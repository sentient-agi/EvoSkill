"""A FAKE `claude` for the tau3 rollout harness's tests (scripts/tau3_cc/tau2_cc_harness.py).

It speaks what Claude Code 2.1.281 speaks on stdin/stdout under `claude -p --input-format stream-json --output-format
stream-json --verbose` with an SDK-hosted MCP server, in the shapes the harness's docstring lists with their sources
(claude_agent_sdk 0.2.91 _internal/query.py, transport/subprocess_cli.py, _internal/client.py; the 2.1.281 binary's
stdin schema and code, read with grep -a): it reads the host's control request `initialize`, holds the MCP handshake
with the host's in-process server "tau2" through `mcp_message` control requests (initialize, notifications/initialized,
tools/list), answers `initialize` with a control_response, and then plays one turn per streamed user message. A turn
emits the system/init event (first turn only), calls the task's expected agent actions as MCP `tools/call` requests
whose params carry _meta {"claudecode/toolUseId": <toolu_ id>} (each preceded by the assistant event holding the
tool_use block, and followed by the user event holding its tool_result), and ends with the assistant's text and a
`result` event. It keeps a session transcript at <HOME>/.claude/projects/<cwd, non-alphanumerics as '-'>/<session>.jsonl,
where Claude Code keeps it. No model, no network: standard library only.

  usage: fake_claude_cc.py SCRIPT.json [the claude argv the harness passes, without argv[0]]
         fake_claude_cc.py --evoctl MAP.json enter --cid C --cwd D [--keep-fd3] -- /usr/bin/env K=V... claude ARGS...
           (stands in for scripts/daytona/evolver_container_ctl.py `enter`: checks the shape, starts from MAP's
           base_env as evoctl's `env -i` does, applies the K=V pairs as /usr/bin/env does, keeps fd 3 only with
           --keep-fd3, enters MAP's dirs[D], records all that in MAP's log and execs itself as claude ARGS)

SCRIPT (JSON):
  log            where the record of everything it saw goes (rewritten after every step, so a crash keeps it)
  session_id     the session id it reports
  actions        the task's expected actions, in order: {"requestor": "assistant"|"user", "name", "arguments", "id"};
                 it plays the assistant's, skipping the customer's (the customer plays those): on every turn but the
                 first it first skips the customer's actions, then calls the agent's until the next is the customer's
                 or none is left, then ends the turn with `to_user` (more to do on the customer's side) or `closing`
  to_user, closing   the two texts it ends turns with
  handshake      "before_init_response" (default: the MCP handshake, then the answer to initialize),
                 "after_init_response" (the answer first; the handshake once the first user message arrived) or
                 "none" (no handshake at all: the server's tools are never listed and it calls none)
  meta_ids       true (default): tools/call carries the toolu_ id in _meta; false: no _meta (the host must match the
                 tool_use block it saw in the stream)
  late_assistant true: the assistant event holding a tool_use block is emitted AFTER its tools/call request
  repeat_first   true: the first agent action is called twice in a row (the second with another toolu_ id)
  extra_calls    [{"name", "arguments", "id"}, ...]: tau2 tools it calls first, at the start of the first turn, before
                 the task's actions (a read-only probe: alltools' shell, KB_search_dense, KB_search_bm25); their
                 answers are recorded in tool_results like the others
  server_status  the tau2 server's status in the init event (default "connected")
  report_cwd     the cwd it reports in the init event and names its transcript by (default: the real cwd)
  turn_sleep_s   seconds it "thinks" at the start of every turn (agent time)
  plant_at_turn  k: at the start of turn k it writes tau2/simulation.json into its cwd (a solver planting a bundle)
  closing_mode   how the closing turn ends: null (the closing text), "empty" (no text), "max_turns" (an
                 error_max_turns result), "end_conversation" (the EndConversation tool, then the session ends),
                 "stop_token" (the closing text + " ###STOP###")
  fail           null or {"kind": K, "after_calls": k, ...}: once k tool calls are done (k=0: at the start of the
                 first turn): crash = exit 1 at once with no result; rate_limit = a 429 error result (subtype
                 success, is_error, api_error_status 429, terminal_reason api_error), then wait for more input as the
                 CLI does; hang = sleep without answering; api_retry_hang = two system/api_retry events (529, then a
                 connection error), then hang; rate_limit_hang = a rate_limit_event with status "rejected", then hang;
                 retry_recovered_hang = an api_retry, then an assistant text event (the API answered), then hang;
                 prompt_too_long / malformed = an is_error result with terminal_reason prompt_too_long /
                 malformed_tool_use_exhausted; signal = send {"signal": "SIGTERM"|"SIGINT"} to its parent (the
                 harness), then hang
The record: argv, argv_problems, env, cwd, pid, the token fd's sha256 (when CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR is
set), every parsed stdin line, the initialize request, the MCP handshake answers and tool list, every tools/call
answer, every user message's text, the transcript path, how it ended.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import sys
import time
import uuid

VALUE_FLAGS = {"--model", "--max-turns", "--output-format", "--permission-mode", "--disallowedTools",
               "--thinking-display", "--input-format", "--mcp-config"}
BARE_FLAGS = {"-p", "--verbose", "--strict-mcp-config"}
BUILTIN_TOOLS = ["Bash", "Edit", "Glob", "Grep", "NotebookEdit", "Read", "Skill", "TodoWrite", "Write"]
CC_VERSION = "2.1.281"
PREFIX = "mcp__tau2__"


def argv_problems(argv: list) -> list:
    P, seen, i = [], {}, 0
    while i < len(argv):
        a = argv[i]
        if a in VALUE_FLAGS:
            if i + 1 >= len(argv):
                P.append(f"{a} has no value")
                break
            seen[a] = argv[i + 1]
            i += 2
        elif a in BARE_FLAGS:
            seen[a] = True
            i += 1
        else:
            P.append(f"unexpected argument {a!r} (a positional prompt or an unknown flag)")
            i += 1
    for flag, want in (("--input-format", "stream-json"), ("--output-format", "stream-json"),
                       ("--permission-mode", "bypassPermissions"), ("--thinking-display", "summarized")):
        if seen.get(flag) != want:
            P.append(f"{flag} is {seen.get(flag)!r}, not {want!r}")
    for flag in ("-p", "--verbose", "--strict-mcp-config", "--model", "--disallowedTools"):
        if flag not in seen:
            P.append(f"{flag} missing")
    if "--max-turns" in seen:
        P.append(f"a turn cap: --max-turns {seen['--max-turns']}")
    try:
        cfg = json.loads(seen.get("--mcp-config") or "null")
        if cfg != {"mcpServers": {"tau2": {"type": "sdk", "name": "tau2"}}}:
            P.append(f"--mcp-config {cfg!r}")
    except ValueError as e:
        P.append(f"--mcp-config is not JSON: {e}")
    return P


def token_sha():
    """sha256 of what the token fd carries (None without the variable; 'unreadable' when the fd is not open)."""
    v = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR")
    if v is None:
        return None
    try:
        chunks = []
        while True:
            b = os.read(int(v), 65536)
            if not b:
                break
            chunks.append(b)
        return hashlib.sha256(b"".join(chunks).strip()).hexdigest()
    except (OSError, ValueError):
        return "unreadable"


class Fake:
    def __init__(self, script: dict, argv: list):
        self.s = script
        self.sid = script.get("session_id") or str(uuid.uuid4())
        self.cwd = script.get("report_cwd") or os.getcwd()
        name = re.sub(r"[^a-zA-Z0-9]", "-", self.cwd)
        self.transcript = os.path.join(os.environ.get("HOME", "/nonexistent"), ".claude", "projects", name,
                                       f"{self.sid}.jsonl")
        self.rec = {"argv": argv, "argv_problems": argv_problems(argv), "env": dict(os.environ), "cwd": self.cwd,
                    "real_cwd": os.getcwd(), "pid": os.getpid(), "token_sha256": token_sha(), "session_id": self.sid,
                    "transcript": self.transcript, "received": [], "initialize": None, "mcp": {}, "tools": None,
                    "tool_results": [], "user_messages": [], "protocol_problems": [], "ended": None, "turns": 0,
                    "turn_started": []}
        self.queue = []                       # user messages that arrived while waiting for a control response
        self.n = 0
        self.calls = 0
        self.cur = 0
        self.turn = 0
        self.mcp_id = 10

    # ── io
    def save(self) -> None:
        tmp = self.s["log"] + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.rec, fh, indent=1)
        os.replace(tmp, self.s["log"])

    def emit(self, ev: dict) -> None:
        sys.stdout.write(json.dumps(ev) + "\n")
        sys.stdout.flush()

    def note(self, entry: dict) -> None:
        os.makedirs(os.path.dirname(self.transcript), exist_ok=True)
        with open(self.transcript, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({**entry, "sessionId": self.sid, "cwd": self.cwd, "uuid": str(uuid.uuid4()),
                                 "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")

    def read(self):
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        try:
            obj = json.loads(line)
        except ValueError:
            self.rec["protocol_problems"].append(f"unparsable stdin line {line[:80]!r}")
            return {}
        self.rec["received"].append(obj)
        return obj

    def request(self, request: dict, after_send=None):
        """Send a control request to the host and return its control_response (None at EOF)."""
        self.n += 1
        rid = f"fake_{self.n}_{uuid.uuid4().hex[:8]}"
        self.emit({"type": "control_request", "request_id": rid, "request": request})
        if after_send is not None:
            after_send()
        while True:
            obj = self.read()
            if obj is None:
                return None
            if obj.get("type") == "control_response" and (obj.get("response") or {}).get("request_id") == rid:
                return obj
            if obj.get("type") == "user":
                self.queue.append(obj)
            else:
                self.rec["protocol_problems"].append(f"while waiting for {rid}: {str(obj)[:200]}")

    def mcp(self, message: dict, after_send=None):
        resp = self.request({"subtype": "mcp_message", "server_name": "tau2", "message": message}, after_send)
        if resp is None:
            return None
        r = resp.get("response") or {}
        if r.get("subtype") != "success" or "mcp_response" not in (r.get("response") or {}):
            self.rec["protocol_problems"].append(f"mcp_message answer {str(resp)[:300]}")
            return None
        return r["response"]["mcp_response"]

    # ── session
    def handshake(self) -> None:
        init = self.mcp({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "claude-code",
                                                                              "version": CC_VERSION}}})
        note = self.mcp({"jsonrpc": "2.0", "method": "notifications/initialized"})
        tools = self.mcp({"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}})
        self.rec["mcp"] = {"initialize": init, "initialized": note, "tools_list": tools}
        self.rec["tools"] = ((tools or {}).get("result") or {}).get("tools")
        self.save()

    def answer_initialize(self, req: dict) -> None:
        self.emit({"type": "control_response", "response": {
            "subtype": "success", "request_id": req.get("request_id"),
            "response": {"commands": [], "agents": [], "output_style": "default",
                         "available_output_styles": ["default"], "models": [], "account": {}, "pid": os.getpid()}}})

    def init_event(self) -> dict:
        tools = BUILTIN_TOOLS + [PREFIX + t["name"] for t in self.rec["tools"] or []]
        return {"type": "system", "subtype": "init", "cwd": self.cwd, "session_id": self.sid, "tools": tools,
                "mcp_servers": [{"name": "tau2", "status": self.s.get("server_status") or "connected",
                                 "source": "sdk"}],
                "model": "claude-sonnet-5-5", "permissionMode": "bypassPermissions", "slash_commands": [],
                "apiKeySource": "none", "claude_code_version": CC_VERSION, "output_style": "default",
                "agents": [], "skills": [], "plugins": [], "uuid": str(uuid.uuid4())}

    def assistant(self, content: list) -> None:
        ev = {"type": "assistant", "message": {"id": f"msg_{uuid.uuid4().hex[:20]}", "type": "message",
                                               "role": "assistant", "model": "claude-sonnet-5-5",
                                               "content": content, "stop_reason": None,
                                               "usage": {"input_tokens": 1, "output_tokens": 1}},
              "parent_tool_use_id": None, "session_id": self.sid, "uuid": str(uuid.uuid4())}
        self.emit(ev)
        self.note({"type": "assistant", "message": ev["message"]})

    def result(self, text, *, subtype="success", is_error=False, **extra) -> None:
        ev = {"type": "result", "subtype": subtype, "is_error": is_error, "duration_ms": 5, "duration_api_ms": 4,
              "num_turns": self.turn, "stop_reason": "end_turn", "session_id": self.sid, "total_cost_usd": 0.0,
              "usage": {"input_tokens": 1, "output_tokens": 1}, "modelUsage": {}, "permission_denials": [],
              "terminal_reason": "completed", "uuid": str(uuid.uuid4())}
        if subtype == "success":
            ev["result"] = text
            ev["api_error_status"] = None
        ev.update(extra)
        self.emit(ev)

    def api_retry(self, attempt: int, status) -> None:
        self.emit({"type": "system", "subtype": "api_retry", "attempt": attempt, "max_retries": 10,
                   "retry_delay_ms": 500 * attempt, "error_status": status,
                   "error": "overloaded_error" if status else "Connection error.", "uuid": str(uuid.uuid4()),
                   "session_id": self.sid})

    def hang(self) -> None:
        time.sleep(3600)
        os._exit(3)

    def maybe_fail(self) -> bool:
        """Apply the scripted failure once `after_calls` tool calls are done; True when the turn is over."""
        fail = self.s.get("fail")
        if not fail or self.calls != int(fail.get("after_calls", 0)) or self.rec.get("failed"):
            return False
        kind = fail["kind"]
        self.rec["failed"] = kind
        self.save()
        if kind == "crash":
            sys.stderr.write("fake claude: crashing as scripted\n")
            sys.stderr.flush()
            os._exit(1)
        if kind == "hang":
            self.hang()
        if kind == "api_retry_hang":
            self.api_retry(1, 529)
            self.api_retry(2, None)
            self.hang()
        if kind == "rate_limit_hang":
            self.emit({"type": "rate_limit_event", "rate_limit_info": {
                "status": "rejected", "resetsAt": int(time.time()) + 3600, "rateLimitType": "five_hour",
                "utilization": 1.0}, "uuid": str(uuid.uuid4()), "session_id": self.sid})
            self.hang()
        if kind == "retry_recovered_hang":
            self.api_retry(1, 529)
            self.assistant([{"type": "text", "text": "Let me look into that."}])
            self.hang()
        if kind == "signal":
            os.kill(os.getppid(), getattr(signal, fail.get("signal") or "SIGTERM"))
            self.hang()
        if kind == "rate_limit":
            self.result("API Error: 429 {\"type\":\"error\",\"error\":{\"type\":\"rate_limit_error\","
                        "\"message\":\"This request would exceed your account's rate limit.\"}}",
                        is_error=True, api_error_status=429, terminal_reason="api_error", stop_reason="stop_sequence")
            return True
        if kind == "prompt_too_long":
            self.result("Prompt is too long", is_error=True, api_error_status=400, terminal_reason="prompt_too_long",
                        stop_reason="stop_sequence")
            return True
        if kind == "malformed":
            self.result("The model's tool call could not be parsed (retry also failed).", is_error=True,
                        terminal_reason="malformed_tool_use_exhausted", stop_reason="stop_sequence")
            return True
        raise SystemExit(f"unknown failure {fail!r}")

    def tool_call(self, a: dict, tid=None) -> None:
        tid = tid or a["id"]
        block = [{"type": "tool_use", "id": tid, "name": PREFIX + a["name"], "input": a["arguments"]}]
        late = bool(self.s.get("late_assistant"))
        if not late:
            self.assistant(block)
        self.mcp_id += 1
        params = {"name": a["name"], "arguments": a["arguments"]}
        if self.s.get("meta_ids", True):
            params["_meta"] = {"claudecode/toolUseId": tid, "progressToken": self.mcp_id}
        resp = self.mcp({"jsonrpc": "2.0", "id": self.mcp_id, "method": "tools/call", "params": params},
                        after_send=(lambda: self.assistant(block)) if late else None)
        res = (resp or {}).get("result") or {}
        content = res.get("content") or []
        text = content[0].get("text") if content and isinstance(content[0], dict) else None
        self.rec["tool_results"].append({"id": tid, "name": a["name"], "jsonrpc_id": (resp or {}).get("id"),
                                         "sent_id": self.mcp_id, "text": text, "isError": res.get("isError"),
                                         "content_types": [c.get("type") for c in content]})
        self.calls += 1
        ev = {"type": "user", "message": {"role": "user", "content": [
            {"tool_use_id": tid, "type": "tool_result", "content": content, "is_error": bool(res.get("isError"))}]},
              "parent_tool_use_id": None, "session_id": self.sid, "uuid": str(uuid.uuid4()),
              "tool_use_result": content}
        self.emit(ev)
        self.note({"type": "user", "message": ev["message"]})
        self.save()

    def play_turn(self, msg: dict) -> bool:
        """One turn for one user message; False when the session is over."""
        self.turn += 1
        self.rec["turns"] = self.turn
        self.rec["turn_started"].append(time.time())
        m = msg.get("message") or {}
        if not (msg.get("parent_tool_use_id", "absent") is None and m.get("role") == "user"
                and isinstance(m.get("content"), str)):
            self.rec["protocol_problems"].append(f"user message shape {str(msg)[:300]}")
        text = m.get("content")
        self.rec["user_messages"].append(text)
        self.note({"type": "user", "message": {"role": "user", "content": text}})
        if self.turn == 1:
            if self.s.get("handshake") == "after_init_response":
                self.handshake()
            self.emit(self.init_event())
        if self.s.get("plant_at_turn") == self.turn:
            os.makedirs("tau2", exist_ok=True)
            with open(os.path.join("tau2", "simulation.json"), "w", encoding="utf-8") as fh:
                json.dump({"planted": True}, fh)
        self.save()
        if self.s.get("turn_sleep_s"):
            time.sleep(float(self.s["turn_sleep_s"]))
        if self.maybe_fail():
            return True
        if self.turn == 1:
            for a in self.s.get("extra_calls") or []:
                self.tool_call(a)
        acts = [] if self.s.get("handshake") == "none" else (self.s.get("actions") or [])
        if self.turn > 1:
            while self.cur < len(acts) and acts[self.cur]["requestor"] == "user":
                self.cur += 1
        while self.cur < len(acts) and acts[self.cur]["requestor"] == "assistant":
            a = acts[self.cur]
            self.tool_call(a)
            if self.s.get("repeat_first") and self.calls == 1:
                self.tool_call(a, tid=a["id"][:-1] + ("A" if a["id"][-1] != "A" else "B"))
            self.cur += 1
            if self.maybe_fail():
                return True
        if self.cur < len(acts):
            final = self.s["to_user"]
        else:
            mode = self.s.get("closing_mode")
            if mode == "empty":
                self.assistant([{"type": "thinking", "thinking": "Nothing more to say.", "signature": "x"}])
                self.result("")
                return True
            if mode == "max_turns":
                self.result(None, subtype="error_max_turns", is_error=True, terminal_reason="max_turns",
                            errors=["Reached maximum number of turns (400)"])
                return True
            if mode == "end_conversation":
                self.assistant([{"type": "tool_use", "id": f"toolu_01end{uuid.uuid4().hex[:16]}",
                                 "name": "EndConversation", "input": {}}])
                self.result("The conversation has ended.")
                self.rec["ended"] = "end_conversation"
                self.save()
                return False
            final = self.s["closing"] + (" ###STOP###" if mode == "stop_token" else "")
        self.assistant([{"type": "text", "text": final}])
        self.result(final)
        return True

    def run(self) -> int:
        first = self.read()
        self.rec["initialize"] = first
        req = (first or {}).get("request") or {}
        if not ((first or {}).get("type") == "control_request" and req.get("subtype") == "initialize"
                and req.get("sdkMcpServers") == ["tau2"] and isinstance(first.get("request_id"), str)):
            self.rec["protocol_problems"].append(f"first line is not the initialize request: {str(first)[:300]}")
        if self.s.get("handshake") in ("after_init_response", "none"):
            self.answer_initialize(first or {})
        else:
            self.handshake()
            self.answer_initialize(first or {})
        self.save()
        while True:
            msg = self.queue.pop(0) if self.queue else self.read()
            if msg is None:
                self.rec["ended"] = self.rec["ended"] or "eof"
                self.save()
                return 0
            if msg.get("type") == "user":
                if not self.play_turn(msg):
                    return 0
            elif msg.get("type") in ("control_request", "control_response"):
                self.rec["protocol_problems"].append(f"unexpected {str(msg)[:300]}")
            self.save()


def evoctl(argv: list) -> int:
    """`evoctl enter` stood in for (see the usage above); execs the fake claude, or exits 97 on a shape problem."""
    with open(argv[0], encoding="utf-8") as fh:
        m = json.load(fh)
    a, rec = argv[1:], {"argv": argv[1:], "problems": [], "fds": None}
    keep3 = "--keep-fd3" in a
    try:
        dd = a.index("--")
        head, cmd = a[:dd], a[dd + 1:]
        if head[:1] != ["enter"] or "--cid" not in head or "--cwd" not in head:
            rec["problems"].append(f"not `enter --cid C --cwd D`: {head}")
        cid, cwd = head[head.index("--cid") + 1], head[head.index("--cwd") + 1]
    except (ValueError, IndexError):
        rec["problems"].append(f"bad evoctl argv {a}")
        cid, cwd, cmd = None, None, []
    env = dict(m["base_env"])                 # evoctl: `env -i` + its own base environment
    if cmd[:1] != ["/usr/bin/env"]:
        rec["problems"].append(f"ARGV does not start with /usr/bin/env: {cmd[:2]}")
    i = 1
    while i < len(cmd) and "=" in cmd[i] and not cmd[i].startswith("/"):
        k, v = cmd[i].split("=", 1)
        env[k] = v
        i += 1
    claude = cmd[i:]
    if claude[:1] != ["claude"]:
        rec["problems"].append(f"the command is not claude: {claude[:1]}")
    rec.update({"cid": cid, "cwd": cwd, "keep_fd3": keep3, "pairs": cmd[1:i], "claude_argv": claude})
    if not keep3:
        try:
            os.close(3)
        except OSError:
            pass
    rec["fds"] = sorted(int(x) for x in os.listdir("/dev/fd") if int(x) > 2)
    with open(m["log"], "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    if rec["problems"] or cwd not in m["dirs"]:
        return 97
    os.chdir(m["dirs"][cwd])
    os.execve(sys.executable, [sys.executable, os.path.abspath(__file__), m["script"]] + claude[1:], env)
    return 127


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "--evoctl":
        return evoctl(sys.argv[2:])
    if len(sys.argv) < 2:
        sys.stderr.write(__doc__)
        return 2
    with open(sys.argv[1], encoding="utf-8") as fh:
        script = json.load(fh)
    return Fake(script, sys.argv[2:]).run()


if __name__ == "__main__":
    sys.exit(main())
