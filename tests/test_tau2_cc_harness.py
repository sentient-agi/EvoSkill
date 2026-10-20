"""The tau3 rollout harness, scripts/tau3_cc/tau2_cc_harness.py: Claude Code as the tau2 agent through the CLI's
stream-json control protocol, with tau2's alltools retrieval through scripts/tau3_cc/tau3_retrieval.py.

OFFLINE: no model, no network, no real `claude`, no Daytona. Claude Code is tests/fake_claude_cc.py, which speaks the
protocol's shapes (initialize, mcp_message with initialize / notifications/initialized / tools/list / tools/call, the
stream-json events and results, api_retry and rate_limit events) and plays a task's expected agent actions as MCP tool
calls; it also stands in for `evoctl enter` (the SSB solver's container). The customer is a scripted golden customer
registered in tau2's registry (the user-side counterpart of tests/fixtures_tau2_scripted.py) that can sleep, raise or
hang on given invocations. Every tau2 step runs in a CHILD process under the tau3-bench venv (tests/tau3_golden.TAU3_PY)
with tau3_golden.child_env() (PYTHON_DOTENV_DISABLED=1, LITELLM_LOCAL_MODEL_COST_MAP=True, TAU2_DATA_DIR and nothing of
this process's environment) plus PATH = TOOL_PATH (tau2's shell sandbox needs srt and rg, found on PATH)
and dead proxies (HTTPS_PROXY / HTTP_PROXY / ALL_PROXY = 127.0.0.1:9: nothing reaches a network); the child calls the
harness's own main() in-process, so the CLI's code path runs (signal handlers included), with the scripted customer
registered first. The data root is the run's curated root third_party/tau3_data_363133a (or
$EVOSKILL_TAU3_DATA_ROOT), the one that holds alltools' precomputed .embeddings_cache. Skipped when the venv, the data
root, its embeddings cache, srt or rg is absent. No OpenRouter key exists here: KB_search_dense is offline (its call
returns tau2's tool error) and every other tool works.

What must hold:
  - the files: the harness imports only the standard library and tau2_bundle at import time; its argv is the SSB
    solver's (cc_solver_lib.claude_argv) without the prompt and WITHOUT --max-turns, plus --input-format stream-json and
    --mcp-config for the sdk server "tau2"; its CC_ENV and disallowed tools are cc_solver_lib's; the first message is
    exactly FIRST_MESSAGE_TEXT (pinned below) filled with the policy and the customer's opening; claude's
    environment is an allow-list; tau2 runs with its own max_steps 200, no tau2 wall, max_errors 10 and EMPTY
    customer llm args; every job field is checked (bad input: exit 2), home is required, key_fd == token_fd is refused,
    read_key is bounded; a job for any retrieval variant but alltools is refused (exit 2); the in-container launch is
    `evoctl enter --keep-fd3 -- /usr/bin/env <claude's vars> claude ...`;
  - alltools: the first message embeds tau2's alltools policy (KB_search_bm25, KB_search_dense with
    "uses **OpenAI API** with embedding model `text-embedding-3-large`", the shell instructions); Claude Code is listed
    17 tools (KB_search_bm25, KB_search_dense, shell and the 14 domain tools; no KB_search, no grep); the shell
    reads the knowledge base (`ls`) but not the run's files (`cd / && cat <data root>/.../task_004.json` gets nothing
    back), while the same probe WITHOUT the hardening reads the task (the control); KB_search_dense with no key is a
    tool error the conversation survives, and with a key whose endpoint is unreachable an error that names no key;
    harness.json and run.json record install()'s record (cache file sha256s, the deny list, key true/false), and
    harness.json the PATH and the srt / rg / bwrap / socat it found; the shell sandboxes' temp dirs are gone when the
    process exits;
  - golden conversations (several non-NL banking tasks; all 96 with EVOSKILL_TAU3_HARNESS_ALL=1): tau2's Orchestrator
    records exactly tests/tau3_golden.py's golden conversation, ended by the customer (user_stop); the bundle grades 1.0
    with the user's grader (strict replay); exit 0; the protocol carried the exact shapes both ways; every MCP tool
    result is tau2's ToolMessage content for that id; run.json and harness.json record the session, the transcript, the
    agent clock and the customer's calls; the same holds launched through the fake evoctl with a token on fd 3;
  - the agent clock: the customer's time never counts (a slow customer, a conversation longer than the limit, still
    graded 1.0); Claude's own time does (timeout, graded 0.0, at the limit);
  - the ends: an empty final text, an error_max_turns result, EndConversation, prompt_too_long or
    malformed_tool_use_exhausted -> agent_error, graded 0.0, exit 0; ###STOP### in Claude's reply -> agent_stop,
    graded 1.0; a crash, a 429 error result, the tau2 server not connected or its tools never listed, a claude that
    does not start, the agent clock running out while Claude Code retries the API or is rate limited, a customer call
    whose every attempt fails, SIGTERM / SIGINT -> infrastructure_error, exit 75, NO tau2/simulation.json (the grader
    raises rather than scoring 0); a hanging claude, or one that hangs after its API retry recovered -> timeout,
    graded 0.0; a customer call that raises and then hangs is retried and the conversation completes;
  - ids: a tool_use block that arrives after its call (no _meta) never hands its id to a later identical call;
  - secrets: the user simulator's key, given on an fd, is loaded into litellm's memory, its fd is closed, and it is in
    no argv, no environment and no file; an OPENROUTER_API_KEY in the harness's environment is dropped (by name);
    the token on fd 3 reaches claude through evoctl and is in no file;
  - a tau2/ planted in the working directory (before, or during the conversation) is replaced, its target untouched,
    and an old bundle never survives a bad job or an infrastructure end.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
HARNESS = ROOT / "scripts" / "tau3_cc" / "tau2_cc_harness.py"
FAKE = TESTS / "fake_claude_cc.py"
GRADER = ROOT / "examples" / "tau3_banking" / "grader.py"
# The solver's first user message, stated here independently of the harness (its bytes and sha256 are the contract).
FIRST_MESSAGE_TEXT = (
    "You are a customer service agent that helps the user according to the <policy> provided below.\n"
    "\n"
    "<policy>\n"
    "{policy}\n"
    "</policy>\n"
    "\n"
    "You started with \"Hi! How can I help you today?\", and the user replied:\n"
    "\n"
    "{customer}\n"
    "\n"
    "Please continue the conversation. Your output is sent to the user, in full, verbatim; any tool calling turns or "
    "reasoning tokens remain invisible to the user.\n")
FIRST_MESSAGE_SHA256 = "a3215d76cc2df8e05a7fd9d4abca283b3c1d2432deffd2f3f0bf224298494f7d"
PLUGIN_API = ROOT / "src" / "orchestrator" / "plugin_api.py"
CC_SOLVER_LIB = ROOT / "scripts" / "cc_solver" / "cc_solver_lib.py"
N_CHILDREN = 4
CUSTOMER = "cc_test_golden_customer"
GOLDEN = ["task_001", "task_017", "task_026", "task_034", "task_043", "task_058", "task_081", "task_083", "task_098",
          "task_101"]
KEY_FD_TEXT = "sk-or-v1-TESTONLY-" + hashlib.sha256(b"fd key").hexdigest()[:40]       # never a real key
KEY_ENV_TEXT = "sk-or-v1-TESTONLY-" + hashlib.sha256(b"env key").hexdigest()[:40]
TOKEN_TEXT = "TESTONLY-cc-token-" + hashlib.sha256(b"fd3 token").hexdigest()[:40]     # never a real token
STDLIB = {"__future__", "argparse", "copy", "hashlib", "json", "math", "os", "re", "select", "shutil", "signal",
          "subprocess", "sys", "threading", "time", "pathlib", "uuid"}
INFRA_FILES = ["final_db.json", "run.json", "simulation.infrastructure_error.json"]
GRADED_FILES = ["final_db.json", "run.json", "simulation.json"]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


G = _load(TESTS / "tau3_golden.py", "tau3_golden")
H = _load(HARNESS, "tau2_cc_harness")
TR = H.TR                                                   # scripts/tau3_cc/tau3_retrieval.py, as the harness has it
CURATED_ROOT = ROOT / "third_party" / "tau3_data_363133a"   # the run's data root: it holds alltools' embeddings cache
DATA_ROOT = Path(os.environ.get(G.DATA_ROOT_ENV) or CURATED_ROOT)
TOOL_PATH = G.tool_path()                                   # srt, rg for tau2's shell (Linux: + bwrap, socat)
DEAD_PROXY = "http://127.0.0.1:9"
ALLTOOLS_KB = {"KB_search_bm25", "KB_search_dense", "shell"}
PROBE_TASK = "task_004"
PROBE_IDS = [f"toolu_01probe{i:017d}" for i in range(4)]
DENSE_ID = "toolu_01dense000000000000000"


def _rel_task_path(task: str) -> str:
    """<data root>/tau2/domains/banking_knowledge/tasks/<task>.json relative to / (tau2's guard blocks absolute paths;
    `cd / && cat <this>` passes it)."""
    return (DATA_ROOT.resolve() / "tau2" / "domains" / G.DOMAIN / "tasks" / f"{task}.json").relative_to("/").as_posix()


SHELL_PROBE = f"cd / && cat {_rel_task_path(PROBE_TASK)} | head -c 200"


def _child_env(**extra) -> dict:
    """tau3_golden.child_env on the alltools data root, with srt / rg on PATH and dead proxies."""
    return G.child_env(DATA_ROOT, **{G.DATA_ROOT_ENV: DATA_ROOT, "PATH": TOOL_PATH, "HTTPS_PROXY": DEAD_PROXY,
                                     "HTTP_PROXY": DEAD_PROXY, "ALL_PROXY": DEAD_PROXY, **extra})


# The child: runs a list of scenarios through the harness's main(), grades, and reports everything as JSON.
CHILD = r'''
import hashlib, importlib.util, json, os, sys, time, uuid
from pathlib import Path

(tests_dir, harness_dir, fake_path, grader_path, api_path, scen_path, out_path, work_base, key_fd, key_sha,
 first_md, token_text) = sys.argv[1:13]
sys.path[:0] = [tests_dir, harness_dir]
import tau3_golden as G
import tau2_cc_harness as H
TR = H.TR
VAR = TR.VARIANT
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.data_model.simulation import SimulationRun
from tau2.registry import registry
from tau2.user.user_simulator_base import HalfDuplexUser, UserState

CUSTOMER = "cc_test_golden_customer"
CURRENT = []
BUILT = []
OPTS = {}
INVOKED = []


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A = load(api_path, "plugin_api")
GR = load(grader_path, "benchmark_grader")


class GoldenCustomer(HalfDuplexUser):
    """Plays the task's customer actions (one ToolCall per message, call_ ids) and lets the agent play its own: on the
    agent's text it skips the agent's actions, then plays its next one, says ###STOP### when none is left, or hands
    the turn back (tau3_golden's lines). OPTS: sleep_s (every invocation), faults (per invocation: "ok" | "raise" |
    "hang"; a hang sleeps hang_s and then raises, so an abandoned attempt never touches the customer)."""

    def __init__(self, tools=None, instructions=None, llm=None, llm_args=None):
        super().__init__(instructions=instructions, tools=tools)
        t = CURRENT[0]
        if instructions != str(t.user_scenario):
            raise ValueError("the golden customer is not built for this task")
        self.task_id = t.id
        self.acts = list(t.evaluation_criteria.actions or [])
        self.cur = 0
        self.first = True
        self.llm, self.llm_args = llm, llm_args
        BUILT.append(self)

    def get_init_state(self, message_history=None):
        return UserState(system_messages=[], messages=list(message_history or []))

    def generate_next_message(self, message, state):
        i = len(INVOKED)
        faults = OPTS.get("faults") or []
        what = faults[i] if i < len(faults) else "ok"
        INVOKED.append(what)
        if OPTS.get("sleep_s"):
            time.sleep(OPTS["sleep_s"])
        if what == "raise":
            raise RuntimeError("scripted customer failure")
        if what == "hang":
            time.sleep(OPTS.get("hang_s", 3.0))
            raise RuntimeError("an abandoned attempt woke up")
        state.messages.append(message)
        if self.first:
            self.first = False
            out = UserMessage(role="user", content=G.OPENING, cost=0.0)
        else:
            if isinstance(message, AssistantMessage):
                while self.cur < len(self.acts) and self.acts[self.cur].requestor == "assistant":
                    self.cur += 1
            if self.cur < len(self.acts) and self.acts[self.cur].requestor == "user":
                a = self.acts[self.cur]
                tc = ToolCall(id=G.call_id(self.task_id, self.cur, "user"), name=a.name,
                              arguments=dict(a.arguments or {}), requestor="user")
                self.cur += 1
                out = UserMessage(role="user", content=None, tool_calls=[tc], cost=0.0)
            elif self.cur >= len(self.acts) and isinstance(message, AssistantMessage):
                out = UserMessage(role="user", content=G.STOP, cost=0.0)
            else:
                out = UserMessage(role="user", content=G.TO_AGENT, cost=0.0)
        state.messages.append(out)
        return out, state


registry.register_user(GoldenCustomer, CUSTOMER)


def ensure_retrieval():
    """alltools environments built here (the policy, the gold conversation, the probe) need install(); the harness's
    main() did it already unless it stopped before (a bad job): then offline, as the grader."""
    if TR._STATE["installed"] is None:
        TR.install(G.data_root(), key=None)


def task_of(f):
    r = G.plugin_row(f, variant=VAR)
    return A.Task(id=r["id"], category=r["category"], instruction=r["instruction"],
                  inputs=tuple(Path(p) for p in r["inputs"]), answer_files=tuple(Path(p) for p in r["answer_files"]),
                  grading=dict(r["grading"]), split=r["split"])


def grade(task, snap):
    try:
        v = GR.grade(task, snap)
    except Exception as e:
        return {"raised": type(e).__name__, "message": str(e)[:300]}
    return {"score": v, "type": type(v).__name__}


def shape(msgs):
    out = []
    for m in msgs:
        if isinstance(m, ToolMessage):
            out.append(["tool", m.id, m.requestor, m.content, m.error])
        else:
            out.append([m.role, m.content, [tc.model_dump(mode="json") for tc in (m.tool_calls or [])]])
    return out


def jload(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def probe_config(sc):
    """What the harness hands tau2: the run config, and the real user simulator built from it (no model call)."""
    from tau2.config import DEFAULT_LLM_ARGS_USER
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner.build import build_user
    ensure_retrieval()
    f = Path(G.data_root()) / "tau2" / "domains" / G.DOMAIN / "tasks" / f"{sc['task']}.json"
    t = G.load_task(f)
    job = dict(H.JOB_DEFAULTS, task_file=str(f), workdir="/w", out_dir="/o", home="/h")
    cfg = TextRunConfig(**H.run_config(job, 12345))
    u = build_user(cfg.user, G.fresh_env(t, variant=VAR), t, llm=cfg.llm_user, llm_args=cfg.llm_args_user)
    rec = {"tau2_default_user_args": DEFAULT_LLM_ARGS_USER, "llm_args_user": cfg.llm_args_user,
           "llm_user": cfg.llm_user, "max_steps": cfg.max_steps, "timeout": cfg.timeout, "max_errors": cfg.max_errors,
           "retrieval_config": cfg.retrieval_config, "user_class": type(u).__name__, "user_llm": u.llm,
           "user_llm_args": dict(u.llm_args)}
    u.set_seed(cfg.seed)
    rec["user_llm_args_seeded"] = dict(u.llm_args)
    calls = H.CustomerCalls(u, timeout_s=1.0, attempts=1, backoff_s=0.0)
    rec["wrapped"] = u.generate_next_message is calls
    return rec


def run_one(sc):
    if sc.get("probe") == "config":
        return {"probe": probe_config(sc)}
    f = Path(G.data_root()) / "tau2" / "domains" / G.DOMAIN / "tasks" / f"{sc['task']}.json"
    t = G.load_task(f)
    d = Path(work_base) / sc["name"].replace(":", "_")
    work, out, home = d / "work", d / "out", d / "home"
    for p in (work, out, home):
        p.mkdir(parents=True)
    actions = [{"requestor": a.requestor, "name": a.name, "arguments": dict(a.arguments or {}),
                "id": G.call_id(t.id, i, a.requestor)} for i, a in enumerate(t.evaluation_criteria.actions or [])]
    sid = f"sess-{uuid.uuid4()}"
    script = {"log": str(d / "fake_log.json"), "session_id": sid, "actions": actions, "to_user": G.TO_USER,
              "closing": G.CLOSING, **sc.get("script", {})}
    jb = dict(sc.get("job", {}))
    missing = jb.pop("claude_cmd", None) == "missing"
    job = {"task_file": str(f), "workdir": str(work), "out_dir": str(out), "home": str(home), "user": CUSTOMER,
           "claude_cmd": [str(d / "no_such_claude")] if missing else [sys.executable, fake_path, str(d / "script.json")],
           "close_grace_s": 5.0, "mid_turn_grace_s": 0.5, **jb}
    saved3 = None
    if sc.get("container"):
        script["report_cwd"] = "/evo/work"
        emap = {"log": str(d / "evoctl_log.json"), "script": str(d / "script.json"),
                "base_env": {"PATH": "/usr/bin:/bin", "LC_ALL": "C.UTF-8", "HOME": "/home/evo"},
                "dirs": {"/evo/work": str(work)}}
        (d / "evoctl_map.json").write_text(json.dumps(emap))
        job["claude_cmd"] = ["claude"]
        job["container"] = {"evoctl": [sys.executable, fake_path, "--evoctl", str(d / "evoctl_map.json")],
                            "cid": "cc-test-" + sc["task"].replace("_", "-"), "cwd": "/evo/work"}
    if sc.get("token"):
        try:
            os.fstat(3)
            saved3 = os.dup(3)
        except OSError:
            saved3 = None
        r, w = os.pipe()
        os.write(w, token_text.encode() + b"\n")
        os.close(w)
        if r != 3:
            os.dup2(r, 3)
            os.close(r)
        job["token_fd"] = 3
    (d / "script.json").write_text(json.dumps(script))
    if sc.get("key"):
        job["key_fd"] = int(key_fd)
    if sc.get("plant"):
        target = d / "planted_target"
        target.mkdir()
        (target / "x").write_text("the solver's")
        (work / "tau2").symlink_to(target, target_is_directory=True)
    if sc.get("plant_old"):
        (work / "tau2").mkdir()
        (work / "tau2" / "simulation.json").write_text(json.dumps({"old": True}))
    (d / "job.json").write_text(json.dumps(job))
    CURRENT[:] = [t]
    BUILT.clear()
    OPTS.clear()
    OPTS.update(sc.get("customer", {}))
    INVOKED.clear()
    saved_or = (TR.OPENROUTER_BASE, TR.ATTEMPTS)
    if sc.get("dense_unreachable"):            # a key, but OpenRouter at a dead local port: no request leaves this host
        TR.OPENROUTER_BASE, TR.ATTEMPTS = "http://127.0.0.1:9/api/v1", 2
    t0 = time.time()
    try:
        rc = H.main(["--job", str(d / "job.json")])
    finally:
        TR.OPENROUTER_BASE, TR.ATTEMPTS = saved_or
    secs = time.time() - t0
    if sc.get("token"):
        try:
            os.fstat(3)
            fd3_open_after = True
        except OSError:
            fd3_open_after = False
        if saved3 is not None:
            os.dup2(saved3, 3)
            os.close(saved3)
    rec = {"rc": rc, "seconds": round(secs, 3), "tau2_id": t.id, "session_id": sid,
           "harness": jload(out / "harness.json"), "fake": jload(d / "fake_log.json"),
           "evoctl": jload(d / "evoctl_log.json"), "customer_invocations": list(INVOKED),
           "token": bool(sc.get("token"))}
    if sc.get("token"):
        rec["token_fd_closed_by_harness"] = not fd3_open_after
    bdir = work / "tau2"
    rec["bundle_is_link"] = bdir.is_symlink()
    rec["bundle_exists"] = bdir.exists() or bdir.is_symlink()
    rec["bundle_files"] = sorted(x.name for x in bdir.iterdir()) if bdir.is_dir() and not bdir.is_symlink() else None
    rec["run"] = jload(bdir / "run.json")
    simf = bdir / "simulation.json"
    rec["simulation_json"] = jload(simf)
    if not simf.is_file():
        simf = bdir / H.INFRA_SIMULATION
    sim = None
    if simf.is_file():
        try:
            sim = SimulationRun.model_validate_json(simf.read_text(encoding="utf-8"))
        except ValueError:
            sim = None
    rec["termination"] = sim.termination_reason.value if sim is not None else None
    rec["n_messages"] = len(sim.messages) if sim is not None else None
    rec["grade"] = grade(task_of(f), work)
    ensure_retrieval()
    env = G.fresh_env(t, variant=VAR)
    policy = env.get_policy()
    rec["expected_first_message"] = H.first_message(policy, G.OPENING)
    rec["first_message_from_file"] = (Path(first_md).read_text(encoding="utf-8").replace("{policy}", policy)
                                      .replace("{customer}", G.OPENING))
    rec["env_tools"] = sorted(x.name for x in env.get_tools())
    gold = G.golden_messages(t, G.fresh_env(t, variant=VAR))
    if sim is not None:
        rec["same_as_golden"] = shape(sim.messages) == shape(gold)
        rec["prefix_of_golden"] = shape(sim.messages)[:-1] == shape(gold)[:len(sim.messages) - 1]
        rec["sim_shape_tail"] = shape(sim.messages)[-2:]
        rec["tool_messages"] = {m.id: [m.content, m.error] for m in sim.messages if isinstance(m, ToolMessage)}
        rec["agent_call_ids"] = [tc.id for m in sim.messages if isinstance(m, AssistantMessage)
                                 for tc in (m.tool_calls or [])]
        rec["duration"] = sim.duration
    rec["golden_agent_ids"] = [a["id"] for a in actions if a["requestor"] == "assistant"]
    stdin_lines = []
    try:
        for line in (out / "claude.stdin.jsonl").read_text(encoding="utf-8").splitlines():
            stdin_lines.append(json.loads(line))
    except OSError:
        pass
    rec["stdin"] = stdin_lines
    rec["stream_types"] = []
    try:
        for line in (out / "claude.stream.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("{"):
                rec["stream_types"].append(json.loads(line).get("type"))
    except OSError:
        pass
    if rec["fake"]:
        rec["fake_alive"] = alive(rec["fake"]["pid"])
        rec["transcript_exists"] = Path(rec["fake"]["transcript"]).is_file()
    if sc.get("plant"):
        rec["planted_target_files"] = sorted(x.name for x in (d / "planted_target").iterdir())
    if sc.get("key"):
        import litellm
        rec["key_loaded"] = hashlib.sha256((litellm.openrouter_key or "").encode()).hexdigest() == key_sha
        try:
            os.fstat(int(key_fd))
            rec["key_fd_closed"] = False
        except OSError:
            rec["key_fd_closed"] = True
        (d / "driver_env_after.json").write_text(json.dumps(dict(os.environ)))
        rec["driver_env_has_openrouter"] = "OPENROUTER_API_KEY" in os.environ
    rec["customer_llm_args"] = [c.llm_args for c in BUILT]
    return rec


out = {}
for sc in json.loads(Path(scen_path).read_text()):
    try:
        out[sc["name"]] = run_one(sc)
    except BaseException as e:
        import traceback
        out[sc["name"]] = {"child_error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]}
# the shell sandboxes this process made (tau3_retrieval removes their temp dirs at exit; the parent checks)
out["__sandboxes__"] = {"dirs": [str(sb.sandbox_dir) for sb in TR._STATE["sandboxes"]]}
Path(out_path).write_text(json.dumps(out, indent=1, default=str))
'''


def _need_tau3():
    if not G.TAU3_PY.is_file():
        pytest.skip(f"the tau3-bench venv interpreter {G.TAU3_PY} is absent")
    if not G.task_files(DATA_ROOT):
        pytest.skip(f"no banking task files under {DATA_ROOT}")
    if not TR.cache_dir(DATA_ROOT).is_dir():
        pytest.skip(f"{TR.cache_dir(DATA_ROOT)} is absent (scripts/tau3_build_embeddings_cache.py)")
    missing = [b for b in ("srt", "rg") if shutil.which(b, path=TOOL_PATH) is None]
    if missing:
        pytest.skip(f"tau2's shell sandbox needs {missing} on {TOOL_PATH}")


def _scenarios() -> tuple:
    names = [p.stem for p in G.graded_task_files(DATA_ROOT)] if os.environ.get("EVOSKILL_TAU3_HARNESS_ALL") else GOLDEN
    golden = []
    for tid in names:
        sc = {"name": f"golden:{tid}", "task": tid}
        if tid == "task_026":
            sc["script"] = {"handshake": "after_init_response"}
        if tid == "task_017":
            sc["script"] = {"meta_ids": False}
        golden.append(sc)
    retry = {"customer_timeout_s": 1.0, "customer_backoff_s": 0.1, "customer_attempts": 3}
    ends = [
        {"name": "empty:task_043", "task": "task_043", "script": {"closing_mode": "empty"}},
        {"name": "max_turns:task_058", "task": "task_058", "script": {"closing_mode": "max_turns"}},
        {"name": "end_conversation:task_098", "task": "task_098", "script": {"closing_mode": "end_conversation"}},
        {"name": "prompt_too_long:task_043", "task": "task_043",
         "script": {"fail": {"kind": "prompt_too_long", "after_calls": 1}}},
        {"name": "malformed:task_026", "task": "task_026", "script": {"fail": {"kind": "malformed", "after_calls": 1}}},
        {"name": "stop_token:task_043", "task": "task_043", "script": {"closing_mode": "stop_token"}},
        {"name": "crash:task_043", "task": "task_043", "script": {"fail": {"kind": "crash", "after_calls": 2}}},
        {"name": "crash0:task_001", "task": "task_001", "script": {"fail": {"kind": "crash", "after_calls": 0}}},
        {"name": "rate_limit:task_026", "task": "task_026",
         "script": {"fail": {"kind": "rate_limit", "after_calls": 1}}},
        {"name": "server_failed:task_001", "task": "task_001", "script": {"server_status": "failed"}},
        {"name": "no_tools:task_001", "task": "task_001", "script": {"handshake": "none", "server_status": "pending"}},
        {"name": "no_claude:task_001", "task": "task_001", "job": {"claude_cmd": "missing"}},
        {"name": "api_retry_hang:task_043", "task": "task_043", "job": {"agent_time_s": 2.0},
         "script": {"fail": {"kind": "api_retry_hang", "after_calls": 1}}},
        {"name": "rate_limit_hang:task_001", "task": "task_001", "job": {"agent_time_s": 2.0},
         "script": {"fail": {"kind": "rate_limit_hang", "after_calls": 0}}},
        {"name": "customer_fails:task_026", "task": "task_026", "job": dict(retry),
         "customer": {"faults": ["ok", "raise", "raise", "raise"]}},
        {"name": "sigterm:task_026", "task": "task_026",
         "script": {"fail": {"kind": "signal", "signal": "SIGTERM", "after_calls": 1}}},
        {"name": "sigint:task_043", "task": "task_043",
         "script": {"fail": {"kind": "signal", "signal": "SIGINT", "after_calls": 0}}},
        {"name": "hang:task_043", "task": "task_043", "script": {"fail": {"kind": "hang", "after_calls": 0}},
         "job": {"agent_time_s": 3.0}},
        {"name": "retry_recovered:task_043", "task": "task_043", "job": {"agent_time_s": 2.0},
         "script": {"fail": {"kind": "retry_recovered_hang", "after_calls": 1}}},
        {"name": "slow_customer:task_026", "task": "task_026", "job": {"agent_time_s": 2.0},
         "customer": {"sleep_s": 0.4}},
        {"name": "agent_clock:task_026", "task": "task_026", "job": {"agent_time_s": 1.2},
         "script": {"turn_sleep_s": 0.8}, "customer": {"sleep_s": 0.3}},
        {"name": "customer_retry:task_026", "task": "task_026", "job": dict(retry),
         "customer": {"faults": ["ok", "raise", "hang", "ok"], "hang_s": 3.0}},
        {"name": "late_ids:task_026", "task": "task_026",
         "script": {"late_assistant": True, "meta_ids": False, "repeat_first": True}},
        {"name": "planted:task_001", "task": "task_001", "plant": True},
        {"name": "planted_during:task_031", "task": "task_031", "script": {"plant_at_turn": 2}},
        {"name": "old_bundle_infra:task_001", "task": "task_001", "plant_old": True,
         "script": {"fail": {"kind": "crash", "after_calls": 0}}},
        {"name": "old_bundle_bad_job:task_001", "task": "task_001", "plant_old": True,
         "job": {"task_id": "banking_knowledge/task_999"}},
        {"name": "container:task_017", "task": "task_017", "container": True, "token": True},
        {"name": "probe:config", "task": "task_026", "probe": "config"},
        {"name": f"probe_tools:{PROBE_TASK}", "task": PROBE_TASK, "script": {"extra_calls": [
            {"name": "shell", "arguments": {"command": SHELL_PROBE}, "id": PROBE_IDS[0]},
            {"name": "shell", "arguments": {"command": "ls | head -3"}, "id": PROBE_IDS[1]},
            {"name": "KB_search_dense", "arguments": {"query": "credit card annual fee"}, "id": PROBE_IDS[2]},
            {"name": "KB_search_bm25", "arguments": {"query": "credit card annual fee"}, "id": PROBE_IDS[3]}]}},
    ]
    secret = [{"name": "secrets:task_026", "task": "task_026", "key": True},
              # a dense search whose embedding request fails every try ends the rollout as an
              # infrastructure end (re-run), never as the solver's 0; here OpenRouter sits at a dead local port
              {"name": "dense_outage:task_026", "task": "task_026", "key": True, "dense_unreachable": True,
               "script": {"extra_calls": [{"name": "KB_search_dense", "arguments": {"query": "card fee"},
                                           "id": DENSE_ID}]}}]
    return golden, ends, secret


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    _need_tau3()
    base = tmp_path_factory.mktemp("tau2_cc_harness")
    golden, ends, secret = _scenarios()
    regular = golden + ends
    child_py = base / "child.py"
    child_py.write_text(CHILD)
    first_md = base / "first_message.md"
    first_md.write_text(FIRST_MESSAGE_TEXT, encoding="utf-8")
    key_sha = hashlib.sha256(KEY_FD_TEXT.encode()).hexdigest()
    rfds = []                                   # one key pipe per secret child (a harness reads and closes its fd)
    for _ in secret:
        rfd, wfd = os.pipe()
        os.write(wfd, KEY_FD_TEXT.encode() + b"\n")
        os.close(wfd)
        rfds.append(rfd)
    procs = []
    try:
        chunks = [regular[k::N_CHILDREN] for k in range(N_CHILDREN)] + [[sc] for sc in secret]
        for k, chunk in enumerate(chunks):
            is_secret = k >= N_CHILDREN
            rfd = rfds[k - N_CHILDREN] if is_secret else None
            (base / f"scen_{k}.json").write_text(json.dumps(chunk))
            env = _child_env(HOME=base / "home")
            if is_secret:
                env["OPENROUTER_API_KEY"] = KEY_ENV_TEXT
            log = base / f"child_{k}.log"
            fh = open(log, "wb")
            procs.append((subprocess.Popen(
                [str(G.TAU3_PY), "-B", str(child_py), str(TESTS), str(HARNESS.parent), str(FAKE), str(GRADER),
                 str(PLUGIN_API), str(base / f"scen_{k}.json"), str(base / f"out_{k}.json"), str(base / "work"),
                 str(rfd if is_secret else -1), key_sha, str(first_md), TOKEN_TEXT],
                cwd=str(base), stdout=fh, stderr=subprocess.STDOUT, env=env,
                pass_fds=(rfd,) if is_secret else ()), fh, log))
        for rfd in rfds:
            os.close(rfd)
        rfds = []
        for p, fh, log in procs:
            rc = p.wait(timeout=1800)
            fh.close()
            if rc != 0:
                raise AssertionError(f"tau3 child failed (rc {rc}):\n{log.read_bytes()[-4000:].decode('utf-8', 'replace')}")
    finally:
        for rfd in rfds:
            os.close(rfd)
        for p, fh, _log in procs:
            if p.poll() is None:
                p.kill()
            fh.close()
    res, sandboxes = {}, []
    for k in range(N_CHILDREN + len(secret)):
        o = json.loads((base / f"out_{k}.json").read_text())
        sandboxes += o.pop("__sandboxes__")["dirs"]
        res.update(o)
    bad = {n: r for n, r in res.items() if "child_error" in r}
    assert not bad, {n: r["trace"] for n, r in bad.items()}
    return {"base": base, "res": res, "golden": [s["name"] for s in golden], "sandboxes": sandboxes}


def _top_imports(path: Path) -> set:
    out = set()
    for n in ast.parse(path.read_text()).body:
        if isinstance(n, ast.Import):
            out |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            out.add(n.module.split(".")[0])
    return out


def _job(**kw) -> dict:
    return {"task_file": "/t", "workdir": "/w", "out_dir": "/o", "home": "/h", **kw}


# ─────────────────────────────────────────────────────────────── the files themselves (no tau2 needed)
def test_the_harness_imports_only_the_standard_library_and_the_bundle_writer():
    assert _top_imports(HARNESS) <= STDLIB | {"tau2_bundle", "tau3_retrieval"}, _top_imports(HARNESS)
    assert _top_imports(FAKE) <= STDLIB, _top_imports(FAKE)
    defined = {n.name for n in ast.parse(HARNESS.read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert {"stream_argv", "claude_env", "first_message", "ClaudeLink", "agent_class", "run_job", "main",
            "container_argv", "run_config", "CustomerCalls", "read_key", "load_job"} <= defined


def test_the_argv_is_the_ssb_solvers_without_the_prompt_or_a_turn_cap_plus_the_stream_flags():
    cl = _load(CC_SOLVER_LIB, "cc_solver_lib_for_tau2_cc_harness")
    ssb = cl.claude_argv("THE PROMPT", model="claude-sonnet-5-5", max_turns=45)
    ssb.remove("THE PROMPT")
    i = ssb.index("--max-turns")
    del ssb[i:i + 2]                                       # no turn cap at all
    ours = H.stream_argv("claude-sonnet-5-5")
    assert ours[:len(ssb)] == ssb, (ours, ssb)
    assert ours[len(ssb):] == ["--input-format", "stream-json", "--mcp-config", H.MCP_CONFIG]
    assert "--max-turns" not in ours and "max_turns" not in H.JOB_DEFAULTS
    assert json.loads(H.MCP_CONFIG) == {"mcpServers": {"tau2": {"type": "sdk", "name": "tau2"}}}
    assert H.CC_ENV == cl.CC_ENV and H.CC_DISALLOWED == cl.CC_DISALLOWED and H.THINKING_DISPLAY == cl.THINKING_DISPLAY
    assert not {"--effort", "--append-system-prompt", "--bare", "--system-prompt"} & set(ours)
    assert H.TAU2_ENV == {"ENABLE_TOOL_SEARCH": "false"}
    assert (H.JOB_DEFAULTS["model"], H.JOB_DEFAULTS["agent_time_s"], H.JOB_DEFAULTS["max_errors"]) == (
        "claude-sonnet-5-5", 1200.0, 10)
    assert (H.JOB_DEFAULTS["user"], H.JOB_DEFAULTS["user_sim_model"]) == ("user_simulator", "openrouter/openai/gpt-5-nano")
    assert (H.JOB_DEFAULTS["domain"], H.JOB_DEFAULTS["retrieval_variant"]) == ("banking_knowledge", "alltools")
    assert TR.VARIANT == "alltools" and Path(TR.__file__) == HARNESS.parent / "tau3_retrieval.py"
    assert H.JOB_DEFAULTS["customer_timeout_s"] == 120.0 and H.JOB_DEFAULTS["customer_attempts"] > 1


def test_the_first_message_is_the_pinned_text_byte_for_byte():
    md = FIRST_MESSAGE_TEXT
    assert hashlib.sha256(md.encode("utf-8")).hexdigest() == FIRST_MESSAGE_SHA256
    assert H.FIRST_MESSAGE_TEMPLATE == md
    assert H.first_message_sha256() == FIRST_MESSAGE_SHA256
    policy, customer = "POLICY with {customer} and {0} braces", "Hi, {policy} is mine."
    want = md.replace("{customer}", "\0C").replace("{policy}", policy).replace("\0C", customer)
    assert H.first_message(policy, customer) == want            # one pass: braces in the values are left alone
    assert H.first_message("P", "C") == md.replace("{policy}", "P").replace("{customer}", "C")
    assert f'"{H.GREETING}"' in md and H.GREETING == G.GREETING
    src = HARNESS.read_text()
    assert "AGENT_INSTRUCTION_CC" not in src


def test_claudes_environment_is_an_allow_list(monkeypatch):
    for k in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN", "AWS_SECRET_ACCESS_KEY",
              "PYTHONPATH", "TAU2_DATA_DIR"):
        monkeypatch.setenv(k, "must-not-pass")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    env = H.claude_env("/h/home", 3)
    assert "must-not-pass" not in json.dumps(env)
    assert set(env) <= set(H.ENV_PASS) | {"HOME", "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR"} | set(H.CC_ENV) | set(H.TAU2_ENV)
    assert env["HOME"] == "/h/home" and env["CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR"] == "3"
    assert "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR" not in H.claude_env("/h", None)


def test_the_container_launch_is_evoctl_enter_with_claudes_variables_in_the_argv(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "must-not-pass")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    job = H.JOB_DEFAULTS | _job(home="/home/evo", token_fd=3, container={
        "evoctl": ["/usr/local/bin/python3", "/opt/evoskill-evolver/bin/evoctl.py"], "cid": "cc-t1-x", "cwd": "/evo/work"})
    plan = H.launch_plan(job)
    a = plan["argv"]
    assert a[:9] == ["/usr/local/bin/python3", "/opt/evoskill-evolver/bin/evoctl.py", "enter", "--cid", "cc-t1-x",
                     "--cwd", "/evo/work", "--keep-fd3", "--"]
    assert a[9] == "/usr/bin/env"
    pairs = dict(x.split("=", 1) for x in a[10:a.index("claude")])
    assert pairs == {"HOME": "/home/evo", **H.CC_ENV, **H.TAU2_ENV, "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR": "3"}
    assert a[a.index("claude"):] == H.stream_argv()
    assert plan["pass_fds"] == (3,) and set(plan["env"]) <= set(H.ENV_PASS)       # evoctl's own environment
    assert "must-not-pass" not in json.dumps(plan)
    local = H.launch_plan(H.JOB_DEFAULTS | _job(claude_cmd=["/x/claude"]))
    assert local["argv"] == ["/x/claude"] + H.stream_argv()[1:] and local["pass_fds"] == ()
    assert "--keep-fd3" not in H.container_argv(job["container"], ["claude"], "m", "/home/evo", None)


def test_tau2_runs_with_no_step_cap_no_wall_and_empty_customer_llm_args():
    cfg = H.run_config(H.JOB_DEFAULTS | _job(), 7)
    assert cfg["max_steps"] == H.MAX_STEPS == 200 and cfg["timeout"] is None      # tau2.config.DEFAULT_MAX_STEPS
    assert cfg["llm_args_user"] == {} and cfg["llm_user"] == "openrouter/openai/gpt-5-nano"
    assert (cfg["max_errors"], cfg["seed"], cfg["retrieval_config"], cfg["agent"]) == (
        10, 7, "alltools", "claude_code_agent")


def test_seeds_paths_and_job_checks():
    assert H.seed_of("banking_knowledge/task_001", 0) == int(hashlib.sha256(
        b"banking_knowledge/task_001#s0").hexdigest(), 16) >> 225 < 2 ** 31
    assert H.seed_of("banking_knowledge/task_001", 0) != H.seed_of("banking_knowledge/task_001", 1)
    assert H.project_dir_name("/evo/work") == "-evo-work" and H.project_dir_name("/" + "a" * 250) is None
    assert H.transcript_path("/h", "/evo/work", "s1") == "/h/.claude/projects/-evo-work/s1.jsonl"
    assert H.transcript_path("/h", None, "s1") is None
    assert issubclass(H.Terminated, BaseException) and not issubclass(H.Terminated, Exception)


@pytest.mark.parametrize("job, why", [
    ({"workdir": "/w", "out_dir": "/o", "home": "/h"}, "missing ['task_file']"),
    ({"task_file": "/t", "workdir": "/w", "out_dir": "/o", "key": "x", "home": "/h"}, "unknown ['key']"),
    ({"task_file": "/t", "workdir": "/w", "out_dir": "/o", "key_fd": 1, "home": "/h"}, "key_fd is an fd number"),
    ({"task_file": "/t", "workdir": "/w", "out_dir": "/o", "claude_cmd": "claude", "home": "/h"},
     "claude_cmd is a non-empty"),
])
def test_a_bad_job_is_refused(tmp_path, job, why):
    p = tmp_path / "job.json"
    p.write_text(json.dumps(job))
    with pytest.raises(H.JobError, match=why.replace("[", r"\[").replace("]", r"\]")):
        H.load_job(p)


@pytest.mark.parametrize("extra, why", [
    ({"home": None}, "missing ['home']"),
    ({"home": "relative/home"}, "home is an absolute path"),
    ({"wall_s": 2400}, "unknown ['wall_s']"),
    ({"max_steps": 200}, "unknown ['max_steps']"),
    ({"max_turns": 400}, "unknown ['max_turns']"),
    ({"agent_time_s": "1200"}, "agent_time_s is a number > 0"),
    ({"agent_time_s": 0}, "agent_time_s is a number > 0"),
    ({"agent_time_s": float("nan")}, "agent_time_s is a number > 0"),
    ({"agent_time_s": True}, "agent_time_s is a number > 0"),
    ({"sample": "0"}, "sample is an integer >= 0"),
    ({"sample": -1}, "sample is an integer >= 0"),
    ({"sample": False}, "sample is an integer >= 0"),
    ({"seed": 2 ** 31}, "seed is null or an integer"),
    ({"seed": 1.5}, "seed is null or an integer"),
    ({"max_errors": 0}, "max_errors is an integer >= 1"),
    ({"customer_timeout_s": -1}, "customer_timeout_s is a number > 0"),
    ({"customer_attempts": 0}, "customer_attempts is an integer >= 1"),
    ({"customer_backoff_s": "5"}, "customer_backoff_s is a number >= 0"),
    ({"task_id": ""}, "task_id is null or a non-empty string"),
    ({"simulation_id": 3}, "simulation_id is null or a non-empty string"),
    ({"domain": 1}, "domain is a non-empty string"),
    ({"user_sim_model": ""}, "user_sim_model is a non-empty string"),
    ({"model": None}, "model is a non-empty string"),
    ({"claude_cmd": ["claude", ""]}, "claude_cmd is a non-empty list"),
    ({"token_fd": 2}, "token_fd is an fd number"),
    ({"key_fd": 5, "token_fd": 5}, "key_fd and token_fd are the same fd"),
    ({"close_grace_s": -1}, "close_grace_s is a number >= 0"),
    ({"mid_turn_grace_s": None}, "mid_turn_grace_s is a number >= 0"),
    ({"workdir": ""}, "workdir is a non-empty path"),
    ({"container": {"evoctl": ["evoctl"], "cid": "cc-x"}}, "container is null or"),
    ({"container": {"evoctl": ["evoctl"], "cid": "Bad Id", "cwd": "/evo/work"}}, "container is null or"),
    ({"container": {"evoctl": ["evoctl"], "cid": "cc-x", "cwd": "evo/work"}}, "container is null or"),
    ({"container": {"evoctl": ["evoctl"], "cid": "cc-x", "cwd": "/evo/work"}, "token_fd": 4}, "token_fd is 3 in a"),
    ({"retrieval_variant": ["alltools"]}, "retrieval_variant is 'alltools' (tau2's alltools is the only variant"),
    ({"retrieval_variant": "bm25_grep"}, "retrieval_variant is 'alltools' (tau2's alltools is the only variant"),
    ({"retrieval_variant": "AllTools"}, "retrieval_variant is 'alltools' (tau2's alltools is the only variant"),     # tau2's alias: one name
    ({"retrieval_variant": "alltools-qwen"}, "retrieval_variant is 'alltools' (tau2's alltools is the only variant"),
])
def test_every_job_field_is_checked(tmp_path, extra, why):
    p = tmp_path / "job.json"
    job = {**_job(), **extra}
    p.write_text(json.dumps({k: v for k, v in job.items() if not (k == "home" and v is None)}))
    with pytest.raises(H.JobError, match=re.escape(why)):
        H.load_job(p)


def test_a_good_job_loads_with_its_defaults(tmp_path):
    p = tmp_path / "job.json"
    p.write_text(json.dumps(_job(token_fd=3, key_fd=4, seed=0, agent_time_s=60)))
    job = H.load_job(p)
    assert job["agent_time_s"] == 60 and job["home"] == "/h" and job["container"] is None
    assert set(job) == set(H.JOB_REQUIRED) | set(H.JOB_DEFAULTS)
    assert job["retrieval_variant"] == "alltools"
    p.write_text(json.dumps(_job(retrieval_variant="alltools")))
    assert H.load_job(p)["retrieval_variant"] == "alltools"


def test_a_job_for_another_retrieval_variant_is_refused_with_exit_2(tmp_path, capsys):
    """One variant; a bm25_grep job is a stale job. The CLI refuses it before anything runs."""
    work = tmp_path / "work"
    work.mkdir()
    for variant in ("bm25_grep", "openai_embeddings", "alltools-qwen"):
        p = tmp_path / f"job_{variant}.json"
        p.write_text(json.dumps(_job(task_file=str(tmp_path / "task.json"), workdir=str(work),
                                     out_dir=str(tmp_path / "out"), home=str(tmp_path), retrieval_variant=variant)))
        assert H.main(["--job", str(p)]) == 2, variant
        err = capsys.readouterr().err
        assert "retrieval_variant is 'alltools' (tau2's alltools is the only variant" in err and "bm25_grep job is stale" in err, err
    assert not (tmp_path / "out").exists()                     # nothing ran: no harness.json, no claude


def test_read_key_is_bounded_and_closes_its_fd():
    r, w = os.pipe()
    os.write(w, b"sk-or-v1-TESTONLY-partial")                 # no newline, writer still open
    t0 = time.monotonic()
    with pytest.raises(H.JobError, match="no newline or EOF within"):
        H.read_key(r, timeout_s=0.5)
    assert 0.4 < time.monotonic() - t0 < 5
    with pytest.raises(OSError):
        os.fstat(r)
    os.close(w)
    r, w = os.pipe()
    os.write(w, b"  sk-or-v1-TESTONLY-line \nrest")            # a newline ends it, the writer may stay open
    assert H.read_key(r, timeout_s=5) == "sk-or-v1-TESTONLY-line"
    os.close(w)
    r, w = os.pipe()
    os.close(w)
    with pytest.raises(H.JobError, match="carried no key"):
        H.read_key(r, timeout_s=5)


def test_home_is_never_the_harness_own(tmp_path, monkeypatch):
    own = tmp_path / "own_home"
    own.mkdir()
    (tmp_path / "work").mkdir()
    (tmp_path / "task.json").write_text("{}")
    monkeypatch.setenv("HOME", str(own))
    job = H.JOB_DEFAULTS | _job(task_file=str(tmp_path / "task.json"), workdir=str(tmp_path / "work"),
                                out_dir=str(tmp_path / "out"), home=str(own))
    with pytest.raises(H.JobError, match="own HOME"):
        H.run_job(job)
    with pytest.raises(H.JobError, match="is not a directory"):
        H.run_job(job | {"home": str(tmp_path / "nope")})


def test_a_late_tool_use_block_never_lends_its_id_to_a_later_call(tmp_path):
    link = H.ClaudeLink([], {}, "/", tmp_path)
    blk = lambda i: {"type": "tool_use", "id": i, "name": "mcp__tau2__get", "input": {"a": 1}}   # noqa: E731
    asst = lambda *bs: {"type": "assistant", "message": {"content": list(bs)}}                  # noqa: E731
    c1 = link._call("r1", 1, {"name": "get", "arguments": {"a": 1}})             # before its block: a counter id
    link._dispatch(asst(blk("toolu_A")))                                          # its block, late
    c2 = link._call("r2", 2, {"name": "get", "arguments": {"a": 1}})             # the same call again
    link._dispatch(asst(blk("toolu_B")))
    assert (c1["id"], c2["id"]) == ("tau2_call_0001", "tau2_call_0002")
    assert link.late_ids == {"tau2_call_0001": "toolu_A", "tau2_call_0002": "toolu_B"}
    link._dispatch(asst(blk("toolu_C")))                                          # a block before its call
    c3 = link._call("r3", 3, {"name": "get", "arguments": {"a": 1}})
    c4 = link._call("r4", 4, {"name": "get", "arguments": {"a": 1}, "_meta": {"claudecode/toolUseId": "toolu_C"}})
    assert c3["id"] == "toolu_C" and c3["id_source"] == "stream"
    assert c4["id"] != "toolu_C" and c4["id_source"] == "counter"               # a _meta id never used twice
    assert len({c["id"] for c in (c1, c2, c3, c4)}) == 4


def test_api_retries_and_rejections_mark_a_stall_until_the_api_answers(tmp_path):
    link = H.ClaudeLink([], {}, "/", tmp_path)
    assert link.api_stall() is None
    link._dispatch({"type": "system", "subtype": "api_retry", "attempt": 1, "max_retries": 10,
                    "retry_delay_ms": 500, "error_status": 529, "error": "overloaded"})
    assert "retrying the API" in link.api_stall() and link.n_api_retries == 1
    link._dispatch({"type": "assistant", "message": {"content": [{"type": "text", "text": "ok"}]}})
    assert link.api_stall() is None
    link._dispatch({"type": "rate_limit_event", "rate_limit_info": {"status": "rejected", "rateLimitType": "five_hour"}})
    assert "rate limited" in link.api_stall()
    link._dispatch({"type": "rate_limit_event", "rate_limit_info": {"status": "allowed"}})
    assert link.api_stall() is None


def test_a_signal_before_the_conversation_exits_75_and_restores_the_handlers(tmp_path):
    import signal
    import threading
    before = signal.getsignal(signal.SIGTERM)
    r, w = os.pipe()                                          # a key fd whose writer never finishes
    p = tmp_path / "job.json"
    p.write_text(json.dumps(_job(key_fd=r)))
    timer = threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGTERM))
    timer.start()
    try:
        t0 = time.monotonic()
        assert H.main(["--job", str(p)]) == 75
        assert time.monotonic() - t0 < 10
    finally:
        timer.cancel()
        os.close(w)
    assert signal.getsignal(signal.SIGTERM) is before
    with pytest.raises(OSError):
        os.fstat(r)                                           # read_key closed the key fd on the way out


def test_an_exception_escaping_run_job_clears_the_bundle_path_and_writes_harness_json(tmp_path, monkeypatch):
    if importlib.util.find_spec("tau2") is not None:
        pytest.skip("this interpreter has tau2: the escape is provoked by its absence")
    for k in ("PYTHON_DOTENV_DISABLED", "LITELLM_LOCAL_MODEL_COST_MAP", "TAU2_DATA_DIR"):
        monkeypatch.setenv(k, "restored-after-the-test")
    monkeypatch.setenv("HOME", str(tmp_path))
    work, home, out = tmp_path / "work", tmp_path / "claude_home", tmp_path / "out"
    work.mkdir()
    home.mkdir()
    task = tmp_path / "a" / "b" / "c" / "d" / "task.json"
    task.parent.mkdir(parents=True)
    task.write_text("{}")
    (work / "tau2").mkdir()
    (work / "tau2" / "simulation.json").write_text('{"old": true}')
    p = tmp_path / "job.json"
    p.write_text(json.dumps(_job(task_file=str(task), workdir=str(work), out_dir=str(out), home=str(home))))
    assert H.main(["--job", str(p)]) == 75                   # `import tau2` fails inside run_job
    assert not (work / "tau2").exists()
    hj = json.loads((out / "harness.json").read_text())
    assert hj["end"]["reason"] == "infrastructure_error" and hj["exit_code"] == 75 and not hj["graded_bundle"]
    assert hj["end"]["detail"].startswith("ModuleNotFoundError")


def test_the_cli_refuses_a_bad_job_with_exit_2_and_drops_a_key_from_its_environment(tmp_path):
    if not G.TAU3_PY.is_file():
        pytest.skip("no tau3 venv")
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"task_file": str(tmp_path / "nope.json"), "workdir": str(tmp_path),
                               "out_dir": str(tmp_path / "out"), "home": str(tmp_path)}))
    env = _child_env(OPENROUTER_API_KEY=KEY_ENV_TEXT)
    p = subprocess.run([str(G.TAU3_PY), "-B", str(HARNESS), "--job", str(job)], capture_output=True, text=True,
                       env=env, timeout=120)
    assert p.returncode == 2, (p.returncode, p.stderr[-2000:])
    assert "task_file" in p.stderr and KEY_ENV_TEXT not in p.stdout + p.stderr


# ─────────────────────────────────────────────────────────────── golden conversations
def test_golden_conversations_are_tau2s_own_and_grade_1(runs):
    res = runs["res"]
    for name in runs["golden"] + ["container:task_017"]:
        r = res[name]
        assert r["rc"] == 0, (name, r["harness"])
        assert r["termination"] == "user_stop" and r["same_as_golden"], (name, r["termination"], r["sim_shape_tail"])
        assert r["grade"] == {"score": 1.0, "type": "float"}, (name, r["grade"])
        assert r["bundle_files"] == GRADED_FILES, (name, r["bundle_files"])
        assert r["agent_call_ids"] == r["golden_agent_ids"], name           # the toolu_ ids Claude Code reported
        assert r["harness"]["end"] == {"reason": "user_stop", "detail": None} and r["harness"]["graded_bundle"], name
    if not os.environ.get("EVOSKILL_TAU3_HARNESS_ALL"):
        assert len(runs["golden"]) == len(GOLDEN)


def test_the_protocol_carried_the_verified_shapes_both_ways(runs):
    res = runs["res"]
    for name in runs["golden"]:
        r, f = res[name], res[name]["fake"]
        assert f["argv_problems"] == [] and f["protocol_problems"] == [], (name, f["argv_problems"], f["protocol_problems"])
        assert f["argv"] == H.stream_argv()[1:], name
        init = f["initialize"]
        assert set(init) == {"type", "request_id", "request"} and init["type"] == "control_request", (name, init)
        assert init["request"] == {"subtype": "initialize", "hooks": None, "sdkMcpServers": ["tau2"]}, name
        assert init["request_id"].startswith("req_1_"), name
        assert r["stdin"][0] == init, name                                  # the harness's own log of what it sent
        assert f["mcp"]["initialize"] == {"jsonrpc": "2.0", "id": 0, "result": {
            "protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
            "serverInfo": {"name": "tau2", "version": "1.0.0"}}}, (name, f["mcp"]["initialize"])
        assert f["mcp"]["initialized"] == {"jsonrpc": "2.0", "result": {}}, name
        assert sorted(t["name"] for t in f["tools"]) == r["env_tools"], name
        assert all(t["inputSchema"].get("type") == "object" and t["description"] for t in f["tools"]), name
        users = [m for m in r["stdin"] if m.get("type") == "user"]
        assert all(set(m) == {"type", "session_id", "message", "parent_tool_use_id"} and m["session_id"] == ""
                   and m["parent_tool_use_id"] is None and m["message"]["role"] == "user" for m in users), name
        assert f["user_messages"] == [m["message"]["content"] for m in users], name
        assert f["user_messages"][0] == r["expected_first_message"] == r["first_message_from_file"], name
        assert set(f["user_messages"][1:]) <= {G.TO_AGENT}, (name, f["user_messages"][1:])
        replies = [m for m in r["stdin"] if m.get("type") == "control_response"]
        assert replies and all(m["response"]["subtype"] == "success" and set(m["response"]) ==
                               {"subtype", "request_id", "response"} for m in replies), name
        assert len(f["tool_results"]) == len(r["golden_agent_ids"]), name
        for tr in f["tool_results"]:
            content, error = r["tool_messages"][tr["id"]]
            assert (tr["text"], tr["isError"], tr["content_types"]) == (content, error, ["text"]), (name, tr["id"])
            assert tr["jsonrpc_id"] == tr["sent_id"], name
        assert r["stream_types"].count("result") == f["turns"] and "system" in r["stream_types"], name
        assert r["fake_alive"] is False, name
    assert res["golden:task_017"]["harness"]["cc"]["tool_calls"] == len(res["golden:task_017"]["golden_agent_ids"])
    assert res["golden:task_017"]["harness"]["cc"]["id_sources"] == {"stream": len(res["golden:task_017"]["golden_agent_ids"])}


def test_the_first_message_starts_with_the_alltools_policy_and_the_customers_opening(runs):
    r = runs["res"]["golden:task_001"]
    first = r["fake"]["user_messages"][0]
    assert first.startswith("You are a customer service agent that helps the user according to the <policy>")
    assert first.endswith("\n\n" + G.OPENING + "\n\nPlease continue the conversation. Your output is sent to the "
                          "user, in full, verbatim; any tool calling turns or reasoning tokens remain invisible to "
                          "the user.\n")
    assert len(first) > 2000 and "{policy}" not in first and "{customer}" not in first and "{{" not in first
    # tau2's alltools policy (prompts/all_tools.md): three knowledge base tools, the dense one as tau2 describes it
    for heading in ("### `KB_search_bm25`", "### `KB_search_dense`", "### `shell`"):
        assert heading in first, heading
    assert ("The `KB_search_dense` tool uses **OpenAI API** with embedding model `text-embedding-3-large` for dense "
            "retrieval.") in first
    shell_md = (DATA_ROOT / "tau2" / "domains" / G.DOMAIN / "prompts" / "components"
                / "shell_instructions.md").read_text()
    assert shell_md.strip() in first and "## Knowledge Base Access (shell)" in first   # the shell instructions, whole
    assert "KB_search_bm25" in first and "KB_search_dense" in first
    assert "`KB_search`" not in first and "`grep`" not in first                          # bm25_grep's tools are gone
    assert first == r["expected_first_message"]                # tau2's own alltools policy for this environment


def test_claude_code_is_listed_alltools_17_tools(runs):
    for name in runs["golden"] + ["container:task_017", f"probe_tools:{PROBE_TASK}"]:
        r = runs["res"][name]
        names = [t["name"] for t in r["fake"]["tools"]]
        assert len(names) == len(set(names)) == 17, (name, names)
        assert ALLTOOLS_KB <= set(names) and not {"KB_search", "grep"} & set(names), (name, names)
        assert sorted(names) == r["env_tools"], name
        cc = r["harness"]["cc"]
        assert cc["tau2_tools_listed"] == sorted(H.TOOL_PREFIX + n for n in names), name


def test_run_json_and_harness_json_record_the_session_and_its_transcript(runs):
    res = runs["res"]
    sha = hashlib.sha256(HARNESS.read_bytes()).hexdigest()
    for name in runs["golden"]:
        r = res[name]
        h, cc = r["run"]["harness"], r["run"]["harness"]["cc"]
        assert (h["name"], h["harness_sha256"], h["user"], h["sample"]) == ("tau2_cc_harness", sha, CUSTOMER, 0), name
        assert h["task_id"] == f"banking_knowledge/{r['tau2_id']}" and h["first_message_sha256"] == H.first_message_sha256()
        assert h["solver"]["argv"] == H.stream_argv() and h["user_sim_model"] == "openrouter/openai/gpt-5-nano", name
        assert h["user_llm_args"] == {} and r["customer_llm_args"] == [{}], name
        assert cc["session_id"] == r["session_id"] and cc["transcript"] == r["fake"]["transcript"], (name, cc)
        assert r["transcript_exists"] and cc["claude_code_version"] == "2.1.281" and cc["rc"] == 0, (name, cc)
        assert cc["mcp_servers"] == [{"name": "tau2", "status": "connected", "source": "sdk"}], name
        assert cc["mcp_tools_listed"] is True and cc["api_retries"]["count"] == 0, name
        assert r["harness"]["cc"]["transcript"] == cc["transcript"], name
        run = r["run"]
        assert run["seed"] == H.seed_of(h["task_id"], 0) and run["simulation_id"] == f"{h['task_id']}#s0", name
        assert (run["wall_s"], run["max_steps"], run["max_errors"]) == (None, 200, 10), name
        assert run["end"] == {"reason": "user_stop", "detail": None}, name
        clock = h["agent_time"]
        assert clock["limit_s"] == 1200.0 and clock["turns"] == r["fake"]["turns"] == len(clock["turns_s"]), name
        assert 0 < clock["used_s"] < 60 and math.isclose(clock["used_s"], sum(clock["turns_s"]), abs_tol=0.01), name
        assert h["customer"]["calls"] >= 2 and h["customer"]["failed_attempts"] == 0, name
        assert r["harness"]["agent_time"] == clock and r["harness"]["cleared_before"] is None, name
        assert run["retrieval_variant"] == "alltools" and run["retrieval_kwargs"] == {}, name
        assert h["retrieval"] == r["harness"]["retrieval"] == _retrieval_record(key=False), (name, h["retrieval"])


def _retrieval_record(*, key: bool) -> dict:
    """What install() records for this data root (scripts/tau3_cc/tau3_retrieval.py; no secret, key a boolean)."""
    return {"variant": "alltools",
            "embedder": {"type": "openai", "model": "text-embedding-3-large", "transport": "openrouter",
                         "openrouter_model": "openai/text-embedding-3-large",
                         "provider": {"order": ["azure"], "allow_fallbacks": False}, "key": key},
            "cache": {"dir": str(TR.cache_dir(DATA_ROOT)), "readonly": True, "files": TR.cache_files(DATA_ROOT)},
            "shell_deny_read": list(TR.deny_read_paths(DATA_ROOT)),
            "module_sha256": hashlib.sha256((HARNESS.parent / "tau3_retrieval.py").read_bytes()).hexdigest()}


def test_the_retrieval_record_names_the_cache_files_the_deny_list_and_no_key():
    if not TR.cache_dir(DATA_ROOT).is_dir():
        pytest.skip(f"{TR.cache_dir(DATA_ROOT)} is absent (scripts/tau3_build_embeddings_cache.py)")
    rec = _retrieval_record(key=False)
    assert len(rec["cache"]["files"]) == 2 and any(k.endswith(".pkl") for k in rec["cache"]["files"])
    assert {"/opt", "/root", "/home", "/Users", str(DATA_ROOT.resolve())} <= set(rec["shell_deny_read"])


def test_the_container_launch_reaches_claude_through_evoctl_with_the_token_on_fd_3(runs):
    r = runs["res"]["container:task_017"]
    e, f = r["evoctl"], r["fake"]
    assert e["problems"] == [] and e["cid"] == "cc-test-task-017" and e["cwd"] == "/evo/work" and e["keep_fd3"]
    assert dict(p.split("=", 1) for p in e["pairs"]) == {
        "HOME": str(Path(r["harness"]["cc"]["transcript"]).parents[3]), **H.CC_ENV, **H.TAU2_ENV,
        "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR": "3"}
    assert e["claude_argv"] == H.stream_argv() and 3 in e["fds"]
    assert f["argv_problems"] == [] and f["token_sha256"] == hashlib.sha256(TOKEN_TEXT.encode()).hexdigest()
    assert set(f["env"]) - {"__CF_USER_TEXT_ENCODING"} == {"PATH", "LC_ALL", "HOME", "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR"} \
        | set(H.CC_ENV) | set(H.TAU2_ENV)
    h = r["harness"]
    assert h["where"] == "container cc-test-task-017" and set(h["env_names"]) <= set(H.ENV_PASS)
    assert h["cc"]["cwd"] == "/evo/work" and h["cc"]["transcript"].endswith(f"/.claude/projects/-evo-work/{r['session_id']}.jsonl")
    assert r["transcript_exists"] and r["fake_alive"] is False
    assert r["token_fd_closed_by_harness"] is True


def test_the_customers_key_reaches_litellm_through_the_fd_and_nowhere_else(runs):
    r = runs["res"]["secrets:task_026"]
    assert r["rc"] == 0 and r["grade"] == {"score": 1.0, "type": "float"}, (r["rc"], r["grade"])
    assert r["key_loaded"] is True and r["key_fd_closed"] is True
    assert r["harness"]["retrieval"] == _retrieval_record(key=True)        # the dense search got the key, in memory
    assert r["harness"]["user_sim_key"] == "fd" and r["harness"]["dropped_env"] == ["OPENROUTER_API_KEY"]
    assert r["driver_env_has_openrouter"] is False
    fake = r["fake"]
    for secret in (KEY_FD_TEXT, KEY_ENV_TEXT):
        assert secret not in json.dumps(fake["argv"]) and secret not in json.dumps(fake["env"])
    leaks = []
    for p in runs["base"].rglob("*"):
        if p.is_file() and not p.is_symlink():
            data = p.read_bytes()
            leaks += [str(p) for s in (KEY_FD_TEXT, KEY_ENV_TEXT, TOKEN_TEXT) if s.encode() in data]
    assert not leaks, leaks
    for name, rr in runs["res"].items():                     # claude's whole environment, in every local run
        if rr.get("fake") and not rr.get("evoctl"):
            got = set(rr["fake"]["env"]) - {"__CF_USER_TEXT_ENCODING"}      # macOS adds this one itself
            assert got == {"PATH", "LC_ALL", "HOME"} | set(H.CC_ENV) | set(H.TAU2_ENV), (name, sorted(got))


def test_tau2_gets_empty_customer_args_and_the_real_user_simulator_keeps_them(runs):
    p = runs["res"]["probe:config"]["probe"]
    assert p["tau2_default_user_args"] != {}                  # tau2's own default (temperature) is really replaced
    assert p["llm_args_user"] == {} and p["user_llm_args"] == {} and p["user_class"] == "UserSimulator"
    assert p["llm_user"] == p["user_llm"] == "openrouter/openai/gpt-5-nano"
    assert p["user_llm_args_seeded"] == {"seed": 12345}       # tau2's set_seed adds only the rollout's seed
    assert (p["max_steps"], p["timeout"], p["max_errors"], p["retrieval_config"]) == (200, None, 10, "alltools")
    assert p["wrapped"] is True


# ─────────────────────────────────────────────────────────────── alltools: the shell and the dense search
def test_the_shell_reads_the_knowledge_base_and_nothing_of_the_run(runs):
    """`cd / && cat <data root>/.../task_004.json` passes tau2's own guard (no '..', '~' or absolute path); the bridge's
    srt denyRead (the data root, /opt, /root, /home, /Users) is what stops it. `ls` still lists the knowledge base."""
    r = runs["res"][f"probe_tools:{PROBE_TASK}"]
    assert r["rc"] == 0 and r["termination"] == "user_stop", (r["rc"], r["termination"], r["harness"]["end"])
    assert r["grade"] == {"score": 1.0, "type": "float"}, r["grade"]           # read-only probes change no state
    res = {tr["id"]: tr for tr in r["fake"]["tool_results"]}
    cat, ls_, dense, bm25 = (res[i] for i in PROBE_IDS)
    task_text = (DATA_ROOT / "tau2" / "domains" / G.DOMAIN / "tasks" / f"{PROBE_TASK}.json").read_text()
    assert task_text[:200] and task_text[:40] not in cat["text"], cat
    assert (cat["text"], cat["isError"]) == ("(no output)", False), cat
    for word in ("evaluation_criteria", "user_scenario", "actions", "task_004"):
        assert word not in cat["text"], (word, cat)
    lines = ls_["text"].splitlines()
    assert ls_["isError"] is False and len(lines) == 3 and all(x.endswith(".md") for x in lines), ls_
    assert {x for x in lines if not x.startswith("doc_")} <= {"INDEX.md"} and any(x.startswith("doc_") for x in lines)
    assert bm25["isError"] is False and "ID: doc_" in bm25["text"], bm25["text"][:300]
    # tau2 recorded the calls and their results, paired by id, ahead of the golden actions
    assert r["agent_call_ids"] == PROBE_IDS + r["golden_agent_ids"], r["agent_call_ids"]
    assert {i: r["tool_messages"][i] for i in PROBE_IDS} == {
        i: [res[i]["text"], res[i]["isError"]] for i in PROBE_IDS}
    h = r["harness"]
    assert h["retrieval"] == _retrieval_record(key=False), h["retrieval"]
    bins = h["shell_bins"]                                    # what the shell ran on: this child's PATH
    assert bins["PATH"] == TOOL_PATH and bins["srt"] == shutil.which("srt", path=TOOL_PATH), bins
    assert bins["rg"] == shutil.which("rg", path=TOOL_PATH) and set(bins) == {"PATH", *H.SHELL_BINS}, bins
    assert h["tool_errors"] == sum(1 for _c, e in r["tool_messages"].values() if e) >= 1, h["tool_errors"]
    assert r["run"]["harness"]["retrieval"] == h["retrieval"]


def test_without_the_hardening_the_same_probe_reads_the_task_file(tmp_path):
    """The control that makes the probe above mean something: tau2's alltools shell with install(harden_shell=False)
    hands the task file (its expected actions) to the agent; with the hardening, nothing."""
    _need_tau3()
    code = r'''
import json, sys
sys.path[:0] = sys.argv[1:3]
import tau3_golden as G
import tau3_retrieval as TR
TR.install(G.data_root(), key=None, harden_shell=sys.argv[4] == "hardened")
t = G.load_task(G.data_root() / "tau2" / "domains" / G.DOMAIN / "tasks" / "task_004.json")
env = G.fresh_env(t, variant=TR.VARIANT)
deny = json.loads(open(env.tools._sandbox.settings_path).read())["filesystem"]["denyRead"]
print(json.dumps({"out": env.tools.shell(command=sys.argv[3]), "deny": deny}))
'''
    py = tmp_path / "probe.py"
    py.write_text(code)
    task_text = (DATA_ROOT / "tau2" / "domains" / G.DOMAIN / "tasks" / f"{PROBE_TASK}.json").read_text()
    got = {}
    for mode in ("bare", "hardened"):
        p = subprocess.run([str(G.TAU3_PY), "-B", str(py), str(TESTS), str(HARNESS.parent), SHELL_PROBE, mode],
                           capture_output=True, text=True, env=_child_env(HOME=tmp_path), cwd=str(tmp_path),
                           timeout=300)
        assert p.returncode == 0, (mode, p.stderr[-3000:])
        got[mode] = json.loads(p.stdout.strip().splitlines()[-1])
    assert got["bare"]["out"] == task_text[:200], got["bare"]["out"]           # the leak tau2 alone allows
    assert got["hardened"]["out"] == "(no output)", got["hardened"]["out"]
    assert set(TR.deny_read_paths(DATA_ROOT)) <= set(got["hardened"]["deny"])
    assert not set(TR.deny_read_paths(DATA_ROOT)) & set(got["bare"]["deny"])


def test_the_dense_search_without_a_key_is_a_tool_error_the_conversation_survives(runs):
    r = runs["res"][f"probe_tools:{PROBE_TASK}"]
    dense = {tr["id"]: tr for tr in r["fake"]["tool_results"]}[PROBE_IDS[2]]
    assert dense["isError"] is True and dense["text"].startswith("Error: an embedding request in an offline process "
                                                                 "(no OpenRouter key)"), dense
    assert r["termination"] == "user_stop" and r["grade"] == {"score": 1.0, "type": "float"}


def test_a_dense_search_outage_is_an_infrastructure_end_and_never_reaches_the_agent(runs):
    """A key, but OpenRouter at a dead local port (the child points tau3_retrieval there, 2 tries): the bridge ends the
    rollout as infrastructure_error (exit 75, no simulation.json, re-run by the leg), the failed call's result never
    reaches Claude Code, and nothing records the key (the leak scan of the key test covers every file)."""
    r = runs["res"]["dense_outage:task_026"]
    assert r["rc"] == H.EXIT_INFRA and r["termination"] == "infrastructure_error", (r["rc"], r["termination"])
    assert r["bundle_files"] == INFRA_FILES and r["grade"]["raised"] == "FileNotFoundError", r["bundle_files"]
    end = r["harness"]["end"]
    assert end["reason"] == "infrastructure_error" and end["detail"] == (
        "KB_search_dense: the embedding request failed 2x (APIConnectionError (status None)); an infrastructure end, "
        "re-run"), end
    o = r["harness"]["dense_outage"]
    assert (o["attempts"], o["timeout_s"], o["error"], o["texts"]) == (2, 10.0, "APIConnectionError (status None)", 1)
    dense = [tr for tr in r["fake"]["tool_results"] if tr["id"] == DENSE_ID]   # the fake logs every call it made
    assert dense and all(tr["jsonrpc_id"] is None and tr["text"] is None for tr in dense), dense  # no reply ever came
    assert KEY_FD_TEXT not in json.dumps(r["harness"]) and KEY_ENV_TEXT not in json.dumps(r["harness"])
    ok = runs["res"]["secrets:task_026"]["harness"]
    assert ok["dense_outage"] is None and ok["end"]["reason"] != "infrastructure_error"

def test_the_shell_sandboxes_temp_dirs_are_gone_when_the_process_exits(runs):
    dirs = runs["sandboxes"]
    assert len(dirs) >= len(runs["res"]), len(dirs)            # one per environment built, harness and grader
    assert all("agentic_search_" in d for d in dirs)
    assert [d for d in dirs if Path(d).exists()] == []


# ─────────────────────────────────────────────────────────────── the agent clock
def test_the_customers_time_never_counts(runs):
    r = runs["res"]["slow_customer:task_026"]
    assert r["rc"] == 0 and r["termination"] == "user_stop" and r["same_as_golden"], (r["termination"], r["harness"]["end"])
    assert r["grade"] == {"score": 1.0, "type": "float"}
    clock, cust = r["harness"]["agent_time"], r["harness"]["customer"]
    assert clock["limit_s"] == 2.0 and clock["used_s"] < 2.0 < cust["seconds"], (clock, cust)
    assert r["duration"] > 2.0                                 # the conversation outlasted the agent's limit


def test_claudes_own_time_runs_out_as_a_timeout_graded_0(runs):
    r = runs["res"]["agent_clock:task_026"]
    assert r["rc"] == 0 and r["termination"] == "timeout", (r["termination"], r["harness"]["end"])
    assert r["harness"]["end"]["detail"] == "Claude Code used its 1.2 s of agent time"
    clock = r["harness"]["agent_time"]
    assert clock["turns"] == 2 and 1.2 <= clock["used_s"] < 1.2 + 0.5, clock
    assert clock["turns_s"][0] >= 0.8 and r["harness"]["customer"]["seconds"] >= 0.3
    assert r["grade"] == {"score": 0.0, "type": "float"} and r["fake_alive"] is False
    assert r["prefix_of_golden"], r["sim_shape_tail"]


def test_a_hanging_claude_is_stopped_at_the_limit_and_graded_0(runs):
    r = runs["res"]["hang:task_043"]
    assert r["rc"] == 0 and r["termination"] == "timeout", (r["termination"], r["harness"]["end"])
    assert r["harness"]["end"]["detail"] == "Claude Code used its 3 s of agent time"
    assert 2.9 <= r["duration"] < 3.0 + 5.0, r["duration"]
    assert r["seconds"] < 3.0 + 30.0, r["seconds"]
    assert r["fake"]["failed"] == "hang" and r["fake_alive"] is False
    assert r["grade"] == {"score": 0.0, "type": "float"}, r["grade"]


def test_a_hang_after_the_api_retry_recovered_is_still_the_solvers_timeout(runs):
    r = runs["res"]["retry_recovered:task_043"]
    assert r["rc"] == 0 and r["termination"] == "timeout", (r["termination"], r["harness"]["end"])
    assert r["harness"]["cc"]["api_retries"]["count"] == 1
    assert r["grade"] == {"score": 0.0, "type": "float"}


# ─────────────────────────────────────────────────────────────── how conversations end
@pytest.mark.parametrize("name, detail", [
    ("empty:task_043", "Claude Code ended its turn with no text"),
    ("max_turns:task_058", "Claude Code reached a turn limit"),
    ("end_conversation:task_098", "Claude Code ended the session with EndConversation"),
    ("prompt_too_long:task_043", "Claude Code's turn ended in its own error {'subtype': 'success', "
                                 "'api_error_status': 400, 'terminal_reason': 'prompt_too_long'"),
    ("malformed:task_026", "Claude Code's turn ended in its own error {'subtype': 'success', 'api_error_status': None, "
                           "'terminal_reason': 'malformed_tool_use_exhausted'"),
])
def test_claude_side_failures_are_agent_errors_that_grade_0(runs, name, detail):
    r = runs["res"][name]
    assert r["rc"] == 0 and r["termination"] == "agent_error", (r["rc"], r["termination"], r["harness"]["end"])
    assert r["harness"]["end"]["detail"].startswith(detail), r["harness"]["end"]
    assert r["run"]["end"] == {"reason": "agent_error", "detail": r["harness"]["end"]["detail"]}
    assert r["bundle_files"] == GRADED_FILES
    assert r["grade"] == {"score": 0.0, "type": "float"}, r["grade"]
    assert r["prefix_of_golden"], r["sim_shape_tail"]


def test_a_stop_token_in_claudes_reply_is_an_agent_stop_that_grades(runs):
    r = runs["res"]["stop_token:task_043"]
    assert r["rc"] == 0 and r["termination"] == "agent_stop", (r["termination"], r["harness"]["end"])
    assert r["sim_shape_tail"][-1][:2] == ["assistant", G.CLOSING + " ###STOP###"]
    assert r["grade"] == {"score": 1.0, "type": "float"}, r["grade"]


@pytest.mark.parametrize("name, detail", [
    ("crash:task_043", "claude exited (rc 1) before ending its turn; stderr: 'fake claude: crashing as scripted"),
    ("crash0:task_001", "claude exited (rc 1) before ending its turn"),
    ("rate_limit:task_026", "Claude Code's turn ended in an error {'subtype': 'success', 'api_error_status': 429"),
    ("server_failed:task_001", "the tau2 MCP server is not connected"),
    ("no_tools:task_001", "the tau2 MCP server never listed its tools before Claude Code answered"),
    ("no_claude:task_001", "claude did not start: FileNotFoundError"),
    ("api_retry_hang:task_043", "the 2 s of agent time ran out while Claude Code was retrying the API"),
    ("rate_limit_hang:task_001", "the 2 s of agent time ran out while Claude Code was rate limited"),
    ("customer_fails:task_026", "CustomerFailed: the customer's call 2 failed all 3 attempts"),
    ("sigterm:task_026", "the harness received SIGTERM"),
    ("sigint:task_043", "the harness received SIGINT"),
])
def test_infrastructure_ends_exit_75_and_leave_nothing_to_grade(runs, name, detail):
    r = runs["res"][name]
    assert r["rc"] == 75 and r["termination"] == "infrastructure_error", (r["rc"], r["termination"], r["harness"])
    assert r["harness"]["end"]["reason"] == "infrastructure_error" and not r["harness"]["graded_bundle"]
    assert r["harness"]["exit_code"] == 75
    assert r["harness"]["end"]["detail"].startswith(detail), r["harness"]["end"]["detail"]
    assert r["bundle_files"] == INFRA_FILES, r["bundle_files"]
    assert r["simulation_json"] is None
    assert r["run"]["end"]["reason"] == "infrastructure_error"
    assert r["grade"]["raised"] == "FileNotFoundError", r["grade"]           # grader_failed, never a 0
    assert r["prefix_of_golden"], r["sim_shape_tail"]
    if r.get("fake"):
        assert r["fake_alive"] is False
    if name == "crash:task_043":
        assert r["agent_call_ids"] == r["golden_agent_ids"][:2]


def test_a_signal_kills_claude_and_writes_harness_json(runs):
    for name, sig in (("sigterm:task_026", "SIGTERM"), ("sigint:task_043", "SIGINT")):
        r = runs["res"][name]
        assert r["harness"]["signal"] == sig and r["fake"]["failed"] == "signal", name
        assert r["fake_alive"] is False and r["harness"]["cc"]["rc"] is not None, name
        assert r["seconds"] < 30, (name, r["seconds"])
        assert r["harness"]["agent_time"]["turns"] == 1, (name, r["harness"]["agent_time"])   # the cut turn counted


def test_api_stalls_at_the_limit_are_infrastructure_with_their_retries_recorded(runs):
    r = runs["res"]["api_retry_hang:task_043"]
    ret = r["harness"]["cc"]["api_retries"]
    assert ret["count"] == 2 and [e["error_status"] for e in ret["events"]] == [529, None]
    assert ret["delay_ms_total"] == 500 + 1000
    assert r["harness"]["agent_time"]["used_s"] >= 2.0
    rl = runs["res"]["rate_limit_hang:task_001"]["harness"]["cc"]["rate_limit"]
    assert rl["status"] == "rejected"


def test_a_customer_call_that_raises_then_hangs_is_retried_as_the_same_call(runs):
    r = runs["res"]["customer_retry:task_026"]
    assert r["rc"] == 0 and r["termination"] == "user_stop" and r["same_as_golden"], (r["termination"], r["harness"]["end"])
    assert r["grade"] == {"score": 1.0, "type": "float"}
    cust = r["harness"]["customer"]
    assert cust["failed_attempts"] == 2 and [f["call"] for f in cust["failures"]] == [2, 2], cust
    assert cust["failures"][0]["error"] == "RuntimeError: scripted customer failure"
    assert cust["failures"][1]["error"] == "no answer within 1 s"
    assert r["customer_invocations"][:4] == ["ok", "raise", "hang", "ok"]
    assert r["harness"]["agent_time"]["used_s"] < 1.0           # the retries never counted as agent time
    f = runs["res"]["customer_fails:task_026"]
    assert f["harness"]["customer"]["failed_attempts"] == 3 and f["customer_invocations"] == ["ok", "raise", "raise", "raise"]


def test_a_late_block_without_meta_keeps_every_id_unique(runs):
    r = runs["res"]["late_ids:task_026"]
    assert r["rc"] == 0 and r["fake"]["protocol_problems"] == []
    ids = r["agent_call_ids"]
    assert len(ids) == len(set(ids)) == len(r["golden_agent_ids"]) + 1
    assert all(i.startswith("tau2_call_") for i in ids), ids
    late = r["harness"]["cc"]["late_ids"]
    sent = [tr["id"] for tr in r["fake"]["tool_results"]]
    assert [late[i] for i in ids] == sent and len(set(sent)) == len(sent), (late, sent)


def test_a_planted_tau2_is_replaced_and_its_target_untouched(runs):
    r = runs["res"]["planted:task_001"]
    assert r["rc"] == 0 and r["grade"] == {"score": 1.0, "type": "float"}, (r["rc"], r["grade"])
    assert r["bundle_is_link"] is False and r["bundle_files"] == GRADED_FILES
    assert r["planted_target_files"] == ["x"] and r["harness"]["cleared_before"] == "link"
    d = runs["res"]["planted_during:task_031"]
    assert d["rc"] == 0 and d["grade"] == {"score": 1.0, "type": "float"}, (d["rc"], d["grade"])
    assert d["harness"]["cleared_before"] is None and d["harness"]["cleared_after"] == "dir"
    assert d["simulation_json"] != {"planted": True} and d["bundle_files"] == GRADED_FILES


def test_an_old_bundle_never_survives_an_infrastructure_end_or_a_bad_job(runs):
    r = runs["res"]["old_bundle_infra:task_001"]
    assert r["rc"] == 75 and r["harness"]["cleared_before"] == "dir" and r["simulation_json"] is None
    assert r["bundle_files"] == INFRA_FILES and r["grade"]["raised"] == "FileNotFoundError"
    b = runs["res"]["old_bundle_bad_job:task_001"]
    assert b["rc"] == 2 and b["bundle_exists"] is False and b["harness"] is None and b["fake"] is None
    assert b["grade"]["raised"] == "FileNotFoundError"
