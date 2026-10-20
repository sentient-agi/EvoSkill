"""The evolver's HIGH-FIDELITY rollout evidence: every train rollout of a batch, in full, with its grade.

WHY. The evolver should see (1) high-fidelity rollout traces, (2) the path to the snapshot of each
epoch's available skills, so it knows which skills were NOT loaded as well as which were, without the
skills' full text in its prompt, and (3) each rollout's score directly. Per-task digests clip every
tool call/output over 1,800 chars to 1,200 + 600 and never show the solver's REASONING (a large share
of the conversation), so they cannot serve (1).

WHAT THIS ADDS, inside each epoch's evidence dir (digests_e<N>_train/, the dir every grant, hash,
launcher gate and upload check already names; nothing else is granted):

  rollouts/<task file stem>/s<k>.trace.jsonl   ONE rollout's whole conversation, one JSON object per
        line (schema TRACE_SCHEMA): a header line, then every message in order -- the task message,
        every assistant message (content, reasoning text as recorded, tool calls with full
        arguments), every tool result in full, every harness note -- then an end line (turns, turn
        cap, how it ended). NOTHING IS CLIPPED. The solver's system prompt and tool definitions are
        stored ONCE per dir under solver_context/ and referenced by sha256 from line 0. Skill bodies
        the solver loaded stay in full (they are the evolver's own earlier skills). The one edit is
        REDACTION: a string that is a non-train task id (val/test/dropped/unused/excluded, either
        spelling) is replaced by REDACTION wherever it occurs (the solver sometimes speculates about
        sibling task names in its reasoning); the count is in the header
        line and in INDEX.json.
  rollouts/<task file stem>/s<k>.grade.txt     the grader's own output for that rollout, byte for byte
        (the DIFF.txt the scorer wrote: verdict, cells matched/graded, every mismatched cell
        expected/got), taken from the snapshot outcomes.py links to the cache record the leg
        recorded. Expected values are of TRAIN tasks only (the surface is train-only).
  solver_context/system_prompt.<sha12>.txt, tools.<sha12>.json
  INDEX.json + INDEX.md   one row per rollout: task, sample, outcome (outcomes.OUTCOMES), cells
        matched/graded, turns, turn cap hit, termination, skills AVAILABLE (the producing library
        snapshot ./_library/ + its skill names, cross-checked against the skill list the solver's
        system prompt actually showed), skills LOADED (re-derived from the trace), skills NOT
        loaded, trace and grade paths, sizes and hashes.
  _library/   unchanged: the exact skill files the epoch's solver carried (skill_library.snapshot).

The per-task *.md digests, COVERAGE.json and DIGEST_INFO.json stay (see the loop: the
coverage step, the producing-library check, the skill-uptake assessor, the launcher's batch-rate gate and
derive_facts all read them); the new files do not depend on them, and check_dir() cross-checks
that the two describe the same verdicts and, when DIGEST_INFO says the grader output is full, the
same grader text.

SOURCES. Everything is derived from the LEG the surface describes (epoch 1: the base leg; epoch N:
skills_v{N-1}'s train leg): its per-sample verdicts, and, through outcomes.leg_outcome_counts, the
final attempt's shard cache records and the snapshot whose DIFF.txt produced each verdict. The
per-epoch cache the digest builder reads can be passed as a cross-check (every record byte-equal).

SOLVER ADAPTERS. `direct_trace` (a direct harness: an OpenAI-format message list in the cache
record) and `cc_trace` (a Claude Code solver: its stream-json output, optionally with the
session JSONL, whose first user message and attachments the stream does not carry). A cache record
is routed by its cache_key.sdk; a Claude Code leg must store the raw stream text in
trace.stream_jsonl (and may add trace.session_jsonl, trace.user_message, trace.system_prompt).

No package imports (outcomes.py and skill_library.py are loaded by path), Python 3.9 compatible.

CLI:
  python src/orchestrator/rollout_evidence.py build --out DIR --run-dir R --leg LEG.json --split S
        --n-samples N [--library skills_vK] [--epoch J] [--cache C [--bucket B]] [--model M]
  python src/orchestrator/rollout_evidence.py check DIR --split S --n-samples N [--library L]
        [--leg LEG.json --run-dir R]
  python src/orchestrator/rollout_evidence.py cc-trace STREAM.jsonl [--session S.jsonl]
        [--user-message F] [--system-prompt F] [--out TRACE.jsonl]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

SCHEMA = "evoskill.rollout_evidence/1"
TRACE_SCHEMA = "evoskill.rollout_trace/1"
INDEX_JSON = "INDEX.json"
INDEX_MD = "INDEX.md"
ROLLOUTS = "rollouts"
CONTEXT = "solver_context"
LIBRARY_DIR = "_library"
# RUNTIME.json key and its values. Absent = "digests": the per-task digests only, every existing run
# unchanged. "rollouts" = this module's files are built and checked in every evidence dir.
LAYOUT_KEY = "evolver_evidence_layout"
LAYOUTS = ("digests", "rollouts")
REDACTION = "[REDACTED: non-train task id]"
# The direct harness's own marker when a tool result exceeded 100,000 chars (executor._truncate).
HARNESS_TRUNCATION = "[OUTPUT TRUNCATED MIDDLE-OUT — "
OUTCOMES = ("passed", "graded_wrong", "not_eligible", "grader_failed")
NON_TRAIN_KEYS = ("val", "test", "dropped", "unused", "excluded_prior_run")
TRACE_NAME_RE = re.compile(r"^s(\d+)\.trace\.jsonl$")
GRADE_NAME_RE = re.compile(r"^s(\d+)\.grade\.txt$")
CONTEXT_NAME_RE = re.compile(r"^(system_prompt\.[0-9a-f]{12}\.txt|tools\.[0-9a-f]{12}\.json)$")
# A skill body echoed into a tool result (skill tool, or `cat .../SKILL.md`): the frontmatter's name.
FRONTMATTER_RE = re.compile(r"(?m)^---[ \t]*\r?\nname:[ \t]*([^\n]+)")
# Text an INDEX.md must never contain: it is a top-level *.md, and these are what the digest
# consumers (a batch-rate gate, derive_facts, digest_coverage, the skill-uptake readers)
# parse out of task digests.
DIGEST_PATTERNS = (re.compile(r"\*\*Result:\s*\d+/\d+\s+samples passed"), re.compile(r"(?m)^## Rollout "),
                   re.compile(r"(?m)^## Scorer diff"), re.compile(r"\[SKILL LOADED:"),
                   re.compile(r"\[DIGEST (?:CLIP|CAP) "), re.compile(r"(?:turns|kind|termination|budget_exhausted)=\["),
                   re.compile(r"\(tool\.skill\)"))


class EvidenceError(ValueError):
    """The evidence cannot be built, or a dir does not hold what its INDEX says."""


def _sibling(name: str):
    spec = importlib.util.spec_from_file_location(f"evoskill_{name}_for_rollout_evidence", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OC = _sibling("outcomes")
SL = _sibling("skill_library")
ES = _sibling("evidence_surface")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stem_of(task_id: str) -> str:
    """The task FILE stem every builder writes ('Debugging/03_01' -> 'Debugging_03_01')."""
    return task_id.replace("/", "_")


def layout_of(runtime: dict | None) -> str:
    """RUNTIME.json evolver_evidence_layout, checked. Absent = "digests"."""
    v = (runtime or {}).get(LAYOUT_KEY, "digests")
    if v not in LAYOUTS:
        raise EvidenceError(f"RUNTIME.json {LAYOUT_KEY}={v!r} is not one of {LAYOUTS}; absent means "
                            f"'digests', and a typo must not silently mean either")
    return v


# ─────────────────────────────────────────────────────────────── the split, and non-train ids in text
# ONE implementation, in evidence_surface.py (the launcher's and the loop's train-only check use it too).
split_key_ids = ES.split_key_ids
split_sets = ES.split_sets
text_id_hits = ES.text_id_hits


class Redactor:
    """ES.IdScanner applied to whole trace objects, with REDACTION as the marker."""

    def __init__(self, scanner):
        self.sc = scanner

    def text(self, s: str) -> tuple:
        return self.sc.redact(s, REDACTION)

    def obj(self, o):
        """(object with every string leaf redacted, count). Keys are left alone (they are ours)."""
        if isinstance(o, str):
            return self.text(o)
        if isinstance(o, list):
            tot, out = 0, []
            for x in o:
                y, n = self.obj(x)
                out.append(y)
                tot += n
            return out, tot
        if isinstance(o, dict):
            tot, out = 0, {}
            for k, v in o.items():
                y, n = self.obj(v)
                out[k] = y
                tot += n
            return out, tot
        return o, 0


# ─────────────────────────────────────────────────────────────── solver adapters
def _args(raw):
    """(parsed arguments, None) when `raw` is JSON (or already an object), else (None, raw)."""
    if isinstance(raw, (dict, list)):
        return raw, None
    if raw is None:
        return {}, None
    try:
        return json.loads(raw), None
    except (TypeError, ValueError):
        return None, raw


def _rd_text(rd) -> str | None:
    """The concatenated reasoning.text of a reasoning_details list, or None when it holds anything else."""
    if not isinstance(rd, list):
        return None
    out = []
    for x in rd:
        if not isinstance(x, dict) or x.get("type") != "reasoning.text":
            return None
        out.append(x.get("text") or "")
    return "".join(out)


def _direct_termination(tr: dict) -> tuple:
    """(termination, detail, turn_cap_hit) from a direct-harness trace (executor.py budget strings)."""
    be = str(tr.get("budget_exhausted") or "")
    if be.startswith("Turn budget exhausted"):
        return "turn_cap", be, True
    if be.startswith("Time budget exhausted"):
        return "time_budget", be, False
    if be.startswith("Cost budget exhausted"):
        return "cost_budget", be, False
    if be.startswith("Compaction failed"):
        return "context_overflow", be, False
    if be:
        return "budget_other", be, False
    if tr.get("parse_error"):
        return "no_final_answer", str(tr.get("parse_error")), False
    return "completed", "", False


def direct_trace(record: dict) -> dict:
    """A direct-harness cache record -> {solver, model, system, tools, lines, end, stats}.

    `lines` are the messages after the system message, each {"type": "message", "n": <index in the
    recorded list>, "role", ...} with every recorded field kept, except: tool calls are flattened to
    {id, name, arguments} (arguments parsed from their JSON string; `arguments_raw` when unparseable),
    a tool result gets the `name` of the call it answers, a user message the harness wrote
    ([HARNESS] ...) gets from_harness=true, and `reasoning_details` is dropped when its text is
    identical to `reasoning` (100% of 10,128 messages in the live run's e0/e1 train records; the
    count is in stats). The system message is replaced by a reference line (the builder fills it)."""
    tr = record.get("trace") or {}
    msgs = list(tr.get("messages") or [])
    stats: collections.Counter = collections.Counter()
    system = None
    if msgs and msgs[0].get("role") == "system":
        system = msgs[0].get("content")
        if not isinstance(system, str):
            system = json.dumps(system, ensure_ascii=False)
    lines, names = [], {}
    for n, m in enumerate(msgs):
        role = m.get("role")
        if n == 0 and role == "system":
            continue
        out = {"type": "message", "n": n, "role": role}
        if role == "assistant":
            out["content"] = m.get("content")
            if "reasoning" in m:
                out["reasoning"] = m.get("reasoning")
            if "reasoning_details" in m:
                rd = m.get("reasoning_details")
                if _rd_text(rd) is not None and _rd_text(rd) == (m.get("reasoning") or ""):
                    stats["reasoning_details_identical_dropped"] += 1
                else:
                    out["reasoning_details"] = rd
            if "tool_calls" in m:
                calls = []
                for tc in m.get("tool_calls") or []:
                    tc = tc or {}
                    fn = tc.get("function") or {}
                    a, raw = _args(fn.get("arguments"))
                    c = {"id": tc.get("id"), "name": fn.get("name")}
                    if raw is None:
                        c["arguments"] = a
                    else:
                        c["arguments_raw"] = raw
                    for k, v in tc.items():
                        if k not in ("id", "type", "function"):
                            c[k] = v
                    calls.append(c)
                    names[tc.get("id")] = fn.get("name")
                    stats["tool_calls"] += 1
                out["tool_calls"] = calls
            for k, v in m.items():
                if k not in ("role", "content", "reasoning", "reasoning_details", "tool_calls"):
                    out[k] = v
            stats["assistant_messages"] += 1
        elif role == "tool":
            out["tool_call_id"] = m.get("tool_call_id")
            out["name"] = names.get(m.get("tool_call_id"))
            out["content"] = m.get("content")
            for k, v in m.items():
                if k not in ("role", "tool_call_id", "content"):
                    out[k] = v
            stats["tool_results"] += 1
            if HARNESS_TRUNCATION in str(m.get("content") or ""):
                stats["harness_truncated_results"] += 1
        else:
            c = m.get("content")
            if role == "user" and isinstance(c, str) and c.startswith("[HARNESS]"):
                out["from_harness"] = True
                stats["harness_notes"] += 1
            out["content"] = c
            for k, v in m.items():
                if k not in ("role", "content"):
                    out[k] = v
        lines.append(out)
    term, detail, cap = _direct_termination(tr)
    output = tr.get("output") if isinstance(tr.get("output"), dict) else {}
    end = {"type": "end", "turns": tr.get("num_turns"), "turn_cap_hit": cap, "termination": term,
           "termination_detail": detail, "finished": output.get("finished"), "is_error": tr.get("is_error"),
           "parse_error": tr.get("parse_error"), "final_text": tr.get("result"),
           "structured_output": tr.get("raw_structured_output"), "cost_usd": tr.get("total_cost_usd"),
           "duration_ms": tr.get("duration_ms"), "usage": tr.get("usage")}
    return {"solver": "direct", "model": tr.get("model"), "system": system,
            "system_note": "the solver's system message, verbatim",
            "tools": tr.get("tool_schemas"), "lines": lines, "end": end, "stats": dict(stats),
            "dropped": ["trace.cached_transcript (a rendering of these same messages)",
                        "trace.uuid / trace.session_id (identifiers)"]}


def _jsonl_events(text: str) -> tuple:
    out, bad = [], 0
    for line in (text or "").split("\n"):          # "\n" only (see _read_trace): U+2028 inside a string is not a line end
        s = line.strip()
        if not s:
            continue
        try:
            e = json.loads(s)
        except ValueError:
            bad += 1
            continue
        if isinstance(e, dict):
            out.append(e)
        else:
            bad += 1
    return out, bad


def _block_text(content) -> tuple:
    """(text, non-text blocks) of a tool_result content: a string, or a list of blocks."""
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    if isinstance(content, list):
        texts, other = [], []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                texts.append(b.get("text") or "")
            else:
                other.append(b)
        return "\n".join(texts), other
    return json.dumps(content, ensure_ascii=False), []


CC_INIT_DROP = ("uuid", "session_id", "messaging_socket_path")
# Stream / session record types that carry no model context: progress counters, quota telemetry,
# queue bookkeeping. Counted in stats, not written.
CC_SKIP_TYPES = {("system", "thinking_tokens"), ("rate_limit_event", None), ("queue-operation", None),
                 ("atis-latch", None), ("last-prompt", None), ("cost-state", None)}


def cc_trace(stream_text: str, *, session_text: str | None = None, user_message: str | None = None,
             system_prompt: str | None = None) -> dict:
    """A Claude Code solver session -> the same shape as direct_trace.

    stream_text  the `claude -p --output-format stream-json --verbose` output (scripts/cc_solver):
                 init, one event per assistant content block, user events carrying tool results, result.
    session_text optional ~/.claude/projects/<cwd>/<session>.jsonl: when given, the MESSAGES come from it
                 (it also holds the task message and the attachments -- environment system-reminders
                 the model saw -- which the stream does not); init and result still come from the stream.
    user_message the task message, when no session file carries it.
    system_prompt the text passed with --append-system-prompt (Claude Code's own system prompt is not
                 in either file; system_note says so).

    Consecutive assistant events with one message id are ONE model reply (one turn under
    --max-turns): content = its text blocks, reasoning = its thinking blocks' text (empty unless
    thinking display is on: `--thinking-display summarized`; reasoning_redacted counts thinking blocks
    that carried only a signature), tool_calls = its tool_use blocks. Every tool_result block is one
    tool line. Unknown event types are kept verbatim as {"type": "event", "raw": ...}."""
    evs, bad = _jsonl_events(stream_text)
    stats: collections.Counter = collections.Counter()
    stats["unparsed_stream_lines"] = bad
    init = next((e for e in evs if e.get("type") == "system" and e.get("subtype") == "init"), None)
    result = next((e for e in reversed(evs) if e.get("type") == "result"), None)
    src = evs
    if session_text is not None:
        src, sbad = _jsonl_events(session_text)
        stats["unparsed_session_lines"] = sbad
    lines: list = []
    names: dict = {}
    cur = None                                   # the assistant reply being assembled

    def flush():
        nonlocal cur
        if cur is not None:
            if not cur["reasoning_parts"] and not cur["redacted"]:
                cur["line"].pop("reasoning", None)
            else:
                cur["line"]["reasoning"] = "\n\n".join(cur["reasoning_parts"])
            if cur["redacted"]:
                cur["line"]["reasoning_redacted"] = cur["redacted"]
            cur["line"]["content"] = "\n\n".join(cur["texts"]) if cur["texts"] else None
            lines.append(cur["line"])
            cur = None

    seen_user = False
    for e in src:
        t, st = e.get("type"), e.get("subtype")
        if (t, st) in CC_SKIP_TYPES or (t, None) in CC_SKIP_TYPES:
            stats[f"skipped:{t}{'/' + st if st else ''}"] += 1
            continue
        if t == "system" and st == "init":
            continue                              # header metadata, below
        if t == "result":
            continue                              # the end line, below
        if t == "assistant":
            msg = e.get("message") or {}
            mid = msg.get("id")
            if cur is None or cur["id"] != mid or mid is None:
                flush()
                cur = {"id": mid, "texts": [], "reasoning_parts": [], "redacted": 0,
                       "line": {"type": "message", "role": "assistant", "content": None, "reasoning": "",
                                "tool_calls": [], "model": msg.get("model")}}
                stats["assistant_messages"] += 1
            for b in msg.get("content") or []:
                if not isinstance(b, dict):
                    continue
                bt = b.get("type")
                if bt == "text":
                    cur["texts"].append(b.get("text") or "")
                elif bt == "thinking":
                    txt = b.get("thinking") or ""
                    if txt:
                        cur["reasoning_parts"].append(txt)
                    else:
                        cur["redacted"] += 1
                elif bt == "redacted_thinking":
                    cur["redacted"] += 1
                elif bt == "tool_use":
                    cur["line"]["tool_calls"].append({"id": b.get("id"), "name": b.get("name"),
                                                      "arguments": b.get("input")})
                    names[b.get("id")] = b.get("name")
                    stats["tool_calls"] += 1
                else:
                    cur["line"].setdefault("other_blocks", []).append(b)
            if msg.get("stop_reason"):
                cur["line"]["stop_reason"] = msg.get("stop_reason")
            continue
        flush()
        if t == "user":
            msg = e.get("message") or {}
            c = msg.get("content")
            if isinstance(c, list) and any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c):
                for b in c:
                    if not (isinstance(b, dict) and b.get("type") == "tool_result"):
                        lines.append({"type": "message", "role": "user", "content": b})
                        continue
                    text, other = _block_text(b.get("content"))
                    ln = {"type": "message", "role": "tool", "tool_call_id": b.get("tool_use_id"),
                          "name": names.get(b.get("tool_use_id")), "content": text,
                          "is_error": bool(b.get("is_error"))}
                    if other:
                        ln["content_blocks"] = other
                    lines.append(ln)
                    stats["tool_results"] += 1
            else:
                if isinstance(c, list):
                    c, other = _block_text(c)
                else:
                    other = []
                ln = {"type": "message", "role": "user", "content": c}
                if other:
                    ln["content_blocks"] = other
                if e.get("isMeta"):
                    ln["from_harness"] = True
                lines.append(ln)
                seen_user = True
            continue
        if t == "attachment":
            rendered = e.get("rendered") or []
            txt = "\n".join(str((r or {}).get("content") or "") for r in rendered if isinstance(r, dict))
            if txt.strip():
                lines.append({"type": "message", "role": "user", "from_harness": True, "kind": "attachment",
                              "content": txt})
                stats["attachments"] += 1
            else:
                stats["attachments_not_rendered"] += 1
            continue
        lines.append({"type": "event", "raw": e})
        stats[f"kept_event:{t}"] += 1
    flush()
    if not seen_user and user_message is not None:
        lines.insert(0, {"type": "message", "role": "user", "content": user_message})
    for i, ln in enumerate(lines, start=1):
        ln["n"] = i
        lines[i - 1] = {"type": ln.pop("type"), "n": ln.pop("n"), **ln}
    res = result or {}
    sub = res.get("subtype")
    term = {"success": "completed", "error_max_turns": "turn_cap"}.get(sub, sub or "no_result_event")
    end = {"type": "end", "turns": res.get("num_turns"), "turn_cap_hit": sub == "error_max_turns",
           "termination": term, "termination_detail": res.get("terminal_reason") or res.get("stop_reason") or "",
           "model_replies": stats.get("assistant_messages", 0), "is_error": res.get("is_error"),
           "final_text": res.get("result"), "cost_usd": res.get("total_cost_usd"),
           "duration_ms": res.get("duration_ms"), "usage": res.get("usage")}
    init_meta = {k: v for k, v in (init or {}).items() if k not in CC_INIT_DROP}
    return {"solver": "claude_code", "model": (init or {}).get("model"), "system": system_prompt,
            "system_note": ("the text appended with --append-system-prompt; Claude Code's own built-in system "
                            "prompt is not recorded in the stream or the session file"),
            "tools": (init or {}).get("tools"), "lines": lines, "end": end, "stats": dict(stats),
            "init": init_meta, "dropped": ["stream thinking_tokens progress events, rate_limit_event quota "
                                           "telemetry, session queue/latch/cost bookkeeping (counted in stats)"]}


def trace_for_record(record: dict) -> dict:
    """Route a cache record to its adapter by cache_key.sdk."""
    sdk = str((record.get("cache_key") or {}).get("sdk") or "")
    tr = record.get("trace") or {}
    if sdk in ("claude_code", "cc", "claude-code"):
        if not isinstance(tr.get("stream_jsonl"), str):
            raise EvidenceError("a Claude Code record must carry its raw stream in trace.stream_jsonl")
        return cc_trace(tr["stream_jsonl"], session_text=tr.get("session_jsonl"),
                        user_message=tr.get("user_message"), system_prompt=tr.get("system_prompt"))
    if sdk and sdk != "direct":
        raise EvidenceError(f"no trace adapter for sdk {sdk!r}")
    return direct_trace(record)


# ─────────────────────────────────────────────────────────────── skills: available (told) and loaded
_SKILL_ITEM = re.compile(r"^- ([A-Za-z0-9][A-Za-z0-9._-]*): ")


def told_skills(system_text: str | None):
    """The skill names the direct harness's system prompt listed under '## Skills' ([] for its
    '(No skills available yet.)'); None when the prompt has no such section (e.g. Claude Code)."""
    if not system_text or "\n## Skills" not in system_text:
        return None
    sec = system_text.split("\n## Skills", 1)[1]
    out = []
    for ln in sec.splitlines()[1:]:
        if ln.startswith("## ") or ln.startswith("IMPORTANT:"):
            break
        m = _SKILL_ITEM.match(ln)
        if m:
            out.append(m.group(1))
    return sorted(out)


def _content_text(ln: dict) -> str:
    c = ln.get("content")
    return c if isinstance(c, str) else ("" if c is None else json.dumps(c, ensure_ascii=False))


def skill_loads(lines, available) -> dict:
    """Which skills the solver LOADED, from the trace alone: a skill-tool call whose result is not an
    error (direct: `skill` {"name"}; Claude Code: `Skill` {"skill"|"name"|"command"}), or a library
    skill's frontmatter echoed into any tool result (`cat .../SKILL.md`). Frontmatter names that are
    not library skills are reported (`foreign_frontmatter`), never counted as loads."""
    avail = set(available or [])
    want = {}
    calls = []
    for ln in lines:
        if ln.get("type") != "message":
            continue
        if ln.get("role") == "assistant":
            for c in ln.get("tool_calls") or []:
                if str(c.get("name") or "").lower() == "skill":
                    a = c.get("arguments") if isinstance(c.get("arguments"), dict) else {}
                    nm = a.get("name") or a.get("skill") or a.get("command")
                    want[c.get("id")] = str(nm).strip() if nm is not None else ""
        elif ln.get("role") == "tool" and ln.get("tool_call_id") in want:
            txt = _content_text(ln)
            ok = not ln.get("is_error") and not txt.lstrip().startswith("Error")
            calls.append((want[ln["tool_call_id"]], ok))
    echoed, foreign = set(), set()
    for ln in lines:
        if ln.get("type") == "message" and ln.get("role") == "tool":
            for m in FRONTMATTER_RE.finditer(_content_text(ln)):
                nm = m.group(1).strip().strip("'\"")
                (echoed if nm in avail else foreign).add(nm)
    loaded = sorted(({n for n, ok in calls if ok and n in avail}) | echoed)
    return {"loaded": loaded, "not_loaded": sorted(avail - set(loaded)), "skill_tool_calls": len(calls),
            "skill_tool_errors": sorted({n for n, ok in calls if not ok}),
            "skill_tool_ok_outside_library": sorted({n for n, ok in calls if ok and n not in avail}),
            "foreign_frontmatter": sorted(foreign)}


def library_info(library) -> dict | None:
    """{label, sha256_12, skills: [{name, path, bytes, description}]} of a promoted skill set."""
    if library is None:
        return None
    lib = Path(library)
    label = SL.label_of(lib)
    rows = []
    for f in sorted(lib.glob("*/SKILL.md")):
        text = f.read_text(errors="replace")
        m = re.search(r"(?m)^description:[ \t]*(.+)$", text.split("\n---", 1)[0] if text.startswith("---") else "")
        rows.append({"name": f.parent.name, "path": f"{LIBRARY_DIR}/{f.parent.name}/SKILL.md",
                     "bytes": len(f.read_bytes()),
                     "description": (m.group(1).strip().strip("'\"") if m else None),
                     "files": sorted(f"{LIBRARY_DIR}/{p.relative_to(lib).as_posix()}"
                                     for p in f.parent.rglob("*") if p.is_file() and not p.is_symlink())})
    return {"label": label, "sha256_12": SL.library_sha256(lib)[:12], "path": LIBRARY_DIR, "skills": rows}


# ─────────────────────────────────────────────────────────────── building one epoch's evidence
def _dump_line(o) -> str:
    return json.dumps(o, ensure_ascii=False, sort_keys=False)


def _leg_sources(run_dir, leg_path, leg: dict, n: int, train: set) -> tuple:
    """({(task, k): outcomes sample}, {(task, k): record path}, outcome result) for the leg."""
    r = OC.leg_outcome_counts(run_dir, leg)
    if r.get("mismatches") or r.get("problems"):
        raise EvidenceError(f"{Path(leg_path).name}: outcomes.py finds the leg's records inconsistent "
                            f"({(r.get('mismatches') or r.get('problems'))[:3]}); the evidence would pair a "
                            f"transcript with a verdict it did not produce")
    samples = {(s["task"], s["sample"]): s for s in r["samples"]}
    attempts = OC.discover_attempts(Path(run_dir), leg)
    finals = [a.dir / "cache" for a in attempts if a.rank == 0]
    idx = OC.index_cache_records(finals, n)
    recs = {}
    for t in leg["task_ids"]:
        for k in range(n):
            got = idx.get((t, k)) or []
            if len(got) != 1:
                raise EvidenceError(f"{t} s{k}: {len(got)} cache record(s) in the leg's final attempt "
                                    f"({[a.label for a in attempts if a.rank == 0][:3]}...); need exactly one")
            recs[(t, k)] = Path(got[0]["path"])
    return samples, recs, r


def build(out_dir, *, run_dir, leg_json, split_path, n_samples: int, library=None, epoch=None,
          cross_check_cache=None, bucket: str = "", expect_model: str | None = None) -> dict:
    """Build rollouts/, solver_context/, INDEX.json and INDEX.md into `out_dir` (and _library/ when a
    library is given and absent) from the leg `leg_json`. Refuses to touch a dir that already holds any
    of them. Built in a sibling temp dir and moved in; INDEX.* are written last; check_dir() must pass
    or everything this call added is removed again. Returns a summary."""
    out_dir = Path(out_dir)
    leg_path = Path(leg_json)
    for name in (INDEX_JSON, INDEX_MD, ROLLOUTS, CONTEXT):
        if (out_dir / name).exists() or (out_dir / name).is_symlink():
            raise EvidenceError(f"{out_dir / name} already exists; refusing to build over evidence an evolver "
                                f"may have read. Move the dir aside and rebuild.")
    if out_dir.is_symlink():
        raise EvidenceError(f"{out_dir} is a symlink")
    leg = json.loads(leg_path.read_text())
    n = int(n_samples)
    if int(leg.get("n_samples") or n) != n:
        raise EvidenceError(f"{leg_path.name} ran n_samples={leg.get('n_samples')}, the spec says {n}")
    train, non = split_sets(split_path)
    red = Redactor(ES.IdScanner(train, non))
    tasks = list(leg.get("task_ids") or [])
    if not tasks:
        raise EvidenceError(f"{leg_path.name} lists no task_ids")
    stray = sorted(set(tasks) - train)
    if stray:
        raise EvidenceError(f"{leg_path.name} ran {len(stray)} task(s) that are NOT train tasks of the split "
                            f"(e.g. a val leg passed as the batch); the evidence is train-only")
    if set(tasks) != train:
        raise EvidenceError(f"{leg_path.name} ran {len(tasks)} tasks, the split's train key has {len(train)}")
    lib_info = library_info(library) if library is not None else None
    available = [s["name"] for s in lib_info["skills"]] if lib_info else []
    if library is None and ((out_dir / LIBRARY_DIR).exists() or (out_dir / LIBRARY_DIR).is_symlink()):
        raise EvidenceError(f"{out_dir.name} holds a {LIBRARY_DIR}/ but no producing library was given "
                            f"(a base surface carries none)")
    samples, recs, oc_result = _leg_sources(run_dir, leg_path, leg, n, train)
    turn_cap = leg.get("budget")
    ctx_files: dict = {}                         # name -> bytes
    ctx_meta: dict = {}                          # name -> {kind, sha256, bytes, rollouts, note}
    rows = []
    created = []
    solvers = set()
    partial = out_dir.parent / f".{out_dir.name}.evidence.partial.{os.getpid()}"
    if partial.exists():
        shutil.rmtree(partial)
    lib_snapshotted = False
    try:
        (partial / ROLLOUTS).mkdir(parents=True)
        (partial / CONTEXT).mkdir()
        for t in sorted(tasks):
            for k in range(n):
                rp = recs[(t, k)]
                raw = rp.read_bytes()
                rec = json.loads(raw)
                q = json.loads((rec.get("cache_key") or {}).get("question") or "{}")
                if q.get("task_id") != t:
                    raise EvidenceError(f"{rp.name}: record is for {q.get('task_id')!r}, not {t}")
                if cross_check_cache:
                    tree = rp.parent.name
                    cand = Path(cross_check_cache) / (bucket or tree) / rp.name
                    if not cand.is_file() or sha256_file(cand) != sha256_bytes(raw):
                        raise EvidenceError(f"{t} s{k}: the leg's record {rp.parent.name}/{rp.name} is not "
                                            f"byte-identical in the digest cache {cand} (the digests and this "
                                            f"evidence would describe different rollouts)")
                if expect_model and (rec.get("cache_key") or {}).get("model") not in (None, expect_model):
                    raise EvidenceError(f"{rp.name}: model {(rec.get('cache_key') or {}).get('model')!r} "
                                        f"!= the spec's {expect_model!r}")
                s = samples[(t, k)]
                score = s.get("recorded")
                rec_score = (rec.get("trace") or {}).get("inline_score")
                if score is not None and rec_score is not None and \
                        (rec_score >= OC.PASS_THRESHOLD) != (score >= OC.PASS_THRESHOLD):
                    raise EvidenceError(f"{t} s{k}: record inline_score {rec_score} disagrees with the leg's {score}")
                tr = trace_for_record(rec)
                solvers.add(tr["solver"])
                header_extra = {}
                sys_text = tr.get("system")
                sys_ref = None
                if sys_text is not None:
                    sys_red, nsys = red.text(sys_text)
                    b = sys_red.encode("utf-8")
                    name = f"system_prompt.{sha256_bytes(b)[:12]}.txt"
                    ctx_files.setdefault(name, b)
                    meta = ctx_meta.setdefault(name, {"kind": "system_prompt", "sha256": sha256_bytes(b),
                                                      "bytes": len(b), "rollouts": 0, "redactions": nsys,
                                                      "note": tr.get("system_note")})
                    meta["rollouts"] += 1
                    sys_ref = f"{CONTEXT}/{name}"
                tools_ref = None
                if tr.get("tools") is not None:
                    tb = (json.dumps(tr["tools"], ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8")
                    name = f"tools.{sha256_bytes(tb)[:12]}.json"
                    ctx_files.setdefault(name, tb)
                    ctx_meta.setdefault(name, {"kind": "tools", "sha256": sha256_bytes(tb), "bytes": len(tb),
                                               "rollouts": 0, "redactions": 0})["rollouts"] += 1
                    tools_ref = f"{CONTEXT}/{name}"
                body, nred = red.obj(tr["lines"])
                end, nred_end = red.obj(tr["end"])
                nred += nred_end
                if tr.get("init"):
                    header_extra["solver_init"], ni = red.obj(tr["init"])
                    nred += ni
                told = told_skills(sys_text)
                sk = skill_loads(body, available)
                if lib_info is None and (sk["loaded"] or sk["foreign_frontmatter"] or
                                         sk["skill_tool_ok_outside_library"]):
                    raise EvidenceError(f"BASE LEG CONTAMINATED: {t} s{k} loaded skill text "
                                        f"({sk['foreign_frontmatter'] or sk['skill_tool_ok_outside_library']}) but a "
                                        f"base solver carries no skills")
                if told is not None and told != sorted(available):
                    raise EvidenceError(f"{t} s{k}: the solver's system prompt listed skills {told}, the "
                                        f"producing library holds {sorted(available)}")
                stem = stem_of(t)
                header = {"type": "header", "schema": TRACE_SCHEMA, "task": t, "sample": k,
                          "solver": tr["solver"], "model": tr.get("model"),
                          "system_prompt": sys_ref, "tools": tools_ref,
                          "record_sha256": sha256_bytes(raw), "created_at": rec.get("created_at"),
                          "messages": len(body), "redactions": nred, "redaction_marker": REDACTION,
                          "stats": tr.get("stats"), "dropped": tr.get("dropped"), **header_extra}
                sys_line = ([{"type": "message", "n": 0, "role": "system", "ref": sys_ref,
                              "sha256": ctx_meta[sys_ref.split("/", 1)[1]]["sha256"]}] if sys_ref else [])
                text = "".join(_dump_line(x) + "\n" for x in [header] + sys_line + body + [end])
                tb = text.encode("utf-8")
                d = partial / ROLLOUTS / stem
                d.mkdir(exist_ok=True)
                (d / f"s{k}.trace.jsonl").write_bytes(tb)
                grade_rel, gsha, gbytes, gred = None, None, 0, 0
                snap = s.get("snapshot")
                diff_text = None
                if snap and (Path(snap) / "DIFF.txt").is_file():
                    gb = (Path(snap) / "DIFF.txt").read_bytes()
                    diff_text = gb.decode("utf-8", errors="replace")
                    gtxt, gred = red.text(diff_text)
                    if gred:
                        gb = gtxt.encode("utf-8")
                    (d / f"s{k}.grade.txt").write_bytes(gb)
                    grade_rel = f"{ROLLOUTS}/{stem}/s{k}.grade.txt"
                    gsha, gbytes = sha256_bytes(gb), len(gb)
                elif s.get("outcome") != "grader_failed":
                    raise EvidenceError(f"{t} s{k}: outcome {s.get('outcome')} but no DIFF.txt at {snap}")
                pd = OC.parse_diff(diff_text)
                rows.append({
                    "task": t, "task_file": stem, "category": t.split("/", 1)[0], "sample": k,
                    "outcome": s["outcome"], "outcome_reason": s.get("reason"), "score": score,
                    "cells_matched": pd["n_match"] if pd else None, "cells_graded": pd["n_total"] if pd else None,
                    "grader_warnings": list(s.get("warnings") or []),
                    "turns": end.get("turns"), "turn_cap": turn_cap, "turn_cap_hit": bool(end.get("turn_cap_hit")),
                    "termination": end.get("termination"), "termination_detail": end.get("termination_detail"),
                    "skills_available": {"library": lib_info["label"] if lib_info else None,
                                         "path": LIBRARY_DIR if lib_info else None, "names": sorted(available)},
                    "skills_loaded": sk["loaded"], "skills_not_loaded": sk["not_loaded"],
                    "skill_tool_calls": sk["skill_tool_calls"], "skill_tool_errors": sk["skill_tool_errors"],
                    "messages": len(body), "tool_calls": (tr.get("stats") or {}).get("tool_calls", 0),
                    "harness_truncated_results": (tr.get("stats") or {}).get("harness_truncated_results", 0),
                    "redactions": nred + gred,
                    "trace": f"{ROLLOUTS}/{stem}/s{k}.trace.jsonl", "trace_bytes": len(tb), "trace_sha256": sha256_bytes(tb),
                    "grade": grade_rel, "grade_bytes": gbytes, "grade_sha256": gsha,
                    "system_prompt": sys_ref, "cost_usd": end.get("cost_usd"), "duration_ms": end.get("duration_ms"),
                })
        for name, b in ctx_files.items():
            (partial / CONTEXT / name).write_bytes(b)
        if library is not None:
            if not (out_dir / LIBRARY_DIR).exists():
                lib_snapshotted = True
            out_dir.mkdir(parents=True, exist_ok=True)
            SL.snapshot(library, out_dir)       # idempotent: an existing copy must be byte-identical
        if len(solvers) != 1:
            raise EvidenceError(f"one batch, one solver: the records name {sorted(solvers)}")
        index = _index(out_dir, rows, ctx_meta, lib_info, leg_path, leg, oc_result, n, epoch, turn_cap,
                       tr_solver=next(iter(solvers)))
        md = render_md(index).encode("utf-8")
        for pat in DIGEST_PATTERNS:
            if pat.search(md.decode("utf-8")):
                raise EvidenceError(f"INDEX.md would contain {pat.pattern!r}, a digest marker another reader parses")
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in (ROLLOUTS, CONTEXT):
            os.replace(str(partial / name), str(out_dir / name))
            created.append(out_dir / name)
        for name, data in ((INDEX_JSON, (json.dumps(index, indent=1, ensure_ascii=False) + "\n").encode("utf-8")),
                           (INDEX_MD, md)):
            tmp = out_dir / f".{name}.{os.getpid()}.tmp"
            tmp.write_bytes(data)
            os.replace(str(tmp), str(out_dir / name))
            created.append(out_dir / name)
        probs = check_dir(out_dir, split_path=split_path, n_samples=n, library=library,
                          leg_json=leg_path, run_dir=run_dir)
        if probs:
            raise EvidenceError("the evidence just built does not check: " + "; ".join(probs[:6]))
    except BaseException:
        for p in created:
            if p.is_dir() and not p.is_symlink():
                shutil.rmtree(p, ignore_errors=True)
            else:
                try:
                    p.unlink()
                except OSError:
                    pass
        if lib_snapshotted:
            shutil.rmtree(out_dir / LIBRARY_DIR, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(partial, ignore_errors=True)
    return summarize(out_dir)


def _index(out_dir: Path, rows, ctx_meta, lib_info, leg_path: Path, leg: dict, oc_result: dict, n: int,
           epoch, turn_cap, tr_solver) -> dict:
    c = collections.Counter(r["outcome"] for r in rows)
    sizes = {"traces": {"files": len(rows), "bytes": sum(r["trace_bytes"] for r in rows)},
             "grades": {"files": sum(1 for r in rows if r["grade"]), "bytes": sum(r["grade_bytes"] for r in rows)},
             "solver_context": {"files": len(ctx_meta), "bytes": sum(m["bytes"] for m in ctx_meta.values())}}
    uptake = sum(1 for r in rows if r["skills_loaded"])
    return {
        "schema": SCHEMA, "evidence_dir": out_dir.name, "epoch": epoch,
        "leg": {"file": leg_path.name, "sha256": sha256_file(leg_path), "tag": leg.get("tag"),
                "outcome_status": oc_result.get("status")},
        "solver": {"kind": tr_solver, "model": leg.get("model"), "turn_cap": turn_cap, "n_samples": n},
        "producing_library": lib_info,
        "solver_context": [{"path": f"{CONTEXT}/{k}", **v} for k, v in sorted(ctx_meta.items())],
        "counts": {"rollouts": len(rows), "tasks": len({r["task"] for r in rows}),
                   **{o: c.get(o, 0) for o in OUTCOMES},
                   "turn_cap_hit": sum(1 for r in rows if r["turn_cap_hit"]),
                   "rollouts_that_loaded_a_skill": uptake,
                   "redactions": sum(r["redactions"] for r in rows),
                   "harness_truncated_results": sum(r["harness_truncated_results"] for r in rows)},
        "sizes": sizes,
        "fidelity": {"clipped": 0, "system_prompt": "stored once per distinct text under solver_context/",
                     "tool_results": "exactly what the solver received (the harness's own middle-out cut of a "
                                     "result over its limit is inside the text, with its own marker)",
                     "redaction": REDACTION, "reasoning": "as recorded by the solver's API"},
        "builder": {"module": "src/orchestrator/rollout_evidence.py", "sha256": sha256_file(Path(__file__))},
        "rows": sorted(rows, key=lambda r: (r["task"], r["sample"])),
    }


def _kb(b: int) -> str:
    return f"{b / 1024:,.0f} KB" if b >= 1024 else f"{b} B"


def render_md(index: dict) -> str:
    """INDEX.md: the same facts as INDEX.json, for reading. Never carries a digest marker (DIGEST_PATTERNS)."""
    c, lib, sol = index["counts"], index.get("producing_library"), index["solver"]
    L = [f"# Rollout evidence: {index['evidence_dir']}" + (f" (epoch {index['epoch']})" if index.get("epoch") else ""),
         "",
         f"{c['rollouts']} rollouts = {c['tasks']} train tasks x {sol['n_samples']} samples, solver "
         f"`{sol.get('model')}` ({sol.get('kind')}), turn cap {sol.get('turn_cap')}.",
         f"Outcomes: {c['passed']} passed, {c['graded_wrong']} graded and wrong, {c['not_eligible']} not eligible "
         f"to grade (no output saved), {c['grader_failed']} grader failed. Turn cap hit in {c['turn_cap_hit']}.",
         ""]
    if lib:
        L += [f"Skills AVAILABLE to every rollout below: the producing library `{lib['label']}` "
              f"(sha256:{lib['sha256_12']}), exact files in `./{lib['path']}/`:"]
        for s in lib["skills"]:
            L.append(f"- `{s['name']}`: `./{s['path']}` ({s['bytes']} B)")
        L += [f"{c['rollouts_that_loaded_a_skill']} of {c['rollouts']} rollouts loaded at least one skill.", ""]
    else:
        L += ["Skills AVAILABLE: none (the solver of this batch carried no skills).", ""]
    for m in index.get("solver_context") or []:
        L.append(f"- {'System prompt' if m['kind'] == 'system_prompt' else 'Tool definitions'} "
                 f"({m['rollouts']} rollouts): `./{m['path']}`")
    sz = index["sizes"]
    L += ["",
          "## Files",
          "",
          f"- `rollouts/<task>/s<k>.trace.jsonl`: the rollout's whole conversation, one JSON object per line: "
          f"a `header` line, then every message in order (`n` = its position; the system message is a reference "
          f"to the file above), then an `end` line with turns, turn cap and how it ended. Nothing is clipped: "
          f"assistant `content`, `reasoning` and `tool_calls` (full `arguments`), every tool result in full. "
          f"The harness itself shortens a tool result over its limit, marked inside that result "
          f"({c['harness_truncated_results']} such results here). Read one with e.g. "
          f"`python3 -c \"import json,sys; [print(json.dumps(json.loads(l), indent=1, ensure_ascii=False)) "
          f"for l in open(sys.argv[1])]\" <file>`; long lines do not fit a line-limited viewer.",
          "- `rollouts/<task>/s<k>.grade.txt`: the grader's own output for that rollout, verbatim.",
          "- `INDEX.json`: this table, machine-readable.",
          f"- Sizes: traces {_kb(sz['traces']['bytes'])} ({sz['traces']['files']} files), grades "
          f"{_kb(sz['grades']['bytes'])}, solver context {_kb(sz['solver_context']['bytes'])}."]
    if c.get("redactions"):
        L.append(f"- {c['redactions']} occurrence(s) of a non-train task id were replaced by `{REDACTION}`.")
    L += ["", "## Per rollout", "",
          "| task | sample | outcome | cells matched/graded | turns | cap hit | ended | skills loaded | "
          "skills not loaded | trace | grade |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in index["rows"]:
        cells = (f"{r['cells_matched']}/{r['cells_graded']}" if r.get("cells_graded") is not None else "-")
        L.append(f"| {r['task']} | s{r['sample']} | {r['outcome']} | {cells} | {r['turns']} | "
                 f"{'yes' if r['turn_cap_hit'] else 'no'} | {r['termination']} | "
                 f"{', '.join(r['skills_loaded']) or '-'} | {', '.join(r['skills_not_loaded']) or '-'} | "
                 f"`{r['trace']}` ({_kb(r['trace_bytes'])}) | "
                 + (f"`{r['grade']}`" if r.get("grade") else "none recorded") + " |")
    return "\n".join(L) + "\n"


# ─────────────────────────────────────────────────────────────── checking a dir (re-derived from bytes)
def _read_trace(p: Path) -> list:
    out = []
    # split on "\n" ONLY: str.splitlines() also splits on U+2028 / U+2029 / U+0085, which json.dumps(ensure_ascii=False)
    # writes raw inside strings -- a spreadsheet text holding one would make a valid trace unreadable
    for i, line in enumerate(ln for ln in p.read_bytes().split(b"\n") if ln.strip()):
        o = json.loads(line)
        if not isinstance(o, dict) or "type" not in o:
            raise EvidenceError(f"{p.name} line {i}: not a typed JSON object")
        out.append(o)
    return out


def check_dir(d, *, split_path, n_samples: int, library=None, leg_json=None, run_dir=None,
              full: bool = True) -> list:
    """Why dir `d` does NOT hold the rollout evidence its INDEX.json describes. [] means it does.

    Re-derived from the bytes: the INDEX schema; one row per train task x sample and nothing else;
    every trace/grade/context file present, a regular file, at the path, size and sha256 the INDEX
    records, and no file under rollouts/ or solver_context/ that the INDEX does not list; the library
    (`library` = skills_v<N>, or None for a base surface) byte-identical under _library/ and equal to
    every row's skills_available; and, with full=True, every row re-derived from its own files: the
    outcome and cells from the grade (outcomes.classify_detail / parse_diff), turns / cap / termination
    and skills loaded from the trace, the header's task / sample / system-prompt sha. No non-train task
    id may occur in ANY file of the dir. With leg_json + run_dir, every row's outcome and recorded score
    agree with outcomes.leg_outcome_counts and the leg's turn_exhausted count. Where a per-task digest
    <stem>.md exists, its verdict per sample (and, when its DIGEST_INFO.json says the grader output is
    full, its scorer-diff text) agrees with the row."""
    d = Path(d)
    probs: list = []
    if d.is_symlink() or not d.is_dir():
        return [f"no evidence dir at {d} (or it is a symlink)"]
    ip = d / INDEX_JSON
    if not ip.is_file() or ip.is_symlink():
        return [f"{d.name} holds no {INDEX_JSON}: it was not built in the rollouts layout"]
    try:
        index = json.loads(ip.read_text())
    except ValueError as e:
        return [f"{d.name}/{INDEX_JSON} is not JSON: {e}"]
    if not isinstance(index, dict) or index.get("schema") != SCHEMA:
        return [f"{d.name}/{INDEX_JSON}: schema {index.get('schema') if isinstance(index, dict) else None!r}, not {SCHEMA}"]
    if index.get("evidence_dir") != d.name:
        probs.append(f"{INDEX_JSON} names {index.get('evidence_dir')!r}, the dir is {d.name}")
    if not (d / INDEX_MD).is_file():
        probs.append(f"no {INDEX_MD}")
    else:
        mdt = (d / INDEX_MD).read_text(errors="replace")
        if mdt != render_md(index):
            probs.append(f"{INDEX_MD} does not re-render from {INDEX_JSON}")
    train, non = split_sets(split_path)
    scanner = ES.IdScanner(train, non)
    n = int(n_samples)
    rows = index.get("rows") or []
    want = {(t, k) for t in train for k in range(n)}
    have = [(r.get("task"), r.get("sample")) for r in rows]
    dup = sorted({x for x in have if have.count(x) > 1})
    if dup:
        probs.append(f"{len(dup)} (task, sample) listed twice, e.g. {dup[:3]}")
    if set(have) != want:
        extra, gone = sorted(set(have) - want, key=str), sorted(want - set(have))
        probs.append(f"rows != train x {n}: {len(gone)} missing (e.g. {gone[:3]}), {len(extra)} not train "
                     f"(e.g. {[x[0] for x in extra[:3]]})")
    # library
    lib_info = None
    if library is not None:
        try:
            lib_info = library_info(library)
        except ValueError as e:
            probs.append(f"producing library: {e}")
        lc = d / LIBRARY_DIR
        if lc.is_symlink() or not lc.is_dir():
            probs.append(f"no {LIBRARY_DIR}/ although the batch was produced by {Path(library).name}")
        elif SL.tree(lc) != SL.tree(library) or SL.symlinks(lc):
            probs.append(f"{LIBRARY_DIR}/ is not byte-identical to {Path(library).name}")
        if lib_info and index.get("producing_library") != lib_info:
            probs.append(f"{INDEX_JSON} producing_library != {Path(library).name} as it is on disk")
    else:
        if (d / LIBRARY_DIR).exists():
            probs.append(f"a base surface holds {LIBRARY_DIR}/")
        if index.get("producing_library") is not None:
            probs.append(f"a base surface's {INDEX_JSON} names a producing library")
    available = sorted(s["name"] for s in lib_info["skills"]) if lib_info else []
    # context files
    ctx = {}
    for m in index.get("solver_context") or []:
        p = d / str(m.get("path"))
        if p.is_symlink() or not p.is_file():
            probs.append(f"{m.get('path')}: missing")
            continue
        b = p.read_bytes()
        if sha256_bytes(b) != m.get("sha256") or len(b) != m.get("bytes"):
            probs.append(f"{m.get('path')}: bytes differ from {INDEX_JSON}")
        ctx[m["path"]] = (m, b)
    # files on disk vs listed
    listed = {r.get("trace") for r in rows} | {r.get("grade") for r in rows if r.get("grade")} | set(ctx)
    on_disk = set()
    for sub in (ROLLOUTS, CONTEXT):
        base = d / sub
        if base.is_symlink():
            probs.append(f"{sub}/ is a symlink")
            continue
        if not base.is_dir():
            probs.append(f"no {sub}/")
            continue
        for root, dirs, files in os.walk(str(base), followlinks=False):
            for nm in dirs + files:
                p = Path(root) / nm
                if p.is_symlink():
                    probs.append(f"symlink {p.relative_to(d)}")
            for nm in files:
                on_disk.add((Path(root) / nm).relative_to(d).as_posix())
    stray = sorted(on_disk - listed)
    if stray:
        probs.append(f"{len(stray)} file(s) under {ROLLOUTS}/ or {CONTEXT}/ not in {INDEX_JSON}: {stray[:3]}")
    for sd in sorted((d / ROLLOUTS).iterdir()) if (d / ROLLOUTS).is_dir() else []:
        if sd.name not in {stem_of(t) for t in train}:
            probs.append(f"{ROLLOUTS}/{sd.name} is not a train task")
    counts = collections.Counter()
    turn_cap_hits = 0
    for r in rows:
        t, k = r.get("task"), r.get("sample")
        tag = f"{t} s{k}"
        stem = stem_of(str(t))
        if r.get("trace") != f"{ROLLOUTS}/{stem}/s{k}.trace.jsonl":
            probs.append(f"{tag}: trace path {r.get('trace')!r}")
            continue
        tp = d / r["trace"]
        if tp.is_symlink() or not tp.is_file():
            probs.append(f"{tag}: trace missing")
            continue
        tb = tp.read_bytes()
        if sha256_bytes(tb) != r.get("trace_sha256") or len(tb) != r.get("trace_bytes"):
            probs.append(f"{tag}: trace bytes differ from {INDEX_JSON}")
        gtext = None
        if r.get("grade") is not None:
            if r["grade"] != f"{ROLLOUTS}/{stem}/s{k}.grade.txt":
                probs.append(f"{tag}: grade path {r.get('grade')!r}")
            gp = d / r["grade"]
            if gp.is_symlink() or not gp.is_file():
                probs.append(f"{tag}: grade missing")
            else:
                gb = gp.read_bytes()
                if sha256_bytes(gb) != r.get("grade_sha256") or len(gb) != r.get("grade_bytes"):
                    probs.append(f"{tag}: grade bytes differ from {INDEX_JSON}")
                gtext = gb.decode("utf-8", errors="replace")
        elif r.get("outcome") != "grader_failed":
            probs.append(f"{tag}: no grade file but outcome {r.get('outcome')}")
        if r.get("outcome") not in OUTCOMES:
            probs.append(f"{tag}: outcome {r.get('outcome')!r}")
        counts[r.get("outcome")] += 1
        turn_cap_hits += bool(r.get("turn_cap_hit"))
        sa = r.get("skills_available") or {}
        if sorted(sa.get("names") or []) != available or (sa.get("library") != (lib_info or {}).get("label")):
            probs.append(f"{tag}: skills_available {sa} != the library's")
        if not full:
            continue
        oc_, reason = OC.classify_detail(gtext, r.get("grader_warnings") or [])
        if (oc_, reason) != (r.get("outcome"), r.get("outcome_reason")):
            probs.append(f"{tag}: the grade re-derives as {oc_}/{reason}, {INDEX_JSON} says "
                         f"{r.get('outcome')}/{r.get('outcome_reason')}")
        pd = OC.parse_diff(gtext)
        if (pd["n_match"] if pd else None, pd["n_total"] if pd else None) != (r.get("cells_matched"), r.get("cells_graded")):
            probs.append(f"{tag}: cells re-derive as {pd and (pd['n_match'], pd['n_total'])}")
        try:
            lines = _read_trace(tp)
        except (ValueError, EvidenceError, UnicodeDecodeError) as e:
            probs.append(f"{tag}: trace unreadable ({e})")
            continue
        h, e = lines[0], lines[-1]
        body = [x for x in lines[1:-1] if not (x.get("role") == "system" and "ref" in x)]
        if h.get("type") != "header" or h.get("schema") != TRACE_SCHEMA or h.get("task") != t or h.get("sample") != k:
            probs.append(f"{tag}: trace header does not name this rollout")
        if e.get("type") != "end":
            probs.append(f"{tag}: trace has no end line")
        for key in ("turns", "termination", "termination_detail"):
            if e.get(key) != r.get(key):
                probs.append(f"{tag}: {key} re-derives as {e.get(key)!r}, {INDEX_JSON} says {r.get(key)!r}")
        if bool(e.get("turn_cap_hit")) != bool(r.get("turn_cap_hit")):
            probs.append(f"{tag}: turn_cap_hit re-derives as {e.get('turn_cap_hit')}")
        sk = skill_loads(body, available)
        if sk["loaded"] != r.get("skills_loaded") or sk["not_loaded"] != r.get("skills_not_loaded"):
            probs.append(f"{tag}: skills loaded re-derive as {sk['loaded']}, {INDEX_JSON} says {r.get('skills_loaded')}")
        if library is None and (sk["foreign_frontmatter"] or sk["skill_tool_ok_outside_library"]):
            probs.append(f"{tag}: a base rollout carries skill text")
        sref = h.get("system_prompt")
        if sref != r.get("system_prompt") or (sref and sref not in ctx):
            probs.append(f"{tag}: system prompt reference {sref!r} not in solver_context")
        elif sref:
            told = told_skills(ctx[sref][1].decode("utf-8"))
            if told is not None and told != available:
                probs.append(f"{tag}: the system prompt listed skills {told}, available is {available}")
        if len(body) != h.get("messages"):
            probs.append(f"{tag}: header says {h.get('messages')} messages, the trace has {len(body)}")
    ic = index.get("counts") or {}
    for o in OUTCOMES:
        if ic.get(o) != counts.get(o, 0):
            probs.append(f"counts.{o} is {ic.get(o)}, the rows say {counts.get(o, 0)}")
    if ic.get("rollouts") != len(rows) or ic.get("turn_cap_hit") != turn_cap_hits:
        probs.append("counts.rollouts / counts.turn_cap_hit do not re-derive from the rows")
    # non-train ids, anywhere in the dir
    hits = text_id_hits(d, scanner)
    if hits:
        probs.append(f"NON-TRAIN TASK IDS in {len(hits)} place(s): {hits[:4]}")
    # the leg
    if leg_json is not None and run_dir is not None:
        lp = Path(leg_json)
        if not lp.is_file():
            probs.append(f"the leg {lp} is gone")
        else:
            if sha256_file(lp) != (index.get("leg") or {}).get("sha256"):
                probs.append(f"{lp.name} changed since the evidence was built")
            leg = json.loads(lp.read_text())
            r = OC.leg_outcome_counts(run_dir, leg)
            by = {(s["task"], s["sample"]): s for s in r["samples"]}
            bad = [f"{x['task']} s{x['sample']}" for x in rows
                   if (by.get((x["task"], x["sample"])) or {}).get("outcome") != x.get("outcome")
                   or (by.get((x["task"], x["sample"])) or {}).get("recorded") != x.get("score")]
            if bad:
                probs.append(f"{len(bad)} row(s) disagree with outcomes.py on the leg: {bad[:4]}")
            if "turn_exhausted" in leg and int(leg["turn_exhausted"]) != turn_cap_hits:
                probs.append(f"the leg records turn_exhausted={leg['turn_exhausted']}, the traces show "
                             f"{turn_cap_hits} turn-cap endings")
    probs += digest_agreement(d, rows)
    return probs


_DIGEST_ROLLOUT = re.compile(r"(?m)^## Rollout s(\d+) — (PASS|FAIL) \(")


def digest_agreement(d, rows) -> list:
    """Where the legacy per-task digest exists, it must tell the same verdict per sample and (when its
    DIGEST_INFO.json records the grader output as full) hold the same grader text."""
    d = Path(d)
    full = False
    try:
        info = json.loads((d / "DIGEST_INFO.json").read_text())
        full = bool((info.get("grader_output") or {}).get("full"))
    except (OSError, ValueError):
        pass
    probs = []
    by_task = collections.defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    for t, rs in sorted(by_task.items()):
        f = d / f"{stem_of(t)}.md"
        if not f.is_file():
            continue
        text = f.read_text(errors="replace")
        verdict = {int(m.group(1)): m.group(2) for m in _DIGEST_ROLLOUT.finditer(text)}
        for r in rs:
            k = r["sample"]
            want = "PASS" if r["outcome"] == "passed" else "FAIL"
            if verdict.get(k) != want:
                probs.append(f"{t} s{k}: the digest says {verdict.get(k)}, the evidence says {r['outcome']}")
            if full and r.get("grade"):
                g = (d / r["grade"]).read_text(errors="replace")
                m = re.search(r"(?m)^## Scorer diff — iter_\d+(?:_midgate)?_s%d \(authoritative\)\n\n(`{3,})\n" % k, text)
                if not m:
                    probs.append(f"{t} s{k}: the digest has no scorer-diff section")
                    continue
                end = text.find("\n" + m.group(1) + "\n", m.end())
                body = text[m.end():end]
                if body not in (g, g.rstrip("\n")) and r.get("redactions", 0) == 0:
                    probs.append(f"{t} s{k}: the digest's scorer diff differs from the grade file")
    return probs


def read_index(d) -> dict:
    """INDEX.json after a light check (every listed file at its recorded sha256). For the prompt
    renderer; the loop's evidence steps run check_dir() in full. Raises EvidenceError."""
    d = Path(d)
    ip = d / INDEX_JSON
    if not ip.is_file():
        raise EvidenceError(f"{d.name}: no {INDEX_JSON}")
    index = json.loads(ip.read_text())
    if index.get("schema") != SCHEMA:
        raise EvidenceError(f"{d.name}/{INDEX_JSON}: schema {index.get('schema')!r}")
    bad = []
    for r in index.get("rows") or []:
        for key in ("trace", "grade"):
            if r.get(key) and (not (d / r[key]).is_file() or sha256_file(d / r[key]) != r.get(f"{key}_sha256")):
                bad.append(r[key])
    for m in index.get("solver_context") or []:
        if not (d / m["path"]).is_file() or sha256_file(d / m["path"]) != m.get("sha256"):
            bad.append(m["path"])
    if bad:
        raise EvidenceError(f"{d.name}: {len(bad)} file(s) differ from {INDEX_JSON}: {bad[:3]}")
    return index


def uptake(index: dict) -> dict:
    """#234's measurement from an INDEX: rollouts that loaded at least one library skill."""
    rows = index.get("rows") or []
    lib = index.get("producing_library") or {}
    avail = [s["name"] for s in lib.get("skills") or []]
    per = collections.Counter(n for r in rows for n in r.get("skills_loaded") or [])
    consulted = sum(1 for r in rows if r.get("skills_loaded"))
    return {"rollouts": len(rows), "consulted": consulted,
            "uptake": (consulted / len(rows)) if rows else 0.0, "per_skill": dict(per),
            "never_loaded": sorted(set(avail) - set(per)), "library": lib.get("label")}


def summarize(d) -> dict:
    """Sizes and counts of a built dir, for the loop's log and the report."""
    d = Path(d)
    index = json.loads((d / INDEX_JSON).read_text())
    tot = 0
    for root, _dirs, files in os.walk(str(d), followlinks=False):
        for n in files:
            tot += (Path(root) / n).stat().st_size
    return {"dir": str(d), "counts": index["counts"], "sizes": index["sizes"],
            "index_json_bytes": (d / INDEX_JSON).stat().st_size, "index_md_bytes": (d / INDEX_MD).stat().st_size,
            "dir_total_bytes": tot}


# ─────────────────────────────────────────────────────────────── CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--run-dir", required=True)
    b.add_argument("--leg", required=True)
    b.add_argument("--split", required=True)
    b.add_argument("--n-samples", type=int, required=True)
    b.add_argument("--library")
    b.add_argument("--epoch", type=int)
    b.add_argument("--cache")
    b.add_argument("--bucket", default="")
    b.add_argument("--model")
    c = sub.add_parser("check")
    c.add_argument("dir")
    c.add_argument("--split", required=True)
    c.add_argument("--n-samples", type=int, required=True)
    c.add_argument("--library")
    c.add_argument("--leg")
    c.add_argument("--run-dir")
    t = sub.add_parser("cc-trace")
    t.add_argument("stream")
    t.add_argument("--session")
    t.add_argument("--user-message")
    t.add_argument("--system-prompt")
    t.add_argument("--out")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        try:
            s = build(a.out, run_dir=a.run_dir, leg_json=a.leg, split_path=a.split, n_samples=a.n_samples,
                      library=a.library, epoch=a.epoch, cross_check_cache=a.cache, bucket=a.bucket,
                      expect_model=a.model)
        except EvidenceError as e:
            print(f"REFUSED: {e}", file=sys.stderr)
            return 2
        print(json.dumps(s, indent=1))
        return 0
    if a.cmd == "check":
        probs = check_dir(a.dir, split_path=a.split, n_samples=a.n_samples, library=a.library,
                          leg_json=a.leg, run_dir=a.run_dir)
        for p in probs:
            print(f"PROBLEM: {p}")
        print(f"{len(probs)} problem(s)")
        return 1 if probs else 0
    tr = cc_trace(Path(a.stream).read_text(errors="replace"),
                  session_text=Path(a.session).read_text(errors="replace") if a.session else None,
                  user_message=Path(a.user_message).read_text() if a.user_message else None,
                  system_prompt=Path(a.system_prompt).read_text() if a.system_prompt else None)
    text = "".join(_dump_line(x) + "\n" for x in [{"type": "header", "schema": TRACE_SCHEMA, "solver": tr["solver"],
                                                   "model": tr["model"], "stats": tr["stats"],
                                                   "solver_init": tr["init"]}] + tr["lines"] + [tr["end"]])
    if a.out:
        Path(a.out).write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
