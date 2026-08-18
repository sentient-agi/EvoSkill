"""Run an evolver on the `direct` harness, emitting a log the existing audits can read.

Peer of the `claude_cli` path in `evolver_dispatch.py`. Selected by `evolver_harness: "direct"`.

── THE HARD REQUIREMENT, AND WHY IT IS FIRST IN THIS DOCSTRING ────────────────────────────────

`loop.py` audits the evolver's LOG for the two contamination modes that invalidate an arm
outright: reading a peer arm's evolved skills (#213) and reaching a held-out task spec (#186).
Both parse JSONL records of a specific shape. An evolver whose log does not carry that shape is
not "untraced" — it is UNAUDITED WHILE APPEARING AUDITED, which is strictly worse.

That is not hypothetical. Building this file is what surfaced #248: `check_no_peer_artifacts_read`
bound its session id only inside a `{"type":"system","subtype":"init"}` record, the real launcher
never emitted one, and so the check skipped every record and returned "no attempt to read another
run's skills" having examined nothing — on all three epochs of a live run. Proved with a
known-negative control (two blatant peer reads: PASS without the init record, FAIL with it).

So this runner emits, and `selftest()` asserts, the records the audits actually consume:

    {"type":"system","subtype":"init","session_id":S}          <- written BEFORE the model runs
    {"type":"assistant","session_id":S,"message":{"content":[
        {"type":"tool_use","name":"Bash","input":{"command":...}}]}}
    {"type":"user","session_id":S,"message":{"content":[
        {"type":"tool_result","content":...}]}}
    {"type":"result","subtype":"success"|"error", ...}

── WHY THE INIT RECORD IS WRITTEN BEFORE THE MODEL IS CALLED ──────────────────────────────────

#244's lesson, applied here: a log written only on completion does not exist for a run that is
killed, and a killed evolver that leaves no log is indistinguishable from one that never started.
Writing `system/init` first means a killed run leaves a log with a session and ZERO tool calls,
which post-#248 makes the audit return SKIP ("the audit did not run") and the loop HALT. That is
the correct outcome: a run that died mid-flight must not be able to present as clean.

── CONTAINMENT ────────────────────────────────────────────────────────────────────────────────

`direct` has no PreToolUse hooks, so the leak/peer guards the CC path gets from
`tau3_evolver.sh` do not exist here. Containment is the deny-first `sandbox-exec` profile from
`evolver_dispatch.build_evolver_sandbox_profile` — kernel-enforced, so an ungranted path cannot
be opened by any command however phrased, which is a stronger guarantee than a hook matching on
command strings. See that module's docstring for why the existing `direct` profile could not be
reused (it whitelists the whole repo, and every held-out spec lives in the repo).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

EVOLVER_TOOLS = ["Bash", "Read", "Write", "Edit", "Glob", "Grep"]


# ── record emission ─────────────────────────────────────────────────────────────────────────

class LogWriter:
    """Append-only JSONL, flushed per record.

    Flushed per record ON PURPOSE. A buffered writer loses exactly the tail that matters when a
    run is killed, and the tail is where a contaminating read would be.
    """

    def __init__(self, path: Path, session_id: str):
        self.path = path
        self.session_id = session_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", buffering=1)

    def write(self, rec: dict) -> None:
        rec.setdefault("session_id", self.session_id)
        self._fh.write(json.dumps(rec, default=str) + "\n")
        self._fh.flush()

    def init(self, model: str, cwd: str, grants: list[str]) -> None:
        # The record #248 showed the audits need in order to bind a session at all.
        self.write({"type": "system", "subtype": "init", "model": model,
                    "cwd": cwd, "grants": grants, "harness": "direct"})

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def messages_to_records(messages: list[Any]) -> list[dict]:
    """Convert the direct executor's returned history into audit-shaped records.

    The executor returns provider-shaped messages (role/content), not SDK stream events, so the
    mapping is explicit rather than a passthrough. Tool CALLS become assistant/tool_use and tool
    RESULTS become user/tool_result — which is what both audits walk.

    ONE INPUT KEY PER TOOL, because the audit blobs `command`, `file_path`, `pattern`, `path`
    and `glob` and searches that string. A tool whose argument lands under a key the audit does
    not read is a tool the audit cannot see — the #229 shape (an audit blind to one call path).
    """
    out: list[dict] = []
    for m in messages or []:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if not role:
            continue
        blocks = content if isinstance(content, list) else ([{"type": "text", "text": content}]
                                                            if content else [])
        if role == "assistant":
            norm = []
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                if b.get("type") in ("tool_use", "tool_call") or b.get("name"):
                    norm.append({"type": "tool_use",
                                 "id": b.get("id") or b.get("tool_use_id") or "",
                                 "name": b.get("name") or b.get("tool") or "",
                                 "input": b.get("input") or b.get("arguments") or {}})
                elif b.get("type") == "text":
                    norm.append({"type": "text", "text": b.get("text", "")})
            if norm:
                out.append({"type": "assistant", "message": {"content": norm}})
        elif role in ("user", "tool"):
            norm = []
            for b in blocks:
                if not isinstance(b, dict):
                    continue
                if b.get("type") in ("tool_result", "tool_response") or b.get("tool_use_id"):
                    norm.append({"type": "tool_result",
                                 "tool_use_id": b.get("tool_use_id") or "",
                                 "content": b.get("content") or b.get("output") or ""})
                elif b.get("type") == "text":
                    norm.append({"type": "text", "text": b.get("text", "")})
            if norm:
                out.append({"type": "user", "message": {"content": norm}})
    return out


# ── selftest ────────────────────────────────────────────────────────────────────────────────

def selftest() -> int:
    """Every check must be able to FAIL on a reachable input (#154).

    The load-bearing one is the LAST: a log this runner produces, carrying a peer-arm read, must
    make the real #213 audit FAIL. Anything less and the harness swap silently removes the guard.
    """
    import shutil
    import tempfile
    from src.orchestrator.invariants import check_no_peer_artifacts_read

    checks: list[tuple[str, bool]] = []
    tmp = Path(tempfile.mkdtemp(prefix=".evodirect_", dir=str(Path.home())))
    try:
        log = tmp / "e1.log"
        w = LogWriter(log, "SESSION-A")
        w.init("m", str(tmp), [str(tmp)])
        checks.append(("init record is written BEFORE any turn",
                       '"subtype": "init"' in log.read_text()))

        # A killed run == init only. The audit must refuse, not pass.
        r = check_no_peer_artifacts_read(log, tmp)
        checks.append(("init-only log (a killed run) -> audit SKIPs, does not PASS",
                       r.status == "SKIP"))

        # Now a clean tool call.
        for rec in messages_to_records([
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t1", "name": "Bash",
                 "input": {"command": "ls ./digests"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "task_004.json"}]},
        ]):
            w.write(rec)
        r2 = check_no_peer_artifacts_read(log, tmp)
        checks.append(("clean tool call -> audit PASSes and says it examined >0",
                       r2.status == "PASS" and "examined" in r2.detail))

        # THE ONE THAT MATTERS: a peer-arm read in THIS runner's format must FAIL the real audit.
        w.write(messages_to_records([
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "t2", "name": "Read",
                 "input": {"file_path": "outputs/tau3_a_sonnet/skills_v3/x/SKILL.md"}}]},
        ])[0])
        r3 = check_no_peer_artifacts_read(log, tmp)
        checks.append(("PEER-ARM READ in this runner's own format -> audit FAILs",
                       r3.status == "FAIL"))

        # Mapping fidelity: the audit blobs specific input keys; ours must land in them.
        recs = messages_to_records([{"role": "assistant", "content": [
            {"type": "tool_call", "name": "Grep", "arguments": {"pattern": "internal_"}}]}])
        checks.append(("tool_call/arguments alias maps to tool_use/input",
                       recs and recs[0]["message"]["content"][0]["type"] == "tool_use"
                       and recs[0]["message"]["content"][0]["input"] == {"pattern": "internal_"}))
        w.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    bad = 0
    for name, good in checks:
        print(f"  [{'ok' if good else 'BAD'}] {name}")
        bad += 0 if good else 1
    print(f"\n{len(checks)} checks, {bad} bad")
    return 1 if bad else 0


# ── entry point ─────────────────────────────────────────────────────────────────────────────

async def _run(a) -> int:
    from src.harness.direct.executor import execute_query
    from src.harness.direct.options import build_direct_options

    work = Path(a.work_dir).resolve()
    log = Path(a.log)
    sid = f"direct-{uuid.uuid4()}"
    w = LogWriter(log, sid)
    grants = [g for g in (a.grant or [])]
    w.init(a.model, str(work), grants)                    # BEFORE the model runs — see docstring

    prompt = Path(a.prompt).read_text() if a.prompt else ""
    t0 = time.time()
    messages: list[Any] = []
    status, err = "success", ""
    try:
        opts = build_direct_options(
            system="", schema={}, tools=EVOLVER_TOOLS,
            project_root=str(work), model=a.model,
            budget_type="time", budget_value=a.timeout_s,
            allowed_write_dirs=[str(work)] + grants,
            inject_skills=False,
        )
        opts["cwd"] = str(work)
        messages = await execute_query(opts, prompt, agent_name=f"evolver_e{a.epoch}")
    except Exception as e:                                # noqa: BLE001 — recorded, not swallowed
        status, err = "error", f"{type(e).__name__}: {e}"
        # The partial buffer is the ONLY record of what a timed-out run touched, and it is
        # exactly what the contamination audits need. Read in THIS task: wrapping the awaited
        # work in ensure_future copies the context and severs the ContextVar handoff.
        try:
            from src.harness.agent import _partial_messages
            messages = _partial_messages.get() or []
        except Exception:
            messages = []
    finally:
        for rec in messages_to_records(messages):
            w.write(rec)
        w.write({"type": "result", "subtype": status, "is_error": status != "success",
                 "error": err, "duration_ms": int((time.time() - t0) * 1000),
                 "num_turns": len(messages)})
        w.close()
    if status != "success":
        print(f"evolver(direct) FAILED: {err}", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, required=False, default=0)
    ap.add_argument("--work-dir")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--log")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--grant", action="append", default=[])
    ap.add_argument("--timeout-s", type=int, default=5400)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.work_dir and a.log):
        ap.error("--work-dir and --log are required unless --selftest")
    import asyncio
    return asyncio.run(_run(a))


if __name__ == "__main__":
    raise SystemExit(main())
