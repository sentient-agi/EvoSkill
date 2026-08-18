#!/usr/bin/env python3
"""PreToolUse hook: DENY any evolver tool call that would read held-out specs.

Wired into `claude -p` via --settings. Reads the hook payload on stdin and exits
2 to block the call, with the reason on stderr (which CC feeds back to the model,
so the evolver learns the boundary rather than silently retrying).

WHY A HOOK AND NOT --add-dir. --add-dir confines Read/Glob/Grep. It does NOT
confine Bash, which can `cd` anywhere the user can. The epoch-1 tau3 evolver
(2026-08-06) read third_party/tau3-bench/{data,src}/... freely despite four
narrow grants, and two directories from what it read sit the full 97-task spec
pool and per-task pytest walkthroughs that spell out held-out end states. The
launcher's own #179 note claimed `cd`-ing into the isolated dir closed repo
readability; it closed session placement and CLAUDE.md loading only. This hook is
the part that actually enforces.

WHAT IS ALLOWED ON PURPOSE. Environment knowledge is NOT leakage: tools.py,
db.json and documents/ describe the world the agent acts in, are identical for
train and held-out tasks, and the agent itself touches them at runtime. Blocking
those would forbid env-aware evolution. Only TASK SPECS and their expected end
states are held-out, and only for ids outside the train arm.

Env:
  EVOSKILL_EVOLVER_SPLIT   path to split json (required; refuses if unset/absent)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evolver_leak_audit import LEAK_PATH_RE, TASK_RE, held_forms  # noqa: E402

# Fields that can carry a path or a pattern, across every tool that reads.
_FIELDS = ("command", "file_path", "pattern", "path", "glob", "notebook_path", "url")


def _deny(msg: str) -> None:
    sys.stderr.write(
        "BLOCKED — this reads held-out (val/test) task material.\n"
        f"{msg}\n"
        "The train task specs in ./oracle/ are the only specs you may read. The "
        "environment itself (tools.py, db.json, the document corpus) is NOT "
        "restricted — read it freely. Do not try to reach the full task pool by "
        "another path; that boundary is what makes this evolution measurable.\n"
    )
    sys.exit(2)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0  # never wedge the evolver on a malformed payload

    split_path = os.environ.get("EVOSKILL_EVOLVER_SPLIT", "")
    if not split_path or not Path(split_path).is_file():
        # Fail CLOSED on the thing this exists to prevent: with no split we
        # cannot tell train from held-out, so block the spec pool wholesale.
        held: set[str] = set()
    else:
        split = json.loads(Path(split_path).read_text())
        held = held_forms(set(split.get("val", [])) | set(split.get("test", [])))

    ti = payload.get("tool_input") or {}
    text = " ".join(str(ti.get(k, "")) for k in _FIELDS if ti.get(k))
    if not text.strip():
        return 0

    m = LEAK_PATH_RE.search(text)
    if m:
        _deny(f"  matched held-out spec path: {m.group(0)!r}")

    # RUNTIME-SUPPLIED DENY PATHS. A benchmark whose ground truth is a data file keeps that file
    # wherever the operator put it (SSB v2: $SSBV2_ROOT), so the launcher passes the absolute
    # directories rather than this module guessing them. Absent is not "nothing to deny" — the
    # launcher is responsible for setting it, and its own deny-probe fails loudly if the result
    # does not actually block.
    for d in filter(None, os.environ.get("EVOSKILL_LEAK_DENY_PATHS", "").split(":")):
        if d in text:
            _deny(f"  reads a declared held-out data root: {d!r}")

    hit = sorted({t for t in TASK_RE.findall(text) if t in held})
    if hit:
        _deny(f"  names held-out task id(s): {hit}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
