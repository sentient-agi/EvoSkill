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
  EVOSKILL_EVOLVER_TRAIN_IDS  whitespace/comma-separated TRAIN ids (not secret --
                              the evolver reads them off its own digest filenames).
                              REQUIRED: unset/empty denies every id, loudly.
  EVOSKILL_EVOLVER_ID_RE      optional narrowed candidate-id regex, derived by the
                              launcher from the train ids' shape. Default: TASK_RE.
"""
from __future__ import annotations

import json
import os
import re
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

    # THE HOOK IS TOLD WHAT IT MAY SEE, NEVER WHAT IT MUST HIDE.
    #
    # This used to read the split named by EVOSKILL_EVOLVER_SPLIT and deny ids in
    # val+test. That question can only be answered by the split -- and the split is
    # THE ONE ARTIFACT CONTAINMENT EXISTS TO HIDE. evolver_dispatch's profile denies
    # the run dir by name, and cli.py writes the split there, precisely because it
    # "LISTS ALL 47 TEST IDS". The hook runs INSIDE that sandbox, so Path.is_file()
    # raised PermissionError (CPython's _ignore_error covers ENOENT/ENOTDIR/EBADF/
    # ELOOP, not EPERM), the hook exited 1, and 1 does not block. Measured: rc=1
    # under the emitted profile vs rc=2 without it, on the same held-out id, with
    # containment ON BY DEFAULT and both shipped smokes recording sandboxed:true.
    #
    # So the question is inverted. "Is this id in this run's TRAIN arm?" has an
    # answer the evolver ALREADY POSSESSES -- the digest filenames it is handed ARE
    # the train ids -- so it is not secret and can be passed IN BAND, through the
    # process environment, with no filesystem read at all. Nothing held-out crosses
    # the boundary. The fix REMOVES a read the kernel forbids; it does not add a
    # grant, which also means it cannot be silently un-fixed by a dropped grant.
    #
    # A SECOND PROPERTY, not asked for: the old denylist was itself a held-out
    # MEMBERSHIP ORACLE. deny <=> id in val+test, so an evolver with Bash could
    # enumerate task_001..task_097 and read the sealed set off the BLOCKED banners.
    # The allowlist answers deny <=> id not in train, and the evolver already knows
    # train exactly. It returns strictly zero new information.
    #
    # NO FALLBACK to reading the split. A "read it if you can, else use the
    # allowlist" branch would rebuild the defect's shape: two paths, one of which
    # is never exercised under the shipped default.
    train_raw = os.environ.get("EVOSKILL_EVOLVER_TRAIN_IDS", "")
    allow = held_forms({t for t in re.split(r"[\s,]+", train_raw) if t})
    blind = not allow
    id_re = TASK_RE
    _custom = os.environ.get("EVOSKILL_EVOLVER_ID_RE", "").strip()
    if _custom:
        # A NARROWED candidate pattern, derived by the launcher from the TRAIN ids'
        # own shape and only after it has proved, against the split it can legally
        # read, that every held-out id still matches. Narrowing matters because
        # inverting to an allowlist inverts the cost of a false positive from
        # harmless to BLOCKED: under a denylist a token that merely looks like an id
        # hits nothing, under an allowlist it is denied, and a denied read is silent
        # (the evolver just works around it).
        try:
            id_re = re.compile(_custom)
        except re.error:
            blind = True

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

    found = id_re.findall(text)
    if blind:
        # DISTINGUISHABLE from a deny-for-cause, and from a crash. python3 exits 2
        # for a missing file too, so exit code alone proves nothing; the launcher's
        # probes require this banner, and require that a DENY probe does NOT see the
        # BLIND one -- otherwise a guard that denies everything passes the deny probe
        # while being useless.
        if found:
            sys.stderr.write(
                "BLOCKED — LEAK GUARD BLIND.\n"
                "  EVOSKILL_EVOLVER_TRAIN_IDS is unset/empty, or EVOSKILL_EVOLVER_ID_RE "
                "did not compile, so this hook cannot tell train from held-out and is "
                "refusing every task id. The launcher sets both; if you are running the "
                "evolver by hand, set them from this run's split.\n")
            sys.exit(2)
        return 0
    bad = sorted({t for t in found if t not in allow})
    if bad:
        _deny(f"  names task id(s) not in this run's train arm: {bad}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
