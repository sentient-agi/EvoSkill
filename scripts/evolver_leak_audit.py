#!/usr/bin/env python3
"""Audit a `claude -p` evolver session for held-out (val/test) leakage.

    evolver_leak_audit.py <session.jsonl> <split.json> [--domain banking_knowledge]

WHY THIS EXISTS. The evolver runs with `--add-dir` pointed at a leaf evidence
directory, which confines the Read tool -- and NOT the Bash tool. Bash can `cd`
anywhere the user can. On 2026-08-06 the epoch-1 evolver read
`third_party/tau3-bench/{data,src}/tau2/domains/banking_knowledge/` freely
(tools.py, db.json, documents/) which is legitimate ENVIRONMENT knowledge, but
two directories away sit:

    data/tau2/domains/<dom>/tasks.json          -- all 97 specs, train+val+test
    data/tau2/domains/<dom>/tasks/task_NNN.json -- per-task specs
    tests/test_domains/test_<dom>/tasks/test_task_NNN.py
        -- a PROSE WALKTHROUGH of the task: user id, entities, and the exact
           expected end state. test_task_028.py names all six corrected reward
           values for a held-out task in its module docstring.

Reading any of those for a held-out id contaminates the evolution: the skill
would encode an answer to a task the val gate is supposed to measure blind.

WHAT THIS CHECKS. Two independent detectors, because either alone misses cases:

  1. PATH -- any command naming a full-pool or per-task spec/test path. Catches
     leakage even when the id never appears literally (`cat tasks.json | jq`).
  2. ID -- any held-out task id appearing in a command OR in a tool RESULT.
     Catches leakage through a path this script does not know about.

Detector 2 scans results, not just commands, because the leak that matters is
what entered the model's context -- a `grep -r` the evolver did not realise was
recursive leaks just as hard as a deliberate read.

FILENAME-ONLY HITS ARE REPORTED SEPARATELY. A `find`/`ls` that returns
`test_task_028.py` reveals that the id exists -- which is public benchmark
structure -- but no spec content. That is a WARN, not a FAIL. The distinction is
made by checking whether the held-out id appears in the result attached to a
directory-listing command with no file body in its output.

Exit codes: 0 clean, 3 warn-only, 4 LEAK.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# BOTH BENCHMARKS' ID SHAPES. tau3 ids are `task_004`; SSB v2 ids are `Debugging/01_02` on the
# split and `Debugging_01_02` as a filename. Until 2026-08-12 this was `task_\d+` alone, so on an
# SSB v2 run the hook matched NOTHING and provided ZERO sealed-test protection — every held-out
# task was nameable and its reference workbook readable. The launcher's own deny-probe caught it
# and refused to dispatch (exit 13), which is the entire reason that probe exists: a guard is not
# protection until it has been seen to DENY.
#
# Both SSB v2 forms are matched rather than normalising one to the other, because stem -> id is
# ambiguous (category names contain underscores: "Financial_Model_01_02" could split two ways) and
# a wrong normalisation would file a held-out task under train — the exact failure being guarded.
# Membership is checked against the split expressed in BOTH forms instead; see _held_forms.
# ONE DEFINITION, imported. These lived here in triplicate -- this module, the live hook
# (via this module) and invariants.check_no_heldout_reachable, which had INVENTED its own
# and missed SSB v2 entirely. A copy is how two guards become one real one. The patterns
# now live in src/leak_patterns.py -- at the TOP level, not under src/harness/, whose
# __init__ imports pydantic: Claude Code runs the hook under the SYSTEM python3, so a
# harness-package home silently killed the hook on import (exit 1, which does not block).
# The names are re-exported so this module's existing consumers are untouched.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.leak_patterns import (                  # noqa: E402
    LEAK_PATH_RE, POOL_RES, TASK_RE, id_forms,
)

held_forms = id_forms          # historical name; the hook and this module both use it


# Directory-listing commands: an id in their output is a filename, not content.
LISTING_RE = re.compile(r"^\s*(find|ls|tree|basename|dirname)\b|\|\s*(ls|find)\b")


def _texts(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _texts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _texts(v)


def load_session(path: Path):
    """Return (commands, results) where results are joined to their command."""
    rows = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    cmds: dict[str, tuple[str, str]] = {}  # tool_use_id -> (tool_name, text)
    for r in rows:
        if r.get("type") != "assistant":
            continue
        content = (r.get("message") or {}).get("content") or []
        for c in content:
            if c.get("type") != "tool_use":
                continue
            i = c.get("input") or {}
            text = " ".join(
                str(i.get(k, "")) for k in ("command", "file_path", "pattern", "path", "glob")
            ).strip()
            if not text:
                text = json.dumps(i)
            cmds[c["id"]] = (c["name"], text)

    results: list[tuple[str, str]] = []  # (tool_use_id, blob)
    for r in rows:
        if r.get("type") != "user":
            continue
        content = (r.get("message") or {}).get("content")
        if not isinstance(content, list):
            continue
        for c in content:
            if c.get("type") == "tool_result":
                results.append((c.get("tool_use_id", ""), "\n".join(_texts(c.get("content")))))
    return cmds, results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("split")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    split = json.loads(Path(args.split).read_text())
    # BOTH SPELLINGS, for MEMBERSHIP. src/leak_patterns.py says the point of matching both
    # forms is that an SSB v2 id is "Debugging/05_05" on the split and "Debugging_05_05" as a
    # filename -- and the consolidation that moved id_forms into the shared module never
    # converted THIS module's own membership tests. So the post-hoc audit, the artifact an
    # operator runs to certify an epoch, reported CLEAN on a real held-out spec-content read
    # written in the filename form, while invariants.check_no_heldout_reachable FAILED the
    # same log because it kept its own converter. A shared definition that only one consumer
    # applies is the copy problem wearing a different hat.
    train_raw = set(split["train"])
    held_raw = set(split.get("val", [])) | set(split.get("test", []))
    train = id_forms(train_raw)
    held = id_forms(held_raw)
    if not held_raw:
        print("REFUSING: split names no val/test arm — nothing to protect", file=sys.stderr)
        return 2

    cmds, results = load_session(Path(args.session))
    fails: list[str] = []
    warns: list[str] = []

    # --- detector 1: leak-bearing paths -------------------------------------
    for tid, (tool, text) in cmds.items():
        m = LEAK_PATH_RE.search(text)
        if m:
            fails.append(f"PATH  {tool}: matched {m.group(0)!r} in: {text[:220]}")

    # --- detector 2: held-out ids in commands -------------------------------
    for tid, (tool, text) in cmds.items():
        hit = {t for t in TASK_RE.findall(text) if t in held}
        if hit:
            fails.append(f"ID-CMD {tool}: {sorted(hit)} in: {text[:220]}")

    # --- detector 2b: held-out ids in tool results --------------------------
    for tid, blob in results:
        hit = {t for t in TASK_RE.findall(blob) if t in held}
        if not hit:
            continue
        tool, text = cmds.get(tid, ("?", ""))
        if LISTING_RE.search(text):
            warns.append(f"ID-LIST {tool}: {sorted(hit)} as filenames from: {text[:160]}")
        else:
            fails.append(f"ID-RES {tool}: {sorted(hit)} in output of: {text[:220]}")

    # COUNT CANONICAL IDS, NOT TOKENS. `train` now holds both spellings, so counting matched
    # tokens against it would double the numerator on SSB v2 and make the ratio below
    # meaningless. Map each match back to the id it names.
    _seen = {t for _, (_, x) in cmds.items() for t in TASK_RE.findall(x)}
    n_train = len({c for c in train_raw
                   if c in _seen or c.replace("/", "_") in _seen})
    if not args.quiet:
        print(f"  session   : {args.session}")
        print(f"  tool calls: {len(cmds)}  results: {len(results)}")
        print(f"  protecting: {len(held_raw)} held-out ids; {n_train}/{len(train_raw)} train ids referenced")

    for w in warns:
        print(f"  WARN  {w}")
    for f in fails:
        print(f"  LEAK  {f}")

    if fails:
        print(f"\nLEAK: {len(fails)} finding(s). Held-out content entered the evolver's "
              f"context. This epoch's skills are CONTAMINATED — discard them and rerun "
              f"with the leak path closed. Do not 'review and keep the good ones'.")
        return 4
    if warns:
        print(f"\nWARN-ONLY: {len(warns)} filename-level exposure(s), no spec content. "
              f"Held-out ids existing is public benchmark structure. CLEAR to accept.")
        return 3
    print("\nCLEAN: no held-out spec content and no held-out ids reached the evolver.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
