#!/usr/bin/env python3
"""Build (or audit) the SSB v2 evolver oracle — TRAIN TASKS ONLY.

    SSBV2_ROOT=... EVOSKILL_SSBV2_ORACLE_SPLIT=... EVOSKILL_SSBV2_ORACLE_DIR=... \
        python3 scripts/make_ssbv2_oracle.py            # build
    ... python3 scripts/make_ssbv2_oracle.py --verify   # audit the oracle on disk

Mirrors scripts/make_tau3_oracle.py: DELEGATES to the tested machinery
(`ssbv2_agent.task_workspace.build_oracle_spec()` + `src.loop.oracle.materialize_oracle()`)
rather than hand-rolling a writer, and adds the same val/test firewall on top — the
oracle names TRAIN tasks only, asserted, because val is the selection signal and an
evolver that can read val's golden ranges can write skills tuned to the very split
meant to adjudicate them.

Env (all required; no guessed defaults — see the #225 incident in the tau3 twin):
    SSBV2_ROOT                    dataset root (the dir CONTAINING spreadsheetbench-v2/)
    EVOSKILL_SSBV2_ORACLE_SPLIT   the frozen split json (train/val/test id lists)
    EVOSKILL_SSBV2_ORACLE_DIR     where to materialize the oracle
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

for _v in ("SSBV2_ROOT", "EVOSKILL_SSBV2_ORACLE_SPLIT", "EVOSKILL_SSBV2_ORACLE_DIR"):
    if _v not in os.environ:
        raise SystemExit(f"{_v} must be set — this script guesses nothing (#225)")

SSBV2_ROOT = Path(os.environ["SSBV2_ROOT"])
SPLIT_FILE = Path(os.environ["EVOSKILL_SSBV2_ORACLE_SPLIT"])
ORACLE = Path(os.environ["EVOSKILL_SSBV2_ORACLE_DIR"])


def _load_split() -> dict:
    if not SPLIT_FILE.exists():
        raise SystemExit(f"no split at {SPLIT_FILE}")
    return json.loads(SPLIT_FILE.read_text())


def _meta_ids() -> set[str]:
    # Read task_id from each meta file's CONTENT. Reconstructing it from the filename
    # is ambiguous: safe_id maps "/" to "_", and category names ("Financial_Model")
    # already contain underscores.
    return {json.loads(p.read_text())["task_id"] for p in (ORACLE / "tasks").glob("*.json")}


def build(split: dict) -> int:
    train = list(split["train"])
    if not train:
        raise SystemExit("split has an empty train list — refusing to build an empty oracle")
    held_out = set(split.get("val", [])) | set(split.get("test", []))

    from src.agent_profiles.ssbv2_agent.dataset import load_ssbv2
    from src.agent_profiles.ssbv2_agent.task_workspace import build_oracle_spec
    from src.loop.oracle import materialize_oracle

    load_ssbv2(SSBV2_ROOT, n_train=80, n_val=60)     # populates TASK_INDEX
    spec = build_oracle_spec(SSBV2_ROOT, allowed_task_ids=set(train))
    out = materialize_oracle(ORACLE, spec)

    # THE FIREWALL, ASSERTED ON THE ARTIFACT, not trusted from the arguments.
    ids = _meta_ids()
    leaked = sorted(ids & held_out)
    if leaked:
        raise SystemExit(f"ORACLE LEAKS HELD-OUT TASKS: {leaked[:5]} — refusing to leave it")
    missing = sorted(set(train) - ids)
    if missing:
        raise SystemExit(f"oracle is missing train task meta: {missing[:5]}")
    if not (ORACLE / "README.md").is_file():
        raise SystemExit("materialize_oracle wrote no README.md — the evolver prompt "
                         "orders it read FIRST, so this oracle is unusable")
    print(f"  materialized {out}")
    print(f"  train tasks : {len(ids)} (train-only verified against {len(held_out)} held-out)")
    print(f"  README.md   : {(ORACLE / 'README.md').stat().st_size} bytes")
    return 0


def audit(split: dict) -> int:
    ids = _meta_ids()
    train, held = set(split["train"]), set(split.get("val", [])) | set(split.get("test", []))
    print(f"  tasks file : {len(ids)} meta file(s)")
    print(f"  train      : {len(ids & train)}/{len(train)}")
    print(f"  val|test   : {len(ids & held)}   (must be 0)")
    print(f"  README.md  : {'present' if (ORACLE / 'README.md').is_file() else 'ABSENT'}")
    ok = ids == train and not (ids & held) and (ORACLE / "README.md").is_file()
    print("  OK — train-only, README present." if ok else "  NOT OK — see above.")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="audit the oracle on disk instead of rebuilding it")
    a = ap.parse_args()
    split = _load_split()
    return audit(split) if a.verify else build(split)


if __name__ == "__main__":
    raise SystemExit(main())
