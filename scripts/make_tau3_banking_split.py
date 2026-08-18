#!/usr/bin/env python3
"""Build a frozen, stratified train/val/test split for tau3 banking_knowledge.

WHY THIS EXISTS. `banking_knowledge` ships no `split_tasks.json` (retail,
airline and telecom do), so there is no canonical split to read. With only 97
tasks and a very wide difficulty spread this is the highest-leverage thing to
get right before spending any rollout budget.

THE SPREAD IS THE PROBLEM. `required_documents` runs from 1 to 30 and the
expected ACTION count from 1 to 37. On a knowledge-retrieval task the pass
probability compounds over hops, so a 1-document task and a 30-document task
are not remotely the same instrument. Drawing 32 tasks at random three times
would hand train, val and test materially different difficulty by luck alone,
and every cross-split comparison after that would be measuring the draw. So the
split is STRATIFIED on binned hop depth.

WHY NOT STRATIFY ON reward_basis. It is near-degenerate here: 87 of 97 tasks are
`('DB',)`, 9 are `('ACTION',)`, 1 is `('DB','NL_ASSERTION')`. Stratifying on it
would buy nothing — the same reason `dataset.py` rejected it for retail, where
112 of 114 tasks share one tuple.

READS THE AUTHORITATIVE PER-TASK FILES, NOT `tasks.json`. The single-file
aggregate disagrees with the per-task files that tau2's own registered loader
actually evaluates against — 13 of 97 tasks in v1.0.1, and 12 of 97 in the
0.2.1-dev checkout, so the drift is not shrinking. `environment.py` globs
`tasks/task_*.json`; this must match it or the split would be built against
metadata the episodes never see.

SELECTION WARNING, recorded in the output. 97 tasks means ~32 per split, and our
own scoreboard says a 40-task val plus frontier max-selection is a winner's
curse: the tau2 `action-efficiency` skill read +20pp with p≈0.01 on 40 val tasks
and tested NULL. Selection off this split must require train+val concordance
and/or repeats, never raw val max.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAU3 = ROOT / "third_party/tau3-bench"
OUT = ROOT / "outputs/tau3_banking/split.json"

# Hop-depth bins over `len(required_documents)`. Chosen so each bin holds enough
# tasks to divide three ways; the raw value has 23 distinct levels over 97 tasks,
# which is far too granular to stratify on directly.
BINS = [(1, 3), (4, 6), (7, 9), (10, 12), (13, 16), (17, 99)]
SEED = 730          # same seed convention as the SSB v2 splits
# NOTE: there is deliberately no FRACTIONS constant. One used to sit here reading
# `(33, 32, 32)  # train, val, test`, and it was BOTH dead code (referenced nowhere --
# the round-robin deal below decides the sizes) AND wrong about the split it claimed to
# describe: the produced split is train 32 / val 32 / test 33, so the constant
# mislabelled which arm got the extra task. A dead constant that disagrees with the
# artifact is worse than no constant, because it reads as the specification.


def bin_of(n: int) -> str:
    for lo, hi in BINS:
        if lo <= n <= hi:
            return f"{lo}-{hi}" if hi < 99 else f"{lo}+"
    raise SystemExit(f"unbinned required_documents count: {n}")


def load_tasks(tau3: Path) -> list[dict]:
    d = tau3 / "data/tau2/domains/banking_knowledge/tasks"
    files = sorted(d.glob("task_*.json"))
    if not files:
        raise SystemExit(f"no task_*.json under {d}")
    return [json.loads(f.read_text()) for f in files]


def build(tau3: Path) -> dict:
    tasks = load_tasks(tau3)
    by_bin: dict[str, list[str]] = defaultdict(list)
    depth: dict[str, int] = {}
    for t in tasks:
        n = len(t.get("required_documents") or [])
        depth[t["id"]] = n
        by_bin[bin_of(n)].append(t["id"])

    rng = random.Random(SEED)
    split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    # Deal round-robin WITHIN each bin, rotating which split gets the remainder
    # so no split systematically collects the leftovers of every bin.
    order = ["train", "val", "test"]
    for i, b in enumerate(sorted(by_bin)):
        ids = sorted(by_bin[b])
        rng.shuffle(ids)
        rot = order[i % 3:] + order[: i % 3]
        for j, tid in enumerate(ids):
            split[rot[j % 3]].append(tid)

    for k in split:
        split[k] = sorted(split[k])

    # ---- assertions: a split that is silently wrong is worse than none -------
    allids = [i for v in split.values() for i in v]
    if len(allids) != len(tasks):
        raise SystemExit(f"split covers {len(allids)} of {len(tasks)} tasks")
    if len(set(allids)) != len(allids):
        raise SystemExit("a task appears in more than one split")
    if set(allids) != {t["id"] for t in tasks}:
        raise SystemExit("split id set does not match the task id set")

    meta = {
        "domain": "banking_knowledge",
        "source": str(tau3),
        "tau_version": (tau3 / "pyproject.toml").read_text().split("version = ")[1]
                        .splitlines()[0].strip().strip('"') if (tau3 / "pyproject.toml").exists() else "unknown",
        "seed": SEED,
        "stratified_on": "binned len(required_documents)",
        "bins": [f"{lo}-{hi}" if hi < 99 else f"{lo}+" for lo, hi in BINS],
        "n": {k: len(v) for k, v in split.items()},
        "depth_mean": {k: round(sum(depth[i] for i in v) / len(v), 2) for k, v in split.items()},
        "depth_median": {k: sorted(depth[i] for i in v)[len(v) // 2] for k, v in split.items()},
        "per_bin": {k: dict(Counter(bin_of(depth[i]) for i in v)) for k, v in split.items()},
        "SELECTION_WARNING": (
            "~32 tasks per split. Our own scoreboard: a 40-task val with "
            "max-selection produced a +20pp/p=0.01 tau2 skill that tested NULL. "
            "Require train+val concordance and/or repeats; never select on raw "
            "val max."
        ),
        "READ_NOTE": (
            "Built from tasks/task_*.json, the files tau2's registered loader "
            "evaluates against. The single-file tasks.json aggregate disagrees "
            "on 13/97 tasks in v1.0.1 — do not rebuild from it."
        ),
        **split,
    }
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau3-root", default=str(DEFAULT_TAU3))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--verify", action="store_true",
                    help="rebuild and confirm it matches the file on disk")
    a = ap.parse_args()

    meta = build(Path(a.tau3_root))
    out = Path(a.out)

    if a.verify:
        if not out.exists():
            raise SystemExit(f"no split at {out}")
        have = json.loads(out.read_text())
        for k in ("train", "val", "test"):
            if have.get(k) != meta[k]:
                raise SystemExit(f"split on disk differs from a fresh rebuild in '{k}' "
                                 f"— it is NOT reproducible from this script + seed")
        print(f"OK — {out} reproduces exactly from seed {SEED}")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {out}")
    print(f"  tau version : {meta['tau_version']}")
    print(f"  n           : {meta['n']}")
    print(f"  depth mean  : {meta['depth_mean']}")
    print(f"  depth median: {meta['depth_median']}")
    for k in ("train", "val", "test"):
        print(f"  {k:<5} per-bin: {meta['per_bin'][k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
