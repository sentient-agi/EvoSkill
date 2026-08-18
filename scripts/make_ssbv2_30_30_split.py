#!/usr/bin/env python3
"""Carve a 30-train / 30-val split out of the FROZEN SSB v2 60/60, as strict subsets.

the maintainer's directive, 2026-08-02: "resample another 30-train 30-val split. the
30-train comes from the current 60-train, the new 30-val comes from the current
60-val. keep distribution the same."

WHY SUBSETS AND NOT A FRESH DRAW. The 60/60/120 in split.json is frozen, and 357
cap-45 base rollouts already exist against 120 of those tasks. Drawing a new 30
from the whole pool would produce tasks with no history; drawing from inside the
frozen 60 means the pre-LibreOffice rollouts for every selected task ALREADY
EXIST. That is the cheap control for the LibreOffice arm — old-image base and
new-image base on the identical task set, one of the two legs already paid for.

WHY THE DISTRIBUTION IS EXACTLY 10/10/10. The frozen 60s are 20/20/20 across
Debugging / Financial_Model / Template, and the categories are not
interchangeable instruments: on the epoch-2 phase the 40-turn cap bit
Financial_Model 40/120, Debugging 25/120 and Template 0/120. A draw that came out
12/9/9 would shift the mix of a cap-sensitive category and any old-vs-new
comparison would partly measure that shift. So the per-category count is asserted,
not hoped for.

AND WHY THAT IS STILL NOT ENOUGH — a warning recorded in the output. Halving the
val set halves the resolution of exactly the quantity used for selection. At a
~20% base rate, 30 tasks x N=3 gives a standard error near 4.2pp on a leg, so
old-vs-new differences below roughly 8-9pp are not separable here. This split is
sized for a CHEAP arm, not for selecting a winner; our own scoreboard has a tau2
skill that read +20pp at p~0.01 on a 40-task val and tested NULL.

TEST IS NEVER TOUCHED. It stays sealed, and that is asserted rather than assumed.

    python3 scripts/make_ssbv2_30_30_split.py
    python3 scripts/make_ssbv2_30_30_split.py --verify
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "outputs/ssbv2_evo_0801/split.json"
OUT = ROOT / "outputs/ssbv2_evo_0801/split_30_30.json"

SEED = 730                    # same seed convention as every other split here
PER_CATEGORY = 10             # 20/20/20 halved -> 10/10/10
CATEGORIES = ("Debugging", "Financial_Model", "Template")


def category(task_id: str) -> str:
    return task_id.split("/")[0]


def build() -> dict:
    frozen = json.loads(FROZEN.read_text())

    # Deal per category from a SORTED list before shuffling. The frozen file's own
    # order came out of a different sampler; sorting first makes this script's
    # output a function of (frozen id set, seed) alone, so `--verify` still
    # reproduces if the frozen file is ever rewritten with the same ids in a
    # different order.
    rng = random.Random(SEED)
    out: dict[str, list[str]] = {}
    for key in ("train", "val"):
        pool = frozen[key]
        chosen: list[str] = []
        for cat in CATEGORIES:
            ids = sorted(t for t in pool if category(t) == cat)
            if len(ids) < PER_CATEGORY:
                raise SystemExit(
                    f"{key}/{cat} has only {len(ids)} tasks, need {PER_CATEGORY}")
            chosen.extend(rng.sample(ids, PER_CATEGORY))
        out[key] = sorted(chosen)

    # ---- assertions. A split that is silently wrong is worse than none. -------
    for key in ("train", "val"):
        got = Counter(category(t) for t in out[key])
        want = {c: PER_CATEGORY for c in CATEGORIES}
        if dict(got) != want:
            raise SystemExit(f"{key} category mix is {dict(got)}, expected {want}")
        if len(set(out[key])) != len(out[key]):
            raise SystemExit(f"{key} contains a duplicate task id")
        # STRICT SUBSET of the frozen split of the SAME name — this is the whole
        # point of the exercise. A new-30-val containing a frozen-TRAIN task would
        # silently break the val firewall the 0801 run is built on.
        extra = sorted(set(out[key]) - set(frozen[key]))
        if extra:
            raise SystemExit(f"{key} is not a subset of frozen {key}: {extra[:5]}")

    # Cross-contamination, both directions, checked explicitly rather than being
    # implied by the subset check above.
    if set(out["train"]) & set(out["val"]):
        raise SystemExit("train and val overlap")
    sealed = set(frozen["test"])
    for key in ("train", "val"):
        leak = sorted(set(out[key]) & sealed)
        if leak:
            raise SystemExit(f"{key} touches the SEALED test split: {leak[:5]}")

    return {
        "derived_from": str(FROZEN.relative_to(ROOT)),
        "derived_from_counts": {k: len(frozen[k]) for k in ("train", "val", "test")},
        "seed": SEED,
        "method": (
            "per-category rng.sample of 10 from each of Debugging/Financial_Model/"
            "Template within the frozen split of the same name; ids sorted before "
            "sampling so the result depends only on the frozen id set and the seed"
        ),
        "counts": {k: len(out[k]) for k in ("train", "val")},
        "category_mix": {k: dict(Counter(category(t) for t in out[k]))
                         for k in ("train", "val")},
        "is_strict_subset": True,
        "test": [],
        "TEST_NOTE": (
            "Deliberately EMPTY, not copied. The frozen test split stays sealed; a "
            "runner pointed at this file cannot reach it even by passing "
            "EVOSKILL_SPLIT_KEY=test. Use split.json for anything test-related, "
            "after epoch 4."
        ),
        "POWER_WARNING": (
            "30 tasks x N=3 at a ~20% base rate is about 4.2pp standard error per "
            "leg, so old-image vs new-image differences under roughly 8-9pp are "
            "not resolvable on this split. Sized for a cheap arm, not for "
            "selection. Precedent: a tau2 skill read +20pp at p~0.01 on a 40-task "
            "val and tested NULL."
        ),
        "REUSE_NOTE": (
            "Every task here is also in the frozen 60/60, so the 357 cap-45 records "
            "in .cache/ssbv2_evo0801_cap45_cache (tree_hash bucket bfb5bd90d1e0, "
            "built on solver image 974b00f17fbe with NO formula engine) cover the "
            "old-image leg. Those records are only valid for the OLD image: "
            "tree_hash does not include the docker image, so never point a "
            "new-image run at that cache dir. Check with "
            "scripts/ssbv2_solver_env_report.py --cache <dir>."
        ),
        "train": out["train"],
        "val": out["val"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="rebuild and confirm it matches the file on disk")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    meta = build()
    out = Path(a.out)

    if a.verify:
        if not out.exists():
            raise SystemExit(f"no split at {out}")
        have = json.loads(out.read_text())
        for k in ("train", "val"):
            if have.get(k) != meta[k]:
                raise SystemExit(f"split on disk differs from a fresh rebuild in "
                                 f"'{k}' — it is NOT reproducible from this "
                                 f"script + seed {SEED}")
        if have.get("test"):
            raise SystemExit("the split on disk has a non-empty test list")
        print(f"OK — {out} reproduces exactly from seed {SEED}, test still empty")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {out}")
    print(f"  counts       : {meta['counts']}")
    for k in ("train", "val"):
        print(f"  {k:<5} mix    : {meta['category_mix'][k]}")
    print(f"  strict subset: {meta['is_strict_subset']} (asserted, both splits)")
    print(f"  test         : {meta['test']} (sealed — see TEST_NOTE)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
