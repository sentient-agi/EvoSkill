#!/usr/bin/env python3
"""Build and freeze the 60/60/rest split for the 2026-08-01 SSB v2 evolution.

Written once, committed to JSON, and never recomputed — every later phase reads
the frozen file. Regenerating a split from a seed at each phase is how a run
silently changes its own train set halfway through.

EXCLUSION. The 30 tasks of the base30_0731b run are dropped from ALL THREE
splits, not just val. I analysed those 30 in depth as evolver on 2026-07-31
(90 rollouts, per-task scorer diffs, an authored skill). Tasks I have already
mined cannot serve as train (epoch-1 analysis would start pre-loaded), as val
(the firewall would leak), or as test (the final number is the one claim this
whole run exists to make). So the usable pool is 297 - 30 = 267.
"""
from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent_profiles.ssbv2_agent.dataset import load_ssbv2  # noqa: E402

if "SSBV2_ROOT" not in os.environ:
    raise SystemExit("SSBV2_ROOT must point to your SpreadsheetBench v2 checkout")
SSBV2_ROOT = Path(os.environ["SSBV2_ROOT"])
OUT = Path(__file__).resolve().parents[1] / "outputs/ssbv2_evo_0801/split.json"

SEED = 801
N_TRAIN = 60
N_VAL = 60
PRIOR_SEED = 731        # the base30_0731b sample, to be excluded
N_PRIOR = 30

# ALL THREE SPLITS SHARE ONE DISTRIBUTION — uniform across categories
# (the maintainer, 2026-08-01: "make sure train/val/test follow the same distribution").
# train 20/20/20, val 20/20/20, test 40/40/40.
#
# This is not cosmetic. The three categories have wildly different base rates
# (0731b run: Template 38%, Financial_Model 22%, Debugging 0%). Under an
# unstratified draw train came out 42% Debugging against val's 32% — i.e. the
# category where almost nothing is winnable was over-represented in precisely
# the split that drives skill edits, so a train->val drop could be a category
# artefact rather than a skill effect. Equal mixes make every cross-split
# comparison a like-for-like one.
N_TEST = 120
TEST_PER_CATEGORY = N_TEST // 3          # 40
TRAIN_PER_CATEGORY = N_TRAIN // 3        # 20
VAL_PER_CATEGORY = N_VAL // 3            # 20


def main() -> int:
    train, val, test = load_ssbv2(SSBV2_ROOT, n_train=80, n_val=60)
    pool = list(train) + list(val) + list(test)

    # Reproduce the 0731b sample exactly as run_ssbv2_base30_dsv4f_n3.py drew it.
    prior = random.Random(PRIOR_SEED).sample(pool, N_PRIOR)
    prior_ids = {json.loads(q)["task_id"] for q, _, _ in prior}
    assert len(prior_ids) == N_PRIOR, f"expected {N_PRIOR} prior ids, got {len(prior_ids)}"

    usable = [t for t in pool if json.loads(t[0])["task_id"] not in prior_ids]
    assert len(usable) == len(pool) - N_PRIOR

    by_cat: dict[str, list] = {}
    for row in usable:
        by_cat.setdefault(json.loads(row[0]).get("category", "?"), []).append(row)

    need = TRAIN_PER_CATEGORY + VAL_PER_CATEGORY + TEST_PER_CATEGORY   # 80
    short = {c: len(v) for c, v in by_cat.items() if len(v) < need}
    if short:
        raise SystemExit(
            f"cannot draw {need}/category — short: {short}. "
            f"Available: { {c: len(v) for c, v in by_cat.items()} }"
        )

    # Draw all three splits per category from ONE shuffle of that category, so
    # every split ends up with the same mix. Seed from the category's INDEX in
    # sorted order, NEVER from hash(cat): Python randomises string hashing per
    # process, so a hash-derived seed would silently produce a different
    # "frozen" split on every run.
    tr, va, te, unused = [], [], [], []
    for i, cat in enumerate(sorted(by_cat)):
        rows = by_cat[cat][:]
        random.Random(SEED + 1000 * (i + 1)).shuffle(rows)
        a, b = TRAIN_PER_CATEGORY, TRAIN_PER_CATEGORY + VAL_PER_CATEGORY
        tr.extend(rows[:a])
        va.extend(rows[a:b])
        te.extend(rows[b:b + TEST_PER_CATEGORY])
        unused.extend(rows[b + TEST_PER_CATEGORY:])

    def ids(rows):
        return [json.loads(q)["task_id"] for q, _, _ in rows]

    def cats(rows):
        c = {}
        for q, _, _ in rows:
            c[json.loads(q).get("category", "?")] = c.get(json.loads(q).get("category", "?"), 0) + 1
        return c

    split = {
        "seed": SEED,
        "excluded_prior_run": {"seed": PRIOR_SEED, "n": N_PRIOR, "task_ids": sorted(prior_ids),
                               "reason": "analysed as evolver 2026-07-31; unusable as train, val or test"},
        "pool_total": len(pool),
        "usable_after_exclusion": len(usable),
        "train": ids(tr), "val": ids(va), "test": ids(te),
        "unused": ids(unused),
        "counts": {"train": len(tr), "val": len(va), "test": len(te),
                   "unused": len(unused)},
        "category_mix": {"train": cats(tr), "val": cats(va), "test": cats(te)},
        "sampling": "stratified; identical category distribution in train/val/test "
                    f"({TRAIN_PER_CATEGORY}/{VAL_PER_CATEGORY}/{TEST_PER_CATEGORY} per category)",
    }

    # Disjointness is the one property everything downstream assumes. Assert it
    # rather than trust the slicing.
    s_tr, s_va, s_te = set(split["train"]), set(split["val"]), set(split["test"])
    s_un = set(split["unused"])
    assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), "splits overlap"
    assert not (s_un & (s_tr | s_va | s_te)), "unused overlaps a split"
    assert not (s_tr | s_va | s_te) & prior_ids, "prior-run task leaked into a split"
    # Test is now a 120-task STRATIFIED DRAW, not the whole remainder, so the
    # four parts partition `usable` — they no longer sum to it three-way.
    assert len(s_tr) + len(s_va) + len(s_te) + len(s_un) == len(usable)
    assert len(s_te) == N_TEST, f"test is {len(s_te)}, expected {N_TEST}"
    # The point of the whole design: all three splits carry the SAME category
    # distribution, so no cross-split comparison can be a category artefact.
    for name, per in (("train", TRAIN_PER_CATEGORY), ("val", VAL_PER_CATEGORY),
                      ("test", TEST_PER_CATEGORY)):
        mix = split["category_mix"][name]
        assert set(mix.values()) == {per}, f"{name} not balanced: {mix}"
    fracs = {n: {c: v / sum(m.values()) for c, v in m.items()}
             for n, m in split["category_mix"].items()}
    assert fracs["train"] == fracs["val"] == fracs["test"], f"distributions differ: {fracs}"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(split, indent=1))
    print(f"wrote {OUT}")
    print(f"  pool {len(pool)} - {N_PRIOR} excluded = {len(usable)} usable")
    print(f"  train {len(tr)}  val {len(va)}  test {len(te)}")
    for k in ("train", "val", "test"):
        print(f"  {k:<6} categories: {split['category_mix'][k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
