"""Build a 10-train + 10-val split from officeqa_pro.csv.

Process:
1. Drop ~10 external-data questions (regex on Macrotrends/BLS/FRED/etc.)
2. Haiku-classify the remaining into 6 categories (using pro7 anchors)
3. Pull hardness signal from eval_pro133_swap_iter1.pkl (timeout > judge=DIFFERENT > EQUIVALENT)
4. Allocate 20 slots proportional to category size
5. Per category, hardest→easiest; first half train, second half val
"""
import asyncio
import os
import pickle
import re
import sys

import pandas as pd
from anthropic import AsyncAnthropic

DS = "/Users/dastin/dev/officeqa/officeqa_pro.csv"
ANCHORS = "/Users/dastin/dev/EvoSkill/.dataset/pro7_train.csv"
EVAL_PKL = "/Users/dastin/dev/EvoSkill/results/eval_pro133_swap_iter1.pkl"
OUT_TRAIN = "/Users/dastin/dev/EvoSkill/.dataset/pro20_train.csv"
OUT_VAL = "/Users/dastin/dev/EvoSkill/.dataset/pro20_val.csv"
JUDGE_MODEL = "claude-haiku-4-5-20251001"

EXTERNAL_PAT = re.compile(
    r"\bmacrotrends\b|\bworld bank\b|\bbls\b|\bbureau of labor\b|\bbea\b|"
    r"\bbureau of economic\b|\bcensus bureau\b|\bnber\b|\bfred\b|"
    r"\bwikipedia\b|\boecd\b|\bimf\b|\bcpi-?u?\b|\bcpi index\b|"
    r"\binflation-adjusted\b|\bofficial.{0,30}cpi\b",
    re.I,
)

CATS = ["simple_lookup", "aggregation", "statistical", "visual_reasoning", "table_parsing", "multi_bulletin"]


async def classify_one(client, sem, question, anchor_block):
    prompt = f"""You are classifying a Treasury Bulletin question into EXACTLY ONE of these categories. Use the FIRST matching rule:

1. **multi_bulletin**: requires combining data from MANY DISTINCT BULLETIN ISSUES (e.g., "January bulletins from 1969 to 1980", "every December bulletin"). Just spanning a date range that fits in a few bulletins does NOT count — the question must explicitly enumerate or sum over many bulletin issues.

2. **statistical**: REQUIRES a true statistical operation by name: variance, standard deviation, R-squared, correlation, regression (linear/OLS), percentile (e.g. "Hazen", "75th percentile"), geometric mean, harmonic mean, CAGR, expected shortfall, VaR, ARIMA, Hodrick-Prescott, Hill estimator, Pareto exponent, Gini coefficient. If the only math is a difference, ratio, sum, log, or simple growth, it is NOT statistical.

3. **visual_reasoning**: explicitly references a chart, plot, figure, graph, or visual element (e.g., "the bar chart on page 5", "from the line graph"). Plain tables do NOT count.

4. **table_parsing**: requires non-trivial column/row navigation across a table with many similar-looking columns or row groupings (e.g., "the 16 NYC Banks among the 30 banks listed", "investor categories that exceed $500M"). Just reading values from a table does NOT count.

5. **aggregation**: sum / average / total / max / min / count over multiple rows, periods, or items. Includes "total expenditures over a year", "sum across X categories", "average across months". Differences between two values are NOT aggregation; that's simple_lookup.

6. **simple_lookup** (fallback): direct extraction of one or two values, possibly with one trivial arithmetic step (subtraction, ratio, difference, log of ratio). When in doubt, use this.

Anchor examples (one per category):
{anchor_block}

Question:
{question}

Reply with ONLY one of: simple_lookup, aggregation, statistical, visual_reasoning, table_parsing, multi_bulletin. Apply the rules in order — first match wins."""
    async with sem:
        m = await client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=20,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in m.content if hasattr(b, "text")).strip().lower()
        for c in CATS:
            if c in text:
                return c
        return "simple_lookup"  # fallback


async def main():
    df = pd.read_csv(DS)
    print(f"Loaded {len(df)} pro questions")

    # Step 1: drop external_data
    is_ext = df["question"].str.contains(EXTERNAL_PAT.pattern, regex=True, case=False, na=False)
    print(f"External-data questions to drop: {is_ext.sum()}")
    df_internal = df[~is_ext].copy()
    print(f"After dropping: {len(df_internal)} questions")

    # Step 2: classify
    anchors_df = pd.read_csv(ANCHORS)
    anchors_df = anchors_df[anchors_df["category"].isin(CATS)]
    anchor_block = "\n".join(
        f"  [{r['category']}] {r['question'][:160]}" for _, r in anchors_df.iterrows()
    )

    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    sem = asyncio.Semaphore(15)
    print("Classifying with Haiku ...")
    cats = await asyncio.gather(
        *[classify_one(client, sem, row["question"], anchor_block) for _, row in df_internal.iterrows()]
    )
    df_internal["category"] = cats

    cat_counts = df_internal["category"].value_counts()
    print("\nCategory counts (after dropping external_data):")
    for c, n in cat_counts.items():
        print(f"  {c:<18} {n}")

    # Step 3: hardness from eval pkl
    with open(EVAL_PKL, "rb") as f:
        eval_results = pickle.load(f)
    full_data = pd.read_csv(DS)
    eval_lookup = {}  # uid -> (hardness_tier, rel_err, turns)
    for r in eval_results:
        if r.index >= len(full_data):
            continue
        uid = full_data.iloc[r.index]["uid"]
        if r.error or not r.trace or not r.trace.output:
            eval_lookup[uid] = ("TIMEOUT", 1.0, 9999)
            continue
        gt = str(full_data.iloc[r.index]["answer"]).strip().lower()
        ag = str(r.trace.output.final_answer).strip().lower()
        # quick rel_err calc (reuse the binary search pattern from runner._measure_rel_error)
        from src.loop.runner import _measure_rel_error
        rel = _measure_rel_error(ag, gt)
        turns = r.trace.num_turns or 0
        if gt == ag:
            tier = "EQUIVALENT"
        elif rel < 0.01:
            tier = "EQUIVALENT"  # close enough
        else:
            tier = "DIFFERENT"
        eval_lookup[uid] = (tier, rel, turns)

    # Tier ordering: TIMEOUT > DIFFERENT > EQUIVALENT (hardest first)
    TIER_RANK = {"TIMEOUT": 0, "DIFFERENT": 1, "EQUIVALENT": 2}

    def hardness_key(uid):
        tier, rel, turns = eval_lookup.get(uid, ("EQUIVALENT", 0.0, 0))
        return (TIER_RANK[tier], -rel, -turns)  # ascending: lower rank = harder, want earliest

    # Step 4: allocate 20 slots, even per category (50/50 split), proportional weighting.
    TOTAL = 20
    MIN_PER_CAT = 2
    cats_present = [c for c in CATS if cat_counts.get(c, 0) > 0]
    n_cats = len(cats_present)
    total_count = sum(cat_counts[c] for c in cats_present)
    # Target = nearest-even of the proportional share, with minimum 2.
    target = {c: cat_counts[c] / total_count * TOTAL for c in cats_present}
    floor_alloc = {
        c: max(MIN_PER_CAT, int(round(target[c] / 2)) * 2)
        for c in cats_present
    }
    # Cap at available count (rounded down to even)
    for c in cats_present:
        cap = (cat_counts[c] // 2) * 2
        floor_alloc[c] = min(floor_alloc[c], cap)
    # Adjust total to exactly 20 by adding/removing pairs from extremes.
    diff = TOTAL - sum(floor_alloc.values())
    while diff != 0:
        if diff > 0:
            # need more — give a pair to the category whose share is most under-allocated
            cands = sorted(
                cats_present,
                key=lambda c: target[c] - floor_alloc[c],  # most under-allocated first
                reverse=True,
            )
            placed = False
            for c in cands:
                if floor_alloc[c] + 2 <= (cat_counts[c] // 2) * 2:
                    floor_alloc[c] += 2
                    diff -= 2
                    placed = True
                    break
            if not placed:
                break
        else:
            # need fewer — remove a pair from the most over-allocated, never below MIN_PER_CAT
            cands = sorted(
                cats_present,
                key=lambda c: floor_alloc[c] - target[c],  # most over-allocated first
                reverse=True,
            )
            placed = False
            for c in cands:
                if floor_alloc[c] - 2 >= MIN_PER_CAT:
                    floor_alloc[c] -= 2
                    diff += 2
                    placed = True
                    break
            if not placed:
                break

    print("\nSlot allocation (per category, total = 20):")
    for c in CATS:
        if c in floor_alloc:
            print(f"  {c:<18} alloc={floor_alloc[c]:<3} (of {cat_counts.get(c, 0)} available)")

    # Step 5: per category, sort hardness key, take top N (hardest), then split
    train_rows = []
    val_rows = []
    for c in CATS:
        if c not in floor_alloc:
            continue
        n = floor_alloc[c]
        cat_df = df_internal[df_internal["category"] == c].copy()
        cat_df["hardness"] = cat_df["uid"].map(lambda u: hardness_key(u))
        cat_df = cat_df.sort_values("hardness").head(n)
        # n is always even — 50/50 split, hardest to train
        n_train = n // 2
        train_part = cat_df.head(n_train)
        val_part = cat_df.iloc[n_train:]
        train_rows.append(train_part)
        val_rows.append(val_part)

    train = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame()
    val = pd.concat(val_rows, ignore_index=True) if val_rows else pd.DataFrame()
    print(f"\nFinal: train={len(train)}, val={len(val)}")
    print(f"\nTrain breakdown: {train['category'].value_counts().to_dict()}")
    print(f"Val breakdown:   {val['category'].value_counts().to_dict()}")

    # Format: uid, question, ground_truth, category (matching pro6_*.csv schema)
    out_cols = ["uid", "question", "ground_truth", "category"]
    train_out = train.rename(columns={"answer": "ground_truth"})[out_cols]
    val_out = val.rename(columns={"answer": "ground_truth"})[out_cols]
    train_out.to_csv(OUT_TRAIN, index=False)
    val_out.to_csv(OUT_VAL, index=False)
    print(f"\nWrote {OUT_TRAIN}")
    print(f"Wrote {OUT_VAL}")

    print("\n=== Train (hardest first per category) ===")
    print(train_out[["uid", "category"]].to_string())
    print("\n=== Val ===")
    print(val_out[["uid", "category"]].to_string())


if __name__ == "__main__":
    asyncio.run(main())
