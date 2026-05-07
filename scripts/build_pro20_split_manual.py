"""Build pro20 split using hand-classified categories (no LLM judge).

Manual classification rules applied:
1. multi_bulletin: explicitly enumerates many distinct bulletin issues
2. statistical: variance, std dev, R-squared, correlation, regression, percentile, geometric mean,
   CAGR, ES/VaR, ARIMA, HP filter, Box-Cox, Hill, Pareto/Zipf, Gini, HHI, MAD, etc.
3. visual_reasoning: explicit chart/graph/figure reference
4. table_parsing: non-trivial multi-column/row table navigation
5. aggregation: sum/avg/total/max/min/count over multiple rows/periods
6. simple_lookup: direct value extraction or trivial math (diff, ratio, log of ratio)
"""
import pickle
from pathlib import Path

import pandas as pd

DS = "/Users/dastin/dev/officeqa/officeqa_pro.csv"
EVAL_PKL = "/Users/dastin/dev/EvoSkill/results/eval_pro133_swap_iter1.pkl"
OUT_TRAIN = "/Users/dastin/dev/EvoSkill/.dataset/pro20_train.csv"
OUT_VAL = "/Users/dastin/dev/EvoSkill/.dataset/pro20_val.csv"

# Hand-classification of all 118 internal questions (118 - 2 ext = 116; -2 more excluded below = 114)
CATEGORIES = {
    # multi_bulletin (5)
    "UID0057": "multi_bulletin",  # Jan 1969-1980, 12 distinct bulletins
    "UID0149": "multi_bulletin",  # Jan 1962 + Jan 1963 surveys
    "UID0150": "multi_bulletin",  # 4 distinct yearly bulletins 1972-1975
    "UID0154": "multi_bulletin",  # 2 specific bulletins covering 24 months
    "UID0183": "multi_bulletin",  # 4 distinct ownership surveys 1964-1966

    # statistical (48)
    "UID0007": "statistical",   # geometric mean
    "UID0013": "statistical",   # OLS linear regression
    "UID0015": "statistical",   # Box-Cox
    "UID0018": "statistical",   # geometric mean
    "UID0022": "statistical",   # linear regression
    "UID0042": "statistical",   # Zipf exponent
    "UID0049": "statistical",   # coefficient of variation
    "UID0053": "statistical",   # CV difference
    "UID0059": "statistical",   # CAGR
    "UID0069": "statistical",   # expected shortfall
    "UID0071": "statistical",   # population std dev
    "UID0073": "statistical",   # population std dev
    "UID0083": "statistical",   # population std dev
    "UID0084": "statistical",   # OLS linear trend
    "UID0094": "statistical",   # geometric annual rate
    "UID0096": "statistical",   # centered moving average
    "UID0100": "statistical",   # OLS regression of ln
    "UID0101": "statistical",   # CAGR + arc elasticity
    "UID0102": "statistical",   # H Spread (percentile)
    "UID0103": "statistical",   # correlation + partial correlation
    "UID0108": "statistical",   # Mean Absolute Deviation
    "UID0109": "statistical",   # 85th Hazen Percentile
    "UID0110": "statistical",   # geometric mean
    "UID0111": "statistical",   # Hodrick-Prescott filter
    "UID0112": "statistical",   # R-squared
    "UID0114": "statistical",   # linear regression + predict
    "UID0117": "statistical",   # sample std dev
    "UID0123": "statistical",   # Pearson correlation
    "UID0136": "statistical",   # geometric mean
    "UID0140": "statistical",   # cubic regression
    "UID0147": "statistical",   # OLS regression
    "UID0155": "statistical",   # geometric mean
    "UID0165": "statistical",   # lower-tail portfolio loss (VaR)
    "UID0168": "statistical",   # quadratic regression
    "UID0174": "statistical",   # arc elasticity
    "UID0175": "statistical",   # variance
    "UID0179": "statistical",   # Pearson correlation
    "UID0187": "statistical",   # Geometric mean
    "UID0194": "statistical",   # CAGR
    "UID0195": "statistical",   # OLS regression + forecast
    "UID0201": "statistical",   # Gini coefficient
    "UID0205": "statistical",   # Pearson correlation
    "UID0212": "statistical",   # continuously-compounded CAGR
    "UID0215": "statistical",   # linear trend
    "UID0222": "statistical",   # CAGR projection
    "UID0226": "statistical",   # standard deviation
    "UID0230": "statistical",   # annualized realized volatility
    "UID0245": "statistical",   # Fisher Ideal symmetric growth rate

    # visual_reasoning (3)
    "UID0030": "visual_reasoning",  # local maxima on line plots (page 5)
    "UID0031": "visual_reasoning",  # Chart TF-G crossover year
    "UID0037": "visual_reasoning",  # payroll employment chart

    # table_parsing (3)
    "UID0035": "table_parsing",   # leading-digit count in tables on PDF page 41
    "UID0121": "table_parsing",   # investor-type categories (multi-column investor table)
    "UID0170": "table_parsing",   # 16 NYC Banks + 14 Chicago Banks (multi-row bank table)

    # aggregation (24)
    "UID0003": "aggregation",   # sum across calendar months 1953
    "UID0004": "aggregation",   # sum across two years
    "UID0009": "aggregation",   # weighted average across denominations
    "UID0012": "aggregation",   # max across departments
    "UID0027": "aggregation",   # max month over 1960-1969 yield spread
    "UID0028": "aggregation",   # min month over 1960-1969 yield spread
    "UID0029": "aggregation",   # avg yield spread across many months
    "UID0032": "aggregation",   # sum across months in two categories
    "UID0039": "aggregation",   # sum across many agencies
    "UID0050": "aggregation",   # total par amount summing issues
    "UID0056": "aggregation",   # max year over 1950-1990 saving rates
    "UID0068": "aggregation",   # min over rents/royalties categories
    "UID0134": "aggregation",   # range over each span + max
    "UID0135": "aggregation",   # arithmetic mean over years
    "UID0148": "aggregation",   # count + sum over 3 years of bills
    "UID0172": "aggregation",   # sum across June 2000/2001/2002
    "UID0180": "aggregation",   # sum over fiscal years 2005-2009
    "UID0182": "aggregation",   # max share over 3 years
    "UID0184": "aggregation",   # arithmetic mean over debentures
    "UID0211": "aggregation",   # arithmetic mean over 5 years
    "UID0216": "aggregation",   # mean of ratios over 3 years
    "UID0224": "aggregation",   # average over 3 months
    "UID0227": "aggregation",   # average over 3 months (Q3 1982)
    "UID0238": "aggregation",   # sum over filtered maturities

    # simple_lookup (31)
    "UID0001": "simple_lookup",   # single-year value
    "UID0017": "simple_lookup",   # 2 lookups
    "UID0019": "simple_lookup",   # diff between two values
    "UID0025": "simple_lookup",   # diff between two years
    "UID0044": "simple_lookup",   # capital flow over date range
    "UID0055": "simple_lookup",   # change between two values
    "UID0058": "simple_lookup",   # single value lookup
    "UID0062": "simple_lookup",   # diff
    "UID0065": "simple_lookup",   # diff between two growth rates
    "UID0077": "simple_lookup",   # diff between two periods
    "UID0085": "simple_lookup",   # diff between two years
    "UID0086": "simple_lookup",   # diff QoQ
    "UID0093": "simple_lookup",   # single value lookup
    "UID0097": "simple_lookup",   # 2 lookups + diff
    "UID0113": "simple_lookup",   # diff between two values
    "UID0124": "simple_lookup",   # diff in share + diff in absolute
    "UID0130": "simple_lookup",   # weighted avg of 2 values, trivial
    "UID0133": "simple_lookup",   # log of ratio (trivial)
    "UID0169": "simple_lookup",   # sum of 2 specific values
    "UID0188": "simple_lookup",   # algebraic conversion (silver oz × rate)
    "UID0190": "simple_lookup",   # diff between two months
    "UID0193": "simple_lookup",   # multi-step extrapolation
    "UID0203": "simple_lookup",   # ratio over fiscal years
    "UID0204": "simple_lookup",   # diff between two values
    "UID0218": "simple_lookup",   # ratio of redemptions
    "UID0220": "simple_lookup",   # diff between two values
    "UID0225": "simple_lookup",   # avg of 2 values, trivial
    "UID0228": "simple_lookup",   # sum of 2 months + currency
    "UID0233": "simple_lookup",   # log of ratio (trivial)
    "UID0237": "simple_lookup",   # mid-point normalized diff (defined formula)
    "UID0240": "simple_lookup",   # Macaulay duration algebra
    "UID0244": "simple_lookup",   # diff + currency
    "UID0246": "simple_lookup",   # diff between two values
}

# Excluded (external data — beyond the regex catch)
EXCLUDED_EXTERNAL = {"UID0036", "UID0196"}

CATS = ["simple_lookup", "aggregation", "statistical", "visual_reasoning", "table_parsing", "multi_bulletin"]


def main():
    df = pd.read_csv(DS)
    print(f"Total pro: {len(df)}")

    # Strict filter
    df = df[~df["uid"].isin(EXCLUDED_EXTERNAL)].copy()

    # Apply categorization (only keep mapped UIDs — external_data ones via regex are absent from CATEGORIES)
    df["category"] = df["uid"].map(CATEGORIES)
    df = df[df["category"].notna()].copy()
    print(f"Internal & categorized: {len(df)}")

    cat_counts = df["category"].value_counts()
    print("\nCategory counts (manual):")
    for c in CATS:
        print(f"  {c:<18} {cat_counts.get(c, 0)}")

    # Hardness from prior 133-eval
    with open(EVAL_PKL, "rb") as f:
        eval_results = pickle.load(f)
    full = pd.read_csv(DS)
    eval_lookup = {}
    from src.loop.runner import _measure_rel_error
    for r in eval_results:
        if r.index >= len(full): continue
        uid = full.iloc[r.index]["uid"]
        if r.error or not r.trace or not r.trace.output:
            eval_lookup[uid] = ("TIMEOUT", 1.0, 9999)
            continue
        gt = str(full.iloc[r.index]["answer"]).strip().lower()
        ag = str(r.trace.output.final_answer).strip().lower()
        rel = _measure_rel_error(ag, gt)
        turns = r.trace.num_turns or 0
        tier = "EQUIVALENT" if (gt == ag or rel < 0.01) else "DIFFERENT"
        eval_lookup[uid] = (tier, rel, turns)

    TIER_RANK = {"TIMEOUT": 0, "DIFFERENT": 1, "EQUIVALENT": 2}
    def hardness_key(uid):
        tier, rel, turns = eval_lookup.get(uid, ("EQUIVALENT", 0.0, 0))
        return (TIER_RANK[tier], -rel, -turns)

    # Allocation: even per category, total=20, hand-tuned proportional
    # statistical 48 → 6, simple_lookup 31 → 4, aggregation 24 → 4,
    # multi_bulletin 5 → 2, visual_reasoning 3 → 2, table_parsing 3 → 2
    allocation = {
        "statistical": 6,
        "simple_lookup": 4,
        "aggregation": 4,
        "multi_bulletin": 2,
        "visual_reasoning": 2,
        "table_parsing": 2,
    }
    assert sum(allocation.values()) == 20

    print("\nAllocation:")
    for c in CATS:
        print(f"  {c:<18} alloc={allocation[c]:<3} (of {cat_counts.get(c, 0)})")

    train_rows, val_rows = [], []
    for c in CATS:
        n = allocation[c]
        cat_df = df[df["category"] == c].copy()
        cat_df["hardness"] = cat_df["uid"].map(hardness_key)
        cat_df = cat_df.sort_values("hardness").head(n)
        n_train = n // 2
        train_rows.append(cat_df.head(n_train))
        val_rows.append(cat_df.iloc[n_train:])

    train = pd.concat(train_rows, ignore_index=True)
    val = pd.concat(val_rows, ignore_index=True)
    print(f"\nFinal: train={len(train)}, val={len(val)}")

    out_cols = ["uid", "question", "ground_truth", "category"]
    train.rename(columns={"answer": "ground_truth"})[out_cols].to_csv(OUT_TRAIN, index=False)
    val.rename(columns={"answer": "ground_truth"})[out_cols].to_csv(OUT_VAL, index=False)
    print(f"\nWrote {OUT_TRAIN}\nWrote {OUT_VAL}")

    print("\n=== TRAIN (hardest first per category) ===")
    print(train.rename(columns={"answer": "ground_truth"})[["uid", "category", "ground_truth"]].to_string())
    print("\n=== VAL ===")
    print(val.rename(columns={"answer": "ground_truth"})[["uid", "category", "ground_truth"]].to_string())


if __name__ == "__main__":
    main()
