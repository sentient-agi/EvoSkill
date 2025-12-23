---
name: hp-balance-calculator
description: >
  Unified tool to fetch U.S. Treasury fiscal data (receipts and outlays) and apply the
  Hodrick-Prescott filter to compute actual balance, structural (trend) balance, and gap
  with validated numerical precision. Use this skill when: (1) calculating structural
  deficits/surpluses for any range of fiscal years, (2) analyzing cyclically-adjusted
  budget positions, (3) answering questions about federal budget balance trends,
  (4) comparing actual vs structural fiscal performance. Combines data fetching and
  HP filter computation in a single reproducible pipeline with consistent methodology.
---

# HP Filter Balance Calculator

Compute structural and actual federal budget balances from Treasury fiscal data with a single command.

## Quick Start

Calculate structural balance for fiscal years 2015-2024:

```bash
python scripts/hp_balance_calculator.py --start-year 2015 --end-year 2024
```

Get result for a specific fiscal year (FY2024):

```bash
python scripts/hp_balance_calculator.py --start-year 2015 --end-year 2024 --target-year 2024
```

## Key Use Cases

### Calculate Current Fiscal Year Balance

```bash
python scripts/hp_balance_calculator.py --start-year 2010 --end-year 2024 --target-year 2024
```

Output includes:
- `actual.balance`: Actual receipts minus actual outlays
- `structural_balance`: Trend receipts minus trend outlays (cyclically-adjusted)
- `gap`: Difference between actual and structural (cyclical component)

### Analyze Balance Trends Over Time

```bash
python scripts/hp_balance_calculator.py --start-year 2015 --end-year 2024
```

Returns all periods with trend decomposition and balance calculations.

### Custom Lambda for Different Analysis

```bash
# Less smooth trend (more cyclical variation)
python scripts/hp_balance_calculator.py --start-year 2015 --end-year 2024 --lambda-param 100

# Standard annual (Ravn-Uhlig recommendation)
python scripts/hp_balance_calculator.py --start-year 2015 --end-year 2024 --lambda-param 6.25
```

### Validate Numerical Precision

```bash
python scripts/hp_balance_calculator.py --start-year 2015 --end-year 2024 --validate
```

Cross-validates results between statsmodels and scipy implementations.

## Output Format

```json
{
  "methodology": {
    "hp_filter_implementation": "statsmodels",
    "filter_type": "two-sided HP filter (standard)",
    "lambda": 6.25,
    "rounding": "ROUND_HALF_UP to integer millions",
    "arithmetic": "Decimal for final balance calculations"
  },
  "data_source": {
    "name": "U.S. Treasury Fiscal Data API",
    "endpoint": "mts_table_9 (September = fiscal year end)"
  },
  "unit": "millions of dollars",
  "periods": [
    {
      "period": "FY2024",
      "actual": {"receipts": 4918746, "outlays": 6751565, "balance": -1832819},
      "trend": {"receipts": 4701548, "outlays": 6751301},
      "structural_balance": -2049753,
      "gap": 216934
    }
  ]
}
```

## Lambda Parameter Guide

| Lambda | Data Type | Use Case |
|--------|-----------|----------|
| 6.25 | Annual | Standard Ravn-Uhlig adjustment for fiscal year data |
| 100 | Annual | Less smooth trend, captures more variation |
| 1600 | Quarterly | Original Hodrick-Prescott recommendation |

## Methodology

See [references/methodology.md](references/methodology.md) for:
- HP filter mathematical formulation
- Boundary condition handling
- Numerical precision considerations
- Interpretation of structural vs actual balance

## CLI Reference

```
--start-year INT      First fiscal year to fetch
--end-year INT        Last fiscal year (inclusive)
--target-year INT     Return only this fiscal year's result
--lambda-param FLOAT  HP filter smoothing parameter (default: 6.25)
--validate            Cross-validate implementations
--implementation      Force "statsmodels" or "scipy"
--full-precision      Keep decimal precision (don't round)
--compact             Compact JSON output
```

## Direct Data Input

Skip API fetch and provide data directly:

```bash
python scripts/hp_balance_calculator.py \
  --receipts 3462282,3266103,3249013,3316235,3523888 \
  --outlays 3854092,3687258,3982189,4001321,4146092 \
  --labels FY2019,FY2020,FY2021,FY2022,FY2023
```
