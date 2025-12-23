---
name: hp-filter-balance
description: >
  Apply the Hodrick-Prescott filter to fiscal time series data (receipts and outlays) to compute
  structural (trend) and actual budget balances with validated numerical precision. Use this skill when:
  (1) Analyzing fiscal data to separate cyclical fluctuations from trends,
  (2) Calculating structural deficits/surpluses for budget analysis,
  (3) Comparing actual vs. potential budget positions,
  (4) Cross-validating HP filter results between implementations to catch numerical discrepancies,
  (5) Performing econometric analysis of government finances.
  The skill uses statsmodels.tsa.filters.hp_filter for verified numerical precision with optional
  scipy cross-validation.
---

# HP Filter Structural Balance Calculator

Decompose fiscal time series into trend and cyclical components to compute structural budget balances with validated numerical precision.

## Quick Start

Calculate structural balance from Treasury fiscal data:

```bash
python scripts/hp_filter.py --receipts 3462282,3266103,3249013,3316235,3523888 \
  --outlays 3854092,3687258,3982189,4001321,4146092 \
  --labels FY2019,FY2020,FY2021,FY2022,FY2023
```

## Validation Mode

Cross-validate results between statsmodels and scipy implementations to detect numerical discrepancies:

```bash
python scripts/hp_filter.py --receipts 3462,3266,3249,3316,3524 \
  --outlays 3854,3687,3982,4001,4146 --validate
```

Output includes validation section:
```json
{
  "validation": {
    "receipts": {
      "is_valid": true,
      "max_trend_diff": 1.2e-10,
      "tolerance": 1e-6
    },
    "outlays": {
      "is_valid": true,
      "max_trend_diff": 8.3e-11,
      "tolerance": 1e-6
    }
  }
}
```

## Force Specific Implementation

```bash
# Use scipy implementation explicitly
python scripts/hp_filter.py --receipts 3462,3266,3249 --outlays 3854,3687,3982 --implementation scipy

# Use statsmodels implementation explicitly
python scripts/hp_filter.py --receipts 3462,3266,3249 --outlays 3854,3687,3982 --implementation statsmodels
```

## Common Use Cases

### Analyze Treasury API Data

First fetch data with treasury-fiscal-data skill, then analyze:

```bash
# Fetch fiscal data
python treasury_api.py --dataset mts_summary --fiscal-year 2019 --fiscal-year 2020 \
  --fiscal-year 2021 --fiscal-year 2022 --fiscal-year 2023 > fiscal_data.json

# Calculate structural balance with validation
python scripts/hp_filter.py --json-input fiscal_data.json --validate
```

### Get Specific Fiscal Year

```bash
python scripts/hp_filter.py --json-input fiscal_data.json --fiscal-year 2023
```

### Custom Lambda for Quarterly Data

```bash
python scripts/hp_filter.py --receipts 1000,1050,1100,1075,1125,1150,1200,1175 \
  --outlays 1050,1100,1150,1125,1175,1200,1250,1225 \
  --lambda-param 1600
```

## Lambda Parameter Guide

| Data Frequency | Recommended Lambda | Use Case |
|----------------|-------------------|----------|
| Annual         | 6.25              | Fiscal year analysis |
| Quarterly      | 1600              | Quarterly budget data |
| Monthly        | 129600            | Monthly Treasury statements |

**Note**: The lambda parameter is passed directly to the filter (not scaled). Some older literature uses different conventions - this skill uses the standard Hodrick-Prescott convention.

## Output Fields

| Field | Description |
|-------|-------------|
| `actual.balance` | Actual receipts minus actual outlays |
| `structural_balance` | Trend receipts minus trend outlays |
| `gap` | Difference between actual and structural balance |
| `trend.receipts` | HP-filtered trend component of receipts |
| `trend.outlays` | HP-filtered trend component of outlays |
| `validation` | Cross-validation results (when --validate used) |

## Technical Notes

- Uses `statsmodels.tsa.filters.hp_filter.hpfilter` as primary implementation
- Falls back to scipy sparse matrix implementation if statsmodels unavailable
- Requires minimum 3 data points for filter computation
- All calculations use `Decimal` for exact arithmetic in final balance computations
- Input values should be in consistent units (typically millions of dollars)
- Cross-validation detects discrepancies > 1e-6 between implementations

## Numerical Precision Considerations

Different HP filter implementations may produce slightly different results due to:

1. **Boundary conditions**: How endpoints of the series are treated
2. **Solver precision**: Floating-point arithmetic in matrix operations
3. **Sparse vs dense matrices**: Representation affecting precision

Use `--validate` to detect such discrepancies. If validation fails, results from both implementations are provided for comparison.

## Reference

See [references/hp_filter_theory.md](references/hp_filter_theory.md) for mathematical background on the Hodrick-Prescott filter and numerical precision considerations.
