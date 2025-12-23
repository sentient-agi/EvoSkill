# HP Filter Balance Methodology

## Overview

This tool computes structural (cyclically-adjusted) budget balances using the Hodrick-Prescott filter to decompose fiscal time series into trend and cyclical components.

## Mathematical Formulation

The HP filter solves:

```
min_τ Σ(y_t - τ_t)² + λ Σ((τ_{t+1} - τ_t) - (τ_t - τ_{t-1}))²
```

Where:
- `y_t` = observed fiscal data (receipts or outlays)
- `τ_t` = trend component
- `λ` = smoothing parameter

The solution is equivalent to solving the linear system:

```
(I + λK'K)τ = y
```

Where K is the (n-2) × n second-difference matrix.

## Implementation Details

### Primary: statsmodels

```python
from statsmodels.tsa.filters.hp_filter import hpfilter
cycle, trend = hpfilter(data, lamb=6.25)
```

Properties:
- Two-sided filter (uses all observations)
- Natural spline boundary conditions
- Sparse matrix algebra for numerical stability
- IEEE 754 double precision (~15-16 significant digits)

### Fallback: scipy

When statsmodels is unavailable, the filter is computed using scipy sparse matrices:

```python
from scipy import sparse
from scipy.sparse.linalg import spsolve

K = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n-2, n))
A = sparse.eye(n) + lamb * K.T @ K
trend = spsolve(A, data)
```

Properties:
- Mathematically identical formulation
- CSC sparse matrix format for efficient solving
- LU decomposition via SuperLU

### Cross-Validation

When `--validate` is used, both implementations are run and compared:
- Maximum trend difference < 1e-6 indicates implementations match
- Differences typically < 1e-10 in practice

## Balance Calculations

### Structural Balance

```
Structural Balance = Trend Receipts - Trend Outlays
```

Represents the budget balance if the economy were at its potential (trend) level.

### Actual Balance

```
Actual Balance = Actual Receipts - Actual Outlays
```

The observed budget balance.

### Gap (Cyclical Component)

```
Gap = Actual Balance - Structural Balance
```

Interpretation:
- Gap > 0: Economy overperforming (cyclical surplus)
- Gap < 0: Economy underperforming (cyclical deficit)

## Numerical Precision

### Sources of Potential Discrepancy

1. **Floating-point arithmetic**: IEEE 754 double precision limits
2. **Sparse solver precision**: LU decomposition rounding
3. **Order of operations**: Accumulation of rounding errors

### Mitigation Strategy

1. HP filter computed in float64 (maximum native precision)
2. Final balance calculations use Python `Decimal` with exact arithmetic
3. Rounding applied only at final step using `ROUND_HALF_UP`
4. Cross-validation available to detect implementation differences

### Expected Precision

| Comparison | Expected Difference |
|------------|---------------------|
| statsmodels vs scipy | < 1e-10 |
| Same implementation, same data | 0 |
| Different λ | Significant (intentional) |

## Lambda Parameter

| Value | Frequency | Source |
|-------|-----------|--------|
| 6.25 | Annual | Ravn-Uhlig (2002) |
| 100 | Annual | Less smooth alternative |
| 1600 | Quarterly | Hodrick-Prescott (1997) |
| 129600 | Monthly | 1600 × 81 scaling |

The λ parameter directly controls the smoothness/variability trade-off:
- Higher λ → smoother trend → more attributed to cycle
- Lower λ → more variable trend → less attributed to cycle

## Data Source

### Treasury Fiscal Data API

- **Base URL**: `https://api.fiscaldata.treasury.gov`
- **Endpoint**: `/v1/accounting/mts/mts_table_9`
- **Filter**: September data (fiscal year end)

Data retrieval:
1. Query MTS Table 9 for September of each fiscal year
2. Find "Receipts" and "Net Outlays" parent classifications
3. Extract "Total" rows under each parent
4. Convert from dollars to millions

### Fiscal Year Definition

- FY runs October 1 to September 30
- FY2024 = October 1, 2023 through September 30, 2024
- September data contains full fiscal year totals

## References

- Hodrick, R.J. and Prescott, E.C. (1997). "Postwar U.S. Business Cycles: An Empirical Investigation."
- Ravn, M.O. and Uhlig, H. (2002). "On adjusting the Hodrick-Prescott filter for the frequency of observations."
- statsmodels documentation: https://www.statsmodels.org/stable/generated/statsmodels.tsa.filters.hp_filter.hpfilter.html
