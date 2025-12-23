# Hodrick-Prescott Filter Theory

## Overview

The Hodrick-Prescott (HP) filter is a data-smoothing technique used to decompose a time series into trend and cyclical components. Developed by economists Robert Hodrick and Edward Prescott in the 1990s, it is widely used in macroeconomics and fiscal analysis.

## Mathematical Formulation

Given a time series {y_t} for t = 1, ..., T, the HP filter finds the trend component {τ_t} by solving:

```
min_τ Σ(y_t - τ_t)² + λ Σ((τ_{t+1} - τ_t) - (τ_t - τ_{t-1}))²
```

Where:
- First term: Penalizes deviation from the original series (goodness of fit)
- Second term: Penalizes changes in the trend's growth rate (smoothness)
- λ: Smoothing parameter controlling the trade-off

## Lambda Parameter

The smoothing parameter λ determines how smooth the trend will be:

| λ Value | Data Frequency | Interpretation |
|---------|---------------|----------------|
| 6.25 | Annual | Standard for fiscal year analysis |
| 1600 | Quarterly | Hodrick-Prescott original recommendation |
| 129600 | Monthly | Scaled from quarterly (1600 × 81) |

Higher λ produces smoother trends; lower λ allows more variation.

### Lambda Convention Warning

This skill uses the **standard λ convention** where the parameter is passed directly to the filter. Some implementations or older literature use:
- λ/100 scaling
- Different base values for the same frequency

Always verify which convention is being used when comparing results across tools.

## Implementation Details

### Statsmodels Implementation

The `statsmodels.tsa.filters.hp_filter.hpfilter` function uses sparse matrix algebra for numerical stability:

```python
from statsmodels.tsa.filters.hp_filter import hpfilter
cycle, trend = hpfilter(data, lamb=6.25)
```

### Scipy Implementation

When statsmodels is unavailable, the filter can be computed by solving:

```
(I + λK'K)τ = y
```

Where K is the second-difference matrix. This is solved efficiently using sparse matrix methods.

## Application to Fiscal Analysis

### Structural Balance

The structural (or cyclically-adjusted) balance removes cyclical fluctuations:

```
Structural Balance = Trend Receipts - Trend Outlays
```

This represents the budget balance that would occur if the economy were operating at its potential (trend) level.

### Interpretation

- **Actual Balance < Structural Balance**: Economy is underperforming; automatic stabilizers increase deficit
- **Actual Balance > Structural Balance**: Economy is overperforming; automatic stabilizers improve balance
- **Gap (Actual - Structural)**: Measures cyclical component of the budget balance

## Numerical Precision Considerations

Different HP filter implementations may produce slightly different results. Understanding these differences is critical for accurate fiscal analysis.

### Sources of Numerical Discrepancy

1. **Boundary Conditions**
   - How endpoints of the series are treated varies by implementation
   - Some use zero-padding, others extrapolate
   - Edge effects are most pronounced in first/last 1-2 observations

2. **Matrix Inversion Methods**
   - Direct vs iterative solvers
   - Sparse vs dense matrix representations
   - Pivoting strategies in LU decomposition

3. **Floating-Point Precision**
   - IEEE 754 double precision (~15-16 significant digits)
   - Order of operations affects accumulation of rounding errors
   - Large values (millions/billions) may lose precision in differences

### Typical Magnitude of Discrepancies

| Scenario | Expected Difference | Impact on Balance |
|----------|--------------------|--------------------|
| Statsmodels vs Scipy | < 1e-10 | Negligible |
| Same library, different versions | < 1e-8 | Negligible |
| Different λ conventions | Up to λ/100 | Significant |
| Boundary condition differences | 0.01-0.1% of value | May be material |

### Validation Strategy

The `--validate` flag computes results using both implementations and reports:

```json
{
  "is_valid": true,
  "tolerance": 1e-6,
  "max_trend_diff": 1.2e-10,
  "max_cycle_diff": 1.2e-10
}
```

If `is_valid` is false, detailed per-element differences are provided for diagnosis.

### Best Practices

1. **Always use validation** for high-stakes fiscal analysis
2. **Document the implementation** used when reporting results
3. **Compare against known values** when possible
4. **Use consistent precision** throughout the calculation pipeline
5. **Apply Decimal arithmetic** for final balance calculations

## References

- Hodrick, R.J. and Prescott, E.C. (1997). "Postwar U.S. Business Cycles: An Empirical Investigation." Journal of Money, Credit and Banking.
- Ravn, M.O. and Uhlig, H. (2002). "On adjusting the Hodrick-Prescott filter for the frequency of observations." Review of Economics and Statistics.
- Hamilton, J.D. (2018). "Why You Should Never Use the Hodrick-Prescott Filter." Review of Economics and Statistics. (Critical perspective on HP filter limitations)
