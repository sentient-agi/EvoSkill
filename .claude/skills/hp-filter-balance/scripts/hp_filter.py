#!/usr/bin/env python3
"""
HP Filter Structural Balance Calculator with Validation

Implements the Hodrick-Prescott filter for fiscal data analysis to compute
structural (trend) and actual budget balances. Uses statsmodels for verified
numerical implementation with optional cross-validation against scipy.

Usage:
    python hp_filter.py --receipts 4439283,4881582,4048000 --outlays 6134432,6821914,5872000
    python hp_filter.py --receipts-file receipts.csv --outlays-file outlays.csv
    python hp_filter.py --json-input data.json --fiscal-year 2023
    python hp_filter.py --validate  # Cross-validate implementations
    python hp_filter.py --help

Examples:
    # Direct input with comma-separated values (in millions of dollars)
    python hp_filter.py --receipts 3462,3266,3249,3316,3524 --outlays 3854,3687,3982,4001,4146

    # With validation to detect numerical discrepancies
    python hp_filter.py --receipts 3462,3266,3249,3316,3524 --outlays 3854,3687,3982,4001,4146 --validate

    # With custom lambda parameter
    python hp_filter.py --receipts 3462,3266,3249 --outlays 3854,3687,3982 --lambda-param 100

    # Input from JSON file (e.g., Treasury API output)
    python hp_filter.py --json-input treasury_data.json

    # Get only specific fiscal year result
    python hp_filter.py --receipts 3462,3266,3249,3316,3524 --outlays 3854,3687,3982,4001,4146 --fiscal-year 2023

Output:
    JSON with trend components and structural/actual balances for each period.
"""

import argparse
import json
import sys
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

import numpy as np

try:
    from statsmodels.tsa.filters.hp_filter import hpfilter
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


def hp_filter_scipy(data: np.ndarray, lamb: float = 1600) -> tuple[np.ndarray, np.ndarray]:
    """
    HP filter implementation using scipy sparse matrices.

    This implements the HP filter by solving the optimization problem:
    min_τ Σ(y_t - τ_t)² + λ Σ((τ_{t+1} - τ_t) - (τ_t - τ_{t-1}))²

    Args:
        data: 1D numpy array of time series data
        lamb: Smoothing parameter (lambda). Common values:
              - 1600 for quarterly data
              - 6.25 for annual data
              - 129600 for monthly data

    Returns:
        Tuple of (trend, cycle) components
    """
    try:
        from scipy import sparse
        from scipy.sparse.linalg import spsolve
    except ImportError:
        raise ImportError("scipy is required for HP filter. Install with: pip install scipy")

    n = len(data)

    # Build the second difference matrix K
    # K is (n-2) x n matrix where K[i] = [0, ..., 1, -2, 1, ..., 0]
    diags = np.array([1, -2, 1])
    offsets = np.array([0, 1, 2])
    K = sparse.diags(diags, offsets, shape=(n - 2, n), format='csc')

    # The trend is solution to: (I + λ K'K) τ = y
    I = sparse.eye(n, format='csc')
    A = I + lamb * K.T @ K

    trend = spsolve(A, data)
    cycle = data - trend

    return trend, cycle


def hp_filter_statsmodels(data: np.ndarray, lamb: float = 1600) -> tuple[np.ndarray, np.ndarray]:
    """
    HP filter using statsmodels implementation.

    Args:
        data: 1D numpy array of time series data
        lamb: Smoothing parameter

    Returns:
        Tuple of (trend, cycle) components
    """
    if not HAS_STATSMODELS:
        raise ImportError("statsmodels is required. Install with: pip install statsmodels")

    cycle, trend = hpfilter(data, lamb=lamb)
    return np.asarray(trend), np.asarray(cycle)


def hp_filter(data: np.ndarray, lamb: float = 1600, implementation: str = "auto") -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Hodrick-Prescott filter to decompose time series into trend and cycle.

    Uses statsmodels.tsa.filters.hp_filter.hpfilter for verified numerical
    precision. Falls back to scipy-based implementation if statsmodels unavailable.

    Args:
        data: 1D numpy array of time series data
        lamb: Smoothing parameter (lambda). Recommended values:
              - 6.25 for annual data (most fiscal data)
              - 1600 for quarterly data
              - 129600 for monthly data
        implementation: "auto" (default), "statsmodels", or "scipy"

    Returns:
        Tuple of (trend, cycle) components as numpy arrays
    """
    if implementation == "statsmodels" or (implementation == "auto" and HAS_STATSMODELS):
        return hp_filter_statsmodels(data, lamb)
    elif implementation == "scipy":
        return hp_filter_scipy(data, lamb)
    elif implementation == "auto":
        return hp_filter_scipy(data, lamb)
    else:
        raise ValueError(f"Unknown implementation: {implementation}. Use 'auto', 'statsmodels', or 'scipy'")


def validate_implementations(data: np.ndarray, lamb: float = 6.25, tolerance: float = 1e-6) -> dict:
    """
    Cross-validate HP filter results between statsmodels and scipy implementations.

    This function catches numerical discrepancies that can arise from:
    - Different boundary condition handling
    - Matrix inversion precision differences
    - Sparse vs dense matrix operations

    Args:
        data: 1D numpy array of time series data
        lamb: Smoothing parameter
        tolerance: Maximum acceptable difference between implementations

    Returns:
        Dictionary with validation results including:
        - max_trend_diff: Maximum absolute difference in trend values
        - max_cycle_diff: Maximum absolute difference in cycle values
        - is_valid: Boolean indicating if differences are within tolerance
        - discrepancy_details: Per-element differences if issues found
    """
    if not HAS_STATSMODELS:
        return {
            "validation_skipped": True,
            "reason": "statsmodels not available for cross-validation",
            "using_implementation": "scipy"
        }

    try:
        trend_sm, cycle_sm = hp_filter_statsmodels(data, lamb)
        trend_scipy, cycle_scipy = hp_filter_scipy(data, lamb)

        trend_diff = np.abs(trend_sm - trend_scipy)
        cycle_diff = np.abs(cycle_sm - cycle_scipy)

        max_trend_diff = float(np.max(trend_diff))
        max_cycle_diff = float(np.max(cycle_diff))
        mean_trend_diff = float(np.mean(trend_diff))
        mean_cycle_diff = float(np.mean(cycle_diff))

        is_valid = max_trend_diff < tolerance and max_cycle_diff < tolerance

        result = {
            "is_valid": is_valid,
            "tolerance": tolerance,
            "max_trend_diff": max_trend_diff,
            "max_cycle_diff": max_cycle_diff,
            "mean_trend_diff": mean_trend_diff,
            "mean_cycle_diff": mean_cycle_diff,
            "statsmodels_version": "available",
            "scipy_version": "available"
        }

        if not is_valid:
            # Provide detailed discrepancy information
            result["discrepancy_details"] = {
                "trend_differences": trend_diff.tolist(),
                "cycle_differences": cycle_diff.tolist(),
                "warning": "Implementations produced different results beyond tolerance"
            }

        return result

    except Exception as e:
        return {
            "validation_error": str(e),
            "is_valid": False
        }


def calculate_structural_balance(
    receipts: list[float],
    outlays: list[float],
    lamb: float = 6.25,
    labels: Optional[list[str]] = None,
    precision: int = 6,
    validate: bool = False,
    implementation: str = "auto"
) -> dict:
    """
    Calculate structural and actual budget balances using HP filter.

    The structural balance is computed as:
        structural_balance = trend_receipts - trend_outlays

    The actual balance is:
        actual_balance = actual_receipts - actual_outlays

    The gap measures how much the actual balance deviates from structural:
        gap = actual_balance - structural_balance

    Args:
        receipts: List of receipt values for each period
        outlays: List of outlay values for each period
        lamb: HP filter smoothing parameter (default 6.25 for annual data)
        labels: Optional list of period labels (e.g., fiscal years)
        precision: Decimal precision for calculations (default 6)
        validate: If True, cross-validate implementations and include results
        implementation: Which implementation to use ("auto", "statsmodels", "scipy")

    Returns:
        Dictionary containing:
        - periods: List of period results with all balance calculations
        - metadata: Information about the calculation parameters
        - summary: Aggregate statistics
        - validation: Cross-validation results (if validate=True)
    """
    if len(receipts) != len(outlays):
        raise ValueError(f"Receipts length ({len(receipts)}) must match outlays length ({len(outlays)})")

    if len(receipts) < 3:
        raise ValueError("At least 3 data points required for HP filter")

    # Convert to numpy arrays with high precision
    receipts_arr = np.array(receipts, dtype=np.float64)
    outlays_arr = np.array(outlays, dtype=np.float64)

    # Optionally validate implementations before proceeding
    validation_results = None
    if validate:
        validation_results = {
            "receipts": validate_implementations(receipts_arr, lamb),
            "outlays": validate_implementations(outlays_arr, lamb)
        }

    # Determine which implementation to use
    impl_used = implementation
    if implementation == "auto":
        impl_used = "statsmodels" if HAS_STATSMODELS else "scipy"

    # Apply HP filter to both series
    trend_receipts, cycle_receipts = hp_filter(receipts_arr, lamb=lamb, implementation=implementation)
    trend_outlays, cycle_outlays = hp_filter(outlays_arr, lamb=lamb, implementation=implementation)

    # Calculate balances with exact arithmetic using Decimal for final values
    n = len(receipts)
    periods = []

    for i in range(n):
        # Use Decimal for precise final calculations
        dec_receipts = Decimal(str(receipts[i]))
        dec_outlays = Decimal(str(outlays[i]))
        dec_trend_receipts = Decimal(str(float(trend_receipts[i])))
        dec_trend_outlays = Decimal(str(float(trend_outlays[i])))

        actual_balance = dec_receipts - dec_outlays
        structural_balance = dec_trend_receipts - dec_trend_outlays
        gap = actual_balance - structural_balance

        # Round to specified precision
        quantize_str = '0.' + '0' * precision

        period_data = {
            "period": labels[i] if labels and i < len(labels) else i + 1,
            "actual": {
                "receipts": float(dec_receipts),
                "outlays": float(dec_outlays),
                "balance": float(actual_balance.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP))
            },
            "trend": {
                "receipts": float(dec_trend_receipts.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)),
                "outlays": float(dec_trend_outlays.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)),
            },
            "structural_balance": float(structural_balance.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)),
            "gap": float(gap.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP))
        }
        periods.append(period_data)

    # Build result
    result = {
        "hp_filter_implementation": impl_used,
        "metadata": {
            "lambda": lamb,
            "lambda_description": "6.25 for annual, 1600 for quarterly, 129600 for monthly",
            "num_periods": n,
            "precision_digits": precision,
            "unit": "same as input (typically millions of dollars)"
        },
        "periods": periods,
        "summary": {
            "avg_structural_balance": float(np.mean([p["structural_balance"] for p in periods])),
            "avg_actual_balance": float(np.mean([p["actual"]["balance"] for p in periods])),
            "avg_gap": float(np.mean([p["gap"] for p in periods])),
            "total_structural_balance": float(sum(p["structural_balance"] for p in periods)),
            "total_actual_balance": float(sum(p["actual"]["balance"] for p in periods))
        }
    }

    if validation_results:
        result["validation"] = validation_results

    return result


def parse_values(value_str: str) -> list[float]:
    """Parse comma-separated values into list of floats."""
    return [float(v.strip()) for v in value_str.split(',')]


def parse_labels(label_str: str) -> list[str]:
    """Parse comma-separated labels into list of strings."""
    return [v.strip() for v in label_str.split(',')]


def load_json_treasury_data(filepath: str) -> tuple[list[float], list[float], list[str]]:
    """
    Load fiscal data from JSON file (e.g., Treasury API output).

    Expected format:
    {
        "fiscal_years": {
            "2019": {"total_receipts_millions": ..., "total_outlays_millions": ...},
            "2020": {...},
            ...
        }
    }

    Returns:
        Tuple of (receipts, outlays, fiscal_year_labels)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    if "fiscal_years" in data:
        # Treasury API format
        receipts = []
        outlays = []
        labels = []

        for fy in sorted(data["fiscal_years"].keys()):
            fy_data = data["fiscal_years"][fy]
            receipts.append(fy_data.get("total_receipts_millions") or fy_data.get("total_receipts"))
            outlays.append(fy_data.get("total_outlays_millions") or fy_data.get("total_outlays"))
            labels.append(f"FY{fy}")

        return receipts, outlays, labels

    elif "periods" in data:
        # Array format
        receipts = [p["receipts"] for p in data["periods"]]
        outlays = [p["outlays"] for p in data["periods"]]
        labels = [p.get("label", str(i+1)) for i, p in enumerate(data["periods"])]
        return receipts, outlays, labels

    else:
        raise ValueError("Unrecognized JSON format. Expected 'fiscal_years' or 'periods' key.")


def main():
    parser = argparse.ArgumentParser(
        description="HP Filter Structural Balance Calculator with Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Direct values (in millions of dollars)
  python hp_filter.py --receipts 3462,3266,3249,3316,3524 --outlays 3854,3687,3982,4001,4146

  # With validation to catch numerical discrepancies
  python hp_filter.py --receipts 3462,3266,3249,3316,3524 --outlays 3854,3687,3982,4001,4146 --validate

  # Force specific implementation
  python hp_filter.py --receipts 3462,3266,3249,3316,3524 --outlays 3854,3687,3982,4001,4146 --implementation scipy

  # With fiscal year labels
  python hp_filter.py --receipts 3462,3266,3249,3316,3524 --outlays 3854,3687,3982,4001,4146 \\
    --labels FY2019,FY2020,FY2021,FY2022,FY2023

  # From Treasury API JSON output
  python hp_filter.py --json-input treasury_data.json

  # Get specific fiscal year
  python hp_filter.py --json-input treasury_data.json --fiscal-year 2023

  # Custom lambda for quarterly data
  python hp_filter.py --receipts 1000,1100,1050,1200 --outlays 1050,1150,1100,1250 --lambda-param 1600

Notes:
  - Default lambda=6.25 is optimal for annual fiscal data
  - Use lambda=1600 for quarterly data, 129600 for monthly
  - All monetary values should be in consistent units (typically millions)
  - At least 3 data points are required for the HP filter
  - Use --validate to cross-check results between implementations
        """
    )

    # Input options
    parser.add_argument("--receipts", type=str,
                        help="Comma-separated receipt values")
    parser.add_argument("--outlays", type=str,
                        help="Comma-separated outlay values")
    parser.add_argument("--labels", type=str,
                        help="Comma-separated period labels (e.g., FY2019,FY2020)")
    parser.add_argument("--json-input", type=str,
                        help="Path to JSON file with fiscal data")

    # Filter parameters
    parser.add_argument("--lambda-param", type=float, default=6.25,
                        help="HP filter smoothing parameter (default: 6.25 for annual)")
    parser.add_argument("--precision", type=int, default=6,
                        help="Decimal precision for output (default: 6)")

    # Validation options
    parser.add_argument("--validate", action="store_true",
                        help="Cross-validate results between statsmodels and scipy implementations")
    parser.add_argument("--implementation", type=str, default="auto",
                        choices=["auto", "statsmodels", "scipy"],
                        help="Which HP filter implementation to use (default: auto)")

    # Output options
    parser.add_argument("--fiscal-year", type=int,
                        help="Return only the result for a specific fiscal year")
    parser.add_argument("--compact", action="store_true",
                        help="Compact JSON output")

    args = parser.parse_args()

    # Parse input data
    try:
        if args.json_input:
            receipts, outlays, labels = load_json_treasury_data(args.json_input)
        elif args.receipts and args.outlays:
            receipts = parse_values(args.receipts)
            outlays = parse_values(args.outlays)
            labels = parse_labels(args.labels) if args.labels else None
        else:
            parser.error("Either --json-input or both --receipts and --outlays are required")
            return

        # Calculate structural balance
        result = calculate_structural_balance(
            receipts=receipts,
            outlays=outlays,
            lamb=args.lambda_param,
            labels=labels,
            precision=args.precision,
            validate=args.validate,
            implementation=args.implementation
        )

        # Filter to specific fiscal year if requested
        if args.fiscal_year:
            fy_label = f"FY{args.fiscal_year}"
            matching = [p for p in result["periods"] if str(p["period"]) == fy_label or str(p["period"]) == str(args.fiscal_year)]
            if matching:
                result = {
                    "hp_filter_implementation": result["hp_filter_implementation"],
                    "metadata": result["metadata"],
                    "period": matching[0]
                }
                if "validation" in result:
                    result["validation"] = result["validation"]
            else:
                print(f"Error: Fiscal year {args.fiscal_year} not found in data", file=sys.stderr)
                sys.exit(1)

        # Output
        indent = None if args.compact else 2
        print(json.dumps(result, indent=indent))

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
