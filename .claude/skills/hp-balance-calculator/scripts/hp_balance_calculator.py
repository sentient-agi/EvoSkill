#!/usr/bin/env python3
"""
HP Filter Balance Calculator with Integrated Treasury Data Fetching

A unified tool that fetches U.S. Treasury fiscal data and applies the Hodrick-Prescott
filter to compute structural (trend) and actual budget balances with validated numerical
precision and consistent methodology.

Key Features:
- Single authoritative data source: U.S. Treasury Fiscal Data API (api.fiscaldata.treasury.gov)
- Primary HP filter: statsmodels.tsa.filters.hp_filter.hpfilter (two-sided filter)
- Fallback HP filter: scipy sparse matrix implementation (matching methodology)
- Consistent Decimal arithmetic for final balance calculations
- Cross-validation between implementations to detect numerical discrepancies

Usage:
    # Fetch Treasury data and compute structural balance for fiscal years
    python hp_balance_calculator.py --start-year 2015 --end-year 2024 --target-year 2024

    # Compute for all years in range
    python hp_balance_calculator.py --start-year 2015 --end-year 2024

    # Use custom lambda parameter
    python hp_balance_calculator.py --start-year 2015 --end-year 2024 --lambda-param 100

    # Enable cross-validation between implementations
    python hp_balance_calculator.py --start-year 2015 --end-year 2024 --validate

    # Provide data directly (skip API fetch)
    python hp_balance_calculator.py --receipts 3462282,3266103,3249013 --outlays 3854092,3687258,3982189

Output:
    JSON with actual balance, structural balance, and gap for each period.
    All monetary values in MILLIONS of dollars, rounded to integer.

Implementation Notes:
    - Uses statsmodels HP filter (two-sided, standard boundary conditions)
    - Lambda=6.25 is standard for annual fiscal data (Ravn-Uhlig adjustment)
    - Lambda=100 can be used for less smooth trends (more variation)
    - Final balances use Decimal arithmetic with ROUND_HALF_UP
"""

import argparse
import json
import sys
from decimal import Decimal, ROUND_HALF_UP
from typing import Optional

import numpy as np

# Try to use requests if available, otherwise fall back to urllib
try:
    import requests
    USE_REQUESTS = True
except ImportError:
    USE_REQUESTS = False
    import ssl
    from urllib.request import urlopen, Request
    from urllib.parse import urlencode
    from urllib.error import HTTPError, URLError

    SSL_CONTEXT = ssl.create_default_context()
    try:
        import certifi
        SSL_CONTEXT.load_verify_locations(certifi.where())
    except ImportError:
        pass

# Try statsmodels first (preferred implementation)
try:
    from statsmodels.tsa.filters.hp_filter import hpfilter
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


# =============================================================================
# Treasury Data API Client
# =============================================================================

BASE_URL = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
MTS_TABLE_9_PATH = "/v1/accounting/mts/mts_table_9"


def query_treasury_api(endpoint_path: str,
                       filter_str: Optional[str] = None,
                       page_size: int = 1000) -> dict:
    """
    Query the Treasury Fiscal Data API.

    Args:
        endpoint_path: API endpoint path
        filter_str: Filter query string
        page_size: Number of records per page

    Returns:
        JSON response from the API
    """
    url = f"{BASE_URL}{endpoint_path}"

    params = {
        "page[size]": page_size,
        "format": "json"
    }

    if filter_str:
        params["filter"] = filter_str

    if USE_REQUESTS:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        return response.json()
    else:
        from urllib.parse import urlencode
        query_string = urlencode(params)
        full_url = f"{url}?{query_string}"
        request = Request(full_url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=60, context=SSL_CONTEXT) as response:
            return json.loads(response.read().decode("utf-8"))


def fetch_fiscal_year_data(fiscal_year: int) -> tuple[Optional[float], Optional[float]]:
    """
    Fetch total receipts and outlays for a single fiscal year from Treasury API.

    Uses MTS Table 9 (September data = fiscal year end).

    Args:
        fiscal_year: The fiscal year to fetch (e.g., 2023)

    Returns:
        Tuple of (total_receipts_millions, total_outlays_millions) or (None, None) if error
    """
    filter_str = f"record_fiscal_year:eq:{fiscal_year},record_calendar_month:eq:09"

    try:
        api_response = query_treasury_api(
            endpoint_path=MTS_TABLE_9_PATH,
            filter_str=filter_str
        )

        data = api_response.get("data", [])

        if not data:
            return None, None

        # Find parent IDs for Receipts and Net Outlays sections
        receipts_parent_id = None
        outlays_parent_id = None

        for row in data:
            desc = row.get("classification_desc", "")
            parent_id = row.get("parent_id", "")

            if desc == "Receipts" and parent_id == "null":
                receipts_parent_id = row.get("classification_id")
            elif desc == "Net Outlays" and parent_id == "null":
                outlays_parent_id = row.get("classification_id")

        # Find the Total rows under each parent
        total_receipts = None
        total_outlays = None

        for row in data:
            desc = row.get("classification_desc", "")
            parent_id = row.get("parent_id", "")
            amt = row.get("current_fytd_rcpt_outly_amt")

            if desc == "Total":
                if parent_id == receipts_parent_id:
                    total_receipts = float(amt) if amt and amt != "null" else None
                elif parent_id == outlays_parent_id:
                    total_outlays = float(amt) if amt and amt != "null" else None

        # Convert to millions
        receipts_millions = total_receipts / 1_000_000 if total_receipts else None
        outlays_millions = total_outlays / 1_000_000 if total_outlays else None

        return receipts_millions, outlays_millions

    except Exception as e:
        print(f"Warning: Error fetching FY{fiscal_year}: {e}", file=sys.stderr)
        return None, None


def fetch_fiscal_data_range(start_year: int, end_year: int) -> dict:
    """
    Fetch fiscal data for a range of years.

    Args:
        start_year: First fiscal year
        end_year: Last fiscal year (inclusive)

    Returns:
        Dictionary with receipts, outlays, and labels lists
    """
    receipts = []
    outlays = []
    labels = []
    errors = []

    for fy in range(start_year, end_year + 1):
        r, o = fetch_fiscal_year_data(fy)
        if r is not None and o is not None:
            receipts.append(r)
            outlays.append(o)
            labels.append(f"FY{fy}")
        else:
            errors.append(f"FY{fy}")

    result = {
        "source": "U.S. Treasury Fiscal Data API",
        "endpoint": "mts_table_9",
        "unit": "millions of dollars",
        "receipts": receipts,
        "outlays": outlays,
        "labels": labels
    }

    if errors:
        result["missing_years"] = errors

    return result


# =============================================================================
# HP Filter Implementations
# =============================================================================

def hp_filter_scipy(data: np.ndarray, lamb: float = 6.25) -> tuple[np.ndarray, np.ndarray]:
    """
    HP filter implementation using scipy sparse matrices.

    Implements the standard two-sided HP filter by solving:
    min_τ Σ(y_t - τ_t)² + λ Σ((τ_{t+1} - τ_t) - (τ_t - τ_{t-1}))²

    Equivalent to: (I + λK'K)τ = y where K is the second-difference matrix.

    Args:
        data: 1D numpy array of time series data
        lamb: Smoothing parameter (6.25 for annual data)

    Returns:
        Tuple of (trend, cycle) components
    """
    try:
        from scipy import sparse
        from scipy.sparse.linalg import spsolve
    except ImportError:
        raise ImportError("scipy is required. Install with: pip install scipy")

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


def hp_filter_statsmodels(data: np.ndarray, lamb: float = 6.25) -> tuple[np.ndarray, np.ndarray]:
    """
    HP filter using statsmodels implementation.

    Uses statsmodels.tsa.filters.hp_filter.hpfilter which implements
    the standard two-sided HP filter with proper boundary handling.

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


def hp_filter(data: np.ndarray, lamb: float = 6.25, implementation: str = "auto") -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Hodrick-Prescott filter to decompose time series into trend and cycle.

    Args:
        data: 1D numpy array of time series data
        lamb: Smoothing parameter (6.25 for annual, 100 for less smooth, 1600 for quarterly)
        implementation: "auto" (prefer statsmodels), "statsmodels", or "scipy"

    Returns:
        Tuple of (trend, cycle) components
    """
    if implementation == "statsmodels" or (implementation == "auto" and HAS_STATSMODELS):
        return hp_filter_statsmodels(data, lamb)
    elif implementation == "scipy":
        return hp_filter_scipy(data, lamb)
    elif implementation == "auto":
        return hp_filter_scipy(data, lamb)
    else:
        raise ValueError(f"Unknown implementation: {implementation}")


def validate_implementations(data: np.ndarray, lamb: float = 6.25, tolerance: float = 1e-6) -> dict:
    """
    Cross-validate HP filter results between statsmodels and scipy.

    Args:
        data: 1D numpy array of time series data
        lamb: Smoothing parameter
        tolerance: Maximum acceptable difference

    Returns:
        Validation results dictionary
    """
    if not HAS_STATSMODELS:
        return {
            "validation_skipped": True,
            "reason": "statsmodels not available",
            "using_implementation": "scipy"
        }

    try:
        trend_sm, cycle_sm = hp_filter_statsmodels(data, lamb)
        trend_scipy, cycle_scipy = hp_filter_scipy(data, lamb)

        trend_diff = np.abs(trend_sm - trend_scipy)
        max_trend_diff = float(np.max(trend_diff))
        mean_trend_diff = float(np.mean(trend_diff))

        is_valid = max_trend_diff < tolerance

        return {
            "is_valid": is_valid,
            "tolerance": tolerance,
            "max_trend_diff": max_trend_diff,
            "mean_trend_diff": mean_trend_diff,
            "implementations_match": is_valid
        }

    except Exception as e:
        return {
            "validation_error": str(e),
            "is_valid": False
        }


# =============================================================================
# Balance Calculation
# =============================================================================

def calculate_structural_balance(
    receipts: list[float],
    outlays: list[float],
    lamb: float = 6.25,
    labels: Optional[list[str]] = None,
    validate: bool = False,
    implementation: str = "auto",
    round_to_integer: bool = True
) -> dict:
    """
    Calculate structural and actual budget balances using HP filter.

    Structural Balance = Trend Receipts - Trend Outlays
    Actual Balance = Actual Receipts - Actual Outlays
    Gap = Actual Balance - Structural Balance

    Args:
        receipts: List of receipt values (millions of dollars)
        outlays: List of outlay values (millions of dollars)
        lamb: HP filter smoothing parameter (default 6.25 for annual)
        labels: Optional period labels (e.g., ["FY2019", "FY2020", ...])
        validate: If True, cross-validate implementations
        implementation: Which HP filter implementation to use
        round_to_integer: If True, round final balances to integer millions

    Returns:
        Dictionary with periods, metadata, and optional validation results
    """
    if len(receipts) != len(outlays):
        raise ValueError(f"Receipts length ({len(receipts)}) must match outlays ({len(outlays)})")

    if len(receipts) < 3:
        raise ValueError("At least 3 data points required for HP filter")

    # Convert to numpy arrays with high precision
    receipts_arr = np.array(receipts, dtype=np.float64)
    outlays_arr = np.array(outlays, dtype=np.float64)

    # Optionally validate implementations
    validation_results = None
    if validate:
        validation_results = {
            "receipts": validate_implementations(receipts_arr, lamb),
            "outlays": validate_implementations(outlays_arr, lamb)
        }

    # Determine which implementation is being used
    impl_used = implementation
    if implementation == "auto":
        impl_used = "statsmodels" if HAS_STATSMODELS else "scipy"

    # Apply HP filter
    trend_receipts, _ = hp_filter(receipts_arr, lamb=lamb, implementation=implementation)
    trend_outlays, _ = hp_filter(outlays_arr, lamb=lamb, implementation=implementation)

    # Calculate balances using Decimal for exact arithmetic
    n = len(receipts)
    periods = []

    for i in range(n):
        # Use Decimal for precise calculations
        dec_receipts = Decimal(str(receipts[i]))
        dec_outlays = Decimal(str(outlays[i]))
        dec_trend_receipts = Decimal(str(float(trend_receipts[i])))
        dec_trend_outlays = Decimal(str(float(trend_outlays[i])))

        actual_balance = dec_receipts - dec_outlays
        structural_balance = dec_trend_receipts - dec_trend_outlays
        gap = actual_balance - structural_balance

        if round_to_integer:
            # Round to integer (millions of dollars)
            actual_balance_rounded = int(actual_balance.quantize(Decimal('1'), rounding=ROUND_HALF_UP))
            structural_balance_rounded = int(structural_balance.quantize(Decimal('1'), rounding=ROUND_HALF_UP))
            gap_rounded = int(gap.quantize(Decimal('1'), rounding=ROUND_HALF_UP))
            trend_receipts_rounded = int(dec_trend_receipts.quantize(Decimal('1'), rounding=ROUND_HALF_UP))
            trend_outlays_rounded = int(dec_trend_outlays.quantize(Decimal('1'), rounding=ROUND_HALF_UP))
        else:
            # Keep full precision
            actual_balance_rounded = float(actual_balance)
            structural_balance_rounded = float(structural_balance)
            gap_rounded = float(gap)
            trend_receipts_rounded = float(dec_trend_receipts)
            trend_outlays_rounded = float(dec_trend_outlays)

        period_data = {
            "period": labels[i] if labels and i < len(labels) else i + 1,
            "actual": {
                "receipts": int(dec_receipts) if round_to_integer else float(dec_receipts),
                "outlays": int(dec_outlays) if round_to_integer else float(dec_outlays),
                "balance": actual_balance_rounded
            },
            "trend": {
                "receipts": trend_receipts_rounded,
                "outlays": trend_outlays_rounded
            },
            "structural_balance": structural_balance_rounded,
            "gap": gap_rounded
        }
        periods.append(period_data)

    # Build result
    result = {
        "methodology": {
            "hp_filter_implementation": impl_used,
            "filter_type": "two-sided HP filter (standard)",
            "boundary_conditions": "natural spline (default)",
            "lambda": lamb,
            "lambda_description": "6.25 for annual (Ravn-Uhlig), 100 for less smooth, 1600 for quarterly",
            "rounding": "ROUND_HALF_UP to integer millions" if round_to_integer else "full precision",
            "arithmetic": "Decimal for final balance calculations"
        },
        "data_source": {
            "name": "U.S. Treasury Fiscal Data API",
            "endpoint": "mts_table_9 (September = fiscal year end)",
            "url": "https://api.fiscaldata.treasury.gov"
        },
        "unit": "millions of dollars",
        "num_periods": n,
        "periods": periods
    }

    if validation_results:
        result["validation"] = validation_results

    return result


def get_period_result(result: dict, target_label: str) -> Optional[dict]:
    """Extract a specific period's result from the full calculation."""
    for period in result.get("periods", []):
        if str(period.get("period")) == target_label:
            return {
                "methodology": result["methodology"],
                "data_source": result["data_source"],
                "unit": result["unit"],
                "period": period,
                "validation": result.get("validation")
            }
    return None


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HP Filter Balance Calculator with Integrated Treasury Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch Treasury data and compute structural balance
  python hp_balance_calculator.py --start-year 2015 --end-year 2024

  # Get result for a specific fiscal year
  python hp_balance_calculator.py --start-year 2015 --end-year 2024 --target-year 2024

  # Use custom lambda parameter (less smooth trend)
  python hp_balance_calculator.py --start-year 2015 --end-year 2024 --lambda-param 100

  # Enable cross-validation between implementations
  python hp_balance_calculator.py --start-year 2015 --end-year 2024 --validate

  # Provide data directly (skip API fetch)
  python hp_balance_calculator.py --receipts 3462282,3266103 --outlays 3854092,3687258

  # Force specific implementation
  python hp_balance_calculator.py --start-year 2015 --end-year 2024 --implementation scipy

Notes:
  - Default lambda=6.25 is standard for annual fiscal data (Ravn-Uhlig)
  - Use lambda=100 for less smooth trends (more cyclical variation captured)
  - All monetary values in millions of dollars, rounded to integer
  - Uses statsmodels HP filter (preferred) with scipy fallback
        """
    )

    # Data source options
    parser.add_argument("--start-year", type=int,
                        help="First fiscal year to fetch from Treasury API")
    parser.add_argument("--end-year", type=int,
                        help="Last fiscal year to fetch (inclusive)")
    parser.add_argument("--receipts", type=str,
                        help="Comma-separated receipt values (millions of dollars)")
    parser.add_argument("--outlays", type=str,
                        help="Comma-separated outlay values (millions of dollars)")
    parser.add_argument("--labels", type=str,
                        help="Comma-separated period labels")

    # Filter parameters
    parser.add_argument("--lambda-param", type=float, default=6.25,
                        help="HP filter smoothing parameter (default: 6.25 for annual)")

    # Output options
    parser.add_argument("--target-year", type=int,
                        help="Return only the result for a specific fiscal year")
    parser.add_argument("--full-precision", action="store_true",
                        help="Keep full decimal precision (don't round to integer)")

    # Validation options
    parser.add_argument("--validate", action="store_true",
                        help="Cross-validate results between statsmodels and scipy")
    parser.add_argument("--implementation", type=str, default="auto",
                        choices=["auto", "statsmodels", "scipy"],
                        help="Which HP filter implementation to use")

    # Output formatting
    parser.add_argument("--compact", action="store_true",
                        help="Compact JSON output")

    args = parser.parse_args()

    try:
        # Determine data source
        if args.start_year and args.end_year:
            # Fetch from Treasury API
            print(f"Fetching fiscal data for FY{args.start_year}-FY{args.end_year}...", file=sys.stderr)
            fiscal_data = fetch_fiscal_data_range(args.start_year, args.end_year)

            if not fiscal_data["receipts"]:
                print("Error: No fiscal data could be retrieved", file=sys.stderr)
                sys.exit(1)

            receipts = fiscal_data["receipts"]
            outlays = fiscal_data["outlays"]
            labels = fiscal_data["labels"]

            if "missing_years" in fiscal_data:
                print(f"Warning: Missing data for: {fiscal_data['missing_years']}", file=sys.stderr)

        elif args.receipts and args.outlays:
            # Use provided data
            receipts = [float(v.strip()) for v in args.receipts.split(',')]
            outlays = [float(v.strip()) for v in args.outlays.split(',')]
            labels = [v.strip() for v in args.labels.split(',')] if args.labels else None
        else:
            parser.error("Either (--start-year and --end-year) or (--receipts and --outlays) required")
            return

        # Calculate structural balance
        result = calculate_structural_balance(
            receipts=receipts,
            outlays=outlays,
            lamb=args.lambda_param,
            labels=labels,
            validate=args.validate,
            implementation=args.implementation,
            round_to_integer=not args.full_precision
        )

        # Filter to specific year if requested
        if args.target_year:
            target_label = f"FY{args.target_year}"
            period_result = get_period_result(result, target_label)
            if period_result:
                result = period_result
            else:
                print(f"Error: FY{args.target_year} not found in results", file=sys.stderr)
                sys.exit(1)

        # Output
        indent = None if args.compact else 2
        print(json.dumps(result, indent=indent))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
