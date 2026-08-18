"""Tests for the staged SSB v2 scorer corrections.

The regression cases use the ACTUAL values observed in the 0801 evolution,
read out of the retained golden and solver workbooks — not invented numbers.
A fix justified by a hand-picked example tends to encode the example; these
encode the defect.

Also asserts the corrections stay NARROW: every real solver error found in the
run's 27 single-cell failures must still fail. A fix that quietly forgives real
errors would inflate every phase and is worse than the defect it replaces.
"""

from __future__ import annotations

import datetime

import pytest

from src.agent_profiles.ssbv2_agent.scorer_fixes import (
    BARE_VOLATILE,
    ITERATIVE_POLICY,
    is_bare_volatile,
    normalize_formula,
    numbers_match,
    values_match,
)
from src.agent_profiles.ssbv2_agent.scorer import _compare_cell_value


# ---------------------------------------------------------------------------
# 1. Rounding boundary
# ---------------------------------------------------------------------------

# Read from the workbooks on 2026-08-01:
#   golden  .../Financial_Model/spreadsheet/08_Project Seafood Model/08_Seafood_golden.xlsx
#   solver  .../_snapshots/Financial_Model_08_04/iter_0_s0/workspace/output/08_04_Seafood_input.xlsx
OBSERVED_BOUNDARY_CASES = [
    # (golden, solver) — differ by ~1e-8, straddling a .xx5 boundary
    (5247785.62500001, 5247785.625),
    (9495255.22500001, 9495255.22499999),
]


@pytest.mark.parametrize("gold,sol", OBSERVED_BOUNDARY_CASES)
def test_observed_boundary_case_currently_fails(gold, sol):
    """Pin the defect: the CURRENT scorer calls these a mismatch.

    If this ever goes green on its own, scorer.py was changed and the fix
    below needs re-deriving rather than blindly applied.
    """
    assert _compare_cell_value(gold, sol) is False


@pytest.mark.parametrize("gold,sol", OBSERVED_BOUNDARY_CASES)
def test_observed_boundary_case_is_forgiven_by_the_fix(gold, sol):
    assert numbers_match(gold, sol) is True
    assert values_match(gold, sol) is True


def test_the_defect_mechanism_is_rounding_not_magnitude():
    """Same number, one ULP apart, on either side of the 2-decimal boundary."""
    gold, sol = 5247785.62500001, 5247785.625
    assert round(gold, 2) != round(sol, 2)          # what the scorer compares
    assert abs(gold - sol) < 1e-7                   # what the values actually are
    assert abs(gold - sol) / gold < 1e-14


# Every one of these is a REAL solver error observed in the run's single-cell
# failures. The fix must not touch any of them.
REAL_ERRORS_THAT_MUST_STILL_FAIL = [
    pytest.param(0.02, 0.01, id="Template_02_01-percentage"),
    pytest.param(50782789.76, -50782789.76, id="Template_15_04-sign"),
    pytest.param(59222921.7, -59222921.7, id="Template_15_01-sign"),
    pytest.param(2682.72, 2682.4, id="Template_06_04-recalc-drift"),
    pytest.param(11152.88, 11878.54, id="Debugging_02_04-6pct-block"),
    pytest.param(137.93, 0.0, id="Debugging_09-datatable-zero"),
    pytest.param(-137.89, -134.35, id="Debugging_09-iterative-drift"),
]


@pytest.mark.parametrize("gold,sol", REAL_ERRORS_THAT_MUST_STILL_FAIL)
def test_real_errors_are_not_forgiven(gold, sol):
    assert numbers_match(gold, sol) is False
    assert values_match(gold, sol) is False


def test_half_cent_absolute_tolerance_would_loosen_the_2dp_contract():
    """Guards the design choice recorded in the module docstring.

    `abs(a-b) <= 0.005` reads like the natural rendering of "same to 2
    decimals", but it forgives pairs that genuinely DIFFER at 2 decimals —
    here 0.03 vs 0.04 — which silently widens the benchmark rather than fixing
    a comparison defect. The relative rule leaves the contract exactly as
    specified and only repairs the float-noise boundary.

    Constructed, not observed: no corpus cell was found in this band. The test
    exists to stop the tolerance being "simplified" to an absolute one later.
    """
    a, b = 0.0349, 0.0399
    assert abs(a - b) <= 0.005 + 1e-12          # naive rule: forgiven
    assert round(a, 2) != round(b, 2)           # but they really differ at 2dp
    assert numbers_match(a, b) is False         # ours keeps them distinct


def test_fix_is_additive_only():
    """Anything the current scorer accepts must still be accepted.

    values_match is designed to be OR-ed with _compare_cell_value, so the
    rescore can only turn FAIL into PASS. That one-directionality is what makes
    the eventual diff auditable.
    """
    for a, b in [(1.0, 1.0), (0.0, 0.0), (-3.25, -3.25), (1.004, 1.0041)]:
        if _compare_cell_value(a, b):
            assert values_match(a, b, already_matched=True) is True


def test_non_numeric_and_bool_are_not_coerced():
    assert numbers_match("abc", "abc") is False      # strings are not this fix's job
    assert numbers_match(None, 0.0) is False
    assert numbers_match(True, 1.0) is False         # bool is not a number here
    assert numbers_match(float("nan"), float("nan")) is False


def test_numeric_strings_are_compared_as_numbers():
    """scorer._transform_value coerces numeric strings; _coerce must match it."""
    assert values_match("5247785.62500001", 5247785.625) is True
    assert values_match("not a number", 1.0) is False


def test_datetime_compared_as_excel_serial():
    d = datetime.datetime(2026, 8, 1)
    serial = (d - datetime.datetime(1899, 12, 30)).days
    assert values_match(d, float(serial)) is True


# ---------------------------------------------------------------------------
# 2. Volatile-function exemption
# ---------------------------------------------------------------------------

def test_lotus_plus_prefix_is_the_defect():
    """The exact miss: lstrip('=') leaves '+TODAY()', which is not in the tuple."""
    assert "=+TODAY()".lstrip("=").upper() not in BARE_VOLATILE   # old behaviour
    assert is_bare_volatile("=+TODAY()") is True                  # new behaviour


@pytest.mark.parametrize("formula", [
    "=TODAY()", "=+TODAY()", "=-TODAY()", "= TODAY() ",
    "=NOW()", "=+NOW()", "=today()", "=+RAND()",
])
def test_bare_volatile_forms_are_detected(formula):
    assert is_bare_volatile(formula) is True


@pytest.mark.parametrize("formula", [
    "=TODAY()+30",                                  # depends on a real base formula
    '=TEXT(TODAY(),"mm/dd/yyyy")',                  # Template/16_09 — separate decision
    '=CONCATENATE("As of Date: ",TEXT(TODAY(),"mm/dd/yyyy"))',   # Template/16_10
    "=SUM(A1:A5)", "=A1", "", "not a formula",
])
def test_non_bare_volatile_is_not_exempted(formula):
    assert is_bare_volatile(formula) is False


def test_normalize_handles_non_strings():
    assert normalize_formula(None) == ""
    assert normalize_formula(42) == ""


def test_the_ten_debugging_09_cells_are_covered():
    """All ten Debugging/09_* graded volatile cells hold this exact formula."""
    assert is_bare_volatile("=+TODAY()") is True


# ---------------------------------------------------------------------------
# 3. Iterative policy stays undecided
# ---------------------------------------------------------------------------

def test_iterative_policy_is_deliberately_unset():
    """Fails if someone quietly picks a tolerance for the open judgment call.

    The flip count across bootstrap+epoch-1 is threshold-sensitive (0 tasks at
    1e-4, 1 at 1e-3, 4 at 1e-1), so setting this without a design decision
    would choose a result rather than fix a bug.
    """
    assert ITERATIVE_POLICY is None
