"""Scorer corrections for SSB v2, staged but NOT yet wired in.

WHY THIS IS A SEPARATE, UNWIRED MODULE. These fix real defects in
`scorer.py`, but the scorer is not part of the RunCache `tree_hash` and cached
rollouts carry their own `inline_score`. Editing `scorer.py` during a live
evolution would leave bootstrap/e1/e2 scored under the old rules while e3/e4 —
which the driver launches as fresh processes — scored under the new ones, i.e.
exactly the cross-epoch comparison the experiment exists to make, measured two
different ways. Nothing here is imported by the scoring path yet.

The wiring plan, once the sealed test eval completes: import these into
`scorer.py`, then rescore **every** phase uniformly from the retained output
workbooks (`_snapshots/<task>/iter_0_s*/workspace/output/*.xlsx`, verified
present for 100% of rollouts that produced a file). No rollout needs re-running
and no LLM spend is involved.

Each correction is written as an ADDITIONAL acceptance path, never a stricter
one, so applying it can only turn a recorded FAIL into a PASS. That keeps the
rescore auditable: any verdict that changes, changes in one direction, and the
count of changes is the whole diff.

Defects addressed
-----------------
1. `values_match` — 2-decimal comparison is discontinuous at the rounding
   boundary (§2b of SCORING_ARTIFACT_AUDIT.md).
2. `is_bare_volatile` / `normalize_formula` — the volatile-function exemption
   misses `=+TODAY()` because it strips only `=` (§1).
3. `workbook_is_iterative` — detection only. The *policy* for iterative blocks
   is deliberately NOT decided here; see the module note on ITERATIVE_POLICY.
"""

from __future__ import annotations

import datetime
import math
from typing import Any

# ---------------------------------------------------------------------------
# 1. Rounding-boundary comparison
# ---------------------------------------------------------------------------

# Chosen from measured evidence, not taste. Financial_Model/08_04 failed all
# three bootstrap samples on two cells out of 200,362:
#
#   'Income Statement'!D59  golden 5247785.62500001  solver 5247785.625
#   'Cashflow'!G14          golden 9495255.22500001  solver 9495255.22499999
#
# The values agree to 1e-8 absolute / 1.9e-15 relative — the same number, one
# float ULP apart — but they straddle a `.xx5` boundary, so `round(v, 2)` sends
# them to 5247785.63 and 5247785.62 and the task scores 0/3.
#
# 1e-9 is ~six orders of magnitude above the observed noise and ~six below the
# smallest disagreement in the corpus that could be a modelling difference
# (Template/06_04, 1.2e-4 relative). Nothing in between was observed, so the
# threshold sits in an empty region rather than on a judgement call.
#
# NOT an absolute half-cent tolerance. `abs(a-b) <= 0.005` reads like the
# natural rendering of "same to 2 decimals", but it forgives pairs that
# genuinely differ at 2 decimals once the values are small (0.0349 vs 0.0399
# rounds to 0.03 vs 0.04). That widens the benchmark instead of repairing a
# comparison defect. The relative form leaves the 2-decimal contract exactly as
# specified and only closes the float-noise boundary.
NUMERIC_REL_TOL = 1e-9
NUMERIC_ABS_TOL = 1e-9


def numbers_match(a: Any, b: Any,
                  rel_tol: float = NUMERIC_REL_TOL,
                  abs_tol: float = NUMERIC_ABS_TOL) -> bool:
    """True when two numerics are the same value up to float representation.

    Deliberately narrow: this is a fix for a *comparison* defect, not a
    loosening of the benchmark's 2-decimal contract. Two genuinely different
    numbers that happen to round alike were already equal under the old rule and
    stay equal; two numbers that differ by more than float noise stay unequal.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return False
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return False
    if math.isnan(a) or math.isnan(b):
        return False
    return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=abs_tol)


def values_match(v1: Any, v2: Any, *, already_matched: bool = False) -> bool:
    """Acceptance path to OR with `scorer._compare_cell_value`.

    `already_matched` lets the caller short-circuit; it exists so the wiring is
    a single `or` at the call site rather than a restructure of the comparison.
    """
    if already_matched:
        return True
    return numbers_match(_coerce(v1), _coerce(v2))


def _coerce(v: Any) -> Any:
    """Mirror scorer._transform_value's type handling WITHOUT the rounding.

    Rounding is precisely what this module is correcting, so the numeric branch
    passes the value through untouched. The datetime branch is kept so date
    cells still compare as Excel serial numbers.
    """
    if isinstance(v, datetime.datetime):
        return (v - datetime.datetime(1899, 12, 30)).days + \
               (v - datetime.datetime(1899, 12, 30)).seconds / 86400.0
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    return v


# ---------------------------------------------------------------------------
# 2. Volatile-function exemption
# ---------------------------------------------------------------------------

# Zero-argument volatile functions only. Grading these against a static golden
# is unfalsifiable: the golden holds the date it was authored, the candidate
# recalculates to the run date. RAND() is included for completeness; it does not
# occur in the graded ranges of the current corpus.
BARE_VOLATILE = ("TODAY()", "NOW()", "RAND()")


def normalize_formula(formula: str) -> str:
    """Strip the leading `=` plus any Lotus-style sign prefix and whitespace.

    `scorer._volatile_function_skip_cells` used `formula.lstrip("=")`, which
    turns `=+TODAY()` into `+TODAY()` — not in the skip tuple, so the cell is
    graded. The leading `+` is a very common Excel idiom carried over from
    Lotus, and the entire Debugging/09_* family uses it: 10 of the 12 graded
    volatile cells the old rule missed.
    """
    if not isinstance(formula, str):
        return ""
    return formula.lstrip("=").lstrip("+-").strip().upper()


def is_bare_volatile(formula: str) -> bool:
    """True for a formula that is *only* a zero-argument volatile call.

    Stays narrow on purpose. `TODAY()+30` legitimately depends on the solver
    having the right base formula, and `TEXT(TODAY(),"mm/dd/yyyy")` embeds one
    inside a string expression — both are excluded here. Template/16_09 and
    16_10 are the two known corpus cells in that second category; they are
    equally unfalsifiable but need a separate decision, recorded in the audit.
    """
    return normalize_formula(formula) in BARE_VOLATILE


# ---------------------------------------------------------------------------
# 3. Iterative workbooks — DETECTION ONLY
# ---------------------------------------------------------------------------

# NO POLICY IS SET HERE, and that is deliberate.
#
# LibreOffice (which recalculates the candidate) and Excel (which produced the
# golden's cached values) land on different fixed points for circular blocks.
# The divergence is real, but "how much divergence is recalculation noise rather
# than a wrong assumption" has no evidence-backed answer yet, and the flip count
# is threshold-sensitive: across bootstrap+epoch-1, forgiving it flips 0 tasks
# at 1e-4, 1 at 1e-3, and 4 at 1e-1. Picking a tolerance would pick a result.
#
# 297 of 321 corpus goldens have iterate=True, so "is this workbook iterative"
# is very nearly a constant and cannot by itself justify forgiving a cell.
#
# The two real options are to recalculate the GOLDEN under the same engine as
# the candidate (removes the asymmetry at its source) or to exclude iterative
# blocks from grading (shrinks the benchmark). Both are design changes.
ITERATIVE_POLICY = None


def workbook_is_iterative(wb) -> bool:
    """True if the workbook has Excel's iterative calculation switched on."""
    calc = getattr(wb, "calculation", None)
    if calc is None:
        return False
    return bool(getattr(calc, "iterate", False))
