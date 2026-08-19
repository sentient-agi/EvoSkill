"""Every outcome of a scoring attempt is exactly one of: a verdict, or no verdict.

The distinction is not cosmetic. An UngradableEpisode lands in the failure ledger and
blocks the leg from writing a result JSON; a False lands in the numbers as a wrong answer.
Putting a case in the wrong bucket is a defect in BOTH directions, and this repo has now
shipped both: a golden workbook that would not open was scored as the solver's 0.0, and
then the guard written to stop that reclassified an ordinary FAIL (solver wrote a new
workbook, so the graded sheet was missing from its OUTPUT) as an infra fault — which,
because a leg with any failed rollout refuses to write its result, voided the whole leg.

So the taxonomy is pinned here case by case, on real workbooks.
"""
import openpyxl
import pytest

from src.agent_profiles.ssbv2_agent.scorer import compare_workbooks_detailed
from src.harness.errors import UngradableEpisode

SHEET = "DCF_Analysis"
AP = f"'{SHEET}'!B2:C2"


@pytest.fixture
def books(tmp_path):
    def mk(name, sheet, vals=None):
        w = openpyxl.Workbook()
        w.active.title = sheet
        for k, v in (vals or {}).items():
            w.active[k] = v
        p = tmp_path / name
        w.save(p)
        return str(p)
    corrupt = tmp_path / "corrupt.xlsx"
    corrupt.write_bytes(b"not a zip file")
    return {
        "gt":      mk("gt.xlsx", SHEET, {"B2": 1, "C2": 2}),
        "same":    mk("same.xlsx", SHEET, {"B2": 1, "C2": 2}),
        "differs": mk("differs.xlsx", SHEET, {"B2": 9, "C2": 2}),
        "fresh":   mk("fresh.xlsx", "Sheet"),      # solver built a new workbook
        "corrupt": str(corrupt),
        "absent":  str(tmp_path / "nope.xlsx"),
    }


@pytest.mark.parametrize("gt,proc,ap,passed,note", [
    ("gt", "same",    AP, True,  "All cells match"),
    ("gt", "differs", AP, False, "B2"),
    # THE SOLVER'S FAULTS ARE VERDICTS, including the one my own guard misfiled.
    ("gt", "fresh",   AP, False, "missing from output"),
    ("gt", "absent",  AP, False, "Output file not found"),
    ("gt", "corrupt", AP, False, "Cannot open output workbook"),
])
def test_solver_outcomes_are_verdicts(books, gt, proc, ap, passed, note):
    ok, _, _, diff = compare_workbooks_detailed(books[gt], books[proc], ap)
    assert ok is passed and note in diff, diff


@pytest.mark.parametrize("gt,proc,ap,kwargs", [
    # GRADING-ARTIFACT FAULTS HAVE NO VERDICT.
    ("absent",  "same", AP, {}),                                  # golden gone
    ("corrupt", "same", AP, {}),                                  # golden unreadable
    ("gt", "same", "'NoSuchSheet'!B2:C2", {}),                    # golden lacks its own sheet
    ("gt", "same", "'DCF_Analysis'!A:A", {}),                     # range unparseable
    ("gt", "same", AP, {"skip_cells": {(SHEET, "B2"), (SHEET, "C2")}}),   # all excluded
])
def test_infra_faults_raise(books, gt, proc, ap, kwargs):
    with pytest.raises(UngradableEpisode):
        compare_workbooks_detailed(books[gt], books[proc], ap, **kwargs)


def test_a_sheet_absent_from_both_blames_the_dataset_not_the_solver(books):
    """Order of the two sheet checks. Asking the OUTPUT first reports 'missing from
    output' for a task whose grading target does not exist anywhere — blaming the solver
    for a dataset fault, and producing a scoreable 0.0 from it."""
    with pytest.raises(UngradableEpisode, match="has no sheet"):
        compare_workbooks_detailed(books["gt"], books["fresh"], "'Nowhere'!B2:C2")


def test_sibling_scorer_never_reports_pass_on_zero_graded_cells(tmp_path):
    """src/evaluation/spreadsheetbench_scorer.py ships to the evolver as
    ./oracle/spreadsheetbench_scorer.py for what-if checks. It returned
    (True, 0, 0, 'All cells match') whenever every range failed to parse, because an
    unparseable range was `continue`d in silence and all_pass starts True."""
    from src.evaluation.spreadsheetbench_scorer import compare_workbooks_detailed as sib

    def mk(name, vals):
        w = openpyxl.Workbook()
        for k, v in vals.items():
            w.active[k] = v
        p = tmp_path / name
        w.save(p)
        return str(p)

    gt, proc = mk("g.xlsx", {"A1": 1}), mk("p.xlsx", {"A1": 999})
    for ap in ("A1:B2:C3", "", "Sheet!"):
        ok, _, _, diff = sib(gt, proc, ap)
        assert ok is False, f"{ap!r} reported PASS: {diff}"
    ok, match, total, _ = sib(gt, proc, "A1:B2")          # ...and real ranges still grade
    assert ok is False and total == 4 and match == 3
