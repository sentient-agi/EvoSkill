"""An episode with no computable verdict must never be scored 0.0.

RecalculationFailed was added so a grading-container fault could not be reported as a
solver's wrong answer. Its own docstring claimed it was "deliberately NOT caught anywhere
in this module" -- and four broad `except Exception: return 0.0` handlers caught it anyway,
including both live SSB v2 scoring call sites. The guard could not fire.

Two kinds of test here, because one alone would not have caught it:
  * BEHAVIOURAL -- make the recalc raise, assert the entry point propagates.
  * STRUCTURAL  -- no `except Exception` may sit in front of a scoring call without an
    `except UngradableEpisode: raise` ahead of it. That is the one that survives someone
    adding a fifth handler.
"""
import ast
from pathlib import Path

import pytest

from src.harness.errors import UngradableEpisode

REPO = Path(__file__).resolve().parents[1]

SCORING_FILES = [
    "src/agent_profiles/ssbv2_agent/scorer.py",
    "src/agent_profiles/ssbv2_agent/ssb_v2_agent.py",
    "src/evaluation/evaluate.py",
]
SCORING_CALLS = ("score_and_diff(", "score_ssbv2(", "scorer(")


def _catches(handler: ast.ExceptHandler, name: str) -> bool:
    t = handler.type
    if t is None:
        return name == "BaseException"
    names = [t] if not isinstance(t, ast.Tuple) else list(t.elts)
    return any(getattr(n, "id", getattr(n, "attr", None)) == name for n in names)


@pytest.mark.parametrize("relpath", SCORING_FILES)
def test_no_broad_handler_swallows_an_ungradable_episode(relpath):
    src = (REPO / relpath).read_text()
    tree = ast.parse(src)
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        body = "\n".join(ast.get_source_segment(src, s) or "" for s in node.body)
        if not any(c in body for c in SCORING_CALLS):
            continue
        for i, h in enumerate(node.handlers):
            if _catches(h, "Exception") or _catches(h, "BaseException"):
                earlier = node.handlers[:i]
                if not any(_catches(e, "UngradableEpisode") for e in earlier):
                    offenders.append(f"{relpath}:{h.lineno}")
                break
    assert not offenders, (
        "broad handler in front of a scoring call with no `except UngradableEpisode: "
        "raise` ahead of it — an infra fault there becomes a 0.0: " + ", ".join(offenders))


def test_recalculation_failed_is_an_ungradable_episode():
    from src.agent_profiles.ssbv2_agent.scorer import RecalculationFailed
    assert issubclass(RecalculationFailed, UngradableEpisode)
    # Exception, NOT BaseException: gather(return_exceptions=True) must still capture it
    # into the failure ledger rather than tearing down the whole leg.
    assert issubclass(RecalculationFailed, Exception)


def test_score_ssbv2_propagates_instead_of_returning_zero(tmp_path, monkeypatch):
    """The auditor's reproduction: make the recalc fail, score_ssbv2 must NOT return 0.0."""
    openpyxl = pytest.importorskip("openpyxl")
    from src.agent_profiles.ssbv2_agent import scorer as S

    gt = tmp_path / "gt.xlsx"; proc = tmp_path / "proc.xlsx"
    for p, v in ((gt, 1), (proc, 999)):
        wb = openpyxl.Workbook(); wb.active["A1"] = v; wb.save(p)

    def boom(path):
        raise S.RecalculationFailed("simulated grading-container fault")
    monkeypatch.setattr(S, "_recalculate_with_libreoffice", boom)

    # score_ssbv2 takes JSON on both sides; passing bare paths makes it return 0.0
    # from the json.loads guard long before it reaches the scorer, which is how the
    # first version of THIS test passed while proving nothing.
    import json
    question = json.dumps({"task_id": "Debugging/01_02", "spreadsheet_path": str(proc)})
    ground_truth = json.dumps({"task_id": "Debugging/01_02", "golden_path": str(gt),
                               "answer_position": "A1"})

    with pytest.raises(UngradableEpisode):
        S.score_ssbv2(question, str(proc), ground_truth)


def test_the_zero_path_is_still_reachable(tmp_path, monkeypatch):
    """Sanity: an ORDINARY scoring error still returns 0.0, so the fix is narrow."""
    openpyxl = pytest.importorskip("openpyxl")
    import json
    from src.agent_profiles.ssbv2_agent import scorer as S
    gt = tmp_path / "gt.xlsx"; proc = tmp_path / "proc.xlsx"
    for p_, v in ((gt, 1), (proc, 999)):
        wb = openpyxl.Workbook(); wb.active["A1"] = v; wb.save(p_)
    monkeypatch.setattr(S, "_recalculate_with_libreoffice",
                        lambda path: (_ for _ in ()).throw(ValueError("ordinary failure")))
    q = json.dumps({"spreadsheet_path": str(proc)})
    g = json.dumps({"golden_path": str(gt), "answer_position": "A1"})
    assert S.score_ssbv2(q, str(proc), g) == 0.0


# ── a zero-cell comparison must never read as a PASS ─────────────────────────────
# compare_workbooks_detailed sets all_pass = True up front and only the per-cell loop
# can falsify it, so every path that graded nothing returned (True, 0, 0, "All cells
# match"). Measured routes: an answer_position the range parser could not read (it
# raised AttributeError on '$A$1' / 'a1:b2' / 'A:A' and the caller dropped the range),
# and every graded cell landing in the volatile-function skip set. A false PASS can
# only move a reported number up.

@pytest.mark.parametrize("answer_position,expect", [
    ("A1", "graded"),          # the plain form always worked
    ("$A$1", "graded"),        # absolute markers: was a FALSE PASS over 0 cells
    ("a1:b2", "graded"),       # lowercase: was a FALSE PASS over 0 cells
    ("Sheet!$A$1", "graded"),  # sheet-qualified + absolute
    ("A:A", "raise"),          # unbounded: cannot be a bounded rectangle
])
def test_no_answer_position_spelling_yields_a_zero_cell_pass(tmp_path, answer_position, expect):
    openpyxl = pytest.importorskip("openpyxl")
    from src.agent_profiles.ssbv2_agent.scorer import compare_workbooks_detailed

    # tmp_path is fine here: compare_workbooks_detailed is pure openpyxl. The container
    # recalc (which DOES need a $HOME-shared path -- /var/folders mounts empty) happens
    # upstream in score_and_diff, not in this function.
    gt, proc = tmp_path / "gt.xlsx", tmp_path / "proc.xlsx"
    for f, v in ((gt, 1), (proc, 999)):
        wb = openpyxl.Workbook(); wb.active["A1"] = v; wb.save(f)

    if expect == "raise":
        with pytest.raises(UngradableEpisode):
            compare_workbooks_detailed(str(gt), str(proc), answer_position)
    else:
        passed, _, total, _ = compare_workbooks_detailed(str(gt), str(proc), answer_position)
        assert total > 0, "graded zero cells — that is the false-PASS shape"
        assert passed is False, "workbooks differ at A1; a PASS here is the defect"


def test_parse_cell_range_normalises_absolute_and_case():
    from src.agent_profiles.ssbv2_agent.scorer import _parse_cell_range
    canonical = _parse_cell_range("A1:B2")
    for spelling in ("$A$1:$B$2", "a1:b2", " A1:B2 ", "$a$1:$b$2"):
        assert _parse_cell_range(spelling) == canonical, spelling
    for bad in ("A:A", "1:1", "", "not-a-range"):
        with pytest.raises(ValueError):
            _parse_cell_range(bad)
