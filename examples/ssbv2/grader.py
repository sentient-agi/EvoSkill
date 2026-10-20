"""SpreadsheetBench v2 grader: grade(task, snapshot) -> score in [0, 1].

Written by the user of the run and handed to the benchmark adapter with the user instruction: the user implements the
grading mechanism, the adapter does not. It is the leg grading of the report's SpreadsheetBench v2 runs, as one
function:

  - task: the plug-in's Task. Read: inputs[0] (the task's input workbook), answer_files[0] (its reference workbook),
    grading["answer_position"].
  - snapshot: the solver's working directory when it exited: spreadsheets/<input name> and output/, where the solver was
    told to save output/<input name>.
  - The output is output/<input name>, else the first *.xlsx in output/ in sorted order (the legs' rule). No output
    scores 0.
  - A copy of the output is graded with score_and_diff(reference, copy, answer_position, orig_input_file=input) from
    src/agent_profiles/ssbv2_agent/scorer.py. It passes -> 1, else 0 (the benchmark is pass/fail).
  - The scorer does not raise when LibreOffice fails: it logs a warning and grades the un-recalculated file. Such a
    warning makes this function raise, which the loop records as grader_failed, never as a 0.

Runs in the grader's setup: the EvoSkill repository at /data/EvoSkill, /opt/evoskill/venv/bin/python (CPython 3.13.9, openpyxl 3.1.5 + lxml), `docker run
ssb-verifier:latest` served by the machine, the leg's grader timeouts in the environment. Standard library at import time;
the scorer (openpyxl) is loaded inside grade().
"""
from __future__ import annotations

import importlib.util
import logging
import shutil
import tempfile
from pathlib import Path

SCORER = Path("/data/EvoSkill/src/agent_profiles/ssbv2_agent/scorer.py")   # the EvoSkill repository in the grader's setup
FAILURE_WARNINGS = ("LibreOffice recalculation failed", "_solve_datatables_with_uno failed")


class GraderFailed(RuntimeError):
    """The grader itself failed on this snapshot (not the solver)."""


class _Warnings(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record):
        try:
            self.messages.append(record.getMessage())
        except Exception:                       # noqa: BLE001 -- a message that cannot be formatted is still kept
            self.messages.append(str(record.msg))


def _output(snapshot: Path, name: str) -> Path | None:
    out = snapshot / "output"
    exp = out / name
    if exp.is_file() and not exp.is_symlink():
        return exp
    xs = sorted(p for p in out.glob("*.xlsx") if p.is_file() and not p.is_symlink())
    return xs[0] if xs else None


def _scorer():
    spec = importlib.util.spec_from_file_location("ssbv2_scorer", SCORER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def grade(task, snapshot: Path) -> float:
    snapshot = Path(snapshot)
    inp, ref = Path(task.inputs[0]), Path(task.answer_files[0])
    out = _output(snapshot, inp.name)
    if out is None:
        return 0.0
    scorer = _scorer()
    tmp = Path(tempfile.mkdtemp(prefix="ssbv2_grade_"))
    root = logging.getLogger()
    cap, level = _Warnings(), root.level
    try:
        copy = tmp / "output" / out.name
        copy.parent.mkdir()
        shutil.copy2(out, copy)                 # the scorer rewrites the file it grades
        copy.chmod(copy.stat().st_mode | 0o200)
        if root.getEffectiveLevel() > logging.WARNING:
            root.setLevel(logging.WARNING)
        root.addHandler(cap)
        passed, _n_match, _n_total, _diff = scorer.score_and_diff(
            str(ref), str(copy), task.grading["answer_position"], orig_input_file=str(inp))
    finally:
        root.removeHandler(cap)
        root.setLevel(level)
        shutil.rmtree(tmp, ignore_errors=True)
    failed = [m for m in cap.messages if any(p in m for p in FAILURE_WARNINGS)]
    if failed:
        raise GraderFailed(failed[0][:500])
    return 1.0 if passed else 0.0
