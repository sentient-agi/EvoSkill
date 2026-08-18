"""SpreadsheetBench v2 scorer.

Uses openpyxl cell-level comparison against golden answers.
Adapted from the official SSB v2 evaluation script.
"""

from __future__ import annotations

import json
import datetime
import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
import colorsys
import xml.etree.ElementTree as ET
from pathlib import Path

import openpyxl
from openpyxl.styles.colors import COLOR_INDEX

logger = logging.getLogger(__name__)

# Root-caused: a large fraction of "TABLE()-wiped" false
# negatives are NOT a fundamental LibreOffice incapability at all -- an
# ISOLATED recalc of a genuinely confirmed false-negative snapshot
# (Financial_Model/19_02, /19_01, /09_01) computed the correct Data Table
# values on the very first attempt, in 1.5-1.8s, nowhere near the 60s
# timeout. The scoring step has ZERO concurrency control of its own: when
# many solver tasks (already concurrency-limited by the AIMD limiter for
# LLM calls) finish scoring around the same time, their `docker run
# ssb-verifier` recalc subprocesses all fire simultaneously, competing for
# the same fixed CPU budget (13 cores) that
# solving ALSO needs -- a computationally expensive recalc (a Data Table
# recomputes the whole model N times) can easily blow past 60s under that
# contention while taking under 2s in isolation. Bounding concurrency here
# (independent of solver concurrency) plus a generous timeout margin fixes
# the ROOT CAUSE for most cases -- the xlfn-prefix fix and the
# restore-from-original-input fallback remain valuable for the separate,
# genuine cases (FILTER/SORT/TEXTSPLIT array-spill gap; tasks whose
# correct Data Table value isn't recoverable via a plain recalc at all),
# but should no longer be the FIRST thing blamed for a None cell.
_RECALC_SEMAPHORE = threading.Semaphore(int(os.environ.get("EVOSKILL_RECALC_CONCURRENCY", "6")))
_RECALC_TIMEOUT_SECONDS = int(os.environ.get("EVOSKILL_RECALC_TIMEOUT_SECONDS", "180"))

# The UNO-driven Data Table solver spins up its own soffice listener process
# and does N+1 document loads (N = number of TABLE() cells found, typically
# tens per table) -- much heavier per-call than a plain recalc, so it gets
# its own, smaller concurrency cap and a more generous timeout.
_DATATABLE_UNO_SEMAPHORE = threading.Semaphore(int(os.environ.get("EVOSKILL_DATATABLE_UNO_CONCURRENCY", "3")))


def configure_multiprocess_semaphores(recalc_sem, uno_sem) -> None:
    """Call once inside each ProcessPoolExecutor worker (via `initializer=`)
    to replace the default threading.Semaphore globals with multiprocessing-
    safe ones created ONCE by the parent process and passed to every worker.

    Why this exists: a bare module-level threading.Semaphore only enforces
    its limit within a single process. The moment score_and_diff is called
    from more than one OS process at a time (e.g. a ProcessPoolExecutor,
    needed because openpyxl cell-comparison is pure-Python and GIL-bound --
    a plain ThreadPoolExecutor can't parallelize that part no matter how
    many threads it has), each worker process re-imports this module under
    'spawn' and gets its OWN independent semaphore starting at the full
    count -- silently multiplying the real concurrency cap by the worker
    count instead of enforcing it globally. That's exactly the CPU-
    contention-causing-false-timeouts failure mode _RECALC_SEMAPHORE exists
    to prevent (see the comment above it), just reintroduced via a
    different door. multiprocessing.Semaphore wraps a real OS-level
    semaphore and stays correctly shared across processes when the SAME
    object is passed in (picklable), which is what this function is for.
    """
    global _RECALC_SEMAPHORE, _DATATABLE_UNO_SEMAPHORE
    _RECALC_SEMAPHORE = recalc_sem
    _DATATABLE_UNO_SEMAPHORE = uno_sem
_DATATABLE_UNO_TIMEOUT_SECONDS = int(os.environ.get("EVOSKILL_DATATABLE_UNO_TIMEOUT_SECONDS", "300"))
_DATATABLE_DOCKER_SCRIPTS_DIR = str((Path(__file__).parent / "docker_scripts").resolve())


# THE IMAGE THAT PRODUCES THE VERDICT. Named once, and overridable, so the runner can
# record its identity in the cache-environment marker -- and so the two sites that run it
# (the LibreOffice recalc and the =TABLE() UNO fix) cannot drift apart. A second copy of a
# convention is a second chance to disagree; that is what resolve_base was about too.
GRADING_IMAGE = os.environ.get("EVOSKILL_GRADING_IMAGE", "ssb-verifier:latest")


class RecalculationFailed(RuntimeError):
    """The grading container could not recalculate a workbook.

    Deliberately NOT caught anywhere in this module. An episode whose formulas
    were never recalculated has no computable verdict, and the failure mode it
    replaces -- degrade to the un-recalculated file and score 0.0 -- puts an
    infrastructure fault into a results table as if it were solver error.
    """


def _recalculate_with_libreoffice(xlsx_path: str) -> str:
    """Recalculate formulas in an xlsx file using LibreOffice via Docker.

    Uses ssb-verifier image which has LibreOffice installed.
    Returns the path to the recalculated file (overwrites in place).

    RAISES RecalculationFailed if the grading container does not succeed. It
    used to log a warning and return the ORIGINAL path, with a docstring saying
    "callers must not treat this silently" -- but all four call sites did
    exactly that, because the contract was enforced only by a human reading
    logs. The consequence is not a missing warning: under data_only=True every
    formula cell of an un-recalculated workbook reads None, so the task scores
    0.0 and is indistinguishable from a solver that answered wrongly. A broken
    verifier image, a docker daemon hiccup or a recalc timeout therefore
    manufactured silent zeros in a measured result. A verdict that cannot be
    computed must stop the run, not be reported as a failing verdict.

    Bounded by _RECALC_SEMAPHORE, independent of solver concurrency -- see
    the module-level comment above for why: this is a blocking subprocess
    call, so a plain threading.Semaphore (not asyncio) correctly throttles
    it regardless of whether the caller is sync or async.
    """
    xlsx_path = str(Path(xlsx_path).resolve())
    xlsx_dir = str(Path(xlsx_path).parent)
    xlsx_name = Path(xlsx_path).name

    with _RECALC_SEMAPHORE:
        try:
            result = subprocess.run(
                ["docker", "run", "--rm",
                 "-v", f"{xlsx_dir}:/work",
                 "-w", "/work",
                 GRADING_IMAGE,
                 "bash", "-c",
                 # cp by exact name: /tmp/*.xlsx skips dot-prefixed (hidden) temp
                 # names, silently failing the copy after a successful convert.
                 # soffice keeps the input basename (already .xlsx).
                 f'libreoffice --headless --calc --convert-to "xlsx:Calc MS Excel 2007 XML" --outdir /tmp "{xlsx_name}" 2>/dev/null && cp "/tmp/{xlsx_name}" "{xlsx_name}"'],
                capture_output=True, text=True, timeout=_RECALC_TIMEOUT_SECONDS,
            )
            if result.returncode == 0:
                return xlsx_path
            raise RecalculationFailed(
                f"LibreOffice recalculation failed (exit {result.returncode}) for {xlsx_path}: "
                f"stdout={result.stdout[-500:]!r} stderr={result.stderr[-500:]!r}. "
                f"Scoring cannot proceed: every formula cell of an un-recalculated workbook "
                f"reads None under data_only=True, which scores 0.0 and is indistinguishable "
                f"from a wrong answer. Check that ssb-verifier:latest exists and runs "
                f"(scripts/build_ssb_solver_lo.sh builds it)."
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            raise RecalculationFailed(
                f"LibreOffice recalculation failed for {xlsx_path}: {e!r}. Scoring cannot "
                f"proceed; see the note above on why this must not degrade to a 0.0."
            ) from e


def _solve_datatables_with_uno(xlsx_path: str, gt_file: str | None = None, answer_position: str | None = None) -> None:
    """Correctly compute Excel two-variable Data Table (=TABLE()) cells that
    LibreOffice's own import/recalc cannot: it never performs the actual
    row/column substitution Excel does, it just reads the corner formula
    once under the base-case input values and copies that single number
    into the entire output range (confirmed empirically: a uniform wrong
    value, or #VALUE!/#DIV0!, across every cell in the range).

    This must run AFTER _recalculate_with_libreoffice -- that step is what
    turns the anchor cell from an openpyxl DataTableFormula OBJECT into the
    plain string '=TABLE($F$9,$C$12,$F10,$C$15,G$9)' that every cell in the
    range carries post-recalc, which is what this function's scan targets
    (fully self-describing per cell: corner/row-input/row-header/col-input/
    col-header cell refs, no separate range-geometry step needed).

    The fix: for each such cell, temporarily overwrite the two input cells
    with THIS cell's row/column header values, force a hard recalculation,
    and read the corner formula's freshly-computed result back -- exactly
    what Excel's Data Table feature does internally. Verified bit-exact
    against golden on real cells (11515.71286542351 vs golden's
    11515.7128654235).

    Bounded but real limitation: LibreOffice's own (non-Data-Table-related)
    formula evaluation has been confirmed to have independent correctness
    bugs on some input values for some models (e.g. a discount-rate chain
    evaluating wrong for one specific rate but not others) -- this fix
    cannot correct those; it only removes the Data-Table-specific gap.
    _restore_unrecalculable_cells runs AFTER this (see score_and_diff) and
    unconditionally prefers the original input's own cached value whenever
    one is available and the formula is unchanged, regardless of what THIS
    function wrote -- so a case where this function's answer is wrong (an
    unrelated LibreOffice engine bug) still gets corrected by restoration
    whenever a valid historical reference exists; this function only ends
    up being the final word for cells where no such reference exists at
    all (the genuinely unrecoverable-by-restoration cases).

    Cheap host-side pre-check avoids ever starting a UNO/soffice session for
    the large majority of tasks that have no Data Tables at all.

    SAFETY NET (v2 — regression-aware, not change-aware): the first version
    of this check flagged ANY value change to a non-target cell as
    "corruption" and rolled back on it. That was too blunt and, on a real
    multi-table file, rejected every single attempt: a full calculateAll()
    pass legitimately recomputes the ENTIRE workbook, so plenty of unrelated
    cells changing value (often becoming MORE correct, since a full
    recalculation is exactly what they need) is normal and desirable, not
    damage. The only change that actually matters for scoring is a cell
    inside the graded answer_position that WAS matching golden and BECOMES
    wrong -- a genuine regression. Cells outside answer_position never
    affect the score at all, so they are not checked. When gt_file/
    answer_position aren't supplied (e.g. a caller with no golden handy),
    this falls back to the original blunt whole-workbook check, which is
    strictly safe (over-conservative) but may under-deliver on complex
    multi-table files the same way the v1 check did.
    """
    try:
        wb = openpyxl.load_workbook(xlsx_path, data_only=False)
        target_cells = set()
        for ws in wb.worksheets:
            for row in ws.iter_rows():
                for cell in row:
                    if isinstance(cell.value, str) and "TABLE(" in cell.value.upper():
                        target_cells.add((ws.title, cell.coordinate))
        if not target_cells:
            return
    except Exception as e:
        logger.warning(f"_solve_datatables_with_uno: pre-check failed for {xlsx_path}: {e}")
        return

    xlsx_path = str(Path(xlsx_path).resolve())
    xlsx_dir = str(Path(xlsx_path).parent)
    xlsx_name = Path(xlsx_path).name

    backup_path = xlsx_path + ".pre_uno_backup"
    shutil.copy2(xlsx_path, backup_path)

    def _graded_cells() -> set[tuple[str, str]]:
        """(sheet, coordinate) for every cell in answer_position."""
        cells = set()
        for sheet_cell_range in _split_answer_position(answer_position):
            if "!" in sheet_cell_range:
                sheet_name, cell_range = sheet_cell_range.rsplit("!", 1)
                sheet_name = sheet_name.strip("'")
            else:
                sheet_name = wb.sheetnames[0]
                cell_range = sheet_cell_range
            cell_range = cell_range.strip("'")
            try:
                min_row, min_col, max_row, max_col = _parse_cell_range(cell_range)
            except Exception:
                continue
            for row in range(min_row, max_row + 1):
                for col in range(min_col, max_col + 1):
                    c = col
                    col_letter = ""
                    while c > 0:
                        c, r = divmod(c - 1, 26)
                        col_letter = chr(65 + r) + col_letter
                    cells.add((sheet_name, f"{col_letter}{row}"))
        return cells

    def _snapshot_graded_match_state(graded_cells: set[tuple[str, str]], wb_gt):
        """{(sheet, coord): matches_golden_bool} for every graded, non-target cell."""
        wb_v = openpyxl.load_workbook(xlsx_path, data_only=True)
        snap = {}
        for sheet_name, coord in graded_cells:
            if (sheet_name, coord) in target_cells:
                continue  # target cells are ALLOWED to change -- that's the whole point
            if sheet_name not in wb_v.sheetnames or sheet_name not in wb_gt.sheetnames:
                continue
            proc_val = wb_v[sheet_name][coord].value
            gt_val = wb_gt[sheet_name][coord].value
            snap[(sheet_name, coord)] = _compare_cell_value(gt_val, proc_val)
        return snap

    use_golden_aware_check = gt_file is not None and answer_position is not None
    graded_cells = None
    wb_gt = None
    if use_golden_aware_check:
        try:
            wb_gt = openpyxl.load_workbook(gt_file, data_only=True)
            graded_cells = _graded_cells()
        except Exception as e:
            logger.warning(f"_solve_datatables_with_uno: golden-aware check setup failed for {xlsx_path}: {e}, falling back to whole-workbook check")
            use_golden_aware_check = False

    def _snapshot_all_non_target_values():
        # _transform_value rounds numerics to 2 decimals -- the SAME
        # tolerance _compare_cell_value uses for actual scoring. Without
        # this, a full recalc's ordinary 13th-decimal-digit floating-point
        # noise (e.g. 11038.4383662343 vs ...44, confirmed live on a real
        # golden Data-Table refresh) reads as "corruption" and rejects an
        # otherwise-correct fix for a difference that could never affect
        # any actual pass/fail decision.
        wb_v = openpyxl.load_workbook(xlsx_path, data_only=True)
        snap = {}
        for ws in wb_v.worksheets:
            for row in ws.iter_rows():
                for cell in row:
                    if (ws.title, cell.coordinate) in target_cells:
                        continue
                    if cell.value is not None:
                        snap[(ws.title, cell.coordinate)] = _transform_value(cell.value)
        return snap

    try:
        before = _snapshot_graded_match_state(graded_cells, wb_gt) if use_golden_aware_check else _snapshot_all_non_target_values()
    except Exception as e:
        logger.warning(f"_solve_datatables_with_uno: pre-snapshot failed for {xlsx_path}: {e}, skipping")
        os.remove(backup_path)
        return

    with _DATATABLE_UNO_SEMAPHORE:
        try:
            result = subprocess.run(
                ["docker", "run", "--rm",
                 "-v", f"{xlsx_dir}:/work",
                 "-v", f"{_DATATABLE_DOCKER_SCRIPTS_DIR}:/scripts:ro",
                 GRADING_IMAGE,
                 "bash", "/scripts/run_datatable_fix.sh", xlsx_name],
                capture_output=True, text=True, timeout=_DATATABLE_UNO_TIMEOUT_SECONDS,
            )
            if result.returncode != 0:
                logger.warning(
                    f"_solve_datatables_with_uno failed (exit {result.returncode}) for {xlsx_path}: "
                    f"stdout={result.stdout[-500:]!r} stderr={result.stderr[-500:]!r} — "
                    f"restoring pre-fix file, falling back to plain recalc + restoration"
                )
                shutil.move(backup_path, xlsx_path)
                return
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            logger.warning(f"_solve_datatables_with_uno failed for {xlsx_path}: {e} — restoring pre-fix file")
            shutil.move(backup_path, xlsx_path)
            return

    try:
        after = _snapshot_graded_match_state(graded_cells, wb_gt) if use_golden_aware_check else _snapshot_all_non_target_values()
    except Exception as e:
        logger.warning(f"_solve_datatables_with_uno: post-snapshot failed for {xlsx_path}: {e} — restoring pre-fix file")
        shutil.move(backup_path, xlsx_path)
        return

    if use_golden_aware_check:
        # A regression is a graded, non-target cell that WAS correct and is
        # NOW wrong. Anything else -- a cell staying wrong, becoming
        # correct, or changing between two different wrong values -- is not
        # a regression and does not trigger a rollback.
        regressions = {k: (before[k], after.get(k)) for k in before if before[k] is True and after.get(k) is not True}
        if regressions:
            logger.warning(
                f"_solve_datatables_with_uno: {len(regressions)} graded cell(s) regressed from "
                f"matching golden to not matching for {xlsx_path} (sample: {dict(list(regressions.items())[:5])}) "
                f"— discarding this attempt, restoring pre-fix file"
            )
            shutil.move(backup_path, xlsx_path)
            return
    else:
        corrupted = {k: (v, after.get(k)) for k, v in before.items() if after.get(k) != v}
        corrupted.update({k: (None, v) for k, v in after.items() if k not in before and k not in target_cells})
        if corrupted:
            logger.warning(
                f"_solve_datatables_with_uno: detected {len(corrupted)} changed non-target cell(s) "
                f"for {xlsx_path} (no golden available for a regression-aware check; sample: "
                f"{dict(list(corrupted.items())[:5])}) — discarding this attempt, restoring pre-fix file"
            )
            shutil.move(backup_path, xlsx_path)
            return

    os.remove(backup_path)


# OOXML spec requires "future functions" (anything added to Excel after the
# 97-2003 function set) to be serialized in formula XML with a literal
# `_xlfn.` prefix. Real Excel's desktop UI adds this silently when you type
# a formula -- a human never sees it -- but openpyxl (what solvers use to
# write .xlsx files) writes the bare function name, which is spec-non-
# compliant XML. LibreOffice's parser is spec-compliant and correctly
# rejects the bare name with #NAME? -- this is NOT a LibreOffice bug, it's
# non-compliant input. Confirmed via SSB v1 forensics (AGGREGATE, IFNA,
# XLOOKUP) and SSB v2 audit (same functions causing #NAME? errors that get
# misread as genuine solver mistakes).
#
# Empirically verified against the actual ssb-verifier image (LibreOffice
# 25.2.3.2) before shipping this -- do not trust the "just needs a prefix"
# claim blindly, some function families need more than a naming fix:
#   WORKS with single `_xlfn.` prefix: AGGREGATE, IFNA, XLOOKUP, XMATCH,
#     LET, UNIQUE, SEQUENCE (all confirmed by direct test).
#   Originally guessed these dynamic-array functions needed a DOUBLE
#     `_xlfn._xlws.` prefix -- WRONG, confirmed by test: XLOOKUP only
#     works with the single prefix; the double prefix produces #NAME? just
#     like no prefix at all. There is no `_xlws`-prefixed variant in use
#     here.
#   STILL BROKEN even with the correct single prefix: FILTER, SORT,
#     TEXTSPLIT (confirmed by direct test -- these remain #NAME?). This
#     looks like a genuine LibreOffice array-spill engine gap, not a
#     naming/spec-compliance issue -- the same class of problem as Excel
#     Data Tables (=TABLE()), which also can't be fixed by touching the
#     formula text. Deliberately excluded from _XLFN_FUNCTIONS below so
#     this fix doesn't claim to solve something it doesn't; SORTBY,
#     RANDARRAY, and other array-spill functions are assumed to share the
#     same gap (untested) and are excluded too, pending direct evidence.
_XLFN_FUNCTIONS = {
    "AGGREGATE", "IFNA", "IFS", "TEXTJOIN", "CONCAT", "SWITCH", "MAXIFS", "MINIFS",
    "NUMBERVALUE", "DAYS", "ISOWEEKNUM", "SHEET", "SHEETS", "RANK.EQ", "RANK.AVG",
    "PERCENTILE.EXC", "PERCENTILE.INC", "QUARTILE.EXC", "QUARTILE.INC",
    "PERCENTRANK.EXC", "PERCENTRANK.INC", "MODE.MULT", "MODE.SNGL",
    "STDEV.P", "STDEV.S", "VAR.P", "VAR.S", "COVARIANCE.P", "COVARIANCE.S",
    "CONFIDENCE.NORM", "CONFIDENCE.T", "CHISQ.DIST", "CHISQ.DIST.RT", "CHISQ.INV",
    "CHISQ.INV.RT", "CHISQ.TEST", "F.DIST", "F.DIST.RT", "F.INV", "F.INV.RT", "F.TEST",
    "T.DIST", "T.DIST.2T", "T.DIST.RT", "T.INV", "T.INV.2T", "T.TEST", "Z.TEST",
    "NORM.DIST", "NORM.INV", "NORM.S.DIST", "NORM.S.INV", "LOGNORM.DIST", "LOGNORM.INV",
    "GAMMA.DIST", "GAMMA.INV", "GAMMALN.PRECISE", "BETA.DIST", "BETA.INV",
    "BINOM.DIST", "BINOM.DIST.RANGE", "BINOM.INV", "POISSON.DIST", "HYPGEOM.DIST",
    "NEGBINOM.DIST", "EXPON.DIST", "WEIBULL.DIST", "SKEW.P", "PERMUTATIONA",
    "COMBINA", "ARABIC", "BASE", "DECIMAL", "BITAND", "BITOR", "BITXOR",
    "BITLSHIFT", "BITRSHIFT", "CEILING.MATH", "CEILING.PRECISE", "FLOOR.MATH",
    "FLOOR.PRECISE", "ACOT", "ACOTH", "COT", "COTH", "CSC", "CSCH", "SEC", "SECH",
    "ERF.PRECISE", "ERFC.PRECISE", "GAUSS", "PHI", "PDURATION", "RRI",
    "FORECAST.LINEAR", "FORECAST.ETS", "FORECAST.ETS.CONFINT",
    "FORECAST.ETS.SEASONALITY", "FORECAST.ETS.STAT", "FILTERXML", "WEBSERVICE",
    "UNICHAR", "UNICODE", "MUNIT", "IMCOSH", "IMCOT", "IMCSC", "IMCSCH", "IMSEC",
    "IMSECH", "IMSINH", "IMTAN", "LET", "LAMBDA", "XLOOKUP", "XMATCH", "UNIQUE",
    "SEQUENCE",
}
_XLFN_PATTERN = None  # built lazily, see _add_xlfn_prefixes


def _add_xlfn_prefixes(xlsx_path: str) -> None:
    """Insert missing _xlfn. prefixes into formula text in-place.

    Skips any formula that already contains "_xlfn" (already prefixed --
    the regex only ever adds, never duplicates, since it matches on the
    non-prefixed function name with a negative lookbehind for identifier
    characters and dots).
    """
    global _XLFN_PATTERN
    import re
    if _XLFN_PATTERN is None:
        all_fns = sorted(_XLFN_FUNCTIONS, key=len, reverse=True)
        _XLFN_PATTERN = re.compile(
            r'(?<![A-Za-z0-9_.])(' + '|'.join(re.escape(f) for f in all_fns) + r')(?=\()',
        )

    def _replace(m: "re.Match") -> str:
        return "_xlfn." + m.group(1)

    try:
        wb = openpyxl.load_workbook(xlsx_path, data_only=False)
    except Exception as e:
        logger.warning(f"_add_xlfn_prefixes: could not open {xlsx_path}: {e}")
        return
    changed = False
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                v = cell.value
                if isinstance(v, str) and v.startswith("=") and "_xlfn" not in v:
                    new_v = _XLFN_PATTERN.sub(_replace, v)
                    if new_v != v:
                        cell.value = new_v
                        changed = True
    if changed:
        wb.save(xlsx_path)


def _force_iterative_calc(xlsx_path: str) -> None:
    """Force Excel's standard iterative-calculation settings on, unless
    already explicitly enabled.

    Root-caused via a 3-skeptic adversarial verification: several confirmed
    SSB v2 false negatives are circular-reference "scenario toggle" cells
    (a standard LBO/financial-model pattern, e.g. `=IF($C$2=3,$R31,AV20)`,
    directly referencing its own cell) whose golden.xlsx has
    `iterate="true"` explicitly set, while the PRISTINE INPUT given to the
    solver is missing the flag entirely (defaults to false) -- confirmed
    across 10 sibling bug-variant files sharing one base template: 8/10
    lack the flag, 2/10 retain it, an inconsistent split uncorrelated with
    any task's declared bug focus, matching the same class of preprocessing
    artifact already documented for cached-value wiping. Without iterative
    calculation, LibreOffice (and real Excel) cannot resolve a circular
    formula at all and errors with #VALUE! regardless of what the solver
    does to individual cell formulas -- proven directly by patching one
    confirmed case's calcPr to iterate=true and re-running the real
    ssb-verifier LibreOffice recalc, which resolved every #VALUE! to a
    correct, golden-matching number.

    Safe to apply unconditionally, not just when circularity is detected:
    enabling iterative calculation is a no-op for an acyclic dependency
    graph (Excel/LibreOffice only iterate when a cycle is actually
    encountered during evaluation) -- there is no formula-detection step
    that could get this wrong in a way that changes non-circular results.
    Uses Excel's own documented defaults (100 iterations, 0.001 max change)
    rather than golden's exact values, since production scoring must work
    without inspecting golden's internal settings for every task.

    A WORKBOOK WITH NO <calcPr> AT ALL was the case this function silently gave
    up on for its entire life. openpyxl exposes a missing calcPr as
    `wb.calculation is None`, so `wb.calculation.iterate` raised
    AttributeError, the `except` below logged a warning, and the function
    returned having done nothing. That is precisely the population it exists to
    serve: the docstring above records that 8 of 10 sibling bug-variant inputs
    LACK the flag, and openpyxl preserves the absence across a load/save, so a
    solver output derived from such an input never had one either. Confirmed
    2026-08-02 by running this function on the pristine
    Financial_Model/02_01 input: warning emitted, iterate still unset. Creating
    the element when it is missing is the whole fix.

    NEVER ROUND-TRIP THROUGH openpyxl HERE. `load_workbook(...).save(...)` blanks
    the cached <v> of EVERY formula cell in the file, and LibreOffice does not
    recompute a cell that still holds a value — so the cached values are what get
    graded wherever they survive. Verified 2026-08-03: a workbook with A1=10,
    A2='=A1*2' and a cached 20, with A1 patched to 100, still grades A2 as 20
    after the real recalc. The first version of this fix used openpyxl.save() and
    therefore destroyed every cached value on any file that lacked <calcPr> —
    trading the circular-reference false negative for a whole-model
    recomputation, which is the SAME false-negative class in a new costume, and
    on precisely the files this function exists to help. Measured: 3 of 169
    rollouts on the 30/30 split.
    So the flag is patched into `xl/workbook.xml` inside the zip and nothing else
    in the archive is touched.
    """
    import re as _re
    import shutil as _shutil
    import zipfile as _zipfile

    ITER_ATTRS = 'iterate="1" iterateCount="100" iterateDelta="0.001"'
    try:
        with _zipfile.ZipFile(xlsx_path) as z:
            items = {n: z.read(n) for n in z.namelist()}
        wbx = items["xl/workbook.xml"].decode("utf-8")
        m = _re.search(r"<calcPr\b[^>]*/>", wbx)
        if m:
            tag = m.group(0)
            if _re.search(r'iterate="(1|true)"', tag):
                return                      # already iterating; leave it alone
            new_tag = _re.sub(r'\s*iterate(Count|Delta)?="[^"]*"', "", tag)
            new_tag = new_tag[:-2].rstrip() + " " + ITER_ATTRS + "/>"
            wbx = wbx[:m.start()] + new_tag + wbx[m.end():]
        else:
            # No calcPr at all — the documented majority case (8 of 10 sibling
            # inputs). Insert one; calcId=124519 is what openpyxl writes for a
            # fresh workbook and an unknown calcId means "recalculate" to Excel.
            anchor = _re.search(r"<sheets\b", wbx)
            if not anchor:
                raise ValueError("workbook.xml has no <sheets> to anchor calcPr")
            tag = f'<calcPr calcId="124519" {ITER_ATTRS}/>'
            # calcPr must follow <sheets> in the schema, so insert after it.
            close = wbx.index("</sheets>") + len("</sheets>")
            wbx = wbx[:close] + tag + wbx[close:]
        items["xl/workbook.xml"] = wbx.encode("utf-8")
        tmp = str(xlsx_path) + ".calcpr.tmp"
        with _zipfile.ZipFile(tmp, "w", _zipfile.ZIP_DEFLATED) as zo:
            for n, b in items.items():
                zo.writestr(n, b)
        _shutil.move(tmp, xlsx_path)
    except Exception as e:
        # Still never fatal — scoring must proceed. But say what was LOST, not
        # just that an operation failed: the old wording ("could not open/save")
        # read like an I/O hiccup, which is why 357 rollouts' worth of this
        # warning class went unexamined.
        logger.warning(
            f"_force_iterative_calc: iterative calculation NOT applied to "
            f"{xlsx_path}: {e} — any circular-reference cell in this workbook "
            f"will recalculate to #VALUE! and score as wrong")


def _datetime_to_float(dt):
    excel_start_date = datetime.datetime(1899, 12, 30)
    delta = dt - excel_start_date
    return delta.days + delta.seconds / 86400.0


def _transform_value(v):
    if isinstance(v, (int, float)):
        v = round(float(v), 2)
    elif isinstance(v, datetime.time):
        v = str(v)[:-3]
    elif isinstance(v, datetime.datetime):
        v = round(_datetime_to_float(v), 0)
    elif isinstance(v, str):
        try:
            v = round(float(v), 2)
        except ValueError:
            pass
    return v


def _compare_cell_value(v1, v2) -> bool:
    v1 = _transform_value(v1)
    v2 = _transform_value(v2)
    if (v1 == "" and v2 is None) or (v1 is None and v2 == ""):
        return True
    if (v1 == "" and v2 == "") or (v1 is None and v2 is None):
        return True
    if type(v1) != type(v2):
        return False
    return v1 == v2


# Default Office theme colors (dk1, lt1, dk2, lt2, accent1-6), used only as a
# fallback if a workbook's own theme XML can't be parsed.
_DEFAULT_THEME_RGB = [
    "000000", "FFFFFF", "44546A", "E7E6E6",
    "4472C4", "ED7D31", "A5A5A5", "FFC000", "5B9BD5", "70AD47",
]

_THEME_CACHE: dict[int, list[str]] = {}


def _load_theme_colors(wb) -> list[str]:
    """Extract the 10 base theme colors from the workbook's theme XML.

    Order matches the OOXML clrScheme child order: dk1, lt1, dk2, lt2,
    accent1-6 — this is also the index space openpyxl's Color.theme uses.
    """
    cache_key = id(wb)
    if cache_key in _THEME_CACHE:
        return _THEME_CACHE[cache_key]
    colors = list(_DEFAULT_THEME_RGB)
    try:
        theme_xml = getattr(wb, "loaded_theme", None)
        if theme_xml:
            ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
            root = ET.fromstring(theme_xml)
            scheme = root.find(".//a:clrScheme", ns)
            if scheme is not None:
                order = ["dk1", "lt1", "dk2", "lt2", "accent1", "accent2",
                         "accent3", "accent4", "accent5", "accent6"]
                found = {}
                for child in scheme:
                    tag = child.tag.split("}")[-1]
                    srgb = child.find("a:srgbClr", ns)
                    sysclr = child.find("a:sysClr", ns)
                    if srgb is not None:
                        found[tag] = srgb.get("val")
                    elif sysclr is not None:
                        found[tag] = sysclr.get("lastClr")
                colors = [found.get(k, _DEFAULT_THEME_RGB[i]) for i, k in enumerate(order)]
    except Exception:
        pass
    _THEME_CACHE[cache_key] = colors
    return colors


def _apply_tint(rgb_hex: str, tint: float) -> str:
    """Apply an OOXML tint/shade factor to an RGB hex color (ECMA-376 HLS-based)."""
    if not tint:
        return rgb_hex
    try:
        r, g, b = (int(rgb_hex[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except (ValueError, IndexError):
        return rgb_hex
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    l = l * (1.0 + tint) if tint < 0 else l * (1.0 - tint) + tint
    l = min(max(l, 0.0), 1.0)
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "".join(f"{round(c * 255):02X}" for c in (r, g, b))


def _get_color_rgb(color, wb=None) -> str:
    """Resolve an openpyxl Color to a 6-hex-digit RGB string.

    Handles explicit RGB, theme-indexed (with tint), and legacy indexed-palette
    colors. Two different theme indices (or a theme color vs. its equivalent
    explicit RGB) previously both collapsed to the same '00000000' sentinel,
    producing both false passes (different themes read as equal) and false
    fails (a theme color vs. its own resolved RGB read as different).
    """
    if color is None:
        return "00000000"
    if isinstance(color.rgb, str) and color.rgb:
        return color.rgb
    # openpyxl's typed descriptors return the descriptor CLASS itself (e.g.
    # <class 'openpyxl.descriptors.base.Integer'>) as the "unset" sentinel for
    # .theme/.indexed on a Color that doesn't use that color type — not None.
    # `getattr(..., None)` doesn't catch this, so `is not None` passed and
    # `0 <= theme_idx` crashed with "'<=' not supported between instances of
    # 'int' and 'Integer'" on any cell whose color wasn't theme-based — this
    # is a real regression discovered live in the Jul 1 SSB v2 run (18 tasks
    # silently mis-scored as 0.0 via score_ssbv2's blanket except/return 0.0
    # before this was caught). Require isinstance(..., int) explicitly.
    theme_idx = getattr(color, "theme", None)
    if isinstance(theme_idx, int) and wb is not None:
        theme_colors = _load_theme_colors(wb)
        if 0 <= theme_idx < len(theme_colors):
            tint = getattr(color, "tint", 0.0)
            tint = tint if isinstance(tint, (int, float)) else 0.0
            return _apply_tint(theme_colors[theme_idx], tint)
    indexed = getattr(color, "indexed", None)
    if isinstance(indexed, int) and 0 <= indexed < len(COLOR_INDEX):
        return COLOR_INDEX[indexed]
    return "00000000"


def _compare_colors(color1, color2, wb1=None, wb2=None) -> bool:
    rgb1 = _get_color_rgb(color1, wb1)
    rgb2 = _get_color_rgb(color2, wb2)
    return rgb1[-6:] == rgb2[-6:]


def _compare_fill_color(fill1, fill2, wb1=None, wb2=None) -> bool:
    return _compare_colors(fill1.fgColor, fill2.fgColor, wb1, wb2) and _compare_colors(
        fill1.bgColor, fill2.bgColor, wb1, wb2
    )


def _compare_font_color(font_gt, font_proc, wb1=None, wb2=None) -> bool:
    return _compare_colors(font_gt.color, font_proc.color, wb1, wb2)


def _col_name2num(name: str) -> int:
    num = 0
    for c in name.upper():
        num = num * 26 + (ord(c) - ord("A") + 1)
    return num


def _parse_cell_range(range_str: str) -> tuple:
    """Parse 'A1:B10' into (min_row, min_col, max_row, max_col)."""
    if ":" in range_str:
        start, end = range_str.split(":")
    else:
        start = end = range_str

    import re
    m = re.match(r"([A-Z]+)(\d+)", start)
    start_col, start_row = _col_name2num(m.group(1)), int(m.group(2))
    m = re.match(r"([A-Z]+)(\d+)", end)
    end_col, end_row = _col_name2num(m.group(1)), int(m.group(2))
    return start_row, start_col, end_row, end_col


def _cell_level_compare(wb_gt, wb_proc, sheet_name: str, cell_range: str) -> bool:
    """Compare cells in a specific range between golden and processed workbooks."""
    if sheet_name not in wb_proc.sheetnames:
        return False

    ws_gt = wb_gt[sheet_name]
    ws_proc = wb_proc[sheet_name]

    min_row, min_col, max_row, max_col = _parse_cell_range(cell_range)

    for row in range(min_row, max_row + 1):
        for col in range(min_col, max_col + 1):
            gt_cell = ws_gt.cell(row=row, column=col)
            proc_cell = ws_proc.cell(row=row, column=col)

            if not _compare_cell_value(gt_cell.value, proc_cell.value):
                return False

            # Fill/font color are deliberately NOT part of scoring — confirmed
            # by direct comparison against the official evaluation script
            # (the upstream SpreadsheetBench-v2 evaluation.py:140-149),
            # where the equivalent calls are commented out by the benchmark's
            # own author. Our earlier port called these unconditionally,
            # scoring strictly stricter than the benchmark actually intends —
            # discovered live on Jul 1 when a 99.7%-cell-match solve (all
            # value mismatches traced to one root-cause formula error) still
            # scored 0, prompting a direct diff against the official script.
            # if not _compare_fill_color(gt_cell.fill, proc_cell.fill, wb_gt, wb_proc):
            #     return False
            # if not _compare_font_color(gt_cell.font, proc_cell.font, wb_gt, wb_proc):
            #     return False

    return True


def _split_answer_position(answer_position: str) -> list[str]:
    """Split answer_position on commas, respecting quoted sheet names.

    E.g. "'IS,BS,CF'!B3:K121,'Revenue'!B1:O23" -> ["'IS,BS,CF'!B3:K121", "'Revenue'!B1:O23"]
    """
    import re
    parts = []
    current = ""
    in_quote = False
    for ch in answer_position:
        if ch == "'" and not in_quote:
            in_quote = True
            current += ch
        elif ch == "'" and in_quote:
            in_quote = False
            current += ch
        elif ch == "," and not in_quote:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())
    return parts


def compare_workbooks_detailed(
    gt_file: str, proc_file: str, answer_position: str, max_mismatches: int = 20,
    skip_cells: set[tuple[str, str]] | None = None,
) -> tuple[bool, int, int, str]:
    """Compare workbooks and return detailed diff.

    Returns (passed, total_match, total_cells, diff_summary).
    diff_summary lists up to max_mismatches mismatched cells for evolver
    feedback; max_mismatches=0 means no cap (list every mismatch).
    Mismatches are collected in answer_position range order (not causal
    order), so on cascade failures the default 20 may show only downstream
    symptoms — re-run with max_mismatches=0 to see the full picture for a
    specific task.

    skip_cells: (sheet, coordinate) pairs to exclude ENTIRELY from grading
    (not counted in total_cells or total_match) -- for cells whose value is
    fundamentally unfalsifiable by a static-snapshot comparison, confirmed
    via adversarial audit: a bare `=TODAY()`/`=NOW()` cell, byte-identical
    between the original input and the solver's output (solver never
    touched it), will always mismatch golden's frozen cached value purely
    because each file was saved on a different calendar day -- this has
    nothing to do with solver or model correctness.
    """
    if not os.path.exists(proc_file):
        return False, 0, 0, "Output file not found"

    try:
        wb_gt = openpyxl.load_workbook(filename=gt_file, data_only=True)
        wb_proc = openpyxl.load_workbook(filename=proc_file, data_only=True)
    except Exception as e:
        return False, 0, 0, f"Cannot open workbook: {e}"

    total_match = 0
    total_cells = 0
    mismatches = []
    all_pass = True

    sheet_cell_ranges = _split_answer_position(answer_position)
    for sheet_cell_range in sheet_cell_ranges:
        if "!" in sheet_cell_range:
            sheet_name, cell_range = sheet_cell_range.rsplit("!", 1)
            sheet_name = sheet_name.strip("'")
        else:
            sheet_name = wb_gt.sheetnames[0]
            cell_range = sheet_cell_range
        cell_range = cell_range.strip("'")

        if sheet_name not in wb_proc.sheetnames:
            all_pass = False
            mismatches.append(f"Sheet '{sheet_name}' missing from output")
            continue

        import re
        ws_gt = wb_gt[sheet_name]
        ws_proc = wb_proc[sheet_name]
        try:
            min_row, min_col, max_row, max_col = _parse_cell_range(cell_range)
        except Exception:
            continue

        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                col_letter = ""
                c = col
                while c > 0:
                    c, r = divmod(c - 1, 26)
                    col_letter = chr(65 + r) + col_letter
                if skip_cells and (sheet_name, f"{col_letter}{row}") in skip_cells:
                    continue
                total_cells += 1
                gt_val = ws_gt.cell(row=row, column=col).value
                proc_val = ws_proc.cell(row=row, column=col).value
                if _compare_cell_value(gt_val, proc_val):
                    total_match += 1
                else:
                    all_pass = False
                    if max_mismatches <= 0 or len(mismatches) < max_mismatches:
                        mismatches.append(
                            f"'{sheet_name}'!{col_letter}{row}: "
                            f"expected={_transform_value(gt_val)!r}, "
                            f"got={_transform_value(proc_val)!r}"
                        )

    diff_summary = "; ".join(mismatches) if mismatches else "All cells match"
    return all_pass, total_match, total_cells, diff_summary


def _formulas_equivalent(proc_formula, orig_formula, in_table_range: bool = False) -> bool:
    """Compare a post-recalc formula against the original, tolerating
    openpyxl's DataTableFormula object representation.

    A Data Table's ANCHOR cell in the pristine input loads as an
    `openpyxl.worksheet.formula.DataTableFormula` object (ref/r1/r2
    attributes, not a string); every OTHER cell in that same table's range
    loads as a plain cached number with no formula marker at all -- it
    isn't even a string, so it can never equal anything the proc file
    holds after recalc. After LibreOffice round-trips the file, ALL cells
    in the range (anchor and siblings alike) become the plain string
    '=TABLE($K$7,$C$25,...)' -- confirmed empirically on
    Financial_Model/19_02's DCF!L8:P12. A strict `==` between these
    representations would reject every real case this safety check exists
    to catch, for both the anchor (object vs string type mismatch) and the
    siblings (number vs string, and the number itself carries no evidence
    of "this used to be part of a formula" without the caller's
    `in_table_range` hint from a whole-sheet anchor scan).

    `in_table_range` is True when this cell's coordinates fall inside a
    DataTableFormula anchor's expanded `.ref` range found anywhere in the
    original sheet (the caller does that scan once per sheet). Combined
    with "proc is a TABLE(...) string," this is safe: the solver cannot
    construct a DataTableFormula object via openpyxl (no such API exists
    for writing one), so a proc-side TABLE() string at a location the
    ORIGINAL already flagged as part of a data table can only be
    LibreOffice's own re-serialization of that pre-existing construct,
    never something the solver introduced.
    """
    if proc_formula == orig_formula:
        return True
    proc_is_table_str = isinstance(proc_formula, str) and "TABLE(" in proc_formula.upper()
    if not proc_is_table_str:
        return False
    orig_is_table_anchor = type(orig_formula).__name__ == "DataTableFormula"
    return orig_is_table_anchor or in_table_range


# FILTER/SORT/TEXTSPLIT remain genuinely broken in LibreOffice even with a
# correct _xlfn. prefix (confirmed by direct test, see _XLFN_FUNCTIONS'
# comment above) -- a real array-spill engine gap, not a naming issue.
_ARRAY_SPILL_FUNCTIONS = ("FILTER(", "SORT(", "TEXTSPLIT(")


def _is_array_spill_formula(formula) -> bool:
    if not isinstance(formula, str):
        return False
    upper = formula.upper()
    return any(f"_XLFN.{fn}" in upper or fn in upper for fn in _ARRAY_SPILL_FUNCTIONS)


def _restore_unrecalculable_cells(
    proc_file: str, orig_input_file: str, answer_position: str, gt_file: str | None = None
) -> None:
    """Restore original cached values LibreOffice's recalc destroyed.

    Root cause (confirmed by hand on Financial_Model/10_03 and via a
    129-task audit of SSB v2 test-set failures -- 10/129 failures traced
    entirely to this): some Excel formulas (`=TABLE(...)` data tables,
    FILTER/SORT/TEXTSPLIT array-spill functions) cannot be evaluated by
    LibreOffice at all. openpyxl never writes a cached value for a formula
    it didn't itself compute, and openpyxl WIPES any pre-existing cached
    value on save even for cells it never touches (confirmed empirically --
    round-tripping a file through openpyxl with a no-op cell reassignment
    elsewhere in the sheet blanks every formula's cached value file-wide).
    So if the solver preserves one of these formulas UNCHANGED from the
    original input (the correct thing to do -- it has no way to compute a
    Data Table's 25 scenario values without reimplementing the model), the
    scorer's own mandatory recalc step is what destroys the already-correct
    cached value, not the solver.

    Fix: for every graded cell where (a) the solver's saved formula is
    BYTE-IDENTICAL to the original input's formula at that cell (solver
    didn't touch it) and (b) the original input has a real, non-None cached
    value there (it was last saved by real Excel, which DID compute it),
    and (c) the post-recalc value is None (LibreOffice couldn't evaluate
    it) -- restore the original's cached value as a literal. This only
    ever restores values LibreOffice's recalc actively DESTROYED on
    formulas the solver never touched; it can't mask a genuine solver
    error, since a cell the solver modified (or one whose original value
    was itself None) is left alone.

    Scoped to just the graded answer_position ranges for speed -- these
    workbooks can have 80+ sheets and this comparison would otherwise be
    the slowest part of scoring.

    gt_file (golden-aware guard, added after an empirical audit found this
    function had NO visibility into golden at all -- unlike
    _solve_datatables_with_uno's regression-aware safety net, it could
    silently turn an already-correct cell wrong with zero detection: if the
    graded cell's correct answer legitimately CHANGED as a result of the
    solver's fix (e.g. a Data Table whose row/column inputs the fix
    touched), but the solver correctly left the Data Table's own formula
    untouched (byte-identical to the original, satisfying (a) below), this
    function would restore the ORIGINAL's stale PRE-FIX cached value over a
    freshly-correct post-fix computation. Spot-checked on 2 real solved
    outputs with successful UNO fixes (Debugging/02_04, Debugging/02_07):
    0 clobbers found in either -- but the restore loop itself has no way to
    know this without checking golden, so absence of gt_file cannot be
    trusted to mean absence of the failure mode on tasks not yet tested.
    When gt_file is supplied, a per-cell guard skips only the specific
    restoration that would turn a currently-matching cell into a
    non-matching one; when omitted, falls back to the original unconditional
    behavior (over-restores in the specific rare case above, never
    under-restores).
    """
    try:
        wb_orig_formulas = openpyxl.load_workbook(orig_input_file, data_only=False)
        wb_orig_values = openpyxl.load_workbook(orig_input_file, data_only=True)
        wb_proc_formulas = openpyxl.load_workbook(proc_file, data_only=False)
        wb_proc_values = openpyxl.load_workbook(proc_file, data_only=True)
        wb_gt_values = openpyxl.load_workbook(gt_file, data_only=True) if gt_file else None
    except Exception as e:
        logger.warning(f"_restore_unrecalculable_cells: could not open workbooks: {e}")
        return

    from openpyxl.utils.cell import range_boundaries

    changed = False
    for sheet_cell_range in _split_answer_position(answer_position):
        if "!" in sheet_cell_range:
            sheet_name, cell_range = sheet_cell_range.rsplit("!", 1)
            sheet_name = sheet_name.strip("'")
        else:
            sheet_name = wb_orig_formulas.sheetnames[0]
            cell_range = sheet_cell_range
        cell_range = cell_range.strip("'")

        if sheet_name not in wb_proc_formulas.sheetnames or sheet_name not in wb_orig_formulas.sheetnames:
            continue
        try:
            min_row, min_col, max_row, max_col = _parse_cell_range(cell_range)
        except Exception:
            continue

        ws_orig_f = wb_orig_formulas[sheet_name]
        ws_orig_v = wb_orig_values[sheet_name]
        ws_proc_f = wb_proc_formulas[sheet_name]
        ws_proc_v = wb_proc_values[sheet_name]

        # Data Tables: openpyxl only tags the ANCHOR cell with a
        # DataTableFormula object (ref/r1/r2 attrs); every other cell in
        # the table's range loads as a plain cached number with no formula
        # marker at all (confirmed empirically: Financial_Model/19_02's
        # DCF!L8 is the anchor, M8:P12 are plain floats). Pre-scan the
        # whole sheet once for anchors and expand their `.ref` into a set
        # of member cells, so sibling cells get the same restoration
        # eligibility as the anchor even though they carry no formula
        # object of their own to compare against.
        table_member_cells: set[tuple[int, int]] = set()
        for row in ws_orig_f.iter_rows():
            for cell in row:
                if type(cell.value).__name__ == "DataTableFormula":
                    try:
                        min_c, min_r, max_c, max_r = range_boundaries(cell.value.ref)
                    except Exception:
                        continue
                    for rr in range(min_r, max_r + 1):
                        for cc in range(min_c, max_c + 1):
                            table_member_cells.add((rr, cc))

        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                in_table_range = (row, col) in table_member_cells
                proc_formula = ws_proc_f.cell(row=row, column=col).value
                orig_formula = ws_orig_f.cell(row=row, column=col).value
                # FILTER/SORT/TEXTSPLIT array-spill formulas fail LibreOffice
                # recalc as a genuine #NAME? error STRING, not None -- this
                # function's own docstring already claimed to cover them, but
                # the gate below only ever widened for in_table_range cells,
                # so a non-None '#NAME?' on one of these was never actually
                # eligible for restoration (confirmed as an active bug: this
                # is the exact false-negative class the whole mechanism
                # exists to catch, just for a different unrecalculable-
                # formula family than Data Tables). Detected directly on the
                # formula text (with or without an _xlfn. prefix) since,
                # unlike Data Tables, there's no separate anchor-object
                # marker to scan for -- only the cell's own formula string.
                in_unrecalculable_range = in_table_range or _is_array_spill_formula(proc_formula) or _is_array_spill_formula(orig_formula)
                proc_val = ws_proc_v.cell(row=row, column=col).value
                # Outside an unrecalculable-formula cell, a non-None recalc
                # result is trustworthy -- leave it alone. INSIDE one,
                # LibreOffice's "successful" (non-None) recalc is NOT
                # trustworthy: confirmed empirically (Debugging/02_07,
                # Debugging/02_04) that it doesn't implement Excel's Data
                # Table row/column substitution at all, and separately
                # confirmed FILTER/SORT/TEXTSPLIT recalc to a #NAME? error
                # string rather than None. So for these cells we still
                # attempt restoration even when proc_val is a concrete
                # (wrong) value or an error string, not just None.
                if proc_val is not None and not in_unrecalculable_range:
                    continue  # recalc produced a real, trustworthy value
                orig_val = ws_orig_v.cell(row=row, column=col).value
                if orig_val is None:
                    continue  # original had nothing cached either -- not this bug
                if not _formulas_equivalent(proc_formula, orig_formula, in_table_range=in_table_range):
                    continue  # solver modified this cell -- a real solver issue, don't mask it
                if wb_gt_values is not None and sheet_name in wb_gt_values.sheetnames:
                    gt_val = wb_gt_values[sheet_name].cell(row=row, column=col).value
                    proc_already_correct = _compare_cell_value(gt_val, proc_val)
                    orig_would_be_correct = _compare_cell_value(gt_val, orig_val)
                    if proc_already_correct and not orig_would_be_correct:
                        # Restoring here would turn an already-correct cell
                        # (recalc or UNO fix legitimately computed the new
                        # post-fix answer) into a wrong one (the original's
                        # stale pre-fix value) -- skip just this cell.
                        continue
                ws_proc_v.cell(row=row, column=col, value=orig_val)
                changed = True

    if changed:
        # Save via the VALUES workbook (wb_proc_values, data_only=True), not
        # the formulas one. This looks backwards at first -- a data_only=True
        # save does turn every formula in the file into a bare literal -- but
        # that is harmless here, since it preserves every cell's CURRENT
        # VALUE exactly (that is the entire point of data_only=True: it
        # already holds each cell's computed result) and
        # compare_workbooks_detailed only ever reads values, never formula
        # text, downstream.
        #
        # The alternative (saving wb_proc_formulas, data_only=False) looks
        # safer but is actually WORSE for this pipeline: a data_only=False
        # load never captures a cell's computed value in the first place, so
        # saving it drops the cached <v> for every UNTOUCHED formula cell in
        # the entire workbook, and re-reading with data_only=True afterward
        # (which is what scoring does) sees None everywhere a real formula
        # used to be -- confirmed empirically as a genuine regression on a
        # real file (260 -> 746 newly-wrong cells, none of them anywhere
        # near a Data Table, from this save call alone).
        #
        # This was gotten backwards once already: an earlier
        # "fix" switched to wb_proc_formulas.save() after repeatedly
        # re-scoring the SAME on-disk file by hand during manual
        # investigation produced different failing cells run-to-run. That
        # symptom was a testing-methodology artifact (once a Data Table
        # formula gets flattened to a literal by the first save, later
        # passes have nothing left to recalculate and just preserve
        # whatever was frozen in) it was never a real bug in the original
        # single-pass save, which is exactly what production scoring always
        # does (score_and_diff runs exactly once per task per iteration).
        wb_proc_values.save(proc_file)


_VOLATILE_ZERO_ARG_FUNCTIONS = ("TODAY()", "NOW()")


def _volatile_function_skip_cells(orig_input_file: str, proc_file: str, answer_position: str) -> set[tuple[str, str]]:
    """Graded cells to exclude entirely: a bare zero-argument volatile
    function (=TODAY(), =NOW()) that the solver left byte-identical to the
    original -- its value is fundamentally unfalsifiable against a static,
    pre-frozen golden file, since each file's cached value simply reflects
    whichever calendar day/time it happened to be saved on. Confirmed via
    adversarial audit (Template/14_07): golden snapshot 2026-05-27, solver's
    real recalc 2026-07-06, formula identical across original/golden/solver
    -- nothing to do with correctness. Deliberately narrow (exact
    zero-argument call only, not e.g. TODAY()+30, which legitimately
    depends on the solver having the right base formula) and does NOT
    require the FORMULA to also match golden -- if the solver touched a
    volatile-function cell at all, whatever it changed it to is a real,
    gradable edit, not this exemption.
    """
    try:
        wb_orig_f = openpyxl.load_workbook(orig_input_file, data_only=False)
        wb_proc_f = openpyxl.load_workbook(proc_file, data_only=False)
    except Exception as e:
        logger.warning(f"_volatile_function_skip_cells: could not open workbooks: {e}")
        return set()

    skip = set()
    for sheet_cell_range in _split_answer_position(answer_position):
        if "!" in sheet_cell_range:
            sheet_name, cell_range = sheet_cell_range.rsplit("!", 1)
            sheet_name = sheet_name.strip("'")
        else:
            sheet_name = wb_orig_f.sheetnames[0]
            cell_range = sheet_cell_range
        cell_range = cell_range.strip("'")

        if sheet_name not in wb_orig_f.sheetnames or sheet_name not in wb_proc_f.sheetnames:
            continue
        try:
            min_row, min_col, max_row, max_col = _parse_cell_range(cell_range)
        except Exception:
            continue

        ws_orig_f = wb_orig_f[sheet_name]
        ws_proc_f = wb_proc_f[sheet_name]
        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                orig_formula = ws_orig_f.cell(row=row, column=col).value
                proc_formula = ws_proc_f.cell(row=row, column=col).value
                if not isinstance(orig_formula, str) or proc_formula != orig_formula:
                    continue
                bare = orig_formula.lstrip("=").upper()
                if bare in _VOLATILE_ZERO_ARG_FUNCTIONS:
                    c = col
                    col_letter = ""
                    while c > 0:
                        c, r = divmod(c - 1, 26)
                        col_letter = chr(65 + r) + col_letter
                    skip.add((sheet_name, f"{col_letter}{row}"))
    return skip


def _datatable_target_cells(xlsx_path: str) -> list[tuple[str, str]]:
    """(sheet_title, coordinate) for every cell holding a live TABLE(...)
    formula or DataTableFormula object. Empty list if the file has none or
    can't be opened."""
    try:
        wb = openpyxl.load_workbook(xlsx_path, data_only=False)
    except Exception:
        return []
    out = []
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                v = cell.value
                if (isinstance(v, str) and "TABLE(" in v.upper()) or type(v).__name__ == "DataTableFormula":
                    out.append((ws.title, cell.coordinate))
    return out


def _has_live_datatable(xlsx_path: str) -> bool:
    try:
        wb = openpyxl.load_workbook(xlsx_path, data_only=False)
    except Exception:
        return False
    return any(
        (isinstance(cell.value, str) and "TABLE(" in cell.value.upper())
        or type(cell.value).__name__ == "DataTableFormula"
        for ws in wb.worksheets for row in ws.iter_rows() for cell in row
    )


def _prepare_normalized_datatable_scratch(orig_input_file: str) -> str:
    """Copy orig_input_file to a scratch path and normalize it exactly the
    way a Data-Table scan needs: add missing _xlfn. prefixes, force
    iterative calc on, then a LibreOffice recalc. Per
    _solve_datatables_with_uno's docstring, this recalc is what turns a
    Data Table's ANCHOR cell from an openpyxl DataTableFormula OBJECT into
    the plain string every member cell in the range carries post-recalc
    (e.g. '=TABLE($F$9,$C$12,$F10,$C$15,G$9)') -- confirmed empirically:
    scanning Debugging/02_04's original input BEFORE this normalization
    finds only 4 anchor cells; scanning the SAME file AFTER it finds all
    100 member cells across the same 4 tables. Callers that need a Data
    Table's full member-cell set (not just its anchors) MUST scan the
    returned scratch file, never the raw orig_input_file directly -- see
    _datatable_target_cells and the bug this fixed in
    _build_hybrid_golden_with_original_datatable below.

    Caller owns cleanup of the returned path (not removed here).
    """
    scratch = f"{orig_input_file}.fresh_datatable_scratch.{uuid.uuid4().hex[:8]}.xlsx"
    shutil.copy2(orig_input_file, scratch)
    _add_xlfn_prefixes(scratch)
    _force_iterative_calc(scratch)
    _recalculate_with_libreoffice(scratch)
    return scratch


# ---------------------------------------------------------------------------
# Formula dependency tracing (replaces the old whole-sheet cell-agreement
# heuristic below with a precise, per-cell BFS over the actual formula
# graph). See _datatable_dependency_safe for the entry point.
# ---------------------------------------------------------------------------

# INDIRECT/ADDRESS build a reference from a runtime string; OFFSET computes
# a moving target -- none of these are statically determinable by regex, so
# any formula containing one is unconditionally opaque.
_OPAQUE_FORMULA_FN_RE = re.compile(r"(?:INDIRECT|OFFSET|ADDRESS)\s*\(", re.IGNORECASE)

# A literal Excel error value embedded in formula text (most commonly #REF!,
# left behind when a referenced cell/sheet was deleted). Confirmed live in
# real data: Debugging/02's "Exhibit 9" sheet has formulas like
# "=(N20-N18)/(#REF!+N20-N18)" -- there is no real cell behind an error
# literal to trace, and a formula containing one is often already damaged
# in a way a static regex cannot reason about. (Even without this explicit
# check, the bareword-classification pass below already rejects these
# tokens too -- e.g. "REF" out of "#REF!" is not a function/name/keyword --
# but it is called out explicitly here rather than relied on as a side
# effect of an unrelated code path.)
_ERROR_LITERAL_RE = re.compile(r"#(?:REF|N/A|VALUE|DIV/0|NAME|NULL|NUM)[!?]", re.IGNORECASE)

# A whole-column ('B:B') or whole-row ('9:9') reference, optionally
# sheet-qualified. Confirmed live in real data (Debugging/02's DCF/WACC
# sheets use e.g. "INDEX('Exhibit 6'!D:D, MATCH(...))"). Expanding one to
# individual cells is unbounded/impractical (a single column is 1,048,576
# cells) and Excel's own semantics for it depend on the sheet's used range,
# which this module has no reliable way to determine -- treated as opaque
# rather than guessed at.
_FULL_COLROW_RE = re.compile(
    r"(?<![A-Za-z0-9_.])"
    r"(?:'(?:[^']|'')+'!|[A-Za-z_][A-Za-z0-9_.]*!)?"
    r"(?:\$?[A-Za-z]{1,3}:\$?[A-Za-z]{1,3}"
    r"|\$?[0-9]{1,7}:\$?[0-9]{1,7})"
    r"(?![A-Za-z0-9_])"
)

_STRING_LITERAL_RE = re.compile(r'"(?:[^"]|"")*"')

# A single cell or same-sheet-qualified range reference, with an optional
# cross-sheet qualifier (quoted -- sheet names may contain commas, spaces,
# ampersands, e.g. this exact dataset's "'PP&E, Debt, NWC'" -- or bare for
# simple names). Left/right boundary lookarounds prevent matching PART of a
# larger identifier (e.g. "TAX10A" must never be read as ref "TAX10" plus a
# stray "A"); the trailing negative lookahead for "(" prevents matching a
# function name that happens to look like a cell ref (e.g. "LOG10(").
_REF_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9_.])"
    r"(?:(?P<sheet>'(?:[^']|'')+'|[A-Za-z_][A-Za-z0-9_.]*)!)?"
    r"(?P<c1col>\$?[A-Za-z]{1,3})(?P<c1row>\$?[0-9]{1,7})"
    r"(?::(?P<c2col>\$?[A-Za-z]{1,3})(?P<c2row>\$?[0-9]{1,7}))?"
    r"(?![A-Za-z0-9_.(])"
)

# Any bareword identifier -- used, after the ref pass above, to find
# leftover tokens that need classifying as a function call, a boolean
# keyword, or a candidate named range.
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")

_EXCEL_BOOL_KEYWORDS = {"TRUE", "FALSE"}

_MAX_EXCEL_COL = 16384   # XFD, Excel 2007+'s max column
_MAX_EXCEL_ROW = 1048576  # Excel 2007+'s max row

_FORMULA_REF_RECURSION_LIMIT = 10  # named-range-resolution recursion guard


def _col_letters_to_num(letters: str) -> int:
    num = 0
    for c in letters.upper():
        num = num * 26 + (ord(c) - ord("A") + 1)
    return num


def _col_num_to_letters(num: int) -> str:
    letters = ""
    while num > 0:
        num, r = divmod(num - 1, 26)
        letters = chr(65 + r) + letters
    return letters


def _strip_string_literals(formula: str) -> str:
    '''Remove Excel double-quoted string literals, honoring the doubled
    double-quote escape Excel uses for a literal quote inside a string
    (so a formula like =SUM(A1)&"a ""quoted"" word" is recognized as ONE
    string literal containing an embedded quote, not two literals split
    around a bare "quoted"). Each literal is replaced with equal-length
    spaces (rather than deleted) so no two originally non-adjacent tokens
    can end up touching after the strip.
    '''
    return _STRING_LITERAL_RE.sub(lambda m: " " * len(m.group(0)), formula)


def _extract_formula_refs(
    formula: str, current_sheet: str, wb, _depth: int = 0,
) -> "set[tuple[str, str]] | None":
    """Every (sheet, cell_coordinate) a formula string depends on, with any
    range reference expanded to individual cells.

    Returns an EMPTY SET if the formula was fully, safely parsed and
    genuinely has zero cell dependencies (e.g. a bare string literal or a
    pure numeric-literal expression). Returns None if the formula could NOT
    be safely, completely parsed -- callers MUST treat None as "unknown
    dependency, fail closed," never as "no dependency." Getting this
    distinction backwards would make the whole dependency-safety mechanism
    silently unsafe.

    Deliberately conservative in several places (opaque functions,
    whole-column/row refs, unresolvable barewords all bail to None rather
    than guess) -- a missed real dependency is the failure mode that
    matters here, an unnecessary refusal is merely inconvenient. Verified
    against real formulas pulled from actual SSB v2 files before being
    wired into any safety-relevant path (Debugging/02's two input files:
    714/720 formulas each fully resolved, the remaining 6 correctly bailed
    to None -- 3 whole-column INDEX/MATCH lookups, 3 literal #REF! error
    fragments; Financial_Model/17's ~78k-formula FMCG model: 0 None, 0
    exceptions, spot-checked by eye).
    """
    if _depth > _FORMULA_REF_RECURSION_LIMIT:
        return None
    if not isinstance(formula, str) or not formula.startswith("="):
        return set()

    stripped = _strip_string_literals(formula)
    if '"' in stripped:
        # Unterminated/malformed string literal -- can't trust the strip.
        return None
    if _ERROR_LITERAL_RE.search(stripped):
        return None
    if "[" in stripped or "]" in stripped:
        # External workbook refs ('[Book.xlsx]Sheet1'!A1) or Excel Table
        # structured references ('Table1[Column1]') -- neither is handled
        # by this parser; fail closed rather than risk mis-parsing either.
        return None
    if _OPAQUE_FORMULA_FN_RE.search(stripped):
        return None
    if _FULL_COLROW_RE.search(stripped):
        return None

    refs: "set[tuple[str, str]]" = set()
    consumed_spans: list[tuple[int, int]] = []

    for m in _REF_TOKEN_RE.finditer(stripped):
        sheet_group = m.group("sheet")
        if sheet_group is not None:
            if sheet_group.startswith("'") and sheet_group.endswith("'"):
                raw_sheet = sheet_group[1:-1].replace("''", "'")
            else:
                raw_sheet = sheet_group
            resolved_sheet = None
            for name in wb.sheetnames:
                if name.lower() == raw_sheet.lower():
                    resolved_sheet = name
                    break
            if resolved_sheet is None:
                # Can't confidently resolve this cross-sheet reference (a
                # renamed/deleted sheet, or a mis-parse) -- fail closed for
                # the WHOLE formula rather than silently drop one ref.
                return None
            sheet_name = resolved_sheet
        else:
            sheet_name = current_sheet

        c1col = m.group("c1col").lstrip("$")
        c1row = m.group("c1row").lstrip("$")
        c2col_g = m.group("c2col")
        c2row_g = m.group("c2row")

        col1_num = _col_letters_to_num(c1col)
        row1_num = int(c1row)
        if not (1 <= col1_num <= _MAX_EXCEL_COL and 1 <= row1_num <= _MAX_EXCEL_ROW):
            consumed_spans.append((m.start(), m.end()))
            continue  # impossible ref -- sanity backstop, skip silently

        if c2col_g is not None:
            c2col = c2col_g.lstrip("$")
            c2row = c2row_g.lstrip("$")
            col2_num = _col_letters_to_num(c2col)
            row2_num = int(c2row)
            if not (1 <= col2_num <= _MAX_EXCEL_COL and 1 <= row2_num <= _MAX_EXCEL_ROW):
                consumed_spans.append((m.start(), m.end()))
                continue
            min_col, max_col = sorted((col1_num, col2_num))
            min_row, max_row = sorted((row1_num, row2_num))
            for r in range(min_row, max_row + 1):
                for c in range(min_col, max_col + 1):
                    refs.add((sheet_name, f"{_col_num_to_letters(c)}{r}"))
        else:
            refs.add((sheet_name, f"{_col_num_to_letters(col1_num)}{row1_num}"))

        consumed_spans.append((m.start(), m.end()))

    # Leftover bareword identifiers: candidate named ranges, function
    # names, or keywords not already consumed by a ref match above (this
    # also naturally covers the sheet-name portion of a cross-sheet ref,
    # e.g. "Sheet1" inside "Sheet1!A1", since that whole span is consumed).
    for m in _IDENTIFIER_RE.finditer(stripped):
        tok_start, tok_end = m.start(), m.end()
        if any(cs <= tok_start and tok_end <= ce for cs, ce in consumed_spans):
            continue
        token = m.group(0)
        if stripped[tok_end:tok_end + 1] == "(":
            continue  # function call syntax -- not a dependency
        if token.upper() in _EXCEL_BOOL_KEYWORDS:
            continue
        # Numeric literals never match _IDENTIFIER_RE (it requires a
        # leading letter/underscore), so no separate numeric check needed.
        defined_name = None
        for name in wb.defined_names:
            if name.lower() == token.lower():
                defined_name = wb.defined_names[name]
                break
        if defined_name is None:
            # Can't classify this bareword confidently -- fail closed.
            return None
        attr_text = defined_name.attr_text
        if attr_text is None:
            return None
        sub_formula = attr_text if attr_text.startswith("=") else "=" + attr_text
        # A workbook-scoped defined name's own target has no single
        # "current sheet" of its own in general; resolve any relative refs
        # inside it against the sheet of the formula THAT USED the name --
        # matches Excel's behavior for the common case (a name pointing at
        # a sheet-qualified range). An unqualified same-sheet target inside
        # a defined name is rare/discouraged and not specially handled.
        sub_refs = _extract_formula_refs(sub_formula, current_sheet, wb, _depth=_depth + 1)
        if sub_refs is None:
            return None
        refs |= sub_refs

    return refs


def _strip_xlfn_prefix(formula: str) -> str:
    """Strip a leading _xlfn./_xlfn._xlws. prefix (case-insensitive) so a
    purely cosmetic OOXML serialization difference between two formula
    strings isn't reported as a real divergence. See the _XLFN_FUNCTIONS
    comment above for why these prefixes exist at all."""
    for prefix in ("_xlfn._xlws.", "_xlfn."):
        if formula.upper().startswith(prefix.upper()):
            return formula[len(prefix):]
    return formula


_DATATABLE_DEPENDENCY_BFS_NODE_CAP = 20000
_DATATABLE_DEPENDENCY_BFS_TIME_CAP_SECONDS = 120


def _datatable_dependency_safe(
    gt_file: str, orig_input_file: str, answer_position: str,
) -> tuple[bool, str]:
    """Precisely determine whether it's safe to splice a Data Table
    recomputed from the PRISTINE ORIGINAL input into golden (see
    _refresh_golden_datatables' "ORIGINAL-INPUT FALLBACK"), by tracing the
    table's actual upstream formula dependency chain instead of a
    whole-sheet statistical proxy.

    Replaces the old _golden_vs_original_agreement + 98% threshold, which
    could not distinguish "this sheet has unrelated changes elsewhere" from
    "this table's own inputs changed" -- confirmed live: on Debugging/02_04
    and 02_07, the whole-sheet-scoped version of that check measured 88.7%
    and 97.1% agreement respectively (both below threshold, refusing the
    fallback), even though this function's precise trace shows the actual
    reason has nothing to do with either task's own fix: the Data Table's
    row-input chain runs DCF!C9 -> WACC!C22 -> WACC!C7, and WACC!C7 is
    '=INDEX(\\'Exhibit 6\\'!D:D, MATCH(...))' -- an unprovable whole-column
    lookup, not a cell either task's injected bug or fix touches (spot
    checked directly: 'Exhibit 6' is byte-for-byte identical between golden
    and original, 232/232 non-blank cells match). This function correctly
    identifies and NAMES that one specific unresolvable cell instead of
    reporting a bare, undebuggable percentage -- a strictly more precise
    (if, for these two tasks, equally conservative) result.

    Algorithm:
      1. Seeds = the Data Table's own member cells' formula args (the 5
         corner/row-input/row-header/col-input/col-header refs each
         TABLE(...) cell carries, see _solve_datatables_with_uno's
         docstring) union the member cells' own coordinates.
      2. BFS the seeds' transitive upstream dependencies over the PRISTINE
         original input's own formula graph (never a recalculated/UNO-
         solved copy -- this comparison must be immune to LibreOffice
         recalc noise, the same class of problem the _transform_value
         rounding fix and the old agreement check both had to work around).
      3. Any cell whose formula can't be SAFELY, COMPLETELY parsed (an
         opaque function, an unresolvable name, a whole-column/row ref, ...)
         makes the whole trace opaque -- fail closed.
      4. Every visited cell's formula text (xlfn-prefix-normalized) or,
         for a non-formula cell, its cached value, must match between
         golden and original exactly. Any divergence -- fail closed, and
         name the exact cell.

    Known limitation (the acceptable, over-conservative direction): this
    is a purely TEXTUAL formula comparison, so two syntactically different
    formulas that happen to be semantically equivalent (e.g. `=A1+A2` vs
    `=SUM(A1:A2)`) would be reported as a false divergence. The dangerous
    direction -- silently treating a genuinely different chain as safe
    (false-safe) -- is what every design choice above (fail-closed on any
    parse uncertainty, the node/time caps below, no cross-file recalc) is
    built to prevent.
    """
    scratch = None
    try:
        # ---- Step 1: seeds ----
        if not _has_live_datatable(orig_input_file):
            return False, "unsafe: original input has no live Data Table to trace dependencies from"

        scratch = _prepare_normalized_datatable_scratch(orig_input_file)
        target_cells = _datatable_target_cells(scratch)
        if not target_cells:
            return False, "unsafe: no Data Table member cells found in original input after normalization"

        wb_scratch = openpyxl.load_workbook(scratch, data_only=False)
        target_cells_set = set(target_cells)
        seeds: "set[tuple[str, str]]" = set(target_cells_set)
        for sheet_name, coord in target_cells:
            if sheet_name not in wb_scratch.sheetnames:
                return False, f"unsafe: sheet {sheet_name!r} vanished while normalizing {orig_input_file}"
            formula = wb_scratch[sheet_name][coord].value
            if not (isinstance(formula, str) and formula.startswith("=")):
                return False, (
                    f"unsafe: {sheet_name}!{coord} has no TABLE() formula string after "
                    f"normalization (got {type(formula).__name__}); cannot trace its inputs"
                )
            refs = _extract_formula_refs(formula, sheet_name, wb_scratch)
            if refs is None:
                return False, f"unsafe: could not parse Data Table formula at {sheet_name}!{coord} ({formula!r})"
            seeds |= refs

        try:
            os.remove(scratch)
        except OSError:
            pass
        scratch = None

        # ---- Steps 2-3: BFS over the PRISTINE original's formula-view ----
        wb_orig_f = openpyxl.load_workbook(orig_input_file, data_only=False)
        wb_orig_v = openpyxl.load_workbook(orig_input_file, data_only=True)
        wb_gt_f = openpyxl.load_workbook(gt_file, data_only=False)
        wb_gt_v = openpyxl.load_workbook(gt_file, data_only=True)

        start_time = time.monotonic()
        visited: "set[tuple[str, str]]" = set()
        opaque: "list[tuple[str, str]]" = []
        queue = list(seeds)
        hit_cap = False
        while queue:
            if (
                len(visited) >= _DATATABLE_DEPENDENCY_BFS_NODE_CAP
                or (time.monotonic() - start_time) > _DATATABLE_DEPENDENCY_BFS_TIME_CAP_SECONDS
            ):
                hit_cap = True
                break
            sheet_name, coord = queue.pop()
            if (sheet_name, coord) in visited:
                continue
            visited.add((sheet_name, coord))

            if sheet_name not in wb_orig_f.sheetnames:
                opaque.append((sheet_name, coord))
                continue
            try:
                formula = wb_orig_f[sheet_name][coord].value
            except Exception:
                opaque.append((sheet_name, coord))
                continue

            if isinstance(formula, str) and formula.startswith("="):
                refs = _extract_formula_refs(formula, sheet_name, wb_orig_f)
                if refs is None:
                    opaque.append((sheet_name, coord))
                    continue
                for r in refs:
                    if r not in visited:
                        queue.append(r)
            elif type(formula).__name__ == "DataTableFormula":
                if (sheet_name, coord) not in target_cells_set:
                    # A DIFFERENT Data Table's anchor, reached transitively.
                    # There's no live formula string to extract refs from on
                    # the raw (unnormalized) file for THIS OTHER table, so
                    # its own inputs can't be traced -- fail closed rather
                    # than assume it's fine.
                    opaque.append((sheet_name, coord))
                # else: our own table's anchor -- its deps were already
                # seeded directly in Step 1 from the normalized scratch;
                # nothing left to expand from the raw object here.
            # else: plain literal or None -- leaf, nothing to expand.

        if hit_cap:
            return False, (
                f"unsafe: dependency closure exceeded node cap ({_DATATABLE_DEPENDENCY_BFS_NODE_CAP}) "
                f"or time cap ({_DATATABLE_DEPENDENCY_BFS_TIME_CAP_SECONDS}s), refusing"
            )

        if opaque:
            o_sheet, o_coord = opaque[0]
            return False, (
                f"unsafe: {len(opaque)} opaque/unresolvable cell(s) in dependency chain, cannot "
                f"prove safety (e.g. {o_sheet}!{o_coord} uses INDIRECT/OFFSET/ADDRESS, a whole "
                f"column/row reference, an unresolvable name, or another Data Table)"
            )

        # ---- Step 4: compare the closure ----
        for sheet_name, coord in visited:
            if sheet_name not in wb_gt_f.sheetnames:
                orig_repr = wb_orig_f[sheet_name][coord].value if sheet_name in wb_orig_f.sheetnames else None
                return False, f"unsafe: diverges at {sheet_name}!{coord} (sheet missing from golden, original={orig_repr!r})"

            orig_formula = wb_orig_f[sheet_name][coord].value

            if isinstance(orig_formula, str) and orig_formula.startswith("="):
                gt_formula = wb_gt_f[sheet_name][coord].value
                o_norm = _strip_xlfn_prefix(orig_formula)
                g_norm = _strip_xlfn_prefix(gt_formula) if isinstance(gt_formula, str) else gt_formula
                if not (isinstance(g_norm, str) and g_norm == o_norm):
                    return False, f"unsafe: diverges at {sheet_name}!{coord} (golden={gt_formula!r}, original={orig_formula!r})"
            elif type(orig_formula).__name__ == "DataTableFormula":
                if (sheet_name, coord) in target_cells_set:
                    # Known, confirmed-empirically limitation: openpyxl
                    # exposes NO comparable cached value for a
                    # DataTableFormula-typed cell under data_only=True (it
                    # reads as None) even though golden retains a real
                    # leftover number there -- comparing them would ALWAYS
                    # spuriously "diverge" at exactly the cells this
                    # fallback exists to override, regardless of actual
                    # safety. Not a real signal; skip. (Non-anchor member
                    # cells of the SAME table, which DO carry a comparable
                    # cached number, are not exempted -- see the `else`
                    # branch below; confirmed on Debugging/02_04 and 02_07
                    # that these all matched: 96/96 and 72/72 respectively.)
                    continue
                # A different table's anchor should already have been
                # caught as opaque above; keep this as a defensive
                # fail-closed fallback in case that logic is ever changed.
                return False, f"unsafe: diverges at {sheet_name}!{coord} (unresolvable Data Table anchor from a different table)"
            else:
                orig_val = wb_orig_v[sheet_name][coord].value
                gt_val = wb_gt_v[sheet_name][coord].value if sheet_name in wb_gt_v.sheetnames else None
                if not _compare_cell_value(gt_val, orig_val):
                    return False, f"unsafe: diverges at {sheet_name}!{coord} (golden={gt_val!r}, original={orig_val!r})"

        return True, f"dependency-safe: {len(visited)} upstream cell(s) traced from {len(seeds)} seed(s), 0 divergent"

    except Exception as e:
        logger.warning(f"_datatable_dependency_safe: exception tracing dependencies for {orig_input_file}: {e}")
        return False, f"unsafe: exception during dependency trace: {e}"
    finally:
        if scratch is not None and os.path.exists(scratch):
            try:
                os.remove(scratch)
            except OSError:
                pass


def _build_hybrid_golden_with_original_datatable(gt_file: str, orig_input_file: str) -> str:
    """Recompute the Data Table live from the ORIGINAL pristine input (which
    still has the formula golden has lost), then overlay ONLY those
    specific target cells onto a scratch copy of GOLDEN -- never swap the
    whole comparison file to original's, since the caller (see
    _refresh_golden_datatables) has only verified that the table's OWN
    traced upstream dependency chain is untouched via
    _datatable_dependency_safe, not that the rest of golden is identical to
    original. Overlaying just the target cells means every other graded
    cell still compares against golden's own authoritative value untouched,
    regardless of how much of the rest of the model legitimately differs.

    Scans for target cells on the NORMALIZED (xlfn-prefixed, iterative-calc-
    forced, LibreOffice-recalculated) scratch copy, not the raw
    orig_input_file -- this is a fix, not just a style choice: scanning the
    raw file only ever finds a Data Table's ANCHOR cell (the one openpyxl
    exposes as a DataTableFormula object), silently missing every other
    member cell in the table's range, which carries no formula marker at
    all until normalization turns it into the plain '=TABLE(...)' string
    every member cell shares post-recalc (see
    _prepare_normalized_datatable_scratch's docstring; confirmed live: the
    raw-file scan found only 4 anchor cells across Debugging/02_04's 4
    tables, the normalized-scratch scan finds all 100 member cells).
    Undercounting here meant most of a multi-cell Data Table's output grid
    was silently left un-spliced, still holding golden's stale/lost value.
    """
    orig_scratch = None
    gold_scratch = f"{gt_file}.fresh_datatable_scratch.{uuid.uuid4().hex[:8]}.xlsx"
    try:
        orig_scratch = _prepare_normalized_datatable_scratch(orig_input_file)
        target_cells = _datatable_target_cells(orig_scratch)
        if not target_cells:
            return gt_file
        _solve_datatables_with_uno(orig_scratch)
        wb_orig_v = openpyxl.load_workbook(orig_scratch, data_only=True)

        shutil.copy2(gt_file, gold_scratch)
        wb_gold_v = openpyxl.load_workbook(gold_scratch, data_only=True)
        for sheet_name, coord in target_cells:
            if sheet_name not in wb_orig_v.sheetnames or sheet_name not in wb_gold_v.sheetnames:
                continue
            wb_gold_v[sheet_name][coord] = wb_orig_v[sheet_name][coord].value
        wb_gold_v.save(gold_scratch)
        return gold_scratch
    except Exception as e:
        logger.warning(f"_build_hybrid_golden_with_original_datatable: failed for {gt_file}: {e}, using raw golden")
        return gt_file
    finally:
        if orig_scratch is not None and os.path.exists(orig_scratch):
            try:
                os.remove(orig_scratch)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# In-process cache for _refresh_golden_datatables' two expensive branches.
#
# Motivation: multiple sibling bug-variant tasks in this dataset share ONE
# golden file (e.g. Financial_Model 17_02/17_03/17_04 all reference
# 17_FMCG_golden.xlsx -- the same sharing pattern that already forced the
# uuid-suffixed scratch names below, see that comment). For the "golden has
# its own live Data Table" branch specifically, every sibling's refresh is
# an IDENTICAL computation (same gt_file content in, same Docker+
# LibreOffice recalc, same UNO Data-Table solve, same bytes out) -- without
# this cache, each sibling independently re-pays a multi-second Docker+
# LibreOffice cold start for work already done by an earlier sibling.
#
# Cache VALUE is the refreshed file's raw BYTES (or None as an explicit
# sentinel meaning "no refresh needed/possible, use gt_file as-is"), never a
# shared file PATH. This is deliberate: score_and_diff's existing contract
# is "I deleted whatever scratch path _refresh_golden_datatables handed me,
# once I'm done comparing against it" -- a contract that's fine for a scratch
# file only ONE caller will ever see, but breaks immediately for a path
# multiple callers share (the second caller to arrive would find the first
# caller's cleanup already deleted the file out from under it, or race the
# delete mid-read). Caching bytes instead of a path sidesteps this entirely:
# every call to _refresh_golden_datatables still returns its OWN freshly
# uuid-suffixed scratch file (written from the cached bytes on a cache hit,
# or from a fresh computation on a cache miss) that IT exclusively owns --
# no caller ever touches another caller's file, so score_and_diff's existing
# per-call "delete if it differs from gt_file" cleanup (see score_and_diff)
# needed NO changes at all to stay correct under sharing. The tradeoff is one
# extra full read+write of the (typically small) xlsx file per cache HIT --
# negligible next to the multi-second Docker/LibreOffice round trip this
# cache exists to avoid, and zero-cost relative to today's code on the
# common case of a cache MISS (a golden needed by only one task), where the
# bytes are simply the same scratch file's content read once and rewritten
# once instead of just... being the scratch file. See constraint #4/#2 in
# the task this was built for.
#
# Two DIFFERENT cache-key shapes for the two branches (deliberately NOT
# collapsed into one -- see _refresh_golden_datatables' branch dispatch
# below for exactly why each shape is what it is):
#   "blunt" (golden has its own live Data Table): keyed on gt_file's
#     fingerprint alone -- this branch never reads orig_input_file or
#     answer_position at all, so those cannot matter to its output.
#   "orig_fallback" (golden has none, falls back to recomputing from
#     orig_input_file): keyed on the fingerprint of BOTH files plus the
#     literal answer_position string -- different sibling tasks sharing one
#     golden can have different original-input bug variants (confirmed
#     live: Debugging/02_04 and 02_07 both reference the same 02_golden.xlsx
#     but from DIFFERENT original inputs, "Inconsistent Color Coding" vs
#     "Incorrect Index Match") and, in principle, different answer_position
#     ranges. Using only gt_file here would risk silently splicing one
#     task's original-input-derived Data-Table values into a different
#     task's golden -- a correctness bug, not just a missed optimization.
_GOLDEN_REFRESH_CACHE: dict[tuple, "bytes | None"] = {}
_GOLDEN_REFRESH_CACHE_GUARD = threading.Lock()  # protects ONLY the key-lock dict below, held briefly
_GOLDEN_REFRESH_KEY_LOCKS: dict[tuple, threading.Lock] = {}


def _file_fingerprint(path: str) -> tuple:
    """(resolved path, size, mtime_ns) -- a cheap (single stat syscall)
    guard against a cache key silently going stale if a file were ever
    mutated in place between two calls that reuse the same path, without
    paying for a full content hash on every call. Scoring runs treat
    golden/original-input files as read-only for their whole duration, so
    this is belt-and-suspenders, not load-bearing -- but it's essentially
    free, so there's no reason to trust the path alone when a stat also
    catches the theoretical case. Fails open to "path alone" (matching the
    pre-cache behavior of just trusting the path) if the stat itself fails
    for any reason -- a stat failure here should never be what breaks
    scoring for an otherwise-fine file."""
    resolved = str(Path(path).resolve())
    try:
        st = os.stat(resolved)
        return (resolved, st.st_size, st.st_mtime_ns)
    except OSError:
        return (resolved,)


def _golden_refresh_key_lock(key: tuple) -> threading.Lock:
    """Per-KEY lock, created on demand. _GOLDEN_REFRESH_CACHE_GUARD is only
    ever held for the few microseconds it takes to look up/insert a lock
    object in a dict -- never for the duration of any expensive work -- so
    two threads refreshing two DIFFERENT golden files never serialize
    behind each other here; only two threads racing on the literal SAME key
    ever block on each other, and only until the first one finishes. This
    mirrors the existing _RECALC_SEMAPHORE/_DATATABLE_UNO_SEMAPHORE idiom
    in this file: bound concurrency/contention precisely where it's needed,
    never more broadly than that."""
    with _GOLDEN_REFRESH_CACHE_GUARD:
        lock = _GOLDEN_REFRESH_KEY_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _GOLDEN_REFRESH_KEY_LOCKS[key] = lock
        return lock


def _get_or_compute_golden_refresh(key: tuple, compute) -> "bytes | None":
    """Return cached bytes (or the None sentinel) for `key`, calling
    `compute()` to produce it AT MOST ONCE across all threads -- even under
    concurrent callers racing on the identical key -- via double-checked
    locking: a lock-free read first (dict reads are atomic under the GIL,
    and the common case is a hit or a miss with no contention at all), then
    the per-KEY lock from _golden_refresh_key_lock guards the actual
    compute-and-store, with a second check immediately after acquiring it
    to catch the case where a concurrent caller already finished the exact
    same work while this thread was waiting on the lock."""
    if key in _GOLDEN_REFRESH_CACHE:
        return _GOLDEN_REFRESH_CACHE[key]
    with _golden_refresh_key_lock(key):
        if key in _GOLDEN_REFRESH_CACHE:
            return _GOLDEN_REFRESH_CACHE[key]
        value = compute()
        _GOLDEN_REFRESH_CACHE[key] = value
        return value


def _compute_blunt_golden_refresh_bytes(gt_file: str) -> "bytes | None":
    """The actual expensive "golden has its own live Data Table" refresh --
    unchanged from _refresh_golden_datatables' pre-cache logic (copy, xlfn-
    prefix, force iterative calc, LibreOffice recalc, UNO Data-Table solve)
    except that the result is read into memory and the scratch file cleaned
    up immediately, so the cache can store/share content rather than a path.
    Returns None (meaning "use raw golden") on any failure, exactly as the
    pre-cache code fell back to returning gt_file itself."""
    scratch = f"{gt_file}.fresh_datatable_scratch.{uuid.uuid4().hex[:8]}.xlsx"
    try:
        shutil.copy2(gt_file, scratch)
        _add_xlfn_prefixes(scratch)
        _force_iterative_calc(scratch)
        _recalculate_with_libreoffice(scratch)
        _solve_datatables_with_uno(scratch)  # no gt_file/answer_position -- see _refresh_golden_datatables docstring
        with open(scratch, "rb") as f:
            return f.read()
    except Exception as e:
        logger.warning(f"_refresh_golden_datatables: failed for {gt_file}: {e}, using raw golden")
        return None
    finally:
        if os.path.exists(scratch):
            try:
                os.remove(scratch)
            except OSError:
                pass


def _compute_orig_fallback_golden_refresh_bytes(
    gt_file: str, orig_input_file: str, answer_position: str,
) -> "bytes | None":
    """The actual expensive "golden has no live Data Table, recompute from
    the pristine original input" refresh, gated by _datatable_dependency_safe
    -- unchanged from _refresh_golden_datatables' pre-cache logic except that
    the result is read into memory and any scratch file cleaned up
    immediately, so the cache can store/share content rather than a path.
    Returns None (meaning "use raw golden") whenever the pre-cache code
    would have returned gt_file itself: the dependency check refuses, or
    anything raises."""
    try:
        safe, reason = _datatable_dependency_safe(gt_file, orig_input_file, answer_position)
    except Exception as e:
        logger.warning(f"_refresh_golden_datatables: original-fallback dependency check failed for {gt_file}: {e}")
        return None
    if not safe:
        logger.warning(
            f"_refresh_golden_datatables: golden has no live Data Table and the "
            f"dependency-safety check refused the original-input fallback for {gt_file} -- {reason} "
            f"-- using raw golden"
        )
        return None
    logger.info(
        f"_refresh_golden_datatables: original-input fallback approved for {gt_file} -- {reason}"
    )
    hybrid_path = _build_hybrid_golden_with_original_datatable(gt_file, orig_input_file)
    if hybrid_path == gt_file:
        # _build_hybrid_golden_with_original_datatable itself failed/no-oped
        # and already logged why -- nothing new to read.
        return None
    try:
        with open(hybrid_path, "rb") as f:
            return f.read()
    except Exception as e:
        # Matches every other failure path in this function: degrade to
        # None ("use raw golden") rather than let an OSError/IOError here
        # propagate up through score_and_diff's broad except-and-score-0.0
        # at the real call sites -- an unreadable scratch file should not
        # silently turn into a false FAIL for the whole task.
        logger.warning(f"_refresh_golden_datatables: could not read computed hybrid golden {hybrid_path} for {gt_file}: {e} -- using raw golden")
        return None
    finally:
        try:
            os.remove(hybrid_path)
        except OSError:
            pass


def _refresh_golden_datatables(
    gt_file: str, orig_input_file: str | None = None, answer_position: str | None = None,
) -> str:
    """Return a path to compare against: if golden itself has stale/never-
    recalculated Data-Table cells, recompute them live via the same UNO fix
    used on solver output, using GOLDEN'S OWN unmodified formulas/inputs --
    never touches or reads anything from proc_file, so this cannot leak
    solver information into the comparison, it only fixes staleness
    internal to golden.

    Root-caused via 3-skeptic adversarial audit on 3 independent tasks
    (Debugging/02_04, 02_07, Financial_Model/14_02): golden's cached
    Data-Table values were stale relative to a live recalculation of
    golden's OWN formula chain -- confirmed present in the PRISTINE
    ORIGINAL input already, i.e. a benchmark-construction artifact (the
    table was probably excluded from auto-recalc and never refreshed
    before golden was frozen), not an intentionally-preserved answer. A
    from-scratch hand-reimplementation of the underlying financial formulas
    in one case matched the solver's live-recomputed value to ~1e-11 and
    diverged from golden's frozen value at every affected cell -- direct
    proof golden itself, not the solver or the scorer's recompute, is wrong
    here.

    Uses the BLUNT (non-golden-aware) safety net deliberately, not the
    regression-aware one: passing gt_file=gt_file here would compare the
    scratch copy against itself (since it starts as a byte copy of golden),
    making every legitimate fix look like a self-regression and rejecting
    all of them. The blunt check still protects every non-Data-Table cell
    in golden from any unintended side effect; only the Data-Table target
    cells are allowed to change, which is the entire point.

    ORIGINAL-INPUT FALLBACK (added after finding golden itself has, for
    some tasks, lost the Data-Table formula entirely -- not just a stale
    cache but zero formula information left at all, e.g. Debugging/02_04
    and 02_07 -- so there's nothing in golden's own file to recompute
    from). When golden has no live Data Table but orig_input_file/
    answer_position are supplied and the PRISTINE ORIGINAL still has one,
    fall back to recomputing from the original's formula instead -- GATED
    on _datatable_dependency_safe(), which traces the table's actual
    upstream formula dependency chain (its row/column input cells and
    everything THEY transitively depend on) and requires every cell in
    that traced closure to match, formula-for-formula or value-for-value,
    between golden and the untouched original.

    This replaced an earlier, coarser gate (_golden_vs_original_agreement:
    "what fraction of graded cells, or of the table's own sheet, have an
    identical cached value between golden and original") that could not
    tell "this sheet has unrelated changes elsewhere" apart from "this
    table's own inputs changed" -- confirmed live on Debugging/02_04
    (88.7%) and 02_07 (97.1%), both below that check's 98% threshold, even
    though neither task's fix touches the table's real dependency chain
    (see _datatable_dependency_safe's docstring for the specific traced
    path and the direct empirical confirmation). The new gate is strictly
    more precise -- it names the exact cell responsible for a refusal
    instead of a bare percentage -- but is NOT guaranteed to pass more
    often: a task whose Data Table genuinely depends on something the
    fix legitimately changed (or on a construct this parser can't
    statically resolve at all, e.g. a whole-column INDEX/MATCH) correctly
    still refuses, exactly as it should.

    Known limitation of the new gate (the acceptable, over-conservative
    direction -- see _datatable_dependency_safe's docstring for the full
    discussion): a purely textual formula comparison will flag two
    semantically-equivalent-but-differently-written formulas as a
    divergence, and any statically-unresolvable construct anywhere in the
    traced chain (INDIRECT/OFFSET/ADDRESS, a whole-column/row reference, an
    unresolvable named range) fails the whole check closed rather than
    guess. It never does the reverse (calling something safe that isn't).

    Falls back to the raw gt_file if golden has no Data Tables at all
    (near-zero cost -- the cheap pre-check inside _solve_datatables_with_uno
    short-circuits immediately), if the original-input fallback doesn't
    apply/qualify, or if anything fails.

    CACHING: the actual expensive work of each branch (blunt refresh /
    original-input fallback) is memoized in-process, keyed on the branch's
    real inputs (see the _GOLDEN_REFRESH_CACHE block above this function for
    the full rationale and the two distinct key shapes) -- this function
    always still returns a freshly uuid-suffixed scratch path that THIS call
    exclusively owns (written from the cached bytes on a hit, or from a
    fresh computation on a miss), so nothing below this docstring, and
    nothing in score_and_diff's cleanup of the path this returns, had to
    change to be cache-safe.
    """
    try:
        has_table = _has_live_datatable(gt_file)
    except Exception:
        return gt_file

    if not has_table:
        if not (orig_input_file and answer_position):
            return gt_file
        cache_key = ("orig_fallback", _file_fingerprint(gt_file), _file_fingerprint(orig_input_file), answer_position)
        logger.info(
            f"_refresh_golden_datatables: cache {'HIT' if cache_key in _GOLDEN_REFRESH_CACHE else 'MISS'} "
            f"(orig_fallback) for {gt_file} / {orig_input_file}"
        )
        refreshed_bytes = _get_or_compute_golden_refresh(
            cache_key,
            lambda: _compute_orig_fallback_golden_refresh_bytes(gt_file, orig_input_file, answer_position),
        )
    else:
        cache_key = ("blunt", _file_fingerprint(gt_file))
        logger.info(
            f"_refresh_golden_datatables: cache {'HIT' if cache_key in _GOLDEN_REFRESH_CACHE else 'MISS'} "
            f"(blunt) for {gt_file}"
        )
        refreshed_bytes = _get_or_compute_golden_refresh(
            cache_key, lambda: _compute_blunt_golden_refresh_bytes(gt_file),
        )

    if refreshed_bytes is None:
        return gt_file

    # Per-call-unique suffix -- multiple bug-variant tasks in the same
    # numbered folder commonly share one golden file (e.g. Financial_Model
    # 17_02/17_03/17_04 all reference 17_FMCG_golden.xlsx), and this
    # function runs concurrently across tasks in production rescoring (6+
    # workers). A purely gt_file-derived scratch name lets two concurrent
    # calls collide on the same path -- confirmed live: one call's
    # shutil.copy2 mid-write corrupted another's in-progress read ("Bad
    # magic number for file header"). uuid4 makes collision the practical
    # impossibility it needs to be. Still true (and still needed) under
    # caching: two sibling tasks sharing a cache HIT each still write their
    # OWN scratch copy of the shared bytes below, so they need their own
    # non-colliding names just as much as two cache MISSES would. The
    # caller (score_and_diff) removes this file once compare_workbooks_detailed
    # has read it -- it must still exist when this function returns, so
    # cleanup does NOT happen here.
    scratch = f"{gt_file}.fresh_datatable_scratch.{uuid.uuid4().hex[:8]}.xlsx"
    try:
        with open(scratch, "wb") as f:
            f.write(refreshed_bytes)
        return scratch
    except OSError as e:
        logger.warning(f"_refresh_golden_datatables: could not write cached scratch copy for {gt_file}: {e}, using raw golden")
        return gt_file


def score_and_diff(
    gt_file: str, proc_file: str, answer_position: str, max_mismatches: int = 0,
    orig_input_file: str | None = None,
) -> tuple[bool, int, int, str]:
    """Recalculate once, compare once — score AND detailed diff in one pass.

    The scoring pipeline already loads both workbooks and walks every graded
    cell; collecting the mismatch list at the same time is free. Persisting
    that diff (see snapshot_output) spares the evolver from re-running the
    same multi-minute comparison over every failed task (Jul 1: one such
    batch re-computation blew the 120s bash timeout).

    orig_input_file: path to the pristine pre-solving input, used to restore
    cells LibreOffice's recalc destroys on formulas the solver never
    touched (see _restore_unrecalculable_cells). Optional so callers that
    don't have this path handy (or don't care about this class of false
    negative) can omit it -- scoring still works, just without the fix.
    """
    _add_xlfn_prefixes(proc_file)
    _force_iterative_calc(proc_file)
    _recalculate_with_libreoffice(proc_file)
    _solve_datatables_with_uno(proc_file, gt_file=gt_file, answer_position=answer_position)
    skip_cells = set()
    if orig_input_file:
        _restore_unrecalculable_cells(proc_file, orig_input_file, answer_position, gt_file=gt_file)
        skip_cells = _volatile_function_skip_cells(orig_input_file, proc_file, answer_position)
    gt_file_for_compare = _refresh_golden_datatables(gt_file, orig_input_file=orig_input_file, answer_position=answer_position)
    try:
        return compare_workbooks_detailed(gt_file_for_compare, proc_file, answer_position, max_mismatches, skip_cells=skip_cells)
    finally:
        if gt_file_for_compare != gt_file:
            try:
                os.remove(gt_file_for_compare)
            except OSError:
                pass


def compare_workbooks(gt_file: str, proc_file: str, answer_position: str) -> bool:
    """Compare golden and processed workbooks at specified cell ranges.

    Recalculates formulas in the output file via LibreOffice before comparing,
    so that formula cells have cached values for openpyxl to read.

    Returns True if all specified cells match.
    """
    if not os.path.exists(proc_file):
        return False

    # Recalculate formulas so openpyxl can read cached values
    _add_xlfn_prefixes(proc_file)
    _recalculate_with_libreoffice(proc_file)

    try:
        wb_gt = openpyxl.load_workbook(filename=gt_file, data_only=True)
        wb_proc = openpyxl.load_workbook(filename=proc_file, data_only=True)
    except Exception:
        return False

    sheet_cell_ranges = _split_answer_position(answer_position)
    for sheet_cell_range in sheet_cell_ranges:
        if "!" in sheet_cell_range:
            sheet_name, cell_range = sheet_cell_range.rsplit("!", 1)
            sheet_name = sheet_name.strip("'")
        else:
            sheet_name = wb_gt.sheetnames[0]
            cell_range = sheet_cell_range

        cell_range = cell_range.strip("'")

        if not _cell_level_compare(wb_gt, wb_proc, sheet_name, cell_range):
            return False

    return True


def score_ssbv2(question: str, predicted: str, ground_truth: str) -> float:
    """Score an SSB v2 task.

    Args:
        question: JSON with task metadata (contains task_id, spreadsheet_path)
        predicted: The solver's final_answer (output file path)
        ground_truth: JSON with task_id, golden_path, and answer_position
    """
    try:
        gt_info = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError):
        return 0.0

    golden_path = gt_info.get("golden_path", "")
    answer_position = gt_info.get("answer_position", "")

    if not golden_path or not answer_position:
        return 0.0

    proc_file = predicted.strip() if predicted else ""
    if not proc_file:
        return 0.0

    # This is the SelfImprovingLoop's own `scorer=` callback -- used by the
    # actual production entrypoints (scripts/run_ssbv2_evo.py,
    # run_ssbv2_smoke.py), not just an internal/dead fallback. Confirmed as
    # an active bug: it called compare_workbooks() (a plain recalc + compare,
    # no xlfn-prefix fix, no Data-Table UNO solve, no restoration fallback),
    # so every fix made to score_and_diff's pipeline was
    # silently invisible to the live production scorer path -- the agent's
    # own inline scoring (ssb_v2_agent.py's _score_fn) already used the fixed
    # score_and_diff, but this separate callback did not.
    orig_input_file = None
    try:
        q_info = json.loads(question)
        orig_input_file = q_info.get("spreadsheet_path") or None
    except (json.JSONDecodeError, TypeError):
        pass

    try:
        passed, _, _, _ = score_and_diff(golden_path, proc_file, answer_position, orig_input_file=orig_input_file)
        return 1.0 if passed else 0.0
    except Exception:
        return 0.0
