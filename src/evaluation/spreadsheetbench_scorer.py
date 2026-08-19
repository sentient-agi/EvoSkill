"""SpreadsheetBench scorer: runs the Harbor verifier (LibreOffice + cell comparison)
in a pre-built Docker container and returns the task reward (0.0-1.0, mean
over test cases; every task in the local dataset has exactly 1 case).

The solver produces an output .xlsx via Python code. This scorer:
  1. Mounts the output file + answer files + evaluate.py into ssb-verifier container
  2. Runs test.sh (LibreOffice recalc + cell comparison)
  3. Reads /logs/verifier/reward.txt → float

Sync interface (score_spreadsheetbench) for the loop's scorer callback.
Async interface (score_spreadsheetbench_async) for inline scoring.

Detailed-diff surface (SSB v2 parity, Jul 2 port):
  compare_workbooks_detailed — exact port of evaluate.py's comparison that
      walks EVERY graded cell and lists mismatches (evaluate.py stops at the
      first one); max_mismatches <= 0 means uncapped.
  render_task_diff — per-test-case diff body for a task's output dir.
  score_and_diff — Docker verifier reward AND full diff in one scoring pass;
      the diff is persisted as DIFF.txt in per-iteration snapshots (see
      src/agent_profiles/spreadsheetbench_agent/task_workspace.py) so the
      evolver never re-runs comparisons.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import openpyxl

logger = logging.getLogger(__name__)

# Root of the SpreadsheetBench checkout. Used only as a fallback when a task id is
# not already in TASK_INDEX; no guessed default — set SSB_TASKS_ROOT explicitly.
_ssb_tasks_root = os.environ.get("SSB_TASKS_ROOT")
SSB_ROOT = Path(_ssb_tasks_root) if _ssb_tasks_root else None
SSB_TASKS_CACHE = Path(os.environ.get(
    "SSB_TASKS_CACHE", Path(__file__).resolve().parents[2] / ".cache/ssb_tasks"))

# Bounds concurrent `docker run` calls (verifier + recalc) against the fixed
# CPU budget of the Colima VM. Confirmed as an active bug during an
# SSB v2 investigation: this exact contention pattern (many solver tasks
# finishing scoring around the same time, each firing its own unthrottled
# docker run) caused a recalc that takes 1.5-2s in isolation to exceed its
# own 60-180s docker timeout, silently misscoring genuinely-correct tasks as
# wrong. SSB v2's sibling scorer already got this fix
# (src/agent_profiles/ssbv2_agent/scorer.py's _RECALC_SEMAPHORE); this one
# had not, despite sharing the identical failure mode. score_spreadsheetbench
# runs these via run_in_executor's default ThreadPoolExecutor, so a plain
# threading.Semaphore (not asyncio) is correct regardless of caller async-ness.
_DOCKER_SEMAPHORE = threading.Semaphore(int(os.environ.get("EVOSKILL_SSB1_DOCKER_CONCURRENCY", "6")))

# Map task_id -> task_dir (populated by the dataset loader)
TASK_INDEX: dict[str, Path] = {}


def _get_task_dir(task_id: str) -> Path | None:
    if task_id in TASK_INDEX:
        return TASK_INDEX[task_id]
    # Fallback: try direct path
    if SSB_ROOT is None:
        raise RuntimeError(
            f"task {task_id!r} not in TASK_INDEX and SSB_TASKS_ROOT is not set; "
            "point SSB_TASKS_ROOT at your SpreadsheetBench checkout")
    candidate = SSB_ROOT / task_id
    if candidate.is_dir() and (candidate / "tests").is_dir():
        return candidate
    return None


def _run_verifier(task_dir: Path, output_dir: Path, timeout: int = 120) -> float:
    """Run the SpreadsheetBench verifier in Docker and return the reward."""
    tests_dir = task_dir / "tests"
    answer_dir = tests_dir / "answer_files"

    if not answer_dir.is_dir():
        logger.warning(f"No answer_files in {tests_dir}")
        return 0.0

    # We need to run test.sh inside the verifier container with:
    # - /app/output/ mounted from solver's output
    # - /tests/ mounted from the task's tests dir
    # - /app/spreadsheets/ mounted from the task's spreadsheets (for reference)
    spreadsheets_dir = task_dir / "environment" / "spreadsheets"
    if not spreadsheets_dir.is_dir():
        spreadsheets_dir = task_dir / "tests" / "spreadsheets"

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{output_dir}:/app/output",
        "-v", f"{tests_dir}:/tests:ro",
    ]
    if spreadsheets_dir.is_dir():
        cmd.extend(["-v", f"{spreadsheets_dir}:/app/spreadsheets:ro"])

    cmd.extend(["ssb-verifier:latest", "bash", "/tests/test.sh"])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        # Read reward from the container's stdout (test.sh prints it)
        # or parse from the expected reward file location
        # The test.sh writes to /logs/verifier/reward.txt inside the container
        # Since we can't read that after --rm, we need a different approach:
        # Mount a local dir for /logs/verifier
    except subprocess.TimeoutExpired:
        logger.warning(f"Verifier timed out ({timeout}s) for {task_dir.name}")
        return 0.0
    except Exception as e:
        logger.warning(f"Verifier failed for {task_dir.name}: {e}")
        return 0.0

    # The approach above loses /logs/verifier/reward.txt on --rm.
    # Better: mount a tmpdir for logs.
    return 0.0  # placeholder — need to restructure


def _normalise_output_formulas(output_dir: Path) -> int:
    """Rewrite LibreOffice-unevaluable formula constructs in solver outputs.

    Closes a scoring asymmetry: goldens are Excel-authored and carry cached
    values, so when LO cannot evaluate a golden formula the harness heals the
    cell back to Excel's own result. openpyxl-written solver files carry no
    cached values, so the same LO limitation reads as a wrong answer instead.

    Today this rewrites this-row structured references (``Table1[@[Col]]``),
    which LO reports as #N/A but Excel generates automatically for any formula
    typed into a table column — verified against ssb-verifier's LibreOffice
    25.2.3.2. The rewrite resolves each reference to the exact cell Excel
    resolves it to, so it changes representation only.

    Set EVOSKILL_SSB_FORMULA_COMPAT=0 to disable and reproduce pre-fix scoring.
    Returns the number of cells rewritten (0 if disabled or nothing matched).
    """
    if os.environ.get("EVOSKILL_SSB_FORMULA_COMPAT", "1") == "0":
        return 0
    try:
        import openpyxl

        from src.agent_profiles.spreadsheetbench_agent.formula_compat import (
            normalise_workbook,
        )
    except Exception as e:
        logger.warning("formula-compat unavailable (%s) — scoring un-normalised", e)
        return 0

    total = 0
    for xlsx in sorted(Path(output_dir).glob("*.xlsx")):
        if xlsx.name.startswith("~$"):
            continue
        try:
            wb = openpyxl.load_workbook(xlsx)
            n = normalise_workbook(wb)
            if n:
                wb.save(xlsx)
                total += n
                logger.info(
                    "formula-compat: rewrote %d this-row reference(s) in %s",
                    n, xlsx.name,
                )
        except Exception as e:
            # Never let normalisation break scoring — fall through to the
            # verifier with the file exactly as the solver left it.
            logger.warning("formula-compat failed on %s: %s", xlsx.name, e)
    return total


def _run_verifier_v2(task_dir: Path, output_dir: Path, timeout: int = 180) -> float:
    """Run the verifier with a mounted logs dir to capture the reward."""
    tests_dir = task_dir / "tests"
    answer_dir = tests_dir / "answer_files"

    if not answer_dir.is_dir():
        logger.warning(f"No answer_files in {tests_dir}")
        return 0.0

    spreadsheets_dir = task_dir / "environment" / "spreadsheets"
    if not spreadsheets_dir.is_dir():
        # Some tasks have spreadsheets directly in tests/
        spreadsheets_dir = tests_dir / "spreadsheets"

    # Use a path under /Users/ — Colima only shares /Users via virtiofs;
    # tempfile.TemporaryDirectory() lands in /tmp which is invisible to Docker.
    import uuid as _uuid
    logs_path = SSB_TASKS_CACHE / "_verifier_logs" / _uuid.uuid4().hex[:12]
    logs_path.mkdir(parents=True, exist_ok=True)
    (logs_path / "verifier").mkdir(exist_ok=True)
    # Symmetric counterpart to the golden heal below: rewrite formula
    # constructs in the SOLVER's output that LibreOffice cannot evaluate but
    # Excel can. Without this the golden is effectively graded under Excel
    # semantics (its Excel-cached values are healed back in) while the solver
    # is graded under LibreOffice semantics, so a correct answer can score 0.
    # Representation-only: it cannot turn a wrong answer into a right one.
    _normalise_output_formulas(output_dir)
    # Healed-answer overlay so in-container LO cannot re-break goldens that
    # LibreOffice itself fails to evaluate (see _heal_golden_lo_errors).
    tests_overlay: Path | None = None
    try:
        tests_overlay = _prepare_healed_tests_overlay(task_dir)

        cmd = [
            "docker", "run", "--rm",
            "-v", f"{output_dir.resolve()}:/app/output",
            "-v", f"{tests_overlay.resolve()}:/tests:ro",
            "-v", f"{logs_path.resolve()}:/logs",
        ]
        if spreadsheets_dir.is_dir():
            cmd.extend(["-v", f"{spreadsheets_dir.resolve()}:/app/spreadsheets:ro"])

        cmd.extend(["ssb-verifier:latest", "bash", "/tests/test.sh"])

        try:
            with _DOCKER_SEMAPHORE:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=timeout,
                )
        except subprocess.TimeoutExpired:
            logger.warning(f"Verifier timed out ({timeout}s) for {task_dir.name}")
            return 0.0
        except Exception as e:
            logger.warning(f"Verifier failed for {task_dir.name}: {e}")
            return 0.0

        reward_path = logs_path / "verifier" / "reward.txt"
        if reward_path.exists():
            try:
                return float(reward_path.read_text().strip())
            except (ValueError, OSError):
                pass

        # Fallback: check if evaluate.py printed the reward to stdout. Only
        # the LAST all-digit line counts — matching the first one found
        # anywhere in stdout risked silently returning an incidental numeric
        # print (a step counter, a row count) as the task's score with no
        # warning (bug M22). This is still a heuristic, not a guarantee, so
        # it's logged whenever it's actually used.
        numeric_lines = [
            line.strip() for line in result.stdout.splitlines()
            if line.strip().replace(".", "", 1).isdigit()
        ]
        if numeric_lines:
            try:
                reward = float(numeric_lines[-1])
                logger.warning(
                    f"No reward.txt for {task_dir.name} — falling back to the "
                    f"last numeric stdout line ({reward!r}); this is a heuristic, "
                    f"not a guarantee it's the real reward"
                )
                return reward
            except ValueError:
                pass

        logger.warning(
            f"No reward found for {task_dir.name}. "
            f"exit={result.returncode}, stdout={result.stdout[-500:]}"
        )
        return 0.0
    finally:
        import shutil as _shutil
        _shutil.rmtree(logs_path, ignore_errors=True)
        if tests_overlay is not None:
            _shutil.rmtree(tests_overlay, ignore_errors=True)


# ---------------------------------------------------------------------------
# Detailed comparison (DIFF.txt) — exact port of the verifier's comparison
# semantics from the per-task tests/evaluate.py (byte-identical across all
# 400 task dirs; verified by md5, 2026-07-02). The Docker verifier above
# remains the authoritative reward; these functions exist so the harness can
# persist the full cell-level mismatch list at scoring time (snapshot
# DIFF.txt) and so the evolver can run what-if comparisons inside its
# container, where no docker CLI is available. Mirrors the SSB v2 pattern
# (src/agent_profiles/ssbv2_agent/scorer.py) — kept self-contained rather
# than importing across profiles.
# ---------------------------------------------------------------------------


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


def _col_num2name(n: int) -> str:
    name = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _col_name2num(name: str) -> int:
    num = 0
    for c in name:
        num = num * 26 + (ord(c) - ord("A") + 1)
    return num


def _parse_cell_range(range_str: str):
    """Parse a v1 cell range, incl. quirks (exact port of evaluate.py).

    Handles column-only ranges ("A:G" -> rows unbounded), end cells without
    a column ("BD2:308" -> BD2:BD308), and start cells without a column.
    Returns ((start_col, start_row), (end_col, end_row)); rows may be None.
    """
    start_cell, end_cell = range_str.split(":")
    start_col, start_row = "", ""
    for char in start_cell:
        if char.isdigit():
            start_row += char
        else:
            start_col += char
    end_col, end_row = "", ""
    for char in end_cell:
        if char.isdigit():
            end_row += char
        else:
            end_col += char
    start_row = int(start_row) if start_row else None
    end_row = int(end_row) if end_row else None
    if not end_col and start_col:
        end_col = start_col
    if not start_col and end_col:
        start_col = end_col
    return (_col_name2num(start_col) if start_col else 1, start_row), (
        _col_name2num(end_col) if end_col else 1,
        end_row,
    )


def _generate_cell_names(range_str: str, max_row: int | None = None) -> list[str]:
    """Expand a range string into cell names (exact port of evaluate.py)."""
    if ":" not in range_str:
        return [range_str]
    (start_col, start_row), (end_col, end_row) = _parse_cell_range(range_str)
    if start_row is None:
        start_row = 1
    if end_row is None:
        end_row = max_row if max_row else 1000
    columns = [_col_num2name(i) for i in range(start_col, end_col + 1)]
    return [f"{col}{row}" for col in columns for row in range(start_row, end_row + 1)]


def parse_answer_position(answer_position: str) -> list[tuple[str | None, str]]:
    """Parse answer_position into (sheet_name_or_None, cell_range) tuples.

    Exact port of evaluate.py's parser — the only parser that handles all of
    the benchmark's quirky formats:
      - Bare range/cell: "A1:B5", "B6"
      - Trailing quote:  "Sheet1'!A1:B5"
      - Bang inside quotes: "'Sheet1!'A1:A50"
      - Properly quoted: "'Sheet 1'!A1:B5"
      - Commas in sheet name: "'b2b, sez, de'!A5:V10"
      - Column-only ranges: "Sheet3'!A:G"
    """
    parts = []
    current: list[str] = []
    in_quote = False
    at_part_start = True
    for ch in answer_position:
        if ch == "'" and at_part_start and not in_quote:
            in_quote = True
            current.append(ch)
            at_part_start = False
        elif ch == "'" and in_quote:
            in_quote = False
            current.append(ch)
        elif ch == "," and not in_quote:
            parts.append("".join(current).strip())
            current = []
            at_part_start = True
        else:
            current.append(ch)
            if ch != " ":
                at_part_start = False
    if current:
        parts.append("".join(current).strip())

    results: list[tuple[str | None, str]] = []
    for part in parts:
        p = part.strip("'")
        if "!" in p:
            sheet, cell_range = p.split("!", 1)
            results.append((sheet.strip("'"), cell_range.strip("'")))
        else:
            results.append((None, p.strip("'")))
    return results


def compare_workbooks_detailed(
    gt_file: str, proc_file: str, answer_position: str, max_mismatches: int = 20
) -> tuple[bool, int, int, str]:
    """Compare workbooks and return a detailed diff.

    Returns (passed, total_match, total_cells, diff_summary). Comparison
    semantics are an exact port of the verifier's evaluate.py (same value
    transforms, same answer_position parser incl. column-only ranges) —
    but where evaluate.py stops at the FIRST mismatch, this walks every
    graded cell and lists up to max_mismatches mismatches for evolver
    feedback; max_mismatches <= 0 means no cap (list every mismatch).
    Mismatches are collected in range order (not causal order), so on
    cascade failures the default 20 may show only downstream symptoms —
    re-run with max_mismatches=0 for the full picture on a specific task.

    NOTE: no recalculation happens here — both files must already carry
    cached formula values (the harness recalculates outputs in place at
    scoring time; goldens go through _recalc_answer_cached host-side).
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
    mismatches: list[str] = []
    all_pass = True

    for sheet_name, cell_range in parse_answer_position(answer_position):
        if sheet_name is None:
            sheet_name = wb_gt.sheetnames[0]
        if sheet_name not in wb_gt.sheetnames:
            all_pass = False
            mismatches.append(f"Sheet '{sheet_name}' missing from GOLDEN workbook")
            continue
        if sheet_name not in wb_proc.sheetnames:
            all_pass = False
            mismatches.append(f"Sheet '{sheet_name}' missing from output")
            continue
        ws_gt = wb_gt[sheet_name]
        ws_proc = wb_proc[sheet_name]
        max_row = max(ws_gt.max_row or 1, ws_proc.max_row or 1)
        try:
            cell_names = _generate_cell_names(cell_range, max_row=max_row)
        except Exception as e:
            # A RANGE THIS FUNCTION CANNOT READ IS NOT AN EMPTY RESULT. `continue` dropped
            # it in silence, and a task whose ranges ALL dropped fell through to
            # `all_pass` still True with total_cells == 0 and the summary "All cells match"
            # -- a false PASS. Measured: answer_position 'A1:B2:C3' on workbooks that
            # differ at A1 returned (True, 0, 0, 'All cells match').
            all_pass = False
            if max_mismatches <= 0 or len(mismatches) < max_mismatches:
                mismatches.append(f"'{sheet_name}'!{cell_range}: UNGRADABLE range ({e})")
            continue
        for cell_name in cell_names:
            total_cells += 1
            try:
                gt_val = ws_gt[cell_name].value
                proc_val = ws_proc[cell_name].value
            except Exception as e:
                all_pass = False
                if max_mismatches <= 0 or len(mismatches) < max_mismatches:
                    mismatches.append(f"'{sheet_name}'!{cell_name}: unreadable cell ({e})")
                continue
            if _compare_cell_value(gt_val, proc_val):
                total_match += 1
            else:
                all_pass = False
                if max_mismatches <= 0 or len(mismatches) < max_mismatches:
                    mismatches.append(
                        f"'{sheet_name}'!{cell_name}: "
                        f"expected={_transform_value(gt_val)!r}, "
                        f"got={_transform_value(proc_val)!r}"
                    )

    # ZERO GRADED CELLS IS NOT A PASS, HERE EITHER. `all_pass` starts True and only the
    # per-cell loop can falsify it, so every path that graded nothing and recorded nothing
    # reported "All cells match". This copy is handed to the evolver as
    # ./oracle/spreadsheetbench_scorer.py for what-if checks (task_workspace.py), so a
    # false PASS here tells the evolver a broken repair works. It raises nothing: the file
    # is imported standalone from the evolver's workspace, where src.harness is not on the
    # path, and the authoritative reward comes from the in-container evaluate.py anyway.
    if total_cells == 0 and not mismatches:
        all_pass = False
        mismatches.append(
            f"graded 0 cells for answer_position={answer_position!r} — nothing was "
            f"compared, so this is NOT a pass. Check the range spelling against "
            f"oracle/tasks/<task_id>.json.")
    diff_summary = "; ".join(mismatches) if mismatches else "All cells match"
    return all_pass, total_match, total_cells, diff_summary


# Excel / LibreOffice error tokens written into cached cell values when a
# formula fails to evaluate. LO recalc of SSB goldens sometimes produces
# these even when the Excel-cached value in the unrecalculated golden is a
# correct number — matching the error then falsely fails a correct solver.
_EXCEL_ERROR_VALUES = frozenset({
    "#VALUE!", "#NAME?", "#REF!", "#DIV/0!", "#N/A", "#NULL!", "#NUM!",
    "#GETTING_DATA", "#SPILL!", "#CALC!", "#FIELD!", "#BLOCKED!", "#UNKNOWN!",
})


def _is_excel_error_value(v) -> bool:
    if not isinstance(v, str):
        return False
    s = v.strip()
    if s in _EXCEL_ERROR_VALUES:
        return True
    # LibreOffice sometimes writes Err:NNN instead of the #TOKEN! form.
    return s.startswith("Err:") or s.startswith("#")


def _heal_golden_lo_errors(recalc_path: Path, original_path: Path) -> int:
    """Replace LO error cells in a recalculated golden with Excel-cached values.

    LibreOffice headless recalc is what test.sh does before compare, but on
    some goldens LO cannot evaluate Excel-valid formulas (date concatenation,
    missing add-in names, etc.) and writes #VALUE!/#NAME? into the graded
    cells. The unrecalculated golden almost always still carries the correct
    Excel-cached value under data_only=True. Prefer that cache over an LO
    error so a solver that writes the right number is not scored as fail.

    Policy:
      - LO error + usable Excel cache → write the cached value (literal)
      - LO error + no usable cache → clear to empty (do not require matching
        `#VALUE!`/`#NAME?`)
      - Successful LO numbers are frozen as literals when any heal runs, so
        openpyxl's save does not wipe neighboring formula caches

    Returns the number of error cells rewritten (healed or cleared).
    """
    recalc_path = Path(recalc_path)
    original_path = Path(original_path)
    try:
        wb_re_vals = openpyxl.load_workbook(recalc_path, data_only=True)
        wb_orig_vals = openpyxl.load_workbook(original_path, data_only=True)
        wb_write = openpyxl.load_workbook(recalc_path, data_only=False)
    except Exception as e:
        logger.warning(
            f"Golden LO-error heal skipped for {recalc_path.name}: cannot open ({e})"
        )
        return 0

    healed = 0
    # openpyxl drops cached formula results on save of a data_only=False
    # workbook. When we rewrite any cell we must also freeze every other
    # non-empty cached value as a literal, or previously-good LO results
    # (e.g. 46646!B11) become None after save and falsely fail compare.
    dirty = False
    for sheet_name in wb_write.sheetnames:
        if sheet_name not in wb_re_vals.sheetnames or sheet_name not in wb_orig_vals.sheetnames:
            continue
        ws_re = wb_re_vals[sheet_name]
        ws_orig = wb_orig_vals[sheet_name]
        ws_w = wb_write[sheet_name]
        sheet_heals: list[tuple[str, object]] = []
        freeze: list[tuple[str, object]] = []
        for row in ws_re.iter_rows(
            min_row=1,
            max_row=ws_re.max_row or 1,
            max_col=ws_re.max_column or 1,
        ):
            for cell in row:
                rv = cell.value
                if rv is None:
                    continue
                coord = cell.coordinate
                if _is_excel_error_value(rv):
                    ov = ws_orig[coord].value
                    if ov is not None and not _is_excel_error_value(ov):
                        sheet_heals.append((coord, ov))
                    else:
                        # No usable Excel cache: do not require the solver to
                        # match LO's error token (false-negative trap). Clear
                        # to empty so only a real value from the solver can
                        # still fail — empty/None matches.
                        sheet_heals.append((coord, None))
                else:
                    freeze.append((coord, rv))
        if not sheet_heals:
            continue
        for coord, val in sheet_heals:
            ws_w[coord].value = val
            healed += 1
        for coord, val in freeze:
            ws_w[coord].value = val
        dirty = True

    if dirty:
        try:
            wb_write.save(recalc_path)
        except Exception as e:
            logger.warning(
                f"Golden LO-error heal failed to save {recalc_path}: {e}"
            )
            return 0
        logger.info(
            f"Healed {healed} LO error cell(s) in golden {recalc_path.name} "
            f"from Excel-cached values in {original_path.name}"
        )
    return healed


def _libreoffice_recalc_inplace(xlsx_path: str, timeout: int = 60) -> bool:
    """Recalculate formulas in an xlsx in place via LibreOffice in Docker.

    Uses the ssb-verifier image (has LibreOffice installed). Returns False
    on failure, leaving the file unchanged — callers must not treat this
    silently: a failed recalc means formula cells will read as None under
    data_only=True, indistinguishable from a genuine wrong answer unless
    this warning is checked in the logs.
    """
    xlsx_path = str(Path(xlsx_path).resolve())
    xlsx_dir = str(Path(xlsx_path).parent)
    xlsx_name = Path(xlsx_path).name
    try:
        with _DOCKER_SEMAPHORE:
            result = subprocess.run(
                ["docker", "run", "--rm",
                 "-v", f"{xlsx_dir}:/work",
                 "-w", "/work",
                 "ssb-verifier:latest",
                 "bash", "-c",
                 # cp by exact name: /tmp/*.xlsx never matches the dot-prefixed
                 # temp names _recalc_answer_cached feeds us (glob skips hidden
                 # files), so every golden recalc "failed" despite a successful
                 # convert. soffice keeps the input basename (already .xlsx).
                 f'libreoffice --headless --calc --convert-to "xlsx:Calc MS Excel 2007 XML" --outdir /tmp "{xlsx_name}" 2>/dev/null && cp "/tmp/{xlsx_name}" "{xlsx_name}"'],
                capture_output=True, text=True, timeout=timeout,
            )
        if result.returncode == 0:
            return True
        logger.warning(
            f"LibreOffice recalculation failed (exit {result.returncode}) for {xlsx_path}: "
            f"stdout={result.stdout[-500:]!r} stderr={result.stderr[-500:]!r}"
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.warning(f"LibreOffice recalculation failed for {xlsx_path}: {e}")
    return False


def _recalc_answer_cached(golden_path: Path, task_id: str) -> Path:
    """Return a formula-recalculated (+ LO-error-healed) golden copy (cached).

    test.sh recalculates answer files inside the verifier container before
    comparing; that recalculated copy never leaves the container. Host-side
    detailed diffs must mirror it, or golden formula cells without cached
    values read as None and produce spurious mismatches. Recalculated
    copies are cached under the tasks cache (wiped per run), so each answer
    file costs one LibreOffice call per run. On recalc failure the plain
    copy is cached with a warning — exactly test.sh's own fallback.

    After a successful LO pass, cells where LO wrote an Excel error token
    but the unrecalculated golden still has a usable Excel-cached value are
    rewritten to that cached value (see `_heal_golden_lo_errors`). Without
    this, tasks like 46646/31011/39903 falsely require the solver to match
    `#VALUE!`/`#NAME?`.
    """
    golden_path = Path(golden_path)
    dst = SSB_TASKS_CACHE / "_answers_recalc" / task_id / golden_path.name
    if dst.exists():
        # Re-heal in place so caches written before this fix (or a LO that
        # re-introduced errors) still score correctly.
        _heal_golden_lo_errors(dst, golden_path)
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    import uuid as _uuid
    tmp = dst.parent / f".recalc_{os.getpid()}_{_uuid.uuid4().hex[:8]}_{golden_path.name}"
    shutil.copy2(golden_path, tmp)
    if not _libreoffice_recalc_inplace(str(tmp)):
        logger.warning(
            f"Answer recalc failed for {golden_path} — caching the un-recalculated "
            f"copy (test.sh fallback semantics); golden formula cells without "
            f"cached values will read as None in DIFF.txt"
        )
    else:
        _heal_golden_lo_errors(tmp, golden_path)
    os.replace(tmp, dst)
    return dst


def _prepare_healed_tests_overlay(task_dir: Path) -> Path:
    """Writable tests/ overlay with LO-error-healed answer files.

    Docker test.sh always re-LO's `/tests/answer_files`. Mounting the stock
    SpreadsheetBench answers therefore re-breaks cells we already healed on
    the host. Overlay the task's tests/ with healed answer copies so the
    in-container LO pass sees literals (or already-good values) instead of
    the original formulas that LO cannot evaluate.
    """
    import uuid as _uuid

    task_dir = Path(task_dir)
    tests_dir = task_dir / "tests"
    task_id = task_dir.name
    overlay = (
        SSB_TASKS_CACHE / "_verifier_tests" / f"{task_id}_{_uuid.uuid4().hex[:10]}"
    )
    if overlay.exists():
        shutil.rmtree(overlay, ignore_errors=True)
    shutil.copytree(tests_dir, overlay)
    answer_dir = overlay / "answer_files"
    if answer_dir.is_dir():
        for ans in sorted(answer_dir.glob("*.xlsx")):
            # Heal from the ORIGINAL SpreadsheetBench golden, not the overlay
            # copy (identical content, but keep the source of truth explicit).
            original = tests_dir / "answer_files" / ans.name
            healed = _recalc_answer_cached(original, task_id)
            shutil.copy2(healed, ans)
    return overlay


def _read_task_config(task_dir: Path) -> dict:
    return json.loads((Path(task_dir) / "tests" / "config.json").read_text())


def render_task_diff(
    task_dir: Path, output_dir: Path, max_mismatches: int = 0
) -> tuple[float, str]:
    """Detailed per-test-case diff body for a task's output dir.

    Returns (soft_estimate, body). soft_estimate is the mean of per-case
    pass booleans from the HOST-side detailed compare — advisory only; the
    Docker verifier's reward is authoritative (score_and_diff reconciles
    the two and flags disagreement). No recalculation is performed here:
    output files must already carry cached values (true after
    _run_verifier_v2 ran — it recalculates them in place on the bind
    mount — and for snapshot dirs, which are captured post-recalc).
    """
    task_dir = Path(task_dir)
    output_dir = Path(output_dir)
    cfg = _read_task_config(task_dir)
    task_id = str(cfg["id"])
    answer_position = cfg.get("answer_position", "")
    num_cases = int(cfg.get("num_test_cases", 3))  # evaluate.py's own default

    sections: list[str] = []
    case_results: list[bool] = []
    for tc in range(1, num_cases + 1):
        out_name = f"{tc}_{task_id}_output.xlsx"
        golden = task_dir / "tests" / "answer_files" / f"{tc}_{task_id}_answer.xlsx"
        out_path = output_dir / out_name
        if not golden.exists():
            case_results.append(False)
            sections.append(
                f"Test case {tc}: SKIP (answer file missing) — counts as FAIL, "
                f"matching evaluate.py"
            )
            continue
        if not out_path.exists():
            case_results.append(False)
            present = sorted(p.name for p in output_dir.glob("*.xlsx")) or "none"
            sections.append(
                f"Test case {tc} ({out_name}): FAIL — output file MISSING. The "
                f"verifier requires this EXACT name; .xlsx files present in "
                f"output/: {present}"
            )
            continue
        gt_path = _recalc_answer_cached(golden, task_id)
        passed, n_match, n_total, diff = compare_workbooks_detailed(
            str(gt_path), str(out_path), answer_position, max_mismatches
        )
        case_results.append(bool(passed))
        sections.append(
            f"Test case {tc} ({out_name}): {'PASS' if passed else 'FAIL'} — "
            f"{n_match}/{n_total} graded cells match\n{diff}"
        )
    soft = sum(case_results) / len(case_results) if case_results else 0.0
    return soft, "\n\n".join(sections)


def score_and_diff(
    task_dir: Path, output_dir: Path, max_mismatches: int = 0
) -> tuple[float, str]:
    """Verifier reward AND detailed diff in one scoring pass.

    Runs the Docker verifier once (authoritative reward — evaluate.py inside
    ssb-verifier, after test.sh recalculates both output and answer files;
    the output recalc happens in place on the bind mount, so the host copies
    the evolver later reads are the recalculated ones), then renders the
    full cell-level mismatch list host-side with the same comparison
    semantics. Persisting that diff (see snapshot_output) spares the evolver
    from re-running the same multi-minute comparison over every failed task
    (Jul 1 SSB v2: one such batch re-computation blew the 120s bash timeout).

    Returns (reward, diff_text). reward is the verifier's soft reward (mean
    over test cases; every task in this dataset has 1 case, so effectively
    0.0/1.0).
    """
    task_dir = Path(task_dir)
    output_dir = Path(output_dir)
    reward = _run_verifier_v2(task_dir, output_dir)
    try:
        soft_est, body = render_task_diff(task_dir, output_dir, max_mismatches)
    except Exception as e:
        soft_est, body = reward, f"(detailed diff unavailable: {e})"
    header = (
        f"{'PASS' if reward >= 0.999 else 'FAIL'} — verifier reward {reward:.2f}\n"
        f"(scorer's own diff, computed at scoring time; full, uncapped)\n"
    )
    if abs(soft_est - reward) > 1e-6:
        header += (
            f"NOTE: host-side diff estimate ({soft_est:.2f}) disagrees with the "
            f"verifier reward ({reward:.2f}) — trust the verifier; the cell diff "
            f"below is advisory.\n"
        )
    return reward, f"{header}\n{body}\n"


def score_spreadsheetbench(question: str, predicted: str, ground_truth: str) -> float:
    """Sync scorer for the EvoSkill loop.

    `ground_truth` is the task_id (e.g., "49237").
    `predicted` is the solver's final_answer text (ignored for scoring —
    the verifier checks the output .xlsx file on disk).

    The scorer looks for output files at SOLVER_OUTPUT_ROOT / task_id / output/.
    Set SOLVER_OUTPUT_ROOT before running.
    """
    # ground_truth is either a plain task_id or JSON with task_id + golden_path
    try:
        gt_info = json.loads(ground_truth)
        task_id = gt_info["task_id"]
    except (json.JSONDecodeError, TypeError, KeyError):
        task_id = ground_truth
    task_dir = _get_task_dir(task_id)
    if task_dir is None:
        logger.warning(f"Unknown task: {task_id}")
        return 0.0

    from src.agent_profiles.spreadsheetbench_agent.task_workspace import get_output_dir
    output_dir = get_output_dir(task_id, SSB_TASKS_CACHE)
    if not output_dir.is_dir() or not any(output_dir.glob("*.xlsx")):
        logger.warning(f"No output .xlsx for task {task_id} at {output_dir}")
        return 0.0

    return _run_verifier_v2(task_dir, output_dir)


async def score_spreadsheetbench_async(
    question: str, predicted: str, ground_truth: str
) -> float:
    """Async wrapper for inline scoring."""
    return await asyncio.get_event_loop().run_in_executor(
        None, score_spreadsheetbench, question, predicted, ground_truth,
    )
