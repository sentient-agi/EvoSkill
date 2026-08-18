"""Normalise Excel formula constructs LibreOffice cannot evaluate.

WHY THIS EXISTS
---------------
SSB v1 grades a solver's .xlsx by recalculating it with LibreOffice and
comparing graded cells against an Excel-authored golden. Those two engines do
not accept the same formula surface, and the mismatch is not symmetric:

  * The GOLDEN was written by Excel, so every formula cell also carries an
    Excel-computed cached value. When LO fails to evaluate a golden formula the
    harness heals the cell back to that cached value (see task_workspace.py).
  * The SOLVER's file is written by openpyxl, which computes nothing. There is
    no cached value to heal from, so any formula LO cannot evaluate reads as an
    error token and scores as a wrong answer.

Net effect: the golden is graded under Excel semantics and the solver under
LibreOffice semantics. A formula that is correct in Excel can score 0.

This module closes that gap for constructs that are *unambiguously* rewritable
into a form LO evaluates identically. It is a *representation* change only --
it never changes what a formula computes, so it cannot turn a wrong answer into
a right one. The rewritten value still has to match the golden.

WHAT IS DELIBERATELY NOT FIXED
------------------------------
Post-2007 function names written without the OOXML ``_xlfn.`` prefix (bare
``XLOOKUP``, ``AGGREGATE``, ``TEXTJOIN``, ...). Such a file is malformed per
the OOXML spec and **Excel also reports #NAME?** on it. The harness agreeing
with Excel is correct behaviour, not a defect; "fixing" it would make the
scorer more lenient than the application the benchmark is about. Verified
against LibreOffice 25.2.3.2 in ssb-verifier:latest -- every one of those
functions evaluates correctly once the prefix is present, so the failure is
about file encoding, not engine capability.
"""

from __future__ import annotations

import logging
import re

from openpyxl.utils import get_column_letter

logger = logging.getLogger(__name__)

# Table1[@[Col Name]]  /  Table1[@Col]  /  Table1[[#This Row],[Col]]
_THIS_ROW = re.compile(
    r"(?P<table>[A-Za-z_][\w.]*)\s*\[\s*(?:"
    r"@\s*\[(?P<b1>[^\]]+)\]"          # Table1[@[Col Name]]
    r"|@\s*(?P<b2>[^\]\[,]+?)"          # Table1[@Col]
    r"|\[\s*\#This\s+Row\s*\]\s*,\s*\[(?P<b3>[^\]]+)\]"   # Table1[[#This Row],[Col]]
    r")\s*\]",
    re.IGNORECASE,
)


def _table_columns(table, ws=None) -> list[str]:
    """Column names in table order.

    openpyxl only materialises ``tableColumns`` for tables round-tripped
    through a file; a Table added in memory has none. Fall back to reading the
    table's own header row off the sheet so both cases behave identically.
    """
    for attr in ("tableColumns", "column_names"):
        try:
            cols = getattr(table, attr, None) or []
            names = [getattr(c, "name", c) for c in cols]
            if names:
                return [str(n) for n in names]
        except Exception:
            pass

    if ws is None:
        return []
    try:
        from openpyxl.utils import range_boundaries

        min_col, min_row, max_col, _ = range_boundaries(str(table.ref))
        header_row = min_row if not getattr(table, "headerRowCount", 1) == 0 else None
        if header_row is None:
            return []
        return [
            str(ws.cell(row=header_row, column=c).value)
            for c in range(min_col, max_col + 1)
        ]
    except Exception:
        return []


def rewrite_this_row_refs(ws) -> int:
    """Rewrite this-row structured references in ``ws`` to plain A1 refs.

    LibreOffice resolves ``Table1[Col]`` (whole column) but NOT the this-row
    forms, which it reports as #N/A -- verified in the scorer image. Excel
    generates the this-row form automatically whenever a formula is typed into
    a table column, so solvers reproduce it constantly.

    The rewrite is exact: ``Table1[@[Col]]`` in row R becomes the address of
    ``Col`` in row R of that table, which is precisely what Excel resolves it
    to. Returns the number of cells changed.
    """
    tables = list(getattr(ws, "tables", {}).values()) if hasattr(ws, "tables") else []
    if not tables:
        return 0

    # name -> (first_col_index, {lowercased column name: absolute col index})
    spec: dict[str, dict[str, int]] = {}
    for t in tables:
        try:
            first_cell, _, _ = str(t.ref).partition(":")
            from openpyxl.utils import coordinate_to_tuple

            _, first_col = coordinate_to_tuple(first_cell)
        except Exception:
            continue
        cols = _table_columns(t, ws)
        if not cols:
            continue
        spec[str(t.displayName).lower()] = {
            str(name).strip().lower(): first_col + i for i, name in enumerate(cols)
        }

    if not spec:
        return 0

    changed = 0
    for row in ws.iter_rows():
        for cell in row:
            f = cell.value
            if not isinstance(f, str) or not f.startswith("="):
                continue

            def _sub(m: re.Match) -> str:
                cols = spec.get(m.group("table").lower())
                if cols is None:
                    return m.group(0)  # unknown table -- leave untouched
                col_name = (m.group("b1") or m.group("b2") or m.group("b3") or "").strip()
                idx = cols.get(col_name.lower())
                if idx is None:
                    return m.group(0)  # unknown column -- leave untouched
                return f"{get_column_letter(idx)}{cell.row}"

            new = _THIS_ROW.sub(_sub, f)
            if new != f:
                cell.value = new
                changed += 1
    return changed


# SUM(IF(...)), MEDIAN(IF(...)), ... -- an aggregate applied to a conditional
# over a RANGE. This is the classic CSE construct: it only evaluates correctly
# under array semantics.
_AGG_OVER_IF = re.compile(
    r"(?<![A-Z_.])(SUM|MEDIAN|MAX|MIN|AVERAGE|COUNT|COUNTA|LARGE|SMALL|PRODUCT"
    r"|STDEV|VAR|AVERAGEA)\s*\(\s*IF\s*\(",
    re.IGNORECASE,
)


def mark_implicit_array_formulas(ws) -> int:
    """Mark aggregate-over-IF formulas as array formulas.

    Excel 365's dynamic arrays evaluate ``=MEDIAN(IF(A2:A6>=C1,A2:A6))`` typed
    normally -- Ctrl+Shift+Enter has not been required since dynamic arrays
    shipped, and the SSB goldens are Excel-authored. LibreOffice has no such
    implicit-array evaluation: the same formula returns ``#VALUE!`` unless the
    cell is stored with ``t="array"``. Verified against ssb-verifier's
    LibreOffice 25.2.3.2 -- identical formula, ``#VALUE!`` unmarked vs the
    correct value marked.

    So an answer that is correct in the application the benchmark is about
    scores 0 purely on how openpyxl serialised it. Marking restores the
    semantics Excel already applies; it does not change what is computed, so a
    wrong formula stays wrong.

    Only formulas matching an aggregate directly wrapping ``IF(`` are touched --
    a construct that genuinely requires array context in every engine. Returns
    the number of cells marked.
    """
    from openpyxl.worksheet.formula import ArrayFormula

    changed = 0
    for row in ws.iter_rows():
        for cell in row:
            f = cell.value
            if not isinstance(f, str) or not f.startswith("="):
                continue  # already an ArrayFormula object, or not a formula
            if not _AGG_OVER_IF.search(f):
                continue
            cell.value = ArrayFormula(cell.coordinate, f)
            changed += 1
    return changed


def normalise_workbook(wb) -> int:
    """Apply every safe rewrite to ``wb`` in place; return cells changed.

    Order matters: this-row references are resolved to A1 form FIRST, so that a
    formula which is both a this-row reference and an aggregate-over-IF is
    array-marked with its references already rewritten.
    """
    total = 0
    for ws in wb.worksheets:
        for fn in (rewrite_this_row_refs, mark_implicit_array_formulas):
            try:
                total += fn(ws)
            except Exception as e:  # never let normalisation break scoring
                logger.warning(
                    "formula normalisation (%s) failed on sheet %r: %s -- "
                    "scoring proceeds on the un-normalised sheet",
                    fn.__name__, getattr(ws, "title", "?"), e,
                )
    return total
