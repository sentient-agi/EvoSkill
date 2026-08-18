"""Stage 2 (must run under /usr/bin/python3 -- the system interpreter
python3-uno was installed against, NOT the image's default python3.11 which
only has openpyxl).

Drives LibreOffice's real (but Excel-import-broken) two-variable Data Table
feature by hand: for every output cell, temporarily overwrite the table's
two input cells with that cell's row/column header values, force a
recalculation, and read the corner formula's freshly-computed result back --
this is what Excel's Data Table feature does internally, and LibreOffice's
own formula engine gets it right once driven this way (confirmed empirically
against golden: 11515.71286542351 vs golden's 11515.7128654235, bit-exact).
LibreOffice's own import of the Excel TABLE() token never performs this
substitution itself -- it just reads the corner formula once under whatever
the input cells' current base-case values are and copies that single number
into the entire output range, which is the bug this script routes around.

CRITICAL: calculateAll() does NOT reliably force a full dependency-graph
recalculation on the second and later call within the same open document --
confirmed empirically (reproducible signature: whichever cell is computed
FIRST in a session comes out correct, every subsequent one in that same
session is wrong, regardless of which cell it is). So each cell's
computation opens its OWN fresh document, computes, and closes without
saving; only a final pass opens the document once more, writes every
computed result as a literal, and saves. This is slower (N+1 document loads
instead of 1) but is the only version that has been verified correct.

Reads the per-cell JSON produced by datatable_scan.py (each entry is fully
self-describing: corner/row_input/row_header/col_input/col_header cell
refs). The final save path never touches openpyxl, so it cannot trigger the
separate "data_only=True save wipes every other formula in the file" bug
found elsewhere in this scorer.

Usage: python3 datatable_uno_solve.py <xlsx_path> <cells_json_path>
Exits 0 with "no tables" if the JSON list is empty (nothing to do).
"""
import json
import sys
import time

sys.path.insert(0, "/usr/lib/python3/dist-packages")
import uno
from com.sun.star.beans import PropertyValue


def mkprop(name, value):
    p = PropertyValue()
    p.Name = name
    p.Value = value
    return p


def connect(port=2002, retries=20, delay=0.5):
    local_ctx = uno.getComponentContext()
    resolver = local_ctx.ServiceManager.createInstanceWithContext(
        "com.sun.star.bridge.UnoUrlResolver", local_ctx)
    last_err = None
    for _ in range(retries):
        try:
            ctx = resolver.resolve(
                f"uno:socket,host=localhost,port={port};urp;StarOffice.ComponentContext")
            desktop = ctx.ServiceManager.createInstanceWithContext(
                "com.sun.star.frame.Desktop", ctx)
            return desktop
        except Exception as e:
            last_err = e
            time.sleep(delay)
    raise RuntimeError(f"could not connect to soffice listener: {last_err}")


def compute_one(desktop, url, entry):
    doc = desktop.loadComponentFromURL(url, "_blank", 0, (mkprop("Hidden", True),))
    try:
        sheet = doc.Sheets.getByName(entry["sheet"])
        row_header_val = sheet.getCellRangeByName(entry["row_header"]).getValue()
        col_header_val = sheet.getCellRangeByName(entry["col_header"]).getValue()
        sheet.getCellRangeByName(entry["row_input"]).setValue(row_header_val)
        sheet.getCellRangeByName(entry["col_input"]).setValue(col_header_val)
        doc.calculateAll()
        return sheet.getCellRangeByName(entry["corner"]).getValue()
    finally:
        doc.close(False)


def solve(xlsx_path: str, cell_entries: list[dict]) -> int:
    desktop = connect()
    url = "file://" + xlsx_path

    results = []
    for e in cell_entries:
        val = compute_one(desktop, url, e)
        results.append((e["sheet"], e["cell"], val))

    doc = desktop.loadComponentFromURL(url, "_blank", 0, (mkprop("Hidden", True),))
    try:
        for sheet_name, cell_ref, val in results:
            doc.Sheets.getByName(sheet_name).getCellRangeByName(cell_ref).setValue(val)
        doc.calculateAll()
        doc.store()
    finally:
        doc.close(False)
    return len(results)


if __name__ == "__main__":
    xlsx_path, cells_json = sys.argv[1], sys.argv[2]
    with open(cells_json) as f:
        cell_entries = json.load(f)
    if not cell_entries:
        print("no tables")
        sys.exit(0)
    n = solve(xlsx_path, cell_entries)
    print(f"solved {n} cell(s)")
