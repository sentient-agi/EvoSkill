#!/bin/bash
# Orchestrates the two-stage Data Table fix inside the ssb-verifier container.
# arg1: xlsx filename (relative to /work, the mounted xlsx directory)
set -e
XLSX_NAME="$1"
TABLES_JSON="/tmp/datatable_scan_$$.json"

soffice --headless --invisible --nologo --nofirststartwizard \
  --accept="socket,host=localhost,port=2002;urp;" --norestore \
  > /tmp/soffice_$$.log 2>&1 &
SOFFICE_PID=$!
trap "kill $SOFFICE_PID 2>/dev/null || true" EXIT

python3 /scripts/datatable_scan.py "/work/$XLSX_NAME" "$TABLES_JSON"

N_TABLES=$(python3 -c "import json; print(len(json.load(open('$TABLES_JSON'))))")
if [ "$N_TABLES" = "0" ]; then
  echo "no data tables found, nothing to do"
  exit 0
fi

/usr/bin/python3 /scripts/datatable_uno_solve.py "/work/$XLSX_NAME" "$TABLES_JSON"
