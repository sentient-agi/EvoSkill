# ssb-verifier:latest — the SpreadsheetBench v2 grading image.
#
# Build:  docker build -t ssb-verifier:latest docker/ssb-verifier/
#
# The scorer runs `libreoffice --convert-to` inside this image to recalculate a
# solver's output workbook before comparing cell values against the reference.
#
# python3-uno is REQUIRED: the scorer's Data-Table solve
# (src/agent_profiles/ssbv2_agent/docker_scripts/run_datatable_fix.sh) calls
# /usr/bin/python3 — Debian's python, the only one with UNO bindings. Without it
# every Data-Table UNO solve fails exit-127 and silently falls back to a plain
# recalc, which cannot compute =TABLE() cells.
# BASE PINNED BY DIGEST, not by tag. `python:3.11-slim` is a moving pointer: two builds
# a month apart inherit different OS packages, so an image built from a tag is not the
# image the numbers were produced on, and nothing in the pipeline would say so. Refresh
# deliberately with:
#     docker manifest inspect -v python:3.11-slim | python3 -c "import json,sys; print(json.load(sys.stdin)['Descriptor']['digest'])"
# Resolved 2026-08-18.
FROM python:3.11-slim@sha256:ff05d1a05204fb9f7444c435db8e8ec104e587a413280dc9ffc27a4797554182
RUN apt-get update && apt-get install -y --no-install-recommends libreoffice-calc python3-uno && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    soffice --version > /etc/evoskill_libreoffice_version
# The grading engine's version, recorded because it cannot be pinned (apt carries one
# libreoffice-calc per Debian release). Recover it from any built image with
# `docker run --rm ssb-verifier:latest cat /etc/evoskill_libreoffice_version`; two results
# graded by different versions are not strictly comparable, and this is what makes that
# answerable rather than unknowable. Same treatment in Dockerfile.libreoffice.
RUN pip install --no-cache-dir openpyxl==3.1.3 pandas==2.2.0 numpy==1.26.4
WORKDIR /app
RUN mkdir -p /app/spreadsheets /app/output /logs/verifier /tests

# Recalc-on-load, always. By default `libreoffice --convert-to` does NOT
# recompute a formula cell that carries a cached <v> value — it trusts the
# cache, so a workbook whose formulas were edited without recomputation would
# be graded on stale numbers. OOXMLRecalcMode=0 / ODFRecalcMode=0 in the user
# profile make LibreOffice recalculate every formula on load. This mutates no
# workbook: cached values are ignored, not stripped.
#
# (Note: patching <calcPr fullCalcOnLoad="1"> into xl/workbook.xml at the zip
# level does NOT work — LibreOffice ignores it. The profile setting is the only
# reliable mechanism.)
RUN mkdir -p /root/.config/libreoffice/4/user \
 && printf '%s\n' \
 '<?xml version="1.0" encoding="UTF-8"?>' \
 '<oor:items xmlns:oor="http://openoffice.org/2001/registry" xmlns:xs="http://www.w3.org/2001/XMLSchema" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">' \
 '<item oor:path="/org.openoffice.Office.Calc/Formula/Load"><prop oor:name="OOXMLRecalcMode" oor:op="fuse"><value>0</value></prop></item>' \
 '<item oor:path="/org.openoffice.Office.Calc/Formula/Load"><prop oor:name="ODFRecalcMode" oor:op="fuse"><value>0</value></prop></item>' \
 '</oor:items>' \
 > /root/.config/libreoffice/4/user/registrymodifications.xcu

# Assert the profile is where LibreOffice will look for it, at BUILD time. A
# silently mislocated profile would leave the stale-cache behaviour in place
# while the image looked fixed.
RUN test -s /root/.config/libreoffice/4/user/registrymodifications.xcu \
 && grep -q 'OOXMLRecalcMode' /root/.config/libreoffice/4/user/registrymodifications.xcu \
 && grep -q '<value>0</value>' /root/.config/libreoffice/4/user/registrymodifications.xcu
