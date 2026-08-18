#!/usr/bin/env bash
# Build ssb-solver-lo:latest — the SSB v2 solver image WITH a formula engine —
# and refuse to hand it over unless every property the experiment depends on
# actually holds inside the built container.
#
# WHY A GATE AND NOT JUST `docker build`. The RunCache tree_hash is
# skills-dir-hash + solver SYSTEM PROMPT (src/cache/run_cache.py::_get_tree_hash).
# It does NOT include the docker image. So a solver image that builds but is
# subtly wrong — Debian's python3 shadowing the one with openpyxl, a soffice that
# resolves but cannot convert, python3-uno missing the way it went missing from
# ssb-verifier on 2026-07-15 — produces rollouts that are cached, replayed and
# reported as if the environment were fine. There is no downstream check that
# would catch it. This is that check, and it runs BEFORE any measurement.
#
# NO OVERRIDE ENV VAR, deliberately. The 2026-07-24 factorial measured an escape
# hatch being taken 72/72 times when one was offered.
#
# Usage:  scripts/build_ssb_solver_lo.sh [--no-cache]

set -euo pipefail

ROOT="${EVOSKILL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
BASE_TAG="ssb-solver:no-libreoffice-0715"
BASE_ID_EXPECTED="974b00f17fbe"      # the 2026-07-15 image every past SSB v2 number was produced on
BASE_LOCAL_TAG="ssb-solver:no-libreoffice-local"   # a base THIS machine built; never the archival name
BASE_USED="$BASE_TAG"                # which of the two the LibreOffice layer is built FROM
IMAGE="ssb-solver-lo:latest"
DF="$ROOT/docker/ssb-solver/Dockerfile.libreoffice"

# Pinned in the baseline image; asserted below because python3-uno drags in a
# SECOND python and a shadowed interpreter is the failure mode with no symptom.
WANT_OPENPYXL="3.1.3"
WANT_PANDAS="2.2.0"
WANT_NUMPY="1.26.4"
WANT_PY="3.11"

fail() { echo "FAILED: $*" >&2; exit 9; }
ok()   { echo "  ok    $*"; }

echo "== preconditions =="

# 1. The corpus-only firewall is a DOCKER-USER default-DROP chain that persists
#    in the Colima VM until a restart. With it installed, apt inside the build
#    dies on a network timeout that reads like a mirror problem.
CHAIN_RULES=$(colima ssh -- sudo iptables -S DOCKER-USER 2>/dev/null | grep -cvE '^-(N|P) ' || true)
: "${CHAIN_RULES:=0}"
if [ "$CHAIN_RULES" -gt 0 ]; then
  fail "the DOCKER-USER firewall chain has $CHAIN_RULES rule(s) — apt egress is blocked.
  This is executor.py::_ensure_corpus_only_firewall's chain, left behind by a run
  with EVOSKILL_CORPUS_ONLY=1. Inspect with
    colima ssh -- sudo iptables -S DOCKER-USER
  and clear it (or restart colima) before building."
fi
ok "DOCKER-USER chain is empty — build egress is open"

# 2. The base. Two situations, and conflating them made this script unrunnable
#    off the machine the images were first built on.
#
#    (a) REPRODUCTION. If the frozen 2026-07-15 image is present, require it by
#        ID. Every SSB v2 number published to date was produced on that exact
#        image, so building the LibreOffice arm on anything else would make the
#        comparison two-variable — which is what the ID pin has always been for.
#
#    (b) FRESH MACHINE. If it is absent, the pin cannot be satisfied by any
#        `docker build`: an image ID is a content hash over layers that include
#        apt's index of the day. Until 2026-08-18 this script simply failed there,
#        which meant step 2 of the documented SSB v2 setup could not succeed for
#        anyone who was not on this laptop. Build the base from its versioned
#        Dockerfile instead and say plainly what that does and does not buy.
#
#    What survives in case (b) is the property that actually matters downstream:
#    the acceptance suite below pins python/openpyxl/pandas/numpy to the same
#    versions and proves the recalc round-trip end to end. What does NOT survive
#    is byte-identity with the historical arm — so a number produced on a fresh
#    base is comparable within its own run, and is NOT directly comparable to a
#    published figure. That is stated at build time rather than left to be
#    rediscovered from a puzzling delta.
BASE_ID=$(docker images --no-trunc --format '{{.ID}}' "$BASE_TAG" 2>/dev/null | head -1 | sed 's/^sha256://' | cut -c1-12)
if [ -z "$BASE_ID" ]; then
  echo "  note  base tag $BASE_TAG is absent — building it from"
  echo "        docker/ssb-solver/Dockerfile (the versioned recipe for that image)."
  echo "        The frozen 2026-07-15 image ($BASE_ID_EXPECTED) cannot be reproduced by"
  echo "        ID on another machine; the library pins asserted below are what make"
  echo "        this base equivalent for running the benchmark."
  echo "        RESULTS FROM THIS BASE ARE NOT BYTE-COMPARABLE TO PUBLISHED NUMBERS."
  # TAG IT LOCALLY, NEVER WITH THE ARCHIVAL NAME. Tagging a fresh build as
  # $BASE_TAG made this script single-use: run 1 built and tagged, run 2 saw that
  # tag holding a non-frozen id and refused at exit 9 ("expected 974b00f17fbe").
  # Since every acceptance failure here exits 9, re-running is the normal operator
  # action, so a fresh machine bricked on the second invocation. The archival tag
  # now means exactly one thing — the frozen image — and a local build gets its
  # own name that the next run can safely rebuild.
  docker build -t "$BASE_LOCAL_TAG" "$ROOT/docker/ssb-solver/" \
    || fail "could not build the base image from docker/ssb-solver/Dockerfile"
  BASE_ID=$(docker images --no-trunc --format '{{.ID}}' "$BASE_LOCAL_TAG" | head -1 | sed 's/^sha256://' | cut -c1-12)
  [ -n "$BASE_ID" ] || fail "base build reported success but $BASE_LOCAL_TAG does not exist"
  BASE_USED="$BASE_LOCAL_TAG"
  ok "base $BASE_LOCAL_TAG = $BASE_ID (freshly built; NOT the frozen 2026-07-15 image)"
elif [ "$BASE_ID" = "$BASE_ID_EXPECTED" ]; then
  BASE_USED="$BASE_TAG"
  ok "base $BASE_TAG = $BASE_ID (the frozen 2026-07-15 solver image)"
else
  # Present but different: someone retagged or rebuilt it. Refuse, because this is
  # the one case where the operator probably BELIEVES they have the frozen image.
  fail "base $BASE_TAG is $BASE_ID, expected $BASE_ID_EXPECTED.
  That tag is supposed to name the frozen 2026-07-15 solver image, and it does
  not. Either it was rebuilt (the historical identity is gone — untag it and let
  this script build a fresh base, accepting that results are not byte-comparable
  to published numbers), or it was tagged onto the wrong image."
fi

echo
echo "== build =="
docker build ${1:-} -f "$DF" -t "$IMAGE" --build-arg "BASE=$BASE_USED" "$ROOT/docker/ssb-solver/" \
  || fail "docker build failed"
NEW_ID=$(docker images --format '{{.ID}}' "$IMAGE" | head -1)
echo "built $IMAGE = $NEW_ID"

# A container that stays up, so every check below runs against ONE instance
# rather than 12 fresh ones (and so a check can leave a file for the next).
CTR="ssb-lo-acceptance-$$"
cleanup() { docker rm -f "$CTR" >/dev/null 2>&1 || true; }
trap cleanup EXIT
docker run -d --rm --name "$CTR" "$IMAGE" sleep 600 >/dev/null \
  || fail "the built image will not even start a container"

dex() { docker exec "$CTR" bash -c "$1"; }

echo
echo "== acceptance: the formula engine =="

SOFF=$(dex 'command -v soffice || true')
[ -n "$SOFF" ] || fail "soffice is not on PATH — this is the whole point of the image"
ok "soffice -> $SOFF"

LOVER=$(dex 'cat /etc/evoskill_libreoffice_version 2>/dev/null || soffice --version')
ok "LibreOffice ${LOVER:-UNKNOWN}  (the grading engine; not version-pinned — see the note
        in Dockerfile.libreoffice. Record this next to any number produced on this image:
        results graded by two different LibreOffice versions are not strictly comparable.)"

LIBRE=$(dex 'command -v libreoffice || true')
[ -n "$LIBRE" ] || fail "libreoffice is not on PATH.
  The scorer invokes the command as \`libreoffice\`, not \`soffice\`
  (scorer.py::_recalculate_with_libreoffice) — the solver must be able to use the
  same spelling it will see in any guidance."
ok "libreoffice -> $LIBRE"

# python3-uno: went missing from ssb-verifier's recipe once and cost 17/157
# verdicts. Check the IMPORT, not the package name — dpkg can be satisfied while
# the bindings are unusable.
dex '/usr/bin/python3 -c "import uno; print(uno.__name__)"' >/dev/null 2>&1 \
  || fail "/usr/bin/python3 cannot import uno — the Data-Table solve path would
  fail exit-127 and fall back silently, exactly as ssb-verifier did on 2026-07-15."
ok "/usr/bin/python3 imports uno"

echo
echo "== acceptance: the spreadsheet stack survived the second python =="

PY3=$(dex 'command -v python3')
[ "$PY3" = "/usr/local/bin/python3" ] || fail "python3 resolves to $PY3, not /usr/local/bin/python3.
  python3-uno installs Debian's python3 at /usr/bin/python3, which has NO
  openpyxl/pandas/numpy. If it shadows the pinned 3.11 the solver's very first
  \`import openpyxl\` fails and every rollout is lost to an environment error."
ok "python3 -> $PY3"

VERS=$(dex 'python3 -c "
import sys, openpyxl, pandas, numpy
print(\"%d.%d\" % sys.version_info[:2], openpyxl.__version__, pandas.__version__, numpy.__version__)"') \
  || fail "python3 cannot import the spreadsheet stack"
read -r GOT_PY GOT_OP GOT_PD GOT_NP <<<"$VERS"
[ "$GOT_PY" = "$WANT_PY" ]       || fail "python $GOT_PY, expected $WANT_PY"
[ "$GOT_OP" = "$WANT_OPENPYXL" ] || fail "openpyxl $GOT_OP, expected $WANT_OPENPYXL"
[ "$GOT_PD" = "$WANT_PANDAS" ]   || fail "pandas $GOT_PD, expected $WANT_PANDAS"
[ "$GOT_NP" = "$WANT_NUMPY" ]    || fail "numpy $GOT_NP, expected $WANT_NUMPY"
ok "python $GOT_PY · openpyxl $GOT_OP · pandas $GOT_PD · numpy $GOT_NP (identical to baseline)"

echo
echo "== acceptance: recalculation actually computes a value =="

# The end-to-end property, not a proxy for it: openpyxl writes a formula with NO
# cached value, so data_only=True reads None — the exact condition the solver was
# stuck in. Then LibreOffice recalculates and the same read returns a number.
# Both halves are asserted, because "reads a number" is only meaningful if the
# pre-state was really None.
dex 'python3 - <<'"'"'PY'"'"'
import subprocess, sys
from openpyxl import Workbook, load_workbook

wb = Workbook(); ws = wb.active
ws["A1"], ws["A2"] = 2, 40
ws["A3"] = "=SUM(A1:A2)"
ws["A4"] = "=IF(A3>10,A3*2,0)"
wb.save("/tmp/p.xlsx")

before = [load_workbook("/tmp/p.xlsx", data_only=True).active[c].value for c in ("A3", "A4")]
if before != [None, None]:
    sys.exit(f"PRE-STATE WRONG: expected [None, None] from openpyxl, got {before!r}. "
             "The test cannot prove LibreOffice did the work if the values were "
             "already there.")

r = subprocess.run(
    ["libreoffice", "--headless", "--calc",
     "--convert-to", "xlsx:Calc MS Excel 2007 XML", "--outdir", "/tmp/out", "/tmp/p.xlsx"],
    capture_output=True, text=True, timeout=180)
if r.returncode != 0:
    sys.exit(f"convert-to failed rc={r.returncode}\nstdout={r.stdout!r}\nstderr={r.stderr!r}")

after = [load_workbook("/tmp/out/p.xlsx", data_only=True).active[c].value for c in ("A3", "A4")]
if after != [42, 84]:
    sys.exit(f"RECALC WRONG: expected [42, 84], got {after!r}")
print("  ok    None,None -> 42,84 via libreoffice --convert-to")
PY' || fail "the recalculation round-trip did not produce computed values"

# Concurrency: one container may run more than one recalc. LibreOffice locks its
# user profile, so two simultaneous default-profile invocations can collide — a
# failure that would show up as an intermittent "verification is broken" the
# solver cannot diagnose. Prove the isolated-profile form works, since that is
# what the prompt will recommend.
dex 'mkdir -p /tmp/c && cd /tmp/c && cp /tmp/p.xlsx a.xlsx && cp /tmp/p.xlsx b.xlsx && \
  ( libreoffice -env:UserInstallation=file:///tmp/lo_a --headless --calc --convert-to xlsx --outdir /tmp/c/oa a.xlsx >/dev/null 2>&1 & \
    libreoffice -env:UserInstallation=file:///tmp/lo_b --headless --calc --convert-to xlsx --outdir /tmp/c/ob b.xlsx >/dev/null 2>&1 & \
    wait ) && \
  python3 -c "
from openpyxl import load_workbook
v=[load_workbook(p, data_only=True).active[\"A3\"].value for p in (\"/tmp/c/oa/a.xlsx\",\"/tmp/c/ob/b.xlsx\")]
assert v==[42,42], v
print(\"  ok    two concurrent recalcs with separate -env:UserInstallation both -> 42\")"' \
  || fail "concurrent recalculation with separate user profiles failed — the
  isolated-profile form the solver prompt recommends does not work here."

echo
echo "== acceptance: layout invariants =="

WD=$(docker image inspect "$IMAGE" --format '{{.Config.WorkingDir}}')
[ "$WD" = "/workspace" ] || fail "WORKDIR is $WD, expected /workspace"
ok "WORKDIR = /workspace"

dex 'test ! -e /app' || fail "/app exists. It must not: a solver that guesses
  \`cd /app\` has to FAIL there, or it writes its output into the container
  instead of the host bind-mount and scores 0 while its own verification passes
  (cost Financial_Model/02_05 and Template/02_04 on 2026-07-01)."
ok "/app absent — a stray cd /app still fails fast"

# The baseline's own libraries must not have gained a formula evaluator by
# accident (a transitive dep of python3-uno, say). If one appeared, the solver
# could verify against an engine that does not grade this benchmark.
LEAKED=$(dex 'python3 -c "
import importlib.util as u
print(\",\".join(m for m in (\"formulas\",\"pycel\",\"xlcalculator\",\"schedula\") if u.find_spec(m)))"')
[ -z "$LEAKED" ] || fail "a pure-Python formula engine leaked in: $LEAKED.
  Only LibreOffice grades this benchmark; a second evaluator with different
  semantics invites the solver to verify against the wrong oracle."
ok "no pure-Python formula engine present (LibreOffice is the only evaluator)"

echo
echo "PASS — $IMAGE ($NEW_ID) meets every acceptance check."
echo
echo "Sizes:"
docker images --format '  {{.Repository}}:{{.Tag}}  {{.Size}}' ssb-solver-lo ssb-solver ssb-verifier 2>/dev/null || true
