#!/bin/bash
# SSB v2 leg launcher, in the shape the ORCHESTRATOR LOOP calls.
#
#   scripts/ssbv2_leg.sh <tag> <train|val> <skills_dir_name>
#
# WHY THIS EXISTS. `loop._leg_argv_env` invokes `bash <leg_script> <tag> <key> <promoted_name>`,
# waits on the pid the launcher prints, and then looks for `<run_dir>/<tag>.json`. The existing
# SSB v2 launcher (ssbv2_arm_launch_rollouts.sh) takes `<arm> <stage> [leg]` instead, and the
# runner writes to a HARDCODED `outputs/ssbv2_base30/<TAG>.json` (run_ssbv2_base30_dsv4f_n3.py
# line 190 — not env-configurable). This adapter bridges both without touching either, because
# the runner owns the org-key refusal, the provider pin, the seat pool and the cache identity,
# and a second copy of any of that is the #154 shape.
#
# THREE THINGS THE LOOP REQUIRES, and each is load-bearing:
#
#  1. IT MUST PRINT `pid=<P>`. The loop takes the pid from that line rather than from a pid FILE,
#     which may be stale from an earlier attempt at the same tag.
#
#  2. THAT PID'S COMMAND LINE MUST CONTAIN `run_ssbv2`. `loop._pid_is_runner_pat` re-derives
#     identity from `ps -o command=` on every poll, because a pid outlives its owner and gets
#     recycled. The backgrounded wrapper below is `bash -c '... run_ssbv2_base30_dsv4f_n3.py ...'`,
#     so its own command line carries the pattern. Wrapping it in anything that hides that string
#     (a `sh -c` that reads a here-doc, say) would make the loop declare a healthy leg dead.
#
#  3. THE RESULT MUST LAND AT `<run_dir>/<tag>.json`. The wrapper copies it there AFTER the
#     runner exits, so the loop's `_await_leg` — which waits for the process THEN judges the
#     artifact — never sees a half-written file. The copy is the last thing the wrapper does.
#
# ENV VAR NAMES. The loop emits `EVOSKILL_TAU3_*` names for run dir, split and cache tag; they
# were named before a second benchmark existed. Translating here rather than renaming them in
# loop.py is deliberate: renaming would touch a live tau3 run mid-flight for a cosmetic gain.
set -euo pipefail
R="${EVOSKILL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"; cd "$R"
TAG=${1:?usage: ssbv2_leg.sh <tag> <train|val> <skills_dir_name>}
KEY=${2:?usage: ssbv2_leg.sh <tag> <train|val> <skills_dir_name>}
SK=${3:-}

RUNDIR_REL=${EVOSKILL_TAU3_RUN_DIR:?loop must set EVOSKILL_TAU3_RUN_DIR}
RUNDIR="$R/$RUNDIR_REL"
mkdir -p "$RUNDIR"
FINAL="$RUNDIR/${TAG}.json"
RUNNER_OUT="$R/outputs/ssbv2_base30/${TAG}.json"

# A COMPLETED LEG IS NOT A STALE ATTEMPT. Refuse rather than re-run: the task dirs hold the only
# copy of that leg's produced workbooks (the cache stores transcripts, not .xlsx bytes), so a
# re-run that cleared them would destroy the next epoch's evidence surface.
if [ -f "$FINAL" ]; then
  echo "REFUSING: $FINAL already exists — that leg is measured." >&2
  echo "  Move it aside deliberately if you mean to redo it." >&2
  exit 4
fi

# ── OPENROUTER KEY. Taken from the environment; EVOSKILL_ENV_FILE (default <repo>/.env) is
#    sourced as a fallback. Refuse to run without a key rather than guess.
ENVF="${EVOSKILL_ENV_FILE:-$R/.env}"
# An EXPORTED value wins over the file -- see scripts/lib/load_env.sh for why this is not
# a plain `. "$ENVF"` any more (the shipped template's empty OPENROUTER_API_KEY= used to
# blank an exported key, and the launcher then refused).
. "$R/scripts/lib/load_env.sh"
evoskill_load_env "$ENVF"
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"
[ -n "${OPENROUTER_API_KEY:-}" ] || { echo "OPENROUTER_API_KEY not set (env or $ENVF)" >&2; exit 4; }

# ── WHAT IS BEING MEASURED. Every value pinned, not defaulted, except where the loop supplies it.
SPLIT=${EVOSKILL_TAU3_SPLIT:?loop must set EVOSKILL_TAU3_SPLIT}
NS=${EVOSKILL_N_SAMPLES:-3}
CONC=${EVO_CONC:-30}
SEATS=${EVO_SEATS:-64}
case "$KEY" in train) PRIO=${EVO_PRIO:-10};; *) PRIO=${EVO_PRIO:-5};; esac

# THE 0731 SNAPSHOT, NOT THE ROLLING TAG. `deepseek/deepseek-v4-flash` exists in this repo's
# history (outputs/ssbv2_base30/CORRUPTED_workspace_collision/base30_0730_CORRUPT.json) and the
# two are different models. The RunCache keys on the model STRING, so the wrong one here does not
# corrupt an existing bucket — it silently opens a new one and re-buys every episode under a
# model nobody chose.
# REQUIRED, not defaulted. `evoskill-loop` exports spec.solver_model for every leg it
# launches (#320), but a HAND-RUN base leg -- which is how the README says to build the
# comparator -- silently fell back to this hardcoded snapshot. So whenever the frozen spec
# names any other solver, the base legs were measured on a different model from the evolved
# legs, the driver compared them anyway, and nothing in the tree noticed. A defaulted model
# is the same class of error as a defaulted cache tag, which this launcher already refuses.
MODEL=${EVOSKILL_SOLVER_MODEL:?set EVOSKILL_SOLVER_MODEL to the solver model from the freeze receipt — a base leg measured on a different model than the evolved legs is not a comparator}
# EXPLICIT ALLOW-LIST of dated snapshots, so no rolling tag can ever slip through. Extend it
# only for a snapshot an operator named explicitly (the maintainer, 0815: deepseek-v4-pro-0813).
case "$MODEL" in
  *-0731|*-0813) : ;;
  *) echo "REFUSING: solver model '$MODEL' is not an allow-listed dated snapshot (*-0731, *-0813)" >&2; exit 4;;
esac

# ...AND IT MUST BE THE MODEL THE SPEC FROZE. Requiring the variable only proves the operator
# typed SOMETHING; the property this guard is for -- "a base leg measured on a different model
# than the evolved legs is not a comparator" -- is about agreement with the frozen spec, and
# nothing compared the two. An operator who exports last week's snapshot gets a full leg,
# correctly cached under that model, silently incomparable with every leg the loop launches
# (which exports spec.solver_model itself, #320). Checked only when the spec is reachable: a
# hand-run against a bare directory is still allowed, it simply cannot be cross-checked.
if [ -f "$RUNDIR/SPEC.json" ]; then
  _SPEC_MODEL=$(python3 - "$RUNDIR/SPEC.json" <<'PYEOF'
import json, sys
try:
    print(json.load(open(sys.argv[1]))["spec"].get("solver_model", "") or "")
except Exception:
    print("")
PYEOF
  )
  if [ -n "$_SPEC_MODEL" ] && [ "$_SPEC_MODEL" != "$MODEL" ]; then
    echo "REFUSING: EVOSKILL_SOLVER_MODEL='$MODEL' but $RUNDIR/SPEC.json froze" >&2
    echo "  solver_model='$_SPEC_MODEL'. These legs would be cached under a model the" >&2
    echo "  experiment did not declare, and comparing them to the loop's legs (which use" >&2
    echo "  the frozen value) would compare two different solvers." >&2
    echo "  Export the frozen model, or freeze a spec that names the one you want." >&2
    exit 4
  fi
  [ -n "$_SPEC_MODEL" ] && echo "  solver model: $MODEL  [matches SPEC.json]"
fi

# BASE vs SKILL LEG. A base leg must NOT set EVOSKILL_SKILLS_DIR: the runner then points at its
# own per-tag empty dir and asserts it empty, and the tree_hash keys on skills CONTENT, so every
# base leg lands in the same bucket regardless of tag. Passing an empty string instead would make
# the runner treat "" as a path.
SKILLENV=()
if [ -n "$SK" ] && [ -d "$RUNDIR/$SK" ] && [ -n "$(ls -A "$RUNDIR/$SK" 2>/dev/null)" ]; then
  SKILLENV=(EVOSKILL_SKILLS_DIR="$RUNDIR/$SK")
  echo "  skill leg: $RUNDIR/$SK"
else
  SKILLENV=(EVOSKILL_BASE_LEG=1)
  echo "  BASE leg (no skills dir)"
fi

# THE CACHE DIR IS DECLARED, NOT DEFAULTED. The runner defaults to a per-tag dir
# (.cache/ssbv2_<TAG>_cache), which is EMPTY for a new tag — so a base leg that could have
# replayed 100% of its episodes from an existing bucket instead re-buys every one of them.
# Caught by the maintainer, 2026-08-12, after I had already launched two from-scratch legs:
# .cache/ssbv2_lo_base2_cache/b4aca32fe1e2 holds 600 base records on dsv4f-0731 covering all
# 120 tasks of this split. The cache stores the solver TRANSCRIPT, not the score, so a replay
# is scored by TODAY's scorer — which is exactly the property that made a fresh base leg look
# necessary in the first place. Both wants are satisfiable at once.
: "${EVOSKILL_CACHE_DIR:?declare EVOSKILL_CACHE_DIR — an undeclared cache dir silently re-buys}"
export EVOSKILL_CACHE_DIR
LOG="$RUNDIR/${TAG}.log"
mkdir -p "$R/outputs/ssbv2_base30"

# THE REPO VENV'S PYTHON, NOT PATH LUCK — the same fix tau3_leg_b2525.sh got, applied here
# for harness parity. Bare `python3` resolves to whatever the shell finds first; on the
# machine this was developed on that was an interpreter carrying every dep from months of
# use, so the leg "worked" while never exercising the shipped venv. On a fresh machine it
# is a system python without the repo's deps and the leg dies at the first import — inside
# a nohup'd subshell, so the only symptom is the DEAD check below. EVOSKILL_PYTHON
# overrides for exotic setups.
PYBIN="${EVOSKILL_PYTHON:-$R/.venv/bin/python}"
[ -x "$PYBIN" ] || PYBIN=python3

# The wrapper: run, then relocate. `exec`-less so the copy happens after the runner exits.
# Its command line contains `run_ssbv2_base30_dsv4f_n3.py`, which is what the loop's liveness
# probe matches on — see note 2 in the header.
nohup env \
  OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  "${SKILLENV[@]}" \
  EVOSKILL_SPLIT_FILE="$SPLIT" \
  EVOSKILL_SPLIT_KEY="$KEY" \
  EVOSKILL_RUN_TAG="$TAG" \
  EVOSKILL_SOLVER_MODEL="$MODEL" \
  EVOSKILL_N_SAMPLES="$NS" \
  EVOSKILL_TURN_BUDGET=45 \
  # A FAILED RUN MUST NOT INHERIT A PREVIOUS ATTEMPT'S NUMBERS.
  # The copy was unconditional on the runner's exit code, and the launcher then exited 0 via
  # the "finished within 5s" branch below. So a runner that died on its first line left the
  # PREVIOUS attempt's result in place, it was copied in as THIS leg's result, and the
  # epoch's gate was computed on numbers this run never measured. The orchestrator cannot
  # catch it: _await_leg polls for the file's existence and never inspects an exit code.
  # Stamp the start, drop any stale runner output for this TAG, and require rc==0 AND a
  # file newer than the stamp before copying.
  STAMP="$(dirname "$FINAL")/.${TAG}.start"
  : > "$STAMP"
  rm -f "$RUNNER_OUT"

  EVOSKILL_SAMPLE_SEED=730 \
  EVOSKILL_DOCKER_IMAGE=ssb-solver-lo:latest \
  EVOSKILL_CONCURRENCY="$CONC" \
  EVOSKILL_SEAT_TOTAL="$SEATS" \
  EVOSKILL_SEAT_MEMBER="$TAG" \
  EVOSKILL_SEAT_PRIORITY="$PRIO" \
  EVOSKILL_SOLVER_TIMEOUT_SECONDS="${EVOSKILL_SOLVER_TIMEOUT_SECONDS:-10800}" \
  PHOENIX_PROJECT_NAME="${PHOENIX_PROJECT_NAME:-ssbv2-${TAG}}" \
  bash -c "'$PYBIN' scripts/run_ssbv2_base30_dsv4f_n3.py; rc=\$?; \
           if [ \$rc -eq 0 ] && [ -f '$RUNNER_OUT' ] && [ '$RUNNER_OUT' -nt '$STAMP' ]; then cp '$RUNNER_OUT' '$FINAL'; fi; exit \$rc" \
  > "$LOG" 2>&1 &
PID=$!
command -v caffeinate >/dev/null 2>&1 && caffeinate -i -w "$PID" >/dev/null 2>&1 &
echo "launched $TAG pid=$PID key=$KEY conc=$CONC seats=$SEATS prio=$PRIO model=$MODEL"
echo "  log:    $LOG"
echo "  result: $FINAL  (copied from $RUNNER_OUT on exit)"

# A DEAD CHILD MUST BE LOUD HERE, NOT AT THE LOOP'S FIRST POLL. `$!` succeeding says nothing
# about whether the child survived its first line — that is how bash 3.2's empty-array-under-
# set-u bug once produced four pid files and zero rollouts while reporting success.
sleep 5
if kill -0 "$PID" 2>/dev/null; then
  echo "  verified: pid $PID alive at +5s"
  elif [ -f "$FINAL" ] && [ "$FINAL" -nt "$STAMP" ]; then
    # ...and the result must postdate THIS launch; without the freshness test
    # this branch reported success for a leftover file from an earlier attempt.
    echo "  note: leg finished within 5s — fully cache-replayed, fresh result written"
else
  echo "DEAD: pid $PID exited within 5s and wrote no result. Last log lines:" >&2
  tail -20 "$LOG" >&2
  exit 5
fi
