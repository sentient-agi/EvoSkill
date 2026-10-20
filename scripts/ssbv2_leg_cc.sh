#!/bin/bash
# The leg launcher the ORCHESTRATOR LOOP runs when the SOLVER IS CLAUDE CODE (`claude -p` in containers on the run's
# ONE Daytona machine; scripts/daytona/cc_leg_driver.py).
#
#   scripts/ssbv2_leg_cc.sh <tag> <train|val|test> <skills_dir_name>
#
# A run selects it with RUNTIME.json `"leg_script"`; the loop's own code is the same for every launcher
# (loop.py: Ctx.leg_script is runtime plumbing, not identity). It runs SSB and tau3 legs alike.
#
# THE THREE THINGS THE LOOP REQUIRES (loop._leg_argv_env / _dispatch_leg / _await_leg):
#  1. It prints `launched <tag> pid=<P>` and DETACHES; the loop waits on that pid by identity.
#  2. That pid is ONE python process (scripts/daytona/cc_leg_driver.py) whose command line names the tag
#     (`--tag <tag>`): the loop's dispatch record keeps the pid with its start time and re-derives both. The
#     driver also holds <run_dir>/.machine/legs/<tag>.pid (an flock) while it runs: the per-run record of the leg.
#  3. `<run_dir>/<tag>.json` appears only when EVERY unit completed AND was graded AND every driver check
#     passed, written atomically, and the process exits right after. A failed leg writes
#     `<tag>.FAILED.json` instead and exits 5 (the loop then HALTs: "exited without writing").
#
# ENV (set by the loop):
#   EVOSKILL_TAU3_RUN_DIR   run dir, repo-relative (absolute also accepted)
#   EVOSKILL_TAU3_SPLIT     the frozen split file
#   EVOSKILL_CACHE_DIR      the PER-EPOCH cache dir (required; train and val of one epoch share it)
#   EVOSKILL_N_SAMPLES, EVO_CONC (in-flight rollouts on the machine, <= 30)
#   EVOSKILL_SKILLS_DIR     set by the loop for every skill leg; must resolve to $RUNDIR/<arg 3>
#   EVOSKILL_SOLVER_MODEL   = spec.solver_model; must be a Claude model (claude-*)
#   EVOSKILL_SOLVER_PROVIDER_* / _EXPECT_*  must NOT be set: those are OpenRouter routing pins, which a Claude Code
#                           solver does not read; a RUNTIME that declares one is refused (infra.agree)
#   EVOSKILL_LEG_DRY_RUN=1  validate + plan + write <run_dir>/<tag>.DRYRUN.json, launch NOTHING, exit 7
#   EVOSKILL_PYTHON         the interpreter (the venv with the `daytona` extra); default <repo>/.venv/bin/python,
#                           else python3 on PATH
#
# BASE vs SKILL LEG: arg 3 empty AND EVOSKILL_SKILLS_DIR unset = base leg (no skills delivered). Otherwise it is a
# skill leg on that dir; an EMPTY promoted dir is a skill leg with no skills; a missing dir is refused. A test leg
# needs the test tasks' files on the pool machine (RUNTIME daytona_leg.pool_keys lists "test"; cc_pool_host).
#
# NO CREDENTIAL IS READ HERE: the driver reads the run's one credential itself, in-process (cc_solver.read_credential:
# INFRA claude_code.token_file, else EVOSKILL_CC_TOKEN_FILE -- a regular file, 0600 -- else ANTHROPIC_API_KEY; never
# printed, handed to Claude Code on fd 3 only). Nothing secret appears on a command line.
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd -P); cd "$R"
TAG=${1:?usage: ssbv2_leg_cc.sh <tag> <train|val> <skills_dir_name>}
KEY=${2:?usage: ssbv2_leg_cc.sh <tag> <train|val> <skills_dir_name>}
SK=${3:-}
PY=${EVOSKILL_PYTHON:-}
if [ -z "$PY" ]; then
  if [ -x "$R/.venv/bin/python" ]; then PY="$R/.venv/bin/python"; else PY=$(command -v python3 || true); fi
fi
DRIVER="$R/scripts/daytona/cc_leg_driver.py"

RUNDIR_REL=${EVOSKILL_TAU3_RUN_DIR:?loop must set EVOSKILL_TAU3_RUN_DIR}
case "$RUNDIR_REL" in /*) RUNDIR="$RUNDIR_REL" ;; *) RUNDIR="$R/$RUNDIR_REL" ;; esac
[ -d "$RUNDIR" ] || { echo "REFUSING: no run dir $RUNDIR" >&2; exit 4; }
# Operator hold: a HOLD_LEGS file in the run dir stops the loop before any leg launches (the loop then HALTs at
# the leg step). Remove the file to launch legs again.
if [ -f "$RUNDIR/HOLD_LEGS" ]; then
  echo "REFUSING: $RUNDIR/HOLD_LEGS is present — legs are on hold: $(head -c 300 "$RUNDIR/HOLD_LEGS")" >&2
  exit 4
fi
FINAL="$RUNDIR/${TAG}.json"
if [ -f "$FINAL" ]; then
  echo "REFUSING: $FINAL already exists — that leg is measured." >&2
  exit 4
fi
case "$KEY" in
  train|val|test) : ;;
  *) echo "REFUSING: key '$KEY' (train|val|test only)" >&2; exit 4 ;;
esac
SPLIT=${EVOSKILL_TAU3_SPLIT:?loop must set EVOSKILL_TAU3_SPLIT}
NS=${EVOSKILL_N_SAMPLES:-3}
CONC=${EVO_CONC:-30}
: "${EVOSKILL_CACHE_DIR:?declare EVOSKILL_CACHE_DIR — an undeclared cache dir silently re-buys}"
MODEL=${EVOSKILL_SOLVER_MODEL:?loop must set EVOSKILL_SOLVER_MODEL}
case "$MODEL" in
  claude-*) : ;;
  *) echo "REFUSING: solver model '$MODEL' is not a Claude model — this launcher runs a Claude Code solver (claude-*)" >&2; exit 4 ;;
esac
# The seven OpenRouter routing pins (infra.SOLVER_ROUTING_KEYS / loop.SOLVER_ENV_KEYS): one set in the loop's
# environment means the wrong RUNTIME solver_env (or shell) is in force; refuse rather than ignore.
for _v in EVOSKILL_SOLVER_PROVIDER_ORDER EVOSKILL_SOLVER_PROVIDER_ONLY EVOSKILL_SOLVER_ALLOW_FALLBACKS \
          EVOSKILL_SOLVER_REQUIRE_PARAMETERS EVOSKILL_SOLVER_PROVIDER_QUANTIZATIONS \
          EVOSKILL_SOLVER_EXPECT_PROVIDER EVOSKILL_SOLVER_EXPECT_MODEL; do
  if [ -n "${!_v:-}" ]; then
    echo "REFUSING: $_v is set (an OpenRouter routing pin): this launcher runs a Claude Code solver; unset it" >&2
    exit 4
  fi
done
case "$CONC" in ''|*[!0-9]*) echo "REFUSING: EVO_CONC='$CONC' is not an integer" >&2; exit 4 ;; esac
if [ "$CONC" -gt 30 ]; then
  echo "REFUSING: EVO_CONC=$CONC exceeds 30 — the Claude Code leg runs at most 30 rollouts in flight (1 GiB each)" >&2
  exit 4
fi

# BASE vs SKILL leg.
SKILLS_ARG=""
if [ -z "$SK" ] && [ -z "${EVOSKILL_SKILLS_DIR:-}" ]; then
  echo "  BASE leg (no skills dir)"
else
  [ -n "$SK" ] || { echo "REFUSING: EVOSKILL_SKILLS_DIR is set but arg 3 (the promoted dir name) is empty" >&2; exit 4; }
  SKDIR="$RUNDIR/$SK"
  if [ -n "${EVOSKILL_SKILLS_DIR:-}" ]; then
    _a=$(cd "$SKDIR" 2>/dev/null && pwd -P || true)
    case "$EVOSKILL_SKILLS_DIR" in /*) _e="$EVOSKILL_SKILLS_DIR" ;; *) _e="$R/$EVOSKILL_SKILLS_DIR" ;; esac
    _b=$(cd "$_e" 2>/dev/null && pwd -P || true)
    if [ -z "$_a" ] || [ "$_a" != "$_b" ]; then
      echo "REFUSING: EVOSKILL_SKILLS_DIR ($EVOSKILL_SKILLS_DIR) and \$RUNDIR/$SK ($SKDIR) are not the same dir" >&2
      exit 4
    fi
  fi
  if [ ! -d "$SKDIR" ]; then
    echo "REFUSING: skill leg $TAG but $SKDIR does not exist" >&2
    exit 4
  fi
  SKILLS_ARG="$SKDIR"
  echo "  skill leg: $SKDIR"
fi

[ -n "$PY" ] && [ -x "$PY" ] || { echo "REFUSING: no interpreter at '$PY' (set EVOSKILL_PYTHON to the venv's python)" >&2; exit 4; }
# (a missing driver is refused by the interpreter itself at the --check-only line below: its error and exit
#  code reach the loop as this launcher's, before anything is detached)
ARGS=(leg --tag "$TAG" --key "$KEY" --run-dir "$RUNDIR" --split "$SPLIT"
      --cache-dir "$EVOSKILL_CACHE_DIR" --skills-dir "$SKILLS_ARG" --n-samples "$NS"
      --concurrency "$CONC")

# DRY RUN: the whole validation and the plan, nothing created, uploaded, spent or launched.
if [ "${EVOSKILL_LEG_DRY_RUN:-0}" = "1" ]; then
  set +e
  "$PY" "$DRIVER" "${ARGS[@]}" --dry-run
  rc=$?
  set -e
  # A refusal in the driver is a refusal of the dry run too (never reported as a clean dry run).
  [ "$rc" = 7 ] || { echo "DRY RUN REFUSED by the driver (rc=$rc) — nothing launched" >&2; exit "$rc"; }
  echo "DRY RUN — no process launched, no container created (record $RUNDIR/${TAG}.DRYRUN.json)"
  exit 7
fi

# Validate BEFORE detaching, so a refusal reaches the loop as a launcher exit code with its reason
# (the loop HALTs with the tail), not as a wrapper that dies at +1s.
"$PY" "$DRIVER" "${ARGS[@]}" --check-only

LOG="$RUNDIR/${TAG}.log"
nohup "$PY" "$DRIVER" "${ARGS[@]}" > "$LOG" 2>&1 < /dev/null &
PID=$!
# best effort: keep the host awake while the leg runs (src/orchestrator/host_tools.keep_awake: caffeinate on macOS,
# systemd-inhibit on Linux when installed, else nothing); in the background, so it never delays the launch line
( "$PY" -c "from src.orchestrator.host_tools import keep_awake; keep_awake($PID)" </dev/null >/dev/null 2>&1 & )
echo "launched $TAG pid=$PID key=$KEY conc=$CONC model=$MODEL daytona=1 harness=claude_code"
echo "  log:    $LOG"
echo "  result: $FINAL  (written by the driver after every unit graded; FAILED -> ${TAG}.FAILED.json)"

sleep 5
if kill -0 "$PID" 2>/dev/null; then
  echo "  verified: pid $PID alive at +5s"
elif [ -f "$FINAL" ]; then
  echo "  note: leg finished within 5s"
else
  echo "DEAD: pid $PID exited within 5s and wrote no result. Last log lines:" >&2
  tail -20 "$LOG" >&2
  exit 5
fi
