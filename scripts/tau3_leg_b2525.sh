#!/bin/bash
# Banking 25/25/47 leg launcher.  tau3_leg_b2525.sh <tag> <train|val> <BASE|skills_vN>
#
# WHY ITS OWN LAUNCHER. tau3_run_leg.sh hardcodes the OLD split file and cache tag with BARE
# exports, which silently clobber a caller's values. Everything tunable here is
# ${VAR:-default} so a smoke test can override without editing the file, and every value that
# defines the INSTRUMENT is pinned, not defaulted, because a drifted instrument is the defect
# class that has cost this project the most (#182 wall, #183 user-sim, #185 agent budget).
set -euo pipefail
R="${EVOSKILL_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"; T=$R/scripts; cd "$R"
# T was a per-job scratch dir, which is deleted when
# that job is. A multi-hour 4-epoch run outliving its launcher would then fail at the ONE
# line that uses it (the detach helper, line ~139) — mid-leg, after spending. Both this
# script and detach.py now live in scripts/ and are version-controlled.
TAG=$1; KEY=$2; SK=$3

# ── THE KEY. Read from an env FILE, never guessed from the ambient environment: the first
# version guarded on a path that did not exist, so the block was a silent no-op and the key
# came from whatever happened to be exported. REFUSE rather than fall back — the failure mode
# is spending on the wrong account.
# EVOSKILL_ENV_FILE overrides the location. OPENROUTER_ORG_KEY, when set, wins over
# OPENROUTER_API_KEY: a machine that holds several keys can point experiment spend at one of
# them without editing anything.
ENVF="${EVOSKILL_ENV_FILE:-$R/.env}"
[ -f "$ENVF" ] || { echo "no env file at $ENVF — refusing to guess the key" >&2; exit 4; }
# An EXPORTED value wins over the file (scripts/lib/load_env.sh); the previous
# unconditional source let the template's empty OPENROUTER_API_KEY= blank an exported key
# and made EVOSKILL_REQUIRE_KEY_SUFFIX fence a key the run did not spend.
. "$R/scripts/lib/load_env.sh"
evoskill_load_env "$ENVF"
export OPENROUTER_API_KEY="${OPENROUTER_ORG_KEY:-${OPENROUTER_API_KEY:-}}"
[ -n "${OPENROUTER_API_KEY:-}" ] || { echo "OPENROUTER_API_KEY empty after sourcing $ENVF" >&2; exit 4; }
# OPTIONAL SPEND FENCE, off unless you ask for it. Set EVOSKILL_REQUIRE_KEY_SUFFIX to the last
# characters of the key you intend to spend on, and this leg refuses to launch under any other.
# Useful when one machine holds a shared/billed key and a personal one whose pricing differs;
# the runner echoes the served key's last 4 on its own [KEY] line so the two can be told apart
# after the fact. Unset = no fence, which is the right default for anyone but the author.
if [ -n "${EVOSKILL_REQUIRE_KEY_SUFFIX:-}" ]; then
  _n=${#EVOSKILL_REQUIRE_KEY_SUFFIX}
  if [ "${OPENROUTER_API_KEY: -$_n}" != "$EVOSKILL_REQUIRE_KEY_SUFFIX" ]; then
    echo "REFUSING: key ...${OPENROUTER_API_KEY: -4} does not end in the required" >&2
    echo "  EVOSKILL_REQUIRE_KEY_SUFFIX (...${EVOSKILL_REQUIRE_KEY_SUFFIX})" >&2
    exit 4
  fi
fi

# ── WHAT IS BEING MEASURED ──────────────────────────────────────────────────────────
# THE THREE IDENTITY-BEARING VALUES REFUSE, THEY DO NOT DEFAULT. They used to fall back to
# one particular run (`outputs/tau3_banking_2525`, split_252547.json, cache tag `b2525`) —
# a directory and split file that exist on no clone. An operator who set two of the three
# and forgot the third got no error: the leg ran, and its episodes went into THAT run's cache
# bucket, since the tag is the only thing separating one configuration's records from
# another's. Silently measuring into someone else's namespace is worse than not running.
# The SSB v2 twin already refuses this way (ssbv2_leg.sh: `${EVOSKILL_TAU3_RUN_DIR:?...}`),
# and the loop supplies all three on every leg it launches.
export EVOSKILL_TAU3_SPLIT="${EVOSKILL_TAU3_SPLIT:?set it to this runs split file — the exact split_NNN.json named by the freeze receipt}"
export EVOSKILL_TAU3_DOMAIN=banking_knowledge
export EVOSKILL_TAU3_VARIANT=bm25_grep
export EVOSKILL_TAU3_RUN_DIR="${EVOSKILL_TAU3_RUN_DIR:?set it to this runs directory, e.g. outputs/my_run}"  # #190
export EVOSKILL_TAU3_CACHE_TAG="${EVOSKILL_TAU3_CACHE_TAG:?set it to this runs cache tag from the freeze receipt — the tag is what keeps these episodes out of another runs cache bucket}"
export TAU2_BENCH_ROOT=$R/third_party/tau3-bench
export EVOSKILL_SPLIT_KEY=$KEY
export EVOSKILL_N_SAMPLES=${EVOSKILL_N_SAMPLES:-3}

# ── FROZEN INSTRUMENT — identical to the 2026-08-06 banking run ─────────────────────
# The DECIDING limits are the agent budget (clean seconds + agent turns); the 5400s episode
# wall is a BACKSTOP and must never be what ends an episode. All three are now in the cache
# IDENTITY (#185), so a drift refuses instead of silently replaying foreign records.
export EVOSKILL_TAU2_AGENT_CLEAN_S=${EVOSKILL_TAU2_AGENT_CLEAN_S:-1200}
export EVOSKILL_TAU2_AGENT_TURNS=${EVOSKILL_TAU2_AGENT_TURNS:-100}
export EVOSKILL_TAU2_EPISODE_TIMEOUT=${EVOSKILL_TAU2_EPISODE_TIMEOUT:-5400}
export EVOSKILL_TAU2_INACTIVITY_S=${EVOSKILL_TAU2_INACTIVITY_S:-300}
# #192, 2026-08-07. PER-LLM-CALL BUDGET, pinned instead of derived from the episode wall.
# `_tau2_protocol.llm_call_budget` defaults to max(60, wall/3), so the 5400s backstop yields
# 1800s -- a single hung LLM call may block for THIRTY MINUTES. And the 1200s agent clean wall
# CANNOT preempt it: tau2 evaluates that wall only at TURN BOUNDARIES
# (orchestrator.py:750, reached only via _check_half_duplex_termination), and a call hung
# mid-turn never reaches one. Observed live on the 25/25 smoke: task_007 sat 1215s having
# emitted ZERO turn spans, blocked in PySSL_select->poll on one socket to OpenRouter.
# THIS ALSO EXPLAINS #147: its unexplained "900.04s" deaths are the same formula under the
# then-current 2700s wall (2700/3 = 900). Not a new bug -- designed behaviour whose cost I
# raised by moving the backstop to 5400s.
#
# 600s is EVIDENCE-BASED, not a guess. `llm_call_budget`'s docstring deliberately withdrew an
# earlier tightening because "NOTHING HERE MEASURES REAL CALL LATENCY". It does now: the
# completed smoke episode's Phoenix agent-turn spans are genuine (durations vary 0.1-14.5s and
# the root span's 153.9s matches the cache record exactly; only user.turn 0.1s values are the
# synthetic replayed ones) -- median 4.0s, max 14.5s over 15 turns. 600s is ~40x the worst
# observed turn, so no legitimate call is truncated, while a hang costs 600s instead of 1800s.
# Worst-case turn becomes 3x600 = 1800s, still well under the 5400s backstop.
# Set BEFORE any leg of this run, so every leg shares it; it is now in the cache IDENTITY too.
export EVOSKILL_TAU2_LLM_TIMEOUT_S=${EVOSKILL_TAU2_LLM_TIMEOUT_S:-600}
export EVOSKILL_TAU2_LLM_RETRIES=${EVOSKILL_TAU2_LLM_RETRIES:-2}
export EVOSKILL_TAU2_USER_MODEL_REFERENCE=openrouter/openai/gpt-5-nano
export EVOSKILL_TAU2_USER_MODEL_CHEAP=openrouter/openai/gpt-5-nano
export EVOSKILL_CACHE_SKIP_WALL_LIMITED=1     # #182 must be ON, not merely assumed
export EVOSKILL_SOLVER_ALLOW_FALLBACKS=0      # half of the DeepSeek hard pin
export EVOSKILL_RUN_TAG=$TAG

# ── THROUGHPUT: every trick established on the 08-06 run ────────────────────────────
# #184 pooled dispatch is unconditional in the runner: all N samples of all tasks go into ONE
# pool, so a pass costs its SLOWEST episode rather than a sum (~3x on a 3-sample leg).
# Concurrency is sized to the POOLED batch (25 x 3 = 75), not to the task count -- sizing to
# 25 would leave two thirds of the batch queued, which is what the first v1 train leg did.
# #195, CORRECTED 2026-08-07. These three defaulted to 75/88/88 and that is what caused the
# memory exhaustion: 75+75 concurrent legs meant 298 workers at ~258 MB each and the machine
# reached 0.10 GB free. ALL TEN legs of this run actually executed at 30 -- the 75 came from a
# stale doc I followed, not from anything that had ever worked. The defaults now match what the
# run really used, so a future launch that passes no override gets the instrument that has
# ten clean legs behind it rather than the one that had to be killed.
export EVOSKILL_CONCURRENCY=${EVO_CONC:-30}
# The AIMD ceiling is EVOSKILL_CONCURRENCY_MAX. EVOSKILL_ADMISSION_MAX, which an earlier
# launcher exported, is read NOWHERE in the codebase. Probe mode governs at PROBE_MAX.
# PINNED TO 30, not 88: an AIMD ceiling above the target lets the controller climb straight
# back into #195 on its own, which is the whole reason the ceiling exists.
export EVOSKILL_CONCURRENCY_MAX=${EVO_CONC_MAX:-30}
export EVOSKILL_ADMISSION_PROBE=1
export EVOSKILL_ADMISSION_PROBE_MAX=${EVO_PROBE_MAX:-30}
# ONE machine-wide seat pool (seats.py:137, fcntl.flock) so train and val can run
# CONCURRENTLY drawing from the same 64 rather than taking 64 each. ~52 MB RSS per worker,
# 2 workers per episode.
export EVOSKILL_SEAT_TOTAL=${EVO_SEATS:-64}
case "$KEY" in train) export EVOSKILL_SEAT_PRIORITY=${EVO_PRIO:-10};;
                *)    export EVOSKILL_SEAT_PRIORITY=${EVO_PRIO:-5};; esac
export EVOSKILL_SEAT_MEMBER=$TAG

# ── SKILLS RESOLVE AGAINST THE RUN DIR, NOT THE HARDCODED 2525 PATH ────────────────
# 2026-08-08, found before the first arm-A skill leg. This read `$RUNDIR/$SK`, and RUNDIR is
# hardcoded to outputs/tau3_banking_2525. So a run redirected with EVOSKILL_TAU3_RUN_DIR
# still loaded ITS SKILLS from the 0807 Opus run's directory -- and the `-d` check PASSED,
# because outputs/tau3_banking_2525/skills_v1 genuinely exists. The 3-arm ablation would have
# rolled out Opus's skills under arm A's cache tag and reported them as Sonnet's, with no
# error anywhere. Silent, and fatal to the only claim the experiment makes.
# RUN_ABS also fixes LOGDIR's latent twin: it prefixed $R unconditionally, so an ABSOLUTE
# EVOSKILL_TAU3_RUN_DIR (which is what tau3_evolver.sh is given) yielded $R/$R/... .
case "${EVOSKILL_TAU3_RUN_DIR}" in
  /*) RUN_ABS="${EVOSKILL_TAU3_RUN_DIR}" ;;
  *)  RUN_ABS="$R/${EVOSKILL_TAU3_RUN_DIR}" ;;
esac
[ -d "$RUN_ABS" ] || { echo "no run dir: $RUN_ABS" >&2; exit 3; }
if [ "$SK" = "BASE" ]; then unset EVOSKILL_SKILLS_DIR || true
else export EVOSKILL_SKILLS_DIR=$RUN_ABS/$SK
     [ -d "$EVOSKILL_SKILLS_DIR" ] || { echo "no skills dir: $EVOSKILL_SKILLS_DIR" >&2; exit 3; }; fi

# #236: the LAST bare export in this file, and the header above says why that is wrong —
# "bare exports silently clobber a caller's values". It clobbered the orchestrator's: the loop
# names the evolver's project `<spec>-e<N>` but had no way to name the SOLVER legs, so solver
# and evolver spans for one epoch landed in unrelated projects and the solver half was findable
# only by guessing a minute-resolution timestamp. Now ${VAR:-default}, like every other tunable
# here, so a caller can name it and a hand launch still gets the old name.
export PHOENIX_PROJECT_NAME="${PHOENIX_PROJECT_NAME:-tau3b2525-$TAG-$(date +%m%d-%H%M)}"
# The log follows the RUN DIR, not the hardcoded $RUNDIR. A smoke that redirects its results
# with EVOSKILL_TAU3_RUN_DIR was still dropping its log into outputs/tau3_banking_2525/, i.e.
# into the published run's directory, where a later reader would find smoke logs sitting beside
# the real legs' and have no way to tell from the filename which was which.
LOGDIR=$RUN_ABS
mkdir -p "$LOGDIR"
LOG=$LOGDIR/${TAG}.log
# The pid file follows the run dir too. It used to land in the hardcoded $RUNDIR, so a
# redirected arm wrote its pids into the published 0807 run's directory alongside that run's.
# THE REPO VENV'S PYTHON, NOT PATH LUCK. Bare `python3` resolves to whatever interpreter
# the shell finds first — on the dev machine that was an anaconda install carrying every
# dep from months of use, so the leg "worked" while never exercising the shipped venv; on
# a fresh machine it is a system python without the repo's deps and the leg dies at
# import. EVOSKILL_PYTHON overrides for exotic setups.
PYBIN="${EVOSKILL_PYTHON:-$R/.venv/bin/python}"
[ -x "$PYBIN" ] || PYBIN=python3
P=$("$PYBIN" "$T/detach.py" "$LOG" "$RUN_ABS/${TAG}.pid" "$PYBIN" scripts/run_tau3_banking.py)
# NON-NEGOTIABLE: keep the box awake for the whole leg, and never pipe this launch --
# caffeinate inherits stdout, so a pipe holds the foreground and the 2-min tool timeout
# kills the child.
command -v caffeinate >/dev/null 2>&1 && caffeinate -i -w "$P" >/dev/null 2>&1 &
echo "launched $TAG pid=$P key=$KEY skills=$SK"
echo "  log=$LOG  phoenix=$PHOENIX_PROJECT_NAME"
echo "  conc=$EVOSKILL_CONCURRENCY seats=$EVOSKILL_SEAT_TOTAL prio=$EVOSKILL_SEAT_PRIORITY N=$EVOSKILL_N_SAMPLES"
echo "  instrument: clean=${EVOSKILL_TAU2_AGENT_CLEAN_S}s turns=${EVOSKILL_TAU2_AGENT_TURNS} backstop=${EVOSKILL_TAU2_EPISODE_TIMEOUT}s"
