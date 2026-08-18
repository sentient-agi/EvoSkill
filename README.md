# EvoSkill v2

EvoSkill v2 is a pair of agents that improves a third. The **evolver** reads
the failed rollouts of a **solver** agent and rewrites the skills the solver
uses — short markdown files loaded into the solver's context. The **agent
orchestrator (AO)** turns a one-page prose brief into a complete, frozen
experiment and drives the loop: rollout legs, evolver dispatch, automated
verification gates, and a sealed held-out test at the end.

This release contains the minimal code to run skill evolution on two
benchmarks: **τ³-bench banking** (multi-turn customer service, graded on
final database state) and **SpreadsheetBench v2** (spreadsheet editing,
graded cell-by-cell against a reference workbook).

## Requirements

- Linux or macOS. Python **3.12+**. [`uv`](https://docs.astral.sh/uv/).
- **Docker** — native on Linux; on macOS use [colima](https://github.com/abiosoft/colima)
  (`colima start --cpu 8 --memory 12` or larger; keep `/Users` shared, the default).
- **Claude Code CLI** (`npm install -g @anthropic-ai/claude-code`). The AO and
  the evolver run through the `claude` binary. Authenticate with either
  a Claude subscription (`claude login`) or an Anthropic API key
  (`ANTHROPIC_API_KEY`).
- An **OpenRouter API key** (`OPENROUTER_API_KEY`) for the solver model.
- `git`.

Copy `.env.example` to `.env` and fill in your keys.

## Setup

```bash
git clone <this-repo> EvoSkill && cd EvoSkill
uv sync                          # main venv
# Optional — Phoenix tracing (live evolver spans + AO session replay into a
# local Phoenix server at :6006). Everything degrades gracefully without it:
#   uv sync --extra phoenix

# 1. Vendor the benchmark repos (clone-don't-vendor policy; exact pins,
#    clone commands, and why: third_party/README.md)
#    - tau2-bench  = the tau2 runtime engine (own .venv: uv sync --all-extras)
#    - tau3-bench  = the banking domain, tasks + code — episodes run in ITS
#      venv, so it needs: uv sync --extra knowledge (rank_bm25 for retrieval)

# 2. Build the docker images
docker build -t evoskill-solver-sandbox:latest docker/solver-sandbox/
docker build -t ssb-solver:latest -f docker/ssb-solver/Dockerfile docker/ssb-solver/
bash scripts/build_ssb_solver_lo.sh          # ssb-solver-lo:latest (LibreOffice arm)
docker build -t ssb-verifier:latest docker/ssb-verifier/   # grading image

# 3. SpreadsheetBench v2 only: point SSBV2_ROOT at your checkout of the
#    SpreadsheetBench v2 dataset (layout: <root>/spreadsheetbench-v2/<Category>/dataset.json
#    plus per-task spreadsheet/golden files).
export SSBV2_ROOT=/path/to/SpreadsheetBench-v2
```

## Start an evolution

The interface is a conversation with the AO. Write a short prose brief and
hand it over:

```bash
cat > brief.txt <<'EOF'
Evolve a skill library for the banking domain of tau3-bench.
Solver is deepseek-v4-flash-0731, pinned to the native DeepSeek provider.
Three epochs. Hold out a test split and do not touch it.
EOF

evoskill new --brief brief.txt \
    --root-hint third_party/tau3-bench \
    --run-dir outputs/my_run \
    --benchmark tau3
```

The AO reads the repository, resolves what the brief means, asks you about
anything it refuses to pick on its own (split sizes, budgets), and freezes
the experiment: the train/validation/test split, the models, the budgets,
and the scorer. (Headless: pass unproposed answers as
`--script '{"seed": 801, "n_samples": 3}'`.)

### Bootstrap epoch 1: base legs, oracle, declarations

The driver measures an epoch against a no-skills BASE comparator, and it
does not run that comparator for you — the base legs are yours, so that
what everything is compared against was launched and read by a person.
For a tau3 run (env vars shown are the ones the smoke run actually used;
the split/cache-tag values come from the AO's freeze receipt):

```bash
export EVOSKILL_ENV_FILE=$PWD/.env             # holds OPENROUTER_API_KEY
export EVOSKILL_TAU3_SPLIT=$PWD/outputs/my_run/split.json
export EVOSKILL_TAU3_RUN_DIR=outputs/my_run
export EVOSKILL_TAU3_CACHE_TAG=<cache tag from the freeze receipt>
scripts/tau3_leg_b2525.sh my_run_base_train train BASE
scripts/tau3_leg_b2525.sh my_run_base_val   val   BASE

# The evolver's ground-truth oracle (train tasks only, by construction):
EVOSKILL_TAU3_ORACLE_SPLIT=$PWD/outputs/my_run/split.json \
EVOSKILL_TAU3_ORACLE_DIR=$PWD/outputs/my_run/oracle \
    python scripts/make_tau3_oracle.py          # then --verify
```

When both base result JSONs exist, declare them once in
`outputs/my_run/RUNTIME.json` — the driver reads it on every invocation:

```json
{
 "base_leg": "outputs/my_run/my_run_base_train.json",
 "base_leg_val": "outputs/my_run/my_run_base_val.json",
 "digest_caches": [".cache/<the cache_dirs recorded in the base train JSON>"],
 "oracle": "outputs/my_run/oracle"
}
```

`digest_caches` must be copied from the `cache_dirs` field of the base
train result JSON — the runner records the real directory names there, and
the driver refuses to re-derive them.

**The same bootstrap for SpreadsheetBench v2.** Same shape, different
launcher, and two env vars that must be present on *every* later
`evoskill-loop` invocation as well (`SSBV2_ROOT` locates the dataset;
`EVOSKILL_CACHE_DIR` names the one run-cache dir — the SSB v2 default is a
per-tag dir that is empty for a new tag, so leaving it unset silently
re-buys every episode):

```bash
export SSBV2_ROOT=/path/to/SpreadsheetBench-v2/data
export EVOSKILL_CACHE_DIR=$PWD/.cache/<cache tag from the freeze receipt>_cache
export EVOSKILL_TAU3_RUN_DIR=outputs/my_ssb_run       # the loop's var names
export EVOSKILL_TAU3_SPLIT=$PWD/outputs/my_ssb_run/split.json  # predate SSB v2
scripts/ssbv2_leg.sh my_ssb_run_base_train train ""   # "" = no skills = BASE
scripts/ssbv2_leg.sh my_ssb_run_base_val   val   ""

EVOSKILL_SSBV2_ORACLE_SPLIT=$PWD/outputs/my_ssb_run/split.json \
EVOSKILL_SSBV2_ORACLE_DIR=$PWD/outputs/my_ssb_run/oracle \
    python scripts/make_ssbv2_oracle.py               # then --verify
```

Its `RUNTIME.json` additionally names the launcher, and `digest_caches`
holds exactly ONE directory (SSB v2 uses a single cache dir, not tau3's
per-sample dirs):

```json
{
 "leg_script": "scripts/ssbv2_leg.sh",
 "base_leg": "outputs/my_ssb_run/my_ssb_run_base_train.json",
 "base_leg_val": "outputs/my_ssb_run/my_ssb_run_base_val.json",
 "digest_caches": [".cache/<cache tag>_cache"],
 "oracle": "outputs/my_ssb_run/oracle"
}
```

### Drive the epoch

```bash
evoskill-loop --run-dir outputs/my_run --epoch 1 --drive --step \
    --note "first epoch"
```

Each `--drive --step` invocation advances one or more steps of the epoch
(evidence surface, AO-authored evolver prompt, evolver dispatch, evolved
legs, verification gates) and stops at the first HALT, question, or
operator gate. Run it repeatedly and read what it printed before
continuing. `--drive` without `--step` deliberately exits with code 64.

Expect to be stopped **twice at the human gate**: once to read the
AO-authored evolver prompt, and once more to read the evolved skills. Each
time, write `outputs/my_run/e1_PROOFREAD.json` with what you checked plus
the sha256 of the prompt (and, the second time, of the staging dir). The
gate is deliberately not automatable; the HALT message names the exact
keys it wants.

## Model routing notes

- **DeepSeek-family solver models** are hard-pinned to the native DeepSeek
  provider (`order=["deepseek"], allow_fallbacks=false`) at every dispatch
  site. Third-party hosts serve these models with measurably different
  quality; the pin is enforced, not advisory.
- **DeepSeek-family models through the `claude` CLI** additionally require
  the local proxy: `python scripts/cc_proxy.py` (default `127.0.0.1:8100`),
  then `ANTHROPIC_BASE_URL=http://127.0.0.1:8100`. The dispatch guard
  (`scripts/lib/deepseek_pin_guard.sh`) refuses to launch without it.

## Platform notes

- **Evolver containment** uses `sandbox-exec` (macOS-only) to kernel-confine
  the evolver away from held-out task specs. On Linux the dispatch refuses by
  default; set `EVOSKILL_EVOLVER_SANDBOX=0` to run with hook-level guards
  only — acceptable for exploratory runs, record it for reported ones.
- **Phoenix tracing is optional.** Without a running Phoenix instance the
  loop's traces-in-phoenix check reports UNMEASURED and everything else runs.
- `caffeinate` (macOS sleep inhibitor) is used when present, skipped on Linux.

## Tests

```bash
uv run pytest
```

## Layout

```
src/orchestrator/   AO front door (cli), v2 epoch loop, evolver dispatch + containment
src/harness/        direct + claude executors, seat pool, admission, liveness, fleet
src/cache/          rollout replay cache + cache-environment guard
src/agent_profiles/ solver agents (ssbv2_agent, tau2_agent = τ³ adapter) + skill_evolver
src/evaluation/     parallel evaluation driver
scripts/            leg launchers, evolver launcher, digest builders, split makers
docker/             solver sandbox, SSB solver images, SSB verifier (grading)
third_party/        vendored benchmark repos (gitignored; see its README)
```

## License

Apache-2.0.
