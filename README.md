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
  the evolver run through the `claude` binary, and they need a **Claude
  subscription** — run `claude login`. An `ANTHROPIC_API_KEY` will NOT work for
  these two steps: `subscription_env()` in `src/orchestrator/quota_scope.py`
  deliberately removes `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` and
  `ANTHROPIC_BASE_URL` from every `claude` child that runs a Claude-family
  model. That is not an oversight (#307): the quota gate reads the subscription
  window, so if the child silently spent API credit instead, the gate would be
  guarding a budget the run does not use. The key is still needed for solver
  models that reach Anthropic through a gateway.
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
#    - tau3-bench  = the banking domain, tasks + code, AND the venv every
#      episode runs in: uv sync --extra knowledge (rank_bm25 for retrieval)
#    - tau2-bench  = the v1.0.0 engine pin. Only needed if you set
#      TAU2_BENCH_ROOT to point episodes at it instead; a plain tau3 run
#      never launches it, so its venv is OPTIONAL.

# 2. Build the docker images. Only SpreadsheetBench v2 needs these — a tau3
#    banking run uses none of them.
docker build -t evoskill-solver-sandbox:latest docker/solver-sandbox/
docker build -t ssb-verifier:latest docker/ssb-verifier/   # grading image
bash scripts/build_ssb_solver_lo.sh          # ssb-solver-lo:latest (LibreOffice arm)
#    build_ssb_solver_lo.sh builds its own base and then runs ~12 acceptance
#    checks INSIDE the built container (soffice present, python3-uno importable,
#    the pinned openpyxl/pandas/numpy not shadowed, and a real
#    formula -> recalculated-value round-trip). It refuses to hand over an image
#    that fails any of them. If you already hold the frozen 2026-07-15 base
#    tagged `ssb-solver:no-libreoffice-0715`, it is used and checked by ID —
#    that is the reproduce-the-published-numbers path, not a build requirement.

# 3. SpreadsheetBench v2 only: point SSBV2_ROOT at your checkout of the
#    SpreadsheetBench v2 dataset. Give it the directory that CONTAINS the
#    category dirs' parent, i.e. either <root>/data/spreadsheetbench-v2/... or
#    <root>/spreadsheetbench-v2/... resolves — the loader tries both.
export SSBV2_ROOT=/path/to/SpreadsheetBench-v2
# Check it before going further: a wrong root does not raise, it warns per
# category and yields an EMPTY task list, which reads downstream as a run with
# nothing in it rather than as a misconfiguration.
uv run python -c "
import os
from pathlib import Path
from src.agent_profiles.ssbv2_agent.dataset import load_ssbv2
tr, va, te = load_ssbv2(Path(os.environ['SSBV2_ROOT']))
assert tr and va and te, 'SSBV2_ROOT is wrong — 0 tasks loaded'
print(f'SSBV2_ROOT OK: {len(tr)} train / {len(va)} val / {len(te)} test')"
```

Everything below runs through `uv run`, because `uv sync` installs the
`evoskill` and `evoskill-loop` entry points into `.venv/bin` and does not put
them on your PATH. `source .venv/bin/activate` once instead if you prefer, and
drop the prefix.

## Start an evolution

The interface is a conversation with the AO. Write a short prose brief and
hand it over:

```bash
cat > brief.txt <<'EOF'
Evolve a skill library for the banking domain of tau3-bench.
Solver is deepseek-v4-flash-0731, pinned to the native DeepSeek provider.
Three epochs. Hold out a test split and do not touch it.
EOF

uv run evoskill new --brief brief.txt \
    --root-hint third_party/tau3-bench \
    --run-dir outputs/my_run \
    --benchmark tau3
```

The AO reads the repository, resolves what the brief means, asks you about
anything it refuses to pick on its own (split sizes, budgets), and freezes
the experiment: the train/validation/test split, the models, the budgets,
and the scorer. (Headless: `--script` takes a PATH to a JSON file of
pre-recorded answers, not inline JSON:
`echo '{"seed": 801, "n_samples": 3}' > script.json` then `--script script.json`.)

**Write down the split filename it prints.** The freeze receipt names the
split it wrote — `split_<n_train><n_val><n_test>.json`, e.g. `split_325.json`
for 3/2/5 — and every bootstrap step below needs that exact name. It is not
`split.json`.

### Bootstrap epoch 1: base legs, oracle, declarations

The driver measures an epoch against a no-skills BASE comparator, and it
does not run that comparator for you — the base legs are yours, so that
what everything is compared against was launched and read by a person.
For a tau3 run (env vars shown are the ones the smoke run actually used;
the split/cache-tag values come from the AO's freeze receipt):

```bash
export EVOSKILL_ENV_FILE=$PWD/.env             # holds OPENROUTER_API_KEY
export EVOSKILL_TAU3_SPLIT=$PWD/outputs/my_run/split_325.json   # EXACT name from the receipt
export EVOSKILL_TAU3_RUN_DIR=outputs/my_run
export EVOSKILL_TAU3_CACHE_TAG=<cache tag from the freeze receipt>
export EVOSKILL_SOLVER_MODEL=<solver model from the freeze receipt>   # same model as the evolved legs
scripts/tau3_leg_b2525.sh my_run_base_train train BASE
scripts/tau3_leg_b2525.sh my_run_base_val   val   BASE

# The evolver's ground-truth oracle (train tasks only, by construction):
EVOSKILL_TAU3_ORACLE_SPLIT=$PWD/outputs/my_run/split_325.json \
EVOSKILL_TAU3_ORACLE_DIR=$PWD/outputs/my_run/oracle \
    uv run python scripts/make_tau3_oracle.py   # then --verify
```

All three `EVOSKILL_TAU3_*` exports **and `EVOSKILL_SOLVER_MODEL`** are
mandatory — the launcher refuses
without them rather than defaulting, because a defaulted cache tag would mix
this run's episodes into another run's cache bucket.

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
launcher. `SSBV2_ROOT` must be exported for *every* later `evoskill-loop`
invocation too — the loop passes it to the evolver's leak hook as the
held-out data root. `EVOSKILL_CACHE_DIR` is required by the leg launcher
here (its default is a per-tag dir that is empty for a new tag, so leaving it
unset silently re-buys every episode); the loop sets its own per-epoch value
when it launches legs itself, so exporting it later is harmless but does
nothing:

```bash
export EVOSKILL_CACHE_DIR=$PWD/.cache/<cache tag from the freeze receipt>_cache
export EVOSKILL_TAU3_RUN_DIR=outputs/my_ssb_run       # the loop's var names
export EVOSKILL_TAU3_SPLIT=$PWD/outputs/my_ssb_run/split_325.json  # predate SSB v2
scripts/ssbv2_leg.sh my_ssb_run_base_train train ""   # "" = no skills = BASE
scripts/ssbv2_leg.sh my_ssb_run_base_val   val   ""

EVOSKILL_SSBV2_ORACLE_SPLIT=$PWD/outputs/my_ssb_run/split_325.json \
EVOSKILL_SSBV2_ORACLE_DIR=$PWD/outputs/my_ssb_run/oracle \
    uv run python scripts/make_ssbv2_oracle.py        # then --verify
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
uv run evoskill-loop --run-dir outputs/my_run --epoch 1 --drive --step \
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
  default. The opt-out is `"evolver_sandbox": false` in the run's
  `RUNTIME.json` — a recorded decision, which is why it lives in the run's
  declaration rather than in the environment. Setting
  `EVOSKILL_EVOLVER_SANDBOX=0` in your shell has no effect on a driven epoch:
  the loop rebuilds that variable from the RUNTIME.json value on every
  dispatch, so an export is silently overwritten. It applies only when you run
  `scripts/tau3_evolver.sh` by hand.
- **The AO front door is also `sandbox-exec`-gated, and it is a SEPARATE gate.**
  `evoskill new --brief` (via the brief/clarify composer) and the loop's
  prompt-authoring step both refuse on a host without `sandbox-exec` — which is
  every Linux host. `"evolver_sandbox": false` does NOT lift this; that key is
  read only for the evolver dispatch. The opt-out here is
  `EVOSKILL_AO_ALLOW_UNCONTAINED=1`, and it is a real trade: the AO then runs
  uncontained, regaining write access to the run directory — including
  `ANSWERS_e{N}.json` and `e{N}_PROOFREAD.json`, the files a later step reads
  back as the operator's own answers. Set it as a recorded decision, not a
  reflex. On Linux, either set it or drive the loop from an existing
  `RUNTIME.json` instead of the brief front door.
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
