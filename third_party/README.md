# third_party/

Vendored clones of third-party benchmark repos that EvoSkill agent profiles
import in-process or drive via subprocess. Not committed to git (see
`.gitignore`: `third_party/*/` is excluded, only this README is tracked) —
each subdirectory is large and fully reproducible from a pinned commit, so it
doesn't belong in version control.

Both directories below are clones of the same upstream repo,
`sierra-research/tau2-bench`, at two different pins. **Only `tau3-bench` is
required for a τ³ banking evolution**; see the role note below.

## tau2-bench — the v1.0.0 engine pin (OPTIONAL)

`src/agent_profiles/tau2_agent/solver.py` launches its subprocess worker with
`uv run --project $TAU2_BENCH_ROOT python <worker.py>`, and the τ³ launchers
export `TAU2_BENCH_ROOT=third_party/tau3-bench` — so on a τ³ run the episodes
execute in **tau3-bench's** venv and this clone is never launched. It is kept
as the exact v1.0.0 provenance pin, and it is what runs if you point
`TAU2_BENCH_ROOT` here (a plain tau2 run, or a fallback when the variable is
unset).

Consequence for setup: `uv sync --all-extras` below, and the portaudio
dependency it drags in, are only needed if you are going to use this clone.
Skip the whole section for a τ³ banking evolution.

- Repo: `sierra-research/tau2-bench` (NOT `amazon-agi/tau2-bench-verified` —
  that fork is stale; NOT `AGI-Eval-Official/tau2-bench-revised` — unrelated).
- Pinned commit: `17e07b1da2bbc0cadfddeea36412686e0604127b`, tag `v1.0.0`.
- To reproduce:
  ```bash
  cd third_party
  git clone https://github.com/sierra-research/tau2-bench.git
  cd tau2-bench
  git checkout v1.0.0
  uv sync --all-extras   # plain `uv sync` is NOT enough — tau2/__init__.py
                          # unconditionally imports a voice stack (scipy,
                          # pyaudio, elevenlabs) even for pure text-mode use.
                          # pyaudio needs the portaudio native lib:
                          #   macOS: brew install portaudio
                          #   Debian/Ubuntu: apt-get install portaudio19-dev
  uv run tau2 --help      # sanity check — should list domains and agents.
  ```
- **Isolated venv, not merged into EvoSkill's own `.venv`.** The
  `--all-extras` dependency surface (elevenlabs, deepgram-sdk, pyaudio,
  boto3, google-cloud-aiplatform, gymnasium, scikit-learn, ...) is irrelevant
  to text-mode use and carries real conflict risk for the main venv. The
  subprocess-worker design keeps it contained.

## tau3-bench — the banking domain (tasks + the code that runs them)

The τ³ banking domain (`banking_knowledge`: task specs, split files, the
document corpus the solver searches, and the domain/retrieval code) landed in
upstream `tau2-bench` main after the v1.0.0 release. It is vendored at a
newer pin under a separate directory so the engine pin above stays frozen.

**This clone is not data-only.** τ³ banking runs set
`TAU2_BENCH_ROOT=third_party/tau3-bench`, and the episode worker is launched
with `uv run --project third_party/tau3-bench` — so episodes execute inside
THIS clone's venv, not tau2-bench's. `uv run` creates that venv implicitly,
but only with the MAIN dependencies; the `banking_knowledge` bm25 retrieval
variants need `rank_bm25`, which upstream ships in the optional `knowledge`
extra. Build the venv with the extra up front, or the first retrieval tool
call of every episode dies with ModuleNotFoundError (the runner's preflight
now refuses at launch instead, but only a synced venv actually runs).

- Repo: `sierra-research/tau2-bench` (same upstream).
- Pinned commit: `363133ada1936491fb5bcec33cd62c3518a99f65`.
- To reproduce:
  ```bash
  cd third_party
  git clone https://github.com/sierra-research/tau2-bench.git tau3-bench
  cd tau3-bench
  git checkout 363133ada1936491fb5bcec33cd62c3518a99f65
  uv sync --extra knowledge   # main deps + rank_bm25/openai for the
                              # banking_knowledge retrieval variants
  ```
- τ³ banking runs point both the data root AND the worker's `--project` here
  (see `scripts/run_tau3_banking.py` and the `TAU2_BENCH_ROOT` environment
  variable).
