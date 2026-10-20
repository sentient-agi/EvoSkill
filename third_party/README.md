# third_party/

Local clones and data that a τ³-bench banking run needs on the host. Nothing here
is committed (`.gitignore`: `third_party/*/` is excluded, only this README is
tracked); every directory is reproducible from a pinned commit.

## tau3-bench: the banking domain and the code that runs it

The τ³ banking domain (`banking_knowledge`: tasks, the knowledge-base documents,
the domain and retrieval code) lives in upstream `sierra-research/tau2-bench` at a
commit after the v1.0.0 release. Rollouts and grading must run the same tau2 code
(the grader replays every conversation strictly), so the commit and the venv are
pinned:

- Repo: `sierra-research/tau2-bench`.
- Pinned commit: `363133ada1936491fb5bcec33cd62c3518a99f65` (tau2 1.0.1).
- The venv: CPython 3.12.13 with the `knowledge` extra (rank_bm25 and the
  embedding retrieval), exactly the 76 packages of
  `scripts/tau3_env/tau3_venv_freeze.txt`.

```bash
cd third_party
git clone https://github.com/sierra-research/tau2-bench.git tau3-bench
cd tau3-bench
git checkout 363133ada1936491fb5bcec33cd62c3518a99f65
git apply ../../scripts/tau3_env/uvlock_1.0.1.patch   # uv.lock names tau2 1.0.1, as pyproject.toml does
uv sync --frozen --no-dev --extra knowledge --python 3.12.13
cd ../..
python3 scripts/tau3_check_venv.py --python third_party/tau3-bench/.venv/bin/python   # rc 0: equal to the pin
```

What uses it:

- `scripts/daytona/tau3_cc_smoke.py` (and `cc_pool_host.py` through it) builds the
  pool machine image's context from a `git archive` of this checkout at the pinned
  commit; `scripts/daytona/Dockerfile.tau3_leg` then builds the same venv at
  `/opt/tau3/venv` and checks it against the same freeze.
- `scripts/tau3_make_data_root.py` copies the domain's data from it.
- The offline tests (`tests/test_tau2_*.py`, `tests/test_tau3_*.py`) run tau2 in
  child processes of `third_party/tau3-bench/.venv/bin/python`; they are skipped
  when it is absent.

## tau3_data_363133a: the curated data root

One copy of the banking data serves the rollouts, the grading and the benchmark
adaptor's `/data` view. It holds the files tau2 reads for `banking_knowledge` and
nothing else (the two aggregate task files, which list every validation and test
task, are left out), a `MANIFEST.sha256`, and the knowledge base's precomputed
document embeddings for the dense search:

```bash
python3 scripts/tau3_make_data_root.py --dest third_party/tau3_data_363133a
third_party/tau3-bench/.venv/bin/python scripts/tau3_build_embeddings_cache.py --root third_party/tau3_data_363133a
```

The embeddings precompute asks OpenRouter for `openai/text-embedding-3-large`
(about $0.02); it reads `OPENROUTER_API_KEY` from the environment, else from the
repository's `.env`. The tests look for the root here, or at
`$EVOSKILL_TAU3_DATA_ROOT`; a run names it in INFRA `data.root` (and
`tau3_cc_smoke.py run --data-root`).

The KB `shell` tool runs commands through Anthropic's sandbox-runtime:
`npm install -g @anthropic-ai/sandbox-runtime@0.0.23` (plus `rg`; on Linux also
`bwrap` and `socat`) on the host that runs tau2.
