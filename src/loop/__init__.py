"""Oracle materialization for evolver ground-truth parity.

In this release the package holds exactly one module, `oracle` (`OracleSpec`,
`materialize_oracle`): the machinery that copies a benchmark's evaluator
sources and golden task records into a run's `oracle/` directory so the
evolver diagnoses failures against the same ground truth the grader uses.

The v1 self-improving loop that used to live here (`SelfImprovingLoop`,
`LoopConfig`, `LoopAgents`, git-versioned skill frontiers) is not part of
this release — the v2 orchestrator (`src/orchestrator/loop.py`) replaced it.
This `__init__` therefore re-exports nothing; import from `src.loop.oracle`
directly, as every in-tree caller does.
"""
