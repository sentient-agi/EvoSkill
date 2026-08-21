# Offline RQGM toy

This deterministic example demonstrates the evaluator-co-evolution mechanism
without an API key or a model call. A weak seed evaluator is challenged at a
validation-outcome checkpoint. The challenger wins on five immutable anchors,
the evaluator slot advances from epoch 1 to epoch 2, the old learned record is
tombstoned, and the fixed record survives.

From the repository root:

```powershell
uv run python examples\rqgm_toy\run.py
```

`build_runtime()` has the same shape as a CLI `runtime_factory`. Real domains
replace the toy executor and proposer with restricted code/model adapters and
keep held-out anchor content outside prompts and committed configuration.
