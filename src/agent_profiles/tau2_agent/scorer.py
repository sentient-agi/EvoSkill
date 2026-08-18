"""Standalone scorer for tau2-bench tasks.

Mirrors swe_rebench_agent/scorer.py's exact docstring convention: the live
run sets `inline_score` itself (`Tau2Solver.run()`, using tau2's own
`RewardInfo.reward` computed in-process by `run_single_task(...,
evaluation_type=EvaluationType.ALL_WITH_NL_ASSERTIONS)` -- NOT plain `ALL`,
which excludes NL_ASSERTION even when it's part of the task's reward_basis
(see `_tau2_worker.py`'s inline comment for the reproduced error) -- no
Docker exec round-trip needed, unlike TB2/SWE-rebench). This function is
NOT called from the live loop;
it exists purely as a correct, self-sufficient fallback for standalone
rescoring given an already-produced `predicted` JSON blob.

`predicted` is the JSON string `Tau2Solver.run()` builds as `final_answer`
-- the full RewardInfo sub-check breakdown plus cost/tier metadata (see
`solver.py`). The scalar reward is just `json.loads(predicted)["reward"]`;
everything else in that blob is diagnostic context for the evolver, not
needed for scoring.
"""
from __future__ import annotations

import json


def score_tau2(question: str, predicted: str, ground_truth: str) -> float:
    """Return the tau2-computed binary reward (0.0/1.0) from `predicted`."""
    try:
        return float(json.loads(predicted)["reward"])
    except Exception:
        return 0.0
