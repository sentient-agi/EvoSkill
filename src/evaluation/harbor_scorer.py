"""Scorer for predictions produced by HarborAgent.

HarborAgent encodes each Harbor trial's outcome as a JSON envelope in
trace.output.final_answer:

    {"reward": 1.0, "metric": "reward.txt", "exit_status": "verified"}

This scorer just decodes that envelope and returns the primary reward. The
ground_truth argument is ignored by design — Harbor's in-container verifier is
the source of truth, and HarborAgent has already collapsed it into a number.
"""

from __future__ import annotations

import json


def harbor_reward_scorer(question: str, predicted: str, ground_truth: str) -> float:
    if not predicted:
        return 0.0
    try:
        envelope = json.loads(predicted)
    except (TypeError, ValueError):
        return 0.0
    if not isinstance(envelope, dict):
        return 0.0
    raw = envelope.get("reward", 0.0)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0
