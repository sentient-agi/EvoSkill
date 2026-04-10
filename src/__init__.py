"""Top-level package exports without import-time side effects."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "FeedbackDescent",
    "EvaluationResult",
    "FeedbackEntry",
    "FeedbackDescentResult",
    "Proposer",
    "Evaluator",
    "EvoSkill",
    "EvalRunner",
    "EvalSummary",
]

_EXPORTS = {
    "FeedbackDescent": ("src.feedback_descent", "FeedbackDescent"),
    "EvaluationResult": ("src.feedback_descent", "EvaluationResult"),
    "FeedbackEntry": ("src.feedback_descent", "FeedbackEntry"),
    "FeedbackDescentResult": ("src.feedback_descent", "FeedbackDescentResult"),
    "Proposer": ("src.feedback_descent", "Proposer"),
    "Evaluator": ("src.feedback_descent", "Evaluator"),
    "EvoSkill": ("src.api", "EvoSkill"),
    "EvalRunner": ("src.api", "EvalRunner"),
    "EvalSummary": ("src.api", "EvalSummary"),
}


def __getattr__(name: str) -> Any:
    """Resolve public exports lazily to keep package imports cheap."""
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
