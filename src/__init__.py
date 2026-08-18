"""Top-level package exports without eager heavy imports."""

from importlib import import_module

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


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
