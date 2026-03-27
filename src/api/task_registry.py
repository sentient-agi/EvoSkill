"""Task configuration registry.

Maps task names to their agent factory, scorer, and column conventions.
Eliminates per-task script duplication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# Type aliases
ScorerFn = Callable[
    [str, str, str], float
]  # (question, predicted, ground_truth) -> score


@dataclass
class TaskConfig:
    """Configuration for a registered task."""

    name: str
    make_agent_options: Callable[
        ..., Any
    ]  # Factory that returns agent options callable
    scorer: ScorerFn | None = None  # None = default multi-tolerance scorer
    question_col: str = "question"
    answer_col: str = "ground_truth"
    category_col: str = "category"
    column_renames: dict[str, str] = field(default_factory=dict)
    default_dataset: str = ""


_REGISTRY: dict[str, TaskConfig] = {}


def register_task(config: TaskConfig) -> None:
    """Register a task configuration."""
    _REGISTRY[config.name] = config


def get_task(name: str) -> TaskConfig:
    """Get a registered task configuration by name.

    Raises:
        KeyError: If the task name is not registered.
    """
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise KeyError(f"Unknown task {name!r}. Available tasks: {available}")
    return _REGISTRY[name]


def list_tasks() -> list[str]:
    """List all registered task names."""
    return sorted(_REGISTRY.keys())


def _sealqa_scorer(question: str, predicted: str, ground_truth: str) -> float:
    """Wrapper around score_sealqa matching the runner's (question, predicted, ground_truth) signature."""
    from src.evaluation.sealqa_scorer import score_sealqa

    return score_sealqa(question, ground_truth, predicted)

def _livecodebench_scorer(question: str, predicted: str, ground_truth: str) -> float:
    """Wrapper around score_livecodebench matching the runner's signature."""
    from src.evaluation.livecodebench import score_livecodebench
    return score_livecodebench(question, ground_truth, predicted)


def _gdpval_scorer(question: str, predicted: str, ground_truth: str) -> float:
    """Wrapper around score_gdpval matching the runner's signature."""
    from src.evaluation.gdpval_scorer import score_gdpval

    return score_gdpval(question, predicted, ground_truth)


def _register_builtins() -> None:
    """Register built-in task configurations."""
    from src.agent_profiles import (
        make_base_agent_options,
        make_dabstep_agent_options,
        make_livecodebench_agent_options,
        make_sealqa_agent_options,
        make_gdpval_agent_options,
        make_frames_agent_options,
    )

    register_task(
        TaskConfig(
            name="base",
            make_agent_options=make_base_agent_options,
            scorer=None,
            default_dataset=".dataset/new_runs_base/solved_dataset.csv",
        )
    )

    register_task(
        TaskConfig(
            name="dabstep",
            make_agent_options=make_dabstep_agent_options,
            scorer=None,
            column_renames={"level": "category", "answer": "ground_truth"},
            default_dataset=".dataset/dabstep_data.csv",
        )
    )

    register_task(
        TaskConfig(
            name="sealqa",
            make_agent_options=make_sealqa_agent_options,
            scorer=_sealqa_scorer,
            column_renames={"topic": "category", "answer": "ground_truth"},
            default_dataset=".dataset/seal-0.csv",
        )
    )

    # Ensure LiveCodeBench dataset is downloaded
    from src.evaluation.livecodebench import ensure_livecodebench_dataset

    livecodebench_dataset = str(ensure_livecodebench_dataset())

    register_task(
        TaskConfig(
            name="livecodebench",
            make_agent_options=make_livecodebench_agent_options,
            scorer=_livecodebench_scorer,
            question_col="formatted_question",
            answer_col="public_test_cases",
            category_col="platform",
            default_dataset=livecodebench_dataset,
        )
    )

    register_task(
        TaskConfig(
            name="gdpval",
            make_agent_options=make_gdpval_agent_options,
            scorer=_gdpval_scorer,
            question_col="prompt",
            answer_col="rubric_json",
            category_col="sector",
            column_renames={"sector": "category"},
            default_dataset=".dataset/gdpval/gdpval.csv",
        )
    )

    register_task(
        TaskConfig(
            name="frames",
            make_agent_options=make_frames_agent_options,
            scorer=None,
            column_renames={"Prompt": "question", "Answer": "ground_truth", "reasoning_types": "category"},
            default_dataset=".dataset/frames_filtered.csv",
        )
    )


_register_builtins()
