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


def _register_builtins() -> None:
    """Register only the base built-in task."""
    from src.agent_profiles import make_base_agent_options

    register_task(
        TaskConfig(
            name="base",
            make_agent_options=make_base_agent_options,
            scorer=None,
            default_dataset=".dataset/new_runs_base/solved_dataset.csv",
        )
    )


_register_builtins()
