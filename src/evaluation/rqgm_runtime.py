"""Runtime loading for TOML/CLI-enabled evaluator evolution."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from src.evaluation.rqgm import (
    CheckpointSchedule,
    EvaluatorSlot,
    RQGMController,
    RoleTask,
)


@dataclass(frozen=True)
class RQGMRuntime:
    """Non-serializable runtime capabilities supplied by a project.

    The CLI owns all statistical and scheduling parameters.  The factory only
    supplies private anchors, executable evaluator adapters, proposer
    capabilities, and the role/task topology.
    """

    slots: Sequence[EvaluatorSlot]
    role_tasks: Sequence[RoleTask]


def load_runtime_factory(spec: str):
    """Load a ``module:function`` RQGM runtime factory."""
    if ":" not in spec:
        raise ValueError("evaluator_evolution.runtime_factory must be module:function")
    module_name, attribute = spec.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"RQGM runtime factory {spec!r} is not callable")
    return factory


def _assert_toml_contract(
    controller: RQGMController, rqgm: Any, schedule: CheckpointSchedule
) -> None:
    expected = (
        rqgm.epsilon,
        rqgm.minimum_anchor_outcomes,
        rqgm.minimum_node_outcomes,
        schedule,
        rqgm.anchor_repeats,
    )
    actual = (
        controller.epsilon,
        controller.minimum_anchor_outcomes,
        controller.minimum_node_outcomes,
        controller.checkpoint_schedule,
        controller.anchor_repeats,
    )
    if actual != expected:
        raise ValueError(
            "saved/runtime RQGM controller disagrees with [evaluator_evolution]"
        )


def build_controller_from_project_config(config: Any) -> RQGMController | None:
    """Build the controller requested by ``[evaluator_evolution]``."""
    rqgm = config.evaluator_evolution
    if not rqgm.enabled:
        return None
    if not rqgm.runtime_factory:
        raise ValueError(
            "evaluator evolution is enabled but runtime_factory is empty; "
            "provide a module:function that builds the private anchors, "
            "executors, candidates, and proposers"
        )
    factory = load_runtime_factory(rqgm.runtime_factory)
    runtime = factory(config)
    if inspect.isawaitable(runtime):
        raise TypeError("CLI runtime_factory must return RQGMRuntime synchronously")
    expected_state = Path(rqgm.state_path)
    if not expected_state.is_absolute():
        expected_state = config.project_root / expected_state

    schedule = CheckpointSchedule(
        explicit=tuple(rqgm.checkpoints),
        start=rqgm.checkpoint_start,
        ratio=rqgm.checkpoint_ratio,
    )
    if isinstance(runtime, RQGMRuntime):
        if expected_state.exists():
            controller = RQGMController.restore(
                expected_state, runtime_slots=runtime.slots
            )
        else:
            controller = RQGMController(
                slots=runtime.slots,
                role_tasks=runtime.role_tasks,
                epsilon=rqgm.epsilon,
                minimum_anchor_outcomes=rqgm.minimum_anchor_outcomes,
                minimum_node_outcomes=rqgm.minimum_node_outcomes,
                checkpoint_schedule=schedule,
                anchor_repeats=rqgm.anchor_repeats,
                state_path=expected_state,
            )
        _assert_toml_contract(controller, rqgm, schedule)
        if tuple(controller.role_tasks) != tuple(runtime.role_tasks):
            raise ValueError(
                "saved RQGM role/task topology disagrees with runtime factory"
            )
        return controller

    # A pre-built controller remains accepted for programmatic compatibility,
    # but the CLI refuses silent disagreement with its TOML contract.
    if not isinstance(runtime, RQGMController):
        raise TypeError("runtime_factory must return RQGMRuntime or RQGMController")
    controller = runtime
    _assert_toml_contract(controller, rqgm, schedule)
    if (
        controller.state_path is not None
        and controller.state_path.resolve() != expected_state.resolve()
    ):
        raise ValueError("runtime_factory controller state_path disagrees with TOML")
    controller.state_path = expected_state
    controller.persist()
    return controller
