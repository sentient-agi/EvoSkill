from pathlib import Path
from types import SimpleNamespace
import importlib

import pytest

from src.cli.config import EvaluatorEvolutionConfig
from src.evaluation.rqgm import (
    AnchorExample,
    CallableEvaluatorExecutor,
    EvaluatorCandidate,
    EvaluatorSlot,
    InMemoryAnchorProvider,
    RoleTask,
)
from src.evaluation.rqgm_runtime import (
    RQGMRuntime,
    build_controller_from_project_config,
)


def runtime() -> RQGMRuntime:
    incumbent = EvaluatorCandidate.create(
        slot_id="judge",
        role_id="solver",
        artifact={"kind": "seed"},
        candidate_id="seed",
    )
    anchor = InMemoryAnchorProvider(
        "private",
        "v1",
        [AnchorExample(str(i), i, True) for i in range(5)],
    )
    executor = CallableEvaluatorExecutor(lambda *_args: True)
    return RQGMRuntime(
        slots=[EvaluatorSlot("judge", "solver", incumbent, anchor, executor)],
        role_tasks=[RoleTask("solver", "task", "learned", "judge")],
    )


def project_config(tmp_path: Path, **overrides):
    values = dict(
        enabled=True,
        epsilon=0.1,
        minimum_anchor_outcomes=5,
        minimum_node_outcomes=3,
        checkpoint_start=7,
        checkpoint_ratio=1.5,
        checkpoints=[],
        anchor_repeats=2,
        state_path=".evoskill/rqgm.json",
        runtime_factory="tests.fake:factory",
    )
    values.update(overrides)
    return SimpleNamespace(
        project_root=tmp_path,
        evaluator_evolution=EvaluatorEvolutionConfig(**values),
    )


def test_cli_constructs_controller_from_toml(monkeypatch, tmp_path):
    runtime_module = importlib.import_module("src.evaluation.rqgm_runtime")
    monkeypatch.setattr(
        runtime_module, "load_runtime_factory", lambda _spec: lambda _config: runtime()
    )
    controller = build_controller_from_project_config(project_config(tmp_path))
    assert controller is not None
    assert controller.epsilon == 0.1
    assert controller.minimum_node_outcomes == 3
    assert controller.checkpoint_schedule.start == 7
    assert controller.anchor_repeats == 2
    assert controller.state_path == tmp_path / ".evoskill/rqgm.json"


def test_cli_restores_state_and_rejects_changed_contract(monkeypatch, tmp_path):
    runtime_module = importlib.import_module("src.evaluation.rqgm_runtime")
    monkeypatch.setattr(
        runtime_module, "load_runtime_factory", lambda _spec: lambda _config: runtime()
    )
    first = build_controller_from_project_config(project_config(tmp_path))
    assert first is not None
    first.validation_outcomes = 4
    first.persist()
    restored = build_controller_from_project_config(project_config(tmp_path))
    assert restored is not None and restored.validation_outcomes == 4

    with pytest.raises(ValueError, match="disagrees"):
        build_controller_from_project_config(project_config(tmp_path, epsilon=0.2))


def test_evaluator_config_rejects_invalid_schedules():
    with pytest.raises(ValueError, match="sorted"):
        EvaluatorEvolutionConfig(checkpoints=[4, 2])
    with pytest.raises(ValueError, match="checkpoint schedule"):
        EvaluatorEvolutionConfig(checkpoint_start=0)
