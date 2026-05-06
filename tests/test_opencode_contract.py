from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.harness import Agent
from src.harness.agent import AgentTrace
from src.harness.sdk_config import set_sdk
from src.loop import LoopAgents
from src.loop.config import LoopConfig
from src.loop.runner import SelfImprovingLoop
from src.registry.models import ProgramConfig
from src.schemas import (
    AgentResponse,
    PromptGeneratorResponse,
    PromptProposerResponse,
    SkillProposerResponse,
    ToolGeneratorResponse,
)


@pytest.fixture(autouse=True)
def _reset_sdk() -> None:
    set_sdk("claude")
    yield
    set_sdk("claude")


def _assert_opencode_options(
    options: dict,
    *,
    project_root: Path,
    model: str,
    required_tools: tuple[str, ...],
) -> None:
    provider_id, model_id = model.split("/", 1)

    assert isinstance(options, dict)
    assert options["cwd"] == str(project_root)
    assert options["provider_id"] == provider_id
    assert options["model_id"] == model_id
    assert isinstance(options["tools"], dict)
    assert all(tool == tool.lower() for tool in options["tools"])
    for tool in required_tools:
        assert options["tools"].get(tool) is True


def _make_dummy_loop(*, manager, base_options: dict) -> SelfImprovingLoop:
    agents = LoopAgents(
        base=Agent(lambda: base_options, AgentResponse),
        skill_proposer=Agent(lambda: base_options, SkillProposerResponse),
        prompt_proposer=Agent(lambda: base_options, PromptProposerResponse),
        skill_generator=Agent(lambda: base_options, ToolGeneratorResponse),
        prompt_generator=Agent(lambda: base_options, PromptGeneratorResponse),
    )
    return SelfImprovingLoop(
        LoopConfig(max_iterations=1, frontier_size=1, concurrency=1),
        agents,
        manager,
        train_pools={"math": [("2 + 2", "4")]},
        val_data=[("2 + 2", "4", "math")],
    )


def test_opencode_base_agent_factories_return_dicts_with_project_root_and_model_split(
    tmp_path: Path,
) -> None:
    set_sdk("opencode")

    from src.agent_profiles.base_agent.base_agent import (
        make_base_agent_options,
        make_base_agent_options_from_task,
    )

    model = "anthropic/claude-sonnet-4-6"

    task_factory = make_base_agent_options_from_task(
        "Answer the question with the final answer only.",
        model=model,
        data_dirs=["/tmp/data"],
        project_root=tmp_path,
    )
    eval_factory = make_base_agent_options(
        model=model,
        data_dirs=["/tmp/data"],
        project_root=tmp_path,
    )

    task_options = task_factory()
    eval_options = eval_factory()

    _assert_opencode_options(
        task_options,
        project_root=tmp_path,
        model=model,
        required_tools=("read", "edit", "bash", "skill"),
    )
    _assert_opencode_options(
        eval_options,
        project_root=tmp_path,
        model=model,
        required_tools=("read", "edit", "bash", "skill"),
    )
    assert task_options["system"].startswith("Answer the question with the final answer only.")
    assert task_options["format"]["type"] == "json_schema"
    assert eval_options["format"]["type"] == "json_schema"


def test_opencode_meta_agent_builders_return_dicts_with_project_root(
    tmp_path: Path,
) -> None:
    set_sdk("opencode")
    model = "anthropic/claude-sonnet-4-6"

    from src.agent_profiles.prompt_generator.prompt_generator import (
        make_prompt_generator_options,
    )
    from src.agent_profiles.prompt_proposer.prompt_proposer import (
        make_prompt_proposer_options,
    )
    from src.agent_profiles.skill_generator.skill_generator import (
        make_skill_generator_options,
    )
    from src.agent_profiles.skill_proposer.skill_proposer import (
        make_skill_proposer_options,
    )

    skill_proposer = make_skill_proposer_options(project_root=tmp_path, model=model)
    skill_generator = make_skill_generator_options(project_root=tmp_path, model=model)
    prompt_proposer = make_prompt_proposer_options(project_root=tmp_path, model=model)
    prompt_generator = make_prompt_generator_options(project_root=tmp_path, model=model)

    _assert_opencode_options(
        skill_proposer,
        project_root=tmp_path,
        model=model,
        required_tools=("read", "bash"),
    )
    _assert_opencode_options(
        skill_generator,
        project_root=tmp_path,
        model=model,
        required_tools=("read", "write", "edit"),
    )
    _assert_opencode_options(
        prompt_proposer,
        project_root=tmp_path,
        model=model,
        required_tools=("read", "bash"),
    )
    _assert_opencode_options(
        prompt_generator,
        project_root=tmp_path,
        model=model,
        required_tools=("read", "bash"),
    )
    assert "YAML frontmatter" in skill_generator["system"]
    assert ".claude/skills/<skill-name>/SKILL.md" in skill_generator["system"]
    assert "skill-creator" not in skill_generator["system"].lower()


def test_self_improving_loop_uses_manager_cwd_for_project_paths(tmp_path: Path) -> None:
    class DummyManager:
        def __init__(self, cwd: Path):
            self.cwd = cwd

    loop = _make_dummy_loop(
        manager=DummyManager(tmp_path),
        base_options={
            "system": "test",
            "format": {"type": "json_schema", "schema": AgentResponse.model_json_schema()},
            "tools": {"read": True},
            "provider_id": "anthropic",
            "model_id": "claude-sonnet-4-6",
            "cwd": str(tmp_path),
        },
    )

    assert loop._project_root == tmp_path
    assert loop._feedback_path == tmp_path / ".evoskill" / "feedback_history.md"
    assert loop._checkpoint_path == tmp_path / ".evoskill" / "loop_checkpoint.json"


def test_ensure_base_program_does_not_call_global_claude_base_factory_in_opencode_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    set_sdk("opencode")

    class DummyManager:
        cwd = tmp_path

        def __init__(self) -> None:
            self.created = []
            self.frontier = []

        def list_programs(self):
            return []

        def create_program(self, name, config, parent=None):
            self.created.append((name, config, parent))

        def switch_to(self, name):
            self.frontier.append(("switch", name))

        def update_frontier(self, name, score, max_size):
            self.frontier.append((name, score, max_size))

        def get_frontier(self):
            return ["base"]

    manager = DummyManager()
    loop = _make_dummy_loop(
        manager=manager,
        base_options={
            "system": "OpenCode base prompt",
            "format": {"type": "json_schema", "schema": AgentResponse.model_json_schema()},
            "tools": {"read": True, "bash": True, "edit": True, "skill": True},
            "provider_id": "anthropic",
            "model_id": "claude-sonnet-4-6",
            "cwd": str(tmp_path),
        },
    )

    async def _fake_evaluate(_data):
        return 0.0

    monkeypatch.setattr(loop, "_evaluate", _fake_evaluate)

    asyncio.run(loop._ensure_base_program())

    assert manager.created
    _, created_config, _ = manager.created[0]
    assert created_config.metadata["sdk"] == "opencode"
    assert created_config.metadata["provider_id"] == "anthropic"
    assert created_config.metadata["model_id"] == "claude-sonnet-4-6"
    assert "OpenCode base prompt" in str(created_config.system_prompt)
    assert set(created_config.allowed_tools) == {"read", "bash", "edit", "skill"}


def test_mutate_switches_to_parent_before_reading_program_config(tmp_path: Path) -> None:
    class DummyAgent:
        def __init__(self, output):
            self.output = output

        async def run(self, _query):
            return AgentTrace(
                duration_ms=1,
                total_cost_usd=0.0,
                num_turns=1,
                usage={},
                result="ok",
                is_error=False,
                output=self.output,
                messages=[],
            )

    class DummyManager:
        cwd = tmp_path

        def __init__(self) -> None:
            self.current = "feature/examples-demo"
            self.switches: list[str] = []
            self.created: list[tuple[str, ProgramConfig, str | None]] = []

        def switch_to(self, name):
            self.current = f"program/{name}"
            self.switches.append(name)

        def get_current(self):
            assert self.current == "program/base"
            return ProgramConfig(
                name="base",
                system_prompt={"type": "preset", "preset": "claude_code"},
                allowed_tools=["Read", "Write"],
            )

        def create_program(self, name, config, parent=None):
            self.created.append((name, config, parent))
            self.current = f"program/{name}"

        def commit(self, _message):
            return False

    manager = DummyManager()
    agents = LoopAgents(
        base=DummyAgent(AgentResponse(final_answer="wrong", reasoning="")),
        skill_proposer=DummyAgent(
            SkillProposerResponse(
                proposed_skill="Create a test skill.",
                justification="It covers the missing behavior.",
            )
        ),
        prompt_proposer=DummyAgent(
            PromptProposerResponse(
                proposed_prompt_change="Be more precise.",
                justification="It covers the missing behavior.",
            )
        ),
        skill_generator=DummyAgent(
            ToolGeneratorResponse(generated_skill="test-skill", reasoning="")
        ),
        prompt_generator=DummyAgent(
            PromptGeneratorResponse(optimized_prompt="Be precise.", reasoning="")
        ),
    )
    loop = SelfImprovingLoop(
        LoopConfig(max_iterations=1, evolution_mode="skill_only"),
        agents,
        manager,
        train_pools={"math": [("2 + 2", "4")]},
        val_data=[("2 + 2", "4", "math")],
    )
    failure_trace = AgentTrace[AgentResponse](
        duration_ms=1,
        total_cost_usd=0.0,
        num_turns=1,
        usage={},
        result="wrong",
        is_error=False,
        output=AgentResponse(final_answer="wrong", reasoning=""),
        messages=[],
    )

    result = asyncio.run(
        loop._mutate("base", [(failure_trace, "wrong", "4", "math")], iteration=1)
    )

    assert result is not None
    assert manager.switches == ["base"]
    assert manager.created[0][0] == "iter-skill-1"
    assert manager.created[0][2] == "base"
