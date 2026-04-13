from __future__ import annotations

from pathlib import Path

import pytest

from src.harness import set_sdk


@pytest.fixture(autouse=True)
def _reset_sdk() -> None:
    set_sdk("claude")
    yield
    set_sdk("claude")


def _assert_openhands_options(
    options: dict,
    *,
    project_root: Path,
    model: str,
    required_tools: tuple[str, ...],
) -> None:
    provider_id, model_id = model.split("/", 1)

    assert isinstance(options, dict)
    assert options["sdk"] == "openhands"
    assert options["cwd"] == str(project_root)
    assert options["provider_id"] == provider_id
    assert options["model_id"] == model_id
    assert options["model"] == model
    assert options["skills_dir"] == str(project_root / ".claude" / "skills")
    assert isinstance(options["tools"], list)
    for tool in required_tools:
        assert tool in options["tools"]


def test_openhands_base_agent_factories_return_dicts_with_project_root_and_model(
    tmp_path: Path,
) -> None:
    set_sdk("openhands")

    from src.agent_profiles.base_agent.base_agent import (
        make_base_agent_options,
        make_base_agent_options_from_task,
    )

    model = "anthropic/claude-sonnet-4-5-20250929"
    external_data_dir = tmp_path.parent / "external-data"
    external_data_dir.mkdir()

    task_factory = make_base_agent_options_from_task(
        "Answer the question with the final answer only.",
        model=model,
        data_dirs=[str(external_data_dir)],
        project_root=tmp_path,
    )
    eval_factory = make_base_agent_options(
        model=model,
        data_dirs=[str(external_data_dir)],
        project_root=tmp_path,
    )

    task_options = task_factory()
    eval_options = eval_factory()

    _assert_openhands_options(
        task_options,
        project_root=tmp_path,
        model=model,
        required_tools=("Read", "Edit", "Bash", "Skill"),
    )
    _assert_openhands_options(
        eval_options,
        project_root=tmp_path,
        model=model,
        required_tools=("Read", "Edit", "Bash", "Skill"),
    )
    assert task_options["format"]["type"] == "json_schema"
    assert eval_options["format"]["type"] == "json_schema"
    mounted_data_dir = Path(task_options["add_dirs"][0])
    assert mounted_data_dir.is_symlink()
    assert mounted_data_dir.resolve() == external_data_dir.resolve()
    assert mounted_data_dir.parent == tmp_path / ".evoskill" / "runtime" / "data_mounts"
    assert mounted_data_dir.relative_to(tmp_path).as_posix() in task_options["system"]
    assert str(external_data_dir) not in task_options["system"]


def test_openhands_meta_agent_builders_return_dicts_with_project_root(
    tmp_path: Path,
) -> None:
    set_sdk("openhands")
    model = "anthropic/claude-sonnet-4-5-20250929"

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

    _assert_openhands_options(
        skill_proposer,
        project_root=tmp_path,
        model=model,
        required_tools=("Read", "Bash"),
    )
    _assert_openhands_options(
        skill_generator,
        project_root=tmp_path,
        model=model,
        required_tools=("Read", "Write", "Edit", "Skill"),
    )
    _assert_openhands_options(
        prompt_proposer,
        project_root=tmp_path,
        model=model,
        required_tools=("Read", "Bash"),
    )
    _assert_openhands_options(
        prompt_generator,
        project_root=tmp_path,
        model=model,
        required_tools=("Read", "Bash"),
    )
    assert "YAML frontmatter" in skill_generator["system"]
    assert ".claude/skills/<skill-name>/SKILL.md" in skill_generator["system"]
