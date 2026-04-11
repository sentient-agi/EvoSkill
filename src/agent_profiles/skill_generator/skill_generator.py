from __future__ import annotations

from pathlib import Path
from typing import Any

from src.harness import build_claudecode_options, build_opencode_options, resolve_project_root, is_claude_sdk
from src.schemas import ToolGeneratorResponse
from src.agent_profiles.skill_generator.prompt import SKILL_GENERATOR_SYSTEM_PROMPT


def get_project_root() -> str:
    """Backward-compatible project-root helper."""
    return str(resolve_project_root())


# Default available tools for skill generator
SKILL_GENERATOR_TOOLS = [
    "Read",
    "Write",
    "Bash",
    "Glob",
    "Grep",
    "Edit",
    "WebFetch",
    "WebSearch",
    "TodoWrite",
    "BashOutput",
    "Skill",
]


def get_skill_generator_options(
    model: str | None = None,
    project_root: str | Path | None = None,
) -> Any:
    if is_claude_sdk():
        return build_claudecode_options(
            system=SKILL_GENERATOR_SYSTEM_PROMPT.strip(),
            schema=ToolGeneratorResponse.model_json_schema(),
            tools=SKILL_GENERATOR_TOOLS,
            project_root=project_root,
            model=model,
            setting_sources=["user", "project"],
            permission_mode="acceptEdits",
        )
    return build_opencode_options(
        system=SKILL_GENERATOR_SYSTEM_PROMPT.strip(),
        schema=ToolGeneratorResponse.model_json_schema(),
        tools=SKILL_GENERATOR_TOOLS,
        project_root=project_root,
        model=model,
    )


def make_skill_generator_options(
    *,
    project_root: str | Path | None = None,
    model: str | None = None,
):
    return get_skill_generator_options(model=model, project_root=project_root)


skill_generator_options = get_skill_generator_options()
