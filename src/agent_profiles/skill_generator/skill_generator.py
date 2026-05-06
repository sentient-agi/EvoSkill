from __future__ import annotations

from pathlib import Path
from typing import Any

from src.harness import build_options, resolve_project_root
from src.schemas import ToolGeneratorResponse
from src.agent_profiles.skill_generator.prompt import SKILL_GENERATOR_SYSTEM_PROMPT


def get_project_root() -> str:
    """Backward-compatible project-root helper."""
    return str(resolve_project_root())


SKILL_GENERATOR_TOOLS = [
    "Read",
    "Write",
    "Glob",
    "Grep",
    "Edit",
]


def get_skill_generator_options(
    model: str | None = None,
    project_root: str | Path | None = None,
) -> Any:
    return build_options(
        system=SKILL_GENERATOR_SYSTEM_PROMPT.strip(),
        schema=ToolGeneratorResponse.model_json_schema(),
        tools=SKILL_GENERATOR_TOOLS,
        project_root=project_root,
        model=model,
        setting_sources=["user", "project"],
        permission_mode="acceptEdits",
    )


def make_skill_generator_options(
    *,
    project_root: str | Path | None = None,
    model: str | None = None,
):
    return get_skill_generator_options(model=model, project_root=project_root)


skill_generator_options = get_skill_generator_options()
