from __future__ import annotations

from typing import Any

from src.harness import build_options
from src.schemas import SkillEvolverResponse
from .prompt import SKILL_EVOLVER_SYSTEM_PROMPT


SKILL_EVOLVER_TOOLS = [
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
    "Skill",
]


def get_skill_evolver_options(
    model: str | None = None,
    project_root: str | None = None,
) -> Any:
    """Factory for the unified skill evolver (proposer + generator in one pass)."""
    return build_options(
        system=SKILL_EVOLVER_SYSTEM_PROMPT.strip(),
        schema=SkillEvolverResponse.model_json_schema(),
        tools=SKILL_EVOLVER_TOOLS,
        project_root=project_root,
        model=model,
        setting_sources=["user", "project"],
        permission_mode="acceptEdits",
    )


def make_skill_evolver_options(
    *,
    project_root: str | None = None,
    model: str | None = None,
):
    return get_skill_evolver_options(model=model, project_root=project_root)


skill_evolver_options = get_skill_evolver_options()
