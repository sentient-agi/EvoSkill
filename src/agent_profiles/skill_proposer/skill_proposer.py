from __future__ import annotations

from typing import Any

from src.harness import build_options
from src.schemas import SkillProposerResponse
from src.agent_profiles.skill_proposer.prompt import SKILL_PROPOSER_SYSTEM_PROMPT


SKILL_PROPOSER_TOOLS = [
    "Read",
    "Bash",
    "Glob",
    "Grep",
    "WebFetch",
    "WebSearch",
    "TodoWrite",
    "BashOutput",
]


def get_skill_proposer_options(
    model: str | None = None,
    project_root: str | None = None,
) -> Any:
    return build_options(
        system=SKILL_PROPOSER_SYSTEM_PROMPT.strip(),
        schema=SkillProposerResponse.model_json_schema(),
        tools=SKILL_PROPOSER_TOOLS,
        project_root=project_root,
        model=model,
    )


def make_skill_proposer_options(
    *,
    project_root: str | None = None,
    model: str | None = None,
):
    return get_skill_proposer_options(model=model, project_root=project_root)


skill_proposer_options = get_skill_proposer_options()
