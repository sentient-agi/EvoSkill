from __future__ import annotations

from typing import Any

from src.harness import build_options
from src.schemas import PromptProposerResponse
from src.agent_profiles.prompt_proposer.prompt import PROMPT_PROPOSER_SYSTEM_PROMPT


PROMPT_PROPOSER_TOOLS = [
    "Read",
    "Bash",
    "Glob",
    "Grep",
    "WebFetch",
    "WebSearch",
    "TodoWrite",
    "BashOutput",
]


def get_prompt_proposer_options(
    model: str | None = None,
    project_root: str | None = None,
) -> Any:
    return build_options(
        system=PROMPT_PROPOSER_SYSTEM_PROMPT.strip(),
        schema=PromptProposerResponse.model_json_schema(),
        tools=PROMPT_PROPOSER_TOOLS,
        project_root=project_root,
        model=model,
    )


def make_prompt_proposer_options(
    *,
    project_root: str | None = None,
    model: str | None = None,
):
    return get_prompt_proposer_options(model=model, project_root=project_root)


prompt_proposer_options = get_prompt_proposer_options()
