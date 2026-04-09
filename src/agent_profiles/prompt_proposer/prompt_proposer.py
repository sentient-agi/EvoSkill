from __future__ import annotations

from typing import Any

from src.agent_profiles.options_utils import build_opencode_options, resolve_project_root
from src.agent_profiles.sdk_config import is_claude_sdk
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
    if is_claude_sdk():
        from claude_agent_sdk import ClaudeAgentOptions

        options = ClaudeAgentOptions(
            output_format={
                "type": "json_schema",
                "schema": PromptProposerResponse.model_json_schema(),
            },
            system_prompt={
                "type": "preset",
                "preset": "claude_code",
                "append": PROMPT_PROPOSER_SYSTEM_PROMPT.strip(),
            },
            allowed_tools=PROMPT_PROPOSER_TOOLS,
            cwd=str(resolve_project_root(project_root)),
        )
        if model:
            options.model = model
        return options

    return build_opencode_options(
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
