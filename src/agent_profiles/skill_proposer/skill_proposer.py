from __future__ import annotations

from typing import Any

from src.agent_profiles.options_utils import build_opencode_options, resolve_project_root
from src.agent_profiles.sdk_config import is_claude_sdk
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
    if is_claude_sdk():
        from claude_agent_sdk import ClaudeAgentOptions

        options = ClaudeAgentOptions(
            output_format={
                "type": "json_schema",
                "schema": SkillProposerResponse.model_json_schema(),
            },
            system_prompt={
                "type": "preset",
                "preset": "claude_code",
                "append": SKILL_PROPOSER_SYSTEM_PROMPT.strip(),
            },
            allowed_tools=SKILL_PROPOSER_TOOLS,
            cwd=str(resolve_project_root(project_root)),
        )
        if model:
            options.model = model
        return options

    return build_opencode_options(
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
