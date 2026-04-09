from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agent_profiles.options_utils import build_opencode_options, resolve_project_root
from src.agent_profiles.sdk_config import is_claude_sdk
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
        from claude_agent_sdk import ClaudeAgentOptions

        options = ClaudeAgentOptions(
            output_format={
                "type": "json_schema",
                "schema": ToolGeneratorResponse.model_json_schema(),
            },
            system_prompt={
                "type": "preset",
                "preset": "claude_code",
                "append": SKILL_GENERATOR_SYSTEM_PROMPT.strip(),
            },
            setting_sources=["user", "project"],
            allowed_tools=SKILL_GENERATOR_TOOLS,
            permission_mode="acceptEdits",
            cwd=str(resolve_project_root(project_root)),
        )
        if model:
            options.model = model
        return options

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
