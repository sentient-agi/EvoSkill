import os
from typing import Union
from pathlib import Path
from claude_agent_sdk import ClaudeAgentOptions

from src.schemas import ToolGeneratorResponse
from src.agent_profiles.skill_generator.prompt import SKILL_GENERATOR_SYSTEM_PROMPT
from src.agent_profiles.sdk_config import is_opencode_sdk


def get_project_root() -> str:
    """Get the project root directory by looking for pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return str(parent)
    # Fallback: go up 3 levels from current file (src/agent_profiles/skill_generator/)
    return str(current.parent.parent.parent)


SKILL_GENERATOR_TOOLS = ["Read", "Write", "Bash", "Glob", "Grep", "Edit", "WebFetch", "WebSearch", "TodoWrite", "BashOutput", "Skill"]


def get_skill_generator_options(
    model: str | None = None,
    provider: str | None = None,
) -> Union[ClaudeAgentOptions, dict]:
    prompt_text = SKILL_GENERATOR_SYSTEM_PROMPT.strip()

    if is_opencode_sdk():
        return {
            "system": prompt_text,
            "model_id": model or "gpt-oss-120b",
            "provider_id": provider or "arc",
            "tools": {tool: True for tool in SKILL_GENERATOR_TOOLS},
            "format": {
                "type": "json_schema",
                "schema": ToolGeneratorResponse.model_json_schema(),
            },
        }

    system_prompt = {
        "type": "preset",
        "preset": "claude_code",
        "append": prompt_text,
    }
    output_format = {
        "type": "json_schema",
        "schema": ToolGeneratorResponse.model_json_schema(),
    }
    options = ClaudeAgentOptions(
        output_format=output_format,
        system_prompt=system_prompt,
        setting_sources=["user", "project"],
        allowed_tools=SKILL_GENERATOR_TOOLS,
        permission_mode='acceptEdits',
        cwd=get_project_root(),
    )
    if model:
        options.model = model
    return options


def make_skill_generator_options(model: str | None = None, provider: str | None = None):
    def factory() -> Union[ClaudeAgentOptions, dict]:
        return get_skill_generator_options(model=model, provider=provider)
    return factory


# Backward compat
skill_generator_options = ClaudeAgentOptions(
    output_format={"type": "json_schema", "schema": ToolGeneratorResponse.model_json_schema()},
    system_prompt={"type": "preset", "preset": "claude_code", "append": SKILL_GENERATOR_SYSTEM_PROMPT.strip()},
    setting_sources=["user", "project"],
    allowed_tools=SKILL_GENERATOR_TOOLS,
    permission_mode='acceptEdits',
    cwd=get_project_root(),
)
