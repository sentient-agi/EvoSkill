import os
from pathlib import Path
from claude_agent_sdk import ClaudeAgentOptions

from src.schemas import ToolGeneratorResponse
from src.agent_profiles.skill_generator.prompt import SKILL_GENERATOR_SYSTEM_PROMPT


def get_project_root() -> str:
    """Get the project root directory by looking for pyproject.toml."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return str(parent)
    # Fallback: go up 3 levels from current file (src/agent_profiles/skill_generator/)
    return str(current.parent.parent.parent)


skill_generator_system_prompt = {
    "type": "preset",
    "preset": "claude_code",
    "append": SKILL_GENERATOR_SYSTEM_PROMPT.strip()
}

skill_generator_output_format = {
    "type": "json_schema",
    "schema": ToolGeneratorResponse.model_json_schema()
}

# Default available tools for skill generator
SKILL_GENERATOR_TOOLS = ["Read", "Write", "Bash", "Glob", "Grep", "Edit", "WebFetch", "WebSearch", "TodoWrite", "BashOutput", "Skill"]

skill_generator_options = ClaudeAgentOptions(
    output_format=skill_generator_output_format,
    system_prompt=skill_generator_system_prompt,
    setting_sources=["user", "project"],
    allowed_tools=SKILL_GENERATOR_TOOLS,
    permission_mode='acceptEdits',
    cwd=get_project_root(),
)
