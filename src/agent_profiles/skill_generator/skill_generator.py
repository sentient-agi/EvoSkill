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


_JSON_INSTRUCTIONS = """

## Output Format

Respond with a JSON object with these fields:
- "skill_name": the name of the skill created or edited
- "skill_path": relative path to the SKILL.md file
- "description": brief description of what the skill does
- "success": true if the skill was written successfully, false otherwise
"""

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


def make_skill_generator_options(harness: str, model: str | None = None):
    """Create skill generator options factory for the given harness.

    Args:
        harness: One of "claude", "opencode", or "openhands".
        model: Model identifier to use. If None, uses harness default.

    Returns:
        For claude: the static skill_generator_options ClaudeAgentOptions.
        For opencode/openhands: a callable returning a dict.
    """
    if harness == "claude":
        if model:
            opts = ClaudeAgentOptions(
                output_format=skill_generator_output_format,
                system_prompt=skill_generator_system_prompt,
                setting_sources=["user", "project"],
                allowed_tools=SKILL_GENERATOR_TOOLS,
                permission_mode='acceptEdits',
                cwd=get_project_root(),
            )
            opts.model = model
            return opts
        return skill_generator_options

    elif harness == "opencode":
        project_root = get_project_root()
        _model = model or "zai-org/GLM-5"
        system = (SKILL_GENERATOR_SYSTEM_PROMPT + _JSON_INSTRUCTIONS).strip()

        def opencode_factory() -> dict:
            return {
                "provider_id": "togetherai",
                "model_id": _model,
                "system": system,
                "mode": "build",
            }

        return opencode_factory

    elif harness == "openhands":
        project_root = get_project_root()
        _model = model or os.environ.get("OPENHANDS_MODEL", "claude-sonnet-4-6")
        _api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        system = (SKILL_GENERATOR_SYSTEM_PROMPT + _JSON_INSTRUCTIONS).strip()

        def openhands_factory() -> dict:
            return {
                "model_id": _model,
                "api_key": _api_key,
                "system": system,
                "cwd": project_root,
                "workspace": project_root,  # must write skills to .agents/skills/
            }

        return openhands_factory

    else:
        raise ValueError(f"Unknown harness: {harness!r}. Must be 'claude', 'opencode', or 'openhands'.")
