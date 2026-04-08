import os
from claude_agent_sdk import ClaudeAgentOptions
from src.schemas import SkillProposerResponse
from src.agent_profiles.skill_proposer.prompt import SKILL_PROPOSER_SYSTEM_PROMPT
from src.agent_profiles.skill_generator import get_project_root


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

_JSON_INSTRUCTIONS = """

## Output Format

Respond with a JSON object with these fields:
- "action": "create" or "edit"
- "target_skill": skill name if action is "edit", else null
- "proposed_skill": detailed description of the skill to create or modifications needed
- "justification": your reasoning
- "related_iterations": list of relevant past iteration names
"""

skill_proposer_system_prompt = {
    "type": "preset",
    "preset": "claude_code",
    "append": SKILL_PROPOSER_SYSTEM_PROMPT.strip(),
}

skill_proposer_output_format = {
    "type": "json_schema",
    "schema": SkillProposerResponse.model_json_schema(),
}

skill_proposer_options = ClaudeAgentOptions(
    output_format=skill_proposer_output_format,
    system_prompt=skill_proposer_system_prompt,
    allowed_tools=SKILL_PROPOSER_TOOLS,
    cwd=get_project_root(),
)


def make_skill_proposer_options(harness: str, model: str | None = None):
    """Create skill proposer options factory for the given harness.

    Args:
        harness: One of "claude", "opencode", or "openhands".
        model: Model identifier to use. If None, uses harness default.

    Returns:
        For claude: the static skill_proposer_options ClaudeAgentOptions.
        For opencode/openhands: a callable returning a dict.
    """
    if harness == "claude":
        if model:
            opts = ClaudeAgentOptions(
                output_format=skill_proposer_output_format,
                system_prompt=skill_proposer_system_prompt,
                allowed_tools=SKILL_PROPOSER_TOOLS,
                cwd=get_project_root(),
            )
            opts.model = model
            return opts
        return skill_proposer_options

    elif harness == "opencode":
        project_root = get_project_root()
        _model = model or "zai-org/GLM-5"
        system = (SKILL_PROPOSER_SYSTEM_PROMPT + _JSON_INSTRUCTIONS).strip()

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
        system = (SKILL_PROPOSER_SYSTEM_PROMPT + _JSON_INSTRUCTIONS).strip()

        def openhands_factory() -> dict:
            return {
                "model_id": _model,
                "api_key": _api_key,
                "system": system,
                "cwd": project_root,
            }

        return openhands_factory

    else:
        raise ValueError(f"Unknown harness: {harness!r}. Must be 'claude', 'opencode', or 'openhands'.")
