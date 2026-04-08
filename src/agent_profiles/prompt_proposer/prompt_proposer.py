import os
from claude_agent_sdk import ClaudeAgentOptions
from src.schemas import PromptProposerResponse
from src.agent_profiles.prompt_proposer.prompt import PROMPT_PROPOSER_SYSTEM_PROMPT
from src.agent_profiles.skill_generator import get_project_root


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

_JSON_INSTRUCTIONS = """

## Output Format

Respond with a JSON object with these fields:
- "proposed_prompt_change": detailed description of the prompt modification needed
- "justification": your reasoning referencing specific trace moments and how this addresses the gap
"""

prompt_proposer_system_prompt = {
    "type": "preset",
    "preset": "claude_code",
    "append": PROMPT_PROPOSER_SYSTEM_PROMPT.strip(),
}

prompt_proposer_output_format = {
    "type": "json_schema",
    "schema": PromptProposerResponse.model_json_schema(),
}

prompt_proposer_options = ClaudeAgentOptions(
    output_format=prompt_proposer_output_format,
    system_prompt=prompt_proposer_system_prompt,
    allowed_tools=PROMPT_PROPOSER_TOOLS,
    cwd=get_project_root(),
)


def make_prompt_proposer_options(harness: str, model: str | None = None):
    """Create prompt proposer options factory for the given harness.

    Args:
        harness: One of "claude", "opencode", or "openhands".
        model: Model identifier to use. If None, uses harness default.

    Returns:
        For claude: the static prompt_proposer_options ClaudeAgentOptions.
        For opencode/openhands: a callable returning a dict.
    """
    if harness == "claude":
        if model:
            opts = ClaudeAgentOptions(
                output_format=prompt_proposer_output_format,
                system_prompt=prompt_proposer_system_prompt,
                allowed_tools=PROMPT_PROPOSER_TOOLS,
                cwd=get_project_root(),
            )
            opts.model = model
            return opts
        return prompt_proposer_options

    elif harness == "opencode":
        project_root = get_project_root()
        _model = model or "zai-org/GLM-5"
        system = (PROMPT_PROPOSER_SYSTEM_PROMPT + _JSON_INSTRUCTIONS).strip()

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
        system = (PROMPT_PROPOSER_SYSTEM_PROMPT + _JSON_INSTRUCTIONS).strip()

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
