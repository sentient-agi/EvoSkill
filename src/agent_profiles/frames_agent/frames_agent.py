from typing import Union
from pathlib import Path
from claude_agent_sdk import ClaudeAgentOptions
from src.schemas import AgentResponse
from src.agent_profiles.skill_generator import get_project_root
from src.agent_profiles.sdk_config import is_opencode_sdk


FRAMES_AGENT_TOOLS = [
    "WebFetch",
    "WebSearch",
]

# Path to the prompt file (read at runtime)
PROMPT_FILE = Path(__file__).parent / "prompt.txt"


def get_frames_agent_options(
    model: str | None = None,
    provider: str | None = None,
) -> Union[ClaudeAgentOptions, dict]:
    """
    Factory function that creates agent options with the current prompt.

    Args:
        model: Model to use. If None, uses SDK default.
        provider: Provider ID for opencode SDK (e.g., 'gemini', 'arc').
    """
    # Read prompt from disk
    prompt_text = PROMPT_FILE.read_text().strip()

    if is_opencode_sdk():
        return {
            "system": prompt_text,
            "model_id": model or "gpt-oss-120b",
            "provider_id": provider or "arc",
            "tools": {tool: True for tool in FRAMES_AGENT_TOOLS},
            "format": {
                "type": "json_schema",
                "schema": AgentResponse.model_json_schema(),
            },
        }
    else:
        output_format = {
            "type": "json_schema",
            "schema": AgentResponse.model_json_schema(),
        }

        options = ClaudeAgentOptions(
            system_prompt=None,
            output_format=output_format,
            allowed_tools=FRAMES_AGENT_TOOLS,
            setting_sources=["user", "project"],
            permission_mode="acceptEdits",
            cwd=get_project_root(),
            max_buffer_size=10 * 1024 * 1024,  # 10MB buffer
        )

        if model:
            options.model = model

        return options


def make_frames_agent_options(model: str | None = None, provider: str | None = None):
    """Create a factory function for agent options."""

    def factory() -> Union[ClaudeAgentOptions, dict]:
        return get_frames_agent_options(model=model, provider=provider)

    return factory


# For backward compatibility, expose the factory as the options
frames_agent_options = get_frames_agent_options
