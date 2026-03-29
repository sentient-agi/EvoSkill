from typing import Union
from pathlib import Path
from claude_agent_sdk import ClaudeAgentOptions
from src.schemas import AgentResponse
from src.agent_profiles.skill_generator import get_project_root
from src.agent_profiles.sdk_config import is_opencode_sdk
import os


BASE_AGENT_TOOLS = ["Read", "Write", "Bash", "Glob", "Grep", "Edit", "WebFetch", "WebSearch", "TodoWrite", "BashOutput", "Skill"]

# Path to the prompt file (read at runtime)
PROMPT_FILE = Path(__file__).parent / "prompt.txt"


def get_base_agent_options(
    model: str | None = None,
    provider: str | None = None,
) -> Union[ClaudeAgentOptions, dict]:
    """
    Factory function that creates agent options with the current prompt.

    Reads prompt.txt from disk each time, allowing dynamic updates
    without restarting the Python process.

    Args:
        model: Model to use (e.g., "opus", "sonnet"). If None, uses SDK default.
        provider: Provider ID for opencode SDK (e.g., 'arc', 'openai'). If None, uses 'arc'.
    """
    # Read prompt from disk
    prompt_text = PROMPT_FILE.read_text().strip()

    if is_opencode_sdk():
        file_path = os.path.join(get_project_root(), "data_directories/treasury_bulletins_parsed/")
        system_with_dir = (
            f"{prompt_text}\n\n"
            f"The treasury bulletins data directory is at: {file_path}\n"
            f"Use Read, Glob, or Bash tools to access files in that directory."
        )
        return {
            "system": system_with_dir,
            "model_id": model or "gpt-oss-120b",
            "provider_id": provider or "arc",
            "tools": {tool: True for tool in BASE_AGENT_TOOLS},
            "format": {
                "type": "json_schema",
                "schema": AgentResponse.model_json_schema(),
            },
        }

    system_prompt = {
        "type": "preset",
        "preset": "claude_code",
        "append": prompt_text
    }

    output_format = {
        "type": "json_schema",
        "schema": AgentResponse.model_json_schema()
    }

    file_path = os.path.join(get_project_root(), "data_directories/treasury_bulletins_parsed/")

    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        output_format=output_format,
        allowed_tools=BASE_AGENT_TOOLS,
        setting_sources=["user", "project"],  # Load Skills from filesystem
        permission_mode='acceptEdits',
        add_dirs=[file_path],
        cwd=get_project_root(),
        max_buffer_size=10 * 1024 * 1024,  # 10MB buffer (default is 1MB)
    )

    if model:
        options.model = model

    return options


def make_base_agent_options(model: str | None = None, provider: str | None = None):
    """Create a factory function for base agent options with a specific model.

    Args:
        model: Model to use (e.g., "opus", "sonnet"). If None, uses SDK default.
        provider: Provider ID for opencode SDK (e.g., 'arc', 'openai'). If None, uses 'arc'.

    Returns:
        A callable that returns agent options configured with the model and provider.
    """
    def factory() -> Union[ClaudeAgentOptions, dict]:
        return get_base_agent_options(model=model, provider=provider)
    return factory


# For backward compatibility, expose the factory as the options
# When passed to Agent, it will be called on each run()
base_agent_options = get_base_agent_options
