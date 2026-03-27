from pathlib import Path
from claude_agent_sdk import ClaudeAgentOptions
from src.schemas import AgentResponse
from src.agent_profiles.skill_generator import get_project_root
import os


BASE_AGENT_TOOLS = ["Read", "Write", "Bash", "Glob", "Grep", "Edit", "WebFetch", "WebSearch", "TodoWrite", "BashOutput", "Skill"]

# Path to the prompt file (read at runtime)
PROMPT_FILE = Path(__file__).parent / "prompt.txt"


def get_base_agent_options(model: str | None = None) -> ClaudeAgentOptions:
    """
    Factory function that creates ClaudeAgentOptions with the current prompt.

    Reads prompt.txt from disk each time, allowing dynamic updates
    without restarting the Python process.

    Args:
        model: Model to use (e.g., "opus", "sonnet"). If None, uses SDK default.
    """
    # Read prompt from disk
    prompt_text = PROMPT_FILE.read_text().strip()

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


def make_base_agent_options(model: str | None = None):
    """Create a factory function for base agent options with a specific model.

    Args:
        model: Model to use (e.g., "opus", "sonnet"). If None, uses SDK default.

    Returns:
        A callable that returns ClaudeAgentOptions configured with the model.
    """
    def factory() -> ClaudeAgentOptions:
        return get_base_agent_options(model=model)
    return factory


# For backward compatibility, expose the factory as the options
# When passed to Agent, it will be called on each run()
base_agent_options = get_base_agent_options
