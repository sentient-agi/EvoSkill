from typing import Union
from pathlib import Path
from claude_agent_sdk import ClaudeAgentOptions
from src.schemas import AgentResponse
from src.agent_profiles.skill_generator import get_project_root
from src.agent_profiles.sdk_config import is_opencode_sdk


GDPVAL_AGENT_TOOLS = ["Read", "Write", "Bash", "Glob", "Grep", "Edit", "WebFetch", "WebSearch", "TodoWrite", "BashOutput", "Skill"]

# Path to the prompt file (read at runtime)
PROMPT_FILE = Path(__file__).parent / "prompt.txt"


def get_gdpval_agent_options(
    model: str | None = None,
    provider: str | None = None,
    data_dir: str | None = None,
    prompt_file: Path | None = None,
    cwd: str | None = None,
) -> Union[ClaudeAgentOptions, dict]:
    """
    Factory function that creates agent options with the current prompt.

    Args:
        model: Model to use. If None, uses SDK default.
        provider: Provider ID for opencode SDK (e.g., 'gemini', 'arc').
        data_dir: Path to the data directory to add via add_dirs. Ignored when
                  cwd is set (reference files are passed as absolute paths in prompt).
        prompt_file: Path to prompt file to use. If None, uses the default prompt.txt.
        cwd: Working directory for the agent. When set, add_dirs is not used
             (best-effort isolation from project data).
    """
    # Read prompt from disk
    prompt_text = (prompt_file or PROMPT_FILE).read_text().strip()

    if is_opencode_sdk():
        return {
            "system": prompt_text,
            "model_id": model or "gpt-oss-120b",
            "provider_id": provider or "arc",
            "tools": {tool: True for tool in GDPVAL_AGENT_TOOLS},
        }
    else:
        system_prompt = {
            "type": "preset",
            "preset": "claude_code",
            "append": prompt_text
        }

        output_format = {
            "type": "json_schema",
            "schema": AgentResponse.model_json_schema()
        }

        # When cwd is explicitly set, skip add_dirs for isolation —
        # reference files are passed as absolute paths in the prompt.
        add_dirs = []
        if not cwd and data_dir:
            add_dirs.append(data_dir)

        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            output_format=output_format,
            allowed_tools=GDPVAL_AGENT_TOOLS,
            setting_sources=["user", "project"],
            permission_mode='acceptEdits',
            add_dirs=add_dirs,
            cwd=cwd or get_project_root(),
            max_buffer_size=10 * 1024 * 1024,  # 10MB buffer (default is 1MB)
        )

        if model:
            options.model = model

        return options


def make_gdpval_agent_options(model: str | None = None, provider: str | None = None, data_dir: str | None = None, prompt_file: Path | None = None, cwd: str | None = None):
    """Create a factory function for gdpval agent options with a specific model.

    Args:
        model: Model to use (e.g., "opus", "sonnet"). If None, uses SDK default.
        provider: Provider ID for opencode SDK (e.g., 'gemini', 'arc').
        data_dir: Path to the data directory to add (contains reference_files).
        prompt_file: Path to prompt file to use. If None, uses the default prompt.txt.
        cwd: Working directory for the agent (best-effort isolation).

    Returns:
        A callable that returns ClaudeAgentOptions configured with the model and data_dir.
    """
    def factory() -> Union[ClaudeAgentOptions, dict]:
        return get_gdpval_agent_options(model=model, provider=provider, data_dir=data_dir, prompt_file=prompt_file, cwd=cwd)
    return factory


# For backward compatibility, expose the factory as the options
gdpval_agent_options = get_gdpval_agent_options
