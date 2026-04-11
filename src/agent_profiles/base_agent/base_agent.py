from __future__ import annotations

from pathlib import Path
from typing import Any

from src.harness import build_claudecode_options, build_opencode_options, is_claude_sdk
from src.schemas import AgentResponse


BASE_AGENT_TOOLS = [
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

# Path to the prompt file (read at runtime)
PROMPT_FILE = Path(__file__).parent / "prompt.txt"


def _build_base_agent_options(
    prompt_text: str,
    *,
    model: str | None = None,
    data_dirs: list[str] | None = None,
    project_root: str | Path | None = None,
) -> Any:
    if is_claude_sdk():
        return build_claudecode_options(
            system=prompt_text,
            schema=AgentResponse.model_json_schema(),
            tools=BASE_AGENT_TOOLS,
            project_root=project_root,
            model=model,
            data_dirs=data_dirs,
            setting_sources=["user", "project"],
            permission_mode="acceptEdits",
            max_buffer_size=10 * 1024 * 1024,
        )
    return build_opencode_options(
        system=prompt_text,
        schema=AgentResponse.model_json_schema(),
        tools=BASE_AGENT_TOOLS,
        project_root=project_root,
        model=model,
        data_dirs=data_dirs,
    )


def get_base_agent_options(
    model: str | None = None,
    data_dirs: list[str] | None = None,
    project_root: str | Path | None = None,
) -> Any:
    """
    Factory function that creates ClaudeAgentOptions with the current prompt.

    Reads prompt.txt from disk each time, allowing dynamic updates
    without restarting the Python process.

    Args:
        model: Model to use (e.g., "opus", "sonnet"). If None, uses SDK default.
        data_dirs: Extra data directories to mount for the agent (from config harness.data_dirs).
    """
    # Read prompt from disk
    prompt_text = PROMPT_FILE.read_text().strip()

    return _build_base_agent_options(
        prompt_text,
        model=model,
        data_dirs=data_dirs,
        project_root=project_root,
    )


def make_base_agent_options_from_task(
    task_description: str,
    model: str | None = None,
    data_dirs: list[str] | None = None,
    project_root: str | Path | None = None,
):
    """Create a factory that uses task_description as the agent system prompt.

    Args:
        task_description: The task description from task.md (replaces prompt.txt).
        model: Model to use. If None, uses SDK default.
        data_dirs: Extra data directories to mount for the agent.

    Returns:
        A callable that returns ClaudeAgentOptions configured for this task.
    """
    def factory() -> Any:
        return _build_base_agent_options(
            task_description,
            model=model,
            data_dirs=data_dirs,
            project_root=project_root,
        )

    return factory


def make_base_agent_options(
    model: str | None = None,
    data_dirs: list[str] | None = None,
    project_root: str | Path | None = None,
):
    """Create a factory function for base agent options with a specific model.

    Args:
        model: Model to use (e.g., "opus", "sonnet"). If None, uses SDK default.
        data_dirs: Extra data directories to mount for the agent (from config harness.data_dirs).

    Returns:
        A callable that returns ClaudeAgentOptions configured with the model.
    """
    def factory() -> Any:
        return get_base_agent_options(
            model=model,
            data_dirs=data_dirs,
            project_root=project_root,
        )

    return factory


# For backward compatibility, expose the factory as the options
# When passed to Agent, it will be called on each run()
base_agent_options = get_base_agent_options
