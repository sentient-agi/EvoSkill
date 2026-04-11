from typing import Any, Union

from src.harness import is_claude_sdk, resolve_project_root
from src.agent_profiles.skill_generator import get_project_root
from src.schemas import AgentResponse


# Use full tool suite for LiveCodeBench (agent can use tools to test/debug)
LIVECODEBENCH_AGENT_TOOLS = [
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

# NOTE: Question formatting (in livecodebench_format.py) matches Artificial Analysis.
# However, we use default Claude Code system prompts and tools for better performance.
# Reference: https://artificialanalysis.ai/benchmarks/livecodebench


def get_livecodebench_agent_options(
    model: str | None = None,
) -> Union[Any, dict[str, Any]]:
    """
    Factory function that creates agent options for LiveCodeBench evaluation.

    Returns ClaudeAgentOptions for Claude SDK or dict for OpenCode SDK.
    Uses default system prompts and full tool access.

    Args:
        model: Model to use (e.g., "opus", "sonnet"). If None, uses SDK default.
    """
    if is_claude_sdk():
        from claude_agent_sdk import ClaudeAgentOptions

        # Use default claude_code preset (no custom append)
        system_prompt = {"type": "preset", "preset": "claude_code"}
        output_format = {
            "type": "json_schema",
            "schema": AgentResponse.model_json_schema(),
        }

        options = ClaudeAgentOptions(
            system_prompt=system_prompt,
            output_format=output_format,
            allowed_tools=LIVECODEBENCH_AGENT_TOOLS,
            setting_sources=["user", "project"],
            permission_mode="acceptEdits",
            cwd=get_project_root(),
            max_buffer_size=10 * 1024 * 1024,  # 10MB buffer (default is 1MB)
        )

        if model:
            options.model = model

        return options
    else:
        # OpenCode SDK - return dict with default system prompt and tools
        return {
            "system": "",  # Use default system prompt
            "format": {
                "type": "json_schema",
                "schema": AgentResponse.model_json_schema(),
            },
            "tools": {tool: True for tool in LIVECODEBENCH_AGENT_TOOLS},
            "mode": "build",
            "model_id": model or "deepseek-ai/DeepSeek-V3",
            "provider_id": "togetherai",
        }


def make_livecodebench_agent_options(model: str | None = None):
    """Create a factory function for LiveCodeBench agent options with a specific model.

    Args:
        model: Model to use (e.g., "opus", "sonnet"). If None, uses SDK default.

    Returns:
        A callable that returns ClaudeAgentOptions configured with the model.
    """

    def factory() -> Union[Any, dict[str, Any]]:
        return get_livecodebench_agent_options(model=model)

    return factory


# For backward compatibility, expose the factory as the options
livecodebench_agent_options = get_livecodebench_agent_options
