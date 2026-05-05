from __future__ import annotations

from pathlib import Path
from typing import Any

from src.harness import build_options
from src.schemas import AgentResponse


# Path to the prompt file (read at runtime)
PROMPT_FILE = Path(__file__).parent / "prompt.txt"

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


def get_livecodebench_agent_options(model: str | None = None) -> Any:
    """Factory that creates agent options for LiveCodeBench evaluation.

    Reads prompt.txt from disk each time, allowing dynamic updates
    without restarting the Python process.
    """
    prompt_text = PROMPT_FILE.read_text().strip()
    return build_options(
        system=prompt_text,
        schema=AgentResponse.model_json_schema(),
        tools=LIVECODEBENCH_AGENT_TOOLS,
        model=model,
        setting_sources=["user", "project"],
        permission_mode="acceptEdits",
        max_buffer_size=10 * 1024 * 1024,
    )


def make_livecodebench_agent_options(model: str | None = None):
    """Create a factory function for LiveCodeBench agent options with a specific model."""
    def factory() -> Any:
        return get_livecodebench_agent_options(model=model)
    return factory


livecodebench_agent_options = get_livecodebench_agent_options
