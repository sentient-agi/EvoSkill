"""Shared harness helpers.

This module centralizes harness-specific decisions so CLI commands and the
Python API do not drift apart.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from src.agent_profiles.base_agent.base_agent import make_base_agent_options_from_task
from src.agent_profiles.opencode_agent.opencode_agent import (
    make_opencode_agent_options_from_task,
)
from src.agent_profiles.openhands_agent.openhands_agent import (
    make_openhands_agent_options,
)


def build_base_agent_factory(
    *,
    harness: str,
    task_description: str,
    model: str | None = None,
    data_dirs: list[str] | None = None,
) -> Any:
    """Return the harness-appropriate base-agent factory."""
    if harness == "openhands":
        return make_openhands_agent_options(
            task_description=task_description,
            model=model,
            data_dirs=data_dirs,
        )
    if harness == "opencode":
        return make_opencode_agent_options_from_task(
            task_description,
            model=model,
            data_dirs=data_dirs,
        )
    return make_base_agent_options_from_task(
        task_description,
        model=model,
        data_dirs=data_dirs,
    )


def resolve_openhands_llm_config(
    *,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, str]:
    """Resolve OpenHands LLM settings from explicit args or environment.

    Current OpenHands docs use the LiteLLM-style `LLM_*` environment variables.
    We still fall back to legacy envs to avoid breaking existing local setups.
    """
    resolved_model = (
        model
        or os.environ.get("LLM_MODEL")
        or os.environ.get("OPENHANDS_MODEL")
        or "anthropic/claude-sonnet-4-5-20250929"
    )
    resolved_api_key = (
        api_key
        or os.environ.get("LLM_API_KEY")
        or os.environ.get("ANTHROPIC_API_KEY")
        or ""
    )
    resolved_base_url = (
        base_url
        or os.environ.get("LLM_BASE_URL")
        or ""
    )
    return {
        "model": resolved_model,
        "api_key": resolved_api_key,
        "base_url": resolved_base_url,
    }


def get_prompt_artifact_path(harness: str, project_root: str | Path) -> Path:
    """Return the harness-specific prompt artifact used for prompt evolution."""
    root = Path(project_root)
    if harness == "openhands":
        return root / ".evoskill" / "prompts" / "openhands.md"
    return root / "src" / "agent_profiles" / "base_agent" / "prompt.txt"


def load_prompt_text(
    *,
    harness: str,
    project_root: str | Path,
    fallback_text: str,
) -> str:
    """Load the active prompt text for a harness, falling back if unset."""
    prompt_path = get_prompt_artifact_path(harness, project_root)
    if prompt_path.exists():
        return prompt_path.read_text().strip()
    return fallback_text.strip()
