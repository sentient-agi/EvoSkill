"""Claude SDK option building.

All Claude-specific construction logic lives here:
    - ClaudeAgentOptions assembly (system_prompt preset, output_format, tools)
    - Optional extras (setting_sources, permission_mode, max_buffer_size)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ..utils import resolve_project_root, resolve_data_dirs


def build_claudecode_options(
    *,
    system: str,
    schema: dict[str, Any],
    tools: Iterable[str],
    project_root: str | Path | None = None,
    model: str | None = None,
    data_dirs: Iterable[str] | None = None,
    setting_sources: list[str] | None = None,
    permission_mode: str | None = None,
    max_buffer_size: int | None = None,
) -> Any:
    """Build ClaudeAgentOptions for the Claude SDK."""
    from claude_agent_sdk import ClaudeAgentOptions

    root = resolve_project_root(project_root)
    system_prompt: dict[str, Any] = {
        "type": "preset",
        "preset": "claude_code",
    }
    if system:
        system_prompt["append"] = system

    kwargs: dict[str, Any] = {
        "system_prompt": system_prompt,
        "output_format": {"type": "json_schema", "schema": schema},
        "allowed_tools": list(tools),
        "cwd": str(root),
    }
    if setting_sources is not None:
        kwargs["setting_sources"] = setting_sources
    if permission_mode is not None:
        kwargs["permission_mode"] = permission_mode
    if max_buffer_size is not None:
        kwargs["max_buffer_size"] = max_buffer_size
    if data_dirs is not None:
        kwargs["add_dirs"] = resolve_data_dirs(root, data_dirs)

    options = ClaudeAgentOptions(**kwargs)
    if model:
        options.model = model
    return options
