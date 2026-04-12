"""Goose SDK option building.

All Goose-specific construction logic lives here.

Key differences from Claude, OpenCode, and Codex:
    - Goose is subprocess-based — no Python SDK to import
    - Model is specified as 'provider/model' (e.g., 'anthropic/claude-sonnet-4-6')
      and split into separate GOOSE_PROVIDER and GOOSE_MODEL env vars
    - A recipe YAML is written to a temp file and passed to goose CLI at runtime
    - Tools are stored as metadata only — Goose has its own built-in toolset
    - No persistent server to manage (unlike OpenCode)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ..utils import resolve_project_root, resolve_data_dirs


# Default model and provider when none is specified.
DEFAULT_GOOSE_MODEL = "claude-sonnet-4-6"
DEFAULT_GOOSE_PROVIDER = "anthropic"


def split_goose_model(model: str | None) -> tuple[str, str]:
    """Split 'provider/model' string or return defaults.

    Examples:
        split_goose_model(None)                    → ("anthropic", "claude-sonnet-4-6")
        split_goose_model("openrouter/gpt-5")      → ("openrouter", "gpt-5")
        split_goose_model("claude-sonnet-4-6")     → ("anthropic", "claude-sonnet-4-6")

    Args:
        model: Model string, optionally prefixed with 'provider/'.

    Returns:
        (provider, model_name) tuple.
    """
    if model is None:
        return DEFAULT_GOOSE_PROVIDER, DEFAULT_GOOSE_MODEL
    if "/" in model:
        provider, model_name = model.split("/", 1)
        return provider, model_name
    return DEFAULT_GOOSE_PROVIDER, model


def build_goose_options(
    *,
    system: str,
    schema: dict[str, Any],
    tools: Iterable[str],
    project_root: str | Path | None = None,
    model: str | None = None,
    data_dirs: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build an options dict for the Goose harness.

    This function has the same signature as build_claudecode_options(),
    build_opencode_options(), and build_codex_options() — it is called by
    build_options() in utils.py when the active SDK is "goose". Agent
    profile factories never call this directly; they all go through
    build_options().

    Args:
        system: System prompt text (from the agent profile's prompt.txt or task.md)
        schema: JSON schema dict for structured output (from Pydantic model_json_schema())
        tools: Tool names from the agent profile (e.g., ["Read", "Write", "Bash"]).
               Stored as metadata only — Goose has its own built-in tools.
        project_root: Path to the project root. Resolved automatically if None.
        model: Model string, optionally prefixed with 'provider/'.
               Defaults to 'anthropic/claude-sonnet-4-6' if None.
        data_dirs: Extra data directories the agent should have access to.
                   If provided, their paths are appended to the system prompt
                   so the agent knows where to look.

    Returns:
        Dict with keys: system, output_schema, provider, model, working_directory,
        tools, data_dirs. This dict is passed to execute_query() in executor.py.
    """
    root = resolve_project_root(project_root)
    resolved_data_dirs = resolve_data_dirs(root, data_dirs)
    provider, model_name = split_goose_model(model)

    # If data directories are provided, append them to the system prompt
    # so the agent knows it can read files from those paths.
    # Same pattern as Codex (see codex/options.py).
    system_with_dirs = system
    if resolved_data_dirs:
        dirs_note = "\n".join(f"- {path}" for path in resolved_data_dirs)
        system_with_dirs = (
            f"{system.rstrip()}\n\n"
            "Additional data directories:\n"
            f"{dirs_note}"
        )

    return {
        # The system prompt — written into the Goose recipe YAML
        "system": system_with_dirs,

        # JSON schema for structured output — embedded in the recipe's response block.
        "output_schema": schema,

        # Provider name — set as GOOSE_PROVIDER env var when invoking the CLI
        "provider": provider,

        # Model name — set as GOOSE_MODEL env var when invoking the CLI
        "model": model_name,

        # Working directory — the Goose CLI subprocess runs in this directory
        "working_directory": str(root),

        # Tool names from the agent profile — stored as metadata for AgentTrace.
        # Goose has its own built-in tools; these are NOT sent to the CLI.
        "tools": list(tools),

        # Resolved absolute paths to extra data directories
        "data_dirs": resolved_data_dirs,
    }
