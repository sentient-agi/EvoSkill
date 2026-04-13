"""Codex SDK option building.

All Codex-specific construction logic lives here.

Key differences from Claude and OpenCode:
    - Codex uses `output_schema` directly (a plain JSON schema dict),
      not wrapped in {"type": "json_schema", "schema": ...} like OpenCode
    - Codex uses `working_directory` (not `cwd` like Claude/OpenCode)
    - Codex tools are built-in (file read/write, bash, etc.) — they can't
      be configured per-request like Claude's `allowed_tools`. We store
      the tools list as metadata so AgentTrace can report what was available.
    - No tool name mapping needed (unlike OpenCode which maps "Read" → "read")
    - No server/permission management needed (unlike OpenCode which manages
      a local HTTP server and writes opencode.json for file access)
    - Model names are passed directly ("codex-mini-latest", "o3", "gpt-5")
      with no provider prefix (unlike OpenCode's "anthropic/claude-sonnet-4-6")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ..utils import resolve_project_root, resolve_data_dirs


# Default model when none is specified. Codex-mini-latest is optimized
# for use with the Codex CLI and is the cheapest option.
DEFAULT_CODEX_MODEL = "codex-mini-latest"


def _make_openai_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert a Pydantic JSON schema to OpenAI strict structured output format.

    OpenAI's Responses API requires:
        - "additionalProperties": false at the top level
        - "required" must list ALL property keys (no optional fields)

    Pydantic's model_json_schema() only puts truly required fields in "required",
    but OpenAI demands every property is listed. Fields with defaults still work
    because the model will always produce them.
    """
    strict = {**schema, "additionalProperties": False}
    if "properties" in strict:
        strict["required"] = list(strict["properties"].keys())
    return strict


def build_codex_options(
    *,
    system: str,
    schema: dict[str, Any],
    tools: Iterable[str],
    project_root: str | Path | None = None,
    model: str | None = None,
    data_dirs: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build an options dict for the Codex SDK.

    This function has the same signature as build_claudecode_options() and
    build_opencode_options() — it's called by build_options() in utils.py
    when the active SDK is "codex". Agent profile factories never call this
    directly; they all go through build_options().

    Args:
        system: System prompt text (from the agent profile's prompt.txt or task.md)
        schema: JSON schema dict for structured output (from Pydantic model_json_schema())
        tools: Tool names from the agent profile (e.g., ["Read", "Write", "Bash"]).
               Stored as metadata only — Codex has its own built-in tools.
        project_root: Path to the project root. Resolved automatically if None.
        model: Codex model name (e.g., "codex-mini-latest", "o3", "gpt-5").
               Defaults to DEFAULT_CODEX_MODEL if None.
        data_dirs: Extra data directories the agent should have access to.
                   If provided, their paths are appended to the system prompt
                   so the agent knows where to look.

    Returns:
        Dict with keys: system, output_schema, model, working_directory, tools, data_dirs.
        This dict is passed to execute_query() in executor.py.
    """
    root = resolve_project_root(project_root)
    resolved_data_dirs = resolve_data_dirs(root, data_dirs)

    # If data directories are provided, append them to the system prompt
    # so the agent knows it can read files from those paths.
    # Same pattern as OpenCode (see opencode/options.py).
    system_with_dirs = system
    if resolved_data_dirs:
        dirs_note = "\n".join(f"- {path}" for path in resolved_data_dirs)
        system_with_dirs = (
            f"{system.rstrip()}\n\n"
            "Additional data directories:\n"
            f"{dirs_note}"
        )

    return {
        # The system prompt — sent to the Codex thread
        "system": system_with_dirs,

        # JSON schema for structured output — passed to thread.run() as output_schema.
        # Unlike OpenCode which wraps it in {"type": "json_schema", "schema": ...},
        # Codex takes the raw schema dict directly.
        # OpenAI's Responses API requires:
        #   - "additionalProperties": false
        #   - "required" must list ALL property keys (no optional fields allowed)
        "output_schema": _make_openai_strict_schema(schema),

        # Model name — passed to Codex thread configuration
        "model": model or DEFAULT_CODEX_MODEL,

        # Working directory — the Codex agent operates within this directory.
        # Set at thread start time, not per-query.
        "working_directory": str(root),

        # Tool names from the agent profile — stored as metadata for AgentTrace.
        # Codex has its own built-in tools; these are NOT sent to the SDK.
        "tools": list(tools),

        # Resolved absolute paths to extra data directories
        "data_dirs": resolved_data_dirs,
    }
