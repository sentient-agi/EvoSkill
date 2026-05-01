"""Utilities for converting between ProgramConfig and runtime agent options."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, TYPE_CHECKING

from .models import ProgramConfig

if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions
else:
    ClaudeAgentOptions = Any


def config_to_options(
    config: ProgramConfig,
    cwd: str,
    *,
    add_dirs: list[Any] | None = None,
    permission_mode: str = "acceptEdits",
) -> ClaudeAgentOptions | dict[str, Any]:
    """Convert a saved ProgramConfig back to runtime agent options.

    Extracts system prompt, schema, tools, and model from the config,
    then delegates to build_options() which routes to the correct harness
    builder. This ensures config_to_options() always produces the same
    output shape as the original build_*_options() functions.

    Args:
        config: The program configuration (loaded from .claude/program.yaml)
        cwd: Working directory for the agent
        add_dirs: Additional directories to add to agent context
        permission_mode: Permission mode for tool execution (Claude-specific)

    Returns:
        Harness-specific options (ClaudeAgentOptions or dict) ready for Agent()
    """
    from src.harness import build_options, set_sdk, get_sdk

    # Headless sandbox: bypass all permission prompts since no one can approve.
    if os.environ.get("EVOSKILL_REMOTE") == "1":
        permission_mode = "bypassPermissions"

    # Extract system prompt text — handles both str and dict formats
    system_prompt = config.system_prompt or ""
    if isinstance(system_prompt, str):
        system_text = system_prompt
    else:
        system_text = system_prompt.get("content") or system_prompt.get("append", "")

    # Extract schema from stored output_format
    # Codex/Goose store as {"schema": {...}}, Claude/OpenCode store the full format
    schema = {}
    if config.output_format:
        schema = config.output_format.get("schema", config.output_format)

    # Extract model — stored differently per harness
    sdk = config.metadata.get("sdk") or "claude"
    model = config.metadata.get("model") or config.metadata.get("model_id")

    # Goose stores provider and model separately — recombine as "provider/model"
    if sdk == "goose" and config.metadata.get("provider"):
        model = f"{config.metadata['provider']}/{model}"

    # OpenCode stores provider_id and model_id separately — recombine
    if sdk == "opencode":
        provider_id = config.metadata.get("provider_id", "anthropic")
        model_id = config.metadata.get("model_id", "claude-sonnet-4-6")
        model = f"{provider_id}/{model_id}"

    # OpenHands stores provider_id and model_id separately — recombine
    if sdk == "openhands":
        provider_id = config.metadata.get("provider_id", "anthropic")
        model_id = config.metadata.get("model_id", "claude-sonnet-4-5-20250929")
        model = config.metadata.get("model") or f"{provider_id}/{model_id}"

    # Temporarily switch SDK so build_options() routes to the correct builder,
    # then restore the original SDK afterwards
    original_sdk = get_sdk()
    try:
        set_sdk(sdk)
        return build_options(
            system=system_text,
            schema=schema,
            tools=config.allowed_tools or [],
            project_root=cwd,
            model=model,
            data_dirs=add_dirs,
            # Claude-specific extras (silently ignored by other harnesses)
            setting_sources=["user", "project"],
            permission_mode=permission_mode,
        )
    finally:
        set_sdk(original_sdk)


def options_to_config(
    options: ClaudeAgentOptions | dict[str, Any],
    name: str,
    *,
    parent: str | None = None,
    generation: int = 0,
    metadata: dict[str, Any] | None = None,
) -> ProgramConfig:
    """Convert runtime agent options to a saveable ProgramConfig.

    Detects which harness produced the options (via explicit "sdk" key or
    heuristic), extracts relevant fields, and packs them into a ProgramConfig
    with metadata tracking the SDK type for round-tripping.

    Args:
        options: The agent options to convert (ClaudeAgentOptions or dict)
        name: Name for the program (e.g., "base", "iter-skill-1")
        parent: Parent program reference (e.g., "program/base")
        generation: Number of mutations from base
        metadata: Additional metadata to include

    Returns:
        ProgramConfig ready for git storage
    """
    base_metadata = {"created_at": datetime.now().isoformat()}
    if metadata:
        base_metadata.update(metadata)

    # Goose / Openhands / Opencode / Goose
    if isinstance(options, dict):

        # PrgConf : system prompt — store as plain string when possible
        system_text = options.get("system", "")
        system_prompt = system_text if isinstance(system_text, (str, dict)) else str(system_text)

        # PrgConf : allowed tools 
        tools = options.get("tools", {})
        if isinstance(tools, dict):
            allowed_tools = list(tools.keys())
        else:
            allowed_tools = list(tools or [])

        # Use the global SDK setting to determine which harness produced these options
        from src.harness import get_sdk
        sdk = options.get("sdk") or get_sdk()
        # Each harness stores different metadata and uses different output_format keys
        output_format = None

        if sdk == "opencode":
            base_metadata.update({
                "sdk": "opencode",
                "mode": options.get("mode", "build"),
                "provider_id": options.get("provider_id"),
                "model_id": options.get("model_id"),
                "cwd": options.get("cwd"),
            })
            output_format = options.get("format")

        elif sdk == "openhands":
            base_metadata.update({
                "sdk": "openhands",
                "provider_id": options.get("provider_id"),
                "model_id": options.get("model_id"),
                "model": options.get("model"),
                "cwd": options.get("cwd"),
                "skills_dir": options.get("skills_dir"),
            })
            output_format = options.get("format")

        elif sdk == "codex":
            base_metadata.update({
                "sdk": "codex",
                "model": options.get("model", "codex-mini-latest"),
                "working_directory": options.get("working_directory"),
            })
            output_format = {"schema": options.get("output_schema", {})}

        elif sdk == "goose":
            base_metadata.update({
                "sdk": "goose",
                "provider": options.get("provider", "anthropic"),
                "model": options.get("model", "claude-sonnet-4-6"),
                "working_directory": options.get("working_directory"),
            })
            output_format = {"schema": options.get("output_schema", {})}

        return ProgramConfig(
            name=name,
            parent=parent,
            generation=generation,
            system_prompt=system_prompt,
            allowed_tools=allowed_tools,
            output_format=output_format,
            metadata=base_metadata,
        )

    # Clause Code
    base_metadata.setdefault("sdk", "claude")
    return ProgramConfig(
        name=name,
        parent=parent,
        generation=generation,
        system_prompt=options.system_prompt or {},
        allowed_tools=options.allowed_tools or [],
        output_format=options.output_format,
        metadata=base_metadata,
    )


def merge_system_prompt(
    base: dict[str, Any],
    *,
    append: str | None = None,
    prepend: str | None = None,
) -> dict[str, Any]:
    """Create a modified system prompt by appending/prepending content.

    Args:
        base: Base system prompt configuration
        append: Text to append to the prompt
        prepend: Text to prepend to the prompt

    Returns:
        New system prompt dict with modifications
    """
    result = dict(base)

    if append:
        existing_append = result.get("append", "")
        if existing_append:
            result["append"] = f"{existing_append}\n\n{append}"
        else:
            result["append"] = append

    if prepend:
        existing_append = result.get("append", "")
        if existing_append:
            result["append"] = f"{prepend}\n\n{existing_append}"
        else:
            result["append"] = prepend

    return result


def add_tools(config: ProgramConfig, tools: list[str]) -> ProgramConfig:
    """Create a new config with additional tools."""
    new_tools = list(set(config.allowed_tools + tools))
    return config.model_copy(update={"allowed_tools": new_tools})


def remove_tools(config: ProgramConfig, tools: list[str]) -> ProgramConfig:
    """Create a new config with tools removed."""
    new_tools = [t for t in config.allowed_tools if t not in tools]
    return config.model_copy(update={"allowed_tools": new_tools})
