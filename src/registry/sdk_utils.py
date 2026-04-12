"""Utilities for converting between ProgramConfig and runtime agent options."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from typing import TYPE_CHECKING

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
    """
    Convert ProgramConfig to ClaudeAgentOptions.

    Args:
        config: The program configuration
        cwd: Working directory for the agent
        add_dirs: Additional directories to add to agent context
        permission_mode: Permission mode for tool execution

    Returns:
        ClaudeAgentOptions ready for use with ClaudeSDKClient
    """
    sdk = config.metadata.get("sdk")
    if sdk == "opencode":
        system_prompt = config.system_prompt
        system_text = system_prompt.get("content") or system_prompt.get("append", "")
        return {
            "system": system_text,
            "tools": {tool: True for tool in config.allowed_tools},
            "format": config.output_format,
            "cwd": cwd,
            "add_dirs": add_dirs or [],
            "mode": config.metadata.get("mode", "build"),
            "provider_id": config.metadata.get("provider_id", "anthropic"),
            "model_id": config.metadata.get("model_id", "claude-sonnet-4-6"),
        }
    if sdk == "openhands":
        system_prompt = config.system_prompt
        system_text = system_prompt.get("content") or system_prompt.get("append", "")
        provider_id = config.metadata.get("provider_id", "anthropic")
        model_id = config.metadata.get("model_id", "claude-sonnet-4-5-20250929")
        model = config.metadata.get("model") or f"{provider_id}/{model_id}"
        return {
            "sdk": "openhands",
            "system": system_text,
            "tools": list(config.allowed_tools),
            "format": config.output_format,
            "cwd": cwd,
            "add_dirs": add_dirs or [],
            "provider_id": provider_id,
            "model_id": model_id,
            "model": model,
            "skills_dir": config.metadata.get("skills_dir") or f"{cwd}/.claude/skills",
        }

    from claude_agent_sdk import ClaudeAgentOptions

    if config.metadata.get("sdk") == "codex":
        system_prompt = config.system_prompt
        system_text = system_prompt.get("content") or system_prompt.get("append", "")
        return {
            "system": system_text,
            "output_schema": config.output_format.get("schema") if config.output_format else {},
            "model": config.metadata.get("model", "codex-mini-latest"),
            "working_directory": cwd,
            "tools": config.allowed_tools,
            "data_dirs": add_dirs or [],
        }

    if config.metadata.get("sdk") == "goose":
        system_prompt = config.system_prompt
        system_text = system_prompt.get("content") or system_prompt.get("append", "")
        provider = config.metadata.get("provider", "anthropic")
        model = config.metadata.get("model", "claude-sonnet-4-6")
        return {
            "system": system_text,
            "output_schema": config.output_format.get("schema") if config.output_format else {},
            "provider": provider,
            "model": model,
            "working_directory": cwd,
            "tools": config.allowed_tools,
            "data_dirs": add_dirs or [],
        }

    return ClaudeAgentOptions(
        system_prompt=config.system_prompt,
        allowed_tools=config.allowed_tools,
        output_format=config.output_format,
        setting_sources=["user", "project"],  # Load skills from .claude/skills/
        permission_mode=permission_mode,
        add_dirs=add_dirs or [],
        cwd=cwd,
    )


def options_to_config(
    options: ClaudeAgentOptions | dict[str, Any],
    name: str,
    *,
    parent: str | None = None,
    generation: int = 0,
    metadata: dict[str, Any] | None = None,
) -> ProgramConfig:
    """
    Convert ClaudeAgentOptions to ProgramConfig.

    Args:
        options: The agent options to convert
        name: Name for the program
        parent: Parent program reference (e.g., 'program/base')
        generation: Number of mutations from base
        metadata: Additional metadata to include

    Returns:
        ProgramConfig ready for registration
    """
    base_metadata = {"created_at": datetime.now().isoformat()}
    if metadata:
        base_metadata.update(metadata)

    if isinstance(options, dict):
        sdk = options.get("sdk", "opencode")
        system_text = options.get("system", "")
        tools = options.get("tools", {})
        if isinstance(tools, dict):
            allowed_tools = list(tools.keys())
        else:
            allowed_tools = list(tools or [])

        system_prompt = (
            system_text
            if isinstance(system_text, dict)
            else {"type": "text", "content": system_text}
        )

        # Detect SDK type:
        #   - goose options have output_schema + working_directory + provider
        #   - codex options have output_schema + working_directory (no provider)
        #   - opencode options have format + cwd
        if "output_schema" in options and "working_directory" in options and "provider" in options:
            sdk_type = "goose"
        elif "output_schema" in options and "working_directory" in options:
            sdk_type = "codex"
        else:
            sdk_type = "opencode"

        if sdk_type == "goose":
            base_metadata.update(
                {
                    "sdk": "goose",
                    "provider": options.get("provider", "anthropic"),
                    "model": options.get("model", "claude-sonnet-4-6"),
                    "working_directory": options.get("working_directory"),
                }
            )
            return ProgramConfig(
                name=name,
                parent=parent,
                generation=generation,
                system_prompt=system_prompt,
                allowed_tools=allowed_tools,
                output_format={"schema": options.get("output_schema", {})},
                metadata=base_metadata,
            )

        if sdk_type == "codex":
            base_metadata.update(
                {
                    "sdk": "codex",
                    "model": options.get("model", "codex-mini-latest"),
                    "working_directory": options.get("working_directory"),
                }
            )
            return ProgramConfig(
                name=name,
                parent=parent,
                generation=generation,
                system_prompt=system_prompt,
                allowed_tools=allowed_tools,
                output_format={"schema": options.get("output_schema", {})},
                metadata=base_metadata,
            )

        base_metadata.update(
            {
                "sdk": sdk,
                "provider_id": options.get("provider_id"),
                "model_id": options.get("model_id"),
                "cwd": options.get("cwd"),
            }
        )
        if sdk == "opencode":
            base_metadata["mode"] = options.get("mode", "build")
        if sdk == "openhands":
            base_metadata["model"] = options.get("model")
            base_metadata["skills_dir"] = options.get("skills_dir")

        return ProgramConfig(
            name=name,
            parent=parent,
            generation=generation,
            system_prompt=system_prompt,
            allowed_tools=allowed_tools,
            output_format=options.get("format"),
            metadata=base_metadata,
        )

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
    """
    Create a modified system prompt by appending/prepending content.

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
    """
    Create a new config with additional tools.

    Args:
        config: Base program configuration
        tools: Tools to add

    Returns:
        New ProgramConfig with additional tools
    """
    new_tools = list(set(config.allowed_tools + tools))
    return config.model_copy(update={"allowed_tools": new_tools})


def remove_tools(config: ProgramConfig, tools: list[str]) -> ProgramConfig:
    """
    Create a new config with tools removed.

    Args:
        config: Base program configuration
        tools: Tools to remove

    Returns:
        New ProgramConfig without specified tools
    """
    new_tools = [t for t in config.allowed_tools if t not in tools]
    return config.model_copy(update={"allowed_tools": new_tools})
