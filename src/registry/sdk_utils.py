"""
Utilities for converting between ProgramConfig and ClaudeAgentOptions.

These helpers allow seamless integration between the program registry
and the Claude Agent SDK.
"""

from datetime import datetime
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions

from .models import ProgramConfig


def config_to_options(
    config: ProgramConfig,
    cwd: str,
    *,
    add_dirs: list[Any] | None = None,
    permission_mode: str = "acceptEdits",
) -> ClaudeAgentOptions:
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
    options: ClaudeAgentOptions,
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
