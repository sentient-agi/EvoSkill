"""
Agent Program Registry - Git-based versioning for self-improving agents.

This module provides tools to track and manage different agent program versions
(configurations of prompts + tools) using git branches.

Example usage:
    from src.registry import ProgramManager, ProgramConfig

    manager = ProgramManager()

    # Create base program
    base_config = ProgramConfig(
        name="base",
        parent=None,
        generation=0,
        system_prompt={"type": "preset", "preset": "claude_code"},
        allowed_tools=["Read", "Write", "Bash"],
        output_format=None,
        metadata={}
    )
    manager.create_program("base", base_config)
    manager.mark_frontier("base")

    # Create a mutation
    mutation = base_config.mutate(
        "iter-1",
        allowed_tools=base_config.allowed_tools + ["WebSearch"]
    )
    manager.create_program("iter-1", mutation, parent="base")

    # Switch between programs
    manager.switch_to("base")
    manager.switch_to("iter-1")

    # Get lineage
    lineage = manager.get_lineage("iter-1")  # ["iter-1", "base"]
"""

from .models import ProgramConfig
from .manager import ProgramManager, ProgramManagerError
from .sdk_utils import (
    config_to_options,
    options_to_config,
    merge_system_prompt,
    add_tools,
    remove_tools,
)

__all__ = [
    # Core classes
    "ProgramConfig",
    "ProgramManager",
    "ProgramManagerError",
    # SDK utilities
    "config_to_options",
    "options_to_config",
    "merge_system_prompt",
    "add_tools",
    "remove_tools",
]
