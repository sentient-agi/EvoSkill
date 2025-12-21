"""
Program configuration models for the agent registry.

A Program represents a specific agent configuration (prompts + tools)
that can be versioned and tracked via git branches.
"""

from pydantic import BaseModel, Field
from typing import Any
from datetime import datetime


class ProgramConfig(BaseModel):
    """
    Program configuration stored in .claude/program.yaml

    Each program represents a distinct agent configuration that includes
    system prompts, allowed tools, and output format specifications.
    """

    name: str = Field(description="Program name (used for branch naming)")
    parent: str | None = Field(
        default=None, description="Parent program branch (e.g., 'program/base')"
    )
    generation: int = Field(
        default=0, description="Number of mutations from base program"
    )

    # Agent configuration
    system_prompt: dict[str, Any] = Field(
        description="System prompt configuration for ClaudeAgentOptions"
    )
    allowed_tools: list[str] = Field(
        default_factory=list, description="List of allowed tool names"
    )
    output_format: dict[str, Any] | None = Field(
        default=None, description="Output format specification (e.g., JSON schema)"
    )

    # Metadata for tracking
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (scores, notes, timestamps)",
    )

    def with_metadata(self, **kwargs: Any) -> "ProgramConfig":
        """Return a copy with updated metadata."""
        new_metadata = {**self.metadata, **kwargs}
        return self.model_copy(update={"metadata": new_metadata})

    def with_timestamp(self) -> "ProgramConfig":
        """Return a copy with current timestamp in metadata."""
        return self.with_metadata(created_at=datetime.now().isoformat())

    def mutate(
        self,
        name: str,
        *,
        system_prompt: dict[str, Any] | None = None,
        allowed_tools: list[str] | None = None,
        output_format: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ProgramConfig":
        """
        Create a child program with mutations.

        Args:
            name: Name for the new program
            system_prompt: New system prompt (or inherit from parent)
            allowed_tools: New tools list (or inherit from parent)
            output_format: New output format (or inherit from parent)
            metadata: Additional metadata for the child

        Returns:
            A new ProgramConfig with this program as parent
        """
        return ProgramConfig(
            name=name,
            parent=f"program/{self.name}",
            generation=self.generation + 1,
            system_prompt=system_prompt or self.system_prompt,
            allowed_tools=allowed_tools or self.allowed_tools,
            output_format=output_format if output_format is not None else self.output_format,
            metadata=metadata or {},
        ).with_timestamp()
