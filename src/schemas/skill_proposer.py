from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SkillProposerResponse(BaseModel):
    """Response from the skill proposer agent.

    This proposer analyzes agent failures and proposes skill additions
    or modifications to existing skills to address capability gaps.
    """

    action: Literal["create", "edit"] = "create"
    """Whether to create a new skill or edit an existing one."""

    target_skill: str | None = None
    """Name of existing skill to modify. Required if action="edit"."""

    proposed_skill: str
    """High-level description of the skill needed or modifications to make."""

    justification: str
    """Explanation of why this skill/modification addresses the identified gap."""

    related_iterations: list[str] = Field(default_factory=list)
    """List of relevant past iterations referenced in the proposal (e.g., ["iter-4", "iter-9"])."""

    @model_validator(mode="after")
    def validate_edit_target(self) -> "SkillProposerResponse":
        if self.action == "edit" and not self.target_skill:
            raise ValueError("target_skill is required when action='edit'")
        return self
