from typing import Literal, Optional
from pydantic import BaseModel


class SkillEvolverResponse(BaseModel):
    """Response from the unified skill evolver agent.

    This agent both analyzes failures AND implements the skill fix
    in a single pass — no proposer/generator handoff.
    """

    action: Literal["create", "edit"]
    """Whether a new skill was created or an existing one was edited."""

    skill_name: str
    """Name of the skill that was created or edited."""

    description: str
    """What the skill does and why it addresses the failure."""

    justification: str
    """Root cause analysis with references to specific trace moments."""
