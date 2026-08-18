from pydantic import BaseModel


class PromptProposerResponse(BaseModel):
    """Response from the prompt proposer agent.

    This proposer analyzes agent failures and proposes prompt modifications
    to improve agent behavior and reasoning.
    """

    proposed_prompt_change: str
    """Description of the prompt modification needed to address the failure."""

    justification: str
    """Explanation of why this prompt change addresses the identified gap."""
