from typing import Literal
from pydantic import BaseModel


class ProposerResponse(BaseModel):
    optimize_prompt_or_skill: Literal["prompt", "skill"]
    proposed_skill_or_prompt: str
    justification: str
