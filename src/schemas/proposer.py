from pydantic import BaseModel

class ProposerResponse(BaseModel):
    proposed_tool_or_skill: str
    justification: str