from pydantic import BaseModel

class ToolGeneratorResponse(BaseModel):
    generated_skill: str
    reasoning: str