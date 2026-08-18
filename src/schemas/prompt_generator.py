from pydantic import BaseModel


class PromptGeneratorResponse(BaseModel):
    optimized_prompt: str
    reasoning: str
