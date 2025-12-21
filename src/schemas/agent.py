from pydantic import BaseModel

class AgentResponse(BaseModel):
    final_answer: str
    reasoning: str