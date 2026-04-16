from pydantic import BaseModel, Field


class AgentResponse(BaseModel):
    reasoning: str = Field(
        description="Step-by-step reasoning that led to the answer."
    )
    final_answer: str = Field(
        description=(
            "The final answer ONLY. Must be a short, precise value — "
            "a number, a name, a date, or a brief phrase. "
            "Do NOT include units, qualifiers like 'approximately', "
            "or restate the question. Examples: '18', '42.5', 'John Smith', '2024-03-15'."
        )
    )
