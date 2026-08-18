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
    confidence: float = Field(
        default=50.0,
        description=(
            "Your confidence that the final_answer is correct, from 0 to 100. "
            "0 = pure guess, 100 = mathematically certain. "
            "Base your confidence on: quality of sources found, consistency across sources, "
            "whether the answer required inference vs. direct lookup, and whether you encountered "
            "conflicting information. Be honest — a well-sourced direct lookup might be 90+, "
            "while an inference from partial data might be 30-50."
        )
    )


class FinishedResponse(BaseModel):
    """Minimal final-message contract for ARTIFACT benchmarks (SSB v1/v2,
    Polyglot, SWE-rebench, TB2.1): the answer is the file(s) the solver saved
    on disk; the final message only signals completion. Adopted 2026-07-22 --
    kills the salvage-turn/final-answer-composition failure class. Old runs
    used AgentResponse; the contract is part of the solver definition, so
    never mix contracts within one comparison.
    """
    finished: bool = Field(
        description="true when you have saved your output and are done."
    )

    # Runner-compat properties (NOT fields: they stay out of
    # model_json_schema(), so the schema-derived prompt suffix and every
    # RunCache key are unchanged). The loop runner reads
    # trace.output.final_answer at ~9 sites for logging and text-scorer
    # fallbacks; artifact benchmarks carry no answer text, so these return
    # empty — same contract SSBv2AgentResponse documents with its
    # empty-default fields. Caught live by the 0723 resume smoke: a solver
    # episode that saved no output file skips inline scoring and hits the
    # runner's final_answer fallback, which crashed on the bare schema.
    @property
    def final_answer(self) -> str:
        return ""

    @property
    def reasoning(self) -> str:
        return ""
