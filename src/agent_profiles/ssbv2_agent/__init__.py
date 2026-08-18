"""SpreadsheetBench v2 agent profile."""

from pathlib import Path

from pydantic import BaseModel, Field

_PROMPT_PATH = Path(__file__).parent / "prompt.md"

def get_solver_prompt() -> str:
    return _PROMPT_PATH.read_text()


class SSBv2AgentResponse(BaseModel):
    """Completion signal for the SSB v2 solver.

    Scoring is entirely file-based (score_and_diff compares the saved
    output workbook against golden) — final_answer/reasoning carry no
    scoring signal here, unlike text-answer benchmarks. Kept as optional
    empty-default fields (not removed) so shared runner.py logging that
    reads trace.output.final_answer doesn't break; the model itself is
    only asked for `finished`+`confidence`.
    """
    finished: bool = Field(
        default=True,
        description=(
            "Set true once you have verified your edits against the "
            "instruction and saved the output file to ./output/."
        ),
    )
    confidence: float = Field(
        default=50.0,
        description="Your confidence (0-100) that the saved output is fully correct.",
    )
    reasoning: str = Field(default="", description="Unused for this benchmark.")
    final_answer: str = Field(default="", description="Unused for this benchmark.")
