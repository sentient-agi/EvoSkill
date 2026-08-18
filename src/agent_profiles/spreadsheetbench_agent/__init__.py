"""SpreadsheetBench agent profile: solver prompt + scorer."""

from pathlib import Path

_PROMPT_PATH = Path(__file__).parent / "prompt.md"


def get_solver_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8").strip()
