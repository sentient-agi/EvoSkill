"""Skill-evolver system prompt — content lives in `prompt.md` for easier
editing. We read the .md file at import time and expose it as the legacy
`SKILL_EVOLVER_SYSTEM_PROMPT` string so callers don't change."""
from pathlib import Path

_PROMPT_PATH = Path(__file__).parent / "prompt.md"
SKILL_EVOLVER_SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")
