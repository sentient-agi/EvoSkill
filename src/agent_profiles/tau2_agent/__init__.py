"""tau2-bench agent profile for EvoSkill.

tau2-bench (sierra-research/tau2-bench, pinned at tag v1.0.0 / commit
17e07b1) is driven via a subprocess worker (`_tau2_worker.py`) run inside
its own isolated venv at `third_party/tau2-bench/.venv` -- see
`third_party/README.md` for provenance and `solver.py`'s module docstring
for why this integration does not merge tau2 into EvoSkill's own venv.
"""
from pathlib import Path

_PROMPT_PATH = Path(__file__).parent / "prompt.md"


def get_solver_prompt() -> str:
    return _PROMPT_PATH.read_text(encoding="utf-8").strip()
