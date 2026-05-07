from __future__ import annotations

from pathlib import Path
from typing import Any

from src.harness import build_options
from src.schemas import AgentResponse


# Dataset is curated to be self-contained within the Treasury Bulletin
# corpus. Web tools are disabled to enforce closed-book solving — prevents
# the agent from wasting turns on external lookups when all required data
# is in the provided documents.
SOLVER_TOOLS = [
    "Read",
    "Write",
    "Bash",
    "Glob",
    "Grep",
    "Edit",
    "TodoWrite",
    "BashOutput",
    "Skill",
    "WebFetch",
    "WebSearch",
]

# Path to the prompt file (read at runtime)
PROMPT_FILE = Path(__file__).parent / "prompt.txt"


def _build_solver_options(
    prompt_text: str,
    *,
    model: str | None = None,
    data_dirs: list[str] | None = None,
    project_root: str | Path | None = None,
    thinking: dict | None = None,
    effort: str | None = None,
) -> Any:
    # Scope the solver's cwd to the DATA root when one is provided. Otherwise
    # `Glob("**/*.pdf")` defaults to EvoSkill and leaks .venv/ files, the
    # tech report PDF, trace caches, and other unrelated material into tool
    # outputs. With cwd = data_root, default globs stay inside the document
    # set. The EvoSkill project root (for skills + scratch) is still reachable
    # via add_dirs. Skills are discovered through <data_root>/.claude/skills
    # which is expected to be a symlink into the evolution workspace.
    effective_cwd = project_root
    effective_add_dirs = list(data_dirs) if data_dirs else None
    if data_dirs:
        effective_cwd = data_dirs[0]
        # Keep the rest of data_dirs (if any) plus the original project_root
        # so scratch dirs in EvoSkill remain accessible via absolute paths.
        extra = list(data_dirs[1:])
        if project_root is not None:
            extra.append(str(project_root))
        effective_add_dirs = extra or None

    return build_options(
        system=prompt_text,
        schema=AgentResponse.model_json_schema(),
        tools=SOLVER_TOOLS,
        project_root=effective_cwd,
        model=model,
        data_dirs=effective_add_dirs,
        setting_sources=["user", "project"],
        permission_mode="acceptEdits",
        max_buffer_size=10 * 1024 * 1024,
        thinking=thinking,
        effort=effort,
    )


def get_solver_options(
    model: str | None = None,
    data_dirs: list[str] | None = None,
    project_root: str | Path | None = None,
    thinking: dict | None = None,
    effort: str | None = None,
) -> Any:
    """Factory that creates agent options with the current prompt.

    Reads prompt.txt from disk each time, allowing dynamic updates
    without restarting the Python process.
    """
    prompt_text = PROMPT_FILE.read_text().strip()
    return _build_solver_options(
        prompt_text, model=model, data_dirs=data_dirs,
        project_root=project_root, thinking=thinking, effort=effort,
    )


def make_solver_options_from_task(
    task_description: str,
    model: str | None = None,
    data_dirs: list[str] | None = None,
    project_root: str | Path | None = None,
    thinking: dict | None = None,
    effort: str | None = None,
):
    """Create a factory that uses task_description as the agent system prompt."""
    def factory() -> Any:
        return _build_solver_options(
            task_description, model=model, data_dirs=data_dirs,
            project_root=project_root, thinking=thinking, effort=effort,
        )
    return factory


def make_solver_options(
    model: str | None = None,
    data_dirs: list[str] | None = None,
    project_root: str | Path | None = None,
    thinking: dict | None = None,
    effort: str | None = None,
):
    """Create a factory function for solver options with a specific model."""
    def factory() -> Any:
        return get_solver_options(
            model=model, data_dirs=data_dirs, project_root=project_root,
            thinking=thinking, effort=effort,
        )
    return factory


solver_options = get_solver_options
