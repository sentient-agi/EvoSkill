from __future__ import annotations

from pathlib import Path
from typing import Any

from src.harness import build_options
from src.schemas import AgentResponse


# Closed-book setup: WebFetch / WebSearch are intentionally absent so the
# agent must answer from the bulletin corpus alone. The dataset is being
# curated (see .dataset/officeqa_pro_audited.csv `internet_required` column)
# to questions whose ground truth is image-verified-correct AND derivable
# from the corpus without external scalars. Web tools are also added to
# disallowed_tools below as belt-and-suspenders against a future skill or
# config that re-introduces them at runtime.
OFFICEQA_AGENT_TOOLS = [
    "Read",
    "Write",
    "Bash",
    "Glob",
    "Grep",
    "Edit",
    "TodoWrite",
    "BashOutput",
    "Skill",
]

PROMPT_FILE = Path(__file__).parent / "prompt.md"


def _build_officeqa_agent_options(
    prompt_text: str,
    *,
    model: str | None = None,
    data_dirs: list[str] | None = None,
    project_root: str | Path | None = None,
    thinking: dict | None = None,
    effort: str | None = None,
) -> Any:
    # Scope cwd to the data root so default `Glob("**/*.pdf")` doesn't leak
    # the EvoSkill .venv / tech-report PDFs / trace caches into tool outputs.
    # Skills + scratch in the EvoSkill project remain reachable via add_dirs.
    effective_cwd = project_root
    effective_add_dirs = list(data_dirs) if data_dirs else None
    if data_dirs:
        effective_cwd = data_dirs[0]
        extra = list(data_dirs[1:])
        if project_root is not None:
            extra.append(str(project_root))
        effective_add_dirs = extra or None

    return build_options(
        system=prompt_text,
        schema=AgentResponse.model_json_schema(),
        tools=OFFICEQA_AGENT_TOOLS,
        project_root=effective_cwd,
        model=model,
        data_dirs=effective_add_dirs,
        setting_sources=["user", "project"],
        permission_mode="acceptEdits",
        max_buffer_size=10 * 1024 * 1024,
        # Task is disabled deliberately — when the solver knows how to use
        # the corpus directly (PDF, JSON, txt), it rarely needs to delegate,
        # and subagents inherit a default system prompt that lacks the
        # dataset context. Re-enable if a future skill genuinely needs
        # parallel sub-task dispatch. WebFetch / WebSearch / shell-net
        # access are also blocked here even though they're already absent
        # from OFFICEQA_AGENT_TOOLS — this protects against a future skill
        # or config that adds them back at runtime.
        disallowed_tools=["Task", "WebFetch", "WebSearch"],
        thinking=thinking,
        effort=effort,
    )


def get_officeqa_agent_options(
    model: str | None = None,
    data_dirs: list[str] | None = None,
    project_root: str | Path | None = None,
    thinking: dict | None = None,
    effort: str | None = None,
) -> Any:
    """Read prompt.md fresh on each call so prompt edits take effect without
    restarting the Python process."""
    prompt_text = PROMPT_FILE.read_text().strip()
    return _build_officeqa_agent_options(
        prompt_text, model=model, data_dirs=data_dirs, project_root=project_root,
        thinking=thinking, effort=effort,
    )


def make_officeqa_agent_options_from_task(
    task_description: str,
    model: str | None = None,
    data_dirs: list[str] | None = None,
    project_root: str | Path | None = None,
    thinking: dict | None = None,
    effort: str | None = None,
):
    def factory() -> Any:
        return _build_officeqa_agent_options(
            task_description, model=model, data_dirs=data_dirs, project_root=project_root,
            thinking=thinking, effort=effort,
        )
    return factory


def make_officeqa_agent_options(
    model: str | None = None,
    data_dirs: list[str] | None = None,
    project_root: str | Path | None = None,
    thinking: dict | None = None,
    effort: str | None = None,
):
    def factory() -> Any:
        return get_officeqa_agent_options(
            model=model, data_dirs=data_dirs, project_root=project_root,
            thinking=thinking, effort=effort,
        )
    return factory


officeqa_agent_options = get_officeqa_agent_options
