from __future__ import annotations

from typing import Any

from src.harness import build_options
from src.schemas import SkillEvolverResponse
from .prompt import SKILL_EVOLVER_SYSTEM_PROMPT


SKILL_EVOLVER_TOOLS = [
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
    "Skill",
]


def get_skill_evolver_options(
    model: str | None = None,
    project_root: str | None = None,
    data_dirs: list[str] | None = None,
    thinking: dict | None = None,
    effort: str | None = None,
    betas: list[str] | None = None,
) -> Any:
    """Factory for the unified skill evolver (proposer + generator in one pass)."""
    return build_options(
        system=SKILL_EVOLVER_SYSTEM_PROMPT.strip(),
        schema=SkillEvolverResponse.model_json_schema(),
        tools=SKILL_EVOLVER_TOOLS,
        project_root=project_root,
        model=model,
        data_dirs=data_dirs,
        setting_sources=["user", "project"],
        permission_mode="bypassPermissions",
        thinking=thinking,
        effort=effort,
        betas=betas,
    )


def make_skill_evolver_options(
    *,
    project_root: str | None = None,
    model: str | None = None,
    data_dirs: list[str] | None = None,
    thinking: dict | None = None,
    effort: str | None = None,
    betas: list[str] | None = None,
):
    return get_skill_evolver_options(
        model=model, project_root=project_root, data_dirs=data_dirs,
        thinking=thinking, effort=effort, betas=betas,
    )


def make_direct_skill_evolver_options(
    *,
    project_root: str | None = None,
    model: str | None = None,
    budget_type: str = "turns",
    budget_value: int | float = 60,
    max_tool_output_chars: int = 100_000,
    compaction_threshold_tokens: int = 150_000,
    enable_thinking: bool = True,
    thinking_effort: str = "xhigh",
    provider_order: list[str] | None = None,
    provider_allow_fallbacks: bool | None = None,
    enable_runtime_switch: bool = False,
    api_key_env: str | None = None,
) -> dict[str, Any]:
    """Factory for a harness-free skill evolver — bash-only tools.

    compaction_threshold_tokens: when cumulative prompt tokens exceed this,
        inject a compaction instruction asking the model to consolidate
        findings before continuing. Set to 0 to disable.
    enable_thinking: enable internal reasoning (DeepSeek thinking mode).
        Default True for evolver — prevents verbose reasoning in content.
    enable_runtime_switch: adds the switch_runtime/exit_runtime tools, which
        let the evolver launch a fresh instance of a SPECIFIC task's own
        docker_image (from its task.toml) and redirect bash there to debug a
        past failure in situ. Built for TB2.1, where (unlike SSB v2's shared-
        image bind-mount setup) the evolver's own container has zero
        filesystem overlap with any task's runtime and that task's actual
        failed container is already destroyed by the time the evolver runs
        -- "switching" only ever means a fresh instance of the same
        environment type, never a resume of the exact failed state. Off by
        default since most benchmarks' evolvers already have adequate
        filesystem access without it (e.g. SSB v2's evolver already gets a
        bind-mounted read-write view of every task's own workspace).
    """
    from src.harness.direct.options import build_direct_options
    from pathlib import Path
    root = Path(project_root) if project_root else Path.cwd()
    skills_dir = str(root / ".claude" / "skills")
    scratch_dir = str(root / ".cache" / "evolver_scratch")
    tools = ["Bash"]
    if enable_runtime_switch:
        tools += ["switch_runtime", "exit_runtime"]
    return build_direct_options(
        system=SKILL_EVOLVER_SYSTEM_PROMPT.strip(),
        schema=SkillEvolverResponse.model_json_schema(),
        tools=tools,
        project_root=project_root,
        model=model,
        inject_skills=False,
        budget_type=budget_type,
        budget_value=budget_value,
        max_tool_output_chars=max_tool_output_chars,
        compaction_threshold_tokens=compaction_threshold_tokens,
        enable_thinking=enable_thinking,
        thinking_effort=thinking_effort,
        allowed_write_dirs=[skills_dir, scratch_dir],
        scratch_dir=scratch_dir,
        provider_order=provider_order,
        # None keeps OpenRouter's default (fallbacks allowed). Pass False to
        # HARD-pin the order -- required when the provider identity is part of
        # the experiment (0722: DSV4 must ride native DeepSeek).
        provider_allow_fallbacks=provider_allow_fallbacks,
        api_key_env=api_key_env,
    )


skill_evolver_options = get_skill_evolver_options()
