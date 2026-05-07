"""Shared utilities and the build_options() router.

Contains:
    - resolve_project_root() — find the repo root
    - resolve_data_dirs() — resolve relative data paths
    - build_options() — routes to the active SDK's builder
    - eval_run_label — contextvar callers can set to label the agent.run
      root span (e.g. with a question UID) so traces are findable in Phoenix.
"""

from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Cross-cutting context: optional label for the agent.run root span.
# Set by callers (e.g. evaluate_full) BEFORE invoking agent.run. Read by
# the SDK executors when constructing the per-run span name + attributes.
# Default None → executors fall back to their stock span name.
# ---------------------------------------------------------------------------

eval_run_label: ContextVar[str | None] = ContextVar("eval_run_label", default=None)
eval_run_uid: ContextVar[str | None] = ContextVar("eval_run_uid", default=None)
eval_run_index: ContextVar[int | None] = ContextVar("eval_run_index", default=None)
# Ground truth, set by callers that know the expected answer (evaluate_full).
# Surfaced as the `eval.ground_truth` span attribute so reviewers can compare
# against the agent's final_answer in Phoenix without re-loading the pkl.
eval_run_ground_truth: ContextVar[str | None] = ContextVar("eval_run_ground_truth", default=None)
# Iter-prefix for span names (set by the runner at the top of each iter,
# e.g., "iter 3"). When set, the executor prepends it to top-level run spans
# and per-turn spans so a Phoenix reader can scan thousands of spans across
# iterations without losing track of which iter each one belongs to.
# Default None means no prefix (e.g., during base eval before iter loop starts).
eval_iter_label: ContextVar[str | None] = ContextVar("eval_iter_label", default=None)
# Scorer callback for in-executor scoring of agent results. Signature:
# (question, agent_answer, ground_truth) -> float in [0.0, 1.0]. When set
# AND a ground_truth is also available (via eval_run_ground_truth), the
# executor computes the score after parsing the agent's final_answer,
# stamps it onto the run-span name (e.g. `[OK 0.997]` / `[FAIL 0.000]`),
# and sets eval.score / eval.passed attributes. Lets a Phoenix scan see
# pass/fail at a glance without a separate per-sample eval span.
# Set by callers that have a scorer; default None = no in-span scoring.
eval_score_callback: ContextVar[object | None] = ContextVar(
    "eval_score_callback", default=None
)


# ---------------------------------------------------------------------------
# Shared helpers (imported by claude/options.py and opencode/options.py)
# ---------------------------------------------------------------------------

def resolve_project_root(project_root: str | Path | None = None) -> Path:
    """Resolve the actual project root for the active EvoSkill run."""
    if project_root is not None:
        return Path(project_root).resolve()

    current = Path.cwd().resolve()
    for parent in [current, *current.parents]:
        if (parent / ".evoskill").exists() or (parent / ".git").exists():
            return parent
    return current


def resolve_data_dirs(
    project_root: str | Path | None,
    data_dirs: Iterable[str] | None = None,
) -> list[str]:
    """Resolve relative data directory paths to absolute paths."""
    root = resolve_project_root(project_root)
    resolved: list[str] = []
    for raw in data_dirs or []:
        path = Path(raw)
        resolved.append(str(path if path.is_absolute() else (root / path).resolve()))
    return resolved


# ---------------------------------------------------------------------------
# Router (SDK builders imported lazily inside the function to avoid cycles)
# ---------------------------------------------------------------------------

def build_options(
    *,
    system: str,
    schema: dict[str, Any],
    tools: Iterable[str],
    project_root: str | Path | None = None,
    model: str | None = None,
    data_dirs: Iterable[str] | None = None,
    # Claude-specific extras — silently ignored on other harnesses
    setting_sources: list[str] | None = None,
    permission_mode: str | None = None,
    max_buffer_size: int | None = None,
    disallowed_tools: Iterable[str] | None = None,
    thinking: dict | None = None,
    effort: str | None = None,
) -> Any:
    """Route to the correct builder for the active SDK.

    Claude-specific parameters (setting_sources, permission_mode,
    max_buffer_size, thinking, effort) are forwarded only when the Claude
    SDK is active. They are silently ignored on other harnesses because
    those runtimes have no equivalent concept.
    """
    from .sdk_config import get_sdk

    sdk = get_sdk()

    if sdk == "claude":
        from .claude.options import build_claudecode_options
        return build_claudecode_options(
            system=system,
            schema=schema,
            tools=tools,
            project_root=project_root,
            model=model,
            data_dirs=data_dirs,
            setting_sources=setting_sources,
            permission_mode=permission_mode,
            max_buffer_size=max_buffer_size,
            disallowed_tools=disallowed_tools,
            thinking=thinking,
            effort=effort,
        )
    
    if sdk == "opencode":
        from .opencode.options import build_opencode_options
        return build_opencode_options(
            system=system,
            schema=schema,
            tools=tools,
            project_root=project_root,
            model=model,
            data_dirs=data_dirs,
        )
    
    if sdk == "openhands":
        from .openhands.options import build_openhands_options
        return build_openhands_options(
            system=system,
            schema=schema,
            tools=tools,
            project_root=project_root,
            model=model,
            data_dirs=data_dirs,
        )
     
    if sdk == "codex":
        from .codex.options import build_codex_options
        return build_codex_options(
            system=system,
            schema=schema,
            tools=tools,
            project_root=project_root,
            model=model,
            data_dirs=data_dirs,
        )
    
    if sdk == "goose":
        from .goose.options import build_goose_options
        return build_goose_options(
            system=system,
            schema=schema,
            tools=tools,
            project_root=project_root,
            model=model,
            data_dirs=data_dirs,
        )
    
    raise ValueError(f"Unknown SDK: {sdk!r}")
