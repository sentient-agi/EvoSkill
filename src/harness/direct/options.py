"""Direct harness option building.

Maps Claude-style PascalCase tool names to the direct executor's
snake_case names, resolves model/provider strings, and builds the
options dict consumed by executor.execute_query().
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable

from ..model_aliases import normalize_harness_model
from ..utils import resolve_project_root, resolve_data_dirs

# Default model when none specified (matches opencode default).
DEFAULT_DIRECT_MODEL = "deepseek/deepseek-v4-flash"

CLAUDE_TO_DIRECT_TOOL: dict[str, str | None] = {
    "Read": "read_file",
    "Write": "write_file",
    "Bash": "bash",
    "Glob": "glob",
    "Grep": "grep",
    "Edit": "edit_file",
    "BashOutput": None,
    "WebFetch": None,
    "WebSearch": None,
    "TodoWrite": None,
    "Skill": "skill",
}


def split_direct_model(model: str | None) -> tuple[str, str]:
    """Parse 'provider/model' into (provider_id, model_id).

    Returns the full 'provider/model' string as model_id (OpenRouter
    expects the slash-prefixed form). provider_id is always "openrouter"
    since all models are accessed through OpenRouter's unified API.
    """
    full = model.strip() if model and model.strip() else DEFAULT_DIRECT_MODEL
    return "openrouter", full


def to_direct_tools(tools: Iterable[str]) -> dict[str, bool]:
    """Map Claude PascalCase tool names to direct executor names."""
    converted: dict[str, bool] = {}
    for tool in tools:
        normalized = CLAUDE_TO_DIRECT_TOOL.get(tool, tool.lower())
        if normalized is not None:
            converted[normalized] = True
    return converted


def is_deepseek_model(model: str | None) -> bool:
    """True for the DeepSeek model family — the ONLY family the pin applies to."""
    return bool(model) and "deepseek" in str(model).lower()


def apply_provider_pin(
    opts: dict[str, Any],
    model: str | None = None,
    env_var: str = "EVOSKILL_SOLVER_ALLOW_FALLBACKS",
) -> dict[str, Any]:
    """Turn `provider_order` from a PREFERENCE into a HARD PIN, in place.

    `order` alone only RANKS providers: OpenRouter reroutes freely to any other
    host when the preferred one is busy or erroring, and third-party hosts of
    the same model produce different outputs.

    **DEEPSEEK FAMILY ONLY.** The standing directive is narrow: DeepSeek models
    (flash included) run order=["deepseek"] AND allow_fallbacks=False. It does
    NOT generalise. GLM 5.2 and friends are deliberately served by the `:nitro`
    multi-provider pool, and hard-pinning them would strip the fallback those
    paths rely on. The gating matters because the launcher scripts `export`
    EVOSKILL_*_ALLOW_FALLBACKS process-wide, so a run with a DeepSeek solver and
    a GLM evolver/judge would otherwise pin the GLM calls too.

    Lives here, in the shared options module, because the same defect appeared
    in several agent-local copies. Agent profiles routinely build options in TWO
    places — an outer `_get_options`/`_make_opts` and an inner one inside
    `run()` — and on 2026-08-01 SSB v2 was found with the hard pin applied only
    to the outer dict, which never executes a query, so `allow_fallbacks` was
    never sent at all.

    `model` defaults to the dict's own `model_id`. Unset env leaves the key
    absent (preference-only, unchanged), so this is opt-in and safe anywhere.
    """
    if not opts.get("provider_order"):
        return opts
    if not is_deepseek_model(model if model is not None else opts.get("model_id")):
        return opts
    fb = os.environ.get(env_var)
    if fb is not None:
        opts["provider_allow_fallbacks"] = fb == "1"
    return opts


def pin_request_body_provider(
    body: dict[str, Any],
    provider_order: list[str] | None,
    env_var: str = "EVOSKILL_SOLVER_ALLOW_FALLBACKS",
    hard: bool = False,
) -> dict[str, Any]:
    """Same rule as `apply_provider_pin`, for hand-rolled OpenRouter bodies.

    Several agents (ALFWorld, TB2's Harbor agent, the tau2 judge) and several
    evolver scripts POST to /chat/completions directly instead of going through
    `build_direct_options`, so they each grew their own
    `body["provider"] = {"order": ...}` line — and every one of them omitted
    `allow_fallbacks`, exactly like the SSB v2 defect found on 2026-08-01.

    Reads the family from `body["model"]`, so a GLM judge sharing a process with
    a DeepSeek solver keeps its multi-provider routing even though the launcher
    exported the env var globally.

    No-op when `provider_order` is falsy, so callers keep their `if` semantics.
    Unset env leaves the pin off (preference-only, unchanged).

    `hard=True` applies the allow_fallbacks half to a NON-DeepSeek model. Needed because
    the DeepSeek-only gate below made this function structurally incapable of hard-pinning
    any other family: it would write `order` and stop, which is precisely the
    preference-only pin this docstring calls a defect -- OpenRouter may still reroute, and
    every log still says "pinned". That is fine as a DEFAULT (GLM 5.2 on :nitro must keep
    multi-provider routing, and silently pinning it would defeat the routing it is chosen
    for), but it must be reachable on request, or cross-family work cannot control which
    endpoint serves it. Opt-in only: callers that do not pass it behave exactly as before.
    """
    if not provider_order:
        return body
    body["provider"] = {"order": list(provider_order)}
    if not (hard or is_deepseek_model(body.get("model"))):
        return body
    fb = os.environ.get(env_var)
    if fb is not None:
        body["provider"]["allow_fallbacks"] = fb == "1"
    return body


def build_direct_options(
    *,
    system: str,
    schema: dict[str, Any],
    tools: Iterable[str],
    project_root: str | Path | None = None,
    model: str | None = None,
    data_dirs: Iterable[str] | None = None,
    budget_type: str = "turns",
    budget_value: int | float = 40,
    max_tool_output_chars: int = 100_000,
    compaction_threshold_tokens: int = 0,
    enable_thinking: bool = True,
    thinking_effort: str = "",
    inject_skills: bool = True,
    allowed_write_dirs: list[str] | None = None,
    scratch_dir: str | None = None,
    bash_read_only: bool = False,
    provider_order: list[str] | None = None,
    provider_allow_fallbacks: bool | None = None,
    api_key_env: str | None = None,
) -> dict[str, Any]:
    """Build an options dict for the direct API executor.

    budget_type: "turns" (max turns), "time" (wall-seconds), "cost" (USD).
    budget_value: numeric limit for the chosen budget type.
    allowed_write_dirs: if set, write_file/edit_file are restricted to these dirs only.
    scratch_dir: if set, created before execution; agent can use it for temp files.
    """
    root = resolve_project_root(project_root)
    provider_id, model_id = split_direct_model(model)
    resolved_add_dirs = resolve_data_dirs(root, data_dirs)

    system_with_dirs = system
    if resolved_add_dirs:
        dirs_note = "\n".join(f"- {path}" for path in resolved_add_dirs)
        system_with_dirs = (
            f"{system.rstrip()}\n\n"
            "Additional accessible data directories (use absolute paths):\n"
            f"{dirs_note}"
        )

    # Bash statefulness contract: the tool description alone is too weak a
    # signal (observed a solver replaying relative `cd ./x` prefixes and
    # breaking when the persisted cwd already WAS ./x). State it where
    # attention is strongest, for every bash-equipped agent uniformly.
    if any(str(t).lower() == "bash" for t in tools):
        # State the agent's OWN bash timeout too. The solver's timeout is
        # documented in the evolver's Reference (under the solver's tools), but
        # an agent was never told its own — so long-running analysis commands
        # could die on a limit it had no way to anticipate (0724 proofread).
        import os as _os
        _bash_to = _os.environ.get("EVOSKILL_BASH_TIMEOUT", "120")
        system_with_dirs = (
            f"{system_with_dirs.rstrip()}\n\n"
            "Bash tool behavior: the working directory PERSISTS between bash "
            "calls — a `cd` in one call carries over to the next. Do not "
            "re-run `cd` before every command; cd once (or use paths relative "
            "to where you already are). Shell variables and functions do NOT "
            f"persist between calls. YOUR bash calls time out after {_bash_to}s "
            "— split long work into steps rather than one command that risks "
            "being killed mid-way."
        )

    return {
        "system": system_with_dirs,
        "format": {
            "type": "json_schema",
            "schema": schema,
        },
        "tools": to_direct_tools(tools),
        "provider_id": provider_id,
        "model_id": model_id,
        "cwd": str(root),
        "skills_dir": str(root / ".claude" / "skills"),
        "add_dirs": resolved_add_dirs,
        "budget_type": budget_type,
        "budget_value": budget_value,
        "max_tool_output_chars": max_tool_output_chars,
        "compaction_threshold_tokens": compaction_threshold_tokens,
        "enable_thinking": enable_thinking,
        "thinking_effort": thinking_effort,
        "inject_skills": inject_skills,
        "allowed_write_dirs": allowed_write_dirs or [],
        "scratch_dir": scratch_dir,
        "bash_read_only": bash_read_only,
        "provider_order": provider_order,
        # None = OpenRouter default (may fall back off `order`). False HARD-pins
        # the listed providers -- use when provider identity is part of the
        # experiment (0722: DSV4 rides native DeepSeek).
        "provider_allow_fallbacks": provider_allow_fallbacks,
        # Name of the env var holding this agent's API key, NOT the key itself
        # -- keeps secrets out of anything that serialises options. None = use
        # the provider default (OPENROUTER_API_KEY). Not in the cache tree_hash.
        "api_key_env": api_key_env,
    }
