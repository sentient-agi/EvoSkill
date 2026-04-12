"""Shared utilities and the build_options() router.

Contains:
    - resolve_project_root() — find the repo root
    - resolve_data_dirs() — resolve relative data paths
    - build_options() — routes to the active SDK's builder
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


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
) -> Any:
    """Route to the correct builder for the active SDK.

    Claude-specific parameters (setting_sources, permission_mode,
    max_buffer_size) are forwarded only when the Claude SDK is active.
    They are silently ignored on other harnesses because those runtimes
    have no equivalent concept.
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
