from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


DEFAULT_OPENCODE_MODEL = "anthropic/claude-sonnet-4-6"

CLAUDE_TO_OPENCODE_TOOL = {
    "Read": "read",
    "Write": "write",
    "Bash": "bash",
    "Glob": "glob",
    "Grep": "grep",
    "Edit": "edit",
    "WebFetch": "webfetch",
    "WebSearch": "websearch",
    "TodoWrite": "todowrite",
    "Skill": "skill",
    # OpenCode does not expose a separate BashOutput tool.
    "BashOutput": None,
}


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
    root = resolve_project_root(project_root)
    resolved: list[str] = []
    for raw in data_dirs or []:
        path = Path(raw)
        resolved.append(str(path if path.is_absolute() else (root / path).resolve()))
    return resolved


def _normalize_permission_block(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        return {"*": value}
    if isinstance(value, dict):
        return dict(value)
    return {}


def ensure_opencode_project_permissions(
    project_root: str | Path | None,
    data_dirs: Iterable[str] | None = None,
) -> None:
    root = resolve_project_root(project_root)
    resolved_add_dirs = resolve_data_dirs(root, data_dirs)
    if not resolved_add_dirs:
        return

    jsonc_path = root / "opencode.jsonc"
    config_path = root / "opencode.json"
    if jsonc_path.exists() and not config_path.exists():
        return

    config: dict[str, Any] = {}
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            return

    config.setdefault("$schema", "https://opencode.ai/config.json")
    permission = _normalize_permission_block(config.get("permission"))
    external_directory = _normalize_permission_block(
        permission.get("external_directory")
    )

    changed = False
    for raw_path in resolved_add_dirs:
        path = str(Path(raw_path).resolve())
        for pattern in (path, f"{path}/**"):
            if external_directory.get(pattern) != "allow":
                external_directory[pattern] = "allow"
                changed = True

    if not changed and config_path.exists():
        return

    permission["external_directory"] = external_directory
    config["permission"] = permission
    config_path.write_text(json.dumps(config, indent=2) + "\n")


def split_opencode_model(model: str | None) -> tuple[str, str]:
    full = model or DEFAULT_OPENCODE_MODEL
    if "/" in full:
        return full.split("/", 1)
    return "anthropic", full


def to_opencode_tools(tools: Iterable[str]) -> dict[str, bool]:
    converted: dict[str, bool] = {}
    for tool in tools:
        normalized = CLAUDE_TO_OPENCODE_TOOL.get(tool, tool.lower())
        if normalized is not None:
            converted[normalized] = True
    return converted


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
        return build_opencode_options(
            system=system,
            schema=schema,
            tools=tools,
            project_root=project_root,
            model=model,
            data_dirs=data_dirs,
        )
    raise ValueError(f"Unknown SDK: {sdk!r}")


def build_claudecode_options(
    *,
    system: str,
    schema: dict[str, Any],
    tools: Iterable[str],
    project_root: str | Path | None = None,
    model: str | None = None,
    data_dirs: Iterable[str] | None = None,
    setting_sources: list[str] | None = None,
    permission_mode: str | None = None,
    max_buffer_size: int | None = None,
) -> Any:
    """Build ClaudeAgentOptions for the Claude SDK."""
    from claude_agent_sdk import ClaudeAgentOptions

    root = resolve_project_root(project_root)
    system_prompt: dict[str, Any] = {
        "type": "preset",
        "preset": "claude_code",
    }
    if system:
        system_prompt["append"] = system

    kwargs: dict[str, Any] = {
        "system_prompt": system_prompt,
        "output_format": {"type": "json_schema", "schema": schema},
        "allowed_tools": list(tools),
        "cwd": str(root),
    }
    if setting_sources is not None:
        kwargs["setting_sources"] = setting_sources
    if permission_mode is not None:
        kwargs["permission_mode"] = permission_mode
    if max_buffer_size is not None:
        kwargs["max_buffer_size"] = max_buffer_size
    if data_dirs is not None:
        kwargs["add_dirs"] = resolve_data_dirs(root, data_dirs)

    options = ClaudeAgentOptions(**kwargs)
    if model:
        options.model = model
    return options


def build_opencode_options(
    *,
    system: str,
    schema: dict[str, Any],
    tools: Iterable[str],
    project_root: str | Path | None = None,
    model: str | None = None,
    mode: str = "build",
    data_dirs: Iterable[str] | None = None,
) -> dict[str, Any]:
    root = resolve_project_root(project_root)
    provider_id, model_id = split_opencode_model(model)
    resolved_add_dirs = resolve_data_dirs(root, data_dirs)
    ensure_opencode_project_permissions(root, resolved_add_dirs)

    system_with_dirs = system
    if resolved_add_dirs:
        dirs_note = "\n".join(f"- {path}" for path in resolved_add_dirs)
        system_with_dirs = (
            f"{system.rstrip()}\n\n"
            "Additional accessible data directories are available outside the project root.\n"
            "Use absolute paths when you need to inspect them:\n"
            f"{dirs_note}"
        )

    return {
        "system": system_with_dirs,
        "format": {
            "type": "json_schema",
            "schema": schema,
        },
        "tools": to_opencode_tools(tools),
        "mode": mode,
        "provider_id": provider_id,
        "model_id": model_id,
        "cwd": str(root),
        "add_dirs": resolved_add_dirs,
    }
