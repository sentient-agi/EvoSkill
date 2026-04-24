"""OpenHands SDK option building."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ..model_aliases import DEFAULT_ANTHROPIC_MODEL, normalize_harness_model
from ..utils import resolve_data_dirs, resolve_project_root
from .workspace import prepare_data_dir_mounts, serialize_data_dir_mounts


DEFAULT_OPENHANDS_MODEL = DEFAULT_ANTHROPIC_MODEL


def split_openhands_model(model: str | None) -> tuple[str, str]:
    """Parse 'provider/model' string into (provider_id, model_id)."""
    full = normalize_harness_model("openhands", model)
    if "/" in full:
        parts = full.split("/", 1)
        return parts[0], parts[1]
    return "anthropic", full


def build_openhands_options(
    *,
    system: str,
    schema: dict[str, Any],
    tools: Iterable[str],
    project_root: str | Path | None = None,
    model: str | None = None,
    data_dirs: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build an options dict for the OpenHands SDK."""
    root = resolve_project_root(project_root)
    provider_id, model_id = split_openhands_model(model)
    full_model = f"{provider_id}/{model_id}"
    source_add_dirs = resolve_data_dirs(root, data_dirs)
    data_dir_mounts = prepare_data_dir_mounts(root, source_add_dirs)
    mounted_add_dirs = [mount.path for mount in data_dir_mounts]

    system_with_dirs = system
    if data_dir_mounts:
        dirs_note = "\n".join(f"- {mount.relative_path}" for mount in data_dir_mounts)
        system_with_dirs = (
            f"{system.rstrip()}\n\n"
            "Additional data directories are mounted inside the workspace.\n"
            "Use these workspace paths when you need reference data:\n"
            f"{dirs_note}"
        )

    return {
        "sdk": "openhands",
        "system": system_with_dirs,
        "format": {
            "type": "json_schema",
            "schema": schema,
        },
        "tools": list(tools),
        "provider_id": provider_id,
        "model_id": model_id,
        "model": full_model,
        "cwd": str(root),
        "skills_dir": str(root / ".claude" / "skills"),
        "add_dirs": mounted_add_dirs,
        "source_add_dirs": source_add_dirs,
        "data_dir_mounts": serialize_data_dir_mounts(data_dir_mounts),
    }
