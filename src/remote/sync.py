"""Sync logic — file lists, exclude patterns, data dir remapping, git bundle helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.cli.config import DownloadConfig

# Directories/patterns excluded from upload to remote.
UPLOAD_EXCLUDES = (
    ".venv",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    ".cache",
    "notebooks",
    "assets",
    "*.pyc",
    "*.egg-info",
)


def should_exclude_upload(rel_path: Path) -> bool:
    """Check whether a relative path should be excluded from upload."""
    parts = rel_path.parts
    for part in parts:
        for pattern in UPLOAD_EXCLUDES:
            if pattern.startswith("*"):
                # Suffix match (e.g. *.pyc)
                if part.endswith(pattern[1:]):
                    return True
            elif part == pattern:
                return True
    return False


def upload_file_list(project_root: Path) -> list[Path]:
    """Walk project_root and return all files that should be uploaded.

    Skips directories/files matching UPLOAD_EXCLUDES.
    Returns absolute paths.
    """
    files: list[Path] = []
    for path in project_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(project_root)
        if not should_exclude_upload(rel):
            files.append(path)
    return sorted(files)


def download_file_list(cfg: DownloadConfig) -> list[str]:
    """Return the list of remote paths to download based on download config.

    Always includes the best program state (skills, program.yaml).
    Optional extras controlled by DownloadConfig flags.
    """
    paths = [
        ".claude/skills/",
        ".claude/program.yaml",
    ]

    if cfg.reports:
        paths.append(".evoskill/reports/")

    if cfg.all_branches:
        paths.append(".git/refs/")
        paths.append(".git/objects/")

    if cfg.cache:
        paths.append(".cache/runs/")

    if cfg.feedback_history:
        paths.append(".claude/feedback_history.md")

    return paths


@dataclass
class DataDirMapping:
    """Maps a host data directory to its container equivalent."""

    host_path: Path
    container_path: str
    needs_upload: bool  # False if already under project root (mounted via /workspace)


def remap_data_dirs(data_dirs: list[str], project_root: Path) -> list[DataDirMapping]:
    """Build a mapping of host data dirs to container paths.

    Dirs inside the project get /workspace/ prefix (no upload needed).
    Dirs outside get /mnt/data/{name} prefix (need upload).
    """
    if not data_dirs:
        return []

    resolved_root = project_root.resolve()
    mappings: list[DataDirMapping] = []

    for d in data_dirs:
        p = Path(d).resolve()
        try:
            rel = p.relative_to(resolved_root)
            # Inside project root — available under /workspace
            mappings.append(DataDirMapping(
                host_path=p,
                container_path=f"/workspace/{rel}",
                needs_upload=False,
            ))
        except ValueError:
            # Outside project root — needs upload to /mnt/data/{name}
            mappings.append(DataDirMapping(
                host_path=p,
                container_path=f"/mnt/data/{p.name}",
                needs_upload=True,
            ))

    return mappings


def bundle_create_args(
    output_path: str,
    all_branches: bool = True,
    branch: str | None = None,
) -> list[str]:
    """Build the git bundle create command args.

    If all_branches=True, bundles everything (--all).
    Otherwise bundles only the specified branch.
    """
    cmd = ["git", "bundle", "create", output_path]
    if all_branches:
        cmd.append("--all")
    elif branch:
        cmd.append(branch)
    return cmd


def bundle_unbundle_args(bundle_path: str) -> list[str]:
    """Build the git bundle unbundle command args."""
    return ["git", "bundle", "unbundle", bundle_path]
