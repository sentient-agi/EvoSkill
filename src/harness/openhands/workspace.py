"""Workspace helpers for exposing EvoSkill data_dirs to OpenHands."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


_MOUNT_ROOT = Path(".evoskill") / "runtime" / "data_mounts"


@dataclass(frozen=True)
class DataDirMount:
    source: str
    path: str
    relative_path: str


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-")
    return slug or "data"


def _mount_alias(source: Path) -> str:
    digest = hashlib.sha256(str(source).encode("utf-8")).hexdigest()[:8]
    return f"{_slugify(source.name)}-{digest}"


def prepare_data_dir_mounts(
    project_root: str | Path,
    data_dirs: Iterable[str],
) -> list[DataDirMount]:
    """Mount external data directories into the repo workspace via symlinks."""
    root = Path(project_root).resolve()
    mount_root = root / _MOUNT_ROOT

    mounts: list[DataDirMount] = []
    seen_sources: set[str] = set()

    for raw_path in data_dirs:
        source = Path(raw_path).resolve()
        source_key = str(source)
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)

        if not source.exists():
            raise FileNotFoundError(f"OpenHands data_dir does not exist: {source}")

        mount_path = mount_root / _mount_alias(source)
        mount_root.mkdir(parents=True, exist_ok=True)
        if mount_path.exists() or mount_path.is_symlink():
            if mount_path.is_symlink():
                if mount_path.resolve() != source:
                    mount_path.unlink()
                    mount_path.symlink_to(source, target_is_directory=source.is_dir())
            else:
                raise FileExistsError(
                    f"OpenHands data mount path already exists and is not a symlink: {mount_path}"
                )
        else:
            mount_path.symlink_to(source, target_is_directory=source.is_dir())

        mounts.append(
            DataDirMount(
                source=source_key,
                path=str(mount_path),
                relative_path=mount_path.relative_to(root).as_posix(),
            )
        )

    return mounts


def serialize_data_dir_mounts(mounts: Iterable[DataDirMount]) -> list[dict[str, str]]:
    return [asdict(mount) for mount in mounts]
