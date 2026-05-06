"""Load a downloaded Harbor task directory as an EvoSkill dataset.

Harbor's `harbor download <dataset>` produces a tree like:

    <root>/<org>/<task_name>/<digest>/
        task.toml
        instruction.md
        environment/Dockerfile
        tests/{test.sh,verify.py,expected.json,config.json}

EvoSkill's loop expects (question, ground_truth, category) triples. We turn each
discovered task into:

    question     -> "<org>/<task_name>"  (Harbor-relative task id)
    ground_truth -> "1.0"                (sentinel; Harbor's verifier is the truth)
    category     -> task.toml [metadata].category, or "default"

The HarborAgent later turns the question (task id) into a `harbor run -p <digest_dir>`
invocation. The harbor scorer reads the agent's reward back from a JSON envelope.
"""

from __future__ import annotations

import fnmatch
import tomllib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Internal envelope used to map task id -> resolved digest dir without polluting
# the downstream tuple shape. `task_paths` is read by HarborAgent.
TASK_PATH_INDEX: dict[str, Path] = {}


@dataclass
class _HarborTask:
    task_id: str          # "arcprize/0934a4d8_0"
    task_dir: Path        # the digest dir containing task.toml
    category: str
    difficulty: str       # "easy" / "medium" / "hard" / "default" if absent


class HarborLoadError(RuntimeError):
    pass


def _iter_task_dirs(root: Path) -> list[_HarborTask]:
    """Walk <root>/<org>/<name>/<digest>/ looking for task.toml.

    A task.toml at depth 3 from root identifies a real task root. Other depths
    are tolerated so the loader still works if Harbor changes its layout.
    """
    found: list[_HarborTask] = []
    if not root.is_dir():
        raise HarborLoadError(f"harbor_tasks_root does not exist: {root}")

    for toml_path in root.rglob("task.toml"):
        task_dir = toml_path.parent
        try:
            data = tomllib.loads(toml_path.read_text())
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise HarborLoadError(f"failed to read {toml_path}: {exc}") from exc

        # Pull task id from [task].name first, fall back to dir layout.
        task_section = data.get("task") or {}
        task_id = task_section.get("name")
        if not task_id:
            # Fallback: <root>/<org>/<name>/<digest> -> "<org>/<name>"
            try:
                rel = task_dir.relative_to(root)
                parts = rel.parts
                if len(parts) >= 2:
                    task_id = f"{parts[0]}/{parts[1]}"
            except ValueError:
                pass
        if not task_id:
            continue

        metadata = data.get("metadata") or {}
        category = str(metadata.get("category") or "default")
        difficulty = str(metadata.get("difficulty") or "default")

        found.append(
            _HarborTask(
                task_id=task_id,
                task_dir=task_dir,
                category=category,
                difficulty=difficulty,
            )
        )

    return found


def _matches_any(task_id: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(task_id, p) for p in patterns)


def load_harbor_tasks(cfg) -> pd.DataFrame:
    """Return a DataFrame with columns (question, ground_truth, category).

    `cfg` is a ProjectConfig. Reads cfg.dataset.harbor_tasks_root, harbor_limit,
    harbor_include, harbor_exclude.

    Side effect: populates the module-level TASK_PATH_INDEX so HarborAgent can
    resolve a task id back to its on-disk digest directory.
    """
    root_str = cfg.dataset.harbor_tasks_root
    if not root_str:
        raise HarborLoadError(
            "dataset.harbor_tasks_root must be set when dataset.source = 'harbor'"
        )

    root = Path(root_str).expanduser()
    if not root.is_absolute():
        root = (cfg.project_root / root).resolve()

    tasks = _iter_task_dirs(root)
    if not tasks:
        raise HarborLoadError(f"no task.toml files found under {root}")

    include = cfg.dataset.harbor_include or []
    exclude = cfg.dataset.harbor_exclude or []
    difficulties = cfg.dataset.harbor_difficulty or []
    if include:
        tasks = [t for t in tasks if _matches_any(t.task_id, include)]
    if exclude:
        tasks = [t for t in tasks if not _matches_any(t.task_id, exclude)]
    if difficulties:
        tasks = [t for t in tasks if t.difficulty in difficulties]

    # Stable order by task id so train/val splits are reproducible across runs.
    tasks.sort(key=lambda t: t.task_id)

    if cfg.dataset.harbor_limit:
        tasks = tasks[: cfg.dataset.harbor_limit]

    if not tasks:
        raise HarborLoadError("harbor task filters left no tasks to run")

    # Refresh the global path index for HarborAgent.
    TASK_PATH_INDEX.clear()
    for t in tasks:
        TASK_PATH_INDEX[t.task_id] = t.task_dir

    return pd.DataFrame(
        {
            "question": [t.task_id for t in tasks],
            "ground_truth": ["1.0"] * len(tasks),
            "category": [t.category for t in tasks],
        }
    )


def resolve_task_dir(task_id: str) -> Path | None:
    """Look up a task id's digest dir. Returns None if unknown."""
    return TASK_PATH_INDEX.get(task_id)
