"""Load and validate .evoskill/config.toml + task.md."""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from src.harness.model_aliases import (
    HarnessName,
    default_model_for_harness,
    normalize_harness_model,
)

def _docker_path_overrides() -> dict[str, str]:
    """Read path overrides injected by the Docker launcher."""
    raw = os.environ.get("EVOSKILL_PATH_OVERRIDES", "")
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}

EVOSKILL_DIR = '.evoskill'


@dataclass
class HarnessConfig:
    name: HarnessName = 'claude'
    model: str | None = field(default_factory=lambda: default_model_for_harness("claude"))
    data_dirs: list[str] = field(default_factory=list)
    timeout_seconds: int = 1200
    max_retries: int = 3


@dataclass
class EvolutionConfig:
    mode: Literal['skill_only', 'prompt_only'] = 'skill_only'
    iterations: int = 20
    frontier_size: int = 3
    concurrency: int = 4
    no_improvement_limit: int = 5
    failure_samples: int = 3


@dataclass
class DatasetConfig:
    path: str = 'data/questions.csv'
    question_column: str = 'question'
    ground_truth_column: str = 'ground_truth'
    category_column: str | None = None
    train_ratio: float = 0.18
    val_ratio: float = 0.12


@dataclass
class ScorerConfig:
    type: Literal['exact', 'multi_tolerance', 'llm', 'script'] = 'multi_tolerance'
    rubric: str | None = None
    model: str | None = None
    provider: str | None = None
    command: str | None = None


@dataclass
class DaytonaConfig:
    api_key: str | None = None
    image: str = ''
    cpu: int = 4
    memory: int = 8        # GB
    disk: int = 10         # GB
    timeout: int = 0       # 0 = no auto-stop


@dataclass
class AWSConfig:
    region: str = 'us-east-1'
    s3_bucket: str = ''
    ecr_repo: str = ''
    ecs_cluster: str = 'default'
    task_cpu: str = '4096'
    task_memory: str = '16384'


@dataclass
class DownloadConfig:
    all_branches: bool = False
    cache: bool = False
    reports: bool = True
    feedback_history: bool = False


_VALID_REMOTE_TARGETS = ('daytona', 'aws')


@dataclass
class RemoteConfig:
    target: str = 'daytona'
    daytona: DaytonaConfig | None = None
    aws: AWSConfig | None = None
    download: DownloadConfig = field(default_factory=DownloadConfig)


@dataclass
class ProjectConfig:
    harness: HarnessConfig = field(default_factory=HarnessConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    scorer: ScorerConfig = field(default_factory=ScorerConfig)
    remote: RemoteConfig | None = None
    project_root: Path = field(default_factory=Path.cwd)
    task_description: str = ''
    task_constraints: str = ''

    @property
    def evoskill_dir(self) -> Path:
        return self.project_root / EVOSKILL_DIR

    @property
    def dataset_path(self) -> Path:
        """Resolve dataset path, with Docker override support."""
        override = _docker_path_overrides().get("dataset_path")
        if override:
            return Path(override)
        p = Path(self.dataset.path)
        if p.is_absolute():
            return p
        return self.evoskill_dir / p


def _find_project_root(start: Path | None = None) -> Path | None:
    """Walk up from start looking for a .evoskill/ directory."""
    current = Path.cwd() if start is None else start
    for parent in [current, *current.parents]:
        if (parent / EVOSKILL_DIR).exists():
            return parent
    return None


def _parse_task_md(text: str) -> tuple[str, str]:
    """Split task.md into (description, constraints) at the --- separator."""
    parts = text.split('\n---\n', maxsplit=1)
    description = parts[0].strip()
    constraints = parts[1].strip() if len(parts) > 1 else ''
    return description, constraints


def load_config(start: Path | None = None) -> ProjectConfig:
    """Find and load the project config. Exits with a message if not found."""
    root = _find_project_root(start)
    if root is None:
        import sys
        print("Error: no .evoskill/ directory found. Run 'evoskill init' first.")
        sys.exit(1)

    config_path = root / EVOSKILL_DIR / 'config.toml'
    if not config_path.exists():
        import sys
        print(f"Error: {config_path} not found. Run 'evoskill init' first.")
        sys.exit(1)

    with open(config_path, 'rb') as f:
        raw = tomllib.load(f)

    harness_raw = dict(raw.get('harness', {}))
    harness_name = harness_raw.get('name', 'claude')
    harness_raw['model'] = normalize_harness_model(harness_name, harness_raw.get('model'))

    # Docker path overrides for data_dirs
    overrides = _docker_path_overrides()
    if "data_dirs" in overrides:
        harness_raw['data_dirs'] = [d.strip() for d in overrides["data_dirs"].split(",") if d.strip()]

    harness = HarnessConfig(**harness_raw)
    evolution = EvolutionConfig(**raw.get('evolution', {}))
    dataset = DatasetConfig(**raw.get('dataset', {}))
    scorer = ScorerConfig(**raw.get('scorer', {}))

    # Parse remote config
    remote: RemoteConfig | None = None
    remote_raw = raw.get('remote')
    if remote_raw is not None:
        target = remote_raw.get('target', 'daytona')
        if target not in _VALID_REMOTE_TARGETS:
            raise ValueError(
                f"Unsupported remote target '{target}'. "
                f"Valid targets: {', '.join(_VALID_REMOTE_TARGETS)}"
            )

        daytona_cfg: DaytonaConfig | None = None
        if 'daytona' in remote_raw:
            daytona_cfg = DaytonaConfig(**remote_raw['daytona'])

        # Daytona API key: toml takes precedence, fall back to env var
        if daytona_cfg is not None and not daytona_cfg.api_key:
            env_key = os.environ.get('DAYTONA_API_KEY')
            if env_key:
                daytona_cfg.api_key = env_key
        elif target == 'daytona' and daytona_cfg is None:
            # target is daytona but no [remote.daytona] section — create with defaults
            env_key = os.environ.get('DAYTONA_API_KEY')
            daytona_cfg = DaytonaConfig(api_key=env_key)

        aws_cfg: AWSConfig | None = None
        if 'aws' in remote_raw:
            aws_cfg = AWSConfig(**remote_raw['aws'])

        download_cfg = DownloadConfig(**remote_raw.get('download', {}))

        remote = RemoteConfig(
            target=target,
            daytona=daytona_cfg,
            aws=aws_cfg,
            download=download_cfg,
        )

    task_path = root / EVOSKILL_DIR / 'task.md'
    description, constraints = _parse_task_md(task_path.read_text()) if task_path.exists() else ('', '')

    return ProjectConfig(
        harness=harness,
        evolution=evolution,
        dataset=dataset,
        scorer=scorer,
        remote=remote,
        project_root=root,
        task_description=description,
        task_constraints=constraints,
    )
