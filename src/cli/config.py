"""Load and validate .evoskill/config.toml + task.md."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from src.harness.model_aliases import (
    HarnessName,
    default_model_for_harness,
    normalize_harness_model,
)

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
    # 'skill_unified' uses a single evolver agent that both proposes the
    # mutation and writes the skill. It's the empirically-better default
    # (see PR #23 A/B); 'skill_only' / 'prompt_only' keep the legacy
    # split-agent paths.
    mode: Literal['skill_only', 'prompt_only', 'skill_unified'] = 'skill_unified'
    iterations: int = 20
    frontier_size: int = 3
    concurrency: int = 4
    no_improvement_limit: int = 5
    failure_samples: int = 3
    # Once frontier accuracy reaches this threshold, switch from optimizing
    # for accuracy to optimizing for cost (Phase 2). None disables Phase 2.
    accuracy_threshold: float | None = None
    # Override the model used by evolver agents (proposer/generator in split
    # modes, unified evolver in skill_unified). None inherits harness.model.
    evolver_model: str | None = None


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
class ProjectConfig:
    harness: HarnessConfig = field(default_factory=HarnessConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    scorer: ScorerConfig = field(default_factory=ScorerConfig)
    project_root: Path = field(default_factory=Path.cwd)
    task_description: str = ''
    task_constraints: str = ''

    @property
    def evoskill_dir(self) -> Path:
        return self.project_root / EVOSKILL_DIR

    @property
    def dataset_path(self) -> Path:
        """Resolve dataset path relative to .evoskill/."""
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
    harness = HarnessConfig(**harness_raw)
    evolution = EvolutionConfig(**raw.get('evolution', {}))
    dataset = DatasetConfig(**raw.get('dataset', {}))
    scorer = ScorerConfig(**raw.get('scorer', {}))

    task_path = root / EVOSKILL_DIR / 'task.md'
    description, constraints = _parse_task_md(task_path.read_text()) if task_path.exists() else ('', '')

    return ProjectConfig(
        harness=harness,
        evolution=evolution,
        dataset=dataset,
        scorer=scorer,
        project_root=root,
        task_description=description,
        task_constraints=constraints,
    )
