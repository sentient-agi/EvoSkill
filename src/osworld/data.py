"""Task loading and stratified splitting for OSWorld."""

import json
import random
from pathlib import Path

from .types import OSWorldTask


def load_osworld_tasks(
    test_all_path: str | Path,
    examples_dir: str | Path,
) -> list[OSWorldTask]:
    """Load OSWorld tasks from the evaluation manifest.

    Args:
        test_all_path: Path to test_all.json (domain -> list of example IDs).
        examples_dir: Path to evaluation_examples/examples/ directory.

    Returns:
        Flat list of OSWorldTask objects.
    """
    test_all_path = Path(test_all_path)
    examples_dir = Path(examples_dir)

    with open(test_all_path, "r", encoding="utf-8") as f:
        test_all = json.load(f)

    tasks: list[OSWorldTask] = []
    for domain, example_ids in test_all.items():
        for example_id in example_ids:
            config_file = examples_dir / domain / f"{example_id}.json"
            if not config_file.exists():
                continue
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            tasks.append(OSWorldTask(
                id=example_id,
                domain=domain,
                instruction=config.get("instruction", ""),
                config=config,
            ))

    return tasks


def stratified_split_tasks(
    tasks: list[OSWorldTask],
    train_ratio: float = 0.18,
    val_ratio: float = 0.12,
    seed: int = 42,
) -> tuple[dict[str, list[OSWorldTask]], list[tuple[OSWorldTask, str]]]:
    """Split tasks ensuring each domain has at least 1 in both train and validation.

    Args:
        tasks: List of all OSWorld tasks.
        train_ratio: Fraction of each domain to use for training.
        val_ratio: Fraction of each domain to use for validation.
        seed: Random seed for reproducibility.

    Returns:
        train_pools: Dict mapping domain -> list of OSWorldTask.
        val_data: List of (OSWorldTask, domain) tuples for validation.
    """
    if train_ratio + val_ratio > 1.0:
        raise ValueError(
            f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) cannot exceed 1.0"
        )

    rng = random.Random(seed)

    # Group by domain
    by_domain: dict[str, list[OSWorldTask]] = {}
    for task in tasks:
        by_domain.setdefault(task.domain, []).append(task)

    train_pools: dict[str, list[OSWorldTask]] = {}
    val_data: list[tuple[OSWorldTask, str]] = []

    for domain, domain_tasks in sorted(by_domain.items()):
        shuffled = domain_tasks[:]
        rng.shuffle(shuffled)

        n_train = max(1, int(len(shuffled) * train_ratio))
        n_val = max(1, int(len(shuffled) * val_ratio))

        train_pools[domain] = shuffled[:n_train]
        val_data.extend(
            (task, domain)
            for task in shuffled[n_train:n_train + n_val]
        )

    return train_pools, val_data
