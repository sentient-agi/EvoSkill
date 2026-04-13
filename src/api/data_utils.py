"""Shared dataset loading and splitting utilities.

Extracts the duplicated stratified_split() from scripts into one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .task_registry import TaskConfig


def load_dataset(path: str, task_config: TaskConfig) -> pd.DataFrame:
    """Load a dataset CSV and apply column renames from the task config.

    Args:
        path: Path to the CSV file.
        task_config: Task configuration with column_renames mapping.

    Returns:
        DataFrame with standardized column names (question, ground_truth, category).
    """
    data = pd.read_csv(path)
    if task_config.column_renames:
        data.rename(columns=task_config.column_renames, inplace=True)
    return data


def stratified_split(
    data: pd.DataFrame,
    train_ratio: float = 0.18,
    val_ratio: float = 0.12,
) -> tuple[dict[str, list[tuple[str, str]]], list[tuple[str, str, str]]]:
    """Split data ensuring each category has at least 1 in both train and validation.

    Args:
        data: DataFrame with 'question', 'ground_truth', 'category' columns.
        train_ratio: Fraction of each category to use for training.
        val_ratio: Fraction of each category to use for validation.

    Returns:
        train_pools: Dict mapping category -> list of (question, answer) tuples.
        val_data: List of (question, answer, category) tuples for validation.
    """
    if train_ratio + val_ratio > 1.0:
        raise ValueError(
            f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) cannot exceed 1.0"
        )

    # Drop rows with missing categories
    data = data.dropna(subset=["category"])
    categories = data["category"].unique()
    train_pools: dict[str, list[tuple[str, str]]] = {}
    val_data: list[tuple[str, str, str]] = []

    for cat in categories:
        cat_data = data[data["category"] == cat].sample(frac=1, random_state=42)
        n_train = max(1, int(len(cat_data) * train_ratio))
        n_val = max(1, int(len(cat_data) * val_ratio))

        # Train comes first, then validation
        train_pools[cat] = [
            (row.question, row.ground_truth)
            for _, row in cat_data.head(n_train).iterrows()
        ]
        val_data.extend(
            [
                (row.question, row.ground_truth, cat)
                for _, row in cat_data.iloc[n_train : n_train + n_val].iterrows()
            ]
        )

    return train_pools, val_data
