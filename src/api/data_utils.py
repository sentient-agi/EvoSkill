"""Shared dataset loading and splitting utilities.

Extracts the duplicated stratified_split() from scripts into one place.
"""

from __future__ import annotations

import pandas as pd

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

import pandas as pd

def stratified_split(
    data: pd.DataFrame,
    train_ratio: float = 0.18,
    val_ratio: float = 0.12,
    max_examples: int | None = None,
    extra_cols: list[str] | None = None
) -> tuple[dict[str, list[tuple]], list[tuple], list[tuple]]:
    """Split data ensuring each category has at least 1 in both train and validation.

    Args:
        data: DataFrame with 'question', 'ground_truth', 'category' columns.
        train_ratio: Fraction of each category to use for training.
        val_ratio: Fraction of each category to use for validation.
        max_examples: Maximum total examples to use across the entire dataset.
        extra_cols: List of additional column names to preserve in the tuples.

    Returns:
        train_pools: Dict mapping category -> list of tuples.
            Each tuple is (question, ground_truth, *extra_cols).
        val_data: List of tuples (question, ground_truth, category, *extra_cols).
        test_data: List of tuples (question, ground_truth, category, *extra_cols).
    """
    if train_ratio + val_ratio > 1.0:
        raise ValueError(
            f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) cannot exceed 1.0"
        )

    # Drop rows with missing categories
    data = data.dropna(subset=["category"])
    total_rows = len(data)
    categories = data["category"].unique()
    
    train_pools: dict[str, list[tuple]] = {}
    val_data: list[tuple] = []
    test_data: list[tuple] = []

    extra_cols = extra_cols or []

    def _make_tuple(row, include_category: bool = False) -> tuple:
        """Build tuple from row data."""
        base = [row.question, row.ground_truth]
        if include_category:
            base.append(row.category)
        for col in extra_cols:
            base.append(row.get(col, None))
        return tuple(base)

    for cat in categories:
        cat_data = data[data["category"] == cat].sample(frac=1, random_state=42)

        if max_examples is not None and total_rows > max_examples:
            cat_proportion = len(cat_data) / total_rows
            cat_limit = max(2, int(round(cat_proportion * max_examples)))
            cat_limit = min(cat_limit, len(cat_data))
            cat_data = cat_data.head(cat_limit)

        n_train = max(1, int(len(cat_data) * train_ratio))
        n_val = max(1, int(len(cat_data) * val_ratio))

        # Train comes first, then validation, then held-out test
        train_pools[cat] = [
            _make_tuple(row)
            for _, row in cat_data.head(n_train).iterrows()
        ]
        val_data.extend(
            [
                _make_tuple(row, include_category=True)
                for _, row in cat_data.iloc[n_train : n_train + n_val].iterrows()
            ]
        )
        test_data.extend(
            [
                _make_tuple(row, include_category=True)
                for _, row in cat_data.iloc[n_train + n_val :].iterrows()
            ]
        )

    return train_pools, val_data, test_data