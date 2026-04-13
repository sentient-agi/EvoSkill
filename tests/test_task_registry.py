"""Tests for src/api/task_registry.py and src/api/data_utils.py."""

import pytest
import pandas as pd
from unittest.mock import MagicMock


# ===========================================================================
# TaskConfig
# ===========================================================================

class TestTaskConfig:
    def _make_config(self, **overrides):
        from src.api.task_registry import TaskConfig

        defaults = {
            "name": "test_task",
            "make_agent_options": lambda: {},
        }
        defaults.update(overrides)
        return TaskConfig(**defaults)

    def test_construction_with_required_fields(self):
        config = self._make_config()
        assert config.name == "test_task"
        assert callable(config.make_agent_options)

    def test_default_scorer_is_none(self):
        config = self._make_config()
        assert config.scorer is None

    def test_default_column_names(self):
        config = self._make_config()
        assert config.question_col == "question"
        assert config.answer_col == "ground_truth"
        assert config.category_col == "category"

    def test_default_column_renames_empty(self):
        config = self._make_config()
        assert config.column_renames == {}

    def test_default_dataset_empty_string(self):
        config = self._make_config()
        assert config.default_dataset == ""

    def test_custom_scorer_accepted(self):
        scorer = lambda q, p, g: 1.0  # noqa: E731
        config = self._make_config(scorer=scorer)
        assert config.scorer is scorer

    def test_custom_column_renames(self):
        renames = {"answer": "ground_truth", "q": "question"}
        config = self._make_config(column_renames=renames)
        assert config.column_renames == renames


# ===========================================================================
# register_task / get_task / list_tasks
# ===========================================================================

class TestTaskRegistry:
    """Use a fresh import in each test to avoid cross-test state contamination."""

    def _get_registry_functions(self):
        """Import registry functions fresh each time."""
        from src.api import task_registry
        return task_registry

    def test_get_builtin_base_task(self):
        from src.api.task_registry import get_task

        task = get_task("base")
        assert task.name == "base"

    def test_list_tasks_contains_base(self):
        from src.api.task_registry import list_tasks

        tasks = list_tasks()
        assert "base" in tasks

    def test_list_tasks_returns_sorted_list(self):
        from src.api.task_registry import list_tasks

        tasks = list_tasks()
        assert tasks == sorted(tasks)

    def test_register_and_get_custom_task(self):
        from src.api.task_registry import register_task, get_task, TaskConfig

        config = TaskConfig(
            name="__test_custom__",
            make_agent_options=lambda: {},
        )
        register_task(config)
        retrieved = get_task("__test_custom__")
        assert retrieved.name == "__test_custom__"

    def test_get_unknown_task_raises_key_error(self):
        from src.api.task_registry import get_task

        with pytest.raises(KeyError, match="Unknown task"):
            get_task("nonexistent_task_xyz")

    def test_error_message_lists_available_tasks(self):
        from src.api.task_registry import get_task

        try:
            get_task("nonexistent_task_xyz")
        except KeyError as e:
            assert "base" in str(e)

    def test_register_overwrites_existing_task(self):
        from src.api.task_registry import register_task, get_task, TaskConfig

        config1 = TaskConfig(name="__overwrite_test__", make_agent_options=lambda: "v1")
        config2 = TaskConfig(name="__overwrite_test__", make_agent_options=lambda: "v2")
        register_task(config1)
        register_task(config2)
        # Second registration should overwrite
        retrieved = get_task("__overwrite_test__")
        assert retrieved.make_agent_options() == "v2"

    def test_list_tasks_includes_custom_registered(self):
        from src.api.task_registry import register_task, list_tasks, TaskConfig

        config = TaskConfig(name="__list_test__", make_agent_options=lambda: {})
        register_task(config)
        assert "__list_test__" in list_tasks()


# ===========================================================================
# data_utils — load_dataset and stratified_split
# ===========================================================================

class TestLoadDataset:
    def _make_csv(self, tmp_path, data=None):
        if data is None:
            data = {
                "question": ["Q1", "Q2"],
                "ground_truth": ["A1", "A2"],
                "category": ["cat_a", "cat_b"],
            }
        df = pd.DataFrame(data)
        path = tmp_path / "dataset.csv"
        df.to_csv(path, index=False)
        return str(path)

    def test_loads_csv_and_returns_dataframe(self, tmp_path):
        from src.api.data_utils import load_dataset
        from src.api.task_registry import TaskConfig

        path = self._make_csv(tmp_path)
        task = TaskConfig(name="t", make_agent_options=lambda: {})
        df = load_dataset(path, task)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

    def test_applies_column_renames(self, tmp_path):
        from src.api.data_utils import load_dataset
        from src.api.task_registry import TaskConfig

        data = {"q": ["Q1"], "ans": ["A1"], "cat": ["c"]}
        df = pd.DataFrame(data)
        path = tmp_path / "data.csv"
        df.to_csv(path, index=False)

        task = TaskConfig(
            name="t",
            make_agent_options=lambda: {},
            column_renames={"q": "question", "ans": "ground_truth", "cat": "category"},
        )
        result = load_dataset(str(path), task)
        assert "question" in result.columns
        assert "ground_truth" in result.columns

    def test_no_renames_preserves_original_columns(self, tmp_path):
        from src.api.data_utils import load_dataset
        from src.api.task_registry import TaskConfig

        path = self._make_csv(tmp_path)
        task = TaskConfig(name="t", make_agent_options=lambda: {})
        result = load_dataset(path, task)
        assert "question" in result.columns
        assert "ground_truth" in result.columns
        assert "category" in result.columns


class TestStratifiedSplit:
    def _make_df(self, n_per_cat=10, categories=None):
        if categories is None:
            categories = ["cat_a", "cat_b", "cat_c"]
        rows = []
        for cat in categories:
            for i in range(n_per_cat):
                rows.append({
                    "question": f"{cat}_Q{i}",
                    "ground_truth": f"{cat}_A{i}",
                    "category": cat,
                })
        return pd.DataFrame(rows)

    def test_returns_correct_types(self):
        from src.api.data_utils import stratified_split

        df = self._make_df()
        train_pools, val_data = stratified_split(df)
        assert isinstance(train_pools, dict)
        assert isinstance(val_data, list)

    def test_train_pools_keyed_by_category(self):
        from src.api.data_utils import stratified_split

        df = self._make_df(categories=["math", "finance"])
        train_pools, _ = stratified_split(df)
        assert "math" in train_pools
        assert "finance" in train_pools

    def test_each_category_has_at_least_one_in_train(self):
        from src.api.data_utils import stratified_split

        df = self._make_df(n_per_cat=3)
        train_pools, _ = stratified_split(df)
        for cat, items in train_pools.items():
            assert len(items) >= 1, f"Category {cat!r} has no training items"

    def test_each_category_has_at_least_one_in_val(self):
        from src.api.data_utils import stratified_split

        df = self._make_df(n_per_cat=3)
        _, val_data = stratified_split(df)
        val_cats = {cat for _, _, cat in val_data}
        for cat in ["cat_a", "cat_b", "cat_c"]:
            assert cat in val_cats

    def test_val_data_tuples_have_three_elements(self):
        from src.api.data_utils import stratified_split

        df = self._make_df()
        _, val_data = stratified_split(df)
        for item in val_data:
            assert len(item) == 3  # (question, ground_truth, category)

    def test_train_items_are_tuples_of_two(self):
        from src.api.data_utils import stratified_split

        df = self._make_df()
        train_pools, _ = stratified_split(df)
        for cat, items in train_pools.items():
            for item in items:
                assert len(item) == 2  # (question, ground_truth)

    def test_invalid_ratio_sum_raises(self):
        from src.api.data_utils import stratified_split

        df = self._make_df()
        with pytest.raises(ValueError, match="cannot exceed 1.0"):
            stratified_split(df, train_ratio=0.7, val_ratio=0.5)

    def test_drops_rows_with_missing_category(self):
        from src.api.data_utils import stratified_split

        df = self._make_df(n_per_cat=5)
        # Inject a row with NaN category
        bad_row = pd.DataFrame([{"question": "Q?", "ground_truth": "A?", "category": None}])
        df = pd.concat([df, bad_row], ignore_index=True)
        # Should not raise
        train_pools, val_data = stratified_split(df)
        # None should not appear as a category key
        assert None not in train_pools

    def test_single_item_per_category_still_works(self):
        from src.api.data_utils import stratified_split

        df = self._make_df(n_per_cat=1)
        # Should not raise even with only 1 item per category
        train_pools, val_data = stratified_split(df)
        for cat, items in train_pools.items():
            assert len(items) >= 1

    def test_large_dataset_proportions_approximate(self):
        from src.api.data_utils import stratified_split

        df = self._make_df(n_per_cat=100)
        train_pools, val_data = stratified_split(df, train_ratio=0.18, val_ratio=0.12)
        # Each category should have ~18 train and ~12 val items
        for cat, items in train_pools.items():
            assert 15 <= len(items) <= 20  # allow for rounding
