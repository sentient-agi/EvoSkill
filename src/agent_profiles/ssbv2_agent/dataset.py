"""SpreadsheetBench v2 dataset loader.

Loads tasks from all non-Visualization categories (Debugging, Financial_Model, Template).
Each task has an instruction, input spreadsheet path, golden response path, and answer_position.

Returns train/val items as (question_json, ground_truth_json, category) tuples.
The question_json embeds the task metadata so the solver can locate files.
The ground_truth_json embeds answer evaluation info for the scorer.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

CATEGORIES = ["Debugging", "Financial_Model", "Template"]
TASK_INDEX: dict[str, dict] = {}


def load_ssbv2(
    data_root: Path,
    *,
    n_train: int = 80,
    n_val: int = 60,
    seed: int = 42,
) -> tuple[list, list, list]:
    """Load SSB v2 tasks, split into train/val/test.

    Stratified by category to maintain distribution.

    Returns:
        (train_items, val_items, test_items) where each item is
        (question_json, ground_truth_json, category)
    """
    data_root = Path(data_root)
    base = data_root / "data" / "spreadsheetbench-v2"
    if not base.exists():
        base = data_root / "spreadsheetbench-v2"
    if not base.exists():
        base = data_root

    all_tasks = []
    for cat in CATEGORIES:
        cat_dir = base / cat
        dataset_path = cat_dir / "dataset.json"
        if not dataset_path.exists():
            print(f"  [WARN] Missing {dataset_path}")
            continue

        tasks = json.loads(dataset_path.read_text())
        for task in tasks:
            task_id = f"{cat}/{task['id']}"
            spreadsheet_path = str(cat_dir / task["spreadsheet_path"])
            golden_path = str(cat_dir / task["golden_response_path"])

            question = json.dumps({
                "task_id": task_id,
                "category": cat,
                "instruction": task["instruction"],
                "spreadsheet_path": spreadsheet_path,
                "spreadsheet_filename": Path(task["spreadsheet_path"]).name,
            })

            ground_truth = json.dumps({
                "task_id": task_id,
                "golden_path": golden_path,
                "answer_position": task["answer_position"],
            })

            TASK_INDEX[task_id] = {
                "spreadsheet_path": spreadsheet_path,
                "golden_path": golden_path,
                "answer_position": task["answer_position"],
                "category": cat,
                "instruction": task["instruction"],
            }

            all_tasks.append((question, ground_truth, cat))

    print(f"  SSB v2: {len(all_tasks)} tasks across {len(CATEGORIES)} categories "
          f"(Debugging={sum(1 for _,_,c in all_tasks if c=='Debugging')}, "
          f"FM={sum(1 for _,_,c in all_tasks if c=='Financial_Model')}, "
          f"Template={sum(1 for _,_,c in all_tasks if c=='Template')})")

    # Stratified split by category
    rng = random.Random(seed)
    by_cat: dict[str, list] = {}
    for item in all_tasks:
        by_cat.setdefault(item[2], []).append(item)

    total = len(all_tasks)
    train, val, test = [], [], []

    for cat, items in sorted(by_cat.items()):
        rng.shuffle(items)
        cat_frac = len(items) / total
        cat_n_train = round(n_train * cat_frac)
        cat_n_val = round(n_val * cat_frac)

        train.extend(items[:cat_n_train])
        val.extend(items[cat_n_train:cat_n_train + cat_n_val])
        test.extend(items[cat_n_train + cat_n_val:])

    # Shuffle within splits
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    print(f"  Split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test
