from scripts.run_eval_comb import EvalSettings
from src.api.data_utils import stratified_split
import pandas as pd
from pathlib import Path
from typing import Literal, Optional

def load_officeqa(data: pd.DataFrame, settings: EvalSettings):
    if settings.held_out:
        data.rename(columns={"answer": "ground_truth", "difficulty": "category"}, inplace=True)
        _train, _val, test_data = stratified_split(data, train_ratio=settings.train_ratio, val_ratio=settings.val_ratio)
        # Rebuild dataframe from held-out tuples
        data = pd.DataFrame(test_data, columns=["question", "answer", "difficulty"])
        print(f"Held-out test set: {len(data)} samples (train={settings.train_ratio:.0%}, val={settings.val_ratio:.0%})")
    else:
        print(f"Full dataset: {len(data)} samples")

    # Filter by difficulty if requested
    if settings.difficulty != "all":
        data = data[data["difficulty"] == settings.difficulty]

    # Limit to num_samples if specified
    if settings.num_samples is not None:
        data = data.head(settings.num_samples)

    print(f"Evaluating: {len(data)} samples (difficulty={settings.difficulty})")

    # Prepare items with index
    items = [
        (int(i), str(row["question"]), str(row["answer"])) for i, row in data.iterrows()
    ]

    return items

def load_sealqa(data: pd.DataFrame, settings: EvalSettings):
    if settings.held_out:
        data.rename(columns={"topic": "category", "answer": "ground_truth"}, inplace=True)
        _train, _val, test_data = stratified_split(data, train_ratio=settings.train_ratio, val_ratio=settings.val_ratio)
        # Rebuild dataframe from held-out tuples
        data = pd.DataFrame(test_data, columns=["question", "answer", "topic"])
        print(f"Held-out test set: {len(data)} samples (train={settings.train_ratio:.0%}, val={settings.val_ratio:.0%})")
    else:
        print(f"Full dataset: {len(data)} samples")

    # Filter by topic if requested
    if settings.topic != "all":
        data = data[data["topic"] == settings.topic]

    # Limit to num_samples if specified
    if settings.num_samples is not None:
        data = data.head(settings.num_samples)

    print(f"Evaluating: {len(data)} samples (topic={settings.topic})")

    items = [
        (idx, row["question"], row["answer"])
        for idx, row in data.iterrows()
    ]

    return items

def load_dabstep(data: pd.DataFrame, settings: EvalSettings, PROMPT):
    # Filter by level if requested
    if settings.level != "all":
        data = data[data["level"].astype(str) == settings.level]

    # Limit to num_samples if specified
    if settings.num_samples is not None:
        data = data.head(settings.num_samples)

    print(f"Dataset: {len(data)} samples (level={settings.level})")

    # Auto-discover context files from data-dir
    data_dir = Path(settings.data_dir).resolve()
    context_file_names = sorted(f.name for f in data_dir.iterdir() if f.is_file())
    context_files_text = "\n".join(f"- {data_dir / name}" for name in context_file_names)
    print(f"Context files ({len(context_file_names)}): {', '.join(context_file_names)}")

    # Prepare items: (task_id, formatted_prompt, answer)
    items = [
        (
            row["task_id"],
            PROMPT.format(
                context_files=context_files_text,
                question=row["question"],
                guidelines=row["guidelines"],
            ),
            row["answer"],
        )
        for _, row in data.iterrows()
    ]

    return items

def load_livecode(data: pd.DataFrame, settings: EvalSettings):
    # Filter by platform if requested
    if settings.platform != "all":
        data = data[data["platform"] == settings.platform]

    # Filter by difficulty if requested
    if settings.difficulty != "all":
        data = data[data["difficulty"] == settings.difficulty]

    # Limit to num_samples if specified
    if settings.num_samples is not None:
        data = data.head(settings.num_samples)

    print(
        f"Dataset: {len(data)} samples (platform={settings.platform}, difficulty={settings.difficulty})"
    )
    # print(f"SDK: {args.sdk}, Model: {args.model}")

    # Prepare items: (index, formatted_question, public_test_cases)
    items = [
        (
            idx,
            row["formatted_question"],
            row["public_test_cases"],
        )
        for idx, row in data.iterrows()
    ]

    return items