#!/usr/bin/env python3
"""Run full evaluation on OfficeQA dataset."""

import asyncio
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)

from src.agent_profiles import (
    Agent,
    base_agent_options,
    make_base_agent_options,
    set_sdk,
)
from src.api.data_utils import stratified_split
from src.evaluation.eval_full import evaluate_full, load_results
from src.schemas import AgentResponse


class EvalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        cli_parse_args=True,
    )
    output: Path = Field(
        default=Path("results/eval_results.pkl"), description="Output pkl file path"
    )
    max_concurrent: int = Field(default=8, description="Max concurrent evaluations")
    resume: bool = Field(default=True, description="Resume from existing results")
    difficulty: Literal["all", "easy", "hard"] = Field(
        default="all", description="Filter by difficulty"
    )
    num_samples: Optional[int] = Field(
        default=None, description="Limit to first N samples"
    )
    model: Optional[str] = Field(
        default="claude-opus-4-5-20251101",
        description="Model for base agent (opus, sonnet, haiku)",
    )
    dataset_path: Path = Field(
        default=Path("~/officeqa/officeqa.csv").expanduser(),
        description="Path to OfficeQA dataset CSV",
    )
    sdk: Literal["claude", "opencode"] = Field(
        default="claude",
        description="SDK to use: 'claude' or 'opencode'",
    )
    held_out: bool = Field(
        default=False,
        description="Evaluate only on the held-out test set (excludes train/val samples)",
    )
    train_ratio: float = Field(
        default=0.18, description="Train ratio for stratified split"
    )
    val_ratio: float = Field(
        default=0.12, description="Val ratio for stratified split"
    )


async def main(settings: EvalSettings):
    set_sdk(settings.sdk)

    # Load dataset
    data = pd.read_csv(settings.dataset_path)

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

    # Create agent and run
    agent_options = (
        make_base_agent_options(model=settings.model)
        if settings.model
        else base_agent_options
    )
    agent = Agent(agent_options, AgentResponse)

    model_info = f" (model: {settings.model})" if settings.model else " (model: opus)"
    print(f"Agent configured{model_info}")

    await evaluate_full(
        agent=agent,
        items=items,
        output_path=settings.output,
        max_concurrent=settings.max_concurrent,
        resume=settings.resume,
    )

    # Summary
    all_results = load_results(settings.output)
    successful = [r for r in all_results if r.error is None]
    failed = [r for r in all_results if r.error is not None]

    print(f"\n{'=' * 50}")
    print(f"Total completed: {len(all_results)}/{len(data)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed indices: {[r.index for r in failed]}")
    print(f"Results saved to: {settings.output}")


if __name__ == "__main__":
    settings = EvalSettings()
    asyncio.run(main(settings))
