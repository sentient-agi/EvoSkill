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
    sdk: Literal["claude", "opencode", "codex"] = Field(
        default="claude",
        description="SDK to use: 'claude', 'opencode', or 'codex'",
    )


async def main(settings: EvalSettings):
    set_sdk(settings.sdk)

    # Load dataset
    data = pd.read_csv(settings.dataset_path)

    # Filter by difficulty if requested
    if settings.difficulty != "all":
        data = data[data["difficulty"] == settings.difficulty]

    # Limit to num_samples if specified
    if settings.num_samples is not None:
        data = data.head(settings.num_samples)

    print(f"Dataset: {len(data)} samples ({settings.difficulty})")

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
