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
    sealqa_agent_options, 
    make_sealqa_agent_options,
    dabstep_agent_options, 
    make_dabstep_agent_options,
    make_livecodebench_agent_options,
    set_sdk,
)
from src.api.data_utils import stratified_split
from src.evaluation.eval_full import evaluate_full, load_results
from src.schemas import AgentResponse
from src.evaluation.sealqa_scorer import score_sealqa
from src.evaluation.reward import score_answer
from src.evaluation.dabstep_scorer import question_scorer
from src.evaluation.livecodebench.livecodebench_scorer import score_livecodebench
from scripts.load_dataset import load_dabstep, load_livecode, load_officeqa, load_sealqa

PROMPT = """You are an expert data analyst and you will answer factoid questions by loading and referencing the files/documents listed below.
You have these files available:
{context_files}

Here is the question you need to answer:
{question}

Here are the guidelines you must follow when answering the question above:
{guidelines}
"""

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
    topic: str = Field(
        default="all", description="Filter by topic"
    )
    level: str = Field(
        default="all", description="Filter by level"
    )
    platform: str = Field(
        default="all", description="Filter by platform"
    )
    num_samples: Optional[int] = Field(
        default=None, description="Limit to first N samples"
    )
    model: Optional[str] = Field(
        default="claude-opus-4-5-20251101",
        description="Model for base agent (opus, sonnet, haiku)",
    )
    dataset_path: Path = Field(
        default=Path(".dataset/officeqa.csv").expanduser(),
        description="Path to evaluation dataset CSV",
    )
    data_dir: str = Field(
        default="DABstep-data/data/context", description="Path to shared context files directory"
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

    if settings.dataset_path.name == "officeqa.csv":
        items = load_officeqa(data, settings)
        agent_options = (
            make_base_agent_options(model=settings.model)
            if settings.model
            else base_agent_options
        )
    elif settings.dataset_path.name == "seal-0.csv":
        items = load_sealqa(data, settings)
        agent_options = make_sealqa_agent_options(model=settings.model)
    elif settings.dataset_path.name == "dabstep_data.csv":
        items = load_dabstep(data, settings, PROMPT)
        agent_options = make_dabstep_agent_options(model=settings.model, data_dir=settings.data_dir)
    elif settings.dataset_path.name == "livecodebench_v6.csv":
        items = load_livecode(data, settings)
        agent_options = make_livecodebench_agent_options(model=settings.model)

    # Create agent and run
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

    # Score successful results (for officeqa use score_answer from reward.py)
    correct = 0
    for r in successful:
        if r.trace and r.trace.output and r.trace.output.final_answer:
            if settings.dataset_path.name == "officeqa.csv":
                score = score_answer(str(r.ground_truth), str(r.trace.output.final_answer))
                if score > 0:
                    correct += 1
            elif settings.dataset_path.name == "seal-0.csv":
                score = score_sealqa(r.question, str(r.ground_truth), str(r.trace.output.final_answer))
                if score > 0:
                    correct += 1
            elif settings.dataset_path.name == "dabstep_data.csv":
                score = question_scorer(str(r.trace.output.final_answer), str(r.ground_truth))
                if score:
                    correct += 1
            elif settings.dataset_path.name == "livecodebench_v6.csv":
                score = score_livecodebench(
                    r.question, str(r.ground_truth), str(r.trace.output.final_answer)
                )
                if score > 0:
                    correct += 1

    print(f"\n{'=' * 50}")
    print(f"Total completed: {len(all_results)}/{len(data)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed indices: {[r.index for r in failed]}")
    print(f"Accuracy: {correct}/{len(successful)} ({correct/len(successful)*100:.1f}%)" if successful else "Accuracy: N/A (no successful results)")
    print(f"Results saved to: {settings.output}")


if __name__ == "__main__":
    settings = EvalSettings()
    asyncio.run(main(settings))
