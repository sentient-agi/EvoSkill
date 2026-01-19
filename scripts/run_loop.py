#!/usr/bin/env python3
"""Run self-improving agent loop."""

import argparse
import asyncio

import pandas as pd

from src.loop import SelfImprovingLoop, LoopConfig, LoopAgents
from src.agent_profiles import (
    Agent,
    base_agent_options,
    skill_proposer_options,
    prompt_proposer_options,
    skill_generator_options,
    prompt_generator_options,
)
from src.agent_profiles.skill_generator import get_project_root
from src.registry import ProgramManager
from src.schemas import (
    AgentResponse,
    SkillProposerResponse,
    PromptProposerResponse,
    ToolGeneratorResponse,
    PromptGeneratorResponse,
)


def interleave_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Interleave rows from two dataframes (easy, hard, easy, hard, ...)"""
    result = []
    for i in range(max(len(df1), len(df2))):
        if i < len(df1):
            result.append(df1.iloc[i])
        if i < len(df2):
            result.append(df2.iloc[i])
    return pd.DataFrame(result)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run self-improving agent loop")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["skill_only", "prompt_only"],
        default="skill_only",
        help="Evolution mode: 'skill_only' or 'prompt_only' (default: skill_only)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="Maximum number of improvement iterations (default: 20)",
    )
    parser.add_argument(
        "--frontier-size",
        type=int,
        default=3,
        help="Number of top-performing programs to keep (default: 3)",
    )
    parser.add_argument(
        "--no-improvement-limit",
        type=int,
        default=5,
        help="Stop after this many iterations without improvement (default: 5)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of concurrent evaluations (default: 4)",
    )
    parser.add_argument(
        "--failure-samples",
        type=int,
        default=3,
        help="Number of samples to test per iteration for pattern detection (default: 3)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable run caching",
    )
    parser.add_argument(
        "--no-reset-feedback",
        action="store_true",
        help="Don't reset feedback history on start",
    )
    parser.add_argument(
        "--continue",
        dest="continue_loop",  # 'continue' is reserved keyword
        action="store_true",
        help="Continue from existing frontier/branch instead of starting fresh",
    )
<<<<<<< Updated upstream
=======
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=None,
        help="Seed for random sample selection (default: None = sequential)",
    )
>>>>>>> Stashed changes
    # Training composition
    parser.add_argument(
        "--train-easy-count",
        type=int,
        default=None,
        help="Number of easy questions for training",
    )
    parser.add_argument(
        "--train-hard-count",
        type=int,
        default=None,
        help="Number of hard questions for training",
    )
    parser.add_argument(
        "--train-easy-ratio",
        type=float,
        default=0.5,
        help="Easy ratio if counts not specified (default: 0.5)",
    )
    parser.add_argument(
        "--train-total",
        type=int,
        default=20,
        help="Total training samples if using ratio (default: 20)",
    )
    # Training order (curriculum)
    parser.add_argument(
        "--curriculum-order",
        type=str,
        choices=["easy_first", "hard_first", "mixed", "none"],
        default="easy_first",
        help="Curriculum order: easy_first, hard_first, mixed, or none (default: easy_first)",
    )
    # Validation composition
    parser.add_argument(
        "--val-easy-ratio",
        type=float,
        default=0.5,
        help="Validation easy/hard balance (default: 0.5)",
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=5,
        help="Total validation samples (default: 5)",
    )
    return parser.parse_args()


async def main(args: argparse.Namespace):
    data = pd.read_csv('.dataset/train_set.csv')

    # Split by difficulty
    easy_pool = data[data['difficulty'] == 'easy'].sample(frac=1, random_state=42)
    hard_pool = data[data['difficulty'] == 'hard'].sample(frac=1, random_state=42)

    # Determine training counts
    if args.train_easy_count is not None and args.train_hard_count is not None:
        # Use explicit counts
        n_train_easy = args.train_easy_count
        n_train_hard = args.train_hard_count
    else:
        # Use ratio
        n_train_easy = int(args.train_total * args.train_easy_ratio)
        n_train_hard = args.train_total - n_train_easy

    # Sample training data
    train_easy = easy_pool.head(min(n_train_easy, len(easy_pool)))
    train_hard = hard_pool.head(min(n_train_hard, len(hard_pool)))

    # Order based on curriculum
    if args.curriculum_order == "easy_first":
        train = pd.concat([train_easy, train_hard])
    elif args.curriculum_order == "hard_first":
        train = pd.concat([train_hard, train_easy])
    elif args.curriculum_order == "mixed":
        train = interleave_dataframes(train_easy, train_hard)
    else:  # "none" - shuffle
        train = pd.concat([train_easy, train_hard]).sample(frac=1, random_state=42)

    # Build balanced validation set (from remaining data)
    val_easy_pool = easy_pool.drop(train_easy.index, errors='ignore')
    val_hard_pool = hard_pool.drop(train_hard.index, errors='ignore')
    n_val_easy = int(args.val_count * args.val_easy_ratio)
    n_val_hard = args.val_count - n_val_easy
    val = pd.concat([
        val_easy_pool.head(min(n_val_easy, len(val_easy_pool))),
        val_hard_pool.head(min(n_val_hard, len(val_hard_pool)))
    ]).sample(frac=1, random_state=77)  # Shuffle validation

    train_data = [(row.question, row.ground_truth) for _, row in train.iterrows()]
    val_data = [(row.question, row.ground_truth) for _, row in val.iterrows()]

    print(f"Training: {len(train_easy)} easy, {len(train_hard)} hard ({args.curriculum_order} order)")
    print(f"Validation: {n_val_easy} easy, {n_val_hard} hard")

    agents = LoopAgents(
        base=Agent(base_agent_options, AgentResponse),
        skill_proposer=Agent(skill_proposer_options, SkillProposerResponse),
        prompt_proposer=Agent(prompt_proposer_options, PromptProposerResponse),
        skill_generator=Agent(skill_generator_options, ToolGeneratorResponse),
        prompt_generator=Agent(prompt_generator_options, PromptGeneratorResponse),
    )
    manager = ProgramManager(cwd=get_project_root())

    config = LoopConfig(
        max_iterations=args.max_iterations,
        frontier_size=args.frontier_size,
        no_improvement_limit=args.no_improvement_limit,
        concurrency=args.concurrency,
        evolution_mode=args.mode,
        failure_sample_count=args.failure_samples,
        sample_seed=args.sample_seed,
        cache_enabled=not args.no_cache,
        reset_feedback=not args.no_reset_feedback,
        continue_mode=args.continue_loop,
    )

    print(f"Running loop with evolution_mode={args.mode}")
    loop = SelfImprovingLoop(config, agents, manager, train_data, val_data)
    result = await loop.run()

    print(f"Best: {result.best_program} ({result.best_score:.2%})")
    print(f"Frontier: {result.frontier}")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
