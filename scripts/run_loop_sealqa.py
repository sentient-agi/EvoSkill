#!/usr/bin/env python3
"""Run self-improving agent loop on SEAL-QA dataset."""

import argparse
import asyncio

import pandas as pd

from src.api.data_utils import stratified_split
from src.loop import SelfImprovingLoop, LoopConfig, LoopAgents
from src.agent_profiles import (
    Agent,
    sealqa_agent_options,
    make_sealqa_agent_options,
    skill_proposer_options,
    prompt_proposer_options,
    skill_generator_options,
    prompt_generator_options,
)
from src.agent_profiles.skill_generator import get_project_root
from src.evaluation.sealqa_scorer import score_sealqa
from src.registry import ProgramManager
from src.schemas import (
    AgentResponse,
    SkillProposerResponse,
    PromptProposerResponse,
    ToolGeneratorResponse,
    PromptGeneratorResponse,
)



def _sealqa_scorer(question: str, predicted: str, ground_truth: str) -> float:
    """Wrapper around score_sealqa matching the runner's (question, predicted, ground_truth) signature."""
    return score_sealqa(question, ground_truth, predicted)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run self-improving agent loop on SEAL-QA")
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
        dest="continue_loop",
        action="store_true",
        help="Continue from existing frontier/branch instead of starting fresh",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=".dataset/seal-0.csv",
        help="Path to SEAL-QA CSV (default: .dataset/seal-0.csv)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.18,
        help="Fraction of each category for training (default: 0.18)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.12,
        help="Fraction of each category for validation (default: 0.12)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["opus", "sonnet", "haiku"],
        default="claude-opus-4-5-20251101",
        help="Model for base agent (default: opus via SDK default)",
    )
    return parser.parse_args()


async def main(args: argparse.Namespace):
    data = pd.read_csv(args.dataset)

    # Rename SEAL-QA columns to match stratified_split expectations
    data.rename(columns={"topic": "category", "answer": "ground_truth"}, inplace=True)

    # Stratified split by category
    train_pools, val_data, _test_data = stratified_split(data, train_ratio=args.train_ratio, val_ratio=args.val_ratio)

    # Print category distribution
    categories = list(train_pools.keys())
    total_train = sum(len(pool) for pool in train_pools.values())
    print(f"Dataset: {args.dataset}")
    print(f"Categories ({len(categories)}): {', '.join(categories)}")
    print(f"Training pools: {', '.join(f'{cat}: {len(pool)}' for cat, pool in train_pools.items())}")
    print(f"Total training samples: {total_train}")
    print(f"Validation samples: {len(val_data)} ({args.val_ratio:.0%} per category, min 1 each)")
    print(f"Split ratios: train={args.train_ratio:.0%}, val={args.val_ratio:.0%} (remaining {1-args.train_ratio-args.val_ratio:.0%} unused)")

    # Use custom model for sealqa agent if specified
    base_options = make_sealqa_agent_options(model=args.model) if args.model else sealqa_agent_options

    agents = LoopAgents(
        base=Agent(base_options, AgentResponse),
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
        categories_per_batch=args.failure_samples,
        cache_enabled=not args.no_cache,
        reset_feedback=not args.no_reset_feedback,
        continue_mode=args.continue_loop,
    )

    model_info = f", model={args.model}" if args.model else ""
    print(f"Running loop with evolution_mode={args.mode}{model_info}")
    loop = SelfImprovingLoop(config, agents, manager, train_pools, val_data, scorer=_sealqa_scorer)
    result = await loop.run()

    print(f"Best: {result.best_program} ({result.best_score:.2%})")
    print(f"Frontier: {result.frontier}")


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
