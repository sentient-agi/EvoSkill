#!/usr/bin/env python3
"""Run self-improving agent loop on SEAL-QA dataset."""

import argparse
import asyncio

import pandas as pd

from src.loop import SelfImprovingLoop, LoopConfig, LoopAgents
from src.agent_profiles import (
    Agent,
    set_sdk,
    make_sealqa_agent_options,
    skill_proposer_options,
    prompt_proposer_options,
    skill_generator_options,
    prompt_generator_options,
)
from src.agent_profiles.sealqa_agent import get_sealqa_agent_options
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


def positional_split(
    data: pd.DataFrame,
    train_end: int,
    val_end: int,
) -> tuple[dict[str, list[tuple[str, str]]], list[tuple[str, str, str]]]:
    """Split data by positional indices instead of stratified sampling.

    Args:
        data: DataFrame with 'question', 'ground_truth', 'category' columns.
        train_end: Index of last training row (exclusive).
        val_end: Index of last validation row (exclusive).

    Returns:
        train_pools: Dict mapping category -> list of (question, answer) tuples.
        val_data: List of (question, answer, category) tuples.
    """
    train_df = data.iloc[:train_end]
    val_df = data.iloc[train_end:val_end]

    # Build train_pools grouped by category
    train_pools: dict[str, list[tuple[str, str]]] = {}
    for _, row in train_df.iterrows():
        cat = row["category"]
        if cat not in train_pools:
            train_pools[cat] = []
        train_pools[cat].append((row["question"], row["ground_truth"]))

    # Build val_data
    val_data = [
        (row["question"], row["ground_truth"], row["category"])
        for _, row in val_df.iterrows()
    ]

    return train_pools, val_data


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
        default=6,
        help="Maximum number of improvement iterations (default: 6)",
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
        default=7,
        help="Stop after this many iterations without improvement (default: 7)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent evaluations (default: 1)",
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
        default=0.13,
        help="Fraction of each category for training (default: 0.13 -> 14 samples)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.13,
        help="Fraction of each category for validation (default: 0.13 -> 14 samples)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-opus-4-5-20251101",
        help="Model for base agent (default: claude-opus-4-5-20251101)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Provider ID for opencode SDK (e.g., gemini, arc)",
    )
    parser.add_argument(
        "--sdk",
        type=str,
        choices=["opencode", "claude"],
        default="claude",
        help="SDK for base agent: 'opencode' or 'claude' (default: claude)",
    )
    return parser.parse_args()


async def main(args: argparse.Namespace):
    # Set SDK for base agent
    set_sdk(args.sdk)

    data = pd.read_csv(args.dataset)
    data.rename(columns={"topic": "category", "answer": "ground_truth"}, inplace=True)

    # Positional split: first N train, next M val
    train_pools, val_data = positional_split(data, args.train_count, args.train_count + args.val_count)

    # Print split info
    total_train = sum(len(pool) for pool in train_pools.values())
    categories = list(train_pools.keys())
    print(f"Dataset: {args.dataset}")
    print(f"Categories ({len(categories)}): {', '.join(categories)}")
    print(f"Training pools: {', '.join(f'{cat}: {len(pool)}' for cat, pool in train_pools.items())}")
    print(f"Total training samples: {total_train}")
    print(f"Validation samples: {len(val_data)} ({args.val_ratio:.0%} per category, min 1 each)")
    print(f"Split ratios: train={args.train_ratio:.0%}, val={args.val_ratio:.0%} (remaining {1-args.train_ratio-args.val_ratio:.0%} unused)")

    # Build base agent options
    base_options = make_sealqa_agent_options(model=args.model, provider=args.provider)

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
        categories_per_batch=2,
        samples_per_category=1,
        cache_enabled=not args.no_cache,
        reset_feedback=not args.no_reset_feedback,
        continue_mode=args.continue_loop,
    )

    # Remember the starting branch so we can export skills back to it
    import subprocess
    starting_branch = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True, text=True, cwd=get_project_root(),
    ).stdout.strip()

    model_info = f", model={args.model}" if args.model else ""
    print(f"Running loop with evolution_mode={args.mode}, sdk={args.sdk}{model_info}")
    loop = SelfImprovingLoop(config, agents, manager, train_pools, val_data, scorer=_sealqa_scorer)
    result = await loop.run()

    print(f"Best: {result.best_program} ({result.best_score:.2%})")
    print(f"Frontier: {result.frontier}")

    # Export best skills back to the starting branch so eval scripts can use them
    if result.best_program != "base" and args.mode == "skill_only":
        exported = loop.export_best_skills(target_branch=starting_branch)
        if exported:
            print(f"Exported skills to {starting_branch}: {exported}")


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(main(args))
    except Exception as e:
        import traceback
        print(f"\n[FATAL] Loop crashed: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise
