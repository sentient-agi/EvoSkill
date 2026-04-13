#!/usr/bin/env python3
"""Run full evaluation on LiveCodeBench v6 dataset."""

import argparse
import asyncio
from pathlib import Path

import pandas as pd

from src.harness import Agent, set_sdk
from src.agent_profiles import make_livecodebench_agent_options
from src.evaluation.eval_full import evaluate_full, load_results
from src.evaluation.livecodebench import (
    score_livecodebench,
    ensure_livecodebench_dataset,
)
from src.schemas import AgentResponse


async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate agent on LiveCodeBench v6 dataset"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=Path,
        default=None,
        help="Path to LiveCodeBench CSV file (default: auto-download to .dataset/livecodebench_v6.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("results/livecodebench_eval_results.pkl"),
        help="Output pkl file path",
    )
    parser.add_argument(
        "--max-concurrent",
        "-c",
        type=int,
        default=4,
        help="Max concurrent evaluations (default: 4)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from existing results (start fresh)",
    )
    parser.add_argument(
        "--platform",
        "-p",
        type=str,
        default="all",
        help="Filter by platform ('all', 'leetcode', 'atcoder', or 'codeforces')",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="all",
        help="Filter by difficulty ('all', 'easy', 'medium', or 'hard')",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=None,
        help="Limit to first N samples (default: all 175)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="claude-opus-4-5-20251101",
        help="Model for agent (default: claude-opus-4-5-20251101)",
    )
    parser.add_argument(
        "--sdk",
        type=str,
        choices=["claude", "opencode"],
        default="claude",
        help="SDK to use: 'claude' or 'opencode' (default: claude)",
    )
    args = parser.parse_args()

    # Set SDK
    set_sdk(args.sdk)

    # Ensure dataset is downloaded
    if args.dataset is None:
        args.dataset = ensure_livecodebench_dataset()

    # Load dataset
    data = pd.read_csv(args.dataset)

    # Filter by platform if requested
    if args.platform != "all":
        data = data[data["platform"] == args.platform]

    # Filter by difficulty if requested
    if args.difficulty != "all":
        data = data[data["difficulty"] == args.difficulty]

    # Limit to num_samples if specified
    if args.num_samples is not None:
        data = data.head(args.num_samples)

    print(
        f"Dataset: {len(data)} samples (platform={args.platform}, difficulty={args.difficulty})"
    )
    print(f"SDK: {args.sdk}, Model: {args.model}")

    # Prepare items: (index, formatted_question, public_test_cases)
    items = [
        (
            idx,
            row["formatted_question"],
            row["public_test_cases"],
        )
        for idx, row in data.iterrows()
    ]

    # Create agent and run
    agent_options_factory = make_livecodebench_agent_options(model=args.model)
    agent = Agent(agent_options_factory, AgentResponse)

    print(f"Agent configured")

    await evaluate_full(
        agent=agent,
        items=items,
        output_path=args.output,
        max_concurrent=args.max_concurrent,
        resume=not args.no_resume,
    )

    # Summary and scoring
    all_results = load_results(args.output)
    successful = [r for r in all_results if r.error is None]
    failed = [r for r in all_results if r.error is not None]

    # Score successful results
    correct = 0
    for r in successful:
        if r.trace and r.trace.output and r.trace.output.final_answer:
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
    print(
        f"Pass@1: {correct}/{len(successful)} ({correct / len(successful) * 100:.1f}%)"
        if successful
        else "Pass@1: N/A (no successful results)"
    )
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
