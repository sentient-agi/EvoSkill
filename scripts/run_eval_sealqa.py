#!/usr/bin/env python3
"""Run full evaluation on SEAL-QA dataset."""
import argparse
import asyncio
from pathlib import Path

import pandas as pd

from src.agent_profiles import Agent, sealqa_agent_options, make_sealqa_agent_options
from src.api.data_utils import stratified_split
from src.evaluation.eval_full import evaluate_full, load_results
from src.evaluation.sealqa_scorer import score_sealqa
from src.schemas import AgentResponse


async def main():
    parser = argparse.ArgumentParser(description="Evaluate agent on SEAL-QA dataset")
    parser.add_argument(
        "--dataset",
        "-d",
        type=Path,
        default=Path(".dataset/seal-0.csv"),
        help="Path to SEAL-QA CSV file (default: .dataset/seal-0.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("results/sealqa_eval_results.pkl"),
        help="Output pkl file path",
    )
    parser.add_argument(
        "--max-concurrent",
        "-c",
        type=int,
        default=8,
        help="Max concurrent evaluations",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from existing results (start fresh)",
    )
    parser.add_argument(
        "--topic",
        "-t",
        type=str,
        default="all",
        help="Filter by topic column ('all' or a specific topic value)",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=None,
        help="Limit to first N samples (default: all)",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="claude-opus-4-5-20251101",
        help="Model for agent (default: claude-opus-4-5-20251101)",
    )
    parser.add_argument(
        "--held-out",
        action="store_true",
        help="Evaluate only on the held-out test set (excludes train/val samples)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.10,
        help="Train ratio for stratified split (default: 0.10)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.12,
        help="Val ratio for stratified split (default: 0.12)",
    )
    args = parser.parse_args()

    # Load dataset
    data = pd.read_csv(args.dataset)

    if args.held_out:
        data.rename(columns={"topic": "category", "answer": "ground_truth"}, inplace=True)
        _train, _val, test_data = stratified_split(data, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
        # Rebuild dataframe from held-out tuples
        data = pd.DataFrame(test_data, columns=["question", "answer", "topic"])
        print(f"Held-out test set: {len(data)} samples (train={args.train_ratio:.0%}, val={args.val_ratio:.0%})")
    else:
        print(f"Full dataset: {len(data)} samples")

    # Filter by topic if requested
    if args.topic != "all":
        data = data[data["topic"] == args.topic]

    # Limit to num_samples if specified
    if args.num_samples is not None:
        data = data.head(args.num_samples)

    print(f"Evaluating: {len(data)} samples (topic={args.topic})")

    items = [
        (idx, row["question"], row["answer"])
        for idx, row in data.iterrows()
    ]

    # Create agent and run
    agent_options_factory = make_sealqa_agent_options(model=args.model)
    agent = Agent(agent_options_factory, AgentResponse)

    model_info = f" (model: {args.model})" if args.model else " (model: opus)"
    print(f"Agent configured{model_info}")

    results = await evaluate_full(
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
            score = score_sealqa(r.question, str(r.ground_truth), str(r.trace.output.final_answer))
            if score > 0:
                correct += 1

    print(f"\n{'='*50}")
    print(f"Total completed: {len(all_results)}/{len(data)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed indices: {[r.index for r in failed]}")
    print(f"Accuracy: {correct}/{len(successful)} ({correct/len(successful)*100:.1f}%)" if successful else "Accuracy: N/A (no successful results)")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
