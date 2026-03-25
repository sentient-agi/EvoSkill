#!/usr/bin/env python3
"""Run full evaluation on SEAL-QA dataset."""
import argparse
import asyncio
import shutil
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from src.agent_profiles import Agent, make_sealqa_agent_options, set_sdk
from src.agent_profiles.skill_generator import get_project_root
from src.api.data_utils import stratified_split
from src.evaluation.eval_full import evaluate_full, load_results
from src.evaluation.sealqa_scorer import score_sealqa
from src.schemas import AgentResponse

SKILLS_DIR = Path(get_project_root()) / ".claude" / "skills"
SKILLS_HIDDEN = SKILLS_DIR.parent / "_skills_hidden"

# Skill names that are meta-tools (not task-specific evolved skills)
META_SKILLS = {"skill-creator", "brainstorming"}


@contextmanager
def hide_skills():
    """Temporarily hide .claude/skills/ so the agent runs without evolved skills."""
    if not SKILLS_DIR.exists():
        yield
        return
    SKILLS_HIDDEN.mkdir(parents=True, exist_ok=True)
    moved = []
    for skill_dir in SKILLS_DIR.iterdir():
        if skill_dir.is_dir() and skill_dir.name not in META_SKILLS:
            dest = SKILLS_HIDDEN / skill_dir.name
            shutil.move(str(skill_dir), str(dest))
            moved.append(skill_dir.name)
    if moved:
        print(f"Hidden skills for baseline: {moved}")
    try:
        yield
    finally:
        for name in moved:
            src = SKILLS_HIDDEN / name
            if src.exists():
                shutil.move(str(src), str(SKILLS_DIR / name))
        if SKILLS_HIDDEN.exists():
            shutil.rmtree(str(SKILLS_HIDDEN), ignore_errors=True)


def list_active_skills() -> list[str]:
    """List non-meta skills currently on disk."""
    if not SKILLS_DIR.exists():
        return []
    return [
        d.name for d in SKILLS_DIR.iterdir()
        if d.is_dir() and d.name not in META_SKILLS and (d / "SKILL.md").exists()
    ]


async def main():
    parser = argparse.ArgumentParser(description="Evaluate agent on SEAL-QA dataset")
    parser.add_argument(
        "--dataset", "-d", type=Path,
        default=Path(".dataset/seal-0.csv"),
        help="Path to SEAL-QA CSV file (default: .dataset/seal-0.csv)",
    )
    parser.add_argument(
        "--output", "-o", type=Path,
        default=Path("results/sealqa_eval_results.pkl"),
        help="Output pkl file path",
    )
    parser.add_argument(
        "--max-concurrent", "-c", type=int, default=1,
        help="Max concurrent evaluations (default: 1)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Don't resume from existing results (start fresh)",
    )
    parser.add_argument(
        "--topic", "-t", type=str, default="all",
        help="Filter by topic column ('all' or a specific topic value)",
    )
    parser.add_argument(
        "--num-samples", "-n", type=int, default=None,
        help="Limit to first N samples (default: all)",
    )
    parser.add_argument(
        "--offset", type=int, default=0,
        help="Skip first N questions (default: 0)",
    )
    parser.add_argument(
        "--model", "-m", type=str, required=True,
        help="Model ID (e.g., claude-opus-4-5-20251101, gemini-3.1-flash-lite-preview, gpt-oss-120b)",
    )
    parser.add_argument(
        "--provider", type=str, default=None,
        help="Provider ID for opencode SDK (e.g., gemini, arc). Required when --sdk=opencode.",
    )
    parser.add_argument(
        "--sdk", type=str, choices=["claude", "opencode"], default="claude",
        help="SDK to use: 'claude' or 'opencode' (default: claude)",
    )
    parser.add_argument(
        "--held-out", action="store_true",
        help="Evaluate only on the held-out test set (excludes train/val samples)",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.12,
        help="Train ratio for stratified split (default: 0.12 -> 14 samples)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.12,
        help="Val ratio for stratified split (default: 0.12 -> 14 samples)",
    )
    parser.add_argument(
        "--no-skills", action="store_true",
        help="Run baseline without evolved skills (temporarily hides .claude/skills/)",
    )
    args = parser.parse_args()

    # Set SDK
    set_sdk(args.sdk)

    # Load dataset
    data = pd.read_csv(args.dataset)

    if args.held_out:
        data.rename(columns={"topic": "category", "answer": "ground_truth"}, inplace=True)
        _train, _val, test_data = stratified_split(data, train_ratio=args.train_ratio, val_ratio=args.val_ratio)
        data = pd.DataFrame(test_data, columns=["question", "answer", "topic"])
        print(f"Held-out test set: {len(data)} samples (train={args.train_ratio:.0%}, val={args.val_ratio:.0%})")
    else:
        print(f"Full dataset: {len(data)} samples")

    # Filter by topic if requested
    if args.topic != "all":
        data = data[data["topic"] == args.topic]

    items = [
        (idx, row["question"], row["answer"])
        for idx, row in data.iterrows()
    ]

    # Apply offset and limit
    if args.offset:
        items = items[args.offset:]
    if args.num_samples is not None:
        items = items[:args.num_samples]

    # Report config
    active = list_active_skills()
    mode = "baseline (no skills)" if args.no_skills else f"skills: {active or 'none'}"
    print(f"Evaluating: {len(items)} samples (topic={args.topic}, {mode})")
    print(f"  sdk={args.sdk} model={args.model} provider={args.provider or 'default'}")

    # Create agent
    agent_options_factory = make_sealqa_agent_options(model=args.model, provider=args.provider)
    agent = Agent(agent_options_factory, AgentResponse)

    # Run evaluation (hide skills if baseline)
    ctx = hide_skills() if args.no_skills else contextmanager(lambda: (yield))()
    with ctx:
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
    scored = 0
    for r in successful:
        predicted = None
        if r.trace:
            if r.trace.output and r.trace.output.final_answer:
                predicted = str(r.trace.output.final_answer)
            elif r.trace.result and r.trace.result.strip():
                predicted = r.trace.result.strip()

        if predicted:
            scored += 1
            score = score_sealqa(r.question, str(r.ground_truth), predicted)
            if score > 0:
                correct += 1

    print(f"\n{'='*50}")
    print(f"Total completed: {len(all_results)}/{len(items)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed indices: {[r.index for r in failed]}")
    print(f"Accuracy: {correct}/{scored} ({correct/scored*100:.1f}%)" if scored else "Accuracy: N/A (no answers to score)")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
