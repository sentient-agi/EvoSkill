import asyncio
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from tqdm.asyncio import tqdm_asyncio

from src.agent_profiles.base import Agent, AgentTrace

T = TypeVar("T")


@dataclass
class IndexedEvalResult(Generic[T]):
    """Result of evaluating a single question with dataset index."""

    index: int  # Dataset row index
    question: str
    ground_truth: str
    trace: AgentTrace[T] | None
    error: str | None  # Error message if failed
    score: float | None = None  # Cached score from grader (None = not yet scored)


def load_results(path: Path) -> list[IndexedEvalResult]:
    """Load results from pkl file."""
    if not path.exists():
        return []
    with open(path, "rb") as f:
        return pickle.load(f)


def get_successful_indices(path: Path) -> set[int]:
    """Get set of indices that completed successfully (no error AND has output)."""
    results = load_results(path)
    return {
        r.index for r in results
        if r.error is None
        and r.trace is not None
        and not r.trace.is_error
        and (r.trace.result or (r.trace.output and r.trace.output.final_answer))
    }


async def evaluate_full(
    agent: Agent[T],
    items: list[tuple[int, str, str]],  # (index, question, ground_truth)
    output_path: Path,
    max_concurrent: int = 5,
    resume: bool = True,
) -> list[IndexedEvalResult[T]]:
    """
    Run agent on multiple questions in parallel, saving incrementally.

    Args:
        agent: The agent to evaluate
        items: List of (index, question, ground_truth) tuples
        output_path: Path to save pkl results
        max_concurrent: Max concurrent agent runs (default 5)
        resume: If True, skip already-processed indices

    Returns:
        List of IndexedEvalResult (only newly processed if resuming)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter out already successful indices if resuming (re-run failed ones)
    if resume:
        successful = get_successful_indices(output_path)
        items_to_run = [(i, q, gt) for i, q, gt in items if i not in successful]
        items = items_to_run
        if successful:
            print(f"Resuming: {len(successful)} successful, {len(items)} to run")

    if not items:
        print("All items already processed!")
        return []

    semaphore = asyncio.Semaphore(max_concurrent)
    lock = asyncio.Lock()

    async def run_one(
        index: int, question: str, ground_truth: str
    ) -> IndexedEvalResult[T]:
        async with semaphore:
            error = None
            trace = None
            try:
                async with asyncio.timeout(2400):  # 40-minute hard limit per eval
                    trace = await agent.run(question)
            except asyncio.TimeoutError:
                error = "TimeoutError: Eval timed out after 40 minutes"
                print(f"[TIMEOUT] Index {index}: {question[:50]}...")
            except Exception as e:
                error = f"{type(e).__name__}: {str(e)}"
                print(f"[ERROR] Index {index}: {question[:50]}... -> {error}")

            result = IndexedEvalResult(
                index=index,
                question=question,
                ground_truth=ground_truth,
                trace=trace,
                error=error,
            )

            # Save result: replace old entry for same index only if new one is better
            async with lock:
                existing = load_results(output_path)
                new_has_output = (
                    result.error is None
                    and result.trace is not None
                    and (result.trace.result or (result.trace.output and result.trace.output.final_answer))
                )
                # Check if we already have a good result for this index
                old = next((r for r in existing if r.index == index), None)
                old_has_output = (
                    old is not None
                    and old.error is None
                    and old.trace is not None
                    and (old.trace.result or (old.trace.output and old.trace.output.final_answer))
                )
                if old is None:
                    # New index, just append
                    existing.append(result)
                elif new_has_output or not old_has_output:
                    # Replace only if new is good, or old was also bad
                    existing = [r for r in existing if r.index != index]
                    existing.append(result)
                # else: keep old good result, discard new bad one

                with open(output_path, "wb") as f:
                    pickle.dump(existing, f)

            return result

    tasks = [run_one(idx, q, gt) for idx, q, gt in items]
    results = await tqdm_asyncio.gather(*tasks, desc="Evaluating")
    return results
