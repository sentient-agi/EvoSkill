import asyncio
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from tqdm.asyncio import tqdm_asyncio

from src.harness import Agent, AgentTrace

T = TypeVar("T")


@dataclass
class IndexedEvalResult(Generic[T]):
    """Result of evaluating a single question with dataset index."""

    index: int  # Dataset row index
    question: str
    ground_truth: str
    trace: AgentTrace[T] | None
    error: str | None  # Error message if failed


def load_results(path: Path) -> list[IndexedEvalResult]:
    """Load results from pkl file."""
    if not path.exists():
        return []
    with open(path, "rb") as f:
        return pickle.load(f)


def get_successful_indices(path: Path) -> set[int]:
    """Get set of indices that completed successfully (no error)."""
    results = load_results(path)
    return {
        r.index for r in results
        if r.error is None and (r.trace is None or not r.trace.is_error)
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

        # Remove failed results from file so they can be replaced
        failed_indices = {i for i, _, _ in items_to_run}
        existing = load_results(output_path)
        kept_results = [r for r in existing if r.index not in failed_indices]
        if len(kept_results) < len(existing):
            # Some failed results were removed, save the cleaned file
            with open(output_path, "wb") as f:
                pickle.dump(kept_results, f)
            print(f"Removed {len(existing) - len(kept_results)} failed results for re-run")

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
                async with asyncio.timeout(1020):  # 17-minute hard limit per eval
                    trace = await agent.run(question)
            except asyncio.TimeoutError:
                error = "TimeoutError: Eval timed out after 17 minutes"
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

            # Append to file immediately (thread-safe)
            async with lock:
                existing = load_results(output_path)
                existing.append(result)
                with open(output_path, "wb") as f:
                    pickle.dump(existing, f)

            return result

    tasks = [run_one(idx, q, gt) for idx, q, gt in items]
    results = await tqdm_asyncio.gather(*tasks, desc="Evaluating")
    return results
