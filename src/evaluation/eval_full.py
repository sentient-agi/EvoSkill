import asyncio
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generic, TypeVar

from tqdm.asyncio import tqdm_asyncio

from src.harness import Agent, AgentTrace
from src.harness.utils import (
    eval_run_label as _eval_run_label,
    eval_run_uid as _eval_run_uid,
    eval_run_index as _eval_run_index,
    eval_run_ground_truth as _eval_run_ground_truth,
)

T = TypeVar("T")


@dataclass
class IndexedEvalResult(Generic[T]):
    """Result of evaluating a single question with dataset index."""

    index: int  # Dataset row index
    question: str
    ground_truth: str
    trace: AgentTrace[T] | None
    error: str | None  # Error message if failed
    uid: str | None = None  # Optional dataset UID (e.g. "UID0042"); used as span label for Phoenix.


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


def _normalize_items(items):
    """Accept either (index, question, gt) 3-tuples or (index, uid, question, gt)
    4-tuples and return a uniform list of (index, uid_or_None, question, gt)."""
    out = []
    for it in items:
        if len(it) == 4:
            i, uid, q, gt = it
            out.append((i, uid, q, gt))
        else:
            i, q, gt = it
            out.append((i, None, q, gt))
    return out


async def evaluate_full(
    agent: Agent[T],
    items: list[tuple[int, str, str]] | list[tuple[int, str, str, str]],
    output_path: Path,
    max_concurrent: int = 5,
    resume: bool = True,
) -> list[IndexedEvalResult[T]]:
    """
    Run agent on multiple questions in parallel, saving incrementally.

    Args:
        agent: The agent to evaluate
        items: Either (index, question, ground_truth) 3-tuples or
            (index, uid, question, ground_truth) 4-tuples. When uid is
            provided, the agent.run root span is named `eval:{uid}` so
            traces are findable per-question in Phoenix; otherwise the
            executor's stock `agent.run:<agent>` name is used.
        output_path: Path to save pkl results
        max_concurrent: Max concurrent agent runs (default 5)
        resume: If True, skip already-processed indices

    Returns:
        List of IndexedEvalResult (only newly processed if resuming)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    items = _normalize_items(items)

    # Filter out already successful indices if resuming (re-run failed ones)
    if resume:
        successful = get_successful_indices(output_path)
        items_to_run = [t for t in items if t[0] not in successful]

        # Remove failed results from file so they can be replaced
        failed_indices = {t[0] for t in items_to_run}
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
        index: int, uid: str | None, question: str, ground_truth: str
    ) -> IndexedEvalResult[T]:
        async with semaphore:
            error = None
            trace = None
            # Stash uid + index in contextvars so the SDK executor can name
            # the agent.run root span `eval:{uid}` and tag it with the index.
            # No-op when uid is None.
            label_token = _eval_run_label.set(f"eval:{uid}" if uid else None)
            uid_token = _eval_run_uid.set(uid)
            idx_token = _eval_run_index.set(index)
            gt_token = _eval_run_ground_truth.set(ground_truth)
            try:
                async with asyncio.timeout(840):  # 14-min hard limit (matches Agent.TIMEOUT_SECONDS)
                    trace = await agent.run(question)
            except asyncio.TimeoutError:
                error = "TimeoutError: Eval timed out after 14 minutes"
                tag = uid or f"index {index}"
                print(f"[TIMEOUT] {tag}: {question[:50]}...")
            except Exception as e:
                error = f"{type(e).__name__}: {str(e)}"
                tag = uid or f"index {index}"
                print(f"[ERROR] {tag}: {question[:50]}... -> {error}")
            finally:
                _eval_run_label.reset(label_token)
                _eval_run_uid.reset(uid_token)
                _eval_run_index.reset(idx_token)
                _eval_run_ground_truth.reset(gt_token)

            result = IndexedEvalResult(
                index=index,
                question=question,
                ground_truth=ground_truth,
                trace=trace,
                error=error,
                uid=uid,
            )

            # Append to file immediately (thread-safe)
            async with lock:
                existing = load_results(output_path)
                existing.append(result)
                with open(output_path, "wb") as f:
                    pickle.dump(existing, f)

            return result

    tasks = [run_one(idx, uid, q, gt) for idx, uid, q, gt in items]
    results = await tqdm_asyncio.gather(*tasks, desc="Evaluating")
    return results
