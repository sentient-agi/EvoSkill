import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

from tqdm.asyncio import tqdm_asyncio

from src.harness import Agent, AgentTrace

if TYPE_CHECKING:
    from src.cache import RunCache

T = TypeVar("T")


@dataclass
class EvalResult(Generic[T]):
    """Result of evaluating a single question."""
    question: str
    ground_truth: str
    trace: AgentTrace[T] | None


async def evaluate_agent_parallel(
    agent: Agent[T],
    items: list[tuple[str, str]],
    max_concurrent: int = 2,
    *,
    cache: "RunCache | None" = None,
) -> list[EvalResult[T]]:
    """
    Run agent on multiple questions in parallel.

    Args:
        agent: The agent to evaluate
        items: List of (question, ground_truth) tuples
        max_concurrent: Max concurrent agent runs (default 2)
        cache: Optional RunCache for caching results (keys on git tree hash)

    Returns:
        List of EvalResult containing question, ground_truth, and trace
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def run_one(question: str, ground_truth: str) -> EvalResult[T]:
        async with semaphore:
            try:
                async with asyncio.timeout(1020):  # 17-minute hard limit per eval
                    # Check cache first
                    trace = None
                    if cache is not None:
                        trace = cache.get(question, agent.response_model)

                    # Cache miss - run agent
                    if trace is None:
                        trace = await agent.run(question)
                        # Store in cache
                        if cache is not None:
                            cache.set(question, trace)

            except asyncio.TimeoutError:
                print(f"Eval timed out (17min) for: {question[:50]}...")
                trace = None
            except Exception as e:
                print(f"Failed on question: {question[:50]}... Error: {e}")
                trace = None
            return EvalResult(question=question, ground_truth=ground_truth, trace=trace)

    tasks = [run_one(q, gt) for q, gt in items]
    results = await tqdm_asyncio.gather(*tasks, desc="Evaluating")
    return results
