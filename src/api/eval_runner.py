"""EvalRunner — standalone evaluation API.

Usage:
    from src.api import EvalRunner
    summary = await EvalRunner(dataset="data.csv", task="sealqa", model="sonnet").run()
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.harness import Agent
from src.evaluation.eval_full import evaluate_full, load_results
from src.schemas import AgentResponse

from .task_registry import TaskConfig, get_task


@dataclass
class EvalSummary:
    """Summary of an evaluation run."""

    total: int
    successful: int
    failed: int
    correct: int
    accuracy: float
    output_path: Path
    failed_indices: list[int]


class EvalRunner:
    """Run standalone evaluation with minimal boilerplate.

    Args:
        dataset: Path to the dataset CSV file. If not provided, uses the task's default.
        task: Registered task name (e.g. "base", "dabstep", "sealqa").
        model: Model for the agent (e.g. "opus", "sonnet", "haiku").
        output: Path for the output pkl file.
        max_concurrent: Max concurrent agent runs.
        resume: If True, skip already-processed indices.
        num_samples: Limit to first N samples (None = all).
        task_config: Custom TaskConfig, overrides the task name lookup.
    """

    def __init__(
        self,
        dataset: str | None = None,
        task: str = "base",
        *,
        model: str | None = None,
        output: str = "results/eval_results.json",
        max_concurrent: int = 8,
        resume: bool = True,
        num_samples: int | None = None,
        task_config: TaskConfig | None = None,
    ) -> None:
        self._task_config = task_config or get_task(task)
        self._dataset_path = dataset or self._task_config.default_dataset
        if not self._dataset_path:
            raise ValueError(
                f"No dataset provided and task {task!r} has no default_dataset."
            )
        self._model = model
        self._output = Path(output)
        self._max_concurrent = max_concurrent
        self._resume = resume
        self._num_samples = num_samples

    def _load_items(self) -> tuple[pd.DataFrame, list[tuple[int, str, str]]]:
        """Load dataset and prepare (index, question, answer) items."""
        data = pd.read_csv(self._dataset_path)
        if self._task_config.column_renames:
            data.rename(columns=self._task_config.column_renames, inplace=True)

        if self._num_samples is not None:
            data = data.head(self._num_samples)

        question_col = self._task_config.question_col
        answer_col = self._task_config.answer_col

        items = [
            (idx, row[question_col], str(row[answer_col]))
            for idx, row in data.iterrows()
        ]
        return data, items

    def _score_results(
        self, all_results: list,
    ) -> tuple[int, list[int]]:
        """Score results and return (correct_count, failed_indices)."""
        scorer = self._task_config.scorer
        correct = 0
        failed_indices = []

        for r in all_results:
            if r.error is not None:
                failed_indices.append(r.index)
                continue
            if r.trace and r.trace.output and r.trace.output.final_answer:
                if scorer is not None:
                    score = scorer(
                        r.question,
                        str(r.trace.output.final_answer),
                        str(r.ground_truth),
                    )
                    if score > 0:
                        correct += 1
                else:
                    # Without a scorer, we count successful completions
                    correct += 1

        return correct, failed_indices

    async def run(self) -> EvalSummary:
        """Run the evaluation.

        Returns:
            EvalSummary with counts, accuracy, and output path.
        """
        data, items = self._load_items()
        print(f"Dataset: {len(data)} samples")

        agent_options = self._task_config.make_agent_options(model=self._model)
        agent = Agent(agent_options, AgentResponse)

        model_info = f" (model: {self._model})" if self._model else ""
        print(f"Agent configured{model_info}")

        await evaluate_full(
            agent=agent,
            items=items,
            output_path=self._output,
            max_concurrent=self._max_concurrent,
            resume=self._resume,
        )

        all_results = load_results(self._output)
        successful = [r for r in all_results if r.error is None]
        failed = [r for r in all_results if r.error is not None]
        correct, failed_indices = self._score_results(all_results)

        accuracy = correct / len(successful) if successful else 0.0

        print(f"\n{'=' * 50}")
        print(f"Total completed: {len(all_results)}/{len(data)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        if failed_indices:
            print(f"Failed indices: {failed_indices}")
        if successful:
            print(
                f"Accuracy: {correct}/{len(successful)} ({accuracy * 100:.1f}%)"
            )
        print(f"Results saved to: {self._output}")

        return EvalSummary(
            total=len(all_results),
            successful=len(successful),
            failed=len(failed),
            correct=correct,
            accuracy=accuracy,
            output_path=self._output,
            failed_indices=failed_indices,
        )

    def run_sync(self) -> EvalSummary:
        """Synchronous wrapper around run(). Calls asyncio.run() internally."""
        return asyncio.run(self.run())
