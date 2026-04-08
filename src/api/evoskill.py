"""EvoSkill — high-level API for the self-improvement loop.

Usage:
    from src.api import EvoSkill
    result = await EvoSkill(dataset="data.csv", task="dabstep").run()
"""

from __future__ import annotations

import asyncio
from typing import Callable, Any

from src.agent_profiles import Agent
from src.agent_profiles.skill_generator import get_project_root
from src.agent_profiles.skill_proposer.skill_proposer import (
    skill_proposer_options,
    make_skill_proposer_options,
)
from src.agent_profiles.skill_generator.skill_generator import (
    skill_generator_options,
    make_skill_generator_options,
)
from src.agent_profiles.prompt_proposer.prompt_proposer import (
    prompt_proposer_options,
    make_prompt_proposer_options,
)
from src.agent_profiles.prompt_generator.prompt_generator import (
    prompt_generator_options,
    make_prompt_generator_options,
)
from src.loop import SelfImprovingLoop, LoopConfig, LoopAgents, LoopResult
from src.registry import ProgramManager
from src.schemas import (
    AgentResponse,
    SkillProposerResponse,
    PromptProposerResponse,
    ToolGeneratorResponse,
    PromptGeneratorResponse,
)

from .data_utils import load_dataset, stratified_split
from .task_registry import TaskConfig, get_task, ScorerFn


class EvoSkill:
    """Run the self-improvement loop with minimal boilerplate.

    Args:
        dataset: Path to the dataset CSV file. If not provided, uses the task's default.
        task: Registered task name (e.g. "base", "dabstep", "sealqa").
        model: Model for the base agent (e.g. "opus", "sonnet", "haiku").
        mode: Evolution mode — "skill_only" or "prompt_only".
        harness: Execution harness — "claude", "opencode", or "openhands".
        max_iterations: Maximum number of improvement iterations.
        frontier_size: Number of top-performing programs to keep.
        no_improvement_limit: Stop after this many iterations without improvement.
        concurrency: Number of concurrent evaluations.
        train_ratio: Fraction of each category to use for training.
        val_ratio: Fraction of each category to use for validation.
        continue_mode: If True, continue from existing frontier instead of starting fresh.
        cache_enabled: Whether to enable run caching.
        reset_feedback: Whether to reset feedback_history.md on fresh loop run.
        failure_samples: Number of failure samples per iteration.
        selection_strategy: Parent selection from frontier — "best" (greedy, default),
            "random" (uniform random), or "round_robin" (cycle through ranked members).
        scorer: Custom scorer function, overrides the task's default.
        task_config: Custom TaskConfig, overrides the task name lookup.
    """

    def __init__(
        self,
        dataset: str | None = None,
        task: str = "base",
        *,
        model: str | None = None,
        mode: str = "skill_only",
        harness: str = "claude",
        max_iterations: int = 20,
        frontier_size: int = 3,
        no_improvement_limit: int = 5,
        concurrency: int = 4,
        train_ratio: float = 0.18,
        val_ratio: float = 0.12,
        continue_mode: bool = False,
        cache_enabled: bool = True,
        reset_feedback: bool = True,
        failure_samples: int = 3,
        selection_strategy: str = "best",
        scorer: ScorerFn | None = None,
        task_config: TaskConfig | None = None,
    ) -> None:
        self._task_config = task_config or get_task(task)
        self._dataset_path = dataset or self._task_config.default_dataset
        if not self._dataset_path:
            raise ValueError(
                f"No dataset provided and task {task!r} has no default_dataset."
            )
        self._model = model
        self._mode = mode
        self._harness = harness
        self._max_iterations = max_iterations
        self._frontier_size = frontier_size
        self._no_improvement_limit = no_improvement_limit
        self._concurrency = concurrency
        self._train_ratio = train_ratio
        self._val_ratio = val_ratio
        self._continue_mode = continue_mode
        self._cache_enabled = cache_enabled
        self._reset_feedback = reset_feedback
        self._failure_samples = failure_samples
        self._selection_strategy = selection_strategy
        self._scorer_override = scorer

    def _build_config(self) -> LoopConfig:
        """Create a LoopConfig from constructor params."""
        return LoopConfig(
            max_iterations=self._max_iterations,
            frontier_size=self._frontier_size,
            no_improvement_limit=self._no_improvement_limit,
            concurrency=self._concurrency,
            evolution_mode=self._mode,
            failure_sample_count=self._failure_samples,
            categories_per_batch=self._failure_samples,
            cache_enabled=self._cache_enabled,
            reset_feedback=self._reset_feedback,
            continue_mode=self._continue_mode,
            selection_strategy=self._selection_strategy,
            harness=self._harness,
        )

    def _build_agents(self) -> LoopAgents:
        """Assemble LoopAgents from task config."""
        base_options = self._task_config.make_agent_options(model=self._model)
        return LoopAgents(
            base=Agent(base_options, AgentResponse),
            skill_proposer=Agent(
                make_skill_proposer_options(self._harness, self._model),
                SkillProposerResponse,
            ),
            prompt_proposer=Agent(
                make_prompt_proposer_options(self._harness, self._model),
                PromptProposerResponse,
            ),
            skill_generator=Agent(
                make_skill_generator_options(self._harness, self._model),
                ToolGeneratorResponse,
            ),
            prompt_generator=Agent(
                make_prompt_generator_options(self._harness, self._model),
                PromptGeneratorResponse,
            ),
        )

    def _load_data(
        self,
    ) -> tuple[dict[str, list[tuple[str, str]]], list[tuple[str, str, str]]]:
        """Load dataset and split into train/val."""
        data = load_dataset(self._dataset_path, self._task_config)
        return stratified_split(data, self._train_ratio, self._val_ratio)

    @property
    def dataset_info(self) -> dict:
        """Preview categories and splits without running the loop."""
        data = load_dataset(self._dataset_path, self._task_config)
        data = data.dropna(subset=["category"])
        categories = data["category"].unique().tolist()
        cat_counts = data["category"].value_counts().to_dict()
        return {
            "dataset": self._dataset_path,
            "task": self._task_config.name,
            "total_rows": len(data),
            "categories": categories,
            "category_counts": cat_counts,
            "train_ratio": self._train_ratio,
            "val_ratio": self._val_ratio,
        }

    async def run(self, max_iterations: int | None = None) -> LoopResult:
        """Run the self-improvement loop.

        Args:
            max_iterations: Override the max_iterations set in constructor.

        Returns:
            LoopResult with frontier, best program, best score, and iterations completed.
        """
        config = self._build_config()
        if max_iterations is not None:
            config.max_iterations = max_iterations

        agents = self._build_agents()
        manager = ProgramManager(cwd=get_project_root())
        train_pools, val_data = self._load_data()

        # Print summary
        total_train = sum(len(pool) for pool in train_pools.values())
        print(f"Dataset: {self._dataset_path}")
        print(
            f"Categories ({len(train_pools)}): {', '.join(train_pools.keys())}"
        )
        print(f"Total training samples: {total_train}")
        print(f"Validation samples: {len(val_data)}")
        print(f"Running loop with evolution_mode={self._mode}, harness={self._harness}")

        scorer = self._scorer_override or self._task_config.scorer
        loop = SelfImprovingLoop(
            config, agents, manager, train_pools, val_data, scorer=scorer
        )
        result = await loop.run()

        print(f"Best: {result.best_program} ({result.best_score:.2%})")
        print(f"Frontier: {result.frontier}")
        return result

    def run_sync(self, max_iterations: int | None = None) -> LoopResult:
        """Synchronous wrapper around run(). Calls asyncio.run() internally."""
        return asyncio.run(self.run(max_iterations=max_iterations))
