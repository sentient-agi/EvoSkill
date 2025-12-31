"""Self-improving agent loop runner."""

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from src.agent_profiles.base import Agent, AgentTrace
from src.cache import RunCache, CacheConfig


def _log(phase: str, message: str = "", indent: int = 0) -> None:
    """Print a structured log message.

    Args:
        phase: Phase marker (e.g., "INIT", "ITER 1/5", "DONE") or empty for continuation.
        message: The message to display.
        indent: Indentation level (each level = 2 spaces).
    """
    prefix = "  " * indent
    if phase:
        print(f"\n{prefix}[{phase}] {message}")
    else:
        print(f"{prefix}{message}")


def _score_multi_tolerance(predicted: str, ground_truth: str) -> float:
    """Score answer using multiple tolerance levels and return average."""
    scores = [score_answer(predicted, ground_truth, tol) for tol in TOLERANCE_LEVELS]
    return sum(scores) / len(scores)


from src.agent_profiles.base_agent import get_base_agent_options
from src.agent_profiles.skill_generator import get_project_root
from src.evaluation import score_answer, evaluate_agent_parallel
from src.registry import ProgramManager, ProgramConfig
from src.schemas import (
    AgentResponse,
    ProposerResponse,
    ToolGeneratorResponse,
    PromptGeneratorResponse,
)

from .config import LoopConfig
from .helpers import (
    build_proposer_query,
    build_skill_query,
    build_prompt_query,
    append_feedback,
    read_feedback_history,
    update_prompt_file,
)


T = TypeVar("T")

TOLERANCE_LEVELS = [0.05, 0.01, 0.1, 0.0, 0.025]


@dataclass
class LoopAgents:
    """Container for the 4 agents used in the loop."""

    base: Agent[AgentResponse]
    proposer: Agent[ProposerResponse]
    skill_generator: Agent[ToolGeneratorResponse]
    prompt_generator: Agent[PromptGeneratorResponse]


@dataclass
class LoopResult:
    """Result of running the self-improving loop."""

    frontier: list[tuple[str, float]]
    best_program: str
    best_score: float
    iterations_completed: int


class SelfImprovingLoop:
    """Self-improving agent loop with git-based versioning.

    This class encapsulates the self-improving loop where:
    1. Base agent attempts to answer questions
    2. Failures are passed to the proposer to suggest skills or prompt changes
    3. Skill/prompt generator creates the proposed changes
    4. New mutations are evaluated and added to frontier if improved
    5. Loop continues until threshold or max iterations
    """

    def __init__(
        self,
        config: LoopConfig,
        agents: LoopAgents,
        manager: ProgramManager,
        train_data: list[tuple[str, str]],
        val_data: list[tuple[str, str]],
    ):
        """Initialize the self-improving loop.

        Args:
            config: Loop configuration parameters.
            agents: Container with the 4 agents (base, proposer, skill_generator, prompt_generator).
            manager: ProgramManager for git-based versioning.
            train_data: Training data as list of (question, answer) tuples.
            val_data: Validation data as list of (question, answer) tuples.
        """
        self.config = config
        self.agents = agents
        self.manager = manager
        self.train_data = train_data
        self.val_data = val_data

        # Paths
        self._project_root = Path(get_project_root())
        self._feedback_path = self._project_root / ".claude" / "feedback_history.md"
        self._prompt_path = (
            self._project_root / "src" / "agent_profiles" / "base_agent" / "prompt.txt"
        )

        # Initialize cache
        if config.cache_enabled:
            cache_config = CacheConfig(
                cache_dir=config.cache_dir,
                enabled=True,
                store_messages=config.cache_store_messages,
                cwd=self._project_root,
            )
            self.cache: RunCache | None = RunCache(cache_config)
        else:
            self.cache = None

    async def run(self) -> LoopResult:
        """Run the full self-improving loop.

        Returns:
            LoopResult with frontier, best program, and iteration count.
        """
        # 0. Reset feedback history if configured
        if self.config.reset_feedback and self._feedback_path.exists():
            self._feedback_path.unlink()

        # 1. Create and evaluate base program if needed
        await self._ensure_base_program()

        # 2. Main loop
        no_improvement_count = 0
        iteration_count = 0

        for i in range(self.config.max_iterations):
            iteration_count = i + 1

            # Select best parent from frontier
            parent = self._get_best_parent()
            self.manager.switch_to(parent)
            _log(f"ITER {iteration_count}/{self.config.max_iterations}", f"Parent: {parent}")

            # Pick a sample question
            question, answer = self.train_data[i % len(self.train_data)]
            _log("", f"  Question: {question[:70]}...")

            # Run agent
            trace = await self.agents.base.run(question)
            agent_answer = (
                trace.output.final_answer if trace.output else "[PARSE FAILED]"
            )

            # Check if correct (average across multiple tolerance levels)
            avg_score = _score_multi_tolerance(
                agent_answer.strip().lower(),
                answer.strip().lower(),
            )
            if avg_score == 1.0:
                _log("", f"  \u2713 Correct")
                continue

            _log("", f"  \u2717 Incorrect (got: \"{agent_answer[:40]}...\", expected: \"{answer[:40]}...\")")

            # Run proposer to suggest improvement
            child_name = await self._mutate(parent, trace, answer, iteration_count)

            if child_name is None:
                no_improvement_count += 1
            else:
                # Evaluate child
                _log("", f"  \u2192 Evaluating {child_name}...")
                child_score = await self._evaluate(self.val_data)

                # Update frontier or discard
                added = self.manager.update_frontier(
                    child_name, child_score, max_size=self.config.frontier_size
                )

                if added:
                    _log("", f"  \u2713 Added to frontier (score: {child_score:.4f})")
                    no_improvement_count = 0
                else:
                    _log("", f"  \u2717 Discarded (score: {child_score:.4f})")
                    self.manager.discard(child_name)
                    no_improvement_count += 1

            # Check early stopping
            if no_improvement_count >= self.config.no_improvement_limit:
                _log("STOP", f"No improvement for {self.config.no_improvement_limit} iterations")
                break

            # Print frontier status
            frontier_str = ", ".join(f"{n}:{s:.2f}" for n, s in self.manager.get_frontier_with_scores())
            _log("", f"  Frontier: [{frontier_str}]")

        # 3. Return results
        frontier = self.manager.get_frontier_with_scores()
        best = self.manager.get_best_from_frontier()
        best_score = frontier[0][1] if frontier else 0.0

        _log("DONE", f"{iteration_count} iterations, best: {best or 'base'} ({best_score:.4f})")

        return LoopResult(
            frontier=frontier,
            best_program=best or "base",
            best_score=best_score,
            iterations_completed=iteration_count,
        )

    async def _ensure_base_program(self) -> None:
        """Create and evaluate base program if it doesn't exist."""
        if "base" not in self.manager.list_programs():
            current_options = get_base_agent_options()

            base_config = ProgramConfig(
                name="base",
                parent=None,
                generation=0,
                system_prompt=current_options.system_prompt,
                allowed_tools=current_options.allowed_tools,
                output_format=current_options.output_format,
                metadata={},
            ).with_timestamp()

            self.manager.create_program("base", base_config)
            _log("INIT", "Created base program")
        else:
            _log("INIT", "Using existing base program")

        # Evaluate and add base to frontier
        self.manager.switch_to("base")
        _log("", f"  \u2192 Evaluating on {len(self.val_data)} samples...")
        base_score = await self._evaluate(self.val_data)
        self.manager.update_frontier(
            "base", base_score, max_size=self.config.frontier_size
        )
        _log("", f"  \u2192 Base score: {base_score:.4f}")
        _log("", f"  \u2192 Frontier: {self.manager.get_frontier()}")

    async def _evaluate(self, data: list[tuple[str, str]]) -> float:
        """Evaluate base agent on data.

        Args:
            data: List of (question, answer) tuples.

        Returns:
            Accuracy score (0.0 to 1.0).
        """
        results = await evaluate_agent_parallel(
            self.agents.base, data, max_concurrent=self.config.concurrency, cache=self.cache
        )

        score = 0.0
        for result in results:
            score += _score_multi_tolerance(
                result.trace.output.final_answer,
                result.ground_truth,
            )
        return score / len(results)

    async def _mutate(
        self,
        parent: str,
        trace: AgentTrace[AgentResponse],
        answer: str,
        iteration: int,
    ) -> str | None:
        """Run proposer and generator to create a mutation.

        Args:
            parent: Name of the parent program.
            trace: Agent trace from the failed attempt.
            answer: Ground truth answer.
            iteration: Current iteration number.

        Returns:
            Child program name if created, None otherwise.
        """
        # Run proposer
        _log("", f"  \u2192 Running proposer...")
        feedback_history = read_feedback_history(self._feedback_path)
        proposer_query = build_proposer_query(trace, answer, feedback_history)
        proposer_trace = await self.agents.proposer.run(proposer_query)

        if proposer_trace.output is None:
            _log("", f"  \u26a0 Proposer failed: {proposer_trace.parse_error}")
            return None

        prompt_or_skill = proposer_trace.output.optimize_prompt_or_skill
        proposed = proposer_trace.output.proposed_skill_or_prompt
        justification = proposer_trace.output.justification

        _log("", f"  \u2192 Proposal: {prompt_or_skill} - {proposed[:50]}...")

        # Create child program branch
        child_name = f"iter-{iteration}"
        parent_config = self.manager.get_current()
        original_prompt = parent_config.system_prompt
        child_config = parent_config.mutate(child_name)
        self.manager.create_program(child_name, child_config, parent=parent)

        # Generate skill or prompt
        if prompt_or_skill == "prompt":
            _log("", f"  \u2192 Generating optimized prompt...")
            prompt_query = build_prompt_query(proposer_trace, original_prompt)
            prompt_trace = await self.agents.prompt_generator.run(prompt_query)
            if prompt_trace.output:
                update_prompt_file(
                    self._prompt_path, prompt_trace.output.optimized_prompt
                )
        else:
            _log("", f"  \u2192 Generating skill...")
            skill_query = build_skill_query(proposer_trace)
            skill_trace = await self.agents.skill_generator.run(skill_query)
            if skill_trace.output:
                pass  # Skill is written to file by the generator

        # Commit changes
        self.manager.commit(f"{child_name}: {proposed[:50]}")

        # Append to feedback history
        append_feedback(self._feedback_path, child_name, proposed, justification)

        return child_name

    def _get_best_parent(self) -> str:
        """Get best program from frontier, or 'base' if frontier is empty."""
        best = self.manager.get_best_from_frontier()
        return best if best else "base"
