"""OSWorld self-improving loop runner.

Follows the same structure as SelfImprovingLoop but uses VM-based evaluation
via EnvPool instead of Q&A text evaluation. Scoring comes from env.evaluate()
instead of text comparison.
"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.agent_profiles.base import Agent, AgentTrace
from src.registry import ProgramManager, ProgramConfig
from src.schemas import (
    SkillProposerResponse,
    PromptProposerResponse,
    ToolGeneratorResponse,
    PromptGeneratorResponse,
)
from src.loop.helpers import (
    build_proposer_query,
    build_skill_query_from_skill_proposer,
    build_prompt_query_from_prompt_proposer,
    append_feedback,
    read_feedback_history,
    update_prompt_file,
)

from .config import OSWorldLoopConfig
from .env_pool import EnvPool
from .types import OSWorldTask


def _log(phase: str, message: str = "") -> None:
    prefix = ""
    if phase:
        print(f"\n{prefix}[{phase}] {message}")
    else:
        print(f"{prefix}{message}")


@dataclass
class OSWorldLoopAgents:
    """Container for agents used in the OSWorld loop."""
    skill_proposer: Agent[SkillProposerResponse]
    prompt_proposer: Agent[PromptProposerResponse]
    skill_generator: Agent[ToolGeneratorResponse]
    prompt_generator: Agent[PromptGeneratorResponse]


@dataclass
class LoopResult:
    """Result of running the self-improving loop."""
    frontier: list[tuple[str, float]]
    best_program: str
    best_score: float
    iterations_completed: int


class OSWorldLoop:
    """Self-improving loop for OSWorld using skill evolution.

    The loop:
    1. Samples tasks from training pools (round-robin by domain)
    2. Runs tasks on VMs via EnvPool, scores via env.evaluate()
    3. Collects failures (score < threshold)
    4. Proposes skill mutations based on AgentTrace failure analysis
    5. Evaluates child program on validation tasks
    6. Updates frontier if improved
    """

    def __init__(
        self,
        config: OSWorldLoopConfig,
        agents: OSWorldLoopAgents,
        manager: ProgramManager,
        base_agent: Any,  # ClaudeCodeAgent
        env_pool: EnvPool,
        train_pools: dict[str, list[OSWorldTask]],
        val_data: list[tuple[OSWorldTask, str]],
    ):
        self.config = config
        self.agents = agents
        self.manager = manager
        self.base_agent = base_agent
        self.env_pool = env_pool
        self.train_pools = train_pools
        self.val_data = val_data

        # Round-robin sampling state
        self._category_offset = 0
        self._per_cat_offset: dict[str, int] = {
            cat: 0 for cat in train_pools.keys()
        }

        # Paths
        self._project_root = Path(manager.cwd)
        self._feedback_path = self._project_root / ".claude" / "feedback_history.md"
        self._prompt_path = None  # Set if prompt evolution is used

        # Iteration offset for continue mode
        self._iteration_offset = 0

    async def run(self) -> LoopResult:
        """Run the full self-improving loop."""
        # 0. Handle feedback reset
        if not self.config.continue_mode:
            if self.config.reset_feedback and self._feedback_path.exists():
                self._feedback_path.unlink()
            self._iteration_offset = 0

        categories = sorted(self.train_pools.keys())

        # 1. Create base program and evaluate
        if self.config.continue_mode and self.manager.get_frontier():
            best = self._get_best_parent()
            self.manager.switch_to(best)
            frontier_str = ", ".join(
                f"{n}:{s:.2f}"
                for n, s in self.manager.get_frontier_with_scores()
            )
            _log("CONTINUE", f"Using existing frontier: [{frontier_str}]")
        else:
            await self._ensure_base_program()

        # 2. Main loop
        no_improvement_count = 0
        iteration_count = 0
        n_cats = len(categories)
        consecutive_proposer_failures = 0

        for i in range(self.config.max_iterations):
            iteration_count = i + 1
            actual_iteration = iteration_count + self._iteration_offset

            parent = self._get_best_parent()
            self.manager.switch_to(parent)
            _log(
                f"ITER {iteration_count}/{self.config.max_iterations}",
                f"Parent: {parent}",
            )

            # Round-robin sampling by domain
            n_cats_this_iter = min(self.config.categories_per_batch, n_cats)
            test_tasks: list[tuple[OSWorldTask, str]] = []
            sampled_cats: list[str] = []

            for j in range(n_cats_this_iter):
                cat_idx = (self._category_offset + j) % n_cats
                cat = categories[cat_idx]
                pool = self.train_pools[cat]
                samples_to_take = min(
                    self.config.samples_per_category, len(pool)
                )

                for _ in range(samples_to_take):
                    sample_idx = self._per_cat_offset[cat] % len(pool)
                    task = pool[sample_idx]
                    test_tasks.append((task, cat))
                    sampled_cats.append(cat)
                    self._per_cat_offset[cat] += 1

            self._category_offset += n_cats_this_iter

            _log(
                "",
                f"  Testing {len(test_tasks)} tasks from domains: "
                f"{', '.join(sampled_cats)}...",
            )

            # Run tasks on VMs
            tasks_only = [t for t, _ in test_tasks]
            results = await self.env_pool.run_batch(
                tasks_only,
                self.base_agent,
                settle_time=self.config.settle_time,
                setup_time=self.config.setup_time,
            )

            # Collect failures
            failures: list[tuple[dict, str, str, str]] = []
            for (task, category), (trace_data, score) in zip(test_tasks, results):
                status = "[OK]" if score >= self.config.failure_threshold else "[FAIL]"
                _log(
                    "",
                    f"    {status} [{category}] {task.instruction[:50]}... "
                    f"(score={score:.2f})",
                )
                if score < self.config.failure_threshold:
                    failures.append((
                        trace_data,
                        f"score={score:.2f}",
                        task.instruction,
                        category,
                    ))

            if len(failures) == 0:
                _log("", "  -> All tasks passed, no proposal needed")
                continue

            _log("", f"  -> {len(failures)} failure(s), proposing improvement...")

            parent_score = next(
                (
                    score
                    for name, score in self.manager.get_frontier_with_scores()
                    if name == parent
                ),
                0.0,
            )

            # Run proposer with failures
            mutation_result = await self._mutate_with_fallback(
                parent, failures, actual_iteration
            )

            if mutation_result is None:
                no_improvement_count += 1
                consecutive_proposer_failures += 1
            else:
                consecutive_proposer_failures = 0
                child_name, proposal, justification = mutation_result

                # Evaluate child on validation set
                _log("", f"  -> Evaluating {child_name}...")
                child_score = await self._evaluate(self.val_data)

                added = self.manager.update_frontier(
                    child_name, child_score, max_size=self.config.frontier_size
                )

                if added:
                    _log("", f"  [OK] Added to frontier (score: {child_score:.4f})")
                    outcome = "improved"
                    no_improvement_count = 0
                else:
                    _log("", f"  [SKIP] Discarded (score: {child_score:.4f})")
                    outcome = "discarded"
                    self.manager.discard(child_name)
                    no_improvement_count += 1

                active_skills = self._get_active_skills()
                append_feedback(
                    self._feedback_path,
                    child_name,
                    proposal,
                    justification,
                    outcome=outcome,
                    score=child_score,
                    parent_score=parent_score,
                    active_skills=active_skills,
                )

            # Check early stopping
            if no_improvement_count >= self.config.no_improvement_limit:
                _log(
                    "STOP",
                    f"No improvement for {self.config.no_improvement_limit} iterations",
                )
                break

            if consecutive_proposer_failures >= self.config.consecutive_proposer_failures_limit:
                _log("STOP", "Too many consecutive proposer failures")
                break

            frontier_str = ", ".join(
                f"{n}:{s:.2f}"
                for n, s in self.manager.get_frontier_with_scores()
            )
            _log("", f"  Frontier: [{frontier_str}]")

        # 3. Return results
        frontier = self.manager.get_frontier_with_scores()
        best = self.manager.get_best_from_frontier()
        best_score = frontier[0][1] if frontier else 0.0

        _log(
            "DONE",
            f"{iteration_count} iterations, best: {best or 'base'} ({best_score:.4f})",
        )

        return LoopResult(
            frontier=frontier,
            best_program=best or "base",
            best_score=best_score,
            iterations_completed=iteration_count,
        )

    async def _ensure_base_program(self) -> None:
        """Create and evaluate base program."""
        if "base" not in self.manager.list_programs():
            base_config = ProgramConfig(
                name="base",
                parent=None,
                generation=0,
                system_prompt="osworld-base",
                allowed_tools=[],
                output_format={},
                metadata={},
            ).with_timestamp()

            self.manager.create_program("base", base_config)
            _log("INIT", "Created base program")
        else:
            _log("INIT", "Using existing base program")

        self.manager.switch_to("base")
        _log("", f"  -> Evaluating on {len(self.val_data)} validation tasks...")
        base_score = await self._evaluate(self.val_data)
        self.manager.update_frontier(
            "base", base_score, max_size=self.config.frontier_size
        )
        _log("", f"  -> Base score: {base_score:.4f}")

    async def _evaluate(
        self, data: list[tuple[OSWorldTask, str]]
    ) -> float:
        """Evaluate agent on a set of tasks.

        Args:
            data: List of (OSWorldTask, domain) tuples.

        Returns:
            Mean score (0.0 to 1.0).
        """
        tasks = [task for task, _ in data]
        results = await self.env_pool.run_batch(
            tasks,
            self.base_agent,
            settle_time=self.config.settle_time,
            setup_time=self.config.setup_time,
        )
        scores = [score for _, score in results]
        return sum(scores) / len(scores) if scores else 0.0

    def _get_best_parent(self) -> str:
        """Get the best-scoring program from the frontier."""
        best = self.manager.get_best_from_frontier()
        return best or "base"

    def _get_active_skills(self) -> list[str]:
        """List currently active skills."""
        skills_dir = self._project_root / ".claude" / "skills"
        if not skills_dir.exists():
            return []
        return [
            d.name
            for d in skills_dir.iterdir()
            if d.is_dir() and (d / "SKILL.md").exists()
        ]

    async def _mutate(
        self,
        parent: str,
        failures: list[tuple[dict, str, str, str]],
        iteration: int,
        truncation_level: int = 0,
    ) -> tuple[str, str, str] | None:
        """Run proposer and generator to create a mutation.

        Args:
            parent: Name of the parent program.
            failures: List of (trace_data, score_str, instruction, domain) tuples.
            iteration: Current iteration number.
            truncation_level: Context reduction level.

        Returns:
            Tuple of (child_name, proposal, justification) if created, None otherwise.
        """
        actual_iteration = iteration

        # Wrap trace dicts into objects with summarize() for build_proposer_query
        adapted_failures = []
        for trace_data, score_str, instruction, domain in failures:
            wrapper = _TraceWrapper(trace_data)
            adapted_failures.append((wrapper, score_str, instruction, domain))

        evolution_mode = self.config.evolution_mode
        _log(
            "",
            f"  -> Running {evolution_mode.replace('_only', '')} proposer "
            f"with {len(failures)} failures...",
        )
        feedback_history = read_feedback_history(self._feedback_path)
        proposer_query = build_proposer_query(
            adapted_failures, feedback_history, evolution_mode, truncation_level
        )

        if evolution_mode == "skill_only":
            proposer_trace = await self.agents.skill_proposer.run(proposer_query)

            if proposer_trace.output is None:
                _log("", f"  [WARN] Skill proposer failed: {proposer_trace.parse_error}")
                return None

            proposed = proposer_trace.output.proposed_skill
            justification = proposer_trace.output.justification
            action_type = proposer_trace.output.action
            target_skill = proposer_trace.output.target_skill

            action_label = (
                f"edit:{target_skill}" if action_type == "edit" else "create"
            )
            _log("", f"  -> Proposal: skill ({action_label}) - {proposed[:50]}...")

            child_name = f"iter-skill-{actual_iteration}"
            parent_config = self.manager.get_current()
            child_config = parent_config.mutate(child_name)
            self.manager.create_program(child_name, child_config, parent=parent)

            if action_type == "edit" and target_skill:
                _log("", f"  -> Editing existing skill: {target_skill}...")
                skill_query = (
                    f"EDIT existing skill: {target_skill}\n\n"
                    f"Modifications needed:\n{proposed}\n\n"
                    f"Justification: {justification}\n\n"
                    f"Read the existing skill at .claude/skills/{target_skill}/SKILL.md "
                    f"and modify it. Preserve existing content that is still relevant."
                )
            else:
                _log("", "  -> Generating new skill...")
                skill_query = build_skill_query_from_skill_proposer(proposer_trace)

            skill_trace = await self.agents.skill_generator.run(skill_query)
            if skill_trace.output:
                pass  # Skill written to file by the generator

        else:  # prompt_only
            proposer_trace = await self.agents.prompt_proposer.run(proposer_query)

            if proposer_trace.output is None:
                _log(
                    "",
                    f"  [WARN] Prompt proposer failed: {proposer_trace.parse_error}",
                )
                return None

            proposed = proposer_trace.output.proposed_prompt_change
            justification = proposer_trace.output.justification
            _log("", f"  -> Proposal: prompt - {proposed[:50]}...")

            child_name = f"iter-prompt-{actual_iteration}"
            parent_config = self.manager.get_current()
            original_prompt = parent_config.system_prompt
            child_config = parent_config.mutate(child_name)
            self.manager.create_program(child_name, child_config, parent=parent)

            _log("", "  -> Generating optimized prompt...")
            prompt_query = build_prompt_query_from_prompt_proposer(
                proposer_trace, original_prompt
            )
            prompt_trace = await self.agents.prompt_generator.run(prompt_query)
            if prompt_trace.output and self._prompt_path:
                update_prompt_file(
                    self._prompt_path, prompt_trace.output.optimized_prompt
                )

        self.manager.commit(f"{child_name}: {proposed[:50]}")
        return (child_name, proposed, justification)

    async def _mutate_with_fallback(
        self,
        parent: str,
        failures: list[tuple[dict, str, str, str]],
        iteration: int,
    ) -> tuple[str, str, str] | None:
        """Try progressive truncation levels, then single-failure fallback."""
        max_level = self.config.proposer_max_truncation_level

        for truncation_level in range(max_level + 1):
            if truncation_level > 0:
                _log("", f"  -> Retrying with truncation level {truncation_level}...")
            result = await self._mutate(
                parent, failures, iteration, truncation_level
            )
            if result is not None:
                return result

        if self.config.proposer_single_failure_fallback and len(failures) > 1:
            _log("", "  -> All levels failed, trying single-failure fallback...")
            shortest = min(failures, key=lambda f: len(str(f[0].get("result", ""))))
            result = await self._mutate(
                parent, [shortest], iteration, truncation_level=max_level
            )
            if result is not None:
                return result

        _log("", "  [WARN] All proposer fallback attempts failed")
        return None


class _TraceWrapper:
    """Wraps a trace_data dict to be compatible with build_proposer_query.

    Implements the .summarize() method that build_proposer_query calls
    on each trace, and an .output attribute with .final_answer.
    """

    def __init__(self, trace_data: dict[str, Any]):
        self._data = trace_data
        self.output = self._make_output()

    def _make_output(self):
        """Create a simple object with final_answer and reasoning."""

        class _Output:
            def __init__(self, data):
                output = data.get("output")
                if output and hasattr(output, "status"):
                    self.final_answer = output.status
                    self.reasoning = output.summary
                else:
                    self.final_answer = "[AGENT ERROR]"
                    self.reasoning = str(data.get("parse_error", "Unknown error"))

        return _Output(self._data)

    def summarize(self, head_chars: int = 60_000, tail_chars: int = 60_000) -> str:
        """Format the trace for proposer consumption."""
        lines = [
            f"Model: {self._data.get('model', 'unknown')}",
            f"Turns: {self._data.get('num_turns', 'unknown')}",
            f"Duration: {self._data.get('duration_ms', 'unknown')}ms",
            f"Is Error: {self._data.get('is_error', False)}",
        ]

        if self._data.get("parse_error"):
            lines.append(f"Parse Error: {self._data['parse_error']}")

        if self._data.get("output"):
            lines.append(f"Output: {self._data['output']}")

        result_str = str(self._data.get("result", ""))

        if len(result_str) > (head_chars + tail_chars):
            truncated = len(result_str) - head_chars - tail_chars
            lines.append(f"\n## Result (truncated, {truncated:,} chars omitted)")
            lines.append(f"### Start:\n{result_str[:head_chars]}")
            lines.append(f"\n[... {truncated:,} characters truncated ...]\n")
            lines.append(f"### End:\n{result_str[-tail_chars:]}")
        else:
            lines.append(f"\n## Full Result\n{result_str}")

        return "\n".join(lines)
