"""Self-improving agent loop runner."""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

from src.harness import Agent, AgentTrace, is_claude_sdk
from src.cache import RunCache, CacheConfig
from src.registry.sdk_utils import options_to_config


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


def _score_multi_tolerance(question: str, predicted: str, ground_truth: str) -> float:
    """Score answer using weighted average across tolerance levels.

    Weights favor stricter tolerances: weight = 1 / (1 + 20 * tolerance)
    This gives approximate weights:
      - 0.0%  tolerance: 1.00 (exact match, highest priority)
      - 1.0%  tolerance: 0.83
      - 2.5%  tolerance: 0.67
      - 5.0%  tolerance: 0.50
      - 10.0% tolerance: 0.33
    """
    weighted_sum = 0.0
    weight_total = 0.0
    for tol in TOLERANCE_LEVELS:
        weight = 1.0 / (1.0 + 20.0 * tol)
        score = score_answer(predicted, ground_truth, tol)
        weighted_sum += weight * score
        weight_total += weight
    return weighted_sum / weight_total


from src.evaluation import score_answer, evaluate_agent_parallel
from src.registry import ProgramManager
from src.schemas import (
    AgentResponse,
    ProposerResponse,
    ToolGeneratorResponse,
    PromptGeneratorResponse,
    SkillProposerResponse,
    PromptProposerResponse,
)

from .config import LoopConfig
from .helpers import (
    build_proposer_query,
    build_skill_query,
    build_prompt_query,
    build_skill_query_from_skill_proposer,
    build_prompt_query_from_prompt_proposer,
    append_feedback,
    normalize_project_skill_frontmatter,
    read_feedback_history,
    update_prompt_file,
)


T = TypeVar("T")

TOLERANCE_LEVELS = [0.05, 0.01, 0.1, 0.0, 0.025]


@dataclass
class LoopAgents:
    """Container for the agents used in the loop."""

    base: Agent[AgentResponse]
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
    total_cost_usd: float = 0.0


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
        train_pools: dict[str, list[tuple[str, str]]],
        val_data: list[tuple[str, str, str]],
        scorer: Callable[[str, str, str], float] | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        task_constraints: str = "",
    ):
        """Initialize the self-improving loop.

        Args:
            config: Loop configuration parameters.
            agents: Container with the 4 agents (base, proposer, skill_generator, prompt_generator).
            manager: ProgramManager for git-based versioning.
            train_pools: Dict mapping category -> list of (question, answer) tuples.
            val_data: Validation data as list of (question, answer, category) tuples.
            scorer: Scoring function (question, predicted, ground_truth) -> float.
                    Defaults to _score_multi_tolerance for backward compatibility.
        """
        self.config = config
        self.agents = agents
        self.manager = manager
        self.train_pools = train_pools
        self.val_data = val_data
        self.scorer = scorer or _score_multi_tolerance
        self.on_event = on_event
        self.task_constraints = task_constraints

        # Round-robin sampling state
        self._category_offset = 0  # Which category to start with next iteration
        self._per_cat_offset: dict[str, int] = {cat: 0 for cat in train_pools.keys()}

        # Paths
        self._project_root = Path(getattr(self.manager, "cwd", Path.cwd())).resolve()
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

        # Iteration offset for continue mode
        self._iteration_offset = 0

        # Checkpoint file for exact resume
        self._checkpoint_path = self._project_root / ".claude" / "loop_checkpoint.json"

        # Cost tracking
        self._total_cost: float = 0.0
        self._iter_cost: float = 0.0

    def _emit(self, event: str, **data: Any) -> None:
        """Fire an event to the display callback if one is registered."""
        if self.on_event is not None:
            self.on_event(event, data)

    def _save_checkpoint(self, iteration: int) -> None:
        """Save sampling state for exact resume.

        Args:
            iteration: The iteration number just completed.
        """
        checkpoint = {
            "iteration": iteration,
            "category_offset": self._category_offset,
            "per_cat_offset": self._per_cat_offset,
        }
        self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpoint_path.write_text(json.dumps(checkpoint, indent=2))

    def _load_checkpoint(self) -> int | None:
        """Load checkpoint if exists.

        Returns:
            Iteration number to resume from, or None if no checkpoint exists.
        """
        if not self._checkpoint_path.exists():
            return None
        try:
            checkpoint = json.loads(self._checkpoint_path.read_text())
            self._category_offset = checkpoint["category_offset"]
            self._per_cat_offset = checkpoint["per_cat_offset"]
            return checkpoint["iteration"]
        except (json.JSONDecodeError, KeyError) as e:
            _log("WARN", f"Invalid checkpoint file, ignoring: {e}")
            return None

    def _delete_checkpoint(self) -> None:
        """Delete checkpoint file if it exists."""
        if self._checkpoint_path.exists():
            self._checkpoint_path.unlink()

    async def run(self) -> LoopResult:
        """Run the full self-improving loop.

        Returns:
            LoopResult with frontier, best program, and iteration count.
        """
        # 0. Handle continue mode and feedback reset
        resume_iteration: int | None = None
        if not self.config.continue_mode:
            # Start fresh: reset feedback if configured
            if self.config.reset_feedback and self._feedback_path.exists():
                self._feedback_path.unlink()
            self._iteration_offset = 0
            # Delete any existing checkpoint on fresh start
            self._delete_checkpoint()
        else:
            # Continue mode: keep feedback, find highest iteration number
            self._iteration_offset = self._get_highest_iteration()
            # Try to load checkpoint for exact sampling state resume
            resume_iteration = self._load_checkpoint()
            if resume_iteration is not None:
                _log("CONTINUE", f"Resuming from iteration {resume_iteration} with exact sampling state")
            else:
                _log("CONTINUE", f"Resuming from iteration {self._iteration_offset} (no checkpoint, sampling state reset)")

        # Get sorted list of categories for deterministic round-robin
        categories = sorted(self.train_pools.keys())

        # 1. Create and evaluate base program if needed (skip in continue mode with existing frontier)
        if self.config.continue_mode and self.manager.get_frontier():
            # Continue mode: use existing frontier, switch to best program
            best = self._select_parent()
            self.manager.switch_to(best)
            frontier_str = ", ".join(f"{n}:{s:.2f}" for n, s in self.manager.get_frontier_with_scores())
            _log("CONTINUE", f"Using existing frontier: [{frontier_str}]")
        else:
            await self._ensure_base_program()

        # 2. Main loop
        no_improvement_count = 0
        iteration_count = 0
        n_cats = len(categories)

        for i in range(self.config.max_iterations):
            iteration_count = i + 1
            actual_iteration = iteration_count + self._iteration_offset

            # Skip already-completed iterations when resuming with checkpoint
            if resume_iteration is not None and actual_iteration <= resume_iteration:
                continue

            # Select parent from frontier using configured strategy
            parent = self._select_parent(iteration_count)
            self.manager.switch_to(parent)
            self._iter_cost = 0.0  # Reset per-iteration cost
            _log(f"ITER {iteration_count}/{self.config.max_iterations}", f"Parent: {parent}")
            self._emit("iter_start", iteration=actual_iteration, total=self.config.max_iterations, parent=parent)

            # Round-robin sampling: pick samples_per_category from each of N categories (cycling)
            n_cats_this_iter = min(self.config.categories_per_batch, n_cats)

            test_samples: list[tuple[str, str, str]] = []
            sampled_cats: list[str] = []
            for j in range(n_cats_this_iter):
                cat_idx = (self._category_offset + j) % n_cats
                cat = categories[cat_idx]
                pool = self.train_pools[cat]

                # Take min(samples_per_category, pool_size) to handle small categories
                samples_to_take = min(self.config.samples_per_category, len(pool))

                for _ in range(samples_to_take):
                    sample_idx = self._per_cat_offset[cat] % len(pool)
                    question, answer = pool[sample_idx]
                    test_samples.append((question, answer, cat))
                    sampled_cats.append(cat)
                    self._per_cat_offset[cat] += 1

            self._category_offset += n_cats_this_iter

            _log("", f"  Testing {len(test_samples)} samples from categories: {', '.join(sampled_cats)}...")

            # Run all samples concurrently
            traces = await asyncio.gather(*[
                self.agents.base.run(question) for question, _, _ in test_samples
            ])
            self._iter_cost += sum(t.total_cost_usd for t in traces)

            # Collect failures
            failures: list[tuple[AgentTrace, str, str, str]] = []  # (trace, agent_answer, ground_truth, category)
            for trace, (question, answer, category) in zip(traces, test_samples):
                agent_answer = (
                    trace.output.final_answer if trace.output else "[PARSE FAILED]"
                )
                avg_score = self.scorer(
                    question,
                    agent_answer.strip().lower(),
                    answer.strip().lower(),
                )
                status = "[OK]" if avg_score >= 0.8 else "[FAIL]"
                _log("", f"    {status} [{category}] {question[:40]}...")
                self._emit("sample", question=question, category=category, score=avg_score, passed=avg_score >= 0.8)
                if avg_score < 0.8:
                    failures.append((trace, agent_answer, answer, category))

            # Always propose if any failures exist
            if len(failures) == 0:
                _log("", f"  -> All samples passed, no proposal needed")
                continue

            _log("", f"  -> {len(failures)} failure(s), proposing improvement...")

            # Get parent's score for comparison
            parent_score = next(
                (score for name, score in self.manager.get_frontier_with_scores() if name == parent),
                0.0
            )

            # Run proposer with all failures (use actual iteration number with offset)
            mutation_result = await self._mutate_with_fallback(parent, failures, actual_iteration)

            if mutation_result is None:
                no_improvement_count += 1
            else:
                child_name, proposal, justification = mutation_result

                # Evaluate child
                _log("", f"  -> Evaluating {child_name}...")
                child_score = await self._evaluate(self.val_data)  # accumulates to self._iter_cost

                # Update frontier or discard
                added = self.manager.update_frontier(
                    child_name, child_score, max_size=self.config.frontier_size
                )

                if added:
                    _log("", f"  [OK] Added to frontier (score: {child_score:.4f})")
                    outcome = "improved" if child_score > parent_score else "kept"
                    no_improvement_count = 0
                else:
                    _log("", f"  [SKIP] Discarded (score: {child_score:.4f})")
                    outcome = "discarded"
                    self.manager.discard(child_name)
                    no_improvement_count += 1

                self._emit(
                    "eval_result",
                    child_name=child_name,
                    score=child_score,
                    parent_score=parent_score,
                    added=added,
                    frontier=self.manager.get_frontier_with_scores(),
                    n_skills=len(self._get_active_skills()),
                )

                # Record feedback with outcome for future proposers to learn from
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
                _log("STOP", f"No improvement for {self.config.no_improvement_limit} iterations")
                break

            # Print frontier status
            frontier_str = ", ".join(f"{n}:{s:.2f}" for n, s in self.manager.get_frontier_with_scores())
            _log("", f"  Frontier: [{frontier_str}]")

            # Report per-iteration and cumulative cost
            self._total_cost += self._iter_cost
            _log("COST", f"Iter {iteration_count} cost: ${self._iter_cost:.4f} | Running total: ${self._total_cost:.4f}")

            # Save checkpoint at end of each successful iteration
            self._save_checkpoint(actual_iteration)

        # 3. Return results
        frontier = self.manager.get_frontier_with_scores()
        best = self.manager.get_best_from_frontier()
        best_score = frontier[0][1] if frontier else 0.0

        _log("DONE", f"{iteration_count} iterations, best: {best or 'base'} ({best_score:.4f})")
        _log("COST", f"Total cost: ${self._total_cost:.4f}")
        self._emit("loop_done", best=best or "base", best_score=best_score, iterations=iteration_count)

        return LoopResult(
            frontier=frontier,
            best_program=best or "base",
            best_score=best_score,
            iterations_completed=iteration_count,
            total_cost_usd=self._total_cost,
        )

    async def _ensure_base_program(self) -> None:
        """Create and evaluate base program if it doesn't exist."""
        if "base" not in self.manager.list_programs():
            current_options = self.agents.base._get_options()
            base_config = options_to_config(current_options, "base")
            self.manager.create_program("base", base_config)
            _log("INIT", "Created base program")
        else:
            _log("INIT", "Using existing base program")

        # Evaluate and add base to frontier
        self.manager.switch_to("base")
        _log("", f"  -> Evaluating on {len(self.val_data)} samples...")
        self._iter_cost = 0.0
        base_score = await self._evaluate(self.val_data)
        self._total_cost += self._iter_cost
        self.manager.update_frontier(
            "base", base_score, max_size=self.config.frontier_size
        )
        _log("", f"  -> Base score: {base_score:.4f}")
        _log("", f"  -> Frontier: {self.manager.get_frontier()}")
        _log("COST", f"Base eval cost: ${self._iter_cost:.4f} | Total: ${self._total_cost:.4f}")
        self._emit("baseline", score=base_score)

    async def _evaluate(self, data: list[tuple[str, str, str]]) -> float:
        """Evaluate base agent on data.

        Args:
            data: List of (question, answer, category) tuples.

        Returns:
            Accuracy score (0.0 to 1.0).
        """
        # Convert to (question, answer) format for evaluate_agent_parallel
        qa_data = [(q, a) for q, a, _ in data]
        results = await evaluate_agent_parallel(
            self.agents.base, qa_data, max_concurrent=self.config.concurrency, cache=self.cache
        )

        score = 0.0
        for result in results:
            if result.trace is not None:
                self._iter_cost += result.trace.total_cost_usd
            if result.trace is None or result.trace.output is None:
                continue  # Timeout/error/parse failed = 0 score
            score += self.scorer(
                result.question,
                result.trace.output.final_answer,
                result.ground_truth,
            )
        return score / len(results)

    async def _mutate(
        self,
        parent: str,
        failures: list[tuple[AgentTrace[AgentResponse], str, str, str]],
        iteration: int,
        truncation_level: int = 0,
    ) -> tuple[str, str, str] | None:
        """Run proposer and generator to create a mutation based on multiple failures.

        Args:
            parent: Name of the parent program.
            failures: List of (trace, agent_answer, ground_truth, category) tuples from failed attempts.
            iteration: Current iteration number.
            truncation_level: Context reduction level (0=full, 1=moderate, 2=aggressive).

        Returns:
            Tuple of (child_name, proposal, justification) if created, None otherwise.
        """
        # Calculate actual iteration number (with offset for continue mode)
        actual_iteration = iteration + self._iteration_offset

        # Run appropriate proposer based on evolution mode
        evolution_mode = self.config.evolution_mode
        _log("", f"  -> Running {evolution_mode.replace('_only', '')} proposer with {len(failures)} failures...")
        feedback_history = read_feedback_history(self._feedback_path)
        proposer_query = build_proposer_query(
            failures,
            feedback_history,
            evolution_mode,
            truncation_level,
            self.task_constraints,
            project_root=self._project_root,
        )

        if evolution_mode == "skill_only":
            proposer_trace = await self.agents.skill_proposer.run(proposer_query)
            self._iter_cost += proposer_trace.total_cost_usd

            if proposer_trace.output is None:
                _log("", f"  [WARN] Skill proposer failed: {proposer_trace.parse_error}")
                return None

            proposer_output = proposer_trace.output
            proposed = proposer_output.proposed_skill
            justification = proposer_output.justification
            action_type = proposer_output.action
            target_skill = proposer_output.target_skill

            action_label = f"edit:{target_skill}" if action_type == "edit" else "create"
            _log("", f"  -> Proposal: skill ({action_label}) - {proposed[:50]}...")
            self._emit("proposal", action=action_type, target_skill=target_skill, summary=proposed[:80])

            # Create child program branch
            child_name = f"iter-skill-{actual_iteration}"
            parent_config = self.manager.get_current()
            child_config = parent_config.mutate(child_name)
            self.manager.create_program(child_name, child_config, parent=parent)

            # Generate skill - use different query for edit vs create
            if action_type == "edit" and target_skill:
                _log("", f"  -> Editing existing skill: {target_skill}...")
                skill_query = f"""EDIT existing skill: {target_skill}

Modifications needed:
{proposed}

Justification: {justification}

Read the existing skill at .claude/skills/{target_skill}/SKILL.md
and modify it to add these capabilities. Preserve all existing content that is still relevant."""
            else:
                _log("", f"  -> Generating new skill...")
                skill_query = build_skill_query_from_skill_proposer(proposer_trace)

            skills_before = set(self._get_active_skills())
            skill_trace = await self.agents.skill_generator.run(skill_query)
            self._iter_cost += skill_trace.total_cost_usd
            skills_after = set(self._get_active_skills())
            new_skills = skills_after - skills_before
            created_skill = next(iter(new_skills)) if new_skills else None

            if not is_claude_sdk():
                skill_descriptions: dict[str, str] = {}
                if target_skill:
                    skill_descriptions[target_skill] = proposed
                if created_skill:
                    skill_descriptions[created_skill] = proposed
                normalize_project_skill_frontmatter(
                    self._project_root,
                    descriptions=skill_descriptions,
                    fallback_description=proposed,
                    compatibility="opencode",
                )

            if skill_trace.output:
                self._emit("skill_written", name=created_skill, action=action_type, target=target_skill)

        else:  # prompt_only
            proposer_trace = await self.agents.prompt_proposer.run(proposer_query)
            self._iter_cost += proposer_trace.total_cost_usd

            if proposer_trace.output is None:
                _log("", f"  [WARN] Prompt proposer failed: {proposer_trace.parse_error}")
                return None

            proposed = proposer_trace.output.proposed_prompt_change
            justification = proposer_trace.output.justification
            _log("", f"  -> Proposal: prompt - {proposed[:50]}...")

            # Create child program branch
            child_name = f"iter-prompt-{actual_iteration}"
            parent_config = self.manager.get_current()
            original_prompt = parent_config.system_prompt
            child_config = parent_config.mutate(child_name)
            self.manager.create_program(child_name, child_config, parent=parent)

            # Generate optimized prompt
            _log("", f"  -> Generating optimized prompt...")
            prompt_query = build_prompt_query_from_prompt_proposer(
                proposer_trace, original_prompt
            )
            prompt_trace = await self.agents.prompt_generator.run(prompt_query)
            self._iter_cost += prompt_trace.total_cost_usd
            if prompt_trace.output:
                update_prompt_file(
                    self._prompt_path, prompt_trace.output.optimized_prompt
                )

        # Commit changes
        self.manager.commit(f"{child_name}: {proposed[:50]}")

        # Return mutation info (feedback will be written by caller with outcome)
        return (child_name, proposed, justification)

    async def _mutate_with_fallback(
        self,
        parent: str,
        failures: list[tuple[AgentTrace[AgentResponse], str, str, str]],
        iteration: int,
    ) -> tuple[str, str, str] | None:
        """Try progressive truncation levels, then single-failure fallback.

        Args:
            parent: Name of the parent program.
            failures: List of (trace, agent_answer, ground_truth, category) tuples.
            iteration: Current iteration number.

        Returns:
            Tuple of (child_name, proposal, justification) if created, None otherwise.
        """
        max_level = self.config.proposer_max_truncation_level

        for truncation_level in range(max_level + 1):
            if truncation_level > 0:
                _log("", f"  -> Retrying with truncation level {truncation_level}...")

            result = await self._mutate(parent, failures, iteration, truncation_level)
            if result is not None:
                return result

        # Final fallback: single failure focus (if enabled and multiple failures)
        if self.config.proposer_single_failure_fallback and len(failures) > 1:
            _log("", f"  -> All truncation levels failed, trying single-failure fallback...")
            single_failure = self._pick_shortest_failure(failures)
            result = await self._mutate(parent, [single_failure], iteration, truncation_level=max_level)
            if result is not None:
                return result

        _log("", f"  [WARN] All proposer fallback attempts failed")
        return None

    def _pick_shortest_failure(
        self,
        failures: list[tuple[AgentTrace[AgentResponse], str, str, str]],
    ) -> tuple[AgentTrace[AgentResponse], str, str, str]:
        """Pick the failure with the shortest trace for fallback.

        Args:
            failures: List of (trace, agent_answer, ground_truth, category) tuples.

        Returns:
            The failure tuple with the shortest trace summary.
        """
        # Estimate trace length by summarizing with default params
        shortest = failures[0]
        shortest_len = len(shortest[0].summarize())

        for failure in failures[1:]:
            length = len(failure[0].summarize())
            if length < shortest_len:
                shortest = failure
                shortest_len = length

        return shortest

    def _select_parent(self, iteration: int = 0) -> str:
        """Select a parent program from the frontier using the configured strategy.

        Args:
            iteration: Current iteration number (used by round_robin strategy).

        Returns:
            Program name to use as parent, or 'base' if frontier is empty.
        """
        selected = self.manager.select_from_frontier(
            self.config.selection_strategy, iteration
        )
        return selected if selected else "base"

    def _get_active_skills(self) -> list[str]:
        """Get list of currently active skills.

        Returns:
            List of skill names that have SKILL.md files.
        """
        skills_dir = self._project_root / ".claude" / "skills"
        active_skills = []
        if skills_dir.exists():
            for skill_dir in skills_dir.iterdir():
                if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                    active_skills.append(skill_dir.name)
        return sorted(active_skills)

    def _get_highest_iteration(self) -> int:
        """Find the highest iteration number across all iter-* branches.

        Returns:
            The highest iteration number found, or 0 if none exist.
        """
        programs = self.manager.list_programs()
        max_iter = 0
        for p in programs:
            # Match iter-skill-N or iter-prompt-N or iter-N
            if p.startswith("iter-"):
                parts = p.split("-")
                try:
                    num = int(parts[-1])
                    max_iter = max(max_iter, num)
                except ValueError:
                    pass
        return max_iter
