"""Self-improving agent loop runner."""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

from opentelemetry import trace as otel_trace

from src.harness import Agent, AgentTrace, is_claude_sdk, is_opencode_sdk, is_openhands_sdk, is_goose_sdk, is_codex_sdk
from src.cache import RunCache, CacheConfig
from src.registry.sdk_utils import options_to_config
from src.loop.trace_db import TraceDB
from src.loop.reviewer import BackgroundReviewer
from src.loop.constraints import gate as constraint_gate

_tracer = otel_trace.get_tracer("evoskill.loop")


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


import math as _math

def _measure_rel_error(predicted: str, ground_truth: str) -> float:
    """Binary-search for the smallest tolerance at which `score_answer` passes.

    That minimum tolerance approximates the relative error between the
    predicted and ground-truth numbers. Returns 0.0 when they match
    exactly, 1.0 when they don't match even at 100% tolerance (e.g. a
    non-numeric string mismatch).
    """
    # Exact match first — avoids search when GT is a year/name/etc. that's right
    if score_answer(predicted, ground_truth, 0.0) >= 1.0:
        return 0.0
    # If even a giant tolerance doesn't help, treat as string-mismatch → 100%
    if score_answer(predicted, ground_truth, 1.0) < 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    # 20 iters = precision ~1e-6 on the crossover point
    for _ in range(20):
        mid = (lo + hi) / 2
        if score_answer(predicted, ground_truth, mid) >= 1.0:
            hi = mid
        else:
            lo = mid
    return hi


def _score_multi_tolerance(question: str, predicted: str, ground_truth: str) -> float:
    """Smooth-decay scorer. Continuous in relative error — no stair-step tiers.

    Shape:
      - rel_err ≤ 0.01  → 1.00    (full credit "soft zone")
      - 0.01 → 0.10     → 1.0 → 0.0 via raised-cosine decay
      - rel_err ≥ 0.10  → 0.00    (hard cutoff)

    The cosine decay has zero slope at both ends, so score is smooth at
    the zone boundaries (no derivative kink). Example scores:

      rel_err | score
      --------|------
       0.010  | 1.000
       0.020  | 0.970
       0.025  | 0.930
       0.040  | 0.793
       0.055  | 0.500
       0.070  | 0.207
       0.085  | 0.030
       0.100  | 0.000

    Non-numeric answers fall through to exact string match (score = 1.0 or
    0.0, since tolerance has no effect on string equality).
    """
    rel_err = _measure_rel_error(predicted, ground_truth)
    soft, hard = 0.01, 0.10
    if rel_err <= soft:
        return 1.0
    if rel_err >= hard:
        return 0.0
    t = (rel_err - soft) / (hard - soft)
    return (1.0 + _math.cos(_math.pi * t)) / 2.0


from src.evaluation import score_answer, evaluate_agent_parallel
from src.registry import ProgramManager
from src.schemas import (
    AgentResponse,
    ProposerResponse,
    ToolGeneratorResponse,
    PromptGeneratorResponse,
    SkillProposerResponse,
    PromptProposerResponse,
    SkillEvolverResponse,
)

from .config import LoopConfig
from .helpers import (
    build_proposer_query,
    build_skill_query,
    build_prompt_query,
    build_skill_query_from_skill_proposer,
    build_prompt_query_from_prompt_proposer,
    append_feedback,
    read_feedback_history,
    update_prompt_file,
)


T = TypeVar("T")


@dataclass
class LoopAgents:
    """Container for the agents used in the loop."""

    solver: Agent[AgentResponse]
    skill_proposer: Agent[SkillProposerResponse]
    prompt_proposer: Agent[PromptProposerResponse]
    skill_generator: Agent[ToolGeneratorResponse]
    prompt_generator: Agent[PromptGeneratorResponse]
    # Unified evolver (proposer+generator in one pass) — used when mode == "skill_unified"
    skill_evolver: Agent[SkillEvolverResponse] | None = None


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
        solver_prompt: str = "",
        data_root: str | Path | None = None,
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
        self.solver_prompt = solver_prompt
        # Forwarded into the evolver prompt so it can interpret the
        # data-root-relative tool paths in failure traces (e.g.
        # `Read(treasury_bulletins_parsed/...)` is relative to this root).
        self._data_root = str(data_root) if data_root else None

        # Publish the scorer to the executor via contextvar so each agent.run
        # span can stamp [OK 0.997] / [FAIL 0.000] onto its name when a
        # ground_truth is in scope. Saves us from creating a separate
        # per-sample eval span just to surface pass/fail in Phoenix's
        # trace list.
        from src.harness.utils import eval_score_callback as _esc
        _esc.set(self.scorer)

        # Detect if validation data overlaps with training data (e.g., tiny datasets)
        train_questions = {q for pool in train_pools.values() for q, _ in pool}
        val_questions = {q for q, _, _ in val_data}
        self._val_is_train_subset = bool(val_questions) and val_questions <= train_questions
        if self._val_is_train_subset:
            _log("INIT", "Validation data overlaps with training — will reuse training traces for evaluation")

        # Round-robin sampling state
        self._category_offset = 0  # Which category to start with next iteration
        self._per_cat_offset: dict[str, int] = {cat: 0 for cat in train_pools.keys()}
        # Per-cycle shuffle of pool indices. Without this, a pool of size P
        # sampled k-at-a-time produces the SAME k samples every (P/k)-th iter
        # — iters 1+3 (or 2+4) would land on identical UIDs in linear order,
        # which we observed in practice. We shuffle once per cycle (each time
        # _per_cat_offset wraps modulo pool size). Seed includes the cycle
        # number so order is deterministic for replays but different across
        # cycles. See _next_train_sample() for the consumption side.
        import random as _rnd
        self._pool_order: dict[str, list[int]] = {}
        self._pool_cycle: dict[str, int] = {cat: 0 for cat in train_pools.keys()}
        for cat, pool in train_pools.items():
            order = list(range(len(pool)))
            _rnd.Random(f"{cat}:0").shuffle(order)
            self._pool_order[cat] = order

        # Proportional sampling state (used when config.proportional_sampling is True).
        # Builds a fixed schedule where each category appears `len(pool)` times,
        # interleaved by Bresenham-style largest-deficit selection. Iteration N
        # consumes `failure_sample_count` slots starting at _schedule_offset.
        self._schedule: list[str] = self._build_proportional_schedule(train_pools)
        self._schedule_offset = 0

        # Paths.
        # `_project_root` here is historically misnamed — it tracks the
        # workspace (where loop state lives: feedback, checkpoint, traces.db),
        # NOT the project source tree. Don't use it to resolve where skills
        # or prompt.txt live; use `_project_skills_dir` for skills (sourced
        # from the manager so it's correct under workspace/project split).
        self._project_root = Path(getattr(self.manager, "cwd", Path.cwd())).resolve()
        self._feedback_path = self._project_root / ".claude" / "feedback_history.md"
        # The actual skills directory — where evolver agents WRITE skills
        # and where the solver reads them via the symlink. Comes from the
        # manager which got it from run_loop.py / cli.run_cmd. Falls back
        # to the workspace's .claude/skills only when the caller didn't
        # split workspace from project_root.
        self._project_skills_dir = getattr(
            self.manager, "_project_skills_dir",
            self._project_root / ".claude" / "skills",
        )
        # Project source tree — for `prompt.txt` and similar repo-shipped
        # files. Derived as the parent of the skills dir's `.claude` parent.
        self._project_source_root = self._project_skills_dir.parent.parent
        self._prompt_path = (
            self._project_source_root / "src" / "agent_profiles" / "base_agent" / "prompt.txt"
        )

        # Initialize cache. CRITICAL: pass `live_skills_dir` and
        # `project_source_root` so the cache key reflects the LIVE skills
        # the agent reads from (not the workspace's stub snapshot dir) and
        # the actual prompt files (in EvoSkill source, not the workspace).
        # Without these, the cache key stays stable across iterations even
        # as the evolver writes new skills → returns stale cached responses
        # generated under different skill conditions → masks the evolver's
        # effect on the agent (e.g. mid-gate's "fixed=0, regressed=0,
        # same=N" artifact when ALL responses came from the cache).
        if config.cache_enabled:
            cache_config = CacheConfig(
                cache_dir=config.cache_dir,
                enabled=True,
                store_messages=config.cache_store_messages,
                cwd=self._project_root,
                live_skills_dir=self._project_skills_dir,
                project_source_root=self._project_source_root,
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

        # Trace DB + background reviewer for cross-iteration learning
        db_path = self._project_root / ".cache" / "traces.db"
        self.trace_db = TraceDB(db_path)
        self.reviewer: BackgroundReviewer | None = None
        if getattr(config, "reviewer_enabled", True):
            try:
                self.reviewer = BackgroundReviewer(db_path)
            except Exception as e:
                _log("WARN", f"BackgroundReviewer disabled: {e}")
                self.reviewer = None

    def _emit(self, event: str, **data: Any) -> None:
        """Fire an event to the display callback if one is registered."""
        if self.on_event is not None:
            self.on_event(event, data)

    def _next_train_sample(self, cat: str) -> tuple[str, str]:
        """Pull the next training sample from `cat`'s pool, with per-cycle shuffling.

        Maintains a shuffled order over pool indices for the current cycle.
        When _per_cat_offset wraps modulo pool size we've consumed one full
        cycle; reshuffle so the next cycle visits samples in a different
        order. Without this, iters at the same offset within consecutive
        cycles would draw identical samples.
        """
        pool = self.train_pools[cat]
        idx_in_cycle = self._per_cat_offset[cat] % len(pool)
        sample_idx = self._pool_order[cat][idx_in_cycle]
        self._per_cat_offset[cat] += 1
        # Crossing the boundary into a new cycle? Reshuffle.
        if self._per_cat_offset[cat] % len(pool) == 0:
            import random as _rnd
            self._pool_cycle[cat] += 1
            order = list(range(len(pool)))
            _rnd.Random(f"{cat}:{self._pool_cycle[cat]}").shuffle(order)
            self._pool_order[cat] = order
        return pool[sample_idx]

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

            # Set the per-iter prefix that the executor will prepend to all
            # span names produced during this iteration. Phoenix shows
            # "iter 3 | agent.run:base [train 2/5]" instead of just
            # "agent.run:base [train 2/5]" — keeps thousands of spans
            # scannable across iterations.
            from src.harness.utils import eval_iter_label as _eval_iter_label
            _iter_label_token = _eval_iter_label.set(f"iter {actual_iteration}")

            test_samples: list[tuple[str, str, str]] = []
            sampled_cats: list[str] = []
            if getattr(self.config, "proportional_sampling", False) and self._schedule:
                # Proportional sampling: pull `failure_sample_count` items from
                # the precomputed schedule (cycling). Each category appears in
                # the schedule proportional to its pool size, so over N iters
                # each category is sampled in proportion. Within a category,
                # _next_train_sample picks the next index from the shuffled
                # per-cycle order.
                n_to_take = self.config.failure_sample_count
                for k in range(n_to_take):
                    cat = self._schedule[(self._schedule_offset + k) % len(self._schedule)]
                    question, answer = self._next_train_sample(cat)
                    test_samples.append((question, answer, cat))
                    sampled_cats.append(cat)
                self._schedule_offset += n_to_take
            else:
                # Round-robin sampling: pick samples_per_category from each of N categories (cycling)
                n_cats_this_iter = min(self.config.categories_per_batch, n_cats)
                for j in range(n_cats_this_iter):
                    cat_idx = (self._category_offset + j) % n_cats
                    cat = categories[cat_idx]
                    pool = self.train_pools[cat]
                    # Take min(samples_per_category, pool_size) to handle small categories
                    samples_to_take = min(self.config.samples_per_category, len(pool))
                    for _ in range(samples_to_take):
                        question, answer = self._next_train_sample(cat)
                        test_samples.append((question, answer, cat))
                        sampled_cats.append(cat)
                self._category_offset += n_cats_this_iter

            _log("", f"  Testing {len(test_samples)} samples from categories: {', '.join(sampled_cats)}...")

            # Run all samples concurrently. Tag each run with a human-readable
            # "train i/N" label so interleaved stdout is scannable.
            # Use return_exceptions=True so a single sample's TimeoutError after
            # exhausted retries doesn't kill the whole iteration. Failed samples
            # become placeholder traces (no output → counted as [FAIL]).
            # Set the per-sample ground_truth contextvar so the executor can
            # surface `eval.ground_truth` on the agent.run span (matches
            # eval_full.py behavior).
            _n_samples = len(test_samples)

            from src.harness.utils import eval_run_ground_truth as _gt_var
            from src.harness.sdk_config import get_sdk
            from src.evaluation.evaluate import _extract_model

            # Cache wrapper around agent.run, mirroring the pattern in
            # src/evaluation/evaluate.py:evaluate_agent_parallel. Without
            # this, training samples re-run full-price on every restart even
            # when the (program, question) pair is unchanged — wasteful for
            # iterative experiments where the same train UID gets sampled
            # against the same parent program multiple times across launches.
            _train_cache = self.cache  # may be None when --cache false
            _train_sdk = get_sdk()
            _train_model = _extract_model(self.agents.solver._get_options())

            async def _run_one_train(i: int, question: str, gt: str):
                tok = _gt_var.set(str(gt))
                try:
                    trace = None
                    if _train_cache is not None:
                        trace = _train_cache.get(
                            question,
                            self.agents.solver.response_model,
                            sdk=_train_sdk,
                            model=_train_model,
                        )
                    if trace is None:
                        trace = await self.agents.solver.run(
                            question, tag=f"train {i + 1}/{_n_samples}"
                        )
                        if _train_cache is not None:
                            _train_cache.set(
                                question, trace,
                                sdk=_train_sdk, model=_train_model,
                            )
                    return trace
                finally:
                    _gt_var.reset(tok)

            raw_results = await asyncio.gather(*[
                _run_one_train(i, question, str(answer))
                for i, (question, answer, _) in enumerate(test_samples)
            ], return_exceptions=True)
            traces = []
            for i, res in enumerate(raw_results):
                if isinstance(res, BaseException):
                    _log("", f"  [WARN] train {i + 1}/{_n_samples} crashed: {type(res).__name__}: {res}")
                    traces.append(
                        AgentTrace(
                            duration_ms=0, total_cost_usd=0.0, num_turns=0,
                            usage={}, result=f"{type(res).__name__}: {res}",
                            is_error=True, output=None,
                            parse_error=f"{type(res).__name__}: {res}",
                            messages=[],
                        )
                    )
                else:
                    traces.append(res)
            self._iter_cost += sum(t.total_cost_usd for t in traces)

            # Backfill base score from first iteration's training traces when deferred
            if iteration_count == 1 and self._val_is_train_subset:
                train_score = 0.0
                for t, (q, a, _) in zip(traces, test_samples):
                    if t.output:
                        train_score += self.scorer(q, str(t.output.final_answer), str(a))
                train_score = train_score / len(traces) if traces else 0.0
                self.manager.update_frontier("base", train_score, max_size=self.config.frontier_size)
                _log("", f"  -> Base score (from training): {train_score:.4f}")

            # Collect failures and persist ALL traces to DB for cross-iteration learning
            active_skills_now = self._get_active_skills()
            # Snapshot skill contents at this iteration so past traces preserve
            # the exact guidance the solver had (skills evolve across iterations).
            active_skill_contents = self._snapshot_active_skills(active_skills_now)
            review_tasks: list[dict] = []
            failures: list[tuple[AgentTrace, str, str, str, str]] = []  # (trace, agent_answer, ground_truth, category, question)
            # Per-sample (question, answer, category, prior_score) — captured here
            # so the mid-gate (post-mutation, pre-val) can re-run the same
            # samples on the new program and compare scores. The Y=0
            # threshold (no regressions on previously-passing samples)
            # requires knowing which samples were originally OK.
            train_sample_scores: list[tuple[str, str, str, float]] = []
            for trace, (question, answer, category) in zip(traces, test_samples):
                agent_answer = (
                    str(trace.output.final_answer) if trace.output else "[PARSE FAILED]"
                )
                avg_score = self.scorer(
                    question,
                    agent_answer.strip().lower(),
                    str(answer).strip().lower(),
                )
                status = "[OK]" if avg_score >= 0.8 else "[FAIL]"
                _log("", f"    {status} score={avg_score:.3f} [{category}] {question[:40]}...")
                self._emit("sample", question=question, category=category, score=avg_score, passed=avg_score >= 0.8)
                train_sample_scores.append((question, str(answer), category, float(avg_score)))

                # Per-sample eval span removed — the score is now stamped onto
                # the agent.run span's name + status by the executor (via the
                # eval_score_callback contextvar set in __init__), so a
                # separate span here would just duplicate the noise we just
                # un-cluttered. Per-sample reasoning + answer remain on the
                # agent.run span's output.value, accessible by clicking in.

                # Persist trace to DB (and individual .md file) for progressive disclosure
                trace_summary = trace.summarize()
                self.trace_db.insert(
                    iteration=f"iter-test-{actual_iteration}",
                    question=question,
                    ground_truth=str(answer),
                    agent_answer=agent_answer,
                    score=avg_score,
                    trace_summary=trace_summary,
                    active_skills=active_skills_now,
                    active_skill_contents=active_skill_contents,
                    num_turns=trace.num_turns,
                    category=category,
                    phase="train",
                )
                # Queue trace for async review (successful traces only — see below)
                if self.reviewer is not None:
                    review_tasks.append({
                        "iteration": f"iter-test-{actual_iteration}",
                        "question": question,
                        "ground_truth": str(answer),
                        "agent_answer": agent_answer,
                        "score": avg_score,
                        "trace_summary": trace_summary,
                        "active_skills": active_skills_now,
                    })

                if avg_score < 0.8:
                    failures.append((trace, agent_answer, answer, category, question))

            # Fire background reviewer ONLY when evolver won't run (no failures).
            # Failures → evolver does deeper analysis. Successes → reviewer captures patterns.
            if self.reviewer is not None and review_tasks and len(failures) == 0:
                asyncio.create_task(self.reviewer.review_traces_batch(review_tasks))

            # Determine optimization phase
            # Phase 1: accuracy is below threshold — evolve for correctness (failures drive it)
            # Phase 2: accuracy met — evolve for efficiency (cost/turns on successful traces)
            best_score = max(
                (s for _, s in self.manager.get_frontier_with_scores()), default=0.0
            )
            threshold = self.config.accuracy_threshold
            in_phase2 = (
                threshold is not None
                and best_score >= threshold
                and len(failures) == 0
            )

            if in_phase2:
                # All training samples passed. Hand the evolver the successful
                # traces so it can look for redundant steps and propose an
                # efficiency-focused skill edit — with accuracy still the hard
                # constraint.
                successful_traces: list[tuple[AgentTrace, str, str, str, str]] = [
                    (t, str(t.output.final_answer) if t.output else "", a, cat, q)
                    for t, (q, a, cat) in zip(traces, test_samples)
                    if t.output is not None
                ]
                avg_turns = sum(t.num_turns for t, _, _, _, _ in successful_traces) / max(len(successful_traces), 1)
                avg_cost = sum(
                    (t.total_cost_usd or 0) for t, _, _, _, _ in successful_traces
                ) / max(len(successful_traces), 1)
                _log("", f"  -> Phase 2: accuracy {best_score:.2f} >= {threshold:.2f}, optimizing efficiency (avg turns: {avg_turns:.0f}, avg cost: ${avg_cost:.3f})")
                mutation_result = await self._mutate_with_fallback(
                    parent, successful_traces, actual_iteration,
                    phase="efficiency",
                    phase_metrics={
                        "accuracy": best_score,
                        "accuracy_threshold": threshold,
                        "avg_turns": avg_turns,
                        "avg_cost_usd": avg_cost,
                        "n_samples": len(successful_traces),
                    },
                )
            elif len(failures) == 0:
                _log("", f"  -> All samples passed, no proposal needed")
                # No mutation, but train evaluation did spend $$$ — accumulate
                # and log so the final report reflects actual API spend
                # (without this, the bottom-of-loop accumulation is skipped by
                # the `continue` and Total cost ends up at $0).
                self._total_cost += self._iter_cost
                _log("COST", f"Iter {iteration_count} cost: ${self._iter_cost:.4f} | Running total: ${self._total_cost:.4f}")
                self._save_checkpoint(actual_iteration)
                continue
            else:
                _log("", f"  -> {len(failures)} failure(s), proposing improvement...")
                mutation_result = await self._mutate_with_fallback(parent, failures, actual_iteration)

            # Get parent's score for comparison
            parent_score = next(
                (score for name, score in self.manager.get_frontier_with_scores() if name == parent),
                0.0
            )

            if mutation_result is None:
                no_improvement_count += 1
            else:
                child_name, proposal, justification = mutation_result

                # Mid-gate: cheap sanity check on the iter's own train
                # samples before paying for the full val GATE. If the
                # proposed skill doesn't fix at least `mid_gate_min_fixed`
                # previously-failing samples, OR causes more than
                # `mid_gate_max_regressions` previously-passing samples to
                # regress, discard the mutation early and skip val eval.
                # Catches the common "mutation looks plausible but breaks
                # the cases it was designed to fix / breaks unrelated cases"
                # failure mode at a fraction of the val-eval cost.
                if self.config.mid_gate_enabled and train_sample_scores:
                    mid_gate_passed = await self._mid_gate_check(
                        child_name, train_sample_scores,
                    )
                    if not mid_gate_passed:
                        _log("", f"  [SKIP] Mid-gate failed — discarding {child_name} without running val eval")
                        self.manager.discard(child_name)
                        no_improvement_count += 1
                        # Switch back to parent so the next iter starts from
                        # the right tree state (mirrors the post-discard
                        # path in the val-gate branch below).
                        self.manager.switch_to(parent)
                        # Still accumulate the mid-gate cost into iter cost.
                        self._total_cost += self._iter_cost
                        _log("COST", f"Iter {iteration_count} cost: ${self._iter_cost:.4f} | Running total: ${self._total_cost:.4f}")
                        self._save_checkpoint(actual_iteration)
                        continue

                # Evaluate child
                _log("", f"  -> Evaluating {child_name}...")
                child_score, child_cost = await self._evaluate(self.val_data)  # accumulates to self._iter_cost

                # Update frontier or discard. Pass cost so the frontier can
                # tie-break on efficiency when accuracy is equal, and
                # parent_score/parent_cost so it can REJECT regressions
                # (a child scoring lower than its parent shouldn't be admitted
                # even when there's room — that pollutes the frontier with
                # bad mutations the proposer then has to "remember not to
                # propose again").
                parent_cost = self.manager._get_program_cost(parent) if parent else None
                added = self.manager.update_frontier(
                    child_name, child_score, max_size=self.config.frontier_size,
                    cost=child_cost,
                    parent_score=parent_score,
                    parent_cost=parent_cost,
                )

                if added:
                    _log("", f"  [OK] Added to frontier (score: {child_score:.4f}, cost: ${child_cost:.4f})")
                    outcome = "improved" if child_score > parent_score else "kept"
                    no_improvement_count = 0
                else:
                    _log("", f"  [SKIP] Discarded (score: {child_score:.4f}, cost: ${child_cost:.4f})")
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
            current_options = self.agents.solver._get_options()
            base_config = options_to_config(current_options, "base")
            self.manager.create_program("base", base_config)
            _log("INIT", "Created base program")
        else:
            _log("INIT", "Using existing base program")

        # Evaluate and add base to frontier
        self.manager.switch_to("base")
        if self._val_is_train_subset:
            _log("", f"  -> Deferring base eval (val overlaps with train; first-iter training traces will backfill)")
            self.manager.update_frontier("base", 0.0, max_size=self.config.frontier_size)
            _log("", f"  -> Frontier: {self.manager.get_frontier()}")
            self._emit("baseline", score=0.0)
            return
        _log("", f"  -> Evaluating on {len(self.val_data)} samples...")
        self._iter_cost = 0.0
        base_score, base_cost = await self._evaluate(self.val_data)
        self._total_cost += self._iter_cost
        self.manager.update_frontier(
            "base", base_score, max_size=self.config.frontier_size,
            cost=base_cost,
        )
        _log("", f"  -> Base score: {base_score:.4f} (avg cost: ${base_cost:.4f})")
        _log("", f"  -> Frontier: {self.manager.get_frontier()}")
        _log("COST", f"Base eval cost: ${self._iter_cost:.4f} | Total: ${self._total_cost:.4f}")
        self._emit("baseline", score=base_score)

    async def _mid_gate_check(
        self,
        child_name: str,
        train_sample_scores: list[tuple[str, str, str, float]],
    ) -> bool:
        """Cheap pre-val sanity check on the iter's training samples.

        Re-runs the same train samples on the new child program and compares
        per-sample scores against the prior (parent) scores. Pass criteria:
            - At least `mid_gate_min_fixed` previously-failing samples now pass
            - At most `mid_gate_max_regressions` previously-passing samples now fail

        A pass means "the proposed skill demonstrably moves the needle on the
        cases it was designed to fix without obvious collateral damage."
        Returns True if mid-gate passes (proceed to full val GATE), False if
        it fails (discard mutation, save val budget).

        Args:
            child_name: branch / program name for logging
            train_sample_scores: list of (question, answer, category, prior_score)
                captured during this iter's train eval

        Cost: ~1 inference per train sample (typically 3-5 samples × ~$1).
        """
        n = len(train_sample_scores)
        _log("", f"  -> Mid-gate: re-running {n} train sample(s) on {child_name}...")

        # Use the same cache+run wrapper as _run_one_train so writes flow
        # to the shared run-cache and future re-runs benefit.
        from src.harness.utils import eval_run_ground_truth as _gt_var
        from src.harness.sdk_config import get_sdk
        from src.evaluation.evaluate import _extract_model

        cache = self.cache
        sdk = get_sdk()
        model = _extract_model(self.agents.solver._get_options())

        async def _run_one(i: int, q: str, gt: str) -> AgentTrace:
            tok = _gt_var.set(gt)
            try:
                trace = None
                if cache is not None:
                    trace = cache.get(q, self.agents.solver.response_model, sdk=sdk, model=model)
                if trace is None:
                    trace = await self.agents.solver.run(q, tag=f"midgate {i + 1}/{n}")
                    if cache is not None:
                        cache.set(q, trace, sdk=sdk, model=model)
                return trace
            finally:
                _gt_var.reset(tok)

        raw = await asyncio.gather(*[
            _run_one(i, q, a) for i, (q, a, _c, _s) in enumerate(train_sample_scores)
        ], return_exceptions=True)

        fixed = 0
        regressed = 0
        same = 0
        sample_lines: list[str] = []
        midgate_cost = 0.0
        for (q, a, c, prior_score), res in zip(train_sample_scores, raw):
            if isinstance(res, BaseException) or res is None or getattr(res, "output", None) is None:
                new_score = 0.0
            else:
                new_score = float(self.scorer(q, str(res.output.final_answer), str(a)))
                midgate_cost += float(getattr(res, "total_cost_usd", 0) or 0)

            prior_passed = prior_score >= 0.8
            now_passed = new_score >= 0.8
            if not prior_passed and now_passed:
                fixed += 1
                marker = "✓ fixed"
            elif prior_passed and not now_passed:
                regressed += 1
                marker = "✗ regressed"
            else:
                same += 1
                marker = "= same"
            sample_lines.append(
                f"      {marker:13s} {prior_score:.2f} → {new_score:.2f}  {q[:50]}..."
            )

        for line in sample_lines:
            _log("", line)

        self._iter_cost += midgate_cost
        _log("", f"  -> Mid-gate result: fixed={fixed}, regressed={regressed}, same={same}, cost=${midgate_cost:.4f}")

        passed = (
            fixed >= self.config.mid_gate_min_fixed
            and regressed <= self.config.mid_gate_max_regressions
        )
        if not passed:
            reasons = []
            if fixed < self.config.mid_gate_min_fixed:
                reasons.append(f"fixed {fixed} < min {self.config.mid_gate_min_fixed}")
            if regressed > self.config.mid_gate_max_regressions:
                reasons.append(f"regressed {regressed} > max {self.config.mid_gate_max_regressions}")
            _log("", f"  -> Mid-gate FAILED ({'; '.join(reasons)})")
        else:
            _log("", f"  -> Mid-gate PASSED — proceeding to full val GATE")
        return passed

    async def _evaluate(self, data: list[tuple[str, str, str]]) -> tuple[float, float]:
        """Evaluate base agent on data.

        Args:
            data: List of (question, answer, category) tuples.

        Returns:
            (accuracy_score, avg_cost_usd) — accuracy in [0.0, 1.0], avg cost
            averaged across the provided samples. avg_cost is 0.0 when no
            samples produced a trace with a non-null cost.
        """
        # Convert to (question, answer) format for evaluate_agent_parallel
        qa_data = [(q, a) for q, a, _ in data]
        results = await evaluate_agent_parallel(
            self.agents.solver, qa_data, max_concurrent=self.config.concurrency, cache=self.cache
        )

        score = 0.0
        total_cost = 0.0
        cost_samples = 0
        for result in results:
            if result.trace is not None:
                c = result.trace.total_cost_usd or 0.0
                self._iter_cost += c
                total_cost += c
                cost_samples += 1
            # Per-question score logging — without it the operator only sees
            # an aggregate `Base score: X.XXXX` after every val run, which
            # hides which specific questions the agent missed and by how
            # much (cosine decay can produce fractional scores too).
            if result.trace is None or result.trace.output is None:
                _log("", f"    score=0.000 [val] {result.question[:60]}... (no output)")
                continue
            q_score = self.scorer(
                result.question,
                str(result.trace.output.final_answer),
                str(result.ground_truth),
            )
            score += q_score
            _log("", f"    score={q_score:.3f} [val] {result.question[:60]}...")
        avg_score = score / len(results) if results else 0.0
        avg_cost = (total_cost / cost_samples) if cost_samples else 0.0
        return avg_score, avg_cost

    async def _mutate(
        self,
        parent: str,
        failures: list[tuple[AgentTrace[AgentResponse], str, str, str, str]],
        iteration: int,
        truncation_level: int = 0,
        *,
        phase: str = "accuracy",
        phase_metrics: dict | None = None,
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
        _log("", f"  -> Running {evolution_mode.replace('_only', '').replace('_unified', ' evolver')} with {len(failures)} failures...")
        feedback_history = read_feedback_history(self._feedback_path)

        # Progressive disclosure: give the evolver a lightweight index of past traces.
        # It can Read any specific trace file on demand via the file paths in the index.
        failed_questions = [q for (_, _, _, _, q) in failures]
        past_traces_index = self.trace_db.generate_index(failed_questions=failed_questions)

        # Unconsumed proposals from the background reviewer. The reviewer
        # writes to the same SQLite DB as TraceDB from a separate context;
        # under WAL with multiple connections we occasionally see
        # `disk I/O error`. Treat it as non-fatal — the evolver can still run
        # without runtime proposals, so just log and continue.
        runtime_proposals = ""
        if self.reviewer is not None:
            try:
                runtime_proposals = self.reviewer.format_proposals_for_reflector(max_chars=10_000)
            except Exception as e:
                _log("", f"  [WARN] Background reviewer unavailable ({type(e).__name__}: {e}) — evolver will run without runtime proposals")
                runtime_proposals = ""

        proposer_query = build_proposer_query(
            failures, feedback_history, evolution_mode, truncation_level,
            self.task_constraints,
            # `project_root` is the workspace (kept for backward compat with
            # other internal lookups). `project_skills_dir` is the load-bearing
            # path for skill enumeration (correct under workspace/project split).
            project_root=self._project_root,
            project_skills_dir=self._project_skills_dir,
            iter_traces_dir=self._project_root / ".cache" / "current_iter_traces",
            past_traces_index=past_traces_index,
            runtime_proposals=runtime_proposals,
            phase=phase,
            phase_metrics=phase_metrics,
            solver_prompt=self.solver_prompt,
            data_root=self._data_root,
        )

        if evolution_mode == "skill_unified":
            if self.agents.skill_evolver is None:
                _log("", f"  [WARN] skill_unified mode requires skill_evolver agent but none configured")
                return None

            # Capture parent config BEFORE the evolver mutates files.
            # The evolver writes skills in-place while running; reading program.yaml
            # after the fact can fail if the working tree has been altered.
            parent_config = self.manager.get_current()
            child_name = f"iter-skill-{actual_iteration}"

            # The evolver agent itself can hit its 12-min wall-clock budget
                # (seen with deep failure-analysis runs that walk many trace
                # files). agent.py raises TimeoutError on budget exhaustion
                # and intentionally does NOT retry — but the evolver call site
                # used to crash the whole loop on that exception. Treat it
                # the same as any other evolver failure: log and skip this
                # iteration's mutation. The next iter will start fresh from
                # the current frontier, so we lose one mutation cycle, not
                # the whole experiment.
            try:
                evolver_trace = await self.agents.skill_evolver.run(proposer_query)
            except TimeoutError as e:
                _log("", f"  [WARN] Skill evolver timed out: {e}")
                return None
            self._iter_cost += evolver_trace.total_cost_usd

            if evolver_trace.output is None:
                _log("", f"  [WARN] Skill evolver failed: {evolver_trace.parse_error}")
                return None

            output = evolver_trace.output
            action_type = output.action
            target_skill = output.skill_name
            proposed = output.description
            justification = output.justification

            action_label = f"edit:{target_skill}" if action_type == "edit" else f"create:{target_skill}"
            _log("", f"  -> Evolved: {action_label} - {proposed[:60]}...")
            self._emit("proposal", action=action_type, target_skill=target_skill, summary=proposed[:80])

            # Now create child branch — skill edits are in the working tree from the evolver
            child_config = parent_config.mutate(child_name)
            self.manager.create_program(child_name, child_config, parent=parent)

            self._emit("skill_written", name=target_skill, action=action_type, target=target_skill)

        elif evolution_mode == "skill_only":
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

            if is_opencode_sdk() or is_openhands_sdk() or is_goose_sdk() or is_codex_sdk():
                from src.harness.opencode.skill_utils import normalize_project_skill_frontmatter
                from src.harness.sdk_config import get_sdk
                skill_descriptions: dict[str, str] = {}
                if target_skill:
                    skill_descriptions[target_skill] = proposed
                if created_skill:
                    skill_descriptions[created_skill] = proposed
                normalize_project_skill_frontmatter(
                    self._project_root,
                    descriptions=skill_descriptions,
                    fallback_description=proposed,
                    compatibility=get_sdk(),
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

        # Constraint gate: validate generated/edited skills BEFORE committing
        # (saves expensive evaluation cost on degenerate mutations)
        if evolution_mode in ("skill_only", "skill_unified"):
            skills_dir = self._project_skills_dir
            gate_failed = False
            if skills_dir.exists():
                for skill_dir in skills_dir.iterdir():
                    skill_file = skill_dir / "SKILL.md"
                    if not skill_dir.is_dir() or not skill_file.exists():
                        continue
                    # Skip meta-skills (brainstorming, skill-creator) that ship with the project
                    if skill_dir.name in ("skill-creator", "brainstorming"):
                        continue
                    passed, summary = constraint_gate(skill_file)
                    if not passed:
                        _log("", f"  [GATE] {skill_dir.name}: FAILED")
                        _log("", summary)
                        gate_failed = True
                    elif "WARNING" in summary:
                        _log("", f"  [GATE] {skill_dir.name}: PASSED with warnings")

            if gate_failed:
                _log("", f"  [GATE] Skill constraint check failed — discarding mutation")
                return None

        # Commit changes
        self.manager.commit(f"{child_name}: {proposed[:50]}")

        # Return mutation info (feedback will be written by caller with outcome)
        return (child_name, proposed, justification)

    async def _mutate_with_fallback(
        self,
        parent: str,
        failures: list[tuple[AgentTrace[AgentResponse], str, str, str, str]],
        iteration: int,
        *,
        phase: str = "accuracy",
        phase_metrics: dict | None = None,
    ) -> tuple[str, str, str] | None:
        """Try progressive truncation levels, then single-failure fallback.

        Args:
            parent: Name of the parent program.
            failures: List of (trace, agent_answer, ground_truth, category) tuples.
            iteration: Current iteration number.
            phase: "accuracy" (default) or "efficiency". Controls the framing
                   of the query sent to the evolver.
            phase_metrics: Optional aggregate metrics (avg_turns, avg_cost_usd,
                           accuracy, accuracy_threshold, n_samples) used to
                           ground the efficiency-phase prompt in real numbers.

        Returns:
            Tuple of (child_name, proposal, justification) if created, None otherwise.
        """
        max_level = self.config.proposer_max_truncation_level

        for truncation_level in range(max_level + 1):
            if truncation_level > 0:
                _log("", f"  -> Retrying with truncation level {truncation_level}...")

            result = await self._mutate(
                parent, failures, iteration, truncation_level,
                phase=phase, phase_metrics=phase_metrics,
            )
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
        failures: list[tuple[AgentTrace[AgentResponse], str, str, str, str]],
    ) -> tuple[AgentTrace[AgentResponse], str, str, str, str]:
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

    @staticmethod
    def _build_proportional_schedule(
        train_pools: dict[str, list],
    ) -> list[str]:
        """Build a category schedule weighted by pool size.

        Each category `c` appears exactly `len(train_pools[c])` times in the
        returned list, interleaved using Bresenham/largest-deficit selection
        so big categories aren't bunched together. The list is then cycled
        across iterations to drive proportional per-category sampling.
        """
        cats = list(train_pools.keys())
        sizes = {c: len(train_pools[c]) for c in cats}
        total = sum(sizes.values())
        if total == 0:
            return []
        targets = {c: 0.0 for c in cats}
        actuals = {c: 0 for c in cats}
        rate = {c: sizes[c] / total for c in cats}
        schedule: list[str] = []
        for _ in range(total):
            for c in cats:
                targets[c] += rate[c]
            # Pick the category with the largest deficit (target - actual).
            # Tiebreak: larger pool first, then insertion order.
            best = max(
                cats,
                key=lambda c: (targets[c] - actuals[c], sizes[c], -cats.index(c)),
            )
            schedule.append(best)
            actuals[best] += 1
        return schedule

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
        skills_dir = self._project_skills_dir
        active_skills = []
        if skills_dir.exists():
            for skill_dir in skills_dir.iterdir():
                if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                    active_skills.append(skill_dir.name)
        return sorted(active_skills)

    def _snapshot_active_skills(self, skill_names: list[str]) -> dict[str, str]:
        """Read the current content of each active SKILL.md file.

        Captures skill content at trace-save time so that past traces
        preserve the exact guidance the solver had at that moment, even
        after the skill has been edited in subsequent iterations.
        """
        skills_dir = self._project_skills_dir
        snapshots: dict[str, str] = {}
        for name in skill_names:
            skill_file = skills_dir / name / "SKILL.md"
            if skill_file.exists():
                try:
                    snapshots[name] = skill_file.read_text()
                except Exception:
                    pass  # best-effort
        return snapshots

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
