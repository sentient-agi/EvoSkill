"""Configuration for the self-improving loop."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


EvolutionMode = Literal["prompt_only", "skill_only", "skill_unified"]
SelectionStrategy = Literal["best", "random", "round_robin"]
CostMetric = Literal["total_cost_usd", "num_turns", "duration_ms"]


@dataclass
class LoopConfig:
    """Configuration parameters for SelfImprovingLoop.

    Attributes:
        max_iterations: Maximum number of improvement iterations.
        frontier_size: Number of top-performing programs to keep.
        no_improvement_limit: Stop early after this many iterations without improvement.
        tolerance: Tolerance for answer matching (0.0 = exact match).
        concurrency: Number of concurrent evaluations.
        evolution_mode: Which dimension to evolve ("prompt_only" or "skill_only").
        selection_strategy: Parent selection from frontier — "best" (greedy, default),
            "random" (uniform random), or "round_robin" (cycle through ranked members).
        reset_feedback: Whether to reset feedback_history.md on fresh loop run.
        cache_enabled: Whether to enable run caching.
        cache_dir: Directory for cache storage.
        cache_store_messages: Whether to store full message history in cache.
    """

    max_iterations: int = 5
    frontier_size: int = 3
    no_improvement_limit: int = 5
    tolerance: float = 0.0
    concurrency: int = 4

    # Evolution mode: which dimension to optimize
    evolution_mode: EvolutionMode = "skill_only"

    # Parent selection strategy: how to pick the next parent from the frontier
    selection_strategy: SelectionStrategy = "best"

    # Multi-sample failure analysis: test this many samples before proposing
    # Helps identify patterns rather than overfitting to single failures
    failure_sample_count: int = 3

    # Category-aware sampling: number of categories to sample per batch
    # (capped by actual number of categories and failure_sample_count)
    categories_per_batch: int = 3

    # Feedback configuration
    reset_feedback: bool = True

    # Continue mode: False = start fresh (reset iteration numbering),
    # True = continue from existing frontier/branch
    continue_mode: bool = False

    # Cache configuration
    cache_enabled: bool = True
    cache_dir: Path = field(default_factory=lambda: Path(".cache/runs"))
    cache_store_messages: bool = False

    # Mid-gate: cheap sanity check between evolver proposal and full val
    # GATE eval. Re-runs the iter's training samples on the new program and
    # requires "fixes ≥ mid_gate_min_fixed AND regressions ≤ mid_gate_max_regressions"
    # before proceeding. Catches obvious bad mutations (no failures fixed,
    # or new regressions on previously-passing train samples) before
    # spending the much more expensive val budget.
    mid_gate_enabled: bool = True
    mid_gate_min_fixed: int = 1     # X — at least N previously-failing samples must now pass
    mid_gate_max_regressions: int = 0  # Y — at most M previously-passing samples may now fail

    # Proposer resilience: adaptive truncation on context limit/timeout
    proposer_max_truncation_level: int = 2  # Max truncation level (0=full, 1=moderate, 2=aggressive)
    proposer_single_failure_fallback: bool = True  # Try single shortest failure if all levels fail
    consecutive_proposer_failures_limit: int = 5  # Stop after N consecutive proposer failures

    # Multi-sample per category: collect N samples per category before proposing
    samples_per_category: int = 2  # Helps identify patterns within categories

    # Proportional sampling: when True, override round-robin and draw
    # `failure_sample_count` samples per iteration from a precomputed schedule
    # weighted by per-category pool size. Over many iterations each category
    # is sampled in proportion to its training-pool size.
    proportional_sampling: bool = False

    # Pareto optimization: track cost alongside accuracy.
    # When set, frontier uses Pareto dominance instead of single-axis score comparison.
    # None = single-axis (accuracy only, original behavior).
    cost_metric: CostMetric | None = None

    # Lexicographic multi-objective optimization:
    # Phase 1: optimize accuracy until accuracy_threshold is met
    # Phase 2: optimize cost/efficiency while maintaining accuracy
    # Set accuracy_threshold to enable. None = single-objective (accuracy only).
    accuracy_threshold: float | None = None  # e.g. 0.8 = switch to phase 2 at 80% accuracy

    # Background reviewer: async LLM extracts insights from successful solver traces.
    # Complements the failure-driven evolver. Set False to disable.
    reviewer_enabled: bool = True
