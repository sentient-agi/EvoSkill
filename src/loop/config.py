"""Configuration for the self-improving loop."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


EvolutionMode = Literal["prompt_only", "skill_only"]


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

    # Multi-sample failure analysis: test this many samples before proposing
    # Helps identify patterns rather than overfitting to single failures
    failure_sample_count: int = 3

    # Sample selection seed: None = sequential sampling, int = random with seed
    sample_seed: int | None = None

    # Feedback configuration
    reset_feedback: bool = True

    # Continue mode: False = start fresh (reset iteration numbering),
    # True = continue from existing frontier/branch
    continue_mode: bool = False

    # Cache configuration
    cache_enabled: bool = True
    cache_dir: Path = field(default_factory=lambda: Path(".cache/runs"))
    cache_store_messages: bool = False
