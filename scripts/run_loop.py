#!/usr/bin/env python3
"""Run self-improving agent loop."""

from src.tracing import init_tracing
init_tracing("evoskill-loop")

import asyncio
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)

from src.loop import SelfImprovingLoop, LoopConfig, LoopAgents
from src.harness import Agent, set_sdk
from src.agent_profiles import (
    base_agent_options,
    make_base_agent_options,
    skill_proposer_options,
    prompt_proposer_options,
    skill_generator_options,
    prompt_generator_options,
    make_skill_evolver_options,
)
from src.agent_profiles.skill_proposer import make_skill_proposer_options
from src.agent_profiles.skill_generator import make_skill_generator_options
from src.agent_profiles.prompt_proposer import make_prompt_proposer_options
from src.agent_profiles.prompt_generator import make_prompt_generator_options
from src.registry import ProgramManager
from src.schemas import (
    AgentResponse,
    SkillProposerResponse,
    PromptProposerResponse,
    ToolGeneratorResponse,
    PromptGeneratorResponse,
    SkillEvolverResponse,
)


class LoopSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        cli_parse_args=True,
        title="Run self-improving agent loop",
    )

    mode: Literal["skill_only", "prompt_only", "skill_unified"] = Field(
        default="skill_unified",
        description="Evolution mode: 'skill_unified' (default, combined evolver), "
                    "'skill_only' (split proposer+generator), or 'prompt_only'",
    )
    max_iterations: int = Field(
        default=20, description="Maximum number of improvement iterations"
    )
    frontier_size: int = Field(
        default=3, description="Number of top-performing programs to keep"
    )
    no_improvement_limit: int = Field(
        default=5, description="Stop after this many iterations without improvement"
    )
    concurrency: int = Field(default=4, description="Number of concurrent evaluations")
    failure_samples: int = Field(
        default=3,
        description="Number of samples to test per iteration for pattern detection",
    )
    cache: bool = Field(default=True, description="Enable run caching")
    reset_feedback: bool = Field(
        default=True, description="Reset feedback history on start"
    )
    continue_loop: bool = Field(
        default=False,
        description="Continue from existing frontier/branch instead of starting fresh",
    )
    fresh: bool = Field(
        default=False,
        description="Wipe all program branches, frontier tags, feedback, checkpoint, and trace DB before running",
    )
    dataset: str = Field(
        default=".dataset/new_runs_base/solved_dataset.csv",
        description="Path to dataset CSV with category column",
    )
    train_ratio: float = Field(
        default=0.18, description="Fraction of each category for training"
    )
    val_ratio: float = Field(
        default=0.12, description="Fraction of each category for validation"
    )
    val_count: Optional[int] = Field(
        default=None, description="Override total validation count"
    )
    model: Optional[str] = Field(
        default=None, description="Model for base/solver agent (opus, sonnet, haiku)"
    )
    evolver_model: Optional[str] = Field(
        default=None,
        description="Model for evolver/reflector agent (defaults to 'opus')",
    )
    sdk: Literal["claude", "opencode", "codex", "goose", "openhands"] = Field(
        default="claude",
        description="SDK to use: 'claude', 'opencode', 'codex', 'goose', or 'openhands'",
    )
    accuracy_threshold: Optional[float] = Field(
        default=None,
        description="Accuracy threshold to switch from accuracy→efficiency optimization (e.g. 0.8)",
    )
    reviewer_enabled: bool = Field(
        default=True,
        description="Enable background reviewer (Haiku) for runtime insight extraction on successful iterations",
    )
    data_root: Optional[str] = Field(
        default=None,
        description="Data root directory for agent cwd and add_dirs (e.g. /path/to/pdf/dataset). Default: project root.",
    )


def stratified_split(
    data: pd.DataFrame, train_ratio: float = 0.18, val_ratio: float = 0.12
) -> tuple[dict[str, list[tuple[str, str]]], list[tuple[str, str, str]]]:
    """Split data ensuring each category has at least 1 in both train and validation."""
    if train_ratio + val_ratio > 1.0:
        raise ValueError(
            f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) cannot exceed 1.0"
        )

    data = data.dropna(subset=["category"])
    categories = data["category"].unique()
    train_pools: dict[str, list[tuple[str, str]]] = {}
    val_data: list[tuple[str, str, str]] = []

    for cat in categories:
        cat_data = data[data["category"] == cat].sample(frac=1, random_state=42)
        n_train = max(1, int(len(cat_data) * train_ratio))
        n_val = max(1, int(len(cat_data) * val_ratio))

        train_pools[cat] = [
            (row.question, row.ground_truth)
            for _, row in cat_data.head(n_train).iterrows()
        ]
        val_data.extend(
            [
                (row.question, row.ground_truth, cat)
                for _, row in cat_data.iloc[n_train : n_train + n_val].iterrows()
            ]
        )

    return train_pools, val_data


def _fresh_reset(project_root: str | Path) -> None:
    """Wipe all evolutionary state: program branches, frontier tags, feedback, checkpoint, trace DB."""
    import subprocess

    root = Path(project_root)
    print("[FRESH] Resetting all evolutionary state...")

    # Switch away from any program/* branch so we can delete them
    try:
        current = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=root, text=True,
        ).strip()
        if current.startswith("program/"):
            branches = subprocess.check_output(
                ["git", "branch", "--format=%(refname:short)"], cwd=root, text=True,
            ).strip().split("\n")
            non_program = [b for b in branches if not b.startswith("program/")]
            if non_program:
                target = non_program[0]
                subprocess.run(["git", "checkout", target], cwd=root, check=False, capture_output=True)
                print(f"[FRESH] Switched from {current} to {target}")
    except subprocess.CalledProcessError:
        pass

    # Delete program/* branches
    try:
        branches = subprocess.check_output(
            ["git", "branch", "--list", "program/*", "--format=%(refname:short)"],
            cwd=root, text=True,
        ).strip().split("\n")
        branches = [b for b in branches if b]
        if branches:
            subprocess.run(["git", "branch", "-D"] + branches, cwd=root, check=False, capture_output=True)
            print(f"[FRESH] Deleted {len(branches)} program branches")
    except subprocess.CalledProcessError:
        pass

    # Delete frontier/* tags
    try:
        tags = subprocess.check_output(
            ["git", "tag", "-l", "frontier/*"], cwd=root, text=True,
        ).strip().split("\n")
        tags = [t for t in tags if t]
        if tags:
            subprocess.run(["git", "tag", "-d"] + tags, cwd=root, check=False, capture_output=True)
            print(f"[FRESH] Deleted {len(tags)} frontier tags")
    except subprocess.CalledProcessError:
        pass

    # Delete state files
    for name in [".claude/feedback_history.md", ".claude/loop_checkpoint.json",
                 ".cache/traces.db", ".cache/traces.db-wal", ".cache/traces.db-shm"]:
        p = root / name
        if p.exists():
            p.unlink()
            print(f"[FRESH] Removed {name}")

    # Wipe individual trace files (kept separate from DB deletion)
    traces_dir = root / ".cache" / "traces"
    if traces_dir.exists():
        import shutil
        shutil.rmtree(traces_dir)
        print(f"[FRESH] Removed .cache/traces/")


async def main(settings: LoopSettings):
    # Set SDK based on CLI argument
    set_sdk(settings.sdk)

    project_root = Path.cwd()
    data_root = settings.data_root or str(project_root)

    # Optional fresh reset: wipe all prior evolutionary state
    if settings.fresh:
        _fresh_reset(project_root)

    data = pd.read_csv(settings.dataset)
    train_pools, val_data = stratified_split(
        data, train_ratio=settings.train_ratio, val_ratio=settings.val_ratio
    )

    categories = list(train_pools.keys())
    total_train = sum(len(pool) for pool in train_pools.values())
    print(f"Dataset: {settings.dataset}")
    print(f"Categories ({len(categories)}): {', '.join(categories)}")
    print(
        f"Training pools: {', '.join(f'{cat}: {len(pool)}' for cat, pool in train_pools.items())}"
    )
    print(f"Total training samples: {total_train}")
    print(
        f"Validation samples: {len(val_data)} ({settings.val_ratio:.0%} per category, min 1 each)"
    )
    print(
        f"Split ratios: train={settings.train_ratio:.0%}, val={settings.val_ratio:.0%} "
        f"(remaining {1 - settings.train_ratio - settings.val_ratio:.0%} unused)"
    )

    # Base agent: cwd = EvoSkill (owns skills + scratch); data_root accessible via add_dirs.
    # Using data_root as cwd would pollute the data folder with agent-generated files.
    data_dirs = [data_root] if settings.data_root else None
    base_options = (
        make_base_agent_options(
            model=settings.model,
            project_root=str(project_root),
            data_dirs=data_dirs,
        )
        if (settings.model or data_dirs)
        else base_agent_options
    )

    # Improvers (proposers/generators/evolver) default to Opus for quality.
    # Override with --evolver_model if you want them cheaper.
    improver_model = settings.evolver_model or "opus"

    # Build agents, each with a name for better OTel span labels
    agents = LoopAgents(
        base=Agent(base_options, AgentResponse, name="base"),
        skill_proposer=Agent(
            make_skill_proposer_options(model=improver_model),
            SkillProposerResponse, name="skill_proposer",
        ),
        prompt_proposer=Agent(
            make_prompt_proposer_options(model=improver_model),
            PromptProposerResponse, name="prompt_proposer",
        ),
        skill_generator=Agent(
            make_skill_generator_options(model=improver_model),
            ToolGeneratorResponse, name="skill_generator",
        ),
        prompt_generator=Agent(
            make_prompt_generator_options(model=improver_model),
            PromptGeneratorResponse, name="prompt_generator",
        ),
        skill_evolver=(
            Agent(
                make_skill_evolver_options(model=improver_model),
                SkillEvolverResponse,
                name="skill_evolver",
            )
            if settings.mode == "skill_unified"
            else None
        ),
    )
    manager = ProgramManager(cwd=project_root)

    config = LoopConfig(
        max_iterations=settings.max_iterations,
        frontier_size=settings.frontier_size,
        no_improvement_limit=settings.no_improvement_limit,
        concurrency=settings.concurrency,
        evolution_mode=settings.mode,
        failure_sample_count=settings.failure_samples,
        categories_per_batch=settings.failure_samples,
        cache_enabled=settings.cache,
        reset_feedback=settings.reset_feedback,
        continue_mode=settings.continue_loop,
        accuracy_threshold=settings.accuracy_threshold,
        cost_metric="total_cost_usd" if settings.accuracy_threshold else None,
        reviewer_enabled=settings.reviewer_enabled,
    )

    model_info = f", model={settings.model}" if settings.model else ""
    evolver_info = f", evolver_model={settings.evolver_model}" if settings.evolver_model else ""
    print(f"Running loop with evolution_mode={settings.mode}{model_info}{evolver_info}")
    loop = SelfImprovingLoop(config, agents, manager, train_pools, val_data)
    result = await loop.run()

    print(f"Best: {result.best_program} ({result.best_score:.2%})")
    print(f"Frontier: {result.frontier}")
    print(f"Total cost: ${result.total_cost_usd:.4f}")


if __name__ == "__main__":
    settings = LoopSettings()
    asyncio.run(main(settings))
