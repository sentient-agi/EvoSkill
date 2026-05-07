#!/usr/bin/env python3
"""Run self-improving agent loop."""

import os

# Raise CLI output-token ceiling before the SDK spawns any subprocess.
# Default is 32k which subagents can blow through on long reads. Sonnet 4.6
# supports up to 64k; Opus 4.7 up to 128k. We set 64k as a safe upper bound
# that works for both. Inherited by every `claude` child process the SDK spawns.
os.environ.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "64000")

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
    officeqa_agent_options,
    make_officeqa_agent_options,
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
        description="Number of categories per iteration (also sets failure_sample_count). Total train samples per iter = failure_samples * samples_per_category.",
    )
    samples_per_category: int = Field(
        default=2,
        description="Number of training samples drawn per category per iteration. Total samples per iter = failure_samples * samples_per_category.",
    )
    proportional_sampling: bool = Field(
        default=False,
        description="If True, draw failure_samples samples per iter from a schedule weighted by per-category pool size (overrides round-robin). Over many iters each category is sampled proportional to its pool.",
    )
    cache: bool = Field(default=True, description="Enable run caching")
    mid_gate: bool = Field(
        default=True,
        description="Re-run iter's train samples after evolver mutation (cheap pre-check). Discards mutations that don't fix the failures they target or regress previously-passing samples — saves the full val-eval cost. Aggressive default: requires ≥1 fix and 0 regressions.",
    )
    mid_gate_min_fixed: int = Field(
        default=1,
        description="Minimum number of previously-failing train samples that must now pass for mid-gate to succeed.",
    )
    mid_gate_max_regressions: int = Field(
        default=0,
        description="Maximum number of previously-passing train samples allowed to regress before mid-gate fails.",
    )
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
        description=(
            "Single-file dataset CSV (question, ground_truth, category). "
            "Auto-split via --train_ratio / --val_ratio. Ignored when both "
            "--train_dataset and --val_dataset are provided."
        ),
    )
    train_dataset: Optional[str] = Field(
        default=None,
        description=(
            "Pre-split training CSV (question, ground_truth, category). "
            "When paired with --val_dataset, disables auto-split and uses "
            "both files as-is."
        ),
    )
    val_dataset: Optional[str] = Field(
        default=None,
        description=(
            "Pre-split validation CSV (question, ground_truth, category). "
            "Must be used together with --train_dataset."
        ),
    )
    train_ratio: float = Field(
        default=0.18,
        description="Fraction of each category for training (auto-split only)",
    )
    val_ratio: float = Field(
        default=0.12,
        description="Fraction of each category for validation (auto-split only)",
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
    workspace: Optional[str] = Field(
        default="/Users/dastin/dev/evoskill-workspace",
        description=(
            "Separate git repo for evolutionary artifacts (program/* branches, "
            "frontier/* tags, feedback, checkpoints). Keeps the EvoSkill source "
            "tree clean. Set to an empty string to fall back to project root."
        ),
    )
    base_thinking: Optional[Literal["adaptive", "enabled", "disabled"]] = Field(
        default=None,
        description="Thinking config for base agent. 'adaptive' lets the model self-pace.",
    )
    base_effort: Optional[Literal["low", "medium", "high", "max"]] = Field(
        default=None,
        description="Effort tier for base agent (layers on top of thinking).",
    )
    evolver_thinking: Optional[Literal["adaptive", "enabled", "disabled"]] = Field(
        default=None,
        description="Thinking config for evolver/improver agents.",
    )
    evolver_effort: Optional[Literal["low", "medium", "high", "max"]] = Field(
        default=None,
        description="Effort tier for evolver/improver agents.",
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
        n_cat = len(cat_data)
        n_train = max(1, int(n_cat * train_ratio))
        n_val = max(1, int(n_cat * val_ratio))

        # Tiny-dataset fallback: when there's only one sample per category, the
        # same question must serve as both train and val. The loop's
        # _val_is_train_subset detection then reuses training traces for
        # evaluation instead of running val separately.
        if n_cat <= 1:
            train_pools[cat] = [
                (row.question, row.ground_truth)
                for _, row in cat_data.iterrows()
            ]
            val_data.extend(
                [
                    (row.question, row.ground_truth, cat)
                    for _, row in cat_data.iterrows()
                ]
            )
            continue

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


def load_presplit(
    train_csv: str | Path, val_csv: str | Path
) -> tuple[dict[str, list[tuple[str, str]]], list[tuple[str, str, str]]]:
    """Load pre-split train and val CSVs as-is (no shuffling, no stratification).

    Both files must have columns: question, ground_truth, category.

    Returns:
        train_pools: Dict[category -> list of (question, ground_truth) tuples].
        val_data: List of (question, ground_truth, category) tuples.
    """
    def _load(path: str | Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        required = {"question", "ground_truth", "category"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing)}. "
                f"Got: {df.columns.tolist()}"
            )
        return df.dropna(subset=["category"])

    train_df = _load(train_csv)
    val_df = _load(val_csv)

    train_pools: dict[str, list[tuple[str, str]]] = {}
    for cat, group in train_df.groupby("category", sort=False):
        train_pools[str(cat)] = [
            (row.question, row.ground_truth) for _, row in group.iterrows()
        ]

    val_data: list[tuple[str, str, str]] = [
        (row.question, row.ground_truth, str(row.category))
        for _, row in val_df.iterrows()
    ]

    # Warn (don't fail) on category mismatch — the loop round-robin-samples
    # from train_pools, so val categories not in train_pools will simply be
    # evaluated but never used for evolution.
    val_cats = {c for _, _, c in val_data}
    train_cats = set(train_pools)
    val_only = val_cats - train_cats
    train_only = train_cats - val_cats
    if val_only:
        print(f"[WARN] val has categories missing from train: {sorted(val_only)}")
    if train_only:
        print(f"[WARN] train has categories missing from val: {sorted(train_only)}")

    return train_pools, val_data


def _sync_subagent_prompt(data_root: Path | None, prompt_text: str) -> None:
    """Mirror the base agent's system prompt into `<data_root>/.claude/agents/general-purpose.md`.

    The bundled `claude` binary's default `general-purpose` Task subagent
    has a hard-coded one-line system prompt with no dataset context — so a
    subagent spawned via Task has no idea the corpus has 4 representations
    or which of them are lossy. This file overrides that default for any
    subagent dispatch from this run, giving the subagent the same priors
    the base agent runs under. `setting_sources=["user", "project"]` (set
    in `officeqa_agent.py`) is what makes the bundled binary pick this up.
    """
    if not data_root or not prompt_text:
        return
    agents_dir = data_root / ".claude" / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    target = agents_dir / "general-purpose.md"
    front_matter = (
        "---\n"
        "name: general-purpose\n"
        "description: General-purpose research subagent for the U.S. Treasury Bulletin corpus. Use for searching, extraction, or multi-step exploration tasks delegated by a parent agent.\n"
        "tools: [\"*\"]\n"
        "---\n\n"
    )
    body = (
        "You are a research subagent dispatched by a parent agent on tasks "
        "over the U.S. Treasury Bulletin corpus. Read the parent's task "
        "carefully, do exactly what's asked, and return a clear writeup of "
        "what you found.\n\n"
        "The parent agent operates under the following system prompt — "
        "you share its dataset, file-system, and closed-book constraints, "
        "and your output will be consumed by a parent that knows these "
        "rules:\n\n"
        f"{prompt_text.strip()}\n"
    )
    target.write_text(front_matter + body)


def _cleanup_scratch(data_root: Path | None) -> None:
    """Wipe `<data_root>/.cache/scratch/` — the agent's intermediate-files dir.

    The system prompt tells the agent to put scratch artifacts (debug images,
    extracted text, ad-hoc Python scripts) here. Cleanup is a harness
    responsibility so individual agent runs don't have to worry about it,
    and so leftover files from a crashed run can't pollute the next one.
    """
    if not data_root:
        return
    scratch = data_root / ".cache" / "scratch"
    if not scratch.exists():
        return
    import shutil
    n = sum(1 for _ in scratch.iterdir())
    shutil.rmtree(scratch, ignore_errors=True)
    if n:
        print(f"[SCRATCH] Cleared {n} item(s) from {scratch}")


def _fresh_reset(workspace_dir: Path, project_root: Path) -> None:
    """Wipe all evolutionary state before a fresh run.

    Targets both the evolution workspace (program branches, frontier tags,
    feedback history, checkpoint) and the project-local trace cache.

    When workspace_dir == project_root (no separation), both sets of cleanup
    apply to the same directory.
    """
    import shutil
    import subprocess

    print("[FRESH] Resetting all evolutionary state...")

    # ── Workspace-side: git branches, tags, and workspace-local state files ──
    if workspace_dir.exists():
        try:
            current = subprocess.check_output(
                ["git", "branch", "--show-current"], cwd=workspace_dir, text=True,
            ).strip()
            if current.startswith("program/"):
                branches = subprocess.check_output(
                    ["git", "branch", "--format=%(refname:short)"],
                    cwd=workspace_dir, text=True,
                ).strip().split("\n")
                non_program = [b for b in branches if b and not b.startswith("program/")]
                if non_program:
                    target = non_program[0]
                    subprocess.run(
                        ["git", "checkout", target], cwd=workspace_dir,
                        check=False, capture_output=True,
                    )
                    print(f"[FRESH] Switched workspace from {current} to {target}")
        except subprocess.CalledProcessError:
            pass

        try:
            branches = subprocess.check_output(
                ["git", "branch", "--list", "program/*", "--format=%(refname:short)"],
                cwd=workspace_dir, text=True,
            ).strip().split("\n")
            branches = [b for b in branches if b]
            if branches:
                subprocess.run(
                    ["git", "branch", "-D"] + branches, cwd=workspace_dir,
                    check=False, capture_output=True,
                )
                print(f"[FRESH] Deleted {len(branches)} program branches in workspace")
        except subprocess.CalledProcessError:
            pass

        try:
            tags = subprocess.check_output(
                ["git", "tag", "-l", "frontier/*"], cwd=workspace_dir, text=True,
            ).strip().split("\n")
            tags = [t for t in tags if t]
            if tags:
                subprocess.run(
                    ["git", "tag", "-d"] + tags, cwd=workspace_dir,
                    check=False, capture_output=True,
                )
                print(f"[FRESH] Deleted {len(tags)} frontier tags in workspace")
        except subprocess.CalledProcessError:
            pass

        for name in [".claude/feedback_history.md", ".claude/loop_checkpoint.json"]:
            p = workspace_dir / name
            if p.exists():
                p.unlink()
                print(f"[FRESH] Removed workspace/{name}")

        # Wipe skills so evolutionary skills from prior runs don't carry over.
        # Skills can live in TWO places depending on the run mode:
        #   1. workspace_dir/.claude/skills/  — when manager.cwd == workspace
        #   2. project_root/.claude/skills/   — when evolver agents have
        #      cwd=project_root (the typical skill_unified setup)
        # We wipe BOTH so a stale evolved skill in the project repo can't
        # warm-start what's supposed to be a cold base eval. Repo-shipped
        # skills with SKILL.md.disabled survive as inert reference docs.
        for label, skills_dir in [
            ("workspace", workspace_dir / ".claude" / "skills"),
            ("project", project_root / ".claude" / "skills"),
        ]:
            if not skills_dir.exists():
                continue
            # Preserve repo-shipped helper skills the evolver depends on.
            preserved = {"brainstorming", "skill-creator"}
            removed = 0
            for child in skills_dir.iterdir():
                if not child.is_dir():
                    continue
                if child.name in preserved:
                    continue
                if (child / "SKILL.md").exists():
                    # Remove the whole skill directory — covers evolution-created
                    # skills (treasury-json-navigation, etc.).
                    shutil.rmtree(child)
                    removed += 1
            if removed:
                print(f"[FRESH] Removed {removed} active skill(s) from {label}/.claude/skills/")

    # ── Cache wipe ──
    # SelfImprovingLoop uses `manager.cwd` (= workspace_dir under workspace
    # separation) as the root for `.cache/{traces.db, traces/, ...}`. Wipe
    # BOTH workspace/.cache AND project_root/.cache so stray state from
    # either layout is gone. No-op when a path doesn't exist.
    cache_roots = {workspace_dir, project_root}  # set → dedupes if they're equal
    for root in cache_roots:
        for name in [".cache/traces.db", ".cache/traces.db-wal", ".cache/traces.db-shm"]:
            p = root / name
            if p.exists():
                p.unlink()
                print(f"[FRESH] Removed {p}")
        # NOTE: `runs/` (the SDK trace cache) is intentionally NOT wiped here.
        # Cache files are content-addressed by hash of (model, system prompt,
        # tools, query, skill contents, ...) — if any input changes, the hash
        # differs and we miss; stale hits are impossible. Keeping the cache
        # across `--fresh` runs lets base eval and any unchanged-config calls
        # reuse prior traces. To force a full cache clear, use `--cache false`
        # or manually `rm -rf .cache/runs/`.
        for subdir in [
            "traces",
            "current_failures",  # legacy name — keep wiping for back-compat
            "current_iter_traces",
            "skill_snapshots",
        ]:
            d = root / ".cache" / subdir
            if d.exists():
                shutil.rmtree(d)
                print(f"[FRESH] Removed {d}")


async def main(settings: LoopSettings):
    # Set SDK based on CLI argument
    set_sdk(settings.sdk)

    project_root = Path.cwd()
    data_root = settings.data_root or str(project_root)

    # Resolve workspace: separate repo for evolutionary artifacts so program/*
    # branches, frontier/* tags, and .claude/feedback_history.md don't pollute
    # the EvoSkill source tree. Empty --workspace falls back to project_root.
    workspace_dir = (
        Path(settings.workspace).expanduser()
        if settings.workspace
        else project_root
    )
    if workspace_dir != project_root and not workspace_dir.exists():
        raise FileNotFoundError(
            f"Workspace {workspace_dir} does not exist. Create it first "
            f"(git init + empty commit), or pass --workspace= to use project root."
        )

    # Optional fresh reset: wipe all prior evolutionary state
    if settings.fresh:
        _fresh_reset(workspace_dir, project_root)

    # Harness-owned scratch cleanup. The agent's system prompt directs it to
    # write intermediate files to `<cwd>/.cache/scratch/`; lifecycle of that
    # dir is the harness's responsibility, not the agent's. Wipe at start to
    # clear any leftovers from a previous run that crashed before its
    # finally-block cleanup; wipe again at the end of this run.
    _cleanup_scratch(Path(data_root))

    # Mirror the base prompt into the project-level Task-subagent override so
    # subagents share the same dataset priors. Done before agents are built.
    from src.agent_profiles.officeqa_agent import PROMPT_FILE as _OFFICEQA_PROMPT_FILE
    _sync_subagent_prompt(Path(data_root), _OFFICEQA_PROMPT_FILE.read_text())

    # Two modes:
    #   1. Pre-split: user supplies --train_dataset AND --val_dataset → load as-is.
    #   2. Auto-split: user supplies --dataset → stratified_split on that one file.
    # Reject the ambiguous half-pre-split case up front.
    presplit = bool(settings.train_dataset) and bool(settings.val_dataset)
    if (settings.train_dataset or settings.val_dataset) and not presplit:
        raise ValueError(
            "--train_dataset and --val_dataset must both be provided together "
            "(or neither, to fall back to --dataset auto-split)."
        )

    if presplit:
        train_pools, val_data = load_presplit(settings.train_dataset, settings.val_dataset)
        dataset_label = f"{settings.train_dataset} + {settings.val_dataset} (pre-split)"
        split_info = "pre-split (no stratification applied)"
    else:
        data = pd.read_csv(settings.dataset)
        train_pools, val_data = stratified_split(
            data, train_ratio=settings.train_ratio, val_ratio=settings.val_ratio
        )
        dataset_label = settings.dataset
        split_info = (
            f"train={settings.train_ratio:.0%}, val={settings.val_ratio:.0%} "
            f"(remaining {1 - settings.train_ratio - settings.val_ratio:.0%} unused)"
        )

    categories = list(train_pools.keys())
    total_train = sum(len(pool) for pool in train_pools.values())
    print(f"Dataset: {dataset_label}")
    print(f"Categories ({len(categories)}): {', '.join(categories)}")
    print(
        f"Training pools: {', '.join(f'{cat}: {len(pool)}' for cat, pool in train_pools.items())}"
    )
    print(f"Total training samples: {total_train}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Split: {split_info}")

    # Base agent: cwd = EvoSkill (owns skills + scratch); data_root accessible via add_dirs.
    # Using data_root as cwd would pollute the data folder with agent-generated files.
    data_dirs = [data_root] if settings.data_root else None
    base_thinking_dict = (
        {"type": settings.base_thinking} if settings.base_thinking else None
    )
    evolver_thinking_dict = (
        {"type": settings.evolver_thinking} if settings.evolver_thinking else None
    )
    base_options = (
        make_officeqa_agent_options(
            model=settings.model,
            project_root=str(project_root),
            data_dirs=data_dirs,
            thinking=base_thinking_dict,
            effort=settings.base_effort,
        )
        if (settings.model or data_dirs or base_thinking_dict or settings.base_effort)
        else officeqa_agent_options
    )

    # Improvers (proposers/generators/evolver) default to Opus for quality.
    # Override with --evolver_model if you want them cheaper.
    improver_model = settings.evolver_model or "opus"

    # Build agents, each with a name for better OTel span labels
    agents = LoopAgents(
        solver=Agent(base_options, AgentResponse, name="solver"),
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
                make_skill_evolver_options(
                    model=improver_model,
                    data_dirs=[data_root] if data_root else None,
                    thinking=evolver_thinking_dict,
                    effort=settings.evolver_effort,
                ),
                SkillEvolverResponse,
                name="evolver",
            )
            if settings.mode == "skill_unified"
            else None
        ),
    )
    # Tell ProgramManager where the *real* skills tree lives so it can
    # snapshot/restore per-program. Skills are written by the evolver agents
    # whose cwd is `project_root` (the EvoSkill repo), NOT the workspace —
    # so without this hint, the workspace's empty stub gets snapshotted and
    # historical iterations become unreproducible.
    manager = ProgramManager(
        cwd=workspace_dir,
        project_skills_dir=project_root / ".claude" / "skills",
    )

    config = LoopConfig(
        max_iterations=settings.max_iterations,
        frontier_size=settings.frontier_size,
        no_improvement_limit=settings.no_improvement_limit,
        concurrency=settings.concurrency,
        evolution_mode=settings.mode,
        failure_sample_count=settings.failure_samples,
        categories_per_batch=settings.failure_samples,
        samples_per_category=settings.samples_per_category,
        proportional_sampling=settings.proportional_sampling,
        cache_enabled=settings.cache,
        mid_gate_enabled=settings.mid_gate,
        mid_gate_min_fixed=settings.mid_gate_min_fixed,
        mid_gate_max_regressions=settings.mid_gate_max_regressions,
        reset_feedback=settings.reset_feedback,
        continue_mode=settings.continue_loop,
        accuracy_threshold=settings.accuracy_threshold,
        cost_metric="total_cost_usd" if settings.accuracy_threshold else None,
        reviewer_enabled=settings.reviewer_enabled,
    )

    model_info = f", model={settings.model}" if settings.model else ""
    evolver_info = f", evolver_model={settings.evolver_model}" if settings.evolver_model else ""
    print(f"Running loop with evolution_mode={settings.mode}{model_info}{evolver_info}")
    from src.evaluation.officeqa_judge import score_officeqa
    from src.agent_profiles.officeqa_agent import PROMPT_FILE as OFFICEQA_PROMPT_FILE
    loop = SelfImprovingLoop(
        config, agents, manager, train_pools, val_data,
        scorer=score_officeqa,
        solver_prompt=OFFICEQA_PROMPT_FILE.read_text(),
        data_root=data_root,
    )
    try:
        result = await loop.run()
    finally:
        _cleanup_scratch(Path(data_root))

    print(f"Best: {result.best_program} ({result.best_score:.2%})")
    print(f"Frontier: {result.frontier}")
    print(f"Total cost: ${result.total_cost_usd:.4f}")


if __name__ == "__main__":
    settings = LoopSettings()
    asyncio.run(main(settings))
