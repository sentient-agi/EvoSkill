#!/usr/bin/env python3
"""PDF-only evolution loop for the 34 failed UIDs from strict-94.

Configuration locked in 2026-05-10 evo experiment:
  - Solver: sonnet 4.6, adaptive thinking, effort=medium
  - Evolver: opus 4.6, adaptive thinking, effort=high (CC harness doesn't support opus 4.7)
  - Per-task timeout: 8 min (Agent.TIMEOUT_SECONDS=480, outer asyncio=600)
  - Concurrency: 20
  - max_iterations: 3
  - 10 train samples per iter (failure_samples=1, samples_per_category=10)
  - mid-gate on (min_fixed=1, max_regressions=0)
  - reviewer off
  - Train: 15 UIDs (.dataset/officeqa_pro_pdf_fail_train_15.csv)
  - Val:   19 UIDs (.dataset/officeqa_pro_pdf_fail_val_19.csv)
  - Workspace: .cache/evo_workspace_pdf_only/ (fresh; PDF symlink + .claude/skills + .cache/scratch)

Built as a standalone runner (separate from run_loop.py) to avoid mixing the
PDF-only single-modality config with the standard multi-modality loop. If the
config proves stable, promote to a --pdf_only flag in run_loop.py.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "64000")

# Override Agent timeout BEFORE any Agent is constructed. The outer
# asyncio.timeout in evaluate.py derives from this + 120s buffer.
from src.harness.agent import Agent
Agent.TIMEOUT_SECONDS = 480  # 8 min per task

from src.tracing import init_tracing
init_tracing("evoskill-evo-pdf-only-34")

import pandas as pd

from src.harness import set_sdk, build_options
from src.cache import RunCache, CacheConfig
from src.schemas import (
    AgentResponse, SkillProposerResponse, PromptProposerResponse,
    ToolGeneratorResponse, PromptGeneratorResponse, SkillEvolverResponse,
)
from src.loop import SelfImprovingLoop, LoopConfig, LoopAgents
from src.registry import ProgramManager
from src.agent_profiles.skill_proposer import make_skill_proposer_options
from src.agent_profiles.prompt_proposer import make_prompt_proposer_options
from src.agent_profiles.skill_generator import make_skill_generator_options
from src.agent_profiles.prompt_generator import make_prompt_generator_options
from src.agent_profiles.skill_evolver import make_skill_evolver_options
from src.officeqa.workspace import build_pdf_only_workspace
from src.evaluation.officeqa_judge import score_officeqa


PROJECT_ROOT = Path("/Users/dastin/dev/EvoSkill")
PDF_SOURCE = Path("/Users/dastin/dev/officeqa/data/treasury_bulletin_pdfs")
PDF_RO_CLONE = PROJECT_ROOT / ".cache" / "pdfs_ro"
# Fresh dedicated workspace for the evo run — keeps the eval workspace
# untouched and gives the evolver a clean .claude/skills/ slate.
WORKSPACE_DIR = PROJECT_ROOT / ".cache" / "evo_workspace_pdf_only"

TRAIN_CSV = PROJECT_ROOT / ".dataset" / "officeqa_pro_pdf_fail_train_15.csv"
VAL_CSV = PROJECT_ROOT / ".dataset" / "officeqa_pro_pdf_fail_val_19.csv"

SOLVER_PROMPT_FILE = (
    PROJECT_ROOT / "src" / "agent_profiles" / "officeqa_agent" / "prompt_pdf_only.md"
)

SOLVER_TOOLS = [
    "Read", "Write", "Bash", "Glob", "Grep", "Edit",
    "TodoWrite", "BashOutput", "Skill",
]


def _build_solver_options():
    prompt_text = SOLVER_PROMPT_FILE.read_text().strip()
    return build_options(
        system=prompt_text,
        schema=AgentResponse.model_json_schema(),
        tools=SOLVER_TOOLS,
        project_root=str(WORKSPACE_DIR),
        model="sonnet",
        setting_sources=["user", "project"],
        permission_mode="acceptEdits",
        max_buffer_size=10 * 1024 * 1024,
        disallowed_tools=["Task", "WebFetch", "WebSearch"],
        thinking={"type": "adaptive"},
        effort="medium",
    )


def _load_presplit(train_csv: Path, val_csv: Path):
    """Load pre-split train + val into (train_pools, val_data) shapes the loop expects."""
    train_df = pd.read_csv(train_csv).dropna(subset=["category"])
    val_df = pd.read_csv(val_csv).dropna(subset=["category"])
    train_pools: dict[str, list[tuple[str, str]]] = {}
    for cat, sub in train_df.groupby("category"):
        train_pools[str(cat)] = [
            (str(r["question"]), str(r["ground_truth"])) for _, r in sub.iterrows()
        ]
    val_data: list[tuple[str, str, str]] = [
        (str(r["question"]), str(r["ground_truth"]), str(r["category"]))
        for _, r in val_df.iterrows()
    ]
    return train_pools, val_data


def _ensure_workspace_git(workspace_dir: Path) -> None:
    """Run_loop's machinery assumes the workspace is a git repo. Init if needed.

    Also commits a tracked `.claude/.keep` so the `.claude/` directory
    survives branch switches. Without this, switching between program
    branches whose `.claude/skills/` content differs leaves `.claude/`
    empty (untracked), and subsequent `append_feedback` calls crash with
    FileNotFoundError when writing `.claude/feedback_history.md`.
    """
    if not (workspace_dir / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=workspace_dir, check=True)
        # Pin `.claude/` to the tree so branch switches preserve it.
        claude_dir = workspace_dir / ".claude"
        claude_dir.mkdir(parents=True, exist_ok=True)
        keep = claude_dir / ".keep"
        keep.touch(exist_ok=True)
        subprocess.run(["git", "add", ".claude/.keep"], cwd=workspace_dir, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "evo workspace init"],
            cwd=workspace_dir, check=True,
        )


def _migrate_strict94_cache_into(tree_dir: Path) -> int:
    """Copy strict-94 cache entries from eda9a973abf4 into the run's
    computed tree_hash dir so iter-0 baseline val + iter-1 train sampling
    hit cache instantly. Idempotent — skips already-present entries.

    Returns count of newly-copied files.
    """
    src = PROJECT_ROOT / ".cache" / "runs" / "eda9a973abf4"
    if not src.exists():
        return 0
    tree_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    copied = 0
    for f in src.glob("*.json"):
        dst = tree_dir / f.name
        if dst.exists():
            continue
        shutil.copy2(f, dst)
        copied += 1
    return copied


def _compute_run_tree_hash() -> str:
    """Recompute the RunCache tree_hash AFTER the workspace is rebuilt so we
    know which dir to migrate strict-94 entries into. Uses the same config
    SelfImprovingLoop will use internally."""
    from src.cache import RunCache, CacheConfig
    skills_dir = WORKSPACE_DIR / ".claude" / "skills"
    cfg = CacheConfig(
        cache_dir=PROJECT_ROOT / ".cache" / "runs",
        enabled=True,
        cwd=WORKSPACE_DIR,
        live_skills_dir=skills_dir,
        project_source_root=WORKSPACE_DIR,
    )
    cache = RunCache(cfg)
    return cache._get_tree_hash(system_prompt=SOLVER_PROMPT_FILE.read_text().strip())[:12]


async def main() -> None:
    set_sdk("claude")

    # Nuke any previous workspace state. Stale "Update score" commits from
    # earlier runs mutate program.yaml → root git tree SHA shifts → cache
    # tree_hash drifts → iter-0 baseline misses cache. A fresh workspace
    # gives a deterministic tree_hash that matches what we migrate into.
    import shutil
    if WORKSPACE_DIR.exists():
        shutil.rmtree(WORKSPACE_DIR)

    # Rebuild: PDFs symlink (read-only), .claude/skills/, .cache/scratch/.
    # The PDF clone refresh inside this call ensures the agent sees the
    # latest source PDFs even after a user edit.
    build_pdf_only_workspace(
        workspace_dir=WORKSPACE_DIR,
        pdf_source=PDF_SOURCE,
        pdf_ro_clone=PDF_RO_CLONE,
    )
    _ensure_workspace_git(WORKSPACE_DIR)

    # Migrate strict-94 cache into the freshly-computed tree_hash so iter-0
    # baseline val (19 UIDs) + iter-1 train sampling (10 of 15) hit cache.
    new_tree = _compute_run_tree_hash()
    new_tree_dir = PROJECT_ROOT / ".cache" / "runs" / new_tree
    migrated = _migrate_strict94_cache_into(new_tree_dir)
    print(f"[CACHE] tree_hash={new_tree}, migrated {migrated} entries from strict-94")

    # Load train + val
    train_pools, val_data = _load_presplit(TRAIN_CSV, VAL_CSV)
    total_train = sum(len(p) for p in train_pools.values())
    print(f"Train pools: {[(c, len(p)) for c, p in train_pools.items()]} (total={total_train})")
    print(f"Val:         {len(val_data)} UIDs")

    # Build agents
    solver_factory = _build_solver_options
    solver = Agent(solver_factory, AgentResponse, name="solver_pdf_only")

    evolver_kwargs = dict(
        model="opus",            # CC harness uses opus 4.6
        project_root=str(WORKSPACE_DIR),  # cwd = workspace (writes .claude/skills there)
        thinking={"type": "adaptive"},
        effort="max",  # max thinking — opus burns turns generating skills
    )
    proposer_kwargs = dict(model="opus")

    agents = LoopAgents(
        solver=solver,
        skill_proposer=Agent(
            make_skill_proposer_options(**proposer_kwargs),
            SkillProposerResponse, name="skill_proposer",
        ),
        prompt_proposer=Agent(
            make_prompt_proposer_options(**proposer_kwargs),
            PromptProposerResponse, name="prompt_proposer",
        ),
        skill_generator=Agent(
            make_skill_generator_options(**proposer_kwargs),
            ToolGeneratorResponse, name="skill_generator",
        ),
        prompt_generator=Agent(
            make_prompt_generator_options(**proposer_kwargs),
            PromptGeneratorResponse, name="prompt_generator",
        ),
        skill_evolver=Agent(
            make_skill_evolver_options(**evolver_kwargs),
            SkillEvolverResponse,
            name="evolver",
            # Evolver needs a longer budget than the solver. Opus@max with
            # deep thinking + 9 failure traces in context regularly burns
            # 100+ turns. Solver class default (Agent.TIMEOUT_SECONDS=480)
            # is too tight for the evolver — override at construction.
            timeout_seconds=1000,
        ),
    )

    # ProgramManager points at the workspace for both cwd AND skills dir —
    # PDF-only setup has them colocated (evolver writes to workspace's
    # .claude/skills/, solver reads from the same path via its cwd).
    manager = ProgramManager(
        cwd=WORKSPACE_DIR,
        project_skills_dir=WORKSPACE_DIR / ".claude" / "skills",
    )

    config = LoopConfig(
        max_iterations=3,
        concurrency=20,
        evolution_mode="skill_unified",
        failure_sample_count=1,
        categories_per_batch=1,
        samples_per_category=10,
        cache_enabled=True,
        mid_gate_enabled=True,
        # Lenient mean-score policy: tolerate a single noise-driven regression
        # as long as net average across re-evaluated train samples stays flat
        # or improves. LLM scores on chart-read / multi-component answers are
        # noisy enough that a strict counts policy (max_regressions=0) blocks
        # iters with a clean net win.
        mid_gate_policy="mean",
        reviewer_enabled=False,
    )

    print(f"Config: iter=3, concurrency=20, 10 train/iter (1 cat × 10), "
          f"timeout=480s, mid-gate=mean (post_mean >= pre_mean), reviewer off")

    solver_prompt_text = SOLVER_PROMPT_FILE.read_text()
    loop = SelfImprovingLoop(
        config, agents, manager, train_pools, val_data,
        scorer=score_officeqa,
        solver_prompt=solver_prompt_text,
        data_root=str(WORKSPACE_DIR),
    )
    result = await loop.run()

    print(f"\n=== Evo PDF-only run complete ===")
    print(f"Best program: {result.best_program} (score={result.best_score:.2%})")
    print(f"Frontier: {result.frontier}")
    print(f"Total cost: ${result.total_cost_usd:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
