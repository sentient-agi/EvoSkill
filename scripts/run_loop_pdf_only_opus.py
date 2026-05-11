#!/usr/bin/env python3
"""Opus-solver evolution loop on opus's own failure set.

Configuration:
  - Solver: opus 4.6, adaptive thinking, effort=medium (CC harness opus 4.6)
  - Evolver: opus 4.6, adaptive thinking, effort=max
  - Per-task timeout: 12 min (Agent.TIMEOUT_SECONDS=720)
  - Evolver timeout: 1000s (instance override)
  - Concurrency: 8 (opus is slower + more expensive than sonnet, lower budget pressure)
  - max_iterations: 4
  - 10 train per iter (1 cat x 10)
  - Mid-gate=mean (lenient)
  - Reviewer off
  - Train: 10 UIDs (.dataset/officeqa_pro_opus_fail_train_10.csv)
  - Val:   16 UIDs (.dataset/officeqa_pro_opus_fail_val_16.csv)
  - Workspace: .cache/evo_workspace_opus_pdf_only/ (fresh each run)
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import shutil
from pathlib import Path

os.environ.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "64000")

from src.harness.agent import Agent
Agent.TIMEOUT_SECONDS = 720  # 12 min per task (solver)

from src.tracing import init_tracing
init_tracing("evoskill-evo-opus-pdf-only")

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
WORKSPACE_DIR = PROJECT_ROOT / ".cache" / "evo_workspace_opus_pdf_only"

TRAIN_CSV = PROJECT_ROOT / ".dataset" / "officeqa_pro_opus_fail_train_10.csv"
VAL_CSV = PROJECT_ROOT / ".dataset" / "officeqa_pro_opus_fail_val_16.csv"

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
        model="opus",
        setting_sources=["user", "project"],
        permission_mode="acceptEdits",
        max_buffer_size=10 * 1024 * 1024,
        disallowed_tools=["Task", "WebFetch", "WebSearch"],
        thinking={"type": "adaptive"},
        effort="medium",
    )


def _load_presplit(train_csv: Path, val_csv: Path):
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
    if not (workspace_dir / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=workspace_dir, check=True)
        claude_dir = workspace_dir / ".claude"
        claude_dir.mkdir(parents=True, exist_ok=True)
        (claude_dir / ".keep").touch(exist_ok=True)
        subprocess.run(["git", "add", ".claude/.keep"], cwd=workspace_dir, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "evo workspace init"],
            cwd=workspace_dir, check=True,
        )


def _migrate_opus34_cache_into(tree_dir: Path) -> int:
    """Copy the opus-34 baseline cache entries into the run's tree_hash dir
    so iter-0 val baseline (16 UIDs, subset of the 34) hits cache instantly.

    The opus-34 entries are scattered across multiple tree_hashes from the
    eval workspace. Walk all of them and pull entries whose cache_key.model
    matches opus.
    """
    import json
    src_root = PROJECT_ROOT / ".cache" / "runs"
    tree_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for f in src_root.rglob("*.json"):
        if f.parent.name == tree_dir.name:
            continue  # don't re-copy from ourselves
        try:
            d = json.load(open(f))
        except Exception:
            continue
        ck = d.get("cache_key", {})
        if "opus" not in ck.get("model", "").lower():
            continue
        # Same question_hash → same filename
        dst = tree_dir / f.name
        if dst.exists():
            continue
        shutil.copy2(f, dst)
        copied += 1
    return copied


def _compute_run_tree_hash() -> str:
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

    if WORKSPACE_DIR.exists():
        shutil.rmtree(WORKSPACE_DIR)

    build_pdf_only_workspace(
        workspace_dir=WORKSPACE_DIR,
        pdf_source=PDF_SOURCE,
        pdf_ro_clone=PDF_RO_CLONE,
    )
    _ensure_workspace_git(WORKSPACE_DIR)

    new_tree = _compute_run_tree_hash()
    new_tree_dir = PROJECT_ROOT / ".cache" / "runs" / new_tree
    migrated = _migrate_opus34_cache_into(new_tree_dir)
    print(f"[CACHE] tree_hash={new_tree}, migrated {migrated} opus entries from prior runs")

    train_pools, val_data = _load_presplit(TRAIN_CSV, VAL_CSV)
    total_train = sum(len(p) for p in train_pools.values())
    print(f"Train pools: {[(c, len(p)) for c, p in train_pools.items()]} (total={total_train})")
    print(f"Val:         {len(val_data)} UIDs")

    solver = Agent(_build_solver_options, AgentResponse, name="solver")

    evolver_kwargs = dict(
        model="opus",
        project_root=str(WORKSPACE_DIR),
        thinking={"type": "adaptive"},
        effort="max",
    )
    proposer_kwargs = dict(model="opus")

    agents = LoopAgents(
        solver=solver,
        skill_proposer=Agent(make_skill_proposer_options(**proposer_kwargs),
                             SkillProposerResponse, name="skill_proposer"),
        prompt_proposer=Agent(make_prompt_proposer_options(**proposer_kwargs),
                              PromptProposerResponse, name="prompt_proposer"),
        skill_generator=Agent(make_skill_generator_options(**proposer_kwargs),
                              ToolGeneratorResponse, name="skill_generator"),
        prompt_generator=Agent(make_prompt_generator_options(**proposer_kwargs),
                               PromptGeneratorResponse, name="prompt_generator"),
        skill_evolver=Agent(
            make_skill_evolver_options(**evolver_kwargs),
            SkillEvolverResponse,
            name="evolver",
            timeout_seconds=1000,
        ),
    )

    manager = ProgramManager(
        cwd=WORKSPACE_DIR,
        project_skills_dir=WORKSPACE_DIR / ".claude" / "skills",
    )

    config = LoopConfig(
        max_iterations=4,
        concurrency=16,
        evolution_mode="skill_unified",
        failure_sample_count=1,
        categories_per_batch=1,
        # 10 train per iter — equals the full train pool, so each iter
        # sees every training sample (no stochastic sampling).
        samples_per_category=10,
        cache_enabled=True,
        mid_gate_enabled=True,
        mid_gate_policy="mean",
        reviewer_enabled=False,
    )

    print(f"Config: iter=4, concurrency=16, 10 train/iter (== full pool, no sub-sampling), "
          f"solver timeout={Agent.TIMEOUT_SECONDS}s, evolver timeout=1000s effort=max, "
          f"mid-gate=mean, reviewer off, cache on")

    solver_prompt_text = SOLVER_PROMPT_FILE.read_text()
    loop = SelfImprovingLoop(
        config, agents, manager, train_pools, val_data,
        scorer=score_officeqa,
        solver_prompt=solver_prompt_text,
        data_root=str(WORKSPACE_DIR),
    )
    result = await loop.run()

    print(f"\n=== Opus evo PDF-only run complete ===")
    print(f"Best program: {result.best_program} (score={result.best_score:.2%})")
    print(f"Frontier: {result.frontier}")
    print(f"Total cost: ${result.total_cost_usd:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
