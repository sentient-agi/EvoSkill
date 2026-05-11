#!/usr/bin/env python3
"""Evaluate the winning evolved skill (treasury-precision-v3 from iter-skill-4)
on the 68 holdout questions that were NOT in train/val during evolution.

This is the uncontaminated generalization test: the evolver never saw these
questions, their traces, or their GTs. If the skill transfers, pass rate on
the 68 should be at least as high as the opus baseline (and ideally higher).

Runs opus@medium with the skill installed, same PDF-only setup as the evo.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "64000")

from src.harness.agent import Agent
Agent.TIMEOUT_SECONDS = 720  # match evo experiment

from src.tracing import init_tracing
init_tracing("evoskill-opus-holdout-68")

import pandas as pd

from src.harness import set_sdk, build_options
from src.cache import RunCache, CacheConfig
from src.evaluation.evaluate import evaluate_agent_parallel
from src.evaluation.officeqa_judge import score_officeqa
from src.harness.utils import eval_score_callback
from src.officeqa.workspace import build_pdf_only_workspace
from src.schemas import AgentResponse


PROJECT_ROOT = Path("/Users/dastin/dev/EvoSkill")
PDF_SOURCE = Path("/Users/dastin/dev/officeqa/data/treasury_bulletin_pdfs")
PDF_RO_CLONE = PROJECT_ROOT / ".cache" / "pdfs_ro"
WORKSPACE_DIR = PROJECT_ROOT / ".cache" / "eval_workspace_holdout_68"

STRICT_CSV = PROJECT_ROOT / ".dataset" / "officeqa_pro_94_strict.csv"
TRAIN_CSV = PROJECT_ROOT / ".dataset" / "officeqa_pro_opus_fail_train_10.csv"
VAL_CSV = PROJECT_ROOT / ".dataset" / "officeqa_pro_opus_fail_val_16.csv"
OUT_CSV = PROJECT_ROOT / ".dataset" / "officeqa_pro_opus_holdout_68_with_skill.csv"

SOLVER_PROMPT_FILE = (
    PROJECT_ROOT / "src" / "agent_profiles" / "officeqa_agent" / "prompt_pdf_only.md"
)
# The winning skill from the evo experiment (iter-skill-4)
EVO_WORKSPACE = PROJECT_ROOT / ".cache" / "evo_workspace_opus_pdf_only"

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


async def main() -> None:
    set_sdk("claude")

    # Build workspace
    if WORKSPACE_DIR.exists():
        shutil.rmtree(WORKSPACE_DIR)
    build_pdf_only_workspace(
        workspace_dir=WORKSPACE_DIR,
        pdf_source=PDF_SOURCE,
        pdf_ro_clone=PDF_RO_CLONE,
    )

    # Copy the winning skill from the evo workspace into this eval workspace.
    # The evo workspace's current branch (iter-skill-4) has the skill(s).
    evo_skills = EVO_WORKSPACE / ".claude" / "skills"
    eval_skills = WORKSPACE_DIR / ".claude" / "skills"
    if evo_skills.exists():
        for skill_dir in evo_skills.iterdir():
            if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                dst = eval_skills / skill_dir.name
                shutil.copytree(skill_dir, dst)
                print(f"Installed skill: {skill_dir.name} ({(skill_dir / 'SKILL.md').stat().st_size} bytes)")
    else:
        # Fall back: checkout iter-skill-4 branch in evo workspace and copy
        subprocess.run(["git", "checkout", "program/iter-skill-4"],
                       cwd=EVO_WORKSPACE, check=True, capture_output=True)
        evo_skills = EVO_WORKSPACE / ".claude" / "skills"
        for skill_dir in evo_skills.iterdir():
            if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                dst = eval_skills / skill_dir.name
                shutil.copytree(skill_dir, dst)
                print(f"Installed skill: {skill_dir.name}")

    # Build holdout set: strict-94 minus train minus val
    strict = pd.read_csv(STRICT_CSV)
    train_uids = set(pd.read_csv(TRAIN_CSV)["uid"])
    val_uids = set(pd.read_csv(VAL_CSV)["uid"])
    holdout = strict[~strict["uid"].isin(train_uids | val_uids)].copy()
    assert len(holdout) == 68, f"Expected 68 holdout, got {len(holdout)}"

    items = [(str(r["question"]), str(r["ground_truth"])) for _, r in holdout.iterrows()]
    print(f"Running {len(items)} opus holdout eval with winning skill @ concurrency=16, timeout=720s ...")

    agent = Agent(_build_solver_options, AgentResponse, name="solver")
    cache = RunCache(CacheConfig(
        cache_dir=PROJECT_ROOT / ".cache" / "runs",
        enabled=True,
        cwd=WORKSPACE_DIR,
    ))

    score_token = eval_score_callback.set(
        lambda q, pred, gt: float(score_officeqa(q, gt, pred))
    )
    try:
        results = await evaluate_agent_parallel(
            agent=agent, items=items,
            max_concurrent=32, tag_prefix="holdout68",
            cache=cache,
        )
    finally:
        eval_score_callback.reset(score_token)

    # Score + write CSV
    rows = []
    total_cost = 0.0
    for uid, gt_status, (_, gt), res in zip(
        holdout["uid"], holdout["gt_status"], items, results,
    ):
        if res.trace is None or res.trace.output is None:
            score = 0.0
            predicted = ""
            cost = 0.0
            no_output = True
        else:
            predicted = str(res.trace.output.final_answer)
            score = score_officeqa(res.question, gt, predicted)
            cost = float(res.trace.total_cost_usd or 0.0)
            no_output = False
        total_cost += cost
        rows.append({
            "uid": uid, "gt_status": gt_status, "ground_truth": gt,
            "opus_pred": predicted[:200], "opus_judged": round(score, 4),
            "cost_usd": round(cost, 4), "no_output": no_output,
        })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)

    pass_n = (out["opus_judged"] >= 0.8).sum()
    no_out = out["no_output"].sum()
    print(f"\n[OK] wrote {OUT_CSV} ({len(out)} rows)")
    print(f"Pass (judged >= 0.8): {pass_n} / {len(out)} = {100*pass_n/len(out):.1f}%")
    print(f"Avg score: {out['opus_judged'].mean():.4f}")
    print(f"no_output: {no_out}")
    print(f"Total cost: ${total_cost:.4f}")

    # Compare to baseline (opus without skill on these same 68)
    # The strict-94 sonnet baseline has these UIDs; opus baseline only has the 34 fails.
    # For the 68 holdout, opus baseline was the "pass" set — all scored ≥0.97.
    print(f"\nNote: these 68 are the 'easy' questions opus already passed at baseline.")
    print(f"Regression check: any score < 0.8 indicates the skill HURT on that UID.")
    regressed = out[out["opus_judged"] < 0.8]
    if len(regressed) > 0:
        print(f"\n⚠ {len(regressed)} REGRESSIONS:")
        for _, r in regressed.iterrows():
            print(f"  {r['uid']}: score={r['opus_judged']}, pred={str(r['opus_pred'])[:60]}")
    else:
        print(f"\n✓ Zero regressions — skill transferred cleanly on all 68 holdout items.")


if __name__ == "__main__":
    asyncio.run(main())
