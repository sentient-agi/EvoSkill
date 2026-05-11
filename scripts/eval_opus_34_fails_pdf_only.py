#!/usr/bin/env python3
"""Opus PDF-only baseline on the 34 fail UIDs (15 train + 19 val from strict-94).

Mirrors eval_sonnet_94_pdf_only.py setup but with model=opus, concurrency=8
(lower to share the API budget with the concurrent sonnet evo run), 480s
per-task timeout, and a dedicated workspace + Phoenix project to avoid
collision with the running evo.

Output: .dataset/officeqa_pro_opus_34_fails_pdf_only.csv
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

os.environ.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "64000")

from src.harness.agent import Agent
Agent.TIMEOUT_SECONDS = 480  # 8 min per task

from src.tracing import init_tracing
init_tracing("evoskill-opus-34-fails-pdf-only")

import pandas as pd

from src.harness import Agent as AgentCls, set_sdk, build_options
from src.cache import RunCache, CacheConfig
from src.evaluation.evaluate import evaluate_agent_parallel
from src.evaluation.officeqa_judge import score_officeqa
from src.harness.utils import eval_score_callback
from src.officeqa.workspace import build_pdf_only_workspace
from src.schemas import AgentResponse


PROJECT_ROOT = Path("/Users/dastin/dev/EvoSkill")
PDF_SOURCE = Path("/Users/dastin/dev/officeqa/data/treasury_bulletin_pdfs")
PDF_RO_CLONE = PROJECT_ROOT / ".cache" / "pdfs_ro"
# Dedicated workspace — keeps the concurrent evo run's workspace untouched.
WORKSPACE_DIR = PROJECT_ROOT / ".cache" / "eval_workspace_opus_34"

TRAIN_CSV = PROJECT_ROOT / ".dataset" / "officeqa_pro_pdf_fail_train_15.csv"
VAL_CSV = PROJECT_ROOT / ".dataset" / "officeqa_pro_pdf_fail_val_19.csv"
OUT_CSV = PROJECT_ROOT / ".dataset" / "officeqa_pro_opus_34_fails_pdf_only.csv"

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
        effort="medium",  # match sonnet baseline's effort for apples-to-apples
    )


async def main() -> None:
    set_sdk("claude")

    # Fresh workspace (no skills, PDF symlink only).
    import shutil
    if WORKSPACE_DIR.exists():
        shutil.rmtree(WORKSPACE_DIR)
    build_pdf_only_workspace(
        workspace_dir=WORKSPACE_DIR,
        pdf_source=PDF_SOURCE,
        pdf_ro_clone=PDF_RO_CLONE,
    )

    # Combine train + val into one 34-UID dataset
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    df = pd.concat([train_df, val_df], ignore_index=True)
    assert len(df) == 34, f"Expected 34 rows, got {len(df)}"
    items = [(str(r["question"]), str(r["ground_truth"])) for _, r in df.iterrows()]

    agent = AgentCls(_build_solver_options, AgentResponse, name="solver_opus_pdf_only")
    cache = RunCache(CacheConfig(
        cache_dir=PROJECT_ROOT / ".cache" / "runs",
        enabled=True,
        cwd=WORKSPACE_DIR,
    ))

    print(f"Running {len(items)} opus PDF-only eval on the 34 fails @ concurrency=8, timeout=480s ...")

    results = await evaluate_agent_parallel(
        agent=agent, items=items,
        max_concurrent=8, tag_prefix="opus34",
        cache=cache,
    )

    # Score with LLM judge
    rows = []
    total_cost = 0.0
    for uid, gt_status, (_, gt), res in zip(df["uid"], df["gt_status"], items, results):
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
            "uid": uid,
            "gt_status": gt_status,
            "ground_truth": gt,
            "opus_pred": predicted[:200],
            "opus_judged": round(score, 4),
            "cost_usd": round(cost, 4),
            "no_output": no_output,
        })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    pass_n = (out["opus_judged"] >= 0.8).sum()
    print(f"\n[OK] wrote {OUT_CSV} ({len(out)} rows)")
    print(f"Pass (judged >= 0.8): {pass_n} / {len(out)} = {100*pass_n/len(out):.1f}%")
    print(f"Avg score: {out['opus_judged'].mean():.4f}")
    print(f"no_output count: {out['no_output'].sum()}")
    print(f"Total cost: ${total_cost:.4f}")
    print()
    print("By split:")
    train_uids = set(train_df["uid"])
    for label, sub in [("train_15", out[out["uid"].isin(train_uids)]),
                       ("val_19", out[~out["uid"].isin(train_uids)])]:
        p = (sub["opus_judged"] >= 0.8).sum()
        print(f"  {label}: pass {p}/{len(sub)} = {100*p/len(sub):.1f}%, avg {sub['opus_judged'].mean():.3f}")


if __name__ == "__main__":
    asyncio.run(main())
