#!/usr/bin/env python3
"""PDF-only closed-book eval for Sonnet on strict-94 (officeqa_pro_94_strict.csv).

Uses a shared workspace pattern (see src/officeqa/workspace.py): both
solver and evolver — when this dataset is later wired into evolution —
share the same cwd, see PDFs at the same relative path
(`treasury_bulletin_pdfs/`), and write to the same `.cache/scratch/` and
`.claude/skills/`. The PDF symlink resolves to a read-only APFS clone of
the source corpus, refreshed on every run startup so the agent always
sees the latest user edits while never being able to write to the corpus
itself.

Sonnet 4.6, adaptive thinking, effort="medium", concurrency=20.
Cache key includes `effort` (per RunCache change in this same commit).

Output: .dataset/officeqa_pro_94_sonnet_pdf_only_baseline.csv
"""
from __future__ import annotations

import asyncio
import os
import sys

os.environ.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "64000")

from src.tracing import init_tracing
init_tracing("evoskill-sonnet-94-pdf-only")

from pathlib import Path

import pandas as pd

from src.harness import Agent, set_sdk, build_options
from src.cache import RunCache, CacheConfig
from src.evaluation.evaluate import evaluate_agent_parallel
from src.evaluation.officeqa_judge import score_officeqa
from src.harness.utils import eval_score_callback
from src.officeqa.workspace import build_pdf_only_workspace
from src.schemas import AgentResponse


PROJECT_ROOT = Path("/Users/dastin/dev/EvoSkill")
PDF_SOURCE = Path("/Users/dastin/dev/officeqa/data/treasury_bulletin_pdfs")
PDF_RO_CLONE = PROJECT_ROOT / ".cache" / "pdfs_ro"
WORKSPACE_DIR = PROJECT_ROOT / ".cache" / "eval_workspace_pdf_only_94"

DATASET_CSV = PROJECT_ROOT / ".dataset" / "officeqa_pro_94_strict.csv"
OUT_CSV = PROJECT_ROOT / ".dataset" / "officeqa_pro_94_sonnet_pdf_only_baseline.csv"

PROMPT_FILE = PROJECT_ROOT / "src" / "agent_profiles" / "officeqa_agent" / "prompt_pdf_only.md"

# Tools list mirrors OFFICEQA_AGENT_TOOLS minus web access.
TOOLS_NO_WEB = [
    "Read", "Write", "Bash", "Glob", "Grep", "Edit",
    "TodoWrite", "BashOutput", "Skill",
]


def build_pdf_only_options():
    prompt_text = PROMPT_FILE.read_text().strip()
    # cwd = the shared workspace (writable). Inside it: treasury_bulletin_pdfs/
    # symlink to a read-only clone of the corpus, .claude/skills/ for skill
    # discovery, .cache/scratch/ for intermediate files. No add_dirs needed —
    # everything the agent needs is rooted in the workspace.
    return build_options(
        system=prompt_text,
        schema=AgentResponse.model_json_schema(),
        tools=TOOLS_NO_WEB,
        project_root=str(WORKSPACE_DIR),
        model="sonnet",
        setting_sources=["user", "project"],
        permission_mode="acceptEdits",
        max_buffer_size=10 * 1024 * 1024,
        disallowed_tools=["Task", "WebFetch", "WebSearch"],
        thinking={"type": "adaptive"},
        effort="medium",
    )


async def main() -> None:
    set_sdk("claude")

    # Build (or refresh) the workspace BEFORE any agent runs. This mirrors
    # the source PDFs into the read-only clone (rsync --delete) and re-
    # applies chmod 555, so the agent always sees the user's latest edits.
    build_pdf_only_workspace(
        workspace_dir=WORKSPACE_DIR,
        pdf_source=PDF_SOURCE,
        pdf_ro_clone=PDF_RO_CLONE,
    )

    df_in = pd.read_csv(DATASET_CSV)
    assert len(df_in) == 94, f"Expected 94 rows, got {len(df_in)}"

    options_factory = build_pdf_only_options
    agent = Agent(options_factory, AgentResponse, name="solver_pdf_only_sonnet")

    # Cache cwd MUST match the agent's cwd so the tree_hash reflects the
    # skills the agent could actually load (`<cwd>/.claude/skills/`), not
    # whatever happens to sit under PROJECT_ROOT/.claude/skills/ — the
    # agent's filesystem reach is scoped to WORKSPACE_DIR + add_dirs, so
    # those PROJECT_ROOT skills are invisible to it and shouldn't enter
    # the cache key. Cache entries written with mismatched cwd become
    # unreachable by any future correctly-configured caller.
    cache = RunCache(CacheConfig(
        cache_dir=PROJECT_ROOT / ".cache" / "runs",
        enabled=True,
        cwd=WORKSPACE_DIR,
    ))

    items = [(str(row["question"]), str(row["ground_truth"])) for _, row in df_in.iterrows()]
    print(f"Running {len(items)} sonnet PDF-only eval @ concurrency=20 ...")
    print(f"  Auth: {'ANTHROPIC_API_KEY' if 'ANTHROPIC_API_KEY' in os.environ else 'CC subscription (no API key in env)'}")

    # No live score callback during the run — the judge needs ANTHROPIC_API_KEY,
    # but if we set that env var before the agent runs, the agent SDK's
    # subprocess inherits it and bills against the API instead of the CC
    # subscription. Defer all judging until after agent runs complete.
    results = await evaluate_agent_parallel(
        agent=agent, items=items,
        max_concurrent=20, tag_prefix="sonnet_pdf94",
        cache=cache,
    )

    # Now activate the API key for the judge. The agent runs are done, so the
    # subprocess they spawned has already returned; setting ANTHROPIC_API_KEY
    # in the parent Python env affects only the upcoming judge calls.
    judge_key = os.environ.pop("ANTHROPIC_API_KEY_FOR_JUDGE", None)
    if judge_key:
        os.environ["ANTHROPIC_API_KEY"] = judge_key
        print("[judge] activated ANTHROPIC_API_KEY for the judge phase")
    elif "ANTHROPIC_API_KEY" not in os.environ:
        print("[judge] WARNING: no ANTHROPIC_API_KEY available; judge will fail")

    rows = []
    total_cost = 0.0
    for (uid, gt_status), (_, gt), res in zip(
        zip(df_in["uid"], df_in["gt_status"]), items, results,
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
            "uid": uid,
            "gt_status": gt_status,
            "ground_truth": gt,
            "sonnet_pred": predicted[:200],
            "sonnet_judged": round(score, 4),
            "cost_usd": round(cost, 4),
            "no_output": no_output,
        })

    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    pass_n = (out["sonnet_judged"] >= 0.8).sum()
    print(f"\n[OK] wrote {OUT_CSV} ({len(out)} rows)")
    print(f"Pass (judged >= 0.8): {pass_n} / {len(out)} = {100*pass_n/len(out):.1f}%")
    print(f"Avg score: {out['sonnet_judged'].mean():.4f}")
    print(f"Total cost: ${total_cost:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
