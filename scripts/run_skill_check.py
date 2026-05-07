#!/usr/bin/env python3
"""Eval the currently-installed `.claude/skills/` against a val CSV.

Use this to verify a skill that an evolution loop produced — without
running another evolution. Loads `officeqa_agent` (so the agent sees the
real system prompt + Task disabled) with the current state of
`<project>/.claude/skills/`. The bundled binary auto-picks-up any skill
files there. Scores each prediction with the OfficeQA LLM judge.
"""

# Phoenix tracing must register BEFORE any LLM SDK import so the bundled
# Anthropic client gets auto-instrumented.
from src.tracing import init_tracing
init_tracing("evoskill-skillcheck")

import asyncio
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.harness import Agent, set_sdk
from src.agent_profiles.officeqa_agent import (
    make_officeqa_agent_options,
    officeqa_agent_options,
)
from src.schemas import AgentResponse
from src.evaluation import evaluate_agent_parallel
from src.evaluation.officeqa_judge import score_officeqa


class CheckSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", cli_parse_args=True,
    )
    val_dataset: Path = Field(
        ..., description="Path to val CSV with columns question + (ground_truth | answer)"
    )
    data_root: str = Field(
        ..., description="Corpus root passed as cwd/data_dir to the agent",
    )
    model: Optional[str] = Field(
        default="opus", description="Base agent model (opus, sonnet, haiku)",
    )
    max_concurrent: int = Field(default=8, description="Parallel agent runs")


async def main(settings: CheckSettings):
    set_sdk("claude")
    df = pd.read_csv(settings.val_dataset)
    gt_col = "ground_truth" if "ground_truth" in df.columns else "answer"
    items = [(str(r["question"]), str(r[gt_col])) for _, r in df.iterrows()]
    print(f"Loaded {len(items)} val samples from {settings.val_dataset}")
    project_root = Path.cwd()
    factory = (
        make_officeqa_agent_options(
            model=settings.model,
            project_root=str(project_root),
            data_dirs=[settings.data_root],
        )
        if settings.model else officeqa_agent_options
    )
    agent = Agent(factory, AgentResponse, name="base")

    t0 = time.time()
    results = await evaluate_agent_parallel(
        agent, items, max_concurrent=settings.max_concurrent, tag_prefix="check",
    )
    elapsed = time.time() - t0

    # Score each prediction with the LLM judge.
    print(f"\nScoring {len(results)} results with LLM judge...")
    rows = []
    total_cost = 0.0
    for r in results:
        if r.trace is None or r.trace.output is None:
            score, predicted = 0.0, "(no output — timeout/error)"
        else:
            predicted = str(r.trace.output.final_answer)
            score = score_officeqa(r.question, r.ground_truth, predicted)
        cost = float(getattr(r.trace, "total_cost_usd", 0) or 0) if r.trace else 0.0
        total_cost += cost
        rows.append((score, predicted, r.ground_truth, cost, r.question[:80]))

    avg = sum(s for s, *_ in rows) / max(len(rows), 1)
    print()
    print(f"{'='*70}")
    for i, (score, pred, gt, cost, q) in enumerate(rows, 1):
        flag = "OK" if score >= 0.99 else ("PARTIAL" if score > 0 else "FAIL")
        print(f"  [{flag}] q{i}: score={score:.2f}  cost=${cost:.4f}")
        print(f"        Q: {q}...")
        print(f"        GT: {gt[:120]}")
        print(f"        P:  {pred[:120]}")
    print(f"{'='*70}")
    print(f"Mean score: {avg:.4f}  |  Total cost: ${total_cost:.4f}  |  Wall: {elapsed:.0f}s")


if __name__ == "__main__":
    settings = CheckSettings()
    asyncio.run(main(settings))
