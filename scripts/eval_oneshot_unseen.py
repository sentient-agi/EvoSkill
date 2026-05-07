#!/usr/bin/env python3
"""One-shot base-agent eval over the unseen image-verified-correct UIDs.

Goal: identify which questions base sonnet (adaptive thinking, medium effort,
web tools enabled) actually fails on, so we can curate a hard-only train+val
pool. Random sampling from the 85-correct pool is dominated by easy questions,
producing 'no proposal' iters that waste evolution budget.

Outputs:
    .dataset/oneshot_scores.csv     — uid, question, ground_truth, score, predicted
    .dataset/oneshot_failures.csv   — same schema, only rows with score < 1.0
"""
from __future__ import annotations

import asyncio
import os

# Bump CLI output ceiling before SDK spawns any subprocess (mirrors run_loop.py).
os.environ.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "64000")

from src.tracing import init_tracing
init_tracing("evoskill-oneshot-eval")

from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.harness import Agent, set_sdk
from src.agent_profiles import make_officeqa_agent_options
from src.cache import RunCache, CacheConfig
from src.evaluation.evaluate import evaluate_agent_parallel
from src.evaluation.officeqa_judge import score_officeqa
from src.schemas import AgentResponse


# 48 UIDs to exclude from the 133-question OfficeQA Pro benchmark, leaving
# 85 image-verified-correct UIDs. Mirrors .dataset/_sample_correct_104.py.
EXCLUDED_UIDS = {
    # 19 numerically wrong
    "UID0018", "UID0032", "UID0037", "UID0053", "UID0055", "UID0062",
    "UID0096", "UID0102", "UID0109", "UID0113", "UID0135", "UID0136",
    "UID0154", "UID0196", "UID0201", "UID0212", "UID0223", "UID0226",
    "UID0244",
    # 1 defective wording
    "UID0216",
    # 9 undecidable
    "UID0005", "UID0044", "UID0077", "UID0114", "UID0140", "UID0147",
    "UID0150", "UID0165", "UID0188",
    # 19 defensible (alternate-but-valid interpretations score 0)
    "UID0173", "UID0174", "UID0207", "UID0213", "UID0240", "UID0245",
    "UID0010", "UID0065", "UID0093", "UID0121", "UID0183", "UID0214",
    "UID0220", "UID0168", "UID0219", "UID0030", "UID0056", "UID0084",
    "UID0225",
}


class OneShotSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore",
        cli_parse_args=True,
    )
    source_csv: Path = Field(
        default=Path("/Users/dastin/dev/officeqa/officeqa_pro.csv"),
        description="OfficeQA Pro source CSV (uid, question, answer, ...)",
    )
    skip_csvs: list[str] = Field(
        default_factory=lambda: [
            ".dataset/correct_train.csv",
            ".dataset/correct_val.csv",
        ],
        description="CSVs whose UIDs to skip (already-evaluated set)",
    )
    out_scores: Path = Field(default=Path(".dataset/oneshot_scores.csv"))
    out_failures: Path = Field(default=Path(".dataset/oneshot_failures.csv"))
    failure_threshold: float = Field(
        default=1.0,
        description="Rows with score < this go to failures CSV (1.0 = anything not perfect)",
    )
    concurrency: int = Field(default=32)
    model: str = Field(default="sonnet")
    base_thinking: str = Field(default="adaptive")
    base_effort: str = Field(default="medium")
    data_root: Path = Field(default=Path("/Users/dastin/dev/officeqa/data"))
    project_root: Path = Field(default=Path("/Users/dastin/dev/EvoSkill"))


def _load_pool(s: OneShotSettings) -> pd.DataFrame:
    full = pd.read_csv(s.source_csv)
    pool = full[~full["uid"].isin(EXCLUDED_UIDS)].copy()
    assert len(pool) == 85, f"Expected 85 correct UIDs, got {len(pool)}"

    skip_uids: set[str] = set()
    for path in s.skip_csvs:
        p = Path(path)
        if not p.exists():
            print(f"[WARN] skip CSV not found: {p}")
            continue
        df = pd.read_csv(p)
        skip_uids |= set(df["uid"].astype(str).tolist())

    unseen = pool[~pool["uid"].isin(skip_uids)].copy().reset_index(drop=True)
    print(f"Pool=85, already-evaluated={len(skip_uids)}, unseen={len(unseen)}")
    return unseen


async def main() -> None:
    s = OneShotSettings()
    set_sdk("claude")

    unseen = _load_pool(s)

    options = make_officeqa_agent_options(
        model=s.model,
        project_root=str(s.project_root),
        data_dirs=[str(s.data_root)],
        thinking={"type": s.base_thinking},
        effort=s.base_effort,
    )
    agent = Agent(options, AgentResponse, name="base_oneshot")

    # Construct a RunCache pointing at the EvoSkill project's `.cache/runs/`
    # — same path the run_loop's RunCache uses, so a subsequent evolution run
    # will hit cache for any question this oneshot evaluates here. Cache key
    # depends on (project tree hash + skills tree hash + prompt hash) × question
    # × sdk × model — see src/cache/run_cache.py.
    cache = RunCache(CacheConfig(
        cache_dir=s.project_root / ".cache" / "runs",
        enabled=True,
        cwd=s.project_root,
    ))

    items: list[tuple[str, str]] = [
        (str(row["question"]), str(row["answer"])) for _, row in unseen.iterrows()
    ]
    print(f"Running {len(items)} questions @ concurrency={s.concurrency} (cache: {cache.config.cache_dir})...")
    results = await evaluate_agent_parallel(
        agent=agent, items=items,
        max_concurrent=s.concurrency, tag_prefix="oneshot",
        cache=cache,
    )

    rows: list[dict] = []
    total_cost = 0.0
    for uid, (_, gt), res in zip(unseen["uid"], items, results):
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
            "question": res.question,
            "ground_truth": gt,
            "category": "all",
            "score": round(score, 4),
            "predicted": predicted,
            "cost_usd": round(cost, 4),
            "no_output": no_output,
        })

    df = pd.DataFrame(rows)
    s.out_scores.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(s.out_scores, index=False)
    print(f"\n[OK] wrote {s.out_scores} ({len(df)} rows)")

    fails = df[df["score"] < s.failure_threshold].copy()
    fails[["uid", "question", "ground_truth", "category"]].to_csv(
        s.out_failures, index=False,
    )
    print(f"[OK] wrote {s.out_failures} ({len(fails)} failures)")

    n_perfect = (df["score"] >= 1.0).sum()
    n_zero = (df["score"] <= 0.0).sum()
    n_partial = len(df) - n_perfect - n_zero
    print(
        f"\nSummary: {n_perfect}/{len(df)} perfect (1.0), "
        f"{n_partial} partial, {n_zero} zero. "
        f"Avg score: {df['score'].mean():.4f}. "
        f"Total cost: ${total_cost:.4f}"
    )


if __name__ == "__main__":
    asyncio.run(main())
