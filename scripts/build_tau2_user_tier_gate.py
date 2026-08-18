#!/usr/bin/env python3
"""Offline, one-time builder for the tau2 user-sim model-tier gate manifest.

Runs `classify_user_tier` over every task in a domain's raw task JSON and
writes `data/tau2_user_tier_gate/<domain>.json`. Committing this file means
the gate costs nothing per evolution run and every verdict is human-
auditable/overridable by hand (see `user_tier_gate.py`'s module docstring
for the full rationale and hard-requirement justification).

Usage:
    python scripts/build_tau2_user_tier_gate.py --domain retail
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent_profiles.tau2_agent.dataset import load_raw_tasks
from src.agent_profiles.tau2_agent.user_tier_gate import classify_user_tier, save_manifest

JUDGE_MODEL = "openrouter/openai/gpt-4.1"  # reference-tier -- misclassifying
                                            # here is exactly the failure
                                            # this gate exists to prevent.
CONCURRENCY = 10


async def _run(domain: str, tau2_root: str | None) -> None:
    api_key = _load_openrouter_key()
    raw_tasks = load_raw_tasks(domain, tau2_root=tau2_root)
    print(f"Classifying {len(raw_tasks)} {domain} tasks with {JUDGE_MODEL}...", flush=True)

    sem = asyncio.Semaphore(CONCURRENCY)
    results: dict[str, dict] = {}

    async def _one(task_id: str, task: dict) -> None:
        async with sem:
            verdict = await classify_user_tier(task, judge_model=JUDGE_MODEL, api_key=api_key)
            results[task_id] = verdict
            tier = "REFERENCE" if verdict["requires_reference_tier"] else "cheap"
            print(f"  [{task_id}] {tier}: {verdict['reasoning'][:100]}", flush=True)

    await asyncio.gather(*(_one(tid, t) for tid, t in raw_tasks.items()))

    n_ref = sum(1 for v in results.values() if v["requires_reference_tier"])
    print(f"\n{n_ref}/{len(results)} tasks classified as requiring reference-tier user-sim.")

    save_manifest(domain, results)
    print(f"Wrote data/tau2_user_tier_gate/{domain}.json")


def _load_openrouter_key() -> str:
    import os
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        return key
    env_path = Path(os.environ.get("EVOSKILL_ENV_FILE",
                                   Path(__file__).resolve().parents[1] / ".env"))
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError(f"OPENROUTER_API_KEY not set and not found in {env_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="retail")
    parser.add_argument("--tau2-root", default=None)
    args = parser.parse_args()
    asyncio.run(_run(args.domain, args.tau2_root))
