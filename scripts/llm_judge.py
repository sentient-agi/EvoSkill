"""LLM-as-judge equivalence check on completed eval results."""
import asyncio
import os
import pickle
import sys

import pandas as pd
from anthropic import AsyncAnthropic

PKL = "/Users/dastin/dev/EvoSkill/results/eval_pro133_swap_iter1.pkl"
DATASET = "/Users/dastin/dev/officeqa/officeqa_pro.csv"
MODEL = "claude-haiku-4-5-20251001"
CONCURRENCY = 10

PROMPT = """You are evaluating whether an agent's answer matches the ground truth.

Question: {question}
Ground truth: {gt}
Agent answer: {ag}

The agent's answer is EQUIVALENT to the ground truth if the values match (allow formatting differences: commas, % signs, brackets, ordering inside lists, decimal precision within ~1%, units like "million" vs raw number, etc.). It is NOT equivalent if the numerical value or text content differs meaningfully.

Reply with ONLY one word: EQUIVALENT or DIFFERENT."""


async def judge_one(client, sem, q, gt, ag):
    async with sem:
        msg = await client.messages.create(
            model=MODEL,
            max_tokens=10,
            messages=[{"role": "user", "content": PROMPT.format(question=q[:500], gt=gt[:300], ag=ag[:300])}],
        )
        text = "".join(b.text for b in msg.content if hasattr(b, "text")).strip().upper()
        return text


async def main():
    with open(PKL, "rb") as f:
        results = pickle.load(f)
    data = pd.read_csv(DATASET)
    pairs = []
    for r in results:
        if r.error or not r.trace or not r.trace.output:
            continue
        row = data.iloc[r.index]
        gt = str(row["answer"]).strip()
        ag = str(r.trace.output.final_answer).strip()
        q = str(row["question"])
        pairs.append((row["uid"], q, gt, ag))
    print(f"Judging {len(pairs)} pairs...")
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    sem = asyncio.Semaphore(CONCURRENCY)
    verdicts = await asyncio.gather(*[judge_one(client, sem, q, gt, ag) for _uid, q, gt, ag in pairs])

    eq = sum(1 for v in verdicts if "EQUIVALENT" in v)
    diff = sum(1 for v in verdicts if "DIFFERENT" in v)
    other = len(verdicts) - eq - diff
    print(f"\nEQUIVALENT: {eq}/{len(verdicts)} ({100*eq/len(verdicts):.1f}%)")
    print(f"DIFFERENT:  {diff}/{len(verdicts)}")
    print(f"OTHER:      {other}/{len(verdicts)}")
    print(f"\nDisagreements (judge vs strict exact):")
    print("  UID       Verdict      GT vs Agent")
    print("  " + "-" * 90)
    for (uid, _q, gt, ag), v in zip(pairs, verdicts):
        strict_match = gt == ag
        judge_match = "EQUIVALENT" in v
        if strict_match != judge_match:
            mk = "+" if judge_match else "-"
            print(f"  {uid:<10} {v:<12} {mk}  GT={gt[:30]:<32} AG={ag[:30]}")


if __name__ == "__main__":
    asyncio.run(main())
