"""Analyze 133-eval failures to classify by root cause."""
import asyncio
import os
import pickle
import re
import sys

import pandas as pd
from anthropic import AsyncAnthropic

PKL = "/Users/dastin/dev/EvoSkill/results/eval_pro133_swap_iter1.pkl"
DATASET = "/Users/dastin/dev/officeqa/officeqa_pro.csv"
JUDGE_MODEL = "claude-haiku-4-5-20251001"
CONCURRENCY = 10

EQUIV_PROMPT = """You are evaluating whether an agent's answer matches the ground truth.

Question: {question}
Ground truth: {gt}
Agent answer: {ag}

The agent's answer is EQUIVALENT to the ground truth if the values match (allow formatting differences). Reply ONLY: EQUIVALENT or DIFFERENT."""

CLASSIFY_PROMPT = """Classify this question by what data sources are needed to answer it.

Question: {question}

Output ONE of these labels:
- EXTERNAL_REQUIRED: question explicitly references an external source (Macrotrends, World Bank, Federal Reserve website, BEA, BLS, Census Bureau, NBER, FRED, OECD, IMF, Wikipedia, "online", "current market", a specific year-of-the-art ML model, etc.) — answering REQUIRES data beyond Treasury Bulletins
- TREASURY_ONLY: question can be answered from Treasury Bulletins alone (Treasury data, FFO, FD, MQ, IFS, CM, TSO, ESF, PDO, OFS tables, federal debt, foreign exchange, etc.)
- AMBIGUOUS: unclear, both could work

Reply with ONLY the label."""


async def call_judge(client, sem, prompt):
    async with sem:
        msg = await client.messages.create(
            model=JUDGE_MODEL, max_tokens=20,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in msg.content if hasattr(b, "text")).strip().upper()


async def main():
    with open(PKL, "rb") as f:
        results = pickle.load(f)
    data = pd.read_csv(DATASET)

    # Build per-result records
    records = []  # (uid, question, gt, ag, status)
    for r in results:
        row = data.iloc[r.index]
        uid = row["uid"]; q = str(row["question"]); gt = str(row["answer"]).strip()
        if r.error or not r.trace or not r.trace.output:
            records.append((uid, q, gt, None, "TIMEOUT"))
        else:
            ag = str(r.trace.output.final_answer).strip()
            records.append((uid, q, gt, ag, "PARSED"))

    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    sem = asyncio.Semaphore(CONCURRENCY)

    # Step 1: Judge equivalence on parsed
    parsed = [(i, rec) for i, rec in enumerate(records) if rec[3] is not None]
    print(f"Judging {len(parsed)} parsed answers...")
    verdicts = await asyncio.gather(*[
        call_judge(client, sem, EQUIV_PROMPT.format(
            question=rec[1][:500], gt=rec[2][:300], ag=rec[3][:300]))
        for _i, rec in parsed
    ])
    judge = {i: v for (i, _rec), v in zip(parsed, verdicts)}

    # Determine which failed (DIFFERENT or TIMEOUT)
    failures = []  # (uid, question, gt, ag, status, judge_verdict)
    for i, (uid, q, gt, ag, status) in enumerate(records):
        if status == "TIMEOUT":
            failures.append((uid, q, gt, ag, status, "—"))
        elif "DIFFERENT" in (judge.get(i) or ""):
            failures.append((uid, q, gt, ag, status, judge[i]))

    print(f"\nFailures: {len(failures)}/{len(records)}")
    print("  Timeouts:    {}".format(sum(1 for f in failures if f[4] == "TIMEOUT")))
    print("  Wrong answers (judge DIFFERENT): {}".format(sum(1 for f in failures if f[4] == "PARSED")))

    # Step 2: Classify each failure by source-needs
    print(f"\nClassifying failures by data-source need...")
    categories = await asyncio.gather(*[
        call_judge(client, sem, CLASSIFY_PROMPT.format(question=f[1][:600]))
        for f in failures
    ])

    ext, trsy, amb = 0, 0, 0
    rows = []
    for f, cat in zip(failures, categories):
        uid, q, gt, ag, status, jv = f
        cat_norm = "EXTERNAL_REQUIRED" if "EXTERNAL" in cat else (
            "TREASURY_ONLY" if "TREASURY" in cat else "AMBIGUOUS"
        )
        if cat_norm == "EXTERNAL_REQUIRED": ext += 1
        elif cat_norm == "TREASURY_ONLY": trsy += 1
        else: amb += 1
        rows.append((uid, status, cat_norm, gt, ag, q))

    print(f"\n=== Failure root-cause breakdown ({len(failures)} failures) ===")
    print(f"  EXTERNAL_REQUIRED: {ext}  (needs Macrotrends/World Bank/Fed Reserve/etc.)")
    print(f"  TREASURY_ONLY:     {trsy}  (data IS in bulletins; agent failed)")
    print(f"  AMBIGUOUS:         {amb}")

    # Print details, grouped
    for label in ["EXTERNAL_REQUIRED", "TREASURY_ONLY", "AMBIGUOUS"]:
        sel = [r for r in rows if r[2] == label]
        if not sel: continue
        print(f"\n--- {label} ({len(sel)}) ---")
        for uid, status, _cat, gt, ag, q in sel:
            ag_disp = (ag or "—")[:30]
            print(f"  {uid:<10} [{status:<8}]  GT={str(gt)[:25]:<27}  AG={ag_disp:<32}  Q: {q[:90]}")


if __name__ == "__main__":
    asyncio.run(main())
