import argparse
import asyncio
import pandas as pd
from pathlib import Path
from src.agent_profiles import set_sdk, Agent
from src.evaluation.eval_full import evaluate_full, load_results
from src.evaluation.sealqa_scorer import score_sealqa
from src.schemas import AgentResponse
from src.agent_profiles.sealqa_agent import get_sealqa_agent_options

# Set SDK to opencode (uses `opencode run` CLI under the hood)
set_sdk("opencode")


async def run(limit: int | None = None, offset: int = 0):
    model = "gemini-3.1-flash-lite-preview"
    provider = "gemini"

    # Options dict for opencode CLI path
    def custom_factory():
        opts = get_sealqa_agent_options(model=model)
        if isinstance(opts, dict):
            opts["provider_id"] = provider
        return opts

    agent = Agent(custom_factory, AgentResponse)

    dataset = Path(".dataset/seal-0.csv")
    output = Path("results/sealqa_opencode_gemini_flash.pkl")

    data = pd.read_csv(dataset)
    items = [(idx, row["question"], row["answer"]) for idx, row in data.iterrows()]
    if offset:
        items = items[offset:]
    if limit:
        items = items[:limit]

    print(f"Running evaluation with model={model} provider={provider} offset={offset} count={len(items)}")

    results = await evaluate_full(
        agent=agent,
        items=items,
        output_path=output,
        max_concurrent=1,  # Exa rate limits with concurrent requests
        resume=True,
    )

    # Simple summary
    all_results = load_results(output)
    successful = [r for r in all_results if r.error is None]

    # Extract predicted answer: prefer structured output, fall back to text result
    correct = 0
    scored = 0
    for r in successful:
        predicted = None
        if r.trace:
            if r.trace.output and r.trace.output.final_answer:
                predicted = str(r.trace.output.final_answer)
            elif r.trace.result and r.trace.result.strip():
                predicted = r.trace.result.strip()

        if predicted:
            scored += 1
            score = score_sealqa(
                r.question, str(r.ground_truth), predicted
            )
            if score > 0:
                correct += 1

    print(f"Completed: {len(all_results)}/{len(items)}")
    print(f"Successful (no errors): {len(successful)}")
    print(f"Scored (had answer): {scored}/{len(successful)}")
    print(
        f"Accuracy: {correct}/{scored} ({correct / scored * 100:.1f}%)"
        if scored
        else "Accuracy: N/A (no answers to score)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Max questions to evaluate")
    parser.add_argument("--offset", type=int, default=0, help="Skip first N questions")
    args = parser.parse_args()
    asyncio.run(run(limit=args.limit, offset=args.offset))
