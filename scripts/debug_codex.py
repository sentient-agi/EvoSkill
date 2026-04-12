"""Debug script to test Codex executor end-to-end.

Run from project root:
    uv run python scripts/debug_codex.py

Output saved to: debug_codex_output.log
"""

import asyncio
import json
import sys
from pathlib import Path


class TeeWriter:
    """Write to both stdout and a log file."""
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def close(self):
        self.log.close()


async def main():
    log_path = Path(__file__).parent.parent / "debug_codex_output.log"
    tee = TeeWriter(log_path)
    sys.stdout = tee

    print(f"Log file: {log_path}")
    print()

    from src.harness import set_sdk, Agent, build_options
    from src.schemas import AgentResponse, SkillProposerResponse

    set_sdk("codex")

    # ===================================================================
    # TEST 1: Simple schema (AgentResponse) — through the full Agent path
    # ===================================================================
    print("=" * 60)
    print("TEST 1: AgentResponse (simple — 2 fields)")
    print("=" * 60)
    print()

    options1 = build_options(
        system="You are a helpful assistant. Answer briefly.",
        schema=AgentResponse.model_json_schema(),
        tools=["Read", "Bash"],
        model="openrouter/anthropic/claude-sonnet-4-6",
    )

    print("OPTIONS:")
    print(json.dumps(options1, indent=2, default=str))
    print()

    agent1 = Agent(options=options1, response_model=AgentResponse)
    print("Running agent with query: 'What is 2+2?'")
    print()

    try:
        trace1 = await agent1.run("What is 2+2?")

        print(f"  output:              {trace1.output}")
        print(f"  output type:         {type(trace1.output).__name__}")
        print(f"  parse_error:         {trace1.parse_error}")
        print(f"  is_error:            {trace1.is_error}")
        print(f"  result (first 200):  {trace1.result[:200]}")
        print(f"  total_cost_usd:      {trace1.total_cost_usd}")
        print(f"  model:               {trace1.model}")
        print(f"  uuid:                {trace1.uuid}")
        print(f"  session_id:          {trace1.session_id}")
        print()

        if trace1.output:
            print(f"  PARSED final_answer: {trace1.output.final_answer}")
            print(f"  PARSED reasoning:    {trace1.output.reasoning}")
            print("  >>> TEST 1 PASSED <<<")
        else:
            print(f"  >>> TEST 1 FAILED — no structured output <<<")
    except Exception as e:
        print(f"  >>> TEST 1 ERROR: {type(e).__name__}: {e} <<<")
    print()

    # ===================================================================
    # TEST 2: Complex schema (SkillProposerResponse) — 5 fields
    # ===================================================================
    print("=" * 60)
    print("TEST 2: SkillProposerResponse (complex — 5 fields)")
    print("=" * 60)
    print()

    options2 = build_options(
        system="You are an agent performance analyst. Propose a skill improvement.",
        schema=SkillProposerResponse.model_json_schema(),
        tools=["Read", "Bash"],
        model="openrouter/anthropic/claude-sonnet-4-6",
    )

    agent2 = Agent(options=options2, response_model=SkillProposerResponse)
    print("Running agent with query: 'Propose a percentage calculation skill'")
    print()

    try:
        trace2 = await agent2.run(
            "The agent failed to calculate year-over-year percentage changes. "
            "It returned 5.2% but the correct answer was 15.2%. "
            "Propose a new skill to fix this."
        )

        print(f"  output:              {trace2.output}")
        print(f"  output type:         {type(trace2.output).__name__}")
        print(f"  parse_error:         {trace2.parse_error}")
        print(f"  is_error:            {trace2.is_error}")
        print(f"  result (first 300):  {trace2.result[:300]}")
        print(f"  total_cost_usd:      {trace2.total_cost_usd}")
        print(f"  model:               {trace2.model}")
        print()

        if trace2.output:
            print(f"  PARSED action:         {trace2.output.action}")
            print(f"  PARSED proposed_skill: {trace2.output.proposed_skill[:100]}")
            print(f"  PARSED justification:  {trace2.output.justification[:100]}")
            print("  >>> TEST 2 PASSED <<<")
        else:
            print(f"  >>> TEST 2 FAILED — no structured output <<<")
    except Exception as e:
        print(f"  >>> TEST 2 ERROR: {type(e).__name__}: {e} <<<")
    print()

    # ===================================================================
    # SUMMARY
    # ===================================================================
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    test1_ok = 'trace1' in dir() and trace1.output is not None
    test2_ok = 'trace2' in dir() and trace2.output is not None
    print(f"  Test 1 (AgentResponse):          {'PASS' if test1_ok else 'FAIL'}")
    print(f"  Test 2 (SkillProposerResponse):   {'PASS' if test2_ok else 'FAIL'}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
    print(f"\nDone. Log saved to: debug_codex_output.log")
