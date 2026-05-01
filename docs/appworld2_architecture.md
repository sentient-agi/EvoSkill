# EvoSkill × HALO AppWorld2: Complete Architecture

How EvoSkill's self-improving loop works on top of HALO's AppWorld pipeline.

---

## The Two Systems

There are **two completely separate AI systems** working together. They share ONE file on disk.

```
┌───────────────────────────────────┐     ┌───────────────────────────────────┐
│         HALO's Pipeline           │     │       EvoSkill's Pipeline          │
│                                   │     │                                   │
│  "Run tasks and evaluate them"    │     │  "Analyze failures and improve    │
│                                   │     │   the system prompt"              │
│  Uses: OpenAI Agents SDK          │     │  Uses: Claude Code SDK            │
│        + LiteLLM → Claude API     │     │        (Anthropic API directly)   │
│        + AppWorld MCP server      │     │                                   │
│        + AppWorld evaluation      │     │                                   │
└───────────────┬───────────────────┘     └───────────────┬───────────────────┘
                │                                         │
                │         ┌───────────────────┐           │
                └────────>│  instructions.txt │<──────────┘
                  reads   │                   │   writes
                          │  (the ONE shared  │
                          │   file on disk)   │
                          └───────────────────┘
```

**HALO reads** `instructions.txt` as the system prompt every time it runs a task.
**EvoSkill writes** a new `instructions.txt` after analyzing failures.

That's the entire connection. Everything else is independent.

---

## Who Runs What

### Task Execution: HALO (OpenAI Agents SDK)

When the agent actually talks to AppWorld's Spotify/Venmo/Gmail APIs, it's running inside **HALO's pipeline**, NOT Claude Code:

```
HALOAgent.run("task_id:::instruction")
    │
    ▼
HALO's run_experiment(task_id="50e1ac9_1")
    │
    ├── OpenAI Agents SDK creates an Agent object
    │   ├── Model: Claude Sonnet (via LiteLLM → Anthropic API)
    │   ├── System prompt: instructions.txt (read from disk)
    │   ├── Few-shot demos: demos.json (3 worked examples)
    │   ├── MCP server: AppWorld HTTP server (Spotify, Venmo, etc.)
    │   └── Tool filtering: API predictor limits to ~20 of 457 tools
    │
    ├── Agent loop (up to 50 turns):
    │   │
    │   │  ┌─────────────────┐    MCP     ┌──────────────┐    HTTP    ┌──────────┐
    │   │  │ OpenAI Agents   │──────────>│ AppWorld MCP │──────────>│ SQLite   │
    │   │  │ SDK + Claude    │<──────────│ Server       │<──────────│ DBs      │
    │   │  │ (via LiteLLM)   │           │              │           │          │
    │   │  └─────────────────┘           └──────────────┘           └──────────┘
    │   │
    │   │  Turn 1: supervisor__show_profile() → {name: Glenn Burton...}
    │   │  Turn 2: supervisor__show_account_passwords() → {spotify: {pw:...}}
    │   │  Turn 3: spotify__login(email, pw) → {access_token: abc123}
    │   │  Turn 4: spotify__show_song_library(token) → {songs: [...]}
    │   │  Turn 5: supervisor__complete_task(answer="Song A, Song B")
    │
    ├── AppWorld evaluates (official eval):
    │   ├── Compares answer to ground truth
    │   ├── Checks DB state changes
    │   └── Writes result to: evaluations/on_only_{task_id}.json
    │
    └── Saves to disk:
        ├── tasks/{task_id}/dbs/supervisor.jsonl    (agent's answer)
        ├── tasks/{task_id}/logs/lm_calls.jsonl     (full conversation)
        └── evaluations/on_only_{task_id}.json       (pass/fail + score)
```

**Key point**: Claude is the LLM brain, but it's orchestrated by HALO's OpenAI Agents SDK, not by Claude Code. HALO manages the MCP connection, tool calling, turn management, and state saving.

### Failure Analysis: EvoSkill (Claude Code SDK / Anthropic API)

When EvoSkill analyzes failures and generates a new prompt, it uses **Claude directly via the Anthropic API**:

```
SelfImprovingLoop._mutate()  (prompt_only mode)
    │
    ├── Prompt Proposer (Claude call #1):
    │   │
    │   │  System: PROMPT_PROPOSER_SYSTEM_PROMPT
    │   │  "You are an expert at analyzing agent failures
    │   │   and proposing prompt improvements..."
    │   │
    │   │  User query:
    │   │  "Here are 2 failures from AppWorld tasks:
    │   │   Failure 1: Agent called 15 APIs but got wrong answer...
    │   │   Failure 2: Agent didn't paginate through all results...
    │   │
    │   │   What prompt change would fix these?"
    │   │
    │   │  Tools available: Read, Bash, Glob, Grep
    │   │  (can explore the codebase for context)
    │   │
    │   └── Output: {
    │         proposed_prompt_change: "Add explicit pagination
    │           instructions: 'Always loop through ALL pages
    │           using page_index until you get an empty result'",
    │         justification: "Both failures occurred because
    │           the agent stopped at page 0..."
    │       }
    │
    └── Prompt Generator (Claude call #2):
        │
        │  System: PROMPT_GENERATOR_SYSTEM_PROMPT
        │  "You are an expert prompt engineer. Given the
        │   original prompt and a proposed change, generate
        │   an optimized prompt..."
        │
        │  User query:
        │  "Original prompt: [current instructions.txt]
        │   Proposed change: Add explicit pagination...
        │   Justification: Both failures occurred because..."
        │
        └── Output: {
              optimized_prompt: "I am your supervisor, and you
                are an AI Assistant...
                [full rewritten instructions.txt with pagination
                 guidance added to Section B]...",
              reasoning: "Added item 8 to Section B: 'Paginated
                APIs: Always process ALL results...'"
            }
```

**Key point**: The proposer and generator are regular Claude API calls (via Anthropic SDK). They don't run inside HALO or OpenAI Agents SDK. They're standard EvoSkill agents — same ones used for the OfficeQA example.

---

## The Complete Iteration Flow

Here's one full iteration with exact data flows:

```
═══════════════════════════════════════════════════════
 ITERATION 1
═══════════════════════════════════════════════════════

 ┌─────────────────────────────────────────────────┐
 │  1. SAMPLE TRAINING TASKS                       │
 │                                                 │
 │  SelfImprovingLoop picks 2 tasks:               │
 │  • "50e1ac9_1:::Give me top 4 R&B songs..."    │
 │  • "fac291d_2:::How many unique songs..."       │
 └────────────────────┬────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────┐
 │  2. RUN TASKS VIA HALO                          │
 │                                                 │
 │  For each task:                                 │
 │    HALOAgent.run(question)                      │
 │      → HALO's run_experiment(task_id)           │
 │      → OpenAI Agents SDK + Claude (via LiteLLM) │
 │      → AppWorld MCP → Spotify/Supervisor APIs   │
 │      → Official evaluation                      │
 │      → Results written to disk                  │
 │                                                 │
 │  SDK: OpenAI Agents SDK                         │
 │  Model: Claude Sonnet (via LiteLLM)             │
 │  Prompt: instructions.txt (current version)     │
 └────────────────────┬────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────┐
 │  3. SCORE & COLLECT FAILURES                    │
 │                                                 │
 │  Read from disk:                                │
 │    evaluations/on_only_50e1ac9_1.json → 100%    │
 │    evaluations/on_only_fac291d_2.json →   0%    │
 │                                                 │
 │  Task 50e1ac9_1: score=1.0 → PASS              │
 │  Task fac291d_2: score=0.0 → FAIL              │
 │                                                 │
 │  Failures: [{trace, answer, ground_truth}]      │
 └────────────────────┬────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────┐
 │  4. PROMPT PROPOSER (Claude call)               │
 │                                                 │
 │  SDK: Anthropic API (direct)                    │
 │  Model: Claude Sonnet                           │
 │                                                 │
 │  Input:                                         │
 │    • Failure trace from fac291d_2               │
 │      (what tools were called, what went wrong)  │
 │    • Feedback history (empty on iter 1)         │
 │    • Current instructions.txt                   │
 │                                                 │
 │  Output:                                        │
 │    "Add pagination guidance..."                 │
 └────────────────────┬────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────┐
 │  5. PROMPT GENERATOR (Claude call)              │
 │                                                 │
 │  SDK: Anthropic API (direct)                    │
 │  Model: Claude Sonnet                           │
 │                                                 │
 │  Input:                                         │
 │    • Current instructions.txt                   │
 │    • Proposed change from step 4                │
 │                                                 │
 │  Output:                                        │
 │    • Complete new instructions.txt              │
 │                                                 │
 │  *** WRITES TO DISK: instructions.txt ***       │
 └────────────────────┬────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────┐
 │  6. EVALUATE ON VALIDATION SET                  │
 │                                                 │
 │  For each of 17 val tasks:                      │
 │    HALOAgent.run(question)                      │
 │      → HALO reads NEW instructions.txt          │
 │      → Executes task with improved prompt       │
 │      → Official evaluation                      │
 │                                                 │
 │  SDK: OpenAI Agents SDK (same as step 2)        │
 │  Model: Claude Sonnet (via LiteLLM)             │
 │  Prompt: NEW instructions.txt (just written)    │
 │                                                 │
 │  Average score: 0.65 (was 0.55)                 │
 └────────────────────┬────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────┐
 │  7. KEEP OR DISCARD                             │
 │                                                 │
 │  0.65 > 0.55 → IMPROVED!                        │
 │  Keep new instructions.txt                      │
 │  Git commit: "iter-prompt-1: Add pagination..." │
 │  Add to frontier                                │
 │                                                 │
 │  → Continue to iteration 2                      │
 └─────────────────────────────────────────────────┘
```

---

## SDK Usage Map

There are exactly **3 places** where an LLM is called:

```
┌──────────────────┬─────────────────────┬──────────────────┬─────────────┐
│ Component        │ SDK                 │ Model            │ Purpose     │
├──────────────────┼─────────────────────┼──────────────────┼─────────────┤
│ HALO Agent       │ OpenAI Agents SDK   │ Claude Sonnet    │ Execute     │
│ (task execution) │ + LiteLLM           │ (via LiteLLM)    │ AppWorld    │
│                  │                     │                  │ tasks       │
├──────────────────┼─────────────────────┼──────────────────┼─────────────┤
│ Prompt Proposer  │ Anthropic API       │ Claude Sonnet    │ Analyze     │
│ (failure         │ (via Claude Code    │ (direct)         │ failures,   │
│  analysis)       │  SDK / harness)     │                  │ propose fix │
├──────────────────┼─────────────────────┼──────────────────┼─────────────┤
│ Prompt Generator │ Anthropic API       │ Claude Sonnet    │ Rewrite     │
│ (prompt          │ (via Claude Code    │ (direct)         │ the system  │
│  rewriting)      │  SDK / harness)     │                  │ prompt      │
└──────────────────┴─────────────────────┴──────────────────┴─────────────┘
```

**Note**: The HALO Agent uses the OpenAI Agents SDK (which is OpenAI's agentic framework), but it calls Claude via LiteLLM (a proxy that routes to Anthropic's API). So **all 3 components ultimately call Claude** — just through different SDKs.

Why not use the same SDK for all? Because HALO's agent loop, MCP integration, tool calling, and evaluation are all built on top of OpenAI Agents SDK. Reimplementing that in Claude Code SDK is what caused our 0% score in `appworld/`. By keeping HALO's proven SDK for execution and only using EvoSkill's SDK for the evolution part, both systems work correctly.

---

## What Gets Evolved

EvoSkill modifies ONE file: `instructions.txt`. This is HALO's system prompt template.

### Before Evolution (base prompt):
```
I am your supervisor, and you are an AI Assistant whose job is to
complete my day-to-day tasks fully autonomously.
...
A. General instructions:
- Act fully on your own...
- Never invent or guess values...

B. App-specific instructions:
- All personal info via Supervisor app...
- Paginated APIs: Always process all results...

C. Task-completion instructions:
- Call supervisor__complete_task when done...
```

### After 1 Iteration (evolved prompt):
```
I am your supervisor, and you are an AI Assistant whose job is to
complete my day-to-day tasks fully autonomously.
...
A. General instructions:
- Act fully on your own...
- Never invent or guess values...

B. App-specific instructions:
- All personal info via Supervisor app...
- Paginated APIs: Always process all results, looping through
  the page_index. Don't stop at the first page. Continue until
  the API returns an empty list or fewer results than the page size.
  ^^^ NEW: Added specific pagination stop condition ^^^
- When searching across multiple libraries (songs, albums, playlists),
  search ALL of them and combine results before answering.
  ^^^ NEW: Cross-library search guidance ^^^

C. Task-completion instructions:
- Call supervisor__complete_task when done...
- Before submitting, verify your answer matches the exact format
  requested (comma-separated, numbered list, etc.)
  ^^^ NEW: Answer format verification ^^^
```

The few-shot demos (`demos.json`) and API predictor prompt (`api_predictor.txt`) are NOT modified — only the system prompt changes.

---

## File Layout

```
HALO repo: /Users/sarveshkhetan/work/HALO/demo/appworld/
├── experiments/
│   ├── prompts/function_calling_agent/
│   │   ├── instructions.txt          ← THE FILE THAT GETS EVOLVED
│   │   └── demos.json                  (read-only, not evolved)
│   ├── prompts/api_predictor.txt       (read-only, not evolved)
│   ├── code/openai_agents/
│   │   └── run.py                      (HALO's runner, not modified)
│   └── outputs/{experiment}/
│       ├── tasks/{task_id}/
│       │   ├── dbs/supervisor.jsonl    (agent's answer)
│       │   └── logs/lm_calls.jsonl     (full conversation trace)
│       └── evaluations/
│           └── on_only_{task_id}.json  (official eval result)
│
├── data/
│   ├── datasets/dev.txt                (57 task IDs)
│   └── tasks/{task_id}/
│       ├── specs.json                  (instruction + user info)
│       └── ground_truth/               (answer + eval code)

EvoSkill repo: /Users/sarveshkhetan/work/EvoSkill/
├── examples/appworld2/scripts/
│   ├── halo_agent.py                   ← HALOAgent wrapping HALO's runner
│   ├── run_evolution.py                ← Wires into SelfImprovingLoop
│   └── build_config.py                 ← Builds HALO's runner config
│
├── src/loop/runner.py                  ← SelfImprovingLoop (prompt_only mode)
├── src/agent_profiles/
│   ├── prompt_proposer/                ← Analyzes failures, proposes changes
│   └── prompt_generator/               ← Rewrites instructions.txt
└── src/harness/agent.py                ← Agent base class
```

---

## Cost & Time Per Iteration

| Step | # of Claude calls | ~Cost | ~Time |
|------|-------------------|-------|-------|
| Train tasks (2-3 samples) | 2-3 calls (via HALO) | ~$3 | ~3 min |
| Prompt proposer | 1 call (Anthropic API) | ~$0.10 | ~30s |
| Prompt generator | 1 call (Anthropic API) | ~$0.10 | ~30s |
| Val tasks (17 samples) | 17 calls (via HALO) | ~$17 | ~17 min |
| **Total per iteration** | **~21 calls** | **~$20** | **~21 min** |
| **5 iterations** | **~105 calls** | **~$100** | **~1.75 hrs** |

The proposer/generator are cheap (~$0.20 combined). The HALO task execution dominates cost and time.

---

## Your Question Answered

> HALO uses OpenAI agents so the tasks will be actually executed in OpenAI agents but then traces and all will be analysed by EvoSkill so agents like Claude Code / OpenHands / OpenCode / Goose will be used for it right?

**Partially correct.** Here's the precise answer:

1. **Task execution**: YES, runs in OpenAI Agents SDK (HALO's pipeline). Claude is the LLM but it's orchestrated by OpenAI's SDK, not Claude Code.

2. **Trace analysis**: The proposer/generator use EvoSkill's harness which calls the **Anthropic API directly** via Claude Code SDK. They don't use OpenHands/OpenCode/Goose — those are alternative harnesses for EvoSkill's base agent, which in this case is HALOAgent (not a Claude Code agent).

3. **New prompt runs again in OpenAI Agents?**: YES, exactly. After the prompt generator writes a new `instructions.txt`, the next HALO run reads that file and uses it as the system prompt — running through the same OpenAI Agents SDK pipeline.

> Once the new prompts are made the agent with new prompt will run again in OpenAI agents?

**YES.** The cycle is:

```
HALO (OpenAI Agents SDK) runs tasks with current prompt
          │
          ▼
    failures collected
          │
          ▼
EvoSkill (Anthropic API) analyzes failures
          │
          ▼
EvoSkill writes new instructions.txt
          │
          ▼
HALO (OpenAI Agents SDK) runs tasks with NEW prompt  ← same SDK, new prompt
          │
          ▼
    improved results (hopefully)
```
