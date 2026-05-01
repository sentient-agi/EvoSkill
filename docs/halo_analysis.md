# HALO: Hierarchical Agent Loop Optimization — Technical Understanding

*A competitive analysis document based on the HALO codebase.*

---

## What HALO Is

HALO (Hierarchical Agent Loop Optimization) is a framework for **automatic optimization of agent harnesses** — the system prompts, tool configurations, and orchestration logic that surround an LLM agent in production. It works by analyzing OpenTelemetry execution traces from failing agents, diagnosing systemic issues, and producing actionable fix reports that a coding agent (Claude Code, Cursor) can apply.

The core thesis: when an LLM agent fails in production, the problem is usually in the **harness** (bad prompt, missing tool, wrong error handling) rather than the model. HALO finds these harness-level patterns by reading execution traces at scale.

---

## The HALO Loop

```
    ┌─────────────────────┐
    │  Agent Harness       │    Production deployment
    │  (prompts + tools)   │
    └──────────┬──────────┘
               │ emits OTel spans
               ▼
    ┌─────────────────────┐
    │  Execution Traces    │    JSONL (OTLP format)
    │  (spans, errors,     │    Can be gigabytes
    │   tool calls, tokens)│
    └──────────┬──────────┘
               │ ingested
               ▼
    ┌─────────────────────┐
    │  HALO Engine         │    Multi-agent RLM analysis
    │  (trace indexing →   │
    │   hierarchical agent │
    │   analysis →         │
    │   findings report)   │
    └──────────┬──────────┘
               │ findings
               ▼
    ┌─────────────────────┐
    │  Coding Agent        │    Claude Code / Cursor
    │  (applies fixes to   │    Modifies prompts, tools,
    │   the harness)       │    error handling
    └──────────┬──────────┘
               │ redeploy
               ▼
         Back to top — iterate
```

This is a **human-in-the-loop** optimization cycle. HALO finds the problems; a coding agent fixes them; humans approve and redeploy.

---

## Core Architecture: Three Pillars

### Pillar 1: Trace Indexing

Production agents generate massive trace files (JSONL, potentially gigabytes). HALO can't parse the full file for every query.

**Solution**: A parallel multi-stage indexer builds a sidecar index file:

```
traces.jsonl                         traces.jsonl.engine-index.jsonl
┌──────────────────────┐             ┌──────────────────────────────────────┐
│ {"trace_id":"abc",   │  ──index──> │ {"trace_id":"abc",                   │
│  "span_id":"s1", ... }│             │  "byte_offsets": [0, 2048, 4096],   │
│ {"trace_id":"abc",   │             │  "span_count": 3,                    │
│  "span_id":"s2", ... }│             │  "has_errors": true,                 │
│ {"trace_id":"def",   │             │  "models": ["gpt-4o"],               │
│  "span_id":"s3", ... }│             │  "total_tokens": 15420}              │
│ ...                   │             │ {"trace_id":"def", ...}              │
└──────────────────────┘             └──────────────────────────────────────┘
```

The indexer runs 4 stages in parallel (up to 8 workers):
1. **Scan** — read JSONL sequentially, record byte offsets
2. **Chunk** — split into parallel work units
3. **Process** — each worker parses spans, groups by trace_id, accumulates metadata
4. **Merge** — combine worker results, write atomic sidecar file

This gives **O(1) random access** to any span via seek + read at the stored byte offset.

### Pillar 2: Hierarchical Multi-Agent Analysis

HALO doesn't use a single LLM call to analyze traces. It runs a **tree of specialized agents**:

```
                    Root Agent (depth 0)
                    "Diagnose errors and suggest fixes"
                   /          |          \
            Subagent 1    Subagent 2    Subagent 3
            (depth 1)     (depth 1)     (depth 1)
            "Analyze      "Analyze      "Analyze
             auth          API 4xx       timeout
             failures"     errors"       patterns"
               |
           Subagent 1a
           (depth 2)
           "Deep-dive trace abc123"
```

**Depth-aware semaphores** prevent deadlock: each depth level has its own concurrency semaphore. A parent at depth 0 can hold its slot while spawning children at depth 1, which wait on a separate semaphore. A single shared semaphore would deadlock (all slots held by parents waiting for children).

Default limits: `maximum_depth=2`, `maximum_parallel_subagents=4`.

Each agent has access to **trace query tools**:

| Tool | Purpose | Cost |
|------|---------|------|
| `get_dataset_overview` | High-level stats, sample trace IDs | Cheap |
| `query_traces` | Paginated trace summaries with filters | Cheap |
| `count_traces` | Fast count without materializing | Cheapest |
| `view_trace` | Load all spans of one trace (4KB/attribute cap) | Medium |
| `search_trace` | Substring search within one trace | Medium |
| `view_spans` | Surgical read of specific spans (16KB/attribute cap) | Expensive |
| `synthesize_traces` | LLM-backed cross-trace summary | Most expensive |

The tools have a **two-tier payload cap**:
- **Discovery tier** (`view_trace`, `search_trace`): 4KB per attribute — good for finding spans, protects context window
- **Surgical tier** (`view_spans`): 16KB per attribute — for detailed inspection of matched spans

If a trace exceeds the budget, the tool returns a summary with a recommendation to use `search_trace` + `view_spans` for targeted access.

### Pillar 3: Context Management

Long agent conversations blow up the context window. HALO uses **message compaction**:

- Keep the last N text messages (default: 12)
- Keep the last M tool-call turns (default: 3), where a "turn" = assistant's tool_calls + matching tool results
- Older items are summarized by a dedicated low-temperature compaction model
- Already-compacted items are skipped on subsequent passes

This keeps the agent's working memory bounded while preserving recent context.

---

## Additional Components

### Synthesis Tool

Agents can call `synthesize_traces` to get an LLM-generated summary across multiple traces. The tool:
1. Renders selected traces as plain text (budget-capped per trace)
2. Sends through a synthesis model with a specialized prompt
3. Returns a short cross-trace pattern summary

This helps agents identify **systemic** issues (e.g., "30% of traces fail on auth token refresh") rather than individual trace bugs.

### Code Execution Sandbox

Agents can run ad-hoc Python analysis in a sandboxed environment:
- **Runtime**: Deno hosting Pyodide (Python in WASM)
- **Available**: numpy, pandas, plus a `trace_store` object for programmatic trace access
- **Constraints**: Read-only, 60s timeout, 64KB stdout/stderr cap
- **Use case**: Statistical analysis ("what % of traces have >10 tool calls?"), pattern extraction

### Output Streaming

All output from root + subagents is streamed via `EngineOutputBus`:
- `AgentTextDelta` — incremental tokens
- `AgentOutputItem` — tool calls, results, messages
- Monotonic sequence numbers preserve ordering across the entire agent tree
- Both async (`stream_engine_async`) and sync (`run_engine`) APIs available

---

## Benchmark Results: AppWorld

HALO's primary benchmark is [AppWorld](https://appworld.dev) (ACL 2024) — 728 tasks across simulated apps (Spotify, Venmo, Gmail, Splitwise, etc.) with 457 APIs.

| Model | Split | Before HALO | After HALO | Gain |
|-------|-------|-------------|------------|------|
| Gemini 3 Flash | dev | 36.8% | 52.6% | **+15.8** |
| Gemini 3 Flash | test_normal | 37.5% | 48.2% | **+10.7** |
| Sonnet 4.6 | dev | 73.7% | 89.5% | **+15.8** |
| Sonnet 4.6 | test_normal | 62.5% | 73.2% | **+10.7** |

Issues found by HALO mapped cleanly to prompt edits: hallucinated tool calls, redundant arguments, refusal loops, semantic correctness gaps.

---

## HALO vs EvoSkill: Conceptual Comparison

| Dimension | HALO | EvoSkill |
|-----------|------|----------|
| **What it optimizes** | The agent harness (prompts, tools, orchestration) | Agent skills (learned behaviors, reusable procedures) |
| **Input** | Production OTel traces from deployed agents | Task execution traces from training questions |
| **Analysis method** | RLM-based hierarchical trace decomposition | Evolutionary loop (mutate skill → evaluate → select) |
| **Who applies fixes** | External coding agent (Claude Code, Cursor) | The evolver agent writes skills directly |
| **Iteration model** | Human-in-the-loop (analyze → fix → redeploy → repeat) | Fully automated (propose → evaluate → keep/discard) |
| **Scope per iteration** | Breadth-first: finds all failure modes in one pass | Depth-first: focuses on current failures, evolves one skill |
| **State across iterations** | Stateless per run (fresh trace analysis each time) | Stateful (TraceDB, frontier, skill versions, feedback history) |
| **Trace handling** | Indexed JSONL with O(1) random access, multi-tier payload caps | SQLite + markdown files with progressive disclosure |
| **Context management** | Message compaction (keep last N messages, summarize older) | Progressive disclosure (compact index → Read on demand) |
| **Multi-agent** | Hierarchical subagents with depth-aware semaphores | Sequential agents (proposer → generator, or unified evolver) |
| **Benchmarks** | AppWorld (728 tasks, 457 APIs, multi-app simulation) | Treasury-bulletin Q&A, SEAL-QA, LiveCodeBench, DABstep |
| **Best gains** | +15.8 points (Sonnet 4.6 on AppWorld dev) | 0.833 → 0.998 on Treasury-bulletin Q&A |

### Where they overlap

Both systems analyze agent execution traces to improve agent performance. Both use LLMs to diagnose failures. Both iterate.

### Where they diverge

**HALO finds problems; EvoSkill fixes them automatically.**

HALO produces a findings report that a human reviews and a coding agent implements. EvoSkill's evolver agent diagnoses AND writes the fix in a single automated pass, then evaluates the fix, and keeps or discards it — no human in the loop.

**HALO analyzes the harness; EvoSkill evolves skills.**

HALO's fixes are prompt edits, tool configuration changes, error handling improvements — changes to the infrastructure surrounding the model. EvoSkill creates new SKILL.md files that teach the agent new procedures — changes to the agent's learned knowledge.

**HALO is stateless; EvoSkill has memory.**

Each HALO run starts fresh with a new trace dataset. EvoSkill's TraceDB persists every trace across iterations, so the evolver can compare iteration 1's behavior to iteration 5's behavior. HALO doesn't need this because it analyzes production traces in bulk rather than iterating on the same questions.

**HALO handles scale differently.**

HALO is built for gigabyte-scale trace files with parallel indexing and O(1) random access. EvoSkill deals with smaller trace volumes (3-6 questions per iteration) but needs richer cross-iteration context. Different scale axes — HALO scales in trace volume, EvoSkill scales in iteration depth.

### Complementary, not competing

Despite being positioned as competitors, the two systems address different parts of the agent optimization lifecycle:

```
    Development                              Production
    ┌──────────────────────────┐            ┌──────────────────────────┐
    │  EvoSkill                │            │  HALO                    │
    │                          │            │                          │
    │  "What new skills does   │  deploy    │  "What's broken in my    │
    │   my agent need?"        │  ──────>   │   deployed agent?"       │
    │                          │            │                          │
    │  Evolves capabilities    │            │  Diagnoses failures      │
    │  on training data        │   <──────  │  on production traces    │
    │                          │  feed back │                          │
    └──────────────────────────┘            └──────────────────────────┘
```

HALO's findings could feed back into EvoSkill's training data — "these are the failure modes users actually hit" — and EvoSkill's evolved skills could be deployed into the harness that HALO monitors. They sit at different stages of the same feedback loop.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| LLM SDK | OpenAI SDK (2.32.0) + OpenAI Agents SDK (0.14.7) |
| Data validation | Pydantic 2.13.3 |
| Trace format | OTLP-compatible JSONL |
| Sandbox | Deno 2.7.14 + Pyodide 0.29.3 (WASM) |
| CLI | Typer 0.25.0 + Rich 15.0.0 |
| Build | Hatchling |
| Linting | Ruff (format + lint) |
| Type checking | BasedPyright (strict) |
| Testing | Pytest + pytest-asyncio |
| Task runner | go-task (Taskfile.yml) |
| Dependencies | uv + uv.lock |
