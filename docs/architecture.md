# EvoSkill Architecture: A Beginner's Guide

## The Big Picture (30-second version)

```
You give EvoSkill:
  - A CSV of questions + correct answers
  - A task description ("answer financial questions")

EvoSkill runs a loop:
  1. Agent tries questions → some fail
  2. "Proposer" agent analyzes WHY they failed
  3. "Generator" agent creates a fix (a new skill file)
  4. Agent retries with the new skill → did score improve?
  5. YES → keep it, NO → throw it away
  6. Repeat up to 20 times or until stuck

You get back:
  - The best-performing agent configuration
  - All discovered skills
  - A score showing how much it improved
```

---

## How to think about the layers

Think of EvoSkill as an **onion with 4 rings**. Each ring builds on the one inside it.

### Ring 1 (innermost): The Core Idea — "Try, Fail, Learn, Repeat"

**File**: `src/feedback_descent.py`

This is the simplest part. It's just a loop:

```
current_best = starting_point
history_of_failures = []

repeat:
    new_attempt = make_something_new(current_best, history_of_failures)

    is_new_better = compare(current_best, new_attempt)

    if yes:
        current_best = new_attempt
        history_of_failures = []   ← start fresh, old failures don't matter anymore
    else:
        history_of_failures.append("this didn't work because...")

    if too_many_failures_in_a_row:
        give up
```

**Key insight**: When something works, you forget all the old failures. Why? Because you have a new baseline now — what failed against the old version might not be relevant anymore.

**This code is abstract** — it doesn't know about agents, skills, or questions. It just knows "try things, keep what's better, remember what didn't work."

---

### Ring 2: The Loop — "Make the Agent Better"

**File**: `src/loop/runner.py` (this is the biggest, most important file)

This ring takes the abstract "try, fail, learn" idea and applies it to **AI agents**.

#### What is a "Program"?

A "program" = everything that defines how your agent behaves:

- Its **system prompt** (instructions telling it how to think)
- Its **skills** (files in `.claude/skills/` that give it new capabilities)

#### What happens each iteration?

Here's one loop iteration, step by step:

**Step 1: Pick a parent to improve**

```
"Which version of the agent should we try to make better?"
→ Usually the best-scoring one (but can also pick randomly for variety)
```

**Step 2: Find what's broken**

```
Run the agent on some training questions
"Hey agent, what's the revenue in Q3?"
Agent answers: "$4.2B"      ← correct, skip
Agent answers: "I don't know" ← WRONG, this is a failure

Collect all the failures (typically 4-6 per iteration)
```

**Step 3: Ask the Proposer "what should we do?"**

```
The Proposer agent gets:
  - The failure details (what the agent tried, what went wrong)
  - History of past attempts ("last time we tried X and it didn't help")
  - List of existing skills

The Proposer says something like:
  "The agent keeps failing on percentage calculations.
   We should CREATE a new skill called 'percentage-calculator'
   that teaches it how to compute year-over-year changes."
```

**Step 4: Ask the Generator to build it**

```
The Generator agent gets the Proposer's idea and actually writes the code.
It creates a file like: .claude/skills/percentage-calculator/SKILL.md
```

**Step 5: Test the new version**

```
Run the agent WITH the new skill on a SEPARATE set of questions (validation set)
"Did the score go up?"
```

**Step 6: Keep or discard**

```
If score improved → keep it! Save it as a git branch.
If score didn't improve → delete it, record "this didn't work"
If nothing has improved for 5 iterations → stop, we're stuck
```

#### Why two sets of questions?

- **Training set**: Used to find failures (Step 2). The proposer sees these.
- **Validation set**: Used to test improvements (Step 5). The proposer NEVER sees these.

This prevents "cheating" — the agent might learn to handle specific training questions without actually getting better in general.

---

### Ring 3: The Supporting Infrastructure

These are the helper systems that make the loop work reliably.

#### 3a. Harness Layer (`src/harness/`)

**What it does**: Wraps agent SDKs (Claude Code, OpenCode, future Goose/OpenHands) so you can run agents and get structured results. This is separated from agent profiles so adding a new harness doesn't touch any profile code.

```python
# Simplified view:
from src.harness import Agent, AgentTrace

agent = Agent(options=how_to_configure_the_agent, response_model=WhatOutputLooksLike)
result = await agent.run("What is 2+2?")

result.output        # → parsed answer (e.g., AgentResponse with final_answer="4")
result.total_cost_usd  # → how much it cost
result.duration_ms     # → how long it took
```

The harness directory contains:

```
src/harness/
├── agent.py               ← Agent class + AgentTrace (the public interface)
├── sdk_config.py          ← Global SDK toggle (set_sdk("claude") / set_sdk("opencode"))
├── options_utils.py       ← build_claudecode_options() / build_opencode_options()
├── _claude_executor.py    ← Claude SDK execution + response parsing
└── _opencode_executor.py  ← OpenCode SDK execution + server management + parsing
```

- **Retries**: If the SDK fails, it retries 3 times with increasing delays (30s, 60s, 120s)
- **Timeout**: 20 minutes max per call
- **Structured output**: Forces the response into a Pydantic model (like a typed dictionary)
- **Per-project servers**: OpenCode runs as a local HTTP server — each project gets its own port

#### 3b. Agent Profiles (`src/agent_profiles/`)

**What it does**: Defines **what** each agent role does (system prompt, tools, output schema). Imports from the harness layer to build options for the active SDK.

There are **5 agent roles**, each with different instructions and tools:

| Who                  | What they do                              | What they output                                           |
| -------------------- | ----------------------------------------- | ---------------------------------------------------------- |
| **Base Agent**       | Answers the actual questions              | `{final_answer, reasoning}`                                |
| **Skill Proposer**   | Analyzes failures, proposes what to build | `{action: "create"/"edit", proposed_skill, justification}` |
| **Skill Generator**  | Actually writes the skill code            | `{generated_skill, reasoning}`                             |
| **Prompt Proposer**  | Suggests how to change the system prompt  | `{proposed_prompt_change, justification}`                  |
| **Prompt Generator** | Rewrites the system prompt                | `{optimized_prompt, reasoning}`                            |

#### 3c. Git Versioning (`src/registry/manager.py`)

**What it does**: Each version of the agent = a git branch.

```
program/base           ← the starting agent
program/iter-skill-1   ← after adding first skill
program/iter-skill-2   ← after adding second skill (might be discarded)
program/iter-skill-3   ← after adding third skill
```

Each branch contains:

- `.claude/program.yaml` — the agent's configuration
- `.claude/skills/` — the skill files it has

**The "frontier"** = the top 3 best-scoring versions, tracked with git tags.

Why git? Because you can:

- Go back to any version: `git checkout program/iter-skill-1`
- See exactly what changed: `git diff program/base program/iter-skill-3`
- Never lose work — discarded branches are deleted but the good ones stay

#### 3d. Caching (`src/cache/run_cache.py`)

**What it does**: Avoids re-running the same question if nothing changed.

The agent might answer 50 validation questions per iteration. If the skills didn't change between iterations, why re-run all 50? Cache the results!

**Smart part**: It only invalidates the cache when "behavior-affecting" files change:

- Skill files changed → cache invalid, re-run everything
- Just updated a score in metadata → cache still valid, reuse results

#### 3e. Scoring (`src/evaluation/`)

**What it does**: Decides if an answer is correct.

The default scorer uses **multi-tolerance matching**:

- Exact match: "4.2" = "4.2" → full credit
- Close match: "4.19" ≈ "4.2" (within 1%) → partial credit
- Way off: "5.0" ≠ "4.2" → no credit

Special scorers for specific tasks:

- **SEAL-QA**: Uses another AI model to judge if the answer is semantically correct
- **LiveCodeBench**: Runs generated code in a Docker container and checks if tests pass

---

### Ring 4 (outermost): How You Use It

There are two ways to use EvoSkill:

#### Way 1: Python API

```python
from src import EvoSkill

result = await EvoSkill(
    dataset="questions.csv",
    task="sealqa",
    mode="skill_only",      # discover new skills (vs rewriting the prompt)
    max_iterations=20,
).run()

print(result.best_score)        # e.g., 0.85
print(result.best_program)      # e.g., "iter-skill-7"
print(result.iterations_completed)  # e.g., 12
```

#### Way 2: CLI

```bash
# 1. Set up your project
evoskill init
# → Creates .evoskill/config.toml (settings) and .evoskill/task.md (what the agent should do)

# 2. Edit the config and task description

# 3. Run the improvement loop
evoskill run
# → Shows a live terminal table with scores, skills discovered, etc.

# 4. Check what was discovered
evoskill skills      # list all skills
evoskill logs        # see run history
evoskill diff base iter-skill-5  # compare two versions
```

The CLI loads settings from `.evoskill/config.toml`:

```toml
[harness]
name = "claude"
model = "sonnet"

[evolution]
mode = "skill_only"
iterations = 20
frontier_size = 3

[dataset]
path = "data/questions.csv"
question_column = "question"
ground_truth_column = "ground_truth"

[scorer]
type = "multi_tolerance"
```

---

## How all the files connect

```
YOU
 │
 ├── Python: EvoSkill(...)          CLI: evoskill run
 │   (src/api/evoskill.py)          (src/cli/commands/run.py)
 │           │                              │
 │           └──────────┬───────────────────┘
 │                      │
 │                      ▼
 │            SelfImprovingLoop        ← The main engine
 │            (src/loop/runner.py)        (Ring 2)
 │                      │
 │      ┌───────────────┼───────────────┐
 │      ▼               ▼               ▼
 │  Agent Profiles   Git Versioning   Caching
 │  (src/agent_      (src/registry/)  (src/cache/)
 │   profiles/)           │               │
 │      │                 │               │
 │      ▼                 ▼               ▼
 │  Harness Layer     Git branches    .cache/runs/
 │  (src/harness/)    program/*       {hash}.json
 │      │             .claude/
 │      ▼             program.yaml
 │  Claude/OpenCode   skills/
 │  (future: Goose,
 │   OpenHands)
 │
 └── Scoring (src/evaluation/)
     └── "Is the answer correct?" → float between 0 and 1
```

---

## Two evolution modes explained

### skill_only (default, recommended)

The system prompt stays the same. The loop discovers new **skill files**.

Think of it like: "The agent's personality stays the same, but it learns new abilities."

A skill file is just a markdown file at `.claude/skills/{name}/SKILL.md` that teaches the agent a specific capability (like how to calculate percentages, or how to read financial tables).

### prompt_only

The skills stay the same. The loop rewrites the **system prompt**.

Think of it like: "The agent's abilities stay the same, but we change how it thinks."

The prompt at `src/agent_profiles/base_agent/prompt.txt` gets rewritten each iteration.

---

## Key numbers to know

| Setting                | Default | What it means                               |
| ---------------------- | ------- | ------------------------------------------- |
| `max_iterations`       | 20      | Maximum improvement attempts                |
| `frontier_size`        | 3       | Keep the top 3 versions                     |
| `no_improvement_limit` | 5       | Stop if 5 iterations in a row don't improve |
| `train_ratio`          | 0.18    | 18% of data for finding failures            |
| `val_ratio`            | 0.12    | 12% of data for testing improvements        |
| `concurrency`          | 4       | Run 4 evaluations in parallel               |
| Pass threshold         | 0.8     | Score below 80% = failure                   |
| Agent timeout          | 20 min  | Max time per agent call                     |
| Agent retries          | 3       | Retry failed API calls 3 times              |

---

## Summary: The one-paragraph version

EvoSkill takes a dataset of questions, splits it into training/validation sets, runs an AI agent on training questions to find failures, uses a "proposer" agent to analyze those failures and suggest a new skill, uses a "generator" agent to build that skill, tests the improved agent on validation questions, keeps it if the score went up (as a git branch), and repeats — all while caching results, tracking costs, and maintaining a "frontier" of the top 3 best versions.
