# Agent Profiles & Schemas

## How schemas, agent profiles, and the harness relate

- **`src/schemas/`** = the output templates. They define what fields each agent must fill out in its response.
- **`src/agent_profiles/`** = the job descriptions + config. They define who each agent is, what tools they get, and which schema they must use.
- **`src/harness/`** = the SDK execution layer. Agent profiles import from here to build options for the active SDK. See `docs/sdk-support.md` for details.

Every agent profile points to exactly one schema as its `response_model`.

---

## The 6 Schemas

### 1. `AgentResponse` — Base Agent output
```
final_answer: str    ← "The revenue was $4.2B"
reasoning: str       ← "I found this in Q3 report page 12..."
```
Used by ALL task agents (base, dabstep, sealqa, livecodebench).

### 2. `ProposerResponse` — Unified router (legacy)
```
optimize_prompt_or_skill: "prompt" or "skill"
proposed_skill_or_prompt: str
justification: str
```
Original "one proposer decides everything" approach. Still exists but newer code uses the specialized proposers.

### 3. `SkillProposerResponse` — Skill Proposer output
```
action: "create" or "edit"     ← make a new skill or fix an existing one?
target_skill: str | None       ← if editing, which skill?
proposed_skill: str            ← description of what to build/change
justification: str             ← why this addresses the failure
related_iterations: list[str]  ← e.g., ["iter-4", "iter-9"]
```
The richest schema. Tracks lineage so the proposer can learn from past failures.

### 4. `ToolGeneratorResponse` — Skill Generator output
```
generated_skill: str   ← the actual skill code/markdown
reasoning: str         ← why it was built this way
```

### 5. `PromptProposerResponse` — Prompt Proposer output
```
proposed_prompt_change: str   ← description of what to change
justification: str            ← why
```

### 6. `PromptGeneratorResponse` — Prompt Generator output
```
optimized_prompt: str   ← the full rewritten system prompt
reasoning: str          ← what was changed and why
```

---

## The Two Pipelines

```
Agent gets question wrong
         │
         ▼
    ┌─────────────────────────────────────────────────┐
    │  SKILL PATH (skill_only mode)                   │
    │                                                 │
    │  Skill Proposer ──► SkillProposerResponse       │
    │    "Create a percentage-calculator skill"        │
    │         │                                       │
    │         ▼                                       │
    │  Skill Generator ──► ToolGeneratorResponse      │
    │    writes .claude/skills/percentage-calculator/  │
    └─────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────┐
    │  PROMPT PATH (prompt_only mode)                 │
    │                                                 │
    │  Prompt Proposer ──► PromptProposerResponse     │
    │    "Agent should always verify calculations"    │
    │         │                                       │
    │         ▼                                       │
    │  Prompt Generator ──► PromptGeneratorResponse   │
    │    rewrites src/agent_profiles/base_agent/      │
    │    prompt.txt                                   │
    └─────────────────────────────────────────────────┘
```

---

## The 9 Agent Profiles

### "Worker" agents — full tools + write access

These answer benchmark questions. All use `AgentResponse` schema.

| Agent | Custom Prompt | Special Config |
|-------|--------------|----------------|
| **Base Agent** | `prompt.txt` ("You are an expert analyst...") | `permission_mode='acceptEdits'`, `max_buffer_size=10MB`, loads skills from disk via `setting_sources=["user", "project"]` |
| **DabStep Agent** | `prompt.txt` (file missing!) | Same as base but with single `data_dir` |
| **SealQA Agent** | `prompt.txt` (file missing!) | Same as base, no data_dirs |
| **LiveCodeBench Agent** | No custom prompt (uses Claude default) | Only agent supporting both Claude + OpenCode SDKs |

### "Meta" agents — limited tools, analyze/improve

| Agent | What it does | Key prompt rule | Tools |
|-------|-------------|-----------------|-------|
| **Proposer** (legacy) | Routes to skill vs prompt path | "If WHAT steps → skill. If HOW to think → prompt" | Read-only (8 tools) |
| **Skill Proposer** | Analyzes failures, proposes skills | "MUST use Brainstorming skill first. Check existing skills. Reference DISCARDED iterations." | Read-only (8 tools) |
| **Skill Generator** | Builds the actual skill file | "Read `.claude/skills/skill-creator/SKILL.md` first. Keep skills concise." | Full tools (11) + `permission_mode='acceptEdits'` |
| **Prompt Proposer** | Analyzes failures, proposes prompt changes | "Only propose if issue is about HOW to think, not WHAT steps" | Read-only (8 tools) |
| **Prompt Generator** | Rewrites the system prompt | "Anti-overfitting rules. Prompts guide HOW, not WHAT. Would this help 10 different tasks?" | Read-only (8 tools) |

---

## Tool Lists

**Full toolbox (11 tools)** — used by worker agents + skill generator:
```
Read, Write, Bash, Glob, Grep, Edit, WebFetch, WebSearch, TodoWrite, BashOutput, Skill
```

**Read-only toolbox (8 tools)** — used by proposers + prompt generator:
```
Read, Bash, Glob, Grep, WebFetch, WebSearch, TodoWrite, BashOutput
```

Key difference: no `Write`, `Edit`, or `Skill`. Proposers can only analyze, not modify files.

---

## Factory Patterns

All profile factories import from `src.harness` and branch on the active SDK:

```python
from src.harness import build_claudecode_options, build_opencode_options, is_claude_sdk

def get_my_agent_options(model=None, project_root=None):
    if is_claude_sdk():
        return build_claudecode_options(system=..., schema=..., tools=..., ...)
    return build_opencode_options(system=..., schema=..., tools=..., ...)
```

There are three calling patterns:

### 1. Lazy singleton (meta agents)
```python
# Evaluated once at import time using the active SDK
skill_proposer_options = get_skill_proposer_options()
```
Used by: skill_proposer, skill_generator, prompt_proposer, prompt_generator.

### 2. Direct factory function (task agents)
```python
def get_base_agent_options(model=None, data_dirs=None):
    prompt_text = PROMPT_FILE.read_text()  # reads from disk each time
    return _build_base_agent_options(prompt_text, model=model, data_dirs=data_dirs)
```
Used by: base_agent, dabstep_agent, sealqa_agent.

### 3. Factory-returning-factory (for Agent[T])
```python
def make_base_agent_options(model=None):
    def factory():
        return get_base_agent_options(model=model)
    return factory  # returns the function, not the result
```
This is passed to `Agent(options=factory, ...)`. Agent calls `factory()` on every `.run()`, giving fresh options each time.

---

## Key Design Details

1. **Proposers are read-only on purpose.** No Write or Edit tools. They should only analyze and propose.

2. **The Skill Generator gets the `Skill` tool.** It reads the `skill-creator` skill to learn the correct format before building a new one.

3. **Base Agent reads its prompt from disk at runtime.** Critical for `prompt_only` mode — Prompt Generator rewrites `prompt.txt`, next run picks it up automatically.

4. **Prompt Generator has anti-overfitting rules.** "BAD: `use np.std(ddof=1)`" vs "GOOD: choose methods appropriate for your sample type."

5. **Skill Proposer must use the Brainstorming skill.** Forces 2-3 alternative approaches before committing.

6. **`related_iterations` tracks lineage.** The proposer references past discarded attempts to avoid repeating mistakes.

7. **`setting_sources=["user", "project"]`** on worker agents is what makes discovered skills loadable from `.claude/skills/`.
