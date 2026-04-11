# SDK / Harness Support

## Architecture

SDK-specific code lives in `src/harness/`, completely separated from agent profiles:

```
src/harness/                         ← "how to talk to SDKs"
├── agent.py                           Agent class + AgentTrace + retry logic
├── sdk_config.py                      Global toggle: set_sdk("claude") / set_sdk("opencode")
├── options_utils.py                   build_claudecode_options() / build_opencode_options()
├── _claude_executor.py                Claude SDK execution + response parsing
└── _opencode_executor.py              OpenCode SDK server management + execution + parsing

src/agent_profiles/                  ← "what each agent role does"
├── base_agent/                        Imports from src.harness, knows nothing about SDKs
├── skill_proposer/                    Same
├── ...                                Same
```

Dependencies flow **one direction only**: `agent_profiles → harness`. The harness never imports from profiles. Adding a new harness (Goose, OpenHands) only touches `src/harness/`.

---

## Supported Harnesses

| Harness | Package | Models | How it runs | Status |
|---------|---------|--------|-------------|--------|
| **Claude Code** | `claude-agent-sdk` | Claude Opus, Sonnet, Haiku | Spawns Claude Code processes | Full support |
| **OpenCode** | `opencode-ai` | Any (Anthropic, DeepSeek, Gemini, etc.) | Local HTTP server per project | Full support |
| **Goose** | TBD | TBD | TBD | Planned |
| **OpenHands** | TBD | TBD | TBD | Planned |

---

## How the Harness Layer Works

### SDK Toggle

```python
from src.harness import set_sdk, is_claude_sdk

set_sdk("claude")    # use Claude Code
set_sdk("opencode")  # use OpenCode
```

Set automatically by the CLI from `.evoskill/config.toml`:
```toml
[harness]
name = "opencode"   # or "claude"
```

### Option Builders

Each profile factory calls the shared builders from `src/harness/options_utils.py`:

```python
from src.harness import build_claudecode_options, build_opencode_options, is_claude_sdk

def get_my_agent_options(model=None, project_root=None):
    if is_claude_sdk():
        return build_claudecode_options(
            system="...", schema=MyResponse.model_json_schema(),
            tools=[...], project_root=project_root, model=model,
        )
    return build_opencode_options(
        system="...", schema=MyResponse.model_json_schema(),
        tools=[...], project_root=project_root, model=model,
    )
```

### Executors

Each SDK has its own executor file with two functions:
- `execute_query(options, query)` → sends the query, returns raw messages
- `parse_response(messages, response_model)` → extracts fields into a dict for AgentTrace

The `Agent` class in `agent.py` delegates to the right executor based on `is_claude_sdk()`.

---

## Adding a New Harness

1. Create `src/harness/_goose_executor.py` with:
   - `execute_query(options, query) → list[Any]`
   - `parse_response(messages, response_model, ...) → dict`

2. Add `build_goose_options()` to `src/harness/options_utils.py`

3. Add the new SDK to `src/harness/sdk_config.py`

4. Add `elif` branches in `agent.py`'s `_execute_query()` and `run()` methods

5. No changes needed to any agent profile, evaluation, loop, cache, or CLI code.

---

## SDK-Agnostic Code (no changes needed when adding harnesses)

| Directory | Why it's safe |
|---|---|
| `src/agent_profiles/` | Imports from `src.harness`, doesn't know about SDKs directly |
| `src/schemas/` | Pure Pydantic models |
| `src/registry/manager.py` | Git operations only |
| `src/cache/run_cache.py` | Serializes AgentTrace which is SDK-neutral |
| `src/evaluation/` | Calls `agent.run()` which handles SDK branching internally |
| `src/loop/` | Uses the Agent abstraction |
| `src/api/` | High-level orchestration |

---

## Message Type Differences Between SDKs

### Claude SDK
Returns a list: `[SystemMessage, ..., ResultMessage]`
- First message: `.data.get("uuid")`, `.data.get("model")`, `.data.get("tools")`
- Last message: `.session_id`, `.duration_ms`, `.total_cost_usd`, `.num_turns`, `.usage`, `.result`, `.structured_output`

### OpenCode SDK
Returns a single `AssistantMessage`:
- `.info.get("structured_output")` or `.info.get("structured")` — structured output
- `.info.get("tokens", {})` — token usage
- `.info.get("cost", 0.0)` — cost
- `.parts[].get("text")` — response text
- `duration_ms` always 0, `num_turns` always 1 (not available from OpenCode)

Both are normalized into the same `AgentTrace` by their respective `parse_response()` functions.

---

## Smoke Testing

Use the smoke test script to verify a harness works end-to-end:

```bash
# Test Claude Code
./scripts/smoke_test.sh claude sonnet

# Test OpenCode
./scripts/smoke_test.sh opencode anthropic/claude-sonnet-4-6

# Future harnesses
./scripts/smoke_test.sh goose
./scripts/smoke_test.sh openhands
```

Runs 3 questions, 1 iteration, ~5-10 minutes. Tests the full pipeline: init → propose → generate → evaluate.
