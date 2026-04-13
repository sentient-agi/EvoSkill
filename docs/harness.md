# Harness Layer — Detailed Guide

The `src/harness/` package handles **how to talk to agent SDKs**. It knows nothing about what each agent does (that's `agent_profiles/`). This document explains every file, every function, every design decision.

## File Overview

```
src/harness/
├── __init__.py              — Package exports (what the outside world imports)
├── agent.py                 — Agent[T] + AgentTrace[T] (the public interface)
├── sdk_config.py            — Global SDK toggle ("claude" / "opencode")
├── options_utils.py         — Option builders + path/tool/permission helpers
├── _claude_executor.py      — Claude SDK: spawn process, parse response
└── _opencode_executor.py    — OpenCode SDK: manage server, send query, parse response
```

The underscore prefix on executor files means "internal" — they're called by `agent.py`, not imported directly by external code.

---

## `sdk_config.py` — The Global Toggle

The simplest file. A module-level variable that the rest of the system checks.

```python
_current_sdk: SDKType = "claude"     # default

set_sdk("opencode")      # switches globally
get_sdk()                 # → "opencode"
is_claude_sdk()           # → False
is_opencode_sdk()         # → True
```

**Who calls `set_sdk()`?**
- The CLI: `src/cli/commands/run.py` reads `config.toml` → `set_sdk(cfg.harness.name)`
- The scripts: `scripts/run_loop.py` passes `--sdk` arg
- Tests: `set_sdk("claude")` in fixtures

**Who checks it?**
- `options_utils.py:build_options()` — routes to the correct builder
- `agent.py:Agent._execute_query()` — picks the correct executor
- `agent.py:Agent.run()` — picks the correct response parser

**Why global state?** Because the SDK choice affects every agent in a run — you don't mix Claude and OpenCode in the same loop. A global toggle is simpler than threading a parameter through every function.

---

## `options_utils.py` — Building Agent Options

This file answers: "Given a system prompt, schema, and tools, how do I build the config object that the SDK needs?"

### `resolve_project_root(project_root=None) → Path`

Finds the repo root by walking up from `cwd` looking for `.evoskill/` or `.git/`. If `project_root` is provided, uses that directly.

```
/Users/me/work/EvoSkill/src/loop/runner.py
    → walks up → finds /Users/me/work/EvoSkill/.git
    → returns Path("/Users/me/work/EvoSkill")
```

Used by: every option builder (to set `cwd` for the agent), the CLI, the API.

### `resolve_data_dirs(project_root, data_dirs=None) → list[str]`

Converts relative data directory paths to absolute paths, resolved against the project root.

```python
resolve_data_dirs("/Users/me/EvoSkill", ["data/treasury"])
# → ["/Users/me/EvoSkill/data/treasury"]

resolve_data_dirs("/Users/me/EvoSkill", ["/absolute/path"])
# → ["/absolute/path"]  (already absolute, left alone)
```

### `build_options(*, system, schema, tools, ...) → Any`

**The main entry point that all 9 agent profiles call.** Routes to the correct SDK-specific builder based on `get_sdk()`.

```python
# What a profile calls:
build_options(
    system="You are an expert analyst...",
    schema=AgentResponse.model_json_schema(),
    tools=["Read", "Write", "Bash", ...],
    model="sonnet",
    setting_sources=["user", "project"],   # Claude-specific, ignored on OpenCode
    permission_mode="acceptEdits",          # Claude-specific, ignored on OpenCode
)

# Internally:
if sdk == "claude":  → build_claudecode_options(system=..., ..., setting_sources=..., permission_mode=...)
if sdk == "opencode": → build_opencode_options(system=..., ...)  # extras dropped
```

**Why Claude-specific extras are silently ignored:** OpenCode has no concept of `permission_mode` or `setting_sources`. Rather than forcing every profile to know this, `build_options` accepts them always and only forwards them when relevant.

### `build_claudecode_options(*, system, schema, tools, ...) → ClaudeAgentOptions`

Builds the `ClaudeAgentOptions` object that `claude-agent-sdk` needs:

```python
ClaudeAgentOptions(
    system_prompt = {"type": "preset", "preset": "claude_code", "append": system},
    output_format = {"type": "json_schema", "schema": schema},
    allowed_tools = ["Read", "Write", "Bash", ...],
    cwd = "/Users/me/EvoSkill",
    # Optional (only passed when not None):
    setting_sources = ["user", "project"],
    permission_mode = "acceptEdits",
    max_buffer_size = 10485760,
    add_dirs = ["/path/to/data"],
)
```

Key details:
- `system_prompt` always uses the `"claude_code"` preset as a base. The custom prompt is appended via `"append"`. If `system` is empty string (livecodebench), no `"append"` key is added.
- `ClaudeAgentOptions` is imported **inside the function**, not at the top of the file. This means the module loads even if `claude-agent-sdk` isn't installed.
- Optional kwargs are only passed when not `None` — avoids sending defaults that might override SDK behavior.
- `model` is set via `options.model = model` after construction (the SDK API requires this).

### `build_opencode_options(*, system, schema, tools, ...) → dict`

Builds a plain dict for the OpenCode SDK:

```python
{
    "system": "You are an expert analyst...\n\nAdditional accessible data directories...",
    "format": {"type": "json_schema", "schema": {...}},
    "tools": {"read": True, "write": True, "bash": True, ...},
    "mode": "build",
    "provider_id": "anthropic",
    "model_id": "claude-sonnet-4-6",
    "cwd": "/Users/me/EvoSkill",
    "add_dirs": ["/path/to/data"],
}
```

Key details:
- **Tool name mapping**: Claude uses PascalCase (`"Read"`), OpenCode uses lowercase (`"read"`). The `to_opencode_tools()` helper converts. `"BashOutput"` maps to `None` (not supported by OpenCode).
- **Model string splitting**: `"anthropic/claude-sonnet-4-6"` → `provider_id="anthropic"`, `model_id="claude-sonnet-4-6"`. Done by `split_opencode_model()`.
- **Data dir injection**: If `data_dirs` are provided, they're appended to the system prompt as a note (since OpenCode doesn't have a native `add_dirs` mechanism for prompt visibility).
- **Permission auto-config**: `ensure_opencode_project_permissions()` writes an `opencode.json` file in the project root that grants read access to the data directories. Without this, OpenCode refuses to read files outside the project.

### `ensure_opencode_project_permissions(project_root, data_dirs=None)`

Auto-creates/updates `opencode.json` to allow file access:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "permission": {
    "external_directory": {
      "/path/to/data": "allow",
      "/path/to/data/**": "allow"
    }
  }
}
```

Only runs if:
- There are data_dirs to configure
- There isn't already an `opencode.jsonc` (user's manual config takes priority)
- The permissions actually need updating

### `to_opencode_tools(tools) → dict[str, bool]`

Maps Claude tool names to OpenCode equivalents:

```python
to_opencode_tools(["Read", "Write", "BashOutput", "Skill"])
# → {"read": True, "write": True, "skill": True}
# Note: BashOutput → None (dropped)
```

### `split_opencode_model(model) → (provider_id, model_id)`

```python
split_opencode_model("anthropic/claude-sonnet-4-6")
# → ("anthropic", "claude-sonnet-4-6")

split_opencode_model("sonnet")
# → ("anthropic", "sonnet")  # default provider

split_opencode_model(None)
# → ("anthropic", "claude-sonnet-4-6")  # default model
```

---

## `agent.py` — The Public Interface

This file defines two things everyone uses: `AgentTrace[T]` (what comes out) and `Agent[T]` (what goes in).

### `OptionsProvider` type

What you can pass to `Agent()` as `options`:

```python
# Option 1: Static ClaudeAgentOptions object
Agent(options=my_claude_options, response_model=AgentResponse)

# Option 2: Static dict (OpenCode)
Agent(options={"system": "...", "tools": {...}}, response_model=AgentResponse)

# Option 3: Factory function (called fresh on every .run())
Agent(options=lambda: build_options(system=...), response_model=AgentResponse)
```

The factory pattern (Option 3) is used by `base_agent` so it re-reads `prompt.txt` from disk on every run — critical for `prompt_only` evolution mode where the Prompt Generator rewrites the file between iterations.

### `AgentTrace[T]` — The Result Object

Every `.run()` call returns one of these:

```
Identity:
  uuid         — unique run ID (from Claude first message, or session_id for OpenCode)
  session_id   — session identifier
  model        — e.g., "claude-sonnet-4-6" (from Claude) or from options (OpenCode)
  tools        — tool list (from Claude first message or options)

Metrics:
  duration_ms    — how long the run took (0 for OpenCode — not reported)
  total_cost_usd — API cost
  num_turns      — how many tool calls the agent made (1 for OpenCode — not reported)
  usage          — token counts dict

Result:
  result          — raw text response
  is_error        — True if the agent errored or structured output parsing failed

Structured Output:
  output: T | None        — the parsed Pydantic model (e.g., AgentResponse with final_answer)
  parse_error: str | None — why parsing failed (e.g., "ValidationError: ...")
  raw_structured_output   — the raw dict before Pydantic validation

Debug:
  messages        — full message list from the SDK (for debugging)
```

The `summarize()` method creates a text version for the proposer agent to read. On success, includes the full trace. On failure, truncates to head + tail (default 60k chars each) to avoid blowing up the proposer's context window.

### `Agent[T]` — The Wrapper

```python
agent = Agent(options=my_factory, response_model=AgentResponse)
trace = await agent.run("What was US defense spending in 1940?")
```

**`__init__(options, response_model)`** — Stores the options provider and the Pydantic model to validate output against.

**`_get_options()`** — If `options` is a callable, calls it. Otherwise returns it directly. This is what makes the factory pattern work.

**`_execute_query(query)`** — The SDK dispatch point:

```python
if is_claude_sdk():
    from . import _claude_executor
    return await _claude_executor.execute_query(options, query)
else:
    from . import _opencode_executor
    return await _opencode_executor.execute_query(options, query)
```

When you add Goose/OpenHands, you add `elif` branches here.

**`_run_with_retry(query)`** — Wraps `_execute_query` with resilience:

```
Attempt 1: run query with 20-minute timeout
  → success? return messages
  → timeout? wait 30s, try again
  → exception? wait 30s, try again

Attempt 2: same, but wait 60s on failure

Attempt 3: same, but wait 120s on failure

All failed? raise the last error
```

**`run(query) → AgentTrace[T]`** — The main entry point:

```python
messages = await self._run_with_retry(query)    # get raw SDK messages

if is_claude_sdk():
    fields = _claude_executor.parse_response(messages, self.response_model)
else:
    fields = _opencode_executor.parse_response(messages, self.response_model, self._get_options)

return AgentTrace(**fields)    # construct the SDK-agnostic result
```

---

## `_claude_executor.py` — Claude SDK Specifics

Two functions, both called only by `agent.py`.

### `execute_query(options, query) → list[messages]`

1. Import `ClaudeSDKClient` (inside function, not at top)
2. If `options` is a dict, convert to `ClaudeAgentOptions` (fallback for registry path)
3. Create `ClaudeSDKClient(options)` — this spawns a Claude Code process
4. `client.query(query)` — send the question
5. `client.receive_response()` — async iterate messages as they stream back
6. Return all messages as a list: `[SystemMessage, AssistantMessage, ..., ResultMessage]`

### `parse_response(messages, response_model) → dict`

Extracts AgentTrace fields from Claude's message format:

```
messages[0] (SystemMessage):
  .data["uuid"]   → uuid
  .data["model"]  → model
  .data["tools"]  → tools

messages[-1] (ResultMessage):
  .session_id          → session_id
  .duration_ms         → duration_ms
  .total_cost_usd      → total_cost_usd
  .num_turns           → num_turns
  .usage               → usage
  .result              → result
  .is_error            → is_error
  .structured_output   → raw_structured_output → validate → output
```

The structured output validation:
- If `structured_output` is not None → try `response_model.model_validate(it)`
  - Success → `output = parsed AgentResponse`
  - Failure → `parse_error = "ValidationError: ..."`
- If `structured_output` is None → `parse_error = "No structured output returned (context limit likely exceeded)"`

Returns a dict that `Agent.run()` unpacks into `AgentTrace(**fields)`.

---

## `_opencode_executor.py` — OpenCode SDK Specifics

More complex than Claude because it manages a server lifecycle.

### Server Management

OpenCode runs as a **local HTTP server** (unlike Claude which spawns a process per query).

**`_SERVER_PORTS: dict[str, int]`** — Module-level dict tracking which port each project uses:
```python
{"/Users/me/project-a": 54321, "/Users/me/project-b": 54322}
```

**`_find_free_port()`** — Asks the OS for a random available port:
```python
sock.bind(("127.0.0.1", 0))  # port 0 = "give me any free port"
return sock.getsockname()[1]  # e.g., 54321
```

**`_server_matches_project(client, expected_cwd)`** — Asks a running server "what directory are you in?":
```python
app_info = await client.app.get()
# Check if app_info.path.cwd == expected_cwd
```
Returns False if server is unreachable or serving a different project.

**`_ensure_server(options)`** — The main server lifecycle function:
```
1. Get requested_cwd from options
2. Look up port for this project (default 4096)
3. Create client at that port
4. Ask server: "are you serving my project?"
   YES → return client (reuse existing server)
   NO →
     a. Find a free port
     b. Start: opencode serve --port {port} --hostname 127.0.0.1
     c. Wait 2 seconds for startup
     d. Create new client
     e. Return client
```

### `execute_query(options, query) → list[message]`

1. Validate options is a dict (OpenCode requires it)
2. `_ensure_server(options)` — get a connected client
3. `client.session.create()` — create a new session
4. `client.session.chat(...)` — send the query with:
   - `model_id` / `provider_id` — which LLM to use
   - `parts` — the query text
   - `system` — system prompt
   - `mode` — "build" (default)
   - `tools` — available tools
   - `extra_body` — structured output format (if configured)
5. Return `[message]` — single message wrapped in list for consistency with Claude

### `parse_response(messages, response_model, get_options) → dict`

Extracts AgentTrace fields from OpenCode's message format:

```
message.info:
  .get("structured_output")  → raw_structured_output (try this first)
  .get("structured")         → raw_structured_output (fallback for older versions)
  .get("tokens", {})         → usage
  .get("cost", 0.0)          → total_cost_usd

message.parts:
  [{type: "text", text: "..."}]  → concatenated into result_text

message.session_id  → uuid, session_id
```

**Key difference from Claude**: OpenCode doesn't return `model` or `tools` in the response. So `parse_response` takes a `get_options` callable and pulls them from the options that were sent:

```python
options = get_options()
model_name = options.get("model_id", "unknown")
tools = list(options.get("tools", {}).keys())
```

Also: `duration_ms=0` and `num_turns=1` — OpenCode doesn't report these.

---

## How It All Flows Together

```
Profile factory calls build_options(system=..., schema=..., tools=...)
    │
    ▼
build_options() checks get_sdk()
    │
    ├── "claude" → build_claudecode_options() → ClaudeAgentOptions
    └── "opencode" → build_opencode_options() → dict
    │
    ▼
Agent(options=result, response_model=AgentResponse)
    │
    ▼
agent.run("What was defense spending?")
    │
    ▼
agent._run_with_retry(query)    ← 3 attempts, 20-min timeout, exponential backoff
    │
    ▼
agent._execute_query(query)     ← checks is_claude_sdk()
    │
    ├── Claude: _claude_executor.execute_query(options, query)
    │     └── ClaudeSDKClient(options).query(query).receive_response()
    │     └── returns [SystemMessage, ..., ResultMessage]
    │
    └── OpenCode: _opencode_executor.execute_query(options, query)
          └── _ensure_server(options) → client
          └── client.session.create() → session
          └── client.session.chat(session.id, ...) → message
          └── returns [AssistantMessage]
    │
    ▼
agent.run() parses response      ← checks is_claude_sdk() again
    │
    ├── Claude: _claude_executor.parse_response(messages, response_model)
    │     └── extracts from SystemMessage + ResultMessage
    │
    └── OpenCode: _opencode_executor.parse_response(messages, response_model, get_options)
          └── extracts from AssistantMessage.info + .parts
    │
    ▼
AgentTrace(**fields)              ← SDK-agnostic result
    │
    ▼
Returned to caller (loop, evaluation, etc.)
```

---

## Adding a New Harness (e.g., Goose)

1. **Create `_goose_executor.py`** with:
   - `execute_query(options, query) → list[Any]`
   - `parse_response(messages, response_model, ...) → dict`

2. **Add `build_goose_options()` to `options_utils.py`**

3. **Add "goose" to `sdk_config.py`**: extend `SDKType` and the `set_sdk` validation

4. **Add branches in `options_utils.py:build_options()`**:
   ```python
   if sdk == "goose":
       return build_goose_options(...)
   ```

5. **Add branches in `agent.py`**:
   ```python
   # In _execute_query:
   elif is_goose_sdk():
       from . import _goose_executor
       return await _goose_executor.execute_query(options, query)

   # In run:
   elif is_goose_sdk():
       from . import _goose_executor
       fields = _goose_executor.parse_response(messages, self.response_model, ...)
   ```

6. **Zero changes to agent profiles, loop, evaluation, cache, CLI, or API.**
