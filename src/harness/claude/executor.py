"""Claude SDK execution and response parsing.

Handles Claude-specific logic:
    - Converting dict options to ClaudeAgentOptions (fallback)
    - Spawning ClaudeSDKClient and streaming messages with live OTel spans
    - Printing turn-by-turn progress to stdout
    - Parsing SystemMessage/ResultMessage into AgentTrace fields
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Type

from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace
from opentelemetry.trace import set_span_in_context
from pydantic import BaseModel, ValidationError

from src.harness.utils import (
    eval_run_label as _eval_run_label,
    eval_run_uid as _eval_run_uid,
    eval_run_index as _eval_run_index,
    eval_run_ground_truth as _eval_run_ground_truth,
)


_tracer = otel_trace.get_tracer("evoskill.harness.claude")


def _print_subagent_to_stdout(agent_id: str, run_tag: str, parent_turn: int) -> int:
    """Replay a subagent's turn-by-turn tool activity to stdout so the live
    `tail -f` log shows what the subagent did (otherwise the parent only
    prints "Task" once and the subagent's 40+ tool calls are silent).

    Format pairs each subagent turn with the parent turn that spawned it:
        [<run_tag>] turn.A/sub.B [<model>]: <Tool>
    where A is the parent's turn number at Task dispatch, B is the subagent's
    internal turn number. The slash mirrors OTel span naming convention.
    """
    matches = list(Path.home().glob(f".claude/projects/*/*/subagents/agent-{agent_id}.jsonl"))
    if not matches:
        return 0
    sub_jsonl = matches[0]
    try:
        events = [json.loads(line) for line in sub_jsonl.read_text().splitlines() if line.strip()]
    except Exception:
        return 0
    n = 0
    for ev in events:
        if ev.get("type") != "assistant":
            continue
        msg = ev.get("message", {})
        content = msg.get("content", []) or []
        if not isinstance(content, list):
            continue
        n += 1
        # Per-turn subagent console prints removed — too noisy with
        # concurrent runs; full detail still in Phoenix OTel spans.
    return n


def _emit_subagent_spans(agent_id: str, parent_span: Any) -> int:
    """Emit nested OTel spans for a subagent's session.

    The bundled Claude binary writes subagent transcripts to
    ~/.claude/projects/<project>/<parent-session>/subagents/agent-<id>.jsonl.
    The parent's stream only sees the Task tool_result (a summary string),
    so without this expansion the subagent is a Phoenix black box.

    For each AssistantMessage in the subagent's JSONL we emit:
        subagent/turn.N         — child of parent_span (the tool.Task span)
        subagent/tool.<name>    — child of the corresponding turn span

    Tool results within subagent turns are correlated by tool_use_id and
    attached to the matching tool span as `tool.output` before closing.

    Returns the number of subagent assistant turns emitted.
    """
    matches = list(Path.home().glob(f".claude/projects/*/*/subagents/agent-{agent_id}.jsonl"))
    if not matches:
        return 0
    sub_jsonl = matches[0]
    try:
        events = [json.loads(line) for line in sub_jsonl.read_text().splitlines() if line.strip()]
    except Exception:
        return 0

    parent_ctx = set_span_in_context(parent_span)
    parent_span.set_attribute("subagent.id", agent_id)
    parent_span.set_attribute("subagent.jsonl", str(sub_jsonl))

    pending_tool_spans: dict[str, Any] = {}
    turn_n = 0
    last_model: str | None = None

    for ev in events:
        t = ev.get("type")
        if t == "assistant":
            msg = ev.get("message", {})
            content = msg.get("content", []) or []
            model = msg.get("model", "") or ""
            if last_model is None and model:
                last_model = model
                parent_span.set_attribute("subagent.model", model)
            turn_n += 1
            t_span = _tracer.start_span(f"subagent/turn.{turn_n}", context=parent_ctx)
            t_span.set_attribute("openinference.span.kind", "CHAIN")
            t_span.set_attribute("turn", turn_n)
            t_span.set_attribute("model", model)
            if model:
                # Match parent-side semantics so Phoenix can price subagent turns.
                t_span.set_attribute("llm.model_name", model)
            t_span.set_attribute("subagent.id", agent_id)
            t_ctx = set_span_in_context(t_span)
            for blk in content:
                if not isinstance(blk, dict):
                    continue
                bt = blk.get("type")
                if bt == "thinking":
                    th_text = (blk.get("thinking", "") or "").strip()
                    if th_text:
                        t_span.set_attribute("turn.thinking", th_text[:5000])
                elif bt == "text":
                    tx = (blk.get("text", "") or "").strip()
                    if tx:
                        t_span.set_attribute("turn.text", tx[:5000])
                elif bt == "tool_use":
                    name = blk.get("name", "?") or "?"
                    inp = blk.get("input", {})
                    try:
                        inp_json = json.dumps(inp, default=str, ensure_ascii=False)
                    except Exception:
                        inp_json = str(inp)[:2000]
                    tspan = _tracer.start_span(f"subagent/tool.{name}", context=t_ctx)
                    tspan.set_attribute("openinference.span.kind", "TOOL")
                    tspan.set_attribute("tool.name", name)
                    tspan.set_attribute("tool.input", inp_json[:8000])
                    tspan.set_attribute("input.value", inp_json[:8000])
                    tspan.set_attribute("input.mime_type", "application/json")
                    tu_id = blk.get("id") or blk.get("tool_use_id")
                    if tu_id:
                        pending_tool_spans[str(tu_id)] = tspan
                    else:
                        tspan.end()
            t_span.end()
        elif t == "user":
            msg = ev.get("message", {})
            content = msg.get("content", []) or []
            if not isinstance(content, list):
                continue
            for blk in content:
                if not isinstance(blk, dict):
                    continue
                if blk.get("type") != "tool_result":
                    continue
                tu_id = blk.get("tool_use_id") or blk.get("id")
                tspan = pending_tool_spans.pop(str(tu_id), None) if tu_id else None
                if tspan is None:
                    continue
                rc = blk.get("content", "")
                if isinstance(rc, list):
                    rs = "\n".join(c.get("text", str(c)) if isinstance(c, dict) else str(c) for c in rc)
                else:
                    rs = str(rc) if rc is not None else ""
                tspan.set_attribute("tool.output", rs[:8000])
                tspan.set_attribute("output.value", rs[:8000])
                tspan.set_attribute("output.mime_type", "text/plain")
                tspan.set_attribute("tool.is_error", bool(blk.get("is_error", False)))
                tspan.end()

    # Close any tool spans whose result never arrived
    for _id, _sp in list(pending_tool_spans.items()):
        _sp.set_attribute("tool.output", "[no result]")
        _sp.end()

    parent_span.set_attribute("subagent.num_turns", turn_n)
    return turn_n


# USD per million tokens. Source: platform.claude.com/docs/en/about-claude/pricing
# Tuple order: (input, output, 5m_cache_write, cache_read). Starting with
# Opus 4.5, Anthropic dropped Opus pricing 3× vs Opus 4 / 4.1.
# Prefix-match on the model_name string. Missing models fall back to Sonnet.
_MODEL_PRICING_USD_PER_MTOK = {
    # (input, output, cache_write, cache_read) — all 5-minute TTL cache writes
    "claude-opus-4-7":      (5.00, 25.00, 6.25, 0.50),
    "claude-opus-4-6":      (5.00, 25.00, 6.25, 0.50),
    "claude-opus-4-5":      (5.00, 25.00, 6.25, 0.50),
    "claude-opus-4-1":      (15.00, 75.00, 18.75, 1.50),
    "claude-opus-4":        (15.00, 75.00, 18.75, 1.50),
    "claude-sonnet-4":      (3.00, 15.00, 3.75, 0.30),  # covers 4, 4.5, 4.6
    "claude-haiku-4":       (1.00, 5.00, 1.25, 0.10),   # covers 4.5
    "claude-3-5-sonnet":    (3.00, 15.00, 3.75, 0.30),
    "claude-3-5-haiku":     (0.80, 4.00, 1.00, 0.08),
}
_DEFAULT_PRICING = (3.00, 15.00, 3.75, 0.30)  # Sonnet-equivalent fallback


def _price_for_model(model: str) -> tuple[float, float, float, float]:
    if not model:
        return _DEFAULT_PRICING
    for prefix, price in _MODEL_PRICING_USD_PER_MTOK.items():
        if model.startswith(prefix):
            return price
    return _DEFAULT_PRICING


def _compute_cost_usd(model: str, usage: dict) -> tuple[float, int, int, int, int]:
    """Return (cost_usd, prompt_tok, completion_tok, cache_read_tok, cache_write_tok)."""
    if not usage:
        return 0.0, 0, 0, 0, 0
    in_tok = int(usage.get("input_tokens", 0) or 0)
    out_tok = int(usage.get("output_tokens", 0) or 0)
    cache_read = int(usage.get("cache_read_input_tokens", usage.get("cache_read", 0)) or 0)
    cache_write = int(usage.get("cache_creation_input_tokens", usage.get("cache_creation", 0)) or 0)
    p_in, p_out, p_cw, p_cr = _price_for_model(model)
    cost = (
        in_tok * p_in +
        out_tok * p_out +
        cache_write * p_cw +
        cache_read * p_cr
    ) / 1_000_000
    return cost, in_tok, out_tok, cache_read, cache_write


def _preview(text: Any, n: int = 80) -> str:
    """Safe one-line preview of a block for terminal/span display."""
    s = str(text).strip().replace("\n", " ")
    return s[:n] + "…" if len(s) > n else s


def _truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"\n[... {len(s) - max_chars} more chars truncated ...]"


def _stringify_tool_result(content: Any) -> str:
    """ToolResultBlock.content can be a string, a list of dicts, or a list of
    SDK content blocks. Normalize to a single string so it lands in a Phoenix
    `message.content` attribute."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for it in content:
            if isinstance(it, dict):
                parts.append(it.get("text") or str(it))
            else:
                parts.append(str(getattr(it, "text", it)))
        return "\n".join(parts)
    return "" if content is None else str(content)


def _emit_blocks_as_messages(
    span: Any,
    blocks: list,
    attr_prefix: str,
    start_idx: int,
    *,
    TextBlock: type,
    ThinkingBlock: type | None,
    ToolUseBlock: type,
    max_chars_per_msg: int,
) -> int:
    """Convert a sequence of SDK assistant content blocks into one or more
    OpenInference messages.

    - `ThinkingBlock` → its own message with `role="thinking"` (Phoenix
      renders the role string verbatim in the message header, so reasoning
      gets a distinct visual section instead of a `[thinking]` prefix
      inside an assistant message).
    - `TextBlock` + `ToolUseBlock` (in order) → grouped into a single
      `role="assistant"` message with `content` = concatenated text and
      `tool_calls` = the tool_use blocks.
    - A new ThinkingBlock encountered mid-stream flushes any pending
      assistant content first so message order matches the model's
      original block order.

    Returns the next available message index after the emitted messages.
    """
    idx = start_idx
    pending_text: list[str] = []
    pending_tools: list[dict[str, str]] = []

    def _flush_assistant() -> None:
        nonlocal idx, pending_text, pending_tools
        if not pending_text and not pending_tools:
            return
        span.set_attribute(f"{attr_prefix}.{idx}.message.role", "assistant")
        joined = "\n".join(p for p in pending_text if p)
        if joined:
            span.set_attribute(
                f"{attr_prefix}.{idx}.message.content",
                _truncate(joined, max_chars_per_msg),
            )
        for j, tc in enumerate(pending_tools):
            base = f"{attr_prefix}.{idx}.message.tool_calls.{j}.tool_call"
            span.set_attribute(f"{base}.id", tc["id"])
            span.set_attribute(f"{base}.function.name", tc["name"])
            span.set_attribute(f"{base}.function.arguments", tc["args"])
        idx += 1
        pending_text = []
        pending_tools = []

    for block in blocks or []:
        if isinstance(block, TextBlock):
            t = getattr(block, "text", "") or ""
            if t:
                pending_text.append(t)
        elif ThinkingBlock is not None and isinstance(block, ThinkingBlock):
            # Flush so this thinking message lands AFTER any earlier text
            # in original order, not lumped together.
            _flush_assistant()
            th = getattr(block, "thinking", "") or ""
            if th:
                span.set_attribute(f"{attr_prefix}.{idx}.message.role", "thinking")
                span.set_attribute(
                    f"{attr_prefix}.{idx}.message.content",
                    _truncate(th, max_chars_per_msg),
                )
                idx += 1
        elif isinstance(block, ToolUseBlock):
            try:
                args_str = json.dumps(
                    getattr(block, "input", None) or {},
                    default=str,
                    ensure_ascii=False,
                )
            except Exception:
                args_str = str(getattr(block, "input", ""))
            pending_tools.append({
                "id": getattr(block, "id", "") or "",
                "name": getattr(block, "name", "") or "",
                "args": _truncate(args_str, max_chars_per_msg),
            })

    _flush_assistant()
    return idx


def _emit_llm_messages_for_turn(
    span: Any,
    sys_prompt_text: str | None,
    query: str,
    prior_messages: list,
    current_msg: Any,
    *,
    AssistantMessage: type,
    UserMessage: type,
    TextBlock: type,
    ThinkingBlock: type | None,
    ToolUseBlock: type,
    ToolResultBlock: type,
    max_chars_per_msg: int = 5000,
    max_chars_per_tool_result: int = 20_000,
) -> None:
    """Emit OpenInference `llm.input_messages.*` and `llm.output_messages.*`
    attrs reconstructing the conversation the model saw on this turn.

    `prior_messages` should be all SDK messages that arrived BEFORE
    `current_msg` (in order). Per-message content is truncated to keep
    Phoenix attribute size manageable even on hundred-turn runs. Tool
    results get a larger budget than text/thinking because they're where
    the actual evidence (Read output, Grep matches, Bash stdout) lives —
    aggressive truncation there hurts trace readability the most.
    """
    msg_idx = 0

    # 1) System
    if sys_prompt_text:
        span.set_attribute(f"llm.input_messages.{msg_idx}.message.role", "system")
        span.set_attribute(
            f"llm.input_messages.{msg_idx}.message.content",
            _truncate(sys_prompt_text, max_chars_per_msg),
        )
        msg_idx += 1

    # 2) Original user query
    span.set_attribute(f"llm.input_messages.{msg_idx}.message.role", "user")
    span.set_attribute(
        f"llm.input_messages.{msg_idx}.message.content",
        _truncate(query, max_chars_per_msg),
    )
    msg_idx += 1

    # 3) Replay prior turns
    for prev in prior_messages:
        if isinstance(prev, AssistantMessage):
            msg_idx = _emit_blocks_as_messages(
                span,
                getattr(prev, "content", []) or [],
                "llm.input_messages",
                msg_idx,
                TextBlock=TextBlock,
                ThinkingBlock=ThinkingBlock,
                ToolUseBlock=ToolUseBlock,
                max_chars_per_msg=max_chars_per_msg,
            )
        elif isinstance(prev, UserMessage):
            for block in getattr(prev, "content", []) or []:
                if isinstance(block, ToolResultBlock):
                    result_str = _stringify_tool_result(getattr(block, "content", None))
                    span.set_attribute(
                        f"llm.input_messages.{msg_idx}.message.role", "tool",
                    )
                    span.set_attribute(
                        f"llm.input_messages.{msg_idx}.message.tool_call_id",
                        getattr(block, "tool_use_id", "") or "",
                    )
                    if result_str:
                        span.set_attribute(
                            f"llm.input_messages.{msg_idx}.message.content",
                            _truncate(result_str, max_chars_per_tool_result),
                        )
                    msg_idx += 1

    # 4) Output: this turn's blocks split into per-block messages, same
    # convention as input — thinking gets its own message header, text +
    # tool_calls group together as assistant.
    _emit_blocks_as_messages(
        span,
        getattr(current_msg, "content", []) or [],
        "llm.output_messages",
        0,
        TextBlock=TextBlock,
        ThinkingBlock=ThinkingBlock,
        ToolUseBlock=ToolUseBlock,
        max_chars_per_msg=max_chars_per_msg,
    )


async def execute_query(
    options: Any,
    query: str,
    *,
    agent_name: str = "agent",
    tag: str | None = None,
) -> list[Any]:
    """Execute a query via Claude SDK, streaming messages with live OTel spans.

    Args:
        options: ClaudeAgentOptions or dict (auto-converted)
        query: The question to send to the agent
        agent_name: Name prefix for spans and console logs (e.g., "base", "skill_evolver")
        tag: Optional human-readable label for stdout prints and span names
             (e.g., "train 3/8", "val 2/8"). Defaults to "q:<sha6>" when omitted.

    Returns:
        List of messages: [SystemMessage, ..., ResultMessage]
    """
    from claude_agent_sdk import (
        ClaudeAgentOptions, ClaudeSDKClient,
        AssistantMessage, UserMessage, ToolUseBlock, ToolResultBlock, TextBlock,
    )
    # ThinkingBlock may not exist in older SDKs
    try:
        from claude_agent_sdk import ThinkingBlock
    except ImportError:
        ThinkingBlock = None  # type: ignore

    # If someone passed a dict to Claude SDK (e.g., from config_to_options),
    # convert it to ClaudeAgentOptions.
    if isinstance(options, dict):
        claude_opts = ClaudeAgentOptions(
            system_prompt=options.get("system"),
            allowed_tools=list(options.get("tools", {}).keys())
            if options.get("tools")
            else [],
            output_format=options.get("format"),
            setting_sources=["user", "project"],
            permission_mode="acceptEdits",
        )
        if "model_id" in options and "claude" in options["model_id"].lower():
            claude_opts.model = options["model_id"]
        options = claude_opts

    # Tighter per-API-request timeout + retry budget for the bundled CLI.
    # Default API_TIMEOUT_MS=600000 (10min) means a single hung HTTP call
    # blocks the whole agent run for 10 minutes before the CLI's internal
    # retry kicks in. Cap at 120s so transient network blips fail fast and
    # the CLI's exponential-backoff retry loop reissues the call against
    # the existing conversation state — no whole-run restart needed.
    # Caller-provided env wins.
    if not getattr(options, "env", None):
        options.env = {}
    options.env.setdefault("API_TIMEOUT_MS", "120000")
    options.env.setdefault("CLAUDE_CODE_MAX_RETRIES", "5")

    # Adaptive wall-clock timeout: forward each stderr line to agent.py's
    # observer, which detects "(attempt N/K)" banners from the bundled CLI
    # and extends the asyncio.timeout deadline by +120s per new retry. This
    # gives the agent back the time consumed by in-CLI retry without us
    # having to budget the worst-case (22min) up front.
    from src.harness.agent import _get_run_stderr_observer
    _prev_stderr = getattr(options, "stderr", None)

    def _stderr_with_retry_detect(line: str) -> None:
        try:
            obs = _get_run_stderr_observer()
            if obs is not None:
                obs(line)
        except Exception:
            pass
        if _prev_stderr is not None:
            try:
                _prev_stderr(line)
            except Exception:
                pass

    options.stderr = _stderr_with_retry_detect

    messages: list[Any] = []
    model_display = getattr(options, "model", None) or "claude"
    # Canonical model name from the SDK response (e.g. "claude-opus-4-6").
    # User-provided aliases like "opus" don't match Phoenix's pricing
    # manifest patterns, so cost shows as $0 in the UI. Capture the real
    # API-returned name from the first AssistantMessage and use it for
    # `llm.model_name` so Phoenix's pricing lookup succeeds.
    canonical_model: str | None = None
    turn_num = 0
    # Map tool_use_id -> span so we can attach results from later UserMessages
    pending_tool_spans: dict[str, Any] = {}
    # Track the most recent Skill tool span across UserMessages — the injected
    # SKILL.md text block ("Base directory for this skill: ...") arrives in a
    # SEPARATE UserMessage from the "Launching skill:" tool_result, so we
    # need a long-lived reference. Cleared once we attach content (or at end).
    last_skill_span: Any = None

    # Per-query tag for disambiguating interleaved stdout when asyncio.gather
    # runs several solvers concurrently. Prefer the caller's human-readable
    # label (e.g. "train 3/8"); fall back to a sha-based id so every trace
    # still has some unique marker.
    query_hash = "q:" + hashlib.sha256(query.encode("utf-8", errors="replace")).hexdigest()[:6]
    run_tag = tag or query_hash

    # Detach parent OTel context so each solver run is its OWN root trace in
    # Phoenix. Concurrent gather'd runs would otherwise nest under a shared
    # parent and appear visually entangled in one tree; as separate roots
    # they show as N independent traces — one per question.
    parent_span = otel_trace.get_current_span()
    parent_ctx = parent_span.get_span_context() if parent_span else None
    # Pull caller-provided eval metadata (set by evaluate_full when running
    # through `evoskill eval` / scripts/run_eval.py). Read BEFORE detaching
    # context — contextvars survive context.attach but reading first keeps
    # the dependency obvious. None when called outside an eval context.
    _eval_label = _eval_run_label.get()
    _eval_uid = _eval_run_uid.get()
    _eval_idx = _eval_run_index.get()
    _eval_gt = _eval_run_ground_truth.get()
    # Iter prefix (e.g., "iter 3") to prepend to span names so Phoenix
    # traces are scannable across iterations. Set by the runner at the
    # top of each iter; None during base eval / outside the iter loop.
    from src.harness.utils import eval_iter_label as _eval_iter_label
    _iter_prefix = _eval_iter_label.get()
    _iter_pfx_str = f"{_iter_prefix} | " if _iter_prefix else ""
    _ctx_token = otel_context.attach(otel_context.Context())
    try:
      # Use the caller-provided label (e.g. "eval:UID0042") as the span name
      # when set, so Phoenix's trace list shows the question UID at a glance.
      # Span name format: drop the `agent.run:` prefix — the OTel `kind`
      # column already labels these as `agent`, so the prefix was redundant
      # noise in Phoenix's trace list. Final form: `iter N | base_agent [val 12/15] [FAIL 0.79]`.
      span_name = _iter_pfx_str + (_eval_label or f"{agent_name} [{run_tag}]")
      with _tracer.start_as_current_span(span_name) as run_span:
        run_span.set_attribute("agent.name", agent_name)
        run_span.set_attribute("run_tag", run_tag)
        run_span.set_attribute("query_hash", query_hash)
        if _eval_uid is not None:
            run_span.set_attribute("eval.uid", _eval_uid)
        if _eval_idx is not None:
            run_span.set_attribute("eval.index", _eval_idx)
        if _eval_gt is not None:
            run_span.set_attribute("eval.ground_truth", _eval_gt)
        # OpenInference semantic conventions — Phoenix shows these in the dedicated
        # Input/Output pane (above "All Attributes" in the detail view).
        run_span.set_attribute("openinference.span.kind", "AGENT")
        run_span.set_attribute("input.value", query)
        # Phoenix's MimeType enum only accepts "text/plain" and "application/json".
        # "text/markdown" crashes its GraphQL resolver (ValueError on the
        # MimeType() enum constructor). Phoenix does render plain text as
        # markdown automatically if it detects headings/bullets — we lose the
        # explicit toggle but get free markdown rendering anyway.
        run_span.set_attribute("input.mime_type", "text/plain")
        # Duplicate under plain names so they're also visible in "All Attributes"
        # (Phoenix hides OpenInference semantic attrs from that pane). Full
        # untruncated query here so debugging can retrieve the exact prompt.
        run_span.set_attribute("query", query)
        # Keep a link back to where this trace originated (for filtering/grouping
        # in Phoenix, since we severed the parent-child relationship).
        if parent_ctx and parent_ctx.is_valid:
            run_span.set_attribute("parent_trace_id", format(parent_ctx.trace_id, "032x"))
            run_span.set_attribute("parent_span_id", format(parent_ctx.span_id, "016x"))

        async with ClaudeSDKClient(options) as client:
            await client.query(query)

            async for msg in client.receive_response():
                messages.append(msg)

                # Only AssistantMessages contain turn-worthy content
                if isinstance(msg, AssistantMessage):
                    # Cooperative pause point: check the operator's pause
                    # flag at every turn boundary. The SDK's receive_response
                    # is pull-based, so blocking here halts the next API
                    # call until the flag is cleared. The current turn's
                    # tokens are already billed (we received its full
                    # response above) — pause activates BEFORE the next turn.
                    # Safe to leave running indefinitely; resumes seamlessly.
                    from src.harness.pause import wait_if_paused
                    await wait_if_paused(reason=f"after turn {turn_num + 1}")

                    turn_num += 1
                    # Push the live turn count up to agent.py so a wall-clock
                    # timeout can report real numbers ("burned 47 turns") in
                    # the partial AgentTrace it builds, instead of the
                    # misleading "turns=0" zero-fill the runner used to emit.
                    try:
                        from src.harness.agent import _push_partial_state
                        _push_partial_state(turn_num)
                    except Exception:
                        pass
                    # Per-turn API-reported model (e.g. "claude-opus-4-6"). Use
                    # this — not the configured alias — so Phoenix can show the
                    # actual model on every turn span and price it correctly.
                    msg_model = getattr(msg, "model", None) or ""
                    if canonical_model is None and msg_model:
                        canonical_model = msg_model
                    # Nest the turn span under the run span explicitly so it's a
                    # direct child (not a sibling of later tool spans).
                    turn_span = _tracer.start_span(
                        f"{_iter_pfx_str}{agent_name}/turn.{turn_num}",
                        context=set_span_in_context(run_span),
                    )
                    turn_span.set_attribute("turn", turn_num)
                    # `model` = canonical per-turn model (authoritative).
                    # `model.alias` = the configured shorthand (opus/sonnet/haiku)
                    # for cross-referencing with the agent's options.
                    turn_span.set_attribute("model", msg_model or model_display)
                    if model_display and model_display != msg_model:
                        turn_span.set_attribute("model.alias", model_display)
                    # Phoenix pricing pane reads `llm.model_name`; set it on EVERY
                    # turn (not just turn 1) so per-turn cost attribution works.
                    if msg_model:
                        turn_span.set_attribute("llm.model_name", msg_model)
                    turn_span.set_attribute("run_tag", run_tag)
                    # Render every turn as an LLM span so Phoenix shows the
                    # chat-bubble UI consistently (turn.1, turn.2, ... all
                    # look the same). `llm.input_messages.*` reconstructs the
                    # full conversation the model saw entering this turn
                    # (system + original query + prior assistant/tool
                    # exchanges); `llm.output_messages.0.*` is this turn's
                    # response. Per-message content is truncated at 5KB to
                    # keep heavy turns within Phoenix's per-attr limit.
                    turn_span.set_attribute("openinference.span.kind", "LLM")
                    turn_span.set_attribute("input.value", query)
                    turn_span.set_attribute("input.mime_type", "text/plain")
                    turn_span.set_attribute("query", query)
                    sys_prompt_obj = getattr(options, "system_prompt", None)
                    sys_prompt_text: str | None = None
                    if isinstance(sys_prompt_obj, str):
                        sys_prompt_text = sys_prompt_obj
                    elif isinstance(sys_prompt_obj, dict):
                        sys_prompt_text = sys_prompt_obj.get("append") or sys_prompt_obj.get("system")
                    if sys_prompt_text:
                        # Plain attr name for the All-Attributes pane (Phoenix
                        # hides OpenInference semantic attrs from that view).
                        turn_span.set_attribute(
                            "system_prompt", _truncate(sys_prompt_text, 5000),
                        )
                    # Reconstruct prior messages = everything in `messages`
                    # before the current AssistantMessage. Filter to the
                    # message types the helper knows how to render.
                    _prior = [
                        m for m in messages[:-1]
                        if isinstance(m, (AssistantMessage, UserMessage))
                    ]
                    _emit_llm_messages_for_turn(
                        turn_span,
                        sys_prompt_text,
                        query,
                        _prior,
                        msg,
                        AssistantMessage=AssistantMessage,
                        UserMessage=UserMessage,
                        TextBlock=TextBlock,
                        ThinkingBlock=ThinkingBlock,
                        ToolUseBlock=ToolUseBlock,
                        ToolResultBlock=ToolResultBlock,
                    )
                    turn_ctx = set_span_in_context(turn_span)

                    # Collect a per-turn output summary (text + tool calls made)
                    turn_texts: list[str] = []
                    turn_tool_names: list[str] = []

                    try:
                        for block in msg.content:
                            if isinstance(block, ToolUseBlock):
                                block_name = getattr(block, "name", None) or "unknown"
                                block_input = getattr(block, "input", None)

                                # Tool span nests UNDER the turn span, stays OPEN until
                                # the matching ToolResultBlock arrives in a later UserMessage.
                                tool_span = _tracer.start_span(
                                    f"{agent_name}/tool.{block_name}",
                                    context=turn_ctx,
                                )
                                tool_span.set_attribute("openinference.span.kind", "TOOL")
                                tool_span.set_attribute("tool.name", block_name)

                                # Serialize input robustly — works for dict/list/primitive/None
                                try:
                                    if block_input is None:
                                        tool_input_json = "{}"
                                    else:
                                        tool_input_json = json.dumps(
                                            block_input, default=str, ensure_ascii=False
                                        )
                                except Exception as e:
                                    tool_input_json = f"[serialization failed: {e}] {str(block_input)[:2000]}"

                                tool_span.set_attribute("tool.input", tool_input_json)
                                tool_span.set_attribute("input.value", tool_input_json)
                                tool_span.set_attribute("input.mime_type", "application/json")
                                tool_span.set_attribute("input", tool_input_json)
                                # Raw block for debugging when fields are missing/weird
                                tool_span.set_attribute("block.type", type(block).__name__)
                                tool_span.set_attribute("block.repr", repr(block)[:2000])

                                # Capture the requested skill name on the span;
                                # the injected SKILL.md content (the actual
                                # payload the LLM saw) is attached when the
                                # following UserMessage TextBlock arrives.
                                if block_name == "Skill" and isinstance(block_input, dict):
                                    _skill_name = block_input.get("skill") or block_input.get("name")
                                    if _skill_name:
                                        tool_span.set_attribute("skill.name", str(_skill_name))

                                # Evolver agents read SKILL.md via the Read tool
                                # directly (no Skill tool call). Tag the span
                                # with skill.name so the file content captured in
                                # `tool.output` is queryable as a skill payload.
                                if block_name == "Read" and isinstance(block_input, dict):
                                    _fp = str(block_input.get("file_path") or "")
                                    if "/.claude/skills/" in _fp and _fp.endswith("/SKILL.md"):
                                        # Extract the skill directory name
                                        _parts = _fp.split("/.claude/skills/", 1)[1].split("/")
                                        if _parts:
                                            tool_span.set_attribute("skill.name", _parts[0])
                                            tool_span.set_attribute("skill.read_path", _fp)

                                tool_use_id = getattr(block, "id", None) or getattr(block, "tool_use_id", None)
                                if tool_use_id:
                                    tool_span.set_attribute("tool.use_id", str(tool_use_id))
                                    pending_tool_spans[str(tool_use_id)] = tool_span
                                else:
                                    # No id to correlate results — end now with a note
                                    tool_span.set_attribute("tool.output", "[no id to correlate result]")
                                    tool_span.end()

                                turn_tool_names.append(block_name)
                            elif isinstance(block, TextBlock):
                                text = (block.text or "").strip()
                                if text:
                                    turn_span.set_attribute("text", text)
                                    turn_texts.append(text)
                            elif ThinkingBlock is not None and isinstance(block, ThinkingBlock):
                                thinking = (getattr(block, "thinking", "") or "").strip()
                                if thinking:
                                    turn_span.set_attribute("thinking", thinking)
                            else:
                                block_type = type(block).__name__
                                turn_span.set_attribute(
                                    f"unknown_block.{block_type}", str(block)[:1000]
                                )

                        # Compose turn output (what the model produced in this turn)
                        output_parts = []
                        if turn_texts:
                            output_parts.append("\n\n".join(turn_texts))
                        if turn_tool_names:
                            output_parts.append("tool_calls: " + ", ".join(turn_tool_names))
                        if output_parts:
                            output_str = "\n\n".join(output_parts)
                            turn_span.set_attribute("output.value", output_str)
                            turn_span.set_attribute("output.mime_type", "text/plain")
                            # Plain name so it also renders in "All Attributes"
                            turn_span.set_attribute("output", output_str)
                    finally:
                        turn_span.end()

                # UserMessages contain ToolResultBlocks — attach to matching tool spans
                elif isinstance(msg, UserMessage):
                    content = getattr(msg, "content", None) or []
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, ToolResultBlock):
                                tool_use_id = getattr(block, "tool_use_id", None) or getattr(block, "id", None)
                                tool_span = pending_tool_spans.pop(str(tool_use_id), None) if tool_use_id else None
                                if tool_span is None:
                                    continue
                                result_content = getattr(block, "content", "")
                                if isinstance(result_content, list):
                                    result_str = "\n".join(
                                        getattr(c, "text", str(c)) for c in result_content
                                    )
                                else:
                                    result_str = str(result_content) if result_content is not None else ""
                                tool_span.set_attribute("tool.output", result_str)
                                tool_span.set_attribute("output.value", result_str)
                                tool_span.set_attribute("output.mime_type", "text/plain")
                                tool_span.set_attribute("output", result_str)
                                is_error = bool(getattr(block, "is_error", False))
                                tool_span.set_attribute("tool.is_error", is_error)
                                # If this is the Task tool's result, expand the
                                # subagent's session as nested spans under this
                                # tool span — without this, the subagent's 40+
                                # tool calls are invisible in Phoenix.
                                _agent_id_match = re.search(r"agentId:\s*([a-f0-9]+)", result_str)
                                if _agent_id_match:
                                    _agent_id = _agent_id_match.group(1)
                                    try:
                                        _emit_subagent_spans(_agent_id, tool_span)
                                    except Exception as _e:
                                        tool_span.set_attribute("subagent.expand_error", f"{type(_e).__name__}: {_e}"[:500])
                                    # Also dump the subagent's turn-by-turn
                                    # activity to stdout so the live tail -f
                                    # log surfaces what the subagent did
                                    # (otherwise it's a black box between Task
                                    # tool_use and tool_result). The parent's
                                    # current turn_num at the time the
                                    # tool_result arrives = the parent turn
                                    # that dispatched this subagent.
                                    try:
                                        _print_subagent_to_stdout(_agent_id, run_tag, turn_num)
                                    except Exception:
                                        pass
                                # If this is the Skill tool's "Launching skill: X"
                                # response, remember the span (don't end yet) so
                                # the next UserMessage's TextBlock — which carries
                                # the injected SKILL.md content — can attach to it.
                                if "Launching skill:" in result_str:
                                    # End any prior pending one defensively
                                    if last_skill_span is not None:
                                        last_skill_span.end()
                                    last_skill_span = tool_span
                                else:
                                    tool_span.end()
                            elif isinstance(block, TextBlock) and last_skill_span is not None:
                                _injected = (block.text or "")
                                # The bundled Claude binary injects the SKILL.md
                                # contents as a TextBlock following the Skill
                                # tool_result, prefixed "Base directory for this
                                # skill: <path>\n\n<full SKILL.md>". Arrives in
                                # a SEPARATE UserMessage from the tool_result.
                                if _injected.startswith("Base directory for this skill:"):
                                    last_skill_span.set_attribute(
                                        "skill.injected_content", _injected[:30_000]
                                    )
                                    last_skill_span.set_attribute(
                                        "skill.injected_bytes", len(_injected)
                                    )
                                    last_skill_span.end()
                                    last_skill_span = None

        # Close any tool spans that never received a result (defensive)
        for _id, _span in list(pending_tool_spans.items()):
            _span.set_attribute("tool.output", "[no result received]")
            _span.end()
        pending_tool_spans.clear()
        # Close a still-open Skill span if its expected injected TextBlock
        # never arrived (rare — only when the run is cut short).
        if last_skill_span is not None:
            last_skill_span.set_attribute("skill.injected_content", "[not captured — stream ended]")
            last_skill_span.end()
            last_skill_span = None

        run_span.set_attribute("num_turns", turn_num)
        # Extract final result from ResultMessage (last msg) for output.value.
        # Prefer the SCHEMA-VALIDATED final_answer over the conversational
        # result, so Phoenix's Output panel shows the exact string the scorer
        # sees. Keep the conversational result separately under `agent.result`
        # for debugging (it often holds useful prose like chart coordinates
        # or table citations that didn't fit in `final_answer`).
        try:
            last = messages[-1] if messages else None
            structured = getattr(last, "structured_output", None) if last is not None else None
            final_answer = None
            if isinstance(structured, dict):
                final_answer = structured.get("final_answer")
                reasoning = structured.get("reasoning")
                if reasoning:
                    run_span.set_attribute("agent.reasoning", str(reasoning))
            if last is not None and hasattr(last, "result") and last.result:
                run_span.set_attribute("agent.result", str(last.result))
            # Output panel: build a glanceable markdown summary so the Info
            # tab shows Question / Answer / Reasoning at a peek instead of
            # raw conversational prose. Fall back to result when no
            # structured output was parsed.
            if final_answer is not None:
                final_str = str(final_answer)
                reasoning_str = str(reasoning) if reasoning else ""
                # Trim very long reasoning so the Info panel stays readable.
                if len(reasoning_str) > 2000:
                    reasoning_str = reasoning_str[:2000] + "…"
                # Markdown summary for the Info tab. Including Ground Truth
                # (when present via the eval contextvar) lets reviewers
                # spot-check correctness without diving into All Attributes.
                # `_eval_gt` was captured earlier from
                # `eval_run_ground_truth.get()`.
                parts = [f"## Question\n{query}"]
                if _eval_gt:
                    parts.append(f"## Ground Truth\n**{_eval_gt}**")
                parts.append(f"## Agent Answer\n**{final_str}**")
                if reasoning_str:
                    parts.append(f"## Reasoning\n{reasoning_str}")
                # Footer with run stats — turns/cost are tiny but useful.
                _stats_bits = [f"turns={turn_num}"]
                if last is not None:
                    _dur = getattr(last, "duration_ms", None)
                    if _dur:
                        _stats_bits.append(f"duration={_dur}ms")
                    _cost = getattr(last, "total_cost_usd", None)
                    if _cost:
                        _stats_bits.append(f"cost=${float(_cost):.4f}")
                parts.append("_" + ", ".join(_stats_bits) + "_")
                summary_md = "\n\n".join(parts)
                # CAREFUL: two known footguns —
                #   1. Setting a scalar "output" attribute alongside the
                #      dotted "output.value" / "output.mime_type" forms
                #      collapses the nested namespace — Phoenix then reads
                #      bare `output` and leaves `output.value` empty, so the
                #      Info panel renders no Output section.
                #   2. Phoenix's MimeType enum only accepts "text/plain" and
                #      "application/json" — "text/markdown" crashes its
                #      GraphQL resolver with a ValueError that breaks the
                #      whole project trace list (per the matching warning at
                #      input.mime_type above).
                # Phoenix auto-renders markdown when the value contains
                # headings/bullets even with mime=text/plain, so we don't
                # actually need text/markdown to get the styled output.
                run_span.set_attribute("output.value", summary_md)
                run_span.set_attribute("output.mime_type", "text/plain")
                run_span.set_attribute("agent.final_answer", final_str)
                # Mirror under `eval.agent_answer` so reviewers can compare
                # head-to-head with `eval.ground_truth` in the All Attributes pane.
                run_span.set_attribute("eval.agent_answer", final_str)

                # In-span scoring: when a scorer + ground_truth are both in
                # scope, compute the score now and STAMP IT INTO THE RUN
                # SPAN'S NAME. Phoenix's trace list shows the span name, so
                # `agent.run:base [iter 1 train 5/5] [OK 0.997]` makes
                # pass/fail scannable without opening the trace. Replaces
                # the per-sample eval [[OK]] / eval [[FAIL]] sibling spans
                # the runner used to emit (now removed as duplicate noise).
                try:
                    from src.harness.utils import eval_score_callback as _esc
                    _scorer = _esc.get()
                    if _scorer is not None and _eval_gt is not None:
                        _score = float(_scorer(query, final_str, str(_eval_gt)))
                        _passed = _score >= 0.8
                        _status = "OK" if _passed else "FAIL"
                        run_span.set_attribute("eval.score", _score)
                        run_span.set_attribute("eval.passed", _passed)
                        run_span.update_name(f"{span_name} [{_status} {_score:.2f}]")
                        from opentelemetry.trace import Status, StatusCode
                        run_span.set_status(
                            Status(StatusCode.OK if _passed else StatusCode.ERROR)
                        )
                except Exception:
                    # Scoring failures must never crash the agent run; the
                    # bare run-span name still gives Phoenix something to
                    # display.
                    pass
            elif last is not None and hasattr(last, "result") and last.result:
                result_str = str(last.result)
                run_span.set_attribute("output.value", result_str)
                run_span.set_attribute("output.mime_type", "text/plain")
            # ── Cost + token accounting for Phoenix ──
            # Prefer the SDK's own total_cost_usd (authoritative, billed by
            # Anthropic) when it's non-zero. Fall back to our local pricing
            # only for Claude Code subscription users, where the SDK reports
            # total_cost_usd=0. Our local table is loose (doesn't model 1h
            # cache-write rates, new-gen premiums, etc.) so it drifts from
            # the true billed number — don't trust it when a real number
            # from the SDK is available.
            #
            # Emit `llm.cost.total` on ONE span only. Phoenix sums this
            # attribute across all spans in the trace, so emitting it on
            # both the AGENT root and a synthetic LLM child was causing
            # double-counting in the UI's Total Cost column.
            sdk_cost = float(getattr(last, "total_cost_usd", None) or 0) if last is not None else 0.0
            if sdk_cost > 0:
                run_span.set_attribute("agent.total_cost_usd", sdk_cost)

            usage = getattr(last, "usage", None) if last is not None else None
            # Prefer the SDK-canonical model (e.g. "claude-opus-4-6") for
            # `llm.model_name` so Phoenix's pricing manifest can match.
            # Fall back to the user-provided alias only if the SDK didn't
            # report one.
            model_name = canonical_model or getattr(options, "model", None) or model_display or ""
            if usage:
                local_cost, in_tok, out_tok, cache_r, cache_w = _compute_cost_usd(model_name, usage)
                total_tok = in_tok + out_tok + cache_r + cache_w
                authoritative_cost = sdk_cost if sdk_cost > 0 else local_cost

                # Tokens on the AGENT root for visibility/debugging. NOT
                # cost — Phoenix's list-view "total cost" column only sums
                # from LLM-kind spans, and emitting on both the AGENT root
                # AND the LLM child causes Phoenix to double-count.
                run_span.set_attribute("llm.model_name", model_name)
                run_span.set_attribute("llm.token_count.prompt", in_tok)
                run_span.set_attribute("llm.token_count.completion", out_tok)
                run_span.set_attribute("llm.token_count.total", total_tok)
                if cache_r:
                    run_span.set_attribute("llm.token_count.prompt_details.cache_read", cache_r)
                if cache_w:
                    run_span.set_attribute("llm.token_count.prompt_details.cache_write", cache_w)
                run_span.set_attribute("agent.computed_cost_usd", local_cost)
                run_span.set_attribute("agent.cost_source", "sdk" if sdk_cost > 0 else "local-estimate")

                # Synthetic LLM-kind child to populate Phoenix's `span_costs`
                # table. Phoenix's pricing pipeline only creates a span_costs
                # row when (a) span_kind=LLM, (b) llm.model_name resolves
                # against the pricing manifest, and (c) per-bucket token
                # counts are present. Without a span_costs row, the trace
                # list-view "Total Cost" column renders `--` even though
                # `llm.cost.total` is set on the span.
                #
                # We emit per-bucket token counts (prompt, completion, cache
                # read, cache write) — NOT `llm.token_count.total`. Setting
                # `total` was the source of an earlier 17× inflation bug:
                # Phoenix mis-read it as completion tokens and priced them
                # at Opus-output rates. Per-bucket alone lets the manifest
                # price each bucket at its correct rate (cache_read is
                # ~10% of prompt; cache_write is ~125%).
                #
                # `llm.cost.total` is kept as a hint/audit attribute; the
                # authoritative number Phoenix shows still comes from the
                # manifest computation over per-bucket counts.
                #
                # No double-counting risk: bundled `claude` binary's API
                # calls aren't auto-instrumented by openinference, so this
                # is the only LLM-kind span in the trace with token counts.
                # Convention from openinference's Anthropic instrumentor:
                # `llm.token_count.prompt` is the FULL prompt-side total
                # (input + cache_read + cache_write), and the cache buckets
                # are *also* emitted under prompt_details. Phoenix's pricing
                # pipeline subtracts detail buckets from the aggregated
                # prompt total to get the remaining "input" bucket — so
                # emitting only `in_tok` would make the remainder negative
                # and drop the input pricing entirely.
                llm_span = _tracer.start_span(
                    f"{agent_name}/llm.cumulative",
                    context=set_span_in_context(run_span),
                )
                llm_span.set_attribute("openinference.span.kind", "LLM")
                llm_span.set_attribute("llm.model_name", model_name)
                llm_span.set_attribute("llm.cost.total", authoritative_cost)
                llm_span.set_attribute(
                    "llm.token_count.prompt", in_tok + cache_r + cache_w
                )
                llm_span.set_attribute("llm.token_count.completion", out_tok)
                if cache_r:
                    llm_span.set_attribute(
                        "llm.token_count.prompt_details.cache_read", cache_r
                    )
                if cache_w:
                    llm_span.set_attribute(
                        "llm.token_count.prompt_details.cache_write", cache_w
                    )
                llm_span.end()
        except Exception:
            pass
    finally:
        otel_context.detach(_ctx_token)

    return messages


def parse_response(
    messages: list[Any],
    response_model: Type[BaseModel],
) -> dict[str, Any]:
    """Parse Claude SDK messages into AgentTrace field values."""
    first = messages[0]
    last = messages[-1]

    # Try to parse structured output from the ResultMessage
    output = None
    parse_error = None
    raw_structured_output = last.structured_output

    if raw_structured_output is not None:
        try:
            output = response_model.model_validate(raw_structured_output)
        except (ValidationError, json.JSONDecodeError, TypeError) as e:
            parse_error = f"{type(e).__name__}: {str(e)}"
    else:
        parse_error = (
            "No structured output returned (context limit likely exceeded)"
        )

    return dict(
        uuid=first.data.get("uuid") or "",
        session_id=last.session_id or "",
        model=first.data.get("model") or "",
        tools=first.data.get("tools") or [],
        duration_ms=last.duration_ms,
        total_cost_usd=last.total_cost_usd,
        num_turns=last.num_turns,
        usage=last.usage,
        result=last.result,
        is_error=last.is_error or parse_error is not None,
        output=output,
        parse_error=parse_error,
        raw_structured_output=raw_structured_output,
        messages=messages,
    )
