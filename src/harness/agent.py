"""Agent wrapper and AgentTrace — the public interface for running agents.

This module provides:
    - AgentTrace[T]: SDK-agnostic result from an agent run
    - Agent[T]: Generic wrapper that delegates to the active SDK's executor
    - OptionsProvider: Type alias for what you can pass as agent options

SDK-specific logic lives in:
    - _claude_executor.py: Claude SDK execution + response parsing
    - _opencode_executor.py: OpenCode SDK execution + server management + response parsing
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, Type, TypeVar, Union
from pydantic import BaseModel, ValidationError
from .sdk_config import get_sdk, is_claude_sdk, is_codex_sdk, is_goose_sdk

logger = logging.getLogger(__name__)

# Adaptive timeout extension: per-agent.run stderr observer that the executor
# invokes for each stderr line from the bundled CLI. The closure tracks the
# max in-CLI retry attempt seen so far; each new attempt extends the
# asyncio.timeout deadline by 120s (one API_TIMEOUT_MS worth) so a hung
# turn that triggered a retry doesn't unfairly trip the wall clock.
# The contextvar is per-attempt; agent.py resets it for each retry attempt
# and the executor reads it through `_get_run_stderr_observer()`.
_run_stderr_observer: ContextVar[Callable[[str], None] | None] = ContextVar(
    "run_stderr_observer", default=None
)


def _get_run_stderr_observer() -> Callable[[str], None] | None:
    """Public read-side accessor for the adaptive-timeout stderr hook."""
    return _run_stderr_observer.get()


# Partial run state: the executor pushes turn count + last-known cumulative
# cost into these contextvars after each completed turn. When `_run_with_retry`
# hits its wall-clock deadline, we read the values and surface them via
# `agent.run()` as a partial AgentTrace — so the evolver sees "agent burned
# 47 turns and $1.20 before timing out", not the misleading "turns=0, cost=$0".
# Cost is not always knowable mid-stream (the SDK only reports total_cost_usd
# in the terminal ResultMessage), so the executor uses its local pricing
# table to estimate as it goes.
_partial_turn_count: ContextVar[int] = ContextVar("partial_turn_count", default=0)
_partial_cost_usd: ContextVar[float] = ContextVar("partial_cost_usd", default=0.0)
# Partial message accumulator. Each message that arrives from the SDK
# stream is appended here. When a run times out, this list is the ONLY
# record of what the agent actually did (the SDK's terminal ResultMessage
# never arrives, so `messages=[]` would leave the failure-trace file
# devoid of transcript — and the evolver hallucinates skills from
# nothing). Reset to a fresh list at the start of each attempt.
_partial_messages: ContextVar[list[Any]] = ContextVar("partial_messages", default=[])


def _push_partial_state(turn_num: int, cost_usd: float = 0.0) -> None:
    """Public write-side hook the executor calls after each turn."""
    _partial_turn_count.set(turn_num)
    if cost_usd > 0:
        _partial_cost_usd.set(cost_usd)


def _push_partial_message(msg: Any) -> None:
    """Append a streamed SDK message to the partial-messages buffer.

    Called by the executor for every message yielded by the SDK's
    `async for msg in client.receive_response()` loop. On a wall-clock
    timeout the subprocess is killed mid-stream and no terminal
    ResultMessage arrives; without this buffer the agent's failure
    trace file would contain only metadata + the timeout error string,
    starving the evolver of all diagnostic signal.
    """
    _partial_messages.get().append(msg)


def make_partial_error_trace(result: str, parse_error: str, duration_ms: int = 0) -> "AgentTrace":
    """Build an error AgentTrace using whatever partial cost/turns/messages
    were tracked before the failure, instead of zero-filling.

    Originally written only for the wall-clock-timeout path (the one case
    where the agent's own internal handling already did this, in `run()`
    below). A Jul 1 audit found the identical omission repeated at every
    sibling generic-`except Exception` handler next to a timeout handler
    across runner.py and evaluate.py — same context vars in scope, just never
    read. Centralizing here so timeout and generic-exception paths use the
    same construction instead of four independently-drifting copies.
    """
    return AgentTrace(
        duration_ms=duration_ms,
        total_cost_usd=_partial_cost_usd.get(),
        num_turns=_partial_turn_count.get(),
        usage={},
        result=result,
        is_error=True,
        output=None,
        parse_error=parse_error,
        messages=list(_partial_messages.get()),
    )

# Generic type variable — every Agent[T] produces AgentTrace[T] where T is a Pydantic model
# (e.g., AgentResponse, SkillProposerResponse, etc.)
T = TypeVar("T", bound=BaseModel)

# Import ClaudeAgentOptions only for type hints (not at runtime).
# This allows the module to load even if claude-agent-sdk is not installed.
if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions as ClaudeAgentOptionsType
else:
    ClaudeAgentOptionsType = Any

# What you can pass to Agent() as options:
#   - ClaudeAgentOptions object (static Claude config)
#   - dict (static OpenCode config)
#   - Callable that returns either (factory pattern — called fresh on each .run())
OptionsProvider = Union[
    ClaudeAgentOptionsType,
    dict[str, Any],
    Callable[[], Union[ClaudeAgentOptionsType, dict[str, Any]]],
]


class AgentTrace(BaseModel, Generic[T]):
    """Metadata and output from a single agent run.

    This is SDK-agnostic — both Claude and OpenCode executors produce the same
    AgentTrace structure, so downstream code (loop, evaluation, cache) never
    needs to know which SDK was used.
    """

    # Identity — who ran and with what config
    uuid: str = ""
    session_id: str = ""
    model: str = ""
    tools: list[str] = []

    # Full tool definitions (OpenAI function-calling schema: name +
    # description + parameters) actually available on this call, not just
    # the bare names `tools` carries. Additive alongside `tools` — nothing
    # reads `tools` expecting schemas (confirmed: the only reader is
    # tests/test_harness.py's `assert trace.tools == []` on an error path),
    # so this is a new field rather than a repurposing of the old one.
    # Populated only where a producer's local scope already holds the real
    # schema list it sent to the model (see each producer for how); left
    # as [] (the default) wherever a site doesn't have that schema in scope
    # (e.g. error-path traces built before any tool was ever selected).
    # Schemas are static across a solver run: the evolver user prompt hoists
    # them once via render_tool_schemas_section(); summarize() omits them
    # by default so every trace.md doesn't repeat the same block.
    tool_schemas: list[dict] = []

    # Metrics — cost and performance
    duration_ms: int
    total_cost_usd: float
    num_turns: int
    usage: dict[str, Any]
    result: str
    is_error: bool

    # Structured output — the main thing downstream code cares about.
    # None if parsing failed (check parse_error for why).
    output: Optional[T] = None

    # Error info when output parsing fails
    parse_error: Optional[str] = None
    raw_structured_output: Optional[Any] = None

    # Full response list for debugging. Note: SDK message objects
    # (AssistantMessage, ToolUseBlock, etc.) don't roundtrip through
    # Pydantic JSON serialization — `model_dump()` falls back to `str()`
    # on the SDK dataclasses, so a cache-replayed trace has structurally
    # useless string-ified content here regardless of `store_messages`.
    # See `cached_transcript` for the diagnostic-grade alternative the
    # cache populates at write time.
    messages: list[Any]

    # Pre-rendered turn-by-turn transcript captured at cache.set() time
    # via `_render_turn_transcript(trace.messages, ...)`. Populated only
    # for cache replays; None for fresh traces. `summarize()` falls back
    # to this when `messages` is empty, so failure-trace files written
    # by the loop for the evolver contain the actual solver tool-use
    # history instead of just metadata + an empty Full Result.
    cached_transcript: Optional[str] = None

    # True when this trace was replayed from RunCache rather than produced
    # by a fresh agent.run. Lets cost accounting distinguish API spend
    # actually paid on this run vs. re-shown numbers from a stored trace.
    # The cache.get setter flips this to True; cache.set always serializes
    # it as False (a freshly stored trace originated from real inference).
    from_cache: bool = False

    # Inline score from the executor's score_fn. When present, the runner
    # skips the batch Gemini call for this trace — no double-scoring.
    inline_score: Optional[float] = None

    # Harness budget-exhaustion marker (e.g. "Turn budget exhausted (30/30
    # turns)"), propagated from the direct executor's _meta record. The
    # executor's parse_response has emitted this key since the budget-marker
    # work, but pydantic silently dropped it here (extra='ignore') — so
    # is_provider_failure() could never distinguish "harness terminated the
    # episode at its turn budget" (a genuine, reproducible, cacheable
    # outcome) from "the API died mid-episode" (transient, retry). Empty
    # string = budget not exhausted.
    budget_exhausted: str = ""

    class Config:
        arbitrary_types_allowed = True

    def summarize(
        self,
        head_chars: int = 0,
        tail_chars: int = 0,
        tool_result_max_chars: int = 0,
        include_tool_schemas: bool = False,
    ) -> str:
        """Create a turn-by-turn transcript for passing to downstream agents.

        Walks through self.messages and renders every block type per turn:
            - ThinkingBlock   -> "(thinking): ..."
            - TextBlock       -> "text: ..."
            - ToolUseBlock    -> "(tool.<name>): <json>"
            - ToolResultBlock -> "(tool.result): <content>" (correlated by id)

        Fallback to the final ResultMessage.result when messages are empty
        or the SDK block types can't be imported.

        Tool schemas default OFF: they are identical across every trace in a
        batch, so the evolver prompt discloses them once (see
        ``build_proposer_query``) instead of repeating them in every
        ``trace.md``. Pass ``include_tool_schemas=True`` only when you need
        a self-contained dump.
        """
        lines: list[str] = []
        if self.parse_error:
            lines.append(f"Parse Error: {self.parse_error}")

        if include_tool_schemas and self.tool_schemas:
            lines.append("\n" + render_tool_schemas_section(self.tool_schemas))
        elif self.tool_schemas:
            # Compact marker so TraceDB can still detect progressive-disclosure
            # skill tools after full schemas were hoisted out of every trace.md.
            _names: list[str] = []
            for spec in self.tool_schemas:
                if not isinstance(spec, dict):
                    continue
                fn = spec.get("function", spec) if isinstance(spec.get("function"), dict) else spec
                n = fn.get("name") if isinstance(fn, dict) else None
                if n:
                    _names.append(str(n))
            if _names:
                lines.append(f"**Tools this run**: {', '.join(_names)}")

        transcript = _render_turn_transcript(self.messages, tool_result_max_chars)

        if not transcript and self.cached_transcript:
            transcript = self.cached_transcript

        if transcript:
            transcript = compact_ssb_in_transcript(transcript)
            if head_chars > 0 and tail_chars > 0 and len(transcript) > head_chars + tail_chars:
                transcript = transcript[:head_chars] + "\n\n...[truncated]...\n\n" + transcript[-tail_chars:]
            elif head_chars > 0 and len(transcript) > head_chars:
                transcript = transcript[:head_chars] + "\n\n...[truncated]..."
            lines.append("\n## Turn-by-turn transcript\n")
            lines.append(transcript)
        else:
            result_str = str(self.result) if self.result else ""
            lines.append(f"\n## Full Result\n{result_str}")

        return "\n".join(lines)


def render_tool_schemas_section(
    tool_schemas: list[dict],
    *,
    audience: str = "solver",
) -> str:
    """Render tool name + description + parameters.

    Default ``audience="solver"`` is for the evolver user prompt's
    ``### Solver agent's tools`` block: headings are ``####`` (children of
    that section, not siblings), and evolver-only affordances
    (``switch_runtime`` / ``timeout_sec``) are stripped so the evolver does
    not invent solver capabilities it does not have.

    Pass ``audience="raw"`` for a self-contained dump of the schemas as
    stored (``####`` still, but no scrubbing).
    """
    scrub = audience == "solver"
    lines: list[str] = []
    for spec in tool_schemas:
        if not isinstance(spec, dict):
            continue
        fn = spec.get("function", spec) if isinstance(spec.get("function"), dict) else spec
        name = fn.get("name", "?")
        description = fn.get("description", "")
        params = fn.get("parameters")
        if scrub:
            description = _scrub_evolver_only_tool_text(
                description if isinstance(description, str) else str(description or "")
            )
            params = _scrub_evolver_only_params(params)
        lines.append(f"#### {name}")
        if description:
            lines.append(description if isinstance(description, str) else str(description))
        if params is not None:
            lines.append("Parameters:")
            lines.append(_format_tool_parameters(params))
        lines.append("")
    return "\n".join(lines).rstrip()


_SWITCH_RUNTIME_SENTENCE = re.compile(
    r"\s*If you've switched into a task's own runtime via switch_runtime\b[^.]*\."
    r"(?:\s*outside a switched runtime[^.]*\.)?",
    re.IGNORECASE,
)


def _scrub_evolver_only_tool_text(text: str) -> str:
    """Drop switch_runtime sentences from solver-facing tool blurbs."""
    if not text:
        return text
    cleaned = _SWITCH_RUNTIME_SENTENCE.sub("", text)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def _scrub_evolver_only_params(params: Any) -> Any:
    """Omit timeout_sec — only meaningful with switch_runtime (evolver tool)."""
    if not isinstance(params, dict):
        return params
    props = params.get("properties")
    if not isinstance(props, dict) or "timeout_sec" not in props:
        return params
    out = dict(params)
    out_props = {k: v for k, v in props.items() if k != "timeout_sec"}
    out["properties"] = out_props
    req = out.get("required")
    if isinstance(req, list):
        out["required"] = [r for r in req if r != "timeout_sec"]
    return out


def _scrub_tool_input_for_transcript(inp: Any) -> Any:
    """Drop evolver-only keys from tool-call args shown in transcripts."""
    if isinstance(inp, dict) and "timeout_sec" in inp:
        return {k: v for k, v in inp.items() if k != "timeout_sec"}
    if isinstance(inp, str) and "timeout_sec" in inp:
        try:
            parsed = json.loads(inp)
        except (ValueError, TypeError):
            return inp
        if isinstance(parsed, dict) and "timeout_sec" in parsed:
            return json.dumps(
                {k: v for k, v in parsed.items() if k != "timeout_sec"},
                default=str,
                ensure_ascii=False,
            )
    return inp


def compact_ssb_user_query(content: str) -> str:
    """Strip shared SSB solver preamble; keep ``### field`` blocks only.

    The long "You are a spreadsheet expert…" template is identical across
    every SSB task and already summarized in the batch index — keeping it
    in every trace User Query doubles instruction cost (user-flagged
    2026-07-23).
    """
    if not content or "###" not in content:
        return content
    # ONLY the documented SSB field marker. There used to be a generic
    # `(###\s+\S+)` fallback here, and it was a silent-data-loss hazard: this
    # function runs on EVERY `role == "user"` message, and tau2/tau3 user
    # messages are a live CUSTOMER speaking, not a task template. Any customer
    # utterance containing "### " followed by a word — e.g. "Please check ###
    # 12345 on my statement" — had everything BEFORE that point deleted, which
    # is invisible afterwards because the result is simply a shorter message.
    #
    # Verified before removal (2026-08-04), because a shared renderer must not
    # be changed on a guess: of 883 SSB v2 `### User Query` sections in the
    # cache, ZERO contain "###" at all, and of 864 tau3 customer messages none
    # is altered by this function either — the tau2 end markers `###STOP###`
    # and `###TRANSFER###` escaped only because they have no whitespace after
    # the hashes. So dropping the fallback is a no-op on both arms' real data.
    #
    # The failure directions are also asymmetric, which settles it: without the
    # fallback an unrecognised template merely stays verbose (more tokens, all
    # information intact); with it, a user's words vanish.
    m = re.search(r"(###\s*instruction\b)", content, re.IGNORECASE)
    if not m:
        return content
    body = content[m.start():].rstrip()
    body = re.sub(
        r"\n*You should generate Python code for the final "
        r"solution of the question\.?\s*$",
        "",
        body,
    ).strip()
    return body


def compact_ssb_in_transcript(transcript: str) -> str:
    """Compact SSB User Query sections inside an already-rendered transcript.

    Stop only at Turn headings — field headings like ``### instruction`` are
    part of the User Query body and must stay inside the match.
    """
    if not transcript or "### User Query" not in transcript:
        return transcript

    def _repl(match: re.Match) -> str:
        header = match.group(1)
        body = match.group(2)
        return header + compact_ssb_user_query(body)

    return re.sub(
        r"(### User Query\n+)(.*?)(?=\n### Turn |\n--- Turn |\n## |\Z)",
        _repl,
        transcript,
        flags=re.DOTALL,
    )


USER_QUERY_PTR = ' … (full text in the "### User Query" section below)'


def compact_question_header_value(raw: str) -> str:
    """One capped line for a ``**Question**`` header — safe to re-apply.

    ``generate_index()`` re-compacts every on-disk trace file it references on
    EVERY evolver call, so this transform is repeatedly fed its own output.
    Without stripping a pointer an earlier pass already appended, each pass
    re-capped the compacted line at 160 chars and appended a SECOND pointer,
    cutting the first one mid-word. Measured on the 0724 resampling demo:
    three stacked pointers, the third truncated at ``… "### User Query" sec``
    (user-flagged cognitive proofread) — a 4-iteration run would stack four.
    Stripping at the first pointer also HEALS files already mangled that way.
    """
    txt = (raw or "").strip()
    if not txt:
        return ""
    cut = txt.find(" … (full text in the")
    if cut != -1:
        txt = txt[:cut].rstrip()
    if not txt:
        return ""
    return txt.splitlines()[0][:160]


def compact_legacy_trace_file(text: str) -> str:
    """Normalize older on-disk traces for evolver Reads.

    Legacy TraceDB files often embed (1) the full SSB question in the
    ``**Question**`` header, (2) a full ``## Tools available`` schemas dump
    (now hoisted once under Reference), and (3) an uncompacted User Query.

    Idempotent: re-running it on its own output is a no-op (see
    ``compact_question_header_value``).
    """
    if not text:
        return text
    out = text
    # Drop per-trace tool schemas (identical across the batch).
    out = re.sub(
        r"\n## Tools available\n.*?(?=\n## Turn-by-turn transcript|\n## Full Result|\Z)",
        "\n",
        out,
        flags=re.DOTALL,
    )
    # Compact **Question**: <full SSB prompt> → one-line pointer when the
    # transcript still carries ### User Query.
    if "### User Query" in out:
        out = re.sub(
            r"(\*\*Question\*\*:)(.*?)(?=\n\*\*Ground Truth\*\*:|\n\*\*Agent Answer\*\*:|\n\*\*Score\*\*:|\n## )",
            lambda m: (
                m.group(1)
                + " "
                + compact_question_header_value(m.group(2))
                + USER_QUERY_PTR
                + "\n"
            ),
            out,
            count=1,
            flags=re.DOTALL,
        )
    # An empty "**Agent Answer**:" value reads as a solver malfunction. It is
    # the norm on artifact benchmarks, where the final message is only a
    # {"finished": true} signal (user-flagged 0724). Heal older traces in
    # place; new inserts render this themselves. Idempotent — the pattern only
    # matches a value that is still empty.
    out = re.sub(
        r"^\*\*Agent Answer\*\*:[ \t]*$",
        "**Agent Answer**: (empty — no answer text in the final message; on "
        "artifact benchmarks the answer is the saved output file, scored on "
        "disk: see this task's DIFF/workspace)",
        out,
        flags=re.M,
    )
    out = compact_ssb_in_transcript(out)
    # Map residual /app paths in the compacted User Query.
    out = out.replace("/app/spreadsheets/", "./spreadsheets/").replace(
        "/app/output/", "./output/"
    )
    return out


def _format_tool_parameters(params: Any) -> str:
    """Render a tool's JSON-Schema ``parameters`` without the object-schema
    wrapper (``type`` / ``properties`` / ``required``). Those keys are
    function-calling ceremony — every tool is an object; the evolver only
    needs each arg's name, type, requiredness, and description.
    """
    if not isinstance(params, dict):
        return str(params)

    props = params.get("properties")
    if not isinstance(props, dict):
        # Non-object / unusual schema — dump as-is minus top-level type.
        _params = {k: v for k, v in params.items() if k != "type"} if params.get("type") == "object" else params
        try:
            return json.dumps(_params, indent=2, ensure_ascii=False)
        except Exception:
            return str(_params)

    required = params.get("required") or []
    if not isinstance(required, list):
        required = []
    required_set = {str(r) for r in required}

    out: list[str] = []
    for pname, pspec in props.items():
        if not isinstance(pspec, dict):
            out.append(f"- `{pname}`: {pspec}")
            continue
        ptype = pspec.get("type", "")
        req = "required" if pname in required_set else "optional"
        type_bit = f"{ptype}, {req}" if ptype else req
        desc = pspec.get("description", "")
        if not isinstance(desc, str):
            desc = str(desc) if desc else ""
        if desc:
            out.append(f"- `{pname}` ({type_bit}): {desc}")
        else:
            out.append(f"- `{pname}` ({type_bit})")
        # Preserve non-boilerplate extras (enum, default, items, …).
        extras = {
            k: v for k, v in pspec.items()
            if k not in ("type", "description")
        }
        if extras:
            try:
                extras_str = json.dumps(extras, ensure_ascii=False)
            except Exception:
                extras_str = str(extras)
            out.append(f"  {extras_str}")
    return "\n".join(out) if out else "(none)"


# Markers that indicate a PROVIDER/transport failure (network, stream timeout,
# retries exhausted) rather than a genuine solver failure (budget exhausted,
# wrong answer). Provider failures must NOT be cached — a transient outage
# would otherwise be baked in as a permanent 0 on re-eval.
_PROVIDER_FAILURE_MARKERS = (
    "[TIMEOUT]",
    "[WALL_TIMEOUT]",
    # Liveness watchdog killed a rollout that stopped making progress
    # (src/harness/liveness.py). Registering it here makes the whole existing
    # never-cache-a-provider-failure machinery apply to stalls for free.
    "[LIVENESS_STALL]",
    "[ERROR]",
    "Max retries exceeded",
    "All retries exhausted",
    "Query timed out",
    "RemoteProtocolError",
    "ConnectError",
    "ConnectTimeout",
    "ReadError",
    "ReadTimeout",
    "Server disconnected",
    "StreamSilenceTimeout",
    "No SSE",
)


def is_provider_failure(trace: "AgentTrace | None") -> bool:
    """True if the trace failed due to provider/transport issues (don't cache).

    A budget-exhausted or genuinely-wrong trace returns False (safe to cache).
    """
    if trace is None:
        return True
    if not getattr(trace, "is_error", False):
        return False
    if getattr(trace, "output", None) is not None:
        return False  # produced a parseable answer — not a transport failure
    # Harness-terminated turn-budget exhaustion is a genuine, reproducible
    # episode outcome (under the artifact contract the harness persists a
    # "[HARNESS] Reached turn budget" message instead of granting a bonus
    # turn) — NOT a transport failure. Without this check every
    # budget-exhausted artifact episode falls through to the
    # has_tool_result + "No substantive assistant message" heuristic below:
    # 3 wasted inline retries + 2 batch rerun rounds per task, and the
    # trace is never cached (2026-07-22 nex evo run: tasks 9569/8942/
    # 51090/54638 re-rolled repeatedly for exactly this reason).
    if getattr(trace, "budget_exhausted", ""):
        return False
    blob = f"{getattr(trace, 'result', '') or ''} {getattr(trace, 'parse_error', '') or ''}"
    if any(m in blob for m in _PROVIDER_FAILURE_MARKERS):
        return True
    # Detect: agent made tool calls (e.g., loaded a skill) but the API
    # failed before the model could produce a final text response. The
    # parse_error says "No substantive assistant message" but the messages
    # contain tool results — meaning the agent was making progress when
    # the API died. This is an infra failure, not a skill quality signal.
    msgs = getattr(trace, "messages", None) or []
    has_tool_result = any(
        isinstance(m, dict) and m.get("role") == "tool" for m in msgs
    )
    no_answer = "No substantive assistant message" in blob
    if has_tool_result and no_answer:
        return True
    # Detect a dangling tool call: the conversation's last message is an
    # assistant tool_calls request with no corresponding {"role": "tool"}
    # reply ever appended. This happens when the wall-clock timeout fires
    # while _execute_tool() is in flight — asyncio.CancelledError propagates
    # through the tool-execution try/except (it's a BaseException, not an
    # Exception, so `except Exception` doesn't catch it), skipping the code
    # that appends the tool result. The resulting "no answer" trace looks
    # identical to a genuine wrong-answer/budget-exhausted case but is
    # actually a process-interruption artifact — always retryable, never a
    # real skill-quality signal, so it must never be cached as a permanent 0.
    if msgs:
        last_msg = msgs[-1]
        if (
            isinstance(last_msg, dict)
            and last_msg.get("role") == "assistant"
            and last_msg.get("tool_calls")
        ):
            return True
    return False


def _render_subagent_transcript(agent_id: str, indent: str, tool_result_max_chars: int) -> str:
    """Render a subagent session's turn-by-turn transcript for inline inclusion.

    The bundled Claude binary writes subagent transcripts to
    ~/.claude/projects/<project>/<parent-session>/subagents/agent-<id>.jsonl.
    The parent's `messages` array does NOT contain subagent AssistantMessages
    — so without expanding from JSONL, the evolver only sees the Task tool's
    summary string, never the subagent's 40+ tool-call breakdown.

    Returns an indented multi-line string. Empty if the JSONL can't be found.
    """
    matches = list(Path.home().glob(f".claude/projects/*/*/subagents/agent-{agent_id}.jsonl"))
    if not matches:
        return ""
    sub_jsonl = matches[0]
    try:
        events = [json.loads(line) for line in sub_jsonl.read_text().splitlines() if line.strip()]
    except Exception:
        return ""

    # Pre-pass: collect tool results keyed by tool_use_id
    tool_results: dict[str, str] = {}
    for ev in events:
        if ev.get("type") != "user":
            continue
        content = ev.get("message", {}).get("content", []) or []
        if not isinstance(content, list):
            continue
        for blk in content:
            if not isinstance(blk, dict) or blk.get("type") != "tool_result":
                continue
            tu_id = blk.get("tool_use_id") or blk.get("id")
            if not tu_id:
                continue
            rc = blk.get("content", "")
            if isinstance(rc, list):
                rs = "\n".join(c.get("text", str(c)) if isinstance(c, dict) else str(c) for c in rc)
            else:
                rs = str(rc) if rc is not None else ""
            if tool_result_max_chars and len(rs) > tool_result_max_chars:
                rs = rs[:tool_result_max_chars] + f"\n...[truncated, {len(rs) - tool_result_max_chars:,} chars]"
            tool_results[str(tu_id)] = rs

    out: list[str] = []
    turn_n = 0
    for ev in events:
        if ev.get("type") != "assistant":
            continue
        msg = ev.get("message", {})
        content = msg.get("content", []) or []
        if not isinstance(content, list):
            continue
        turn_n += 1
        model = msg.get("model", "?") or "?"
        turn_lines: list[str] = []
        for blk in content:
            if not isinstance(blk, dict):
                continue
            bt = blk.get("type")
            if bt == "thinking":
                t = (blk.get("thinking") or "").strip()
                if t:
                    turn_lines.append(f"{indent}(thinking): {t}\n{indent}(end thinking)")
            elif bt == "text":
                t = (blk.get("text") or "").strip()
                if t:
                    turn_lines.append(f"{indent}text: {t}")
            elif bt == "tool_use":
                name = blk.get("name", "?") or "?"
                inp = blk.get("input", {})
                try:
                    inp_str = json.dumps(inp, default=str, ensure_ascii=False)
                except Exception:
                    inp_str = str(inp)
                turn_lines.append(f"{indent}(tool.{name}): {inp_str}")
                tu_id = blk.get("id") or blk.get("tool_use_id")
                if tu_id and str(tu_id) in tool_results:
                    turn_lines.append(f"{indent}(tool.result): {tool_results[str(tu_id)]}")
        if turn_lines:
            out.append(f"{indent}--- subagent turn {turn_n} [{model}] ---")
            out.extend(turn_lines)
    return "\n".join(out)


def _render_dict_transcript(messages: list[dict], tool_result_max_chars: int) -> str:
    """Render dict-based messages (from direct/OpenHands/OpenCode executors)."""
    lines: list[str] = []
    turn_num = 0
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls")

        if role == "system":
            continue
        elif role == "user":
            # Messages injected by the harness itself (budget-termination
            # marker etc.) are sent with role=user for API compatibility,
            # but labeling them "User Query" misattributes them — the
            # evolver reads these transcripts and must know the difference
            # between the task author and the harness speaking.
            if isinstance(content, str) and content.lstrip().startswith("[HARNESS]"):
                lines.append("### Harness Message\n")
                lines.append(content)
            else:
                lines.append("### User Query\n")
                lines.append(
                    compact_ssb_user_query(content)
                    if isinstance(content, str)
                    else content
                )
            lines.append("")
        elif role == "assistant":
            turn_num += 1
            lines.append(f"### Turn {turn_num}\n")
            # Solver/evolver chain-of-thought. Rendered FIRST (thinking precedes
            # action) so the evolver can see WHY each tool call / answer was made,
            # not just what. OpenRouter gives a `reasoning` string; fall back to
            # native `reasoning_content`, then to text in `reasoning_details`.
            reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
            if not reasoning and isinstance(msg.get("reasoning_details"), list):
                reasoning = "\n".join(
                    d.get("text", "") for d in msg["reasoning_details"]
                    if isinstance(d, dict) and d.get("text")
                )
            if isinstance(reasoning, str) and reasoning.strip():
                lines.append(f"(thinking): {reasoning.strip()}")
                lines.append(f"(end thinking)")
            if content:
                lines.append(f"text: {content}")
            if tool_calls:
                for tc in tool_calls:
                    func = tc.get("function", {})
                    name = func.get("name", "?")
                    args = _scrub_tool_input_for_transcript(func.get("arguments", ""))
                    if isinstance(args, dict):
                        try:
                            args = json.dumps(args, default=str, ensure_ascii=False)
                        except Exception:
                            args = str(args)
                    lines.append(f"(tool.{name}): {args}")
            lines.append("")
        elif role == "tool":
            text = content
            if tool_result_max_chars and len(text) > tool_result_max_chars:
                text = text[:tool_result_max_chars] + f"\n...[truncated, {len(text) - tool_result_max_chars:,} chars]"
            lines.append(f"(tool.result): {text}")
            lines.append("")

    return "\n".join(lines)


def _render_turn_transcript(messages: list[Any], tool_result_max_chars: int) -> str:
    """Render messages as a turn-by-turn transcript with subagent hierarchy.

    Subagent boundaries are detected by model-switch (model differs from the
    primary agent's model) and bracketed with ▶▶▶ SUBAGENT START / ◀◀◀
    SUBAGENT END markers so the evolver can see who delegated to whom and
    when control returned. Subagent content is indented 4 spaces.
    """
    try:
        from claude_agent_sdk import (
            AssistantMessage, UserMessage, ToolUseBlock, ToolResultBlock, TextBlock,
        )
        try:
            from claude_agent_sdk import ThinkingBlock
        except ImportError:
            ThinkingBlock = None
    except ImportError:
        AssistantMessage = None

    if not messages:
        return ""

    # Dict-based messages (direct executor, OpenHands, etc.)
    if isinstance(messages[0], dict):
        return _render_dict_transcript(messages, tool_result_max_chars)

    if AssistantMessage is None:
        return ""

    # First pass: collect tool results keyed by tool_use_id for inline correlation
    tool_results: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, UserMessage):
            content = getattr(msg, "content", None) or []
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, ToolResultBlock):
                    continue
                tool_use_id = getattr(block, "tool_use_id", None) or getattr(block, "id", None)
                if not tool_use_id:
                    continue
                raw = getattr(block, "content", "")
                if isinstance(raw, list):
                    text = "\n".join(getattr(c, "text", str(c)) for c in raw)
                else:
                    text = str(raw) if raw is not None else ""
                if tool_result_max_chars and len(text) > tool_result_max_chars:
                    text = text[:tool_result_max_chars] + f"\n...[truncated, {len(text) - tool_result_max_chars:,} chars]"
                is_error = getattr(block, "is_error", False)
                marker = "[ERROR] " if is_error else ""
                tool_results[str(tool_use_id)] = f"{marker}{text}"

    # Pre-pass: identify subagent message ranges by tracking Task tool_use_ids.
    # Works regardless of whether the subagent uses the same model.
    subagent_msg_indices: set[int] = set()
    subagent_start_desc: dict[int, str] = {}  # first subagent msg idx -> task description
    pending_task_ids: dict[str, tuple[int, str]] = {}  # tool_use_id -> (dispatch_idx, desc)
    for global_idx, msg in enumerate(messages):
        if isinstance(msg, AssistantMessage):
            for block in getattr(msg, "content", []) or []:
                if isinstance(block, ToolUseBlock) and getattr(block, "name", "") == "Task":
                    tid = str(getattr(block, "id", None) or getattr(block, "tool_use_id", None) or "")
                    if tid:
                        task_input = getattr(block, "input", None)
                        desc = "(subtask)"
                        if isinstance(task_input, dict):
                            desc = task_input.get("description") or task_input.get("prompt", "")[:60] or "(subtask)"
                        pending_task_ids[tid] = (global_idx, desc)
        elif isinstance(msg, UserMessage):
            for block in getattr(msg, "content", []) or []:
                if isinstance(block, ToolResultBlock):
                    tid = str(getattr(block, "tool_use_id", None) or getattr(block, "id", None) or "")
                    if tid in pending_task_ids:
                        dispatch_idx, desc = pending_task_ids.pop(tid)
                        first_sub = None
                        for inner_idx in range(dispatch_idx + 1, global_idx):
                            if isinstance(messages[inner_idx], AssistantMessage):
                                subagent_msg_indices.add(inner_idx)
                                if first_sub is None:
                                    first_sub = inner_idx
                        if first_sub is not None:
                            subagent_start_desc[first_sub] = desc

    # Second pass: render each AssistantMessage as a turn with hierarchical
    # numbering. Primary agent turns: "Turn 1", "Turn 2", etc. Subagent turns
    # nest under the dispatching turn: "Turn 3, subagent turn 1", "Turn 3,
    # subagent turn 2", etc.
    out_lines: list[str] = []
    primary_turn = 0
    sub_turn = 0
    in_subagent = False
    pending_task_desc: str | None = None
    indent = ""
    # Map content-digest -> (turn_label, tool_name, char_count)
    seen_results: dict[str, tuple[str, str, int]] = {}

    for global_idx, msg in enumerate(messages):
        if not isinstance(msg, AssistantMessage):
            continue
        msg_model = (getattr(msg, "model", "") or "").strip()
        is_subagent_msg = global_idx in subagent_msg_indices

        # Subagent entry
        if is_subagent_msg and not in_subagent:
            desc = pending_task_desc or subagent_start_desc.get(global_idx, "(subtask)")
            out_lines.append(f"\n    ▶▶▶ SUBAGENT START: {desc}   [{msg_model}]")
            in_subagent = True
            sub_turn = 0
            indent = "    "
            pending_task_desc = None

        # Subagent exit
        if not is_subagent_msg and in_subagent:
            out_lines.append(f"    ◀◀◀ SUBAGENT END\n")
            in_subagent = False
            sub_turn = 0
            indent = ""

        if in_subagent:
            sub_turn += 1
            turn_label = f"Turn {primary_turn}, subagent turn {sub_turn}"
        else:
            primary_turn += 1
            turn_label = f"Turn {primary_turn}"

        # Render blocks into a temp buffer; skip the turn entirely if empty
        turn_lines: list[str] = []
        for block in getattr(msg, "content", []) or []:
            if ThinkingBlock is not None and isinstance(block, ThinkingBlock):
                text = (getattr(block, "thinking", "") or "").strip()
                if text:
                    turn_lines.append(f"{indent}(thinking): {text}\n{indent}(end thinking)")
            elif isinstance(block, TextBlock):
                text = (getattr(block, "text", "") or "").strip()
                if text:
                    turn_lines.append(f"{indent}text: {text}")
            elif isinstance(block, ToolUseBlock):
                name = getattr(block, "name", "?")
                raw_inp = _scrub_tool_input_for_transcript(getattr(block, "input", None))
                try:
                    inp = json.dumps(raw_inp, default=str, ensure_ascii=False)
                except Exception:
                    inp = str(raw_inp)
                turn_lines.append(f"{indent}(tool.{name}): {inp}")
                if name == "Task":
                    task_input = getattr(block, "input", None)
                    if isinstance(task_input, dict):
                        pending_task_desc = task_input.get("description") or task_input.get("prompt", "")[:60] or "(subtask)"
                tool_use_id = getattr(block, "id", None) or getattr(block, "tool_use_id", None)
                if tool_use_id and str(tool_use_id) in tool_results:
                    result_text = tool_results[str(tool_use_id)]
                    # For Task tools, try to render the subagent's transcript
                    # from its JSONL file (the SDK runs subagents internally,
                    # so their turns aren't in the parent's messages list).
                    if name == "Task":
                        agent_id_match = re.search(r"agentId:\s*([a-f0-9]+)", result_text)
                        if agent_id_match:
                            sub_indent = indent + "    "
                            sub_transcript = _render_subagent_transcript(
                                agent_id_match.group(1), sub_indent, tool_result_max_chars
                            )
                            if sub_transcript:
                                desc = pending_task_desc or "(subtask)"
                                turn_lines.append(
                                    f"{indent}▶▶▶ SUBAGENT START: {desc}"
                                )
                                turn_lines.append(sub_transcript)
                                turn_lines.append(
                                    f"{indent}◀◀◀ SUBAGENT END"
                                )
                                pending_task_desc = None
                    if len(result_text) >= 120:
                        digest = hashlib.sha256(
                            result_text.encode("utf-8", errors="replace")
                        ).hexdigest()[:16]
                        prior = seen_results.get(digest)
                        if prior is not None:
                            prior_label, prior_name, prior_len = prior
                            turn_lines.append(
                                f"{indent}(tool.result): [identical to {prior_label}'s "
                                f"{prior_name} result — {prior_len:,} chars omitted]"
                            )
                            continue
                        seen_results[digest] = (turn_label, name, len(result_text))
                    turn_lines.append(f"{indent}(tool.result): {result_text}")
            else:
                turn_lines.append(f"{indent}({type(block).__name__}): {str(block)}")

        if turn_lines:
            model_tag = f" [{msg_model}]" if msg_model else ""
            out_lines.append(f"\n{indent}--- {turn_label}{model_tag} ---")
            out_lines.extend(turn_lines)

    if in_subagent:
        out_lines.append(f"    ◀◀◀ SUBAGENT END\n")

    return "\n".join(out_lines).strip()


class Agent(Generic[T]):
    """Generic wrapper for running agents via Claude SDK or OpenCode SDK.

    Usage:
        agent = Agent(options=my_options, response_model=AgentResponse)
        trace = await agent.run("What is 2+2?")
        print(trace.output.final_answer)  # "4"

    The Agent handles:
        - SDK selection (Claude vs OpenCode) based on global sdk_config
        - Retry with exponential backoff (3 attempts, 30s → 60s → 120s)
        - 20-minute timeout per attempt
        - Structured output validation via the SDK-specific executor
    """

    TIMEOUT_SECONDS = 720    # 12 minutes per attempt — base budget for productive work
    # The wall-clock budget extends ADAPTIVELY by +120s per observed in-CLI
    # retry (see _run_stderr_observer below). API_TIMEOUT_MS=120s ×
    # CLAUDE_CODE_MAX_RETRIES=5 (set in executor.py) bounds retries at K≤5,
    # so the worst-case effective budget is 12 + 2×5 = 22 min on a single
    # severely network-stuck turn — but healthy runs stay at 12 min and
    # don't pay for retry capacity they didn't use.
    MAX_RETRIES = 2          # Total attempts before giving up
    INITIAL_BACKOFF = 30     # Seconds to wait after first failure (doubles each retry)

    def __init__(
        self,
        options: OptionsProvider,
        response_model: Type[T],
        *,
        name: str = "agent",
        timeout_seconds: int | None = None,
        max_retries: int | None = None,
    ):
        self._options = options
        self.response_model = response_model
        self.name = name
        self.timeout_seconds = (
            self.TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        )
        self.max_retries = self.MAX_RETRIES if max_retries is None else max_retries
        self.sdk_override: str | None = None
        # Per-call options a CALLER supplies between runs, merged over whatever
        # the provider yields. Needed because every real launcher passes a
        # FACTORY (`Agent(_build_evolver_options, ...)`) that returns a brand-new
        # dict per call: writing into the dict `_get_options()` handed back was
        # writing into a throwaway, so anything set that way silently vanished
        # before the executor ever saw it. Merging here is the only place that
        # holds for both static-dict and factory providers.
        self.runtime_options: dict[str, Any] = {}

    def _get_options(self) -> Union[ClaudeAgentOptionsType, dict[str, Any]]:
        """Resolve options — if it's a factory (callable), call it to get fresh options."""
        opts = self._options() if callable(self._options) else self._options
        # Only dict-shaped options can carry these; the Claude-SDK dataclass has
        # nowhere to put them, and silently dropping beats raising on a path
        # that has no use for them.
        if self.runtime_options and isinstance(opts, dict):
            opts = {**opts, **self.runtime_options}
        return opts

    async def _execute_query(self, query: str, tag: str | None = None) -> list[Any]:
        """Execute a single query by delegating to the active SDK's executor."""
        options = self._get_options()

        sdk = self.sdk_override or get_sdk()
        if sdk == "claude":
            from .claude import executor as _claude_executor
            return await _claude_executor.execute_query(options, query, agent_name=self.name, tag=tag)
        if sdk == "opencode":
            raise NotImplementedError("the opencode harness is not included in this release")
        if sdk == "openhands":
            from .openhands import executor as _openhands_executor
            return await _openhands_executor.execute_query(options, query)
        if sdk == "codex":
            from .codex import executor as _codex_executor
            return await _codex_executor.execute_query(options, query)
        if sdk == "goose":
            from .goose import executor as _goose_executor
            return await _goose_executor.execute_query(options, query)
        if sdk == "direct":
            from .direct import executor as _direct_executor
            return await _direct_executor.execute_query(options, query, agent_name=self.name, tag=tag)
        else:
            raise ValueError(f"Unknown SDK: {sdk!r}")

    async def _run_with_retry(self, query: str, tag: str | None = None) -> list[Any]:
        """Execute query with timeout, retrying only on subprocess-level errors.

        TimeoutError = the agent legitimately exhausted its 12-min wall-clock
        budget (too many turns / heavy thinking). Network blips are already
        handled inside the bundled CLI's own per-request retry loop
        (API_TIMEOUT_MS + CLAUDE_CODE_MAX_RETRIES, set in the executor), so a
        timeout here is a real over-budget condition — restarting from turn 0
        won't recover, just burns the same budget again. Surface immediately.

        Other exceptions (subprocess crash, SDK init error, etc.) still get
        one retry with backoff since those are usually transient and a fresh
        process tends to recover.
        """
        last_error: Exception | None = None
        backoff = self.INITIAL_BACKOFF

        # Detects in-CLI retry banners. Bundled CLI emits "(attempt N/K)"
        # when its API client retries a hung HTTP request. We dedupe by
        # only acting when the attempt number we see is strictly higher
        # than any previous one (multiple stderr lines per retry are common).
        attempt_pat = re.compile(r"\(attempt\s+(\d+)/\d+\)", re.IGNORECASE)

        for attempt in range(self.max_retries):
            # Reset partial-state contextvars at the start of every attempt
            # so a previous attempt's stale numbers don't leak into a fresh
            # run's timeout report.
            _partial_turn_count.set(0)
            _partial_cost_usd.set(0.0)
            _partial_messages.set([])  # fresh list per attempt
            try:
                async with asyncio.timeout(self.timeout_seconds) as tm:
                    # Publish the active timeout to wait_if_paused so a
                    # cooperative pause can refund the wall-clock budget
                    # consumed during the pause. See src/harness/pause.py.
                    from src.harness.pause import active_timeout as _active_tm
                    _tm_token = _active_tm.set(tm)
                    max_seen = [1]  # closed-over per-attempt state

                    def _observe_stderr(line: str) -> None:
                        m = attempt_pat.search(line)
                        if not m:
                            return
                        n = int(m.group(1))
                        if n <= max_seen[0]:
                            return
                        max_seen[0] = n
                        current = tm.when()
                        if current is None:
                            return
                        # +120s per new attempt = matches API_TIMEOUT_MS so
                        # the agent gets back the time the retry consumed.
                        tm.reschedule(current + 120.0)
                        logger.warning(
                            f"Adaptive timeout: in-CLI retry attempt {n} "
                            f"detected — extending wall-clock budget +120s"
                        )

                    obs_token = _run_stderr_observer.set(_observe_stderr)
                    try:
                        return await self._execute_query(query, tag=tag)
                    finally:
                        _run_stderr_observer.reset(obs_token)
                        _active_tm.reset(_tm_token)
            except asyncio.TimeoutError:
                # Surface partial state in the error message so callers
                # (and the evolver) see "agent burned N turns and $X" rather
                # than the misleading "turns=0, cost=$0" zero-fill.
                n_turns = _partial_turn_count.get()
                spent = _partial_cost_usd.get()
                logger.warning(
                    f"Query timed out after {self.timeout_seconds}s — "
                    f"agent over budget, not retrying (turns={n_turns}, cost=${spent:.4f})"
                )
                raise TimeoutError(
                    f"Query timed out after {self.timeout_seconds}s "
                    f"(burned {n_turns} turns, ${spent:.4f} before timeout)"
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed: {e}. Retrying in {backoff}s..."
                )
                # Defense-in-depth alongside executor.py's own
                # execute_query `finally` fix (2026-07-10): that fix already
                # covers this case (any exception raised inside
                # execute_query's try body passes through its finally before
                # reaching here), but a bare `except Exception` retry loop
                # reusing the same coroutine/context is exactly the shape of
                # bug the switch_runtime contextvar leak came from, so a
                # second, idempotent no-op-if-already-clean call here costs
                # nothing and removes any dependency on that invariant
                # holding forever as this code evolves.
                try:
                    from src.harness.direct.executor import _force_cleanup_switched_runtime
                    await _force_cleanup_switched_runtime()
                except Exception:
                    pass

            if attempt < self.max_retries - 1:
                await asyncio.sleep(backoff)
                backoff *= 2  # Exponential backoff: 30s → 60s → 120s

        raise last_error if last_error else RuntimeError("All retries exhausted")

    async def run(self, query: str, tag: str | None = None) -> AgentTrace[T]:
        """Run the agent on a query and return a structured AgentTrace.

        This is the main entry point. It:
            1. Calls _run_with_retry to get raw SDK messages
            2. Delegates response parsing to the active SDK's executor
            3. Returns an AgentTrace with all metadata + parsed output

        Args:
            query: The question for the agent.
            tag: Optional human-readable label (e.g. "train 3/8") shown in
                 stdout prints and attached to spans. Defaults to a sha256
                 hash of the query when omitted.
        """
        try:
            messages = await self._run_with_retry(query, tag=tag)
        except TimeoutError as e:
            # Convert the wall-clock-budget TimeoutError into a partial
            # AgentTrace so callers see real turn / cost numbers in the
            # failure record. The evolver renders these into the "Failure N
            # (turns=X, cost=$Y)" header — without this conversion the
            # exception path in runner.py zero-filled the placeholder, and
            # the evolver was misled into treating timeouts as zero-work
            # parse failures rather than the over-budget runs they actually
            # are.
            n_turns = _partial_turn_count.get()
            spent = _partial_cost_usd.get()
            partial_msgs = list(_partial_messages.get())  # snapshot
            err_msg = str(e)
            return AgentTrace(
                duration_ms=int(self.timeout_seconds * 1000),
                total_cost_usd=spent,
                num_turns=n_turns,
                usage={},
                result=err_msg,
                is_error=True,
                output=None,
                parse_error=err_msg,
                messages=partial_msgs,
            )

        sdk = self.sdk_override or get_sdk()
        if sdk == "claude":
            from .claude import executor as _claude_executor
            fields = _claude_executor.parse_response(messages, self.response_model)
        elif sdk == "opencode":
            raise NotImplementedError("the opencode harness is not included in this release")
        elif sdk == "openhands":
            from .openhands import executor as _openhands_executor
            fields = await _openhands_executor.parse_response(messages,self.response_model,self._get_options,query)
        elif sdk == "codex":
            from .codex import executor as _codex_executor
            fields = _codex_executor.parse_response(messages, self.response_model, self._get_options)
        elif sdk == "goose":
            from .goose import executor as _goose_executor
            fields = _goose_executor.parse_response(messages, self.response_model, self._get_options)
        elif sdk == "direct":
            from .direct import executor as _direct_executor
            fields = _direct_executor.parse_response(messages, self.response_model, self._get_options)
        else:
            raise ValueError(f"Unknown SDK: {sdk!r}")

        try:
            return AgentTrace(**fields)
        except (ValidationError, TypeError) as e:
            # Non-Claude models via CC SDK may return fields that don't
            # match AgentTrace schema. Build a minimal valid trace.
            import uuid as _uuid
            return AgentTrace(
                uuid=str(_uuid.uuid4()),
                session_id=fields.get("session_id", "unknown"),
                model=fields.get("model", "unknown"),
                tools=fields.get("tools", []),
                tool_schemas=fields.get("tool_schemas", []),
                duration_ms=fields.get("duration_ms", 0),
                total_cost_usd=fields.get("total_cost_usd", 0.0),
                num_turns=fields.get("num_turns", 0),
                usage=fields.get("usage", {}),
                result=fields.get("result", ""),
                is_error=True,
                output=None,
                parse_error=f"AgentTrace validation: {e}",
                raw_structured_output=fields.get("raw_structured_output"),
                messages=messages,
            )
