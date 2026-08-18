"""Shared OTel/Phoenix tracing helpers for non-Claude executors.

Every executor needs the same pattern: read eval contextvars, open an
AGENT root span, stamp eval attributes, emit an LLM cost child, and
optionally run the in-span scorer.  This module extracts that boilerplate
so Direct, OpenHands, OpenCode, and Goose can share it.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from typing import Any, Generator

from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace
from opentelemetry.trace import Span, set_span_in_context

# Throttle for in-place mid-turn transcript updates on the open turn span.
# We mutate ONE span's output attributes as tokens arrive (no live.N.pK clones).
# OTEL still exports that span only when it ends — Phoenix then shows the
# finished turn once, with the full grown text.
_PHOENIX_LIVE_MS = int(os.environ.get("EVOSKILL_PHOENIX_LIVE_MS", "750"))
_PHOENIX_LIVE = os.environ.get("EVOSKILL_PHOENIX_LIVE", "1") == "1"

from .utils import (
    eval_iter_label,
    eval_run_ground_truth,
    eval_run_index,
    eval_run_label,
    eval_run_uid,
    eval_score_callback,
)

# Per-harness tracers are created by callers; this is the fallback.
_default_tracer = otel_trace.get_tracer("evoskill.harness")


def flush_spans(timeout_ms: int = 2000) -> None:
    """Force-flush all pending spans so Phoenix shows them immediately."""
    try:
        provider = otel_trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush(timeout_millis=timeout_ms)
    except Exception:
        pass


# ── Cost estimation ──────────────────────────────────────────────────
# OpenRouter returns the actual cost in usage.cost on every response.
# Use that as the single source of truth. estimate_cost() is only a
# last-resort fallback for non-OpenRouter providers that don't report cost.

_FALLBACK_USD_PER_MTOK = (0.10, 0.30)  # conservative fallback — should rarely fire


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Fallback cost estimate when the API doesn't report cost.

    Prefer usage.cost from the API response (OpenRouter always provides it).
    This function uses a single conservative fallback rate — no per-model table,
    since those are always stale and wrong.
    """
    p_in, p_out = _FALLBACK_USD_PER_MTOK
    return (prompt_tokens * p_in + completion_tokens * p_out) / 1_000_000


# ── Eval context snapshot ────────────────────────────────────────────

class EvalContext:
    """Snapshot of eval contextvars at span-creation time."""
    __slots__ = ("label", "uid", "index", "ground_truth", "iter_prefix", "scorer")

    def __init__(self) -> None:
        self.label: str | None = eval_run_label.get()
        self.uid: str | None = eval_run_uid.get()
        self.index: int | None = eval_run_index.get()
        self.ground_truth: str | None = eval_run_ground_truth.get()
        _iter = eval_iter_label.get()
        self.iter_prefix: str = f"{_iter} | " if _iter else ""
        self.scorer = eval_score_callback.get()


def _stamp_eval_attrs(span: Span, ctx: EvalContext) -> None:
    if ctx.uid is not None:
        span.set_attribute("eval.uid", ctx.uid)
    if ctx.index is not None:
        span.set_attribute("eval.index", ctx.index)
    if ctx.ground_truth is not None:
        span.set_attribute("eval.ground_truth", ctx.ground_truth)


# ── Root span context manager ────────────────────────────────────────

@contextmanager
def agent_root_span(
    tracer: otel_trace.Tracer,
    *,
    harness_name: str,
    agent_name: str,
    query: str,
    run_tag: str = "",
    session_id: str = "",
) -> Generator[tuple[Span, EvalContext, str], None, None]:
    """Open an isolated AGENT root span with eval attributes.

    Yields (span, eval_ctx, session_id). The session_id groups the root
    agent trace with its turn children in Phoenix's Sessions tab.
    """
    import uuid as _uuid
    ctx = EvalContext()
    span_name = ctx.iter_prefix + (ctx.label or f"{agent_name} [{run_tag}]")
    sid = session_id or _uuid.uuid4().hex[:12]

    token = otel_context.attach(otel_context.Context())
    try:
        with tracer.start_as_current_span(span_name) as span:
            # AGENT (not LLM): Phoenix only writes span_costs for LLM spans.
            # Cost is priced on turn children; marking the root LLM double-counted.
            span.set_attribute("openinference.span.kind", "AGENT")
            span.set_attribute("agent.name", agent_name)
            span.set_attribute("agent.harness", harness_name)
            span.set_attribute("session.id", sid)
            span.set_attribute("input.value", query)
            span.set_attribute("input.mime_type", "text/plain")
            span.set_attribute("query", query)
            if run_tag:
                span.set_attribute("run_tag", run_tag)
            _stamp_eval_attrs(span, ctx)
            yield span, ctx, sid
    finally:
        otel_context.detach(token)


# ── Turn + tool child spans ──────────────────────────────────────────

def start_turn_span(
    tracer: otel_trace.Tracer,
    parent: Span,
    *,
    turn_num: int,
    model: str,
    eval_ctx: EvalContext,
    agent_name: str = "",
    run_tag: str = "",
) -> Span:
    label = eval_ctx.label or agent_name
    tag_suffix = f" [{run_tag}]" if run_tag else ""
    name = f"{eval_ctx.iter_prefix}{label}{tag_suffix}/turn.{turn_num}"
    span = tracer.start_span(name, context=set_span_in_context(parent))
    span.set_attribute("openinference.span.kind", "LLM")
    # Phoenix only writes span_costs for LLM spans that carry llm.model_name
    # (see phoenix.db.insertion.helpers.should_calculate_span_cost). CHAIN
    # roots are never priced, so turn spans must arm the pricer. Prefer
    # llm.provider=openrouter so custom OpenRouter price rows win over
    # provider-agnostic / wrong-provider matches.
    span.set_attribute("turn", turn_num)
    span.set_attribute("llm.model_name", model)
    span.set_attribute("llm.provider", "openrouter")
    span.set_attribute("model", model)
    return span


def emit_live_turn(
    tracer: otel_trace.Tracer,
    *,
    turn_num: int,
    model: str,
    eval_ctx: EvalContext,
    agent_name: str = "",
    session_id: str = "",
    system_prompt: str = "",
    messages: list[dict[str, Any]] | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cumulative_prompt_tokens: int = 0,
    cumulative_completion_tokens: int = 0,
) -> None:
    """Emit an independent root-level span for real-time Phoenix visibility.

    Shows the full conversation history up to this turn so clicking any
    live.N span gives the complete transcript through turn N.
    All live spans share the same session_id so Phoenix groups them.
    """
    label = eval_ctx.label or agent_name
    name = f"{eval_ctx.iter_prefix}{label}/live.{turn_num}"
    token = otel_context.attach(otel_context.Context())
    try:
        span = tracer.start_span(name)
        span.set_attribute("openinference.span.kind", "CHAIN")
        span.set_attribute("turn", turn_num)
        span.set_attribute("llm.model_name", model)
        span.set_attribute("live", True)
        if session_id:
            span.set_attribute("session.id", session_id)
        _stamp_eval_attrs(span, eval_ctx)

        # Full conversation as input messages (system + all turns so far)
        all_msgs: list[dict[str, Any]] = []
        if system_prompt:
            all_msgs.append({"role": "system", "content": system_prompt})
        if messages:
            for m in messages:
                role = m.get("role", "")
                content = m.get("content", "") or ""
                if role and content:
                    all_msgs.append({"role": role, "content": content})
        _set_oi_messages(span, "llm.input_messages", all_msgs)

        # Output = last assistant message
        last_assistant = ""
        if messages:
            for m in reversed(messages):
                if m.get("role") == "assistant" and m.get("content"):
                    last_assistant = m["content"]
                    break
        if last_assistant:
            span.set_attribute("output.value", span_text(last_assistant))
            _set_oi_messages(span, "llm.output_messages", [
                {"role": "assistant", "content": last_assistant},
            ])

        span.set_attribute("llm.token_count.prompt", cumulative_prompt_tokens)
        span.set_attribute("llm.token_count.completion", cumulative_completion_tokens)
        # Without this, Phoenix independently estimates cost from model +
        # cumulative token counts (its own stale/default pricing) on EVERY
        # live.N span, then rolls these up alongside the root's own correct
        # llm.cost.total -- multi-counting that compounds with turn count
        # (confirmed: a 26-turn GLM 5.2 evolver call showed $1.44 real cost
        # but $2.58 on the root span badge and $5.15 as the trace total).
        # The real cost is already attributed exactly once on the root span;
        # these live spans exist purely for real-time streaming visibility.
        span.set_attribute("llm.cost.total", 0.0)
        span.end()
    finally:
        otel_context.detach(token)
    flush_spans()


def emit_live_partial(
    *args: Any,
    **kwargs: Any,
) -> None:
    """Deprecated no-op — mid-turn clones (live.N.pK) were removed.

    Use LiveTranscriptEmitter, which mutates the single open turn span in place.
    """
    return None


class LiveTranscriptEmitter:
    """Grow ONE open turn span's output in place as the assistant streams.

    Writes OpenInference roles separately (reasoning / assistant / tool_call)
    — never a single flattened blob. Does not create live.N.pK root spans.
    """

    __slots__ = (
        "turn_span", "interval_s", "_last_emit", "_last_text", "_oi_count",
    )

    def __init__(
        self,
        turn_span: Span,
        *,
        interval_ms: int | None = None,
        # Backward-compat kwargs from older call sites (ignored).
        tracer: otel_trace.Tracer | None = None,
        turn_num: int = 0,
        model: str = "",
        eval_ctx: Any = None,
        agent_name: str = "",
        session_id: str = "",
    ) -> None:
        self.turn_span = turn_span
        ms = _PHOENIX_LIVE_MS if interval_ms is None else interval_ms
        self.interval_s = max(0.1, ms / 1000.0)
        self._last_emit = 0.0
        self._last_text = ""
        self._oi_count = 0

    @staticmethod
    def enabled() -> bool:
        return _PHOENIX_LIVE

    def update(
        self,
        content: str = "",
        reasoning: str = "",
        tool_calls: dict[int, dict[str, Any]] | None = None,
        *,
        force: bool = False,
    ) -> None:
        # Fingerprint without flattening into one Phoenix block.
        tc_sig = ""
        if tool_calls:
            for idx in sorted(tool_calls):
                fn = (tool_calls[idx].get("function") or {})
                tc_sig += f"{fn.get('name')}|{fn.get('arguments')}|"
        fingerprint = f"{reasoning}\0{content}\0{tc_sig}"
        if not (reasoning or content or tool_calls) or fingerprint == self._last_text:
            return
        now = time.monotonic()
        if not force and (now - self._last_emit) < self.interval_s:
            return
        self._last_text = fingerprint
        self._last_emit = now
        try:
            self._write_structured(content, reasoning, tool_calls)
        except Exception:
            pass

    def flush(
        self,
        content: str = "",
        reasoning: str = "",
        tool_calls: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        self.update(content, reasoning, tool_calls, force=True)

    def _write_structured(
        self,
        content: str,
        reasoning: str,
        tool_calls: dict[int, dict[str, Any]] | None,
    ) -> None:
        """Mirror executor._emit_messages_on_span role split for Phoenix UI."""
        prefix = "llm.output_messages"
        idx = 0
        preview_parts: list[str] = []
        if reasoning:
            self.turn_span.set_attribute(f"{prefix}.{idx}.message.role", "reasoning")
            self.turn_span.set_attribute(
                f"{prefix}.{idx}.message.content", span_text(reasoning)
            )
            preview_parts.append(reasoning)
            idx += 1
        if content:
            self.turn_span.set_attribute(f"{prefix}.{idx}.message.role", "assistant")
            self.turn_span.set_attribute(
                f"{prefix}.{idx}.message.content", span_text(content)
            )
            preview_parts.append(content)
            idx += 1
        if tool_calls:
            for i in sorted(tool_calls):
                tc = tool_calls[i]
                fn = tc.get("function") or {}
                name = fn.get("name") or "?"
                args = fn.get("arguments") or ""
                block = f"Tool: {name}\n\n{args}"
                self.turn_span.set_attribute(f"{prefix}.{idx}.message.role", "tool_call")
                self.turn_span.set_attribute(
                    f"{prefix}.{idx}.message.content", span_text(block)
                )
                preview_parts.append(f"[tool_call: {name}]")
                idx += 1
        # Blank any stale higher-index blocks from an earlier longer partial.
        for stale in range(idx, self._oi_count):
            self.turn_span.set_attribute(f"{prefix}.{stale}.message.role", "")
            self.turn_span.set_attribute(f"{prefix}.{stale}.message.content", "")
        self._oi_count = idx
        if preview_parts:
            self.turn_span.set_attribute(
                "output.value", span_text("\n\n".join(preview_parts))
            )


# 0 = no tracing-side limit (record exactly what the model saw).
SPAN_TEXT_LIMIT = int(os.environ.get("EVOSKILL_SPAN_TEXT_LIMIT", "0") or 0)


def span_text(value: Any, limit: int | None = None) -> str:
    """Stringify a value for a span attribute WITHOUT silently losing content.

    A span must show what the model actually saw (feedback: "Phoenix must show
    authentic LLM context — no truncation mismatch"). Every tool-output site
    used to slice `[:8000]`, which made a genuine full read and a capped read
    look IDENTICAL: during the 2026-07-24 evolver audit every large read
    reported exactly 8,000 bytes, so a complete `cat` of a 14KB trace was
    indistinguishable from a truncated one, and the real cap (100,000, applied
    upstream by `_execute_tool`) had to be recovered from source to rule out a
    fake "full read".

    So: record it whole by default. `EVOSKILL_SPAN_TEXT_LIMIT` can impose a
    ceiling if span volume ever becomes a problem, and when it bites the marker
    states the TRUE length inline — a reader always knows a cut happened.
    """
    if isinstance(value, str):
        s = value
    else:
        try:
            s = json.dumps(value, default=str, ensure_ascii=False)
        except Exception:
            s = str(value)
    cap = SPAN_TEXT_LIMIT if limit is None else limit
    if cap and cap > 0 and len(s) > cap:
        return (
            f"{s[:cap]}\n\n[span-truncated for display: showing {cap:,} of "
            f"{len(s):,} chars. This is a TRACING limit "
            f"(EVOSKILL_SPAN_TEXT_LIMIT), not what the model received — set it "
            f"to 0 to record the full value.]"
        )
    return s


def start_tool_span(
    tracer: otel_trace.Tracer,
    parent: Span,
    *,
    tool_name: str,
    tool_input: Any = None,
) -> Span:
    span = tracer.start_span(f"tool.{tool_name}", context=set_span_in_context(parent))
    span.set_attribute("openinference.span.kind", "TOOL")
    span.set_attribute("tool.name", tool_name)
    if tool_input is not None:
        inp = span_text(tool_input)
        span.set_attribute("tool.input", inp)
        span.set_attribute("input.value", inp)
        span.set_attribute("input.mime_type", "application/json")
    return span


# ── Finalise root span (scoring, cost, output) ──────────────────────

def _msg_content(msg: dict[str, Any]) -> str:
    """Extract readable content from any message type, including tool calls."""
    content = msg.get("content", "") or ""
    tool_calls = msg.get("tool_calls")
    if tool_calls and not content:
        parts = []
        for tc in tool_calls:
            func = tc.get("function", {})
            name = func.get("name", "?")
            args = func.get("arguments", "")
            parts.append(f"[tool_call: {name}({args})]")
        content = "\n".join(parts)
    elif tool_calls and content:
        tc_summary = ", ".join(
            tc.get("function", {}).get("name", "?") for tc in tool_calls
        )
        content += f"\n[tool_calls: {tc_summary}]"
    return content


def _set_oi_messages(
    span: Span,
    prefix: str,
    messages: list[dict[str, Any]],
    max_per_msg: int | None = None,
) -> None:
    """Set OpenInference llm.input_messages / llm.output_messages on a span.

    Phoenix reads these to populate the Input/Output Messages panes.
    Handles all message types: system, user, assistant (with tool_calls), tool.

    Unlimited by default (`None` -> EVOSKILL_SPAN_TEXT_LIMIT, itself 0). The old
    30,000-per-message default was a silent trap for the standing "audit the
    evolver prompt at EVERY evolver call" directive: the dump tool reads exactly
    these attributes, so any prompt over 30K would have been audited in
    truncated form with nothing saying so. A 2-task batch renders ~23K, but a
    full 20-task batch does not.
    """
    for i, msg in enumerate(messages):
        role = msg.get("role", "unknown")
        content = _msg_content(msg)
        if not content:
            continue
        span.set_attribute(f"{prefix}.{i}.message.role", role)
        span.set_attribute(f"{prefix}.{i}.message.content", span_text(str(content), max_per_msg))


def finalise_root_span(
    tracer: otel_trace.Tracer,
    span: Span,
    eval_ctx: EvalContext,
    *,
    agent_answer: str = "",
    result_text: str = "",
    model: str = "",
    num_turns: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cost_usd: float = 0.0,
    system_prompt: str = "",
    messages: list[dict[str, Any]] | None = None,
    query: str = "",
    inference_provider: str = "",
    inline_score: float | None = None,
    budget_exhausted: bool = False,
) -> None:
    """Stamp final attributes on the AGENT root span + emit LLM cost child."""
    span.set_attribute("num_turns", num_turns)
    span.set_attribute("output.mime_type", "text/plain")

    # -- OpenInference Input/Output Messages (Phoenix Info tab) --
    # Cost + token accounting — compute FIRST so transcript can use it
    total_tokens = prompt_tokens + completion_tokens
    if cost_usd <= 0 and total_tokens > 0:
        cost_usd = estimate_cost(model, prompt_tokens, completion_tokens)

    # Eval: GT + agent answer
    if agent_answer:
        span.set_attribute("eval.agent_answer", agent_answer[:5000])
    gt = eval_ctx.ground_truth
    if gt:
        span.set_attribute("eval.ground_truth", gt)

    # In-span scoring
    scorer = eval_ctx.scorer
    current_name = getattr(span, "_name", None) or getattr(span, "name", "")
    _score_result: float | None = None
    _score_answer = agent_answer
    if agent_answer and "final_answer" in agent_answer:
        try:
            import re as _re_score
            _m = _re_score.search(r'"final_answer"\s*:\s*"([^"]*)"', agent_answer)
            if _m:
                _score_answer = _m.group(1)
        except Exception:
            pass

    # Prefer inline_score (async, computed by executor) over sync scorer callback
    _score: float | None = inline_score
    if _score is None and callable(scorer) and gt is not None and _score_answer:
        try:
            _score = float(scorer(query or "", _score_answer, gt))
        except Exception:
            pass

    if _score is not None:
        _score_result = _score
        _passed = _score >= 0.5
        _status = "OK" if _passed else "WRONG"
        span.set_attribute("eval.score", _score)
        span.set_attribute("eval.passed", _passed)
        span.update_name(f"{current_name} [{_status} {_score:.2f}] [{num_turns}T{' BUDGET' if budget_exhausted else ''}]")
        from opentelemetry.trace import StatusCode
        # description is only meaningful on ERROR — the OTEL SDK logs a
        # warning if set alongside OK, which fires on every genuine pass.
        if _passed:
            span.set_status(StatusCode.OK)
        else:
            span.set_status(StatusCode.ERROR, f"score={_score:.3f}")
    elif gt is not None and not agent_answer:
        _score_result = 0.0
        span.set_attribute("eval.score", 0.0)
        span.set_attribute("eval.passed", False)
        span.update_name(f"{current_name} [NO ANSWER] [{num_turns}T{' BUDGET' if budget_exhausted else ''}]")
        from opentelemetry.trace import StatusCode
        span.set_status(StatusCode.ERROR, "no answer")

    # Set llm.input_messages with ALL conversation messages.
    # LLM-kind spans render these as collapsible blocks in Phoenix.
    # Split assistant messages with tool_calls into separate blocks.
    all_msgs: list[dict[str, Any]] = []
    if system_prompt:
        all_msgs.append({"role": "system", "content": system_prompt})
    if messages:
        for m in messages:
            role = m.get("role", "")
            if role == "system" and system_prompt:
                continue  # already added above
            elif role == "assistant":
                reasoning = m.get("reasoning") or m.get("reasoning_content") or ""
                if not reasoning and isinstance(m.get("reasoning_details"), list):
                    reasoning = "\n".join(
                        d.get("text", "") for d in m["reasoning_details"]
                        if isinstance(d, dict) and d.get("text")
                    )
                if reasoning:
                    all_msgs.append({"role": "reasoning", "content": reasoning})
                text = m.get("content", "") or ""
                tc = m.get("tool_calls")
                if text:
                    all_msgs.append({"role": "assistant", "content": text})
                if tc:
                    for call in tc:
                        fn = call.get("function", {})
                        fname = fn.get("name", "?")
                        fargs = fn.get("arguments", "")
                        all_msgs.append({
                            "role": "tool_call",
                            "content": f"Tool: {fname}\n\n{fargs}",
                        })
            elif role == "tool":
                content = m.get("content", "") or ""
                all_msgs.append({"role": "tool", "content": content})
            else:
                content = m.get("content", "") or ""
                if content:
                    all_msgs.append({"role": role, "content": content})
    _set_oi_messages(span, "llm.input_messages", all_msgs)

    # Output: parse structured answer, show as clean summary block
    output_text = agent_answer or result_text
    gt_str = gt or "N/A"

    # Try to extract structured fields from the full result text (not truncated).
    # Search for JSON containing "final_answer" — ignore other JSON fragments.
    full_text = result_text or agent_answer or ""
    parsed_answer = ""
    parsed_reasoning = ""
    parsed_confidence = ""
    if full_text and "final_answer" in full_text:
        import re
        # Find all { ... } candidates that contain "final_answer"
        for m in re.finditer(r'\{[^{}]*final_answer[^{}]*\}', full_text):
            candidate = m.group(0)
            for attempt in (candidate, candidate.replace("'", '"')):
                try:
                    _parsed = json.loads(attempt)
                    if "properties" in _parsed:
                        _parsed = _parsed["properties"]
                    if "final_answer" in _parsed:
                        parsed_answer = str(_parsed.get("final_answer", ""))
                        parsed_reasoning = str(_parsed.get("reasoning", ""))
                        parsed_confidence = str(_parsed.get("confidence", ""))
                        break
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
            if parsed_answer:
                break

    _agent_name = span.attributes.get("agent.name", "") if hasattr(span, "attributes") else ""
    is_evolver = _agent_name == "evolver" or (gt is None and eval_ctx.scorer is None)
    display_answer = (parsed_answer or agent_answer[:200] or "(no answer)").replace("\n", " ").strip()

    if is_evolver:
        provider_row = f"| **Provider** | {inference_provider} |\n" if inference_provider else ""
        # Extract structured fields from evolver JSON for cleaner display
        _evo_action = _evo_skill = _evo_summary = _evo_justification = ""
        if agent_answer:
            try:
                _evo_parsed = json.loads(agent_answer)
                _evo_action = str(_evo_parsed.get("action", ""))
                _evo_skill = str(_evo_parsed.get("skill_name", ""))
                _evo_summary = str(_evo_parsed.get("summary", ""))
                _evo_justification = str(_evo_parsed.get("justification", ""))
            except (json.JSONDecodeError, TypeError):
                pass
        if _evo_action:
            evo_output = (
                f"| **Action** | {_evo_action} |\n"
                f"| **Skill** | {_evo_skill} |\n\n"
                f"**Summary:** {_evo_summary}\n\n"
                f"**Justification:** {_evo_justification}"
            )
        else:
            evo_output = agent_answer or "(no output)"
        summary = (
            f"| Field | Value |\n"
            f"|---|---|\n"
            f"| **Cost** | ${cost_usd:.6f} |\n"
            f"| **Tokens** | {prompt_tokens:,} prompt + {completion_tokens:,} completion |\n"
            f"| **Turns** | {num_turns} |\n"
            f"| **Model** | {model} |\n"
            f"{provider_row}"
            f"{evo_output}"
        )
    else:
        score_str = f"{_score_result:.2f}" if _score_result is not None else "N/A"
        provider_row = f"| **Provider** | {inference_provider} |\n" if inference_provider else ""
        summary = (
            f"| Field | Value |\n"
            f"|---|---|\n"
            f"| **Answer** | {display_answer} |\n"
            f"| **GT** | {gt_str} |\n"
            f"| **Score** | {score_str} |\n"
            f"| **Confidence** | {parsed_confidence or 'N/A'} |\n"
            f"| **Cost** | ${cost_usd:.6f} |\n"
            f"| **Tokens** | {prompt_tokens:,} prompt + {completion_tokens:,} completion |\n"
            f"| **Turns** | {num_turns} |\n"
            f"| **Model** | {model} |\n"
            f"{provider_row}\n"
            f"**Reasoning:** {parsed_reasoning or 'N/A'}\n\n"
            f"**Question:** {query or 'N/A'}"
        )
    span.set_attribute("output.value", summary)
    _set_oi_messages(span, "llm.output_messages", [
        {"role": "assistant", "content": summary},
    ])

    span.set_attribute("agent.total_cost_usd", cost_usd)
    # Do NOT set llm.cost.total on the AGENT root. Phoenix Total Cost sums
    # llm.cost.total across the whole trace (including AGENT parents), so
    # stamping the rollup here double-counts against turn LLM spans
    # (Jul 23 glm52 evolver: turns ~$1.84 + root $1.84 → UI Total ~$3.68).
    # Authoritative spend stays on turn spans (llm.cost.total / turn.cost_usd)
    # and on agent.total_cost_usd + the output.value Cost row for display.
    if model:
        span.set_attribute("agent.model", model)
    span.set_attribute("agent.tokens.prompt", prompt_tokens)
    span.set_attribute("agent.tokens.completion", completion_tokens)
    # Non-standard keys only — llm.token_count.* on a non-LLM root still
    # inflate cumulativeTokenCount* (own + children) in the Spans table.
