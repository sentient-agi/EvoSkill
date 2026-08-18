#!/usr/bin/env python3
"""Tee an evolver's stream-json to its log AND emit Phoenix spans LIVE.

WHY THIS EXISTS ALONGSIDE THE REPLAY. `phoenix_replay_evolver.py` reconstructs a
finished evolver's trace after the fact, which is enough to audit an iteration but
not to watch one: for ~18 minutes the only visible signal is a growing log. This
sits in the pipe instead, so each span appears in Phoenix the moment the unit it
describes is complete.

    claude -p "$PROMPT" ... --output-format stream-json --verbose \
      | python3 scripts/phoenix_stream_evolver.py --log "$LOG" --project ssbv2-evo-e3

THE LOG IS WRITTEN VERBATIM, byte for byte, before anything else happens to a line.
Everything downstream — the replay script, the infra scans, the cost accounting —
reads that file, so this must not become a second, subtly different record. If span
emission raises, the tee keeps going and the exception is reported to stderr: losing
observability is acceptable, losing the evolver's only durable record is not.

SPAN LIFETIMES, which are the whole difficulty of doing this live:
  * a TURN span closes when the next event arrives (its end is not in the record, so
    the next event's timestamp is the closest honest bound);
  * a TOOL span closes when its tool_result arrives — EXCEPT for a spawning call
    (Agent/Task), which must outlive the subagent turns nested inside it, so it
    closes at max(result_ts, last_child_end);
  * the ROOT closes at EOF. OTel exports a span when it ENDS, so children reach
    Phoenix before their parent and the trace fills in top-down late. That is a
    display quirk, not a loss — and it is the price of live emission.

Turn labels are hierarchical (`40-3` = 3rd turn of the agent spawned by main turn
40). A global counter is actively misleading here: measured on the 2026-08-01
evolver, 6 subagents ran CONCURRENTLY and their turns interleaved, so consecutive
global numbers belonged to different agents at unrelated points in their own work.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from datetime import datetime
from pathlib import Path

# TRACING IMPORTS ARE OPTIONAL. This process is first a TEE — the verbatim stream log is
# the load-bearing artifact every downstream audit reads — and only second a Phoenix
# emitter. A module-scope import of the tracing stack made the tee die at import on any
# machine without the `phoenix` extra, and because `claude` writes stream-json into this
# process's stdin, a dead tee SIGPIPEs the evolver itself: missing observability killing
# the run it observes. main() already degrades to tee-only when the tracer cannot
# construct; these guards make a missing package reach that same path.
try:
    from openinference.semconv.trace import SpanAttributes as SA
    from opentelemetry import trace as otel_trace
    from opentelemetry.trace import StatusCode
    from phoenix.otel import register as phoenix_register
    _TRACING_IMPORT_ERROR = None
except ImportError as _e:
    SA = otel_trace = StatusCode = phoenix_register = None
    _TRACING_IMPORT_ERROR = _e


def ns(ts: str | None) -> int | None:
    if not ts:
        return None
    try:
        return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1e9)
    except ValueError:
        return None


def blocks_text(content) -> str:
    if isinstance(content, str):
        return content
    out = []
    for c in content or []:
        t = c.get("type")
        if t == "text":
            out.append(c.get("text", ""))
        elif t == "thinking":
            # claude-code >=2.1.8 blanks thinking text in stream-json AND in its own
            # session transcript (anthropics/claude-code#20127) — only the signature
            # survives, so there is nothing to render and nothing to backfill from.
            # Say so, or every reader re-debugs this tee (verified empty 23/23 on e1).
            body = c.get("thinking") or ""
            out.append("[thinking]\n" + body if body.strip() else
                       "[thinking — text withheld by claude-code stream-json "
                       ">=2.1.8 (issue #20127); signature only]")
        elif t == "tool_use":
            out.append(f"[tool_use {c.get('name')} {c.get('id')}]")
        elif t == "tool_result":
            body = c.get("content")
            if isinstance(body, list):
                body = "\n".join(x.get("text", "") for x in body if isinstance(x, dict))
            out.append(str(body or ""))
    return "\n".join(x for x in out if x)


class LiveTracer:
    def __init__(self, project: str, endpoint: str, prompt: str | None, epoch: str | None):
        if _TRACING_IMPORT_ERROR is not None:
            raise ImportError(f"tracing deps unavailable ({_TRACING_IMPORT_ERROR}); "
                              f"install them with: uv sync --extra phoenix")
        tp = phoenix_register(project_name=project, endpoint=f"{endpoint}/v1/traces",
                              set_global_tracer_provider=False, batch=True)
        self.tp = tp
        self.tracer = tp.get_tracer("evolver-live")
        self.root = None
        self.prompt = prompt
        self.epoch = epoch
        self.open_turn = None            # (span, label)
        self.turn_label: dict[str, str] = {}   # tool_use_id -> issuing turn's label
        self.local = collections.Counter()     # parent tool_use_id -> child count
        self.main_n = 0
        self.tool_spans: dict[str, dict] = {}  # tool_use_id -> {span, end, spawning}
        self.n_turns = 0

    # -- root ---------------------------------------------------------------
    def ensure_root(self, ev: dict) -> None:
        if self.root is not None:
            return
        t = ns(ev.get("timestamp"))
        self.root = self.tracer.start_span(
            f"evolver e{self.epoch or '?'} | LIVE", start_time=t)
        self.root.set_attribute(SA.OPENINFERENCE_SPAN_KIND, "CHAIN")
        if ev.get("session_id"):
            self.root.set_attribute("evolver.session_id", ev["session_id"])
        if self.epoch:
            self.root.set_attribute("evolver.epoch", self.epoch)
        if self.prompt:
            self.root.set_attribute(SA.INPUT_VALUE, self.prompt)
            self.root.set_attribute("evolver.prompt_chars", len(self.prompt))

    # -- per-event ----------------------------------------------------------
    def handle(self, ev: dict) -> None:
        kind = ev.get("type")
        t = ns(ev.get("timestamp"))
        if kind in ("assistant", "user"):
            self.ensure_root(ev)
        # Close the previous turn at this event's timestamp: a turn's own end is not
        # in the record, and the next event is the tightest honest bound available.
        if self.open_turn is not None and t is not None:
            self.open_turn[0].end(end_time=t)
            self.open_turn = None

        if kind == "assistant":
            self._assistant(ev, t)
        elif kind == "user":
            self._tool_results(ev, t)
        elif kind == "result":
            self._finish(ev, t)

    def _label_for(self, ev: dict) -> str:
        ptu = ev.get("parent_tool_use_id")
        if ptu:
            self.local[ptu] += 1
            return f"{self.turn_label.get(ptu, '?')}-{self.local[ptu]}"
        self.main_n += 1
        return str(self.main_n)

    def _assistant(self, ev: dict, t: int | None) -> None:
        msg = ev.get("message", {}) or {}
        label = self._label_for(ev)
        ptu = ev.get("parent_tool_use_id")
        parent = (self.tool_spans.get(ptu, {}) or {}).get("span") or self.root
        span = self.tracer.start_span(
            f"turn {label}{' (subagent)' if ptu else ''}",
            context=otel_trace.set_span_in_context(parent), start_time=t)
        span.set_attribute(SA.OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute("evolver.turn_label", label)
        if msg.get("model"):
            span.set_attribute(SA.LLM_MODEL_NAME, msg["model"])
        txt = blocks_text(msg.get("content"))
        if txt:
            span.set_attribute(SA.OUTPUT_VALUE, txt)
            span.set_attribute(f"{SA.LLM_OUTPUT_MESSAGES}.0.message.role", "assistant")
            span.set_attribute(f"{SA.LLM_OUTPUT_MESSAGES}.0.message.content", txt)
        u = msg.get("usage") or {}
        if u.get("input_tokens") is not None:
            span.set_attribute(SA.LLM_TOKEN_COUNT_PROMPT, u["input_tokens"])
        if u.get("output_tokens") is not None:
            span.set_attribute(SA.LLM_TOKEN_COUNT_COMPLETION, u["output_tokens"])
        self.n_turns += 1

        # A spawning call's span must stay OPEN while its subagent streams.
        for c in msg.get("content") or []:
            if not (isinstance(c, dict) and c.get("type") == "tool_use"):
                continue
            tuid = c["id"]
            self.turn_label[tuid] = label
            tspan = self.tracer.start_span(
                str(c.get("name")), context=otel_trace.set_span_in_context(span),
                start_time=t)
            tspan.set_attribute(SA.OPENINFERENCE_SPAN_KIND, "TOOL")
            tspan.set_attribute(SA.TOOL_NAME, str(c.get("name")))
            tspan.set_attribute(SA.INPUT_VALUE, json.dumps(c.get("input", {}), indent=1))
            tspan.set_attribute(SA.INPUT_MIME_TYPE, "application/json")
            self.tool_spans[tuid] = {
                "span": tspan, "end": t,
                "spawning": c.get("name") in ("Agent", "Task")}
        # Track the open turn so the NEXT event can close it. Subagent turns are
        # tracked the same way; their interleaving is why the bound is per-event
        # rather than per-turn-index.
        self.open_turn = (span, label)

    def _tool_results(self, ev: dict, t: int | None) -> None:
        for c in (ev.get("message", {}) or {}).get("content", []) or []:
            if not (isinstance(c, dict) and c.get("type") == "tool_result"):
                continue
            rec = self.tool_spans.pop(c.get("tool_use_id"), None)
            if not rec:
                continue
            span = rec["span"]
            body = blocks_text([c])
            if body:
                span.set_attribute(SA.OUTPUT_VALUE, body)
                span.set_attribute(f"{SA.LLM_INPUT_MESSAGES}.0.message.role", "tool")
            if c.get("is_error"):
                span.set_status(StatusCode.ERROR, "tool reported is_error")
            # A parent must contain its children; children of a spawning call have
            # already advanced `rec["end"]` past the result in the nested case.
            span.end(end_time=max(t or rec["end"], rec["end"]))

    def _finish(self, ev: dict, t: int | None) -> None:
        if self.root is None:
            return
        cost = ev.get("total_cost_usd")
        if cost is not None:
            self.root.set_attribute("evolver.cost_usd", cost)
        self.root.set_attribute("evolver.n_turns", self.n_turns)
        if ev.get("result"):
            self.root.set_attribute(SA.OUTPUT_VALUE, str(ev["result"]))
        if ev.get("is_error") or ev.get("subtype") not in (None, "success"):
            self.root.set_status(StatusCode.ERROR, str(ev.get("subtype")))
        self.close(t)

    def close(self, t: int | None = None) -> None:
        if self.open_turn is not None:
            self.open_turn[0].end(end_time=t)
            self.open_turn = None
        for rec in self.tool_spans.values():
            rec["span"].set_status(StatusCode.ERROR, "no tool_result (stream ended)")
            rec["span"].end(end_time=t or rec["end"])
        self.tool_spans.clear()
        if self.root is not None:
            self.root.end(end_time=t)
            self.root = None
        self.tp.force_flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="verbatim stream-json log to write")
    ap.add_argument("--project", required=True)
    ap.add_argument("--endpoint", default="http://localhost:6006")
    ap.add_argument("--prompt", help="rendered prompt, attached to the root span")
    ap.add_argument("--epoch")
    a = ap.parse_args()

    prompt = None
    if a.prompt and Path(a.prompt).exists():
        prompt = Path(a.prompt).read_text()

    tr = None
    try:
        tr = LiveTracer(a.project, a.endpoint, prompt, a.epoch)
    except Exception as e:                        # noqa: BLE001
        print(f"[phoenix-stream] tracer init failed, tee-only: {e}", file=sys.stderr)

    n_bad = 0
    with open(a.log, "a", buffering=1) as log:
        for line in sys.stdin:
            log.write(line)                       # VERBATIM, and first
            if tr is None:
                continue
            s = line.strip()
            if not s.startswith("{"):
                continue
            try:
                tr.handle(json.loads(s))
            except Exception as e:                # noqa: BLE001
                n_bad += 1
                if n_bad <= 5:
                    print(f"[phoenix-stream] span emit failed: {type(e).__name__}: {e}",
                          file=sys.stderr)
    if tr is not None:
        try:
            tr.close()
        except Exception as e:                    # noqa: BLE001
            print(f"[phoenix-stream] close failed: {e}", file=sys.stderr)
    if n_bad:
        print(f"[phoenix-stream] {n_bad} line(s) failed span emission "
              f"(log is complete regardless)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
