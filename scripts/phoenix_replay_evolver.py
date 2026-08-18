#!/usr/bin/env python3
"""Replay an isolated `claude -p` evolver's stream-json log into Phoenix as spans.

WHY A REPLAY AND NOT LIVE INSTRUMENTATION. The evolver is a headless top-level
Claude Code process, deliberately so: an Agent-tool subagent lacks `Workflow` and
`TaskCreate`, so only a real session gives the evolver the full toolset. Nothing in
our harness sits in that process, so there is no place to hang a tracer. Claude
Code's own OTEL support exports METRICS and LOG EVENTS, not spans — useful for
token/cost counters, useless for a trace tree — while Phoenix is span-native. So
the evolver was a black box in Phoenix while every solver rollout beside it was
fully traced.

The stream-json log closes that gap: its `assistant` and `user` events carry
`timestamp`, `uuid`, and `parent_tool_use_id`, which is exactly enough to
reconstruct turns, tool calls, and nested subagent trees after the fact. Same
approach already used for Harbor subprocesses, whose spans are likewise lost.

    python3 scripts/phoenix_replay_evolver.py \
        --log outputs/ssbv2_evo_0801/evolver_e1.log \
        --project ssbv2-evo0803-e1-evolver

ONE-SHOT BY DESIGN. Spans are immutable once exported, so an incremental emitter
would either duplicate spans or need to rebuild parent contexts from a state file.
Run it when the evolver finishes; pass `--project ...-partial` for a mid-flight
look at a still-running evolver, into a throwaway project.

FIDELITY RULE: nothing here is truncated, and nothing is invented. Per-turn INPUT
context is not present in the stream record — only outputs and usage are — so turn
spans carry an explicit `evolver.input_not_in_stream_record` attribute rather than a
reconstructed prompt that would look authentic and not be. The one real input, the
rendered prompt, is attached verbatim to the root.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from datetime import datetime
from pathlib import Path

from openinference.semconv.trace import SpanAttributes as SA
from opentelemetry import trace as otel_trace
from opentelemetry.trace import StatusCode
from phoenix.otel import register as phoenix_register


def iso_to_ns(ts: str | None) -> int | None:
    if not ts:
        return None
    try:
        return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1e9)
    except ValueError:
        return None


def blocks_text(content) -> str:
    """Flatten a message's content blocks to text, keeping thinking visible.

    Thinking is labelled rather than dropped: for an evolver, the reasoning IS the
    artifact under review, and a trace that hides it cannot answer the question the
    orchestrator actually asks of it ("did it reason correctly from the evidence?").
    """
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--project", required=True)
    ap.add_argument("--endpoint", default="http://localhost:6006")
    ap.add_argument("--prompt", help="rendered prompt file, attached verbatim to the root")
    ap.add_argument("--trace-json", help="the launcher's evolver_eN.trace.json sidecar")
    # --role (2026-08-10). This replay was written for the evolver and hardcoded the string
    # "evolver" into the root label AND every attribute key. Pointed at the AO's authoring
    # log it produced a trace labelled "evolver | 13 turns", i.e. it named the ORCHESTRATOR
    # as the OPTIMIZER — the single distinction an AO audit exists to make. Default is
    # "evolver", so all existing invocations are byte-identical.
    ap.add_argument("--role", default="evolver",
                    help="component being replayed; sets the root label and attribute prefix")
    a = ap.parse_args()

    log = Path(a.log)
    if not log.exists():
        print(f"no such log: {log}", file=sys.stderr)
        return 5

    events = []
    for line in log.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue          # a partially-written last line on a live log
    if not events:
        print("log has no parseable events", file=sys.stderr)
        return 5

    sidecar = {}
    if a.trace_json and Path(a.trace_json).exists():
        sidecar = json.loads(Path(a.trace_json).read_text())

    init = next((e for e in events if e.get("type") == "system" and e.get("cwd")), {})
    result = next((e for e in reversed(events) if e.get("type") == "result"), {})
    turns = [e for e in events if e.get("type") == "assistant"]
    users = [e for e in events if e.get("type") == "user"]

    # tool_use_id -> the user event carrying its result, for pairing call to return.
    result_by_tuid: dict[str, dict] = {}
    for u in users:
        for c in (u.get("message", {}) or {}).get("content", []) or []:
            if isinstance(c, dict) and c.get("type") == "tool_result":
                result_by_tuid[c.get("tool_use_id")] = u

    times = [iso_to_ns(e.get("timestamp")) for e in events]
    times = [t for t in times if t]
    if not times:
        print("no timestamps in log — cannot place spans in time", file=sys.stderr)
        return 5
    t0, t1 = min(times), max(times)

    tp = phoenix_register(project_name=a.project, endpoint=f"{a.endpoint}/v1/traces",
                          set_global_tracer_provider=False, batch=True)
    tracer = tp.get_tracer(f"{a.role}-replay")

    cost = result.get("total_cost_usd")
    n_err = 0
    label = f"{a.role} | {len(turns)} turns"
    if cost is not None:
        label += f" | ${cost:.2f}"
    if not result:
        label += " | INCOMPLETE (no result event — still running or died)"

    root = tracer.start_span(label, start_time=t0)
    root.set_attribute(SA.OPENINFERENCE_SPAN_KIND, "CHAIN")
    for k, v in {
        f"{a.role}.session_id": init.get("session_id") or sidecar.get("session_id"),
        f"{a.role}.model": init.get("model"),
        f"{a.role}.cwd": init.get("cwd"),
        f"{a.role}.epoch": sidecar.get("epoch"),
        f"{a.role}.digests": sidecar.get("digests"),
        f"{a.role}.staging": sidecar.get("staging"),
        f"{a.role}.n_digests": sidecar.get("n_digests"),
        f"{a.role}.n_turns": len(turns),
        f"{a.role}.n_tool_calls": len(result_by_tuid),
        f"{a.role}.stream_log": str(log),
        f"{a.role}.cost_usd": cost,
        f"{a.role}.completed": bool(result),
    }.items():
        if v is not None:
            root.set_attribute(k, v)
    if a.prompt and Path(a.prompt).exists():
        p = Path(a.prompt).read_text()
        root.set_attribute(SA.INPUT_VALUE, p)           # verbatim, never clipped
        root.set_attribute(f"{a.role}.prompt_chars", len(p))
    if result.get("result"):
        root.set_attribute(SA.OUTPUT_VALUE, str(result["result"]))

    # ---- TURN LABELS AND SPAN GEOMETRY, computed before anything is emitted.
    #
    # TWO DEFECTS THIS FIXES, both visible in the first real trace.
    #
    # 1. A GLOBAL TURN COUNTER LIES ABOUT SUBAGENTS. Measured on the 2026-08-01
    #    evolver: 282 of 403 turns belonged to 6 subagents, and the turn order
    #    switched parent 200 times — subagent turns INTERLEAVE, because agents spawn
    #    agents and the streams merge. So "turn 41" and "turn 86" can be different
    #    agents at unrelated points in their own work, and reading them as a
    #    sequence invites exactly the wrong conclusion about what happened when.
    #    Labels are therefore hierarchical and LOCAL: the 3rd turn of the agent
    #    spawned by main turn 40 is "40-3", and a nested agent's is "40-3-2".
    #
    # 2. A PARENT SPAN MUST CONTAIN ITS CHILDREN. The Agent tool span was ended at
    #    its tool_result timestamp, which for a spawning call arrives before the
    #    subagent's turns are placed — Phoenix rendered `Agent` at 9ms with an
    #    887ms child inside it. Latency on a parent is meaningless if the subtree
    #    outlives it, so a tool span now ends at the later of its own result and
    #    its last descendant.
    turn_label: dict[int, str] = {}          # index in `turns` -> label
    label_of_tuid: dict[str, str] = {}       # tool_use_id -> label of issuing turn
    local_counter: collections.Counter = collections.Counter()
    main_n = 0
    for i, ev in enumerate(turns):
        ptu = ev.get("parent_tool_use_id")
        if ptu and ptu in label_of_tuid:
            local_counter[ptu] += 1
            turn_label[i] = f"{label_of_tuid[ptu]}-{local_counter[ptu]}"
        elif ptu:
            # Spawned by a call we never saw (log truncated at the head). Keep it
            # distinguishable rather than silently numbering it as a main turn.
            local_counter[ptu] += 1
            turn_label[i] = f"?-{local_counter[ptu]}"
        else:
            main_n += 1
            turn_label[i] = str(main_n)
        for c in (ev.get("message", {}) or {}).get("content", []) or []:
            if isinstance(c, dict) and c.get("type") == "tool_use":
                label_of_tuid[c["id"]] = turn_label[i]

    # Latest end time of everything descended from each tool_use_id.
    subtree_end: dict[str, int] = {}
    for i, ev in enumerate(turns):
        ptu = ev.get("parent_tool_use_id")
        if not ptu:
            continue
        t = iso_to_ns(ev.get("timestamp"))
        nxt = next((iso_to_ns(e.get("timestamp"))
                    for e in events[events.index(ev) + 1:] if iso_to_ns(e.get("timestamp"))), None)
        for cand in (t, nxt):
            if cand and cand > subtree_end.get(ptu, 0):
                subtree_end[ptu] = cand

    # parent_tool_use_id -> the tool span that spawned it, so subagent turns nest
    # under the Task/Agent call that created them instead of flattening to the root.
    tool_span_by_tuid: dict[str, object] = {}

    for i, ev in enumerate(turns):
        msg = ev.get("message", {}) or {}
        start = iso_to_ns(ev.get("timestamp")) or t0
        # A turn's end is not recorded; the next event's start is the closest honest
        # bound. Better than a fabricated duration, and stated in an attribute.
        end = next((iso_to_ns(e.get("timestamp")) for e in events[events.index(ev) + 1:]
                    if iso_to_ns(e.get("timestamp"))), t1)

        parent = tool_span_by_tuid.get(ev.get("parent_tool_use_id"))
        ctx = otel_trace.set_span_in_context(parent or root)
        nested = " (subagent)" if ev.get("parent_tool_use_id") else ""
        span = tracer.start_span(f"turn {turn_label[i]}{nested}", context=ctx,
                                 start_time=start)
        span.set_attribute(f"{a.role}.turn_label", turn_label[i])
        if ev.get("parent_tool_use_id"):
            span.set_attribute(f"{a.role}.spawned_by_turn",
                              label_of_tuid.get(ev["parent_tool_use_id"], "?"))
            span.set_attribute(f"{a.role}.agent_tool_use_id", ev["parent_tool_use_id"])
        span.set_attribute(SA.OPENINFERENCE_SPAN_KIND, "LLM")
        if msg.get("model"):
            span.set_attribute(SA.LLM_MODEL_NAME, msg["model"])
        txt = blocks_text(msg.get("content"))
        if txt:
            span.set_attribute(SA.OUTPUT_VALUE, txt)
            span.set_attribute(f"{SA.LLM_OUTPUT_MESSAGES}.0.message.role", "assistant")
            span.set_attribute(f"{SA.LLM_OUTPUT_MESSAGES}.0.message.content", txt)
        span.set_attribute(
            f"{a.role}.input_not_in_stream_record",
            "The stream-json record carries only this turn's OUTPUT and usage. The "
            "conversation prefix the model actually saw is not in it, so no input "
            "messages are attached here rather than showing a reconstruction. The "
            "run's one real input, the rendered prompt, is on the root span.")
        u = msg.get("usage") or {}
        for key, attr in (("input_tokens", SA.LLM_TOKEN_COUNT_PROMPT),
                          ("output_tokens", SA.LLM_TOKEN_COUNT_COMPLETION)):
            if u.get(key) is not None:
                span.set_attribute(attr, u[key])
        if u.get("input_tokens") is not None and u.get("output_tokens") is not None:
            span.set_attribute(SA.LLM_TOKEN_COUNT_TOTAL,
                               u["input_tokens"] + u["output_tokens"])

        for c in msg.get("content") or []:
            if not (isinstance(c, dict) and c.get("type") == "tool_use"):
                continue
            tuid = c.get("id")
            res_ev = result_by_tuid.get(tuid)
            tstart = start
            tend = iso_to_ns((res_ev or {}).get("timestamp")) or end
            # A spawning call must outlive its subtree — see note above.
            tend = max(tend, subtree_end.get(tuid, 0))
            tspan = tracer.start_span(
                f"{c.get('name')}", context=otel_trace.set_span_in_context(span),
                start_time=tstart)
            tspan.set_attribute(SA.OPENINFERENCE_SPAN_KIND, "TOOL")
            tspan.set_attribute(SA.TOOL_NAME, str(c.get("name")))
            tspan.set_attribute(SA.INPUT_VALUE, json.dumps(c.get("input", {}), indent=1))
            tspan.set_attribute(SA.INPUT_MIME_TYPE, "application/json")
            if res_ev:
                body = None
                for rc in (res_ev.get("message", {}) or {}).get("content", []) or []:
                    if isinstance(rc, dict) and rc.get("tool_use_id") == tuid:
                        body = blocks_text([rc])
                        # Role "tool", not "tool_result" — the latter renders as an
                        # unstyled unknown role in Phoenix.
                        tspan.set_attribute(
                            f"{SA.LLM_INPUT_MESSAGES}.0.message.role", "tool")
                        if rc.get("is_error"):
                            tspan.set_status(StatusCode.ERROR, "tool reported is_error")
                if body is not None:
                    tspan.set_attribute(SA.OUTPUT_VALUE, body)
            else:
                tspan.set_status(StatusCode.ERROR, "no tool_result in log (truncated?)")
            tspan.end(end_time=tend)
            tool_span_by_tuid[tuid] = tspan

        span.end(end_time=end)

    if result.get("is_error") or (result.get("subtype") not in (None, "success")):
        root.set_status(StatusCode.ERROR, str(result.get("subtype")))
        n_err += 1
    root.end(end_time=t1)
    tp.force_flush()

    print(f"replayed {log.name} -> project {a.project!r}")
    print(f"  turns={len(turns)} tool_calls={len(result_by_tuid)} "
          f"subagent_turns={sum(1 for e in turns if e.get('parent_tool_use_id'))}")
    print(f"  cost={'$%.2f' % cost if cost is not None else 'n/a (no result event)'}"
          f"  span_errors={n_err}")
    print(f"  view: {a.endpoint}/projects")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
