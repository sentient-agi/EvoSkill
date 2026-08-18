"""Independent OpenRouter-based cost tracking for tau2-bench episodes.

Runs ONLY inside the `_tau2_worker.py` subprocess, under tau2-bench's own
isolated venv -- this module imports tau2 internals and is never imported
from EvoSkill's own venv/process.

Why this exists: tau2's own `agent_cost`/`user_cost`/`Message.cost` fields
silently read 0.0 for every OpenRouter-routed model (confirmed empirically:
`litellm.completion_cost()` raises "This model isn't mapped yet" inside
`get_response_cost()`, `src/tau2/utils/llm_utils.py:119-131`, which
swallows the exception and returns 0.0 -- for gpt-4.1-2025-04-14,
deepseek/deepseek-v4-flash, and gpt-5-nano alike). This module patches
`generate` and reconstructs cost from each response's real token usage x a
static OpenRouter pricing table, so `final_answer["cost_usd"]` in the
solver's trace is a number that can actually be trusted.

Chokepoint (confirmed via `grep -rln "from tau2.utils.llm_utils import"
src/tau2`, then manually checking each hit's imported names): `generate` is
value-imported into exactly 3 modules on our call path (text mode,
evaluation_type=ALL, auto_review=False) --
`tau2.agent.llm_agent`, `tau2.user.user_simulator`,
`tau2.evaluator.evaluator_nl_assertions`. Patching
`tau2.utils.llm_utils.generate` alone is a no-op for episodes: each of
those 3 modules already bound its OWN local name `generate` at import
time, so the origin module's attribute being reassigned later does not
affect calls made through the importing modules' own bound names.
(`orchestrator.py`/`runner/batch.py` also match that grep but only import
`get_cost`/logging helpers, not `generate` -- verified, not a patch
target.)

Deviation from the original design doc: the design assumed tau2 would be
merged into EvoSkill's main venv and driven via `asyncio.to_thread` from a
shared thread pool, which is why it specified a `threading.local()` ledger
(a `ContextVar` can't carry data back OUT of a `to_thread` call -- the
context is copied into the thread, not shared). Here, each task instead
gets its own OS subprocess (one `run_single_task` call per process
invocation, via `_tau2_worker.py`), so there is no concurrent-access hazard
within a process to guard against -- a plain module-level list is a
sufficient ledger.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

_LEDGER: list[dict[str, Any]] = []

# ---------------------------------------------------------------------------
# CLEAN TIME, per role. the maintainer's directive 2026-08-06: "let's make clean time of
# agent turns only the metrics".
#
# WHY NOT WALL-CLOCK. The 0806 base train leg killed 7/60 episodes (11.7%) on a
# 1800s total-DURATION wall while the 300s inactivity watchdog never fired once.
# Duration counts the user simulator's latency, tool/env time, and host queueing
# against the agent -- so at ~10x concurrency a slow-but-progressing episode dies
# on the clock and is scored 0, which biases the baseline DOWN and inflates any
# later gain. Clean time counts ONLY the seconds spent inside the agent's own
# `generate()` calls, so it is contention-immune BY CONSTRUCTION: adding load
# cannot consume the agent's budget.
#
# This wrapper is the right place because it already brackets every generate()
# call and receives tau2's own `call_name` -- "agent_response",
# "user_simulator_response", "nl_assertions_eval" -- so attribution is exact
# rather than inferred.
_CLEAN_S: dict[str, float] = {}
_CALLS: dict[str, int] = {}


def reset_clean_time() -> None:
    _CLEAN_S.clear()
    _CALLS.clear()


def clean_time_s(call_name: str = "agent_response") -> float:
    """Seconds spent INSIDE that role's LLM calls this episode."""
    return _CLEAN_S.get(call_name, 0.0)


def call_count(call_name: str = "agent_response") -> int:
    """Completed calls for that role -- the AGENT's turn count when it is the agent."""
    return _CALLS.get(call_name, 0)


def clean_time_report() -> dict[str, Any]:
    """Everything needed to audit, and later re-calibrate, the budget.

    Recorded on EVERY episode, not just capped ones: the 1200s budget was set
    from an ESTIMATED 50-70% agent share of wall time, because no per-role
    timing existed when it was chosen. These fields are what replace that
    estimate with a measurement.
    """
    return {
        "agent_clean_s": round(_CLEAN_S.get("agent_response", 0.0), 3),
        "user_sim_clean_s": round(_CLEAN_S.get("user_simulator_response", 0.0), 3),
        "judge_clean_s": round(_CLEAN_S.get("nl_assertions_eval", 0.0), 3),
        "agent_calls": _CALLS.get("agent_response", 0),
        "user_sim_calls": _CALLS.get("user_simulator_response", 0),
    }

# USD per token (NOT per-million-token), sourced from OpenRouter's
# GET /api/v1/models `pricing.prompt` / `pricing.completion` fields,
# snapshotted 2026-07-10. ":nitro"/":floor" OpenRouter routing suffixes are
# stripped before lookup -- OpenRouter does not expose separate pricing
# for the throughput/price-routed variant of a model, so this is the base
# model's advertised per-token price: an approximation of, not identical
# to, the actual routed-provider's real price. Add an entry here for every
# new model string used as agent/user-sim/judge before running it --
# `price_usd` raises on an unknown model rather than silently reporting
# $0.00 (the exact bug this module exists to route around).
_PRICING_USD_PER_TOKEN: dict[str, tuple[float, float]] = {
    # NOTE 2026-08-04: OpenRouter now prices the UNDATED id at
    # (0.00000014, 0.00000028) -- this entry holds the -0731 SNAPSHOT's price,
    # so anything routed to the undated id under-reports cost by ~1.56x. Left
    # as-is deliberately: correcting it changes how legacy runs' costs compare,
    # which is the maintainer's call, not a silent edit. Verified against
    # GET /api/v1/models on 2026-08-04.
    "deepseek/deepseek-v4-flash": (0.00000009, 0.00000018),
    # The DATED SNAPSHOT is a distinct model id and needs its own entry:
    # `_strip_routing_suffix` removes `openrouter/` and `:nitro`/`:floor` but
    # deliberately NOT a date suffix, because a snapshot can be priced
    # differently from the moving id -- and here it is. Without this row,
    # tau3's standing DSV4F pin made `price_usd` return its NaN sentinel on
    # EVERY agent response, which (a) NaN-poisoned every [COST] rollup to
    # `$nan` and (b) pushed 76 copies of the same warning string into the
    # evolver's user prompt -- 20,368 of 34,765 chars, 58.6% of the whole
    # message. Price from GET /api/v1/models, fetched 2026-08-04.
    "deepseek/deepseek-v4-flash-0731": (0.00000009, 0.00000018),
    # Confirmed missing 2026-07-12: the DSV4P solver relaunch's every
    # agent-response cost NaN'd (loudly, per this module's own design --
    # see price_usd's docstring) because this entry didn't exist, poisoning
    # every downstream [COST] rollup line (base/midgate/train/val all showed
    # $nan, only the evolver's separate OpenRouter-billed cost stayed
    # readable). Price from GET /api/v1/models, fetched 2026-07-12.
    "deepseek/deepseek-v4-pro": (0.000000435, 0.00000087),
    # #207, added 2026-08-08. Both are tau3 3-arm-ablation SOLVERS (Arm B and
    # Arm C). Neither had an entry, so `price_usd` returned its NaN sentinel on
    # EVERY agent response and `total_cost_usd` came out NaN in 300/300 Arm B
    # records and 300/300 Arm C records -- base and skills-v1 alike. The
    # sentinel worked exactly as designed (loud, not a silent $0.00); what was
    # missing was these two rows. Consequence while missing: neither arm's
    # solver spend was attributable at all, and the org key is not a management
    # key so /api/v1/activity (403) could not substitute. Prices from
    # GET /api/v1/models, fetched 2026-08-08.
    "nex-agi/nex-n2-mini": (0.000000025, 0.0000001),
    "inclusionai/ling-3.0-flash": (0.000000021, 0.000000063),
    "z-ai/glm-5.2": (0.00000077, 0.00000242),
    "openai/gpt-4.1": (0.000002, 0.000008),
    "openai/gpt-4.1-mini": (0.0000004, 0.0000016),
    "openai/gpt-4.1-nano": (0.0000001, 0.0000004),
    "openai/gpt-5-nano": (0.00000005, 0.0000004),
}


def _strip_routing_suffix(model: str) -> str:
    base = model[len("openrouter/"):] if model.startswith("openrouter/") else model
    for suffix in (":nitro", ":floor"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base


def price_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    base = _strip_routing_suffix(model)
    if base not in _PRICING_USD_PER_TOKEN:
        raise ValueError(
            f"No pricing entry for model '{model}' (looked up as '{base}'). "
            f"Add it to _PRICING_USD_PER_TOKEN in cost_tracking.py before "
            f"using this model -- refusing to silently report $0.00."
        )
    prompt_price, completion_price = _PRICING_USD_PER_TOKEN[base]
    return prompt_tokens * prompt_price + completion_tokens * completion_price


def reset_ledger() -> None:
    _LEDGER.clear()


def read_ledger() -> list[dict[str, Any]]:
    return list(_LEDGER)


# Confirmed live 2026-07-12 via Phoenix (161/1502 = 10.7% of episodes across
# base/iter-1..4, both train+val): DSV4P (reasoning model, via OpenRouter)
# occasionally returns a response with NEITHER text content NOR tool_calls
# -- only internal reasoning, nothing surfaced. tau2's own orchestrator.py
# unconditionally calls `agent_msg.validate()` right after generate() returns
# (data_model/message.py's Message.validate(), which requires has_content()
# or is_tool_call()) and raises an unrecoverable ValueError otherwise,
# crashing the whole multi-turn episode. This is NOT caught by
# `is_provider_failure()` (src/harness/agent.py's _PROVIDER_FAILURE_MARKERS
# doesn't match "ValueError: AssistantMessage must have either content or
# tool_calls"), so every hit was permanently scored reward=0.0 with zero
# retry -- confirmed non-deterministic per task (e.g. task_071 crashed on
# iter-1/3/4 but produced a normal real response on base/iter-2), meaning a
# same-task retry would very likely succeed. Retrying HERE (one LLM call)
# rather than widening tau2's task-level retry eligibility is deliberate:
# a crash on turn 15 of 20 would otherwise force re-running the entire
# episode from scratch to get one more attempt at the single bad turn.
_EMPTY_RESPONSE_MAX_RETRIES = 2  # up to 3 total attempts per generate() call


def _is_empty_response(msg) -> bool:
    """Mirrors tau2's own Message.has_content()/is_tool_call() checks
    (data_model/message.py) -- True means `agent_msg.validate()` /
    `user_msg.validate()` would raise in orchestrator.py.
    """
    has_text = bool(msg.content and msg.content.strip())
    has_tool_calls = bool(getattr(msg, "tool_calls", None))
    return not (has_text or has_tool_calls)


def _wrap(real_generate: Callable) -> Callable:
    def wrapped(model, messages, tools=None, tool_choice=None, call_name=None, **kwargs):
        attempt = 0
        while True:
            # PER-TURN HEARTBEAT for the solver's inactivity watchdog (tau3 parity with
            # SSB v2's src/harness/liveness.py). This wrapper is the ONLY per-turn hook
            # available: tau2 runs its whole orchestrator loop inside `run_single_task`,
            # so there is no turn callback, and the worker's stdout is a single framed
            # payload read via `proc.communicate()` — streaming it instead would mean
            # rewriting the pipe plumbing, whose classic failure is deadlocking when one
            # pipe fills while the other is being read. A file mtime crosses the process
            # boundary with none of that risk.
            # Touched BEFORE the call as well as after, so a turn that hangs INSIDE the
            # LLM call still shows the watchdog that work had begun -- the alternative
            # (touch only on return) makes a slow call indistinguishable from a dead one,
            # which is the exact confusion this whole change exists to remove.
            _hb = os.environ.get("EVOSKILL_TAU2_HEARTBEAT_FILE")
            if _hb:
                try:
                    Path(_hb).write_text(f"{time.time():.3f} {call_name or 'call'}\n")
                except OSError:
                    pass          # a heartbeat that cannot be written must not kill a run
            # CLEAN TIME is measured around the call itself, so it excludes
            # queueing, the other role's latency, and everything the host is
            # doing. monotonic(), not time(), because a clock step must not be
            # able to hand an episode extra budget or steal it.
            _t_call = time.monotonic()
            try:
                msg = real_generate(model, messages, tools=tools, tool_choice=tool_choice, call_name=call_name, **kwargs)
            finally:
                # `finally`, so a failed/retried call still costs its real time.
                # Charging only successful calls would let a pathological retry
                # loop run unbounded inside a budget meant to bound exactly that.
                _k = call_name or "call"
                _CLEAN_S[_k] = _CLEAN_S.get(_k, 0.0) + (time.monotonic() - _t_call)
            _CALLS[_k] = _CALLS.get(_k, 0) + 1
            if _hb:
                try:
                    Path(_hb).write_text(f"{time.time():.3f} {call_name or 'call'}:done\n")
                except OSError:
                    pass
            usage = msg.usage or {}
            prompt_tokens = usage.get("prompt_tokens", 0) or 0
            completion_tokens = usage.get("completion_tokens", 0) or 0
            try:
                cost = price_usd(model, prompt_tokens, completion_tokens)
            except ValueError as e:
                # Loud, not silent: printed to stderr (visible in the worker's
                # subprocess stderr, which the solver surfaces on failure) AND
                # recorded as an explicit NaN sentinel in the ledger rather
                # than a $0.00 that would look like a real, trustworthy zero.
                print(f"[cost_tracking] {call_name}: {e}", file=sys.stderr)
                cost = float("nan")
            msg.cost = cost  # belt-and-suspenders repair of tau2's own
                             # agent_cost/user_cost aggregation (get_cost(),
                             # src/tau2/utils/llm_utils.py:472-490) -- a free
                             # side-effect fix, but NOT the source of truth EvoSkill
                             # reads; read_ledger() is, per the brief's explicit
                             # instruction not to trust tau2's own cost fields
                             # even after this patch.
            entry: dict[str, Any] = {
                "call_name": call_name,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cost_usd": cost,
            }
            if call_name == "nl_assertions_eval":
                # The NL-assertion judge call is the one `generate()` call this
                # module intercepts that Phoenix needs FULL prompt/response
                # content for (to build a judge span, see solver.py's
                # `_build_success_trace`) -- agent/user turns already carry
                # their own content back via `_tau2_worker._messages_to_turn_log`
                # (much cheaper, since it doesn't re-serialize the whole running
                # transcript on every turn). Judge calls happen at most once per
                # episode, so capturing the full input `messages` + raw output
                # here is cheap and doesn't blow up the ledger's JSON payload.
                entry["prompt_messages"] = [
                    {"role": m.role, "content": m.content} for m in messages
                ]
                entry["response_content"] = msg.content
                entry["generation_time_seconds"] = msg.generation_time_seconds
            _LEDGER.append(entry)  # every attempt is real spend -- record all, not just the last

            if not _is_empty_response(msg):
                if attempt > 0:
                    print(f"[cost_tracking] {call_name}: empty response recovered after {attempt} retry/retries", file=sys.stderr)
                return msg
            if attempt >= _EMPTY_RESPONSE_MAX_RETRIES:
                print(f"[cost_tracking] {call_name}: empty response persisted after {attempt} retries, giving up", file=sys.stderr)
                return msg
            attempt += 1

    wrapped._evoskill_wrapped = True  # idempotency marker, see install_cost_tracking
    return wrapped


def install_cost_tracking() -> None:
    """Monkeypatch `generate` in the 3 modules that import it by value.

    Must be called after those 3 modules have been imported (they will
    have been, transitively, by the time `run_single_task` is reachable).
    Idempotent -- safe to call more than once in the same process.
    """
    import tau2.agent.llm_agent as _agent_mod
    import tau2.evaluator.evaluator_nl_assertions as _judge_mod
    import tau2.user.user_simulator as _user_mod

    for mod in (_agent_mod, _user_mod, _judge_mod):
        if getattr(mod.generate, "_evoskill_wrapped", False):
            continue
        mod.generate = _wrap(mod.generate)
