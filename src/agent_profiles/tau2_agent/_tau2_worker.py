#!/usr/bin/env python3
"""tau2-bench episode worker -- runs ONE task under tau2-bench's own
isolated venv (`third_party/tau2-bench/.venv`), invoked by `solver.py` as:

    uv run --project <tau2_root> python <this file>

with the request JSON on stdin and the result JSON (one line) on stdout.

Reads a single JSON request, drives it through tau2's own orchestrator via
`run_single_task` (`src/tau2/runner/batch.py:336` -- the confirmed correct
integration point: builds via `build_orchestrator` then `run_simulation`;
we never touch `Orchestrator`/`run_simulation` directly), and writes back
the full `RewardInfo` sub-check breakdown, the converted message
transcript, and an independently-tracked cost ledger.

Deviation from the original design doc: the design recommended merging
tau2 directly into EvoSkill's own venv (ALFWorld precedent: `import
alfworld` straight from `episode.py`) and driving `run_single_task` via
`asyncio.to_thread`. This build instead keeps tau2-bench in its own
isolated venv and drives it via this subprocess worker -- see
`third_party/README.md`'s tau2-bench section and `solver.py`'s module
docstring for the full justification (tau2's `--all-extras` dependency
surface -- elevenlabs, deepgram-sdk, pyaudio, boto3,
google-cloud-aiplatform, gymnasium, scikit-learn, ... -- is irrelevant to
the text-mode retail domain and carries real conflict/bloat risk for the
production venv every other EvoSkill benchmark shares). Trade-off:
per-episode `uv run` startup overhead (~1-2s) instead of an in-process
call -- negligible next to a multi-turn tau2 episode's LLM round-trip
latency.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

# agent_wiring.py / cost_tracking.py live next to this file in the EvoSkill
# repo (not inside tau2-bench) but have zero EvoSkill-internal imports
# (only tau2 + stdlib), so they import cleanly once this directory is on
# sys.path -- this is the only bridge needed between the two venvs' code.
sys.path.insert(0, str(Path(__file__).parent))

import agent_wiring  # noqa: E402
# ABSOLUTE, not `from ._tau2_protocol import ...`. This file is executed as a SCRIPT
# (`uv run --project <tau2_root> python _tau2_worker.py`), so it has no parent
# package and a relative import raises ImportError before main() runs. Cost of
# getting this wrong, 2026-08-06: the relative form was copied in from solver.py —
# where it is correct, because THAT module is imported as part of a package — and the
# worker died with "exited 1 with no output" on its first episode. Both names come
# through this one line so the next addition cannot reintroduce the wrong form.
from _tau2_protocol import RESULT_MARKER, llm_call_budget  # noqa: E402
import cost_tracking  # noqa: E402
import tau2.evaluator.evaluator_nl_assertions as _nl_mod  # noqa: E402
from tau2.data_model.message import (  # noqa: E402
    AssistantMessage, MultiToolMessage, SystemMessage, ToolMessage, UserMessage,
)
from tau2.data_model.simulation import TextRunConfig  # noqa: E402
from tau2.evaluator.evaluator import EvaluationType  # noqa: E402
from tau2.runner.batch import run_single_task  # noqa: E402
from tau2.runner.helpers import get_tasks  # noqa: E402


def _messages_to_dicts(messages: list) -> list[dict]:
    """Convert tau2's own Message objects (SystemMessage | AssistantMessage
    | UserMessage | ToolMessage, `src/tau2/data_model/message.py`) into the
    plain `{"role":.., "content":..}` dict shape EvoSkill's
    `_render_dict_transcript` (`src/harness/agent.py`) knows how to render.
    """
    out: list[dict] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            continue  # shown via Tau2Solver._get_options()["system"], not duplicated
        if isinstance(m, ToolMessage):
            out.append({"role": "tool", "content": m.content or ""})
            continue
        d: dict = {"role": m.role, "content": m.content or ""}
        if m.tool_calls:
            d["tool_calls"] = [
                {"function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)}}
                for tc in m.tool_calls
            ]
            if m.role != "assistant":
                # _render_dict_transcript only renders tool_calls for
                # role == "assistant" -- fold user-side tool calls (dual
                # control domains, not exercised by retail today but kept
                # for forward compatibility) into content too, so a
                # dual-control task never silently drops them.
                extra = "\n".join(
                    f"[tool_call: {tc.name}({json.dumps(tc.arguments, ensure_ascii=False)})]"
                    for tc in m.tool_calls
                )
                d["content"] = f"{d['content']}\n{extra}" if d["content"] else extra
        out.append(d)
    return out


def _extract_system_prompt(messages: list) -> str | None:
    """Return the REAL system-prompt text tau2 fed the model this episode,
    read directly off the `SystemMessage` tau2 itself inserts into
    `simulation.get_messages()` -- the byte-exact text `EvoSkillLLMAgent`
    built from tau2's own `SYSTEM_PROMPT`/`AGENT_INSTRUCTION` constants
    (`agent_wiring.py`), NOT solver.py's `_build_system_prompt()`
    hand-transcribed copy (that module's own docstring flags it as
    "can drift" since it's not an import -- fine as a cache-key input, not
    good enough as the thing Phoenix displays as ground truth for what the
    model actually saw). Returns None if no SystemMessage is present
    (should not happen for a text-mode retail episode -- `EvoSkillLLMAgent`
    always seeds one -- guarded defensively rather than assumed).
    """
    for m in messages:
        if isinstance(m, SystemMessage) and m.content:
            return m.content
    return None


def _messages_to_turn_log(messages: list, agent_model: str, user_model: str) -> list[dict]:
    """Ordered per-event data for Phoenix tracing: one entry per message in
    `messages` (tau2's own causal order), EXCLUDING only `SystemMessage`
    (captured separately, see `_extract_system_prompt`) -- kept separate
    from `_messages_to_dicts` (transcript rendering) so that function's
    existing consumers/shape are untouched.

    Each entry carries `kind`:
      - "llm" for AssistantMessage (role="assistant", the solver) / UserMessage
        (role="user", the role-play/user-simulator) -- these carry a real
        `generate()` call each turn, and solver.py opens one Phoenix span per
        "llm" entry.
      - "tool" for ToolMessage/MultiToolMessage (tool-call results) -- no
        `generate()` call backs these, so they get no span of their own, but
        their content is still needed INLINE (in causal order) so that a
        later turn's reconstructed "history so far" faithfully includes the
        tool result the model actually saw, instead of silently skipping
        every tool round-trip (which `turn_log` used to do entirely, since
        it only kept the AssistantMessage/UserMessage subset).

    `reasoning_tokens` is pulled from `raw_data` (the full litellm response
    dict, always populated by tau2's own `generate()` -- see
    `AssistantMessage.raw_data`/`UserMessage.raw_data` in message.py) because
    tau2's own `get_response_usage()` (llm_utils.py:134-141) only keeps
    prompt_tokens/completion_tokens and discards the
    completion_tokens_details.reasoning_tokens sub-field that reasoning
    models (DeepSeek-V4, GLM 5.2) report.
    """

    def _tool_entry(tm: ToolMessage) -> dict:
        # `requestor` decides who is allowed to SEE this tool result. tau2's
        # `is_valid_user_history_message` admits a ToolMessage into the user
        # simulator's history ONLY when requestor == "user" (dual-control domains
        # where the user operates their own tools). Without this field the tracer
        # cannot reproduce that rule and shows the user sim documents it never saw.
        return {"kind": "tool", "role": "tool", "content": tm.content or "",
                "requestor": getattr(tm, "requestor", None)}

    out: list[dict] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            continue
        if isinstance(m, ToolMessage):
            out.append(_tool_entry(m))
            continue
        if isinstance(m, MultiToolMessage):
            out.extend(_tool_entry(tm) for tm in m.tool_messages)
            continue
        if not isinstance(m, (AssistantMessage, UserMessage)):
            continue  # forward-compat guard against any future Message variant
        usage = m.usage or {}
        reasoning_tokens = None
        try:
            raw_usage = (m.raw_data or {}).get("usage") or {}
            details = raw_usage.get("completion_tokens_details") or {}
            reasoning_tokens = details.get("reasoning_tokens")
        except Exception:
            pass
        entry = {
            "kind": "llm",
            "role": m.role,  # "assistant" (solver) or "user" (role-play agent)
            "model": agent_model if m.role == "assistant" else user_model,
            "content": m.content or "",
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": reasoning_tokens,
            "cost_usd": m.cost,
            # Real wall-clock seconds this specific generate() call took --
            # needed so solver.py can give each per-turn span a real,
            # non-zero duration instead of collapsing every turn to ~0ms
            # (all spans were being created+ended back-to-back in a
            # synchronous loop AFTER the episode already finished, so their
            # OTEL start/end timestamps were only microseconds apart
            # regardless of how long the real LLM call took).
            "generation_time_seconds": m.generation_time_seconds,
        }
        if m.tool_calls:
            entry["tool_calls"] = [
                {"name": tc.name, "arguments": tc.arguments} for tc in m.tool_calls
            ]
        out.append(entry)
    return out


def _bound_llm_call_budget(wall: float) -> str:
    """Give litellm an explicit per-call timeout so a hung call cannot outlive
    tau2's own wall. Returns a one-line description for the result payload.

    #147, found 2026-08-04. tau2 evaluates `episode_timeout` in exactly ONE
    place — `orchestrator._check_timeout()`, reached only from the
    step-completion path. So the check runs at TURN BOUNDARIES. If the FIRST
    LLM call hangs, no step ever completes, the check never runs, and
    `episode_timeout` is structurally unreachable no matter how high it is set.

    Measured signature, identical across two 18-episode runs: 4 episodes ending
    at 900.02-900.05s (jittered, so a real timer), span status UNSET, attributes
    stopping at the pre-episode set, ZERO child spans, no cost, no
    termination_reason. Nothing our side owns fired: inner 1800, derived outer
    2100, evaluate.py wrapper 2220. The only remaining bound was litellm's
    inherited default multiplied by `tau2.config.DEFAULT_MAX_RETRIES = 3`.

    Why fix it HERE. This file is ours and runs INSIDE tau2's venv, so it can
    configure litellm before tau2 touches it — no edit to the pinned vendored
    clone, and no reliance on tau2 growing a knob. Setting a per-call timeout
    turns an unbounded hang into a normal LLM error, which tau2 counts through
    `num_errors` -> `max_errors` and reports as a real `termination_reason`.
    An attributable failure is the point: an episode we can classify is worth
    more than one that merely dies sooner.

    The budget is derived from the episode wall, not fixed, so the two cannot
    drift apart the way the walls did: each attempt gets at most a third of the
    wall, floored at 60s so a short smoke wall stays usable.

    `wall` comes from the REQUEST, not from the environment. First draft read
    `EVOSKILL_TAU2_EPISODE_TIMEOUT` here — but the orchestrator's wall comes
    from the CONSTRUCTOR argument, so any caller passing `episode_timeout=`
    without also exporting that variable would have put the two budgets back out
    of step. That is the identical drift this whole fix exists to remove;
    threading the same number the orchestrator receives makes it structural
    rather than a convention.
    """
    try:
        import litellm as _ll
    except Exception as e:                      # pragma: no cover
        return f"litellm not importable ({type(e).__name__}); call budget UNBOUNDED"
    # The numbers come from `_tau2_protocol.llm_call_budget`, NOT from a formula
    # here: `solver.py` sizes its outer wall from the same call, and #175 was
    # caused by exactly this pair of numbers living in two places. See that
    # function's docstring for why the fraction is DELIBERATELY wall/3. (This comment
    # used to say "wall/12 and not wall/3", describing a FIRST DRAFT that was
    # withdrawn -- the opposite of what the code does. `_tau2_protocol` computes
    # max(60, wall/3) with 2 retries, so one call chain may consume the ENTIRE
    # episode wall, and that docstring explains why that is the chosen trade. A
    # comment asserting the reverse of its own code is the #175 shape: prose stating
    # an invariant that the code does not derive.)
    per_call, retries, worst_turn = llm_call_budget(wall)
    _ll.request_timeout = per_call
    _ll.num_retries = retries
    # `drop_params` unchanged; only the time budget is being bounded here.
    return (f"litellm.request_timeout={per_call:.0f}s "
            f"num_retries={retries} (wall={wall:.0f}s) "
            f"-> worst-case per call {worst_turn:.0f}s")


def _install_agent_budget() -> None:
    """Bound the episode by the AGENT's own clean time and turn count.

    the maintainer, 2026-08-06: "let's make clean time of agent turns only the metrics"
    and "max turn should only apply to agent".

    WHY THIS REPLACES max_steps AS THE DECIDING LIMIT. tau2's max_steps counts
    every message hop between ANY two roles (orchestrator.py:899), and the user
    simulator has its own tools (USER->ENV, orchestrator.py:562-566), so the
    agent's share is neither half nor fixed. Measured over 82 banking
    transcripts: the agent gets 38% of steps in normally-terminated episodes and
    45% in capped ones. A budget the user simulator can spend is not a budget.

    WHY CLEAN TIME AND NOT WALL-CLOCK. The 0806 base train leg lost 7/60
    episodes (11.7%) to a 1800s total-DURATION wall while the 300s inactivity
    watchdog never fired. Duration charges the agent for the user simulator's
    latency, tool time and host queueing; clean time cannot be consumed by load.

    BOTH LIMITS ARE OFF UNLESS SET, so an unconfigured run behaves EXACTLY as
    before -- the val leg of the 0806 base was in flight when this landed, and a
    default-on change would have straddled one leg across two instruments, which
    is precisely the #182/#183 failure this codebase keeps re-learning.

    The stop goes through tau2's OWN termination path (done + MAX_STEPS) rather
    than an exception, so the simulation finalizes normally and scores 0 under
    tau2's own rule (evaluator.py:119) -- identical semantics to the limit it
    replaces. Raising instead would surface as an is_error and be EXCLUDED from
    the denominator, silently converting a capability limit into an infra
    exclusion and inflating the rate.
    """
    budget_s = float(os.environ.get("EVOSKILL_TAU2_AGENT_CLEAN_S", "0") or 0)
    max_turns = int(os.environ.get("EVOSKILL_TAU2_AGENT_TURNS", "0") or 0)
    if budget_s <= 0 and max_turns <= 0:
        return
    from tau2.orchestrator.orchestrator import Orchestrator, Role
    from tau2.data_model.simulation import TerminationReason

    if getattr(Orchestrator, "_evoskill_agent_budget", False):
        return  # idempotent: one episode per worker, but re-entry must be safe
    _orig = Orchestrator._check_termination

    def _patched(self) -> None:
        _orig(self)
        if self.done:
            return
        # RESPECT tau2's OWN mid-tool-call guard. `_check_termination` opens with
        # `if self.to_role == Role.ENV: return` -- "skip termination checks if we're
        # waiting for environment to respond". Calling _orig() honours it for tau2's
        # limits, but _orig() RETURNS rather than raising, so control fell through to
        # our checks and we could set done=True while a tool call was still in flight.
        # Finalisation then routes an AssistantMessage-with-ToolCalls to the env and
        # tau2 raises "Environment should not receive the last message".
        # Observed live: e0_base_val task_090 sample 3 died exactly this way, the only
        # failure in 59 episodes, and it took the whole leg's canonical JSON with it.
        # Terminating one hop later costs nothing; terminating mid-hop corrupts the run.
        if self.to_role == Role.ENV:
            return
        spent = cost_tracking.clean_time_s("agent_response")
        turns = cost_tracking.call_count("agent_response")
        hit = None
        if budget_s > 0 and spent >= budget_s:
            hit = f"agent_clean_time {spent:.0f}s >= {budget_s:.0f}s"
        elif max_turns > 0 and turns >= max_turns:
            hit = f"agent_turns {turns} >= {max_turns}"
        if hit:
            self.done = True
            # #197: BOTH budgets land on MAX_STEPS, so `termination_reason` alone cannot say
            # WHICH wall fired. On the 25/25 banking run 22 of 108 capped episodes ended below
            # the 100-turn cap, every one of them at agent_clean_s ~= 1200 -- i.e. the CLEAN-TIME
            # wall, not the turn cap. Those are different failures (a loop at 101 turns vs ~37 s
            # per turn at 33 turns) and they want different remedies, but the record could not
            # tell them apart. `hit` already distinguishes them; it was assigned here and NEVER
            # READ ANYWHERE -- the same shape as #157, "recorded the reason and discarded the
            # only copy". Publish it on the orchestrator so the worker can put it in the result.
            self.termination_reason = TerminationReason.MAX_STEPS
            self._evoskill_budget_hit = hit
            self._evoskill_budget_kind = ("clean_time" if hit.startswith("agent_clean_time")
                                          else "turns")
            # main() builds the result dict and has no handle on this orchestrator instance,
            # so publish at module scope. Safe because this worker process runs exactly ONE
            # episode; a second write would mean two simulations in one process, which is why
            # the assert below exists rather than a silent overwrite.
            global _BUDGET_WALL_HIT, _BUDGET_WALL_DETAIL
            if _BUDGET_WALL_HIT is not None and _BUDGET_WALL_DETAIL != hit:
                raise RuntimeError(
                    f"two different budget-wall hits in one worker process: "
                    f"{_BUDGET_WALL_DETAIL!r} then {hit!r} -- the one-episode-per-process "
                    f"assumption behind this module global is broken")
            _BUDGET_WALL_HIT = self._evoskill_budget_kind
            _BUDGET_WALL_DETAIL = hit

    Orchestrator._check_termination = _patched
    Orchestrator._evoskill_agent_budget = True
    print(f"[agent-budget] clean_time={budget_s or 'off'}s turns={max_turns or 'off'}",
          file=sys.stderr)


# #197: which budget wall ended the episode, set by the _check_termination patch and read
# by main() when it builds the result. None on a voluntary stop.
_BUDGET_WALL_HIT: str | None = None
_BUDGET_WALL_DETAIL: str | None = None

# #198: the USER SIMULATOR's own system prompt, captured byte-exact (see
# _install_user_prompt_capture). None if the capture never fired.
_USER_SYSTEM_PROMPT: str | None = None
_USER_SYSTEM_PROMPT_VARIANTS: int = 0


_TRANSIENT_400 = "invalid request error trace_id"
_INT32_MAX = 2**31 - 1


def _install_seed_clamp() -> None:
    """Clamp `seed` into signed-32-bit range on every agent completion call.

    #203 -- THE root cause of the ling-3.0-flash base-rollout failures.

    tau2's `generate()` passes a per-episode `seed` drawn from the full UNSIGNED 32-bit range.
    Novita's endpoint accepts a SIGNED 32-bit seed and answers anything larger with
    HTTP 400 `{"message":"invalid request error trace_id: <hex>"}` -- an opaque message that
    names neither the field nor the constraint.

    MEASURED BOUNDARY, by bisecting one real failing request against the pinned endpoint:
        seed 0, 1, 12345, 2147483646, 2147483647  -> OK
        seed 2147483648, 3618361324, 4294967295   -> 400
    i.e. exactly 2**31-1. Half the unsigned range is rejected, which is why ~50% of episodes
    failed, why the failure was DETERMINISTIC per episode (same seed each retry, so all four
    retry attempts re-sent the same bad value), and why every replay I did succeeded -- my
    replay body omitted `seed` entirely.

    WHY CLAMP AND NOT DROP. `seed` is what makes an episode reproducible; dropping it would
    trade a crash for silent non-determinism, which is worse in a benchmark. `& _INT32_MAX`
    is a pure function of the original, so the same episode still gets the same seed on every
    run -- reproducibility is preserved, just inside the range the provider accepts.

    NOT ling-specific: any provider with a signed-32-bit seed hits this, so the clamp is
    unconditional rather than gated on a model name.
    """
    try:
        from tau2.utils import llm_utils as _lu
        orig = getattr(_lu, "completion", None)
        if orig is None or getattr(orig, "_evoskill_seed", False):
            return

        def _patched(*a, **kw):
            s = kw.get("seed")
            if isinstance(s, int) and not isinstance(s, bool) and not (0 <= s <= _INT32_MAX):
                kw["seed"] = s & _INT32_MAX
            return orig(*a, **kw)

        _patched._evoskill_seed = True
        _lu.completion = _patched
        print(f"[seed-clamp] armed (seed & {_INT32_MAX})", file=sys.stderr)
    except Exception as e:
        print(f"[seed-clamp] NOT armed: {e}", file=sys.stderr)


def _install_transient_400_retry(attempts: int = 4, backoff: float = 1.5) -> None:
    """Retry the provider's transient 400 on the AGENT's completion call.

    #202. Novita intermittently answers a perfectly valid request with
    HTTP 400 `{"message":"invalid request error trace_id: <hex>",
    "type":"invalid_request_error"}`. litellm maps that to BadRequestError, and
    BadRequestError is a 4xx -- a CLIENT error -- so litellm's own `num_retries` (3, set by
    tau2's `generate`) deliberately does NOT retry it. One flake therefore kills a whole
    episode, and tau2 surfaces it as `is_error`, which the runner correctly refuses to score.

    EVIDENCE IT IS TRANSIENT, NOT MALFORMED. The exact failing request, captured verbatim by
    `_install_request_dump` and replayed against the same pinned endpoint:
      * 14/14 OK serially;
      * 30/30 OK at 30-way concurrency (so it is not load, and not our conc=30);
      * all 17 tool schemas accepted, individually and together;
      * every observed failure has the SAME shape -- n_messages=3,
        [system, assistant-greeting, user] -- i.e. the episode's opening call, never a
        deep-history one. A malformed request would fail on replay. This one never does.

    So the request is valid, the endpoint accepts it, and the 400 is noise. Retrying is the
    correct response and does not paper over a real defect.

    NARROW BY DESIGN: only this exact message substring, only the agent's completion path.
    A genuine BadRequestError (bad schema, unsupported feature -- e.g. the structured-outputs
    400 this model really does return) does not match, and still fails fast and loudly. If the
    substring ever stops matching, this becomes a no-op rather than a silent swallow.
    """
    try:
        from tau2.utils import llm_utils as _lu
        orig = getattr(_lu, "completion", None)
        if orig is None or getattr(orig, "_evoskill_t400", False):
            return

        def _patched(*a, **kw):
            last = None
            for i in range(attempts):
                try:
                    return orig(*a, **kw)
                except Exception as e:
                    _s = str(e)
                    # RateLimitError joined the surface once the seed 400s stopped
                    # dominating (train 6, val 7). It is transient by definition and
                    # litellm does retry 429 -- but tau2 caps at num_retries=3, which a
                    # 30-way burst can exhaust. Back off further rather than lose an episode.
                    if _TRANSIENT_400 not in _s and "RateLimitError" not in _s:
                        raise
                    last = e
                    _dbg = os.environ.get("EVOSKILL_TAU2_REQ_DUMP", "")
                    if _dbg:
                        try:
                            with open(_dbg + ".retry", "a") as fh:
                                fh.write(f"attempt {i+1}/{attempts} matched transient-400\n")
                        except Exception:
                            pass
                    if i < attempts - 1:
                        time.sleep(backoff * (2 ** i))
                        print(f"[transient-400] retry {i+1}/{attempts-1}", file=sys.stderr)
            raise last

        _patched._evoskill_t400 = True
        _lu.completion = _patched
        print(f"[transient-400] armed: {attempts-1} retries on {_TRANSIENT_400!r}",
              file=sys.stderr)
    except Exception as e:
        print(f"[transient-400] NOT armed: {e}", file=sys.stderr)


def _install_request_dump() -> None:
    """On any LLM call failure, write the FULL request and the FULL exception to a file.

    #201. Three separate hypotheses about a 400 were wrong today because the only copy of the
    provider's reason was destroyed: litellm's exception string arrives truncated (~190 chars,
    mid-word), the runner's `errors[...]` slice is downstream of that so raising it changed
    nothing, and `.PARTIAL.json` records `episodes_excluded_infra` with no per-episode detail.
    An error surface that cannot be read is the same class of defect as #147 (deaths with "zero
    diagnostic trail") and #157 (the runner kept `is_error` and discarded the reason) -- and here
    it actively MISDIRECTED, because the untruncated ValueError was visible while the truncated
    400 was not, so the rarer failure looked like the whole story.

    Writes to $EVOSKILL_TAU2_REQ_DUMP (one JSON per line). Off unless that env var is set, so
    production legs are unaffected. Never raises: a diagnostic that can break a run is worse
    than no diagnostic.
    """
    path = os.environ.get("EVOSKILL_TAU2_REQ_DUMP", "").strip()
    if not path:
        return
    try:
        import litellm
    except Exception as e:                       # pragma: no cover
        print(f"[req-dump] litellm unavailable: {e}", file=sys.stderr)
        return

    def _wrap(name: str) -> None:
        orig = getattr(litellm, name, None)
        if orig is None or getattr(orig, "_evoskill_dump", False):
            return

        def _dump(exc, kwargs):
            try:
                with open(path, "a") as fh:
                    fh.write(json.dumps({
                        "fn": name,
                        "exc_type": type(exc).__name__,
                        # str(exc) is what gets truncated downstream -- keep every field the
                        # exception object carries, not just its rendered form.
                        "exc_str": str(exc),
                        "exc_body": str(getattr(exc, "response", "") or "")[:8000],
                        "exc_message": str(getattr(exc, "message", "") or "")[:8000],
                        "model": kwargs.get("model"),
                        "n_messages": len(kwargs.get("messages") or []),
                        "n_tools": len(kwargs.get("tools") or []),
                        "request_keys": sorted(kwargs.keys()),
                        # the non-message scalars are what a 400 is usually about
                        "scalars": {k: v for k, v in kwargs.items()
                                    if k not in ("messages", "tools") and not callable(v)
                                    and isinstance(v, (str, int, float, bool, type(None), dict, list))},
                    }, default=str) + "\n")
            except Exception:
                pass

        if name.startswith("a"):
            async def patched(*a, **kw):
                try:
                    return await orig(*a, **kw)
                except Exception as e:
                    _dump(e, kw); raise
        else:
            def patched(*a, **kw):
                try:
                    return orig(*a, **kw)
                except Exception as e:
                    _dump(e, kw); raise
        patched._evoskill_dump = True
        setattr(litellm, name, patched)

    for n in ("completion", "acompletion"):
        _wrap(n)

    # AND the BOUND name. `tau2/utils/llm_utils.py:15` does
    # `from litellm import completion`, which copies the function object into that module's
    # namespace at import time -- so patching `litellm.completion` afterwards is invisible to
    # every real call. That is exactly why the first version of this dump wrote nothing while
    # 74 episodes failed. Patch the binding tau2 actually calls.
    try:
        from tau2.utils import llm_utils as _lu
        orig_lu = getattr(_lu, "completion", None)
        if orig_lu is not None and not getattr(orig_lu, "_evoskill_dump", False):
            def _lu_patched(*a, **kw):
                try:
                    return orig_lu(*a, **kw)
                except Exception as e:
                    try:
                        with open(path, "a") as fh:
                            fh.write(json.dumps({
                                "fn": "tau2.utils.llm_utils.completion",
                                "exc_type": type(e).__name__,
                                "exc_str": str(e),
                                "exc_message": str(getattr(e, "message", "") or "")[:8000],
                                "model": kw.get("model"),
                                "n_messages": len(kw.get("messages") or []),
                                "n_tools": len(kw.get("tools") or []),
                                "request_keys": sorted(kw.keys()),
                                "scalars": {k: v for k, v in kw.items()
                                            if k not in ("messages", "tools") and not callable(v)},
                                # the FULL tool payload: a 400 on a tool-calling request is
                                # usually a schema the provider rejects, and the names alone
                                # are not enough to find it.
                                "tools": kw.get("tools"),
                                "messages": kw.get("messages"),
                                # the LAST message is what a mid-episode 400 usually turns on
                                "last_message": (kw.get("messages") or [None])[-1],
                            }, default=str) + "\n")
                    except Exception:
                        pass
                    raise
            _lu_patched._evoskill_dump = True
            _lu.completion = _lu_patched
            print("[req-dump] patched tau2.utils.llm_utils.completion", file=sys.stderr)
    except Exception as e:
        print(f"[req-dump] could not patch llm_utils: {e}", file=sys.stderr)

    print(f"[req-dump] armed -> {path}", file=sys.stderr)


def _install_user_prompt_capture() -> None:
    """Record the user simulator's OWN system prompt, byte-exact.

    the maintainer, 2026-08-07: "you should also log the system prompt of the user simulation agent."

    WHY A PATCH AND NOT A RECONSTRUCTION. The agent's system prompt is recoverable from the
    transcript -- tau2 inserts a `SystemMessage` into `simulation.get_messages()`, which is what
    `_extract_system_prompt` reads. The user simulator's is NOT: it lives only in the
    `UserState.system_messages` that `UserSimulator.get_init_state` builds
    (user_simulator.py:177) and never enters the message stream solver.py sees. So the choice is
    between reconstructing it here from tau2's `SYSTEM_PROMPT` template plus the task's user
    instructions, or capturing the real value at its source. Reconstruction is what
    `_extract_system_prompt`'s docstring already rejects for the agent ("not good enough as the
    thing Phoenix displays as ground truth"), and it would be even less safe here: the prompt is
    assembled from THREE independently-varying inputs -- the global guidelines markdown under
    `data/tau2/user_simulator/*.md`, the task's baked-in `UserScenario.persona`, and a runtime
    `PersonaConfig` -- with a `<PERSONA_GUIDELINES>` placeholder substitution in between
    (user_simulator.py:130-162). A copy of that assembly would drift silently the first time any
    of the three changed, and the drift would be invisible precisely because the trace would
    still show a plausible-looking prompt. Wrapping the property returns the exact string tau2
    puts in the SystemMessage, or nothing.

    NOT FATAL ON FAILURE. This is display-only instrumentation; a tracing concern must never
    cost an episode. A prompt that differs between accesses within one episode is recorded as a
    variant count rather than raised (unlike #197's budget-wall guard, where a second differing
    value means a MEASUREMENT is wrong). None surfaces as an absent system prompt in Phoenix --
    the same fail-visible state as before this existed.

    THE WHOLE INSTALL IS GUARDED, not just the import. `system_prompt` is a property on
    `UserSimulator` in both third_party trees today, so `.fget` resolves -- but if it ever
    becomes a plain method or attribute, an unguarded `.fget` would raise AttributeError and
    kill the episode over a display field. `UserSimulatorStreaming` already OVERRIDES the
    property, so a streaming user class silently keeps its own and this capture yields None:
    correct-by-absence, and visible as `tau2.user_system_prompt_captured=False` on the episode
    span rather than as a wrong prompt.
    """
    try:
        from tau2.user.user_simulator import UserSimulator

        if getattr(UserSimulator, "_evoskill_prompt_capture", False):
            return                               # idempotent, mirroring _install_agent_budget

        _orig = UserSimulator.system_prompt.fget

        def _patched(self):
            global _USER_SYSTEM_PROMPT, _USER_SYSTEM_PROMPT_VARIANTS
            value = _orig(self)
            if _USER_SYSTEM_PROMPT is None:
                _USER_SYSTEM_PROMPT = value
            elif value != _USER_SYSTEM_PROMPT:
                _USER_SYSTEM_PROMPT_VARIANTS += 1
            return value

        UserSimulator.system_prompt = property(_patched)
        UserSimulator._evoskill_prompt_capture = True
    except Exception as e:                       # pragma: no cover -- tau2's shape, not ours
        print(f"[user-prompt-capture] unavailable, user spans will show no system prompt: {e}",
              file=sys.stderr)


def _install_provider_pin(pin: dict | None, pin_model: str | None = None) -> str:
    """Apply the provider routing the PARENT computed, at tau2's own call site.

    NO RULE LIVES HERE. `solver.py` calls the shared `pin_request_body_provider` in
    EvoSkill's venv and ships the resulting block as `provider_pin`; this worker only
    merges it. That keeps the family gate and the allow_fallbacks contract in one place
    (tests/test_ssbv2_provider_pin.py forbids a seventh hand-rolled block) and respects
    this file's architecture: it runs in tau2-bench's venv and must not import EvoSkill.

    WHY IT PATCHES `tau2.utils.llm_utils.completion`, NOT JUST `litellm.completion`.
    tau2 does `from litellm import completion` (llm_utils.py:15) -- a FROM-import that
    binds the function into tau2's namespace at ITS import time, which happens at this
    module's import (agent_wiring, line 44) and therefore BEFORE this ever runs.
    Rebinding only `litellm.completion` would be completely INERT for every call tau2
    actually makes: it would log "installed" and pin nothing. The first draft did exactly
    that and the test for this function is what caught it.
    """
    if not pin:
        return "no provider pin supplied (routing left to OpenRouter)"

    def _wrap(orig):
        def _pinned(*args, **kwargs):
            # SCOPE THE PIN TO THE MODEL IT WAS COMPUTED FOR. This wrapper sits on the
            # shared `completion`, so it sees the agent, the USER SIMULATOR and the JUDGE.
            # Applying an agent-family provider block to the others is not merely untidy:
            # OpenRouter rejects a deepseek `order` on openai/gpt-5-nano with
            # "No endpoints found", which 404s the episode. The standing directive is
            # explicit that the DeepSeek pin must not extend outside its family, and the
            # family gate itself lives in the parent's `_provider_pin_for`.
            _m = str(kwargs.get("model") or (args[0] if args else ""))
            if pin_model and _m != pin_model:
                return orig(*args, **kwargs)
            eb = dict(kwargs.get("extra_body") or {})
            merged = dict(pin)
            merged.update(eb.get("provider") or {})     # caller's own keys win...
            if "allow_fallbacks" in pin:
                merged["allow_fallbacks"] = pin["allow_fallbacks"]  # ...except the pin
            eb["provider"] = merged
            kwargs["extra_body"] = eb
            return orig(*args, **kwargs)
        return _pinned

    patched = []
    try:
        import litellm as _ll
    except Exception:                            # pragma: no cover
        return "litellm not importable; provider pin NOT applied"
    if not getattr(_ll, "_evoskill_pinned", False):
        _ll.completion = _wrap(_ll.completion); _ll._evoskill_pinned = True
        patched.append("litellm.completion")
    try:
        from tau2.utils import llm_utils as _lu       # the LIVE call site
        if not getattr(_lu, "_evoskill_pinned", False):
            _lu.completion = _wrap(_lu.completion); _lu._evoskill_pinned = True
            patched.append("tau2.utils.llm_utils.completion")
    except Exception as e:
        return (f"provider pin INCOMPLETE — patched {patched or 'nothing'} but could NOT "
                f"reach tau2's binding ({type(e).__name__}: {e})")
    if not patched:
        return "provider pin already active on both bindings (no-op)"
    return f"provider pin {pin} INSTALLED on {patched}"


def main() -> None:
    req = json.loads(sys.stdin.read())
    domain = req["domain"]
    task_id = req["task_id"]
    agent_model = req["agent_model"]
    user_model = req["user_model"]
    judge_model = req["judge_model"]
    skills = req.get("skills", {})
    seed = req.get("seed")
    max_steps = req.get("max_steps", 200)
    # WHEN THE AGENT BUDGET IS ACTIVE, tau2's own max_steps MUST NOT BE THE THING THAT
    # DECIDES. It counts every role's hops, so with the agent at 38-45% of steps it fires
    # at ~80-95 agent turns -- BEFORE our 100-turn cap -- and the shortfall grows as the
    # user simulator gets chattier. That is the precise failure this change exists to
    # remove, and leaving 200 in place left it half-fixed.
    #
    # CAUGHT IN FLIGHT, not by review: an episode came back terminated at 901s
    # (75% of the clean-time budget) with 99/100 agent turns -- neither of our limits at
    # cap. The only remaining candidate was tau2's 200-hop ceiling, and it was.
    #
    # Raised to a NON-BINDING backstop rather than removed: an unbounded step count would
    # hand a pathological ping-pong loop the whole clean-time budget one cheap hop at a
    # time. 100 agent turns needs ~263 hops at the worst measured 38% share; 2000 clears
    # that by ~8x while still bounding the absurd.
    if (float(os.environ.get("EVOSKILL_TAU2_AGENT_CLEAN_S", "0") or 0) > 0
            or int(os.environ.get("EVOSKILL_TAU2_AGENT_TURNS", "0") or 0) > 0):
        max_steps = int(os.environ.get("EVOSKILL_TAU2_MAX_STEPS_BACKSTOP", "2000"))
    max_errors = req.get("max_errors", 10)
    timeout = req.get("timeout", 460)
    # Bound the per-call LLM budget from the SAME wall the orchestrator
    # gets (#147) — see `_bound_llm_call_budget`.
    print(f"[worker] {_bound_llm_call_budget(float(timeout))}", file=sys.stderr)
    # Standing directive: EVERY DeepSeek model rides native DeepSeek with fallbacks OFF.
    # The block itself is computed by the PARENT with the shared helper and arrives in the
    # request; installed here because this is the one place the episode path touches
    # litellm before tau2's `generate()` starts calling it.
    print(f"[worker] {_install_provider_pin(req.get('provider_pin'), req.get('provider_pin_model'))}", file=sys.stderr)
    # banking_knowledge-only: which retrieval variant builds the agent's
    # knowledge-base policy/tools (`bm25`, `full_kb`, `golden_retrieval`,
    # ...). Every other domain leaves both fields at their TextRunConfig
    # default of None -- confirmed harmless: `_build_env_kwargs`
    # (tau2/runner/build.py) only populates `env_kwargs["retrieval_variant"]`
    # when `retrieval_config is not None`, so passing None here reproduces
    # today's exact retail/airline/telecom behavior (no env_kwargs at all).
    retrieval_config = req.get("retrieval_config")
    retrieval_config_kwargs = req.get("retrieval_config_kwargs")

    try:
        # Re-route the NL-assertion judge through openrouter. It is a
        # value-imported module constant (confirmed:
        # `evaluator_nl_assertions.py:8`, `from tau2.config import
        # DEFAULT_LLM_NL_ASSERTIONS`), so patching `tau2.config` after this
        # module has already imported does nothing -- the module-local
        # name must be patched directly. There is no RunConfig/CLI-level
        # override point for this (confirmed by reading the source).
        _nl_mod.DEFAULT_LLM_NL_ASSERTIONS = judge_model

        cost_tracking.reset_ledger()
        cost_tracking.reset_clean_time()
        cost_tracking.install_cost_tracking()
        _install_request_dump()
        _install_seed_clamp()
        _install_transient_400_retry()
        _install_agent_budget()
        # Must precede run_single_task -- that is what constructs the UserSimulator and calls
        # get_init_state, which is the only access to the property we wrap.
        _install_user_prompt_capture()

        agent_name = agent_wiring.register_evoskill_agent(skills)

        tasks = get_tasks(domain, task_ids=[task_id])
        if not tasks:
            raise ValueError(f"Task '{task_id}' not found in domain '{domain}'")
        task = tasks[0]

        # #200: extra kwargs for the AGENT's LLM calls, as JSON. Task-universal by design --
        # it carries whatever the chosen solver needs, with no model-specific branching here.
        #
        # WHY IT EXISTS. `inclusionai/ling-3.0-flash` is a reasoning-ON-by-default hybrid that,
        # with reasoning enabled, returns an assistant message with NO content and NO tool_calls
        # -- it spends the completion on reasoning tokens. tau2's own data model then raises
        # `ValueError: AssistantMessage must have either content or tool_calls` BEFORE any turn
        # is recorded, so the episode dies at num_turns=0 and RunCache correctly refuses to
        # persist it. Measured: 131/150 episodes lost this way. Passing
        # {"reasoning": {"enabled": false}} zeroes reasoning tokens and the model returns both
        # content and tool_calls (verified by direct probe against the pinned Novita endpoint).
        #
        # THIS IS AN INSTRUMENT INPUT. It changes what the model does, so a leg run with it set
        # is NOT comparable to one without, and it belongs in the cache identity for the same
        # reason the user-sim model and the walls do (#182/#183/#185). That wiring is NOT done
        # here -- use a distinct EVOSKILL_TAU3_CACHE_TAG per setting until it is, or a changed
        # value will silently replay foreign records.
        _agent_llm_args: dict = {}   # merged onto DEFAULT_LLM_ARGS_AGENT below
        _raw_llm_args = os.environ.get("EVOSKILL_TAU2_AGENT_LLM_ARGS", "").strip()
        if _raw_llm_args:
            try:
                _agent_llm_args = json.loads(_raw_llm_args)
            except json.JSONDecodeError as e:
                raise SystemExit(
                    f"EVOSKILL_TAU2_AGENT_LLM_ARGS is not valid JSON ({e}); refusing to run "
                    f"rather than silently drop an instrument setting: {_raw_llm_args!r}")
            if not isinstance(_agent_llm_args, dict):
                raise SystemExit(
                    f"EVOSKILL_TAU2_AGENT_LLM_ARGS must be a JSON object, got "
                    f"{type(_agent_llm_args).__name__}: {_raw_llm_args!r}")
            print(f"[agent-llm-args] {json.dumps(_agent_llm_args, sort_keys=True)}", file=sys.stderr)

        # MERGE ONTO tau2's DEFAULT, never replace it. `TextRunConfig.llm_args_agent` is a
        # `dict` with default_factory=DEFAULT_LLM_ARGS_AGENT = {"temperature": 0.0}. Passing a
        # bare {"reasoning": ...} REPLACES that dict, silently dropping temperature=0.0 -- so
        # the leg runs at the provider's default temperature while every other arm runs at 0.0.
        # That is an instrument difference between arms in an ABLATION, i.e. exactly the
        # confound the ablation exists to avoid. Caught 2026-08-08 when the nex-n2-mini base
        # leg died on `llm_args_agent Input should be a valid dictionary [input_value=None]`:
        # the unset case was passing None, which the same field rejects outright.
        from tau2.config import DEFAULT_LLM_ARGS_AGENT as _DEF
        _agent_llm_args = {**_DEF, **_agent_llm_args}
        print(f"[agent-llm-args] MERGED onto default -> "
              f"{json.dumps(_agent_llm_args, sort_keys=True, default=str)}", file=sys.stderr)

        config = TextRunConfig(
            domain=domain,
            agent=agent_name,
            llm_agent=agent_model,
            llm_args_agent=_agent_llm_args,
            user="user_simulator",
            llm_user=user_model,
            max_steps=max_steps,
            max_errors=max_errors,
            num_trials=1,
            hallucination_retries=0,  # explicit -- not exercised in text
                                       # mode anyway (gated on
                                       # is_full_duplex at its only call
                                       # site, batch.py:680), disabled here
                                       # for clarity, not correctness.
            timeout=timeout,
            seed=seed,
            retrieval_config=retrieval_config,
            retrieval_config_kwargs=retrieval_config_kwargs,
        )

        # NOT EvaluationType.ALL -- confirmed by reading evaluator.py's own
        # docstring + evaluate_simulation: ALL only evaluates
        # ENV+COMMUNICATE+ACTION and explicitly excludes NL_ASSERTION even
        # when it's part of the task's reward_basis (raises "NL assertions
        # are part of the reward basis, but they are not being evaluated"
        # otherwise -- reproduced empirically on retail task "0", which
        # like 112/114 retail tasks has reward_basis=(DB,NL_ASSERTION)).
        # ALL_WITH_NL_ASSERTIONS is tau2's own (WIP-labeled, but
        # functional) evaluation type that adds NL_ASSERTION when present
        # in reward_basis -- this is a deviation from the original design
        # doc, which assumed EvaluationType.ALL was sufficient.
        simulation = run_single_task(
            config, task, seed=seed,
            evaluation_type=EvaluationType.ALL_WITH_NL_ASSERTIONS, save_dir=None,
        )

        reward_info = simulation.reward_info
        result = {
            "ok": True,
            "task_id": task.id,
            "domain": domain,
            "reward": reward_info.reward if reward_info else 0.0,
            "reward_basis": (
                [r.value for r in (reward_info.reward_basis or [])] if reward_info else []
            ),
            "reward_breakdown": (
                {k.value: v for k, v in (reward_info.reward_breakdown or {}).items()}
                if reward_info else {}
            ),
            "db_check": (
                reward_info.db_check.model_dump(mode="json")
                if reward_info and reward_info.db_check else None
            ),
            "action_checks": (
                [a.model_dump(mode="json") for a in (reward_info.action_checks or [])]
                if reward_info else []
            ),
            "communicate_checks": (
                [c.model_dump(mode="json") for c in (reward_info.communicate_checks or [])]
                if reward_info else []
            ),
            "nl_assertions": (
                [n.model_dump(mode="json") for n in (reward_info.nl_assertions or [])]
                if reward_info else []
            ),
            "termination_reason": (
                simulation.termination_reason.value if simulation.termination_reason else None
            ),
            # #197: WHICH budget produced a max_steps termination. `termination_reason` cannot
            # distinguish the 100-turn cap from the 1200s agent clean-time wall -- both are
            # MAX_STEPS -- so an analysis that calls every capped episode "a loop" is wrong for
            # roughly a fifth of them. None on episodes that ended voluntarily.
            "budget_wall_hit": _BUDGET_WALL_HIT,
            "budget_wall_detail": _BUDGET_WALL_DETAIL,
            "messages": _messages_to_dicts(simulation.get_messages()),
            "turn_log": _messages_to_turn_log(simulation.get_messages(), agent_model, user_model),
            "system_prompt": _extract_system_prompt(simulation.get_messages()),
            # #198: the user simulator's own system prompt. Distinct from `system_prompt` above,
            # which is the AGENT's -- conflating the two is the #146 failure mode (showing a
            # reader a prompt the participant never received).
            "user_system_prompt": _USER_SYSTEM_PROMPT,
            "user_system_prompt_variants": _USER_SYSTEM_PROMPT_VARIANTS,
            "cost_ledger": cost_tracking.read_ledger(),
            # Recorded on EVERY episode, not only capped ones: the 1200s budget
            # was chosen from an ESTIMATED 50-70% agent share of wall time
            # because no per-role timing existed yet. These fields are what turn
            # that estimate into a measurement, so the number can be re-derived
            # from one real leg instead of re-argued.
            "clean_time": cost_tracking.clean_time_report(),
        }
    except Exception as e:
        # Worker-boundary catch-all, not a silent swallow: the full
        # traceback is included in the JSON payload itself, so nothing is
        # lost -- an uncaught exception here would instead look like a
        # bare subprocess crash to solver.py with no diagnostic payload at
        # all, which is strictly worse for debugging.
        # #204 CONTEXT OVERFLOW IS A CAPABILITY FAILURE, NOT AN INFRA ERROR.
        # A model whose own window cannot hold the conversation has FAILED the task, exactly
        # as exhausting the turn cap is a failure -- and tau2 already scores `max_steps` as 0
        # (evaluator.py:119). Classifying it as `ok: False` makes the runner refuse to score
        # the episode and refuse a canonical result JSON, which turns a real, reportable
        # capability limit into an unrunnable leg.
        #
        # MEASURED 2026-08-08 on two DIFFERENT 262,144-token models, and the two overflow in
        # DIFFERENT ways, which is why a max_tokens change is NOT the fix:
        #   nex-n2-mini: "262885 total: 197349 input + 65536 completion"  -> the RESERVATION
        #                overflows; a smaller max_tokens would fit.
        #   ling-3.0-flash: "266500 requested (264365 text + 2135 tool input)" -> the INPUT
        #                ALONE exceeds the window. NOTHING fits. Not tunable.
        # DSV4F never hit this: 1M window. So this is a benchmark-vs-model-class limit.
        #
        # Scoring 0 does NOT alter the instrument -- the model saw exactly what it saw and
        # could not answer. It records the truth instead of discarding the episode.
        _emsg = f"{type(e).__name__}: {e}"
        _overflow = ("maximum context length" in _emsg
                     or "Requested token count exceeds" in _emsg
                     or "context length is" in _emsg)
        if _overflow:
            result = {
                "ok": True,
                "task_id": task_id,
                "domain": domain,
                "reward": 0.0,
                "reward_basis": [],
                "reward_breakdown": {},
                "db_check": None, "action_checks": [], "communicate_checks": [],
                "nl_assertions": [],
                "termination_reason": "context_overflow",
                "budget_wall_hit": None, "budget_wall_detail": None,
                "messages": [], "turn_log": [],
                "system_prompt": None, "user_system_prompt": None,
                "user_system_prompt_variants": 0,
                "cost_ledger": cost_tracking.read_ledger(),
                "clean_time": cost_tracking.clean_time_report(),
                "capability_limit": _emsg[:2000],
            }
        else:
            result = {
                "ok": False,
                "error": _emsg,
                "traceback": traceback.format_exc(),
            }

    # FRAME THE RESULT. Bare stdout is a SHARED channel: this worker imports tau2, litellm
    # and the domain modules, and any one of them may print. Measured 2026-08-06 on the tau3
    # smoke: 3 of 15 episodes (20%) died with "Worker produced non-JSON stdout: Expecting
    # value: line 2 column 1" and a stderr tail full of policy/transaction text — the episode
    # ran to completion and its result was destroyed in transit by somebody else's print.
    # Only this worker writes the marker, so the parent can find the payload no matter what
    # else lands on the stream. Written on its own line, after a newline, so a library that
    # leaves a partial line cannot merge into it.
    sys.stdout.write("\n" + RESULT_MARKER + "\n")
    sys.stdout.write(json.dumps(result, default=str))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
