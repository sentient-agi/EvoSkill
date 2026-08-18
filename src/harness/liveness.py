"""Progress-based liveness watchdog for solver rollouts.

WHY THIS EXISTS (measured 2026-07-29). Rollout budgets were duration-based: a
single wall-clock cap per rollout. That cap cannot distinguish productive work
from a stuck process, because it measures elapsed time rather than progress.
Phoenix span forensics over 53 solver rollouts that exceeded 720s found two
populations living in the SAME duration band:

  - 43 rollouts that COMPLETED between 720s and 1643s ran at 98-100%
    merged-interval occupancy — genuine long generation over 20-60 turns (one
    burned 1,006,491 prompt tokens).
  - 11 of the 14 that hit the 1800s wall were HUNG: 690-1452s of dead air at
    the TAIL, occupancy 19-61%, with lead gap 0s and max INTERNAL gap 1-2s.
    They worked normally, then stopped emitting anything at all.

So any single duration threshold either kills the first group or tolerates the
second. Raising the wall to 1800s stopped truncating real work and bought
~3.4 hours of dead waiting instead.

THE FIX: bound INACTIVITY, not duration.

  - inactivity budget (short, default 120s): killed if nothing productive
    happens for this long, no matter how little total time has elapsed.
  - total budget (generous, default 3600s): a blast-radius backstop only.

A 25-minute generation that keeps emitting tokens survives. A dead socket or a
wedged post-processing step dies in ~2 minutes instead of ~20.

EXIT REASON IS THE OTHER HALF. A rollout now reports WHY it ended, and callers
must treat the reasons differently:

  - COMPLETED       -> a real observation, score it, cache it.
  - INACTIVITY_KILL -> infra failure. NO OBSERVATION. Exclude it from means,
                       do not insert it as a scored row, never cache it, retry it.
  - TOTAL_WALL      -> the rollout was making progress and simply ran out of
                       budget. That is a genuine capability limit and is a real,
                       cacheable 0.
Conflating these is what produced phantom failures: an infra-killed sample was
scored 0.0 at full weight, which caps an N=3 task at 0.667 against an 0.8 [OK]
threshold and deterministically relabels a passing task as [FAIL] in the
evolver's failure list.

DESIGN NOTES

- Heartbeats are cheap: `touch()` is one ContextVar read plus two assignments,
  so it is safe to call per streamed chunk.
- `touch()` is a no-op outside a guard, so instrumentation can be sprinkled
  into shared code paths without caring whether a guard is active.
- The Progress object is bound INSIDE the guarded task, not in the parent.
  asyncio.Task snapshots the context at creation, so setting the ContextVar as
  the task's first action isolates concurrent rollouts from each other. Binding
  it in the parent would let sibling rollouts share one Progress — the same
  shape as this repo's earlier switch_runtime ContextVar leak.
- time.monotonic() throughout: wall-clock jumps (NTP, sleep/wake) must not
  fabricate or mask a stall.
- The watchdog never runs longer than the rollout it guards; it is cancelled in
  a finally block.
"""
from __future__ import annotations

import asyncio
import contextvars
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable


class ExitReason(str, Enum):
    COMPLETED = "completed"
    INACTIVITY_KILL = "inactivity_kill"
    CLEAN_BUDGET = "clean_budget"
    TOTAL_WALL = "total_wall"

    @property
    def is_infra(self) -> bool:
        """True when the rollout produced NO valid observation.

        Callers use this to decide exclude-vs-score.

        CLEAN_BUDGET and TOTAL_WALL are deliberately NOT infra: in both the
        rollout was making progress and genuinely ran out of budget, which is a
        real result about the task's difficulty and is a legitimate, cacheable 0.
        Only INACTIVITY_KILL means "we learned nothing" — exclude it from every
        mean, never insert it as a scored row, never cache it, retry it.
        """
        return self is ExitReason.INACTIVITY_KILL


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


# Stall detector: no productive event for this long => dead, kill and retry.
# Must exceed the longest legitimate silence (time-to-first-token on a
# reasoning model, a slow tool call), hence 120s rather than the SSE layer's 45s.
DEFAULT_INACTIVITY_S = _env_float("EVOSKILL_INACTIVITY_TIMEOUT", 120.0)
# The real budget: productive seconds a rollout is allowed. Set to the old
# 720s solver wall so a HEALTHY rollout gets exactly the budget it always had —
# the change is that stalled time no longer counts against it.
DEFAULT_CLEAN_BUDGET_S = _env_float("EVOSKILL_CLEAN_BUDGET", 720.0)
# Blast radius only. Pure clean-time budgeting is UNBOUNDED in wall clock: a
# cycle that stalls, is killed, retries, emits two tokens and stalls again
# accrues clean time arbitrarily slowly while burning real hours and real
# money. This backstop should essentially never fire.
DEFAULT_TOTAL_S = _env_float("EVOSKILL_TOTAL_WALL", 3600.0)
# Longest single gap charged as productive. Covers time-to-first-token and
# tool latency; anything beyond is presumed dead socket and is not charged.
DEFAULT_MAX_CLEAN_GAP_S = _env_float("EVOSKILL_MAX_CLEAN_GAP", 90.0)
DEFAULT_POLL_S = _env_float("EVOSKILL_LIVENESS_POLL_S", 10.0)


@dataclass
class Progress:
    """Per-rollout liveness state. One instance per guarded rollout.

    Tracks THREE clocks, because they answer different questions:

      elapsed_s — wall clock. Only used as a blast-radius backstop.
      idle_s    — time since the last productive event. The stall DETECTOR.
      clean_s   — accumulated PRODUCTIVE time. The actual BUDGET (the maintainer's
                  "clean time", 2026-07-29).

    clean_s is the sum of inter-heartbeat gaps, each capped at max_clean_gap:

        clean_s = Σ min(gap_i, max_clean_gap)

    The cap is what makes this work without every call site reporting its own
    duration. A gap shorter than the cap is normal cadence — streaming chunks,
    a tool call returning, a model thinking before its first token — and is
    charged in full. Time beyond the cap in a single gap is presumed dead
    socket and is NOT charged. So genuine latency counts against the budget
    while stalls are free, which is exactly the asymmetry we want: a rollout is
    not punished for the provider's bad night, but a model cannot think
    forever for free either.

    Why budget on clean time at all: a wall-clock budget makes a rollout's
    outcome depend on infra weather, so the same task can pass on a quiet night
    and fail on a flaky one. That variance lands directly in the evolver's
    evidence and in every val/test number. Budgeting productive time removes it.
    """

    name: str
    max_clean_gap: float = 90.0
    started: float = field(default_factory=time.monotonic)
    last: float = field(default_factory=time.monotonic)
    events: int = 0
    last_kind: str = ""
    # Per-kind counts make it possible to say WHICH instrumentation fired,
    # which is how we tell "the model streamed for 20 min" from "the scorer
    # churned for 20 min" after the fact.
    kinds: dict[str, int] = field(default_factory=dict)
    _clean: float = 0.0
    wasted_s: float = 0.0

    def touch(self, kind: str = "") -> None:
        now = time.monotonic()
        gap = now - self.last
        charged = gap if gap <= self.max_clean_gap else self.max_clean_gap
        self._clean += charged
        self.wasted_s += gap - charged
        self.last = now
        self.events += 1
        if kind:
            self.last_kind = kind
            self.kinds[kind] = self.kinds.get(kind, 0) + 1

    @property
    def idle_s(self) -> float:
        return time.monotonic() - self.last

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.started

    @property
    def clean_s(self) -> float:
        """Productive time so far, including the in-flight gap.

        The open gap is included (capped) so the budget check cannot be evaded
        by a call that never emits another heartbeat — without this, a rollout
        sitting at 99% of budget could hang forever and never trip the gate,
        which is the very failure mode this class exists to remove.
        """
        gap = self.idle_s
        return self._clean + min(gap, self.max_clean_gap)


_current: contextvars.ContextVar[Progress | None] = contextvars.ContextVar(
    "evoskill_liveness_progress", default=None
)


def touch(kind: str = "") -> None:
    """Record productive activity. No-op when no guard is active.

    Call this from every place that constitutes progress: a streamed chunk
    arriving, a tool call starting or returning, one step of a long
    post-processing loop. Over-calling is harmless; under-calling causes false
    kills, so prefer instrumenting too much.
    """
    p = _current.get()
    if p is not None:
        p.touch(kind)


def current() -> Progress | None:
    """The active rollout's Progress, or None outside a guard."""
    return _current.get()


async def await_beating(awaitable, kind: str, interval: float = 30.0):
    """Await `awaitable`, emitting a heartbeat every `interval` seconds.

    Bracketing an opaque phase with start/done touches names the culprit after
    the fact but still leaves a gap as long as the phase, so a healthy-but-slow
    phase is indistinguishable from a wedged one and dies on the idle gate.
    Measured 2026-07-30: v4 lost three midgate rollouts at idle=125-128s with
    last_kind='score_start', and 4 of v3's 13 kills were tool_start/tool_done —
    i.e. long bash commands, whose real bound is EVOSKILL_BASH_TIMEOUT.

    Beat from the AWAITING side: it needs no knowledge of the callee and works
    for thread-offloaded work, where the callee cannot reach the ContextVar at
    all. Phases whose duration is legitimately unbounded must be bounded by
    CLEAN_BUDGET and TOTAL_WALL, never by the stall detector.

    Each beat is a gap <= max_clean_gap, so it is charged in full — beating
    cannot be used to run forever for free.

    CRITICAL: `awaitable` is awaited in the CALLER's context, and only the
    beater runs in a task. Wrapping the work itself in
    `asyncio.ensure_future` copies the context (PEP 567), so any ContextVar
    the work SETS is written into that copy and lost on return. A first cut
    did exactly that around the tool-call path and broke the executor's
    `_docker_agent_container` handoff: every bash call then saw no container
    and created another, while only the last was ever released — the host went
    from 128 to 983 containers in minutes. Beating in the side task is safe
    because `touch()` only READS the ContextVar and mutates the Progress
    object it points at, which the copied context shares.
    """
    stop = asyncio.Event()

    async def _beat() -> None:
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
                return
            except asyncio.TimeoutError:
                touch(kind)

    beater = asyncio.ensure_future(_beat())
    try:
        return await awaitable
    finally:
        stop.set()
        beater.cancel()
        try:
            await beater
        except BaseException:
            pass


@dataclass
class LivenessResult:
    """Outcome of a guarded rollout.

    clean_s and wasted_s are worth persisting on every trace, not just failures:
    clean_s is a far better difficulty signal than wall time (it is invariant to
    infra weather), and wasted_s quantifies how much of the night the provider
    cost us — the number this whole mechanism exists to drive to zero.
    """

    value: Any
    exit_reason: ExitReason
    elapsed_s: float
    clean_s: float
    wasted_s: float
    idle_s_at_exit: float
    events: int
    kinds: dict[str, int]

    @property
    def is_infra(self) -> bool:
        return self.exit_reason.is_infra


async def run_with_liveness(
    factory: Callable[[], Awaitable[Any]],
    *,
    name: str = "rollout",
    inactivity_s: float | None = None,
    clean_budget_s: float | None = None,
    total_s: float | None = None,
    max_clean_gap_s: float | None = None,
    poll_s: float | None = None,
    on_kill: Callable[[ExitReason, Progress], None] | None = None,
) -> LivenessResult:
    """Run `factory()` under an inactivity watchdog and a total-time backstop.

    `factory` is a zero-arg callable returning the coroutine to guard; taking a
    factory rather than a coroutine means the coroutine is created inside this
    function, so it can never be left un-awaited if validation fails first.

    Exceptions raised by the guarded coroutine propagate unchanged — this
    function only converts the two BUDGET outcomes into LivenessResult, and
    deliberately does not swallow real errors.
    """
    inact = DEFAULT_INACTIVITY_S if inactivity_s is None else inactivity_s
    clean_budget = DEFAULT_CLEAN_BUDGET_S if clean_budget_s is None else clean_budget_s
    total = DEFAULT_TOTAL_S if total_s is None else total_s
    gap_cap = DEFAULT_MAX_CLEAN_GAP_S if max_clean_gap_s is None else max_clean_gap_s
    poll = DEFAULT_POLL_S if poll_s is None else poll_s
    if poll <= 0:
        raise ValueError("poll_s must be > 0")
    if gap_cap <= 0:
        raise ValueError(f"max_clean_gap_s ({gap_cap}) must be > 0")
    # No gap_cap-vs-inactivity validation (proofread C-1, 2026-07-29: the
    # original `gap_cap < inact -> raise` was INVERTED and crashed on the
    # module's own defaults 90 vs 120). Neither ordering is broken:
    # gap_cap < inact just counts long-surviving gaps conservatively
    # (wasted > 0); gap_cap > inact is moot since gaps longer than inact are
    # killed first. Production exports 120/120.

    prog = Progress(name=name, max_clean_gap=gap_cap)

    async def _body() -> Any:
        # Bound INSIDE the task so concurrent rollouts cannot share it.
        _current.set(prog)
        return await factory()

    task = asyncio.ensure_future(_body())
    verdict: ExitReason = ExitReason.COMPLETED

    async def _watch() -> None:
        nonlocal verdict
        while True:
            await asyncio.sleep(poll)
            if task.done():
                return
            # Order matters: inactivity is checked FIRST so a stalled rollout
            # is classified as infra (retry, never cache) rather than as a
            # budget exhaustion (a real, cacheable 0). Getting this backwards
            # reintroduces the phantom-failure bug through the back door.
            if prog.idle_s > inact:
                verdict = ExitReason.INACTIVITY_KILL
            elif prog.clean_s > clean_budget:
                verdict = ExitReason.CLEAN_BUDGET
            elif prog.elapsed_s > total:
                verdict = ExitReason.TOTAL_WALL
            else:
                continue
            if on_kill:
                on_kill(verdict, prog)
            task.cancel()
            return

    watcher = asyncio.ensure_future(_watch())
    try:
        value = await task
        return LivenessResult(value, ExitReason.COMPLETED, prog.elapsed_s,
                              prog.clean_s, prog.wasted_s, prog.idle_s,
                              prog.events, dict(prog.kinds))
    except asyncio.CancelledError:
        # Distinguish OUR cancellation from an outer one. If verdict is still
        # COMPLETED, something above us cancelled and we must not swallow it —
        # swallowing an outer cancel is how a "stopped" run keeps running.
        if verdict is ExitReason.COMPLETED:
            raise
        return LivenessResult(None, verdict, prog.elapsed_s, prog.clean_s,
                              prog.wasted_s, prog.idle_s, prog.events,
                              dict(prog.kinds))
    finally:
        watcher.cancel()
        # Await the watcher so it cannot outlive the rollout it guards.
        try:
            await watcher
        except (asyncio.CancelledError, Exception):
            pass
        # Outer-cancellation leak fix (proofread C-2, 2026-07-29): when an
        # OUTER wait_for cancels us, the re-raise path above never cancelled
        # the guarded task — it kept running detached (no watchdog, outside
        # the rollout semaphore, still writing to the task's output dir) for
        # up to its inner wall. On the normal/watchdog paths task is already
        # done and this is a no-op.
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
