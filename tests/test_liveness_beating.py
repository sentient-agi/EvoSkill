"""Regression tests for `_await_beating` (executor.py) under the real watchdog.

Context (2026-07-30): the inline scorer is bracketed by score_start/score_done
heartbeats but is internally ONE opaque blocking call, so a legitimately slow
scoring phase left a heartbeat gap as long as the phase itself and was killed as
a stall. Three v4 midgate rollouts died this way at idle=125-128s, every one
reporting last_kind='score_start'.

These tests drive the PRODUCTION guard (`run_with_liveness`), not a stand-in, so
they demonstrate the defect and the fix rather than a model of them.
"""

import asyncio
import time

import pytest

from src.harness.direct.executor import _await_beating
from src.harness.liveness import (
    ExitReason,
    await_beating,
    current,
    run_with_liveness,
    touch,
)

# Scaled-down analogue of production's 120s gate: the guard must be given a
# window shorter than the slow phase, or there is nothing to prove.
INACT = 0.3
SLOW = 1.0
BEAT = 0.1


async def _slow_phase_unbeaten():
    """A scorer as it behaved BEFORE the fix: one touch, then a long call."""
    touch("score_start")
    await asyncio.sleep(SLOW)
    touch("score_done")
    return 0.75


async def _slow_phase_beaten():
    """The same phase awaited through the fix."""
    touch("score_start")
    await _await_beating(asyncio.sleep(SLOW), "score_running", interval=BEAT)
    touch("score_done")
    return 0.75


def _guard(factory):
    return run_with_liveness(
        factory,
        name="test",
        inactivity_s=INACT,
        clean_budget_s=30.0,
        total_s=30.0,
        max_clean_gap_s=INACT,
        poll_s=0.02,
    )


def test_unbeaten_slow_phase_is_killed_as_a_stall():
    """The defect. Without interior heartbeats the phase looks like a stall."""
    res = asyncio.run(_guard(_slow_phase_unbeaten))
    assert res.exit_reason is ExitReason.INACTIVITY_KILL
    assert res.exit_reason.is_infra
    assert res.kinds.get("score_start") == 1
    assert "score_done" not in res.kinds


def test_beaten_slow_phase_survives_and_returns_its_value():
    """The fix. Identical phase, identical gate, now completes."""
    res = asyncio.run(_guard(_slow_phase_beaten))
    assert res.exit_reason is ExitReason.COMPLETED
    assert res.value == 0.75
    assert res.kinds.get("score_done") == 1
    # Roughly SLOW/BEAT beats; assert a floor rather than an exact count so the
    # test does not become a scheduler-latency detector.
    assert res.kinds.get("score_running", 0) >= 3


def test_beating_charges_scoring_against_the_clean_budget():
    """Beats must not create a way to run forever for free.

    Each beat is a gap <= gap_cap, so it is charged in full — scoring time
    counts against CLEAN_BUDGET, which is the correct bound for a phase whose
    duration is legitimately unbounded.
    """
    res = asyncio.run(_guard(_slow_phase_beaten))
    assert res.clean_s >= SLOW * 0.8


def test_returns_value_and_propagates_exceptions():
    """Checks BOTH entry points: liveness.await_beating and the executor alias.

    The executor keeps a thin local wrapper so the harness still runs when the
    liveness module is unavailable, so the alias needs its own coverage.
    """

    async def _body():
        async def _ok():
            return 42

        async def _boom():
            raise RuntimeError("scorer exploded")

        assert await await_beating(_ok(), "k", interval=0.05) == 42
        assert await _await_beating(_ok(), "k", interval=0.05) == 42
        for fn in (await_beating, _await_beating):
            with pytest.raises(RuntimeError, match="scorer exploded"):
                await fn(_boom(), "k", interval=0.05)

    asyncio.run(_body())


def test_fast_work_emits_no_beats():
    """No heartbeat inflation when the phase is quick."""

    async def _phase():
        touch("score_start")
        await _await_beating(asyncio.sleep(0.01), "score_running", interval=5.0)
        return 1.0

    res = asyncio.run(_guard(_phase))
    assert res.exit_reason is ExitReason.COMPLETED
    assert "score_running" not in res.kinds


def test_outer_cancellation_does_not_leak_the_inner_task():
    """The finally block must cancel AND await the inner task.

    A leaked task would keep running past the kill and could still mutate the
    workspace of a rollout the runner has already written off — the same shape
    as the C-2 leak found in the watchdog itself.
    """

    async def _body():
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def _inner():
            started.set()
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled.set()
                raise

        async def _outer():
            await _await_beating(_inner(), "k", interval=0.05)

        task = asyncio.ensure_future(_outer())
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert cancelled.is_set(), "inner task was never cancelled"

    asyncio.run(_body())


def _victim_factory(gaps: list, beats: int = 80, cadence: float = 0.02):
    """A healthy rollout that beats steadily, like an SSE read loop.

    Records its own inter-beat gaps so starvation is measured directly rather
    than inferred from whether it happened to be killed.
    """

    async def _victim():
        last = time.monotonic()
        for _ in range(beats):
            touch("sse")
            now = time.monotonic()
            gaps.append(now - last)
            last = now
            await asyncio.sleep(cadence)
        return "victim-ok"

    return _victim


def test_inline_blocking_call_starves_a_CONCURRENT_rollout():
    """Cross-rollout starvation: a sync call on the loop freezes everyone else.

    Deliberately asserts STARVATION (the victim's heartbeat gap balloons past
    the idle gate), not death. Whether starvation escalates to a kill is a race:
    once the loop is released, the victim's own touch and the watchdog's poll are
    both runnable, and if the victim touches first its idle timer resets and it
    survives. Measured here, the victim usually survives — so loop-blocking is a
    real hazard that can produce misleading last_kind='sse' kills, but it must
    NOT be asserted as the explanation for any particular observed kill.
    """
    gaps: list = []

    async def _body():
        async def _blocker():
            touch("snapshot_start")
            time.sleep(INACT * 2)  # inline: freezes the loop
            touch("snapshot_done")
            return "blocked"

        await asyncio.gather(_guard(_victim_factory(gaps)), _guard(_blocker))

    asyncio.run(_body())
    assert max(gaps) > INACT, (
        f"expected a starvation gap > {INACT}s, saw max {max(gaps):.3f}s"
    )


def test_offloaded_blocking_call_does_not_starve_a_concurrent_rollout():
    """The fix: the same blocking work in a thread starves nobody."""
    gaps: list = []

    async def _body():
        async def _blocker():
            touch("snapshot_start")
            await _await_beating(
                asyncio.to_thread(time.sleep, INACT * 2),
                "snapshot_running",
                interval=BEAT,
            )
            touch("snapshot_done")
            return "offloaded"

        return await asyncio.gather(_guard(_victim_factory(gaps)), _guard(_blocker))

    victim, blocker = asyncio.run(_body())
    assert max(gaps) < INACT, (
        f"offloading should keep gaps small, saw max {max(gaps):.3f}s"
    )
    assert victim.exit_reason is ExitReason.COMPLETED
    assert victim.value == "victim-ok"
    assert blocker.exit_reason is ExitReason.COMPLETED
    assert blocker.kinds.get("snapshot_done") == 1


def test_contextvar_written_by_the_awaited_work_REACHES_the_caller():
    """The regression that caused the container leak, in one assertion.

    The executor hands a rollout's Docker container between bash calls through a
    ContextVar that `_execute_tool` sets. If the awaited work is wrapped in
    `asyncio.ensure_future`, it runs in a COPIED context (PEP 567) and that write
    is discarded on return — so every bash call saw no container, created another,
    and only the last was released. Host went 128 -> 983 containers in minutes.

    So `await_beating` must await the work in the caller's own context and put
    only the heartbeat loop in a task.
    """
    import contextvars

    var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
        "test_container", default=None
    )

    async def _body():
        async def _sets_the_var():
            await asyncio.sleep(0.12)
            var.set("container-abc")
            return "done"

        assert var.get() is None
        for fn in (await_beating, _await_beating):
            var.set(None)
            assert await fn(_sets_the_var(), "k", interval=0.03) == "done"
            assert var.get() == "container-abc", (
                "ContextVar set by the awaited work did not reach the caller — "
                "the work is being wrapped in a task and its context copied"
            )

    asyncio.run(_body())


def test_beats_land_on_the_active_progress_object():
    """Guards against a silently-dead heartbeat path (the C-3 failure mode).

    `_await_beating` beats from the awaiting side, inside the guarded context,
    so it must reach the same Progress the watchdog reads.
    """

    async def _phase():
        p = current()
        assert p is not None, "no active Progress inside the guard"
        before = p.events
        await _await_beating(asyncio.sleep(0.3), "score_running", interval=0.05)
        assert p.events > before
        return 1.0

    res = asyncio.run(_guard(_phase))
    assert res.exit_reason is ExitReason.COMPLETED
