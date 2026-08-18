"""Tests for the cross-process rollout-seat pool.

Each test is written to FAIL if the specific rule it names is broken, and several
were built by first breaking the implementation to check they fire — a scheduler test
that passes for both the right and the wrong behaviour is worse than no test, which
this project has already paid for twice (a `--expect-not` comparing a 64-char digest
to a 12-char prefix, and an iterate-flag assertion that held for either value).
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.harness.seats import SeatPool  # noqa: E402


def _pool(tmp_path, member, priority, total=3, **kw):
    return SeatPool(total=total, member=member, priority=priority,
                    path=tmp_path / "seats.json",
                    trajectory=tmp_path / "traj.jsonl", **kw)


# --- the hard ceiling --------------------------------------------------------

def test_never_exceeds_total_within_one_process(tmp_path):
    p = _pool(tmp_path, "a", 0, total=3)
    asyncio.run(_take(p, 3))
    assert p.snapshot()["used"] == 3
    # a 4th acquire must not be granted
    assert asyncio.run(_try_for(p, 0.6)) is False


def test_never_exceeds_total_across_two_members(tmp_path):
    """The whole point: two POOLS (as two processes would) share one ceiling."""
    a = _pool(tmp_path, "a", 0, total=3)
    b = _pool(tmp_path, "b", 0, total=3)
    asyncio.run(_take(a, 2))
    asyncio.run(_take(b, 1))
    assert a.snapshot()["used"] == 3
    assert asyncio.run(_try_for(b, 0.6)) is False


# --- strict priority --------------------------------------------------------

def test_may_take_refuses_lower_priority_WHILE_A_SEAT_IS_FREE(tmp_path):
    """THE priority test, and the only one that isolates the rule.

    Every behavioural test below fills the pool first, so a refusal there is
    explained by the CEILING and says nothing about priority — verified by
    mutation: deleting the priority branch entirely left all of them green. The
    rule only bites when a seat IS available and the asker is outranked, so the
    state has to be built that way explicitly.
    """
    lo = _pool(tmp_path, "val", 1, total=3)
    state = {"members": {
        # a higher-priority member holding 1 seat with unmet demand
        "train#1": {"pid": os.getpid(), "member": "train", "priority": 0,
                    "held": 1, "waiting": 4, "ts": 9e12},
        # us: one seat held, and the pool has a free seat (1+1 < 3)
        lo._key(): {"pid": os.getpid(), "member": "val", "priority": 1,
                    "held": 1, "waiting": 1, "ts": 9e12},
    }}
    used = sum(m["held"] for m in state["members"].values())
    assert used < lo.total, "test is meaningless unless a seat is genuinely free"
    assert lo._may_take(state) is False, \
        "outranked member took a free seat while a higher priority was waiting"

    # ...and the same member MAY take it once the higher rank's demand is met.
    state["members"]["train#1"]["waiting"] = 0
    assert lo._may_take(state) is True, "work-conservation broken: seat left idle"


def test_released_seat_goes_to_waiter_not_to_low_priority_reacquire(tmp_path):
    """Behavioural form of the same rule: on release, the waiter wins the seat.

    This is the one that would catch a real regression in the handoff, because the
    freed seat makes the pool non-full at the instant both members want it.
    """
    lo = _pool(tmp_path, "val", 1, total=3)
    hi = _pool(tmp_path, "train", 0, total=3)
    asyncio.run(_take(lo, 3))                     # val fills the pool

    async def scenario():
        hi_task = asyncio.create_task(hi.acquire())
        await asyncio.sleep(0.3)                  # let train's demand publish
        lo.release()                              # a val rollout finishes
        # val immediately tries to reuse its own freed seat; it must lose.
        got_lo = await _try_for(lo, 1.2)
        await asyncio.wait_for(hi_task, timeout=5)
        return got_lo
    got_lo = asyncio.run(scenario())
    assert got_lo is False, "val re-took the freed seat ahead of waiting train"
    assert hi.snapshot()["me"]["held"] == 1, "freed seat did not go to train"


def test_freed_seat_goes_to_high_priority_not_back_to_low(tmp_path):
    """the maintainer's worked example, in miniature: 2 val in flight, train arrives."""
    lo = _pool(tmp_path, "val", 1, total=3)
    hi = _pool(tmp_path, "train", 0, total=3)
    asyncio.run(_take(lo, 3))

    async def scenario():
        waiters = [asyncio.create_task(hi.acquire()) for _ in range(2)]
        await asyncio.sleep(0.3)
        lo.release(); lo.release()                # two val rollouts complete
        await asyncio.wait_for(asyncio.gather(*waiters), timeout=5)
    asyncio.run(scenario())
    assert hi.snapshot()["me"]["held"] == 2
    assert lo._held == 1                          # the survivor was NOT preempted


def test_no_preemption_of_held_seats(tmp_path):
    """A held seat is only ever released by its holder."""
    lo = _pool(tmp_path, "val", 1, total=2)
    hi = _pool(tmp_path, "train", 0, total=2)
    asyncio.run(_take(lo, 2))

    async def scenario():
        t = asyncio.create_task(hi.acquire())
        await asyncio.sleep(0.8)
        assert not t.done(), "high priority acquired without a seat being released"
        assert lo._held == 2, "a held seat was taken away"
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
    asyncio.run(scenario())


def test_work_conserving_when_high_priority_idle(tmp_path):
    """With no unmet high-priority demand, the low rank may use the whole pool."""
    hi = _pool(tmp_path, "train", 0, total=3)
    lo = _pool(tmp_path, "val", 1, total=3)
    asyncio.run(_take(hi, 1))
    asyncio.run(_take(lo, 2))                     # fills remaining seats
    assert lo.snapshot()["used"] == 3


# --- crash safety -----------------------------------------------------------

def test_dead_member_seats_are_reclaimed(tmp_path):
    """A killed run must not permanently shrink the pool."""
    live = _pool(tmp_path, "live", 0, total=3)
    ghost = _pool(tmp_path, "ghost", 0, total=3)
    asyncio.run(_take(ghost, 3))
    # simulate the process dying: rewrite its record with an impossible pid
    d = json.loads((tmp_path / "seats.json").read_text())
    key = [k for k in d["members"] if k.startswith("ghost#")][0]
    d["members"][key]["pid"] = 999_999_999
    (tmp_path / "seats.json").write_text(json.dumps(d))
    assert asyncio.run(_try_for(live, 1.0)) is True, "dead member's seats not reclaimed"


def test_leave_removes_member(tmp_path):
    a = _pool(tmp_path, "a", 0, total=2)
    asyncio.run(_take(a, 2))
    a.leave()
    b = _pool(tmp_path, "b", 0, total=2)
    assert asyncio.run(_try_for(b, 0.6)) is True


# --- opt-in behaviour -------------------------------------------------------

def test_from_env_returns_none_without_total(monkeypatch):
    """Absent env must mean 'unchanged behaviour', never a permissive pool."""
    monkeypatch.delenv("EVOSKILL_SEAT_TOTAL", raising=False)
    assert SeatPool.from_env(member="x") is None


def test_from_env_reads_priority(monkeypatch, tmp_path):
    monkeypatch.setenv("EVOSKILL_SEAT_TOTAL", "7")
    monkeypatch.setenv("EVOSKILL_SEAT_PRIORITY", "2")
    monkeypatch.setenv("EVOSKILL_SEAT_MEMBER", "val_leg")
    p = SeatPool.from_env(path=tmp_path / "s.json", trajectory=tmp_path / "t.jsonl")
    assert (p.total, p.priority, p.member) == (7, 2, "val_leg")


# --- telemetry --------------------------------------------------------------

def test_trajectory_records_squeeze(tmp_path):
    """A squeezed run must be able to PROVE it afterwards, not just look slow."""
    lo = _pool(tmp_path, "val", 1, total=1)
    asyncio.run(_take(lo, 1))
    lo.release()
    lines = [json.loads(x) for x in (tmp_path / "traj.jsonl").read_text().splitlines()]
    assert [x["event"] for x in lines] == ["acquire", "release"]
    assert lines[0]["fleet_used"] == 1 and lines[0]["total"] == 1


# --- helpers ----------------------------------------------------------------

async def _take(pool, n):
    for _ in range(n):
        await asyncio.wait_for(pool.acquire(), timeout=5)


async def _try_for(pool, seconds):
    """Try to acquire, giving up after `seconds`. Returns whether it succeeded."""
    t = asyncio.create_task(pool.acquire())
    try:
        await asyncio.wait_for(asyncio.shield(t), timeout=seconds)
        return True
    except asyncio.TimeoutError:
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
        return False


if __name__ == "__main__":
    raise SystemExit(pytest.main([os.path.abspath(__file__), "-v"]))


# --- the 2026-08-03 overshoot: a LIVE member must survive a stale heartbeat ---
# Production defect. `ts` advances only inside `_mutate`, so a member holding its
# last seats with an EMPTY queue makes no calls and goes legitimately quiet for as
# long as its slowest rollout. `_reap` used `not _alive(pid) OR ts stale`, so the
# stale branch alone evicted it — while its rollouts kept running, uncounted. The
# other member then filled the pool beside them. Measured: e3_train evicted holding
# 10 seats, real concurrency 35 against a cap of 30.

def test_live_member_with_stale_heartbeat_is_NOT_reaped(tmp_path):
    """The exact production shape: alive, holding seats, silent past the TTL."""
    p = _pool(tmp_path, "val", 1, total=30, ttl_s=1.0)
    state = {"members": {
        "train#%d" % os.getpid(): {
            "pid": os.getpid(), "member": "train", "priority": 0,
            "held": 10, "waiting": 0, "ts": 0.0,      # ts=0 => maximally stale
        }}}
    p._reap(state)
    assert "train#%d" % os.getpid() in state["members"], (
        "a LIVE member holding 10 seats was evicted for being quiet; its rollouts "
        "keep running uncounted and the pool will overshoot")
    assert state["members"]["train#%d" % os.getpid()]["held"] == 10


def test_dead_member_is_reaped_immediately_without_waiting_out_the_ttl(tmp_path):
    """The other side: a dead PID is proof, and must not wait out a 3600s TTL."""
    p = _pool(tmp_path, "val", 1, total=30, ttl_s=1e9)
    state = {"members": {"train#999999999": {
        "pid": 999999999, "member": "train", "priority": 0,
        "held": 10, "waiting": 0, "ts": time.time(),   # heartbeat is FRESH
    }}}
    p._reap(state)
    assert "train#999999999" not in state["members"], (
        "a dead member's seats must be reclaimed immediately, fresh ts or not")


def test_the_overshoot_is_unreachable_via_reap_plus_acquire(tmp_path):
    """End to end: quiet live holder + a full pool must refuse, not overshoot."""
    val = _pool(tmp_path, "val", 1, total=30, ttl_s=1.0)
    val._mutate(lambda st: st["members"].update({
        "train#%d" % os.getpid(): {
            "pid": os.getpid(), "member": "train", "priority": 0,
            "held": 30, "waiting": 0, "ts": 0.0}}))
    # val must see a FULL pool: train is quiet but alive and holds every seat.
    got = val._mutate(val._try_take)
    assert got is False, "val took a 31st seat because a quiet live holder was reaped"
    snap = val.snapshot()
    assert snap["used"] == 30, f"pool reports {snap['used']} of 30"


def test_default_ttl_exceeds_a_realistic_rollout():
    """Defence in depth. The TTL no longer gates live members, but it still gates
    unverifiable ones, and 300s was under the ~470s rollout median that produced the
    2026-08-03 overshoot. Keep it sized against a rollout so a future change that
    reintroduces a TTL path cannot inherit the old, too-small number."""
    from src.harness.seats import DEFAULT_TTL_S
    assert DEFAULT_TTL_S >= 1800.0, (
        f"DEFAULT_TTL_S={DEFAULT_TTL_S}s must exceed the slowest gap between two "
        f"_mutate calls, i.e. one rollout (SSB v2: ~470s median, ~834s p90)")


def test_missing_pid_is_not_read_as_alive(tmp_path):
    """os.kill(-1, 0) signals every process the user owns and SUCCEEDS, so a record
    with no pid must not reach _alive and be mistaken for a live holder."""
    p = _pool(tmp_path, "val", 1, total=30, ttl_s=1e9)
    state = {"members": {"ghost#x": {"member": "train", "priority": 0,
                                     "held": 30, "waiting": 0, "ts": 0.0}}}
    p._reap(state)                     # ts=0 is older than any TTL
    assert "ghost#x" not in state["members"], (
        "a pid-less member held 30 seats and was treated as alive")
