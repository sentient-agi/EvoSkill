"""Cross-process ROLLOUT-SEAT leases with strict priority. Drop-in for a Semaphore.

WHAT THIS IS FOR, and why `FleetBudget` is not it. `fleet.py` caps in-flight LLM
CALLS and divides the pool proportionally by weight — deliberately, because the
2026-07-29 measurement showed provider failure rate, not local CPU, is what scales
with load. This module caps a different resource: concurrent ROLLOUTS (one seat per
task-sample in flight), across every process on the machine, with STRICT PRIORITY
rather than proportional shares.

THE RULE IT IMPLEMENTS (the maintainer, 2026-08-03), stated so the code can be checked
against it:

  * One fleet-wide pool of N seats. The sum of in-flight rollouts across ALL
    participating processes never exceeds N.
  * Priority is a total order (lower number wins). Iteration-N+1 train outranks
    iteration-N train, which outranks iteration-N val, because only the train half
    blocks the next evolver.
  * Admission-time only: **a rollout already holding a seat is never preempted.**
    A deprioritised member simply stops being granted new seats and drains.
  * Work-conserving: if the top-priority member has no unmet demand, its seats go
    to the next member down rather than idling.

Worked example from the spec: 20 val rollouts in flight when a train phase starts.
Train immediately gets the 10 free seats. Every seat a val rollout releases goes to
train, not back to val. Val's remaining tasks wait until train has no unmet demand.

WHY WEIGHTS CANNOT DO THIS. `FleetBudget`'s share is a ceiling each process grants
ITSELF, not a count of global in-flight work. In the example above, val's computed
share would drop to 2 while it is already 18 over — and nothing may kill those. Train
would read its own share as 27 and start admitting, peaking at 47. A per-rollout
lease against a shared counter cannot overshoot, because the counter is the thing
being checked.

OPT-IN, so this file cannot affect a run that does not ask for it: `from_env` returns
None unless EVOSKILL_SEAT_TOTAL is set, and callers fall back to their existing
`asyncio.Semaphore`. That matters because it is being introduced while a measured leg
is live.

    pool = SeatPool.from_env(member="e2_train", priority=0)
    gate = pool or asyncio.Semaphore(CONCURRENCY)
    async with gate:
        ...one rollout...
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("EVOSKILL_ROOT", str(Path(__file__).resolve().parents[2])))
DEFAULT_PATH = ROOT / ".cache/fleet_rollout_seats.json"
DEFAULT_TRAJECTORY = ROOT / ".cache/fleet_rollout_seats_trajectory.jsonl"

# A member whose PID is ALIVE is evicted only after this long without a heartbeat.
# It exists solely to guard PID reuse and genuinely wedged processes; a dead PID is
# reclaimed immediately and does not wait it out (see `_reap`).
#
# THE YARDSTICK IS A ROLLOUT, NOT THE POLL INTERVAL — this is what the 2026-08-03
# overshoot got wrong. `ts` is refreshed only inside `_mutate`, i.e. on acquire or
# release, so a member that has taken its last task and is holding seats with an
# EMPTY queue makes no calls at all and goes legitimately quiet for as long as its
# slowest in-flight rollout. At the old 300s this was not merely possible but
# near-certain at the end of every leg, because an SSB v2 rollout has a ~470s median
# and an ~834s p90: e3_train was evicted while alive holding 10 seats, e3_val then
# filled the pool to 30 beside it, and real concurrency hit 35 against a cap of 30.
# The old comment reasoned "much longer than the poll interval (0.5s)" — true, and
# irrelevant. Compare against the longest thing that can happen between two calls.
DEFAULT_TTL_S = 3600.0
POLL_S = 0.5


def _alive(pid: int) -> bool:
    """Same liveness test fleet.py uses — signal 0 probes without delivering."""
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError, TypeError):
        return False
    return True


class SeatPool:
    """A priority-ordered, cross-process seat pool with an asyncio-compatible API."""

    def __init__(self, total: int, member: str, *, priority: int = 0,
                 path: Path | None = None, ttl_s: float = DEFAULT_TTL_S,
                 trajectory: Path | None = None) -> None:
        self.total = max(1, int(total))
        self.member = member
        self.priority = int(priority)
        self.path = Path(path or DEFAULT_PATH)
        self.ttl_s = max(1.0, float(ttl_s))
        self.trajectory = Path(trajectory or DEFAULT_TRAJECTORY)
        self.pid = os.getpid()
        self._held = 0          # seats this process holds, for crash accounting
        self._waiting = 0       # unmet demand, published so lower ranks can yield
        self._lock = asyncio.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, member: str | None = None, priority: int | None = None,
                 **kw: Any) -> "SeatPool | None":
        """Build from env, or None when seats are not in use.

        Returning None rather than a permissive pool is the safe default: an absent
        EVOSKILL_SEAT_TOTAL means "this run predates seats", and it must behave
        exactly as it did before, not acquire from a pool nobody else respects.
        """
        total = int(os.environ.get("EVOSKILL_SEAT_TOTAL", "0") or 0)
        if total <= 0:
            return None
        return cls(
            total=total,
            member=member or os.environ.get("EVOSKILL_SEAT_MEMBER") or f"pid-{os.getpid()}",
            priority=(priority if priority is not None
                      else int(os.environ.get("EVOSKILL_SEAT_PRIORITY", "0") or 0)),
            ttl_s=float(os.environ.get("EVOSKILL_SEAT_TTL_S", DEFAULT_TTL_S)),
            **kw,
        )

    # -- shared-state plumbing ---------------------------------------------

    def _key(self) -> str:
        return f"{self.member}#{self.pid}"

    def _mutate(self, fn):
        """Run fn(state) under an exclusive flock, persisting whatever it returns.

        Every read-modify-write goes through here. Doing the check and the increment
        in separate locked sections is how a seat pool overshoots.
        """
        with open(self.path, "a+") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                fh.seek(0)
                raw = fh.read().strip()
                try:
                    state = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    state = {}
                state.setdefault("members", {})
                self._reap(state)
                out = fn(state)
                fh.seek(0)
                fh.truncate()
                json.dump(state, fh, indent=1, sort_keys=True)
                fh.flush()
                os.fsync(fh.fileno())
                return out
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def _reap(self, state: dict) -> None:
        """Drop members whose process is gone, or that are wedged past the TTL.

        A dead member's seats MUST be reclaimed: it cannot release them itself, and
        without this one killed run would permanently shrink the pool. This is the
        same pattern fleet.py uses for budget shares, and the container reaper uses
        for orphaned sandboxes.

        THE TWO CONDITIONS ARE NOT INTERCHANGEABLE, and conflating them with `or` is
        what let the pool overshoot on 2026-08-03. A dead PID is PROOF the seats are
        abandoned. A stale timestamp is not proof of anything: `ts` only advances
        inside `_mutate`, so a live member holding its last seats with an empty queue
        is silent by design. Evicting it does not free its rollouts — they keep
        running, uncounted — so the pool admits replacements beside them and exceeds
        its own cap. Measured: e3_train evicted while alive holding 10 seats, real
        concurrency 35 against a cap of 30.

        So: **a live PID is never evicted, at any age.** The TTL applies only to a
        record whose PID cannot be checked at all (missing or malformed), which is the
        one case where liveness is genuinely unknown.

        That leaves PID reuse unhandled on purpose, because the two failure directions
        are not symmetric. A leaked seat costs throughput: the pool admits FEWER than
        its cap, visibly, and one `leave()` or process exit reclaims it. An eviction
        costs the invariant: the pool admits MORE than its cap while real rollouts run
        uncounted, which is unbounded and silent. When a guard must be wrong, it
        should be wrong in the direction that under-admits.
        """
        now = time.time()
        for k, m in list(state["members"].items()):
            pid = m.get("pid")
            # A checkable PID is the ONLY authority. Note `_alive` must never be
            # handed a non-positive pid: os.kill(-1, 0) signals every process the
            # user owns and reports success, so a missing pid defaulted to -1 would
            # read as ALIVE and pin a phantom member's seats forever.
            if isinstance(pid, int) and pid > 0:
                if not _alive(pid):
                    state["members"].pop(k, None)
                continue
            # Unverifiable record: fall back to the TTL, and say so.
            if (now - float(m.get("ts", 0))) > self.ttl_s:
                print(f"  [SEATS] dropping member {k} with unusable pid={pid!r} "
                      f"after {now - float(m.get('ts', 0)):.0f}s without a "
                      f"heartbeat", flush=True)
                state["members"].pop(k, None)

    def _self_record(self, state: dict) -> dict:
        rec = state["members"].setdefault(self._key(), {})
        rec.update(pid=self.pid, member=self.member, priority=self.priority,
                   held=self._held, waiting=self._waiting, ts=time.time())
        return rec

    # -- the admission decision -------------------------------------------

    def _may_take(self, state: dict) -> bool:
        """True iff this member may take a seat right now.

        Two conditions, and both are the rule rather than a heuristic:

        1. The pool is not full. `used` counts seats held by LIVE members only, so a
           crashed process cannot strand capacity.
        2. No STRICTLY higher-priority member has unmet demand. This is the squeeze
           and the whole point: a lower rank stops acquiring while a higher rank still
           wants seats, and drains on its own as its in-flight rollouts finish. It is
           also why nothing needs to be killed — yielding happens by not taking, and
           a seat is only ever surrendered voluntarily on completion.
        """
        used = sum(int(m.get("held", 0)) for m in state["members"].values())
        if used >= self.total:
            return False
        for k, m in state["members"].items():
            if k == self._key():
                continue
            if int(m.get("priority", 0)) < self.priority and int(m.get("waiting", 0)) > 0:
                return False
        return True

    async def acquire(self) -> None:
        # Publish demand BEFORE testing, so a lower-priority member polling
        # concurrently can already see that we want a seat. Announcing only after a
        # failed attempt would let it keep taking seats in the gap.
        async with self._lock:
            self._waiting += 1
        try:
            while True:
                got = self._mutate(self._try_take)
                if got:
                    return
                await asyncio.sleep(POLL_S)
        finally:
            async with self._lock:
                self._waiting = max(0, self._waiting - 1)
            self._mutate(self._self_record)   # republish reduced demand promptly

    def _try_take(self, state: dict) -> bool:
        self._self_record(state)              # make our demand visible first
        if not self._may_take(state):
            return False
        self._held += 1
        self._self_record(state)
        self._log(state, "acquire")
        return True

    def release(self) -> None:
        def _fn(state):
            self._held = max(0, self._held - 1)
            self._self_record(state)
            self._log(state, "release")
        self._mutate(_fn)

    # -- asyncio.Semaphore compatibility ----------------------------------

    async def __aenter__(self) -> "SeatPool":
        await self.acquire()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.release()

    def leave(self) -> None:
        """Remove this member entirely. Call at process end so the next run's reap
        does not have to wait out the TTL."""
        def _fn(state):
            state["members"].pop(self._key(), None)
        self._held = 0
        self._waiting = 0
        self._mutate(_fn)

    # -- observability ----------------------------------------------------

    def _log(self, state: dict, event: str) -> None:
        """Append a line per admission event.

        Not optional decoration: the failure mode of a scheduler is running at the
        wrong concurrency and LOOKING merely slow. A run that was squeezed must be
        able to prove it afterwards.
        """
        try:
            used = sum(int(m.get("held", 0)) for m in state["members"].values())
            rec = {"ts": round(time.time(), 3), "event": event, "member": self.member,
                   "pid": self.pid, "priority": self.priority, "held": self._held,
                   "waiting": self._waiting, "fleet_used": used, "total": self.total,
                   "members": {k: {"p": v.get("priority"), "h": v.get("held"),
                                   "w": v.get("waiting")}
                               for k, v in state["members"].items()}}
            with open(self.trajectory, "a") as fh:
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
        except OSError:
            pass          # telemetry must never break admission

    def snapshot(self) -> dict:
        def _fn(state):
            self._self_record(state)
            return {
                "total": self.total,
                "used": sum(int(m.get("held", 0)) for m in state["members"].values()),
                "me": {"member": self.member, "priority": self.priority,
                       "held": self._held, "waiting": self._waiting},
                "members": {k: {"priority": v.get("priority"), "held": v.get("held"),
                                "waiting": v.get("waiting")}
                            for k, v in state["members"].items()},
            }
        return self._mutate(_fn)
