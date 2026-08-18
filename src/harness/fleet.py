"""Cross-process fleet budget for LLM-call concurrency.

THE HOLE THIS CLOSES (measured 2026-07-29). Concurrency control today is
per-process at two layers, and the lower one has a hardcoded ceiling:
executor.py's _AdaptiveConcurrencyLimiter is constructed with
max_concurrent=40 as a module-global singleton, with no env knob. So N
concurrently-running experiments silently grant themselves 40*N in-flight LLM
calls against one shared provider account. Tonight three processes were live at
once -> a 120-call nominal ceiling nobody chose.

That matters because provider failures are the ONE resource that measurably
scales with load. Same host, same model, same wall-clock window: arm a at
rollout cap 15 saw 6 retries / 3,968 LLM calls (0.0015/call, 4 AIMD halvings),
while arm c at cap 30 saw 45 / 9,008 (0.0050/call, 36 halvings) — 3.3x the
failure rate. Meanwhile local resources were NOT the constraint: 57 containers
totalled 78 MiB, the colima VM had 11.8 of 13.9 GB free, and host load sat at
0.59/core. So the fleet needs a ceiling on CALLS, not on containers or CPU.

DESIGN, and why each choice is what it is:

- A file + advisory lock, NOT a daemon. This host has a documented recurring
  failure where every background process dies at once; a broker would become a
  single point whose death silently freezes or unbounds the fleet. A file has
  no liveness of its own to lose.

- OFF unless EVOSKILL_FLEET_LLM_TOTAL is set. Default import changes nothing,
  so it can land next to a frozen experiment without altering its behavior.

- Shares are recomputed from LIVE members only (PID liveness + heartbeat TTL),
  so the mass-death case self-heals: when one process survives and the others
  are gone, it legitimately owns the whole budget. There is no stale claim to
  leak, and a relaunched process does not inherit a phantom entitlement.

- NOT on the hot path. share() caches for refresh_s and does no I/O in between,
  so before_call() reads a plain int. Per-call broker round-trips were
  explicitly rejected: the median rollout is ~60s and needs no sub-second
  arbitration, while before_call is already a jittered poll loop with a 180s
  starvation deadline — adding a queueing stage there is how provider jitter
  becomes a starvation cascade.

- Pins are first-class. A member may declare weight=0 with a fixed reservation
  (`pinned`), meaning "this experiment must run at exactly this cap for
  comparability"; its reservation is subtracted from the pool before the
  remainder is divided. Concurrency has demonstrably changed outcomes in this
  lab, so an experiment whose validity depends on a known cap must be able to
  opt out of arbitration without opting out of accounting.

- Never revokes running work. A shrinking share applies to future admissions
  only. Revoking in-flight calls would mean the bookkeeping says "within
  budget" while the client keeps running what it already started — an audit
  that lies exactly when it matters most.

- Every share change is appended to a JSONL trajectory, because any run must be
  able to prove what its limit was at every moment.
"""
from __future__ import annotations

import errno
import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any

_ROOT = Path(os.environ.get("EVOSKILL_ROOT", str(Path(__file__).resolve().parents[2])))
DEFAULT_PATH = _ROOT / ".cache/fleet_llm_budget.json"
DEFAULT_TRAJECTORY = _ROOT / ".cache/fleet_llm_budget_trajectory.jsonl"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _alive(pid: int) -> bool:
    """True if pid exists. signal 0 never delivers a signal.

    EPERM means the pid exists but belongs to another user — still alive, so a
    foreign process can never be reaped as stale.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError as e:
        return e.errno == errno.EPERM


class FleetBudget:
    """Fair-share partition of a global in-flight LLM-call budget.

    Usage from a participating process:

        fb = FleetBudget.from_env(member="arm-b-e2")
        if fb:                      # None when the feature is off
            fb.join()
            cap = fb.share()        # cheap; refreshes at most every refresh_s

    share() is the only call on any hot-ish path and is memoized.
    """

    def __init__(self, total: int, member: str, *, weight: float = 1.0,
                 pinned: int | None = None, min_share: int = 2,
                 refresh_s: float = 10.0, ttl_s: float = 90.0,
                 path: Path | None = None, trajectory: Path | None = None) -> None:
        self.total = max(1, int(total))
        self.member = member
        self.weight = 0.0 if pinned is not None else max(0.0, float(weight))
        self.pinned = None if pinned is None else max(1, int(pinned))
        self.min_share = max(1, int(min_share))
        self.refresh_s = max(0.0, float(refresh_s))
        self.ttl_s = max(1.0, float(ttl_s))
        self.path = Path(path or DEFAULT_PATH)
        self.trajectory = Path(trajectory or DEFAULT_TRAJECTORY)
        self.pid = os.getpid()
        self._cached_share: int | None = None
        self._cached_at = 0.0
        self._last_logged: int | None = None

    # -- construction -------------------------------------------------------

    @classmethod
    def from_env(cls, member: str, **kw: Any) -> "FleetBudget | None":
        """Build from EVOSKILL_FLEET_* env vars, or None if the feature is off.

        Returning None rather than a no-op object keeps the call site honest:
        `if fb:` is visibly a feature gate, not an always-taken branch.
        """
        total = _env_int("EVOSKILL_FLEET_LLM_TOTAL", 0)
        if total <= 0:
            return None
        pin = _env_int("EVOSKILL_FLEET_LLM_PIN", 0)
        # Pop the kwarg unconditionally (proofread m-4, 2026-07-30): popping
        # only in the else-branch left a duplicate `pinned` kwarg -> TypeError
        # whenever the env pin AND a caller kwarg were both present.
        _pin_kw = kw.pop("pinned", None)
        return cls(
            total=total,
            member=member,
            weight=_env_float("EVOSKILL_FLEET_WEIGHT", 1.0),
            pinned=pin if pin > 0 else _pin_kw,
            min_share=_env_int("EVOSKILL_FLEET_MIN_SHARE", 2),
            refresh_s=_env_float("EVOSKILL_FLEET_REFRESH_S", 10.0),
            ttl_s=_env_float("EVOSKILL_FLEET_TTL_S", 90.0),
            **kw,
        )

    # -- state file ---------------------------------------------------------

    def _read_locked(self, fh) -> dict:
        fh.seek(0)
        raw = fh.read()
        if not raw.strip():
            return {"members": {}}
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            # A torn file must not wedge the fleet; treat it as empty and let
            # live members re-register on their next heartbeat.
            return {"members": {}}
        if not isinstance(d.get("members"), dict):
            d["members"] = {}
        return d

    def _write_locked(self, fh, d: dict) -> None:
        fh.seek(0)
        fh.truncate()
        fh.write(json.dumps(d, indent=1))
        fh.flush()
        os.fsync(fh.fileno())

    def _mutate(self, fn) -> Any:
        """Run fn(state) under an exclusive lock and persist the result.

        Opened 'a+' so the file is created if absent without truncating an
        existing one; flock is advisory but every participant goes through here.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a+") as fh:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                d = self._read_locked(fh)
                out = fn(d)
                self._write_locked(fh, d)
                return out
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)

    def _reap(self, d: dict, now: float) -> None:
        """Drop members whose PID is gone or whose heartbeat expired.

        PID liveness is the primary signal; the TTL only covers a process that
        is alive but wedged. A PID that was reused by an unrelated process is
        caught by the member name recorded alongside it.
        """
        for key, m in list(d["members"].items()):
            if not _alive(int(m.get("pid", -1))) or now - float(m.get("ts", 0)) > self.ttl_s:
                d["members"].pop(key, None)

    # -- membership ---------------------------------------------------------

    def _key(self) -> str:
        return f"{self.member}:{self.pid}"

    def join(self) -> None:
        def fn(d: dict) -> None:
            self._reap(d, time.time())
            d["members"][self._key()] = {
                "pid": self.pid, "member": self.member, "weight": self.weight,
                "pinned": self.pinned, "ts": time.time(),
            }
            d["total"] = self.total
        self._mutate(fn)
        self._cached_share = None  # force a fresh computation after joining

    def leave(self) -> None:
        def fn(d: dict) -> None:
            d["members"].pop(self._key(), None)
        try:
            self._mutate(fn)
        except OSError:
            pass  # exiting anyway; the reaper will collect it by PID liveness

    # -- share computation --------------------------------------------------

    def _compute(self, d: dict) -> int:
        pins = sum(int(m["pinned"]) for m in d["members"].values() if m.get("pinned"))
        if self.pinned is not None:
            return self.pinned
        pool = max(0, self.total - pins)
        wsum = sum(float(m.get("weight", 0.0)) for m in d["members"].values()
                   if not m.get("pinned"))
        if wsum <= 0:
            return max(self.min_share, pool)
        return max(self.min_share, int(pool * (self.weight / wsum)))

    def share(self, force: bool = False) -> int:
        """This process's allowed in-flight LLM calls.

        Memoized for refresh_s. The heartbeat rides along with the refresh, so
        a process that keeps asking stays alive in the ledger and one that
        stops asking ages out — no separate heartbeat thread to leak.
        """
        now = time.monotonic()
        if not force and self._cached_share is not None and now - self._cached_at < self.refresh_s:
            return self._cached_share

        def fn(d: dict) -> int:
            wall = time.time()
            self._reap(d, wall)
            key = self._key()
            if key not in d["members"]:  # reaped while wedged, or first call
                d["members"][key] = {"pid": self.pid, "member": self.member,
                                     "weight": self.weight, "pinned": self.pinned}
            d["members"][key]["ts"] = wall
            d["total"] = self.total
            return self._compute(d)

        val = self._mutate(fn)
        self._cached_share, self._cached_at = val, now
        if val != self._last_logged:
            self._log(val)
            self._last_logged = val
        return val

    def _log(self, val: int) -> None:
        try:
            self.trajectory.parent.mkdir(parents=True, exist_ok=True)
            with open(self.trajectory, "a") as fh:
                fh.write(json.dumps({
                    "t": round(time.time(), 1), "member": self.member, "pid": self.pid,
                    "share": val, "total": self.total,
                    "pinned": self.pinned, "weight": self.weight,
                }) + "\n")
        except OSError:
            pass  # auditing must never break the run it audits

    def snapshot(self) -> dict:
        def fn(d: dict) -> dict:
            self._reap(d, time.time())
            return {"total": d.get("total", self.total),
                    "members": {k: dict(v) for k, v in d["members"].items()}}
        return self._mutate(fn)
