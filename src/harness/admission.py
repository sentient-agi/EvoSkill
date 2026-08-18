"""Dynamic task-admission controller — AIMD concurrency for eval waves.

Governs how many solver TASKS (and hence sandboxes) run at once, adjusting
the limit from live infra signals instead of a static EVOSKILL_CONCURRENCY.
Mirrors the AIMD pattern proven at the LLM-call layer by executor.py's
_AdaptiveConcurrencyLimiter, one level up: that limiter protects OpenRouter,
this one protects sandbox capacity (Daytona quota, local Docker). The two
compose — a task slot here may make several LLM calls there.

Design rules (agreed 2026-07-09):
- React ONLY to infra-class signals (sandbox-create failures, transient
  502-burst rates). Task-level failures never reduce concurrency — a hard
  task says nothing about capacity, and TB2.1 batches are full of hard tasks.
- Multiplicative decrease (halve, floored) with a refractory window so one
  burst doesn't cascade to the floor; additive increase (+1) only after a
  streak of clean completions.
- Every transition is printed (lands in the tee'd run log) and kept in an
  in-memory trajectory for post-run auditing — concurrency has demonstrably
  affected task outcomes on local backends before, so any run using this
  must be able to prove what the limit was at every point.
- On by default (2026-07-11, the maintainer's explicit instruction: "turn on AIMD by
  default in all future runs"). Opt out via EVOSKILL_DYNAMIC_CONCURRENCY=0
  for a launch that needs the byte-identical prior behavior (plain
  asyncio.Semaphore, no adaptive throttling) -- e.g. a smoke test comparing
  against historical fixed-concurrency numbers.
- Probe mode (EVOSKILL_ADMISSION_PROBE=1, default OFF, 2026-07-29): the
  launch-time cap stops being the operating point and becomes a starting
  guess — clean streaks may raise the target PAST it (up to
  EVOSKILL_ADMISSION_PROBE_MAX) as long as the freshest host load sample
  shows CPU headroom (load/core < EVOSKILL_ADMISSION_PROBE_LOAD_MAX), and
  sustained saturation (load/core >= EVOSKILL_ADMISSION_SAT_LOAD on
  consecutive samples) steps the target DOWN by 1. Additive down for load,
  halving reserved for infra bursts: load is a slow, partly self-inflicted
  signal, and halving on it oscillates. Load per core is the primary
  stress signal because the suspected mechanism behind long-tail solver
  timeouts is local CPU contention (containers on the colima VM), which
  connect-error bursts never see. Two processes probing on the same host
  converge TCP-style — both read the same load and both back off on it.
- Grant staggering: even after a slot is admitted, release-to-caller is
  paced against a shared virtual clock (EVOSKILL_ADMISSION_STAGGER_S) so a
  burst of simultaneously-eligible acquisitions (e.g. a full train+val pool
  with nothing previously in flight) doesn't fire N sandbox-creation calls
  in the same instant. The Docker daemon / Daytona control plane is itself
  a finite, rate-sensitive resource, independent of CPU/RAM capacity.
  Bounded by EVOSKILL_ADMISSION_STAGGER_MAX_WAIT_S so no single waiter is
  ever delayed beyond a fixed cap regardless of burst size. Purely additive
  to the AIMD ceiling above — never touches _target/_in_flight.

Threading: acquire()/release() run on the event loop; report_*() may be
called from asyncio.to_thread workers (all Daytona SDK calls are), so state
is guarded by a threading.Lock and waiter wakeup goes through
loop.call_soon_threadsafe.
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any


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


class GrantStagger:
    """Standalone minimum-inter-grant-spacing pacer, shared by every
    admission path that hands out sandbox-creation slots.

    Extracted from DynamicAdmission (2026-07-10) after an adversarial review
    found the original stagger logic lived ONLY inside DynamicAdmission,
    while evaluate.py's MemoryBudgetSemaphore (the docker-backend gate used
    whenever memory_budget_mb+task_memory_fn are set, e.g. every current
    TB2.1/SWE-rebench docker launch) called MemoryBudgetSemaphore.acquire()
    directly and never touched DynamicAdmission at all — so those runs got
    ZERO burst pacing regardless of EVOSKILL_DYNAMIC_CONCURRENCY. Pulling
    the mechanism out into its own lockable, reusable primitive lets both
    gates use the identical pacing behavior instead of one silently
    bypassing it.

    Owns its own threading.Lock rather than reusing a caller's lock: callers
    use different lock types (DynamicAdmission a threading.Lock since
    report_*() runs from asyncio.to_thread workers; MemoryBudgetSemaphore an
    asyncio.Condition) that can't be shared, and this class's own critical
    section is pure arithmetic on two floats — cheap enough that a private
    lock adds no meaningful contention, and it's never held while trying to
    acquire a caller's lock, so nesting introduces no deadlock ordering risk.
    """

    def __init__(self) -> None:
        self.stagger_s = _env_float("EVOSKILL_ADMISSION_STAGGER_S", 0.1)
        self.stagger_max_wait_s = _env_float("EVOSKILL_ADMISSION_STAGGER_MAX_WAIT_S", 5.0)
        self._lock = threading.Lock()
        self._next_grant_ok_at = 0.0  # monotonic; virtual clock, 0.0 = "no debt"
        self.events = 0  # count of reserve() calls actually delayed
        self.total_s = 0.0  # cumulative seconds delayed (for snapshot())

    def reserve(self) -> float:
        """Reserve the next grant slot; returns seconds the caller should
        sleep (outside any lock) before proceeding. Pure pacing layer —
        never touches admission/budget accounting, so it cannot change
        whether a slot is granted, only when its holder gets to run.

        Bursts within stagger_max_wait_s of the virtual clock's current
        debt get pure stagger_s spacing (unchanged from the original
        design). Bursts that would push a waiter's ideal slot PAST the cap
        get their delay wrapped into the [0, cap) window via modulo instead
        of being clamped to exactly `cap` — clamping made every waiter past
        the ~(cap/stagger_s)'th collapse onto the exact same instant
        (t=cap), reproducing the very thundering-herd this mechanism exists
        to prevent, just delayed by `cap` seconds instead of eliminated.
        Modulo keeps them spread at stagger_s spacing, colliding only in
        groups of ceil(overflow / (cap/stagger_s)) — a small, bounded
        cluster instead of the entire overflow tail at once. Confirmed safe
        against the cap invariant: Python's `%` on two positive floats
        always returns a value in [0, divisor), so wrapped delays never
        exceed (in fact never even reach) stagger_max_wait_s.
        """
        if self.stagger_s <= 0.0:
            return 0.0
        with self._lock:
            now = time.monotonic()
            grant_at = max(now, self._next_grant_ok_at)
            self._next_grant_ok_at = grant_at + self.stagger_s
            raw_delay = grant_at - now
            if raw_delay <= self.stagger_max_wait_s:
                delay = raw_delay
            else:
                delay = raw_delay % self.stagger_max_wait_s
            if delay > 0.0:
                self.events += 1
                self.total_s += delay
            return delay

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"stagger_events": self.events, "stagger_total_s": round(self.total_s, 2)}


class DynamicAdmission:
    """AIMD task gate. Drop-in for asyncio.Semaphore's async-acquire /
    sync-release / async-context-manager usage patterns."""

    def __init__(self, initial: int):
        initial = max(1, int(initial))
        # Bounds clamp AROUND the initial value so a caller-provided
        # concurrency outside [min, max] never starts out-of-bounds.
        self.min_limit = min(_env_int("EVOSKILL_CONCURRENCY_MIN", 4), initial)
        self.max_limit = max(_env_int("EVOSKILL_CONCURRENCY_MAX", 16), initial)
        self.refractory_s = _env_float("EVOSKILL_ADMISSION_REFRACTORY_S", 30.0)
        self.clean_streak_target = _env_int("EVOSKILL_ADMISSION_CLEAN_STREAK", 3)
        self.transient_burst = _env_int("EVOSKILL_ADMISSION_TRANSIENT_BURST", 6)
        self.transient_window_s = _env_float("EVOSKILL_ADMISSION_TRANSIENT_WINDOW_S", 60.0)
        # Max events a single source key (one task's sandbox) may contribute
        # to the burst count within a window. A dead/self-broken sandbox
        # fast-fails through up to ~5 exec cycles x 3 retries (~15 events in
        # <60s) — without this cap ONE pathological task (OOM'd its own env,
        # firewall task cutting the toolbox) would halve fleet concurrency,
        # violating the rule that task-level failures never reduce it.
        self.transient_per_key = _env_int("EVOSKILL_ADMISSION_TRANSIENT_PER_KEY", 2)
        # Probe mode (see module docstring). probe_max is clamped to at
        # least max_limit so enabling probe can never LOWER the ceiling.
        self.probe_enabled = os.environ.get("EVOSKILL_ADMISSION_PROBE", "0") == "1"
        self.probe_max = max(_env_int("EVOSKILL_ADMISSION_PROBE_MAX", 60), self.max_limit)
        self.probe_load_max = _env_float("EVOSKILL_ADMISSION_PROBE_LOAD_MAX", 0.85)
        self.sat_load = _env_float("EVOSKILL_ADMISSION_SAT_LOAD", 1.2)
        self.sat_strikes_needed = _env_int("EVOSKILL_ADMISSION_SAT_STRIKES", 2)
        self.load_sample_interval_s = _env_float("EVOSKILL_ADMISSION_LOAD_SAMPLE_S", 5.0)
        self._last_load_sample = 0.0  # monotonic; rate-limits getloadavg
        self._last_load: float | None = None  # freshest load/core, None until sampled
        self._sat_strikes = 0
        # Wall-timeout signal (2026-07-29): windowed like transients but with
        # its own (much lower) threshold and per-source count of 1.
        self.timeout_burst = _env_int("EVOSKILL_ADMISSION_TIMEOUT_BURST", 2)
        self.timeout_window_s = _env_float("EVOSKILL_ADMISSION_TIMEOUT_WINDOW_S", 600.0)
        self._timeout_events: deque[tuple[float, str]] = deque()
        # Minimum spacing enforced between consecutive slot grants, and the
        # hard cap on how long any one acquire() may be delayed enforcing it.
        # Tradeoff point: at these defaults a burst of up to 50 simultaneous
        # acquisitions spreads smoothly ((50-1)*0.1s = 4.9s <= cap); larger
        # task pools that want the same smooth-spread property should raise
        # the cap or lower the spacing rather than rediscovering this math.
        # (Delegated to GrantStagger — see that class for the mechanism;
        # kept as attributes here too since callers/tests read them off the
        # controller directly.)
        self._stagger = GrantStagger()
        self.stagger_s = self._stagger.stagger_s
        self.stagger_max_wait_s = self._stagger.stagger_max_wait_s

        self._lock = threading.Lock()
        self._target = initial
        self._in_flight = 0
        self._clean_streak = 0
        self._last_decrease = 0.0  # monotonic; 0 = never
        self._transient_events: deque[tuple[float, str]] = deque()  # (ts, key)
        self.trajectory: list[dict[str, Any]] = [self._entry(initial, initial, "init")]

        # Lazily bound to the running loop on first acquire.
        self._wake: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    # -- gating ------------------------------------------------------------

    async def acquire(self) -> None:
        loop = asyncio.get_running_loop()
        if self._wake is None or self._loop is not loop:
            # (Re)bind to the current loop — scripts that run multiple
            # sequential asyncio.run() calls would otherwise await an Event
            # attached to a dead loop.
            self._loop = loop
            self._wake = asyncio.Event()
        delay = 0.0
        while True:
            self._wake.clear()
            with self._lock:
                if self._in_flight < self._target:
                    self._in_flight += 1
                    delay = self._reserve_stagger_slot_locked()
                    break
            await self._wake.wait()
        if delay > 0.0:
            try:
                await asyncio.sleep(delay)
            except BaseException:
                # Reserved the slot (in_flight already incremented) but got
                # cancelled/errored mid-stagger. Callers only ever call
                # release() after acquire() RETURNS (evaluate.py's run_one:
                # acquire() sits outside the try/finally that calls release();
                # runner.py's `async with self._solver_semaphore` never enters
                # __aexit__ if __aenter__ raises) — so if we don't roll the
                # reservation back here ourselves, it leaks forever, silently
                # shrinking effective concurrency by one per cancellation.
                self.release()
                raise

    def _reserve_stagger_slot_locked(self) -> float:
        """Call with self._lock held, immediately after incrementing
        _in_flight for a newly granted slot. Returns seconds the caller
        should sleep (outside the lock) before proceeding to solver.run().
        Pure pacing layer over an already-decided admission: never reads or
        writes _in_flight/_target, so it cannot change AIMD ceiling
        behavior — it only delays when an already-admitted slot's holder
        gets to run. Delegates to GrantStagger, which has its own private
        lock (nested safely inside self._lock — GrantStagger never touches
        self._lock, so there's no reverse-order acquisition to deadlock
        on)."""
        return self._stagger.reserve()

    def release(self) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
        self._notify()

    async def __aenter__(self) -> "DynamicAdmission":
        await self.acquire()
        return self

    async def __aexit__(self, *exc) -> None:
        self.release()

    def _notify(self) -> None:
        """Wake waiters; safe from any thread."""
        wake, loop = self._wake, self._loop
        if wake is None or loop is None:
            return
        try:
            if loop.is_closed():
                return
            loop.call_soon_threadsafe(wake.set)
        except RuntimeError:
            pass

    # -- signals (thread-safe) ----------------------------------------------

    def report_infra_failure(self, reason: str) -> None:
        """Hard infra failure (e.g. sandbox creation failed) — halve."""
        self._decrease(f"infra failure: {reason}")

    def report_transient(self, reason: str, key: str | None = None) -> None:
        """Soft infra event (e.g. one 502 retry). Windowed: only a burst
        (>= transient_burst events within transient_window_s) decreases.

        ``key`` identifies the source (e.g. task name / sandbox). Events
        sharing a key count at most ``transient_per_key`` times toward the
        burst, so a burst requires MULTIPLE distinct sources struggling at
        once — the fleet-wide signature of real capacity trouble — and one
        pathological task can never halve concurrency by itself. Callers
        that pass no key fall back to the reason string."""
        now = time.monotonic()
        k = key if key is not None else reason
        with self._lock:
            self._transient_events.append((now, k))
            while self._transient_events and now - self._transient_events[0][0] > self.transient_window_s:
                self._transient_events.popleft()
            counts: dict[str, int] = {}
            for _, ek in self._transient_events:
                counts[ek] = counts.get(ek, 0) + 1
            effective = sum(min(c, self.transient_per_key) for c in counts.values())
            burst = effective >= self.transient_burst
            if burst:
                self._transient_events.clear()
        if burst:
            self._decrease(
                f"transient burst (>= {self.transient_burst}/{self.transient_window_s:.0f}s "
                f"across {len(counts)} sources, <= {self.transient_per_key}/source): {reason}"
            )

    def _sample_load(self) -> float | None:
        """Rate-limited host load-average-per-core sample. Returns the fresh
        value, or None when inside the sample interval or getloadavg is
        unavailable. The rate limit keeps completion storms from hammering
        the syscall AND gives saturation strikes a time dimension: two
        strikes means >= 2 * interval of sustained saturation, not two
        completions in the same instant."""
        now = time.monotonic()
        with self._lock:
            if now - self._last_load_sample < self.load_sample_interval_s:
                return None
            self._last_load_sample = now
        try:
            load = os.getloadavg()[0] / max(1, os.cpu_count() or 1)
        except OSError:
            return None
        with self._lock:
            self._last_load = load
        return load

    def report_timeout(self, reason: str, key: str | None = None) -> None:
        """A sample hit its wall-clock timeout. Empirically (2026-07-29 span
        audit, 2161 v1 rollouts) p50 runtime is ~1 min and p90 ~3-4 min, so a
        720s+ wall hit is a >10x outlier — strong congestion evidence, but
        not proof: singleton wall hits occurred even at concurrency 15.
        Windowed with a low burst threshold (default 2 distinct sources /
        600s) and a HALVE — by the time two distinct tasks blow the wall the
        fleet has been degraded for many minutes. Each source counts once,
        so one intrinsically-slow task rerun repeatedly can never halve the
        fleet (same design rule as transients)."""
        now = time.monotonic()
        k = key if key is not None else reason
        with self._lock:
            self._timeout_events.append((now, k))
            while self._timeout_events and now - self._timeout_events[0][0] > self.timeout_window_s:
                self._timeout_events.popleft()
            distinct = len({ek for _, ek in self._timeout_events})
            burst = distinct >= self.timeout_burst
            if burst:
                self._timeout_events.clear()
        if burst:
            self._decrease(
                f"wall-timeout burst ({distinct} distinct sources"
                f"/{self.timeout_window_s:.0f}s): {reason}")

    def report_clean_completion(self) -> None:
        """A task finished with no infra trouble. After a streak, +1.

        Probe mode: the effective ceiling is probe_max instead of max_limit,
        but ANY increase additionally requires the freshest load sample to
        show headroom — increasing into known saturation is wrong at every
        ceiling. Sustained saturation steps the target down by 1 (additive;
        halving stays reserved for infra bursts). A saturated completion
        neither feeds the clean streak nor counts as clean evidence."""
        if self.probe_enabled:
            load = self._sample_load()
            if load is not None and load >= self.sat_load:
                with self._lock:
                    self._sat_strikes += 1
                    sat = self._sat_strikes >= self.sat_strikes_needed
                    if sat:
                        self._sat_strikes = 0
                if sat:
                    self._decrease_step(
                        f"load saturation ({load:.2f}/core >= {self.sat_load:.2f})")
                return
            if load is not None:
                with self._lock:
                    self._sat_strikes = 0
        changed = False
        with self._lock:
            if self.probe_enabled:
                headroom = self._last_load is not None and self._last_load < self.probe_load_max
                ceiling = self.probe_max if headroom else 0  # no sample / no headroom: hold
            else:
                ceiling = self.max_limit
            self._clean_streak += 1
            now = time.monotonic()
            if (
                self._clean_streak >= self.clean_streak_target
                and self._target < ceiling
                and (self._last_decrease == 0.0 or now - self._last_decrease >= self.refractory_s)
            ):
                old, self._target = self._target, self._target + 1
                self._clean_streak = 0
                self.trajectory.append(self._entry(old, self._target, f"clean streak x{self.clean_streak_target}"))
                changed = True
                new = self._target
        if changed:
            print(f"  [admission] concurrency {old}→{new} (clean streak)", flush=True)
            self._notify()

    def _decrease_step(self, reason: str) -> None:
        """Additive step-down (-1), the decrease half of probe mode. Shares
        the refractory window with _decrease so load stepping and burst
        halving can never stack inside one window."""
        with self._lock:
            now = time.monotonic()
            if self._last_decrease and now - self._last_decrease < self.refractory_s:
                return
            old = self._target
            new = max(self.min_limit, old - 1)
            if new >= old:
                return
            self._target = new
            self._last_decrease = now
            self._clean_streak = 0
            self.trajectory.append(self._entry(old, new, reason))
        print(f"  [admission] concurrency {old}→{new} ({reason})", flush=True)

    def _decrease(self, reason: str) -> None:
        with self._lock:
            now = time.monotonic()
            if self._last_decrease and now - self._last_decrease < self.refractory_s:
                return  # refractory: at most one decrease per window
            old = self._target
            new = max(self.min_limit, old // 2)
            if new >= old:
                return
            self._target = new
            self._last_decrease = now
            self._clean_streak = 0
            self.trajectory.append(self._entry(old, new, reason))
        print(f"  [admission] concurrency {old}→{new} ({reason})", flush=True)

    # -- introspection -------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        stagger_snap = self._stagger.snapshot()
        with self._lock:
            return {
                "target": self._target,
                "in_flight": self._in_flight,
                "clean_streak": self._clean_streak,
                "min": self.min_limit,
                "max": self.probe_max if self.probe_enabled else self.max_limit,
                "probe": self.probe_enabled,
                "last_load_per_core": self._last_load,
                "transitions": len(self.trajectory) - 1,
                "stagger_events": stagger_snap["stagger_events"],
                "stagger_total_s": stagger_snap["stagger_total_s"],
            }

    @staticmethod
    def _entry(old: int, new: int, reason: str) -> dict[str, Any]:
        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "old": old,
            "new": new,
            "reason": reason,
        }


# -- module singleton ---------------------------------------------------------

_controller: DynamicAdmission | None = None
_controller_lock = threading.Lock()


def dynamic_enabled() -> bool:
    return os.environ.get("EVOSKILL_DYNAMIC_CONCURRENCY", "1") != "0"


def get_task_gate(max_concurrent: int):
    """Return the task-concurrency gate for an eval wave.

    Dynamic mode (EVOSKILL_DYNAMIC_CONCURRENCY=1): one process-wide
    DynamicAdmission shared by every wave — task slots model a global
    resource (sandbox capacity), so the runner's failure-batch gate and
    evaluate_agent_parallel's per-call gate must draw from the same pool.
    The first caller's limit seeds the initial target; later callers with a
    different limit get the same shared instance (logged once).

    Default (off): a fresh asyncio.Semaphore(max_concurrent), byte-identical
    to the behavior before this module existed.
    """
    if not dynamic_enabled():
        return asyncio.Semaphore(max_concurrent)
    global _controller
    with _controller_lock:
        if _controller is None:
            _controller = DynamicAdmission(max_concurrent)
            print(
                f"  [admission] dynamic concurrency ON: init={max_concurrent} "
                f"range=[{_controller.min_limit},{_controller.max_limit}] "
                f"refractory={_controller.refractory_s:.0f}s "
                f"streak={_controller.clean_streak_target} "
                f"burst={_controller.transient_burst}/{_controller.transient_window_s:.0f}s "
                f"(<= {_controller.transient_per_key}/source) "
                f"stagger={_controller.stagger_s:.2f}s/cap={_controller.stagger_max_wait_s:.1f}s"
                + (
                    f" probe=ON max={_controller.probe_max}"
                    f" load<{_controller.probe_load_max:.2f}"
                    f" sat>={_controller.sat_load:.2f}x{_controller.sat_strikes_needed}"
                    if _controller.probe_enabled else ""
                ),
                flush=True,
            )
        elif _controller.snapshot()["target"] != max_concurrent:
            pass  # shared instance intentionally keeps its own adjusted target
        return _controller


def get_controller() -> DynamicAdmission | None:
    """The live controller if dynamic mode is on and a gate was created."""
    return _controller


def report_infra_failure(reason: str) -> None:
    c = _controller
    if c is not None:
        c.report_infra_failure(reason)


def report_transient(reason: str, key: str | None = None) -> None:
    c = _controller
    if c is not None:
        c.report_transient(reason, key=key)


def report_timeout(reason: str, key: str | None = None) -> None:
    c = _controller
    if c is not None:
        c.report_timeout(reason, key=key)


def report_clean_completion() -> None:
    c = _controller
    if c is not None:
        c.report_clean_completion()
