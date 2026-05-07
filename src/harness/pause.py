"""Cooperative pause/resume for long-running evolution.

Use case: the operator wants to physically move the laptop (e.g., briefly
walk into a no-wifi area) without losing in-flight work. They drop a flag
file before disconnecting; the harness blocks the next turn's API call
until the flag is removed; on return, the loop continues seamlessly.

Mechanism:

    Pause:
        $ touch /tmp/evoskill_pause            # any running run picks it up
        # log shows: [PAUSE] Pause requested. Holding next API call...
        # log shows: [PAUSE] PAUSED. Safe to disconnect. (after current
        #            turn's response is fully received)
    Resume:
        $ rm /tmp/evoskill_pause
        # log shows: [PAUSE] Resumed.

Granularity: pause activates AFTER the SDK delivers the current turn's
AssistantMessage but BEFORE the harness pulls the next event from the
async iterator. The SDK's underlying conversation is async-iterator-driven
(pull-based), so blocking the iterator effectively blocks the next
`messages.create` API call. In-flight HTTP requests are not aborted —
the current turn's tokens are still billed.

Override the flag location with EVOSKILL_PAUSE_FLAG env var if you need
to coordinate multiple concurrent runs (each can watch a distinct file).
"""
from __future__ import annotations

import asyncio
import contextvars
import os
import sys
import time
from pathlib import Path

_DEFAULT_FLAG = "/tmp/evoskill_pause"
# Poll interval for the flag-file. Short enough to feel responsive on
# resume; long enough to keep CPU near-zero during long pauses.
_POLL_INTERVAL_S = 1.0
# Track whether we've already logged "PAUSED" for the current pause cycle.
# Reset to False every time the flag is removed. Prevents repeated log
# spam while paused, which would happen if multiple turns simultaneously
# hit the wait point.
_already_logged_paused: dict[str, bool] = {}

# Contextvar carrying the currently-active asyncio.Timeout for the
# enclosing agent.run() invocation. When set, wait_if_paused() will extend
# the deadline by the pause duration so a long pause doesn't eat into the
# 720s wall-clock budget. Set by src/harness/agent.py inside its
# `async with asyncio.timeout(...) as tm:` block. Reads None when no
# active timeout is in scope (e.g. external callers using the pause
# helper outside an agent.run).
active_timeout: contextvars.ContextVar["asyncio.Timeout | None"] = (
    contextvars.ContextVar("evoskill_active_timeout", default=None)
)


def pause_flag_path() -> Path:
    """Return the active pause-flag path (env-overridable)."""
    return Path(os.environ.get("EVOSKILL_PAUSE_FLAG", _DEFAULT_FLAG))


def is_pause_requested() -> bool:
    """True if a pause has been requested via the flag file."""
    return pause_flag_path().exists()


async def wait_if_paused(reason: str = "") -> None:
    """Block until the pause flag is removed. No-op if no flag.

    Also FREEZES the agent.run() wall-clock budget (720s by default)
    while paused: on resume, the active asyncio.timeout's deadline is
    extended by exactly the pause duration so the budget consumed during
    the pause is refunded. This means a multi-hour pause (e.g. overnight)
    won't time out the in-flight sample on resume — the agent picks up
    with the same remaining budget it had at pause time.

    Args:
        reason: short context string (e.g. "between turns", "between
                samples") shown in the [PAUSE] log line so the operator
                knows where the loop is parked.
    """
    flag = pause_flag_path()
    if not flag.exists():
        return

    flag_key = str(flag)
    if not _already_logged_paused.get(flag_key, False):
        _already_logged_paused[flag_key] = True
        ctx = f" ({reason})" if reason else ""
        print(
            f"[PAUSE] PAUSED{ctx}. Safe to disconnect. "
            f"`rm {flag}` to resume. (wall-clock budget will be refunded "
            f"on resume — pause for as long as you need)",
            file=sys.stderr,
            flush=True,
        )

    paused_since = time.time()
    while flag.exists():
        await asyncio.sleep(_POLL_INTERVAL_S)

    elapsed = time.time() - paused_since

    # Refund the pause duration to the active asyncio.timeout, if any.
    # `tm.when()` is the loop-monotonic deadline; bumping it by `elapsed`
    # gives back exactly the time we spent parked. If no active timeout
    # is in scope (caller used the pause helper outside agent.run), this
    # is a silent no-op.
    tm = active_timeout.get()
    refunded = False
    if tm is not None:
        try:
            current_deadline = tm.when()
            if current_deadline is not None:
                tm.reschedule(current_deadline + elapsed)
                refunded = True
        except Exception:
            # Defensive: if the timeout is in a state where rescheduling
            # fails (e.g., already fired), don't crash the agent.
            pass

    _already_logged_paused[flag_key] = False
    refund_note = " (wall-clock budget refunded)" if refunded else ""
    print(
        f"[PAUSE] Resumed after {elapsed:.0f}s.{refund_note}",
        file=sys.stderr,
        flush=True,
    )
