"""Bug M-10: SSE keep-alives must not count as generation progress.

`Progress.touch()` does double duty -- it resets the stall detector AND charges
the elapsed gap to the clean budget. Touching on every SSE line therefore made
a hung-but-still-pinging provider invisible twice: idle never grew (no
INACTIVITY kill) while clean_s accrued at wall-clock rate until it hit
CLEAN_BUDGET. CLEAN_BUDGET is deliberately NOT infra, so the phantom failure
was scored as a genuine 0 and cached -- a wrong answer made permanent.

These tests pin the predicate and the resulting liveness behaviour.
"""

from __future__ import annotations

import asyncio

import pytest

from src.harness import liveness
from src.harness.direct.executor import _is_sse_progress, _iter_sse_lines


# --------------------------------------------------------------------------
# the predicate
# --------------------------------------------------------------------------

@pytest.mark.parametrize("line", [
    'data: {"choices":[{"delta":{"content":"hi"}}]}',
    "data: [DONE]",
])
def test_payload_lines_are_progress(line):
    assert _is_sse_progress(line) is True


@pytest.mark.parametrize("line", [
    "",                              # frame separator
    ": OPENROUTER PROCESSING",       # the keep-alive that caused M-10
    ":",                             # bare comment
    ": PROCESSING",
    "event: message",                # metadata, not payload
    "id: 42",
    "retry: 1000",
])
def test_keepalives_blanks_and_metadata_are_not_progress(line):
    assert _is_sse_progress(line) is False


def test_predicate_matches_what_the_consumer_accepts():
    """The consumer skips anything not starting with 'data: ' -- stay in sync."""
    for line in ['data: {"a":1}', "data: [DONE]"]:
        assert _is_sse_progress(line) == line.startswith("data: ")
    for line in ["", ": PROCESSING", "event: x", "data:no-space"]:
        assert _is_sse_progress(line) == line.startswith("data: ")


# --------------------------------------------------------------------------
# end-to-end through _iter_sse_lines
# --------------------------------------------------------------------------

class _FakeResp:
    """Minimal httpx.Response stand-in exposing aiter_lines()."""

    def __init__(self, lines):
        self._lines = list(lines)

    def aiter_lines(self):
        async def gen():
            for ln in self._lines:
                yield ln
        return gen()


async def _drain(lines):
    out = []
    async for ln in _iter_sse_lines(_FakeResp(lines), silence_timeout=30):
        out.append(ln)
    return out


def test_all_lines_are_still_yielded_downstream():
    """The fix must change heartbeats only -- never what the consumer sees."""
    lines = [": OPENROUTER PROCESSING", "", 'data: {"x":1}', "", "data: [DONE]"]
    assert asyncio.run(_drain(lines)) == lines


def test_keepalive_only_stream_records_no_heartbeat():
    prog = liveness.Progress(name="t", max_clean_gap=90.0)
    token = liveness._current.set(prog)
    try:
        asyncio.run(_drain([": OPENROUTER PROCESSING"] * 25 + [""] * 25))
    finally:
        liveness._current.reset(token)
    assert prog.events == 0, "keep-alives must not touch the liveness heartbeat"
    assert prog.kinds.get("sse", 0) == 0


def test_payload_stream_records_one_heartbeat_per_payload_line():
    prog = liveness.Progress(name="t", max_clean_gap=90.0)
    token = liveness._current.set(prog)
    try:
        asyncio.run(_drain([
            ": PROCESSING", "", 'data: {"a":1}', "", 'data: {"b":2}', "",
            ": PROCESSING", "data: [DONE]",
        ]))
    finally:
        liveness._current.reset(token)
    assert prog.kinds.get("sse", 0) == 3, "3 payload lines -> 3 heartbeats"


def test_mixed_stream_ignores_the_keepalives_but_counts_the_data():
    prog = liveness.Progress(name="t", max_clean_gap=90.0)
    token = liveness._current.set(prog)
    try:
        asyncio.run(_drain([": PROCESSING"] * 100 + ['data: {"a":1}']))
    finally:
        liveness._current.reset(token)
    assert prog.kinds.get("sse", 0) == 1


def test_a_keepalive_only_stall_is_classified_INFRA_not_a_genuine_zero():
    """The whole point: such a stall must be retried, never cached as a 0.

    Before the fix the keep-alives kept idle_s pinned near zero, so the guard
    could only ever exit via CLEAN_BUDGET (is_infra False -> cacheable 0).
    With the fix nothing touches, idle_s grows, and INACTIVITY_KILL fires --
    and INACTIVITY_KILL is the one reason the harness treats as "we learned
    nothing": excluded from means, never cached, retried.
    """
    async def work():
        # a provider that pings forever and never emits a token
        while True:
            await _drain([": OPENROUTER PROCESSING"])
            await asyncio.sleep(0.02)

    async def scenario():
        return await liveness.run_with_liveness(
            work, name="stalled",
            inactivity_s=0.3, clean_budget_s=30.0, total_s=30.0,
            max_clean_gap_s=5.0, poll_s=0.05,
        )

    result = asyncio.run(scenario())
    assert result.exit_reason is liveness.ExitReason.INACTIVITY_KILL, (
        f"keep-alive-only stall must be INACTIVITY_KILL (infra), got {result.exit_reason}"
    )
    assert result.exit_reason.is_infra is True, "must be excluded and retried, not cached as 0"


def test_a_STREAMING_provider_is_left_alone_by_the_same_watchdog():
    """Control: real payload lines keep the stream alive past the inactivity window."""
    async def work():
        for _ in range(12):
            await _drain(['data: {"delta":"tok"}'])
            await asyncio.sleep(0.05)
        return "finished"

    async def scenario():
        return await liveness.run_with_liveness(
            work, name="streaming",
            inactivity_s=0.3, clean_budget_s=30.0, total_s=30.0,
            max_clean_gap_s=5.0, poll_s=0.05,
        )

    result = asyncio.run(scenario())
    assert result.exit_reason is liveness.ExitReason.COMPLETED
    assert result.value == "finished"
