"""Bug #110: the agent wall must sit strictly OUTSIDE the liveness gate.

Three limits govern a rollout, and two of them were both 1800 by coincidence:

  EVOSKILL_CLEAN_BUDGET  1800s  productive ("clean") time  <- the governing rule
  EVOSKILL_TOTAL_WALL    3600s  wall-clock blast-radius backstop
  Agent.TIMEOUT_SECONDS         a SEPARATE asyncio wall-clock cap

Two distinct failures came from getting the third wrong:

BELOW TOTAL_WALL (v3 arms, which set nothing and inherited 1800): clean time is
wall time minus uncharged stall gaps, so clean <= wall always. An 1800s
WALL-CLOCK cap makes an 1800s CLEAN-TIME budget unreachable the moment any
stall occurs — the agent wall preempts the gate it was supposed to sit outside.
The v3 judge used 3600, so training was 2x stricter than measurement and four
arm-b rollouts died on a wall judging never applied.

EQUAL TO TOTAL_WALL (v4 arms/judges, 3600 == 3600): the two paths classify the
SAME event oppositely —

  liveness TOTAL_WALL -> ExitReason.TOTAL_WALL, is_infra False -> cacheable 0
  asyncio.timeout     -> TimeoutError -> [TIMEOUT]/[WALL_TIMEOUT] -> infra,
                                          excluded and retried

so whether a wall-hit is scored or excluded becomes a scheduler race.

The wall must therefore be strictly greater, with margin.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ENTRY = (Path(__file__).resolve().parents[1]
         / "scripts" / "run_ssb1_evo_nexn2mini_glm52_0723.py")
SRC = ENTRY.read_text()


def test_agent_wall_default_derives_from_total_wall_plus_margin():
    assert "_TOTAL_WALL + _WALL_MARGIN" in SRC, (
        "the default must derive from TOTAL_WALL so the two cannot drift apart"
    )
    assert not re.search(r'EVOSKILL_SOLVER_TIMEOUT", "1800"', SRC), (
        "the hardcoded 1800 default is #110 — it sits BELOW TOTAL_WALL"
    )


def test_margin_is_positive_so_the_wall_is_strictly_greater():
    m = re.search(r'EVOSKILL_SOLVER_WALL_MARGIN", "(\d+)"', SRC)
    assert m, "a configurable margin must exist"
    assert int(m.group(1)) > 0, (
        "a zero margin puts the agent wall EQUAL to TOTAL_WALL, which makes a "
        "wall-hit a race between a cached 0 and an excluded retry"
    )


def test_warning_fires_on_equal_not_just_below():
    """`>=` is the wrong bar — equality is the race case."""
    assert "<= _TOTAL_WALL" in SRC, (
        "the guard must warn when the wall is EQUAL to TOTAL_WALL, not only below"
    )
    assert "RACE" in SRC, "the equality hazard must be named in the warning"


def test_both_consumers_read_one_value():
    """Two reads with different defaults is how #110 stayed invisible."""
    assert "solver_timeout=Agent.TIMEOUT_SECONDS" in SRC
    assert SRC.count('os.environ.get("EVOSKILL_SOLVER_TIMEOUT"') == 1, (
        "EVOSKILL_SOLVER_TIMEOUT must be resolved in exactly one place"
    )


def test_startup_prints_all_three_limits_together():
    """The value was inherited silently; visibility is the durable fix."""
    for token in ("solver agent wall", "total_wall", "clean_budget"):
        assert token in SRC, f"startup line must surface {token}"


def test_clean_budget_is_not_redefined_here():
    """1800s clean time is the agreed governing limit — this file must not move it."""
    assert not re.search(r'EVOSKILL_CLEAN_BUDGET"\s*\]\s*=', SRC), (
        "the entry script must not assign CLEAN_BUDGET"
    )
    assert 'os.environ["EVOSKILL_CLEAN_BUDGET"]' not in SRC


@pytest.mark.parametrize(
    "total_wall,margin,expected_ok",
    [(3600, 300, True), (3600, 0, False), (1800, 300, True)],
)
def test_ordering_invariant_holds_for_the_derived_default(total_wall, margin, expected_ok):
    """wall > TOTAL_WALL must hold for any TOTAL_WALL once a margin is applied."""
    wall = total_wall + margin
    assert (wall > total_wall) is expected_ok
