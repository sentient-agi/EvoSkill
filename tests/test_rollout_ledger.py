"""#140: a tolerant gather is only safe if failures stay distinct from real zeros.

The property under test is not "the ledger lists exceptions" — it is the pair of
confusions that would corrupt a measurement:

    a failed rollout counted as a WRONG ANSWER   (score 0.0)  -> silent, fatal
    a real wrong answer counted as a FAILURE     (retried)    -> inflates scores

Both are single-character mistakes in `failure_ledger`, so each has a test that
FAILS if that character changes. Mutants exercised at the bottom.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.harness.rollout_ledger import failure_ledger, summarise  # noqa: E402

N = 3


def _idx(n_tasks: int, n_samples: int = N) -> list:
    """The submission order the runner uses: task outer, sample inner."""
    return [(i, s) for i in range(n_tasks) for s in range(n_samples)]


def _scores(n_tasks: int, n_samples: int = N) -> dict:
    return {i: [None] * n_samples for i in range(n_tasks)}


def _all_scored(n_tasks: int, value: float = 1.0, n_samples: int = N) -> dict:
    return {i: [value] * n_samples for i in range(n_tasks)}


def test_a_clean_leg_produces_an_empty_ledger():
    """Empty is the ONLY state in which a canonical result JSON may be written."""
    n = 4
    assert failure_ledger(_idx(n), [None] * (n * N), _all_scored(n)) == []


def test_a_genuine_zero_is_NOT_a_failure():
    """THE CORRUPTION GUARD, direction 1.

    A rollout that ran and got the answer wrong scores 0.0. If the ledger treated
    falsy scores as missing, every wrong answer would be reported as an infra
    failure, the leg would never be allowed to complete, and rerunning would
    resample only the failures — inflating the score without bound.
    """
    n = 3
    scores = _all_scored(n, value=0.0)
    assert failure_ledger(_idx(n), [None] * (n * N), scores) == []


def test_a_failed_rollout_is_never_reported_as_a_zero():
    """THE CORRUPTION GUARD, direction 2.

    The failed rollout must surface in the ledger, and its score must remain None
    rather than being coerced to 0.0 by anything the ledger does. A provider 503
    recorded as a wrong answer is unrecoverable and invisible.
    """
    n = 2
    scores = _all_scored(n)
    scores[1][2] = None                     # the rollout that raised
    results = [None] * (n * N)
    results[1 * N + 2] = RuntimeError("503 upstream")

    led = failure_ledger(_idx(n), results, scores)

    assert len(led) == 1
    i, s, kind, msg = led[0]
    assert (i, s) == (1, 2)
    assert kind == "RuntimeError"
    assert "503" in msg
    assert scores[1][2] is None, "the ledger must not write a score"


def test_the_exception_maps_back_to_the_RIGHT_rollout():
    """Submission order is the only link between a returned exception and its
    rollout. An off-by-one here would blame the wrong task, and since the ledger
    drives which rollouts a rerun retries, it would retry the wrong one forever
    while the real failure stayed unfixed."""
    n = 5
    scores = _all_scored(n)
    # task 3, sample 1 -> flat position 3*3+1 = 10
    scores[3][1] = None
    results = [None] * (n * N)
    results[10] = ValueError("boom")

    led = failure_ledger(_idx(n), results, scores)
    assert [(f[0], f[1]) for f in led] == [(3, 1)]


def test_a_None_score_with_no_exception_is_caught_as_a_logic_hole():
    """The second source. A rollout that returns early without assigning a score
    raises nothing, so gather reports success — but the None then reaches the
    aggregation and raises TypeError far from the cause. Caught here instead."""
    n = 2
    scores = _all_scored(n)
    scores[0][1] = None

    led = failure_ledger(_idx(n), [None] * (n * N), scores)

    assert len(led) == 1
    assert led[0][:3] == (0, 1, "NoScoreNoException")


def test_an_exception_AND_a_None_for_one_rollout_is_reported_ONCE():
    """Both sources fire on the same rollout — the usual case, since a raising
    coroutine also leaves its score unset. Reporting it twice would double the
    failure count in the PARTIAL file and in the operator-facing summary."""
    n = 1
    scores = {0: [1.0, None, 1.0]}
    results = [None, OSError("disk"), None]

    led = failure_ledger(_idx(n), results, scores)

    assert len(led) == 1
    assert led[0][2] == "OSError", "the cause must win over the generic sweep"


def test_an_exception_raised_AFTER_the_score_landed_still_counts_as_a_failure():
    """Why both sources are needed rather than just the None-sweep. In the real
    coroutine the score is assigned and THEN `cache.set` runs; if the cache write
    raises, the score exists but the rollout never got persisted. It is not a
    clean rollout and must not be silently accepted."""
    n = 1
    scores = {0: [1.0, 1.0, 1.0]}           # every score landed
    results = [None, IOError("cache write failed"), None]

    led = failure_ledger(_idx(n), results, scores)
    assert [(f[0], f[1], f[2]) for f in led] == [(0, 1, "OSError")]


def test_ledger_is_sorted_so_two_sources_do_not_interleave_arbitrarily():
    n = 3
    scores = _all_scored(n)
    scores[0][0] = None                     # source 2 (no exception)
    scores[2][1] = None                     # source 1 (exception)
    results = [None] * (n * N)
    results[2 * N + 1] = RuntimeError("x")

    led = failure_ledger(_idx(n), results, scores)
    assert [(f[0], f[1]) for f in led] == [(0, 0), (2, 1)]


def test_long_provider_html_is_truncated():
    """Provider errors can carry a whole HTML error page; the ledger goes into a
    log and a json and neither is improved by 40KB of markup."""
    n = 1
    scores = {0: [None, 1.0, 1.0]}
    results = [RuntimeError("<html>" + "x" * 40_000), None, None]

    led = failure_ledger(_idx(n), results, scores)
    assert len(led[0][3]) == 200


def test_multiline_provider_error_is_flattened_to_one_line():
    """Regression from the #140 fault-injection smoke test: httpx's real 400 text
    is two lines, and the ledger prints one line per failed rollout. Left
    unflattened, a 3-failure report renders as 6 lines with no rollout label on
    half of them — someone triaging an outage then miscounts the failures."""
    n = 1
    real_httpx_text = ("Client error '400 Bad Request' for url "
                       "'https://openrouter.ai/api/v1/chat/completions'\n"
                       "For more information check: https://developer.mozilla.org/")
    scores = {0: [None, 1.0, 1.0]}
    results = [RuntimeError(real_httpx_text), None, None]

    msg = failure_ledger(_idx(n), results, scores)[0][3]
    assert "\n" not in msg and "\r" not in msg
    assert "400 Bad Request" in msg
    assert "For more information" in msg, "flattening must not drop the tail"


def test_summarise_counts_by_kind():
    led = [(0, 0, "RuntimeError", "a"), (0, 1, "RuntimeError", "b"),
           (1, 0, "OSError", "c")]
    assert summarise(led) == {"RuntimeError": 2, "OSError": 1}


def test_BaseException_not_just_Exception_is_caught():
    """gather(return_exceptions=True) can hand back a BaseException subclass, and
    `except Exception` style filtering would let it through as a success — the
    score would then be None and the aggregation would raise."""
    n = 1
    scores = {0: [None, 1.0, 1.0]}
    results = [KeyboardInterrupt(), None, None]

    led = failure_ledger(_idx(n), results, scores)
    assert led[0][2] == "KeyboardInterrupt"


def test_the_None_sentinel_is_load_bearing():
    """Documents WHY None (not 0.0) is the right initial value: the aggregation
    the runner performs is a TypeError on None, so an unscored rollout can never
    slip through as a legitimate result. If this ever stops raising, the ledger
    becomes the only guard instead of one of two."""
    with pytest.raises(TypeError):
        _ = None >= 0.999


# ---------------------------------------------------------------------------
# MUTANTS. A test suite that passes against a broken implementation is worthless,
# so each mutant is the specific wrong version I might plausibly have written.
# ---------------------------------------------------------------------------

def _mutant_falsy_instead_of_is_none(idx, results, scores):
    """`if not scores[i][s]` instead of `is None` — the classic. Treats a real
    0.0 (wrong answer) as a missing score."""
    failures, seen = [], set()
    for (i, s), r in zip(idx, results):
        if isinstance(r, BaseException):
            failures.append((i, s, type(r).__name__, str(r)[:200]))
            seen.add((i, s))
    for (i, s) in idx:
        if (i, s) not in seen and not scores[i][s]:
            failures.append((i, s, "NoScoreNoException", "x"))
    return sorted(failures, key=lambda f: (f[0], f[1]))


def _mutant_no_seen_set(idx, results, scores):
    """Drops the dedup — double-counts the common case."""
    failures = []
    for (i, s), r in zip(idx, results):
        if isinstance(r, BaseException):
            failures.append((i, s, type(r).__name__, str(r)[:200]))
    for (i, s) in idx:
        if scores[i][s] is None:
            failures.append((i, s, "NoScoreNoException", "x"))
    return sorted(failures, key=lambda f: (f[0], f[1]))


def _mutant_exception_only(idx, results, scores):
    """Trusts gather alone — misses the no-exception logic hole."""
    return sorted([(i, s, type(r).__name__, str(r)[:200])
                   for (i, s), r in zip(idx, results)
                   if isinstance(r, BaseException)],
                  key=lambda f: (f[0], f[1]))


@pytest.mark.parametrize("mutant,should_fail_on", [
    (_mutant_falsy_instead_of_is_none, "genuine_zero"),
    (_mutant_no_seen_set, "double_count"),
    (_mutant_exception_only, "logic_hole"),
])
def test_each_mutant_is_killed(mutant, should_fail_on):
    """Every mutant must differ from the real implementation on at least one of
    the scenarios above. If a mutant survives, the corresponding test is vacuous."""
    n = 3
    if should_fail_on == "genuine_zero":
        idx, results, scores = _idx(n), [None] * (n * N), _all_scored(n, 0.0)
    elif should_fail_on == "double_count":
        idx, results, scores = _idx(1), [None, OSError("d"), None], {0: [1.0, None, 1.0]}
    else:
        idx, results, scores = _idx(1), [None] * N, {0: [1.0, None, 1.0]}

    real = failure_ledger(idx, results, scores)
    mut = mutant(idx, results, scores)
    assert real != mut, f"mutant survived on {should_fail_on} — that test is vacuous"


# --- SOURCE 3: WALL-CLOCK KILLS (#138/#152) ----------------------------------
# Written because the FIRST implementation could never fire. It walked `results`
# for a trace with is_error + a timeout parse_error — but the SSB v2 runner's
# rollout coroutine assigns into scores[i][s] and returns None, so `results` is a
# list of Nones and there was no trace to inspect. Third can-never-fire check in
# one day (cf. #154's tree_hash identity, and the pooled hard-zero set in
# ssbv2_dbg_compare.py). These tests exist so the wiring is PROVEN live, not
# assumed: a mutation that stops the runner passing `killed` turns them red.

def test_wall_kill_is_a_failure_even_though_it_scored():
    """The defining property: sources 1 and 2 key on a MISSING score; this has one."""
    idx = [(0, 0), (0, 1)]
    results = [None, None]                  # what the real coroutine returns
    scores = {0: [0.0, 0.0]}                # both scored — neither is None
    assert failure_ledger(idx, results, scores) == []          # invisible without it
    f = failure_ledger(idx, results, scores,
                       killed=[(0, 1, "Query timed out after 3600s (burned 43 turns)")])
    assert len(f) == 1
    assert f[0][:3] == (0, 1, "WallClockKill")
    assert "burned 43 turns" in f[0][3]
    assert "must not decide an outcome" in f[0][3] or "Do NOT report the leg" in f[0][3]


def test_wall_kill_does_not_double_report_an_exception():
    """A rollout that raised is already Source 1; it must not appear twice."""
    idx = [(0, 0)]
    boom = RuntimeError("503")
    f = failure_ledger(idx, [boom], {0: [None]},
                       killed=[(0, 0, "Query timed out after 3600s")])
    assert len(f) == 1 and f[0][2] == "RuntimeError"


def test_no_killed_argument_preserves_old_behaviour():
    """Omitting `killed` must not invent failures — it degrades, never masks."""
    idx = [(0, 0)]
    assert failure_ledger(idx, [None], {0: [1.0]}) == []


def test_the_runner_actually_passes_killed():
    """The wiring, asserted against the runner's SOURCE — not its behaviour.

    A perfect ledger is useless if nothing calls it with `killed`. This is the
    check whose absence made version one vacuous.
    """
    import re
    root = Path(__file__).resolve().parent.parent
    src = (root / "scripts/run_ssbv2_base30_dsv4f_n3.py").read_text()
    assert re.search(r"failure_ledger\([^)]*killed=wall_killed", src), \
        "the SSB v2 runner must pass killed= to failure_ledger"
    # BOTH scoring paths must detect: fresh rollouts AND cache hits. A wall kill
    # cached by an earlier leg replays as an ordinary 0.0 — that is how all 5
    # census records reached reported numbers.
    assert len(re.findall(r'"timed out" in _\w+\.lower\(\)', src)) == 2, \
        "wall-kill detection must cover the fresh path AND the cache-hit path"
