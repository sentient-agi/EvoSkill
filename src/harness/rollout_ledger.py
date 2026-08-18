"""Which rollouts failed, and why — the guard that makes a tolerant gather safe. (#140)

THE FAILURE THIS CLOSES. Every rollout runner in this repo fans out with a bare
`asyncio.gather(*coros)`. `gather` without `return_exceptions=True` cancels every
sibling the moment one coroutine raises, so a SINGLE transient provider error
destroys an entire leg's in-flight work and the run writes no result JSON.

Measured 2026-08-03 on the sealed SSB v2 test split: `test_skills_v4` died at
64/360 when an `httpx.HTTPStatusError: 503` outlived its retry budget and
propagated out of the rollout coroutine. ~30 in-flight rollouts were lost. The 64
completed ones survived only because each cached record carries
`trace.inline_score`, so the scores could be reconstructed from the RunCache.

WHY `return_exceptions=True` ALONE IS A TRAP, in both directions:

  * Default a failed rollout's score to 0.0 and a provider outage is silently
    recorded as a WRONG ANSWER. That corrupts the exact quantity being measured,
    invisibly and unrecoverably — strictly worse than the crash it replaces,
    because a crash is loud and the cache makes it recoverable.
  * Leave the score `None` and the aggregation (`x >= 0.999`) raises
    `TypeError: '>=' not supported between instances of 'NoneType' and 'float'` —
    the same lost leg, now with a stack trace pointing far from the cause.

So a tolerant gather is only safe when paired with a ledger that keeps failures
DISTINGUISHABLE from real zeros, and a caller that refuses to write a canonical
result JSON while the ledger is non-empty. A leg with failed rollouts is not a
measurement. Relaunching is cheap and monotone: a failed rollout is never cached
(the cache write is unreachable on raise), so a rerun retries exactly the
failures and replays every completed rollout from cache.
"""
from __future__ import annotations

# Message cap. Provider errors can carry an entire HTML error page; the ledger is
# printed to a log and serialised into a PARTIAL json, and neither is improved by
# 40KB of markup. The kind + first 200 chars have always been enough to classify.
_MSG_CHARS = 200


def _one_line(exc: BaseException) -> str:
    """Exception text as ONE line, truncated.

    Not cosmetic. Provider errors are routinely multi-line — httpx's 400/503 text
    is `Client error ... for url ...\\nFor more information check: <link>` — and the
    ledger prints one line per failed rollout. An embedded newline turns a 3-row
    report into 6 rows with no rollout label on half of them, which is exactly
    when someone triaging an outage miscounts how many rollouts actually failed.
    Observed 2026-08-03 in the #140 fault-injection smoke test.
    """
    return " ".join(str(exc).split())[:_MSG_CHARS]


def failure_ledger(idx: list, results: list, scores: dict,
                   killed: list | None = None) -> list:
    """Every rollout that did NOT produce a score, with why.

    idx      — [(task_index, sample_index)] in the SAME ORDER the coroutines were
               submitted to gather. gather preserves submission order in its
               results list, and that ordering is the only thing tying a returned
               exception back to the rollout that raised it.
    results  — gather(..., return_exceptions=True) output.
    scores   — {task_index: [score_or_None] * n_samples}, as the runner built it.
    killed   — [(task_index, sample_index, why)] the runner detected as
               wall-clock kills. Explicit because the trace is only visible
               at the scoring site; see SOURCE 3.

    Returns [(task_index, sample_index, kind, message)], sorted by (task, sample).
    Empty means every rollout scored — the ONLY state in which a caller may write
    a canonical result JSON.
    """
    failures = []
    seen = set()
    # SOURCE 1: an exception object — a failure WITH a cause attached.
    for (i, s), r in zip(idx, results):
        if isinstance(r, BaseException):
            failures.append((i, s, type(r).__name__, _one_line(r)))
            seen.add((i, s))
    # SOURCE 2: a score still None with no exception to explain it. That would be
    # a logic hole in the rollout coroutine — an early return that skips the score
    # assignment. Both sources are consulted because either alone can miss: an
    # exception can be raised after a score is assigned, and a score can be left
    # unset without anything raising. `None` is what makes the aggregation blow up
    # downstream, so it is caught here whether or not a cause is known.
    for (i, s) in idx:
        if (i, s) not in seen and scores[i][s] is None:
            failures.append((i, s, "NoScoreNoException",
                             "rollout returned without assigning a score — "
                             "logic hole in the rollout coroutine"))
    # SOURCE 3: A WALL-CLOCK KILL THAT DID PRODUCE A SCORE (#138/#152).
    #
    # Sources 1 and 2 both key on the ABSENCE of a score, and this failure has one
    # — which is exactly why it slipped past. `Agent.run` converts its wall-clock
    # TimeoutError into a partial AgentTrace rather than re-raising, so the rollout
    # returns normally, the runner takes `inline_score or 0.0`, and a rollout that
    # was KILLED mid-work counts as a solver that tried and failed. It is then
    # cached under a valid key, so every later leg reads it back as a real 0.0.
    #
    # Why that is a measurement error, not merely a lost rollout: wall-clock
    # duration depends on API contention, hence on CONCURRENCY. A binding wall makes
    # the score a function of how many rollouts run in parallel, promoting a
    # throughput knob into a scientific axis. The turn cap is the budget being
    # measured; the wall is a hang guard and must never decide an outcome.
    #
    # Census over 8,464 SSB v2 cache records on 2026-08-04: 5 such rollouts, all
    # burning 41-45 turns, all scored 0.0, one of them in the v1 sealed-test leg
    # whose numbers were reported. Small — and it grows with concurrency, which is
    # exactly what raising throughput does.
    #
    # WHY `killed` IS AN EXPLICIT ARGUMENT AND NOT SNIFFED FROM `results`. The
    # first version of this source walked `results` looking for a trace with
    # is_error + a timeout parse_error. It could never fire: the SSB v2 runner's
    # rollout coroutine assigns into `scores[i][s]` and returns None, so `results`
    # is a list of Nones and there is no trace to inspect. That is the third
    # can-never-fire check written in one day (see #154 and the pooled hard-zero
    # set in ssbv2_dbg_compare.py). Detection therefore lives where the trace
    # actually exists — at the scoring site — and the ledger only aggregates what
    # the runner explicitly reports. A caller that forgets to pass `killed` gets
    # the old behaviour, so this cannot mask an existing failure; the test suite is
    # what proves the wiring is live.
    for (i, s, why) in (killed or []):
        if (i, s) in seen:
            continue
        failures.append((
            i, s, "WallClockKill",
            f"{_one_line(why)} — KILLED by the wall, not by the turn cap, and then "
            f"scored. Raise the wall (derive it from the turn cap) and re-run this "
            f"rollout. Do NOT report the leg."))
        seen.add((i, s))
    # Stable order: gather returns in submission order but the None-sweep appends
    # after it, which would otherwise interleave the two sources arbitrarily.
    return sorted(failures, key=lambda f: (f[0], f[1]))



def summarise(failures: list) -> dict:
    """{kind: count}, for a one-line triage of what went wrong."""
    by_kind: dict[str, int] = {}
    for _i, _s, kind, _m in failures:
        by_kind[kind] = by_kind.get(kind, 0) + 1
    return by_kind
