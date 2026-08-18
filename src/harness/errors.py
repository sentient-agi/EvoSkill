"""Exceptions that must never be turned into a score.

WHY THIS EXISTS. On 2026-08-18 `RecalculationFailed` was added to the SSB v2 scorer so a
grading-container fault could not be reported as a solver's 0.0. Its docstring said
"deliberately NOT caught anywhere in this module". It was caught four times -- by broad
`except Exception: return 0.0` handlers at both live scoring call sites, at `score_ssbv2`,
and in the generic evaluator -- and every one of them converted it straight back into the
0.0 it was written to prevent. Measured: with the verifier image made unresolvable,
`score_ssbv2` returned 0.0.

The lesson is that a convention held only by a docstring is not held at all. An exception
class alone cannot express "do not swallow me"; the CALLERS have to agree, and there were
four of them with more to come. So the contract gets a name that a caller can catch
explicitly, and every broad handler in a scoring path re-raises it first.

An UngradableEpisode means: this episode has NO computable verdict. It is not a wrong
answer and it is not a zero. Let it propagate. The runners gather with
return_exceptions=True and `src/harness/rollout_ledger.failure_ledger` records any
exception as a failure -- and an empty ledger is the only state in which a caller may
write a canonical result JSON. So propagation lands the episode in the ledger, with its
cause attached, and blocks the leg from producing a result that silently contains it.
"""
from __future__ import annotations


class UngradableEpisode(RuntimeError):
    """No verdict could be computed for this episode.

    Deliberately an Exception (not BaseException) so `asyncio.gather(...,
    return_exceptions=True)` still captures it into the failure ledger rather than
    tearing down the whole leg. What must not happen is a broad `except Exception`
    silently scoring it; every such handler in a scoring path carries an explicit
    `except UngradableEpisode: raise` ahead of itself, and tests assert that.
    """
