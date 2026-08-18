"""The tau2 worker<->parent wire protocol. DEPENDENCY-FREE ON PURPOSE.

The worker (`_tau2_worker.py`) runs inside tau2-bench's own venv and imports `tau2` at
module scope; the parent (`solver.py`) runs in EvoSkill's venv where `tau2` does not exist.
So neither file can import the other, and a constant defined in one and retyped in the other
is the two-copies defect that has already bitten this project repeatedly (four walls in #147,
the derived-vs-effective solver wall on 2026-08-06). This module imports NOTHING, so both
venvs can load it, and the constant exists once.

WHY A MARKER AT ALL. The worker returned its result as bare stdout. stdout is a SHARED
channel: the worker's process also hosts tau2, litellm and the domain modules, any of which
may print. Measured 2026-08-06 on the tau3 smoke: 3 of 15 episodes (20%) failed with
"Worker produced non-JSON stdout: Expecting value: line 2 column 1" while their stderr tail
carried policy and transaction text. Those episodes RAN TO COMPLETION and were graded as
infrastructure failures because somebody else wrote to the pipe. Framing the payload makes
the result channel unambiguous without giving any dependency a way to break it.
"""

RESULT_MARKER = "<<<TAU2_WORKER_RESULT_JSON>>>"


def llm_call_budget(wall):
    """The per-LLM-call time budget for an episode whose inner wall is `wall` seconds.

    Returns `(per_call_s, retries, worst_case_turn_s)`. THE THIRD VALUE IS THE POINT:
    it is the longest a single turn can take, and therefore the longest the inner
    wall can go un-noticed after it has already been exceeded.

    WHY THIS LIVES HERE AND NOT IN THE WORKER. Two consumers need the same numbers:
    the worker configures litellm with them, and `solver.py` must size its OUTER
    wall above them. Computing them in the worker and hardcoding a margin in the
    solver is the two-copies defect this module exists to prevent — and it DID
    happen. Measured 2026-08-06 on the tau3 smoke: `wall/3` with `num_retries=2`
    gave a worst-case turn of 1800s against a hardcoded 300s outer margin.

    #175, THE VACUOUS ORDERING. `solver.py` derived `outer = inner + 300` and its
    comment claimed "the outer must always leave the inner room to fire". It does
    not. tau2 evaluates the inner wall in exactly ONE place — `_check_timeout()`,
    called only from `_check_half_duplex_termination` (tau3-bench
    `orchestrator/orchestrator.py:750`), which additionally RETURNS EARLY while
    `to_role == Role.ENV`. So the inner wall can only fire at a turn boundary. If a
    call chain hangs, the next boundary is up to `worst_case_turn` away; when that
    exceeds the margin, the OUTER wall fires first and yields an `is_error` with no
    termination_reason, no cost and no score — an unscoreable episode where tau2's
    own rule says TIMEOUT scores 0.0. Cost: 3 voided episodes (task_094 x2 in the
    e1 val leg, task_040 x1 in e2 train), each of which blocked a whole leg's
    result JSON. Same shape as the four walls of #147 and the derived-vs-effective
    solver wall: an ordering asserted in a comment instead of derived in code.

    THE FRACTION IS LEFT AT wall/3 ON PURPOSE, and this is the part worth reading.
    wall/3 with 2 retries means one call chain may consume the ENTIRE episode wall,
    which is why the ordering has to be established in the margin rather than by
    making calls cheap. The first draft of this fix tightened it to wall/12 (a 150s
    per-call budget at wall=1800) so the worst-case turn would shrink to 450s.
    That was withdrawn: NOTHING HERE MEASURES REAL CALL LATENCY. Phoenix cannot
    settle it either — tau3's per-turn spans are retroactively replayed from
    `evoskill.jsonl` with synthetic ~0.1s durations, so their durations are
    fabricated and a percentile taken from them would be an artifact. Lowering the
    budget on that basis would truncate legitimate calls and silently change what
    the solver is allowed to do — an INSTRUMENT CHANGE disguised as a bug fix,
    across legs already compared against each other. So the LLM budget is
    byte-identical to what every leg so far ran under, and only the outer wall
    moves. Tightening it is a live option once real latency is instrumented, and
    is deliberately a separate change from this one.

    The 60s floor keeps short smoke walls usable, and is why `worst_case_turn` is
    RETURNED rather than recomputed by callers: at small walls the floor makes it
    not a fixed fraction of the wall.
    """
    import os as _os

    per_call = float(_os.environ.get(
        "EVOSKILL_TAU2_LLM_TIMEOUT_S", max(60.0, float(wall) / 3.0)))
    retries = int(_os.environ.get("EVOSKILL_TAU2_LLM_RETRIES", "2"))
    return per_call, retries, (retries + 1) * per_call
