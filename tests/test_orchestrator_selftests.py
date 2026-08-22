"""The AO's own guards, run by the suite.

Every module under `src/orchestrator/` that owns a correctness guard ships it as a
`selftest()` behind a `__main__` entry point, and each of those guards is built the
right way round: it pairs every check with a KNOWN-BAD fixture and asserts the check
fails on it. `spec.selftest`'s docstring says so outright, and `loop.py`'s section
header reads "every verify must fail on a known-bad input".

They were never collected by pytest. Ten guards, ~1s of total runtime, sitting behind
`python3 -m ...` invocations that only ran when someone typed them. A guard nobody runs
is #154 with extra steps -- one of these modules contains a `selftest()` whose own
docstring says "A guard that cannot fail is #154", and that selftest was itself never
executed by the suite. This file closes that: a fresh clone now runs all ten.

WHY SUBPROCESS AND NOT `from ... import selftest`. Two reasons. The `python3 -m` form
is the documented invocation in each module's docstring, and per the standing rule a
verification that does not run the thing in the environment that matters is not one.
And these guards mutate process state -- tempdirs, cwd, os.environ, `sys.exit` -- so
importing ten of them into one interpreter would let one guard's leftovers decide
another's verdict.

WHAT EACH ASSERTION IS FOR. `returncode == 0` alone would be vacuous in the one
direction that matters: a guard edited down to zero checks, or a `__main__` block that
stopped calling its selftest, exits 0 and would look green forever. So each case also
requires the guard's own summary line in stdout, and where a guard reports how many
checks it ran, a floor on that count. Raise a floor when you add checks; never lower
one without saying why.

WHAT THIS FILE DOES NOT COVER. It does not test the AO's behaviour -- it asserts that
the AO's own guards ran and behaved. If a guard's coverage is thin, this file will
still be green; that is the guard's problem, not this file's, and
`test_every_selftest_in_the_tree_is_listed_here` is what stops a NEW thin guard from
also being an unrun one.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# name, argv after the interpreter, expected stdout marker, min checks (None = the
# guard reports no count), needs_sandbox_exec.
#
# The argv here is each module's OWN documented form -- `cli` takes a positional
# `selftest`, while `prompt_author` and `skill_delivery_shim` take no flag at all
# because their `__main__` blocks call the guard unconditionally. Do not "normalise"
# these to --selftest: on the two flagless modules argparse does not exist, so a flag
# is silently ignored rather than honoured, and the next reader would believe a flag
# is being tested.
GUARDS = [
    ("spec", ["-m", "src.orchestrator.spec", "--selftest"],
     "ALL CHECKS BEHAVED", None, False),
    ("invariants", ["-m", "src.orchestrator.invariants", "--selftest"],
     "ALL CHECKS BEHAVED", None, False),
    ("loop", ["-m", "src.orchestrator.loop", "--selftest"],
     "ALL VERIFIES BEHAVED", None, False),
    ("cli", ["-m", "src.orchestrator.cli", "selftest"],
     "ALL PASS", None, False),
    ("quota_scope", ["-m", "src.orchestrator.quota_scope", "--selftest"],
     "ALL CASES BEHAVED", None, False),
    ("prompt_author", ["-m", "src.orchestrator.prompt_author"],
     "ALL FACTS CHECKS BEHAVED", None, False),
    ("evolver_dispatch", ["-m", "src.orchestrator.evolver_dispatch", "--selftest"],
     "0 bad", 9, True),
    ("evolver_direct_runner", ["-m", "src.orchestrator.evolver_direct_runner", "--selftest"],
     "0 bad", 5, False),
    ("skill_delivery_shim", ["-m", "src.orchestrator.skill_delivery_shim"],
     "SELFTEST PASSED", None, False),
    ("tau3_skill_delivery", ["scripts/tau3_skill_delivery.py", "--selftest"],
     "0 bad", 6, False),
]

_COUNT = re.compile(r"(\d+) checks, (\d+) bad")


def _run(argv: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Invoke a guard from the repo root, whatever directory pytest was started in."""
    return subprocess.run(
        [sys.executable, *argv],
        cwd=REPO, capture_output=True, text=True, timeout=600, env=env,
    )


def _report(name: str, r: subprocess.CompletedProcess) -> str:
    return (f"{name} guard: rc={r.returncode}\n"
            f"--- stdout ---\n{r.stdout}\n--- stderr ---\n{r.stderr}")


@pytest.mark.parametrize(
    "name,argv,marker,min_checks,needs_sandbox", GUARDS, ids=[g[0] for g in GUARDS])
def test_guard_runs_and_behaves(name, argv, marker, min_checks, needs_sandbox):
    if needs_sandbox and shutil.which("sandbox-exec") is None:
        # NOT a pass. This guard's first check is "sandbox-exec present (containment
        # enforceable)", and it correctly calls its own absence BAD and exits 1 --
        # refusing to certify confinement it cannot enforce. On a host without
        # sandbox-exec that exit is a true statement about the host, so asserting rc==0
        # would be wrong and asserting nothing would be a silent pass. Skip, and say so.
        pytest.skip("sandbox-exec unavailable -- containment is not enforceable on this host")

    r = _run(argv)
    assert r.returncode == 0, _report(name, r)
    assert marker in r.stdout, (
        f"{name} exited 0 but never printed its own summary line {marker!r} -- "
        f"the guard may no longer be running.\n{_report(name, r)}")

    if min_checks is not None:
        m = _COUNT.search(r.stdout)
        assert m, f"{name} reports a check count; this run printed none.\n{_report(name, r)}"
        ran, bad = int(m.group(1)), int(m.group(2))
        assert bad == 0, _report(name, r)
        assert ran >= min_checks, (
            f"{name} ran {ran} checks, floor is {min_checks}. Checks were removed, or "
            f"the environment silenced some. Confirm which before lowering the floor.\n"
            f"{_report(name, r)}")


def test_the_harness_notices_a_guard_that_fails():
    """#154: a check that reports absence must be shown able to see presence.

    The known-bad input is real, not synthetic: hide `sandbox-exec` from PATH and
    `evolver_dispatch`'s guard loses its containment probes and reports the loss. If
    the assertions above can be satisfied by a failing guard, this test fails first.
    Portable by construction -- on a host that never had sandbox-exec, the emptied PATH
    changes nothing and the guard reports the same BAD.
    """
    r = _run(["-m", "src.orchestrator.evolver_dispatch", "--selftest"], env={"PATH": ""})
    assert r.returncode != 0, (
        "a guard that lost its containment probes still exited 0 -- the rc assertion "
        f"above proves nothing.\n{_report('evolver_dispatch(no sandbox-exec)', r)}")
    assert "[BAD]" in r.stdout, _report("evolver_dispatch(no sandbox-exec)", r)
    m = _COUNT.search(r.stdout)
    assert m and int(m.group(2)) >= 1, _report("evolver_dispatch(no sandbox-exec)", r)


def test_every_selftest_in_the_tree_is_listed_here():
    """The gap this file closes was that a guard could exist and never run.

    Adding a `selftest()` and forgetting to wire it in is exactly how all ten came to
    be uncollected, so the wiring is asserted rather than remembered. If this fails,
    add the module to GUARDS -- with its marker and its check-count floor.
    """
    defined = set()
    for d in ("src/orchestrator", "scripts"):
        for p in sorted((REPO / d).glob("*.py")):
            if re.search(r"^def _?selftest\(", p.read_text(encoding="utf-8"), re.M):
                defined.add(p.stem)
    missing = defined - {g[0] for g in GUARDS}
    assert not missing, (
        f"these modules own a selftest that this file never runs: {sorted(missing)}")
