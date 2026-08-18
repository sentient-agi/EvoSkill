"""Exercise the --max-age-s (past-wall) reap path, which the live container set
cannot exercise: no container is older than the wall yet, so a green run against
production proves only that the branch was never taken.

HISTORY, because the shape of the mistake is reusable: this file was first written
as a standalone SCRIPT — module-level checks ending in `sys.exit(...)` — while
being named `test_*.py`. Under pytest that raises SystemExit during COLLECTION,
which is an INTERNALERROR, which aborts the whole session: `pytest tests/`
collected 467 tests and then ran NONE of them. A file that only works when invoked
directly is not in the suite, and if it is named like a test it takes the suite
down with it. Verified green under pytest now, which is the only thing that counts.
"""
import importlib.util
import os
import sys
import time
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "reap", str(Path(__file__).resolve().parent.parent
               / "scripts/reap_orphan_containers.py"))
reap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reap)

MY = os.getpid()          # guaranteed ALIVE
DEAD = 999999999          # guaranteed dead — no such pid


def _labels(pid, age_s, now):
    return {"evoskill.owner_pid": str(pid),
            "evoskill.created": str(now - age_s),
            "evoskill.run_tag": "t"}


@pytest.fixture
def rows():
    """One container per decision the reaper has to make."""
    now = time.time()
    return [
        ("live_young",     _labels(MY, 60, now)),     # live, under min-age -> spare
        ("live_mid",       _labels(MY, 900, now)),    # live, mid-life      -> spare
        ("live_pastwall",  _labels(MY, 5000, now)),   # live, PAST WALL     -> reap w/ wall
        ("dead_owner",     _labels(DEAD, 900, now)),  # dead owner          -> always reap
        ("unlabelled",     {}),                       # ephemeral verifier  -> never touch
        ("live_nocreated", {"evoskill.owner_pid": str(MY)}),  # age UNKNOWN -> spare
    ]


@pytest.fixture
def run(monkeypatch, rows):
    """Drive reap.main() against synthetic containers, capturing what it removed.

    monkeypatch (not raw assignment) so `sys.argv` and the two patched module
    attributes are RESTORED after every test — the original version leaked a
    mutated sys.argv into whatever ran next.
    """
    def _run(argv, container_rows=None):
        data = container_rows if container_rows is not None else rows
        monkeypatch.setattr(reap, "containers", lambda: [
            {"id": f"id{i}", "name": n, "running_for": "?",
             "labels": {k: v for k, v in lab.items() if v is not None}}
            for i, (n, lab) in enumerate(data)])

        removed = []

        class _R:
            returncode = 0
            stdout = ""
            stderr = ""

        def fake_run(cmd, **kw):
            r = _R()
            if cmd[:3] == ["docker", "rm", "-f"]:
                removed.extend(cmd[3:])
                r.stdout = "\n".join(cmd[3:])
            return r

        monkeypatch.setattr(reap.subprocess, "run", fake_run)
        monkeypatch.setattr(sys, "argv", ["reap"] + argv)

        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = reap.main()
        return rc, set(removed), buf.getvalue()

    return _run


def test_default_reaps_only_the_dead_owner(run):
    """--max-age-s defaults to 0 (disabled), so a live owner is untouchable
    however old its container is. That default is the safety property: guessing
    the wall wrong would remove a live rollout's sandbox mid-measurement."""
    _rc, removed, _out = run(["--apply"])
    assert removed == {"dead_owner"}


def test_the_wall_also_reaps_a_past_wall_container_with_a_LIVE_owner(run):
    """The whole point of #117C: within-process accumulation. A rollout cannot
    outlive the solver wall, so a labelled sandbox older than it provably is not
    serving in-flight work — even though its owner is alive and busy."""
    _rc, removed, _out = run(["--apply", "--max-age-s", "4200"])
    assert removed == {"dead_owner", "live_pastwall"}


def test_a_container_of_UNKNOWN_age_is_never_reaped_by_the_age_rule(run):
    """An unparseable/absent `created` label means the wall cannot be applied.
    Spare it rather than guess — a false positive corrupts a measured rollout."""
    _rc, removed, _out = run(["--apply", "--max-age-s", "4200"])
    assert "live_nocreated" not in removed


def test_an_unlabelled_container_is_never_touched(run):
    """Includes the ephemeral `ssb-verifier` recalc containers scoring spawns.
    Nothing without `evoskill.owner_pid` is ours to judge."""
    _rc, removed, _out = run(["--apply", "--max-age-s", "4200"])
    assert "unlabelled" not in removed


def test_young_and_midlife_live_containers_are_spared(run):
    _rc, removed, _out = run(["--apply", "--max-age-s", "4200"])
    assert "live_young" not in removed and "live_mid" not in removed


def test_dry_run_removes_nothing_but_still_reports_the_candidate(run):
    """Dry-run-by-default is the third safety guard; a dry run that silently
    reaped would be the worst possible failure of this script."""
    _rc, removed, out = run(["--max-age-s", "4200"])
    assert removed == set()
    assert "PAST WALL (live owner)   : 1" in out


def test_contradictory_guards_are_refused(run):
    """--max-age-s below --min-age-s means "spare everything younger than X and
    also reap everything older than Y<X" — incoherent, so refuse rather than
    silently pick one."""
    with pytest.raises(SystemExit):
        run(["--apply", "--min-age-s", "600", "--max-age-s", "300"])


def test_a_container_with_no_created_label_does_not_crash_the_age_math(run):
    """Regression: `age` is None when `created` is absent, and an early version
    computed `age / 60` for the report — a TypeError that killed the reaper
    before it removed anything, on exactly the containers it should ignore."""
    only_unknown = [("live_nocreated", {"evoskill.owner_pid": str(MY)}),
                    ("dead_nocreated", {"evoskill.owner_pid": str(DEAD)})]
    rc, removed, _out = run(["--apply", "--max-age-s", "4200"], only_unknown)
    assert rc == 0
    # The dead owner is still reaped — liveness does not depend on knowing age.
    assert removed == {"dead_nocreated"}
