"""The ONE opt-in that lets anything under tests/ start the real `claude` CLI.

WHY THIS EXISTS. pytest's default `*_test.py` pattern collects any such file, and a module that
dispatches a REAL `claude -p` session at import time would then spend real money and leave
sessions behind on a mere `pytest --co`. Collection must never be able to do that.

Every file named in LIVE_FILES starts the real CLI (at import, or in a test) or drives a
launcher that can. It is collected, and the live scripts run, only when
EVOSKILL_ALLOW_LIVE_CLAUDE=1 is set explicitly. tests/conftest.py enforces this for pytest
(directory walks AND explicit paths). EVERY one of them also calls require_live_claude() as
its first statement, and every LIVE_ITEMS test checks allowed() as its first statement: the
conftest is switched off by `--noconftest` or a `--confcutdir` below tests/, and a module
imported another way (`-p <module>`, `--doctest-modules`, `python <file>`) never meets it.

This file is deliberately not named test_*.py or *_test.py, so pytest never collects it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

LIVE_ENV = "EVOSKILL_ALLOW_LIVE_CLAUDE"

TESTS = Path(__file__).resolve().parent

# Paths relative to tests/. Whole files: each one launches the real CLI at import or in its tests, or runs a
# launcher or suite that can. None of the offline tests does; a test that starts the CLI is listed here.
LIVE_FILES = ()

# Single tests (pytest node ids relative to tests/) in files that are otherwise offline.
LIVE_ITEMS = ()

REASON = (f"starts (or can start) the real claude CLI; set {LIVE_ENV}=1 to opt in "
          f"(see tests/live_claude_gate.py)")


def allowed(environ=None) -> bool:
    """True only for the exact opt-in value "1". Anything else, including "0", is no."""
    return (os.environ if environ is None else environ).get(LIVE_ENV) == "1"


def _resolved(p) -> Path:
    try:
        return Path(p).resolve()
    except OSError:
        return Path(os.path.abspath(p))


_LIVE_PATHS = frozenset(_resolved(TESTS / rel) for rel in LIVE_FILES)


def is_live_file(path) -> bool:
    return _resolved(path) in _LIVE_PATHS


_LIVE_TESTS = frozenset((_resolved(TESTS / f), n)
                        for f, _, n in (i.partition("::") for i in LIVE_ITEMS))


def is_live_test(path, name) -> bool:
    """`path` is the test's file (pytest's item.path), `name` its function name (item.originalname;
    a parametrisation suffix is ignored). Never the nodeid string: `--rootdir` or `-c` elsewhere
    rewrites it (to `::test_...`), and a match on it then silently matched nothing."""
    if not name:
        return False
    return (_resolved(path), str(name).split("[", 1)[0]) in _LIVE_TESTS


def require_live_claude(what: str) -> None:
    """Call at the top of a live script, before anything that could launch the CLI.

    Direct run without the opt-in: exit 3 with the reason. Under pytest (explicit path, or a
    conftest bypass): a module-level skip, so the session is not torn down by SystemExit.
    """
    if allowed():
        return
    msg = f"{what}: NOT RUN. It {REASON}."
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(msg, allow_module_level=True)
    print(msg, file=sys.stderr)
    raise SystemExit(3)
