"""No module in this repo may reference a name that does not exist.

Added 2026-08-18 after a one-line edit to scripts/run_tau3_banking.py left `_key`
bound on only ONE of two branches, so the `[KEY]` line raised NameError on exactly
the path the edit was written to enable. The pytest suite ran green through it --
that file is a runner, not an importable unit, so nothing exercised the line. A
second re-audit found it by reading.

pyflakes finds this class in about a second across the whole tree. Style findings
(unused imports, f-strings without placeholders) are deliberately NOT enforced:
this asserts the one property that is always a bug.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_no_undefined_names():
    r = subprocess.run([sys.executable, "-m", "pyflakes", "scripts", "src", "tests"],
                       cwd=REPO, capture_output=True, text=True)
    bad = [ln for ln in (r.stdout or "").splitlines()
           if "undefined name" in ln or "undefined local" in ln]
    assert not bad, "pyflakes found undefined names:\n  " + "\n  ".join(bad)
