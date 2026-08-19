"""No module in this repo may reference a name that does not exist.

pyflakes finds that class in about a second across the whole tree. Style findings
(unused imports, f-strings without placeholders) are deliberately NOT enforced: this
asserts the one property that is always a bug.

WHAT THIS TEST DOES NOT COVER, corrected from its first version. It was added after a
one-line edit to scripts/run_tau3_banking.py left `_key` bound on only ONE of two
branches, so the `[KEY]` line raised NameError on exactly the path the edit enabled --
and the commit said this test "now enforces it". It does not. pyflakes does not report
conditionally-bound names. Measured against that exact file
(`git show dba0941^:scripts/run_tau3_banking.py`, `_key` bound only in the else at line
453, read at line 692): pyflakes exits 0 with no output, so `bad` is empty and the
assertion below passes on the defect. Two minimal repros, module-level and
function-level, reproduce the same silence.

Covering it needs possibly-undefined dataflow (mypy's --enable-error-code
possibly-undefined, or an equivalent), which is not among this project's dependencies.
Two hand-written AST approximations were tried and discarded here: the first flagged
reads inside the binding branch itself, the second stopped finding the real `_key` case
while still reporting on live code. A checker whose own accuracy is unverified is the
vacuous guard this repo keeps rebuilding, so the gap is stated rather than papered over.
What DOES defend that class today is code review plus the runner smokes; nothing static.
"""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_no_undefined_names():
    r = subprocess.run([sys.executable, "-m", "pyflakes", "scripts", "src", "tests"],
                       cwd=REPO, capture_output=True, text=True)
    # THE LINTER MUST HAVE RUN. Only stdout was read before, so on an interpreter without
    # pyflakes the call returned rc=1 with "No module named pyflakes" on stderr and empty
    # stdout -- `bad` was [] and this test PASSED having checked nothing. pyflakes exits 0
    # for clean, 1 for findings; anything else is the tool failing, not the code passing.
    assert "No module named pyflakes" not in (r.stderr or ""), (
        "pyflakes is not installed in this interpreter, so this test would pass without "
        "checking anything. Install the dev extra (uv sync --extra dev).")
    assert r.returncode in (0, 1), (
        f"pyflakes itself failed (rc={r.returncode}); nothing was checked.\n{r.stderr}")

    bad = [ln for ln in (r.stdout or "").splitlines()
           if "undefined name" in ln or "undefined local" in ln]
    assert not bad, "pyflakes found undefined names:\n  " + "\n  ".join(bad)


def test_this_test_can_fail():
    """A file with a genuinely undefined name must be reported — otherwise the assertion
    above is decoration. (Kept separate from the tree scan so it cannot be masked by it.)"""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "broken.py"
        p.write_text("def f():\n    return does_not_exist_anywhere\n")
        r = subprocess.run([sys.executable, "-m", "pyflakes", str(p)],
                           capture_output=True, text=True)
        assert "undefined name" in (r.stdout or ""), (
            f"pyflakes did not flag an undefined name; rc={r.returncode}\n{r.stdout}{r.stderr}")
