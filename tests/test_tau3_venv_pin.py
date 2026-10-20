"""The tau3-bench venv pin: scripts/tau3_check_venv.py and scripts/tau3_env/.

What must hold: a local venv third_party/tau3-bench/.venv holds exactly the pinned 76 packages on CPython 3.12.13, and
rank_bm25 (the one package `--extra knowledge` adds) imports in it; the pin names every package of the knowledge extra
and none of the other extras; uvlock_1.0.1.patch turns uv.lock at commit 363133a into the local checkout's uv.lock; the
check reports each kind of difference with rc 1 and an unreadable pin with rc 2.

OFFLINE. This file imports under the repo's interpreter, which has no tau2. Everything that runs in the tau3 venv runs
in a child process of third_party/tau3-bench/.venv/bin/python with PYTHON_DOTENV_DISABLED=1 and
LITELLM_LOCAL_MODEL_COST_MAP=True; no child imports tau2, calls a user simulator, an LLM judge or the network. Tests
that need that interpreter (or git and the checkout) are skipped when it is absent. Nothing under third_party/ is written.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "scripts" / "tau3_check_venv.py"
ENV_DIR = ROOT / "scripts" / "tau3_env"
PIN = ENV_DIR / "tau3_venv_freeze.txt"
PATCH = ENV_DIR / "uvlock_1.0.1.patch"
TAU3 = ROOT / "third_party" / "tau3-bench"
TAU3_PY = TAU3 / ".venv" / "bin" / "python"
COMMIT = "363133ada1936491fb5bcec33cd62c3518a99f65"
LOCK_AT_COMMIT_SHA = "62d3a8c4807b89e85703b3c03f2c21048a2da9736ca83d9af6f61adc74ac5314"
LOCK_PATCHED_SHA = "b7c738079f883646b50d93f72caaa256dee5cf7e130baa68cb460cfc43e18e14"
CHILD_ENV = {**os.environ, "PYTHON_DOTENV_DISABLED": "1", "LITELLM_LOCAL_MODEL_COST_MAP": "True"}

needs_venv = pytest.mark.skipif(not TAU3_PY.exists(), reason=f"no tau3-bench venv at {TAU3_PY}")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C = _load(CHECK, "tau3_check_venv_under_test")


def _check(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(CHECK), *args], capture_output=True, text=True, input=stdin,
                          env=CHILD_ENV, timeout=180)


def _pin_lines() -> list[str]:
    return [ln for ln in PIN.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]


# ───────────────────────────────────────────────────────────────────────────── the pinned files
def test_pin_is_76_packages_on_cpython_3_12_13_with_tau2_1_0_1():
    specs, unnamed, python = C.parse(PIN.read_text(encoding="utf-8"), str(PIN), strict=True)
    assert len(specs) == 76 and unnamed == []
    assert python == "CPython 3.12.13"
    assert specs["tau2"] == "==1.0.1"
    assert specs["rank-bm25"] == "==0.2.2"
    assert all(spec.startswith("==") for spec in specs.values())      # no machine path, no direct reference
    assert _pin_lines() == sorted(_pin_lines(), key=lambda ln: C.canon(ln.split("==")[0]))


def test_pin_holds_the_knowledge_extra_and_no_other_extra():
    """The venv build's `--extra knowledge`: every requirement of that extra is pinned, no requirement only another extra
    brings is (openai is also a litellm dependency, so only names no base package needs are checked for absence)."""
    if not (TAU3 / "pyproject.toml").exists():
        pytest.skip(f"no tau3-bench checkout at {TAU3}")
    extras = tomllib.loads((TAU3 / "pyproject.toml").read_text(encoding="utf-8"))["project"]["optional-dependencies"]
    specs, _, _ = C.parse(PIN.read_text(encoding="utf-8"), str(PIN), strict=True)

    def names(extra: str) -> set[str]:
        return {C.canon(re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", req).group(0))
                for req in extras[extra] if not req.startswith("tau2")}

    assert names("knowledge") == {"rank-bm25", "openai"} and names("knowledge") <= specs.keys()
    for extra, absent in (("dev", {"pytest", "pytest-xdist", "ruff", "pre-commit"}), ("gym", {"gymnasium"}),
                          ("voice", {"elevenlabs", "deepgram-sdk", "scipy", "pyaudio", "livekit-agents"}),
                          ("experiments", {"plotly", "matplotlib", "seaborn", "scikit-learn"})):
        assert absent <= names(extra), extra
        assert not (absent & specs.keys()), (extra, absent & specs.keys())


@pytest.mark.skipif(shutil.which("git") is None or not (TAU3 / ".git").exists(),
                    reason="needs git and the tau3-bench checkout")
def test_patch_turns_the_commits_uv_lock_into_the_mac_checkouts(tmp_path):
    body = PATCH.read_text(encoding="utf-8")
    changed = [ln for ln in body.splitlines() if ln[:1] in ("+", "-") and not ln.startswith(("+++", "---"))]
    assert changed == ['-version = "1.0.0"', '+version = "1.0.1"']
    lock = subprocess.run(["git", "-C", str(TAU3), "show", f"{COMMIT}:uv.lock"], capture_output=True, check=True,
                          timeout=60).stdout
    assert hashlib.sha256(lock).hexdigest() == LOCK_AT_COMMIT_SHA
    (tmp_path / "uv.lock").write_bytes(lock)
    env = {**CHILD_ENV, "GIT_CEILING_DIRECTORIES": str(tmp_path.parent)}   # never find a repository above tmp_path
    subprocess.run(["git", "apply", str(PATCH)], cwd=tmp_path, env=env, check=True, capture_output=True, timeout=60)
    assert hashlib.sha256((tmp_path / "uv.lock").read_bytes()).hexdigest() == LOCK_PATCHED_SHA
    assert hashlib.sha256((TAU3 / "uv.lock").read_bytes()).hexdigest() == LOCK_PATCHED_SHA


# ───────────────────────────────────────────────────────────────────────────── the local venv
@needs_venv
def test_mac_venv_equals_the_pin():
    cp = _check("--python", str(TAU3_PY))
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert cp.stdout.startswith("tau3_check_venv: equal: 76 packages, CPython 3.12.13 ")


@needs_venv
def test_rank_bm25_imports_in_the_mac_venv():
    code = ("import importlib.metadata as m, sys, rank_bm25\n"
            "from rank_bm25 import BM25Okapi\n"
            "scores = BM25Okapi([['card', 'fee'], ['wire', 'limit'], ['card', 'limit']]).get_scores(['fee'])\n"
            "print(m.version('rank-bm25'), m.version('tau2'), rank_bm25.__file__.startswith(sys.prefix), "
            "len(scores), scores[0] > scores[1], 'tau2' in sys.modules)\n")
    cp = subprocess.run([str(TAU3_PY), "-I", "-c", code], capture_output=True, text=True, env=CHILD_ENV,
                        cwd=str(ENV_DIR), timeout=120)
    assert cp.returncode == 0, cp.stderr
    assert cp.stdout.split() == ["0.2.2", "1.0.1", "True", "3", "True", "False"]


@needs_venv
def test_a_different_interpreter_in_the_pin_is_a_difference(tmp_path):
    pin = tmp_path / "pin.txt"
    pin.write_text(PIN.read_text(encoding="utf-8").replace("# python: CPython 3.12.13", "# python: CPython 3.12.12"),
                   encoding="utf-8")
    cp = _check("--python", str(TAU3_PY), "--pin", str(pin))
    assert cp.returncode == 1
    assert cp.stdout.splitlines()[1:] == ["  python   pin CPython 3.12.12, venv CPython 3.12.13"]


# ───────────────────────────────────────────────────────────────────────────── the check itself (no venv needed)
def test_the_pins_own_lines_are_equal_and_names_compare_in_pep503_form(tmp_path):
    lines = [ln.replace("pyyaml==", "PyYAML==").replace("pydantic-core==", "pydantic_core==") for ln in _pin_lines()]
    got = tmp_path / "freeze.txt"
    got.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cp = _check("--freeze", str(got))
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert cp.stdout.startswith("tau3_check_venv: equal: 76 packages (")
    cp = _check("--freeze", "-", stdin="\n".join(lines))
    assert cp.returncode == 0 and "(stdin vs " in cp.stdout


def test_each_kind_of_difference_is_one_line_and_rc_1(tmp_path):
    lines = [ln for ln in _pin_lines() if not ln.startswith("rank-bm25==")]
    lines = [("tau2==1.0.0" if ln == "tau2==1.0.1" else ln) for ln in lines]
    lines += ["pytest==8.3.5", "-e file:///opt/tau3/src/tau2-bench", "numpy==2.3.4"]
    got = tmp_path / "freeze.txt"
    got.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cp = _check("--freeze", str(got))
    assert cp.returncode == 1
    out = cp.stdout.splitlines()
    assert out[0].startswith("tau3_check_venv: 5 difference(s) between ")
    assert out[1:] == [
        "  extra    pytest==8.3.5  (in the venv, not in the pin)",
        "  missing  rank-bm25==0.2.2  (in the pin, not in the venv)",
        "  changed  tau2: pin ==1.0.1, venv ==1.0.0",
        "  unnamed  -e file:///opt/tau3/src/tau2-bench  (an editable from `uv pip freeze`: list with "
        "`uv pip list --format freeze`, or use --python)",
        "  unnamed  numpy==2.3.4 (a second entry for numpy)",
    ]


def test_an_unreadable_pin_or_list_is_rc_2(tmp_path):
    got = tmp_path / "freeze.txt"
    got.write_text("\n".join(_pin_lines()) + "\n", encoding="utf-8")
    for name, text in (("dup.txt", "addict==2.4.0\naddict==2.4.0\n"), ("editable.txt", "-e file:///x\n"),
                       ("empty.txt", "# python: CPython 3.12.13\n")):
        (tmp_path / name).write_text(text, encoding="utf-8")
        cp = _check("--freeze", str(got), "--pin", str(tmp_path / name))
        assert cp.returncode == 2 and cp.stderr.startswith("tau3_check_venv: "), (name, cp.stdout, cp.stderr)
    assert _check("--freeze", str(got), "--pin", str(tmp_path / "absent.txt")).returncode == 2
    assert _check("--freeze", str(tmp_path / "absent.txt")).returncode == 2
    assert _check("--python", str(tmp_path / "absent" / "python")).returncode == 2
    assert _check("--freeze", str(got), "--print").returncode == 2
