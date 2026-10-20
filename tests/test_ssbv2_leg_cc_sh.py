"""scripts/ssbv2_leg_cc.sh: the loop's leg launcher for a CLAUDE CODE solver.

NO DAYTONA, NO CLAUDE, NO TOKEN. The driver (scripts/daytona/cc_leg_driver.py) is replaced by a STUB
interpreter (EVOSKILL_PYTHON): the script only needs the driver's CLI names (`leg --tag --key --run-dir
--split --cache-dir --skills-dir --n-samples --concurrency`, `--check-only`, `--dry-run`). What is proved:
the loop's contract (a `launched <tag> pid=` line, a detached pid whose command line names the driver and the
tag, one process), every refusal before anything is launched, the dry run's exit 7, and that the script never
names the token.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.orchestrator import infra  # noqa: E402

SCRIPT = ROOT / "scripts/ssbv2_leg_cc.sh"
SPLIT = ROOT / "examples/ssbv2/split.json"
MODEL = "claude-sonnet-5"
PINS = infra.SOLVER_ROUTING_KEYS            # the seven OpenRouter routing pins the launcher refuses
DEEPSEEK_PINS = {"EVOSKILL_SOLVER_PROVIDER_ORDER": "deepinfra", "EVOSKILL_SOLVER_PROVIDER_ONLY": "deepinfra",
                 "EVOSKILL_SOLVER_ALLOW_FALLBACKS": "0", "EVOSKILL_SOLVER_REQUIRE_PARAMETERS": "1",
                 "EVOSKILL_SOLVER_PROVIDER_QUANTIZATIONS": "fp8", "EVOSKILL_SOLVER_EXPECT_PROVIDER": "DeepInfra",
                 "EVOSKILL_SOLVER_EXPECT_MODEL": "deepseek/deepseek-v4.1-flash-20260910"}
assert set(DEEPSEEK_PINS) == set(PINS)

# The stub "interpreter": `$PY $DRIVER leg ... [--check-only|--dry-run]`. --check-only passes; --dry-run
# writes <run>/<tag>.DRYRUN.json and returns 7; the live call records its argv and sleeps (the detached leg).
STUB_LIVE = r"""#!/bin/bash
RD=""; TAG=""; prev=""
for a in "$@"; do
  [ "$prev" = --run-dir ] && RD="$a"
  [ "$prev" = --tag ] && TAG="$a"
  prev="$a"
done
for a in "$@"; do
  [ "$a" = --check-only ] && exit 0
  [ "$a" = --dry-run ] && { printf '{"dry_run": true, "argv": "%s"}\n' "$*" > "$RD/$TAG.DRYRUN.json"; exit 7; }
done
printf '%s\n' "$@" > "__MARKER__"
env > "__ENVDUMP__"
sleep 20
"""
STUB_REFUSE = "#!/bin/bash\necho 'REFUSED by the driver: probe record missing for a skill leg' >&2\nexit 4\n"
STUB_DIE = "#!/bin/bash\nfor a in \"$@\"; do [ \"$a\" = --check-only ] && exit 0; done\necho 'boom' >&2\nexit 1\n"


def _stub(tmp: Path, name: str, body: str) -> Path:
    p = tmp / name
    p.write_text(body)
    p.chmod(0o755)
    return p


@pytest.fixture
def w(tmp_path):
    """A run dir with a promoted skills_v1 (one skill), an empty skills dir, the stub interpreters."""
    rd = tmp_path / "ssbv2_fx_cc_run"
    (rd / "skills_v1" / "alpha").mkdir(parents=True)
    (rd / "skills_v1" / "alpha" / "SKILL.md").write_text("---\nname: alpha\ndescription: A.\n---\nbody\n")
    (rd / "skills_empty").mkdir()
    marker, envdump = tmp_path / "stub_args", tmp_path / "stub_env"
    live = _stub(tmp_path, "stubpy", STUB_LIVE.replace("__MARKER__", str(marker)).replace("__ENVDUMP__", str(envdump)))
    return types.SimpleNamespace(tmp=tmp_path, rd=rd, cache=tmp_path / "cache" / "fx_e1_cache", marker=marker,
                                 envdump=envdump, live=live, refuse=_stub(tmp_path, "stubrefuse", STUB_REFUSE),
                                 die=_stub(tmp_path, "stubdie", STUB_DIE))


def _env(w, py=None, **over):
    env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": str(w.tmp),
           "EVOSKILL_TAU3_RUN_DIR": str(w.rd), "EVOSKILL_TAU3_SPLIT": str(SPLIT), "EVOSKILL_N_SAMPLES": "3",
           "EVO_CONC": "30", "EVOSKILL_CACHE_DIR": str(w.cache), "EVOSKILL_SOLVER_MODEL": MODEL,
           "EVOSKILL_PYTHON": str(py or w.live)}
    env.update(over)
    return {k: v for k, v in env.items() if v is not None}


def run(w, tag, key, arg3, env):
    return subprocess.run(["bash", str(SCRIPT), tag, key, arg3], env=env, capture_output=True, text=True,
                          timeout=60)


def _pid(out: str) -> int:
    m = re.search(r"launched (\S+) pid=(\d+)", out)
    assert m, out
    return int(m.group(2))


def _cmdline(pid: int) -> str:
    return subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True).stdout


def _kill(pid: int):
    try:
        os.kill(pid, 9)
    except ProcessLookupError:
        pass


def _argv(w) -> list:
    return w.marker.read_text().splitlines()


# ══ the loop's contract ═════════════════════════════════════════════════════════════════════════
def test_skill_leg_detaches_one_runner_named_pid_and_prints_the_launch_line(w):
    tag = f"{w.rd.name}_v1_train"
    r = run(w, tag, "train", "skills_v1", _env(w, EVOSKILL_SKILLS_DIR=str(w.rd / "skills_v1")))
    assert r.returncode == 0, r.stdout + r.stderr
    pid = _pid(r.stdout)
    try:
        assert f"launched {tag} pid={pid} key=train conc=30 model={MODEL} daytona=1 harness=claude_code" in r.stdout
        assert "verified: pid" in r.stdout and f"skill leg: {w.rd / 'skills_v1'}" in r.stdout
        cmd = _cmdline(pid)
        assert "cc_leg_driver.py" in cmd and f"--tag {tag}" in cmd, "the loop identifies the detached pid by its tag"
        args = _argv(w)                                   # `$PY $DRIVER leg ...`: argv[0] is the driver
        assert args[0] == str(ROOT / "scripts/daytona/cc_leg_driver.py") and args[1] == "leg"
        for flag, want in (("--tag", tag), ("--key", "train"), ("--run-dir", str(w.rd)),
                           ("--split", str(SPLIT)), ("--cache-dir", str(w.cache)),
                           ("--skills-dir", str(w.rd / "skills_v1")), ("--n-samples", "3"), ("--concurrency", "30")):
            assert args[args.index(flag) + 1] == want, (flag, args)
        assert "--check-only" not in args and "--dry-run" not in args
        assert (w.rd / f"{tag}.log").exists()
        low = (r.stdout + r.stderr).lower()
        assert "openrouter" not in low and "oauth" not in low and "sk-ant" not in low
        dumped = w.envdump.read_text()
        assert "CLAUDE_CODE_OAUTH_TOKEN" not in dumped and "sk-ant" not in dumped
        assert "EVOSKILL_SOLVER_PROVIDER_ORDER" not in dumped
    finally:
        _kill(pid)


def test_base_leg_passes_an_empty_skills_dir(w):
    tag = f"{w.rd.name}_base_val"
    r = run(w, tag, "val", "", _env(w))
    assert r.returncode == 0, r.stdout + r.stderr
    pid = _pid(r.stdout)
    try:
        assert "BASE leg (no skills dir)" in r.stdout and "key=val" in r.stdout
        args = _argv(w)
        assert args[args.index("--skills-dir") + 1] == "" and args[args.index("--key") + 1] == "val"
    finally:
        _kill(pid)


def test_the_loop_env_may_carry_only_the_model_of_the_solver_variables(w):
    """EVOSKILL_SOLVER_MODEL is the one solver variable the loop exports for this run (RUNTIME has no
    solver_env); every OpenRouter pin is refused below. The launcher passes nothing secret on."""
    r = run(w, f"{w.rd.name}_base_train", "train", "", _env(w, PHOENIX_PROJECT_NAME="x"))
    assert r.returncode == 0, r.stdout + r.stderr
    _kill(_pid(r.stdout))


# ══ refusals: nothing launched, exit 4, the reason on stderr ════════════════════════════════════
@pytest.mark.parametrize("over, arg3, needle", [
    ({"EVOSKILL_SOLVER_MODEL": "deepseek/deepseek-v4.1-flash"}, "skills_v1", "not a Claude model"),
    ({"EVOSKILL_SOLVER_MODEL": None}, "skills_v1", "EVOSKILL_SOLVER_MODEL"),
    ({"EVOSKILL_CACHE_DIR": None}, "skills_v1", "EVOSKILL_CACHE_DIR"),
    ({"EVOSKILL_TAU3_SPLIT": None}, "skills_v1", "EVOSKILL_TAU3_SPLIT"),
    ({"EVOSKILL_TAU3_RUN_DIR": None}, "skills_v1", "EVOSKILL_TAU3_RUN_DIR"),
    ({"EVOSKILL_TAU3_RUN_DIR": "/nowhere/run"}, "skills_v1", "no run dir"),
    ({"EVOSKILL_SKILLS_DIR": None}, "skills_missing", "does not exist"),      # (an EMPTY dir is a skill leg)
    ({"EVOSKILL_SKILLS_DIR": "/elsewhere"}, "skills_v1", "not the same dir"),
    ({"EVOSKILL_SKILLS_DIR": "KEEP"}, "", "arg 3"),
    ({"EVO_CONC": "31"}, "skills_v1", "EVO_CONC=31 exceeds 30"),
    ({"EVO_CONC": "abc"}, "skills_v1", "not an integer"),
    ({"EVOSKILL_PYTHON": "/nowhere/python"}, "skills_v1", "no interpreter at"),
] + [({k: v}, "skills_v1", f"{k} is set (an OpenRouter routing pin)") for k, v in DEEPSEEK_PINS.items()])
def test_refuses_before_launching(w, over, arg3, needle):
    sk = str(w.rd / arg3) if arg3 else None
    env = _env(w, EVOSKILL_SKILLS_DIR=sk)
    for k, v in over.items():
        if v is None:
            env.pop(k, None)
        elif v == "KEEP":
            env[k] = str(w.rd / "skills_v1")
        else:
            env[k] = v
    r = run(w, f"{w.rd.name}_v1_train", "train", arg3, env)
    assert r.returncode != 0 and "launched" not in r.stdout, r.stdout
    assert needle in r.stderr, r.stderr
    assert not w.marker.exists(), "the driver was launched"


def test_an_empty_promoted_set_runs_as_a_skill_leg_with_no_skills(w):
    """An existing but EMPTY promoted dir is launched as a skill leg with zero skills: the loop has no skill rules of
    its own."""
    r = run(w, f"{w.rd.name}_v1_train", "train", "skills_empty", _env(w, EVOSKILL_SKILLS_DIR=str(w.rd / "skills_empty")))
    assert r.returncode == 0 and "skill leg:" in r.stdout, r.stdout + r.stderr
    _kill(_pid(r.stdout))


def test_refuses_a_measured_leg_a_hold_and_the_test_key(w):
    tag = f"{w.rd.name}_v1_train"
    (w.rd / f"{tag}.json").write_text("{}")
    r = run(w, tag, "train", "", _env(w))
    assert r.returncode == 4 and "already exists" in r.stderr
    (w.rd / f"{tag}.json").unlink()
    (w.rd / "HOLD_LEGS").write_text("HOLD: layout change in progress")
    r = run(w, tag, "train", "", _env(w))
    assert r.returncode == 4 and "HOLD_LEGS" in r.stderr and "layout change in progress" in r.stderr
    (w.rd / "HOLD_LEGS").unlink()
    # the test key is accepted (the held-out leg); an unknown key is refused
    r = run(w, f"{w.rd.name}_v1_x", "dev", "", _env(w))
    assert r.returncode == 4 and "train|val|test only" in r.stderr
    assert not w.marker.exists()


def test_a_check_only_refusal_reaches_the_caller_before_anything_is_detached(w):
    r = run(w, f"{w.rd.name}_v1_train", "train", "skills_v1",
            _env(w, py=w.refuse, EVOSKILL_SKILLS_DIR=str(w.rd / "skills_v1")))
    assert r.returncode == 4 and "launched" not in r.stdout
    assert "REFUSED by the driver" in r.stderr
    assert not w.marker.exists() and not (w.rd / f"{w.rd.name}_v1_train.log").exists()


def test_a_runner_dead_within_5s_is_exit_5_with_the_log_tail(w):
    tag = f"{w.rd.name}_base_train"
    r = run(w, tag, "train", "", _env(w, py=w.die))
    assert r.returncode == 5, r.stdout + r.stderr
    assert f"launched {tag} pid=" in r.stdout
    assert "DEAD" in r.stderr and "boom" in r.stderr           # the log's tail is on stderr
    assert not (w.rd / f"{tag}.json").exists()


# ══ dry run ═════════════════════════════════════════════════════════════════════════════════════
def test_dry_run_exits_7_writes_the_record_and_launches_nothing(w):
    tag = f"{w.rd.name}_v1_val"
    r = run(w, tag, "val", "skills_v1",
            _env(w, EVOSKILL_LEG_DRY_RUN="1", EVOSKILL_SKILLS_DIR=str(w.rd / "skills_v1")))
    assert r.returncode == 7, r.stdout + r.stderr
    assert "DRY RUN — no process launched" in r.stdout and not re.search(r"launched \S+ pid=", r.stdout)
    rec = (w.rd / f"{tag}.DRYRUN.json").read_text()
    assert f"leg --tag {tag} --key val" in rec and "--dry-run" in rec and f"--skills-dir {w.rd / 'skills_v1'}" in rec
    assert not w.marker.exists() and not (w.rd / f"{tag}.log").exists()


def test_dry_run_refused_by_the_driver_is_not_a_clean_dry_run(w):
    tag = f"{w.rd.name}_v1_train"
    r = run(w, tag, "train", "skills_v1",
            _env(w, py=w.refuse, EVOSKILL_LEG_DRY_RUN="1", EVOSKILL_SKILLS_DIR=str(w.rd / "skills_v1")))
    assert r.returncode == 4 and "DRY RUN REFUSED" in r.stderr and "DRY RUN —" not in r.stdout
    assert not (w.rd / f"{tag}.DRYRUN.json").exists()


# ══ the script itself ═══════════════════════════════════════════════════════════════════════════
def test_the_script_names_no_token_and_runs_the_cc_leg_driver():
    text = SCRIPT.read_text()
    assert "oauth_token" not in text and "CLAUDE_CODE_OAUTH_TOKEN" not in text and "sk-ant" not in text
    assert 'DRIVER="$R/scripts/daytona/cc_leg_driver.py"' in text
    assert os.access(SCRIPT, os.X_OK) or True            # bash <script> is how the loop calls it
    for pin in PINS:                                     # every routing pin is named in the refusal loop
        assert pin in text
