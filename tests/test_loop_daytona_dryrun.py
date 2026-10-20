"""`loop --plan` through main() on the report's SSB v2 configuration, with nothing live.

`_freeze` writes the frozen SPEC of the report's SpreadsheetBench v2 run (Claude Code solver, Opus evolver,
3 epochs x 3 attempts, the shipped examples/ssbv2/split.json); other test modules build their run dirs with it.

The plan test runs main() in a child process on that SPEC and a RUNTIME.json of the report's shape (template
prompt, stateful evolver session, pool-container evolver on Daytona, the Claude Code leg launcher, pipelined val,
skip_last_train) with every path re-pointed into tmp_path. `--plan` evaluates every verify and changes nothing:
it must exit 0, print the pipelined order, start no process that reaches a model or a sandbox (a refusing `claude`
stub is first on PATH, and its marker must not appear) and leave no prompt, skills or BLOCKED file behind.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.orchestrator.spec import ExperimentSpec, split_digest  # noqa: E402

SPLIT = ROOT / "examples" / "ssbv2" / "split.json"
NAME = "ssbv2_dryrunfx"
LEG_SNAP = "daytona-88539fb69117901dc638e4b0caca454435a04ac5b7158c3fb68fea4d4dbdeed9:daytona"
EVOLVER_SNAP = "daytona-94fe0b343fb85ffcf7ec8334352accaa515b5bf46533d2eb06c3a8bb3829f293:daytona"
LO = "LibreOffice 25.2.3.2 520(Build:2)"


def _freeze(rd: Path) -> ExperimentSpec:
    """The report's SSB v2 SPEC, frozen into `rd`."""
    sp = json.loads(SPLIT.read_text())
    spec = ExperimentSpec(
        name=NAME, benchmark="ssbv2", domain="spreadsheetbench-v2", split_path=str(SPLIT),
        split_sha256=split_digest(SPLIT), n_train=len(sp["train"]), n_val=len(sp["val"]),
        n_test=len(sp["test"]), n_samples=3, solver_model="claude-sonnet-5-5",
        solver_provider_order=(), solver_allow_fallbacks=False, user_sim_model="none",
        judge_model="none", agent_turns=45, episode_timeout_s=10800, solver_harness="claude_code",
        evolver_model="claude-opus-5-5", evolver_harness="claude_cli", n_epochs=3,
        cost_ceiling_usd=200.0)
    assert spec.validate() == [], spec.validate()
    spec.freeze(rd)
    return spec


def report_runtime(rd: Path, tmp: Path, token: Path) -> dict:
    """A RUNTIME.json of the report's run configuration, every path inside tmp_path."""
    return {
        "leg_script": str(ROOT / "scripts" / "ssbv2_leg_cc.sh"),
        "shared_dir": str(rd), "oracle": str(rd / "oracle"),
        "prompt_mode": "template", "prompt_author_model": "claude-sonnet-5-5", "evolver_host": "daytona", "evolver_sandbox": True,
        "evolver_timeout_s": 7200, "leg_timeout_s": 43200,
        "evolver_evidence": "all_epochs", "evolver_evidence_layout": "evo", "evolver_session": "stateful",
        "evolver_daytona": {"placement": "pool_container", "oauth_token_file": str(token),
                            "ttl_minutes": 180, "require_libreoffice": LO, "require_snapshot_ref": EVOLVER_SNAP,
                            "require_leg_snapshot_ref": LEG_SNAP, "machine_wait_s": 900,
                            "container": {"cpus": 2, "mem_max_gib": 6, "pids_max": 2048, "tmp_gib": 1,
                                          "rw_max_gib": 5}},
        "digest_caches": [str(tmp / "cache" / f"{NAME}_e0_cache")],
        "pipeline_val": True, "skip_last_train": True,
        "daytona_leg": {"pool": True, "require_snapshot_ref": LEG_SNAP, "require_libreoffice": LO,
                        "max_sandboxes": 6, "concurrency": 30, "max_heavy_per_shard": 2, "leg_spend_cap_usd": 40.0,
                        "run_spend_cap_usd": 200.0, "min_account_balance_usd": 100.0, "ttl_minutes": 300,
                        "shard_wall_min": 240,
                        "pool_keys": ["train", "val", "test"], "pool_sandboxes": 1, "pool_ttl_minutes": 1440},
        "claude": {"effort": "default"},
        "base_leg": str(rd / f"{NAME}_base_train.json"),
        "base_leg_val": str(rd / f"{NAME}_base_val.json"),
    }


DRIVER = r'''
import sys
sys.path.insert(0, %(repo)r)
import src.orchestrator.loop as L
sys.argv = ["loop", "--run-dir", %(rd)r, "--epoch", "1", "--plan"]
sys.exit(L.main())
'''


@pytest.fixture
def fx(tmp_path):
    rd = tmp_path / NAME
    rd.mkdir()
    _freeze(rd)
    tok = tmp_path / "cc_token"
    tok.write_text("sk-ant-oat01-FAKE")
    os.chmod(tok, 0o600)
    (rd / "RUNTIME.json").write_text(json.dumps(report_runtime(rd, tmp_path, tok), indent=1))
    (rd / "oracle").mkdir()
    stub = tmp_path / "stubbin"
    stub.mkdir()
    marker = tmp_path / "CLAUDE_WAS_CALLED"
    (stub / "claude").write_text(f"#!/bin/sh\ntouch {marker}\necho refused >&2\nexit 97\n")
    (stub / "claude").chmod(0o755)
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("EVOSKILL_", "CLAUDE", "ANTHROPIC_", "DAYTONA_", "OPENROUTER_"))}
    env.update({"PATH": f"{stub}:{os.environ.get('PATH', '/usr/bin:/bin')}", "HOME": str(tmp_path / "home"),
                "EVOSKILL_CLAUDE_EFFORT": "default"})
    drv = tmp_path / "_drv.py"
    drv.write_text(DRIVER % {"repo": str(ROOT), "rd": str(rd)})
    yield rd, env, drv
    assert not marker.exists(), "something invoked `claude`"


def test_the_report_spec_freezes_on_the_shipped_split(tmp_path):
    spec = _freeze(tmp_path)
    loaded, _ = ExperimentSpec.load(tmp_path)
    assert loaded.split_sha256 == split_digest(SPLIT) == spec.split_sha256
    assert (loaded.n_train, loaded.n_val, loaded.n_test) == (57, 57, 104)
    assert loaded.solver_harness == "claude_code" and loaded.n_epochs == 3 and loaded.n_samples == 3


def test_plan_on_the_report_configuration_changes_nothing_and_starts_nothing(fx):
    rd, env, drv = fx
    before = sorted(p.name for p in rd.iterdir())
    r = subprocess.run([sys.executable, str(drv)], cwd=str(ROOT), env=env, capture_output=True, text=True,
                       timeout=300)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    # the preflight accepted every declaration (a refused one exits 1 before the board), so the board is the
    # stateful template program of the report, in its pipelined order
    assert f"LOOP — {NAME} epoch 1" in out, out[-3000:]
    import src.orchestrator.loop as L
    for name in ("infra declaration agrees", L.STATEFUL_LAYOUT_STEP, L.STATEFUL_RENDER_STEP,
                 L.STATEFUL_CONTINUED_STEP, L.VAL_TRACKED_STEP):
        assert name in out, (name, out[-3000:])
    assert "PIPELINE (RUNTIME pipeline_val=true" in out and "skip_last_train (RUNTIME): epoch 3" in out
    # --plan wrote nothing a step would write
    after = sorted(p.name for p in rd.iterdir())
    new = set(after) - set(before)
    assert not [n for n in new if n.startswith(("ITER", "skills_v", "BLOCKED", "evo_e"))], new
