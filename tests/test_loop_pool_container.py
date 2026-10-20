"""The loop side of the evolver CONTAINER on the run pool's machine (RUNTIME evolver_host "daytona" +
evolver_daytona.placement "pool_container": the solver containers and the evolver share ONE machine) and the
val-only pool flag of a skip-train epoch.

  * placement is validated in _evolver_host BEFORE any spend: pool_container needs daytona_leg.pool true, a leg
    snapshot declaration, valid container limits and machine wait;
  * the dispatch picks the driver by placement (scripts/daytona/evolver_container.py vs tau3_evolver_daytona.py),
    with the loop's wait extended by the driver's machine wait;
  * one evolver dispatch per run at a time: a second dispatch of the same run HALTs naming the holder of the
    run's EVOLVER.lock, and another run's lock never counts;
  * _v_evolver_contained routes a pool_container receipt to check_pool_container_receipt and refuses a placement
    mismatch both ways;
  * EVOSKILL_POOL_NO_TRAIN_SIBLING is "1" only for the val leg of the skip-train epoch.

No sandbox, no model, no network: subprocess.Popen is faked where a dispatch would start.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import src.orchestrator.loop as L  # noqa: E402
from src.orchestrator import daytona_evolver as de  # noqa: E402
from src.orchestrator import pool_container_evolver as pce  # noqa: E402

LEG_SNAP = "daytona-88539fb69117901dc638e4b0caca454435a04ac5b7158c3fb68fea4d4dbdeed9:daytona"
LO = "LibreOffice 25.2.3.2 520(Build:2)"


def _spec(tmp_path, **kw):
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["Template/01_01"], "val": [], "test": []}))
    d = dict(name="r", benchmark="ssbv2", split_path=str(split), n_samples=3, cache_tag="t", solver_model="m",
             solver_provider_order=(), n_epochs=3, evolver_harness="claude_cli", evolver_model="claude-opus-5-5",
             domain="ssbv2")
    d.update(kw)
    return NS(**d)


def _run(tmp_path, *, placement="pool_container", leg_pool=True, runtime=None, ed=None, pipeline_val=True, epoch=1, **kw):
    rd = tmp_path / f"run-{uuid.uuid4().hex[:6]}"          # one run dir per ctx (its own RUNTIME.json)
    rd.mkdir(parents=True)
    rt = {"daytona_leg": {"pool": leg_pool, "require_snapshot_ref": LEG_SNAP}} if runtime is None else runtime
    (rd / "RUNTIME.json").write_text(json.dumps(rt))
    tok = tmp_path / "tok"
    tok.write_text("sk-ant-oat01-FAKE")
    os.chmod(tok, 0o600)
    cfg = {"oauth_token_file": str(tok), "require_libreoffice": LO,
           "require_snapshot_ref": "daytona-OWN-EVOLVER-VM-SNAPSHOT:daytona"}
    if placement:
        cfg["placement"] = placement
    cfg.update(ed or {})
    return L.Ctx(rd, epoch, _spec(tmp_path), rd / "oracle", evolver_host="daytona", evolver_daytona=cfg,
                 leg_pool=leg_pool, pipeline_val=pipeline_val, **kw)


# ══ 1. placement validation, before any spend ═════════════════════════════════════════════════════════════════
def test_pool_container_placement_is_validated_in_evolver_host(tmp_path):
    ctx = _run(tmp_path)
    assert L._evolver_host(ctx) == "daytona" and L._evolver_placement(ctx) == "pool_container"
    for kw, word in ((dict(leg_pool=False), "needs the leg pool"),
                     (dict(placement="pool"), "is not one of"),
                     (dict(ed={"container": {"cpu": 2}}), "unknown keys"),
                     (dict(ed={"container": "big"}), "must be an object"),
                     (dict(ed={"machine_wait_s": -5}), "machine_wait_s"),
                     (dict(ed={"machine_wait_s": "900"}), "machine_wait_s")):
        c = _run(tmp_path, **kw)
        with pytest.raises(L.Halt, match=word):
            L._evolver_host(c)
    # no declared LEG snapshot = not pinned (a new user's first build): accepted; the receipt records the ref
    c = _run(tmp_path, runtime={"daytona_leg": {"pool": True}})
    assert L._evolver_host(c) == "daytona" and L._pool_container_check_kw(c)["require_snapshot_ref"] is None
    # a declared leg snapshot in evolver_daytona also counts; the OWN VM's require_snapshot_ref never does
    c = _run(tmp_path, runtime={"daytona_leg": {"pool": True}}, ed={"require_leg_snapshot_ref": LEG_SNAP})
    assert L._evolver_host(c) == "daytona" and L._pool_container_check_kw(c)["require_snapshot_ref"] == LEG_SNAP
    assert L._pool_container_check_kw(ctx) == {"require_libreoffice": LO, "require_snapshot_ref": LEG_SNAP}
    # the own-sandbox placement (absent) is unchanged: no pool needed
    own = _run(tmp_path, placement=None, leg_pool=False)
    assert L._evolver_host(own) == "daytona" and L._evolver_placement(own) == "own_sandbox"


# ══ 2. the dispatch: driver by placement ════════════════════════════════════════════════════════════════════
_REAL_POPEN = subprocess.Popen


class _FakePopen:
    """The evolver driver and its caffeinate are recorded, never started; any other command (subprocess.run's
    helpers inside the loop) runs for real."""
    calls: list = []

    def __new__(cls, argv, *a, **kw):
        if not (str(argv[0]) == "caffeinate" or any("evolver" in str(x) for x in argv[1:2])):
            return _REAL_POPEN(argv, *a, **kw)
        return super().__new__(cls)

    def __init__(self, argv, **kw):
        self.argv, self.kw, self.pid = list(argv), kw, 424242
        _FakePopen.calls.append(self)

    def wait(self, timeout=None):
        self.timeout = timeout
        return 3                                   # "refused": the loop HALTs right after the dispatch


def _dispatch(tmp_path, monkeypatch, ctx):
    ctx.paths.prompt.write_text("the prompt " * 50)
    _FakePopen.calls = []
    monkeypatch.setattr(L.subprocess, "Popen", _FakePopen)
    real_run = subprocess.run
    monkeypatch.setattr(L.subprocess, "run", lambda argv, **k: NS(stdout="", returncode=1)
                        if argv[0] == "pgrep" else real_run(argv, **k))
    with pytest.raises(L.Halt, match="evolver exited 3"):
        L._run_evolver(ctx)
    evo = [c for c in _FakePopen.calls if c.argv[0] != "caffeinate"]
    assert len(evo) == 1
    return evo[0]


def test_dispatch_picks_the_container_driver_and_waits_for_the_machine_too(tmp_path, monkeypatch):
    ctx = _run(tmp_path, ed={"machine_wait_s": 600})
    p = _dispatch(tmp_path, monkeypatch, ctx)
    assert p.argv[1:] == ["scripts/daytona/evolver_container.py", "1", str(ctx.paths.prompt), str(ctx.paths.digests),
                          str(ctx.paths.staging)]
    assert json.loads(p.kw["env"]["EVOSKILL_EVOLVER_DAYTONA_JSON"])["placement"] == "pool_container"
    assert p.kw["env"]["EVOSKILL_EVOLVER_TIMEOUT_S"] == str(ctx.evolver_timeout_s)
    assert p.timeout == ctx.evolver_timeout_s + 3600 + 600
    d = _run(tmp_path / "d")
    assert _dispatch(tmp_path, monkeypatch, d).timeout == d.evolver_timeout_s + 3600 + pce.MACHINE_WAIT_S


def test_dispatch_keeps_the_own_sandbox_driver_when_placement_is_absent(tmp_path, monkeypatch):
    ctx = _run(tmp_path, placement=None, leg_pool=False)
    p = _dispatch(tmp_path, monkeypatch, ctx)
    assert p.argv[1] == "scripts/daytona/tau3_evolver_daytona.py" and p.timeout == ctx.evolver_timeout_s + 3600
    assert "placement" not in json.loads(p.kw["env"]["EVOSKILL_EVOLVER_DAYTONA_JSON"])


# ══ 3. serialisation: one evolver dispatch per run (the run's EVOLVER.lock) ═════════════════════════════════
def test_a_second_evolver_dispatch_of_the_same_run_halts_naming_the_holder(tmp_path, monkeypatch):
    """While one dispatch holds <run>.operator/EVOLVER.lock, a second dispatch of the same run HALTs before it
    starts anything and names the holder from EVOLVER.pid.json; another run's lock never counts."""
    ctx = _run(tmp_path, pipeline_val=False)
    ctx.paths.prompt.write_text("the prompt " * 50)
    od = L.operator_dir(ctx.run_dir)
    od.mkdir(parents=True, exist_ok=True)
    (od / L.EVOLVER_PID).write_text(json.dumps({"pid": 777001, "epoch": 1, "at": "t0"}))
    monkeypatch.setattr(L.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("started")))
    with L._flock(od / L.EVOLVER_LOCK, blocking=False) as got:
        assert got
        with pytest.raises(L.Halt, match="another evolver dispatch of this run is live") as ei:
            L._run_evolver(ctx)
        assert "pid 777001" in str(ei.value) and L.EVOLVER_LOCK in str(ei.value)
        # the holder's record is left for the holder to remove
        assert json.loads((od / L.EVOLVER_PID).read_text())["pid"] == 777001
    # another run's lock is not this run's
    other = _run(tmp_path / "other", pipeline_val=False)
    with L._flock(L.operator_dir(other.run_dir) / L.EVOLVER_LOCK, blocking=False) as got:
        assert got
        _FakePopen.calls = []
        monkeypatch.setattr(L.subprocess, "Popen", _FakePopen)
        with pytest.raises(L.Halt, match="evolver exited 3"):
            L._run_evolver(ctx)
    assert not (od / L.EVOLVER_PID).exists()          # the dispatch removed its own record on the way out


# ══ 4. the receipts: routed by placement, both ways ═════════════════════════════════════════════════════════
def _containment(ctx, placement):
    d = {"sandboxed": True, "containment": "daytona", "daytona": {"sandbox_id": "s1"}}
    if placement:
        d["daytona"]["placement"] = placement
    (ctx.run_dir / f"e{ctx.epoch}_EVOLVER_CONTAINMENT.json").write_text(json.dumps(d))


def test_evolver_containment_routes_a_pool_container_receipt(tmp_path, monkeypatch):
    got = {}

    def fake_check(receipt, **kw):
        got.update(kw, receipt=receipt)
        return True, "container ok (fake)"
    monkeypatch.setattr(pce, "check_pool_container_receipt", fake_check)
    monkeypatch.setattr(de, "check_daytona_receipt", lambda *a, **k: (_ for _ in ()).throw(AssertionError("own")))
    ctx = _run(tmp_path)
    _containment(ctx, "pool_container")
    ok, det = L._v_evolver_contained(ctx)
    assert ok and "container ok (fake)" in det, det
    assert got["staging"] == ctx.paths.staging and got["require_snapshot_ref"] == LEG_SNAP
    assert got["require_libreoffice"] == LO and got["log"] == L._evolver_log_path(ctx)
    # a failing container check fails the step
    monkeypatch.setattr(pce, "check_pool_container_receipt", lambda r, **k: (False, "probe: /opt visible"))
    ok, det = L._v_evolver_contained(ctx)
    assert not ok and det == "pool-container containment: probe: /opt visible"
    # refused both ways
    _containment(ctx, None)
    ok, det = L._v_evolver_contained(ctx)
    assert not ok and "placement MISMATCH" in det and "'own_sandbox'" in det
    own = _run(tmp_path / "own", placement=None, leg_pool=False)
    _containment(own, "pool_container")
    ok, det = L._v_evolver_contained(own)
    assert not ok and "placement MISMATCH" in det and "'pool_container'" in det


def test_evolver_containment_keeps_the_own_sandbox_check(tmp_path, monkeypatch):
    got = {}
    monkeypatch.setattr(de, "check_daytona_receipt", lambda d, **k: (got.update(k) or True, "own vm ok"))
    monkeypatch.setattr(pce, "check_pool_container_receipt",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("container")))
    own = _run(tmp_path, placement=None, leg_pool=False)
    _containment(own, None)
    ok, det = L._v_evolver_contained(own)
    assert ok and "own vm ok" in det and got["require_libreoffice"] == LO


# ══ 5. the val-only pool flag ═══════════════════════════════════════════════════════════════════════════════
def test_no_train_sibling_flag_only_for_the_val_leg_of_the_skip_train_epoch(tmp_path, monkeypatch):
    rd = tmp_path / "r"
    rd.mkdir()
    script = tmp_path / "leg.sh"
    script.write_text("#!/bin/sh\n")
    monkeypatch.setattr(L, "_live_runners", lambda ctx=None: [])
    sp = NS(name="r", benchmark="ssbv2", split_path="s.json", n_samples=3, cache_tag="t", solver_model="m",
            solver_provider_order=(), n_epochs=3)
    got = {}
    for e in (1, 2, 3):
        for name, kw in (("pipeline+skip", dict(pipeline_val=True, skip_last_train=True, leg_pool=True)),
                         ("pipeline", dict(pipeline_val=True, leg_pool=True)), ("serial", dict())):
            for key in ("train", "val"):
                ctx = L.Ctx(rd, e, sp, rd / "oracle", leg_script=str(script), **kw)
                _argv, env = L._leg_argv_env(ctx, key)
                got[(e, name, key)] = env["EVOSKILL_POOL_NO_TRAIN_SIBLING"]
    assert sorted(k for k, v in got.items() if v == "1") == [(3, "pipeline+skip", "val")]
    assert set(got.values()) == {"0", "1"} and len(got) == 18


# ══ placement_from_epoch (earlier epochs on their own VM, the pool container from epoch N) ═══════════════
def test_placement_from_epoch_keeps_earlier_epochs_on_the_own_sandbox_check(tmp_path, monkeypatch):
    got = {}
    monkeypatch.setattr(de, "check_daytona_receipt", lambda d, **k: (got.update(k) or True, "own vm ok"))
    monkeypatch.setattr(pce, "check_pool_container_receipt", lambda d, **k: (True, "container ok"))
    e1 = _run(tmp_path / "a", ed={"placement_from_epoch": 2}, epoch=1)
    assert L._evolver_placement(e1) == "own_sandbox" and L._evolver_placement(e1, configured=True) == "pool_container"
    assert L._evolver_host(e1) == "daytona"                       # the pool-container config is still validated
    _containment(e1, None)                                        # e1's receipt: its own VM
    ok, det = L._v_evolver_contained(e1)
    assert ok and "own vm ok" in det, det
    _containment(e1, "pool_container")                            # a container receipt for e1 is a mismatch
    ok, det = L._v_evolver_contained(e1)
    assert not ok and "placement MISMATCH" in det
    e2 = _run(tmp_path / "b", ed={"placement_from_epoch": 2}, epoch=2)
    assert L._evolver_placement(e2) == "pool_container"
    for bad in (0, "2", True, 1.5):
        with pytest.raises(L.Halt, match="placement_from_epoch"):
            L._evolver_placement(_run(tmp_path / f"c{bad}", ed={"placement_from_epoch": bad}))
