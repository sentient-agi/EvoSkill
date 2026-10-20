"""check_pool_container_receipt's STATEFUL branch (container layout v2) on hand-built receipts:

  - a KEPT epoch (no `down`, no ledger row) PASSES with `session="stateful"` and every stateful rule is named on
    its mutation (session record, session_id, the rw pair /evo/staging + /evo/evolver_workspace, creation-time
    probe sha, epoch-end flags and scans, egress from the epoch-end log, lease released, a `down` inside a kept
    epoch, the recorded path_view);
  - the TEARDOWN variants: a verified wipe needs the ledgered container_down; a machine_absent record needs
    machine.json to show the sandbox gone; a wrong cid / sandbox is refused;
  - the DEFAULT arguments stay today's per-epoch check: the same stateful receipt fails it (no down, no ledger).

The receipts the real driver writes (e1 kept, e2 kept, the finish record) are tests/test_evolver_session.py's
(FakeMachine). No network, no model, tmp only.

"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.orchestrator import daytona_evolver as de  # noqa: E402
from src.orchestrator import evidence_surface as es  # noqa: E402
from src.orchestrator import evo_layout as EL  # noqa: E402
from src.orchestrator import pool_container_evolver as pce  # noqa: E402
from src.orchestrator import stateful_layout as SL  # noqa: E402
from tests.test_evolver_container import FAKE_FACTS, _set, good_probe, plan_of  # noqa: E402

ctl = pce.load_ctl()
CID = "run-20260927t120000-abc123"
SID = "11111111-2222-4333-8444-555555555555"
SB = "sbx0001stateful"
NAME = pce.POOL_PREFIX + "v1pl-m1-120000"
RW_PATHS = [r["path"] for r in SL.STATEFUL_RW]
EVO_ROLES = (SL.ROLE_TRACES, SL.ROLE_SOLVER_WS, SL.ROLE_SCORES, SL.ROLE_REFERENCE)


def _spec(rw=None) -> dict:
    """The stateful container spec (layout v2): rw = the two evo dirs, no re-bind, the container-native
    read-only roots of epoch 2."""
    return {"schema": ctl.SCHEMA, "cid": CID, "rw": rw if rw is not None else [dict(d) for d in SL.STATEFUL_RW],
            "ro_rebind": [], "readonly_roots": [SL.traces_dir(0), SL.traces_dir(1), SL.solver_ws_dir(0),
                                                SL.solver_ws_dir(1), SL.VM_SCORES, SL.VM_REFERENCE],
            "limits": {}, "cc": {"version": ctl.CC_VERSION, "sha256": ctl.CC_SHA256},
            "egress": {"allow": ctl.ALLOW, "must_reach": list(de.MUST_REACH), "must_refuse": list(de.MUST_REFUSE),
                       "max_seconds": 86400, "idle_timeout_s": 1800}}


def _evo(rd: Path, K: int) -> Path:
    """<run>/evo_e{K}: one raw log, one solver workspace, scores.json (evo_layout)."""
    d = EL.epoch_dir(rd, K)
    (d / EL.TRACES).mkdir(parents=True)
    (d / EL.TRACES / "Template_01_01_s1.jsonl").write_text('{"type": "result"}\n')
    (d / EL.WORKSPACES / "Template_01_01_s1").mkdir(parents=True)
    (d / EL.WORKSPACES / "Template_01_01_s1" / "out.xlsx").write_bytes(f"OUT e{K}".encode())
    (d / EL.SCORES_JSON).write_text('[{"task_id": "Template_01_01", "sample": 1, "score": 1.0}]\n')
    return d


def _rows(pm: SL.PathMap) -> list:
    if pm.role in SL.FILE_ROLES:
        return [dict(de._row(pm.vm, pm.role, pm.mac), epoch=pm.epoch)]
    return [dict(de._row(f"{pm.vm}/{f.relative_to(pm.mac).as_posix()}", pm.role, str(f)), epoch=pm.epoch)
            for f in sorted(Path(pm.mac).rglob("*")) if f.is_file()]


@pytest.fixture
def kept(tmp_path, monkeypatch):
    """An epoch-2 receipt of a KEPT stateful container, as Appendix A describes it on layout v2 (evo_e0 / evo_e1 and
    the adapter's reference/ mapped by the run's path view), plus its log, an (absent) ledger and a finish record."""
    rd = tmp_path / "run"
    e0, e1 = _evo(rd, 0), _evo(rd, 1)
    (rd / SL.REFERENCE_DIR_NAME).mkdir()
    (rd / SL.REFERENCE_DIR_NAME / "README.md").write_text("the adapter's copy\n")
    spec = _spec()
    pl, lim = plan_of(spec)
    probe = good_probe(pl, spec)
    view = SL.path_view(rd, 2)
    up1 = [r for pm in view if pm.epoch == 1 and pm.role in EVO_ROLES for r in _rows(pm)]
    up2 = [r for pm in view if pm.epoch == 2 and pm.role in EVO_ROLES for r in _rows(pm)]
    before = es.hashes([e0, e1])
    log = tmp_path / "tau3_evolver_e2.log"
    lines = [{"type": "system", "subtype": "init", "apiKeySource": "none", "mcp_servers": [], "tools": ["Bash"],
              "session_id": SID, "model": "claude-opus-5-5", "claude_code_version": "2.1.281"},
             {"type": "result", "subtype": "success", "is_error": False}]
    log.write_bytes("".join(json.dumps(x) + "\n" for x in lines).encode())
    rb = {"refreshed": True, "state": "SandboxState.STARTED", "domain_allow_list": pce.MACHINE_ALLOW_STR,
          "network_allow_list": None, "network_block_all": False, "snapshot_ref": "daytona-leg:snap"}
    ee = {"ok": True, "cid": CID, "kept": True, "frozen": True, "thawed": True, "init_alive": True,
          "tmpfs_scan": {"files": 4, "hits": []}, "scan": {k: {"files": 9, "hits": []} for k in ("fs", "rw", "state")},
          "egress_log": {"requests": 5, "by_host": {"api.anthropic.com:allowed": 4, "openrouter.ai:refused": 1}},
          "tgz": "/opt/evoskill-evolver/c/x/state/out.tgz", "bytes": 10, "packed": [SL.VM_STAGING], "refused": []}
    rec = {"containment": "daytona", "sandboxed": True, "upload_manifest": up2, "upload_manifest_cumulative": up1 + up2,
           "path_view": [pm.as_dict() for pm in view], "evidence_sha256_before": before,
           "evidence_sha256_after_vm": dict(before), "token_scan": {"hits": [], "scanned_files": 7},
           "daytona": {"placement": pce.PLACEMENT, "sandbox_id": SB, "sandbox_name": NAME, "session_id": SID,
                       "pool_machine": {"format": "pool", "row": {"sandbox_id": SB, "state": "ready"}},
                       "lease": {"acknowledged": True, "released": True, "sandbox_id": SB},
                       "readback_create": dict(rb), "readback_end": dict(rb),
                       "launch_argv": ["claude", "-p", "__PROMPT__", "--resume", SID, "--append-system-prompt", "__SYSPROMPT__"],
                       "prompt_delivery": "message + appended system prompt as base64 shell variables (no file)",
                       "log_round_trip": {"equal": True, "streamed_sha256": de.sha256_bytes(log.read_bytes())},
                       "session_audit": {"files": 2, "denied_tool_uses": [], "nested_claude": [], "api_or_token_refs": []},
                       "container": {"cid": CID, "spec": spec, "facts": FAKE_FACTS, "limits": lim, "plan": pl,
                                     "init_sha256": hashlib.sha256(ctl.render_init(spec, FAKE_FACTS, lim, pl).encode()).hexdigest(),
                                     "leg_dockerfile_sha256": de.BASE_DOCKERFILE_SHA, "probe_pre": probe,
                                     "probe_pre_sha256": SL.probe_sha256(probe),
                                     "probe_post": {"phase": "post", "view": {"top": probe["view"]["top"],
                                                                              "mounts": probe["view"]["mounts"],
                                                                              "ifaces": ["lo"]}},
                                     "session": {"mode": "stateful", "cid": CID, "sid": SID, "created_epoch": 1,
                                                 "epoch": 2, "resume": True, "kept": True, "rebuilt": None},
                                     "epoch_end": ee}}}
    teardown = {"schema": "evoskill.evolver_session_teardown/1", "cid": CID, "sandbox_id": SB, "sid": SID,
                "epochs": [1, 2], "down": {"cid": CID, "verified": True, "wiped": True}, "ledgered": True,
                "utc": "2026-09-27T12:00:00Z", "abort": False}
    return SimpleNamespace(rec=rec, log=log, ledger=tmp_path / "ledger.jsonl", teardown=teardown, probe=probe, spec=spec)


def _check(k, rec=None, **kw) -> tuple:
    base = dict(log=k.log, ledger_path=k.ledger, session="stateful", rw_paths=list(RW_PATHS),
                probe_pre_sha256=SL.probe_sha256(k.probe))
    base.update(kw)
    return pce.check_pool_container_receipt(rec if rec is not None else k.rec, **base)


def _ledger_down(k) -> None:
    k.ledger.write_text(json.dumps({"event": "container_down", "driver": pce.DRIVER, "tag": "run", "id": SB,
                                    "cid": CID, "verified": True}) + "\n")


# ══ 1. the kept epoch ═══════════════════════════════════════════════════════════════════════════════════════
def test_kept_epoch_passes_with_no_down_and_no_ledger_row(kept):
    ok, det = _check(kept)
    assert ok, det
    assert "kept for the next epoch" in det and f"stateful session {SID[:8]} epoch 2" in det
    assert "torn down" not in det and not kept.ledger.exists()
    # rw_paths None defaults to the stateful pair; the state's probe sha may be omitted (the receipt's own still holds)
    ok, det = _check(kept, rw_paths=None, probe_pre_sha256=None)
    assert ok, det
    # the DEFAULT arguments are today's per-epoch check: this receipt fails it exactly on the teardown rules
    ok, det = pce.check_pool_container_receipt(kept.rec, log=kept.log, ledger_path=kept.ledger)
    assert not ok and "teardown not verified" in det and "container_down" in det
    with pytest.raises(ValueError, match="session="):
        pce.check_pool_container_receipt(kept.rec, log=kept.log, session="statefull")


def test_an_unpinned_leg_snapshot_is_accepted_and_recorded_a_pinned_one_compared(kept):
    """require_snapshot_ref None (the shipped templates: a new user's first build has no known ref) = accept the ref
    Daytona reported, named in the detail; the receipt must record one. A pin is compared exactly."""
    ok, det = _check(kept)
    assert ok and "snapshot daytona-leg:snap, not pinned" in det, det
    ok, det = _check(kept, require_snapshot_ref="daytona-leg:snap")
    assert ok and "snapshot daytona-leg:snap;" in det, det
    ok, det = _check(kept, require_snapshot_ref="daytona-other:snap")
    assert not ok and "!= the leg image's daytona-other:snap" in det
    rec = copy.deepcopy(kept.rec)
    rec["daytona"]["readback_create"]["snapshot_ref"] = None
    ok, det = _check(kept, rec)
    assert not ok and "records no snapshot_ref" in det


def test_the_per_epoch_receipt_shape_is_refused_as_a_stateful_one(kept):
    """A per-epoch receipt (down + wipe, no session record, no epoch_end) is not a stateful epoch."""
    rec = copy.deepcopy(kept.rec)
    c = rec["daytona"]["container"]
    c.pop("session")
    c.pop("epoch_end")
    c["down"] = {"verified": True, "wiped": True, "scan": {"fs": {"files": 1, "hits": []}},
                 "egress_log": {"by_host": {"api.anthropic.com:allowed": 1}}}
    ok, det = _check(kept, rec=rec)
    assert not ok
    for needle in ("session:", "epoch-end", "kept epoch"):
        assert needle in det, (needle, det)


MUTATIONS = [
    ("daytona.container.session.mode", "per_epoch", "session:"),
    ("daytona.container.session.cid", "run-other", "session:"),
    ("daytona.container.session.sid", "other-sid", "session_id"),
    ("daytona.session_id", "other-sid", "session_id"),
    ("daytona.container.spec.rw", lambda rw: rw[:1], "writable dirs"),
    ("daytona.container.spec.rw", lambda rw: rw + [{"path": "/evo/notes", "owner": "evo", "mode": "0755"}],
     "writable dirs"),                                                          # the older notes dir is no rw dir
    ("daytona.container.probe_pre_sha256", "0" * 64, "does not hash"),
    ("daytona.container.probe_pre.identity.uid", 0, "creation-time probe"),
    ("daytona.container.epoch_end.thawed", False, "thawed"),
    ("daytona.container.epoch_end.kept", False, "kept"),
    ("daytona.container.epoch_end.frozen", False, "frozen"),
    ("daytona.container.epoch_end.init_alive", False, "init_alive"),
    ("daytona.container.epoch_end.tmpfs_scan", {"files": 4, "hits": ["/tmp/x"]}, "state scan"),
    ("daytona.container.epoch_end.tmpfs_scan", {}, "state scan"),
    ("daytona.container.epoch_end.scan", {"rw": {"files": 1, "hits": [f"{SL.VM_WORKSPACE}/t"]}}, "state scan"),
    ("daytona.container.epoch_end.scan", {}, "no scan"),
    ("daytona.container.epoch_end", {}, "epoch-end"),
    ("daytona.container.epoch_end.egress_log", {"by_host": {"api.anthropic.com:allowed": 1, "example.com:allowed": 1}},
     "tunnels"),
    ("daytona.lease.released", False, "not released"),
    ("daytona.lease.acknowledged", False, "never listed"),
    ("daytona.container.down", {"verified": True, "wiped": True}, "kept epoch"),
    ("evidence_sha256_after_vm", {}, "evidence hashes"),
    ("evidence_sha256_before", {}, "evidence hashes"),
    ("token_scan", {"hits": [f"{SL.VM_WORKSPACE}/x"], "scanned_files": 1}, "token scan"),
]


@pytest.mark.parametrize("path,value,needle", MUTATIONS, ids=[f"{p}->{n}" for p, _v, n in MUTATIONS])
def test_every_stateful_rule_is_named_on_its_mutation(kept, path, value, needle):
    rec = copy.deepcopy(kept.rec)
    _set(rec, path, value)
    ok, det = _check(kept, rec=rec)
    assert not ok and needle in det, (path, det)


def test_probe_sha_from_the_state_record_is_checked_against_the_recorded_probe(kept):
    ok, det = _check(kept, probe_pre_sha256="0" * 64)
    assert not ok and "creation-time probe" in det
    ok, det = _check(kept, probe_pre_sha256=SL.probe_sha256(json.loads(json.dumps(kept.probe))))
    assert ok, det


# ══ 2. the teardown record ═══════════════════════════════════════════════════════════════════════════════════
def test_teardown_record_needs_a_verified_wipe_and_the_ledger_row(kept):
    ok, det = _check(kept, teardown=kept.teardown)
    assert not ok and "container_down" in det                      # the wipe alone is not enough: ledgered
    _ledger_down(kept)
    ok, det = _check(kept, teardown=kept.teardown)
    assert ok, det
    assert "torn down (ledgered)" in det and "kept for the next epoch" not in det
    bad = copy.deepcopy(kept.teardown)
    bad["down"]["wiped"] = False
    ok, det = _check(kept, teardown=bad)
    assert not ok and "teardown not verified" in det
    bad = copy.deepcopy(kept.teardown)
    bad["down"] = None
    ok, det = _check(kept, teardown=bad)
    assert not ok and "teardown not verified" in det
    for k, v in (("cid", "run-other"), ("sandbox_id", "sbx-other")):
        bad = copy.deepcopy(kept.teardown)
        bad[k] = v
        ok, det = _check(kept, teardown=bad)
        assert not ok and "teardown record names" in det, det
    # a ledger row for ANOTHER container / machine does not count
    kept.ledger.write_text(json.dumps({"event": "container_down", "driver": pce.DRIVER, "id": SB, "cid": "run-other",
                                       "verified": True}) + "\n")
    ok, det = _check(kept, teardown=kept.teardown)
    assert not ok and "container_down" in det
    # the ledger row alone never turns a kept epoch into a teardown (nothing is demanded without the record)
    _ledger_down(kept)
    ok, det = _check(kept)
    assert ok and "kept for the next epoch" in det


def test_teardown_with_the_machine_absent_needs_machine_json_to_show_the_sandbox_gone(kept):
    assert not kept.ledger.exists()
    for row in (None, {"sandbox_id": SB, "state": "deleted"}, {"sandbox_id": "another", "state": "ready"}):
        ma = {"schema": "evoskill.evolver_session_teardown/1", "cid": CID, "sandbox_id": SB, "sid": SID,
              "machine_absent": {"machine_json": row, "utc": "2026-09-27T13:00:00Z"}, "abort": False}
        ok, det = _check(kept, teardown=ma)
        assert ok, (row, det)
        assert "machine absent" in det
    ma["machine_absent"]["machine_json"] = {"sandbox_id": SB, "state": "ready"}
    ok, det = _check(kept, teardown=ma)
    assert not ok and "still shows the sandbox" in det
    ma["machine_absent"] = "yes"
    ok, det = _check(kept, teardown=ma)
    assert not ok and "still shows the sandbox" in det
