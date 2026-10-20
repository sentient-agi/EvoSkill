"""src/orchestrator/stateful_layout.py (container layout v2): the RUNTIME
reader, the container-native path view (cumulative over the run) with vm_of / mac_of, the state record + snapshot
helpers, forbidden_mac_strings, build_stateful_manifest, after_hashes_vm and check_stateful_upload.

Fixture: a tiny run dir under tmp_path -- evo_e0..evo_e2 written directly in the evo_layout shape (traces/,
solver_workspaces/, scores.json), skills_v1 / skills_v2, the adapter's reference/ (holding a dataset.json: it is NOT
scanned), a RUNTIME.json declaring the stateful session, a split and a one-task fake dataset root. The view is a
pure mapping: the tests assert the exact table of the module docstring, never re-deriving it with the module's code.
Offline; nothing written outside tmp_path.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.orchestrator.loop as L  # noqa: E402
from src.orchestrator import daytona_evolver as de  # noqa: E402
from src.orchestrator import evidence_surface as es  # noqa: E402
from src.orchestrator import evo_layout as EL  # noqa: E402
from src.orchestrator import stateful_layout as SL  # noqa: E402

TASKS = ("Cat/01_01", "Cat/01_02")
VENV = {"HOME": de.VM_HOME}


def R(p) -> str:
    return os.path.realpath(str(p))


def _evo(rd: Path, K: int) -> Path:
    """<run>/evo_e{K} in the evo_layout shape: 2 tasks x 2 samples."""
    d = EL.epoch_dir(rd, K)
    (d / EL.TRACES).mkdir(parents=True)
    (d / EL.WORKSPACES).mkdir()
    rows = []
    for t in TASKS:
        for i in (1, 2):
            u = EL.unit_of(t, i)
            (d / EL.TRACES / f"{u}.jsonl").write_text(f'{{"type":"system","subtype":"init","epoch":{K},"unit":"{u}"}}\n')
            w = d / EL.WORKSPACES / u / "out"
            w.mkdir(parents=True)
            (w.parent / "task.txt").write_text(f"{t} sample {i} epoch {K}\n")
            (w / "answer.xlsx").write_bytes(b"XLSX " + u.encode() + str(K).encode())
            rows.append({"task_id": EL.safe_id(t), "sample": i, "score": 1.0 if i == 1 else 0.0})
    (d / EL.SCORES_JSON).write_text(json.dumps(rows, indent=1) + "\n")
    return d


def _skills(rd: Path, K: int, names) -> Path:
    d = rd / f"skills_v{K}"
    for n in names:
        (d / n).mkdir(parents=True)
        (d / n / "SKILL.md").write_text(f"---\nname: {n}\ndescription: Use when {n} applies (v{K}).\n---\n# {n}\n")
    (d / names[0] / "scripts").mkdir()
    (d / names[0] / "scripts" / "x.py").write_text("print(1)\n")
    return d


@pytest.fixture
def run(tmp_path, monkeypatch):
    rd = tmp_path / "outputs" / "runX"
    rd.mkdir(parents=True)
    for K in (0, 1, 2):
        _evo(rd, K)
    _skills(rd, 1, ["alpha", "beta"])
    _skills(rd, 2, ["alpha", "gamma"])
    ref = rd / SL.REFERENCE_DIR_NAME
    (ref / "grader").mkdir(parents=True)
    (ref / "README.md").write_text("what the adapter kept\n")
    (ref / "grader" / "score.py").write_text("def score(x):\n    return 1\n")
    (ref / "dataset.json").write_text("[]\n")                 # the adapter's copy is uploaded UNSCANNED
    (rd / "RUNTIME.json").write_text(json.dumps({"prompt_mode": "template", "evolver_session": "stateful",
                                                 "evolver_evidence": "all_epochs", "evolver_evidence_layout": "evo"}))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": list(TASKS), "val": ["Cat/02_01"], "test": ["Cat/03_01"]}))
    data = tmp_path / "data" / "Cat" / "s"                    # the dataset root the forbidden-sha rule reads
    data.mkdir(parents=True)
    (data / "01_01_input.xlsx").write_bytes(b"INPUT 01_01")
    (data / "01_01_golden.xlsx").write_bytes(b"GOLDEN 01_01")
    (data.parent / "dataset.json").write_text(json.dumps([{"id": "01_01", "spreadsheet_path": "s/01_01_input.xlsx",
                                                           "golden_response_path": "s/01_01_golden.xlsx"}]))
    monkeypatch.setenv("EVOSKILL_SSBV2_DATA", str(tmp_path / "data"))
    return SimpleNamespace(rd=rd, split=split)


# ══ 1. the RUNTIME reader ═══════════════════════════════════════════════════════════════════════════════════
def test_evolver_session_reads_the_file_and_refuses_a_typo(tmp_path):
    rd = tmp_path / "r"
    rd.mkdir()
    assert SL.evolver_session(rd) == "per_epoch"                       # no RUNTIME.json
    assert SL.evolver_session(None) == "per_epoch"
    (rd / "RUNTIME.json").write_text(json.dumps({"prompt_mode": "template"}))
    assert SL.evolver_session(rd) == "per_epoch"                       # absent key
    (rd / "RUNTIME.json").write_text(json.dumps({"evolver_session": "stateful"}))
    assert SL.evolver_session(rd) == "stateful"
    (rd / "RUNTIME.json").write_text(json.dumps({"evolver_session": "per_epoch"}))
    assert SL.evolver_session(rd) == "per_epoch"
    assert SL.evolver_session(runtime={"evolver_session": "stateful"}) == "stateful"
    for bad in ("statefull", "Stateful", "", True, None, ["stateful"], 1):
        with pytest.raises(ValueError, match="evolver_session"):
            SL.evolver_session(runtime={"evolver_session": bad})
    (rd / "RUNTIME.json").write_text("{not json")
    with pytest.raises(ValueError, match="unreadable"):
        SL.evolver_session(rd)
    with pytest.raises(ValueError, match="object"):
        SL.evolver_session(runtime=["stateful"])
    assert SL.SESSIONS == ("per_epoch", "stateful") and SL.STATEFUL == "stateful" and SL.SESSION_KEY == "evolver_session"


# ══ 2. the layout constants and the small maps ══════════════════════════════════════════════════════════════
def test_layout_constants_and_epoch_maps():
    assert SL.traces_dir(0) == "/evo/traces/epoch_0" and SL.traces_dir(3) == "/evo/traces/epoch_3"
    assert SL.scores_file(0) == "/evo/scores/epoch_0.json" and SL.solver_ws_dir(2) == "/evo/solver_workspaces/epoch_2"
    assert SL.skills_dir(1) == "/evo/skills/epoch_1"
    for bad in (-1, True, "1", 1.0, None):
        with pytest.raises(ValueError):
            SL.traces_dir(bad)
    with pytest.raises(ValueError):
        SL.skills_dir(0)                                               # epoch 0 has no skill set
    assert SL.evo_epoch("evo_e0") == 0 and SL.evo_epoch(Path("/x/run/evo_e12")) == 12
    for bad in ("evo_e", "evo_e1_train", "digests_e1_train", "skills_v1", "evo_e1.bak"):
        with pytest.raises(ValueError):
            SL.evo_epoch(bad)
    assert SL.EVOLVER_WRITABLE == ("/evo/staging", "/evo/evolver_workspace", "/evo/work", "/evo/out")
    assert [d["path"] for d in SL.STATEFUL_RW] == ["/evo/staging", "/evo/evolver_workspace"]
    assert not hasattr(SL, "GRADE_CMD_VM") and not hasattr(SL, "VM_GRADING")   # the evolver has no grader
    assert all(d["owner"] == "evo" and d["mode"] == "0755" for d in SL.STATEFUL_RW)
    assert SL.VM_REFERENCE == "/evo/reference" and SL.REFERENCE_DIR_NAME == "reference"
    assert SL.STATE_ROOT == de.REPO / ".cache/evolver_state"
    assert SL.WORKSPACE_DIR.format(epoch=3) == "e3_evolver_workspace" and SL.SNAPSHOT_WORKSPACE == "workspace"
    assert SL.SYSTEM_PROMPT_FILE == "EVOLVER_SYSTEM.md"
    assert SL.ENV_SESSION == "EVOSKILL_EVOLVER_SESSION" and SL.ENV_SYSTEM_PROMPT == "EVOSKILL_EVOLVER_SYSTEM_PROMPT"
    assert SL.STATE_SCHEMA == "evoskill.evolver_session/1" and SL.SNAPSHOT_SCHEMA == "evoskill.evolver_snapshot/1"
    # the roles: the per-epoch material, the dir / file split, and what is never scanned
    assert SL.EVO_ROLES == ("traces", "solver_workspaces", "scores") and SL.FILE_ROLES == ("scores",)
    assert SL.DIR_ROLES == ("traces", "solver_workspaces", "skills_snapshot", "seed", "reference")
    assert SL.STATEFUL_ROLES == {"traces", "solver_workspaces", "scores", "skills_snapshot", "seed", "reference",
                                 "hook", "env", "policy"}                       # no "tool" role: no grader
    assert SL.UNSCANNED_ROLES == {"traces", "solver_workspaces", "scores", "reference"}
    # no grader / grants / prompt files in the container any more
    for gone in ("VM_GRADER", "VM_MESSAGE", "VM_SYSTEM", "NOTES_DIR", "MAC_NAMES_FORBIDDEN", "evidence_epoch"):
        assert not hasattr(SL, gone), gone


# ══ 3. the path view: the exact table of the module docstring for epochs 1..3 ══════════════════════════════
def _rows(view):
    return [(pm.mac, pm.vm, pm.role, pm.epoch) for pm in view]


def _evo_rows(rd, K):
    d = rd / f"evo_e{K}"
    return [(R(d / "traces"), f"/evo/traces/epoch_{K}", "traces", K + 1),
            (R(d / "solver_workspaces"), f"/evo/solver_workspaces/epoch_{K}", "solver_workspaces", K + 1),
            (R(d / "scores.json"), f"/evo/scores/epoch_{K}.json", "scores", K + 1)]


def test_path_view_epoch1_is_the_base_material_and_the_reference(run):
    rd = run.rd
    view = SL.path_view(rd, 1)
    assert _rows(view) == _evo_rows(rd, 0) + [(R(rd / "reference"), "/evo/reference", "reference", 1)]
    assert all(isinstance(pm, SL.PathMap) for pm in view)
    assert view[0].as_dict() == {"mac": R(rd / "evo_e0/traces"), "vm": "/evo/traces/epoch_0", "role": "traces", "epoch": 1}
    assert {pm.epoch for pm in view} == {1}                            # every entry travels at epoch 1


def test_path_view_epoch2_adds_the_delta_and_keeps_epoch1(run):
    rd = run.rd
    view = SL.path_view(rd, 2)
    assert _rows(view) == _evo_rows(rd, 0) + _evo_rows(rd, 1) + [
        (R(rd / "skills_v1"), "/evo/skills/epoch_1", "skills_snapshot", 2),
        (R(rd / "skills_v1"), "/evo/staging", "seed", 2),
        (R(rd / "reference"), "/evo/reference", "reference", 1),
    ]
    delta = [(pm.vm, pm.role) for pm in view if pm.epoch == 2]
    assert delta == [("/evo/traces/epoch_1", "traces"), ("/evo/solver_workspaces/epoch_1", "solver_workspaces"),
                     ("/evo/scores/epoch_1.json", "scores"), ("/evo/skills/epoch_1", "skills_snapshot"),
                     ("/evo/staging", "seed")]
    # cumulative: everything epoch 1 uploaded is still in the view, unchanged
    assert [pm for pm in view if pm.epoch == 1] == SL.path_view(rd, 1)


def test_path_view_epoch3_is_cumulative_and_reseeds_from_skills_v2(run):
    rd = run.rd
    v2, v3 = SL.path_view(rd, 2), SL.path_view(rd, 3)
    assert [(pm.mac, pm.vm, pm.role) for pm in v3 if pm.epoch == 3] == \
        [(m, v, r) for m, v, r, _e in _evo_rows(rd, 2)] + [(R(rd / "skills_v2"), "/evo/skills/epoch_2", "skills_snapshot"),
                                                          (R(rd / "skills_v2"), "/evo/staging", "seed")]
    # the seed is the ONLY entry that changes between epochs (/evo/staging is cleared and re-seeded)
    assert [pm for pm in v3 if pm.epoch <= 2 and pm.role != "seed"] == [pm for pm in v2 if pm.role != "seed"]
    assert [pm for pm in v2 if pm.role == "seed"] == [SL.PathMap(R(rd / "skills_v1"), "/evo/staging", "seed", 2)]
    assert [pm for pm in v3 if pm.role == "seed"] == [SL.PathMap(R(rd / "skills_v2"), "/evo/staging", "seed", 3)]
    assert [(pm.vm, pm.epoch) for pm in v3 if pm.role == "skills_snapshot"] == [("/evo/skills/epoch_1", 2),
                                                                                ("/evo/skills/epoch_2", 3)]
    assert [(pm.vm, pm.epoch) for pm in v3 if pm.role == "reference"] == [("/evo/reference", 1)]
    assert [pm.role for pm in v3] == ["traces", "solver_workspaces", "scores"] * 3 + ["skills_snapshot"] * 2 + ["seed", "reference"]
    # no vm root is shared, and no vm path is a prefix of another (the container holds them side by side)
    vms = [pm.vm for pm in v3]
    assert len(set(vms)) == len(vms) and not any(a != b and b.startswith(a + "/") for a in vms for b in vms)
    # the Mac side is the loop's own naming, never a second format string
    macs = {pm.mac for pm in v3}
    for want in (L.EpochPaths(rd, 1).evo, L.EpochPaths(rd, 3).evo, EL.epoch_dir(rd, 1) / EL.SCORES_JSON,
                 L.EpochPaths(rd, 2).promoted, rd / SL.REFERENCE_DIR_NAME):
        assert R(want) in macs or any(m.startswith(R(want) + "/") for m in macs), want


def test_path_view_refuses_bad_epochs_and_a_dir_outside_the_run(run, tmp_path):
    rd = run.rd
    for bad in (0, -1, True, "1"):
        with pytest.raises(ValueError):
            SL.path_view(rd, bad)
    # a symlinked evo dir that resolves OUTSIDE the run dir is not this run's (realpath, then the check)
    other = tmp_path / "elsewhere" / "evo_e3"
    (other / "traces").mkdir(parents=True)
    (rd / "evo_e3").symlink_to(other)
    with pytest.raises(ValueError, match="outside this run dir"):
        SL.path_view(rd, 4)
    assert SL.path_view(rd, 3)                                         # a future epoch's dir is not looked at
    (rd / "evo_e3").unlink()
    (rd / "evo_e3.real" / "traces").mkdir(parents=True)                # ... while one INSIDE resolves to its target
    (rd / "evo_e3").symlink_to(rd / "evo_e3.real")
    assert [pm.mac for pm in SL.path_view(rd, 4) if pm.vm == "/evo/traces/epoch_3"] == [R(rd / "evo_e3.real/traces")]


# ══ 4. vm_of / mac_of ═══════════════════════════════════════════════════════════════════════════════════════
def test_vm_of_and_mac_of_round_trip_every_kind_of_path(run):
    rd = run.rd
    view = SL.path_view(rd, 2)
    t = rd / "evo_e0/traces/Cat_01_01_s1.jsonl"
    assert SL.vm_of(view, t) == "/evo/traces/epoch_0/Cat_01_01_s1.jsonl" and SL.mac_of(view, SL.vm_of(view, t)) == R(t)
    assert SL.vm_of(view, rd / "evo_e1/solver_workspaces/Cat_01_02_s2/out/answer.xlsx") == \
        "/evo/solver_workspaces/epoch_1/Cat_01_02_s2/out/answer.xlsx"
    assert SL.vm_of(view, rd / "evo_e1/scores.json") == "/evo/scores/epoch_1.json"
    assert SL.vm_of(view, rd / "evo_e1/traces") == "/evo/traces/epoch_1"                 # the dir itself
    assert SL.mac_of(view, "/evo/traces/epoch_1") == R(rd / "evo_e1/traces")
    assert SL.mac_of(view, "/evo/traces/epoch_1/") == R(rd / "evo_e1/traces")            # normalised
    assert SL.mac_of(view, "/evo/scores/epoch_0.json") == R(rd / "evo_e0/scores.json")
    assert SL.mac_of(view, "/evo/staging/alpha/SKILL.md") == R(rd / "skills_v1/alpha/SKILL.md")
    assert SL.mac_of(view, "/evo/skills/epoch_1/alpha/SKILL.md") == R(rd / "skills_v1/alpha/SKILL.md")
    assert SL.mac_of(view, "/evo/reference/grader/score.py") == R(rd / "reference/grader/score.py")
    assert SL.vm_of(view, rd / "reference/README.md") == "/evo/reference/README.md"
    # skills_v1 is BOTH the read-only snapshot and the writable seed: the snapshot wins the tie; `role` picks
    assert SL.vm_of(view, rd / "skills_v1") == "/evo/skills/epoch_1"
    assert SL.vm_of(view, rd / "skills_v1/alpha/SKILL.md", role="seed") == "/evo/staging/alpha/SKILL.md"
    assert SL.vm_of(view, rd / "skills_v1", role="skills_snapshot") == "/evo/skills/epoch_1"
    for vmp in ("/evo/traces/epoch_0", "/evo/solver_workspaces/epoch_1", "/evo/scores/epoch_1.json",
                "/evo/skills/epoch_1", "/evo/reference"):
        assert SL.vm_of(view, SL.mac_of(view, vmp)) == vmp
    assert SL.vm_of(view, SL.mac_of(view, "/evo/staging")) == "/evo/skills/epoch_1"    # the tie rule


def test_vm_of_and_mac_of_raise_keyerror_on_anything_unmapped(run):
    rd = run.rd
    v1, v2 = SL.path_view(rd, 1), SL.path_view(rd, 2)
    for mac in (rd, rd / "runX_v1_train.json", rd / "skills_v1_staging", rd / "evo_e1", rd / "evo_e1.bak/traces/x",
                rd / "EVOLVER_SYSTEM.md", rd / "ITER2_PROMPT.md", rd / "e2_evolver_workspace", "/etc/passwd"):
        with pytest.raises(KeyError):
            SL.vm_of(v2, mac)
    with pytest.raises(KeyError):
        SL.vm_of(v1, rd / "skills_v1")                       # not in the container at epoch 1
    with pytest.raises(KeyError):
        SL.vm_of(v1, rd / "evo_e1/traces")                   # a FUTURE epoch's material is not mapped
    with pytest.raises(KeyError):
        SL.vm_of(v2, rd / "skills_v1", role="inputs")        # no entry of that role
    for vm in ("/evo/traces/epoch_2", "/evo/traces/epoch_10/x.jsonl", "/evo/traces", "/evo/scores", "/evo/work",
               "/evo/work/MESSAGE_e2.md", "/evo/evolver_workspace/x.md", "/evo/grader/x.py", "/evo",
               "/data/x/run/evo_e0/traces", "relative/path", "/evo/skills/epoch_2"):
        with pytest.raises(KeyError):
            SL.mac_of(v2, vm)
    with pytest.raises(KeyError):
        SL.mac_of(v2, "/evo/traces/epoch_1/x", role="seed")


# ══ 5. the state record ═════════════════════════════════════════════════════════════════════════════════════
def _state(rd, **kw):
    st = {"schema": SL.STATE_SCHEMA, "run": rd.name, "run_dir": str(rd), "cid": "run-20260927t120000-abc123",
          "sandbox_id": "sb-1", "sandbox_name": "paper-ssb-0924-loop-x", "sid": str(uuid.uuid4()),
          "created_epoch": 1, "created_utc": "2026-09-27T12:00:00Z", "system_prompt_sha256": "0" * 64,
          "container": {}, "uploaded": [], "epochs": {}, "rebuilds": [], "torn_down": None}
    st.update(kw)
    return st


def test_state_record_round_trip_atomic_replace_and_private_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(SL, "STATE_ROOT", tmp_path / "cache" / "evolver_state")
    rd = tmp_path / "outputs" / "runX"
    rd.mkdir(parents=True)
    assert SL.state_path(rd) == tmp_path / "cache/evolver_state/runX/container.json"
    assert SL.snapshot_dir(rd, 2) == tmp_path / "cache/evolver_state/runX/e2"
    assert SL.load_state(rd) is None                                 # never written
    st = _state(rd)
    SL.write_state(rd, st)
    p = SL.state_path(rd)
    assert p.is_file() and json.loads(p.read_text()) == st and SL.load_state(rd) == st
    for d in (SL.STATE_ROOT, p.parent):
        assert stat.S_IMODE(d.stat().st_mode) == 0o700, d
    assert stat.S_IMODE(p.stat().st_mode) == 0o600
    assert [x.name for x in p.parent.iterdir()] == ["container.json"]   # no tmp file left behind
    # a second write REPLACES (the reader never sees a partial file; the old bytes are gone)
    st2 = _state(rd, sid=st["sid"], epochs={"1": {"status": "collected"}})
    SL.write_state(rd, st2)
    assert SL.load_state(rd) == st2 and [x.name for x in p.parent.iterdir()] == ["container.json"]
    # refused writes leave the record untouched
    for bad in ({}, {"schema": "other/1", "run": "runX"}, _state(rd, schema="evoskill.evolver_session/2"),
                _state(rd, run="runY"), [st2], "x"):
        with pytest.raises(ValueError):
            SL.write_state(rd, bad)
    assert SL.load_state(rd) == st2
    # a record that exists but is corrupt / another run's is an error, never "absent"
    p.write_text("{corrupt")
    with pytest.raises(ValueError, match="unreadable"):
        SL.load_state(rd)
    p.write_text(json.dumps(_state(rd, run="runY")))
    with pytest.raises(ValueError, match="runY"):
        SL.load_state(rd)
    p.write_text(json.dumps({"schema": "x", "run": "runX"}))
    with pytest.raises(ValueError, match="schema"):
        SL.load_state(rd)
    with pytest.raises(ValueError):
        SL.snapshot_dir(rd, 0)


# ══ 6. snapshots ════════════════════════════════════════════════════════════════════════════════════════════
def _snapshot(d: Path, sid="11111111-2222-4333-8444-555555555555") -> Path:
    (d / "projects" / "-evo-work").mkdir(parents=True)
    (d / "projects" / "-evo-work" / f"{sid}.jsonl").write_text('{"type":"user"}\n{"type":"assistant"}\n')
    (d / "projects" / "-evo-work" / "quota-probe.jsonl").write_text("{}\n")
    (d / SL.SNAPSHOT_WORKSPACE).mkdir()
    (d / SL.SNAPSHOT_WORKSPACE / "PLAN.md").write_text("# notes\n")
    (d / "receipt.json").write_text("{}")
    return d


def _write_manifest(d: Path, **extra):
    man = SL.snapshot_manifest(d)
    man.update({"run": "r", "cid": "run-x", "sandbox_id": "sb", "sid": "s", "epoch": 1, "utc": "u", "claude_rc": 0})
    man.update(extra)
    (d / SL.SNAPSHOT_MANIFEST).write_text(json.dumps(man, indent=1))
    return man


def test_snapshot_manifest_lists_every_regular_file_and_verify_rehashes(tmp_path):
    d = _snapshot(tmp_path / "e1")
    man = _write_manifest(d)
    assert man["schema"] == SL.SNAPSHOT_SCHEMA
    assert sorted(man["files"]) == ["projects/-evo-work/11111111-2222-4333-8444-555555555555.jsonl",
                                    "projects/-evo-work/quota-probe.jsonl", "receipt.json", "workspace/PLAN.md"]
    assert man["n_files"] == 4 and man["bytes"] == sum(v["bytes"] for v in man["files"].values())
    assert all(len(v["sha256"]) == 64 for v in man["files"].values())
    assert man["files"]["workspace/PLAN.md"]["bytes"] == len("# notes\n")
    assert SL.verify_snapshot(d) == []
    # MANIFEST.json itself is never listed, and a re-run of the manifest is byte-stable
    assert SL.SNAPSHOT_MANIFEST not in SL.snapshot_manifest(d)["files"]
    assert SL.snapshot_manifest(d)["files"] == man["files"]


def test_verify_snapshot_fails_on_a_changed_byte_an_unlisted_file_a_missing_file_a_symlink_and_contamination(tmp_path):
    d = _snapshot(tmp_path / "e1")
    ws = d / SL.SNAPSHOT_WORKSPACE
    _write_manifest(d)
    assert SL.verify_snapshot(d) == []
    (ws / "PLAN.md").write_text("# notes!\n")                          # changed (size + sha)
    probs = SL.verify_snapshot(d)
    assert any("workspace/PLAN.md" in p and "bytes" in p for p in probs) and any("sha256" in p for p in probs)
    (ws / "PLAN.md").write_text("# NOTES\n")                           # same size, other bytes
    assert [p for p in SL.verify_snapshot(d) if "sha256" in p] and not [p for p in SL.verify_snapshot(d) if "bytes" in p]
    (ws / "PLAN.md").write_text("# notes\n")
    assert SL.verify_snapshot(d) == []
    (ws / "extra.md").write_text("x")                                  # unlisted
    assert SL.verify_snapshot(d) == ["unlisted file in the snapshot: workspace/extra.md"]
    (ws / "extra.md").unlink()
    (d / "receipt.json").unlink()                                      # listed, missing
    assert SL.verify_snapshot(d) == ["listed file missing: receipt.json"]
    (d / "receipt.json").write_text("{}")
    (ws / "link.md").symlink_to(ws / "PLAN.md")                        # a symlink is never copied, never OK
    assert any("symlink" in p for p in SL.verify_snapshot(d))
    (ws / "link.md").unlink()
    (d / "ext").symlink_to(tmp_path)                                   # a symlinked DIR is refused, not walked
    assert any("symlink" in p for p in SL.verify_snapshot(d))
    (d / "ext").unlink()
    assert SL.verify_snapshot(d) == []
    _write_manifest(d, contaminated=True)                              # unusable for a resume
    assert any("contaminated" in p for p in SL.verify_snapshot(d))
    _write_manifest(d)
    assert SL.verify_snapshot(d) == []
    # no manifest / wrong schema / no dir
    (d / SL.SNAPSHOT_MANIFEST).unlink()
    assert SL.verify_snapshot(d) == [f"{d / SL.SNAPSHOT_MANIFEST} missing"]
    (d / SL.SNAPSHOT_MANIFEST).write_text(json.dumps({"schema": "x", "files": {}}))
    assert any("schema" in p for p in SL.verify_snapshot(d))
    (d / SL.SNAPSHOT_MANIFEST).write_text("{nope")
    assert any("unreadable" in p for p in SL.verify_snapshot(d))
    (d / SL.SNAPSHOT_MANIFEST).write_text(json.dumps({"schema": SL.SNAPSHOT_SCHEMA,
                                                      "files": {"../x": {"sha256": "0" * 64, "bytes": 1}}}))
    assert any("unsafe path" in p for p in SL.verify_snapshot(d))
    assert SL.verify_snapshot(tmp_path / "nope") == [f"snapshot dir {tmp_path / 'nope'} does not exist"]
    with pytest.raises(ValueError):
        SL.snapshot_manifest(tmp_path / "nope")
    (ws / "link.md").symlink_to(ws / "PLAN.md")
    with pytest.raises(ValueError, match="symlink"):
        SL.snapshot_manifest(d)


# ══ 7. forbidden Mac strings (the renderer refuses a text that names any) ═══════════════════════════════════
def test_forbidden_mac_strings_names_each_needle_and_allows_the_container_paths(monkeypatch):
    monkeypatch.setenv("EVOSKILL_SSBV2_DATA", "/opt/fake_dataset_root")
    clean = ("Read /evo/traces/epoch_1/Cat_01_01_s1.jsonl and /evo/scores/epoch_1.json; the set `skills_v1` is at "
             "/evo/skills/epoch_1/; write to /evo/staging/ and keep notes in /evo/evolver_workspace/; "
             "/evo/reference/ is the adapter's copy. skills_v2. `skills_v12` skills_v{N}\n")
    assert SL.forbidden_mac_strings(clean) == []
    for needle in ("/Users/someone/x", "/users/x", "evo_e1/traces", "EVO_E0", "digests_e2_train", "skills_v1_staging",
                   "skills_v1/", "skills_v12_", "/opt/fake_dataset_root/Template", "/OPT/FAKE_DATASET_ROOT"):
        hits = SL.forbidden_mac_strings(f"line one\nsee {needle} here\nline three\n")
        assert hits and all("line 2" in h and "see " in h for h in hits), (needle, hits)
    hits = SL.forbidden_mac_strings("a /Users/b\nc /Users/d\n")
    assert len(hits) == 1 and "2 occurrence(s)" in hits[0] and "line 1" in hits[0] and hits[0].startswith("/Users/")
    assert [h for h in SL.forbidden_mac_strings("skills_v1/a") if "Mac dir" in h]
    assert [h for h in SL.forbidden_mac_strings("x /opt/fake_dataset_root y") if "dataset root" in h]
    assert SL.forbidden_mac_strings("") == []


# ══ 8. the manifest ═════════════════════════════════════════════════════════════════════════════════════════
def _build(r, epoch, before, **over):
    kw = dict(run_dir=r.rd, epoch=epoch, env_text=de.render_env_file(VENV), hooks_text=de.hook_settings_text(),
              split=r.split, uploaded_before=before)
    kw.update(over)
    return SL.build_stateful_manifest(**kw)


def _nfiles(d) -> int:
    return sum(1 for p in Path(d).rglob("*") if p.is_file())


def _roles(rows) -> dict:
    out = {}
    for r in rows:
        out[r["role"]] = out.get(r["role"], 0) + 1
    return out


def _lib(d: Path) -> dict:
    return {f.relative_to(d).as_posix(): de.sha256_file(f) for f in d.rglob("*") if f.is_file()}


def test_build_stateful_manifest_epoch1_is_the_base_material_the_reference_and_hook_env_policy(run):
    rd = run.rd
    rows, gen = _build(run, 1, [])
    assert _roles(rows) == {"traces": 4, "solver_workspaces": 8, "scores": 1, "reference": 3, "hook": 2, "env": 1, "policy": 1}
    assert not [r for r in rows if r["vm_path"].startswith(("/evo/bin", "/evo/grading"))]   # no grader
    assert sorted(gen) == sorted([f"{de.VM_HOOKS}/settings.json", de.VM_ENV_FILE, de.VM_POLICY])
    assert gen[de.VM_POLICY] == de.policy_settings_text().encode()
    assert all(r["epoch"] == 1 for r in rows)
    assert all(r["vm_path"].startswith(("/evo/", "/etc/claude-code/")) for r in rows)
    assert set(rows[0]) == {"vm_path", "role", "source", "sha256", "bytes", "epoch"}
    for r in rows:
        if r["source"] != "generated":
            assert r["source"] == R(r["source"]) and de.sha256_file(r["source"]) == r["sha256"]
            assert os.path.getsize(r["source"]) == r["bytes"]
    view = SL.path_view(rd, 1)
    tr = [r for r in rows if r["role"] == "traces"]
    assert {SL.mac_of(view, r["vm_path"]) for r in tr} == {R(p) for p in (rd / "evo_e0/traces").iterdir()}
    assert {r["vm_path"] for r in rows if r["role"] == "scores"} == {"/evo/scores/epoch_0.json"}
    assert {r["vm_path"] for r in rows if r["role"] == "reference"} == {f"/evo/reference/{rel}" for rel in _lib(rd / "reference")}
    assert "/evo/reference/dataset.json" in {r["vm_path"] for r in rows}         # unscanned: uploaded as it is
    assert all(r["vm_path"].startswith("/evo/solver_workspaces/epoch_0/Cat_01_0") for r in rows if r["role"] == "solver_workspaces")
    assert [r["source"] for r in rows if r["vm_path"].endswith("evolver_sync_agent_hook.py")] == [str(de.SYNC_HOOK)]
    order = ["traces", "solver_workspaces", "scores", "reference", "hook", "env", "policy", "tool"]
    assert [r["role"] for r in rows] == sorted((r["role"] for r in rows), key=order.index)
    assert not [r for r in rows if r["role"] in ("prompt", "system", "grader", "train_reference", "inputs")]
    assert _build(run, 1, [])[0] == rows                              # byte-stable


def test_build_stateful_manifest_epoch2_delta_is_the_new_material_the_snapshot_and_the_seed(run):
    rd = run.rd
    e1, _ = _build(run, 1, [])
    rows, gen = _build(run, 2, e1)
    assert gen == {}
    lib = _lib(rd / "skills_v1")
    assert len(lib) == 3
    assert _roles(rows) == {"traces": 4, "solver_workspaces": 8, "scores": 1, "skills_snapshot": 3, "seed": 3}
    assert all(r["epoch"] == 2 for r in rows)
    assert {r["vm_path"]: r["sha256"] for r in rows if r["role"] == "seed"} == {f"/evo/staging/{k}": v for k, v in lib.items()}
    assert {r["vm_path"]: r["sha256"] for r in rows if r["role"] == "skills_snapshot"} == \
        {f"/evo/skills/epoch_1/{k}": v for k, v in lib.items()}
    assert all(r["vm_path"].startswith("/evo/traces/epoch_1/") for r in rows if r["role"] == "traces")
    assert [r["vm_path"] for r in rows if r["role"] == "scores"] == ["/evo/scores/epoch_1.json"]
    assert [r["role"] for r in rows] == sorted((r["role"] for r in rows),
                                               key=["traces", "solver_workspaces", "scores", "skills_snapshot", "seed"].index)
    assert not ({r["vm_path"] for r in rows} & {r["vm_path"] for r in e1})     # nothing of epoch 1 travels again


def test_build_stateful_manifest_epoch3_reseeds_staging_over_epoch2s_seed(run):
    rd = run.rd
    e1, _ = _build(run, 1, [])
    e2, _ = _build(run, 2, e1)
    rows, gen = _build(run, 3, e1 + e2)
    assert gen == {}
    lib2 = _lib(rd / "skills_v2")
    seeds = {r["vm_path"]: r["sha256"] for r in rows if r["role"] == "seed"}
    assert seeds == {f"/evo/staging/{k}": v for k, v in lib2.items()}
    # the SAME /evo/staging paths as epoch 2's seed: allowed, epoch-begin cleared the dir
    assert set(seeds) & {r["vm_path"] for r in e2 if r["role"] == "seed"}
    assert {r["vm_path"]: r["sha256"] for r in rows if r["role"] == "skills_snapshot"} == \
        {f"/evo/skills/epoch_2/{k}": v for k, v in lib2.items()}
    assert all(r["vm_path"].startswith("/evo/traces/epoch_2/") for r in rows if r["role"] == "traces")
    # a repeated path OUTSIDE /evo/staging is still refused (extract --incremental is create-only)
    with pytest.raises(ValueError, match="already in the container"):
        _build(run, 3, e1 + e2 + [dict(rows[0])])
    with pytest.raises(ValueError, match="already in the container"):
        _build(run, 3, e1 + e2 + [dict(r) for r in rows if r["role"] == "skills_snapshot"][:1])


def test_build_stateful_manifest_refusals(run, tmp_path):
    rd = run.rd
    e1, _ = _build(run, 1, [])

    def refused(match, epoch=2, before=None, **over):
        with pytest.raises(ValueError, match=match):
            _build(run, epoch, e1 if before is None else before, **over)
    refused("does not hold", before=[])                                             # epoch 1 never uploaded
    refused("does not hold /evo/reference", before=[r for r in e1 if r["role"] != "reference"])
    refused("already in the container", before=e1 + [dict(e1[0], vm_path="/evo/scores/epoch_1.json")])
    refused("already in the container", epoch=1, before=e1)
    refused("uploaded_before", before=["x"])
    refused("uploaded_before", before=e1 + [{"role": "seed"}])
    for bad in (0, True, "2"):
        refused("epoch index", epoch=bad)
    # a symlink in the previous library / in this epoch's traces; a scores.json that is no regular file
    (rd / "skills_v1" / "link.md").symlink_to(rd / "reference" / "README.md")
    refused("symlink")
    (rd / "skills_v1" / "link.md").unlink()
    (rd / "evo_e1" / "traces" / "link.jsonl").symlink_to(rd / "evo_e0" / "traces" / "Cat_01_01_s1.jsonl")
    refused("symlink")
    (rd / "evo_e1" / "traces" / "link.jsonl").unlink()
    (rd / "evo_e1" / "scores.json").rename(rd / "evo_e1" / "scores.json.aside")
    refused("not a regular file")
    (rd / "evo_e1" / "scores.json.aside").rename(rd / "evo_e1" / "scores.json")
    (rd / "evo_e1" / "solver_workspaces").rename(rd / "evo_e1" / "ws.aside")
    refused("not a plain directory")
    (rd / "evo_e1" / "ws.aside").rename(rd / "evo_e1" / "solver_workspaces")
    # a dir outside this run (a symlinked skills_v1) is refused before anything is read
    (rd / "skills_v1").rename(tmp_path / "elsewhere_v1")
    (rd / "skills_v1").symlink_to(tmp_path / "elsewhere_v1")
    refused("outside this run dir")
    (rd / "skills_v1").unlink()
    (tmp_path / "elsewhere_v1").rename(rd / "skills_v1")
    # the scanned roles (skills / seed): grading-validation material and the split file are refused ...
    (rd / "skills_v1" / "alpha" / "GRADING_VALIDATION.json").write_text("{}")
    refused("grading-validation")
    (rd / "skills_v1" / "alpha" / "GRADING_VALIDATION.json").unlink()
    # ... while the adapter's reference/ and the solver's own material are NOT scanned (dataset.json is in it)
    (rd / "evo_e1" / "traces" / "GRADING_VALIDATION.json").write_text("{}")
    assert _roles(_build(run, 2, e1)[0])["traces"] == 5
    (rd / "evo_e1" / "traces" / "GRADING_VALIDATION.json").unlink()
    assert _build(run, 2, e1)[0] and _build(run, 1, [])[0] == e1


# ══ 9. after_hashes_vm / probe_sha256 ═══════════════════════════════════════════════════════════════════════
def test_after_hashes_vm_hashes_the_container_listing_at_each_mapped_dirs_root(run):
    rd = run.rd
    e1, _ = _build(run, 1, [])
    e2, _ = _build(run, 2, e1)
    view = SL.path_view(rd, 2)
    listing = [[r["vm_path"], "F", r["sha256"], r["bytes"]] for r in e1 + e2] + \
        [["/evo/out", "D", "", 0], ["/evo/evolver_workspace/x.md", "F", "a" * 64, 1], ["/evo/staging/new/SKILL.md", "F", "b" * 64, 2]]
    got = SL.after_hashes_vm(listing, view)
    assert set(got) == {pm.vm for pm in view if pm.role in SL.DIR_ROLES} == \
        {"/evo/traces/epoch_0", "/evo/solver_workspaces/epoch_0", "/evo/traces/epoch_1", "/evo/solver_workspaces/epoch_1",
         "/evo/skills/epoch_1", "/evo/staging", "/evo/reference"}
    for pm in view:
        if pm.role in SL.DIR_ROLES and pm.role != "seed":       # the container's bytes == the Mac dir's content hash
            assert got[pm.vm] == es.dir_sha256(pm.mac), pm.vm
    assert got["/evo/staging"] != es.dir_sha256(rd / "skills_v1")   # the evolver's new SKILL.md is in the listing
    # a byte changed in the container shows in that dir's hash only
    bad = [row if row[0] != listing[0][0] else [row[0], "F", "0" * 64, row[3]] for row in listing]
    got2 = SL.after_hashes_vm(bad, view)
    assert got2["/evo/traces/epoch_0"] != got["/evo/traces/epoch_0"]
    assert {k: v for k, v in got2.items() if k != "/evo/traces/epoch_0"} == {k: v for k, v in got.items() if k != "/evo/traces/epoch_0"}


def test_probe_sha256_is_canonical():
    a = {"b": 1, "a": {"y": [1, 2], "x": "s"}}
    b = {"a": {"x": "s", "y": [1, 2]}, "b": 1}
    assert SL.probe_sha256(a) == SL.probe_sha256(b) and len(SL.probe_sha256(a)) == 64
    assert SL.probe_sha256({"b": 2, "a": {"y": [1, 2], "x": "s"}}) != SL.probe_sha256(a)
    assert SL.probe_sha256({}) == hashlib.sha256(b"{}").hexdigest()


# ══ 10. check_stateful_upload ═══════════════════════════════════════════════════════════════════════════════
def _receipt(rd, epoch, rows, cum, view) -> dict:
    return {"containment": "daytona", "sandboxed": True, "upload_manifest": rows, "upload_manifest_cumulative": cum,
            "path_view": [pm.as_dict() for pm in view],
            "evidence_sha256_before": es.hashes([rd / f"evo_e{K}" for K in range(epoch)]),
            "daytona": {"vm_env": dict(VENV)}}


def _listing(cum) -> dict:
    return {"pre_upload": {"users_exists": False},
            "after_upload": [[r["vm_path"], "F", r["sha256"], r["bytes"]] for r in cum]}


@pytest.fixture
def two_epochs(run):
    rd = run.rd
    e1, _ = _build(run, 1, [])
    rc1 = _receipt(rd, 1, e1, list(e1), SL.path_view(rd, 1))
    (rd / SL.RECEIPT_FILE.format(epoch=1)).write_text(json.dumps(rc1))
    e2, _ = _build(run, 2, e1)
    rc2 = _receipt(rd, 2, e2, e1 + e2, SL.path_view(rd, 2))
    return SimpleNamespace(r=run, rd=rd, e1=e1, e2=e2, rc1=rc1, rc2=rc2, st1=_state(rd, uploaded=list(e1)),
                           st2=_state(rd, uploaded=e1 + e2), l1=_listing(e1), l2=_listing(e1 + e2))


def _check(t, epoch=2, **over):
    kw = dict(run_dir=t.rd, epoch=epoch, split=t.r.split, state=t.st2 if epoch == 2 else t.st1,
              receipt=t.rc2 if epoch == 2 else t.rc1, listing=t.l2 if epoch == 2 else t.l1)
    kw.update(over)
    return SL.check_stateful_upload(**kw)


def test_check_stateful_upload_passes_epoch1_and_epoch2_cumulative(two_epochs):
    t = two_epochs
    c1 = _check(t, 1)
    assert c1.status == "PASS", c1.detail
    assert c1.evidence["counts"] == _roles(t.e1) and c1.evidence["cumulative"] == len(t.e1)
    c2 = _check(t, 2)
    assert c2.status == "PASS", c2.detail
    assert c2.evidence["counts"] == _roles(t.e2) and c2.evidence["cumulative"] == len(t.e1) + len(t.e2)
    assert c2.evidence["template_mode"] is True and c2.defect == "#213"
    for s in ("8 raw session log(s)", "reference/ mirrored whole (unscanned)", "no prompt file in the container",
              "nothing under /Users"):
        assert s in c2.detail, s
    # files the evolver wrote under the writable trees never fail it (they are its own)
    lst = copy.deepcopy(t.l2)
    lst["after_upload"] += [["/evo/evolver_workspace/PLAN.md", "F", "a" * 64, 3], ["/evo/staging/leftover/SKILL.md", "F", "b" * 64, 5],
                            ["/evo/work/scratch.txt", "F", "c" * 64, 1], ["/evo/out/evolver.log", "F", "d" * 64, 9],
                            ["/evo/out", "D", "", 0], ["/evo/evolver_workspace", "D", "", 0]]
    assert _check(t, listing=lst).status == "PASS"
    # no recorded path_view is fine (an older receipt); not a daytona receipt / an empty manifest -> SKIP
    assert _check(t, receipt={k: v for k, v in t.rc2.items() if k != "path_view"}).status == "PASS"
    assert _check(t, receipt={**t.rc2, "containment": "sandbox-exec"}).status == "SKIP"
    assert _check(t, receipt={**t.rc2, "upload_manifest": []}).status == "SKIP"


def test_check_stateful_upload_fails_each_rule(two_epochs):
    t = two_epochs
    rd = t.rd

    def fail(needle, **over):
        c = _check(t, **over)
        text = c.detail + " || " + " || ".join(c.evidence.get("problems") or [])
        assert c.status == "FAIL" and needle in text, (needle, c.status, text[:600])

    def LST(mut):
        lst = copy.deepcopy(t.l2)
        mut(lst)
        return lst

    def RC(mut):
        rc = copy.deepcopy(t.rc2)
        mut(rc)
        return rc

    def row(vm_path, role, epoch=2, **kw):
        return {"vm_path": vm_path, "role": role, "source": "generated", "sha256": "a" * 64, "bytes": 1, "epoch": epoch, **kw}
    seed_vm = next(r["vm_path"] for r in t.e2 if r["role"] == "seed")
    # the listing
    fail("not in the manifest", listing=LST(lambda l: l["after_upload"].append(["/evo/traces/epoch_0/extra.jsonl", "F", "a" * 64, 1])))
    fail("not in the manifest", listing=LST(lambda l: l["after_upload"].append(["/data/x/run/ITER1_PROMPT.md", "F", "a" * 64, 1])))
    fail("absent in the container", listing=LST(lambda l: l["after_upload"].pop(0)))
    fail("differ from the manifest", listing=LST(lambda l: l["after_upload"][0].__setitem__(2, "0" * 64)))
    fail("symlink", listing=LST(lambda l: l["after_upload"].append(["/evo/evolver_workspace/l", "L", "/etc", 0])))
    fail("/Users existed", listing=LST(lambda l: l["pre_upload"].__setitem__("users_exists", True)))
    fail("/Users existed", listing=LST(lambda l: l.pop("pre_upload")))
    fail("no container listing", listing={})
    fail("seed rows under /evo/staging not at their sha", listing=LST(lambda l: l["after_upload"].remove(
        next(x for x in l["after_upload"] if x[0] == seed_vm))))
    fail("seed rows under /evo/staging not at their sha", listing=LST(lambda l: [x.__setitem__(2, "0" * 64) for x in l["after_upload"] if x[0] == seed_vm]))
    # the delta
    fail("/Users", receipt=RC(lambda rc: rc["upload_manifest"].append(row("/Users/x/a.md", "traces"))))
    fail("delta rebuilt now", receipt=RC(lambda rc: rc["upload_manifest"].__setitem__(0, dict(rc["upload_manifest"][0], sha256="0" * 64))))
    fail("delta rebuilt now", receipt=RC(lambda rc: rc["upload_manifest"].append(dict(t.e1[0], vm_path="/evo/traces/epoch_2/x.jsonl", epoch=2))))
    fail("delta rebuilt now", receipt=RC(lambda rc: rc["upload_manifest"].pop()))
    fail("another epoch than 2", receipt=RC(lambda rc: rc["upload_manifest"][0].__setitem__("epoch", 1)))
    fail("unknown manifest role", receipt=RC(lambda rc: rc["upload_manifest"].append(row("/evo/work/x", "oracle"))))
    fail("travel once, at epoch 1", receipt=RC(lambda rc: rc["upload_manifest"].append(next(r for r in t.e1 if r["role"] == "policy"))))
    fail("seed row(s) at epoch 1", epoch=1, receipt={**t.rc1, "upload_manifest": t.e1 + [row("/evo/staging/a/SKILL.md", "seed", 1)]})
    # the cumulative manifest
    fail("/evo/skills/epoch_1 rows != skills_v1 file-for-file", receipt=RC(lambda rc: [r.__setitem__("sha256", "0" * 64) for r in rc["upload_manifest_cumulative"] if r["role"] == "skills_snapshot"]))
    fail("state record's uploaded rows", receipt=RC(lambda rc: rc["upload_manifest_cumulative"].pop()))
    fail("no upload_manifest_cumulative", receipt=RC(lambda rc: rc.pop("upload_manifest_cumulative")))
    fail("receipts' upload_manifest concatenated", state=dict(t.st2, uploaded=t.e1 + t.e2 + [t.e2[0]]),
         receipt=RC(lambda rc: rc["upload_manifest_cumulative"].append(t.e2[0])))
    fail("another epoch than the receipt", state=dict(t.st2, uploaded=[dict(r, epoch=2) for r in t.e1] + t.e2),
         receipt=RC(lambda rc: [r.__setitem__("epoch", 2) for r in rc["upload_manifest_cumulative"][:len(t.e1)]]))
    fail("sha equals a forbidden file", receipt=RC(lambda rc: next(r for r in rc["upload_manifest_cumulative"] if r["role"] == "skills_snapshot")
                                                  .__setitem__("sha256", hashlib.sha256(b"GOLDEN 01_01").hexdigest())))
    fail("forbidden name", receipt=RC(lambda rc: rc["upload_manifest_cumulative"].append(row("/evo/skills/epoch_1/x_golden.xlsx", "skills_snapshot"))))
    fail("source bytes changed", receipt=RC(lambda rc: rc["upload_manifest_cumulative"][0].__setitem__("sha256", "0" * 64)))
    fail("/evo/reference rows != reference file-for-file", receipt=RC(lambda rc: rc["upload_manifest_cumulative"].remove(
        next(r for r in rc["upload_manifest_cumulative"] if r["role"] == "reference"))))
    fail("reference row not mirrored", receipt=RC(lambda rc: next(r for r in rc["upload_manifest_cumulative"] if r["vm_path"] == "/evo/reference/README.md")
                                                 .__setitem__("source", str(rd / "reference/dataset.json"))))
    fail("/evo/scores/epoch_0.json: not exactly scores.json at its current sha", receipt=RC(lambda rc: next(
        r for r in rc["upload_manifest_cumulative"] if r["role"] == "scores").__setitem__("sha256", "0" * 64)))
    fail("/evo/traces/epoch_0 rows != evo_e0/traces file-for-file", receipt=RC(lambda rc: rc["upload_manifest_cumulative"].remove(
        next(r for r in rc["upload_manifest_cumulative"] if r["role"] == "traces"))))
    fail("outside the mapped /evo/traces", receipt=RC(lambda rc: rc["upload_manifest_cumulative"].append(dict(t.e1[0], vm_path="/evo/traces/epoch_7/x.jsonl"))))
    fail("the receipt's path_view is not the run's path view", receipt=RC(lambda rc: rc["path_view"].pop()))
    fail("env file does not re-render", receipt=RC(lambda rc: rc["daytona"].__setitem__("vm_env", {"HOME": "/other"})))
    fail("hook rows", receipt=RC(lambda rc: [rc["upload_manifest_cumulative"].remove(r) for r in list(rc["upload_manifest_cumulative"]) if r["role"] == "hook"]))
    fail("policy rows", receipt=RC(lambda rc: [r.__setitem__("sha256", "0" * 64) for r in rc["upload_manifest_cumulative"] if r["role"] == "policy"]))
    fail("travel on the launch command line", receipt=RC(lambda rc: rc["upload_manifest_cumulative"].append(row("/evo/work/MESSAGE_e1.md", "prompt", 1))))
    fail("e2's receipt records no pre-dispatch hash", receipt=RC(lambda rc: rc.__setitem__("evidence_sha256_before", {"/x": "0" * 64})))
    # the state record
    fail("state record", state=dict(t.st2, uploaded=list(t.e1)))
    fail("no state record", state=None)
    # on disk: the material, the library, the earlier receipt, RUNTIME.json, a scanned dir
    tr = rd / "evo_e0/traces/Cat_01_01_s1.jsonl"
    keep_b = tr.read_bytes()
    tr.write_bytes(keep_b + b"\n{}")
    fail("/evo/traces/epoch_0 rows != evo_e0/traces file-for-file")
    tr.write_bytes(keep_b)
    tr = rd / "evo_e1/traces/Cat_01_01_s1.jsonl"
    keep_b = tr.read_bytes()
    tr.write_bytes(keep_b + b"\n{}")
    fail("delta rebuilt now")
    tr.write_bytes(keep_b)
    sk = rd / "skills_v1/alpha/SKILL.md"
    keep = sk.read_text()
    sk.write_text(keep + "\nmore\n")
    fail("skills_v1")
    sk.write_text(keep)
    (rd / "evo_e0/scores.json").rename(rd / "evo_e0/scores.json.aside")
    fail("/evo/scores/epoch_0.json: not exactly scores.json")
    (rd / "evo_e0/scores.json.aside").rename(rd / "evo_e0/scores.json")
    r1 = rd / SL.RECEIPT_FILE.format(epoch=1)
    keep = r1.read_text()
    r1.unlink()
    fail("unreadable")
    r1.write_text(keep)
    rt = rd / "RUNTIME.json"
    keep = rt.read_text()
    d = json.loads(keep)
    d["evolver_session"] = "per_epoch"
    rt.write_text(json.dumps(d))
    fail("stateful session only")
    d["evolver_session"] = "stateful"
    d["prompt_mode"] = "ao"
    rt.write_text(json.dumps(d))
    fail("prompt_mode=template")
    rt.write_text(keep)
    v = rd / "skills_v1/alpha/GRADING_VALIDATION.json"
    v.write_text("{}")
    fail("grading-validation")
    v.unlink()
    # every mutation restored: the check passes again, at both epochs
    assert _check(t).status == "PASS" and _check(t, 1).status == "PASS"
