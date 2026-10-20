"""The STATEFUL evolver session (container layout v2): one container for the whole run.

THIS FILE HAS TWO PARTS:
  * the ctl part (A + B): scripts/daytona/evolver_container_ctl.py's stateful additions, run on macOS against tmp
    trees with chown/lchown/alive monkeypatched as tests/test_evolver_container.py does -- validate_spec's /evo rw
    rule, the plan/render_init of the stateful spec, `_extract(..., incremental=True)`, `_pack_parts`, `_start_proxy`,
    `epoch-begin`, `epoch-end`, `restore`, the argparse registrations, and the FakeMachine's handlers for the same
    contracts (what the host driver's tests run against);
  * the driver part (C): scripts/daytona/evolver_session.py's StatefulEvolverRun on the FakeMachine (two epochs,
    finish, SIGTERM, rebuild, the base64 delivery of the message and the system prompt).
NO root, NO Linux, NO network, NO Daytona, NO model call.
"""
from __future__ import annotations

import inspect
import io
import json
import os
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "daytona"))
sys.path.insert(0, str(ROOT / "scripts"))

import evolver_container_ctl as ctl  # noqa: E402
from src.orchestrator import stateful_layout as SL  # noqa: E402
from tests.test_evolver_container import CNS, FAKE_FACTS, HNS, FakeMachine, plan_of, spec_for  # noqa: E402

RCID = "run-20260927t120000-abc123"                 # the run-scoped cid: CID_RE ok
STAGING, WORKSPACE = SL.VM_STAGING, SL.VM_WORKSPACE  # /evo/staging, /evo/evolver_workspace (was /evo/notes)
RW = [dict(r) for r in SL.STATEFUL_RW]
RO_ROOTS = [SL.traces_dir(0), SL.solver_ws_dir(0), SL.VM_SCORES, SL.VM_REFERENCE]


def stateful_spec(cid=RCID, rw=None) -> dict:
    return spec_for(rw=list(RW) if rw is None else rw, ro_roots=list(RO_ROOTS), cid=cid)


def tgz_bytes(members) -> bytes:
    """[(name, data | None for a symlink, kind)] -> tgz bytes; kind in {"file", "sym", "dir"}."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data, kind in members:
            ti = tarfile.TarInfo(name)
            if kind == "sym":
                ti.type, ti.linkname = tarfile.SYMTYPE, "/etc/passwd"
                tf.addfile(ti)
            elif kind == "dir":
                ti.type = tarfile.DIRTYPE
                tf.addfile(ti)
            else:
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════ A. the ctl on a tmp machine tree
@pytest.fixture
def box(tmp_path, monkeypatch):
    """A prepared (not yet up) stateful container under a tmp BASE: state/{spec,facts,limits}.json, the fs/ tree
    `prepare` lays out (Users, evo/{work,out,staging,evolver_workspace}, etc/claude-code), rw/home, a fake cgroup
    dir. chown/lchown are recorded (never applied), `alive` answers from `live` (pids), jprint output is captured."""
    monkeypatch.setattr(ctl, "BASE", str(tmp_path / "base"))
    monkeypatch.setattr(ctl, "CG_PARENT", str(tmp_path / "cg"))
    monkeypatch.setattr(ctl.time, "sleep", lambda s: None)
    b = SimpleNamespace(chown=[], out=[], live=set())
    monkeypatch.setattr(ctl.os, "chown", lambda p, u, g, *a, **k: b.chown.append(("c", str(p), u, g)))
    monkeypatch.setattr(ctl.os, "lchown", lambda p, u, g, *a, **k: b.chown.append(("l", str(p), u, g)))
    monkeypatch.setattr(ctl, "alive", lambda pid, start=None: int(pid or 0) in b.live)
    monkeypatch.setattr(ctl, "jprint", lambda obj: b.out.append(json.loads(json.dumps(obj, default=str))))
    spec = stateful_spec()
    assert ctl.validate_spec(spec) == []
    P = ctl.paths(RCID)
    for d in (P["state"], P["fs"] + "/Users", P["fs"] + ctl.VM_WORK, P["fs"] + ctl.VM_OUT,
              *(P["fs"] + r["path"] for r in spec["rw"]), P["fs"] + ctl.VM_HOOKS, P["fs"] + ctl.POLICY_DIR,
              P["home"], P["cg"]):
        os.makedirs(d, exist_ok=True)
    pl, lim = plan_of(spec)
    for n, obj in (("spec", spec), ("facts", FAKE_FACTS), ("limits", lim), ("plan", pl)):
        Path(P["state"], f"{n}.json").write_text(json.dumps(obj))
    Path(P["cg"], "cgroup.procs").write_text("100\n101\n")
    Path(P["cg"], "cgroup.freeze").write_text("0\n")
    b.P, b.spec, b.pl, b.lim, b.tmp = P, spec, pl, lim, tmp_path
    return b


def bring_up(b) -> dict:
    """state.json as `up` leaves it: holder 100, init 101, proxy 102 (alive), the proxy listening in the container's
    network namespace."""
    st = {"cid": RCID, "holder_pid": 100, "holder_start": 1, "init_pid": 101, "init_start": 2, "ns": CNS,
          "machine_ns": HNS, "cgroup": {"path": b.P["cg"]}, "proxy_pid": 102, "proxy_start": 3,
          "proxy": {"pid": 102, "listen": "127.0.0.1:3128", "listen_netns": CNS["net"], "machine_netns": HNS["net"]}}
    Path(b.P["state"], "state.json").write_text(json.dumps(st))
    Path(b.P["state"], "proxy.ready").write_text(json.dumps(st["proxy"]))
    b.live |= {100, 101, 102}
    return st


def last(b) -> dict:
    return b.out[-1]


# ── 1. validate_spec: rw below /evo (outside work/out/hooks) OR below /Users; plan + init unchanged in shape
def test_validate_spec_accepts_the_stateful_rw_dirs_and_refuses_the_fixed_trees():
    assert ctl.validate_spec(stateful_spec()) == []
    assert ctl.validate_spec(spec_for()) == []                                   # the per-epoch spec, unchanged
    for bad in ("/evo", ctl.VM_WORK, ctl.VM_OUT, f"{ctl.VM_HOOKS}/x", ctl.VM_HOOKS, "/etc", f"{ctl.VM_WORK}/sub",
                f"{ctl.VM_OUT}/sub", "/Users", "/home/evo", "/opt/x"):
        probs = ctl.validate_spec(stateful_spec(rw=[{"path": bad, "owner": "evo", "mode": "0755"}]))
        assert any("not strictly below /Users or /evo (outside work/out/hooks)" in p for p in probs), (bad, probs)
    # the other rw rules still apply to /evo paths
    probs = ctl.validate_spec(stateful_spec(rw=RW + [{"path": f"{STAGING}/sub", "owner": "evo", "mode": "0755"}]))
    assert any("nests" in p for p in probs)
    probs = ctl.validate_spec(stateful_spec(rw=[{"path": STAGING, "owner": "evo", "mode": "0777"}]))
    assert any("owner/mode" in p for p in probs)
    s = stateful_spec()
    s["readonly_roots"].append("/evo")                                          # /evo contains the rw dirs
    assert any("contains a rw path" in p for p in ctl.validate_spec(s))


def test_plan_binds_the_stateful_rw_dirs_after_evo_out_and_the_top_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(ctl, "BASE", str(tmp_path / "base"))
    spec = stateful_spec()
    pl, lim = plan_of(spec)
    dsts = [m["dst"] for m in pl]
    i_out, i_ws, i_stg = dsts.index(ctl.VM_OUT), dsts.index(WORKSPACE), dsts.index(STAGING)
    assert i_out < i_ws < i_stg                                                 # depth, then name
    c = ctl.cdir(RCID)
    for p in (WORKSPACE, STAGING):
        m = pl[dsts.index(p)]
        assert m["kind"] == "bind" and m["mk"] == "exists" and m["ro"] is False and m["src"] == f"{c}/fs{p}"
    assert ctl.expected_top(pl) == ctl.expected_top(plan_of(spec_for(cid=RCID))[0])
    assert ("/evo", True) in ctl.expected_mount_points(pl) and (WORKSPACE, False) in ctl.expected_mount_points(pl)
    init = ctl.render_init(spec, FAKE_FACTS, lim, pl)
    for p in (WORKSPACE, STAGING):
        assert f'[ -d "$R"{p} ] || {{ echo "EVOCTL_FAIL missing mount point {p}" >&2; exit 3; }}' in init
        assert f'mount -n --bind {c}/fs{p} "$R"{p}' in init
    assert init.index(f'"$R"{ctl.VM_OUT}\n') < init.index(f'"$R"{WORKSPACE}\n') if f'"$R"{WORKSPACE}\n' in init else True
    assert init == ctl.render_init(spec, FAKE_FACTS, lim)                       # re-renders identically


# ── 2. extract --incremental: create-only, ownership on what it wrote only, seed rows to evo
def test_extract_incremental_touches_only_what_it_wrote_and_hands_the_seed_to_evo(box):
    P = box.P
    old = Path(P["fs"], "evo/traces/epoch_0/Template_01_01_s1.jsonl")           # epoch 1's upload, already there
    old.parent.mkdir(parents=True)
    old.write_bytes(b"old")
    Path(P["fs"], "evo/evolver_workspace/mine.md").write_bytes(b"the evolver's own")   # never touched
    members = [("evo/traces/epoch_1/Template_01_01_s1.jsonl", b"new", "file"),
               ("evo/solver_workspaces/epoch_1/Template_01_01_s1/out.xlsx", b"x", "file"),
               ("evo/scores/epoch_1.json", b"[]", "file"),
               ("evo/staging/seed-skill/SKILL.md", b"seed", "file")]
    Path(P["state"], "upload.tgz").write_bytes(tgz_bytes(members))
    assert ctl.cmd_extract(SimpleNamespace(cid=RCID, incremental=True)) == 0
    res = last(box)
    assert res["ok"] is True and res["files"] == 4 and res["incremental"] is True
    assert sorted(res["dirs_created"]) == ["/evo/scores", "/evo/solver_workspaces", "/evo/solver_workspaces/epoch_1",
                                           "/evo/solver_workspaces/epoch_1/Template_01_01_s1", "/evo/staging/seed-skill",
                                           "/evo/traces/epoch_1"]
    assert not Path(P["state"], "upload.tgz").exists()
    fs = P["fs"]
    touched = {p for _k, p, _u, _g in box.chown}
    assert touched == {f"{fs}{d}" for d in res["dirs_created"]} | {f"{fs}/{m}" for m, _d, _k in members}
    owner = {p: (u, g) for _k, p, u, g in box.chown}
    assert owner[f"{fs}/evo/staging/seed-skill"] == owner[f"{fs}/evo/staging/seed-skill/SKILL.md"] == (1000, 1000)
    assert owner[f"{fs}/evo/traces/epoch_1/Template_01_01_s1.jsonl"] == owner[f"{fs}/evo/scores/epoch_1.json"] == (0, 0)
    assert owner[f"{fs}/evo/traces/epoch_1"] == owner[f"{fs}/evo/solver_workspaces"] == (0, 0)
    assert old.read_bytes() == b"old" and Path(fs, "evo/evolver_workspace/mine.md").read_bytes() == b"the evolver's own"
    assert oct(os.stat(f"{fs}/evo/staging/seed-skill").st_mode & 0o777) == "0o755"
    assert oct(os.stat(f"{fs}/evo/staging/seed-skill/SKILL.md").st_mode & 0o777) == "0o644"
    # a second extract of the same member is refused BEFORE anything is written (the other member stays absent)
    box.chown.clear()
    Path(P["state"], "upload.tgz").write_bytes(tgz_bytes([("evo/scores/epoch_2.json", b"e2", "file"),
                                                          ("evo/traces/epoch_1/Template_01_01_s1.jsonl", b"again", "file")]))
    assert ctl.cmd_extract(SimpleNamespace(cid=RCID, incremental=True)) == 2
    assert "already present" in last(box)["problems"][0] and "Template_01_01_s1.jsonl" in last(box)["problems"][0]
    assert not Path(fs, "evo/scores/epoch_2.json").exists() and box.chown == []
    assert Path(fs, "evo/traces/epoch_1/Template_01_01_s1.jsonl").read_bytes() == b"new"
    # a symlink planted by the evolver in its writable dir is never followed by the root-side extraction
    os.symlink(str(box.tmp / "outside"), f"{fs}/evo/staging/planted")
    Path(P["state"], "upload.tgz").write_bytes(tgz_bytes([("evo/staging/planted/SKILL.md", b"x", "file")]))
    assert ctl.cmd_extract(SimpleNamespace(cid=RCID, incremental=True)) == 2
    assert "not a plain directory" in last(box)["problems"][0] and not (box.tmp / "outside").exists()
    os.unlink(f"{fs}/evo/staging/planted")
    # the member rules are the plain extract's
    for bad in [("opt/leg/leg.env", b"x", "file"), ("evo/../etc/passwd", b"x", "file"),
                ("evo/evolver_workspace/l", b"", "sym")]:
        Path(P["state"], "upload.tgz").write_bytes(tgz_bytes([bad]))
        assert ctl.cmd_extract(SimpleNamespace(cid=RCID, incremental=True)) == 2
        assert "refused tar members" in last(box)["problems"][0]


def test_extract_default_path_is_the_per_epoch_one(box):
    """No --incremental (and an argparse namespace without the attribute, as the existing test builds it): the
    whole upload tree is re-owned by root and the rw tops handed back, exactly as before."""
    P, fs = box.P, box.P["fs"]
    old = Path(fs, "evo/traces/epoch_0/Template_01_01_s1.jsonl")
    old.parent.mkdir(parents=True)
    old.write_bytes(b"old")
    Path(P["state"], "upload.tgz").write_bytes(tgz_bytes([("evo/work/PROMPT.md", b"p", "file")]))
    assert ctl.cmd_extract(SimpleNamespace(cid=RCID)) == 0
    assert last(box) == {"ok": True, "files": 1}
    touched = {p for _k, p, _u, _g in box.chown}
    assert str(old) in touched and f"{fs}/evo" in touched                     # the pre-existing entries, re-owned
    handed = [(p, u) for k, p, u, _g in box.chown if k == "c"]
    assert (f"{fs}{STAGING}", 1000) in handed and (f"{fs}{WORKSPACE}", 1000) in handed
    assert (f"{fs}{ctl.VM_WORK}", 1000) in handed and (f"{fs}{ctl.VM_OUT}", 1000) in handed


# ── 3. _pack_parts (the pack's tar loop) and cmd_pack's refusal while up
def test_pack_parts_refuses_a_symlink_component_and_stores_inner_links_as_links(box):
    P, fs = box.P, box.P["fs"]
    outside = box.tmp / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_bytes(b"machine secret")
    Path(fs, "evo/staging/skill-a").mkdir()
    Path(fs, "evo/staging/skill-a/SKILL.md").write_bytes(b"body")
    os.symlink(str(outside), f"{fs}/evo/staging/lnk")
    Path(fs, "evo/evolver_workspace/n.md").write_bytes(b"note")
    os.symlink("/etc/passwd", f"{fs}/evo/evolver_workspace/pw")
    packed, refused = ctl._pack_parts(P, box.pl, RCID, [STAGING, f"{STAGING}/lnk/secret.txt", WORKSPACE,
                                                         SL.traces_dir(0), "/usr/bin"])
    assert packed == [STAGING, WORKSPACE]
    assert [r["part"] for r in refused] == [f"{STAGING}/lnk/secret.txt", SL.traces_dir(0), "/usr/bin"]
    assert "symlink" in refused[0]["why"] and "absent" in refused[1]["why"] and "own binds" in refused[2]["why"]
    with tarfile.open(f"{P['state']}/out.tgz", "r:gz") as tf:
        names = {m.name: m for m in tf.getmembers()}
    assert names["evo/staging/skill-a/SKILL.md"].isfile() and names["evo/evolver_workspace/n.md"].isfile()
    assert names["evo/staging/lnk"].issym() and names["evo/evolver_workspace/pw"].issym()
    assert "evo/staging/lnk/secret.txt" not in names and names["evo/staging"].isdir()
    assert all(m.uid == m.gid == 0 and m.uname == m.gname == "root" for m in names.values())
    assert b"machine secret" not in Path(P["state"], "out.tgz").read_bytes()
    # cmd_pack: the same loop, still refused while the container is up
    bring_up(box)
    assert ctl.cmd_pack(SimpleNamespace(cid=RCID, part=[WORKSPACE])) == 2
    assert "pack before down" in last(box)["problems"][0]
    box.live.clear()
    assert ctl.cmd_pack(SimpleNamespace(cid=RCID, part=[WORKSPACE])) == 0
    assert last(box)["packed"] == [WORKSPACE] and last(box)["refused"] == [] and last(box)["tgz"] == f"{P['state']}/out.tgz"
    assert "_pack_parts(" in inspect.getsource(ctl.cmd_pack) and "_start_proxy(" in inspect.getsource(ctl.cmd_up)


# ── 4. _start_proxy (extracted from `up`)
def test_start_proxy_records_pid_start_and_ready_in_state(box, monkeypatch):
    P = box.P
    spawned = []

    class FakePopen:
        pid = 777

        def __init__(self, argv, **kw):
            spawned.append(argv)
            assert kw["start_new_session"] is True and kw["close_fds"] is True
            Path(P["state"], "proxy.ready").write_text(json.dumps({"pid": 777, "listen_netns": CNS["net"]}))

        def poll(self):
            return None
    monkeypatch.setattr(ctl.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(ctl, "pid_start", lambda pid: 4321)
    state = {"cid": RCID, "init_pid": 101, "ns": CNS}
    ready = ctl._start_proxy(P, RCID, 101, state)
    assert ready == {"pid": 777, "listen_netns": CNS["net"]}
    assert spawned[0][-4:] == ["--cid", RCID, "--init-pid", "101"] and spawned[0][-5] == "proxy"
    assert state["proxy_pid"] == 777 and state["proxy_start"] == 4321 and state["proxy"] == ready
    assert json.loads(Path(P["state"], "state.json").read_text()) == state


# ── 5. epoch-begin
def test_epoch_begin_resets_out_clears_the_staging_without_following_links_and_keeps_the_workspace(box):
    fs = box.P["fs"]
    bring_up(box)
    outside = box.tmp / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_bytes(b"machine file")
    for rel, data in (("evo/staging/skill-a/SKILL.md", b"v1"), ("evo/staging/skill-a/sub/x.py", b"x"),
                      ("evo/evolver_workspace/e1.md", b"word: ALPHA"), ("evo/out/rc", b"0\n"), ("evo/out/pid", b"55\n"),
                      ("evo/out/evolver.log", b"log"), ("evo/out/other.txt", b"kept")):
        Path(fs, rel).parent.mkdir(parents=True, exist_ok=True)
        Path(fs, rel).write_bytes(data)
    os.symlink(str(outside), f"{fs}/evo/staging/lnk-dir")
    os.symlink("/etc/passwd", f"{fs}/evo/staging/skill-a/lnk-file")
    assert ctl.cmd_epoch_begin(SimpleNamespace(cid=RCID, clear=[STAGING])) == 0
    res = last(box)
    assert res["ok"] is True and res["init_alive"] is True and res["thawed"] is False and res["proxy_restarted"] is False
    assert res["proxy"]["listen_netns"] == CNS["net"]
    assert res["out_reset"] == ["pid", "rc", "evolver.log"]                    # OUT_RESET_NAMES order; absent ones skipped
    assert res["cleared"] == {STAGING: 6}                                       # SKILL.md, lnk-file, x.py, sub, skill-a, lnk-dir
    assert os.path.isdir(f"{fs}{STAGING}") and os.listdir(f"{fs}{STAGING}") == []
    assert (outside / "keep.txt").read_bytes() == b"machine file"                # the link was removed, never followed
    assert Path(fs, "evo/evolver_workspace/e1.md").read_bytes() == b"word: ALPHA"   # the workspace is never cleared
    assert Path(fs, "evo/out/other.txt").read_bytes() == b"kept" and not Path(fs, "evo/out/rc").exists()
    assert ("c", f"{fs}{STAGING}", 1000, 1000) in box.chown
    assert oct(os.stat(f"{fs}{STAGING}").st_mode & 0o777) == "0o755"
    # no --clear at all: nothing cleared, the out still reset (idempotent on an empty out)
    assert ctl.cmd_epoch_begin(SimpleNamespace(cid=RCID, clear=[])) == 0
    assert last(box)["cleared"] == {} and last(box)["out_reset"] == []


def test_epoch_begin_refusals_thaw_and_proxy_restart(box, monkeypatch):
    P, fs = box.P, box.P["fs"]
    # not up
    Path(P["state"], "state.json").write_text(json.dumps({"cid": RCID, "init_pid": 101, "init_start": 2}))
    assert ctl.cmd_epoch_begin(SimpleNamespace(cid=RCID, clear=[])) == 2
    assert last(box) == {"ok": False, "problems": ["container not up"]}
    st = bring_up(box)
    # a session still running: a pid in the cgroup other than holder/init
    Path(P["cg"], "cgroup.procs").write_text("100\n101\n555\n")
    assert ctl.cmd_epoch_begin(SimpleNamespace(cid=RCID, clear=[])) == 2
    assert "a session is still running in the container; stop-session first" in last(box)["problems"][0]
    assert last(box)["procs"] == [555]
    Path(P["cg"], "cgroup.procs").write_text("100\n101\n")
    # frozen -> thawed
    Path(P["cg"], "cgroup.freeze").write_text("1\n")
    assert ctl.cmd_epoch_begin(SimpleNamespace(cid=RCID, clear=[])) == 0
    assert last(box)["thawed"] is True and Path(P["cg"], "cgroup.freeze").read_text() == "0"
    # a symlink among the /evo/out names is refused
    os.symlink("/etc/passwd", f"{fs}/evo/out/rc")
    assert ctl.cmd_epoch_begin(SimpleNamespace(cid=RCID, clear=[])) == 2
    assert last(box)["problems"] == [f"symlink in {ctl.VM_OUT}: rc"]
    os.unlink(f"{fs}/evo/out/rc")
    # --clear of anything but an evo-owned rw path of the spec
    for bad in (ctl.VM_WORK, SL.traces_dir(0), f"{STAGING}/x", "/Users"):
        assert ctl.cmd_epoch_begin(SimpleNamespace(cid=RCID, clear=[bad])) == 2
        assert "not an evo-owned rw path of the spec" in last(box)["problems"][0]
    # the proxy died between epochs: restarted through _start_proxy, the stale ready record unlinked first
    box.live.discard(102)
    calls = []

    def fake_start(Pp, cid, init_pid, state):
        assert not Path(Pp["state"], "proxy.ready").exists(), "the dead proxy's ready record must be gone"
        calls.append((cid, init_pid))
        state.update(proxy_pid=103, proxy_start=9, proxy={"pid": 103, "listen_netns": state["ns"]["net"]})
        ctl._jsave(f"{Pp['state']}/state.json", state)
        return state["proxy"]
    monkeypatch.setattr(ctl, "_start_proxy", fake_start)
    assert ctl.cmd_epoch_begin(SimpleNamespace(cid=RCID, clear=[STAGING])) == 0
    assert calls == [(RCID, 101)] and last(box)["proxy_restarted"] is True and last(box)["proxy"]["pid"] == 103
    assert json.loads(Path(P["state"], "state.json").read_text())["proxy_pid"] == 103
    # a restarted proxy that listens elsewhere is a refusal (rc 3)
    box.live.discard(103)

    def wrong_ns(Pp, cid, init_pid, state):
        state.update(proxy_pid=104, proxy={"pid": 104, "listen_netns": HNS["net"]})
        return state["proxy"]
    monkeypatch.setattr(ctl, "_start_proxy", wrong_ns)
    assert ctl.cmd_epoch_begin(SimpleNamespace(cid=RCID, clear=[])) == 3
    assert "egress proxy not listening in the container's network namespace" in last(box)["problems"][0]
    assert st["cid"] == RCID


# ── 6. epoch-end
def test_epoch_end_packs_while_frozen_scans_thaws_and_keeps_the_container(box, monkeypatch):
    P, fs = box.P, box.P["fs"]
    bring_up(box)
    for rel, data in (("evo/staging/skill-a/SKILL.md", b"v2"), ("evo/evolver_workspace/e1.md", b"word: ALPHA\nturn: 2\n"),
                      ("evo/out/evolver.log", b"log"), ("evo/out/rc", b"0\n")):
        Path(fs, rel).parent.mkdir(parents=True, exist_ok=True)
        Path(fs, rel).write_bytes(data)
    sess = Path(P["home"], ".claude/projects/-evo-work/sid.jsonl")
    sess.parent.mkdir(parents=True)
    sess.write_bytes(b'{"type":"system"}\n')
    Path(P["state"], "egress.jsonl").write_text(json.dumps({"host": "api.anthropic.com", "allowed": True}) + "\n"
                                                 + json.dumps({"host": "openrouter.ai", "allowed": False}) + "\n")
    seen_frozen = []
    real_pack = ctl._pack_parts

    def pack_while_frozen(*a, **k):
        seen_frozen.append(Path(P["cg"], "cgroup.freeze").read_text())
        return real_pack(*a, **k)
    monkeypatch.setattr(ctl, "_pack_parts", pack_while_frozen)
    parts = [STAGING, WORKSPACE, "/home/evo/.claude/projects", ctl.VM_OUT, SL.traces_dir(0)]
    assert ctl.cmd_epoch_end(SimpleNamespace(cid=RCID, part=parts)) == 0
    res = last(box)
    assert set(res) >= {"ok", "cid", "kept", "frozen", "thawed", "init_alive", "tmpfs_scan", "scan", "egress_log",
                        "tgz", "bytes", "packed", "refused"}
    assert res["ok"] is True and res["cid"] == RCID and res["kept"] is True
    assert res["frozen"] is True and res["thawed"] is True and res["init_alive"] is True
    assert seen_frozen == ["1"] and Path(P["cg"], "cgroup.freeze").read_text() == "0"
    assert res["packed"] == parts[:4] and [r["part"] for r in res["refused"]] == [SL.traces_dir(0)]
    assert res["tmpfs_scan"] == {"files": 0, "hits": []} and all(v["hits"] == [] for v in res["scan"].values())
    assert res["egress_log"] == {"requests": 2, "by_host": {"api.anthropic.com:allowed": 1, "openrouter.ai:refused": 1}}
    assert res["tgz"] == f"{P['state']}/out.tgz" and res["bytes"] == os.path.getsize(res["tgz"])
    assert res["session_procs"] == []
    with tarfile.open(res["tgz"], "r:gz") as tf:
        names = {m.name for m in tf.getmembers()}
    assert {"evo/staging/skill-a/SKILL.md", "evo/evolver_workspace/e1.md", "home/evo/.claude/projects/-evo-work/sid.jsonl",
            "evo/out/evolver.log", "evo/out/rc", "evo/evolver_workspace", "evo/staging"} <= names
    assert json.loads(Path(P["state"], "epoch_end.json").read_text()) == res
    assert box.live == {100, 101, 102}                                          # nothing was killed
    # the writable state is scanned: a token prefix planted in the workspace is named (never its bytes)
    Path(fs, "evo/evolver_workspace/leak.md").write_bytes(b"x " + ctl.TOKEN_PREFIX + b"LEAKLEAKLEAK y")
    assert ctl.cmd_epoch_end(SimpleNamespace(cid=RCID, part=[WORKSPACE])) == 0
    assert last(box)["scan"]["fs"]["hits"] == ["fs/evo/evolver_workspace/leak.md:token-prefix"]
    assert b"LEAKLEAK" not in json.dumps(last(box)).encode()


def test_epoch_end_refuses_when_not_up_and_thaws_even_when_the_pack_fails(box, monkeypatch):
    P = box.P
    Path(P["state"], "state.json").write_text(json.dumps({"cid": RCID, "init_pid": 101, "init_start": 2}))
    assert ctl.cmd_epoch_end(SimpleNamespace(cid=RCID, part=[WORKSPACE])) == 2
    assert last(box)["problems"] == ["container not up"] and last(box)["kept"] is True
    bring_up(box)
    Path(P["cg"], "cgroup.procs").write_text("100\n101\n556\n")                 # a straggler: reported, not fatal

    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(ctl, "_pack_parts", boom)
    assert ctl.cmd_epoch_end(SimpleNamespace(cid=RCID, part=[WORKSPACE])) == 1
    res = last(box)
    assert res["ok"] is False and res["problems"] == ["epoch-end: disk full"]
    assert res["frozen"] is True and res["thawed"] is True and res["init_alive"] is True
    assert res["session_procs"] == [556] and Path(P["cg"], "cgroup.freeze").read_text() == "0"
    # init gone after the thaw -> not ok
    monkeypatch.setattr(ctl, "_pack_parts", lambda P_, pl, cid, parts: (parts, []))
    Path(P["state"], "out.tgz").write_bytes(b"")
    answers = iter([True, False])                                               # alive at the start, gone after the thaw
    monkeypatch.setattr(ctl, "_init_alive", lambda st: next(answers))
    assert ctl.cmd_epoch_end(SimpleNamespace(cid=RCID, part=[WORKSPACE])) == 1
    assert last(box)["init_alive"] is False and last(box)["ok"] is False
    # no cgroup.freeze on the machine: recorded as the failure text, never True
    os.unlink(f"{P['cg']}/cgroup.freeze")
    monkeypatch.setattr(ctl, "_init_alive", lambda st: True)
    assert ctl.cmd_epoch_end(SimpleNamespace(cid=RCID, part=[WORKSPACE])) == 1
    assert last(box)["frozen"] == "failed: no cgroup.freeze" and last(box)["ok"] is False


def test_freeze_waits_for_the_cgroup_events_to_settle(box):
    cg = box.P["cg"]
    Path(cg, "cgroup.events").write_text("populated 1\nfrozen 1\n")
    assert ctl._freeze(cg, True) is True and Path(cg, "cgroup.freeze").read_text() == "1"
    Path(cg, "cgroup.events").write_text("populated 1\nfrozen 0\n")
    assert ctl._freeze(cg, False) is True and Path(cg, "cgroup.freeze").read_text() == "0"
    assert ctl._freeze(str(box.tmp / "nope"), True) == "failed: no cgroup.freeze"


# ── 7. restore (before `up`)
def test_restore_writes_the_snapshot_into_home_and_the_workspace_create_only_as_evo(box):
    P, fs = box.P, box.P["fs"]
    members = [("home/evo/.claude/projects/-evo-work/sid.jsonl", b'{"a":1}\n', "file"),
               ("home/evo/.claude/projects/-tmp-q_evo/q.jsonl", b"{}\n", "file"),
               ("evo/evolver_workspace/e1.md", b"word: ALPHA\n", "file"),
               ("evo/evolver_workspace/sub/deep.md", b"d", "file")]
    Path(P["state"], "restore.tgz").write_bytes(tgz_bytes(members))
    assert ctl.cmd_restore(SimpleNamespace(cid=RCID)) == 0
    assert last(box) == {"ok": True, "files": 4, "dirs": 5}                    # .claude, projects, -evo-work, -tmp-q_evo, sub
    assert Path(P["home"], ".claude/projects/-evo-work/sid.jsonl").read_bytes() == b'{"a":1}\n'
    assert Path(fs, "evo/evolver_workspace/e1.md").read_bytes() == b"word: ALPHA\n"
    assert Path(fs, "evo/evolver_workspace/sub/deep.md").read_bytes() == b"d"
    assert not Path(P["state"], "restore.tgz").exists()
    owned = {p: (u, g) for _k, p, u, g in box.chown}
    for rel in (".claude", ".claude/projects", ".claude/projects/-evo-work", ".claude/projects/-evo-work/sid.jsonl"):
        assert owned[str(Path(P["home"], rel))] == (1000, 1000)
    assert owned[f"{fs}/evo/evolver_workspace/sub"] == owned[f"{fs}/evo/evolver_workspace/e1.md"] == (1000, 1000)
    assert f"{fs}/evo/evolver_workspace" not in owned and P["home"] not in owned   # pre-existing dirs untouched
    assert oct(os.stat(f"{fs}/evo/evolver_workspace/sub").st_mode & 0o777) == "0o755"
    assert oct(os.stat(f"{fs}/evo/evolver_workspace/e1.md").st_mode & 0o777) == "0o644"
    # create-only: the same member again is refused before anything is written
    Path(P["state"], "restore.tgz").write_bytes(tgz_bytes([("evo/evolver_workspace/new.md", b"n", "file"), members[2]]))
    assert ctl.cmd_restore(SimpleNamespace(cid=RCID)) == 2
    assert "already present" in last(box)["problems"][0] and not Path(fs, "evo/evolver_workspace/new.md").exists()


def test_restore_refusals(box):
    P, fs = box.P, box.P["fs"]
    for bad in [("evo/staging/s/SKILL.md", b"x", "file"), ("evo/evolver_workspace/../work/x", b"x", "file"),
                ("/evo/evolver_workspace/x", b"x", "file"), ("home/evo/.bashrc", b"x", "file"), ("Users/x", b"x", "file"),
                ("evo/evolver_workspace/l", b"", "sym"), ("evo/evolver_workspace/d", b"", "dir"),
                ("home/evo/.claude/settings.json", b"{}", "file"),
                ("evo/notes/x.md", b"x", "file")]:                              # the older notes dir is no root
        Path(P["state"], "restore.tgz").write_bytes(tgz_bytes([bad, ("evo/evolver_workspace/ok.md", b"ok", "file")]))
        assert ctl.cmd_restore(SimpleNamespace(cid=RCID)) == 2, bad
        assert "refused tar members" in last(box)["problems"][0] and not Path(fs, "evo/evolver_workspace/ok.md").exists()
    # a symlink planted where a restored dir must go is never followed
    os.symlink(str(box.tmp / "outside"), f"{fs}/evo/evolver_workspace/sub")
    Path(P["state"], "restore.tgz").write_bytes(tgz_bytes([("evo/evolver_workspace/sub/x.md", b"x", "file")]))
    assert ctl.cmd_restore(SimpleNamespace(cid=RCID)) == 2
    assert "not a plain directory" in last(box)["problems"][0] and not (box.tmp / "outside").exists()
    os.unlink(f"{fs}/evo/evolver_workspace/sub")
    # the workspace without an evo rw grant for /evo/evolver_workspace
    spec = stateful_spec(rw=[RW[0]])
    Path(P["state"], "spec.json").write_text(json.dumps(spec))
    Path(P["state"], "restore.tgz").write_bytes(tgz_bytes([("evo/evolver_workspace/x.md", b"x", "file")]))
    assert ctl.cmd_restore(SimpleNamespace(cid=RCID)) == 2
    assert f"{WORKSPACE} is not an evo-owned rw path" in last(box)["problems"][0]
    Path(P["state"], "spec.json").write_text(json.dumps(box.spec))
    # only before up
    bring_up(box)
    Path(P["state"], "restore.tgz").write_bytes(tgz_bytes([("evo/evolver_workspace/x.md", b"x", "file")]))
    assert ctl.cmd_restore(SimpleNamespace(cid=RCID)) == 2
    assert "already up" in last(box)["problems"][0] and not Path(fs, "evo/evolver_workspace/x.md").exists()
    os.unlink(f"{P['state']}/state.json")
    # a missing tgz is a refusal, not a traceback
    os.unlink(f"{P['state']}/restore.tgz")
    assert ctl.cmd_restore(SimpleNamespace(cid=RCID)) == 2 and last(box)["problems"][0].startswith("restore:")


# ── 8. main(): the registrations
def test_main_registers_the_three_subcommands_and_the_incremental_flag(monkeypatch):
    seen = {}
    for name in ("cmd_epoch_begin", "cmd_epoch_end", "cmd_restore", "cmd_extract"):
        monkeypatch.setattr(ctl, name, (lambda n: lambda a: seen.__setitem__(n, a) or 0)(name))
    assert ctl.main(["epoch-begin", "--cid", RCID, "--clear", STAGING, "--clear", WORKSPACE]) == 0
    assert seen["cmd_epoch_begin"].cid == RCID and seen["cmd_epoch_begin"].clear == [STAGING, WORKSPACE]
    assert ctl.main(["epoch-begin", "--cid", RCID]) == 0 and seen["cmd_epoch_begin"].clear == []
    assert ctl.main(["epoch-end", "--cid", RCID, "--part", STAGING, "--part", WORKSPACE]) == 0
    assert seen["cmd_epoch_end"].part == [STAGING, WORKSPACE]
    with pytest.raises(SystemExit):
        ctl.main(["epoch-end", "--cid", RCID])                                  # --part required
    assert ctl.main(["restore", "--cid", RCID]) == 0 and seen["cmd_restore"].cid == RCID
    assert ctl.main(["extract", "--cid", RCID, "--incremental"]) == 0 and seen["cmd_extract"].incremental is True
    assert ctl.main(["extract", "--cid", RCID]) == 0 and seen["cmd_extract"].incremental is False
    for name in ("epoch-begin", "epoch-end", "restore"):
        with pytest.raises(SystemExit):
            ctl.main([name])                                                    # --cid required
    assert "epoch-begin --cid C [--clear P]..." in ctl.__doc__ and "restore --cid C" in ctl.__doc__


# ══════════════════════════════════════════════════════════════════ B. the FakeMachine's handlers (the contracts)
def test_fake_machine_plays_the_stateful_contracts():
    """What the driver tests run against; the real ctl above answers the same shapes."""
    import base64
    fm = FakeMachine()
    C = f"{ctl.PY} {ctl.CTL}"
    h = lambda c: fm.handle(c, {})                                              # noqa: E731
    j = lambda c: json.loads(h(c)[1])                                           # noqa: E731
    assert j(f"{C} list") == {"containers": [], "cgroups": []}
    spec = stateful_spec()
    b64 = base64.b64encode(json.dumps(spec).encode()).decode()
    assert j(f"{C} prepare --spec-b64 {b64}")["ok"] is True
    cid, cdir = RCID, ctl.cdir(RCID)
    assert j(f"{C} list") == {"containers": [cid], "cgroups": []}
    st = j(f"{C} status --cid {cid}")
    assert st["exists"] is True and st["init_alive"] is False and st["proxy_alive"] is False
    # restore before up: transcripts + the workspace, create-only; the older notes dir is no root
    fm.files[f"{cdir}/state/restore.tgz"] = tgz_bytes([("home/evo/.claude/projects/-evo-work/s.jsonl", b"a", "file"),
                                                       ("evo/evolver_workspace/e1.md", b"n", "file")])
    assert j(f"{C} restore --cid {cid}") == {"ok": True, "files": 2, "dirs": 3}
    assert fm.cfs[f"{WORKSPACE}/e1.md"] == b"n" and fm.cfs["/home/evo/.claude/projects/-evo-work/s.jsonl"] == b"a"
    fm.files[f"{cdir}/state/restore.tgz"] = tgz_bytes([("evo/evolver_workspace/e1.md", b"n", "file")])
    code, out = h(f"{C} restore --cid {cid}")
    assert code == 2 and "already present" in out
    for member in ("evo/staging/x", "evo/notes/e1.md"):
        fm.files[f"{cdir}/state/restore.tgz"] = tgz_bytes([(member, b"n", "file")])
        code, out = h(f"{C} restore --cid {cid}")
        assert code == 2 and "refused tar members" in out
    # epoch-begin needs the container up
    code, out = h(f"{C} epoch-begin --cid {cid} --clear {STAGING}")
    assert code == 2 and "container not up" in out
    assert j(f"{C} up --cid {cid}")["ok"] is True
    assert j(f"{C} status --cid {cid}")["init_alive"] is True and j(f"{C} list")["cgroups"] == [cid]
    code, out = h(f"{C} restore --cid {cid}")
    assert code == 2 and "already up" in out
    # a crashed driver's live session blocks epoch-begin until stop-session
    fm.cfs[f"{ctl.VM_OUT}/evolver.log"] = b"old log"
    fm.cfs[f"{STAGING}/table-audit/SKILL.md"] = b"v1"
    fm.session_live = True
    code, out = h(f"{C} epoch-begin --cid {cid} --clear {STAGING}")
    assert code == 2 and "stop-session first" in out
    assert j(f"{C} stop-session --cid {cid}")["ok"] is True and fm.cfs[f"{ctl.VM_OUT}/rc"] == b"124\n"
    res = j(f"{C} epoch-begin --cid {cid} --clear {STAGING}")
    assert res["ok"] is True and res["out_reset"] == ["rc", "evolver.log"] and res["cleared"] == {STAGING: 1}
    assert res["proxy"]["listen_netns"] == CNS["net"] and res["thawed"] is False and res["proxy_restarted"] is False
    assert not any(k.startswith(STAGING + "/") for k in fm.cfs) and f"{WORKSPACE}/e1.md" in fm.cfs
    code, out = h(f"{C} epoch-begin --cid {cid} --clear {ctl.VM_WORK}")
    assert code == 2 and "not an evo-owned rw path" in out
    # extract --incremental: create-only
    fm.files[f"{cdir}/state/upload.tgz"] = tgz_bytes([("evo/traces/epoch_1/Template_01_01_s1.jsonl", b"i", "file"),
                                                      ("evo/staging/seed/SKILL.md", b"seed", "file")])
    res = j(f"{C} extract --cid {cid} --incremental")
    assert res == {"ok": True, "files": 2, "incremental": True,
                   "dirs_created": ["/evo/staging/seed", "/evo/traces", "/evo/traces/epoch_1"]}
    fm.files[f"{cdir}/state/upload.tgz"] = tgz_bytes([("evo/traces/epoch_1/Template_01_01_s1.jsonl", b"i", "file")])
    code, out = h(f"{C} extract --cid {cid} --incremental")
    assert code == 2 and "already present" in out
    # epoch-end: packed with dir entries, no down
    fm.cfs[f"{ctl.VM_OUT}/evolver.log"] = b"new log"
    res = j(f"{C} epoch-end --cid {cid} --part {STAGING} --part {WORKSPACE} --part /home/evo/.claude/projects "
            f"--part {ctl.VM_OUT} --part /evo/traces/epoch_9")
    assert res["ok"] is True and res["kept"] is True and res["frozen"] is True and res["thawed"] is True
    assert res["packed"] == [STAGING, WORKSPACE, "/home/evo/.claude/projects", ctl.VM_OUT]
    assert [r["part"] for r in res["refused"]] == ["/evo/traces/epoch_9"]
    with tarfile.open(fileobj=io.BytesIO(fm.files[f"{cdir}/state/out.tgz"]), mode="r:gz") as tf:
        mem = {m.name: m for m in tf.getmembers()}
    assert mem["evo/evolver_workspace"].isdir() and mem["evo/evolver_workspace/e1.md"].isfile()
    assert mem["evo/staging/seed/SKILL.md"].isfile() and mem["evo/out/evolver.log"].isfile()
    assert mem["home/evo/.claude/projects/-evo-work/s.jsonl"].isfile()
    assert j(f"{C} status --cid {cid}")["init_alive"] is True                   # still up
    # the teardown: down --wipe, then the machine lists nothing
    assert j(f"{C} down --cid {cid} --wipe")["wiped"] is True
    assert j(f"{C} list") == {"containers": [], "cgroups": []}
    st = j(f"{C} status --cid {cid}")
    assert st["exists"] is False and st["init_alive"] is False
    assert f"{WORKSPACE}/e1.md" in fm.cfs                                       # kept for inspection after the wipe


# ══════════════════════════════════════════════════════════════════ C. the host driver (layout v2)
# scripts/daytona/evolver_session.py's StatefulEvolverRun on the FakeMachine: two epochs in ONE container and ONE
# claude session, `finish`, a SIGTERM mid-launch, a replaced pool machine (rebuild from the snapshot); the routing
# and cleanup guard of evolver_container.py; the base64 delivery of the message and the system prompt; the two
# seams of tau3_evolver_daytona.py. The FakeMachine side of the ctl contracts is part B above; StatefulFakeMachine
# adds the SESSION side.
import base64  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

import daytona_lib_evo as dle  # noqa: E402
import evolver_container as EC  # noqa: E402
import evolver_session as ES  # noqa: E402
import ssb_leg_pool as P  # noqa: E402
import tau3_evolver_daytona as T  # noqa: E402
from src.orchestrator import claude_pins  # noqa: E402
from src.orchestrator import daytona_evolver as de  # noqa: E402
from src.orchestrator import evo_layout as EL  # noqa: E402
from src.orchestrator import pool_container_evolver as pce  # noqa: E402
from tests.test_evolver_container import TOKEN, FakePoolHost, machines_json, pool_row  # noqa: E402
from tests.test_train_references import _fake  # noqa: E402

C = f"{ctl.PY} {ctl.CTL}"
LO = "LibreOffice 25.2.3.2 520(Build:2)"
SYSTEM_TEXT = ("# Evolver\n\nYou write skill sets into /evo/staging/ and keep your own files in "
               "/evo/evolver_workspace/.\n")
MESSAGE = ("Please analyze the solver rollouts (epoch {prev}'s are the newest; earlier epochs are there too if you "
           "want to compare), diagnose the pitfalls that may be fixed by a dedicated skill, and compose "
           "`skills_v{N}` under `/evo/staging/`.\n")
TASKS = ("Template_01_01", "Template_01_02")        # the fake split's train ids, file-safe (evo_layout.safe_id)


class StatefulFakeMachine(FakeMachine):
    """The FakeMachine plus the SESSION side of the contracts: the played claude honours `--session-id` / `--resume`
    -- ONE transcript at /home/evo/.claude/projects/-evo-work/<sid>.jsonl that GROWS every turn (e1's bytes are a
    prefix of e2's), a log naming the sid and the turn, a file per turn in /evo/evolver_workspace (never cleared),
    the staging skill rewritten (body v1, v2, ...). The member list of every `extract --incremental` and `restore`
    is kept (the ctl consumes the tgz), `on_play` runs while the session is "live", `crash` plays a SIGTERM
    mid-launch."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.turns, self.launches, self.extracts, self.restores = 0, [], [], []
        self.on_play = None
        self.crash = None

    def _members(self, remote: str) -> list:
        with tarfile.open(fileobj=io.BytesIO(self.files[remote]), mode="r:gz") as tf:
            return sorted(m.name for m in tf.getmembers())

    def handle(self, cmd: str, env: dict) -> tuple:
        cid = self.spec["cid"] if self.spec else None
        if cid and cmd == f"{C} extract --cid {cid} --incremental":
            self.extracts.append(self._members(f"{ctl.cdir(cid)}/state/upload.tgz"))
        elif cid and cmd == f"{C} restore --cid {cid}":
            self.restores.append(self._members(f"{ctl.cdir(cid)}/state/restore.tgz"))
        return super().handle(cmd, env)

    def play_session(self):
        if self.on_play:
            self.on_play()
        cmd = self.execs[-1][0]
        m = re.search(r"--(session-id|resume) ([0-9a-f-]{36})", cmd)
        assert m, cmd[:300]
        flag, sid = m.group(1), m.group(2)
        self.turns += 1
        self.launches.append({"flag": flag, "sid": sid, "cmd": cmd})
        lines = [{"type": "system", "subtype": "init", "apiKeySource": "none", "mcp_servers": [],
                  "tools": ["Bash", "Read", "Write"], "claude_code_version": "2.1.281", "model": "claude-opus-5-5",
                  "session_id": sid},
                 {"type": "assistant", "message": {"model": "claude-opus-5-5", "content": [
                     {"type": "tool_use", "name": "Bash",
                      "input": {"command": f"ls /evo/traces/epoch_{self.turns - 1}"}}]}},
                 {"type": "result", "subtype": "success", "is_error": False, "result": f"turn {self.turns} done"}]
        body = "".join(json.dumps(x) + "\n" for x in lines).encode()
        self.cfs[f"{ctl.VM_OUT}/evolver.log"] = body
        self.cfs[f"{ctl.VM_OUT}/evolver.stderr"] = b""
        self.cfs[f"{ctl.VM_OUT}/rc"] = b"0\n"
        tp = f"{de.VM_SESSION_DIR}/{sid}.jsonl"
        self.cfs[tp] = self.cfs.get(tp, b"") + body                          # the ONE transcript grows
        self.cfs[f"{WORKSPACE}/turn_{self.turns}.md"] = f"turn {self.turns}: workspace kept\n".encode()
        self.cfs[f"{STAGING}/table-audit/SKILL.md"] = \
            f"---\nname: table-audit\ndescription: d\n---\nbody v{self.turns}\n".encode()
        if self.crash is not None:
            exc, self.crash = self.crash, None
            raise exc


def evo_epoch(rd: Path, K: int) -> Path:
    """<run>/evo_e{K} as evo_layout.build writes it: one raw session log and one solver workspace per rollout,
    scores.json with task id / sample / score only."""
    d = EL.epoch_dir(rd, K)
    (d / EL.TRACES).mkdir(parents=True)
    rows = []
    for i, t in enumerate(TASKS):
        u = EL.unit_of(t, 1)
        (d / EL.TRACES / f"{u}.jsonl").write_text(json.dumps({"type": "result", "epoch": K, "unit": u}) + "\n")
        (d / EL.WORKSPACES / u).mkdir(parents=True)
        (d / EL.WORKSPACES / u / "output.xlsx").write_bytes(f"OUTPUT e{K} {u}".encode())
        rows.append({"task_id": t, "sample": 1, "score": 1.0 if i == K else 0.0})
    (d / EL.SCORES_JSON).write_text(json.dumps(rows, indent=1) + "\n")
    return d


@pytest.fixture
def sess(tmp_path, monkeypatch):
    """A two-epoch stateful run on a tmp tree (STATE_ROOT patched: nothing under .cache/evolver_state): evo_e0 and
    evo_e1 (the base leg's and skills_v1's rollouts), skills_v1, the adapter's reference/, RUNTIME.json, the epoch-1
    system prompt and the two epoch messages; its pool (a FakePoolHost publishing machine.json with the pool's own
    code), one StatefulFakeMachine, a tmp ledger, the env the loop exports (EVOSKILL_EVOLVER_SESSION, the system
    prompt, EVOSKILL_CLAUDE_EFFORT=default, the split of a fake dataset)."""
    f = _fake(tmp_path / "data", monkeypatch)                    # EVOSKILL_SSBV2_DATA -> the fake dataset; split.json
    rd = tmp_path / "run"
    rd.mkdir()
    for K in (0, 1):
        evo_epoch(rd, K)
    (rd / "skills_v1" / "table-audit").mkdir(parents=True)
    (rd / "skills_v1" / "table-audit" / "SKILL.md").write_text("---\nname: table-audit\ndescription: d\n---\nbody v1\n")
    ref = rd / SL.REFERENCE_DIR_NAME
    ref.mkdir()
    (ref / "README.md").write_text("what the adapter handed over (never scanned by the loop)\n")
    (ref / "grader.py").write_text("def grade(out, ref):\n    return out == ref\n")
    tok = tmp_path / "fake_token"
    tok.write_text(TOKEN)
    os.chmod(tok, 0o600)
    cfg = {"placement": "pool_container", "oauth_token_file": str(tok), "require_libreoffice": LO}
    rt = {"evolver_session": "stateful", "prompt_mode": "template", "evolver_host": "daytona",
          "evolver_evidence": "all_epochs", "evolver_evidence_layout": "evo", "claude": {"effort": "default"},
          "daytona_leg": {"store_root": str(tmp_path / "store"), "pool": True, "pool_ttl_minutes": 180},
          "evolver_daytona": cfg}
    (rd / "RUNTIME.json").write_text(json.dumps(rt))
    (rd / SL.SYSTEM_PROMPT_FILE).write_text(SYSTEM_TEXT)
    for k in (1, 2):
        (rd / f"ITER{k}_PROMPT.md").write_text(MESSAGE.format(N=k, prev=k - 1))
    monkeypatch.setenv("EVOSKILL_TAU3_EVOLVER_SPLIT", str(f.split))
    monkeypatch.setenv(SL.ENV_SESSION, SL.STATEFUL)
    monkeypatch.setenv(SL.ENV_SYSTEM_PROMPT, str(rd / SL.SYSTEM_PROMPT_FILE))
    monkeypatch.setenv(claude_pins.EFFORT_ENV, "default")
    monkeypatch.delenv("EVOSKILL_EVOLVER_DAYTONA_JSON", raising=False)
    monkeypatch.setattr(SL, "STATE_ROOT", tmp_path / "cache" / "evolver_state")
    ledger = tmp_path / "ledger.jsonl"
    monkeypatch.setattr(dle, "LEDGER", str(ledger))
    fm = StatefulFakeMachine()
    machines = {fm.id: fm}
    client = SimpleNamespace(get=lambda sid: machines[sid],
                             delete=lambda *a, **k: (_ for _ in ()).throw(AssertionError("machine deleted")))
    monkeypatch.setattr(dle, "client", lambda: client)
    monkeypatch.setattr(dle, "delete_by_id", lambda *a, **k: (_ for _ in ()).throw(AssertionError("delete_by_id")))
    cc = tmp_path / "cc_fake"
    cc.write_bytes(b"ELF fake claude")
    monkeypatch.setattr(EC, "ensure_cc_binary", lambda cfg, run_dir=None: cc)
    monkeypatch.setattr(T, "POLL_S", 0)
    monkeypatch.setattr(EC, "POLL_S", 0)
    monkeypatch.setattr(pce, "LEASE_ACK_S", 5.0)
    regdir = pce.pool_run_dir(rd)
    host = FakePoolHost(regdir, {"m01": pool_row(fm.id, fm.name)}).start()
    s = SimpleNamespace(rd=rd, split=f.split, tok=tok, cfg=cfg, ledger=ledger, fm=fm, machines=machines,
                        host=host, regdir=regdir, tmp=tmp_path, runs={}, rt=rt)
    try:
        yield s
    finally:
        host.stop()


def make_run(s, epoch: int):
    """A StatefulEvolverRun as cmd_dispatch builds one (a new Python object per epoch, as the loop spawns
    processes): $3 = evo_e{N-1} (the newest rollouts), the earlier evo dirs as the prior evidence."""
    rd = s.rd
    run = ES.StatefulEvolverRun(
        epoch=epoch, prompt=rd / f"ITER{epoch}_PROMPT.md", digests=EL.epoch_dir(rd, epoch - 1),
        staging=rd / f"skills_v{epoch}_staging", run_dir=rd,
        prior=[str(EL.epoch_dir(rd, j)) for j in range(epoch - 1)], model="claude-opus-5-5",
        oracle=rd / "oracle", cfg=dict(s.cfg), timeout_s=900,
        receipt_path=rd / f"e{epoch}_EVOLVER_CONTAINMENT.json", trace_path=rd / f"tau3_evolver_e{epoch}.trace.json",
        listing_path=rd / f"e{epoch}_EVOLVER_VM_LISTING.json", log_path=rd / f"tau3_evolver_e{epoch}.log",
        quarantine=s.tmp / "q" / f"e{epoch}", mode="all_epochs")
    run.cinit()
    return run


def play(s, epoch: int) -> int:
    run = make_run(s, epoch)
    s.runs[epoch] = run
    return run.run(gate_argv=None)


def labels(execs) -> list:
    """The machine commands as the spec's step names (tail polls collapsed; the installs' mkdir/sha lines dropped)."""
    out = []
    for cmd, _env in execs:
        if "setsid nohup" in cmd:
            lab = "launch"
        elif cmd.startswith(C + " "):
            words = cmd[len(C) + 1:].split()
            lab = words[0]
            if lab == "probe":
                lab += " " + words[words.index("--phase") + 1]
            elif lab == "extract" and "--incremental" in words:
                lab += " --incremental"
            elif lab == "enter":
                lab = "listing"
            elif lab == "down" and "--wipe" in words:
                lab += " --wipe"
        else:
            continue
        if not (lab == "tail" and out and out[-1] == "tail"):
            out.append(lab)
    return out


def receipt(run) -> dict:
    return json.loads(run.receipt_path.read_text())


def ledger_events(s, cid=None) -> list:
    rows = [json.loads(ln) for ln in s.ledger.read_text().splitlines() if ln.strip()]
    return [(r["event"], r.get("cid"), r.get("verified")) for r in rows if cid is None or r.get("cid") == cid]


def check_receipt(s, run, *, probe_sha=None, **kw):
    """The loop's #294 half for a stateful epoch (pool_container_evolver.check_pool_container_receipt, as
    stateful_evolver.v_session_contained calls it)."""
    st = SL.load_state(s.rd)
    return pce.check_pool_container_receipt(
        receipt(run), log=run.log_path, ledger_path=str(s.ledger), require_libreoffice=LO, session=SL.STATEFUL,
        rw_paths=[r["path"] for r in SL.STATEFUL_RW], probe_pre_sha256=probe_sha or st["container"]["probe_pre_sha256"], **kw)


def check_upload(s, run):
    """The loop's #213 half (stateful_layout.check_stateful_upload) with the state as it was at that epoch."""
    st = SL.load_state(s.rd)
    st_n = dict(st, uploaded=[r for r in st["uploaded"] if int(r.get("epoch", 0)) <= run.epoch])
    return SL.check_stateful_upload(run_dir=s.rd, epoch=run.epoch, split=s.split, state=st_n, receipt=receipt(run),
                                    listing=json.loads(run.listing_path.read_text()))


def add_dirs(argv) -> list:
    return [d for i, d in enumerate(argv) if i and argv[i - 1] == "--add-dir"]


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


# the "status" after the tail polls is collect()'s container_alive() guard (the base's salvage path) before the
# after-run listing; the second listing is the read-only roots inside the container after the session
E1_STEPS = ["list", "prepare", "up", "probe pre", "extract", "listing", "launch", "tail",
            "status", "listing", "probe post", "epoch-end"]
E2_STEPS = ["status", "list", "epoch-begin", "extract --incremental", "listing", "launch",
            "tail", "status", "listing", "probe post", "epoch-end"]
COLLECT_STEPS = E2_STEPS[E2_STEPS.index("listing"):]           # listing, launch, tail, status, listing, ...
E1_ROOTS = [SL.traces_dir(0), SL.solver_ws_dir(0), SL.VM_SCORES, SL.VM_REFERENCE]
E2_ROOTS = [SL.traces_dir(0), SL.traces_dir(1), SL.solver_ws_dir(0), SL.solver_ws_dir(1), SL.VM_SCORES,
            SL.skills_dir(1), SL.VM_REFERENCE]


# ── 1. two epochs: one container, one session
def test_two_epochs_one_container_one_claude_session(sess):
    s, fm, rd = sess, sess.fm, sess.rd
    seen = {}

    def look():                                   # while each session runs, the pool honours this epoch's lease
        seen[fm.turns + 1] = (P.Registry(s.regdir).leases(time.time()), s.host.last["machines"]["m01"]["leased_by"])
    fm.on_play = look
    assert play(s, 1) == 0
    n1 = len(fm.execs)
    assert play(s, 2) == 0
    r1, r2 = s.runs[1], s.runs[2]
    st = SL.load_state(rd)
    # ONE container (the run-scoped cid) and ONE session (the sid) for both epochs; both collected
    assert r1.cid == r2.cid == st["cid"] and r1.cid.startswith("run-") and ctl.CID_RE.match(r1.cid)
    assert r1.sid_session == r2.sid_session == st["sid"] and st["created_epoch"] == 1 and st["torn_down"] is None
    assert {k: v["status"] for k, v in st["epochs"].items()} == {"1": "collected", "2": "collected"}
    assert st["run"] == rd.name and st["sandbox_id"] == fm.id and st["rebuilds"] == []
    # the command order; never a `down`, never a `pack`; one download per epoch; the machine stays
    assert labels(fm.execs[:n1]) == E1_STEPS
    assert labels(fm.execs[n1:]) == E2_STEPS
    every = labels(fm.execs)
    assert "down" not in every and "down --wipe" not in every and "pack" not in every
    assert [x for x in fm.log if x[0] == "download"] == [("download", f"{ctl.cdir(r1.cid)}/state/out.tgz")] * 2
    assert not fm.deleted and fm.up and not fm.wiped
    assert f"{C} epoch-begin --cid {r1.cid} --clear {STAGING}" in [c for c, _e in fm.execs]
    assert (f"{C} epoch-end --cid {r1.cid} --part {STAGING} --part {WORKSPACE} --part /home/evo/.claude/projects "
            f"--part {ctl.VM_OUT}") in [c for c, _e in fm.execs]
    # the ledger: attach + up once, then reattach; a lease per epoch, listed by the pool while each session ran
    assert ledger_events(s) == [("container_attach", r1.cid, None), ("container_up", r1.cid, None),
                                ("container_reattach", r1.cid, None)]
    holder = pce.lease_holder(r1.cid)
    assert seen == {1: ({fm.id: [holder]}, [holder]), 2: ({fm.id: [holder]}, [holder])}
    assert P.Registry(s.regdir).leases(time.time()) == {}
    assert not list((s.regdir / pce.POOL_LEASES_DIR).glob("*.json"))
    rec1, rec2 = receipt(r1), receipt(r2)
    assert rec1["daytona"]["lease"]["released"] is True and rec2["daytona"]["lease"]["released"] is True
    # the token: in the env of exactly the 2 launch execs, never in a command, a file or an upload
    with_tok = [c for c, e in fm.execs if e]
    assert len(with_tok) == 2 and all("setsid nohup" in c for c in with_tok)
    assert not [c for c, _e in fm.execs if TOKEN in c or de.TOKEN_PREFIX in c]
    assert not [p for p, b in fm.files.items() if TOKEN.encode() in b or de.TOKEN_PREFIX.encode() in b]
    assert not [p for p in s.tmp.rglob("*")
                if p.is_file() and not p.is_symlink() and p != s.tok and TOKEN.encode() in p.read_bytes()]
    # the launch (3.5): --session-id at e1, --resume at e2, the system prompt appended, container --add-dirs, no --effort
    a1, a2 = rec1["daytona"]["launch_argv"], rec2["daytona"]["launch_argv"]
    assert a1[a1.index("--session-id") + 1] == st["sid"] and "--resume" not in a1
    assert a2[a2.index("--resume") + 1] == st["sid"] and "--session-id" not in a2
    for a in (a1, a2):
        assert a[a.index("--append-system-prompt") + 1] == "__SYSPROMPT__" and "--effort" not in a
        assert a[:3] == ["claude", "-p", "__PROMPT__"] and "--disallowed-tools" in a
    assert add_dirs(a1) == E1_ROOTS + [STAGING, WORKSPACE]
    assert add_dirs(a2) == E2_ROOTS + [STAGING, WORKSPACE]
    assert [x["flag"] for x in fm.launches] == ["session-id", "resume"]
    assert {x["sid"] for x in fm.launches} == {st["sid"]}
    # the message and the system prompt travel as base64 shell variables: no file in the container, no plain text
    # and no host path in the command, the same system prompt bytes every epoch
    for i, x in enumerate(fm.launches, 1):
        cmd = x["cmd"]
        assert f"printf %s {b64(rd / f'ITER{i}_PROMPT.md')} | base64 -d" in cmd and '"$EVO_MESSAGE"' in cmd
        assert f"printf %s {b64(rd / SL.SYSTEM_PROMPT_FILE)} | base64 -d" in cmd and '"$EVO_SYSTEM"' in cmd
        assert "__SYSPROMPT__" not in cmd and "__PROMPT__" not in cmd and "/Users" not in cmd and "$(cat " not in cmd
        assert "skills_v" not in cmd and SYSTEM_TEXT.splitlines()[0] not in cmd
    assert not [k for k in fm.cfs if k.startswith(de.VM_WORK + "/")]
    assert "base64" in rec1["daytona"]["prompt_delivery"] and rec2["daytona"]["prompt_delivery"] == rec1["daytona"]["prompt_delivery"]
    # e2's extract holds ONLY the delta: evo_e1's rollouts, the skills_v1 snapshot, the re-seed; no prompt file
    (e2m,) = fm.extracts
    want = ("evo/traces/epoch_1/", "evo/solver_workspaces/epoch_1/", "evo/scores/epoch_1.json", "evo/skills/epoch_1/",
            "evo/staging/")
    assert all(m.startswith(want) for m in e2m) and all(any(m.startswith(w) for m in e2m) for w in want)
    assert sorted(m for m in e2m if m.startswith("evo/traces/")) == [f"evo/traces/epoch_1/{EL.unit_of(t, 1)}.jsonl" for t in TASKS]
    lib = sorted(p.relative_to(rd / "skills_v1").as_posix() for p in (rd / "skills_v1").rglob("*") if p.is_file())
    assert sorted(m[len("evo/staging/"):] for m in e2m if m.startswith("evo/staging/")) == lib
    assert sorted(m[len("evo/skills/epoch_1/"):] for m in e2m if m.startswith("evo/skills/epoch_1/")) == lib
    assert {r["role"] for r in rec2["upload_manifest"]} == {"traces", "solver_workspaces", "scores", "skills_snapshot", "seed"}
    assert {r["role"] for r in rec1["upload_manifest"]} == {"traces", "solver_workspaces", "scores", "reference",
                                                            "hook", "env", "policy"}      # no grader
    assert not [r for r in rec1["upload_manifest"] + rec2["upload_manifest"] if r["vm_path"].startswith("/Users")]
    # epoch-begin cleared /evo/staging (e1's skill gone before the seed) and the stale rc/log: e2 streamed e2's log
    eb = rec2["daytona"]["container"]["epoch_begin"]
    assert eb["ok"] is True and eb["cleared"] == {STAGING: 1} and set(eb["out_reset"]) == {"rc", "evolver.log",
                                                                                            "evolver.stderr"}
    assert b"ls /evo/traces/epoch_1" in r2.log_path.read_bytes() and b"epoch_0" not in r2.log_path.read_bytes()
    assert rec2["daytona"]["log_round_trip"]["equal"] is True and rec2["daytona"]["claude_rc"] == 0
    # the workspace persists across epochs; the transcript grew as a prefix; the snapshots verify; the installs
    d1, d2 = SL.snapshot_dir(rd, 1), SL.snapshot_dir(rd, 2)
    assert SL.verify_snapshot(d1) == [] and SL.verify_snapshot(d2) == []
    m1 = json.loads((d1 / SL.SNAPSHOT_MANIFEST).read_text())
    assert m1["sid"] == st["sid"] and m1["cid"] == r1.cid and m1["epoch"] == 1 and m1["contaminated"] is False
    t1, t2 = (d / "projects" / "-evo-work" / f"{st['sid']}.jsonl" for d in (d1, d2))
    assert t2.read_bytes().startswith(t1.read_bytes()) and len(t2.read_bytes()) > len(t1.read_bytes())
    assert t2.read_bytes() == (rd / "e2_evolver_session" / "-evo-work" / f"{st['sid']}.jsonl").read_bytes()
    assert sorted(p.name for p in (d2 / SL.SNAPSHOT_WORKSPACE).iterdir()) == ["turn_1.md", "turn_2.md"]
    w1, w2 = (rd / SL.WORKSPACE_DIR.format(epoch=k) for k in (1, 2))
    assert sorted(p.name for p in w2.iterdir()) == ["turn_1.md", "turn_2.md"]
    assert (w1 / "turn_1.md").is_file() and not (w1 / "turn_2.md").exists()
    assert (rd / "skills_v1_staging/table-audit/SKILL.md").read_text().endswith("body v1\n")
    assert (rd / "skills_v2_staging/table-audit/SKILL.md").read_text().endswith("body v2\n")
    assert (d2 / "receipt.json").is_file() and st["epochs"]["2"]["snapshot"] == str(d2)
    # the receipts: the upload check (cumulative, re-derived from the host bytes) passes for both epochs
    for run in (r1, r2):
        c = check_upload(s, run)
        assert c.status == "PASS", c.detail
    c1, c2 = rec1["daytona"]["container"], rec2["daytona"]["container"]
    assert {k: c1["session"][k] for k in ("mode", "epoch", "resume", "kept", "rebuilt", "created_epoch")} == \
        {"mode": "stateful", "epoch": 1, "resume": False, "kept": True, "rebuilt": None, "created_epoch": 1}
    assert c2["session"]["resume"] is True and c2["session"]["cid"] == r1.cid and c2["session"]["epoch"] == 2
    assert c2["attach"]["ledger"] == "container_reattach" and c2["attach"]["list"] == [r1.cid]
    assert c2["probe_pre_sha256"] == c1["probe_pre_sha256"] == st["container"]["probe_pre_sha256"]
    assert c2["probe_pre"] == c1["probe_pre"] and c2["plan"] == c1["plan"] and c2["spec"] == c1["spec"]
    assert "down" not in c1 and "down" not in c2 and c2["epoch_end"]["kept"] is True and c1["kept"] is True
    assert rec2["upload_manifest_cumulative"] == st["uploaded"]
    assert len(st["uploaded"]) == len(rec1["upload_manifest"]) + len(rec2["upload_manifest"])
    assert rec1["daytona"]["session_id"] == rec2["daytona"]["session_id"] == st["sid"]
    assert rec2["daytona"]["workspace_dir"] == str(w2) and rec2["daytona"]["workspace_files"] == 2
    assert rec2["daytona"]["snapshot_dir"] == str(d2) and rec2["daytona"]["driver"] == "scripts/daytona/evolver_session.py"
    assert [pm["vm"] for pm in rec2["path_view"] if pm["epoch"] == 2] == \
        [SL.traces_dir(1), SL.solver_ws_dir(1), SL.scores_file(1), SL.skills_dir(1), STAGING]
    assert rec2["readonly_roots"] == E2_ROOTS
    # the evidence hashes: before = the host evo dirs (the loop's judge); after = the container roots (informational)
    assert sorted(rec2["evidence_sha256_before"]) == [os.path.realpath(EL.epoch_dir(rd, K)) for K in (0, 1)]
    assert len(rec1["evidence_sha256_before"]) == 1
    for rec in (rec1, rec2):
        assert set(rec["evidence_sha256_after_vm"]) == {pm["vm"] for pm in rec["path_view"] if pm["role"] in SL.DIR_ROLES}


# ── 2. a turn is taken once; a session cannot go back or skip
def test_an_epoch_runs_once_in_a_session_and_epochs_never_skip(sess):
    s, fm = sess, sess.fm
    assert play(s, 1) == 0
    n = len(fm.execs)
    assert play(s, 1) == ES.EXIT_USAGE                        # already collected: refused before any machine command
    assert play(s, 3) == ES.EXIT_USAGE                        # epoch 2 never ran
    assert len(fm.execs) == n and fm.up
    assert {k: v["status"] for k, v in SL.load_state(s.rd)["epochs"].items()} == {"1": "collected"}


# ── 3. the driver refuses before any machine command when the run / env is not a stateful one
@pytest.mark.parametrize("mut,needle", [
    ("per_epoch", "drives only a 'stateful' run"),
    ("typo", "evolver_session"),
    ("no_env", SL.ENV_SESSION),
    ("effort", "EVOSKILL_CLAUDE_EFFORT must opt out"),
    ("sysprompt_env", SL.ENV_SYSTEM_PROMPT),
    ("sysprompt_missing", "missing"),
    ("ao_mode", "prompt_mode must be 'template'"),
])
def test_prepare_refusals_touch_no_machine(sess, monkeypatch, capsys, mut, needle):
    s, fm, rd = sess, sess.fm, sess.rd
    rt = dict(s.rt)
    if mut == "per_epoch":
        rt["evolver_session"] = "per_epoch"
    elif mut == "typo":
        rt["evolver_session"] = "statefull"
    elif mut == "no_env":
        monkeypatch.delenv(SL.ENV_SESSION)
    elif mut == "effort":
        monkeypatch.setenv(claude_pins.EFFORT_ENV, "high")
    elif mut == "sysprompt_env":
        monkeypatch.setenv(SL.ENV_SYSTEM_PROMPT, str(rd / "OTHER.md"))
    elif mut == "sysprompt_missing":
        (rd / SL.SYSTEM_PROMPT_FILE).unlink()
    elif mut == "ao_mode":
        rt["prompt_mode"] = "ao"
    (rd / "RUNTIME.json").write_text(json.dumps(rt))
    assert play(s, 1) == ES.EXIT_USAGE
    assert fm.execs == [] and SL.load_state(rd) is None and not (rd / "e1_EVOLVER_CONTAINMENT.json").exists()
    assert needle in capsys.readouterr().out                    # the driver's log() prints to stdout


# ── 4. finish: down --wipe LAST, ledgered, the teardown record; refusals; --machine-absent-ok
def test_finish_tears_the_container_down_last_ledgers_and_records(sess, monkeypatch, capsys):
    s, fm, rd = sess, sess.fm, sess.rd
    assert play(s, 1) == 0 and play(s, 2) == 0
    r2 = s.runs[2]
    n = len(fm.execs)
    args = lambda **kw: SimpleNamespace(**{"run_dir": str(rd), "abort": False, "machine_absent_ok": False, **kw})  # noqa: E731
    # refused while a dispatch of the run is live (it holds the run's dispatch lock shared): nothing touched
    import fcntl
    lp = rd / T.DISPATCH_LOCK
    lp.parent.mkdir(parents=True, exist_ok=True)
    with open(lp, "a") as held:
        fcntl.flock(held, fcntl.LOCK_SH)
        assert ES.cmd_finish(args()) == T.EXIT_LIVE
    assert len(fm.execs) == n and not (rd / SL.TEARDOWN_RECORD).exists()
    assert ES.cmd_finish(args()) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["ok"] is True and out["cid"] == r2.cid and out["abort"] is False
    assert labels(fm.execs[n:]) == ["down --wipe"] and fm.wiped and not fm.up
    assert ledger_events(s)[-1] == ("container_down", r2.cid, True)
    assert pce.ledger_container_down(str(s.ledger), sandbox_id=fm.id, cid=r2.cid)
    rec = json.loads((rd / SL.TEARDOWN_RECORD).read_text())
    assert rec["schema"] == ES.TEARDOWN_SCHEMA and rec["cid"] == r2.cid and rec["sid"] == r2.sid_session
    assert rec["sandbox_id"] == fm.id and rec["run"] == rd.name
    assert rec["down"]["verified"] is True and rec["down"]["wiped"] is True and rec["ledgered"] is True
    assert [e["epoch"] for e in rec["epochs"]] == [1, 2] and rec["problems"] == [] and rec["abort"] is False
    assert ES.teardown_record_problems(rec, str(s.ledger)) == []
    assert ES.teardown_record_problems({**rec, "down": {"verified": True, "wiped": False}}, str(s.ledger))
    assert ES.teardown_record_problems({**rec, "cid": "run-other"}, str(s.ledger))
    st = SL.load_state(rd)
    assert st["torn_down"]["record"] == str(rd / SL.TEARDOWN_RECORD) and st["torn_down"]["abort"] is False
    # the teardown's own lease is gone; a second finish is a no-op
    assert P.Registry(s.regdir).leases(time.time()) == {}
    assert ES.cmd_finish(args()) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["already"] is True
    assert labels(fm.execs[n:]) == ["down --wipe"]
    # --machine-absent-ok: machine.json no longer names the session's sandbox -> recorded, no down, --abort noted
    st["torn_down"] = None
    SL.write_state(rd, st)
    s.host.stop()
    pce.write_json_atomic(s.regdir / pce.POOL_MACHINES_FILE,
                          machines_json({"m02": pool_row("ffff2222eeee", pce.POOL_PREFIX + "v1pl-m2-130000")},
                                        current="m02"))
    assert ES.cmd_finish(args()) == ES.EXIT_IDENTITY
    assert SL.load_state(rd)["torn_down"] is None and "machine-absent-ok" in capsys.readouterr().out
    assert ES.cmd_finish(args(abort=True, machine_absent_ok=True)) == 0
    rec = json.loads((rd / SL.TEARDOWN_RECORD).read_text())
    assert rec["machine_absent"]["machine_json"]["sandbox_id"] == "ffff2222eeee" and rec["down"] is None
    assert rec["abort"] is True and rec["problems"] == [] and SL.load_state(rd)["torn_down"]["abort"] is True
    assert labels(fm.execs[n:]) == ["down --wipe"]              # nothing more touched the old machine


# ── 5. SIGTERM mid-launch: the finally keeps the container, releases the lease, writes the state
def test_sigterm_mid_launch_keeps_the_container_and_the_next_epoch_stops_the_straggler(sess):
    s, fm, rd = sess, sess.fm, sess.rd
    fm.crash = SystemExit(128 + 15)                             # _on_sigterm inside run(): the launch exec is cut
    r1 = make_run(s, 1)
    s.runs[1] = r1
    with pytest.raises(SystemExit):
        r1.run(gate_argv=None)
    n = len(fm.execs)
    every = labels(fm.execs)
    # the finally keeps the container
    assert every[-1] == "launch" and "down" not in every and "down --wipe" not in every and fm.up
    assert P.Registry(s.regdir).leases(time.time()) == {}      # the finally released the lease
    rec1 = receipt(r1)
    assert rec1["daytona"]["lease"]["released"] is True and rec1["daytona"]["container"]["kept"] is True
    assert rec1["daytona"]["container_removed_utc"] is None
    st = SL.load_state(rd)
    assert st["epochs"]["1"]["status"] == "launched" and st["torn_down"] is None and len(st["uploaded"]) > 0
    assert not SL.snapshot_dir(rd, 1).exists()
    # the crashed driver's claude is still running: epoch-begin refuses until stop-session, then e2 proceeds
    fm.session_live = True
    assert play(s, 2) == 0
    assert labels(fm.execs[n:]) == ["status", "list", "epoch-begin", "stop-session", "epoch-begin",
                                    "extract --incremental", *COLLECT_STEPS]
    r2 = s.runs[2]
    rec2 = receipt(r2)
    c2 = rec2["daytona"]["container"]
    assert c2["epoch_begin_stop_session"]["ok"] is True and c2["epoch_begin"]["ok"] is True
    a2 = rec2["daytona"]["launch_argv"]
    assert a2[a2.index("--resume") + 1] == st["sid"] and "--session-id" not in a2
    st = SL.load_state(rd)
    assert {k: v["status"] for k, v in st["epochs"].items()} == {"1": "launched", "2": "collected"}
    d2 = SL.snapshot_dir(rd, 2)
    assert SL.verify_snapshot(d2) == []
    t2 = (d2 / "projects" / "-evo-work" / f"{st['sid']}.jsonl").read_bytes()
    assert t2.count(b'"subtype": "init"') == 2                  # e1's turn (written before the cut) + e2's
    c = check_upload(s, r2)
    assert c.status == "PASS", c.detail


# ── 6. a replaced pool machine: rebuild from the newest verified snapshot (7.3), then --resume
def test_a_replaced_pool_machine_rebuilds_the_container_from_the_snapshot(sess):
    s, fm1, rd = sess, sess.fm, sess.rd
    assert play(s, 1) == 0
    r1 = s.runs[1]
    n1 = len(fm1.execs)
    fm2 = StatefulFakeMachine(sid="bbbb2222cccc", name=pce.POOL_PREFIX + "v1pl-m2-130000")
    fm2.turns = fm1.turns                                       # the SESSION continues: its next turn is the 2nd
    s.machines[fm2.id] = fm2
    s.host.machines, s.host.current = {"m02": pool_row(fm2.id, fm2.name)}, "m02"
    time.sleep(0.2)                                             # the host publishes the replacement
    assert play(s, 2) == 0
    r2 = s.runs[2]
    st = SL.load_state(rd)
    assert r2.cid != r1.cid and st["cid"] == r2.cid and st["sandbox_id"] == fm2.id and st["sid"] == r1.sid_session
    assert len(fm1.execs) == n1 and fm1.up                      # the old machine is gone from the pool: untouched
    assert labels(fm2.execs) == ["list", "prepare", "restore", "up", "probe pre", "epoch-begin",
                                 "extract --incremental", *COLLECT_STEPS]
    d1 = SL.snapshot_dir(rd, 1)
    want = sorted(("home/evo/.claude/projects/" if rel.startswith("projects/") else WORKSPACE.lstrip("/") + "/")
                  + rel.split("/", 1)[1]
                  for rel in SL.snapshot_manifest(d1)["files"] if rel.startswith(("projects/", SL.SNAPSHOT_WORKSPACE + "/")))
    assert fm2.restores == [want] and any(m.endswith(f"{st['sid']}.jsonl") for m in want)
    assert "evo/evolver_workspace/turn_1.md" in want
    (cum,) = fm2.extracts                                       # ONE cumulative re-upload: e1's rows + e2's delta
    rec2 = receipt(r2)
    assert sorted(cum) == sorted(r["vm_path"].lstrip("/") for r in rec2["upload_manifest_cumulative"])
    assert {r["epoch"] for r in rec2["upload_manifest_cumulative"]} == {1, 2}
    a2 = rec2["daytona"]["launch_argv"]
    assert a2[a2.index("--resume") + 1] == st["sid"] and "--session-id" not in a2
    rb = rec2["daytona"]["container"]["session"]["rebuilt"]
    assert rb["old_cid"] == r1.cid and rb["old_sandbox_id"] == fm1.id and rb["from_epoch"] == 1
    assert rb["new_cid"] == r2.cid and rb["from_snapshot"] == str(d1) and fm1.id[:8] in rb["why"]
    assert st["rebuilds"] == [{"epoch": 2, **rb}] and rec2["daytona"]["container"]["upload"]["how"].startswith("cumulative")
    assert rec2["daytona"]["container"]["restore"]["ok"] is True and rec2["daytona"]["container"]["cid"] == r2.cid
    assert st["container"]["probe_pre_sha256"] == rec2["daytona"]["container"]["probe_pre_sha256"]
    assert ledger_events(s, r1.cid) == [("container_attach", r1.cid, None), ("container_up", r1.cid, None)]
    assert ledger_events(s, r2.cid) == [("container_attach", r2.cid, None), ("container_up", r2.cid, None)]
    for run in (r1, r2):
        c = check_upload(s, run)
        assert c.status == "PASS", c.detail
    d2 = SL.snapshot_dir(rd, 2)
    assert SL.verify_snapshot(d2) == []
    t1, t2 = (d / "projects" / "-evo-work" / f"{st['sid']}.jsonl" for d in (d1, d2))
    assert t2.read_bytes().startswith(t1.read_bytes()) and len(t2.read_bytes()) > len(t1.read_bytes())
    assert sorted(p.name for p in (d2 / SL.SNAPSHOT_WORKSPACE).iterdir()) == ["turn_1.md", "turn_2.md"]
    # a contaminated (or absent) snapshot is never rebuilt from
    (d1 / SL.SNAPSHOT_MANIFEST).write_text(json.dumps({**json.loads((d1 / SL.SNAPSHOT_MANIFEST).read_text()),
                                                       "contaminated": True}))
    assert SL.verify_snapshot(d1) and SL.verify_snapshot(d2) == []
    m = make_run(s, 3)                                          # a run object only: nothing is launched
    m.receipt = {"daytona": {"container": {}}}
    assert m._newest_verified_snapshot() == (d2, 2)
    (d2 / SL.SNAPSHOT_MANIFEST).unlink()
    assert m._newest_verified_snapshot() == (None, None)
    assert len(m.c()["snapshots_skipped"]) == 2


# ── 7. cleanup refuses a live stateful session's container unless --force
def test_cmd_cleanup_refuses_a_live_stateful_session_unless_forced(sess, monkeypatch):
    s, fm, rd = sess, sess.fm, sess.rd
    assert play(s, 1) == 0
    r1 = s.runs[1]
    n = len(fm.execs)
    args = lambda **kw: SimpleNamespace(**{"sandbox_id": None, "registry": None, "run_dir": str(rd), "force": False, **kw})  # noqa: E731
    assert EC.cmd_cleanup(args()) == 4                                     # by the run's record
    other = s.tmp / "other_run"
    other.mkdir()
    assert EC.cmd_cleanup(args(run_dir=str(other), sandbox_id=fm.id)) == 4  # by the sandbox the record names
    assert len(fm.execs) == n and fm.up and not fm.wiped
    assert EC._live_stateful_session(str(rd), None).startswith(f"a live stateful session's container {r1.cid}")
    assert EC._live_stateful_session(None, fm.id).startswith(f"a live stateful session's container {r1.cid}")
    assert EC._live_stateful_session(None, "nosuchsandbox") == "" and EC._live_stateful_session(str(s.tmp), None) == ""
    p = SL.state_path(rd)                                                  # a corrupt record is never read as absent
    keep = p.read_bytes()
    p.write_text("{not json")
    assert "unreadable" in EC._live_stateful_session(str(rd), None) and EC.cmd_cleanup(args(run_dir=str(rd))) == 4
    p.write_bytes(keep)
    assert EC.cmd_cleanup(args(run_dir=str(rd), force=True)) == 0          # --force wipes it, ledgered
    assert labels(fm.execs[n:]) == ["list", "down --wipe"] and fm.wiped
    assert ledger_events(s)[-1] == ("container_down", r1.cid, True)
    st = SL.load_state(rd)
    st["torn_down"] = {"utc": "x", "record": "y"}
    SL.write_state(rd, st)
    assert EC.cmd_cleanup(args(run_dir=str(rd))) == 0                      # torn down: cleanup is the manual wipe again
    seen = []
    monkeypatch.setattr(EC, "cmd_cleanup", lambda a: seen.append(a) or 0)
    monkeypatch.setattr(EC.signal, "signal", lambda *a, **k: None)
    assert EC.main(["cleanup", "--run-dir", str(rd), "--force"]) == 0 and seen[-1].force is True
    assert EC.main(["cleanup", "--run-dir", str(rd), "--sandbox-id", fm.id]) == 0 and seen[-1].force is False


# ── 8. cmd_dispatch routes on the RUNTIME key (the loop's argv unchanged: $3 = evo_e{N-1})
def test_cmd_dispatch_routes_on_the_runtime_session_key(sess, monkeypatch):
    s, rd = sess, sess.rd
    calls = []

    class FakeRun:
        def __init__(self, **kw):
            self.kw = kw

        def cinit(self, **kw):
            pass

        def run(self, gate_argv=None):
            calls.append((type(self).__name__, self.kw, gate_argv))
            return 0

    class Per(FakeRun):
        pass

    class Stateful(FakeRun):
        pass
    monkeypatch.setattr(EC, "ContainerEvolverRun", Per)
    monkeypatch.setattr(ES, "StatefulEvolverRun", Stateful)
    monkeypatch.setenv("EVOSKILL_TAU3_RUN_DIR", str(rd))
    monkeypatch.setenv("EVOSKILL_EVOLVER_MODEL", "claude-opus-5-5")
    monkeypatch.setenv("EVOSKILL_EVOLVER_EVIDENCE", "all_epochs")
    monkeypatch.setenv("EVOSKILL_PRIOR_DIGESTS", str(EL.epoch_dir(rd, 0)))
    a = SimpleNamespace(epoch="2", prompt=str(rd / "ITER2_PROMPT.md"), digests=str(EL.epoch_dir(rd, 1)),
                        staging=str(rd / "skills_v2_staging"))

    def set_session(v):
        r = dict(s.rt)
        r.pop(SL.SESSION_KEY, None)
        if v is not None:
            r[SL.SESSION_KEY] = v
        (rd / "RUNTIME.json").write_text(json.dumps(r))
    set_session("stateful")
    assert EC.cmd_dispatch(a) == 0 and calls[-1][0] == "Stateful"
    kw, gate = calls[-1][1], calls[-1][2]
    assert kw["epoch"] == 2 and kw["prompt"] == rd / "ITER2_PROMPT.md" and kw["prior"] == [str(EL.epoch_dir(rd, 0))]
    assert kw["digests"] == EL.epoch_dir(rd, 1) and kw["mode"] == "all_epochs"
    assert kw["receipt_path"] == rd / "e2_EVOLVER_CONTAINMENT.json"
    assert kw["listing_path"] == rd / "e2_EVOLVER_VM_LISTING.json" and kw["log_path"] == rd / "tau3_evolver_e2.log"
    # a stateful session skips the launcher's digest-layout gate (tau3_evolver.sh); the driver's own
    # checks stand. The per-epoch driver still runs it with the same positional argv.
    assert gate is None
    monkeypatch.delenv(SL.ENV_SESSION)
    for v in ("per_epoch", None):                              # today's driver, the same kwargs
        set_session(v)
        assert EC.cmd_dispatch(a) == 0 and calls[-1][0] == "Per"
        assert {k: v_ for k, v_ in calls[-1][1].items() if k != "quarantine"} == \
            {k: v_ for k, v_ in kw.items() if k != "quarantine"}
        g = calls[-1][2]
        assert g[:3] == ["bash", str(EC.REPO / "scripts/tau3_evolver.sh"), "2"] and g[3] == str(rd / "ITER2_PROMPT.md")
    n = len(calls)
    set_session("statefull")
    assert EC.cmd_dispatch(a) == EC.EXIT_USAGE and len(calls) == n           # a typo never means either
    set_session("per_epoch")
    monkeypatch.setenv(SL.ENV_SESSION, SL.STATEFUL)
    assert EC.cmd_dispatch(a) == EC.EXIT_USAGE and len(calls) == n           # env and file disagree
    set_session("stateful")
    monkeypatch.delenv(SL.ENV_SESSION)
    assert EC.cmd_dispatch(a) == 0 and calls[-1][0] == "Stateful"            # the driver's prepare refuses the env


# ── 9. the stateful launch: the message and the system prompt as base64 shell variables, never a file
def test_stateful_launch_delivers_the_message_and_the_system_prompt_as_base64_variables(tmp_path):
    msg, sysp = tmp_path / "ITER1_PROMPT.md", tmp_path / SL.SYSTEM_PROMPT_FILE
    msg.write_text(MESSAGE.format(N=1, prev=0))
    sysp.write_text(SYSTEM_TEXT)
    base = ["claude", "-p", "__PROMPT__", "--model", "claude-opus-5-5", "--resume", "x" * 36,
            "--append-system-prompt", "__SYSPROMPT__"]
    r = object.__new__(ES.StatefulEvolverRun)
    r.claude_argv = lambda: list(base)
    r.prompt, r.system_prompt, r.state = msg, sysp, None
    r.cid, r.cdir, r.token = RCID, ctl.cdir(RCID), TOKEN
    r.receipt, r.save_receipt, r.fd_env = {"daytona": {}}, lambda: None, T.FD_ENV["token_file"]
    got = []
    r.sh = lambda cmd, **kw: (got.append((cmd, kw.get("env"))), (0, ""))[1]
    r.launch()
    (cmd, env), = got
    assert f'EVO_MESSAGE="$(printf %s {b64(msg)} | base64 -d; printf x)"' in cmd and '"$EVO_MESSAGE"' in cmd
    assert f'EVO_SYSTEM="$(printf %s {b64(sysp)} | base64 -d; printf x)"' in cmd and '"$EVO_SYSTEM"' in cmd
    assert cmd.index('"$EVO_MESSAGE"') > cmd.index("EVO_MESSAGE=") and cmd.index('"$EVO_SYSTEM"') > cmd.index("EVO_SYSTEM=")
    assert "__SYSPROMPT__" not in cmd and "__PROMPT__" not in cmd and TOKEN not in cmd and "$(cat " not in cmd
    assert "skills_v" not in cmd and "# Evolver" not in cmd                  # the texts travel encoded, never plain
    assert env == {T.SECRET_ENV: TOKEN} and r.launched is True
    assert f'{T.FD_ENV["token_file"]}=3' in cmd
    assert r.receipt["daytona"]["launch_argv"] == base and "base64" in r.receipt["daytona"]["prompt_delivery"]
    # an argv without both placeholders is refused before anything runs
    got.clear()
    r.claude_argv = lambda: base[:-2]
    with pytest.raises(ES.Refuse) as ei:
        r.launch()
    assert ei.value.code == ES.EXIT_USAGE and "__SYSPROMPT__" in str(ei.value) and got == []
    # no prompt / system-prompt FILE is uploaded: the mixin names none (the per-epoch driver keeps its PROMPT.md)
    assert ES.StatefulMixin.sysprompt_vm_file is None and ES.StatefulMixin.prompt_vm_file is None
    assert T.EvolverRun.sysprompt_vm_file is None and T.EvolverRun.prompt_vm_file == f"{de.VM_WORK}/PROMPT.md"


# ── 10. the two seams of the per-epoch driver are today's manifest and add-dir list
def test_the_per_epoch_seams_are_todays_manifest_and_add_dirs(monkeypatch, tmp_path):
    r = object.__new__(T.EvolverRun)
    r.run_dir, r.epoch, r.prompt = tmp_path, 1, tmp_path / "ITER1_PROMPT.md"
    r.evidence_dirs, r.oracle, r.staging = ["/x/digests_e1_train"], tmp_path / "oracle", tmp_path / "skills_v1_staging"
    r.inputs, r.references, r.grader = None, "/x/refs", "/x/grader"
    r.grants, r.grader_grant = {"train_inputs": False, "train_references": True}, True
    r.template_mode, r.env, r.split = True, {"HOME": "/home/evo"}, tmp_path / "split.json"
    assert r.add_dirs_for() == ["/x/digests_e1_train", str(r.staging), "/x/refs", "/x/grader"]
    r.template_mode = False
    assert r.add_dirs_for() == ["/x/digests_e1_train", str(r.staging), str(r.oracle), "/x/refs", "/x/grader"]
    seen = {}
    monkeypatch.setattr(de, "build_upload_manifest", lambda **kw: seen.update(kw) or (["rows"], {"g": b""}))
    assert r.build_manifest(tmp_path / "skills_v0") == (["rows"], {"g": b""})
    assert seen["prev_library"] == tmp_path / "skills_v0" and seen["template_mode"] is False and seen["split"] == r.split
    assert set(seen) == {"run_dir", "epoch", "prompt", "evidence_dirs", "oracle", "staging", "prev_library", "inputs",
                         "references", "grants", "grader", "grader_granted", "template_mode", "env_text", "hooks_text",
                         "split"}
    monkeypatch.setattr(de, "build_upload_manifest", lambda **kw: (_ for _ in ()).throw(ValueError("bad row")))
    with pytest.raises(T.Refuse) as ei:
        r.build_manifest(tmp_path / "skills_v0")
    assert ei.value.code == T.EXIT_MANIFEST and "upload manifest: bad row" in str(ei.value)
    src = inspect.getsource(T.EvolverRun.prepare)
    assert "self.build_manifest(prev_lib)" in src and "self.add_dirs_for()" in src
    assert "build_upload_manifest(" not in src                  # the call moved into the seam
    assert ES.StatefulMixin.build_manifest is not T.EvolverRun.build_manifest
    assert ES.StatefulMixin.add_dirs_for is not T.EvolverRun.add_dirs_for


# ── 11. the driver's small helpers
def test_driver_helpers(tmp_path):
    cid = ES.new_cid()
    assert cid.startswith("run-") and ctl.CID_RE.match(cid) and cid != ES.new_cid()
    assert ES.probe_sha256({"b": 1, "a": [2]}) == SL.probe_sha256({"a": [2], "b": 1})
    rd = tmp_path / "r"
    rd.mkdir()
    assert ES.pool_ttl_seconds(rd) == ES.DEFAULT_MAX_SECONDS                          # no RUNTIME.json
    for v, want in ((180, 10800), (1440, 86400), (0, ES.DEFAULT_MAX_SECONDS), (True, ES.DEFAULT_MAX_SECONDS),
                    ("x", ES.DEFAULT_MAX_SECONDS), (None, ES.DEFAULT_MAX_SECONDS), (1.5, 90)):
        (rd / "RUNTIME.json").write_text(json.dumps({"daytona_leg": {"pool_ttl_minutes": v}}))
        assert ES.pool_ttl_seconds(rd) == want, v
    src = tmp_path / "src"
    (src / "a").mkdir(parents=True)
    (src / "a" / "f.txt").write_text("x")
    assert ES.copy_tree_nofollow(src, tmp_path / "dst") == 1 and (tmp_path / "dst/a/f.txt").read_text() == "x"
    assert ES.copy_tree_nofollow(tmp_path / "absent", tmp_path / "dst2") == 0 and (tmp_path / "dst2").is_dir()
    os.symlink(str(tmp_path / "dst"), str(src / "link"))
    with pytest.raises(ES.Refuse):
        ES.copy_tree_nofollow(src, tmp_path / "dst3")
    for name in ("status", "finish"):
        assert f"evolver_session.py {name}" in ES.__doc__


# ── 12. the loop's #294 half on the receipts this driver writes (layout v2: before = the host evo dirs,
# after_vm = the container roots, informational; no INDEX.json cross-check)
def test_the_loops_receipt_check_accepts_the_drivers_receipts_and_the_teardown(sess, capsys):
    s, rd = sess, sess.rd
    assert play(s, 1) == 0 and play(s, 2) == 0
    r1, r2 = s.runs[1], s.runs[2]
    for run in (r1, r2):
        ok, det = check_receipt(s, run)
        assert ok, det
    assert ES.cmd_finish(SimpleNamespace(run_dir=str(rd), abort=False, machine_absent_ok=False)) == 0
    rec = json.loads((rd / SL.TEARDOWN_RECORD).read_text())
    ok, det = check_receipt(s, r2, teardown=rec)
    assert ok, det
    ok, det = check_receipt(s, r2, teardown={**rec, "down": {"verified": True, "wiped": False}})
    assert not ok and "teardown not verified" in det
