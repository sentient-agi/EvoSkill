"""The evolver CONTAINER on the run's pool machine (scripts/daytona/evolver_container_ctl.py, the machine side;
scripts/daytona/evolver_container.py, the host driver; src/orchestrator/pool_container_evolver.py, the checks).

NO ROOT, NO LINUX, NO NETWORK, NO SANDBOX, NO MODEL CALL:
  * the generated init script is RUN by bash with mount/mkdir/touch/ln/cd/[/python3/exec stubbed, and the command
    sequence bash really executes is replayed in a MOUNT-NAMESPACE SIMULATOR over a fake machine tree (fake dataset
    goldens, runner caches, /opt/evoskill, leg/pool scripts, the shim's state, /root, /tmp, /workspace): pivot_root
    re-roots the mounts, `umount -l` of the old root drops every machine mount, and the container's view is then
    checked path by path (what is visible, from where, read-only or not, and that nothing else is);
  * the egress proxy (a subclass of scripts/evolver_egress_proxy.Proxy) runs on loopback against local fake
    upstreams: allowlist, port rule, the chained CONNECT through the machine's own proxy, its auth header never
    logged;
  * the probe verdicts: a good in-container probe passes, and each planted violation is named;
  * the host driver runs END TO END against a scripted fake pool machine (fake Daytona client): the receipts it writes
    carry every key the own-sandbox receipts carry, the loop's checks (check_upload_manifest unchanged +
    check_pool_container_receipt) pass on them, the machine is never deleted, and the OAuth token appears in no
    command string and no file -- only in the env of exactly the launch exec.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "daytona"))
sys.path.insert(0, str(ROOT / "scripts"))

import evolver_container_ctl as ctl  # noqa: E402
import ssb_leg_daytona_loop as M  # noqa: E402  (the pool's constants: the registry format is THEIRS)
import ssb_leg_pool as P  # noqa: E402
from src.orchestrator import daytona_evolver as de  # noqa: E402
from src.orchestrator import pool_container_evolver as pce  # noqa: E402

CID = "e1-20260925t203000-abc123"
STAGING = "/evo/mirror/outputs/run_x/skills_v1_staging"
RUN_MIRROR = "/evo/mirror/outputs/run_x"
# The leg machine's own tree of mirrored host paths (the dataset, the repo, caches), all of which the container must
# not see; a neutral stand-in user.
MACHINE_TREE = "Users/runner"
FAKE_FACTS = {"sysdirs": {"bin": "symlink:usr/bin", "lib": "symlink:usr/lib", "lib64": "symlink:usr/lib64",
                          "sbin": "symlink:usr/sbin"},
              "masks_present": ["/usr/local/bin/docker"], "arch": "x86_64", "kernel": "6.8.0", "ncpu": 8,
              "mem_total": 32 << 30, "cg_root_memory_max": "max", "userns_max": "63000",
              "unprivileged_userns_clone": ""}
OWN_SANDBOX_KEYS = json.loads((ROOT / "tests/fixtures/evolver_container/own_sandbox_receipt_keys.json").read_text())
# keys the own-sandbox receipt carried that the driver no longer writes (a staged file equal to a benchmark workbook
# is not refused: the run imposes no such rule on what the evolver writes)
RETIRED_DAYTONA_KEYS = {"staging_dataset_clashes"}


def spec_for(rw=None, ro_rebind=(), ro_roots=None, cid=CID, limits=None) -> dict:
    rw = rw if rw is not None else [{"path": STAGING, "owner": "evo", "mode": "0755"}]
    return {"schema": ctl.SCHEMA, "cid": cid, "rw": rw, "ro_rebind": list(ro_rebind),
            "readonly_roots": ro_roots if ro_roots is not None else [f"{RUN_MIRROR}/digests_e1_train",
                                                                     f"{RUN_MIRROR}/grader_clean"],
            "limits": limits or {}, "cc": {"version": ctl.CC_VERSION, "sha256": ctl.CC_SHA256},
            "egress": {"allow": ctl.ALLOW, "must_reach": list(de.MUST_REACH), "must_refuse": list(de.MUST_REFUSE)}}


def ao_spec() -> dict:
    ev = f"{RUN_MIRROR}/digests_e1_train"
    return spec_for(rw=[{"path": RUN_MIRROR, "owner": "root", "mode": "1777"}], ro_rebind=[ev, f"{RUN_MIRROR}/oracle"],
                    ro_roots=[ev, f"{RUN_MIRROR}/oracle"])


def plan_of(spec, facts=FAKE_FACTS):
    lim = ctl.resolve_limits(facts, spec.get("limits"))
    return ctl.plan(spec, facts, lim), lim


# ═════════════════════════════════════════════════════════════════════════ a fake machine + the simulator
def _w(p: Path, data: bytes = b"x", link: str | None = None):
    p.parent.mkdir(parents=True, exist_ok=True)
    if link is not None:
        p.symlink_to(link)
    else:
        p.write_bytes(data)


@pytest.fixture
def machine(tmp_path):
    """The pool machine's filesystem, rooted at H: what the leg image + leg bundle + pool + this controller lay out."""
    H = tmp_path / "H"
    m = SimpleNamespace(H=H)
    for rel in ("usr/bin/bash", "usr/bin/setpriv", "usr/bin/env", "usr/bin/nsenter", "usr/local/bin/python3",
                "usr/local/bin/docker", "usr/lib/x86_64-linux-gnu/libc.so.6", "usr/lib64/ld-linux-x86-64.so.2",
                "usr/sbin/pivot_root", "usr/lib/libreoffice/program/soffice"):
        _w(H / rel)
    for n, t in (("bin", "usr/bin"), ("lib", "usr/lib"), ("lib64", "usr/lib64"), ("sbin", "usr/sbin")):
        (H / n).symlink_to(t)
    for n in ctl.DEV_NODES:
        _w(H / "dev" / n)
    _w(H / "etc/passwd", b"root:x:0:0:root:/root:/bin/bash\n")
    _w(H / "etc/shadow", b"root:*:1::::::\n")
    # the dataset mirror (every golden!), the repo, runner caches, a leg's skills dir, an episode workspace
    m.canaries = []
    for rel in (f"{MACHINE_TREE}/SpreadsheetBench-v2/data/spreadsheetbench-v2/Template/spreadsheet/02_b/02_01_golden.xlsx",
                f"{MACHINE_TREE}/SpreadsheetBench-v2/data/spreadsheetbench-v2/Template/spreadsheet/03_c/03_01_golden.xlsx",
                f"{MACHINE_TREE}/SpreadsheetBench-v2/data/spreadsheetbench-v2/Template/dataset.json",
                f"{MACHINE_TREE}/EvoSkill/outputs/split_x/split.json",
                f"{MACHINE_TREE}/EvoSkill/src/agent_profiles/ssbv2_agent/scorer.py",
                f"{MACHINE_TREE}/EvoSkill/.cache/run_x_e1_cache/Template_02_01/rec.json",
                f"{MACHINE_TREE}/EvoSkill/outputs/run_x/skills_v1/some-skill/SKILL.md",
                f"{MACHINE_TREE}/EvoSkill/outputs/run_x/run_x_v1_val.json",
                "opt/evoskill/venv/bin/python", "opt/evoskill/verifier-lo-profile/4/user/registrymodifications.xcu",
                "opt/leg/leg.env", "opt/pool/units/v1_val/s01.t1/out/runner.log",
                "var/lib/evoshim/state/evoskill-sandbox-1.json", "var/lib/evoshim/priv/evoskill-sandbox-1/tmp/w.xlsx",
                "var/log/evoshim/containers.jsonl", "root/.config/libreoffice/4/user/x.xcu", "tmp/bundle.tgz",
                "workspace/episode.xlsx"):
        _w(H / rel)
        m.canaries.append("/" + rel)
    m.canaries += ["/opt/evoskill", "/opt/leg", "/opt/pool", "/var/lib/evoshim", "/root", "/workspace",
                   f"/{MACHINE_TREE}/SpreadsheetBench-v2", "/etc/shadow", ctl.BASE + "/cc"]
    # this controller's state (what `prepare` + `extract` leave)
    c = H / ctl.cdir(CID).lstrip("/")
    for d in ("root", "etc/claude-code", "fs/Users", "fs/evo/work", "fs/evo/out", "fs/etc/claude-code", "rw/home",
              "state", "fs" + STAGING):
        (c / d).mkdir(parents=True, exist_ok=True)
    _w(c / "etc/passwd", b"root:x:0:0::/root:/bin/bash\nevo:x:1000:1000::/home/evo:/bin/bash\n")
    _w(H / ctl.CC_HOST.lstrip("/"), b"ELF claude")
    m.uploaded = {f"{RUN_MIRROR}/ITER1_PROMPT.md": b"prompt", f"{RUN_MIRROR}/digests_e1_train/Template_01_01.md": b"d1",
                  f"{RUN_MIRROR}/grader_clean/scorer.py": b"code", f"{ctl.VM_WORK}/PROMPT.md": b"prompt",
                  f"{ctl.VM_ROOT}/evo.env": b"IS_SANDBOX=1\n", f"{ctl.VM_ROOT}/hooks/settings.json": b"{}",
                  f"{ctl.POLICY_DIR}/managed-settings.json": b"{}"}
    for p, b in m.uploaded.items():
        _w(c / ("fs" + p).lstrip("/"), b)
    m.c = c
    return m


def _join(*parts) -> str:
    return "/" + "/".join(x for p in parts for x in str(p).split("/") if x)


class Sim:
    """A private mount namespace replayed from the commands bash ran. fs "host" = the fake machine tree."""

    def __init__(self, H: Path):
        self.H = H
        self.mounts = [{"point": "/", "fs": "host", "base": "/", "ro": False}]
        self.tfs: dict = {}
        self.cwd = "/"
        self.violations: list = []
        self.tests: list = []
        self.pivoted = False
        self.exited = None

    def norm(self, p: str) -> str:
        return _join(self.cwd, p) if not p.startswith("/") else _join(p)

    def where(self, p: str):
        p = _join(p)
        best = max((m for m in self.mounts if ctl._under(p, m["point"])), key=lambda m: len(m["point"]))
        rest = p[len(best["point"]):] if best["point"] != "/" else p
        return best["fs"], _join(best["base"], rest), best

    def kind(self, fs: str, fsp: str):
        if fs == "host":
            hp = self.H / fsp.lstrip("/")
            return "link" if hp.is_symlink() else ("dir" if hp.is_dir() else ("file" if hp.exists() else None))
        if fs == "proc":
            return "dir" if fsp in ("/", "/sys", "/self") else "file"
        t = self.tfs[fs]
        return "link" if fsp in t["links"] else ("dir" if fsp in t["dirs"] else ("file" if fsp in t["files"] else None))

    def resolve(self, p: str, hops: int = 0) -> str:
        assert hops < 40
        parts = [x for x in p.split("/") if x]
        cur = "/"
        for i, part in enumerate(parts):
            cur = _join(cur, part)
            fs, fsp, _ = self.where(cur)
            if self.kind(fs, fsp) == "link":
                tgt = os.readlink(self.H / fsp.lstrip("/")) if fs == "host" else self.tfs[fs]["links"][fsp]
                nxt = tgt if tgt.startswith("/") else _join(os.path.dirname(cur), tgt)
                return self.resolve(_join(nxt, *parts[i + 1:]), hops + 1)
        return cur

    def lookup(self, p: str):
        r = self.resolve(self.norm(p))
        fs, fsp, m = self.where(r)
        return fs, fsp, m, self.kind(fs, fsp), r

    def _create(self, p: str, what: str, target=None):
        p = self.norm(p)
        fs, fsp, m = self.where(p)
        if fs in ("host", "proc"):
            self.violations.append((what, p, fs))
            return
        t = self.tfs[fs]
        parent = os.path.dirname(fsp)
        assert what == "mkdir -p" or parent in t["dirs"], f"{what} {p}: parent missing on {fs}"
        if what.startswith("mkdir"):
            parts = [x for x in fsp.split("/") if x]
            t["dirs"].update(_join(*parts[:i]) for i in range(len(parts) + 1))
        elif what == "touch":
            t["files"].add(fsp)
        else:
            t["links"][fsp] = target

    def add_mount(self, point, fs, base, ro):
        self.mounts = [m for m in self.mounts if not (ctl._under(m["point"], point) and m["point"] != point)]
        self.mounts.append({"point": point, "fs": fs, "base": base, "ro": ro})

    def run(self, cmds):
        for c in cmds:
            op, a = c[0], c[1:]
            if op in ("mount", "umount"):
                assert a[0] == "-n", f"{op} without -n writes an mtab/utab: {c}"
                a = a[1:]
            if op == "mkdir":
                self._create(a[-1], "mkdir -p" if a[0] == "-p" else "mkdir")
            elif op == "touch":
                self._create(a[0], "touch")
            elif op == "ln":
                assert a[0] == "-s"
                self._create(a[2], "ln", a[1])
            elif op == "cd":
                self.cwd = self.norm(a[0])
            elif op == "[":
                fs, fsp, m, k, r = self.lookup(a[1])
                ok = {"-d": k == "dir", "-f": k == "file", "-e": k is not None}[a[0]]
                self.tests.append((a[0], a[1], ok))
                if not ok and not a[1].split("/")[-1] in ("kcore", "keys", "timer_list", "sysrq-trigger"):
                    self.exited = 3                      # the real script: `|| { echo EVOCTL_FAIL ...; exit 3; }`
                    return self
            elif op == "mount" and a[:2] == ["--make-rprivate", "/"]:
                pass
            elif op == "mount" and a[0] == "--bind":
                sfs, sfsp, sm, sk, _ = self.lookup(a[1])
                dfs, dfsp, dm, dk, dres = self.lookup(a[2])
                assert sk is not None, f"bind source {a[1]} missing"
                assert dk is not None, f"bind target {a[2]} missing"
                assert (sk == "dir") == (dk == "dir"), f"bind type mismatch {a}"
                self.add_mount(dres, sfs, sfsp, sm["ro"] if sfs != "host" or sm["point"] != "/" else False)
            elif op == "mount" and a[0] == "-o" and a[1].startswith("remount"):
                pt = self.resolve(self.norm(a[2]))
                top = [m for m in self.mounts if m["point"] == pt]
                assert top, f"remount of {pt}, not a mount point"
                top[-1]["ro"] = "ro" in a[1].split(",")
            elif op == "mount" and a[:2] == ["-t", "tmpfs"]:
                fid = f"tmpfs{len(self.tfs)}:{a[4]}"
                dres = self.lookup(a[5])[4]
                assert self.lookup(a[5])[3] == "dir", f"tmpfs mount point {a[5]} missing"
                self.tfs[fid] = {"dirs": {"/"}, "files": set(), "links": {}}
                self.add_mount(dres, fid, "/", False)
            elif op == "mount" and a[:2] == ["-t", "proc"]:
                assert self.lookup(a[5])[3] == "dir"
                self.add_mount(self.lookup(a[5])[4], "proc", "/", False)
            elif op == "py":
                code = a[1] if len(a) > 1 else ""
                if "pivot_root" in code or "155" in code:
                    self.pivot()
            elif op == "umount":
                p = self.norm(a[-1])
                self.mounts = [m for m in self.mounts if not ctl._under(m["point"], p)]
            elif op == "rmdir":
                fs, fsp, m = self.where(self.norm(a[0]))
                self.tfs[fs]["dirs"].discard(fsp)
            else:
                raise AssertionError(f"simulator does not know {c}")
        return self

    def pivot(self):
        new_root = self.cwd
        assert any(m["point"] == new_root for m in self.mounts), "new root is not a mount point"
        assert self.lookup(_join(new_root, ".old_root"))[3] == "dir", "put_old missing"
        out = []
        for m in self.mounts:
            if ctl._under(m["point"], new_root):
                out.append({**m, "point": _join(m["point"][len(new_root):])})
            else:
                out.append({**m, "point": _join("/.old_root", m["point"])})
        self.mounts = out
        self.cwd = "/"
        self.pivoted = True

    def top(self):
        fs, fsp, m = self.where("/")
        t = self.tfs[fs]
        names = {x.strip("/").split("/")[0] for x in t["dirs"] | t["files"] | set(t["links"]) if x != "/"}
        names |= {m["point"].split("/")[1] for m in self.mounts if m["point"] != "/"}
        return sorted(names)


PRE = r"""exec 9>&1
_r() { { printf '\035'; printf '%s\037' "$@"; printf '\036'; } >&9; }
mount() { _r mount "$@"; }
mkdir() { _r mkdir "$@"; }
touch() { _r touch "$@"; }
ln() { _r ln "$@"; }
umount() { _r umount "$@"; }
rmdir() { _r rmdir "$@"; }
cd() { _r cd "$@"; }
[() { _r [ "$@"; return 0; }
function /usr/local/bin/python3 { _r py "$@"; }
exec() { _r exec "$@"; exit 0; }
"""


def bash_commands(script: str) -> list:
    r = subprocess.run(["/bin/bash", "-c", PRE + script], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    rows = [ch.split("\x1d", 1)[1].split("\x1f")[:-1] for ch in r.stdout.split("\x1e") if "\x1d" in ch]
    assert rows and rows[-1][0] == "exec", "the script did not reach its final exec"
    assert "EVOCTL_READY" in r.stdout
    return rows


def replay(machine, spec):
    pl, lim = plan_of(spec)
    script = ctl.render_init(spec, FAKE_FACTS, lim, pl)
    rows = bash_commands(script)
    sim = Sim(machine.H).run([r for r in rows if r[0] != "exec"])
    return sim, pl, rows


# ═════════════════════════════════════════════════════════════════════════ 1. the container's filesystem
def test_container_sees_exactly_the_plan_and_nothing_of_the_machine(machine):
    spec = spec_for()
    sim, pl, rows = replay(machine, spec)
    assert sim.pivoted and not sim.violations, sim.violations          # nothing created on the machine's disk
    assert all(ok for _f, _p, ok in sim.tests), [t for t in sim.tests if not t[2]]
    # the old root is gone: no machine mount remains in the namespace
    assert not [m for m in sim.mounts if m["point"].startswith("/.old_root")]
    assert sim.lookup("/.old_root")[3] is None
    # the exact mount table and top level
    pts = {m["point"]: m["ro"] for m in sim.mounts}
    want = dict(ctl.expected_mount_points(pl))
    assert set(pts) - {"/"} - set(ctl.PROC_MASKS) == set(want)
    assert all(pts[p] == ro for p, ro in want.items()), {p: (pts[p], ro) for p, ro in want.items() if pts[p] != ro}
    assert pts["/"] is True                                             # remounted read-only after the pivot
    assert sim.top() == ctl.expected_top(pl) == ["Users", "bin", "dev", "etc", "evo", "home", "lib", "lib64", "opt",
                                                   "proc", "run", "sbin", "tmp", "usr", "var"]
    # EVERY machine path the fixture planted is invisible (dataset goldens, split, repo, caches, skills dirs of
    # legs, val results, grader tooling, leg/pool scripts, shim state, /root, host /tmp, /workspace, /etc/shadow)
    visible = [p for p in machine.canaries if sim.lookup(p)[3] is not None]
    assert visible == []
    # what IS visible under /Users is exactly the upload set, served from the container's own fs/ dir
    for p in machine.uploaded:
        fs, fsp, m, k, _ = sim.lookup(p)
        assert (fs, k) == ("host", "file") and fsp.startswith(ctl.cdir(CID) + "/fs/"), (p, fs, fsp)
    for p, ro in ((f"{RUN_MIRROR}/digests_e1_train/Template_01_01.md", True), (STAGING, False),
                  (ctl.VM_WORK, False), (ctl.VM_OUT, False), ("/evo/hooks/settings.json", True),
                  ("/etc/claude-code/managed-settings.json", True), ("/usr/local/bin/python3", True),
                  ("/home/evo", False), ("/etc/passwd", True), (ctl.CC_CONTAINER, True)):
        fs, fsp, m, k, _ = sim.lookup(p)
        assert k is not None and m["ro"] is ro, (p, m)
    # system dirs through the image's own symlinks; the docker shim masked; Claude Code from its host path
    assert sim.lookup("/bin/bash")[:2] == ("host", "/usr/bin/bash")
    assert sim.lookup("/lib64/ld-linux-x86-64.so.2")[:2] == ("host", "/usr/lib64/ld-linux-x86-64.so.2")
    assert sim.lookup("/usr/local/bin/docker")[:2] == ("host", "/dev/null")
    assert sim.lookup(ctl.CC_CONTAINER)[:2] == ("host", ctl.CC_HOST)
    assert sim.lookup("/opt/evoskill")[3] is None and sim.lookup("/opt/claude-code")[3] == "dir"
    assert sim.lookup("/tmp")[0].startswith("tmpfs") and sim.lookup("/dev/shm")[0].startswith("tmpfs")
    assert sim.lookup("/proc/sys")[2]["ro"] is True and sim.lookup("/proc/kcore")[:2] == ("host", "/dev/null")
    assert sim.lookup("/sys")[3] is None and sim.lookup("/root")[3] is None
    # every machine source bound in: the image's system dirs, device nodes, the CC binary, the container's own dir
    allowed = ("/usr", "/dev/null", "/dev/zero", "/dev/full", "/dev/random", "/dev/urandom", "/dev/tty", ctl.CC_HOST)
    srcs = sorted({m["base"] for m in sim.mounts if m["fs"] == "host"})
    assert all(s.startswith(ctl.cdir(CID) + "/") or s.startswith(allowed) for s in srcs), srcs


def test_root_is_read_only_and_the_reaper_drops_everything(machine):
    sim, pl, rows = replay(machine, spec_for())
    # a BIND remount of / (per-mount flags): a plain superblock remount fails in a user-namespaced Daytona sandbox
    remounts = [r for r in rows if r[:4] == ["mount", "-n", "-o", "remount,bind,ro,nosuid,nodev"]]
    assert remounts and remounts[-1][-1] == "/"
    assert not [r for r in rows if r[:4] == ["mount", "-n", "-o", "remount,ro,nosuid,nodev"]]
    order = [r[0] if r[0] != "py" else ("pivot" if "pivot_root" in r[-1] else "py") for r in rows]
    assert order.index("pivot") > max(i for i, r in enumerate(rows) if r[:3] == ["mount", "-n", "--bind"])
    assert rows[0][:4] == ["mount", "-n", "--make-rprivate", "/"]
    ex = rows[-1]
    assert ex[1] == "/usr/bin/setpriv" and f"--reuid={ctl.REAPER_UID}" in ex and f"--regid={ctl.REAPER_GID}" in ex
    for f in ctl.DROP + ("--clear-groups",):
        assert f in ex
    assert ex[ex.index("--") + 1:ex.index("--") + 3] == ["/usr/bin/env", "-i"]
    assert ctl.INIT in ex


def test_every_mount_point_below_a_bind_is_checked_never_created(machine):
    spec = spec_for()
    pl, lim = plan_of(spec)
    script = ctl.render_init(spec, FAKE_FACTS, lim, pl)
    for m in pl:
        if m["mk"] == "exists" and not (m["kind"] == "mask" and m["dst"].startswith("/proc/")):
            assert f'[ {"-e" if m["kind"] == "mask" else ("-f" if m["node"] == "file" else "-d")} "$R"{shlex.quote(m["dst"])} ]' in script
            assert f'mkdir -p "$R"{shlex.quote(m["dst"])}' not in script
    # a missing staging dir: the check fails in the simulator (the real script exits 3 there)
    shutil.rmtree(machine.c / ("fs" + STAGING).lstrip("/"))
    sim, _pl, _rows = replay(machine, spec)
    assert sim.exited == 3 and not sim.pivoted
    assert [p for _f, p, ok in sim.tests if not ok] == [f"{ctl.cdir(CID)}/root{STAGING}"]


def test_ao_run_dir_is_writable_but_its_evidence_stays_read_only(machine):
    spec = ao_spec()
    for p in spec["ro_rebind"]:
        (machine.c / ("fs" + p).lstrip("/")).mkdir(parents=True, exist_ok=True)
    (machine.c / ("fs" + RUN_MIRROR).lstrip("/")).mkdir(parents=True, exist_ok=True)
    assert ctl.validate_spec(spec) == []
    sim, pl, rows = replay(machine, spec)
    assert sim.lookup(RUN_MIRROR)[2]["ro"] is False
    assert sim.lookup(f"{RUN_MIRROR}/digests_e1_train/Template_01_01.md")[2]["ro"] is True
    assert sim.lookup(f"{RUN_MIRROR}/oracle")[2]["ro"] is True


@pytest.mark.parametrize("mut,why", [
    (lambda s: s.update(cid="../x"), "bad cid"),
    (lambda s: s["rw"].append({"path": "/etc", "owner": "evo", "mode": "0755"}), "not strictly below /Users"),
    (lambda s: s["rw"].append({"path": STAGING + "/sub", "owner": "evo", "mode": "0755"}), "nests"),
    (lambda s: s["rw"].__setitem__(0, {"path": STAGING + "/../x", "owner": "evo", "mode": "0755"}), "plain absolute"),
    (lambda s: s["rw"].__setitem__(0, {"path": STAGING, "owner": "evo", "mode": "0777"}), "owner/mode"),
    (lambda s: s["readonly_roots"].append(RUN_MIRROR), "contains a rw path"),
    (lambda s: s["readonly_roots"].append(STAGING + "/x"), "not re-bound read-only"),
    (lambda s: s["ro_rebind"].append("/evo/elsewhere"), "not below a rw path"),
    (lambda s: s["limits"].update(bogus=1), "unknown limit"),
    (lambda s: s["cc"].update(sha256="0" * 64), "cc pin"),
    (lambda s: s["rw"].__setitem__(0, {"path": "/evo/a b", "owner": "evo", "mode": "0755"}), "plain absolute"),
])
def test_spec_refusals(mut, why):
    s = spec_for()
    mut(s)
    probs = ctl.validate_spec(s)
    assert any(why in p for p in probs), probs


def test_plan_refuses_odd_machine_facts():
    for facts in ({**FAKE_FACTS, "sysdirs": {"bin": "symlink:../../etc"}},
                  {**FAKE_FACTS, "sysdirs": {"opt": "dir"}}, {**FAKE_FACTS, "masks_present": ["/etc/passwd"]}):
        with pytest.raises(ValueError):
            plan_of(spec_for(), facts)


def test_limits_leave_the_rollouts_the_machine():
    lim = ctl.resolve_limits(FAKE_FACTS, {})
    assert lim["cpus"] == 2.0 and lim["cpu_max"] == "200000 100000" and lim["cpu_weight"] == 50
    # page-aligned (the kernel rounds memory.max/high down to 4096 B, so an unaligned value never reads back equal)
    assert lim["mem_max_bytes"] == int(0.2 * (32 << 30)) // 4096 * 4096
    assert lim["mem_high_bytes"] == int(lim["mem_max_bytes"] * 0.85) // 4096 * 4096
    assert lim["mem_max_bytes"] % 4096 == 0 and lim["mem_high_bytes"] % 4096 == 0
    # the sandbox's cgroup cpu quota wins over the host's cpu count (Daytona: nproc 48, cpu.max 8 cores)
    q = ctl.resolve_limits({**FAKE_FACTS, "ncpu": 48, "cg_root_cpu_max": "800000 100000"}, {})
    assert q["cpus"] == 2.0 and q["cpu_max"] == "200000 100000"
    assert lim["tmp_bytes"] <= 2 << 30 and lim["pids_max"] == 2048
    big = ctl.resolve_limits({**FAKE_FACTS, "ncpu": 32, "mem_total": 128 << 30}, {})
    assert big["cpus"] == 4.0 and big["mem_max_bytes"] == 8 << 30
    assert ctl.resolve_limits(FAKE_FACTS, {"cpus": 1.5, "mem_max_bytes": 3 << 30})["cpu_max"] == "150000 100000"
    assert pce.container_limits_cfg({"container": {"cpus": 3, "mem_max_gib": 4}}) == {"cpus": 3.0,
                                                                                    "mem_max_bytes": 4 << 30}
    with pytest.raises(ValueError):
        pce.container_limits_cfg({"container": {"cores": 3}})


# ═════════════════════════════════════════════════════════════════════════ 2. entering, and the token
def test_enter_argv_all_namespaces_no_privilege_clean_env():
    argv = ctl.enter_argv(4242, ["/bin/true"], "/evo/work")
    assert argv[:2] == ["nsenter", "--target=4242"]
    for f in ("--mount", "--pid", "--net", "--ipc", "--uts", "--root", "--wd"):
        assert f in argv[:argv.index("--")]
    sp = argv[argv.index("/usr/bin/setpriv"):]
    assert f"--reuid={ctl.EVO_UID}" in sp and f"--regid={ctl.EVO_GID}" in sp and "--clear-groups" in sp
    for f in ctl.DROP:
        assert f in sp
    env = argv[argv.index("-i") + 1:argv.index("/bin/bash")]
    assert env == list(ctl.EVO_ENV)
    assert not [e for e in env if any(k in e.upper() for k in ("TOKEN", "KEY", "SECRET", "ANTHROPIC"))]
    # pool_cmdwatch.py counts `nsenter ... 'echo $$; ...'` processes as SOLVER commands: never look like one
    assert not [a for a in argv if a.startswith("echo $$;")]
    assert argv[-2:] == ["/evo/work", "/bin/true"]


def test_enter_keeps_only_stdio_and_the_token_pipe(tmp_path, monkeypatch):
    cg = tmp_path / "cg"
    cg.mkdir()
    st = {"init_pid": 77, "init_start": 1, "cgroup": {"path": str(cg)}}
    monkeypatch.setattr(ctl, "load_all", lambda cid: ({}, {}, {}, {}, [], st))
    monkeypatch.setattr(ctl, "alive", lambda pid, start=None: True)
    seen = {}
    monkeypatch.setattr(ctl.os, "closerange", lambda lo, hi: seen.setdefault("closed_from", lo))

    def fake_exec(f, argv):
        seen["argv"] = argv
        raise SystemExit(0)
    monkeypatch.setattr(ctl.os, "execvp", fake_exec)
    for keep, want in ((True, 4), (False, 3)):
        seen.clear()
        a = SimpleNamespace(cid=CID, argv=["--", "/bin/true"], cwd="/tmp", keep_fd3=keep, log="")
        with pytest.raises(SystemExit):
            ctl.cmd_enter(a)
        assert seen["closed_from"] == want and seen["argv"][0] == "nsenter"
        assert (cg / "cgroup.procs").read_text() == str(os.getpid())


def test_token_command_shapes(monkeypatch):
    import evolver_container as EC
    r = SimpleNamespace(cid=CID, cdir=ctl.cdir(CID))
    fg = EC.ContainerMixin.enter_cmd(r, "claude -p x", cwd="/tmp", detached=False)
    bg = EC.ContainerMixin.enter_cmd(r, "claude -p x", cwd="/evo/work", detached=True)
    sec = EC.T.SECRET_ENV
    for cmd in (fg, bg):
        assert cmd.count(f'"${sec}"') == 1 and cmd.count(sec) == 2
        assert cmd.index(f"env -u {sec}") < cmd.index(ctl.CTL)
        assert f' 3<<<"${sec}"' in cmd and "--keep-fd3" in cmd
        assert "</dev/null" in cmd
    assert bg.startswith("(setsid nohup ") and bg.endswith("&)") and ">/dev/null 2>&1" in bg
    assert f"--log {ctl.cdir(CID)}/state/launch.err" in bg


# ═════════════════════════════════════════════════════════════════════════ 3. the egress proxy
def _eep():
    return ctl._import_eep()


def _serve(px):
    t = threading.Thread(target=px.serve_forever, daemon=True)
    t.start()
    return t


def _connect(port: int, req: bytes, then: bytes = b"") -> bytes:
    s = socket.create_connection(("127.0.0.1", port), timeout=10)
    s.sendall(req)
    buf = b""
    while b"\r\n\r\n" not in buf:
        b = s.recv(4096)
        if not b:
            break
        buf += b
    if then and buf.startswith(b"HTTP/1.1 200"):
        s.sendall(then)
        got = b""
        while len(got) < len(then):
            b = s.recv(4096)
            if not b:
                break
            got += b
        buf += b"|" + got
    s.close()
    return buf


class FakeUpstream:
    """A machine-side HTTP proxy (Daytona's egress proxy stand-in): records each CONNECT head; answers `code`;
    then echoes."""

    def __init__(self, code=b"200 Connection Established"):
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(8)
        self.port = self.srv.getsockname()[1]
        self.heads, self.code = [], code
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while True:
            try:
                c, _ = self.srv.accept()
            except OSError:
                return
            buf = b""
            while b"\r\n\r\n" not in buf:
                b = c.recv(4096)
                if not b:
                    break
                buf += b
            self.heads.append(buf.decode("latin-1"))
            c.sendall(b"HTTP/1.1 " + self.code + b"\r\n\r\n")
            if self.code.startswith(b"200"):
                while True:
                    b = c.recv(4096)
                    if not b:
                        break
                    c.sendall(b)
            c.close()


def test_proxy_allowlist_and_chained_connect_with_auth_never_logged(tmp_path):
    eep = _eep()
    Chain = ctl.make_chain_proxy(eep)
    up = FakeUpstream()
    logf = tmp_path / "egress.jsonl"
    px = Chain("api.anthropic.com", str(logf), 60.0, 0, 443)
    px.upstream = ctl.upstream_from_env({"HTTPS_PROXY": f"http://user:s3cr3tpw@127.0.0.1:{up.port}"})
    _serve(px)
    ok = _connect(px.port, b"CONNECT api.anthropic.com:443 HTTP/1.1\r\nHost: api.anthropic.com:443\r\n\r\n", b"hello")
    assert ok.startswith(b"HTTP/1.1 200") and ok.endswith(b"|hello")
    assert up.heads[0].startswith("CONNECT api.anthropic.com:443 HTTP/1.1\r\n")
    assert "Proxy-Authorization: Basic " + base64.b64encode(b"user:s3cr3tpw").decode() in up.heads[0]
    n_up = len(up.heads)
    for req, code in ((b"CONNECT openrouter.ai:443 HTTP/1.1\r\n\r\n", b"403"),
                      (b"CONNECT api.anthropic.com:80 HTTP/1.1\r\n\r\n", b"403"),
                      (b"CONNECT API.ANTHROPIC.COM.evil.com:443 HTTP/1.1\r\n\r\n", b"403"),
                      (b"CONNECT 160.79.104.10:443 HTTP/1.1\r\n\r\n", b"403"),
                      (b"GET http://api.anthropic.com/ HTTP/1.1\r\nHost: api.anthropic.com\r\n\r\n", b"403"),
                      (b"CONNECT api.anthropic.com HTTP/1.1\r\n\r\n", b"400"),
                      (b"garbage\r\n\r\n", b"400")):
        assert _connect(px.port, req).split(b" ")[1] == code, req
    time.sleep(0.3)
    assert len(up.heads) == n_up, "a refused request reached the upstream"
    px.stop()
    text = logf.read_text()
    assert "s3cr3tpw" not in text and "Proxy-Authorization" not in text
    rows = [json.loads(ln) for ln in text.splitlines()]
    assert sorted({r["host"] for r in rows if r["allowed"]}) == ["api.anthropic.com"]
    assert sum(1 for r in rows if not r["allowed"]) == 7
    assert ctl.egress_summary(str(logf))["by_host"]["api.anthropic.com:allowed"] == 1


def test_proxy_upstream_refusal_is_a_502_and_direct_mode(tmp_path):
    eep = _eep()
    Chain = ctl.make_chain_proxy(eep)
    up = FakeUpstream(code=b"403 Forbidden")
    px = Chain("api.anthropic.com", str(tmp_path / "e.jsonl"), 60.0, 0, 443)
    px.upstream = {"host": "127.0.0.1", "port": up.port, "auth": None}
    _serve(px)
    assert _connect(px.port, b"CONNECT api.anthropic.com:443 HTTP/1.1\r\n\r\n").split(b" ")[1] == b"502"
    px.stop()
    echo = FakeUpstream()                               # a plain echo server after its fake "200" line
    px2 = Chain("localhost", str(tmp_path / "e2.jsonl"), 60.0, 0, echo.port)
    _serve(px2)
    got = _connect(px2.port, f"CONNECT localhost:{echo.port} HTTP/1.1\r\n\r\n".encode())
    assert got.startswith(b"HTTP/1.1 200")
    px2.stop()


def test_upstream_from_env():
    assert ctl.upstream_from_env({}) is None
    u = ctl.upstream_from_env({"https_proxy": "http://10.1.2.3:3128"})
    assert u == {"host": "10.1.2.3", "port": 3128, "auth": None}
    assert ctl.upstream_from_env({"HTTPS_PROXY": "10.1.2.3:8080"})["port"] == 8080
    with pytest.raises(ValueError):
        ctl.upstream_from_env({"HTTPS_PROXY": "https://10.1.2.3:443"})


# ═════════════════════════════════════════════════════════════════════════ 4. registry + lease
def _row(**kw):
    r = {"sandbox_id": "aaaa1111", "sandbox_name": pce.POOL_PREFIX + "v1pl-m1", "state": "ready",
         "created_ts": 1000.0, "ttl_minutes": 1440, "domain_allow_list": "openrouter.ai,api.anthropic.com",
         "evolver_container": True, "snapshot_ref": "snap:1"}
    r.update(kw)
    return r


def test_registry_pick_machine():
    now, need = 2000.0, 9000.0
    reg = {"schema": pce.REGISTRY_SCHEMA, "machines": [_row()]}
    row, why = pce.pick_machine(reg, now=now, need_s=need)
    assert row["sandbox_id"] == "aaaa1111"
    for bad, word in ((dict(state="retiring"), "state"), (dict(evolver_container=False), "evolver_container"),
                      (dict(domain_allow_list="openrouter.ai"), "allow list"), (dict(ttl_minutes=100), "TTL")):
        row, why = pce.pick_machine({"schema": pce.REGISTRY_SCHEMA, "machines": [_row(**bad)]}, now=now, need_s=need)
        assert row is None and word in why, why
    reg2 = {"schema": pce.REGISTRY_SCHEMA, "machines": [_row(), _row(sandbox_id="bbbb2222", ttl_minutes=2000)]}
    assert pce.pick_machine(reg2, now=now, need_s=need)[0]["sandbox_id"] == "bbbb2222"
    assert pce.pick_machine(reg2, now=now, need_s=need, sandbox_id="aaaa1111")[0]["sandbox_id"] == "aaaa1111"
    assert pce.pick_machine({"schema": "other"}, now=now, need_s=need)[0] is None
    assert pce.parse_registry({"schema": pce.REGISTRY_SCHEMA, "machines": [{"x": 1}, _row()]})[1]
    # allow-list order and case do not matter; an extra domain does
    assert pce.pick_machine({"schema": pce.REGISTRY_SCHEMA, "machines": [
        _row(domain_allow_list="API.anthropic.com, openrouter.ai")]}, now=now, need_s=need)[0]
    assert pce.pick_machine({"schema": pce.REGISTRY_SCHEMA, "machines": [
        _row(domain_allow_list="openrouter.ai,api.anthropic.com,pypi.org")]}, now=now, need_s=need)[0] is None


def test_machine_problems_and_lease():
    rb = {"refreshed": True, "domain_allow_list": "openrouter.ai,api.anthropic.com", "network_allow_list": None,
          "network_block_all": False, "snapshot_ref": "snap:1"}
    ok = pce.machine_problems(rb, name=pce.POOL_PREFIX + "x", state="SandboxState.STARTED", row=_row(),
                              require_snapshot_ref="snap:1")
    assert ok == []
    assert pce.machine_problems(rb, name=pce.SMOKE_PREFIX + "x", state="started", row=None, require_snapshot_ref=None)
    assert pce.machine_problems({**rb, "domain_allow_list": "openrouter.ai"}, name=pce.POOL_PREFIX, state="started",
                                row=None, require_snapshot_ref=None)
    assert pce.machine_problems({**rb, "network_block_all": True}, name=pce.POOL_PREFIX, state="started", row=None,
                                require_snapshot_ref=None)
    assert pce.machine_problems(rb, name=pce.POOL_PREFIX, state="started", row=None, require_snapshot_ref="snap:2")
    le = pce.lease_record(sandbox_id="aaaa1111", cid=CID, run="r", epoch=1, pid=os.getpid(), expires_ts=time.time() + 60)
    assert pce.lease_live(le, now=time.time(), pid_alive=lambda p: True)
    assert not pce.lease_live(le, now=time.time() + 120, pid_alive=lambda p: True)
    assert not pce.lease_live(le, now=time.time(), pid_alive=lambda p: False)


def test_placement_values():
    assert pce.placement({}) == "own_sandbox" and pce.placement({"placement": "pool_container"}) == "pool_container"
    with pytest.raises(ValueError):
        pce.placement({"placement": "pool"})


# ═════════════════════════════════════════════════════════════════════════ 4b. the RUN POOL's machine.json + leases
def pool_row(sid="aaaa1111bbbb", name=pce.POOL_PREFIX + "rpabcde-m01-120000", **kw):
    """A machine as ssb_leg_pool.Machine.public_record() shows it (a ready machine, 285 min to retire_at)."""
    now = time.time()
    r = {"sandbox_id": sid, "sandbox_name": name, "state": "ready", "created_ts": now - 600, "ready_ts": now - 300,
         "ttl_minutes": 600, "ttl_expires_ts": now + 540 * 60, "retire_at_ts": now + 285 * 60,
         "domain_allow_list": ["openrouter.ai", "api.anthropic.com"], "units_running": 57, "inflight": 120,
         "cap": 120, "disk_free_gib": 30.0, "leased_by": []}
    r.update(kw)
    return r


def machines_json(machines: dict, current="m01", **kw) -> dict:
    """ssb_leg_pool.PoolHost.machines_record()'s shape."""
    d = {"schema": P.MACHINES_SCHEMA, "run": "r", "pool_root": "/x", "pool_dir": "/x", "host_pid": os.getpid(),
         "host_state": "open", "current": current, "machines": machines, "leases_dir": "/x/leases", "generation": 1}
    d.update(kw)
    return d


class FakePoolHost:
    """The pool host's side of the registry, with the POOL's OWN code: every tick the leases are read with
    ssb_leg_pool.Registry.leases() and machine.json is rewritten with each machine's leased_by. `script(host, t)`
    may change the machines between ticks."""

    def __init__(self, root: Path, machines: dict, current="m01", tick=0.02, script=None):
        self.root, self.machines, self.current, self.tick, self.script = Path(root), machines, current, tick, script
        self.reg = P.Registry(self.root)
        self.gen, self.last, self.lease_log = 0, None, []
        self._stop = threading.Event()
        self.publish()

    def publish(self):
        leases = self.reg.leases(time.time())
        ms = {n: dict(r, leased_by=leases.get(str(r.get("sandbox_id")), [])) for n, r in self.machines.items()}
        self.gen += 1
        self.last = machines_json(ms, current=self.current, generation=self.gen)
        if leases:
            self.lease_log.append(leases)
        pce.write_json_atomic(self.root / P.MACHINES_FILE, self.last)

    def loop(self):
        t0 = time.time()
        while not self._stop.wait(self.tick):
            if self.script:
                self.script(self, time.time() - t0)
            self.publish()

    def start(self):
        self.th = threading.Thread(target=self.loop, daemon=True)
        self.th.start()
        return self

    def stop(self):
        self._stop.set()
        self.th.join(timeout=5)


def test_registry_constants_and_paths_are_the_pools_own(tmp_path):
    assert (pce.POOL_MACHINES_SCHEMA, pce.POOL_MACHINES_FILE, pce.POOL_LEASES_DIR, pce.POOL_SCOPE) == \
        (P.MACHINES_SCHEMA, P.MACHINES_FILE, P.LEASES_DIR, P.POOL_SCOPE)
    assert pce.POOL_SYNC_MARGIN_MIN == M.POOL_SYNC_MARGIN_MIN
    if hasattr(P, "PLACEMENTS"):          # the pool's egress by placement (final integration): the same two domains
        assert tuple(P.PLACEMENTS) == pce.PLACEMENTS and P.EVOLVER_DOMAIN == pce.API_HOST
        assert frozenset(P.allow_domains(pce.PLACEMENT)) == pce.MACHINE_ALLOW
        assert pce.API_HOST not in P.allow_domains(pce.OWN_SANDBOX)
    rd = tmp_path / "outputs" / "runq"
    for rt, store in (({}, M.DEFAULTS["store_root"]), ({"daytona_leg": {"store_root": str(tmp_path / "st")}},
                                                        str(tmp_path / "st"))):
        root = P.pool_root(SimpleNamespace(rt={"store_root": store}, run_dir=rd))
        assert pce.pool_run_dir(rd, rt) == root
        assert pce.pool_machines_file(rd, rt) == root / P.MACHINES_FILE
    reg = P.Registry(tmp_path / "pool" / "run")
    assert (reg.machines_file, reg.leases_dir) == (tmp_path / "pool/run" / pce.POOL_MACHINES_FILE,
                                                   tmp_path / "pool/run" / pce.POOL_LEASES_DIR)


def test_pick_pool_machine_takes_the_ready_current_machine_only():
    now, need = time.time(), 7200.0 + pce.EVOLVER_MARGIN_S
    ok = machines_json({"m01": pool_row()})
    row, why = pce.pick_pool_machine(ok, now=now, need_s=need)
    assert row["sandbox_id"] == "aaaa1111bbbb" and row["name"] == "m01", why
    # the SDK's comma string works as well as the pool's list
    assert pce.pick_pool_machine(machines_json({"m01": pool_row(domain_allow_list="openrouter.ai,API.anthropic.com")}),
                                 now=now, need_s=need)[0]
    for rec, word in (
            (machines_json({"m01": pool_row(state="starting")}), "'starting'"),
            (machines_json({"m01": pool_row(state="retiring")}), "'retiring'"),
            (machines_json({"m01": pool_row(state="released")}), "'released'"),
            (machines_json({"m01": pool_row()}, current=None), "no current machine"),
            (machines_json({"m01": pool_row(state="starting")}, current=None), "'starting'"),
            (machines_json({"m01": pool_row()}, current="m02"), "no row"),
            (machines_json({"m01": pool_row()}, host_state="closed"), "'closed'"),
            (machines_json({"m01": pool_row()}, schema="evoskill-pool-machines/1"), "schema"),
            ({"schema": pce.REGISTRY_SCHEMA, "machines": [{"sandbox_id": "x"}]}, "schema"),
            (machines_json({"m01": pool_row(retire_at_ts="soon")}), "retire_at_ts 'soon'"),
            (machines_json({"m01": pool_row(retire_at_ts=None)}), "retire_at_ts None"),
            (machines_json({"m01": pool_row(ttl_expires_ts=now + need)}), "honoured lease"),
            (machines_json({"m01": pool_row(sandbox_id=None)}), "no row with a sandbox_id")):
        row, why = pce.pick_pool_machine(rec, now=now, need_s=need)
        assert row is None and word in why, (word, why)
    row, why = pce.pick_pool_machine(ok, now=now, need_s=need, alive=lambda pid: False)
    assert row is None and "not alive" in why
    assert pce.pick_pool_machine(ok, now=now, need_s=need, sandbox_id="aaaa1111bbbb")[0]
    row, why = pce.pick_pool_machine(ok, now=now, need_s=need, sandbox_id="cccc")
    assert row is None and "not the requested" in why
    assert pce.pick_pool_machine("not json", now=now, need_s=need)[0] is None


def test_pick_pool_machine_refuses_a_machine_that_retires_before_the_evolver_ends():
    """The pool's rule: the holder checks retire_at_ts before starting work that would outlive it (a
    lease past it holds up the replacement and queued rollouts wait). Needed: evolver timeout + 30 min."""
    now, timeout = time.time(), 7200
    need = timeout + pce.EVOLVER_MARGIN_S
    edge = machines_json({"m01": pool_row(retire_at_ts=now + need + 5)})
    assert pce.pick_pool_machine(edge, now=now, need_s=need)[0], "exactly enough is enough"
    # the builder's own simulation (TTL 600, margin 255): the v1 train result lands at ~206 min, when the first
    # machine has 139 min to retire_at -- less than 150: REFUSED, whatever else is right about it
    late = machines_json({"m01": pool_row(retire_at_ts=now + 139 * 60)})
    row, why = pce.pick_pool_machine(late, now=now, need_s=need)
    assert row is None and "to retire_at_ts < 9000 s" in why, why
    assert "retire_at" not in pce.pick_pool_machine(machines_json({"m01": pool_row(retire_at_ts=now + 139 * 60)}),
                                                    now=now, need_s=3600 + pce.EVOLVER_MARGIN_S)[1].split(":")[0]


def test_pick_pool_machine_refuses_an_allow_list_without_api_anthropic():
    now, need = time.time(), 9000.0
    for al in (["openrouter.ai"], "openrouter.ai", [], None, ["openrouter.ai", "api.anthropic.com.evil"],
               ["anthropic.com"]):
        row, why = pce.pick_pool_machine(machines_json({"m01": pool_row(domain_allow_list=al)}), now=now, need_s=need)
        assert row is None and "lacks api.anthropic.com" in why, (al, why)


def test_pool_lease_is_the_pools_format_and_the_pool_honours_it(tmp_path):
    root = tmp_path / "pool" / "run"
    reg = P.Registry(root)
    cap = time.time() + 3600
    lease = pce.PoolLease(reg.leases_dir, holder=pce.lease_holder(CID), sandbox_id="aaaa1111bbbb", cap_ts=cap,
                          info={"cid": CID, "epoch": 1}, window_s=30, every_s=0.05)
    rec = lease.write()
    on_disk = json.loads((reg.leases_dir / f"evolver-{CID}.json").read_text())
    assert on_disk == rec and {"holder", "sandbox_id", "until_ts", "pid"} <= set(on_disk)
    assert on_disk["pid"] == os.getpid() and on_disk["holder"] == f"evolver-{CID}"
    assert time.time() + 25 < on_disk["until_ts"] <= time.time() + 30
    assert reg.leases(time.time()) == {"aaaa1111bbbb": [f"evolver-{CID}"]}           # THE POOL'S OWN READER
    assert not list(reg.leases_dir.glob(".*"))                                        # no tmp left beside it
    # renewed from its thread
    u0 = on_disk["until_ts"]
    lease.start()
    time.sleep(0.4)
    assert json.loads(lease.path.read_text())["until_ts"] > u0 and lease.writes >= 3
    # never past the cap
    capped = pce.PoolLease(reg.leases_dir, holder="evolver-capped", sandbox_id="x", cap_ts=time.time() + 5,
                           window_s=900)
    assert capped.write()["until_ts"] <= time.time() + 5
    capped.release()
    # released: gone, and the pool sees none
    assert lease.release() is True and not lease.path.exists()
    assert reg.leases(time.time()) == {}
    w = lease.writes
    time.sleep(0.2)
    assert lease.writes == w and not lease.path.exists(), "no renewal after release"


def test_pool_lease_of_a_dead_driver_is_void_and_another_holders_file_is_kept(tmp_path):
    root = tmp_path / "pool" / "run"
    reg = P.Registry(root)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    gone = pce.PoolLease(reg.leases_dir, holder="evolver-dead", sandbox_id="s1", cap_ts=time.time() + 3600,
                         pid=dead.pid)
    gone.write()
    assert reg.leases(time.time()) == {}, "a lease whose pid is dead holds nothing (the pool's rule)"
    mine = pce.PoolLease(reg.leases_dir, holder="evolver-mine", sandbox_id="s1", cap_ts=time.time() + 3600)
    mine.write()
    pce.write_json_atomic(mine.path, {"holder": "someone-else", "sandbox_id": "s1", "until_ts": time.time() + 60})
    assert mine.release() is False and mine.path.exists(), "a file that is no longer ours is never removed"


def test_lease_acknowledged_verdicts():
    h, sid = "evolver-x", "aaaa1111bbbb"
    assert pce.lease_acknowledged(machines_json({"m01": pool_row(leased_by=[h])}), name="m01", sandbox_id=sid,
                                  holder=h)[0] is True
    assert pce.lease_acknowledged(machines_json({"m01": pool_row(leased_by=[])}), name="m01", sandbox_id=sid,
                                  holder=h)[0] is None
    for rec in (machines_json({"m01": pool_row(leased_by=[h], state="released")}),
                machines_json({"m01": pool_row(leased_by=[h], state="retiring")}),
                machines_json({"m02": pool_row(leased_by=[h])}),
                machines_json({"m01": pool_row(sandbox_id="other", leased_by=[h])}),
                machines_json({"m01": pool_row(leased_by=[h])}, host_state="closed"), None):
        assert pce.lease_acknowledged(rec, name="m01", sandbox_id=sid, holder=h)[0] is False, rec


def test_machine_wait_values():
    assert pce.machine_wait_s({}) == pce.MACHINE_WAIT_S and pce.machine_wait_s({"machine_wait_s": 0}) == 0
    for bad in (-1, 3601, "900", True, None, float("nan")):
        with pytest.raises(ValueError):
            pce.machine_wait_s({"machine_wait_s": bad})


def _attach_run(tmp_path, *, cfg=None, timeout_s=7200):
    """Just enough of a ContainerEvolverRun for attach_machine / release_lease (its pool under tmp_path)."""
    import evolver_container as EC
    r = object.__new__(EC.ContainerEvolverRun)
    r.epoch, r.timeout_s, r.cfg = 1, timeout_s, dict(cfg or {})
    r.run_dir = tmp_path / "outputs" / "runq"
    r.run_dir.mkdir(parents=True, exist_ok=True)
    (r.run_dir / "RUNTIME.json").write_text(json.dumps({"daytona_leg": {"store_root": str(tmp_path / "store")}}))
    r.receipt = {"daytona": {}}
    r.receipt_path = tmp_path / "receipt.json"
    r.cinit()
    return r


def test_attach_waits_for_the_rollover_replacement_then_leases_it(tmp_path, monkeypatch):
    """The pool rolls an idle, short-of-time machine over right when a train leg ends -- when the evolver starts:
    the replacement is `starting` for minutes. The driver re-reads machine.json until it is ready, then leases it."""
    monkeypatch.setattr(pce, "MACHINE_POLL_S", 0.02)
    monkeypatch.setattr(pce, "LEASE_ACK_S", 2.0)
    r = _attach_run(tmp_path, cfg={"machine_wait_s": 5})
    root = pce.pool_run_dir(r.run_dir)
    now = time.time()

    def script(host, t):
        if t > 0.3:
            host.machines["m02"]["state"] = "ready"
            host.current = "m02"
    host = FakePoolHost(root, {"m01": pool_row(state="deleted", retire_at_ts=now + 60),
                               "m02": pool_row(sid="bbbb2222", state="starting")}, current=None,
                        script=script).start()
    try:
        row = r.attach_machine(pce.pool_machines_file(r.run_dir))
        assert row["sandbox_id"] == "bbbb2222" and row["name"] == "m02"
        pm, le = r.receipt["daytona"]["pool_machine"], r.receipt["daytona"]["lease"]
        assert pm["format"] == "pool" and pm["waited_s"] >= 0.2 and pm["attempts"]
        assert "no current machine" in pm["attempts"][0]["why"]
        assert le["acknowledged"] is True and le["sandbox_id"] == "bbbb2222"
        time.sleep(0.1)
        assert host.last["machines"]["m02"]["leased_by"] == [pce.lease_holder(r.cid)]
        r.release_lease()
        assert r.receipt["daytona"]["lease"]["released"] is True
        time.sleep(0.1)
        assert host.last["machines"]["m02"]["leased_by"] == [] and P.Registry(root).leases(time.time()) == {}
    finally:
        host.stop()


def test_attach_refuses_after_the_wait_and_leaves_no_lease(tmp_path, monkeypatch):
    import evolver_container as EC
    monkeypatch.setattr(pce, "MACHINE_POLL_S", 0.02)
    r = _attach_run(tmp_path, cfg={"machine_wait_s": 0.2})
    root = pce.pool_run_dir(r.run_dir)
    host = FakePoolHost(root, {"m01": pool_row(retire_at_ts=time.time() + 100 * 60)}).start()
    try:
        with pytest.raises(EC.Refuse) as ei:
            r.attach_machine(pce.pool_machines_file(r.run_dir))
        assert ei.value.code == EC.EXIT_IDENTITY and "retire_at_ts" in str(ei.value)
        assert r.receipt["daytona"]["pool_machine"]["row"] is None
        assert json.loads(r.receipt_path.read_text())["daytona"]["pool_machine"]["why"]
        assert not list((root / pce.POOL_LEASES_DIR).glob("*.json")) and host.lease_log == []
        # no machine.json at all (the pool host has not written one yet): also a refusal after the wait
        r2 = _attach_run(tmp_path / "b", cfg={"machine_wait_s": 0})
        with pytest.raises(EC.Refuse) as ei:
            r2.attach_machine(pce.pool_machines_file(r2.run_dir))
        assert "no readable pool machine.json" in str(ei.value)
    finally:
        host.stop()


def test_attach_refuses_a_lease_the_host_never_lists_and_repicks_after_a_release_race(tmp_path, monkeypatch):
    import evolver_container as EC
    monkeypatch.setattr(pce, "MACHINE_POLL_S", 0.02)
    monkeypatch.setattr(pce, "LEASE_ACK_S", 0.3)
    # (a) a host that publishes machine.json but never reads leases/ (e.g. a pool without the lease feature)
    r = _attach_run(tmp_path / "a", cfg={"machine_wait_s": 0})
    root = pce.pool_run_dir(r.run_dir)
    root.mkdir(parents=True)
    pce.write_json_atomic(root / pce.POOL_MACHINES_FILE, machines_json({"m01": pool_row()}))
    with pytest.raises(EC.Refuse) as ei:
        r.attach_machine(pce.pool_machines_file(r.run_dir))
    assert "not acknowledged" in str(ei.value) and not list((root / pce.POOL_LEASES_DIR).glob("*.json"))
    # (b) the host released m01 in the tick that saw no lease yet: the lease goes, and the pick starts over
    r = _attach_run(tmp_path / "b", cfg={"machine_wait_s": 5})
    root = pce.pool_run_dir(r.run_dir)
    state = {"released": False}

    def script(host, t):
        if host.reg.leases(time.time()) and not state["released"]:
            state["released"] = True
            host.machines["m01"]["state"] = "released"
            host.machines["m02"] = pool_row(sid="bbbb2222")
            host.current = "m02"
    host = FakePoolHost(root, {"m01": pool_row()}, script=script).start()
    try:
        row = r.attach_machine(pce.pool_machines_file(r.run_dir))
        assert row["name"] == "m02" and state["released"]
        assert any("not acknowledged" in a["why"] and "released" in a["why"]
                   for a in r.receipt["daytona"]["pool_machine"]["attempts"])
        assert [p.name for p in (root / pce.POOL_LEASES_DIR).glob("*.json")] == [f"{pce.lease_holder(r.cid)}.json"]
        r.release_lease()
    finally:
        host.stop()


def test_the_loop_check_wants_the_pools_acknowledged_lease():
    """check_pool_container_receipt's machine half, on its own (the full receipt passes in the end-to-end test)."""
    base = {"containment": "daytona", "sandboxed": True, "daytona": {"placement": pce.PLACEMENT, "sandbox_id": "s1"}}

    def machine_probs(d, **kw):
        ok, det = pce.check_pool_container_receipt({**base, "daytona": {**base["daytona"], **d}}, log="/nonexistent",
                                                   ledger_path="/nonexistent", **kw)
        return det
    good = {"pool_machine": {"format": "pool", "row": {"sandbox_id": "s1", "state": "ready"}},
            "lease": {"acknowledged": True, "sandbox_id": "s1"}}
    assert "pool machine:" not in machine_probs(good)
    assert "never listed" in machine_probs({**good, "lease": {"acknowledged": False, "sandbox_id": "s1"}})
    assert "not the ready machine.json row" in machine_probs({**good, "pool_machine": {
        "format": "pool", "row": {"sandbox_id": "s2", "state": "ready"}}})
    assert "not the run pool's machine.json" in machine_probs({"pool_machine": {"format": "smoke"}})
    assert "pool machine:" not in machine_probs({"pool_machine": {"format": "smoke"}},
                                                prefixes=(pce.POOL_PREFIX, pce.SMOKE_PREFIX))


# ═════════════════════════════════════════════════════════════════════════ 5. the probe verdicts
CNS = {n: f"{n}:[40265320{i}]" for i, n in enumerate(("mnt", "pid", "net", "ipc", "uts"))}
CNS.update(user="user:[4026531837]", cgroup="cgroup:[4026531835]")
HNS = {n: f"{n}:[40265310{i}]" for i, n in enumerate(("mnt", "pid", "net", "ipc", "uts"))}
HNS.update(user="user:[4026531837]", cgroup="cgroup:[4026531835]")


def good_probe(pl, spec) -> dict:
    rw = ["/tmp", "/var/tmp", ctl.EVO_HOME, ctl.VM_WORK, ctl.VM_OUT] + [r["path"] for r in spec["rw"]]
    ro = ["/", "/usr", "/usr/local/bin", "/etc", ctl.POLICY_DIR, "/Users", ctl.VM_ROOT, "/opt", ctl.CC_DIR_CONTAINER,
          "/dev", "/run", "/var", "/home", "/proc/sys"] + list(spec["ro_rebind"])
    px = {k: ctl.PROXY_URL for k in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy")}
    px.update(NO_PROXY="", no_proxy="")
    tcp = ["1.1.1.1:443", "8.8.8.8:443", "169.254.169.254:80", "example.com:22", "[2606:4700:4700::1111]:443",
           "openrouter 104.18.2.115:443", "anthropic-direct 160.79.104.10:443", "machine 10.0.0.5:22",
           "loopback 127.0.0.1:2280"]
    return {
        "phase": "pre",
        "identity": {"uid": 1000, "euid": 1000, "gid": 1000, "groups": [],
                     "caps": {k: pce.ZERO_CAPS for k in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")},
                     "no_new_privs": "1", "seccomp": "0", "arch": "x86_64", "hostname": "evolver", "user": "evo",
                     "libreoffice": "LibreOffice 25.2.3.2 520(Build:2)", "cc_sha256": de.CC_LINUX_X64_SHA,
                     "claude_on_path": ctl.CC_CONTAINER, "docker_on_path": None, "cc_version": "2.1.281 (Claude Code)",
                     "solver_stack": "3.1.3 2.2.0 1.26.4 3.11.16", "soffice_convert_ok": True,
                     "opt_entries": ["claude-code"], "users_entries": 1, "users_files": 0, "secret_env_names": [],
                     "proxy_env": px,
                     "evo_lo_profile_owner": "evo", "policy_dir_entries": 0, "files_named_claude": [ctl.CC_CONTAINER]},
        "view": {"top": ctl.expected_top(pl), "home": ["evo"], "var": ["tmp"], "run": [],
                 "dev": sorted(pce.DEV_ALLOWED), "opt": ["claude-code"],
                 "mounts": [{"point": "/", "ro": True}] + [{"point": p, "ro": r} for p, r in
                                                          ctl.expected_mount_points(pl)]
                           + [{"point": m, "ro": False} for m in ctl.PROC_MASKS],
                 "canaries_checked": 5321, "canaries_visible_n": 0, "canaries_visible": [],
                 "procs": [{"pid": 1, "comm": "python3", "start": 1, "uid": 65534},
                           {"pid": 9, "comm": "python3", "start": 7, "uid": 1000}],
                 "host_pid_matches": [], "ns": dict(CNS)},
        "writes": {"rw": {d: "WRITABLE" for d in rw}, "ro": {d: "EROFS" for d in ro}},
        "priv": {"setuid0": "EPERM", "chroot": "EPERM", "mount_rc": 32, "umount_users_rc": 32, "kill_pid1": "EPERM",
                 "ptrace_pid1": "EPERM", "pid1_environ": "EACCES", "kcore": "READ 0 B", "sysrq": "READ 0 B",
                 "proc_sys_write": "EROFS", "userns_rc": 0, "userns_view_equal": True},
        "net": {"ifaces": ["lo"], "routes": 0, "inet6": ["lo"],
                "http": [{"url": u, "reachable": True, "detail": "HTTP 401"} for u in de.MUST_REACH]
                + [{"url": u, "reachable": False, "detail": "URLError: Tunnel connection failed: 403"}
                   for u in de.MUST_REFUSE],
                "direct": [{"target": t, "reached": False, "detail": "URLError: gaierror"} for t in
                           ("https://example.com", "https://1.1.1.1", "https://github.com", "https://openrouter.ai",
                            "https://api.anthropic.com")],
                "raw": [{"target": t, "tcp_connected": False, "detail": "ENETUNREACH"} for t in tcp]
                + [{"target": "udp 1.1.1.1:53", "answered": False}, {"target": "udp 8.8.8.8:53", "answered": False}],
                "abstract": [{"name": "containerd-shim", "connected": False}],
                "dns": {"example.com": "FAILED: gaierror", "api.anthropic.com": "FAILED: gaierror",
                        "openrouter.ai": "FAILED: gaierror"}},
        "host": {"machine_ns": dict(HNS), "container_ns": dict(CNS), "here_string_fd": "pipe:[88123]",
                 "proxy": {"listen_netns": CNS["net"], "allow": ["api.anthropic.com"],
                           "upstream": {"kind": "http-proxy", "host": "10.0.0.1", "port": 3128, "auth": False}},
                 "episode_view": {"rc": 0, "opt_entries": "0", "evolver_dir_visible": "0", "procs_visible": "2"}}}


def _verdict(p, pl, spec):
    return pce.container_probe_problems(p, expected_top=ctl.expected_top(pl),
                                        expected_mounts=ctl.expected_mount_points(pl),
                                        rw_paths=[r["path"] for r in spec["rw"]], ro_rebind=spec["ro_rebind"],
                                        require_libreoffice="LibreOffice 25.2.3.2 520(Build:2)")


def _set(d, path, v):
    ks = path.split(".")
    for k in ks[:-1]:
        d = d[k]
    d[ks[-1]] = v(d[ks[-1]]) if callable(v) else v


MUTATIONS = [
    ("identity.uid", 0, "uid/euid"), ("identity.caps.CapBnd", "000001ffffffffff", "capabilities"),
    ("identity.no_new_privs", "0", "no_new_privs"), ("identity.cc_sha256", "0" * 64, "Claude Code sha"),
    ("identity.docker_on_path", "/usr/local/bin/docker", "docker is on PATH"),
    ("identity.opt_entries", ["claude-code", "evoskill"], "/opt holds"),
    ("identity.secret_env_names", ["OPENROUTER_API_KEY"], "secret-shaped"),
    ("identity.solver_stack", "3.1.5 2.2.0 1.26.4 3.13.9", "solver stack"),
    ("identity.libreoffice", "LibreOffice 24", "LibreOffice"), ("identity.soffice_convert_ok", False, "convert"),
    ("identity.users_files", 3, "/Users held"), ("identity.proxy_env", {}, "proxy env"),
    ("identity.groups", [0, 1000], "uid/euid"), ("priv.kcore", "READ 64 B", "kcore readable"),
    ("view.top", lambda t: t + ["workspace"], "view: / ="),
    ("view.mounts", lambda m: m + [{"point": f"/{MACHINE_TREE}/SpreadsheetBench-v2", "ro": True}], "mount table"),
    ("view.mounts", lambda m: [x if x["point"] != "/Users" else {**x, "ro": False} for x in m], "read-only flag"),
    ("view.canaries_visible_n", 1, "machine path(s) visible"), ("view.canaries_checked", 0, "machine path(s) visible"),
    ("view.host_pid_matches", [{"pid": 812, "comm": "python", "start": 5}], "machine processes visible"),
    ("view.procs", lambda p: p + [{"pid": 40, "comm": "bash", "start": 2, "uid": 0}], "process(es) visible"),
    ("view.ns.net", HNS["net"], "namespaces shared"), ("view.dev", ["null", "sda"], "/dev ="),
    ("view.home", ["evo", "runner"], "/home"),
    ("writes.rw./evo/out", "EACCES", "not exactly writable"), ("writes.ro./Users", "WRITABLE", "read-only dir"),
    ("priv.setuid0", "SUCCEEDED", "setuid0"), ("priv.mount_rc", 0, "mount_rc"),
    ("priv.ptrace_pid1", "SUCCEEDED", "ptrace"), ("priv.pid1_environ", "READ 64 B", "pid1_environ"),
    ("priv.userns_view_equal", False, "user namespace"),
    ("net.ifaces", ["lo", "eth0"], "interfaces"), ("net.routes", 1, "interfaces"),
    ("net.http", lambda h: [{**x, "reachable": True} if x["url"] == "https://openrouter.ai" else x for x in h],
     "not refused"),
    ("net.http", lambda h: [{**x, "reachable": False} if x["url"] in de.MUST_REACH else x for x in h],
     "not reachable through the proxy"),
    ("net.direct", lambda d: [{**d[0], "reached": True}] + d[1:], "proxy-less"),
    ("net.raw", lambda r: [{**x, "tcp_connected": True} if x["target"] == "169.254.169.254:80" else x for x in r],
     "reachable without the proxy"),
    ("net.raw", lambda r: [{**x, "answered": True} if x["target"] == "udp 1.1.1.1:53" else x for x in r],
     "reachable without the proxy"),
    ("net.raw", lambda r: [x for x in r if x["target"] != "udp 8.8.8.8:53"], "not probed"),
    ("net.abstract", [{"name": "x", "connected": True}], "abstract"),
    ("net.dns", {"example.com": ["93.184.216.34"]}, "DNS resolved"),
    ("host.proxy.listen_netns", HNS["net"], "proxy not listening"),
    ("host.proxy.allow", ["api.anthropic.com", "openrouter.ai"], "proxy not listening"),
    ("host.here_string_fd", "/tmp/sh-thd.1 (deleted)", "here-strings"),
    ("host.episode_view.opt_entries", "3", "episode view"),
]


def test_good_probe_passes_and_every_violation_is_named():
    spec = spec_for()
    pl, lim = plan_of(spec)
    assert _verdict(good_probe(pl, spec), pl, spec) == []
    for path, v, why in MUTATIONS:
        p = good_probe(pl, spec)
        _set(p, path, v)
        probs = _verdict(p, pl, spec)
        assert any(why in x for x in probs), (path, probs)


FB_TUNNELS = ["gre0", "gretap0", "erspan0", "ip6tnl0", "ip6gre0", "tunl0", "sit0", "ip_vti0", "ip6_vti0"]


def test_kernel_fallback_tunnel_devices_pass_only_without_routes_or_addresses():
    """Daytona machines whose kernel has the tunnel modules loaded give every new network namespace these nine
    fallback devices; they pass by name, and only so."""
    spec = spec_for()
    pl, _ = plan_of(spec)
    p = good_probe(pl, spec)
    p["net"]["ifaces"] = ["lo"] + FB_TUNNELS
    assert _verdict(p, pl, spec) == []
    for path, v in (("net.ifaces", ["lo", "eth0"] + FB_TUNNELS), ("net.routes", 1), ("net.inet6", ["lo", "sit0"]),
                    ("net.ifaces", FB_TUNNELS), ("net.ifaces", ["lo", "gre1"])):
        q = good_probe(pl, spec)
        q["net"]["ifaces"] = ["lo"] + FB_TUNNELS
        _set(q, path, v)
        assert any("interfaces" in x for x in _verdict(q, pl, spec)), (path, v)
    post = {"view": {"top": p["view"]["top"], "mounts": p["view"]["mounts"], "ifaces": ["lo"] + FB_TUNNELS}}
    assert pce.post_probe_problems(p, post) == []
    post["view"]["ifaces"] = ["lo"] + FB_TUNNELS + ["veth0"]
    assert any("interfaces" in x for x in pce.post_probe_problems(p, post))


def test_post_probe_must_match_pre():
    spec = spec_for()
    pl, _ = plan_of(spec)
    pre = good_probe(pl, spec)
    post = {"view": {"top": pre["view"]["top"], "mounts": pre["view"]["mounts"], "ifaces": ["lo"]}}
    assert pce.post_probe_problems(pre, post) == []
    post2 = json.loads(json.dumps(post))
    post2["view"]["mounts"].append({"point": "/mnt", "ro": False})
    assert pce.post_probe_problems(pre, post2)
    post3 = json.loads(json.dumps(post))
    post3["view"]["ifaces"] = ["lo", "veth0"]
    assert pce.post_probe_problems(pre, post3)


# ═════════════════════════════════════════════════════════════════════════ 6. host-side reads never follow links
def test_scan_and_tail_never_follow_symlinks(tmp_path):
    root = tmp_path / "c"
    (root / "rw/home/.claude").mkdir(parents=True)
    outside = tmp_path / "machine_secret.txt"
    outside.write_bytes(b"xx sk-ant-oat01-SECRETSECRET yy")
    (root / "rw/home/link").symlink_to(outside)
    (root / "rw/home/.claude/transcript.jsonl").write_bytes(b'{"a": "sk-ant-oat01-LEAK"}')
    os.mkfifo(root / "rw/home/fifo")
    hits, n = ctl.scan_tree(str(root), (ctl.TOKEN_PREFIX,))
    assert hits == [("rw/home/.claude/transcript.jsonl", 0)] and n == 1
    log = root / "evolver.log"
    log.symlink_to(outside)
    assert ctl._read_nofollow(str(log), 0, 100) == (0, b"")
    real = root / "real.log"
    real.write_bytes(b"0123456789")
    assert ctl._read_nofollow(str(real), 3, 4) == (10, b"3456")


def test_safe_host_path_refuses_planted_links(tmp_path, monkeypatch):
    monkeypatch.setattr(ctl, "BASE", str(tmp_path / "base"))
    spec = spec_for()
    pl, _ = plan_of(spec)
    home = Path(ctl.cdir(CID)) / "rw/home"
    (home / ".claude/projects").mkdir(parents=True)
    hp, why = ctl.safe_host_path(pl, "/home/evo/.claude/projects", CID)
    assert hp == str(home / ".claude/projects") and why == ""
    shutil.rmtree(home / ".claude")
    (home / ".claude").symlink_to(f"/{MACHINE_TREE}/SpreadsheetBench-v2")
    hp, why = ctl.safe_host_path(pl, "/home/evo/.claude/projects", CID)
    assert hp is None and "symlink" in why
    assert ctl.safe_host_path(pl, "/usr/bin", CID)[0] is None            # only the container's own dirs


def test_build_etc_copies_the_image_etc_minus_secrets(tmp_path):
    src, dst = tmp_path / "etc", tmp_path / "out"
    for rel, data in (("passwd", b"root:x:0:0::/root:/bin/bash\ndaytona:x:1000:1000::/home/daytona:/bin/sh\n"),
                      ("group", b"root:x:0:\ndaytona:x:1000:\n"), ("shadow", b"root:$6$h"),
                      ("ssh/ssh_host_ed25519_key", b"k"), ("ssl/private/x.key", b"k"), ("ssl/certs/ca.pem", b"CERT"),
                      ("hostname", b"box"), ("resolv.conf", b"nameserver 1.1.1.1\n"), ("profile.d/daytona.sh", b"x"),
                      ("fonts/fonts.conf", b"<fontconfig/>"), ("evoskill_libreoffice_version", b"LibreOffice 25")):
        _w(src / rel, data)
    (src / "alternatives").mkdir()
    (src / "alternatives/python").symlink_to("/usr/bin/python3.11")
    rec = ctl.build_etc(str(dst), str(src))
    for gone in ("shadow", "ssh", "ssl/private", "profile.d"):
        assert not (dst / gone).exists(), gone
    assert (dst / "ssl/certs/ca.pem").read_bytes() == b"CERT" and (dst / "fonts/fonts.conf").exists()
    assert os.readlink(dst / "alternatives/python") == "/usr/bin/python3.11"
    pw = (dst / "passwd").read_text()
    assert "evo:x:1000:1000::/home/evo:/bin/bash" in pw and "daytona" not in pw
    assert (dst / "resolv.conf").read_text() == ctl.RESOLV_CONF and (dst / "hostname").read_text() == "evolver\n"
    assert (dst / "claude-code").is_dir() and rec["files_copied"] >= 4
    _w(src / "environment.d/10-key.conf", b"OPENROUTER_API_KEY=sk-or-v1-abcdef")
    with pytest.raises(RuntimeError):
        ctl.build_etc(str(tmp_path / "out2"), str(src))


def test_probe_scripts_compile_under_the_machine_python():
    import ast
    for code in (ctl.PROBE_PY, ctl.POST_PROBE_PY, ctl.INIT):
        ast.parse(code, feature_version=(3, 9))
    ast.parse(Path(ctl.__file__).read_text(), feature_version=(3, 9))
    assert "kill -TERM" in ctl.STOP_SH and f"{ctl.VM_OUT}/rc" in ctl.STOP_SH


# ═════════════════════════════════════════════════════════════════════════ 7. the driver, end to end, offline
TOKEN = de.TOKEN_PREFIX + "FAKE" + "x" * 90


class FakeProc:
    def __init__(self, m):
        self.m = m

    def exec(self, cmd, timeout=None, env=None):
        parts = shlex.split(cmd)
        assert parts[:2] == ["bash", "-c"] and parts[-1] == "2>&1"
        inner = parts[2].rstrip("\n")
        code, out = self.m.handle(inner, env or {})
        return SimpleNamespace(exit_code=code, result=out)


class FakeFS:
    def __init__(self, m):
        self.m = m

    def upload_file(self, local, remote, timeout=None):
        self.m.files[remote] = Path(local).read_bytes()
        self.m.log.append(("upload", remote))

    def download_file(self, remote, local, timeout=None):
        Path(local).write_bytes(self.m.files[remote])
        self.m.log.append(("download", remote))


class FakeMachine:
    """The pool machine as the driver sees it through the SDK: every evoctl subcommand answered as the real one
    would on a contained machine; the evolver's session is played when the launch arrives."""

    def __init__(self, sid="aaaa1111bbbb", name=pce.POOL_PREFIX + "v1pl-m1-120000"):
        self.id, self.name = sid, name
        self.cpu, self.memory, self.disk = 8, 32, 40
        self.state = "SandboxState.STARTED"
        self.domain_allow_list = "openrouter.ai,api.anthropic.com"
        self.network_allow_list, self.network_block_all = None, False
        self.build_info = SimpleNamespace(snapshot_ref="daytona-leg:snap")
        self.files, self.cfs, self.log, self.execs = {}, {}, [], []
        self.process, self.fs = FakeProc(self), FakeFS(self)
        self.spec = self.pl = None
        self.deleted = False
        # the stateful session: `cfs` is kept between the driver's epochs (new Python objects, one
        # machine); `up` / `down` / `down --wipe` flip these, a test sets `session_live` to play a crashed driver's
        # still-running claude (epoch-begin then refuses until stop-session)
        self.up = self.wiped = self.session_live = False

    def refresh_data(self):
        pass

    def __getattr__(self, k):                       # readback's other getattr()s
        if k in ("sandbox_class", "target", "runner_id", "created_at", "auto_stop_interval", "auto_delete_interval",
                 "auto_destroy_at", "snapshot"):
            return None
        raise AttributeError(k)

    # the machine
    def handle(self, cmd: str, env: dict) -> tuple:
        self.execs.append((cmd, dict(env)))
        C = f"{ctl.PY} {ctl.CTL}"
        if cmd.startswith(f"mkdir -p {ctl.BIN}") or cmd.startswith(f"mkdir -p {os.path.dirname(ctl.CC_HOST)}"):
            return 0, ""
        if cmd.startswith(f"sha256sum {ctl.CTL}"):
            return 0, "".join(f"{hashlib.sha256(self.files[p]).hexdigest()}  {p}\n" for p in (ctl.CTL, ctl.PROXY_FILE))
        if cmd.startswith(f"sha256sum {ctl.CC_HOST}"):
            return 0, ""
        if cmd.startswith('test "$(sha256sum ') and ctl.CC_HOST + ".part" in cmd:
            assert ctl.CC_HOST + ".part" in self.files
            return 0, "CC_OK\n"
        if cmd == f"{C} list":
            have = [self.spec["cid"]] if self.spec and not self.wiped else []   # a kept container is listed
            return 0, json.dumps({"containers": have, "cgroups": have if self.up else []})
        if cmd.startswith(f"{C} prepare --spec-b64 "):
            self.spec = json.loads(base64.b64decode(cmd.split()[-1]))
            assert ctl.validate_spec(self.spec) == []
            lim = ctl.resolve_limits(FAKE_FACTS, self.spec.get("limits"))
            self.pl = ctl.plan(self.spec, FAKE_FACTS, lim)
            self.lim = lim
            self.cfs, self.up, self.wiped, self.session_live = {}, False, False, False   # a new container
            return 0, json.dumps({"ok": True, "cdir": ctl.cdir(self.spec["cid"]), "facts": FAKE_FACTS,
                                  "limits": lim, "etc": {"files_copied": 600}, "plan": self.pl,
                                  "expected_top": ctl.expected_top(self.pl), "cc_host_sha256": ctl.CC_SHA256})
        cid = self.spec["cid"] if self.spec else None
        if cmd.startswith(f"{C} up --cid {cid}"):
            self.up, self.down = True, False
            return 0, json.dumps({"ok": True, "state": {"init_pid": 4242, "ns": CNS, "machine_ns": HNS,
                                                        "proxy": self._proxy_ready()}})
        if cmd == f"{C} probe --cid {cid} --phase pre":
            return 0, "noise on stderr\n" + json.dumps({"ok": True, "probe": good_probe(self.pl, self.spec)})
        if cmd == f"{C} probe --cid {cid} --phase post":
            pre = good_probe(self.pl, self.spec)
            return 0, json.dumps({"ok": True, "probe": {"phase": "post", "view": {
                "top": pre["view"]["top"], "mounts": pre["view"]["mounts"], "ifaces": ["lo"]}}})
        if cmd == f"{C} extract --cid {cid}":
            with tarfile.open(fileobj=io.BytesIO(self.files.pop(f"{ctl.cdir(cid)}/state/upload.tgz")), mode="r:gz") as tf:
                for mm in tf.getmembers():
                    self.cfs["/" + mm.name] = tf.extractfile(mm).read()
            return 0, json.dumps({"ok": True, "files": len(self.cfs)})
        if cmd == f"{C} extract --cid {cid} --incremental":       # create-only into a kept container
            code, res = self._extract_into(f"{ctl.cdir(cid)}/state/upload.tgz", ctl.UPLOAD_TOPS, "extract --incremental")
            return code, json.dumps({**res, "incremental": True} if res.get("ok") else res)
        if cmd == f"{C} restore --cid {cid}":                     # before `up` only
            if self.up:
                return 2, json.dumps({"ok": False, "problems": ["container already up (state.json exists); restore "
                                                                "runs before up"]})
            code, res = self._extract_into(f"{ctl.cdir(cid)}/state/restore.tgz", ctl.RESTORE_PREFIXES, "restore")
            if not res.get("ok"):
                return code, json.dumps(res)
            return 0, json.dumps({"ok": True, "files": res["files"], "dirs": len(res["dirs_created"])})
        if cmd == f"{C} stop-session --cid {cid}":
            self.session_live = False
            self.cfs.setdefault(f"{ctl.VM_OUT}/rc", b"124\n")                  # STOP_SH: rc 124 unless claude wrote one
            return 0, json.dumps({"ok": True, "rc": 0})
        if cmd.startswith(f"{C} epoch-begin --cid {cid}"):
            args = shlex.split(cmd)
            clear = [x for i, x in enumerate(args) if args[i - 1] == "--clear"]
            if not self.up:
                return 2, json.dumps({"ok": False, "problems": ["container not up"]})
            if self.session_live:
                return 2, json.dumps({"ok": False, "problems": ["a session is still running in the container; "
                                                                "stop-session first"]})
            removed = [n for n in ctl.OUT_RESET_NAMES if self.cfs.pop(f"{ctl.VM_OUT}/{n}", None) is not None]
            evo_rw = [r["path"] for r in self.spec["rw"] if r["owner"] == "evo"]
            cleared = {}
            for p in clear:
                if p not in evo_rw:
                    return 2, json.dumps({"ok": False, "problems": [f"--clear {p}: not an evo-owned rw path of the "
                                                                    f"spec ({evo_rw})"]})
                gone = [k for k in self.cfs if ctl._under(k, p) and k != p]
                for k in gone:
                    del self.cfs[k]
                cleared[p] = len(gone)
            return 0, json.dumps({"ok": True, "init_alive": True, "thawed": False, "proxy_restarted": False,
                                  "proxy": self._proxy_ready(), "out_reset": removed, "cleared": cleared})
        if cmd.startswith(f"{C} epoch-end --cid {cid} "):          # pack while frozen, no down
            args = shlex.split(cmd)
            parts = [x for i, x in enumerate(args) if args[i - 1] == "--part"]
            if not self.up:
                return 2, json.dumps({"ok": False, "cid": cid, "kept": True, "problems": ["container not up"]})
            packed, refused, nbytes = self._pack_parts(cid, parts, dir_entries=True)
            return 0, json.dumps({"ok": True, "cid": cid, "kept": True, "frozen": True, "thawed": True,
                                  "init_alive": True, "tmpfs_scan": {"files": 4, "hits": []},
                                  "scan": {k: {"files": 9, "hits": []} for k in ("fs", "rw", "state")},
                                  "egress_log": {"requests": 5, "by_host": {"api.anthropic.com:allowed": 4,
                                                                            "openrouter.ai:refused": 1}},
                                  "tgz": f"{ctl.cdir(cid)}/state/out.tgz", "bytes": nbytes, "packed": packed,
                                  "refused": refused})
        if cmd.startswith(f"{C} enter --cid {cid} --cwd /tmp -- {ctl.PY} -c "):
            roots = shlex.split(cmd)[len(shlex.split(f"{C} enter --cid {cid} --cwd /tmp --")) + 3:]
            rows = [[p, "F", hashlib.sha256(b).hexdigest(), len(b)] for p, b in sorted(self.cfs.items())
                    if any(ctl._under(p, r) for r in roots)]
            rows += [[ctl.VM_OUT, "D", "", 0]] if any(ctl._under(ctl.VM_OUT, r) for r in roots) else []
            return 0, json.dumps(rows)
        if "Reply with exactly: OK" in cmd and "enter --cid" in cmd:
            self.check_token_cmd(cmd, env)
            return 0, json.dumps({"type": "rate_limit_event", "rate_limit_info": {"status": "allowed"}}) + "\nrc=0\n"
        if "setsid nohup" in cmd:
            self.check_token_cmd(cmd, env)
            self.play_session()
            return 0, self.tail(0, 1500)
        if cmd.startswith(f"{C} tail --cid {cid} "):
            a = shlex.split(cmd)
            return 0, self.tail(int(a[a.index("--offset") + 1]), int(a[a.index("--max") + 1]))
        if cmd == f"{C} status --cid {cid}":
            return 0, json.dumps({"cid": cid, "exists": not self.wiped, "init_alive": self.up, "proxy_alive": self.up,
                                  "state": ({"init_pid": 4242, "ns": CNS, "machine_ns": HNS} if self.up else {})})
        if cmd == f"{C} down --cid {cid}":
            self.down, self.up = True, False
            return 0, json.dumps({"cid": cid, "verified": True, "wiped": False,
                                  "tmpfs_scan": {"files": 4, "hits": []},
                                  "scan": {k: {"files": 9, "hits": []} for k in ("fs", "rw", "state")},
                                  "egress_log": {"requests": 5, "by_host": {"api.anthropic.com:allowed": 4,
                                                                            "openrouter.ai:refused": 1}}})
        if cmd.startswith(f"{C} pack --cid {cid} "):
            assert getattr(self, "down", False), "pack before down"
            parts = [a for i, a in enumerate(shlex.split(cmd)) if shlex.split(cmd)[i - 1] == "--part"]
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tf:
                for p, b in sorted(self.cfs.items()):
                    if any(ctl._under(p, part) for part in parts):
                        ti = tarfile.TarInfo(p.lstrip("/"))
                        ti.size = len(b)
                        tf.addfile(ti, io.BytesIO(b))
            self.files[f"{ctl.cdir(cid)}/state/out.tgz"] = buf.getvalue()
            return 0, json.dumps({"ok": True, "packed": parts, "refused": []})
        if cmd == f"{C} down --cid {cid} --wipe":
            self.up, self.wiped, self.session_live = False, True, False        # c/<cid> gone: `list` is empty again
            return 0, json.dumps({"cid": cid, "verified": True, "wiped": True, "procs_in_container_ns": [],
                                  "cgroup_removed": True, "proxy_alive": False,
                                  "scan": {k: {"files": 9, "hits": []} for k in ("fs", "rw", "state")},
                                  "egress_log": {"requests": 5, "by_host": {"api.anthropic.com:allowed": 4,
                                                                            "openrouter.ai:refused": 1}}})
        raise AssertionError(f"unexpected command on the machine: {cmd[:200]}")

    # ── the stateful session's helpers (the ctl's contracts; the real ctl must match)
    @staticmethod
    def _proxy_ready() -> dict:
        return {"pid": 4300, "listen": f"{ctl.PROXY_HOST}:{ctl.PROXY_PORT}", "listen_netns": CNS["net"],
                "machine_netns": HNS["net"], "allow": [ctl.ALLOW], "upstream": {"kind": "direct"}}

    @staticmethod
    def _ancestors(p: str) -> set:
        out, d = set(), os.path.dirname(p)
        while d and d != "/":
            out.add(d)
            d = os.path.dirname(d)
        return out

    def _present_dirs(self) -> set:
        """Directories the container already has: the plan's fixed ones, the spec's rw dirs, every ancestor of a file."""
        fixed = {"/Users", ctl.VM_ROOT, ctl.VM_WORK, ctl.VM_OUT, ctl.POLICY_DIR, ctl.EVO_HOME}
        fixed |= {r["path"] for r in (self.spec or {}).get("rw") or []}
        have = set(fixed)
        for p in list(fixed) + list(self.cfs):
            have |= self._ancestors(p)
        return have

    def _extract_into(self, remote_tgz: str, tops: tuple, what: str) -> tuple:
        """Create-only extraction of a machine-side tgz into cfs (the incremental extract and restore share it):
        plain members below `tops` only, nothing already present, the dirs it creates reported."""
        with tarfile.open(fileobj=io.BytesIO(self.files.pop(remote_tgz)), mode="r:gz") as tf:
            mem = tf.getmembers()
            bad = [m.name for m in mem if m.name.startswith("/") or ".." in m.name.split("/")
                   or not m.name.startswith(tuple(tops)) or not m.isfile()]
            if bad:
                return 2, {"ok": False, "problems": [f"refused tar members {bad[:5]}"]}
            present = [m.name for m in mem if "/" + m.name in self.cfs]
            if present:
                return 2, {"ok": False, "problems": [f"{what}: members already present {present[:5]}"]}
            have, new_dirs = self._present_dirs(), set()
            for m in mem:
                self.cfs["/" + m.name] = tf.extractfile(m).read()
                new_dirs |= self._ancestors("/" + m.name) - have
                have |= new_dirs
        return 0, {"ok": True, "files": len(mem), "dirs_created": sorted(new_dirs)}

    def _pack_parts(self, cid: str, parts: list, *, dir_entries: bool) -> tuple:
        """state/out.tgz of the container paths in `parts` (files under each; with dir_entries the part's own
        directory entry too, as tar's recursive add stores it); a part the container does not have is refused."""
        have, packed, refused = self._present_dirs(), [], []
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for part in parts:
                if part not in have and not any(ctl._under(p, part) for p in self.cfs):
                    refused.append({"part": part, "why": f"{part}: absent"})
                    continue
                if dir_entries:
                    ti = tarfile.TarInfo(part.lstrip("/"))
                    ti.type = tarfile.DIRTYPE
                    ti.mode = 0o755
                    tf.addfile(ti)
                for p, b in sorted(self.cfs.items()):
                    if ctl._under(p, part):
                        ti = tarfile.TarInfo(p.lstrip("/"))
                        ti.size = len(b)
                        tf.addfile(ti, io.BytesIO(b))
                packed.append(part)
        self.files[f"{ctl.cdir(cid)}/state/out.tgz"] = buf.getvalue()
        return packed, refused, len(buf.getvalue())

    def check_token_cmd(self, cmd, env):
        import tau3_evolver_daytona as T
        assert env == {T.SECRET_ENV: TOKEN}
        assert TOKEN not in cmd and de.TOKEN_PREFIX not in cmd

    def play_session(self):
        sid = str(uuid.uuid4())
        lines = [{"type": "system", "subtype": "init", "apiKeySource": "none", "mcp_servers": [],
                  "tools": ["Bash", "Read", "Write"], "claude_code_version": "2.1.281", "model": "claude-opus-5-5",
                  "session_id": sid},
                 {"type": "assistant", "message": {"model": "claude-opus-5-5", "content": [
                     {"type": "tool_use", "name": "Bash", "input": {"command": "ls /Users"}}]}},
                 {"type": "result", "subtype": "success", "is_error": False}]
        body = "".join(json.dumps(x) + "\n" for x in lines).encode()
        self.cfs[f"{ctl.VM_OUT}/evolver.log"] = body
        self.cfs[f"{ctl.VM_OUT}/evolver.stderr"] = b""
        self.cfs[f"{ctl.VM_OUT}/rc"] = b"0\n"
        self.cfs[f"{de.VM_SESSION_DIR}/{sid}.jsonl"] = body
        st = self.spec["rw"][0]["path"]
        self.cfs[f"{st}/table-audit/SKILL.md"] = b"---\nname: table-audit\ndescription: d\n---\nbody\n"

    def tail(self, off, mx):
        log = self.cfs.get(f"{ctl.VM_OUT}/evolver.log", b"")
        rc = self.cfs.get(f"{ctl.VM_OUT}/rc", b"").decode().strip() or "RUNNING"
        return f"{len(log)} {rc} 4096 1\n@@B64@@{base64.b64encode(log[off:off + mx]).decode()}\n"


@pytest.fixture
def repo_run(monkeypatch):
    """A fake run under the repository (a per-epoch container mirrors host paths below /Users, where the repository
    lives on a macOS host; pytest's tmp_path is not there)."""
    base = ROOT / ".cache" / "test_evolver_container" / uuid.uuid4().hex[:10]
    base.mkdir(parents=True)
    yield base
    shutil.rmtree(base, ignore_errors=True)


def test_driver_end_to_end_on_a_fake_pool_machine(repo_run, monkeypatch):
    import daytona_lib_evo as dle
    import evolver_container as EC
    import tau3_evolver_daytona as T
    from tests.test_template_oracle_and_grader_wording import _grader, _runtime
    from tests.test_train_references import _fake
    f = _fake(repo_run, monkeypatch)
    _grader(f.rd, f.split)
    de.build_references_dir(f.rd, f.split)
    _runtime(f, template=True)
    rt = json.loads((f.rd / "RUNTIME.json").read_text())
    rt["daytona_leg"] = {"store_root": str(repo_run / "store")}          # the pool's registry lives here
    (f.rd / "RUNTIME.json").write_text(json.dumps(rt))
    monkeypatch.setenv("EVOSKILL_TAU3_EVOLVER_SPLIT", str(f.split))
    tok = repo_run / "fake_token"
    tok.write_text(TOKEN)
    os.chmod(tok, 0o600)
    ledger = repo_run / "ledger.jsonl"
    monkeypatch.setattr(dle, "LEDGER", str(ledger))
    fm = FakeMachine()
    client = SimpleNamespace(get=lambda sid: fm if sid == fm.id else (_ for _ in ()).throw(KeyError(sid)),
                             delete=lambda *a, **k: (_ for _ in ()).throw(AssertionError("machine deleted")))
    monkeypatch.setattr(dle, "client", lambda: client)
    monkeypatch.setattr(dle, "delete_by_id", lambda *a, **k: (_ for _ in ()).throw(AssertionError("delete_by_id")))
    cc = repo_run / "cc_fake"
    cc.write_bytes(b"ELF fake claude")
    monkeypatch.setattr(EC, "ensure_cc_binary", lambda cfg, run_dir=None: cc)
    monkeypatch.setattr(T, "POLL_S", 0)
    # THE RUN POOL (long-lived): its host publishes machine.json and reads leases/ with the pool's OWN
    # Registry code; the driver must be listed in leased_by before anything runs on the machine
    regdir = pce.pool_run_dir(f.rd)
    host = FakePoolHost(regdir, {"m01": pool_row(fm.id, fm.name)}).start()
    seen = {}
    _play = fm.play_session

    def _play_and_look():                         # while the session runs, the pool honours the lease
        seen["leases"] = P.Registry(regdir).leases(time.time())
        seen["leased_by"] = host.last["machines"]["m01"]["leased_by"]
        _play()
    fm.play_session = _play_and_look
    monkeypatch.setattr(pce, "LEASE_ACK_S", 5.0)
    q = repo_run / "q"
    staging = f.rd / "skills_v1_staging"
    run = EC.ContainerEvolverRun(
        epoch=1, prompt=f.rd / "ITER1_PROMPT.md", digests=f.ev, staging=staging, run_dir=f.rd, prior=[],
        model="claude-opus-5-5", oracle=f.rd / "oracle",
        cfg={"oauth_token_file": str(tok), "placement": "pool_container", "train_references": True,
             "grader_clean": True, "require_libreoffice": "LibreOffice 25.2.3.2 520(Build:2)"},
        timeout_s=900, receipt_path=f.rd / "e1_EVOLVER_CONTAINMENT.json", trace_path=f.rd / "tau3_evolver_e1.trace.json",
        listing_path=f.rd / "e1_EVOLVER_VM_LISTING.json", log_path=f.rd / "tau3_evolver_e1.log", quarantine=q, mode="all_epochs")
    run.cinit()
    try:
        rc = run.run(gate_argv=None)
    finally:
        host.stop()
    assert rc == 0
    rec = json.loads(run.receipt_path.read_text())
    # (0) THE POOL'S MACHINE AND LEASE: picked from machine.json, leased in the pool's format, honoured by the
    # pool's own Registry while the session ran, listed in leased_by, and gone at the end
    holder = pce.lease_holder(run.cid)
    assert seen["leases"] == {fm.id: [holder]} and seen["leased_by"] == [holder]
    pm, le = rec["daytona"]["pool_machine"], rec["daytona"]["lease"]
    assert pm["format"] == "pool" and pm["name"] == "m01" and pm["row"]["sandbox_id"] == fm.id
    assert le["acknowledged"] is True and le["released"] is True and le["holder"] == holder
    assert not (regdir / pce.POOL_LEASES_DIR / f"{holder}.json").exists()
    assert P.Registry(regdir).leases(time.time()) == {}
    assert not (pce.registry_dir(f.rd) / pce.REGISTRY_NAME).exists() and not (regdir / pce.LEASE_NAME).exists()
    # (a) SCHEMA: every key today's own-sandbox receipt carries (the loop adds evidence_sha256_after itself)
    assert set(OWN_SANDBOX_KEYS["receipt"]) - {"evidence_sha256_after"} <= set(rec)
    assert set(OWN_SANDBOX_KEYS["daytona"]) - RETIRED_DAYTONA_KEYS <= set(rec["daytona"])
    assert not RETIRED_DAYTONA_KEYS & set(rec["daytona"])
    assert set(OWN_SANDBOX_KEYS["identity"]) <= set(rec["daytona"]["identity"])
    assert set(OWN_SANDBOX_KEYS["listing"]) <= set(json.loads(run.listing_path.read_text()))
    assert set(OWN_SANDBOX_KEYS["trace"]) - {"evidence_sha256_after"} <= set(json.loads(run.trace_path.read_text()))
    assert rec["containment"] == "daytona" and rec["daytona"]["placement"] == "pool_container"
    # (b) the loop's checks: the manifest check UNCHANGED, the container check instead of check_daytona_receipt
    m = de.check_upload_manifest(run_dir=f.rd, epoch=1, split=f.split, evidence_dirs=run.evidence_dirs,
                                 prev_library=f.rd / "skills_v0", prompt=f.rd / "ITER1_PROMPT.md", oracle=f.rd / "oracle",
                                 staging=staging)
    assert m.status == "PASS", m.detail
    ok, det = pce.check_pool_container_receipt(rec, log=run.log_path, ledger_path=ledger, staging=staging,
                                               require_libreoffice="LibreOffice 25.2.3.2 520(Build:2)")
    assert ok, det
    assert de.check_daytona_receipt(rec, log=run.log_path, ledger_path=ledger)[0] is False   # the other world
    # (c) the machine: never deleted; down before pack, pack before download, the wipe last
    assert not fm.deleted and (staging / "table-audit/SKILL.md").is_file()
    order = [c for c, _e in fm.execs]
    i_down = order.index(f"{ctl.PY} {ctl.CTL} down --cid {run.cid}")
    i_pack = next(i for i, c in enumerate(order) if c.startswith(f"{ctl.PY} {ctl.CTL} pack"))
    assert i_down < i_pack and order[-1] == f"{ctl.PY} {ctl.CTL} down --cid {run.cid} --wipe"
    assert ("download", f"{ctl.cdir(run.cid)}/state/out.tgz") in fm.log
    rows = [json.loads(ln) for ln in ledger.read_text().splitlines()]
    assert [r["event"] for r in rows] == ["container_attach", "container_up", "container_down"]
    assert rows[-1]["verified"] is True
    # (d) THE TOKEN: in the env of exactly the launch exec, never in a command, a file or an upload
    with_tok = [c for c, e in fm.execs if e]
    assert len(with_tok) == 1 and "setsid nohup" in with_tok[0]
    assert not [c for c, _e in fm.execs if TOKEN in c or de.TOKEN_PREFIX in c]
    assert not [p for p, b in fm.files.items() if TOKEN.encode() in b or de.TOKEN_PREFIX.encode() in b]
    leaked = []
    for p in list(repo_run.rglob("*")):
        if p.is_file() and not p.is_symlink() and p != tok and TOKEN.encode() in p.read_bytes():
            leaked.append(str(p))
    assert leaked == []


def test_a_turn_stopped_at_its_deadline_installs_the_staging_as_it_stands(repo_run, monkeypatch):
    """The epoch message says /evo/staging/ is taken as it stands at the deadline. A claude still running at the
    deadline is stopped (rc 124, no `result` event), its staging -- a half-written set included -- is installed,
    the receipt records the deadline stop, and the driver exits 0 (not the timeout exit 28)."""
    import daytona_lib_evo as dle
    import evolver_container as EC
    import tau3_evolver_daytona as T
    from tests.test_template_oracle_and_grader_wording import _grader, _runtime
    from tests.test_train_references import _fake
    f = _fake(repo_run, monkeypatch)
    _grader(f.rd, f.split)
    de.build_references_dir(f.rd, f.split)
    _runtime(f, template=True)
    rt = json.loads((f.rd / "RUNTIME.json").read_text())
    rt["daytona_leg"] = {"store_root": str(repo_run / "store")}
    (f.rd / "RUNTIME.json").write_text(json.dumps(rt))
    monkeypatch.setenv("EVOSKILL_TAU3_EVOLVER_SPLIT", str(f.split))
    tok = repo_run / "fake_token"
    tok.write_text(TOKEN)
    os.chmod(tok, 0o600)
    monkeypatch.setattr(dle, "LEDGER", str(repo_run / "ledger.jsonl"))
    fm = FakeMachine()
    client = SimpleNamespace(get=lambda sid: fm if sid == fm.id else (_ for _ in ()).throw(KeyError(sid)),
                             delete=lambda *a, **k: (_ for _ in ()).throw(AssertionError("machine deleted")))
    monkeypatch.setattr(dle, "client", lambda: client)
    monkeypatch.setattr(dle, "delete_by_id", lambda *a, **k: (_ for _ in ()).throw(AssertionError("delete_by_id")))
    cc = repo_run / "cc_fake"
    cc.write_bytes(b"ELF fake claude")
    monkeypatch.setattr(EC, "ensure_cc_binary", lambda cfg, run_dir=None: cc)
    monkeypatch.setattr(T, "POLL_S", 0)
    monkeypatch.setattr(EC, "POLL_S", 0, raising=False)

    def still_running():                       # the turn has not ended: no `result` event, no rc file
        sid = str(uuid.uuid4())
        lines = [{"type": "system", "subtype": "init", "apiKeySource": "none", "mcp_servers": [],
                  "tools": ["Bash", "Read", "Write"], "claude_code_version": "2.1.281", "model": "claude-opus-5-5",
                  "session_id": sid},
                 {"type": "assistant", "message": {"model": "claude-opus-5-5", "content": [
                     {"type": "tool_use", "name": "Write", "input": {"file_path": "/evo/staging/half/SKILL.md"}}]}}]
        body = "".join(json.dumps(x) + "\n" for x in lines).encode()
        fm.cfs[f"{ctl.VM_OUT}/evolver.log"] = body
        fm.cfs[f"{ctl.VM_OUT}/evolver.stderr"] = b""
        fm.cfs[f"{de.VM_SESSION_DIR}/{sid}.jsonl"] = body
        st = fm.spec["rw"][0]["path"]
        fm.cfs[f"{st}/half/SKILL.md"] = b"---\nname: half\ndescription: d\n---\nhalf-writ"
    fm.play_session = still_running
    regdir = pce.pool_run_dir(f.rd)
    host = FakePoolHost(regdir, {"m01": pool_row(fm.id, fm.name)}).start()
    monkeypatch.setattr(pce, "LEASE_ACK_S", 5.0)
    staging = f.rd / "skills_v1_staging"
    run = EC.ContainerEvolverRun(
        epoch=1, prompt=f.rd / "ITER1_PROMPT.md", digests=f.ev, staging=staging, run_dir=f.rd, prior=[],
        model="claude-opus-5-5", oracle=f.rd / "oracle",
        cfg={"oauth_token_file": str(tok), "placement": "pool_container", "train_references": True,
             "grader_clean": True, "require_libreoffice": "LibreOffice 25.2.3.2 520(Build:2)"},
        timeout_s=0, receipt_path=f.rd / "e1_EVOLVER_CONTAINMENT.json", trace_path=f.rd / "tau3_evolver_e1.trace.json",
        listing_path=f.rd / "e1_EVOLVER_VM_LISTING.json", log_path=f.rd / "tau3_evolver_e1.log", quarantine=repo_run / "q", mode="all_epochs")
    run.cinit()
    try:
        rc = run.run(gate_argv=None)
    finally:
        host.stop()
    assert rc == 0 and run.deadline_stop is True and run.claude_rc == 124
    rec = json.loads((f.rd / "e1_EVOLVER_CONTAINMENT.json").read_text())
    assert rec["daytona"]["deadline_stop"]["after_s"] == 0 and rec["daytona"]["claude_rc"] == 124
    assert rec["daytona"]["staging_installed"] is True
    assert (staging / "half/SKILL.md").read_bytes().endswith(b"half-writ")          # taken as it stands
    assert any(c == f"{ctl.PY} {ctl.CTL} stop-session --cid {run.cid}" for c, _e in fm.execs)


def test_extract_writes_only_plain_files_through_plain_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(ctl, "BASE", str(tmp_path / "base"))
    monkeypatch.setattr(ctl.os, "chown", lambda *a, **k: None)
    monkeypatch.setattr(ctl.os, "lchown", lambda *a, **k: None)
    spec = spec_for()
    P = ctl.paths(CID)
    for d in (P["state"], P["fs"] + "/Users", P["fs"] + ctl.VM_WORK, P["fs"] + ctl.VM_OUT, P["fs"] + STAGING):
        os.makedirs(d, exist_ok=True)
    pl, lim = plan_of(spec)
    for n, obj in (("spec", spec), ("facts", FAKE_FACTS), ("limits", lim)):
        Path(P["state"], f"{n}.json").write_text(json.dumps(obj))

    def tgz(members):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tf:
            for name, data, kind in members:
                ti = tarfile.TarInfo(name)
                if kind == "sym":
                    ti.type, ti.linkname = tarfile.SYMTYPE, "/etc/passwd"
                    tf.addfile(ti)
                else:
                    ti.size = len(data)
                    tf.addfile(ti, io.BytesIO(data))
        Path(P["state"], "upload.tgz").write_bytes(buf.getvalue())
    run = lambda: ctl.cmd_extract(SimpleNamespace(cid=CID))
    tgz([("Users/x/../../etc/passwd", b"x", "file")])
    assert run() == 2
    tgz([("opt/leg/leg.env", b"x", "file")])
    assert run() == 2
    tgz([("Users/a/link", b"", "sym")])
    assert run() == 2
    # a symlink planted in a writable dir: never followed by the root-owned extraction
    os.symlink(str(tmp_path / "outside"), P["fs"] + STAGING + "/seed")
    tgz([(STAGING.lstrip("/") + "/seed/SKILL.md", b"x", "file")])
    assert run() == 2 and not (tmp_path / "outside").exists()
    os.unlink(P["fs"] + STAGING + "/seed")
    tgz([(STAGING.lstrip("/") + "/seed/SKILL.md", b"body", "file"), ("evo/work/PROMPT.md", b"p", "file"),
         ("etc/claude-code/managed-settings.json", b"{}", "file")])
    assert run() == 0
    assert Path(P["fs"] + STAGING + "/seed/SKILL.md").read_bytes() == b"body"
    assert not Path(P["state"], "upload.tgz").exists()


def test_attach_refuses_at_once_when_the_pool_host_is_closed_or_dead(tmp_path, monkeypatch):
    """A closed / closing / dead pool host is never re-opened while the loop waits on the
    evolver, so the driver refuses on the first read instead of waiting machine_wait_s (900 s) and names the remedy."""
    import evolver_container as EC
    monkeypatch.setattr(pce, "MACHINE_POLL_S", 0.02)
    for i, (kw, row) in enumerate(((dict(host_state="closed"), pool_row()),
                                   (dict(host_state="closing"), pool_row()),
                                   (dict(), pool_row()))):
        r = _attach_run(tmp_path / str(i), cfg={"machine_wait_s": 30})
        root = pce.pool_run_dir(r.run_dir)
        root.mkdir(parents=True)
        mj = machines_json({"m01": row}, **kw)
        if i == 2:
            mj["host_pid"] = 2 ** 22 + 12345          # no such process: a dead host
        pce.write_json_atomic(root / pce.POOL_MACHINES_FILE, mj)
        t0 = time.time()
        with pytest.raises(EC.Refuse) as ei:
            r.attach_machine(pce.pool_machines_file(r.run_dir))
        assert time.time() - t0 < 5, "no machine_wait_s wait for a closed host"
        assert ei.value.code == EC.EXIT_IDENTITY and "any leg join" in str(ei.value)
        assert r.receipt["daytona"]["pool_machine"]["host_closed"] is True
        assert not list((root / pce.POOL_LEASES_DIR).glob("*.json"))


def test_solver_containers_on_the_shared_machine_are_not_an_evolver_conflict():
    """The Claude Code solver legs run `cc-*` containers on the evolver's machine
    (a val leg's 30 may be live when the epoch's evolver starts); create() must refuse only ANOTHER evolver container."""
    import scripts.daytona.evolver_container as EC
    f = EC.evolver_container_conflicts
    assert f(None) == [] and f([]) == []
    assert f(["cc-Template_01_02__s0-0927t1-ab12", "cc-other-0927t185950-4601"]) == []
    assert f([{"cid": "cc-x-1"}, {"cid": "e1-0927t1-deadbeef"}]) == [{"cid": "e1-0927t1-deadbeef"}]
    assert f(["run-20260927t193158-bf8851"]) == ["run-20260927t193158-bf8851"]
    assert f(["ao-0927-1", "cc-y"]) == ["ao-0927-1"]
