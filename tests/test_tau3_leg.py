"""scripts/tau3_cc/tau3_leg.py and the tau3 branch of scripts/daytona/cc_leg_driver.py, plus the pool-wide rollout
cap. NO DAYTONA, NO CLAUDE, NO KEY, NO NETWORK.

The pool machine is Tau3Machine: tests/test_cc_leg_driver.py's FakeHandle that also answers, by the `what` of each
exec, every command a REAL Tau3Box issues (the unit dir, job.json and its sha, the bridge launch, the batched poll,
the SIGTERM, down / ship / pack / the unit tar / rm / wipe) against an in-memory file map, and logs them in order.
Only the container's own setup (prepare / up / PRE probe) is faked (FakeTau3Box), as test_cc_leg_driver's FakeBox
fakes it; upload / toolchain / launch / stop / collect / wipe / result, poll_batch, judge and init_problems are real.
The customer's key is injected (key_fn) and openrouter_key.openrouter_key raises if anything calls it, so no test
reads a .env file; the token and the key are fake strings.
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import threading
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts" / "daytona", ROOT / "scripts" / "cc_solver", ROOT / "scripts" / "tau3_cc",
           ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import test_cc_leg_driver as TD  # noqa: E402  (the driver tests' fakes: FakeHandle, FakeGrader, World)
import tau3_leg as T  # noqa: E402

D, G, H, C, L = TD.D, TD.G, TD.H, TD.C, TD.L
S, HB = T.S, T.HB
ctl = L.ctl

RUN3 = "tau3_drv_fx_1001"
MODEL3 = "claude-sonnet-5-5"
CUSTOMER = "openrouter/openai/gpt-5-nano"
TOKEN = TD.TOKEN
KEY = "sk-or-v1-" + "f" * 64                     # a fake org key with the OpenRouter prefix
TRAIN3 = ["banking_knowledge/task_004", "banking_knowledge/task_005"]
VAL3 = ["banking_knowledge/task_006"]
N3 = 2
EVENTS = TD.EVENTS
FIRST = HB.first_message("POLICY TEXT", "Hi, I need to change my email.")


def no_key_file_read():
    raise AssertionError("no test reads the OpenRouter key file")


def clean_env(monkeypatch):
    """What a leg reads from the process environment, unset (an earlier test module may leave the SSB data overlay's
    variable in os.environ; INFRA here declares none, which the driver would refuse)."""
    from src.orchestrator import data_overlay
    for k in ("OPENROUTER_API_KEY", "EVOSKILL_SOLVER_MODEL", data_overlay.ENV) + D.OPENROUTER_PINS + \
            (D.SUBSET_ENV,):
        monkeypatch.delenv(k, raising=False)


# ══ the world: a tau3 run dir (SPEC benchmark tau3), its data root, frozen plug-in rows ══════════════════════════
class Tau3World:
    def __init__(self, tmp: Path, monkeypatch):
        self.tmp = tmp
        self.root = tmp / "tau3_data_fx"
        tdir = self.root / "tau2" / "domains" / T.DOMAIN / "tasks"
        tdir.mkdir(parents=True)
        for t in TRAIN3 + VAL3 + ["banking_knowledge/task_007"]:
            (tdir / f"{T.short_id(t)}.json").write_text(json.dumps({"id": T.short_id(t)}))
        self.rd = tmp / "outputs" / RUN3
        setup = self.rd / "setup"
        setup.mkdir(parents=True)
        # the freeze's split (plug-in ids '<domain>/task_NNN'): what SPEC split_path names (run setup, not code)
        self.split = setup / "split.json"
        self.split.write_text(json.dumps({"train": TRAIN3, "val": VAL3, "test": ["banking_knowledge/task_007"],
                                          "unused": []}))
        self.split_sha = TD.sha(self.split.read_bytes())
        (self.rd / "SPEC.json").write_text(json.dumps({"spec": {
            "name": RUN3, "solver_model": MODEL3, "solver_harness": "claude_code", "n_samples": N3, "agent_turns": 45,
            "split_path": str(self.split), "split_sha256": self.split_sha, "benchmark": "tau3",
            "user_sim_model": CUSTOMER}}))
        self.token = tmp / "cc_oauth_token"
        self.token.write_text(TOKEN + "\n")
        os.chmod(self.token, 0o600)
        inf = json.loads(TD.INFRA_SRC.read_text())
        inf["solver"]["key_file"] = inf["claude_code"]["token_file"] = str(self.token)
        inf["data"]["overlay_patches_json"] = inf["data"]["overlay_patches_sha256"] = None
        inf["data"]["root"] = str(self.root)
        self.inf = inf
        (self.rd / "INFRA.json").write_text(json.dumps(inf))
        self.store_root, self.tasks_root = tmp / "store", tmp / "tasks_root"
        self.rt = {"prompt_mode": "template", "leg_timeout_s": 43200, "evolver_evidence_layout": "evo",
                   "shared_dir": str(self.rd),
                   "daytona_leg": {"store_root": str(self.store_root), "tasks_root": str(self.tasks_root), "pool": True,
                                   "pool_keys": ["train", "val"], "pool_sandboxes": 1, "pool_ttl_minutes": 1440,
                                   "pool_ttl_margin_min": 255},
                   "evolver_daytona": {"placement": "pool_container", "oauth_token_file": str(self.token),
                                       "machine_wait_s": 900},
                   "cc_leg": {"concurrency": 30, "wall_min": 20,
                              "machine_wait_s": 1800, "grade_max_procs": 4, "grade_max_heavy": 1}}
        self.write_rt()
        self.tasks = [{"id": t, "category": T.DOMAIN, "instruction": "",
                       "inputs": [], "answer_files": [str(S.task_file(T.short_id(t), self.root))],
                       "grading": {"domain": T.DOMAIN, "retrieval_variant": T.VARIANT},
                       "split": "train" if t in TRAIN3 else "validation"} for t in TRAIN3 + VAL3]
        files = {}
        for n in ("benchmark_plugin.py", "benchmark_grader.py", "plugin_api.py"):
            (setup / n).write_text(f"# {n}\n")
            files[n] = {"path": str(setup / n), "sha256": TD.sha((setup / n).read_bytes())}
        (setup / "tasks.json").write_text(json.dumps(self.tasks))
        self.plugin = {"files": files, "tasks": self.tasks, "tasks_json_sha256": TD.sha((setup / "tasks.json").read_bytes())}
        cc_local = tmp / "claude-fake"
        cc_local.write_bytes(b"#!fake claude")
        monkeypatch.setattr(H, "CC_LOCAL", cc_local)
        monkeypatch.setattr(G, "frozen_plugin", lambda rd: json.loads(json.dumps(self.plugin)))
        monkeypatch.setattr(S.ORK, "openrouter_key", no_key_file_read)
        clean_env(monkeypatch)
        EVENTS.clear()

    def write_rt(self):
        (self.rd / "RUNTIME.json").write_text(json.dumps(self.rt))

    def write_spec(self, **over):
        sp = json.loads((self.rd / "SPEC.json").read_text())
        sp["spec"].update(over)
        (self.rd / "SPEC.json").write_text(json.dumps(sp))

    def args(self, key="train", *, epoch=0, **over):
        a = dict(tag=G.leg_tag(RUN3, epoch, key), key=key, run_dir=str(self.rd), split=str(self.split),
                 cache_dir=str(self.tmp / "cache" / f"{RUN3}_e{epoch}_cache"), skills_dir="", n_samples=N3,
                 concurrency=30, check_only=False, dry_run=False)
        a.update(over)
        return SimpleNamespace(**a)


@pytest.fixture
def w3(tmp_path, monkeypatch):
    return Tau3World(tmp_path, monkeypatch)


# ══ what a bridge leaves: its stream, the stdin it sent, harness.json, the bundle ════════════════════════════════
def t3_stream(*, servers=None, five=0.2, status="allowed", text="I've updated your email.") -> str:
    init = {"type": "system", "subtype": "init", "cwd": "/evo/work", "session_id": "S1", "uuid": "U1", "model": MODEL3,
            "tools": ["Bash", "Edit", "Read", "Skill", "Write", "mcp__tau2__get_user_information_by_email"],
            "mcp_servers": [{"name": "tau2", "status": "connected", "source": "sdk"}] if servers is None else servers,
            "claude_code_version": "2.1.281", "permissionMode": "bypassPermissions", "apiKeySource": "none",
            "skills": [], "plugins": [{"name": "agents-md", "path": "builtin"}], "memory_paths": {}}
    ev = [{"type": "control_request", "request_id": "r1", "request": {
              "subtype": "mcp_message", "server_name": "tau2", "message": {"method": "initialize", "jsonrpc": "2.0",
                                                                           "id": 0}}},
          init,
          {"type": "assistant", "message": {"id": "m1", "model": MODEL3, "content": [
              {"type": "thinking", "thinking": "verify the customer first", "signature": "sig"}]}},
          {"type": "assistant", "message": {"id": "m1", "model": MODEL3, "content": [
              {"type": "tool_use", "id": "toolu_01", "name": "mcp__tau2__get_user_information_by_email",
               "input": {"email": "a@b.c"}}]}},
          {"type": "rate_limit_event", "rate_limit_info": {"status": status, "unifiedWindows": {
              "five_hour": {"utilization": five, "resetsAt": 1790842800},
              "seven_day": {"utilization": 0.3, "resetsAt": 1791259200}}}},
          {"type": "user", "message": {"role": "user", "content": [
              {"type": "tool_result", "tool_use_id": "toolu_01", "content": [{"type": "text", "text": "{}"}]}]}},
          {"type": "assistant", "message": {"id": "m2", "model": MODEL3, "content": [{"type": "text", "text": text}],
                                            "stop_reason": "end_turn"}},
          {"type": "result", "subtype": "success", "is_error": False, "num_turns": 2, "terminal_reason": "completed",
           "result": text, "total_cost_usd": 0.1, "duration_ms": 5000, "usage": {"output_tokens": 3}}]
    return "\n".join(json.dumps(e) for e in ev) + "\n"


OK = {"rc": 0, "die": False, "never": False, "servers": None, "secret": False}


def harness_json(rc: int, spec: dict) -> dict:
    srv = [{"name": "tau2", "status": "connected", "source": "sdk"}] if spec["servers"] is None else spec["servers"]
    return {"exit_code": rc, "end": {"reason": "user_stop" if rc == 0 else "infrastructure_error", "detail": None},
            "graded_bundle": rc == 0, "agent_time": {"limit_s": 1200.0, "used_s": 23.2, "turns": 2},
            "customer": {"calls": 3, "seconds": 20.0, "failed_attempts": 0},
            "cc": {"mcp_servers": srv, "id_sources": {"_meta": 1}, "session_id": "S1",
                   "transcript": "/home/evo/.claude/projects/-evo-work/S1.jsonl"},
            "note": KEY if spec["secret"] else "no secret here"}


def tgz_of(files: dict, dirs=()) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for d in dirs:
            ti = tarfile.TarInfo(d)
            ti.type, ti.mode = tarfile.DIRTYPE, 0o755
            tf.addfile(ti)
        for n, data in sorted(files.items()):
            ti = tarfile.TarInfo(n)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


# ══ the pool machine: every command a real Tau3Box issues, answered in memory ═══════════════════════════════════
class Tau3Machine(TD.FakeHandle):
    """FakeHandle + the tau3 commands. SCRIPT[unit] = [attempt spec, ...] (OK's keys: rc, die = the bridge dies
    without an rc, never = it runs until SIGTERMed, servers = the init's mcp_servers, secret = the key's bytes in
    harness.json). `log` = (kind, cid or unit) in order: launch, sigterm, down, ship, pack, tar, wipe."""
    SCRIPT: dict = {}

    def __init__(self, tmp: Path):
        super().__init__(tmp)
        self.files: dict = {}
        self.log: list = []
        self.units: dict = {}
        self.cfs: dict = {}                      # cid -> {path in the container's pack: bytes}
        self.attempts: Counter = Counter()
        self.launches: list = []
        self.ctl_calls: list = []
        self.extracted: dict = {}                # /evo/... -> bytes of the last `evoctl extract`
        self.stray: list = []                    # listing rows nothing put there
        self.lk = threading.Lock()
        self.sb = SimpleNamespace(fs=SimpleNamespace(upload_file=self.upload, download_file=self.download))

    def upload(self, src, dst, timeout=None):
        with self.lk:
            self.files[dst] = Path(src).read_bytes()

    def download(self, src, dst):
        with self.lk:
            Path(dst).write_bytes(self.files[src])

    def unit_of_vm(self, vm: str) -> dict:
        return self.units[vm]

    def sh(self, cmd, *, timeout=600, env=None, what="", retries=3):
        with self.lk:
            self.sh_calls.append((what, cmd))
        if what == "bridge sha":
            paths = shlex.split(cmd.split(" 2>/dev/null")[0])[1:]
            return 0, "".join(f"{TD.sha(self.files[p])}  {p}\n" for p in paths if p in self.files)
        if what == "bridge dir":
            return 0, ""
        if what == "install bridge":
            _mv, _f, part, dst = shlex.split(cmd)
            with self.lk:
                self.files[dst] = self.files.pop(part)
            return 0, ""
        if what == "leftover bridges":
            self.log.append(("leftover", re.search(r"R=(\S+);", cmd).group(1)))
            return 0, "LEFTOVER_CLEAN\n"
        if what == "workspace owner":
            assert "/output" not in cmd, "a tau3 workspace gets no output/"
            return 0, "d 1000 755 0 \nWS_OK\n"
        if what == "listing":                     # the container as evoctl extract + the skills link left it
            rows = [["/evo/work", "D", "", 0], ["/evo/out", "D", "", 0]]
            for vm_path, data in sorted(self.extracted.items()):
                rows.append([vm_path, "F", TD.sha(data), len(data)])
                d = os.path.dirname(vm_path)
                while d not in ("/evo", "/") and [d, "D", "", 0] not in rows:
                    rows.append([d, "D", "", 0])
                    d = os.path.dirname(d)
            if self.extracted:
                rows += [["/evo/work/.claude", "D", "", 0], ["/evo/work/.claude/skills", "L", "/evo/skills", 0]]
            return 0, json.dumps(rows + self.stray) + "\n"
        if what == "unit dir":
            vm = re.search(r"rm -rf -- (\S+) &&", cmd).group(1)
            with self.lk:
                self.units[vm] = {"vm": vm, "state": "new"}
            return 0, ""
        if what == "job sha":
            p = re.search(r"sha256sum (\S+) ", cmd).group(1)
            return 0, TD.sha(self.files[p]) + "\n"
        if what == "launch":
            return self.launch(cmd, env)
        if what == "poll":
            return 0, self.poll(cmd)
        if what == "stop bridge":
            vm = re.search(r"balive (\S+?)\);", cmd).group(1)
            st = self.units[vm]
            st["stopped"] = True
            self.log.append(("sigterm", st["unit"]))
            return 0, "SIGTERM 4242\n"
        if what in ("end bridge", "bridge wipe"):
            return 0, "BRIDGE_GONE\n"
        if what == "ship bundle":
            vm = re.search(r"U=(\S+);", cmd).group(1)
            st = self.units[vm]
            assert st.get("rc") == 0 and f"{ctl.cdir(st['cid'])}/fs/evo/work" in cmd
            assert "rm -rf -- \"$W/tau2\"" in cmd and "cp -R -P --" in cmd
            self.log.append(("ship", st["cid"]))
            with self.lk:
                for n, data in self.bundle(st).items():
                    self.cfs[st["cid"]][f"evo/work/tau2/{n}"] = data
            return 0, "SHIP_OK\n"
        if what == "unit tar":
            vm = re.search(r"-C (\S+) \.", cmd).group(1)
            st = self.units[vm]
            self.log.append(("tar", st["cid"]))
            with self.lk:
                self.files[vm + ".tgz"] = self.unit_tgz(st)
            return 0, "TAR_OK\n"
        if what == "unit rm":
            return 0, "RM_OK\n"
        return super().sh(cmd, timeout=timeout, env=env, what=what, retries=retries)

    # -- the bridge
    def launch(self, cmd: str, env) -> tuple:
        assert env == {"CLAUDE_CODE_OAUTH_TOKEN": TOKEN, "TAU3_OR_KEY": KEY}, "the secrets travel in the exec's env"
        assert TOKEN not in cmd and KEY not in cmd and "sk-or-v1-" not in cmd
        vm = re.search(r"cat (\S+)/bridge\.pid", cmd).group(1)
        job = json.loads(self.files[vm + "/job.json"])
        unit = f"{job['task_id']}#s{job['sample']}"
        with self.lk:
            self.attempts[unit] += 1
            script = self.SCRIPT.get(unit) or [{}]
            spec = dict(OK, **script[min(self.attempts[unit], len(script)) - 1])
            st = self.units[vm]
            st.update(unit=unit, cid=job["container"]["cid"], job=job, spec=spec, polls=0, stopped=False, rc=None,
                      stream=t3_stream(servers=spec["servers"]).encode())
            self.launches.append({"cmd": cmd, "vm": vm, "job": job, "unit": unit})
        self.log.append(("launch", unit))
        return 0, "BRIDGE_PID=4242\n"

    def poll(self, cmd: str) -> str:
        assert "--name evolver.stderr --offset 0 --max 0 --du" in cmd and "out/claude.stream.jsonl" in cmd
        out = []
        for it in shlex.split(re.search(r"for s in (.*?); do", cmd).group(1)):
            cid, off, vm = it.split(":", 2)
            st = self.units[vm]
            st["polls"] += 1
            sp, stream, off = st["spec"], st["stream"], int(off)
            if st["polls"] < 2 or (sp["never"] and not st["stopped"]):
                line, data = "0 RUNNING 0 1", b""
            elif sp["die"] and not st["stopped"]:
                line, data = f"{len(stream)} RUNNING 0 0", stream[off:]
            else:
                st["rc"] = 75 if st["stopped"] else sp["rc"]
                line, data = f"{len(stream)} {st['rc']} 0 0", stream[off:]
            out.append(f"@@CID@@ {cid}\n{line}\n@@B64@@{base64.b64encode(data).decode()}\n")
        return "".join(out)

    def bundle(self, st) -> dict:
        tid = T.short_id(st["job"]["task_id"])
        return {"simulation.json": json.dumps({"task_id": tid, "sample": st["job"]["sample"]}).encode(),
                "run.json": b'{"tau2": {"version": "1.0.1"}}', "final_db.json": b"{}"}

    def unit_tgz(self, st) -> bytes:
        rc, sp = st.get("rc"), st["spec"]
        stdin = json.dumps({"type": "user", "session_id": "", "message": {"role": "user", "content": FIRST},
                            "parent_tool_use_id": None}) + "\n"
        files = {"./job.json": self.files[st["vm"] + "/job.json"], "./bridge.pid": b"4242\n",
                 "./bridge.log": b"tau2_cc_harness: ok\n", "./out/claude.stream.jsonl": st["stream"],
                 "./out/claude.stdin.jsonl": stdin.encode(), "./out/claude.stderr": b"",
                 "./out/harness.json": json.dumps(harness_json(rc, sp)).encode()}
        if rc is not None:
            files["./rc"] = f"{rc}\n".encode()
        if rc == 0:
            files.update({f"./snapshot/tau2/{n}": d for n, d in self.bundle(st).items()})
        return tgz_of(files, dirs=(".", "./out", "./snapshot"))

    # -- evoctl
    def ctl(self, args, *, timeout=600, what="", retries=2):
        with self.lk:
            self.ctl_calls.append(args)
        cid = re.search(r"--cid (\S+)", args).group(1)
        if args.startswith("stop-session"):
            self.log.append(("stop-session", cid))
            return 0, "", {"ok": True}
        if args.startswith("probe"):
            return 0, "", {"probe": {"view": {"ifaces": ["lo"]}}}
        if args.startswith("down") and "--wipe" in args:
            self.log.append(("wipe", cid))
            with self.lk:
                if cid in self.cc_containers:
                    self.cc_containers.remove(cid)
            return 0, "", {"verified": True, "wiped": True}
        if args.startswith("down"):
            self.log.append(("down", cid))
            return 0, "", {"ok": True, "scan": {}, "tmpfs_scan": {"hits": []}}
        if args.startswith("pack"):
            assert args.endswith("--part /evo/work --part /home/evo/.claude"), args
            self.log.append(("pack", cid))
            st = next(s for s in self.units.values() if s.get("cid") == cid)
            with self.lk:
                files = dict(self.cfs[cid])
                files["home/evo/.claude/projects/-evo-work/S1.jsonl"] = b'{"type":"user","transcript":true}\n' + st["stream"]
                self.files[f"{ctl.cdir(cid)}/state/out.tgz"] = tgz_of(files, dirs=("evo/work", "home/evo/.claude"))
            return 0, "", {"ok": True, "bytes": 1, "packed": ["/evo/work", "/home/evo/.claude"], "refused": []}
        if args.startswith("extract"):
            with tarfile.open(fileobj=io.BytesIO(self.files[f"{ctl.cdir(cid)}/state/upload.tgz"]), mode="r:gz") as tf:
                self.extracted = {"/" + mb.name: tf.extractfile(mb).read() for mb in tf.getmembers() if mb.isfile()}
            return 0, "", {"ok": True}
        raise AssertionError(f"unexpected evoctl {args}")


class FakeTau3Box(T.Tau3Box):
    """The container's own setup faked (as test_cc_leg_driver.FakeBox fakes it); every tau3 step is real."""

    def prepare(self):
        TD.ev("prepare", self.unit)
        self.rec["prepare"] = {"ok": True, "facts": {}, "limits": self.spec["limits"]}
        self.status = "prepared"

    def up(self, init_root):
        assert init_root is False
        self.status = "up"
        self.m.cc_containers.append(self.cid)
        self.m.cfs[self.cid] = {}

    def probe(self):
        self.rec["probe_pre"] = {"view": {"ifaces": ["lo"]}}
        self.rec["probe_pre_problems"] = []


class FakeTau3Grader(TD.FakeGrader):
    """InPlaceGrader's surface for a tau3 leg: built with runner_py = the tau3 venv (the P2 interface); a kept
    snapshot scores 1.0 when it holds evo/work/tau2/simulation.json (the user's grader reads it), else 0.0."""
    KEPT: dict = {}

    def __init__(self, handle, *, run_name, tag, knobs, plugin, task_rows, layout, overlay=None, max_procs=4,
                 max_heavy=1, heavy_bytes=None, job_timeout=5400, store=None, log=None, runner_py=None):
        assert runner_py == T.VM_TAU3_PY == "/opt/tau3/venv/bin/python"
        assert run_name == RUN3 and set(task_rows) == set(TRAIN3 + VAL3)
        assert layout[str(Path(plugin["files"]["benchmark_grader.py"]["path"]).parents[3] / "tau3_data_fx")] \
            == "/data/tau3_data_fx"
        self.runner_py = runner_py
        self.h, self.run_name, self.tag, self.task_rows, self.layout = handle, run_name, tag, task_rows, dict(layout)
        self.plugin = plugin
        self.max_procs, self.max_heavy, self.job_timeout = max_procs, max_heavy, job_timeout
        self.store = Path(store)
        self.store.mkdir(parents=True, exist_ok=True)
        self.setup = None
        self.jobs_sent, self.seen, self.regrades, self.statuses = {}, {}, {}, Counter()
        self.queue, self.has_output, self.released = [], {}, set()
        self.launches, self.cleaned = 0, False

    def enqueue_unit(self, *, job_id, task_id, cdir, snapshot_sha256, mac_tgz=None, cid=None, unit=None,
                     collected_utc=None):
        r = super().enqueue_unit(job_id=job_id, task_id=task_id, cdir=cdir, snapshot_sha256=snapshot_sha256,
                                 mac_tgz=mac_tgz, cid=cid, unit=unit, collected_utc=collected_utc)
        with tarfile.open(mac_tgz, "r:gz") as tf:
            names = tf.getnames()
        self.has_output[job_id] = "evo/work/tau2/simulation.json" in names
        FakeTau3Grader.KEPT[unit] = names
        return r

    def grader_record(self):
        return dict(super().grader_record(), interpreter=self.runner_py)


def make_run3(w: Tau3World, a, *, handle=None, poll_s=0, **kw) -> tuple:
    handle = handle or Tau3Machine(w.tmp / "m")
    TD.FakeGrader.SCORES, TD.FakeGrader.DELAY, TD.FakeGrader.INSTALL_ERROR = {}, set(), None
    FakeTau3Grader.KEPT = {}
    kw.setdefault("key_fn", lambda: KEY)
    run = D.LegRun(a, join_fn=TD.make_join(handle), box_cls=FakeTau3Box, grader_cls=FakeTau3Grader, plugin=w.plugin, poll_s=poll_s, grade_poll_s=0.02, **kw)
    return run, handle


def units_of(keys=TRAIN3, n=N3) -> list:
    return [f"{t}#s{k + 1}" for k in range(n) for t in keys]


# ══ (b) validate / plan / check-only / dry-run: the tau3 switch, no key read ═════════════════════════════════════
def test_tau3_validate_plan_check_only_and_dry_run_read_no_key(w3, capsys):
    run = D.LegRun(w3.args(), plugin=w3.plugin)
    plan = run.validate()
    assert run.tau3 and run.T is T and run.box_base is T.Tau3Box and run.poll_fn is T.poll_batch
    assert run.judge_fn is T.judge and run.init_problems_fn is T.init_problems
    # the wall: the bridge's agent time + 40 min, never cc_leg.wall_min (20)
    assert run.wall_s == 3600 and plan["wall_min"] == 60 and plan["need_s"] == 3600 + G.WAIT_MAX_S + 1800
    t3 = plan["tau3"]
    assert t3["agent_time_s"] == 1200.0 and t3["hard_stop_s"] == 3600.0 and t3["customer_model"] == CUSTOMER
    assert t3["data_root"] == str(w3.root) and t3["interpreter"] == "/opt/tau3/venv/bin/python"
    assert t3["bridge_sha256"] == {"tau2_cc_harness.py": L.sha256_file(HB.__file__),
                                   "tau2_bundle.py": L.sha256_file(HB.B.__file__),
                                   "tau3_retrieval.py": L.sha256_file(S.TR.__file__)}
    assert t3["first_message_template_sha256"] == HB.first_message_sha256() and t3["bridge_dir"] == "/opt/cc/tau3bin"
    # the bucket is keyed by the first message's template: never an SSB leg's
    assert run.instruction == HB.FIRST_MESSAGE_TEMPLATE
    assert plan["bucket"] == G.bucket_for(MODEL3, HB.FIRST_MESSAGE_TEMPLATE, None)
    assert plan["bucket"] != G.bucket_for(MODEL3, L.SOLVER_PREAMBLE, None)
    assert run.units == units_of() and run.questions[TRAIN3[0]] == json.dumps(
        {"task_id": TRAIN3[0], "category": T.DOMAIN, "instruction": ""})
    # check-only and dry-run through the launcher's argv: no key read (openrouter_key raises), nothing in the store
    argv = ["leg", "--tag", f"{RUN3}_base_train", "--key", "train", "--run-dir", str(w3.rd),
            "--split", str(w3.split), "--cache-dir", str(w3.tmp / "cache" / f"{RUN3}_e0_cache"), "--n-samples",
            str(N3), "--concurrency", "30"]
    assert D.main(argv + ["--check-only"]) == 0 and "check-only OK" in capsys.readouterr().out
    assert D.main(argv + ["--dry-run"]) == 7
    d = json.loads((w3.rd / f"{RUN3}_base_train{G.DRYRUN_SUFFIX}").read_text())
    assert d["tau3"]["hard_stop_s"] == 3600.0 and d["wall_min"] == 60 and d["bucket"] == plan["bucket"]
    assert not (w3.store_root / RUN3).exists() and not EVENTS


def test_tau3_validate_refusals(w3):
    def refused(match):
        with pytest.raises(D.Refuse, match=match):
            D.LegRun(w3.args(), plugin=w3.plugin).validate()
    w3.write_spec(user_sim_model="none")
    refused("user_sim_model 'none'")
    w3.write_spec(user_sim_model=CUSTOMER)
    for bad in ({"inputs": ["/x/in.xlsx"]}, {"answer_files": ["/elsewhere/task_004.json"]},
                {"grading": {"domain": T.DOMAIN, "retrieval_variant": "bm25_grep"}}):     # the old variant (refused)
        tasks = [dict(w3.tasks[0], **bad)] + w3.tasks[1:]
        with pytest.raises(D.Refuse, match="not tau3 banking_knowledge tasks"):
            D.LegRun(w3.args(), plugin=dict(w3.plugin, tasks=tasks)).validate()
    with pytest.raises(D.Refuse, match="'airline/task_009' is not a banking_knowledge/<task> id"):
        D.LegRun(w3.args(), plugin=dict(w3.plugin, tasks=w3.tasks + [dict(w3.tasks[0], id="airline/task_009")])).validate()
    # the switch is SPEC benchmark only: the same run dir as an SSB run takes none of the tau3 path
    w3.write_spec(benchmark="ssbv2")
    r = D.LegRun(w3.args(), plugin=w3.plugin)
    p = r.validate()
    assert not r.tau3 and r.T is None and "tau3" not in p and p["wall_min"] == 20 and r.box_base is D.LegBox


def test_an_ssb_leg_imports_nothing_of_tau3():
    code = ("import sys, types; sys.path[:0] = [{r!r}, {d!r}, {c!r}]; "
            "m = types.ModuleType('daytona'); m.CreateSandboxFromImageParams = m.Image = m.Resources = None; "
            "sys.modules.setdefault('daytona', m); import cc_leg_driver; "
            "bad = sorted(n for n in sys.modules if n in ('tau3_leg', 'tau3_cc_smoke', 'tau2_cc_harness', 'tau2_bundle', "
            "'tau3_retrieval')); "
            "print('TAU3_MODULES', bad)").format(r=str(ROOT), d=str(ROOT / "scripts" / "daytona"),
                                                 c=str(ROOT / "scripts" / "cc_solver"))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, cwd=str(ROOT))
    assert "TAU3_MODULES []" in r.stdout, r.stdout + r.stderr


# ══ (d) + (c) a whole tau3 leg: launch, poll, rc 0, the bundle shipped after down and before pack, graded ═════════
def test_tau3_leg_runs_ships_the_bundle_grades_and_feeds_the_evolver(w3):
    Tau3Machine.SCRIPT = {}
    run, m = make_run3(w3, w3.args())
    run.validate()
    rc = run.run()
    assert rc == 0, run.fatal + run.problems
    leg = json.loads((w3.rd / f"{RUN3}_base_train.json").read_text())
    assert leg["per_task"] == {"0": [1.0, 1.0], "1": [1.0, 1.0]} and leg["sample_pass"] == 4 and leg["grader_failed"] == []
    assert run.layout.verify(leg) == []
    # (c) the launch: neither secret in the command (asserted by the machine), the job = unit_job's + the customer
    assert len(m.launches) == 4
    for la in m.launches:
        task, k0 = G.sample_index(la["unit"])
        want = S.unit_job(T.short_id(task), k0 + 1, la["job"]["container"]["cid"], la["vm"], model=MODEL3,
                          overrides={"user_sim_model": CUSTOMER}, root=w3.root)
        assert la["job"] == want and la["job"]["user_sim_model"] == CUSTOMER and la["job"]["sample"] == k0 + 1
        assert la["vm"] == f"/opt/cc/{RUN3}/{RUN3}_base_train/bridge/{L.unit_dir_name(la['unit'])}"
        assert la["job"]["token_fd"] == 3 and la["job"]["key_fd"] == 4 and la["job"]["workdir"] == la["vm"] + "/snapshot"
        cmd = la["cmd"]
        assert "/opt/tau3/venv/bin/python /opt/cc/tau3bin/tau2_cc_harness.py --job job.json" in cmd
        assert '3<<<"$CLAUDE_CODE_OAUTH_TOKEN" 4<<<"$TAU3_OR_KEY"' in cmd
        assert "/sys/fs/cgroup/evobridge/cgroup.procs" in cmd and "BRIDGE_PID=" in cmd
    # the bridge installed once (sha-verified, renamed into place) and the leftover bridges of this tag cleared
    assert m.files["/opt/cc/tau3bin/tau2_cc_harness.py"] == Path(HB.__file__).read_bytes()
    assert m.files["/opt/cc/tau3bin/tau2_bundle.py"] == Path(HB.B.__file__).read_bytes()
    assert m.files["/opt/cc/tau3bin/tau3_retrieval.py"] == Path(S.TR.__file__).read_bytes()
    assert ("leftover", f"/opt/cc/{RUN3}/{RUN3}_base_train/bridge") in m.log
    st = G.state_read(run.store / "STATE.json")
    assert sorted(st["tau3"]["bridge"]["uploaded"]) == ["/opt/cc/tau3bin/tau2_bundle.py", "/opt/cc/tau3bin/tau2_cc_harness.py",
                                                       "/opt/cc/tau3bin/tau3_retrieval.py"]
    # (d) per unit: down -> ship -> pack (the bundle copied after down, before pack), then the unit tar and the wipe
    for b in run.queue.boxes:
        seq = [k for k, x in m.log if x == b.cid]
        assert seq == ["down", "ship", "pack", "tar", "wipe"], (b.unit, seq)
        assert "evo/work/tau2/simulation.json" in FakeTau3Grader.KEPT[b.unit]
        res = json.loads((b.rd / "RESULT.json").read_text())
        assert res["completed"] is True and res["claude_rc"] == 0 and res["infra_error"] is False
        assert res["init_problems"] == [] and res["bridge"]["mcp_servers"][0]["name"] == "tau2"
        assert res["bundle_shipped"] is True and "--max-turns" not in b.rec["argv"]
        assert (b.rd / "first_message.txt").read_text() == FIRST
        assert (b.rd / "bridge" / "snapshot" / "tau2" / "simulation.json").is_file()
        assert b.rec["log_round_trip"]["equal"] and b.rec["token_scan"]["hits"] == []
    # every unit graded through the grader built with the tau3 interpreter; the record's user message = FIRST
    assert run.grader.runner_py == "/opt/tau3/venv/bin/python" and len(run.grader.jobs_sent) == 4
    rec = json.loads(run.layout.record_paths(G.record_stem(run.questions[TRAIN3[0]], MODEL3, 0))[0].read_text())
    assert rec["trace"]["messages"][0] == {"role": "user", "content": FIRST}
    assert rec["trace"]["inline_score"] == 1.0
    assert leg["solver_harness"]["prompt_delivery"] == T.PROMPT_DELIVERY
    assert leg["runtime_env"]["egress"]["grading"] == T.EGRESS_GRADING
    assert leg["runtime_env"]["grader"]["interpreter"] == "/opt/tau3/venv/bin/python"
    assert leg["daytona"]["plan"]["wall_min"] == 60 and leg["daytona"]["containment"]["init_problems"] == 0
    assert not any(a.startswith("stop-session") for a in m.ctl_calls) and m.cc_containers == []
    # no secret anywhere: not on the machine's files, not in the store
    blob = b"".join(m.files.values())
    assert TOKEN.encode() not in blob and KEY.encode() not in blob
    assert S.secret_scan([run.store], {"t": TOKEN, "k": KEY, "p": S.OR_PREFIX})["hits"] == []
    # the evolver's material from this leg: the workspace holds tau2/, the trace is the CC transcript
    from src.orchestrator import evo_layout as EL
    out = w3.rd / "evo_e0"
    summ = EL.build(out, leg_json=w3.rd / f"{RUN3}_base_train.json", epoch=0)
    assert summ["rollouts"] == 4 and summ["mean_score"] == 1.0 and EL.check(out, leg_json=w3.rd / f"{RUN3}_base_train.json") == []
    ws = out / EL.WORKSPACES / f"{EL.safe_id(TRAIN3[0])}_s1"
    assert json.loads((ws / "tau2" / "simulation.json").read_text()) == {"task_id": "task_004", "sample": 1}
    assert sorted(p.name for p in (ws / "tau2").iterdir()) == ["final_db.json", "run.json", "simulation.json"]
    tr = (out / EL.TRACES / f"{EL.safe_id(TRAIN3[0])}_s1.jsonl").read_text()
    assert tr.startswith('{"type":"user","transcript":true}')


# ══ a skill leg's tau3 box: the promoted set and the one .claude/skills link, as an SSB skill leg; no output/ ═══════
def test_a_skill_legs_tau3_box_uploads_the_set_and_the_link_and_no_output(w3, tmp_path):
    import test_cc_leg_lib as TL
    sd = TL.make_skills(tmp_path / "sk")
    rows = G.skill_rows(sd)
    run = SimpleNamespace(store=tmp_path / "store", questions={TRAIN3[0]: json.dumps({"task_id": TRAIN3[0], "category":
                          T.DOMAIN, "instruction": ""})}, instruction=HB.FIRST_MESSAGE_TEMPLATE,
                          task_rows={TRAIN3[0]: w3.tasks[0]}, tag=f"{RUN3}_v1_train", key="train", epoch=1, skills_dir=sd)
    u = f"{TRAIN3[0]}#s2"
    T.prepare_units(run, [u])
    prep = json.loads((L.run_dir(run.store, u) / "PREP.json").read_text())
    assert prep["upload_rows"] == [] and prep["sample_idx"] == 2 and prep["prompt_delivery"] == T.PROMPT_DELIVERY
    assert prep["user_message_sha256"] == L.sha256_bytes(b"") and prep["task"] == w3.tasks[0]
    m = Tau3Machine(tmp_path / "m")
    B = FakeTau3Box.for_leg(rows, model=MODEL3, ctx={"run": RUN3, "tag": run.tag, "data_root": str(w3.root),
                                                     "user_sim_model": CUSTOMER, "key_fn": lambda: KEY})
    assert "key_fn" not in vars(B) and B.ctx["key_fn"]() == KEY and B.skill_leg and B.model == MODEL3
    b = B(m, run.store, u, "1001t010000", 3600, L.limits_for(1))
    assert b.vm_unit == f"/opt/cc/{RUN3}/{RUN3}_v1_train/bridge/banking_knowledge_task_004__s2" and b.sample == 2
    assert b.rec["argv"] == HB.stream_argv(MODEL3) and b.rec["env"]["CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR"] == "3"
    b.prepare()
    b.up(init_root=False)
    b.upload()
    b.toolchain()
    assert b.status == "ready" and b.rec["listing_problems"] == [] and b.rec["skills_link"]["rows"] == len(rows)
    assert set(m.extracted) == {r["vm_path"] for r in rows}
    assert not any("/output" in c for w_, c in m.sh_calls if w_ == "workspace owner")
    # the listing check is SSB's minus the one 'output/ missing' item: a stray file is still a problem
    m.stray = [["/evo/work/stray.txt", "F", TD.sha(b"x"), 1]]
    with pytest.raises(C.Refuse, match="listing inside != manifest"):
        b.upload()


# ══ (e) rc 75 twice / (f) a bridge gone without rc / (g) the wall: SIGTERM to the bridge ═════════════════════════
def test_rc_75_is_never_shipped_nor_graded_re_run_once_then_the_leg_fails(w3):
    u = f"{TRAIN3[1]}#s1"
    Tau3Machine.SCRIPT = {u: [{"rc": 75}, {"rc": 75}]}
    run, m = make_run3(w3, w3.args())
    run.validate()
    assert run.run() == 5
    f = json.loads((w3.rd / f"{RUN3}_base_train{G.FAILED_SUFFIX}").read_text())
    assert any(f"{u}: not completed after 2 attempt(s)" in p for p in f["problems"])
    cids = [b.cid for b in run.queue.boxes if b.unit == u]
    assert len(cids) == 2 and m.attempts[u] == 2 and run.retries == {u: 1}
    assert not [x for k, x in m.log if k == "ship" and x in cids], "an infrastructure end is never shipped"
    assert u not in run.jobs.values() and u not in run.samples, "nor graded"
    for c in cids:
        assert [k for k, x in m.log if x == c] == ["down", "pack", "tar", "wipe"]
    res = json.loads((next(b for b in reversed(run.queue.boxes) if b.unit == u).rd / "RESULT.json").read_text())
    assert res["completed"] is False and res["infra_error"] is True and res["claude_rc"] == 75


def test_a_bridge_gone_without_rc_is_re_run(w3):
    u = f"{TRAIN3[0]}#s2"
    Tau3Machine.SCRIPT = {u: [{"die": True}, {}]}
    run, m = make_run3(w3, w3.args())
    run.validate()
    assert run.run() == 0, run.fatal + run.problems
    first = next(b for b in run.queue.boxes if b.unit == u)
    assert first.status_ended == "container_died" and first.res["completed"] is False
    assert run.retries == {u: 1} and m.attempts[u] == 2
    assert not [x for k, x in m.log if k == "ship" and x == first.cid]
    assert json.loads((w3.rd / f"{RUN3}_base_train.json").read_text())["sample_pass"] == 4


def test_the_wall_sends_sigterm_to_the_bridge_not_stop_session(w3):
    u = f"{TRAIN3[0]}#s1"
    Tau3Machine.SCRIPT = {u: [{"never": True}, {}]}
    run, m = make_run3(w3, w3.args(), poll_s=0.005)
    run.validate()
    run.wall_s = 1                                   # the backstop (agent_time_s + 40 min), shortened for the test
    assert run.run() == 0, run.fatal + run.problems
    first = next(b for b in run.queue.boxes if b.unit == u)
    assert ("sigterm", u) in m.log and first.stopped == "wall 0 min" and first.rec["stopped"]["out"] == "SIGTERM 4242"
    assert not any(a.startswith("stop-session") for a in m.ctl_calls)
    stop_cmd = next(c for w_, c in m.sh_calls if w_ == "stop bridge")
    assert "kill -TERM" in stop_cmd and first.vm_unit in stop_cmd
    assert first.rc == 75 and not first.res["completed"] and run.retries == {u: 1}


# ══ (h) the init: one tau2 server, connected ══════════════════════════════════════════════════════════════════════
def test_init_problems_want_exactly_one_connected_tau2_server():
    init = json.loads(t3_stream().splitlines()[1])
    assert T.init_problems(init, model=MODEL3) == []
    assert any(p.startswith("mcp servers ") for p in L.init_problems(init, model=MODEL3))   # SSB's check refuses it
    for sv in ([], [{"name": "tau2", "status": "failed"}], [{"name": "other", "status": "connected"}],
               [{"name": "tau2", "status": "connected"}, {"name": "x", "status": "connected"}]):
        probs = T.init_problems(dict(init, mcp_servers=sv), model=MODEL3)
        assert len(probs) == 1 and probs[0].startswith("mcp servers "), (sv, probs)
    assert T.init_problems(None) == ["no init event"]
    assert any("disallowed tools" in p for p in T.init_problems(dict(init, tools=init["tools"] + ["WebFetch"]),
                                                                model=MODEL3))


def test_judge_recomputes_the_init_reason(tmp_path):
    def box(text, rc=0, status="finished"):
        (tmp_path / "stream.jsonl").write_text(text)
        return SimpleNamespace(rd=tmp_path, model=MODEL3, unit="banking_knowledge/task_004#s1", status=status, rc=rc,
                               rec={"argv": HB.stream_argv(MODEL3)})
    g = T.judge(box(t3_stream()), 0.85)
    assert g["stop_reasons"] == [] and g["init_problems"] == []
    g = T.judge(box(t3_stream(servers=[{"name": "tau2", "status": "failed"}], five=0.9)), 0.85)
    assert [r.split(" ")[0] for r in g["stop_reasons"]] == ["init"]      # a utilization level alone stops nothing
    g = T.judge(box(t3_stream(servers=[{"name": "tau2", "status": "failed"}], status="rejected")), 0.85)
    assert g["stop_reasons"][0].startswith("quota rejected") and g["stop_reasons"][1].startswith("init ")
    # an infrastructure end before claude started (no init) is a re-run, not a defect; a graded end without one is
    assert T.judge(box("", rc=75), 0.85)["stop_reasons"] == []
    assert T.judge(box("", rc=0), 0.85)["stop_reasons"] == ["init ['no init event']"]


def test_a_failed_tau2_server_is_a_defect_that_fails_the_leg(w3):
    u = f"{TRAIN3[0]}#s1"
    Tau3Machine.SCRIPT = {u: [{"servers": [{"name": "tau2", "status": "failed"}]}]}
    run, m = make_run3(w3, w3.args(concurrency=1))
    run.validate()
    assert run.run() == 5
    f = json.loads((w3.rd / f"{RUN3}_base_train{G.FAILED_SUFFIX}").read_text())
    g = f["daytona"]["guard"]
    assert g["kind"] == "defect" and g["reason"].startswith("init ") and "tau2" in g["reason"]


# ══ (i) a secret's bytes in a downloaded file: TokenHit, the leg FAILS ═════════════════════════════════════════════
def test_a_secret_in_a_downloaded_file_fails_the_leg(w3):
    u = f"{TRAIN3[0]}#s1"
    Tau3Machine.SCRIPT = {u: [{"secret": True}]}
    run, m = make_run3(w3, w3.args(concurrency=1))
    run.validate()
    assert run.run() == 5
    b = next(x for x in run.queue.boxes if x.unit == u)
    hits = b.rec["token_scan"]["hits"]
    assert any(h.endswith("harness.json:openrouter-key") for h in hits) and any("openrouter-prefix" in h for h in hits)
    f = json.loads((w3.rd / f"{RUN3}_base_train{G.FAILED_SUFFIX}").read_text())
    assert f["daytona"]["guard"]["kind"] == "defect" and "token hit" in f["daytona"]["guard"]["reason"]
    assert u not in run.samples


# ══ the key: read ONCE at run time, never by validate; a missing key refuses the leg before the machine ════════════
def test_the_key_is_read_once_at_run_time_and_a_missing_key_refuses(w3, monkeypatch):
    calls = []

    def missing():
        calls.append(1)
        raise S.ORK.KeyMissing("OPENROUTER_API_KEY is set neither in the environment nor in /x/.env")
    monkeypatch.setattr(S.ORK, "openrouter_key", missing)
    run, m = make_run3(w3, w3.args(), key_fn=None)
    run.validate()
    assert calls == []
    assert run.run() == 5 and calls == [1]
    f = json.loads((w3.rd / f"{RUN3}_base_train{G.FAILED_SUFFIX}").read_text())
    assert f["problems"][0].startswith("refused: the customer's OpenRouter key: OPENROUTER_API_KEY is set neither")
    assert not [k for k, _ in EVENTS if k == "join"] and not m.launches
    # with the key: read once for the whole leg (four units)
    (w3.rd / f"{RUN3}_base_train{G.FAILED_SUFFIX}").unlink()
    n = []
    monkeypatch.setattr(S.ORK, "openrouter_key", lambda: n.append(1) or KEY)
    Tau3Machine.SCRIPT = {}
    run2, m2 = make_run3(w3, w3.args(), handle=Tau3Machine(w3.tmp / "m2"), key_fn=None)
    run2.validate()
    assert run2.run() == 0, run2.fatal + run2.problems
    assert n == [1] and len(m2.launches) == 4


# ══ the bridge's install, the leftover bridges, the poll ══════════════════════════════════════════════════════════
def test_install_bridge_is_idempotent_and_refuses_a_mismatch(tmp_path):
    m = Tau3Machine(tmp_path)
    r1 = T.install_bridge(m)
    assert len(r1["uploaded"]) == 3 and set(r1["sha256"]) == {"/opt/cc/tau3bin/tau2_cc_harness.py",
                                                              "/opt/cc/tau3bin/tau2_bundle.py",
                                                              "/opt/cc/tau3bin/tau3_retrieval.py"}
    assert r1["sha256"]["/opt/cc/tau3bin/tau3_retrieval.py"] == L.sha256_file(S.TR.__file__)
    assert T.BRIDGE_FILES is S.BRIDGE_FILES and len(T.BRIDGE_FILES) == 3
    assert T.install_bridge(m)["uploaded"] == []
    m.files["/opt/cc/tau3bin/tau2_bundle.py"] = b"tampered"
    assert T.install_bridge(m)["uploaded"] == ["/opt/cc/tau3bin/tau2_bundle.py"]
    m.files["/opt/cc/tau3bin/tau3_retrieval.py"] = b"tampered"                 # the alltools module is checked too
    assert T.install_bridge(m)["uploaded"] == ["/opt/cc/tau3bin/tau3_retrieval.py"]
    del m.files["/opt/cc/tau3bin/tau3_retrieval.py"]                            # and a missing one is re-uploaded
    assert T.install_bridge(m)["uploaded"] == ["/opt/cc/tau3bin/tau3_retrieval.py"]
    m.upload = lambda src, dst, timeout=None: m.files.__setitem__(dst, b"partial")
    m.sb = SimpleNamespace(fs=SimpleNamespace(upload_file=m.upload, download_file=m.download))
    m.files["/opt/cc/tau3bin/tau2_bundle.py"] = b"tampered"
    with pytest.raises(C.Refuse, match="does not verify"):
        T.install_bridge(m)


def test_kill_leftover_bridges_and_the_shell_helpers():
    calls = []
    h = SimpleNamespace(sh=lambda cmd, **kw: calls.append(cmd) or (0, "KILLED 77 /opt/cc/r/t/bridge/x__s1\nLEFTOVER_CLEAN\n"))
    r = T.kill_leftover_bridges(h, "/opt/cc/r/t/bridge")
    assert r == {"root": "/opt/cc/r/t/bridge", "killed": ["77"]}
    cmd = calls[0]
    assert 'kill -TERM "$P"' in cmd and 'kill -KILL "$P"' in cmd and 'rm -rf -- "$R"' in cmd
    assert '[ "$(readlink "/proc/$p/cwd" 2>/dev/null)" = "$1" ]' in cmd, "a reused pid is never killed"
    h2 = SimpleNamespace(sh=lambda cmd, **kw: (0, "rm: cannot remove"))
    with pytest.raises(C.Refuse, match="could not be cleaned"):
        T.kill_leftover_bridges(h2, "/opt/cc/r/t/bridge")
    assert T.bridge_root("r", "t") == "/opt/cc/r/t/bridge" and T.short_id("banking_knowledge/task_010") == "task_010"
    with pytest.raises(C.Refuse):
        T.short_id("airline/task_010")


def test_poll_batch_reads_rc_first_and_alive_needs_container_and_bridge():
    boxes = [SimpleNamespace(cid="cc-a-1", off=0, vm_unit="/opt/cc/r/t/bridge/a__s1"),
             SimpleNamespace(cid="cc-b-1", off=7, vm_unit="/opt/cc/r/t/bridge/b__s1")]
    seen = []

    def sh(cmd, **kw):
        seen.append(cmd)
        return 0, ("@@CID@@ cc-a-1\n12 RUNNING 4096 1\n@@B64@@" + base64.b64encode(b"hello").decode() + "\n"
                   "@@CID@@ cc-b-1\n40 0 100 0\n@@B64@@\n")
    res = T.poll_batch(SimpleNamespace(sh=sh), boxes)
    assert res == {"cc-a-1": (12, "RUNNING", 4096, "1", b"hello"), "cc-b-1": (40, "0", 100, "0", b"")}
    cmd = seen[0]
    assert "cc-a-1:0:/opt/cc/r/t/bridge/a__s1" in cmd and "cc-b-1:7:/opt/cc/r/t/bridge/b__s1" in cmd
    assert cmd.index('rc=$(cat "$u/rc"') < cmd.index('n=$(stat -c %s "$f"') < cmd.index("tail -c +$((o + 1))")
    assert '[ "$ca" = 1 ] && [ -n "$b" ] && al=1' in cmd and f"head -c {C.CHUNK_B}" in cmd

    def broken(cmd, **kw):
        raise H.HandleError("exec 'poll' failed 2x")
    with pytest.raises(C.Refuse):                     # poll_once counts it per box; never a crash of the queue
        T.poll_batch(SimpleNamespace(sh=broken), boxes)


# ══ (j) the pool-wide cap of rollout containers, all legs together ═══════════════════════════════════════════════
class MiniBox:
    def __init__(self, m, root, unit, stamp, wall_s, limits):
        self.unit, self.rd, self.rec = unit, Path(root) / "runs" / L.unit_dir_name(unit), {}
        self.rd.mkdir(parents=True, exist_ok=True)


class SetupFreeQueue(D.LegQueue):
    def _setup(self, b):                              # the slot is held; nothing runs
        pass


def pooled(world, key, cap, holder, pool_root, sleeps):
    run = D.LegRun(world.args(key, skills=False), plugin=world.plugin, sleep=sleeps.append)
    run.validate()
    run.cc["concurrency"] = cap
    run.holder, run.pool_reg = holder, H.P.Registry(pool_root)
    units = run.units
    q = SetupFreeQueue(run, SimpleNamespace(t_created=time.time(), ttl=10 ** 6, lock=threading.Lock()), run.store,
                       list(units), "20261001T010000", 3600, run.limits, TOKEN, run.conc, box_cls=MiniBox,
                       ttl_s=10 ** 7, poll_s=0)
    return run, q


def inflight_files(pool_root: Path) -> dict:
    d = pool_root / D.POOL_INFLIGHT_DIR
    return {json.loads(f.read_text())["holder"]: json.loads(f.read_text())["n"] for f in sorted(d.glob("*.json"))}


def test_the_pool_cap_holds_across_legs_ignores_dead_holders_and_sleeps_when_full(tmp_path, monkeypatch):
    clean_env(monkeypatch)
    world = TD.World(tmp_path, monkeypatch)
    pool = tmp_path / "pool_run"
    sleeps = []
    r1, q1 = pooled(world, "train", 3, "ccleg-a", pool, sleeps)        # 4 units, own ceiling 30
    r2, q2 = pooled(world, "val", 3, "ccleg-b", pool, sleeps)          # 2 units
    # a dead holder's file (pid gone) is never counted, and is removed
    (pool / D.POOL_INFLIGHT_DIR).mkdir(parents=True)
    (pool / D.POOL_INFLIGHT_DIR / "ghost.json").write_text(json.dumps({"holder": "ghost", "pid": 999999, "n": 3}))
    q1.admit()
    assert len(q1.inflight) == 3 and q1.conc == 3 and inflight_files(pool) == {"ccleg-a": 3}
    q2.admit()                                                       # no room, nothing in flight: it waits, no spin
    assert len(q2.inflight) == 0 and q2.capped and sleeps == [D.PRIORITY_HOLD_S]
    assert inflight_files(pool) == {"ccleg-a": 3, "ccleg-b": 0}
    q1.inflight.pop()                                                # one of leg 1's rollouts is collected + wiped
    q1.admit()                                                       # leg 1 refills its own slot first (1 pending)
    assert len(q1.inflight) == 3 and len(q1.pending) == 0
    q2.admit()
    assert len(q2.inflight) == 0 and sleeps == [D.PRIORITY_HOLD_S] * 2
    q1.inflight[:] = q1.inflight[:1]                                 # two more of leg 1's end
    q1.admit()
    assert inflight_files(pool)["ccleg-a"] == 1
    q2.admit()
    assert len(q2.inflight) == 2 and len(q1.inflight) + len(q2.inflight) == 3 and not q2.capped
    assert inflight_files(pool) == {"ccleg-a": 1, "ccleg-b": 2}
    # a leg's own concurrency still bounds it under a larger cap; teardown removes the holder's file
    r1.cc["concurrency"] = 30
    r1.conc = 1
    q1.inflight.clear()
    q1.pending.extend(["Debugging/03_08#s1", "Debugging/03_08#s2"])
    q1.admit()
    assert len(q1.inflight) == 1 and not q1.capped
    r1.queue = None
    r1.teardown()
    assert "ccleg-a" not in inflight_files(pool)


def test_two_legs_on_one_pool_never_exceed_the_cap_together(tmp_path, monkeypatch):
    clean_env(monkeypatch)
    world = TD.World(tmp_path, monkeypatch)
    pool = tmp_path / "pool_run"
    sleeps = []
    r1, q1 = pooled(world, "train", 3, "ccleg-a", pool, sleeps)
    r2, q2 = pooled(world, "val", 3, "ccleg-b", pool, sleeps)
    for q in (q1, q2):
        q.pending.extend(f"Debugging/03_08#s{k}" for k in range(3, 40))
    peak, bad = [0], []
    real = D.LegRun.pool_note

    def note(self, n):                                               # called under the registry flock
        tot = len(q1.inflight) + len(q2.inflight)
        peak[0] = max(peak[0], tot)
        if tot > 3:
            bad.append(tot)
        real(self, n)
    monkeypatch.setattr(D.LegRun, "pool_note", note)

    def drive(q):
        for i in range(150):
            q.admit()
            if q.inflight and i % 2:
                q.inflight.pop(0)
    th = [threading.Thread(target=drive, args=(q,)) for q in (q1, q2)]
    for t in th:
        t.start()
    for t in th:
        t.join(timeout=60)
    assert not bad and peak[0] == 3, (bad, peak)
