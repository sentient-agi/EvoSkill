"""scripts/daytona/cc_leg_driver.py -- the Claude Code leg process.

NO DAYTONA, NO CLAUDE, NO KEY. The pool machine is a FakeHandle (cc_pool_host.PoolMachineHandle's surface, an event
log for the wipe / release order), the join a function returning it, the boxes a FakeBox in the shape of
tests/test_cc_solver.py's (cc_solver.Box's steps answered locally -- collect writes a real container_out.tgz and its
extraction -- the REAL Box.__init__ / Box.result run on the driver's own PREP.json), the in-place grader a
FakeGrader with the InPlaceGrader surface (install, enqueue_unit, poll, regrade) and scripted scores, the clock a fake
for the rate-limit wait. Everything else -- LegQueue, the grade poller, the samples, LegLayout and its verify, the
loop's own leg checks, evo_layout -- is real. Every completed unit is graded by the run's grade(task, snapshot) (the fake scores a
snapshot with no output 0.0, as the user's SSB grader does), a score is a number from 0 to 1, a twice-failed grade is
recorded as grader_failed and the leg continues, a `refused` line or an install() fault FAILS the leg.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sys
import tarfile
import threading
import time
import types
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "daytona"))
sys.path.insert(0, str(ROOT / "scripts" / "cc_solver"))
sys.path.insert(0, str(ROOT / "tests"))

class _Params:                                   # the same stub shape as tests/test_cc_pool_host.py's (one session)
    def __init__(self, **kw):
        self.__dict__.update(kw)


fake_daytona = types.ModuleType("daytona")
fake_daytona.CreateSandboxFromImageParams = _Params
fake_daytona.Image = types.SimpleNamespace(from_dockerfile=lambda p: ("dockerfile", p))
fake_daytona.Resources = lambda **kw: kw
sys.modules.setdefault("daytona", fake_daytona)

import cc_leg_driver as D  # noqa: E402
import cc_leg_lib as G  # noqa: E402
import cc_pool_host as H  # noqa: E402
import cc_solver as C  # noqa: E402
import cc_solver_lib as L  # noqa: E402
import test_cc_leg_lib as TL  # noqa: E402  (the lib tests' fixtures: streams, split, skills)
from src.orchestrator import pool_container_evolver as pce  # noqa: E402

RUN = "cc_drv_fx"
MODEL = TL.MODEL
TRAIN, VAL, N = TL.TRAIN, TL.VAL, TL.N
GF = G.GRADER_FAILED
TOKEN = "sk-ant-oat01-" + "k" * 60
INFRA_SRC = ROOT / "tests" / "fixtures" / "ssb_leg_pool" / "infra_cc.json"
EVENTS: list = []
EV_LOCK = threading.Lock()


def ev(kind, unit=None):
    with EV_LOCK:
        EVENTS.append((kind, unit))


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def stream(*, five=0.2, status="allowed", reset=1790495400, capped=False, skill=None, skills_listed=(),
           api_error=False, thinking_text="look at the workbook first") -> str:
    """test_cc_leg_lib.cc_stream with the guard reading and the api_error / thinking variants the driver reacts to."""
    s = TL.cc_stream(capped=capped, skill=skill, skills_listed=skills_listed)
    evs = [json.loads(ln) for ln in s.splitlines()]
    for e in evs:
        if e.get("type") == "rate_limit_event":
            e["rate_limit_info"] = {"status": status, "unifiedWindows": {
                "five_hour": {"utilization": five, "resetsAt": reset},
                "seven_day": {"utilization": 0.3, "resetsAt": reset + 100000}}}
        if e.get("type") == "assistant":
            for c in e["message"]["content"]:
                if c.get("type") == "thinking":
                    c["thinking"] = thinking_text
        if api_error and e.get("type") == "result":
            e.update(subtype="success", is_error=True, terminal_reason="api_error", api_error_status=529)
    return "\n".join(json.dumps(e) for e in evs) + "\n"


# ══ the world: run dir, split, frozen plug-in, token, skills, probe record ════════════════════════
class World:
    def __init__(self, tmp: Path, monkeypatch, *, skills=True, probe_ok=True):
        self.tmp = tmp
        self.split, self.split_sha = TL.make_split(tmp)
        self.rd = tmp / "outputs" / RUN
        self.rd.mkdir(parents=True)
        (self.rd / "SPEC.json").write_text(json.dumps({"spec": {
            "name": RUN, "solver_model": MODEL, "solver_harness": "claude_code", "n_samples": N, "agent_turns": 45,
            "split_path": str(self.split), "split_sha256": self.split_sha, "benchmark": "ssbv2"}}))
        self.token = tmp / "cc_token"
        self.token.write_text(TOKEN + "\n")
        os.chmod(self.token, 0o600)
        inf = json.loads(INFRA_SRC.read_text())
        inf["solver"]["key_file"] = inf["claude_code"]["token_file"] = str(self.token)
        inf["data"]["overlay_patches_json"] = None
        inf["data"]["overlay_patches_sha256"] = None
        self.inf = inf
        self.write_inf()
        self.store_root, self.tasks_root = tmp / "store", tmp / "tasks_root"
        self.cache_e1 = tmp / "cache" / f"{RUN}_e1_cache"
        self.rt = {"prompt_mode": "template", "leg_timeout_s": 43200, "evolver_evidence_layout": "rollouts",
                   "evolver_evidence": "all_epochs", "evolver_host": "daytona", "evolver_sandbox": True,
                   "shared_dir": str(self.rd),
                   "daytona_leg": {"store_root": str(self.store_root), "tasks_root": str(self.tasks_root), "pool": True,
                                   "pool_keys": ["train", "val"], "pool_sandboxes": 1, "pool_ttl_minutes": 1440,
                                   "pool_ttl_margin_min": 255},
                   "evolver_daytona": {"placement": "pool_container", "oauth_token_file": str(self.token),
                                       "machine_wait_s": 900},
                   "cc_leg": {"concurrency": 30, "wall_min": 90,
                              "machine_wait_s": 1800, "grade_max_procs": 4, "grade_max_heavy": 1},
                   "base_leg": str(self.rd / f"{RUN}_base_train.json"), "base_leg_val": str(self.rd / f"{RUN}_base_val.json"),
                   "digest_caches": [str(tmp / "cache" / f"{RUN}_e0_cache")], "digest_bucket": ""}
        self.write_rt()
        # the task files the frozen rows name (input + reference per task)
        data = tmp / "data"
        data.mkdir()
        self.tasks = []
        for t in TRAIN + VAL:
            d = data / t.replace("/", "_")
            d.mkdir()
            (d / "in.xlsx").write_bytes(b"PK-in-" + t.encode())
            (d / "golden.xlsx").write_bytes(b"PK-golden-" + t.encode())
            self.tasks.append({"id": t, "category": t.split("/")[0], "instruction": "Fill the model.",
                               "inputs": [str(d / "in.xlsx")], "answer_files": [str(d / "golden.xlsx")],
                               "grading": {"answer_position": "A1"}, "split": "train" if t in TRAIN else "validation"})
        # The frozen plug-in in cc_leg_lib.frozen_plugin's shape -- the three pinned setup
        # files (the user's grader among them) and tasks.json -- handed to the driver (plugin=) or returned by the
        # monkeypatched G.frozen_plugin on the command path
        setup = self.rd / "setup"
        setup.mkdir()
        files = {}
        for n in ("benchmark_plugin.py", "benchmark_grader.py", "plugin_api.py"):
            (setup / n).write_text(f"# {n}\n")
            files[n] = {"path": str(setup / n), "sha256": sha((setup / n).read_bytes())}
        (setup / "tasks.json").write_text(json.dumps(self.tasks))
        self.plugin = {"files": files, "tasks": self.tasks, "tasks_json_sha256": sha((setup / "tasks.json").read_bytes())}
        self.task_rows = {t["id"]: t for t in self.tasks}
        self.questions, self.idx = G.tasks_json_questions(self.tasks)
        for t in TRAIN + VAL:
            d = data / t.replace("/", "_")
            assert self.questions[t] == json.dumps({"task_id": t, "category": t.split("/")[0],
                                                    "instruction": "Fill the model.", "spreadsheet_path": str(d / "in.xlsx"),
                                                    "spreadsheet_filename": "in.xlsx"})
            assert self.idx[t]["input"] == d / "in.xlsx" and self.idx[t]["golden"] == d / "golden.xlsx"
        self.skills = None
        if skills:
            self.skills = TL.make_skills(self.rd)
            self.skills.rename(self.rd / "skills_v1")
            self.skills = self.rd / "skills_v1"
            self.write_probe(ok=probe_ok)
        cc_local = tmp / "claude-fake"
        cc_local.write_bytes(b"#!fake claude")
        monkeypatch.setattr(H, "CC_LOCAL", cc_local)
        # the command path (no injected plug-in) reads the fake frozen plug-in too
        monkeypatch.setattr(G, "frozen_plugin", lambda rd: json.loads(json.dumps(self.plugin)))
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.delenv("EVOSKILL_SOLVER_MODEL", raising=False)
        for k in D.OPENROUTER_PINS + (D.SUBSET_ENV,):
            monkeypatch.delenv(k, raising=False)
        EVENTS.clear()

    def write_inf(self):
        (self.rd / "INFRA.json").write_text(json.dumps(self.inf))

    def write_rt(self):
        (self.rd / "RUNTIME.json").write_text(json.dumps(self.rt))

    def write_probe(self, ok=True):
        p = self.rd / D.PROBE_RECORD
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"cc_version": "2.1.281", "utc": "2026-09-27T12:00:00Z", "sandbox_id": "sb-probe",
                                 "primary": {"ok": ok, "problems": [] if ok else ["init.skills [] != ['probe-skill']"]}}))

    def layout_map(self) -> dict:
        """What the driver hands the grader: plugin_runner.material_layout(INFRA data.root, the split, the repo)."""
        return {self.inf["data"]["root"]: "/data/spreadsheetbench-v2", str(self.split.resolve()): "/data/split.json",
                str(D.REPO): "/data/EvoSkill"}

    def args(self, key="train", *, skills=True, epoch=1, **over):
        tag = G.leg_tag(RUN, epoch, key)
        a = dict(tag=tag, key=key, run_dir=str(self.rd), split=str(self.split),
                 cache_dir=str(self.tmp / "cache" / f"{RUN}_e{epoch}_cache"),
                 skills_dir=str(self.rd / f"skills_v{epoch}") if skills else "", n_samples=N, concurrency=30,
                 check_only=False, dry_run=False)
        a.update(over)
        return SimpleNamespace(**a)


@pytest.fixture
def world(tmp_path, monkeypatch):
    return World(tmp_path, monkeypatch)


# ══ the fakes ═══════════════════════════════════════════════════════════════════════════════════
class FakeLease:
    def __init__(self, holder, path):
        self.holder, self.path, self.cap_ts, self.writes, self.errors = holder, path, time.time() + 90000, 1, []

    def summary(self):
        return {"holder": self.holder, "path": str(self.path), "cap_ts": self.cap_ts, "writes": self.writes,
                "errors": self.errors}

    def release(self):
        return True


class FakeHandle:
    """cc_pool_host.PoolMachineHandle's surface on no machine; `sh` answers the two host-side commands the driver's
    boxes issue for real (the skills link, the cgroup peak)."""

    def __init__(self, tmp: Path):
        self.sid = "11111111-fake-machine"
        self.name = f"{pce.POOL_PREFIX}rpabcde-m01-120000"
        self.machine_name = "m01"
        now = time.time()
        self.row = {"name": "m01", "sandbox_id": self.sid, "sandbox_name": self.name, "state": "ready", "created_ts": now - 600,
                    "ready_ts": now - 300, "ttl_minutes": 1440, "ttl_expires_ts": now + 1440 * 60,
                    "retire_at_ts": now + (1440 - 255) * 60, "domain_allow_list": ["openrouter.ai", "api.anthropic.com"],
                    "leased_by": []}
        self.identity = {"libreoffice": "LibreOffice 25.2.3.2 520(Build:2)", "solver_stack": "3.1.3 2.2.0 1.26.4",
                         "solver_python": "Python 3.11.9", "runner_python": "Python 3.13.9",
                         "runner_openpyxl": "3.1.5 lxml 6.1.1"}
        self.summary = {"identity": self.identity, "egress": {"must_reach": ["anthropic", "openrouter"]},
                        "grade_selftest": {"golden_vs_golden_passed": True}, "bundle": {"shim_sha256": "ab" * 32},
                        "readback": {"snapshot_ref": "daytona-88:daytona"}, "install": {}, "sandbox_id": self.sid,
                        "sandbox_name": self.name, "granted_ttl_min": 1440, "ready_utc": "2026-09-27T12:00:00Z"}
        self.rec = {"registry": {"host_pid": 4242, "pool_dir": str(tmp / "pool"), "host_script": "cc_pool_host.py",
                                 "run_dir": str(tmp), "owner": "cc_pool_host"},
                    "readback_join": {"refreshed": True, "snapshot_ref": "daytona-88:daytona"},
                    "join": {"spawned": False, "waited_s": 0.1}}
        self.dir = tmp / "handle"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.inf = None
        self.dle = SimpleNamespace(ledger=lambda **kw: None)
        self.ttl = 1440
        self.t_created = self.row["created_ts"]
        self.retire_at_ts = self.row["retire_at_ts"]
        self.token_name = None
        self.cc_containers: list = []
        self.sh_calls: list = []

    def save(self):
        pass

    def sh(self, cmd, *, timeout=600, env=None, what="", retries=3):
        self.sh_calls.append((what, cmd))
        assert not env or "CLAUDE_CODE_OAUTH_TOKEN" in env
        if "ln -s" in cmd:
            assert "ln -s /evo/skills " in cmd and "chown -h 1000:1000" in cmd and "install -d -o 1000 -g 1000" in cmd
            return 0, "LINK_OK\n"
        if "memory.peak" in cmd:
            return 0, "== memory.peak\n300000000\n== pids.peak\n12\n"
        return 0, ""

    def containers(self):
        return {"cc": list(self.cc_containers), "other": ["e1-evolver-x"]}

    def wipe(self, cids, *, tag=""):
        ev("handle_wipe", tuple(cids))
        for c in cids:
            assert c.startswith("cc-")
            if c in self.cc_containers:
                self.cc_containers.remove(c)
        return {c: {"verified": True, "wiped": True} for c in cids}

    def release(self):
        ev("release", None)
        return True


def make_join(handle):
    def join(run_dir, holder, need_s, **kw):
        assert holder.startswith("ccleg-") or holder.startswith("ccprobe-")
        assert need_s >= (90 * 60 + 1800 if holder.startswith("ccleg-") else 1800)
        ev("join", holder)
        handle.inf = json.loads((Path(run_dir) / "INFRA.json").read_text())
        return handle, FakeLease(holder, handle.dir / f"{holder}.json")
    return join


def pack_out(rd: Path, task: str, stream_bytes: bytes, output) -> Path:
    """<rd>/container_out.tgz as `evoctl pack` makes it (evo/work + evo/out + home/evo/.claude), extracted to
    <rd>/container with Box.collect's member rule: the snapshot a unit's grade job keeps and the evolver reads."""
    files = {"evo/work/spreadsheets/in.xlsx": b"PK-in-" + task.encode(), "evo/out/stream.jsonl": stream_bytes,
             "home/evo/.claude/projects/-evo-work/S1.jsonl": stream_bytes}
    if output is not None:
        files["evo/work/output/in.xlsx"] = output
    dirs = sorted({str(Path(n).parents[i]) for n in files for i in range(len(Path(n).parents) - 1)})
    tgz = rd / "container_out.tgz"
    with tarfile.open(tgz, "w:gz") as tf:
        for d in dirs:
            ti = tarfile.TarInfo(d)
            ti.type, ti.mode = tarfile.DIRTYPE, 0o755
            tf.addfile(ti)
        for n, data in sorted(files.items()):
            ti = tarfile.TarInfo(n)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    ex = rd / "container"
    if ex.exists():
        shutil.rmtree(ex)
    with tarfile.open(tgz, "r:gz") as tf:
        tf.extractall(ex, filter="data")
    return tgz


class FakeBox(D.LegBox):
    """cc_solver.Box's steps answered locally; Box.__init__ (the driver's PREP.json) and Box.result() are REAL.
    STREAMS[unit] = the stream (a callable gets the attempt number); OUTPUT[unit] = the workbook bytes (None = the
    solver saved nothing); FAIL_SETUP = units whose PRE probe fails. collect writes container_out.tgz + container/."""
    STREAMS: dict = {}
    OUTPUT: dict = {}
    FAIL_SETUP: set = set()
    ATTEMPTS: Counter = Counter()

    def __init__(self, m, root, unit, stamp, wall_s, limits):
        super().__init__(m, root, unit, stamp, wall_s, limits)
        self.polls = 0
        FakeBox.ATTEMPTS[unit] += 1
        self.attempt_n = FakeBox.ATTEMPTS[unit]

    def stream_text(self) -> str:
        s = self.STREAMS.get(self.unit)
        if callable(s):
            s = s(self.attempt_n)
        return s or stream()

    def prepare(self):
        ev("prepare", self.unit)
        self.rec["prepare"] = {"ok": True, "facts": {"cg_root_memory_max": "34359738368"}, "limits": self.spec["limits"]}
        self.status = "prepared"

    def up(self, init_root):
        assert init_root is False
        self.status = "up"
        self.m.cc_containers.append(self.cid)

    def probe(self):
        if self.unit in self.FAIL_SETUP:
            raise C.Refuse(f"{self.unit}: PRE probe: boom")
        self.rec["probe_pre_problems"] = []

    def upload(self):
        assert [r["rel"] for r in self.rows] == ["spreadsheets/in.xlsx"]
        self.after_extract()                      # the real skills link command against the fake handle
        self.rec["listing_problems"] = []
        self.rec["listing_after_upload"] = []

    def toolchain(self):
        self.rec["toolchain_problems"] = []
        self.status = "ready"

    def launch(self, tok):
        assert tok == TOKEN
        self.stream_fh = open(self.rd / "stream.jsonl", "wb")
        self.t_launch = time.time()
        ev("launch", self.unit)
        self.status = "running"

    def stop(self, why):
        self.stopped = why

    def collect(self, tok):
        self.stream_fh.close()
        data = self.OUTPUT.get(self.unit, b"PK-out-" + self.unit.encode())
        pack_out(self.rd, self.task, (self.rd / "stream.jsonl").read_bytes(), data)
        od = self.rd / "output"
        if od.exists():
            shutil.rmtree(od)
        if data is not None:
            od.mkdir()
            (od / "in.xlsx").write_bytes(data)
            self.rec["output"] = {"path": str(od / "in.xlsx"), "name": "in.xlsx", "expected_name": True,
                                  "sha256": sha(data), "bytes": len(data)}
        else:
            self.rec["output"] = None
        code, out = self.m.sh(f"for f in memory.peak memory.events cpu.stat pids.peak; do echo \"== $f\"; "
                              f"cat /sys/fs/cgroup/evoevolver/{self.cid}/$f 2>/dev/null; done", timeout=60, what="cgroup")
        self.rec["cgroup_end"] = out.strip()[-1500:]
        self.rec["container_times"] = {"t_start": str(int(self.t_launch)), "t_end": str(int(time.time()) + 60), "rc": "0"}
        self.rec["log_round_trip"] = {"streamed_sha256": "a", "container_sha256": "a", "equal": True}
        self.rec["token_scan"] = {"hits": [], "scanned_files": 4}
        self.rec["container_token_hits"] = []
        self.rec["probe_post_problems"] = []
        self.rec["down_before_pack"] = {"ok": True}
        ev("collect", self.unit)
        self.status = "collected"

    def wipe(self):
        ev("wipe", self.unit)
        self.rec["down_wipe"] = {"verified": True, "wiped": True}
        if self.cid in self.m.cc_containers:
            self.m.cc_containers.remove(self.cid)
        return True


def fake_poll(m, boxes):
    out = {}
    for b in boxes:
        b.polls += 1
        data = b.stream_text().encode()
        out[b.cid] = (len(data), "0", 0, "1", data) if b.polls >= 2 else (0, "RUNNING", 0, "1", b"")
    return out


class FakeGrader:
    """cc_pool_host.InPlaceGrader's surface with scripted scores. By default a job scores
    as the user's SSB grader would: 1.0 when the kept snapshot holds evo/work/output/*.xlsx, 0.0 when it holds none;
    SCORES[job_id] = (status, score) overrides it ("ok" with a number, "grader_failed" or "refused" with None);
    INSTALL_ERROR = a GradeError text install() raises."""
    SCORES: dict = {}
    DELAY: set = set()                            # job ids whose line lands only after `release_delayed`
    INSTALL_ERROR = None

    def __init__(self, handle, *, run_name, tag, knobs, plugin, task_rows, layout, overlay=None, max_procs=4,
                 max_heavy=1, heavy_bytes=None, job_timeout=5400, store=None, log=None):
        assert run_name == RUN and tag.startswith(RUN) and knobs["EVOSKILL_RECALC_TIMEOUT_SECONDS"] == "900"
        assert set(plugin["files"]) == {"benchmark_plugin.py", "benchmark_grader.py", "plugin_api.py"}
        assert set(task_rows) == set(TRAIN + VAL) and all(v.startswith("/data/") for v in layout.values())
        self.h, self.run_name, self.tag, self.task_rows, self.layout = handle, run_name, tag, task_rows, dict(layout)
        self.plugin = plugin
        self.max_procs, self.max_heavy, self.job_timeout = max_procs, max_heavy, job_timeout
        self.store = Path(store)
        self.store.mkdir(parents=True, exist_ok=True)
        self.setup = None
        self.jobs_sent: dict = {}
        self.seen: dict = {}
        self.regrades: dict = {}
        self.statuses: Counter = Counter()
        self.queue: list = []
        self.has_output: dict = {}
        self.launches, self.cleaned = 0, False
        self.released: set = set()

    def install(self, task_ids):
        ev("install", None)
        if self.INSTALL_ERROR:
            raise H.GradeError(self.INSTALL_ERROR)
        assert set(task_ids) <= set(self.task_rows)
        self.setup = {"runner": {"uploaded": True}, "links": {vm: {"target": mac, "action": "made"}
                                                              for mac, vm in self.layout.items()},
                      "task_files": {"n": 2 * len(task_ids)}}
        return self.setup

    def enqueue_unit(self, *, job_id, task_id, cdir, snapshot_sha256, mac_tgz=None, cid=None, unit=None,
                     collected_utc=None):
        assert Path(mac_tgz).is_file() and sha(Path(mac_tgz).read_bytes()) == snapshot_sha256
        assert cid in self.h.cc_containers, "the snapshot is kept and the job enqueued BEFORE the container is wiped"
        assert task_id in self.task_rows
        ev("enqueue", unit)
        with tarfile.open(mac_tgz, "r:gz") as tf:
            self.has_output[job_id] = any(n.startswith("evo/work/output/") and n.endswith(".xlsx") for n in tf.getnames())
        j = {"job_id": job_id, "task_id": task_id, "snapshot_tgz": f"/opt/cc/{self.run_name}/{self.tag}/units/{job_id}/out.tgz",
             "snapshot_sha256": snapshot_sha256, "snapshot_root": "evo/work",
             "meta": {"unit": unit, "tag": self.tag, "cid": cid, "collected_utc": collected_utc}}
        H.PGR.validate_job(j)
        self.jobs_sent[job_id] = j
        self.queue.append(job_id)
        self.launches += 1
        return {"kept": {"matches": True, "reuploaded": False, "sha256": snapshot_sha256}, "job": j,
                "worker": {"launched": True}}

    def poll(self):
        new = []
        for jid in list(self.queue):
            if jid in self.seen or (jid in self.DELAY and jid not in self.released):
                continue
            j = self.jobs_sent[jid]
            status, score = self.SCORES.get(jid) or ("ok", 1.0 if self.has_output[jid.split("#regrade", 1)[0]] else 0.0)
            line = {"job_id": jid, "task_id": j["task_id"], "worker": "plugin_grade/1", "status": status,
                    "score": score if status == "ok" else None,
                    "reason": None if status == "ok" else ("grade() raised ValueError: boom" if status == GF
                                                           else "snapshot sha256 differs from the job's"),
                    "snapshot_tgz": j["snapshot_tgz"], "snapshot_sha256": j["snapshot_sha256"],
                    "snapshot_skipped": [], "started_utc": L.now_utc(), "ended_utc": L.now_utc(), "seconds": 1.5,
                    "meta": j["meta"]}
            self.seen[jid] = line
            self.statuses[status] += 1
            new.append(line)
        return new

    def pending(self):
        return [j for j in self.jobs_sent if j not in self.seen]

    def ensure_worker(self):
        return {"launched": False, "running": False}

    def regrade(self, job_id):
        base = job_id.split("#regrade", 1)[0]
        if base in self.regrades:
            raise H.GradeError(f"{base}: already re-graded once")
        j = dict(self.jobs_sent[base], job_id=f"{base}#regrade1")
        self.regrades[base] = j["job_id"]
        self.jobs_sent[j["job_id"]] = j
        self.queue.append(j["job_id"])
        ev("regrade", base)
        return j

    def cleanup(self):
        self.cleaned = True
        return {"ok": True}

    def grader_record(self):
        return {"worker": "plugin_grade/1", "runner_sha256": "cd" * 32,
                "plugin": {"files": {n: f["sha256"] for n, f in self.plugin["files"].items()},
                           "tasks_json_sha256": self.plugin["tasks_json_sha256"]}, "interpreter": H.RUNNER_PY}

    def summary(self):
        return {"worker": "plugin_grade/1", "runner_sha256": "cd" * 32, "layout": dict(self.layout), "setup": self.setup,
                "results": str(self.store / "results.jsonl"), "jobs": len(self.jobs_sent),
                "statuses": dict(self.statuses), "in_place": True, "max_procs": self.max_procs,
                "max_heavy": self.max_heavy, "launches": self.launches, "regrades": dict(self.regrades),
                "pending": len(self.pending())}


class FakeClock:
    def __init__(self, t0=1_800_000_000.0):
        self.t = float(t0)
        self.slept = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += float(s)
        self.slept += float(s)


def make_run(w: World, a, *, handle=None, clock=None, **kw) -> tuple:
    handle = handle or FakeHandle(w.tmp)
    FakeBox.ATTEMPTS = Counter()
    run = D.LegRun(a, join_fn=make_join(handle), box_cls=FakeBox, grader_cls=FakeGrader,
                   poll_fn=fake_poll, clock=clock, sleep=clock.sleep if clock else None,
                   plugin=w.plugin, poll_s=0, grade_poll_s=0.02, **kw)
    return run, handle


def set_streams(**per_unit):
    """The four units of TRAIN x N (skill leg default: every init lists the promoted set)."""
    base = {f"{t}#s{k + 1}": stream(skills_listed=["alpha", "beta"], skill="alpha" if k == 0 else None)
            for t in TRAIN for k in range(N)}
    base.update(per_unit)
    FakeBox.STREAMS = base
    FakeBox.OUTPUT = {}
    FakeBox.FAIL_SETUP = set()
    FakeGrader.SCORES = {}
    FakeGrader.DELAY = set()
    FakeGrader.INSTALL_ERROR = None


def leg_json(w: World, key="train", epoch=1) -> dict:
    return json.loads((w.rd / f"{G.leg_tag(RUN, epoch, key)}.json").read_text())


def failed_json(w: World, key="train", epoch=1) -> dict:
    return json.loads((w.rd / f"{G.leg_tag(RUN, epoch, key)}{G.FAILED_SUFFIX}").read_text())


def record_of(run, task: str, k: int) -> dict:
    return json.loads(run.layout.record_paths(G.record_stem(run.questions[task], MODEL, k))[0].read_text())


# ══ validate / check-only / dry-run ══════════════════════════════════════════════════════════════
def test_validate_plan_check_only_and_dry_run(world, capsys):
    run = D.LegRun(world.args(), plugin=world.plugin)
    plan = run.validate()
    assert plan["tag"] == f"{RUN}_v1_train" and plan["n_units"] == 4 and plan["skills"] == ["alpha", "beta"]
    assert plan["bucket"] == G.bucket_for(MODEL, L.SOLVER_PREAMBLE, world.skills)
    assert plan["limits"]["mem_max_bytes"] == 1 << 30 and plan["limits"]["tmp_bytes"] <= (1 << 30) * 2 // 3
    assert plan["need_s"] == 90 * 60 + G.WAIT_MAX_S + 1800 and plan["leg_timeout_s"] == 43200
    assert plan["store"] == str(world.store_root / RUN / plan["tag"]) and plan["skills_probe"]["sandbox_id"] == "sb-probe"
    assert run.units == [f"{t}#s{k + 1}" for k in range(N) for t in TRAIN]        # sample-major
    assert run.expect_skills == ["alpha", "beta"] and run.kept == [] and run.requeue == run.units
    # The frozen plug-in's rows by id, and the grader's /data view (material_layout)
    assert run.plugin is world.plugin and run.task_rows == world.task_rows and run.questions == world.questions
    assert run.layout_map == world.layout_map() and run.layout.questions == world.questions
    # a base leg: no rows, no expectations, no probe needed
    rb = D.LegRun(world.args(skills=False), plugin=world.plugin)
    pb = rb.validate()
    assert pb["skills"] == [] and pb["skills_dir"] is None and pb["bucket"] != plan["bucket"]
    assert rb.expect_skills == [] and rb.probe_rec is None
    # check-only and dry-run through the command line (the launcher's exact argv; the frozen plug-in read from the run)
    argv = ["leg", "--tag", f"{RUN}_v1_train", "--key", "train", "--run-dir", str(world.rd),
            "--split", str(world.split), "--cache-dir", str(world.tmp / "cache" / f"{RUN}_e1_cache"),
            "--skills-dir", str(world.skills), "--n-samples", str(N), "--concurrency", "30"]
    assert D.main(argv + ["--check-only"]) == 0
    assert "check-only OK" in capsys.readouterr().out
    assert D.main(argv + ["--dry-run"]) == 7
    d = json.loads((world.rd / f"{RUN}_v1_train{G.DRYRUN_SUFFIX}").read_text())
    assert d["DRY_RUN"] is True and d["n_units"] == 4 and d["driver"] == "scripts/daytona/cc_leg_driver.py"
    assert d["credential"] == {"kind": "token_file", "from": str(world.token)} and d["pool_machine"] is None
    assert not (world.rd / f"{RUN}_v1_train.json").exists() and not EVENTS
    assert not (world.store_root / RUN).exists(), "check-only / dry-run write nothing into the store"


def test_refusals_before_anything_runs(world, monkeypatch):
    def refused(match, **over):
        with pytest.raises(D.Refuse, match=match):
            D.LegRun(world.args(**over), plugin=world.plugin).validate()
    refused("not <run>_v<N>_", tag=f"{RUN}_vX_train")
    # a base tag (<run>_base_<key>) is a leg this driver RUNS (epoch 0)
    assert D.parse_tag(f"{RUN}_base_train") == (RUN, 0, "train") and D.parse_tag(f"{RUN}_v2_val") == (RUN, 2, "val")
    refused("names run", tag="other_v1_train")
    refused("but --key", tag=f"{RUN}_v1_val", key="train")
    refused("split_path", split=str(world.tmp / "nope.json"))
    refused("n_samples", n_samples=3)
    refused("concurrency", concurrency=0)
    clamped = D.LegRun(world.args(concurrency=31), plugin=world.plugin).validate()
    assert clamped["concurrency"] == 30 and clamped["concurrency_requested"] == 31 and clamped["concurrency_ceiling"] == 30
    refused("cache dir name", cache_dir=str(world.tmp / "cache" / f"{RUN}_e2_cache"))
    refused("must be absolute", cache_dir="rel/cache")
    refused("promoted dir", skills_dir=str(world.rd))
    (world.rd / f"{RUN}_v1_train.json").write_text("{}")
    refused("is measured")
    (world.rd / f"{RUN}_v1_train.json").unlink()
    monkeypatch.setenv("EVOSKILL_SOLVER_PROVIDER_ORDER", "deepinfra")
    refused("OpenRouter pin")
    monkeypatch.delenv("EVOSKILL_SOLVER_PROVIDER_ORDER")
    monkeypatch.setenv("EVOSKILL_SOLVER_MODEL", "deepseek/x")
    refused("EVOSKILL_SOLVER_MODEL")
    monkeypatch.delenv("EVOSKILL_SOLVER_MODEL")
    monkeypatch.setenv(D.SUBSET_ENV, TRAIN[0])
    refused("accepted only for a run dir")
    monkeypatch.delenv(D.SUBSET_ENV)
    # the skill set: no SKILL.md policy of the loop's own (Claude Code is canonical), then the probe record
    (world.skills / "alpha" / "SKILL.md").write_text("---\nname: alpha\ndescription: d\nmodel: opus\n---\nbody\n")
    D.LegRun(world.args(), plugin=world.plugin).validate()
    (world.skills / "alpha" / "SKILL.md").write_text("---\nname: alpha\ndescription: d\n---\nbody\n")
    world.write_probe(ok=False)
    refused("primary skill placement did not pass")
    (world.rd / D.PROBE_RECORD).unlink()
    refused("no skills probe record")
    world.write_probe(ok=True)
    # the token file's mode, the RUNTIME token, HOLD_LEGS
    os.chmod(world.token, 0o644)
    refused("readable or writable by group/other")
    os.chmod(world.token, 0o600)
    (world.rd / "HOLD_LEGS").write_text("hold")
    refused("HOLD_LEGS")
    assert D.LegRun(world.args(check_only=True), plugin=world.plugin).validate()   # check-only may look
    (world.rd / "HOLD_LEGS").unlink()
    # a resume STATE.json of another leg in this store is refused
    st = G.state_new(tag="x_v1_train", key="train", epoch=1, run_dir=world.rd, bucket="0" * 12, skills_dir=None,
                     stamp="s", units=[])
    p = world.store_root / RUN / f"{RUN}_v1_train" / "STATE.json"
    G.state_write(p, st)
    refused("belongs to tag")
    p.unlink()
    # A frozen plug-in that does not verify (a pin drifted) refuses the leg -- it runs the
    # plug-in's grader; and a /data view with two sources on one path is refused before anything is linked
    def drifted(rd):
        raise ValueError("the run's frozen plug-in does not verify (benchmark_grader.py changed); re-freeze it")
    monkeypatch.setattr(G, "frozen_plugin", drifted)
    with pytest.raises(D.Refuse, match="frozen plug-in is not usable.*benchmark_grader.py changed"):
        D.LegRun(world.args()).validate()
    world.inf["data"]["root"] = str(world.tmp / "elsewhere" / "split.json")
    world.write_inf()
    with pytest.raises(D.Refuse, match="grader's /data view.*two sources map to one path"):
        D.LegRun(world.args(), plugin=world.plugin).validate()


def test_cli_returns_4_on_a_refusal_and_has_no_import_base(world, capsys):
    rc = D.main(["leg", "--tag", f"{RUN}_v1_train", "--key", "train", "--run-dir",
                 str(world.rd), "--split", str(world.split), "--cache-dir", str(world.tmp / "c" / f"{RUN}_e1_cache"),
                 "--n-samples", "3", "--concurrency", "1", "--check-only"])
    assert rc == 4 and "REFUSED" in capsys.readouterr().err
    # There is no import-base (a second grading path); base legs are RUN by this driver
    with pytest.raises(SystemExit) as e:
        D.main(["import-base", "--help"])
    assert e.value.code == 2 and "invalid choice" in capsys.readouterr().err
    src = Path(D.__file__).read_text()
    assert "import_base" not in src and "import-base" not in src


# ══ a whole leg with fakes: in-place grading, samples at grade time, the leg JSON ═════════════════
def run_skill_leg(world, *, clock=None, **kw):
    a = world.args()
    run, handle = make_run(world, a, clock=clock, **kw)
    run.validate()
    rc = run.run()
    return rc, run, handle


def test_leg_runs_grades_in_place_and_writes_the_loop_leg(world, capsys):
    set_streams(**{f"{TRAIN[0]}#s2": stream(capped=True, skills_listed=["alpha", "beta"])})
    FakeBox.OUTPUT = {f"{TRAIN[1]}#s1": None}                    # no workbook saved: graded all the same (0.0)
    jid_half = H.job_id_for(TRAIN[0], 1)
    FakeGrader.SCORES = {jid_half: ("ok", 0.5)}                  # a fraction: recorded, never a pass
    rc, run, handle = run_skill_leg(world)
    out = capsys.readouterr().out
    assert rc == 0, run.fatal + run.problems
    assert "[SKILL] alpha: Use alpha when the sheet needs it." in out and "[SKILL] beta:" in out
    leg = leg_json(world)
    assert leg["tag"] == f"{RUN}_v1_train" and leg["task_ids"] == TRAIN and leg["n_samples"] == N and leg["epoch"] == 1
    assert leg["per_task"] == {"0": [1.0, 0.5], "1": [0.0, 1.0]} and leg["sample_pass"] == 2 and leg["sample_total"] == 4
    assert leg["turn_exhausted"] == 1 and leg["pass3"] == 2 and leg["all3"] == 0 and leg["episodes_excluded_infra"] == 0
    assert leg["grader_failed"] == []
    assert leg["skills_dir"] == str(world.skills) and leg["model"] == MODEL and leg["budget"] == 45
    h = leg["solver_harness"]
    assert h["kind"] == "claude_code" and "Base directory for this skill: /evo/work/.claude/skills/<name>" in h["skills_delivery"]
    assert h["skills_delivery_facts"]["init_skills_seen"] == {"alpha": 4, "beta": 4} and h["skills_delivery_facts"]["sessions"] == 4
    assert h["skills_delivery_facts"]["probe"] == str(world.rd / D.PROBE_RECORD) and h["tools"] == ["Bash", "Edit", "Read", "Skill", "Write"]
    d = leg["daytona"]
    assert d["driver"] == "scripts/daytona/cc_leg_driver.py" and d["bucket"] == run.bucket and d["store"] == str(run.store)
    # EVERY completed unit was graded -- the one that saved nothing too -- by the run's grade()
    assert d["grade"]["statuses"] == {"ok": 4} and d["grade"]["jobs"] == 4 and d["grade"]["in_place"] is True
    assert d["grade"]["worker"] == "plugin_grade/1" and d["grade"]["layout"] == world.layout_map()
    assert d["grade"]["setup"]["links"]["/data/EvoSkill"] == {"target": str(D.REPO), "action": "made"}
    assert d["pool"]["result_guarantee"] == G.CC_RESULT_GUARANTEE and d["lease"]["acknowledged"] is True
    assert d["containment"]["units"] == 4 and d["containment"]["down_wipe_verified"] == 4 and d["containment"]["token_scan_hits"] == 0
    assert d["containment"]["skills_link"] == 4 and d["containment"]["memory_peak_max_bytes"] == 300000000
    assert d["spend"]["pauses"] == [] and d["spend"]["credential"] == {"kind": "token_file", "from": str(world.token)}
    assert d["machine"]["sandbox_id"] == handle.sid and d["plan"]["concurrency"] == 30 and d["skills"]["names"] == ["alpha", "beta"]
    env = leg["runtime_env"]
    assert env["solver_stack"] == "3.1.3 2.2.0 1.26.4" and env["sandbox_id"] == handle.sid
    assert env["shim_sha256"] == "ab" * 32 and env["skills"]["names"] == ["alpha", "beta"]
    assert env["scorer_sha256"] is None and env["grader"]["worker"] == "plugin_grade/1"
    assert env["grader"]["plugin"]["files"]["benchmark_grader.py"] == world.plugin["files"]["benchmark_grader.py"]["sha256"]
    # the leg's own re-read: STATE <-> both records <-> per_task
    assert run.layout.verify(leg) == []
    # records in both places, byte-identical; created_at = the grade line's ended_utc; the cc block carries the grade
    recs = sorted((run.cache_dir / run.bucket).glob("*.json"))
    assert len(recs) == 4 and (run.cache_dir / "CACHE_ENV.json").is_file()
    stem = G.record_stem(world.questions[TRAIN[0]], MODEL, 1)
    rec = json.loads((run.cache_dir / run.bucket / f"{stem}.json").read_text())
    assert rec["cache_key"]["tree_hash"] == run.bucket and rec["trace"]["budget_exhausted"].startswith("Turn budget exhausted")
    assert rec["trace"]["inline_score"] == 0.5 and rec["trace"]["output"] == {"finished": True}
    cc = rec["trace"]["cc"]
    assert cc["grade_job_id"] == jid_half and cc["grade_status"] == "ok" and cc["grade_score"] == 0.5 and cc["grade_reason"] is None
    tgz = run.store / "runs" / f"{TRAIN[0].replace('/', '_')}__s2" / "container_out.tgz"
    assert cc["snapshot_sha256"] == sha(tgz.read_bytes())
    m0 = rec["trace"]["messages"][0]           # no system row, no preamble: the task's own message is the first message
    assert m0["role"] == "user" and m0["content"] == L.task_user_message("Fill the model.", "in.xlsx")
    assert leg["solver_harness"]["prompt_delivery"] == L.PROMPT_DELIVERY
    assert (run.store / "s01" / "cache" / run.bucket / f"{stem}.json").read_bytes() == (run.cache_dir / run.bucket / f"{stem}.json").read_bytes()
    none = record_of(run, TRAIN[1], 0)["trace"]
    assert none["inline_score"] == 0.0 and none["output"] == {"finished": False} and none["cc"]["grade_status"] == "ok"
    # nothing SSB-shaped is written any more: no snapshot dirs / DIFF.txt, no runner.log
    assert not (run.tasks_dir / "_snapshots").exists() and not (run.store / "s01" / "runner.log").exists()
    # STATE.json: every unit graded with its score and the sha of the snapshot that was graded
    st = G.state_read(run.store / "STATE.json")
    assert {u: d["state"] for u, d in st["units"].items()} == {u: "graded" for u in run.units}
    assert {u: d["score"] for u, d in st["units"].items()} == {f"{TRAIN[0]}#s1": 1.0, f"{TRAIN[0]}#s2": 0.5,
                                                              f"{TRAIN[1]}#s1": 0.0, f"{TRAIN[1]}#s2": 1.0}
    for u, row in st["units"].items():
        assert "verdict" not in row and "output_sha256" not in row
        ud = L.unit_dir_name(u)
        assert row["snapshot_sha256"] == sha((run.store / "runs" / ud / "container_out.tgz").read_bytes())
        assert row["grade_job_id"] == H.job_id_for(*G.sample_index(u))
    assert st["schema"] == G.STATE_SCHEMA and st["machine"]["sandbox_id"] == handle.sid and st["ended_utc"]
    assert st["guard"]["admitting"] is True
    # the order: the grader installed BEFORE any box; every job (the no-output one too) enqueued BEFORE its wipe;
    # the lease released after the LAST wipe; the machine cleaned
    kinds = [k for k, _ in EVENTS]
    assert kinds.count("install") == 1 and kinds.index("join") < kinds.index("install") < kinds.index("prepare")
    for u in run.units:
        seq = [k for k, x in EVENTS if x == u]
        assert seq == ["prepare", "launch", "collect", "enqueue", "wipe"], (u, seq)
    assert kinds.index("release") > max(i for i, k in enumerate(kinds) if k == "wipe") and kinds.count("release") == 1
    assert run.grader.cleaned and handle.cc_containers == []
    assert run.queue.max_inflight == 4 and run.early == []
    # the PREP the boxes ran on is cc_solver's: ONE user message = the task's own text + its paths, nothing before it
    # (bare metal); nothing appended to Claude Code's system prompt (L.PROMPT_DELIVERY)
    prep = json.loads((run.store / "runs" / f"{TRAIN[0].replace('/', '_')}__s1" / "PREP.json").read_text())
    assert prep["upload_rows"][0]["rel"] == "spreadsheets/in.xlsx" and prep["sample_idx"] == 1
    um = L.task_user_message("Fill the model.", "in.xlsx")
    assert prep["user_message_sha256"] == sha(um.encode()) and prep["prompt_delivery"] == L.PROMPT_DELIVERY
    assert prep["instruction_sha256"] == sha(run.instruction.encode()) and "append_prompt_sha256" not in prep
    assert not (run.store / "runs" / f"{TRAIN[0].replace('/', '_')}__s1" / "append_system_prompt.txt").exists()


def test_base_leg_runs_without_skills_and_a_val_leg_shares_the_epoch_cache(world):
    set_streams(**{u: stream() for u in [f"{t}#s{k + 1}" for t in TRAIN for k in range(N)]})
    a = world.args(skills=False)
    run, handle = make_run(world, a)
    run.validate()
    assert run.run() == 0, run.fatal + run.problems
    leg = leg_json(world)
    assert leg["skills_dir"] is None and "skills_delivery" not in leg["solver_harness"] and leg["daytona"]["skills"] is None
    assert leg["daytona"]["containment"]["skills_link"] == 0 and not any("ln -s" in c for _, c in handle.sh_calls)
    assert leg["sample_pass"] == 4
    # the val leg (its own tag, the same epoch cache and bucket)
    FakeBox.STREAMS = {f"{VAL[0]}#s{k + 1}": stream() for k in range(N)}
    FakeBox.OUTPUT = {}
    EVENTS.clear()
    rv, hv = make_run(world, world.args("val", skills=False), handle=FakeHandle(world.tmp / "v"))
    rv.validate()
    assert rv.task_ids == VAL and rv.bucket == run.bucket and rv.cache_dir == run.cache_dir
    assert rv.run() == 0, rv.fatal + rv.problems
    lv = leg_json(world, "val")
    assert lv["key"] == "val" and lv["task_ids"] == VAL and lv["sample_total"] == 2
    assert len(list((run.cache_dir / run.bucket).glob("*.json"))) == 6


def test_a_task_subset_is_smoke_only(world, monkeypatch):
    monkeypatch.setenv(D.SUBSET_ENV, TRAIN[1])
    with pytest.raises(D.Refuse, match="accepted only"):
        D.LegRun(world.args(skills=False), plugin=world.plugin).validate()
    # a smoke run dir accepts them (the name says smoke)
    smoke = world.rd.parent / f"{RUN}_smoke"
    world.rd.rename(smoke)
    sp = json.loads((smoke / "SPEC.json").read_text())
    sp["spec"]["name"] = smoke.name
    (smoke / "SPEC.json").write_text(json.dumps(sp))
    rt = json.loads((smoke / "RUNTIME.json").read_text())
    (smoke / "RUNTIME.json").write_text(json.dumps(rt))
    a = world.args(skills=False, run_dir=str(smoke), tag=f"{smoke.name}_v1_train",
                   cache_dir=str(world.tmp / "cache" / f"{smoke.name}_e1_cache"), n_samples=1)
    run = D.LegRun(a, plugin=world.plugin)
    plan = run.validate()
    assert plan["n_tasks"] == 1 and plan["n_units"] == 1 and plan["smoke"] == {"run": True, "task_subset": TRAIN[1]}


# ══ the usage limit: a rejected session closes admission, the leg waits for the reported reset; a defect ═════════
def rejected(reset, **kw):
    """A session the API rejected for a usage limit; its last reading reports when the five-hour window resets."""
    return stream(status="rejected", five=1.0, reset=reset, skills_listed=["alpha", "beta"], **kw)


def test_a_rejected_session_waits_for_the_reported_reset_and_admits_again(world):
    clk = FakeClock()
    unit = f"{TRAIN[1]}#s1"
    set_streams(**{unit: lambda n: rejected(clk.t + 100) if n == 1 else stream(skills_listed=["alpha", "beta"])})
    a = world.args(concurrency=1)                                    # one at a time: the stop lands with units pending
    run, handle = make_run(world, a, clock=clk)
    run.validate()
    rc = run.run()
    assert rc == 0, run.fatal + run.problems
    leg = leg_json(world)
    p = leg["daytona"]["spend"]["pauses"]
    assert len(p) == 1 and p[0]["window"] == "five_hour" and p[0]["pending"] >= 1 and "quota rejected" in p[0]["reason"]
    assert p[0]["rejected_units"] == [unit] and p[0]["ended_utc"]
    assert clk.slept >= 100 + G.PAUSE_MARGIN_S                       # no new attempt before the reset + the margin
    g = leg["daytona"]["guard"]
    assert g["admitting"] is True and len(g["reopenings"]) == 1 and "usage limit's reset" in g["reopenings"][0]["note"]
    assert sorted(u for k, u in EVENTS if k == "launch") == sorted(run.units) and leg["sample_pass"] == 4
    st = G.state_read(run.store / "STATE.json")
    assert len(st["pauses"]) == 1
    assert handle.rec["pauses"][0]["until_utc"] == p[0]["until_utc"]


def test_a_session_rejected_again_after_the_wait_waits_again(world):
    clk = FakeClock()
    first, second = f"{TRAIN[0]}#s1", f"{TRAIN[1]}#s1"              # admitted one after the other (sample-major)
    set_streams(**{first: lambda n: rejected(clk.t + 50), second: lambda n: rejected(clk.t + 50)})
    run, handle = make_run(world, world.args(concurrency=1), clock=clk)
    run.validate()
    assert run.run() == 0, run.fatal + run.problems
    p = leg_json(world)["daytona"]["spend"]["pauses"]
    assert len(p) == 2 and p[1]["n"] == 2 and [x["rejected_units"] for x in p] == [[first], [second]]
    assert sorted(u for k, u in EVENTS if k == "launch") == sorted(run.units)   # a rejected session is still graded


def test_a_reset_beyond_the_leg_deadline_writes_failed_with_the_reason(world):
    clk = FakeClock()
    set_streams(**{f"{TRAIN[0]}#s1": rejected(clk.t + 4 * 3600)})
    world.rt["leg_timeout_s"] = 4 * 3600                             # the reset lands after this leg's deadline
    world.write_rt()
    run, handle = make_run(world, world.args(concurrency=1), clock=clk)
    run.validate()
    rc = run.run()
    assert rc == 5 and not (world.rd / f"{RUN}_v1_train.json").exists()
    f = failed_json(world)
    assert f["INCOMPLETE"] is True and f["problems"][0].startswith("usage limit: the limit resets at ")
    assert "beyond this leg's deadline" in f["problems"][0] and f["daytona"]["guard"]["kind"] == "quota"
    assert f["daytona"]["guard"]["admitting"] is False and "quota rejected" in f["daytona"]["guard"]["reason"]
    assert clk.slept == 0
    kinds = [k for k, _ in EVENTS]
    assert kinds.count("launch") == 1 and kinds.count("wipe") == 1 and kinds.index("release") > kinds.index("wipe")
    st = G.state_read(run.store / "STATE.json")
    assert st["units"][f"{TRAIN[0]}#s1"]["state"] == "graded" and any(v["state"] == "queued" for v in st["units"].values())


def test_a_reset_further_away_than_the_longest_wait_fails_the_leg(world):
    clk = FakeClock()
    set_streams(**{f"{TRAIN[0]}#s1": rejected(clk.t + G.WAIT_MAX_S + 3600)})        # e.g. a weekly limit
    run, handle = make_run(world, world.args(concurrency=1), clock=clk)
    run.validate()
    assert run.run() == 5 and f"at most {G.WAIT_MAX_S // 3600} h" in failed_json(world)["problems"][0]
    assert clk.slept == 0


def test_a_defect_stop_writes_failed_and_tears_everything_down(world):
    # a thinking block without text (a real defect; the skills Claude Code lists are its own business)
    set_streams(**{f"{TRAIN[0]}#s1": stream(thinking_text="")})
    run, handle = make_run(world, world.args(concurrency=2))
    run.validate()
    rc = run.run()
    assert rc == 5
    f = failed_json(world)
    assert any("thinking" in p for p in f["problems"]) or any("never ran" in p for p in f["problems"])
    assert f["daytona"]["guard"]["admitting"] is False and f["daytona"]["guard"]["kind"] == "defect"
    kinds = [k for k, _ in EVENTS]
    launched = [u for k, u in EVENTS if k == "launch"]
    assert 1 <= len(launched) <= 3 and all(("wipe", u) in EVENTS for u in launched)
    assert kinds.index("release") > max(i for i, k in enumerate(kinds) if k == "wipe")
    assert not (world.rd / f"{RUN}_v1_train.json").exists() and handle.cc_containers == []


def test_a_stop_leg_file_aborts_a_wait(world):
    clk = FakeClock()
    set_streams(**{f"{TRAIN[0]}#s1": rejected(clk.t + 100)})
    run, handle = make_run(world, world.args(concurrency=1), clock=clk)
    run.validate()
    (run.store / D.STOP_LEG_FILE).parent.mkdir(parents=True, exist_ok=True)
    (run.store / D.STOP_LEG_FILE).write_text("stop")
    assert run.run() == 5 and D.STOP_LEG_FILE in failed_json(world)["problems"][0]


# ══ grading: a grader failure is re-graded once, then recorded; a loop fault fails the leg ════════
def test_grader_failure_is_regraded_once_then_recorded_and_the_leg_continues(world):
    set_streams()
    jid = H.job_id_for(TRAIN[0], 0)
    FakeGrader.SCORES = {jid: (GF, None), jid + "#regrade1": ("ok", 0.25)}
    run, handle = make_run(world, world.args())
    run.validate()
    assert run.run() == 0, run.fatal + run.problems
    leg = leg_json(world)
    assert leg["daytona"]["grade"]["regrades"] == {jid: jid + "#regrade1"}
    assert leg["daytona"]["grade"]["statuses"] == {"ok": 4, GF: 1} and [u for k, u in EVENTS if k == "regrade"] == [jid]
    rec = record_of(run, TRAIN[0], 0)
    assert rec["trace"]["cc"]["grade_job_id"] == jid + "#regrade1" and rec["trace"]["inline_score"] == 0.25
    assert leg["per_task"]["0"][0] == 0.25 and leg["grader_failed"] == []          # the re-grade's score is kept
    st = G.state_read(run.store / "STATE.json")
    assert st["units"][f"{TRAIN[0]}#s1"]["grade_job_id"] == jid + "#regrade1" and st["units"][f"{TRAIN[0]}#s1"]["score"] == 0.25
    # twice: RECORDED as grader_failed (never a number) and the leg is WRITTEN
    (world.rd / f"{RUN}_v1_train.json").unlink()
    shutil.rmtree(run.store)
    shutil.rmtree(run.cache_dir)
    set_streams()
    FakeGrader.SCORES = {jid: (GF, None), jid + "#regrade1": (GF, None)}
    run2, h2 = make_run(world, world.args(), handle=FakeHandle(world.tmp / "h2"))
    run2.validate()
    assert run2.run() == 0, run2.fatal + run2.problems
    leg2 = leg_json(world)
    unit = f"{TRAIN[0]}#s1"
    assert leg2["per_task"] == {"0": [None, 1.0], "1": [1.0, 1.0]} and leg2["grader_failed"] == [unit]
    assert leg2["sample_pass"] == 3 and leg2["pass3"] == 2 and leg2["all3"] == 1 and leg2["sample_total"] == 4
    assert leg2["daytona"]["grade"]["statuses"] == {"ok": 3, GF: 2} and leg2["episodes_excluded_infra"] == 0
    st2 = G.state_read(run2.store / "STATE.json")
    assert st2["units"][unit]["state"] == "graded" and st2["units"][unit]["score"] == GF
    tr = record_of(run2, TRAIN[0], 0)["trace"]
    assert tr["inline_score"] is None and tr["cc"]["grade_score"] == GF and tr["cc"]["grade_status"] == GF
    assert "ValueError: boom" in tr["cc"]["grade_reason"]
    assert run2.layout.verify(leg2) == [] and not run2.fatal and run2.samples[unit]["score"] is None


def test_a_refused_line_fails_the_leg(world):
    """A `refused` line is a loop fault (the snapshot sha, an unknown task, the staged plug-in, the extraction), never a
    grade: no re-grade, the unit fails and the leg is FAILED."""
    set_streams()
    jid = H.job_id_for(TRAIN[1], 1)
    FakeGrader.SCORES = {jid: ("refused", None)}
    run, handle = make_run(world, world.args())
    run.validate()
    assert run.run() == 5
    f = failed_json(world)
    assert any("refused by the runner" in p and "snapshot sha256 differs" in p for p in f["problems"])
    assert f["daytona"]["grade"]["regrades"] == {} and not [u for k, u in EVENTS if k == "regrade"]
    assert G.state_read(run.store / "STATE.json")["units"][f"{TRAIN[1]}#s2"]["state"] == "failed"
    assert not (world.rd / f"{RUN}_v1_train.json").exists()


def test_an_install_error_fails_the_leg_before_any_box_launches(world):
    set_streams()
    FakeGrader.INSTALL_ERROR = "task file /x/golden.xlsx on the machine (missing) is not the Mac's bytes (ab12)"
    run, handle = make_run(world, world.args())
    run.validate()
    assert run.run() == 5
    f = failed_json(world)
    assert f["problems"][0].startswith("refused: the grader's setup on the machine: task file /x/golden.xlsx")
    kinds = [k for k, _ in EVENTS]
    assert kinds == ["join", "install", "release"], kinds                     # no box
    assert not run.samples and not run.jobs
    assert not (world.rd / f"{RUN}_v1_train.json").exists()


def test_a_grade_line_that_lands_before_its_result_is_taken_at_the_next_poll(world):
    """The job is enqueued BEFORE the wipe and the RESULT written after it; a fast grade (a rollout that saved nothing)
    can land in between. Its line waits for the next poll -- it is not 'a line without a RESULT' -- until the queue has
    ended (then every RESULT is written)."""
    run = D.LegRun(world.args(), plugin=world.plugin)
    run.validate()
    unit, jid = f"{TRAIN[0]}#s1", H.job_id_for(TRAIN[0], 0)
    lines = [[{"job_id": jid, "status": "ok", "score": 0.0}], [], [{"job_id": "x__s0", "status": "ok", "score": 1.0}]]
    run.grader = SimpleNamespace(poll=lambda: lines.pop(0), pending=lambda: [], regrades={})
    run.jobs = {jid: unit, "x__s0": "X/1#s1"}
    took = []
    run.accept_grade = lambda u, line, score: took.append((u, score))
    assert run.poll_grades() == 1 and took == [] and run.early and not run.fatal
    run.results[unit] = {"completed": True}
    assert run.poll_grades() == 0 and took == [(unit, 0.0)] and run.early == []
    run.queue_ended = True                                           # after the queue: no line waits
    assert run.poll_grades() == 1 and took[-1] == ("X/1#s1", 1.0)


def test_an_api_error_session_is_re_run_once_then_the_leg_fails(world):
    u = f"{TRAIN[1]}#s2"
    set_streams(**{u: lambda n: stream(api_error=(n == 1), skills_listed=["alpha", "beta"])})
    run, handle = make_run(world, world.args())
    run.validate()
    assert run.run() == 0, run.fatal + run.problems
    leg = leg_json(world)
    assert leg["daytona"]["retries"] == {u: 1} and [x for k, x in EVENTS if k == "launch"].count(u) == 2
    assert leg["sample_pass"] == 4 and leg["daytona"]["containment"]["units"] == 5
    rd = run.store / "runs" / f"{TRAIN[1].replace('/', '_')}__s2"
    assert list(rd.glob("RESULT.prev-*.json")) and json.loads((rd / "RESULT.json").read_text())["completed"]
    (world.rd / f"{RUN}_v1_train.json").unlink()
    shutil.rmtree(run.store)
    set_streams(**{u: stream(api_error=True, skills_listed=["alpha", "beta"])})
    run2, h2 = make_run(world, world.args(), handle=FakeHandle(world.tmp / "h2"))
    run2.validate()
    assert run2.run() == 5
    assert any("not completed after 2 attempt(s)" in p for p in failed_json(world)["problems"])


def test_a_setup_failure_is_re_queued_once(world):
    set_streams()
    u = f"{TRAIN[0]}#s1"
    FakeBox.FAIL_SETUP = {u}
    seen = []
    orig = FakeBox.probe

    def probe(self):
        seen.append(self.unit)
        if self.unit == u and seen.count(u) == 1:
            raise C.Refuse(f"{u}: PRE probe: boom")
        self.rec["probe_pre_problems"] = []
    FakeBox.probe = probe
    try:
        run, handle = make_run(world, world.args())
        run.validate()
        assert run.run() == 0, run.fatal + run.problems
        assert seen.count(u) == 2 and leg_json(world)["daytona"]["retries"] == {u: 1}
    finally:
        FakeBox.probe = orig
        FakeBox.FAIL_SETUP = set()


# ══ resume after a crash: graded samples kept, the rest re-run, leftover containers wiped ══════════
def test_resume_keeps_graded_and_grader_failed_samples_and_wipes_leftovers_of_its_tag(world):
    u = f"{TRAIN[1]}#s2"
    gf = f"{TRAIN[0]}#s2"
    gj = H.job_id_for(TRAIN[0], 1)
    set_streams()
    FakeGrader.SCORES = {gj: (GF, None), gj + "#regrade1": (GF, None)}
    run, handle = make_run(world, world.args())
    run.validate()
    assert run.run() == 0, run.fatal + run.problems              # four graded, one of them grader_failed
    st = G.state_read(run.store / "STATE.json")
    assert sum(1 for v in st["units"].values() if v["state"] == "graded") == 4 and st["attempt"] == 1
    assert st["units"][gf]["score"] == GF
    (world.rd / f"{RUN}_v1_train.json").unlink()
    # as after a crash with u mid-flight: its container of this tag left behind -- and one of another tag, left alone
    st["units"][u].update(state="running", cid="cc-financial-model-06-01-s2-x-ab12", score=None, graded_utc=None)
    G.state_write(run.store / "STATE.json", st)
    EVENTS.clear()
    set_streams()
    h2 = FakeHandle(world.tmp / "h2")
    h2.cc_containers = ["cc-financial-model-06-01-s2-x-ab12", "cc-other-tag-x-1"]
    run2, _ = make_run(world, world.args(), handle=h2)
    plan = run2.validate()
    assert plan["units_kept_from_earlier_attempt"] == 3 and plan["units_to_run"] == 1
    assert run2.run() == 0, run2.fatal + run2.problems
    assert [x for k, x in EVENTS if k == "launch"] == [u]
    assert ("handle_wipe", ("cc-financial-model-06-01-s2-x-ab12",)) in EVENTS and h2.cc_containers == ["cc-other-tag-x-1"]
    leg = leg_json(world)
    assert leg["sample_pass"] == 3 and leg["grader_failed"] == [gf] and leg["per_task"]["0"] == [1.0, None]
    assert leg["daytona"]["attempt"] == 2 and sorted(leg["daytona"]["kept_from_earlier_attempt"]) == \
        sorted(x for x in run.units if x != u)
    assert run2.samples[gf] == {"score": None, "turn_exhausted": False} and run2.layout.verify(leg) == []
    st2 = G.state_read(run2.store / "STATE.json")
    assert st2["attempt"] == 2 and st2["resumed_utc"] and st2["leftover_wipes"] and st2["units"][gf]["score"] == GF
    # a kept sample whose record is gone is re-run, not trusted
    (world.rd / f"{RUN}_v1_train.json").unlink()
    st2["units"][u]["state"] = "queued"
    G.state_write(run2.store / "STATE.json", st2)
    for p in run2.layout.record_paths(G.record_stem(world.questions[TRAIN[0]], MODEL, 0)):
        p.unlink()
    EVENTS.clear()
    run3, _ = make_run(world, world.args(), handle=FakeHandle(world.tmp / "h3"))
    run3.validate()
    assert run3.run() == 0, run3.fatal + run3.problems
    assert sorted(x for k, x in EVENTS if k == "launch") == sorted([u, f"{TRAIN[0]}#s1"])
    assert leg_json(world)["grader_failed"] == [gf]


# ══ the loop's own checks on the driver's legs (the stateful program's readers) ═══════════════════
def test_loop_acceptance_of_the_drivers_leg(world):
    """The base leg and the v1 skill leg, both RUN by the driver, read by the stateful program's readers: the leg
    pairing, the promoted set, no infra exclusion, the outcomes reconcile (a grader_failed rollout = never-graded), and
    evo_layout's scores.json = the STATE scores verbatim, grader_failed included."""
    from src.orchestrator import evo_layout as EL
    from src.orchestrator import invariants as INV
    from src.orchestrator import loop as LP
    # the base leg (epoch 0, no skills): one fraction, one grader_failed, one rollout that saved nothing
    set_streams(**{u: stream() for u in [f"{t}#s{k + 1}" for t in TRAIN for k in range(N)]})
    FakeBox.OUTPUT = {f"{TRAIN[1]}#s1": None}
    bj = H.job_id_for(TRAIN[0], 1)
    FakeGrader.SCORES = {bj: (GF, None), bj + "#regrade1": (GF, None), H.job_id_for(TRAIN[1], 1): ("ok", 0.5)}
    rb, _ = make_run(world, world.args(skills=False, epoch=0), handle=FakeHandle(world.tmp / "hb"))
    rb.validate()
    assert rb.tag == f"{RUN}_base_train" and rb.run() == 0, rb.fatal + rb.problems
    base = world.rd / f"{RUN}_base_train.json"
    assert json.loads(base.read_text())["grader_failed"] == [f"{TRAIN[0]}#s2"]
    # the v1 skill leg
    EVENTS.clear()
    set_streams(**{f"{TRAIN[0]}#s2": stream(capped=True, skills_listed=["alpha", "beta"])})
    FakeBox.OUTPUT = {f"{TRAIN[1]}#s1": None}
    j = H.job_id_for(TRAIN[0], 1)
    FakeGrader.SCORES = {j: (GF, None), j + "#regrade1": (GF, None)}
    run, handle = make_run(world, world.args())
    run.validate()
    assert run.run() == 0, run.fatal + run.problems
    spec = SimpleNamespace(n_samples=N, split_path=str(world.split), solver_model=MODEL, name=RUN, benchmark="ssbv2")
    ctx = SimpleNamespace(run_dir=world.rd, epoch=1, spec=spec, paths=LP.EpochPaths(world.rd, 1),
                          base_leg=str(base), base_leg_val="", evidence_layout="rollouts")
    leg = ctx.paths.leg("train")
    assert LP._leg_on_split(leg, world.split, "train", N)[0] == "ok" and LP._leg_on_split(base, world.split, "train", N)[0] == "ok"
    c = INV.check_legs_paired_comparably(base, leg, want_samples=N)
    assert c.status == "PASS", c.detail
    for fn in (LP._v_legs_comparable, LP._v_leg_loaded_promoted, LP._v_no_infra_exclusions, LP._v_outcomes_reconcile):
        ok, det = fn(ctx, "train")
        assert ok, (fn.__name__, det)
    ok, det = LP._v_outcomes_reconcile(ctx, "train")
    assert "pass 2, graded-fail 1, never-graded 1 = 4, capped-but-graded 1" in det
    # the evolver's material for epoch 1 (evo_e0) from the base leg's store: scores.json = the STATE scores verbatim
    out = ctx.paths.evo
    summ = EL.build(out, leg_json=base, epoch=0)
    rows = json.loads((out / EL.SCORES_JSON).read_text())
    st = G.state_read(rb.store / "STATE.json")
    assert rows == [{"task_id": EL.safe_id(t), "sample": k + 1, "score": st["units"][f"{t}#s{k + 1}"]["score"]}
                    for t in TRAIN for k in range(N)]
    assert [r["score"] for r in rows] == [1.0, GF, 0.0, 0.5]
    assert summ["rollouts"] == 4 and summ["scored"] == 3 and summ["grader_failed"] == 1 and summ["mean_score"] == 0.5
    assert EL.check(out, leg_json=base) == []
    ws = out / EL.WORKSPACES / f"{EL.safe_id(TRAIN[0])}_s1"
    assert (ws / "output" / "in.xlsx").read_bytes() == b"PK-out-" + f"{TRAIN[0]}#s1".encode()
    assert not (out / EL.WORKSPACES / f"{EL.safe_id(TRAIN[1])}_s1" / "output").exists()


# ══ the skills probe ══════════════════════════════════════════════════════════════════════════════
def probe_stream(*, two_turn: bool, base_dir="/evo/work/.claude/skills", skills=("probe-skill",), listing_ok=True):
    init = {"type": "system", "subtype": "init", "cwd": "/evo/work", "session_id": "S", "model": "claude-haiku-4-5",
            "tools": ["Bash", "Read", "Skill"] if listing_ok else ["Bash", "Read"], "claude_code_version": "2.1.281",
            "permissionMode": "bypassPermissions", "apiKeySource": "none", "mcp_servers": [], "skills": list(skills),
            "plugins": [], "memory_paths": {}}
    line = f"Base directory for this skill: {base_dir}/probe-skill"
    ev_ = [init]
    if two_turn:
        ev_ += [{"type": "assistant", "message": {"id": "m1", "model": "claude-haiku-4-5", "content": [
                    {"type": "tool_use", "id": "t1", "name": "Skill", "input": {"skill": "probe-skill"}}]}},
                {"type": "user", "message": {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "t1", "content": "Launching skill: probe-skill", "is_error": False}]}},
                {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": line + "\nWhen invoked..."}]}},
                {"type": "assistant", "message": {"id": "m2", "model": "claude-haiku-4-5", "content": [{"type": "text", "text": line}]}},
                {"type": "result", "subtype": "success", "is_error": False, "num_turns": 2, "result": line}]
    else:
        ev_ += [{"type": "assistant", "message": {"id": "m1", "model": "claude-haiku-4-5", "content": [{"type": "text", "text": "OK"}]}},
                {"type": "result", "subtype": "success", "is_error": False, "num_turns": 1, "result": "OK"}]
    return "\n".join(json.dumps(e) for e in ev_) + "\n"


def test_probe_verdict_reads_the_four_proofs():
    v = D.probe_verdict(probe_stream(two_turn=False), G.SKILLS_LINK, two_turn=False)
    assert v["ok"] and v["init_skills"] == ["probe-skill"] and v["tools_has_skill"]
    v2 = D.probe_verdict(probe_stream(two_turn=True), G.SKILLS_LINK, two_turn=True)
    assert v2["ok"] and v2["skill_loads"]["loaded"] == ["probe-skill"] and v2["launching_ok"] and v2["result_has_base_line"]
    bad = D.probe_verdict(probe_stream(two_turn=True, base_dir="/home/evo/.claude/skills"), G.SKILLS_LINK, two_turn=True)
    assert not bad["ok"] and any("no user text starting with" in p for p in bad["problems"])
    none = D.probe_verdict(probe_stream(two_turn=False, skills=()), G.SKILLS_LINK, two_turn=False)
    assert any("init.skills [] !=" in p for p in none["problems"])
    assert D.probe_argv("x", 3)[:6] == ["claude", "-p", "x", "--model", "haiku", "--max-turns"]
    inner = D.probe_inner("Reply", 1)
    assert "cd /evo/work" in inner and "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR=3" in inner and "--bare" not in inner


class FakeProbeBox(D.ProbeBox):
    STREAMS: dict = {}
    STEPS: list = []

    def prepare(self):
        self.STEPS.append("prepare")
        self.status = "prepared"

    def up(self, init_root):
        assert init_root is False
        self.status = "up"

    def probe(self):
        self.STEPS.append("probe")

    def upload(self):
        assert self.rows == [] and [r["rel"] for r in self.skill_rows] == ["probe-skill/SKILL.md"]
        assert self.expect_links == {"/evo/work/.claude/skills": "/evo/skills"} and self.expect_link_count == 1
        self.after_extract()
        self.STEPS.append("upload")

    def toolchain(self):
        self.status = "ready"

    def call(self, tok, label, prompt, max_turns, model=D.PROBE_MODEL):
        assert tok == TOKEN and model == "haiku"
        self.STEPS.append(f"call:{label}:{max_turns}")
        text = self.STREAMS[label]
        s = L.parse_stream(text)
        return {"label": label, "stream_text": text, "init": s["init"], "result": s["result"]}

    def write_fallback(self, text):
        self.STEPS.append("fallback")
        assert text.startswith("---\nname: probe-skill")
        return f"{D.FALLBACK_SKILLS_DIR}/probe-skill"

    def remove_primary_link(self):
        self.STEPS.append("unlink")

    def wipe(self):
        self.STEPS.append("wipe")
        return True


def test_probe_skills_writes_the_record_and_gates_the_skill_leg(world):
    (world.rd / D.PROBE_RECORD).unlink()
    FakeProbeBox.STEPS = []
    FakeProbeBox.STREAMS = {"primary_one": probe_stream(two_turn=False), "primary_two": probe_stream(two_turn=True),
                            "both_one": probe_stream(two_turn=False, skills=("probe-skill", "probe-skill")),
                            "fallback_one": probe_stream(two_turn=False),
                            "fallback_two": probe_stream(two_turn=True, base_dir=D.FALLBACK_SKILLS_DIR)}
    handle = FakeHandle(world.tmp / "p")
    a = SimpleNamespace(run_dir=str(world.rd), skills_dir="")
    rc = D.SkillsProbe(a, join_fn=make_join(handle), box_cls=FakeProbeBox).run()
    assert rc == 0
    rec = json.loads((world.rd / D.PROBE_RECORD).read_text())
    assert rec["primary"]["ok"] is True and rec["cc_version"] == "2.1.281" and rec["sandbox_id"] == handle.sid
    assert rec["fallback"]["ok"] is True and rec["both"]["init_skills"] == ["probe-skill", "probe-skill"]
    assert rec["lease_released"] is True and rec["wiped"] is True and rec["skill_names"] == ["probe-skill"]
    assert rec["streams"]["primary_two"][1] == {"type": "tool_use", "name": "Skill", "input": {"skill": "probe-skill"}}
    assert FakeProbeBox.STEPS == ["prepare", "probe", "upload", "call:primary_one:1", "call:primary_two:3", "fallback",
                                  "call:both_one:1", "unlink", "call:fallback_one:1", "call:fallback_two:3", "wipe"]
    assert rec["fallback"]["note"].startswith("measured ALONE")
    assert [k for k, _ in EVENTS] == ["join", "release"]
    assert D.LegRun(world.args(), plugin=world.plugin).validate()["skills_probe"]["path"] == str(world.rd / D.PROBE_RECORD)
    # the primary failing: rc 5, the record says so, and a skill leg refuses
    FakeProbeBox.STREAMS["primary_two"] = probe_stream(two_turn=True, base_dir="/elsewhere")
    rc = D.SkillsProbe(a, join_fn=make_join(FakeHandle(world.tmp / "p2")), box_cls=FakeProbeBox).run()
    assert rc == 5 and json.loads((world.rd / D.PROBE_RECORD).read_text())["primary"]["ok"] is False
    with pytest.raises(D.Refuse, match="did not pass"):
        D.LegRun(world.args(), plugin=world.plugin).validate()


# ══ status ════════════════════════════════════════════════════════════════════════════════════════
def test_status_prints_every_leg_state(world, capsys):
    set_streams()
    run, handle = make_run(world, world.args())
    run.validate()
    assert run.run() == 0
    assert D.main(["status", "--run-dir", str(world.rd)]) == 0
    out = capsys.readouterr().out
    assert f"{RUN}_v1_train: attempt 1" in out and "'graded': 4" in out and "WRITTEN" in out


# ══ a resumed attempt never inherits an earlier attempt's grade line ══════════════════════════════
def test_resume_grades_a_re_run_unit_under_an_attempt_unique_job_id(world, tmp_path):
    """The runner's RESUME skips a job_id that already has a line in results.jsonl (grade_worker.done_ids). A relaunch
    that re-ran a unit under the plain id would therefore take attempt 1's score (of attempt 1's snapshot) for attempt
    2's session. On a resume the job id carries "#a<attempt>", and the real InPlaceGrader.poll returns no line whose id
    this process never sent."""
    u = f"{TRAIN[1]}#s2"
    task, k0 = G.sample_index(u)
    plain = H.job_id_for(task, k0)
    set_streams(**{u: stream(api_error=True, skills_listed=["alpha", "beta"])})
    run, handle = make_run(world, world.args())
    run.validate()
    assert run.run() == 5                                        # three graded under plain ids, u failed twice
    assert plain not in run.jobs and all("#a" not in j for j in run.jobs)
    (world.rd / f"{RUN}_v1_train{G.FAILED_SUFFIX}").unlink()
    st = G.state_read(run.store / "STATE.json")
    # as after a crash with u collected and its job on the machine (its line may land there any time)
    st["units"][u].update(state="collected", cid="cc-financial-model-06-01-s2-x-ab12", grade_job_id=plain)
    G.state_write(run.store / "STATE.json", st)
    EVENTS.clear()
    set_streams()
    h2 = FakeHandle(world.tmp / "h2")
    h2.cc_containers = ["cc-financial-model-06-01-s2-x-ab12"]
    run2, _ = make_run(world, world.args(), handle=h2)
    run2.validate()
    assert run2.run() == 0, run2.fatal + run2.problems
    assert run2.jobs == {plain + "#a2": u}
    st2 = G.state_read(run2.store / "STATE.json")
    assert st2["attempt"] == 2 and st2["units"][u]["state"] == "graded" and st2["units"][u]["grade_job_id"] == plain + "#a2"
    rec = json.loads(run2.layout.record_paths(G.record_stem(world.questions[task], MODEL, k0))[0].read_text())
    assert rec["trace"]["cc"]["grade_job_id"] == plain + "#a2"
    assert run2.layout.verify(leg_json(world)) == []
    # the real poll filter: attempt 1's stale line under the plain id is not returned, attempt 2's is
    text = (json.dumps({"job_id": plain, "status": "ok", "score": 0.0}) + "\n"
            + json.dumps({"job_id": plain + "#a2", "status": "ok", "score": 1.0}) + "\n")
    fs = SimpleNamespace(download_file=lambda src, dst: Path(dst).write_text(text))
    g = H.InPlaceGrader(SimpleNamespace(sb=SimpleNamespace(fs=fs), dir=tmp_path), run_name=RUN, tag=f"{RUN}_v1_train",
                        knobs={}, plugin=world.plugin, task_rows=world.task_rows, layout=world.layout_map(),
                        store=tmp_path / "g")
    g.jobs_sent[plain + "#a2"] = {"job_id": plain + "#a2"}
    assert [ln["job_id"] for ln in g.poll()] == [plain + "#a2"] and plain not in g.seen


def test_a_legs_boxes_carry_the_specs_solver_model():
    """The leg's boxes launch and check the run's SPEC solver_model;
    cc_solver_lib.MODEL is only the default (a base leg without a model keeps it byte for byte)."""
    B = D.LegBox.for_leg([], model="claude-sonnet-5-5")
    assert B.model == "claude-sonnet-5-5" and D.LegBox.for_leg([]).model == L.MODEL == C.Box.model
    inner = L.launch_inner("U", model=B.model)
    assert "--model claude-sonnet-5-5" in inner and f"--model {L.MODEL} " not in inner
    assert L.launch_inner("U") == L.launch_inner("U", model=L.MODEL)
    ok, old = {"model": "claude-sonnet-5-5"}, {"model": "claude-sonnet-5"}
    assert not any("init model" in p for p in L.init_problems(ok, model=B.model, cwd=None))
    assert any("init model" in p for p in L.init_problems(old, model=B.model, cwd=None))   # a Sonnet 5 session fails
