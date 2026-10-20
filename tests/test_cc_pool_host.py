"""scripts/daytona/cc_pool_host.py -- the ONE pool machine of a Claude Code solver run.

NO DAYTONA, NO CLAUDE, NO KEY. The Daytona SDK is an in-memory fake whose sandbox is a temp dir: `process.exec`
recognises the real Shard steps by their `: EVOLEG=<step>` marker (identity / egress / unpack / cgroups / selftest /
grade / cpu weights / installs answered as the real machine would, the bundle really extracted and re-hashed, the
cpu.weight files really written; tests/test_cc_pool_host_tau3.py adds the tau3 steps) and the leg-side
commands (evoctl, the grader's install -- runner, plug-in dir, /data links, task-file shas --, the out.tgz kept by
cp, job enqueue, runner launch, cleanup) by their text; a fake plug-in grade runner checks its --expect against the
plug-in dir as the real one does and writes plugin_grade-shaped results lines. The
real ssb_leg_daytona_loop.Shard is exercised through cc_pool_host.CCMachine (a subclass); the
registry / lease code is the pool's own (ssb_leg_pool.Registry, pool_container_evolver). The ledger is redirected
into the test's temp dir.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "daytona"))
os.environ.setdefault("EVOSKILL_DAYTONA_POLL_S", "0")


# ══ fake daytona SDK module (only what Shard._create_once imports) ═══════════════════════════════
class _Params:
    def __init__(self, **kw):
        self.__dict__.update(kw)


fake_daytona = types.ModuleType("daytona")
fake_daytona.CreateSandboxFromImageParams = _Params
fake_daytona.Image = types.SimpleNamespace(from_dockerfile=lambda p: ("dockerfile", p))
fake_daytona.Resources = lambda **kw: kw
sys.modules.setdefault("daytona", fake_daytona)

import cc_pool_host as H  # noqa: E402
import daytona_lib_evo as dle_real  # noqa: E402
import daytona_lib_loop as dlx  # noqa: E402
import evolver_container_ctl as ctl  # noqa: E402
import plugin_grade_runner as PGR  # noqa: E402
import ssb_leg_daytona_loop as M  # noqa: E402
import ssb_leg_pool as P  # noqa: E402
from src.orchestrator import data_overlay  # noqa: E402
from src.orchestrator import plugin_runner as PR  # noqa: E402
from src.orchestrator import pool_container_evolver as pce  # noqa: E402

SPLIT = ROOT / M.PROJECT_SPLIT[0]
INFRA_SRC = ROOT / "tests" / "fixtures" / "ssb_leg_pool" / "infra_cc.json"
RUN_NAME = "ssbv2_cc_fx"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ══ the fake sandbox ════════════════════════════════════════════════════════════════════════════
class FakeProc:
    def __init__(self, sb):
        self.sb = sb

    def exec(self, command, timeout=None, env=None):
        parts = shlex.split(command)
        script = parts[2]
        if script.startswith(": EVOLEG="):
            step = script.split("\n", 1)[0].replace(": EVOLEG=", "").strip()
        else:
            step = self.sb.route(script)
        self.sb.calls.append((step, script))
        out, rc = self.sb.play(step, script, env or {})
        return types.SimpleNamespace(exit_code=rc, result=out)


class FakeFS:
    def __init__(self, sb):
        self.sb = sb

    def upload_file(self, src, dst, timeout=None):
        p = self.sb.path(dst)
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(src, (bytes, bytearray)):
            p.write_bytes(src)
        else:
            shutil.copyfile(src, p)
        self.sb.uploaded.append(str(dst))

    def download_file(self, src, dst):
        p = self.sb.path(src)
        if not p.is_file():
            raise FileNotFoundError(src)
        shutil.copyfile(p, dst)


ROUTES = (("readlink", "data links"), ("ls -A", "plugin state"), ("sha256sum -- ", "task files sha"),
          (f"sha256sum {H.RUNNER_VM}", "grader code sha"), ("mv -f ", "mv"), ("rm -rf -- ", "rm"),
          ("/state/out.tgz", "keep"),
          (f"{ctl.PY} {ctl.CTL} ", "evoctl"), ("sha256sum /opt/evoskill-evolver/bin/evoctl.py", "ctl sha"),
          ("mkdir -p /opt/evoskill-evolver/bin", "mkdir bin"), ('test "$(sha256sum', "cc verify"),
          (".part", "cc dir"), ("stat -c %s", "cc sha"), ("base64 -d >>", "enqueue"),
          ("pgrep -f", "worker state"), ("setsid nohup", "launch worker"), ("rm -rf", "cleanup"),
          ("df -k", "df"), ("sha256sum ", "rehash"), ("mkdir -p", "mkdir"))


class FakeSandbox:
    def __init__(self, client, name, sid):
        self.client, self.name, self.id = client, name, sid
        self.cpu, self.memory, self.disk = 8, 32, 40
        self.state = "started"
        self.build_info = types.SimpleNamespace(snapshot_ref=client.snapshot_ref)
        self.root = client.base / sid
        self.root.mkdir(parents=True)
        self.process, self.fs = FakeProc(self), FakeFS(self)
        self.calls, self.uploaded, self.downs = [], [], []
        self.containers: list = []
        self.claude_binaries = "0"
        self.disk_kib = 30 * (1 << 20)
        self.worker_running, self.hold_worker, self.launches = False, False, 0
        self.grade_status, self.grade_score = {}, {}
        self.worker_args: dict = {}
        self.fail_keep = False

    def refresh_data(self):
        pass

    def path(self, p):
        return self.root / str(p).lstrip("/")

    @staticmethod
    def route(script: str) -> str:
        for needle, step in ROUTES:
            if needle in script:
                return step
        raise AssertionError(f"unrouted command {script[:120]!r}")

    # -- steps
    def play(self, step, script, env):
        c = self.client
        assert "OPENROUTER_API_KEY" not in env and not any("TOKEN" in k for k in env), "no secret reaches the machine"
        if step == "identity":
            return ("whoami=root\narch=x86_64\ncpu_model=AMD EPYC 9354P\nnproc=64\nlibreoffice=" + c.libreoffice +
                    "\nrunner_openpyxl=3.1.5 lxml 6.1.1 LXML True DEFUSEDXML True\n"
                    "solver_stack=3.1.3 2.2.0 1.26.4\nrunner_python=Python 3.13.9\nsolver_python=Python 3.11.9\n"
                    f"claude_binaries={self.claude_binaries}\n"), 0
        if step == "egress":
            eg = {"openrouter": {"ok": True, "status": 200}, "anthropic": {"ok": True, "status": 200},
                  "example_com": {"ok": False, "error": "403"}, "pypi": {"ok": False, "error": "x"},
                  "github": {"ok": False, "error": "x"}, "google_ip": {"ok": False, "error": "x"}}
            if c.egress_answer:
                eg = c.egress_answer(self, eg)
            return "EGRESS " + json.dumps(eg) + "\n", 0
        if step == "unpack":
            with tarfile.open(self.path("/tmp/bundle.tgz")) as tf:
                self.bundle_names = tf.getnames()
                tf.extractall(self.root, filter="data")
            self.path("/tmp/bundle.tgz").unlink()
            return "/usr/local/bin/docker\nabc  /usr/local/bin/docker\n", 0
        if step == "verify":
            out = []
            for vm, local in (("/usr/local/bin/docker", "/opt/leg/docker_shim.py"),
                              (H.GRADE_WORKER_VM, H.GRADE_WORKER_VM)):
                p = self.path(local)
                if p.is_file():
                    out.append(f"{sha(p.read_bytes())}  {vm}")
            env_text = self.path("/opt/leg/leg.env").read_text() if self.path("/opt/leg/leg.env").is_file() else ""
            if env_text and not re.search(r"OPENROUTER|TOKEN|EVOSKILL_SOLVER", env_text):
                out.append("LEGENV_OK")
            return "\n".join(out) + "\n", 0
        if step == "cgroups":
            return "root=memory cpu pids evoshim=memory cpu pids\n", 0
        if step == "cpu weights":             # mkdir -p + `echo W > <cg>/cpu.weight` each, then `WEIGHT <g> <readback>`
            for w, g in re.findall(r"echo (\d+) > /sys/fs/cgroup/([\w-]+)/cpu\.weight", script):
                p = self.path(f"/sys/fs/cgroup/{g}/cpu.weight")
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(str(c.cpu_weight_kept.get(g, w)) + "\n")     # a kernel that keeps another value
            out = ""
            for g in re.findall(r"echo WEIGHT ([\w-]+) ", script):
                p = self.path(f"/sys/fs/cgroup/{g}/cpu.weight")
                out += f"WEIGHT {g} {p.read_text().strip() if p.is_file() else ''}\n"
            return out, 0
        if step == "selftest":
            img = shlex.split(script.split("bash -s ", 1)[1].split(" <<", 1)[0])[0]
            kv = {"DATASET": "HIDDEN", "UMOUNT": "DENIED", "NET": "NONE", "CGROUP_WRITE": "DENIED",
                  "WS_WRITE": "OK", "RUNNING": "true", "PROBE": "/usr/bin/soffice", "IMAGES": img,
                  "USERS_ENTRIES": "0", "OPT_ENTRIES": "0", "HOME_ENTRIES": "0", "SHIMSTATE_ENTRIES": "0",
                  "CGROUPS_LEFT": "0", "EXEC_RC": "0", "KEY_IN_ENV": "0"}
            return "".join(f"{k}={v}\n" for k, v in kv.items()), 0
        if step == "grade":
            task = shlex.split(script.split("bash -s ", 1)[1].split(" <<", 1)[0])[1]
            return "GRADE_SELFTEST " + json.dumps({"task": task, "golden_vs_golden_passed": True, "matched": 1,
                                                   "total": 1, "seconds": 1.0}) + "\n", 0
        if step == "df":
            return f"{self.disk_kib}\n", 0
        if step == "evoctl":
            args = shlex.split(script.split("evoctl.py", 1)[1].strip().splitlines()[0])
            if args[0] == "list":
                return json.dumps({"containers": list(self.containers), "cgroups": []}) + "\n", 0
            if args[0] == "down":
                cid = args[args.index("--cid") + 1]
                self.downs.append(cid)
                if cid in self.containers:
                    self.containers.remove(cid)
                return json.dumps({"ok": True, "verified": True, "wiped": "--wipe" in args}) + "\n", 0
            raise AssertionError(f"unexpected evoctl {args}")
        if step == "ctl sha":
            out = ""
            for vm in (ctl.CTL, ctl.PROXY_FILE):
                p = self.path(vm)
                if p.is_file():
                    out += f"{sha(p.read_bytes())}  {vm}\n"
            return out, 0
        if step in ("mkdir bin", "cc dir", "mkdir"):
            for m in re.finditer(r"mkdir -p (\S+)", script):
                self.path(shlex.split(m.group(1))[0]).mkdir(parents=True, exist_ok=True)
            return "", 0
        if step == "cc sha":
            p = self.path(ctl.CC_HOST)
            return (f"{sha(p.read_bytes())}\n{p.stat().st_size}\n" if p.is_file() else ""), 0
        if step == "cc verify":
            m = re.search(r'= ([0-9a-f]{64}) && test "\$\(stat -c %s [^)]*\)" = (\d+)', script)
            part = self.path(ctl.CC_HOST + ".part")
            if m and part.is_file() and sha(part.read_bytes()) == m.group(1) and part.stat().st_size == int(m.group(2)):
                part.rename(self.path(ctl.CC_HOST))
                return "CC_OK\n", 0
            return "mismatch\n", 1
        if step == "keep":                    # mkdir -p <unit> && cp <cdir>/state/out.tgz <unit>/out.tgz && sha256sum
            src, dst = (shlex.split(x)[0] for x in re.search(r"cp (\S+) (\S+) &&", script).groups())
            if self.fail_keep or not self.path(src).is_file():
                return "cp: cannot stat: No such file or directory\n", 1
            self.path(dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.path(src), self.path(dst))
            return sha(self.path(dst).read_bytes()) + "\n", 0
        if step == "grader code sha":
            return "".join(f"{sha(self.path(vm).read_bytes())}  {vm}\n" for vm in (H.RUNNER_VM, H.GRADE_WORKER_VM)
                           if self.path(vm).is_file()), 0
        if step == "mv":
            src, dst = (shlex.split(x)[0] for x in re.search(r"mv -f (\S+) (\S+)", script).groups())
            os.replace(self.path(src), self.path(dst))
            return "", 0
        if step == "plugin state":           # mkdir -p D; echo "FILES=$(ls -A D | tr ...)"; sha256sum D/a D/b
            d = self.path(shlex.split(re.match(r"mkdir -p (\S+);", script).group(1))[0])
            d.mkdir(parents=True, exist_ok=True)
            out = "FILES=" + "".join(n + "/" for n in sorted(os.listdir(d))) + "\n"
            for p in shlex.split(script.split("; sha256sum ", 1)[1].rsplit(" 2>/dev/null", 1)[0]):
                if self.path(p).is_file():
                    out += f"{sha(self.path(p).read_bytes())}  {p}\n"
            return out, 0
        if step == "rm":
            for p in shlex.split(script.split("rm -rf -- ", 1)[1]):
                shutil.rmtree(self.path(p)) if self.path(p).is_dir() else self.path(p).unlink(missing_ok=True)
            return "", 0
        if step == "data links":             # emulates cc_pool_host.LINK_FN for each `link <mac> <vm>`
            for d in shlex.split(re.search(r"mkdir -p (.*?) \|\| exit 1", script).group(1)):
                self.path(d).mkdir(parents=True, exist_ok=True)
            out = ""
            for call in script.split(H.LINK_FN, 1)[1].split("; link ")[1:]:
                mac, vm = shlex.split(call.strip())
                p = self.path(vm)
                if p.is_symlink() and os.readlink(p) == mac:
                    out += f"KEPT {vm}\n"
                elif p.exists() and not p.is_symlink():
                    out += f"NOT_A_LINK {vm}\n"
                else:
                    if p.is_symlink():
                        p.unlink()
                    os.symlink(mac, p)
                    out += f"MADE {vm}\n"
                out += f"READLINK {vm} -> {os.readlink(p) if p.is_symlink() else ''}\n"
            return out, 0
        if step == "task files sha":
            out = ""
            for p in shlex.split(script[len("sha256sum -- "):].rsplit(" 2>/dev/null", 1)[0]):
                if self.path(p).is_file():
                    out += f"{sha(self.path(p).read_bytes())}  {p}\n"
            return out, 0
        if step == "rehash":
            p = shlex.split(re.search(r"sha256sum ([^\s;|]+)", script).group(1))[0]
            return (sha(self.path(p).read_bytes()) if self.path(p).is_file() else "") + "\n", 0
        if step == "enqueue":
            m = re.search(r"echo (\S+) \| base64 -d >> ([^\s;]+)", script)
            jobs = self.path(shlex.split(m.group(2))[0])
            jobs.parent.mkdir(parents=True, exist_ok=True)
            with open(jobs, "ab") as fh:
                fh.write(base64.b64decode(m.group(1)))
            return f"{len(jobs.read_text().splitlines())}\n", 0
        if step == "worker state":
            rcf = self.path(shlex.split(re.search(r"cat (\S+) 2>/dev/null", script).group(1))[0])
            res = self.path(shlex.split(re.search(r"test -f (\S+)", script).group(1))[0])
            rc = rcf.read_text().strip() if rcf.is_file() else ""
            n = len(res.read_text().splitlines()) if res.is_file() else 0
            return f"WORKER={'RUNNING' if self.worker_running else 'IDLE'}\nRC={rc}\nRESULTS={n}\n", 0
        if step == "launch worker":
            assert "env -u OPENROUTER_API_KEY" in script and ". /opt/leg/leg.env" in script
            inner = script.split("bash -c '", 1)[1].split("' </dev/null", 1)[0]
            argv = shlex.split(inner.split(f"{H.RUNNER_VM} run ", 1)[1].split(" >> ", 1)[0])
            args = {"expect": {}}
            for k, v in zip(argv[::2], argv[1::2]):
                if k == "--expect":
                    args["expect"][v.split("=", 1)[0]] = v.split("=", 1)[1]
                else:
                    args[k[2:]] = v
            args["rc"] = inner.rsplit("echo $? > ", 1)[1].strip()
            self.worker_args = args
            self.launches += 1
            self.path(args["rc"]).unlink(missing_ok=True)
            d = self.path(args["plugin-dir"])     # the runner's own refusal: rc 3, no line (plugin_grade_runner)
            if not d.is_dir() or sorted(os.listdir(d)) != sorted(args["expect"]) or any(
                    sha((d / n).read_bytes()) != s for n, s in args["expect"].items()):
                self.path(args["rc"]).write_text("3\n")
                return "", 0
            if self.hold_worker:
                self.worker_running = True
            else:
                self.run_worker()
            return "", 0
        if step == "cleanup":
            for m in re.finditer(r"rm -rf ([^\s;]+) ([^\s;]+)", script):
                for tok in m.groups():
                    shutil.rmtree(self.path(shlex.split(tok)[0]), ignore_errors=True)
            return "CLEAN_RC=0\n", 0
        raise AssertionError(f"unexpected step {step!r}: {script[:100]!r}")

    def run_worker(self):
        """The fake plug-in grade runner: one results line per job without one (RESUME by job_id), then rc 0. The kept
        tgz must be the job's bytes (else 'refused', the loop's fault); grade() 'returns' sb.grade_score[jid] (1.0 by
        default) or, with sb.grade_status[jid] = 'grader_failed', 'raises'."""
        a = self.worker_args
        jobs_f, res_f = self.path(a["jobs"]), self.path(a["results"])
        jobs = [json.loads(ln) for ln in jobs_f.read_text().splitlines() if ln.strip()]
        done = set()
        if res_f.is_file():
            done = {json.loads(ln)["job_id"] for ln in res_f.read_text().splitlines() if ln.strip()}
        with open(res_f, "a") as fh:
            for j in jobs:
                jid = j["job_id"]
                if jid in done:
                    continue
                tgz = self.path(j["snapshot_tgz"])
                status, score, reason = self.grade_status.get(jid, "ok"), None, None
                if not tgz.is_file() or sha(tgz.read_bytes()) != j["snapshot_sha256"]:
                    status, reason = "refused", "snapshot sha256 differs from the job's"
                elif status == "ok":
                    score = float(self.grade_score.get(jid, 1.0))
                else:
                    reason = "grade() raised ValueError: toy says no"
                fh.write(json.dumps({"job_id": jid, "task_id": j["task_id"], "worker": PGR.WORKER_VERSION,
                                     "status": status, "score": score, "reason": reason, "log_records": [],
                                     "snapshot_tgz": j["snapshot_tgz"], "snapshot_sha256": j["snapshot_sha256"],
                                     "snapshot_files": 1, "snapshot_skipped": [], "plugin": a["expect"],
                                     "started_utc": dlx.now(), "ended_utc": dlx.now(), "meta": j["meta"]}) + "\n")
        self.path(a["rc"]).write_text("0\n")
        (res_f.parent / "worker_meta.json").write_text(json.dumps({"worker": PGR.WORKER_VERSION}))
        (res_f.parent / "worker.log").write_text("done\n")
        self.worker_running = False

    def finish_worker(self):
        self.run_worker()


class FakeClient:
    sandbox_cls = FakeSandbox                # tests/test_cc_pool_host_tau3.py: a sandbox that answers the tau3 steps

    def __init__(self, base: Path, tag: str = "t"):
        self.base, self.tag = base, tag
        self.live, self.deleted, self.created = {}, [], []
        self.snapshot_ref = M.EXPECTED["snapshot_ref"]
        self.libreoffice = M.EXPECTED["libreoffice"]
        self.egress_answer = None
        self.cpu_weight_kept = {}            # {cgroup: the cpu.weight a write leaves} (a readback that differs)
        self.delete_refuses = set()          # ids whose delete silently does nothing (a listing that lags forever)
        self.list_fails = False
        self._n = 0
        self.lock = threading.Lock()

    def create(self, params, timeout=None, on_snapshot_create_logs=None):
        import datetime as _dt
        with self.lock:
            self._n += 1
            sid = f"{self._n:08d}-fake-{params.name[-10:]}"
        sb = self.sandbox_cls(self, params.name, sid)
        sb.domain_allow_list = getattr(params, "domain_allow_list", None)
        now = _dt.datetime.now(_dt.timezone.utc)
        sb.created_at = now.isoformat().replace("+00:00", "Z")
        sb.auto_destroy_at = (now + _dt.timedelta(minutes=float(params.ttl_minutes))).isoformat().replace("+00:00", "Z")
        self.live[sid] = sb
        self.created.append(params)
        return sb

    def adopt(self, name: str, sid: str) -> FakeSandbox:
        """A sandbox that exists before any host (an orphan of a crashed host)."""
        sb = self.sandbox_cls(self, name, sid)
        self.live[sid] = sb
        return sb

    def get(self, key):
        for sb in self.live.values():
            if key in (sb.id, sb.name):
                return sb
        raise RuntimeError("Sandbox not found (404)")

    def delete(self, sb, timeout=None, wait=True):
        if sb.id in self.delete_refuses:
            return
        self.live.pop(sb.id, None)
        self.deleted.append(sb.id)

    def list(self):
        if self.list_fails:
            raise RuntimeError("listing failed")
        return list(self.live.values())


def fake_dle(client):
    """daytona_lib_evo as the handle sees it: run + readback (fresh) + a ledger list."""
    rows = []
    return types.SimpleNamespace(run=dle_real.run, readback=dle_real.readback, now=dlx.now,
                                 ledger=lambda **kw: rows.append(kw), rows=rows)


# ══ fixtures ════════════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(dlx, "LEDGER", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setattr(dle_real, "LEDGER", str(tmp_path / "ledger_evo.jsonl"))
    monkeypatch.delenv(data_overlay.ENV, raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("EVOSKILL_KEEP_AWAKE", "0")
    monkeypatch.setenv("EVOSKILL_CC_HANDLE_RETRY_S", "0")
    monkeypatch.delenv(H.HOST_SCRIPT_ENV, raising=False)
    rd = tmp_path / "outputs" / RUN_NAME
    rd.mkdir(parents=True)
    spec = {"spec": {"name": RUN_NAME, "split_path": str(SPLIT), "split_sha256": M.sha256_file(SPLIT), "n_samples": 3,
                     "solver_model": "claude-sonnet-5", "solver_harness": "claude_code", "agent_turns": 45}}
    (rd / "SPEC.json").write_text(json.dumps(spec))
    inf = json.loads(INFRA_SRC.read_text())
    inf["data"]["overlay_patches_json"] = None
    inf["data"]["overlay_patches_sha256"] = None
    (rd / "INFRA.json").write_text(json.dumps(inf))
    rt = {"prompt_mode": "template",
          "daytona_leg": {"store_root": str(tmp_path / "store"), "tasks_root": str(tmp_path / "tasks_root"),
                          "pool": True, "pool_keys": ["train", "val"], "pool_sandboxes": 1, "pool_ttl_minutes": 1440,
                          "pool_ttl_margin_min": 255, "pool_max_heavy_units": 1,
                          "require_snapshot_ref": M.EXPECTED["snapshot_ref"],
                          "require_libreoffice": M.EXPECTED["libreoffice"]},
          "evolver_daytona": {"placement": "pool_container", "machine_wait_s": 900},
          "cc_leg": {"concurrency": 30, "machine_wait_s": 1800, "_note": "fixture"}}
    (rd / "RUNTIME.json").write_text(json.dumps(rt))
    ssb = tmp_path / "ssb_data"                     # the SpreadsheetBench v2 tree: one dataset.json per category
    for cat in M.CATEGORIES:
        (ssb / cat).mkdir(parents=True)
        (ssb / cat / "dataset.json").write_text("[]")
    monkeypatch.setattr(M, "DATA", ssb)
    data = tmp_path / "data"
    data.mkdir()
    split = json.loads(SPLIT.read_text())
    idx = {}
    for t in sorted(set(split["train"]) | set(split["val"])):
        f = data / (t.replace("/", "_") + ".xlsx")
        f.write_bytes(b"x" + t.encode())
        idx[t] = {"input": f, "golden": f, "answer_position": "A1"}
    cc_local = tmp_path / "claude-fake"
    cc_local.write_bytes(b"#!fake claude " + b"z" * 100)
    client = FakeClient(tmp_path / "sandboxes", tag=f"{RUN_NAME}_v1_train")
    return types.SimpleNamespace(tmp=tmp_path, rd=rd, idx=idx, client=client, cc_local=cc_local, split=split,
                                 store=tmp_path / "store")


def make_owner(w, pool_dir=None, tick=0.01, client=None):
    o = H.PoolMachineOwner(w.rd, pool_dir=pool_dir, client=client or w.client, idx=w.idx, tick_s=tick,
                           cc_local=w.cc_local, keep_awake=False)
    o.cc_sha256, o.cc_bytes = sha(w.cc_local.read_bytes()), w.cc_local.stat().st_size
    return o


def live_host(pdir, secs=60.0, script=None):
    """A live process whose argv IS the registered host's (ssb_leg_pool.host_alive's identity rule)."""
    script = str(script or H._host_script())
    p = subprocess.Popen([sys.executable, "-c", "import sys, time; time.sleep(float(sys.argv[1]))", str(secs),
                          script, "host", "--run-dir", "x", "--pool-dir", str(pdir)])
    threading.Thread(target=p.wait, daemon=True).start()
    # until the child exec'd, `ps` shows the parent's argv: wait for the identity to be visible
    wait_for(lambda: P.host_alive({"host_pid": p.pid, "host_script": script, "pool_dir": str(pdir)}),
             "the stand-in host's argv", 10)
    return p


def cc_kw(w) -> dict:
    """join()'s CC-binary pins for the fixture's small fake binary."""
    return {"cc_local": w.cc_local, "cc_sha256": sha(w.cc_local.read_bytes()), "cc_bytes": w.cc_local.stat().st_size}


def wait_for(cond, what, timeout=30.0):
    t0 = time.time()
    while not cond():
        if time.time() - t0 > timeout:
            raise AssertionError(f"timed out waiting for {what}")
        time.sleep(0.01)


def machines_json(reg: P.Registry) -> dict:
    return json.loads(reg.machines_file.read_text())


def ledger(w):
    p = Path(dlx.LEDGER)
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


# ══ settings / cfg ══════════════════════════════════════════════════════════════════════════════
def test_cc_leg_settings_defaults_and_refusals(world):
    s = H.cc_leg_settings(world.rd)
    assert s["concurrency"] == 30 and s["machine_wait_s"] == 1800 and "quota_stop" not in s
    assert s["grade_max_procs"] == 4 and s["grade_max_heavy"] == 1 and s["cc_version"] == ctl.CC_VERSION
    assert set(s) == set(H.CC_LEG_DEFAULTS), "the block is exactly the known keys (no _note)"
    rt = json.loads((world.rd / "RUNTIME.json").read_text())
    for bad, word in (({"typo": 1}, "unknown key"), ({"concurrency": 31}, "at most 30 rollouts"),
                      ({"quota_stop": 0.8}, "no usage threshold exists"), ({"max_pause_s": 600}, "max_pause_s"),
                      ({"machine_wait_s": "soon"}, "machine_wait_s")):
        rt2 = dict(rt, cc_leg=dict(rt["cc_leg"], **bad))
        (world.rd / "RUNTIME.json").write_text(json.dumps(rt2))
        with pytest.raises(M.Refused, match=word):
            H.cc_leg_settings(world.rd)
    (world.rd / "RUNTIME.json").write_text(json.dumps({k: v for k, v in rt.items() if k != "cc_leg"}))
    assert H.cc_leg_settings(world.rd) == H.CC_LEG_DEFAULTS


def test_machine_cfg_reads_the_run_and_refuses_drift(world, monkeypatch):
    cfg = H.machine_cfg(world.rd)
    assert cfg.tag == f"pool_{RUN_NAME}" and cfg.code == P.pool_code(RUN_NAME) and cfg.key == "pool"
    assert cfg.rt["ttl_minutes"] == 1440 and cfg.rt["require_snapshot_ref"] == M.EXPECTED["snapshot_ref"]
    assert cfg.image_id == M.EXPECTED["image_id"] and cfg.skills_dir is None and cfg.data_overlay is None
    assert cfg.tasks[:len(cfg.train)] == world.split["train"] and set(cfg.tasks) == set(world.split["train"]) | set(world.split["val"])
    assert cfg.n_samples == 3 and cfg.split_sha256 == M.sha256_file(SPLIT) and cfg.prompt_mode == "template"
    assert cfg.knobs["EVOSKILL_RECALC_TIMEOUT_SECONDS"] == "900"       # INFRA's timeout joined the knobs
    assert H.pool_root(world.rd) == pce.pool_run_dir(world.rd) == world.store / RUN_NAME / "pool" / "run"
    # the shipped leg snapshot is not pinned (INFRA null, EXPECTED None): a RUNTIME pin alone is then the pin
    assert M.EXPECTED["snapshot_ref"] is None and cfg.rt["require_snapshot_ref"] is None
    rt = json.loads((world.rd / "RUNTIME.json").read_text())
    rt["daytona_leg"]["require_snapshot_ref"] = "daytona-other:daytona"
    (world.rd / "RUNTIME.json").write_text(json.dumps(rt))
    assert H.machine_cfg(world.rd).rt["require_snapshot_ref"] == "daytona-other:daytona"
    # drift refusals: a pinned INFRA snapshot and a RUNTIME pin that disagrees
    inf = json.loads((world.rd / "INFRA.json").read_text())
    inf["leg_sandbox"]["snapshot_ref"] = "daytona-pinned:daytona"
    (world.rd / "INFRA.json").write_text(json.dumps(inf))
    with pytest.raises(M.Refused, match="require_snapshot_ref"):
        H.machine_cfg(world.rd)
    inf["leg_sandbox"]["snapshot_ref"] = None
    (world.rd / "INFRA.json").write_text(json.dumps(inf))
    rt["daytona_leg"]["require_snapshot_ref"] = M.EXPECTED["snapshot_ref"]
    (world.rd / "RUNTIME.json").write_text(json.dumps(rt))
    sp = json.loads((world.rd / "SPEC.json").read_text())
    sp["spec"]["split_sha256"] = "0" * 64
    (world.rd / "SPEC.json").write_text(json.dumps(sp))
    with pytest.raises(M.Refused, match="split_sha256"):
        H.machine_cfg(world.rd)
    (world.rd / "INFRA.json").unlink()
    with pytest.raises(M.Refused, match="INFRA.json"):
        H.machine_cfg(world.rd)


# ══ the registry format is the pool's own ═══════════════════════════════════════════════════════
def test_machine_json_schema_and_row_keys_equal_the_pools(world):
    o = make_owner(world, pool_dir=world.store / RUN_NAME / "pool" / "run" / "p1")
    o.reap()
    m = o.create_machine("m01")
    rec = machines_json(o.reg)
    assert rec["schema"] == pce.POOL_MACHINES_SCHEMA == P.MACHINES_SCHEMA
    assert rec["host_state"] == "open" and rec["current"] == "m01" and rec["owner"] == H.OWNER
    assert rec["evolver_placement"] == pce.PLACEMENT and set(rec["domain_allow_list"]) == set(pce.MACHINE_ALLOW)
    assert rec["host_pid"] == os.getpid() and rec["leases_dir"] == str(o.reg.leases_dir) and rec["generation"] >= 2
    # the row keys are exactly ssb_leg_pool.Machine.public_record's (called unbound on a stand-in)
    stub = types.SimpleNamespace(sid="x", summary={}, state_name=lambda: "ready", t_created=1.0, t_ready=2.0,
                                 cfg=types.SimpleNamespace(rt={"ttl_minutes": 5}), ttl_expires=lambda: 3.0,
                                 retire_at=lambda: 2.5, allow_csv="a,b", active=lambda: [], inflight=lambda: 0,
                                 ramp=None, disk_free_gib=None)
    want = set(P.Machine.public_record(stub))
    row = rec["machines"]["m01"]
    assert set(row) == want, (set(row) ^ want)
    assert row["state"] == "ready" and row["sandbox_id"] == m.sid and row["sandbox_name"].startswith(
        f"{pce.POOL_PREFIX}{P.pool_code(RUN_NAME)}-m01-")
    assert row["domain_allow_list"] == ["openrouter.ai", "api.anthropic.com"] == row["domain_allow_list_readback"].split(",")
    assert row["ttl_minutes"] == 1440 and abs(row["retire_at_ts"] - (row["created_ts"] + (1440 - 255) * 60)) < 1
    assert row["cap"] == 30 and row["leased_by"] == []
    # pick_pool_machine (the evolver's and the leg driver's gate) accepts the published row
    need = 7200 + pce.EVOLVER_MARGIN_S
    picked, why = pce.pick_pool_machine(rec, now=time.time(), need_s=need)
    assert picked and picked["sandbox_id"] == m.sid and picked["name"] == "m01", why
    # the machine's summary carries what the leg driver copies into runtime_env
    s = json.loads((o.store / "m01" / "summary.json").read_text())
    assert s["identity"]["solver_stack"] == "3.1.3 2.2.0 1.26.4" and s["identity"]["libreoffice"] == M.EXPECTED["libreoffice"]
    assert s["egress"]["must_reach"] == ["anthropic", "openrouter"] and s["grade_selftest"]["golden_vs_golden_passed"]
    assert s["bundle"]["verified_on_machine"][H.GRADE_WORKER_VM] == M.sha256_file(H.GRADE_WORKER_SRC)
    assert s["install"]["cc"]["cc"].startswith("uploaded") and s["install"]["ctl"]["ctl"].startswith("uploaded")
    assert s["readback"]["snapshot_ref"] == M.EXPECTED["snapshot_ref"] and s["granted_ttl_min"] == 1440
    o.close()


def test_the_bundle_is_the_grading_path_and_holds_no_solver_setting(world):
    o = make_owner(world, pool_dir=world.store / RUN_NAME / "pool" / "run" / "p1")
    o.reap()
    m = o.create_machine("m01")
    sb = world.client.get(m.sid)
    names = set(sb.bundle_names)
    assert "opt/leg/leg.env" in names and "opt/grade/grade_worker.py" in names and "opt/leg/docker_shim.py" in names
    assert "opt/leg/subset.json" in names
    for t in world.split["train"][:2] + world.split["val"][:2]:
        assert str(world.idx[t]["input"]).lstrip("/") in names, t
    env = {ln.split("=", 1)[0][len("export "):]: shlex.split(ln.split("=", 1)[1])[0]
           for ln in sb.path("/opt/leg/leg.env").read_text().splitlines()}
    assert env["EVOSKILL_RECALC_TIMEOUT_SECONDS"] == "900" and env["EVOSHIM_IMAGE_ID"] == M.EXPECTED["image_id"]
    assert not any(k.startswith(("EVOSKILL_SOLVER", "OPENROUTER")) or "TOKEN" in k for k in env)
    assert sb.path(ctl.CC_HOST).read_bytes() == world.cc_local.read_bytes()
    assert sb.path(ctl.CTL).read_bytes() == H.CTL_SRC.read_bytes()
    steps = [s for s, _ in sb.calls]
    assert steps.index("identity") < steps.index("unpack") < steps.index("cgroups") < steps.index("cc verify")
    assert steps.index("egress") < steps.index("unpack")
    o.close()


def test_every_machine_writes_the_cpu_weights_right_after_its_cgroups_and_a_readback_that_differs_is_kept(world):
    """Rollouts come before grading: the machine setup itself writes cpu.weight
    evoevolver 10000 / evoshim 1 / evoshim_runner 1 right after the cgroup step (the SSB machine: after
    cgroups_and_selftest, i.e. its grading self-test) and before the installs; the external keeper is not needed."""
    o = make_owner(world, pool_dir=world.store / RUN_NAME / "pool" / "run" / "p1")
    o.reap()
    m = o.create_machine("m01")
    sb = world.client.get(m.sid)
    steps = [s for s, _ in sb.calls]
    assert steps.count("cpu weights") == 1 and steps.index("cpu weights") == steps.index("grade") + 1
    assert steps.index("cpu weights") < steps.index("ctl sha")
    script = dict(sb.calls)["cpu weights"]
    assert "evobridge" not in script, "the bridges' cgroup is a tau3 machine's only"
    for g, w in H.CPU_WEIGHTS.items():
        assert f"mkdir -p /sys/fs/cgroup/{g} && echo {w} > /sys/fs/cgroup/{g}/cpu.weight" in script
        assert sb.path(f"/sys/fs/cgroup/{g}/cpu.weight").read_text().strip() == str(w)
    assert H.CPU_WEIGHTS == {"evoevolver": 10000, "evoshim": 1, "evoshim_runner": 1}
    s = json.loads((o.store / "m01" / "summary.json").read_text())
    assert s["cgroup_weights"] == {"want": H.CPU_WEIGHTS, "readback": H.CPU_WEIGHTS, "equal": True}
    o.close()
    # a readback that differs is logged and recorded; the machine is still proven and served (no run-halting guard)
    world.client.cpu_weight_kept = {"evoshim": 100}
    o2 = make_owner(world, pool_dir=world.store / RUN_NAME / "pool" / "run" / "p2")
    o2.reap()
    m2 = o2.create_machine("m01")
    assert m2.usable() and not m2.failed and machines_json(o2.reg)["current"] == "m01"
    rec = json.loads((o2.store / "m01" / "summary.json").read_text())["cgroup_weights"]
    assert rec["equal"] is False and rec["readback"] == dict(H.CPU_WEIGHTS, evoshim=100) and rec["want"] == H.CPU_WEIGHTS
    assert "cpu.weight readback" in (o2.store / "m01" / "driver.log").read_text()
    o2.close()


def test_the_cgroup_step_text_is_the_shards_own():
    """CCMachine.cgroups (a tau3 machine's) runs the delegation half of Shard.cgroups_and_selftest: the same text."""
    seen = []

    class Stop(Exception):
        pass

    def sh(step, cmd, timeout=600, env=None, check=False):
        seen.append((step, cmd))
        if step != "cgroups":
            raise Stop
        return 0, "root=memory cpu pids evoshim=memory cpu pids\n"
    stub = types.SimpleNamespace(sh=sh, summary={}, cfg=types.SimpleNamespace(image_id="x"), tasks=["t"])
    with pytest.raises(Stop):
        M.Shard.cgroups_and_selftest(stub)
    assert seen[0] == ("cgroups", H.CGROUP_CMD) and seen[1][0] == "selftest"


def test_a_machine_that_fails_its_proof_is_deleted_and_the_host_refuses(world):
    world.client.egress_answer = lambda sb, eg: dict(eg, anthropic={"ok": False, "error": "refused"})
    o = make_owner(world, pool_dir=world.store / RUN_NAME / "pool" / "run" / "p1")
    o.reap()
    with pytest.raises(M.Abort, match="api.anthropic.com NOT reachable"):
        o.create_machine("m01")
    m = o.machines[0]
    assert m.failed and m.deleted is True and world.client.deleted == [m.sid] and not world.client.live
    rec = machines_json(o.reg)
    assert rec["current"] is None and rec["machines"]["m01"]["state"] == "deleted"


# ══ registry / spawn / attach ═══════════════════════════════════════════════════════════════════
def test_host_argv_is_this_program_with_the_run_and_pool_dirs(world, monkeypatch, tmp_path):
    argv = H.host_argv(tmp_path / "p", world.rd)
    assert argv[0] == sys.executable and argv[1].endswith("cc_pool_host.py")
    assert argv[2:] == ["host", "--run-dir", str(world.rd), "--pool-dir", str(tmp_path / "p")]
    monkeypatch.setenv(H.HOST_SCRIPT_ENV, str(tmp_path / "standin_host.py"))
    assert H.host_argv(tmp_path / "p", world.rd)[1] == str(tmp_path / "standin_host.py")


def test_spawn_host_detaches_a_host_the_registry_identifies(world, monkeypatch, tmp_path):
    stand_in = tmp_path / "standin_cc_pool_host.py"
    stand_in.write_text("import json, sys, time\n"
                        "open(sys.argv[sys.argv.index('--pool-dir') + 1] + '/argv.json', 'w').write(json.dumps(sys.argv))\n"
                        "time.sleep(3)\n")
    monkeypatch.setenv(H.HOST_SCRIPT_ENV, str(stand_in))
    spawned = []
    reg, new = H.attach_or_spawn(world.rd, tag="t1", spawn=lambda d, r: spawned.append((d, r)) or H.spawn_host(d, r))
    assert new and spawned and spawned[0][1] == world.rd.resolve()
    pdir = Path(reg["pool_dir"])
    wait_for(lambda: (pdir / "argv.json").exists(), "the stand-in host's argv")
    argv = json.loads((pdir / "argv.json").read_text())
    assert argv[1:] == ["host", "--run-dir", str(world.rd.resolve()), "--pool-dir", str(pdir)]
    assert not any(P.RUNNER_PATTERN in a for a in argv)
    assert reg["state"] == "accepting" and reg["owner"] == H.OWNER and reg["host_script"] == str(stand_in)
    assert reg["run_dir"] == str(world.rd.resolve()) and reg["spawned_by"] == {"tag": "t1", "pid": os.getpid()}
    assert P.host_alive(reg), "ssb_leg_pool.host_alive identifies the CC host by program name + pool dir"
    # a second driver joins the same host: no spawn
    reg2, new2 = H.attach_or_spawn(world.rd, tag="t2", spawn=lambda d, r: pytest.fail("spawned a second host"))
    assert not new2 and reg2["host_pid"] == reg["host_pid"]
    os.kill(reg["host_pid"], 9)


def test_attach_refuses_another_runs_host_and_a_deepseek_host_and_waits_for_a_closing_one(world, tmp_path):
    reg = P.Registry(H.pool_root(world.rd))
    old = reg.root / "closinghost"
    old.mkdir(parents=True)
    hp = live_host(old)
    try:
        base = {"state": "accepting", "host_pid": hp.pid, "pool_dir": str(old), "scope": "run",
                "host_script": str(H._host_script()), "owner": H.OWNER, "run_dir": str(world.rd)}
        reg.write(dict(base, run_dir=str(tmp_path / "outputs" / "other_run")))
        with pytest.raises(M.Refused, match="serves run dir"):
            H.attach_or_spawn(world.rd, spawn=lambda d, r: pytest.fail("spawned"))
        ds = ROOT / "scripts/daytona/ssb_leg_pool.py"           # a live DeepSeek pool host of THIS run
        hp_ds = live_host(old, script=ds)
        try:
            reg.write(dict(base, host_pid=hp_ds.pid, owner=None, host_script=str(ds)))
            with pytest.raises(M.Refused, match="not 'cc_pool_host'"):
                H.attach_or_spawn(world.rd, spawn=lambda d, r: pytest.fail("spawned"))
            assert H.attach_or_spawn(world.rd, spawn=lambda d, r: pytest.fail("spawned"), require_owner=None)[1] is False
        finally:
            hp_ds.kill()
        reg.write(dict(base, state="closing"))
        notes = []
        with pytest.raises(M.Refused, match="still alive in state 'closing'.*SECOND machine"):
            H.attach_or_spawn(world.rd, spawn=lambda d, r: pytest.fail("spawned"), closing_wait_s=0.3, poll_s=0.05,
                              note=notes.append)
        assert notes and "waiting up to" in notes[0]
    finally:
        hp.kill()
    hp.wait()
    time.sleep(0.05)
    spawned = []
    rec, new = H.attach_or_spawn(world.rd, tag="t3", spawn=lambda d, r: spawned.append(d) or os.getpid())
    assert new and spawned == [Path(rec["pool_dir"])] and rec["previous"]["state"] == "closing"


def test_reap_deletes_orphans_by_id_and_refuses_while_one_stays_listed(world):
    code = P.pool_code(RUN_NAME)
    orphan = world.client.adopt(f"{dlx.PREFIX}{code}-m01-101010", "orphan01-fake")
    other = world.client.adopt(f"{dlx.PREFIX}rpzzzzz-m01-101010", "otherrun-fake")     # another run's pool machine
    o = make_owner(world, pool_dir=world.store / RUN_NAME / "pool" / "run" / "p1")
    found = o.reap()
    assert [f["id"] for f in found] == [orphan.id] and world.client.deleted == [orphan.id]
    assert other.id in world.client.live, "only THIS run's pool sandboxes are reaped"
    ev = [r["event"] for r in ledger(world)]
    assert "pool_orphan_found" in ev and "deleted" in ev
    # one that cannot be deleted blocks the creation (fail closed)
    stuck = world.client.adopt(f"{dlx.PREFIX}{code}-m01-121212", "stuck01-fake")
    world.client.delete_refuses.add(stuck.id)
    o2 = make_owner(world, pool_dir=world.store / RUN_NAME / "pool" / "run" / "p2")
    with pytest.raises(M.Refused, match="still listed after delete-by-id"):
        o2.reap()
    assert not o2.machines and len(world.client.created) == 0
    world.client.delete_refuses.clear()
    world.client.list_fails = True
    with pytest.raises(M.Refused, match="listing .* failed"):
        make_owner(world, pool_dir=world.store / RUN_NAME / "pool" / "run" / "p3").reap()


def test_register_keeps_the_spawners_record_and_never_displaces_a_live_host(world):
    reg = P.Registry(H.pool_root(world.rd))
    rec, _ = H.attach_or_spawn(world.rd, tag="t", spawn=lambda d, r: os.getpid())
    o = make_owner(world, pool_dir=Path(rec["pool_dir"]))
    assert o.register() == rec, "the spawner's record for this pid is kept as it is"
    other = reg.root / "other"
    other.mkdir()
    hp = live_host(other)
    try:
        reg.write(dict(rec, host_pid=hp.pid, pool_dir=str(other)))
        with pytest.raises(M.Refused, match="live host"):
            o.register()
    finally:
        hp.kill()
    hp.wait()
    reg.write(dict(rec, host_pid=hp.pid, pool_dir=str(other)))
    r = o.register()
    assert r["host_pid"] == os.getpid() and r["pool_dir"] == str(o.pdir) and r["previous"]["pool_dir"] == str(other)


# ══ the host's life: serve, leases, STOP ════════════════════════════════════════════════════════
def test_host_run_serves_until_stop_then_wipes_cc_and_deletes_exactly_its_machine(world):
    other = world.client.adopt(f"{dlx.PREFIX}rpother-m01-101010", "other-fake")
    rec, _ = H.attach_or_spawn(world.rd, tag="t", spawn=lambda d, r: os.getpid())
    o = make_owner(world, pool_dir=Path(rec["pool_dir"]))
    reg = o.reg
    seen = {}

    def driver():
        wait_for(lambda: reg.machines_file.is_file() and machines_json(reg).get("current") == "m01", "ready", 60)
        mj = machines_json(reg)
        sb = world.client.get(mj["machines"]["m01"]["sandbox_id"])
        sb.containers = ["cc-template-01-02-s1-x-ab12", "e1-20260927t000000-abcdef", "ao-1-x"]
        seen["mj"] = mj
        seen["reg"] = reg.read()
        (reg.root / H.STOP_FILE).write_text("{}")
    th = threading.Thread(target=driver, daemon=True)
    th.start()
    rc = o.run()
    th.join(5)
    assert rc == 0 and not seen.get("err")
    assert seen["reg"]["state"] == "accepting" and seen["mj"]["host_state"] == "open"
    m = o.machines[0]
    assert world.client.deleted == [m.sid] and other.id in world.client.live, "exactly the host's sandbox was deleted"
    final = machines_json(reg)
    assert final["host_state"] == "closed" and final["current"] is None and final["machines"]["m01"]["state"] == "deleted"
    r = reg.read()
    assert r["state"] == "closed" and r["closed_utc"] and r["machines_not_verified_deleted"] == []
    pr = json.loads((o.pdir / "pool_record.json").read_text())
    assert pr["machines"]["m01"]["delete"]["verified_gone"] is True and pr["machines_still_listed"] == []
    assert pr["stop_file"] and pr["problems"] == [] and pr["orphans_deleted_at_init"] == []
    # cc-* containers downed, the evolver's left alone (the fake keeps the machine dir after delete)
    sbx = [s for s in Path(world.client.base).iterdir()]
    assert sbx
    summ = json.loads((o.store / "m01" / "summary.json").read_text())
    assert summ["cleanup"]["downed"] == {"cc-template-01-02-s1-x-ab12": {"verified": True, "wiped": True}}
    assert summ["cleanup"]["left_alone"] == ["ao-1-x", "e1-20260927t000000-abcdef"]
    ev = [r for r in ledger(world) if r["event"] == "container_down"]
    assert [e["cid"] for e in ev] == ["cc-template-01-02-s1-x-ab12"] and ev[0]["verified"] is True
    assert [r["event"] for r in ledger(world)].count("cc_pool_machine_ready") == 1
    assert (o.pdir / "status.json").is_file() and (o.pdir / "host.log").is_file()


def test_a_valid_lease_keeps_the_machine_through_stop_until_it_expires(world):
    o = make_owner(world, pool_dir=world.store / RUN_NAME / "pool" / "run" / "p1")
    o.reap()
    m = o.create_machine("m01")
    lease = pce.PoolLease(o.reg.leases_dir, holder="evolver-e1-test", sandbox_id=m.sid, cap_ts=time.time() + 0.6,
                          window_s=0.6)
    lease.write()
    (o.reg.root / H.STOP_FILE).write_text("{}")
    states = []

    def watch():
        while True:
            try:
                mj = machines_json(o.reg)
            except (OSError, ValueError):
                continue
            row = mj["machines"]["m01"]
            states.append((mj["host_state"], row["state"], tuple(row["leased_by"]), m.sid in world.client.live))
            if mj["host_state"] == "closed":
                return
            time.sleep(0.02)
    th = threading.Thread(target=watch, daemon=True)
    th.start()
    t0 = time.time()
    assert o.serve() == "stop"
    o.close()
    th.join(5)
    waited = time.time() - t0
    assert waited >= 0.5, f"the delete waited for the lease ({waited:.2f}s)"
    assert any(s[0] == "closing" and s[2] == ("evolver-e1-test",) and s[3] for s in states), states[:5]
    assert states[-1][0] == "closed" and states[-1][1] == "deleted" and not states[-1][3]
    assert o.lease_waits and o.lease_waits[0]["waited_s"] >= 0.5 and o.lease_waits[0]["holders_at_end"] == []
    # while closing, the pool refuses a new lease (lease_acknowledged says the host is not open)
    closing = [s for s in states if s[0] == "closing"]
    assert closing and pce.lease_acknowledged(dict(machines_json(o.reg), host_state="closing"), name="m01",
                                              sandbox_id=m.sid, holder="x")[0] is False


def test_lease_wait_is_bounded_and_a_dead_holder_pid_frees_the_machine(world):
    o = make_owner(world, pool_dir=world.store / RUN_NAME / "pool" / "run" / "p1")
    o.p = dict(o.p, pool_ttl_margin_min=0.005)          # bound 0.3 s
    o.reap()
    m = o.create_machine("m01")
    o.reg.leases_dir.mkdir(exist_ok=True)
    pce.write_json_atomic(o.reg.leases_dir / "ghost.json",
                          {"holder": "ghost", "sandbox_id": m.sid, "until_ts": time.time() + 3600, "pid": 1})
    t0 = time.time()
    o.close()
    assert time.time() - t0 >= 0.25 and any("still valid after" in p for p in o.problems)
    assert m.deleted is True
    o2 = make_owner(world, pool_dir=world.store / RUN_NAME / "pool" / "run" / "p2")
    o2.reap()
    m2 = o2.create_machine("m01")
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    o2.reg.leases_dir.mkdir(exist_ok=True)
    pce.write_json_atomic(o2.reg.leases_dir / "dead.json",
                          {"holder": "dead", "sandbox_id": m2.sid, "until_ts": time.time() + 3600, "pid": dead.pid})
    t0 = time.time()
    o2.close()
    assert time.time() - t0 < 2.0 and o2.lease_waits[0]["holders_at_end"] == [] and m2.deleted is True


def test_rollover_replaces_the_machine_only_after_it_is_deleted(world):
    o = make_owner(world, pool_dir=world.store / RUN_NAME / "pool" / "run" / "p1")
    o.reap()
    m1 = o.create_machine()
    assert m1.name == "m01"
    (o.reg.root / H.ROLLOVER_FILE).write_text("{}")
    alive_at_create = []
    orig = world.client.create

    def create(params, timeout=None, on_snapshot_create_logs=None):
        alive_at_create.append(sorted(world.client.live))
        return orig(params, timeout, on_snapshot_create_logs)
    world.client.create = create
    th = threading.Thread(target=lambda: (wait_for(lambda: machines_json(o.reg).get("current") == "m02", "m02", 60),
                                          (o.reg.root / H.STOP_FILE).write_text("{}")), daemon=True)
    th.start()
    assert o.serve() == "stop"
    th.join(5)
    assert [m.name for m in o.machines] == ["m01", "m02"] and m1.deleted is True
    assert alive_at_create == [[]], "m02 was created only once m01 was gone (never two alive)"
    assert not (o.reg.root / H.ROLLOVER_FILE).exists()
    with pytest.raises(M.Refused, match="never two"):
        o.create_machine()                            # m02 is alive
    o.close()
    assert world.client.deleted == [m1.sid, o.machines[1].sid] and not world.client.live


def test_stop_command_writes_stop_and_waits_for_closed(world, capsys):
    reg = P.Registry(H.pool_root(world.rd))
    hp = live_host(reg.root / "h")
    try:
        reg.write({"state": "accepting", "host_pid": hp.pid, "pool_dir": str(reg.root / "h"),
                   "host_script": str(H._host_script()), "owner": H.OWNER})
        (reg.root / "h").mkdir(parents=True, exist_ok=True)

        def closer():
            wait_for(lambda: (reg.root / H.STOP_FILE).exists(), "STOP")
            (reg.root / "h" / "pool_record.json").write_text(json.dumps({"machines": {"m01": {"state": "deleted"}},
                                                                          "machines_still_listed": [], "problems": []}))
            reg.write(dict(reg.read(), state="closed", closed_utc="now", machines_not_verified_deleted=[]))
        threading.Thread(target=closer, daemon=True).start()
        rc = H.main(["stop", "--run-dir", str(world.rd), "--wait-s", "10"])
        out = capsys.readouterr().out
        assert rc == 0 and "STOP written" in out and "closed at now" in out and '"m01": "deleted"' in out
        assert H.main(["status", "--run-dir", str(world.rd)]) == 0 and "STOP present" in capsys.readouterr().out
    finally:
        hp.kill()
    hp.wait()
    assert H.main(["stop", "--run-dir", str(world.rd), "--wait-s", "1"]) == 0
    assert "nothing to stop" in capsys.readouterr().out


# ══ join (the leg driver's side) ════════════════════════════════════════════════════════════════
class HostThread:
    """A PoolMachineOwner served from a thread (the fake client), with a live stand-in process as the registry's
    host pid (host_alive's argv identity); the owner publishes machine.json as the real host does."""

    def __init__(self, w, pdir=None):
        self.w = w
        self.reg = P.Registry(H.pool_root(w.rd))
        self.pdir = Path(pdir) if pdir else self.reg.root / "h1"
        self.pdir.mkdir(parents=True, exist_ok=True)
        self.proc = live_host(self.pdir)
        self.reg.write({"state": "accepting", "host_pid": self.proc.pid, "pool_dir": str(self.pdir), "scope": "run",
                        "host_script": str(H._host_script()), "owner": H.OWNER, "run_dir": str(w.rd),
                        "machines_file": str(self.reg.machines_file), "leases_dir": str(self.reg.leases_dir)})
        self.owner = make_owner(w, pool_dir=self.pdir)
        self.th = None

    def start(self, prove=True):
        self.owner.reap()
        if prove:
            self.owner.create_machine("m01")
        self.th = threading.Thread(target=self.owner.serve, daemon=True)
        self.th.start()
        return self

    @property
    def machine(self):
        return self.owner.machines[0]

    def stop(self):
        (self.reg.root / H.STOP_FILE).write_text("{}")
        if self.th:
            self.th.join(10)
        self.proc.kill()
        self.proc.wait()


def test_join_picks_the_ready_machine_leases_it_acknowledged_and_installs_idempotently(world, tmp_path):
    h = HostThread(world).start()
    try:
        dle = fake_dle(world.client)
        handle, lease = H.join(world.rd, "ccleg-t1-20260927", 90 * 60 + 21600 + 1800, tag="t1", epoch=1, key="train",
                               store_dir=tmp_path / "handle", client=world.client, dle=dle, poll_s=0.02,
                               spawn=lambda d, r: pytest.fail("spawned beside a live host"), **cc_kw(world))
        try:
            assert handle.sid == h.machine.sid and handle.machine_name == "m01" and handle.row["state"] == "ready"
            assert lease.path == h.reg.leases_dir / "ccleg-t1-20260927.json" and lease.writes >= 1
            rec = json.loads(lease.path.read_text())
            assert rec["holder"] == "ccleg-t1-20260927" and rec["role"] == "cc_leg" and rec["pid"] == os.getpid()
            assert rec["epoch"] == 1 and rec["key"] == "train" and rec["until_ts"] <= lease.cap_ts
            wait_for(lambda: "ccleg-t1-20260927" in machines_json(h.reg)["machines"]["m01"]["leased_by"], "ack")
            assert handle.rec["lease"]["holder"] == "ccleg-t1-20260927" and handle.rec["join"]["spawned"] is False
            # the handle's surface (cc_solver.Machine's) and the identity from the host's summary
            assert handle.identity["solver_stack"] == "3.1.3 2.2.0 1.26.4" and handle.ttl == 1440
            assert handle.t_created == h.machine.t_created and handle.inf["leg_sandbox"]["image_id"] == M.EXPECTED["image_id"]
            assert handle.rec["readback_join"]["refreshed"] is True and handle.rec["host_summary"]["grade_selftest"]
            assert json.loads((handle.dir / "MACHINE.json").read_text())["sandbox_id"] == handle.sid
            assert handle.rec["install"]["ctl"]["ctl"].startswith("already") and \
                handle.rec["install"]["cc"]["cc"].startswith("already"), "the host installed them: skipped by sha"
            sb = world.client.get(handle.sid)
            assert not [u for u in sb.uploaded if u.endswith(".part") or u.endswith("evoctl.py")][2:], "no re-upload"
            code, out = handle.sh("df -k / | tail -1", what="df")
            assert code == 0 and out.strip()
            with pytest.raises(H.HandleError, match="never deletes"):
                handle.delete()
            # cc-* listing + explicit wipe of the leg's own containers only
            sb.containers = ["cc-a-x-1", "cc-b-y-2", "e1-zzz"]
            assert handle.containers() == {"cc": ["cc-a-x-1", "cc-b-y-2"], "other": ["e1-zzz"]}
            with pytest.raises(H.HandleError, match="non-CC"):
                handle.wipe(["cc-a-x-1", "e1-zzz"])
            res = handle.wipe(["cc-a-x-1"], tag="t1")
            assert res == {"cc-a-x-1": {"verified": True, "wiped": True}} and sb.containers == ["cc-b-y-2", "e1-zzz"]
            assert dle.rows[-1]["event"] == "container_down" and dle.rows[-1]["cid"] == "cc-a-x-1"
        finally:
            assert handle.release() is True and not lease.path.exists()
        wait_for(lambda: machines_json(h.reg)["machines"]["m01"]["leased_by"] == [], "lease gone")
        ev = [r for r in ledger(world) if r["event"] == "cc_leg_attach"]
        assert ev and ev[0]["holder"] == "ccleg-t1-20260927"
    finally:
        h.stop()


def test_join_refuses_a_wrong_prefix_machine_and_an_identity_drift(world, tmp_path, monkeypatch):
    h = HostThread(world).start()
    try:
        dle = fake_dle(world.client)
        sb = world.client.get(h.machine.sid)
        real_name = sb.name
        sb.name = "other-prefix-evo-rp-m01-1"            # not the pool's prefix (machine_problems)
        with pytest.raises(M.Refused, match="has none of the prefixes"):
            H.join(world.rd, "ccleg-bad1", 3600, store_dir=tmp_path / "h1", client=world.client, dle=dle, poll_s=0.02)
        assert not list(h.reg.leases_dir.glob("*.json")), "the lease was released on the refusal"
        sb.name = real_name
        sb.domain_allow_list = "openrouter.ai"
        with pytest.raises(M.Refused, match="domain_allow_list"):
            H.join(world.rd, "ccleg-bad2", 3600, store_dir=tmp_path / "h2", client=world.client, dle=dle, poll_s=0.02)
        sb.domain_allow_list = pce.MACHINE_ALLOW_STR
        summ = h.owner.store / "m01" / "summary.json"
        s = json.loads(summ.read_text())
        s["sandbox_id"] = "someone-else"
        summ.write_text(json.dumps(s))
        with pytest.raises(M.Refused, match="identity != the host's summary"):
            H.join(world.rd, "ccleg-bad3", 3600, store_dir=tmp_path / "h3", client=world.client, dle=dle, poll_s=0.02)
        assert not list(h.reg.leases_dir.glob("*.json"))
        with pytest.raises(M.Refused, match="not a plain file name"):
            H.join(world.rd, "../x", 3600, store_dir=tmp_path / "h4", client=world.client, dle=dle)
    finally:
        h.stop()


def test_join_waits_for_a_starting_machine_and_refuses_at_machine_wait(world, tmp_path):
    h = HostThread(world)
    h.owner.reap()
    h.th = threading.Thread(target=h.owner.serve, daemon=True)     # serving, but no machine yet
    h.th.start()
    try:
        dle = fake_dle(world.client)
        with pytest.raises(M.Refused, match="no readable pool machine.json|no current machine"):
            H.join(world.rd, "ccleg-w1", 3600, store_dir=tmp_path / "w1", client=world.client, dle=dle,
                   machine_wait_s=0.3, poll_s=0.05, spawn=lambda d, r: pytest.fail("spawned beside a live host"))
        notes = []
        th = threading.Thread(target=lambda: (time.sleep(0.3), h.owner.create_machine("m01")), daemon=True)
        th.start()
        handle, lease = H.join(world.rd, "ccleg-w2", 3600, store_dir=tmp_path / "w2", client=world.client, dle=dle,
                               machine_wait_s=30, poll_s=0.05, note=notes.append, **cc_kw(world))
        try:
            assert handle.sid == h.machine.sid and handle.rec["join"]["waited_s"] >= 0.2
            assert any("not usable yet" in n for n in notes)
        finally:
            handle.release()
    finally:
        h.stop()


def test_join_spawns_a_host_when_none_is_alive_and_respawns_a_dead_one(world, tmp_path):
    reg = P.Registry(H.pool_root(world.rd))
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    reg.write({"state": "accepting", "host_pid": dead.pid, "pool_dir": str(reg.root / "dead"), "scope": "run",
               "host_script": str(H._host_script()), "owner": H.OWNER, "run_dir": str(world.rd)})
    hosts = []

    def spawn(pdir, run_dir):
        h = HostThread.__new__(HostThread)
        h.w, h.reg, h.pdir = world, reg, Path(pdir)
        h.proc = live_host(h.pdir)
        h.owner = make_owner(world, pool_dir=h.pdir)
        hosts.append(h)
        h.th = threading.Thread(target=lambda: (h.owner.reap(), h.owner.create_machine("m01"), h.owner.serve()),
                                daemon=True)
        h.th.start()
        return h.proc.pid
    dle = fake_dle(world.client)
    handle, lease = H.join(world.rd, "ccleg-s1", 3600, tag="t", store_dir=tmp_path / "s1", client=world.client,
                           dle=dle, machine_wait_s=60, poll_s=0.05, spawn=spawn, **cc_kw(world))
    try:
        assert len(hosts) == 1 and handle.rec["join"]["spawned"] is True
        r = reg.read()
        assert r["host_pid"] == hosts[0].proc.pid and r["previous"]["host_pid"] == dead.pid
        assert handle.rec["registry"]["pool_dir"] == str(hosts[0].pdir)
    finally:
        handle.release()
        for h in hosts:
            h.stop()


# ══ in-place grading ════════════════════════════════════════════════════════════════════════════
def make_handle(w, tmp_path, sb=None):
    """A handle on a fake machine without the join (the grader's unit tests)."""
    if sb is None:
        p = _Params(name=f"{dlx.PREFIX}{P.pool_code(RUN_NAME)}-m01-000000", ttl_minutes=1440,
                    domain_allow_list=pce.MACHINE_ALLOW_STR)
        sb = w.client.create(p)
    row = {"sandbox_id": sb.id, "sandbox_name": sb.name, "name": "m01", "state": "ready", "created_ts": time.time(),
           "ttl_minutes": 1440, "retire_at_ts": time.time() + 3600, "ttl_expires_ts": time.time() + 7200}
    inf = json.loads((w.rd / "INFRA.json").read_text())
    return H.PoolMachineHandle(sb, row=row, registry={}, machine_rec={}, summary={"identity": {}}, dir=tmp_path / "h",
                               inf=inf, dle=fake_dle(w.client))


TOY_PLUGIN = "from pathlib import Path\n\n\ndef list_tasks(root: Path) -> list:\n    return []\n"
TOY_GRADER = "from pathlib import Path\n\n\ndef grade(task, snapshot: Path) -> float:\n    return 1.0\n"
# the grader's setup prefix, byte for byte: the leg's knobs, the runner cgroup, the repo mirror, no OpenRouter key
RUNNER_PREFIX = ("set -uo pipefail; . /opt/leg/leg.env; echo $$ > /sys/fs/cgroup/evoshim_runner/cgroup.procs "
                 f"2>/dev/null; cd {P.SANDBOX_REPO} && env -u OPENROUTER_API_KEY /opt/evoskill/venv/bin/python -u ")


def snapshot_tgz(files: dict) -> bytes:
    """An out.tgz as evoctl pack builds it: evo/work (the solver's working directory) + evo/out + home/evo/.claude."""
    bio = io.BytesIO()
    with tarfile.open(fileobj=bio, mode="w:gz") as tf:
        for name, data in {**files, "evo/out/RESULT.json": b"{}", "home/evo/.claude/settings.json": b"{}"}.items():
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
    return bio.getvalue()


def put_tgz(sb, cdir: str, data: bytes) -> None:
    p = sb.path(f"{cdir}/state/out.tgz")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def grading_setup(w, tmp_path, n_tasks: int = 3):
    """A fake machine as the bundle leaves it (grade_worker.py, every task file at its frozen path) and a frozen
    plug-in as cc_leg_lib.frozen_plugin returns it: setup/{plugin_api.py (the canonical file), benchmark_plugin.py,
    benchmark_grader.py, tasks.json}; each task has an input and a reference file (with a space in its name)."""
    handle = make_handle(w, tmp_path)
    sb = handle.sb
    tasks = w.split["train"][:n_tasks]
    setup = tmp_path / "run" / "setup"
    setup.mkdir(parents=True)
    rows = []
    for t in tasks:
        inp = Path(w.idx[t]["input"])
        gold = inp.with_name(inp.stem + " golden.xlsx")
        gold.write_bytes(b"gold " + t.encode())
        rows.append({"id": t, "category": t.split("/")[0], "instruction": f"do {t}", "inputs": [str(inp)],
                     "answer_files": [str(gold)], "grading": {"answer_position": "A1"}, "split": "train"})
        for p in (inp, gold):
            sb.path(p).parent.mkdir(parents=True, exist_ok=True)
            sb.path(p).write_bytes(p.read_bytes())
    (setup / "plugin_api.py").write_bytes((ROOT / "src" / "orchestrator" / "plugin_api.py").read_bytes())
    (setup / "benchmark_plugin.py").write_text(TOY_PLUGIN)
    (setup / "benchmark_grader.py").write_text(TOY_GRADER)
    (setup / "tasks.json").write_text(json.dumps(rows))
    plugin = {"files": {n: {"path": str(setup / n), "sha256": sha((setup / n).read_bytes())} for n in PR.SETUP_FILES},
              "tasks": rows, "tasks_json_sha256": sha((setup / "tasks.json").read_bytes())}
    sb.path(H.GRADE_WORKER_VM).parent.mkdir(parents=True, exist_ok=True)
    sb.path(H.GRADE_WORKER_VM).write_bytes(H.GRADE_WORKER_SRC.read_bytes())
    return types.SimpleNamespace(handle=handle, sb=sb, tasks=tasks, rows={r["id"]: r for r in rows}, plugin=plugin,
                                 setup=setup, layout=PR.material_layout(w.tmp / "data", SPLIT, ROOT),
                                 knobs=M.leg_knobs(M.load_infra(w.rd)[0], {}))


def make_grader(w, gs, store, **kw):
    return H.InPlaceGrader(gs.handle, run_name=RUN_NAME, tag=w.client.tag, knobs=gs.knobs, plugin=gs.plugin,
                           task_rows=gs.rows, layout=gs.layout, store=store, **kw)


def test_grader_install_is_proven_idempotent_and_refuses_before_any_rollout(world, tmp_path, monkeypatch):
    """install() puts the grader's setup of the leg on the machine and proves it: the
    runner next to the bundle's grade_worker.py, the plug-in dir holding exactly the pinned files + tasks.json, the
    user's material at the adapter's /data paths, every task file of the leg == the Mac's bytes."""
    monkeypatch.setattr(H, "TASK_FILES_PER_CALL", 4)
    gs = grading_setup(world, tmp_path)
    sb = gs.sb
    g = make_grader(world, gs, tmp_path / "sg")
    assert g.plugin_dir == f"/opt/cc/{RUN_NAME}/{world.client.tag}/plugin" and g.setup is None
    rec = g.install(gs.tasks)
    assert rec["runner"] == {"uploaded": True, "sha256": {H.RUNNER_VM: sha(H.RUNNER_SRC.read_bytes()),
                                                         H.GRADE_WORKER_VM: sha(H.GRADE_WORKER_SRC.read_bytes())}}
    assert sb.path(H.RUNNER_VM).read_bytes() == H.RUNNER_SRC.read_bytes()
    assert not list(sb.path("/opt/grade").glob("*.part")), "the runner was put in place by a rename"
    names = sorted([*PR.SETUP_FILES, "tasks.json"])
    assert sorted(os.listdir(sb.path(g.plugin_dir))) == names == rec["plugin"]["uploaded"]
    for n in names:
        assert sb.path(f"{g.plugin_dir}/{n}").read_bytes() == (gs.setup / n).read_bytes(), n
    assert rec["plugin"]["removed"] == [] and rec["plugin"]["sha256"] == g.expect
    assert g.expect == {**{n: f["sha256"] for n, f in gs.plugin["files"].items()},
                        "tasks.json": gs.plugin["tasks_json_sha256"]}
    # the user's material at the adapter's /data paths (plugin_runner.material_layout), links to the mirrors
    assert set(rec["links"]) == {"/data/data", f"/data/{SPLIT.name}", "/data/EvoSkill"}
    for mac, vm in gs.layout.items():
        assert os.readlink(sb.path(vm)) == mac and rec["links"][vm] == {"target": mac, "action": "made"}
    # every input and reference file of the leg, a name with a space included
    paths = sorted(p for t in gs.tasks for p in gs.rows[t]["inputs"] + gs.rows[t]["answer_files"])
    assert rec["task_files"]["n"] == len(paths) == 2 * len(gs.tasks) and any(" " in p for p in paths)
    assert rec["task_files"]["sha256"] == {p: sha(Path(p).read_bytes()) for p in paths}
    assert [s for s, _ in sb.calls].count("task files sha") == 2, "6 paths, 4 per exec"
    assert g.setup is rec and g.summary()["setup"] is rec
    # idempotent: a relaunch's grader uploads nothing and keeps the links
    n_up = len(sb.uploaded)
    g2 = make_grader(world, gs, tmp_path / "sg2")
    rec2 = g2.install(gs.tasks)
    assert len(sb.uploaded) == n_up and rec2["runner"]["uploaded"] is False and rec2["plugin"]["uploaded"] == []
    assert {v["action"] for v in rec2["links"].values()} == {"kept"}
    # a stale runner is re-uploaded, a stranger in the plug-in dir removed, a link pointing elsewhere replaced
    vm = "/data/EvoSkill"
    sb.path(vm).unlink()
    os.symlink("/elsewhere", sb.path(vm))
    sb.path(H.RUNNER_VM).write_text("an older runner")
    sb.path(f"{g.plugin_dir}/stray.py").write_text("x")
    sb.path(f"{g.plugin_dir}/tasks.json").write_text("[]")
    rec3 = g2.install(gs.tasks)
    assert rec3["runner"]["uploaded"] and sb.path(H.RUNNER_VM).read_bytes() == H.RUNNER_SRC.read_bytes()
    assert rec3["plugin"]["removed"] == ["stray.py"] and rec3["plugin"]["uploaded"] == ["tasks.json"]
    assert sorted(os.listdir(sb.path(g.plugin_dir))) == names
    assert rec3["links"][vm]["action"] == "made" and os.readlink(sb.path(vm)) == str(ROOT)
    # refusals: a GradeError FAILS the leg before any rollout
    gold = gs.rows[gs.tasks[1]]["answer_files"][0]
    good = sb.path(gold).read_bytes()
    sb.path(gold).write_bytes(b"other bytes")
    with pytest.raises(H.GradeError, match="golden.xlsx on the machine .* is not the Mac's bytes"):
        g2.install(gs.tasks)
    sb.path(gold).unlink()
    with pytest.raises(H.GradeError, match=r"golden.xlsx on the machine \(missing\)"):
        g2.install(gs.tasks)
    sb.path(gold).write_bytes(good)
    with pytest.raises(H.GradeError, match="not in the frozen tasks.json"):
        g2.install(gs.tasks + ["Nope/1"])
    sb.path(vm).unlink()
    sb.path(vm).mkdir()
    with pytest.raises(H.GradeError, match="real directory or file, not a link"):
        g2.install(gs.tasks)
    assert sb.path(vm).is_dir() and not sb.path(vm).is_symlink(), "a real directory is never replaced"
    sb.path(vm).rmdir()
    frozen = (gs.setup / "benchmark_grader.py").read_bytes()
    (gs.setup / "benchmark_grader.py").write_text("edited after the freeze")
    sb.path(f"{g.plugin_dir}/benchmark_grader.py").write_text("x")
    with pytest.raises(H.GradeError, match="plug-in dir .* does not verify"):
        g2.install(gs.tasks)
    (gs.setup / "benchmark_grader.py").write_bytes(frozen)
    sb.path(H.GRADE_WORKER_VM).write_text("another grade worker")
    with pytest.raises(H.GradeError, match="does not verify"):
        g2.install(gs.tasks)
    sb.path(H.GRADE_WORKER_VM).write_bytes(H.GRADE_WORKER_SRC.read_bytes())
    assert g2.install(gs.tasks)["links"][vm]["action"] == "made"
    with pytest.raises(H.GradeError, match="plain"):
        make_grader(world, types.SimpleNamespace(**{**vars(gs), "plugin": {**gs.plugin, "files": {
            "../x.py": gs.plugin["files"]["plugin_api.py"]}}}), tmp_path / "sg3")


def test_grader_install_checks_the_task_files_through_the_data_overlay(world, tmp_path):
    gs = grading_setup(world, tmp_path, n_tasks=1)
    t = gs.tasks[0]
    inp = Path(gs.rows[t]["inputs"][0])
    rel = inp.relative_to(world.tmp / "data")
    patched = tmp_path / "patched" / rel
    patched.parent.mkdir(parents=True)
    patched.write_bytes(b"patched bytes")
    pj = tmp_path / "PATCHES.json"
    pj.write_text(json.dumps({"src": str(world.tmp / "data"), "dst": str(tmp_path / "patched"),
                              "files": [{"path": str(rel), "sha256_before": sha(inp.read_bytes()),
                                         "sha256_after": sha(b"patched bytes")}]}))
    g = make_grader(world, gs, tmp_path / "sg", overlay=str(pj))
    with pytest.raises(H.GradeError, match="not the Mac's bytes through the data overlay"):
        g.install([t])                               # the machine holds the canonical bytes, not the overlay's
    gs.sb.path(inp).write_bytes(b"patched bytes")
    assert g.install([t])["task_files"]["sha256"][str(inp)] == sha(b"patched bytes")


def test_grader_keeps_the_whole_snapshot_builds_the_job_and_the_runner_command(world, tmp_path):
    gs = grading_setup(world, tmp_path)
    sb = gs.sb
    g = make_grader(world, gs, tmp_path / "sg", max_procs=4, max_heavy=1)
    task = gs.tasks[0]
    jid = H.job_id_for(task, 0)
    assert jid == task.replace("/", "_") + "__s0"
    data = snapshot_tgz({"evo/work/output/book.xlsx": b"PK-fake-workbook"})
    mac = tmp_path / "container_out.tgz"             # the Mac's runs/<ud>/container_out.tgz: the same bytes
    mac.write_bytes(data)
    put_tgz(sb, ctl.cdir("cc-unit-1"), data)
    kept = g.keep_snapshot(jid, ctl.cdir("cc-unit-1"), sha(data), mac_tgz=mac)
    assert kept["path"] == f"/opt/cc/{RUN_NAME}/{world.client.tag}/units/{jid}/out.tgz" == g.tgz_path(jid) \
        == g.tgz_path(jid + "#regrade1")
    assert kept["matches"] and not kept["reuploaded"] and kept["from"] == "container" and kept["rc"] == 0
    assert sb.path(kept["path"]).read_bytes() == data and not [u for u in sb.uploaded if u.endswith("out.tgz")]
    # a mismatch: the Mac copy is uploaded; wrong again (or no Mac copy) is an error
    put_tgz(sb, ctl.cdir("cc-unit-2"), b"not the collected bytes")
    kept2 = g.keep_snapshot(jid + "#a2", ctl.cdir("cc-unit-2"), sha(data), mac_tgz=mac)
    assert kept2["matches"] and kept2["reuploaded"] and sb.path(kept2["path"]).read_bytes() == data
    with pytest.raises(H.GradeError, match="is not the Mac's container_out.tgz"):
        g.keep_snapshot(jid + "#a3", ctl.cdir("cc-unit-2"), sha(data))
    with pytest.raises(H.GradeError, match="is not the Mac's container_out.tgz"):
        g.keep_snapshot(jid + "#a4", ctl.cdir("cc-unit-2"), "0" * 64, mac_tgz=mac)
    sb.fail_keep = True                               # the container's copy cannot be read: the Mac copy
    assert g.keep_snapshot(jid + "#a5", ctl.cdir("cc-unit-1"), sha(data), mac_tgz=mac)["reuploaded"]
    sb.fail_keep = False
    # cdir=None (regrade_leg.py): the Mac copy, directly
    kept6 = g.keep_snapshot(H.job_id_for(task, 1), None, sha(data), mac_tgz=mac)
    assert kept6["matches"] and kept6["from"] == "mac" and not kept6["reuploaded"]
    assert sb.path(kept6["path"]).read_bytes() == data
    # the job: six keys, nothing benchmark-specific
    j = g.job(job_id=jid, task_id=task, snapshot_sha256=sha(data), cid="cc-unit-1",
              collected_utc="2026-09-30T10:00:00Z", unit=f"{task}#s1")
    assert PGR.validate_job(j) is j and set(j) == {"job_id", "task_id", "snapshot_tgz", "snapshot_sha256",
                                                    "snapshot_root", "meta"}
    assert j["snapshot_tgz"] == kept["path"] and j["snapshot_root"] == "evo/work" and j["snapshot_sha256"] == sha(data)
    assert j["meta"] == {"unit": f"{task}#s1", "tag": world.client.tag, "cid": "cc-unit-1",
                         "collected_utc": "2026-09-30T10:00:00Z"}
    with pytest.raises(H.GradeError, match="not in the frozen tasks.json"):
        g.job(job_id="x__s0", task_id="Nope/1", snapshot_sha256=sha(data))
    with pytest.raises(H.GradeError, match="snapshot_sha256"):
        g.job(job_id=jid, task_id=task, snapshot_sha256="not-a-sha")
    # the launch: the grader's setup prefix byte for byte, then the runner; argv exact
    cmd = g.launch_cmd()
    assert cmd.startswith("(setsid nohup bash -c '") and cmd.endswith("' </dev/null >/dev/null 2>&1 &); sleep 2")
    inner = cmd[len("(setsid nohup bash -c '"):-len("' </dev/null >/dev/null 2>&1 &); sleep 2")]
    assert P.SANDBOX_REPO == str(M.REPO) and H.runner_shell(["x"]) == RUNNER_PREFIX + H.RUNNER_VM + " x"
    assert inner.startswith(RUNNER_PREFIX + "/opt/grade/plugin_grade_runner.py run --jobs ")
    assert "env -u OPENROUTER_API_KEY /opt/evoskill/venv/bin/python -u /opt/grade/plugin_grade_runner.py run" in inner
    argv = shlex.split(inner.split(" run ", 1)[1].split(" >> ", 1)[0])
    f = gs.plugin["files"]
    want = ["--jobs", g.jobs_file, "--results", g.results_file, "--work", f"{H.GRADE_WORK_VM}/{world.client.tag}",
            "--plugin-dir", g.plugin_dir,
            "--expect", f"benchmark_grader.py={f['benchmark_grader.py']['sha256']}",
            "--expect", f"benchmark_plugin.py={f['benchmark_plugin.py']['sha256']}",
            "--expect", f"plugin_api.py={f['plugin_api.py']['sha256']}",
            "--expect", f"tasks.json={gs.plugin['tasks_json_sha256']}",
            "--max-procs", "4", "--max-heavy", "1", "--heavy-bytes", "5000000", "--job-timeout", "5400"]
    assert argv == want and PGR.parse_expect(argv[i + 1] for i, a in enumerate(argv) if a == "--expect") == g.expect
    assert inner.endswith(f">> {g.worker_log} 2>&1; echo $? > {g.rc_file}")
    assert g.jobs_file == f"/opt/cc/{RUN_NAME}/{world.client.tag}/grade/jobs.jsonl"
    # a runner launched before install() refuses (its plug-in dir does not verify): no line, rc 3
    res = g.enqueue(g.job(job_id=H.job_id_for(task, 1), task_id=task, snapshot_sha256=sha(data)))
    assert res["launched"] and g.worker_state()["rc"] == "3" and g.poll() == []
    # the driver's one call per collected unit, once installed: keep + job + enqueue (+ the runner's launch)
    g.install(gs.tasks)
    put_tgz(sb, ctl.cdir("cc-unit-9"), data)
    res = g.enqueue_unit(job_id=H.job_id_for(task, 2), task_id=task, cdir=ctl.cdir("cc-unit-9"),
                         snapshot_sha256=sha(data), mac_tgz=mac, cid="cc-unit-9", unit=f"{task}#s3")
    assert res["kept"]["matches"] and res["worker"]["launched"] and PGR.validate_job(res["job"])
    assert sb.worker_args["plugin-dir"] == g.plugin_dir and sb.worker_args["expect"] == g.expect
    lines = g.poll()
    assert [ln["job_id"] for ln in lines] == [H.job_id_for(task, 1), H.job_id_for(task, 2)]
    assert [H.score_of(ln) for ln in lines] == [("ok", 1.0, None)] * 2 and g.summary()["statuses"] == {"ok": 2}
    assert H.score_of is PGR.score_of
    assert H.PoolMachineOwner.join is H.join and H.PoolMachineOwner.attach_or_spawn is H.attach_or_spawn


def test_grader_enqueue_polls_new_lines_once_regrades_once_and_cleans_up(world, tmp_path):
    gs = grading_setup(world, tmp_path)
    sb = gs.sb
    g = make_grader(world, gs, tmp_path / "sg")
    g.install(gs.tasks)
    jobs = []
    for i, t in enumerate(gs.tasks):
        jid = H.job_id_for(t, i % 3)
        data = snapshot_tgz({f"evo/work/output/b{i}.xlsx": f"wb{i}".encode()})
        put_tgz(sb, ctl.cdir(f"cc-u{i}"), data)
        g.keep_snapshot(jid, ctl.cdir(f"cc-u{i}"), sha(data))
        jobs.append(g.job(job_id=jid, task_id=t, snapshot_sha256=sha(data), cid=f"cc-u{i}"))
    sb.hold_worker = True                                # the runner runs "in the background"
    sb.grade_score[jobs[1]["job_id"]] = 0.25
    sb.grade_status[jobs[2]["job_id"]] = "grader_failed"
    r = g.enqueue(jobs[0])
    assert r["launched"] and r["launch_n"] == 1 and sb.launches == 1
    r = g.enqueue(jobs[1])
    assert not r["launched"] and r["running"], "one runner per leg: the running one picks the new line up (RESUME)"
    with pytest.raises(H.GradeError, match="already enqueued"):
        g.enqueue(jobs[0])
    on_machine = [json.loads(ln) for ln in sb.path(g.jobs_file).read_text().splitlines()]
    assert on_machine == [jobs[0], jobs[1]] == [json.loads(ln) for ln in (g.store / "jobs.jsonl").read_text().splitlines()]
    assert g.poll() == [] and g.pending() == [jobs[0]["job_id"], jobs[1]["job_id"]]
    sb.finish_worker()                                    # both lines land, rc written
    new = g.poll()
    assert [r["job_id"] for r in new] == [jobs[0]["job_id"], jobs[1]["job_id"]]
    assert g.poll() == [] and g.pending() == [] and (g.store / "worker_meta.json").is_file()
    assert [H.score_of(x) for x in new] == [("ok", 1.0, None), ("ok", 0.25, None)]
    # a job enqueued after the runner exited relaunches it; a grader_failed line -> exactly ONE regrade, same tgz
    sb.hold_worker = False
    r = g.enqueue(jobs[2])
    assert r["launched"] and r["launch_n"] == 2 and r["rc"] == "0"
    line = g.poll()[0]
    assert H.score_of(line) == ("grader_failed", None, "grade() raised ValueError: toy says no")
    sb.grade_status.pop(jobs[2]["job_id"])
    rg = g.regrade(jobs[2]["job_id"])
    assert rg["job_id"] == jobs[2]["job_id"] + "#regrade1" and rg["meta"]["regrade_of"] == jobs[2]["job_id"]
    assert rg["snapshot_tgz"] == jobs[2]["snapshot_tgz"] == g.tgz_path(rg["job_id"]) and PGR.validate_job(rg)
    assert rg["snapshot_sha256"] == jobs[2]["snapshot_sha256"]
    assert [r["job_id"] for r in g.poll()] == [rg["job_id"]] and H.score_of(g.seen[rg["job_id"]]) == ("ok", 1.0, None)
    with pytest.raises(H.GradeError, match="already re-graded once"):
        g.regrade(jobs[2]["job_id"])
    with pytest.raises(H.GradeError, match="already re-graded once"):
        g.regrade(rg["job_id"])
    s = g.summary()
    assert s["worker"] == "plugin_grade/1" == PGR.WORKER_VERSION and s["jobs"] == 4
    assert s["statuses"] == {"ok": 3, "grader_failed": 1} and s["in_place"] is True and s["launches"] == 3
    assert s["regrades"] == {jobs[2]["job_id"]: rg["job_id"]} and s["pending"] == 0
    assert s["runner_sha256"] == sha(H.RUNNER_SRC.read_bytes()) and s["results"] == str(g.store / "results.jsonl")
    assert s["plugin"] == {"files": {n: gs.plugin["files"][n]["sha256"] for n in PR.SETUP_FILES},
                           "tasks_json_sha256": gs.plugin["tasks_json_sha256"]}
    assert s["layout"] == gs.layout and s["setup"] is g.setup and s["machine"]["plugin_dir"] == g.plugin_dir
    assert (s["max_procs"], s["max_heavy"], s["heavy_bytes"], s["job_timeout_s"]) == (4, 1, 5_000_000, 5400)
    assert g.grader_record() == {"worker": "plugin_grade/1", "runner_sha256": s["runner_sha256"],
                                 "plugin": s["plugin"], "interpreter": "/opt/evoskill/venv/bin/python"}
    # the kept tgz changed under the runner: 'refused', the loop's own fault (the driver fails the leg on it)
    sb.path(jobs[0]["snapshot_tgz"]).write_bytes(b"changed")
    g.enqueue(dict(jobs[0], job_id=jobs[0]["job_id"] + "#a2"))
    assert H.score_of(g.poll()[0])[0] == "refused"
    # cleanup: units (the kept snapshots) + work dir go; results and the plug-in dir stay
    assert sb.path(g.units_dir).is_dir()
    sb.path(g.work).mkdir(parents=True, exist_ok=True)
    assert g.cleanup()["ok"] and not sb.path(g.units_dir).exists() and not sb.path(g.work).exists()
    assert sb.path(g.results_file).is_file() and sorted(os.listdir(sb.path(g.plugin_dir))) == sorted(
        [*PR.SETUP_FILES, "tasks.json"])
    with pytest.raises(H.GradeError, match="must be"):
        H.InPlaceGrader(gs.handle, run_name="bad name", tag="t", knobs=gs.knobs, plugin=gs.plugin, task_rows=gs.rows,
                        layout=gs.layout)


def test_install_helpers_skip_when_the_sha_matches_and_refuse_a_bad_binary(world, tmp_path):
    handle = make_handle(world, tmp_path)
    sb = handle.sb
    fake = world.cc_local
    rec = handle.install(fake, sha256=sha(fake.read_bytes()), nbytes=fake.stat().st_size)
    assert rec["ctl"]["ctl"].startswith("uploaded") and rec["cc"]["cc"].startswith("uploaded")
    assert sb.path(ctl.CC_HOST).read_bytes() == fake.read_bytes() and sb.path(ctl.PROXY_FILE).is_file()
    n = len(sb.uploaded)
    rec2 = handle.install(fake, sha256=sha(fake.read_bytes()), nbytes=fake.stat().st_size)
    assert rec2["ctl"]["ctl"].startswith("already") and rec2["cc"]["cc"].startswith("already") and len(sb.uploaded) == n
    with pytest.raises(M.Abort, match="not the pinned Claude Code"):
        handle.install(fake, sha256="0" * 64, nbytes=fake.stat().st_size)
    bad = tmp_path / "bad"
    bad.write_bytes(b"other")
    with pytest.raises(M.Abort, match="not the pinned"):
        handle.install(bad, sha256=sha(fake.read_bytes()), nbytes=fake.stat().st_size)
    assert H.cc_cids({"containers": ["e1-a", "cc-z", "ao-b", "cc-a"]}) == ["cc-a", "cc-z"]
    assert H.other_cids({"containers": ["e1-a", "cc-z", "ao-b"]}) == ["ao-b", "e1-a"]


def test_a_new_host_replacing_a_dead_one_starts_without_the_dead_hosts_stop_or_rollover_file(world):
    """`stop` writes <pool root>/STOP and the host never removes it; the next attach_or_spawn after that host died
    would spawn a host that creates a machine and then closes at its first serve-loop check ("STOP file present:
    closing"). A STOP / ROLLOVER addressed to a dead host must not
    outlive it."""
    reg = P.Registry(H.pool_root(world.rd))
    old = reg.root / "deadhost"
    old.mkdir(parents=True)
    reg.write({"state": "accepting", "host_pid": 999999, "pool_dir": str(old),     # no such host: not alive
               "scope": "run", "host_script": str(H._host_script()), "owner": H.OWNER, "run_dir": str(world.rd)})
    (reg.root / H.STOP_FILE).write_text('{"utc": "x", "by_pid": 1}\n')
    (reg.root / H.ROLLOVER_FILE).write_text("x\n")
    spawned = []
    rec, new = H.attach_or_spawn(world.rd, tag="t4", spawn=lambda d, r: spawned.append(d) or os.getpid())
    assert new and spawned and rec["state"] == "accepting"
    assert not (reg.root / H.STOP_FILE).exists() and not (reg.root / H.ROLLOVER_FILE).exists()


def test_worker_pattern_cannot_match_the_probe_that_greps_for_it(world):
    """`pgrep -f` sees its own shell (the exec's argv carries the pattern) and would report the grade worker RUNNING
    while none exists, so no worker would ever be launched. The pattern's first character is
    bracketed: the regex never matches its own text."""
    g = H.InPlaceGrader.__new__(H.InPlaceGrader)
    g.jobs_file = "/opt/cc/r/t/grade/jobs.jsonl"
    pat = g.worker_pattern()
    assert pat.startswith("[/]opt/grade/plugin_grade_runner.py run --jobs ")       # the plug-in runner
    import re
    assert re.search(pat, "bash -c \"if pgrep -f '" + pat + "' >/dev/null; then echo WORKER=RUNNING; fi\"") is None
    assert re.search(pat, "/opt/evoskill/venv/bin/python -u /opt/grade/plugin_grade_runner.py run --jobs " +
                     g.jobs_file) is not None
    assert re.search(pat, "/opt/evoskill/venv/bin/python -u /opt/grade/grade_worker.py run --jobs " + g.jobs_file) is None


def frozen_launch_cmd(g) -> str:
    """InPlaceGrader.launch_cmd() with the default interpreter, spelled out for grader `g`: the default must stay this
    string byte for byte."""
    argv = ["run", "--jobs", g.jobs_file, "--results", g.results_file, "--work", g.work, "--plugin-dir", g.plugin_dir]
    for n, s in sorted(g.expect.items()):
        argv += ["--expect", f"{n}={s}"]
    argv += ["--max-procs", "4", "--max-heavy", "1", "--heavy-bytes", "5000000", "--job-timeout", "5400"]
    return ("(setsid nohup bash -c 'set -uo pipefail; . /opt/leg/leg.env; echo $$ > "
            f"/sys/fs/cgroup/evoshim_runner/cgroup.procs 2>/dev/null; cd {P.SANDBOX_REPO} && env -u "
            "OPENROUTER_API_KEY /opt/evoskill/venv/bin/python -u /opt/grade/plugin_grade_runner.py " + " ".join(argv) +
            f" >> {g.worker_log} 2>&1; echo $? > {g.rc_file}' </dev/null >/dev/null 2>&1 &); sleep 2")


def test_the_grader_interpreter_is_a_parameter_and_its_default_is_todays_command(world, tmp_path):
    """InPlaceGrader(runner_py=...) runs the plug-in runner (and so the grader's child) under another
    interpreter -- a tau3 leg's /opt/tau3/venv/bin/python, where tau2 is; without it every command string is the
    default one and grader_record() names RUNNER_PY."""
    gs = grading_setup(world, tmp_path)
    g = make_grader(world, gs, tmp_path / "sg")
    assert g.runner_py == H.RUNNER_PY == "/opt/evoskill/venv/bin/python"
    assert g.launch_cmd() == frozen_launch_cmd(g)
    assert H.runner_shell(["x"]) == H.runner_shell(["x"], H.RUNNER_PY) == RUNNER_PREFIX + H.RUNNER_VM + " x"
    assert g.grader_record()["interpreter"] == H.RUNNER_PY
    tau3_py = "/opt/tau3/venv/bin/python"
    g3 = make_grader(world, gs, tmp_path / "sg3", runner_py=tau3_py)
    cmd = g3.launch_cmd()
    assert f"env -u OPENROUTER_API_KEY {tau3_py} -u {H.RUNNER_VM} run --jobs " in cmd and H.RUNNER_PY not in cmd
    assert cmd == frozen_launch_cmd(g3).replace(H.RUNNER_PY, tau3_py), "only the interpreter differs"
    assert H.runner_shell(["x"], tau3_py) == RUNNER_PREFIX.replace(H.RUNNER_PY, tau3_py) + H.RUNNER_VM + " x"
    assert g3.grader_record()["interpreter"] == tau3_py and "interpreter" not in g3.summary()
    # the runner is launched with that interpreter on the machine, and grades as before
    g3.install(gs.tasks)
    data = snapshot_tgz({"evo/work/tau2/simulation.json": b"{}"})
    put_tgz(gs.sb, ctl.cdir("cc-unit-t3"), data)
    res = g3.enqueue_unit(job_id=H.job_id_for(gs.tasks[0], 0), task_id=gs.tasks[0], cdir=ctl.cdir("cc-unit-t3"),
                          snapshot_sha256=sha(data), cid="cc-unit-t3")
    assert res["worker"]["launched"]
    launch = [s for st, s in gs.sb.calls if st == "launch worker"][-1]
    assert f"{tau3_py} -u {H.RUNNER_VM} run --jobs {g3.jobs_file}" in launch
    assert [H.score_of(x) for x in g3.poll()] == [("ok", 1.0, None)]
