"""scripts/daytona/ssb_grade_daytona.py -- the grading sandbox's driver.

NO DAYTONA, NO KEY, NO LIBRELOFFICE. The Daytona SDK is an in-memory fake whose sandboxes are temp
dirs: `process.exec` recognises each step by its `: EVOLEG=<step>` marker (the driver reuses the
leg driver's own identity and self-test methods, so their steps are played exactly as for a leg),
`fs` copies bytes. `launch` runs the REAL scripts/daytona/grade_worker.py on the uploaded job list
with a fake scorer (so the driver's worker command line and job schema are proven against the
worker, not assumed). The ledger goes to a temp file.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import threading
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "daytona"))

import daytona_lib_loop as dlx  # noqa: E402
import ssb_grade_daytona as G  # noqa: E402

WORKER = ROOT / "scripts" / "daytona" / "grade_worker.py"
FAKE_SCORER = '''
import logging
def score_and_diff(gt_file, proc_file, answer_position, max_mismatches=0, orig_input_file=None):
    logging.getLogger("fake").warning("graded %s", proc_file)
    ok = open(proc_file, "rb").read().startswith(b"GOLD")
    return ok, (2 if ok else 0), 2, "All cells match" if ok else "'S'!A1: expected=1, got=None"
'''


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class _Params:
    def __init__(self, **kw):
        self.__dict__.update(kw)


FAKE_DAYTONA = types.ModuleType("daytona")
FAKE_DAYTONA.CreateSandboxFromImageParams = _Params
FAKE_DAYTONA.Image = types.SimpleNamespace(from_dockerfile=lambda p: ("dockerfile", p))
FAKE_DAYTONA.Resources = lambda **kw: kw


class FakeSandbox:
    def __init__(self, world, params, sid):
        self.world, self.name, self.id = world, params.name, sid
        self.cpu, self.memory, self.disk = 8, 32, 40
        self.state = "started"
        self.network_block_all = (getattr(params, "network_block_all", None)
                                  if world.honour_block_all else None)
        self.domain_allow_list = getattr(params, "domain_allow_list", None)
        self.build_info = types.SimpleNamespace(snapshot_ref=world.snapshot_ref)
        self.root = world.base / sid
        self.root.mkdir(parents=True)
        self.process = types.SimpleNamespace(exec=self._exec)
        self.fs = types.SimpleNamespace(upload_file=self._up, download_file=self._down)
        self.steps = []
        self.exec_envs = []

    def refresh_data(self):
        pass

    def path(self, p):
        return self.root / str(p).lstrip("/")

    def _up(self, src, dst, timeout=None):
        p = self.path(dst)
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(src, (bytes, bytearray)):
            p.write_bytes(src)
        else:
            shutil.copyfile(src, p)

    def _down(self, src, dst):
        s = self.path(src)
        if not s.is_file():
            raise RuntimeError(f"file not found: {src}")
        shutil.copyfile(s, dst)

    def _exec(self, command, timeout=None, env=None):
        script = shlex.split(command)[2]
        step = script.split("\n", 1)[0].replace(": EVOLEG=", "").strip()
        self.steps.append(step)
        self.exec_envs.append(env)
        out, rc = self.play(step, script)
        return types.SimpleNamespace(exit_code=rc, result=out)

    def play(self, step, script):
        w = self.world
        if step == "identity":
            return ("arch=x86_64\nlibreoffice=" + G.EXPECTED["libreoffice"] +
                    "\nrunner_openpyxl=3.1.5 lxml 6.1.1 LXML True DEFUSEDXML True\n"
                    "claude_binaries=0\n"), 0
        if step == "egress":
            eg = {k: {"ok": False, "error": "URLError"} for k in
                  ("openrouter", "example_com", "pypi", "github", "google_ip")}
            if w.leak:
                eg["pypi"] = {"ok": True, "status": 200}
            return "EGRESS " + json.dumps(eg) + "\n", 0
        if step == "unpack":
            with tarfile.open(self.path("/tmp/bundle.tgz")) as tf:
                w.uploaded = tf.getnames()
                tf.extractall(self.root, filter="data")
            d = self.path("/usr/local/bin/docker")
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.path("/opt/leg/docker_shim.py"), d)
            return "/usr/local/bin/docker\n", 0
        if step == "shas":
            out = []
            for p in shlex.split(script.split("\n", 1)[1])[1:]:
                f = self.path(p)
                if f.is_file():
                    out.append(f"{sha(f)}  {p}")
            return "\n".join(out) + "\n", 0
        if step == "cgroups":
            return "root=memory cpu pids evoshim=memory cpu pids\n", 0
        if step == "selftest":
            img = shlex.split(script.split("bash -s ", 1)[1].split(" <<", 1)[0])[0]
            kv = {"DATASET": "HIDDEN", "UMOUNT": "DENIED", "NET": "NONE", "CGROUP_WRITE": "DENIED",
                  "WS_WRITE": "OK", "RUNNING": "true", "PROBE": "/usr/bin/soffice", "IMAGES": img,
                  "USERS_ENTRIES": "0", "OPT_ENTRIES": "0", "HOME_ENTRIES": "0",
                  "SHIMSTATE_ENTRIES": "0", "CGROUPS_LEFT": "0", "EXEC_RC": "0", "KEY_IN_ENV": "0"}
            return "".join(f"{k}={v}\n" for k, v in kv.items()), 0
        if step == "grade":
            w.probe = shlex.split(script.split("bash -s ", 1)[1].split(" <<", 1)[0])[1]
            return "GRADE_SELFTEST " + json.dumps({"task": w.probe, "golden_vs_golden_passed": True,
                                                   "matched": 2, "total": 2}) + "\n", 0
        if step == "launch":
            if w.worker_hangs:
                return "launched\n", 0
            self.run_worker()
            return "launched\n", 0
        if step == "poll":
            rc = self.path("/opt/grade/out/rc")
            state = rc.read_text().strip() if rc.is_file() else "RUNNING"
            res = self.path("/opt/grade/out/results.jsonl")
            n = len(res.read_text().splitlines()) if res.is_file() else 0
            log = self.path("/opt/grade/out/worker.log")
            return f"{state}\nLINES={n}\nSIZE={log.stat().st_size if log.is_file() else 0}\n", 0
        if step == "logtail":
            log = self.path("/opt/grade/out/worker.log")
            return (log.read_text() if log.is_file() else ""), 0
        if step == "kill":
            w.killed = True
            return "killed\n", 0
        if step == "pack":
            out = self.path("/tmp/grade_logs.tgz")
            with tarfile.open(out, "w:gz") as tf:
                od = self.path("/opt/grade/out")
                for f in sorted(od.rglob("*")) if od.is_dir() else []:
                    if f.is_file():
                        tf.add(str(f), arcname=str(f.relative_to(self.root)))
            return "packed\n", 0
        raise AssertionError(f"unexpected step {step!r}")

    def run_worker(self):
        """The REAL worker on the uploaded job list, with the driver's own flags, except the paths
        that only exist inside a sandbox and the scorer (a fake, so its sha pin is swapped)."""
        launch = self.path("/opt/grade/launch.sh").read_text()
        line = next(ln for ln in launch.splitlines() if ln.startswith("env -u OPENROUTER_API_KEY"))
        toks = shlex.split(line)                # env -u KEY python -u WORKER <args> > log 2>&1
        assert toks[:6] == ["env", "-u", "OPENROUTER_API_KEY", "/opt/evoskill/venv/bin/python",
                            "-u", G.SANDBOX_WORKER], toks[:6]
        argv = toks[6:]
        argv = argv[: argv.index(">")]
        swap = {"--jobs": str(self.path(G.SANDBOX_JOBS)),
                "--results": str(self.path(G.SANDBOX_OUT) / "results.jsonl"),
                "--work": str(self.root / "work"), "--repo": str(self.world.fake_repo),
                "--scorer-module": "fake_scorer",
                "--expect-scorer-sha256": sha(self.world.fake_repo / "fake_scorer.py")}
        for i, a in enumerate(argv):
            if a in swap:
                argv[i + 1] = swap[a]
        self.world.worker_argv = argv
        out = self.path(G.SANDBOX_OUT)
        out.mkdir(parents=True, exist_ok=True)
        env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
        p = subprocess.run([sys.executable, str(WORKER), *argv], capture_output=True, text=True,
                           env=env, timeout=120)
        (out / "worker.log").write_text(p.stdout + p.stderr)
        (out / "rc").write_text(f"{p.returncode}\n")


class FakeClient:
    def __init__(self, base: Path, fake_repo: Path):
        self.base, self.fake_repo = base, fake_repo
        self.live, self.deleted, self.created = {}, set(), []
        self.snapshot_ref = G.EXPECTED["snapshot_ref"]
        self.honour_block_all, self.leak, self.worker_hangs = True, False, False
        self.killed = False
        self.uploaded, self.probe, self.worker_argv = [], None, None
        self._n = 0
        self.lock = threading.Lock()

    def create(self, params, timeout=None, on_snapshot_create_logs=None):
        with self.lock:
            self._n += 1
            sid = f"{self._n:08d}-fakegrade"
        sb = FakeSandbox(self, params, sid)
        self.live[sid] = sb
        self.created.append(params)
        return sb

    def get(self, key):
        for sb in self.live.values():
            if key in (sb.id, sb.name):
                return sb
        raise RuntimeError("Sandbox not found (404)")

    def delete(self, sb, timeout=None, wait=True):
        self.live.pop(sb.id, None)
        self.deleted.add(sb.id)

    def list(self):
        return list(self.live.values())


# ══ fixtures ════════════════════════════════════════════════════════════════════════════════════
@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "daytona", FAKE_DAYTONA)
    monkeypatch.setattr(dlx, "LEDGER", str(tmp_path / "ledger.jsonl"))
    monkeypatch.setattr(G, "POLL_S", 0)
    for k in ("EVOSKILL_RECALC_TIMEOUT_SECONDS", "EVOSKILL_DATATABLE_UNO_TIMEOUT_SECONDS",
              G.data_overlay.ENV):
        monkeypatch.delenv(k, raising=False)
    rd = tmp_path / "run"
    rd.mkdir()
    write_infra(rd / "INFRA.json")
    ssb = tmp_path / "ssb_data"                      # the SpreadsheetBench v2 tree: one dataset.json per category
    for cat in G.CATEGORIES:
        (ssb / cat).mkdir(parents=True)
        (ssb / cat / "dataset.json").write_text("[]")
    monkeypatch.setattr(G, "DATA", ssb)
    data = tmp_path / "ds"
    idx = {}
    for t in ("Cat/01_01", "Cat/01_02"):
        d = data / t.replace("/", "_")
        d.mkdir(parents=True)
        (d / "golden.xlsx").write_bytes(b"GOLD-" + t.encode())
        (d / "input.xlsx").write_bytes(b"IN-" + t.encode())
        idx[t] = {"golden": d / "golden.xlsx", "input": d / "input.xlsx",
                  "answer_position": "S!A1:A2"}
    saved = tmp_path / "saved" / "Cat_01_01__s0" / "output_raw"
    saved.mkdir(parents=True)
    (saved / "input.xlsx").write_bytes(b"WRONG")
    stray = data / "Cat_01_02" / "unrelated.xlsx"
    stray.write_bytes(b"not needed by any job")
    fake_repo = tmp_path / "fakerepo"
    fake_repo.mkdir()
    (fake_repo / "fake_scorer.py").write_text(FAKE_SCORER)
    client = FakeClient(tmp_path / "sandboxes", fake_repo)
    return types.SimpleNamespace(tmp=tmp_path, run_dir=rd, idx=idx, saved=saved / "input.xlsx",
                                 stray=stray, client=client, out=tmp_path / "out",
                                 ledger=tmp_path / "ledger.jsonl")


INFRA_FIXTURE = ROOT / "tests" / "fixtures" / "ssb_leg_pool" / "infra_cc.json"


def real_infra() -> dict:
    """A valid INFRA.json of an SSB v2 run (the shipped test fixture), as a fresh dict."""
    return json.loads(INFRA_FIXTURE.read_text())


def write_infra(path: Path, inf: dict | None = None) -> None:
    """The run's INFRA.json as the operator declares it."""
    path.write_text(json.dumps(real_infra() if inf is None else inf, indent=1))


def write_jobs(w, jobs) -> Path:
    f = w.tmp / "jobs.jsonl"
    f.write_text("".join(json.dumps(j) + "\n" for j in jobs))
    return f


def base_jobs(w):
    return [{"job_id": "Cat/01_01/golden/grade", "mode": "grade", "task_id": "Cat/01_01",
             "output": str(w.idx["Cat/01_01"]["golden"])},
            {"job_id": "Cat/01_01/saved/s0/grade", "mode": "grade", "task_id": "Cat/01_01",
             "output": str(w.saved), "meta": {"sample": 0}},
            {"job_id": "Cat/01_02/input/open", "mode": "open",
             "path": str(w.idx["Cat/01_02"]["input"])}]


def cfg_for(w, jobs, **kw) -> G.GradeConfig:
    return G.GradeConfig(jobs_file=write_jobs(w, jobs), out_dir=w.out, run_dir=w.run_dir,
                         label=kw.pop("label", "t"), idx=w.idx, env={}, **kw)


def ledger_rows(w):
    return [json.loads(ln) for ln in w.ledger.read_text().splitlines()] if w.ledger.is_file() else []


# ══ configuration (no sandbox) ══════════════════════════════════════════════════════════════════
def test_the_config_takes_infra_timeouts_and_leg_knobs(world):
    c = cfg_for(world, base_jobs(world))
    r = c.record()
    assert r["knobs"]["EVOSKILL_RECALC_TIMEOUT_SECONDS"] == "900"
    assert r["knobs"]["EVOSKILL_DATATABLE_UNO_TIMEOUT_SECONDS"] == "900"
    assert r["knobs"]["EVOSKILL_HEAVY_COMPARE_CONCURRENCY"] == "2"
    assert (c.max_procs, c.max_heavy, c.job_timeout, c.ttl_minutes) == (4, 2, 5400, 150)
    assert r["egress"] == "block-all" and r["probe_task"] == "Cat/01_01" and r["n_to_run"] == 3
    env = G.grade_env_text(c.knobs, c.image_id)
    assert "export EVOSKILL_RECALC_TIMEOUT_SECONDS=900\n" in env
    assert f"export EVOSHIM_IMAGE_ID={G.EXPECTED['image_id']}\n" in env
    assert "OPENROUTER" not in env and "EVOSKILL_SOLVER_MODEL" not in env
    assert "network_block_all" not in env


def test_no_infra_or_a_clashing_env_is_refused(world):
    (world.run_dir / "INFRA.json").unlink()
    with pytest.raises(G.Refused, match="no INFRA.json"):
        cfg_for(world, base_jobs(world))
    write_infra(world.run_dir / "INFRA.json")
    with pytest.raises(G.Refused, match="disagree with INFRA"):
        G.GradeConfig(jobs_file=write_jobs(world, base_jobs(world)), out_dir=world.out,
                      run_dir=world.run_dir, label="t", idx=world.idx,
                      env={"EVOSKILL_RECALC_TIMEOUT_SECONDS": "180"})


def test_an_infra_that_states_another_image_is_refused(world):
    inf = real_infra()
    inf["leg_sandbox"]["snapshot_ref"] = "daytona-other:daytona"
    write_infra(world.run_dir / "INFRA.json", inf)
    with pytest.raises(G.Refused, match="snapshot_ref"):
        cfg_for(world, base_jobs(world))


@pytest.mark.parametrize("bad,match", [
    ({"answer_position": "S!A1:Z99"}, "differs from the dataset"),
    ({"gt_file": "/tmp/other_golden.xlsx"}, "differs from the dataset"),
    ({"task_id": "Cat/99_99"}, "not a dataset task"),
    ({"output": "/tmp/x.txt"}, "not a workbook"),
    ({"output": "/data/x/.env"}, "not a workbook"),
    ({"mode": "regrade"}, "mode"),
])
def test_a_job_that_disagrees_with_the_dataset_or_is_not_a_workbook_is_refused(world, bad, match):
    j = dict(base_jobs(world)[0], **bad)
    with pytest.raises(G.Refused, match=match):
        cfg_for(world, [j])


def test_duplicate_ids_are_refused_and_done_jobs_are_not_rerun(world):
    jobs = base_jobs(world)
    with pytest.raises(G.Refused, match="duplicate"):
        cfg_for(world, jobs + [jobs[0]])
    world.out.mkdir()
    (world.out / "results.jsonl").write_text(json.dumps({"job_id": jobs[0]["job_id"],
                                                         "status": "ok"}) + "\n")
    c = cfg_for(world, jobs)
    assert c.n_done_before == 1 and [j["job_id"] for j in c.jobs] == [j["job_id"] for j in jobs[1:]]


def test_the_bundle_holds_exactly_the_workbooks_the_jobs_need(world):
    c = cfg_for(world, base_jobs(world))
    b = world.tmp / "b.tgz"
    man = G.build_bundle(c, b)
    with tarfile.open(b) as tf:
        names = set(tf.getnames())
        jobs = [json.loads(ln) for ln in tf.extractfile("opt/grade/jobs.jsonl").read().decode()
                .splitlines()]
        env = tf.extractfile("opt/leg/leg.env").read().decode()
        launch = tf.extractfile("opt/grade/launch.sh").read().decode()
    books = {n for n in names if n.endswith(".xlsx")}
    want = {G.loop._mirror(Path(p)) for p in (
        world.idx["Cat/01_01"]["golden"], world.idx["Cat/01_01"]["input"], world.saved,
        world.idx["Cat/01_02"]["input"])}
    assert books == want and G.loop._mirror(world.stray) not in names
    assert set(man["workbooks"]) == {str(Path("/" + p)) for p in want}
    for cat in G.CATEGORIES:
        assert G.loop._mirror(G.DATA / cat / "dataset.json") in names
    assert G.loop._mirror(ROOT / "src/agent_profiles/ssbv2_agent/scorer.py") in names
    assert G.loop._mirror(ROOT / "src/agent_profiles/ssbv2_agent/docker_scripts/datatable_uno_solve.py") in names
    assert {"opt/leg/docker_shim.py", "opt/grade/grade_worker.py"} <= names
    assert not any(n.endswith("evaluation.py") for n in names), "no Stage B job, no evaluation.py"
    assert not any(G.SECRET_NAME_RE.search(Path(n).name) or "oauth" in n.lower()
                   for n in names if n != "opt/leg/leg.env")
    assert all("_heavy" not in j for j in jobs) and len(jobs) == 3
    assert "OPENROUTER" not in env and "env -u OPENROUTER_API_KEY" in launch
    assert f"--expect-scorer-sha256 {G.EXPECTED['baseline_runtime']['scorer_sha256'][1]}" in launch
    assert "--job-timeout 5400" in launch and "--max-heavy 2" in launch


def test_a_stageb_job_uploads_the_pinned_evaluation_file(world):
    if G.EVALUATION is None or not G.EVALUATION.is_file():
        pytest.skip("evaluation.py not present")
    jobs = [{"job_id": "Cat/01_01/golden/stageB", "mode": "stageB", "task_id": "Cat/01_01",
             "output": str(world.idx["Cat/01_01"]["golden"])}]
    c = cfg_for(world, jobs)
    b = world.tmp / "b.tgz"
    G.build_bundle(c, b)
    with tarfile.open(b) as tf:
        assert G.loop._mirror(G.EVALUATION) in tf.getnames()


# ══ a whole run on the fake SDK ═════════════════════════════════════════════════════════════════
def test_a_run_blocks_egress_proves_identity_grades_merges_and_deletes(world):
    c = cfg_for(world, base_jobs(world))
    run = G.GradeRun(c, world.client)
    rc = run.run()
    assert rc == G.EXIT_OK, run.summary.get("error")
    p = world.client.created[0]
    assert p.network_block_all is True and getattr(p, "domain_allow_list", None) is None
    assert p.name.startswith(G.PREFIX + "t-") and p.labels["owner"] == G.OWNER_LABEL
    assert p.ttl_minutes == 150 and p.ephemeral is True and p.auto_stop_interval == 0
    sb_id = run.sid
    assert sb_id in world.client.deleted and not world.client.live
    sb = [s for s in [run.sb] if s][0]
    assert all(e is None for e in sb.exec_envs), "no exec call may carry an environment"
    assert sb.steps[:5] == ["identity", "egress", "unpack", "shas", "cgroups"]
    assert "selftest" in sb.steps and "grade" in sb.steps and world.client.probe == "Cat/01_01"
    rows = ledger_rows(world)
    assert {r["driver"] for r in rows} == {"ssb_grade_daytona"}
    ev = [r["event"] for r in rows]
    assert ev[0] == "create_requested" and "created" in ev and "deleted" in ev
    assert ev[-1] == "grade_run_end" and rows[0]["network_block_all"] is True
    res = {json.loads(ln)["job_id"]: json.loads(ln)
           for ln in (world.out / "results.jsonl").read_text().splitlines()}
    assert set(res) == {j["job_id"] for j in base_jobs(world)}
    g, s, o = (res["Cat/01_01/golden/grade"], res["Cat/01_01/saved/s0/grade"],
               res["Cat/01_02/input/open"])
    assert g["status"] == "ok" and g["passed"] is True and g["host"]["outcome"] == "passed"
    assert s["passed"] is False and s["host"]["outcome"] == "graded_wrong"
    assert s["meta"] == {"sample": 0} and s["host"]["sandbox_id"] == sb_id
    assert o["mode"] == "open" and o["status"] == "ok"
    assert run.summary["egress"]["mode"] == "block-all" and run.summary["deleted"] is True
    assert run.summary["results_this_run"]["grade_outcomes"] == {"passed": 1, "graded_wrong": 1}
    assert run.summary["worker_meta"]["max_procs"] == 4 and run.summary["keylike_hits"] == []
    assert (run.dir / "sandbox" / "opt" / "grade" / "out" / "worker.log").is_file()
    assert "--max-procs" in world.client.worker_argv


def test_block_all_not_read_back_aborts_before_anything_is_uploaded(world):
    world.client.honour_block_all = False
    run = G.GradeRun(cfg_for(world, base_jobs(world)), world.client)
    assert run.run() == G.EXIT_INCOMPLETE
    assert "network_block_all" in run.summary["error"]
    assert run.sid in world.client.deleted and world.client.uploaded == []
    assert not (world.out / "results.jsonl").exists()


def test_an_egress_leak_aborts_and_deletes(world):
    world.client.leak = True
    run = G.GradeRun(cfg_for(world, base_jobs(world)), world.client)
    assert run.run() == G.EXIT_INCOMPLETE
    assert "egress NOT blocked" in run.summary["error"] and "pypi" in run.summary["error"]
    assert run.sid in world.client.deleted and world.client.uploaded == []


def test_the_spend_guard_kills_the_worker_keeps_results_and_deletes(world, monkeypatch):
    world.client.worker_hangs = True
    monkeypatch.setattr(G, "USD_PER_HOUR", 1e9)
    run = G.GradeRun(cfg_for(world, base_jobs(world), max_usd=0.5), world.client)
    assert run.run() == G.EXIT_INCOMPLETE
    assert "spend guard" in (run.summary.get("aborted") or run.summary.get("error"))
    assert run.sid in world.client.deleted
    assert run.summary["missing_after_run"] == 3


def test_a_rerun_uploads_only_the_jobs_without_a_result(world):
    run = G.GradeRun(cfg_for(world, base_jobs(world)[:1]), world.client)
    assert run.run() == G.EXIT_OK
    c2 = cfg_for(world, base_jobs(world))
    assert [j["job_id"] for j in c2.jobs] == [j["job_id"] for j in base_jobs(world)[1:]]
    run2 = G.GradeRun(c2, world.client)
    assert run2.run() == G.EXIT_OK
    up = [json.loads(ln) for ln in (run2.sb.root / "opt/grade/jobs.jsonl").read_text().splitlines()]
    assert [j["job_id"] for j in up] == [j["job_id"] for j in base_jobs(world)[1:]]
    assert len((world.out / "results.jsonl").read_text().splitlines()) == 3
    assert G.GradeRun(cfg_for(world, base_jobs(world)), world.client).cfg.jobs == []


def test_merge_never_rewrites_a_line_and_skips_torn_or_unknown_ones(world):
    local, dl = world.tmp / "res.jsonl", world.tmp / "dl.jsonl"
    local.write_text(json.dumps({"job_id": "a", "status": "ok", "mode": "open"}) + "\n")
    dl.write_text(json.dumps({"job_id": "a", "status": "exception", "mode": "open"}) + "\n" +
                  json.dumps({"job_id": "b", "status": "ok", "mode": "open"}) + "\n" +
                  json.dumps({"job_id": "zz", "status": "ok", "mode": "open"}) + "\n" +
                  '{"job_id": "c", "sta')
    rep = G.merge_results(local, dl, {"a", "b", "c"}, {"sandbox_id": "x"})
    assert rep == {"new": 1, "already": 1, "unknown": ["zz"], "torn": 1}
    rows = [json.loads(ln) for ln in local.read_text().splitlines()]
    assert [r["job_id"] for r in rows] == ["a", "b"] and rows[0]["status"] == "ok"


def test_host_outcomes_follow_d4_and_flag_the_open_question():
    hdr = "FAIL — 1/2 graded cells match\n(scorer's own diff, computed at scoring time; full, uncapped)\n\nx\n"
    ok = {"mode": "grade", "status": "ok", "diff_file_text": hdr, "warnings": []}
    assert G.annotate(ok, {})["outcome"] == "graded_wrong"
    failw = dict(ok, warnings=[{"message": "LibreOffice recalculation failed for /x.xlsx: boom"}])
    assert G.annotate(failw, {})["outcome"] == "grader_failed"
    raised = {"mode": "grade", "status": "exception", "warnings": []}
    assert (G.annotate(raised, {})["outcome"], G.annotate(raised, {})["outcome_reason"]) == (
        "grader_failed", "exception")
    to = {"mode": "grade", "status": "timeout"}
    assert G.annotate(to, {})["outcome_reason"] == "timeout"
    openq = dict(ok, scorer_exception_warnings=["_restore_unrecalculable_cells: could not open"])
    h = G.annotate(openq, {})
    assert h["outcome"] == "graded_wrong" and h["open_question"] is True
    ref = {"mode": "grade", "status": "refused", "refused": "output sha256 x != y"}
    assert G.annotate(ref, {})["outcome"] is None


def test_regrade_jobs_are_new_ids_for_grader_failed_grades_only(world):
    world.out.mkdir()
    jobs = G.normalize_jobs(base_jobs(world), world.idx)
    (world.out / "jobs_normalized.jsonl").write_text("".join(json.dumps(j) + "\n" for j in jobs))
    (world.out / "results.jsonl").write_text(
        json.dumps({"job_id": jobs[0]["job_id"], "mode": "grade", "task_id": "Cat/01_01",
                    "host": {"outcome": "grader_failed"}}) + "\n" +
        json.dumps({"job_id": jobs[1]["job_id"], "mode": "grade", "task_id": "Cat/01_01",
                    "host": {"outcome": "graded_wrong"}}) + "\n")
    out = world.tmp / "regrade.jsonl"
    rc = G.main(["regrade-jobs", "--results", str(world.out / "results.jsonl"), "--out", str(out)])
    assert rc == 0
    rows = [json.loads(ln) for ln in out.read_text().splitlines()]
    assert rows == [{"job_id": jobs[0]["job_id"] + "#regrade1", "mode": "grade",
                     "task_id": "Cat/01_01", "output": jobs[0]["output"],
                     "meta": {"regrade_of": jobs[0]["job_id"]}}]
    assert G.normalize_jobs(rows, world.idx)[0]["output_sha256"] == jobs[0]["output_sha256"]


def test_cleanup_lists_only_grade_sandboxes_never_verified_deleted(world):
    rows = [{"driver": "ssb_grade_daytona", "event": "created", "id": "g1", "name": G.PREFIX + "a"},
            {"driver": "ssb_grade_daytona", "event": "deleted", "id": "g1"},
            {"driver": "ssb_grade_daytona", "event": "created", "id": "g2", "name": G.PREFIX + "b"},
            {"driver": "ssb_leg_daytona_loop", "event": "created", "id": "leg1", "name": "x"}]
    world.ledger.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert G.ledger_open_sandboxes(world.ledger) == [{"id": "g2", "name": G.PREFIX + "b"}]


def test_delete_refuses_a_sandbox_that_is_not_ours(world):
    other = types.SimpleNamespace(id="00000009-x", name="evoskill-loop-v1tr-s01-1",
                                  state="started")
    world.client.live[other.id] = other
    assert G.delete_by_id(world.client, other.id, "t") is False and other.id in world.client.live


def test_check_only_creates_nothing(world, capsys, monkeypatch):
    monkeypatch.setattr(G.base, "load_index", lambda: world.idx)
    f = write_jobs(world, base_jobs(world))
    rc = G.main(["run", "--jobs", str(f), "--out-dir", str(world.out), "--run-dir",
                 str(world.run_dir), "--label", "t", "--check-only"])
    assert rc == G.EXIT_CHECK_ONLY and "CHECK-ONLY: 3 job(s)" in capsys.readouterr().out
    assert world.client.created == [] and not world.ledger.exists()


# ══ make-jobs --key: the split is the run's ══════════════════════════════════════════════════════
def test_make_jobs_key_checks_against_the_runs_declared_split(tmp_path, capsys, monkeypatch):
    new = ROOT / G.loop.PROJECT_SPLIT[0]
    assert G.DEFAULT_SPLIT == new
    rd = tmp_path / "run"
    rd.mkdir()
    out = tmp_path / "jobs.jsonl"
    dropped = "Template/16_09"                   # dropped on the project split; train on the other split below
    assert dropped in json.loads(new.read_text())["dropped"]["task_ids"]
    other = json.loads(new.read_text())
    other["train"] = other["train"] + [dropped]
    old = tmp_path / "other_split.json"
    old.write_text(json.dumps(other))
    book = tmp_path / "book.xlsx"
    book.write_bytes(b"PK")
    monkeypatch.setattr(G.base, "load_index",
                        lambda: {dropped: {"golden": book, "input": book, "answer_position": "A1"}})
    argv = ["make-jobs", "--tasks", dropped, "--kinds", "open", "--key", "train", "--out", str(out),
            "--run-dir", str(rd)]
    # no SPEC.json: the project split, where the task is dropped
    assert G.main(argv) == G.EXIT_REFUSED and "not in split[train]" in capsys.readouterr().err
    # the run declares the other split: accepted there
    (rd / "SPEC.json").write_text(json.dumps({"spec": {"split_path": str(old),
                                                        "split_sha256": sha(old)}}))
    assert G.main(argv) == G.EXIT_OK and out.is_file()
    # the run declares the project split: refused again; an explicit --split still wins
    (rd / "SPEC.json").write_text(json.dumps({"spec": {"split_path": str(new),
                                                        "split_sha256": G.loop.PROJECT_SPLIT[1]}}))
    assert G.main(argv) == G.EXIT_REFUSED
    assert G.main(argv + ["--split", str(old)]) == G.EXIT_OK
