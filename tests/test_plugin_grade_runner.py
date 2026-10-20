"""scripts/daytona/plugin_grade_runner.py -- the run's frozen grade(task, snapshot), one child per job.

OFFLINE: real subprocesses under this interpreter; no machine, no LibreOffice, no model. Every test writes its own
toy plug-in dir: a COPY of plugin_api v3 written here (never the canonical src/orchestrator/plugin_api.py), a
list_tasks-only benchmark_plugin.py, a toy benchmark_grader.py whose grade() follows task.grading["mode"], and
tasks.json. Snapshot tgz files are built the way evoctl pack builds out.tgz (evo/work + evo/out + home/evo/.claude,
no symlink followed).

What must hold: a number from 0 to 1 is the score and anything else grade() does is grader_failed, never a number;
the kept tgz is never changed nor handed over (a #regrade1 sees the original bytes); the extraction is
cc_solver.Box.collect's rule and creates no link; the loop's own faults are refused before any child; RESUME, the
signals and the caps are grade_worker's; score_of is the one decision on a line; the runner never imports
benchmark_plugin.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import io
import json
import os
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "daytona" / "plugin_grade_runner.py"
GRADE_WORKER = ROOT / "scripts" / "daytona" / "grade_worker.py"
CC_SOLVER = ROOT / "scripts" / "cc_solver" / "cc_solver.py"
PLUGIN_API = ROOT / "src" / "orchestrator" / "plugin_api.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load(RUNNER, "plugin_grade_runner_under_test")
GF = "grader_failed"

TOY_API = '''"""The loop's side of the benchmark plug-in interface, version 3 (a test copy)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

API_VERSION = 3
SPLITS = ("train", "validation", "test")
GRADER_FAILED = "grader_failed"


@dataclass(frozen=True)
class Task:
    id: str
    category: str
    instruction: str
    inputs: tuple[Path, ...]
    answer_files: tuple[Path, ...]
    grading: Mapping[str, str]
    split: str | None
'''

TOY_PLUGIN = '''from pathlib import Path


def list_tasks(root: Path) -> list:
    return []
'''

TOY_GRADER = r'''"""A toy grader: what grade() does follows task.grading["mode"]."""
import ast
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

TRACE = os.environ.get("TOY_TRACE")


def _trace(suffix, obj):
    if TRACE:
        Path(TRACE, Path.cwd().name + suffix).write_text(json.dumps(obj))


def grade(task, snapshot):
    g = task.grading
    mode = g["mode"]
    _trace(".start.json", {"pid": os.getpid(), "start": time.time()})
    if mode == "return":
        out = float("nan") if g["value"] == "nan" else ast.literal_eval(g["value"])
    elif mode == "raise":
        raise ValueError("toy says no")
    elif mode == "sysexit":
        raise SystemExit(4)
    elif mode == "exit":
        os._exit(9)
    elif mode == "warn":
        logging.getLogger("toy.scorer").warning("marker %s", Path.cwd().name)
        out = 0.0
    elif mode == "sleep":
        if not os.environ.get("TOY_NO_SLEEP"):
            if g.get("grandchild"):
                _trace(".grandchild.json", {"pid": subprocess.Popen(["sleep", "300"]).pid})
            time.sleep(float(g["seconds"]))
        out = 1.0
    elif mode == "inspect":
        import plugin_api
        snapshot = Path(snapshot)
        found = []
        for d, dirs, files in os.walk(snapshot):
            for n in dirs + files:
                p = Path(d, n)
                kind = "link" if p.is_symlink() else "dir" if p.is_dir() else "file"
                found.append([str(p.relative_to(snapshot)), kind])
        seen = (snapshot / "output" / "out.xlsx").read_text()
        _trace(".inspect.json", {
            "task_is_staged": type(task) is plugin_api.Task, "plugin_api_file": plugin_api.__file__,
            "inputs": [type(task.inputs).__name__] + [type(p).__name__ for p in task.inputs],
            "answer_files": [type(task.answer_files).__name__] + [type(p).__name__ for p in task.answer_files],
            "id": task.id, "split": task.split, "grading": dict(task.grading), "snapshot": str(snapshot),
            "cwd": os.getcwd(), "env": sorted(os.environ),
            "benchmark_plugin_imported": "benchmark_plugin" in sys.modules, "found": sorted(found), "seen": seen})
        if g.get("mutate"):
            shutil.rmtree(snapshot)
            (snapshot / "output").mkdir(parents=True)
            (snapshot / "output" / "out.xlsx").write_text("WRECKED")
        out = 1.0 if seen == "ORIGINAL" else 0.0
    else:
        raise KeyError(mode)
    _trace(".end.json", {"end": time.time()})
    return out
'''


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture
def box(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "in.xlsx").write_bytes(b"IN")
    (data / "gold.xlsx").write_bytes(b"GOLD")
    trace = tmp_path / "trace"
    trace.mkdir()
    return {"tmp": tmp_path, "data": data, "trace": trace, "results": tmp_path / "grade" / "results.jsonl",
            "work": tmp_path / "work"}


def row(box, tid, *, answer=None, **grading) -> dict:
    return {"id": tid, "category": tid.split("/")[0], "instruction": f"do {tid}",
            "inputs": [str(box["data"] / "in.xlsx")], "answer_files": [str(answer or box["data"] / "gold.xlsx")],
            "grading": {k: str(v) for k, v in grading.items()}, "split": "train"}


def plugin_dir(box, rows, *, grader=TOY_GRADER, name="plugin") -> Path:
    d = box["tmp"] / name
    d.mkdir()
    (d / "plugin_api.py").write_text(TOY_API)
    (d / "benchmark_plugin.py").write_text(TOY_PLUGIN)
    (d / "benchmark_grader.py").write_text(grader)
    (d / "tasks.json").write_text(json.dumps(rows, sort_keys=True, indent=1) + "\n")
    return d


def expect(d) -> list:
    out = []
    for p in sorted(Path(d).iterdir()):
        out += ["--expect", f"{p.name}={sha(p)}"]
    return out


def _root(ti):
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = "root"
    return ti


def snapshot_tgz(box, name, work=None, *, crafted=(), hardlink=False) -> Path:
    """An out.tgz the way evoctl pack writes it: /evo/work, /evo/out and /home/evo/.claude read with no symlink
    followed (the skill leg's /evo/work/.claude/skills -> /evo/skills link is stored as a link)."""
    src = box["tmp"] / "containers" / name
    files = {"output/out.xlsx": b"ORIGINAL", "spreadsheets/in.xlsx": b"IN"} if work is None else work
    for rel, data in files.items():
        p = src / "evo" / "work" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    (src / "evo/work/.claude").mkdir(parents=True, exist_ok=True)
    os.symlink("/evo/skills", src / "evo/work/.claude/skills")
    if hardlink:            # sorted after output/, so tarfile stores zz_hard as the LNKTYPE member
        os.link(src / "evo/work/output/out.xlsx", src / "evo/work/zz_hard")
    (src / "evo/out").mkdir(parents=True)
    (src / "evo/out/stream.jsonl").write_text("{}\n")
    (src / "home/evo/.claude").mkdir(parents=True)
    (src / "home/evo/.claude/settings.json").write_text("{}")
    tgz = box["tmp"] / "tgz" / name / "out.tgz"
    tgz.parent.mkdir(parents=True)
    with tarfile.open(tgz, "w:gz", format=tarfile.PAX_FORMAT, dereference=False) as tf:
        for part in ("evo/work", "evo/out", "home/evo/.claude"):
            tf.add(src / part, arcname=part, recursive=True, filter=_root)
        for ti, data in crafted:
            tf.addfile(ti, None if data is None else io.BytesIO(data))
    return tgz


def job(jid, tid, tgz, **over) -> dict:
    j = {"job_id": jid, "task_id": tid, "snapshot_tgz": str(tgz),
         "snapshot_sha256": over.pop("snapshot_sha256", None) or sha(tgz), "snapshot_root": "evo/work",
         "meta": {"unit": jid, "tag": "t_train"}}
    j.update(over)
    return j


def runner_cmd(box, d, jobs_file, *extra, pins=None) -> list:
    return [sys.executable, str(RUNNER), "run", "--jobs", str(jobs_file), "--results", str(box["results"]),
            "--work", str(box["work"]), "--plugin-dir", str(d), *(expect(d) if pins is None else pins), *extra]


def runner_env(box, extra=None) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
    env["TOY_TRACE"] = str(box["trace"])
    env.update(extra or {})
    return env


def read_results(box):
    lines = []
    if box["results"].is_file():
        lines = [json.loads(ln) for ln in box["results"].read_text().splitlines() if ln.strip()]
    mf = box["results"].with_name("worker_meta.json")
    return {r["job_id"]: r for r in lines}, (json.loads(mf.read_text()) if mf.is_file() else None)


def run_runner(box, d, jobs, *extra, env_extra=None, pins=None, timeout=120):
    if isinstance(jobs, Path):
        jf = jobs
    else:
        jf = box["tmp"] / f"jobs_{time.time_ns()}.jsonl"
        jf.write_text("".join(json.dumps(j) + "\n" for j in jobs))
    p = subprocess.run(runner_cmd(box, d, jf, *extra, pins=pins), capture_output=True, text=True,
                       env=runner_env(box, env_extra), timeout=timeout)
    return (p, *read_results(box))


def jd(box, jid) -> Path:
    return box["work"] / R.gw.safe_name(jid)


def trace(box, jid, suffix) -> dict:
    return json.loads((box["trace"] / (R.gw.safe_name(jid) + suffix)).read_text())


def wait_for(cond, seconds: float) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.1)
    return cond()


def gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


LINE_KEYS = {"job_id", "task_id", "worker", "status", "score", "reason", "log_records", "snapshot_tgz",
             "snapshot_sha256", "snapshot_files", "snapshot_skipped", "plugin", "grade_s", "child_pid", "child_rc",
             "started_utc", "ended_utc", "seconds", "heavy", "job_timeout_s", "meta"}


# ─────────────────────────────────────────────────────────────── (a) scores
def test_a_number_from_0_to_1_is_the_score_and_anything_else_is_grader_failed(box):
    cases = {"quarter": ("0.25", "ok", 0.25), "int1": ("1", "ok", 1.0), "true": ("True", "ok", 1.0),
             "zero": ("0", "ok", 0.0), "big": ("1.5", GF, None), "neg": ("-0.1", GF, None),
             "nan": ("nan", GF, None), "none": ("None", GF, None), "str": ("'x'", GF, None)}
    rows = [row(box, f"C/{k}", mode="return", value=v) for k, (v, _st, _sc) in cases.items()]
    rows += [row(box, "C/raise", mode="raise"), row(box, "C/sysexit", mode="sysexit"), row(box, "C/warn", mode="warn")]
    d = plugin_dir(box, rows)
    tgz = snapshot_tgz(box, "u")
    jobs = [job(f"C_{r['id'][2:]}__s0", r["id"], tgz) for r in rows]
    p, res, meta = run_runner(box, d, jobs)
    assert p.returncode == 0, p.stderr
    assert len(res) == len(jobs) and meta["statuses"] == {"ok": 5, GF: 7}
    pins = {p.name: sha(p) for p in d.iterdir()}
    for k, (v, st, sc) in cases.items():
        r = res[f"C_{k}__s0"]
        assert (r["status"], r["score"]) == (st, sc), (k, r)
        assert r["worker"] == "plugin_grade/1" and r["plugin"] == pins and r["snapshot_sha256"] == sha(tgz)
        assert r["snapshot_files"] == 2 and r["snapshot_skipped"] == ["evo/work/.claude/skills"]
        if st == "ok":
            assert set(r) == LINE_KEYS and r["reason"] is None and isinstance(r["score"], float)
        else:
            assert set(r) == LINE_KEYS | {"log_tail"}
            assert r["reason"].startswith("grade() returned ") and r["reason"].endswith("(not a number from 0 to 1)")
    assert res["C_none__s0"]["reason"] == "grade() returned None (not a number from 0 to 1)"
    assert res["C_str__s0"]["reason"] == "grade() returned 'x' (not a number from 0 to 1)"
    r = res["C_raise__s0"]
    assert (r["status"], r["score"], r["reason"]) == (GF, None, "grade() raised ValueError: toy says no")
    assert r["exception"]["type"] == "ValueError" and r["exception"]["phase"] == "grade"
    assert "toy says no" in r["exception"]["traceback"]
    r = res["C_sysexit__s0"]
    assert (r["status"], r["reason"]) == (GF, "grade() raised SystemExit: 4")
    r = res["C_warn__s0"]
    assert (r["status"], r["score"]) == ("ok", 0.0)
    assert r["log_records"] == [f"WARNING toy.scorer: marker {R.gw.safe_name('C_warn__s0')}"]


def test_a_grader_that_cannot_be_imported_is_grader_failed(box):
    d = plugin_dir(box, [row(box, "C/01", mode="return", value="1")],
                   grader="import no_such_module_for_the_toy_grader  # noqa: F401\n" + TOY_GRADER)
    p, res, _ = run_runner(box, d, [job("C_01__s0", "C/01", snapshot_tgz(box, "u"))])
    assert p.returncode == 0, p.stderr
    r = res["C_01__s0"]
    assert (r["status"], r["score"]) == (GF, None)
    assert r["reason"].startswith("grade() raised ModuleNotFoundError: ")
    assert r["exception"]["type"] == "ModuleNotFoundError" and r["exception"]["phase"] == "import"
    assert r["grade_s"] is None


def test_a_timeout_is_grader_failed_and_the_whole_process_group_is_killed(box):
    d = plugin_dir(box, [row(box, "C/01", mode="sleep", seconds=60, grandchild=1)])
    t = time.time()
    p, res, _ = run_runner(box, d, [job("C_01__s0", "C/01", snapshot_tgz(box, "u"))], "--job-timeout", "3")
    assert p.returncode == 0, p.stderr
    assert time.time() - t < 40
    r = res["C_01__s0"]
    assert (r["status"], r["score"], r["reason"]) == (GF, None, "timeout after 3 s (process group killed)")
    assert r["job_timeout_s"] == 3 and "log_tail" in r
    pids = [trace(box, "C_01__s0", ".start.json")["pid"], trace(box, "C_01__s0", ".grandchild.json")["pid"]]
    assert pids[0] == r["child_pid"]
    assert wait_for(lambda: all(gone(x) for x in pids), 10), "a process of the job outlived its timeout"


def test_a_child_that_dies_without_a_result_is_grader_failed(box):
    d = plugin_dir(box, [row(box, "C/01", mode="exit")])
    p, res, _ = run_runner(box, d, [job("C_01__s0", "C/01", snapshot_tgz(box, "u"))])
    assert p.returncode == 0, p.stderr
    r = res["C_01__s0"]
    assert (r["status"], r["score"], r["reason"]) == (GF, None, "crashed (child rc 9, no result)")
    assert r["child_rc"] == 9 and "log_tail" in r


# ─────────────────────────────────────────────────────────────── (b) isolation
def test_grade_may_wreck_its_snapshot_the_kept_tgz_never_changes_and_a_regrade_sees_the_original(box):
    d = plugin_dir(box, [row(box, "C/01", mode="inspect", mutate=1)])
    tgz = snapshot_tgz(box, "u")
    before = sha(tgz)
    jobs = [job("C_01__s0", "C/01", tgz),
            job("C_01__s0#regrade1", "C/01", tgz, meta={"unit": "C/01#s1", "regrade_of": "C_01__s0"})]
    p, res, _ = run_runner(box, d, jobs, "--max-procs", "1", "--keep-copies")
    assert p.returncode == 0, p.stderr
    assert sha(tgz) == before
    for jid in ("C_01__s0", "C_01__s0#regrade1"):
        r = res[jid]
        assert (r["status"], r["score"]) == ("ok", 1.0), r
        assert trace(box, jid, ".inspect.json")["seen"] == "ORIGINAL"
        assert (jd(box, jid) / "tree/evo/work/output/out.xlsx").read_text() == "WRECKED"
    assert res["C_01__s0#regrade1"]["meta"] == {"unit": "C/01#s1", "regrade_of": "C_01__s0"}


def test_the_child_gets_the_staged_task_its_own_job_dir_and_no_secret(box):
    rw = row(box, "C/01", mode="inspect")
    d = plugin_dir(box, [rw])
    pins = expect(d)
    tgz = snapshot_tgz(box, "u")
    p, res, _ = run_runner(box, d, [job("C_01__s0", "C/01", tgz)], "--keep-copies",
                           env_extra={"SOME_API_KEY": "sekrit-9f3", "GH_TOKEN": "sekrit-9f3", "EVOSKILL_X": "1"})
    assert p.returncode == 0, p.stderr
    assert res["C_01__s0"]["status"] == "ok"
    t = trace(box, "C_01__s0", ".inspect.json")
    j = jd(box, "C_01__s0").resolve()
    assert t["task_is_staged"] is True and Path(t["plugin_api_file"]).resolve() == j / "stage" / "plugin_api.py"
    assert t["inputs"] == ["tuple", "PosixPath"] and t["answer_files"] == ["tuple", "PosixPath"]
    assert (t["id"], t["split"], t["grading"]) == ("C/01", "train", {"mode": "inspect"})
    assert Path(t["cwd"]).resolve() == j and Path(t["snapshot"]).resolve() == j / "tree" / "evo" / "work"
    assert t["benchmark_plugin_imported"] is False
    assert "EVOSKILL_X" in t["env"] and not any(R.gw.SECRET_ENV_RE.search(k) for k in t["env"])
    assert sorted(x.name for x in (j / "stage").iterdir()) == ["benchmark_grader.py", "benchmark_plugin.py",
                                                               "plugin_api.py"]      # -B: no __pycache__ either
    jt = (j / "job.json").read_text()
    assert str(tgz) not in jt and set(json.loads(jt)) == {"job_id", "task", "snapshot"}
    assert json.loads(jt)["task"] == rw
    assert expect(d) == pins                            # the pinned files were never touched


# ─────────────────────────────────────────────────────────────── (c) extraction
def test_the_snapshot_is_extracted_with_box_collects_rule_and_no_link_is_ever_created(box):
    crafted = []
    for name in ("/evo/work/abs.txt", "evo/work/../escape.txt", "../x", "/abs"):
        ti = tarfile.TarInfo(name)
        ti.size = 3
        crafted.append((ti, b"bad"))
    fifo = tarfile.TarInfo("evo/work/fifo")
    fifo.type = tarfile.FIFOTYPE
    crafted.append((fifo, None))
    tgz = snapshot_tgz(box, "x", crafted=crafted, hardlink=True)
    d = plugin_dir(box, [row(box, "C/01", mode="inspect")])
    p, res, _ = run_runner(box, d, [job("C_01__s0", "C/01", tgz)], "--keep-copies")
    assert p.returncode == 0, p.stderr
    r = res["C_01__s0"]
    assert (r["status"], r["score"]) == ("ok", 1.0), r
    assert sorted(r["snapshot_skipped"]) == sorted(["evo/work/.claude/skills", "evo/work/zz_hard", "/evo/work/abs.txt",
                                                    "evo/work/../escape.txt", "evo/work/fifo"])
    assert r["snapshot_files"] == 2
    tree = jd(box, "C_01__s0") / "tree"
    assert not [x for x in tree.rglob("*") if x.is_symlink()]
    on_disk = sorted(str(x.relative_to(tree)) for x in tree.rglob("*"))
    with tarfile.open(tgz, "r:gz") as tf:               # cc_solver.Box.collect's rule on the whole tgz ...
        kept = [m.name for m in tf.getmembers()
                if not (m.name.startswith("/") or ".." in Path(m.name).parts or not (m.isfile() or m.isdir()))]
    under = [n for n in kept if n == "evo/work" or n.startswith("evo/work/")]    # ... restricted to evo/work
    assert on_disk == sorted(["evo"] + under)
    found = trace(box, "C_01__s0", ".inspect.json")["found"]
    assert found == [[".claude", "dir"], ["output", "dir"], ["output/out.xlsx", "file"], ["spreadsheets", "dir"],
                     ["spreadsheets/in.xlsx", "file"]]


def test_the_extraction_rule_is_cc_solvers():
    assert ('mbr.name.startswith("/") or ".." in Path(mbr.name).parts or not (mbr.isfile() or mbr.isdir())'
            in CC_SOLVER.read_text())
    assert ('m.name.startswith("/") or ".." in PurePosixPath(m.name).parts or not (m.isfile() or m.isdir())'
            in inspect.getsource(R.skipped_member))


# ─────────────────────────────────────────────────────────────── (d) refusals
def test_the_loops_own_faults_are_refused_before_any_child(box):
    d = plugin_dir(box, [row(box, "C/01", mode="inspect")])
    good = snapshot_tgz(box, "g")
    corrupt = box["tmp"] / "corrupt.tgz"
    corrupt.write_bytes(b"not a gzip stream at all")
    jobs = [job("C_01__s0", "C/01", good, snapshot_sha256="0" * 64), job("C_01__s1", "C/99", good),
            job("C_01__s2", "C/01", corrupt),
            job("C_01__s3", "C/01", box["tmp"] / "none.tgz", snapshot_sha256="1" * 64)]
    p, res, meta = run_runner(box, d, jobs)
    assert p.returncode == 0, p.stderr
    assert meta["statuses"] == {"refused": 4}
    for jid, why in (("C_01__s0", "snapshot sha256 "), ("C_01__s1", "task 'C/99' is not in tasks.json"),
                     ("C_01__s2", "extraction failed: ReadError"), ("C_01__s3", "unreadable")):
        r = res[jid]
        assert (r["status"], r["score"], r["child_pid"]) == ("refused", None, None), r
        assert why in r["reason"], (jid, r["reason"])
        assert R.score_of(r) == ("refused", None, r["reason"])
    assert not list(box["trace"].iterdir()), "grade() must not have been called"


@pytest.mark.parametrize("bad", ["model_key", "extra_file", "sha", "missing"])
def test_the_runner_refuses_to_start_and_writes_no_line(box, bad):
    d = plugin_dir(box, [row(box, "C/01", mode="return", value="1")])
    pins, env = expect(d), None
    if bad == "model_key":
        env = {"OPENROUTER_API_KEY": "sk-or-not-a-key"}
    elif bad == "extra_file":
        (d / "notes.txt").write_text("x")
    elif bad == "sha":
        (d / "benchmark_grader.py").write_text(TOY_GRADER + "\n# edited after the pin\n")
    else:
        (d / "benchmark_grader.py").unlink()
    p, res, meta = run_runner(box, d, [job("C_01__s0", "C/01", snapshot_tgz(box, "u"))], env_extra=env, pins=pins)
    assert p.returncode == 3 and "REFUSED" in p.stderr and not res and meta is None


@pytest.mark.parametrize("bad", ["dup", "relative", "sha", "root_abs", "root_dotdot", "meta", "missing", "json"])
def test_a_bad_job_list_is_refused_whole(box, bad):
    d = plugin_dir(box, [row(box, "C/01", mode="return", value="1")])
    j = job("C_01__s0", "C/01", snapshot_tgz(box, "u"))
    jobs = {"dup": [j, dict(j)], "relative": [dict(j, snapshot_tgz="units/out.tgz")],
            "sha": [dict(j, snapshot_sha256="xyz")], "root_abs": [dict(j, snapshot_root="/evo/work")],
            "root_dotdot": [dict(j, snapshot_root="evo/../..")], "meta": [dict(j, meta="m")],
            "missing": [{k: v for k, v in j.items() if k != "task_id"}], "json": None}[bad]
    jf = box["tmp"] / "jobs.jsonl"
    jf.write_text("not json\n" if jobs is None else "".join(json.dumps(x) + "\n" for x in jobs))
    p, res, meta = run_runner(box, d, jf)
    assert p.returncode == 2 and "REFUSED" in p.stderr and not res and meta is None


# ─────────────────────────────────────────────────────────────── (e) RESUME and signals
def test_a_rerun_skips_done_jobs_and_leaves_their_lines_alone(box):
    d = plugin_dir(box, [row(box, f"C/0{i}", mode="return", value="1") for i in range(4)])
    tgz = snapshot_tgz(box, "u")
    first = [job(f"C_0{i}__s0", f"C/0{i}", tgz) for i in range(2)]
    p, res, meta = run_runner(box, d, first)
    assert p.returncode == 0 and set(res) == {"C_00__s0", "C_01__s0"} and meta["n_todo"] == 2
    before = box["results"].read_text().splitlines()
    for f in box["trace"].iterdir():
        f.unlink()
    p, res, meta = run_runner(box, d, first + [job(f"C_0{i}__s0", f"C/0{i}", tgz) for i in (2, 3)])
    assert p.returncode == 0, p.stderr
    assert meta["n_skipped_done"] == 2 and meta["n_todo"] == 2
    assert box["results"].read_text().splitlines()[:2] == before
    assert sorted(f.name for f in box["trace"].glob("*.start.json")) == sorted(
        R.gw.safe_name(x) + ".start.json" for x in ("C_02__s0", "C_03__s0"))


def test_a_torn_last_line_is_moved_aside_and_its_job_graded_again(box):
    d = plugin_dir(box, [row(box, "C/01", mode="return", value="1")])
    tgz = snapshot_tgz(box, "u")
    a, b = job("C_01__s0", "C/01", tgz), job("C_01__s1", "C/01", tgz)
    run_runner(box, d, [a])
    with open(box["results"], "a") as fh:
        fh.write('{"job_id": "C_01__s1", "status": "o')        # the machine died mid-write
    p, res, meta = run_runner(box, d, [a, b])
    assert p.returncode == 0, p.stderr
    assert meta["torn_line_moved_to"] and Path(meta["torn_line_moved_to"]).read_text().startswith(
        '{"job_id": "C_01__s1"')
    assert res["C_01__s1"]["status"] == "ok"
    for ln in box["results"].read_text().splitlines():
        json.loads(ln)                                          # every line parses again


def test_sigterm_kills_the_running_children_writes_no_line_and_a_relaunch_grades_them(box):
    d = plugin_dir(box, [row(box, f"C/0{i}", mode="sleep", seconds=60, grandchild=1) for i in range(3)])
    tgz = snapshot_tgz(box, "u")
    jf = box["tmp"] / "jobs.jsonl"
    jf.write_text("".join(json.dumps(job(f"C_0{i}__s0", f"C/0{i}", tgz)) + "\n" for i in range(2)))
    proc = subprocess.Popen(runner_cmd(box, d, jf), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env=runner_env(box))
    try:
        assert wait_for(lambda: len(list(box["trace"].glob("*.grandchild.json"))) == 2, 30), "children never started"
        pids = [json.loads(f.read_text())["pid"] for f in box["trace"].glob("*.json")]
        proc.send_signal(signal.SIGTERM)
        _out, err = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            proc.kill()
    assert proc.returncode == 143, err
    res, meta = read_results(box)
    assert res == {} and meta["interrupted"] is True
    assert len(pids) == 4 and wait_for(lambda: all(gone(x) for x in pids), 10), "a child outlived the SIGTERM"
    with open(jf, "a") as fh:                                   # a job appended after the exit
        fh.write(json.dumps(job("C_02__s0", "C/02", tgz)) + "\n")
    p = subprocess.run(runner_cmd(box, d, jf), capture_output=True, text=True,
                       env=runner_env(box, {"TOY_NO_SLEEP": "1"}), timeout=120)
    assert p.returncode == 0, p.stderr
    res, meta = read_results(box)
    assert sorted(res) == ["C_00__s0", "C_01__s0", "C_02__s0"] and {r["status"] for r in res.values()} == {"ok"}
    assert (meta["n_skipped_done"], meta["n_todo"]) == (0, 3)


# ─────────────────────────────────────────────────────────────── (f) caps
def _span(box, jid) -> tuple:
    return trace(box, jid, ".start.json")["start"], trace(box, jid, ".end.json")["end"]


def test_max_procs_caps_the_concurrent_children(box):
    d = plugin_dir(box, [row(box, "C/01", mode="sleep", seconds=1)])
    tgz = snapshot_tgz(box, "u")
    jobs = [job(f"C_01__s{i}", "C/01", tgz) for i in range(4)]
    p, res, _ = run_runner(box, d, jobs, "--max-procs", "2")
    assert p.returncode == 0, p.stderr
    spans = [_span(box, j["job_id"]) for j in jobs]
    peak = max(sum(1 for s, e in spans if s <= t < e) for t, _e in spans)
    assert peak == 2


def test_the_heavy_cap_holds_while_light_jobs_use_the_other_slots(box):
    big = box["data"] / "big_gold.xlsx"
    big.write_bytes(b"G" * 400)
    d = plugin_dir(box, [row(box, "H/01", answer=big, mode="sleep", seconds=0.5),
                         row(box, "L/01", mode="sleep", seconds=0.5)])
    small = snapshot_tgz(box, "small")
    large = snapshot_tgz(box, "large", work={"output/out.xlsx": b"x" * 400})
    heavy = [job("H_01__s0", "H/01", small), job("H_01__s1", "H/01", small),    # by the task's answer file
             job("L_01__big", "L/01", large)]                                   # by the snapshot's largest file
    light = [job(f"L_01__s{i}", "L/01", small) for i in range(3)]
    p, res, meta = run_runner(box, d, heavy + light, "--max-procs", "4", "--max-heavy", "1", "--heavy-bytes", "300")
    assert p.returncode == 0, p.stderr
    assert meta["n_heavy_todo"] == 3 and meta["max_heavy_concurrent_seen"] == 1
    assert all(res[j["job_id"]]["heavy"] for j in heavy) and not any(res[j["job_id"]]["heavy"] for j in light)
    hs = sorted(_span(box, j["job_id"]) for j in heavy)
    for (_s1, e1), (s2, _e2) in zip(hs, hs[1:]):
        assert s2 >= e1 - 0.05, "two heavy grades overlapped"
    assert any(_span(box, j["job_id"])[0] < hs[0][1] for j in light), "light jobs should run beside a heavy one"


# ─────────────────────────────────────────────────────────────── (g) decisions, meta, pins
def test_score_of_is_the_one_decision_on_a_line():
    assert R.score_of({"status": "ok", "score": 0.25}) == ("ok", 0.25, None)
    assert R.score_of({"status": "ok", "score": 1}) == ("ok", 1.0, None)
    assert R.score_of({"status": GF, "score": None, "reason": "grade() raised X: y"}) == (GF, None,
                                                                                         "grade() raised X: y")
    assert R.score_of({"status": "refused", "reason": "snapshot sha256 a != b"}) == ("refused", None,
                                                                                    "snapshot sha256 a != b")
    for bad in (None, [], "x", {}, {"status": "ok"}, {"status": "ok", "score": 1.5}, {"status": "ok", "score": "1"},
                {"status": "ok", "score": float("nan")}, {"status": "exception"}, {"status": "timeout"},
                {"status": "ok", "passed": True}):                              # a grade_worker/1 line is not one
        st, sc, why = R.score_of(bad)
        assert (st, sc) == ("refused", None) and why, bad
    assert R.check_score(True) == 1.0 and R.check_score(10 ** 400) is None and R.check_score(-0.0) == 0.0


def test_validate_job_takes_the_six_key_line():
    good = {"job_id": "Cat_01__s0#a1#regrade1", "task_id": "Cat/01",
            "snapshot_tgz": "/opt/cc/run/tag/units/Cat_01__s0#a1/out.tgz", "snapshot_sha256": "a" * 64,
            "snapshot_root": "evo/work", "meta": {"unit": "Cat/01#s1"}}
    assert R.validate_job(dict(good)) == good
    for over in ({"job_id": "bad id"}, {"task_id": ""}, {"snapshot_tgz": "rel/out.tgz"}, {"snapshot_sha256": "A" * 64},
                 {"snapshot_sha256": "a" * 63}, {"snapshot_root": "/evo/work"}, {"snapshot_root": "evo/../work"},
                 {"snapshot_root": "."}, {"meta": None}, {"meta": []}):
        with pytest.raises(R.gw.JobError):
            R.validate_job({**good, **over})
    with pytest.raises(R.gw.JobError):
        R.validate_job({k: v for k, v in good.items() if k != "meta"})


def test_worker_meta_records_the_graders_setup_and_no_secret(box):
    d = plugin_dir(box, [row(box, "C/01", mode="return", value="1")])
    p, res, meta = run_runner(box, d, [job("C_01__s0", "C/01", snapshot_tgz(box, "u"))],
                              env_extra={"SOME_API_KEY": "sekrit-9f3", "EVOSKILL_X": "1", "EVOSHIM_Y": "2"})
    assert p.returncode == 0, p.stderr
    for k in ("worker", "runner_sha256", "grade_worker_sha256", "python", "executable", "distributions", "expect",
              "env", "max_procs", "max_heavy", "heavy_bytes", "job_timeout_s", "n_jobs", "n_skipped_done", "n_todo",
              "n_heavy_todo", "torn_line_moved_to", "nproc", "statuses", "interrupted", "max_heavy_concurrent_seen"):
        assert k in meta, k
    assert (meta["worker"], meta["runner_sha256"], meta["grade_worker_sha256"]) == (
        "plugin_grade/1", sha(RUNNER), sha(GRADE_WORKER))
    assert meta["python"] == sys.version and meta["distributions"].get("pytest")
    assert meta["expect"] == {x.name: sha(x) for x in d.iterdir()}
    assert meta["env"]["EVOSKILL_X"] == "1" and meta["env"]["EVOSHIM_Y"] == "2"
    assert "sekrit-9f3" not in json.dumps(meta)
    assert (meta["max_procs"], meta["max_heavy"], meta["heavy_bytes"], meta["job_timeout_s"]) == (4, 1, 5_000_000, 5400)
    assert meta["statuses"] == {"ok": 1} and meta["interrupted"] is False


def test_the_runner_imports_only_the_standard_library_and_grade_worker():
    tree = ast.parse(RUNNER.read_text())
    top, inner = set(), set()
    for node in tree.body:
        into = inner if isinstance(node, (ast.FunctionDef, ast.ClassDef)) else top
        for sub in ast.walk(node):
            if isinstance(sub, ast.Import):
                into |= {a.name.split(".")[0] for a in sub.names}
            elif isinstance(sub, ast.ImportFrom):
                into.add((sub.module or "").split(".")[0])
    assert top - set(sys.stdlib_module_names) - {"__future__"} == {"grade_worker"}
    assert inner == {"plugin_api", "benchmark_grader"}              # never benchmark_plugin


def test_the_runner_is_grade_workers_runner_with_only_the_job_replaced():
    assert issubclass(R.PluginRunner, R.gw.Runner)
    for name in ("dispatch", "_start", "_finish", "write", "kill_all", "log"):
        assert name not in R.PluginRunner.__dict__ and getattr(R.PluginRunner, name) is getattr(R.gw.Runner, name)
    assert "run_job" in R.PluginRunner.__dict__
    assert Path(R.gw.__file__).resolve() == GRADE_WORKER


def _assigned(source: str, name: str):
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    return None


def test_grader_failed_is_plugin_apis_label():
    assert R.GRADER_FAILED == "grader_failed" == _assigned(TOY_API, "GRADER_FAILED")
    assert _assigned(PLUGIN_API.read_text(), "GRADER_FAILED") == R.GRADER_FAILED     # the canonical file
