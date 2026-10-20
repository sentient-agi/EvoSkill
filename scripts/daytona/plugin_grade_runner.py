#!/usr/bin/env python3
"""Grade rollouts with the run's frozen grade(task, snapshot), on the pool machine, one child per job.

The grader is an EvoSkill v2 primitive, grade(task, snapshot) -> a
number from 0 to 1, written by the run's user and handed to the benchmark adapter. It lives in the run's pinned
setup/benchmark_grader.py (benchmark_plugin.py only lists the tasks, and this runner never imports it). The loop
never knows the benchmark: it keeps each rollout's whole out.tgz, and this runner hands grade() a fresh extraction
of the solver's working directory and records the number, or GRADER_FAILED when grade() fails.

Runs as root on the pool machine in the grader's setup, launched by cc_pool_host.InPlaceGrader exactly as the SSB
worker was (leg.env sourced, the runner cgroup joined, `env -u OPENROUTER_API_KEY`, /opt/evoskill/venv/bin/python).
Standard library plus grade_worker (read-only reuse of its Runner, which schedules, caps and writes the lines).

  plugin_grade_runner.py run   --jobs J --results R --work W --plugin-dir D --expect NAME=SHA256 [...]
                               [--max-procs 4] [--max-heavy 1] [--heavy-bytes 5000000] [--job-timeout 5400]
                               [--keep-copies]
  plugin_grade_runner.py child --job-file F --result-file R --stage S   (internal: exactly one grade() call)

--expect names every file of D (each file PLUGIN.json pins, plus tasks.json). rc 3, no line written: a model key in
the environment, a file in D no --expect names, a missing file or a sha that differs. rc 2: a malformed or duplicate
job line. rc 0: the list drained. rc 143: SIGTERM/SIGINT (running children are killed and write no line).

JOB LINE (jobs.jsonl; cc_pool_host.InPlaceGrader.job builds it, validate_job checks it):
  {"job_id", "task_id", "snapshot_tgz" (absolute), "snapshot_sha256" (the Mac's container_out.tgz), "snapshot_root"
   ("evo/work"), "meta" {...}}

PER JOB (the parent; every step before the child is the loop's own integrity, and a failure there is "refused"):
the tgz sha must equal the job's, the task must be in tasks.json, the members under snapshot_root are extracted into
<work>/<job>/tree with cc_solver.Box.collect's rule (no link is ever created), the *.py of D are copied into
<work>/<job>/stage, and one child imports the STAGED plugin_api and benchmark_grader, rebuilds the Task from the
tasks.json row and calls grade(task, <work>/<job>/tree/<snapshot_root>). The kept tgz is never handed over, so a
#regrade1 job re-extracts the untouched bytes.

RESULT LINE (results.jsonl, appended and fsynced per job): status is
  ok             grade() returned a number from 0 to 1 (score = float);
  grader_failed  grade() raised (at import or in the call), returned anything else, outlived --job-timeout (its
                 process group killed), or its child ended without a result;
  refused        a loop-side fault (the snapshot sha, an unknown task, the staged plugin_api, the extraction).
score_of(line) is the ONE decision function the Mac applies to a line. RESUME: a job_id already answered is skipped;
a torn last line is moved aside first.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import math
import numbers
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import grade_worker as gw  # noqa: E402  (READ-ONLY reuse: Runner, helpers, JOB_ID_RE)

WORKER_VERSION = "plugin_grade/1"
GRADER_FAILED = "grader_failed"          # == plugin_api.GRADER_FAILED (tests/test_plugin_grade_runner.py pins it)
STATUSES = ("ok", GRADER_FAILED, "refused")
RUNNER = str(Path(__file__).resolve())
TASKS_JSON = "tasks.json"
SHA_RE = re.compile(r"[0-9a-f]{64}")
MAX_LOG_RECORDS = 200
JobError = gw.JobError                   # what validate_job raises (cc_pool_host catches it through this module)


class SetupError(ValueError):
    pass


# ───────────────────────────────────────────────────────────────────────────── shared with the Mac
def check_score(v) -> float | None:
    """The contract's return value: a real number from 0 to 1 (int and bool included) -> float(v); else None."""
    try:
        if isinstance(v, numbers.Real) and math.isfinite(v) and 0 <= v <= 1:
            return float(v)
    except Exception:  # noqa: BLE001 -- a value the check cannot handle is not a score
        pass
    return None


def score_of(line) -> tuple:
    """THE decision on one results.jsonl line: ('ok', float, None) | ('grader_failed', None, reason) |
    ('refused', None, reason). A malformed line counts as refused (a loop fault, never a grader verdict)."""
    if not isinstance(line, dict):
        return "refused", None, "malformed result line (not a JSON object)"
    st = line.get("status")
    if st == "ok":
        s = check_score(line.get("score"))
        if s is None:
            return "refused", None, f"malformed result line: status ok with score {line.get('score')!r}"
        return "ok", s, None
    if st in (GRADER_FAILED, "refused"):
        return st, None, str(line.get("reason") or f"{st} (no reason recorded)")
    return "refused", None, f"malformed result line: status {st!r}"


def validate_job(j) -> dict:
    if not isinstance(j, dict):
        raise JobError("a job must be a JSON object")
    jid = j.get("job_id")
    if not isinstance(jid, str) or not gw.JOB_ID_RE.fullmatch(jid):
        raise JobError(f"bad job_id {jid!r}")
    miss = [k for k in ("task_id", "snapshot_tgz", "snapshot_sha256", "snapshot_root")
            if not isinstance(j.get(k), str) or not j.get(k)]
    if miss:
        raise JobError(f"{jid}: missing {miss}")
    if not os.path.isabs(j["snapshot_tgz"]):
        raise JobError(f"{jid}: snapshot_tgz must be absolute")
    if not SHA_RE.fullmatch(j["snapshot_sha256"]):
        raise JobError(f"{jid}: snapshot_sha256 must be 64 lowercase hex characters")
    root = PurePosixPath(j["snapshot_root"])
    if root.is_absolute() or ".." in root.parts or not root.parts:
        raise JobError(f"{jid}: snapshot_root must be a relative posix path without '..'")
    if not isinstance(j.get("meta"), dict):
        raise JobError(f"{jid}: meta must be an object")
    return j


def load_jobs(path) -> list[dict]:
    jobs, seen = [], set()
    for n, ln in enumerate(Path(path).read_text().splitlines(), 1):
        if not ln.strip():
            continue
        try:
            j = validate_job(json.loads(ln))
        except ValueError as e:                 # JobError is a ValueError
            raise JobError(f"{path}:{n}: {e}")
        if j["job_id"] in seen:
            raise JobError(f"{path}:{n}: duplicate job_id {j['job_id']!r}")
        seen.add(j["job_id"])
        jobs.append(j)
    return jobs


def root_of(job: dict) -> str:
    return PurePosixPath(job["snapshot_root"]).as_posix()


def in_snapshot(name: str, root: str) -> bool:
    n = name.lstrip("/")
    return n == root or n.startswith(root + "/")


def skipped_member(m: tarfile.TarInfo) -> bool:
    """cc_solver.Box.collect's rule (cc_solver.py:667; PurePosixPath there is Path): an absolute name, a '..' part, or
    anything but a regular file or a directory (a symlink, a hard link, a device) is never extracted."""
    return m.name.startswith("/") or ".." in PurePosixPath(m.name).parts or not (m.isfile() or m.isdir())


def extract_snapshot(tgz, root: str, dest: Path) -> tuple:
    """The members named `root` or under `root/` into dest, with Box.collect's rule and tarfile's 'data' filter. A
    skill leg's evo/work/.claude/skills -> /evo/skills link is therefore never created. (regular files written,
    skipped member names). dest/root always exists afterwards: a rollout that left nothing is still a snapshot."""
    dest.mkdir(parents=True)
    n, skipped = 0, []
    with tarfile.open(tgz, "r:gz") as tf:
        for m in tf.getmembers():
            if not in_snapshot(m.name, root):
                continue
            if skipped_member(m):
                skipped.append(m.name)
                continue
            tf.extract(m, dest, filter="data")
            n += 1 if m.isfile() else 0
    (dest / root).mkdir(parents=True, exist_ok=True)
    return n, skipped


def snapshot_bytes(tgz, root: str) -> int:
    """The largest regular file the extraction would write (0 when the tgz cannot be read: the job refuses later)."""
    best = 0
    try:
        with tarfile.open(tgz, "r:gz") as tf:
            for m in tf.getmembers():
                if in_snapshot(m.name, root) and not skipped_member(m) and m.isfile():
                    best = max(best, m.size)
    except Exception:  # noqa: BLE001 -- a size estimate only
        pass
    return best


def answer_bytes(row) -> int:
    best = 0
    for p in (row or {}).get("answer_files") or []:
        try:
            best = max(best, os.path.getsize(p))
        except OSError:
            pass
    return best


def parse_expect(items) -> dict:
    out = {}
    for it in items or []:
        name, sep, sha = str(it).partition("=")
        if not sep or not name or "/" in name or name in (".", "..") or not SHA_RE.fullmatch(sha):
            raise SetupError(f"--expect {it!r} is not NAME=SHA256")
        if name in out:
            raise SetupError(f"--expect names {name} twice")
        out[name] = sha
    return out


def check_plugin_dir(d: Path, expect: dict) -> None:
    """Every file of the plug-in dir is named by --expect and hashes to it; every --expect file is there."""
    if not d.is_dir():
        raise SetupError(f"plug-in dir {d} does not exist")
    extra = sorted({p.name for p in d.iterdir()} - set(expect))
    if extra:
        raise SetupError(f"plug-in dir {d} holds {extra}, which no --expect names")
    for name, want in sorted(expect.items()):
        p = d / name
        if not p.is_file():
            raise SetupError(f"{p} is missing")
        got = gw.sha256_file(p)
        if got != want:
            raise SetupError(f"{name} sha256 {got[:12]} != expected {want[:12]}")


def load_tasks(path: Path) -> dict:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise SetupError(f"{path}: {e}")
    if not isinstance(rows, list) or not all(isinstance(r, dict) and isinstance(r.get("id"), str) for r in rows):
        raise SetupError(f"{path}: not a list of task rows")
    return {r["id"]: r for r in rows}


def distributions() -> dict:
    out = {}
    try:
        for d in importlib.metadata.distributions():
            name = d.metadata["Name"] if d.metadata else None
            if name:
                out[name] = d.version
    except Exception:  # noqa: BLE001 -- evidence only
        pass
    return dict(sorted(out.items(), key=lambda kv: kv[0].lower()))


# ───────────────────────────────────────────────────────────────────────────── the child
class _Records(logging.Handler):
    """This job's WARNING+ records as text (the first MAX_LOG_RECORDS): evidence only, the loop never reads them."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.lines: list[str] = []

    def emit(self, record):
        if len(self.lines) >= MAX_LOG_RECORDS:
            return
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001 -- a bad format string still gets recorded
            msg = str(record.msg)
        self.lines.append(f"{record.levelname} {record.name}: {msg}"[:2000])


def _repr(v) -> str:
    try:
        return repr(v)[:200]
    except Exception:  # noqa: BLE001
        return f"<{type(v).__name__}>"


def child_grade(job: dict, stage: Path, out: dict) -> None:
    sys.path.insert(0, str(stage))
    try:
        import plugin_api as A
    except BaseException as e:  # noqa: BLE001 -- the canonical file: a loop fault
        out.update(status="refused", reason=f"the staged plugin_api cannot be imported: {type(e).__name__}: "
                                            f"{str(e)[:300]}", exception=gw._exc(e, "plugin_api"))
        return
    got = Path(getattr(A, "__file__", None) or "").resolve()
    if got != (stage / "plugin_api.py").resolve():
        out.update(status="refused", reason=f"plugin_api was imported from {got}, not the staged copy")
        return
    row = job["task"]
    try:
        task = A.Task(id=row["id"], category=row["category"], instruction=row["instruction"],
                      inputs=tuple(Path(p) for p in row["inputs"]),
                      answer_files=tuple(Path(p) for p in row["answer_files"]),
                      grading=dict(row["grading"]), split=row["split"])
    except Exception as e:  # noqa: BLE001 -- a frozen row that is not a Task: a loop fault
        out.update(status="refused", reason=f"the tasks.json row is not a Task: {type(e).__name__}: {str(e)[:300]}",
                   exception=gw._exc(e, "task"))
        return
    phase, t = "import", time.time()
    try:
        import benchmark_grader as GR           # the user's grader file; benchmark_plugin is never imported
        phase, t = "grade", time.time()
        v = GR.grade(task, Path(job["snapshot"]))
    except BaseException as e:  # noqa: BLE001 -- raising is how grade() says the grader failed (SystemExit too)
        out.update(status=GRADER_FAILED, reason=f"grade() raised {type(e).__name__}: {str(e)[:300]}",
                   exception=gw._exc(e, phase))
        return
    finally:
        if phase == "grade":
            out["grade_s"] = round(time.time() - t, 2)
    s = check_score(v)
    if s is None:
        out.update(status=GRADER_FAILED, reason=f"grade() returned {_repr(v)} (not a number from 0 to 1)")
    else:
        out.update(status="ok", score=s)


def child_main(args) -> int:
    job = json.loads(Path(args.job_file).read_text())
    cap = _Records()
    echo = logging.StreamHandler(sys.stderr)
    echo.setLevel(logging.WARNING)
    root = logging.getLogger()
    root.addHandler(cap)
    root.addHandler(echo)
    logging.captureWarnings(True)
    out = {"status": GRADER_FAILED, "score": None, "reason": None, "grade_s": None}
    try:
        child_grade(job, Path(args.stage), out)
    except BaseException as e:  # noqa: BLE001 -- the child's own fault, still a result
        out.update(status="refused", score=None, reason=f"runner child error: {type(e).__name__}: {str(e)[:300]}",
                   exception=gw._exc(e, "child"))
    out["log_records"] = cap.lines
    res = Path(args.result_file)
    tmp = res.with_name(res.name + ".tmp")
    tmp.write_text(json.dumps(out, default=str))
    os.replace(tmp, res)
    return 0


# ───────────────────────────────────────────────────────────────────────────── the parent
class PluginRunner(gw.Runner):
    """grade_worker's Runner (dispatch, the caps, write, kill_all) with run_job calling grade() in a child."""

    def __init__(self, args, jobs: list[dict], tasks: dict, expect: dict):
        super().__init__(args, jobs)
        self.tasks = tasks
        self.plugin = dict(expect)
        self.plugin_dir = Path(args.plugin_dir)

    def prepare(self, job: dict, jd: Path, rec: dict) -> str | None:
        """The loop's own integrity before any child: the refusal reason, or None with jd/job.json written."""
        tgz = job["snapshot_tgz"]
        try:
            got = gw.sha256_file(tgz)
        except OSError as e:
            return f"snapshot {tgz} unreadable: {e}"
        if got != job["snapshot_sha256"]:
            return f"snapshot sha256 {got[:12]} != job's {job['snapshot_sha256'][:12]}"
        row = self.tasks.get(job["task_id"])
        if row is None:
            return f"task {job['task_id']!r} is not in tasks.json"
        root = root_of(job)
        try:
            rec["snapshot_files"], rec["snapshot_skipped"] = extract_snapshot(tgz, root, jd / "tree")
        except Exception as e:  # noqa: BLE001 -- any extraction error is the loop's
            return f"extraction failed: {type(e).__name__}: {str(e)[:300]}"
        (jd / "stage").mkdir()
        for p in sorted(self.plugin_dir.glob("*.py")):
            shutil.copy2(p, jd / "stage" / p.name)   # the child imports this copy: the pinned files are never touched
        (jd / "job.json").write_text(json.dumps({"job_id": job["job_id"], "task": row,
                                                 "snapshot": str(jd / "tree" / root)}))
        return None

    def call_child(self, jd: Path, rec: dict) -> bool:
        """One child, one grade() call. False: interrupted without a result (no line, so a relaunch grades it)."""
        jid = rec["job_id"]
        res = jd / "child_result.json"
        cmd = [sys.executable, "-u", "-B", RUNNER, "child", "--job-file", str(jd / "job.json"),
               "--result-file", str(res), "--stage", str(jd / "stage")]
        with open(jd / "child.log", "wb") as logf:
            p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=str(jd), env=self.child_env,
                                 start_new_session=True)
            self.procs[jid] = p
            if self.stop.is_set():
                gw._killpg(p)                   # the stop landed while this child was starting
            timed_out = False
            try:
                rc = p.wait(timeout=self.args.job_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                gw._killpg(p)
                rc = p.wait()
            finally:
                self.procs.pop(jid, None)
        rec.update(child_pid=p.pid, child_rc=rc)
        if self.stop.is_set() and not res.is_file():
            return False
        if timed_out:
            rec.update(status=GRADER_FAILED, reason=f"timeout after {self.args.job_timeout} s (process group killed)")
            return True
        try:
            out = json.loads(res.read_text())
        except (OSError, ValueError):
            rec.update(status=GRADER_FAILED, reason=f"crashed (child rc {rc}, no result)")
            return True
        rec.update(status=out.get("status"), score=out.get("score"), reason=out.get("reason"),
                   log_records=out.get("log_records") or [], grade_s=out.get("grade_s"))
        if out.get("exception"):
            rec["exception"] = out["exception"]
        return True

    def run_job(self, job: dict, heavy: bool) -> None:
        jid = job["job_id"]
        jd = self.work / gw.safe_name(jid)
        started, t0 = gw.utcnow(), time.time()
        rec = {"job_id": jid, "task_id": job["task_id"], "worker": WORKER_VERSION, "status": "refused", "score": None,
               "reason": None, "log_records": [], "snapshot_tgz": job["snapshot_tgz"],
               "snapshot_sha256": job["snapshot_sha256"], "snapshot_files": None, "snapshot_skipped": [],
               "plugin": self.plugin, "grade_s": None, "child_pid": None, "child_rc": None}
        try:
            try:
                if jd.exists():
                    shutil.rmtree(jd)           # a leftover from an interrupted run: start fresh
                jd.mkdir(parents=True)
                why = self.prepare(job, jd, rec)
                if why:
                    rec["reason"] = why
                elif self.stop.is_set() or not self.call_child(jd, rec):
                    return                      # interrupted: no line, so a relaunch grades it again
            except Exception as e:  # noqa: BLE001 -- the parent must never lose a job silently; a loop fault
                rec.update(status="refused", score=None, reason=f"runner error: {type(e).__name__}: {str(e)[:300]}",
                           exception=gw._exc(e, "runner"))
            rec.update(started_utc=started, ended_utc=gw.utcnow(), seconds=round(time.time() - t0, 2), heavy=heavy,
                       job_timeout_s=self.args.job_timeout)
            if rec["status"] != "ok":
                try:
                    rec["log_tail"] = (jd / "child.log").read_bytes()[-gw.LOG_TAIL:].decode("utf-8", "replace")
                except OSError:
                    rec["log_tail"] = ""
            rec["meta"] = job.get("meta")
            self.write(rec)
            self.log(f"[{self.n_done}/{len(self.jobs)}] {rec['status']:<13} {rec['seconds']:>8.1f}s "
                     f"heavy={int(heavy)} {jid}")
        finally:
            if not self.args.keep_copies:
                for sub in ("tree", "stage"):
                    shutil.rmtree(jd / sub, ignore_errors=True)


def run_main(args) -> int:
    if os.environ.get("OPENROUTER_API_KEY"):
        print("REFUSED: OPENROUTER_API_KEY is in the environment; the grader's setup never holds a model key",
              file=sys.stderr)
        return 3
    try:
        expect = parse_expect(args.expect)
        check_plugin_dir(Path(args.plugin_dir), expect)
        tasks = load_tasks(Path(args.plugin_dir) / TASKS_JSON)
    except SetupError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    try:
        jobs = load_jobs(args.jobs)
    except (JobError, OSError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    args.max_procs, args.max_heavy = max(1, args.max_procs), max(1, args.max_heavy)
    results = Path(args.results)
    results.parent.mkdir(parents=True, exist_ok=True)
    Path(args.work).mkdir(parents=True, exist_ok=True)
    torn = gw.repair_results(results)
    done = gw.done_ids(results)
    todo = [j for j in jobs if j["job_id"] not in done]
    for j in todo:      # heavy: today's max(reference, output) rule on the plug-in's terms
        j["_heavy"] = max(snapshot_bytes(j["snapshot_tgz"], root_of(j)),
                          answer_bytes(tasks.get(j["task_id"]))) >= args.heavy_bytes
    meta = {"worker": WORKER_VERSION, "runner_sha256": gw.sha256_file(RUNNER),
            "grade_worker_sha256": gw.sha256_file(gw.__file__), "started_utc": gw.utcnow(), "python": sys.version,
            "executable": sys.executable, "distributions": distributions(), "argv": sys.argv,
            "plugin_dir": args.plugin_dir, "expect": expect,
            "env": {k: v for k, v in sorted(os.environ.items())
                    if k.startswith(("EVOSKILL_", "EVOSHIM_", "OMP_", "OPENBLAS_", "MKL_", "NUMEXPR_",
                                     "MAX_CONCURRENCY", "PYTHON"))
                    and not gw.SECRET_ENV_RE.search(k)},
            "max_procs": args.max_procs, "max_heavy": args.max_heavy, "heavy_bytes": args.heavy_bytes,
            "job_timeout_s": args.job_timeout, "n_jobs": len(jobs), "n_skipped_done": len(jobs) - len(todo),
            "n_todo": len(todo), "n_heavy_todo": sum(1 for j in todo if j["_heavy"]), "torn_line_moved_to": torn,
            "nproc": os.cpu_count()}
    meta_path = results.with_name("worker_meta.json")
    meta_path.write_text(json.dumps(meta, indent=1, default=str))
    r = PluginRunner(args, todo, tasks, expect)
    r.log(f"{WORKER_VERSION}: {len(todo)} job(s) to run ({meta['n_heavy_todo']} heavy), {meta['n_skipped_done']} "
          f"already done; max_procs={args.max_procs} max_heavy={args.max_heavy} job_timeout={args.job_timeout}s")

    def on_sig(signum, _frm):
        # Only set the flag: dispatch() polls it every 0.5 s, kills the children and joins.
        r.stop.set()
        print(f"{gw.utcnow()} signal {signum}: stopping; running children are killed and write no line", flush=True)
    signal.signal(signal.SIGTERM, on_sig)
    signal.signal(signal.SIGINT, on_sig)
    r.dispatch()
    meta.update(ended_utc=gw.utcnow(), statuses=dict(r.statuses), n_written=r.n_done, interrupted=r.stop.is_set(),
                max_heavy_concurrent_seen=r.max_heavy_seen)
    meta_path.write_text(json.dumps(meta, indent=1, default=str))
    r.log(f"done: {dict(r.statuses)} interrupted={r.stop.is_set()}")
    return 143 if r.stop.is_set() else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--jobs", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--plugin-dir", required=True)
    p.add_argument("--expect", action="append", default=[], metavar="NAME=SHA256")
    p.add_argument("--max-procs", type=int, default=4)
    p.add_argument("--max-heavy", type=int, default=1)
    p.add_argument("--heavy-bytes", type=int, default=5_000_000)
    p.add_argument("--job-timeout", type=int, default=5400)
    p.add_argument("--keep-copies", action="store_true")
    p = sub.add_parser("child")
    p.add_argument("--job-file", required=True)
    p.add_argument("--result-file", required=True)
    p.add_argument("--stage", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "child":
        return child_main(args)
    return run_main(args)


if __name__ == "__main__":
    sys.exit(main())
