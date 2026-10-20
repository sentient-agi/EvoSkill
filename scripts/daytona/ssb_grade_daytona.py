#!/usr/bin/env python3
"""The SSB v2 GRADING SANDBOX: re-grade saved workbooks with the grader of record, in
the leg image, on one Daytona sandbox -- and nothing else.

WHAT IT IS FOR. The grader of record is score_and_diff in src/agent_profiles/ssbv2_agent/scorer.py
(sha 2442e5aa...) run in the leg image (scripts/daytona/Dockerfile.ssb_leg) with
its LibreOffice grading profile. It calls `docker run ssb-verifier`, which exists only inside a
sandbox with the docker shim, so every re-grade -- the validation before freeze (reference and
input workbooks, the saved outputs), the re-grade of a "grader failed" sample, Stage B -- runs
here. The worker (scripts/daytona/grade_worker.py) grades one job per child process; this driver
builds the sandbox, proves it is the leg's, runs the worker, brings the results home, deletes the
sandbox and writes the ledger.

ONE SANDBOX PER INVOCATION, IDENTICAL TO A LEG'S (same checks, reused read-only):
  * image: Dockerfile.ssb_leg (sha pinned in ssb_leg_daytona_loop.EXPECTED, restated in the run's
    INFRA.json leg_sandbox -- the three must agree), built by content, and the Daytona snapshot
    ref READ BACK must be the baseline's; resources 8 vCPU / 32 GiB / 40 GB read back;
  * identity probe, LibreOffice build, grading openpyxl 3.1.5, no claude binary:
    ssb_leg_daytona_loop.Shard.identity, run unchanged on this sandbox;
  * docker shim installed exactly as a leg installs it; cgroup delegation, the shim isolation
    self-test and the golden-vs-golden grading self-test: Shard.cgroups_and_selftest, unchanged;
  * the leg's knobs (ssb_leg_daytona.KNOBS) plus INFRA.json's grader timeouts (900/900), via
    ssb_leg_daytona_loop.leg_knobs (an env clash or any other knob drift is refused);
  * the data overlay (repaired input workbooks, src/orchestrator/data_overlay.py) from INFRA.json's
    block "data", by the leg's own rule (ssb_leg_daytona_loop.data_overlay_from_infra: the variable
    set from it when unset, refused when it names another file or is set while INFRA declares none,
    PATCHES.json re-hashed and every entry verified); job shas and uploaded bytes use that path and
    the summary records it.
DIFFERENT FROM A LEG, deliberately:
  * NO model key anywhere: the driver never reads OPENROUTER_API_KEY, no exec call carries an env,
    and the worker refuses to start if one is present;
  * EGRESS BLOCKED: created with network_block_all=True, asserted on readback; the leg's egress
    probe must then reach NOTHING (openrouter.ai included). `--egress unresolvable-allow-list` is
    the fallback if the API ever refuses block-all: an allow list holding only the reserved,
    never-resolving name `invalid.invalid` (an EMPTY list might mean "no restriction"), with the
    same all-blocked probe. The summary records which one ran;
  * the upload is src/ (the frozen grader + its docker_scripts, sha-checked before and inside the
    sandbox), the three dataset.json files, and EXACTLY the workbooks the job list needs (plus the
    self-test task's two), mirrored at their Mac paths; evaluation.py only when a Stage B job exists;
  * a SPEND GUARD on sandbox time ($0.926/sandbox-hour list price x wall; --max-usd) and a HARD
    WALL (--wall-min; the sandbox's ttl is wall + 30 min as the backstop);
  * sandboxes are named evoskill-grade-<label>-<HHMMSS>; the ledger rows go to the Daytona ledger
    (daytona_lib_loop.LEDGER) with driver "ssb_grade_daytona"; delete is by recorded id only, verified, in a
    finally.

RESUMABLE. <out-dir>/results.jsonl accumulates every result (one line per job_id, never
rewritten). A rerun uploads only the jobs with no line there; inside the sandbox the worker also
skips any job already in its own results file. Results are synced home every --sync-s while the
worker runs, so a crashed driver loses at most that much.

HOST-SIDE ANNOTATION. Each merged line gets a "host" block: for a grade, the outcome from
src/orchestrator/outcomes.py classify_detail on the DIFF.txt text the runner would have written
and the job's own WARNING lines (an exception / timeout / crash = grader failed); the scorer's
caught-exception warnings are only flagged (`open_question`), never judged.

  make-jobs    --tasks A,B,C [--kinds golden,input,open,stageB_golden] [--saved-json F]
               [--split F | --run-dir R] [--key train] --out JOBS.jsonl
  run          --jobs JOBS.jsonl --out-dir D [--run-dir R] [--label L] [--max-procs 4]
               [--max-heavy 2] [--job-timeout S] [--wall-min 120] [--max-usd 3]
               [--egress block-all|unresolvable-allow-list] [--probe-task T] [--check-only]
  status       --out-dir D [--jobs JOBS.jsonl]
  regrade-jobs --results D/results.jsonl --out JOBS.jsonl [--suffix '#regrade1']
  cleanup      [--force]     # grade sandboxes the ledger shows created and not deleted

Run with the venv that has the `daytona` extra (the Daytona SDK + python-dotenv); make-jobs / status /
regrade-jobs / run --check-only never import the SDK or read any key.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shlex
import signal
import sys
import tarfile
import threading
import time
import traceback
import types
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(1, str(HERE.parents[1]))
import daytona_lib_loop as dlx  # noqa: E402  (READ-ONLY reuse: client, readback, run, now)
import grade_worker as gw  # noqa: E402  (stdlib only: job schema, constants)
import ssb_leg_daytona as base  # noqa: E402  (READ-ONLY reuse: image, knobs, commands, index)
import ssb_leg_daytona_loop as loop  # noqa: E402  (READ-ONLY reuse: pins, knobs, Shard checks)
from src.orchestrator import data_overlay  # noqa: E402  (repaired files: bytes and shas, path unchanged)

REPO = base.REPO
DATA = base.DATA
DOCKERFILE = base.DOCKERFILE
EXPECTED = loop.EXPECTED
RESOURCES = dict(base.RESOURCES)
CATEGORIES = base.CATEGORIES
PREFIX = "evoskill-grade-"
OWNER_LABEL = "evoskill-grade"
DRIVER = "ssb_grade_daytona"
USD_PER_HOUR = loop.DAYTONA_USD_PER_SANDBOX_HOUR
# the official evaluation.py of the SpreadsheetBench v2 checkout the data comes from (Stage B jobs only)
EVALUATION = (Path(DATA).parents[1] / "evaluation.py") if DATA is not None else None
# make-jobs --key checks tasks against a split: by default the one --run-dir's SPEC.json declares
# (loop.default_split), else loop.PROJECT_SPLIT.
DEFAULT_SPLIT = REPO / loop.PROJECT_SPLIT[0]
SANDBOX_WORK = f"{REPO}/.cache/_grade_work"        # inside the repo mirror (the host's repo path on the machine)
SANDBOX_OUT = "/opt/grade/out"
SANDBOX_JOBS = "/opt/grade/jobs.jsonl"
SANDBOX_WORKER = "/opt/grade/grade_worker.py"
POLL_S = int(os.environ.get("EVOSKILL_DAYTONA_POLL_S", "30"))
UNRESOLVABLE = "invalid.invalid"          # RFC 6761/2606: never resolves
WORKBOOK_SUFFIXES = (".xlsx", ".xlsm")
SECRET_NAME_RE = re.compile(r"(^\.env)|oauth_token|api_key", re.I)
KEYLIKE_RE = re.compile(rb"sk-or-v1-[0-9a-f]{16}|sk-ant-[A-Za-z0-9_-]{16}")
EXIT_OK, EXIT_REFUSED, EXIT_INCOMPLETE, EXIT_CHECK_ONLY = 0, 4, 5, 7

# The grader's code (scorer + shim from EXPECTED; the three docker_scripts the scorer runs, pinned here).
GRADER_CODE_PINS = {
    EXPECTED["baseline_runtime"]["scorer_sha256"][0]: EXPECTED["baseline_runtime"]["scorer_sha256"][1],
    "src/agent_profiles/ssbv2_agent/docker_scripts/datatable_scan.py":
        "69ae5cb116d59f1be48983769bc58ac5b68bb7df454701cdd96cd199657d2328",
    "src/agent_profiles/ssbv2_agent/docker_scripts/datatable_uno_solve.py":
        "a4c777fe0278437d7d0c7cfd5822ad1cbeca8d44412b817276c6ca81b12aa6fe",
    "src/agent_profiles/ssbv2_agent/docker_scripts/run_datatable_fix.sh":
        "07e0e70a87fde505d42cf4e7b998e7c4d63fe66e6a2f169cbba1ec53c156f8cb",
    EXPECTED["baseline_runtime"]["shim_sha256"][0]: EXPECTED["baseline_runtime"]["shim_sha256"][1],
}
# Recorded, not refused: code on the grading path that nothing pinned at the baseline.
RECORDED_CODE = ("src/agent_profiles/ssbv2_agent/dataset.py",
                 "src/agent_profiles/ssbv2_agent/__init__.py", "src/agent_profiles/__init__.py",
                 "scripts/daytona/grade_worker.py", "scripts/daytona/ssb_grade_daytona.py")

Refused = loop.Refused
Abort = loop.Abort


def sha256_file(p) -> str:
    return gw.sha256_file(p)


def utcnow() -> str:
    return dlx.now()


# ───────────────────────────────────────────────────────────────────────────── ledger + sandbox
def ledger(**kw) -> None:
    """The Daytona ledger (dlx.LEDGER, read at call time so tests can redirect it), driver field ours."""
    kw = {"ts": utcnow(), "driver": DRIVER, **kw}
    with open(dlx.LEDGER, "a") as fh:
        fh.write(json.dumps(kw, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def create_sandbox(d, params, requested: dict, tag: str, timeout: float = 3600, on_logs=None):
    """daytona_lib_loop.create with this driver's prefix, owner label and ledger driver field:
    ledger request -> id as soon as visible -> readback. Returns (sb, id, err, seconds)."""
    name = params.name
    assert name and name.startswith(PREFIX), name
    params.labels = dict(params.labels or {})
    params.labels.update({"owner": OWNER_LABEL, "tag": tag})
    ledger(event="create_requested", tag=tag, name=name, requested=requested,
           ttl_minutes=getattr(params, "ttl_minutes", None),
           auto_stop_interval=getattr(params, "auto_stop_interval", None),
           network_block_all=getattr(params, "network_block_all", None),
           domain_allow_list=getattr(params, "domain_allow_list", None))
    box: dict = {}
    t0 = time.time()

    def _go():
        try:
            if on_logs is not None:
                box["sb"] = d.create(params, timeout=timeout, on_snapshot_create_logs=on_logs)
            else:
                box["sb"] = d.create(params, timeout=timeout)
        except BaseException as e:  # noqa: BLE001
            box["err"] = e

    th = threading.Thread(target=_go, daemon=True)
    th.start()
    seen = None
    while th.is_alive():
        th.join(10)
        if seen is None:
            try:
                s = d.get(name)
                seen = s.id
                ledger(event="id_seen_during_create", tag=tag, name=name, id=seen, state=str(s.state))
            except Exception:  # noqa: BLE001
                pass
    dt = round(time.time() - t0, 1)
    if "err" in box:
        sid = seen
        if sid is None:
            try:
                sid = d.get(name).id
            except Exception:  # noqa: BLE001
                sid = None
        ledger(event="create_failed", tag=tag, name=name, id=sid, seconds=dt,
               error=f"{type(box['err']).__name__}: {str(box['err'])[:1500]}")
        return None, sid, box["err"], dt
    sb = box["sb"]
    rb = dlx.readback(sb)
    ledger(event="created", tag=tag, name=name, id=sb.id, seconds=dt, readback=rb,
           requested=requested)
    return sb, sb.id, None, dt


def delete_by_id(d, sid: str, tag: str) -> bool:
    """Delete (not stop) the sandbox with THIS id if it carries our prefix; verify; ledger."""
    if not sid:
        return True
    ok = False
    for attempt in range(6):
        try:
            sb = d.get(sid)
        except Exception as e:  # noqa: BLE001
            if "not found" in str(e).lower() or "404" in str(e):
                ok = True
                break
            time.sleep(5)
            continue
        if not str(sb.name).startswith(PREFIX):
            ledger(event="delete_refused_not_ours", tag=tag, id=sid, name=sb.name)
            return False
        try:
            d.delete(sb, timeout=180, wait=True)
            ok = True
            break
        except Exception as e:  # noqa: BLE001
            ledger(event="delete_retry", tag=tag, id=sid, attempt=attempt, error=str(e)[:300])
            time.sleep(10)
    try:
        state = str(d.get(sid).state)
    except Exception as e:  # noqa: BLE001
        state = f"get-failed: {str(e)[:160]}"
    gone = ok and any(w in state.lower() for w in ("destroy", "not found", "get-failed"))
    ledger(event="deleted" if gone else "delete_UNVERIFIED", tag=tag, id=sid, post_state=state)
    return gone


# ───────────────────────────────────────────────────────────────────────────── jobs
def _workbook_ok(p: Path, what: str) -> Path:
    if not p.is_absolute():
        raise Refused(f"{what} {p} must be an absolute path")
    if SECRET_NAME_RE.search(p.name) or p.suffix.lower() not in WORKBOOK_SUFFIXES:
        raise Refused(f"{what} {p} is not a workbook (.xlsx/.xlsm) -- refusing to upload it")
    if p.is_symlink() or not p.is_file():
        raise Refused(f"{what} {p} is missing or a symlink")
    return p


def normalize_jobs(raw: list, idx: dict, overlay=None) -> list:
    """Minimal jobs ({job_id, mode, task_id, output} / {job_id, mode:"open", path}) -> the worker's
    full schema. The reference, the input and answer_position ALWAYS come from the dataset index;
    a job that states them differently is refused. Every sha is computed here, of the
    bytes the sandbox gets (`overlay`: the PATCHES.json GradeConfig took from INFRA.json; None = the
    bare EVOSKILL_SSBV2_DATA_OVERLAY, e.g. make-jobs)."""
    def src(p):
        return data_overlay.source(p, overlay)
    out, seen = [], set()
    for j in raw:
        if not isinstance(j, dict):
            raise Refused(f"a job must be an object: {j!r}"[:200])
        jid, mode = j.get("job_id"), j.get("mode")
        if not isinstance(jid, str) or not gw.JOB_ID_RE.fullmatch(jid):
            raise Refused(f"bad job_id {jid!r}")
        if jid in seen:
            raise Refused(f"duplicate job_id {jid!r}")
        seen.add(jid)
        if mode not in gw.MODES:
            raise Refused(f"{jid}: mode {mode!r} not in {gw.MODES}")
        n = {"job_id": jid, "mode": mode}
        if mode in ("grade", "stageB"):
            tid = j.get("task_id")
            if tid not in idx:
                raise Refused(f"{jid}: task_id {tid!r} is not a dataset task")
            info = idx[tid]
            want = {"gt_file": str(info["golden"]), "answer_position": info["answer_position"]}
            if mode == "grade":
                want["input_file"] = str(info["input"])
            for k, v in want.items():
                if k in j and j[k] != v:
                    raise Refused(f"{jid}: {k} {j[k]!r} differs from the dataset's {v!r}")
            out_p = _workbook_ok(Path(str(j.get("output") or "")), f"{jid}: output")
            n.update(task_id=tid, **want, output=str(out_p), output_sha256=sha256_file(src(out_p)),
                     gt_sha256=sha256_file(src(want["gt_file"])))
            if mode == "grade":
                n["input_sha256"] = sha256_file(src(want["input_file"]))
            if j.get("output_sha256") and j["output_sha256"] != n["output_sha256"]:
                raise Refused(f"{jid}: output_sha256 differs from the file on disk now")
        else:
            p = _workbook_ok(Path(str(j.get("path") or "")), f"{jid}: path")
            n.update(path=str(p), path_sha256=sha256_file(src(p)))
            if j.get("task_id"):
                n["task_id"] = j["task_id"]
            if j.get("path_sha256") and j["path_sha256"] != n["path_sha256"]:
                raise Refused(f"{jid}: path_sha256 differs from the file on disk now")
        if "meta" in j:
            if not isinstance(j["meta"], dict):
                raise Refused(f"{jid}: meta must be an object")
            n["meta"] = j["meta"]
        gw.validate_job(n)
        out.append(n)
    return out


def read_jsonl(path: Path) -> list:
    rows = []
    for k, ln in enumerate(Path(path).read_text().splitlines(), 1):
        if ln.strip():
            try:
                rows.append(json.loads(ln))
            except ValueError as e:
                raise Refused(f"{path}:{k}: not JSON: {e}")
    return rows


def make_jobs(tasks: list, kinds: list, saved: list, idx: dict) -> list:
    """Job list for dataset workbooks (the reference workbook must pass, the untouched input, opens, Stage B on the
    golden) and for saved outputs ({task, output, label?, modes?, meta?})."""
    known = {"golden", "input", "open", "stageB_golden"}
    bad = sorted(set(kinds) - known)
    if bad:
        raise Refused(f"unknown kind(s) {bad}; known {sorted(known)}")
    jobs = []
    for t in tasks:
        if t not in idx:
            raise Refused(f"{t} is not a dataset task")
        g, i = str(idx[t]["golden"]), str(idx[t]["input"])
        if "golden" in kinds:
            jobs.append({"job_id": f"{t}/golden/grade", "mode": "grade", "task_id": t, "output": g,
                         "meta": {"kind": "golden_as_output"}})
        if "input" in kinds:
            jobs.append({"job_id": f"{t}/input/grade", "mode": "grade", "task_id": t, "output": i,
                         "meta": {"kind": "input_as_output"}})
        if "open" in kinds:
            jobs.append({"job_id": f"{t}/golden/open", "mode": "open", "task_id": t, "path": g,
                         "meta": {"kind": "golden"}})
            jobs.append({"job_id": f"{t}/input/open", "mode": "open", "task_id": t, "path": i,
                         "meta": {"kind": "input"}})
        if "stageB_golden" in kinds:
            jobs.append({"job_id": f"{t}/golden/stageB", "mode": "stageB", "task_id": t,
                         "output": g, "meta": {"kind": "golden_as_output"}})
    for s in saved:
        t, o = s.get("task"), s.get("output")
        label = s.get("label") or hashlib.sha256(str(o).encode()).hexdigest()[:10]
        for mode in s.get("modes") or ("grade", "stageB"):
            j = {"job_id": f"{t}/saved/{label}/{mode}", "mode": mode, "task_id": t,
                 "meta": dict(s.get("meta") or {}, kind="saved_output", label=label)}
            j["path" if mode == "open" else "output"] = o
            jobs.append(j)
    return jobs


# ───────────────────────────────────────────────────────────────────────────── configuration
class GradeConfig:
    """Everything decided before any sandbox exists. Refused = nothing created, nothing spent."""

    def __init__(self, *, jobs_file, out_dir, run_dir, label="grade", env=None,
                 max_procs=0, max_heavy=0, job_timeout=0, wall_min=120, max_usd=3.0,
                 egress="block-all", probe_task=None, sync_s=300, idx=None):
        process_env = env is None               # only then is the data overlay set in os.environ
        env = dict(os.environ if env is None else env)
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,23}", label or ""):
            raise Refused(f"label {label!r}: 1-24 chars of [a-z0-9-] (it names the sandbox)")
        if egress not in ("block-all", "unresolvable-allow-list"):
            raise Refused(f"egress {egress!r}")
        self.label, self.egress = label, egress
        self.run_dir = Path(run_dir).resolve()
        self.out_dir = Path(out_dir).resolve()
        self.infra, self.infra_record = loop.load_infra(self.run_dir)
        if self.infra is None:
            raise Refused(f"no INFRA.json in {self.run_dir}: the grader timeouts come from it")
        self.knobs = loop.leg_knobs(self.infra, env)
        # Before any job sha is computed: the overlay INFRA.json declares (the leg's rule).
        self.data_overlay = loop.data_overlay_from_infra(self.infra, env, set_process_env=process_env)
        ls = self.infra["leg_sandbox"]
        df_sha = sha256_file(DOCKERFILE)
        agree = {
            "dockerfile_sha256": (EXPECTED["dockerfile_sha256"], ls.get("dockerfile_sha256"), df_sha),
            "snapshot_ref": (EXPECTED["snapshot_ref"], ls.get("snapshot_ref")),
            "image_id": (EXPECTED["image_id"], ls.get("image_id"), "x86daytona-" + df_sha[:12]),
            "libreoffice": (EXPECTED["libreoffice"], ls.get("libreoffice")),
            "resources": (RESOURCES, ls.get("resources")),
        }
        bad = {k: v for k, v in agree.items() if any(x != v[0] for x in v[1:])}
        if not str(ls.get("grading_openpyxl", "")).startswith(EXPECTED["grading_openpyxl_prefix"]):
            bad["grading_openpyxl"] = (EXPECTED["grading_openpyxl_prefix"], ls.get("grading_openpyxl"))
        if bad:
            raise Refused(f"the leg image is stated differently by the leg driver (EXPECTED), "
                          f"INFRA.json and the disk: {bad}")
        self.image_id = EXPECTED["image_id"]
        drift = {rel: (sha256_file(REPO / rel)[:12], want[:12]) for rel, want in GRADER_CODE_PINS.items()
                 if sha256_file(REPO / rel) != want}
        if drift:
            raise Refused(f"grader code differs from what graded the baseline: {drift}")
        self.code_sha256 = {rel: sha256_file(REPO / rel) for rel in
                            list(GRADER_CODE_PINS) + list(RECORDED_CODE) if (REPO / rel).is_file()}
        self.grader_clean = self._grader_clean_agreement()
        self.recalc_timeout = int(self.knobs[loop.infra_cfg.RECALC_TIMEOUT_ENV])
        self.uno_timeout = int(self.knobs[loop.infra_cfg.DATATABLE_UNO_TIMEOUT_ENV])
        self.max_procs = int(max_procs or self.knobs["EVOSKILL_RECALC_CONCURRENCY"])
        self.max_heavy = int(max_heavy or self.knobs["EVOSKILL_HEAVY_COMPARE_CONCURRENCY"])
        self.heavy_bytes = int(self.knobs["EVOSKILL_HEAVY_COMPARE_BYTES"])
        self.job_timeout = int(job_timeout or 3 * (self.recalc_timeout + self.uno_timeout))
        if not 1 <= self.max_procs <= 8 or not 1 <= self.max_heavy <= self.max_procs:
            raise Refused(f"max_procs {self.max_procs} / max_heavy {self.max_heavy}: 1 <= heavy <= "
                          f"procs <= 8 (an 8 vCPU / 32 GiB sandbox; a heavy grade holds ~6 GB)")
        self.wall_min, self.max_usd = int(wall_min), float(max_usd)
        if self.wall_min < 15 or self.max_usd <= 0:
            raise Refused("wall_min must be >= 15 and max_usd > 0")
        self.ttl_minutes = self.wall_min + 30
        self.sync_s = int(sync_s)
        self.idx = idx if idx is not None else base.load_index()
        self.jobs_file = Path(jobs_file)
        self.all_jobs = normalize_jobs(read_jsonl(self.jobs_file), self.idx, self.data_overlay)
        self.results = self.out_dir / "results.jsonl"
        done = gw.done_ids(self.results)
        self.jobs = [j for j in self.all_jobs if j["job_id"] not in done]
        self.n_done_before = len(self.all_jobs) - len(self.jobs)
        for j in self.jobs:
            j["_heavy"] = gw.job_size(j) >= self.heavy_bytes
        self.has_stageb = any(j["mode"] == "stageB" for j in self.jobs)
        if self.has_stageb:
            if EVALUATION is None or not EVALUATION.is_file() or sha256_file(EVALUATION) != gw.EVALUATION_SHA256:
                raise Refused(f"{EVALUATION} is missing or not the pinned evaluation.py "
                              f"({gw.EVALUATION_SHA256[:12]}) -- Stage B cannot run")
        self.probe_task = probe_task or self._pick_probe()
        if self.jobs and self.probe_task not in self.idx:
            raise Refused(f"probe task {self.probe_task!r} is not a dataset task")

    def _pick_probe(self):
        for j in self.jobs:
            t = j.get("task_id")
            if t in self.idx and not base.is_heavy(self.idx[t]):
                return t
        if self.jobs:
            raise Refused("no light task among the jobs for the grading self-test: pass --probe-task")
        return None

    def _grader_clean_agreement(self) -> dict:
        m = self.run_dir / "grader_clean" / "MANIFEST.json"
        if not m.is_file():
            return {"manifest": None}
        try:
            files = json.loads(m.read_text()).get("files") or []
        except ValueError:
            return {"manifest": str(m), "error": "unreadable"}
        rows = {f.get("source"): f.get("source_sha256") for f in files if f.get("source")}
        return {"manifest": str(m),
                "agrees": all(sha256_file(REPO / s) == h for s, h in rows.items()
                              if (REPO / s).is_file()),
                "files": {s: h[:12] for s, h in rows.items() if h}}

    def estimate(self) -> dict:
        light = sum(1 for j in self.jobs if not j["_heavy"])
        heavy = len(self.jobs) - light
        secs = 600 + max(light * 60 / self.max_procs, heavy * 900 / self.max_heavy)
        return {"jobs": len(self.jobs), "light": light, "heavy": heavy,
                "sandbox_hours_est": round(secs / 3600, 2),
                "usd_est": round(secs / 3600 * USD_PER_HOUR, 2),
                "basis": "10 min setup + 60 s per light / 900 s per heavy grade (measured: "
                         "golden self-tests 21-72 s, heavy grades 712-720 s)"}

    def record(self) -> dict:
        return {"label": self.label, "run_dir": str(self.run_dir), "out_dir": str(self.out_dir),
                "jobs_file": str(self.jobs_file), "n_jobs": len(self.all_jobs),
                "n_done_before": self.n_done_before, "n_to_run": len(self.jobs),
                "modes": dict(Counter(j["mode"] for j in self.jobs)),
                "n_heavy": sum(1 for j in self.jobs if j["_heavy"]),
                "knobs": self.knobs, "infra": self.infra_record, "image_id": self.image_id,
                "data_overlay": data_overlay.record(self.data_overlay),
                "expected": {k: EXPECTED[k] for k in ("dockerfile_sha256", "snapshot_ref",
                                                      "libreoffice", "grading_openpyxl_prefix")},
                "code_sha256": self.code_sha256, "grader_clean": self.grader_clean,
                "max_procs": self.max_procs, "max_heavy": self.max_heavy,
                "heavy_bytes": self.heavy_bytes, "job_timeout_s": self.job_timeout,
                "recalc_timeout_s": self.recalc_timeout, "uno_timeout_s": self.uno_timeout,
                "wall_min": self.wall_min, "ttl_minutes": self.ttl_minutes,
                "max_usd": self.max_usd, "egress": self.egress, "probe_task": self.probe_task,
                "stageB": self.has_stageb, "estimate": self.estimate()}


def grade_env_text(knobs: dict, image_id: str) -> str:
    """/opt/leg/leg.env for the grading sandbox: the leg's knobs (INFRA's timeouts included), the
    thread env, the shim's image id / extra env. No solver settings, NO key."""
    env = dict(knobs)
    env.update(base.THREAD_ENV)
    env.update({"EVOSHIM_IMAGE_ID": image_id,
                "EVOSHIM_EXTRA_ENV": json.dumps(base.THREAD_ENV, sort_keys=True)})
    bad = [k for k in env if gw.SECRET_ENV_RE.search(k) or k.startswith("OPENROUTER")]
    assert not bad, bad
    return "".join(f"export {k}={shlex.quote(str(v))}\n" for k, v in sorted(env.items()))


def worker_args(cfg: GradeConfig) -> list:
    return ["run", "--jobs", SANDBOX_JOBS, "--results", f"{SANDBOX_OUT}/results.jsonl",
            "--work", SANDBOX_WORK, "--repo", str(REPO), "--scorer-module", gw.DEFAULT_SCORER_MODULE,
            "--evaluation", str(EVALUATION or ""), "--evaluation-sha256", gw.EVALUATION_SHA256,
            "--max-procs", str(cfg.max_procs), "--max-heavy", str(cfg.max_heavy),
            "--heavy-bytes", str(cfg.heavy_bytes), "--job-timeout", str(cfg.job_timeout),
            "--expect-scorer-sha256", GRADER_CODE_PINS[EXPECTED["baseline_runtime"]["scorer_sha256"][0]]]


def launch_script(cfg: GradeConfig) -> str:
    args = " ".join(shlex.quote(a) for a in worker_args(cfg))
    return f"""#!/bin/bash
# The grading worker, detached (written by scripts/daytona/ssb_grade_daytona.py). No key: the
# driver's exec calls carry no env, and `env -u` makes sure.
set -uo pipefail
OUT={SANDBOX_OUT}
mkdir -p "$OUT" {shlex.quote(SANDBOX_WORK)}
rm -f "$OUT/rc"
. /opt/leg/leg.env
echo $$ > /sys/fs/cgroup/evoshim_runner/cgroup.procs 2>/dev/null || echo "runner cgroup join failed" >> "$OUT/launch_notes.txt"
cd {shlex.quote(str(REPO))} || {{ echo 3 > "$OUT/rc"; exit 3; }}
env -u OPENROUTER_API_KEY /opt/evoskill/venv/bin/python -u {SANDBOX_WORKER} {args} > "$OUT/worker.log" 2>&1
RC=$?
echo "$RC" > "$OUT/rc"
exit "$RC"
"""


def build_bundle(cfg: GradeConfig, out: Path) -> dict:
    """src/ + 3 dataset.json + exactly the workbooks the jobs (and the self-test) need, mirrored at
    their Mac paths; the shim, the env, the worker, the job list and its launcher under /opt."""
    files, nbytes, workbooks = 0, 0, {}
    wanted = set()
    for j in cfg.jobs:
        for k in ("gt_file", "input_file", "output", "path"):
            if j.get(k):
                wanted.add(j[k])
    if cfg.probe_task:
        wanted.update({str(cfg.idx[cfg.probe_task]["golden"]), str(cfg.idx[cfg.probe_task]["input"])})
    jobs_text = "".join(json.dumps({k: v for k, v in j.items() if not k.startswith("_")}) + "\n"
                        for j in cfg.jobs)
    with tarfile.open(out, "w:gz") as tf:
        for p in sorted((REPO / "src").rglob("*")):
            if "__pycache__" in p.parts or p.suffix == ".pyc" or not p.is_file():
                continue
            if SECRET_NAME_RE.search(p.name):
                raise Refused(f"refusing to upload {p}: it looks like a secret file")
            nbytes += loop._add_file(tf, p, loop._mirror(p))
            files += 1
        for cat in CATEGORIES:
            loop._add_file(tf, DATA / cat / "dataset.json", loop._mirror(DATA / cat / "dataset.json"))
            files += 1
        for w in sorted(wanted):
            p = _workbook_ok(Path(w), "bundle workbook")
            nbytes += loop._add_file(tf, data_overlay.source(p, cfg.data_overlay), loop._mirror(p))
            workbooks[w] = sha256_file(data_overlay.source(p, cfg.data_overlay))
            files += 1
        if cfg.has_stageb:
            loop._add_file(tf, EVALUATION, loop._mirror(EVALUATION))
            files += 1
        loop._add_file(tf, HERE / "docker_shim.py", "opt/leg/docker_shim.py")
        loop._add_file(tf, HERE / "grade_worker.py", SANDBOX_WORKER.lstrip("/"))
        loop._add_bytes(tf, "opt/leg/leg.env", grade_env_text(cfg.knobs, cfg.image_id).encode())
        loop._add_bytes(tf, SANDBOX_JOBS.lstrip("/"), jobs_text.encode())
        loop._add_bytes(tf, "opt/grade/launch.sh", launch_script(cfg).encode())
        files += 5
    return {"files": files, "payload_bytes": nbytes, "bundle_bytes": out.stat().st_size,
            "bundle_sha256": sha256_file(out), "workbooks": workbooks,
            "jobs_sha256": hashlib.sha256(jobs_text.encode()).hexdigest(),
            "evaluation_uploaded": cfg.has_stageb}


# ───────────────────────────────────────────────────────────────────────────── host annotation
_OUTCOMES = None


def outcomes_mod():
    global _OUTCOMES
    if _OUTCOMES is None:
        spec = importlib.util.spec_from_file_location("evoskill_outcomes",
                                                      REPO / "src" / "orchestrator" / "outcomes.py")
        _OUTCOMES = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_OUTCOMES)
    return _OUTCOMES


def annotate(rec: dict, info: dict) -> dict:
    """The host block of a merged line. Grade: the outcome (outcomes.classify_detail) from the
    DIFF.txt text and the job's own WARNING lines; a grade that raised / timed out / crashed has no
    diff = grader failed. Caught-exception warnings are flagged (open_question), not judged."""
    h = dict(info, merged_utc=utcnow())
    st = rec.get("status")
    if rec.get("mode") == "grade":
        msgs = [w.get("message", "") for w in rec.get("warnings") or []]
        if st == "refused":
            h.update(outcome=None, outcome_reason="refused: " + str(rec.get("refused"))[:200])
        else:
            text = rec.get("diff_file_text") if st == "ok" else None
            oc, why = outcomes_mod().classify_detail(text, msgs)
            if st in ("exception", "timeout", "crashed") and why == "no_diff_recorded":
                why = st
            h.update(outcome=oc, outcome_reason=why)
        h["open_question"] = bool(rec.get("scorer_exception_warnings"))
    elif rec.get("mode") == "stageB":
        h["stageB_recalc_ok"] = (rec.get("recalc") or {}).get("ok")
    return h


def merge_results(local: Path, downloaded: Path, known_ids: set, info: dict) -> dict:
    """Append every COMPLETE line of the sandbox's results file whose job_id is new here. Lines are
    never rewritten; a job_id already present is skipped (the first result stands)."""
    rep = {"new": 0, "already": 0, "unknown": [], "torn": 0}
    if not downloaded.is_file():
        return rep
    local.parent.mkdir(parents=True, exist_ok=True)
    gw.repair_results(local)
    have = gw.done_ids(local)
    data = downloaded.read_bytes()
    if data and not data.endswith(b"\n"):
        data = data[: data.rfind(b"\n") + 1]
        rep["torn"] = 1
    with open(local, "a") as fh:
        for ln in data.decode("utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                rec = json.loads(ln)
            except ValueError:
                rep["torn"] += 1
                continue
            jid = rec.get("job_id")
            if jid in have:
                rep["already"] += 1
                continue
            if jid not in known_ids:
                rep["unknown"].append(jid)
                continue
            rec["host"] = annotate(rec, info)
            fh.write(json.dumps(rec, default=str, ensure_ascii=True) + "\n")
            have.add(jid)
            rep["new"] += 1
        fh.flush()
        os.fsync(fh.fileno())
    return rep


def summarize_results(path: Path, ids=None) -> dict:
    st, oc, modes = Counter(), Counter(), Counter()
    openq = 0
    if path.is_file():
        for ln in path.read_text().splitlines():
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if ids is not None and r.get("job_id") not in ids:
                continue
            st[r.get("status")] += 1
            modes[r.get("mode")] += 1
            h = r.get("host") or {}
            if r.get("mode") == "grade":
                oc[str(h.get("outcome"))] += 1
                openq += bool(h.get("open_question"))
    return {"status": dict(st), "grade_outcomes": dict(oc), "modes": dict(modes),
            "grades_with_scorer_exception_warnings": openq}


# ───────────────────────────────────────────────────────────────────────────── one run
class _ShardView:
    """What ssb_leg_daytona_loop.Shard.identity / .cgroups_and_selftest read from `self`, so the
    leg's own identity checks and self-tests run on this sandbox UNCHANGED (same `need` table, same
    failure patterns, same GRADE_SELFTEST)."""

    def __init__(self, run: "GradeRun"):
        self._run = run
        self.sb = run.sb
        self.summary = run.summary
        self.name = "grade"
        self.tasks = [run.cfg.probe_task]
        self.cfg = types.SimpleNamespace(image_id=run.cfg.image_id,
                                         rt={"require_libreoffice": EXPECTED["libreoffice"],
                                             "require_snapshot_ref": EXPECTED["snapshot_ref"]})

    def log(self, msg):
        self._run.log(msg)

    def sh(self, step, cmd, timeout=600, env=None, check=False):
        return self._run.sh(step, cmd, timeout=timeout, env=env, check=check)


class GradeRun:
    def __init__(self, cfg: GradeConfig, client):
        self.cfg, self.client = cfg, client
        self.stamp = time.strftime("%Y%m%dT%H%M%S")
        self.dir = cfg.out_dir / f"run_{self.stamp}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.logf = open(self.dir / "driver.log", "a")
        self.tag = f"grade_{cfg.label}"
        self.sid = None
        self.sb = None
        self.t0 = None
        self.abort = threading.Event()
        self.abort_reason = ""
        self.log_offset = 0
        self.summary: dict = {"driver": DRIVER, "stamp": self.stamp, "config": cfg.record()}

    def log(self, msg):
        line = f"{utcnow()} [grade/{self.cfg.label}] {msg}"
        print(line, flush=True)
        self.logf.write(line + "\n")
        self.logf.flush()

    def sh(self, step, cmd, timeout=600, env=None, check=False):
        if env:
            raise Abort(f"{step}: no exec call of the grading sandbox carries an environment")
        rc, out, _dt = dlx.run(self.sb, loop._tagged(step, cmd), timeout=timeout, env=None)
        if check and rc != 0:
            raise Abort(f"{step} failed rc={rc}: {out[-1500:]}")
        return rc, out

    def save(self):
        (self.dir / "summary.json").write_text(json.dumps(self.summary, indent=1, default=str))

    # -- guard
    def cost_usd(self) -> float:
        return 0.0 if self.t0 is None else round((time.time() - self.t0) / 3600 * USD_PER_HOUR, 4)

    def breach(self) -> str:
        if self.abort.is_set():
            return self.abort_reason or "abort requested"
        if self.t0 is None:
            return ""
        if time.time() - self.t0 > self.cfg.wall_min * 60:
            return f"hard wall {self.cfg.wall_min} min reached"
        if self.cost_usd() > self.cfg.max_usd:
            return f"spend guard: ${self.cost_usd():.3f} > ${self.cfg.max_usd}"
        return ""

    def guard(self, where: str):
        why = self.breach()
        if why:
            raise Abort(f"{why} (at {where})")

    # -- steps
    def create(self):
        from daytona import CreateSandboxFromImageParams, Image, Resources
        for attempt in range(3):
            name = f"{PREFIX}{self.cfg.label}-{time.strftime('%H%M%S')}"
            net = ({"network_block_all": True} if self.cfg.egress == "block-all"
                   else {"domain_allow_list": UNRESOLVABLE})
            params = CreateSandboxFromImageParams(
                name=name, image=Image.from_dockerfile(str(DOCKERFILE)),
                resources=Resources(**RESOURCES), ephemeral=True, auto_stop_interval=0,
                ttl_minutes=self.cfg.ttl_minutes, public=False, labels={"role": "grade"}, **net)
            self.log(f"creating {name} ({RESOURCES}, ttl {self.cfg.ttl_minutes} min, egress "
                     f"{self.cfg.egress} {net})")
            if self.t0 is None:
                self.t0 = time.time()
            with open(self.dir / "image_build.log", "a") as blog:
                sb, sid, err, dt = create_sandbox(
                    self.client, params, RESOURCES, tag=self.tag, timeout=3600,
                    on_logs=lambda c: (blog.write(c + "\n"), blog.flush()))
            self.sid = sid
            self.summary.update(sandbox_id=sid, sandbox_name=name, create_s=dt)
            if err is None:
                self.sb = sb
                break
            msg = str(err).lower()
            transient = any(w in msg for w in ("quota", "limit", "exceed", "capacity",
                                               "insufficient", "no available", "429", "timed out"))
            if not transient or attempt == 2 or self.breach():
                raise Abort(f"create failed after {dt}s: {type(err).__name__}: {str(err)[:600]}")
            self.log(f"create refused (attempt {attempt + 1}/3), retry in 60s: {str(err)[:300]}")
            if sid:
                delete_by_id(self.client, sid, self.tag)
                self.sid = None
            time.sleep(60)
        rb = dlx.readback(self.sb)
        self.summary["readback"] = rb
        self.log(f"created {self.sid} readback cpu={rb['cpu']} mem={rb['memory']} disk={rb['disk']} "
                 f"snapshot_ref={rb.get('snapshot_ref')} network_block_all={rb.get('network_block_all')} "
                 f"domain_allow_list={rb.get('domain_allow_list')}")
        mism = {k: (RESOURCES[k], rb.get(k)) for k in RESOURCES if rb.get(k) != RESOURCES[k]}
        if mism:
            raise Abort(f"resource readback mismatch {mism}")
        if EXPECTED["snapshot_ref"] and rb.get("snapshot_ref") != EXPECTED["snapshot_ref"]:
            raise Abort(f"Daytona snapshot {rb.get('snapshot_ref')} != the baseline's "
                        f"{EXPECTED['snapshot_ref']} -- not the leg image")
        if self.cfg.egress == "block-all" and rb.get("network_block_all") is not True:
            raise Abort(f"network_block_all reads back {rb.get('network_block_all')!r}, not True -- "
                        f"egress is not blocked (rerun with --egress unresolvable-allow-list only "
                        f"deliberately)")
        if self.cfg.egress == "unresolvable-allow-list" and rb.get("domain_allow_list") != UNRESOLVABLE:
            raise Abort(f"domain_allow_list reads back {rb.get('domain_allow_list')!r}")

    def egress_check(self):
        rc, out = self.sh("egress", base.EGRESS_CMD, timeout=180)
        m = re.search(r"^EGRESS (.*)$", out, re.M)
        if not m:
            raise Abort("egress probe failed: " + out[-800:])
        eg = json.loads(m.group(1))
        self.summary["egress"] = {"mode": self.cfg.egress, "probe": eg,
                                  "network_block_all": (self.summary.get("readback") or {}).get(
                                      "network_block_all")}
        self.log("egress " + json.dumps(eg))
        reached = [k for k, v in eg.items() if v.get("ok")]
        if reached or not eg:
            raise Abort(f"egress NOT blocked: reachable {reached or 'probe empty'}")

    def upload(self):
        b = self.dir / "bundle.tgz"
        man = build_bundle(self.cfg, b)
        self.summary["bundle"] = {k: v for k, v in man.items() if k != "workbooks"}
        self.summary["bundle"]["n_workbooks"] = len(man["workbooks"])
        (self.dir / "bundle_workbooks.json").write_text(json.dumps(man["workbooks"], indent=1))
        t = time.time()
        self.sb.fs.upload_file(str(b), "/tmp/bundle.tgz", timeout=1800)
        self.log(f"uploaded bundle {man['bundle_bytes'] / 1e6:.1f} MB ({man['files']} files, "
                 f"{len(man['workbooks'])} workbooks) in {time.time() - t:.0f}s")
        b.unlink()
        self.sh("unpack", "tar xzf /tmp/bundle.tgz -C / && rm /tmp/bundle.tgz && "
                "install -m 0755 /opt/leg/docker_shim.py /usr/local/bin/docker && "
                "mkdir -p /opt/leg/out /var/log/evoshim /var/lib/evoshim/state /var/lib/evoshim/priv "
                f"{SANDBOX_OUT} {shlex.quote(str(REPO))}/.cache && "
                "which docker && sha256sum /usr/local/bin/docker", timeout=600, check=True)
        shim = GRADER_CODE_PINS["scripts/daytona/docker_shim.py"]
        paths = {str(REPO / rel): h for rel, h in GRADER_CODE_PINS.items() if rel.startswith("src/")}
        paths["/opt/leg/docker_shim.py"] = shim            # uploaded as a leg uploads it ...
        paths["/usr/local/bin/docker"] = shim              # ... and installed as a leg installs it
        paths[SANDBOX_WORKER] = sha256_file(HERE / "grade_worker.py")
        paths[SANDBOX_JOBS] = man["jobs_sha256"]
        if self.cfg.has_stageb:
            paths[str(EVALUATION)] = gw.EVALUATION_SHA256
        rc, out = self.sh("shas", "sha256sum " + " ".join(shlex.quote(p) for p in paths), timeout=120)
        got = {}
        for ln in out.splitlines():
            parts = ln.split(None, 1)
            if len(parts) == 2 and re.fullmatch(r"[0-9a-f]{64}", parts[0]):
                got[parts[1].strip()] = parts[0]
        bad = {p: (got.get(p, "missing")[:12], h[:12]) for p, h in paths.items() if got.get(p) != h}
        self.summary["sandbox_code_sha256"] = got
        if bad:
            raise Abort(f"code in the sandbox differs from the pins: {bad}")

    def selftests(self):
        view = _ShardView(self)
        loop.Shard.identity(view)
        self.guard("identity")
        self.egress_check()
        self.guard("egress")
        self.upload()
        self.guard("upload")
        loop.Shard.cgroups_and_selftest(view)

    def launch(self):
        rc, out = self.sh("launch", "(setsid nohup bash /opt/grade/launch.sh > "
                          f"{SANDBOX_OUT}/launch.log 2>&1 < /dev/null &) ; sleep 5; "
                          f"ls {SANDBOX_OUT}; tail -3 {SANDBOX_OUT}/worker.log 2>/dev/null",
                          timeout=120)
        self.summary["launched_utc"] = utcnow()
        self.log("worker launched: " + out.strip()[-400:].replace("\n", " | "))

    def poll(self) -> str:
        rc, out = self.sh("poll", f"cat {SANDBOX_OUT}/rc 2>/dev/null || echo RUNNING; "
                          f"echo LINES=$(cat {SANDBOX_OUT}/results.jsonl 2>/dev/null | wc -l); "
                          f"echo SIZE=$(stat -c %s {SANDBOX_OUT}/worker.log 2>/dev/null || echo 0)",
                          timeout=120)
        lines = out.strip().splitlines()
        state = lines[0].strip() if lines else "?"
        kv = dict(ln.split("=", 1) for ln in lines[1:] if "=" in ln)
        size = int((kv.get("SIZE") or "0").strip() or 0)
        if size > self.log_offset:
            _rc, chunk = self.sh("logtail", f"tail -c +{self.log_offset + 1} {SANDBOX_OUT}/worker.log "
                                 f"| head -c 1000000", timeout=120)
            with open(self.dir / "worker.log", "a") as fh:
                fh.write(chunk)
            self.log_offset += len(chunk.encode())
        self.summary["progress"] = {"lines": (kv.get("LINES") or "").strip(), "of": len(self.cfg.jobs),
                                    "state": state, "cost_usd": self.cost_usd()}
        self.log(f"progress {kv.get('LINES', '?').strip()}/{len(self.cfg.jobs)} | state {state} | "
                 f"${self.cost_usd():.3f}")
        return state

    def sync_results(self) -> dict:
        local = self.dir / "results.sandbox.jsonl"
        try:
            self.sb.fs.download_file(f"{SANDBOX_OUT}/results.jsonl", str(local))
        except Exception as e:  # noqa: BLE001 — nothing written yet is not an error
            self.log(f"results download: {type(e).__name__}: {str(e)[:160]}")
            return {}
        rep = merge_results(self.cfg.results, local, {j["job_id"] for j in self.cfg.all_jobs},
                            {"sandbox_id": self.sid, "run": self.stamp, "label": self.cfg.label})
        self.log(f"results merged: {rep}")
        return rep

    def kill_worker(self):
        self.sh("kill", "pkill -TERM -f '[g]rade_worker.py run'; sleep 12; "
                "pkill -KILL -f '[g]rade_worker.py'; echo killed", timeout=120)

    def final_sync(self):
        self.summary["merge_final"] = self.sync_results()
        spec = {"out": "/tmp/grade_logs.tgz"}
        rc, out = self.sh("pack", f"# {json.dumps(spec)}\ncd / && "
                          f"find opt/grade/out var/log/evoshim {SANDBOX_WORK.lstrip('/')} -type f "
                          r"\( -name '*.log' -o -name '*.json' -o -name '*.jsonl' -o -name '*.txt' \) "
                          "-print0 2>/dev/null | tar czf /tmp/grade_logs.tgz --null -T - && "
                          "ls -la /tmp/grade_logs.tgz", timeout=600)
        local = self.dir / "grade_logs.tgz"
        try:
            self.sb.fs.download_file("/tmp/grade_logs.tgz", str(local))
            with tarfile.open(local) as tf:
                for m in tf.getmembers():
                    if m.isfile() and not m.name.lower().endswith(WORKBOOK_SUFFIXES):
                        tf.extract(m, self.dir / "sandbox", filter="data")
            local.unlink()
        except Exception as e:  # noqa: BLE001
            self.log(f"log sync failed: {type(e).__name__}: {str(e)[:200]}")
        meta = self.dir / "sandbox" / "opt" / "grade" / "out" / "worker_meta.json"
        if meta.is_file():
            wm = json.loads(meta.read_text())
            self.summary["worker_meta"] = {k: wm.get(k) for k in (
                "worker", "worker_sha256", "python", "openpyxl", "code_sha256", "evaluation_sha256",
                "env", "max_procs", "max_heavy", "job_timeout_s", "n_jobs", "n_skipped_done",
                "n_todo", "statuses", "interrupted", "max_heavy_concurrent_seen", "nproc")}

    def secret_scan(self):
        hits = []
        for p in self.dir.rglob("*"):
            if p.is_file():
                try:
                    if KEYLIKE_RE.search(p.read_bytes()):
                        hits.append(str(p))
                except OSError:
                    pass
        self.summary["keylike_hits"] = hits

    # -- orchestration
    def run(self) -> int:
        cfg = self.cfg
        (cfg.out_dir / "jobs_normalized.jsonl").write_text(
            "".join(json.dumps({k: v for k, v in j.items() if not k.startswith("_")}) + "\n"
                    for j in cfg.all_jobs))
        self.log(f"{len(cfg.jobs)} job(s) to run ({cfg.n_done_before} already done); estimate "
                 f"{cfg.estimate()}")
        try:
            self.create()
            self.guard("create")
            self.selftests()
            self.guard("self-tests")
            self.launch()
            last_sync = time.time()
            while True:
                why = self.breach()
                if why:
                    self.log(f"ABORT: {why} -- killing the worker")
                    self.kill_worker()
                    self.summary["aborted"] = why
                    break
                if self.poll() != "RUNNING":
                    break
                if time.time() - last_sync >= cfg.sync_s:
                    self.sync_results()
                    last_sync = time.time()
                self.save()
                time.sleep(POLL_S)
            self.final_sync()
        except Abort as e:
            self.summary["error"] = str(e)
            self.log(f"ABORT: {e}")
            if self.sb is not None and self.summary.get("launched_utc"):
                try:
                    self.kill_worker()
                    self.final_sync()
                except Exception as e2:  # noqa: BLE001
                    self.log(f"sync after abort failed: {e2}")
        except Exception as e:  # noqa: BLE001
            self.summary["error"] = f"{type(e).__name__}: {e}"
            self.log("CRASH: " + traceback.format_exc()[-2000:])
            if self.sb is not None and self.summary.get("launched_utc"):
                try:
                    self.final_sync()
                except Exception as e2:  # noqa: BLE001
                    self.log(f"sync after crash failed: {e2}")
        finally:
            if self.sid:
                ok = delete_by_id(self.client, self.sid, self.tag)
                self.summary["deleted"] = ok
                self.log(f"DELETED {self.sid}: {ok}")
            self.summary["wall_s"] = round(time.time() - self.t0, 1) if self.t0 else 0
            self.summary["cost_usd_est"] = self.cost_usd()
            ids = {j["job_id"] for j in cfg.jobs}
            self.summary["results_this_run"] = summarize_results(cfg.results, ids)
            have = gw.done_ids(cfg.results)
            missing = sorted(ids - have)
            self.summary["missing_after_run"] = len(missing)
            self.secret_scan()
            self.save()
            ledger(event="grade_run_end", tag=self.tag, id=self.sid, label=cfg.label,
                   n_to_run=len(ids), n_missing=len(missing), deleted=self.summary.get("deleted"),
                   wall_s=self.summary["wall_s"], cost_usd_est=self.summary["cost_usd_est"],
                   error=self.summary.get("error"), aborted=self.summary.get("aborted"),
                   egress=cfg.egress)
            self.logf.close()
        if self.summary.get("error") or self.summary.get("aborted") or missing:
            return EXIT_INCOMPLETE
        return EXIT_OK


# ───────────────────────────────────────────────────────────────────────────── commands
def cmd_make_jobs(args) -> int:
    idx = base.load_index()
    tasks = [t.strip() for t in (args.tasks or "").split(",") if t.strip()]
    if args.key:
        split = Path(args.split) if args.split else loop.default_split(Path(args.run_dir) if args.run_dir else None)
        allowed = set(json.loads(split.read_text())[args.key])
        off = [t for t in tasks if t not in allowed]
        if off:
            raise Refused(f"not in split[{args.key}]: {off}")
    saved = json.loads(Path(args.saved_json).read_text()) if args.saved_json else []
    if args.key:
        off = [s.get("task") for s in saved if s.get("task") not in allowed]
        if off:
            raise Refused(f"saved outputs of tasks not in split[{args.key}]: {off}")
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    jobs = normalize_jobs(make_jobs(tasks, kinds, saved, idx), idx)
    Path(args.out).write_text("".join(json.dumps(j) + "\n" for j in jobs))
    print(json.dumps({"out": args.out, "jobs": len(jobs),
                      "modes": dict(Counter(j["mode"] for j in jobs))}))
    return EXIT_OK


def _cfg(args) -> GradeConfig:
    return GradeConfig(jobs_file=args.jobs, out_dir=args.out_dir, run_dir=args.run_dir,
                       label=args.label, max_procs=args.max_procs, max_heavy=args.max_heavy,
                       job_timeout=args.job_timeout, wall_min=args.wall_min, max_usd=args.max_usd,
                       egress=args.egress, probe_task=args.probe_task, sync_s=args.sync_s)


def cmd_run(args) -> int:
    cfg = _cfg(args)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    rec = cfg.record()
    if args.check_only:
        print(json.dumps(rec, indent=1, default=str))
        print(f"CHECK-ONLY: {len(cfg.jobs)} job(s) would run; nothing created, no key read")
        return EXIT_CHECK_ONLY
    if not cfg.jobs:
        print(f"nothing to do: all {len(cfg.all_jobs)} job(s) already have a result in {cfg.results}")
        return EXIT_OK
    if rec["estimate"]["usd_est"] > cfg.max_usd:
        print(f"WARNING: estimate ${rec['estimate']['usd_est']} exceeds --max-usd {cfg.max_usd}; the "
              f"spend guard will stop the run there (results so far are kept; rerun resumes)")
    run = GradeRun(cfg, dlx.client())

    def on_sig(*_a):
        run.abort_reason = "SIGINT/SIGTERM"
        run.abort.set()
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)
    rc = run.run()
    print(json.dumps({"run_dir": str(run.dir), "sandbox_id": run.sid,
                      "deleted": run.summary.get("deleted"),
                      "results": run.summary.get("results_this_run"),
                      "missing": run.summary.get("missing_after_run"),
                      "error": run.summary.get("error"), "aborted": run.summary.get("aborted"),
                      "cost_usd_est": run.summary.get("cost_usd_est")}, indent=1, default=str))
    return rc


def cmd_status(args) -> int:
    res = Path(args.out_dir) / "results.jsonl"
    ids = None
    if args.jobs:
        ids = {j["job_id"] for j in read_jsonl(Path(args.jobs))}
    s = summarize_results(res, ids)
    if ids is not None:
        s["missing"] = len(ids - gw.done_ids(res))
    print(json.dumps(s, indent=1))
    return EXIT_OK


def cmd_regrade_jobs(args) -> int:
    res = Path(args.results)
    norm = res.with_name("jobs_normalized.jsonl")
    paths = {j["job_id"]: j.get("output") for j in read_jsonl(norm)} if norm.is_file() else {}
    rows = []
    for r in read_jsonl(res):
        if r.get("mode") == "grade" and (r.get("host") or {}).get("outcome") == "grader_failed":
            if r["job_id"] not in paths:
                raise Refused(f"{r['job_id']}: not in {norm}")
            rows.append({"job_id": r["job_id"] + args.suffix, "mode": "grade",
                         "task_id": r["task_id"], "output": paths[r["job_id"]],
                         "meta": dict(r.get("meta") or {}, regrade_of=r["job_id"])})
    Path(args.out).write_text("".join(json.dumps(j) + "\n" for j in rows))
    print(json.dumps({"out": args.out, "regrade_jobs": len(rows)}))
    return EXIT_OK


def ledger_open_sandboxes(path) -> list:
    """Grade sandboxes this driver ledgered as created (or seen) and never verified deleted."""
    created, gone = {}, set()
    p = Path(path)
    if not p.is_file():
        return []
    for ln in p.read_text().splitlines():
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        if r.get("driver") != DRIVER or not r.get("id"):
            continue
        if r.get("event") in ("created", "id_seen_during_create", "create_failed"):
            created.setdefault(r["id"], r.get("name"))
        if r.get("event") == "deleted":
            gone.add(r["id"])
    return [{"id": i, "name": n} for i, n in created.items() if i not in gone]


def cmd_cleanup(args) -> int:
    rows = ledger_open_sandboxes(dlx.LEDGER)
    print(json.dumps(rows, indent=1))
    if not rows or not args.force:
        if rows:
            print("(listed only; --force deletes them by id)")
        return EXIT_OK
    d = dlx.client()
    for r in rows:
        print(r["id"], delete_by_id(d, r["id"], "grade_cleanup"))
    return EXIT_OK


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make-jobs")
    p.add_argument("--tasks", default="")
    p.add_argument("--kinds", default="golden,input,open,stageB_golden")
    p.add_argument("--saved-json", default=None)
    p.add_argument("--split", default=None, help="default: --run-dir's SPEC.json split_path, else "
                   f"{DEFAULT_SPLIT.relative_to(REPO)}")
    p.add_argument("--run-dir", default=None, help="whose declared split --key reads")
    p.add_argument("--key", default=None)
    p.add_argument("--out", required=True)
    p = sub.add_parser("run")
    p.add_argument("--jobs", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-dir", required=True, help="the run whose INFRA.json names the grader timeouts")
    p.add_argument("--label", default="grade")
    p.add_argument("--max-procs", type=int, default=0)
    p.add_argument("--max-heavy", type=int, default=0)
    p.add_argument("--job-timeout", type=int, default=0)
    p.add_argument("--wall-min", type=int, default=120)
    p.add_argument("--max-usd", type=float, default=3.0)
    p.add_argument("--egress", default="block-all", choices=("block-all", "unresolvable-allow-list"))
    p.add_argument("--probe-task", default=None)
    p.add_argument("--sync-s", type=int, default=300)
    p.add_argument("--check-only", action="store_true")
    p = sub.add_parser("status")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--jobs", default=None)
    p = sub.add_parser("regrade-jobs")
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--suffix", default="#regrade1")
    p = sub.add_parser("cleanup")
    p.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    try:
        return {"make-jobs": cmd_make_jobs, "run": cmd_run, "status": cmd_status,
                "regrade-jobs": cmd_regrade_jobs, "cleanup": cmd_cleanup}[args.cmd](args)
    except Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
