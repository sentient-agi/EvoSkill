#!/usr/bin/env python3
"""Grade saved SSB v2 workbooks INSIDE the grading sandbox, one child process per job.

Runs under the leg image's runner interpreter (/opt/evoskill/venv/bin/python: CPython 3.13.9,
openpyxl 3.1.5 + lxml), launched by scripts/daytona/ssb_grade_daytona.py with the leg's knobs
sourced (INFRA.json's grader timeouts included) and NO model key in the environment. Standard
library only; the grader (src/agent_profiles/ssbv2_agent/scorer.py, frozen) is imported in the
child, never in the parent.

  grade_worker.py run   --jobs JOBS.jsonl --results RESULTS.jsonl --work DIR --repo R
                        [--max-procs 4] [--max-heavy 2] [--heavy-bytes 5000000] [--job-timeout S]
                        [--expect-scorer-sha256 H] [--evaluation PATH] [--keep-copies]
  grade_worker.py child --job-file J --result-file R --repo R ...   (internal: one job)

WHY ONE CHILD PER JOB. The scorer reports its failures only as WARNING log lines (a failed
LibreOffice recalculation, a failed data-table pass, an input workbook openpyxl cannot open) and
carries on. In a shared process those lines interleave across concurrent grades and cannot be
pinned to one workbook (a leg's shared runner.log has exactly that problem). Here every
job gets a fresh interpreter whose root logger is captured, so every WARNING line in a result
belongs to that job. The scorer's own concurrency semaphores are per process, so the parent caps
concurrency itself: at most --max-procs children (default: the leg's EVOSKILL_RECALC_CONCURRENCY)
and at most --max-heavy HEAVY ones (default: EVOSKILL_HEAVY_COMPARE_CONCURRENCY), where heavy
means the reference or the graded workbook is >= --heavy-bytes (EVOSKILL_HEAVY_COMPARE_BYTES) --
the same rule the leg's heavy-grade gate applies (scripts/daytona/sandbox_entry.py).

JOB MODES (one JSON object per line of JOBS.jsonl; every path absolute; shas are sha256 hex):
  grade   {"job_id","mode":"grade","task_id","gt_file","gt_sha256","input_file","input_sha256",
           "answer_position","output","output_sha256"}
          A FRESH copy of `output` (shutil.copy2, into <work>/<job>/output/<its own name>) is
          graded by score_and_diff(gt_file, copy, answer_position, orig_input_file=input_file),
          the call the SSB runner makes. Recorded: passed, n_match, n_total, the diff text and its
          sha256, the DIFF.txt text the runner would have written (diff_file_text) and its sha256,
          every WARNING line of the job (labelled by kind, see WARNING_KINDS), any exception with
          its traceback, and the
          sha256 of the source, reference and input before and after (the source must never
          change: only the copy is graded).
  stageB  {"job_id","mode":"stageB","task_id","gt_file","gt_sha256","answer_position","output",
           "output_sha256"}
          A fresh copy of `output` is recalculated by the image's recalc-always LibreOffice
          profile and NOTHING else of ours: the scorer's own `docker run ssb-verifier:latest ...
          --convert-to xlsx` command line (scorer.py:107-115), which the docker shim runs with a
          private copy of /opt/evoskill/verifier-lo-profile (OOXMLRecalcMode=0). No _xlfn fix, no
          iterative-calc switch, no data-table solve, no restore, no volatile skip, no golden
          refresh. The recalculated copy is then compared with the ORIGINAL reference by the
          official rules: SpreadsheetBench-v2/evaluation.py, loaded by path after its sha256 is
          checked (tqdm stubbed if absent; it is used only in evaluation()). Two columns:
            passed          evaluation.compare_workbooks with ONE change -- answer_position split
                            on commas outside single quotes (the official split breaks sheet names
                            that contain a comma); every other step is the file's own code;
            verbatim_passed evaluation.compare_workbooks unchanged, wrapped in try/except exactly
                            as evaluation():216-219 does.
          Reported beside our grader's number, never the verdict.
  open    {"job_id","mode":"open","path","path_sha256"}
          Can openpyxl open (a fresh copy of) the workbook with data_only=True and with
          data_only=False? Each attempt's error is recorded.
  Any job may carry "meta": {...}; it is copied into the result unread.

RESULTS.jsonl: one line per job, appended (and fsynced) as each job ends. status is one of
  ok        the grader / comparison / open attempts ran to completion (a grade that the scorer
            completed with failure warnings is still "ok" here: the warnings are in the line, and
            the host classifies the four outcomes);
  exception the grader (or the comparison) raised -- type, message, traceback recorded;
  timeout   the child exceeded --job-timeout and was killed with its process group;
  crashed   the child died without writing its result (rc recorded; e.g. an OOM kill);
  refused   a precondition failed before any grading (a sha mismatch, a missing file).
RESUME: a job whose job_id already has a line in RESULTS.jsonl is skipped. A torn last line (the
sandbox died mid-write) is moved aside first. A job interrupted by SIGTERM writes no line, so a
rerun grades it again.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import importlib
import importlib.util
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import types
from collections import Counter, deque
from pathlib import Path

WORKER_VERSION = "grade_worker/1"
MODES = ("grade", "stageB", "open")
STATUSES = ("ok", "exception", "timeout", "crashed", "refused")
# No default repo or evaluation.py path: both sit where the driver mirrored them on the machine (the host's own paths),
# so the driver passes them (--repo; --evaluation only when a stageB job needs it: without it such a job is refused).
DEFAULT_SCORER_MODULE = "src.agent_profiles.ssbv2_agent.scorer"
DEFAULT_EVALUATION = ""
# sha256 of SpreadsheetBench-v2/evaluation.py as hashed for the design (plugin_design/design.md s5)
EVALUATION_SHA256 = "4ae77cee8df01d1f34684fceab972810d696886533d33be2e89373de6b4d3de3"
JOB_ID_RE = re.compile(r"[A-Za-z0-9._:/=+@#-]{1,240}")
SECRET_ENV_RE = re.compile(r"(API_KEY|_TOKEN|SECRET|PASSWORD|OAUTH)", re.I)
LOG_TAIL = 4000

# The grader-failure warnings (mirror of src/orchestrator/outcomes.py
# GRADER_FAILURE_PATTERNS; tests/test_grade_worker.py pins the two equal).
GRADER_FAILURE_PATTERNS = (
    "LibreOffice recalculation failed",
    "_solve_datatables_with_uno failed",
    "Inline SSBv2 scoring failed",
)

# Every WARNING the frozen scorer can log, labelled (message prefix -> kind, class). The class is
# DESCRIPTIVE, not a verdict: a scorer_exception is not counted as "grader failed", so each line is recorded
# with its label and the decision stays with whoever reads it.
#   d4_failure        a grader-failure warning (counts as grader failed);
#   scorer_exception  the scorer caught an exception and carried on (not counted);
#   scorer_decision   a designed rollback/refusal with no exception.
# tests/test_grade_worker.py checks every logger.warning in scorer.py maps to a kind here.
WARNING_KINDS = (
    ("LibreOffice recalculation failed", "recalc_failed", "d4_failure"),
    ("_solve_datatables_with_uno failed", "uno_failed", "d4_failure"),
    ("Inline SSBv2 scoring failed", "inline_scoring_failed", "d4_failure"),
    ("_solve_datatables_with_uno: pre-check failed", "uno_precheck_failed", "scorer_exception"),
    ("_solve_datatables_with_uno: golden-aware check setup failed", "uno_golden_aware_setup_failed",
     "scorer_exception"),
    ("_solve_datatables_with_uno: pre-snapshot failed", "uno_pre_snapshot_failed", "scorer_exception"),
    ("_solve_datatables_with_uno: post-snapshot failed", "uno_post_snapshot_failed",
     "scorer_exception"),
    ("_add_xlfn_prefixes: could not open", "xlfn_open_failed", "scorer_exception"),
    ("_force_iterative_calc: iterative calculation NOT applied", "iterative_calc_not_applied",
     "scorer_exception"),
    ("_restore_unrecalculable_cells: could not open workbooks", "restore_open_failed",
     "scorer_exception"),
    ("_volatile_function_skip_cells: could not open workbooks", "volatile_open_failed",
     "scorer_exception"),
    ("_datatable_dependency_safe: exception tracing dependencies", "dependency_trace_exception",
     "scorer_exception"),
    ("_build_hybrid_golden_with_original_datatable: failed", "hybrid_golden_failed",
     "scorer_exception"),
    ("_refresh_golden_datatables: failed for", "golden_refresh_failed", "scorer_exception"),
    ("_refresh_golden_datatables: original-fallback dependency check failed",
     "golden_refresh_dependency_check_failed", "scorer_exception"),
    ("_refresh_golden_datatables: could not read computed hybrid golden",
     "golden_refresh_read_failed", "scorer_exception"),
    ("_refresh_golden_datatables: could not write cached scratch copy",
     "golden_scratch_write_failed", "scorer_exception"),
    ("_refresh_golden_datatables: golden has no live Data Table and the dependency-safety check "
     "refused", "golden_refresh_refused", "scorer_decision"),
)
_REGRESSED_RE = re.compile(r"^_solve_datatables_with_uno: \d+ graded cell\(s\) regressed")
_NONTARGET_RE = re.compile(r"^_solve_datatables_with_uno: detected \d+ changed non-target")

# Stage B's recalculation: the scorer's own command line (scorer.py:107-115, pinned by a test),
# which the docker shim runs in the verifier profile. {name} is the workbook's file name.
STAGEB_IMAGE = "ssb-verifier:latest"
STAGEB_BASH = ('libreoffice --headless --calc --convert-to "xlsx:Calc MS Excel 2007 XML" '
               '--outdir /tmp "{name}" 2>/dev/null && cp "/tmp/{name}" "{name}"')


# ───────────────────────────────────────────────────────────────────────────── helpers
def utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z"


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def safe_name(job_id: str) -> str:
    """A work-dir name for a job: readable prefix + a hash, so two ids never collide."""
    return (re.sub(r"[^A-Za-z0-9._-]", "_", job_id)[:80] + "-"
            + hashlib.sha256(job_id.encode()).hexdigest()[:10])


def quote_aware_split(answer_position: str) -> list[str]:
    """answer_position split on commas OUTSIDE single quotes; each part otherwise VERBATIM (no
    strip, unlike scorer._split_answer_position): the only change made to the official
    `answer_position.split(',')`, so a sheet name like 'IS,BS,CF' stays one range."""
    parts, cur, in_q = [], "", False
    for ch in answer_position:
        if ch == "'":
            in_q = not in_q
            cur += ch
        elif ch == "," and not in_q:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return parts


def warning_kind(logger_name: str, message: str) -> tuple[str, str]:
    if logger_name == "py.warnings":
        return "python_warning", "python_warning"
    for prefix, kind, cls in WARNING_KINDS:
        if message.startswith(prefix):
            return kind, cls
    if _REGRESSED_RE.match(message):
        return "uno_regression_rollback", "scorer_decision"
    if _NONTARGET_RE.match(message):
        return "uno_nontarget_rollback", "scorer_decision"
    if any(p in message for p in GRADER_FAILURE_PATTERNS):
        return "d4_pattern_elsewhere", "d4_failure"
    return "other", "other"


def scrubbed_env(env=None) -> dict:
    env = dict(os.environ if env is None else env)
    return {k: v for k, v in env.items() if not SECRET_ENV_RE.search(k)}


class JobError(ValueError):
    pass


def validate_job(j) -> dict:
    if not isinstance(j, dict):
        raise JobError("a job must be a JSON object")
    jid = j.get("job_id")
    if not isinstance(jid, str) or not JOB_ID_RE.fullmatch(jid):
        raise JobError(f"bad job_id {jid!r}")
    mode = j.get("mode")
    if mode not in MODES:
        raise JobError(f"{jid}: mode {mode!r} not in {MODES}")
    need = {"grade": ("task_id", "gt_file", "input_file", "answer_position", "output",
                      "output_sha256"),
            "stageB": ("task_id", "gt_file", "answer_position", "output", "output_sha256"),
            "open": ("path", "path_sha256")}[mode]
    miss = [k for k in need if not isinstance(j.get(k), str) or not j.get(k)]
    if miss:
        raise JobError(f"{jid}: missing {miss}")
    for k in ("gt_file", "input_file", "output", "path"):
        if k in j and not os.path.isabs(j[k]):
            raise JobError(f"{jid}: {k} must be absolute")
    if "meta" in j and not isinstance(j["meta"], dict):
        raise JobError(f"{jid}: meta must be an object")
    return j


def load_jobs(path) -> list[dict]:
    jobs, seen = [], set()
    for n, ln in enumerate(Path(path).read_text().splitlines(), 1):
        if not ln.strip():
            continue
        try:
            j = validate_job(json.loads(ln))
        except (ValueError, JobError) as e:
            raise JobError(f"{path}:{n}: {e}")
        if j["job_id"] in seen:
            raise JobError(f"{path}:{n}: duplicate job_id {j['job_id']!r}")
        seen.add(j["job_id"])
        jobs.append(j)
    return jobs


def repair_results(path: Path) -> str | None:
    """A results file whose last line is torn (no trailing newline) gets that fragment moved to
    <file>.torn.<stamp>, so the next append starts on a fresh line. Returns the fragment's file."""
    if not path.is_file() or path.stat().st_size == 0:
        return None
    data = path.read_bytes()
    if data.endswith(b"\n"):
        return None
    cut = data.rfind(b"\n") + 1
    torn = path.with_name(f"{path.name}.torn.{time.strftime('%Y%m%dT%H%M%S')}")
    torn.write_bytes(data[cut:])
    with open(path, "r+b") as fh:
        fh.truncate(cut)
    return str(torn)


def done_ids(path: Path) -> set:
    out = set()
    if not path.is_file():
        return out
    for ln in path.read_text().splitlines():
        try:
            d = json.loads(ln)
        except ValueError:
            continue
        if isinstance(d, dict) and d.get("job_id"):
            out.add(d["job_id"])
    return out


def job_size(job: dict) -> int:
    best = 0
    for k in ("gt_file", "output", "path"):
        p = job.get(k)
        if p:
            try:
                best = max(best, os.path.getsize(p))
            except OSError:
                pass
    return best


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def default_job_timeout() -> int:
    """3 x (recalc + data-table timeouts): score_and_diff can run up to ~5 LibreOffice passes
    (the copy's recalc and data-table solve, the input's normalised scratch, the refreshed
    golden's recalc and solve), each bounded by those two timeouts. 5400 s at INFRA's 900/900."""
    return 3 * (_env_int("EVOSKILL_RECALC_TIMEOUT_SECONDS", 180)
                + _env_int("EVOSKILL_DATATABLE_UNO_TIMEOUT_SECONDS", 300))


def _exc(e: BaseException, phase: str) -> dict:
    return {"phase": phase, "type": type(e).__name__, "message": str(e)[:4000],
            "traceback": traceback.format_exc()[-6000:]}


# ───────────────────────────────────────────────────────────────────────────── the child
class _Capture(logging.Handler):
    """Every record >= WARNING from any logger, in order, for THIS job only."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.records: list[dict] = []

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001 — a bad format string still gets recorded
            msg = str(record.msg)
        kind, cls = warning_kind(record.name, msg)
        self.records.append({"t": round(time.time(), 3), "logger": record.name,
                             "level": record.levelname, "kind": kind, "class": cls,
                             "message": msg})


def _fresh_copy(src: str, dst_dir: Path) -> Path:
    """copy2 (bytes + mtime, like the runner's output_raw copy), then owner-writable: the grader
    rewrites its copy in place, as it rewrote the solver's own file in the leg."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / os.path.basename(src)
    shutil.copy2(src, dst)
    mode = os.stat(dst).st_mode
    if not mode & 0o200:
        os.chmod(dst, (mode & 0o7777) | 0o200)
    return dst


def _check_sha(rec: dict, job: dict, path_key: str, sha_key: str, label: str) -> bool:
    p = job.get(path_key)
    if not p:
        return True
    if not os.path.isfile(p):
        rec["status"] = "refused"
        rec["refused"] = f"{label} {p} does not exist"
        return False
    got = sha256_file(p)
    rec[f"{label}_sha256_before"] = got
    want = job.get(sha_key)
    if want and got != want:
        rec["status"] = "refused"
        rec["refused"] = f"{label} sha256 {got[:12]} != job's {want[:12]}"
        return False
    return True


def _after_shas(rec: dict, job: dict, pairs) -> None:
    for path_key, label in pairs:
        p = job.get(path_key)
        if p and f"{label}_sha256_before" in rec:
            try:
                rec[f"{label}_sha256_after"] = sha256_file(p)
            except OSError as e:
                rec[f"{label}_sha256_after"] = f"unreadable: {e}"
    rec["originals_unchanged"] = all(
        rec.get(f"{lab}_sha256_after") == rec.get(f"{lab}_sha256_before")
        for _k, lab in pairs if f"{lab}_sha256_before" in rec)


def diff_file_text(passed: bool, n_match: int, n_total: int, diff: str) -> str:
    """The DIFF.txt the SSB runner writes for this result (the header outcomes.parse_diff reads)."""
    return (f"{'PASS' if passed else 'FAIL'} — {n_match}/{n_total} graded cells match\n"
            f"(scorer's own diff, computed at scoring time; full, uncapped)\n\n{diff}\n")


def child_grade(job: dict, work: Path, rec: dict, scorer_module: str) -> None:
    pairs = (("output", "output"), ("gt_file", "gt"), ("input_file", "input"))
    for pk, sk, lab in (("output", "output_sha256", "output"), ("gt_file", "gt_sha256", "gt"),
                        ("input_file", "input_sha256", "input")):
        if not _check_sha(rec, job, pk, sk, lab):
            return
    proc = _fresh_copy(job["output"], work / "output")
    rec["copy"] = str(proc)
    rec["copy_sha256"] = sha256_file(proc)
    rec["output_name"] = proc.name
    rec["output_name_matches_input"] = proc.name == os.path.basename(job["input_file"])
    t = time.time()
    try:
        scorer = importlib.import_module(scorer_module)
    except BaseException as e:  # noqa: BLE001
        rec.update(status="exception", exception=_exc(e, "import"))
        return
    rec["import_s"] = round(time.time() - t, 2)
    root = logging.getLogger()
    if root.getEffectiveLevel() > logging.WARNING:      # an import raised the root level: the
        rec["root_log_level_lowered_from"] = root.level  # scorer's WARNINGs would be dropped
        root.setLevel(logging.WARNING)
    rec["scorer_file"] = getattr(scorer, "__file__", None)
    if rec["scorer_file"]:
        rec["scorer_sha256"] = sha256_file(rec["scorer_file"])
    t = time.time()
    try:
        passed, n_match, n_total, diff = scorer.score_and_diff(
            job["gt_file"], str(proc), job["answer_position"], orig_input_file=job["input_file"])
    except Exception as e:  # noqa: BLE001 — "the grader raised" is a recorded outcome
        rec.update(status="exception", exception=_exc(e, "score_and_diff"),
                   grade_s=round(time.time() - t, 2))
    else:
        diff = diff if isinstance(diff, str) else str(diff)
        text = diff_file_text(bool(passed), int(n_match), int(n_total), diff)
        rec.update(status="ok", grade_s=round(time.time() - t, 2), passed=bool(passed),
                   n_match=int(n_match), n_total=int(n_total), diff_sha256=sha256_text(diff),
                   diff_text=diff, diff_file_sha256=sha256_text(text), diff_file_text=text)
    finally:
        _after_shas(rec, job, pairs)
        if proc.exists():
            rec["copy_sha256_after"] = sha256_file(proc)


def load_evaluation(path: str, expect_sha256: str) -> tuple:
    """(module, tqdm_stubbed). The official evaluation.py, executed from its own file after its
    sha256 is checked. It imports tqdm at line 7 but uses it only in evaluation(); the runner venv
    has no tqdm, so a pass-through stub stands in when the import fails."""
    got = sha256_file(path)
    if expect_sha256 and got != expect_sha256:
        raise JobError(f"evaluation.py sha256 {got[:12]} != pinned {expect_sha256[:12]}")
    stubbed = False
    try:
        importlib.import_module("tqdm")
    except ImportError:
        stub = types.ModuleType("tqdm")
        stub.tqdm = lambda it=None, *a, **k: it
        stub.__evoskill_stub__ = True
        sys.modules["tqdm"] = stub
        stubbed = True
    spec = importlib.util.spec_from_file_location("ssb_official_evaluation", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, stubbed


def official_compare_quote_aware(ev, gt_file: str, proc_file: str, instruction_type,
                                 answer_position: str) -> tuple:
    """evaluation.compare_workbooks (lines 155-188) step for step, the comma split excepted (see
    quote_aware_split). Uses the module's own openpyxl and cell_level_compare. Returns
    (result, msg, per-range list)."""
    if not os.path.exists(proc_file):
        return False, "File not exist", []
    try:
        wb_gt = ev.openpyxl.load_workbook(filename=gt_file, data_only=True)
        wb_proc = ev.openpyxl.load_workbook(filename=proc_file, data_only=True)
    except Exception as e:  # noqa: BLE001 — as the official function does
        return False, str(e), []
    ranges = []
    for sheet_cell_range in quote_aware_split(answer_position):
        if "!" in sheet_cell_range:
            sheet_name, cell_range = sheet_cell_range.split("!")
            sheet_name = sheet_name.lstrip("'").rstrip("'")
        else:
            sheet_name = wb_gt.sheetnames[0]
            cell_range = sheet_cell_range
        sheet_name = sheet_name.lstrip("'").rstrip("'")
        cell_range = cell_range.lstrip("'").rstrip("'")
        result, msg = ev.cell_level_compare(wb_gt, wb_proc, sheet_name, cell_range)
        ranges.append({"range": sheet_cell_range, "sheet": sheet_name, "cells": cell_range,
                       "result": bool(result), "msg": str(msg)[:2000]})
    return all(r["result"] for r in ranges), "", ranges


def stageb_recalc(proc: Path) -> dict:
    """The scorer's recalculation command line, alone, on the copy. Recorded, never judged here."""
    timeout = _env_int("EVOSKILL_RECALC_TIMEOUT_SECONDS", 180)
    cmd = ["docker", "run", "--rm", "-v", f"{proc.parent}:/work", "-w", "/work", STAGEB_IMAGE,
           "bash", "-c", STAGEB_BASH.format(name=proc.name)]
    before = sha256_file(proc)
    t = time.time()
    out = {"cmd": cmd, "timeout_s": timeout, "sha256_before": before}
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out.update(rc=r.returncode, ok=r.returncode == 0, stdout_tail=r.stdout[-1500:],
                   stderr_tail=r.stderr[-1500:])
    except subprocess.TimeoutExpired as e:
        out.update(rc=None, ok=False, error=f"TimeoutExpired: {e}")
    except OSError as e:
        out.update(rc=None, ok=False, error=f"{type(e).__name__}: {e}")
    out["seconds"] = round(time.time() - t, 2)
    out["sha256_after"] = sha256_file(proc) if proc.exists() else None
    out["changed"] = out["sha256_after"] != before
    return out


def child_stageb(job: dict, work: Path, rec: dict, evaluation: str, evaluation_sha: str) -> None:
    pairs = (("output", "output"), ("gt_file", "gt"))
    for pk, sk, lab in (("output", "output_sha256", "output"), ("gt_file", "gt_sha256", "gt")):
        if not _check_sha(rec, job, pk, sk, lab):
            return
    try:
        ev, stubbed = load_evaluation(evaluation, evaluation_sha)
    except (OSError, JobError) as e:
        rec.update(status="refused", refused=f"evaluation.py: {e}")
        return
    rec["evaluation"] = evaluation
    rec["evaluation_sha256"] = sha256_file(evaluation)
    rec["tqdm_stubbed"] = stubbed
    proc = _fresh_copy(job["output"], work / "output")
    rec["copy"] = str(proc)
    rec["copy_sha256"] = sha256_file(proc)
    try:
        rec["recalc"] = stageb_recalc(proc)
        itype = job.get("instruction_type")
        try:
            vres, vmsg = ev.compare_workbooks(job["gt_file"], str(proc), itype,
                                              job["answer_position"])
            rec["verbatim"] = {"result": bool(vres), "msg": str(vmsg)[:2000], "exception": None}
        except Exception as e:  # noqa: BLE001 — evaluation():216-219 counts a raise as False
            rec["verbatim"] = {"result": False, "msg": "", "exception": _exc(e, "verbatim")}
        try:
            qres, qmsg, ranges = official_compare_quote_aware(
                ev, job["gt_file"], str(proc), itype, job["answer_position"])
            rec["quote_aware"] = {"result": bool(qres), "msg": str(qmsg)[:2000], "ranges": ranges,
                                  "exception": None}
        except Exception as e:  # noqa: BLE001 — same wrapping as the official loop
            rec["quote_aware"] = {"result": False, "msg": "", "ranges": [],
                                  "exception": _exc(e, "quote_aware")}
        rec["passed"] = rec["quote_aware"]["result"]
        rec["verbatim_passed"] = rec["verbatim"]["result"]
        rec["status"] = "ok"
    finally:
        _after_shas(rec, job, pairs)


def child_open(job: dict, work: Path, rec: dict) -> None:
    if not _check_sha(rec, job, "path", "path_sha256", "path"):
        return
    copy = _fresh_copy(job["path"], work / "open")
    rec["copy_sha256"] = sha256_file(copy)
    try:
        import openpyxl
    except BaseException as e:  # noqa: BLE001
        rec.update(status="exception", exception=_exc(e, "import openpyxl"))
        return
    rec["openpyxl_version"] = getattr(openpyxl, "__version__", None)
    try:
        import openpyxl.xml as ox
        rec["openpyxl_lxml"] = bool(getattr(ox, "LXML", False))
    except Exception:  # noqa: BLE001
        rec["openpyxl_lxml"] = None
    for data_only in (True, False):
        key = "data_only_true" if data_only else "data_only_false"
        t = time.time()
        try:
            wb = openpyxl.load_workbook(str(copy), data_only=data_only)
            rec[key] = {"ok": True, "sheets": len(wb.sheetnames), "error": None}
            try:
                wb.close()
            except Exception:  # noqa: BLE001
                pass
        except Exception as e:  # noqa: BLE001 — the point of this mode
            rec[key] = {"ok": False, "sheets": None, "error": f"{type(e).__name__}: {str(e)[:2000]}",
                        "traceback": traceback.format_exc()[-3000:]}
        rec[key]["seconds"] = round(time.time() - t, 2)
    rec["opens"] = rec["data_only_true"]["ok"] and rec["data_only_false"]["ok"]
    rec["status"] = "ok"
    _after_shas(rec, job, (("path", "path"),))


def child_main(args) -> int:
    job = json.loads(Path(args.job_file).read_text())
    work = Path(args.job_file).parent
    cap = _Capture()
    echo = logging.StreamHandler(sys.stderr)
    echo.setLevel(logging.WARNING)
    root = logging.getLogger()
    root.addHandler(cap)
    root.addHandler(echo)
    logging.captureWarnings(True)
    if args.repo and args.repo not in sys.path:
        sys.path.insert(0, args.repo)
    rec = {"job_id": job["job_id"], "mode": job["mode"], "task_id": job.get("task_id"),
           "status": "crashed", "child_pid": os.getpid(), "child_started_utc": utcnow()}
    t0 = time.time()
    try:
        if job["mode"] == "grade":
            child_grade(job, work, rec, args.scorer_module)
        elif job["mode"] == "stageB":
            child_stageb(job, work, rec, args.evaluation, args.evaluation_sha256)
        else:
            child_open(job, work, rec)
    except BaseException as e:  # noqa: BLE001 — the worker's own failure, still one line
        rec.update(status="exception", exception=_exc(e, "worker"))
    rec["child_seconds"] = round(time.time() - t0, 2)
    rec["warnings"] = cap.records
    rec["warning_kinds"] = dict(Counter(w["kind"] for w in cap.records))
    rec["d4_failure_warnings"] = [w["message"] for w in cap.records if w["class"] == "d4_failure"]
    rec["scorer_exception_warnings"] = [w["message"] for w in cap.records
                                        if w["class"] == "scorer_exception"]
    out = Path(args.result_file)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(rec, default=str))
    os.replace(tmp, out)
    return 0


# ───────────────────────────────────────────────────────────────────────────── the parent
class Runner:
    def __init__(self, args, jobs: list[dict]):
        self.args = args
        self.jobs = jobs
        self.work = Path(args.work)
        self.results = Path(args.results)
        self.lock = threading.Lock()
        self.cv = threading.Condition()
        self.stop = threading.Event()
        self.procs: dict[str, subprocess.Popen] = {}
        self.running = 0
        self.heavy_running = 0
        self.max_heavy_seen = 0
        self.n_done = 0
        self.statuses: Counter = Counter()
        self.threads: list[threading.Thread] = []
        self.child_env = scrubbed_env()
        self.child_env["PYTHONUNBUFFERED"] = "1"

    def log(self, msg: str) -> None:
        print(f"{utcnow()} {msg}", flush=True)

    def write(self, rec: dict) -> None:
        line = json.dumps(rec, default=str, ensure_ascii=True) + "\n"
        with self.lock:
            with open(self.results, "a") as fh:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            self.n_done += 1
            self.statuses[rec.get("status")] += 1

    def kill_all(self) -> None:
        for p in list(self.procs.values()):
            _killpg(p)

    def run_job(self, job: dict, heavy: bool) -> None:
        jd = self.work / safe_name(job["job_id"])
        started = utcnow()
        t0 = time.time()
        try:
            if jd.exists():
                shutil.rmtree(jd)          # a leftover from an interrupted run: start fresh
            jd.mkdir(parents=True)
            (jd / "job.json").write_text(json.dumps(job))
            res = jd / "child_result.json"
            cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "child",
                   "--job-file", str(jd / "job.json"), "--result-file", str(res),
                   "--repo", self.args.repo, "--scorer-module", self.args.scorer_module,
                   "--evaluation", self.args.evaluation,
                   "--evaluation-sha256", self.args.evaluation_sha256]
            with open(jd / "child.log", "wb") as logf:
                p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=self.args.repo,
                                     env=self.child_env, start_new_session=True)
                self.procs[job["job_id"]] = p
                timed_out = False
                try:
                    rc = p.wait(timeout=self.args.job_timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    _killpg(p)
                    rc = p.wait()
                finally:
                    self.procs.pop(job["job_id"], None)
            if self.stop.is_set() and not res.is_file():
                return                      # interrupted: no line, so a rerun grades it again
            if res.is_file() and not timed_out:
                rec = json.loads(res.read_text())
            else:
                rec = {"job_id": job["job_id"], "mode": job["mode"], "task_id": job.get("task_id"),
                       "status": "timeout" if timed_out else "crashed"}
            rec.update(child_rc=rc, started_utc=started, ended_utc=utcnow(),
                       seconds=round(time.time() - t0, 2), heavy=heavy,
                       job_size_bytes=job_size(job), worker=WORKER_VERSION,
                       job_timeout_s=self.args.job_timeout)
            if timed_out:
                rec["timeout"] = f"killed after {self.args.job_timeout}s with its process group"
            if rec["status"] not in ("ok",):
                try:
                    rec["child_log_tail"] = (jd / "child.log").read_bytes()[-LOG_TAIL:].decode(
                        "utf-8", "replace")
                except OSError:
                    pass
            if "meta" in job:
                rec["meta"] = job["meta"]
            self.write(rec)
            self.log(f"[{self.n_done}/{len(self.jobs)}] {rec['status']:<9} {job['mode']:<6} "
                     f"{rec['seconds']:>8.1f}s heavy={int(heavy)} {job['job_id']}")
        except Exception as e:  # noqa: BLE001 — the parent must never lose a job silently
            self.write({"job_id": job["job_id"], "mode": job["mode"], "task_id": job.get("task_id"),
                        "status": "crashed", "exception": _exc(e, "parent"),
                        "started_utc": started, "ended_utc": utcnow(), "heavy": heavy,
                        "worker": WORKER_VERSION})
        finally:
            if not self.args.keep_copies:
                for sub in ("output", "open"):
                    shutil.rmtree(jd / sub, ignore_errors=True)

    def _finish(self, heavy: bool) -> None:
        with self.cv:
            self.running -= 1
            if heavy:
                self.heavy_running -= 1
            self.cv.notify_all()

    def _start(self, job: dict, heavy: bool) -> None:
        self.running += 1
        if heavy:
            self.heavy_running += 1
            self.max_heavy_seen = max(self.max_heavy_seen, self.heavy_running)

        def go():
            try:
                self.run_job(job, heavy)
            finally:
                self._finish(heavy)
        th = threading.Thread(target=go, name=job["job_id"][:40], daemon=True)
        self.threads.append(th)
        th.start()

    def dispatch(self) -> None:
        """Heavy jobs first while a heavy slot is free (long jobs early shorten the tail), light
        jobs fill every other slot; never more than max_procs children or max_heavy heavy ones."""
        heavy_q = deque(j for j in self.jobs if j["_heavy"])
        light_q = deque(j for j in self.jobs if not j["_heavy"])
        with self.cv:
            while not self.stop.is_set():
                if self.running < self.args.max_procs:
                    if heavy_q and self.heavy_running < self.args.max_heavy:
                        self._start(heavy_q.popleft(), True)
                        continue
                    if light_q:
                        self._start(light_q.popleft(), False)
                        continue
                if not heavy_q and not light_q and self.running == 0:
                    break
                self.cv.wait(timeout=0.5)
        if self.stop.is_set():
            self.kill_all()
        for th in self.threads:
            th.join()


def _killpg(p: subprocess.Popen) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(p.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            p.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            continue


def file_shas(repo: str, scorer_module: str) -> dict:
    out = {}
    scorer = Path(repo) / (scorer_module.replace(".", "/") + ".py")
    if scorer.is_file():
        out[str(scorer)] = sha256_file(scorer)
    ds = scorer.parent / "docker_scripts"
    if ds.is_dir():
        for p in sorted(ds.iterdir()):
            if p.is_file():
                out[str(p)] = sha256_file(p)
    return out


def run_main(args) -> int:
    if os.environ.get("OPENROUTER_API_KEY"):
        print("REFUSED: OPENROUTER_API_KEY is in the environment; the grading sandbox never holds "
              "a model key", file=sys.stderr)
        return 3
    try:
        jobs = load_jobs(args.jobs)
    except JobError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    args.job_timeout = args.job_timeout or default_job_timeout()
    args.max_procs = max(1, args.max_procs or _env_int("EVOSKILL_RECALC_CONCURRENCY", 4))
    args.max_heavy = max(1, args.max_heavy or _env_int("EVOSKILL_HEAVY_COMPARE_CONCURRENCY", 2))
    args.heavy_bytes = args.heavy_bytes or _env_int("EVOSKILL_HEAVY_COMPARE_BYTES", 5_000_000)
    shas = file_shas(args.repo, args.scorer_module)
    scorer_path = str(Path(args.repo) / (args.scorer_module.replace(".", "/") + ".py"))
    if args.expect_scorer_sha256 and shas.get(scorer_path) != args.expect_scorer_sha256:
        print(f"REFUSED: scorer {scorer_path} sha256 {str(shas.get(scorer_path))[:12]} != "
              f"expected {args.expect_scorer_sha256[:12]}", file=sys.stderr)
        return 3
    results = Path(args.results)
    results.parent.mkdir(parents=True, exist_ok=True)
    Path(args.work).mkdir(parents=True, exist_ok=True)
    torn = repair_results(results)
    done = done_ids(results)
    todo = [j for j in jobs if j["job_id"] not in done]
    for j in todo:
        j["_heavy"] = job_size(j) >= args.heavy_bytes
    try:
        import openpyxl
        opx = getattr(openpyxl, "__version__", None)
    except ImportError:
        opx = None
    meta = {"worker": WORKER_VERSION, "worker_sha256": sha256_file(__file__),
            "started_utc": utcnow(), "python": sys.version, "executable": sys.executable,
            "openpyxl": opx, "argv": sys.argv, "repo": args.repo,
            "scorer_module": args.scorer_module, "code_sha256": shas,
            "evaluation": args.evaluation,
            "evaluation_sha256": (sha256_file(args.evaluation) if args.evaluation and os.path.isfile(args.evaluation)
                                  else None),
            "env": {k: v for k, v in sorted(os.environ.items())
                    if k.startswith(("EVOSKILL_", "EVOSHIM_", "OMP_", "OPENBLAS_", "MKL_",
                                     "NUMEXPR_", "MAX_CONCURRENCY", "PYTHON"))
                    and not SECRET_ENV_RE.search(k)},
            "max_procs": args.max_procs, "max_heavy": args.max_heavy,
            "heavy_bytes": args.heavy_bytes, "job_timeout_s": args.job_timeout,
            "n_jobs": len(jobs), "n_skipped_done": len(jobs) - len(todo), "n_todo": len(todo),
            "n_heavy_todo": sum(1 for j in todo if j["_heavy"]), "torn_line_moved_to": torn,
            "nproc": os.cpu_count()}
    meta_path = results.with_name("worker_meta.json")
    meta_path.write_text(json.dumps(meta, indent=1, default=str))
    r = Runner(args, todo)
    r.log(f"{WORKER_VERSION}: {len(todo)} job(s) to run ({meta['n_heavy_todo']} heavy), "
          f"{meta['n_skipped_done']} already done; max_procs={args.max_procs} "
          f"max_heavy={args.max_heavy} job_timeout={args.job_timeout}s")

    def on_sig(signum, _frm):
        # Only set the flag: dispatch() polls it every 0.5 s, kills the children and joins.
        r.stop.set()
        print(f"{utcnow()} signal {signum}: stopping; running children are killed and write "
              f"no line", flush=True)
    signal.signal(signal.SIGTERM, on_sig)
    signal.signal(signal.SIGINT, on_sig)
    r.dispatch()
    meta.update(ended_utc=utcnow(), statuses=dict(r.statuses), n_written=r.n_done,
                interrupted=r.stop.is_set(), max_heavy_concurrent_seen=r.max_heavy_seen)
    meta_path.write_text(json.dumps(meta, indent=1, default=str))
    r.log(f"done: {dict(r.statuses)} interrupted={r.stop.is_set()}")
    return 143 if r.stop.is_set() else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--repo", required=True)
        p.add_argument("--scorer-module", default=DEFAULT_SCORER_MODULE)
        p.add_argument("--evaluation", default=DEFAULT_EVALUATION)
        p.add_argument("--evaluation-sha256", default=EVALUATION_SHA256)

    p = sub.add_parser("run")
    common(p)
    p.add_argument("--jobs", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--max-procs", type=int, default=0)
    p.add_argument("--max-heavy", type=int, default=0)
    p.add_argument("--heavy-bytes", type=int, default=0)
    p.add_argument("--job-timeout", type=int, default=0)
    p.add_argument("--expect-scorer-sha256", default="")
    p.add_argument("--keep-copies", action="store_true")
    p = sub.add_parser("child")
    common(p)
    p.add_argument("--job-file", required=True)
    p.add_argument("--result-file", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "child":
        return child_main(args)
    return run_main(args)


if __name__ == "__main__":
    sys.exit(main())
