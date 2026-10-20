"""scripts/daytona/grade_worker.py -- the grading sandbox's in-sandbox worker.

OFFLINE: no Daytona, no LibreOffice, no model. The grader is a FAKE scorer module (written into a
temp "repo" and passed as --scorer-module) that mutates the copy it is given in place, logs a
WARNING naming that copy, and on request raises, sleeps, crashes or logs the scorer's own failure
strings. Stage B runs the REAL SpreadsheetBench-v2/evaluation.py (read-only; skipped when absent)
with a fake `docker` on PATH standing in for the shim's LibreOffice recalculation.

What must hold: every WARNING is captured by the job that caused it (one child per job, even under
concurrency); an exception is a recorded result, not a lost job; the saved output, the reference
and the input are never touched (only a fresh copy is graded); a rerun skips done jobs; the heavy
cap holds; a timeout kills the job's process group; the worker's pins agree with the frozen
sources (the grader-failure patterns, the DIFF.txt header, the recalculation command, every scorer warning).
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "daytona" / "grade_worker.py"
SCORER = ROOT / "src" / "agent_profiles" / "ssbv2_agent" / "scorer.py"
OUTCOMES = ROOT / "src" / "orchestrator" / "outcomes.py"
# the official evaluation.py of a SpreadsheetBench v2 checkout ($SSBV2_ROOT); the Stage B tests skip without it
EVALUATION = Path(os.environ.get("SSBV2_ROOT", "/nonexistent-ssbv2-checkout")) / "evaluation.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


W = _load(WORKER, "grade_worker_under_test")

FAKE_SCORER = r'''
import json, logging, os, time
logger = logging.getLogger("src.agent_profiles.ssbv2_agent.scorer")
TRACE = os.environ.get("FAKE_TRACE_DIR")

def score_and_diff(gt_file, proc_file, answer_position, max_mismatches=0, orig_input_file=None):
    tag = os.path.basename(proc_file)
    start = time.time()
    if TRACE:
        with open(os.path.join(TRACE, tag + ".start.json"), "w") as fh:
            json.dump({"pid": os.getpid(), "proc_file": proc_file, "gt": gt_file,
                       "orig": orig_input_file, "start": start}, fh)
    with open(proc_file, "ab") as fh:            # in place, like the real recalculation
        fh.write(b"RECALCULATED")
    logger.warning(f"marker {tag}")
    if "RECALCFAIL" in answer_position:
        logger.warning(f"LibreOffice recalculation failed (exit 1) for {proc_file}: stdout='' "
                       f"stderr='' - scoring will proceed on the UN-recalculated file")
    if "PRECHECK" in answer_position:
        logger.warning(f"_solve_datatables_with_uno: pre-check failed for {proc_file}: "
                       f"Namespace prefix dc on creator is not defined")
    if "SLEEP" in answer_position:
        time.sleep(float(answer_position.split("SLEEP", 1)[1].split(",")[0]))
    if "RAISE" in answer_position:
        raise RuntimeError(f"fake grader raised for {tag}")
    if "CRASH" in answer_position:
        os._exit(9)
    data = open(proc_file, "rb").read()
    passed = data.startswith(b"GOOD")
    if TRACE:
        with open(os.path.join(TRACE, tag + ".end.json"), "w") as fh:
            json.dump({"end": time.time()}, fh)
    return (passed, 3 if passed else 1, 3,
            "All cells match" if passed else "'S'!A1: expected=1, got=2")
'''


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture
def box(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "fake_scorer.py").write_text(FAKE_SCORER)
    trace = tmp_path / "trace"
    trace.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    gt = data / "golden.xlsx"
    gt.write_bytes(b"GT-bytes")
    inp = data / "input.xlsx"
    inp.write_bytes(b"IN-bytes")
    return {"tmp": tmp_path, "repo": repo, "trace": trace, "data": data, "gt": gt, "inp": inp,
            "results": tmp_path / "out" / "results.jsonl", "work": tmp_path / "work"}


def grade_job(box, jid, content: bytes, ap="S!A1:A3", name=None, meta=None) -> dict:
    d = box["data"] / "saved" / jid.replace("/", "_")
    d.mkdir(parents=True, exist_ok=True)
    out = d / (name or f"{jid.replace('/', '_')}.xlsx")
    out.write_bytes(content)
    j = {"job_id": jid, "mode": "grade", "task_id": "Cat/01_01", "gt_file": str(box["gt"]),
         "gt_sha256": sha(box["gt"]), "input_file": str(box["inp"]),
         "input_sha256": sha(box["inp"]), "answer_position": ap, "output": str(out),
         "output_sha256": sha(out)}
    if meta:
        j["meta"] = meta
    return j


def run_worker(box, jobs, *extra, env_extra=None, timeout=120):
    jf = box["tmp"] / f"jobs_{time.time_ns()}.jsonl"
    jf.write_text("".join(json.dumps(j) + "\n" for j in jobs))
    env = {k: v for k, v in os.environ.items() if k != "OPENROUTER_API_KEY"}
    env["FAKE_TRACE_DIR"] = str(box["trace"])
    env.update(env_extra or {})
    cmd = [sys.executable, str(WORKER), "run", "--jobs", str(jf), "--results", str(box["results"]),
           "--work", str(box["work"]), "--repo", str(box["repo"]), "--scorer-module", "fake_scorer",
           *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    res = []
    if box["results"].is_file():
        res = [json.loads(ln) for ln in box["results"].read_text().splitlines() if ln.strip()]
    meta_f = box["results"].with_name("worker_meta.json")
    meta = json.loads(meta_f.read_text()) if meta_f.is_file() else None
    return p, {r["job_id"]: r for r in res}, meta


# ─────────────────────────────────────────────────────────────── per-job capture, exceptions
def test_every_warning_is_captured_by_the_job_that_caused_it(box):
    jobs = [grade_job(box, f"j{i}", b"GOOD" if i % 2 else b"BAD", ap="S!A1:A3,SLEEP0.4")
            for i in range(6)]
    p, res, meta = run_worker(box, jobs, "--max-procs", "6")
    assert p.returncode == 0, p.stderr
    assert len(res) == 6 and meta["max_procs"] == 6
    starts = sorted(json.loads(f.read_text())["start"] for f in box["trace"].glob("*.start.json"))
    assert starts[-1] - starts[0] < 0.4, "jobs must actually overlap for this test to mean anything"
    for j in jobs:
        r = res[j["job_id"]]
        mine = f"marker {Path(j['output']).name}"
        markers = [w["message"] for w in r["warnings"] if w["message"].startswith("marker ")]
        assert markers == [mine], (j["job_id"], markers)
        assert r["status"] == "ok" and r["passed"] is (int(j["job_id"][1:]) % 2 == 1)
        assert r["warnings"][0]["logger"] == "src.agent_profiles.ssbv2_agent.scorer"


def test_an_exception_is_a_recorded_result_and_the_other_jobs_still_run(box):
    jobs = [grade_job(box, "boom", b"GOOD", ap="S!A1,RAISE"), grade_job(box, "fine", b"GOOD")]
    p, res, _ = run_worker(box, jobs)
    assert p.returncode == 0, p.stderr
    r = res["boom"]
    assert r["status"] == "exception"
    assert r["exception"]["type"] == "RuntimeError" and r["exception"]["phase"] == "score_and_diff"
    assert "fake grader raised for" in r["exception"]["message"]
    assert "fake grader raised" in r["exception"]["traceback"]
    assert "passed" not in r and "diff_text" not in r
    assert [w["message"] for w in r["warnings"]] == [f"marker {Path(jobs[0]['output']).name}"]
    assert res["fine"]["status"] == "ok" and res["fine"]["passed"] is True


def test_grade_records_the_diff_and_the_exact_diff_txt_the_runner_writes(box):
    j = grade_job(box, "d", b"BAD")
    _p, res, _ = run_worker(box, [j])
    r = res["d"]
    assert (r["passed"], r["n_match"], r["n_total"]) == (False, 1, 3)
    assert r["diff_text"] == "'S'!A1: expected=1, got=2"
    assert r["diff_sha256"] == hashlib.sha256(r["diff_text"].encode()).hexdigest()
    assert r["diff_file_text"].startswith("FAIL — 1/3 graded cells match\n(scorer's own diff")
    assert r["diff_file_sha256"] == hashlib.sha256(r["diff_file_text"].encode()).hexdigest()
    O = _load(OUTCOMES, "outcomes_for_worker_test")
    assert O.classify_detail(r["diff_file_text"], []) == ("graded_wrong", "fail")


# ─────────────────────────────────────────────────────────────── originals never touched
def test_only_a_fresh_copy_is_graded_and_the_originals_never_change(box):
    j = grade_job(box, "c", b"GOOD-original")
    src = Path(j["output"])
    for f in (src, box["gt"], box["inp"]):
        f.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)   # a write to an original would raise
    before = {f: sha(f) for f in (src, box["gt"], box["inp"])}
    p, res, _ = run_worker(box, [j], "--keep-copies")
    assert p.returncode == 0, p.stderr
    r = res["c"]
    assert r["status"] == "ok" and r["passed"] is True
    assert {f: sha(f) for f in before} == before
    assert r["output_sha256_before"] == r["output_sha256_after"] == before[src]
    assert r["gt_sha256_before"] == r["gt_sha256_after"] and r["originals_unchanged"] is True
    assert r["copy_sha256"] == before[src] and r["copy_sha256_after"] != before[src]
    seen = json.loads((box["trace"] / f"{src.name}.start.json").read_text())
    assert Path(seen["proc_file"]).resolve() != src.resolve()
    assert str(box["work"]) in seen["proc_file"] and "/output/" in seen["proc_file"]
    assert (seen["gt"], seen["orig"]) == (str(box["gt"]), str(box["inp"]))
    copy = Path(r["copy"])
    assert copy.read_bytes() == b"GOOD-originalRECALCULATED" and src.read_bytes() == b"GOOD-original"
    assert r["output_name"] == src.name and r["output_name_matches_input"] is False


def test_copies_are_removed_unless_kept(box):
    j = grade_job(box, "rm", b"GOOD")
    _p, res, _ = run_worker(box, [j])
    assert res["rm"]["status"] == "ok" and not Path(res["rm"]["copy"]).exists()


def test_a_sha_mismatch_is_refused_before_the_grader_runs(box):
    j = grade_job(box, "bad", b"GOOD")
    j["output_sha256"] = "0" * 64
    j2 = grade_job(box, "badgt", b"GOOD")
    j2["gt_sha256"] = "f" * 64
    _p, res, _ = run_worker(box, [j, j2])
    assert res["bad"]["status"] == "refused" and "output sha256" in res["bad"]["refused"]
    assert res["badgt"]["status"] == "refused" and "gt sha256" in res["badgt"]["refused"]
    assert not list(box["trace"].glob("*.start.json")), "the grader must not have been called"


# ─────────────────────────────────────────────────────────────── resume
def test_a_rerun_skips_done_jobs_and_leaves_their_lines_alone(box):
    first = [grade_job(box, "a", b"GOOD"), grade_job(box, "b", b"BAD")]
    p, res, meta = run_worker(box, first)
    assert p.returncode == 0 and set(res) == {"a", "b"} and meta["n_todo"] == 2
    lines_before = box["results"].read_text().splitlines()
    for f in box["trace"].glob("*"):
        f.unlink()
    second = first + [grade_job(box, "c", b"GOOD"), grade_job(box, "d", b"BAD")]
    p, res, meta = run_worker(box, second)
    assert p.returncode == 0, p.stderr
    assert meta["n_skipped_done"] == 2 and meta["n_todo"] == 2
    assert box["results"].read_text().splitlines()[:2] == lines_before
    assert set(res) == {"a", "b", "c", "d"}
    graded = sorted(f.name for f in box["trace"].glob("*.start.json"))
    assert graded == ["c.xlsx.start.json", "d.xlsx.start.json"]


def test_a_torn_last_line_is_moved_aside_and_its_job_graded_again(box):
    a, b = grade_job(box, "a", b"GOOD"), grade_job(box, "b", b"GOOD")
    run_worker(box, [a])
    with open(box["results"], "a") as fh:
        fh.write('{"job_id": "b", "status": "o')          # the sandbox died mid-write
    p, res, meta = run_worker(box, [a, b])
    assert p.returncode == 0, p.stderr
    assert meta["torn_line_moved_to"] and Path(meta["torn_line_moved_to"]).read_text().startswith(
        '{"job_id": "b"')
    assert res["b"]["status"] == "ok"
    for ln in box["results"].read_text().splitlines():
        json.loads(ln)                                   # every line parses again


# ─────────────────────────────────────────────────────────────── timeouts, crashes, heavy cap
def test_a_timeout_kills_the_job_and_its_process_group(box):
    j = grade_job(box, "slow", b"GOOD", ap="S!A1,SLEEP60")
    t = time.time()
    p, res, _ = run_worker(box, [j], "--job-timeout", "2")
    assert time.time() - t < 40
    r = res["slow"]
    assert r["status"] == "timeout" and r["job_timeout_s"] == 2 and r["child_rc"] != 0
    pid = json.loads((box["trace"] / f"{Path(j['output']).name}.start.json").read_text())["pid"]
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_a_crashed_child_is_recorded_with_its_exit_code(box):
    _p, res, _ = run_worker(box, [grade_job(box, "dead", b"GOOD", ap="S!A1,CRASH")])
    r = res["dead"]
    assert r["status"] == "crashed" and r["child_rc"] == 9 and "child_log_tail" in r


def test_the_heavy_cap_holds_while_light_jobs_use_the_other_slots(box):
    heavy = [grade_job(box, f"h{i}", b"GOOD" + b"x" * 400, ap="S!A1,SLEEP0.5") for i in range(3)]
    light = [grade_job(box, f"l{i}", b"GOOD", ap="S!A1,SLEEP0.5") for i in range(3)]
    p, res, meta = run_worker(box, heavy + light, "--max-procs", "4", "--max-heavy", "1",
                              "--heavy-bytes", "300")
    assert p.returncode == 0, p.stderr
    assert meta["n_heavy_todo"] == 3 and meta["max_heavy_concurrent_seen"] == 1
    assert all(res[j["job_id"]]["heavy"] for j in heavy)
    assert not any(res[j["job_id"]]["heavy"] for j in light)

    def span(j):
        n = Path(j["output"]).name
        return (json.loads((box["trace"] / f"{n}.start.json").read_text())["start"],
                json.loads((box["trace"] / f"{n}.end.json").read_text())["end"])
    hs = sorted(span(j) for j in heavy)
    for (s1, e1), (s2, _e2) in zip(hs, hs[1:]):
        assert s2 >= e1 - 0.05, "two heavy grades overlapped"
    ls = [span(j) for j in light]
    assert any(s < hs[0][1] for s, _e in ls), "light jobs should run beside the first heavy one"


# ─────────────────────────────────────────────────────────────── warning labels (OPEN data)
def test_failure_and_exception_warnings_are_labelled_not_judged(box):
    j = grade_job(box, "w", b"GOOD", ap="S!A1,RECALCFAIL,PRECHECK")
    _p, res, _ = run_worker(box, [j])
    r = res["w"]
    assert r["status"] == "ok" and r["passed"] is True       # the worker records; the host judges
    assert len(r["d4_failure_warnings"]) == 1
    assert r["d4_failure_warnings"][0].startswith("LibreOffice recalculation failed")
    assert len(r["scorer_exception_warnings"]) == 1
    assert "pre-check failed" in r["scorer_exception_warnings"][0]
    assert r["warning_kinds"] == {"other": 1, "recalc_failed": 1, "uno_precheck_failed": 1}
    O = _load(OUTCOMES, "outcomes_for_worker_test2")
    msgs = [w["message"] for w in r["warnings"]]
    assert O.classify_detail(r["diff_file_text"], msgs)[0] == "grader_failed"


# ─────────────────────────────────────────────────────────────── open mode
def test_open_mode_records_whether_openpyxl_can_open_each_way(box):
    openpyxl = pytest.importorskip("openpyxl")
    good = box["data"] / "good.xlsx"
    wb = openpyxl.Workbook()
    wb.active["A1"] = 1
    wb.save(good)
    bad = box["data"] / "bad.xlsx"
    bad.write_bytes(b"PK\x03\x04 not really a workbook")
    jobs = [{"job_id": f"open/{p.stem}", "mode": "open", "path": str(p), "path_sha256": sha(p)}
            for p in (good, bad)]
    before = {p: sha(p) for p in (good, bad)}
    p, res, _ = run_worker(box, jobs)
    assert p.returncode == 0, p.stderr
    g, b = res["open/good"], res["open/bad"]
    assert g["status"] == "ok" and g["opens"] is True
    assert g["data_only_true"]["ok"] and g["data_only_false"]["ok"] and g["data_only_true"]["sheets"] == 1
    assert b["status"] == "ok" and b["opens"] is False
    assert not b["data_only_true"]["ok"] and not b["data_only_false"]["ok"]
    assert b["data_only_true"]["error"] and b["path_sha256_before"] == b["path_sha256_after"]
    assert {q: sha(q) for q in before} == before


# ─────────────────────────────────────────────────────────────── Stage B
FAKE_DOCKER = """#!/bin/sh
printf '%s\\n' "$@" > "$FAKE_DOCKER_LOG"
exit 0
"""


def _stageb_setup(box, gt_vals, out_vals, sheet="A,B"):
    openpyxl = pytest.importorskip("openpyxl")
    if not EVALUATION.is_file():
        pytest.skip("SpreadsheetBench-v2/evaluation.py not present")
    pytest.importorskip("numpy")                       # evaluation.py imports it
    for path, vals in ((box["data"] / "gt_b.xlsx", gt_vals), (box["data"] / "out_b.xlsx", out_vals)):
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = sheet
        for i, v in enumerate(vals, 1):
            ws[f"A{i}"] = v
        wb.save(path)
    fb = box["tmp"] / "fakebin"
    fb.mkdir(exist_ok=True)
    (fb / "docker").write_text(FAKE_DOCKER)
    (fb / "docker").chmod(0o755)
    return box["data"] / "gt_b.xlsx", box["data"] / "out_b.xlsx", fb


def test_stageB_recalculates_a_copy_and_compares_with_the_quote_aware_official_rules(box):
    gt, out, fb = _stageb_setup(box, [1, 2], [1, 2])
    log = box["tmp"] / "docker_argv.txt"
    j = {"job_id": "sb", "mode": "stageB", "task_id": "Cat/01_01", "gt_file": str(gt),
         "gt_sha256": sha(gt), "answer_position": "'A,B'!A1:A2", "output": str(out),
         "output_sha256": sha(out)}
    before = sha(out)
    p, res, _ = run_worker(box, [j], "--evaluation", str(EVALUATION), env_extra={
        "PATH": f"{fb}{os.pathsep}{os.environ['PATH']}", "FAKE_DOCKER_LOG": str(log),
        "EVOSKILL_RECALC_TIMEOUT_SECONDS": "900"})
    assert p.returncode == 0, p.stderr
    r = res["sb"]
    assert r["status"] == "ok" and r["evaluation_sha256"] == W.EVALUATION_SHA256
    assert r["recalc"]["ok"] is True and r["recalc"]["timeout_s"] == 900
    argv = log.read_text().splitlines()
    assert argv[:2] == ["run", "--rm"] and "ssb-verifier:latest" in argv
    assert argv[-1] == W.STAGEB_BASH.format(name=out.name)
    assert r["passed"] is True, r["quote_aware"]
    assert r["quote_aware"]["ranges"][0]["sheet"] == "A,B"
    assert r["verbatim_passed"] is False, "the official comma split must break 'A,B'"
    assert sha(out) == before and r["originals_unchanged"] is True


def test_stageB_fails_a_wrong_value_in_both_columns(box):
    gt, out, fb = _stageb_setup(box, [1, 2], [1, 3], sheet="Sheet1")
    j = {"job_id": "sbw", "mode": "stageB", "task_id": "Cat/01_01", "gt_file": str(gt),
         "answer_position": "Sheet1!A1:A2", "output": str(out), "output_sha256": sha(out)}
    _p, res, _ = run_worker(box, [j], "--evaluation", str(EVALUATION), env_extra={
        "PATH": f"{fb}{os.pathsep}{os.environ['PATH']}",
        "FAKE_DOCKER_LOG": str(box["tmp"] / "argv2.txt")})
    r = res["sbw"]
    assert r["status"] == "ok" and r["passed"] is False and r["verbatim_passed"] is False
    assert "A2" in r["quote_aware"]["ranges"][0]["msg"]


def test_stageB_refuses_an_evaluation_file_with_another_sha(box):
    gt, out, fb = _stageb_setup(box, [1], [1], sheet="S")
    j = {"job_id": "sbx", "mode": "stageB", "task_id": "Cat/01_01", "gt_file": str(gt),
         "answer_position": "S!A1", "output": str(out), "output_sha256": sha(out)}
    _p, res, _ = run_worker(box, [j], "--evaluation", str(EVALUATION),
                            "--evaluation-sha256", "0" * 64)
    assert res["sbx"]["status"] == "refused" and "evaluation.py" in res["sbx"]["refused"]


def test_tqdm_is_stubbed_only_when_it_cannot_be_imported(monkeypatch):
    if not EVALUATION.is_file():
        pytest.skip("SpreadsheetBench-v2/evaluation.py not present")
    pytest.importorskip("numpy")                       # evaluation.py imports it
    monkeypatch.setitem(sys.modules, "tqdm", None)            # `import tqdm` now raises
    mod, stubbed = W.load_evaluation(str(EVALUATION), W.EVALUATION_SHA256)
    assert stubbed is True and getattr(sys.modules["tqdm"], "__evoskill_stub__", False)
    assert mod.tqdm([1, 2]) == [1, 2] and callable(mod.compare_workbooks)


def test_quote_aware_split_changes_only_commas_inside_quotes():
    assert W.quote_aware_split("'IS,BS,CF'!B3:K121,'Revenue'!B1:O23") == [
        "'IS,BS,CF'!B3:K121", "'Revenue'!B1:O23"]
    for ap in ("A1:B2", "S!A1,S!B2", "A1:B2, C3"):
        assert W.quote_aware_split(ap) == ap.split(",")      # identical where no quotes


# ─────────────────────────────────────────────────────────────── refusals, validation
def test_the_worker_refuses_to_run_with_a_model_key_in_its_environment(box):
    p, res, _ = run_worker(box, [grade_job(box, "k", b"GOOD")],
                           env_extra={"OPENROUTER_API_KEY": "sk-or-not-a-key"})
    assert p.returncode == 3 and not res


def test_the_worker_refuses_a_scorer_with_another_sha(box):
    p, res, _ = run_worker(box, [grade_job(box, "s", b"GOOD")],
                           "--expect-scorer-sha256", "0" * 64)
    assert p.returncode == 3 and "REFUSED: scorer" in p.stderr and not res


@pytest.mark.parametrize("bad", ["dup", "relative", "mode", "missing"])
def test_a_bad_job_list_is_refused_whole(box, bad):
    j = grade_job(box, "x", b"GOOD")
    jobs = {"dup": [j, dict(j)], "relative": [dict(j, output="rel/x.xlsx")],
            "mode": [dict(j, mode="regrade")], "missing": [{k: v for k, v in j.items()
                                                              if k != "answer_position"}]}[bad]
    p, res, _ = run_worker(box, jobs)
    assert p.returncode == 2 and "REFUSED" in p.stderr and not res


def test_meta_is_copied_and_no_secret_env_reaches_the_child(box):
    j = grade_job(box, "m", b"GOOD", meta={"leg": "e0_train", "sample": 2})
    p, res, meta = run_worker(box, [j], env_extra={"SOME_API_KEY": "zzz", "EVOSKILL_X": "1"})
    assert res["m"]["meta"] == {"leg": "e0_train", "sample": 2}
    assert "SOME_API_KEY" not in json.dumps(meta) and meta["env"].get("EVOSKILL_X") == "1"
    assert W.scrubbed_env({"A_API_KEY": "1", "CC_OAUTH_X": "2", "PATH": "/bin"}) == {"PATH": "/bin"}


# ─────────────────────────────────────────────────────────────── pins against frozen sources
def test_the_d4_patterns_match_the_outcomes_classifier():
    O = _load(OUTCOMES, "outcomes_for_pins")
    assert tuple(W.GRADER_FAILURE_PATTERNS) == tuple(O.GRADER_FAILURE_PATTERNS)


def test_the_diff_txt_header_is_the_one_outcomes_parses():
    O = _load(OUTCOMES, "outcomes_for_hdr")
    d = O.parse_diff(W.diff_file_text(True, 5, 5, "All cells match"))
    assert d == {"verdict": "PASS", "n_match": 5, "n_total": 5, "body": "All cells match"}


def test_stageB_recalculates_with_the_scorers_own_command_line():
    src = SCORER.read_text()
    assert '"ssb-verifier:latest",' in src and W.STAGEB_IMAGE == "ssb-verifier:latest"
    assert "f'" + W.STAGEB_BASH.replace("{name}", "{xlsx_name}") + "'" in src


def _render(node) -> str:
    """The message a logger.warning(...) argument produces, every {expr} rendered as '7'."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(_render(v) if not isinstance(v, ast.FormattedValue) else "7"
                       for v in node.values)
    raise AssertionError(f"unexpected warning argument {ast.dump(node)[:80]}")


def test_every_scorer_warning_has_a_label():
    tree = ast.parse(SCORER.read_text())
    msgs = [_render(n.args[0]) for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "warning" and isinstance(n.func.value, ast.Name)
            and n.func.value.id == "logger"]
    assert len(msgs) >= 20
    unlabelled = [m for m in msgs if W.warning_kind("src.agent_profiles.ssbv2_agent.scorer", m)[0]
                  in ("other", "d4_pattern_elsewhere")]
    assert not unlabelled, unlabelled
    kinds = {W.warning_kind("x", m)[0] for m in msgs}
    assert {"recalc_failed", "uno_failed", "uno_regression_rollback",
            "uno_nontarget_rollback", "golden_refresh_refused"} <= kinds
