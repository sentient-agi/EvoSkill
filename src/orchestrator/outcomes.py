"""Per-sample OUTCOMES of an SSB v2 leg, read from the records the leg left on disk.

WHY THIS EXISTS. A leg JSON records one float per sample (1.0 / 0.0). A 0.0 hides three different
things: the solver's workbook was graded and is wrong; the solver saved no workbook at all, so
there was nothing to grade; or the grader itself broke (it raised, or LibreOffice / the data-table
pass failed, so formula cells read as None and a right answer can score as wrong). Only the first
is a measurement of the solver. So each sample gets one of four outcomes:

  passed         the scorer's DIFF.txt header says PASS;
  graded_wrong   the header says FAIL and the grader compared a workbook;
  not_eligible   no output workbook was saved: the recorded diff is exactly "Output file not found";
  grader_failed  no diff was recorded (the grader raised, or the snapshot never got its DIFF.txt),
                 OR a recalculation / data-table FAILURE warning names this sample's workbook.

A grader_failed sample is re-graded once from its saved output; while one remains the leg
carries no number (`reportable` is False).

THE RECORDED FORMS (read-only sources):
  * DIFF.txt header, src/agent_profiles/ssbv2_agent/ssb_v2_agent.py `_diff_header`:
        "PASS — 12/12 graded cells match\\n(scorer's own diff, computed at scoring time; full, uncapped)\\n\\n<diff>\\n"
    with "FAIL — 0/0 ..." + body "Output file not found" when the solver saved no workbook.
  * failure warnings, src/agent_profiles/ssbv2_agent/scorer.py (logged to the runner's stderr, which
    the leg driver keeps as <shard>/runner.log):
        "LibreOffice recalculation failed [(exit N)] for <xlsx>: ..."
        "_solve_datatables_with_uno failed [(exit N)] for <xlsx>: ..."
    and ssb_v2_agent.py's "Inline SSBv2 scoring failed for <task_id>: <e>" (the grader raised).
    ONLY these. The scorer's other warnings (the data-table pass rolling back a regression, the
    golden-aware check falling back, the golden refresh using the raw golden) are its designed
    fallbacks, not failures, so they do not count.
  * where the diffs live: <attempt>/tasks/_snapshots/<task with / -> _>/iter_<N>_s<k>/DIFF.txt,
    written by task_workspace.snapshot_output in the same second the runner caches the rollout.

WHICH ATTEMPT. A leg can run more than once (a failed shard, a --resume, a whole-leg rerun). The
earlier attempt is moved aside (<store>._prev_<stamp>/ for a loop leg, <shard>/_prev_<stamp>/ for
the held-out baseline), and a rerun REPLAYS every rollout that reached the cache, so a replayed
sample's verdict and DIFF.txt belong to the attempt that produced its cache record, not to the
final one (which wrote no snapshot for it). The e0 train leg is the extreme case: the final
attempt wrote 1 DIFF.txt, the attempt before it 177. The merged <repo>/.cache/ssbv2_<tag>_tasks
dir is NOT used: it is a file-by-file union, so a snapshot that lost its DIFF.txt in the newer
attempt would show the older attempt's DIFF.txt.
  Link rule: the final attempt's shard cache holds exactly the record whose inline_score the leg
recorded (replayed or fresh). Its `created_at` is the second the runner cached the rollout, and
snapshot_output wrote DIFF.txt / LAYOUT.md just before. The sample's snapshot is the one (in any
attempt) whose DIFF.txt / LAYOUT.md mtime falls in [created_at - LINK_WINDOW_S, created_at +
LINK_SLACK_S], closest first. No record -> the newest attempt's snapshot, and the result says so
per sample (`link`). Every sample's DIFF verdict and record score are then checked against the leg
JSON's recorded verdict; any disagreement is a `mismatch` and the leg is not reportable.

WARNINGS -> SAMPLES. A failure warning names the workbook it was about. The solver's output lives
at <tasks cache>/<task with / -> _>__s<k>/output/<file>, so "<ws>__s<k>/output/" names exactly one
sample. A warning counts for a sample only if it comes from the attempt that produced that
sample's verdict. A warning that names no single sample of this leg (the reference workbook's
scratch copy, which every sample of every task sharing it depends on; "Inline SSBv2 scoring failed
for <task>", which names the task but not the sample) is NOT guessed onto a sample: it goes to
`unattributed_warnings` and the leg's status becomes "unattributable_warning".

No package imports, stdlib only (like digest_coverage.py): the loop and the tests load it by path.

CLI:  python src/orchestrator/outcomes.py <run_dir> <leg_json> [--json] [--samples]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import NamedTuple

OUTCOMES = ("passed", "graded_wrong", "not_eligible", "grader_failed")

# The recalculation / data-table FAILURE warnings, plus the grader-raised line.
GRADER_FAILURE_PATTERNS = (
    "LibreOffice recalculation failed",
    "_solve_datatables_with_uno failed",
    "Inline SSBv2 scoring failed",
)

DIFF_HEADER_RE = re.compile(r"^(PASS|FAIL) \u2014 (\d+)/(\d+) graded cells match\s*$")
DIFF_PREAMBLE = "(scorer's own diff"
NO_OUTPUT_BODY = "Output file not found"
OUTPUT_UNREADABLE_PREFIX = "Cannot open workbook:"

# "<ws>__s<k>/output/" or ".../output_raw/" (the raw copy preserved before the in-place recalc).
_SAMPLE_PATH_RE = re.compile(r"/([^/\s'\"]+?)__s(\d+)/output(?:_raw)?/")
_SNAP_DIR_RE = re.compile(r"^iter_(\d+)_s(\d+)$")   # midgate snapshots (iter_N_midgate_sK) excluded

LINK_WINDOW_S = 600.0   # DIFF.txt may precede the cache record by the workspace copy; seconds in practice
LINK_SLACK_S = 5.0      # ... and never follow it (mtime granularity / clock rounding only)
PASS_THRESHOLD = 0.999  # the runner's own "pass" test (merge_testbase.py, run_ssbv2_base30_dsv4f_n3.py)
CACHE_HASH_LENGTH = 12  # src/cache/run_cache.py CacheConfig.hash_length


# ---------------------------------------------------------------------------------------------
# One sample: the DIFF.txt text + the warnings attributed to it -> one of OUTCOMES
# ---------------------------------------------------------------------------------------------

def parse_diff(diff_text: str | None) -> dict | None:
    """{"verdict": "PASS"|"FAIL", "n_match", "n_total", "body"} or None (no text / unknown header)."""
    if diff_text is None:
        return None
    lines = diff_text.splitlines()
    if not lines:
        return None
    m = DIFF_HEADER_RE.match(lines[0].lstrip("\ufeff"))
    if not m:
        return None
    rest = lines[1:]
    if rest and rest[0].startswith(DIFF_PREAMBLE):
        rest = rest[1:]
    return {"verdict": m.group(1), "n_match": int(m.group(2)), "n_total": int(m.group(3)),
            "body": "\n".join(rest).strip()}


def failure_warnings(lines) -> list[str]:
    """The lines that carry one of GRADER_FAILURE_PATTERNS."""
    return [ln for ln in lines if any(p in ln for p in GRADER_FAILURE_PATTERNS)]


def classify_detail(diff_text: str | None, warnings: list[str] | None = None) -> tuple[str, str]:
    """(outcome, reason). `warnings` = the runner-log lines attributed to THIS sample."""
    if failure_warnings(warnings or []):
        return "grader_failed", "failure_warning"
    if diff_text is None:
        return "grader_failed", "no_diff_recorded"
    d = parse_diff(diff_text)
    if d is None:
        # Not a form the grader writes: nothing trustworthy was recorded. Re-grade, never guess.
        return "grader_failed", "unrecognised_diff_header"
    if d["verdict"] == "PASS":
        return "passed", "pass"
    if d["n_match"] == 0 and d["n_total"] == 0 and d["body"] == NO_OUTPUT_BODY:
        return "not_eligible", "output_file_not_found"
    if d["n_total"] == 0 and d["body"].startswith(OUTPUT_UNREADABLE_PREFIX):
        # The reference workbooks are validated to open, and a golden refresh that cannot
        # be read falls back to the raw golden, so the unreadable file is the solver's output.
        return "graded_wrong", "output_unreadable"
    return "graded_wrong", "fail"


def classify(diff_text: str | None, warnings: list[str] | None = None) -> str:
    """One of OUTCOMES."""
    return classify_detail(diff_text, warnings)[0]


# ---------------------------------------------------------------------------------------------
# Warnings -> samples
# ---------------------------------------------------------------------------------------------

def ws_name(task_id: str, sample_idx: int) -> str:
    """Mirror of ssbv2_agent.task_workspace.task_ws_name (pinned by a parity test)."""
    return f"{task_id.replace('/', '_')}__s{sample_idx}"


def attribute_warning(line: str, ws_to_task: dict) -> tuple | None:
    """(task_id, sample_idx) when the line names exactly one sample's workbook of this leg, else None."""
    hits = set()
    for m in _SAMPLE_PATH_RE.finditer(line):
        task = ws_to_task.get(m.group(1))
        if task is None:
            return None          # a sample path, but not one of this leg's tasks: do not guess
        hits.add((task, int(m.group(2))))
    return next(iter(hits)) if len(hits) == 1 else None


def runner_log(attempt_dir: Path) -> Path | None:
    """The shard's runner log: <shard>/runner.log (the driver's copy), else <shard>/out/runner.log.
    Exactly one of the two, never both (they are the same stream)."""
    for p in (attempt_dir / "runner.log", attempt_dir / "out" / "runner.log"):
        if p.is_file():
            return p
    return None


# ---------------------------------------------------------------------------------------------
# Attempts, snapshots, cache records
# ---------------------------------------------------------------------------------------------

class Attempt(NamedTuple):
    """One shard of one attempt. NamedTuple, not a dataclass: a dataclass under `from __future__
    import annotations` needs its module in sys.modules, which a load-by-path does not give."""
    dir: Path      # tasks/_snapshots, runner.log, cache/
    rank: int      # 0 = the final attempt; larger = older
    label: str


def _prev_stamp(p: Path) -> str:
    return p.name.rsplit("_prev_", 1)[-1]


def discover_attempts(run_dir: Path, leg: dict, repo_root: Path | None = None) -> list[Attempt]:
    """Every shard-attempt dir of this leg, final attempts first.

    Loop leg (the result carries daytona.store): <store>/s*  then  <store>._prev_<stamp>/s*  (newest
    stamp first). Merged held-out baseline (result carries shards{...}): <run_dir>/<shard> then
    <run_dir>/<shard>/_prev_<stamp> (newest first) and any earlier_attempts[].dir."""
    run_dir = Path(run_dir)
    out: list[Attempt] = []
    dy = leg.get("daytona") if isinstance(leg.get("daytona"), dict) else None
    if (dy and dy.get("store")) or (not isinstance(leg.get("shards"), dict) and leg.get("tag")):
        if dy and dy.get("store"):
            store = Path(dy["store"])
        else:
            root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
            store = root / ".cache" / "ssbv2_daytona" / run_dir.name / leg["tag"]
        stores = [store] + sorted(store.parent.glob(f"{store.name}._prev_*"), key=_prev_stamp,
                                  reverse=True)
        for rank, st in enumerate(stores):
            for sd in sorted(p for p in st.glob("s[0-9]*") if p.is_dir()):
                out.append(Attempt(sd, rank, f"{st.name}/{sd.name}"))
        return out
    shards = leg.get("shards")
    if isinstance(shards, dict):
        for shard, meta in sorted(shards.items()):
            sd = run_dir / shard
            prevs = {p.resolve(): p for p in sd.glob("_prev_*") if p.is_dir()}
            for e in (meta or {}).get("earlier_attempts") or []:
                p = Path(e.get("dir") or "")
                if e.get("dir") and p.is_dir():
                    prevs.setdefault(p.resolve(), p)
            out.append(Attempt(sd, 0, shard))
            for rank, p in enumerate(sorted(prevs.values(), key=_prev_stamp, reverse=True), 1):
                out.append(Attempt(p, rank, f"{shard}/{p.name}"))
        return out
    raise ValueError("cannot locate this leg's attempts: the result has neither daytona.store nor "
                     "shards{}; pass attempt_dirs explicitly")


def snapshot_dirs(attempt: Attempt, task_id: str, sample_idx: int) -> list[Path]:
    root = attempt.dir / "tasks" / "_snapshots" / task_id.replace("/", "_")
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and (m := _SNAP_DIR_RE.match(p.name)) and int(m.group(2)) == sample_idx)


def snapshot_time(snap: Path) -> float:
    """When snapshot_output wrote this snapshot: DIFF.txt / LAYOUT.md mtime (the dir's own mtime
    moves when the leg driver downloads / merges it)."""
    ts = [p.stat().st_mtime for p in (snap / "DIFF.txt", snap / "LAYOUT.md") if p.is_file()]
    return max(ts) if ts else snap.stat().st_mtime


def _record_path_stem(question: str, sdk: str, model: str, effort: str | None, sample_idx: int,
                      n: int = CACHE_HASH_LENGTH) -> str:
    """Mirror of RunCache._get_cache_path's file stem (pinned by a parity test)."""
    keyed = f"{question.strip()}|sdk={sdk}|model={model}"
    if effort:
        keyed += f"|effort={effort}"
    keyed += f"|sample={sample_idx}"
    return hashlib.sha256(keyed.encode("utf-8")).hexdigest()[:n]


def index_cache_records(cache_dirs, n_samples: int) -> dict:
    """{(task_id, sample_idx): [record, ...]} over <cache_dir>/<tree>/<hash>.json.
    record = {"created_at", "inline_score", "path"}."""
    idx: dict = {}
    for cd in cache_dirs:
        cd = Path(cd)
        if not cd.is_dir():
            continue
        for f in sorted(cd.glob("*/*.json")):
            try:
                d = json.loads(f.read_text())
                ck = d["cache_key"]
                q = ck["question"]
                task = json.loads(q).get("task_id")
            except Exception:
                continue
            if not task:
                continue
            for k in range(n_samples):
                if _record_path_stem(q, ck.get("sdk", ""), ck.get("model", ""), ck.get("effort"),
                                     k, len(f.stem)) == f.stem:
                    rec = {"created_at": d.get("created_at"),
                           "inline_score": (d.get("trace") or {}).get("inline_score"),
                           "path": str(f)}
                    lst = idx.setdefault((task, k), [])
                    if not any(r["created_at"] == rec["created_at"] for r in lst):
                        lst.append(rec)
                    break
    return idx


def _created_epochs(created_at: str | None) -> list[float]:
    """created_at is a naive datetime.now(): UTC inside a Daytona sandbox, local time on a Mac leg.
    Both readings, UTC first."""
    if not created_at:
        return []
    try:
        t = _dt.datetime.fromisoformat(created_at)
    except ValueError:
        return []
    if t.tzinfo is not None:
        return [t.timestamp()]
    utc = t.replace(tzinfo=_dt.timezone.utc).timestamp()
    local = time.mktime(t.timetuple()) + t.microsecond / 1e6
    return [utc] if abs(utc - local) < 1 else [utc, local]


def link_snapshot(cands: list, record: dict) -> tuple | None:
    """The (attempt, snapdir) whose snapshot time sits just before the record's created_at."""
    for t in _created_epochs(record.get("created_at")):
        best = None
        for att, snap in cands:
            st = snapshot_time(snap)
            if t - LINK_WINDOW_S <= st <= t + LINK_SLACK_S:
                if best is None or abs(t - st) < best[0]:
                    best = (abs(t - st), att, snap)
        if best:
            return best[1], best[2]
    return None


# ---------------------------------------------------------------------------------------------
# One leg
# ---------------------------------------------------------------------------------------------

def _empty_counts() -> dict:
    return {o: 0 for o in OUTCOMES}


def leg_outcome_counts(run_dir, leg_json, *, attempt_dirs=None, repo_root=None,
                       use_cache_records: bool = True) -> dict:
    """Four-outcome counts of one leg, per task and per category, from its on-disk records.

    run_dir     the run dir the leg belongs to (outputs/<run>; for the merged baseline, the dir
                holding its shard dirs).
    leg_json    path to the leg's result JSON, or the loaded dict.
    attempt_dirs  optional explicit shard-attempt dirs, FINAL attempts first (overrides discovery).

    Returns {passed, graded_wrong, not_eligible, grader_failed, n_episodes, per_task{task: [outcome
    per sample]}, per_category{cat: counts}, unattributed_warnings[], superseded_warnings[],
    recorded_pass, diff_pass, pass_matches_recorded, mismatches[], samples[], links{}, attempts[],
    status, reportable}."""
    leg = json.loads(Path(leg_json).read_text()) if not isinstance(leg_json, dict) else leg_json
    run_dir = Path(run_dir)
    task_ids = list(leg["task_ids"])
    n_samples = int(leg.get("n_samples") or 3)
    per_task_scores = {task_ids[int(i)]: v for i, v in leg["per_task"].items()}
    ws_to_task = {t.replace("/", "_"): t for t in task_ids}

    if attempt_dirs is not None:
        attempts = [Attempt(Path(p), r, Path(p).name) for r, p in enumerate(attempt_dirs)]
    else:
        attempts = discover_attempts(run_dir, leg, repo_root)

    records = {}
    if use_cache_records:
        final_caches = [a.dir / "cache" for a in attempts if a.rank == 0]
        records = index_cache_records(final_caches, n_samples)

    # failure warnings, per attempt: attributed {(task, k): [line]} and unattributed [line]
    attributed: dict = {}
    unattributed: dict = {}
    for a in attempts:
        log = runner_log(a.dir)
        if log is None:
            continue
        for ln in failure_warnings(log.read_text(errors="replace").splitlines()):
            key = attribute_warning(ln, ws_to_task)
            if key is None:
                unattributed.setdefault(a, []).append(ln.strip())
            else:
                attributed.setdefault(a, {}).setdefault(key, []).append(ln.strip())

    counts, per_cat = _empty_counts(), {}
    per_task: dict = {}
    samples, mismatches, problems = [], [], []
    links: Counter = Counter()
    chosen_attempts = set()
    recorded_pass = diff_pass = 0
    for task in task_ids:
        cat = task.split("/", 1)[0]
        pc = per_cat.setdefault(cat, {**_empty_counts(), "n_episodes": 0})
        scores = per_task_scores.get(task) or []
        row = []
        for k in range(n_samples):
            score = scores[k] if k < len(scores) else None
            if score is None:
                problems.append(f"{task} s{k}: no recorded verdict in the leg JSON")
            rec_pass = score is not None and score >= PASS_THRESHOLD
            recorded_pass += rec_pass
            cands = [(a, s) for a in attempts for s in snapshot_dirs(a, task, k)]
            # newest = the most recent attempt's snapshot (latest write within it)
            newest = max(cands, key=lambda c: (-c[0].rank, snapshot_time(c[1]))) if cands else None
            recs = records.get((task, k)) or []
            record = recs[0] if len(recs) == 1 else None
            chosen, link = None, "none"
            if record is not None:
                hit = link_snapshot(cands, record)
                if hit:
                    chosen, link = hit, "cache_record"
                else:
                    link = "cache_record_without_snapshot"
            elif cands:
                chosen, link = newest, ("newest_snapshot" if not recs else "ambiguous_records_newest_snapshot")
            links[link] += 1

            diff_text, warns, att = None, [], None
            if chosen is not None:
                att, snap = chosen
                chosen_attempts.add(att)
                f = snap / "DIFF.txt"
                diff_text = f.read_text(errors="replace") if f.is_file() else None
                warns = attributed.get(att, {}).get((task, k), [])
            outcome, reason = classify_detail(diff_text, warns)
            d = parse_diff(diff_text)
            if d and d["verdict"] == "PASS":
                diff_pass += 1
            rec_score = record.get("inline_score") if record else None
            bad = []
            if score is not None and d is not None and (d["verdict"] == "PASS") != rec_pass:
                bad.append(f"DIFF says {d['verdict']}, leg JSON records {score}")
            if score is not None and record is not None and \
                    ((rec_score or 0.0) >= PASS_THRESHOLD) != rec_pass:
                bad.append(f"cache record inline_score {rec_score}, leg JSON records {score}")
            if bad:
                mismatches.append({"task": task, "sample": k, "problems": bad,
                                   "diff": str(chosen[1] / "DIFF.txt") if chosen else None})
            row.append(outcome)
            counts[outcome] += 1
            pc[outcome] += 1
            pc["n_episodes"] += 1
            samples.append({"task": task, "sample": k, "outcome": outcome, "reason": reason,
                            "recorded": score, "link": link,
                            "attempt": att.label if att else None,
                            "snapshot": str(chosen[1]) if chosen else None,
                            "warnings": warns})
        per_task[task] = row

    # Warnings that concern a graded verdict of this leg but name no single sample.
    unattr_out = [{"attempt": a.label, "line": ln}
                  for a, lines in unattributed.items() if a in chosen_attempts or a.rank == 0
                  for ln in lines]
    chosen_by_sample = {(s["task"], s["sample"]): s["attempt"] for s in samples}
    superseded = [{"attempt": a.label, "task": t, "sample": k, "line": ln}
                  for a, m in attributed.items() for (t, k), lines in m.items()
                  if chosen_by_sample.get((t, k)) != a.label for ln in lines]

    if mismatches or problems:
        status = "mismatch"
    elif unattr_out:
        status = "unattributable_warning"
    elif counts["grader_failed"]:
        status = "grader_failed"
    else:
        status = "ok"
    return {
        "tag": leg.get("tag"),
        **counts,
        "n_episodes": sum(counts.values()),
        "per_task": per_task,
        "per_category": per_cat,
        "unattributed_warnings": unattr_out,
        "superseded_warnings": superseded,
        "recorded_pass": recorded_pass,
        "recorded_sample_pass": leg.get("sample_pass"),
        "diff_pass": diff_pass,
        "pass_matches_recorded": counts["passed"] == recorded_pass == diff_pass
                                 and (leg.get("sample_pass") is None or leg["sample_pass"] == recorded_pass),
        "passed_but_grader_failed": sum(1 for s in samples
                                        if s["outcome"] == "grader_failed" and s["recorded"] is not None
                                        and s["recorded"] >= PASS_THRESHOLD),
        "mismatches": mismatches,
        "problems": problems,
        "links": dict(links),
        "attempts": [a.label for a in attempts],
        "samples": samples,
        "status": status,
        "reportable": status == "ok",
    }


def summary_line(r: dict) -> str:
    return (f"{r['tag']}: passed {r['passed']}  graded_wrong {r['graded_wrong']}  not_eligible "
            f"{r['not_eligible']}  grader_failed {r['grader_failed']}  (n={r['n_episodes']}; recorded "
            f"pass {r['recorded_sample_pass']}, DIFF pass {r['diff_pass']}; match="
            f"{r['pass_matches_recorded']}; links {r['links']}; unattributed "
            f"{len(r['unattributed_warnings'])}; mismatches {len(r['mismatches'])}) -> {r['status']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("run_dir")
    ap.add_argument("leg_json")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    ap.add_argument("--samples", action="store_true", help="with --json: keep the per-sample list")
    ap.add_argument("--no-cache-records", action="store_true",
                    help="link by newest snapshot only (skip the cache-record link)")
    a = ap.parse_args(argv)
    r = leg_outcome_counts(a.run_dir, a.leg_json, use_cache_records=not a.no_cache_records)
    if a.json:
        if not a.samples:
            r = {k: v for k, v in r.items() if k != "samples"}
        print(json.dumps(r, indent=1))
    else:
        print(summary_line(r))
        for cat, c in sorted(r["per_category"].items()):
            print(f"  {cat}: " + "  ".join(f"{o} {c[o]}" for o in OUTCOMES) + f"  (n={c['n_episodes']})")
        for w in r["unattributed_warnings"]:
            print(f"  UNATTRIBUTED [{w['attempt']}] {w['line'][:300]}")
        for m in r["mismatches"]:
            print(f"  MISMATCH {m}")
    return 0 if r["reportable"] else 3


if __name__ == "__main__":
    sys.exit(main())
