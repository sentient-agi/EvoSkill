"""The evolver's per-epoch material, built on the host from a Claude Code leg's store. Rollouts land in one directory
whatever the harness: each solver's session log, a snapshot of its own file system at the time it exits, and the task id,
sample and score only:

    <run>/evo_e{K}/traces/<id>_s<i>.jsonl           the solver harness's session log: Claude Code's own session transcript,
                                                    every user message included (bytes)
    <run>/evo_e{K}/traces/<id>_s<i>/                the files that log refers to in the harness's own directory (large
                                                    tool outputs, tool-results/*.txt); only when the session wrote any
    <run>/evo_e{K}/solver_workspaces/<id>_s<i>/     the solver's working directory exactly as it was when it exited
                                                    (the unit's container/evo/work tree)
    <run>/evo_e{K}/scores.json                      [{"task_id": <id>, "sample": <i>, "score": 0..1 | "grader_failed"}]

K = the epoch whose rollouts these are: 0 = the base train leg (the solver with no skills), K >= 1 = skills_vK's train
leg. The loop uploads them into the evolver's container as /evo/traces/epoch_K/, /evo/solver_workspaces/epoch_K/ and
/evo/scores/epoch_K.json (stateful_layout). <id> is the file-safe task id (the task id with "/" -> "_"), the SAME string
in every file name and in scores.json (the evolver never needs the benchmark's own spelling); <i> = the sample index the
solver's unit carried (1..N). The score is a number from 0 to 1 (a score generalises across graders better than a
pass/fail verdict), or "grader_failed" when the grader itself failed. The score is the run's own grader's, grade(task, snapshot), as the leg driver recorded it per unit in STATE.json, copied here VERBATIM: the loop maps
no verdict (a rollout that saved nothing scores whatever the grader returns for it), and a rollout whose grade() raised
twice stays "grader_failed" (a grader fault says nothing about the solver, so it is never a 0). Nothing else about the
grade reaches the evolver.

The source is the leg JSON's store (leg["daytona"]["store"]: <store>/STATE.json units[<task>#s<i>].score and
<store>/runs/<id>__s<i>/{container/home/evo/.claude/projects/<cwd>/<session>.jsonl + <session>/, container/evo/work/}),
exactly what scripts/daytona/cc_leg_driver.py wrote. Nothing is converted, digested or summarised here; a graded unit
whose score is anything else (a verdict-era row, a number outside 0..1) is refused, naming the unit.
The trace is the session transcript, not the unit's stream.jsonl (`claude -p --output-format stream-json`): the stream
never carries the first user message -- the solver's whole instruction and the task text -- and the trace must contain
every user message. Claude Code's session transcript carries every message the stream carries plus the user messages.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path

SCORES_JSON = "scores.json"
TRACES = "traces"
WORKSPACES = "solver_workspaces"
DIR_NAME = "evo_e{epoch}"
GRADER_FAILED = "grader_failed"      # = plugin_api.GRADER_FAILED: the STATE score of a rollout whose grade() raised twice
CONTAINER_WORK = ("container", "evo", "work")      # the solver's cwd inside its container, as the driver extracts it
CC_PROJECTS = ("container", "home", "evo", ".claude", "projects")   # Claude Code's session transcripts, as extracted


class LayoutError(RuntimeError):
    pass


def epoch_dir(run_dir, epoch: int) -> Path:
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError(f"epoch {epoch!r}: an integer >= 0")
    return Path(run_dir) / DIR_NAME.format(epoch=epoch)


def safe_id(task_id: str) -> str:
    return str(task_id).replace("/", "_")


def unit_of(task_id: str, sample: int) -> str:
    return f"{safe_id(task_id)}_s{int(sample)}"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_score(s) -> bool:
    """A STATE score as the leg driver records it: a finite number from 0 to 1 (not a bool), or GRADER_FAILED."""
    if isinstance(s, str):
        return s == GRADER_FAILED
    return isinstance(s, (int, float)) and not isinstance(s, bool) and math.isfinite(s) and 0 <= s <= 1


def _units_of(leg: dict, store: Path) -> list:
    """[(task_id, sample, score, unit_run_dir)] for every rollout the leg measured, from the store's STATE.json."""
    st_p = store / "STATE.json"
    if not st_p.is_file():
        raise LayoutError(f"{st_p} missing: the leg's store holds no state record")
    st = json.loads(st_p.read_text(encoding="utf-8"))
    units = st.get("units") or {}
    n = int(leg.get("n_samples") or 0)
    out = []
    for task in leg.get("task_ids") or []:
        for k in range(1, n + 1):
            unit = f"{task}#s{k}"
            row = units.get(unit)
            if not isinstance(row, dict) or row.get("state") != "graded":
                raise LayoutError(f"{unit}: not graded in the store ({(row or {}).get('state')!r})")
            s = row.get("score")
            if not _is_score(s):
                raise LayoutError(f"{unit}: score {s!r} is not a number from 0 to 1 or {GRADER_FAILED!r}")
            out.append((task, k, s, store / "runs" / f"{safe_id(task)}__s{k}"))
    return out


def transcript_of(rd: Path) -> tuple:
    """(the unit's Claude Code session transcript, its companion dir or None). Exactly one transcript per unit."""
    d = rd.joinpath(*CC_PROJECTS)
    found = sorted(d.glob("*/*.jsonl")) if d.is_dir() and not d.is_symlink() else []
    if len(found) != 1 or found[0].is_symlink() or not found[0].is_file():
        raise LayoutError(f"{rd.name}: {len(found)} session transcript(s) under {d} (exactly one expected)")
    comp = found[0].with_suffix("")
    return found[0], (comp if comp.is_dir() and not comp.is_symlink() else None)


def _tree(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): _sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}


def _store_of(leg_json) -> tuple:
    leg = json.loads(Path(leg_json).read_text(encoding="utf-8"))
    store = Path(str(((leg.get("daytona") or {}).get("store")) or ""))
    if not store.is_dir():
        raise LayoutError(f"the leg {Path(leg_json).name} names no store dir on this Mac ({store})")
    return leg, store


def _copy_tree(src: Path, dst: Path) -> int:
    """Copy a directory tree with no symlink followed (a symlink anywhere is refused). Returns the file count."""
    n = 0
    for dirpath, dirnames, filenames in os.walk(src, followlinks=False):
        rel = Path(dirpath).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for d in dirnames:
            if (Path(dirpath) / d).is_symlink():
                raise LayoutError(f"symlink in the solver workspace: {Path(dirpath, d).relative_to(src)}")
        for f in filenames:
            p = Path(dirpath) / f
            if p.is_symlink():
                raise LayoutError(f"symlink in the solver workspace: {p.relative_to(src)}")
            if p.is_file():
                shutil.copyfile(p, dst / rel / f)
                n += 1
    return n


def build(out, *, leg_json, epoch: int) -> dict:
    """Write <out>/{traces/, solver_workspaces/, scores.json} from the leg's store. Refuses an existing `out`
    (evidence an evolver may have read is never rebuilt in place). Returns a summary."""
    out = Path(out)
    if out.exists():
        raise LayoutError(f"{out} exists: move it aside (never delete material an evolver read) and rebuild")
    leg, store = _store_of(leg_json)
    units = _units_of(leg, store)
    tmp = out.with_name(out.name + f".building.{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / TRACES).mkdir(parents=True)
    (tmp / WORKSPACES).mkdir(parents=True)
    rows, n_files, n_bytes = [], 0, 0
    try:
        for task, k, score, rd in units:
            log, comp = transcript_of(rd)
            work = rd.joinpath(*CONTAINER_WORK)
            if not work.is_dir() or work.is_symlink():
                raise LayoutError(f"{task}#s{k}: no solver workspace at {work}")
            u = unit_of(task, k)
            shutil.copyfile(log, tmp / TRACES / f"{u}.jsonl")
            n_bytes += log.stat().st_size
            n_files += 1 + _copy_tree(work, tmp / WORKSPACES / u)
            if comp is not None:
                n_files += _copy_tree(comp, tmp / TRACES / u)
            rows.append({"task_id": safe_id(task), "sample": k, "score": score})
        (tmp / SCORES_JSON).write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
        os.rename(tmp, out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return {"dir": str(out), "epoch": int(epoch), "leg": str(leg_json), "store": str(store), **_score_counts(rows),
            "files": n_files + 1, "trace_bytes": n_bytes}


def check(out, *, leg_json) -> list:
    """Problems ([] = the dir is exactly what build() writes from the leg's store now): every measured rollout has
    its trace (byte-identical to the unit's session transcript, its companion files identical) and its workspace dir, scores.json holds exactly the
    store's STATE scores in leg order, nothing else is in traces/ or scores.json, no symlink anywhere."""
    out = Path(out)
    probs = []
    if not out.is_dir():
        return [f"{out} is not a directory"]
    try:
        leg, store = _store_of(leg_json)
        units = _units_of(leg, store)
    except (LayoutError, OSError, ValueError) as e:
        return [str(e)]
    sp = out / SCORES_JSON
    try:
        rows = json.loads(sp.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [f"{sp} unreadable ({type(e).__name__})"]
    want = [{"task_id": safe_id(t), "sample": k, "score": s} for t, k, s, _ in units]
    if rows != want:
        probs.append(f"{SCORES_JSON} differs from the store's scores ({len(rows)} rows vs {len(want)} rollouts)")
    traces = sorted(p.name for p in (out / TRACES).iterdir()) if (out / TRACES).is_dir() else None
    if traces is None:
        probs.append(f"no {TRACES}/ dir")
    else:
        allowed = {f"{unit_of(t, k)}.jsonl" for t, k, _, _ in units} | {unit_of(t, k) for t, k, _, _ in units}
        extra = sorted(set(traces) - allowed)
        if extra:
            probs.append(f"{len(extra)} file(s) in {TRACES}/ that are no measured rollout: {extra[:3]}")
    for task, k, _s, rd in units:
        u = unit_of(task, k)
        tr = out / TRACES / f"{u}.jsonl"
        try:
            src, comp = transcript_of(rd)
        except LayoutError as e:
            probs.append(str(e))
            continue
        if tr.is_symlink() or not tr.is_file():
            probs.append(f"{u}: trace missing")
        elif _sha256(tr) != _sha256(src):
            probs.append(f"{u}: trace differs from the unit's session transcript")
        cd = out / TRACES / u
        if (comp is None) != (not cd.exists()) or (comp is not None and (cd.is_symlink() or _tree(cd) != _tree(comp))):
            probs.append(f"{u}: the trace's companion files differ from the session's")
        ws = out / WORKSPACES / u
        if ws.is_symlink() or not ws.is_dir():
            probs.append(f"{u}: solver workspace missing")
        if len(probs) > 20:
            probs.append("...")
            break
    for p in out.rglob("*"):
        if p.is_symlink():
            probs.append(f"symlink in the layout: {p.relative_to(out)}")
            break
    return probs


def _score_counts(rows) -> dict:
    scored = [r["score"] for r in rows if r["score"] != GRADER_FAILED]
    return {"rollouts": len(rows), "scored": len(scored), "grader_failed": len(rows) - len(scored),
            "mean_score": round(sum(scored) / len(scored), 4) if scored else None}


def summary(out) -> dict:
    out = Path(out)
    return _score_counts(json.loads((out / SCORES_JSON).read_text(encoding="utf-8")))
