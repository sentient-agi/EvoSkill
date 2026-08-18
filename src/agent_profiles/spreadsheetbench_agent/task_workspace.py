"""Per-task workspace management for SpreadsheetBench (v1).

Each task gets an isolated directory that is bind-mounted as the SOLVER's
container workspace — so it must contain ONLY what the solver is allowed
to see (no golden answer, no cross-iteration history):

    .cache/ssb_tasks/{task_id}/
        spreadsheets/       <- input xlsx copied from the task dir
        output/             <- solver writes {tc}_{task_id}_output.xlsx here
        .claude/skills/     <- symlink to shared evolving skills dir

Evolver-only data lives OUTSIDE the solver's mount, under a sibling root
that only the evolver can reach (its container mounts the whole
`ssb_tasks/` root at /workspace/solver_outputs; each solver's mount is
scoped to its own task dir):

    .cache/ssb_tasks/_snapshots/{task_id}/iter_N/   <- per-iteration output
                                                       snapshots (+ DIFF.txt)

Golden answers are NOT staged anywhere the solver can see (v1 never staged
them; kept deliberately after the Jul 1 SSB v2 trace audit caught solvers
opening a staged golden file mid-rollout — GT leakage). The evolver reads
golden files from its read-only dataset mount (/workspace/ssb_data/...),
with the exact paths given in oracle/tasks/{task_id}.json.

Layout and conventions mirror src/agent_profiles/ssbv2_agent/task_workspace.py
(the SSB v2 environment-aware-evolver stack); differences are v1's scoring
mechanism only — see build_oracle_spec.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from pathlib import Path

_log = logging.getLogger(__name__)

# Generic per-task/per-root locking lives in src.harness.task_workspace,
# shared across every benchmark — see that module's docstring for why.

SNAPSHOTS_ROOT_NAME = "_snapshots"


def build_oracle_spec(ssb_root: Path, allowed_task_ids=None):
    """SSB v1's oracle declaration — see src/loop/oracle.py.

    Shared by the run scripts so they can't drift. Modeled on the SSB v2
    spec (src/agent_profiles/ssbv2_agent/task_workspace.py) with v1's
    scoring semantics: the benchmark's own Docker verifier (test.sh +
    evaluate.py inside ssb-verifier), per-test-case REQUIRED output file
    names, and a soft (mean-over-cases) reward.
    """
    from src.loop.oracle import OracleSpec

    ssb_root = Path(ssb_root)

    # Harness-side scorer: invokes the Docker verifier for the authoritative
    # reward and provides compare_workbooks_detailed (an exact port of
    # evaluate.py's comparison) for DIFF.txt rendering / what-if checks.
    wrapper_src = (
        Path(__file__).resolve().parents[2] / "evaluation" / "spreadsheetbench_scorer.py"
    )
    # In-container scoring authority. evaluate.py and test.sh are
    # byte-identical across all 400 task dirs (verified by md5, 2026-07-02),
    # so one representative copy stands for every task.
    eval_candidates = sorted(ssb_root.glob("*/tests/evaluate.py"))
    if not eval_candidates:
        raise FileNotFoundError(f"No */tests/evaluate.py found under {ssb_root}")
    evaluate_src = eval_candidates[0]
    test_sh_src = evaluate_src.parent / "test.sh"

    return OracleSpec(
        scoring_note=(
            "Scoring runs the benchmark's own Docker verifier (identical "
            "`test.sh` + `evaluate.py` in every task's tests/ dir; copies "
            "here): `test.sh` formula-recalculates BOTH the solver's output "
            "files and the golden answer files with LibreOffice, then "
            "`evaluate.py` compares cell VALUES only — no formatting, fill, "
            "or font is graded — inside each task's `answer_position` "
            "ranges. Value rules (see `evaluate.py`): numbers rounded to 2 "
            "decimals; datetimes -> Excel serial rounded to 0 decimals; "
            "times -> string with the last 3 chars dropped; strings that "
            "parse as numbers are compared as numbers; '' equals None; "
            "otherwise type-strict. `answer_position` can hold several "
            "comma-separated ranges with quirky sheet-name quoting, "
            "including column-only ranges like 'A:G' (expanded to rows "
            "1..max_row of either workbook). A task has `num_test_cases` "
            "graded cases (1 for EVERY task in this dataset; the mechanism "
            "supports more). Case `tc` compares "
            "`output/{tc}_{task_id}_output.xlsx` against golden "
            "`{tc}_{task_id}_answer.xlsx` — the output file NAME is "
            "required verbatim; a missing or misnamed file fails the case. "
            "Within a case a single wrong cell fails the ENTIRE case — no "
            "partial credit, so a sheet can be 99%+ cell-correct and still "
            "score 0 from one cascading formula error. The task reward is "
            "the MEAN over its cases, so with 1 case it is strictly 0 or 1. "
            "Files you see in `output/` (and in snapshots) are already "
            "formula-recalculated in place by the harness at scoring time — "
            "diff them as-is. If you create xlsx files with formulas "
            "yourself, openpyxl will NOT compute their values (no recalc "
            "engine in this container)."
        ),
        scoring_code=[
            evaluate_src,
            test_sh_src,
            (wrapper_src, "a host-side PORT of evaluate.py's comparison logic "
             "used to render DIFF.txt and for your own what-if checks — NOT "
             "literally what the Docker verifier executes (that's evaluate.py "
             "inside the ssb-verifier container)"),
        ],
        # Moved out of the launchers' evolver-prompt text 2026-07-24: these are
        # SSB-instance path/artifact facts, so they belong beside the scoring
        # code they describe, not in the universal prompt. The prompt keeps only
        # the generic pointer ("read ./oracle/README.md first") plus the
        # benchmark-agnostic pinned-path/live-dir contract.
        artifact_note=(
            "Per task, relative to the `Task workspace` path pinned in that "
            "task's block (`$WS` below — copy the pinned path verbatim):\n"
            "- `$WS/spreadsheets/` — the input xlsx the solver received\n"
            "- `$WS/output/` — the xlsx the solver produced, formula-"
            "recalculated IN PLACE at scoring time, so these are the graded "
            "bytes\n"
            "- `./solver_outputs/_snapshots/{task_id}/...` — per-sample output "
            "snapshots, each carrying the scorer's own `DIFF.txt`\n"
            "- `./oracle/tasks/{task_id}.json` — graded ranges "
            "(`answer_position`), `golden_paths`, and `expected_output_names`\n\n"
            "**Path dialects inside solver traces.** The solver's cwd WAS its "
            "own task workspace, so its traces say `./spreadsheets/` and "
            "`./output/`, and occasionally the absolute container forms "
            "`/app/spreadsheets/` and `/app/output/`. Map either onto "
            "`$WS/...`. Oracle `golden_paths` use `/workspace/ssb_data/...`, a "
            "Docker mount readable from the evolver workspace exactly as "
            "written.\n\n"
            "**Artifact-level failure mode to check before diagnosing "
            "contents.** The verifier accepts only the exact filename "
            "`{tc}_{task_id}_output.xlsx` per test case (`{tc}` = 1-based "
            "test-case number; the accepted names are in "
            "`expected_output_names`). A solver that wrote a perfectly correct "
            "sheet under the wrong name scores 0 — so when a DIFF reports "
            "EVERY graded cell missing, check the filename first."
        ),
        diff_recipe=(
            "EVERY snapshot dir already contains `DIFF.txt` — the scorer's own "
            "diff for that iteration's output, computed at scoring time, full "
            "and uncapped: `./solver_outputs/_snapshots/{task_id}/iter_N/DIFF.txt` "
            "(mid-gate re-evals use `iter_N_midgate` so they do not overwrite "
            "train/base DIFFs). "
            "READ THOSE FIRST — they are free and authoritative; do NOT re-run "
            "comparisons just to see what failed.\n\n"
            "**LibreOffice golden-error heal (harness-side):** after LO "
            "recalculates a golden, any graded cell where LO wrote an Excel "
            "error token (`#VALUE!`, `#NAME?`, ...) but the unrecalculated "
            "golden still has a usable Excel-cached value is rewritten to "
            "that cached value before compare (host DIFF and Docker verifier "
            "both use the healed copy under `_answers_recalc/`). So a solver "
            "that writes the correct number must PASS even when LO cannot "
            "evaluate the golden formula. Residual `expected='#…'` in DIFF.txt "
            "means heal could not recover a cached value (no Excel cache / "
            "orig also errored) — still not a fixable solver mistake. "
            "`expected=None` is NOT this signal by itself — check the cell "
            "content before calling it an artifact.\n\n"
            "Only compute a diff yourself for what-if checks (e.g. after you "
            "modify a workbook, or to double-check a DIFF.txt classification). "
            "When you do, use the scorer's own function — do NOT hand-roll "
            "workbook comparisons (your rounding/type/range rules WILL "
            "differ; only this parser handles `answer_position` quirks like "
            "column-only ranges) and do NOT diff cells outside answer_position "
            "(they are not graded). CRITICAL: `m['golden_paths']` (from "
            "`oracle/tasks/{task_id}.json`) points at the STORED, "
            "un-recalculated golden — comparing against it directly will "
            "DISAGREE with DIFF.txt and can misclassify a genuine LO artifact "
            "as a real solver failure (or vice versa). Use "
            "`solver_outputs/_answers_recalc/{task_id}/{golden_filename}` "
            "instead — the exact LibreOffice-recalculated golden DIFF.txt "
            "itself was generated from (same cache, written lazily the first "
            "time any task is scored; present for every task in this batch "
            "since scoring already ran). Complete example (note the types: "
            "each meta `m` is a dict loaded from its JSON file; `diff` is a "
            "plain STRING summary — print it, do not index into it):\n\n"
            "```python\n"
            "import sys, json, os\n"
            "sys.path.insert(0, './oracle')\n"
            "from spreadsheetbench_scorer import compare_workbooks_detailed\n"
            "for task_id in ['10452', '32438']:  # use EXACT ids from the failure list\n"
            "    m = json.load(open(f'oracle/tasks/{task_id}.json'))  # dict: task_id, answer_position, golden_paths, expected_output_names, ...\n"
            "    out_dir = f'solver_outputs/{task_id}/output'\n"
            "    for golden, out_name in zip(m['golden_paths'], m['expected_output_names']):\n"
            "        out = f'{out_dir}/{out_name}'\n"
            "        if not os.path.exists(out):\n"
            "            print(task_id, out_name, 'MISSING (the verifier requires this EXACT name) — files present:',\n"
            "                  os.listdir(out_dir) if os.path.isdir(out_dir) else 'no output dir'); continue\n"
            "        recalc_golden = f'solver_outputs/_answers_recalc/{task_id}/' + os.path.basename(golden)\n"
            "        if not os.path.exists(recalc_golden):\n"
            "            print(task_id, 'no recalculated golden cached yet (task never scored this run) — falling back to stored (may disagree with DIFF.txt)')\n"
            "            recalc_golden = golden\n"
            "        passed, n_match, n_total, diff = compare_workbooks_detailed(recalc_golden, out, m['answer_position'])\n"
            "        print(task_id, out_name, passed, f'{n_match}/{n_total} graded cells match')\n"
            "        print(diff)  # STRING; 20 mismatches by default — pass max_mismatches=0 to list ALL\n"
            "```\n\n"
            "Mismatches are listed in range order, not causal order — on a "
            "cascade failure the first 20 may all be downstream symptoms. Pass "
            "`max_mismatches=0` (no cap) when you need the full picture. "
            "IMPORTANT: diff in batches of ~10-15 tasks per bash call — each "
            "task loads two workbooks (seconds each for big models), so one "
            "call covering all failures will exceed the bash timeout and lose "
            "its output."
        ),
        # allowed_task_ids: pass the TRAIN split so val/test tasks get no meta
        # file at all (0725 contamination control — see write_all_task_meta).
        write_task_meta=lambda tasks_dir: write_all_task_meta(
            tasks_dir, ssb_root, allowed_task_ids=allowed_task_ids),
    )


def write_all_task_meta(
    meta_dir: Path,
    ssb_root: Path,
    container_data_root: str = "/workspace/ssb_data",
    allowed_task_ids: set[str] | frozenset[str] | None = None,
) -> int:
    """Write per-task metadata JSON files for the evolver.

    One `{task_id}.json` per task with the machine-readable inputs the
    scorer's diff needs: answer_position (the graded ranges), per-test-case
    golden paths (container paths) and the REQUIRED output file names.
    Scans ssb_root directly (same discovery rule as the dataset loader: a
    task dir has tests/config.json). `allowed_task_ids` RESTRICTS which tasks
    get a meta file — pass the TRAIN split and val/test tasks get none.

    That restriction is a contamination control, not tidiness (user directive
    0725: "oracle for val and test shouldn't be accessible by evolver by any
    means"). Before it, this wrote all 400 tasks regardless of split, so the
    evolver could open `oracle/tasks/<any-val-id>.json` and follow its
    golden_paths into the dataset mount. Removing the pointers is only half
    the fix — the mount itself must also be train-only (see
    `stage_split_dataset`), or the road survives without the signpost.

    Written into the benchmark's oracle/tasks/ dir — inside the
    evolver's workspace, so never solver-visible (solvers mount only their
    own task dir). Golden paths are translated from the host dataset path
    to the evolver container's read-only mount.
    """
    meta_dir = Path(meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    ssb_root = Path(ssb_root)
    host_prefix = str(ssb_root.resolve())
    n = 0
    for cfg_path in sorted(ssb_root.glob("*/tests/config.json")):
        try:
            cfg = json.loads(cfg_path.read_text())
        except (ValueError, OSError):
            continue
        task_id = str(cfg.get("id") or cfg_path.parent.parent.name)
        if allowed_task_ids is not None and task_id not in allowed_task_ids:
            continue
        # evaluate.py's own default when the field is absent is 3 (all 400
        # local tasks carry an explicit value, and it is always 1).
        num_cases = int(cfg.get("num_test_cases", 3))
        answer_dir = cfg_path.parent / "answer_files"
        golden_paths: list[str] = []
        output_names: list[str] = []
        for tc in range(1, num_cases + 1):
            golden = (answer_dir / f"{tc}_{task_id}_answer.xlsx").resolve()
            golden_paths.append(str(golden).replace(host_prefix, container_data_root))
            output_names.append(f"{tc}_{task_id}_output.xlsx")
        (meta_dir / f"{task_id}.json").write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "instruction_type": cfg.get("instruction_type", "unknown"),
                    "answer_position": cfg.get("answer_position", ""),
                    "num_test_cases": num_cases,
                    "golden_paths": golden_paths,
                    "expected_output_names": output_names,
                },
                indent=1,
            )
        )
        n += 1
    return n


def stage_split_dataset(
    ssb_root: Path,
    dest: Path,
    allowed_task_ids: set[str] | frozenset[str],
) -> tuple[int, int]:
    """Mirror ONLY `allowed_task_ids`' task dirs into `dest`, for the evolver mount.

    The launcher used to bind-mount the whole dataset read-only
    (`{SSB_ROOT}:/workspace/ssb_data:ro`), which put every val and test task's
    golden answer one path away from an agent whose only restraint was a
    sentence in its prompt. Mount THIS instead: val/test task dirs simply do
    not exist inside it, so no instruction is load-bearing.

    Files are HARD-LINKED, not copied — same inode, so an 80-task mirror costs
    kilobytes of directory entries rather than duplicating the dataset, and it
    can be rebuilt per run without thought. Falls back to a copy when the
    destination is on a different filesystem (os.link raises OSError/EXDEV).

    Returns (tasks_mirrored, files_linked).
    """
    ssb_root, dest = Path(ssb_root), Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    n_tasks = n_files = 0
    for cfg_path in sorted(ssb_root.glob("*/tests/config.json")):
        task_dir = cfg_path.parent.parent
        if task_dir.name not in allowed_task_ids:
            continue
        n_tasks += 1
        for src in task_dir.rglob("*"):
            if not src.is_file():
                continue
            tgt = dest / task_dir.name / src.relative_to(task_dir)
            tgt.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(src, tgt)
            except OSError:
                shutil.copy2(src, tgt)
            n_files += 1
    return n_tasks, n_files


def stage_inputs(task_dir: Path, ss_dir: Path) -> None:
    """Copy a task's input spreadsheets into ss_dir (skip if already staged).

    Inputs are static dataset files. Shared by setup_task_workspace and the
    run script's cache_on_replay hook: on a cache hit the solver agent never
    runs, so artifact restore alone would leave the workspace with output/
    but no spreadsheets/ — breaking the evolver's `$WS/spreadsheets/`
    manifest contract (the SSB v2 Jul 1 lesson: 140/140 replayed dirs were
    input-less when the iter-1 evolver ran).
    """
    ss_dir = Path(ss_dir)
    if ss_dir.exists() and any(ss_dir.iterdir()):
        return
    ss_dir.mkdir(parents=True, exist_ok=True)
    for src in [Path(task_dir) / "environment" / "spreadsheets",
                Path(task_dir) / "tests" / "spreadsheets"]:
        if src.is_dir():
            for f in src.iterdir():
                if f.is_file():
                    shutil.copy2(f, ss_dir / f.name)
            break


def task_ws_name(task_id: str, sample_idx: int | None = None) -> str:
    """Workspace dir name for a rollout — per SAMPLE when resampling.

    With ``n_samples>1`` every sample of a task used to share one workspace,
    and ``setup_task_workspace`` rmtree's ``output/`` on entry — so a second
    sample starting up DELETED a still-running sibling's output directory.
    Verified 2026-07-24 on a real N=3 run: task 398-14 sample 0 ran 22 turns,
    scored, and left no snapshot because a sibling wiped its output. Beyond
    losing artifacts that also means a sample could be SCORED against another
    sample's output file (both wrote ``1_{task_id}_output.xlsx`` to the same
    path), so parallel resampling results were not trustworthy.

    ``sample_idx=None`` keeps the exact historical name, so every
    ``n_samples=1`` run is byte-for-byte unchanged.
    """
    return task_id if sample_idx is None else f"{task_id}__s{sample_idx}"


def setup_task_workspace(
    task_id: str,
    task_dir: Path,
    tasks_cache_root: Path,
    shared_skills_dir: Path,
    sample_idx: int | None = None,
) -> Path:
    """Create/reset a per-task (per-SAMPLE when resampling) workspace.

    The workspace is the solver's project_root — Docker mounts it at
    /workspace. Cross-iteration snapshots live OUTSIDE this dir (see module
    docstring), so output/ is reset completely on every rollout — which is
    exactly why concurrent samples need their own workspace (see
    ``task_ws_name``).
    """
    ws = tasks_cache_root / task_ws_name(task_id, sample_idx)

    out_dir = ws / "output"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_inputs(task_dir, ws / "spreadsheets")

    # Symlink skills dir so the solver loads the shared evolving skills
    skills_link = ws / ".claude" / "skills"
    skills_link.parent.mkdir(parents=True, exist_ok=True)
    if skills_link.is_symlink() or skills_link.exists():
        if skills_link.is_symlink():
            skills_link.unlink()
        else:
            shutil.rmtree(skills_link)
    skills_link.symlink_to(shared_skills_dir.resolve())

    return ws


# Never snapshot these: `.claude/skills` is a SYMLINK to the shared, evolving
# skills dir (see build_task_workspace). Following it would copy the skill set
# into all 60 per-sample snapshots AND capture state that changes every
# iteration — destroying the immutability the snapshot exists to provide.
_SNAPSHOT_EXCLUDE_DIRS = frozenset({".claude", "__pycache__", ".git"})


def _snapshot_sample_workspace(ws: Path, dest: Path) -> int:
    """Freeze one sample's whole working dir at `dest`. Returns files captured.

    COPIES every file. Hard-linking was tried first and is WRONG here, caught
    by this module's own test suite on its first run: a hard link shares the
    inode, and the harness rewrites outputs with `Path.write_bytes`, which
    truncates the EXISTING inode rather than creating a new file — so a later
    re-run silently mutated the frozen snapshot through the link, defeating the
    whole purpose. Sharing an inode with a mutable file cannot produce an
    immutable snapshot. (`stage_split_dataset` still links, correctly: the
    dataset is never rewritten, and its mount is read-only.)

    On APFS/XFS `shutil.copy2` lands on a copy-on-write clone, so the cost is
    close to a link anyway while the inode stays independent.

    Symlinks are SKIPPED rather than followed: a link's target is outside the
    frozen tree and can change under us, which is exactly the staleness this
    function removes.
    """
    ws, dest = Path(ws), Path(dest)
    if not ws.is_dir():
        return 0
    n = 0
    for src in sorted(ws.rglob("*")):
        rel = src.relative_to(ws)
        if any(part in _SNAPSHOT_EXCLUDE_DIRS for part in rel.parts):
            continue
        if src.is_symlink() or not src.is_file():
            continue
        tgt = dest / rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, tgt)   # NOT os.link — see docstring
        n += 1
    return n


def snapshot_output(
    task_id: str,
    tasks_cache_root: Path,
    iteration: int,
    diff_text: str | None = None,
    *,
    phase: str = "",
    sample_idx: int | None = None,
) -> None:
    """Snapshot current output/ for cross-iteration comparison.

    Snapshots are written OUTSIDE the solver's task workspace (see module
    docstring) so future rollouts of the same task can't read their own
    prior outputs — only the evolver sees them.

    diff_text, when provided, is the scorer's own output-vs-golden diff
    computed at scoring time; it lands as DIFF.txt in the snapshot dir so
    the evolver reads exact mismatches for free instead of re-running the
    comparison. Captured in the cache artifact tars like everything else
    in this dir, so replays keep it.

    Mid-gate re-evals write ``iter_{N}_midgate`` (not ``iter_{N}``) so they
    do not overwrite the train/base DIFF for the same loop iteration —
    otherwise Stage-3 v0/v1 DIFF pins collide on one path (Jul 23 audit).

    sample_idx (2026-07-24, resampling infra): when a task is rolled out
    N>1 times under LoopConfig.n_samples for the SAME iteration, each
    sample needs its own snapshot dir or they'd collide on one `iter_N`
    path (same class of bug the midgate suffix above already fixes for
    train-vs-midgate). Pass the 0-based sample index to get `iter_N_sK`;
    leave None (the default, used by every existing n_samples=1 call site)
    to keep today's exact `iter_N` naming unchanged.
    """
    # Read the SAMPLE's own output dir (task_ws_name): with n_samples>1
    # each sample has its own workspace, so the shared path would either
    # be missing or hold a sibling's file.
    out_dir = Path(tasks_cache_root) / task_ws_name(task_id, sample_idx) / "output"
    label = f"{iteration}_midgate" if str(phase).startswith("midgate") else str(iteration)
    if sample_idx is not None:
        label = f"{label}_s{sample_idx}"
    snap_dir = Path(tasks_cache_root) / SNAPSHOTS_ROOT_NAME / task_id / f"iter_{label}"
    if snap_dir.exists():
        shutil.rmtree(snap_dir)
    snap_dir.mkdir(parents=True, exist_ok=True)
    if out_dir.is_dir():
        for f in out_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, snap_dir / f.name)
    if diff_text is not None:
        (snap_dir / "DIFF.txt").write_text(diff_text)
    # TRUE-STATIC ROLLOUT SNAPSHOT (0725). The flat output files + DIFF.txt above
    # stay exactly where they are — 8+ rescore/reconstruction scripts index
    # `_snapshots/{id}/iter_{n}/<name>.xlsx` and would break if outputs moved.
    # What is NEW is `workspace/`: the whole of what this sample's solver could
    # see and write, frozen. The evolver was previously pointed at the LIVE
    # `{task_id}__s{K}/` dir for its Stage 1 input read, which a later re-run
    # overwrites — so it could diagnose one rollout using another's bytes. This
    # gives it an immutable copy instead, and adds evidence the trace only
    # described: any script or intermediate file the solver wrote.
    _snapshot_sample_workspace(
        Path(tasks_cache_root) / task_ws_name(task_id, sample_idx),
        snap_dir / "workspace",
    )


def get_output_dir(
    task_id: str, tasks_cache_root: Path, sample_idx: int | None = None
) -> Path:
    """Return the output dir for a rollout (for the scorer).

    sample_idx must match what setup_task_workspace was given, or the scorer
    reads a different sample's output file (see task_ws_name).
    """
    return tasks_cache_root / task_ws_name(task_id, sample_idx) / "output"


def ensure_snapshot_iter_alias(
    task_id: str,
    tasks_cache_root: Path,
    target_iter: int,
) -> Path | None:
    """If ``iter_{target}`` is missing, copy the newest lower snapshot to it.

    Cache-hit train under loop iter K restores only the snapshot iters that
    were captured when the cache entry was written (typically ``iter_0`` from
    base). TraceDB then pins DIFF to ``iter_K``, which does not exist — the
    evolver's Stage-1 "Read first — DIFF" paths come back empty (Jul 23
    glm52). Same-content alias keeps the pin honest without re-scoring.
    """
    if target_iter < 0:
        return None
    snap_root = Path(tasks_cache_root) / SNAPSHOTS_ROOT_NAME / task_id
    target = snap_root / f"iter_{target_iter}"
    if _snapshot_dir_usable(target):
        return target
    if not snap_root.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for d in snap_root.iterdir():
        m = re.fullmatch(r"iter_(\d+)", d.name)
        if not m or not d.is_dir():
            continue
        n = int(m.group(1))
        if n == target_iter or not _snapshot_dir_usable(d):
            continue
        candidates.append((n, d))
    if not candidates:
        return None
    lower = [c for c in candidates if c[0] < target_iter]
    src = max(lower, key=lambda c: c[0])[1] if lower else min(candidates, key=lambda c: c[0])[1]
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(src, target)
    return target


def _snapshot_dir_usable(path: Path) -> bool:
    return path.is_dir() and (
        (path / "DIFF.txt").is_file() or any(path.glob("*.xlsx"))
    )


def resolve_snapshot_iter(
    task_id: str,
    tasks_cache_root: Path | str | None,
    claimed: int,
) -> int:
    """Map a claimed snapshot iter to one that actually has DIFF/xlsx on disk."""
    if tasks_cache_root is None:
        return claimed
    snap_root = Path(tasks_cache_root) / SNAPSHOTS_ROOT_NAME / task_id
    if not snap_root.is_dir():
        return claimed
    existing = sorted(
        int(m.group(1))
        for d in snap_root.iterdir()
        if (m := re.fullmatch(r"iter_(\d+)", d.name)) and _snapshot_dir_usable(d)
    )
    if not existing:
        return claimed
    le = [n for n in existing if n <= claimed]
    return max(le) if le else max(existing)


def backfill_snapshot_diffs(task_id: str, tasks_cache_root: Path) -> None:
    """Write DIFF.txt into snapshot dirs that predate diff persistence.

    Cache-replayed snapshots recorded before DIFF.txt persistence carry
    outputs but no DIFF.txt; computing it host-side at replay costs one
    comparison per missing file (snapshot xlsx were recalculated in place
    by the verifier before capture, so the detailed compare applies
    directly; goldens go through the answer recalc cache).

    Also aliases the current loop iter's snapshot dir when a cache hit
    restored only an older iter_* (see ``ensure_snapshot_iter_alias``).
    """
    from src.evaluation.spreadsheetbench_scorer import _get_task_dir, render_task_diff

    task_dir = _get_task_dir(task_id)
    if task_dir is None:
        return
    snap_root = Path(tasks_cache_root) / SNAPSHOTS_ROOT_NAME / task_id
    if not snap_root.exists():
        return
    for it_dir in sorted(snap_root.iterdir()):
        if not it_dir.is_dir() or (it_dir / "DIFF.txt").exists():
            continue
        if not any(it_dir.glob("*.xlsx")):
            continue
        try:
            soft, body = render_task_diff(task_dir, it_dir, max_mismatches=0)
            (it_dir / "DIFF.txt").write_text(
                f"{'PASS' if soft >= 0.999 else 'FAIL'} — host-side diff estimate "
                f"{soft:.2f} (backfilled at cache replay; full, uncapped; the verifier "
                f"reward recorded in the trace is authoritative)\n\n{body}\n"
            )
        except Exception as e:
            _log.warning(
                f"DIFF.txt backfill failed for {task_id} {it_dir.name}: {e}"
            )

    # Materialize iter_K when this replay is for loop iter K (eval_iter_label).
    try:
        from src.harness.utils import eval_iter_label

        label = eval_iter_label.get() or ""
        m = re.search(r"iter\s*(\d+)", label, re.I)
        if m:
            ensure_snapshot_iter_alias(task_id, tasks_cache_root, int(m.group(1)))
    except Exception as e:
        _log.warning(f"snapshot iter alias failed for {task_id}: {e}")
