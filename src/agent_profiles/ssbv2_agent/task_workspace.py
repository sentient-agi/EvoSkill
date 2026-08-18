"""Per-task workspace management for SpreadsheetBench v2.

Each task gets an isolated directory that is bind-mounted as the SOLVER's
container workspace — so it must contain ONLY what the solver is allowed
to see (no golden answer, no cross-iteration history):

    .cache/ssbv2_tasks/{task_id}/
        spreadsheets/       <- input xlsx copied from source
        output/             <- solver writes modified xlsx here (current iter)
        .claude/skills/     <- symlink to shared evolving skills dir

Evolver-only data lives OUTSIDE the solver's mount, under a sibling root
that only the evolver can reach (its container mounts the whole
`ssbv2_tasks/` root at /workspace/solver_outputs; the solver's mount is
scoped to its own task dir):

    .cache/ssbv2_tasks/_snapshots/{task_id}/iter_N/   <- per-iteration output snapshots

The golden answer is NOT staged anywhere on disk for either agent; the
evolver reads it directly from the read-only dataset mount
(/workspace/ssbv2_data/...), with the exact path given in its manifest.
A Jul 1 trace audit found solvers opening the previously staged
$WS/golden/ file during rollouts (28/407 traces; 32% pass rate vs 13%
for non-peekers) — ground-truth leakage, hence this layout.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path

_log = logging.getLogger(__name__)

# Generic per-task/per-root locking now lives in src.harness.task_workspace,
# shared across every benchmark (SSB v2, Polyglot, ...) instead of a
# benchmark-specific copy — see that module's docstring for why.

SNAPSHOTS_ROOT_NAME = "_snapshots"


# The static-snapshot and split-scoping mechanics are IDENTICAL across SSB v1
# and v2 — import them rather than fork, so a fix to one is a fix to both
# (0725: the v1 copy is the one with 28 tests behind it).
from src.agent_profiles.spreadsheetbench_agent.task_workspace import (  # noqa: E402
    _snapshot_sample_workspace,
    stage_split_dataset,
    task_ws_name as _v1_task_ws_name,
)


def task_ws_name(task_id: str, sample_idx: int | None = None) -> str:
    """Workspace dir name for a rollout — per SAMPLE when resampling.

    PARITY PORT (2026-07-31). SSB v1 fixed this on 2026-07-24; v2 was missed,
    and the gap was found the hard way by a 30-task x N=3 DSV4F run whose log
    carried ``File is not a zip file`` and ``LibreOffice recalculation failed``
    for the same task. Root cause: this module named workspaces by task_id
    alone, so with ``n_samples>1`` all N samples of a task bind-mounted the SAME
    directory and all wrote the SAME ``output/<name>.xlsx``. Consequences, all
    silent:

      * ``setup_task_workspace`` rmtree's ``output/`` on entry, so a sample
        starting up wiped a still-running sibling's output directory;
      * a sample could be SCORED against a sibling's file, or against one
        caught mid-write (hence the not-a-zip error);
      * a mid-write file makes the LibreOffice recalc produce nothing, and the
        scorer then reads formula cells as ``None`` — a GUARANTEED 0 on work
        that may have been correct. The bias is therefore downward.

    Sanitisation happens here (v2 task ids contain ``/``, e.g.
    ``Financial_Model/07_02``) and the ``__sK`` suffix is delegated to v1's
    namer so the two benchmarks cannot drift apart again.

    ``sample_idx=None`` reproduces the exact historical name, so every
    ``n_samples=1`` call site — which is every SSB v2 evolution and rollout
    script — is byte-for-byte unchanged.
    """
    return _v1_task_ws_name(task_id.replace("/", "_"), sample_idx)


def build_oracle_spec(ssbv2_root: Path, allowed_task_ids=None):
    """SSB v2's oracle declaration — see src/loop/oracle.py.

    Shared by the smoke and full run scripts so the two can't drift.
    """
    from src.loop.oracle import OracleSpec

    scorer_src = Path(__file__).parent / "scorer.py"
    data_host_prefix = str((Path(ssbv2_root) / "data" / "spreadsheetbench-v2").resolve())
    return OracleSpec(
        scoring_note=(
            "Only cell VALUES within each task's `answer_position` ranges are "
            "checked (rounded to 2 decimals, type-strict); fill color and font "
            "color are NOT part of scoring. A single wrong cell fails the "
            "ENTIRE task — no partial credit, so a task can be 99%+ "
            "cell-correct and still score 0 from one cascading formula error. "
            "Output files are formula-recalculated (LibreOffice) by the "
            "harness BEFORE comparison; the files you see in `output/` are "
            "already recalculated — diff them as-is. If you create xlsx files "
            "with formulas yourself, openpyxl will NOT compute their values "
            "(no recalc engine in this container)."
        ),
        scoring_code=[scorer_src],
        diff_recipe=(
            "EVERY snapshot dir already contains `DIFF.txt` — the scorer's own "
            "diff for that iteration's output, computed at scoring time, full "
            "and uncapped: `./solver_outputs/_snapshots/{task_id}/iter_N/DIFF.txt`. "
            "READ THOSE FIRST — they are free and authoritative; do NOT re-run "
            "comparisons just to see what failed.\n\n"
            "Only compute a diff yourself for what-if checks (e.g. after you "
            "modify a workbook). When you do, use the scorer's own function — "
            "do NOT hand-roll workbook comparisons (your rounding/type rules "
            "WILL differ) and do NOT diff cells outside answer_position (they "
            "are not graded). Complete example (note the types: each meta `m` "
            "is a dict loaded from its JSON file; `diff` is a plain STRING "
            "summary — print it, do not index into it):\n\n"
            "```python\n"
            "import sys, json, glob\n"
            "sys.path.insert(0, './oracle')\n"
            "from scorer import compare_workbooks_detailed\n"
            "for task_id in ['Template/04_04', 'Debugging/02_01']:  # use EXACT ids from the failure list\n"
            "    safe_id = task_id.replace('/', '_')\n"
            "    m = json.load(open(f'oracle/tasks/{safe_id}.json'))  # dict: task_id, answer_position, golden_path\n"
            "    outs = glob.glob(f'solver_outputs/{safe_id}/output/*.xlsx')\n"
            "    if not outs:\n"
            "        print(task_id, 'NO OUTPUT FILE (solver never saved one)'); continue\n"
            "    passed, n_match, n_total, diff = compare_workbooks_detailed(\n"
            "        m['golden_path'], outs[0], m['answer_position'])\n"
            "    print(task_id, passed, f'{n_match}/{n_total} graded cells match')\n"
            "    print(diff)  # STRING; 20 mismatches by default — pass max_mismatches=0 to list ALL\n"
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
        # file at all (0725 contamination control — parity with SSB v1).
        write_task_meta=lambda tasks_dir: write_all_task_meta(
            tasks_dir, data_host_prefix, allowed_task_ids=allowed_task_ids,
        ),
    )


def write_all_task_meta(
    meta_dir: Path,
    data_host_prefix: str,
    container_data_root: str = "/workspace/ssbv2_data",
    allowed_task_ids: set[str] | frozenset[str] | None = None,
) -> int:
    """Write per-task metadata JSON files for the evolver.

    One `{safe_id}.json` per task with the machine-readable inputs the
    scorer's diff needs: answer_position (the graded ranges) and golden_path
    (container path). Written into the benchmark's oracle/tasks/ dir —
    inside the evolver's workspace, so never solver-visible (solvers mount
    only their own task dir). golden_path is translated from the host
    dataset path to the evolver container's read-only mount.
    """
    from .dataset import TASK_INDEX

    meta_dir = Path(meta_dir)
    meta_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for task_id, info in TASK_INDEX.items():
        # 0725 contamination control (parity with v1): val/test tasks get no
        # meta file at all, so no pointer to their golden exists. See the v1
        # docstring — removing pointers is only half; the mount must be scoped
        # too (stage_split_dataset).
        if allowed_task_ids is not None and task_id not in allowed_task_ids:
            continue
        golden_ctr = (info.get("golden_path") or "").replace(
            data_host_prefix, container_data_root
        )
        safe_id = task_id.replace("/", "_")
        (meta_dir / f"{safe_id}.json").write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "answer_position": info.get("answer_position", ""),
                    "golden_path": golden_ctr,
                },
                indent=1,
            )
        )
        n += 1
    return n


def setup_task_workspace(
    task_id: str,
    spreadsheet_path: str,
    tasks_cache_root: Path,
    shared_skills_dir: Path,
    sample_idx: int | None = None,
) -> Path:
    """Create/reset a per-task (per-SAMPLE when resampling) workspace.

    Pass ``sample_idx`` whenever N>1 samples of a task can be in flight at
    once, or they share one directory and destroy each other's output — see
    ``task_ws_name``. Whatever is passed here must also be passed to
    ``get_output_dir``/``snapshot_output``, or the scorer reads a different
    sample's file than the solver wrote.
    """
    ws = tasks_cache_root / task_ws_name(task_id, sample_idx)

    # Reset output completely — cross-iteration snapshots live outside the
    # solver-visible workspace now, so nothing in output/ needs preserving.
    # (Also clears stale output/iter_N dirs left by the pre-Jul-1 layout.)
    out_dir = ws / "output"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # `output_raw/` holds the solver's file as it was written, copied by
    # _preserve_raw_output before score_and_diff recalculates output/ in place.
    # It MUST be reset here for the same reason output/ is: it lives inside the
    # solver-visible workspace, so a survivor from an earlier iteration would
    # let a later rollout read a previous attempt's output — and would be
    # snapshotted as though it belonged to this rollout. Not resetting it would
    # reintroduce, in a new file, exactly the cross-rollout contamination the
    # module docstring exists to prevent.
    raw_dir = ws / "output_raw"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)

    # Stage input spreadsheet
    ss_dir = ws / "spreadsheets"
    ss_dir.mkdir(parents=True, exist_ok=True)
    src = Path(spreadsheet_path)
    if src.exists():
        dst = ss_dir / src.name
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)

    # Remove any golden dir staged by the pre-Jul-1 layout (defense in depth;
    # the run scripts wipe the whole cache root anyway).
    legacy_golden = ws / "golden"
    if legacy_golden.exists():
        shutil.rmtree(legacy_golden)

    # Symlink skills dir
    skills_link = ws / ".claude" / "skills"
    skills_link.parent.mkdir(parents=True, exist_ok=True)
    if skills_link.is_symlink() or skills_link.exists():
        if skills_link.is_symlink():
            skills_link.unlink()
        else:
            shutil.rmtree(skills_link)
    skills_link.symlink_to(shared_skills_dir.resolve())

    return ws


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

    Mid-gate re-evals write ``iter_{N}_midgate`` so they do not overwrite
    the train/base DIFF for the same loop iteration.

    ``sample_idx`` (parity with v1): with ``n_samples>1`` each sample needs its
    own ``iter_N_sK`` snapshot or the N samples overwrite one another and the
    evolver sees a single arbitrary survivor instead of the spread. It must
    match what ``setup_task_workspace`` was given — the source ``output/`` dir
    is resolved through the same ``task_ws_name``.
    """
    safe_id = task_id.replace("/", "_")
    out_dir = tasks_cache_root / task_ws_name(task_id, sample_idx) / "output"
    label = f"{iteration}_midgate" if str(phase).startswith("midgate") else str(iteration)
    if sample_idx is not None:
        label = f"{label}_s{sample_idx}"
    snap_dir = tasks_cache_root / SNAPSHOTS_ROOT_NAME / safe_id / f"iter_{label}"
    if snap_dir.exists():
        shutil.rmtree(snap_dir)
    snap_dir.mkdir(parents=True, exist_ok=True)
    if out_dir.is_dir():
        for f in out_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, snap_dir / f.name)
    if diff_text is not None:
        (snap_dir / "DIFF.txt").write_text(diff_text)

    # LAYOUT.md — provenance, because the filenames alone actively mislead.
    #
    # The flat copies written just above come from output/ but keep the INPUT's
    # filename (the solver saves as `<original>_input.xlsx`), so a top-level
    # `..._input.xlsx` sitting beside `workspace/spreadsheets/..._input.xlsx` is
    # the OUTPUT. On 2026-08-01 that cost two epochs: epoch 2 and epoch 3 both
    # concluded "the rollouts changed nothing" from an input-vs-output diff that
    # was really output-vs-output, and shipped opposite prescriptions from it
    # ("never re-save" then "re-saving is harmless, freeze everything"). The
    # orchestrator repeated the error while auditing them.
    #
    # Renaming the flat copy would break the rescore scripts that glob it, so
    # name the provenance instead. Costs ~1KB per rollout.
    (snap_dir / "LAYOUT.md").write_text(
        "# What each file here is\n\n"
        "Filenames mislead: the solver writes its output under the INPUT's\n"
        "filename, so `*_input.xlsx` at this level is the OUTPUT.\n\n"
        "| path | what it actually is |\n"
        "|---|---|\n"
        "| `DIFF.txt` | the scorer's own diff, computed at scoring time, uncapped. Authoritative — read this before recomputing anything. |\n"
        "| `*.xlsx` (this level) | flat copy of `workspace/output/` (POST-recalculation), kept for the rescore scripts. NOT the input. |\n"
        "| `workspace/spreadsheets/*.xlsx` | the input the solver was GIVEN |\n"
        "| `workspace/output_raw/*.xlsx` | what the solver ACTUALLY WROTE, copied before scoring transformed it |\n"
        "| `workspace/output/*.xlsx` | the same file AFTER the scorer's in-place recalculation — the bytes that were graded |\n\n"
        "`score_and_diff` rewrites the candidate in place before comparing:\n"
        "`_add_xlfn_prefixes` -> `_force_iterative_calc` ->\n"
        "`_recalculate_with_libreoffice` -> `_solve_datatables_with_uno` ->\n"
        "`_restore_unrecalculable_cells`. So `output/` is not what the solver\n"
        "wrote, and its checksum says nothing about whether the solver re-saved.\n\n"
        "Two different questions, two different comparisons:\n\n"
        "- *Did the solver's own edit break this?* -> `spreadsheets/` vs `output_raw/`\n"
        "- *What did the grader see?* -> golden vs `output/`\n\n"
        "`output_raw/` exists only for rollouts run after 2026-08-02; earlier\n"
        "snapshots have no pre-recalculation copy, and for those the first\n"
        "question is simply not answerable.\n"
    )
    # TRUE-STATIC ROLLOUT SNAPSHOT (0725, parity with v1). Flat outputs +
    # DIFF.txt above stay put for the rescore scripts; `workspace/` is the new
    # frozen copy of everything this rollout could see or write, so the evolver
    # never diagnoses one rollout using a later re-run's bytes.
    _snapshot_sample_workspace(
        tasks_cache_root / task_ws_name(task_id, sample_idx), snap_dir / "workspace"
    )


def get_output_dir(
    task_id: str, tasks_cache_root: Path, sample_idx: int | None = None
) -> Path:
    """Output dir for a rollout.

    ``sample_idx`` MUST match what ``setup_task_workspace`` was given, or the
    scorer reads a different sample's output file (see ``task_ws_name``).
    """
    return tasks_cache_root / task_ws_name(task_id, sample_idx) / "output"


def ensure_snapshot_iter_alias(
    task_id: str,
    tasks_cache_root: Path,
    target_iter: int,
) -> Path | None:
    """If ``iter_{target}`` is missing, copy the newest lower snapshot to it.

    See SSB v1 ``ensure_snapshot_iter_alias`` — same cache-replay / TraceDB
    DIFF-pin mismatch (Jul 23).
    """
    if target_iter < 0:
        return None
    safe_id = task_id.replace("/", "_")
    snap_root = Path(tasks_cache_root) / SNAPSHOTS_ROOT_NAME / safe_id
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
    safe_id = task_id.replace("/", "_")
    snap_root = Path(tasks_cache_root) / SNAPSHOTS_ROOT_NAME / safe_id
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

    Cache-replayed snapshots recorded before Jul 1 carry outputs but no
    DIFF.txt; computing it host-side at replay costs one comparison per
    missing file (the snapshot xlsx was recalculated before capture, so
    compare_workbooks_detailed applies directly — no LibreOffice needed).

    Also aliases the current loop iter's snapshot dir when a cache hit
    restored only an older iter_* (see ``ensure_snapshot_iter_alias``).
    """
    from .dataset import TASK_INDEX
    from .scorer import compare_workbooks_detailed

    info = TASK_INDEX.get(task_id)
    if not info:
        return
    safe_id = task_id.replace("/", "_")
    snap_root = Path(tasks_cache_root) / SNAPSHOTS_ROOT_NAME / safe_id
    if not snap_root.exists():
        return
    for it_dir in sorted(snap_root.iterdir()):
        if not it_dir.is_dir() or (it_dir / "DIFF.txt").exists():
            continue
        xlsx = sorted(it_dir.glob("*.xlsx"))
        if not xlsx:
            continue
        try:
            passed, n_match, n_total, diff = compare_workbooks_detailed(
                info["golden_path"], str(xlsx[0]), info["answer_position"], 0
            )
            (it_dir / "DIFF.txt").write_text(
                f"{'PASS' if passed else 'FAIL'} — {n_match}/{n_total} graded cells match\n"
                f"(scorer's own diff, computed at scoring time; full, uncapped)\n\n{diff}\n"
            )
        except Exception as e:
            _log.warning(
                f"DIFF.txt backfill failed for {task_id} {it_dir.name}: {e}"
            )

    try:
        from src.harness.utils import eval_iter_label

        label = eval_iter_label.get() or ""
        m = re.search(r"iter\s*(\d+)", label, re.I)
        if m:
            ensure_snapshot_iter_alias(task_id, tasks_cache_root, int(m.group(1)))
    except Exception as e:
        _log.warning(f"snapshot iter alias failed for {task_id}: {e}")
