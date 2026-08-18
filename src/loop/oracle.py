"""First-class "oracle" artifact class.

The oracle is the umbrella for every artifact an evolver needs to answer
one question: "what IS the correct answer here, and how is correctness
computed?" Before this existed, that evidence was scattered per benchmark
(GT strings in trace files, per-task meta JSONs, a scorer source mounted
at an ad-hoc path, scoring semantics prose hand-written into each
manifest) — and every path drift between them silently starved the
evolver (Jul 1: evolvers burned turns on `find / -name scorer.py`; a
manifest pointed at a repo path that didn't exist in the container).

A benchmark declares an OracleSpec; materialize_oracle() renders it as a
single evolver-visible directory:

    oracle/
        README.md          <- generated: scoring semantics + artifact/path
                              conventions + diff recipe + inventory
        <scorer sources>   <- the EXACT scoring code the harness runs (copies)
        tasks/<id>.json    <- per-task metadata (graded ranges, golden paths)
        gt/<qhash>.txt     <- per-question ground-truth records (TraceDB writes
                              these as traces accumulate when configured with
                              gt_dir=<oracle>/gt; trace files + index point here)

The directory lives inside the evolver's workspace (solver containers
mount only their own task dir, so it is never solver-visible). The
universal evolver prompt references `./oracle/` generically — contents
are benchmark-specific, the convention is not.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class OracleSpec:
    """Benchmark declaration of its ground truth + scoring mechanism."""

    # Short prose: what is compared, tolerances, partial credit or not.
    scoring_note: str
    # Host paths of scoring source files, copied (snapshot semantics) into
    # the oracle dir under their basenames. Each entry is either a bare
    # Path (rendered with the default "EXACT scoring code the harness
    # runs" label) or a (Path, label) tuple for files that are NOT
    # literally what the harness executes — e.g. a host-side re-port used
    # for diffs/what-if checks — where the default label would overclaim
    # authority the file doesn't have (2026-07-23 oracle audit: all three
    # SSB v1 scoring-file bullets carried the identical "EXACT" phrasing
    # even though one of them is a host-side port, not the Docker
    # verifier's own code).
    scoring_code: list[Path | tuple[Path, str]] = field(default_factory=list)
    # Ready-to-run snippet showing how to check correctness for one task.
    diff_recipe: str = ""
    # Benchmark-specific artifact/path conventions the evolver needs in order
    # to READ the evidence: filename requirements, per-runtime path dialects
    # that show up inside solver traces, known artifact-level failure modes.
    # Lives here rather than in the evolver prompt (2026-07-24): it is
    # benchmark-instance documentation, so it belongs beside the scoring code
    # it describes, and the universal prompt keeps only the pointer. Verified
    # before the move that a real GLM-5.2 episode reads README.md on turn 1.
    artifact_note: str = ""
    # Callable(tasks_dir) -> count; writes per-task metadata JSONs into
    # oracle/tasks/. None for benchmarks without per-task metadata.
    write_task_meta: Callable[[Path], int] | None = None


def materialize_oracle(oracle_dir: Path, spec: OracleSpec) -> Path:
    """Render an OracleSpec into a fresh oracle/ directory.

    gt/ contents are preserved across regenerations: gt files are
    content-addressed by question hash, and continue-mode runs keep their
    trace DB — whose trace files point at oracle/gt paths. Wiping gt/ here
    would dangle every pointer in a continued run until each question
    happened to be re-inserted.
    """
    oracle_dir = Path(oracle_dir)
    stash: Path | None = None
    if oracle_dir.exists():
        gt_dir = oracle_dir / "gt"
        if gt_dir.exists() and any(gt_dir.iterdir()):
            stash = oracle_dir.parent / ".oracle_gt_stash"
            if stash.exists():
                shutil.rmtree(stash)
            gt_dir.rename(stash)
        shutil.rmtree(oracle_dir)
    oracle_dir.mkdir(parents=True)
    if stash is not None:
        stash.rename(oracle_dir / "gt")

    code_entries: list[tuple[str, str]] = []  # (filename, label)
    for entry in spec.scoring_code:
        src, label = entry if isinstance(entry, tuple) else (entry, "the EXACT scoring code the harness runs")
        src = Path(src)
        shutil.copy2(src, oracle_dir / src.name)
        code_entries.append((src.name, label))

    n_tasks = 0
    if spec.write_task_meta is not None:
        tasks_dir = oracle_dir / "tasks"
        tasks_dir.mkdir()
        n_tasks = spec.write_task_meta(tasks_dir)

    # Populated lazily by TraceDB (gt_dir points here) as traces accumulate.
    (oracle_dir / "gt").mkdir(exist_ok=True)

    lines = [
        "# Oracle — ground truth & exact scoring mechanism",
        "",
        "Everything needed to determine the CORRECT answer for any task and",
        "how correctness is computed. Read this file before diagnosing failures.",
        "",
        "## Scoring",
        "",
        spec.scoring_note.strip(),
        "",
        "## Files",
        "",
    ]
    for name, label in code_entries:
        lines.append(f"- `{name}` — {label}")
    if n_tasks:
        # DESCRIBE THE DISK; DO NOT ASSERT A CONVENTION. `n_tasks` is the TASK count
        # write_task_meta returned, NOT a file count, and the layouts differ by
        # harness: SSB v2 writes one file per task (`tasks/<task_id>.json`, 30 files),
        # while tau2/tau3 write one file per DOMAIN keyed by task_id (telecom task_ids
        # are 150+ char composite strings that make terrible filenames — see
        # tau2_agent/oracle.py). Rendering "(N files)" from a task count told a tau3
        # evolver to expect 32 files where exactly 1 exists, in the very file the
        # prompt orders it to read FIRST.
        tfiles = sorted(q.name for q in (oracle_dir / "tasks").glob("*.json"))
        if len(tfiles) == 1:
            lines.append(
                f"- `tasks/{tfiles[0]}` — ONE JSON object keyed by task_id, covering "
                f"{n_tasks} task(s): the complete golden spec the evaluator grades against"
            )
        else:
            lines.append(
                f"- `tasks/<task_id>.json` — per-task metadata ({len(tfiles)} files, "
                f"{n_tasks} task(s)): machine-readable inputs for the scorer "
                "(graded ranges, golden paths)"
            )
    # Only advertise `gt/` when it HAS something. It is created empty and filled
    # lazily by TraceDB, and promising ground-truth files that are not there is the
    # same "prompt describes evidence that is not on disk" defect the evidence
    # preflight exists to catch.
    if any((oracle_dir / "gt").iterdir()):
        lines.append(
            "- `gt/<qhash>.txt` — optional per-question ground-truth text when a "
            "trace header or index line points here (artifact benchmarks usually "
            "rely on `tasks/*.json` + snapshot `DIFF.txt` instead)"
        )
    if spec.artifact_note.strip():
        lines += ["", "## Artifacts & paths", "", spec.artifact_note.strip()]
    if spec.diff_recipe.strip():
        lines += ["", "## How to check correctness", "", spec.diff_recipe.strip()]
    lines.append("")
    (oracle_dir / "README.md").write_text("\n".join(lines))
    return oracle_dir
