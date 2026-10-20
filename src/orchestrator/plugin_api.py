"""The loop's side of the benchmark plug-in interface, version 3.

The plug-in the BENCHMARK ADAPTER writes is the single source of the benchmark's tasks AND their sets. The grader is
an EvoSkill primitive: a function that takes a state snapshot and returns a score. It is a file of its own, written
by the run's user and handed to the adapter, which neither writes, copies nor tests it. A run's setup/ holds three
files, all pinned at freeze:

    plugin_api.py          this module (the canonical copy, given)
    benchmark_plugin.py    the adapter's:  def list_tasks(root: Path) -> list[Task]
    benchmark_grader.py    the user's:     def grade(task: Task, snapshot: Path) -> float

list_tasks(root) describes the benchmark found under `root`: every task with its set, each Task carrying the fields
the grader reads. The loop runs it read-only and offline in a sandbox (plugin_runner.run_plugin, `python -I -S`: the
standard library and this module only), freezes what it returns as setup/tasks.json + setup/split.json, and runs
every leg from that list. The module is imported there, so its top level imports only the standard library and
plugin_api. Where the solver saves its output is the harness's business, not the plug-in's.

grade(task, snapshot) scores ONE rollout. `task` is the Task as frozen in setup/tasks.json. `snapshot` is a
directory: a copy, made for this call only, of the solver's working directory as it was when the solver exited;
grade() may change it. It returns a number from 0 to 1. A rollout that left nothing gradable is a score (normally
0), not a failure. When the grader itself cannot produce a score, grade() RAISES (any exception); the loop grades
that rollout once more and, if it raises again, records GRADER_FAILED for it -- never a number. The loop calls
grade() for every rollout, each call in its own process with a time limit, in the run's grader's setup (the
interpreter, programs and /data paths the user instruction describes), and never knows the benchmark. The grader
file is imported only there, so its top level may import what that setup provides; elsewhere the loop only reads
its source (plugin_runner.static_check_grader) and pins its bytes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

API_VERSION = 3
SPLITS = ("train", "validation", "test")
GRADER_FAILED = "grader_failed"      # what the loop records for a rollout whose grade() raised twice


@dataclass(frozen=True)
class Task:
    id: str                          # the benchmark's own task id
    category: str                    # a label the reports group by
    instruction: str                 # the task text, verbatim
    inputs: tuple[Path, ...]         # absolute paths of the files the solver gets; never an answer file
    answer_files: tuple[Path, ...]   # absolute paths of the reference files; the evolver sees the training ones only
    grading: Mapping[str, str]       # the grader's per-task arguments other than files
    split: str | None                # "train" | "validation" | "test" | None (a task the run does not use)
