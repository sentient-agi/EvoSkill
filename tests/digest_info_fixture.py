"""Test helper for DIGEST_INFO.json.

write_truthful_info(d)   a header computed from a hand-written synthetic digest dir's own bytes, for
                         fixtures that write task files by hand. Labelled as a fixture.

Nothing here writes outside the `d` it is given.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.orchestrator import digest_info as DI  # noqa: E402


def write_truthful_info(d: Path) -> dict:
    """A DIGEST_INFO.json that states exactly what dir `d`'s hand-written task files show (every
    tool clip counted as a result clip). For synthetic fixtures only."""
    files, gclip = {}, 0
    for p in DI.task_files(d):
        raw = p.read_bytes()
        s = DI.scan_text(raw.decode("utf-8", errors="replace"))
        files[p.name] = {"sha256": DI.sha256_bytes(raw), "bytes": len(raw),
                         "grader_sections": s["grader_sections"], "grader_clipped": s["grader_clipped"],
                         "changed_sections": s["changed_sections"], "changed_capped": s["changed_capped"],
                         "tool_results_clipped": s["tool_clipped"], "tool_calls_clipped": 0}
        gclip += s["grader_clipped"]
    tot = lambda k: sum(f[k] for f in files.values())  # noqa: E731
    info = {
        "schema": DI.SCHEMA, "builder": "tests/digest_info_fixture.py (SYNTHETIC FIXTURE)",
        "task_files": len(files),
        "grader_output": {"full": gclip == 0, "clip_head": None if gclip == 0 else 2500,
                          "clip_tail": None if gclip == 0 else 800,
                          "sections": tot("grader_sections"), "clipped": gclip, "chars_not_shown": 0},
        "changed_cells": {"on": tot("changed_sections") > 0,
                          "max_cells": 60 if tot("changed_sections") else None,
                          "sections": tot("changed_sections"), "lists_capped": tot("changed_capped"),
                          "cells_not_listed": 0},
        "tool_output": {"clip_head": 1200, "clip_tail": 600, "marker": DI.CLIP_PREFIX,
                        "results": {"items": 0, "clipped": tot("tool_results_clipped"),
                                    "chars_not_shown": 0},
                        "calls": {"items": 0, "clipped": 0, "chars_not_shown": 0}},
        "transcript": {"system_prompt": "not shown", "skill_loads": "shown like any other tool output"},
        "clipped_items_total": (gclip + tot("changed_capped") + tot("tool_results_clipped")),
        "files": files,
    }
    (Path(d) / DI.INFO_NAME).write_text(json.dumps(info, indent=1, sort_keys=True) + "\n")
    assert DI.problems(d) == [], DI.problems(d)
    return info


__all__ = ["write_truthful_info"]
