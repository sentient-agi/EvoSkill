"""The loop-rendered evolver prompt: src/orchestrator/evolver_template.py.

The AO paragraph rules, and wording that must stay exactly true: what the transcript shows of a
tool call, what the grader sentence claims about a local run, and the batch line both prompt
gates match. No network, no model call, no sandbox, no LibreOffice.
"""
from __future__ import annotations

import re
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.orchestrator.evolver_template as ET  # noqa: E402

FIXROOT = Path(tempfile.gettempdir()) / "evolver-template-fixtures"


@pytest.fixture
def fx():
    FIXROOT.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="et_", dir=str(FIXROOT))).resolve()
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


@pytest.mark.parametrize("body,want", [
    ("one.\n\ntwo.", "more than one paragraph"),
    ("# heading\nprose", "markdown heading"),
    ("The grader is in {GRADER_DIR}.", "slot marker"),
    ("", "empty"),
    ("x" * (ET.MAX_PARAGRAPH_CHARS + 1), "the cap is"),
])
def test_template_paragraph_must_be_one_paragraph(fx, body, want):
    p = fx / ET.TEMPLATE_NAME
    p.write_text(body)
    with pytest.raises(ET.RenderError, match=want):
        ET.read_template_paragraph(p)


def test_template_paragraph_is_verbatim(fx):
    p = fx / ET.TEMPLATE_NAME
    p.write_text("\n  A verdict is produced by comparing values; `x` stays as written.  \n\n")
    assert ET.read_template_paragraph(p) == "A verdict is produced by comparing values; `x` stays as written."


def test_fence_outlasts_inner_fences():
    assert ET._fence("plain") == "```"
    assert ET._fence("a\n```py\nx\n```\n") == "````"


def _info(calls_clipped=0, results_clipped=0):
    return {"grader_output": {"sections": 3, "clipped": 0, "full": True},
            "changed_cells": {"on": False, "sections": 0, "lists_capped": 0},
            "tool_output": {"clip_head": 1200, "clip_tail": 600,
                            "results": {"items": 9, "clipped": results_clipped},
                            "calls": {"items": 9, "clipped": calls_clipped}},
            "transcript": {"system_prompt": "not shown",
                           "skill_loads": "shown like any other tool output"}}


def test_the_transcript_sentence_says_what_a_tool_call_shows():
    """The builder shows a call's command alone when it has one (timeout_sec dropped) and its JSON
    arguments otherwise; "each tool call's arguments" was true of neither half."""
    for info in (_info(), _info(calls_clipped=2, results_clipped=3)):
        s = ET._holds_text([("d", info)])
        assert ("the solver's messages, each tool call's command (or its JSON arguments when it has "
                "no command; a call that has a command is shown without its other arguments, such "
                "as `timeout_sec`) and each tool's output, without the solver's system prompt.") in s
        assert "each tool call's arguments" not in s


def test_the_grader_sentence_never_says_a_local_run_cannot_agree():
    """Un-recalculated, the grader can still agree with the recorded verdict (no formulas, say)."""
    s = ET.grader_sentence("/r/grader_clean", True, True)
    assert "so it is not guaranteed to reproduce a recorded verdict." in s
    assert "does not reproduce" not in s
    assert ET.GRADER_NOT_HERE not in ET.grader_sentence("/r/grader_clean", True, None)


def test_batch_line_is_the_shape_both_gates_match():
    line = ET.BATCH_LINE.format(T=59, S=3, E=177, P=57)
    assert line == ("This batch is **59 train tasks** × 3 samples = 177 episodes, of which "
                    "**57 of 177 episodes** passed.")
    plain = re.sub(r"\*|__", "", line)                    # the gates strip emphasis first
    assert re.search(r"(\d+)\s+train\s+tasks\b", plain).group(1) == "59"
    assert re.search(r"(\d+)\s+of\s+(\d+)\s+episodes?\b", line).groups() == ("57", "177")

