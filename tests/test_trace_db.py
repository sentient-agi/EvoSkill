"""Unit tests for TraceDB — no API calls, fast."""

import tempfile
from pathlib import Path

import pytest

from src.loop.trace_db import TraceDB


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmp:
        yield TraceDB(Path(tmp) / "test.db")


def test_insert_and_retrieve(db):
    db.insert(
        iteration="iter-1",
        question="What is 2+2?",
        ground_truth="4",
        agent_answer="4",
        score=1.0,
        trace_summary="Turn 1: I think it's 4.",
        active_skills=["math"],
        num_turns=2,
        category="arithmetic",
        phase="train",
    )
    rows = db.query_by_question("What is 2+2?")
    assert len(rows) == 1
    assert rows[0]["score"] == 1.0
    assert rows[0]["agent_answer"] == "4"
    assert rows[0]["num_turns"] == 2


def test_trace_file_written(db):
    db.insert(
        iteration="iter-1", question="Q1", ground_truth="A", agent_answer="A",
        score=1.0, trace_summary="the trace", active_skills=[], num_turns=5,
        category="x", phase="train",
    )
    rows = db.query_by_question("Q1")
    trace_file = Path(rows[0]["trace_file"])
    assert trace_file.exists()
    content = trace_file.read_text()
    assert "the trace" in content
    assert "iter-1" in content
    assert "Score**: 1.0" in content


def test_query_by_question_no_match(db):
    db.insert(
        iteration="iter-1", question="Q1", ground_truth="A", agent_answer="A",
        score=1.0, trace_summary="trace", active_skills=[], num_turns=1,
        category="x", phase="train",
    )
    assert db.query_by_question("Q2") == []


def test_generate_index_empty(db):
    result = db.generate_index(failed_questions=["Q1"])
    assert result == ""


def test_generate_index_with_traces(db):
    for i, score in enumerate([0.0, 0.5], start=1):
        db.insert(
            iteration=f"iter-{i}",
            question="Count peaks",
            ground_truth="22",
            agent_answer=str(10 + i),
            score=score,
            trace_summary=f"Turn 1: counted {10+i} peaks",
            active_skills=["counting"] if i == 2 else [],
            num_turns=i * 10,
            category="hard",
            phase="train",
        )
    index = db.generate_index(failed_questions=["Count peaks"])
    assert "iter-1" in index
    assert "iter-2" in index
    assert "counting" in index
    assert ".cache/traces/" in index or "traces/" in index
    # Should have table header
    assert "Iteration" in index
    assert "Score" in index


def test_generate_index_includes_other_traces(db):
    db.insert(
        iteration="iter-1", question="Q1", ground_truth="A", agent_answer="B",
        score=0.0, trace_summary="trace1", active_skills=[], num_turns=5,
        category="x", phase="train",
    )
    db.insert(
        iteration="iter-1", question="Q2", ground_truth="C", agent_answer="D",
        score=1.0, trace_summary="trace2", active_skills=[], num_turns=3,
        category="y", phase="train",
    )
    # Query for Q1 — should also show Q2 as cross-question context
    index = db.generate_index(failed_questions=["Q1"])
    assert "Q1" in index
    assert "Q2" in index
