"""Tests for HALOAgent — wraps HALO's runner as EvoSkill Agent.

All HALO interactions are mocked. Tests verify:
    - Task ID parsing from encoded questions
    - Answer extraction from supervisor.jsonl
    - Trace summary from lm_calls.jsonl
    - Eval result reading from JSON
    - AgentTrace construction
    - HALOAgent.run() lifecycle
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from examples.appworld2.scripts.halo_agent import (
    SEPARATOR,
    HALOAgent,
    build_agent_trace,
    parse_task_id,
    read_answer_from_jsonl,
    read_eval_result,
    read_trace_summary,
)
from src.harness.agent import AgentTrace
from src.schemas import AgentResponse


# ---------------------------------------------------------------------------
# TestParseTaskId
# ---------------------------------------------------------------------------

class TestParseTaskId:

    def test_basic_split(self):
        task_id, instruction = parse_task_id(f"abc_1{SEPARATOR}How many songs?")
        assert task_id == "abc_1"
        assert instruction == "How many songs?"

    def test_separator_in_instruction(self):
        task_id, instruction = parse_task_id(f"abc_1{SEPARATOR}What is 2{SEPARATOR}3?")
        assert task_id == "abc_1"
        assert instruction == f"What is 2{SEPARATOR}3?"

    def test_raises_on_missing_separator(self):
        with pytest.raises(ValueError):
            parse_task_id("no_separator")

    def test_strips_whitespace(self):
        task_id, instruction = parse_task_id(f" abc_1 {SEPARATOR} Some task ")
        assert task_id == "abc_1"
        assert instruction == "Some task"


# ---------------------------------------------------------------------------
# TestReadAnswerFromJsonl
# ---------------------------------------------------------------------------

class TestReadAnswerFromJsonl:

    def test_extracts_answer(self, tmp_path: Path):
        jsonl = tmp_path / "supervisor.jsonl"
        data = ["UPDATE tasks SET record_hash=?, status=?, answer=? WHERE tasks.id = ?",
                ["hash", "success", '"The Answer"', 1], False]
        jsonl.write_text(json.dumps(data) + "\n")
        assert read_answer_from_jsonl(jsonl) == "The Answer"

    def test_returns_none_for_missing_file(self, tmp_path: Path):
        assert read_answer_from_jsonl(tmp_path / "nonexistent.jsonl") is None

    def test_returns_none_for_empty_file(self, tmp_path: Path):
        jsonl = tmp_path / "supervisor.jsonl"
        jsonl.write_text("")
        assert read_answer_from_jsonl(jsonl) is None

    def test_handles_unquoted_answer(self, tmp_path: Path):
        jsonl = tmp_path / "supervisor.jsonl"
        data = ["UPDATE tasks SET record_hash=?, status=?, answer=? WHERE tasks.id = ?",
                ["hash", "success", "42", 1], False]
        jsonl.write_text(json.dumps(data) + "\n")
        assert read_answer_from_jsonl(jsonl) == "42"

    def test_handles_null_answer(self, tmp_path: Path):
        """Side-effect tasks submit null/None answer."""
        jsonl = tmp_path / "supervisor.jsonl"
        data = ["UPDATE tasks SET record_hash=?, status=?, answer=? WHERE tasks.id = ?",
                ["hash", "success", None, 1], False]
        jsonl.write_text(json.dumps(data) + "\n")
        result = read_answer_from_jsonl(jsonl)
        assert result is None


# ---------------------------------------------------------------------------
# TestReadTraceSummary
# ---------------------------------------------------------------------------

class TestReadTraceSummary:

    def test_summarizes_tool_calls(self, tmp_path: Path):
        jsonl = tmp_path / "lm_calls.jsonl"
        trajectory = [
            {"type": "function_call", "name": "supervisor__show_profile", "arguments": "{}"},
            {"type": "function_call_output", "output": '{"first_name": "Glenn"}'},
            {"type": "function_call", "name": "spotify__login", "arguments": '{"email": "a@b.com"}'},
            {"type": "function_call_output", "output": '{"access_token": "abc"}'},
        ]
        jsonl.write_text(json.dumps(trajectory))
        summary = read_trace_summary(jsonl)
        assert "supervisor__show_profile" in summary
        assert "spotify__login" in summary

    def test_returns_empty_for_missing_file(self, tmp_path: Path):
        summary = read_trace_summary(tmp_path / "nonexistent.jsonl")
        assert summary == ""

    def test_counts_turns(self, tmp_path: Path):
        jsonl = tmp_path / "lm_calls.jsonl"
        trajectory = [
            {"type": "function_call", "name": "api_1", "arguments": "{}"},
            {"type": "function_call_output", "output": "{}"},
            {"type": "function_call", "name": "api_2", "arguments": "{}"},
            {"type": "function_call_output", "output": "{}"},
            {"type": "function_call", "name": "api_3", "arguments": "{}"},
            {"type": "function_call_output", "output": "{}"},
        ]
        jsonl.write_text(json.dumps(trajectory))
        summary = read_trace_summary(jsonl)
        assert "3" in summary  # 3 tool calls


# ---------------------------------------------------------------------------
# TestReadEvalResult
# ---------------------------------------------------------------------------

class TestReadEvalResult:

    def test_reads_passed(self, tmp_path: Path):
        eval_json = tmp_path / "eval.json"
        eval_json.write_text(json.dumps({
            "aggregate": {"task_goal_completion": 100.0, "scenario_goal_completion": 100.0}
        }))
        passed, score = read_eval_result(eval_json)
        assert passed is True
        assert score == 1.0

    def test_reads_failed(self, tmp_path: Path):
        eval_json = tmp_path / "eval.json"
        eval_json.write_text(json.dumps({
            "aggregate": {"task_goal_completion": 0.0, "scenario_goal_completion": 0.0}
        }))
        passed, score = read_eval_result(eval_json)
        assert passed is False
        assert score == 0.0

    def test_returns_false_for_missing_file(self, tmp_path: Path):
        passed, score = read_eval_result(tmp_path / "nonexistent.json")
        assert passed is False
        assert score == 0.0


# ---------------------------------------------------------------------------
# TestBuildAgentTrace
# ---------------------------------------------------------------------------

class TestBuildAgentTrace:

    def test_creates_trace_with_answer(self):
        trace = build_agent_trace(
            task_id="abc_1",
            answer="42",
            trace_summary="Called 3 APIs",
            passed=True,
            score=1.0,
            num_turns=3,
        )
        assert isinstance(trace, AgentTrace)
        assert trace.output is not None
        assert trace.output.final_answer == "42"
        assert trace.num_turns == 3

    def test_creates_trace_without_answer(self):
        trace = build_agent_trace(
            task_id="abc_1",
            answer=None,
            trace_summary="Agent failed",
            passed=False,
            score=0.0,
        )
        assert trace.output is not None
        assert trace.output.final_answer == "[NO ANSWER]"

    def test_trace_has_reasoning(self):
        trace = build_agent_trace(
            task_id="abc_1",
            answer="42",
            trace_summary="Step 1: login\nStep 2: query",
            passed=True,
            score=1.0,
        )
        assert "login" in trace.output.reasoning
        assert "query" in trace.output.reasoning


# ---------------------------------------------------------------------------
# TestHALOAgent
# ---------------------------------------------------------------------------

class TestHALOAgent:

    def test_init_stores_config(self):
        agent = HALOAgent(
            halo_root="/fake/halo",
            model="claude-sonnet-4-20250514-no-reasoning",
            experiment_name="test_exp",
        )
        assert agent._halo_root == Path("/fake/halo")
        assert agent._model == "claude-sonnet-4-20250514-no-reasoning"
        assert agent._experiment_name == "test_exp"

    def _setup_fake_outputs(self, tmp_path: Path, task_id: str, exp_name: str,
                            answer: str | None = '"test answer"', tgc: float = 100.0):
        """Create fake HALO output files on disk."""
        halo_root = tmp_path / "halo"
        exp_dir = halo_root / "experiments" / "outputs" / exp_name / "tasks" / task_id

        dbs_dir = exp_dir / "dbs"
        dbs_dir.mkdir(parents=True)
        if answer is not None:
            sup_data = ["UPDATE tasks SET record_hash=?, status=?, answer=? WHERE tasks.id = ?",
                         ["hash", "success", answer, 1], False]
            (dbs_dir / "supervisor.jsonl").write_text(json.dumps(sup_data) + "\n")
        else:
            (dbs_dir / "supervisor.jsonl").write_text("")

        logs_dir = exp_dir / "logs"
        logs_dir.mkdir(parents=True)
        (logs_dir / "lm_calls.jsonl").write_text(json.dumps([
            {"type": "function_call", "name": "supervisor__complete_task", "arguments": "{}"},
        ]))

        eval_dir = halo_root / "experiments" / "outputs" / exp_name / "evaluations"
        eval_dir.mkdir(parents=True)
        (eval_dir / f"on_only_{task_id}.json").write_text(json.dumps({
            "aggregate": {"task_goal_completion": tgc, "scenario_goal_completion": tgc}
        }))

        # Create ground truth for fallback scorer
        gt_dir = halo_root / "data" / "tasks" / task_id / "ground_truth"
        gt_dir.mkdir(parents=True)
        (gt_dir / "answer.json").write_text(json.dumps("test answer"))

        return halo_root

    def test_run_returns_agent_trace(self, tmp_path: Path):
        """Mock _run_halo_and_evaluate to test the read-back logic."""
        halo_root = self._setup_fake_outputs(tmp_path, "abc_1", "test_exp",
                                              answer='"my answer"', tgc=100.0)

        agent = HALOAgent(halo_root=halo_root, experiment_name="test_exp")
        # Mock the heavy function that calls HALO + evaluates
        agent._run_halo_and_evaluate = lambda task_id, config: None

        # Patch _sync_prompt_to_halo to no-op
        agent._sync_prompt_to_halo = lambda: None

        trace = asyncio.run(agent.run(f"abc_1{SEPARATOR}Some task"))

        assert isinstance(trace, AgentTrace)
        assert trace.output.final_answer == "my answer"

    def test_run_handles_failed_task(self, tmp_path: Path):
        halo_root = self._setup_fake_outputs(tmp_path, "abc_1", "test_exp",
                                              answer=None, tgc=0.0)

        agent = HALOAgent(halo_root=halo_root, experiment_name="test_exp")
        agent._run_halo_and_evaluate = lambda task_id, config: None
        agent._sync_prompt_to_halo = lambda: None

        trace = asyncio.run(agent.run(f"abc_1{SEPARATOR}Some task"))

        assert trace.output.final_answer == "[NO ANSWER]"

    def test_run_reads_eval_score(self, tmp_path: Path):
        halo_root = self._setup_fake_outputs(tmp_path, "abc_1", "test_exp",
                                              answer='"correct"', tgc=100.0)

        agent = HALOAgent(halo_root=halo_root, experiment_name="test_exp")
        agent._run_halo_and_evaluate = lambda task_id, config: None
        agent._sync_prompt_to_halo = lambda: None

        trace = asyncio.run(agent.run(f"abc_1{SEPARATOR}Some task"))

        assert trace.is_error is False
