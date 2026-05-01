"""Tests for run_evolution.py — HALOAgent + SelfImprovingLoop wiring."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from examples.appworld2.scripts.halo_agent import SEPARATOR
from examples.appworld2.scripts.run_evolution import (
    halo_scorer,
    make_halo_scorer,
    build_evolution_loop,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_gt_answer(tmp_path: Path, task_id: str, answer: str):
    """Write a fake ground truth answer.json."""
    gt_dir = tmp_path / "data" / "tasks" / task_id / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)
    (gt_dir / "answer.json").write_text(json.dumps(answer))


# ---------------------------------------------------------------------------
# TestHALOScorer
# ---------------------------------------------------------------------------

class TestHALOScorer:

    def test_returns_1_for_correct_answer(self, tmp_path: Path):
        _write_gt_answer(tmp_path, "abc_1", "42")
        score = halo_scorer(
            question=f"abc_1{SEPARATOR}Some task",
            predicted="42",
            ground_truth="abc_1",
            halo_root=tmp_path,
            experiment_name="exp",
        )
        assert score == 1.0

    def test_returns_0_for_wrong_answer(self, tmp_path: Path):
        _write_gt_answer(tmp_path, "abc_1", "42")
        score = halo_scorer(
            question=f"abc_1{SEPARATOR}Some task",
            predicted="wrong",
            ground_truth="abc_1",
            halo_root=tmp_path,
            experiment_name="exp",
        )
        assert score == 0.0

    def test_comma_list_match(self, tmp_path: Path):
        _write_gt_answer(tmp_path, "abc_1", "Song A,Song B,Song C")
        score = halo_scorer(
            question=f"abc_1{SEPARATOR}Some task",
            predicted="song b, song a, song c",
            ground_truth="abc_1",
            halo_root=tmp_path,
            experiment_name="exp",
        )
        assert score == 1.0

    def test_returns_0_for_missing_gt(self, tmp_path: Path):
        score = halo_scorer(
            question=f"abc_1{SEPARATOR}Some task",
            predicted="x",
            ground_truth="abc_1",
            halo_root=tmp_path,
            experiment_name="exp",
        )
        assert score == 0.0

    def test_extracts_task_id_from_ground_truth(self, tmp_path: Path):
        _write_gt_answer(tmp_path, "xyz_99", "correct")
        score = halo_scorer(
            question=f"xyz_99{SEPARATOR}Task",
            predicted="correct",
            ground_truth="xyz_99",
            halo_root=tmp_path,
            experiment_name="exp",
        )
        assert score == 1.0


class TestMakeHALOScorer:

    def test_returns_callable(self, tmp_path: Path):
        scorer = make_halo_scorer(tmp_path, "exp")
        assert callable(scorer)

    def test_scorer_has_correct_signature(self, tmp_path: Path):
        _write_gt_answer(tmp_path, "abc_1", "answer")
        scorer = make_halo_scorer(tmp_path, "exp")
        result = scorer(f"abc_1{SEPARATOR}Task", "answer", "abc_1")
        assert isinstance(result, float)
        assert result == 1.0


# ---------------------------------------------------------------------------
# TestBuildEvolutionLoop
# ---------------------------------------------------------------------------

class TestBuildEvolutionLoop:

    @patch("examples.appworld2.scripts.run_evolution.ProgramManager")
    @patch("examples.appworld2.scripts.run_evolution.Agent")
    def test_returns_loop_instance(self, MockAgent, MockPM, tmp_path: Path):
        MockPM.return_value = MagicMock()
        MockAgent.return_value = MagicMock()

        # Create minimal prompt file
        prompt_dir = tmp_path / "experiments" / "prompts" / "function_calling_agent"
        prompt_dir.mkdir(parents=True)
        (prompt_dir / "instructions.txt").write_text("You are an agent.")

        loop = build_evolution_loop(
            halo_root=tmp_path,
            train_pools={"easy": [("t1:::Q1", "t1")]},
            val_data=[("t1:::Q1", "t1", "easy")],
        )

        from src.loop import SelfImprovingLoop
        assert isinstance(loop, SelfImprovingLoop)

    @patch("examples.appworld2.scripts.run_evolution.ProgramManager")
    @patch("examples.appworld2.scripts.run_evolution.Agent")
    def test_loop_uses_prompt_only_mode(self, MockAgent, MockPM, tmp_path: Path):
        MockPM.return_value = MagicMock()
        MockAgent.return_value = MagicMock()

        prompt_dir = tmp_path / "experiments" / "prompts" / "function_calling_agent"
        prompt_dir.mkdir(parents=True)
        (prompt_dir / "instructions.txt").write_text("You are an agent.")

        loop = build_evolution_loop(
            halo_root=tmp_path,
            train_pools={"easy": [("t1:::Q1", "t1")]},
            val_data=[("t1:::Q1", "t1", "easy")],
        )
        assert loop.config.evolution_mode == "prompt_only"

    @patch("examples.appworld2.scripts.run_evolution.ProgramManager")
    @patch("examples.appworld2.scripts.run_evolution.Agent")
    def test_loop_uses_concurrency_1(self, MockAgent, MockPM, tmp_path: Path):
        MockPM.return_value = MagicMock()
        MockAgent.return_value = MagicMock()

        prompt_dir = tmp_path / "experiments" / "prompts" / "function_calling_agent"
        prompt_dir.mkdir(parents=True)
        (prompt_dir / "instructions.txt").write_text("You are an agent.")

        loop = build_evolution_loop(
            halo_root=tmp_path,
            train_pools={"easy": [("t1:::Q1", "t1")]},
            val_data=[("t1:::Q1", "t1", "easy")],
        )
        assert loop.config.concurrency == 1

    @patch("examples.appworld2.scripts.run_evolution.ProgramManager")
    @patch("examples.appworld2.scripts.run_evolution.Agent")
    def test_loop_prompt_path_points_to_instructions_txt(self, MockAgent, MockPM, tmp_path: Path):
        MockPM.return_value = MagicMock()
        MockAgent.return_value = MagicMock()

        prompt_dir = tmp_path / "experiments" / "prompts" / "function_calling_agent"
        prompt_dir.mkdir(parents=True)
        (prompt_dir / "instructions.txt").write_text("You are an agent.")

        loop = build_evolution_loop(
            halo_root=tmp_path,
            train_pools={"easy": [("t1:::Q1", "t1")]},
            val_data=[("t1:::Q1", "t1", "easy")],
        )
        assert "instructions.txt" in str(loop._prompt_path)
