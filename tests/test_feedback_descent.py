"""Tests for src/feedback_descent.py — FeedbackDescent core algorithm.

All proposer/evaluator dependencies are mocked so no API calls are made.
"""

import pytest
from unittest.mock import MagicMock

from src.feedback_descent import (
    EvaluationResult,
    FeedbackDescent,
    FeedbackDescentResult,
    FeedbackEntry,
)


# ===========================================================================
# Data classes
# ===========================================================================

class TestEvaluationResult:
    def test_construction_basic(self):
        result = EvaluationResult(preference_for_candidate=True, rationale="better")
        assert result.preference_for_candidate is True
        assert result.rationale == "better"

    def test_scores_default_to_none(self):
        result = EvaluationResult(preference_for_candidate=False, rationale="worse")
        assert result.score_best is None
        assert result.score_candidate is None

    def test_optional_scores_set(self):
        result = EvaluationResult(
            preference_for_candidate=True,
            rationale="better",
            score_best=0.7,
            score_candidate=0.9,
        )
        assert result.score_best == pytest.approx(0.7)
        assert result.score_candidate == pytest.approx(0.9)


class TestFeedbackEntry:
    def test_construction(self):
        entry = FeedbackEntry(candidate="candidate text", rationale="it improved X")
        assert entry.candidate == "candidate text"
        assert entry.rationale == "it improved X"

    def test_generic_with_int(self):
        entry: FeedbackEntry[int] = FeedbackEntry(candidate=42, rationale="highest score")
        assert entry.candidate == 42


class TestFeedbackDescentResult:
    def test_construction(self):
        result = FeedbackDescentResult(
            best="best prompt",
            feedback_history=[],
            iterations=3,
            improved=True,
        )
        assert result.best == "best prompt"
        assert result.iterations == 3
        assert result.improved is True


# ===========================================================================
# Helper: mock proposer and evaluator builders
# ===========================================================================

def _make_proposer(initial="initial", proposals=None):
    """Build a mock Proposer that yields canned proposals."""
    proposer = MagicMock()
    proposer.generate_initial.return_value = initial
    if proposals is not None:
        proposer.propose.side_effect = proposals
    else:
        proposer.propose.return_value = "proposed"
    return proposer


def _make_evaluator(preferences=None):
    """Build a mock Evaluator that returns canned EvaluationResults.

    Args:
        preferences: list of bool — True means candidate wins, False means no win.
    """
    evaluator = MagicMock()
    if preferences is not None:
        evaluator.evaluate.side_effect = [
            EvaluationResult(preference_for_candidate=p, rationale=f"rationale_{i}")
            for i, p in enumerate(preferences)
        ]
    else:
        evaluator.evaluate.return_value = EvaluationResult(
            preference_for_candidate=False, rationale="no improvement"
        )
    return evaluator


# ===========================================================================
# FeedbackDescent.run
# ===========================================================================

class TestFeedbackDescentRun:
    # --- Basic mechanics ---

    def test_returns_feedback_descent_result(self):
        fd = FeedbackDescent(
            proposer=_make_proposer(),
            evaluator=_make_evaluator(),
            max_iterations=1,
        )
        result = fd.run("problem description")
        assert isinstance(result, FeedbackDescentResult)

    def test_initial_candidate_generated_once(self):
        proposer = _make_proposer()
        fd = FeedbackDescent(proposer=proposer, evaluator=_make_evaluator(), max_iterations=2)
        fd.run("problem")
        proposer.generate_initial.assert_called_once_with("problem")

    def test_propose_called_once_per_iteration(self):
        proposer = _make_proposer()
        fd = FeedbackDescent(proposer=proposer, evaluator=_make_evaluator(), max_iterations=3)
        fd.run("problem")
        assert proposer.propose.call_count == 3

    def test_evaluate_called_once_per_iteration(self):
        evaluator = _make_evaluator()
        fd = FeedbackDescent(proposer=_make_proposer(), evaluator=evaluator, max_iterations=3)
        fd.run("problem")
        assert evaluator.evaluate.call_count == 3

    # --- Best updated on win ---

    def test_best_updated_when_candidate_wins(self):
        proposer = _make_proposer(initial="v0", proposals=["v1", "v2"])
        evaluator = _make_evaluator(preferences=[True, False])
        fd = FeedbackDescent(
            proposer=proposer,
            evaluator=evaluator,
            max_iterations=2,
            no_improvement_limit=5,
        )
        result = fd.run("problem")
        assert result.best == "v1"

    def test_best_unchanged_when_no_win(self):
        fd = FeedbackDescent(
            proposer=_make_proposer(initial="original"),
            evaluator=_make_evaluator(preferences=[False, False]),
            max_iterations=2,
            no_improvement_limit=5,
        )
        result = fd.run("problem")
        assert result.best == "original"

    # --- Feedback history management ---

    def test_feedback_history_reset_after_improvement(self):
        """After a win, feedback_history is reset and the next propose sees []."""
        calls = []

        def capture_propose(current_best, history):
            calls.append(list(history))
            return f"proposal_{len(calls)}"

        proposer = _make_proposer()
        proposer.propose.side_effect = capture_propose

        # Win on first iteration, no win on second
        evaluator = _make_evaluator(preferences=[True, False])

        fd = FeedbackDescent(
            proposer=proposer,
            evaluator=evaluator,
            max_iterations=2,
            no_improvement_limit=5,
        )
        fd.run("problem")

        # After the win (iter 1), history was reset → propose in iter 2 sees []
        assert calls[1] == []

    def test_feedback_history_accumulates_on_no_win(self):
        """Without a win, each iteration appends to history."""
        calls = []

        def capture_propose(current_best, history):
            calls.append(len(history))
            return "candidate"

        proposer = _make_proposer()
        proposer.propose.side_effect = capture_propose

        evaluator = _make_evaluator(preferences=[False, False, False])

        fd = FeedbackDescent(
            proposer=proposer,
            evaluator=evaluator,
            max_iterations=3,
            no_improvement_limit=10,
        )
        fd.run("problem")

        # Iteration 1 sees 0 history, iteration 2 sees 1, iteration 3 sees 2
        assert calls == [0, 1, 2]

    # --- Early stopping ---

    def test_early_stop_after_no_improvement_limit(self):
        fd = FeedbackDescent(
            proposer=_make_proposer(),
            evaluator=_make_evaluator(preferences=[False] * 10),
            max_iterations=10,
            no_improvement_limit=3,
        )
        result = fd.run("problem")
        # Should stop at iteration 3, not 10
        assert result.iterations == 3

    def test_no_early_stop_when_improvements_keep_coming(self):
        # Candidate always wins → no_improvement_count never increments
        proposer = _make_proposer(initial="v0")
        proposer.propose.side_effect = [f"v{i}" for i in range(1, 11)]
        evaluator = _make_evaluator(preferences=[True] * 10)

        fd = FeedbackDescent(
            proposer=proposer,
            evaluator=evaluator,
            max_iterations=5,
            no_improvement_limit=2,
        )
        result = fd.run("problem")
        assert result.iterations == 5  # All 5 ran

    def test_improved_flag_true_when_all_win(self):
        proposer = _make_proposer()
        proposer.propose.side_effect = ["v1"]
        evaluator = _make_evaluator(preferences=[True])

        fd = FeedbackDescent(
            proposer=proposer, evaluator=evaluator, max_iterations=1
        )
        result = fd.run("problem")
        assert result.improved is True

    def test_improved_flag_false_when_no_win(self):
        fd = FeedbackDescent(
            proposer=_make_proposer(),
            evaluator=_make_evaluator(preferences=[False]),
            max_iterations=1,
        )
        result = fd.run("problem")
        assert result.improved is False

    # --- Iterations count ---

    def test_iterations_count_equals_actual_runs(self):
        fd = FeedbackDescent(
            proposer=_make_proposer(),
            evaluator=_make_evaluator(preferences=[False] * 5),
            max_iterations=5,
            no_improvement_limit=10,
        )
        result = fd.run("problem")
        assert result.iterations == 5

    def test_zero_max_iterations_returns_initial(self):
        proposer = _make_proposer(initial="initial_candidate")
        fd = FeedbackDescent(
            proposer=proposer,
            evaluator=_make_evaluator(),
            max_iterations=0,
        )
        result = fd.run("problem")
        assert result.best == "initial_candidate"
        assert result.iterations == 0

    # --- Feedback history in result ---

    def test_result_feedback_history_contains_entries(self):
        evaluator = _make_evaluator(preferences=[False, False])
        fd = FeedbackDescent(
            proposer=_make_proposer(),
            evaluator=evaluator,
            max_iterations=2,
            no_improvement_limit=5,
        )
        result = fd.run("problem")
        assert len(result.feedback_history) == 2

    def test_result_feedback_history_empty_after_final_win(self):
        """If the last action was a win, feedback_history should be reset (empty)."""
        proposer = _make_proposer()
        proposer.propose.side_effect = ["v1"]
        evaluator = _make_evaluator(preferences=[True])

        fd = FeedbackDescent(
            proposer=proposer,
            evaluator=evaluator,
            max_iterations=1,
        )
        result = fd.run("problem")
        assert result.feedback_history == []

    # --- Proposer receives correct best ---

    def test_propose_receives_updated_best_after_win(self):
        received_bests = []

        def track_propose(current_best, history):
            received_bests.append(current_best)
            return f"candidate_from_{current_best}"

        proposer = _make_proposer(initial="v0")
        proposer.propose.side_effect = track_propose

        # Win iter 1, no win iter 2
        evaluator = _make_evaluator(preferences=[True, False])

        fd = FeedbackDescent(
            proposer=proposer,
            evaluator=evaluator,
            max_iterations=2,
            no_improvement_limit=5,
        )
        fd.run("problem")

        # Iter 1 propose gets "v0", iter 2 propose gets the winner "candidate_from_v0"
        assert received_bests[0] == "v0"
        assert received_bests[1] == "candidate_from_v0"

    # --- Evaluator receives correct arguments ---

    def test_evaluator_receives_best_and_candidate(self):
        proposer = _make_proposer(initial="best_v0")
        proposer.propose.return_value = "candidate_v1"

        calls = []

        def track_evaluate(best, candidate):
            calls.append((best, candidate))
            return EvaluationResult(preference_for_candidate=False, rationale="nope")

        evaluator = MagicMock()
        evaluator.evaluate.side_effect = track_evaluate

        fd = FeedbackDescent(
            proposer=proposer,
            evaluator=evaluator,
            max_iterations=1,
        )
        fd.run("problem")

        assert calls[0] == ("best_v0", "candidate_v1")

    # --- Generic type parameter works ---

    def test_works_with_integer_type(self):
        proposer = _make_proposer(initial=0)
        proposer.propose.side_effect = [10, 20]
        evaluator = _make_evaluator(preferences=[True, False])

        fd: FeedbackDescent[int] = FeedbackDescent(
            proposer=proposer,
            evaluator=evaluator,
            max_iterations=2,
            no_improvement_limit=5,
        )
        result = fd.run("maximize integer")
        assert result.best == 10  # Won on iter 1, then no win on iter 2
