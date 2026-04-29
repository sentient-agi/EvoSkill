"""Tests for src/loop/ — LoopConfig, helpers (build_proposer_query, feedback management)."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock


# ===========================================================================
# LoopConfig — dataclass defaults and field types
# ===========================================================================

class TestLoopConfig:
    def test_default_max_iterations(self):
        from src.loop.config import LoopConfig

        config = LoopConfig()
        assert config.max_iterations == 5

    def test_default_frontier_size(self):
        from src.loop.config import LoopConfig

        config = LoopConfig()
        assert config.frontier_size == 3

    def test_default_evolution_mode(self):
        from src.loop.config import LoopConfig

        config = LoopConfig()
        assert config.evolution_mode == "skill_only"

    def test_default_selection_strategy(self):
        from src.loop.config import LoopConfig

        config = LoopConfig()
        assert config.selection_strategy == "best"

    def test_default_tolerance_is_zero(self):
        from src.loop.config import LoopConfig

        config = LoopConfig()
        assert config.tolerance == 0.0

    def test_default_cache_enabled(self):
        from src.loop.config import LoopConfig

        config = LoopConfig()
        assert config.cache_enabled is True

    def test_default_cache_dir(self):
        from src.loop.config import LoopConfig

        config = LoopConfig()
        assert config.cache_dir == Path(".cache/runs")

    def test_default_reset_feedback(self):
        from src.loop.config import LoopConfig

        config = LoopConfig()
        assert config.reset_feedback is True

    def test_custom_values_accepted(self):
        from src.loop.config import LoopConfig

        config = LoopConfig(
            max_iterations=20,
            frontier_size=5,
            evolution_mode="prompt_only",
            selection_strategy="random",
            tolerance=0.05,
        )
        assert config.max_iterations == 20
        assert config.frontier_size == 5
        assert config.evolution_mode == "prompt_only"
        assert config.selection_strategy == "random"
        assert config.tolerance == pytest.approx(0.05)

    def test_cache_dir_is_path_instance(self):
        from src.loop.config import LoopConfig

        config = LoopConfig()
        assert isinstance(config.cache_dir, Path)

    def test_default_concurrency(self):
        from src.loop.config import LoopConfig

        assert LoopConfig().concurrency == 4

    def test_default_no_improvement_limit(self):
        from src.loop.config import LoopConfig

        assert LoopConfig().no_improvement_limit == 5

    def test_default_failure_sample_count(self):
        from src.loop.config import LoopConfig

        assert LoopConfig().failure_sample_count == 3

    def test_default_samples_per_category(self):
        from src.loop.config import LoopConfig

        assert LoopConfig().samples_per_category == 2

    def test_proposer_max_truncation_level_default(self):
        from src.loop.config import LoopConfig

        assert LoopConfig().proposer_max_truncation_level == 2

    def test_consecutive_proposer_failures_limit_default(self):
        from src.loop.config import LoopConfig

        assert LoopConfig().consecutive_proposer_failures_limit == 5


def test_multi_tolerance_scorer_empty_prediction_scores_zero() -> None:
    from src.loop.runner import _score_multi_tolerance

    assert _score_multi_tolerance("question", "", "42") == 0.0
    assert _score_multi_tolerance("question", None, "42") == 0.0


# ===========================================================================
# build_proposer_query
# ===========================================================================

def _make_trace(result_text="Agent result", parse_error=None):
    """Create a lightweight AgentTrace mock."""
    from src.harness.agent import AgentTrace

    trace = AgentTrace(
        duration_ms=1000,
        total_cost_usd=0.01,
        num_turns=2,
        usage={},
        result=result_text,
        is_error=False,
        messages=[],
        model="claude-opus-4-5",
        parse_error=parse_error,
    )
    return trace


class TestBuildProposerQuery:
    def test_returns_string(self, tmp_path):
        from src.loop.helpers import build_proposer_query

        traces = [(_make_trace(), "agent_answer", "ground_truth", "math")]
        result = build_proposer_query(
            traces, "No previous attempts.", project_root=tmp_path
        )
        assert isinstance(result, str)

    def test_includes_failure_section(self, tmp_path):
        from src.loop.helpers import build_proposer_query

        traces = [(_make_trace("Agent said X"), "X", "Y", "category_a")]
        result = build_proposer_query(
            traces, "No previous attempts.", project_root=tmp_path
        )
        assert "Failure" in result

    def test_includes_agent_answer_and_ground_truth(self, tmp_path):
        from src.loop.helpers import build_proposer_query

        traces = [(_make_trace(), "predicted_42", "true_100", "finance")]
        result = build_proposer_query(
            traces, "No previous attempts.", project_root=tmp_path
        )
        assert "predicted_42" in result
        assert "true_100" in result

    def test_includes_categories_summary(self, tmp_path):
        from src.loop.helpers import build_proposer_query

        traces = [
            (_make_trace(), "a1", "gt1", "math"),
            (_make_trace(), "a2", "gt2", "finance"),
        ]
        result = build_proposer_query(traces, "", project_root=tmp_path)
        assert "math" in result
        assert "finance" in result

    def test_includes_feedback_history(self, tmp_path):
        from src.loop.helpers import build_proposer_query

        traces = [(_make_trace(), "a", "gt", "cat")]
        result = build_proposer_query(
            traces,
            "iter-1: tried numeric extraction",
            project_root=tmp_path,
        )
        assert "iter-1" in result

    def test_skill_only_mode_default(self, tmp_path):
        from src.loop.helpers import build_proposer_query

        traces = [(_make_trace(), "a", "gt", "cat")]
        result = build_proposer_query(
            traces, "", evolution_mode="skill_only", project_root=tmp_path
        )
        assert isinstance(result, str)

    def test_prompt_only_mode(self, tmp_path):
        from src.loop.helpers import build_proposer_query

        traces = [(_make_trace(), "a", "gt", "cat")]
        result = build_proposer_query(
            traces, "", evolution_mode="prompt_only", project_root=tmp_path
        )
        assert isinstance(result, str)

    def test_truncation_level_1_limits_failures(self, tmp_path):
        from src.loop.helpers import build_proposer_query

        # Truncation level 1 limits to max_failures=3
        traces = [(_make_trace(f"result {i}"), f"a{i}", f"gt{i}", "cat") for i in range(6)]
        result = build_proposer_query(
            traces, "", truncation_level=1, project_root=tmp_path
        )
        # Should mention at most 3 failures
        assert "Failure 4" not in result

    def test_truncation_level_2_aggressive(self, tmp_path):
        from src.loop.helpers import build_proposer_query

        traces = [(_make_trace("x" * 10000), "a", "gt", "cat")]
        # Should not raise and should produce a shorter query than level 0
        result_full = build_proposer_query(
            traces, "", truncation_level=0, project_root=tmp_path
        )
        result_agg = build_proposer_query(
            traces, "", truncation_level=2, project_root=tmp_path
        )
        assert len(result_agg) <= len(result_full)

    def test_task_constraints_included(self, tmp_path):
        from src.loop.helpers import build_proposer_query

        traces = [(_make_trace(), "a", "gt", "cat")]
        result = build_proposer_query(
            traces,
            "",
            task_constraints="Only use Python tools.",
            project_root=tmp_path,
        )
        assert "Only use Python tools." in result

    def test_existing_skills_listed(self, tmp_path):
        from src.loop.helpers import build_proposer_query

        # Create a fake skill directory
        skills_dir = tmp_path / ".claude" / "skills" / "my-skill"
        skills_dir.mkdir(parents=True)
        (skills_dir / "SKILL.md").write_text("# My Skill")

        traces = [(_make_trace(), "a", "gt", "cat")]
        result = build_proposer_query(traces, "", project_root=tmp_path)
        assert "my-skill" in result

    def test_no_skills_shows_none(self, tmp_path):
        from src.loop.helpers import build_proposer_query

        traces = [(_make_trace(), "a", "gt", "cat")]
        result = build_proposer_query(traces, "", project_root=tmp_path)
        assert "None" in result


# ===========================================================================
# append_feedback
# ===========================================================================

class TestAppendFeedback:
    def test_creates_file_if_not_exists(self, tmp_path):
        from src.loop.helpers import append_feedback

        path = tmp_path / "feedback.md"
        append_feedback(path, "iter-1", "skill proposal", "justification")
        assert path.exists()

    def test_appends_entry_to_existing_file(self, tmp_path):
        from src.loop.helpers import append_feedback

        path = tmp_path / "feedback.md"
        path.write_text("# History\n")
        append_feedback(path, "iter-1", "proposal A", "reason A")
        append_feedback(path, "iter-2", "proposal B", "reason B")
        content = path.read_text()
        assert "iter-1" in content
        assert "iter-2" in content

    def test_entry_contains_proposal_and_justification(self, tmp_path):
        from src.loop.helpers import append_feedback

        path = tmp_path / "feedback.md"
        append_feedback(path, "iter-1", "my proposal", "my reason")
        content = path.read_text()
        assert "my proposal" in content
        assert "my reason" in content

    def test_outcome_section_when_provided(self, tmp_path):
        from src.loop.helpers import append_feedback

        path = tmp_path / "feedback.md"
        append_feedback(
            path,
            "iter-1",
            "proposal",
            "reason",
            outcome="improved",
            score=0.85,
            parent_score=0.72,
        )
        content = path.read_text()
        assert "IMPROVED" in content
        assert "0.8500" in content

    def test_outcome_no_improvement(self, tmp_path):
        from src.loop.helpers import append_feedback

        path = tmp_path / "feedback.md"
        append_feedback(
            path, "iter-1", "proposal", "reason", outcome="no_improvement"
        )
        assert "NO_IMPROVEMENT" in path.read_text()

    def test_active_skills_included(self, tmp_path):
        from src.loop.helpers import append_feedback

        path = tmp_path / "feedback.md"
        append_feedback(
            path,
            "iter-1",
            "proposal",
            "reason",
            active_skills=["skill-a", "skill-b"],
        )
        content = path.read_text()
        assert "skill-a" in content
        assert "skill-b" in content

    def test_failure_category_included(self, tmp_path):
        from src.loop.helpers import append_feedback

        path = tmp_path / "feedback.md"
        append_feedback(
            path, "iter-1", "proposal", "reason", failure_category="formatting"
        )
        assert "formatting" in path.read_text()

    def test_root_cause_included(self, tmp_path):
        from src.loop.helpers import append_feedback

        path = tmp_path / "feedback.md"
        append_feedback(
            path, "iter-1", "proposal", "reason", root_cause="Agent skips steps"
        )
        assert "Agent skips steps" in path.read_text()

    def test_delta_displayed_with_sign(self, tmp_path):
        from src.loop.helpers import append_feedback

        path = tmp_path / "feedback.md"
        append_feedback(
            path,
            "iter-1",
            "proposal",
            "reason",
            outcome="improved",
            score=0.90,
            parent_score=0.80,
        )
        content = path.read_text()
        assert "+0.1000" in content


# ===========================================================================
# read_feedback_history
# ===========================================================================

class TestReadFeedbackHistory:
    def test_returns_file_contents_when_exists(self, tmp_path):
        from src.loop.helpers import read_feedback_history

        path = tmp_path / "feedback.md"
        path.write_text("## iter-1\nsome content")
        result = read_feedback_history(path)
        assert "iter-1" in result

    def test_returns_default_message_when_file_missing(self, tmp_path):
        from src.loop.helpers import read_feedback_history

        path = tmp_path / "nonexistent.md"
        result = read_feedback_history(path)
        assert "No previous attempts" in result


# ===========================================================================
# build_skill_query_from_skill_proposer
# ===========================================================================

class TestBuildSkillQuery:
    def test_includes_proposed_skill(self):
        from src.loop.helpers import build_skill_query_from_skill_proposer
        from src.schemas import SkillProposerResponse
        from src.harness.agent import AgentTrace

        proposer_output = SkillProposerResponse(
            proposed_skill="Build a numeric extraction tool",
            justification="Agent lacks unit handling",
        )
        trace = AgentTrace(
            duration_ms=500,
            total_cost_usd=0.005,
            num_turns=1,
            usage={},
            result="",
            is_error=False,
            messages=[],
            output=proposer_output,
        )
        query = build_skill_query_from_skill_proposer(trace)
        assert "numeric extraction tool" in query
        assert "unit handling" in query


# ===========================================================================
# build_prompt_query_from_prompt_proposer
# ===========================================================================

class TestBuildPromptQuery:
    def test_includes_proposed_change_and_original(self):
        from src.loop.helpers import build_prompt_query_from_prompt_proposer
        from src.schemas import PromptProposerResponse
        from src.harness.agent import AgentTrace

        proposer_output = PromptProposerResponse(
            proposed_prompt_change="Add step-by-step instructions",
            justification="Agent skips reasoning",
        )
        trace = AgentTrace(
            duration_ms=500,
            total_cost_usd=0.005,
            num_turns=1,
            usage={},
            result="",
            is_error=False,
            messages=[],
            output=proposer_output,
        )
        query = build_prompt_query_from_prompt_proposer(trace, "Original prompt text")
        assert "Original prompt text" in query
        assert "step-by-step" in query
        assert "Agent skips reasoning" in query


# ===========================================================================
# ensure_skill_frontmatter
# ===========================================================================

class TestEnsureSkillFrontmatter:
    def test_adds_frontmatter_when_missing(self, tmp_path):
        from src.harness.opencode.skill_utils import ensure_skill_frontmatter

        skill_dir = tmp_path / "my-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("# Skill body content\n")

        result = ensure_skill_frontmatter(
            skill_file, description="A useful skill"
        )
        assert result is True
        content = skill_file.read_text()
        assert "---" in content
        assert "my-skill" in content

    def test_returns_false_when_file_missing(self, tmp_path):
        from src.harness.opencode.skill_utils import ensure_skill_frontmatter

        result = ensure_skill_frontmatter(
            tmp_path / "nonexistent" / "SKILL.md",
            description="desc",
        )
        assert result is False

    def test_preserves_existing_description(self, tmp_path):
        from src.harness.opencode.skill_utils import ensure_skill_frontmatter

        skill_dir = tmp_path / "skill-a"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("---\nname: skill-a\ndescription: Old description\n---\n\nBody")

        result = ensure_skill_frontmatter(skill_file, description="New description")
        # Description already exists → no rewrite
        assert result is False
        content = skill_file.read_text()
        assert "Old description" in content

    def test_description_truncated_if_too_long(self, tmp_path):
        from src.harness.opencode.skill_utils import ensure_skill_frontmatter

        skill_dir = tmp_path / "long-skill"
        skill_dir.mkdir()
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text("# Body")

        long_desc = "x " * 600  # > 1024 chars
        ensure_skill_frontmatter(skill_file, description=long_desc)
        content = skill_file.read_text()
        # Description in frontmatter should be truncated with ellipsis
        assert "..." in content
