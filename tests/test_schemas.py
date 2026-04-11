"""Tests for src/schemas/ — Pydantic model validation, defaults, and required fields."""

import pytest
from pydantic import ValidationError


# ===========================================================================
# AgentResponse
# ===========================================================================

class TestAgentResponse:
    def test_valid_construction(self):
        from src.schemas import AgentResponse

        resp = AgentResponse(final_answer="42", reasoning="Because math.")
        assert resp.final_answer == "42"
        assert resp.reasoning == "Because math."

    def test_missing_final_answer_raises(self):
        from src.schemas import AgentResponse

        with pytest.raises(ValidationError):
            AgentResponse(reasoning="no answer provided")  # type: ignore[call-arg]

    def test_missing_reasoning_raises(self):
        from src.schemas import AgentResponse

        with pytest.raises(ValidationError):
            AgentResponse(final_answer="42")  # type: ignore[call-arg]

    def test_empty_strings_accepted(self):
        from src.schemas import AgentResponse

        resp = AgentResponse(final_answer="", reasoning="")
        assert resp.final_answer == ""

    def test_long_strings_accepted(self):
        from src.schemas import AgentResponse

        long_str = "x" * 10_000
        resp = AgentResponse(final_answer=long_str, reasoning=long_str)
        assert len(resp.final_answer) == 10_000

    def test_json_serialization_round_trip(self):
        from src.schemas import AgentResponse

        original = AgentResponse(final_answer="42", reasoning="Because math.")
        dumped = original.model_dump()
        restored = AgentResponse.model_validate(dumped)
        assert restored == original


# ===========================================================================
# ProposerResponse
# ===========================================================================

class TestProposerResponse:
    def test_valid_prompt_mode(self):
        from src.schemas import ProposerResponse

        resp = ProposerResponse(
            optimize_prompt_or_skill="prompt",
            proposed_skill_or_prompt="Improve CoT reasoning",
            justification="Agent failed on multi-step problems",
        )
        assert resp.optimize_prompt_or_skill == "prompt"

    def test_valid_skill_mode(self):
        from src.schemas import ProposerResponse

        resp = ProposerResponse(
            optimize_prompt_or_skill="skill",
            proposed_skill_or_prompt="Build a numeric extraction skill",
            justification="Agent missed decimal numbers",
        )
        assert resp.optimize_prompt_or_skill == "skill"

    def test_invalid_mode_raises(self):
        from src.schemas import ProposerResponse

        with pytest.raises(ValidationError):
            ProposerResponse(
                optimize_prompt_or_skill="invalid_value",  # type: ignore[arg-type]
                proposed_skill_or_prompt="something",
                justification="reason",
            )

    def test_missing_justification_raises(self):
        from src.schemas import ProposerResponse

        with pytest.raises(ValidationError):
            ProposerResponse(
                optimize_prompt_or_skill="skill",
                proposed_skill_or_prompt="something",
                # justification missing
            )  # type: ignore[call-arg]


# ===========================================================================
# SkillProposerResponse
# ===========================================================================

class TestSkillProposerResponse:
    def test_default_action_is_create(self):
        from src.schemas import SkillProposerResponse

        resp = SkillProposerResponse(
            proposed_skill="A numeric comparison skill",
            justification="Agent misses unit differences",
        )
        assert resp.action == "create"

    def test_edit_action_with_target_skill(self):
        from src.schemas import SkillProposerResponse

        resp = SkillProposerResponse(
            action="edit",
            target_skill="numeric-extraction",
            proposed_skill="Extend to handle percentages",
            justification="Missing percentage support",
        )
        assert resp.action == "edit"
        assert resp.target_skill == "numeric-extraction"

    def test_target_skill_defaults_to_none(self):
        from src.schemas import SkillProposerResponse

        resp = SkillProposerResponse(
            proposed_skill="New skill",
            justification="New capability needed",
        )
        assert resp.target_skill is None

    def test_related_iterations_defaults_to_empty_list(self):
        from src.schemas import SkillProposerResponse

        resp = SkillProposerResponse(
            proposed_skill="New skill",
            justification="reason",
        )
        assert resp.related_iterations == []

    def test_related_iterations_accepts_list(self):
        from src.schemas import SkillProposerResponse

        resp = SkillProposerResponse(
            proposed_skill="New skill",
            justification="reason",
            related_iterations=["iter-1", "iter-5"],
        )
        assert resp.related_iterations == ["iter-1", "iter-5"]

    def test_invalid_action_raises(self):
        from src.schemas import SkillProposerResponse

        with pytest.raises(ValidationError):
            SkillProposerResponse(
                action="delete",  # type: ignore[arg-type]
                proposed_skill="something",
                justification="reason",
            )

    def test_missing_proposed_skill_raises(self):
        from src.schemas import SkillProposerResponse

        with pytest.raises(ValidationError):
            SkillProposerResponse(justification="reason")  # type: ignore[call-arg]


# ===========================================================================
# PromptProposerResponse
# ===========================================================================

class TestPromptProposerResponse:
    def test_valid_construction(self):
        from src.schemas import PromptProposerResponse

        resp = PromptProposerResponse(
            proposed_prompt_change="Add step-by-step reasoning instructions",
            justification="Agent skips intermediate steps",
        )
        assert "step-by-step" in resp.proposed_prompt_change

    def test_missing_proposed_prompt_change_raises(self):
        from src.schemas import PromptProposerResponse

        with pytest.raises(ValidationError):
            PromptProposerResponse(justification="reason")  # type: ignore[call-arg]

    def test_missing_justification_raises(self):
        from src.schemas import PromptProposerResponse

        with pytest.raises(ValidationError):
            PromptProposerResponse(proposed_prompt_change="change")  # type: ignore[call-arg]


# ===========================================================================
# ToolGeneratorResponse
# ===========================================================================

class TestToolGeneratorResponse:
    def test_valid_construction(self):
        from src.schemas import ToolGeneratorResponse

        resp = ToolGeneratorResponse(
            generated_skill="# SKILL\n\nExtract numbers from text.",
            reasoning="The agent needs number extraction.",
        )
        assert "SKILL" in resp.generated_skill

    def test_missing_generated_skill_raises(self):
        from src.schemas import ToolGeneratorResponse

        with pytest.raises(ValidationError):
            ToolGeneratorResponse(reasoning="reason")  # type: ignore[call-arg]

    def test_missing_reasoning_raises(self):
        from src.schemas import ToolGeneratorResponse

        with pytest.raises(ValidationError):
            ToolGeneratorResponse(generated_skill="skill text")  # type: ignore[call-arg]


# ===========================================================================
# PromptGeneratorResponse
# ===========================================================================

class TestPromptGeneratorResponse:
    def test_valid_construction(self):
        from src.schemas import PromptGeneratorResponse

        resp = PromptGeneratorResponse(
            optimized_prompt="You are a helpful agent. Always show your reasoning.",
            reasoning="Added explicit reasoning instruction.",
        )
        assert "helpful" in resp.optimized_prompt

    def test_missing_optimized_prompt_raises(self):
        from src.schemas import PromptGeneratorResponse

        with pytest.raises(ValidationError):
            PromptGeneratorResponse(reasoning="reason")  # type: ignore[call-arg]

    def test_missing_reasoning_raises(self):
        from src.schemas import PromptGeneratorResponse

        with pytest.raises(ValidationError):
            PromptGeneratorResponse(optimized_prompt="prompt text")  # type: ignore[call-arg]

    def test_json_round_trip(self):
        from src.schemas import PromptGeneratorResponse

        original = PromptGeneratorResponse(
            optimized_prompt="prompt", reasoning="reasoning"
        )
        restored = PromptGeneratorResponse.model_validate(original.model_dump())
        assert restored == original
