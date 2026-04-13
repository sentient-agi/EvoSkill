"""Shared pytest fixtures for the EvoSkill test suite."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# AgentTrace fixture helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_trace_fields():
    """Return the minimum dict of fields required to construct an AgentTrace."""
    return {
        "duration_ms": 1500,
        "total_cost_usd": 0.01,
        "num_turns": 3,
        "usage": {"input_tokens": 100, "output_tokens": 50},
        "result": "some result string",
        "is_error": False,
        "messages": [],
    }


@pytest.fixture
def mock_agent_trace(minimal_trace_fields):
    """Return a fully constructed AgentTrace with no structured output."""
    from src.harness.agent import AgentTrace

    return AgentTrace(**minimal_trace_fields)


@pytest.fixture
def mock_agent_trace_with_output(minimal_trace_fields):
    """Return an AgentTrace whose output is an AgentResponse."""
    from src.harness.agent import AgentTrace
    from src.schemas import AgentResponse

    fields = {
        **minimal_trace_fields,
        "model": "claude-opus-4-5",
        "output": AgentResponse(final_answer="42", reasoning="Because."),
    }
    return AgentTrace(**fields)


# ---------------------------------------------------------------------------
# ProgramConfig fixture helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def base_program_config():
    """Return a minimal ProgramConfig ready for use in tests."""
    from src.registry.models import ProgramConfig

    return ProgramConfig(
        name="base",
        system_prompt={"type": "preset", "preset": "claude_code"},
        allowed_tools=["Read", "Bash"],
    )


# ---------------------------------------------------------------------------
# Temp-directory fixture (avoids polluting cwd during cache tests)
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Return a temporary cache directory path."""
    return tmp_path / ".cache" / "runs"
