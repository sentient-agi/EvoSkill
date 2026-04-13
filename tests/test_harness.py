"""Tests for src/harness/ — AgentTrace, Agent, sdk_config, options_utils."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.harness.sdk_config import (
    get_sdk,
    is_claude_sdk,
    is_openhands_sdk,
    is_opencode_sdk,
    set_sdk,
)
from src.harness.agent import AgentTrace, Agent


# ===========================================================================
# sdk_config — pure state-management functions
# ===========================================================================

class TestSdkConfig:
    """Test the global SDK toggle — always reset after each test to avoid bleed."""

    def setup_method(self):
        """Ensure we start each test in Claude SDK mode."""
        set_sdk("claude")

    def teardown_method(self):
        """Reset to Claude SDK after each test."""
        set_sdk("claude")

    def test_default_sdk_is_claude(self):
        assert get_sdk() == "claude"

    def test_is_claude_sdk_true_by_default(self):
        assert is_claude_sdk() is True

    def test_is_opencode_sdk_false_by_default(self):
        assert is_opencode_sdk() is False

    def test_is_openhands_sdk_false_by_default(self):
        assert is_openhands_sdk() is False

    def test_set_sdk_to_opencode(self):
        set_sdk("opencode")
        assert get_sdk() == "opencode"
        assert is_opencode_sdk() is True
        assert is_claude_sdk() is False
        assert is_openhands_sdk() is False

    def test_set_sdk_to_openhands(self):
        set_sdk("openhands")
        assert get_sdk() == "openhands"
        assert is_openhands_sdk() is True
        assert is_claude_sdk() is False
        assert is_opencode_sdk() is False

    def test_set_sdk_back_to_claude(self):
        set_sdk("opencode")
        set_sdk("claude")
        assert is_claude_sdk() is True

    def test_invalid_sdk_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid SDK"):
            set_sdk("unknown_sdk")  # type: ignore[arg-type]

    def test_invalid_sdk_does_not_change_state(self):
        try:
            set_sdk("bad_value")  # type: ignore[arg-type]
        except ValueError:
            pass
        assert get_sdk() == "claude"


# ===========================================================================
# AgentTrace — construction and methods
# ===========================================================================

class TestAgentTrace:
    """Test AgentTrace data model construction and the summarize() method."""

    def _make_trace(self, **overrides):
        defaults = {
            "duration_ms": 1000,
            "total_cost_usd": 0.01,
            "num_turns": 2,
            "usage": {"input_tokens": 100, "output_tokens": 50},
            "result": "The answer is 42.",
            "is_error": False,
            "messages": [],
        }
        defaults.update(overrides)
        return AgentTrace(**defaults)

    def test_minimal_construction(self):
        trace = self._make_trace()
        assert trace.is_error is False
        assert trace.output is None

    def test_default_uuid_is_empty_string(self):
        trace = self._make_trace()
        assert trace.uuid == ""

    def test_default_model_is_empty_string(self):
        trace = self._make_trace()
        assert trace.model == ""

    def test_default_tools_is_empty_list(self):
        trace = self._make_trace()
        assert trace.tools == []

    def test_parse_error_defaults_to_none(self):
        trace = self._make_trace()
        assert trace.parse_error is None

    def test_output_can_be_pydantic_model(self):
        from src.schemas import AgentResponse

        trace = self._make_trace(
            output=AgentResponse(final_answer="42", reasoning="math")
        )
        assert trace.output.final_answer == "42"

    # --- summarize() ---

    def test_summarize_contains_model_info(self):
        trace = self._make_trace(model="claude-opus-4-5")
        summary = trace.summarize()
        assert "claude-opus-4-5" in summary

    def test_summarize_contains_turns(self):
        trace = self._make_trace(num_turns=5)
        summary = trace.summarize()
        assert "5" in summary

    def test_summarize_contains_full_result_when_no_error(self):
        trace = self._make_trace(result="The answer is 42.")
        summary = trace.summarize()
        assert "The answer is 42." in summary

    def test_summarize_shows_parse_error(self):
        trace = self._make_trace(parse_error="JSON decode failed")
        summary = trace.summarize()
        assert "JSON decode failed" in summary

    def test_summarize_truncates_large_result_on_parse_error(self):
        big_result = "X" * 200_000
        trace = self._make_trace(result=big_result, parse_error="failed")
        summary = trace.summarize(head_chars=100, tail_chars=100)
        assert "truncated" in summary.lower() or "omitted" in summary.lower()

    def test_summarize_no_truncation_without_parse_error(self):
        # Even a longish result should not be truncated when there is no parse_error
        result = "A" * 500
        trace = self._make_trace(result=result)
        summary = trace.summarize(head_chars=100, tail_chars=100)
        assert "truncated" not in summary.lower()

    def test_summarize_shows_output_when_present(self):
        from src.schemas import AgentResponse

        trace = self._make_trace(
            output=AgentResponse(final_answer="42", reasoning="Because")
        )
        summary = trace.summarize()
        assert "42" in summary

    def test_is_error_true_reflected(self):
        trace = self._make_trace(is_error=True)
        summary = trace.summarize()
        assert "True" in summary


# ===========================================================================
# Agent._get_options — callable vs static resolution
# ===========================================================================

class TestAgentGetOptions:
    """Test that Agent._get_options correctly resolves callable vs static options."""

    def _make_agent(self, options):
        from src.schemas import AgentResponse

        return Agent(options=options, response_model=AgentResponse)

    def test_static_dict_returned_as_is(self):
        opts = {"system": "prompt", "tools": {}}
        agent = self._make_agent(opts)
        assert agent._get_options() is opts

    def test_callable_is_called_on_each_access(self):
        call_count = {"n": 0}

        def factory():
            call_count["n"] += 1
            return {"system": "dynamic", "call": call_count["n"]}

        agent = self._make_agent(factory)
        result1 = agent._get_options()
        result2 = agent._get_options()

        assert call_count["n"] == 2
        assert result1["call"] == 1
        assert result2["call"] == 2

    def test_mock_object_returned_directly(self):
        mock_opts = MagicMock()
        agent = self._make_agent(mock_opts)
        # MagicMock is callable, so it will be called and return mock_opts()
        # The factory pattern means callable options get invoked
        result = agent._get_options()
        mock_opts.assert_called_once()


# ===========================================================================
# Agent._run_with_retry — retry + timeout logic (no real API calls)
# ===========================================================================

class TestAgentRetryLogic:
    """Test retry and timeout behaviour by mocking _execute_query."""

    def _make_agent(self):
        from src.schemas import AgentResponse

        return Agent(options={"system": "x"}, response_model=AgentResponse)

    def test_success_on_first_attempt(self):
        agent = self._make_agent()
        expected = [MagicMock()]

        async def run():
            with patch.object(agent, "_execute_query", AsyncMock(return_value=expected)):
                return await agent._run_with_retry("hello?")

        result = asyncio.run(run())
        assert result is expected

    def test_retries_on_failure_then_succeeds(self):
        agent = self._make_agent()
        expected = [MagicMock()]
        call_count = {"n": 0}

        async def flaky_execute(query):
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise RuntimeError("transient failure")
            return expected

        async def run():
            with (
                patch.object(agent, "_execute_query", side_effect=flaky_execute),
                patch("asyncio.sleep", AsyncMock()),
            ):
                return await agent._run_with_retry("hello?")

        result = asyncio.run(run())
        assert result is expected
        assert call_count["n"] == 2

    def test_raises_after_all_retries_exhausted(self):
        agent = self._make_agent()

        async def always_fail(query):
            raise RuntimeError("permanent failure")

        async def run():
            with (
                patch.object(agent, "_execute_query", side_effect=always_fail),
                patch("asyncio.sleep", AsyncMock()),
            ):
                return await agent._run_with_retry("hello?")

        with pytest.raises(RuntimeError, match="permanent failure"):
            asyncio.run(run())

    def test_timeout_triggers_retry(self):
        agent = self._make_agent()
        call_count = {"n": 0}
        expected = [MagicMock()]

        async def timeout_then_succeed(query):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise asyncio.TimeoutError()
            return expected

        async def run():
            with (
                patch.object(agent, "_execute_query", side_effect=timeout_then_succeed),
                patch("asyncio.sleep", AsyncMock()),
            ):
                return await agent._run_with_retry("hello?")

        result = asyncio.run(run())
        assert result is expected


# ===========================================================================
# options_utils — pure utility functions (no real SDK import needed)
# ===========================================================================

class TestOptionsUtils:
    """Test options_utils helpers that don't require the Claude SDK."""

    def test_resolve_project_root_explicit_path(self, tmp_path):
        from src.harness.utils import resolve_project_root

        result = resolve_project_root(tmp_path)
        assert result == tmp_path.resolve()

    def test_resolve_data_dirs_absolute_paths(self, tmp_path):
        from src.harness.utils import resolve_data_dirs

        abs_path = str(tmp_path)
        result = resolve_data_dirs(tmp_path, [abs_path])
        assert abs_path in result

    def test_split_opencode_model_with_slash(self):
        from src.harness.opencode.options import split_opencode_model

        provider, model = split_opencode_model("anthropic/claude-opus-4-5")
        assert provider == "anthropic"
        assert model == "claude-opus-4-5"

    def test_split_opencode_model_without_slash_uses_default_provider(self):
        from src.harness.opencode.options import split_opencode_model

        provider, model = split_opencode_model("claude-sonnet-4-6")
        assert provider == "anthropic"
        assert model == "claude-sonnet-4-6"

    def test_split_opencode_model_none_uses_default(self):
        from src.harness.opencode.options import split_opencode_model, DEFAULT_OPENCODE_MODEL

        provider, model = split_opencode_model(None)
        full = f"{provider}/{model}"
        assert full == DEFAULT_OPENCODE_MODEL

    def test_to_opencode_tools_basic(self):
        from src.harness.opencode.options import to_opencode_tools

        result = to_opencode_tools(["Read", "Bash", "Write"])
        assert result["read"] is True
        assert result["bash"] is True
        assert result["write"] is True

    def test_to_opencode_tools_skips_none_mappings(self):
        from src.harness.opencode.options import to_opencode_tools

        # BashOutput maps to None → should be excluded
        result = to_opencode_tools(["BashOutput", "Read"])
        assert "bashoutput" not in result
        assert "BashOutput" not in result
        assert "read" in result

    def test_to_opencode_tools_unknown_tool_lowercased(self):
        from src.harness.opencode.options import to_opencode_tools

        result = to_opencode_tools(["CustomTool"])
        assert "customtool" in result

    def test_normalize_permission_block_none(self):
        from src.harness.opencode.options import _normalize_permission_block

        assert _normalize_permission_block(None) == {}

    def test_normalize_permission_block_string(self):
        from src.harness.opencode.options import _normalize_permission_block

        assert _normalize_permission_block("allow") == {"*": "allow"}

    def test_normalize_permission_block_dict(self):
        from src.harness.opencode.options import _normalize_permission_block

        d = {"path": "allow"}
        assert _normalize_permission_block(d) == d

    def test_build_opencode_options_structure(self, tmp_path):
        from src.harness.opencode.options import build_opencode_options

        result = build_opencode_options(
            system="You are helpful.",
            schema={"type": "object"},
            tools=["Read", "Bash"],
            project_root=tmp_path,
            model="anthropic/claude-sonnet-4-6",
        )

        assert result["system"] == "You are helpful."
        assert result["provider_id"] == "anthropic"
        assert result["model_id"] == "claude-sonnet-4-6"
        assert "read" in result["tools"]
        assert "bash" in result["tools"]
        assert result["format"] == {"type": "json_schema", "schema": {"type": "object"}}

    def test_build_opencode_options_with_data_dirs(self, tmp_path):
        from src.harness.opencode.options import build_opencode_options

        data_dir = tmp_path / "data"
        data_dir.mkdir()

        result = build_opencode_options(
            system="prompt",
            schema={},
            tools=[],
            project_root=tmp_path,
            data_dirs=[str(data_dir)],
        )

        # Data dir paths should appear in system prompt
        assert str(data_dir) in result["system"]
        assert str(data_dir) in result["add_dirs"]


# ===========================================================================
# Claude executor — parse_response
# ===========================================================================

import types
from src.harness.claude.executor import parse_response as claude_parse
from src.harness.opencode.executor import parse_response as opencode_parse
from src.schemas import AgentResponse, SkillProposerResponse


def _make_claude_messages(structured_output=None, is_error=False):
    first = types.SimpleNamespace(
        data={"uuid": "test-uuid", "model": "sonnet", "tools": ["Read", "Bash"]}
    )
    last = types.SimpleNamespace(
        session_id="sess-123",
        duration_ms=500,
        total_cost_usd=0.05,
        num_turns=3,
        usage={"input": 100, "output": 50},
        result="some result text",
        is_error=is_error,
        structured_output=structured_output,
    )
    return [first, last]


class TestClaudeParseResponse:
    def test_parses_structured_output_successfully(self):
        msgs = _make_claude_messages({"final_answer": "4", "reasoning": "math"})
        fields = claude_parse(msgs, AgentResponse)
        assert fields["output"] is not None
        assert fields["output"].final_answer == "4"
        assert fields["parse_error"] is None
        assert fields["is_error"] is False

    def test_extracts_metadata(self):
        msgs = _make_claude_messages({"final_answer": "4", "reasoning": "math"})
        fields = claude_parse(msgs, AgentResponse)
        assert fields["uuid"] == "test-uuid"
        assert fields["model"] == "sonnet"
        assert fields["tools"] == ["Read", "Bash"]
        assert fields["session_id"] == "sess-123"
        assert fields["duration_ms"] == 500
        assert fields["total_cost_usd"] == 0.05
        assert fields["num_turns"] == 3

    def test_handles_none_structured_output(self):
        msgs = _make_claude_messages(structured_output=None)
        fields = claude_parse(msgs, AgentResponse)
        assert fields["output"] is None
        assert "No structured output" in fields["parse_error"]
        assert fields["is_error"] is True

    def test_handles_invalid_structured_output(self):
        msgs = _make_claude_messages({"wrong_field": "value"})
        fields = claude_parse(msgs, AgentResponse)
        assert fields["output"] is None
        assert "ValidationError" in fields["parse_error"]

    def test_complex_schema(self):
        data = {
            "action": "create",
            "target_skill": None,
            "proposed_skill": "calculator",
            "justification": "needed for math",
            "related_iterations": ["iter-1"],
        }
        msgs = _make_claude_messages(structured_output=data)
        fields = claude_parse(msgs, SkillProposerResponse)
        assert fields["output"].action == "create"
        assert fields["output"].proposed_skill == "calculator"
        assert fields["output"].related_iterations == ["iter-1"]

    def test_is_error_propagates_from_last_message(self):
        msgs = _make_claude_messages(
            {"final_answer": "4", "reasoning": "math"}, is_error=True
        )
        fields = claude_parse(msgs, AgentResponse)
        assert fields["is_error"] is True


# ===========================================================================
# OpenCode executor — parse_response
# ===========================================================================

def _make_opencode_message(info=None, parts=None, session_id="sess-456"):
    return types.SimpleNamespace(
        info=info,
        parts=parts or [],
        session_id=session_id,
    )


def _make_opencode_get_options(model_id="claude-sonnet-4-6", tools=None):
    return lambda: {
        "model_id": model_id,
        "tools": tools or {"read": True, "bash": True},
    }


class TestOpencodeParseResponse:
    def test_parses_structured_output_from_info(self):
        msg = _make_opencode_message(
            info={"structured_output": {"final_answer": "4", "reasoning": "math"}, "cost": 0.05, "tokens": {}}
        )
        fields = opencode_parse([msg], AgentResponse, _make_opencode_get_options())
        assert fields["output"] is not None
        assert fields["output"].final_answer == "4"
        assert fields["parse_error"] is None

    def test_extracts_cost_from_info(self):
        msg = _make_opencode_message(
            info={"structured_output": {"final_answer": "4", "reasoning": "math"}, "cost": 0.123, "tokens": {"input": 10}}
        )
        fields = opencode_parse([msg], AgentResponse, _make_opencode_get_options())
        assert fields["total_cost_usd"] == 0.123
        assert fields["usage"] == {"input": 10}

    def test_handles_missing_structured_output(self):
        msg = _make_opencode_message(info={"cost": 0.01, "tokens": {}})
        fields = opencode_parse([msg], AgentResponse, _make_opencode_get_options())
        assert fields["output"] is None
        assert fields["parse_error"] is not None

    def test_handles_missing_info(self):
        msg = types.SimpleNamespace(parts=[], session_id="sess-456")
        fields = opencode_parse([msg], AgentResponse, _make_opencode_get_options())
        assert fields["output"] is None
        assert fields["parse_error"] is not None

    def test_handles_invalid_structured_output(self):
        msg = _make_opencode_message(
            info={"structured_output": {"wrong": "fields"}, "cost": 0, "tokens": {}}
        )
        fields = opencode_parse([msg], AgentResponse, _make_opencode_get_options())
        assert fields["output"] is None
        assert "ValidationError" in fields["parse_error"]

    def test_complex_schema(self):
        data = {
            "action": "edit",
            "target_skill": "math-helper",
            "proposed_skill": "improved calculator",
            "justification": "needs fixing",
            "related_iterations": [],
        }
        msg = _make_opencode_message(
            info={"structured_output": data, "cost": 0.05, "tokens": {}}
        )
        fields = opencode_parse([msg], SkillProposerResponse, _make_opencode_get_options())
        assert fields["output"].action == "edit"
        assert fields["output"].target_skill == "math-helper"

    def test_extracts_text_from_parts(self):
        msg = _make_opencode_message(
            info={"structured_output": {"final_answer": "4", "reasoning": "math"}, "cost": 0, "tokens": {}},
            parts=[
                {"type": "step-start", "id": "1"},
                {"type": "text", "text": "hello "},
                {"type": "text", "text": "world"},
                {"type": "step-finish", "id": "2"},
            ],
        )
        fields = opencode_parse([msg], AgentResponse, _make_opencode_get_options())
        assert fields["result"] == "hello world"

    def test_structured_key_fallback(self):
        msg = _make_opencode_message(
            info={"structured": {"final_answer": "4", "reasoning": "math"}, "cost": 0, "tokens": {}}
        )
        fields = opencode_parse([msg], AgentResponse, _make_opencode_get_options())
        assert fields["output"] is not None
        assert fields["output"].final_answer == "4"

    def test_model_from_options(self):
        msg = _make_opencode_message(
            info={"structured_output": {"final_answer": "4", "reasoning": "math"}, "cost": 0, "tokens": {}}
        )
        fields = opencode_parse([msg], AgentResponse, _make_opencode_get_options(model_id="opus"))
        assert fields["model"] == "opus"


# ===========================================================================
# TestSdkConfig — codex extension (appended to TestSdkConfig pattern above)
# ===========================================================================

class TestSdkConfigCodex:
    """Test codex SDK toggle and is_codex_sdk() helper."""

    def setup_method(self):
        set_sdk("claude")

    def teardown_method(self):
        set_sdk("claude")

    def test_set_sdk_to_codex(self):
        from src.harness.sdk_config import is_codex_sdk
        set_sdk("codex")
        assert get_sdk() == "codex"
        assert is_codex_sdk() is True
        assert is_claude_sdk() is False
        assert is_opencode_sdk() is False

    def test_is_codex_sdk_false_by_default(self):
        from src.harness.sdk_config import is_codex_sdk
        assert is_codex_sdk() is False

    def test_set_sdk_codex_then_back_to_claude(self):
        from src.harness.sdk_config import is_codex_sdk
        set_sdk("codex")
        set_sdk("claude")
        assert is_claude_sdk() is True
        assert is_codex_sdk() is False

    def test_invalid_sdk_still_raises(self):
        with pytest.raises(ValueError, match="Invalid SDK"):
            set_sdk("unknown_harness")  # type: ignore[arg-type]


# ===========================================================================
# TestCodexOptions — build_codex_options()
# ===========================================================================

class TestCodexOptions:
    """Test the Codex options builder — pure dict construction, no SDK calls."""

    def test_basic_structure(self, tmp_path):
        from src.harness.codex.options import build_codex_options

        result = build_codex_options(
            system="You are helpful.",
            schema={"type": "object"},
            tools=["Read", "Bash"],
            project_root=tmp_path,
            model="codex-mini-latest",
        )

        assert result["system"] == "You are helpful."
        assert result["output_schema"]["type"] == "object"
        assert result["output_schema"]["additionalProperties"] is False
        assert result["model"] == "codex-mini-latest"
        assert result["working_directory"] == str(tmp_path.resolve())
        assert "Read" in result["tools"]
        assert "Bash" in result["tools"]

    def test_default_model_used_when_none(self, tmp_path):
        from src.harness.codex.options import build_codex_options, DEFAULT_CODEX_MODEL

        result = build_codex_options(
            system="prompt",
            schema={},
            tools=[],
            project_root=tmp_path,
            model=None,
        )

        assert result["model"] == DEFAULT_CODEX_MODEL

    def test_data_dirs_appended_to_system(self, tmp_path):
        from src.harness.codex.options import build_codex_options

        data_dir = tmp_path / "extra_data"
        data_dir.mkdir()

        result = build_codex_options(
            system="base system",
            schema={},
            tools=[],
            project_root=tmp_path,
            data_dirs=[str(data_dir)],
        )

        assert str(data_dir) in result["system"]
        assert "Additional data directories" in result["system"]
        assert str(data_dir) in result["data_dirs"]

    def test_tools_stored_as_list(self, tmp_path):
        from src.harness.codex.options import build_codex_options

        result = build_codex_options(
            system="s",
            schema={},
            tools=["Read", "Write", "Bash"],
            project_root=tmp_path,
        )

        assert isinstance(result["tools"], list)
        assert set(result["tools"]) == {"Read", "Write", "Bash"}

    def test_no_data_dirs_system_unchanged(self, tmp_path):
        from src.harness.codex.options import build_codex_options

        result = build_codex_options(
            system="clean prompt",
            schema={},
            tools=[],
            project_root=tmp_path,
            data_dirs=None,
        )

        assert result["system"] == "clean prompt"
        assert result["data_dirs"] == []


# ===========================================================================
# TestCodexParseResponse — parse_response()
# ===========================================================================

import json as _json


def _make_codex_turn(final_response=None, turn_id="turn-abc", thread_id="thread-xyz"):
    import types
    return types.SimpleNamespace(
        final_response=final_response,
        id=turn_id,
        thread_id=thread_id,
    )


def _make_codex_get_options(model="codex-mini-latest", tools=None):
    return lambda: {
        "model": model,
        "tools": tools or ["Read", "Bash"],
        "working_directory": "/tmp",
        "output_schema": {},
    }


class TestCodexParseResponse:
    """Test parse_response for the Codex harness."""

    def test_parses_valid_json_response(self):
        from src.harness.codex.executor import parse_response

        payload = _json.dumps({"final_answer": "42", "reasoning": "math"})
        turn = _make_codex_turn(final_response=payload)
        fields = parse_response([turn], AgentResponse, _make_codex_get_options())

        assert fields["output"] is not None
        assert fields["output"].final_answer == "42"
        assert fields["parse_error"] is None
        assert fields["is_error"] is False

    def test_handles_non_json_response(self):
        from src.harness.codex.executor import parse_response

        turn = _make_codex_turn(final_response="not JSON at all")
        fields = parse_response([turn], AgentResponse, _make_codex_get_options())

        assert fields["output"] is None
        assert "JSONDecodeError" in fields["parse_error"]
        assert fields["is_error"] is True

    def test_handles_empty_final_response(self):
        from src.harness.codex.executor import parse_response

        turn = _make_codex_turn(final_response="")
        fields = parse_response([turn], AgentResponse, _make_codex_get_options())

        assert fields["output"] is None
        assert "No response from Codex" in fields["parse_error"]
        assert fields["is_error"] is True

    def test_handles_none_final_response(self):
        from src.harness.codex.executor import parse_response

        turn = _make_codex_turn(final_response=None)
        fields = parse_response([turn], AgentResponse, _make_codex_get_options())

        assert fields["output"] is None
        assert fields["is_error"] is True

    def test_model_comes_from_options(self):
        from src.harness.codex.executor import parse_response

        payload = _json.dumps({"final_answer": "x", "reasoning": "y"})
        turn = _make_codex_turn(final_response=payload)
        fields = parse_response([turn], AgentResponse, _make_codex_get_options(model="my-model"))

        assert fields["model"] == "my-model"

    def test_result_text_equals_final_response(self):
        from src.harness.codex.executor import parse_response

        payload = _json.dumps({"final_answer": "z", "reasoning": "q"})
        turn = _make_codex_turn(final_response=payload)
        fields = parse_response([turn], AgentResponse, _make_codex_get_options())

        assert fields["result"] == payload

    def test_cost_is_always_zero(self):
        from src.harness.codex.executor import parse_response

        turn = _make_codex_turn(final_response="")
        fields = parse_response([turn], AgentResponse, _make_codex_get_options())

        assert fields["total_cost_usd"] == 0.0

    def test_uuid_from_turn_id(self):
        from src.harness.codex.executor import parse_response

        payload = _json.dumps({"final_answer": "a", "reasoning": "b"})
        turn = _make_codex_turn(final_response=payload, turn_id="my-turn-id")
        fields = parse_response([turn], AgentResponse, _make_codex_get_options())

        assert fields["uuid"] == "my-turn-id"

    def test_session_id_from_thread_id(self):
        from src.harness.codex.executor import parse_response

        payload = _json.dumps({"final_answer": "a", "reasoning": "b"})
        turn = _make_codex_turn(final_response=payload, thread_id="my-thread-id")
        fields = parse_response([turn], AgentResponse, _make_codex_get_options())

        assert fields["session_id"] == "my-thread-id"

    def test_json_schema_validation_error_recorded(self):
        from src.harness.codex.executor import parse_response

        # Valid JSON but wrong schema fields
        payload = _json.dumps({"wrong_field": "value"})
        turn = _make_codex_turn(final_response=payload)
        fields = parse_response([turn], AgentResponse, _make_codex_get_options())

        assert fields["output"] is None
        assert "ValidationError" in fields["parse_error"]


# ===========================================================================
# TestSdkConfigGoose — goose SDK toggle and is_goose_sdk() helper
# ===========================================================================

class TestSdkConfigGoose:
    """Test goose SDK toggle and is_goose_sdk() helper."""

    def setup_method(self):
        set_sdk("claude")

    def teardown_method(self):
        set_sdk("claude")

    def test_set_sdk_to_goose(self):
        from src.harness.sdk_config import is_goose_sdk
        set_sdk("goose")
        assert get_sdk() == "goose"
        assert is_goose_sdk() is True
        assert is_claude_sdk() is False
        assert is_opencode_sdk() is False

    def test_is_goose_sdk_false_by_default(self):
        from src.harness.sdk_config import is_goose_sdk
        assert is_goose_sdk() is False

    def test_set_sdk_goose_then_back_to_claude(self):
        from src.harness.sdk_config import is_goose_sdk
        set_sdk("goose")
        set_sdk("claude")
        assert is_claude_sdk() is True
        assert is_goose_sdk() is False

    def test_invalid_sdk_still_raises(self):
        with pytest.raises(ValueError, match="Invalid SDK"):
            set_sdk("notreal")  # type: ignore[arg-type]


# ===========================================================================
# TestGooseOptions — build_goose_options()
# ===========================================================================

class TestGooseSplitModel:
    """Test split_goose_model() directly for all branches."""

    def test_none_returns_defaults(self):
        from src.harness.goose.options import split_goose_model, DEFAULT_GOOSE_PROVIDER, DEFAULT_GOOSE_MODEL

        provider, model = split_goose_model(None)
        assert provider == DEFAULT_GOOSE_PROVIDER
        assert model == DEFAULT_GOOSE_MODEL

    def test_with_slash_splits_on_first_slash(self):
        from src.harness.goose.options import split_goose_model

        provider, model = split_goose_model("openrouter/gpt-4-turbo")
        assert provider == "openrouter"
        assert model == "gpt-4-turbo"

    def test_without_slash_uses_default_provider(self):
        from src.harness.goose.options import split_goose_model, DEFAULT_GOOSE_PROVIDER

        provider, model = split_goose_model("claude-sonnet-4-6")
        assert provider == DEFAULT_GOOSE_PROVIDER
        assert model == "claude-sonnet-4-6"

    def test_multiple_slashes_splits_only_on_first(self):
        from src.harness.goose.options import split_goose_model

        provider, model = split_goose_model("some-provider/model/version")
        assert provider == "some-provider"
        assert model == "model/version"


class TestGooseExecuteQueryErrors:
    """Test execute_query error path when goose CLI is not installed."""

    def test_raises_runtime_error_when_goose_not_found(self):
        import asyncio
        from unittest.mock import patch
        from src.harness.goose.executor import execute_query

        async def run():
            with patch("shutil.which", return_value=None):
                await execute_query({"system": "test", "output_schema": {}}, "query")

        with pytest.raises(RuntimeError, match="Goose CLI not found"):
            asyncio.run(run())


class TestGooseOptions:
    """Test the Goose options builder — pure dict construction, no subprocess calls."""

    def test_basic_structure(self, tmp_path):
        from src.harness.goose.options import build_goose_options

        result = build_goose_options(
            system="You are helpful.",
            schema={"type": "object"},
            tools=["Read", "Bash"],
            project_root=tmp_path,
            model="anthropic/claude-sonnet-4-6",
        )

        assert result["system"] == "You are helpful."
        assert result["output_schema"]["type"] == "object"
        assert result["output_schema"]["additionalProperties"] is False
        assert result["provider"] == "anthropic"
        assert result["model"] == "claude-sonnet-4-6"
        assert result["working_directory"] == str(tmp_path.resolve())
        assert "Read" in result["tools"]
        assert "Bash" in result["tools"]

    def test_default_model_and_provider_when_none(self, tmp_path):
        from src.harness.goose.options import (
            build_goose_options,
            DEFAULT_GOOSE_MODEL,
            DEFAULT_GOOSE_PROVIDER,
        )

        result = build_goose_options(
            system="prompt",
            schema={},
            tools=[],
            project_root=tmp_path,
            model=None,
        )

        assert result["model"] == DEFAULT_GOOSE_MODEL
        assert result["provider"] == DEFAULT_GOOSE_PROVIDER

    def test_model_with_slash_splits(self, tmp_path):
        from src.harness.goose.options import build_goose_options

        result = build_goose_options(
            system="s",
            schema={},
            tools=[],
            project_root=tmp_path,
            model="openrouter/gpt-5",
        )

        assert result["provider"] == "openrouter"
        assert result["model"] == "gpt-5"

    def test_data_dirs_appended_to_system(self, tmp_path):
        from src.harness.goose.options import build_goose_options

        data_dir = tmp_path / "extra_data"
        data_dir.mkdir()

        result = build_goose_options(
            system="base system",
            schema={},
            tools=[],
            project_root=tmp_path,
            data_dirs=[str(data_dir)],
        )

        assert str(data_dir) in result["system"]
        assert "Additional data directories" in result["system"]
        assert str(data_dir) in result["data_dirs"]

    def test_tools_stored_as_list(self, tmp_path):
        from src.harness.goose.options import build_goose_options

        result = build_goose_options(
            system="s",
            schema={},
            tools=["Read", "Write", "Bash"],
            project_root=tmp_path,
        )

        assert isinstance(result["tools"], list)
        assert set(result["tools"]) == {"Read", "Write", "Bash"}

    def test_no_data_dirs_system_unchanged(self, tmp_path):
        from src.harness.goose.options import build_goose_options

        result = build_goose_options(
            system="clean prompt",
            schema={},
            tools=[],
            project_root=tmp_path,
            data_dirs=None,
        )

        assert result["system"] == "clean prompt"
        assert result["data_dirs"] == []


# ===========================================================================
# TestGooseParseResponse — parse_response()
# ===========================================================================

import types as _types


def _make_goose_result(stdout="", stderr="", returncode=0):
    return _types.SimpleNamespace(
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
    )


def _make_goose_conversation_stdout(structured_output: dict) -> str:
    """Build a fake Goose --output-format json stdout with the conversation JSON."""
    conversation = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "query"}]},
            {"role": "assistant", "content": [
                {"type": "toolRequest", "id": "tool1", "toolCall": {
                    "status": "success",
                    "value": {
                        "name": "recipe__final_output",
                        "arguments": structured_output,
                    }
                }}
            ]},
            {"role": "user", "content": [{"type": "toolResponse", "id": "tool1", "toolResult": {"status": "success"}}]},
            {"role": "assistant", "content": [
                {"type": "text", "text": _json.dumps(structured_output)}
            ]},
        ]
    }
    return f"starting session | provider: anthropic model: test\n{_json.dumps(conversation)}"


def _make_goose_get_options(provider="anthropic", model="claude-sonnet-4-6", tools=None):
    return lambda: {
        "provider": provider,
        "model": model,
        "tools": tools or ["Read", "Bash"],
        "working_directory": "/tmp",
        "output_schema": {},
    }


class TestGooseParseResponse:
    """Test parse_response for the Goose harness."""

    def test_parses_structured_output_from_conversation(self):
        from src.harness.goose.executor import parse_response as goose_parse

        stdout = _make_goose_conversation_stdout({"final_answer": "4", "reasoning": "math"})
        result = _make_goose_result(stdout=stdout)
        fields = goose_parse([result], AgentResponse, _make_goose_get_options())

        assert fields["output"] is not None
        assert fields["output"].final_answer == "4"
        assert fields["parse_error"] is None
        assert fields["is_error"] is False

    def test_handles_no_json_in_stdout(self):
        from src.harness.goose.executor import parse_response as goose_parse

        result = _make_goose_result(stdout="not JSON at all, no braces anywhere")
        fields = goose_parse([result], AgentResponse, _make_goose_get_options())

        assert fields["output"] is None
        assert fields["parse_error"] is not None
        assert fields["is_error"] is True

    def test_handles_empty_stdout(self):
        from src.harness.goose.executor import parse_response as goose_parse

        result = _make_goose_result(stdout="")
        fields = goose_parse([result], AgentResponse, _make_goose_get_options())

        assert fields["output"] is None
        assert fields["parse_error"] is not None
        assert fields["is_error"] is True

    def test_handles_nonzero_returncode(self):
        from src.harness.goose.executor import parse_response as goose_parse

        result = _make_goose_result(stdout="some output", stderr="critical error occurred", returncode=1)
        fields = goose_parse([result], AgentResponse, _make_goose_get_options())

        assert fields["is_error"] is True
        assert "critical error occurred" in fields["parse_error"]

    def test_model_comes_from_options(self):
        from src.harness.goose.executor import parse_response as goose_parse

        stdout = _make_goose_conversation_stdout({"final_answer": "x", "reasoning": "y"})
        result = _make_goose_result(stdout=stdout)
        fields = goose_parse([result], AgentResponse, _make_goose_get_options(model="my-model"))

        assert fields["model"] == "my-model"

    def test_provider_comes_from_options(self):
        from src.harness.goose.executor import parse_response as goose_parse

        stdout = _make_goose_conversation_stdout({"final_answer": "x", "reasoning": "y"})
        result = _make_goose_result(stdout=stdout)
        opts_fn = _make_goose_get_options(provider="openrouter")
        fields = goose_parse([result], AgentResponse, opts_fn)

        # provider is stored in the options dict, verify it's accessible
        assert opts_fn()["provider"] == "openrouter"
        # output should still parse fine
        assert fields["output"] is not None

    def test_cost_is_always_zero(self):
        from src.harness.goose.executor import parse_response as goose_parse

        result = _make_goose_result(stdout="")
        fields = goose_parse([result], AgentResponse, _make_goose_get_options())

        assert fields["total_cost_usd"] == 0.0

    def test_uuid_and_session_id_are_unknown(self):
        from src.harness.goose.executor import parse_response as goose_parse

        stdout = _make_goose_conversation_stdout({"final_answer": "a", "reasoning": "b"})
        result = _make_goose_result(stdout=stdout)
        fields = goose_parse([result], AgentResponse, _make_goose_get_options())

        assert fields["uuid"] == "unknown"
        assert fields["session_id"] == "unknown"

    def test_json_validation_error_recorded(self):
        from src.harness.goose.executor import parse_response as goose_parse

        # Valid conversation JSON but tool call has wrong schema fields
        stdout = _make_goose_conversation_stdout({"wrong_field": "value"})
        result = _make_goose_result(stdout=stdout)
        fields = goose_parse([result], AgentResponse, _make_goose_get_options())

        assert fields["output"] is None
        assert fields["parse_error"] is not None
        assert fields["is_error"] is True

    def test_extracts_from_tool_call_arguments(self):
        from src.harness.goose.executor import parse_response as goose_parse

        stdout = _make_goose_conversation_stdout({"final_answer": "7", "reasoning": "logic"})
        result = _make_goose_result(stdout=stdout)
        fields = goose_parse([result], AgentResponse, _make_goose_get_options())

        assert fields["output"] is not None
        assert fields["output"].final_answer == "7"
        assert fields["parse_error"] is None
        assert fields["raw_structured_output"] == {"final_answer": "7", "reasoning": "logic"}
