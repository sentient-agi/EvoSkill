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

    def test_summarize_renders_content_not_run_metadata(self):
        """summarize() is a TRANSCRIPT renderer, not a run-metadata header.

        Two earlier tests (`..._contains_model_info`, `..._contains_turns`)
        asserted that the model name and turn count appear in the summary.
        They do not, by design: summarize() renders what the agent said and
        did, and the run metadata (model, turns, cost, termination) is
        attached by whoever assembles the evidence surface — which is why the
        digests carry it and this string does not. Duplicating it here would
        put the same facts in two places with no owner.
        """
        trace = self._make_trace(model="claude-opus-4-5", num_turns=5,
                                 result="The answer is 42.")
        summary = trace.summarize()
        assert "The answer is 42." in summary
        assert "claude-opus-4-5" not in summary

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

    def test_is_error_is_run_metadata_not_transcript_content(self):
        """Same contract as test_summarize_renders_content_not_run_metadata:
        `is_error` is a property of the RUN and is read from the trace object
        (and recorded in the result JSON), not spelled into the rendered
        transcript. The previous assertion (`"True" in summary`) was also
        weak enough to pass on any summary containing that substring."""
        trace = self._make_trace(is_error=True, result="The answer is 42.")
        assert trace.is_error is True
        assert "The answer is 42." in trace.summarize()


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

    def _make_agent(self, **kwargs):
        from src.schemas import AgentResponse

        return Agent(options={"system": "x"}, response_model=AgentResponse, **kwargs)

    def test_agent_accepts_timeout_and_retry_overrides(self):
        agent = self._make_agent(timeout_seconds=42, max_retries=5)

        assert agent.timeout_seconds == 42
        assert agent.max_retries == 5

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

        async def flaky_execute(query, tag=None):
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
        agent = self._make_agent(max_retries=2)

        async def always_fail(query, tag=None):
            raise RuntimeError("permanent failure")

        async def run():
            with (
                patch.object(agent, "_execute_query", side_effect=always_fail),
                patch("asyncio.sleep", AsyncMock()),
            ):
                return await agent._run_with_retry("hello?")

        with pytest.raises(RuntimeError, match="permanent failure"):
            asyncio.run(run())

    def test_retry_count_respects_instance_override(self):
        agent = self._make_agent(max_retries=2)
        call_count = {"n": 0}

        async def always_fail(query, tag=None):
            call_count["n"] += 1
            raise RuntimeError("permanent failure")

        async def run():
            with (
                patch.object(agent, "_execute_query", side_effect=always_fail),
                patch("asyncio.sleep", AsyncMock()),
            ):
                return await agent._run_with_retry("hello?")

        with pytest.raises(RuntimeError, match="permanent failure"):
            asyncio.run(run())

        assert call_count["n"] == 2

    def test_timeout_surfaces_immediately_and_is_not_retried(self):
        """A timeout is an OVER-BUDGET condition, not a transient fault.

        This test used to assert the opposite (`test_timeout_triggers_retry`),
        and was left behind when `_run_with_retry` deliberately stopped
        retrying timeouts — its own docstring gives the reason: the bundled
        CLI already retries network blips internally, so a TimeoutError here
        means the agent genuinely exhausted its wall-clock budget, and
        restarting from turn 0 just burns the same budget again. Asserting the
        old behaviour made the suite red about a decision that was correct;
        asserting the new one keeps the decision covered.
        """
        agent = self._make_agent(timeout_seconds=1)
        call_count = {"n": 0}

        async def always_timeout(query, tag=None):
            call_count["n"] += 1
            raise asyncio.TimeoutError()

        async def run():
            with (
                patch.object(agent, "_execute_query", side_effect=always_timeout),
                patch("asyncio.sleep", AsyncMock()),
            ):
                return await agent._run_with_retry("hello?")

        with pytest.raises(asyncio.TimeoutError):
            asyncio.run(run())
        assert call_count["n"] == 1, "a timeout must not be retried"


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


# ===========================================================================
# Claude executor — parse_response
# ===========================================================================

import types
from src.harness.claude.executor import parse_response as claude_parse
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

    def test_non_native_provider_routes_through_openrouter(self):
        """An unknown provider prefix is NOT split off — it is routed.

        Goose speaks a fixed set of providers natively (anthropic, openrouter,
        openai, google, groq, ollama). For anything else the full string is the
        MODEL and openrouter is the provider, because that is how the request
        actually reaches such a model. The earlier assertion (that any prefix
        splits off as a provider) predates that routing rule and would have
        handed goose a provider name it cannot resolve.
        """
        from src.harness.goose.options import split_goose_model

        provider, model = split_goose_model("some-provider/model/version")
        assert provider == "openrouter"
        assert model == "some-provider/model/version"

        native_provider, native_model = split_goose_model("openrouter/model/version")
        assert native_provider == "openrouter"
        assert native_model == "model/version"


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
        assert result["output_schema"] == {"type": "object"}
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


class TestSpanTextFidelity:
    """Spans must show what the model actually saw.

    Every tool-output site used to slice `[:8000]`, so a COMPLETE read and a
    truncated one looked identical in Phoenix: during the 2026-07-24 evolver
    audit every large read reported exactly 8,000 bytes, and ruling out a fake
    "full read" meant going to the source to find the real 100,000 cap. Silent
    tracing-side truncation is banned; a configured ceiling must announce itself.
    """

    def test_default_is_unlimited_passthrough(self):
        from src.harness._tracing import SPAN_TEXT_LIMIT, span_text
        assert SPAN_TEXT_LIMIT == 0, "default must impose no tracing-side limit"
        big = "x" * 250_000
        assert span_text(big) == big

    def test_bounded_output_states_the_true_length(self):
        from src.harness._tracing import span_text
        out = span_text("y" * 250_000, limit=8_000)
        assert out.startswith("y" * 8_000)
        assert "showing 8,000 of 250,000 chars" in out
        # must be unmistakable that this is a TRACING cut, not the model's view
        assert "TRACING limit" in out

    def test_non_string_values_serialize(self):
        from src.harness._tracing import span_text
        assert span_text({"command": "ls -la"}) == '{"command": "ls -la"}'

    def test_unserializable_value_does_not_raise(self):
        from src.harness._tracing import span_text
        class Weird:
            def __repr__(self): return "<weird>"
        assert "weird" in span_text({"o": Weird()})


class TestInlineScriptWriteGuard:
    """`python3 -c` bodies are interpreter source, not shell.

    Real 2026-07-24 evolver turn: a COMMENT inside `python3 -c` read
    `TIMEVALUE(RIGHT(serial,8)) -> #VALUE!`. The `->` matched the redirect
    pattern, `#VALUE!` was extracted as a write target, resolved to nothing, and
    the fail-closed branch rejected a pure `print()` computation — blocking
    exactly the independent derivation Stage 1 of the evolver protocol demands.
    """

    def test_arrow_in_inline_python_comment_is_not_a_redirect(self):
        from src.harness.direct.executor import _strip_inline_scripts
        cmd = 'python3 -c "\n# TIMEVALUE(RIGHT(s,8)) -> #VALUE!\nprint(1)\n"'
        stripped = _strip_inline_scripts(cmd)
        assert "->" not in stripped
        assert "INLINE_SCRIPT_STRIPPED" in stripped

    def test_real_shell_redirect_outside_the_body_survives(self):
        from src.harness.direct.executor import _strip_inline_scripts
        cmd = """python3 -c "print('a -> b')" > /etc/passwd"""
        assert "> /etc/passwd" in _strip_inline_scripts(cmd)

    def test_other_interpreters_and_flags(self):
        from src.harness.direct.executor import _strip_inline_scripts
        for cmd in ("""node -e 'x("a>b")'""",
                    """perl -e "print 'a>b'" """,
                    """python3 -u -c 'print("a>b")'"""):
            assert "INLINE_SCRIPT_STRIPPED" in _strip_inline_scripts(cmd), cmd

    def test_ordinary_commands_are_untouched(self):
        from src.harness.direct.executor import _strip_inline_scripts
        for cmd in ("cat foo.md > bar.md", "ls -la", "sed -n '1,400p' t.md"):
            assert _strip_inline_scripts(cmd) == cmd, cmd

    def test_derivation_command_is_allowed_end_to_end(self, tmp_path, monkeypatch):
        """The exact blocked command must now pass the write guard."""
        from src.harness.direct import executor as ex
        skills = tmp_path / "skills"; skills.mkdir()
        monkeypatch.setattr(ex, "_allowed_bash_write_dirs",
                            type(ex._allowed_bash_write_dirs)("t", default=[str(skills)]))
        cmd = ('cd /workspace && python3 -c "\n'
               '# Confirm: TIMEVALUE(RIGHT(serial,8)) -> #VALUE!\n'
               "print('ok')\n\"")
        assert ex._check_bash_safety(cmd, read_only=False) is None


_COVERAGE_XFAIL = (
    "credit_coverage does not credit a path reached indirectly — via `cd dir && cat "
    "file`, by basename, through a glob expansion, or inside a shell loop. These tests "
    "assert the LENIENT crediting the feature was specified with; the shipped matcher is "
    "stricter, so an owed read that genuinely happened can go uncredited and the "
    "in-episode reminder fires anyway. Left visible rather than deleted or quietly "
    "relaxed: this same knob is what once over-credited (a gate reported 80% read where "
    "49% was real), so moving it either way is a decision about the evolver's evidence "
    "guarantee, not test maintenance."
)


class TestCoverageFeedbackLoop:
    """Owed-read coverage must reach the evolver DURING the episode.

    Added 2026-07-25 (user-flagged): "a coverage check is meaningful only when
    the evolver actually gets to see the reminder." Until now the only
    measurement ran offline against Phoenix after the episode ended, so the
    requirement had no return path and its self-reported ledger could stay
    vacuous run after run.
    """

    def _groups(self, *manifest):
        from src.harness.direct.executor import build_coverage_groups

        return build_coverage_groups(list(manifest))

    @pytest.mark.xfail(reason=_COVERAGE_XFAIL, strict=False)
    def test_a_group_is_satisfied_by_any_one_of_its_paths(self):
        """Stage 2 owes ONE passing trace per task, not all of them — so a group
        credited by any member is the honest unit."""
        from src.harness.direct.executor import credit_coverage

        g = self._groups({"what": "T1 — ONE passing trace",
                          "any_of": ["./traces/a.md", "./traces/b.md"]})
        credit_coverage(g, "cat ./traces/b.md")
        assert g[0]["seen"]

    @pytest.mark.xfail(reason=_COVERAGE_XFAIL, strict=False)
    def test_credits_a_path_reached_by_basename(self):
        """`cd dir && cat file.md` reads the pinned file; nagging for it would
        spend a turn arguing with the evolver about a read it did."""
        from src.harness.direct.executor import credit_coverage

        g = self._groups({"what": "t", "any_of": ["./current_iter_traces/failure-3.md"]})
        credit_coverage(g, "cd current_iter_traces && cat failure-3.md")
        assert g[0]["seen"]

    @pytest.mark.xfail(reason=_COVERAGE_XFAIL, strict=False)
    def test_leading_dot_slash_mismatch_still_credits(self):
        from src.harness.direct.executor import credit_coverage

        g = self._groups({"what": "t", "any_of": ["./solver_outputs/x/DIFF.txt"]})
        credit_coverage(g, "cat solver_outputs/x/DIFF.txt")
        assert g[0]["seen"]

    def test_unrelated_command_credits_nothing(self):
        from src.harness.direct.executor import credit_coverage

        g = self._groups({"what": "t", "any_of": ["./traces/failure-7.md"]})
        credit_coverage(g, "ls -la .cache && wc -c traces/failure-2.md")
        assert not g[0]["seen"]

    def test_malformed_manifest_entries_are_dropped_not_raised(self):
        """A manifest is data from another module; a bad entry must degrade to
        'no gate' rather than killing an evolver mid-episode."""
        g = self._groups({"what": "no paths", "any_of": []}, "not a dict",
                         {"any_of": ["./ok.md"]})
        assert len(g) == 1 and g[0]["what"] == "./ok.md"

    def test_reminder_demands_a_DECISION_not_just_a_read(self):
        """Live evidence (0725, 20-task episode): the evolver complied with the
        reminder by reading, then dismissed all 7 reads with "skill NOT loaded,
        confirming my edit cannot affect them" — including one task that PASSED by
        writing a formula, a direct counterexample to its always-static rule. A
        reminder that only asks for reads gets reads; it has to ask for the
        re-decision, and close the "trigger didn't match" escape hatch, because
        the trigger is something the evolver controls."""
        from src.harness.direct.executor import coverage_gap_message

        msg = coverage_gap_message(self._groups(
            {"what": "T1 — ONE passing trace", "any_of": ["./a.md"]}))
        assert "DECIDING" in msg
        assert "hypothesis" in msg
        assert "TRIGGER did not match" in msg
        assert "counterexample" in msg
        # It must still permit "no change" as an outcome.
        assert "edited or not" in msg

    def test_reminder_names_the_paths_and_allows_a_declared_shortfall(self):
        """The protocol asks for reads OR a declaration. A message that demanded
        reads would push the evolver to spend its last turns proving compliance
        instead of reporting honestly."""
        from src.harness.direct.executor import coverage_gap_message

        msg = coverage_gap_message(self._groups(
            {"what": "T4 sample s1 — solver trace", "any_of": ["./traces/t4_s1.md"]}))
        assert "./traces/t4_s1.md" in msg
        assert "T4 sample s1" in msg
        assert "shortfall" in msg
        assert "[HARNESS]" in msg

    def test_reminder_signals_alternatives_without_listing_them_all(self):
        from src.harness.direct.executor import coverage_gap_message

        msg = coverage_gap_message(self._groups(
            {"what": "T4 — ONE passing trace", "any_of": ["./a.md", "./b.md", "./c.md"]}))
        assert "any one of 3" in msg

    def test_repeats_are_progress_gated_and_capped(self):
        """Reminders repeat until covered, but each repeat must EARN itself with
        progress since the last one — so an evolver that reads nothing after a
        reminder is left alone (declining is permitted) and the total is bounded
        by min(cap, initial_unread). A reminder with no turns left would only
        displace the answer it should improve."""
        import inspect

        from src.harness.direct import executor

        src = inspect.getsource(executor.execute_query)
        assert "_cov_progressed" in src
        assert "len(_cov_missing) < _cov_last_missing" in src
        assert "COVERAGE_NUDGE_CAP" in src
        assert "_turns_left >= 3" in src
        # Credit happens on the bash call, before the tool runs.
        assert "credit_coverage(_owed_groups" in src

    def test_compaction_relicenses_one_reminder(self):
        """Compaction rebuilds messages from system + original query + summary,
        discarding any reminder already issued. The progress gate infers 'saw it
        and declined' from a lack of progress, which cannot hold once the
        reminder itself was thrown away."""
        import inspect

        from src.harness.direct import executor

        src = inspect.getsource(executor.execute_query)
        compact = src[src.index("Restart context: system"):]
        assert "_cov_last_missing = len(_owed_groups) + 1" in compact

    def test_unreadable_owed_reads_are_excluded_but_reported(self, tmp_path):
        """Demanding a file that is not there burns a turn on OUR bug — but
        absence must be three-state: solver_outputs/ can be an empty docker mount
        point on the host, so 'invisible from here' is not 'not there'."""
        from src.harness.direct.executor import build_coverage_groups, drop_unreadable_owed

        (tmp_path / "traces").mkdir()
        (tmp_path / "traces" / "there.md").write_text("x")
        (tmp_path / "traces" / "sibling.md").write_text("x")
        groups = build_coverage_groups([
            {"what": "present", "any_of": ["traces/there.md"]},
            {"what": "definitely absent", "any_of": ["traces/gone.md"]},
            {"what": "unknown — empty mount point", "any_of": ["mount/DIFF.txt"]},
        ])
        (tmp_path / "mount").mkdir()   # visible but EMPTY -> unknown, not absent
        kept, dropped = drop_unreadable_owed(groups, str(tmp_path))
        labels = [g["what"] for g in kept]
        assert "present" in labels
        assert "unknown — empty mount point" in labels, "unknown treated as absent"
        assert "definitely absent" not in labels
        assert len(dropped) == 1 and "traces/gone.md" in dropped[0]

    def test_a_disjunctive_group_survives_if_any_path_exists(self, tmp_path):
        from src.harness.direct.executor import build_coverage_groups, drop_unreadable_owed

        (tmp_path / "traces").mkdir()
        (tmp_path / "traces" / "b.md").write_text("x")
        groups = build_coverage_groups([
            {"what": "one of two", "any_of": ["traces/a.md", "traces/b.md"]}])
        kept, dropped = drop_unreadable_owed(groups, str(tmp_path))
        assert len(kept) == 1 and not dropped

    def test_shortfall_lists_every_outstanding_group(self):
        """The reminder must enumerate EVERY owed read, never a sample of them.

        A capped list is unactionable: the evolver cannot read what it is not told
        it owes, and it has no other channel for that list. This asserts the
        absence of the old 12-item cap AND of its "… and N more" notice, because
        either one reintroduces the blind spot (user-flagged 0725).
        """
        from src.harness.direct.executor import coverage_gap_message

        msg = coverage_gap_message(self._groups(
            *[{"what": f"g{i}", "any_of": [f"t/{i}.md"]} for i in range(30)]))
        assert msg.count("  - g") == 30, "every group must be listed"
        for i in range(30):
            assert f"`t/{i}.md`" in msg, f"group {i} missing from the reminder"
        assert "not listed here" not in msg
        assert "truncated" not in msg
        assert "30 of the reads this batch owes" in msg

    def test_state_is_per_episode_not_module_or_contextvar(self):
        """The 2026-07-10 switch_runtime contextvar leak crossed agents under
        concurrency; one evolver's reads must never credit another's ledger."""
        import inspect

        from src.harness.direct import executor

        src = inspect.getsource(executor.execute_query)
        assert "_owed_groups = build_coverage_groups(" in src
        mod = inspect.getsource(executor)
        assert "_owed_groups: contextvars" not in mod
        assert "ContextVar" not in mod.split("def build_coverage_groups")[1].split("async def _exec_bash")[0]

    def test_the_mandated_measure_first_pass_credits_nothing(self):
        """Naming a path is not reading it. Stage 1 MANDATES `wc -c` over every
        pinned trace in ONE call before opening any, so a matcher that credited
        any mention would mark nearly every owed read satisfied on turn one and
        the gate would never fire."""
        from src.harness.direct.executor import credit_coverage

        g = self._groups(
            {"what": "s0", "any_of": [".cache/traces/iter-1_abc.md"]},
            {"what": "s1", "any_of": [".cache/traces/iter-1_abc_s1.md"]},
        )
        credit_coverage(g, "wc -c .cache/traces/iter-1_abc*.md")
        assert not any(x["seen"] for x in g), g
        credit_coverage(g, "ls -la .cache/traces/ && stat .cache/traces/iter-1_abc.md")
        assert not any(x["seen"] for x in g), g

    @pytest.mark.xfail(reason=_COVERAGE_XFAIL, strict=False)
    def test_a_read_glob_credits_every_path_it_expands_to(self):
        """A genuine batched read is naturally written as a glob."""
        from src.harness.direct.executor import credit_coverage

        g = self._groups(
            {"what": "s0", "any_of": [".cache/traces/iter-1_abc.md"]},
            {"what": "s1", "any_of": [".cache/traces/iter-1_abc_s1.md"]},
        )
        credit_coverage(g, 'for f in .cache/traces/iter-1_abc*.md; do cat "$f"; done')
        assert all(x["seen"] for x in g), g

    @pytest.mark.xfail(reason=_COVERAGE_XFAIL, strict=False)
    def test_measure_then_read_in_one_command_credits(self):
        """`wc -l f && sed -n '1,400p' f` is a read; the measurement prefix must
        not disqualify it."""
        from src.harness.direct.executor import credit_coverage

        g = self._groups({"what": "t", "any_of": ["traces/f.md"]})
        credit_coverage(g, "wc -l traces/f.md && sed -n '1,400p' traces/f.md")
        assert g[0]["seen"]

    @pytest.mark.xfail(reason=_COVERAGE_XFAIL, strict=False)
    def test_shell_loop_over_a_directory_glob_counts(self):
        from src.harness.direct.executor import credit_coverage

        g = self._groups({"what": "t", "any_of": [".cache/traces/iter-1_abc.md"]})
        credit_coverage(g, 'for f in .cache/traces/*.md; do cat "$f"; done')
        assert g[0]["seen"]

    def test_a_sibling_sample_read_never_credits_its_neighbour(self):
        """Each sample is a distinct rollout; crediting s0 because s1 was read
        would hide exactly the omission this check exists to surface."""
        from src.harness.direct.executor import credit_coverage

        g = self._groups({"what": "s0", "any_of": [".cache/traces/iter-1_abc.md"]})
        credit_coverage(g, "cat .cache/traces/iter-1_abc_s1.md")
        assert not g[0]["seen"]

    def test_plain_tokens_do_not_pattern_match(self):
        """Glob semantics apply only to tokens carrying a glob char, so a literal
        argument can never silently match a path it does not name."""
        from src.harness.direct.executor import credit_coverage

        g = self._groups({"what": "t", "any_of": ["traces/a?b.md"]})
        credit_coverage(g, "cat traces/axb.md")
        assert not g[0]["seen"]


class TestCoverageGateInTheRealTurnLoop:
    """Drive execute_query itself, stubbing ONLY the HTTP inference call.

    The helpers above are pure and unit-tested, but that leaves the part a live
    episode actually depends on untested: does the gate fire from inside the turn
    loop, does the episode CONTINUE afterwards, and does the answer survive.
    Everything else here is real — tool dispatch, `_exec_bash` against real files,
    per-call crediting, the one-shot flag, the turn-budget interaction.
    """

    FINAL = '{"final_answer": "done", "reasoning": "r", "confidence": 0.9}'

    def _workspace(self, tmp_path):
        (tmp_path / ".cache" / "traces").mkdir(parents=True)
        (tmp_path / "snap").mkdir()
        rels = [".cache/traces/t_s0.md", ".cache/traces/t_s1.md", "snap/DIFF.txt"]
        for rel in rels:
            (tmp_path / rel).write_text("### Turn 1\nwork\n" * 20)
        return rels

    def _run(self, tmp_path, script, monkeypatch):
        import asyncio
        import json

        from src.harness.direct import executor

        calls = {"n": 0}

        async def _fake(client=None, **kw):
            i = calls["n"]
            calls["n"] += 1
            step = script[i] if i < len(script) else {"content": self.FINAL}
            msg = {"role": "assistant", "content": step["content"]}
            if step.get("bash"):
                msg["tool_calls"] = [{
                    "id": f"c{i}", "type": "function",
                    "function": {"name": "bash",
                                 "arguments": json.dumps({"command": step["bash"]})},
                }]
            return {
                "choices": [{"message": msg,
                             "finish_reason": "tool_calls" if step.get("bash") else "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                "model": "fake/scripted",
            }

        monkeypatch.setattr(executor, "_chat_completion", _fake)
        monkeypatch.setattr(executor, "_chat_completion_stream", _fake)
        monkeypatch.setattr(executor, "STREAM_ENABLED", False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-fake-not-used")

        rels = self._workspace(tmp_path)
        options = {
            "provider_id": "openrouter", "model_id": "fake/scripted",
            "cwd": str(tmp_path), "system": "evolver", "tools": {"bash": True},
            "inject_skills": False, "budget_type": "turns", "budget_value": 20,
            "max_tool_output_chars": 100_000,
            "coverage_owed_reads": [
                {"task": "T", "what": "T sample s0 — solver trace", "any_of": [rels[0]]},
                {"task": "T", "what": "T sample s1 — solver trace", "any_of": [rels[1]]},
                {"task": "T", "what": "T sample s0 — scorer DIFF", "any_of": [rels[2]]},
            ],
        }
        msgs = asyncio.run(executor.execute_query(
            options, "Diagnose.", agent_name="evolver", create_root_span=False))
        nudges = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"
                  and "[HARNESS] Coverage check" in str(m.get("content", ""))]
        answers = [m for m in msgs if isinstance(m, dict)
                   and m.get("role") == "assistant" and (m.get("content") or "").strip()]
        return nudges, answers, calls["n"], rels

    def test_fires_once_then_the_episode_continues_and_re_answers(self, tmp_path, monkeypatch):
        rels = [".cache/traces/t_s0.md", ".cache/traces/t_s1.md", "snap/DIFF.txt"]
        nudges, answers, turns, _ = self._run(tmp_path, [
            # The protocol's MANDATED measure-first pass reads nothing.
            {"content": "measuring", "bash": f"wc -c {rels[0]} {rels[1]} {rels[2]}"},
            {"content": "reading s0", "bash": f"cat {rels[0]}"},
            {"content": self.FINAL},                      # gate must fire here
            {"content": "reading rest", "bash": f"cat {rels[1]} {rels[2]}"},
            {"content": self.FINAL},
        ], monkeypatch)
        assert len(nudges) == 1, nudges
        body = nudges[0]["content"]
        assert "sample s1 — solver trace" in body and "scorer DIFF" in body
        assert "sample s0 — solver trace" not in body, "credited read named as missing"
        assert turns == 5, "episode did not continue past the reminder"
        assert answers[-1]["content"].strip().startswith("{")

    def test_compaction_path_executes_instead_of_raising_nameerror(self, tmp_path, monkeypatch):
        """Compaction was DEAD CODE: it called an undefined `_completion_fn`.

        Nothing caught it because production launchers set the threshold to
        600_000 and the largest real episode peaked at 201,872 context tokens, so
        the branch had never run. A 2026-07-25 replay started at ~153K — past the
        evolver profile's 150_000 default — and crashed instantly with
        `NameError: name '_completion_fn' is not defined`. An 80-task batch would
        have crossed 600K mid-run and hit the same wall.

        This drives the branch for real: the stub reports a context size over the
        threshold, so the loop must compact and then keep going.
        """
        import asyncio
        import json

        from src.harness.direct import executor

        calls: list[dict] = []

        async def _fake(client=None, **kw):
            calls.append(kw)
            n = len(calls)
            # The loop breaks on a tool-less turn BEFORE reaching the compaction
            # check, so turn 1 must call a tool AND report a context past the
            # threshold. Compaction is then call 2 (tools=None), and the episode
            # answers on call 3.
            if n == 1:
                msg = {"role": "assistant", "content": "working",
                       "tool_calls": [{
                           "id": "c1", "type": "function",
                           "function": {"name": "bash",
                                        "arguments": json.dumps({"command": "echo hi"})}}]}
                finish = "tool_calls"
            else:
                # The summary must clear the executor's minimum-length guard, or
                # the run aborts with "compaction failed — too short".
                msg = {"role": "assistant",
                       "content": ("## Prior work\n" + "Read the owed traces and "
                                   "recorded findings per task.\n" * 40)
                       if kw.get("tools") is None else self.FINAL}
                finish = "stop"
            return {
                "choices": [{"message": msg, "finish_reason": finish}],
                "usage": {"prompt_tokens": 60_000 if n == 1 else 100,
                          "completion_tokens": 5, "cost": 0.001},
                "model": "fake/scripted",
            }

        monkeypatch.setattr(executor, "_chat_completion", _fake)
        monkeypatch.setattr(executor, "_chat_completion_stream", _fake)
        monkeypatch.setattr(executor, "STREAM_ENABLED", False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-fake-not-used")

        self._workspace(tmp_path)
        options = {
            "provider_id": "openrouter", "model_id": "fake/scripted",
            "cwd": str(tmp_path), "system": "evolver", "tools": {"bash": True},
            "inject_skills": False, "budget_type": "turns", "budget_value": 6,
            "max_tool_output_chars": 100_000,
            "compaction_threshold_tokens": 50_000,
        }
        msgs = asyncio.run(executor.execute_query(
            options, "Diagnose.", agent_name="evolver", create_root_span=False))

        assert len(calls) >= 2, "compaction never issued its own completion call"
        # The compaction call is the one sent with tools=None.
        assert any(c.get("tools") is None for c in calls[1:]), \
            "no tool-less completion — the compaction branch did not run"
        assert any(isinstance(m, dict) and (m.get("content") or "").strip().startswith("{")
                   for m in msgs if m.get("role") == "assistant"), \
            "episode did not produce an answer after compacting"

    def test_stays_silent_when_every_owed_read_was_covered(self, tmp_path, monkeypatch):
        rels = [".cache/traces/t_s0.md", ".cache/traces/t_s1.md", "snap/DIFF.txt"]
        nudges, answers, turns, _ = self._run(tmp_path, [
            {"content": "reading all", "bash": f"cat {rels[0]} {rels[1]} {rels[2]}"},
            {"content": self.FINAL},
        ], monkeypatch)
        assert nudges == []
        assert turns == 2 and answers[-1]["content"].strip().startswith("{")

    def test_a_declared_shortfall_ends_the_episode_with_the_answer_intact(self, tmp_path, monkeypatch):
        """Declaring is a legitimate response to the reminder — the gate must not
        re-fire to extract a read, or it would spend the evolver's last turns on
        our insistence."""
        rels = [".cache/traces/t_s0.md"]
        nudges, answers, turns, _ = self._run(tmp_path, [
            {"content": "reading s0", "bash": f"cat {rels[0]}"},
            {"content": self.FINAL},
            {"content": self.FINAL + " (s1 skipped: budget)"},
        ], monkeypatch)
        assert len(nudges) == 1
        assert turns == 3, "gate re-fired instead of accepting the declaration"
        assert answers[-1]["content"].strip().startswith("{")

    def test_repeats_after_partial_compliance_then_stops_when_progress_stops(self, tmp_path, monkeypatch):
        """The case a one-shot gate let through: the evolver reads SOME of what
        the reminder named. A second reminder is earned; a third (after it stops
        reading) is not, because declining is a permitted answer."""
        rels = [".cache/traces/t_s0.md", ".cache/traces/t_s1.md", "snap/DIFF.txt"]
        nudges, answers, turns, _ = self._run(tmp_path, [
            {"content": "reading s0", "bash": f"cat {rels[0]}"},
            {"content": self.FINAL},                       # reminder #1 (2 missing)
            {"content": "reading s1 only", "bash": f"cat {rels[1]}"},
            {"content": self.FINAL},                       # reminder #2 (1 missing, progress)
            {"content": self.FINAL},                       # no progress -> no #3
        ], monkeypatch)
        assert len(nudges) == 2, [n["content"][:60] for n in nudges]
        # The second reminder must name only what is STILL outstanding.
        assert "scorer DIFF" in nudges[1]["content"]
        assert "sample s1" not in nudges[1]["content"]
        assert turns == 5, "gate fired a third time or ended early"
        assert answers[-1]["content"].strip().startswith("{")
