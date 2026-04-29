"""Tests for the LLM and Script scorers in make_scorer()."""

from __future__ import annotations

import asyncio
import sys
import types
from dataclasses import dataclass, field

import pytest


# ── Minimal config stubs ────────────────────────────────────────────────────

@dataclass
class FakeScorerConfig:
    type: str = "multi_tolerance"
    rubric: str | None = None
    model: str | None = None
    provider: str | None = None
    command: str | None = None


@dataclass
class FakeProjectConfig:
    scorer: FakeScorerConfig = field(default_factory=FakeScorerConfig)


# ── LLM Scorer Tests ────────────────────────────────────────────────────────

class TestLLMScorer:
    """Tests for make_scorer with type='llm'."""

    def _make_fake_openai(self, response_text: str):
        """Build a fake openai module that returns *response_text*."""
        captured = {}

        class FakeCompletions:
            async def create(self, *, model, max_tokens, messages):
                captured["model"] = model
                captured["messages"] = messages
                return types.SimpleNamespace(
                    choices=[
                        types.SimpleNamespace(
                            message=types.SimpleNamespace(content=response_text)
                        )
                    ]
                )

        class FakeChat:
            completions = FakeCompletions()

        class FakeAsyncOpenAI:
            def __init__(self, **kwargs):
                captured["init_kwargs"] = kwargs
                self.chat = FakeChat()

        return types.SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI), captured

    def test_llm_scorer_happy_path(self, monkeypatch):
        """LLM scorer returns the float the LLM provides."""
        fake_openai, captured = self._make_fake_openai("0.85")
        monkeypatch.setitem(sys.modules, "openai", fake_openai)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        from src.cli.shared import make_scorer

        cfg = FakeProjectConfig(
            scorer=FakeScorerConfig(
                type="llm",
                model="openrouter/openai/gpt-5-mini",
                rubric="Award 1.0 if correct, 0.0 if wrong.",
            )
        )
        scorer = make_scorer(cfg)
        score = scorer("What is 2+2?", "4", "4")

        assert score == 0.85

    def test_llm_scorer_non_numeric_response_logs_warning(self, monkeypatch, caplog):
        """LLM scorer returns 0.0 and logs a warning when response isn't a float."""
        import logging

        fake_openai, _ = self._make_fake_openai("I think the answer is correct")
        monkeypatch.setitem(sys.modules, "openai", fake_openai)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        from src.cli.shared import make_scorer

        cfg = FakeProjectConfig(
            scorer=FakeScorerConfig(
                type="llm",
                model="openrouter/openai/gpt-5-mini",
            )
        )
        scorer = make_scorer(cfg)

        with caplog.at_level(logging.WARNING):
            score = scorer("Q?", "A", "A")

        assert score == 0.0
        assert "could not parse response as float" in caplog.text

    def test_llm_scorer_api_error_logs_error(self, monkeypatch, caplog):
        """LLM scorer returns 0.0 and logs an error on API failures."""
        import logging

        # Build a fake openai that raises on create()
        class FakeCompletions:
            async def create(self, **kwargs):
                raise ConnectionError("network down")

        class FakeChat:
            completions = FakeCompletions()

        class FakeAsyncOpenAI:
            def __init__(self, **kwargs):
                self.chat = FakeChat()

        fake_openai = types.SimpleNamespace(AsyncOpenAI=FakeAsyncOpenAI)
        monkeypatch.setitem(sys.modules, "openai", fake_openai)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        from src.cli.shared import make_scorer

        cfg = FakeProjectConfig(
            scorer=FakeScorerConfig(
                type="llm",
                model="openrouter/openai/gpt-5-mini",
            )
        )
        scorer = make_scorer(cfg)

        with caplog.at_level(logging.ERROR):
            score = scorer("Q?", "A", "A")

        assert score == 0.0
        assert "LLM scorer failed" in caplog.text
        assert "network down" in caplog.text

    def test_llm_scorer_defaults(self, monkeypatch):
        """LLM scorer falls back to default rubric, model, and provider."""
        # Stub anthropic for the default provider path
        captured = {}

        class FakeResponse:
            content = [types.SimpleNamespace(text="1.0")]

        class FakeMessages:
            async def create(self, **kwargs):
                captured["kwargs"] = kwargs
                return FakeResponse()

        class FakeAsyncAnthropic:
            def __init__(self, *, api_key=None):
                captured["api_key"] = api_key
                self.messages = FakeMessages()

        fake_anthropic = types.SimpleNamespace(AsyncAnthropic=FakeAsyncAnthropic)
        monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")

        from src.cli.shared import make_scorer

        cfg = FakeProjectConfig(
            scorer=FakeScorerConfig(type="llm")
            # model, rubric, provider all None → defaults
        )
        scorer = make_scorer(cfg)
        score = scorer("Q?", "A", "A")

        assert score == 1.0
        assert captured["api_key"] == "test-anthropic-key"

    def test_llm_scorer_works_inside_running_event_loop(self, monkeypatch):
        """LLM scorer works when called from inside a running event loop."""
        fake_openai, _ = self._make_fake_openai("1.0")
        monkeypatch.setitem(sys.modules, "openai", fake_openai)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

        from src.cli.shared import make_scorer

        cfg = FakeProjectConfig(
            scorer=FakeScorerConfig(
                type="llm",
                model="openrouter/openai/gpt-5-mini",
            )
        )
        scorer = make_scorer(cfg)

        # Simulate calling the scorer inside a running event loop
        # (as happens in SelfImprovingLoop._evaluate_train_samples)
        async def call_from_async():
            return scorer("Q?", "A", "A")

        score = asyncio.run(call_from_async())
        assert score == 1.0


# ── Script Scorer Tests ──────────────────────────────────────────────────────

class TestScriptScorer:
    """Tests for make_scorer with type='script'."""

    def test_script_scorer_happy_path(self):
        """Script that echoes 1.0 for matching answers."""
        from src.cli.shared import make_scorer

        cfg = FakeProjectConfig(
            scorer=FakeScorerConfig(
                type="script",
                command="echo 1.0",
            )
        )
        scorer = make_scorer(cfg)
        score = scorer("Q?", "correct answer", "correct answer")

        assert score == 1.0

    def test_script_scorer_uses_format_placeholders(self):
        """Script scorer substitutes {predicted} and {expected} into the command."""
        from src.cli.shared import make_scorer

        # Use python -c to compare the two values
        cfg = FakeProjectConfig(
            scorer=FakeScorerConfig(
                type="script",
                command='python3 -c "print(1.0 if \'{predicted}\' == \'{expected}\' else 0.0)"',
            )
        )
        scorer = make_scorer(cfg)

        assert scorer("Q?", "hello", "hello") == 1.0
        assert scorer("Q?", "hello", "world") == 0.0

    def test_script_scorer_non_numeric_output(self):
        """Script scorer returns 0.0 when command output isn't a valid float."""
        from src.cli.shared import make_scorer

        cfg = FakeProjectConfig(
            scorer=FakeScorerConfig(
                type="script",
                command="echo not_a_number",
            )
        )
        scorer = make_scorer(cfg)
        score = scorer("Q?", "A", "A")

        assert score == 0.0

    def test_script_scorer_command_failure(self):
        """Script scorer returns 0.0 when the command fails (non-zero exit)."""
        from src.cli.shared import make_scorer

        cfg = FakeProjectConfig(
            scorer=FakeScorerConfig(
                type="script",
                command="false",  # always exits 1, empty stdout
            )
        )
        scorer = make_scorer(cfg)
        score = scorer("Q?", "A", "A")

        assert score == 0.0

    def test_script_scorer_none_command_raises_at_creation(self):
        """Script scorer raises ValueError at creation time if command is None."""
        from src.cli.shared import make_scorer

        cfg = FakeProjectConfig(
            scorer=FakeScorerConfig(
                type="script",
                command=None,
            )
        )

        with pytest.raises(ValueError, match="scorer.command is not set"):
            make_scorer(cfg)

    def test_script_scorer_unknown_placeholder_left_as_literal(self):
        """Unknown placeholders like {score} are left as literal text (not expanded)."""
        from src.cli.shared import make_scorer

        # {score} is not a recognized placeholder — str.replace won't touch it
        cfg = FakeProjectConfig(
            scorer=FakeScorerConfig(
                type="script",
                command="echo {predicted} {score}",
            )
        )
        scorer = make_scorer(cfg)
        # {score} is left as the literal string "{score}" in the command
        score = scorer("Q?", "hello", "hello")
        assert score == 0.0  # "hello {score}" is not a valid float

    def test_script_scorer_braces_in_answer(self):
        """Curly braces in answers don't crash the scorer (no str.format)."""
        from src.cli.shared import make_scorer

        cfg = FakeProjectConfig(
            scorer=FakeScorerConfig(
                type="script",
                command="echo 1.0",
            )
        )
        scorer = make_scorer(cfg)
        # Previously this could trigger KeyError/ValueError via str.format()
        score = scorer("Q?", '{"key": "value"}', '{"key": "value"}')
        assert score == 1.0

    def test_script_scorer_timeout(self):
        """Script scorer returns 0.0 when the command hangs past the timeout."""
        from src.cli.shared import make_scorer

        cfg = FakeProjectConfig(
            scorer=FakeScorerConfig(
                type="script",
                command="sleep 60",
            )
        )
        scorer = make_scorer(cfg)
        score = scorer("Q?", "A", "A")
        assert score == 0.0

    def test_script_scorer_shell_metachar_in_answer(self):
        """Script scorer may misbehave with shell metacharacters in answers."""
        from src.cli.shared import make_scorer

        # Since subprocess.run uses a list (not shell=True), this should be safe
        # but .format() happens before shlex.split, so quotes can break tokenization
        cfg = FakeProjectConfig(
            scorer=FakeScorerConfig(
                type="script",
                command='python3 -c "print(1.0 if \'{predicted}\' == \'{expected}\' else 0.0)"',
            )
        )
        scorer = make_scorer(cfg)

        # Answer with a single quote breaks the command template
        score = scorer("Q?", "it's correct", "it's correct")
        # This may produce unexpected results or crash depending on the shell
        # The key point: the scorer doesn't sanitize inputs
        assert isinstance(score, float)
