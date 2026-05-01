from __future__ import annotations

import asyncio

import pytest


def test_resolve_openrouter_api_key_prefers_provider_specific_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "provider-key")
    monkeypatch.setenv("LLM_API_KEY", "generic-key")

    from src.harness.provider_auth import resolve_openrouter_api_key

    value, source = resolve_openrouter_api_key()

    assert value == "provider-key"
    assert source == "OPENROUTER_API_KEY"


def test_apply_openrouter_env_mirrors_key_into_both_common_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "provider-key")

    from src.harness.provider_auth import apply_openrouter_env

    env: dict[str, str] = {}
    apply_openrouter_env("openrouter", env)

    assert env["OPENROUTER_API_KEY"] == "provider-key"
    assert env["LLM_API_KEY"] == "provider-key"


def test_ensure_provider_api_key_uses_provider_specific_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")

    from src.harness.provider_auth import ensure_provider_api_key

    assert ensure_provider_api_key("anthropic") == "anthropic-key"


def test_apply_provider_auth_env_mirrors_all_accepted_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "google-key")

    from src.harness.provider_auth import apply_provider_auth_env

    env: dict[str, str] = {}
    apply_provider_auth_env("google", env)

    assert env["GOOGLE_API_KEY"] == "google-key"
    assert env["GEMINI_API_KEY"] == "google-key"


def test_ensure_provider_api_key_raises_clear_error_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from src.harness.provider_auth import ensure_provider_api_key

    with pytest.raises(RuntimeError, match="openai API key not configured"):
        ensure_provider_api_key("openai")


def test_ensure_openrouter_api_key_raises_clear_error_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)

    from src.harness.provider_auth import ensure_openrouter_api_key

    with pytest.raises(RuntimeError, match="OpenRouter API key not configured"):
        ensure_openrouter_api_key("openrouter")


def test_claude_harness_requires_anthropic_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    from src.harness.claude.executor import execute_query

    with pytest.raises(RuntimeError, match="anthropic API key not configured"):
        asyncio.run(execute_query({}, "hello"))


def test_codex_harness_requires_openai_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from src.harness.codex.executor import execute_query

    with pytest.raises(RuntimeError, match="openai API key not configured"):
        asyncio.run(execute_query({}, "hello"))
