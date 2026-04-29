"""Provider-specific auth helpers for harness runtimes."""

from __future__ import annotations

import os


PROVIDER_ENV_KEYS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "openrouter": ("OPENROUTER_API_KEY", "LLM_API_KEY"),
    "groq": ("GROQ_API_KEY",),
    "mistral": ("MISTRAL_API_KEY",),
    "together": ("TOGETHER_API_KEY",),
    "deepseek": ("DEEPSEEK_API_KEY",),
    "xai": ("XAI_API_KEY",),
}


def normalize_provider(provider: str | None) -> str:
    return str(provider or "").strip().lower()


def resolve_provider_api_key(provider: str | None) -> tuple[str | None, str | None]:
    """Return the configured API key for a provider and its source env var."""
    normalized = normalize_provider(provider)
    env_names = PROVIDER_ENV_KEYS.get(normalized)
    if not env_names:
        return None, None

    for env_name in env_names:
        value = os.environ.get(env_name)
        if value:
            return value, env_name
    return None, None


def ensure_provider_api_key(provider: str | None) -> str:
    """Validate that the selected provider has a configured API key."""
    normalized = normalize_provider(provider)
    env_names = PROVIDER_ENV_KEYS.get(normalized)
    if not env_names:
        raise ValueError(f"Unknown provider: {provider}")

    value, _source = resolve_provider_api_key(normalized)
    if value:
        return value

    expected = " or ".join(env_names)
    display_provider = "OpenRouter" if normalized == "openrouter" else normalized
    raise RuntimeError(
        f"{display_provider} API key not configured. Set {expected}."
    )


def apply_provider_auth_env(provider: str | None, env: dict[str, str]) -> None:
    """Mirror a provider key into all accepted env names for child processes."""
    normalized = normalize_provider(provider)
    value = ensure_provider_api_key(normalized)
    for env_name in PROVIDER_ENV_KEYS[normalized]:
        env.setdefault(env_name, value)


def resolve_openrouter_api_key() -> tuple[str | None, str | None]:
    """Return the first configured OpenRouter-compatible API key and its source env var."""
    return resolve_provider_api_key("openrouter")


def ensure_openrouter_api_key(provider: str | None) -> str | None:
    """Validate OpenRouter credentials when the selected provider is OpenRouter."""
    if normalize_provider(provider) != "openrouter":
        return None

    return ensure_provider_api_key("openrouter")


def apply_openrouter_env(provider: str | None, env: dict[str, str]) -> None:
    """Mirror the OpenRouter API key into both common env var names for child processes."""
    if normalize_provider(provider) != "openrouter":
        return

    apply_provider_auth_env("openrouter", env)
