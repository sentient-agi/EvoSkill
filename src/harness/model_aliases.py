from __future__ import annotations

from typing import Literal

HarnessName = Literal["claude", "opencode", "codex", "goose", "openhands"]

DEFAULT_ANTHROPIC_MODEL = "anthropic/claude-sonnet-4-6"
DEFAULT_CODEX_MODEL = "gpt-5.1-codex-mini"

_DEFAULT_MODELS: dict[HarnessName, str] = {
    "claude": DEFAULT_ANTHROPIC_MODEL,
    "opencode": DEFAULT_ANTHROPIC_MODEL,
    "codex": DEFAULT_CODEX_MODEL,
    "goose": DEFAULT_ANTHROPIC_MODEL,
    "openhands": DEFAULT_ANTHROPIC_MODEL,
}


def default_model_for_harness(harness: HarnessName) -> str:
    return _DEFAULT_MODELS[harness]


def normalize_harness_model(harness: HarnessName, model: str | None) -> str:
    if model is None or not model.strip():
        return default_model_for_harness(harness)

    normalized = model.strip()
    if harness == "codex":
        return normalized
    if normalized == "sonnet":
        return DEFAULT_ANTHROPIC_MODEL
    if normalized.startswith("claude-"):
        return f"anthropic/{normalized}"
    return normalized


def strip_model_provider(model: str | None, provider: str) -> str | None:
    if not model:
        return model
    prefix = f"{provider}/"
    if model.startswith(prefix):
        return model[len(prefix):]
    return model
