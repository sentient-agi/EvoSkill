"""Compatibility wrapper around the harness SDK configuration."""

from src.harness.sdk_config import (
    SDKType,
    get_sdk,
    is_claude_sdk,
    is_openhands_sdk,
    is_opencode_sdk,
    set_sdk,
)

__all__ = [
    "SDKType",
    "get_sdk",
    "is_claude_sdk",
    "is_openhands_sdk",
    "is_opencode_sdk",
    "set_sdk",
]
