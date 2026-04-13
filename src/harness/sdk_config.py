"""SDK configuration and selection logic."""

from typing import Literal

SDKType = Literal["claude", "opencode", "codex", "goose", "openhands"]

# Global SDK selection (can be overridden via CLI arguments)
_current_sdk: SDKType = "claude"

_VALID_SDKS = ("claude", "opencode", "codex", "goose", "openhands")


def set_sdk(sdk: SDKType) -> None:
    """Set the current SDK to use globally."""
    global _current_sdk
    if sdk not in _VALID_SDKS:
        raise ValueError(
            f"Invalid SDK type: {sdk}. Must be one of: {', '.join(repr(s) for s in _VALID_SDKS)}"
        )
    _current_sdk = sdk


def get_sdk() -> SDKType:
    """Get the currently configured SDK."""
    return _current_sdk


def is_claude_sdk() -> bool:
    """Check if claude-agent-sdk is the current SDK."""
    return _current_sdk == "claude"


def is_opencode_sdk() -> bool:
    """Check if opencode-ai is the current SDK."""
    return _current_sdk == "opencode"


def is_openhands_sdk() -> bool:
    """Check if OpenHands is the current SDK."""
    return _current_sdk == "openhands"
  
  
def is_codex_sdk() -> bool:
    """Check if codex is the current SDK."""
    return _current_sdk == "codex"


def is_goose_sdk() -> bool:
    """Check if goose is the current SDK."""
    return _current_sdk == "goose"
