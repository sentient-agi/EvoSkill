"""Harness layer — SDK executors, options builders, and the Agent abstraction.

This package handles HOW to talk to different agent SDKs (Claude, OpenCode,
future Goose/OpenHands). It knows nothing about specific agent roles
(that's agent_profiles/).

Key exports:
    Agent[T]        — generic wrapper that delegates to the active SDK
    AgentTrace[T]   — SDK-agnostic result from an agent run
    set_sdk/get_sdk — global SDK toggle
    build_claudecode_options/build_opencode_options — option builders
"""

from .agent import Agent, AgentTrace, OptionsProvider
from .sdk_config import set_sdk, get_sdk, is_claude_sdk, is_opencode_sdk, is_codex_sdk, is_goose_sdk
from .utils import build_options, resolve_project_root, resolve_data_dirs
from .claude.options import build_claudecode_options
from .opencode.options import build_opencode_options
from .codex.options import build_codex_options
from .goose.options import build_goose_options

__all__ = [
    "Agent",
    "AgentTrace",
    "OptionsProvider",
    "set_sdk",
    "get_sdk",
    "is_claude_sdk",
    "is_opencode_sdk",
    "is_codex_sdk",
    "is_goose_sdk",
    "build_options",
    "build_claudecode_options",
    "build_opencode_options",
    "build_codex_options",
    "build_goose_options",
    "resolve_project_root",
    "resolve_data_dirs",
]
