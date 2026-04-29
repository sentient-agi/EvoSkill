"""Claude SDK execution and response parsing.

This module handles all Claude-specific logic:
    - Converting dict options to ClaudeAgentOptions (fallback)
    - Spawning ClaudeSDKClient and streaming messages
    - Parsing SystemMessage/ResultMessage into AgentTrace fields
"""

from __future__ import annotations

import json
from typing import Any, Type

from pydantic import BaseModel, ValidationError

from ..provider_auth import ensure_provider_api_key


async def execute_query(options: Any, query: str) -> list[Any]:
    """Execute a query via Claude SDK.

    Args:
        options: ClaudeAgentOptions or dict (auto-converted)
        query: The question to send to the agent

    Returns:
        List of messages: [SystemMessage, ..., ResultMessage]
    """
    ensure_provider_api_key("anthropic")

    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

    # If someone passed a dict to Claude SDK (e.g., from config_to_options),
    # convert it to ClaudeAgentOptions. This is a fallback — normally profiles
    # use build_claudecode_options() which returns the right type directly.
    if isinstance(options, dict):
        claude_opts = ClaudeAgentOptions(
            system_prompt=options.get("system"),
            allowed_tools=list(options.get("tools", {}).keys())
            if options.get("tools")
            else [],
            output_format=options.get("format"),
            setting_sources=["user", "project"],
            permission_mode="acceptEdits",
        )
        if "model_id" in options and "claude" in options["model_id"].lower():
            claude_opts.model = options["model_id"]
        options = claude_opts

    # ClaudeSDKClient spawns a Claude Code process, sends the query,
    # and streams back messages (SystemMessage, AssistantMessages, ResultMessage)
    async with ClaudeSDKClient(options) as client:
        await client.query(query)
        return [msg async for msg in client.receive_response()]


def parse_response(
    messages: list[Any],
    response_model: Type[BaseModel],
) -> dict[str, Any]:
    """Parse Claude SDK messages into AgentTrace field values.

    Claude returns: [SystemMessage, ..., ResultMessage]
        - First message has identity info (uuid, model, tools)
        - Last message has results (cost, duration, structured_output)

    Returns:
        Dict of field values ready to pass to AgentTrace(**fields).
    """
    first = messages[0]
    last = messages[-1]

    # Try to parse structured output from the ResultMessage
    output = None
    parse_error = None
    raw_structured_output = last.structured_output

    if raw_structured_output is not None:
        try:
            output = response_model.model_validate(raw_structured_output)
        except (ValidationError, json.JSONDecodeError, TypeError) as e:
            parse_error = f"{type(e).__name__}: {str(e)}"
    else:
        # No structured output usually means the agent hit the context limit
        parse_error = (
            "No structured output returned (context limit likely exceeded)"
        )

    return dict(
        uuid=first.data.get("uuid") or "",
        session_id=last.session_id or "",
        model=first.data.get("model") or "",
        tools=first.data.get("tools") or [],
        duration_ms=last.duration_ms,
        total_cost_usd=last.total_cost_usd,
        num_turns=last.num_turns,
        usage=last.usage,
        result=last.result,
        is_error=last.is_error or parse_error is not None,
        output=output,
        parse_error=parse_error,
        raw_structured_output=raw_structured_output,
        messages=messages,
    )
