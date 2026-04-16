"""Claude SDK execution and response parsing.

Handles Claude-specific logic:
    - Converting dict options to ClaudeAgentOptions (fallback)
    - Spawning ClaudeSDKClient and streaming messages with live OTel spans
    - Printing turn-by-turn progress to stdout
    - Parsing SystemMessage/ResultMessage into AgentTrace fields
"""

from __future__ import annotations

import json
from typing import Any, Type

from opentelemetry import trace as otel_trace
from pydantic import BaseModel, ValidationError


_tracer = otel_trace.get_tracer("evoskill.harness.claude")


def _preview(text: Any, n: int = 80) -> str:
    """Safe one-line preview of a block for terminal/span display."""
    s = str(text).strip().replace("\n", " ")
    return s[:n] + "…" if len(s) > n else s


async def execute_query(
    options: Any,
    query: str,
    *,
    agent_name: str = "agent",
) -> list[Any]:
    """Execute a query via Claude SDK, streaming messages with live OTel spans.

    Args:
        options: ClaudeAgentOptions or dict (auto-converted)
        query: The question to send to the agent
        agent_name: Name prefix for spans and console logs (e.g., "base", "skill_evolver")

    Returns:
        List of messages: [SystemMessage, ..., ResultMessage]
    """
    from claude_agent_sdk import (
        ClaudeAgentOptions, ClaudeSDKClient,
        AssistantMessage, ToolUseBlock, TextBlock,
    )

    # If someone passed a dict to Claude SDK (e.g., from config_to_options),
    # convert it to ClaudeAgentOptions.
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

    messages: list[Any] = []
    model_display = getattr(options, "model", None) or "claude"
    turn_num = 0

    with _tracer.start_as_current_span(f"agent.run:{agent_name}") as run_span:
        run_span.set_attribute("agent.name", agent_name)
        run_span.set_attribute("agent.query_preview", _preview(query, 200))

        async with ClaudeSDKClient(options) as client:
            await client.query(query)

            async for msg in client.receive_response():
                messages.append(msg)

                # Only AssistantMessages contain turn-worthy content
                if isinstance(msg, AssistantMessage):
                    turn_num += 1
                    turn_span = _tracer.start_span(f"{agent_name}/turn.{turn_num}")
                    turn_span.set_attribute("turn", turn_num)
                    turn_span.set_attribute("model", model_display)

                    try:
                        for block in msg.content:
                            if isinstance(block, ToolUseBlock):
                                # Tool use — emit a child span + print
                                tool_span = _tracer.start_span(
                                    f"{agent_name}/tool.{block.name}"
                                )
                                tool_span.set_attribute("tool.name", block.name)
                                try:
                                    input_preview = _preview(json.dumps(block.input), 300)
                                    tool_span.set_attribute("tool.input_preview", input_preview)
                                except Exception:
                                    pass
                                tool_span.end()
                                print(f"      turn.{turn_num} [{model_display}]: {block.name}", flush=True)
                            elif isinstance(block, TextBlock):
                                text = (block.text or "").strip()
                                if text:
                                    print(
                                        f"      turn.{turn_num} [{model_display}]: {_preview(text, 80)}",
                                        flush=True,
                                    )
                                    turn_span.set_attribute("text_preview", _preview(text, 300))
                    finally:
                        turn_span.end()

        run_span.set_attribute("num_turns", turn_num)

    return messages


def parse_response(
    messages: list[Any],
    response_model: Type[BaseModel],
) -> dict[str, Any]:
    """Parse Claude SDK messages into AgentTrace field values."""
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
