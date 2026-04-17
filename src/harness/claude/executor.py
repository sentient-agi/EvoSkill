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
from opentelemetry.trace import set_span_in_context
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
        AssistantMessage, UserMessage, ToolUseBlock, ToolResultBlock, TextBlock,
    )
    # ThinkingBlock may not exist in older SDKs
    try:
        from claude_agent_sdk import ThinkingBlock
    except ImportError:
        ThinkingBlock = None  # type: ignore

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
    # Map tool_use_id -> span so we can attach results from later UserMessages
    pending_tool_spans: dict[str, Any] = {}

    with _tracer.start_as_current_span(f"agent.run:{agent_name}") as run_span:
        run_span.set_attribute("agent.name", agent_name)
        # OpenInference semantic conventions — Phoenix shows these in input/output columns
        run_span.set_attribute("openinference.span.kind", "AGENT")
        run_span.set_attribute("input.value", query)
        run_span.set_attribute("input.mime_type", "text/plain")
        # Keep agent.query for backward compat / structured queries
        run_span.set_attribute("agent.query", query)

        async with ClaudeSDKClient(options) as client:
            await client.query(query)

            async for msg in client.receive_response():
                messages.append(msg)

                # Only AssistantMessages contain turn-worthy content
                if isinstance(msg, AssistantMessage):
                    turn_num += 1
                    # Nest the turn span under the run span explicitly so it's a
                    # direct child (not a sibling of later tool spans).
                    turn_span = _tracer.start_span(
                        f"{agent_name}/turn.{turn_num}",
                        context=set_span_in_context(run_span),
                    )
                    turn_span.set_attribute("openinference.span.kind", "CHAIN")
                    turn_span.set_attribute("turn", turn_num)
                    turn_span.set_attribute("model", model_display)
                    # First turn's input = the original query (so the user can see
                    # the actual prompt when drilling into turn.1)
                    if turn_num == 1:
                        turn_span.set_attribute("input.value", query)
                        turn_span.set_attribute("input.mime_type", "text/plain")
                    turn_ctx = set_span_in_context(turn_span)

                    # Collect a per-turn output summary (text + tool calls made)
                    turn_texts: list[str] = []
                    turn_tool_names: list[str] = []

                    try:
                        for block in msg.content:
                            if isinstance(block, ToolUseBlock):
                                # Tool span nests UNDER the turn span, stays OPEN until
                                # the matching ToolResultBlock arrives in a later UserMessage.
                                tool_span = _tracer.start_span(
                                    f"{agent_name}/tool.{block.name}",
                                    context=turn_ctx,
                                )
                                tool_span.set_attribute("openinference.span.kind", "TOOL")
                                tool_span.set_attribute("tool.name", block.name)
                                try:
                                    tool_input_json = json.dumps(block.input)
                                except Exception:
                                    tool_input_json = str(block.input)
                                tool_span.set_attribute("tool.input", tool_input_json)
                                tool_span.set_attribute("input.value", tool_input_json)
                                tool_span.set_attribute("input.mime_type", "application/json")

                                tool_use_id = getattr(block, "id", None) or getattr(block, "tool_use_id", None)
                                if tool_use_id:
                                    pending_tool_spans[tool_use_id] = tool_span
                                else:
                                    # No id to correlate results — just end the span now
                                    tool_span.end()

                                turn_tool_names.append(block.name)
                                print(f"      turn.{turn_num} [{model_display}]: {block.name}", flush=True)
                            elif isinstance(block, TextBlock):
                                text = (block.text or "").strip()
                                if text:
                                    print(
                                        f"      turn.{turn_num} [{model_display}]: {_preview(text, 80)}",
                                        flush=True,
                                    )
                                    turn_span.set_attribute("text", text)
                                    turn_texts.append(text)
                            elif ThinkingBlock is not None and isinstance(block, ThinkingBlock):
                                thinking = (getattr(block, "thinking", "") or "").strip()
                                if thinking:
                                    print(
                                        f"      turn.{turn_num} [{model_display}] (thinking): {_preview(thinking, 80)}",
                                        flush=True,
                                    )
                                    turn_span.set_attribute("thinking", thinking)
                            else:
                                block_type = type(block).__name__
                                turn_span.set_attribute(
                                    f"unknown_block.{block_type}", str(block)[:1000]
                                )
                                print(
                                    f"      turn.{turn_num} [{model_display}] ({block_type})",
                                    flush=True,
                                )

                        # Compose turn output (what the model produced in this turn)
                        output_parts = []
                        if turn_texts:
                            output_parts.append("\n\n".join(turn_texts))
                        if turn_tool_names:
                            output_parts.append("tool_calls: " + ", ".join(turn_tool_names))
                        if output_parts:
                            turn_span.set_attribute("output.value", "\n\n".join(output_parts))
                            turn_span.set_attribute("output.mime_type", "text/plain")
                    finally:
                        turn_span.end()

                # UserMessages contain ToolResultBlocks — attach to matching tool spans
                elif isinstance(msg, UserMessage):
                    content = getattr(msg, "content", None) or []
                    if isinstance(content, list):
                        for block in content:
                            if isinstance(block, ToolResultBlock):
                                tool_use_id = getattr(block, "tool_use_id", None) or getattr(block, "id", None)
                                tool_span = pending_tool_spans.pop(tool_use_id, None) if tool_use_id else None
                                if tool_span is None:
                                    continue
                                result_content = getattr(block, "content", "")
                                if isinstance(result_content, list):
                                    result_str = "\n".join(
                                        getattr(c, "text", str(c)) for c in result_content
                                    )
                                else:
                                    result_str = str(result_content) if result_content is not None else ""
                                tool_span.set_attribute("tool.output", result_str)
                                tool_span.set_attribute("output.value", result_str)
                                tool_span.set_attribute("output.mime_type", "text/plain")
                                is_error = bool(getattr(block, "is_error", False))
                                tool_span.set_attribute("tool.is_error", is_error)
                                tool_span.end()

        # Close any tool spans that never received a result (defensive)
        for _id, _span in list(pending_tool_spans.items()):
            _span.set_attribute("tool.output", "[no result received]")
            _span.end()
        pending_tool_spans.clear()

        run_span.set_attribute("num_turns", turn_num)
        # Extract final result from ResultMessage (last msg) for output.value
        try:
            last = messages[-1] if messages else None
            if last is not None and hasattr(last, "result") and last.result:
                run_span.set_attribute("output.value", str(last.result))
                run_span.set_attribute("output.mime_type", "text/plain")
            if last is not None and hasattr(last, "total_cost_usd"):
                run_span.set_attribute("agent.total_cost_usd", float(last.total_cost_usd or 0))
        except Exception:
            pass

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
