"""Goose harness execution and response parsing.

This module handles all Goose-specific logic:
    - Building a recipe YAML and invoking the Goose CLI as a subprocess
    - Parsing stdout into AgentTrace fields

How Goose differs from Claude, OpenCode, and Codex:

    Claude SDK:
        - Spawns a Claude Code process per query
        - Returns [SystemMessage, ..., ResultMessage]
        - Structured output in ResultMessage.structured_output

    OpenCode SDK:
        - Manages a persistent HTTP server
        - Returns AssistantMessage with info/parts as extra fields
        - Structured output in info["structured_output"]

    Codex SDK:
        - Creates a thread, runs prompts within it
        - Returns a turn object with .final_response (JSON string)

    Goose (subprocess):
        - No Python SDK — invoked as `goose run --recipe <yaml> --output-format json`
        - Query is embedded in the recipe YAML's "prompt" field (--recipe and --text can't be combined)
        - Returns a SimpleNamespace with .stdout, .stderr, .returncode
        - Structured JSON output is the LAST non-empty line of stdout
        - No cost/session/uuid info available (all set to zero/"unknown")
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Callable, Type

import yaml
from opentelemetry import trace as otel_trace
from pydantic import BaseModel, ValidationError

from ..provider_auth import apply_provider_auth_env
from .._tracing import (
    agent_root_span,
    finalise_root_span,
    start_tool_span,
    start_turn_span,
)

logger = logging.getLogger(__name__)
_tracer = otel_trace.get_tracer("evoskill.harness.goose")


async def execute_query(options: dict[str, Any], query: str) -> list[Any]:
    """Execute a query by invoking the Goose CLI as an async subprocess.

    Writes a temporary recipe YAML file containing the system prompt and
    output schema, then runs `goose run --recipe <path> -t <query>`.
    The recipe file is cleaned up in a finally block even if execution fails.

    Args:
        options: Dict built by build_goose_options() with keys:
            - system, output_schema, provider, model, working_directory, tools, data_dirs
        query: The question/task to send to the agent

    Returns:
        Single-item list containing a SimpleNamespace with .stdout, .stderr,
        .returncode, and .duration_ms.

    Raises:
        RuntimeError: If the goose CLI is not found on PATH.
    """
    if not shutil.which("goose"):
        raise RuntimeError(
            "Goose CLI not found. Install with: brew install block-goose-cli\n"
            "Requires v1.25.0+ for skill discovery support."
        )

    recipe: dict[str, Any] = {
        "version": 1,
        "title": "EvoSkill Agent Query",
        "description": "Auto-generated recipe for EvoSkill agent execution",
        "instructions": options.get("system", ""),
        "prompt": query,
        "response": {
            "json_schema": options.get("output_schema", {}),
        },
    }

    model_name = options.get("model", "unknown")

    fd, recipe_path = tempfile.mkstemp(suffix=".yaml", prefix="goose_recipe_")
    try:
      with agent_root_span(_tracer, harness_name="goose", agent_name="goose_agent", query=query) as (run_span, eval_ctx, _session_id):
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(recipe, f)

        env = dict(os.environ)
        apply_provider_auth_env(options.get("provider"), env)
        if options.get("provider"):
            env["GOOSE_PROVIDER"] = options["provider"]
        if options.get("model"):
            env["GOOSE_MODEL"] = options["model"]

        t0 = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            "goose",
            "run",
            "--recipe", recipe_path,
            "--output-format", "json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=options.get("working_directory", "."),
            env=env,
        )
        try:
            stdout_bytes, stderr_bytes = await proc.communicate()
        except (asyncio.CancelledError, Exception):
            proc.kill()
            await proc.wait()
            raise

        duration_ms = int((time.monotonic() - t0) * 1000)
        stdout_str = stdout_bytes.decode("utf-8", errors="replace")
        stderr_str = stderr_bytes.decode("utf-8", errors="replace")

        # Emit child spans from Goose conversation JSON
        _emit_goose_child_spans(run_span, stdout_str, model_name, eval_ctx)

        # Extract result text for root span
        result_text = _extract_goose_final_text(stdout_str)
        system_text = options.get("system", "")
        goose_msgs = _parse_goose_conversation(stdout_str)
        finalise_root_span(
            _tracer, run_span, eval_ctx,
            agent_answer=result_text,
            result_text=result_text,
            model=model_name,
            num_turns=_count_goose_turns(stdout_str),
            system_prompt=system_text,
            messages=goose_msgs,
        )

        return [SimpleNamespace(
            stdout=stdout_str,
            stderr=stderr_str,
            returncode=proc.returncode,
            duration_ms=duration_ms,
        )]
    finally:
        try:
            os.unlink(recipe_path)
        except OSError:
            pass


def _parse_goose_conversation(stdout: str) -> list[dict[str, Any]]:
    """Extract the messages list from Goose JSON output."""
    json_start = stdout.find("{")
    if json_start == -1:
        return []
    try:
        conversation = json.loads(stdout[json_start:])
        return conversation.get("messages", []) if isinstance(conversation, dict) else []
    except json.JSONDecodeError:
        return []


def _extract_goose_final_text(stdout: str) -> str:
    for msg in reversed(_parse_goose_conversation(stdout)):
        if msg.get("role") == "assistant":
            for content in msg.get("content", []):
                if content.get("type") == "text":
                    return content.get("text", "")
    return ""


def _count_goose_turns(stdout: str) -> int:
    return max(1, sum(1 for m in _parse_goose_conversation(stdout) if m.get("role") == "assistant"))


def _emit_goose_child_spans(
    parent: Any,
    stdout: str,
    model: str,
    eval_ctx: Any,
) -> None:
    """Emit turn + tool child spans from Goose conversation JSON."""
    msgs = _parse_goose_conversation(stdout)
    turn_num = 0
    for msg in msgs:
        role = msg.get("role", "")
        if role != "assistant":
            continue
        turn_num += 1
        turn_sp = start_turn_span(
            _tracer, parent,
            turn_num=turn_num, model=model,
            eval_ctx=eval_ctx, agent_name="goose_agent",
        )
        for content in msg.get("content", []):
            ctype = content.get("type", "")
            if ctype == "text":
                turn_sp.set_attribute("llm.output_messages.0.message.content", content.get("text", ""))
            elif ctype == "toolRequest":
                tc = content.get("toolCall", {}).get("value", {})
                tname = tc.get("name", "unknown")
                targs = tc.get("arguments")
                ts = start_tool_span(_tracer, turn_sp, tool_name=tname, tool_input=targs)
                ts.end()
        turn_sp.end()


def parse_response(
    messages: list[Any],
    response_model: Type[BaseModel],
    get_options: Callable[[], Any],
) -> dict[str, Any]:
    """Parse a Goose subprocess result into AgentTrace field values.

    Goose writes structured JSON output as the last non-empty line of stdout.
    All preceding lines are treated as informational log output and are ignored.

    When a non-zero return code is detected, the stderr is captured as the
    parse_error rather than attempting to parse stdout.

    Args:
        messages: Single-item list containing a SimpleNamespace result
                  (from execute_query) with .stdout, .stderr, .returncode.
        response_model: Pydantic model to validate the parsed JSON against
                       (e.g., AgentResponse, SkillProposerResponse)
        get_options: Callable that returns the options dict (for model/tools/provider
                    metadata). Same pattern as Codex and OpenCode executors.

    Returns:
        Dict of field values ready to unpack into AgentTrace(**fields).
    """
    result = messages[0]

    output = None
    parse_error = None
    raw_structured_output = None
    result_text = getattr(result, "stdout", "") or ""

    # Check for subprocess failure first — non-zero exit is always an error
    if getattr(result, "returncode", 0) != 0:
        stderr = getattr(result, "stderr", "")
        parse_error = f"Goose exited with code {result.returncode}: {stderr[:500]}"
    else:
        # Goose --output-format json returns log lines at the top, then a big
        # JSON object containing the full conversation with messages array.
        # The structured output is in the recipe__final_output tool call arguments.
        #
        # Strategy:
        #   1. Find the JSON blob in stdout (skip leading log lines)
        #   2. Parse it and look for recipe__final_output tool call
        #   3. Extract the arguments dict — that's our structured output
        #   4. Fallback: try the last assistant message's text content

        # Find where the JSON starts (first "{" on a line)
        json_start = -1
        for i, char in enumerate(result_text):
            if char == "{":
                json_start = i
                break

        if json_start == -1:
            parse_error = "No JSON found in Goose output"
        else:
            try:
                conversation = json.loads(result_text[json_start:])
            except json.JSONDecodeError as e:
                parse_error = f"JSONDecodeError parsing Goose output: {e}"
                conversation = None

            if conversation and isinstance(conversation, dict):
                messages_list = conversation.get("messages", [])

                # Strategy 1: Find recipe__final_output tool call
                for msg in messages_list:
                    if msg.get("role") != "assistant":
                        continue
                    for content in msg.get("content", []):
                        if content.get("type") == "toolRequest":
                            tool_call = content.get("toolCall", {}).get("value", {})
                            if tool_call.get("name") == "recipe__final_output":
                                raw_structured_output = tool_call.get("arguments", {})
                                break
                    if raw_structured_output:
                        break

                # Strategy 2: Fallback — try last assistant text content as JSON
                if raw_structured_output is None:
                    for msg in reversed(messages_list):
                        if msg.get("role") != "assistant":
                            continue
                        for content in msg.get("content", []):
                            if content.get("type") == "text":
                                try:
                                    raw_structured_output = json.loads(content["text"])
                                except (json.JSONDecodeError, KeyError):
                                    pass
                        if raw_structured_output:
                            break

                # Validate against the Pydantic model
                if raw_structured_output is not None:
                    try:
                        output = response_model.model_validate(raw_structured_output)
                    except (ValidationError, TypeError) as e:
                        parse_error = f"{type(e).__name__}: {str(e)}"
                elif not parse_error:
                    parse_error = "No structured output found in Goose conversation"

    # Get model name, provider, and tools from the options we sent.
    # Goose doesn't return these in the response, so we read from our config.
    options = get_options()
    model_name = options.get("model", "unknown") if isinstance(options, dict) else "unknown"
    tools = options.get("tools", []) if isinstance(options, dict) else []

    # Build the AgentTrace fields dict.
    # Limitations vs other harnesses:
    #   - uuid="unknown": Goose CLI doesn't expose a session UUID
    #   - session_id="unknown": no persistent session tracking
    #   - duration_ms=0: not reported by Goose CLI
    #   - total_cost_usd=0.0: not exposed by Goose CLI
    #   - num_turns=1: Goose runs as a single invocation
    #   - usage={}: token counts not exposed by Goose CLI
    duration_ms = getattr(result, "duration_ms", 0) or 0

    return dict(
        uuid="unknown",
        session_id="unknown",
        model=model_name,
        tools=tools,
        duration_ms=duration_ms,
        total_cost_usd=0.0,
        num_turns=1,
        usage={},
        result=result_text,
        is_error=parse_error is not None,
        output=output,
        parse_error=parse_error,
        raw_structured_output=raw_structured_output,
        messages=messages,
        budget_exhausted="",
    )
