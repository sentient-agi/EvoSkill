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
import os
import shutil
import tempfile
from types import SimpleNamespace
from typing import Any, Callable, Type

import yaml
from pydantic import BaseModel, ValidationError

from ..provider_auth import apply_provider_auth_env


async def execute_query(options: dict[str, Any], query: str) -> list[Any]:
    """Execute a query by invoking the Goose CLI as an async subprocess.

    Writes a temporary recipe YAML file containing the system prompt and
    output schema, then runs `goose run --recipe <path> -t <query>`.
    The recipe file is cleaned up in a finally block even if execution fails.

    The Goose CLI is imported lazily (checked via shutil.which) so the module
    can be loaded even when goose is not installed. Only fails when you
    actually try to use the Goose harness.

    Args:
        options: Dict built by build_goose_options() with keys:
            - system: system prompt text
            - output_schema: JSON schema dict for structured output
            - provider: provider name (e.g., "anthropic")
            - model: model name (e.g., "claude-sonnet-4-6")
            - working_directory: path where the agent operates
            - tools: tool names (metadata only, not sent to CLI)
            - data_dirs: extra data directory paths
        query: The question/task to send to the agent

    Returns:
        Single-item list containing a SimpleNamespace with .stdout, .stderr,
        and .returncode. Wrapped in a list for consistency with other executors.

    Raises:
        RuntimeError: If the goose CLI is not found on PATH.
    """
    if not shutil.which("goose"):
        raise RuntimeError(
            "Goose CLI not found. Install with: brew install block-goose-cli\n"
            "Requires v1.25.0+ for skill discovery support."
        )

    # Build the recipe YAML structure.
    # NOTE: --recipe and --text/-t CANNOT be combined in Goose CLI.
    # The query is embedded in the recipe's "prompt" field instead.
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

    # Write recipe to a temp file — cleaned up in finally regardless of errors
    fd, recipe_path = tempfile.mkstemp(suffix=".yaml", prefix="goose_recipe_")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(recipe, f)

        # Build environment with provider/model overrides.
        # We copy os.environ so we never mutate the global process environment.
        env = dict(os.environ)
        apply_provider_auth_env(options.get("provider"), env)
        if options.get("provider"):
            env["GOOSE_PROVIDER"] = options["provider"]
        if options.get("model"):
            env["GOOSE_MODEL"] = options["model"]

        # Run goose as an async subprocess so we don't block the event loop.
        # No -t/--text flag — the query is in the recipe's "prompt" field.
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
            # Kill the subprocess if we're cancelled (timeout) or error
            proc.kill()
            await proc.wait()
            raise

        return [SimpleNamespace(
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            returncode=proc.returncode,
        )]
    finally:
        # Always remove the temp recipe file, even if execution raised
        try:
            os.unlink(recipe_path)
        except OSError:
            pass


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
    return dict(
        uuid="unknown",
        session_id="unknown",
        model=model_name,
        tools=tools,
        duration_ms=0,
        total_cost_usd=0.0,
        num_turns=1,
        usage={},
        result=result_text,
        is_error=parse_error is not None,
        output=output,
        parse_error=parse_error,
        raw_structured_output=raw_structured_output,
        messages=messages,
    )
