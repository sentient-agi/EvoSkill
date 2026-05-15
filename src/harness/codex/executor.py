"""Codex SDK execution and response parsing.

This module handles all Codex-specific logic:
    - Creating Codex threads and sending queries
    - Parsing turn results into AgentTrace fields

How Codex differs from Claude and OpenCode:

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
        - Structured output parsed from final_response text
        - No external server to manage (Codex handles its own process)
        - No cost/token reporting (set to 0 in AgentTrace)
"""

from __future__ import annotations

import json
from typing import Any, Callable, Type

from pydantic import BaseModel, ValidationError

from ..provider_auth import ensure_provider_api_key


async def execute_query(options: dict[str, Any], query: str) -> list[Any]:
    """Execute a query via the Codex SDK.

    Creates a new Codex instance, starts a thread with the configured
    working directory, and runs the query with an output schema for
    structured JSON output.

    The Codex SDK is imported lazily (inside this function) so the module
    can be loaded even when openai-codex-sdk is not installed. Only fails
    when you actually try to use the Codex harness.

    Args:
        options: Dict built by build_codex_options() with keys:
            - system: system prompt text
            - output_schema: JSON schema dict for structured output
            - model: model name (e.g., "gpt-5.1-codex-mini")
            - working_directory: path where the agent operates
            - tools: tool names (metadata only, not sent to SDK)
            - data_dirs: extra data directory paths
        query: The question/task to send to the agent

    Returns:
        Single-item list containing the turn result object.
        Wrapped in a list for consistency with Claude ([messages])
        and OpenCode ([message]) executors.
    """
    api_key = ensure_provider_api_key("openai")

    # Lazy import — only fails if someone actually uses the codex harness
    # without installing the SDK. Same pattern as claude/executor.py and
    # opencode/executor.py.
    from openai_codex_sdk import Codex

    if not isinstance(options, dict):
        raise TypeError(f"Codex SDK requires dict options, got {type(options)}")

    # Ensure Codex can discover skills written to .claude/skills/ by
    # symlinking .agents/skills/ -> .claude/skills/ before starting the
    # thread. Codex scans .agents/skills/ but EvoSkill writes to .claude/skills/.
    from pathlib import Path
    from .skill_discovery import ensure_agents_skills_symlink
    ensure_agents_skills_symlink(Path(options.get("working_directory", ".")))

    # Create a new Codex instance and start a thread.
    # Unlike OpenCode (which manages a persistent HTTP server), Codex
    # handles its own process lifecycle internally.
    codex = Codex({"api_key": api_key})
    thread_opts: dict[str, Any] = {
        "working_directory": options.get("working_directory", "."),
    }
    if options.get("model"):
        thread_opts["model"] = options["model"]
    if options.get("data_dirs"):
        thread_opts["additional_directories"] = options["data_dirs"]

    thread = codex.start_thread(thread_opts)

    # Pass the output_schema so Codex constrains the model's response
    # to match our Pydantic schema (e.g., AgentResponse, SkillProposerResponse).
    # This is equivalent to Claude's output_format and OpenCode's format parameter.
    run_opts: dict[str, Any] = {}
    if "output_schema" in options:
        run_opts["output_schema"] = options["output_schema"]

    # Run the query. The Codex SDK returns a turn object with:
    #   .final_response — the model's text output (JSON string when schema is set)
    #   .id — unique turn identifier
    #   .thread_id — the thread this turn belongs to
    #   .items — tool call results (file reads, bash executions, etc.)
    system_prompt = str(options.get("system") or "").strip()
    prompt = f"{system_prompt}\n\n{query}" if system_prompt else query
    turn = await thread.run(prompt, run_opts)

    # Wrap in list for consistency with other executors.
    # Agent.run() always receives list[Any] and passes it to parse_response().
    return [turn]


def parse_response(
    messages: list[Any],
    response_model: Type[BaseModel],
    get_options: Callable[[], Any],
) -> dict[str, Any]:
    """Parse a Codex turn object into AgentTrace field values.

    The Codex SDK returns structured output differently from Claude and OpenCode:
        - Claude: structured output is a dict in ResultMessage.structured_output
        - OpenCode: structured output is a dict in info["structured_output"]
        - Codex: structured output is a JSON STRING in turn.final_response
                 that we need to json.loads() and then validate

    When output_schema is passed to thread.run(), the Codex CLI constrains
    the model to produce valid JSON matching the schema. We parse it here.

    Args:
        messages: Single-item list containing the turn object (from execute_query)
        response_model: Pydantic model to validate the parsed JSON against
                       (e.g., AgentResponse, SkillProposerResponse)
        get_options: Callable that returns the options dict (for model/tools metadata).
                    Same pattern as OpenCode executor.

    Returns:
        Dict of field values ready to unpack into AgentTrace(**fields).
    """
    turn = messages[0]

    output = None
    parse_error = None
    raw_structured_output = None

    # Step 1: Get the text response from the turn object.
    # final_response is a string — either JSON (when output_schema was set)
    # or plain text (when no schema was set).
    result_text = ""
    if hasattr(turn, "final_response") and turn.final_response:
        result_text = turn.final_response

    # Step 2: Try to parse the text as JSON and validate against the schema.
    # This is the main difference from Claude/OpenCode: they get structured
    # output as a pre-parsed dict, while Codex gives us a JSON string.
    if result_text:
        try:
            # Parse the JSON string into a dict
            parsed = json.loads(result_text)
            raw_structured_output = parsed

            # Validate against the Pydantic model (e.g., AgentResponse)
            # This is the same validation step that Claude and OpenCode do.
            output = response_model.model_validate(parsed)
        except json.JSONDecodeError as e:
            # The model returned non-JSON text (shouldn't happen with output_schema,
            # but can happen if the schema wasn't enforced or the model errored)
            parse_error = f"JSONDecodeError: {e}"
        except (ValidationError, TypeError) as e:
            # Valid JSON but doesn't match the expected schema
            # (e.g., missing required fields, wrong types)
            parse_error = f"{type(e).__name__}: {str(e)}"
    else:
        # No response at all — the model didn't produce output.
        # Could be a timeout, context limit, or SDK error.
        parse_error = "No response from Codex (final_response is empty)"

    # Step 3: Get model name and tools from the options we sent.
    # Codex doesn't return these in the response (unlike Claude which puts
    # model in the first message). So we read them from our original config.
    options = get_options()
    model_name = options.get("model", "unknown") if isinstance(options, dict) else "unknown"
    tools = options.get("tools", []) if isinstance(options, dict) else []

    # Step 4: Build the AgentTrace fields dict.
    # Note the limitations vs other harnesses:
    #   - duration_ms=0: Codex SDK doesn't report execution time
    #   - total_cost_usd=0.0: Codex SDK doesn't expose API cost
    #   - num_turns=1: Codex runs as a single turn (no multi-turn count)
    #   - usage={}: Codex SDK doesn't expose token counts
    return dict(
        uuid=getattr(turn, "id", "") or "unknown",
        session_id=getattr(turn, "thread_id", "") or "unknown",
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
