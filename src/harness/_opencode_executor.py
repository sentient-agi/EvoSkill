"""OpenCode SDK execution, server management, and response parsing.

This module handles all OpenCode-specific logic:
    - Starting/reusing OpenCode servers (per-project port management)
    - Sending queries via AsyncOpencode
    - Parsing AssistantMessage into AgentTrace fields
"""

from __future__ import annotations

import json
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Type, Union

from pydantic import BaseModel, ValidationError

# Tracks which port each project's OpenCode server is running on.
# Key = resolved project directory path, Value = port number.
# Prevents port collisions when running multiple EvoSkill projects simultaneously.
_SERVER_PORTS: dict[str, int] = {}

# HTTP request timeout for OpenCode API calls (10 minutes).
# Some treasury questions require reading large documents and take a while.
_REQUEST_TIMEOUT_SECONDS = 600


# ---------------------------------------------------------------------------
# Server lifecycle helpers
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    """Ask the OS for a random available port by binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def _server_matches_project(client: Any, expected_cwd: str | None) -> bool:
    """Check if a running OpenCode server is serving the expected project directory.

    Returns True if the server's cwd matches expected_cwd, or if expected_cwd is None.
    Returns False if the server is unreachable or serving a different directory.
    """
    app_api = getattr(client, "app", None)
    if app_api is None or not hasattr(app_api, "get"):
        return False

    try:
        app_info = await app_api.get()
    except Exception:
        return False

    if not expected_cwd:
        return True

    # Handle both dict and object response formats from the OpenCode API
    if isinstance(app_info, dict):
        path_info = app_info.get("path", {})
        directory = path_info.get("cwd")
    else:
        path_info = getattr(app_info, "path", None)
        directory = getattr(path_info, "cwd", None)

    if not directory:
        return False

    return Path(directory).resolve() == Path(expected_cwd).resolve()


async def _ensure_server(options: dict[str, Any]) -> Any:
    """Ensure an OpenCode server is running for this project and return a client.

    Steps:
        1. Resolve the project directory from options["cwd"]
        2. Look up existing port for this project, default to 4096
        3. Check if server at that port is serving the right project
        4. If not, find a free port and start a new server
        5. Return an AsyncOpencode client connected to the right port
    """
    from opencode_ai import AsyncOpencode

    requested_cwd = options.get("cwd")
    if requested_cwd:
        requested_cwd = str(Path(requested_cwd).resolve())

    # Look up if we already have a port for this project, default to 4096
    port = _SERVER_PORTS.get(requested_cwd or "", 4096)
    client = AsyncOpencode(
        base_url=f"http://127.0.0.1:{port}",
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )

    # Check if the server at this port is actually serving our project.
    # If not (wrong project, or server not running), start a new one.
    if not await _server_matches_project(client, requested_cwd):
        if requested_cwd:
            port = _SERVER_PORTS.get(requested_cwd, port)
            if port == 4096:
                # Port 4096 might be taken by another project — find a free one
                port = _find_free_port()
                _SERVER_PORTS[requested_cwd] = port

        # Start OpenCode server as a background process
        subprocess.Popen(
            ["opencode", "serve", "--port", str(port), "--hostname", "127.0.0.1"],
            cwd=requested_cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        time.sleep(2)  # Give the server time to start
        client = AsyncOpencode(
            base_url=f"http://127.0.0.1:{port}",
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )

    return client


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------

async def execute_query(options: dict[str, Any], query: str) -> list[Any]:
    """Execute a query via OpenCode SDK.

    Args:
        options: Dict with keys: system, tools, format, model_id, provider_id, cwd, mode
        query: The question to send to the agent

    Returns:
        List containing a single AssistantMessage (wrapped in list for
        consistency with Claude SDK path).
    """
    if not isinstance(options, dict):
        raise TypeError(
            f"OpenCode SDK requires dict options, got {type(options)}"
        )

    client = await _ensure_server(options)

    # Create a session and send the query
    session = await client.session.create(extra_body={})

    # Pass structured output format if configured
    extra_body = {}
    if "format" in options:
        extra_body["format"] = options["format"]

    message = await client.session.chat(
        id=session.id,
        model_id=options.get("model_id", "zai-org/GLM-5"),
        provider_id=options.get("provider_id", "togetherai"),
        parts=[{"type": "text", "text": query}],
        system=options.get("system"),
        mode=options.get("mode", "build"),
        tools=options.get("tools", {}),
        extra_body=extra_body if extra_body else None,
    )

    # Wrap in list for consistency with Claude SDK path
    return [message]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_response(
    messages: list[Any],
    response_model: Type[BaseModel],
    get_options: Callable[[], Any],
) -> dict[str, Any]:
    """Parse OpenCode SDK response into AgentTrace field values.

    OpenCode returns a single AssistantMessage with:
        .info dict → structured output, tokens, cost
        .parts list → text content segments
        .session_id → session identifier

    Args:
        messages: Single-item list containing the AssistantMessage
        response_model: Pydantic model to validate structured output against
        get_options: Callable to retrieve agent options (for model/tools metadata)

    Returns:
        Dict of field values ready to pass to AgentTrace(**fields).
    """
    message = messages[0]

    # Extract structured output — check both key names because
    # different OpenCode versions use different keys
    output = None
    parse_error = None
    raw_structured_output = None

    if hasattr(message, "info") and message.info:
        raw_structured_output = message.info.get("structured_output")
        if raw_structured_output is None:
            raw_structured_output = message.info.get("structured")

    if raw_structured_output is not None:
        try:
            output = response_model.model_validate(raw_structured_output)
        except (ValidationError, json.JSONDecodeError, TypeError) as e:
            parse_error = f"{type(e).__name__}: {str(e)}"
    else:
        parse_error = (
            "No structured output returned (context limit likely exceeded)"
        )

    # Reconstruct the result text from message parts
    result_text = ""
    if hasattr(message, "parts"):
        for part in message.parts:
            if isinstance(part, dict) and part.get("type") == "text":
                result_text += part.get("text", "")

    # Extract cost and token usage from info dict
    info = message.info if hasattr(message, "info") else {}
    usage = info.get("tokens", {}) if info else {}
    cost = info.get("cost", 0.0) if info else 0.0

    # OpenCode doesn't return model/tools in the response,
    # so we pull them from the options we sent
    options = get_options()
    model_name = (
        options.get("model_id", "unknown")
        if isinstance(options, dict)
        else "unknown"
    )
    tools = (
        list(options.get("tools", {}).keys())
        if isinstance(options, dict) and options.get("tools")
        else []
    )

    # Note: duration_ms=0 and num_turns=1 because OpenCode
    # doesn't report these metrics
    return dict(
        uuid=message.session_id or "unknown",
        session_id=message.session_id or "unknown",
        model=model_name,
        tools=tools,
        duration_ms=0,
        total_cost_usd=cost,
        num_turns=1,
        usage=usage,
        result=result_text,
        is_error=parse_error is not None,
        output=output,
        parse_error=parse_error,
        raw_structured_output=raw_structured_output,
        messages=messages,
    )
