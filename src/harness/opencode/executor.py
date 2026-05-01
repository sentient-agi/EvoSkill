"""OpenCode harness — server management, query execution, response parsing.

Uses raw httpx to talk to the opencode server (the Python SDK sends
model/provider as flat fields which the server ignores; the correct
format is a nested ``model: {providerID, modelID}`` object).
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Type

import httpx
from pydantic import BaseModel, ValidationError

from ..provider_auth import (
    PROVIDER_ENV_KEYS,
    apply_provider_auth_env,
    ensure_provider_api_key,
)

# ── module-level state ────────────────────────────────────────────────
_SERVER_PORTS: dict[str, int] = {}
_SERVER_PIDS: dict[str, int] = {}
_SPAWNED_THIS_RUN: set[str] = set()

_TIMEOUT = 1800  # per-request HTTP timeout (30 min — opencode agents can take 15+ min on complex queries)


# ── provider auth ─────────────────────────────────────────────────────

def _push_provider_auth(base_url: str) -> None:
    """Push all available API keys from env into the opencode server's auth store.

    The opencode server reads credentials from its own auth store, not env vars.
    This syncs any provider keys found in the environment so the server can
    authenticate regardless of which provider the user configures.
    """
    for provider, env_vars in PROVIDER_ENV_KEYS.items():
        for var in env_vars:
            key = os.environ.get(var)
            if key:
                try:
                    httpx.put(
                        f"{base_url}/auth/{provider}",
                        json={"type": "api", "key": key},
                        timeout=5,
                    )
                except Exception:
                    pass
                break


# ── server lifecycle ──────────────────────────────────────────────────

def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _resolve_key(cwd: str | Path | None) -> str:
    return str(Path(cwd).resolve()) if cwd else ""


def _kill_pid(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    for _ in range(10):
        time.sleep(0.1)
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
    try:
        os.kill(pid, signal.SIGKILL)
    except Exception:
        pass


def _kill_all_opencode_servers() -> None:
    """Kill all opencode serve processes on this machine."""
    try:
        subprocess.run(
            ["pkill", "-f", "opencode serve"],
            capture_output=True, timeout=5,
        )
    except Exception:
        pass


def shutdown_project_server(project_root: str | Path | None) -> None:
    key = _resolve_key(project_root)
    pid = _SERVER_PIDS.pop(key, None)
    if pid is not None:
        _kill_pid(pid)
    _SERVER_PORTS.pop(key, None)
    _SPAWNED_THIS_RUN.discard(key)


def shutdown_all_servers() -> None:
    for key in list(set(_SERVER_PORTS) | set(_SERVER_PIDS) | set(_SPAWNED_THIS_RUN)):
        shutdown_project_server(key)


def _wait_for_port(port: int, timeout: float = 15) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.5)


def _ensure_server(options: dict[str, Any]) -> str:
    """Return ``http://127.0.0.1:<port>`` for a running opencode server.

    First call per project per process kills ALL stale opencode servers,
    then spawns a fresh one. Subsequent calls reuse the same server.
    """
    key = _resolve_key(options.get("cwd"))

    if key in _SPAWNED_THIS_RUN:
        port = _SERVER_PORTS.get(key)
        if port is not None:
            return f"http://127.0.0.1:{port}"

    # kill ALL opencode servers from previous runs
    _kill_all_opencode_servers()
    time.sleep(0.5)

    port = _find_free_port()
    env = dict(os.environ)
    apply_provider_auth_env(options.get("provider_id"), env)

    proc = subprocess.Popen(
        ["opencode", "serve", "--port", str(port), "--hostname", "127.0.0.1"],
        cwd=options.get("cwd"),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    _SERVER_PORTS[key] = port
    _SERVER_PIDS[key] = proc.pid
    _SPAWNED_THIS_RUN.add(key)
    _wait_for_port(port)

    base_url = f"http://127.0.0.1:{port}"
    _push_provider_auth(base_url)
    return base_url


# ── query execution ──────────────────────────────────────────────────

async def execute_query(options: dict[str, Any], query: str) -> list[Any]:
    if not isinstance(options, dict):
        raise TypeError(f"OpenCode executor requires dict options, got {type(options)}")

    ensure_provider_api_key(options.get("provider_id"))
    base_url = _ensure_server(options)

    async with httpx.AsyncClient(base_url=base_url, timeout=_TIMEOUT) as client:
        # 1. create session
        r = await client.post("/session", json={})
        r.raise_for_status()
        session_id = r.json()["id"]

        # 2. send message (nested model object — required by the server)
        body: dict[str, Any] = {
            "parts": [{"type": "text", "text": query}],
            "model": {
                "providerID": options.get("provider_id", "anthropic"),
                "modelID": options.get("model_id", "claude-sonnet-4-6"),
            },
        }
        if options.get("system"):
            body["system"] = options["system"]
        if options.get("tools"):
            body["tools"] = options["tools"]
        if options.get("mode"):
            body["mode"] = options["mode"]
        if options.get("format"):
            body["format"] = options["format"]

        r = await client.post(f"/session/{session_id}/message", json=body)
        r.raise_for_status()
        chat_info = r.json()

        # 3. fetch full messages (with parts + structured output)
        r = await client.get(f"/session/{session_id}/message")
        r.raise_for_status()
        messages = r.json()

    return [{"session_id": session_id, "chat_info": chat_info, "messages": messages}]


# ── response parsing ─────────────────────────────────────────────────

def parse_response(
    messages: list[Any],
    response_model: Type[BaseModel],
    get_options: Callable[[], Any],
) -> dict[str, Any]:
    payload = messages[0]
    all_msgs: list[dict] = payload.get("messages", [])

    # find last assistant message
    assistant_info: dict = {}
    assistant_parts: list[dict] = []
    for msg in reversed(all_msgs):
        info = msg.get("info", {})
        if info.get("role") == "assistant":
            assistant_info = info
            assistant_parts = msg.get("parts", [])
            break

    # extract text
    result_text = "".join(
        p.get("text", "") for p in assistant_parts if p.get("type") == "text"
    )

    # structured output
    output = None
    parse_error = None
    raw_structured = assistant_info.get("structured")

    if raw_structured is not None:
        try:
            output = response_model.model_validate(raw_structured)
        except (ValidationError, TypeError, ValueError) as e:
            parse_error = f"{type(e).__name__}: {e}"

    # fallback: parse JSON from text (even if structured failed)
    if output is None and result_text.strip():
        text = result_text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            lines = lines[1:] if lines[0].startswith("```") else lines
            lines = lines[:-1] if lines and lines[-1].strip() == "```" else lines
            text = "\n".join(lines).strip()
        try:
            parsed = json.loads(text)
            output = response_model.model_validate(parsed)
            raw_structured = parsed
            parse_error = None
        except (json.JSONDecodeError, ValidationError, TypeError, ValueError) as e:
            if parse_error is None:
                parse_error = f"{type(e).__name__}: {e}"

    if output is None and parse_error is None:
        parse_error = "No structured output returned (context limit likely exceeded)"

    # cost / usage
    cost = assistant_info.get("cost", 0.0) or 0.0
    usage = assistant_info.get("tokens", {}) or {}

    # model / tools from options
    opts = get_options()
    model = opts.get("model", "unknown") if isinstance(opts, dict) else "unknown"
    tools = list(opts.get("tools", {}).keys()) if isinstance(opts, dict) and opts.get("tools") else []

    session_id = payload.get("session_id", "unknown")

    return dict(
        uuid=session_id,
        session_id=session_id,
        model=model,
        tools=tools,
        duration_ms=0,
        total_cost_usd=cost,
        num_turns=1,
        usage=usage,
        result=result_text,
        is_error=parse_error is not None,
        output=output,
        parse_error=parse_error,
        raw_structured_output=raw_structured,
        messages=messages,
    )


atexit.register(shutdown_all_servers)
