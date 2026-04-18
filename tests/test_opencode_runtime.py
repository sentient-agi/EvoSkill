"""Tests for the opencode harness executor (httpx-based, no Python SDK)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import src.harness.opencode.executor as executor
from src.harness.sdk_config import set_sdk
from src.schemas import AgentResponse


@pytest.fixture(autouse=True)
def _reset_sdk():
    set_sdk("claude")
    yield
    set_sdk("claude")


@pytest.fixture(autouse=True)
def _reset_executor_state():
    executor._SERVER_PORTS.clear()
    executor._SERVER_PIDS.clear()
    executor._SPAWNED_THIS_RUN.clear()
    yield
    executor._SERVER_PORTS.clear()
    executor._SERVER_PIDS.clear()
    executor._SPAWNED_THIS_RUN.clear()


def _fake_httpx_response(json_data, status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    return resp


def _make_server_responses():
    """Return (session_create, chat, messages) httpx responses."""
    session = _fake_httpx_response({"id": "ses-1"})
    chat = _fake_httpx_response({
        "info": {
            "role": "assistant",
            "modelID": "minimax/minimax-m2.7",
            "providerID": "openrouter",
            "cost": 0.05,
            "tokens": {"input": 10, "output": 5},
            "structured": {"final_answer": "4", "reasoning": "basic arithmetic"},
        }
    })
    messages = _fake_httpx_response([
        {
            "info": {
                "role": "assistant",
                "cost": 0.05,
                "tokens": {"input": 10, "output": 5},
                "structured": {"final_answer": "4", "reasoning": "basic arithmetic"},
            },
            "parts": [{"type": "text", "text": "4"}],
        }
    ])
    return session, chat, messages


class TestExecuteQuery:
    def test_sends_nested_model_and_parses_structured_output(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ):
        set_sdk("opencode")
        popen_calls = []
        session_resp, chat_resp, msgs_resp = _make_server_responses()

        monkeypatch.setattr(executor, "_find_free_port", lambda: 5555)
        monkeypatch.setattr(executor, "_kill_all_opencode_servers", lambda: None)
        monkeypatch.setattr(executor, "_push_provider_auth", lambda *a: None)
        monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: (popen_calls.append(kw), SimpleNamespace(pid=99))[1])
        monkeypatch.setattr("time.sleep", lambda _: None)
        monkeypatch.setattr(executor, "_wait_for_port", lambda *a, **kw: None)

        async def fake_post(url, **kwargs):
            if url == "/session":
                return session_resp
            if "/message" in url:
                body = kwargs.get("json", {})
                assert "model" in body
                assert body["model"]["providerID"] == "openrouter"
                assert body["model"]["modelID"] == "minimax/minimax-m2.7"
                return chat_resp
            raise ValueError(f"unexpected url: {url}")

        async def fake_get(url, **kwargs):
            return msgs_resp

        mock_client = AsyncMock()
        mock_client.post = fake_post
        mock_client.get = fake_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            options = {
                "system": "Answer questions.",
                "format": {"type": "json_schema", "schema": AgentResponse.model_json_schema()},
                "tools": {"read": True, "bash": True},
                "mode": "build",
                "provider_id": "openrouter",
                "model_id": "minimax/minimax-m2.7",
                "model": "openrouter/minimax/minimax-m2.7",
                "cwd": str(tmp_path),
            }
            result = asyncio.run(executor.execute_query(options, "What is 2+2?"))

        assert popen_calls
        assert popen_calls[0]["cwd"] == str(tmp_path)

        fields = executor.parse_response(result, AgentResponse, lambda: options)
        assert fields["output"] is not None
        assert fields["output"].final_answer == "4"
        assert fields["total_cost_usd"] == 0.05
        assert fields["parse_error"] is None

    def test_reuses_server_on_concurrent_calls(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ):
        set_sdk("opencode")
        popen_count = 0

        def fake_popen(*a, **kw):
            nonlocal popen_count
            popen_count += 1
            return SimpleNamespace(pid=100 + popen_count)

        monkeypatch.setattr(executor, "_find_free_port", lambda: 6666)
        monkeypatch.setattr(executor, "_kill_all_opencode_servers", lambda: None)
        monkeypatch.setattr(executor, "_push_provider_auth", lambda *a: None)
        monkeypatch.setattr("subprocess.Popen", fake_popen)
        monkeypatch.setattr("time.sleep", lambda _: None)
        monkeypatch.setattr(executor, "_wait_for_port", lambda *a, **kw: None)

        options = {"cwd": str(tmp_path), "provider_id": "anthropic", "model_id": "claude-sonnet-4-6"}

        url1 = executor._ensure_server(options)
        assert popen_count == 1

        url2 = executor._ensure_server(options)
        assert popen_count == 1
        assert url1 == url2


class TestShutdown:
    def test_shutdown_project_server_kills_pid(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        key = str(tmp_path.resolve())
        executor._SERVER_PIDS[key] = 1234
        executor._SERVER_PORTS[key] = 7777
        executor._SPAWNED_THIS_RUN.add(key)

        kill_calls = []

        def fake_kill(pid, sig):
            kill_calls.append((pid, sig))
            if sig == 0:
                raise ProcessLookupError

        monkeypatch.setattr("os.kill", fake_kill)
        monkeypatch.setattr("time.sleep", lambda _: None)

        executor.shutdown_project_server(tmp_path)

        assert (1234, executor.signal.SIGTERM) in kill_calls
        assert key not in executor._SERVER_PIDS
        assert key not in executor._SERVER_PORTS
        assert key not in executor._SPAWNED_THIS_RUN

    def test_shutdown_all_servers(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        key1 = str((tmp_path / "a").resolve())
        key2 = str((tmp_path / "b").resolve())
        executor._SERVER_PIDS[key1] = 111
        executor._SERVER_PIDS[key2] = 222
        executor._SERVER_PORTS[key1] = 8001
        executor._SERVER_PORTS[key2] = 8002
        executor._SPAWNED_THIS_RUN.update({key1, key2})

        killed = []

        def fake_kill(pid, sig):
            killed.append(pid)
            if sig == 0:
                raise ProcessLookupError

        monkeypatch.setattr("os.kill", fake_kill)
        monkeypatch.setattr("time.sleep", lambda _: None)

        executor.shutdown_all_servers()

        assert 111 in killed
        assert 222 in killed
        assert not executor._SERVER_PIDS
        assert not executor._SERVER_PORTS
        assert not executor._SPAWNED_THIS_RUN


class TestParseResponse:
    def test_parse_error_when_no_assistant_message(self):
        payload = {"session_id": "s1", "chat_info": {}, "messages": []}
        fields = executor.parse_response(
            [payload], AgentResponse, lambda: {"model": "test", "tools": {}}
        )
        assert fields["output"] is None
        assert fields["parse_error"] is not None

    def test_text_fallback_when_structured_is_invalid(self):
        payload = {
            "session_id": "s1",
            "chat_info": {},
            "messages": [{
                "info": {
                    "role": "assistant",
                    "structured": {"wrong": "fields"},
                    "cost": 0, "tokens": {},
                },
                "parts": [{"type": "text", "text": '{"final_answer": "7", "reasoning": "fallback"}'}],
            }],
        }
        fields = executor.parse_response(
            [payload], AgentResponse, lambda: {"model": "test", "tools": {}}
        )
        assert fields["output"] is not None
        assert fields["output"].final_answer == "7"
        assert fields["parse_error"] is None
