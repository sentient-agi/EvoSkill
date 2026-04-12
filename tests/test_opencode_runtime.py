from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.agent_profiles.base as base_module
from src.agent_profiles.base import Agent
from src.agent_profiles.sdk_config import set_sdk
from src.schemas import AgentResponse


@pytest.fixture(autouse=True)
def _reset_sdk() -> None:
    set_sdk("claude")
    yield
    set_sdk("claude")


def test_opencode_runtime_uses_options_cwd_and_parses_structured_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    set_sdk("opencode")

    popen_calls: list[dict] = []

    class FakeSessionApi:
        def __init__(self, should_fail_first: bool):
            self._should_fail_first = should_fail_first
            self._create_calls = 0

        async def create(self, *, extra_body=None, **_kwargs):
            self._create_calls += 1
            if self._should_fail_first and self._create_calls == 1:
                raise RuntimeError("server not running")
            return SimpleNamespace(id="session-1")

        async def chat(self, **_kwargs):
            return SimpleNamespace(
                session_id="session-1",
                parts=[{"type": "text", "text": "4"}],
                info={
                    "structured_output": {
                        "final_answer": "4",
                        "reasoning": "basic arithmetic",
                    },
                    "tokens": {"input": 10, "output": 5},
                    "cost": 0.25,
                },
            )

    class FakeAsyncOpencode:
        _instances = 0

        def __init__(self, base_url=None):
            type(self)._instances += 1
            self.base_url = base_url
            self.session = FakeSessionApi(should_fail_first=type(self)._instances == 1)

    monkeypatch.setitem(
        sys.modules,
        "opencode_ai",
        SimpleNamespace(AsyncOpencode=FakeAsyncOpencode),
    )
    monkeypatch.setattr(base_module, "_find_free_port", lambda: 4241, raising=False)

    def fake_popen(args, **kwargs):
        popen_calls.append({"args": args, "kwargs": kwargs})
        return SimpleNamespace(pid=1234)

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    agent = Agent(
        {
            "system": "Answer the question with the final answer only.",
            "format": {
                "type": "json_schema",
                "schema": AgentResponse.model_json_schema(),
            },
            "tools": {
                "read": True,
                "bash": True,
                "edit": True,
                "skill": True,
            },
            "mode": "build",
            "provider_id": "anthropic",
            "model_id": "claude-sonnet-4-6",
            "cwd": str(tmp_path),
        },
        AgentResponse,
    )

    trace = asyncio.run(agent.run("What is 2 + 2?"))

    assert popen_calls
    assert popen_calls[0]["kwargs"]["cwd"] == str(tmp_path)
    assert trace.output is not None
    assert trace.output.final_answer == "4"
    assert trace.output.reasoning == "basic arithmetic"
    assert trace.total_cost_usd == 0.25


def test_opencode_runtime_starts_fresh_server_when_existing_server_points_to_wrong_repo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    set_sdk("opencode")

    wrong_root = tmp_path / "wrong-repo"
    right_root = tmp_path / "right-repo"
    wrong_root.mkdir()
    right_root.mkdir()

    popen_calls: list[dict] = []

    class FakeSessionApi:
        async def create(self, *, extra_body=None, **_kwargs):
            return SimpleNamespace(id="session-1")

        async def chat(self, **_kwargs):
            return SimpleNamespace(
                session_id="session-1",
                parts=[{"type": "text", "text": "4"}],
                info={
                    "structured_output": {
                        "final_answer": "4",
                        "reasoning": "basic arithmetic",
                    },
                    "tokens": {"input": 10, "output": 5},
                    "cost": 0.25,
                },
            )

    class FakePathApi:
        def __init__(self, directory: str):
            self._directory = directory

        async def get(self):
            return {"directory": self._directory}

    class FakeAsyncOpencode:
        def __init__(self, base_url=None):
            self.base_url = base_url
            self.session = FakeSessionApi()
            if base_url == "http://127.0.0.1:4242":
                self.path = FakePathApi(str(right_root))
            else:
                self.path = FakePathApi(str(wrong_root))

    monkeypatch.setitem(
        sys.modules,
        "opencode_ai",
        SimpleNamespace(AsyncOpencode=FakeAsyncOpencode),
    )
    monkeypatch.setattr(base_module, "_find_free_port", lambda: 4242, raising=False)

    def fake_popen(args, **kwargs):
        popen_calls.append({"args": args, "kwargs": kwargs})
        return SimpleNamespace(pid=5678)

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    agent = Agent(
        {
            "system": "Answer the question with the final answer only.",
            "format": {
                "type": "json_schema",
                "schema": AgentResponse.model_json_schema(),
            },
            "tools": {
                "read": True,
                "bash": True,
                "edit": True,
                "skill": True,
            },
            "mode": "build",
            "provider_id": "anthropic",
            "model_id": "claude-sonnet-4-6",
            "cwd": str(right_root),
        },
        AgentResponse,
    )

    trace = asyncio.run(agent.run("What is 2 + 2?"))

    assert popen_calls
    assert popen_calls[0]["kwargs"]["cwd"] == str(right_root)
    assert "4242" in [str(part) for part in popen_calls[0]["args"]]
    assert trace.output is not None
    assert trace.output.final_answer == "4"
