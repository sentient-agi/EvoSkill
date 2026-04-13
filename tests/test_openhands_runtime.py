from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.harness import Agent, set_sdk
from src.schemas import AgentResponse


@pytest.fixture(autouse=True)
def _reset_sdk() -> None:
    set_sdk("claude")
    yield
    set_sdk("claude")


def _install_fake_openhands(
    monkeypatch: pytest.MonkeyPatch,
    *,
    final_message: str,
    extracted_message: str | None = None,
    metrics: dict | None = None,
) -> dict[str, object]:
    captured: dict[str, object] = {}

    class FakeTool:
        def __init__(self, name: str):
            self.name = name

    class FakeAgentContext:
        def __init__(self, *, skills=None, system_message_suffix=None, **_kwargs):
            self.skills = list(skills or [])
            self.system_message_suffix = system_message_suffix

    class FakeLocalWorkspace:
        def __init__(self, working_dir: str):
            self.working_dir = working_dir

    class FakeLLM:
        def __init__(self, **kwargs):
            captured["llm_init"] = kwargs
            self.model = kwargs["model"]
            self.metrics = SimpleNamespace(
                get=lambda: metrics
                or {
                    "accumulated_cost": 0.42,
                    "accumulated_token_usage": {"input_tokens": 10, "output_tokens": 5},
                }
            )
            self.calls: list[dict] = []
            captured["llm"] = self

        def completion(self, **kwargs):
            self.calls.append(kwargs)
            content = extracted_message or '{"final_answer":"4","reasoning":"basic arithmetic"}'
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=content)
                    )
                ]
            )

    class FakeConversation:
        def __init__(self, *, agent, workspace, callbacks=None, **kwargs):
            captured["conversation_init"] = {
                "agent": agent,
                "workspace": workspace,
                "callbacks": callbacks,
                "kwargs": kwargs,
            }
            self.agent = agent
            self.workspace = workspace
            self.callbacks = list(callbacks or [])
            self.messages = []
            self.state = SimpleNamespace(events=[])
            self.conversation_stats = SimpleNamespace(
                usage_to_metrics={"input_tokens": 10, "output_tokens": 5}
            )

        def send_message(self, message, sender=None):
            captured["sent_message"] = {"message": message, "sender": sender}

        def run(self):
            msg = SimpleNamespace(role="assistant", content=final_message)
            self.messages = [msg]
            self.state.events = [msg]
            for callback in self.callbacks:
                callback(msg)

    def fake_get_default_agent(**kwargs):
        captured["agent_kwargs"] = kwargs
        return SimpleNamespace(**kwargs)

    def fake_load_skills_from_dir(path):
        captured["skills_dir"] = str(path)
        return {}, {}, {
            "repo-skill": SimpleNamespace(
                name="repo-skill",
                description="Repo-local skill",
            )
        }

    def fake_register_default_tools(*, enable_browser=False):
        captured["register_default_tools"] = {"enable_browser": enable_browser}

    class FakeTextContent:
        def __init__(self, text: str):
            self.text = text

    class FakeMessage:
        def __init__(self, *, role: str, content):
            self.role = role
            self.content = content
    sdk_module = SimpleNamespace(
        LLM=FakeLLM,
        AgentContext=FakeAgentContext,
        LocalWorkspace=FakeLocalWorkspace,
        Conversation=FakeConversation,
        Tool=FakeTool,
        get_default_agent=fake_get_default_agent,
    )
    monkeypatch.setitem(sys.modules, "openhands.sdk", sdk_module)
    monkeypatch.setitem(
        sys.modules,
        "openhands.sdk.context.skills",
        SimpleNamespace(load_skills_from_dir=fake_load_skills_from_dir),
    )
    monkeypatch.setitem(
        sys.modules,
        "openhands.sdk.llm",
        SimpleNamespace(Message=FakeMessage, TextContent=FakeTextContent),
    )
    monkeypatch.setitem(
        sys.modules,
        "openhands.tools",
        SimpleNamespace(register_default_tools=fake_register_default_tools),
    )
    monkeypatch.setitem(
        sys.modules,
        "openhands.tools.file_editor",
        SimpleNamespace(FileEditorTool=SimpleNamespace(name="file_editor")),
    )
    monkeypatch.setitem(
        sys.modules,
        "openhands.tools.terminal",
        SimpleNamespace(TerminalTool=SimpleNamespace(name="terminal")),
    )
    monkeypatch.setitem(
        sys.modules,
        "openhands.tools.task_tracker",
        SimpleNamespace(TaskTrackerTool=SimpleNamespace(name="task_tracker")),
    )
    return captured


def test_openhands_runtime_uses_direct_json_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    set_sdk("openhands")
    skills_dir = tmp_path / ".claude" / "skills" / "repo-skill"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("---\nname: repo-skill\ndescription: Repo-local skill\n---\n")

    captured = _install_fake_openhands(
        monkeypatch,
        final_message='{"final_answer":"4","reasoning":"basic arithmetic"}',
    )

    agent = Agent(
        {
            "sdk": "openhands",
            "system": "Answer the question with the final answer only.",
            "format": {
                "type": "json_schema",
                "schema": AgentResponse.model_json_schema(),
            },
            "tools": ["Read", "Edit", "Bash", "TodoWrite", "Skill"],
            "provider_id": "anthropic",
            "model_id": "claude-sonnet-4-5-20250929",
            "model": "anthropic/claude-sonnet-4-5-20250929",
            "cwd": str(tmp_path),
            "skills_dir": str(tmp_path / ".claude" / "skills"),
            "add_dirs": [],
        },
        AgentResponse,
    )

    trace = asyncio.run(agent.run("What is 2 + 2?"))

    assert trace.output is not None
    assert trace.output.final_answer == "4"
    assert trace.output.reasoning == "basic arithmetic"
    assert trace.result == '{"final_answer":"4","reasoning":"basic arithmetic"}'
    assert getattr(captured["llm"], "calls") == []
    assert captured["conversation_init"]["workspace"].working_dir == str(tmp_path)
    assert captured["skills_dir"] == str(tmp_path / ".claude" / "skills")
    assert captured["register_default_tools"] == {"enable_browser": False}


def test_openhands_runtime_falls_back_to_strict_extraction_when_final_text_is_not_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    set_sdk("openhands")
    skills_dir = tmp_path / ".claude" / "skills" / "repo-skill"
    skills_dir.mkdir(parents=True)
    (skills_dir / "SKILL.md").write_text("---\nname: repo-skill\ndescription: Repo-local skill\n---\n")

    captured = _install_fake_openhands(
        monkeypatch,
        final_message="The final answer is 4. Reasoning: basic arithmetic.",
    )

    agent = Agent(
        {
            "sdk": "openhands",
            "system": "Answer the question with the final answer only.",
            "format": {
                "type": "json_schema",
                "schema": AgentResponse.model_json_schema(),
            },
            "tools": ["Read", "Edit", "Bash", "TodoWrite", "Skill"],
            "provider_id": "anthropic",
            "model_id": "claude-sonnet-4-5-20250929",
            "model": "anthropic/claude-sonnet-4-5-20250929",
            "cwd": str(tmp_path),
            "skills_dir": str(tmp_path / ".claude" / "skills"),
            "add_dirs": [],
        },
        AgentResponse,
    )

    trace = asyncio.run(agent.run("What is 2 + 2?"))

    assert trace.output is not None
    assert trace.output.final_answer == "4"
    assert trace.output.reasoning == "basic arithmetic"
    llm_calls = getattr(captured["llm"], "calls")
    assert len(llm_calls) == 1
    assert llm_calls[0]["messages"][0].role == "system"
    assert llm_calls[0]["messages"][1].role == "user"
    assert llm_calls[0]["response_format"]["type"] == "json_schema"
    assert llm_calls[0]["response_format"]["json_schema"]["name"] == "AgentResponse"
    tool_names = [tool.name for tool in captured["agent_kwargs"]["tools"]]
    assert tool_names == ["file_editor", "terminal", "task_tracker"]
    assert captured["conversation_init"]["workspace"].working_dir == str(tmp_path)
