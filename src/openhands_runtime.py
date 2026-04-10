"""OpenHands runtime helpers built on the public SDK surface."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _get_openhands_context_api():
    from openhands.sdk import AgentContext
    from openhands.sdk.context.skills import load_project_skills, load_skills_from_dir

    return AgentContext, load_project_skills, load_skills_from_dir


def _get_openhands_sdk_api():
    from pydantic import SecretStr
    from openhands.sdk import LLM, Agent, Conversation
    from openhands.sdk.tool import Tool
    from openhands.tools.terminal import TerminalTool
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.task_tracker import TaskTrackerTool

    return (
        SecretStr,
        LLM,
        Agent,
        Tool,
        Conversation,
        TerminalTool,
        FileEditorTool,
        TaskTrackerTool,
    )


def build_openhands_agent_context(
    *,
    workspace: Path,
    system_prompt: str,
):
    """Build OpenHands AgentContext with native project and repo skills."""
    AgentContext, load_project_skills, load_skills_from_dir = _get_openhands_context_api()

    project_skills = load_project_skills(str(workspace))

    skills_dir = workspace / ".agents" / "skills"
    try:
        _, _, agent_skills = load_skills_from_dir(skills_dir)
    except Exception:
        agent_skills = {}

    return AgentContext(
        skills=[*project_skills, *agent_skills.values()],
        system_message_suffix=system_prompt,
        load_public_skills=False,
    )


def _collect_openhands_result(conversation: Any) -> dict[str, Any]:
    """Collect the final OpenHands result in a generic shape for Agent.run()."""
    events = list(getattr(conversation, "_evoskill_events", []))
    result_text = ""

    state = getattr(conversation, "state", None)
    if state is not None and hasattr(state, "get_last_agent_message"):
        try:
            last_message = state.get_last_agent_message()
            if last_message is not None:
                result_text = getattr(last_message, "content", "") or str(last_message)
        except Exception:
            result_text = ""

    if not result_text:
        for event in reversed(events):
            content = getattr(event, "content", None)
            if content:
                result_text = content
                break
            if hasattr(event, "to_llm_message"):
                try:
                    result_text = str(event.to_llm_message())
                    break
                except Exception:
                    continue

    agent = getattr(conversation, "agent", None)
    llm = getattr(agent, "llm", None)
    metrics = getattr(llm, "metrics", None)
    cost = getattr(metrics, "accumulated_cost", 0.0) if metrics is not None else 0.0
    model = getattr(llm, "model", "unknown") if llm is not None else "unknown"

    return {
        "__openhands__": True,
        "result_text": result_text,
        "messages": events,
        "cost": float(cost or 0.0),
        "usage": {},
        "model": model,
    }


def run_openhands_query(
    *,
    query: str,
    workspace: str | Path,
    model: str,
    api_key: str,
    base_url: str | None,
    system_prompt: str,
    cwd: str | Path | None = None,
) -> dict[str, Any]:
    """Run a query through OpenHands using the public SDK."""
    (
        SecretStr,
        LLM,
        Agent,
        Tool,
        Conversation,
        TerminalTool,
        FileEditorTool,
        TaskTrackerTool,
    ) = _get_openhands_sdk_api()

    workspace_path = Path(workspace)
    agent_context = build_openhands_agent_context(
        workspace=workspace_path,
        system_prompt=system_prompt,
    )

    llm = LLM(
        usage_id="agent",
        model=model,
        api_key=SecretStr(api_key),
        base_url=base_url,
    )
    tools = [
        Tool(name=TerminalTool.name, params={"terminal_type": "subprocess"}),
        Tool(name=FileEditorTool.name),
        Tool(name=TaskTrackerTool.name),
    ]
    agent = Agent(llm=llm, tools=tools, agent_context=agent_context)

    events: list[Any] = []

    def callback(event: Any) -> None:
        events.append(event)

    conversation = Conversation(
        agent=agent,
        workspace=str(workspace_path),
        callbacks=[callback],
    )
    setattr(conversation, "_evoskill_events", events)
    conversation.send_message(query)
    conversation.run()
    return _collect_openhands_result(conversation)
