"""OpenHands SDK execution and structured-output fallback."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Type

from pydantic import BaseModel, SecretStr, ValidationError


_DIRECT_PARSE_RESPONSE_MODELS = {
    "AgentResponse",
    "SkillProposerResponse",
    "PromptProposerResponse",
    "PromptGeneratorResponse",
    "ToolGeneratorResponse",
}

_EXTRACTION_SYSTEM_PROMPT = (
    "Convert the assistant's final answer into a schema-valid JSON object. "
    "Preserve the original meaning. Output JSON only."
)


def _import_openhands_sdk() -> Any:
    try:
        return importlib.import_module("openhands.sdk")
    except ImportError as exc:
        raise ImportError(
            "OpenHands SDK is not installed. Add 'openhands-sdk' and 'openhands-tools' "
            "to the environment to use the OpenHands harness."
        ) from exc


def _import_openhands_skills() -> Any:
    return importlib.import_module("openhands.sdk.context.skills")


def _import_openhands_llm() -> Any:
    return importlib.import_module("openhands.sdk.llm")


def _import_openhands_tools() -> Any:
    try:
        return importlib.import_module("openhands.tools")
    except ImportError as exc:
        raise ImportError(
            "OpenHands tools are not installed. Add 'openhands-tools' "
            "to the environment to use the OpenHands harness."
        ) from exc


def _register_openhands_tools() -> dict[str, str]:
    tools_module = _import_openhands_tools()
    register_default_tools = getattr(tools_module, "register_default_tools", None)
    if callable(register_default_tools):
        register_default_tools(enable_browser=False)

    file_editor_module = importlib.import_module("openhands.tools.file_editor")
    terminal_module = importlib.import_module("openhands.tools.terminal")
    task_tracker_module = importlib.import_module("openhands.tools.task_tracker")

    file_editor_name = file_editor_module.FileEditorTool.name
    terminal_name = terminal_module.TerminalTool.name
    task_tracker_name = task_tracker_module.TaskTrackerTool.name

    return {
        "Read": file_editor_name,
        "Write": file_editor_name,
        "Edit": file_editor_name,
        "Bash": terminal_name,
        "Glob": terminal_name,
        "Grep": terminal_name,
        "WebFetch": terminal_name,
        "WebSearch": terminal_name,
        "BashOutput": terminal_name,
        "TodoWrite": task_tracker_name,
    }


def _build_tool_objects(
    raw_tools: list[str],
    tool_cls: Any,
    tool_name_map: dict[str, str],
) -> list[Any]:
    tool_names: list[str] = []
    for raw_tool in raw_tools:
        mapped = tool_name_map.get(raw_tool)
        if mapped and mapped not in tool_names:
            tool_names.append(mapped)
    return [tool_cls(name=name) for name in tool_names]


def _resolve_api_key(options: dict[str, Any]) -> SecretStr | None:
    provider_id = str(options.get("provider_id", "")).lower()
    if provider_id == "anthropic":
        value = os.environ.get("ANTHROPIC_API_KEY")
        if value:
            return SecretStr(value)
    value = os.environ.get("LLM_API_KEY")
    if value:
        return SecretStr(value)
    return None


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "content" in item:
                    parts.append(_extract_text(item["content"]))
            elif hasattr(item, "text"):
                parts.append(str(getattr(item, "text")))
            elif hasattr(item, "content"):
                parts.append(_extract_text(getattr(item, "content")))
        return "".join(parts)
    if isinstance(value, dict):
        if "content" in value:
            return _extract_text(value["content"])
        if value.get("type") == "text":
            return str(value.get("text", ""))
    if hasattr(value, "content"):
        return _extract_text(getattr(value, "content"))
    if hasattr(value, "text"):
        return str(getattr(value, "text"))
    if hasattr(value, "message"):
        return _extract_text(getattr(value, "message"))
    return str(value)


def _extract_final_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        role = getattr(message, "role", None)
        if role in (None, "assistant"):
            text = _extract_text(message)
            if text:
                return text
    return _extract_text(messages[-1]) if messages else ""


def _extract_json_candidate(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def _parse_output(
    candidate: Any,
    response_model: Type[BaseModel],
) -> tuple[BaseModel | None, Any | None, str | None]:
    try:
        if isinstance(candidate, response_model):
            return candidate, candidate.model_dump(), None
        if isinstance(candidate, dict):
            output = response_model.model_validate(candidate)
            return output, candidate, None
        if isinstance(candidate, str):
            json_candidate = _extract_json_candidate(candidate)
            output = response_model.model_validate_json(json_candidate)
            return output, output.model_dump(), None
        output = response_model.model_validate(candidate)
        return output, output.model_dump(), None
    except (ValidationError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def _build_agent_context(sdk_module: Any, options: dict[str, Any]) -> Any:
    skills_module = _import_openhands_skills()
    _, _, agent_skills = skills_module.load_skills_from_dir(Path(options["skills_dir"]))
    return sdk_module.AgentContext(
        skills=list(agent_skills.values()),
        system_message_suffix=options.get("system", ""),
    )


def _build_workspace(sdk_module: Any, options: dict[str, Any]) -> Any:
    cwd = options.get("cwd")
    workspace_cls = getattr(sdk_module, "LocalWorkspace", None)
    if workspace_cls is None:
        return cwd
    try:
        return workspace_cls(working_dir=cwd)
    except TypeError:
        return workspace_cls(cwd)


async def execute_query(options: dict[str, Any], query: str) -> list[Any]:
    """Execute a query via OpenHands and return execution context for parsing."""
    sdk_module = _import_openhands_sdk()
    tool_name_map = _register_openhands_tools()
    llm_kwargs: dict[str, Any] = {
        "model": options.get("model"),
        "service_id": "agent",
    }
    api_key = _resolve_api_key(options)
    if api_key is not None:
        llm_kwargs["api_key"] = api_key

    llm = sdk_module.LLM(**llm_kwargs)
    agent_context = _build_agent_context(sdk_module, options)
    tools = _build_tool_objects(
        list(options.get("tools", [])),
        sdk_module.Tool,
        tool_name_map,
    )

    if hasattr(sdk_module, "get_default_agent"):
        agent = sdk_module.get_default_agent(
            llm=llm,
            tools=tools,
            agent_context=agent_context,
            cli_mode=True,
        )
    else:
        agent = sdk_module.Agent(llm=llm, tools=tools, agent_context=agent_context)

    collected_messages: list[Any] = []

    def _collector(event: Any) -> None:
        collected_messages.append(event)

    conversation_kwargs = {
        "agent": agent,
        "workspace": _build_workspace(sdk_module, options),
        "callbacks": [_collector],
    }

    try:
        conversation = sdk_module.Conversation(**conversation_kwargs)
    except TypeError:
        conversation_kwargs.pop("callbacks")
        conversation = sdk_module.Conversation(**conversation_kwargs)

    send_result = conversation.send_message(query)
    if inspect.isawaitable(send_result):
        await send_result

    start = time.perf_counter()
    if inspect.iscoroutinefunction(conversation.run):
        await conversation.run()
    else:
        await asyncio.to_thread(conversation.run)
    duration_ms = int((time.perf_counter() - start) * 1000)

    raw_messages = list(
        getattr(conversation, "messages", None)
        or getattr(getattr(conversation, "state", None), "events", None)
        or getattr(getattr(conversation, "state", None), "history", None)
        or collected_messages
    )

    return [{
        "conversation": conversation,
        "llm": llm,
        "raw_messages": raw_messages,
        "duration_ms": duration_ms,
    }]


def _extract_metrics(payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
    llm = payload["llm"]
    metrics = {}
    if hasattr(llm, "metrics") and hasattr(llm.metrics, "get"):
        raw = llm.metrics.get() or {}
        if isinstance(raw, dict):
            metrics = raw

    usage = metrics.get("accumulated_token_usage") or metrics.get("usage") or {}
    total_cost = metrics.get("accumulated_cost")
    if total_cost is None:
        total_cost = metrics.get("cost", 0.0)
    return usage, float(total_cost or 0.0)


async def _run_fallback_extraction(
    llm: Any,
    *,
    query: str,
    result_text: str,
    response_model: Type[BaseModel],
) -> tuple[BaseModel | None, Any | None, str | None]:
    schema = response_model.model_json_schema()
    extraction_query = (
        f"Original user query:\n{query}\n\n"
        f"Assistant final answer:\n{result_text}\n"
    )
    llm_module = _import_openhands_llm()
    response = await asyncio.to_thread(
        llm.completion,
        messages=[
            llm_module.Message(
                role="system",
                content=[llm_module.TextContent(text=_EXTRACTION_SYSTEM_PROMPT)],
            ),
            llm_module.Message(
                role="user",
                content=[llm_module.TextContent(text=extraction_query)],
            ),
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "schema": schema,
                "strict": True,
            },
        },
        temperature=0,
    )
    extracted_text = _extract_text(response.message)
    return _parse_output(extracted_text, response_model)


async def parse_response(
    messages: list[Any],
    response_model: Type[BaseModel],
    get_options: Callable[[], Any],
    query: str,
) -> dict[str, Any]:
    """Parse OpenHands execution results into AgentTrace field values."""
    payload = messages[0]
    raw_messages = list(payload.get("raw_messages", []))
    result_text = _extract_final_text(raw_messages)

    output, raw_structured_output, parse_error = _parse_output(result_text, response_model)

    if output is None and response_model.__name__ in _DIRECT_PARSE_RESPONSE_MODELS:
        fallback_output, fallback_raw, fallback_error = await _run_fallback_extraction(
            payload["llm"],
            query=query,
            result_text=result_text,
            response_model=response_model,
        )
        if fallback_output is not None:
            output = fallback_output
            raw_structured_output = fallback_raw
            parse_error = None
        else:
            parse_error = fallback_error or parse_error

    options = get_options()
    usage, total_cost = _extract_metrics(payload)

    return dict(
        uuid="",
        session_id="",
        model=options.get("model", options.get("model_id", "")),
        tools=list(options.get("tools", [])),
        duration_ms=payload.get("duration_ms", 0),
        total_cost_usd=total_cost,
        num_turns=max(1, len(raw_messages)) if raw_messages else 1,
        usage=usage,
        result=result_text,
        is_error=parse_error is not None,
        output=output,
        parse_error=parse_error,
        raw_structured_output=raw_structured_output,
        messages=raw_messages,
    )
