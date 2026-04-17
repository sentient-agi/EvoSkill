"""Agent wrapper and AgentTrace — the public interface for running agents.

This module provides:
    - AgentTrace[T]: SDK-agnostic result from an agent run
    - Agent[T]: Generic wrapper that delegates to the active SDK's executor
    - OptionsProvider: Type alias for what you can pass as agent options

SDK-specific logic lives in:
    - _claude_executor.py: Claude SDK execution + response parsing
    - _opencode_executor.py: OpenCode SDK execution + server management + response parsing
"""

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, Type, TypeVar, Union
from pydantic import BaseModel
from .sdk_config import get_sdk, is_claude_sdk, is_codex_sdk, is_goose_sdk

logger = logging.getLogger(__name__)

# Generic type variable — every Agent[T] produces AgentTrace[T] where T is a Pydantic model
# (e.g., AgentResponse, SkillProposerResponse, etc.)
T = TypeVar("T", bound=BaseModel)

# Import ClaudeAgentOptions only for type hints (not at runtime).
# This allows the module to load even if claude-agent-sdk is not installed.
if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions as ClaudeAgentOptionsType
else:
    ClaudeAgentOptionsType = Any

# What you can pass to Agent() as options:
#   - ClaudeAgentOptions object (static Claude config)
#   - dict (static OpenCode config)
#   - Callable that returns either (factory pattern — called fresh on each .run())
OptionsProvider = Union[
    ClaudeAgentOptionsType,
    dict[str, Any],
    Callable[[], Union[ClaudeAgentOptionsType, dict[str, Any]]],
]


class AgentTrace(BaseModel, Generic[T]):
    """Metadata and output from a single agent run.

    This is SDK-agnostic — both Claude and OpenCode executors produce the same
    AgentTrace structure, so downstream code (loop, evaluation, cache) never
    needs to know which SDK was used.
    """

    # Identity — who ran and with what config
    uuid: str = ""
    session_id: str = ""
    model: str = ""
    tools: list[str] = []

    # Metrics — cost and performance
    duration_ms: int
    total_cost_usd: float
    num_turns: int
    usage: dict[str, Any]
    result: str
    is_error: bool

    # Structured output — the main thing downstream code cares about.
    # None if parsing failed (check parse_error for why).
    output: Optional[T] = None

    # Error info when output parsing fails
    parse_error: Optional[str] = None
    raw_structured_output: Optional[Any] = None

    # Full response list for debugging
    messages: list[Any]

    class Config:
        arbitrary_types_allowed = True

    def summarize(
        self,
        head_chars: int = 60_000,
        tail_chars: int = 60_000,
        tool_result_max_chars: int = 4_000,
    ) -> str:
        """Create a turn-by-turn transcript for passing to downstream agents.

        Walks through self.messages and renders every block type per turn:
            - ThinkingBlock   -> "(thinking): ..."
            - TextBlock       -> "text: ..."
            - ToolUseBlock    -> "(tool.<name>): <json>"
            - ToolResultBlock -> "(tool.result): <content>" (correlated by id)

        Fallback to the final ResultMessage.result when messages are empty
        or the SDK block types can't be imported.
        """
        lines = [
            f"Model: {self.model}",
            f"Turns: {self.num_turns}",
            f"Duration: {self.duration_ms}ms",
            f"Is Error: {self.is_error}",
        ]
        if self.parse_error:
            lines.append(f"Parse Error: {self.parse_error}")
        if self.output:
            lines.append(f"Output: {self.output}")

        transcript = _render_turn_transcript(self.messages, tool_result_max_chars)

        if transcript:
            lines.append("\n## Turn-by-turn transcript\n")
            lines.append(transcript)
        else:
            result_str = str(self.result) if self.result else ""
            if self.parse_error and len(result_str) > (head_chars + tail_chars):
                truncated_middle = len(result_str) - head_chars - tail_chars
                lines.append(f"\n## Result (truncated, {truncated_middle:,} chars omitted)")
                lines.append(f"### Start:\n{result_str[:head_chars]}")
                lines.append(f"\n[... {truncated_middle:,} characters truncated ...]\n")
                lines.append(f"### End:\n{result_str[-tail_chars:]}")
            else:
                lines.append(f"\n## Full Result\n{result_str}")

        return "\n".join(lines)


def _render_turn_transcript(messages: list[Any], tool_result_max_chars: int) -> str:
    """Render messages as a turn-by-turn transcript. Returns '' on failure."""
    try:
        from claude_agent_sdk import (
            AssistantMessage, UserMessage, ToolUseBlock, ToolResultBlock, TextBlock,
        )
        try:
            from claude_agent_sdk import ThinkingBlock
        except ImportError:
            ThinkingBlock = None
    except ImportError:
        return ""

    if not messages:
        return ""

    # First pass: collect tool results keyed by tool_use_id for inline correlation
    tool_results: dict[str, str] = {}
    for msg in messages:
        if isinstance(msg, UserMessage):
            content = getattr(msg, "content", None) or []
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, ToolResultBlock):
                    continue
                tool_use_id = getattr(block, "tool_use_id", None) or getattr(block, "id", None)
                if not tool_use_id:
                    continue
                raw = getattr(block, "content", "")
                if isinstance(raw, list):
                    text = "\n".join(getattr(c, "text", str(c)) for c in raw)
                else:
                    text = str(raw) if raw is not None else ""
                if len(text) > tool_result_max_chars:
                    text = text[:tool_result_max_chars] + f"\n...[truncated, {len(text) - tool_result_max_chars:,} chars]"
                is_error = getattr(block, "is_error", False)
                marker = "[ERROR] " if is_error else ""
                tool_results[str(tool_use_id)] = f"{marker}{text}"

    # Second pass: render each AssistantMessage as a turn
    out_lines: list[str] = []
    turn_num = 0
    for msg in messages:
        if not isinstance(msg, AssistantMessage):
            continue
        turn_num += 1
        out_lines.append(f"\n--- Turn {turn_num} ---")
        for block in getattr(msg, "content", []) or []:
            if ThinkingBlock is not None and isinstance(block, ThinkingBlock):
                text = (getattr(block, "thinking", "") or "").strip()
                if text:
                    out_lines.append(f"(thinking): {text}")
            elif isinstance(block, TextBlock):
                text = (getattr(block, "text", "") or "").strip()
                if text:
                    out_lines.append(f"text: {text}")
            elif isinstance(block, ToolUseBlock):
                name = getattr(block, "name", "?")
                try:
                    inp = json.dumps(getattr(block, "input", None), default=str, ensure_ascii=False)
                except Exception:
                    inp = str(getattr(block, "input", None))
                out_lines.append(f"(tool.{name}): {inp}")
                tool_use_id = getattr(block, "id", None) or getattr(block, "tool_use_id", None)
                if tool_use_id and str(tool_use_id) in tool_results:
                    out_lines.append(f"(tool.result): {tool_results[str(tool_use_id)]}")
            else:
                out_lines.append(f"({type(block).__name__}): {str(block)[:500]}")

    return "\n".join(out_lines).strip()


class Agent(Generic[T]):
    """Generic wrapper for running agents via Claude SDK or OpenCode SDK.

    Usage:
        agent = Agent(options=my_options, response_model=AgentResponse)
        trace = await agent.run("What is 2+2?")
        print(trace.output.final_answer)  # "4"

    The Agent handles:
        - SDK selection (Claude vs OpenCode) based on global sdk_config
        - Retry with exponential backoff (3 attempts, 30s → 60s → 120s)
        - 20-minute timeout per attempt
        - Structured output validation via the SDK-specific executor
    """

    TIMEOUT_SECONDS = 1200   # 20 minutes per attempt
    MAX_RETRIES = 3          # Total attempts before giving up
    INITIAL_BACKOFF = 30     # Seconds to wait after first failure (doubles each retry)

    def __init__(
        self,
        options: OptionsProvider,
        response_model: Type[T],
        *,
        name: str = "agent",
        timeout_seconds: int | None = None,
        max_retries: int | None = None,
    ):
        self._options = options
        self.response_model = response_model
        self.name = name
        self.timeout_seconds = (
            self.TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        )
        self.max_retries = self.MAX_RETRIES if max_retries is None else max_retries

    def _get_options(self) -> Union[ClaudeAgentOptionsType, dict[str, Any]]:
        """Resolve options — if it's a factory (callable), call it to get fresh options."""
        if callable(self._options):
            return self._options()
        return self._options

    async def _execute_query(self, query: str) -> list[Any]:
        """Execute a single query by delegating to the active SDK's executor."""
        options = self._get_options()

        sdk = get_sdk()
        if sdk == "claude":
            from .claude import executor as _claude_executor
            return await _claude_executor.execute_query(options, query, agent_name=self.name)
        if sdk == "opencode":
            from .opencode import executor as _opencode_executor
            return await _opencode_executor.execute_query(options, query)
        if sdk == "openhands":
            from .openhands import executor as _openhands_executor
            return await _openhands_executor.execute_query(options, query)
        if sdk == "codex":
            from .codex import executor as _codex_executor
            return await _codex_executor.execute_query(options, query)
        if sdk == "goose":
            from .goose import executor as _goose_executor
            return await _goose_executor.execute_query(options, query)
        else:
            raise ValueError(f"Unknown SDK: {sdk!r}")

    async def _run_with_retry(self, query: str) -> list[Any]:
        """Execute query with timeout and exponential backoff retry.

        Attempts up to max_retries times. On each failure:
            - Logs a warning
            - Waits (30s → 60s → 120s)
            - Tries again

        Raises the last error if all retries are exhausted.
        """
        last_error: Exception | None = None
        backoff = self.INITIAL_BACKOFF

        for attempt in range(self.max_retries):
            try:
                async with asyncio.timeout(self.timeout_seconds):
                    return await self._execute_query(query)
            except asyncio.TimeoutError:
                last_error = TimeoutError(
                    f"Query timed out after {self.timeout_seconds}s"
                )
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} timed out. Retrying in {backoff}s..."
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt + 1}/{self.max_retries} failed: {e}. Retrying in {backoff}s..."
                )

            if attempt < self.max_retries - 1:
                await asyncio.sleep(backoff)
                backoff *= 2  # Exponential backoff: 30s → 60s → 120s

        raise last_error if last_error else RuntimeError("All retries exhausted")

    async def run(self, query: str) -> AgentTrace[T]:
        """Run the agent on a query and return a structured AgentTrace.

        This is the main entry point. It:
            1. Calls _run_with_retry to get raw SDK messages
            2. Delegates response parsing to the active SDK's executor
            3. Returns an AgentTrace with all metadata + parsed output
        """
        messages = await self._run_with_retry(query)

        sdk = get_sdk()
        if sdk == "claude":
            from .claude import executor as _claude_executor
            fields = _claude_executor.parse_response(messages, self.response_model)
        elif sdk == "opencode":
            from .opencode import executor as _opencode_executor
            fields = _opencode_executor.parse_response(messages, self.response_model, self._get_options)
        elif sdk == "openhands":
            from .openhands import executor as _openhands_executor
            fields = await _openhands_executor.parse_response(messages,self.response_model,self._get_options,query)
        elif sdk == "codex":
            from .codex import executor as _codex_executor
            fields = _codex_executor.parse_response(messages, self.response_model, self._get_options)
        elif sdk == "goose":
            from .goose import executor as _goose_executor
            fields = _goose_executor.parse_response(messages, self.response_model, self._get_options)
        else:
            raise ValueError(f"Unknown SDK: {sdk!r}")

        return AgentTrace(**fields)
