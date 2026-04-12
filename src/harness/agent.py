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
import logging
from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, Type, TypeVar, Union
from pydantic import BaseModel
from .sdk_config import is_claude_sdk, is_codex_sdk

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
    ) -> str:
        """Create a text summary of this trace for passing to downstream agents.

        The proposer agent reads these summaries to understand what went wrong.

        On success: returns full trace (the proposer needs all the details).
        On failure (parse_error): truncates to head + tail to avoid blowing up
        the proposer's context window with a massive failed response.
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

        result_str = str(self.result) if self.result else ""

        # Only truncate on failure — successful traces are usually small enough
        if self.parse_error and len(result_str) > (head_chars + tail_chars):
            truncated_middle = len(result_str) - head_chars - tail_chars
            lines.append(f"\n## Result (truncated, {truncated_middle:,} chars omitted)")
            lines.append(f"### Start:\n{result_str[:head_chars]}")
            lines.append(f"\n[... {truncated_middle:,} characters truncated ...]\n")
            lines.append(f"### End:\n{result_str[-tail_chars:]}")
        else:
            lines.append(f"\n## Full Result\n{result_str}")

        return "\n".join(lines)


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

    def __init__(self, options: OptionsProvider, response_model: Type[T]):
        self._options = options
        self.response_model = response_model

    def _get_options(self) -> Union[ClaudeAgentOptionsType, dict[str, Any]]:
        """Resolve options — if it's a factory (callable), call it to get fresh options."""
        if callable(self._options):
            return self._options()
        return self._options

    async def _execute_query(self, query: str) -> list[Any]:
        """Execute a single query by delegating to the active SDK's executor."""
        options = self._get_options()

        if is_claude_sdk():
            from .claude import executor as _claude_executor
            return await _claude_executor.execute_query(options, query)
        elif is_codex_sdk():
            from .codex import executor as _codex_executor
            return await _codex_executor.execute_query(options, query)
        else:
            from .opencode import executor as _opencode_executor
            return await _opencode_executor.execute_query(options, query)

    async def _run_with_retry(self, query: str) -> list[Any]:
        """Execute query with timeout and exponential backoff retry.

        Attempts up to MAX_RETRIES times. On each failure:
            - Logs a warning
            - Waits (30s → 60s → 120s)
            - Tries again

        Raises the last error if all retries are exhausted.
        """
        last_error: Exception | None = None
        backoff = self.INITIAL_BACKOFF

        for attempt in range(self.MAX_RETRIES):
            try:
                async with asyncio.timeout(self.TIMEOUT_SECONDS):
                    return await self._execute_query(query)
            except asyncio.TimeoutError:
                last_error = TimeoutError(
                    f"Query timed out after {self.TIMEOUT_SECONDS}s"
                )
                logger.warning(
                    f"Attempt {attempt + 1}/{self.MAX_RETRIES} timed out. Retrying in {backoff}s..."
                )
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt + 1}/{self.MAX_RETRIES} failed: {e}. Retrying in {backoff}s..."
                )

            if attempt < self.MAX_RETRIES - 1:
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

        if is_claude_sdk():
            from .claude import executor as _claude_executor
            fields = _claude_executor.parse_response(messages, self.response_model)
        elif is_codex_sdk():
            from .codex import executor as _codex_executor
            fields = _codex_executor.parse_response(messages, self.response_model, self._get_options)
        else:
            from .opencode import executor as _opencode_executor
            fields = _opencode_executor.parse_response(messages, self.response_model, self._get_options,)

        return AgentTrace(**fields)
