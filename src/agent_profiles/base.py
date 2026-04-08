import asyncio
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, Type, TypeVar, Union

from pydantic import BaseModel, ValidationError

from src.openhands_runtime import run_openhands_query

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Import ClaudeAgentOptions at module level for type hints only
if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions as ClaudeAgentOptionsType
else:
    ClaudeAgentOptionsType = Any

# Type alias for options that can be static or dynamically generated
# Supports ClaudeAgentOptions, dict (for opencode), and dict without provider_id (for openhands)
OptionsProvider = Union[
    ClaudeAgentOptionsType,
    dict[str, Any],
    Callable[[], Union[ClaudeAgentOptionsType, dict[str, Any]]],
]


class AgentTrace(BaseModel, Generic[T]):
    """Metadata and output from an agent run."""

    # From first message (SystemMessage)
    uuid: str
    session_id: str
    model: str
    tools: list[str]

    # From last message (ResultMessage)
    duration_ms: int
    total_cost_usd: float
    num_turns: int
    usage: dict[str, Any]
    result: str
    is_error: bool

    # The validated structured output (None if parsing failed)
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
        """
        Create a summary of the trace for passing to downstream agents.

        - On success: returns full trace
        - On failure (parse_error): truncates to head + tail to avoid context exhaustion

        Args:
            head_chars: Characters to keep from start (only used on failure)
            tail_chars: Characters to keep from end (only used on failure)
        """
        # Build the core info
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

        # Convert result to string
        result_str = str(self.result) if self.result else ""

        # Only truncate on failure
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
    """Simple wrapper for running Claude/OpenCode/OpenHands agents.

    Args:
        options: Either a ClaudeAgentOptions instance (static) or a callable
                 that returns ClaudeAgentOptions or dict (dynamic, called on each run).
        response_model: Pydantic model for structured output validation.
    """

    TIMEOUT_SECONDS = 1200  # 20 minutes
    MAX_RETRIES = 3
    INITIAL_BACKOFF = 30  # seconds

    def __init__(self, options: OptionsProvider, response_model: Type[T]):
        self._options = options
        self.response_model = response_model

    def _get_options(self) -> Union[ClaudeAgentOptionsType, dict[str, Any]]:
        """Get options, calling the provider if it's a callable."""
        if callable(self._options):
            return self._options()
        return self._options

    @staticmethod
    def _extract_json(text: str) -> Any:
        """Extract the last valid JSON object from text.

        Scans backwards through all '{' positions and attempts to parse each
        one. Returns the first (rightmost) successful parse. This handles
        cases where '{' appears inside string literals.

        Args:
            text: String that may contain a JSON object.

        Returns:
            Parsed JSON value.

        Raises:
            ValueError: If no JSON object is found.
        """
        decoder = json.JSONDecoder()
        # Find all '{' positions, right-to-left, and try each
        pos = len(text)
        while True:
            pos = text.rfind("{", 0, pos)
            if pos == -1:
                break
            try:
                obj, _ = decoder.raw_decode(text, pos)
                return obj
            except json.JSONDecodeError:
                pass  # This '{' wasn't the start of valid JSON; try the previous one
        raise ValueError("No JSON object found in text")

    async def _execute_query(self, query: str) -> list[Any]:
        """Execute a single query attempt, routing by options type."""
        options = self._get_options()

        # Route by options type
        # ClaudeAgentOptions instance → Claude SDK
        # dict with provider_id → OpenCode
        # dict without provider_id → OpenHands

        if not isinstance(options, dict):
            # Claude SDK path
            from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

            async with ClaudeSDKClient(options) as client:
                await client.query(query)
                return [msg async for msg in client.receive_response()]

        elif "provider_id" in options:
            # OpenCode SDK path
            from opencode_ai import AsyncOpencode

            import subprocess

            try:
                client = AsyncOpencode(base_url="http://127.0.0.1:4096")
                await client.session.create(extra_body={})
            except Exception:
                subprocess.Popen(
                    ["opencode", "serve", "--port", "4096", "--hostname", "127.0.0.1"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                await asyncio.sleep(2)
                client = AsyncOpencode(base_url="http://127.0.0.1:4096")

            session = await client.session.create(extra_body={})

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
                timeout=self.TIMEOUT_SECONDS,
            )

            # Return as single-item list for consistency with Claude SDK
            return [message]

        else:
            # OpenHands path
            model_id = options.get(
                "model_id", "anthropic/claude-sonnet-4-5-20250929"
            )
            api_key = options.get("api_key", "")
            base_url = options.get("base_url")
            system = options.get("system", "")
            cwd = options.get("cwd", ".")

            # Workspace selection:
            # - If options includes "workspace" key → use that path (skill/prompt
            #   generators need project root to write files).
            # - Otherwise → temp copy so the base agent can see the repo context
            #   and skills without mutating the real workspace.
            _explicit_workspace = options.get("workspace")
            _workspace = _explicit_workspace or tempfile.mkdtemp(prefix="evoskill_oh_")
            _cleanup_workspace = _explicit_workspace is None  # only clean up temp dirs

            if _cleanup_workspace:
                shutil.copytree(
                    cwd,
                    _workspace,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns(
                        ".git",
                        ".venv",
                        "__pycache__",
                        ".pytest_cache",
                        ".mypy_cache",
                    ),
                )

            try:
                runtime_result = run_openhands_query(
                    query=query,
                    workspace=_workspace,
                    model=model_id,
                    api_key=api_key,
                    base_url=base_url,
                    system_prompt=system,
                    cwd=cwd,
                )
            finally:
                # Clean up temp workspace (never clean up the real project root)
                if _cleanup_workspace:
                    try:
                        shutil.rmtree(_workspace, ignore_errors=True)
                    except Exception:
                        pass

            return [runtime_result]

    async def _run_with_retry(self, query: str) -> list[Any]:
        """Execute query with timeout and exponential backoff retry."""
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
                backoff *= 2  # Exponential backoff

        raise last_error if last_error else RuntimeError("All retries exhausted")

    async def run(self, query: str) -> "AgentTrace[T]":
        messages = await self._run_with_retry(query)

        if not messages:
            return AgentTrace(
                uuid="unknown",
                session_id="unknown",
                model="unknown",
                tools=[],
                duration_ms=0,
                total_cost_usd=0.0,
                num_turns=0,
                usage={},
                result="",
                is_error=True,
                output=None,
                parse_error="No messages returned",
                raw_structured_output=None,
                messages=messages,
            )

        first_msg = messages[0]

        # Detect response type by message shape
        if hasattr(first_msg, "structured_output"):
            # Claude SDK path: messages list with SystemMessage, AssistantMessage, ResultMessage
            first = messages[0]
            last = messages[-1]

            output = None
            parse_error = None
            raw_structured_output = last.structured_output

            if raw_structured_output is not None:
                try:
                    output = self.response_model.model_validate(raw_structured_output)
                except (ValidationError, json.JSONDecodeError, TypeError) as e:
                    parse_error = f"{type(e).__name__}: {str(e)}"
            else:
                parse_error = (
                    "No structured output returned (context limit likely exceeded)"
                )

            return AgentTrace(
                uuid=first.data.get("uuid"),
                session_id=last.session_id,
                model=first.data.get("model"),
                tools=first.data.get("tools", []),
                duration_ms=last.duration_ms,
                total_cost_usd=last.total_cost_usd,
                num_turns=last.num_turns,
                usage=last.usage,
                result=last.result,
                is_error=last.is_error or parse_error is not None,
                output=output,
                parse_error=parse_error,
                raw_structured_output=raw_structured_output,
                messages=messages,
            )

        elif isinstance(first_msg, dict) and first_msg.get("__openhands__"):
            output = None
            parse_error = None
            raw_structured_output = None
            result_text = first_msg.get("result_text", "")

            try:
                raw_structured_output = self._extract_json(result_text)
            except ValueError:
                parse_error = "No structured output found in OpenHands response"

            if raw_structured_output is not None:
                try:
                    output = self.response_model.model_validate(raw_structured_output)
                    parse_error = None
                except (ValidationError, json.JSONDecodeError, TypeError) as e:
                    parse_error = f"{type(e).__name__}: {str(e)}"

            return AgentTrace(
                uuid="openhands",
                session_id="openhands",
                model=first_msg.get("model", "unknown"),
                tools=["terminal", "file_editor", "task_tracker"],
                duration_ms=0,
                total_cost_usd=first_msg.get("cost", 0.0),
                num_turns=1,
                usage=first_msg.get("usage", {}),
                result=result_text,
                is_error=parse_error is not None,
                output=output,
                parse_error=parse_error,
                raw_structured_output=raw_structured_output,
                messages=first_msg.get("messages", messages),
            )

        elif hasattr(first_msg, "history"):
            # OpenHands path: run_controller returns a State object directly.
            # State has .history (list of events) and .get_last_agent_message().
            from openhands.events.action import MessageAction
            from openhands.events.event import EventSource

            state = first_msg

            output = None
            parse_error = None
            raw_structured_output = None
            result_text = ""

            try:
                # Use get_last_agent_message() first (fastest path)
                last_msg = state.get_last_agent_message()
                if last_msg is not None:
                    result_text = last_msg.content
                    try:
                        raw_structured_output = self._extract_json(result_text)
                    except ValueError:
                        pass

                # If that didn't yield JSON, scan all agent messages newest-first
                if raw_structured_output is None:
                    for event in reversed(state.view):
                        if (
                            isinstance(event, MessageAction)
                            and getattr(event, "source", None) == EventSource.AGENT
                        ):
                            if not result_text:
                                result_text = event.content
                            try:
                                raw_structured_output = self._extract_json(event.content)
                                break
                            except ValueError:
                                continue

            except Exception as e:
                logger.warning(f"OpenHands result extraction failed: {e}")

            if raw_structured_output is not None:
                try:
                    output = self.response_model.model_validate(raw_structured_output)
                except (ValidationError, json.JSONDecodeError, TypeError) as e:
                    parse_error = f"{type(e).__name__}: {str(e)}"
            else:
                parse_error = "No structured output found in OpenHands response"

            options = self._get_options()
            model_name = options.get("model_id", "unknown") if isinstance(options, dict) else "unknown"

            return AgentTrace(
                uuid="openhands",
                session_id="openhands",
                model=model_name,
                tools=["terminal", "file_editor"],
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

        else:
            # OpenCode SDK path: single AssistantMessage with extra fields
            message = messages[0]

            output = None
            parse_error = None
            raw_structured_output = None

            if hasattr(message, "info") and message.info:
                raw_structured_output = message.info.get("structured")

            if raw_structured_output is not None:
                try:
                    output = self.response_model.model_validate(raw_structured_output)
                except (ValidationError, json.JSONDecodeError, TypeError) as e:
                    parse_error = f"{type(e).__name__}: {str(e)}"
            else:
                parse_error = (
                    "No structured output returned (context limit likely exceeded)"
                )

            # Extract text from parts (extra field)
            result_text = ""
            if hasattr(message, "parts"):
                for part in message.parts:
                    if isinstance(part, dict) and part.get("type") == "text":
                        result_text += part.get("text", "")

            # Get metadata from info dict
            info = message.info if hasattr(message, "info") else {}
            usage = info.get("tokens", {}) if info else {}
            cost = info.get("cost", 0.0) if info else 0.0

            options = self._get_options()
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

            return AgentTrace(
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
