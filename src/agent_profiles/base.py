import asyncio
import json
import logging
import socket
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, Type, TypeVar, Union

from pydantic import BaseModel, ValidationError

from .sdk_config import is_claude_sdk

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
_OPENCODE_SERVER_PORTS: dict[str, int] = {}
_OPENCODE_REQUEST_TIMEOUT_SECONDS = 600

# Import ClaudeAgentOptions at module level for type hints only
if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions as ClaudeAgentOptionsType
else:
    ClaudeAgentOptionsType = Any

# Type alias for options that can be static or dynamically generated
# Supports both ClaudeAgentOptions and dict (for opencode)
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


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def _server_matches_project(client: Any, expected_cwd: str | None) -> bool:
    app_api = getattr(client, "app", None)
    if app_api is None or not hasattr(app_api, "get"):
        return False

    try:
        app_info = await app_api.get()
    except Exception:
        return False

    if not expected_cwd:
        return True

    if isinstance(app_info, dict):
        path_info = app_info.get("path", {})
        directory = path_info.get("cwd")
    else:
        path_info = getattr(app_info, "path", None)
        directory = getattr(path_info, "cwd", None)

    if not directory:
        return False

    return Path(directory).resolve() == Path(expected_cwd).resolve()


class Agent(Generic[T]):
    """Simple wrapper for running Claude agents.

    Args:
        options: Either a ClaudeAgentOptions instance (static) or a callable
                 that returns ClaudeAgentOptions (dynamic, called on each run).
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

    async def _execute_query(self, query: str) -> list[Any]:
        """Execute a single query attempt."""
        options = self._get_options()

        if is_claude_sdk():
            # Claude SDK path
            from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

            # Convert dict to ClaudeAgentOptions if needed
            if isinstance(options, dict):
                claude_opts = ClaudeAgentOptions(
                    system_prompt=options.get("system"),
                    allowed_tools=list(options.get("tools", {}).keys())
                    if options.get("tools")
                    else [],
                    output_format=options.get("format"),
                    setting_sources=["user", "project"],
                    permission_mode="acceptEdits",
                )
                if "model_id" in options and "claude" in options["model_id"].lower():
                    claude_opts.model = options["model_id"]
                options = claude_opts

            async with ClaudeSDKClient(options) as client:
                await client.query(query)
                return [msg async for msg in client.receive_response()]
        else:
            # OpenCode SDK path
            from opencode_ai import AsyncOpencode

            if not isinstance(options, dict):
                raise TypeError(
                    f"OpenCode SDK requires dict options, got {type(options)}"
                )

            # Start opencode server if needed
            import subprocess
            import time

            requested_cwd = options.get("cwd")
            if requested_cwd:
                requested_cwd = str(Path(requested_cwd).resolve())

            port = _OPENCODE_SERVER_PORTS.get(requested_cwd or "", 4096)
            client = AsyncOpencode(
                base_url=f"http://127.0.0.1:{port}",
                timeout=_OPENCODE_REQUEST_TIMEOUT_SECONDS,
            )

            if not await _server_matches_project(client, requested_cwd):
                if requested_cwd:
                    port = _OPENCODE_SERVER_PORTS.get(requested_cwd, port)
                    if port == 4096:
                        port = _find_free_port()
                        _OPENCODE_SERVER_PORTS[requested_cwd] = port

                subprocess.Popen(
                    ["opencode", "serve", "--port", str(port), "--hostname", "127.0.0.1"],
                    cwd=requested_cwd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                time.sleep(2)
                client = AsyncOpencode(
                    base_url=f"http://127.0.0.1:{port}",
                    timeout=_OPENCODE_REQUEST_TIMEOUT_SECONDS,
                )

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
            )

            # Return as single-item list for consistency with Claude SDK
            return [message]

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

    async def run(self, query: str) -> AgentTrace[T]:
        messages = await self._run_with_retry(query)

        if is_claude_sdk():
            # Claude SDK: messages list with SystemMessage, AssistantMessage, ResultMessage
            first = messages[0]
            last = messages[-1]

            # Try to parse structured output
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
        else:
            # OpenCode SDK: single AssistantMessage with extra fields
            message = messages[0]

            # Extract structured output from info dict (extra field)
            output = None
            parse_error = None
            raw_structured_output = None

            if hasattr(message, "info") and message.info:
                raw_structured_output = message.info.get("structured_output")
                if raw_structured_output is None:
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
