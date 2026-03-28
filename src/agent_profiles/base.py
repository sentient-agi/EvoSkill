import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, Type, TypeVar, Union

from pydantic import BaseModel, ValidationError

from .sdk_config import is_claude_sdk

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OpenCodeMessage:
    """Simple container for opencode CLI JSON output, pickleable."""

    def __init__(self):
        self.parts = []
        self.info = {}
        self.session_id = None

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

        # Determine SDK per-agent: use Claude SDK if options is ClaudeAgentOptions,
        # opencode if options is dict. Global setting is the fallback.
        from claude_agent_sdk import ClaudeAgentOptions as _CAO
        use_claude = isinstance(options, _CAO) or (is_claude_sdk() and not isinstance(options, dict))

        if use_claude:
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
            # OpenCode CLI path — uses `opencode run` for full agentic loop
            # (including web search, tool use, multi-turn reasoning)
            import subprocess
            import os

            if not isinstance(options, dict):
                raise TypeError(
                    f"OpenCode SDK requires dict options, got {type(options)}"
                )

            model = options.get("model_id")
            provider = options.get("provider_id")
            model_flag = f"{provider}/{model}" if provider and model else model

            opencode_bin = "opencode.cmd" if os.name == "nt" else "opencode"
            cmd = [opencode_bin, "run", "--format", "json"]
            if model_flag:
                cmd.extend(["-m", model_flag])

            # Isolate each opencode process with its own data dir
            # to avoid SQLite WAL conflicts from concurrent runs
            import uuid as _uuid
            run_id = _uuid.uuid4().hex[:8]
            env = {**os.environ}
            env["XDG_DATA_HOME"] = os.path.join(
                os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share")),
                f"opencode-run-{run_id}",
            )

            # Allow callers to set a custom working directory for opencode
            run_dir = options.get("run_dir")

            proc = await asyncio.create_subprocess_exec(
                *cmd, query,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=run_dir,
            )
            stdout, stderr = await proc.communicate()

            if proc.returncode != 0:
                raise RuntimeError(
                    f"opencode run failed (exit {proc.returncode}): {stderr.decode()[:500]}"
                )

            # Parse JSON lines output from opencode run
            import json as _json

            parts = []
            info = {}
            for line in stdout.decode().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                etype = event.get("type")
                if etype == "text":
                    parts.append(event.get("part", {}))
                elif etype == "tool_use":
                    parts.append(event.get("part", {}))
                elif etype == "step_start":
                    parts.append(event.get("part", {}))
                elif etype == "step_finish":
                    parts.append(event.get("part", {}))
                elif etype == "assistant":
                    info = event.get("message", event)

            msg = OpenCodeMessage()
            msg.parts = parts
            msg.info = info
            msg.session_id = info.get("sessionID")
            return [msg]

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

        # Detect which SDK was used based on the options type (same logic as _execute_query)
        options = self._get_options()
        from claude_agent_sdk import ClaudeAgentOptions as _CAO
        use_claude = isinstance(options, _CAO) or (is_claude_sdk() and not isinstance(options, dict))

        if use_claude:
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
            # OpenCode CLI: parse parts from `opencode run --format json`
            message = messages[0]

            # Extract structured output from tool parts or info
            output = None
            parse_error = None
            raw_structured_output = None

            # Check StructuredOutput tool part first
            if hasattr(message, "parts"):
                for part in message.parts:
                    if isinstance(part, dict) and part.get("tool") == "StructuredOutput":
                        state = part.get("state", {})
                        inp = state.get("input")
                        if inp:
                            raw_structured_output = inp
                            break

            # Fall back to info.structured
            if raw_structured_output is None:
                info_dict = message.info if hasattr(message, "info") else {}
                if isinstance(info_dict, dict):
                    raw_structured_output = info_dict.get("structured")

            if raw_structured_output is not None:
                try:
                    output = self.response_model.model_validate(raw_structured_output)
                except (ValidationError, json.JSONDecodeError, TypeError) as e:
                    parse_error = f"{type(e).__name__}: {str(e)}"
            else:
                parse_error = (
                    "No structured output returned (context limit likely exceeded)"
                )

            # Extract text from parts
            result_text = ""
            tools_used = []
            total_cost = 0.0
            total_tokens = {}
            num_turns = 0
            if hasattr(message, "parts"):
                for part in message.parts:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text":
                        result_text += part.get("text", "")
                    elif part.get("type") == "tool":
                        tools_used.append(part.get("tool", ""))
                    elif part.get("type") == "step-finish":
                        num_turns += 1
                        total_cost += part.get("cost", 0.0)
                        tokens = part.get("tokens", {})
                        for k, v in tokens.items():
                            if isinstance(v, (int, float)):
                                total_tokens[k] = total_tokens.get(k, 0) + v

            usage = total_tokens
            cost = total_cost

            options = self._get_options()
            model_name = (
                options.get("model_id", "unknown")
                if isinstance(options, dict)
                else "unknown"
            )

            return AgentTrace(
                uuid=message.session_id or "unknown",
                session_id=message.session_id or "unknown",
                model=model_name,
                tools=tools_used,
                duration_ms=0,
                total_cost_usd=cost,
                num_turns=num_turns,
                usage=usage,
                result=result_text,
                is_error=parse_error is not None,
                output=output,
                parse_error=parse_error,
                raw_structured_output=raw_structured_output,
                messages=messages,
            )
