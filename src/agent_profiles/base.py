from typing import TypeVar, Type, Generic, Any, Callable, Union, Optional
from pydantic import BaseModel, ValidationError
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
import json

T = TypeVar('T', bound=BaseModel)

# Type alias for options that can be static or dynamically generated
OptionsProvider = Union[ClaudeAgentOptions, Callable[[], ClaudeAgentOptions]]


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

    def __init__(self, options: OptionsProvider, response_model: Type[T]):
        self._options = options
        self.response_model = response_model

    def _get_options(self) -> ClaudeAgentOptions:
        """Get options, calling the provider if it's a callable."""
        if callable(self._options):
            return self._options()
        return self._options

    async def run(self, query: str) -> AgentTrace[T]:
        async with ClaudeSDKClient(self._get_options()) as client:
            await client.query(query)
            messages = [msg async for msg in client.receive_response()]

            first = messages[0]
            last = messages[-1]

            # Try to parse structured output, gracefully handle failures
            output = None
            parse_error = None
            raw_structured_output = last.structured_output

            if raw_structured_output is not None:
                try:
                    output = self.response_model.model_validate(raw_structured_output)
                except (ValidationError, json.JSONDecodeError, TypeError) as e:
                    parse_error = f"{type(e).__name__}: {str(e)}"
            else:
                parse_error = "No structured output returned (context limit likely exceeded)"

            return AgentTrace(
                uuid=first.data.get('uuid'),
                session_id=last.session_id,
                model=first.data.get('model'),
                tools=first.data.get('tools', []),
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
