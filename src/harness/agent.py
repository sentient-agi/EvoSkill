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
import hashlib
import json
import logging
import re
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generic, Optional, Type, TypeVar, Union
from pydantic import BaseModel
from .sdk_config import get_sdk, is_claude_sdk, is_codex_sdk, is_goose_sdk

logger = logging.getLogger(__name__)

# Adaptive timeout extension: per-agent.run stderr observer that the executor
# invokes for each stderr line from the bundled CLI. The closure tracks the
# max in-CLI retry attempt seen so far; each new attempt extends the
# asyncio.timeout deadline by 120s (one API_TIMEOUT_MS worth) so a hung
# turn that triggered a retry doesn't unfairly trip the wall clock.
# The contextvar is per-attempt; agent.py resets it for each retry attempt
# and the executor reads it through `_get_run_stderr_observer()`.
_run_stderr_observer: ContextVar[Callable[[str], None] | None] = ContextVar(
    "run_stderr_observer", default=None
)


def _get_run_stderr_observer() -> Callable[[str], None] | None:
    """Public read-side accessor for the adaptive-timeout stderr hook."""
    return _run_stderr_observer.get()


# Partial run state: the executor pushes turn count + last-known cumulative
# cost into these contextvars after each completed turn. When `_run_with_retry`
# hits its wall-clock deadline, we read the values and surface them via
# `agent.run()` as a partial AgentTrace — so the evolver sees "agent burned
# 47 turns and $1.20 before timing out", not the misleading "turns=0, cost=$0".
# Cost is not always knowable mid-stream (the SDK only reports total_cost_usd
# in the terminal ResultMessage), so the executor uses its local pricing
# table to estimate as it goes.
_partial_turn_count: ContextVar[int] = ContextVar("partial_turn_count", default=0)
_partial_cost_usd: ContextVar[float] = ContextVar("partial_cost_usd", default=0.0)


def _push_partial_state(turn_num: int, cost_usd: float = 0.0) -> None:
    """Public write-side hook the executor calls after each turn."""
    _partial_turn_count.set(turn_num)
    if cost_usd > 0:
        _partial_cost_usd.set(cost_usd)

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


def _render_subagent_transcript(agent_id: str, indent: str, tool_result_max_chars: int) -> str:
    """Render a subagent session's turn-by-turn transcript for inline inclusion.

    The bundled Claude binary writes subagent transcripts to
    ~/.claude/projects/<project>/<parent-session>/subagents/agent-<id>.jsonl.
    The parent's `messages` array does NOT contain subagent AssistantMessages
    — so without expanding from JSONL, the evolver only sees the Task tool's
    summary string, never the subagent's 40+ tool-call breakdown.

    Returns an indented multi-line string. Empty if the JSONL can't be found.
    """
    matches = list(Path.home().glob(f".claude/projects/*/*/subagents/agent-{agent_id}.jsonl"))
    if not matches:
        return ""
    sub_jsonl = matches[0]
    try:
        events = [json.loads(line) for line in sub_jsonl.read_text().splitlines() if line.strip()]
    except Exception:
        return ""

    # Pre-pass: collect tool results keyed by tool_use_id
    tool_results: dict[str, str] = {}
    for ev in events:
        if ev.get("type") != "user":
            continue
        content = ev.get("message", {}).get("content", []) or []
        if not isinstance(content, list):
            continue
        for blk in content:
            if not isinstance(blk, dict) or blk.get("type") != "tool_result":
                continue
            tu_id = blk.get("tool_use_id") or blk.get("id")
            if not tu_id:
                continue
            rc = blk.get("content", "")
            if isinstance(rc, list):
                rs = "\n".join(c.get("text", str(c)) if isinstance(c, dict) else str(c) for c in rc)
            else:
                rs = str(rc) if rc is not None else ""
            if len(rs) > tool_result_max_chars:
                rs = rs[:tool_result_max_chars] + f"\n...[truncated, {len(rs) - tool_result_max_chars:,} chars]"
            tool_results[str(tu_id)] = rs

    out: list[str] = []
    turn_n = 0
    for ev in events:
        if ev.get("type") != "assistant":
            continue
        msg = ev.get("message", {})
        content = msg.get("content", []) or []
        if not isinstance(content, list):
            continue
        turn_n += 1
        model = msg.get("model", "?") or "?"
        out.append(f"{indent}--- Turn {turn_n} [{model}] ---")
        for blk in content:
            if not isinstance(blk, dict):
                continue
            bt = blk.get("type")
            if bt == "thinking":
                t = (blk.get("thinking") or "").strip()
                if t:
                    out.append(f"{indent}(thinking): {t}")
            elif bt == "text":
                t = (blk.get("text") or "").strip()
                if t:
                    out.append(f"{indent}text: {t}")
            elif bt == "tool_use":
                name = blk.get("name", "?") or "?"
                inp = blk.get("input", {})
                try:
                    inp_str = json.dumps(inp, default=str, ensure_ascii=False)
                except Exception:
                    inp_str = str(inp)
                out.append(f"{indent}(tool.{name}): {inp_str}")
                tu_id = blk.get("id") or blk.get("tool_use_id")
                if tu_id and str(tu_id) in tool_results:
                    out.append(f"{indent}(tool.result): {tool_results[str(tu_id)]}")
    return "\n".join(out)


def _render_turn_transcript(messages: list[Any], tool_result_max_chars: int) -> str:
    """Render messages as a turn-by-turn transcript with subagent hierarchy.

    Subagent boundaries are detected by model-switch (model differs from the
    primary agent's model) and bracketed with ▶▶▶ SUBAGENT START / ◀◀◀
    SUBAGENT END markers so the evolver can see who delegated to whom and
    when control returned. Subagent content is indented 4 spaces.
    """
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

    # Second pass: render each AssistantMessage as a turn, tracking subagent
    # entry/exit via model-switch. Dedup repeat tool results (sha256 of content)
    # so the transcript doesn't balloon when the solver re-reads the same file.
    out_lines: list[str] = []
    turn_num = 0
    primary_model: str | None = None
    in_subagent = False
    pending_task_desc: str | None = None  # From most recent Task tool call
    indent = ""
    # Map content-digest -> (first_turn_num, first_tool_name, char_count)
    seen_results: dict[str, tuple[int, str, int]] = {}

    for msg in messages:
        if not isinstance(msg, AssistantMessage):
            continue
        turn_num += 1
        msg_model = (getattr(msg, "model", "") or "").strip()

        # Establish primary model from first turn
        if not primary_model and msg_model:
            primary_model = msg_model

        # Subagent entry: model differs from primary
        if primary_model and msg_model and msg_model != primary_model and not in_subagent:
            desc = pending_task_desc or "(subtask)"
            out_lines.append(f"\n▶▶▶ SUBAGENT START: {desc}   [{msg_model}]")
            in_subagent = True
            indent = "    "
            pending_task_desc = None

        # Subagent exit: model returns to primary
        if in_subagent and msg_model == primary_model:
            out_lines.append(f"◀◀◀ SUBAGENT END — back to [{primary_model}]\n")
            in_subagent = False
            indent = ""

        model_tag = f" [{msg_model}]" if msg_model else ""
        out_lines.append(f"\n{indent}--- Turn {turn_num}{model_tag} ---")

        for block in getattr(msg, "content", []) or []:
            if ThinkingBlock is not None and isinstance(block, ThinkingBlock):
                text = (getattr(block, "thinking", "") or "").strip()
                if text:
                    out_lines.append(f"{indent}(thinking): {text}")
            elif isinstance(block, TextBlock):
                text = (getattr(block, "text", "") or "").strip()
                if text:
                    out_lines.append(f"{indent}text: {text}")
            elif isinstance(block, ToolUseBlock):
                name = getattr(block, "name", "?")
                try:
                    inp = json.dumps(getattr(block, "input", None), default=str, ensure_ascii=False)
                except Exception:
                    inp = str(getattr(block, "input", None))
                out_lines.append(f"{indent}(tool.{name}): {inp}")
                # If this is a Task dispatch, remember its description so we
                # can label the upcoming SUBAGENT START block.
                if name == "Task":
                    task_input = getattr(block, "input", None)
                    if isinstance(task_input, dict):
                        pending_task_desc = task_input.get("description") or task_input.get("prompt", "")[:60] or "(subtask)"
                tool_use_id = getattr(block, "id", None) or getattr(block, "tool_use_id", None)
                if tool_use_id and str(tool_use_id) in tool_results:
                    result_text = tool_results[str(tool_use_id)]
                    # If this is a Task tool_result, inline-expand the
                    # subagent's session as a nested block so the evolver can
                    # see the subagent's turn-by-turn behavior — not just its
                    # final summary string.
                    if name == "Task":
                        agent_id_match = re.search(r"agentId:\s*([a-f0-9]+)", result_text)
                        if agent_id_match:
                            sub_indent = indent + "    "
                            sub_transcript = _render_subagent_transcript(
                                agent_id_match.group(1), sub_indent, tool_result_max_chars
                            )
                            if sub_transcript:
                                out_lines.append(
                                    f"{indent}▶▶▶ SUBAGENT START: {pending_task_desc or '(subtask)'}"
                                )
                                out_lines.append(sub_transcript)
                                out_lines.append(
                                    f"{indent}◀◀◀ SUBAGENT END — back to [{primary_model}]"
                                )
                                pending_task_desc = None
                    # Only dedup non-trivial results.
                    if len(result_text) >= 120:
                        digest = hashlib.sha256(
                            result_text.encode("utf-8", errors="replace")
                        ).hexdigest()[:16]
                        prior = seen_results.get(digest)
                        if prior is not None:
                            prior_turn, prior_name, prior_len = prior
                            out_lines.append(
                                f"{indent}(tool.result): [identical to turn {prior_turn}'s "
                                f"{prior_name} result — {prior_len:,} chars omitted]"
                            )
                            continue
                        seen_results[digest] = (turn_num, name, len(result_text))
                    out_lines.append(f"{indent}(tool.result): {result_text}")
            else:
                out_lines.append(f"{indent}({type(block).__name__}): {str(block)[:500]}")

    # If we ended inside a subagent block (solver didn't fully return), close it
    if in_subagent:
        out_lines.append(f"◀◀◀ SUBAGENT END — transcript ended inside subagent\n")

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

    TIMEOUT_SECONDS = 720    # 12 minutes per attempt — base budget for productive work
    # The wall-clock budget extends ADAPTIVELY by +120s per observed in-CLI
    # retry (see _run_stderr_observer below). API_TIMEOUT_MS=120s ×
    # CLAUDE_CODE_MAX_RETRIES=5 (set in executor.py) bounds retries at K≤5,
    # so the worst-case effective budget is 12 + 2×5 = 22 min on a single
    # severely network-stuck turn — but healthy runs stay at 12 min and
    # don't pay for retry capacity they didn't use.
    MAX_RETRIES = 2          # Total attempts before giving up
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

    async def _execute_query(self, query: str, tag: str | None = None) -> list[Any]:
        """Execute a single query by delegating to the active SDK's executor."""
        options = self._get_options()

        sdk = get_sdk()
        if sdk == "claude":
            from .claude import executor as _claude_executor
            return await _claude_executor.execute_query(options, query, agent_name=self.name, tag=tag)
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

    async def _run_with_retry(self, query: str, tag: str | None = None) -> list[Any]:
        """Execute query with timeout, retrying only on subprocess-level errors.

        TimeoutError = the agent legitimately exhausted its 12-min wall-clock
        budget (too many turns / heavy thinking). Network blips are already
        handled inside the bundled CLI's own per-request retry loop
        (API_TIMEOUT_MS + CLAUDE_CODE_MAX_RETRIES, set in the executor), so a
        timeout here is a real over-budget condition — restarting from turn 0
        won't recover, just burns the same budget again. Surface immediately.

        Other exceptions (subprocess crash, SDK init error, etc.) still get
        one retry with backoff since those are usually transient and a fresh
        process tends to recover.
        """
        last_error: Exception | None = None
        backoff = self.INITIAL_BACKOFF

        # Detects in-CLI retry banners. Bundled CLI emits "(attempt N/K)"
        # when its API client retries a hung HTTP request. We dedupe by
        # only acting when the attempt number we see is strictly higher
        # than any previous one (multiple stderr lines per retry are common).
        attempt_pat = re.compile(r"\(attempt\s+(\d+)/\d+\)", re.IGNORECASE)

        for attempt in range(self.max_retries):
            # Reset partial-state contextvars at the start of every attempt
            # so a previous attempt's stale numbers don't leak into a fresh
            # run's timeout report.
            _partial_turn_count.set(0)
            _partial_cost_usd.set(0.0)
            try:
                async with asyncio.timeout(self.timeout_seconds) as tm:
                    # Publish the active timeout to wait_if_paused so a
                    # cooperative pause can refund the wall-clock budget
                    # consumed during the pause. See src/harness/pause.py.
                    from src.harness.pause import active_timeout as _active_tm
                    _tm_token = _active_tm.set(tm)
                    max_seen = [1]  # closed-over per-attempt state

                    def _observe_stderr(line: str) -> None:
                        m = attempt_pat.search(line)
                        if not m:
                            return
                        n = int(m.group(1))
                        if n <= max_seen[0]:
                            return
                        max_seen[0] = n
                        current = tm.when()
                        if current is None:
                            return
                        # +120s per new attempt = matches API_TIMEOUT_MS so
                        # the agent gets back the time the retry consumed.
                        tm.reschedule(current + 120.0)
                        logger.warning(
                            f"Adaptive timeout: in-CLI retry attempt {n} "
                            f"detected — extending wall-clock budget +120s"
                        )

                    obs_token = _run_stderr_observer.set(_observe_stderr)
                    try:
                        return await self._execute_query(query, tag=tag)
                    finally:
                        _run_stderr_observer.reset(obs_token)
                        _active_tm.reset(_tm_token)
            except asyncio.TimeoutError:
                # Surface partial state in the error message so callers
                # (and the evolver) see "agent burned N turns and $X" rather
                # than the misleading "turns=0, cost=$0" zero-fill.
                n_turns = _partial_turn_count.get()
                spent = _partial_cost_usd.get()
                logger.warning(
                    f"Query timed out after {self.timeout_seconds}s — "
                    f"agent over budget, not retrying (turns={n_turns}, cost=${spent:.4f})"
                )
                raise TimeoutError(
                    f"Query timed out after {self.timeout_seconds}s "
                    f"(burned {n_turns} turns, ${spent:.4f} before timeout)"
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

    async def run(self, query: str, tag: str | None = None) -> AgentTrace[T]:
        """Run the agent on a query and return a structured AgentTrace.

        This is the main entry point. It:
            1. Calls _run_with_retry to get raw SDK messages
            2. Delegates response parsing to the active SDK's executor
            3. Returns an AgentTrace with all metadata + parsed output

        Args:
            query: The question for the agent.
            tag: Optional human-readable label (e.g. "train 3/8") shown in
                 stdout prints and attached to spans. Defaults to a sha256
                 hash of the query when omitted.
        """
        try:
            messages = await self._run_with_retry(query, tag=tag)
        except TimeoutError as e:
            # Convert the wall-clock-budget TimeoutError into a partial
            # AgentTrace so callers see real turn / cost numbers in the
            # failure record. The evolver renders these into the "Failure N
            # (turns=X, cost=$Y)" header — without this conversion the
            # exception path in runner.py zero-filled the placeholder, and
            # the evolver was misled into treating timeouts as zero-work
            # parse failures rather than the over-budget runs they actually
            # are.
            n_turns = _partial_turn_count.get()
            spent = _partial_cost_usd.get()
            err_msg = str(e)
            return AgentTrace(
                duration_ms=int(self.timeout_seconds * 1000),
                total_cost_usd=spent,
                num_turns=n_turns,
                usage={},
                result=err_msg,
                is_error=True,
                output=None,
                parse_error=err_msg,
                messages=[],
            )

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
