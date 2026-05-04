"""Wraps HALO's AppWorld runner as an EvoSkill Agent.

Calls HALO's run_agent_on_tasks() for each task, reads results from disk,
and packages them into AgentTrace — making HALO's proven pipeline
compatible with EvoSkill's SelfImprovingLoop.

The "skill" evolved by EvoSkill is the system prompt (instructions.txt).
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Type, TypeVar

from pydantic import BaseModel

from src.harness.agent import Agent, AgentTrace
from src.schemas import AgentResponse

T = TypeVar("T", bound=BaseModel)

SEPARATOR = ":::"


def parse_task_id(question: str) -> tuple[str, str]:
    """Parse 'task_id:::instruction' into (task_id, instruction)."""
    if not question or SEPARATOR not in question:
        raise ValueError(f"Question must contain '{SEPARATOR}': {question[:80] if question else ''}")
    task_id, instruction = question.split(SEPARATOR, 1)
    return task_id.strip(), instruction.strip()


def read_answer_from_jsonl(supervisor_jsonl: Path) -> str | None:
    """Extract the agent's submitted answer from supervisor.jsonl."""
    if not supervisor_jsonl.exists():
        return None
    content = supervisor_jsonl.read_text().strip()
    if not content:
        return None

    for line in content.splitlines():
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        sql = data[0] if isinstance(data, list) and len(data) > 0 else ""
        params = data[1] if isinstance(data, list) and len(data) > 1 else []
        if "answer" in sql and len(params) >= 3:
            answer = params[2]
            if answer is None:
                return None
            answer = str(answer)
            if answer.startswith('"') and answer.endswith('"'):
                answer = answer[1:-1]
            return answer
    return None


def read_trace_summary(lm_calls_jsonl: Path) -> str:
    """Read lm_calls.jsonl and produce a human-readable trace summary.

    HALO's lm_calls.jsonl is JSONL format (one JSON object per line).
    The file includes demo messages first, then the real task execution.
    We skip demo entries and only summarize the real task.
    """
    if not lm_calls_jsonl.exists():
        return ""

    content = lm_calls_jsonl.read_text().strip()
    if not content:
        return ""

    # Parse JSONL (one JSON per line) or single JSON array
    entries: list[dict] = []
    try:
        # Try as single JSON array first (test fixtures use this)
        parsed = json.loads(content)
        if isinstance(parsed, list):
            entries = [e for e in parsed if isinstance(e, dict)]
        else:
            entries = [parsed] if isinstance(parsed, dict) else []
    except json.JSONDecodeError:
        # JSONL format (one JSON per line) — this is HALO's real format
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if isinstance(entry, dict):
                    entries.append(entry)
            except json.JSONDecodeError:
                continue

    if not entries:
        return ""

    # Find where the real task starts (after demo messages).
    real_start = 0
    for i, entry in enumerate(entries):
        content_str = str(entry.get("content", ""))
        if "Real Task Instruction" in content_str or "Disclaimer: This is a real task" in content_str:
            real_start = i + 1
            break

    # Extract only the real task's tool calls
    real_entries = entries[real_start:]
    lines = []
    tool_call_count = 0
    for entry in real_entries:
        entry_type = entry.get("type", "")

        if entry_type == "function_call":
            tool_call_count += 1
            name = entry.get("name", "?")
            args = entry.get("arguments", "{}")
            if len(str(args)) > 150:
                args = str(args)[:150] + "..."
            lines.append(f"Step {tool_call_count}: {name}({args})")
        elif entry_type == "function_call_output":
            output = entry.get("output", "")
            if len(output) > 300:
                output = output[:300] + "..."
            lines.append(f"  → {output}")

    summary = "\n".join(lines)
    if tool_call_count:
        summary = f"Agent made {tool_call_count} tool calls:\n\n{summary}"
    return summary


def read_eval_result(eval_json: Path) -> tuple[bool, float]:
    """Read AppWorld's evaluation JSON and return (passed, score)."""
    if not eval_json.exists():
        return False, 0.0

    try:
        data = json.loads(eval_json.read_text())
    except (json.JSONDecodeError, OSError):
        return False, 0.0

    metrics = data.get("aggregate", {})
    tgc = metrics.get("task_goal_completion", 0.0)
    passed = tgc >= 100.0
    score = tgc / 100.0
    return passed, score


def build_agent_trace(
    task_id: str,
    answer: str | None,
    trace_summary: str,
    passed: bool,
    score: float,
    num_turns: int = 0,
) -> AgentTrace[AgentResponse]:
    """Construct an AgentTrace from HALO's disk outputs."""
    final_answer = answer if answer is not None else "[NO ANSWER]"
    reasoning = trace_summary if trace_summary else f"Task {task_id}: no trace available"

    return AgentTrace(
        uuid=str(uuid.uuid4()),
        session_id=f"halo-{task_id}",
        model="halo-runner",
        tools=[],
        duration_ms=0,
        total_cost_usd=0.0,
        num_turns=num_turns,
        usage={},
        result=final_answer,
        is_error=not passed,
        output=AgentResponse(final_answer=final_answer, reasoning=reasoning),
        parse_error=None,
        raw_structured_output={"final_answer": final_answer, "reasoning": reasoning},
        messages=[],
    )


class HALOAgent(Agent[AgentResponse]):
    """EvoSkill Agent that delegates to HALO's AppWorld runner.

    Uses a serialization lock so that concurrent run() calls from
    asyncio.gather() are executed one at a time — preventing AppWorld
    server conflicts. Each task gets its own AppWorld.initializer()
    context with fresh servers.
    """

    def __init__(
        self,
        halo_root: str | Path,
        model: str | None = None,
        provider: str | None = None,
        experiment_name: str | None = None,
        max_steps: int = 50,
    ) -> None:
        from examples.appworld2.scripts.build_config import get_default_model, get_default_experiment_name
        self._halo_root = Path(halo_root)
        self._model = model or get_default_model()
        self._provider = provider
        self._experiment_name = experiment_name or get_default_experiment_name()
        self._max_steps = max_steps
        self.response_model = AgentResponse
        # Serialize all run() calls — prevents AppWorld server conflicts
        self._lock = threading.Lock()

    def _get_options(self) -> dict:
        """Return a minimal options dict for ProgramManager compatibility."""
        appworld2_root = Path(__file__).resolve().parent.parent
        prompt_path = appworld2_root / ".claude" / "prompts" / "instructions.txt"
        prompt_text = prompt_path.read_text() if prompt_path.exists() else ""
        return {
            "system": prompt_text,
            "model": self._model,
        }

    def _get_output_dir(self, task_id: str) -> Path:
        return self._halo_root / "experiments" / "outputs" / self._experiment_name / "tasks" / task_id

    def _get_eval_path(self, task_id: str) -> Path:
        return (
            self._halo_root / "experiments" / "outputs" / self._experiment_name
            / "evaluations" / f"on_only_{task_id}.json"
        )

    def _build_config(self) -> dict[str, Any]:
        """Build runner_config matching HALO's setup."""
        from examples.appworld2.scripts.build_config import build_runner_config
        return build_runner_config(model=self._model, provider=self._provider)

    def _sync_prompt_to_halo(self) -> None:
        """Copy the evolved prompt from .claude/ to where HALO reads it."""
        import shutil
        appworld2_root = Path(__file__).resolve().parent.parent
        src = appworld2_root / ".claude" / "prompts" / "instructions.txt"
        dst = appworld2_root / "experiments" / "prompts" / "function_calling_agent" / "instructions.txt"
        if src.exists():
            shutil.copy2(src, dst)

    def _run_halo_and_evaluate(self, task_id: str, config: dict | None) -> None:
        """Run HALO's agent AND evaluate inside the AppWorld context.

        Serialized via self._lock — only one task runs at a time.
        Each call gets its own AppWorld.initializer() with fresh servers.
        """
        from copy import deepcopy
        from appworld import AppWorld

        with self._lock:
            config = config or self._build_config()
            cfg = deepcopy(config)
            api_predictor_config = cfg.pop("api_predictor")
            agent_config = cfg.pop("agent")
            appworld_config = cfg.pop("appworld")
            logger_config = cfg.pop("logger")
            _ = cfg.pop("dataset")

            with AppWorld.initializer(
                update_defaults=True,
                experiment_name=self._experiment_name,
                **appworld_config,
            ):
                from appworld_agents.code.openai_agents.run import run_agent_on_tasks
                asyncio.run(
                    run_agent_on_tasks(
                        experiment_name=self._experiment_name,
                        task_ids=[task_id],
                        api_predictor_config=api_predictor_config,
                        agent_config=agent_config,
                        appworld_config=appworld_config,
                        logger_config=logger_config,
                        num_processes=1,
                        process_index=0,
                    )
                )

                try:
                    from appworld.evaluator import evaluate_dataset
                    evaluate_dataset(
                        experiment_name=self._experiment_name,
                        dataset_name=None,
                        task_id=task_id,
                        suppress_errors=True,
                    )
                except Exception:
                    pass

    async def run(self, question: str) -> AgentTrace[AgentResponse]:
        """Run a single AppWorld task via HALO's pipeline."""
        task_id, instruction = parse_task_id(question)

        # Sync the evolved prompt to where HALO reads it
        self._sync_prompt_to_halo()

        # Set env vars
        os.environ["APPWORLD_ROOT"] = str(self._halo_root)
        os.environ.setdefault("OPENAI_API_KEY", "unused-for-litellm")

        # Run in a thread (HALO calls asyncio.run() internally).
        # The lock inside _run_halo_and_evaluate ensures only one task
        # runs at a time, even if asyncio.gather() launches multiple.
        import concurrent.futures
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            await loop.run_in_executor(
                pool, self._run_halo_and_evaluate, task_id, None
            )

        # Read results from disk
        output_dir = self._get_output_dir(task_id)
        answer = read_answer_from_jsonl(output_dir / "dbs" / "supervisor.jsonl")
        trace_summary = read_trace_summary(output_dir / "logs" / "lm_calls.jsonl")

        # Read official evaluation result
        passed, score = read_eval_result(self._get_eval_path(task_id))

        # Fallback: if official evaluation wasn't written, compare answers
        if score == 0.0 and answer is not None:
            gt_path = self._halo_root / "data" / "tasks" / task_id / "ground_truth" / "answer.json"
            if gt_path.exists():
                import json as _json
                gt_answer = _json.loads(gt_path.read_text())
                from appworld.common.evaluation import do_answers_match
                if do_answers_match(answer, gt_answer):
                    passed = True
                    score = 1.0
                elif isinstance(gt_answer, str) and "," in gt_answer:
                    gt_items = sorted(s.strip().lower() for s in gt_answer.split(",") if s.strip())
                    pred_items = sorted(s.strip().lower() for s in answer.split(",") if s.strip())
                    if gt_items == pred_items:
                        passed = True
                        score = 1.0

        # Count tool calls from trace
        num_turns = trace_summary.count("Step ") if trace_summary else 0

        return build_agent_trace(
            task_id=task_id,
            answer=answer,
            trace_summary=trace_summary,
            passed=passed,
            score=score,
            num_turns=num_turns,
        )
