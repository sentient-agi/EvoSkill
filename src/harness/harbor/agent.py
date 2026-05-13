"""HarborAgent: drop-in replacement for the base Agent that runs Harbor trials.

For each query (a Harbor task id), this agent:

  1. Resolves the task id to a local digest directory (populated by harbor_loader).
  2. Mounts EvoSkill's current `.claude/skills/` into the container at a known
     path via `--mounts-json`.
  3. Tells the inner agent (claude-code, openhands, ...) to read skills from
     that mount via `--ak skills_dir=<mount>`.
  4. Spawns `harbor run` as a subprocess.
  5. Reads the trial's verifier reward (reward.txt or reward.json) from the
     emitted jobs directory.
  6. Returns an AgentTrace whose output.final_answer is a JSON envelope:
        {"reward": <float>, "task_id": <id>, "exit_status": <str>}

The harbor scorer (src.evaluation.harbor_scorer) reads that envelope back.

Why a JSON envelope instead of returning a float directly: EvoSkill's loop
expects a string in `output.final_answer` and computes scores via a scorer
function. We keep that contract intact.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from src.harness.agent import Agent, AgentTrace
from src.schemas import AgentResponse

logger = logging.getLogger(__name__)


class HarborRunError(RuntimeError):
    pass


class HarborAgent(Agent[AgentResponse]):
    """Run a Harbor task as an EvoSkill agent."""

    def __init__(
        self,
        *,
        project_root: Path,
        skills_source_dir: Path,
        inner_agent: str,
        inner_model: str,
        env: str = "docker",
        n_concurrent: int = 1,
        timeout_seconds: int = 1800,
        max_retries: int = 1,
        jobs_dir: Path | None = None,
        container_skills_path: str = "/skills",
        timeout_multiplier: float = 1.0,
        extra_args: list[str] | None = None,
    ):
        # The base Agent class expects options + response_model. We pass placeholders
        # because we override run() entirely and never delegate to an SDK executor.
        super().__init__(
            options=lambda: {},
            response_model=AgentResponse,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
        self.project_root = Path(project_root)
        self.skills_source_dir = Path(skills_source_dir)
        self.inner_agent = inner_agent
        self.inner_model = inner_model
        self.env = env
        self.n_concurrent = max(1, int(n_concurrent))
        self.jobs_dir = (
            Path(jobs_dir)
            if jobs_dir
            else (self.project_root / ".evoskill" / "harbor_jobs")
        )
        self.container_skills_path = container_skills_path
        self.timeout_multiplier = float(timeout_multiplier)
        self.extra_args = list(extra_args or [])

    # ------------------------------------------------------------------ public
    async def run(self, query: str) -> AgentTrace[AgentResponse]:
        """Execute the Harbor task identified by `query` and return a trace."""
        from src.api.harbor_loader import resolve_task_dir

        started_ms = time.monotonic()
        task_dir = resolve_task_dir(query)
        if task_dir is None:
            return self._error_trace(
                query, started_ms, f"unknown harbor task id: {query!r}"
            )
        if not (task_dir / "task.toml").is_file():
            return self._error_trace(
                query, started_ms, f"no task.toml in {task_dir}"
            )

        job_name = f"evoskill-{uuid.uuid4().hex[:12]}"
        job_dir = self.jobs_dir / job_name
        job_dir.parent.mkdir(parents=True, exist_ok=True)

        cmd = self._build_command(task_dir, job_name)
        env_vars = self._build_env()

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env_vars,
            )
        except FileNotFoundError as exc:
            return self._error_trace(
                query,
                started_ms,
                f"`harbor` CLI not found on PATH: {exc}. "
                "Install with: uv tool install harbor",
            )

        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout_seconds,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return self._error_trace(
                query,
                started_ms,
                f"harbor run timed out after {self.timeout_seconds}s",
            )

        stdout = stdout_b.decode("utf-8", errors="replace")
        stderr = stderr_b.decode("utf-8", errors="replace")
        exit_code = proc.returncode or 0

        reward, exit_status = self._read_reward(job_dir)
        if reward is None:
            # Surface harbor's stderr tail so failures are diagnosable upstream.
            tail = stderr[-4000:] if stderr else stdout[-4000:]
            return self._error_trace(
                query,
                started_ms,
                f"harbor exit={exit_code}; no reward found under {job_dir}.\n"
                f"--- harbor stderr/stdout tail ---\n{tail}",
            )

        envelope = {
            "reward": float(reward),
            "task_id": query,
            "exit_status": exit_status,
            "job_dir": str(job_dir),
        }
        duration_ms = int((time.monotonic() - started_ms) * 1000)
        return AgentTrace(
            uuid=job_name,
            session_id=job_name,
            model=f"harbor:{self.inner_agent}:{self.inner_model}",
            tools=[],
            duration_ms=duration_ms,
            total_cost_usd=0.0,
            num_turns=1,
            usage={},
            result=stdout[-2000:],
            is_error=exit_code != 0 and reward == 0.0,
            output=AgentResponse(
                final_answer=json.dumps(envelope),
                reasoning=f"harbor:{self.inner_agent} task={query}",
            ),
            messages=[],
        )

    # ------------------------------------------------------------ command build
    def _build_command(self, task_dir: Path, job_name: str) -> list[str]:
        skills_mount = self._build_skills_mount()
        cmd: list[str] = [
            "harbor",
            "run",
            "-p",
            str(task_dir),
            "-a",
            self.inner_agent,
            "-m",
            self.inner_model,
            "-e",
            self.env,
            "-n",
            str(self.n_concurrent),
            "--quiet",
            "--yes",
            "--job-name",
            job_name,
            "--jobs-dir",
            str(self.jobs_dir),
            "--max-retries",
            "0",
            "--timeout-multiplier",
            f"{self.timeout_multiplier}",
        ]
        if skills_mount is not None:
            cmd.extend(["--mounts-json", json.dumps([skills_mount])])
            cmd.extend(["--ak", f"skills_dir={self.container_skills_path}"])
        cmd.extend(self.extra_args)
        return cmd

    def _build_skills_mount(self) -> dict[str, Any] | None:
        if not self.skills_source_dir.is_dir():
            return None
        try:
            has_content = any(self.skills_source_dir.iterdir())
        except OSError:
            has_content = False
        if not has_content:
            return None
        return {
            "type": "bind",
            "source": str(self.skills_source_dir.resolve()),
            "target": self.container_skills_path,
            "read_only": True,
        }

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        # Pass the current working dir as context but don't leak project secrets.
        # The inner agent (claude-code, etc.) needs its own provider key in env.
        return env

    # -------------------------------------------------------------- reward read
    def _read_reward(self, job_dir: Path) -> tuple[float | None, str]:
        """Return (reward, exit_status). reward is None if no file was found."""
        if not job_dir.is_dir():
            return None, "no_job_dir"

        # Prefer the structured trial result.json (TrialResult schema).
        for result_path in sorted(job_dir.rglob("result.json")):
            try:
                payload = json.loads(result_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            verifier = payload.get("verifier_result") or {}
            rewards = verifier.get("rewards") or {}
            if rewards:
                # Pick "reward" if present, else the first value.
                if "reward" in rewards:
                    return float(rewards["reward"]), "verified"
                first = next(iter(rewards.values()))
                try:
                    return float(first), "verified"
                except (TypeError, ValueError):
                    pass

        # Fallback: read reward.txt directly from any verifier dir.
        for reward_path in sorted(job_dir.rglob("verifier/reward.txt")):
            try:
                return float(reward_path.read_text().strip()), "verified"
            except (OSError, ValueError):
                continue

        # Reward.json fallback
        for reward_json in sorted(job_dir.rglob("verifier/reward.json")):
            try:
                payload = json.loads(reward_json.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and payload:
                first = next(iter(payload.values()))
                try:
                    return float(first), "verified"
                except (TypeError, ValueError):
                    continue

        return None, "no_reward"

    # ------------------------------------------------------------- error helper
    def _error_trace(
        self, query: str, started_ms: float, message: str
    ) -> AgentTrace[AgentResponse]:
        envelope = {
            "reward": 0.0,
            "task_id": query,
            "exit_status": "error",
            "error": message,
        }
        duration_ms = int((time.monotonic() - started_ms) * 1000)
        return AgentTrace(
            uuid="",
            session_id="",
            model=f"harbor:{self.inner_agent}:{self.inner_model}",
            tools=[],
            duration_ms=duration_ms,
            total_cost_usd=0.0,
            num_turns=0,
            usage={},
            result=message,
            is_error=True,
            output=AgentResponse(
                final_answer=json.dumps(envelope),
                reasoning=f"harbor:{self.inner_agent} task={query}",
            ),
            parse_error=None,
            messages=[],
        )

    # Optional cleanup hook (not currently called, but useful for tests).
    def cleanup_jobs_dir(self) -> None:
        if self.jobs_dir.is_dir():
            shutil.rmtree(self.jobs_dir, ignore_errors=True)
