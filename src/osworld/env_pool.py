"""VM environment pool for concurrent OSWorld evaluation."""

import asyncio
import copy
import logging
import sys
import time
from pathlib import Path
from typing import Any

from .types import OSWorldTask

logger = logging.getLogger(__name__)


class EnvPool:
    """Manages a pool of DesktopEnv instances for concurrent task evaluation.

    Each DesktopEnv wraps a VM. Tasks are dispatched to available VMs,
    and the pool handles allocation/release automatically.

    Args:
        num_envs: Number of parallel VM environments.
        env_kwargs: Keyword arguments passed to DesktopEnv constructor.
        osworld_root: Path to OSWorld project root (for imports).
    """

    def __init__(
        self,
        num_envs: int,
        env_kwargs: dict[str, Any],
        osworld_root: str | Path,
    ):
        self.num_envs = num_envs
        self.env_kwargs = env_kwargs
        self.osworld_root = str(osworld_root)

        # Add OSWorld to import path
        if self.osworld_root not in sys.path:
            sys.path.insert(0, self.osworld_root)

        from desktop_env.desktop_env import DesktopEnv

        logger.info(f"Creating {num_envs} VM environment(s)...")
        self.envs: list[DesktopEnv] = []
        for i in range(num_envs):
            env = DesktopEnv(**env_kwargs)
            self.envs.append(env)
            logger.info(f"  VM {i} created")

        # Track availability via async queue
        self._available: asyncio.Queue[int] = asyncio.Queue()
        for i in range(num_envs):
            self._available.put_nowait(i)

    async def run_task(
        self,
        task: OSWorldTask,
        agent: Any,
        settle_time: float = 20.0,
        setup_time: float = 60.0,
    ) -> tuple[dict[str, Any], float]:
        """Run a single task on an available VM.

        Acquires a VM, resets it to the task snapshot, runs the agent
        autonomously, evaluates, and releases the VM.

        Args:
            task: The OSWorld task to execute.
            agent: ClaudeCodeAgent instance.
            settle_time: Seconds to wait before evaluation.
            setup_time: Seconds to wait after env.reset().

        Returns:
            Tuple of (trace_data dict, score float).
        """
        idx = await self._available.get()
        env = self.envs[idx]
        try:
            logger.info(f"[VM {idx}] Running task {task.domain}/{task.id}")

            # Reset environment with retry (VMware can race after snapshot revert)
            for attempt in range(3):
                try:
                    await asyncio.to_thread(env.reset, task_config=task.config)
                    break
                except Exception as reset_err:
                    if attempt < 2:
                        logger.warning(
                            f"[VM {idx}] env.reset attempt {attempt + 1} failed: {reset_err}. "
                            f"Retrying in 10s..."
                        )
                        await asyncio.to_thread(time.sleep, 10)
                    else:
                        raise
            await asyncio.to_thread(time.sleep, setup_time)

            # Create per-task agent copy to avoid shared state across concurrent tasks
            task_agent = copy.copy(agent)
            server_port = getattr(env, "server_port", 5000)
            task_agent.reset(vm_ip=env.vm_ip, server_port=server_port)

            # Agent runs autonomously
            trace_data = await task_agent.run(task.instruction)

            # Wait for environment to settle, then evaluate
            await asyncio.to_thread(time.sleep, settle_time)
            score = await asyncio.to_thread(env.evaluate)

            logger.info(
                f"[VM {idx}] Task {task.domain}/{task.id}: score={score:.2f}"
            )
            return trace_data, score

        except Exception as e:
            logger.error(f"[VM {idx}] Task {task.domain}/{task.id} failed: {e}")
            return {"is_error": True, "result": str(e), "messages": []}, 0.0

        finally:
            self._available.put_nowait(idx)

    async def run_batch(
        self,
        tasks: list[OSWorldTask],
        agent: Any,
        settle_time: float = 20.0,
        setup_time: float = 60.0,
    ) -> list[tuple[dict[str, Any], float]]:
        """Run a batch of tasks concurrently (bounded by pool size).

        Args:
            tasks: List of tasks to execute.
            agent: ClaudeCodeAgent instance.
            settle_time: Seconds to wait before evaluation.
            setup_time: Seconds to wait after env.reset().

        Returns:
            List of (trace_data, score) tuples, one per task.
        """
        coros = [
            self.run_task(task, agent, settle_time, setup_time)
            for task in tasks
        ]
        return await asyncio.gather(*coros)

    def close(self):
        """Shut down all VM environments."""
        for i, env in enumerate(self.envs):
            try:
                env.close()
                logger.info(f"VM {i} closed")
            except Exception as e:
                logger.error(f"Error closing VM {i}: {e}")
