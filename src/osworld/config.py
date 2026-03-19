"""Configuration for the OSWorld self-improving loop."""

from dataclasses import dataclass
from typing import Optional

from src.loop.config import LoopConfig


@dataclass
class OSWorldLoopConfig(LoopConfig):
    """Extended configuration for OSWorld evaluation.

    Inherits all LoopConfig fields (max_iterations, frontier_size,
    evolution_mode, etc.) and adds OSWorld-specific settings.
    """

    # VM pool
    num_envs: int = 1
    provider_name: str = "vmware"
    path_to_vm: Optional[str] = None
    headless: bool = False

    # Task execution
    max_steps_hint: int = 15
    setup_time: float = 60.0     # seconds to wait after env.reset()
    settle_time: float = 20.0    # seconds to wait before env.evaluate()
    failure_threshold: float = 1.0  # score < this = failure (1.0 = only perfect passes)

    # Agent
    agent_model: str = "claude-sonnet-4-5-20250929"
    agent_timeout: int = 1200  # seconds

    # Paths
    osworld_root: str = ""       # path to OSWorld project
    dataset_path: str = ""       # path to task dataset JSON (test_nogdrive.json)
    examples_dir: str = ""       # path to evaluation_examples/examples/
