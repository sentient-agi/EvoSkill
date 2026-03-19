"""Data types for OSWorld integration."""

from dataclasses import dataclass, field


@dataclass
class OSWorldTask:
    """A single OSWorld benchmark task.

    Attributes:
        id: Unique task identifier (UUID).
        domain: Application domain (e.g., "chrome", "gimp", "libreoffice_calc").
        instruction: Natural language task instruction.
        config: Full example JSON dict, passed to DesktopEnv.reset(task_config=...).
    """
    id: str
    domain: str
    instruction: str
    config: dict = field(repr=False)
