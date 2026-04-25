"""Remote execution backend interface."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from src.cli.config import ProjectConfig


@dataclass
class RunInfo:
    """Persisted to .evoskill/remote_run.json to track an active remote run."""

    run_id: str
    target: str  # "daytona"
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "running"
    # Backend-specific fields
    extra: dict = field(default_factory=dict)

    def save(self, project_root: Path) -> None:
        path = project_root / ".evoskill" / "remote_run.json"
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, project_root: Path) -> RunInfo | None:
        path = project_root / ".evoskill" / "remote_run.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text())
        return cls(**data)

    @classmethod
    def clear(cls, project_root: Path) -> None:
        path = project_root / ".evoskill" / "remote_run.json"
        if path.exists():
            path.unlink()


class RemoteBackend(ABC):
    """Abstract interface for remote execution backends.

    All backends must implement these methods. The CLI calls them
    in order: setup → upload → run, then status/logs/stop as needed,
    then download.
    """

    @abstractmethod
    def setup(self, cfg: ProjectConfig) -> None:
        """One-time setup: validate credentials, create resources."""

    @abstractmethod
    def upload(self, cfg: ProjectConfig) -> None:
        """Upload project state (code, config, data, git) to remote."""

    @abstractmethod
    def run(self, cfg: ProjectConfig, extra_args: list[str] | None = None) -> RunInfo:
        """Start the EvoSkill loop on remote. Returns RunInfo for tracking."""

    @abstractmethod
    def status(self, cfg: ProjectConfig, run_info: RunInfo) -> str:
        """Check run status. Returns a human-readable status string."""

    @abstractmethod
    def logs(self, cfg: ProjectConfig, run_info: RunInfo, follow: bool = False) -> Iterator[str]:
        """Stream log lines from the remote run."""

    @abstractmethod
    def download(self, cfg: ProjectConfig, run_info: RunInfo) -> None:
        """Download results from remote to local project."""

    @abstractmethod
    def stop(self, cfg: ProjectConfig, run_info: RunInfo) -> None:
        """Stop the remote run."""
