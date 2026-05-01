"""Cycle 3: Tests for RemoteBackend ABC and RunInfo."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.remote.base import RemoteBackend, RunInfo


# ── RemoteBackend ABC ────────────────────────────────────────────────────────

def test_backend_cannot_instantiate():
    """RemoteBackend is abstract — can't instantiate directly."""
    with pytest.raises(TypeError):
        RemoteBackend()


def test_backend_incomplete_subclass():
    """Subclass without all methods → TypeError on instantiation."""
    class Partial(RemoteBackend):
        async def setup(self, cfg):
            pass
        # Missing: upload, run, status, logs, download, stop

    with pytest.raises(TypeError):
        Partial()


# ── RunInfo persistence ──────────────────────────────────────────────────────

def test_run_info_save_load(tmp_path):
    """RunInfo round-trips through save/load."""
    (tmp_path / ".evoskill").mkdir()
    info = RunInfo(
        run_id="test-123",
        target="daytona",
        extra={"sandbox_id": "sb_abc"},
    )
    info.save(tmp_path)

    loaded = RunInfo.load(tmp_path)
    assert loaded is not None
    assert loaded.run_id == "test-123"
    assert loaded.target == "daytona"
    assert loaded.extra["sandbox_id"] == "sb_abc"
    assert loaded.status == "running"


def test_run_info_load_missing(tmp_path):
    """Load returns None when no remote_run.json exists."""
    (tmp_path / ".evoskill").mkdir()
    assert RunInfo.load(tmp_path) is None


def test_run_info_clear(tmp_path):
    """Clear removes the file."""
    evoskill_dir = tmp_path / ".evoskill"
    evoskill_dir.mkdir()
    (evoskill_dir / "remote_run.json").write_text("{}")

    RunInfo.clear(tmp_path)
    assert not (evoskill_dir / "remote_run.json").exists()


def test_run_info_clear_missing(tmp_path):
    """Clear is a no-op if file doesn't exist."""
    (tmp_path / ".evoskill").mkdir()
    RunInfo.clear(tmp_path)  # should not raise
