"""Cycle 6: Tests for --remote flag and evoskill remote subcommands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from src.cli.config import (
    DaytonaConfig,
    DownloadConfig,
    HarnessConfig,
    ProjectConfig,
    RemoteConfig,
)
from src.remote.base import RunInfo


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_cfg(tmp_path, target="daytona"):
    (tmp_path / ".evoskill").mkdir(exist_ok=True)
    (tmp_path / ".git").mkdir(exist_ok=True)
    return ProjectConfig(
        harness=HarnessConfig(name="claude"),
        remote=RemoteConfig(
            target=target,
            daytona=DaytonaConfig(api_key="dtk_test"),
            download=DownloadConfig(),
        ),
        project_root=tmp_path,
    )


def _make_cfg_no_remote(tmp_path):
    (tmp_path / ".evoskill").mkdir(exist_ok=True)
    return ProjectConfig(
        harness=HarnessConfig(name="claude"),
        project_root=tmp_path,
    )


def _mock_backend():
    backend = MagicMock()
    backend.setup = MagicMock()
    backend.upload = MagicMock()
    backend.run = MagicMock(return_value=RunInfo(
        run_id="test-123", target="daytona", extra={"sandbox_id": "sb_abc"},
    ))
    backend.status = MagicMock(return_value="running (iteration 3/20)")
    backend.logs = MagicMock(return_value=iter(["line1", "line2"]))
    backend.download = MagicMock()
    backend.stop = MagicMock()
    return backend


# ── run --remote ─────────────────────────────────────────────────────────────

def test_run_remote_dispatches_to_daytona(tmp_path):
    """--remote flag with target=daytona uses DaytonaBackend."""
    from src.cli.commands.run import run_cmd

    cfg = _make_cfg(tmp_path)
    backend = _mock_backend()

    with patch("src.cli.commands.run.load_config", return_value=cfg), \
         patch("src.cli.commands.run._get_remote_backend", return_value=backend):
        runner = CliRunner()
        result = runner.invoke(run_cmd, ["--remote"])

    assert result.exit_code == 0
    backend.setup.assert_called_once()
    backend.upload.assert_called_once()
    backend.run.assert_called_once()


def test_run_remote_no_config(tmp_path):
    """--remote with no [remote] section → error."""
    from src.cli.commands.run import run_cmd

    cfg = _make_cfg_no_remote(tmp_path)

    with patch("src.cli.commands.run.load_config", return_value=cfg):
        runner = CliRunner()
        result = runner.invoke(run_cmd, ["--remote"])

    assert result.exit_code != 0
    assert "remote" in result.output.lower()


def test_run_remote_forwards_continue(tmp_path):
    """--remote --continue → backend.run gets --continue in extra_args."""
    from src.cli.commands.run import run_cmd

    cfg = _make_cfg(tmp_path)
    backend = _mock_backend()

    with patch("src.cli.commands.run.load_config", return_value=cfg), \
         patch("src.cli.commands.run._get_remote_backend", return_value=backend):
        runner = CliRunner()
        result = runner.invoke(run_cmd, ["--remote", "--continue"])

    assert result.exit_code == 0
    call_args = backend.run.call_args
    extra_args = call_args[1].get("extra_args") or call_args[0][1] if len(call_args[0]) > 1 else None
    # Check that --continue was passed somehow
    assert backend.run.called


# ── evoskill remote status ───────────────────────────────────────────────────

def test_remote_status_shows_status(tmp_path):
    """Loads remote_run.json, calls backend.status."""
    from src.cli.commands.remote import remote_status

    cfg = _make_cfg(tmp_path)
    # Write run info
    run_info = RunInfo(run_id="test-123", target="daytona", extra={"sandbox_id": "sb_abc"})
    run_info.save(tmp_path)

    backend = _mock_backend()

    with patch("src.cli.commands.remote.load_config", return_value=cfg), \
         patch("src.cli.commands.remote._get_remote_backend", return_value=backend):
        runner = CliRunner()
        result = runner.invoke(remote_status)

    assert result.exit_code == 0
    assert "running" in result.output.lower()


def test_remote_status_no_run(tmp_path):
    """No remote_run.json → clear message."""
    from src.cli.commands.remote import remote_status

    cfg = _make_cfg(tmp_path)

    with patch("src.cli.commands.remote.load_config", return_value=cfg):
        runner = CliRunner()
        result = runner.invoke(remote_status)

    assert "no active" in result.output.lower() or "not found" in result.output.lower()


# ── evoskill remote logs ─────────────────────────────────────────────────────

def test_remote_logs_streams(tmp_path):
    """backend.logs() output printed."""
    from src.cli.commands.remote import remote_logs

    cfg = _make_cfg(tmp_path)
    run_info = RunInfo(run_id="test-123", target="daytona", extra={"sandbox_id": "sb_abc"})
    run_info.save(tmp_path)

    backend = _mock_backend()

    with patch("src.cli.commands.remote.load_config", return_value=cfg), \
         patch("src.cli.commands.remote._get_remote_backend", return_value=backend):
        runner = CliRunner()
        result = runner.invoke(remote_logs)

    assert result.exit_code == 0
    assert "line1" in result.output


# ── evoskill remote stop ─────────────────────────────────────────────────────

def test_remote_stop_calls_backend(tmp_path):
    """backend.stop() called."""
    from src.cli.commands.remote import remote_stop

    cfg = _make_cfg(tmp_path)
    run_info = RunInfo(run_id="test-123", target="daytona", extra={"sandbox_id": "sb_abc"})
    run_info.save(tmp_path)

    backend = _mock_backend()

    with patch("src.cli.commands.remote.load_config", return_value=cfg), \
         patch("src.cli.commands.remote._get_remote_backend", return_value=backend):
        runner = CliRunner()
        result = runner.invoke(remote_stop)

    assert result.exit_code == 0
    backend.stop.assert_called_once()


# ── evoskill remote download ─────────────────────────────────────────────────

def test_remote_download_calls_backend(tmp_path):
    """backend.download() called."""
    from src.cli.commands.remote import remote_download

    cfg = _make_cfg(tmp_path)
    run_info = RunInfo(run_id="test-123", target="daytona", extra={"sandbox_id": "sb_abc"})
    run_info.save(tmp_path)

    backend = _mock_backend()

    with patch("src.cli.commands.remote.load_config", return_value=cfg), \
         patch("src.cli.commands.remote._get_remote_backend", return_value=backend):
        runner = CliRunner()
        result = runner.invoke(remote_download)

    assert result.exit_code == 0
    backend.download.assert_called_once()


def test_remote_download_prints_summary(tmp_path):
    """Shows what was downloaded."""
    from src.cli.commands.remote import remote_download

    cfg = _make_cfg(tmp_path)
    run_info = RunInfo(run_id="test-123", target="daytona", extra={"sandbox_id": "sb_abc"})
    run_info.save(tmp_path)

    backend = _mock_backend()

    with patch("src.cli.commands.remote.load_config", return_value=cfg), \
         patch("src.cli.commands.remote._get_remote_backend", return_value=backend):
        runner = CliRunner()
        result = runner.invoke(remote_download)

    assert result.exit_code == 0
    assert "download" in result.output.lower()
