"""Cycle 4: Tests for Daytona backend — all SDK calls mocked."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.cli.config import (
    DaytonaConfig,
    DownloadConfig,
    HarnessConfig,
    ProjectConfig,
    RemoteConfig,
)
from src.remote.base import RunInfo


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_cfg(tmp_path, api_key="dtk_test", image="evoskill:latest",
              data_dirs=None, download=None):
    (tmp_path / ".evoskill").mkdir(exist_ok=True)
    (tmp_path / ".git").mkdir(exist_ok=True)
    return ProjectConfig(
        harness=HarnessConfig(name="claude", data_dirs=data_dirs or []),
        remote=RemoteConfig(
            target="daytona",
            daytona=DaytonaConfig(api_key=api_key, image=image),
            download=download or DownloadConfig(),
        ),
        project_root=tmp_path,
    )


def _mock_sandbox():
    sb = MagicMock()
    sb.id = "sb_test123"
    sb.process.exec.return_value = MagicMock(result="ok")
    sb.fs.upload_file = MagicMock()
    sb.fs.download_file = MagicMock(return_value=b"content")
    sb.fs.create_folder = MagicMock()
    sb.stop = MagicMock()
    return sb


def _mock_params(**kwargs):
    m = MagicMock()
    for k, v in kwargs.items():
        setattr(m, k, v)
    return m


class _Patches:
    """Patches _make_client, _create_sandbox_params, and optionally subprocess."""

    def __init__(self, client, mock_subprocess=True):
        self.client = client
        self._patches = [
            patch("src.remote.daytona._make_client", return_value=client),
            patch("src.remote.daytona._create_sandbox_params",
                  side_effect=lambda **kw: _mock_params(**kw)),
        ]
        if mock_subprocess:
            self.sub_patch = patch("subprocess.run", return_value=MagicMock(returncode=0))
            self._patches.append(self.sub_patch)
        else:
            self.sub_patch = None
        self.mock_sub = None

    def __enter__(self):
        for p in self._patches:
            m = p.__enter__()
            if p is getattr(self, "sub_patch", None):
                self.mock_sub = m
        return self

    def __exit__(self, *args):
        for p in reversed(self._patches):
            p.__exit__(*args)


def _setup_backend(client, cfg):
    """Create and setup a DaytonaBackend."""
    from src.remote.daytona import DaytonaBackend
    backend = DaytonaBackend()
    backend.setup(cfg)
    return backend


# ── Setup ────────────────────────────────────────────────────────────────────

def test_setup_creates_sandbox(tmp_path):
    from src.remote.daytona import DaytonaBackend
    sb = _mock_sandbox()
    client = MagicMock()
    client.create.return_value = sb
    cfg = _make_cfg(tmp_path)

    with _Patches(client):
        backend = DaytonaBackend()
        backend.setup(cfg)

    client.create.assert_called_once()
    params = client.create.call_args[0][0]
    assert params.image == "evoskill:latest"


def test_setup_validates_api_key(tmp_path):
    from src.remote.daytona import DaytonaBackend
    cfg = _make_cfg(tmp_path, api_key=None)
    cfg.remote.daytona.api_key = None

    with pytest.raises(ValueError, match="API key"):
        DaytonaBackend().setup(cfg)


def test_setup_sets_auto_stop_zero(tmp_path):
    from src.remote.daytona import DaytonaBackend
    sb = _mock_sandbox()
    client = MagicMock()
    client.create.return_value = sb
    cfg = _make_cfg(tmp_path)

    with _Patches(client):
        DaytonaBackend().setup(cfg)

    params = client.create.call_args[0][0]
    assert params.auto_stop_interval == 0


# ── Upload ───────────────────────────────────────────────────────────────────

def test_upload_sends_bundle(tmp_path):
    sb = _mock_sandbox()
    client = MagicMock()
    client.create.return_value = sb
    cfg = _make_cfg(tmp_path)
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main")

    with _Patches(client) as p:
        backend = _setup_backend(client, cfg)
        backend.upload(cfg)

    bundle_calls = [c for c in p.mock_sub.call_args_list if "bundle" in str(c)]
    assert len(bundle_calls) >= 1
    sb.fs.upload_file.assert_called()


def test_upload_sends_project_files(tmp_path):
    sb = _mock_sandbox()
    client = MagicMock()
    client.create.return_value = sb
    cfg = _make_cfg(tmp_path)  # creates .evoskill/ and .git/
    (tmp_path / ".evoskill" / "config.toml").write_text("[harness]")
    (tmp_path / ".evoskill" / "task.md").write_text("Test")
    (tmp_path / "pyproject.toml").write_text("[project]")
    (tmp_path / "src").mkdir(exist_ok=True)
    (tmp_path / "src" / "main.py").write_text("pass")

    with _Patches(client):
        backend = _setup_backend(client, cfg)
        backend.upload(cfg)

    assert sb.fs.upload_file.call_count > 0


def test_upload_remaps_data_dirs(tmp_path):
    sb = _mock_sandbox()
    client = MagicMock()
    client.create.return_value = sb
    # Create a truly external directory (outside project root)
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    ext_dir = tmp_path / "external_data"
    ext_dir.mkdir()
    (ext_dir / "file.csv").write_text("data")
    cfg = _make_cfg(project_dir, data_dirs=[str(ext_dir)])

    with _Patches(client):
        backend = _setup_backend(client, cfg)
        backend.upload(cfg)

    # Data dir is now uploaded as a tar, then extracted via exec
    # Check that tar was uploaded and extract command was run
    upload_paths = [str(c[0][1]) if len(c[0]) > 1 else "" for c in sb.fs.upload_file.call_args_list]
    assert any("external_data" in p and ".tar.gz" in p for p in upload_paths)
    exec_calls = [str(c) for c in sb.process.exec.call_args_list]
    assert any("tar xzf" in c and "/mnt/data/" in c for c in exec_calls)


def test_upload_skips_excluded(tmp_path):
    sb = _mock_sandbox()
    client = MagicMock()
    client.create.return_value = sb
    (tmp_path / ".venv" / "bin").mkdir(parents=True)
    (tmp_path / ".venv" / "bin" / "python").write_text("")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "mod.pyc").write_text("")
    (tmp_path / "pyproject.toml").write_text("[project]")
    cfg = _make_cfg(tmp_path)

    with _Patches(client):
        backend = _setup_backend(client, cfg)
        backend.upload(cfg)

    joined = " ".join(str(c) for c in sb.fs.upload_file.call_args_list)
    assert ".venv" not in joined
    assert "__pycache__" not in joined


# ── Run ──────────────────────────────────────────────────────────────────────

def test_run_installs_and_starts(tmp_path):
    sb = _mock_sandbox()
    client = MagicMock()
    client.create.return_value = sb
    cfg = _make_cfg(tmp_path)

    with _Patches(client):
        backend = _setup_backend(client, cfg)
        run_info = backend.run(cfg)

    exec_calls = [str(c) for c in sb.process.exec.call_args_list]
    assert any("pip install" in c for c in exec_calls)
    assert any("evoskill" in c for c in exec_calls)
    assert run_info.target == "daytona"


def test_run_passes_continue_flag(tmp_path):
    sb = _mock_sandbox()
    client = MagicMock()
    client.create.return_value = sb
    cfg = _make_cfg(tmp_path)

    with _Patches(client):
        backend = _setup_backend(client, cfg)
        backend.run(cfg, extra_args=["--continue"])

    # --continue is now in the uploaded run.sh script
    upload_calls = sb.fs.upload_file.call_args_list
    script_uploads = [c for c in upload_calls if len(c[0]) >= 2 and "run.sh" in str(c[0][1])]
    assert len(script_uploads) == 1
    script_content = script_uploads[0][0][0]
    if isinstance(script_content, bytes):
        script_content = script_content.decode()
    assert "--continue" in script_content


def test_run_saves_run_info(tmp_path):
    sb = _mock_sandbox()
    client = MagicMock()
    client.create.return_value = sb
    cfg = _make_cfg(tmp_path)

    with _Patches(client):
        backend = _setup_backend(client, cfg)
        run_info = backend.run(cfg)

    assert run_info.extra["sandbox_id"] == "sb_test123"
    saved = json.loads((tmp_path / ".evoskill" / "remote_run.json").read_text())
    assert saved["extra"]["sandbox_id"] == "sb_test123"


# ── Status ───────────────────────────────────────────────────────────────────

def test_status_running(tmp_path):
    from src.remote.daytona import DaytonaBackend
    sb = _mock_sandbox()
    sb.process.exec.side_effect = [
        MagicMock(result="alive"),          # pgrep check
        MagicMock(result='{"iteration": 3}'),  # checkpoint
        MagicMock(result="Score: 0.75"),     # tail log
    ]
    run_info = RunInfo(run_id="test", target="daytona", extra={"sandbox_id": "sb_test123"})
    cfg = _make_cfg(tmp_path)

    with patch("src.remote.daytona._get_sandbox", return_value=sb):
        backend = DaytonaBackend()
        backend._client = MagicMock()
        status = backend.status(cfg, run_info)

    assert "running" in status.lower()
    assert "iteration 3" in status


def test_status_completed(tmp_path):
    from src.remote.daytona import DaytonaBackend
    sb = _mock_sandbox()
    sb.process.exec.side_effect = [
        MagicMock(result="done"),            # pgrep check
        MagicMock(result='{"iteration": 20}'),  # checkpoint
        MagicMock(result="Score: 0.85"),      # tail log
    ]
    run_info = RunInfo(run_id="test", target="daytona", extra={"sandbox_id": "sb_test123"})
    cfg = _make_cfg(tmp_path)

    with patch("src.remote.daytona._get_sandbox", return_value=sb):
        backend = DaytonaBackend()
        backend._client = MagicMock()
        status = backend.status(cfg, run_info)

    assert "completed" in status.lower()


# ── Download ─────────────────────────────────────────────────────────────────

def test_download_best_only(tmp_path):
    from src.remote.daytona import DaytonaBackend
    sb = _mock_sandbox()
    sb.process.exec.return_value = MagicMock(result="frontier/iter-skill-3:0.85")
    run_info = RunInfo(run_id="test", target="daytona", extra={"sandbox_id": "sb_test123"})
    cfg = _make_cfg(tmp_path)
    (tmp_path / ".claude").mkdir(exist_ok=True)

    with patch("src.remote.daytona._get_sandbox", return_value=sb), \
         patch("subprocess.run", return_value=MagicMock(returncode=0)):
        backend = DaytonaBackend()
        backend.download(cfg, run_info)

    assert sb.fs.download_file.call_count > 0


def test_download_unbundles_git(tmp_path):
    from src.remote.daytona import DaytonaBackend
    sb = _mock_sandbox()
    sb.process.exec.return_value = MagicMock(result="frontier/iter-skill-3:0.85")
    sb.fs.download_file.return_value = b"fake-bundle"
    run_info = RunInfo(run_id="test", target="daytona", extra={"sandbox_id": "sb_test123"})
    cfg = _make_cfg(tmp_path)
    (tmp_path / ".claude").mkdir(exist_ok=True)

    with patch("src.remote.daytona._get_sandbox", return_value=sb), \
         patch("subprocess.run") as mock_sub:
        mock_sub.return_value = MagicMock(returncode=0)
        backend = DaytonaBackend()
        backend.download(cfg, run_info)

    bundle_calls = [c for c in mock_sub.call_args_list if "unbundle" in str(c)]
    assert len(bundle_calls) >= 1


# ── Stop ─────────────────────────────────────────────────────────────────────

def test_stop_stops_sandbox(tmp_path):
    from src.remote.daytona import DaytonaBackend
    sb = _mock_sandbox()
    run_info = RunInfo(run_id="test", target="daytona", extra={"sandbox_id": "sb_test123"})
    cfg = _make_cfg(tmp_path)

    with patch("src.remote.daytona._get_sandbox", return_value=sb):
        backend = DaytonaBackend()
        backend._client = MagicMock()
        backend.stop(cfg, run_info)

    sb.stop.assert_called_once()


def test_stop_no_active_run(tmp_path):
    from src.remote.daytona import DaytonaBackend
    run_info = RunInfo(run_id="test", target="daytona", extra={})
    cfg = _make_cfg(tmp_path)

    with pytest.raises(ValueError, match="sandbox"):
        DaytonaBackend().stop(cfg, run_info)
