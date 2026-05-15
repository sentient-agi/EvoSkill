"""Tests for Daytona backend — all SDK calls mocked."""

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

def _make_cfg(tmp_path, api_key="dtk_test", image="evoskill:test",
              harness_name="claude", data_dirs=None, download=None):
    (tmp_path / ".evoskill").mkdir(exist_ok=True)
    (tmp_path / ".git").mkdir(exist_ok=True)
    # Create a dummy dataset so upload doesn't fail reading it
    dataset = tmp_path / "data.csv"
    dataset.write_text("q,a\nhello,world")
    from src.cli.config import DatasetConfig
    return ProjectConfig(
        harness=HarnessConfig(name=harness_name, data_dirs=data_dirs or []),
        dataset=DatasetConfig(path=str(dataset)),
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
    sb.process.create_session = MagicMock()
    sb.process.delete_session = MagicMock()
    sb.process.execute_session_command.return_value = MagicMock(cmd_id="cmd_abc")
    # Default to completed/success so _exec_async polling returns immediately.
    # Tests that need to simulate "still running" override this explicitly.
    sb.process.get_session_command.return_value = MagicMock(exit_code=0)
    sb.process.get_session_command_logs.return_value = MagicMock(
        output="", stdout="", stderr=""
    )
    sb.fs.upload_file = MagicMock()
    sb.fs.download_file = MagicMock(return_value=b"content")
    sb.fs.create_folder = MagicMock()
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
            self.sub_patch = patch("subprocess.run",
                                   return_value=MagicMock(returncode=0, stdout="", stderr=""))
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
    assert params.image == "evoskill:test"


def test_setup_validates_api_key(tmp_path):
    from src.remote.daytona import DaytonaBackend
    cfg = _make_cfg(tmp_path, api_key=None)
    cfg.remote.daytona.api_key = None

    with pytest.raises(ValueError, match="API key"):
        DaytonaBackend().setup(cfg)


def test_setup_validates_image(tmp_path):
    from src.remote.daytona import DaytonaBackend
    sb = _mock_sandbox()
    client = MagicMock()
    client.create.return_value = sb
    cfg = _make_cfg(tmp_path, image="")

    with _Patches(client):
        with pytest.raises(ValueError, match="image"):
            DaytonaBackend().setup(cfg)


def test_setup_sets_env_vars(tmp_path):
    from src.remote.daytona import DaytonaBackend
    sb = _mock_sandbox()
    client = MagicMock()
    client.create.return_value = sb
    cfg = _make_cfg(tmp_path)

    with _Patches(client), patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}):
        DaytonaBackend().setup(cfg)

    params = client.create.call_args[0][0]
    assert params.env_vars["CLAUDE_CODE_ACCEPT_TOS"] == "yes"
    assert params.env_vars["EVOSKILL_REMOTE"] == "1"
    assert params.env_vars["ANTHROPIC_API_KEY"] == "sk-test"


# ── Upload ───────────────────────────────────────────────────────────────────

def test_upload_sends_bundle(tmp_path):
    sb = _mock_sandbox()
    client = MagicMock()
    client.create.return_value = sb
    cfg = _make_cfg(tmp_path)

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
    cfg = _make_cfg(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]")
    (tmp_path / "src").mkdir(exist_ok=True)
    (tmp_path / "src" / "main.py").write_text("pass")

    with _Patches(client):
        backend = _setup_backend(client, cfg)
        backend.upload(cfg)

    assert sb.fs.upload_file.call_count > 0


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


def test_upload_remaps_external_data_dirs(tmp_path):
    sb = _mock_sandbox()
    client = MagicMock()
    client.create.return_value = sb
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    ext_dir = tmp_path / "external_data"
    ext_dir.mkdir()
    (ext_dir / "file.csv").write_text("data")
    cfg = _make_cfg(project_dir, data_dirs=[str(ext_dir)])

    with _Patches(client):
        backend = _setup_backend(client, cfg)
        backend.upload(cfg)

    upload_paths = [str(c[0][1]) if len(c[0]) > 1 else "" for c in sb.fs.upload_file.call_args_list]
    assert any("external_data" in p and ".tar.gz" in p for p in upload_paths)


# ── Run ──────────────────────────────────────────────────────────────────────

def test_run_uses_session_api(tmp_path):
    sb = _mock_sandbox()
    client = MagicMock()
    client.create.return_value = sb
    cfg = _make_cfg(tmp_path)

    with _Patches(client):
        backend = _setup_backend(client, cfg)
        run_info = backend.run(cfg)

    sb.process.create_session.assert_called_once_with("evoskill-run")
    sb.process.execute_session_command.assert_called_once()
    req = sb.process.execute_session_command.call_args[0][1]
    assert req.run_async is True


def test_run_saves_run_info_with_cmd_id(tmp_path):
    sb = _mock_sandbox()
    client = MagicMock()
    client.create.return_value = sb
    cfg = _make_cfg(tmp_path)

    with _Patches(client):
        backend = _setup_backend(client, cfg)
        run_info = backend.run(cfg)

    assert run_info.extra["sandbox_id"] == "sb_test123"
    assert run_info.extra["cmd_id"] == "cmd_abc"
    assert run_info.extra["session_id"] == "evoskill-run"
    saved = json.loads((tmp_path / ".evoskill" / "remote_run.json").read_text())
    assert saved["extra"]["cmd_id"] == "cmd_abc"


def test_run_passes_extra_args(tmp_path):
    sb = _mock_sandbox()
    client = MagicMock()
    client.create.return_value = sb
    cfg = _make_cfg(tmp_path)

    with _Patches(client):
        backend = _setup_backend(client, cfg)
        backend.run(cfg, extra_args=["--continue", "--verbose"])

    req = sb.process.execute_session_command.call_args[0][1]
    assert "--continue" in req.command
    assert "--verbose" in req.command


def test_run_includes_path_overrides(tmp_path):
    sb = _mock_sandbox()
    client = MagicMock()
    client.create.return_value = sb
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    ext_dataset = tmp_path / "data.csv"
    ext_dataset.write_text("q,a\nhello,world")
    cfg = _make_cfg(project_dir)
    cfg.dataset.path = str(ext_dataset)

    with _Patches(client):
        backend = _setup_backend(client, cfg)
        backend.upload(cfg)
        backend.run(cfg)

    req = sb.process.execute_session_command.call_args[0][1]
    assert "EVOSKILL_PATH_OVERRIDES" in req.command


@pytest.mark.parametrize("harness", ["claude", "opencode", "codex", "goose", "openhands"])
def test_run_works_for_all_harnesses(tmp_path, harness):
    sb = _mock_sandbox()
    client = MagicMock()
    client.create.return_value = sb
    cfg = _make_cfg(tmp_path, harness_name=harness)

    with _Patches(client):
        backend = _setup_backend(client, cfg)
        run_info = backend.run(cfg)

    assert run_info.target == "daytona"
    sb.process.create_session.assert_called_once()
    sb.process.execute_session_command.assert_called_once()


# ── Status ───────────────────────────────────────────────────────────────────

def test_status_running(tmp_path):
    from src.remote.daytona import DaytonaBackend
    sb = _mock_sandbox()
    sb.process.get_session_command.return_value = MagicMock(exit_code=None)
    sb.process.get_session_command_logs.return_value = MagicMock(
        output="Evaluating: 50%", stdout="", stderr=""
    )
    run_info = RunInfo(run_id="test", target="daytona",
                       extra={"sandbox_id": "sb_test123", "session_id": "evoskill-run", "cmd_id": "cmd_abc"})
    cfg = _make_cfg(tmp_path)

    with patch("src.remote.daytona._get_sandbox", return_value=sb):
        backend = DaytonaBackend()
        backend._client = MagicMock()
        status = backend.status(cfg, run_info)

    assert "running" in status.lower()


def test_status_completed(tmp_path):
    from src.remote.daytona import DaytonaBackend
    sb = _mock_sandbox()
    sb.process.get_session_command.return_value = MagicMock(exit_code=0)
    sb.process.get_session_command_logs.return_value = MagicMock(
        output="Score: 0.85\nRun complete", stdout="", stderr=""
    )
    run_info = RunInfo(run_id="test", target="daytona",
                       extra={"sandbox_id": "sb_test123", "session_id": "evoskill-run", "cmd_id": "cmd_abc"})
    cfg = _make_cfg(tmp_path)

    with patch("src.remote.daytona._get_sandbox", return_value=sb):
        backend = DaytonaBackend()
        backend._client = MagicMock()
        status = backend.status(cfg, run_info)

    assert "completed" in status.lower()


def test_status_failed(tmp_path):
    from src.remote.daytona import DaytonaBackend
    sb = _mock_sandbox()
    sb.process.get_session_command.return_value = MagicMock(exit_code=1)
    sb.process.get_session_command_logs.return_value = MagicMock(
        output="Traceback: something broke", stdout="", stderr=""
    )
    run_info = RunInfo(run_id="test", target="daytona",
                       extra={"sandbox_id": "sb_test123", "session_id": "evoskill-run", "cmd_id": "cmd_abc"})
    cfg = _make_cfg(tmp_path)

    with patch("src.remote.daytona._get_sandbox", return_value=sb):
        backend = DaytonaBackend()
        backend._client = MagicMock()
        status = backend.status(cfg, run_info)

    assert "failed" in status.lower()


# ── Logs ─────────────────────────────────────────────────────────────────────

def test_logs_returns_content(tmp_path):
    from src.remote.daytona import DaytonaBackend
    sb = _mock_sandbox()
    sb.process.get_session_command_logs.return_value = MagicMock(
        output="line1\nline2\nline3", stdout="", stderr=""
    )
    run_info = RunInfo(run_id="test", target="daytona",
                       extra={"sandbox_id": "sb_test123", "session_id": "evoskill-run", "cmd_id": "cmd_abc"})
    cfg = _make_cfg(tmp_path)

    with patch("src.remote.daytona._get_sandbox", return_value=sb):
        backend = DaytonaBackend()
        backend._client = MagicMock()
        lines = list(backend.logs(cfg, run_info, follow=False))

    assert len(lines) == 3
    assert "line1" in lines[0]


def test_logs_follow_exits_on_completion(tmp_path):
    from src.remote.daytona import DaytonaBackend
    sb = _mock_sandbox()
    # First call: still running, second call: done
    sb.process.get_session_command.side_effect = [
        MagicMock(exit_code=None),
        MagicMock(exit_code=0),
    ]
    sb.process.get_session_command_logs.return_value = MagicMock(
        output="progress\ndone", stdout="", stderr=""
    )
    run_info = RunInfo(run_id="test", target="daytona",
                       extra={"sandbox_id": "sb_test123", "session_id": "evoskill-run", "cmd_id": "cmd_abc"})
    cfg = _make_cfg(tmp_path)

    with patch("src.remote.daytona._get_sandbox", return_value=sb), \
         patch("time.sleep"):
        backend = DaytonaBackend()
        backend._client = MagicMock()
        lines = list(backend.logs(cfg, run_info, follow=True))

    assert any("Run completed" in l for l in lines)


# ── Download ─────────────────────────────────────────────────────────────────

def test_download_creates_branches(tmp_path):
    from src.remote.daytona import DaytonaBackend
    sb = _mock_sandbox()
    sb.process.exec.return_value = MagicMock(result="frontier/iter-skill-3:0.85")
    run_info = RunInfo(run_id="test", target="daytona",
                       extra={"sandbox_id": "sb_test123", "session_id": "evoskill-run", "cmd_id": "cmd_abc"})
    cfg = _make_cfg(tmp_path)

    unbundle_output = "abc123 refs/heads/program/iter-skill-3\n"
    with patch("src.remote.daytona._get_sandbox", return_value=sb), \
         patch("subprocess.run") as mock_sub:
        mock_sub.return_value = MagicMock(returncode=0, stdout=unbundle_output, stderr="")
        backend = DaytonaBackend()
        backend._client = MagicMock()
        backend.download(cfg, run_info)

    # Should have called git branch -f for the unbundled ref
    branch_calls = [c for c in mock_sub.call_args_list if "branch" in str(c)]
    assert len(branch_calls) >= 1


# ── Stop ─────────────────────────────────────────────────────────────────────

def test_stop_deletes_sandbox(tmp_path):
    from src.remote.daytona import DaytonaBackend
    sb = _mock_sandbox()
    client = MagicMock()
    run_info = RunInfo(run_id="test", target="daytona",
                       extra={"sandbox_id": "sb_test123"})
    cfg = _make_cfg(tmp_path)

    with patch("src.remote.daytona._get_sandbox", return_value=sb):
        backend = DaytonaBackend()
        backend._client = client
        backend.stop(cfg, run_info)

    client.delete.assert_called_once_with(sb)


def test_cleanup_current_deletes_sandbox(tmp_path):
    from src.remote.daytona import DaytonaBackend
    sb = _mock_sandbox()
    client = MagicMock()
    client.create.return_value = sb
    cfg = _make_cfg(tmp_path)

    with _Patches(client):
        backend = _setup_backend(client, cfg)

    backend._client = client
    backend.cleanup_current(cfg)
    client.delete.assert_called_once_with(sb)


# ── Permission mode ──────────────────────────────────────────────────────────

def test_remote_env_sets_bypass_permissions(tmp_path):
    """EVOSKILL_REMOTE=1 should set permission_mode to bypassPermissions."""
    from src.registry.sdk_utils import config_to_options
    from src.registry.models import ProgramConfig

    config = ProgramConfig(
        name="test",
        system_prompt={"type": "preset", "preset": "claude_code"},
        allowed_tools=[],
        schema={},
        metadata={"sdk": "claude"},
    )

    with patch.dict("os.environ", {"EVOSKILL_REMOTE": "1"}), \
         patch("src.harness.build_options") as mock_build, \
         patch("src.harness.set_sdk"), \
         patch("src.harness.get_sdk", return_value="claude"):
        mock_build.return_value = MagicMock()
        config_to_options(config, str(tmp_path))

    assert mock_build.called
    _, kwargs = mock_build.call_args
    assert kwargs["permission_mode"] == "bypassPermissions"


# ── _exec_async helper ───────────────────────────────────────────────────────

def test_exec_async_returns_on_success():
    """Polls until exit_code becomes 0, then returns."""
    from src.remote.daytona import _exec_async

    sb = _mock_sandbox()
    sb.process.execute_session_command.return_value = MagicMock(cmd_id="c1")
    sb.process.get_session_command.side_effect = [
        MagicMock(exit_code=None),
        MagicMock(exit_code=None),
        MagicMock(exit_code=0),
    ]

    with patch("src.remote.daytona.time.sleep"):
        _exec_async(sb, "sess", "do something")

    assert sb.process.execute_session_command.called
    req = sb.process.execute_session_command.call_args[0][1]
    assert req.run_async is True
    assert req.command == "do something"
    assert sb.process.get_session_command.call_count == 3


def test_exec_async_raises_on_nonzero_exit():
    """Includes log tail in the raised error message."""
    from src.remote.daytona import _exec_async

    sb = _mock_sandbox()
    sb.process.execute_session_command.return_value = MagicMock(cmd_id="c1")
    sb.process.get_session_command.return_value = MagicMock(exit_code=2)
    sb.process.get_session_command_logs.return_value = MagicMock(
        output="boom: file not found", stdout="", stderr=""
    )

    with patch("src.remote.daytona.time.sleep"):
        with pytest.raises(RuntimeError, match="exit=2"):
            _exec_async(sb, "sess", "bad cmd")


def test_exec_async_includes_log_in_error():
    from src.remote.daytona import _exec_async

    sb = _mock_sandbox()
    sb.process.execute_session_command.return_value = MagicMock(cmd_id="c1")
    sb.process.get_session_command.return_value = MagicMock(exit_code=1)
    sb.process.get_session_command_logs.return_value = MagicMock(
        output="", stdout="some stdout", stderr="critical stderr msg"
    )

    with patch("src.remote.daytona.time.sleep"):
        with pytest.raises(RuntimeError, match="some stdout"):
            _exec_async(sb, "sess", "bad cmd")


def test_exec_async_times_out():
    """Raises if exit_code never gets set within max_wait_seconds."""
    from src.remote.daytona import _exec_async

    sb = _mock_sandbox()
    sb.process.execute_session_command.return_value = MagicMock(cmd_id="c1")
    sb.process.get_session_command.return_value = MagicMock(exit_code=None)

    with patch("src.remote.daytona.time.sleep"):
        with pytest.raises(RuntimeError, match="timed out"):
            _exec_async(sb, "sess", "slow cmd",
                        poll_interval=0.001, max_wait_seconds=0.01)


def test_exec_async_uses_provided_session_id():
    from src.remote.daytona import _exec_async

    sb = _mock_sandbox()
    sb.process.execute_session_command.return_value = MagicMock(cmd_id="c1")
    sb.process.get_session_command.return_value = MagicMock(exit_code=0)

    _exec_async(sb, "my-session", "do it")

    args, _ = sb.process.execute_session_command.call_args
    assert args[0] == "my-session"


# ── Upload uses async session for heavy ops ──────────────────────────────────

def test_upload_creates_upload_session(tmp_path):
    sb = _mock_sandbox()
    client = MagicMock()
    client.create.return_value = sb
    cfg = _make_cfg(tmp_path)

    with _Patches(client):
        backend = _setup_backend(client, cfg)
        backend.upload(cfg)

    create_calls = [c.args for c in sb.process.create_session.call_args_list]
    assert ("evoskill-upload",) in create_calls


def test_upload_dispatches_git_unbundle_via_session(tmp_path):
    """Git unbundle goes through execute_session_command, not blocking exec."""
    sb = _mock_sandbox()
    client = MagicMock()
    client.create.return_value = sb
    cfg = _make_cfg(tmp_path)

    with _Patches(client):
        backend = _setup_backend(client, cfg)
        backend.upload(cfg)

    # Look for the git unbundle command dispatched via session
    session_cmds = [
        call.args[1].command
        for call in sb.process.execute_session_command.call_args_list
    ]
    assert any("git bundle unbundle" in c for c in session_cmds)


def test_upload_single_chunk_data_uses_async_for_extract(tmp_path):
    """Even single-chunk extract goes through async session, not blocking exec."""
    sb = _mock_sandbox()
    client = MagicMock()
    client.create.return_value = sb
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    ext_dir = tmp_path / "external_data"
    ext_dir.mkdir()
    (ext_dir / "file.csv").write_text("data")
    cfg = _make_cfg(project_dir, data_dirs=[str(ext_dir)])

    with _Patches(client):
        backend = _setup_backend(client, cfg)
        backend.upload(cfg)

    session_cmds = [
        call.args[1].command
        for call in sb.process.execute_session_command.call_args_list
    ]
    # Extract command should go through async session
    assert any("tar xzf /tmp/external_data.tar.gz" in c for c in session_cmds)
