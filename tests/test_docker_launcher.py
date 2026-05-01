"""Tests for Docker launcher — compose generation and launch flow."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.cli.config import HarnessConfig, ProjectConfig, DatasetConfig


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_cfg(tmp_path, harness_name="claude", data_dirs=None, dataset_path=None):
    (tmp_path / ".evoskill").mkdir(exist_ok=True)
    ds_path = dataset_path or str(tmp_path / "data.csv")
    Path(ds_path).write_text("q,a\nhello,world")
    return ProjectConfig(
        harness=HarnessConfig(name=harness_name, data_dirs=data_dirs or []),
        dataset=DatasetConfig(path=ds_path),
        project_root=tmp_path,
    )


# ── Compose generation ───────────────────────────────────────────────────────

def test_compose_has_correct_image(tmp_path):
    from src.docker.launcher import _build_compose
    cfg = _make_cfg(tmp_path)
    compose = _build_compose(cfg, [])
    assert compose["services"]["evoskill"]["image"] == "evoskill"


def test_compose_mounts_project_root(tmp_path):
    from src.docker.launcher import _build_compose
    cfg = _make_cfg(tmp_path)
    compose = _build_compose(cfg, [])
    volumes = compose["services"]["evoskill"]["volumes"]
    assert any("/workspace" in v for v in volumes)


def test_compose_mounts_external_dataset(tmp_path):
    from src.docker.launcher import _build_compose
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / ".evoskill").mkdir()
    ext_dataset = tmp_path / "external.csv"
    ext_dataset.write_text("q,a\nhello,world")
    cfg = _make_cfg(project_dir, dataset_path=str(ext_dataset))

    compose = _build_compose(cfg, [])
    volumes = compose["services"]["evoskill"]["volumes"]
    assert any("external.csv" in v and ":ro" in v for v in volumes)


def test_compose_mounts_external_data_dirs(tmp_path):
    from src.docker.launcher import _build_compose
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / ".evoskill").mkdir()
    ext_dir = tmp_path / "my_data"
    ext_dir.mkdir()
    cfg = _make_cfg(project_dir, data_dirs=[str(ext_dir)])

    compose = _build_compose(cfg, [])
    volumes = compose["services"]["evoskill"]["volumes"]
    assert any("my_data" in v and ":ro" in v for v in volumes)


def test_compose_sets_path_overrides_for_external(tmp_path):
    from src.docker.launcher import _build_compose
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / ".evoskill").mkdir()
    ext_dataset = tmp_path / "data.csv"
    ext_dataset.write_text("q,a")
    cfg = _make_cfg(project_dir, dataset_path=str(ext_dataset))

    compose = _build_compose(cfg, [])
    env = compose["services"]["evoskill"]["env_with_values"]
    overrides = [e for e in env if "EVOSKILL_PATH_OVERRIDES" in e]
    assert len(overrides) == 1
    parsed = json.loads(overrides[0].split("=", 1)[1])
    assert "dataset_path" in parsed


def test_compose_forwards_api_keys_by_name(tmp_path):
    from src.docker.launcher import _build_compose
    cfg = _make_cfg(tmp_path)

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test", "OPENAI_API_KEY": "sk-oai"}):
        compose = _build_compose(cfg, [])

    forward = compose["services"]["evoskill"]["env_forward"]
    assert "ANTHROPIC_API_KEY" in forward
    assert "OPENAI_API_KEY" in forward
    # Values should NOT be in the compose
    env_values = compose["services"]["evoskill"]["env_with_values"]
    assert not any("sk-test" in e for e in env_values)


def test_compose_does_not_forward_unset_keys(tmp_path):
    from src.docker.launcher import _build_compose
    cfg = _make_cfg(tmp_path)

    with patch.dict("os.environ", {}, clear=True):
        compose = _build_compose(cfg, [])

    assert compose["services"]["evoskill"]["env_forward"] == []


def test_compose_includes_extra_args(tmp_path):
    from src.docker.launcher import _build_compose
    cfg = _make_cfg(tmp_path)
    compose = _build_compose(cfg, ["--continue", "--verbose"])
    cmd = compose["services"]["evoskill"]["command"]
    assert "--continue" in cmd
    assert "--verbose" in cmd


def test_compose_sets_evoskill_remote(tmp_path):
    from src.docker.launcher import _build_compose
    cfg = _make_cfg(tmp_path)
    compose = _build_compose(cfg, [])
    env = compose["services"]["evoskill"]["env_with_values"]
    assert "EVOSKILL_REMOTE=1" in env
    assert "CLAUDE_CODE_ACCEPT_TOS=yes" in env


@pytest.mark.parametrize("harness", ["claude", "opencode", "codex", "goose", "openhands"])
def test_compose_works_for_all_harnesses(tmp_path, harness):
    from src.docker.launcher import _build_compose
    cfg = _make_cfg(tmp_path, harness_name=harness)
    compose = _build_compose(cfg, [])
    assert compose["services"]["evoskill"]["image"] == "evoskill"
    assert "evoskill run" in compose["services"]["evoskill"]["command"]


# ── Compose file writing ────────────────────────────────────────────────────

def test_write_compose_creates_valid_yaml(tmp_path):
    from src.docker.launcher import _build_compose, _write_compose
    cfg = _make_cfg(tmp_path)

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}):
        compose = _build_compose(cfg, [])
    path = _write_compose(cfg, compose)

    assert path.exists()
    content = path.read_text()
    assert "services:" in content
    assert "evoskill:" in content
    assert "ANTHROPIC_API_KEY" in content
    # API key value should NOT be in file
    assert "sk-test" not in content


def test_write_compose_handles_json_in_env(tmp_path):
    """EVOSKILL_PATH_OVERRIDES contains JSON with braces — must not break YAML."""
    from src.docker.launcher import _build_compose, _write_compose
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    (project_dir / ".evoskill").mkdir()
    ext = tmp_path / "ext.csv"
    ext.write_text("q,a")
    cfg = _make_cfg(project_dir, dataset_path=str(ext))

    compose = _build_compose(cfg, [])
    path = _write_compose(cfg, compose)

    content = path.read_text()
    # Should use single quotes to protect JSON braces
    assert "EVOSKILL_PATH_OVERRIDES=" in content


# ── Launch flow ──────────────────────────────────────────────────────────────

def test_launch_builds_image_if_missing(tmp_path):
    from src.docker.launcher import launch_docker
    cfg = _make_cfg(tmp_path)
    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim")

    with patch("subprocess.run") as mock_run:
        # images -q returns empty (no image), build succeeds, down succeeds, up succeeds
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),   # images -q
            MagicMock(returncode=0, stdout="", stderr=""),   # build
            MagicMock(returncode=0, stdout="", stderr=""),   # down
            MagicMock(returncode=0, stdout="", stderr=""),   # up -d
        ]
        launch_docker(cfg)

    build_calls = [c for c in mock_run.call_args_list if "docker', 'build" in str(c)]
    assert len(build_calls) == 1


def test_launch_skips_build_if_image_exists(tmp_path):
    from src.docker.launcher import launch_docker
    cfg = _make_cfg(tmp_path)
    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim")

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="abc123\n", stderr=""),  # images -q (exists)
            MagicMock(returncode=0, stdout="", stderr=""),           # down
            MagicMock(returncode=0, stdout="", stderr=""),           # up -d
        ]
        launch_docker(cfg)

    build_calls = [c for c in mock_run.call_args_list if "docker', 'build" in str(c)]
    assert len(build_calls) == 0


def test_launch_rebuilds_when_flag_set(tmp_path):
    from src.docker.launcher import launch_docker
    cfg = _make_cfg(tmp_path)
    (tmp_path / "Dockerfile").write_text("FROM python:3.12-slim")

    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="abc123\n", stderr=""),  # images -q (exists)
            MagicMock(returncode=0, stdout="", stderr=""),           # build
            MagicMock(returncode=0, stdout="", stderr=""),           # down
            MagicMock(returncode=0, stdout="", stderr=""),           # up -d
        ]
        launch_docker(cfg, rebuild=True)

    build_calls = [c for c in mock_run.call_args_list if "docker', 'build" in str(c)]
    assert len(build_calls) == 1


def test_launch_fails_without_dockerfile(tmp_path):
    from src.docker.launcher import launch_docker
    cfg = _make_cfg(tmp_path)
    # No Dockerfile

    with pytest.raises(SystemExit):
        launch_docker(cfg)
