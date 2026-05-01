"""Cycle 2: Tests for sync logic — file lists, path remapping, git bundle commands."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.remote.sync import (
    UPLOAD_EXCLUDES,
    bundle_create_args,
    bundle_unbundle_args,
    download_file_list,
    remap_data_dirs,
    should_exclude_upload,
    upload_file_list,
)
from src.cli.config import DownloadConfig


# ── Upload excludes ──────────────────────────────────────────────────────────

def test_upload_excludes_venv():
    assert should_exclude_upload(Path(".venv/lib/python3.12"))


def test_upload_excludes_pycache():
    assert should_exclude_upload(Path("src/__pycache__/config.cpython-312.pyc"))


def test_upload_excludes_cache_runs():
    assert should_exclude_upload(Path(".cache/runs/abc123/q1.json"))


def test_upload_excludes_node_modules():
    assert should_exclude_upload(Path("node_modules/express/index.js"))


def test_upload_excludes_pytest_cache():
    assert should_exclude_upload(Path(".pytest_cache/v/cache"))


def test_upload_excludes_notebooks():
    assert should_exclude_upload(Path("notebooks/analysis.ipynb"))


def test_upload_excludes_assets():
    assert should_exclude_upload(Path("assets/logo.png"))


def test_upload_includes_git():
    assert not should_exclude_upload(Path(".git/HEAD"))


def test_upload_includes_evoskill():
    assert not should_exclude_upload(Path(".evoskill/config.toml"))


def test_upload_includes_claude():
    assert not should_exclude_upload(Path(".claude/skills/brainstorming/SKILL.md"))


def test_upload_includes_src():
    assert not should_exclude_upload(Path("src/loop/runner.py"))


def test_upload_includes_pyproject():
    assert not should_exclude_upload(Path("pyproject.toml"))


# ── Upload file list ─────────────────────────────────────────────────────────

def test_upload_file_list_essentials(tmp_path):
    """All essential directories/files are included in upload list."""
    # Create a minimal project structure
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: refs/heads/main")
    (tmp_path / ".evoskill").mkdir()
    (tmp_path / ".evoskill" / "config.toml").write_text("[harness]")
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "skills").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("pass")
    (tmp_path / "pyproject.toml").write_text("[project]")
    (tmp_path / "Dockerfile").write_text("FROM python:3.12")
    # Excluded dirs
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "bin").mkdir()
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "node_modules").mkdir()

    files = upload_file_list(tmp_path)
    rel_paths = {str(f.relative_to(tmp_path)) for f in files}

    # Essentials present
    assert ".evoskill/config.toml" in rel_paths
    assert "src/main.py" in rel_paths
    assert "pyproject.toml" in rel_paths
    assert "Dockerfile" in rel_paths

    # Excluded not present
    for f in files:
        rel = str(f.relative_to(tmp_path))
        assert not rel.startswith(".venv")
        assert not rel.startswith("__pycache__")
        assert not rel.startswith("node_modules")


# ── Download file list ───────────────────────────────────────────────────────

def test_download_default_paths():
    """Default download includes only best program state + reports."""
    cfg = DownloadConfig()
    paths = download_file_list(cfg)
    assert ".claude/skills/" in paths
    assert ".claude/program.yaml" in paths
    assert ".evoskill/reports/" in paths
    assert ".cache/runs/" not in paths
    assert ".claude/feedback_history.md" not in paths


def test_download_all_branches():
    """all_branches=true adds git refs."""
    cfg = DownloadConfig(all_branches=True)
    paths = download_file_list(cfg)
    assert any("git" in p for p in paths)


def test_download_with_cache():
    """cache=true adds .cache/runs/."""
    cfg = DownloadConfig(cache=True)
    paths = download_file_list(cfg)
    assert ".cache/runs/" in paths


def test_download_with_feedback():
    """feedback_history=true adds the file."""
    cfg = DownloadConfig(feedback_history=True)
    paths = download_file_list(cfg)
    assert ".claude/feedback_history.md" in paths


# ── Data dir remapping ───────────────────────────────────────────────────────

def test_data_dir_remapping_absolute():
    """/Users/me/treasury → /mnt/data/treasury + override."""
    project_root = Path("/home/user/project")
    data_dirs = ["/Users/me/treasury", "/Users/me/reports"]
    mappings = remap_data_dirs(data_dirs, project_root)

    assert len(mappings) == 2
    assert mappings[0].host_path == Path("/Users/me/treasury")
    assert mappings[0].container_path == "/mnt/data/treasury"
    assert mappings[1].host_path == Path("/Users/me/reports")
    assert mappings[1].container_path == "/mnt/data/reports"


def test_data_dir_inside_project():
    """Relative data_dir inside project → no remapping, uses /workspace/ prefix."""
    project_root = Path("/home/user/project")
    data_dirs = ["/home/user/project/data/local"]
    mappings = remap_data_dirs(data_dirs, project_root)

    assert len(mappings) == 1
    assert mappings[0].needs_upload is False
    assert mappings[0].container_path == "/workspace/data/local"


def test_data_dir_mixed():
    """Mix of internal and external dirs."""
    project_root = Path("/home/user/project")
    data_dirs = ["/home/user/project/data", "/external/data"]
    mappings = remap_data_dirs(data_dirs, project_root)

    internal = [m for m in mappings if not m.needs_upload]
    external = [m for m in mappings if m.needs_upload]
    assert len(internal) == 1
    assert len(external) == 1


def test_data_dir_empty():
    """No data dirs → empty list."""
    mappings = remap_data_dirs([], Path("/project"))
    assert mappings == []


# ── Git bundle commands ──────────────────────────────────────────────────────

def test_git_bundle_create_all():
    """Bundle all branches."""
    args = bundle_create_args("/tmp/repo.bundle", all_branches=True)
    assert args == ["git", "bundle", "create", "/tmp/repo.bundle", "--all"]


def test_git_bundle_create_best_only():
    """Bundle only best branch + frontier tag."""
    args = bundle_create_args(
        "/tmp/best.bundle",
        all_branches=False,
        branch="program/iter-skill-7",
    )
    assert "program/iter-skill-7" in args
    assert "--all" not in args


def test_git_bundle_unbundle():
    """Unbundle command."""
    args = bundle_unbundle_args("/tmp/repo.bundle")
    assert args == ["git", "bundle", "unbundle", "/tmp/repo.bundle"]
