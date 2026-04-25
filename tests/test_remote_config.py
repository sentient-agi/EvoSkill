"""Cycle 1: Tests for remote execution config parsing."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.cli.config import (
    AWSConfig,
    DaytonaConfig,
    DownloadConfig,
    RemoteConfig,
    load_config,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _write_config(tmp_path: Path, toml_content: str) -> Path:
    """Set up a minimal project with the given config.toml content."""
    evoskill_dir = tmp_path / ".evoskill"
    evoskill_dir.mkdir()
    (evoskill_dir / "config.toml").write_text(toml_content)
    (evoskill_dir / "task.md").write_text("Test task\n---\nNo constraints")
    # load_config needs a git root or .evoskill dir to find project root
    return tmp_path


MINIMAL_TOML = """\
[harness]
name = "claude"
"""


# ── Daytona config ───────────────────────────────────────────────────────────

def test_remote_config_daytona_parses(tmp_path):
    """Daytona config is parsed with all fields."""
    root = _write_config(tmp_path, MINIMAL_TOML + """
[remote]
target = "daytona"

[remote.daytona]
api_key = "dtk_test123"
image = "evoskill:latest"
cpu = 2
memory = 4
disk = 8
timeout = 0
""")
    cfg = load_config(start=root)
    assert cfg.remote is not None
    assert cfg.remote.target == "daytona"
    assert cfg.remote.daytona is not None
    assert cfg.remote.daytona.api_key == "dtk_test123"
    assert cfg.remote.daytona.image == "evoskill:latest"
    assert cfg.remote.daytona.cpu == 2
    assert cfg.remote.daytona.memory == 4
    assert cfg.remote.daytona.disk == 8
    assert cfg.remote.daytona.timeout == 0


# ── AWS config ───────────────────────────────────────────────────────────────

def test_remote_config_aws_parses(tmp_path):
    """AWS config is parsed with all fields."""
    root = _write_config(tmp_path, MINIMAL_TOML + """
[remote]
target = "aws"

[remote.aws]
region = "us-west-2"
s3_bucket = "my-bucket"
ecr_repo = "my-ecr"
ecs_cluster = "prod"
task_cpu = "2048"
task_memory = "8192"
""")
    cfg = load_config(start=root)
    assert cfg.remote is not None
    assert cfg.remote.target == "aws"
    assert cfg.remote.aws is not None
    assert cfg.remote.aws.region == "us-west-2"
    assert cfg.remote.aws.s3_bucket == "my-bucket"
    assert cfg.remote.aws.ecr_repo == "my-ecr"
    assert cfg.remote.aws.ecs_cluster == "prod"
    assert cfg.remote.aws.task_cpu == "2048"
    assert cfg.remote.aws.task_memory == "8192"


# ── Download config defaults ─────────────────────────────────────────────────

def test_remote_config_download_defaults(tmp_path):
    """Missing [remote.download] uses sensible defaults."""
    root = _write_config(tmp_path, MINIMAL_TOML + """
[remote]
target = "daytona"

[remote.daytona]
api_key = "dtk_test"
""")
    cfg = load_config(start=root)
    dl = cfg.remote.download
    assert dl.all_branches is False
    assert dl.cache is False
    assert dl.reports is True
    assert dl.feedback_history is False


def test_remote_config_download_overrides(tmp_path):
    """[remote.download] values override defaults."""
    root = _write_config(tmp_path, MINIMAL_TOML + """
[remote]
target = "daytona"

[remote.daytona]
api_key = "dtk_test"

[remote.download]
all_branches = true
cache = true
reports = false
feedback_history = true
""")
    cfg = load_config(start=root)
    dl = cfg.remote.download
    assert dl.all_branches is True
    assert dl.cache is True
    assert dl.reports is False
    assert dl.feedback_history is True


# ── Missing remote config ────────────────────────────────────────────────────

def test_remote_config_missing(tmp_path):
    """No [remote] section → remote is None."""
    root = _write_config(tmp_path, MINIMAL_TOML)
    cfg = load_config(start=root)
    assert cfg.remote is None


# ── API key from env var ─────────────────────────────────────────────────────

def test_daytona_api_key_from_env(tmp_path):
    """api_key not in toml but DAYTONA_API_KEY in env → picked up."""
    root = _write_config(tmp_path, MINIMAL_TOML + """
[remote]
target = "daytona"

[remote.daytona]
image = "evoskill:latest"
""")
    with patch.dict(os.environ, {"DAYTONA_API_KEY": "dtk_from_env"}):
        cfg = load_config(start=root)
    assert cfg.remote.daytona.api_key == "dtk_from_env"


def test_daytona_toml_key_overrides_env(tmp_path):
    """api_key in toml takes precedence over env var."""
    root = _write_config(tmp_path, MINIMAL_TOML + """
[remote]
target = "daytona"

[remote.daytona]
api_key = "dtk_from_toml"
""")
    with patch.dict(os.environ, {"DAYTONA_API_KEY": "dtk_from_env"}):
        cfg = load_config(start=root)
    assert cfg.remote.daytona.api_key == "dtk_from_toml"


# ── Invalid target ───────────────────────────────────────────────────────────

def test_remote_config_invalid_target(tmp_path):
    """Unsupported target raises ValueError."""
    root = _write_config(tmp_path, MINIMAL_TOML + """
[remote]
target = "gcp"
""")
    with pytest.raises(ValueError, match="gcp"):
        load_config(start=root)


# ── Daytona defaults ─────────────────────────────────────────────────────────

def test_daytona_config_defaults(tmp_path):
    """Daytona config uses defaults for optional fields."""
    root = _write_config(tmp_path, MINIMAL_TOML + """
[remote]
target = "daytona"

[remote.daytona]
api_key = "dtk_test"
""")
    cfg = load_config(start=root)
    d = cfg.remote.daytona
    assert d.image == ""
    assert d.cpu == 4
    assert d.memory == 8
    assert d.disk == 10
    assert d.timeout == 0


# ── AWS defaults ─────────────────────────────────────────────────────────────

def test_aws_config_defaults(tmp_path):
    """AWS config uses defaults for optional fields."""
    root = _write_config(tmp_path, MINIMAL_TOML + """
[remote]
target = "aws"

[remote.aws]
region = "us-east-1"
s3_bucket = "my-bucket"
ecr_repo = "my-ecr"
""")
    cfg = load_config(start=root)
    a = cfg.remote.aws
    assert a.ecs_cluster == "default"
    assert a.task_cpu == "4096"
    assert a.task_memory == "16384"
