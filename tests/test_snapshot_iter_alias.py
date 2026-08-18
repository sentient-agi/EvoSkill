"""Tests for cache-replay snapshot iter aliasing / DIFF pin resolution."""

from pathlib import Path

from src.agent_profiles.spreadsheetbench_agent.task_workspace import (
    ensure_snapshot_iter_alias,
    resolve_snapshot_iter,
)


def _write_snap(root: Path, task_id: str, n: int, body: str = "FAIL\n") -> Path:
    d = root / "_snapshots" / task_id / f"iter_{n}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "DIFF.txt").write_text(body)
    (d / "out.xlsx").write_bytes(b"fake")
    return d


def test_ensure_snapshot_iter_alias_copies_lower(tmp_path: Path):
    tid = "46646"
    src = _write_snap(tmp_path, tid, 0, "FAIL — iter0\n")
    out = ensure_snapshot_iter_alias(tid, tmp_path, 1)
    assert out is not None
    assert out.name == "iter_1"
    assert (out / "DIFF.txt").read_text() == "FAIL — iter0\n"
    # source untouched
    assert src.is_dir()


def test_ensure_snapshot_iter_alias_noop_when_present(tmp_path: Path):
    tid = "46646"
    _write_snap(tmp_path, tid, 0)
    _write_snap(tmp_path, tid, 1, "FAIL — iter1\n")
    out = ensure_snapshot_iter_alias(tid, tmp_path, 1)
    assert out is not None
    assert (out / "DIFF.txt").read_text() == "FAIL — iter1\n"


def test_resolve_snapshot_iter_falls_back(tmp_path: Path):
    tid = "46646"
    _write_snap(tmp_path, tid, 0)
    assert resolve_snapshot_iter(tid, tmp_path, 1) == 0
    assert resolve_snapshot_iter(tid, tmp_path, 0) == 0
