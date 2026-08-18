"""SSBV2_ROOT resolution must produce ONE spelling of the corpus directory.

resolve_base() is the single place that decides which of the two checkout layouts a
root uses. task_workspace.py then calls .resolve() on the result and uses it as a
str.replace() prefix to rewrite host golden paths into container paths, while
TASK_INDEX's golden paths come from resolve_base() itself. If those two disagree by so
much as a `..` segment the rewrite matches nothing -- which was a silent no-op that
shipped host paths into containers, and is now a hard failure that blames a root the
operator never configured.

There was no coverage of any of this before 2026-08-18.
"""
import os
from pathlib import Path

import pytest

from src.agent_profiles.ssbv2_agent.dataset import resolve_base


def _corpus(tmp_path: Path, layout: str) -> Path:
    """Build a minimal checkout in one of the two layouts; return the ROOT."""
    root = tmp_path / "SpreadsheetBench-v2"
    base = root / "data" / "spreadsheetbench-v2" if layout == "data" else root / "spreadsheetbench-v2"
    (base / "Debugging" / "01_02").mkdir(parents=True)
    (base / "Debugging" / "dataset.json").write_text("[]")
    return root


@pytest.mark.parametrize("layout", ["data", "flat"])
def test_canonical_absolute_root(tmp_path, layout):
    root = _corpus(tmp_path, layout)
    got = resolve_base(root)
    assert got.is_absolute()
    assert got == got.resolve(), "resolve_base must return a canonical path"
    assert (got / "Debugging" / "dataset.json").is_file()


@pytest.mark.parametrize("layout", ["data", "flat"])
def test_root_with_dotdot_segment(tmp_path, layout):
    """`/a/b/../SpreadsheetBench-v2` must give the same answer as the clean path."""
    root = _corpus(tmp_path, layout)
    (tmp_path / "side").mkdir()
    noisy = tmp_path / "side" / ".." / "SpreadsheetBench-v2"
    assert resolve_base(noisy) == resolve_base(root)
    assert ".." not in str(resolve_base(noisy))


@pytest.mark.parametrize("layout", ["data", "flat"])
def test_symlinked_root(tmp_path, layout):
    root = _corpus(tmp_path, layout)
    link = tmp_path / "ssb-link"
    link.symlink_to(root, target_is_directory=True)
    assert resolve_base(link) == resolve_base(root)


@pytest.mark.parametrize("layout", ["data", "flat"])
def test_relative_root(tmp_path, layout, monkeypatch):
    root = _corpus(tmp_path, layout)
    monkeypatch.chdir(tmp_path)
    assert resolve_base(Path("SpreadsheetBench-v2")) == resolve_base(root)


def test_prefix_agrees_with_task_workspace(tmp_path):
    """The exact agreement the container-path rewrite depends on.

    task_workspace.py builds its replace() prefix as
    `str(resolve_base(root).resolve())`; golden paths come from resolve_base(root).
    A golden path must therefore START with that prefix, for every spelling of root.
    """
    root = _corpus(tmp_path, "data")
    (tmp_path / "side").mkdir()
    link = tmp_path / "link"; link.symlink_to(root, target_is_directory=True)
    for spelling in (root, tmp_path / "side" / ".." / "SpreadsheetBench-v2", link):
        golden = resolve_base(spelling) / "Debugging" / "01_02" / "1_answer.xlsx"
        prefix = str(resolve_base(spelling).resolve())
        assert str(golden).startswith(prefix), f"prefix mismatch for {spelling}"


def test_unknown_layout_falls_back_to_root(tmp_path):
    empty = tmp_path / "nothing-here"; empty.mkdir()
    assert resolve_base(empty) == empty.resolve()
