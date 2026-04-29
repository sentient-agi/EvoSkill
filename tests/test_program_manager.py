from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from src.registry.manager import ProgramManager, ProgramManagerError


def test_discard_checks_out_parent_program_when_no_non_program_branch(
    monkeypatch,
    tmp_path,
) -> None:
    manager = ProgramManager(cwd=tmp_path)
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(manager, "_git_current_branch", lambda: "program/iter-skill-1")
    monkeypatch.setattr(
        manager,
        "_git_list_branches",
        lambda: ["program/base", "program/iter-skill-1"],
    )
    monkeypatch.setattr(
        manager,
        "_read_config_from_branch",
        lambda branch: SimpleNamespace(parent="program/base"),
    )
    monkeypatch.setattr(
        manager,
        "_git_checkout",
        lambda branch: calls.append(("checkout", branch)),
    )
    monkeypatch.setattr(
        manager,
        "_git_branch_delete",
        lambda branch: calls.append(("delete", branch)),
    )
    monkeypatch.setattr(manager, "_git_list_tags", lambda: [])
    monkeypatch.setattr(
        manager,
        "_git_tag_delete",
        lambda tag: calls.append(("tag_delete", tag)),
    )

    manager.discard("iter-skill-1")

    assert calls[:2] == [
        ("checkout", "program/base"),
        ("delete", "program/iter-skill-1"),
    ]


def test_git_checkout_raises_clear_error_when_stash_fails(monkeypatch, tmp_path) -> None:
    manager = ProgramManager(cwd=tmp_path)

    monkeypatch.setattr(manager, "_git_current_branch", lambda: "branch-name")

    def fake_run_git(args, check=True):
        if args == ["status", "--porcelain"]:
            return subprocess.CompletedProcess(args, 0, stdout=" M file.py\n", stderr="")
        if args[:3] == ["stash", "push", "-u"]:
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="cannot stash")
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(manager, "_run_git", fake_run_git)

    with pytest.raises(ProgramManagerError, match="could not stash"):
        manager._git_checkout("program/base")


def test_git_checkout_leaves_stash_when_apply_fails(monkeypatch, tmp_path) -> None:
    manager = ProgramManager(cwd=tmp_path)
    calls: list[list[str]] = []

    monkeypatch.setattr(manager, "_git_current_branch", lambda: "branch-name")

    def fake_run_git(args, check=True):
        calls.append(args)
        if args == ["status", "--porcelain"]:
            return subprocess.CompletedProcess(args, 0, stdout=" M file.py\n", stderr="")
        if args[:3] == ["stash", "push", "-u"]:
            return subprocess.CompletedProcess(args, 0, stdout="Saved", stderr="")
        if args == ["checkout", "program/base"]:
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if args == ["stash", "apply"]:
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="conflict")
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(manager, "_run_git", fake_run_git)

    with pytest.raises(ProgramManagerError, match="auto-stash was left"):
        manager._git_checkout("program/base")

    assert ["stash", "drop"] not in calls
