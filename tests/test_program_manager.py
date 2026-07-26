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


def test_git_commit_raises_clear_error_when_commit_fails(monkeypatch, tmp_path) -> None:
    manager = ProgramManager(cwd=tmp_path)

    def fake_run_git(args, check=True):
        if args == ["diff", "--cached", "--quiet"]:
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="")
        if args == ["commit", "-m", "save program"]:
            return subprocess.CompletedProcess(
                args,
                128,
                stdout="",
                stderr=(
                    "fatal: cannot update ref 'refs/heads/program/iter-skill-1': "
                    "trying to write ref 'refs/heads/program/iter-skill-1' "
                    "with nonexistent object 2e1c9de11e3256cd9efd81abcf625f9da47ffd1b"
                ),
            )
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(manager, "_run_git", fake_run_git)

    with pytest.raises(ProgramManagerError, match="Cannot save EvoSkill program changes"):
        manager._git_commit("save program")


# ---------------------------------------------------------------------------
# update_frontier — frontier size must never exceed max_size, even when a
# member has no readable score. get_frontier_with_scores() drops score-less /
# unreadable members, so sizing the frontier off that filtered list undercounts
# a full frontier and lets update_frontier push it past max_size.
# ---------------------------------------------------------------------------


def _init_repo_with_programs(tmp_path, n):
    """A real git repo with a base program and n iter-* programs."""
    from src.registry.models import ProgramConfig

    def _cfg(name):
        return ProgramConfig(name=name, system_prompt={"text": "p"}, allowed_tools=[])

    for cmd in (
        ["git", "init", "-q", "-b", "main"],
        ["git", "config", "user.email", "t@example.com"],
        ["git", "config", "user.name", "t"],
    ):
        subprocess.run(cmd, cwd=tmp_path, check=True)
    (tmp_path / "readme.md").write_text("x")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=tmp_path, check=True)

    manager = ProgramManager(cwd=tmp_path)
    manager.create_program("base", _cfg("base"))
    for i in range(1, n + 1):
        manager.create_program(f"iter-{i}", _cfg(f"iter-{i}"), parent="base")
    return manager


def test_update_frontier_does_not_overflow_when_member_has_no_score(tmp_path) -> None:
    manager = _init_repo_with_programs(tmp_path, 5)

    # A full frontier whose members were marked before evaluation (the base is
    # marked via mark_frontier() before its score is committed) — none carry a
    # readable score, so get_frontier_with_scores() returns [].
    for i in (1, 2, 3):
        manager.mark_frontier(f"iter-{i}")
    assert len(manager.get_frontier()) == 3

    # With no scored member to compare against, a newcomer must be declined
    # rather than pushing the frontier past max_size.
    added = manager.update_frontier("iter-4", 0.9, max_size=3)
    assert added is False
    assert len(manager.get_frontier()) == 3


def test_update_frontier_does_not_overflow_on_transient_read_failure(
    monkeypatch, tmp_path
) -> None:
    manager = _init_repo_with_programs(tmp_path, 5)
    for i, s in ((1, 0.5), (2, 0.6), (3, 0.7)):
        assert manager.update_frontier(f"iter-{i}", s, max_size=3) is True
    assert len(manager.get_frontier()) == 3

    # One frontier member's config becomes unreadable — get_frontier_with_scores
    # silently drops it, so len(scored) undercounts the full frontier.
    original = manager._read_config_from_branch

    def flaky(branch):
        if branch.endswith("iter-2"):
            raise RuntimeError("transient read failure")
        return original(branch)

    monkeypatch.setattr(manager, "_read_config_from_branch", flaky)

    manager.update_frontier("iter-4", 0.65, max_size=3)
    assert len(manager.get_frontier()) == 3


def test_update_frontier_rescoring_existing_member_keeps_size(tmp_path) -> None:
    manager = _init_repo_with_programs(tmp_path, 5)
    for i, s in ((1, 0.5), (2, 0.6), (3, 0.7)):
        manager.update_frontier(f"iter-{i}", s, max_size=3)

    before = sorted(manager.get_frontier())
    # Re-evaluating a program already on the frontier updates its score in
    # place; it must not change frontier membership or size.
    assert manager.update_frontier("iter-2", 0.99, max_size=3) is True
    assert sorted(manager.get_frontier()) == before
    assert dict(manager.get_frontier_with_scores())["iter-2"] == 0.99


def test_update_frontier_evicts_worst_when_beaten(tmp_path) -> None:
    manager = _init_repo_with_programs(tmp_path, 5)
    for i, s in ((1, 0.30), (2, 0.50), (3, 0.70)):
        manager.update_frontier(f"iter-{i}", s, max_size=3)

    # Beats the worst (0.30) -> evicts iter-1.
    assert manager.update_frontier("iter-4", 0.60, max_size=3) is True
    assert "iter-1" not in manager.get_frontier()
    assert len(manager.get_frontier()) == 3

    # Worse than the worst remaining (0.50) -> declined.
    assert manager.update_frontier("iter-5", 0.40, max_size=3) is False
    assert len(manager.get_frontier()) == 3
