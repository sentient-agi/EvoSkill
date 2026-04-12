from __future__ import annotations

from types import SimpleNamespace

from src.registry.manager import ProgramManager


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
