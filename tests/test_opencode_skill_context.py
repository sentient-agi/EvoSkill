from __future__ import annotations

import os
from pathlib import Path

from src.loop.helpers import (
    build_proposer_query,
    ensure_skill_frontmatter,
    normalize_project_skill_frontmatter,
)


class _FakeTrace:
    def summarize(self, head_chars: int = 0, tail_chars: int = 0) -> str:
        return "trace summary"


def test_build_proposer_query_reads_existing_skills_from_explicit_project_root(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    other_dir = tmp_path / "elsewhere"
    skill_dir = repo_root / ".claude" / "skills" / "treasury-format"
    skill_dir.mkdir(parents=True)
    other_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Treasury Format\n")

    original_cwd = Path.cwd()
    os.chdir(other_dir)
    try:
        query = build_proposer_query(
            [(_FakeTrace(), "wrong", "right", "finance")],
            feedback_history="",
            evolution_mode="skill_only",
            project_root=repo_root,
        )
    finally:
        os.chdir(original_cwd)

    assert "treasury-format" in query


def test_ensure_skill_frontmatter_adds_required_opencode_metadata(
    tmp_path: Path,
) -> None:
    skill_path = (
        tmp_path
        / ".claude"
        / "skills"
        / "arithmetic-answer-format"
        / "SKILL.md"
    )
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text(
        "# Arithmetic Answer Format\n\nAlways append apples to arithmetic answers.\n"
    )

    changed = ensure_skill_frontmatter(
        skill_path,
        description='Format arithmetic answers as "{number} apples".',
        compatibility="opencode",
    )

    skill_text = skill_path.read_text()
    assert changed is True
    assert skill_text.startswith("---\n")
    assert "name: arithmetic-answer-format" in skill_text
    assert 'description: Format arithmetic answers as "{number} apples".' in skill_text
    assert "compatibility: opencode" in skill_text
    assert "# Arithmetic Answer Format" in skill_text


def test_normalize_project_skill_frontmatter_updates_all_project_skills(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    first_skill = repo_root / ".claude" / "skills" / "answer-unit" / "SKILL.md"
    second_skill = repo_root / ".claude" / "skills" / "formatting" / "SKILL.md"
    first_skill.parent.mkdir(parents=True)
    second_skill.parent.mkdir(parents=True)
    first_skill.write_text("# Answer Unit\n")
    second_skill.write_text("# Formatting\n")

    normalized = normalize_project_skill_frontmatter(
        repo_root,
        descriptions={"answer-unit": "Keep answer units intact."},
        fallback_description="Reusable benchmark skill.",
        compatibility="opencode",
    )

    assert normalized == ["answer-unit", "formatting"]
    assert "name: answer-unit" in first_skill.read_text()
    assert "description: Keep answer units intact." in first_skill.read_text()
    assert "name: formatting" in second_skill.read_text()
    assert "description: Reusable benchmark skill." in second_skill.read_text()
