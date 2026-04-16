"""Unit tests for skill constraint gating."""

import pytest

from src.loop.constraints import gate, check_all, MAX_SKILL_SIZE, MAX_GROWTH_RATIO


VALID_SKILL = """---
name: test-skill
description: A skill that does testing
---

# How to Test

This skill describes how to write tests.

## Steps
1. Write a test
2. Run it
3. Verify it passes
"""


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_valid_skill_passes(tmp_path):
    skill = _write(tmp_path / "SKILL.md", VALID_SKILL)
    passed, summary = gate(skill)
    assert passed
    assert "frontmatter: name=test-skill" in summary


def test_missing_file_fails(tmp_path):
    passed, summary = gate(tmp_path / "nonexistent.md")
    assert not passed
    assert "not found" in summary


def test_size_limit_enforced(tmp_path):
    # Too big
    huge = "---\nname: big\ndescription: huge\n---\n\n" + ("X" * MAX_SKILL_SIZE * 2)
    skill = _write(tmp_path / "SKILL.md", huge)
    passed, summary = gate(skill)
    assert not passed
    assert "size_limit" in summary
    assert "too large" in summary.lower()


def test_growth_limit_enforced(tmp_path):
    parent_content = "---\nname: small\ndescription: parent\n---\n\n" + ("X" * 1000)
    parent = _write(tmp_path / "parent.md", parent_content)

    # Child 3x parent — exceeds 2x growth limit
    child_content = "---\nname: big-child\ndescription: child\n---\n\n" + ("X" * 3500)
    child = _write(tmp_path / "child.md", child_content)
    passed, summary = gate(child, parent_path=parent)
    assert not passed
    assert "growth_limit" in summary


def test_growth_limit_allowed_within_2x(tmp_path):
    parent_content = "---\nname: parent\ndescription: parent\n---\n\n" + ("X" * 1000)
    parent = _write(tmp_path / "parent.md", parent_content)

    # Child 1.5x parent — within 2x growth limit
    child_content = "---\nname: child\ndescription: child\n---\n\n" + ("X" * 500)
    child = _write(tmp_path / "child.md", child_content)
    passed, _ = gate(child, parent_path=parent)
    assert passed


def test_missing_frontmatter_fails(tmp_path):
    skill = _write(tmp_path / "SKILL.md", "# No Frontmatter Here\n\nJust plain markdown.")
    passed, summary = gate(skill)
    assert not passed
    assert "frontmatter" in summary


def test_invalid_yaml_frontmatter_fails(tmp_path):
    content = "---\nname: test\n  bad: :indent:\n---\n\nbody"
    skill = _write(tmp_path / "SKILL.md", content)
    passed, summary = gate(skill)
    assert not passed


def test_missing_name_field_fails(tmp_path):
    content = "---\ndescription: no name\n---\n\nbody must be long enough for body check to pass"
    skill = _write(tmp_path / "SKILL.md", content)
    passed, summary = gate(skill)
    assert not passed
    assert "name" in summary


def test_binary_content_fails(tmp_path):
    # Include null bytes and other non-printable junk
    content = "---\nname: binary\ndescription: test\n---\n\n"
    content += "\x00\x01\x02\x03" * 100  # binary garbage
    skill = _write(tmp_path / "SKILL.md", content)
    passed, summary = gate(skill)
    assert not passed
    assert "no_binary" in summary or "binary" in summary.lower()


def test_answer_leakage_warns(tmp_path):
    # Skill contains a training answer — should pass but warn
    content = VALID_SKILL + "\n\nThe answer is always 42.5 for this category.\n"
    skill = _write(tmp_path / "SKILL.md", content)
    passed, summary = gate(skill, training_answers=["42.5", "100"])
    assert passed
    assert "WARNING" in summary


def test_no_leakage_no_warning(tmp_path):
    skill = _write(tmp_path / "SKILL.md", VALID_SKILL)
    passed, summary = gate(skill, training_answers=["42.5", "99.99"])
    assert passed
    assert "WARNING" not in summary
