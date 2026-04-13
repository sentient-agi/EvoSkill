"""Tests for Codex skill discovery via symlink.

EvoSkill writes skills to .claude/skills/.
Codex scans .agents/skills/ (not .claude/skills/).
The fix: symlink .agents/skills/ -> .claude/skills/.

All tests use pytest's tmp_path fixture to avoid touching the real filesystem.
"""

import os
import pytest
from pathlib import Path


class TestEnsureAgentsSkillsSymlink:

    def test_creates_symlink_when_nothing_exists(self, tmp_path):
        from src.harness.codex.skill_discovery import ensure_agents_skills_symlink
        result = ensure_agents_skills_symlink(tmp_path)
        link = tmp_path / ".agents" / "skills"
        source = tmp_path / ".claude" / "skills"
        assert result is True
        assert link.is_symlink()
        assert link.resolve() == source.resolve()
        assert source.is_dir()

    def test_noop_when_correct_symlink_exists(self, tmp_path):
        from src.harness.codex.skill_discovery import ensure_agents_skills_symlink
        ensure_agents_skills_symlink(tmp_path)  # first call creates
        result = ensure_agents_skills_symlink(tmp_path)  # second call is noop
        assert result is False

    def test_replaces_stale_symlink(self, tmp_path):
        from src.harness.codex.skill_discovery import ensure_agents_skills_symlink
        link = tmp_path / ".agents" / "skills"
        link.parent.mkdir(parents=True)
        os.symlink("/tmp/wrong_target", str(link))
        result = ensure_agents_skills_symlink(tmp_path)
        assert result is True
        assert link.resolve() == (tmp_path / ".claude" / "skills").resolve()

    def test_skips_real_directory(self, tmp_path):
        from src.harness.codex.skill_discovery import ensure_agents_skills_symlink
        link = tmp_path / ".agents" / "skills"
        link.mkdir(parents=True)
        (link / "existing_file.txt").write_text("data")
        result = ensure_agents_skills_symlink(tmp_path)
        assert result is False
        assert not link.is_symlink()
        assert (link / "existing_file.txt").read_text() == "data"

    def test_creates_claude_skills_if_missing(self, tmp_path):
        from src.harness.codex.skill_discovery import ensure_agents_skills_symlink
        ensure_agents_skills_symlink(tmp_path)
        source = tmp_path / ".claude" / "skills"
        assert source.is_dir()

    def test_idempotent_multiple_calls(self, tmp_path):
        from src.harness.codex.skill_discovery import ensure_agents_skills_symlink
        ensure_agents_skills_symlink(tmp_path)
        ensure_agents_skills_symlink(tmp_path)
        ensure_agents_skills_symlink(tmp_path)
        link = tmp_path / ".agents" / "skills"
        assert link.is_symlink()
        assert link.resolve() == (tmp_path / ".claude" / "skills").resolve()

    def test_skill_files_visible_through_symlink(self, tmp_path):
        from src.harness.codex.skill_discovery import ensure_agents_skills_symlink
        ensure_agents_skills_symlink(tmp_path)
        # Write a skill to .claude/skills/
        skill_dir = tmp_path / ".claude" / "skills" / "test-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# Test")
        # Verify visible through .agents/skills/ symlink
        agents_skill = tmp_path / ".agents" / "skills" / "test-skill" / "SKILL.md"
        assert agents_skill.exists()
        assert agents_skill.read_text() == "# Test"

    def test_returns_bool(self, tmp_path):
        from src.harness.codex.skill_discovery import ensure_agents_skills_symlink
        result_first = ensure_agents_skills_symlink(tmp_path)
        result_second = ensure_agents_skills_symlink(tmp_path)
        assert isinstance(result_first, bool)
        assert isinstance(result_second, bool)

    def test_agents_parent_dir_created_if_missing(self, tmp_path):
        from src.harness.codex.skill_discovery import ensure_agents_skills_symlink
        # Ensure .agents/ does not exist at all
        agents_dir = tmp_path / ".agents"
        assert not agents_dir.exists()
        ensure_agents_skills_symlink(tmp_path)
        assert agents_dir.is_dir()

    def test_symlink_target_is_absolute_path(self, tmp_path):
        from src.harness.codex.skill_discovery import ensure_agents_skills_symlink
        ensure_agents_skills_symlink(tmp_path)
        link = tmp_path / ".agents" / "skills"
        # os.readlink returns the raw target string stored in the symlink
        raw_target = os.readlink(str(link))
        assert os.path.isabs(raw_target), (
            f"Symlink target must be absolute, got: {raw_target!r}"
        )
