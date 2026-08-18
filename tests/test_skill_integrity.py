"""Tests for the merge-conflict guard on skill files.

Regression target: the 0726 incident where `git stash pop` (check=False) wrote
conflict markers into a tracked SKILL.md and 28 solver rollouts read it without
anything raising. The fixture text below is the REAL shape git produced in that
run, not an invented approximation.
"""
from __future__ import annotations

import pytest

from src.skill_integrity import (
    SkillConflictError,
    assert_skills_clean,
    assert_text_clean,
    find_conflict_markers,
    scan_skills_dir,
)

# The literal marker triple git emitted in the incident (manager.py stash pop).
CORRUPT = """---
name: libreoffice-formula-compatibility
description: MUST invoke before writing ANY formula.
---
# LibreOffice Formula Compatibility

After deleting rows, clear them:

```python
<<<<<<< Updated upstream
# SAFE — simple IF based on condition
ws['D2'] = '=IF(NOT(ISNA(A2)),A2,"")'
=======
>>>>>>> Stashed changes
if cell.value == '#N/A':
    cell.value = None
```
"""

CLEAN = """---
name: libreoffice-formula-compatibility
description: MUST invoke before writing ANY formula.
---
# LibreOffice Formula Compatibility

Compute values in Python and write them as static values.
"""


class TestMarkerDetection:
    def test_detects_the_real_incident_text(self):
        markers = find_conflict_markers(CORRUPT)
        assert "<<<<<<< Updated upstream" in markers
        assert ">>>>>>> Stashed changes" in markers

    def test_clean_text_has_no_markers(self):
        assert find_conflict_markers(CLEAN) == []

    def test_bare_equals_line_is_NOT_a_marker(self):
        """A `=======` line is a valid setext H1 underline. Rejecting it would
        fail ordinary skill prose, so it must not count on its own."""
        setext = "My Heading\n=======\n\nbody text\n"
        assert find_conflict_markers(setext) == []

    def test_markers_must_be_at_line_start(self):
        """Prose that merely mentions the marker mid-line is not a conflict."""
        prose = "Use `git diff` and look for a <<<<<<< marker in the output.\n"
        assert find_conflict_markers(prose) == []

    def test_deduped_and_order_preserving(self):
        txt = "<<<<<<< a\n>>>>>>> b\n<<<<<<< a\n"
        assert find_conflict_markers(txt) == ["<<<<<<< a", ">>>>>>> b"]

    def test_eight_or_more_angle_brackets_still_caught(self):
        assert find_conflict_markers("<<<<<<<<< x\n") != []


class TestAssertTextClean:
    def test_raises_on_corrupt(self):
        with pytest.raises(SkillConflictError) as ei:
            assert_text_clean(CORRUPT, origin="/tmp/SKILL.md")
        msg = str(ei.value)
        assert "/tmp/SKILL.md" in msg
        assert "Stashed changes" in msg
        # The message must tell the operator how to recover, not just complain.
        assert "evolver_sessions" in msg or "skill_snapshots" in msg

    def test_silent_on_clean(self):
        assert_text_clean(CLEAN, origin="/tmp/SKILL.md")


class TestScanSkillsDir:
    def _mk(self, tmp_path, name, text):
        d = tmp_path / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(text)
        return d / "SKILL.md"

    def test_finds_only_the_corrupt_file(self, tmp_path):
        self._mk(tmp_path, "good", CLEAN)
        bad_path = self._mk(tmp_path, "bad", CORRUPT)
        bad = scan_skills_dir(tmp_path)
        assert list(bad) == [str(bad_path)]

    def test_empty_and_missing_dirs_are_clean(self, tmp_path):
        assert scan_skills_dir(tmp_path) == {}
        assert scan_skills_dir(tmp_path / "nope") == {}

    def test_scans_nested_and_non_SKILL_markdown(self, tmp_path):
        """Harnesses glob differently (alfworld uses rglob('*.md')), so a
        corrupt sibling .md must not slip through."""
        d = tmp_path / "s"
        d.mkdir()
        (d / "REFERENCE.md").write_text(CORRUPT)
        assert len(scan_skills_dir(tmp_path)) == 1

    def test_assert_skills_clean_raises_with_stage_and_path(self, tmp_path):
        self._mk(tmp_path, "bad", CORRUPT)
        with pytest.raises(SkillConflictError) as ei:
            assert_skills_clean(tmp_path, stage="restore of program 'iter-skill-7'")
        msg = str(ei.value)
        assert "iter-skill-7" in msg
        assert "bad/SKILL.md" in msg

    def test_assert_skills_clean_silent_on_clean_tree(self, tmp_path):
        self._mk(tmp_path, "good", CLEAN)
        assert_skills_clean(tmp_path, stage="test")


class TestNoEscapeHatch:
    """The three guards that failed open in this codebase all had an opt-out.
    This one must not acquire one by accident."""

    def test_env_vars_cannot_disable_the_guard(self, tmp_path, monkeypatch):
        d = tmp_path / "bad"
        d.mkdir()
        (d / "SKILL.md").write_text(CORRUPT)
        for var in (
            "EVOSKILL_ALLOW_SKILL_CONFLICTS",
            "EVOSKILL_ALLOW_STALE_SKILLS",
            "EVOSKILL_SKIP_SKILL_CHECK",
        ):
            monkeypatch.setenv(var, "1")
        with pytest.raises(SkillConflictError):
            assert_skills_clean(tmp_path, stage="test")


class TestDiscoverySiteWired:
    def test_direct_harness_discovery_rejects_corrupt_skill(self, tmp_path):
        from src.harness.direct.executor import _discover_skills

        d = tmp_path / "libreoffice-formula-compatibility"
        d.mkdir()
        (d / "SKILL.md").write_text(CORRUPT)
        with pytest.raises(SkillConflictError):
            _discover_skills(tmp_path)

    def test_direct_harness_discovery_accepts_clean_skill(self, tmp_path):
        from src.harness.direct.executor import _discover_skills

        d = tmp_path / "libreoffice-formula-compatibility"
        d.mkdir()
        (d / "SKILL.md").write_text(CLEAN)
        found = _discover_skills(tmp_path)
        assert "libreoffice-formula-compatibility" in found
