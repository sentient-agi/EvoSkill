"""The skills-load check asks the skills' real consumer: Claude Code.

Claude Code reads <name>/SKILL.md's frontmatter and reaches a skill's companion files through the skill's
base directory, so a companion `.md` is the skill's own material. The check re-implements none of Claude
Code's rules; it records the count. A consumer without a shipped loader SKIPs (the loop turns that into a
HALT: an audit that cannot run must not read as one that passed).
"""
from __future__ import annotations

from pathlib import Path

from src.orchestrator import invariants as I


def _skill(root: Path, name: str, description: str = "Use when the task needs it.", *, extra_fm: str = "",
           companion: str | None = None, body: str = "# body\n") -> Path:
    d = root / name
    d.mkdir(parents=True)
    fm = f"---\nname: {name}\n"
    if description is not None:
        fm += f"description: {description}\n"
    fm += extra_fm + "---\n"
    (d / "SKILL.md").write_text(fm + body)
    if companion:
        (d / companion).write_text("# companion notes\nno frontmatter here\n")
    return d


def test_a_companion_md_is_the_skills_own_material_for_claude_code(tmp_path):
    st = tmp_path / "staging"
    _skill(st, "excel-audit-fix")
    _skill(st, "excel-model-completion", companion="accounting-conventions.md")
    cc = I.check_skills_load_through_consumer(st, consumer="claude_code")
    assert cc.status == "PASS", cc.detail
    assert cc.evidence["written"] == ["excel-audit-fix", "excel-model-completion"]


def test_default_consumer_is_claude_code(tmp_path):
    st = tmp_path / "staging"
    _skill(st, "one", companion="notes.md")
    c = I.check_skills_load_through_consumer(st)
    assert c.status == "PASS" and c.evidence["consumer"] == I.CONSUMER_CLAUDE_CODE


def test_a_consumer_without_a_shipped_loader_skips_and_a_missing_staging_skips(tmp_path):
    st = tmp_path / "staging"
    _skill(st, "fine")
    c = I.check_skills_load_through_consumer(st, consumer="direct")
    assert c.status == "SKIP" and "no skill loader for consumer 'direct'" in c.detail
    assert I.check_skills_load_through_consumer(tmp_path / "nowhere").status == "SKIP"


def test_claude_code_is_its_own_judge(tmp_path):
    """For Claude Code the check re-implements nothing: a blank description (Claude Code uses the body's first
    line), a flat .md in the root (not a skill), colliding names and extra frontmatter keys all PASS, with a
    count of the <name>/SKILL.md written."""
    st = tmp_path / "staging"
    _skill(st, "fine")
    _skill(st, "mute", description="")
    _skill(st, "sneaky", extra_fm="allowed-tools: Bash\n")
    (st / "loose.md").write_text("---\nname: loose\ndescription: x\n---\nbody\n")
    c = I.check_skills_load_through_consumer(st, consumer="claude_code")
    assert c.status == "PASS" and c.evidence["written"] == ["fine", "mute", "sneaky"], c.detail
    assert not hasattr(I, "_claude_code_skill_frontmatter") and not hasattr(I, "_cc_skill_policy_problems")
