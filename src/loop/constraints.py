"""Constraint gating for evolved skills.

Validates skills BEFORE expensive evaluation to catch degenerate mutations early.
All checks are free (no LLM calls) — pure text/structure analysis.
"""

import re
import yaml
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConstraintResult:
    passed: bool
    constraint: str
    message: str


MAX_SKILL_SIZE = 15_000       # 15KB — forces concise methodology
MAX_GROWTH_RATIO = 2.0        # Edit can't more than double the parent's size
MIN_SKILL_SIZE = 50           # Must have meaningful content
MAX_DESCRIPTION_LENGTH = 500  # Frontmatter description


def check_all(
    skill_path: Path,
    parent_path: Path | None = None,
    training_answers: list[str] | None = None,
) -> list[ConstraintResult]:
    """Run all constraint checks on a skill file.

    Args:
        skill_path: Path to the SKILL.md file to validate.
        parent_path: Path to the parent skill (for growth limit on edits). None for new skills.
        training_answers: Ground truth answers from training set (for leakage detection).

    Returns:
        List of ConstraintResult. All must have passed=True for the skill to proceed.
    """
    results = []

    if not skill_path.exists():
        return [ConstraintResult(False, "exists", f"Skill file not found: {skill_path}")]

    content = skill_path.read_text()

    # 1. Non-empty
    if len(content.strip()) < MIN_SKILL_SIZE:
        results.append(ConstraintResult(False, "non_empty", f"Skill too small ({len(content.strip())} chars, min {MIN_SKILL_SIZE})"))
    else:
        results.append(ConstraintResult(True, "non_empty", "OK"))

    # 2. Size limit
    if len(content) > MAX_SKILL_SIZE:
        results.append(ConstraintResult(False, "size_limit", f"Skill too large ({len(content):,} chars, max {MAX_SKILL_SIZE:,})"))
    else:
        results.append(ConstraintResult(True, "size_limit", f"{len(content):,} chars"))

    # 3. Growth limit (edits only)
    if parent_path is not None and parent_path.exists():
        parent_size = len(parent_path.read_text())
        if parent_size > 0 and len(content) > parent_size * MAX_GROWTH_RATIO:
            results.append(ConstraintResult(
                False, "growth_limit",
                f"Skill grew {len(content)/parent_size:.1f}x (max {MAX_GROWTH_RATIO}x). "
                f"Parent: {parent_size:,} chars → New: {len(content):,} chars"
            ))
        else:
            results.append(ConstraintResult(True, "growth_limit", "OK"))

    # 4. YAML frontmatter structure
    fm_result = _check_frontmatter(content)
    results.append(fm_result)

    # 5. No binary/non-printable content
    non_printable = sum(1 for c in content if not c.isprintable() and c not in '\n\r\t')
    if non_printable > max(len(content) * 0.01, 10):
        results.append(ConstraintResult(False, "no_binary", f"{non_printable} non-printable chars detected"))
    else:
        results.append(ConstraintResult(True, "no_binary", "OK"))

    # 6. Answer leakage (soft — warn but pass)
    if training_answers:
        leaked = []
        content_lower = content.lower()
        for answer in training_answers:
            answer_str = str(answer).strip()
            # Only flag exact matches for non-trivial answers (>3 chars)
            if len(answer_str) > 3 and answer_str.lower() in content_lower:
                leaked.append(answer_str)
        if leaked:
            results.append(ConstraintResult(
                True, "answer_leakage",  # Soft: passes but warns
                f"WARNING: skill contains training answers: {leaked[:5]}. Possible overfitting."
            ))
        else:
            results.append(ConstraintResult(True, "answer_leakage", "OK"))

    return results


def _check_frontmatter(content: str) -> ConstraintResult:
    """Validate YAML frontmatter structure."""
    # Must start with ---
    if not content.strip().startswith("---"):
        return ConstraintResult(False, "frontmatter", "Missing YAML frontmatter (no opening ---)")

    # Find closing ---
    parts = content.split("---", 2)
    if len(parts) < 3:
        return ConstraintResult(False, "frontmatter", "Missing closing --- for YAML frontmatter")

    yaml_text = parts[1].strip()
    if not yaml_text:
        return ConstraintResult(False, "frontmatter", "Empty YAML frontmatter")

    # Parse YAML
    try:
        fm = yaml.safe_load(yaml_text)
    except yaml.YAMLError as e:
        return ConstraintResult(False, "frontmatter", f"Invalid YAML: {e}")

    if not isinstance(fm, dict):
        return ConstraintResult(False, "frontmatter", "Frontmatter must be a YAML mapping")

    # Required fields
    if "name" not in fm:
        return ConstraintResult(False, "frontmatter", "Missing required field: name")
    if "description" not in fm:
        return ConstraintResult(False, "frontmatter", "Missing required field: description")

    # Description length
    desc = str(fm.get("description", ""))
    if len(desc) > MAX_DESCRIPTION_LENGTH:
        return ConstraintResult(False, "frontmatter", f"Description too long ({len(desc)} chars, max {MAX_DESCRIPTION_LENGTH})")

    # Body after frontmatter
    body = parts[2].strip()
    if len(body) < 20:
        return ConstraintResult(False, "frontmatter", "Skill body after frontmatter is too short")

    return ConstraintResult(True, "frontmatter", f"name={fm['name']}")


def gate(
    skill_path: Path,
    parent_path: Path | None = None,
    training_answers: list[str] | None = None,
) -> tuple[bool, str]:
    """Run all constraints and return (passed, summary).

    Returns:
        (True, summary) if all hard constraints passed.
        (False, summary) if any hard constraint failed.
    """
    results = check_all(skill_path, parent_path, training_answers)
    failed = [r for r in results if not r.passed]
    warnings = [r for r in results if r.passed and "WARNING" in r.message]

    lines = []
    for r in results:
        icon = "✓" if r.passed else "✗"
        lines.append(f"  {icon} {r.constraint}: {r.message}")

    summary = "\n".join(lines)

    if failed:
        return False, f"Constraint check FAILED:\n{summary}"
    elif warnings:
        return True, f"Constraint check PASSED with warnings:\n{summary}"
    else:
        return True, f"Constraint check PASSED:\n{summary}"
