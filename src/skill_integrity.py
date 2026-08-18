"""Reject skill files corrupted by a failed VCS merge.

WHY THIS EXISTS: `.claude/skills/` is git-tracked, so the registry's
stash -> checkout -> stash-pop cycle can leave 3-way-merge conflict markers
*inside* a SKILL.md. `_git_checkout` ran `git stash pop` with `check=False`, so a
conflicting pop reported nothing. The result was iter-skill-7 carrying 133 of its
228 lines inside a `<<<<<<< Updated upstream / ======= / >>>>>>> Stashed changes`
block, spliced through a python code fence — and every one of the 28 iteration-8
rollouts that opened the skill read it. Nothing raised; the run completed; the
numbers looked normal. It also injected phantom "Rule 3/4/5" headings that a trace
audit initially blamed on the model.

DESIGN CHOICES, both deliberate:

1. RAISE, DO NOT REPAIR. A conflict block holds two candidate texts and no way to
   know which the author meant. Keeping either one silently ships a skill the
   evolver did not write, which is quietly-invalid science — strictly worse than a
   stopped run. (Contrast `read_feedback_history`, which strips markers on read:
   that file is regenerable narration, not the artifact under test.)

2. NO ESCAPE-HATCH ENV VAR. Three earlier guards in this codebase failed open, and
   an opt-out is how the fourth would. There is no legitimate reason to roll out a
   conflicted skill, so there is no flag to allow it.

Detection keys on a line starting with seven `<` or seven `>`. Those are git's own
markers and are effectively impossible in legitimate markdown. A bare `=======`
line is deliberately NOT treated as a marker on its own: it is a valid setext H1
underline, and matching it would reject ordinary skill prose.
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = [
    "CONFLICT_LINE_RE",
    "SkillConflictError",
    "find_conflict_markers",
    "assert_text_clean",
    "scan_skills_dir",
    "assert_skills_clean",
]

# Seven-or-more '<' or '>' at line start. Git writes exactly seven.
CONFLICT_LINE_RE = re.compile(r"^(?:<{7,}|>{7,})", re.MULTILINE)


class SkillConflictError(RuntimeError):
    """A skill file contains unresolved VCS merge-conflict markers."""


def find_conflict_markers(text: str) -> list[str]:
    """Return the offending lines (stripped, deduped, order-preserving).

    Empty list means clean.
    """
    out: list[str] = []
    seen: set[str] = set()
    for line in text.splitlines():
        if CONFLICT_LINE_RE.match(line):
            s = line.strip()
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


def assert_text_clean(text: str, *, origin: str) -> None:
    """Raise SkillConflictError if `text` carries conflict markers."""
    markers = find_conflict_markers(text)
    if markers:
        raise SkillConflictError(
            f"[SKILL-CONFLICT] {origin} contains unresolved merge-conflict "
            f"markers: {markers}. A skill in this state was never authored by the "
            f"evolver — it is the residue of a failed `git stash pop` (see "
            f"src/registry/manager.py::_git_checkout). Refusing to continue, "
            f"because rolling this out silently corrupts the experiment. Recover "
            f"the intended text from the evolver session's write command "
            f"(.cache/evolver_sessions/*.json) or from a clean snapshot under "
            f".cache/skill_snapshots/, then restart."
        )


def scan_skills_dir(skills_dir: str | Path) -> dict[str, list[str]]:
    """Map {path: marker lines} for every corrupted markdown file under a tree.

    Returns {} when the tree is clean, absent, or empty. Reads every `*.md`
    (not just `SKILL.md`) because harnesses differ in what they glob, and an
    unreadable file is skipped rather than masking a real conflict elsewhere.
    """
    root = Path(skills_dir)
    bad: dict[str, list[str]] = {}
    if not root.is_dir():
        return bad
    for md in sorted(root.rglob("*.md")):
        try:
            text = md.read_text(errors="replace")
        except OSError:
            continue
        markers = find_conflict_markers(text)
        if markers:
            bad[str(md)] = markers
    return bad


def assert_skills_clean(skills_dir: str | Path, *, stage: str) -> None:
    """Raise if any skill under `skills_dir` carries conflict markers.

    `stage` names the pipeline step for the error message, so a failure says
    where corruption entered rather than only that it exists.
    """
    bad = scan_skills_dir(skills_dir)
    if bad:
        detail = "; ".join(f"{p} -> {m}" for p, m in bad.items())
        raise SkillConflictError(
            f"[SKILL-CONFLICT] at {stage}: {len(bad)} skill file(s) under "
            f"{skills_dir} carry unresolved merge-conflict markers ({detail}). "
            f"Refusing to continue — a conflicted skill silently corrupts every "
            f"rollout that loads it. Root cause is normally `git stash pop` "
            f"failing during a program checkout; recover the intended text from "
            f"the evolver session's write command or a clean "
            f".cache/skill_snapshots/ copy, then restart."
        )
