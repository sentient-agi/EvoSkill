"""Codex skill discovery — symlink .agents/skills/ to .claude/skills/.

Codex scans .agents/skills/ for SKILL.md files, but EvoSkill writes
skills to .claude/skills/. This module creates a symlink so both paths
point to the same directory.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_agents_skills_symlink(project_root: Path) -> bool:
    """Create .agents/skills/ symlink pointing to .claude/skills/.

    Idempotent: no-op if correct symlink already exists.
    Never deletes real directories (logs warning, skips).

    Args:
        project_root: Path to the project root where .claude/ and .agents/ live.

    Returns:
        True if a symlink was created or updated, False if no-op.
    """
    source = project_root / ".claude" / "skills"
    link = project_root / ".agents" / "skills"

    # Ensure the source directory exists
    source.mkdir(parents=True, exist_ok=True)

    # Check if link already exists
    if link.is_symlink():
        if link.resolve() == source.resolve():
            return False  # correct symlink already exists
        # Stale symlink pointing elsewhere — replace it
        link.unlink()
    elif link.exists():
        # Real directory — don't destroy user data
        logger.warning(
            "%s exists as a real directory, not creating symlink. "
            "To enable Codex skill discovery, replace it with: "
            "ln -sf %s %s",
            link,
            source.resolve(),
            link,
        )
        return False

    # Create parent directory and symlink
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(source.resolve()), str(link))
    return True
