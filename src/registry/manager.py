"""
Program manager for tracking agent versions via git branches.

Each program is stored as a git branch with:
- .claude/program.yaml: Program configuration (prompts, tools)
- .claude/skills/: Generated skills for this program
"""

import subprocess
from pathlib import Path
from typing import Any

import yaml

from .models import ProgramConfig


class ProgramManager:
    """
    Manages program branches via git.

    Programs are stored as git branches with the prefix 'program/'.
    Switching between programs is done via git checkout.
    Frontier members are tracked via git tags with prefix 'frontier/'.
    """

    PROGRAM_FILE = ".claude/program.yaml"
    BRANCH_PREFIX = "program/"
    FRONTIER_PREFIX = "frontier/"

    def __init__(self, cwd: str | Path | None = None):
        """
        Initialize ProgramManager.

        Args:
            cwd: Working directory for git operations. Defaults to current directory.
        """
        self.cwd = Path(cwd) if cwd else Path.cwd()

    def create_program(
        self,
        name: str,
        config: ProgramConfig,
        parent: str | None = None,
    ) -> str:
        """
        Create a new program branch from parent.

        Args:
            name: Program name (will be prefixed with 'program/')
            config: Program configuration to save
            parent: Parent program name (without prefix) to branch from

        Returns:
            Full branch name (e.g., 'program/base')
        """
        branch_name = f"{self.BRANCH_PREFIX}{name}"

        # If parent specified, checkout parent first
        if parent:
            self._git_checkout(f"{self.BRANCH_PREFIX}{parent}")

        # Create and checkout new branch
        self._git_checkout_new(branch_name)

        # Write program config
        self._write_config(config)

        # Stage and commit
        self._git_add(self.PROGRAM_FILE)
        # Also stage any skills that might exist
        skills_dir = self.cwd / ".claude" / "skills"
        if skills_dir.exists():
            self._git_add(".claude/skills/")
        self._git_commit(f"Create program: {name}")

        return branch_name

    def switch_to(self, name: str) -> None:
        """
        Switch to a program (git checkout).

        Args:
            name: Program name (without 'program/' prefix)
        """
        self._git_checkout(f"{self.BRANCH_PREFIX}{name}")

    def get_current(self) -> ProgramConfig:
        """
        Get current program config from disk.

        Returns:
            ProgramConfig loaded from .claude/program.yaml
        """
        return self._read_config()

    def get_current_name(self) -> str:
        """
        Get current program name.

        Returns:
            Program name (without 'program/' prefix), or branch name if not a program
        """
        branch = self._git_current_branch()
        if branch.startswith(self.BRANCH_PREFIX):
            return branch[len(self.BRANCH_PREFIX) :]
        return branch

    def list_programs(self) -> list[str]:
        """
        List all program branches.

        Returns:
            List of program names (without 'program/' prefix)
        """
        branches = self._git_list_branches()
        return [
            b[len(self.BRANCH_PREFIX) :]
            for b in branches
            if b.startswith(self.BRANCH_PREFIX)
        ]

    def get_lineage(self, name: str) -> list[str]:
        """
        Get parent chain by reading program.yaml from each program.

        Args:
            name: Program name to get lineage for

        Returns:
            List of program names from child to root (e.g., ['iter-1', 'base'])
        """
        lineage = [name]
        current = name
        while True:
            config = self._read_config_from_branch(f"{self.BRANCH_PREFIX}{current}")
            if config.parent is None:
                break
            parent = config.parent.replace(self.BRANCH_PREFIX, "")
            lineage.append(parent)
            current = parent
        return lineage

    def get_children(self, name: str) -> list[str]:
        """
        Get programs that have this program as parent.

        Args:
            name: Program name to find children of

        Returns:
            List of child program names
        """
        parent_ref = f"{self.BRANCH_PREFIX}{name}"
        children = []
        for program in self.list_programs():
            if program == name:
                continue
            config = self._read_config_from_branch(f"{self.BRANCH_PREFIX}{program}")
            if config.parent == parent_ref:
                children.append(program)
        return children

    def discard(self, name: str) -> None:
        """
        Delete a program branch.

        Args:
            name: Program name to delete
        """
        branch = f"{self.BRANCH_PREFIX}{name}"
        # Switch away if currently on this branch
        if self._git_current_branch() == branch:
            self._git_checkout("main")
        self._git_branch_delete(branch)

        # Also remove frontier tag if exists
        tag = f"{self.FRONTIER_PREFIX}{name}"
        if tag in self._git_list_tags():
            self._git_tag_delete(tag)

    def mark_frontier(self, name: str) -> None:
        """
        Tag a program as part of the frontier.

        Args:
            name: Program name to mark as frontier
        """
        # Make sure we're on the right branch
        current = self._git_current_branch()
        target = f"{self.BRANCH_PREFIX}{name}"
        if current != target:
            self._git_checkout(target)

        self._git_tag(f"{self.FRONTIER_PREFIX}{name}")

        # Switch back if needed
        if current != target:
            self._git_checkout(current)

    def unmark_frontier(self, name: str) -> None:
        """
        Remove a program from the frontier.

        Args:
            name: Program name to remove from frontier
        """
        tag = f"{self.FRONTIER_PREFIX}{name}"
        if tag in self._git_list_tags():
            self._git_tag_delete(tag)

    def get_frontier(self) -> list[str]:
        """
        Get all frontier-tagged programs.

        Returns:
            List of program names in the frontier
        """
        tags = self._git_list_tags()
        return [
            t[len(self.FRONTIER_PREFIX) :]
            for t in tags
            if t.startswith(self.FRONTIER_PREFIX)
        ]

    def update_config(self, config: ProgramConfig, message: str | None = None) -> None:
        """
        Update the current program's configuration.

        Args:
            config: New configuration to save
            message: Commit message (defaults to 'Update program: {name}')
        """
        self._write_config(config)
        self._git_add(self.PROGRAM_FILE)
        self._git_commit(message or f"Update program: {config.name}")

    def save_skill_changes(self, message: str | None = None) -> None:
        """
        Commit any changes to skills directory.

        Args:
            message: Commit message (defaults to 'Update skills')
        """
        skills_dir = self.cwd / ".claude" / "skills"
        if skills_dir.exists():
            self._git_add(".claude/skills/")
            self._git_commit(message or "Update skills")

    # -------------------------------------------------------------------------
    # Internal: Config I/O
    # -------------------------------------------------------------------------

    def _write_config(self, config: ProgramConfig) -> None:
        """Write program config to YAML file."""
        config_path = self.cwd / self.PROGRAM_FILE
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)

    def _read_config(self) -> ProgramConfig:
        """Read program config from YAML file."""
        config_path = self.cwd / self.PROGRAM_FILE
        with open(config_path) as f:
            data = yaml.safe_load(f)
        return ProgramConfig.model_validate(data)

    def _read_config_from_branch(self, branch: str) -> ProgramConfig:
        """Read program config from a specific branch without checking out."""
        result = self._run_git(["show", f"{branch}:{self.PROGRAM_FILE}"])
        data = yaml.safe_load(result.stdout)
        return ProgramConfig.model_validate(data)

    # -------------------------------------------------------------------------
    # Internal: Git operations
    # -------------------------------------------------------------------------

    def _run_git(
        self, args: list[str], check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        """Run a git command."""
        return subprocess.run(
            ["git"] + args,
            cwd=self.cwd,
            capture_output=True,
            text=True,
            check=check,
        )

    def _git_checkout(self, branch: str) -> None:
        """Checkout a branch."""
        self._run_git(["checkout", branch])

    def _git_checkout_new(self, branch: str) -> None:
        """Create and checkout a new branch."""
        self._run_git(["checkout", "-b", branch])

    def _git_current_branch(self) -> str:
        """Get current branch name."""
        result = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        return result.stdout.strip()

    def _git_list_branches(self) -> list[str]:
        """List all local branches."""
        result = self._run_git(["branch", "--format=%(refname:short)"])
        return [b.strip() for b in result.stdout.strip().split("\n") if b.strip()]

    def _git_branch_delete(self, branch: str) -> None:
        """Delete a branch."""
        self._run_git(["branch", "-D", branch])

    def _git_add(self, path: str) -> None:
        """Stage a file or directory."""
        self._run_git(["add", path])

    def _git_commit(self, message: str) -> None:
        """Create a commit."""
        self._run_git(["commit", "-m", message])

    def _git_tag(self, tag: str) -> None:
        """Create a tag at current HEAD."""
        self._run_git(["tag", tag])

    def _git_tag_delete(self, tag: str) -> None:
        """Delete a tag."""
        self._run_git(["tag", "-d", tag])

    def _git_list_tags(self) -> list[str]:
        """List all tags."""
        result = self._run_git(["tag", "-l"])
        return [t.strip() for t in result.stdout.strip().split("\n") if t.strip()]
