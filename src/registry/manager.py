"""
Program manager for tracking agent versions via git branches.

Each program is stored as a git branch with:
- .claude/program.yaml: Program configuration (prompts, tools)
- .claude/skills/: Generated skills for this program
"""

import logging
import random
import subprocess
from pathlib import Path
from typing import Any

import yaml

from .models import ProgramConfig

logger = logging.getLogger(__name__)


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
    # Workspace-relative location for per-program skill snapshots. Persists
    # the project's `.claude/skills/` tree at every create_program() call so
    # switch_to() can restore the *exact* skill state that produced a given
    # program's score. Without this, evolved skills get clobbered by later
    # iterations and historical scores become unreproducible.
    SKILL_SNAPSHOT_DIR = ".cache/skill_snapshots"

    def __init__(
        self,
        cwd: str | Path | None = None,
        project_skills_dir: str | Path | None = None,
    ):
        """
        Initialize ProgramManager.

        Args:
            cwd: Working directory for git operations (the workspace).
                Defaults to git repo root.
            project_skills_dir: Absolute path to the project's `.claude/skills/`
                directory — the *real* place evolver agents write skill files.
                When None, defaults to `<cwd>/.claude/skills/` (legacy single-
                root mode where workspace == project_root). When the workspace
                and project root are split (typical with --workspace flag),
                callers MUST pass the project's path so snapshots track the
                live skill tree, not an empty workspace stub.
        """
        if cwd:
            self.cwd = Path(cwd)
        else:
            self.cwd = self._find_repo_root()
        self._project_skills_dir = (
            Path(project_skills_dir).resolve()
            if project_skills_dir is not None
            else self.cwd / ".claude" / "skills"
        )
        # Ensure the workspace's .gitignore covers loop-managed cache dirs.
        # Without this, our defensive `_git_checkout` (which calls
        # `git clean -fd` as a fallback when checkout fails on untracked
        # files) would wipe `.cache/skill_snapshots/`, destroying the
        # reproducibility snapshots created at every program. The seed
        # entries here are minimal — append-safe for users who later
        # add their own ignores.
        self._ensure_workspace_gitignore()

    def _ensure_workspace_gitignore(self) -> None:
        """Idempotently add `.cache/` to the workspace's .gitignore.

        Safe to call repeatedly — appends only entries that aren't already
        present. Only operates inside a real git repo (cwd has .git/).
        """
        if not (self.cwd / ".git").exists():
            return
        gitignore = self.cwd / ".gitignore"
        required = [".cache/", "logs/"]
        existing_lines = (
            gitignore.read_text().splitlines() if gitignore.exists() else []
        )
        existing_set = {line.strip() for line in existing_lines}
        to_add = [entry for entry in required if entry not in existing_set]
        if not to_add:
            return
        with open(gitignore, "a") as f:
            if existing_lines and existing_lines[-1].strip():
                f.write("\n")  # ensure newline before appending
            f.write("# evoskill: loop-managed state — do not commit, do not clean\n")
            for entry in to_add:
                f.write(f"{entry}\n")

    @staticmethod
    def _find_repo_root() -> Path:
        """Find the git repository root by looking for .git directory."""
        current = Path.cwd()
        for parent in [current, *current.parents]:
            if (parent / ".git").exists():
                return parent
        # Fallback to cwd if no .git found
        return current

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

        # Snapshot the live project skills tree so this program can be exactly
        # reproduced later, even after subsequent iterations mutate the same
        # files. Snapshot AFTER the commit so it captures the post-evolver
        # state when create_program is called from inside _mutate.
        self._snapshot_skills(name)

        return branch_name

    def switch_to(self, name: str) -> None:
        """
        Switch to a program and restore its skill state.

        First does git checkout (which restores program.yaml + anything
        committed under .claude/skills/ in the workspace). Then restores the
        project-level skills tree from the snapshot taken at create time —
        this is the actually-load-bearing step, since evolver agents write
        skills outside the workspace and git can't track them there.

        Args:
            name: Program name (without 'program/' prefix)
        """
        self._git_checkout(f"{self.BRANCH_PREFIX}{name}")
        self._restore_skills(name)

    # ---------------------------------------------------------------------
    # Skill snapshot helpers
    # ---------------------------------------------------------------------

    def _snapshot_dir_for(self, name: str) -> Path:
        return self.cwd / self.SKILL_SNAPSHOT_DIR / name

    def _snapshot_skills(self, name: str) -> None:
        """Capture the live project skills tree under workspace snapshot store.

        Always creates the snapshot directory, even when the project skills
        tree is empty or missing. An empty snapshot is the *correct* record
        for a program that has no skills (e.g., the base program created
        before any evolver mutation), and is what `_restore_skills` needs
        to wipe leftover skills from a discarded child mutation when the
        loop switches back to the parent.
        """
        import shutil
        src = self._project_skills_dir
        dest = self._snapshot_dir_for(name)
        # Wipe any prior snapshot for this program (e.g., from an aborted run)
        # so the fresh copy is authoritative.
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True, exist_ok=True)
        if src.exists():
            # copytree requires an empty target on macOS without dirs_exist_ok.
            shutil.copytree(src, dest, dirs_exist_ok=True)
        # If src doesn't exist, dest is left as an empty dir — the marker
        # that "this program has no skills" — which restore will use.

    def _restore_skills(self, name: str) -> None:
        """Restore the project skills tree from the snapshot taken at create.

        Replaces the live tree wholesale to match the snapshot. When the
        snapshot is empty (e.g., the base program), the live tree is wiped
        — this is load-bearing: without it, a discarded child mutation's
        skill files would persist on disk and bleed into subsequent
        iterations.

        No-op only if no snapshot exists at all (e.g., a program created by
        an older version that didn't snapshot). That case is for backward
        compat — modern programs always have at least an empty snapshot.
        """
        import shutil
        snap = self._snapshot_dir_for(name)
        if not snap.exists():
            return
        dest = self._project_skills_dir
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True, exist_ok=True)
        # When snap is empty, this copies nothing — leaving dest empty,
        # which is exactly what we want.
        shutil.copytree(snap, dest, dirs_exist_ok=True)

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

    def reset_all(self) -> dict[str, int]:
        """Delete all program branches, frontier tags, and loop state files.

        Returns:
            Dict with counts: {"branches": N, "tags": N, "files": N}
        """
        # Detect a safe branch to land on (first non-program local branch)
        all_branches = self._git_list_branches()
        safe_branch = next(
            (b for b in all_branches if not b.startswith(self.BRANCH_PREFIX)),
            "main",
        )
        self._git_checkout(safe_branch)

        # Delete all frontier/* tags
        tags_deleted = 0
        for tag in self._git_list_tags():
            if tag.startswith(self.FRONTIER_PREFIX):
                self._git_tag_delete(tag)
                tags_deleted += 1

        # Delete all program/* branches
        branches_deleted = 0
        for branch in self._git_list_branches():
            if branch.startswith(self.BRANCH_PREFIX):
                self._git_branch_delete(branch)
                branches_deleted += 1

        # Delete loop state files
        files_deleted = 0
        for rel_path in [
            ".claude/loop_checkpoint.json",
            ".claude/feedback_history.md",
        ]:
            p = self.cwd / rel_path
            if p.exists():
                p.unlink()
                files_deleted += 1

        return {"branches": branches_deleted, "tags": tags_deleted, "files": files_deleted}

    def discard(self, name: str) -> None:
        """
        Delete a program branch.

        Args:
            name: Program name to delete
        """
        branch = f"{self.BRANCH_PREFIX}{name}"
        # Switch away if currently on this branch
        if self._git_current_branch() == branch:
            checkout_target = self._get_discard_checkout_target(branch)
            self._git_checkout(checkout_target)
        self._git_branch_delete(branch)

        # Also remove frontier tag if exists
        tag = f"{self.FRONTIER_PREFIX}{name}"
        if tag in self._git_list_tags():
            self._git_tag_delete(tag)

    def _get_discard_checkout_target(self, branch: str) -> str:
        """Choose a safe checkout target before deleting the current branch."""
        branches = self._git_list_branches()

        non_program_branches = [
            candidate
            for candidate in branches
            if candidate != branch and not candidate.startswith(self.BRANCH_PREFIX)
        ]
        if non_program_branches:
            return non_program_branches[0]

        try:
            config = self._read_config_from_branch(branch)
            if config.parent and config.parent != branch and config.parent in branches:
                return config.parent
        except Exception:
            pass

        sibling_programs = [candidate for candidate in branches if candidate != branch]
        if sibling_programs:
            return sibling_programs[0]

        raise RuntimeError(f"Cannot discard the only remaining branch: {branch}")

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

    def get_frontier_with_scores(self) -> list[tuple[str, float]]:
        """
        Get frontier programs with their scores, sorted by score descending.

        Returns:
            List of (program_name, score) tuples, highest score first.
            Programs without scores are excluded.
        """
        frontier = self.get_frontier()
        scored: list[tuple[str, float, int]] = []
        for index, name in enumerate(frontier):
            try:
                config = self._read_config_from_branch(f"{self.BRANCH_PREFIX}{name}")
                score = config.get_score()
                if score is not None:
                    scored.append((name, score, index))
            except Exception:
                continue
        scored.sort(key=lambda item: (item[1], item[2]), reverse=True)
        return [(name, score) for name, score, _ in scored]

    def select_from_frontier(self, strategy: str, iteration: int = 0) -> str | None:
        """Select a program from the frontier using the given strategy.

        Args:
            strategy: Selection strategy — "best", "random", or "round_robin".
            iteration: Current iteration number (used by round_robin).

        Returns:
            Program name, or None if frontier is empty.
        """
        scored = self.get_frontier_with_scores()
        if not scored:
            return None
        if strategy == "random":
            return random.choice(scored)[0]
        if strategy == "round_robin":
            return scored[iteration % len(scored)][0]
        # Default: "best"
        return scored[0][0]

    def get_best_from_frontier(self) -> str | None:
        """
        Get the program with highest score from the frontier.

        Returns:
            Program name with highest score, or None if frontier is empty.
        """
        scored = self.get_frontier_with_scores()
        if scored:
            return scored[0][0]
        return None

    def update_frontier(
        self, name: str, score: float, max_size: int = 5,
        *, cost: float | None = None,
        parent_score: float | None = None,
        parent_cost: float | None = None,
    ) -> bool:
        """
        Add program to frontier if it qualifies, pruning worst if over max_size.

        A program qualifies if (in priority order):
        1. (Regression guard, if parent_score is provided) score is NOT
           materially worse than parent's score. In Phase 2 (efficiency)
           a regression is rejected outright — the score MUST be at least
           as good as the parent's.
        2. Frontier has fewer than max_size members, OR
        3. Score is strictly higher than the lowest score in frontier, OR
        4. Score ties the worst AND cost is lower (efficiency tie-break).

        Args:
            name: Program name to potentially add
            score: Score for this program
            max_size: Maximum frontier size
            cost: Average evaluation cost in USD for this program. When
                  present, used to break ties on equal score — the lower
                  cost wins. Stored on the program's config metadata for
                  future comparisons.
            parent_score: Optional parent program's score. When provided,
                  any mutation with score < parent_score is REJECTED before
                  being considered for frontier admission. This prevents
                  Phase 2 efficiency mutations from polluting the frontier
                  with accuracy regressions.
            parent_cost: Optional parent program's cost. When score ties
                  parent_score AND cost is provided, requires the new cost
                  to be strictly lower than parent_cost (otherwise the
                  mutation isn't actually an improvement).

        Returns:
            True if program was added to frontier, False otherwise
        """
        # First, update the program's config with the score (and cost, if provided)
        current_branch = self._git_current_branch()
        target_branch = f"{self.BRANCH_PREFIX}{name}"

        # Switch to target branch to update config
        if current_branch != target_branch:
            self._git_checkout(target_branch)

        config = self._read_config()
        updated_config = config.with_score(score)
        if cost is not None:
            updated_config = updated_config.with_metadata(cost=float(cost))
        self._write_config(updated_config)
        self._git_add(self.PROGRAM_FILE)
        commit_msg = f"Update score: {score:.4f}"
        if cost is not None:
            commit_msg += f" cost: ${cost:.4f}"
        self._git_commit(commit_msg)

        # Switch back
        if current_branch != target_branch:
            self._git_checkout(current_branch)

        # Regression guard (rule 1). Apply BEFORE checking frontier room —
        # a regression should never be admitted, even if there's space.
        # Tolerance is exact (no floating-point slack) because the cosine
        # scorer already smooths small differences.
        if parent_score is not None and score < parent_score:
            # Special case: in Phase 2, a tie on score with strictly lower
            # cost is still acceptable (cost-improvement is the whole point).
            cost_improves = (
                parent_cost is not None
                and cost is not None
                and cost < parent_cost
                and score == parent_score
            )
            if not cost_improves:
                return False

        # Now check frontier membership
        scored = self.get_frontier_with_scores()

        # If frontier has room, add unconditionally (subject to the
        # regression guard above which already filtered out bad mutations).
        if len(scored) < max_size:
            self.mark_frontier(name)
            return True

        # Otherwise, compare against the worst member (last in sorted list)
        worst_name, worst_score = scored[-1]
        if score > worst_score:
            self.unmark_frontier(worst_name)
            self.mark_frontier(name)
            return True

        # Tie-break on cost when scores are equal and cost is available on
        # both sides. Lower cost wins — this is what lets phase-2 efficiency
        # evolution actually replace incumbents when accuracy is already
        # saturated.
        if cost is not None and score == worst_score:
            worst_cost = self._get_program_cost(worst_name)
            if worst_cost is not None and cost < worst_cost:
                self.unmark_frontier(worst_name)
                self.mark_frontier(name)
                return True

        return False

    def _get_program_cost(self, name: str) -> float | None:
        """Read the `cost` metadata from a program's config without switching branches."""
        target_branch = f"{self.BRANCH_PREFIX}{name}"
        try:
            import subprocess
            out = subprocess.check_output(
                ["git", "show", f"{target_branch}:{self.PROGRAM_FILE}"],
                cwd=self.cwd, text=True, stderr=subprocess.DEVNULL,
            )
            import yaml
            data = yaml.safe_load(out) or {}
            meta = (data.get("metadata") or {}) if isinstance(data, dict) else {}
            cost = meta.get("cost")
            return float(cost) if cost is not None else None
        except Exception:
            return None

    def commit(self, message: str | None = None) -> bool:
        """
        Commit any changes in the repo.

        Only commits if there are actual changes. Safe to call anytime.

        Args:
            message: Commit message (defaults to 'Update program: {name}')

        Returns:
            True if a commit was made, False if nothing to commit
        """
        # Check if there are any changes
        result = self._run_git(["status", "--porcelain"], check=False)
        if not result.stdout.strip():
            return False  # Nothing to commit

        # Stage all changes
        self._git_add(".")

        # Get program name for default message
        try:
            config = self._read_config()
            default_msg = f"Update program: {config.name}"
        except Exception:
            default_msg = "Update program"

        self._git_commit(message or default_msg)
        return True

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
        """Checkout a branch, auto-stashing any uncommitted changes.

        Robust against leftover merge conflicts: if the working tree is in
        an unresolved state (e.g. from a prior failed `stash pop`), git
        refuses to stash again. We detect that and clear only the
        regenerable state files (`.claude/feedback_history.md`,
        `.claude/loop_checkpoint.json`) — these get rewritten on the next
        iteration anyway. Evolver-produced skill edits are preserved.

        Known race (mitigated, not eliminated): `feedback_history.md` is
        committed on each branch, so when we stash → checkout target →
        pop, git's 3-way merge can leave conflict markers in the file
        when both branches diverged. We do NOT resolve here because
        we don't know the merge intent. Instead, `read_feedback_history`
        in src/loop/helpers.py strips markers on read and warns loudly.
        Long-term fix: move feedback_history.md out of git-tracked space
        (e.g., into .cache/) so it doesn't participate in checkout merges.
        """
        # 1. Clear any unresolved merge state on regenerable files.
        #    Porcelain codes for conflicts: DD, AU, UD, UA, DU, AA, UU.
        result = self._run_git(["status", "--porcelain"], check=False)
        conflicted_paths = [
            line[3:].strip()
            for line in result.stdout.splitlines()
            if line[:2] in ("DD", "AU", "UD", "UA", "DU", "AA", "UU")
        ]
        if conflicted_paths:
            regenerable = {
                ".claude/feedback_history.md",
                ".claude/loop_checkpoint.json",
            }
            for p in conflicted_paths:
                if p in regenerable:
                    # Resolve by removing the file — it'll be rewritten next iter.
                    self._run_git(["rm", "-f", p], check=False)
                else:
                    # Keep non-regenerable conflicts visible; surface loudly.
                    logger.warning(
                        f"Unresolved merge conflict on {p} — manual resolution required"
                    )

        # 2. Re-check for uncommitted changes after conflict cleanup.
        result = self._run_git(["status", "--porcelain"], check=False)
        has_changes = bool(result.stdout.strip())

        # 3. Stash if dirty (include untracked with -u so no files are lost)
        if has_changes:
            self._run_git(["stash", "push", "-u", "-m", "auto-stash before checkout"], check=False)

        # 4. Perform checkout
        self._run_git(["checkout", branch])

        # 5. Pop stash if we stashed (best-effort — conflicts leave stash for manual review)
        if has_changes:
            self._run_git(["stash", "pop"], check=False)

    def _git_checkout_new(self, branch: str) -> None:
        """Create and checkout a new branch.

        If the branch already exists (e.g., a prior mutation in this iteration
        was created and then GATE-rejected, leaving a stale branch behind),
        force-recreate it from the current HEAD. Without this, a retry of the
        same iteration crashes with `git checkout -b` exit 128.
        """
        existing = set(self._git_list_branches())
        if branch in existing:
            current = self._git_current_branch()
            if current == branch:
                # Already on the stale branch (shouldn't normally happen, but
                # be defensive). Step off it before deleting.
                self._run_git(["checkout", "main"], check=False)
            self._git_branch_delete(branch)
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
        """Create a commit if there are staged changes."""
        # Check if there are staged changes
        result = self._run_git(["diff", "--cached", "--quiet"], check=False)
        if result.returncode != 0:
            # There are staged changes, commit them
            commit_result = self._run_git(["commit", "-m", message], check=False)
            if commit_result.returncode != 0:
                # Log error but don't crash - common issues: no user config, lock file, etc.
                import logging
                logging.warning(
                    f"Git commit failed (exit {commit_result.returncode}): {commit_result.stderr.strip()}"
                )

    def _git_tag(self, tag: str) -> None:
        """Create a tag at current HEAD."""
        # Check if tag already exists
        existing_tags = self._git_list_tags()
        if tag in existing_tags:
            # Tag already exists, force update to current HEAD
            self._run_git(["tag", "-f", tag])
        else:
            # Create new tag
            self._run_git(["tag", tag])

    def _git_tag_delete(self, tag: str) -> None:
        """Delete a tag."""
        self._run_git(["tag", "-d", tag])

    def _git_list_tags(self) -> list[str]:
        """List all tags."""
        result = self._run_git(["tag", "-l"])
        return [t.strip() for t in result.stdout.strip().split("\n") if t.strip()]
