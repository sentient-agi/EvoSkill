"""
Program-aware run caching for agent evaluations.

Cache invalidates automatically when behavior-affecting files change:
- .claude/skills/** (skill definitions)
- src/agent_profiles/base_agent/prompt.txt (legacy fallback) and
  src/agent_profiles/officeqa_agent/prompt.md (prompt text)

Excludes metadata files like .claude/program.yaml to avoid unnecessary
cache invalidation when only scores or timestamps change.
"""

import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from src.harness import AgentTrace

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class CacheConfig(BaseModel):
    """Configuration for RunCache behavior."""

    cache_dir: Path = Path(".cache/runs")
    enabled: bool = True
    store_messages: bool = False  # Whether to cache the full messages list
    hash_length: int = 12  # Length of hash prefix for filenames
    cwd: Path = Path(".")  # Working directory for git commands
    # Path to the LIVE `.claude/skills/` the agent reads from at run time.
    # When workspace ≠ project source root (typical with `--workspace`),
    # the workspace's `.claude/skills/` only contains stub snapshot files
    # — the real skills evolve in the project's `.claude/skills/`. Hashing
    # the workspace dir alone produces a stable cache key even as the
    # actual loaded skills change between iterations, returning stale
    # responses generated under different skill conditions. Pointing here
    # at the live dir fixes that. Defaults to `<cwd>/.claude/skills` for
    # backward compatibility (single-root setups where workspace == project).
    live_skills_dir: Path | None = None
    # Path to the project source root (parent of `src/agent_profiles/...`)
    # so the cache hashes the agent's actual prompt files. Without this,
    # `_get_tree_hash` looks for prompts at `<cwd>/src/agent_profiles/...`
    # which DOES NOT EXIST in a workspace setup → prompt edits go undetected.
    # Defaults to `cwd` for single-root backward compatibility.
    project_source_root: Path | None = None

    class Config:
        arbitrary_types_allowed = True


class CacheEntry(BaseModel):
    """Schema for a cached run entry."""

    version: str = "1.0"
    created_at: str

    cache_key: dict[str, str]  # tree_hash, question_hash, question
    trace: dict[str, Any]  # Serialized AgentTrace


class RunCache:
    """
    Program-aware cache for agent run results.

    Keys on hash(behavior-affecting files) + hash(question) to automatically
    invalidate when skills or prompts change. Ignores metadata changes
    (scores, timestamps) to avoid unnecessary cache invalidation.

    Usage:
        cache = RunCache()

        # Check cache (sdk + model are required keyword args — they're part of the key)
        trace = cache.get(question, sdk="claude", model="claude-opus-4-6")
        if trace is None:
            trace = await agent.run(question)
            cache.set(question, trace, sdk="claude", model="claude-opus-4-6")
    """

    def __init__(self, config: CacheConfig | None = None):
        """
        Initialize RunCache.

        Args:
            config: Cache configuration. Uses defaults if not provided.
        """
        self.config = config or CacheConfig()
        self._ensure_cache_dir()

    def _ensure_cache_dir(self) -> None:
        """Create cache directory if it doesn't exist."""
        if self.config.enabled:
            self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_tree_hash(self) -> str:
        """
        Get combined hash of files that affect agent behavior.

        Primary source of truth: the workspace's current git HEAD commit +
        the tree-hash of `.claude/skills/` on that commit. Since every
        program is a distinct branch with its own committed skill state,
        git already knows when things actually changed — using git's hashes
        sidesteps pathlib glob edge cases, stat-cache freshness, and
        uncommitted-working-tree drift.

        Falls back to the prior content-hashing logic only when git queries
        fail (e.g. workspace isn't a git repo).

        Returns:
            Combined hash of behavior-affecting state.
        """
        # Three ingredients go into the hash:
        #   (a) git HEAD — captures branch switches (each program = new branch).
        #   (b) live file content of .claude/skills/ — captures uncommitted
        #       working-tree edits (evolver writes before branch is created).
        #   (c) prompt.txt content — captures prompt edits.
        # Mixing (a) and (b) means we detect BOTH "switched to a different
        # program branch" AND "wrote uncommitted skill changes" — either
        # forces a cache miss.
        git_token = self._git_behavior_token() or "no-git"

        # Hash the LIVE skills directory (where the agent actually reads
        # skill files from). When the loop runs with --workspace, the
        # live dir is on the project side (e.g. <EvoSkill>/.claude/skills),
        # not the workspace side (which carries only stub snapshots). If
        # we hash the workspace dir, the cache key stays constant across
        # iterations even as skills change, masking the evolver's effect.
        skills_dir = self.config.live_skills_dir or (
            self.config.cwd / ".claude" / "skills"
        )
        skills_h = self._hash_files(skills_dir, "**/*") if skills_dir.exists() else ""

        # Hash whichever agent prompts exist. Use the project source root
        # (where the actual `src/agent_profiles/...` tree lives), not cwd
        # — same workspace/project split issue as skills_dir above.
        source_root = self.config.project_source_root or self.config.cwd
        prompt_paths = [
            source_root / "src" / "agent_profiles" / "base_agent" / "prompt.txt",
            source_root / "src" / "agent_profiles" / "officeqa_agent" / "prompt.md",
        ]
        hasher = hashlib.sha256()
        for prompt_path in prompt_paths:
            try:
                if prompt_path.is_file():
                    with open(prompt_path, "rb") as f:
                        hasher.update(prompt_path.name.encode("utf-8"))
                        hasher.update(f.read())
            except (IOError, OSError):
                pass
        prompt_h = hasher.hexdigest()

        combined = f"git:{git_token}|skills:{skills_h}|prompt:{prompt_h}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def _git_behavior_token(self) -> str | None:
        """Return a git-derived token for the workspace's current skill state.

        Uses the **root tree SHA** (`git rev-parse HEAD^{tree}`) plus the
        skills tree SHA. Tree SHAs are content-addressed: identical content
        produces the same SHA regardless of when the commit was made. Using
        the commit SHA (HEAD) instead would break cache reuse across
        `--fresh` runs because each run creates a new "Create program: base"
        commit with a fresh timestamp → different commit SHA, identical
        content. Tree SHAs sidestep this by hashing the snapshot, not the
        commit metadata.

        Returns None when git isn't available, the cwd isn't a repo, or
        `.claude/skills` isn't tracked.
        """
        try:
            import subprocess
            root_tree = subprocess.check_output(
                ["git", "rev-parse", "HEAD^{tree}"],
                cwd=self.config.cwd, text=True, stderr=subprocess.DEVNULL,
            ).strip()
            try:
                skills_tree = subprocess.check_output(
                    ["git", "rev-parse", "HEAD:.claude/skills"],
                    cwd=self.config.cwd, text=True, stderr=subprocess.DEVNULL,
                ).strip()
            except subprocess.CalledProcessError:
                # Skills dir not tracked on this commit → use empty-tree SHA
                skills_tree = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
            return f"{root_tree}:{skills_tree}"
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return None

    def _hash_files(self, dir_path: Path, pattern: str = "**/*") -> str:
        """
        Compute content hash of files matching a pattern in a directory.

        Args:
            dir_path: Path to directory to search in
            pattern: Glob pattern for files to include (default: all files)

        Returns:
            SHA256 hash of matching file contents
        """
        hasher = hashlib.sha256()

        # Get matching files, sorted for consistency
        files = sorted(dir_path.glob(pattern))
        for file_path in files:
            if file_path.is_file():
                # Include relative path and content in hash
                rel_path = file_path.relative_to(dir_path)
                hasher.update(str(rel_path).encode("utf-8"))
                try:
                    with open(file_path, "rb") as f:
                        hasher.update(f.read())
                except (IOError, OSError):
                    # Skip files that can't be read
                    pass

        return hasher.hexdigest()

    @staticmethod
    def _compute_question_hash(question: str) -> str:
        """Compute hash for a question string."""
        return hashlib.sha256(question.strip().encode("utf-8")).hexdigest()

    def _get_cache_dir_for_tree(self, tree_hash: str) -> Path:
        """Get the cache directory for a specific tree hash."""
        return self.config.cache_dir / tree_hash[: self.config.hash_length]

    def _get_cache_path(self, tree_hash: str, question: str, sdk: str, model: str) -> Path:
        """Get the cache file path for a tree hash + question + sdk + model."""
        tree_dir = self._get_cache_dir_for_tree(tree_hash)
        keyed = f"{question.strip()}|sdk={sdk}|model={model}"
        question_hash = hashlib.sha256(keyed.encode("utf-8")).hexdigest()[: self.config.hash_length]
        return tree_dir / f"{question_hash}.json"

    def get(
        self,
        question: str,
        response_model: type[T] | None = None,
        *,
        sdk: str,
        model: str,
    ) -> AgentTrace[T] | None:
        """
        Retrieve cached trace for current .claude/ content + question + sdk + model.

        Args:
            question: The question/query string
            response_model: Optional Pydantic model for output validation
            sdk: Active harness SDK (e.g. "claude", "opencode") — part of cache key
            model: Model identifier — part of cache key

        Returns:
            AgentTrace if cache hit, None if miss or disabled
        """
        if not self.config.enabled:
            return None

        try:
            tree_hash = self._get_tree_hash()
        except RuntimeError as e:
            logger.debug(f"Cache get failed to compute tree hash: {e}")
            return None

        cache_path = self._get_cache_path(tree_hash, question, sdk, model)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, "r") as f:
                entry_data = json.load(f)

            entry = CacheEntry.model_validate(entry_data)

            # Reconstruct AgentTrace from cached data
            trace_data = entry.trace

            # Reconstruct output if response_model provided
            if response_model and trace_data.get("output"):
                trace_data["output"] = response_model.model_validate(trace_data["output"])

            return AgentTrace.model_validate(trace_data)

        except (json.JSONDecodeError, ValueError, KeyError):
            # Corrupted cache entry - delete and return None
            cache_path.unlink(missing_ok=True)
            return None

    def set(
        self,
        question: str,
        trace: AgentTrace[Any],
        *,
        sdk: str,
        model: str,
    ) -> None:
        """
        Cache a trace for current .claude/ content + question + sdk + model.

        Args:
            question: The question/query string
            trace: The AgentTrace to cache
            sdk: Active harness SDK (e.g. "claude", "opencode") — part of cache key
            model: Model identifier — part of cache key
        """
        if not self.config.enabled:
            return

        try:
            tree_hash = self._get_tree_hash()
        except RuntimeError as e:
            logger.debug(f"Cache set failed to compute tree hash: {e}")
            return

        tree_dir = self._get_cache_dir_for_tree(tree_hash)
        tree_dir.mkdir(parents=True, exist_ok=True)

        cache_path = self._get_cache_path(tree_hash, question, sdk, model)

        # Serialize trace
        trace_dict = trace.model_dump()

        # Optionally strip messages to save space
        if not self.config.store_messages:
            trace_dict["messages"] = []

        # Serialize output if it's a Pydantic model
        if trace.output is not None:
            trace_dict["output"] = trace.output.model_dump()

        entry = CacheEntry(
            version="1.0",
            created_at=datetime.now().isoformat(),
            cache_key={
                "tree_hash": tree_hash[: self.config.hash_length],
                "question_hash": hashlib.sha256(
                    f"{question.strip()}|sdk={sdk}|model={model}".encode("utf-8")
                ).hexdigest()[: self.config.hash_length],
                "question": question,
                "sdk": sdk,
                "model": model,
            },
            trace=trace_dict,
        )

        # Write atomically (write to temp, then rename)
        temp_path = cache_path.with_suffix(".tmp")
        with open(temp_path, "w") as f:
            json.dump(entry.model_dump(), f, indent=2, default=str)
        temp_path.rename(cache_path)

    def clear(self, tree_hash: str | None = None) -> int:
        """
        Clear cached entries.

        Args:
            tree_hash: If provided, only clear entries for this tree hash.
                      If None, clear all cached entries.

        Returns:
            Number of entries deleted
        """
        if not self.config.enabled:
            return 0

        count = 0

        if tree_hash is not None:
            # Clear specific tree's cache
            tree_dir = self._get_cache_dir_for_tree(tree_hash)
            if tree_dir.exists():
                count = len(list(tree_dir.glob("*.json")))
                shutil.rmtree(tree_dir)
        else:
            # Clear all caches
            if self.config.cache_dir.exists():
                for tree_dir in self.config.cache_dir.iterdir():
                    if tree_dir.is_dir():
                        count += len(list(tree_dir.glob("*.json")))
                shutil.rmtree(self.config.cache_dir)
                self._ensure_cache_dir()

        return count

    def stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats (total entries, size, programs)
        """
        if not self.config.cache_dir.exists():
            return {"total_entries": 0, "total_size_bytes": 0, "programs": 0}

        total_entries = 0
        total_size = 0
        programs = 0

        for tree_dir in self.config.cache_dir.iterdir():
            if tree_dir.is_dir():
                programs += 1
                for cache_file in tree_dir.glob("*.json"):
                    total_entries += 1
                    total_size += cache_file.stat().st_size

        return {
            "total_entries": total_entries,
            "total_size_bytes": total_size,
            "programs": programs,
        }
