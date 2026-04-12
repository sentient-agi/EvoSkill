"""
Program-aware run caching for agent evaluations.

Cache invalidates automatically when behavior-affecting files change:
- .claude/skills/** (skill definitions)
- src/agent_profiles/base_agent/prompt.txt (prompt text)

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

        # Check cache
        trace = cache.get(question)
        if trace is None:
            trace = await agent.run(question)
            cache.set(question, trace)
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

        Only hashes files that actually affect agent behavior:
        - .claude/skills/** - skill definitions
        - src/agent_profiles/base_agent/prompt.txt - prompt text

        Excludes metadata files like .claude/program.yaml which contain
        mutable fields (score, created_at) that don't affect behavior.

        Returns:
            Combined hash of behavior-affecting files.
        """
        # Define paths that affect agent behavior
        # (directory, glob_pattern) tuples
        behavior_paths = [
            (".claude/skills", "**/*"),  # All skill files
            ("src/agent_profiles/base_agent", "prompt.txt"),  # Prompt text
        ]

        content_hashes = []
        for base_dir, pattern in behavior_paths:
            dir_path = self.config.cwd / base_dir
            if dir_path.exists():
                content_hashes.append(self._hash_files(dir_path, pattern))
            else:
                content_hashes.append("")

        # Combine hashes into a single hash
        combined = ":".join(content_hashes)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

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

    def _get_cache_path(self, tree_hash: str, question: str) -> Path:
        """Get the cache file path for a tree hash + question."""
        tree_dir = self._get_cache_dir_for_tree(tree_hash)
        question_hash = self._compute_question_hash(question)[: self.config.hash_length]
        return tree_dir / f"{question_hash}.json"

    def get(
        self,
        question: str,
        response_model: type[T] | None = None,
    ) -> AgentTrace[T] | None:
        """
        Retrieve cached trace for current .claude/ content + question.

        Args:
            question: The question/query string
            response_model: Optional Pydantic model for output validation

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

        cache_path = self._get_cache_path(tree_hash, question)

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
    ) -> None:
        """
        Cache a trace for current .claude/ content + question.

        Args:
            question: The question/query string
            trace: The AgentTrace to cache
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

        cache_path = self._get_cache_path(tree_hash, question)

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
                "question_hash": self._compute_question_hash(question)[
                    : self.config.hash_length
                ],
                "question": question,
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
