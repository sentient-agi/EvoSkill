"""Tests for src/cache/run_cache.py — CacheConfig and RunCache."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from src.cache.run_cache import CacheConfig, CacheEntry, RunCache
from src.harness.agent import AgentTrace


# ===========================================================================
# CacheConfig
# ===========================================================================

class TestCacheConfig:
    def test_defaults(self):
        config = CacheConfig()
        assert config.enabled is True
        assert config.store_messages is False
        assert config.hash_length == 12
        assert config.cache_dir == Path(".cache/runs")
        assert config.cwd == Path(".")

    def test_custom_values(self, tmp_path):
        config = CacheConfig(
            cache_dir=tmp_path / "cache",
            enabled=False,
            store_messages=True,
            hash_length=8,
            cwd=tmp_path,
        )
        assert config.enabled is False
        assert config.store_messages is True
        assert config.hash_length == 8
        assert config.cwd == tmp_path

    def test_arbitrary_types_allowed(self):
        # Path is an arbitrary type — should not raise
        config = CacheConfig(cache_dir=Path("/tmp/cache"), cwd=Path("/tmp"))
        assert isinstance(config.cache_dir, Path)


# ===========================================================================
# Helpers for building minimal AgentTrace instances
# ===========================================================================

def _make_trace(result="result text", is_error=False, output=None):
    return AgentTrace(
        duration_ms=100,
        total_cost_usd=0.001,
        num_turns=1,
        usage={"input_tokens": 10, "output_tokens": 5},
        result=result,
        is_error=is_error,
        messages=[],
        output=output,
    )


# ===========================================================================
# RunCache._compute_question_hash (static method)
# ===========================================================================

class TestComputeQuestionHash:
    def test_same_question_same_hash(self):
        h1 = RunCache._compute_question_hash("What is 2+2?")
        h2 = RunCache._compute_question_hash("What is 2+2?")
        assert h1 == h2

    def test_different_questions_different_hashes(self):
        h1 = RunCache._compute_question_hash("Question A")
        h2 = RunCache._compute_question_hash("Question B")
        assert h1 != h2

    def test_leading_trailing_whitespace_normalized(self):
        h1 = RunCache._compute_question_hash("  hello  ")
        h2 = RunCache._compute_question_hash("hello")
        assert h1 == h2

    def test_returns_hex_string(self):
        h = RunCache._compute_question_hash("test")
        int(h, 16)  # Should not raise


# ===========================================================================
# RunCache.set / RunCache.get — happy path
# ===========================================================================

class TestRunCacheSetGet:
    def _make_cache(self, tmp_path, **kwargs):
        config = CacheConfig(cache_dir=tmp_path / ".cache" / "runs", **kwargs)
        return RunCache(config)

    def test_cache_miss_returns_none(self, tmp_path):
        cache = self._make_cache(tmp_path)
        result = cache.get("unknown question")
        assert result is None

    def test_set_then_get_returns_trace(self, tmp_path):
        cache = self._make_cache(tmp_path)
        trace = _make_trace(result="cached answer")
        cache.set("my question", trace)
        retrieved = cache.get("my question")
        assert retrieved is not None
        assert retrieved.result == "cached answer"

    def test_get_respects_content_boundary(self, tmp_path):
        cache = self._make_cache(tmp_path)
        cache.set("question A", _make_trace("answer A"))
        cache.set("question B", _make_trace("answer B"))
        a = cache.get("question A")
        b = cache.get("question B")
        assert a.result == "answer A"
        assert b.result == "answer B"

    def test_set_when_disabled_is_noop(self, tmp_path):
        cache = self._make_cache(tmp_path, enabled=False)
        cache.set("q", _make_trace())
        # cache dir should not have been written to
        assert not (tmp_path / ".cache" / "runs").exists() or \
               len(list((tmp_path / ".cache" / "runs").glob("**/*.json"))) == 0

    def test_get_when_disabled_returns_none(self, tmp_path):
        cache = self._make_cache(tmp_path, enabled=False)
        result = cache.get("some question")
        assert result is None

    def test_messages_not_stored_by_default(self, tmp_path):
        cache = self._make_cache(tmp_path)
        trace = AgentTrace(
            duration_ms=100,
            total_cost_usd=0.001,
            num_turns=1,
            usage={},
            result="result",
            is_error=False,
            messages=[{"role": "user", "content": "hello"}],
        )
        cache.set("q", trace)
        retrieved = cache.get("q")
        assert retrieved.messages == []

    def test_messages_stored_when_option_enabled(self, tmp_path):
        cache = self._make_cache(tmp_path, store_messages=True)
        trace = AgentTrace(
            duration_ms=100,
            total_cost_usd=0.001,
            num_turns=1,
            usage={},
            result="result",
            is_error=False,
            messages=[{"role": "user", "content": "hello"}],
        )
        cache.set("q", trace)
        retrieved = cache.get("q")
        assert len(retrieved.messages) == 1

    def test_corrupted_cache_entry_returns_none(self, tmp_path):
        cache = self._make_cache(tmp_path)
        trace = _make_trace()
        cache.set("q", trace)

        # Corrupt the cache file
        tree_hash = cache._get_tree_hash()
        cache_path = cache._get_cache_path(tree_hash, "q")
        cache_path.write_text("{ invalid json }")

        result = cache.get("q")
        assert result is None

    def test_structured_output_round_trip(self, tmp_path):
        from src.schemas import AgentResponse

        cache = self._make_cache(tmp_path)
        output = AgentResponse(final_answer="42", reasoning="math")
        trace = _make_trace(output=output)
        cache.set("q with output", trace)

        retrieved = cache.get("q with output", response_model=AgentResponse)
        assert retrieved is not None
        assert retrieved.output.final_answer == "42"


# ===========================================================================
# RunCache.clear
# ===========================================================================

class TestRunCacheClear:
    def _make_cache(self, tmp_path):
        config = CacheConfig(cache_dir=tmp_path / "runs")
        return RunCache(config)

    def test_clear_all_returns_count(self, tmp_path):
        cache = self._make_cache(tmp_path)
        cache.set("q1", _make_trace())
        cache.set("q2", _make_trace())
        count = cache.clear()
        assert count >= 2

    def test_clear_all_empties_cache(self, tmp_path):
        cache = self._make_cache(tmp_path)
        cache.set("q", _make_trace())
        cache.clear()
        assert cache.get("q") is None

    def test_clear_when_disabled_returns_zero(self, tmp_path):
        config = CacheConfig(cache_dir=tmp_path / "runs", enabled=False)
        cache = RunCache(config)
        assert cache.clear() == 0

    def test_clear_specific_tree_hash(self, tmp_path):
        cache = self._make_cache(tmp_path)
        cache.set("q", _make_trace())
        tree_hash = cache._get_tree_hash()

        count = cache.clear(tree_hash=tree_hash)
        assert count >= 1
        assert cache.get("q") is None


# ===========================================================================
# RunCache.stats
# ===========================================================================

class TestRunCacheStats:
    def _make_cache(self, tmp_path):
        config = CacheConfig(cache_dir=tmp_path / "runs")
        return RunCache(config)

    def test_stats_empty_cache(self, tmp_path):
        cache = self._make_cache(tmp_path)
        stats = cache.stats()
        assert stats["total_entries"] == 0
        assert stats["total_size_bytes"] == 0
        assert stats["programs"] == 0

    def test_stats_after_set(self, tmp_path):
        cache = self._make_cache(tmp_path)
        cache.set("q1", _make_trace())
        stats = cache.stats()
        assert stats["total_entries"] >= 1
        assert stats["total_size_bytes"] > 0
        assert stats["programs"] >= 1

    def test_stats_missing_dir(self, tmp_path):
        config = CacheConfig(cache_dir=tmp_path / "nonexistent" / "runs")
        # Don't initialize so directory doesn't exist
        cache = RunCache.__new__(RunCache)
        cache.config = config
        stats = cache.stats()
        assert stats["total_entries"] == 0


# ===========================================================================
# RunCache._hash_files
# ===========================================================================

class TestHashFiles:
    def _make_cache(self, tmp_path):
        config = CacheConfig(cache_dir=tmp_path / "runs")
        return RunCache(config)

    def test_empty_directory_consistent_hash(self, tmp_path):
        cache = self._make_cache(tmp_path)
        d = tmp_path / "empty"
        d.mkdir()
        h1 = cache._hash_files(d)
        h2 = cache._hash_files(d)
        assert h1 == h2

    def test_hash_changes_when_file_content_changes(self, tmp_path):
        cache = self._make_cache(tmp_path)
        d = tmp_path / "dir"
        d.mkdir()
        f = d / "skill.md"
        f.write_text("version 1")
        h1 = cache._hash_files(d)
        f.write_text("version 2")
        h2 = cache._hash_files(d)
        assert h1 != h2

    def test_hash_changes_when_file_added(self, tmp_path):
        cache = self._make_cache(tmp_path)
        d = tmp_path / "dir"
        d.mkdir()
        (d / "a.md").write_text("file a")
        h1 = cache._hash_files(d)
        (d / "b.md").write_text("file b")
        h2 = cache._hash_files(d)
        assert h1 != h2
