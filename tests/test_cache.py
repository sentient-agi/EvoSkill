"""Tests for src/cache/run_cache.py — CacheConfig and RunCache."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from src.cache.run_cache import CacheConfig, CacheEntry, RunCache
from src.harness.agent import AgentTrace

# Constants used to satisfy required sdk/model cache-key dimensions in tests.
_SDK = "test"
_MODEL = "test"


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

def _make_trace(result="result text", is_error=False, output=None, num_turns=3):
    # num_turns defaults to 3, not 1: RunCache.set deliberately SKIPS a trace
    # with output=None and num_turns<=1 (it reads as a transient startup
    # failure, logged as "RunCache.set SKIP"). A fixture meant to represent a
    # normal cacheable rollout must clear that guard — before 2026-07-24 these
    # tests asserted the opposite of the guard's intent and failed.
    return AgentTrace(
        duration_ms=100,
        total_cost_usd=0.001,
        num_turns=num_turns,
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
        result = cache.get("unknown question", sdk=_SDK, model=_MODEL)
        assert result is None

    def test_set_then_get_returns_trace(self, tmp_path):
        cache = self._make_cache(tmp_path)
        trace = _make_trace(result="cached answer")
        cache.set("my question", trace, sdk=_SDK, model=_MODEL)
        retrieved = cache.get("my question", sdk=_SDK, model=_MODEL)
        assert retrieved is not None
        assert retrieved.result == "cached answer"

    def test_get_respects_content_boundary(self, tmp_path):
        cache = self._make_cache(tmp_path)
        cache.set("question A", _make_trace("answer A"), sdk=_SDK, model=_MODEL)
        cache.set("question B", _make_trace("answer B"), sdk=_SDK, model=_MODEL)
        a = cache.get("question A", sdk=_SDK, model=_MODEL)
        b = cache.get("question B", sdk=_SDK, model=_MODEL)
        assert a.result == "answer A"
        assert b.result == "answer B"

    def test_set_when_disabled_is_noop(self, tmp_path):
        cache = self._make_cache(tmp_path, enabled=False)
        cache.set("q", _make_trace(), sdk=_SDK, model=_MODEL)
        # cache dir should not have been written to
        assert not (tmp_path / ".cache" / "runs").exists() or \
               len(list((tmp_path / ".cache" / "runs").glob("**/*.json"))) == 0

    def test_get_when_disabled_returns_none(self, tmp_path):
        cache = self._make_cache(tmp_path, enabled=False)
        result = cache.get("some question", sdk=_SDK, model=_MODEL)
        assert result is None

    def test_messages_not_stored_by_default(self, tmp_path):
        cache = self._make_cache(tmp_path)
        trace = AgentTrace(
            duration_ms=100,
            total_cost_usd=0.001,
            num_turns=3,  # > 1 so RunCache.set's transient-failure guard passes
            usage={},
            result="result",
            is_error=False,
            messages=[{"role": "user", "content": "hello"}],
        )
        cache.set("q", trace, sdk=_SDK, model=_MODEL)
        retrieved = cache.get("q", sdk=_SDK, model=_MODEL)
        assert retrieved.messages == []

    def test_messages_stored_when_option_enabled(self, tmp_path):
        cache = self._make_cache(tmp_path, store_messages=True)
        trace = AgentTrace(
            duration_ms=100,
            total_cost_usd=0.001,
            num_turns=3,  # > 1 so RunCache.set's transient-failure guard passes
            usage={},
            result="result",
            is_error=False,
            messages=[{"role": "user", "content": "hello"}],
        )
        cache.set("q", trace, sdk=_SDK, model=_MODEL)
        retrieved = cache.get("q", sdk=_SDK, model=_MODEL)
        assert len(retrieved.messages) == 1

    def test_corrupted_cache_entry_returns_none(self, tmp_path):
        cache = self._make_cache(tmp_path)
        trace = _make_trace()
        cache.set("q", trace, sdk=_SDK, model=_MODEL)

        # Corrupt the cache file
        tree_hash = cache._get_tree_hash()
        cache_path = cache._get_cache_path(tree_hash, "q", _SDK, _MODEL)
        cache_path.write_text("{ invalid json }")

        result = cache.get("q", sdk=_SDK, model=_MODEL)
        assert result is None

    def test_structured_output_round_trip(self, tmp_path):
        from src.schemas import AgentResponse

        cache = self._make_cache(tmp_path)
        output = AgentResponse(final_answer="42", reasoning="math")
        trace = _make_trace(output=output)
        cache.set("q with output", trace, sdk=_SDK, model=_MODEL)

        retrieved = cache.get("q with output", response_model=AgentResponse, sdk=_SDK, model=_MODEL)
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
        cache.set("q1", _make_trace(), sdk=_SDK, model=_MODEL)
        cache.set("q2", _make_trace(), sdk=_SDK, model=_MODEL)
        count = cache.clear()
        assert count >= 2

    def test_clear_all_empties_cache(self, tmp_path):
        cache = self._make_cache(tmp_path)
        cache.set("q", _make_trace(), sdk=_SDK, model=_MODEL)
        cache.clear()
        assert cache.get("q", sdk=_SDK, model=_MODEL) is None

    def test_clear_when_disabled_returns_zero(self, tmp_path):
        config = CacheConfig(cache_dir=tmp_path / "runs", enabled=False)
        cache = RunCache(config)
        assert cache.clear() == 0

    def test_clear_specific_tree_hash(self, tmp_path):
        cache = self._make_cache(tmp_path)
        cache.set("q", _make_trace(), sdk=_SDK, model=_MODEL)
        tree_hash = cache._get_tree_hash()

        count = cache.clear(tree_hash=tree_hash)
        assert count >= 1
        assert cache.get("q", sdk=_SDK, model=_MODEL) is None


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
        cache.set("q1", _make_trace(), sdk=_SDK, model=_MODEL)
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


class TestRunCacheSetSkipGuards:
    """The skip paths are deliberate and each logs a WARNING. Pin them so a
    future change can't silently start caching transient failures (or silently
    stop caching real ones — the 2026-07-22 nex run lost 71 of 117 rollouts to
    a silent no-write, which is why every skip path is logged)."""

    def _make_cache(self, tmp_path, **kwargs):
        config = CacheConfig(cache_dir=tmp_path / ".cache" / "runs", **kwargs)
        return RunCache(config)

    def test_skips_no_output_single_turn_trace(self, tmp_path):
        cache = self._make_cache(tmp_path)
        trace = _make_trace(output=None, num_turns=1)
        assert cache.set("q", trace, sdk=_SDK, model=_MODEL) is False
        assert cache.get("q", sdk=_SDK, model=_MODEL) is None

    def test_caches_no_output_multi_turn_trace(self, tmp_path):
        """A genuine wrong-answer/budget-exhausted episode DOES get cached —
        it recurs with the same skill+question, so re-running wastes money."""
        cache = self._make_cache(tmp_path)
        trace = _make_trace(output=None, num_turns=12)
        assert cache.set("q", trace, sdk=_SDK, model=_MODEL) is True
        assert cache.get("q", sdk=_SDK, model=_MODEL) is not None

    def test_caches_timeout_trace_even_at_one_turn(self, tmp_path):
        cache = self._make_cache(tmp_path)
        trace = _make_trace(result="[WALL_TIMEOUT] exceeded", num_turns=1)
        assert cache.set("q", trace, sdk=_SDK, model=_MODEL) is True

    def test_skip_when_disabled(self, tmp_path):
        cache = self._make_cache(tmp_path, enabled=False)
        assert cache.set("q", _make_trace(), sdk=_SDK, model=_MODEL) is False


class TestOnReplaySampleAwareness:
    """A replay hook must be told WHICH sample it is staging for.

    Before this, artifact restore put each sample's output under its own
    per-sample workspace while on_replay staged the inputs into the bare
    `{task_id}/` dir — so a cache-replayed multi-sample task came back with an
    output but no inputs, breaking the evolver's `$WS/spreadsheets/` contract
    (2026-07-24, fallout of the task #108 per-sample workspaces).
    """

    def _cache(self, tmp_path, hook):
        from src.cache.run_cache import CacheConfig, RunCache
        return RunCache(CacheConfig(cache_dir=tmp_path / "c", on_replay=hook))

    def test_sample_aware_hook_receives_index(self, tmp_path):
        seen = []
        cache = self._cache(
            tmp_path, lambda q, sample_idx=None: seen.append((q, sample_idx))
        )
        cache._run_on_replay("q", 2)
        assert seen == [("q", 2)]

    def test_legacy_one_arg_hook_still_works(self, tmp_path):
        seen = []
        cache = self._cache(tmp_path, lambda q: seen.append(q))
        cache._run_on_replay("q", 2)
        assert seen == ["q"]

    def test_typeerror_inside_hook_is_not_mistaken_for_old_signature(self, tmp_path):
        """The old duck-typing caught TypeError and re-called the 1-arg form —
        so a bug inside a sample-aware hook silently degraded to staging the
        SHARED directory. Signature inspection must not do that."""
        calls = []

        def hook(q, sample_idx=None):
            calls.append(sample_idx)
            raise TypeError("boom from inside the hook")

        cache = self._cache(tmp_path, hook)
        cache._run_on_replay("q", 1)  # swallowed + logged, never retried
        assert calls == [1]

    def test_kwargs_hook_counts_as_sample_aware(self, tmp_path):
        seen = []
        cache = self._cache(tmp_path, lambda q, **kw: seen.append(kw.get("sample_idx")))
        cache._run_on_replay("q", 3)
        assert seen == [3]


# ===========================================================================
# cached_transcript render budget (#149)
# ===========================================================================

class TestTranscriptRenderBudget:
    """The transcript is the ONLY structurally-usable record of a rollout after
    a cache write (`messages` does not round-trip through Pydantic and
    `store_messages` defaults off). Until 2026-08-04 the cap was head-only and
    hardcoded at both write sites, so an over-budget transcript lost its
    ENDING — measured on tau3: 26 of 55 stored rollouts, median 40% of turns
    gone, worst case 79 of 86 on a rollout that scored 1.0.

    These tests pin the three properties that made it a silent defect:
    the tail survives, the gap is LABELLED with the missing turn range, and an
    under-budget transcript is passed through byte-identically so the fix
    cannot alter a transcript that never needed truncating.
    """

    def _cache(self, tmp_path, **cfg):
        return RunCache(CacheConfig(cache_dir=tmp_path / "c", **cfg))

    @staticmethod
    def _trace(rendered: str):
        """A stand-in whose `.messages` IS the pre-rendered string, paired with
        the monkeypatched renderer below. Keeps the test on the cap logic
        rather than on SDK block shapes."""
        class _T:
            messages = rendered
        return _T()

    @pytest.fixture(autouse=True)
    def _identity_renderer(self):
        import src.harness.agent as A
        orig = A._render_turn_transcript
        A._render_turn_transcript = lambda messages, cap: messages
        yield
        A._render_turn_transcript = orig

    @staticmethod
    def _body(n_turns: int, pad: int = 80) -> str:
        return "".join(f"### Turn {i}\ntext: {'x' * pad}\n\n"
                       for i in range(1, n_turns + 1))

    def test_defaults_match_the_previously_hardcoded_values(self):
        c = CacheConfig()
        assert c.transcript_tool_result_max_chars == 20_000
        assert c.transcript_max_chars == 200_000

    def test_under_cap_is_byte_identical(self, tmp_path):
        cache = self._cache(tmp_path)
        body = self._body(3)
        assert cache._render_transcript_for_cache(self._trace(body)) == body

    def test_tail_is_preserved(self, tmp_path):
        """The regression that mattered: the END used to be the part dropped."""
        cache = self._cache(tmp_path, transcript_max_chars=1000)
        body = self._body(60)
        out = cache._render_transcript_for_cache(self._trace(body))
        assert len(out) <= 1000 + 400          # cap + the elision marker
        assert body[-60:] in out, "tail lost — this is exactly #149"
        assert body[:60] in out, "head lost"

    def test_marker_names_the_missing_turn_range(self, tmp_path):
        """A bare char count lets a reader treat 'Turn 7' -> 'Turn 55' as an
        episode that had 8 turns. The range is the part that prevents that."""
        cache = self._cache(tmp_path, transcript_max_chars=1000)
        out = cache._render_transcript_for_cache(self._trace(self._body(60)))
        assert "TRANSCRIPT ELIDED IN THE MIDDLE" in out
        import re as _re
        m = _re.search(r"turns (\d+)-(\d+) are NOT SHOWN", out)
        assert m, out[:400]
        lo, hi = int(m.group(1)), int(m.group(2))
        assert lo < hi
        # The named range must be genuinely absent, and its neighbours present.
        assert f"### Turn {lo}\n" not in out
        assert f"### Turn {hi}\n" not in out
        assert f"### Turn {lo - 1}\n" in out
        assert f"### Turn {hi + 1}\n" in out

    def test_zero_cap_disables_truncation(self, tmp_path):
        cache = self._cache(tmp_path, transcript_max_chars=0)
        body = self._body(60)
        assert cache._render_transcript_for_cache(self._trace(body)) == body

    def test_empty_render_returns_none(self, tmp_path):
        cache = self._cache(tmp_path)
        assert cache._render_transcript_for_cache(self._trace("")) is None

    def test_renderer_exception_does_not_lose_the_cache_write(self, tmp_path):
        import src.harness.agent as A
        A._render_turn_transcript = lambda messages, cap: (_ for _ in ()).throw(
            RuntimeError("boom"))
        cache = self._cache(tmp_path)
        assert cache._render_transcript_for_cache(self._trace("x")) is None

    def test_tool_result_cap_is_passed_through_to_the_renderer(self, tmp_path):
        seen = {}
        import src.harness.agent as A
        A._render_turn_transcript = lambda messages, cap: seen.setdefault("cap", cap) and ""
        cache = self._cache(tmp_path, transcript_tool_result_max_chars=1234)
        cache._render_transcript_for_cache(self._trace("x"))
        assert seen["cap"] == 1234

    def test_set_and_set_n_share_one_render_path(self, tmp_path):
        """Two copies of the same cap logic is the #147 trap: fixing the site
        you found leaves the symptom untouched. Assert there is one site."""
        import inspect
        import re as _re
        src = inspect.getsource(RunCache)
        assert len(_re.findall(r"def _render_transcript_for_cache", src)) == 1
        assert len(_re.findall(r"self\._render_transcript_for_cache\(", src)) == 2, (
            "expected exactly two call sites: set() and set_n()")
        # The caps must come from config, not from a literal that one site can
        # drift away from.
        assert "200_000" not in src and "20_000" not in src, (
            "a literal transcript cap crept back into RunCache")
