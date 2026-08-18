"""The cache-environment marker must refuse a dir built under other conditions.

WHY. `RunCache`'s key cannot see the docker image or the turn budget, and it only
sees the system prompt when the caller remembers to pass it — which no SSB v2
runner did, so every SSB v2 cache dir shares bucket `bfb5bd90d1e0` regardless of
what the solver prompt said. Reusing such a dir across an environment change
replays the other environment's rollouts and reports them as this run's: no
error, no missing data, just a number describing a mixture.

These tests pin the refusal, and pin that there is NO override.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cache.cache_env import (  # noqa: E402
    MARKER, CacheEnvMismatch, assert_cache_env, sha12,
)

ENV_A = {"solver_image": "ssb-solver", "turn_budget": 45,
         "solver_prompt_sha256_12": "a85009cb8279"}
ENV_B = {**ENV_A, "solver_image": "ssb-solver-lo:latest"}


def test_first_use_stamps_the_marker(tmp_path):
    m = assert_cache_env(tmp_path / "c", ENV_A, label="run1")
    assert m.name == MARKER
    have = json.loads(m.read_text())
    assert have["solver_image"] == "ssb-solver"
    assert have["_label"] == "run1"


def test_same_environment_is_allowed(tmp_path):
    d = tmp_path / "c"
    assert_cache_env(d, ENV_A)
    assert_cache_env(d, ENV_A)          # must not raise


def test_key_order_does_not_count_as_a_change(tmp_path):
    d = tmp_path / "c"
    assert_cache_env(d, ENV_A)
    assert_cache_env(d, dict(reversed(list(ENV_A.items()))))


def test_a_changed_docker_image_is_refused(tmp_path):
    """The exact 2026-08-02 case: same prompt, same budget, new solver image."""
    d = tmp_path / "c"
    assert_cache_env(d, ENV_A)
    with pytest.raises(CacheEnvMismatch) as e:
        assert_cache_env(d, ENV_B)
    msg = str(e.value)
    assert "solver_image" in msg
    assert "ssb-solver-lo:latest" in msg
    # The message has to say WHY silence is the danger, or the next reader
    # assumes a stale marker and deletes it.
    assert "NOT part of the RunCache key" in msg


def test_a_changed_turn_budget_is_refused(tmp_path):
    d = tmp_path / "c"
    assert_cache_env(d, ENV_A)
    with pytest.raises(CacheEnvMismatch):
        assert_cache_env(d, {**ENV_A, "turn_budget": 40})


def test_a_changed_solver_prompt_is_refused(tmp_path):
    d = tmp_path / "c"
    assert_cache_env(d, ENV_A)
    with pytest.raises(CacheEnvMismatch):
        assert_cache_env(d, {**ENV_A,
                             "solver_prompt_sha256_12": sha12("something else")})


def test_a_newly_tracked_axis_is_refused(tmp_path):
    """Adding an axis must fail closed, not be treated as compatible."""
    d = tmp_path / "c"
    assert_cache_env(d, ENV_A)
    with pytest.raises(CacheEnvMismatch):
        assert_cache_env(d, {**ENV_A, "thinking_effort": "high"})


def test_no_override_parameter_exists():
    """The 2026-07-24 factorial measured an escape hatch taken 72/72."""
    import inspect
    sig = inspect.signature(assert_cache_env)
    assert set(sig.parameters) == {"cache_dir", "env", "label"}, (
        f"assert_cache_env grew a parameter: {list(sig.parameters)} — if one of "
        f"them bypasses the check, the guard is decorative")


def test_preexisting_records_are_counted_not_rejected(tmp_path):
    """A dir that predates the guard gets stamped, and says so.

    Refusing here would make introducing the guard a breaking change for every
    in-flight cache, so the compromise is recorded explicitly in the marker
    rather than left implicit.
    """
    d = tmp_path / "c"
    (d / "bfb5bd90d1e0").mkdir(parents=True)
    (d / "bfb5bd90d1e0" / "aa.json").write_text("{}")
    (d / "bfb5bd90d1e0" / "bb.json").write_text("{}")

    m = assert_cache_env(d, ENV_A)

    have = json.loads(m.read_text())
    assert have["_preexisting_records"] == 2
    assert "assumed to match" in have["_note"]
