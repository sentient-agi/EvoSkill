"""Regression tests for the two firewalls in scripts/build_rollout_digests.py.

WHY THESE EXIST. Both guards were added on 2026-08-01 after each had already
silently corrupted a digest set, and both were shipped as bare `raise SystemExit`
lines inside a top-level script. They fired correctly in live use, but nothing
would have caught their *removal* — which is the failure mode that matters,
because neither defect crashes. Each one produces a full, plausible-looking set
of digests that misrepresents which rollouts the evolver is reading:

  * SKILL-VERSION firewall — one cache bucket per tree_hash, i.e. per skill
    version. Records key on (task_id, sample_idx), which is identical across
    buckets, so an unfiltered scan overwrites one version's rollouts with the
    other's and labels pre-edit traces as post-edit.
  * VAL firewall — an epoch's train and val legs share a skills dir, therefore
    the same bucket. A bare scan sweeps val rollouts into the train analysis,
    breaking the run's first design decision (edit from train only).

The script is top-level (it executes on import), so these run it as a
subprocess against a synthetic cache — which also means they exercise the real
script through its real env-var interface rather than a reimplementation of it.

Each test asserts the guard FIRES on the bad input. A guard that is deleted or
weakened turns these red.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Overridable so the guards can be MUTATION-TESTED against a throwaway copy.
# A guard test that has never been seen to fail proves nothing, and mutating the
# real script is unsafe while a run is live (the driver invokes it between
# epochs). See scripts/mutation_check_digest_firewalls.sh.
SCRIPT = Path(os.environ.get("EVOSKILL_TEST_DIGEST_SCRIPT",
                             ROOT / "scripts" / "build_rollout_digests.py"))
MODEL = "deepseek/deepseek-v4-flash-0731"


def _question(task_id: str) -> str:
    """The JSON blob the script parses out of cache_key.question."""
    return json.dumps(
        {
            "task_id": task_id,
            "instruction": f"do the thing for {task_id}",
            "category": "Debugging",
            "spreadsheet_filename": f"{task_id}.xlsx",
        }
    )


def _record_name(question: str, sample: int) -> str:
    """Reproduce RunCache's filename so true_sample_idx() can recover the index.

    Mirrors the keying in build_rollout_digests.true_sample_idx: if that
    convention ever changes, these tests go red rather than silently building
    records the script skips as `unresolved`.
    """
    keyed = f"{question.strip()}|sdk=direct|model={MODEL}|sample={sample}"
    return hashlib.sha256(keyed.encode("utf-8")).hexdigest()[:12]


def _messages(skill: str | None):
    """A rollout transcript, optionally one that LOADED a skill body.

    A skill reaches the solver by progressive disclosure: it reads the file, so
    the body arrives as a tool RESULT carrying SKILL.md frontmatter. That is the
    evidence the leg-identity assertion keys on — behaviour the record cannot
    fake, unlike the tree_hash it merely carries.
    """
    msgs = [{"role": "user", "content": "hello"}]
    if skill:
        msgs.append({"role": "tool", "content": [
            {"type": "tool_result",
             "content": f"---\nname: {skill}\ndescription: x\n---\n# Title\nbody"}]})
    return msgs


def _write_record(bucket: Path, task_id: str, sample: int, score: float = 0.0,
                  skill: str | None = None) -> Path:
    q = _question(task_id)
    rec = {
        "cache_key": {"question": q, "sdk": "direct", "model": MODEL},
        "trace": {
            "inline_score": score,
            "num_turns": 7,
            "budget_exhausted": "",
            "total_cost_usd": 0.01,
            "messages": _messages(skill),
        },
    }
    bucket.mkdir(parents=True, exist_ok=True)
    p = bucket / f"{_record_name(q, sample)}.json"
    p.write_text(json.dumps(rec))
    return p


def _run(cache: Path, out: Path, snaps: Path, **env_extra) -> subprocess.CompletedProcess:
    env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": str(Path.home()),
        "EVOSKILL_DIGEST_CACHE": str(cache),
        "EVOSKILL_DIGEST_SNAPS": str(snaps),
        "EVOSKILL_DIGEST_OUT": str(out),
        "EVOSKILL_SOLVER_MODEL": MODEL,
    }
    env.update({k: str(v) for k, v in env_extra.items()})
    out.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        env=env, capture_output=True, text=True, timeout=120,
    )


@pytest.fixture
def workspace(tmp_path: Path):
    cache = tmp_path / "cache"
    snaps = tmp_path / "snaps"
    out = tmp_path / "out"
    snaps.mkdir(parents=True, exist_ok=True)
    return cache, snaps, out


# --------------------------------------------------------------------------
# SKILL-VERSION firewall
# --------------------------------------------------------------------------

def test_refuses_multiple_buckets_when_bucket_unset(workspace):
    """Two skill versions in the cache and no bucket named => refuse to guess.

    This is the exact 0801 situation: by epoch 1 the same train tasks exist in
    both the no-skill and v1 buckets.
    """
    cache, snaps, out = workspace
    _write_record(cache / "aaaaaaaaaaaa", "Debugging/01_01", 0)
    _write_record(cache / "bbbbbbbbbbbb", "Debugging/01_01", 0)

    r = _run(cache, out, snaps)

    assert r.returncode != 0, "two buckets with no BUCKET set must NOT succeed"
    assert "refusing to mix skill versions" in r.stderr
    assert "aaaaaaaaaaaa" in r.stderr and "bbbbbbbbbbbb" in r.stderr


def test_single_bucket_needs_no_bucket_env(workspace):
    """One bucket is unambiguous — the guard must not block ordinary use."""
    cache, snaps, out = workspace
    for s in range(3):
        _write_record(cache / "aaaaaaaaaaaa", "Debugging/01_01", s)

    r = _run(cache, out, snaps)

    assert r.returncode == 0, r.stderr
    assert "BUCKET: aaaaaaaaaaaa (of 1 in cache)" in r.stdout
    assert (out / "Debugging_01_01.md").exists()


def test_named_bucket_scopes_the_scan(workspace):
    """With BUCKET set, only that bucket's rollouts are digested."""
    cache, snaps, out = workspace
    for s in range(3):
        _write_record(cache / "aaaaaaaaaaaa", "Debugging/01_01", s)
        _write_record(cache / "bbbbbbbbbbbb", "Debugging/02_02", s)

    r = _run(cache, out, snaps, EVOSKILL_DIGEST_BUCKET="aaaaaaaaaaaa")

    assert r.returncode == 0, r.stderr
    assert (out / "Debugging_01_01.md").exists()
    assert not (out / "Debugging_02_02.md").exists(), "other bucket leaked in"


def test_nonexistent_bucket_is_rejected(workspace):
    """A typo'd bucket name must fail loudly, not scan nothing and 'succeed'."""
    cache, snaps, out = workspace
    _write_record(cache / "aaaaaaaaaaaa", "Debugging/01_01", 0)

    r = _run(cache, out, snaps, EVOSKILL_DIGEST_BUCKET="deadbeefcafe")

    assert r.returncode != 0
    assert "no such cache bucket" in r.stderr


def test_duplicate_task_sample_collision_is_fatal(workspace):
    """The general guard: ANY duplicate (task, sample) aborts.

    Backstop for a future cache layout that reintroduces cross-version merging
    by some route other than sibling top-level buckets — here, two records for
    the same (task, sample) reachable under one bucket via a subdirectory,
    which rglob() walks into.
    """
    cache, snaps, out = workspace
    bucket = cache / "aaaaaaaaaaaa"
    _write_record(bucket, "Debugging/01_01", 0)
    _write_record(bucket / "nested_version_dir", "Debugging/01_01", 0)

    r = _run(cache, out, snaps, EVOSKILL_DIGEST_BUCKET="aaaaaaaaaaaa")

    assert r.returncode != 0, "duplicate (task, sample) must abort"
    assert "DUPLICATE (task, sample)" in r.stderr
    assert "Debugging/01_01 s0" in r.stderr


# --------------------------------------------------------------------------
# VAL firewall
# --------------------------------------------------------------------------

def _split_file(tmp: Path) -> Path:
    p = tmp / "split.json"
    p.write_text(json.dumps({
        "train": ["Debugging/01_01", "Debugging/01_02"],
        "val": ["Debugging/09_09"],
        "test": ["Debugging/10_10"],
    }))
    return p


def test_val_rollouts_are_excluded_from_train_digests(workspace, tmp_path):
    """Train and val share a bucket; only the named split may reach the digests."""
    cache, snaps, out = workspace
    bucket = cache / "aaaaaaaaaaaa"
    for s in range(3):
        _write_record(bucket, "Debugging/01_01", s)
        _write_record(bucket, "Debugging/09_09", s)   # val, same bucket
        _write_record(bucket, "Debugging/10_10", s)   # test, same bucket

    r = _run(cache, out, snaps,
             EVOSKILL_DIGEST_BUCKET="aaaaaaaaaaaa",
             EVOSKILL_DIGEST_SPLIT_FILE=str(_split_file(tmp_path)),
             EVOSKILL_DIGEST_SPLIT_KEY="train")

    assert r.returncode == 0, r.stderr
    assert (out / "Debugging_01_01.md").exists()
    assert not (out / "Debugging_09_09.md").exists(), "VAL task reached the digests"
    assert not (out / "Debugging_10_10.md").exists(), "TEST task reached the digests"
    assert "excluded (other splits): 6 rollouts" in r.stdout
    assert "firewall OK: 0 of 2 out-of-split tasks present" in r.stdout


def test_split_file_without_split_key_is_fatal(workspace, tmp_path):
    """Half-configured firewall must abort rather than silently pass everything."""
    cache, snaps, out = workspace
    _write_record(cache / "aaaaaaaaaaaa", "Debugging/01_01", 0)

    r = _run(cache, out, snaps,
             EVOSKILL_DIGEST_BUCKET="aaaaaaaaaaaa",
             EVOSKILL_DIGEST_SPLIT_FILE=str(_split_file(tmp_path)))

    assert r.returncode != 0
    assert "SPLIT_FILE set without EVOSKILL_DIGEST_SPLIT_KEY" in r.stderr


def test_breach_assertion_fires_when_the_filter_is_bypassed(workspace, tmp_path, monkeypatch):
    """The post-hoc assertion must catch a leak the pre-filter missed.

    The filter and the assertion are deliberately redundant: the filter can be
    defeated by a task_id that is spelled differently in the split file than in
    the cache record. Here the split lists 'Debugging/01_01' while the cache
    holds the same task under a trailing-space variant, so the pre-filter
    excludes the *real* train task and the breach check is what has to fire.
    """
    cache, snaps, out = workspace
    bucket = cache / "aaaaaaaaaaaa"
    split = tmp_path / "split.json"
    # 'val' contains a task that IS present in the cache and IS allowed by the
    # train key -- i.e. the two sets overlap, which the filter cannot resolve.
    split.write_text(json.dumps({
        "train": ["Debugging/01_01"],
        "val": ["Debugging/01_01"],
        "test": [],
    }))
    for s in range(3):
        _write_record(bucket, "Debugging/01_01", s)

    r = _run(cache, out, snaps,
             EVOSKILL_DIGEST_BUCKET="aaaaaaaaaaaa",
             EVOSKILL_DIGEST_SPLIT_FILE=str(split),
             EVOSKILL_DIGEST_SPLIT_KEY="train")

    assert r.returncode != 0, "overlapping splits must not produce digests"
    assert "VAL FIREWALL BREACH" in r.stderr


def test_missing_split_tasks_are_reported(workspace, tmp_path):
    """Silence about absent tasks would read as full coverage."""
    cache, snaps, out = workspace
    bucket = cache / "aaaaaaaaaaaa"
    for s in range(3):
        _write_record(bucket, "Debugging/01_01", s)   # 01_02 has no rollouts

    r = _run(cache, out, snaps,
             EVOSKILL_DIGEST_BUCKET="aaaaaaaaaaaa",
             EVOSKILL_DIGEST_SPLIT_FILE=str(_split_file(tmp_path)),
             EVOSKILL_DIGEST_SPLIT_KEY="train")

    assert r.returncode == 0, r.stderr
    assert "WARNING: 1 split tasks have NO rollouts" in r.stdout
    assert "Debugging/01_02" in r.stdout


# --------------------------------------------------------------------------
# sample-index recovery (the pairing bug the firewalls sit on top of)
# --------------------------------------------------------------------------

def test_sample_index_comes_from_the_filename_not_scan_order(workspace):
    """Rollouts must be labelled s0/s1/s2 by hash, never by rglob() order.

    Inferring the index from iteration order mislabelled 53 of 90 rollouts on
    0731. Writing the records so that filename order and sample order disagree
    would silently pass under an order-based implementation.
    """
    cache, snaps, out = workspace
    bucket = cache / "aaaaaaaaaaaa"
    scores = {0: 0.0, 1: 1.0, 2: 0.0}
    paths = {s: _write_record(bucket, "Debugging/01_01", s, score=scores[s])
             for s in (0, 1, 2)}
    # Only meaningful if the on-disk ordering isn't already 0,1,2.
    by_name = sorted(paths, key=lambda s: paths[s].name)
    assert by_name != [0, 1, 2], "fixture no longer exercises the ordering hazard"

    r = _run(cache, out, snaps, EVOSKILL_DIGEST_BUCKET="aaaaaaaaaaaa")

    assert r.returncode == 0, r.stderr
    assert "unresolved sample_idx: 0" in r.stdout
    body = (out / "Debugging_01_01.md").read_text()
    assert "## Rollout s1 — PASS" in body, "the PASS must be attributed to s1"
    assert "## Rollout s0 — FAIL" in body
    assert "## Rollout s2 — FAIL" in body


# --- LEG IDENTITY -----------------------------------------------------------
# HARNESS PARITY, added 2026-08-04. `tau3_build_digests.py` carried this
# assertion; the SSB v2 path — whose contamination motivated it — did not.
#
# The check it REPLACES is the instructive part. "Every record in this bucket
# shares one tree_hash" is true BY CONSTRUCTION: the bucket directory *is*
# `tree_hash[:12]` and the stored tree_hash is that same truncated string. It
# reads like a contamination check, can never fail for any input, and is why 8
# skills_v4 rollouts sat in SSB v2's BASE bucket counted as base. Record COUNT is
# no proof either — a foreign write lands on a slot the leg never ran, so the
# count staying right is exactly what hides it.
#
# The assertion below keys on BEHAVIOUR instead: a base solver is never handed a
# skill, so skill frontmatter in a base rollout's tool result is evidence the
# record cannot fake. It works because contamination is one-directional — an
# empty/missing skills dir hashes to precisely the base tree_hash.

def test_base_leg_refuses_a_rollout_that_loaded_a_skill(workspace):
    """The 8-contaminated-records case, as a test."""
    cache, snaps, out = workspace
    B = cache / "aaaaaaaaaaaa"
    _write_record(B, "Debugging/01_01", 0)
    _write_record(B, "Debugging/01_01", 1, skill="xlsx-save-output-first")
    r = _run(cache, out, snaps,
             EVOSKILL_DIGEST_BUCKET=B.name, EVOSKILL_DIGEST_LEG="base")
    assert r.returncode != 0, r.stdout
    assert "BASE LEG CONTAMINATED" in r.stderr
    assert "1 of 2 rollouts" in r.stderr


def test_base_leg_passes_when_no_rollout_loaded_a_skill(workspace):
    cache, snaps, out = workspace
    B = cache / "aaaaaaaaaaaa"
    for s in (0, 1, 2):
        _write_record(B, "Debugging/01_01", s)
    r = _run(cache, out, snaps,
             EVOSKILL_DIGEST_BUCKET=B.name, EVOSKILL_DIGEST_LEG="base")
    assert r.returncode == 0, r.stderr
    assert "BASE asserted — 0/3 rollouts loaded a skill body [OK]" in r.stdout


def test_skill_leg_reports_partial_uptake_and_does_NOT_assert_it(workspace):
    """The symmetric assertion would be WRONG: honest uptake is below 100%.

    SSB v1 loaded a skill body in 330 of 360 rollouts, so demanding 3/3 here would
    fire on good data and get switched off — which is how a real guard dies.
    """
    cache, snaps, out = workspace
    B = cache / "aaaaaaaaaaaa"
    _write_record(B, "Debugging/01_01", 0, skill="xlsx-save-output-first")
    _write_record(B, "Debugging/01_01", 1)          # no uptake, legitimate
    _write_record(B, "Debugging/01_01", 2, skill="xlsx-audit-minimal-repair")
    r = _run(cache, out, snaps,
             EVOSKILL_DIGEST_BUCKET=B.name, EVOSKILL_DIGEST_LEG="skill")
    assert r.returncode == 0, r.stderr
    assert "SKILL leg — 2/3 rollouts loaded a skill body" in r.stdout
    assert "xlsx-audit-minimal-repair" in r.stdout and "xlsx-save-output-first" in r.stdout


def test_unset_leg_reports_but_does_not_gate(workspace):
    """Unset must never silently pass a contaminated set off as checked."""
    cache, snaps, out = workspace
    B = cache / "aaaaaaaaaaaa"
    _write_record(B, "Debugging/01_01", 0, skill="xlsx-save-output-first")
    r = _run(cache, out, snaps,
             EVOSKILL_DIGEST_BUCKET=B.name)
    assert r.returncode == 0, r.stderr
    assert "NOT ASSERTED" in r.stdout
    assert "1/1 rollouts loaded a skill body" in r.stdout
