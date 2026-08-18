"""SSB v2 per-sample workspace isolation (parity port of the v1 0724 fix).

The bug these guard against: SSB v2 named workspaces by task_id alone, so with
n_samples>1 every sample of a task bind-mounted the SAME directory and wrote the
SAME output/<name>.xlsx. Found 2026-07-31 on a real 30x3 DSV4F run.

Two properties matter and are tested separately, because only one of them is
obvious:

  1. Distinct samples get distinct directories (obvious).
  2. sample_idx=None reproduces the EXACT historical path (not obvious, and the
     more important one — every SSB v2 evolution script runs n_samples=1, so a
     naming change there would silently invalidate the whole existing corpus of
     v2 results, including the GLM 5.2 headline).
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from src.agent_profiles.ssbv2_agent.task_workspace import (
    get_output_dir,
    setup_task_workspace,
    snapshot_output,
    task_ws_name,
)


TASK = "Financial_Model/07_02"
SAFE = "Financial_Model_07_02"


# --- naming ---------------------------------------------------------------

def test_none_reproduces_historical_name():
    """n_samples=1 layout must be byte-for-byte what it always was."""
    assert task_ws_name(TASK) == SAFE
    assert task_ws_name(TASK, None) == SAFE


def test_samples_get_distinct_names():
    names = {task_ws_name(TASK, s) for s in range(3)}
    assert len(names) == 3
    assert names == {f"{SAFE}__s0", f"{SAFE}__s1", f"{SAFE}__s2"}


def test_slashes_sanitised_before_suffix():
    """v2 task ids contain '/'. The suffix must not reintroduce a path level."""
    n = task_ws_name(TASK, 1)
    assert "/" not in n
    assert n.startswith(SAFE)


def test_sample_zero_is_not_treated_as_none():
    """0 is falsy — an `if sample_idx:` check would collapse s0 onto the shared
    dir, putting sample 0 back in collision with every n_samples=1 rollout."""
    assert task_ws_name(TASK, 0) != task_ws_name(TASK, None)
    assert task_ws_name(TASK, 0) == f"{SAFE}__s0"


# --- workspace creation ---------------------------------------------------

@pytest.fixture()
def staged(tmp_path: Path):
    src = tmp_path / "src" / "07_02_RealEstate_input.xlsx"
    src.parent.mkdir(parents=True)
    src.write_bytes(b"PK\x03\x04original")
    skills = tmp_path / "skills"
    skills.mkdir()
    return src, skills, tmp_path / "tasks"


def _setup(staged, sample_idx):
    src, skills, root = staged
    return setup_task_workspace(
        task_id=TASK,
        spreadsheet_path=str(src),
        tasks_cache_root=root,
        shared_skills_dir=skills,
        sample_idx=sample_idx,
    )


def test_concurrent_samples_do_not_share_a_directory(staged):
    ws = [_setup(staged, s) for s in range(3)]
    assert len({str(w) for w in ws}) == 3
    for w in ws:
        assert (w / "spreadsheets" / "07_02_RealEstate_input.xlsx").exists()


def test_sibling_setup_does_not_wipe_a_running_samples_output(staged):
    """The concrete 0724/0731 failure: setup rmtree's output/ on entry, so a
    sample starting up destroyed a still-running sibling's work."""
    ws0 = _setup(staged, 0)
    inflight = ws0 / "output" / "07_02_RealEstate_input.xlsx"
    inflight.write_bytes(b"PK\x03\x04sample-0-work-in-progress")

    _setup(staged, 1)          # sibling starts while sample 0 is mid-rollout

    assert inflight.exists(), "sibling setup wiped a live sample's output"
    assert inflight.read_bytes() == b"PK\x03\x04sample-0-work-in-progress"


def test_each_sample_scores_its_own_file(staged):
    """A sample must never be scored against a sibling's output."""
    for s in range(3):
        ws = _setup(staged, s)
        (ws / "output" / "07_02_RealEstate_input.xlsx").write_bytes(
            f"PK\x03\x04sample-{s}".encode()
        )
    for s in range(3):
        out = get_output_dir(TASK, staged[2], s) / "07_02_RealEstate_input.xlsx"
        assert out.read_bytes() == f"PK\x03\x04sample-{s}".encode()


def test_get_output_dir_matches_setup_for_every_index(staged):
    for s in (None, 0, 1, 2):
        ws = _setup(staged, s)
        assert get_output_dir(TASK, staged[2], s) == ws / "output"


# --- snapshots ------------------------------------------------------------

def test_snapshots_do_not_overwrite_across_samples(staged):
    """Without a per-sample label the N samples collapse to one arbitrary
    survivor and the evolver sees a point instead of the spread."""
    root = staged[2]
    for s in range(3):
        ws = _setup(staged, s)
        (ws / "output" / "out.xlsx").write_bytes(f"s{s}".encode())
        snapshot_output(TASK, root, 1, diff_text=f"diff-{s}", sample_idx=s)

    snaps = sorted(p.name for p in (root / "_snapshots" / SAFE).iterdir())
    assert snaps == ["iter_1_s0", "iter_1_s1", "iter_1_s2"]
    for s in range(3):
        d = root / "_snapshots" / SAFE / f"iter_1_s{s}"
        assert (d / "DIFF.txt").read_text() == f"diff-{s}"
        assert (d / "out.xlsx").read_bytes() == f"s{s}".encode()


def test_snapshot_without_sample_idx_keeps_plain_iter_label(staged):
    root = staged[2]
    ws = _setup(staged, None)
    (ws / "output" / "out.xlsx").write_bytes(b"single")
    snapshot_output(TASK, root, 2, diff_text="d")
    assert (root / "_snapshots" / SAFE / "iter_2" / "DIFF.txt").read_text() == "d"


def test_midgate_and_sample_labels_compose(staged):
    root = staged[2]
    ws = _setup(staged, 1)
    (ws / "output" / "out.xlsx").write_bytes(b"x")
    snapshot_output(TASK, root, 3, diff_text="d", phase="midgate", sample_idx=1)
    assert (root / "_snapshots" / SAFE / "iter_3_midgate_s1").is_dir()


def test_snapshot_reads_the_matching_samples_output(staged):
    """snapshot_output must resolve output/ through task_ws_name too — reading
    the shared path would snapshot a sibling's bytes under this sample's label."""
    root = staged[2]
    _setup(staged, 0)
    ws1 = _setup(staged, 1)
    (ws1 / "output" / "out.xlsx").write_bytes(b"belongs-to-s1")
    snapshot_output(TASK, root, 1, sample_idx=1)
    snap = root / "_snapshots" / SAFE / "iter_1_s1" / "out.xlsx"
    assert snap.read_bytes() == b"belongs-to-s1"


# --- call-site contract ---------------------------------------------------

def test_agent_run_accepts_sample_idx():
    """The plumbing is only useful if the agent can actually be told the index;
    a correct task_workspace with an agent that never passes through is the
    same bug with extra steps."""
    from src.agent_profiles.ssbv2_agent.ssb_v2_agent import SSBv2SolverAgent

    sig = inspect.signature(SSBv2SolverAgent.run)
    assert "sample_idx" in sig.parameters
    assert sig.parameters["sample_idx"].default is None


def test_run_script_passes_sample_idx():
    """Regression guard on the launcher itself: the harness fix does nothing
    if the N=3 script keeps calling run() without the index."""
    src = Path("scripts/run_ssbv2_base30_dsv4f_n3.py").read_text()
    assert "sample_idx=s" in src
