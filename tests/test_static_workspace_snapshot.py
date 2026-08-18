"""The evolver's evidence must be IMMUTABLE and TRAIN-ONLY, mechanically.

Two properties, both enforced by construction rather than by prompt text
(0725 goal). A real solver and evolver are simulated with plain file I/O — no
API calls — because every property under test is a filesystem invariant:

  1. TRUE-STATIC SNAPSHOT. Each rollout sample's whole working dir is frozen at
     scoring time. A later re-run of the same task must not change one byte the
     evolver can reach. Before this, the evolver was pointed at the LIVE
     `{task_id}__s{K}/` dir for Stage 1's input read, so it could diagnose one
     rollout using a different rollout's bytes.

  2. TRAIN-ONLY ORACLE. Val and test golden answers must be unreachable — not
     merely unmentioned. The launcher used to bind-mount the entire dataset
     read-only while `write_all_task_meta` wrote metas for all 400 tasks
     "regardless of split", so the only restraint was a sentence in a prompt.

The tests are written against the PROPERTY, not the implementation: they read
what an evolver could read and assert what it must not find.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.agent_profiles.spreadsheetbench_agent.task_workspace import (
    SNAPSHOTS_ROOT_NAME,
    _snapshot_sample_workspace,
    snapshot_output,
    stage_split_dataset,
    task_ws_name,
    write_all_task_meta,
)

# --------------------------------------------------------------------------
# Simulated benchmark: a dataset, a solver that writes into its workspace,
# and an evolver that may only read what it is handed.
# --------------------------------------------------------------------------


def _make_dataset(root: Path, ids) -> None:
    """A miniature SSB-shaped dataset: one dir per task, with a golden answer."""
    for tid in ids:
        tests = root / tid / "tests"
        (tests / "answer_files").mkdir(parents=True, exist_ok=True)
        (tests / "config.json").write_text(json.dumps({
            "id": tid, "num_test_cases": 1, "answer_position": "A1",
            "instruction_type": "Cell-Level Manipulation",
        }))
        (tests / "answer_files" / f"1_{tid}_answer.xlsx").write_bytes(
            f"GOLDEN::{tid}".encode())
        (root / tid / "environment").mkdir(parents=True, exist_ok=True)
        (root / tid / "environment" / "spreadsheets").mkdir(exist_ok=True)
        (root / tid / "environment" / "spreadsheets" / f"{tid}.xlsx").write_bytes(
            f"INPUT::{tid}".encode())


def _simulate_solver(cache: Path, task_id: str, sample_idx: int, *,
                     answer: bytes, scratch: str | None = None,
                     link_skills_to: Path | None = None) -> Path:
    """Build a sample workspace the way the real harness does, then 'solve'.

    Mirrors build_task_workspace's shape: `spreadsheets/` (input), `output/`
    (what the solver writes), and a `.claude/skills` SYMLINK to the shared
    evolving skills dir.
    """
    ws = cache / task_ws_name(task_id, sample_idx)
    (ws / "spreadsheets").mkdir(parents=True, exist_ok=True)
    (ws / "output").mkdir(parents=True, exist_ok=True)
    (ws / "spreadsheets" / f"{task_id}.xlsx").write_bytes(f"INPUT::{task_id}".encode())
    (ws / "output" / f"1_{task_id}_output.xlsx").write_bytes(answer)
    if scratch is not None:
        (ws / "solve.py").write_text(scratch)
    if link_skills_to is not None:
        link = ws / ".claude" / "skills"
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(link_skills_to, target_is_directory=True)
    return ws


def _snap_dir(cache: Path, task_id: str, iteration: int, sample_idx: int) -> Path:
    return (cache / SNAPSHOTS_ROOT_NAME / task_id
            / f"iter_{iteration}_s{sample_idx}")


def _evolver_readable(snap: Path) -> dict[str, bytes]:
    """Everything an evolver could read under one snapshot, as {relpath: bytes}."""
    return {
        str(p.relative_to(snap)): p.read_bytes()
        for p in sorted(snap.rglob("*")) if p.is_file()
    }


# ==========================================================================
# 1 · TRUE-STATIC SNAPSHOT
# ==========================================================================


class TestStaticSnapshot:
    def test_captures_input_output_and_scratch(self, tmp_path):
        """The snapshot must hold everything the rollout could see or write.

        The input matters most: Stage 1's derivation gate reads it, and it was
        previously reachable ONLY through the live working dir.
        """
        cache = tmp_path / "tasks"
        _simulate_solver(cache, "T1", 0, answer=b"OUT-A", scratch="print(1)\n")
        snapshot_output("T1", cache, 1, diff_text="mismatch at A1", sample_idx=0)

        got = _evolver_readable(_snap_dir(cache, "T1", 1, 0))
        assert got["workspace/spreadsheets/T1.xlsx"] == b"INPUT::T1"
        assert got["workspace/output/1_T1_output.xlsx"] == b"OUT-A"
        assert got["workspace/solve.py"] == b"print(1)\n"
        assert got["DIFF.txt"] == b"mismatch at A1"

    def test_backward_compatible_flat_output_and_diff(self, tmp_path):
        """8+ rescore scripts index `_snapshots/{id}/iter_{n}/<name>.xlsx`.

        Widening the snapshot must not relocate what they read.
        """
        cache = tmp_path / "tasks"
        _simulate_solver(cache, "T1", 0, answer=b"OUT-A")
        snapshot_output("T1", cache, 1, diff_text="d", sample_idx=0)
        snap = _snap_dir(cache, "T1", 1, 0)
        assert (snap / "1_T1_output.xlsx").read_bytes() == b"OUT-A"
        assert (snap / "DIFF.txt").read_text() == "d"

    def test_a_later_rerun_cannot_change_the_snapshot(self, tmp_path):
        """THE core property. A re-run overwrites the LIVE workspace; every byte
        the evolver can reach must be unaffected."""
        cache = tmp_path / "tasks"
        _simulate_solver(cache, "T1", 0, answer=b"FIRST", scratch="v1\n")
        snapshot_output("T1", cache, 1, diff_text="first", sample_idx=0)
        before = _evolver_readable(_snap_dir(cache, "T1", 1, 0))

        # A later iteration re-rolls the same task into the same live dir.
        ws = _simulate_solver(cache, "T1", 0, answer=b"SECOND", scratch="v2\n")
        (ws / "spreadsheets" / "T1.xlsx").write_bytes(b"MUTATED-INPUT")

        after = _evolver_readable(_snap_dir(cache, "T1", 1, 0))
        assert after == before, "a re-run mutated bytes the evolver had frozen"
        assert after["workspace/output/1_T1_output.xlsx"] == b"FIRST"
        assert after["workspace/spreadsheets/T1.xlsx"] == b"INPUT::T1"

    def test_deleting_the_live_workspace_leaves_the_snapshot_intact(self, tmp_path):
        """Workspaces get cleaned between runs (tonight's 20x3 workspace had
        empty solver_outputs/). The snapshot must survive that."""
        import shutil
        cache = tmp_path / "tasks"
        ws = _simulate_solver(cache, "T1", 0, answer=b"OUT", scratch="s\n")
        snapshot_output("T1", cache, 1, diff_text="d", sample_idx=0)
        shutil.rmtree(ws)
        got = _evolver_readable(_snap_dir(cache, "T1", 1, 0))
        assert got["workspace/output/1_T1_output.xlsx"] == b"OUT"
        assert got["workspace/solve.py"] == b"s\n"

    def test_siblings_do_not_collide_and_keep_their_own_bytes(self, tmp_path):
        """Per-sample snapshots: s0/s1/s2 each frozen separately. A shared path
        would show one sibling's file for another's diagnosis."""
        cache = tmp_path / "tasks"
        for si, ans in enumerate((b"A", b"B", b"C")):
            _simulate_solver(cache, "T1", si, answer=ans, scratch=f"s{si}\n")
            snapshot_output("T1", cache, 1, diff_text=f"d{si}", sample_idx=si)
        for si, ans in enumerate((b"A", b"B", b"C")):
            got = _evolver_readable(_snap_dir(cache, "T1", 1, si))
            assert got["workspace/output/1_T1_output.xlsx"] == ans
            assert got["workspace/solve.py"] == f"s{si}\n".encode()
            assert got["DIFF.txt"] == f"d{si}".encode()

    def test_midgate_snapshot_does_not_overwrite_the_train_one(self, tmp_path):
        """Stage 3 compares v0 vs v1 DIFFs; a shared path collides (Jul 23)."""
        cache = tmp_path / "tasks"
        _simulate_solver(cache, "T1", 0, answer=b"TRAIN")
        snapshot_output("T1", cache, 1, diff_text="train", sample_idx=0)
        _simulate_solver(cache, "T1", 0, answer=b"MIDGATE")
        snapshot_output("T1", cache, 1, diff_text="mid", sample_idx=0, phase="midgate")

        train = _snap_dir(cache, "T1", 1, 0)
        mid = cache / SNAPSHOTS_ROOT_NAME / "T1" / "iter_1_midgate_s0"
        assert train.exists() and mid.exists()
        assert (train / "workspace/output/1_T1_output.xlsx").read_bytes() == b"TRAIN"
        assert (mid / "workspace/output/1_T1_output.xlsx").read_bytes() == b"MIDGATE"

    def test_skills_symlink_is_never_followed(self, tmp_path):
        """`.claude/skills` points at the SHARED, evolving skills dir. Following
        it would copy skills into all 60 snapshots AND capture state that
        changes every iteration — destroying the immutability being bought."""
        cache = tmp_path / "tasks"
        shared = tmp_path / "shared_skills"
        (shared / "write-static-values").mkdir(parents=True)
        (shared / "write-static-values" / "SKILL.md").write_text("v1 body")

        _simulate_solver(cache, "T1", 0, answer=b"OUT", link_skills_to=shared)
        snapshot_output("T1", cache, 1, sample_idx=0)
        snap = _snap_dir(cache, "T1", 1, 0)

        got = _evolver_readable(snap)
        assert not any(".claude" in k for k in got), f"skills leaked: {list(got)}"
        assert not any("SKILL.md" in k for k in got)
        # And mutating the shared skills dir changes nothing under the snapshot.
        before = dict(got)
        (shared / "write-static-values" / "SKILL.md").write_text("v2 body")
        assert _evolver_readable(snap) == before

    def test_snapshot_files_have_their_own_inode(self, tmp_path):
        """Deliberately the OPPOSITE of the dataset mirror's contract. Sharing an
        inode with a live, rewritable file cannot yield an immutable snapshot —
        the first run of this suite proved it by mutating one."""
        cache = tmp_path / "tasks"
        ws = _simulate_solver(cache, "T1", 0, answer=b"OUT")
        snapshot_output("T1", cache, 1, sample_idx=0)
        src = ws / "spreadsheets" / "T1.xlsx"
        dst = _snap_dir(cache, "T1", 1, 0) / "workspace" / "spreadsheets" / "T1.xlsx"
        assert src.stat().st_ino != dst.stat().st_ino, (
            "snapshot shares an inode with the live workspace — a re-run will "
            "mutate it in place")
        assert dst.read_bytes() == src.read_bytes()

    def test_hardlink_does_not_make_the_snapshot_writable_through_the_live_dir(
            self, tmp_path):
        """A hard link shares the inode, so REPLACING the live file is safe
        (new inode) but TRUNCATING IT IN PLACE would mutate the snapshot. The
        harness always rewrites via write_bytes (open 'wb' → new content on the
        same inode), so this asserts the real risk, not a theoretical one."""
        cache = tmp_path / "tasks"
        ws = _simulate_solver(cache, "T1", 0, answer=b"OUT")
        snapshot_output("T1", cache, 1, sample_idx=0)
        dst = _snap_dir(cache, "T1", 1, 0) / "workspace" / "output" / "1_T1_output.xlsx"
        live = ws / "output" / "1_T1_output.xlsx"

        live.write_bytes(b"REWRITTEN-IN-PLACE")
        assert dst.read_bytes() == b"OUT", (
            "in-place rewrite of a hard-linked file mutated the snapshot — the "
            "snapshot must be copy-on-write against this, not just re-link")

    def test_empty_or_missing_workspace_is_not_an_error(self, tmp_path):
        """Timeout/no-output rollouts exist (55427 s0 tonight). Snapshotting
        must degrade, not raise, or one bad sample kills the batch."""
        cache = tmp_path / "tasks"
        (cache / task_ws_name("T1", 0)).mkdir(parents=True)
        snapshot_output("T1", cache, 1, diff_text="no output", sample_idx=0)
        snap = _snap_dir(cache, "T1", 1, 0)
        assert snap.exists()
        assert (snap / "DIFF.txt").read_text() == "no output"
        assert _snapshot_sample_workspace(cache / "does_not_exist", snap / "w2") == 0


# ==========================================================================
# 2 · TRAIN-ONLY ORACLE
# ==========================================================================


class TestTrainOnlyOracle:
    IDS = ("tr_a", "tr_b", "va_a", "te_a")
    TRAIN = frozenset({"tr_a", "tr_b"})

    def test_only_train_tasks_get_oracle_metadata(self, tmp_path):
        root = tmp_path / "ds"
        _make_dataset(root, self.IDS)
        meta = tmp_path / "oracle_tasks"
        n = write_all_task_meta(meta, root, allowed_task_ids=self.TRAIN)
        assert n == 2
        assert {p.stem for p in meta.glob("*.json")} == set(self.TRAIN)

    def test_no_val_or_test_golden_is_reachable_under_the_mount(self, tmp_path):
        """The decisive assertion: walk the whole mount as an evolver would."""
        root = tmp_path / "ds"
        _make_dataset(root, self.IDS)
        mirror = tmp_path / "mirror"
        n_tasks, n_files = stage_split_dataset(root, mirror, self.TRAIN)
        assert n_tasks == 2 and n_files > 0

        reachable = [p for p in mirror.rglob("*") if p.is_file()]
        blob = b"".join(p.read_bytes() for p in reachable)
        assert b"GOLDEN::va_a" not in blob, "a val golden is readable"
        assert b"GOLDEN::te_a" not in blob, "a test golden is readable"
        assert b"GOLDEN::tr_a" in blob, "train goldens must still work"
        assert {d.name for d in mirror.iterdir()} == set(self.TRAIN)

    def test_no_oracle_meta_points_outside_the_train_split(self, tmp_path):
        """Layer (a) and (b) must agree: every golden_path a meta names has to
        resolve inside the mirror. A mismatch means the evolver is handed a
        pointer that either dangles or escapes."""
        root = tmp_path / "ds"
        _make_dataset(root, self.IDS)
        meta = tmp_path / "oracle_tasks"
        write_all_task_meta(meta, root, allowed_task_ids=self.TRAIN,
                            container_data_root="/workspace/ssb_data")
        mirror = tmp_path / "mirror"
        stage_split_dataset(root, mirror, self.TRAIN)

        for j in meta.glob("*.json"):
            spec = json.loads(j.read_text())
            for gp in spec["golden_paths"]:
                assert gp.startswith("/workspace/ssb_data/"), gp
                rel = gp[len("/workspace/ssb_data/"):]
                assert (mirror / rel).is_file(), f"meta names unmirrored golden: {gp}"
                assert rel.split("/")[0] in self.TRAIN

    def test_default_still_writes_every_task(self, tmp_path):
        """allowed_task_ids=None must keep the old behaviour, so benchmarks that
        have not adopted the split control are not silently emptied."""
        root = tmp_path / "ds"
        _make_dataset(root, self.IDS)
        meta = tmp_path / "oracle_tasks"
        assert write_all_task_meta(meta, root) == len(self.IDS)

    def test_mirror_is_rebuilt_not_merged(self, tmp_path):
        """A stale mirror from a previous run must not survive into a new split
        — otherwise yesterday's val task is today's leak."""
        root = tmp_path / "ds"
        _make_dataset(root, self.IDS)
        mirror = tmp_path / "mirror"
        stage_split_dataset(root, mirror, frozenset({"tr_a", "va_a"}))
        assert (mirror / "va_a").exists()
        stage_split_dataset(root, mirror, self.TRAIN)
        assert not (mirror / "va_a").exists(), "stale val task survived a rebuild"
        assert {d.name for d in mirror.iterdir()} == set(self.TRAIN)

    def test_train_oracle_remains_fully_usable(self, tmp_path):
        """The control must not break what it protects: an evolver diagnosing a
        train task still needs its golden and its graded range."""
        root = tmp_path / "ds"
        _make_dataset(root, self.IDS)
        meta = tmp_path / "oracle_tasks"
        write_all_task_meta(meta, root, allowed_task_ids=self.TRAIN)
        spec = json.loads((meta / "tr_a.json").read_text())
        assert spec["answer_position"] == "A1"
        assert spec["num_test_cases"] == 1
        assert spec["expected_output_names"] == ["1_tr_a_output.xlsx"]


# ==========================================================================
# 3 · END-TO-END: a simulated iteration, then a simulated evolver
# ==========================================================================


def test_simulated_iteration_gives_the_evolver_static_train_only_evidence(tmp_path):
    """One full pass with no API calls: dataset → 3 solver samples → snapshots →
    a second iteration that re-rolls everything → an 'evolver' that reads only
    what it is handed and must find its original evidence unchanged, with no
    val/test golden anywhere in reach.
    """
    root, cache = tmp_path / "ds", tmp_path / "tasks"
    _make_dataset(root, ("tr_a", "va_a", "te_a"))
    train = frozenset({"tr_a"})

    mirror = tmp_path / "mirror"
    stage_split_dataset(root, mirror, train)
    meta = tmp_path / "oracle" / "tasks"
    write_all_task_meta(meta, root, allowed_task_ids=train)

    # iteration 1 — three samples, differing answers and scratch
    for si, ans in enumerate((b"WRONG", b"RIGHT", b"WRONG2")):
        _simulate_solver(cache, "tr_a", si, answer=ans, scratch=f"attempt {si}\n")
        snapshot_output("tr_a", cache, 1, diff_text=f"diff-{si}", sample_idx=si)
    frozen = {si: _evolver_readable(_snap_dir(cache, "tr_a", 1, si)) for si in range(3)}

    # iteration 2 — same task re-rolled; live dirs now hold different bytes
    for si in range(3):
        _simulate_solver(cache, "tr_a", si, answer=b"ITER2", scratch="iter2\n")
        snapshot_output("tr_a", cache, 2, diff_text=f"i2-{si}", sample_idx=si)

    # the "evolver" re-reads iteration 1's evidence
    for si in range(3):
        assert _evolver_readable(_snap_dir(cache, "tr_a", 1, si)) == frozen[si], (
            f"iteration 1 sample {si} was mutated by iteration 2")
    assert frozen[1]["workspace/output/1_tr_a_output.xlsx"] == b"RIGHT"

    # and everything reachable — snapshots + mount + metas — is train-only
    reach = [p for p in (list(cache.rglob("*")) + list(mirror.rglob("*"))
                         + list(meta.rglob("*"))) if p.is_file()]
    blob = b"".join(p.read_bytes() for p in reach)
    assert b"GOLDEN::va_a" not in blob and b"GOLDEN::te_a" not in blob
    assert not (meta / "va_a.json").exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ==========================================================================
# 5 · THE CENTRAL GUARD — unbypassable, because every container goes through it
# ==========================================================================


class TestDatasetMountGuard:
    """11 SSB launchers each build their own EXTRA_MOUNTS string. Enforcement
    lives at the single point where containers are created, so no launcher can
    opt out and no future edit can quietly re-widen the mount.
    """

    @staticmethod
    def _guard():
        from src.harness.direct.executor import _assert_dataset_mount_is_train_only
        return _assert_dataset_mount_is_train_only

    def _ds(self, tmp_path, ids):
        root = tmp_path / "ds"
        for t in ids:
            (root / t).mkdir(parents=True)
        return root

    def test_raises_when_the_mount_contains_a_val_task(self, tmp_path, monkeypatch):
        root = self._ds(tmp_path, ["tr_a", "tr_b", "va_a"])
        monkeypatch.setenv("EVOSKILL_TRAIN_TASK_IDS", "tr_a,tr_b")
        with pytest.raises(RuntimeError, match="CONTAMINATION"):
            self._guard()(f"{root}:/workspace/ssb_data:ro")

    def test_passes_for_a_train_only_mirror(self, tmp_path, monkeypatch):
        root = self._ds(tmp_path, ["tr_a", "tr_b"])
        monkeypatch.setenv("EVOSKILL_TRAIN_TASK_IDS", "tr_a,tr_b")
        self._guard()(f"{root}:/workspace/ssb_data:ro")   # must not raise

    def test_a_mirror_missing_a_train_task_is_still_allowed(self, tmp_path, monkeypatch):
        """The guard forbids EXTRA tasks, not missing ones — an incomplete mirror
        is a different failure, caught by the launcher's own count assertion."""
        root = self._ds(tmp_path, ["tr_a"])
        monkeypatch.setenv("EVOSKILL_TRAIN_TASK_IDS", "tr_a,tr_b")
        self._guard()(f"{root}:/workspace/ssb_data:ro")

    def test_unscoped_launcher_warns_but_does_not_block(self, tmp_path, monkeypatch, caplog):
        """10 launchers are still unconverted. They must be VISIBLE, not silently
        trusted — and not broken either, or the fix becomes un-shippable."""
        import logging
        root = self._ds(tmp_path, ["tr_a", "va_a", "te_a"])
        monkeypatch.delenv("EVOSKILL_TRAIN_TASK_IDS", raising=False)
        with caplog.at_level(logging.WARNING):
            self._guard()(f"{root}:/workspace/ssb_data:ro")
        assert any("CONTAMINATION" in r.message for r in caplog.records), caplog.text

    def test_non_dataset_mounts_are_ignored(self, tmp_path, monkeypatch):
        """solver_outputs carries the snapshots and must not be split-gated."""
        root = self._ds(tmp_path, ["anything", "at_all"])
        monkeypatch.setenv("EVOSKILL_TRAIN_TASK_IDS", "tr_a")
        self._guard()(f"{root}:/workspace/solver_outputs")   # must not raise

    def test_hidden_dirs_do_not_trip_the_guard(self, tmp_path, monkeypatch):
        root = self._ds(tmp_path, ["tr_a"])
        (root / ".DS_Store_dir").mkdir()
        monkeypatch.setenv("EVOSKILL_TRAIN_TASK_IDS", "tr_a")
        self._guard()(f"{root}:/workspace/ssb_data:ro")


def test_both_ssb_profiles_share_one_implementation():
    """v1 and v2 must not fork the mechanism — a fix to one has to be a fix to
    both, which is only true while they import the same helpers."""
    from src.agent_profiles.spreadsheetbench_agent import task_workspace as v1
    from src.agent_profiles.ssbv2_agent import task_workspace as v2
    import inspect

    assert v2._snapshot_sample_workspace is v1._snapshot_sample_workspace
    assert v2.stage_split_dataset is v1.stage_split_dataset
    for mod in (v1, v2):
        assert "allowed_task_ids" in inspect.signature(mod.write_all_task_meta).parameters
        assert "allowed_task_ids" in inspect.signature(mod.build_oracle_spec).parameters
