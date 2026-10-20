"""The evidence surface must be COMPLETE, not just train-only.

A digest surface can pass a membership check while half of its samples carry no scorer diff (a
builder that skips missing snapshots in silence). src/orchestrator/digest_coverage.py holds the rules
the loop's `_v_digests_complete` step applies to the digest BYTES: every train task of the split,
every one of n samples, each with its grade evidence. The module is loaded BY PATH
(EVOSKILL_TEST_COVERAGE_MODULE) so a mutated copy can be shown to turn these tests red.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_MOD = Path(os.environ.get("EVOSKILL_TEST_COVERAGE_MODULE",
                           ROOT / "src" / "orchestrator" / "digest_coverage.py"))
_spec = importlib.util.spec_from_file_location("digest_coverage_under_test", _MOD)
cov = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cov)


# ── synthetic surfaces, in each builder's exact heading format ─────────────────────────────────
def ssb_digest(tid: str, samples=(0, 1, 2), diffs=(0, 1, 2), *, extra_pre="",
               in_rollout="") -> str:
    L = [f"# {tid}  (Debugging)", "", "**Result: 0/3 samples passed.**  turns=[7]", ""]
    for k in diffs:
        L += [f"## Scorer diff — iter_0_s{k} (authoritative)", "", "```", "0/1 match", "```", ""]
    L += [extra_pre] if extra_pre else []
    for k in samples:
        L += [f"## Rollout s{k} — FAIL (7 turns)", "", "### assistant", "", "did it", ""]
        if in_rollout:
            L += ["### tool", "", in_rollout, ""]
    return "\n".join(L)


def t3_digest(tid: str, verdicts=("ok", "ok", "ok"), *, transcript="") -> str:
    """verdicts: 'ok' a graded rollout, 'none' the no-verdict marker, 'ungraded' an empty basis
    (a timeout: a real outcome, not a gap), 'nograde' no grade heading at all."""
    L = [f"# {tid}  (banking, variant x)", "", "**Result: 0/3 samples passed.**", ""]
    for i, v in enumerate(verdicts):
        L += [f"## Rollout s{i} — FAIL [kind=fail] (score=0.0, 9 turns, 1.0s)", ""]
        if v != "nograde":
            L += ["### Grade (authoritative — this is what produced the score)", ""]
        if v == "none":
            L += ["_No structured verdict on this trace — the episode produced no evaluator "
                  "output at all._", ""]
        elif v == "ungraded":
            L += ["- reward **0.0**  = PRODUCT of [] -> {}", "", "**NOT GRADED — no check ran.**"]
        elif v == "ok":
            L += ["- reward **0.0**  = PRODUCT of ['DB'] -> {}", ""]
        L += ["### Transcript", "", transcript or "### Turn 1", ""]
    return "\n".join(L)


def surface(d: Path, stems, render) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for s in stems:
        (d / f"{s}.md").write_text(render(s))
    return d


STEMS = ["Debugging_01_01", "Debugging_01_02"]


# ── the rules (module under test) ──────────────────────────────────────────────────────────────
def test_ssb_complete_surface_passes(tmp_path):
    d = surface(tmp_path / "e1", STEMS, ssb_digest)
    (d / "COVERAGE.json").write_text("{}")                     # sidecars are not task files
    (d / "REGRESSIONS.md").write_text("## Rollout s0 — FAIL (1 turns)")
    ok, detail = cov.check([d], STEMS, 3, "ssbv2")
    assert ok, detail
    assert "2/2 train task files, 6/6 samples complete" in detail


def test_ssb_missing_diff_fails(tmp_path):
    d = surface(tmp_path / "e1", STEMS, ssb_digest)
    (d / "Debugging_01_02.md").write_text(ssb_digest("Debugging/01_02", diffs=(0, 2)))
    ok, detail = cov.check([d], STEMS, 3, "ssbv2")
    assert not ok
    assert "1 of 6 samples" in detail and "Debugging_01_02 s1 no scorer diff" in detail
    assert "digest_snapshot_roots" in detail                   # the HALT names the remedy


def test_ssb_builder_marker_is_a_gap_not_a_diff(tmp_path):
    marker = ("## NO scorer diff — s1 (no workbook snapshot recorded)\n\n"
              "[No scorer diff for this sample.]\n")
    d = surface(tmp_path / "e1", STEMS,
                lambda s: ssb_digest(s, diffs=(0, 2), extra_pre=marker))
    ok, detail = cov.check([d], STEMS, 3, "ssbv2")
    assert not ok
    assert "s1 no scorer diff (builder marker)" in detail


def test_ssb_n_minus_one_samples_fails(tmp_path):
    d = surface(tmp_path / "e1", STEMS, ssb_digest)
    (d / "Debugging_01_01.md").write_text(ssb_digest("x", samples=(0, 1)))
    ok, detail = cov.check([d], STEMS, 3, "ssbv2")
    assert not ok and "Debugging_01_01 s2 no rollout" in detail


def test_ssb_diff_heading_inside_a_rollout_does_not_count(tmp_path):
    """Rollout sections paste tool output verbatim, so the heading can occur there."""
    pasted = "## Scorer diff — iter_0_s1 (authoritative)"
    d = surface(tmp_path / "e1", STEMS,
                lambda s: ssb_digest(s, diffs=(0, 2), in_rollout=pasted))
    ok, detail = cov.check([d], STEMS, 3, "ssbv2")
    assert not ok and "s1 no scorer diff" in detail


def test_ssb_two_diffs_for_one_sample_fails(tmp_path):
    two = "## Scorer diff — iter_0_midgate_s0 (authoritative)\n"
    d = surface(tmp_path / "e1", STEMS, lambda s: ssb_digest(s, extra_pre=two))
    ok, detail = cov.check([d], STEMS, 3, "ssbv2")
    assert not ok and "ambiguous scorer diff" in detail


def test_ssb_missing_task_file_fails(tmp_path):
    d = surface(tmp_path / "e1", STEMS[:1], ssb_digest)
    ok, detail = cov.check([d], STEMS, 3, "ssbv2")
    assert not ok and "Debugging_01_02 s0 no digest file" in detail
    assert "3 of 6 samples" in detail


def test_tau3_complete_surface_passes_and_ungraded_is_not_a_gap(tmp_path):
    stems = ["task_001", "task_002"]
    d = surface(tmp_path / "e1", stems, lambda s: t3_digest(s, ("ok", "ungraded", "ok")))
    ok, detail = cov.check([d], stems, 3, "tau3")
    assert ok, detail


def test_tau3_no_verdict_fails(tmp_path):
    stems = ["task_001", "task_002"]
    d = surface(tmp_path / "e1", stems, t3_digest)
    (d / "task_002.md").write_text(t3_digest("task_002", ("ok", "ok", "none")))
    ok, detail = cov.check([d], stems, 3, "tau3")
    assert not ok and "task_002 s2 no structured verdict" in detail


def test_tau3_grade_heading_inside_transcript_does_not_count(tmp_path):
    stems = ["task_001"]
    pasted = "### Grade (authoritative — this is what produced the score)"
    d = surface(tmp_path / "e1", stems,
                lambda s: t3_digest(s, ("ok", "nograde", "ok"), transcript=pasted))
    ok, detail = cov.check([d], stems, 3, "tau3")
    assert not ok and "task_001 s1 no grade block" in detail


def test_every_dir_in_the_list_is_checked(tmp_path):
    good = surface(tmp_path / "digests_e1_train", STEMS, ssb_digest)
    good2 = surface(tmp_path / "digests_e2_train", STEMS, ssb_digest)
    ok, detail = cov.check([good, good2], STEMS, 3, "ssbv2")
    assert ok, detail
    assert "digests_e1_train" in detail and "digests_e2_train" in detail
    bad = surface(tmp_path / "digests_e3_train", STEMS, lambda s: ssb_digest(s, diffs=()))
    ok, detail = cov.check([good, bad, good2], STEMS, 3, "ssbv2")
    assert not ok and "EVIDENCE GAPS in digests_e3_train" in detail
    ok, detail = cov.check([good, tmp_path / "digests_e4_train"], STEMS, 3, "ssbv2")
    assert not ok and "no digest dir at" in detail


def test_nothing_to_check_is_never_a_pass(tmp_path):
    d = surface(tmp_path / "e1", STEMS, ssb_digest)
    assert not cov.check([], STEMS, 3, "ssbv2")[0]
    assert not cov.check([d], [], 3, "ssbv2")[0]
    assert not cov.check([d], STEMS, 0, "ssbv2")[0]


def test_unknown_benchmark_is_refused(tmp_path):
    d = surface(tmp_path / "e1", STEMS, ssb_digest)
    ok, detail = cov.check([d], STEMS, 3, "arc_agi")
    assert not ok and "no coverage rule for benchmark 'arc_agi'" in detail


def test_train_stems_maps_ids_to_filenames(tmp_path):
    p = tmp_path / "split.json"
    p.write_text(json.dumps({"train": ["Debugging/01_02", "Financial_Model/03_04"],
                             "val": ["Debugging/09_09"]}))
    assert cov.train_stems(p) == ["Debugging_01_02", "Financial_Model_03_04"]
    p.write_text(json.dumps({"train": [{"id": "task_001"}], "val": []}))
    assert cov.train_stems(p) == ["task_001"]


# ── the loop ───────────────────────────────────────────────────────────────────────────────────
import src.orchestrator.loop as L  # noqa: E402  (a failed import must fail, never skip)


def test_step_sits_between_train_only_and_the_ao():
    names = [s.name for s in L.epoch_steps()]
    i = names.index("evidence surface is complete")
    assert names[i - 1] == "evidence surface is train-only"
    assert i < names.index("evolver prompt authored by the AO")
    step = L.epoch_steps()[i]
    assert step.run is None, "no runner: the remedy is upstream and the HALT names it"


