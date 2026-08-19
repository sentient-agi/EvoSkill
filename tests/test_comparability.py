"""Laws for the comparison gate (L1).

The gate's whole value is that it REFUSES. A gate that cannot fail is #154 wearing a
new name, so most of what follows drives it to refuse and asserts it did, and asserts
WHICH field it named — a refusal that does not say what differed cannot be acted on.

WHAT THESE TESTS DO NOT COVER (stated, not implied — see the module docstring's third
bucket): they do not check the served provider (#169) or scorer bytes, because no
artifact records either. `test_unverifiable_is_never_empty` pins that gap open so a
future edit cannot quietly drop the disclosure and leave a pass looking complete.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.orchestrator.comparability import (
    PROTECTED,
    Comparison,
    RunsNotComparable,
    SpecUnavailable,
    VacuousComparison,
    compare_runs,
    load_spec,
)
from src.orchestrator.spec import ExperimentSpec

BASE = {
    "name": "arm_a",
    "benchmark": "tau3",
    "domain": "banking_knowledge",
    "variant": "bm25_grep",
    "solver_model": "deepseek/deepseek-v4-flash-0731",
    "solver_provider_order": ["deepseek"],
    "solver_allow_fallbacks": False,
    "user_sim_model": "openai/gpt-5-nano",
    "judge_model": "openai/gpt-4.1",
    "agent_clean_s": 1200,
    "agent_turns": 100,
    "episode_timeout_s": 5400,
    "solver_harness": "direct",
    "split_sha256": "abc123",
    "test_sealed": True,
    "n_samples": 3,
    "evolver_model": "claude-opus-5",
    "n_epochs": 4,
}


def _run(tmp_path: Path, name: str, **over) -> Path:
    """A run directory carrying a SPEC.json. `over` may set a field to `...` to DELETE it."""
    spec = dict(BASE, name=name)
    for k, v in over.items():
        if v is ...:
            spec.pop(k, None)
        else:
            spec[k] = v
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SPEC.json").write_text(
        json.dumps({"cache_tag": f"{name}_deadbeef", "config_sha256": "0" * 64, "spec": spec})
    )
    return d


# ── the gate licenses what it should ─────────────────────────────────────────────────

def test_identical_specs_are_comparable(tmp_path):
    a, b = _run(tmp_path, "a"), _run(tmp_path, "b")
    c = compare_runs(a, b)
    assert isinstance(c, Comparison)
    # every protected field was actually looked at, not skipped
    assert set(c.matched) == set(PROTECTED)
    assert c.reported == {}  # only `name` differs, and name is not a spec-diff we report
    assert "COMPARABLE" in c.render()


def test_list_and_tuple_provider_order_compare_equal(tmp_path):
    a = _run(tmp_path, "a", solver_provider_order=["deepseek"])
    b = _run(tmp_path, "b", solver_provider_order=["deepseek"])
    assert compare_runs(a, b).matched["solver_provider_order"] == ["deepseek"]


def test_non_protected_difference_is_reported_not_refused(tmp_path):
    a = _run(tmp_path, "a", n_epochs=4)
    b = _run(tmp_path, "b", n_epochs=8)
    c = compare_runs(a, b)
    assert c.reported["n_epochs"] == (4, 8)


# ── the gate refuses what it should — one test per real defect class ─────────────────

def test_differing_solver_model_is_refused(tmp_path):
    a = _run(tmp_path, "a")
    b = _run(tmp_path, "b", solver_model="z-ai/glm-5.2")
    with pytest.raises(RunsNotComparable) as e:
        compare_runs(a, b)
    assert "solver_model" in str(e.value)


def test_differing_provider_order_is_refused(tmp_path):
    """#129: the pin that landed on the non-executing dict would pass a naive diff."""
    a = _run(tmp_path, "a", solver_provider_order=["deepseek"])
    b = _run(tmp_path, "b", solver_provider_order=["novita"])
    with pytest.raises(RunsNotComparable) as e:
        compare_runs(a, b)
    assert "solver_provider_order" in str(e.value)


def test_differing_allow_fallbacks_is_refused(tmp_path):
    a = _run(tmp_path, "a", solver_allow_fallbacks=False)
    b = _run(tmp_path, "b", solver_allow_fallbacks=True)
    with pytest.raises(RunsNotComparable):
        compare_runs(a, b)


def test_differing_split_is_refused(tmp_path):
    a, b = _run(tmp_path, "a", split_sha256="aaa"), _run(tmp_path, "b", split_sha256="bbb")
    with pytest.raises(RunsNotComparable) as e:
        compare_runs(a, b)
    assert "split_sha256" in str(e.value)


def test_sealed_vs_unsealed_is_refused(tmp_path):
    a, b = _run(tmp_path, "a", test_sealed=True), _run(tmp_path, "b", test_sealed=False)
    with pytest.raises(RunsNotComparable):
        compare_runs(a, b)


def test_differing_n_samples_is_refused(tmp_path):
    """Power differs, so the discordant-pair gate is not measuring the same thing."""
    a, b = _run(tmp_path, "a", n_samples=3), _run(tmp_path, "b", n_samples=5)
    with pytest.raises(RunsNotComparable):
        compare_runs(a, b)


# ── THE property: absence is not a match ─────────────────────────────────────────────

def test_field_absent_from_both_specs_refuses(tmp_path):
    """The vacuous-guard case. A naive dict diff calls this 'equal'; it is 'blind'.

    This is the single most important test in the file: it is the failure mode that
    #154/#227/#228/#248 all shared — a check reporting success while observing nothing.
    """
    a = _run(tmp_path, "a", solver_model=...)
    b = _run(tmp_path, "b", solver_model=...)
    with pytest.raises(RunsNotComparable) as e:
        compare_runs(a, b)
    msg = str(e.value)
    assert "NOT RECORDED" in msg and "solver_model" in msg


def test_field_absent_from_one_side_refuses(tmp_path):
    a = _run(tmp_path, "a")
    b = _run(tmp_path, "b", judge_model=...)
    with pytest.raises(RunsNotComparable) as e:
        compare_runs(a, b)
    assert "judge_model" in str(e.value)


def test_allow_unrecorded_passes_but_discloses(tmp_path):
    """The escape hatch must leave a trace in the receipt, not a silent pass."""
    a = _run(tmp_path, "a", solver_model=...)
    b = _run(tmp_path, "b", solver_model=...)
    c = compare_runs(a, b, allow_unrecorded=True)
    names = {n for n, _ in c.unverifiable}
    assert "unrecorded_protected_fields" in names
    assert "solver_model" in c.render()


# ── treatment ────────────────────────────────────────────────────────────────────────

def test_treatment_exempts_a_protected_field_and_records_the_exemption(tmp_path):
    a = _run(tmp_path, "a", solver_model="deepseek/deepseek-v4-flash-0731")
    b = _run(tmp_path, "b", solver_model="z-ai/glm-5.2")
    c = compare_runs(a, b, treatment=["solver_model"])
    assert c.treatment["solver_model"] == (
        "deepseek/deepseek-v4-flash-0731", "z-ai/glm-5.2")
    assert "solver_model" in c.exempted
    assert "exempted by caller declaration" in c.render()


def test_declared_treatment_that_did_not_vary_is_vacuous(tmp_path):
    """#120 at the claim layer: nothing manipulated means no declared cause."""
    a, b = _run(tmp_path, "a"), _run(tmp_path, "b")
    with pytest.raises(VacuousComparison) as e:
        compare_runs(a, b, treatment=["solver_model"])
    assert "did not vary" in str(e.value)


def test_treatment_typo_is_refused(tmp_path):
    """A misspelled treatment would silently exempt nothing and re-protect the field."""
    a, b = _run(tmp_path, "a"), _run(tmp_path, "b", solver_model="z-ai/glm-5.2")
    with pytest.raises(RunsNotComparable) as e:
        compare_runs(a, b, treatment=["solver_modle"])
    assert "solver_modle" in str(e.value)


def test_treatment_works_on_a_reported_field(tmp_path):
    """Arm A (#190): evolver differs by design; assert it actually did."""
    a = _run(tmp_path, "a", evolver_model="claude-opus-5")
    b = _run(tmp_path, "b", evolver_model="claude-sonnet-5")
    c = compare_runs(a, b, treatment=["evolver_model"])
    assert c.treatment["evolver_model"] == ("claude-opus-5", "claude-sonnet-5")
    assert "evolver_model" not in c.exempted  # not protected, so not an exemption


# ── missing / malformed input must not read as agreement ─────────────────────────────

def test_missing_spec_file_raises(tmp_path):
    a = _run(tmp_path, "a")
    (tmp_path / "b").mkdir()
    with pytest.raises(SpecUnavailable):
        compare_runs(a, tmp_path / "b")


def test_empty_spec_block_raises_rather_than_comparing_nothing(tmp_path):
    a = _run(tmp_path, "a")
    d = tmp_path / "empty"
    d.mkdir()
    (d / "SPEC.json").write_text(json.dumps({"spec": {}}))
    with pytest.raises(SpecUnavailable):
        compare_runs(a, d)


def test_unreadable_spec_raises(tmp_path):
    a = _run(tmp_path, "a")
    d = tmp_path / "bad"
    d.mkdir()
    (d / "SPEC.json").write_text("{not json")
    with pytest.raises(SpecUnavailable):
        load_spec(d)


# ── laws about the gate itself ───────────────────────────────────────────────────────

def test_protected_covers_every_instrument_field(tmp_path):
    """Single source of truth. If _INSTRUMENT grows, PROTECTED grows with it.

    Without this, someone argues a new field into _INSTRUMENT for cache correctness and
    the claim layer silently keeps ignoring it — the exact split that let #129 through.
    """
    assert set(ExperimentSpec._INSTRUMENT) <= set(PROTECTED)


def test_unverifiable_is_never_empty(tmp_path):
    """Silence must never imply verification.

    A clean comparison still has to say what it did not check. If someone removes the
    served-provider disclosure, this fails and they have to think about it.
    """
    a, b = _run(tmp_path, "a"), _run(tmp_path, "b")
    c = compare_runs(a, b)
    names = {n for n, _ in c.unverifiable}
    assert "served_provider" in names
    assert all(why.strip() for _, why in c.unverifiable)
    assert "NOT VERIFIED" in c.render()


@pytest.mark.parametrize("f", PROTECTED)
def test_every_protected_field_can_actually_refuse(tmp_path, f):
    """The meta-test: each protected field must be able to FAIL the gate on its own.

    A field listed as protected but never compared would be a guard that cannot fail —
    exactly what #154 was. Parametrising over PROTECTED means a newly added field is
    covered the moment it is added, with no second list to remember.
    """
    original = BASE.get(f)
    mutated = (not original) if isinstance(original, bool) else \
              (original + 1) if isinstance(original, int) else \
              (list(original) + ["MUTATED"]) if isinstance(original, list) else \
              f"{original}-MUTATED"
    a = _run(tmp_path, "a")
    b = _run(tmp_path, "b", **{f: mutated})
    with pytest.raises(RunsNotComparable) as e:
        compare_runs(a, b)
    assert f in str(e.value), f"refusal message must name the offending field {f!r}"
