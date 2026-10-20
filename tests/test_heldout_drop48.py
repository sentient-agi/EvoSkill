"""Offline tests for the run's DECLARED split in the loop (a split may carry a "dropped" key: tasks whose own
reference fails self-grading):

  * every leg the loop reads a number from (the prompt figures' source, the render's base/batch legs, the
    pairing) must be ON the run's declared split, compared by TASK ID — SSB v2 legs key per_task by index, so a
    per_task key-set comparison cannot tell two equal-size task sets apart;
  * the prompt leakage check also refuses the split's dropped / unused / excluded ids;
  * the held-out program (`--heldout`): the rule's epoch, the baseline on the split's test key (a measured leg or
    a restricted record re-derived from its source), the skill leg on the same key, and HELDOUT_COMPARISON.json on
    ONE task set;
  * scripts/daytona/ssb_leg_daytona.py plans only on a named split and bundles THAT split file.

No sandbox, no model, no network; everything is written under tmp_path.
"""
from __future__ import annotations

import hashlib
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "daytona"))

import src.orchestrator.loop as L  # noqa: E402

NEW_SPLIT = REPO / "examples/ssbv2/split.json"

ONE, ZERO, ALL = [1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]


def _split(p: Path, **kw) -> Path:
    d = {"train": ["A/01", "A/02", "A/03"], "val": ["A/04", "A/05"], "test": ["B/01", "B/02", "C/01"],
         "unused": ["A/09"], "dropped": {"task_ids": ["B/09", "C/09"], "reason": "reference fails"},
         "excluded_prior_run": {"task_ids": ["C/08"], "reason": "prior run"}}
    d.update(kw)
    p.write_text(json.dumps(d))
    return p


def _leg(p: Path, tasks, scores, **extra) -> Path:
    d = {"task_ids": list(tasks), "per_task": {str(i): s for i, s in enumerate(scores)}, "n_samples": 3,
         "model": "m", "provider_order": ["p"], "skills_dir": None, "n": len(tasks),
         "sample_pass": sum(1 for s in scores for x in s if x >= 0.999), "sample_total": 3 * len(tasks),
         "turn_exhausted": 0}
    d.update(extra)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d))
    return p


def _spec(split: Path, **kw):
    base = dict(benchmark="ssbv2", split_path=str(split), n_samples=3, n_epochs=3, name="r",
                solver_model="m", solver_provider_order=("p",))
    base.update(kw)
    return SimpleNamespace(**base)


# ══ the split, read from the file ════════════════════════════════════════════════════════════════
def test_split_key_ids_reads_every_shape(tmp_path):
    sp = json.loads(_split(tmp_path / "s.json").read_text())
    assert L._split_key_ids(sp, "test") == ["B/01", "B/02", "C/01"]
    assert L._split_key_ids(sp, "dropped") == ["B/09", "C/09"]
    assert L._split_key_ids(sp, "excluded_prior_run") == ["C/08"]
    assert L._split_key_ids({"train": [{"id": "x"}]}, "train") == ["x"]
    assert L._split_key_ids(sp, "nope") == []


def test_leg_on_split_compares_task_ids_not_sizes(tmp_path):
    sp = _split(tmp_path / "s.json")
    ok = _leg(tmp_path / "ok.json", ["A/03", "A/01", "A/02"], [ONE, ZERO, ZERO])
    assert L._leg_on_split(ok, sp, "train", 3)[0] == "ok"
    same_size = _leg(tmp_path / "ss.json", ["A/01", "A/02", "A/04"], [ONE, ZERO, ZERO])
    st, d = L._leg_on_split(same_size, sp, "train", 3)
    assert st == "bad" and "A/04" in d and "A/03" in d
    assert L._leg_on_split(ok, sp, "train", 1)[0] == "bad"                 # N differs from the spec
    short = _leg(tmp_path / "sh.json", ["A/01", "A/02", "A/03"], [ONE, [0.0, 1.0], ZERO])
    assert L._leg_on_split(short, sp, "train", 3)[0] == "bad"              # a task without N scores
    broken = _leg(tmp_path / "br.json", ["A/01", "A/02", "A/03"], [ONE, ZERO])   # 3 ids, 2 per_task rows
    assert L._leg_on_split(broken, sp, "train", 3)[0] == "bad"
    tau3 = tmp_path / "t3.json"
    tau3.write_text(json.dumps({"n_samples": 3, "per_task": {"task_1": {}}}))
    assert L._leg_on_split(tau3, sp, "train", 3)[0] == "unknown"           # no task_ids: not judged
    assert L._leg_on_split(ok, "", "train", 3)[0] == "unknown"             # no split declared


def test_pairing_catches_equal_size_legs_on_different_tasks(tmp_path):
    sp = _split(tmp_path / "s.json")
    rd = tmp_path / "r"
    rd.mkdir()
    base = _leg(rd / "base_train.json", ["A/01", "A/02", "A/03"], [ONE, ZERO, ZERO])
    ctx = L.Ctx(rd, 1, _spec(sp), rd / "oracle", base_leg=str(base))
    _leg(ctx.paths.leg("train"), ["A/01", "A/02", "A/04"], [ONE, ONE, ZERO])
    from src.orchestrator.invariants import check_legs_paired_comparably
    assert check_legs_paired_comparably(base, ctx.paths.leg("train"), 3).status == "PASS", \
        "the index-keyed per_task hides the difference from the shared invariant"
    ok, d = L._v_legs_comparable(ctx, "train")
    assert not ok and "A/04" in d
    _leg(ctx.paths.leg("train"), ["A/03", "A/02", "A/01"], [ONE, ONE, ZERO])
    assert L._v_legs_comparable(ctx, "train")[0]
    # a base left on the OLD split (4 train tasks) against a skill leg on the new one
    _leg(base, ["A/01", "A/02", "A/03", "A/07"], [ONE, ZERO, ZERO, ZERO])
    ok, d = L._v_legs_comparable(ctx, "train")
    assert not ok


def test_prompt_numbers_refuse_a_base_leg_off_the_split(tmp_path):
    sp = _split(tmp_path / "s.json")
    rd = tmp_path / "r"
    rd.mkdir()
    base = _leg(rd / "base_train.json", ["A/01", "A/02", "A/03", "A/07"], [ONE, ZERO, ZERO, ONE])
    ctx = L.Ctx(rd, 1, _spec(sp), rd / "oracle", base_leg=str(base))
    ctx.paths.prompt.write_text("The solver passed **4 of 12 episodes**.\n")
    ok, d = L._v_prompt_numbers(ctx)
    assert not ok and "not on the run's split" in d
    _leg(base, ["A/01", "A/02", "A/03"], [ONE, ZERO, ZERO])                 # 2/9 on the split
    ok, d = L._v_prompt_numbers(ctx)
    assert not ok and "no /9 episode fraction" in d, d                      # 4/12 is not on /9
    ctx.paths.prompt.write_text("The solver passed **2 of 9 episodes**.\n")
    ok, d = L._v_prompt_numbers(ctx)
    assert ok and "2/9" in d, d


def test_prompt_numbers_on_171_sample_legs(tmp_path):
    """The prompt-figures check on 171-sample legs: 57 train tasks x 3."""
    tasks = [f"A/{i:02d}" for i in range(57)]
    sp = _split(tmp_path / "s.json", train=tasks)
    rd = tmp_path / "r"
    rd.mkdir()
    scores = [ALL] * 19 + [[1.0, 0.0, 0.0]] + [ZERO] * 37                   # 58 passes
    base = _leg(rd / "base_train.json", tasks, scores)
    assert json.loads(base.read_text())["sample_pass"] == 58
    ctx = L.Ctx(rd, 1, _spec(sp), rd / "oracle", base_leg=str(base))
    ctx.paths.prompt.write_text("Base: 58 of 171 episodes passed.\n")
    ok, d = L._v_prompt_numbers(ctx)
    assert ok and "58/171" in d, d
    ctx.paths.prompt.write_text("Base: 58 of 177 episodes passed.\n")
    assert not L._v_prompt_numbers(ctx)[0]


def test_no_leakage_covers_dropped_unused_and_excluded_ids(tmp_path):
    sp = _split(tmp_path / "s.json")
    rd = tmp_path / "r"
    rd.mkdir()
    ctx = L.Ctx(rd, 1, _spec(sp), rd / "oracle")
    for tid in ("B/09", "A/09", "C/08"):
        ctx.paths.prompt.write_text(f"Look at {tid}.\n")
        ok, d = L._v_prompt_no_leakage(ctx)
        assert not ok and tid in d, (tid, d)
    ctx.paths.prompt.write_text("Scored 2/9 on train.\n")
    ok, d = L._v_prompt_no_leakage(ctx)
    assert ok and "0 of 4 dropped/unused/excluded" in d, d


# ══ the restricted held-out baseline ═════════════════════════════════════════════════════════════
def test_restriction_is_exact_and_refuses_what_it_cannot_restate(tmp_path):
    sp = _split(tmp_path / "s.json")
    src = _leg(tmp_path / "full.json", ["B/01", "B/09", "B/02", "C/01"], [ONE, ZERO, ZERO, ALL],
               episode_status={"counts": {"pass": 5, "graded_fail": 7}, "no_output_file": ["B/09 s1", "B/02 s0"],
                               "source": "x"},
               served_providers={"P": 99}, provider_violations=[], turn_exhausted=3, tag="tb")
    rec = L.derive_restricted_leg(src, sp, "test")
    assert rec["task_ids"] == ["B/01", "B/02", "C/01"]                       # the split's order
    assert (rec["n"], rec["sample_pass"], rec["sample_total"], rec["pass3"], rec["all3"]) == (3, 5, 9, 2, 1)
    assert rec["episode_status"]["no_output_file"] == ["B/02 s0"]
    assert rec["episode_status"]["counts"] == {"graded_fail": 4, "pass": 5}
    assert "served_providers" not in rec and "served_providers" in rec["derived_from"]["source_fields_not_restated"]
    assert "turn_exhausted" not in rec and rec["derived_from"]["turn_exhausted"].startswith("ABSENT")
    assert rec["derived_from"]["removed_task_ids"] == ["B/09"]
    assert rec["derived_from"]["removed_sample_passes"] == 0
    assert rec["provider_violations"] == [] and rec["split_key"] == "test"
    rp = tmp_path / "rec.json"
    rp.write_text(json.dumps(rec))
    assert L._restricted_leg_problems(json.loads(rp.read_text())) == []
    # a source whose recorded figures do not re-derive is refused
    bad = json.loads(src.read_text())
    bad["sample_pass"] = 99
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="does not re-derive"):
        L.derive_restricted_leg(tmp_path / "bad.json", sp, "test")
    # a restriction cannot add tasks
    small = _leg(tmp_path / "small.json", ["B/01", "B/02"], [ONE, ZERO])
    with pytest.raises(ValueError, match="cannot add tasks"):
        L.derive_restricted_leg(small, sp, "test")
    # the source changed after the derivation
    src.write_text(src.read_text().replace('"tb"', '"tb2"'))
    assert any("no longer hashes" in x for x in L._restricted_leg_problems(rec))


# ══ the held-out program ═════════════════════════════════════════════════════════════════════════
def _held_run(tmp_path, epochs=(1, 2), val_for=(1, 2)):
    sp = _split(tmp_path / "s.json")
    rd = tmp_path / "r"
    rd.mkdir()
    for e in epochs:
        ep = L.EpochPaths(rd, e)
        for d in (ep.staging, ep.promoted):
            (d / "k").mkdir(parents=True, exist_ok=True)
            (d / "k" / "SKILL.md").write_text(f"---\nname: k\ndescription: v{e}\n---\nbody\n")
        _leg(ep.leg("train"), ["A/01", "A/02", "A/03"], [ONE, ZERO, ZERO], skills_dir=str(ep.promoted))
        if e in val_for:
            _leg(ep.leg("val"), ["A/04", "A/05"], [ONE, ZERO], skills_dir=str(ep.promoted))
    src = _leg(tmp_path / "tb_full.json", ["B/01", "B/09", "B/02", "C/01"], [ONE, ZERO, ZERO, ZERO])
    rec = L.derive_restricted_leg(src, sp, "test")
    base = tmp_path / "tb.json"
    base.write_text(json.dumps(rec))
    return sp, rd, base


def test_last_successful_epoch_follows_the_rule(tmp_path):
    sp, rd, _ = _held_run(tmp_path, epochs=(1, 2, 3), val_for=(1, 2))
    e, why = L.last_successful_epoch(rd, 3, sp, 3)
    assert e == 2 and why[0][0] == 3 and "no val leg" in why[0][1]
    # epoch 2's promoted set drifts from its staging -> epoch 1
    (L.EpochPaths(rd, 2).promoted / "k" / "SKILL.md").write_text("edited")
    assert L.last_successful_epoch(rd, 3, sp, 3)[0] == 1
    # epoch 1's train leg is on another task set -> nothing qualifies
    _leg(L.EpochPaths(rd, 1).leg("train"), ["A/01", "A/02", "A/07"], [ONE, ZERO, ZERO])
    assert L.last_successful_epoch(rd, 3, sp, 3)[0] is None


def test_heldout_program_end_to_end_on_one_task_set(tmp_path):
    sp, rd, base = _held_run(tmp_path)
    ctx = L.Ctx(rd, 2, _spec(sp), rd / "oracle", heldout_base_leg=str(base))
    steps = L.heldout_steps(ctx)
    assert [s.name for s in steps] == ["held-out epoch is the last successful one",
                                       "held-out baseline is the split's test tasks",
                                       "held-out leg loaded what was promoted",
                                       "held-out leg is the split's test tasks",
                                       "held-out comparison uses the same tasks"]
    rows, halt = L.evaluate(ctx, steps)
    assert halt[0].name == "held-out leg loaded what was promoted"
    assert "launch the leg with tag r_v2_test" in halt[1] and "test key" in halt[1]
    with pytest.raises(L.Halt, match="no runner"):
        L.advance(ctx, steps)                                               # the loop cannot launch it
    _leg(ctx.paths.leg("test"), ["C/01", "B/02", "B/01"], [ALL, ONE, ZERO], skills_dir=str(ctx.paths.promoted))
    s, d = L.advance(ctx, steps)
    assert s.name == "held-out comparison uses the same tasks", d
    c = json.loads((L.operator_dir(rd) / L.HELDOUT_COMPARISON).read_text())
    assert c["n_tasks"] == 3 and c["same_tasks"] and c["epoch"] == 2
    assert (c["base"]["sample_pass"], c["base"]["sample_total"]) == (2, 9)
    assert (c["skill"]["sample_pass"], c["skill"]["sample_total"]) == (5, 9)
    assert c["base"]["derived"] is True and list(c["per_task"]) == ["B/01", "B/02", "C/01"]
    # B/01 base [1,0,1] vs skill [0,0,0]; B/02 [0,0,0] vs [1,0,1]; C/01 [0,0,0] vs [1,1,1]
    assert (c["tasks_skill_more_passes"], c["tasks_skill_fewer_passes"], c["tasks_same_passes"]) == (2, 1, 0)
    assert (c["samples_base_pass_skill_fail"], c["samples_skill_pass_base_fail"]) == (2, 5)
    assert c["split"]["sha256"] == hashlib.sha256(sp.read_bytes()).hexdigest()
    assert L.advance(ctx, steps)[0] is None                                 # idempotent
    # the comparison edited by hand, or the base swapped for the unrestricted superset: refused
    p = L.operator_dir(rd) / L.HELDOUT_COMPARISON
    p.write_text(p.read_text().replace('"delta_sample_pass": 3', '"delta_sample_pass": 4'))
    assert not L._v_heldout_comparison(ctx)[0]
    ctx.heldout_base_leg = str(tmp_path / "tb_full.json")
    ok, d = L._v_heldout_base(ctx)
    assert not ok and "B/09" in d


def test_heldout_baseline_refusals(tmp_path):
    sp, rd, base = _held_run(tmp_path)
    ctx = L.Ctx(rd, 2, _spec(sp), rd / "oracle")
    ok, d = L._v_heldout_base(ctx)
    assert not ok and "heldout_base_leg" in d
    ctx.heldout_base_leg = str(base)
    assert L._v_heldout_base(ctx)[0]
    rec = json.loads(base.read_text())
    base.write_text(json.dumps(dict(rec, skills_dir="/x/skills_v1")))
    assert "no-skill" in L._v_heldout_base(ctx)[1]
    base.write_text(json.dumps(dict(rec, model="other")))
    ok, d = L._v_heldout_base(ctx)
    assert not ok and "model" in d
    base.write_text(json.dumps(rec))
    ok, d = L._v_heldout_epoch(L.Ctx(rd, 1, _spec(sp), rd / "oracle"))
    assert not ok and "selects epoch 2" in d


def test_template_mode_regrades_the_heldout_leg_before_the_comparison(tmp_path):
    ctx = SimpleNamespace(prompt_mode="template")
    names = [s.name for s in L.heldout_steps(ctx)]
    assert names.index(L.LEG_D4_STEP.format(key="test")) == len(names) - 2
    assert L.LEG_D4_STEP.format(key="test") not in [s.name for s in L.heldout_steps(SimpleNamespace())]


DSV = "deepseek/deepseek-v4.1-flash"
REAL = {"model": DSV, "provider_order": ["deepinfra"]}


def _frozen_run(tmp_path):
    from src.orchestrator.spec import ExperimentSpec, split_digest
    sp = _split(tmp_path / "s.json")
    d = json.loads(sp.read_text())
    rd = tmp_path / "hrun"
    rd.mkdir()
    spec = ExperimentSpec(
        name="hrun", benchmark="ssbv2", domain="spreadsheetbench-v2", split_path=str(sp),
        split_sha256=split_digest(sp), n_train=len(d["train"]), n_val=len(d["val"]), n_test=len(d["test"]),
        n_samples=3, solver_model=DSV, solver_provider_order=("deepinfra",), solver_allow_fallbacks=False,
        user_sim_model="none", judge_model="none", agent_turns=45, episode_timeout_s=10800,
        solver_harness="direct", evolver_model="claude-opus-5-5", evolver_harness="claude_cli", n_epochs=2,
        cost_ceiling_usd=150.0)
    assert spec.validate() == [], spec.validate()
    spec.freeze(rd)
    return sp, rd


def test_main_heldout_plan_and_step(tmp_path, monkeypatch, capsys):
    sp, rd = _frozen_run(tmp_path)
    monkeypatch.setattr(sys, "argv", ["loop", "--run-dir", str(rd), "--heldout", "--plan"])
    assert L.main() == 1                                                    # no epoch qualifies yet
    assert "no epoch qualifies" in capsys.readouterr().out
    for e in (1, 2):
        ep = L.EpochPaths(rd, e)
        for d in (ep.staging, ep.promoted):
            (d / "k").mkdir(parents=True, exist_ok=True)
            (d / "k" / "SKILL.md").write_text("---\nname: k\ndescription: d\n---\nb\n")
        _leg(ep.leg("train"), ["A/01", "A/02", "A/03"], [ONE, ZERO, ZERO], skills_dir=str(ep.promoted), **REAL)
        _leg(ep.leg("val"), ["A/04", "A/05"], [ONE, ZERO], skills_dir=str(ep.promoted), **REAL)
    src = _leg(tmp_path / "tb_full.json", ["B/01", "B/09", "B/02", "C/01"], [ONE, ZERO, ZERO, ZERO], **REAL)
    out = tmp_path / "tb_restricted.json"
    monkeypatch.setattr(sys, "argv", ["loop", "--derive-restricted-leg", str(src), "--split", str(sp),
                                      "--out", str(out)])
    assert L.main() == 0 and out.is_file()
    assert "2/9 samples passed" in capsys.readouterr().out
    assert L.main() == 1 and "exists" in capsys.readouterr().out            # written once
    (rd / "RUNTIME.json").write_text(json.dumps({"heldout_base_leg": str(out)}))
    _leg(L.EpochPaths(rd, 2).leg("test"), ["B/01", "B/02", "C/01"], [ALL, ZERO, ZERO],
         skills_dir=str(L.EpochPaths(rd, 2).promoted), **REAL)
    monkeypatch.setattr(sys, "argv", ["loop", "--run-dir", str(rd), "--heldout", "--plan"])
    assert L.main() == 0
    board = capsys.readouterr().out
    assert "epoch 2" in board and "HALT  held-out comparison uses the same tasks" in board
    monkeypatch.setattr(sys, "argv", ["loop", "--run-dir", str(rd), "--heldout", "--step"])
    assert L.main() == 0
    assert "RAN + VERIFIED: held-out comparison uses the same tasks" in capsys.readouterr().out
    c = json.loads((L.operator_dir(rd) / L.HELDOUT_COMPARISON).read_text())
    assert (c["base"]["sample_pass"], c["skill"]["sample_pass"], c["n_tasks"]) == (2, 3, 3)
    monkeypatch.setattr(sys, "argv", ["loop", "--run-dir", str(rd), "--heldout", "--epoch", "1", "--plan"])
    assert L.main() == 0 and "the rule selects epoch 2" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", ["loop", "--run-dir", str(rd), "--heldout", "--step", "--drive"])
    assert L.main() == 64


# ══ the held-out baseline's driver ═══════════════════════════════════════════════════════════════
def test_leg_driver_plans_only_on_a_named_split(tmp_path):
    import ssb_leg_daytona as leg
    a = SimpleNamespace(split=None, key="test", tasks=None, n_shards=2, max_heavy=2, shard_name=None,
                        run_dir=str(tmp_path), cmd="plan")
    with pytest.raises(SystemExit, match="--split is required"):
        leg.plan(a)
    assert not hasattr(leg, "DEFAULT_SPLIT")


def test_leg_driver_bundles_the_plans_split_and_no_other(tmp_path, monkeypatch):
    import ssb_leg_daytona as leg
    data = tmp_path / "data" / "spreadsheetbench-v2"
    for cat in leg.CATEGORIES:
        (data / cat).mkdir(parents=True)
        (data / cat / "dataset.json").write_text("[]")
    monkeypatch.setattr(leg, "DATA", data)
    with pytest.raises(ValueError, match="split_file"):
        leg.build_bundle("s01", [], "", tmp_path / "x.tgz")
    man = leg.build_bundle("s01", [], "", tmp_path / "b.tgz", split_file=str(NEW_SPLIT))
    assert man["split_sha256"] == hashlib.sha256(NEW_SPLIT.read_bytes()).hexdigest()
    with tarfile.open(tmp_path / "b.tgz") as tf:
        names = set(tf.getnames())
    assert [n for n in names if n.endswith("split.json")] == [str(NEW_SPLIT.resolve()).lstrip("/")]
