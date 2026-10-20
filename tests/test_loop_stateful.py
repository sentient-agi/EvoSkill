"""Stateful evolver -- the LOOP side: `_evolver_session` / stateful_evolver.validate_session (one HALT per rule, the
needles taken from the code's messages), main()'s presence read of RUNTIME `evolver_session`, the evo layout's dirs
(_evidence_layout / _evidence_dirs / EpochPaths.evo), the stateful epoch program (_stateful_steps), the _run_evolver
env + argv, the _drive DRIVE-DONE hook, and run_session_finish's Mac-side rules on a hand-written session state
record.

The fixture is test_loop_pool_container's shape (a pool-container Daytona run) with the stateful declarations added:
RUNTIME evolver_session / prompt_mode / evolver_evidence / evolver_evidence_layout=evo / claude.effort / daytona_leg
(pool + TTL), evolver_daytona WITHOUT grants (the stateful session has none), the effort opt-out in the
environment, an explicit claude-* evolver model.

The after-run verifies on a REAL driver run (v_session_contained / _read_nothing_forbidden / _continued /
_snapshotted, and run_session_finish's actual teardown) were checked here on tests/
test_evolver_session's two-epoch FakeMachine fixture; that fixture predates the evo layout, so those checks are not
here until it is adapted (they belong beside that driver's suite).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.orchestrator.loop as L  # noqa: E402
from src.orchestrator import claude_pins  # noqa: E402
from src.orchestrator import daytona_evolver as de  # noqa: E402
from src.orchestrator import pool_container_evolver as pce  # noqa: E402
from src.orchestrator import stateful_evolver as SE  # noqa: E402
from src.orchestrator import stateful_layout as SL  # noqa: E402
from tests.test_evolver_container import machines_json, pool_row  # noqa: E402
from tests.test_loop_pool_container import LEG_SNAP, LO, _dispatch, _spec  # noqa: E402
from tests.test_loop_template_mode import _ctx as _tm_ctx  # noqa: E402
from tests.test_loop_template_mode import _frozen_rd, _sig  # noqa: E402


def _runtime(**over) -> dict:
    rt = {"daytona_leg": {"pool": True, "require_snapshot_ref": LEG_SNAP, "pool_ttl_minutes": 1440},
          "claude": {"effort": "default"}, "evolver_session": "stateful", "prompt_mode": "template",
          "evolver_host": "daytona", "evolver_evidence": "all_epochs", "evolver_evidence_layout": "evo",
          "evolver_daytona": {"placement": "pool_container"}}
    rt.update(over)
    return rt


def per_epoch_kw() -> dict:
    """The same run declared per_epoch (on the digests layout: "evo" belongs to the stateful session)."""
    return dict(evolver_session="per_epoch", evidence_layout="digests",
                runtime=_runtime(evolver_session="per_epoch", evolver_evidence_layout="digests"))


def stateful_parts(tmp_path, monkeypatch, *, runtime=None, cfg_extra=None, cfg_drop=(), spec_kw=None,
                   effort="default", **kw) -> tuple:
    """(run dir, spec, Ctx kwargs) of a run that satisfies every rule of validate_session; each argument breaks
    exactly one. The drive test rebuilds a Ctx per epoch from the same kwargs, as _drive does."""
    rd = tmp_path / f"run-{uuid.uuid4().hex[:6]}"
    rd.mkdir(parents=True)
    (rd / "RUNTIME.json").write_text(json.dumps(_runtime() if runtime is None else runtime))
    tok = tmp_path / "tok"
    tok.write_text("sk-ant-oat01-FAKE")
    os.chmod(tok, 0o600)
    cfg = {"oauth_token_file": str(tok), "require_libreoffice": LO, "placement": "pool_container"}
    cfg.update(cfg_extra or {})
    for k in cfg_drop:
        cfg.pop(k, None)
    if effort is None:
        monkeypatch.delenv(claude_pins.EFFORT_ENV, raising=False)
    else:
        monkeypatch.setenv(claude_pins.EFFORT_ENV, effort)
    base = dict(evolver_host="daytona", evolver_daytona=cfg, leg_pool=True, pipeline_val=True,
                prompt_mode="template", evolver_evidence="all_epochs", evidence_layout="evo",
                evolver_session="stateful")
    base.update(kw)
    return rd, _spec(tmp_path, **(spec_kw or {})), base


def stateful_ctx(tmp_path, monkeypatch, *, epoch=1, **kw):
    """A Ctx that satisfies every rule of validate_session (the PASS case); each argument breaks exactly one."""
    rd, spec, base = stateful_parts(tmp_path, monkeypatch, **kw)
    return L.Ctx(rd, epoch, spec, rd / "oracle", **base)


# ══ 1. the PASS case, the evo layout's dirs, and the per-epoch path untouched ═══════════════════════════════
def test_a_full_stateful_ctx_passes_and_reads_the_evo_layout(tmp_path, monkeypatch):
    ctx = stateful_ctx(tmp_path, monkeypatch)
    assert L._evolver_session(ctx) == "stateful"
    assert SE.validate_session(ctx) is None
    assert L._evolver_host(ctx) == "daytona" and L._prompt_mode(ctx) == "template"
    assert L._evidence_mode(ctx) == "all_epochs" and L._evidence_layout(ctx) == "evo"
    assert SL.evolver_session(ctx.run_dir) == "stateful"        # the driver reads the same answer from the file
    # no grants: an evolver_daytona without them reads as none (the adapter's reference/ is the evolver's view)
    assert not any(L._grants(ctx)[k] for k in ("train_inputs", "train_references", "grader_clean"))
    # the evolver's material: epoch N reads <run>/evo_e{N-1} (0 = the base train leg), all of them cumulatively
    rd = Path(ctx.run_dir).resolve()
    assert ctx.paths.evo == ctx.run_dir / "evo_e0" and L._evidence_dirs(ctx) == [rd / "evo_e0"]
    c3 = L.Ctx(ctx.run_dir, 3, ctx.spec, ctx.oracle, **stateful_parts(tmp_path / "k", monkeypatch)[2])
    assert c3.paths.evo == ctx.run_dir / "evo_e2"
    assert L._evidence_dirs(c3) == [rd / f"evo_e{k}" for k in range(3)] == c3.paths.evo_dirs()
    # the effort opt-out spellings all pass (claude_pins._OPT_OUT); the flag is what matters, not the spelling
    for sp in ("", "none", "off", "DEFAULT"):
        assert L._evolver_session(stateful_ctx(tmp_path, monkeypatch, effort=sp)) == "stateful"


def test_the_evo_layout_belongs_to_the_stateful_session_only(tmp_path, monkeypatch):
    ctx = stateful_ctx(tmp_path, monkeypatch, **per_epoch_kw())
    ctx.evidence_layout = "evo"
    with pytest.raises(L.Halt, match="needs evolver_session=stateful"):
        L._evidence_layout(ctx)
    for lay in ("digests", "rollouts"):                          # the per-epoch driver's layouts, unchanged
        ctx.evidence_layout = lay
        assert L._evidence_layout(ctx) == lay
    ctx.evidence_layout = "evoo"
    with pytest.raises(L.Halt, match="is not one of"):
        L._evidence_layout(ctx)


def test_per_epoch_is_the_default_and_checks_nothing_else(tmp_path, monkeypatch):
    # absent = "per_epoch": a Ctx with every stateful precondition broken still passes as per_epoch
    ctx = stateful_ctx(tmp_path, monkeypatch, runtime={"daytona_leg": {"pool": True, "require_snapshot_ref": LEG_SNAP}},
                       evolver_session="per_epoch", prompt_mode="ao", evolver_evidence="current_epoch",
                       evidence_layout="digests", evolver_host="local", effort="high")
    assert L._evolver_session(ctx) == "per_epoch"
    assert L.Ctx.__dataclass_fields__["evolver_session"].default == "per_epoch"
    assert L.Ctx(ctx.run_dir, 1, ctx.spec, ctx.oracle).evolver_session == "per_epoch"
    # a bare namespace (the shape test_loop_template_mode's _ctx builds) is per_epoch too
    assert L._evolver_session(NS(prompt_mode="template", evolver_host="daytona")) == "per_epoch"
    # a run dir with no RUNTIME.json: the Ctx value stands (offline callers, fixtures)
    rd = tmp_path / "bare"
    rd.mkdir()
    assert L._evolver_session(NS(run_dir=rd, evolver_session="per_epoch")) == "per_epoch"


# ══ 2. one HALT per rule of validate_session (each mutation breaks exactly that rule) ══════════════════════
def _drop(rt: dict, *keys) -> dict:
    d = json.loads(json.dumps(rt))
    for k in keys:
        a, _, b = k.partition(".")
        (d[a].pop(b, None) if b else d.pop(a, None))
    return d


def _set(rt: dict, path: str, value) -> dict:
    d = json.loads(json.dumps(rt))
    a, _, b = path.partition(".")
    if b:
        d[a][b] = value
    else:
        d[a] = value
    return d


CASES = [
    # rule 1: the loop renders the message
    ("prompt_mode ao", dict(prompt_mode="ao", runtime=_runtime(prompt_mode="ao")), "needs prompt_mode=template"),
    # rule 2: daytona + pool container
    ("evolver_host local", dict(evolver_host="local"), "evolver_daytona.placement=pool_container"),
    ("placement own_sandbox", dict(cfg_extra={"placement": "own_sandbox"}), "evolver_daytona.placement=pool_container"),
    ("placement absent", dict(cfg_drop=("placement",)), "evolver_daytona.placement=pool_container"),
    ("placement typo", dict(cfg_extra={"placement": "pool"}), "is not one of"),      # _evolver_placement's own HALT
    # rule 3: the container must exist from epoch 1
    ("placement_from_epoch", dict(cfg_extra={"placement_from_epoch": 2}), "placement_from_epoch"),
    # rule 4: every epoch's evidence
    ("evolver_evidence current_epoch", dict(evolver_evidence="current_epoch"), "needs evolver_evidence=all_epochs"),
    # rule 5: the evo layout (the per-epoch driver's layouts are refused)
    ("evidence_layout rollouts", dict(evidence_layout="rollouts"), "needs evolver_evidence_layout=evo"),
    ("evidence_layout digests", dict(evidence_layout="digests"), "needs evolver_evidence_layout=evo"),
    # rule 6: claude_cli on an explicit model id
    ("harness direct", dict(spec_kw={"evolver_harness": "direct"}), "needs evolver_harness=claude_cli"),
    ("model alias", dict(spec_kw={"evolver_model": "opus[1m]"}), "explicit claude-"),
    ("model empty", dict(spec_kw={"evolver_model": ""}), "explicit claude-"),
    # rule 7: no --effort flag (Anthropic's default effort); an UNSET variable means "high"
    ("effort high", dict(effort="high"), "export EVOSKILL_CLAUDE_EFFORT=default"),
    ("effort unset", dict(effort=None), "export EVOSKILL_CLAUDE_EFFORT=default"),
    ("effort max", dict(effort="max"), "Anthropic's default effort"),
    ("effort bogus", dict(effort="bogus"), "EVOSKILL_CLAUDE_EFFORT"),
    # rule 8: the documentation block agrees with the live channel
    ("claude.effort high", dict(runtime=_set(_runtime(), "claude.effort", "high")), "claude.effort='high'"),
    ("claude.effort absent", dict(runtime=_drop(_runtime(), "claude.effort")), "claude.effort=None"),
    ("claude block absent", dict(runtime=_drop(_runtime(), "claude")), "claude.effort"),
    # rule 9: the leg pool with a TTL
    ("pool false", dict(runtime=_set(_runtime(), "daytona_leg.pool", False)), "daytona_leg.pool: true"),
    ("pool_ttl_minutes absent", dict(runtime=_drop(_runtime(), "daytona_leg.pool_ttl_minutes")), "pool_ttl_minutes"),
    ("pool_ttl_minutes zero", dict(runtime=_set(_runtime(), "daytona_leg.pool_ttl_minutes", 0)), "pool_ttl_minutes"),
    ("pool_ttl_minutes string", dict(runtime=_set(_runtime(), "daytona_leg.pool_ttl_minutes", "1440")), "pool_ttl_minutes"),
    ("pool_ttl_minutes bool", dict(runtime=_set(_runtime(), "daytona_leg.pool_ttl_minutes", True)), "pool_ttl_minutes"),
]


@pytest.mark.parametrize("name,mut,needle", CASES, ids=[c[0] for c in CASES])
def test_each_rule_halts_with_its_needle(tmp_path, monkeypatch, name, mut, needle):
    ctx = stateful_ctx(tmp_path, monkeypatch, **mut)
    with pytest.raises(L.Halt, match=needle) as ei:
        L._evolver_session(ctx)
    msg = str(ei.value)
    # names the key, the channel, or (the placement typo) the placement key whose own HALT fires first
    assert any(s in msg for s in ("evolver_session", "EVOSKILL_CLAUDE_EFFORT", "evolver_daytona.placement")), msg
    # the same Ctx as per_epoch passes: nothing here touches today's path
    ctx.evolver_session = "per_epoch"
    (ctx.run_dir / "RUNTIME.json").write_text(json.dumps(_set(json.loads((ctx.run_dir / "RUNTIME.json").read_text()),
                                                              "evolver_session", "per_epoch")))
    assert L._evolver_session(ctx) == "per_epoch"


def test_the_effort_message_states_the_rule_and_the_remedy(tmp_path, monkeypatch):
    ctx = stateful_ctx(tmp_path, monkeypatch, effort="high")
    with pytest.raises(L.Halt) as ei:
        L._evolver_session(ctx)
    msg = str(ei.value)
    assert "claude-opus-5-5" in msg and "unset (use anthropic's default)" in msg
    assert "an unset variable means 'high'" in msg and "claude_pins.DEFAULT_EFFORT" in msg
    assert "--effort high" in msg


def test_rule_order_is_the_codes_and_the_first_broken_rule_speaks(tmp_path, monkeypatch):
    # everything broken at once: rule 1 (prompt_mode) is the one that HALTs
    ctx = stateful_ctx(tmp_path, monkeypatch, prompt_mode="ao", evolver_host="local", evolver_evidence="current_epoch",
                       evidence_layout="digests", effort="high",
                       runtime=_set(_runtime(prompt_mode="ao"), "claude.effort", "high"))
    with pytest.raises(L.Halt, match="needs prompt_mode=template"):
        L._evolver_session(ctx)


# ══ 3. the value itself, and the file the driver reads ══════════════════════════════════════════════════════
def test_a_typo_halts_and_a_ctx_that_disagrees_with_runtime_json_halts(tmp_path, monkeypatch):
    for bad in ("statefull", "Stateful", "", None, True, 1):
        ctx = stateful_ctx(tmp_path, monkeypatch, evolver_session=bad, runtime=_runtime(evolver_session=bad))
        with pytest.raises(L.Halt, match="is not one of"):
            L._evolver_session(ctx)
    # the Ctx says stateful, the file says nothing: the driver and the upload check would run per-epoch
    ctx = stateful_ctx(tmp_path, monkeypatch, runtime=_drop(_runtime(), "evolver_session"))
    with pytest.raises(L.Halt, match="must agree"):
        L._evolver_session(ctx)
    # the file says stateful, the Ctx says per_epoch
    ctx = stateful_ctx(tmp_path, monkeypatch, evolver_session="per_epoch")
    with pytest.raises(L.Halt, match="must agree"):
        L._evolver_session(ctx)
    # the file carries a typo while the Ctx is fine: the file's error is the HALT
    ctx = stateful_ctx(tmp_path, monkeypatch, runtime=_runtime(evolver_session="statefull"))
    with pytest.raises(L.Halt, match="is not one of"):
        L._evolver_session(ctx)
    # an unreadable RUNTIME.json is a HALT, never a guess
    ctx = stateful_ctx(tmp_path, monkeypatch)
    (ctx.run_dir / "RUNTIME.json").write_text("{nope")
    with pytest.raises(L.Halt, match="unreadable"):
        L._evolver_session(ctx)


# ══ 4. main()'s presence read ═══════════════════════════════════════════════════════════════════════════════
def test_main_reads_evolver_session_by_presence_and_validates_it_in_the_preflight(tmp_path, monkeypatch, capsys):
    seen = []

    def rec(ctx):
        seen.append(ctx.evolver_session)
        raise L.Halt(f"captured evolver_session={ctx.evolver_session!r}")

    monkeypatch.setattr(L, "_evolver_session", rec)
    for i, (runtime, want) in enumerate((({}, "per_epoch"), ({"evolver_session": "stateful"}, "stateful"),
                                         ({"evolver_session": "per_epoch"}, "per_epoch"))):
        (tmp_path / f"c{i}").mkdir()
        rd = _frozen_rd(tmp_path / f"c{i}", runtime)
        monkeypatch.setattr(sys, "argv", ["loop", "--run-dir", str(rd), "--epoch", "1", "--plan"])
        assert L.main() == 1
        out = capsys.readouterr().out
        assert "HALT" in out and f"captured evolver_session={want!r}" in out, out
    assert seen == ["per_epoch", "stateful", "per_epoch"]


@pytest.mark.parametrize("runtime,needle", [
    ({"evolver_session": "statefull"}, "evolver_session='statefull'"),
    ({"evolver_session": True}, "evolver_session=True"),
    ({"evolver_session": "stateful"}, "needs prompt_mode=template"),      # the real validator, rule 1 first
])
def test_main_refuses_a_bad_or_unmet_session_before_anything_is_planned(tmp_path, monkeypatch, capsys, runtime, needle):
    monkeypatch.setenv(claude_pins.EFFORT_ENV, "default")
    rd = _frozen_rd(tmp_path, runtime)
    monkeypatch.setattr(sys, "argv", ["loop", "--run-dir", str(rd), "--epoch", "1", "--plan"])
    assert L.main() == 1
    out = capsys.readouterr().out
    assert "HALT" in out and needle in out, out
    assert not list(rd.glob("BLOCKED_*")) and not list(rd.glob("ITER*_PROMPT.md"))


# ══ 5. the epoch program (_stateful_steps over the template program) ════════════════════════════════════════
# The template-mode step each stateful step replaces, by NAME; the steps the stateful session drops (the loop
# computes NOTHING for the evolver: no digest surface, no grants, no prompt checks; and no loop re-grade: the leg
# driver grades every rollout with the frozen grade() and re-grades once itself); and the ones inserted after the
# read step. Everything else is the same Step object in the
# same order.
REGRADE_STEPS = {"newest batch has no grader failure", "train leg has no grader failure", "val leg has no grader failure"}
REPLACED = {"evidence surface is train-only": L.STATEFUL_LAYOUT_STEP,
            "evolver prompt rendered from the template": L.STATEFUL_RENDER_STEP,
            "evolver ran under declared containment": L.STATEFUL_CONTAINED_STEP,
            "evolver read nothing forbidden": L.STATEFUL_READ_STEP}
GONE = {"evidence surface is complete", "evidence names its producing library", "evolver inputs are train-only",
        "evolver references are train-only", "clean grader is granted and code only",
        "evolver prompt numbers re-derive", "prompt names no held-out task id, no held-out figure",
        "prompt is raw information, not analysis", "prompt names every granted evidence dir",
        "prompt names exactly the granted dirs"} | REGRADE_STEPS
INSERTED = [L.STATEFUL_CONTINUED_STEP, L.STATEFUL_SNAPSHOT_STEP]      # no held-out id scan of what it wrote
PROMOTE = "promoted set matches staging"


def test_per_epoch_program_is_todays_template_program(tmp_path, monkeypatch):
    want = _sig(L._template_steps(L.epoch_steps()))
    assert _sig(L._steps_for(_tm_ctx())) == want                                   # no run dir: per_epoch
    ctx = stateful_ctx(tmp_path, monkeypatch, pipeline_val=False, **per_epoch_kw())
    assert _sig(L._serial_steps_for(ctx)) == want
    names = [s.name for s in L._steps_for(ctx)]
    assert not any(n in names for n in [*REPLACED.values(), *INSERTED]) and GONE < set(names)
    # the per-epoch program keeps its grader-failure steps, each with its re-grade runner
    by = {s.name: s for s in L._steps_for(ctx)}
    assert by["newest batch has no grader failure"].run is L._run_batch_regrade
    assert all(by[n].run is not None for n in REGRADE_STEPS)


def test_stateful_program_replaces_by_name_drops_the_computed_checks_and_inserts_two(tmp_path, monkeypatch):
    tmpl = L._template_steps(L.epoch_steps())
    ctx = stateful_ctx(tmp_path, monkeypatch, pipeline_val=False)
    steps = L._stateful_steps(tmpl, ctx)
    # _serial_steps_for applies exactly this transform (epoch_steps() builds fresh objects per call: code pins)
    assert _sig(L._serial_steps_for(ctx)) == _sig(steps)
    exp = []                                                   # the transform, applied here by hand
    for s in tmpl:
        if s.name in GONE:
            continue
        exp.append(REPLACED.get(s.name, s.name))
        if s.name == "evolver read nothing forbidden":
            exp += INSERTED
    assert [s.name for s in steps] == exp
    assert not (GONE | set(REPLACED)) & set(exp)
    i = exp.index
    assert exp[:3] == ["infra declaration agrees", "benchmark plug-in is frozen and signed", L.STATEFUL_LAYOUT_STEP]
    assert i(L.STATEFUL_RENDER_STEP) + 1 == i(L.TEMPLATE_CONVERSATION_STEP)
    # no prompt check between the render and the evolver
    assert i(L.TEMPLATE_CONVERSATION_STEP) + 1 == i("skills load through the real consumer")
    assert i(L.STATEFUL_READ_STEP) + 1 == i(L.STATEFUL_CONTINUED_STEP)
    assert i(L.STATEFUL_SNAPSHOT_STEP) + 1 == i(PROMOTE)
    # every other Step is the SAME object, in the same order
    by = {s.name: s for s in steps}
    tby = {s.name: s for s in tmpl}
    for n in exp:
        if n in tby:
            assert by[n] is tby[n], n
    # the replacements bind the evo layout builder, the renderer and the stateful verifies; the inserted two have no runner
    lay, ren = by[L.STATEFUL_LAYOUT_STEP], by[L.STATEFUL_RENDER_STEP]
    assert lay.run is SE.run_build_evo_layout and lay.verify is SE.v_evo_layout
    assert ren.run is SE.run_render_session_prompt and ren.verify is SE.v_session_prompt_rendered and ren.defect == "loop-rendered prompt"
    for n, fn, defect in ((L.STATEFUL_CONTAINED_STEP, SE.v_session_contained, "containment"),
                          (L.STATEFUL_READ_STEP, SE.v_session_read_nothing_forbidden, "forbidden read"),
                          (L.STATEFUL_CONTINUED_STEP, SE.v_session_continued, "resume"),
                          (L.STATEFUL_SNAPSHOT_STEP, SE.v_session_snapshotted, "resume material")):
        assert by[n].verify is fn and by[n].run is None and by[n].defect == defect, n
    # kept
    assert by["skills load through the real consumer"].run is L._run_evolver
    assert by["evolver ran on the declared model"].verify is L._v_evolver_model
    assert by[PROMOTE].run is L._run_promote
    assert not any(s.operator_gate for s in steps)                # the run is automatic: nothing waits on a person
    # no re-grade step (the leg driver re-grades once itself); the reconciles stay
    assert not REGRADE_STEPS & set(exp) and {"train outcomes reconcile", "val outcomes reconcile"} < set(exp)
    # nothing computed for the evolver survives under another name
    assert not [n for n in exp if any(w in n for w in ("prompt names", "re-derive", "raw information", "inputs",
                                                        "references", "grader is granted", "producing library"))]
    # pipeline mode keeps the stateful steps in order, ahead of the promotion
    pnames = [s.name for s in L._steps_for(stateful_ctx(tmp_path, monkeypatch))]
    mine = (*REPLACED.values(), *INSERTED)
    assert [n for n in pnames if n in mine] == [n for n in exp if n in mine]
    assert pnames.index(L.STATEFUL_SNAPSHOT_STEP) < pnames.index(PROMOTE)
    # ... and the val leg's checks, taken BY NAME from the stateful program, carry no re-grade step either
    assert [s.name for s in L._val_bundle(stateful_ctx(tmp_path, monkeypatch))] == [
        L.VAL_LEG_STEP, "val pairs comparably with base", "val: no infra exclusions", "val outcomes reconcile"]


def test_reconcile_counts_grader_failed_as_never_graded(tmp_path, monkeypatch):
    """A rollout whose grade() raised twice is listed in the leg's grader_failed and
    is the never-graded channel; graded-fail = total - pass - never. A leg without the key has none."""
    ctx = stateful_ctx(tmp_path, monkeypatch)
    leg = {"n": 4, "n_samples": 3, "sample_pass": 5, "sample_total": 12, "turn_exhausted": 1,
           "grader_failed": ["a#s1", "b#s3"]}

    def reconcile(**over):
        ctx.paths.leg("train").write_text(json.dumps({**leg, **over}))
        return L._v_outcomes_reconcile(ctx, "train")
    assert reconcile() == (True, "train: pass 5, graded-fail 5, never-graded 2 = 12, capped-but-graded 1")
    assert reconcile(grader_failed=[]) == (True, "train: pass 5, graded-fail 7, never-graded 0 = 12, "
                                                 "capped-but-graded 1")
    ctx.paths.leg("train").write_text(json.dumps({k: v for k, v in leg.items() if k != "grader_failed"}))
    assert L._v_outcomes_reconcile(ctx, "train")[1].startswith("train: pass 5, graded-fail 7, never-graded 0 = 12")
    ok, det = reconcile(sample_pass=11)
    assert not ok and det == "train: pass(11) + never-graded(2) exceeds total(12)"


def test_stateful_steps_anchor_on_a_reduced_list_too():
    only_promote = [L.Step(PROMOTE, lambda c: (True, ""))]
    assert [s.name for s in L._stateful_steps(only_promote)] == [*INSERTED, PROMOTE]
    both = [L.Step("evolver read nothing forbidden", lambda c: (True, "")), L.Step("x", lambda c: (True, "")),
            L.Step(PROMOTE, lambda c: (True, ""))]
    assert [s.name for s in L._stateful_steps(both)] == [L.STATEFUL_READ_STEP, *INSERTED, "x", PROMOTE]
    neither = [L.Step("y", lambda c: (True, "")), L.Step("prompt names exactly the granted dirs", lambda c: (True, ""))]
    assert [s.name for s in L._stateful_steps(neither)] == ["y"]


# ══ 6. the dispatch env and argv ════════════════════════════════════════════════════════════════════════════
def test_run_evolver_exports_the_session_and_the_system_prompt_only_when_stateful(tmp_path, monkeypatch):
    monkeypatch.setenv(SL.ENV_SESSION, "stateful")                 # stale exports in the operator's shell
    monkeypatch.setenv(SL.ENV_SYSTEM_PROMPT, "/nope/EVOLVER_SYSTEM.md")
    monkeypatch.setenv("EVOSKILL_PRIOR_DIGESTS", "/nope/digests_e9_train")
    ctx = stateful_ctx(tmp_path, monkeypatch)
    p = _dispatch(tmp_path, monkeypatch, ctx)
    env = p.kw["env"]
    assert env[SL.ENV_SESSION] == "stateful"
    assert env[SL.ENV_SYSTEM_PROMPT] == str(Path(ctx.run_dir).resolve() / SL.SYSTEM_PROMPT_FILE)
    assert "CLAUDE_CODE_EFFORT_LEVEL" not in env and env["EVOSKILL_EVOLVER_EVIDENCE"] == "all_epochs"
    # epoch 1: $3 is the base leg's evo_e0 (the newest material), nothing earlier travels
    assert env["EVOSKILL_PRIOR_DIGESTS"] == ""
    assert p.argv[1:] == ["scripts/daytona/evolver_container.py", "1", str(ctx.paths.prompt),
                          str(ctx.run_dir / "evo_e0"), str(ctx.paths.staging)]
    # epoch 3: $3 is evo_e2; evo_e0 and evo_e1 travel in EVOSKILL_PRIOR_DIGESTS, resolved
    c3 = stateful_ctx(tmp_path / "e3", monkeypatch, epoch=3)
    p3 = _dispatch(tmp_path / "e3", monkeypatch, c3)
    assert p3.argv[2] == "3" and p3.argv[4] == str(c3.run_dir / "evo_e2")
    assert p3.kw["env"]["EVOSKILL_PRIOR_DIGESTS"] == os.pathsep.join(
        str(Path(c3.run_dir).resolve() / f"evo_e{k}") for k in (0, 1))
    # per_epoch: the session variables are never inherited, and $3 is the digests dir as before
    per = stateful_ctx(tmp_path / "per", monkeypatch, **per_epoch_kw())
    p2 = _dispatch(tmp_path / "per", monkeypatch, per)
    assert SL.ENV_SESSION not in p2.kw["env"] and SL.ENV_SYSTEM_PROMPT not in p2.kw["env"]
    assert p2.argv[1:2] == ["scripts/daytona/evolver_container.py"] and p2.argv[4] == str(per.paths.digests)
    # a stateful run the loop cannot serve HALTs before any spend (rule 7 here)
    bad = stateful_ctx(tmp_path / "bad", monkeypatch, effort="high")
    bad.paths.prompt.write_text("the prompt " * 50)
    monkeypatch.setattr(L.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(AssertionError("spent")))
    with pytest.raises(L.Halt, match="EVOSKILL_CLAUDE_EFFORT=default"):
        L._run_evolver(bad)


# ══ 7. the DRIVE DONE hook ══════════════════════════════════════════════════════════════════════════════════
def _drive(rd, spec, kw, *, epoch=1, until=None):
    said = []
    rc = L._drive(L.Ctx(rd, epoch, spec, rd / "oracle", **kw), NS(until_epoch=until, max_steps=9), rd, spec,
                  rd / "oracle", kw, lambda text, key, c=None: said.append(key), lambda c: "")
    return rc, said


def test_drive_calls_the_session_finish_once_at_drive_done_and_a_halt_is_rc_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(L, "advance", lambda ctx, steps=None: (None, "every step verifies (fake)"))
    calls = []
    monkeypatch.setattr(SE, "run_session_finish", lambda ctx: calls.append(ctx.epoch))
    rd, spec, kw = stateful_parts(tmp_path, monkeypatch, pipeline_val=False)
    rc, said = _drive(rd, spec, kw)
    out = capsys.readouterr().out
    assert rc == 0 and calls == [1] and "DRIVE DONE" in out, out
    # a multi-epoch drive tears down ONCE, after the last epoch (a fresh Ctx per epoch, as _drive builds it)
    calls.clear()
    rc, said = _drive(rd, spec, kw, until=3)
    assert rc == 0 and calls == [3] and "DRIVE DONE" in capsys.readouterr().out
    # a HALT from the finish: rc 1, the HALT block, no DRIVE DONE
    calls.clear()

    def boom(ctx):
        raise L.Halt("finish exited 4; the container is NOT torn down")
    monkeypatch.setattr(SE, "run_session_finish", boom)
    rc, said = _drive(rd, spec, kw)
    out = capsys.readouterr().out
    assert rc == 1 and "HALT" in out and "NOT torn down" in out and "DRIVE DONE" not in out, out
    assert any(k.startswith("halt:1:0:finish:") for k in said)
    # per_epoch never calls it
    monkeypatch.setattr(SE, "run_session_finish", lambda ctx: calls.append("per_epoch"))
    rd2, spec2, kw2 = stateful_parts(tmp_path / "p", monkeypatch, pipeline_val=False, **per_epoch_kw())
    rc, said = _drive(rd2, spec2, kw2)
    assert rc == 0 and calls == [] and "DRIVE DONE" in capsys.readouterr().out


# ══ 8. DRIVE DONE: run_session_finish's Mac-side rules ══════════════════════════════════════════════════════
# The finish subprocess is faked (its exit code set per case); the session's state record is written by hand as
# the driver writes it at the first dispatch. The container-side teardown itself is scripts/daytona/
# evolver_session.py's, tested with its FakeMachine in tests/test_evolver_session.
CID, SANDBOX, SID = "run-20260928t000000-abcdef", "aaaa1111bbbb", str(uuid.uuid4())
TEARDOWN_SCHEMA = "evoskill.evolver_session_teardown/1"


class _FinishPopen:
    calls: list = []
    rc = 0

    def __init__(self, argv, **kw):
        self.argv, self.kw, self.pid = list(argv), kw, 4242
        _FinishPopen.calls.append(self)

    def wait(self, timeout=None):
        self.timeout = timeout
        return _FinishPopen.rc


def _state(rd, **over) -> dict:
    st = {"schema": SL.STATE_SCHEMA, "run": SL.run_name(rd), "run_dir": str(rd), "cid": CID, "sandbox_id": SANDBOX,
          "sid": SID, "created_epoch": 1, "uploaded": [], "epochs": {}, "rebuilds": [], "torn_down": None}
    st.update(over)
    SL.write_state(rd, st)
    return st


@pytest.fixture
def live(tmp_path, monkeypatch):
    """A stateful run whose session is live (its state record on a tmp STATE_ROOT, the pool's machine present)."""
    monkeypatch.setattr(SL, "STATE_ROOT", tmp_path / "evolver_state")
    monkeypatch.setattr(subprocess, "Popen", _FinishPopen)
    monkeypatch.setenv("EVOSKILL_PYTHON", sys.executable)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-MUST-NOT-TRAVEL")
    monkeypatch.setattr(SE, "_machine_absent", lambda rd, st: (False, {"state": "ready"}))
    _FinishPopen.calls, _FinishPopen.rc = [], 0
    rd, spec, kw = stateful_parts(tmp_path, monkeypatch, pipeline_val=False)
    _state(rd)
    return NS(rd=rd, spec=spec, kw=kw, ctx=lambda **over: L.Ctx(rd, 2, spec, rd / "oracle", **dict(kw, **over)))


def test_run_session_finish_runs_the_driver_finish_without_tokens_and_requires_its_record(live):
    with pytest.raises(L.Halt, match="finish exited 0 but no EVOLVER_SESSION_TEARDOWN.json"):
        SE.run_session_finish(live.ctx())
    (p,) = _FinishPopen.calls
    assert p.argv[0] == sys.executable and p.argv[1] == str(de.REPO / SE.FINISH_SCRIPT)
    assert p.argv[2:] == ["finish", "--run-dir", str(live.rd.resolve())] and p.timeout == SE.FINISH_TIMEOUT_S
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in p.kw["env"] and p.kw["cwd"] == str(de.REPO)
    assert "argv" in (live.rd / SE.FINISH_OUT).read_text() and SL.load_state(live.rd)["torn_down"] is None
    # a refusal (exit 4: a dispatch or another session command is live) is named; the container is not torn down
    _FinishPopen.calls, _FinishPopen.rc = [], 4
    with pytest.raises(L.Halt, match="exited 4 \\(a dispatch or another session command is live"):
        SE.run_session_finish(live.ctx())
    assert len(_FinishPopen.calls) == 1 and SL.load_state(live.rd)["torn_down"] is None


def test_finish_passes_machine_absent_ok_only_on_a_reading_that_shows_the_machine_gone(live, monkeypatch):
    for reading, flag in (((True, None), True), ((False, {"state": "ready"}), False), ((None, None), False)):
        monkeypatch.setattr(SE, "_machine_absent", lambda rd, st, r=reading: r)
        _FinishPopen.calls, _FinishPopen.rc = [], 0
        with pytest.raises(L.Halt, match="finish exited 0 but"):
            SE.run_session_finish(live.ctx())
        (p,) = _FinishPopen.calls
        assert ("--machine-absent-ok" in p.argv) is flag, (reading, p.argv)


def test_run_session_finish_halts_or_returns_before_any_call(live, tmp_path, monkeypatch):
    # a Ctx that disagrees with the run's file HALTs before anything; a per_epoch run returns without a call; a
    # stateful run without a state record HALTs (nothing to tear down)
    with pytest.raises(L.Halt, match="must agree"):
        SE.run_session_finish(live.ctx(evolver_session="per_epoch"))
    rd1, spec1, kw1 = stateful_parts(tmp_path / "per", monkeypatch, pipeline_val=False, **per_epoch_kw())
    assert SE.run_session_finish(L.Ctx(rd1, 1, spec1, rd1 / "oracle", **kw1)) is None
    rd2, spec2, kw2 = stateful_parts(tmp_path / "fresh", monkeypatch, pipeline_val=False)
    with pytest.raises(L.Halt, match="nothing shows the run's evolver container was ever created"):
        SE.run_session_finish(L.Ctx(rd2, 1, spec2, rd2 / "oracle", **kw2))
    # a session recorded torn down is never re-finished: its record must exist and verify
    _state(live.rd, torn_down={"record": str(live.rd / SL.TEARDOWN_RECORD), "utc": "t"})
    with pytest.raises(L.Halt, match="says the session is torn down, but no EVOLVER_SESSION_TEARDOWN.json"):
        SE.run_session_finish(live.ctx())
    (live.rd / SL.TEARDOWN_RECORD).write_text(json.dumps({"schema": TEARDOWN_SCHEMA, "cid": CID, "sandbox_id": SANDBOX,
                                                          "sid": SID, "down": {"verified": True, "wiped": False}}))
    with pytest.raises(L.Halt, match="does not verify"):
        SE.run_session_finish(live.ctx())
    assert _FinishPopen.calls == []


def test_machine_absent_reads_the_pools_machine_json(tmp_path):
    rd = tmp_path / "r"
    rd.mkdir()
    (rd / "RUNTIME.json").write_text(json.dumps({"daytona_leg": {"store_root": str(tmp_path / "store")}}))
    st = {"sandbox_id": "aaaa1111bbbb"}
    assert SE._machine_absent(rd, st) == (None, None)                     # no machine.json: not a reading
    f = pce.pool_machines_file(rd)
    f.parent.mkdir(parents=True)

    def pub(machines, current="m01", **kw):
        f.write_text(json.dumps(machines_json(machines, current=current, **kw)))
    pub({"m01": pool_row("aaaa1111bbbb")})
    absent, row = SE._machine_absent(rd, st)
    assert absent is False and row["sandbox_id"] == "aaaa1111bbbb"
    pub({"m01": pool_row("aaaa1111bbbb", state="deleted")})
    assert SE._machine_absent(rd, st)[0] is True
    for state in ("released", "failed"):
        pub({"m01": pool_row("aaaa1111bbbb", state=state)})
        assert SE._machine_absent(rd, st)[0] is True
    pub({"m02": pool_row("cccc2222dddd")}, current="m02")                 # another machine is current
    absent, row = SE._machine_absent(rd, st)
    assert absent is True and row["sandbox_id"] == "cccc2222dddd"
    pub({}, current="")
    assert SE._machine_absent(rd, st) == (True, None)
    pub({"m01": pool_row("aaaa1111bbbb")}, schema="other/1")
    assert SE._machine_absent(rd, st) == (None, None)
    f.write_text("{nope")
    assert SE._machine_absent(rd, st) == (None, None)


def test_teardown_record_problems_rules(tmp_path):
    st = {"cid": CID, "sandbox_id": SANDBOX, "sid": SID}
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("")
    base = {"schema": TEARDOWN_SCHEMA, "cid": CID, "sandbox_id": SANDBOX, "sid": SID,
            "down": {"verified": True, "wiped": True}}
    assert SE.teardown_record_problems({"schema": "x"}) == ["not a teardown record (schema)"]
    assert SE.teardown_record_problems(base, str(ledger), state=st) == \
        ["no ledgered, verified 'container_down' row for this container"]
    ledger.write_text(json.dumps({"event": "container_down", "driver": pce.DRIVER, "id": SANDBOX, "cid": CID,
                                  "verified": True}) + "\n")
    assert SE.teardown_record_problems(base, str(ledger), state=st) == []
    assert SE.teardown_record_problems(dict(base, cid="run-20260927t000000-ffffff"), str(ledger), state=st) == \
        [f"the record's cid 'run-20260927' != the session's {CID[:12]!r}",
         "no ledgered, verified 'container_down' row for this container"]
    assert SE.teardown_record_problems(dict(base, down={"verified": True, "wiped": False}), str(ledger), state=st) == \
        ["down not verified (verified True, wiped False)"]
    ma = dict(base, down=None, machine_absent={"machine_json": None, "utc": "t"})
    assert SE.teardown_record_problems(ma, str(ledger), state=st) == []
    ma["machine_absent"] = {"machine_json": {"sandbox_id": SANDBOX, "state": "deleted"}, "utc": "t"}
    assert SE.teardown_record_problems(ma, str(ledger), state=st) == []
    ma["machine_absent"] = {"machine_json": {"sandbox_id": SANDBOX, "state": "ready"}, "utc": "t"}
    assert SE.teardown_record_problems(ma, str(ledger), state=st) == \
        ["machine_absent claimed, but the record carries no machine.json read showing the sandbox gone"]
    ma["machine_absent"] = {"utc": "t"}
    assert SE.teardown_record_problems(ma, str(ledger), state=st)[0].startswith("machine_absent claimed")


# ══ an empty or unchanged staging is a result ═══════════
def test_an_empty_or_unchanged_staging_loads_and_an_empty_one_is_promoted(tmp_path):
    """The evolver's system prompt says "If you change nothing, the solver carries the same set again": a staging
    byte-identical to the previous set, and an EMPTY one (the solver runs with no skills), both pass the skills
    step, and an empty staging is promoted as an empty set (neither stops the run for a person)."""
    from src.orchestrator.spec import tau3_banking_2525
    rd = tmp_path / "run"
    rd.mkdir()
    ctx = L.Ctx(rd, 2, tau3_banking_2525(), rd / "oracle")
    prev = L.EpochPaths(rd, 1).promoted
    (prev / "alpha").mkdir(parents=True)
    (prev / "alpha" / "SKILL.md").write_text("---\nname: alpha\ndescription: when x\n---\nbody\n")
    st = ctx.paths.staging
    (st / "alpha").mkdir(parents=True)
    (st / "alpha" / "SKILL.md").write_bytes((prev / "alpha" / "SKILL.md").read_bytes())
    ok, det = L._v_skills_load(ctx)
    assert ok, det                                                       # unchanged: the same set runs again
    (st / "alpha" / "SKILL.md").unlink()
    (st / "alpha").rmdir()
    assert not any(st.iterdir())
    ok, det = L._v_skills_load(ctx)
    assert ok, det                                                       # empty: no skills
    L._run_promote(ctx)
    assert ctx.paths.promoted.is_dir() and not any(ctx.paths.promoted.iterdir())
    assert L._v_promoted_matches_staging(ctx)[0]
    ok, det = L._v_skills_load(L.Ctx(rd, 3, tau3_banking_2525(), rd / "oracle"))
    assert not ok and "no staging dir" in det                            # no installed staging is still a failure
