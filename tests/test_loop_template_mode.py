"""Offline tests for RUNTIME prompt_mode = "template" in src/orchestrator/loop.py, and the clean-grader upload
grant in src/orchestrator/daytona_evolver.py.

  * prompt_mode absent / "ao": the epoch program is exactly epoch_steps(), the AO path unchanged;
  * prompt_mode "template": infra agreement FIRST, the frozen plug-in SECOND, the grader-failure outcomes before
    the render and after each leg, the references/grader dirs, the loop-rendered prompt in place of the AO step,
    and a naming check that follows the grants; every other step is the same object; nothing waits on a person;
  * the re-grade of a grader-failed sample runs ONCE through the grading-sandbox driver's CLI (a fake driver here);
  * grader_clean travels to the evolver VM only under RUNTIME evolver_daytona.grader_clean.

No sandbox, no model, no network; everything is written under tmp_path.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import src.orchestrator.loop as L  # noqa: E402
from src.orchestrator import daytona_evolver as de  # noqa: E402
from src.orchestrator import evidence_surface as es  # noqa: E402
from tests.test_evolver_daytona import _good_receipt, _log  # noqa: E402
from tests.test_train_references import _fake  # noqa: E402



def _cp(x):
    return json.loads(json.dumps(x))


def _ctx(**kw):
    base = dict(prompt_mode="template", evolver_host="daytona", epoch=1, base_leg="",
                spec=SimpleNamespace(split_path="", benchmark="ssbv2"))
    base.update(kw)
    return SimpleNamespace(**base)


# ══ the epoch program ════════════════════════════════════════════════════════════════════════════
def _sig(steps):
    """(name, verify code, run code, gate, defect) per step: epoch_steps() builds fresh lambdas on
    every call, so the code objects, not the function objects, are what must be identical."""
    code = lambda f: None if f is None else getattr(f, "__code__", f)      # noqa: E731
    return [(s.name, code(s.verify), code(s.run), s.operator_gate, s.defect) for s in steps]


def test_ao_mode_is_exactly_epoch_steps():
    base = _sig(L.epoch_steps())
    for ctx in (_ctx(prompt_mode="ao"), SimpleNamespace(evolver_host="daytona")):
        assert _sig(L._steps_for(ctx)) == base


def test_template_steps_order_and_replacements():
    base = L.epoch_steps()
    names = [s.name for s in L._steps_for(_ctx())]
    assert names[:2] == ["infra declaration agrees", "benchmark plug-in is frozen and signed"]
    for gone in (L.AO_AUTHOR_STEP, L.AO_INPUTS_NAMING_STEP, L.AO_CONVERSATION_STEP):
        assert gone not in names
    i = names.index
    assert i("newest batch has no grader failure") + 1 == i(L.INPUTS_STEP)
    assert i(L.INPUTS_STEP) < i("evolver references are train-only") \
        < i("clean grader is granted and code only") < i("evolver prompt rendered from the template")
    # the AO conversation step is REPLACED in place, right after the render
    assert i("evolver prompt rendered from the template") + 1 == i(L.TEMPLATE_CONVERSATION_STEP)
    assert i(L.TEMPLATE_CONVERSATION_STEP) < i("evolver prompt numbers re-derive")
    assert i("prompt names exactly the granted dirs") + 1 == i("skills load through the real consumer")
    # both legs' grader-failure checks come after the legs ran and BEFORE any pairing or reconcile
    assert i("val leg loaded what was promoted") + 1 == i("train leg has no grader failure")
    assert i("train leg has no grader failure") + 1 == i("val leg has no grader failure")
    assert i("val leg has no grader failure") + 1 == i("train pairs comparably with base")
    for later in ("train pairs comparably with base", "val pairs comparably with base",
                  "train outcomes reconcile", "val outcomes reconcile"):
        assert i("val leg has no grader failure") < i(later)
    # every base step except the three replaced is present, as the same object, in the same order
    kept = [s.name for s in base
            if s.name not in (L.AO_AUTHOR_STEP, L.AO_INPUTS_NAMING_STEP, L.AO_CONVERSATION_STEP)]
    assert [n for n in names if n in kept] == kept
    tsteps = {s.name: s for s in L._template_steps(base)}
    for s in base:
        if s.name in tsteps:
            assert tsteps[s.name] is s
    assert not any(s.operator_gate for s in tsteps.values())      # the run is automatic: nothing waits on a person
    assert tsteps["evolver prompt rendered from the template"].run is L._run_render_prompt
    assert tsteps["newest batch has no grader failure"].run is L._run_batch_regrade
    for k in ("train", "val"):                          # the runner re-grades once
        assert tsteps[f"{k} leg has no grader failure"].run is not None
    for n in ("infra declaration agrees", "benchmark plug-in is frozen and signed",
              "clean grader is granted and code only", L.TEMPLATE_CONVERSATION_STEP):
        assert tsteps[n].run is None, f"{n} must have no runner"


def test_d4_steps_anchor_on_a_reduced_list_too():
    """A caller's reduced list (no pairing steps) still gets the grader-failure check BEFORE its first reconcile."""
    rec = [L.Step("train outcomes reconcile", lambda c: (True, "")),
           L.Step("x", lambda c: (True, "")),
           L.Step("val outcomes reconcile", lambda c: (True, ""))]
    names = [s.name for s in L._template_steps(rec)][2:]
    assert names == ["train leg has no grader failure", "val leg has no grader failure",
                     "train outcomes reconcile", "x", "val outcomes reconcile"]
    only_val = [s.name for s in L._template_steps([L.Step("val pairs comparably with base",
                                                          lambda c: (True, ""))])][2:]
    assert only_val == ["val leg has no grader failure", "val pairs comparably with base"]
    assert [s.name for s in L._template_steps([L.Step("y", lambda c: (True, ""))])][2:] == ["y"]


def test_a_replaced_epoch_steps_is_honoured_in_both_modes(monkeypatch):
    """Fixtures install `L.epoch_steps = lambda: [...]`; _steps_for looks it up at call time."""
    seen = []
    spy = L.Step("spy", lambda c: (seen.append(getattr(c, "prompt_mode", "ao")) or True, "ok"))
    monkeypatch.setattr(L, "epoch_steps", lambda: [spy])
    ao = L._steps_for(_ctx(prompt_mode="ao"))
    assert ao == [spy]
    tm = L._steps_for(_ctx())
    assert [s.name for s in tm] == ["infra declaration agrees", "benchmark plug-in is frozen and signed", "spy"]
    monkeypatch.setattr(L, "_v_infra_agrees", lambda c: (True, "i"))
    monkeypatch.setattr(L, "_v_plugin_frozen", lambda c: (True, "p"))
    ctx = _ctx()
    ctx._asking_step = ""
    rows, halt = L.evaluate(ctx)
    assert halt is None and [r[0].name for r in rows][-1] == "spy" and seen == ["template"]


def test_prompt_mode_validation():
    assert L._prompt_mode(SimpleNamespace()) == "ao"
    assert L._prompt_mode(_ctx()) == "template"
    for bad in ("Template", "templates", True, None):
        with pytest.raises(L.Halt):
            L._prompt_mode(_ctx(prompt_mode=bad))
    with pytest.raises(L.Halt, match="evolver_host=daytona"):
        L._prompt_mode(_ctx(evolver_host="local"))
    with pytest.raises(L.Halt):
        L._steps_for(_ctx(evolver_host="local"))


def _frozen_rd(tmp_path, runtime):
    from tests.test_loop_daytona_dryrun import NAME, _freeze
    rd = tmp_path / NAME
    rd.mkdir()
    _freeze(rd)
    (rd / "RUNTIME.json").write_text(json.dumps(runtime))
    return rd


@pytest.mark.parametrize("runtime,needle", [
    ({"prompt_mode": "tempate"}, "prompt_mode='tempate'"),
    ({"prompt_mode": "template"}, "needs evolver_host=daytona"),
    ({"evolver_train_references": True}, "at the top level"),
    ({"train_inputs": False}, "at the top level"),
])
def test_main_refuses_bad_prompt_mode_and_misplaced_grants(tmp_path, monkeypatch, capsys, runtime, needle):
    rd = _frozen_rd(tmp_path, runtime)
    monkeypatch.setattr(sys, "argv", ["loop", "--run-dir", str(rd), "--epoch", "1", "--plan"])
    assert L.main() == 1
    out = capsys.readouterr().out
    assert "HALT" in out and needle in out, out


# ══ the template-only verifies ═══════════════════════════════════════════════════════════════════
def test_infra_step(tmp_path, monkeypatch):
    from src.orchestrator import infra
    ctx = _ctx(run_dir=tmp_path)
    ok, d = L._v_infra_agrees(ctx)
    assert not ok and "INFRA.json is missing or invalid" in d
    (tmp_path / "INFRA.json").write_text("{}")
    monkeypatch.setattr(infra, "agree", lambda rd: [])
    ok, d = L._v_infra_agrees(ctx)
    assert ok and "agrees" in d
    monkeypatch.setattr(infra, "agree", lambda rd: [infra.Disagreement("RUNTIME.json", "leg_timeout_s",
                                                                       21600, 3600)])
    ok, d = L._v_infra_agrees(ctx)
    assert not ok and "leg_timeout_s" in d and "1 statement(s) disagree" in d


def test_plugin_step(tmp_path, monkeypatch):
    from src.orchestrator import plugin_runner as pr
    ctx = _ctx(run_dir=tmp_path)
    ok, d = L._v_plugin_frozen(ctx)                    # no setup/PLUGIN.json at all
    assert not ok and "REMEDY" in d and "freeze" in d

    def unsigned(rd, **kw):
        raise pr.PluginDrift("x", {"problems": ["operator signature missing (signed_by / signed_utc empty)"]})
    monkeypatch.setattr(pr, "verify_frozen", unsigned)
    ok, d = L._v_plugin_frozen(ctx)
    assert not ok and "operator signature missing" in d
    monkeypatch.setattr(pr, "verify_frozen", lambda rd, **kw: {"record_sha256": "ab" * 32, "ok": True})
    ok, d = L._v_plugin_frozen(ctx)
    assert ok and "abababababab" in d


def _counts(**kw):
    r = {"passed": 57, "graded_wrong": 94, "not_eligible": 26, "grader_failed": 0, "n_episodes": 177,
         "status": "ok", "reportable": True, "pass_matches_recorded": True, "mismatches": [],
         "unattributed_warnings": [], "problems": [], "samples": [], "recorded_pass": 57,
         "recorded_sample_pass": None}
    r.update(kw)
    return r


def _snap(root: Path, task: str, k: int, data: bytes = b"PK-saved-workbook") -> str:
    """A snapshot dir holding one saved workbook in workspace/output_raw (the loop's re-grade source)."""
    d = root / task.replace("/", "_") / f"iter_1_s{k}"
    (d / "workspace" / "output_raw").mkdir(parents=True, exist_ok=True)
    (d / "workspace" / "output_raw" / f"{task.replace('/', '_')}_in.xlsx").write_bytes(data + task.encode())
    return str(d)


def test_outcome_steps_halt_on_grader_failed_and_unreportable(tmp_path, monkeypatch):
    from src.orchestrator import outcomes
    leg = tmp_path / "run_base_train.json"
    leg.write_text("{}")
    ctx = _ctx(run_dir=tmp_path, base_leg=str(leg))
    monkeypatch.setattr(outcomes, "leg_outcome_counts", lambda rd, lj: _counts())
    ok, d = L._v_batch_outcomes(ctx)
    assert ok and "grader failed 0" in d and "passed 57" in d
    gf = [{"task": f"Template/0{i}_01", "sample": 0, "outcome": "grader_failed", "reason": "failure_warning",
           "recorded": 0.0, "snapshot": _snap(tmp_path / "snaps", f"Template/0{i}_01", 0)} for i in (1, 2)]
    monkeypatch.setattr(outcomes, "leg_outcome_counts",
                        lambda rd, lj: _counts(grader_failed=2, graded_wrong=92, status="grader_failed",
                                               reportable=False, samples=gf))
    ok, d = L._v_batch_outcomes(ctx)
    assert not ok and "RE-GRADE NEEDED" in d and "2 sample(s) were GRADER FAILED" in d
    monkeypatch.setattr(outcomes, "leg_outcome_counts",
                        lambda rd, lj: _counts(status="mismatch", reportable=False,
                                               mismatches=[{"task": "t", "sample": 0}]))
    ok, d = L._v_batch_outcomes(ctx)
    assert not ok and "not reportable" in d and "'mismatch'" in d
    monkeypatch.setattr(outcomes, "leg_outcome_counts", lambda rd, lj: _counts(pass_matches_recorded=False))
    assert not L._v_batch_outcomes(ctx)[0]

    def boom(rd, lj):
        raise ValueError("cannot locate this leg's attempts")
    monkeypatch.setattr(outcomes, "leg_outcome_counts", boom)
    ok, d = L._v_batch_outcomes(ctx)
    assert not ok and "not a zero" in d
    assert not L._v_batch_outcomes(_ctx(run_dir=tmp_path, base_leg=""))[0]
    # epoch N reads skills_v{N-1}'s train leg; the post-leg step reads this epoch's leg
    ctx2 = L.Ctx(tmp_path, 2, SimpleNamespace(), tmp_path / "oracle", prompt_mode="template",
                 evolver_host="daytona")
    assert L._batch_leg(ctx2) == L.EpochPaths(tmp_path, 1).leg("train")
    ok, d = L._v_leg_outcomes(ctx2, "val")
    assert not ok and "no leg result" in d


def test_references_step_follows_the_grant(tmp_path, monkeypatch):
    f = _fake(tmp_path, monkeypatch, grants={})
    ctx = _ctx(run_dir=f.rd, spec=SimpleNamespace(split_path=str(f.split)))
    ok, d = L._v_references_train_only(ctx)
    assert ok and "not granted" in d
    L._run_build_references(ctx)
    assert not de.references_dir(f.rd).exists()
    (f.rd / "RUNTIME.json").write_text(json.dumps({"evolver_daytona": {"train_references": True}}))
    assert not L._v_references_train_only(ctx)[0]          # granted, not built yet
    L._run_build_references(ctx)
    ok, d = L._v_references_train_only(ctx)
    assert ok and "2 reference workbooks" in d


def test_inputs_step_template_follows_grant_and_ao_mode_needs_the_inputs_grant(tmp_path, monkeypatch):
    f = _fake(tmp_path, monkeypatch, grants={})
    spec = SimpleNamespace(split_path=str(f.split))
    t = _ctx(run_dir=f.rd, spec=spec)
    ok, d = L._v_inputs_train_only(t)
    assert ok and "neither uploaded nor named" in d
    L._run_build_inputs(t)
    assert not de.inputs_dir(f.rd).exists()
    ao = _ctx(prompt_mode="ao", run_dir=f.rd, spec=spec)
    ok, d = L._v_inputs_train_only(ao)
    assert not ok and "train_inputs is not granted" in d and "REMEDY" in d
    with pytest.raises(L.Halt, match="train_inputs is not granted"):
        L._run_build_inputs(ao)
    (f.rd / "RUNTIME.json").write_text(json.dumps({"evolver_daytona": {"train_inputs": True,
                                                                       "grader_clean": True}}))
    ok, d = L._v_inputs_train_only(ao)
    assert not ok and "grader_clean is granted" in d
    (f.rd / "RUNTIME.json").write_text(json.dumps({"evolver_daytona": {"train_inputs": True}}))
    L._run_build_inputs(ao)
    assert L._v_inputs_train_only(ao)[0]
    (f.rd / "RUNTIME.json").unlink()                        # no RUNTIME.json: inputs on, unchanged
    assert L._v_inputs_train_only(ao)[0]


# ══ the clean grader: the dir, the grant, the upload ═════════════════════════════════════════════
def _grader(rd: Path, split: Path) -> Path:
    g = rd / de.GRADER_DIRNAME
    (g / "docker_scripts").mkdir(parents=True)
    files = {"scorer.py": "def score_and_diff(a, b, c, d):\n    return (True, 1, 1, '')\n",
             "docker_scripts/scan.py": "print('scan')\n"}
    for rel, body in files.items():
        (g / rel).write_text(body)
    (g / "README.md").write_text("# The grader\n")
    (g / "MANIFEST.json").write_text(json.dumps(
        {"what": "clean grader", "split": str(split),
         "files": [{"file": rel, "clean_sha256": hashlib.sha256(body.encode()).hexdigest(),
                    "task_id_hits": []} for rel, body in files.items()]}))
    return g


def test_grader_grant_is_a_runtime_boolean_absent_off(tmp_path):
    assert de.evolver_grader_grant(tmp_path) is False
    (tmp_path / "RUNTIME.json").write_text(json.dumps({"evolver_daytona": {}}))
    assert de.evolver_grader_grant(tmp_path) is False
    (tmp_path / "RUNTIME.json").write_text(json.dumps({"evolver_daytona": {"grader_clean": True}}))
    assert de.evolver_grader_grant(tmp_path) is True
    for bad in ({"evolver_daytona": {"grader_clean": "true"}}, {"evolver_daytona": ["grader_clean"]}):
        (tmp_path / "RUNTIME.json").write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            de.evolver_grader_grant(tmp_path)
    assert de.GRADER_GRANT not in de.GRANT_KEYS, "the two-key grant records must keep their meaning"


def test_verify_grader_dir_and_every_planted_violation(tmp_path, monkeypatch):
    f = _fake(tmp_path, monkeypatch)
    g = _grader(f.rd, f.split)
    c = de.verify_grader_dir(f.rd, f.split)
    assert c.status == "PASS", c.detail
    assert de.verify_grader_dir(tmp_path / "nowhere", f.split).status == "SKIP"
    cases = {
        "workbook": lambda: (g / "ref.xlsx").write_bytes(b"x"),
        "unlisted": lambda: (g / "extra.py").write_text("x = 1\n"),
        "edited": lambda: (g / "scorer.py").write_text("tampered\n"),
        "held-out id": lambda: (g / "docker_scripts/scan.py").write_text("# Template/02_01\n"),
        "dataset bytes": lambda: (g / "docker_scripts/scan.py").write_bytes(b"GOLDEN 02_01"),
        "symlink": lambda: (g / "link.py").symlink_to(g / "scorer.py"),
        "other split": lambda: (g / "MANIFEST.json").write_text(
            (g / "MANIFEST.json").read_text().replace(str(f.split), "/x/split.json")),
        "manifest": lambda: (g / "MANIFEST.json").write_text("not json"),
    }
    for label, plant in cases.items():
        shutil.rmtree(g)
        g = _grader(f.rd, f.split)
        if label in ("held-out id", "dataset bytes"):
            man = json.loads((g / "MANIFEST.json").read_text())
            plant()
            for e in man["files"]:
                e["clean_sha256"] = hashlib.sha256((g / e["file"]).read_bytes()).hexdigest()
            (g / "MANIFEST.json").write_text(json.dumps(man))
        else:
            plant()
        c = de.verify_grader_dir(f.rd, f.split)
        assert c.status == "FAIL", f"{label}: {c.detail}"


def _grader_manifest(f, *, grader=True, references=False):
    rt = {"evolver_daytona": {"train_references": references, "grader_clean": grader}}
    (f.rd / "RUNTIME.json").write_text(json.dumps(rt))
    g = de.evolver_grants(f.rd)
    gg = de.evolver_grader_grant(f.rd)
    if references:
        de.build_references_dir(f.rd, f.split)
    venv = {"HOME": de.VM_HOME}
    rows, _gen = de.build_upload_manifest(
        run_dir=f.rd, epoch=1, prompt=f.rd / "ITER1_PROMPT.md", evidence_dirs=[f.ev],
        oracle=f.rd / "oracle", staging=f.rd / "skills_v1_staging", prev_library=f.rd / "skills_v0",
        references=de.references_dir(f.rd) if references else None, grants=g,
        grader=de.grader_dir(f.rd) if grader else None, grader_granted=gg,
        env_text=de.render_env_file(venv), hooks_text=de.hook_settings_text(), split=f.split)
    receipt = {"containment": "daytona", "sandboxed": True, "upload_manifest": rows, "grants": dict(g),
               "grader_granted": gg, "evidence_sha256_before": es.hashes([f.ev]),
               "daytona": {"vm_env": venv},
               "references_sha256_before": es.dir_sha256(de.references_dir(f.rd)) if references else None,
               "grader_sha256_before": es.dir_sha256(de.grader_dir(f.rd)) if grader else None}
    listing = {"pre_upload": {"users_exists": False},
               "after_upload": [[r["vm_path"], "F", r["sha256"], r["bytes"]] for r in rows]}
    return receipt, listing


def _check(f, receipt, listing, **kw):
    return de.check_upload_manifest(run_dir=f.rd, epoch=1, split=f.split, evidence_dirs=[f.ev],
                                    prev_library=f.rd / "skills_v0", prompt=f.rd / "ITER1_PROMPT.md",
                                    oracle=f.rd / "oracle", staging=f.rd / "skills_v1_staging",
                                    receipt=receipt, listing=listing, **kw)


def test_upload_carries_the_grader_only_under_its_grant(tmp_path, monkeypatch):
    f = _fake(tmp_path, monkeypatch)
    _grader(f.rd, f.split)
    receipt, listing = _grader_manifest(f, grader=True, references=True)
    roles = [r["role"] for r in receipt["upload_manifest"]]
    assert roles.count("grader") == 4 and roles.count("train_reference") == 3
    c = _check(f, receipt, listing)
    assert c.status == "PASS", c.detail
    assert c.evidence["grader_granted"] is True
    assert c.evidence["grants"] == {"train_inputs": False, "train_references": True}
    # grant withdrawn in RUNTIME while the grader was on the box
    (f.rd / "RUNTIME.json").write_text(json.dumps({"evolver_daytona": {"train_references": True}}))
    c = _check(f, receipt, listing)
    assert c.status == "FAIL" and "does not grant grader_clean" in c.detail
    assert "receipt records grader_clean=True" in c.detail
    (f.rd / "RUNTIME.json").write_text(json.dumps({"evolver_daytona": {"train_references": True,
                                                                       "grader_clean": True}}))
    # a grader file dropped from the upload
    x, lx = _cp(receipt), _cp(listing)
    drop = next(r for r in x["upload_manifest"] if r["role"] == "grader")
    x["upload_manifest"].remove(drop)
    lx["after_upload"] = [e for e in lx["after_upload"] if e[0] != drop["vm_path"]]
    c = _check(f, x, lx)
    assert c.status == "FAIL" and "grader rows != the whole grader_clean dir" in c.detail
    # the grader dir changed after dispatch
    (de.grader_dir(f.rd) / "README.md").write_text("changed\n")
    c = _check(f, receipt, listing)
    assert c.status == "FAIL" and "source bytes changed since upload" in c.detail
    # a grader file travelling under another role is refused at build time
    kw = dict(run_dir=f.rd, epoch=1, prompt=f.rd / "ITER1_PROMPT.md", evidence_dirs=[f.ev],
              oracle=f.rd / "oracle", staging=f.rd / "skills_v1_staging", prev_library=f.rd / "skills_v0",
              env_text="", hooks_text="", split=f.split)
    rows = [de._row(str(de.grader_dir(f.rd) / "scorer.py"), "evidence", str(de.grader_dir(f.rd) / "scorer.py"))]
    with pytest.raises(ValueError, match="grader_clean"):
        de._refuse_forbidden_sources(rows, f.split)
    with pytest.raises(ValueError, match="grader"):
        de._refuse_forbidden_sources([de._row(str(f.rd / "oracle/README.md"), "grader",
                                              str(f.rd / "oracle/README.md"))], f.split)
    with pytest.raises(ValueError):                             # granted but omitted
        de.build_upload_manifest(grader_granted=True, **kw)
    with pytest.raises(ValueError):                             # passed but not granted
        de.build_upload_manifest(grader=de.grader_dir(f.rd), grader_granted=False, **kw)
    with pytest.raises(ValueError):                             # not this run's dir
        de.build_upload_manifest(grader=f.ev, **kw)


def test_upload_without_the_grader_grant_is_unchanged(tmp_path, monkeypatch):
    f = _fake(tmp_path, monkeypatch)
    _grader(f.rd, f.split)
    receipt, listing = _grader_manifest(f, grader=False, references=False)
    assert "grader" not in {r["role"] for r in receipt["upload_manifest"]}
    c = _check(f, receipt, listing)
    assert c.status == "PASS", c.detail
    assert c.evidence["grader_granted"] is False


def test_receipt_hashes_the_grader_before_and_after(tmp_path):
    led, lg = tmp_path / "ledger.log", _log(tmp_path)
    r = _good_receipt(tmp_path, led, lg)
    r.update(grader_dir="/data/x/grader_clean", grader_sha256_before="g", grader_sha256_after_vm="g")
    ok, det = de.check_daytona_receipt(r, log=lg, ledger_path=led)
    assert ok, det
    x = _cp(r)
    x["grader_sha256_after_vm"] = "changed"
    ok, det = de.check_daytona_receipt(x, log=lg, ledger_path=led)
    assert not ok and "clean grader changed on the VM" in det
    x = _cp(r)
    x.pop("grader_sha256_after_vm")
    assert not de.check_daytona_receipt(x, log=lg, ledger_path=led)[0]


# ══ the naming check in template mode ══════════════════════════════════════
def test_prompt_names_exactly_the_granted_dirs(tmp_path, monkeypatch):
    f = _fake(tmp_path, monkeypatch)
    (f.rd / "RUNTIME.json").write_text(json.dumps({"evolver_daytona": {"train_references": True,
                                                                       "grader_clean": True}}))
    pr = f.rd / "ITER1_PROMPT.md"
    ctx = _ctx(run_dir=f.rd, paths=SimpleNamespace(prompt=pr))
    refs, grader, inputs = (str(de.references_dir(f.rd)), str(de.grader_dir(f.rd)),
                            str(de.inputs_dir(f.rd)))
    good = (f"The grader's code is in `{grader}`. The references are in `{refs}`, laid out as "
            f"`{refs}/<evidence file name without .md>/<reference workbook file>`.\n")
    pr.write_text(good)
    ok, d = L._v_prompt_names_granted_dirs(ctx)
    assert ok, d
    for bad, needle in (
            (f"The references are in `{refs}`.\n", "never names the granted clean grader dir"),
            (good + f"Inputs: `{inputs}`.\n", "which RUNTIME does not grant"),
            (good + "see evolver_references_train/ too\n", "without this run's absolute path"),
            (good + f"`{refs}/Template_01_01`\n", "subpath below the references dir"),
            (good + "Template_01_01_golden.xlsx\n", "dataset/golden path"),
            (good + f"`{refs}_old`\n", "sibling")):
        pr.write_text(bad)
        ok, d = L._v_prompt_names_granted_dirs(ctx)
        assert not ok and needle in d, (needle, d)


# ── (a) the conversation step in template mode ───────────────────────────────────────────────────
def _conv(rd: Path, *turns):
    from src.orchestrator import conversation
    for sp, ch, ep in turns:
        conversation.record(rd, speaker=sp, channel=ch, content=f"{sp}/{ch}", epoch=ep)
    return conversation.path_for(rd)


def test_template_conversation_step_checks_what_is_true_not_the_ao_exchange(tmp_path):
    rd = tmp_path / "run"
    rd.mkdir()
    ctx = L.Ctx(rd, 1, SimpleNamespace(), rd / "oracle", prompt_mode="template", evolver_host="daytona")
    ok, d = L._v_operator_turns_recorded(ctx)
    assert not ok and "NO TRANSCRIPT" in d
    ctx.readonly = True
    ok, d = L._v_operator_turns_recorded(ctx)
    assert ok and d.startswith(L.UNMEASURED)
    ctx.readonly = False
    # THE WRONG REASON (review #6): the epoch-less intake turn + the loop's own "ao" report turn make the
    # AO-mode check pass although no AO spoke this epoch and no operator turn exists for it.
    _conv(rd, ("operator", "intake", None), ("ao", "report", 1))
    assert L._v_conversation_recorded(ctx)[0], "the AO-mode check passes for the wrong reason"
    ok, d = L._v_operator_turns_recorded(ctx)
    assert not ok and "no operator turn is recorded for epoch 1" in d
    _conv(rd, ("operator", "invocation", 1))
    ok, d = L._v_operator_turns_recorded(ctx)
    assert ok and "does not apply" in d and "1 operator turn(s)" in d and "invocation" in d
    assert not d.startswith(L.UNMEASURED)
    _conv(rd, ("ao", "ask", 1), ("ao", "report", 1))          # the loop's own turns: not an AO
    assert L._v_operator_turns_recorded(ctx)[0]
    _conv(rd, ("ao", "authoring", 2))                           # another epoch's: not this one's
    assert L._v_operator_turns_recorded(ctx)[0]
    _conv(rd, ("ao", "authoring", 1))
    ok, d = L._v_operator_turns_recorded(ctx)
    assert not ok and "AO AUTHORING turn" in d
    p = _conv(tmp_path / "run2")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"speaker": "operator", "channel": "invocation", "epoch": 1, "content": "x", "ts_ns": 1}\nnot json\n')
    ok, d = L._v_operator_turns_recorded(L.Ctx(tmp_path / "run2", 1, SimpleNamespace(), rd))
    assert not ok and "unparseable" in d


FAKE_DRIVER = r'''
import hashlib, json, os, sys
from pathlib import Path
a = sys.argv[1:]
assert a[0] == "run", a
kv = {a[i]: a[i + 1] for i in range(1, len(a) - 1, 2)}
here = Path(__file__).parent
beh = json.loads((here / "behaviour.json").read_text())
out = Path(kv["--out-dir"]); out.mkdir(parents=True, exist_ok=True)
rp = out / "REGRADE.json"
disp = sorted((json.loads(rp.read_text()).get("dispatched") or {})) if rp.is_file() else None
with open(here / "calls.jsonl", "a") as fh:
    fh.write(json.dumps({"argv": a, "env_keys": sorted(os.environ), "dispatched_at_launch": disp}) + "\n")
res = out / "results.jsonl"
have = {json.loads(l)["job_id"] for l in res.read_text().splitlines()} if res.is_file() else set()
jobs = [json.loads(l) for l in Path(kv["--jobs"]).read_text().splitlines() if l.strip()]
with open(res, "a") as fh:
    for j in jobs:
        if j["job_id"] in have or j["job_id"] in beh.get("skip", []):
            continue
        b = beh.get("jobs", {}).get(j["job_id"], beh.get("default", {}))
        sha = hashlib.sha256(Path(j["output"]).read_bytes()).hexdigest()
        rec = {"job_id": j["job_id"], "mode": "grade", "task_id": j["task_id"], "status": b.get("status", "ok"),
               "worker": b.get("worker", "grade_worker/1"),
               "d4_failure_warnings": b.get("d4", []), "scorer_exception_warnings": b.get("scorer_exc", [])}
        # the driver's host block: the sandbox the line was merged from (grade_worker/1 + ssb_grade_daytona)
        host = {"sandbox_id": b.get("sandbox", "sb-fake-0001"), "run": "20260925T000000", "label": kv["--label"]}
        if rec["status"] in ("ok", "exception"):            # the child ran: its sha checks and the scorer
            rec.update(output_sha256_before=b.get("sha", sha), originals_unchanged=True,
                       gt_sha256_before=b.get("gt", j.get("gt_sha256")),
                       input_sha256_before=b.get("input", j.get("input_sha256")),
                       scorer_sha256=b.get("scorer", "2442e5aa0e6172dbf62be5849a00fedf1b53e390d093d65ac7bb19da9b807f81"))
        if rec["status"] == "ok":
            p = b.get("passed", False)
            rec.update(passed=p, n_match=2 if p else 0, n_total=2)
            host.update(outcome=b.get("host", "passed" if p else "graded_wrong"), outcome_reason="x")
        else:
            host.update(outcome="grader_failed", outcome_reason=rec["status"])
            if rec["status"] == "exception":
                rec["exception"] = {"type": "RuntimeError", "message": "recalc died"}
        rec["host"] = {k: v for k, v in host.items() if v is not None}
        for k in b.get("drop", []):
            rec.pop(k, None)
        fh.write(json.dumps(rec) + "\n")
sys.exit(beh.get("rc", 0))
'''


def _fake_sources(task, run_dir=None):
    """L._regrade_sources without the dataset: a stable reference / input sha per task."""
    h = lambda x: hashlib.sha256(x.encode()).hexdigest()           # noqa: E731
    return {"gt_file": f"/data/{task}_golden.xlsx", "gt_sha256": h("gt:" + task),
            "input_file": f"/data/{task}_init.xlsx", "input_sha256": h("in:" + task)}


@pytest.fixture
def rg(tmp_path, monkeypatch):
    """A run dir with an epoch-1 train leg whose records (outcomes.leg_outcome_counts, faked) carry
    grader-failed samples with saved workbooks, and a FAKE grading-sandbox driver speaking the real
    driver's CLI (run --jobs --out-dir --run-dir --label ...; results.jsonl, first line per job wins)."""
    from src.orchestrator import outcomes
    rd = tmp_path / "ssbv2_rg_run"
    rd.mkdir()
    ctx = L.Ctx(rd, 1, SimpleNamespace(), rd / "oracle", prompt_mode="template", evolver_host="daytona")
    leg = ctx.paths.leg("train")
    leg.write_text(json.dumps({"tag": leg.stem, "per_task": {}}))
    drv = tmp_path / "fake_driver" / "ssb_grade_daytona.py"
    drv.parent.mkdir()
    drv.write_text(FAKE_DRIVER)
    monkeypatch.setattr(L, "GRADE_DRIVER", drv)
    monkeypatch.setenv("EVOSKILL_PYTHON", sys.executable)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-" + "0" * 20)       # must NOT reach the driver
    state = {"samples": []}

    def fake_counts(run_dir, leg_json, **kw):
        smp = [dict(x) for x in state["samples"]]
        c = {o: sum(1 for x in smp if x["outcome"] == o) for o in outcomes.OUTCOMES}
        rp = sum(1 for x in smp if (x["recorded"] or 0) >= 0.999)
        bad = c["grader_failed"] > 0
        return {**c, "n_episodes": len(smp), "samples": smp, "mismatches": [], "problems": [],
                "unattributed_warnings": [], "recorded_pass": rp, "recorded_sample_pass": rp,
                "pass_matches_recorded": c["passed"] == rp, "status": "grader_failed" if bad else "ok",
                "reportable": not bad}
    monkeypatch.setattr(outcomes, "leg_outcome_counts", fake_counts)
    monkeypatch.setattr(L, "_regrade_sources", _fake_sources)

    def samples(*rows):
        """rows: (task, k, outcome, recorded, has_workbook)"""
        state["samples"] = [{"task": t, "sample": k, "outcome": o, "reason": "failure_warning" if o ==
                             "grader_failed" else "fail", "recorded": rec, "warnings": [],
                             "snapshot": _snap(tmp_path / "snaps", t, k) if wb else None}
                            for t, k, o, rec, wb in rows]

    def behave(**b):
        (drv.parent / "behaviour.json").write_text(json.dumps(b))

    def calls():
        f = drv.parent / "calls.jsonl"
        return [json.loads(x) for x in f.read_text().splitlines()] if f.is_file() else []
    behave()
    return SimpleNamespace(rd=rd, ctx=ctx, leg=leg, samples=samples, behave=behave, calls=calls,
                           step=next(s for s in L._template_steps(L.epoch_steps())
                                     if s.name == "train leg has no grader failure"))


def _advance_one(rg):
    rg.ctx._asking_step = ""
    return L.advance(rg.ctx, steps=[rg.step])


def test_regrade_runs_once_through_the_driver_and_the_counts_apply(rg):
    rg.samples(("Template/01_01", 0, "graded_wrong", 0.0, True), ("Template/01_01", 1, "passed", 1.0, True),
               ("Template/02_01", 0, "grader_failed", 0.0, True), ("Template/03_01", 2, "grader_failed", 1.0, True),
               ("Debugging/01_01", 0, "not_eligible", 0.0, False))
    rg.behave(jobs={"Template/02_01/s0/regrade1": {"passed": False, "scorer_exc": ["restore_open_failed"]},
                    "Template/03_01/s2/regrade1": {"passed": True}})
    ok, d = L._v_leg_outcomes(rg.ctx, "train")
    assert not ok and "RE-GRADE NEEDED" in d and "2 sample(s) were GRADER FAILED" in d
    step, det = _advance_one(rg)                            # runner = ONE driver run, then the verify
    assert step is rg.step and "after the re-grade" in det, det
    assert "passed 2, graded and wrong 2, not eligible to grade 1, grader failed 0 (n=5)" in det
    assert "1 of the re-grades logged scorer exception warnings, which are NOT counted" in det
    (call,) = rg.calls()
    a = call["argv"]
    kv = {a[i]: a[i + 1] for i in range(1, len(a) - 1, 2)}
    assert kv["--run-dir"] == str(rg.rd.resolve()) and kv["--label"] == "rg-v1tr"
    assert kv["--wall-min"] == str(L.REGRADE_WALL_MIN) and kv["--max-usd"] == str(L.REGRADE_MAX_USD)
    assert not [k for k in call["env_keys"] if "API_KEY" in k or "TOKEN" in k], "a key reached the driver"
    d_ = L._regrade_dir(rg.ctx, rg.leg)
    assert d_.parent.parent == L.operator_dir(rg.rd), "re-grade results live in the operator dir"
    jobs = [json.loads(x) for x in (d_ / "jobs.jsonl").read_text().splitlines()]
    assert sorted(j["job_id"] for j in jobs) == ["Template/02_01/s0/regrade1", "Template/03_01/s2/regrade1"]
    assert all(j["mode"] == "grade" and j["output"].endswith("_in.xlsx") and "/output_raw/" in j["output"]
               for j in jobs)
    assert all(j["gt_sha256"] == _fake_sources(j["task_id"])["gt_sha256"]
               and j["input_sha256"] == _fake_sources(j["task_id"])["input_sha256"] for j in jobs)
    rec = json.loads((d_ / L.REGRADE_RECORD).read_text())
    assert rec["schema"] == L.REGRADE_SCHEMA and rec["attempts_per_sample"] == 1
    assert [i["rc"] for i in rec["invocations"]] == [0]
    # both jobs were on disk as DISPATCHED when the driver started
    assert call["dispatched_at_launch"] == ["Template/02_01/s0/regrade1", "Template/03_01/s2/regrade1"]
    assert {k: v["invocation"] for k, v in rec["dispatched"].items()} == {
        "Template/02_01/s0/regrade1": 1, "Template/03_01/s2/regrade1": 1}
    # ONCE: every later check reads the same results; no second sandbox is ever started
    assert L._v_leg_outcomes(rg.ctx, "train")[0]
    assert _advance_one(rg) == (None, "every step verifies — nothing to do")
    L._run_leg_regrade(rg.ctx, "train")
    assert len(rg.calls()) == 1
    st = L._leg_outcomes_after_regrade(rg.ctx, rg.leg, "train leg")
    assert st["counts"]["regraded"] == 2 and st["counts"]["grader_failed"] == 0 and st["counts"]["reportable"]


def test_regrade_that_fails_again_halts_and_is_never_repeated(rg):
    rg.samples(("Template/02_01", 0, "grader_failed", 0.0, True), ("Template/03_01", 0, "graded_wrong", 0.0, True))
    rg.behave(default={"status": "exception"})
    with pytest.raises(L.Halt, match="RAN, and its verify still fails.*RE-GRADE FAILED AGAIN"):
        _advance_one(rg)
    with pytest.raises(L.Halt, match="RE-GRADE FAILED AGAIN"):
        _advance_one(rg)                                    # the runner now HALTs itself
    assert len(rg.calls()) == 1, "a failed re-grade is never re-run"
    for status, needle in (("timeout", "timeout"), ("crashed", "crashed")):
        rg.leg.write_text(json.dumps({"tag": rg.leg.stem, "variant": status}))   # a new leg -> a new dir
        rg.behave(default={"status": status})
        with pytest.raises(L.Halt, match="RE-GRADE FAILED AGAIN"):
            _advance_one(rg)
    rg.leg.write_text(json.dumps({"tag": rg.leg.stem, "variant": "d4"}))
    rg.behave(default={"passed": False, "d4": ["LibreOffice recalculation failed for x"]})
    with pytest.raises(L.Halt, match="RE-GRADE FAILED AGAIN.*failure_warning"):
        _advance_one(rg)


def test_regrade_that_changes_a_recorded_verdict_halts(rg):
    """A flip HALTs for a person. The message names every flipped sample, both verdicts, and the operator's
    two choices."""
    rg.samples(("Template/02_01", 0, "grader_failed", 0.0, True), ("Template/05_01", 2, "grader_failed", 1.0, True))
    rg.behave(jobs={"Template/02_01/s0/regrade1": {"passed": True},
                    "Template/05_01/s2/regrade1": {"passed": False}})
    with pytest.raises(L.Halt, match="CHANGES the recorded verdict"):
        _advance_one(rg)
    ok, d = L._v_leg_outcomes(rg.ctx, "train")
    assert not ok and "never rewrites a recorded verdict" in d
    assert ("Template/02_01 sample 0: recorded verdict FAIL (recorded score 0.0; first grade failure_warning), "
            "re-grade verdict PASS (passed, job Template/02_01/s0/regrade1)") in d, d
    assert ("Template/05_01 sample 2: recorded verdict PASS (recorded score 1.0; first grade failure_warning), "
            "re-grade verdict FAIL (graded_wrong, job Template/05_01/s2/regrade1)") in d, d
    assert "of 2 sample(s)" in d and "THE OPERATOR DECIDES" in d
    assert "a correction of the recorded verdict written by a person" in d and f"re-run of the leg {rg.leg.name}" in d
    assert "results.jsonl" in d and "no second re-grade is started" in d
    with pytest.raises(L.Halt, match="THE OPERATOR DECIDES"):
        L._run_leg_regrade(rg.ctx, "train")
    assert len(rg.calls()) == 1


def test_regrade_without_a_saved_workbook_halts_before_any_sandbox(rg):
    rg.samples(("Template/02_01", 0, "grader_failed", 0.0, False))
    with pytest.raises(L.Halt, match="cannot be re-graded.*no saved workbook"):
        _advance_one(rg)
    assert rg.calls() == []
    # two files in output_raw is not "the" saved workbook either (critique #7)
    rg.samples(("Template/02_01", 0, "grader_failed", 0.0, True))
    snap = Path(L._leg_outcomes_after_regrade(rg.ctx, rg.leg, "x")["rows"][0]["output"]).parent
    (snap / "second.xlsx").write_bytes(b"x")
    with pytest.raises(L.Halt, match="holds 2 file"):
        L._run_leg_regrade(rg.ctx, "train")
    assert rg.calls() == []


def test_regrade_result_of_other_bytes_is_invalid_not_a_verdict(rg):
    rg.samples(("Template/02_01", 0, "grader_failed", 0.0, True))
    rg.behave(default={"passed": False, "sha": "0" * 64})
    with pytest.raises(L.Halt, match="not a grade of the saved workbook"):
        _advance_one(rg)
    assert len(rg.calls()) == 1


def _good_line(task="Template/02_01", out_sha="o" * 64, **over):
    """A re-grade line as grade_worker/1 writes it and ssb_grade_daytona merges it (host block)."""
    src = _fake_sources(task)
    r = {"job_id": f"{task}/s0/regrade1", "mode": "grade", "task_id": task, "status": "ok",
         "worker": "grade_worker/1", "output_sha256_before": out_sha, "gt_sha256_before": src["gt_sha256"],
         "input_sha256_before": src["input_sha256"], "originals_unchanged": True,
         "scorer_sha256": L.GRADE_SCORER_SHA256, "passed": False, "n_match": 0, "n_total": 2,
         "d4_failure_warnings": [], "host": {"sandbox_id": "sb-1", "outcome": "graded_wrong", "outcome_reason": "fail"}}
    r.update(over)
    return r


def test_regrade_outcome_requires_the_validators_provenance():
    """A line counts only if the grading sandbox's worker graded it with the
    pinned scorer, the driver merged it from a named sandbox, and the reference and input it was graded
    against hash to the job's. Each missing piece makes it INVALID (a HALT), never a verdict."""
    want = _fake_sources("Template/02_01")
    oc = lambda r: L._regrade_outcome("Template/02_01", "o" * 64, r, want)        # noqa: E731
    assert oc(_good_line()) == ("graded_wrong", "fail")
    assert oc(_good_line(passed=True, host={"sandbox_id": "sb-1", "outcome": "passed", "outcome_reason": "p"}))[0] \
        == "passed"
    for over, needle in (({"worker": "grade_worker/0"}, "worker 'grade_worker/0'"),
                         ({"worker": None}, "not the grading sandbox's grade_worker/1"),
                         ({"host": {"outcome": "graded_wrong"}}, "no host.sandbox_id"),
                         ({"host": None}, "no host.sandbox_id"),
                         ({"scorer_sha256": "f" * 64}, "scorer sha256 ffffffffffff, not the pinned 2442e5aa0e61"),
                         ({"scorer_sha256": None}, "not the pinned"),
                         ({"gt_sha256_before": "1" * 64}, "against a reference hashing to 111111111111"),
                         ({"gt_sha256_before": None}, "against a reference"),
                         ({"input_sha256_before": "2" * 64}, "against a input hashing to 222222222222"),
                         ({"output_sha256_before": "3" * 64}, "graded bytes 333333333333")):
        got = oc(_good_line(**over))
        assert got[0] == "invalid" and needle in got[1], (over, got)
    assert L._regrade_outcome("Template/02_01", "o" * 64, _good_line(), None)[0] == "invalid", \
        "no stated job sources: nothing to compare against, never a verdict"
    # a timed-out / crashed line comes from the worker's parent: worker + sandbox, no child fields
    assert oc({"job_id": "x", "mode": "grade", "task_id": "Template/02_01", "status": "timeout",
               "worker": "grade_worker/1", "host": {"sandbox_id": "sb-1"}}) == ("grader_failed", "timeout")
    assert oc({"job_id": "x", "mode": "grade", "task_id": "Template/02_01", "status": "crashed",
               "worker": "grade_worker/1", "host": {}})[0] == "invalid"
    # the scorer never imported (it raised at import): no sha, still grader failed; another scorer: invalid
    exc = _good_line(status="exception", exception={"type": "ImportError"})
    exc.pop("scorer_sha256")
    for k in ("passed", "n_match", "n_total"):
        exc.pop(k)
    assert oc(exc)[0] == "grader_failed"
    assert oc({**exc, "scorer_sha256": "f" * 64})[0] == "invalid"


def test_regrade_line_without_provenance_halts_as_invalid_and_is_not_resent(rg):
    for i, beh in enumerate(({"worker": "hand_run"}, {"sandbox": None}, {"scorer": "e" * 64},
                             {"gt": "9" * 64}, {"input": "8" * 64})):
        rg.leg.write_text(json.dumps({"tag": rg.leg.stem, "variant": i}))          # a new leg -> a new dir
        rg.samples(("Template/02_01", 0, "grader_failed", 0.0, True))
        rg.behave(default={"passed": False, **beh})
        with pytest.raises(L.Halt, match="not a grade of the saved workbook by the grading sandbox"):
            _advance_one(rg)
        with pytest.raises(L.Halt, match="not a grade of the saved workbook"):
            L._run_leg_regrade(rg.ctx, "train")
        assert len(rg.calls()) == i + 1


def test_grade_provenance_constants_are_the_workers_and_the_pinned_scorer():
    import re as _re
    src = (REPO / "scripts/daytona/grade_worker.py").read_text()
    assert _re.search(r'^WORKER_VERSION = "([^"]+)"', src, _re.M).group(1) == L.GRADE_WORKER
    leg = (REPO / "scripts/daytona/ssb_leg_daytona_loop.py").read_text()
    assert f'"{L.GRADE_SCORER_SHA256}"' in leg.split('"scorer_sha256"', 1)[1][:200]
    assert hashlib.sha256((REPO / "src/agent_profiles/ssbv2_agent/scorer.py").read_bytes()).hexdigest() \
        == L.GRADE_SCORER_SHA256


def test_regrade_sources_follow_infra_data_and_the_overlay(tmp_path, monkeypatch):
    """_regrade_sources states the job's reference / input shas by the grade driver's rule: INFRA.json's
    "data" block (root; overlay, null = none), else data_root() and the bare overlay variable."""
    from src.orchestrator import data_overlay, infra
    root = tmp_path / "data"
    (root / "Cat").mkdir(parents=True)
    (root / "Cat" / "dataset.json").write_text(json.dumps([{"id": "01_01", "spreadsheet_path": "01_01_init.xlsx",
                                                          "golden_response_path": "01_01_golden.xlsx",
                                                          "answer_position": "A1"}]))
    (root / "Cat" / "01_01_init.xlsx").write_bytes(b"broken-input")
    (root / "Cat" / "01_01_golden.xlsx").write_bytes(b"golden")
    fixed = tmp_path / "patched"
    (fixed / "Cat").mkdir(parents=True)
    (fixed / "Cat" / "01_01_init.xlsx").write_bytes(b"repaired-input")
    h = lambda b: hashlib.sha256(b).hexdigest()                         # noqa: E731
    pj = tmp_path / "PATCHES.json"
    pj.write_text(json.dumps({"src": str(root), "dst": str(fixed), "files": [
        {"path": "Cat/01_01_init.xlsx", "sha256_before": h(b"broken-input"), "sha256_after": h(b"repaired-input")}]}))
    monkeypatch.setenv("EVOSKILL_SSBV2_DATA", str(root))
    monkeypatch.delenv(data_overlay.ENV, raising=False)
    rd = tmp_path / "run"
    rd.mkdir()
    s = L._regrade_sources("Cat/01_01", rd)                             # no INFRA.json, no overlay
    assert s["input_sha256"] == h(b"broken-input") and s["gt_sha256"] == h(b"golden")
    assert s["input_file"] == str(root / "Cat" / "01_01_init.xlsx")
    monkeypatch.setenv(data_overlay.ENV, str(pj))                       # the bare variable
    assert L._regrade_sources("Cat/01_01", rd)["input_sha256"] == h(b"repaired-input")
    with pytest.raises(ValueError, match="not a task of the dataset"):
        L._regrade_sources("Cat/99_01", rd)
    # INFRA.json decides when the run has one (load_run faked: the rest of INFRA is not under test here)
    (rd / "INFRA.json").write_text("{}")
    blk = {"root": str(root), "overlay_patches_json": None, "overlay_patches_sha256": None}
    monkeypatch.setattr(infra, "load_run", lambda r: {"data": dict(blk)})
    assert L._regrade_sources("Cat/01_01", rd)["input_sha256"] == h(b"broken-input"), \
        "INFRA declares no overlay: the bare variable does not substitute bytes"
    blk.update(overlay_patches_json=str(pj), overlay_patches_sha256=h(pj.read_bytes()))
    monkeypatch.delenv(data_overlay.ENV)
    assert L._regrade_sources("Cat/01_01", rd)["input_sha256"] == h(b"repaired-input")
    blk.update(overlay_patches_sha256="0" * 64)
    with pytest.raises(ValueError, match="does not hash to INFRA.json data.overlay_patches_sha256"):
        L._regrade_sources("Cat/01_01", rd)

    def bad(r):
        raise infra.InfraError(["data.root must be absolute"])
    monkeypatch.setattr(infra, "load_run", bad)
    with pytest.raises(ValueError, match="INFRA.json is invalid"):
        L._regrade_sources("Cat/01_01", rd)


def test_regrade_driver_failure_halts_and_a_dispatched_job_without_result_is_never_regraded(rg):
    """The driver dies after the sandbox graded a job but before its line was merged. The
    job was recorded as dispatched before launch, so the next invocation HALTs instead of grading it again."""
    rg.samples(("Template/02_01", 0, "grader_failed", 0.0, True), ("Template/04_01", 1, "grader_failed", 0.0, True))
    rg.behave(rc=5, skip=["Template/04_01/s1/regrade1"])
    with pytest.raises(L.Halt, match="re-grade exited 5 \\(incomplete\\).*recorded as dispatched"):
        _advance_one(rg)
    ok, d = L._v_leg_outcomes(rg.ctx, "train")
    assert not ok and L.REGRADE_LOST in d and "Template/04_01 s1 (Template/04_01/s1/regrade1)" in d, d
    assert "invocation 1: driver exit 5 (incomplete)" in d and "RE-GRADE NEEDED" not in d
    rg.behave()                                             # a driver that WOULD now grade it
    with pytest.raises(L.Halt, match="dispatched, no result: a lost re-grade may not be retried silently"):
        _advance_one(rg)
    with pytest.raises(L.Halt, match="DISPATCHED and have NO RESULT"):
        L._run_leg_regrade(rg.ctx, "train")
    assert len(rg.calls()) == 1, "a dispatched job is never handed to the driver again"
    d_ = L._regrade_dir(rg.ctx, rg.leg)
    res = [json.loads(x)["job_id"] for x in (d_ / "results.jsonl").read_text().splitlines()]
    assert res == ["Template/02_01/s0/regrade1"]
    assert [i["rc"] for i in json.loads((d_ / L.REGRADE_RECORD).read_text())["invocations"]] == [5]


def test_regrade_lost_after_a_clean_driver_exit_halts_too(rg):
    """Exit 0 but a line missing (the sync lost it): the runner's own verify HALTs, nothing is re-sent."""
    rg.samples(("Template/02_01", 0, "grader_failed", 0.0, True))
    rg.behave(skip=["Template/02_01/s0/regrade1"])
    with pytest.raises(L.Halt, match="RAN, and its verify still fails.*dispatched, no result"):
        _advance_one(rg)
    with pytest.raises(L.Halt, match="dispatched, no result"):
        _advance_one(rg)
    assert len(rg.calls()) == 1


def test_regrade_dispatches_only_what_was_never_dispatched(rg):
    """A second grader-failed sample appearing for the SAME leg bytes is dispatched on its own; the first
    (dispatched and answered) is not sent again, and both stay in the dispatch record."""
    rg.samples(("Template/02_01", 0, "grader_failed", 0.0, True))
    step, _ = _advance_one(rg)
    assert step is rg.step
    rg.samples(("Template/02_01", 0, "grader_failed", 0.0, True), ("Template/06_01", 1, "grader_failed", 0.0, True))
    ok, d = L._v_leg_outcomes(rg.ctx, "train")
    assert not ok and "1 of them have no re-grade yet" in d
    step, det = _advance_one(rg)
    assert step is rg.step and "2 grader-failed sample(s) re-graded ONCE" in det, det
    c1, c2 = rg.calls()
    assert c1["dispatched_at_launch"] == ["Template/02_01/s0/regrade1"]
    assert c2["dispatched_at_launch"] == ["Template/02_01/s0/regrade1", "Template/06_01/s1/regrade1"]
    d_ = L._regrade_dir(rg.ctx, rg.leg)
    assert [json.loads(x)["job_id"] for x in (d_ / "jobs.jsonl").read_text().splitlines()] == \
        ["Template/06_01/s1/regrade1"]
    rec = json.loads((d_ / L.REGRADE_RECORD).read_text())
    assert {k: v["invocation"] for k, v in rec["dispatched"].items()} == {
        "Template/02_01/s0/regrade1": 1, "Template/06_01/s1/regrade1": 2}
    assert sorted(x["job_id"] for x in rec["samples"]) == sorted(rec["dispatched"])


def test_regrade_record_unreadable_or_from_before_the_dispatch_list(rg):
    rg.samples(("Template/02_01", 0, "grader_failed", 0.0, True))
    d_ = L._regrade_dir(rg.ctx, rg.leg)
    d_.mkdir(parents=True)
    (d_ / L.REGRADE_RECORD).write_text("{torn")
    with pytest.raises(L.Halt, match="REGRADE.json is unreadable, so which re-grades were dispatched"):
        _advance_one(rg)
    assert rg.calls() == []
    # a record the step-2 loop wrote (no "dispatched"; its samples were written before its launch)
    (d_ / L.REGRADE_RECORD).write_text(json.dumps({
        "schema": L.REGRADE_SCHEMA, "invocations": [{"rc": None}],
        "samples": [{"task": "Template/02_01", "sample": 0, "job_id": "Template/02_01/s0/regrade1"}]}))
    with pytest.raises(L.Halt, match="dispatched, no result"):
        _advance_one(rg)
    assert rg.calls() == []


def test_regrade_driver_that_cannot_start_leaves_its_jobs_dispatched(rg, monkeypatch, tmp_path):
    rg.samples(("Template/02_01", 0, "grader_failed", 0.0, True))
    monkeypatch.setenv("EVOSKILL_PYTHON", str(tmp_path / "no_such_python"))
    with pytest.raises(L.Halt, match="could not be started.*a person confirms no driver ran"):
        L._run_leg_regrade(rg.ctx, "train")
    rec = json.loads((L._regrade_dir(rg.ctx, rg.leg) / L.REGRADE_RECORD).read_text())
    assert list(rec["dispatched"]) == ["Template/02_01/s0/regrade1"]
    assert rec["invocations"][0]["rc_meaning"].startswith("not started")
    ok, d = L._v_leg_outcomes(rg.ctx, "train")
    assert not ok and "dispatched, no result" in d


def test_regrade_sources_unstatable_halts_before_any_dispatch(rg, monkeypatch):
    rg.samples(("Template/02_01", 0, "grader_failed", 0.0, True))

    def boom(task, run_dir=None):
        raise ValueError(f"{task} is not a task of the dataset at /nowhere")
    monkeypatch.setattr(L, "_regrade_sources", boom)
    with pytest.raises(L.Halt, match="cannot be stated.*not a task of the dataset.*nothing dispatched"):
        L._run_leg_regrade(rg.ctx, "train")
    assert rg.calls() == [] and not (L._regrade_dir(rg.ctx, rg.leg) / L.REGRADE_RECORD).exists()


def test_regrade_helpers():
    assert L._regrade_attempts() == 1
    assert L._regrade_job_id("Financial_Model/06_01", 2) == "Financial_Model/06_01/s2/regrade1"
    assert L._regrade_label(Path("x/ssbv2_r_v12_val.json")) == "rg-v12va"
    assert L._regrade_label(Path("ssbv2_r_base_train.json")) == "rg-b0tr"
    lab = L._regrade_label(Path("odd.json"))
    import re as _re
    assert _re.fullmatch(r"[a-z0-9][a-z0-9-]{0,23}", lab)
    from src.orchestrator import evolver_template as ET
    assert "once" in ET.GRADER_FAILED_POLICY and ET.REGRADE_ATTEMPTS == 1


def test_no_grader_failure_means_no_regrade_and_no_sandbox(rg):
    rg.samples(("Template/01_01", 0, "graded_wrong", 0.0, True), ("Template/01_01", 1, "passed", 1.0, True))
    ok, d = L._v_leg_outcomes(rg.ctx, "train")
    assert ok and "grader failed 0" in d and "re-grade" not in d
    L._run_leg_regrade(rg.ctx, "train")
    assert rg.calls() == [] and not L._regrade_dir(rg.ctx, rg.leg).exists()


# ── (b) the grading validation record (plugin_runner) ─────────────────────────────────
def _validation(rec: dict, run_name: str, infra=None, **over) -> dict:
    """A PLUGIN_VALIDATION.json shaped exactly as scripts/plugin_validate.py evaluate writes it; `infra`
    (a path) records that INFRA.json's sha in its environment block, as evaluate does since review #1."""
    from src.orchestrator import plugin_runner as pr
    gates = {g: {"hard": True, "pass": True, "detail": {}} for g in pr.GRADING_REQUIRED_HARD_GATES}
    gates.update({"V2_untouched_inputs_listed": {"hard": False, "pass": True, "detail": {}},
                  "counts_as_expected": {"hard": True, "pass": True, "detail": {}}})
    v = {"kind": "grading_validation", "schema": "evoskill.plugin_grading_validation/1",
         "generated_utc": "2026-09-25T00:00:00Z", "run": run_name, "subset": False,
         "jobs_sha256": "j" * 64, "freeze_allowed": True, "gates": gates,
         "split_sha256": rec["split"]["sha256"],
         "grader": {"path": rec["grader"]["declared_path"], "sha256": rec["grader"]["sha256"],
                    "entry": rec["grader"]["entry"]},
         "plugin": {"plugin_sha256": rec["files"]["benchmark_plugin.py"]["sha256"],
                    "tasks_sha256": rec["validation"]["tasks_sha256"], "n_tasks": rec["validation"]["n_tasks"]},
         "V1": {"n": 297, "passed": 297}, "V4": {"n": 592, "exact": 592}}
    if infra is not None:
        v["environment"] = {"infra_sha256": hashlib.sha256(Path(infra).read_bytes()).hexdigest()}
    v.update(over)
    return v


def test_grading_validation_problems_every_refusal():
    from src.orchestrator import plugin_runner as pr
    rec = {"split": {"sha256": "s" * 64}, "grader": {"declared_path": "g.py", "sha256": "g" * 64, "entry": "e"},
           "files": {"benchmark_plugin.py": {"sha256": "p" * 64}},
           "validation": {"tasks_sha256": "t" * 64, "n_tasks": 3}}
    kw = dict(run_name="R", split_sha256="s" * 64, grader_sha256="g" * 64, plugin_sha256="p" * 64,
              tasks_sha256="t" * 64)
    assert pr.grading_validation_problems(_validation(rec, "R"), **kw) == []
    import copy
    cases = {
        "kind": lambda v: v.update(kind="front_door"),
        "schema": lambda v: v.update(schema="x/1"),
        "SUBSET": lambda v: v.update(subset=True),
        "freeze_allowed": lambda v: v.update(freeze_allowed=False),
        "V4_exact_parity": lambda v: v["gates"]["V4_exact_parity"].update({"pass": False}),
        "missing": lambda v: v["gates"].pop("V1_every_golden_passes"),
        "hard=False": lambda v: v["gates"]["no_grader_failure_after_regrade"].update({"hard": False}),
        "counts_as_expected": lambda v: v["gates"]["counts_as_expected"].update({"pass": False}),
        "run": lambda v: v.update(run="other"),
        "split": lambda v: v.update(split_sha256="x"),
        "grader": lambda v: v["grader"].update(sha256="x"),
        "benchmark_plugin.py": lambda v: v["plugin"].update(plugin_sha256="x"),
        "task list": lambda v: v["plugin"].update(tasks_sha256="x"),
    }
    for needle, plant in cases.items():
        v = copy.deepcopy(_validation(rec, "R"))
        plant(v)
        probs = pr.grading_validation_problems(v, **kw)
        assert probs and any(needle in p for p in probs), (needle, probs)
    assert pr.grading_validation_problems([], **kw)


def test_digest_options_follow_the_mode_and_the_changed_cells_grant(tmp_path, monkeypatch):
    rd = tmp_path / "run"
    rd.mkdir()
    ctx = _ctx(run_dir=rd)
    ao = {"EVOSKILL_DIGEST_GRADER_OUTPUT_FULL": "0", "EVOSKILL_DIGEST_CHANGED_CELLS": "1",
          "EVOSKILL_DIGEST_CHANGED_CELLS_MAX": "60"}
    assert L._digest_options(_ctx(prompt_mode="ao", run_dir=rd)) == ao == L.AO_DIGEST_OPTIONS
    assert L._digest_options(SimpleNamespace(run_dir=rd)) == ao, "prompt_mode absent = ao"
    assert L._digest_options(ctx) == {"EVOSKILL_DIGEST_GRADER_OUTPUT_FULL": "1",
                                      "EVOSKILL_DIGEST_CHANGED_CELLS": "0"}
    (rd / "RUNTIME.json").write_text(json.dumps({"evolver_daytona": {"changed_cells": True}}))
    assert L._digest_options(ctx)["EVOSKILL_DIGEST_CHANGED_CELLS"] == "1"
    assert L._changed_cells_grant(ctx) is True
    (rd / "RUNTIME.json").write_text(json.dumps({"evolver_daytona": {"changed_cells": "yes"}}))
    with pytest.raises(L.Halt, match="changed_cells must be true or false"):
        L._digest_options(ctx)
    for k in L._DIGEST_OPTION_KEYS:
        assert k in L._DIGEST_ENV_OVERRIDES, "an export must be scrubbed from the builder's env"
    # an exported option that disagrees with the run's declaration HALTs; an equal one is a note
    monkeypatch.setenv("EVOSKILL_DIGEST_CHANGED_CELLS", "1")
    with pytest.raises(L.Halt, match="EVOSKILL_DIGEST_CHANGED_CELLS='1' is exported"):
        L._refuse_inherited_digest_env({"EVOSKILL_DIGEST_CHANGED_CELLS": "0"})
    L._refuse_inherited_digest_env({"EVOSKILL_DIGEST_CHANGED_CELLS": "1"})
    monkeypatch.setenv("EVOSKILL_DIGEST_CHANGED_CELLS", "0")
    with pytest.raises(L.Halt):                              # a key the caller declares no value for
        L._refuse_inherited_digest_env({})
    # AO mode declares CHANGED_CELLS=1: an exported 0 disagrees and HALTs, an exported 1 is a note
    with pytest.raises(L.Halt, match="EVOSKILL_DIGEST_CHANGED_CELLS='0' is exported"):
        L._refuse_inherited_digest_env(L._digest_options(_ctx(prompt_mode="ao", run_dir=rd)))
    assert L._digest_options(_ctx(prompt_mode="ao", run_dir=rd)) == ao, "an AO run ignores the template grant"


def _ssb_digest_ctx(tmp_path, prompt_mode):
    """An epoch-1 SSB v2 context whose builder call can be observed: a declared cache + snapshots."""
    rd = tmp_path / f"run_{prompt_mode}"
    rd.mkdir()
    cache = tmp_path / f"x_{prompt_mode}_cache"
    (tmp_path / f"x_{prompt_mode}_tasks" / "_snapshots").mkdir(parents=True)
    cache.mkdir()
    spec = SimpleNamespace(name="x", split_path=str(tmp_path / "split.json"), solver_model="m", n_samples=3,
                           benchmark="ssbv2")
    return L.Ctx(rd, 1, spec, rd / "oracle", digest_caches=(cache,), digest_bucket="b",
                 prompt_mode=prompt_mode, evolver_host="daytona")


@pytest.mark.parametrize("prompt_mode,want,absent", [
    ("ao", {"EVOSKILL_DIGEST_GRADER_OUTPUT_FULL": "0", "EVOSKILL_DIGEST_CHANGED_CELLS": "1",
            "EVOSKILL_DIGEST_CHANGED_CELLS_MAX": "60"}, ()),
    ("template", {"EVOSKILL_DIGEST_GRADER_OUTPUT_FULL": "1", "EVOSKILL_DIGEST_CHANGED_CELLS": "0"},
     ("EVOSKILL_DIGEST_CHANGED_CELLS_MAX",)),
])
def test_the_builder_gets_the_modes_digest_options(tmp_path, monkeypatch, prompt_mode, want, absent):
    """An AO-mode epoch is built with AO_DIGEST_OPTIONS, set explicitly -- not the builder's defaults;
    template mode keeps the defaults. Observed at the builder's own env, with a stray export of the cap
    scrubbed."""
    for k in L._DIGEST_OPTION_KEYS:
        monkeypatch.delenv(k, raising=False)
    ctx = _ssb_digest_ctx(tmp_path, prompt_mode)
    seen = {}

    def fake_run(argv, cwd=None, env=None, **kw):
        seen.update(env)
        (ctx.paths.digests).mkdir(parents=True, exist_ok=True)
        (ctx.paths.digests / "t.md").write_text("x")
        return SimpleNamespace(returncode=0, stdout="DIGEST OPTIONS: stub\n", stderr="")
    monkeypatch.setattr(L.subprocess, "run", fake_run)
    monkeypatch.setattr(L, "_halt_on_digest_gaps", lambda c, dirs: None)
    monkeypatch.setattr(L, "_assert_surface_is_current", lambda c, caches: None)
    L._run_build_digests_ssbv2(ctx)
    assert {k: seen.get(k) for k in want} == want
    for k in absent:
        assert k not in seen
    if prompt_mode == "ao":                                  # an export that disagrees HALTs, before any build
        monkeypatch.setenv("EVOSKILL_DIGEST_GRADER_OUTPUT_FULL", "1")
        shutil.rmtree(ctx.paths.digests)
        seen.clear()
        with pytest.raises(L.Halt, match="EVOSKILL_DIGEST_GRADER_OUTPUT_FULL='1' is exported"):
            L._run_build_digests_ssbv2(ctx)
        assert not seen


def test_epoch_grants_drop_the_oracle_in_template_mode(tmp_path):
    """EpochPaths.grants() lists no oracle in template mode (the upload carries none)."""
    ep = L.EpochPaths(tmp_path, 2)
    orc = tmp_path / "oracle"
    assert ep.grants(orc) == [ep.digests, ep.staging, orc]
    assert ep.grants(orc, template_mode=True) == [ep.digests, ep.staging]
    assert orc not in ep.grants(orc, all_epochs=True, template_mode=True)
    assert ep.grants(orc, all_epochs=True, template_mode=True) == [*ep.evidence_dirs(), ep.staging]


def test_operator_turns_ignore_ao_authoring_before_a_recorded_template_switch(tmp_path):
    """A run switched from prompt_mode "ao" to "template" keeps its AO-era authoring turns; only
    authoring turns after the latest note turn with meta.mode_switch == "template" halt the step."""
    import types
    from src.orchestrator import conversation
    from src.orchestrator import loop as L
    rd = tmp_path / "run"
    rd.mkdir()
    conversation.record(rd, speaker="operator", channel="invocation", content="old", epoch=1)
    conversation.record(rd, speaker="ao", channel="authoring", content="old prompt", epoch=1)
    ctx = types.SimpleNamespace(run_dir=rd, epoch=1, readonly=False)
    ok, why = L._v_operator_turns_recorded(ctx)
    assert not ok and "AO AUTHORING" in why
    conversation.record(rd, speaker="operator", channel="note", content="switch", epoch=1,
                        meta={"mode_switch": "template"})
    ok, why = L._v_operator_turns_recorded(ctx)
    assert ok and "1 earlier AO-era authoring" in why, why
    conversation.record(rd, speaker="ao", channel="authoring", content="new prompt", epoch=1)
    ok, why = L._v_operator_turns_recorded(ctx)
    assert not ok and "AO AUTHORING" in why
