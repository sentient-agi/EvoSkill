"""What the epoch-K evolver is told about EVERY epoch's evidence.

RUNTIME `evolver_evidence: "all_epochs"` grants the epoch-K evolver digests_e1..eK of its own run.
Two things must then be true, and these tests pin both. Each test calls the real function or runs
the real step; none reads source text.

The producing library. digests_e{j}_train carries a byte-identical copy of skills_v{j-1} under
_library/, and every task file and skill-load marker in it names that version and hash. A marker
that says every body "is the file you were handed in your staging dir" is false on every surface
but the newest. The rules live in src/orchestrator/skill_library.py (loaded here by path:
EVOSKILL_TEST_SKILL_LIBRARY). The loop verifies them with the "evidence names its producing
library" step.

The prompt side, in all-epochs mode only:
  * a separate policy file (PROMPT_POLICY.md is named by earlier runs and its bytes are pinned);
  * a deterministic EVIDENCE INDEX in FACTS, cross-checked against each directory's own leg;
  * an author message that names every directory by absolute path with its library;
  * the policy's sha256 recorded (always);
  * `_v_prompt_names_evidence`: every granted dir is named by absolute path, and no other digest
    dir is named.
Current-epoch mode must stay byte-identical. Frozen copies of the earlier message functions pin
that.

loop.py and prompt_author.py can each be loaded from a mutated COPY (EVOSKILL_TEST_LOOP,
EVOSKILL_TEST_AUTHOR_D7) for a mutation harness. The loop's own imports (prompt_author,
skill_library) always resolve to the real modules, and every test that could reach a dispatch
patches the REAL prompt_author. No test launches `claude`: the one that runs author_prompt()
replaces subprocess.run with a fake that refuses anything else.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXROOT = Path(tempfile.gettempdir()) / "fix-evidence-fixtures"


def _load(env: str, default: Path, name: str):
    p = Path(os.environ.get(env, str(default)))
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SL = _load("EVOSKILL_TEST_SKILL_LIBRARY", ROOT / "src/orchestrator/skill_library.py",
           "skill_library_under_test")
import src.orchestrator.prompt_author as PA_REAL  # noqa: E402  (a failed import must fail)
PA = (_load("EVOSKILL_TEST_AUTHOR_D7", ROOT / "src/orchestrator/prompt_author.py",
            "src.orchestrator._pa_under_test_d7")
      if os.environ.get("EVOSKILL_TEST_AUTHOR_D7") else PA_REAL)
if os.environ.get("EVOSKILL_TEST_LOOP"):
    L = _load("EVOSKILL_TEST_LOOP", ROOT / "src/orchestrator/loop.py",
              "src.orchestrator._loop_under_test_d7")
else:
    import src.orchestrator.loop as L  # noqa: E402

TRAIN = ["Debugging/01_01", "Debugging/01_02"]
# e1 1+2 = 3/6, e2 2+2 = 4/6, e3 (the current batch) 2+0 = 2/6: every figure distinct
RATES = {1: (1, 2), 2: (2, 2), 3: (2, 0), 4: (0, 1)}


@pytest.fixture
def fx():
    FIXROOT.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="d7_", dir=str(FIXROOT))).resolve()
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _digest(tid: str, passed: int, header: str | None = None, marker: str | None = None) -> str:
    out = [f"# {tid}  (Debugging)", "", f"**Result: {passed}/3 samples passed.**  turns=[7, 8, 9]",
           ""]
    if header:
        out += [header, ""]
    for k in range(3):
        out += [f"## Scorer diff — iter_0_s{k} (authoritative)", "", "0/1 match", ""]
    for k in range(3):
        out += [f"## Rollout s{k} — {'PASS' if k < passed else 'FAIL'} (7 turns)", ""]
        if marker and k == 0:
            out += ["### tool", "", marker, ""]
        out += ["did it", ""]
    return "\n".join(out)


def _lib_body(v: int) -> str:
    return f"---\nname: a\ndescription: when x\n---\nThe v{v} body.\n"


def make_run(root: Path, epochs: int, *, name="run", library=True, base_leg=True):
    """A run dir as an all-epochs build leaves it: digests_e1..e<epochs>_train (e1 from the base
    leg, e_j from skills_v{j-1} with its _library/ and anchored markers), the legs each one digests,
    and the promoted sets."""
    rd = root / name
    rd.mkdir(parents=True)
    split = root / f"{name}_split.json"
    split.write_text(json.dumps({"train": TRAIN, "val": ["Debugging/09_09"],
                                 "test": ["Debugging/08_08"]}))
    for j in range(1, epochs + 1):
        d = rd / f"digests_e{j}_train"
        d.mkdir()
        header = marker = None
        if j >= 2:
            lib = rd / f"skills_v{j - 1}"
            (lib / "a").mkdir(parents=True)
            (lib / "a" / "SKILL.md").write_text(_lib_body(j - 1))
            label, sha12 = f"skills_v{j - 1}", SL.library_sha256(lib)[:12]
            if library:
                header = SL.header_line(label, sha12)
                marker = SL.marker("a", len((lib / "a" / "SKILL.md").read_bytes()), label, sha12,
                                   "a/SKILL.md")
                SL.snapshot(lib, d)
            else:
                marker = ("[SKILL LOADED: a — 40 B. Body omitted here; it is the file you were "
                          "handed in your staging dir, so read it there rather than from this "
                          "truncated echo.]")
        p1, p2 = RATES[j]
        for tid, p in zip(TRAIN, (p1, p2)):
            (d / f"{tid.replace('/', '_')}.md").write_text(_digest(tid, p, header, marker))
        leg = rd / (f"{name}_e0_base_train.json" if j == 1 else f"{name}_v{j - 1}_train.json")
        if j > 1 or base_leg:
            leg.write_text(json.dumps({"per_task": {tid: [1] * p + [0] * (3 - p)
                                                    for tid, p in zip(TRAIN, (p1, p2))}}))
    (rd / "oracle").mkdir()
    return SimpleNamespace(rd=rd, split=split,
                           base=(rd / f"{name}_e0_base_train.json") if base_leg else None)


def ctx_for(r, epoch: int, mode="all_epochs", **kw):
    spec = SimpleNamespace(benchmark="ssbv2", split_path=str(r.split), n_samples=3, name="run",
                           solver_model="m", domain="ssbv2", variant="",
                           evolver_harness="claude_cli", prompt_author_model="opus[1m]")
    extra = {} if mode is None else {"evolver_evidence": mode}
    return L.Ctx(r.rd, epoch, spec, r.rd / "oracle", base_leg=str(r.base) if r.base else "",
                 **extra, **kw)


# ══ skill_library: the rules ════════════════════════════════════════════════════════════════════
def test_library_hash_is_the_countersignature_hash(fx):
    """ONE definition. The <12> in a marker must be the hash the proofread names for that set, so
    skill_library.library_sha256 must equal loop.staging_sha256 and the original formula."""
    lib = fx / "skills_v1"
    for n, body in (("b", "B"), ("a", "A")):
        (lib / n).mkdir(parents=True)
        (lib / n / "SKILL.md").write_text(body)
    (lib / "a" / "notes.md").write_text("not a SKILL.md, not in the countersigned hash")
    frozen = hashlib.sha256(json.dumps(
        {f.parent.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16]
         for f in sorted(lib.glob("*/SKILL.md"))}, sort_keys=True).encode()).hexdigest()
    assert SL.library_sha256(lib) == frozen == L.staging_sha256(lib)


def test_snapshot_copies_the_whole_tree_and_refuses_drift_and_links(fx):
    lib = fx / "run" / "skills_v1"
    (lib / "a").mkdir(parents=True)
    (lib / "a" / "SKILL.md").write_text(_lib_body(1))
    (lib / "a" / "ref.md").write_text("supporting file")
    d = fx / "run" / "digests_e2_train"
    assert SL.snapshot(lib, d) == ("skills_v1", SL.library_sha256(lib)[:12])
    assert SL.tree(d / "_library") == SL.tree(lib)
    assert SL.snapshot(lib, d)[0] == "skills_v1", "re-running over an identical copy is fine"
    (d / "_library" / "a" / "SKILL.md").write_text("an older body")
    with pytest.raises(ValueError, match="NOT byte-identical"):
        SL.snapshot(lib, d)
    (lib / "a" / "link.md").symlink_to(lib / "a" / "ref.md")
    with pytest.raises(ValueError, match="symlink"):
        SL.snapshot(lib, fx / "run" / "digests_e3_train")
    stag = fx / "run" / "skills_v1_staging"
    stag.mkdir()
    with pytest.raises(ValueError, match="not a promoted skill set"):
        SL.snapshot(stag, fx / "run" / "digests_e4_train")


def test_label_rules():
    assert SL.label_for_epoch(2) == "skills_v1" and SL.label_for_epoch(5) == "skills_v4"
    with pytest.raises(ValueError):
        SL.label_for_epoch(1)
    assert SL.label_of("/x/run/skills_v3") == "skills_v3"
    for bad in ("/x/run/skills_v3_staging", "/x/run/skills", "/x/run/_library"):
        with pytest.raises(ValueError):
            SL.label_of(bad)
    m = SL.marker("excel-a", 10, "skills_v1", "0123456789ab", "excel-a/SKILL.md")
    assert m.startswith("[SKILL LOADED: excel-a — ")
    assert m == ("[SKILL LOADED: excel-a — 10 B; library skills_v1 sha256:0123456789ab; body in "
                 "./_library/excel-a/SKILL.md of this digest dir]")


def test_problems_accepts_an_honest_surface(fx):
    r = make_run(fx, 3)
    assert SL.problems(r.rd / "digests_e1_train", None) == []
    for j in (2, 3):
        assert SL.problems(r.rd / f"digests_e{j}_train", r.rd / f"skills_v{j - 1}",
                           f"skills_v{j - 1}") == []


def _break(r, case):
    d = r.rd / "digests_e2_train"
    f = d / "Debugging_01_01.md"
    if case == "no_library":
        shutil.rmtree(d / "_library")
    elif case == "library_edited_after_build":
        (r.rd / "skills_v1" / "a" / "SKILL.md").write_text("edited after the surface was built")
    elif case == "legacy_marker":
        f.write_text(f.read_text() + "\n[SKILL LOADED: a — 40 B. Body omitted here; it is the file "
                     "you were handed in your staging dir, so read it there.]\n")
    elif case == "marker_of_another_version":
        f.write_text(f.read_text().replace("library skills_v1 sha256:",
                                           "library skills_v2 sha256:"))
    elif case == "no_header":
        t = f.read_text()
        f.write_text("\n".join(ln for ln in t.splitlines() if "Producing library" not in ln))
    elif case == "dangling_body":
        f.write_text(f.read_text().replace("./_library/a/SKILL.md", "./_library/zz/SKILL.md"))
    elif case == "symlink_in_library":
        (d / "_library" / "a" / "x.md").symlink_to(r.rd / "run_v1_train.json")
    return d


@pytest.mark.parametrize("case,want", [
    ("no_library", "holds no _library/"),
    ("library_edited_after_build", "NOT byte-identical"),
    ("legacy_marker", "legacy marker"),
    ("marker_of_another_version", "do not name skills_v1"),
    ("no_header", "do not name the producing library"),
    ("dangling_body", "point at a body missing"),
    ("symlink_in_library", "symlink"),
])
def test_problems_flags_every_way_a_surface_misstates_its_library(fx, case, want):
    r = make_run(fx, 2)
    d = _break(r, case)
    probs = SL.problems(d, r.rd / "skills_v1", "skills_v1")
    assert any(want in p for p in probs), probs


def test_a_base_surface_carries_no_library_and_no_marker(fx):
    r = make_run(fx, 2)
    e1 = r.rd / "digests_e1_train"
    SL.snapshot(r.rd / "skills_v1", e1)
    assert any("base surface but holds a _library/" in p for p in SL.problems(e1, None))
    shutil.rmtree(e1 / "_library")
    f = e1 / "Debugging_01_01.md"
    f.write_text(f.read_text() + "\n[SKILL LOADED: a — 1 B]\n")
    assert any("skill-load marker" in p for p in SL.problems(e1, None))


# ══ the loop: steps, builders' env, the library step ════════════════════════════════════════════
def test_new_steps_sit_before_the_ao_and_before_the_gate():
    names = [s.name for s in L.epoch_steps()]
    i = names.index("evidence names its producing library")
    assert names[i - 1] == "evidence surface is complete"
    assert i < names.index("evolver prompt authored by the AO")
    k = names.index("prompt names every granted evidence dir")
    assert names[k - 1] == "prompt is raw information, not analysis"
    # The daytona-host inputs-dir check (a no-op on local runs) follows it; both path checks run
    # before the skills are loaded through the real consumer.
    assert names[k + 1] == "prompt names the granted inputs dir"
    assert names[k + 2] == "skills load through the real consumer"
    steps = {s.name: s for s in L.epoch_steps()}
    assert steps["evidence names its producing library"].run is None
    assert steps["prompt names every granted evidence dir"].run is None


def test_library_step(fx):
    r = make_run(fx, 3)
    ok, det = L._v_evidence_libraries(ctx_for(r, 3, mode=None))
    assert ok and "current_epoch" in det
    ok, det = L._v_evidence_libraries(ctx_for(r, 3))
    assert ok and "3 evidence dir(s)" in det and "digests_e1_train <- no skills" in det, det
    assert f"digests_e3_train <- skills_v2 sha256:{SL.library_sha256(r.rd / 'skills_v2')[:12]}" \
        in det
    shutil.rmtree(r.rd / "digests_e2_train" / "_library")
    ok, det = L._v_evidence_libraries(ctx_for(r, 3))
    assert not ok and "digests_e2_train holds no _library/" in det and "REMEDY" in det, det


def test_library_step_checks_the_base_surface_too(fx):
    r = make_run(fx, 2)
    SL.snapshot(r.rd / "skills_v1", r.rd / "digests_e1_train")
    ok, det = L._v_evidence_libraries(ctx_for(r, 2))
    assert not ok and "digests_e1_train is a base surface" in det, det


def test_a_surface_built_in_current_epoch_mode_halts_all_epochs(fx):
    """A mid-arm flip: e2 was built with the legacy marker, and all-epochs mode must refuse it."""
    r = make_run(fx, 2, library=False)
    ok, det = L._v_evidence_libraries(ctx_for(r, 2))
    assert not ok and "holds no _library/" in det, det


# ══ the loop: which policy ══════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("mode,declared,want", [
    (None, "", None),
    (None, "default", "default"),
    (None, "custom", "custom"),
    (None, "all", "halt"),
    ("all_epochs", "", "all"),
    ("all_epochs", "all", "all"),
    ("all_epochs", "custom", "custom"),
    ("all_epochs", "default", "halt"),
])
def test_policy_selection(fx, mode, declared, want):
    r = make_run(fx, 1)
    custom = fx / "MY_POLICY.md"
    custom.write_text("x")
    paths = {"": "", "default": str(PA_REAL.DEFAULT_POLICY), "custom": str(custom),
             "all": str(PA_REAL.ALL_EPOCHS_POLICY)}
    ctx = ctx_for(r, 1, mode=mode, prompt_policy=paths[declared])
    if want == "halt":
        with pytest.raises(L.Halt):
            L._prompt_policy(ctx)
        return
    got = L._prompt_policy(ctx)
    expect = {None: None, "default": PA_REAL.DEFAULT_POLICY, "custom": custom,
              "all": PA_REAL.ALL_EPOCHS_POLICY}[want]
    assert (got is None and expect is None) or Path(got).resolve() == Path(expect).resolve()


def test_the_single_batch_policy_file_is_unchanged():
    """PROMPT_POLICY.md is named by the RUNTIME.json of earlier runs, so its bytes are pinned."""
    assert hashlib.sha256(PA_REAL.DEFAULT_POLICY.read_bytes()).hexdigest() == (
        "4f7e91aa2dbe507145c3e19d47b0368e6b4668f92f3ee4daba4cf13f09696bd0")


# ══ the loop: the AO call site ══════════════════════════════════════════════════════════════════
def _patch_ao(monkeypatch):
    got = {}

    def fake_author(**kw):
        got.update(kw)
        return {}
    monkeypatch.setattr(PA_REAL, "author_prompt", fake_author)
    return got


def test_ao_call_site_current_epoch_is_unchanged(fx, monkeypatch):
    r = make_run(fx, 2)
    got = _patch_ao(monkeypatch)
    calls = []
    real = PA_REAL.derive_facts

    def spy(*a, **kw):
        calls.append(kw)
        return real(*a, **kw)
    monkeypatch.setattr(PA_REAL, "derive_facts", spy)
    L._run_author_prompt(ctx_for(r, 2, mode=None))
    assert calls == [{"variant": ""}], "derive_facts is called exactly as before"
    assert got["policy"] is None and got["evidence_dirs"] is None
    assert "EVIDENCE INDEX" not in got["facts"].as_block()


def test_a_single_batch_policy_in_all_epochs_mode_halts_before_any_dispatch(fx, monkeypatch):
    r = make_run(fx, 2)
    got = _patch_ao(monkeypatch)
    with pytest.raises(L.Halt, match="single-batch"):
        L._run_author_prompt(ctx_for(r, 2, prompt_policy=str(PA_REAL.DEFAULT_POLICY)))
    assert got == {}


def test_ao_call_site_refuses_an_earlier_surface_without_its_library(fx, monkeypatch):
    r = make_run(fx, 3)
    shutil.rmtree(r.rd / "digests_e2_train" / "_library")
    got = _patch_ao(monkeypatch)
    with pytest.raises(ValueError, match="does not carry its producing library"):
        L._run_author_prompt(ctx_for(r, 3))
    assert got == {}, "no AO is dispatched over a misdescribed surface"


# ══ the loop: the prompt names exactly the granted evidence ═════════════════════════════════════
def _prompt_naming(rd: Path, dirs, extra="") -> str:
    lines = ["This batch is 2 train tasks x 3 samples = 6 episodes, of which 2 of 6 episodes passed."]
    for d in dirs:
        lines.append(f"Evidence: `{d}` holds one file per train task.")
    return "\n".join(lines) + "\n" + extra


@pytest.mark.parametrize("case,ok,want", [
    ("all", True, "names all 3"),
    ("all_with_library_paths", True, "names all 3"),
    ("sentence_end_dot", True, "names all 3"),
    ("missing_e1", False, "never names ['digests_e1_train']"),
    ("later_epoch", False, "LATER epoch"),
    ("relative", False, "without this run's absolute path"),
    ("bare", False, "without this run's absolute path"),
    ("another_run", False, "without this run's absolute path"),
    ("stray_sibling", False, "digests_e2_train.062721.bak"),
    ("e0_hand_naming", False, "not granted at epoch 3"),
])
def test_prompt_names_evidence(fx, case, ok, want):
    r = make_run(fx, 3)
    ctx = ctx_for(r, 3)
    dirs = ctx.paths.evidence_dirs()
    extra = ""
    if case == "missing_e1":
        dirs = dirs[1:]
    elif case == "all_with_library_paths":
        extra = f"The v1 library is at {dirs[1]}/_library/ and v2's at {dirs[2]}/_library/a/SKILL.md."
    elif case == "sentence_end_dot":
        extra = f"The newest directory is {dirs[2]}."
    elif case == "later_epoch":
        extra = f"Also read {r.rd.resolve()}/digests_e4_train/."
    elif case == "relative":
        extra = "Relative: outputs/run/digests_e2_train/Debugging_01_01.md"
    elif case == "bare":
        extra = "(that is digests_e2_train)"
    elif case == "another_run":
        extra = f"Peer: {fx / 'run_f2'}/digests_e2_train"
    elif case == "stray_sibling":
        extra = f"Old copy: {r.rd.resolve()}/digests_e2_train.062721.bak"
    elif case == "e0_hand_naming":
        extra = f"Base: {r.rd.resolve()}/digests_e0_train"
    ctx.paths.prompt.write_text(_prompt_naming(r.rd, dirs, extra))
    got, det = L._v_prompt_names_evidence(ctx)
    assert got is ok and want in det, det


def test_prompt_names_evidence_failure_names_the_remedy(fx):
    """L2: a no-runner Step's HALT must say what to do, not only 'This step has no runner'."""
    r = make_run(fx, 3)
    ctx = ctx_for(r, 3)
    ctx.paths.prompt.write_text(_prompt_naming(r.rd, ctx.paths.evidence_dirs(),
                                               "(that is digests_e2_train)"))
    ok, det = L._v_prompt_names_evidence(ctx)
    assert ok is False
    want = (f"REMEDY: move {Path(ctx.run_dir).absolute() / 'ITER3_PROMPT.md'} aside (keep it) "
            f"and re-run --step; the AO re-authors against the EVIDENCE INDEX paths.")
    assert det.endswith(want), det
    ctx.paths.prompt.write_text(_prompt_naming(r.rd, ctx.paths.evidence_dirs()))
    ok, det = L._v_prompt_names_evidence(ctx)
    assert ok and "REMEDY" not in det, det


def test_prompt_names_evidence_is_all_epochs_only_and_needs_a_prompt(fx):
    r = make_run(fx, 2)
    ctx = ctx_for(r, 2, mode=None)
    ctx.paths.prompt.write_text("names nothing at all")
    ok, det = L._v_prompt_names_evidence(ctx)
    assert ok and "current_epoch" in det
    ok, det = L._v_prompt_names_evidence(ctx_for(r, 2, prompt_policy=""))
    assert ok is False, det                  # all-epochs: names none of the granted dirs
    (r.rd / "ITER2_PROMPT.md").unlink()
    ok, det = L._v_prompt_names_evidence(ctx_for(r, 2))
    assert not ok and "no prompt" in det


def test_prompt_names_evidence_accepts_the_unresolved_absolute_run_path(fx):
    """A run dir reached through a symlinked parent: the unresolved absolute form names the same
    directory the evolver is granted, and is accepted as well as the resolved one."""
    real = fx / "real"
    real.mkdir()
    r = make_run(real, 2)
    (fx / "link").symlink_to(real)
    ctx = ctx_for(SimpleNamespace(rd=fx / "link" / "run", split=r.split, base=r.base), 2)
    unresolved = [fx / "link" / "run" / f"digests_e{j}_train" for j in (1, 2)]
    ctx.paths.prompt.write_text(_prompt_naming(ctx.run_dir, unresolved))
    ok, det = L._v_prompt_names_evidence(ctx)
    assert ok, det


# ══ prompt_author: the EVIDENCE INDEX ═══════════════════════════════════════════════════════════
def _evidence(r, epochs):
    return [(j, (r.rd / f"digests_e{j}_train").resolve(),
             r.base if j == 1 else r.rd / f"run_v{j - 1}_train.json",
             None if j == 1 else r.rd / f"skills_v{j - 1}") for j in range(1, epochs + 1)]


def test_index_rows(fx):
    r = make_run(fx, 3)
    f = PA.derive_facts(r.rd / "digests_e3_train", r.rd / "run_v2_train.json",
                        evidence=_evidence(r, 3))
    rows = f.evidence_index
    assert [x.epoch for x in rows] == [1, 2, 3]
    assert [x.path for x in rows] == [str((r.rd / f"digests_e{j}_train").resolve()) for j in (1, 2, 3)]
    assert [x.library for x in rows] == ["", "skills_v1", "skills_v2"]
    assert [(x.passed, x.episodes) for x in rows] == [(3, 6), (4, 6), (2, 6)]
    assert [x.task_files for x in rows] == [2, 2, 2]
    assert all(x.rate_checked for x in rows) and [x.current for x in rows] == [False, False, True]
    block = f.as_block()
    assert block.startswith(f._batch_block()), "the batch block is untouched; the index follows"
    assert "3 of 6 episodes passed" in block and "(THIS BATCH)" in block
    # The index must not use a phrase the launcher reads as a claim about THIS batch (rule 3), or a
    # coverage fraction next to "scor…" (rule 1).
    idx = block[len(f._batch_block()):]
    assert "train tasks" not in idx and "tasks in this batch" not in idx and "scor" not in idx


def test_without_evidence_the_facts_block_is_unchanged(fx):
    r = make_run(fx, 2)
    f = PA.derive_facts(r.rd / "digests_e2_train", r.rd / "run_v1_train.json")
    assert f.evidence_index == [] and f.as_block() == f._batch_block()
    assert "EVIDENCE INDEX" not in f.as_block()


def test_index_withholds_a_count_no_gate_could_check(fx):
    r = make_run(fx, 2, base_leg=False)
    ev = _evidence(r, 2)
    f = PA.derive_facts(r.rd / "digests_e2_train", r.rd / "run_v1_train.json", evidence=ev)
    assert f.evidence_index[0].rate_checked is False
    block = f.as_block()
    assert "3 of 6 episodes passed" not in block and "NOT STATED" in block
    assert "1 count(s) withheld" in block


@pytest.mark.parametrize("case,want", [
    ("leg_mismatch", "EVIDENCE MISMATCH"),
    ("leg_missing", "does not exist"),
    ("no_library", "does not carry its producing library"),
    ("gap", "epochs 1..K"),
    ("last_is_not_current", "newest directory must be the batch"),
    ("no_result_header", "Result header"),
    ("library_not_given", "no producing library given"),
])
def test_index_refuses_to_describe_a_directory_wrongly(fx, case, want):
    r = make_run(fx, 3)
    ev = _evidence(r, 3)
    cur, leg = r.rd / "digests_e3_train", r.rd / "run_v2_train.json"
    if case == "leg_mismatch":
        (r.rd / "run_v1_train.json").write_text(json.dumps({"per_task": {TRAIN[0]: [1, 1, 1]}}))
    elif case == "leg_missing":
        (r.rd / "run_v1_train.json").unlink()
    elif case == "no_library":
        shutil.rmtree(r.rd / "digests_e2_train" / "_library")
    elif case == "gap":
        ev = [ev[0], ev[2]]
    elif case == "last_is_not_current":
        cur, leg = r.rd / "digests_e2_train", r.rd / "run_v1_train.json"
    elif case == "no_result_header":
        (r.rd / "digests_e2_train" / "Debugging_01_02.md").write_text(
            SL.header_line("skills_v1", SL.library_sha256(r.rd / "skills_v1")[:12]) + "\nno header")
    elif case == "library_not_given":
        ev[1] = (ev[1][0], ev[1][1], ev[1][2], None)
    with pytest.raises(ValueError, match=want):
        PA.derive_facts(cur, leg, evidence=ev)


# ══ prompt_author: the author message ═══════════════════════════════════════════════════════════
def _frozen_prior_note(prior):
    """_prior_note as of HEAD 9948ffa, verbatim. Current-epoch mode must reproduce it."""
    if not (prior and prior.is_dir()):
        return "NO skill set exists yet. This is the first iteration; say so in the prompt."
    files = sorted(prior.glob("*/SKILL.md"))
    if not files:
        return "NO skill set exists yet. This is the first iteration; say so in the prompt."
    out = [
        f"A PRIOR SKILL SET OF {len(files)} SKILL(S) EXISTS, and the evolver is revising it — so the "
        "prompt must say what changed since it was written, and must judge each prior rule against "
        "this batch's evidence of that rule IN USE.",
        "",
        "*** REPRODUCE THE PRIOR SKILL SET VERBATIM INSIDE THE PROMPT. *** The evolver CANNOT OPEN "
        "it: the evolver is granted only the digests, its staging dir and the oracle, and a hook "
        "denies every skills_v* read — including this run's own previous version. A prompt that "
        f"says 'read them at `{prior}`' is telling the evolver to do something it cannot do, and "
        "no gate catches that. Quote each file's full text in a fenced block, under its name, so "
        "the prompt is self-contained. Do not name the path at all.",
        "",
        "THE PRIOR SET, which you must carry into the prompt:",
    ]
    for f in files:
        out += ["", f"--- {f.parent.name}/SKILL.md ---", f.read_text(errors="replace").rstrip()]
    return "\n".join(out)


def _frozen_author_user_prompt(policy_text, facts, out, epoch, digests, oracle, prior, staging):
    """_author_user_prompt as of HEAD 9948ffa, verbatim."""
    prior_note = _frozen_prior_note(prior)
    return "\n".join([
        "You are assembling the evolver's user prompt for iteration " + str(epoch) + ".",
        "",
        "Follow the POLICY below exactly. Ground every quantity in the FACTS block, verbatim.",
        f"Read the evidence at `{digests}` and the golden specifications at `{oracle}` to describe the",
        "failure modes QUALITATIVELY — that is what the traces are for, and it is the part no number",
        "can supply. Those directories are TRAIN-only by construction; do not look for anything else.",
        "",
        prior_note,
        "",
        "THE EVOLVER MUST BE TOLD TO WRITE ITS SKILL SET INTO THIS EXACT DIRECTORY:",
        f"    {staging}",
        "State that literal path in the prompt. Do NOT tell it to write where skills live at solver",
        "RUNTIME (a .claude/skills/ or similar) — the pipeline collects the deliverable from the",
        "staging path above, and a skill set written anywhere else is invisible to every later step.",
        "",
        f"WRITE THE FINISHED PROMPT TO `{out}` using the Write tool. Write that file and nothing else.",
        "Do not create scratch files. Your final message should be one short paragraph saying what you",
        "emphasised and why — it is read by an operator, not by the evolver.",
        "",
        "=" * 78, "POLICY", "=" * 78, "",
        policy_text,
        "", "=" * 78, "FACTS", "=" * 78, "",
        facts.as_block(),
    ])


@pytest.mark.parametrize("with_prior", [True, False])
def test_current_epoch_message_is_byte_identical(fx, with_prior):
    r = make_run(fx, 2)
    f = PA.derive_facts(r.rd / "digests_e2_train", r.rd / "run_v1_train.json")
    prior = r.rd / "skills_v1" if with_prior else None
    args = ("POLICY TEXT", f, r.rd / "ITER2_PROMPT.md", 2, Path("outputs/run/digests_e2_train"),
            r.rd / "oracle", prior, Path("outputs/run/skills_v2_staging"))
    assert PA._author_user_prompt(*args) == _frozen_author_user_prompt(*args)
    assert PA._prior_note(prior) == _frozen_prior_note(prior)


def test_all_epochs_message_names_every_dir_with_its_library(fx):
    r = make_run(fx, 3)
    ev = [(r.rd / f"digests_e{j}_train").resolve() for j in (1, 2, 3)]
    f = PA.derive_facts(r.rd / "digests_e3_train", r.rd / "run_v2_train.json",
                        evidence=_evidence(r, 3))
    msg = PA._author_user_prompt("POLICY TEXT", f, r.rd / "ITER3_PROMPT.md", 3,
                                 r.rd / "digests_e3_train", r.rd / "oracle", r.rd / "skills_v2",
                                 Path("outputs/run/skills_v3_staging"), evidence_dirs=ev)
    assert f"    e1  {ev[0]}  — rollouts of the solver carrying NO skills (the base leg)" in msg
    assert (f"    e2  {ev[1]}  — rollouts of the solver carrying skills_v1; that library's exact "
            f"files are in {ev[1]}/_library/") in msg
    assert f"    e3  {ev[2]}  — rollouts of the solver carrying skills_v2" in msg
    assert "THIS BATCH" in msg.split(f"e3  {ev[2]}")[1].splitlines()[0]
    assert "ON DEMAND" in msg and "QUALITATIVELY" not in msg
    assert f"    {Path('outputs/run/skills_v3_staging').resolve()}" in msg, "staging made absolute"
    assert "## EVIDENCE INDEX" in msg and "POLICY TEXT" in msg
    note = PA._prior_note(r.rd / "skills_v2", all_epochs=True)
    assert note in msg
    assert "cannot open" in note.lower() and "VERBATIM" in note and "_library/" in note
    assert "The v2 body." in note, "the set being revised is still carried verbatim"
    assert "The v1 body." not in msg, "earlier libraries are read on demand, not pasted"


def test_all_epochs_message_refuses_a_non_canonical_list(fx):
    r = make_run(fx, 2)
    f = PA.derive_facts(r.rd / "digests_e2_train", r.rd / "run_v1_train.json")
    for bad in ([r.rd / "digests_e2_train"], [r.rd / "digests_e1_train", r.rd / "oracle"]):
        with pytest.raises(ValueError):
            PA._author_user_prompt("P", f, r.rd / "ITER2_PROMPT.md", 2, r.rd / "digests_e2_train",
                                   r.rd / "oracle", None, r.rd / "s", evidence_dirs=bad)


@pytest.mark.skipif(shutil.which("sandbox-exec") is None, reason="the AO refuses uncontained")
def test_author_prompt_records_the_policy_hash_and_picks_the_policy(fx, monkeypatch):
    """author_prompt() end to end with subprocess.run replaced: the fake answers ONLY the claude
    dispatch (writing the artifact the AO would) and refuses any other call."""
    r = make_run(fx, 2)
    out = r.rd / "ITER2_PROMPT.md"
    f = PA.derive_facts(r.rd / "digests_e2_train", r.rd / "run_v1_train.json")
    sent = {}

    def fake_run(argv, *a, **kw):
        if "claude" not in argv:
            raise AssertionError(f"unexpected subprocess.run: {argv[:4]}")
        sent["msg"] = argv[argv.index("-p") + 1]
        out.write_text("x" * 2000)
        kw["stdout"].write('{"type":"result","result":"done"}\n')
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(subprocess, "run", fake_run)
    info = PA.author_prompt(facts=f, out=out, epoch=2, digests=r.rd / "digests_e2_train",
                            oracle=r.rd / "oracle", log=r.rd / "e2_prompt_author.log",
                            staging=r.rd / "skills_v2_staging", evidence_dirs=None,
                            prior_skills=r.rd / "skills_v1")
    want_pol = PA.DEFAULT_POLICY
    assert info["policy"] == str(want_pol)
    assert info["policy_sha256"] == hashlib.sha256(want_pol.read_bytes()).hexdigest()
    assert want_pol.read_text() in sent["msg"]
    assert "evidence_dirs" not in info and "QUALITATIVELY" in sent["msg"]


@pytest.mark.skipif(shutil.which("sandbox-exec") is None, reason="the AO refuses uncontained")
def test_author_prompt_never_hands_the_parent_session_to_the_ao(fx, monkeypatch):
    """L6: the AO's `claude` must not inherit what identifies or reaches the operator's own
    session. Names (never values) are recorded in the return value (e{N}_PROMPT_AUTHOR.json)."""
    from src.orchestrator.evolver_dispatch import parent_session_env_names
    # every value is distinctive: a bare "bg" once matched a random fixture dir name in `info`
    parent = {"CLAUDE_CODE_MESSAGING_SOCKET": "/tmp/SECRET-SOCK",
              "CLAUDE_CODE_MESSAGING_TOKEN": "SECRET-TOKEN", "CLAUDE_JOB_DIR": "/x/SECRET-JOB",
              "CLAUDE_CODE_SESSION_KIND": "SECRET-KIND-bg"}
    for k, v in parent.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "kept-token")
    want = parent_session_env_names(os.environ)
    r = make_run(fx, 1)
    out = r.rd / "ITER1_PROMPT.md"
    f = PA.derive_facts(r.rd / "digests_e1_train", r.base)
    got = {}

    def fake_run(argv, *a, **kw):
        if "claude" not in argv:
            raise AssertionError(f"unexpected subprocess.run: {argv[:4]}")
        got["env"] = dict(kw["env"])
        out.write_text("x" * 2000)
        kw["stdout"].write('{"type":"result","result":"done"}\n')
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(subprocess, "run", fake_run)
    info = PA.author_prompt(facts=f, out=out, epoch=1, digests=r.rd / "digests_e1_train",
                            oracle=r.rd / "oracle", log=r.rd / "e1_prompt_author.log",
                            staging=r.rd / "skills_v1_staging")
    assert not set(want) & set(got["env"]), sorted(set(want) & set(got["env"]))
    assert got["env"]["CLAUDE_CODE_OAUTH_TOKEN"] == "kept-token"
    assert info["parent_env_scrubbed"] == want
    assert not any(v in json.dumps(info) for v in parent.values())
