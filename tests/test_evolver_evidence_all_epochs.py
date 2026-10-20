"""All-epochs evidence (RUNTIME `evolver_evidence`) and the containment that goes with it. Every
test here either EXECUTES the boundary (a real sandbox-exec) or calls the real function -- never the
profile text alone, where a test can pass for a reason other than the one it claims.

Pinned:
  * src/orchestrator/evidence_surface.py -- which dirs are one run's evidence at iteration K, the
    train-only / no-symlink rules, the per-dir hash. Loaded BY PATH (EVOSKILL_TEST_EVIDENCE_MODULE).
  * invariants.check_no_peer_artifacts_read -- own-run by path component, not substring
    (EVOSKILL_TEST_INVARIANTS).
  * evolver_dispatch.build_evolver_sandbox_profile -- ~/.claude/jobs & co. denied, later narrow
    grants still win (EVOSKILL_TEST_DISPATCH).
  * prompt_author._ao_sandbox -- the AO does not read the whole run dir.
  * loop -- mode refusals, the receipt check.

Fixture dirs live under the per-uid CLI temp root (/tmp/claude-<uid>/fix-evidence-fixtures/), the
geometry the sandbox tests need, and are removed afterwards.
"""
from __future__ import annotations

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


def _load(env: str, default: Path, name: str):
    p = Path(os.environ.get(env, str(default)))
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ES = _load("EVOSKILL_TEST_EVIDENCE_MODULE", ROOT / "src/orchestrator/evidence_surface.py",
           "evidence_surface_under_test")
INV = _load("EVOSKILL_TEST_INVARIANTS", ROOT / "src/orchestrator/invariants.py",
            "invariants_under_test")
DISPATCH = _load("EVOSKILL_TEST_DISPATCH", ROOT / "src/orchestrator/evolver_dispatch.py",
                 "evolver_dispatch_under_test")
# The sandbox tests stand the fixture in for other sessions' dirs, so it must sit under the
# per-uid CLI temp root the profiles deny.
FIXROOT = DISPATCH.cli_uid_roots()[0] / "fix-evidence-fixtures"

import src.orchestrator.prompt_author as PA  # noqa: E402  (a failed import must fail, never skip)
# The AO-sandbox test alone may use a mutated COPY of prompt_author. Everything that could reach a
# real dispatch (the AO call-site test) always patches the REAL module above.
PA_SB = (_load("EVOSKILL_TEST_AUTHOR", ROOT / "src/orchestrator/prompt_author.py",
               "src.orchestrator._pa_under_test")
         if os.environ.get("EVOSKILL_TEST_AUTHOR") else PA)
if os.environ.get("EVOSKILL_TEST_LOOP"):
    # a mutated COPY of loop.py (scripts/mutation_check_digest_firewalls.sh). Named inside the
    # package so its relative imports resolve to the real siblings.
    L = _load("EVOSKILL_TEST_LOOP", ROOT / "src/orchestrator/loop.py",
              "src.orchestrator._loop_under_test")
else:
    import src.orchestrator.loop as L  # noqa: E402

SANDBOX = shutil.which("sandbox-exec") is not None
needs_sandbox = pytest.mark.skipif(not SANDBOX, reason="sandbox-exec unavailable on this host")

TRAIN = ["Debugging/01_01", "Debugging/01_02"]
VAL = ["Debugging/09_09"]
TEST = ["Debugging/08_08"]


@pytest.fixture
def fx():
    FIXROOT.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="evid_", dir=str(FIXROOT))).resolve()
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def digest(tid: str, passed: int, n: int = 3) -> str:
    """An SSB-shaped digest: the header line every consumer parses, n rollouts, n diffs."""
    L_ = [f"# {tid}  (Debugging)", "", f"**Result: {passed}/{n} samples passed.**  turns=[7, 8, 9]", ""]
    for k in range(n):
        L_ += [f"## Scorer diff — iter_0_s{k} (authoritative)", "", "0/1 match", ""]
    for k in range(n):
        L_ += [f"## Rollout s{k} — {'PASS' if k < passed else 'FAIL'} (7 turns)", "", "did it", ""]
    return "\n".join(L_)


def make_run(root: Path, epochs: int, rates=None, name: str = "run") -> dict:
    """A run dir with digests_e1..e<epochs>_train, the things containment must keep hidden, and a
    split OUTSIDE the run dir (as ssbv2_ao1's is)."""
    rd = root / name
    rd.mkdir(parents=True)
    split = root / f"{name}_split.json"
    split.write_text(json.dumps({"train": TRAIN, "val": VAL, "test": TEST}))
    rates = rates or {}
    for j in range(1, epochs + 1):
        d = rd / f"digests_e{j}_train"
        d.mkdir()
        p1, p2 = rates.get(j, (1, 1))
        (d / "Debugging_01_01.md").write_text(digest("Debugging/01_01", p1))
        (d / "Debugging_01_02.md").write_text(digest("Debugging/01_02", p2))
    (rd / "oracle").mkdir()
    (rd / "oracle" / "README.md").write_text("no golden specs are exposed\n")
    # held-out material and everything else a hidden run dir must keep hidden
    (rd / f"{name}_v1_val.json").write_text('{"sample_pass": 5, "sample_total": 9, "VAL": 1}')
    (rd / "E3_GATE_RESULT.json").write_text('{"verdict": "VAL GATE"}')
    (rd / "SEALED_TEST_RESULT.json").write_text('{"sealed": "TEST"}')
    (rd / f"{name}_test_v3.json").write_text('{"test": 1}')
    (rd / "drive_e1.out").write_text("val verdict")
    (rd / "tau3_evolver_e1.log").write_text("earlier evolver transcript")
    (rd / "e1_prompt_author.log").write_text("earlier AO transcript")
    (rd / "skills_v1" / "a").mkdir(parents=True)
    (rd / "skills_v1" / "a" / "SKILL.md").write_text("---\nname: a\ndescription: x\n---\nV1 BODY\n")
    return {"rd": rd, "split": split}


# ══ evidence_surface: the one definition ═══════════════════════════════════════════════════════
def test_validate_accepts_exactly_the_earlier_epochs(fx):
    r = make_run(fx, 3)
    rd = r["rd"]
    prior = f"{rd / 'digests_e1_train'}{os.pathsep}{rd / 'digests_e2_train'}"
    ok, errs = ES.validate(rd, 3, rd / "digests_e3_train", prior, r["split"])
    assert errs == []
    assert [j for j, _ in ok] == [1, 2]
    assert [p for _, p in ok] == [os.path.realpath(rd / f"digests_e{j}_train") for j in (1, 2)]


def test_validate_refuses_an_incomplete_list(fx):
    """Every entry valid, one earlier epoch simply ABSENT: a launcher must not dispatch on that
    partial surface (otherwise only _v_evolver_contained would notice, after the spend)."""
    r = make_run(fx, 4)
    rd = r["rd"]
    for prior, missing in (([1], "digests_e2_train, digests_e3_train"), ([1, 3], "digests_e2_train"),
                           ([2, 3], "digests_e1_train")):
        ok, errs = ES.validate(rd, 4, rd / "digests_e4_train",
                               os.pathsep.join(str(rd / f"digests_e{j}_train") for j in prior),
                               r["split"])
        assert any(f"omits {missing} " in e for e in errs), (prior, errs)
    ok, errs = ES.validate(rd, 4, rd / "digests_e4_train",
                           os.pathsep.join(str(rd / f"digests_e{j}_train") for j in (3, 1, 2)),
                           r["split"])
    assert errs == [] and [j for j, _ in ok] == [1, 2, 3]
    # iteration 1: nothing earlier exists, so the empty list is complete
    ok, errs = ES.validate(rd, 1, rd / "digests_e1_train", "", r["split"])
    assert errs == [] and ok == []


def test_validate_refuses_a_dir_outside_the_run(fx):
    r = make_run(fx, 3)
    peer = make_run(fx, 2, name="run_f2")               # a PEER whose name extends ours
    rd = r["rd"]
    prior = f"{peer['rd'] / 'digests_e1_train'}{os.pathsep}{rd / 'digests_e2_train'}"
    _ok, errs = ES.validate(rd, 3, rd / "digests_e3_train", prior, r["split"])
    assert any("OUTSIDE this run dir" in e for e in errs), errs


def test_validate_refuses_a_symlinked_dir(fx):
    r = make_run(fx, 3)
    rd = r["rd"]
    real = fx / "elsewhere_e1"
    shutil.move(str(rd / "digests_e1_train"), str(real))
    (rd / "digests_e1_train").symlink_to(real)
    prior = f"{rd / 'digests_e1_train'}{os.pathsep}{rd / 'digests_e2_train'}"
    _ok, errs = ES.validate(rd, 3, rd / "digests_e3_train", prior, r["split"])
    assert any("SYMLINK" in e for e in errs), errs


def test_validate_refuses_a_link_elsewhere_that_points_at_a_real_dir(fx):
    """The entry must BE the canonical path. A link outside the run whose target is a real
    digests_e1_train resolves inside the run — and would still hand the kernel a path the loop
    never named."""
    r = make_run(fx, 3)
    rd = r["rd"]
    (fx / "innocent_looking").symlink_to(rd / "digests_e1_train")
    ok, errs = ES.validate(rd, 3, rd / "digests_e3_train", str(fx / "innocent_looking"),
                           r["split"])
    # refused AT THE ENTRY, before its target is ever resolved or accepted as epoch 1
    assert any("SYMLINK — the profile builder resolves it" in e for e in errs), errs
    assert ok == [], ok


def test_validate_refuses_the_current_or_a_later_epoch(fx):
    r = make_run(fx, 4)
    rd = r["rd"]
    for bad in ("digests_e3_train", "digests_e4_train"):
        _ok, errs = ES.validate(rd, 3, rd / "digests_e3_train",
                                f"{rd / 'digests_e1_train'}{os.pathsep}{rd / bad}", r["split"])
        assert any("not digests_e<j>_train of this run" in e for e in errs), (bad, errs)


def test_validate_refuses_stray_siblings_relative_empty_and_duplicates(fx):
    r = make_run(fx, 3)
    rd = r["rd"]
    shutil.copytree(rd / "digests_e1_train", rd / "digests_e1_train.062721.bak")
    cases = {
        str(rd / "digests_e1_train.062721.bak"): "not digests_e<j>_train",
        "digests_e1_train": "not absolute",
        f"{rd / 'digests_e1_train'}{os.pathsep}{os.pathsep}{rd / 'digests_e2_train'}": "EMPTY entry",
        f"{rd / 'digests_e1_train'}{os.pathsep}{rd / 'digests_e1_train'}": "listed twice",
    }
    for prior, want in cases.items():
        _ok, errs = ES.validate(rd, 3, rd / "digests_e3_train", prior, r["split"])
        assert any(want in e for e in errs), (prior, errs)


def test_validate_refuses_non_train_stems_and_symlinked_files(fx):
    r = make_run(fx, 3)
    rd = r["rd"]
    (rd / "digests_e1_train" / "Debugging_09_09.md").write_text(digest("Debugging/09_09", 1))
    _ok, errs = ES.validate(rd, 3, rd / "digests_e3_train", str(rd / "digests_e1_train"),
                            r["split"])
    assert any("HELD-OUT MATERIAL" in e for e in errs), errs
    (rd / "digests_e1_train" / "Debugging_09_09.md").unlink()
    (rd / "digests_e1_train" / "Stray_07_07.md").write_text("x")
    _ok, errs = ES.validate(rd, 3, rd / "digests_e3_train", str(rd / "digests_e1_train"),
                            r["split"])
    assert any("NOT train tasks" in e for e in errs), errs
    (rd / "digests_e1_train" / "Stray_07_07.md").unlink()
    (rd / "digests_e2_train" / "_library").mkdir()
    (rd / "digests_e2_train" / "_library" / "SKILL.md").symlink_to(rd / "skills_v1/a/SKILL.md")
    _ok, errs = ES.validate(rd, 3, rd / "digests_e3_train", str(rd / "digests_e2_train"),
                            r["split"])
    assert any("symlink(s) inside" in e for e in errs), errs


def test_validate_refuses_without_a_split(fx):
    r = make_run(fx, 3)
    rd = r["rd"]
    _ok, errs = ES.validate(rd, 3, rd / "digests_e3_train", str(rd / "digests_e1_train"),
                            str(fx / "no_such_split.json"))
    assert any("no split file" in e for e in errs), errs


def test_validate_requires_the_current_dir_to_be_canonical(fx):
    r = make_run(fx, 3)
    rd = r["rd"]
    _ok, errs = ES.validate(rd, 3, rd / "digests_e2_train", str(rd / "digests_e1_train"),
                            r["split"])
    assert any("current digest dir" in e for e in errs), errs


def test_dir_hash_tracks_content_and_never_follows_links(fx):
    r = make_run(fx, 1)
    d = r["rd"] / "digests_e1_train"
    h0 = ES.dir_sha256(d)
    assert h0 == ES.dir_sha256(d)
    (d / "Debugging_01_01.md").write_text("changed")
    assert ES.dir_sha256(d) != h0
    h1 = ES.dir_sha256(d)
    (d / "l").symlink_to(r["rd"] / "E3_GATE_RESULT.json")
    h2 = ES.dir_sha256(d)
    assert h2 != h1                                         # the link is recorded...
    (r["rd"] / "E3_GATE_RESULT.json").write_text("target changed")
    assert ES.dir_sha256(d) == h2                           # ...but never followed


# ══ the peer-artifact audit: own run by PATH COMPONENT, never by substring ═════════════════════
def _log_with(tmp: Path, *cmds) -> Path:
    recs = [{"type": "system", "subtype": "init", "session_id": "S"}]
    for c in cmds:
        recs.append({"type": "assistant", "session_id": "S", "message": {"content": [
            {"type": "tool_use", "name": "Bash", "input": {"command": c}}]}})
    p = tmp / "e.log"
    p.write_text("\n".join(json.dumps(r) for r in recs))
    return p


def test_peer_audit_is_not_fooled_by_a_peer_whose_name_extends_ours(fx):
    own = fx / "outputs" / "run_a"
    peer = fx / "outputs" / "run_a_f2"
    own.mkdir(parents=True)
    peer.mkdir(parents=True)
    for c in (f"cat {peer}/skills_v1/x/SKILL.md",
              f"cd {peer} && cat skills_v1/x/SKILL.md",
              "cat outputs/run_a_f2/skills_v2/y/SKILL.md"):
        r = INV.check_no_peer_artifacts_read(_log_with(fx, c), own)
        assert r.status == "FAIL", (c, r.detail)
    # the run's own files, spelled every way, stay allowed
    for c in (f"cat {own}/skills_v1/x/SKILL.md",
              "cat outputs/run_a/skills_v1/x/SKILL.md",
              f"ls {own}/skills_v2_staging",
              f"cd {own}/digests_e1_train && wc -l *.md"):
        r = INV.check_no_peer_artifacts_read(_log_with(fx, c), own)
        assert r.status == "PASS", (c, r.detail)


# '..' is collapsed and '~' expanded BEFORE ownership is judged.
def test_peer_audit_normalises_dotdot_before_judging_ownership(fx):
    own = fx / "outputs" / "run_a"
    (fx / "outputs" / "run_a_f2").mkdir(parents=True)
    own.mkdir(parents=True)
    for c in (f"cat {own}/digests_e1_train/../../run_a_f2/skills_v1/x/SKILL.md",
              "cat outputs/run_a/digests_e1_train/../../run_a_f2/skills_v1/x/SKILL.md",
              f"cd {own}/digests_e1_train/../../run_a_f2 && cat skills_v1/x/SKILL.md",
              "cd outputs/run_a/../run_a_f2 && cat skills_v1/x/SKILL.md"):
        r = INV.check_no_peer_artifacts_read(_log_with(fx, c), own)
        assert r.status == "FAIL", (c, r.detail)
    # '..' that stays inside the run is still own
    for c in (f"cat {own}/digests_e2_train/../skills_v1/x/SKILL.md",
              f"cd {own}/digests_e2_train/../digests_e1_train && wc -l *.md"):
        r = INV.check_no_peer_artifacts_read(_log_with(fx, c), own)
        assert r.status == "PASS", (c, r.detail)


def test_peer_audit_a_relative_path_that_climbs_out_is_not_own(fx):
    own = fx / "outputs" / "run_a"
    own.mkdir(parents=True)
    r = INV.check_no_peer_artifacts_read(
        _log_with(fx, "cat ../outputs/run_a/skills_v1/x/SKILL.md"), own)
    assert r.status == "FAIL", r.detail


def test_peer_audit_expands_tilde(fx, monkeypatch):
    home = fx / "home"
    own = home / "work" / "repo" / "outputs" / "run_a"
    own.mkdir(parents=True)
    (own.parent / "run_a_f2").mkdir()
    monkeypatch.setenv("HOME", str(home))
    for c in ("cat ~/work/repo/outputs/run_a/skills_v1/x/SKILL.md",
              "cd ~/work/repo/outputs/run_a && cat skills_v1/x/SKILL.md"):
        r = INV.check_no_peer_artifacts_read(_log_with(fx, c), own)
        assert r.status == "PASS", (c, r.detail)
    # '~' is HOME, not a relative dir: a same-named run elsewhere under HOME is not this run
    for c in ("cat ~/work/repo/outputs/run_a_f2/skills_v1/x/SKILL.md",
              "cd ~/work/repo/outputs/run_a_f2 && cat skills_v1/x/SKILL.md",
              "cat ~/elsewhere/outputs/run_a/skills_v1/x/SKILL.md"):
        r = INV.check_no_peer_artifacts_read(_log_with(fx, c), own)
        assert r.status == "FAIL", (c, r.detail)


def _read_log(tmp: Path, path: str) -> Path:
    p = tmp / "e.log"
    p.write_text("\n".join(json.dumps(r) for r in (
        {"type": "system", "subtype": "init", "session_id": "S"},
        {"type": "assistant", "session_id": "S", "message": {"content": [
            {"type": "tool_use", "name": "Read", "input": {"file_path": path}}]}})))
    return p


# A 'staging' SUBSTRING is not ownership, and spelling around `outputs/` is not
# leaving. Found live: tau3_lingt0 e1 read outputs/tau3_a_sonnet/skills_v1_staging/*/SKILL.md "as
# an example", and the old exemption reported PASS.
def test_peer_audit_staging_is_own_only_by_path(fx):
    own = fx / "outputs" / "ownrun"
    peer = fx / "outputs" / "peerrun"
    (own / "skills_v3_staging").mkdir(parents=True)
    peer.mkdir(parents=True)
    for path in (f"{own}/skills_v3_staging/../../peerrun/skills_v1/a/SKILL.md",
                 f"{peer}/skills_v2_staging/a/SKILL.md"):
        r = INV.check_no_peer_artifacts_read(_read_log(fx, path), own)
        assert r.status == "FAIL", (path, r.detail)
    r = INV.check_no_peer_artifacts_read(
        _log_with(fx, f"cd {peer} && ls skills_v1_staging/"), own)
    assert r.status == "FAIL", r.detail
    # the run's OWN staging, by any spelling of its path, is still own
    for path in (f"{own}/skills_v3_staging/a/SKILL.md",
                 f"{own}/digests_e1_train/../skills_v3_staging/a/SKILL.md",
                 "outputs/ownrun/skills_v3_staging/a/SKILL.md"):
        r = INV.check_no_peer_artifacts_read(_read_log(fx, path), own)
        assert r.status == "PASS", (path, r.detail)


def test_peer_audit_resolves_each_cd_against_the_one_before(fx):
    own = fx / "outputs" / "ownrun"
    (own / "digests_e1_train").mkdir(parents=True)
    (fx / "outputs" / "peerrun").mkdir(parents=True)
    r = INV.check_no_peer_artifacts_read(_log_with(
        fx, f"cd {own}/digests_e1_train && cd ../../peerrun && cat skills_v1/a/SKILL.md"), own)
    assert r.status == "FAIL", r.detail
    # ...and a relative cd that lands back INSIDE the run is not leaving it
    r = INV.check_no_peer_artifacts_read(_log_with(
        fx, f"cd {own}/digests_e2_train && cd ../digests_e1_train && cat ../skills_v1/a/SKILL.md"),
        own)
    assert r.status == "PASS", r.detail


def test_peer_audit_leaving_does_not_need_the_word_outputs(fx):
    """A run dir NOT under a dir named outputs: a cd to its parent, a climbing cd, a globbed cd, a
    quoted cd, and an absolute path spelled with a glob all reach a peer run."""
    runs = fx / "runs"
    own = runs / "ownrun"
    (own / "skills_v2").mkdir(parents=True)
    (runs / "peerrun").mkdir()
    for c in (f"cd {runs} && cat peerrun/skills_v1/a/SKILL.md",
              "cd ../../peerrun && cat skills_v1/a/SKILL.md",
              f"cd {fx}/r*s/peerrun && cat skills_v1/a/SKILL.md",
              f'cd "{runs}/peerrun" && cat skills_v1/a/SKILL.md',
              f"cat {fx}/run?/peerrun/skills_v1/a/SKILL.md"):
        r = INV.check_no_peer_artifacts_read(_log_with(fx, c), own)
        assert r.status == "FAIL", (c, r.detail)
    # not leaving: the run's own dirs, a cd somewhere unrelated, a URL
    for c in (f"cd {own} && cat skills_v2/a/SKILL.md",
              f"cd {fx}/work && mkdir -p skills_draft/a",
              "curl -s https://example.com/docs/skills/index.html"):
        r = INV.check_no_peer_artifacts_read(_log_with(fx, c), own)
        assert r.status == "PASS", (c, r.detail)


# ══ other sessions' material under ~/.claude is denied; narrow grants still win ══════════
def _sb_ok(prof, *cmd) -> bool:
    return subprocess.run(["sandbox-exec", "-f", str(prof), *cmd],
                          capture_output=True, text=True).returncode == 0


@needs_sandbox
def test_profile_denies_other_sessions_under_dot_claude(fx, monkeypatch):
    home = fx / "home"
    cl = home / ".claude"
    work = cl / "jobs" / "56847add" / "tmp" / "tau3_evolver_run_e2"
    peer_work = cl / "jobs" / "56847add" / "tmp" / "tau3_evolver_peer_e2"
    projdir = cl / "projects" / "-own-session"
    for d in (work, peer_work, projdir, cl / "projects" / "-other", cl / "jobs" / "882075be"):
        d.mkdir(parents=True)
    files = {
        "own work PROMPT": (work / "PROMPT.md", True),
        "own session transcript": (projdir / "s.jsonl", True),
        "settings.json (CLI startup)": (cl / "settings.json", True),
        # The CLI `source`s its OWN snapshot on every Bash call and swallows the
        # failure, so a deny silently strips PATH and its wrappers. Aliases, not conversations.
        "shell-snapshots (the CLI sources its own)": (cl / "shell-snapshots" / "snapshot-zsh-1-a.sh",
                                                       True),
        "PEER work PROMPT": (peer_work / "PROMPT.md", False),
        "orchestrator job scratch": (cl / "jobs" / "882075be" / "state.json", False),
        "history.jsonl": (cl / "history.jsonl", False),
        "other project transcript": (cl / "projects" / "-other" / "t.jsonl", False),
    }
    (cl / "shell-snapshots").mkdir()
    for n in ("file-history", "paste-cache", "debug", "sessions", "backups"):
        (cl / n).mkdir()
        files[n] = (cl / n / "x.txt", False)
    for p, _ in files.values():
        p.write_text("SECRET" if not _ else "ok")
    monkeypatch.setenv("HOME", str(home))
    log = fx / "run" / "e2.log"
    log.parent.mkdir()
    log.write_text("")
    job = DISPATCH.EvolverJob(epoch=2, work_dir=work, prompt="", model="", log_path=log,
                              grants=[projdir], hide=[fx / "run"])
    prof = fx / "p.sb"
    prof.write_text(DISPATCH.build_evolver_sandbox_profile(job))
    for label, (p, want) in files.items():
        r = subprocess.run(["sandbox-exec", "-f", str(prof), "/bin/cat", str(p)],
                           capture_output=True, text=True)
        assert (r.returncode == 0) is want, (label, r.returncode, r.stderr)
    # Every ancestor of the evolver's cwd under jobs/ is still a readable NODE
    for anc in (cl / "jobs", cl / "jobs" / "56847add", cl / "jobs" / "56847add" / "tmp"):
        r = subprocess.run(["sandbox-exec", "-f", str(prof), "/bin/ls", str(anc)],
                           capture_output=True, text=True)
        assert r.returncode == 0, (anc, r.stderr)


@needs_sandbox
def test_profile_denies_writes_to_the_operators_config_but_not_the_clis_own_state(fx, monkeypatch):
    """~/.claude must not be writable wholesale. Everything the OPERATOR's sessions load or run is
    write-denied by name; the CLI's own run-time state and this job's work dir
    (under jobs/) stay writable, because the job's writes are granted again after the denies."""
    home = fx / "home"
    cl = home / ".claude"
    work = cl / "jobs" / "56847add" / "tmp" / "tau3_evolver_run_e2"
    other_job = cl / "jobs" / "882075be" / "tmp"
    projdir = cl / "projects" / "-own-session"
    for d in (work, other_job, projdir, cl / "skills" / "real", cl / "hooks", cl / "plugins",
              cl / "commands", cl / "agents", cl / "todos", cl / "shell-snapshots"):
        d.mkdir(parents=True, exist_ok=True)
    (cl / "settings.json").write_text("{}")
    monkeypatch.setenv("HOME", str(home))
    log = fx / "run" / "e2.log"
    log.parent.mkdir()
    log.write_text("")
    job = DISPATCH.EvolverJob(epoch=2, work_dir=work, prompt="", model="", log_path=log,
                              grants=[projdir], writes=[work], hide=[fx / "run"])
    prof = fx / "p.sb"
    prof.write_text(DISPATCH.build_evolver_sandbox_profile(job))

    def wr(p):
        return _sb_ok(prof, "/bin/sh", "-c", f'echo x > "{p}"') and Path(p).exists()
    denied = {
        "settings.json": cl / "settings.json",
        "settings.local.json": cl / "settings.local.json",
        "an atomic-write sibling of settings.json": cl / "settings.json.tmp.123",
        "CLAUDE.md": cl / "CLAUDE.md",
        "statusline.sh": cl / "statusline.sh",
        "a user skill": cl / "skills" / "real" / "SKILL.md",
        "a hook script": cl / "hooks" / "gate.sh",
        "a plugin": cl / "plugins" / "p.json",
        "a command": cl / "commands" / "c.md",
        "an agent": cl / "agents" / "a.md",
        "ANOTHER job's work dir": other_job / "PROMPT.md",
    }
    for label, p in denied.items():
        assert not wr(p), label
    assert (cl / "settings.json").read_text() == "{}"
    assert not _sb_ok(prof, "/bin/mkdir", str(cl / "skills" / "evil"))
    assert not _sb_ok(prof, "/bin/mv", str(cl / "skills"), str(cl / "skills.old"))
    allowed = {
        "own work dir (under jobs/, re-granted after the deny)": work / "PROMPT.md",
        "own session transcript dir": projdir / "s.jsonl",
        "the CLI's own run-time state": cl / "todos" / "t.json",
        "its own shell snapshot": cl / "shell-snapshots" / "snapshot-zsh-1-a.sh",
    }
    for label, p in allowed.items():
        assert wr(p), label


@needs_sandbox
def test_profile_denies_the_per_uid_cli_temp_root_and_grants_the_childs_own(fx, monkeypatch):
    """Every session's Bash task outputs and scratchpads live under /tmp/claude-<uid>, which a
    broad /private allow would leave readable AND writable. The root is denied on both sides; only
    the child's own CLAUDE_CODE_TMPDIR is granted back. fx itself lies under the per-uid root, so
    it stands in for other sessions' dirs."""
    root = DISPATCH.cli_uid_roots()[0]
    assert fx.is_relative_to(root), (fx, root)                 # the fixture geometry this needs
    other = fx / "-Users-someone-dev" / "56847add-x" / "tasks"
    other.mkdir(parents=True)
    (other / "b1.output").write_text("SEALED TEST 12/40")
    home = fx / "home"
    (home / ".claude").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    work = home / ".claude" / "jobs" / "j" / "tmp" / "w"
    work.mkdir(parents=True)
    own = fx / "ev.test"
    log = home / "e.log"
    log.write_text("")
    job = DISPATCH.EvolverJob(epoch=1, work_dir=work, prompt="", model="", log_path=log,
                              cli_tmpdir=own)
    prof = fx / "p.sb"
    body = DISPATCH.build_evolver_sandbox_profile(job)
    prof.write_text(body)
    assert f'(deny file-read* (subpath "{root}"))' in body
    assert f'(deny file-write* (subpath "{root}"))' in body
    assert not _sb_ok(prof, "/bin/cat", str(other / "b1.output")), "another session's task output"
    assert not _sb_ok(prof, "/bin/sh", "-c", f'echo x >> "{other / "b1.output"}"')
    assert (other / "b1.output").read_text() == "SEALED TEST 12/40"
    assert not _sb_ok(prof, "/bin/ls", str(fx)), "the root's entries name other sessions"
    assert not _sb_ok(prof, "/bin/mkdir", str(fx / "ev.squat")), "no other dir may be created"
    # the child's own root: created the way the CLI creates it (the missing leaves only)
    assert _sb_ok(prof, "/bin/sh", "-c", f'mkdir "{own}" && mkdir "{own}/claude-1" && '
                                         f'echo ok > "{own}/claude-1/t.output" && '
                                         f'cat "{own}/claude-1/t.output"')
    assert _sb_ok(prof, "/usr/bin/stat", str(root)), "the root NODE stays stat-able (realpath)"
    # a profile for a child that is NOT the CLI (the direct runner) grants no temp root at all
    job2 = DISPATCH.EvolverJob(epoch=1, work_dir=work, prompt="", model="", log_path=log)
    prof.write_text(DISPATCH.build_evolver_sandbox_profile(job2))
    assert not _sb_ok(prof, "/bin/cat", str(own / "claude-1" / "t.output"))


def test_minted_cli_tmpdir_fits_the_clis_socket_budget_and_is_fresh(monkeypatch):
    uid = os.getuid()
    a, b = DISPATCH.mint_cli_tmpdir("ev"), DISPATCH.mint_cli_tmpdir("ev")
    assert a != b and not Path(a).exists()
    assert a.startswith(f"/tmp/claude-{uid}/ev.")
    assert len(f"{a}/claude-{uid}".encode()) <= DISPATCH.CLI_TMP_BUDGET
    assert Path(a).resolve().parent == DISPATCH.cli_uid_roots()[0]
    for bad in ("", "a/b", "toolong"):
        with pytest.raises(ValueError):
            DISPATCH.mint_cli_tmpdir(bad)
    # over budget (a long uid): refused, never handed to a CLI whose children would then fall
    # back to the denied shared root
    monkeypatch.setattr(DISPATCH.os, "getuid", lambda: 1234567890)
    with pytest.raises(ValueError, match="budget"):
        DISPATCH.mint_cli_tmpdir("ev")


# ══ the AO does not read the whole run dir ══════════════════════════════════════════════════════
# RUNTIME.json as real runs can carry it: free-text notes that quote val results.
REAL_RUNTIME = {
    "leg_script": "/r/scripts/ssbv2_leg.sh", "oracle": "/r/outputs/x/oracle",
    "evolver_timeout_s": 3600, "digest_caches": ["/r/.cache/c1"],
    "base_leg": "/r/outputs/x/x_e0_base_train.json",
    "base_leg_val": "/r/outputs/x/x_e0_base_val.json",
    "_the_comparison_this_run_supports": "the reused base scored val 40/180 against 43 of 45",
    "_prose_note": "the val leg passed forty of one hundred and eighty",
    "_test_split": "sealed",
}


@needs_sandbox
@pytest.mark.parametrize("all_epochs", [False, True])
def test_ao_cannot_read_val_gate_or_sealed_results(fx, monkeypatch, all_epochs):
    import src.orchestrator.evolver_dispatch as ED_REAL
    r = make_run(fx, 3)
    rd = r["rd"]
    (rd / "SPEC.json").write_text("{}")
    (rd / "RUNTIME.json").write_text(json.dumps(REAL_RUNTIME))
    (rd / "ITER2_PROMPT.md").write_text("previous prompt")
    out, log = rd / "ITER3_PROMPT.md", rd / "e3_prompt_author.log"
    log.write_text("")
    # the AO CLI's private temp root, placed in the fixture (still under the per-uid root)
    monkeypatch.setattr(ED_REAL, "mint_cli_tmpdir", lambda tag: str(fx / f"{tag}.test"))
    ev = [(rd / f"digests_e{j}_train").resolve() for j in (1, 2, 3)] if all_epochs else None
    argv, prof = PA_SB._ao_sandbox(out, rd / "digests_e3_train", rd / "oracle", log, 3,
                                   evidence_dirs=ev)
    try:
        def rd_ok(p):
            return subprocess.run([*argv, "/bin/cat", str(p)], capture_output=True).returncode == 0

        def wr_ok(p):
            q = subprocess.run([*argv, "/bin/sh", "-c", f'echo x > "{p}"'], capture_output=True)
            return q.returncode == 0 and Path(p).exists()
        for hidden in ("run_v1_val.json", "E3_GATE_RESULT.json", "SEALED_TEST_RESULT.json",
                       "run_test_v3.json", "drive_e1.out", "tau3_evolver_e1.log",
                       "e1_prompt_author.log", "skills_v1/a/SKILL.md",
                       # the raw RUNTIME quotes val results in its notes
                       "RUNTIME.json"):
            assert not rd_ok(rd / hidden), hidden
        for granted in ("digests_e3_train/Debugging_01_01.md", "oracle/README.md", "SPEC.json",
                        "e3_AO_RUNTIME_VIEW.json", "ITER2_PROMPT.md", "e3_prompt_author.log"):
            assert rd_ok(rd / granted), granted
        seen = subprocess.run([*argv, "/bin/cat", str(rd / "e3_AO_RUNTIME_VIEW.json")],
                              capture_output=True, text=True).stdout
        assert "40/180" not in seen and "forty" not in seen and "base_leg_val" not in seen, seen
        assert json.loads(seen)["leg_script"] == REAL_RUNTIME["leg_script"]
        assert rd_ok(rd / "digests_e1_train/Debugging_01_01.md") is all_epochs
        assert wr_ok(out) and rd_ok(out), "its own artifact: write AND read back"
        assert not wr_ok(rd / "ANSWERS_e3.json")
        assert not wr_ok(rd / "e3_PROOFREAD.json")
        body = Path(prof).read_text()
        assert f'(allow file-read* (subpath "{rd.resolve()}"))' not in body
        assert f'(deny file-read* (subpath "{rd.resolve()}"))' in body
        # The AO's CLI is handed a private temp root, outside the run dir, and may use it
        tmp = PA_SB.ao_cli_tmpdir(argv)
        assert tmp == (fx / "ao.test").resolve() and not tmp.is_relative_to(rd.resolve())
        assert argv[:2] == ["/usr/bin/env", f"CLAUDE_CODE_TMPDIR={fx / 'ao.test'}"], argv
        q = subprocess.run([*argv, "/bin/sh", "-c", f'mkdir "{tmp}" && mkdir "{tmp}/claude-1" '
                                                   f'&& echo ok > "{tmp}/claude-1/o" && '
                                                   f'cat "{tmp}/claude-1/o"'], capture_output=True)
        assert q.returncode == 0, q.stderr
        (fx / "-other-session").mkdir()
        (fx / "-other-session" / "t.output").write_text("x")
        assert not rd_ok(fx / "-other-session" / "t.output"), "another session's task output"
    finally:
        PA_SB.ao_sandbox_cleanup(argv, prof)
    assert not os.path.exists(prof)


def test_ao_runtime_view_keeps_operational_keys_only():
    v = PA_SB.ao_runtime_view(REAL_RUNTIME)
    assert set(v) == {"_about", "leg_script", "oracle", "evolver_timeout_s", "digest_caches",
                      "base_leg"}, sorted(v)
    assert not INV.held_out_figures(json.dumps(v))
    # a value that carries a held-out figure is withheld even under an operational key
    assert "cache_tag" not in PA_SB.ao_runtime_view({"cache_tag": "val 3/9"})
    # structured values are withheld (a dict can hold a note); an unreadable RUNTIME yields nothing
    assert "digest_caches" not in PA_SB.ao_runtime_view({"digest_caches": {"note": "x"}})
    assert set(PA_SB.ao_runtime_view(None)) == {"_about"}


def test_ao_sandbox_refuses_a_spec_that_states_a_held_out_figure(fx, monkeypatch):
    import src.orchestrator.evolver_dispatch as ED_REAL
    monkeypatch.setattr(ED_REAL, "_sandbox_exec_available", lambda: True)
    r = make_run(fx, 1)
    rd = r["rd"]
    (rd / "SPEC.json").write_text('{"notes": "val 12 of 40"}')
    before = set(Path(tempfile.gettempdir()).glob("*.ao.sb"))
    try:
        res = PA_SB._ao_sandbox(rd / "ITER1_PROMPT.md", rd / "digests_e1_train", rd / "oracle",
                                rd / "e1.log", 1)
    except ValueError as e:
        assert "held-out figure" in str(e), e
    else:                                   # never leave a profile behind, even when this fails
        PA_SB.ao_sandbox_cleanup(*res)
        pytest.fail("a SPEC.json stating a held-out figure was granted to the AO")
    assert set(Path(tempfile.gettempdir()).glob("*.ao.sb")) == before, "refusal left a profile"


def test_ao_sandbox_creates_nothing_before_the_profile_is_built(fx, monkeypatch):
    """A builder exception must leave no 0-byte .ao.sb and no .work dir behind."""
    import src.orchestrator.evolver_dispatch as ED_REAL
    monkeypatch.setattr(ED_REAL, "_sandbox_exec_available", lambda: True)

    def boom(job):
        raise RuntimeError("builder failed")
    monkeypatch.setattr(ED_REAL, "build_evolver_sandbox_profile", boom)
    r = make_run(fx, 1)
    rd = r["rd"]
    tmp = Path(tempfile.gettempdir())
    before = set(tmp.glob("*.ao.sb*"))
    try:
        res = PA_SB._ao_sandbox(rd / "ITER1_PROMPT.md", rd / "digests_e1_train", rd / "oracle",
                                rd / "e1.log", 1)
    except RuntimeError:
        pass
    else:
        PA_SB.ao_sandbox_cleanup(*res)
        pytest.fail("the builder's exception did not propagate")
    left = set(tmp.glob("*.ao.sb*")) - before
    for x in left:                           # clean up before failing, never after
        shutil.rmtree(x, ignore_errors=True) if x.is_dir() else x.unlink()
    assert not left, f"a failed build left {sorted(map(str, left))}"


# ══ the loop ═══════════════════════════════════════════════════════════════════════════════════
def _ctx(fx, epoch, *, harness="claude_cli", mode=None, sandbox=True):
    r = make_run(fx, epoch)
    spec = SimpleNamespace(evolver_harness=harness, evolver_model="deepseek/deepseek-v4-flash-0731",
                           domain="ssbv2", variant="", split_path=str(r["split"]),
                           benchmark="ssbv2", n_samples=3, name="run",
                           prompt_author_model="opus[1m]")
    kw = {"evolver_sandbox": sandbox}
    if mode is not None:
        kw["evolver_evidence"] = mode
    ctx = L.Ctx(r["rd"], epoch, spec, r["rd"] / "oracle", **kw)
    ctx.evolver_timeout_s = 0
    ctx.paths.prompt.write_text("the countersigned prompt")
    return ctx, r


def _capture_dispatch(monkeypatch):
    """Replace ONLY the loop's view of subprocess: Popen records instead of launching, pgrep
    reports no live evolver. Nothing is ever executed by the code under test."""
    seen = []

    class _P:
        pid = 999999

        def wait(self, timeout=None):
            return 0

    def popen(argv, **kw):
        seen.append((list(argv), dict(kw.get("env") or {})))
        return _P()

    def run(argv, *a, **kw):
        if argv and argv[0] == "pgrep":
            return subprocess.CompletedProcess(argv, 1, "", "")
        raise AssertionError(f"unexpected subprocess.run in the loop: {argv}")
    proxy = SimpleNamespace(Popen=popen, run=run, STDOUT=subprocess.STDOUT, PIPE=subprocess.PIPE,
                            DEVNULL=subprocess.DEVNULL, TimeoutExpired=subprocess.TimeoutExpired,
                            CompletedProcess=subprocess.CompletedProcess)
    monkeypatch.setattr(L, "subprocess", proxy)
    return seen


def _dispatch(ctx):
    try:
        L._run_evolver(ctx)
    except L.Halt as e:
        return str(e)
    return ""


@pytest.mark.parametrize("mode,sandbox,want", [
    ("all_epochs", False, "evolver_sandbox=false"),
    ("everything", True, "is not one of"),
    (True, True, "is not one of"),
])
def test_bad_or_uncontained_all_epochs_halts_before_any_dispatch(fx, monkeypatch, mode, sandbox,
                                                                 want):
    ctx, _r = _ctx(fx, 3, mode=mode, sandbox=sandbox)
    seen = _capture_dispatch(monkeypatch)
    msg = _dispatch(ctx)
    assert want in msg and not seen, (msg, seen)


def _receipt(ctx, **extra):
    prof = ctx.run_dir / "p.sb"
    prof.write_text("(version 1)\n")
    d = {"sandboxed": True, "containment": "sandbox-exec", "profile": str(prof), **extra}
    (ctx.run_dir / f"e{ctx.epoch}_EVOLVER_CONTAINMENT.json").write_text(json.dumps(d))


def test_contained_verify_checks_the_evidence_it_records(fx):
    ctx, r = _ctx(fx, 3, mode="all_epochs")
    _receipt(ctx)                                             # an OLD receipt: unchanged verdict
    ok, det = L._v_evolver_contained(ctx)
    assert ok and det == "evolver ran under sandbox-exec (p.sb)", det
    ev = [str(p) for p in ctx.paths.evidence_dirs()]
    before = ES.hashes(ev)
    _receipt(ctx, evidence_mode="all_epochs", evidence_dirs=ev, evidence_sha256_before=before,
             evidence_sha256_after=dict(before))
    ok, det = L._v_evolver_contained(ctx)
    assert ok and "3 dir(s), unchanged" in det, det
    _receipt(ctx, evidence_mode="current_epoch", evidence_dirs=ev[-1:],
             evidence_sha256_before=ES.hashes(ev[-1:]))
    ok, det = L._v_evolver_contained(ctx)
    assert not ok and "did not grant the evidence this run declared" in det, det
    for bad_grant, why in ((str(r["rd"]), "run dir or a parent"),
                           (str(r["rd"].parent), "run dir or a parent"),
                           (str(r["rd"] / "digests_e4_train"), "outside the declared evidence")):
        _receipt(ctx, evidence_mode="all_epochs", evidence_dirs=ev, evidence_sha256_before=before,
                 evidence_sha256_after=dict(before), sandbox_grants=ev + [bad_grant])
        ok, det = L._v_evolver_contained(ctx)
        assert not ok and why in det, (bad_grant, det)
    _receipt(ctx, evidence_mode="all_epochs", evidence_dirs=ev, evidence_sha256_before=before)
    (r["rd"] / "digests_e1_train" / "Debugging_01_01.md").write_text("edited after dispatch")
    ok, det = L._v_evolver_contained(ctx)
    assert not ok and "CHANGED" in det and "digests_e1_train" in det, det


def test_train_only_and_coverage_cover_every_granted_dir(fx):
    ctx, r = _ctx(fx, 3, mode="all_epochs")
    ok, det = L._v_digests_train_only(ctx)
    assert ok and "3 dir(s)" in det, det
    ok, det = L._v_digests_complete(ctx)
    assert ok, det
    (r["rd"] / "digests_e1_train" / "Debugging_01_02.md").write_text(
        digest("Debugging/01_02", 1).replace("## Scorer diff — iter_0_s1 (authoritative)", ""))
    ok, det = L._v_digests_complete(ctx)
    assert not ok and "digests_e1_train" in det, det
    ok, _ = L._v_digests_complete(L.Ctx(r["rd"], 3, ctx.spec, r["rd"] / "oracle"))
    assert ok, "current-epoch mode still checks only the current dir"
    (r["rd"] / "digests_e2_train" / "Debugging_09_09.md").write_text("x")
    ok, det = L._v_digests_train_only(ctx)
    assert not ok and "HELD-OUT" in det and "digests_e2_train" in det, det


@pytest.mark.parametrize("mode", [None, "all_epochs"])
def test_ao_call_site_passes_the_evidence_list(fx, monkeypatch, mode):
    ctx, r = _ctx(fx, 2, mode=mode)
    ctx.paths.prompt.unlink()
    got = {}
    fake_facts = SimpleNamespace(as_block=lambda: "FACTS")
    monkeypatch.setattr(PA, "derive_facts", lambda *a, **k: fake_facts)

    def fake_author(**kw):
        got.update(kw)
        return {}
    monkeypatch.setattr(PA, "author_prompt", fake_author)
    L._run_author_prompt(ctx)
    if mode is None:
        assert got["evidence_dirs"] is None
    else:
        assert got["evidence_dirs"] == [(r["rd"] / f"digests_e{j}_train").resolve()
                                        for j in (1, 2)]


def test_post_run_hash_lands_in_the_receipt_and_the_trace(fx):
    ctx, r = _ctx(fx, 2, mode="all_epochs")
    ev = [str(p) for p in ctx.paths.evidence_dirs()]
    before = ES.hashes(ev)
    for name in ("e2_EVOLVER_CONTAINMENT.json", "tau3_evolver_e2.trace.json"):
        (r["rd"] / name).write_text(json.dumps({"evidence_sha256_before": before}))
    (r["rd"] / "digests_e1_train" / "Debugging_01_01.md").write_text("the evolver wrote here")
    L._record_evidence_after(ctx)
    for name in ("e2_EVOLVER_CONTAINMENT.json", "tau3_evolver_e2.trace.json"):
        after = json.loads((r["rd"] / name).read_text())["evidence_sha256_after"]
        assert after[ev[0]] != before[ev[0]] and after[ev[1]] == before[ev[1]], name


def test_ao_sandbox_refuses_a_profile_that_regrants_the_run_dir(fx, monkeypatch):
    """CHECK THE ARTIFACT: if the shared builder ever re-grants the run dir as a tree (the old
    work_dir=rd shape), the AO refuses to dispatch — and leaves no profile or scratch dir."""
    import src.orchestrator.evolver_dispatch as ED
    r = make_run(fx, 1)
    rd = r["rd"].resolve()
    real = ED.build_evolver_sandbox_profile
    monkeypatch.setattr(ED, "build_evolver_sandbox_profile",
                        lambda job: real(job) + f'\n(allow file-read* (subpath "{rd}"))\n')
    tmp = Path(tempfile.gettempdir())
    before = set(tmp.glob("*.ao.sb")) | set(tmp.glob("*.ao.sb.work"))
    try:
        res = PA_SB._ao_sandbox(rd / "ITER1_PROMPT.md", rd / "digests_e1_train", rd / "oracle",
                                rd / "e1.log", 1)
    except ValueError as e:
        assert "the run dir must stay hidden" in str(e), e
    else:                                   # never leave a profile behind, even when this fails
        PA_SB.ao_sandbox_cleanup(*res)
        pytest.fail("a profile that re-grants the run dir was accepted")
    after = set(tmp.glob("*.ao.sb")) | set(tmp.glob("*.ao.sb.work"))
    assert after <= before, after - before


# ══ the producing library travels inside each digest dir ═════════════════════
@needs_sandbox
def test_preflight_probes_a_task_file_when_the_library_sorts_first(fx):
    """"_" sorts before lowercase names, so in a tau3 digest dir (task_004.md) the first entry of
    sorted(rglob) is the _library DIRECTORY, and preflight skipped the grant without probing it.
    Made observable with an unreadable task file: the probe must reach it and report the gap."""
    d = fx / "run" / "digests_e2_train"
    (d / "_library" / "a").mkdir(parents=True)
    (d / "_library" / "a" / "SKILL.md").write_text("---\nname: a\n---\nbody\n")
    t = d / "task_004.md"
    t.write_text("# task_004\n")
    (fx / "work").mkdir()
    job = DISPATCH.EvolverJob(epoch=2, work_dir=fx / "work", prompt="", model="",
                              log_path=fx / "run" / "e2.log", grants=[d])
    os.chmod(t, 0)
    try:
        gaps = DISPATCH.preflight(job)
    finally:
        os.chmod(t, 0o644)
    assert any(f"cannot READ a required input: {t}" in g for g in gaps), gaps


# ══ the profile write-denies the run dir and the operator channel BY NAME ══════════════════════
OUTSIDE_ROOT = Path("/var/tmp").resolve() / "followups"   # outside HOME, the repo and the per-uid root


@pytest.fixture
def fx_outside():
    """A fixture OUTSIDE the home tree, the repo AND the per-uid CLI temp root. The profile denies the
    per-uid root as a whole, so a write-deny probe on a run dir inside it (fx) would pass for the
    wrong reason; here only a deny BY NAME can stop the write."""
    OUTSIDE_ROOT.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="evid_", dir=str(OUTSIDE_ROOT))).resolve()
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)
        try:
            OUTSIDE_ROOT.rmdir()                  # only when no other fixture is using it
        except OSError:
            pass


def _sb_writes(prof, path: Path) -> bool:
    q = subprocess.run(["sandbox-exec", "-f", str(prof), "/bin/sh", "-c", f'echo x >> "{path}"'],
                       capture_output=True, text=True)
    return q.returncode == 0 and path.exists()


# ══ one rule for a missing pre-dispatch hash, launcher and loop agree ════════════════════════
def test_annotate_fails_without_a_usable_before_hash(fx, monkeypatch):
    r = make_run(fx, 2)
    rd = r["rd"]
    receipt = fx / "receipt.json"
    argv = ["--annotate", str(receipt), "", "all_epochs", str(rd / "digests_e2_train"),
            str(rd / "digests_e1_train"), "--", "--", "--grant", str(rd / "digests_e1_train"),
            "--no-write", str(rd)]
    assert ES.main(argv) == 0
    rec = json.loads(receipt.read_text())
    assert set(rec["evidence_sha256_before"]) == set(rec["evidence_dirs"])
    assert rec["sandbox_no_writes"] == [str(rd)]
    # hashing raised: rc 3, and the receipt says why
    bad = rd / "digests_e2_train" / "zzz_unreadable.bin"
    bad.write_text("x")
    os.chmod(bad, 0)
    try:
        receipt.unlink()
        assert ES.main(argv) == 3
        assert "evidence_sha256_error" in json.loads(receipt.read_text())
    finally:
        os.chmod(bad, 0o644)
    # hashing "succeeded" with nothing to show for it: still rc 3 (the old exit was 0)
    receipt.unlink()
    monkeypatch.setattr(ES, "hashes", lambda dirs: {})
    assert ES.main(argv) == 3
    assert ES._before_hash_gap(str(receipt), ["/x"]) == "the before-hash is empty"
    receipt.write_text(json.dumps({"evidence_sha256_before": {"/x": "h"}}))
    assert ES._before_hash_gap(str(receipt), ["/x", "/y"]) == "no before-hash for ['/y']"
    assert ES._before_hash_gap(str(receipt), ["/x"]) == ""


def test_contained_verify_needs_the_before_hash_only_in_all_epochs_mode(fx):
    ctx, _r = _ctx(fx, 3, mode="all_epochs")
    ev = [str(p) for p in ctx.paths.evidence_dirs()]
    _receipt(ctx, evidence_mode="all_epochs", evidence_dirs=ev, evidence_sha256_before={},
             evidence_sha256_error="PermissionError: x")
    ok, det = L._v_evolver_contained(ctx)
    assert not ok and "records no pre-dispatch hash (PermissionError: x)" in det, det
    ctx, _r = _ctx(fx / "cur", 3)
    cur = [str(ctx.paths.digests)]
    _receipt(ctx, evidence_mode="current_epoch", evidence_dirs=cur, evidence_sha256_before={},
             evidence_sha256_error="PermissionError: x")
    ok, det = L._v_evolver_contained(ctx)
    assert ok and "NOTE: no pre-dispatch evidence hash recorded (PermissionError: x)" in det, det
    # a hash that IS recorded is still compared in current-epoch mode
    _receipt(ctx, evidence_mode="current_epoch", evidence_dirs=cur,
             evidence_sha256_before=ES.hashes(cur))
    (ctx.paths.digests / "Debugging_01_01.md").write_text("edited after dispatch")
    ok, det = L._v_evolver_contained(ctx)
    assert not ok and "CHANGED" in det, det


# ══ the AO's parent-directory literals stay under HOME ════════════════════════════════════════
@needs_sandbox
def test_ao_ancestor_nodes_never_reopen_the_per_uid_root(fx):
    """fx lies under /tmp/claude-<uid>. The AO's cwd (the run dir) is granted as a node, but its
    ancestors outside HOME are not: one of them is the per-uid root, whose listing names every
    session on this host."""
    r = make_run(fx, 1)
    rd = r["rd"]
    root = DISPATCH.cli_uid_roots()[0]
    assert rd.resolve().is_relative_to(root), "the fixture must sit under the per-uid root"
    argv, prof = PA_SB._ao_sandbox(rd / "ITER1_PROMPT.md", rd / "digests_e1_train",
                                   rd / "oracle", rd / "e1.log", 1)
    try:
        body = Path(prof).read_text()
        assert f'(allow file-read* (literal "{rd.resolve()}"))' in body, "cwd node kept"
        for anc in list(rd.resolve().parents):
            assert f'(allow file-read* (literal "{anc}"))' not in body, anc
        assert subprocess.run([*argv, "/bin/ls", str(root)], capture_output=True).returncode != 0
        assert subprocess.run([*argv, "/bin/ls", str(fx)], capture_output=True).returncode != 0
        assert subprocess.run([*argv, "/bin/ls", str(rd)], capture_output=True).returncode == 0
    finally:
        PA_SB.ao_sandbox_cleanup(argv, prof)
    # a cwd under HOME keeps its ancestors up to HOME (a denied ancestor is fatal there)
    argv, prof = PA_SB._ao_sandbox(rd / "ITER1_PROMPT.md", rd / "digests_e1_train",
                                   rd / "oracle", rd / "e1.log", 1, cwd=ROOT)
    try:
        body = Path(prof).read_text()
        for anc in (ROOT.resolve(), ROOT.resolve().parent, Path.home()):
            assert f'(allow file-read* (literal "{anc}"))' in body, anc
    finally:
        PA_SB.ao_sandbox_cleanup(argv, prof)


# ══ the parent session's environment never reaches a child ═══════════════════════════════════
PARENT_VARS = {"CLAUDE_CODE_MESSAGING_SOCKET": "/tmp/SECRET-SOCK",
               "CLAUDE_CODE_MESSAGING_TOKEN": "SECRET-TOKEN",
               "CLAUDE_JOB_DIR": "/x/SECRET-JOB",
               # distinctive: a bare "bg" matched random fixture names in the receipt
               "CLAUDE_CODE_SESSION_KIND": "SECRET-KIND-bg",
               "CLAUDE_CODE_SESSION_ID": "SECRET-SID"}


def test_parent_session_env_names_are_exactly_the_session_ones():
    env = {**PARENT_VARS, "CLAUDECODE": "1", "CLAUDE_PID": "1", "CLAUDE_CODE_ENTRYPOINT": "cli",
           "CLAUDE_CODE_EXECPATH": "/x", "CLAUDE_CODE_BRIDGE_SESSION_ID": "b",
           "CLAUDE_CODE_CHILD_SESSION": "1", "CLAUDE_CODE_SESSION_ATTENDED": "1",
           "CLAUDE_CODE_FUTURE_SESSION_THING": "1",              # caught by the pattern
           # kept: auth, the child's own temp root, model/config
           "CLAUDE_CODE_OAUTH_TOKEN": "t", "CLAUDE_CODE_TMPDIR": "/tmp/claude-1/ev.x",
           "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "1", "CLAUDE_EFFORT": "high", "PATH": "/bin",
           "ANTHROPIC_MODEL": "m"}
    kept = {"CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_TMPDIR", "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
            "CLAUDE_EFFORT", "PATH", "ANTHROPIC_MODEL"}
    names = DISPATCH.parent_session_env_names(env)
    assert names == sorted(set(env) - kept), names
    out, scrubbed = DISPATCH.scrub_parent_session_env(env)
    assert scrubbed == names and set(out) == kept


# ══ a dry run's preflight never lists the real per-uid CLI temp root ══════════════════════════
@needs_sandbox
@pytest.mark.parametrize("dry", [True, False])
def test_preflight_step5_touches_the_real_uid_root_only_outside_dry_runs(fx, monkeypatch, dry):
    real_root = DISPATCH.cli_uid_roots()[0]
    real_run = subprocess.run
    probed = []

    def spy(argv, *a, **kw):
        probed.append(list(argv))
        if argv and argv[-1] == str(real_root):
            # never execute a probe of the REAL root from a test: answer as a working deny would
            return subprocess.CompletedProcess(argv, 0 if argv[-2] == "/usr/bin/stat" else 1,
                                               "", "")
        return real_run(argv, *a, **kw)
    monkeypatch.setattr(DISPATCH.subprocess, "run", spy)
    (fx / "work").mkdir()
    job = DISPATCH.EvolverJob(epoch=1, work_dir=fx / "work", prompt="", model="",
                              log_path=fx / "run" / "e1.log",
                              cli_tmpdir=Path(DISPATCH.mint_cli_tmpdir("ev")), dry_run=dry)
    before = set(Path(tempfile.gettempdir()).glob("evodisp_uidroot.*"))
    gaps = DISPATCH.preflight(job)
    assert not [g for g in gaps if "per-uid" in g], gaps
    targets = [a[-1] for a in probed if a[:1] == ["sandbox-exec"]]
    if dry:
        assert str(real_root) not in targets, "a dry run listed the REAL per-uid root"
        stand = [t for t in targets if "evodisp_uidroot." in t]
        assert stand, "the dry run probed no stand-in at all"
        assert set(Path(tempfile.gettempdir()).glob("evodisp_uidroot.*")) == before, "leftover"
    elif real_root.is_dir():
        assert str(real_root) in targets
        assert not [t for t in targets if "evodisp_uidroot." in t]


@needs_sandbox
def test_dry_run_stand_in_probe_catches_broken_rules(fx, monkeypatch):
    """The stand-in probe is not decoration: with the per-uid rules dropped from the builder it
    reports the root listable and another session's output readable."""
    real_build = DISPATCH.build_evolver_sandbox_profile

    def no_root_rules(job, uid_roots=None):
        return real_build(job, uid_roots=[] if uid_roots is not None else None)
    monkeypatch.setattr(DISPATCH, "build_evolver_sandbox_profile", no_root_rules)
    (fx / "work").mkdir()
    job = DISPATCH.EvolverJob(epoch=1, work_dir=fx / "work", prompt="", model="",
                              log_path=fx / "run" / "e1.log", dry_run=True)
    gaps = DISPATCH._probe_uid_root_standin(job, real_build(job))
    assert any("LISTABLE" in g for g in gaps) and any("READABLE" in g for g in gaps), gaps


# ══ the receipt's by-name write denies are checked, not only recorded ══════════════════════════
def test_contained_verify_requires_the_recorded_write_denies(fx):
    ctx, r = _ctx(fx, 3)
    rd = r["rd"].resolve()
    op = L.operator_dir(rd)
    _receipt(ctx)                                             # no field at all (older receipt): passes
    assert L._v_evolver_contained(ctx)[0]
    _receipt(ctx, sandbox_no_writes=[str(rd), str(op)])
    ok, det = L._v_evolver_contained(ctx)
    assert ok, det
    for nw, lost in (([str(rd)], "the operator channel"), ([str(op)], "the run dir"),
                     ([], "the run dir or the operator channel"),
                     ([f"{rd}/.operator", str(rd)], "the operator channel")):
        _receipt(ctx, sandbox_no_writes=nw)
        ok, det = L._v_evolver_contained(ctx)
        assert not ok and f"no by-name write deny on {lost}" in det, (nw, det)
    # an UNSANDBOXED receipt carries no denies and is judged on containment alone
    (ctx.run_dir / "e3_EVOLVER_CONTAINMENT.json").write_text(json.dumps(
        {"sandboxed": False, "containment": "NONE", "sandbox_no_writes": []}))
    ctx.evolver_sandbox = False
    assert L._v_evolver_contained(ctx)[0]


# Preflight step 6 probes every declared no-write under the real profile.
@needs_sandbox
def test_preflight_probes_the_declared_no_writes(fx_outside, monkeypatch):
    fx = fx_outside
    rd = fx / "run"
    (rd / "work").mkdir(parents=True)
    op = fx / "run.operator"                                  # absent: probed with its own mkdir
    job = DISPATCH.EvolverJob(epoch=1, work_dir=rd / "work", prompt="", model="",
                              log_path=rd / "e1.log", writes=[rd / "work"], hide=[rd],
                              no_write=[rd, op], dry_run=True)
    gaps = DISPATCH.preflight(job)
    assert not [g for g in gaps if "no-write" in g], gaps
    real_build = DISPATCH.build_evolver_sandbox_profile

    def no_denies(job, uid_roots=None):                     # the by-name write denies dropped
        return "\n".join(ln for ln in real_build(job, uid_roots=uid_roots).splitlines()
                         if not ln.startswith("(deny file-write* (subpath \"" + str(fx)))
    monkeypatch.setattr(DISPATCH, "build_evolver_sandbox_profile", no_denies)
    gaps = DISPATCH.preflight(job)
    bad = [g for g in gaps if "no-write" in g]
    assert len(bad) == 2 and str(rd) in bad[0] and str(op) in bad[1], gaps
    assert not op.exists() and not list(rd.glob(".evodisp_nowrite_probe.*")), "probe left behind"


# The IDE-integration channel.
def test_parent_session_env_includes_the_ide_channel_by_whole_token():
    env = {"CLAUDE_CODE_SSE_PORT": "1", "ENABLE_IDE_INTEGRATION": "true",
           "CLAUDE_CODE_AUTO_CONNECT_IDE": "1", "CLAUDE_CODE_IDE_SKIP_AUTO_INSTALL": "1",
           "CLAUDE_CODE_IDE_HOST_OVERRIDE": "h",
           # configuration that merely CONTAINS the letters: kept
           "CLAUDE_CODE_OVERRIDE_X": "1", "CLAUDE_CODE_MAX_PROCESSES": "1",
           "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"}
    names = DISPATCH.parent_session_env_names(env)
    assert names == sorted(["CLAUDE_CODE_SSE_PORT", "ENABLE_IDE_INTEGRATION",
                            "CLAUDE_CODE_AUTO_CONNECT_IDE", "CLAUDE_CODE_IDE_SKIP_AUTO_INSTALL",
                            "CLAUDE_CODE_IDE_HOST_OVERRIDE"]), names


# Whatever HOME says, the AO's parent nodes never reopen the per-uid CLI temp root.
@needs_sandbox
def test_ao_ancestor_nodes_ignore_a_home_at_or_above_the_per_uid_root(fx, monkeypatch):
    r = make_run(fx, 1)
    rd = r["rd"]
    root = DISPATCH.cli_uid_roots()[0]
    assert rd.resolve().is_relative_to(root)
    monkeypatch.setenv("HOME", str(root.parent))              # e.g. HOME=/tmp
    argv, prof = PA_SB._ao_sandbox(rd / "ITER1_PROMPT.md", rd / "digests_e1_train",
                                   rd / "oracle", rd / "e1.log", 1)
    try:
        body = Path(prof).read_text()
        assert f'(allow file-read* (literal "{rd.resolve()}"))' in body, "cwd node kept"
        for anc in (root, *root.parents):
            assert f'(allow file-read* (literal "{anc}"))' not in body, anc
        assert f'(allow file-read* (literal "{rd.resolve().parent}"))' in body, \
            "vacuous: HOME did not reach the filter (nodes under it are kept)"
        assert subprocess.run([*argv, "/bin/ls", str(root)], capture_output=True).returncode != 0
    finally:
        PA_SB.ao_sandbox_cleanup(argv, prof)


@needs_sandbox
def test_ao_ancestor_nodes_outside_home_and_the_uid_root_are_not_granted(fx_outside):
    """The HOME filter on its own: a run dir outside both HOME and the per-uid root keeps no
    ancestor node either (the broad system allow already covers them)."""
    r = make_run(fx_outside, 1)
    rd = r["rd"]
    argv, prof = PA_SB._ao_sandbox(rd / "ITER1_PROMPT.md", rd / "digests_e1_train",
                                   rd / "oracle", rd / "e1.log", 1)
    try:
        body = Path(prof).read_text()
        for anc in list(rd.resolve().parents):
            assert f'(allow file-read* (literal "{anc}"))' not in body, anc
    finally:
        PA_SB.ao_sandbox_cleanup(argv, prof)


@needs_sandbox
def test_builder_ancestor_nodes_ignore_a_home_at_or_above_the_per_uid_root(fx, monkeypatch):
    """The shared builder's own ancestor list (cwd and repo nodes): a work dir beneath the
    per-uid root with HOME above it must not re-grant the root after its deny."""
    root = DISPATCH.cli_uid_roots()[0]
    work = fx / "work"
    work.mkdir()
    assert work.resolve().is_relative_to(root)
    monkeypatch.setenv("HOME", str(root.parent))
    job = DISPATCH.EvolverJob(epoch=1, work_dir=work, prompt="", model="",
                              log_path=fx / "run" / "e1.log")
    body = DISPATCH.build_evolver_sandbox_profile(job)
    for anc in (root, *root.parents):
        assert f'(allow file-read* (literal "{anc}"))' not in body, anc
    assert f'(allow file-read* (literal "{work.resolve().parent}"))' in body, \
        "vacuous: HOME did not reach the filter (nodes under it are kept)"
    assert f'(allow file-read-metadata (literal "{root}"))' in body, "the root's stat node stays"
    prof = fx / "b.sb"
    prof.write_text(body)
    assert subprocess.run(["sandbox-exec", "-f", str(prof), "/bin/ls", str(root)],
                          capture_output=True).returncode != 0, "the per-uid root is listable"
    assert subprocess.run(["sandbox-exec", "-f", str(prof), "/bin/ls", str(work)],
                          capture_output=True).returncode == 0, "the work dir must stay readable"
