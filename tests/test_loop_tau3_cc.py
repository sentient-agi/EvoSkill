"""A tau3 banking run driven by the SAME loop as SSB. SPEC benchmark "tau3" + solver_harness "claude_code" marks
the run; the leg launcher is the benchmark-agnostic scripts/ssbv2_leg_cc.sh (cc_leg_driver.py ... --runner
run_ssbv2_claude_code).

What is checked, on a scratch tau3 run dir (the report's SPEC and RUNTIME shape from test_loop_daytona_dryrun, the
neutral INFRA fixture tests/fixtures/ssb_leg_pool/infra_cc.json with the tau3 declarations, the token a 0600
stand-in, an empty stand-in data root):
  * infra.agree == [] (check_env=False): the SSB-only statements (MAX_TURNS, EXPECTED, the SSB data root) are not
    compared for tau3, and Dockerfile.tau3_leg (the SSB leg Dockerfile's bytes + the /opt/tau3 layer) satisfies the
    Claude-image base rule through its base;
  * _evolver_host passes; the leg env carries EVOSKILL_CACHE_DIR; RUNTIME without skip_last_train + pipeline_val
    true gives skip_last_train true (the last epoch's train leg feeds no later evolver);
  * the pipelined 3-epoch drive (test_loop_pipeline's simulated world) with a tau3 claude_code spec and legs whose
    command line is the CC driver's: train+val at epochs 1-2, ONLY val at epoch 3;
  * an SSB run: the same cache env, an explicit skip_last_train false honoured;
  * the task rows a tau3 plug-in returns pass the freeze's validation, and their ids pass the leg driver's unit and
    the grade worker's job patterns.

OFFLINE: no Daytona, no model call, no network; agree() never opens the token stand-in.
"""
from __future__ import annotations

import builtins
import hashlib
import io
import json
import shutil
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

REPO = Path(__file__).resolve().parents[1]

import src.orchestrator.loop as L  # noqa: E402
from src.orchestrator import infra  # noqa: E402
from src.orchestrator.spec import ExperimentSpec, split_digest  # noqa: E402
from tests import test_loop_daytona_dryrun as D  # noqa: E402  (the report's SPEC / RUNTIME shape)
from tests import test_loop_pipeline as P  # noqa: E402  (the simulated pipelined world)

TAU3_DF = "scripts/daytona/Dockerfile.tau3_leg"
TAU3_SNAP = "daytona-70c842b98285d64d2944f338e2a51956857c8084a524915dd8ab765d28f6a25a:daytona"
TAU3_SPLIT = REPO / "examples" / "tau3_banking" / "split.json"
INFRA_FIXTURE = REPO / "tests" / "fixtures" / "ssb_leg_pool" / "infra_cc.json"
USER_SIM = "openrouter/openai/gpt-5-nano"
SOLVER = "claude-sonnet-5-5"
RUN = "tau3_cc_scratch"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _jload(p: Path) -> dict:
    return json.loads(p.read_text())


def _jdump(p: Path, d) -> None:
    p.write_text(json.dumps(d, indent=1))


def _token(tmp_path: Path) -> Path:
    t = tmp_path / "secrets" / "cc_token_standin"
    t.parent.mkdir(parents=True, exist_ok=True)
    t.write_text("sk-ant-oat01-NOT-A-TOKEN")
    t.chmod(0o600)
    return t


def tau3_run(tmp_path: Path) -> NS:
    """A scratch tau3 run dir: SPEC benchmark tau3 / banking_knowledge / claude_code / user sim gpt-5-nano, split =
    <run>/setup/split.json with plug-in ids; INFRA leg_sandbox = Dockerfile.tau3_leg (its real sha), data.root = a
    stand-in dir, no overlay; RUNTIME = the report's pool_container + stateful + evo layout + pipeline_val true,
    NO skip_last_train, the tau3 leg snapshot."""
    rd = tmp_path / "runs" / RUN
    (rd / "setup").mkdir(parents=True)
    tok = _token(tmp_path)
    user_split = _jload(TAU3_SPLIT)
    split = {k: [f"banking_knowledge/{t}" for t in user_split[k]] for k in ("train", "val", "test")}
    split["unused"] = [f"banking_knowledge/{t}" for t in user_split["dropped"]["task_ids"]]
    sp_path = rd / "setup" / "split.json"
    _jdump(sp_path, split)
    spec = ExperimentSpec(
        name=RUN, benchmark="tau3", domain="banking_knowledge", split_path=str(sp_path),
        split_sha256=split_digest(sp_path), n_train=len(split["train"]), n_val=len(split["val"]),
        n_test=len(split["test"]), n_samples=3, solver_model=SOLVER, solver_provider_order=(),
        solver_allow_fallbacks=False, user_sim_model=USER_SIM, judge_model="none", agent_turns=45,
        episode_timeout_s=10800, solver_harness="claude_code", evolver_model="claude-opus-5-5",
        evolver_harness="claude_cli", n_epochs=3, cost_ceiling_usd=200.0)
    assert spec.validate() == [], spec.validate()
    spec.freeze(rd)

    inf = _jload(INFRA_FIXTURE)
    inf["run"] = RUN
    df_sha = _sha(REPO / TAU3_DF)
    inf["leg_sandbox"].update(dockerfile=TAU3_DF, dockerfile_sha256=df_sha, image_id="x86daytona-" + df_sha[:12],
                              snapshot_ref=TAU3_SNAP)
    inf["claude_sandbox"]["snapshot_ref"] = D.EVOLVER_SNAP    # the report RUNTIME below pins one (the fixture: null)
    data_root = tmp_path / "tau3_data"
    data_root.mkdir()
    inf["data"] = {"root": str(data_root), "overlay_patches_json": None, "overlay_patches_sha256": None}
    inf["other_models"]["user_sim"] = USER_SIM
    inf["solver"]["model"] = inf["solver"]["expect_served_model"] = inf["ao"]["model"] = SOLVER
    inf["claude_code"]["token_file"] = inf["solver"]["key_file"] = str(tok)
    _jdump(rd / "INFRA.json", inf)

    rt = D.report_runtime(rd, tmp_path, tok)
    for k in ("skip_last_train", "base_leg", "base_leg_val", "digest_caches"):
        rt.pop(k, None)
    rt["daytona_leg"]["require_snapshot_ref"] = TAU3_SNAP
    rt["daytona_leg"]["pool_keys"] = ["train", "val"]
    rt["evolver_daytona"]["require_leg_snapshot_ref"] = TAU3_SNAP
    _jdump(rd / "RUNTIME.json", rt)
    loaded, _ = ExperimentSpec.load(rd)
    return NS(rd=rd, token=tok, spec=loaded, rt=rt)


def ctx_kwargs(rt: dict) -> dict:
    """Ctx kwargs from a RUNTIME dict by main()'s presence reads (the keys this run declares), plus the real
    _pipeline_runtime (the skip_last_train default lives there)."""
    kw = {"leg_script": rt["leg_script"], "shared_dir": rt["shared_dir"], "evolver_sandbox": rt["evolver_sandbox"],
          "evolver_evidence": rt["evolver_evidence"], "evidence_layout": rt["evolver_evidence_layout"],
          "evolver_session": rt["evolver_session"], "evolver_host": rt["evolver_host"],
          "evolver_daytona": dict(rt["evolver_daytona"]), "evolver_timeout_s": int(rt["evolver_timeout_s"]),
          "prompt_mode": rt["prompt_mode"], "leg_timeout_s": rt["leg_timeout_s"]}
    kw.update(L._pipeline_runtime(rt))
    return kw


@pytest.fixture
def tau3(tmp_path):
    return tau3_run(tmp_path)


# ══ 1. infra: the tau3 declaration agrees ════════════════════════════════════════════════════════════════════
def test_the_tau3_run_infra_agrees_and_never_opens_the_token(tau3, monkeypatch):
    banned = {str(tau3.token)}
    real_open = io.open

    def guarded(file, *a, **k):
        if str(file) in banned:
            raise AssertionError(f"agree() opened the token {file}")
        return real_open(file, *a, **k)
    monkeypatch.setattr(io, "open", guarded)
    monkeypatch.setattr(builtins, "open", guarded)
    assert infra.validate(_jload(tau3.rd / "INFRA.json")) == []
    assert infra.agree(tau3.rd, {}, check_env=False) == []


def test_the_ssb_only_statements_are_not_compared_for_tau3_and_the_rest_still_are(tau3):
    inf = _jload(tau3.rd / "INFRA.json")
    # what an SSB-only comparison would refuse: the tau3 leg image is not the SSB leg image (EXPECTED)
    assert inf["leg_sandbox"]["dockerfile_sha256"] != inf["claude_sandbox"]["base_dockerfile_sha256"]
    assert infra._benchmark(tau3.rd) == "tau3" and not infra._ssb("tau3")
    assert infra._ssb("ssbv2") and infra._ssb(None)
    # an agent_turns that MAX_TURNS would refuse is no disagreement for tau3 (the bridge passes no --max-turns)
    sp = _jload(tau3.rd / "SPEC.json")
    sp["spec"]["agent_turns"] = 7
    _jdump(tau3.rd / "SPEC.json", sp)
    assert infra.agree(tau3.rd, {}, check_env=False) == []
    # ... and is one again for an SSB run (the same files, benchmark ssbv2)
    sp["spec"]["benchmark"] = "ssbv2"
    _jdump(tau3.rd / "SPEC.json", sp)
    got = {(d.source, d.field) for d in infra.agree(tau3.rd, {}, check_env=False)}
    assert {(f"{infra.CC_SOLVER_LIB} MAX_TURNS", "agent_turns"),
            (f"{infra.LEG_DRIVER_LOOP} EXPECTED", "snapshot_ref")} <= got, got
    # every benchmark: SPEC/RUNTIME statements, the disk shas, the data root on disk
    sp["spec"].update(benchmark="tau3", agent_turns=45, user_sim_model="openrouter/openai/gpt-5")
    _jdump(tau3.rd / "SPEC.json", sp)
    rt = _jload(tau3.rd / "RUNTIME.json")
    rt["daytona_leg"]["require_snapshot_ref"] = "daytona-other:daytona"
    _jdump(tau3.rd / "RUNTIME.json", rt)
    inf["data"]["root"] = str(tau3.rd / "no_such_data_root")
    inf["leg_sandbox"]["dockerfile_sha256"] = "0" * 64
    inf["leg_sandbox"]["image_id"] = "x86daytona-" + "0" * 12
    _jdump(tau3.rd / "INFRA.json", inf)
    got = {(d.source, d.field) for d in infra.agree(tau3.rd, {}, check_env=False)}
    assert got == {("SPEC.json", "user_sim_model"), ("RUNTIME.json", "daytona_leg.require_snapshot_ref"),
                   ("disk", f"sha256({TAU3_DF})"), ("disk", "data.root")}, got


def test_the_layered_leg_dockerfile_rule(tau3, tmp_path):
    inf = _jload(tau3.rd / "INFRA.json")
    # only a LAYERED leg Dockerfile may differ from the Claude image's base; any other still may not
    other = json.loads(json.dumps(inf))
    other["leg_sandbox"]["dockerfile"] = "scripts/daytona/Dockerfile.other_leg"
    assert any("claude_sandbox.base_dockerfile_sha256 != leg_sandbox.dockerfile_sha256" in p
               for p in infra.validate(other))
    # its base is re-checked on disk: the base sha and the byte prefix (a repo mirror with drift planted)
    repo = tmp_path / "repo"
    for rel in (infra.LEG_DRIVER_BASE, infra.LEG_DRIVER_LOOP, infra.LEG_LIB_LOOP, infra.EVOLVER_MODULE,
                infra.PINS_MODULE, infra.RUNNER, infra.CC_SOLVER_LIB, TAU3_DF, "scripts/daytona/Dockerfile.ssb_leg",
                "scripts/daytona/Dockerfile.ssb_evolver"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, repo / rel)
    assert infra.agree(tau3.rd, {}, repo=repo, check_env=False) == []
    t = repo / TAU3_DF
    t.write_bytes(b"# not the base first\n" + t.read_bytes())
    inf["leg_sandbox"]["dockerfile_sha256"] = _sha(t)
    inf["leg_sandbox"]["image_id"] = "x86daytona-" + inf["leg_sandbox"]["dockerfile_sha256"][:12]
    _jdump(tau3.rd / "INFRA.json", inf)
    got = {(d.source, d.field) for d in infra.agree(tau3.rd, {}, repo=repo, check_env=False)}
    assert got == {("disk", f"{TAU3_DF} prefix")}, got
    b = repo / "scripts/daytona/Dockerfile.ssb_leg"
    b.write_text(b.read_text() + "# drift\n")
    got = {(d.source, d.field) for d in infra.agree(tau3.rd, {}, repo=repo, check_env=False)}
    assert ("disk", "sha256(scripts/daytona/Dockerfile.ssb_leg)") in got


# ══ 2. the loop's switches ═══════════════════════════════════════════════════════════════════════════════════
def _ctx(run, epoch=1, **over):
    kw = ctx_kwargs(run.rt)
    kw.update(over)
    return L.Ctx(run.rd, epoch, run.spec, run.rd / "oracle", **kw)


def test_evolver_host_cache_env_and_skip_last_train(tau3, monkeypatch):
    ctx = _ctx(tau3)
    assert ctx.skip_last_train is True and ctx.pipeline_val is True and ctx.leg_pool is True     # the default
    assert L._evolver_host(ctx) == "daytona"
    monkeypatch.setattr(L, "_live_runners", lambda c=None: [])
    argv, env = L._leg_argv_env(ctx, "train")
    assert argv == ["bash", tau3.rt["leg_script"], f"{RUN}_v1_train", "train", "skills_v1"]
    assert env["EVOSKILL_CACHE_DIR"] == str(REPO / ".cache" / f"{RUN}_e1_cache")
    assert env["EVOSKILL_SOLVER_MODEL"] == SOLVER and env["EVOSKILL_POOL_NO_TRAIN_SIBLING"] == "0"
    c3 = _ctx(tau3, epoch=3)
    assert L._skip_train_epoch(c3) == 3 and L._skip_train_here(c3)
    assert L._leg_argv_env(c3, "val")[1]["EVOSKILL_POOL_NO_TRAIN_SIBLING"] == "1"


def test_the_stateful_template_pipeline_program_is_the_ssb_one(tau3, monkeypatch):
    """The epoch program a tau3 claude_code run gets is the SSB run's, step for step (no step keys on benchmark)."""
    monkeypatch.setattr(L, "_prompt_mode", lambda c: "template")
    monkeypatch.setattr(L, "_evolver_session", lambda c: "stateful")
    ssb = NS(**{**vars(tau3.spec), "benchmark": "ssbv2", "domain": "spreadsheetbench-v2"})
    for e in (1, 2, 3):
        t, s = _ctx(tau3, epoch=e), _ctx(tau3, epoch=e)
        s.spec = ssb
        assert P._sig(L._steps_for(t)) == P._sig(L._steps_for(s))
    names = [st.name for st in L._steps_for(_ctx(tau3, epoch=3))]
    assert L.STATEFUL_LAYOUT_STEP in names and "evidence surface is complete" not in names
    assert "evidence surface is train-only" not in names and L.VAL_TRACKED_STEP in names


def test_the_evidence_layout_and_digest_branches_are_unreachable_for_the_evo_layout(tau3):
    ctx = _ctx(tau3, epoch=2)
    assert L._evidence_layout(ctx) == "evo" and L._evidence_mode(ctx) == "all_epochs"
    assert L._evidence_dirs(ctx) == [Path(tau3.rd).resolve() / f"evo_e{k}" for k in (0, 1)]
    assert L._layout_missing(ctx, ctx.paths.evo, 2) == []


# ══ 3. the pipelined drive: train+val at epochs 1-2, ONLY val at epoch 3 ══════════════════════════════════════
def test_the_tau3_pipelined_drive_skips_the_last_train_leg(tau3, tmp_path, monkeypatch, capsys):
    spec = NS(**{**vars(P.SPEC), "benchmark": "tau3", "solver_harness": "claude_code", "n_epochs": 3})
    rd = tmp_path / "simrun"
    rd.mkdir()
    rt = {"pipeline_val": True, "daytona_leg": {"pool": True}}           # no skip_last_train key: the default
    kw = dict(leg_timeout_s=100000, **L._pipeline_runtime(rt))
    assert kw["skip_last_train"] is True
    W = P.World(rd, P.DUR)

    def dispatch(ctx, key):
        pid = W.dispatch(ctx, key)
        if pid:      # the leg's command line is the Claude Code driver's, whatever the benchmark
            W.procs[pid]["cmd"] = (f"/py scripts/daytona/cc_leg_driver.py leg --runner run_ssbv2_claude_code "
                                   f"--tag {ctx.paths.leg(key).stem} --key {key}")
        return pid
    monkeypatch.setattr(L, "epoch_steps", lambda: P.fake_steps(W))
    monkeypatch.setattr(L, "_dispatch_leg", dispatch)
    monkeypatch.setattr(L, "_proc_cmd", W.cmd)
    monkeypatch.setattr(L, "_proc_lstart", W.lstart)
    monkeypatch.setattr(L, "time", P.FakeTime(W))

    def no_spawn(c):
        raise AssertionError("no background re-grade expected")
    monkeypatch.setattr(L, "_pipeline_spawn_remedy", no_spawn)
    L._conv_once(L.Ctx(rd, 1, spec, rd / "oracle", **kw), "invocation:1:sim", speaker="operator",
                 channel="invocation", content="simulated drive",
                 meta={"argv": f"--run-dir {rd} --epoch 1 --step --drive --until-epoch 3", "mode": "drive"})
    said = []
    rc = L._drive(L.Ctx(rd, 1, spec, rd / "oracle", **kw), NS(until_epoch=3, max_steps=500), rd, spec,
                  rd / "oracle", kw, lambda text, key, c=None: said.append((key, text)), lambda c: "")
    out = capsys.readouterr().out
    assert rc == 0, out
    stem = lambda n, k: L.EpochPaths(rd, n).leg(k).stem       # noqa: E731
    assert [W.launches.get(stem(n, "train"), 0) for n in (1, 2, 3)] == [1, 1, 0]
    assert [W.launches.get(stem(n, "val"), 0) for n in (1, 2, 3)] == [1, 1, 1]
    assert [s for e, s in P.ran_steps(said) if e == 3][-1] == L.VAL_TRACKED_STEP
    r = P.reg(rd)
    assert r["train"]["3"]["skipped_by_design"] and all(r["legs"][str(n)]["state"] == "passed" for n in (1, 2, 3))
    assert {r["legs"][str(n)]["tag"] for n in (1, 2, 3)} == {stem(n, "val") for n in (1, 2, 3)}
    assert "skip_last_train: epoch 3 launches ONLY its val leg" in out
    # a drive past the last epoch is refused before any step (the key is on unless RUNTIME says false)
    assert L._drive(L.Ctx(rd, 1, spec, rd / "oracle", **kw), NS(until_epoch=4, max_steps=5), rd, spec,
                    rd / "oracle", kw, lambda *a, **k: None, lambda c: "") == 1


# ══ 4. SSB: the same cache env, an explicit skip_last_train false honoured ═══════════════════════════════════
def test_an_ssb_run_keeps_its_cache_env_and_an_explicit_skip_last_train_false(tmp_path, monkeypatch):
    rd = tmp_path / D.NAME
    rd.mkdir()
    D._freeze(rd)
    rt = D.report_runtime(rd, tmp_path, _token(tmp_path))
    rt["skip_last_train"] = False
    kw = L._pipeline_runtime(rt)
    assert kw == {"pipeline_val": True, "skip_last_train": False, "leg_pool": True}, "explicit false is honoured"
    spec, _ = ExperimentSpec.load(rd)
    ctx = L.Ctx(rd, 3, spec, rd / "oracle", leg_script=rt["leg_script"], **kw)
    assert L._skip_train_epoch(ctx) is None
    monkeypatch.setattr(L, "_live_runners", lambda c=None: [])
    env = L._leg_argv_env(ctx, "train")[1]
    assert env["EVOSKILL_CACHE_DIR"] == str(REPO / ".cache" / f"{D.NAME}_e3_cache")
    # pipeline off / absent: no skip_last_train key appears
    assert L._pipeline_runtime({"pipeline_val": False}) == {"pipeline_val": False}
    assert L._pipeline_runtime({"daytona_leg": {"pool": True}}) == {"leg_pool": True}


# ══ 5. freeze compatibility: hand-built tau3 task rows ═════════════════════════════════════════════════════════
def tau3_rows(tmp_path: Path) -> list:
    """The task list a tau3 plug-in would return: one row per banking task, id 'banking_knowledge/task_NNN', no
    input, the task file as the answer file (stand-in files here), grading {domain, retrieval_variant alltools},
    the family split (dropped = None)."""
    sp = _jload(TAU3_SPLIT)
    role = {**{t: "train" for t in sp["train"]}, **{t: "validation" for t in sp["val"]},
            **{t: "test" for t in sp["test"]}, **{t: None for t in sp["dropped"]["task_ids"]}}
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()
    rows = []
    for t in sorted(role):
        f = tasks_dir / f"{t}.json"
        f.write_text(json.dumps({"id": t}))
        rows.append({"id": f"banking_knowledge/{t}", "category": "banking_knowledge",
                     "instruction": f"banking conversation {t}", "inputs": [], "answer_files": [str(f)],
                     "grading": {"domain": "banking_knowledge", "retrieval_variant": "alltools"}, "split": role[t]})
    return rows


def test_plugin_validate_accepts_tau3_rows_and_the_ids_pass_the_unit_and_job_patterns(tmp_path, monkeypatch):
    from src.orchestrator import plugin_runner as PR
    monkeypatch.syspath_prepend(str(REPO / "scripts" / "cc_solver"))
    monkeypatch.syspath_prepend(str(REPO / "scripts" / "daytona"))
    import cc_solver_lib as CL
    import grade_worker as GW
    sp = _jload(TAU3_SPLIT)
    rows = tau3_rows(tmp_path)
    n = {k: len(sp[k]) for k in ("train", "val", "test")}
    assert len(rows) == sum(n.values()) + len(sp["dropped"]["task_ids"])
    assert not [r for r in rows if PR._check_task_dict(r)]
    rep = PR.validate(rows)
    assert rep["ok"], rep["problems"]
    assert rep["split_counts"] == {**n, "unused": len(sp["dropped"]["task_ids"])}
    assert rep["n_answer_groups"] == len(rows) and rep["answer_groups_spanning_sets"] == 0
    split = PR.split_from_tasks(rows)
    assert split["train"][0] == f"banking_knowledge/{sorted(sp['train'])[0]}"
    for tid in (t for k in ("train", "val", "test") for t in split[k]):
        assert CL.UNIT_RE.match(tid) and CL.UNIT_RE.match(f"{tid}#s3").group("k") == "3"
        assert GW.JOB_ID_RE.fullmatch(tid) and GW.JOB_ID_RE.fullmatch(f"{tid}#s1")
