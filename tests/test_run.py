"""`evoskill run` (src/run.py), offline: nothing here reaches Daytona, a model or the network.

  * the Run size section of an instruction (both shipped instructions, the accepted forms, the refusals);
  * init on both task types writes a run the port's own checks accept: INFRA.json validates, RUNTIME.json is the
    report's run configuration, and once a frozen split exists the spec validates and freezes and infra.agree finds
    no disagreement;
  * the adapt step after the adaptor (a stand-in plug-in and reference/ in its place): the real plug-in freeze and
    verify, the split check, REFERENCE_CHECK.json and `infra check`, all local (macOS: the freeze needs sandbox-exec);
  * the step order, the skip of a complete step, --stop-after, and the refusal of a changed input;
  * --dry-run prints the commands and never a secret, writes nothing and starts nothing;
  * the run's environment; `stop` holds legs, stops live legs and keeps the evolver session; the leg wait.
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.run as R  # noqa: E402
from src.orchestrator import infra, plugin_runner  # noqa: E402
from src.orchestrator.spec import ExperimentSpec  # noqa: E402

EX = {"workbook": ROOT / "examples" / "ssbv2", "tau3": ROOT / "examples" / "tau3_banking"}
DATA_NAME = {"workbook": "spreadsheetbench-v2", "tau3": "tau3_data_363133a"}   # what the instructions name
TOKEN_SECRET = "sk-ant-oat01-TOKENCONTENT-must-never-print"
API_SECRET = "sk-ant-api03-APIKEY-must-never-print"
OR_SECRET = "sk-or-v1-OPENROUTER-must-never-print"
DT_SECRET = "dtn_DAYTONA-must-never-print"
needs_sandbox = pytest.mark.skipif(not os.path.exists("/usr/bin/sandbox-exec"), reason="macOS sandbox-exec only")


def _plugin_id(task_type: str, tid: str) -> str:
    return f"banking_knowledge/{tid}" if task_type == "tau3" else tid


@pytest.fixture(autouse=True)
def _private_caches(tmp_path, monkeypatch):
    """The run's caches and machine registry live under tmp_path, never the repository's .cache."""
    monkeypatch.setattr(R, "CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(R, "STORE_ROOT", tmp_path / "cache" / "ssbv2_daytona")


def world(tmp_path: Path, task_type: str, *, name: str = None, env_extra: dict = None, **over) -> NS:
    """A stand-in data root named as the instruction names it, a 0600 token file, the example's instruction,
    split and grader, and the arguments of `evoskill run`."""
    data = tmp_path / "material" / DATA_NAME[task_type]
    data.mkdir(parents=True)
    tok = tmp_path / "secrets" / "cc_token"
    tok.parent.mkdir()
    tok.write_text(TOKEN_SECRET)
    tok.chmod(0o600)
    a = NS(instruction=str(EX[task_type] / "instruction.md"), data=str(data), split=str(EX[task_type] / "split.json"),
           grader=str(EX[task_type] / "grader.py"), run_dir=str(tmp_path / "runs" / (name or f"fx_{task_type}")),
           task_type=task_type, solver_model=None, evolver_model=None,
           adaptor_model=None, token_file=str(tok), concurrency=R.DEFAULT_CONCURRENCY,
           stop_after=None, dry_run=False)
    for k, v in over.items():
        setattr(a, k, v)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path / "home"),
           "DAYTONA_API_KEY": DT_SECRET, "OPENROUTER_API_KEY": OR_SECRET, **(env_extra or {})}
    plan = R.make_plan(a, env)
    out = []
    run = R.Run(plan, R.run_env(plan, env), out=out.append, sleep=lambda s: None)
    return NS(a=a, env=env, plan=plan, run=run, out=out, data=data, token=tok, rd=plan.run_dir)


def frozen_split(w: NS) -> Path:
    """setup/split.json as plugin_runner.freeze writes it from a plug-in that lists the example split's tasks."""
    user = json.loads(Path(w.a.split).read_text())
    tasks = []
    for key, s in (("train", "train"), ("val", "validation"), ("test", "test")):
        tasks += [{"id": _plugin_id(w.plan.task_type, t), "split": s} for t in R.split_ids(user, key)]
    p = w.rd / "setup" / plugin_runner.SPLIT_JSON
    p.write_bytes(plugin_runner.canonical_json(plugin_runner.split_from_tasks(tasks)))
    return p


# ══ the Run size section ═════════════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("task_type", sorted(EX))
def test_both_shipped_instructions_declare_three_epochs_of_three_samples(task_type):
    assert R.parse_run_size((EX[task_type] / "instruction.md").read_text()) == (3, 3)


@pytest.mark.parametrize("task_type,want", [
    ("workbook", ("claude-sonnet-5-5", "claude-sonnet-5-5", "claude-sonnet-5-5")),
    ("tau3", ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-sonnet-5-5")),
])
def test_default_models_are_the_reports_per_task_type(tmp_path, task_type, want):
    w = world(tmp_path, task_type)
    assert (w.plan.solver_model, w.plan.evolver_model, w.plan.adaptor_model) == want
    # an explicit model wins over the task type's default
    w2 = world(tmp_path / "x", task_type, solver_model="claude-opus-5-5")
    assert w2.plan.solver_model == "claude-opus-5-5" and w2.plan.evolver_model == want[1]
    helptext = " ".join(R.build_parser()._subparsers._group_actions[0].choices["run"].format_help().split())
    assert "tau3 claude- haiku-5-5" in helptext or "tau3 claude-haiku-5-5" in helptext


def test_run_size_forms_and_refusals():
    assert R.parse_run_size("intro\n## Run size\n\n2 epochs, 5 attempts per task.\n") == (2, 5)
    assert R.parse_run_size("### 4) Run Size\n1 epoch · 1 sample per task\n### 5. Other\n9 epochs") == (1, 1)
    # numbers outside the section do not count
    with pytest.raises(R.RunError, match="must give both numbers"):
        R.parse_run_size("### Run size\n3 epochs\n### Notes\n3 samples per task\n")
    with pytest.raises(R.RunError, match="no 'Run size' section"):
        R.parse_run_size("### 1. Benchmark\n3 epochs · 3 samples per task\n")
    with pytest.raises(R.RunError, match="at least 1"):
        R.parse_run_size("### Run size\n0 epochs · 3 samples per task\n")


def test_the_instructions_data_paths_must_be_what_the_adaptor_gets(tmp_path):
    text = (EX["workbook"] / "instruction.md").read_text()
    split = EX["workbook"] / "split.json"
    assert R.instruction_path_problems(text, tmp_path / "spreadsheetbench-v2", split) == []
    probs = R.instruction_path_problems(text, tmp_path / "ssbv2", split)
    assert len(probs) == 1 and "/data/spreadsheetbench-v2" in probs[0] and "/data/ssbv2" in probs[0]
    text = (EX["tau3"] / "instruction.md").read_text()
    assert R.instruction_path_problems(text, tmp_path / "tau3_data_363133a", EX["tau3"] / "split.json") == []


# ══ init: the port's own checks accept the run ═════════════════════════════════════════════════════════════
@pytest.mark.parametrize("task_type", sorted(EX))
def test_init_writes_a_run_the_ports_spec_and_infra_checks_accept(tmp_path, monkeypatch, task_type):
    w = world(tmp_path, task_type)
    assert R.preflight(w.plan, w.run.env) == []
    done, why = w.run.check_init()
    assert not done and why == "no RUN.json"
    w.run.run_init()
    done, why = w.run.check_init()
    assert done, why
    rd = w.rd
    inf = infra.load(rd / "INFRA.json")
    assert inf["run"] == rd.name and inf["claude_code"]["token_file"] == str(w.token)
    assert inf["solver"]["harness"] == "claude_code" and inf["claude_code"]["version"] == R.CC_VERSION
    assert inf["data"]["root"] == str(w.data) and inf["data"]["overlay_patches_json"] is None
    rt = json.loads((rd / "RUNTIME.json").read_text())
    # the report's run configuration
    assert (rt["prompt_mode"], rt["evolver_session"], rt["evolver_daytona"]["placement"]) == \
        ("template", "stateful", "pool_container")
    assert rt["pipeline_val"] is True and rt["skip_last_train"] is True
    assert Path(rt["leg_script"]) == R.LEG_SCRIPT and rt["cc_leg"]["concurrency"] == 30
    assert "test" in rt["daytona_leg"]["pool_keys"]
    import src.orchestrator.loop as L
    assert L._pipeline_runtime(rt)["skip_last_train"] is True
    sys.path.insert(0, str(ROOT / "scripts" / "daytona"))
    import cc_pool_host as H
    assert H.cc_leg_settings(rd)["concurrency"] == 30            # no retired or unknown cc_leg key
    assert (rd / "setup" / "benchmark_grader.py").read_bytes() == Path(w.a.grader).read_bytes()
    rec = json.loads((rd / R.RUN_RECORD).read_text())
    assert rec["run_size"] == {"n_epochs": 3, "n_samples": 3}
    assert TOKEN_SECRET not in json.dumps(rec) + json.dumps(inf) + json.dumps(rt)
    # after the adaptor: SPEC.json on the frozen split validates and freezes, and INFRA agrees with all of it
    spec = R.spec_for(w.plan, frozen_split(w))
    assert spec.validate() == [], spec.validate()
    spec.freeze(rd)
    loaded, _ = ExperimentSpec.load(rd)
    assert (loaded.n_epochs, loaded.n_samples, loaded.solver_harness) == (3, 3, "claude_code")
    assert loaded.benchmark == ("ssbv2" if task_type == "workbook" else "tau3")
    monkeypatch.setenv("EVOSKILL_CC_VERSION", R.CC_VERSION)
    assert infra.agree(rd, check_env=False) == []


def test_api_key_run_declares_no_token_file(tmp_path):
    w = world(tmp_path, "workbook", token_file=None, env_extra={"ANTHROPIC_API_KEY": API_SECRET})
    assert w.plan.credential == {"kind": "api_key", "token_file": None}
    assert w.run.env["ANTHROPIC_API_KEY"] == API_SECRET and "EVOSKILL_CC_TOKEN_FILE" not in w.run.env
    w.run.run_init()
    inf = infra.load(w.rd / "INFRA.json")
    assert inf["claude_code"]["token_file"] is None and inf["solver"]["key_file"] is None
    assert API_SECRET not in (w.rd / "INFRA.json").read_text() + (w.rd / R.RUN_RECORD).read_text()


# ══ adapt after the adaptor: freeze, verify, split, reference, SPEC, infra check ═══════════════════════════════
PLUGIN = '''from pathlib import Path
from plugin_api import Task

SETS = %(sets)r


def list_tasks(root: Path):
    root = Path(root)
    return [Task(id=t, category=t.split("/")[0], instruction="do " + t, inputs=(),
                 answer_files=(root / t / "answer.txt",), grading={}, split=s) for t, s in SETS.items()]
'''


def adaptor_output(w: NS, *, leak: bool = False) -> None:
    """What the adaptor would leave: a plug-in listing the example split's tasks over the stand-in data root, and a
    reference/ with the training answers (and, with leak, a held-out task's answer)."""
    user = json.loads(Path(w.a.split).read_text())
    sets = {}
    for key, s in (("train", "train"), ("val", "validation"), ("test", "test")):
        for t in R.split_ids(user, key):
            tid = _plugin_id(w.plan.task_type, t)
            sets[tid] = s
            (w.data / tid).mkdir(parents=True, exist_ok=True)
            (w.data / tid / "answer.txt").write_text(f"the answer of {tid}\n")
    (w.rd / "setup" / "benchmark_plugin.py").write_text(PLUGIN % {"sets": sets})
    # the canonical interface the adaptor's driver puts next to the grader (setup_ao_probe.ensure_given_files)
    shutil.copyfile(plugin_runner.API_FILE, w.rd / "setup" / "plugin_api.py")
    ref = w.rd / "reference"
    ref.mkdir()
    (ref / "README.md").write_text("training material\n")
    first_train = next(t for t, s in sets.items() if s == "train")
    shutil.copyfile(w.data / first_train / "answer.txt", ref / "train_answer.txt")
    if leak:
        first_test = next(t for t, s in sets.items() if s == "test")
        shutil.copyfile(w.data / first_test / "answer.txt", ref / "leaked.txt")


@needs_sandbox
@pytest.mark.parametrize("task_type", sorted(EX))
def test_adapt_freezes_checks_and_agrees_after_the_adaptor(tmp_path, task_type):
    w = world(tmp_path, task_type)
    w.run.run_init()
    adaptor_output(w)
    called = []
    real_exec = w.run.exec

    def exec_no_adaptor(cmd, what, **kw):
        assert "setup_ao_probe" not in cmd.text(), "the adaptor must not run when its output exists"
        called.append(what)
        return real_exec(cmd, what, **kw)

    w.run.exec = exec_no_adaptor
    assert not w.run.check_adapt()[0]
    w.run.run_adapt()
    assert called == ["plugin_runner freeze", "plugin_runner verify", "infra check"]
    done, why = w.run.check_adapt()
    assert done, why
    spec, _ = ExperimentSpec.load(w.rd)
    assert Path(spec.split_path) == w.rd / "setup" / "split.json" and spec.n_test == len(
        R.split_ids(json.loads(Path(w.a.split).read_text()), "test"))
    rc = json.loads((w.rd / R.REFERENCE_CHECK).read_text())
    assert rc["ok"] and rc["files"] == 2


@needs_sandbox
def test_adapt_refuses_a_reference_holding_a_held_out_answer(tmp_path):
    w = world(tmp_path, "tau3")
    w.run.run_init()
    adaptor_output(w, leak=True)
    with pytest.raises(R.RunError, match="leaked.txt is an answer file of test task"):
        w.run.run_adapt()
    assert not (w.rd / "SPEC.json").exists()


def test_adapt_refuses_a_partial_adaptor_output(tmp_path):
    w = world(tmp_path, "workbook")
    w.run.run_init()
    (w.rd / "reference").mkdir()
    with pytest.raises(R.RunError, match="partial"):
        w.run.run_adapt()


def test_split_check_maps_the_plugins_domain_prefix_and_names_a_moved_task():
    user = {"train": ["task_001"], "val": ["task_002"], "test": ["task_003"]}
    frozen = {"train": ["banking_knowledge/task_001"], "val": ["banking_knowledge/task_002"],
              "test": ["banking_knowledge/task_003"], "unused": []}
    assert R.split_problems(frozen, user) == []
    frozen["val"], frozen["test"] = frozen["test"], frozen["val"]
    probs = R.split_problems(frozen, user)
    assert len(probs) == 2 and probs[0].startswith("val")


# ══ steps: order, skip, --stop-after, refusal ═════════════════════════════════════════════════════════════════
def fake_steps(state: dict, calls: list, *, broken=()):
    def step(name):
        def check():
            return state.get(name, False), name
        def run():
            calls.append(name)
            if name not in broken:
                state[name] = True
        return (name, check, run)
    return [step(s) for s in R.STEPS]


def test_steps_run_in_order_and_a_complete_step_is_skipped():
    state, calls, out = {}, [], []
    R.run_steps(fake_steps(state, calls), None, out=out.append)
    assert calls == list(R.STEPS)
    calls.clear()
    R.run_steps(fake_steps(state, calls), None, out=out.append)
    assert calls == [] and all("skipped" in ln for ln in out[-5:])
    state, calls = {"init": True, "adapt": True}, []
    R.run_steps(fake_steps(state, calls), None, out=out.append)
    assert calls == ["base", "evolve", "test"]


def test_stop_after_and_an_incomplete_step_end_the_sequence():
    state, calls, out = {}, [], []
    R.run_steps(fake_steps(state, calls), "adapt", out=out.append)
    assert calls == ["init", "adapt"] and "stopped after adapt" in out[-1]
    state, calls = {"init": True}, []
    with pytest.raises(R.RunError, match="step base ran but its output is not complete"):
        R.run_steps(fake_steps(state, calls, broken=("base",)), None, out=out.append)
    assert calls == ["adapt", "base"]


def test_a_changed_input_is_refused(tmp_path):
    w = world(tmp_path, "workbook")
    w.run.run_init()
    # the instruction's content
    edited = tmp_path / "instruction.md"
    edited.write_text(Path(w.a.instruction).read_text() + "\nOne more sentence.\n")
    w2 = R.Run(R.make_plan(NS(**{**vars(w.a), "instruction": str(edited)}), w.env), w.run.env, out=[].append)
    with pytest.raises(R.RunError, match="instruction_sha256: the content changed"):
        w2.check_init()
    # a model
    w3 = R.Run(R.make_plan(NS(**{**vars(w.a), "evolver_model": "claude-haiku-5-5"}), w.env), w.run.env,
               out=[].append)
    with pytest.raises(R.RunError, match="evolver_model: was 'claude-sonnet-5-5', now 'claude-haiku-5-5'"):
        w3.check_init()
    # the data
    (w.data / "new_file").write_text("x")
    with pytest.raises(R.RunError, match="data_fingerprint"):
        w.run.check_init()


def test_init_refuses_a_foreign_non_empty_dir_and_a_name_with_caches(tmp_path):
    w = world(tmp_path, "workbook")
    w.rd.mkdir(parents=True)
    (w.rd / "something").write_text("x")
    with pytest.raises(R.RunError, match="not a run dir made by"):
        w.run.run_init()
    w2 = world(tmp_path / "b", "workbook", name="taken")
    (R.CACHE_ROOT / "taken_e0_cache").mkdir(parents=True)
    assert any("already has caches" in p for p in R.preflight(w2.plan, w2.run.env))


# ══ --dry-run ═══════════════════════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("task_type", sorted(EX))
def test_dry_run_prints_every_command_and_no_secret(tmp_path, monkeypatch, capsys, task_type):
    tok = tmp_path / "tok"
    tok.write_text(TOKEN_SECRET)
    tok.chmod(0o600)
    for k, v in (("ANTHROPIC_API_KEY", API_SECRET), ("OPENROUTER_API_KEY", OR_SECRET),
                 ("DAYTONA_API_KEY", DT_SECRET), ("EVOSKILL_DIGEST_CACHE", "/stray")):
        monkeypatch.setenv(k, v)

    def no_process(*a, **k):
        raise AssertionError(f"a dry run started a process: {a[:1]}")

    monkeypatch.setattr(subprocess, "Popen", no_process)
    monkeypatch.setattr(subprocess, "run", no_process)
    rd = tmp_path / "runs" / "dry"
    fake_data = tmp_path / "nowhere" / DATA_NAME[task_type]            # never created: a dry run needs no data
    rc = R.main(["run", "--dry-run", "--instruction", str(EX[task_type] / "instruction.md"), "--data", str(fake_data),
                 "--split", str(EX[task_type] / "split.json"), "--grader", str(EX[task_type] / "grader.py"),
                 "--run-dir", str(rd), "--task-type", task_type, "--token-file", str(tok)])
    out = capsys.readouterr()
    text = out.out + out.err
    assert rc == 0, text
    for secret in (TOKEN_SECRET, API_SECRET, OR_SECRET, DT_SECRET):
        assert secret not in text
    assert not rd.exists() and not (tmp_path / "nowhere").exists()
    for needle in ("setup_ao_probe.py", "plugin_runner freeze", "plugin_runner verify", "infra check",
                   "probe-skills", "ssbv2_leg_cc.sh dry_base_train train", "ssbv2_leg_cc.sh dry_base_test test",
                   "src.orchestrator.loop --run-dir", "--drive --until-epoch 3",
                   "ssbv2_leg_cc.sh dry_v3_test test skills_v3", "cc_pool_host.py stop",
                   "EVOSKILL_CLAUDE_EFFORT=default", f"EVOSKILL_CC_VERSION={R.CC_VERSION}", "DISABLE_AUTOUPDATER=1",
                   "ANTHROPIC_API_KEY (unset)", "EVOSKILL_DIGEST_CACHE (unset)", "no such directory"):
        assert needle in text, needle
    assert ("OPENROUTER_API_KEY (unset)" in text) == (task_type == "workbook")


def test_dry_run_reports_a_missing_credential_instead_of_refusing(tmp_path, monkeypatch, capsys):
    for k in ("ANTHROPIC_API_KEY", "EVOSKILL_CC_TOKEN_FILE"):
        monkeypatch.delenv(k, raising=False)
    rc = R.main(["run", "--dry-run", "--instruction", str(EX["workbook"] / "instruction.md"), "--data", str(tmp_path),
                 "--split", str(EX["workbook"] / "split.json"), "--grader", str(EX["workbook"] / "grader.py"),
                 "--run-dir", str(tmp_path / "r"), "--task-type", "workbook", "--stop-after", "init"])
    text = capsys.readouterr().out
    assert rc == 0 and "credential: no Claude Code credential" in text


# ══ the environment ═════════════════════════════════════════════════════════════════════════════════════════
def test_run_env_strips_overrides_and_routes_the_credential(tmp_path):
    w = world(tmp_path, "workbook")
    base = {**w.env, "ANTHROPIC_API_KEY": API_SECRET, "EVOSKILL_SKILLS_DIR": "/x", "EVOSKILL_PYTHON": "/py",
            "ANTHROPIC_BASE_URL": "http://proxy", "EVO_CONC": "7", "SSBV2_ROOT": "/elsewhere"}
    env = R.run_env(w.plan, base)
    assert env["EVOSKILL_CC_TOKEN_FILE"] == str(w.token) and "ANTHROPIC_API_KEY" not in env
    assert "EVOSKILL_SKILLS_DIR" not in env and env["EVOSKILL_PYTHON"] == "/py"
    assert not {"ANTHROPIC_BASE_URL", "EVO_CONC", "SSBV2_ROOT", "OPENROUTER_API_KEY"} & set(env)
    assert (env["EVOSKILL_CLAUDE_EFFORT"], env["EVOSKILL_CC_VERSION"], env["EVOSKILL_EXPECT_CLAUDE_CODE_VERSION"],
            env["DISABLE_AUTOUPDATER"]) == ("default", R.CC_VERSION, R.CC_VERSION, "1")
    assert env["EVOSKILL_SSBV2_DATA"] == str(w.data)
    t = world(tmp_path / "t", "tau3")
    env = R.run_env(t.plan, {**t.env, "EVOSKILL_SSBV2_DATA": "/x"})
    assert env["OPENROUTER_API_KEY"] == OR_SECRET and "EVOSKILL_SSBV2_DATA" not in env


# ══ stop, status, legs ══════════════════════════════════════════════════════════════════════════════════════
def test_stop_holds_legs_stops_live_legs_and_keeps_the_evolver_session(tmp_path):
    w = world(tmp_path, "workbook")
    w.run.run_init()
    rd = w.rd
    (rd / "e1_evolver_session").mkdir()                     # an open evolver session
    with pytest.raises(R.RunError, match="evolver session of this run is open"):
        R.cmd_stop(NS(run_dir=str(rd), machine=True), out=[].append, kill=lambda *a: None)
    pid_file = rd / R.LEG_PID_DIR / f"{rd.name}_v1_train.pid"
    pid_file.parent.mkdir(parents=True)
    pid_file.write_text(json.dumps({"pid": 424242, "tag": f"{rd.name}_v1_train"}))
    held = open(pid_file)
    fcntl.flock(held, fcntl.LOCK_EX)
    killed, out = [], []
    try:
        assert R.cmd_stop(NS(run_dir=str(rd), machine=False), out=out.append,
                          kill=lambda pid, sig: killed.append((pid, sig))) == 0
    finally:
        held.close()
    assert killed == [(424242, R.signal.SIGTERM)]
    assert (rd / R.HOLD_FILE).read_text().startswith(R.HOLD_MARK)
    assert any("evolver session and the run's machine are kept" in ln for ln in out)
    # the next `evoskill run` lifts its own hold, never someone else's
    w.run.release_hold()
    assert not (rd / R.HOLD_FILE).exists()
    (rd / R.HOLD_FILE).write_text("held by hand\n")
    with pytest.raises(R.RunError, match="not written by `evoskill stop`"):
        w.run.release_hold()


def test_status_reads_the_run_dir_only(tmp_path):
    w = world(tmp_path, "tau3")
    w.run.run_init()
    leg = w.rd / f"{w.rd.name}_base_train.json"
    leg.write_text(json.dumps({"sample_pass": 19, "sample_total": 57, "pass3": 9, "all3": 4, "grader_failed": 0}))
    out = []
    assert R.cmd_status(NS(run_dir=str(w.rd)), out=out.append) == 0
    text = "\n".join(out)
    assert "[init] complete" in text and "[adapt] not complete" in text
    assert "19/57 attempts passed" in text and "evolver session: not started" in text


def test_wait_legs_returns_on_the_result_and_stops_on_a_failure(tmp_path):
    w = world(tmp_path, "workbook")
    w.rd.mkdir(parents=True)
    tag = f"{w.rd.name}_base_train"

    def appear(_s):
        w.run.leg_path(tag).write_text(json.dumps({"sample_pass": 1, "sample_total": 3}))

    w.run.sleep = appear
    w.run.wait_legs([tag], {tag: R.time.time()})
    assert any("1/3 attempts passed" in ln for ln in w.out)
    tag2 = f"{w.rd.name}_base_val"
    w.run.leg_path(tag2, ".FAILED.json").write_text(json.dumps({"INCOMPLETE": True, "problems": ["machine gone"]}))
    with pytest.raises(R.RunError, match="machine gone"):
        w.run.wait_legs([tag2], {tag2: R.time.time()})
    tag3 = f"{w.rd.name}_base_test"
    with pytest.raises(R.RunError, match="is not running and wrote no result"):
        w.run.wait_legs([tag3], {tag3: 0.0})


@pytest.mark.parametrize("task_type", R.TASK_TYPES)
def test_templates_leave_every_daytona_snapshot_ref_unpinned(tmp_path, task_type):
    """A Daytona snapshot ref is known only after Daytona builds the Dockerfile, so the shipped templates pin none:
    null = accept the ref Daytona reports and record it (the Dockerfile sha256 stays the image's identity)."""
    w = world(tmp_path, task_type)
    inf, rt = R.infra_for(w.plan), R.runtime_for(w.plan)
    assert inf["leg_sandbox"]["snapshot_ref"] is None and inf["claude_sandbox"]["snapshot_ref"] is None
    assert len(inf["leg_sandbox"]["dockerfile_sha256"]) == 64
    assert rt["daytona_leg"]["require_snapshot_ref"] is None
    assert rt["evolver_daytona"]["require_snapshot_ref"] is None and rt["evolver_daytona"]["require_leg_snapshot_ref"] is None
