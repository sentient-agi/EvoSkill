"""The tau3 rollout bundle writer, scripts/tau3_cc/tau2_bundle.py, and the scripted golden pair,
tests/fixtures_tau2_scripted.py. The conversations run on tau2's alltools retrieval variant, the rollouts', made
process-local by
scripts/tau3_cc/tau3_retrieval.install(root, key=None) in the child, as the bridge does (no key: offline). The
scripted pair calls only the tasks' expected actions (domain tools; no knowledge base tool), all of which alltools has.

OFFLINE: every tau2 step runs in a CHILD process under the tau3-bench venv (tests/tau3_golden.TAU3_PY, CPython 3.12,
tau2 1.0.1), with tau3_golden.child_env(): PYTHON_DOTENV_DISABLED=1, LITELLM_LOCAL_MODEL_COST_MAP=True, TAU2_DATA_DIR
and nothing of this process's environment, so no .env file is read and no key reaches tau2, plus PATH = TOOL_PATH
(alltools' shell sandbox needs srt and rg) and dead proxies. No model is called: the agent and the customer are the
scripted pair, and the only judged banking task (task_102, reward basis NL_ASSERTION) is left out. The data root is
the run's curated root third_party/tau3_data_363133a, or $EVOSKILL_TAU3_DATA_ROOT: it must hold alltools'
precomputed .embeddings_cache. Skipped when the venv, the data root, its cache, srt or rg is absent.

What must hold, for every banking task but task_102 (96): build_text_orchestrator with the scripted pair and
tau2_bundle.run_conversation play tau3_golden's golden conversation message for message, ended by the customer
(user_stop); write_bundle writes simulation.json, run.json and final_db.json, and the user's grader
(examples/tau3_banking/grader.py) scores the bundle 1.0, twice; final_db.json's
hashes equal the gold environment's (the task's expected actions applied to a fresh environment, as tau2's
EnvironmentEvaluator builds it) and its dumps reproduce them; simulation.json has no raw_data key although the
conversation's messages carried one, and it reads back to the same SimulationRun; run.json records how the
conversation ran; a second write into the same place, or a write where tau2/ is a planted link to a directory, is
refused and writes nothing there. And ending the conversation from inside the agent: a
HarnessEnd on the agent's closing turn (every expected action already played, the databases already the gold ones)
gives agent_error and a bundle that grades 0.0; one on the agent's first turn gives its own reason
(infrastructure_error) and grades 0.0 too; any other exception propagates.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
BUNDLE_DIR = ROOT / "scripts" / "tau3_cc"
BUNDLE_PY = BUNDLE_DIR / "tau2_bundle.py"
FIXTURE = TESTS / "fixtures_tau2_scripted.py"
GRADER = ROOT / "examples" / "tau3_banking" / "grader.py"
PLUGIN_API = ROOT / "src" / "orchestrator" / "plugin_api.py"
N_CHILDREN = 4
WALL_S = 2400.0                 # a rollout wall limit, tau2's own timeout (40 minutes)
FILES = ["final_db.json", "run.json", "simulation.json"]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


G = _load(TESTS / "tau3_golden.py", "tau3_golden")
TR = _load(BUNDLE_DIR / "tau3_retrieval.py", "tau3_retrieval_for_bundle_tests")   # standard library at import
VARIANT = TR.VARIANT                                        # "alltools", the rollouts' variant
CURATED_ROOT = ROOT / "third_party" / "tau3_data_363133a"   # the run's data root: it holds alltools' embeddings cache
DATA_ROOT = Path(os.environ.get(G.DATA_ROOT_ENV) or CURATED_ROOT)
TOOL_PATH = G.tool_path()                                   # srt, rg for tau2's shell (Linux: + bwrap, socat)
DEAD_PROXY = "http://127.0.0.1:9"


def seed_of(task_id: str, sample: int = 0) -> int:
    """A seed per (task, sample): the first 31 bits of sha256('<task id>#s<k>')."""
    return int(hashlib.sha256(f"{task_id}#s{sample}".encode()).hexdigest(), 16) >> 225


# One child runs a chunk of tasks: the scripted conversation through tau2's Orchestrator, the bundle, the grades, the
# harness ends; the results as JSON.
CHILD = r'''
import importlib.util, json, sys, time
from pathlib import Path

tests_dir, bundle_dir, grader_path, api_path, chunk_path, out_path, work, wall = sys.argv[1:9]
sys.path[:0] = [tests_dir, bundle_dir]
import tau3_golden as G
import fixtures_tau2_scripted as S
import tau2_bundle as B
import tau3_retrieval as TR
from tau2.data_model.message import ToolMessage
from tau2.data_model.simulation import SimulationRun
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.runner.build import build_text_orchestrator, build_user
from tau2.utils.utils import get_dict_hash


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A = load(api_path, "plugin_api")
GR = load(grader_path, "benchmark_grader")
S.register()
S.register()                                            # a second registration in one process is a no-op
VAR = TR.VARIANT
RETRIEVAL = TR.install(G.data_root(), key=None)         # as the bridge, offline: before any environment is built


def cfg(**kw):
    """The scripted pair's TextRunConfig on the rollouts' retrieval variant."""
    return S.config(retrieval_config=VAR, **kw)


def task_of(f):
    r = G.plugin_row(f, variant=VAR)
    return A.Task(id=r["id"], category=r["category"], instruction=r["instruction"],
                  inputs=tuple(Path(p) for p in r["inputs"]), answer_files=tuple(Path(p) for p in r["answer_files"]),
                  grading=dict(r["grading"]), split=r["split"])


def grade(task, snap):
    try:
        v = GR.grade(task, snap)
    except Exception as e:
        return {"raised": type(e).__name__, "message": str(e)[:400]}
    return {"score": v, "type": type(v).__name__}


def shape(msgs):
    """What a conversation says: roles, text, tool calls, tool results (not timestamps, turn_idx, cost, raw_data)."""
    out = []
    for m in msgs:
        if isinstance(m, ToolMessage):
            out.append(["tool", m.id, m.requestor, m.content, m.error])
        else:
            out.append([m.role, m.content, [tc.model_dump(mode="json") for tc in (m.tool_calls or [])]])
    return out


def strip(d):
    for m in d.get("messages") or []:
        m.pop("raw_data", None)
    return d


def gold_hashes(t):
    """The gold environment as tau2's EnvironmentEvaluator builds it: a fresh environment in the task's initial state,
    then every expected action (an action that raises is skipped, as there)."""
    env = G.fresh_env(t, variant=VAR)
    for a in t.evaluation_criteria.actions or []:
        try:
            env.make_tool_call(tool_name=a.name, requestor=a.requestor, **a.arguments)
        except Exception:
            pass
    return [env.get_db_hash(), env.get_user_db_hash()]


def converse(t, seed, sim_id, end=None):
    orch = build_text_orchestrator(cfg(seed=seed, timeout=float(wall)), t, seed=seed, simulation_id=sim_id)
    if end is not None:
        orch.agent.end_on_call(*end)
    sim, detail = B.run_conversation(orch)
    return orch, sim, detail


def files(p):
    return {n: json.loads((p / n).read_text(encoding="utf-8")) for n in B.FILES}


def final_of(fdb):
    return [fdb["agent"]["hash"], fdb["user"]["hash"]]


def rehash(orch, fdb):
    DB = type(orch.environment.tools.db)
    return [get_dict_hash(DB.model_validate(fdb[s]["db"]).model_dump()) for s in ("agent", "user")]


out = {"tasks": {}, "misc": {}}
first = None
for f, seed in json.loads(Path(chunk_path).read_text()):
    t0 = time.time()
    t = G.load_task(f)
    task = task_of(f)
    d = Path(work) / t.id
    meta = {"task_file": f, "retrieval_variant": VAR, "name": "fixtures_tau2_scripted", "sample": 0}
    orch, sim, detail = converse(t, seed, f"scripted-{t.id}")
    gold = G.golden_messages(t, G.fresh_env(t, variant=VAR))
    try:
        Orchestrator.validate_message_history(sim.messages)
        valid = True
    except Exception as e:
        valid = f"{type(e).__name__}: {e}"
    p = B.write_bundle(orch, sim, d / "golden", meta)
    fl = files(p)
    back = SimulationRun.model_validate_json((p / "simulation.json").read_text(encoding="utf-8"))
    rec = {"tau2_id": t.id, "termination": sim.termination_reason.value, "end_detail": detail,
           "history_valid": valid, "same_as_golden": shape(sim.messages) == shape(gold),
           "n_messages": len(sim.messages), "n_golden": len(gold),
           "n_tool_errors": sum(1 for m in sim.messages if isinstance(m, ToolMessage) and m.error),
           "agent_calls": orch.agent.calls, "user_calls": orch.user.calls,
           "seeds_seen": [orch.agent.seed, orch.user.seed, sim.seed],
           "raw_data_in_memory": sum(1 for m in sim.messages if getattr(m, "raw_data", None)),
           "raw_data_keys_in_file": sum(1 for m in fl["simulation.json"]["messages"] if "raw_data" in m),
           "file_is_the_run": fl["simulation.json"] == strip(sim.model_dump(mode="json")),
           "round_trip_equal": strip(back.model_dump(mode="json")) == strip(sim.model_dump(mode="json")),
           "reward_info": fl["simulation.json"].get("reward_info"),
           "files": sorted(x.name for x in p.iterdir()),
           "bundle_bytes_equal": {n: (p / n).read_bytes() == b for n, b in B.bundle(orch, sim, meta).items()},
           "run": fl["run.json"], "final_hashes": final_of(fl["final_db.json"]),
           "live_hashes": [orch.environment.get_db_hash(), orch.environment.get_user_db_hash()],
           "gold_hashes": gold_hashes(t), "dump_rehash": rehash(orch, fl["final_db.json"]),
           "same_db_object": orch.environment.tools.db is orch.environment.user_tools.db,
           "tools": sorted(x.name for x in orch.environment.get_tools())}
    rec["golden"] = grade(task, d / "golden")
    rec["golden_again"] = grade(task, d / "golden")
    try:
        B.write_bundle(orch, sim, d / "golden", meta)
        rec["rewrite"] = "written"
    except FileExistsError:
        rec["rewrite"] = "FileExistsError"
    n = orch.agent.calls                                   # the agent's last turn: the closing line
    for kind, end in (("end_last", (n, B.HarnessEnd("claude ended its turn with no text"))),
                      ("end_first", (1, B.HarnessEnd("claude exited", reason="infrastructure_error")))):
        o2, s2, det2 = converse(t, seed, f"scripted-{t.id}-{kind}", end)
        p2 = B.write_bundle(o2, s2, d / kind, {**meta, "end_detail": det2})
        f2 = files(p2)
        rec[kind] = {"termination": s2.termination_reason.value, "detail": det2, "run_end": f2["run.json"]["end"],
                     "steps": f2["run.json"]["steps"], "n_messages": len(s2.messages),
                     "prefix_of_golden": shape(s2.messages) == shape(sim.messages)[:len(s2.messages)],
                     "final_hashes": final_of(f2["final_db.json"]), "grade": grade(task, d / kind)}
    o3 = build_text_orchestrator(cfg(seed=seed, timeout=float(wall)), t, seed=seed)
    o3.agent.end_on_call(1, RuntimeError("not a harness end"))
    try:
        B.run_conversation(o3)
        rec["other_exception"] = "returned"
    except Exception as e:
        rec["other_exception"] = f"{type(e).__name__}: {e}"
    rec["seconds"] = round(time.time() - t0, 2)
    out["tasks"][task.id] = rec
    TR._cleanup_sandboxes()     # each environment's shell sandbox: a 698-file KB copy in the temp dir; no shell runs later
    if first is None:
        first = (t, seed, orch, sim, meta, d)

# the refusals, once per child, on its first task
t, seed, orch, sim, meta, d = first
misc = out["misc"]
for label, fn in (("unknown_end_reason", lambda: B.HarnessEnd("x", reason="user_stop")),
                  ("customer_without_agent", lambda: build_user(S.USER, orch.environment, t)),
                  ("sim_of_another_orchestrator", lambda: B.write_bundle(
                      build_text_orchestrator(cfg(seed=seed), t, seed=seed, simulation_id="other"), sim,
                      d / "other", meta))):
    try:
        fn()
        misc[label] = "accepted"
    except Exception as e:
        misc[label] = f"{type(e).__name__}: {e}"
target = d / "planted_target"                          # a tau2/ planted as a link to a directory elsewhere
target.mkdir()
(d / "planted").mkdir()
(d / "planted" / "tau2").symlink_to(target, target_is_directory=True)
try:
    B.write_bundle(orch, sim, d / "planted", meta)
    misc["planted_link"] = "written"
except Exception as e:
    misc["planted_link"] = f"{type(e).__name__}: {e}"
misc["planted_target_files"] = sorted(x.name for x in target.iterdir())
misc["default_end_reason"] = B.HarnessEnd("x").reason
misc["registered"] = [S.AGENT in S.registry.get_agents(), S.USER in S.registry.get_users()]
misc["other_dir_written"] = (d / "other").exists()
misc["retrieval"] = RETRIEVAL
Path(out_path).write_text(json.dumps(out, indent=1))
'''


def _need_tau3():
    if not G.TAU3_PY.is_file():
        pytest.skip(f"the tau3-bench venv interpreter {G.TAU3_PY} is absent")
    if not G.task_files(DATA_ROOT):
        pytest.skip(f"no banking task files under {DATA_ROOT}")
    if not TR.cache_dir(DATA_ROOT).is_dir():
        pytest.skip(f"{TR.cache_dir(DATA_ROOT)} is absent (scripts/tau3_build_embeddings_cache.py)")
    missing = [b for b in ("srt", "rg") if shutil.which(b, path=TOOL_PATH) is None]
    if missing:
        pytest.skip(f"alltools' shell sandbox needs {missing} on {TOOL_PATH}")


@pytest.fixture(scope="module")
def matrix(tmp_path_factory):
    _need_tau3()
    files = G.graded_task_files(DATA_ROOT)
    base = tmp_path_factory.mktemp("tau2_bundle")
    home = base / "home"
    home.mkdir()
    rows = [[str(f), seed_of(f"{G.DOMAIN}/{f.stem}")] for f in files]
    child_py = base / "child.py"
    child_py.write_text(CHILD)
    procs = []
    for k in range(N_CHILDREN):
        chunk = base / f"chunk_{k}.json"
        chunk.write_text(json.dumps(rows[k::N_CHILDREN]))
        log = base / f"child_{k}.log"
        fh = open(log, "wb")
        procs.append((subprocess.Popen(
            [str(G.TAU3_PY), "-B", str(child_py), str(TESTS), str(BUNDLE_DIR), str(GRADER), str(PLUGIN_API),
             str(chunk), str(base / f"out_{k}.json"), str(base / "bundles"), str(WALL_S)],
            cwd=str(base), stdout=fh, stderr=subprocess.STDOUT,
            env=G.child_env(DATA_ROOT, HOME=home, **{G.DATA_ROOT_ENV: DATA_ROOT, "PATH": TOOL_PATH,
                                                     "HTTPS_PROXY": DEAD_PROXY, "HTTP_PROXY": DEAD_PROXY,
                                                     "ALL_PROXY": DEAD_PROXY})), fh, log))
    try:
        for p, fh, log in procs:
            rc = p.wait(timeout=900)
            fh.close()
            if rc != 0:
                raise AssertionError(f"tau3 child failed (rc {rc}):\n"
                                     f"{log.read_bytes()[-4000:].decode('utf-8', 'replace')}")
    finally:
        for p, fh, _log in procs:
            if p.poll() is None:
                p.kill()
            fh.close()
    res, misc = {}, []
    for k in range(N_CHILDREN):
        o = json.loads((base / f"out_{k}.json").read_text())
        res.update(o["tasks"])
        misc.append(o["misc"])
    return {"base": base, "files": files, "res": res, "misc": misc}


def _id(f) -> str:
    return f"{G.DOMAIN}/{Path(f).stem}"


def _top_imports(path: Path) -> set:
    out = set()
    for n in ast.parse(path.read_text()).body:
        if isinstance(n, ast.Import):
            out |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            out.add(n.module.split(".")[0])
    return out


# ─────────────────────────────────────────────────────────────── the files themselves (no tau2 needed)
def test_the_writer_imports_no_tau2_at_import_time_and_the_fixture_is_a_tau2_module():
    assert _top_imports(BUNDLE_PY) <= {"__future__", "hashlib", "json", "pathlib"}, _top_imports(BUNDLE_PY)
    defined = {n.name for n in ast.parse(BUNDLE_PY.read_text()).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert {"HarnessEnd", "run_conversation", "bundle", "write_bundle"} <= defined, defined
    assert {"tau2", "tau3_golden"} <= _top_imports(FIXTURE), _top_imports(FIXTURE)
    assert "tau2" not in _top_imports(Path(__file__))
    assert seed_of("banking_knowledge/task_001") == seed_of("banking_knowledge/task_001") < 2 ** 31
    assert seed_of("banking_knowledge/task_001") != seed_of("banking_knowledge/task_001", 1)


# ─────────────────────────────────────────────────────────────── every banking task but the judged one
def test_the_scripted_pair_runs_every_banking_task_but_the_judged_one(matrix):
    assert len(G.task_files(DATA_ROOT)) == 97 and len(matrix["files"]) == 96
    assert sorted(matrix["res"]) == sorted(_id(f) for f in matrix["files"])


def test_tau2s_orchestrator_plays_the_golden_conversation(matrix):
    for tid, r in matrix["res"].items():
        seed = seed_of(tid)
        assert r["tau2_id"] == tid.split("/", 1)[1], (tid, r["tau2_id"])
        assert (r["termination"], r["end_detail"]) == ("user_stop", None), (tid, r["termination"], r["end_detail"])
        assert r["history_valid"] is True and r["same_as_golden"], (tid, r["history_valid"], r["same_as_golden"])
        assert r["n_messages"] == r["n_golden"] and r["agent_calls"] >= 1 and r["user_calls"] >= 2, (tid, r)
        assert r["seeds_seen"] == [seed, seed, seed], (tid, r["seeds_seen"])


def test_the_bundle_grades_1_on_every_task_twice(matrix):
    bad = {tid: (r["golden"], r["golden_again"]) for tid, r in matrix["res"].items()
           if not r["golden"] == r["golden_again"] == {"score": 1.0, "type": "float"}}
    assert not bad, bad


def test_final_db_holds_the_gold_environments_databases(matrix):
    for tid, r in matrix["res"].items():
        assert r["final_hashes"] == r["live_hashes"] == r["gold_hashes"], (tid, r["final_hashes"], r["gold_hashes"])
        assert r["dump_rehash"] == r["final_hashes"], (tid, r["dump_rehash"])
        assert r["same_db_object"] is True, tid              # banking: one database behind both toolkits


def test_simulation_json_has_no_raw_data_and_is_otherwise_the_run(matrix):
    for tid, r in matrix["res"].items():
        assert r["raw_data_in_memory"] > 0 and r["raw_data_keys_in_file"] == 0, (tid, r["raw_data_in_memory"])
        assert r["file_is_the_run"] and r["round_trip_equal"] and r["reward_info"] is None, tid
        assert r["files"] == FILES and all(r["bundle_bytes_equal"].values()), (tid, r["files"])


def test_run_json_records_how_the_conversation_ran(matrix):
    by_id = {_id(f): f for f in matrix["files"]}
    for tid, r in matrix["res"].items():
        f = by_id[tid]
        run = r["run"]
        assert run["tau2"] == {"version": "1.0.1", "commit": "363133a"}, (tid, run["tau2"])
        assert Path(run["data_dir"]).resolve() == DATA_ROOT.resolve(), (tid, run["data_dir"])
        assert (run["domain"], run["retrieval_variant"], run["retrieval_kwargs"]) == (G.DOMAIN, "alltools", {}), tid
        assert (run["task_id"], run["task_file"]) == (r["tau2_id"], str(f)), (tid, run["task_id"], run["task_file"])
        assert run["task_file_sha256"] == G.sha256_file(f), tid
        assert run["simulation_id"] == f"scripted-{r['tau2_id']}" and run["seed"] == seed_of(tid), tid
        assert (run["max_steps"], run["max_errors"], run["wall_s"]) == (200, 10, WALL_S), tid
        assert isinstance(run["duration_s"], float) and 0 <= run["duration_s"] < WALL_S, (tid, run["duration_s"])
        assert (run["steps"], run["tool_errors"]) == (r["n_messages"] - 1, r["n_tool_errors"]), (tid, run["steps"])
        assert run["end"] == {"reason": "user_stop", "detail": None}, (tid, run["end"])
        assert run["harness"] == {"name": "fixtures_tau2_scripted", "sample": 0}, (tid, run["harness"])


def test_a_bundle_is_never_written_over_an_existing_one(matrix):
    assert {r["rewrite"] for r in matrix["res"].values()} == {"FileExistsError"}


def test_a_harness_end_on_the_closing_turn_is_an_agent_error_that_grades_0(matrix):
    """Every expected action is played and the databases are the gold ones, yet the conversation did not end with a
    stop: tau2 scores it 0."""
    for tid, r in matrix["res"].items():
        e = r["end_last"]
        assert (e["termination"], e["detail"]) == ("agent_error", "claude ended its turn with no text"), (tid, e)
        assert e["run_end"] == {"reason": "agent_error", "detail": "claude ended its turn with no text"}, (tid, e)
        assert e["n_messages"] == r["n_messages"] - 2 and e["prefix_of_golden"], (tid, e)   # no closing line, no STOP
        assert e["steps"] == e["n_messages"] - 1, (tid, e)
        assert e["final_hashes"] == r["gold_hashes"], (tid, e)
        assert e["grade"] == {"score": 0.0, "type": "float"}, (tid, e["grade"])


def test_a_harness_end_on_the_first_turn_keeps_its_reason_and_grades_0(matrix):
    for tid, r in matrix["res"].items():
        e = r["end_first"]
        assert (e["termination"], e["detail"]) == ("infrastructure_error", "claude exited"), (tid, e)
        assert e["run_end"] == {"reason": "infrastructure_error", "detail": "claude exited"}, (tid, e)
        assert e["n_messages"] == 2 and e["prefix_of_golden"], (tid, e)                  # the greeting, the opening
        assert e["grade"] == {"score": 0.0, "type": "float"}, (tid, e["grade"])


def test_any_other_exception_in_the_agent_propagates(matrix):
    assert {r["other_exception"] for r in matrix["res"].values()} == {"RuntimeError: not a harness end"}


def test_the_helpers_refuse_what_they_cannot_record(matrix):
    for m in matrix["misc"]:
        assert m["unknown_end_reason"].startswith("ValueError: a harness end is one of"), m
        assert m["customer_without_agent"].startswith("RuntimeError: the scripted customer is built after"), m
        assert m["sim_of_another_orchestrator"].startswith("ValueError: the SimulationRun 'scripted-task_"), m
        assert m["other_dir_written"] is False, m
        assert m["planted_link"].startswith("FileExistsError") and m["planted_target_files"] == [], m
        assert m["default_end_reason"] == "agent_error" and m["registered"] == [True, True], m


def test_the_conversations_ran_on_alltools_offline(matrix):
    """The scripted conversations' environments are alltools' (17 tools), built as the bridge builds them: the
    documents' embeddings read-only from the data root's cache, no key."""
    for tid, r in matrix["res"].items():
        names = set(r["tools"])
        assert len(names) == 17 and {"KB_search_bm25", "KB_search_dense", "shell"} <= names, (tid, r["tools"])
        assert not {"KB_search", "grep"} & names, tid
    for m in matrix["misc"]:
        rec = m["retrieval"]
        assert rec["variant"] == VARIANT == "alltools" and rec["embedder"]["key"] is False, rec
        assert rec["cache"] == {"dir": str(TR.cache_dir(DATA_ROOT)), "readonly": True,
                                "files": TR.cache_files(DATA_ROOT)}
        assert rec["shell_deny_read"] == list(TR.deny_read_paths(DATA_ROOT)), rec["shell_deny_read"]
