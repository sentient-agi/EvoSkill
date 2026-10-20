"""The user's tau3 grader, examples/tau3_banking/grader.py (retrieval alltools): grade(task, snapshot) is tau2's own evaluate_simulation on the tau2/simulation.json the rollout harness writes into the solver's
working directory. Every task row, golden reference and environment here uses tau2's "alltools" retrieval.

OFFLINE: every tau2 step runs in a CHILD process under the tau3-bench venv (tests/tau3_golden.TAU3_PY, CPython 3.12,
tau2 1.0.1), with tau3_golden.child_env(): PYTHON_DOTENV_DISABLED=1, LITELLM_LOCAL_MODEL_COST_MAP=True and nothing of
this process's environment, so no .env file is read and no key reaches tau2; plus HTTPS_PROXY / HTTP_PROXY / ALL_PROXY
= a dead local port (any request would fail), PATH = /usr/bin:/bin and the dirs holding srt, rg and node (tau2 builds
the alltools shell sandbox, which checks for srt and rg) and TMPDIR = a test dir (no shell sandbox may be left in it).
No model is called: the references are built from each task's own expected actions (tests/tau3_golden.py), the
document vectors come from the data root's read-only .embeddings_cache, and the only judged banking task (task_102,
reward basis NL_ASSERTION) is left out of every grading test. The golden references are built in a child that
installed scripts/tau3_cc/tau3_retrieval.py with no key, as the bridge builds its environment; the grader installs
its own offline changes over it. Skipped when the venv, the data root's embeddings cache or srt is absent. The data
root is the curated third_party/tau3_data_363133a (scripts/tau3_make_data_root.py + the embeddings precompute),
or $EVOSKILL_TAU3_DATA_ROOT (a copy of it with its .embeddings_cache).

What must hold, for every banking task but task_102 (96): the golden reference scores 1.0 and two grades of it agree;
the greeting followed by ###STOP### scores 0.0; the same conversation ended by agent_error or max_steps scores 0.0; a
missing simulation.json raises; a conversation recorded for another task raises; a changed ToolMessage on a
state-changing call raises (strict replay); the grader refuses an answer file outside the tau2 layout and a tau2
already imported with another data root; no shell sandbox dir is left behind. alltools: a golden conversation with a
KB_search_dense and a shell call added still scores 1.0, also when those two recorded outputs are changed (replay
never re-runs a read); a data root without the cache entry, or with a stale one (other document ids), raises
OfflineError and nothing is written or deleted there; the grader file imports only the standard library. The grader
ALONE (no tau3_retrieval in the process): the sandboxes grade() builds are gone when it returns, one the caller builds
outside grade() is kept, tau2's "openai" embedder is the grader's OfflineEmbedder everywhere and refuses (also through
KB_search_dense), the cache refuses put/clear, and a second copy of the grader module (another name) cleans up its own
sandboxes too, one wrapper per copy. And end to
end: scripts/daytona/plugin_grade_runner.py, run under the tau3 venv with a plug-in dir holding a copy of
plugin_api.py, the grader as benchmark_grader.py and three hand-built tau3 rows, gives the same scores from snapshot
tgz files whose members sit under evo/work/tau2/simulation.json, with no TAU2_DATA_DIR in its environment (the grader
derives it from the task's answer file). It runs twice: the rows' answer files in the test data root, and in a copy
at another path (with the cache), which only a grader that derives the root can score.
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
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
GRADER = ROOT / "examples" / "tau3_banking" / "grader.py"
PLUGIN_API = ROOT / "src" / "orchestrator" / "plugin_api.py"
RUNNER = ROOT / "scripts" / "daytona" / "plugin_grade_runner.py"
TR_DIR = ROOT / "scripts" / "tau3_cc"                    # tau3_retrieval.py: the bridge's alltools changes
CURATED_ROOT = ROOT / "third_party" / "tau3_data_363133a"   # scripts/tau3_make_data_root.py + the embeddings precompute
VARIANT = "alltools"
CACHE_DIRNAME = ".embeddings_cache"
DEAD = "http://127.0.0.1:9"                              # a proxy on a dead port: any request fails
N_CHILDREN = 4
GF = "grader_failed"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


G = _load(TESTS / "tau3_golden.py", "tau3_golden")


def data() -> Path:
    """The data root graded here: $EVOSKILL_TAU3_DATA_ROOT, else the curated root (alltools needs its cache)."""
    return Path(os.environ.get(G.DATA_ROOT_ENV) or CURATED_ROOT)


def cache_pkl(root=None):
    """The root's one document-embeddings entry (None when absent)."""
    pk = sorted((Path(root or data()) / CACHE_DIRNAME).glob("*.pkl"))
    return pk[0] if len(pk) == 1 else None


def srt_path():
    """PATH for a tau2 child: /usr/bin:/bin and the dirs holding srt, rg and node (+ bwrap, socat on Linux); None
    when one is missing."""
    need = ("srt", "rg", "node") + (("bwrap", "socat") if sys.platform.startswith("linux") else ())
    search = os.pathsep.join([os.environ.get("PATH", ""), "/usr/local/bin"])
    dirs = ["/usr/bin", "/bin"]
    for b in need:
        w = shutil.which(b, path=search)
        if w is None:
            return None
        if str(Path(w).parent) not in dirs:
            dirs.append(str(Path(w).parent))
    return ":".join(dirs)


def offline_env(root, **extra) -> dict:
    """tau3_golden.child_env plus the dead proxies and the srt PATH."""
    return G.child_env(root, PATH=srt_path(), HTTPS_PROXY=DEAD, HTTP_PROXY=DEAD, ALL_PROXY=DEAD,
                       https_proxy=DEAD, http_proxy=DEAD, all_proxy=DEAD, **extra)


def sandboxes_in(d: Path) -> list:
    return sorted(p.name for p in Path(d).iterdir() if p.name.startswith("agentic_search_")) if Path(d).is_dir() else []

# One child grades a chunk of tasks: every kind of bundle for each, one grade() call per bundle, the results as JSON.
CHILD = r'''
import importlib.util, json, sys, time
from pathlib import Path

tests_dir, grader_path, api_path, chunk_path, out_path, work, tr_dir, variant = sys.argv[1:9]
sys.path[:0] = [tests_dir, tr_dir]
import os
import tau3_golden as G
import tau3_retrieval as TR

TR.install(Path(os.environ["TAU2_DATA_DIR"]), key=None)    # the rollout's environment: the read-only cache, no key


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A = load(api_path, "plugin_api")
GR = load(grader_path, "benchmark_grader")


def task_of(f):
    r = G.plugin_row(f, variant=variant)
    return A.Task(id=r["id"], category=r["category"], instruction=r["instruction"],
                  inputs=tuple(Path(p) for p in r["inputs"]), answer_files=tuple(Path(p) for p in r["answer_files"]),
                  grading=dict(r["grading"]), split=r["split"])


def grade(task, snap):
    try:
        v = GR.grade(task, snap)
    except Exception as e:
        return {"raised": type(e).__name__, "message": str(e)[:400]}
    return {"score": v, "type": type(v).__name__}


out = {}
for f, other in json.loads(Path(chunk_path).read_text()):
    t0 = time.time()
    t = G.load_task(f)
    task = task_of(f)
    d = Path(work) / t.id
    gold = G.golden_run(t, variant=variant)
    p = G.write_bundle(gold, d / "golden")
    again = G.write_bundle(G.golden_run(t, variant=variant), d / "golden_rebuilt")
    back = type(gold).model_validate_json(p.read_text())
    calls = [(m.role, len(m.tool_calls), m.tool_calls[0].id) for m in gold.messages if getattr(m, "tool_calls", None)]
    rec = {"task_file": f, "tau2_id": t.id, "sim_task_id": gold.task_id,
           "basis": [b.value for b in t.evaluation_criteria.reward_basis],
           "n_actions": len(t.evaluation_criteria.actions or []), "n_calls": len(calls),
           "one_call_per_message": all(n == 1 for _r, n, _i in calls),
           "ids_ok": all(i.startswith("toolu_01" if r == "assistant" else "call_") for r, _n, i in calls),
           "roles": sorted({r for r, _n, _i in calls}), "termination": gold.termination_reason.value,
           "rebuilt_identical": p.read_bytes() == again.read_bytes(),
           "round_trip_identical": back.model_dump(mode="json") == gold.model_dump(mode="json")}
    rec["golden"] = grade(task, d / "golden")
    rec["golden_again"] = grade(task, d / "golden")
    G.write_bundle(G.stop_run(t), d / "stop")
    rec["stop"] = grade(task, d / "stop")
    for reason in ("agent_error", "max_steps"):
        G.write_bundle(G.ended(gold, reason), d / reason)
        rec[reason] = grade(task, d / reason)
    (d / "missing").mkdir()
    rec["missing"] = grade(task, d / "missing")
    G.write_bundle(G.for_task(gold, other), d / "mismatch")
    rec["mismatch"] = grade(task, d / "mismatch")
    tam = G.tampered(gold, t, variant=variant)
    if tam is None:
        rec["tampered"] = None
    else:
        G.write_bundle(tam, d / "tampered")
        rec["tampered"] = grade(task, d / "tampered")
    rec["seconds"] = round(time.time() - t0, 2)
    out[task.id] = rec
Path(out_path).write_text(json.dumps(out, indent=1))
'''

# The grader's own refusals that need a tau2 import: an answer file outside the tau2 layout, and a tau2 already
# imported in the process with another data root.
REFUSALS = r'''
import importlib.util, json, shutil, sys
from pathlib import Path

tests_dir, grader_path, api_path, task_file, snap, other_root, out_path = sys.argv[1:8]
sys.path.insert(0, tests_dir)
import tau3_golden as G


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A = load(api_path, "plugin_api")
GR = load(grader_path, "benchmark_grader")


def task_of(answer):
    r = G.plugin_row(task_file, variant="alltools")
    return A.Task(id=r["id"], category=r["category"], instruction="", inputs=(), answer_files=(Path(answer),),
                  grading=dict(r["grading"]), split="train")


def grade(task):
    try:
        return {"score": GR.grade(task, Path(snap))}
    except Exception as e:
        return {"raised": type(e).__name__, "message": str(e)[:400]}


out = {}
loose = Path(other_root) / "loose" / Path(task_file).name
loose.parent.mkdir(parents=True)
shutil.copy2(task_file, loose)
out["layout"] = grade(task_of(loose))
import tau2.utils.utils as U                            # imported with TAU2_DATA_DIR = other_root (the environment)
out["tau2_data_dir"] = str(U.DATA_DIR)
out["other_root"] = grade(task_of(task_file))
Path(out_path).write_text(json.dumps(out))
'''

# alltools reads: the golden conversation with a KB_search_dense and a shell call added right after the customer's
# opening (as the agent's first two steps), graded; then the same bundle with those two recorded outputs changed.
READS = r'''
import importlib.util, json, os, sys
from pathlib import Path

tests_dir, grader_path, api_path, tr_dir, files_json, out_path, work = sys.argv[1:8]
sys.path[:0] = [tests_dir, tr_dir]
import tau3_golden as G
import tau3_retrieval as TR

TR.install(Path(os.environ["TAU2_DATA_DIR"]), key=None)
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage

QUERY = "How do I dispute a transaction on my credit card?"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A = load(api_path, "plugin_api")
GR = load(grader_path, "benchmark_grader")


def grade(task, snap):
    try:
        return {"score": GR.grade(task, snap)}
    except Exception as e:
        return {"raised": type(e).__name__, "message": str(e)[:400]}


out = {}
for f in json.loads(files_json):
    t = G.load_task(f)
    r = G.plugin_row(f, variant="alltools")
    task = A.Task(id=r["id"], category=r["category"], instruction="", inputs=(), answer_files=(Path(f),),
                  grading=dict(r["grading"]), split="train")
    env = G.fresh_env(t, variant="alltools")
    msgs = G.golden_messages(t, env)
    bm25 = env.get_response(ToolCall(id="toolu_01bm25", name="KB_search_bm25", arguments={"query": QUERY},
                                     requestor="assistant"))
    reads = []
    for i, (name, args) in enumerate((("KB_search_dense", {"query": QUERY}), ("shell", {"command": "ls"}))):
        tc = ToolCall(id=f"toolu_01read{i:018d}", name=name, arguments=args, requestor="assistant")
        # the shell output is the sandbox's own (srt runs here); the dense one needs the network, so its recorded
        # output is BM25's for the same query: replay never re-runs a read nor compares its output
        tm = (env.get_response(tc) if name == "shell" else
              ToolMessage(id=tc.id, role="tool", content=bm25.content, requestor="assistant"))
        reads += [AssistantMessage(role="assistant", content=None, tool_calls=[tc]), tm]
    run = G._run(t.id, msgs[:2] + reads + msgs[2:], "user_stop")
    d = Path(work) / t.id
    G.write_bundle(run, d / "reads")
    changed = run.model_copy(deep=True)
    for m in changed.messages:
        if isinstance(m, ToolMessage) and m.id.startswith("toolu_01read"):
            m.content = "changed after the conversation"
    G.write_bundle(changed, d / "reads_changed")
    rec = {"calls": [tc.name for m in run.messages for tc in (getattr(m, "tool_calls", None) or [])],
           "shell_output": next(m.content for m in run.messages if isinstance(m, ToolMessage)
                                and m.id == f"toolu_01read{1:018d}")}
    rec["reads"] = grade(task, d / "reads")
    rec["reads_changed"] = grade(task, d / "reads_changed")
    out[t.id] = rec
Path(out_path).write_text(json.dumps(out))
'''

# The grader ALONE (no tau3_retrieval in the process, so nothing else removes sandboxes): what it does to tau2 and to
# the sandboxes, looked at while the process still runs. A counter on SandboxManager.__init__ shows sandboxes ARE built
# inside grade() (so "none left" is not vacuous); a second copy of the grader module (another name) grades too.
ALONE = r'''
import importlib.util, json, os, shutil, sys
from pathlib import Path

tests_dir, grader_path, api_path, task_file, snap, out_path = sys.argv[1:7]
sys.path.insert(0, tests_dir)
import tau3_golden as G


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def raised(fn):
    try:
        fn()
    except Exception as e:
        return {"raised": type(e).__name__, "message": str(e)[:400]}
    return {"raised": None}


A = load(api_path, "plugin_api")
GR = load(grader_path, "benchmark_grader")
GR2 = load(grader_path, "benchmark_grader_second_copy")
tmp = Path(os.environ["TMPDIR"])
left = lambda: sorted(p.name for p in tmp.iterdir() if p.name.startswith("agentic_search_"))
from tau2.knowledge import sandbox_manager as SM            # TAU2_DATA_DIR = the root (the environment), as grade() sets
built = []
_init = SM.SandboxManager.__init__


def counted(self, *a, **k):
    _init(self, *a, **k)
    built.append(str(self.sandbox_dir))


SM.SandboxManager.__init__ = counted
r = G.plugin_row(task_file, variant="alltools")
task = A.Task(id=r["id"], category=r["category"], instruction="", inputs=(), answer_files=(Path(task_file),),
              grading=dict(r["grading"]), split="train")
out = {"left_before": left()}
out["first"] = {"score": GR.grade(task, Path(snap))}
out["built_first"] = len(built)
out["left_after_first"] = left()
out["live_after_first"] = len(GR._SANDBOXES["live"])
out["grading_flag_after"] = GR._SANDBOXES["grading"]
from tau2.knowledge import embedders as E
from tau2.knowledge import embeddings_cache as EC
from tau2.knowledge.document_preprocessors import embedding_indexer as EI
from tau2.knowledge.embedders import openai_embedder as OE
from tau2.knowledge.input_preprocessors import embedding_encoder as EE
out["embedder_everywhere"] = sorted({c.__name__ for c in (E.OpenAIEmbedder, OE.OpenAIEmbedder,
                                                         EI.EMBEDDER_REGISTRY["openai"], EE.EMBEDDER_REGISTRY["openai"])})
out["embed"] = raised(lambda: E.OpenAIEmbedder(model="text-embedding-3-large").embed(["a query"]))
cache = EC.get_embeddings_cache()
out["cache_type"] = type(cache).__name__
out["put"] = raised(lambda: cache.put([], "openai", None, [], {"model": "text-embedding-3-large"}))
out["clear"] = raised(lambda: cache.clear())
# an environment the caller builds outside grade(): its sandbox stays the caller's; its dense search is refused
from tau2.data_model.message import ToolCall
from tau2.domains.banking_knowledge.environment import get_environment
env = get_environment(retrieval_variant="alltools")
tm = env.get_response(ToolCall(id="toolu_01dense", name="KB_search_dense", arguments={"query": "card dispute"},
                               requestor="assistant"))
out["dense_outside"] = {"error": tm.error, "content": tm.content[:400]}
outside = built[-1]
out["second"] = {"score": GR2.grade(task, Path(snap))}      # the second copy, the first copy's wrapper still in place
out["built_second"] = len(built)
out["left_after_second"] = left()
out["outside_kept"] = Path(outside).is_dir() and left() == [Path(outside).name]
out["live_after_second"] = [len(GR._SANDBOXES["live"]), len(GR2._SANDBOXES["live"])]
out["third"] = {"score": GR.grade(task, Path(snap))}        # the first copy again: no new wrapper, still cleaned
out["left_after_third"] = left()
from tau2.domains.banking_knowledge import retrieval as R
chain, f = [], R._create_sandbox
while f is not None:
    t = getattr(f, "_grader_tracks", None)
    chain.append("GR" if t is GR._SANDBOXES else "GR2" if t is GR2._SANDBOXES else "tau2")
    f = getattr(f, "_grader_inner", None)
out["wrappers"] = chain
shutil.rmtree(Path(outside), ignore_errors=True)
Path(out_path).write_text(json.dumps(out))
'''

# The grader on a data root whose embeddings cache is absent (mode "missing") or stale (mode "stale": the child writes
# the real entry under its own name with the document ids reversed, then grades): a golden bundle must raise.
CACHE = r'''
import hashlib, importlib.util, json, os, pickle, sys
from pathlib import Path

tests_dir, grader_path, api_path, task_file, snap, real_pkl, mode, out_path = sys.argv[1:9]
sys.path.insert(0, tests_dir)
import tau3_golden as G


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


A = load(api_path, "plugin_api")
GR = load(grader_path, "benchmark_grader")
root = Path(os.environ["TAU2_DATA_DIR"])
cache = root / ".embeddings_cache"
out = {}
if mode == "stale":
    with open(real_pkl, "rb") as fh:
        data = pickle.load(fh)
    # another document SET (one id replaced): a mere re-ordering is the same vectors and is served (in_kb_order)
    data["doc_ids"] = list(data["doc_ids"][:-1]) + ["doc_not_in_this_knowledge_base"]
    cache.mkdir()
    with open(cache / Path(real_pkl).name, "wb") as fh:
        pickle.dump(data, fh)
    out["stale_sha256"] = hashlib.sha256((cache / Path(real_pkl).name).read_bytes()).hexdigest()
r = G.plugin_row(task_file, variant="alltools")
task = A.Task(id=r["id"], category=r["category"], instruction="", inputs=(), answer_files=(Path(task_file),),
              grading=dict(r["grading"]), split="train")
try:
    out["grade"] = {"score": GR.grade(task, Path(snap))}
except Exception as e:
    out["grade"] = {"raised": type(e).__name__, "message": str(e)[:600]}
Path(out_path).write_text(json.dumps(out))
'''


def _need_tau3():
    if not G.TAU3_PY.is_file():
        pytest.skip(f"the tau3-bench venv interpreter {G.TAU3_PY} is absent")
    if not G.task_files(data()):
        pytest.skip(f"no banking task files under {data()}")
    if cache_pkl() is None:
        pytest.skip(f"{data() / CACHE_DIRNAME} holds no single embeddings entry (scripts/tau3_build_embeddings_cache.py)")
    if srt_path() is None:
        pytest.skip("srt, rg or node is absent (tau2 builds the alltools shell sandbox with srt)")


def _child(args: list, *, cwd: Path, log: Path, root=None, timeout: int = 900, **env) -> None:
    with open(log, "wb") as fh:
        p = subprocess.run([str(G.TAU3_PY), "-B", *map(str, args)], cwd=str(cwd), stdout=fh,
                           stderr=subprocess.STDOUT, env=offline_env(root, **env), timeout=timeout)
    if p.returncode != 0:
        tail = log.read_bytes()[-4000:].decode("utf-8", "replace")
        raise AssertionError(f"tau3 child failed (rc {p.returncode}):\n{tail}")


@pytest.fixture(scope="module")
def matrix(tmp_path_factory):
    _need_tau3()
    files = G.graded_task_files(data())
    base = tmp_path_factory.mktemp("tau3_grader")
    home = base / "home"
    home.mkdir()
    tmp = base / "tmp"                                   # TMPDIR of every child: tau2's shell sandboxes go here
    tmp.mkdir()
    pairs = [[str(f), files[(i + 1) % len(files)].stem] for i, f in enumerate(files)]
    child_py = base / "child.py"
    child_py.write_text(CHILD)
    procs = []
    for k in range(N_CHILDREN):
        chunk = base / f"chunk_{k}.json"
        chunk.write_text(json.dumps(pairs[k::N_CHILDREN]))
        log = base / f"child_{k}.log"
        fh = open(log, "wb")
        procs.append((subprocess.Popen(
            [str(G.TAU3_PY), "-B", str(child_py), str(TESTS), str(GRADER), str(PLUGIN_API), str(chunk),
             str(base / f"out_{k}.json"), str(base / "bundles"), str(TR_DIR), VARIANT],
            cwd=str(base), stdout=fh, stderr=subprocess.STDOUT, env=offline_env(data(), HOME=home, TMPDIR=tmp)),
            fh, log))
    res = {}
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
    for k in range(N_CHILDREN):
        res.update(json.loads((base / f"out_{k}.json").read_text()))
    return {"base": base, "home": home, "tmp": tmp, "files": files, "res": res}


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
def test_the_grader_file_is_the_primitive_with_a_standard_library_top_level():
    from src.orchestrator.plugin_runner import static_check_grader
    src = GRADER.read_text()
    assert static_check_grader(src)["ok"], static_check_grader(src)
    assert _top_imports(GRADER) == {"__future__", "atexit", "os", "pathlib", "pickle"}, _top_imports(GRADER)
    assert _top_imports(GRADER) <= set(sys.stdlib_module_names) | {"__future__"}
    assert "tau2" in src and "PYTHON_DOTENV_DISABLED" in src and "LITELLM_LOCAL_MODEL_COST_MAP" in src
    assert "RETRIEVAL alltools" in src and "srt" in src and CACHE_DIRNAME in src
    assert "tau3_retrieval" not in src and "scripts" not in src     # self-contained: a run's copy has no repo beside it
    tr = _load(TR_DIR / "tau3_retrieval.py", "tau3_retrieval_for_grader_test")
    assert tr.CACHE_DIRNAME == CACHE_DIRNAME and tr.VARIANT == VARIANT == G.plugin_row("x/t.json", variant=VARIANT)[
        "grading"]["retrieval_variant"]


def test_importing_the_grader_under_the_tau3_venv_loads_nothing_but_the_standard_library(tmp_path):
    if not G.TAU3_PY.is_file():
        pytest.skip(f"the tau3-bench venv interpreter {G.TAU3_PY} is absent")
    code = ("import importlib.util, json, sys\n"
            "before = set(sys.modules)\n"
            f"spec = importlib.util.spec_from_file_location('benchmark_grader', {str(GRADER)!r})\n"
            "m = importlib.util.module_from_spec(spec); sys.modules['benchmark_grader'] = m; spec.loader.exec_module(m)\n"
            "new = sorted(set(sys.modules) - before)\n"
            "print(json.dumps([n for n in new if n.split('.')[0] not in sys.stdlib_module_names]))\n")
    r = subprocess.run([str(G.TAU3_PY), "-I", "-B", "-c", code], cwd=str(tmp_path), capture_output=True, text=True,
                       env=G.child_env(None, HOME=tmp_path), timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert json.loads(r.stdout.strip().splitlines()[-1]) == ["benchmark_grader"]


def test_the_golden_helper_imports_under_the_repo_venv_without_tau2():
    assert _top_imports(TESTS / "tau3_golden.py") <= {
        "__future__", "argparse", "hashlib", "json", "os", "shutil", "sys", "datetime", "pathlib"}
    row = G.plugin_row(G.DEFAULT_DATA_ROOT / "tau2" / "domains" / G.DOMAIN / "tasks" / "task_007.json",
                       variant=VARIANT)
    assert row == {"id": "banking_knowledge/task_007", "category": "banking_knowledge", "instruction": "",
                   "inputs": [], "answer_files": [str(G.DEFAULT_DATA_ROOT / "tau2/domains/banking_knowledge/tasks/"
                                                      "task_007.json")],
                   "grading": {"domain": "banking_knowledge", "retrieval_variant": "alltools"}, "split": "train"}
    assert G.root_of(row["answer_files"][0]) == G.DEFAULT_DATA_ROOT
    ids = {G.call_id("task_007", i, r) for i in range(40) for r in ("assistant", "user")}
    assert len(ids) == 80
    assert all(len(i) == 30 and i.startswith("toolu_01") or len(i) == 29 and i.startswith("call_") for i in ids)
    env = G.child_env("/x")
    assert env["PYTHON_DOTENV_DISABLED"] == "1" and env["LITELLM_LOCAL_MODEL_COST_MAP"] == "True"
    assert env["TAU2_DATA_DIR"] == "/x" and "TAU2_DATA_DIR" not in G.child_env()
    assert sorted(env) == ["HOME", "LC_ALL", "LITELLM_LOCAL_MODEL_COST_MAP", "PATH", "PYTHONDONTWRITEBYTECODE",
                           "PYTHON_DOTENV_DISABLED", "TAU2_DATA_DIR"]


# ─────────────────────────────────────────────────────────────── every banking task but the judged one
def test_the_graded_set_is_every_banking_task_but_the_judged_one(matrix):
    allf = G.task_files(data())
    assert len(allf) == 97 and len(matrix["files"]) == 96
    assert [p.stem for p in allf if G.needs_judge(p)] == ["task_102"]
    assert sorted(matrix["res"]) == sorted(_id(f) for f in matrix["files"])


def test_the_golden_reference_is_a_well_formed_harness_recording(matrix):
    for tid, r in matrix["res"].items():
        assert r["sim_task_id"] == r["tau2_id"] == tid.split("/", 1)[1], (tid, r)
        assert r["n_calls"] == r["n_actions"] > 0 and r["one_call_per_message"] and r["ids_ok"], (tid, r)
        assert r["termination"] == "user_stop" and "NL_ASSERTION" not in r["basis"], (tid, r)
        assert r["rebuilt_identical"] and r["round_trip_identical"], (tid, r)
    roles = {tuple(r["roles"]) for r in matrix["res"].values()}
    assert ("assistant", "user") in roles           # customer tool calls are in the references too


def test_the_golden_reference_scores_1_on_every_task(matrix):
    bad = {tid: r["golden"] for tid, r in matrix["res"].items() if r["golden"] != {"score": 1.0, "type": "float"}}
    assert not bad, bad


def test_two_grades_of_one_bundle_agree(matrix):
    bad = {tid: (r["golden"], r["golden_again"]) for tid, r in matrix["res"].items()
           if r["golden"] != r["golden_again"]}
    assert not bad, bad


def test_the_greeting_then_stop_scores_0(matrix):
    bad = {tid: r["stop"] for tid, r in matrix["res"].items() if r["stop"] != {"score": 0.0, "type": "float"}}
    assert not bad, bad


@pytest.mark.parametrize("reason", ["agent_error", "max_steps"])
def test_a_conversation_that_did_not_end_with_a_stop_scores_0(matrix, reason):
    bad = {tid: r[reason] for tid, r in matrix["res"].items() if r[reason] != {"score": 0.0, "type": "float"}}
    assert not bad, bad


def test_a_missing_simulation_json_raises(matrix):
    for tid, r in matrix["res"].items():
        assert r["missing"]["raised"] == "FileNotFoundError", (tid, r["missing"])
        assert "tau2/simulation.json is missing" in r["missing"]["message"], (tid, r["missing"])


def test_a_conversation_for_another_task_raises(matrix):
    for tid, r in matrix["res"].items():
        assert r["mismatch"]["raised"] == "ValueError", (tid, r["mismatch"])
        assert r["mismatch"]["message"].startswith("the conversation is for 'task_"), (tid, r["mismatch"])


def test_a_changed_tool_output_on_a_state_changing_call_raises(matrix):
    none = sorted(tid for tid, r in matrix["res"].items() if r["tampered"] is None)
    # these four expect only transfer_to_human_agents, which changes no state, so replay never re-runs a call
    assert none == [f"banking_knowledge/task_{n}" for n in ("004", "008", "012", "014")], none
    for tid, r in matrix["res"].items():
        if r["tampered"] is not None:
            assert r["tampered"]["raised"] == "ValueError", (tid, r["tampered"])
            assert r["tampered"]["message"].startswith("Tool call:"), (tid, r["tampered"])


def test_the_grader_refuses_a_foreign_layout_and_a_tau2_on_another_root(matrix, tmp_path):
    f = matrix["files"][0]
    snap = matrix["base"] / "bundles" / f.stem / "golden"
    other = tmp_path / "other_root"
    other.mkdir()
    child = tmp_path / "refusals.py"
    child.write_text(REFUSALS)
    out = tmp_path / "out.json"
    _child([child, TESTS, GRADER, PLUGIN_API, f, snap, other, out], cwd=tmp_path, log=tmp_path / "child.log",
           root=other, HOME=matrix["home"])
    r = json.loads(out.read_text())
    assert r["layout"]["raised"] == "ValueError" and "is not <root>/tau2/domains/banking_knowledge/tasks/<file>" \
        in r["layout"]["message"], r
    assert r["tau2_data_dir"] == str(other)
    assert r["other_root"]["raised"] == "RuntimeError" and "tau2 was imported with data root" in \
        r["other_root"]["message"], r


def test_no_shell_sandbox_is_left_behind(matrix):
    """Every child built alltools environments (each one a KB shell sandbox in TMPDIR); none is left at exit. (Those
    children also installed tau3_retrieval, whose own exit hook removes sandboxes: the grader's own removal is shown
    by the next test, without it.)"""
    assert sandboxes_in(matrix["tmp"]) == []


def test_the_grader_alone_removes_its_sandboxes_when_grade_returns_and_refuses_every_embedding(matrix, tmp_path):
    f = _picks(matrix)[0]
    snap = matrix["base"] / "bundles" / f.stem / "golden"
    child = tmp_path / "alone.py"
    child.write_text(ALONE)
    out = tmp_path / "out.json"
    (tmp_path / "tmp").mkdir()
    _child([child, TESTS, GRADER, PLUGIN_API, f, snap, out], cwd=tmp_path, log=tmp_path / "child.log", root=data(),
           HOME=matrix["home"], TMPDIR=tmp_path / "tmp")
    r = json.loads(out.read_text())
    assert r["left_before"] == []
    assert r["first"] == r["second"] == r["third"] == {"score": 1.0}
    assert r["built_first"] >= 1, r                         # grade() built sandboxes ...
    assert r["left_after_first"] == [] and r["live_after_first"] == 0 and r["grading_flag_after"] is False  # ... gone
    assert r["embedder_everywhere"] == ["OfflineEmbedder"]
    assert r["embed"]["raised"] == "OfflineError" and "offline" in r["embed"]["message"]
    assert r["cache_type"] == "ReadOnlyEmbeddingsCache"
    assert r["put"]["raised"] == r["clear"]["raised"] == "OfflineError"
    assert r["dense_outside"]["error"] is True and "offline" in r["dense_outside"]["content"], r["dense_outside"]
    assert r["built_second"] > r["built_first"] + 1          # the outside environment's, then the second copy's
    assert r["outside_kept"] is True, r                      # the caller's own sandbox is never touched
    assert r["left_after_second"] == r["left_after_third"] and len(r["left_after_second"]) == 1
    assert r["live_after_second"] == [0, 0]
    assert r["wrappers"] == ["GR2", "GR", "tau2"], r["wrappers"]   # one wrapper per copy, none added on regrading
    assert sandboxes_in(tmp_path / "tmp") == []


def test_a_conversation_with_dense_and_shell_reads_still_grades_offline(matrix, tmp_path):
    """The agent's KB_search_dense and shell calls are reads: strict replay skips them, so the grader (offline, no
    key, its embedder refusing every request) scores the conversation as the golden one, also with their recorded
    outputs changed."""
    picks = _picks(matrix)
    child = tmp_path / "reads.py"
    child.write_text(READS)
    out = tmp_path / "out.json"
    (tmp_path / "tmp").mkdir()
    _child([child, TESTS, GRADER, PLUGIN_API, TR_DIR, json.dumps([str(f) for f in picks]), out, tmp_path / "work"],
           cwd=tmp_path, log=tmp_path / "child.log", root=data(), HOME=matrix["home"], TMPDIR=tmp_path / "tmp")
    res = json.loads(out.read_text())
    assert sorted(res) == sorted(f.stem for f in picks)
    for tid, r in res.items():
        assert r["calls"][:2] == ["KB_search_dense", "shell"], (tid, r["calls"])
        assert not {"KB_search_dense", "shell"} & set(r["calls"][2:]), (tid, r["calls"])   # the golden's: no reads
        assert "INDEX.md" in r["shell_output"] and ".md" in r["shell_output"].replace("INDEX.md", ""), r
        assert r["reads"] == {"score": 1.0} == r["reads_changed"], (tid, r)
        assert matrix["res"][_id(f"{tid}.json")]["golden"]["score"] == 1.0
    assert sandboxes_in(tmp_path / "tmp") == []


def _cache_case(matrix, tmp_path, mode: str) -> tuple:
    f = _picks(matrix)[0]
    root = _data_copy([f], tmp_path / "root", cache=False)
    child = tmp_path / "cache.py"
    child.write_text(CACHE)
    out = tmp_path / "out.json"
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    snap = matrix["base"] / "bundles" / f.stem / "golden"
    _child([child, TESTS, GRADER, PLUGIN_API, root / "tau2" / "domains" / G.DOMAIN / "tasks" / f.name, snap,
            cache_pkl(), mode, out], cwd=cwd, log=tmp_path / "child.log", root=root, HOME=matrix["home"],
           TMPDIR=tmp_path)
    assert matrix["res"][_id(f)]["golden"] == {"score": 1.0, "type": "float"}   # the same bundle, on the real root
    assert list(cwd.iterdir()) == [], "the grader wrote into its working directory (tau2's default cache?)"
    return root, json.loads(out.read_text())


def test_a_data_root_without_the_embeddings_cache_raises_and_gets_none(matrix, tmp_path):
    root, r = _cache_case(matrix, tmp_path, "missing")
    assert r["grade"]["raised"] == "OfflineError" and "is absent in" in r["grade"]["message"], r
    assert not (root / CACHE_DIRNAME).exists(), "the grader created the cache dir"
    assert sandboxes_in(tmp_path) == []


def test_a_stale_embeddings_entry_raises_and_is_neither_deleted_nor_rewritten(matrix, tmp_path):
    root, r = _cache_case(matrix, tmp_path, "stale")
    assert r["grade"]["raised"] == "OfflineError" and "document ids differ" in r["grade"]["message"], r
    left = sorted((root / CACHE_DIRNAME).iterdir())
    assert [p.name for p in left] == [cache_pkl().name], left            # tau2's own cache would unlink it
    assert _sha(left[0]) == r["stale_sha256"]
    assert sandboxes_in(tmp_path) == []


# ─────────────────────────────────────────────────────────────── once end to end, through plugin_grade_runner
def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _snapshot_tgz(bundle_dir, dest: Path) -> Path:
    """An out.tgz as evoctl pack writes one: evo/work (with the harness's tau2/ when the bundle has one) and evo/out."""
    src = dest.parent / (dest.name + ".src")
    work = src / "evo" / "work"
    work.mkdir(parents=True)
    (work / "notes.md").write_text("the solver's own file\n")
    if bundle_dir is not None:
        shutil.copytree(bundle_dir / "tau2", work / "tau2")
    (src / "evo" / "out").mkdir(parents=True)
    (src / "evo" / "out" / "stream.jsonl").write_text("{}\n")
    with tarfile.open(dest, "w:gz", format=tarfile.PAX_FORMAT, dereference=False) as tf:
        for part in ("evo/work", "evo/out"):
            tf.add(src / part, arcname=part, recursive=True)
    return dest


def _picks(matrix) -> list:
    """Three tasks of different shapes: a DB task with customer tool calls, a DB task with agent calls only, and an
    ACTION task."""
    out = []
    for want in ((("DB",), ("assistant", "user")), (("DB",), ("assistant",)), (("ACTION",), None)):
        f = next((f for f in matrix["files"] if tuple(matrix["res"][_id(f)]["basis"]) == want[0]
                  and (want[1] is None or tuple(matrix["res"][_id(f)]["roles"]) == want[1])), None)
        assert f is not None, want
        out.append(f)
    return out


def _data_copy(files, dest: Path, *, cache: bool = True) -> Path:
    """A data root at another path: the banking db, documents and prompts, the picked task files and (cache) the
    root's .embeddings_cache."""
    src = G.root_of(files[0]) / "tau2" / "domains" / G.DOMAIN
    dom = dest / "tau2" / "domains" / G.DOMAIN
    dom.mkdir(parents=True)
    shutil.copy2(src / "db.json", dom / "db.json")
    for sub in ("documents", "prompts"):
        shutil.copytree(src / sub, dom / sub)
    (dom / "tasks").mkdir()
    for f in files:
        shutil.copy2(f, dom / "tasks" / f.name)
    if cache:
        shutil.copytree(G.root_of(files[0]) / CACHE_DIRNAME, dest / CACHE_DIRNAME)
    return dest


@pytest.mark.parametrize("where", ["data_root", "relocated"])
def test_end_to_end_through_plugin_grade_runner_gives_the_same_scores(matrix, tmp_path, where):
    """The runner under the tau3 venv, its environment WITHOUT TAU2_DATA_DIR. data_root: the rows' answer files are
    the test data root's own. relocated: they sit in a copy at another path than tau2's own fallback
    (<tau3-bench>/data), so a grader that did not derive the root from answer_files[0] would raise (tau2 imported
    with another data root): equal scores show it did."""
    res = matrix["res"]
    picks = _picks(matrix)
    root = _data_copy(picks, tmp_path / "data_copy") if where == "relocated" else data()
    rows = [G.plugin_row(root / "tau2" / "domains" / G.DOMAIN / "tasks" / f.name, variant=VARIANT) for f in picks]
    assert all(G.root_of(r["answer_files"][0]) == root and Path(r["answer_files"][0]).is_file() for r in rows)
    d = tmp_path / "plugin"
    d.mkdir()
    shutil.copy2(PLUGIN_API, d / "plugin_api.py")
    shutil.copy2(GRADER, d / "benchmark_grader.py")
    (d / "tasks.json").write_text(json.dumps(rows, sort_keys=True, indent=1) + "\n")
    expect = []
    for p in sorted(d.iterdir()):
        expect += ["--expect", f"{p.name}={_sha(p)}"]
    (tmp_path / "tgz").mkdir()
    jobs, want = [], {}
    for f in picks:
        tid = _id(f)
        for kind in ("golden", "stop", "agent_error", "missing"):
            bundle = None if kind == "missing" else matrix["base"] / "bundles" / f.stem / kind
            tgz = _snapshot_tgz(bundle, tmp_path / "tgz" / f"{f.stem}_{kind}.tgz")
            jid = f"{tid}__{kind}"
            jobs.append({"job_id": jid, "task_id": tid, "snapshot_tgz": str(tgz), "snapshot_sha256": _sha(tgz),
                         "snapshot_root": "evo/work", "meta": {"kind": kind}})
            want[jid] = res[tid][kind]
    jf = tmp_path / "jobs.jsonl"
    jf.write_text("".join(json.dumps(j) + "\n" for j in jobs))
    results = tmp_path / "grade" / "results.jsonl"
    (tmp_path / "tmp").mkdir()
    env = offline_env(None, HOME=matrix["home"], TMPDIR=tmp_path / "tmp")
    assert "TAU2_DATA_DIR" not in env and not any(k in env for k in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"))
    with open(tmp_path / "runner.log", "wb") as fh:
        p = subprocess.run([str(G.TAU3_PY), str(RUNNER), "run", "--jobs", str(jf), "--results", str(results),
                            "--work", str(tmp_path / "work"), "--plugin-dir", str(d), *expect, "--max-procs", "4",
                            "--job-timeout", "600"], cwd=str(tmp_path), stdout=fh, stderr=subprocess.STDOUT,
                           env=env, timeout=900)
    assert p.returncode == 0, (tmp_path / "runner.log").read_text()[-4000:]
    lines = {j["job_id"]: j for j in map(json.loads, results.read_text().splitlines())}
    assert sorted(lines) == sorted(want)
    for jid, w in want.items():
        got = lines[jid]
        if "score" in w:
            assert (got["status"], got["score"]) == ("ok", w["score"]), (jid, got.get("reason"), w)
        else:
            assert got["status"] == GF and got["reason"].startswith(f"grade() raised {w['raised']}: "), (jid, got)
    assert sorted(jid for jid, g in lines.items() if g["score"] == 1.0) == sorted(f"{_id(f)}__golden" for f in picks)
    meta = json.loads(results.with_name("worker_meta.json").read_text())
    assert meta["statuses"] == {"ok": 9, GF: 3} and meta["python"].startswith("3.12")
    assert Path(meta["executable"]) == G.TAU3_PY
    assert sandboxes_in(tmp_path / "tmp") == []
