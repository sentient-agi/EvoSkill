"""scripts/tau3_cc/tau3_retrieval.py: tau2's "alltools" retrieval with its three process-local changes: the dense
embeddings asked through
OpenRouter (openai/text-embedding-3-large on Azure, the key in memory only), tau2's global embeddings cache pointed
READ-ONLY at <data root>/.embeddings_cache, and every KB shell sandbox denied reads of the data root, /opt, /root,
/home and /Users.

OFFLINE. tau2 reads TAU2_DATA_DIR once per process and install() patches tau2 process-wide, so every check that
imports tau2 runs in its own CHILD under the tau3-bench venv (third_party/tau3-bench/.venv, tau2 1.0.1) with an
emptied environment: PYTHON_DOTENV_DISABLED=1, LITELLM_LOCAL_MODEL_COST_MAP=True, TAU2_DATA_DIR, HOME and TMPDIR in
the test's tmp dir, HTTPS_PROXY / HTTP_PROXY / ALL_PROXY on a dead local port, and PATH = /usr/bin:/bin plus the dirs
holding srt, rg and node (tau2 builds the alltools shell sandbox, which checks for srt and rg; srt is a node script).
No key exists anywhere here: the transport is exercised against a fake openai.OpenAI (monkeypatched in the child) with
a made-up "sk-or-v1-TESTONLY" string. Nothing reads a .env file or a token file. The data root is the
curated third_party/tau3_data_363133a (or $EVOSKILL_TAU3_DATA_ROOT, a copy with its .embeddings_cache), only
read; the cases that need another root build a temporary one.

What must hold:
  - the module imports only the standard library; its constants are the pinned ones;
  - install() on the root returns the record (variant, the embedder spec with transport, provider and key False, the
    cache dir, read-only, with the precompute's two files and their shas, the deny list, the module's sha), and tau2's
    "openai" embedder is the transport everywhere tau2 looks it up;
  - the read-only cache hits (698 x 3072 float64, ids in the knowledge base's order); a miss and a stale entry (other
    document ids) raise and delete nothing; put and the incremental API raise; an absent cache dir raises;
  - offline (key None) any embedding request raises RetrievalError, also through tau2's own KB_search_dense tool, while
    KB_search_bm25 answers; the environment has 17 tools;
  - with a fake client: model openai/text-embedding-3-large, extra_body {"provider": {"order": ["azure"],
    "allow_fallbacks": false}}, the key, OpenRouter's base URL, timeout 120 s and max_retries 0; batches of 64; vectors
    put back in input order by index; a failing request is tried ATTEMPTS times (waits 2, 4, 8 s) and then raises
    with the error's type and HTTP status only (an upstream body quoting "sk-proj-..." never reaches the message); a
    wrong dimension raises at once; tau2's KB_search_dense answers through it;
  - the hardened sandbox hides the task file (expected actions), the bank's db.json and /opt from `cd / && ...`
    while `ls` lists the knowledge base, and the SAME commands without the hardening leak them (the control that shows
    the probe is not vacuous when /opt holds anything); skipped only when srt is absent;
  - a sandbox whose knowledge-base dir would fall under a denied path (TMPDIR inside the data root) is refused, and
    every sandbox's temp dir is removed at exit, hardened or not;
  - install() refuses a tau2 whose alltools variant is not BM25 + dense openai/text-embedding-3-large + a read-only
    shell (RETRIEVAL_VARIANTS monkeypatched in the child), and a tau2 imported with another data root.
  - no capabilities (Linux): srt's `bwrap` is a wrapper that runs the real one with
    --cap-drop ALL first and srt's arguments verbatim (checked against a fake bwrap, spaces and quotes included);
    on Linux the wrapper dir leads the PATH srt gets, and install() runs probe_shell and raises unless the shell
    shows CapPrm = CapEff = 0. A macOS host has no root and no /proc, so: the forced-Linux path is run with a
    fake bwrap -- the KB shell's own `bwrap` IS the wrapper, and the probe fails closed (no capability line can be
    read, so install() raises); parse_caps reads /proc status text; the real capability drop is NOT tested here
    (it needs a Linux host as root: the machine smoke);
  - probe_shell on macOS: in the hardened sandbox a marker file reads back and, after `umount` attempts, the data
    root, /opt and /Users list nothing; the SAME probe without the hardening lists them (the control);
  - the hardened shell's environment is SHELL_ENV_KEEP only (no TAU2_DATA_DIR, proxies or tau2 switches), while the
    unhardened one shows them; install(harden_shell=False) after a hardened install() restores tau2's own settings
    and subprocess and records shell_deny_read None.
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

REPO = Path(__file__).resolve().parents[1]
TR_DIR = REPO / "scripts" / "tau3_cc"
TR_FILE = TR_DIR / "tau3_retrieval.py"
TAU3_PY = REPO / "third_party" / "tau3-bench" / ".venv" / "bin" / "python"
CURATED_ROOT = REPO / "third_party" / "tau3_data_363133a"   # scripts/tau3_make_data_root.py + the embeddings precompute
DOMAIN = "tau2/domains/banking_knowledge"
DEAD = "http://127.0.0.1:9"
FAKE_KEY = "sk-or-v1-TESTONLY-not-a-key"
CACHE_ROWS = {
    ".embeddings_cache/62eca72522cacac783980da07ba707af2dfa2df0aec9def44ea0fcb260127fc2_edd03519131f2b849e5b90cd0e673657.pkl":
        "b3c69d7afea9441b4e1e61c06f5a132ac6fbf46031c210623824db37b436b046",
    ".embeddings_cache/metadata.json": "e4392566a6fc664722fbe5858cc27a03697e08398b4cb2e644c014a6d0216775",
}


def _load():
    spec = importlib.util.spec_from_file_location("tau3_retrieval_under_test", TR_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


TR = _load()


def data() -> Path:
    return Path(os.environ.get("EVOSKILL_TAU3_DATA_ROOT") or CURATED_ROOT)


def srt_path():
    """/usr/bin:/bin and the dirs holding srt, rg and node (+ bwrap, socat on Linux); None when one is missing."""
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


def _need(root=None):
    if not TAU3_PY.is_file():
        pytest.skip(f"the tau3-bench venv interpreter {TAU3_PY} is absent")
    r = Path(root or data())
    if len(list((r / TR.CACHE_DIRNAME).glob("*.pkl"))) != 1:
        pytest.skip(f"{r / TR.CACHE_DIRNAME} holds no single embeddings entry")
    if srt_path() is None:
        pytest.skip("srt, rg or node is absent (tau2 builds the alltools shell sandbox with srt)")


# The prelude of every child: argv[1] = the dir of tau3_retrieval.py, argv[2] = a JSON object of parameters.
PRELUDE = r'''
import hashlib, json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import tau3_retrieval as TR
P = json.loads(sys.argv[2])
ROOT = Path(os.environ["TAU2_DATA_DIR"])
OUT = {}


def emit():
    print("RESULT " + json.dumps(OUT, default=str))


def raised(fn):
    try:
        fn()
    except Exception as e:
        return {"raised": type(e).__name__, "message": str(e)[:600]}
    return {"raised": None}


def docs():
    from tau2.domains.banking_knowledge import retrieval as R
    from tau2.domains.banking_knowledge.environment import get_knowledge_base
    return [{"id": d["id"], "text": d["text"]} for d in R.get_or_create_docs(get_knowledge_base())]


def call(env, name, **args):
    from tau2.data_model.message import ToolCall
    tm = env.get_response(ToolCall(id=f"toolu_01{name}", name=name, arguments=args, requestor="assistant"))
    return {"error": tm.error, "content": tm.content}
'''


def _child(tmp_path: Path, body: str, params: dict | None = None, *, root=None, tmpdir=None, path=None) -> dict:
    cwd, home = tmp_path / "cwd", tmp_path / "home"
    tmpd = Path(tmpdir) if tmpdir else tmp_path / "tmp"
    for d in (cwd, home, tmpd):
        d.mkdir(parents=True, exist_ok=True)
    script = tmp_path / "child.py"
    script.write_text(PRELUDE + body + "\nemit()\n")
    env = {"PATH": path or srt_path(), "HOME": str(home), "TMPDIR": str(tmpd), "LC_ALL": "C.UTF-8",
           "PYTHON_DOTENV_DISABLED": "1", "LITELLM_LOCAL_MODEL_COST_MAP": "True", "PYTHONDONTWRITEBYTECODE": "1",
           "TAU2_DATA_DIR": str(root or data()), "HTTPS_PROXY": DEAD, "HTTP_PROXY": DEAD, "ALL_PROXY": DEAD,
           "https_proxy": DEAD, "http_proxy": DEAD, "all_proxy": DEAD}
    r = subprocess.run([str(TAU3_PY), "-I", "-B", str(script), str(TR_DIR), json.dumps(params or {})], cwd=str(cwd),
                       capture_output=True, text=True, env=env, timeout=600)
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
    assert r.returncode == 0 and len(lines) == 1, (r.returncode, r.stdout[-3000:], r.stderr[-3000:])
    assert FAKE_KEY not in r.stdout + r.stderr, "the (fake) key reached the child's output"
    assert list(cwd.iterdir()) == [], "the child wrote into its working directory (tau2's default cache?)"
    out = json.loads(lines[0][len("RESULT "):])
    out["_tmp_left"] = sorted(p.name for p in tmpd.iterdir() if p.name.startswith("agentic_search_"))
    return out


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _root_copy(dest: Path, *, cache: bool = True) -> Path:
    """A data root at another path: the banking db, documents, prompts, tasks, user_simulator (and the cache)."""
    src = data()
    for rel in (f"{DOMAIN}/documents", f"{DOMAIN}/prompts", f"{DOMAIN}/tasks", "tau2/user_simulator"):
        shutil.copytree(src / rel, dest / rel)
    shutil.copy2(src / DOMAIN / "db.json", dest / DOMAIN / "db.json")
    if cache:
        shutil.copytree(src / TR.CACHE_DIRNAME, dest / TR.CACHE_DIRNAME)
    return dest


# ───────────────────────────────────────────────────────────────────────────── no tau2 needed
def test_the_module_imports_only_the_standard_library_and_states_the_pinned_constants():
    tops = set()
    for n in ast.parse(TR_FILE.read_text()).body:
        if isinstance(n, ast.Import):
            tops |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            tops.add(n.module.split(".")[0])
    assert tops <= set(sys.stdlib_module_names) | {"__future__"}, tops
    assert (TR.VARIANT, TR.EMBEDDER_TYPE, TR.TAU2_MODEL, TR.OR_MODEL) == (
        "alltools", "openai", "text-embedding-3-large", "openai/text-embedding-3-large")
    assert TR.PROVIDER == {"order": ["azure"], "allow_fallbacks": False}
    assert (TR.CACHE_DIRNAME, TR.EMBED_DIM, TR.BATCH, TR.ATTEMPTS, TR.REQUEST_TIMEOUT_S, TR.BACKOFF_S) == (
        ".embeddings_cache", 3072, 64, 3, 10.0, (1.0, 2.0))       # 10 s x 3 tries, waits 1 s and 2 s
    assert TR.DENY_READ_FIXED == ("/opt", "/root", "/home", "/Users")
    assert TR.OPENROUTER_BASE == "https://openrouter.ai/api/v1"


def test_paths_and_the_deny_list(tmp_path):
    real = tmp_path / "real_root"
    real.mkdir()
    link = tmp_path / "link_root"
    link.symlink_to(real)
    assert TR.cache_dir(link) == link / ".embeddings_cache"
    assert TR.deny_read_paths(link) == (str(real.resolve()), str(link), "/opt", "/root", "/home", "/Users")
    assert TR.deny_read_paths(real.resolve()) == (str(real.resolve()), "/opt", "/root", "/home", "/Users")


def test_the_real_roots_cache_files():
    if not (data() / TR.CACHE_DIRNAME).is_dir():
        pytest.skip(f"{data()} has no {TR.CACHE_DIRNAME}")
    assert TR.cache_files(data()) == CACHE_ROWS


def test_env_without_keys(monkeypatch):
    for k in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "OR_API_KEY"):
        monkeypatch.setenv(k, "x")
    monkeypatch.setenv("TAU3_RETRIEVAL_TEST_KEEP", "1")
    TR.env_without_keys()
    assert not {"OPENAI_API_KEY", "OPENROUTER_API_KEY", "OR_API_KEY"} & set(os.environ)
    assert os.environ["TAU3_RETRIEVAL_TEST_KEEP"] == "1"


FAKE_BWRAP = "#!/bin/sh\nfor a in \"$@\"; do printf '[%s]\\n' \"$a\"; done\n"   # prints each argument it got


def _fake_bwrap(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    f = d / "bwrap"
    f.write_text(FAKE_BWRAP)
    f.chmod(0o755)
    return f


def test_the_bwrap_wrapper_drops_every_capability_and_passes_srts_arguments_verbatim(tmp_path):
    assert TR.BWRAP_CAP_DROP == ("--cap-drop", "ALL")
    real = _fake_bwrap(tmp_path / "real bin's dir")                      # a space and a quote in the path
    w = tmp_path / "w" / "bwrap"
    w.parent.mkdir()
    w.write_text(TR.bwrap_wrapper_text(str(real)))
    w.chmod(0o755)
    args = ["--new-session", "--ro-bind", "/", "/", "--tmpfs", "/Users", "--", "/bin/bash", "-c",
            "cd / && echo 'a b' \"$HOME\" $x; ls -A", ""]
    r = subprocess.run([str(w), *args], capture_output=True, text=True, timeout=30, env={"PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [f"[{a}]" for a in ("--cap-drop", "ALL", *args)]


def test_the_wrapper_dir_and_the_shell_environment(tmp_path, monkeypatch):
    """_bwrap_wrapper_dir: a private dir holding only `bwrap` = the wrapper around the real bwrap on PATH (made once);
    shell_env: SHELL_ENV_KEEP only, PATH led by that dir on Linux and untouched elsewhere."""
    fake = _fake_bwrap(tmp_path / "fakebin")
    monkeypatch.setitem(TR._STATE, "bwrap_dir", None)
    monkeypatch.setenv("PATH", f"{fake.parent}:/usr/bin:/bin")
    monkeypatch.setenv("TAU2_DATA_DIR", "/somewhere")
    monkeypatch.setenv("HTTPS_PROXY", DEAD)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(TR, "_linux", lambda: False)
    env = TR.shell_env()
    assert set(env) <= set(TR.SHELL_ENV_KEEP) and env["PATH"] == f"{fake.parent}:/usr/bin:/bin"
    assert env["HOME"] == str(tmp_path) and TR._STATE["bwrap_dir"] is None
    monkeypatch.setattr(TR, "_linux", lambda: True)
    d = tmp_path / "unset"
    try:
        env = TR.shell_env()
        d = Path(TR._STATE["bwrap_dir"])
        assert env["PATH"] == f"{d}:{fake.parent}:/usr/bin:/bin" and set(env) <= set(TR.SHELL_ENV_KEEP)
        assert [p.name for p in d.iterdir()] == ["bwrap"] and (d.stat().st_mode & 0o777) == 0o700
        assert (d / "bwrap").read_text() == TR.bwrap_wrapper_text(os.path.realpath(fake))
        assert os.access(d / "bwrap", os.X_OK)
        assert TR.shell_env()["PATH"].split(":")[0] == str(d)            # made once
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        monkeypatch.setitem(TR._STATE, "bwrap_dir", None)
        with pytest.raises(TR.RetrievalError, match="bwrap is not on PATH"):
            TR.shell_env()
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_parse_caps_reads_proc_status():
    none = "CapInh:\t0000000000000000\nCapPrm:\t0000000000000000\nCapEff:\t0000000000000000\n" \
           "CapBnd:\t0000000000000000\nCapAmb:\t0000000000000000\nNoNewPrivs:\t1\n"
    assert TR.parse_caps(none) == {"CapInh": 0, "CapPrm": 0, "CapEff": 0, "CapBnd": 0, "CapAmb": 0, "NoNewPrivs": 1}
    root = "CapPrm:\t000001ffffffffff\nCapEff:\t000001ffffffffff\nNoNewPrivs:\t0\nName:\tgrep\nCapBnd: zz\n\n"
    assert TR.parse_caps(root) == {"CapPrm": 0x1ffffffffff, "CapEff": 0x1ffffffffff, "NoNewPrivs": 0}
    assert TR.parse_caps("") == {} and TR.parse_caps(None) == {}
    assert TR.CAP_PROBE.startswith("cd / && grep ") and " /" not in TR.CAP_PROBE.replace("cd / ", "")


# ───────────────────────────────────────────────────────────────────────────── install() offline
OFFLINE = r'''
rec = TR.install(ROOT, key=None)
OUT["rec"] = rec
from tau2.knowledge import embedders as E
from tau2.knowledge import embeddings_cache as EC
from tau2.knowledge.document_preprocessors import embedding_indexer as EI
from tau2.knowledge.embedders import openai_embedder as OE
from tau2.knowledge.input_preprocessors import embedding_encoder as EE
OUT["embedder_everywhere"] = sorted({c.__name__ for c in (E.OpenAIEmbedder, OE.OpenAIEmbedder,
                                                         EI.EMBEDDER_REGISTRY["openai"], EE.EMBEDDER_REGISTRY["openai"])})
D = docs()
cache = EC.get_embeddings_cache()
OUT["cache_type"] = type(cache).__name__
emb, ids = cache.get(D, "openai", {"model": "text-embedding-3-large"})
OUT["hit"] = {"shape": list(emb.shape), "dtype": str(emb.dtype), "ids_in_kb_order": ids == [d["id"] for d in D]}
listing = lambda d: sorted((p.name, hashlib.sha256(p.read_bytes()).hexdigest()) for p in Path(d).iterdir())
before = listing(cache.cache_dir)
OUT["miss"] = raised(lambda: cache.get(D, "openai", {"model": "text-embedding-3-small"}))
OUT["put"] = raised(lambda: cache.put(D, "openai", emb, ids, {"model": "text-embedding-3-large"}))
OUT["get_incremental"] = raised(lambda: cache.get_incremental(D, "openai", {"model": "text-embedding-3-large"}))
OUT["put_incremental"] = raised(lambda: cache.put_incremental(D, "openai", {}, {"model": "text-embedding-3-large"}))
OUT["cache_dir_unchanged"] = listing(cache.cache_dir) == before

# a stale entry (the real vectors under the real key, one document id replaced) in a temporary cache dir
import pickle, tempfile
stale_dir = Path(tempfile.mkdtemp(prefix="stale_cache_"))
pkl = next(cache.cache_dir.glob("*.pkl"))
with open(pkl, "rb") as fh:
    d = pickle.load(fh)
# another document SET (one id replaced): a mere re-ordering is the same vectors and is served (in_kb_order)
d["doc_ids"] = list(d["doc_ids"][:-1]) + ["doc_not_in_this_knowledge_base"]
with open(stale_dir / pkl.name, "wb") as fh:
    pickle.dump(d, fh)
stale_sha = hashlib.sha256((stale_dir / pkl.name).read_bytes()).hexdigest()
RO = TR.readonly_cache_class()
OUT["stale"] = raised(lambda: RO(str(stale_dir)).get(D, "openai", {"model": "text-embedding-3-large"}))
OUT["stale_kept"] = [p.name for p in stale_dir.iterdir()] == [pkl.name] and \
    hashlib.sha256((stale_dir / pkl.name).read_bytes()).hexdigest() == stale_sha
(stale_dir / pkl.name).unlink()
stale_dir.rmdir()
OUT["absent_dir"] = raised(lambda: RO(str(stale_dir)))
OUT["absent_dir_created"] = stale_dir.exists()

OUT["offline_embed"] = raised(lambda: TR.embedder_class()(model=TR.TAU2_MODEL).embed(["a query"]))
OUT["other_model"] = raised(lambda: TR.embedder_class()(model="text-embedding-3-small"))
from tau2.domains.banking_knowledge.environment import get_environment
env = get_environment(retrieval_variant="alltools")
OUT["tools"] = sorted(t.name for t in env.get_tools())
OUT["dense_tool"] = call(env, "KB_search_dense", query="How do I dispute a card transaction?")
bm = call(env, "KB_search_bm25", query="How do I dispute a card transaction?")
OUT["bm25_tool"] = {"error": bm["error"], "results": bm["content"].count("\n   ID: doc_")}
sb = TR._STATE["sandboxes"]
OUT["sandboxes"] = [str(s.sandbox_dir) for s in sb]
OUT["deny_read"] = json.loads(Path(sb[0].settings_path).read_text())["filesystem"]["denyRead"]
'''


@pytest.fixture(scope="module")
def offline(tmp_path_factory):
    _need()
    return _child(tmp_path_factory.mktemp("tau3_retrieval_offline"), OFFLINE)


def test_install_returns_the_record(offline):
    root = data()
    rec = offline["rec"]
    assert rec == {"variant": "alltools",
                   "embedder": {"type": "openai", "model": "text-embedding-3-large", "transport": "openrouter",
                                "openrouter_model": "openai/text-embedding-3-large",
                                "provider": {"order": ["azure"], "allow_fallbacks": False}, "key": False},
                   "cache": {"dir": str(root / ".embeddings_cache"), "readonly": True, "files": CACHE_ROWS},
                   "shell_deny_read": list(TR.deny_read_paths(root)),
                   "module_sha256": _sha(TR_FILE)}
    assert rec["shell_deny_read"][1:] == ["/opt", "/root", "/home", "/Users"] or \
        rec["shell_deny_read"][2:] == ["/opt", "/root", "/home", "/Users"]
    assert str(root.resolve()) in rec["shell_deny_read"]
    assert offline["embedder_everywhere"] == ["OpenRouterTransportEmbedder"]


def test_the_read_only_cache_hits_and_refuses_everything_else(offline):
    assert offline["cache_type"] == "ReadOnlyEmbeddingsCache"
    assert offline["hit"] == {"shape": [698, 3072], "dtype": "float64", "ids_in_kb_order": True}
    assert offline["miss"]["raised"] == "RetrievalError" and "no cached embeddings" in offline["miss"]["message"]
    for k in ("put", "get_incremental", "put_incremental"):
        assert offline[k]["raised"] == "RetrievalError", (k, offline[k])
    assert offline["cache_dir_unchanged"]
    assert offline["stale"]["raised"] == "RetrievalError" and "document ids differ" in offline["stale"]["message"]
    assert offline["stale_kept"], "a stale entry was deleted or rewritten (tau2's own cache unlinks it)"
    assert offline["absent_dir"]["raised"] == "RetrievalError" and "is absent" in offline["absent_dir"]["message"]
    assert offline["absent_dir_created"] is False


def test_offline_every_embedding_request_raises(offline):
    assert offline["offline_embed"]["raised"] == "RetrievalError"
    assert "offline process (no OpenRouter key)" in offline["offline_embed"]["message"]
    assert offline["other_model"]["raised"] == "RetrievalError"
    assert offline["dense_tool"]["error"] is True and "no OpenRouter key" in offline["dense_tool"]["content"]
    assert offline["bm25_tool"] == {"error": False, "results": 10}


def test_the_environment_has_the_17_alltools_tools_and_the_hardened_sandbox(offline):
    tools = offline["tools"]
    assert len(tools) == 17 and {"KB_search_bm25", "KB_search_dense", "shell"} <= set(tools)
    assert not {"KB_search", "grep"} & set(tools)
    assert len(offline["sandboxes"]) == 1
    deny = offline["deny_read"]
    assert deny[-len(TR.deny_read_paths(data())):] == list(TR.deny_read_paths(data()))
    assert {"~/.ssh", "/etc/passwd"} <= set(deny)           # tau2's own entries are kept
    assert offline["_tmp_left"] == [] and not any(Path(p).exists() for p in offline["sandboxes"])


# ───────────────────────────────────────────────────────────────────────────── the transport, with a fake client
FAKE = r'''
import types
import openai

CALLS = {"init": [], "create": []}
MODE = {"m": "ok"}


class Item:
    def __init__(self, index, embedding):
        self.index, self.embedding = index, embedding


class Upstream(Exception):
    status_code = 401


def vec(text):
    w = text.split()[-1]
    return [float(w) if w.isdigit() else 0.5] + [1.0] * 3071


class FakeEmbeddings:
    def create(self, input, model, extra_body=None, **kw):
        CALLS["create"].append({"n": len(input), "model": model, "extra_body": extra_body, "kw": sorted(kw)})
        if MODE["m"] == "fail":
            raise Upstream("Error code: 401 - {'error': {'message': 'Incorrect API key provided: "
                           "sk-proj-abcd************************wxyz'}}")
        if MODE["m"] == "dim":
            return types.SimpleNamespace(data=[Item(i, [1.0] * 3071) for i in range(len(input))])
        return types.SimpleNamespace(data=list(reversed([Item(i, vec(t)) for i, t in enumerate(input)])))


class FakeOpenAI:
    def __init__(self, **kw):
        CALLS["init"].append({k: (v == P["key"] if k == "api_key" else v) for k, v in kw.items()})
        self.embeddings = FakeEmbeddings()


openai.OpenAI = FakeOpenAI
SLEEPS = []
TR.time = types.SimpleNamespace(sleep=SLEEPS.append)
OUT["rec_key"] = TR.install(ROOT, key=P["key"])["embedder"]["key"]
em = TR.embedder_class()(model=TR.TAU2_MODEL)
arr = em.embed([f"text {i}" for i in range(130)])
OUT["shape"] = list(arr.shape)
OUT["in_input_order"] = [int(x) for x in arr[:, 0]] == list(range(130))
OUT["batches"] = [c["n"] for c in CALLS["create"]]
OUT["requests"] = sorted({json.dumps({k: c[k] for k in ("model", "extra_body", "kw")}, sort_keys=True)
                          for c in CALLS["create"]})
OUT["init"] = list(CALLS["init"])
CALLS["create"].clear()
MODE["m"] = "fail"
OUT["fail"] = raised(lambda: em.embed(["a query 1"]))
OUT["fail_attempts"] = len(CALLS["create"])
OUT["sleeps"] = list(SLEEPS)
OUT["outage"] = dict(TR._STATE["dense_outage"] or {})
try:
    em.embed(["a query 1"])
except TR.RetrievalError as e:
    OUT["fail_chained"] = e.__cause__ is not None or e.__context__ is not None
CALLS["create"].clear()
MODE["m"] = "dim"
OUT["dim"] = raised(lambda: em.embed(["a query 1"]))
OUT["dim_attempts"] = len(CALLS["create"])
MODE["m"] = "ok"
CALLS["create"].clear()
from tau2.domains.banking_knowledge.environment import get_environment
env = get_environment(retrieval_variant="alltools")
OUT["build_requests"] = len(CALLS["create"])              # the documents come from the cache: none
dense = call(env, "KB_search_dense", query="How do I dispute a card transaction 3")
OUT["dense_tool"] = {"error": dense["error"], "results": dense["content"].count("\n   ID: doc_")}
OUT["query_requests"] = [{"n": c["n"], "model": c["model"], "extra_body": c["extra_body"]} for c in CALLS["create"]]
OUT["all_inits"] = CALLS["init"]                         # ours, then the one tau2's query encoder built
'''


def test_the_transport_asks_openrouter_for_the_same_model_on_azure(tmp_path):
    _need()
    out = _child(tmp_path, FAKE, {"key": FAKE_KEY})
    assert out["rec_key"] is True
    assert out["shape"] == [130, 3072] and out["in_input_order"] and out["batches"] == [64, 64, 2]
    assert [json.loads(r) for r in out["requests"]] == [
        {"model": "openai/text-embedding-3-large", "extra_body": {"provider": {"order": ["azure"],
                                                                               "allow_fallbacks": False}}, "kw": []}]
    client = {"api_key": True, "base_url": "https://openrouter.ai/api/v1", "timeout": 10.0, "max_retries": 0}
    assert out["init"] == [client] and out["all_inits"] == [client, client]
    assert out["fail"]["raised"] == "RetrievalError" and out["fail_attempts"] == TR.ATTEMPTS == 3
    assert out["fail"]["message"] == "OpenRouter openai/text-embedding-3-large failed 3x: Upstream (status 401)"
    o = out["outage"]                                          # the flag the bridge turns into an infrastructure end
    assert (o["attempts"], o["timeout_s"], o["error"], o["texts"]) == (3, 10.0, "Upstream (status 401)", 1), o
    assert "sk-proj-" not in json.dumps(out) and "Incorrect API key" not in json.dumps(out)
    assert out["sleeps"][:2] == [1.0, 2.0] and out["fail_chained"] is False
    assert out["dim"]["raised"] == "RetrievalError" and "want 3072" in out["dim"]["message"]
    assert out["dim_attempts"] == 1
    assert out["build_requests"] == 0
    assert out["dense_tool"] == {"error": False, "results": 10}
    assert out["query_requests"] == [{"n": 1, "model": "openai/text-embedding-3-large",
                                      "extra_body": {"provider": {"order": ["azure"], "allow_fallbacks": False}}}]
    assert out["_tmp_left"] == []


# ───────────────────────────────────────────────────────────────────────────── the shell sandbox
SHELL = r'''
TR.install(ROOT, key=None, harden_shell=P["harden"])
from tau2.domains.banking_knowledge.environment import get_environment
env = get_environment(retrieval_variant="alltools")
rel = ROOT.resolve().relative_to("/").as_posix()
for k, cmd in {"task": f"cd / && cat {rel}/tau2/domains/banking_knowledge/tasks/task_004.json",
               "db": f"cd / && head -c 300 {rel}/tau2/domains/banking_knowledge/db.json",
               "opt": "cd / && ls opt",
               "ls": "ls",
               "env": "env"}.items():
    OUT[k] = call(env, "shell", command=cmd)
OUT["sandboxes"] = [str(s.sandbox_dir) for s in TR._STATE["sandboxes"]]
OUT["probe"] = TR.probe_shell(ROOT)
OUT["shell_probe_state"] = TR._STATE["shell_probe"]
from tau2.knowledge import sandbox_manager as SM
OUT["guard"] = [SM.SandboxManager._has_escape_pattern(None, c) for c in
                (TR.CAP_PROBE, OUT["probe"]["deny_listing"]["cmd"], "cat " + TR.PROBE_FILE)]
OUT["probe_left"] = sorted(p.name for p in Path(os.environ["TMPDIR"]).iterdir() if p.name.startswith("agentic_"))
'''


@pytest.fixture(scope="module")
def shells(tmp_path_factory):
    """The SHELL child twice: hardened (True) and with tau2's own sandbox settings (False, the control)."""
    _need()
    base = tmp_path_factory.mktemp("tau3_retrieval_shell")
    return {h: _child(base / str(h), SHELL, {"harden": h}) for h in (True, False)}


def test_the_hardened_shell_hides_the_run_and_the_unhardened_one_leaks_it(shells):
    task_raw = (data() / DOMAIN / "tasks" / "task_004.json").read_text()
    task = json.loads(task_raw)
    db_head = (data() / DOMAIN / "db.json").read_bytes()[:300].decode("utf-8", "replace")
    opt = sorted(os.listdir("/opt")) if os.path.isdir("/opt") else []
    got = shells
    hard, soft = got[True], got[False]
    # the control: without the hardening, `cd / && <relative path>` passes tau2's own guard and reads the run
    assert json.loads(soft["task"]["content"]) == task, soft["task"]        # the expected actions, verbatim
    assert soft["db"]["content"].strip() == db_head.strip() and len(db_head.strip()) > 100, soft["db"]
    assert not opt or any(name in soft["opt"]["content"].split() for name in opt), soft["opt"]
    # hardened: nothing of it (on macOS a denied read answers "Operation not permitted", naming the path only)
    for k in ("task", "db", "opt"):
        c = hard[k]["content"]
        assert "evaluation_criteria" not in c and db_head.strip()[:60] not in c, (k, c)
        assert not any(n in c.split() for n in opt), (k, c)
        if sys.platform == "darwin":                      # srt's sandbox-exec profile: the read is denied
            assert c == "(no output)" or "Operation not permitted" in c, (k, c)
    assert "evaluation_criteria" not in json.dumps(hard) and task["user_scenario"]["instructions"][:80] not in \
        "".join(hard[k]["content"] for k in ("task", "db", "opt"))
    # the knowledge base itself stays readable, in both
    for out in (hard, soft):
        names = out["ls"]["content"].split()
        assert out["ls"]["error"] is False and "INDEX.md" in names and sum(n.endswith(".md") for n in names) > 600
    for out in (hard, soft):                              # removed at exit, hardened or not
        assert out["_tmp_left"] == [] and len(out["sandboxes"]) == 1
        assert not Path(out["sandboxes"][0]).exists()
    # the environment: SHELL_ENV_KEEP only when hardened; the bridge's whole environment without
    keys = {h: {ln.split("=", 1)[0] for ln in got[h]["env"]["content"].splitlines() if "=" in ln} for h in got}
    leaky = {"TAU2_DATA_DIR", "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "PYTHON_DOTENV_DISABLED",
             "LITELLM_LOCAL_MODEL_COST_MAP"}
    assert leaky <= keys[False], keys[False]
    assert not leaky & keys[True] and {"PATH", "HOME"} <= keys[True], keys[True]
    shell_added = {"PWD", "SHLVL", "_", "OLDPWD", "SANDBOX_RUNTIME", "__CF_USER_TEXT_ENCODING"}   # bash, env, srt, macOS
    assert keys[True] <= set(TR.SHELL_ENV_KEEP) | shell_added, keys[True] - set(TR.SHELL_ENV_KEEP)


def test_the_shell_probe_passes_hardened_and_fails_without_the_hardening(shells):
    """probe_shell (what install() requires on Linux) on macOS, in the two SHELL children: hardened, the marker
    reads back and every denied dir lists nothing after the `umount` attempts; unhardened (the control), the same
    probe lists the data root, /opt and /Users. Its commands pass tau2's own guard, and its sandbox is removed when
    it returns. install() runs no probe here (not Linux)."""
    got = shells
    hard, soft = got[True]["probe"], got[False]["probe"]
    linux = sys.platform.startswith("linux")
    assert hard["platform"] == sys.platform
    assert hard["ok"] == dict({"commands_run": True, "denied_dirs_unreadable": True},
                              **({"no_capabilities": True} if linux else {})) and hard["passed"] is True, hard
    assert hard["marker"]["stdout"] == TR.PROBE_TEXT and hard["deny_listing"]["stdout"] == ""
    rel = data().resolve().relative_to("/").as_posix()
    assert hard["deny_listing"]["cmd"].startswith("cd / && umount ") and f"ls -A {rel}" in hard["deny_listing"]["cmd"]
    for d in ("opt", "root", "home", "Users"):
        assert f"umount {d} ;" in hard["deny_listing"]["cmd"] and f"ls -A {d}" in hard["deny_listing"]["cmd"]
    assert soft["ok"]["commands_run"] is True and soft["ok"]["denied_dirs_unreadable"] is False, soft
    assert soft["passed"] is False
    names = soft["deny_listing"]["stdout"].split()
    assert {"tau2", "MANIFEST.sha256", ".embeddings_cache"} <= set(names), names     # the data root, listed
    assert not os.path.isdir("/opt") or set(os.listdir("/opt")) & set(names)
    for out in got.values():
        assert out["guard"] == [None, None, None]          # tau2's escape guard lets every probe command through
        assert out["probe_left"] == [Path(out["sandboxes"][0]).name]     # only the environment's own, until exit
    assert got[False]["shell_probe_state"] is None
    assert (got[True]["shell_probe_state"] or {}).get("passed") is (True if linux else None)


BURIED = r'''
TR.install(ROOT, key=None)
from tau2.domains.banking_knowledge.environment import get_environment
OUT["env"] = raised(lambda: get_environment(retrieval_variant="alltools"))
OUT["sandboxes"] = len(TR._STATE["sandboxes"])
'''


def test_a_sandbox_under_a_denied_path_is_refused_and_removed(tmp_path):
    """TMPDIR inside the data root (denied): srt would hide the knowledge base and every `shell` call would answer
    "(no output)" without an error; install's hardening refuses it loudly instead, and leaves no dir behind."""
    _need()
    root = _root_copy(tmp_path / "root")
    tmpd = root / "tmp"
    out = _child(tmp_path / "c", BURIED, root=root, tmpdir=tmpd)
    assert out["env"]["raised"] == "RetrievalError" and "is under the denied path(s)" in out["env"]["message"]
    assert out["sandboxes"] == 0 and list(tmpd.iterdir()) == []


# The Linux path on macOS: TR._linux forced True, a fake `bwrap` (prints its arguments) first on PATH. srt here is
# still macOS's (sandbox-exec), so the shell's commands run; what is checked is the plumbing and the fail-closed probe.
FORCED_LINUX = r'''
TR._linux = lambda: True
from tau2.knowledge import sandbox_manager as SM
OUT["install"] = raised(lambda: TR.install(ROOT, key=None))     # the real probe: no /proc here -> no capability line
OUT["installed_after_refusal"] = TR._STATE["installed"]
OUT["bwrap_dir"] = TR._STATE["bwrap_dir"]
OUT["srt_env_path_head"] = TR.shell_env()["PATH"].split(":")[0]
OUT["subprocess_shim"] = type(SM.subprocess).__name__
TR._require_shell_probe = lambda root: {"stubbed": True}          # past the probe, to look at the plumbing itself
OUT["record_deny"] = TR.install(ROOT, key=None)["shell_deny_read"]
from tau2.domains.banking_knowledge.environment import get_environment
env = get_environment(retrieval_variant="alltools")
OUT["which"] = call(env, "shell", command="command -v bwrap")
OUT["run"] = call(env, "shell", command="bwrap --ro-bind 'a b' c")
OUT["ls"] = call(env, "shell", command="ls | head -3")
'''


def test_forced_linux_the_shell_runs_the_wrapper_and_the_probe_fails_closed(tmp_path):
    _need()
    fake = _fake_bwrap(tmp_path / "fakebin")
    tmpd = tmp_path / "tmp"
    tmpd.mkdir()
    out = _child(tmp_path / "c", FORCED_LINUX, tmpdir=tmpd, path=f"{fake.parent}:{srt_path()}")
    ins = out["install"]
    assert ins["raised"] == "RetrievalError" and "failed its probe on this host (no_capabilities)" in ins["message"]
    assert out["installed_after_refusal"] is None
    d = out["bwrap_dir"]
    assert d and Path(d).parent == tmpd and Path(d).name.startswith("tau3_bwrap_")
    assert out["srt_env_path_head"] == d and out["subprocess_shim"] == "_SrtSubprocess"
    assert out["record_deny"] == list(TR.deny_read_paths(data()))
    assert out["which"]["error"] is False and out["which"]["content"].strip() == f"{d}/bwrap", out["which"]
    assert out["run"]["content"].splitlines() == ["[--cap-drop]", "[ALL]", "[--ro-bind]", "[a b]", "[c]"], out["run"]
    assert out["ls"]["error"] is False and len(out["ls"]["content"].split()) == 3
    assert not Path(d).exists(), "the wrapper dir outlived the process"
    assert out["_tmp_left"] == []


UNDO = r'''
from tau2.knowledge import sandbox_manager as SM
import subprocess
OUT["hard"] = TR.install(ROOT, key=None)["shell_deny_read"]
OUT["shim"] = SM.subprocess is not subprocess
OUT["soft"] = TR.install(ROOT, key=None, harden_shell=False)["shell_deny_read"]
OUT["restored"] = SM.subprocess is subprocess and TR._STATE["deny_read"] == ()
sb = SM.SandboxManager(allow_writes=False)
OUT["deny_after"] = json.loads(Path(sb.settings_path).read_text())["filesystem"]["denyRead"]
sb.cleanup()
OUT["again"] = TR.install(ROOT, key=None)["shell_deny_read"]
sb = SM.SandboxManager(allow_writes=False)
OUT["deny_again"] = json.loads(Path(sb.settings_path).read_text())["filesystem"]["denyRead"]
OUT["shim_again"] = SM.subprocess is not subprocess
sb.cleanup()
'''


def test_install_without_hardening_undoes_an_earlier_hardening(tmp_path):
    _need()
    out = _child(tmp_path, UNDO)
    deny = list(TR.deny_read_paths(data()))
    assert out["hard"] == deny and out["shim"] is True
    assert out["soft"] is None and out["restored"] is True
    assert out["deny_after"] == ["~/.ssh", "~/.aws", "~/.config", "~/.gnupg", "/etc/passwd", "/etc/shadow"]
    assert out["again"] == deny and out["deny_again"][-len(deny):] == deny and out["shim_again"] is True


# ───────────────────────────────────────────────────────────────────────────── refusals
REFUSE = r'''
import dataclasses
from tau2.domains.banking_knowledge import retrieval as R
orig = R.RETRIEVAL_VARIANTS["alltools"]
dense, shell = orig.kb_search_dense, orig.shell
cases = {
    "other_model": dataclasses.replace(orig, kb_search_dense=dataclasses.replace(dense,
                                                                                embedder_model="text-embedding-3-small")),
    "other_embedder": dataclasses.replace(orig, kb_search_dense=dataclasses.replace(dense, embedder_type="openrouter")),
    "writable_shell": dataclasses.replace(orig, shell=dataclasses.replace(shell, allow_writes=True)),
    "no_shell": dataclasses.replace(orig, shell=None),
    "no_dense": dataclasses.replace(orig, kb_search_dense=None),
    "no_bm25": dataclasses.replace(orig, kb_search_bm25=None),
    "absent": None,
}
for name, v in cases.items():
    if v is None:
        del R.RETRIEVAL_VARIANTS["alltools"]
    else:
        R.RETRIEVAL_VARIANTS["alltools"] = v
    OUT[name] = raised(lambda: TR.install(ROOT, key=None))
    R.RETRIEVAL_VARIANTS["alltools"] = orig
OUT["installed_after"] = TR._STATE["installed"]
OUT["other_root"] = raised(lambda: TR.install(ROOT / "elsewhere", key=None))
OUT["original"] = TR.install(ROOT, key=None)["variant"]
'''


def test_install_refuses_a_changed_variant_and_another_data_root(tmp_path):
    _need()
    out = _child(tmp_path, REFUSE)
    for k in ("other_model", "other_embedder", "writable_shell", "no_shell", "no_dense", "no_bm25", "absent"):
        assert out[k]["raised"] == "RetrievalError", (k, out[k])
        assert "variant is not BM25 + dense openai/text-embedding-3-large + a read-only shell" in out[k]["message"]
    assert out["installed_after"] is None                  # nothing was installed by a refused call
    assert out["other_root"]["raised"] == "RetrievalError" and "tau2 was imported with data root" in \
        out["other_root"]["message"]
    assert out["original"] == "alltools"


# tau2 loads the KB documents with an unordered Path.glob, so Linux reads them in another order than a Mac that wrote
# the cache, and tau2's get() would discard the entry. tau3_retrieval.in_kb_order (and the grader's and the machine check's copies) re-order instead.
REVERSE_KB = r'''
from tau2.domains.banking_knowledge import data_model as DM
_orig_load = DM.KnowledgeBase.load.__func__
def _reversed_load(cls, documents_dir):
    kb = _orig_load(cls, documents_dir)
    return cls(documents=dict(reversed(list(kb.documents.items()))))
DM.KnowledgeBase.load = classmethod(_reversed_load)
'''


def test_the_cache_serves_the_same_documents_read_in_another_order(tmp_path):
    _need()
    out = _child(tmp_path, REVERSE_KB + r'''
import pickle, numpy as np
TR.install(ROOT, key=None)
from tau2.knowledge.embeddings_cache import get_embeddings_cache
d = docs()
f = next((ROOT / TR.CACHE_DIRNAME).glob("*.pkl"))
data = pickle.loads(f.read_bytes())
OUT["kb_order_differs"] = [x["id"] for x in d] != data["doc_ids"]
emb, ids = get_embeddings_cache().get(d, TR.EMBEDDER_TYPE, {"model": TR.TAU2_MODEL})
OUT["ids_follow_kb"] = ids == [x["id"] for x in d]
pos = {k: i for i, k in enumerate(data["doc_ids"])}
OUT["rows_follow_ids"] = all(np.array_equal(emb[j], np.asarray(data["embeddings"])[pos[k]]) for j, k in enumerate(ids))
from tau2.domains.banking_knowledge.environment import get_environment
env = get_environment(retrieval_variant=TR.VARIANT)
OUT["bm25"] = call(env, "KB_search_bm25", query="dispute a card transaction")["error"]
try:
    TR.in_kb_order({"doc_ids": data["doc_ids"][:-1], "embeddings": data["embeddings"][:-1]}, ids, "x", TR.RetrievalError)
    OUT["missing_id_refused"] = False
except TR.RetrievalError:
    OUT["missing_id_refused"] = True
''')
    assert out["kb_order_differs"] is True            # the reversal really changed the order (not a vacuous check)
    assert out["ids_follow_kb"] and out["rows_follow_ids"]
    assert out["bm25"] is False and out["missing_id_refused"] is True
    assert out["_tmp_left"] == []


def test_the_grader_and_the_machine_check_accept_another_document_order(tmp_path):
    _need()
    grader = REPO / "examples" / "tau3_banking" / "grader.py"
    sys.path.insert(0, str(REPO / "scripts" / "daytona"))
    try:
        import tau3_cc_smoke as S
    finally:
        sys.path.pop(0)
    out = _child(tmp_path, REVERSE_KB + r"""
import importlib.util, io, contextlib
spec = importlib.util.spec_from_file_location("benchmark_grader", P["grader"])
G = importlib.util.module_from_spec(spec); spec.loader.exec_module(G)
from tau2.utils.utils import DATA_DIR
G._offline(Path(DATA_DIR))
from tau2.knowledge import embeddings_cache as EC
d = docs()
emb, ids = EC.get_embeddings_cache().get(d, "openai", {"model": "text-embedding-3-large"})
OUT["grader_cache"] = type(EC.get_embeddings_cache()).__name__
OUT["grader_ids_follow_kb"] = ids == [x["id"] for x in d]
OUT["grader_shape"] = list(emb.shape)
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    try:
        exec(compile(P["check"], "TAU3_CHECK_PY", "exec"), {"__name__": "__main__"})
    except SystemExit as e:
        OUT["check_exit"] = e.code
lines = buf.getvalue().splitlines()
ok = [l for l in lines if l.startswith("TAU3_CHECK ")]
OUT["check_fail"] = [l for l in lines if l.startswith("TAU3_CHECK_FAIL")]
OUT["check_embeddings"] = json.loads(ok[0][len("TAU3_CHECK "):]).get("embeddings") if ok else None
""", {"grader": str(grader), "check": S.TAU3_CHECK_PY})
    assert out["grader_cache"] == "ReadOnlyEmbeddingsCache" and out["grader_ids_follow_kb"] is True
    assert out["grader_shape"] == [698, 3072]
    emb = out["check_embeddings"]
    assert not [f for f in out["check_fail"] if "embedding" in f], out["check_fail"]
    assert emb and emb["cached_order_equals_kb_order"] is False and emb["ids_match"] is True and emb["shape"] == [698, 3072]
