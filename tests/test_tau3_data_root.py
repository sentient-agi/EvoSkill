"""The curated tau3 data root built by scripts/tau3_make_data_root.py: what it holds, and that tau2 runs from it.

One copy of the banking_knowledge data serves a tau3 run's rollouts, its grading and the adapter's /data view, so it
must be exactly the checkout's bytes at 363133a, minus the two stale aggregates that hold every validation and test
task. For retrieval alltools the root also holds the knowledge base's document embeddings, computed once by
scripts/tau3_build_embeddings_cache.py into .embeddings_cache/ (2 files) and listed in its MANIFEST.sha256: 819 rows,
817 from git + 2 derived.

  - the manifest is the one written after the precompute (sha 061cb180...), its 817 rows outside .embeddings_cache/
    match the checkout's bytes (sha256 of every file on both sides, and git's blob at the commit) and list exactly the
    kept files, its 2 cache rows are the precompute's files, and the root holds nothing else;
  - neither aggregate (tasks.json, tasks_voice.json) is in the root or its manifest;
  - with TAU2_DATA_DIR = the root, the domain's get_tasks() and the registry's task loader both load all 97 tasks,
    the alltools environment (17 tools: KB_search_bm25, KB_search_dense, shell and the bank's) builds from it with
    scripts/tau3_cc/tau3_retrieval.py installed without a key (the document vectors from the read-only cache) and
    leaves no shell sandbox behind, and the user simulator's guidelines load from it too. That runs in a child under
    the tau3 venv (the repo's interpreter has no tau2): PYTHON_DOTENV_DISABLED=1, LITELLM_LOCAL_MODEL_COST_MAP=True,
    an emptied environment (PATH = /usr/bin:/bin and the dirs of srt, rg and node, which tau2's shell sandbox checks
    for; the proxies on a dead port) and, on macOS, sandbox-exec with the network denied, .env and token files
    unreadable, and the repo and the root unwritable;
  - the builder: a fresh build gives the git rows of the manifest, a second build changes nothing, --check passes, a
    root with derived .embeddings_cache/ rows is "unchanged" and checked, and a root whose manifest differs outside
    .embeddings_cache/, whose files differ from it, that has an unlisted file, or that has no manifest is refused (rc 3)
    and left as it was.

The expected contents are written out here, not read from the builder, so the two state the contract separately.
No test names a task id, and nothing calls an LLM or a user simulator.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BUILDER = REPO / "scripts" / "tau3_make_data_root.py"
TAU3 = REPO / "third_party" / "tau3-bench"
TAU3_PY = TAU3 / ".venv" / "bin" / "python"
SRC = TAU3 / "data"
ROOT = Path(os.environ.get("EVOSKILL_TAU3_DATA_ROOT") or REPO / "third_party" / "tau3_data_363133a")
COMMIT = "363133ada1936491fb5bcec33cd62c3518a99f65"
MANIFEST = "MANIFEST.sha256"
N_TASKS = 97

N_ROWS, N_GIT_ROWS = 819, 817
MANIFEST_SHA256 = "061cb180384e813f207071805470a882966c6f4d164829d511487431396617e9"   # after the precompute
CACHE_DIR = ".embeddings_cache"
CACHE_ROWS = {                                            # the precompute's files (698 x 3072 float64 vectors)
    f"{CACHE_DIR}/62eca72522cacac783980da07ba707af2dfa2df0aec9def44ea0fcb260127fc2_edd03519131f2b849e5b90cd0e673657.pkl":
        "b3c69d7afea9441b4e1e61c06f5a132ac6fbf46031c210623824db37b436b046",
    f"{CACHE_DIR}/metadata.json": "e4392566a6fc664722fbe5858cc27a03697e08398b4cb2e644c014a6d0216775",
}
TR_DIR = REPO / "scripts" / "tau3_cc"
DEAD = "http://127.0.0.1:9"

DOMAIN = "tau2/domains/banking_knowledge"
KEPT_FILES = (f"{DOMAIN}/db.json",)
KEPT_DIRS = (f"{DOMAIN}/documents", f"{DOMAIN}/prompts", f"{DOMAIN}/tasks", "tau2/user_simulator")
AGGREGATES = ("tasks.json", "tasks_voice.json")          # banking_knowledge/<name> in the checkout; never copied
SANDBOX_EXEC = Path("/usr/bin/sandbox-exec")

CHILD = r"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import tau3_retrieval as TR
from tau2.utils.utils import DATA_DIR
rec = TR.install(Path(DATA_DIR), key=None)               # the read-only cache, no key, the hardened shell
from tau2.domains.banking_knowledge import utils as bu
from tau2.domains.banking_knowledge.environment import get_environment, get_tasks
from tau2.registry import registry
from tau2.user.user_simulator import get_global_user_sim_guidelines

tasks = get_tasks()
by_registry = registry.get_tasks_loader("banking_knowledge")()
env = get_environment(retrieval_variant="alltools")
print("RESULT " + json.dumps({
    "data_dir": str(DATA_DIR),
    "paths": [str(p) for p in (bu.KNOWLEDGE_DB_PATH, bu.KNOWLEDGE_DOCUMENTS_DIR, bu.KNOWLEDGE_TASK_SET_PATH)],
    "ids": [t.id for t in tasks],
    "registry_ids": [t.id for t in by_registry],
    "tools": sorted(t.name for t in env.get_tools()),
    "policy_chars": len(env.policy),
    "guidelines": [get_global_user_sim_guidelines(False), get_global_user_sim_guidelines(True)],
    "cache_files": rec["cache"]["files"],
    "sandboxes": [str(sb.sandbox_dir) for sb in TR._STATE["sandboxes"]],
}))
"""


# ───────────────────────────────────────────────────────────────────────────── helpers
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _files(top: Path) -> set[str]:
    """Relative posix paths of every file under top; a symlink or any other kind of entry fails the test."""
    out = set()
    for p in top.rglob("*"):
        assert not p.is_symlink() and (p.is_file() or p.is_dir()), f"{p} is not a plain file or dir"
        if p.is_file():
            out.add(p.relative_to(top).as_posix())
    return out


def _manifest(root: Path) -> dict[str, str]:
    lines = (root / MANIFEST).read_text(encoding="utf-8").splitlines()
    rows = [line.split("  ", 1) for line in lines]
    assert all(len(r) == 2 and len(r[0]) == 64 for r in rows), "not sha256sum lines"
    rels = [r[1] for r in rows]
    assert rels == sorted(rels) and len(set(rels)) == len(rels), "manifest paths are not sorted and unique"
    return {rel: sha for sha, rel in rows}


def _srt_path():
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


def _git_rows(rows: dict) -> dict:
    return {rel: sha for rel, sha in rows.items() if not rel.startswith(CACHE_DIR + "/")}


def _write_manifest(d: Path, rows: dict) -> None:
    (d / MANIFEST).write_text("".join(f"{rows[rel]}  {rel}\n" for rel in sorted(rows)), encoding="utf-8")


def _expected_from_checkout() -> set[str]:
    out = {f for f in KEPT_FILES if (SRC / f).is_file()}
    for d in KEPT_DIRS:
        out |= {f"{d}/{rel}" for rel in _files(SRC / d)}
    return out


def _build(dest: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(BUILDER), "--dest", str(dest), *extra],
                          capture_output=True, text=True, timeout=300)


def _snapshot(top: Path) -> dict[str, str]:
    return {rel: _sha256(top / rel) for rel in _files(top)}


@pytest.fixture
def root() -> Path:
    if not (ROOT / MANIFEST).is_file():
        pytest.skip(f"{ROOT} not built on this machine (run scripts/tau3_make_data_root.py once)")
    return ROOT


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    """A root built by the builder into a temporary dir (the checkout is only read)."""
    if not SRC.is_dir():
        pytest.skip(f"{SRC} absent")
    dest = tmp_path_factory.mktemp("tau3_root") / "root"
    r = _build(dest)
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("created: "), r.stdout
    return dest


# ───────────────────────────────────────────────────────────────────────────── the root's contents
def test_manifest_matches_the_checkouts_bytes(root):
    if not SRC.is_dir():
        pytest.skip(f"{SRC} absent")
    rows = _manifest(root)
    assert _sha256(root / MANIFEST) == MANIFEST_SHA256
    assert len(rows) == N_ROWS and len(_git_rows(rows)) == N_GIT_ROWS
    assert set(_git_rows(rows)) == _expected_from_checkout(), "the manifest does not list exactly the kept files"
    assert {rel: rows[rel] for rel in set(rows) - set(_git_rows(rows))} == CACHE_ROWS
    assert _files(root) - {MANIFEST} == set(rows), "the root holds a file its manifest does not list, or lacks one"
    for rel, sha in rows.items():
        assert _sha256(root / rel) == sha, f"{rel}: the root's bytes differ from its manifest"
        if rel not in CACHE_ROWS:
            assert _sha256(SRC / rel) == sha, f"{rel}: the checkout's bytes differ from the manifest"
    assert not (SRC / CACHE_DIR).exists() and not (SRC.parent / CACHE_DIR).exists()


def test_the_cache_rows_name_the_alltools_embedder_and_the_retrieval_modules_cache_dir(root):
    """The .pkl's name is tau2's cache key: <document hash>_<md5 of {"type": "openai", "config": {"model":
    "text-embedding-3-large"}}>, the embedder tau2's alltools variant names; metadata.json records the same."""
    sys.path.insert(0, str(TR_DIR))
    try:
        import tau3_retrieval as TR
    finally:
        sys.path.remove(str(TR_DIR))
    assert TR.CACHE_DIRNAME == CACHE_DIR and TR.cache_files(root) == CACHE_ROWS
    pkl = next(rel for rel in CACHE_ROWS if rel.endswith(".pkl"))
    doc_hash, emb_hash = Path(pkl).stem.split("_")
    spec = json.dumps({"type": TR.EMBEDDER_TYPE, "config": {"model": TR.TAU2_MODEL}}, sort_keys=True)
    assert hashlib.md5(spec.encode()).hexdigest() == emb_hash
    meta = json.loads((root / CACHE_DIR / "metadata.json").read_text())
    assert meta == {Path(pkl).stem: {**meta[Path(pkl).stem], "embedder_type": "openai", "num_documents": 698,
                                     "embedder_config": {"model": "text-embedding-3-large"},
                                     "doc_hash": doc_hash, "embedder_hash": emb_hash}}


def test_manifest_is_git_commit_363133a(root):
    """Every listed file is git's blob at 363133a (read-only git, no index lock), and every file the commit
    tracks under the kept paths is listed, the aggregates aside."""
    if shutil.which("git") is None or not (TAU3 / ".git").exists():
        pytest.skip("no git checkout of tau3-bench")
    out = subprocess.run(["git", "--no-optional-locks", "-C", str(TAU3), "ls-tree", "-r", "-z", "--full-tree",
                          COMMIT, "--", *(f"data/{p}" for p in KEPT_FILES + KEPT_DIRS)],
                         capture_output=True, check=True).stdout
    tree = {}
    for rec in filter(None, out.split(b"\0")):
        meta, path = rec.split(b"\t", 1)
        tree[path.decode().removeprefix("data/")] = meta.decode().split()[2]
    for name in AGGREGATES:
        tree.pop(f"{DOMAIN}/{name}", None)
    rows = _git_rows(_manifest(root))
    assert set(rows) == set(tree) and len(rows) == N_GIT_ROWS
    assert not any(p.startswith(CACHE_DIR) or f"/{CACHE_DIR}" in p for p in tree)     # derived: never in git
    for rel in rows:
        data = (root / rel).read_bytes()
        assert hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == tree[rel], rel


def test_neither_aggregate_is_present(root):
    rows = _manifest(root)
    names = {Path(rel).name for rel in _files(root)} | {Path(rel).name for rel in rows}
    assert not names & set(AGGREGATES)
    assert not (root / DOMAIN / "tasks.json").exists() and not (root / DOMAIN / "tasks_voice.json").exists()
    kept = KEPT_FILES + tuple(d + "/" for d in KEPT_DIRS)
    assert all(rel in KEPT_FILES or rel.startswith(kept) for rel in _git_rows(rows)), "a file outside the kept paths"
    assert set(rows) - set(_git_rows(rows)) == set(CACHE_ROWS)
    tasks = sorted((root / DOMAIN / "tasks").iterdir())
    assert len(tasks) == N_TASKS and all(p.name.startswith("task_") and p.suffix == ".json" for p in tasks)
    assert [json.loads(p.read_text())["id"] for p in tasks] == [p.stem for p in tasks]


# ───────────────────────────────────────────────────────────────────────────── tau2 runs from the root
def test_tau2_loads_all_97_tasks_and_the_alltools_environment_from_the_root(root, tmp_path):
    if not TAU3_PY.exists():
        pytest.skip(f"{TAU3_PY} absent (the tau3 venv)")
    if _srt_path() is None:
        pytest.skip("srt, rg or node is absent (tau2 builds the alltools shell sandbox with srt)")
    cwd, home, tmpd = tmp_path / "cwd", tmp_path / "home", tmp_path / "tmp"
    for d in (cwd, home, tmpd):
        d.mkdir()
    env = ["env", "-i", f"PATH={_srt_path()}", f"HOME={home}", f"TMPDIR={tmpd}", "PYTHON_DOTENV_DISABLED=1",
           "LITELLM_LOCAL_MODEL_COST_MAP=True", f"TAU2_DATA_DIR={root}", f"HTTPS_PROXY={DEAD}",
           f"HTTP_PROXY={DEAD}", f"ALL_PROXY={DEAD}"]
    argv = env + [str(TAU3_PY), "-I", "-B", "-c", CHILD, str(TR_DIR)]
    if sys.platform == "darwin" and SANDBOX_EXEC.exists():
        q = json.dumps                                    # SBPL string literals quote these paths as JSON does
        profile = ("(version 1)\n(allow default)\n(deny network*)\n"
                   '(deny file-read* (regex #"/\\.env$") (regex #"oauth_token"))\n'
                   f"(deny file-write* (subpath {q(os.path.realpath(REPO))}))\n"
                   f"(deny file-write* (subpath {q(os.path.realpath(root))}))\n")
        argv = [str(SANDBOX_EXEC), "-p", profile] + argv
    r = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-4000:]
    assert "Failed to load" not in r.stdout, r.stdout[-4000:]     # get_tasks prints, and skips, a bad file
    lines = [line for line in r.stdout.splitlines() if line.startswith("RESULT ")]
    assert len(lines) == 1, r.stdout[-4000:]
    got = json.loads(lines[0][len("RESULT "):])

    assert got["data_dir"] == str(root)
    assert all(p.startswith(str(root) + "/") for p in got["paths"]), got["paths"]
    stems = sorted(p.stem for p in (root / DOMAIN / "tasks").glob("task_*.json"))
    assert len(got["ids"]) == N_TASKS and sorted(got["ids"]) == stems
    assert sorted(got["registry_ids"]) == stems
    assert len(got["tools"]) == 17 and {"KB_search_bm25", "KB_search_dense", "shell"} <= set(got["tools"])
    assert not {"KB_search", "grep"} & set(got["tools"]) and got["policy_chars"] > 0
    assert got["guidelines"] == [(root / "tau2/user_simulator" / n).read_text()
                                 for n in ("simulation_guidelines.md", "simulation_guidelines_tools.md")]
    assert got["cache_files"] == CACHE_ROWS
    assert len(got["sandboxes"]) == 1 and Path(got["sandboxes"][0]).parent == tmpd
    assert list(tmpd.iterdir()) == [], "the shell sandbox was not removed at exit"
    assert not list(cwd.iterdir()), "the child wrote into its working directory (tau2's default cache?)"


# ───────────────────────────────────────────────────────────────────────────── the builder
def test_build_is_deterministic_and_idempotent(built):
    rows = _manifest(built)
    if (ROOT / MANIFEST).is_file():                       # the real root = these rows + the precompute's cache rows
        assert rows == _git_rows(_manifest(ROOT))
    assert set(rows) == _expected_from_checkout() and len(rows) == N_GIT_ROWS
    assert not (built / CACHE_DIR).exists()               # a fresh build has no derived rows
    before = {rel: (built / rel).stat().st_mtime_ns for rel in _files(built)}

    again = _build(built)
    assert again.returncode == 0 and again.stdout.startswith("unchanged: "), again.stdout + again.stderr
    assert {rel: (built / rel).stat().st_mtime_ns for rel in _files(built)} == before
    assert sorted(p.name for p in built.parent.iterdir()) == ["root"], "a temporary dir was left beside the root"

    checked = _build(built, "--check")
    assert checked.returncode == 0 and checked.stdout.startswith("checked: "), checked.stdout + checked.stderr


def _with_cache(built: Path, dest: Path) -> Path:
    """A copy of the built root plus a derived .embeddings_cache (two files) listed in its manifest, as the precompute
    leaves a root (scripts/tau3_build_embeddings_cache.rewrite_manifest)."""
    shutil.copytree(built, dest)
    c = dest / CACHE_DIR
    c.mkdir()
    (c / "aa_bb.pkl").write_bytes(b"\x80\x04vectors")
    (c / "metadata.json").write_text("{}\n")
    rows = _manifest(dest)
    rows.update({f"{CACHE_DIR}/{p.name}": _sha256(p) for p in c.iterdir()})
    _write_manifest(dest, rows)
    return dest


def test_a_root_with_derived_cache_rows_is_unchanged_and_checked(built, tmp_path):
    dest = _with_cache(built, tmp_path / "root")
    before = {rel: (dest / rel).stat().st_mtime_ns for rel in _files(dest)}
    snap = _snapshot(dest)
    again = _build(dest)
    assert again.returncode == 0 and again.stdout.startswith("unchanged: "), again.stdout + again.stderr
    assert f"{N_GIT_ROWS + 2} data files" in again.stdout and f"{N_GIT_ROWS} from git, 2 under {CACHE_DIR}/" in \
        again.stdout, again.stdout
    checked = _build(dest, "--check")
    assert checked.returncode == 0 and checked.stdout.startswith("checked: "), checked.stdout + checked.stderr
    assert f"2 under {CACHE_DIR}/" in checked.stdout
    assert _snapshot(dest) == snap and {rel: (dest / rel).stat().st_mtime_ns for rel in _files(dest)} == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["root"]


def test_check_on_the_real_root_reports_its_cache_rows(root):
    if not SRC.is_dir():
        pytest.skip(f"{SRC} absent")
    before = _snapshot(root)
    r = _build(root, "--check")
    assert r.returncode == 0 and r.stdout.startswith(f"checked: {root} "), r.stdout + r.stderr
    assert f"({N_ROWS} data files + {MANIFEST}: {N_GIT_ROWS} from git, 2 under {CACHE_DIR}/;" in r.stdout
    assert f"manifest sha256 {MANIFEST_SHA256}" in r.stdout
    assert _snapshot(root) == before


def _change_a_cache_file(d: Path):
    p = d / CACHE_DIR / "aa_bb.pkl"
    p.write_bytes(p.read_bytes() + b"x")


def _unlisted_cache_file(d: Path):
    (d / CACHE_DIR / "cc_dd.pkl").write_bytes(b"more")


def _listed_extra_file_outside_the_cache(d: Path):
    (d / "notes.txt").write_text("x")
    rows = _manifest(d)
    rows["notes.txt"] = _sha256(d / "notes.txt")
    _write_manifest(d, rows)


def _listed_nested_cache_dir(d: Path):
    p = d / "tau2" / CACHE_DIR / "x.pkl"                  # only <root>/.embeddings_cache/ is derived
    p.parent.mkdir()
    p.write_bytes(b"x")
    rows = _manifest(d)
    rows[f"tau2/{CACHE_DIR}/x.pkl"] = _sha256(p)
    _write_manifest(d, rows)


def _a_git_file_changed_and_relisted(d: Path):
    p = d / DOMAIN / "db.json"
    p.write_bytes(p.read_bytes() + b"\n")
    rows = _manifest(d)
    rows[f"{DOMAIN}/db.json"] = _sha256(p)                 # self-consistent, but not the checkout's bytes
    _write_manifest(d, rows)


def _a_git_row_and_its_file_dropped(d: Path):
    """A git row dropped from the manifest of a root with cache rows (its file deleted too, so the root is
    self-consistent): the rows outside .embeddings_cache/ are no longer the checkout's."""
    rows = _manifest(d)
    rel = next(r for r in sorted(rows) if not r.startswith(CACHE_DIR + "/"))
    (d / rel).unlink()
    del rows[rel]
    _write_manifest(d, rows)


@pytest.mark.parametrize("damage", [_change_a_cache_file, _unlisted_cache_file, _listed_extra_file_outside_the_cache,
                                    _listed_nested_cache_dir, _a_git_file_changed_and_relisted, _a_git_row_and_its_file_dropped],
                         ids=lambda f: f.__name__.strip("_"))
def test_build_refuses_a_cache_root_that_is_not_the_checkout_plus_its_cache(built, tmp_path, damage):
    dest = _with_cache(built, tmp_path / "root")
    damage(dest)
    before = _snapshot(dest)
    for extra in ((), ("--check",)):
        r = _build(dest, *extra)
        assert r.returncode == 3 and "REFUSED" in r.stderr, (extra, r.stdout, r.stderr)
        assert _snapshot(dest) == before, "a refused build changed the root"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["root"]


def _edit_manifest(d: Path):
    mf = d / MANIFEST
    text = mf.read_text(encoding="utf-8")
    mf.write_text(("0" if text[0] != "0" else "1") + text[1:], encoding="utf-8")


def _drop_manifest_line(d: Path):
    mf = d / MANIFEST
    mf.write_text("".join(mf.read_text(encoding="utf-8").splitlines(keepends=True)[1:]), encoding="utf-8")



def _change_a_file(d: Path):
    p = d / DOMAIN / "db.json"
    p.write_bytes(p.read_bytes() + b"\n")


def _add_a_file(d: Path):
    (d / DOMAIN / "tasks.json").write_text("[]")


def _no_manifest(d: Path):
    (d / MANIFEST).unlink()


@pytest.mark.parametrize("damage", [_edit_manifest, _drop_manifest_line, _change_a_file, _add_a_file, _no_manifest])
def test_build_refuses_a_root_it_did_not_make(built, tmp_path, damage):
    dest = tmp_path / "root"
    shutil.copytree(built, dest)
    damage(dest)
    before = _snapshot(dest)
    for extra in ((), ("--check",)):
        r = _build(dest, *extra)
        assert r.returncode == 3 and "REFUSED" in r.stderr, (extra, r.stdout, r.stderr)
        assert _snapshot(dest) == before, "a refused build changed the root"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["root"]


def test_build_refuses_a_plain_file_or_a_foreign_dir(tmp_path):
    if not SRC.is_dir():
        pytest.skip(f"{SRC} absent")
    (tmp_path / "file").write_text("x")
    (tmp_path / "dir").mkdir()
    (tmp_path / "dir" / "note.txt").write_text("y")
    for name in ("file", "dir"):
        r = _build(tmp_path / name)
        assert r.returncode == 3 and "REFUSED" in r.stderr, r.stderr
    assert (tmp_path / "file").read_text() == "x"
    assert sorted(p.name for p in (tmp_path / "dir").iterdir()) == ["note.txt"]
