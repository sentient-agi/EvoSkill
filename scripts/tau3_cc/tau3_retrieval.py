"""tau3 ALLTOOLS retrieval (the leaderboard's retrieval), process-local.

The variant is tau2's own "alltools", UNCHANGED (retrieval.py:595: KB_search_bm25 + KB_search_dense over OpenAI
text-embedding-3-large + a read-only `shell` over the knowledge base through Anthropic's sandbox-runtime, policy
prompts/all_tools.md). Its name, prompt text ("uses **OpenAI API** with embedding model `text-embedding-3-large`"),
embedder type and params -- what tau2 records, hashes and shows the agent -- stay tau2's. install() changes three
things in THIS process only:

1. TRANSPORT. tau2's OpenAIEmbedder (embedder_type "openai") is replaced by OpenRouterTransportEmbedder, which asks
   OpenRouter for the same model ("openai/text-embedding-3-large") with the OpenRouter key we already carry for the
   customer, held in memory (never env, argv or disk). key=None (the grader, the offline tests) makes any embedding
   request raise: the documents come from the cache, and only an agent's KB_search_dense query needs the network.
2. CACHE. The 698 documents' embeddings are computed ONCE (scripts/tau3_build_embeddings_cache.py) into
   <data root>/.embeddings_cache in tau2's EmbeddingsCache format (key = tau2's document hash + the embedder hash of
   ("openai", {"model": "text-embedding-3-large"})), listed in the data root's MANIFEST.sha256. tau2's global cache
   (default cwd-relative data/.embeddings_cache) is pointed there READ-ONLY: a miss or a stale entry raises instead
   of re-embedding (or deleting) anything.
3. SHELL. Every KB shell sandbox's srt settings also deny reads of the data root (task files with the expected
   actions, the bank's db.json), /opt (bridge unit dirs, containers, the tau3 venv), /root, /home and /Users, so the
   agent's `shell` reads the exported knowledge base and nothing of the run. (tau2's own guard blocks "..", "~" and
   absolute paths, but `cd / && cat <relative path>` passes it; srt reads are deny-only.) A sandbox whose own
   knowledge-base dir would fall under a denied path (a TMPDIR under /opt, /home, /Users or the data root) is refused
   with RetrievalError: srt would hide the documents and every `shell` call would answer "(no output)" without an
   error. Each sandbox's temp dir is removed at process exit, hardened or not.
   Two more things make that deny list hold:
   - NO CAPABILITIES (Linux). The bridge runs as root, and bubblewrap started by uid 0 (not setuid) keeps the
     caller's capabilities unless told otherwise (srt 0.0.23 never passes --cap-drop or --unshare-user). With
     CAP_SYS_ADMIN the agent's shell could `umount` the tmpfs srt lays over each denied dir (or /proc, or remount /
     read-write) and read what it hides. So srt finds, first on its PATH, a private `bwrap` wrapper that runs the
     real one as `bwrap --cap-drop ALL ...` (bwrap_wrapper_text). And install() PROVES it on the host before any
     environment exists (probe_shell, Linux): in a hardened sandbox a marker file reads back, `grep ... /proc/self/
     status` shows CapPrm = CapEff = 0, and after the shell tries to `umount` every denied dir, listing the data
     root, /opt, /root, /home and /Users prints nothing. A failed probe raises RetrievalError (no rollout runs on a
     host where the hardening does not hold). On macOS srt uses sandbox-exec as the calling user: nothing to drop.
   - A MINIMAL ENVIRONMENT. tau2 starts srt with this process's whole environment, which the shell inherits; a
     hardened process gives srt only SHELL_ENV_KEEP (PATH, HOME, TMPDIR, locale, user name) instead.
   Both act through tau2's own subprocess.run(["srt", ...]) call (a stand-in for the subprocess module inside
   tau2.knowledge.sandbox_manager); nothing of tau2's command guard, settings or output handling changes.
   install(..., harden_shell=False) undoes the hardening an earlier install() of this process applied.

Call install(root, key=...) after tau2 was imported with TAU2_DATA_DIR = root and before any environment is built.
Standard library at import time; tau2, numpy and openai are imported inside install() / the embedder.
"""
from __future__ import annotations

import atexit
import datetime
import hashlib
import json
import os
import pickle
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

VARIANT = "alltools"                                   # tau2's name; what run.json / simulation.json record
EMBEDDER_TYPE = "openai"                               # tau2's spec for the variant (retrieval.py:595-599)
TAU2_MODEL = "text-embedding-3-large"
OR_MODEL = "openai/text-embedding-3-large"             # the same model, asked through OpenRouter
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
EMBED_DIM = 3072
CACHE_DIRNAME = ".embeddings_cache"                    # <data root>/.embeddings_cache
BATCH = 64                                             # documents per embeddings request
# 10 s per try, 3 tries with 1 s and 2 s waits, all failing = an infrastructure end: an embedding is one forward
# pass (a query measured 0.19-1.24 s, median 0.39 s, through OpenRouter Azure), so a dense-search outage costs the solver at most ~33 s of its 20 minutes, and then the bridge ends the rollout as
# infrastructure_error (re-run, never graded) instead of the solver taking a 0 for it.
ATTEMPTS = 3                                           # per request
BACKOFF_S = (1.0, 2.0)                                 # the waits before tries 2 and 3
REQUEST_TIMEOUT_S = 10.0
DENY_READ_FIXED = ("/opt", "/root", "/home", "/Users")  # + the data root itself (install adds it)
# OpenRouter serves this model from two providers, OpenAI and Azure (GET /api/v1/models/openai/text-embedding-3-large/
# endpoints). Through the OpenAI provider an OpenRouter account's own OpenAI key ("bring your own key") may be used, and
# OpenAI can reject it (401); Azure serves the same model on OpenRouter's credits. Documents and
# queries both go to Azure, so every vector the cosine search compares comes from one deployment.
PROVIDER = {"order": ["azure"], "allow_fallbacks": False}
BWRAP_CAP_DROP = ("--cap-drop", "ALL")                 # what srt's bwrap gets first (Linux; module doc, item 3)
SHELL_ENV_KEEP = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE", "USER", "LOGNAME")
PROBE_FILE = "TAU3_SHELL_PROBE.txt"                    # written into the probe's own sandbox only
PROBE_TEXT = "tau3 shell probe: the sandbox runs commands\n"
CAP_PROBE = "cd / && grep -E '^(Cap(Inh|Prm|Eff|Bnd|Amb)|NoNewPrivs):' proc/self/status"

_STATE: dict = {"key": None, "installed": None, "sandboxes": [], "deny_read": (), "bwrap_dir": None,
                "shell_probe": None, "dense_outage": None}


class RetrievalError(RuntimeError):
    """An alltools precondition failed (no cache entry, an offline embedding request, a changed variant)."""


def cache_dir(root) -> Path:
    return Path(root) / CACHE_DIRNAME


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ───────────────────────────────────────────────────────────────────────────── 1. transport
def embedder_class():
    """OpenRouterTransportEmbedder: tau2's OpenAIEmbedder signature (model, api_key), OpenRouter transport, the key
    from install() (api_key is ignored: tau2 never passes one for this variant)."""
    import numpy as np
    from tau2.knowledge.embedders.base import BaseEmbedder

    class OpenRouterTransportEmbedder(BaseEmbedder):
        def __init__(self, model: str = TAU2_MODEL, api_key: str | None = None):
            if model != TAU2_MODEL:
                raise RetrievalError(f"alltools asks for {model!r}; only {TAU2_MODEL!r} is carried through OpenRouter")
            self.model = model
            self.or_model = OR_MODEL
            self._client = None

        def _cli(self):
            key = _STATE["key"]
            if not key:
                raise RetrievalError("an embedding request in an offline process (no OpenRouter key): the document "
                                     f"embeddings must come from {CACHE_DIRNAME}, and only a solver's KB_search_dense "
                                     "query may reach OpenRouter")
            if self._client is None:
                from openai import OpenAI
                self._client = OpenAI(api_key=key, base_url=OPENROUTER_BASE, timeout=REQUEST_TIMEOUT_S, max_retries=0)
            return self._client

        def _one(self, texts: list) -> list:
            last = None
            for attempt in range(ATTEMPTS):
                try:
                    resp = self._cli().embeddings.create(input=texts, model=self.or_model, extra_body={"provider": PROVIDER})
                    data = sorted(resp.data, key=lambda d: d.index)
                    if len(data) != len(texts) or any(len(d.embedding) != EMBED_DIM for d in data):
                        raise RetrievalError(f"OpenRouter returned {len(data)} embeddings for {len(texts)} texts "
                                             f"(dims {sorted({len(d.embedding) for d in data})}, want {EMBED_DIM})")
                    return [d.embedding for d in data]
                except RetrievalError:
                    raise
                except Exception as e:          # transport / rate limit / 5xx: same request again
                    last = e
                    if attempt + 1 < ATTEMPTS:
                        time.sleep(BACKOFF_S[min(attempt, len(BACKOFF_S) - 1)])
            # the error's type and HTTP status only: an upstream body can quote (masked) provider keys, and this text
            # reaches the solver as the tool's error
            err = f"{type(last).__name__} (status {getattr(last, 'status_code', None)})"
            # the bridge reads this flag when tau2 hands back the failed call's result and ends the rollout as an
            # infrastructure end (re-run), never as the solver's 0
            _STATE["dense_outage"] = {"attempts": ATTEMPTS, "timeout_s": REQUEST_TIMEOUT_S, "error": err,
                                      "texts": len(texts),
                                      "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
            raise RetrievalError(f"OpenRouter {self.or_model} failed {ATTEMPTS}x: {err}")

        def embed(self, texts):
            if not texts:
                raise ValueError("No text to embed.")
            out = []
            for i in range(0, len(texts), BATCH):
                out.extend(self._one(list(texts[i:i + BATCH])))
            return np.array(out)

        def get_name(self) -> str:
            return f"openai_{self.model}"

    return OpenRouterTransportEmbedder


# ───────────────────────────────────────────────────────────────────────────── 2. cache
def in_kb_order(data: dict, ids: list, name: str, err=RuntimeError):
    """The cached embeddings in the knowledge base's CURRENT document order. tau2 loads the documents with an unordered
    Path.glob (domains/banking_knowledge/data_model.py:46), so the order is the filesystem's: the cache written on the
    Mac (APFS) lists the ids in another order than a Linux machine reads them (a cache written on one and read on the
    other otherwise has no usable entry). tau2's own get() then discards the entry. The cache key already proves the
    same documents with the same texts (tau2's document hash: sorted id:md5(text)), so the same set of ids in another
    order is the same vectors re-ordered -- exactly what embedding them in this order would give. Anything else raises."""
    import numpy as np
    cached = list(data.get("doc_ids") or [])
    if cached == ids:
        return data["embeddings"], cached
    pos = {d: k for k, d in enumerate(cached)}
    if len(pos) != len(cached) or len(cached) != len(ids) or set(pos) != set(ids):
        raise err(f"{name}: cached document ids differ from the knowledge base's (not only in order)")
    return np.asarray(data["embeddings"])[[pos[d] for d in ids]], list(ids)


def readonly_cache_class():
    from tau2.knowledge.embeddings_cache import EmbeddingsCache

    class ReadOnlyEmbeddingsCache(EmbeddingsCache):
        """tau2's EmbeddingsCache on an existing dir; get() raises on a miss or stale entry (never unlinks), put()
        raises. The incremental per-document API is refused too (tau2's pipelines do not use it)."""

        def __init__(self, cache_dir: str):
            d = Path(cache_dir)
            if not d.is_dir():
                raise RetrievalError(f"{d} is absent: run scripts/tau3_build_embeddings_cache.py")
            self.cache_dir = d
            self.metadata_file = d / "metadata.json"
            self.metadata = self._load_metadata()

        def get(self, documents, embedder_type, embedder_config=None):
            key = self._get_cache_key(self._compute_document_hash(documents),
                                      self._compute_embedder_hash(embedder_type, embedder_config))
            f = self._get_cache_file(key)
            if not f.is_file():
                raise RetrievalError(f"no cached embeddings for {embedder_type} {embedder_config} over these "
                                     f"{len(documents)} documents ({f.name} absent in {self.cache_dir})")
            with open(f, "rb") as fh:
                data = pickle.load(fh)
            return in_kb_order(data, [doc["id"] for doc in documents], f.name, RetrievalError)

        def put(self, *a, **k):
            raise RetrievalError(f"{self.cache_dir} is read-only here (a miss must never re-embed the knowledge base)")

        def get_incremental(self, *a, **k):
            raise RetrievalError("incremental embedding cache is not used by alltools")

        def put_incremental(self, *a, **k):
            raise RetrievalError(f"{self.cache_dir} is read-only here")

    return ReadOnlyEmbeddingsCache


def cache_files(root) -> dict:
    """{relative path under the data root: sha256} of the cache dir's files."""
    d = cache_dir(root)
    return {p.relative_to(Path(root)).as_posix(): _sha256(p) for p in sorted(d.rglob("*")) if p.is_file()}


# ───────────────────────────────────────────────────────────────────────────── 3. shell
def deny_read_paths(root) -> tuple:
    r = str(Path(root).resolve())
    return tuple(dict.fromkeys((r, str(Path(root)), *DENY_READ_FIXED)))


def _under(p: Path, d: str) -> bool:
    d = Path(os.path.expanduser(d))
    return any(p == q or p.is_relative_to(q) for q in {d, d.resolve()})


def _linux() -> bool:
    return sys.platform.startswith("linux")


def bwrap_wrapper_text(real_bwrap: str) -> str:
    """The `bwrap` srt finds first on its PATH on Linux: the real one, every capability dropped (BWRAP_CAP_DROP before
    srt's own arguments, which pass through verbatim)."""
    return f'#!/bin/sh\nexec {shlex.quote(real_bwrap)} {" ".join(BWRAP_CAP_DROP)} "$@"\n'


def _bwrap_wrapper_dir() -> str:
    """A private dir (mkdtemp: 0700, this process's user) holding only `bwrap` = bwrap_wrapper_text(the real bwrap on
    this process's PATH). Made once per process, removed at exit."""
    d = _STATE["bwrap_dir"]
    if d and (Path(d) / "bwrap").is_file():
        return d
    real = shutil.which("bwrap")
    if real is None:
        raise RetrievalError("bwrap is not on PATH: srt runs the KB shell inside it on Linux")
    d = tempfile.mkdtemp(prefix="tau3_bwrap_")
    w = Path(d) / "bwrap"
    w.write_text(bwrap_wrapper_text(os.path.realpath(real)))
    w.chmod(0o755)
    _STATE["bwrap_dir"] = d
    return d


def shell_env() -> dict:
    """The environment srt -- and through it the agent's `shell` -- gets in a hardened process: SHELL_ENV_KEEP from
    this process's environment and nothing else; on Linux PATH is led by the bwrap wrapper dir."""
    env = {k: os.environ[k] for k in SHELL_ENV_KEEP if os.environ.get(k)}
    path = os.environ.get("PATH") or os.defpath
    env["PATH"] = os.pathsep.join([_bwrap_wrapper_dir(), path]) if _linux() else path
    return env


class _SrtSubprocess:
    """Stands in for the subprocess module inside tau2.knowledge.sandbox_manager in a hardened process: its one
    subprocess.run(["srt", ...]) call (sandbox_manager.py:428-437, no env=) gets shell_env(); every other name is
    the real module's (subprocess.TimeoutExpired included)."""

    def __getattr__(self, name):
        return getattr(subprocess, name)

    @staticmethod
    def run(args, *a, **k):
        if isinstance(args, (list, tuple)) and args and args[0] == "srt" and k.get("env") is None:
            k["env"] = shell_env()
        return subprocess.run(args, *a, **k)


def _patch_sandbox(root) -> None:
    from tau2.knowledge import sandbox_manager as SM

    if _linux():
        _bwrap_wrapper_dir()            # a missing bwrap fails here, before any environment is built
    deny = deny_read_paths(root)
    _STATE["deny_read"] = deny
    cls = SM.SandboxManager
    if not getattr(cls, "_tau3_orig_settings", None):
        cls._tau3_orig_settings = cls._create_srt_settings

    def _create_srt_settings(self):
        cls._tau3_orig_settings(self)
        s = json.loads(Path(self.settings_path).read_text())
        fs = s.setdefault("filesystem", {})
        fs["denyRead"] = list(dict.fromkeys([*fs.get("denyRead", []), *_STATE["deny_read"]]))
        kb = Path(self.kb_dir).resolve()
        hit = [d for d in fs["denyRead"] if _under(kb, d)]
        if hit:                         # SandboxManager.__init__ made the dir; nothing else will remove it
            shutil.rmtree(self.sandbox_dir, ignore_errors=True)
            raise RetrievalError(f"the knowledge-base sandbox {kb} is under the denied path(s) {hit}: srt would hide "
                                 "the documents from `shell` (set TMPDIR outside them)")
        Path(self.settings_path).write_text(json.dumps(s, indent=2))

    cls._create_srt_settings = _create_srt_settings
    SM.subprocess = _SrtSubprocess()


def _unpatch_sandbox() -> None:
    """install(harden_shell=False): tau2's own settings and subprocess again (sandboxes already built keep theirs)."""
    from tau2.knowledge import sandbox_manager as SM

    cls = SM.SandboxManager
    orig = getattr(cls, "_tau3_orig_settings", None)
    if orig is not None:
        cls._create_srt_settings = orig
    if SM.subprocess is not subprocess:
        SM.subprocess = subprocess
    _STATE["deny_read"] = ()


def parse_caps(text: str) -> dict:
    """{"CapPrm": int, "CapEff": int, ..., "NoNewPrivs": int} from /proc/<pid>/status lines (capability sets in hex)."""
    out = {}
    for line in (text or "").splitlines():
        name, sep, val = line.partition(":")
        name, val = name.strip(), val.strip()
        if not sep or not val:
            continue
        try:
            if name.startswith("Cap"):
                out[name] = int(val, 16)
            elif name == "NoNewPrivs":
                out[name] = int(val)
        except ValueError:
            pass
    return out


def probe_shell(root) -> dict:
    """The hardening checked where it runs: a KB sandbox built in THIS process (tau2's SandboxManager, so hardened
    exactly as the agent's when install() hardened this process), and through tau2's own run_command (its guard,
    srt, its environment):
      commands_run            -- a marker file written into the sandbox reads back (the probe is not vacuous);
      denied_dirs_unreadable  -- after `umount` of every denied dir (outermost first, twice: stacked tmpfs), listing
                                 the data root, /opt, /root, /home and /Users prints nothing;
      no_capabilities (Linux) -- the shell's own process has CapPrm = CapEff = 0 (an exec there gains none).
    Returns {"platform", "marker", "deny_listing", ["caps"], "ok": {check: bool}, "passed"}; never raises for a
    failed check (install() does). Outputs are kept to 600 / 300 characters."""
    from tau2.knowledge.sandbox_manager import SandboxManager

    real = Path(os.path.realpath(root))

    def rel(p) -> str:
        return Path(p).relative_to("/").as_posix()

    denied = sorted({rel(d) for d in deny_read_paths(root)}, key=lambda s: (s.count("/"), s))
    targets = list(dict.fromkeys([rel(real), *(rel(d) for d in DENY_READ_FIXED)]))
    listing = (f"cd / && {' ; '.join('umount ' + shlex.quote(d) for d in denied * 2)} ; "
               f"{' ; '.join('ls -A ' + shlex.quote(t) for t in targets)}")
    sb = SandboxManager(allow_writes=False)
    try:
        (Path(sb.kb_dir) / PROBE_FILE).write_text(PROBE_TEXT)

        def run(cmd):
            rc, so, se = sb.run_command(cmd)
            return so or "", {"cmd": cmd, "rc": rc, "stdout": (so or "")[:600], "stderr": (se or "")[:300]}

        so, marker = run(f"cat {PROBE_FILE}")
        out = {"platform": sys.platform, "marker": marker}
        ok = {"commands_run": PROBE_TEXT.strip() in so}
        so, out["deny_listing"] = run(listing)
        ok["denied_dirs_unreadable"] = so.strip() == ""
        if _linux():
            so, out["caps"] = run(CAP_PROBE)
            caps = out["caps"]["parsed"] = parse_caps(so)
            ok["no_capabilities"] = caps.get("CapPrm") == 0 and caps.get("CapEff") == 0
    finally:
        sb.cleanup()
    out["ok"] = ok
    out["passed"] = all(ok.values())
    return out


def _require_shell_probe(root) -> dict:
    p = probe_shell(root)
    if not p["passed"]:
        bad = sorted(k for k, v in p["ok"].items() if not v)
        raise RetrievalError(f"the hardened KB shell failed its probe on this host ({', '.join(bad)}): "
                             f"{json.dumps(p)[:2000]}")
    return p


def _track_sandboxes() -> None:
    """Every sandbox tau2 builds (one per alltools environment) is recorded, hardened or not, so that its temp dir
    is removed at exit."""
    from tau2.domains.banking_knowledge import retrieval as R

    if getattr(R, "_tau3_orig_create_sandbox", None):
        return
    R._tau3_orig_create_sandbox = R._create_sandbox

    def _create_sandbox(knowledge_base, spec):
        sb = R._tau3_orig_create_sandbox(knowledge_base, spec)
        _STATE["sandboxes"].append(sb)
        return sb

    R._create_sandbox = _create_sandbox


@atexit.register
def _cleanup_sandboxes() -> None:
    for sb in _STATE["sandboxes"]:
        try:
            sb.cleanup()
        except Exception:
            pass
    _STATE["sandboxes"].clear()
    if _STATE["bwrap_dir"]:
        shutil.rmtree(_STATE["bwrap_dir"], ignore_errors=True)
        _STATE["bwrap_dir"] = None


# ───────────────────────────────────────────────────────────────────────────── install
def install(root, *, key: str | None = None, harden_shell: bool = True, readonly_cache: bool = True) -> dict:
    """Apply the three changes (module doc) to the already-imported tau2. Returns a record (no secret) for harness.json.
    readonly_cache=False is for the precompute only (tau2's own EmbeddingsCache on the same dir, writable)."""
    root = Path(root)
    from tau2.utils.utils import DATA_DIR
    if Path(DATA_DIR).resolve() != root.resolve():
        raise RetrievalError(f"tau2 was imported with data root {DATA_DIR}, not {root}")
    from tau2.domains.banking_knowledge import retrieval as R
    from tau2.knowledge import embedders as E
    from tau2.knowledge import embeddings_cache as EC
    from tau2.knowledge.document_preprocessors import embedding_indexer as EI
    from tau2.knowledge.embedders import openai_embedder as OE
    from tau2.knowledge.input_preprocessors import embedding_encoder as EE

    v = R.RETRIEVAL_VARIANTS.get(VARIANT)
    dense = getattr(v, "kb_search_dense", None)
    if (v is None or dense is None or v.kb_search_bm25 is None or v.shell is None
            or (dense.embedder_type, dense.embedder_model) != (EMBEDDER_TYPE, TAU2_MODEL) or v.shell.allow_writes):
        raise RetrievalError(f"tau2's {VARIANT!r} variant is not BM25 + dense {EMBEDDER_TYPE}/{TAU2_MODEL} + a "
                             f"read-only shell: {v!r}")
    _STATE["key"] = key or None
    _STATE["dense_outage"] = None                # one rollout per install; a new one starts clean
    cls = embedder_class()
    E.OpenAIEmbedder = cls          # embeddings_cache._compute_and_cache_embeddings imports it from here at call time
    OE.OpenAIEmbedder = cls
    EI.EMBEDDER_REGISTRY[EMBEDDER_TYPE] = cls
    EE.EMBEDDER_REGISTRY[EMBEDDER_TYPE] = cls
    d = cache_dir(root)
    if readonly_cache:
        EC._global_cache = readonly_cache_class()(str(d))
    else:
        EC._global_cache = EC.EmbeddingsCache(str(d))
    _track_sandboxes()
    if harden_shell:
        _patch_sandbox(root)
        # Linux: proven on this host before any environment is built (module doc, item 3); raises if it does not hold
        _STATE["shell_probe"] = _require_shell_probe(root) if _linux() else None
    else:
        _unpatch_sandbox()
        _STATE["shell_probe"] = None
    rec = {"variant": VARIANT, "embedder": {"type": EMBEDDER_TYPE, "model": TAU2_MODEL, "transport": "openrouter",
                                            "openrouter_model": OR_MODEL, "provider": PROVIDER,
                                            "key": bool(_STATE["key"])},
           "cache": {"dir": str(d), "readonly": readonly_cache,
                     "files": cache_files(root) if d.is_dir() else {}},
           "shell_deny_read": list(_STATE["deny_read"]) if harden_shell else None,
           "module_sha256": _sha256(Path(__file__))}
    _STATE["installed"] = rec
    return rec


def env_without_keys() -> None:
    """Drop the provider key names from this process's environment (the openai client reads OPENAI_API_KEY)."""
    for k in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "OR_API_KEY"):
        os.environ.pop(k, None)
