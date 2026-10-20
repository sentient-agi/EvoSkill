"""tau3 banking_knowledge grader: grade(task, snapshot) -> score in [0, 1].

Written by the user of the run, who implements the grading and hands it to the benchmark adapter with the user
instruction. A run's setup/benchmark_grader.py is a byte copy of this file. It is tau2's own scoring of one
conversation, unchanged (tau2 1.0.1, tau3-bench commit 363133a, retrieval variant alltools), as one function:

  - task: the plug-in's Task. Read: answer_files[0] (the task's tau2 file,
    <root>/tau2/domains/<domain>/tasks/task_NNN.json), grading["domain"], grading["retrieval_variant"].
  - snapshot: the solver's working directory when it exited. After the conversation the rollout harness writes
    tau2/simulation.json there: tau2's SimulationRun (every message with its tool-call ids, requestor, content and error
    flag, and the termination reason). The harness writes it for every rollout, a conversation that never started
    included, so a missing file is a grader failure, never the solver's.
  - The data root is answer_files[0].parents[4] (tasks/, <domain>/, domains/, tau2/, then the root). It becomes
    TAU2_DATA_DIR before tau2 is imported: tau2 reads it once, at import (tau2/utils/utils.py:18-28), and the banking
    domain reads <root>/tau2/domains/banking_knowledge (domains/banking_knowledge/utils.py:38-43). So one file grades on
    the Mac and on the pool machine, wherever the frozen Task's answer files live.
  - PYTHON_DOTENV_DISABLED=1 (tau2 calls load_dotenv() at import, which would read any .env above the venv) and
    LITELLM_LOCAL_MODEL_COST_MAP=True (litellm must not fetch its price map) are set before the import too.
  - The tau2 Task is the answer file, read as the domain's own loader reads it (Task.model_validate on the JSON,
    domains/banking_knowledge/environment.py:113-117). A conversation recorded for another task raises.
  - evaluate_simulation(EvaluationType.ALL, HALF_DUPLEX, solo_mode=False, strict_replay=True) with the environment built
    as the rollout built it: env_kwargs = tau2's _build_env_kwargs(TextRunConfig(domain, retrieval_config=variant), task),
    i.e. the task's retrieval variant and, for banking, the read-log allowlist taken from the task's expected actions
    (tau2/runner/build.py:310-353). The variant is passed explicitly because a banking config without one silently
    becomes 'alltools' (tau2/data_model/simulation.py:467-472).
  - tau2 replays the conversation's state-changing tool calls on a fresh environment, compares the databases with the
    ones the task's expected actions produce, and checks the expected actions, as the task's reward basis says; the
    score is the product, 0 or 1. A conversation that did not end with the customer or the agent stopping scores 0
    (tau2/evaluator/evaluator.py:119-128). Reads are never replayed (environment.py:388: a tool whose mutates_state is
    False is skipped), so a conversation's KB_search_bm25, KB_search_dense and shell calls are never re-run here.
  - Anything that goes wrong raises, which the loop records as grader_failed, never as a number: a missing
    simulation.json, an answer file outside the tau2 layout, a tau2 already imported with another data root, a
    conversation for another task, a replayed tool output that differs from the recorded one (strict replay,
    tau2/environment/environment.py:399-403), a document-embeddings cache entry that is absent or stale, an embedding
    request, or anything else tau2 raises.

RETRIEVAL alltools (tau2's own variant, unchanged: KB_search_bm25, KB_search_dense over OpenAI
text-embedding-3-large, and a read-only `shell` over the knowledge base through Anthropic's sandbox-runtime). Every
environment tau2 builds for it indexes the knowledge base densely and builds a shell sandbox, so before any is built
grade() changes three things in its own process (_offline):
  1. tau2's global embeddings cache (tau2.knowledge.embeddings_cache._global_cache; by default a cwd-relative
     data/.embeddings_cache that tau2 creates, fills and prunes) is <root>/.embeddings_cache, READ-ONLY: the document
     vectors the run's precompute stored there (listed in the data root's MANIFEST.sha256) are read; an entry that is
     absent or whose document ids differ from the knowledge base's raises. Nothing is ever written or deleted there.
  2. tau2's "openai" embedder refuses every request (an embedding request here would be a bug: the documents come
     from the cache and replay never re-runs a KB_search_dense query). It replaces OpenAIEmbedder in
     tau2.knowledge.embedders, its openai_embedder module and the EMBEDDER_REGISTRY of embedding_indexer and
     embedding_encoder, and it never builds an OpenAI client, so no key is ever asked for.
  3. The knowledge-base shell sandboxes tau2 builds during a grade() call (one temp dir per environment, holding the
     exported documents; grading never runs a command in one) are removed when the call returns, and any left at
     process exit. A sandbox the caller builds outside grade() is never touched. A second copy of this module in the
     same process (loaded under another name) tracks and removes its own.
A variant with neither a dense index nor a shell (bm25_grep) never reaches the cache, the embedder or srt, and grades
as before.

Runs in the grader's setup: a tau2 1.0.1 venv (CPython 3.12.13, tau3-bench at 363133a with the knowledge extra), the
data root the Task's answer files name, with its .embeddings_cache, and, for alltools, srt (sandbox-runtime 0.0.23)
and rg on PATH (and bwrap and socat on Linux): tau2's SandboxManager checks for them when it builds the sandbox and
raises without them. No network and no model key: the embeddings come from the cache, and EvaluationType.ALL calls
tau2's LLM judge only for a task whose reward basis holds NL_ASSERTION (evaluator.py:215-220), and the run's split has
none (banking task_102, the only one, is left out). Standard library at import time; tau2 is loaded inside grade().
"""
from __future__ import annotations

import atexit
import os
import pickle
from pathlib import Path

BUNDLE = Path("tau2") / "simulation.json"    # written into the solver's working directory by the rollout harness
CACHE_DIRNAME = ".embeddings_cache"          # <root>/.embeddings_cache: the precomputed document vectors
_SANDBOXES: dict = {"grading": False, "live": []}   # the shell sandboxes tau2 built inside grade(), not yet removed


class OfflineError(RuntimeError):
    """The grader needed something it never has: an embedding request, or a cache entry that is absent or stale."""


def _data_root(answer: Path, domain: str) -> Path:
    """<root>/tau2/domains/<domain>/tasks/<file> -> <root>; anything else raises."""
    p = answer.parents
    if len(p) < 5 or (p[3].name, p[2].name, p[1].name, p[0].name) != ("tau2", "domains", domain, "tasks"):
        raise ValueError(f"{answer} is not <root>/tau2/domains/{domain}/tasks/<file>")
    return p[4]


def _offline(root: Path) -> None:
    """The three changes of the module doc, applied to the imported tau2 (again on every call: idempotent)."""
    from tau2.domains.banking_knowledge import retrieval as R
    from tau2.knowledge import embedders as E
    from tau2.knowledge import embeddings_cache as EC
    from tau2.knowledge.document_preprocessors import embedding_indexer as EI
    from tau2.knowledge.embedders import openai_embedder as OE
    from tau2.knowledge.embedders.base import BaseEmbedder
    from tau2.knowledge.input_preprocessors import embedding_encoder as EE

    class OfflineEmbedder(BaseEmbedder):
        def __init__(self, model: str = "text-embedding-ada-002", api_key: str | None = None):
            self.model = model                  # tau2's OpenAIEmbedder signature; no client, so no key is read

        def embed(self, texts):
            raise OfflineError(f"an embedding request ({len(texts)} texts, {self.model}) in the grader, which is "
                               f"offline: the documents come from {CACHE_DIRNAME}, and replay never re-runs a read")

        def get_name(self) -> str:
            return f"openai_{self.model}"

    class ReadOnlyEmbeddingsCache(EC.EmbeddingsCache):
        def __init__(self, cache_dir: str):    # tau2's __init__ would create the dir
            self.cache_dir = Path(cache_dir)
            self.metadata_file = self.cache_dir / "metadata.json"
            self.metadata = {}

        def get(self, documents, embedder_type, embedder_config=None):
            f = self._get_cache_file(self._get_cache_key(self._compute_document_hash(documents),
                                                         self._compute_embedder_hash(embedder_type, embedder_config)))
            if not f.is_file():
                raise OfflineError(f"no cached embeddings for {embedder_type} {embedder_config} over these "
                                   f"{len(documents)} documents: {f.name} is absent in {self.cache_dir}")
            with open(f, "rb") as fh:
                data = pickle.load(fh)
            # tau2 lists the documents in filesystem order (an unordered glob), so a cache written on the Mac lists
            # them in another order than Linux reads them; the key already proves the same texts, so the same ids in
            # another order are the same vectors re-ordered. Anything else raises.
            ids, cached = [d["id"] for d in documents], list(data.get("doc_ids") or [])
            if cached == ids:
                return data["embeddings"], cached
            pos = {d: k for k, d in enumerate(cached)}
            if len(pos) != len(cached) or len(cached) != len(ids) or set(pos) != set(ids):
                raise OfflineError(f"{f}: the cached document ids differ from the knowledge base's")
            import numpy as np
            return np.asarray(data["embeddings"])[[pos[d] for d in ids]], ids

        def _refuse(self, *a, **k):
            raise OfflineError(f"{self.cache_dir} is read-only in the grader")

        put = get_incremental = put_incremental = clear = _refuse

    E.OpenAIEmbedder = OE.OpenAIEmbedder = OfflineEmbedder   # embeddings_cache imports it from E at call time
    EI.EMBEDDER_REGISTRY["openai"] = EE.EMBEDDER_REGISTRY["openai"] = OfflineEmbedder
    EC._global_cache = ReadOnlyEmbeddingsCache(str(Path(root) / CACHE_DIRNAME))
    f = R._create_sandbox                       # wrapped once per copy of this module (a second copy chains on top)
    while f is not None and getattr(f, "_grader_tracks", None) is not _SANDBOXES:
        f = getattr(f, "_grader_inner", None)
    if f is None:
        made = R._create_sandbox

        def _create_sandbox(knowledge_base, spec):
            sb = made(knowledge_base, spec)
            if _SANDBOXES["grading"]:           # a sandbox the caller builds outside grade() stays the caller's
                _SANDBOXES["live"].append(sb)
            return sb

        _create_sandbox._grader_tracks = _SANDBOXES
        _create_sandbox._grader_inner = made
        R._create_sandbox = _create_sandbox


@atexit.register
def _remove_sandboxes() -> None:
    live = _SANDBOXES["live"]
    while live:
        sb = live.pop()
        try:
            sb.cleanup()
        except Exception:
            pass


def grade(task, snapshot: Path) -> float:
    domain = task.grading["domain"]
    variant = task.grading["retrieval_variant"]
    answer = Path(task.answer_files[0])
    root = _data_root(answer, domain)
    bundle = Path(snapshot) / BUNDLE
    if not bundle.is_file():
        raise FileNotFoundError(f"{bundle} is missing (the harness writes it for every rollout)")
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["TAU2_DATA_DIR"] = str(root)
    from tau2.utils.utils import DATA_DIR       # the first tau2 import reads TAU2_DATA_DIR, once per process

    if Path(DATA_DIR).resolve() != root.resolve():
        raise RuntimeError(f"tau2 was imported with data root {DATA_DIR}, not {root}")
    _offline(root)
    from tau2.data_model.simulation import SimulationRun, TextRunConfig
    from tau2.data_model.tasks import Task as TauTask
    from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
    from tau2.orchestrator.modes import CommunicationMode
    from tau2.runner.build import _build_env_kwargs

    t = TauTask.model_validate_json(answer.read_text(encoding="utf-8"))
    sim = SimulationRun.model_validate_json(bundle.read_text(encoding="utf-8"))
    if sim.task_id != t.id:
        raise ValueError(f"the conversation is for {sim.task_id!r}, not {t.id!r}")
    env_kwargs = _build_env_kwargs(TextRunConfig(domain=domain, retrieval_config=variant), t)
    _SANDBOXES["grading"] = True
    try:
        ri = evaluate_simulation(simulation=sim, task=t, evaluation_type=EvaluationType.ALL, solo_mode=False,
                                 domain=domain, mode=CommunicationMode.HALF_DUPLEX, env_kwargs=env_kwargs,
                                 strict_replay=True)
    finally:
        _SANDBOXES["grading"] = False
        _remove_sandboxes()
    return float(ri.reward)
