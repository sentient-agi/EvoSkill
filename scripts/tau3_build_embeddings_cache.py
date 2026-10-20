#!/usr/bin/env python3
"""Compute the tau3 knowledge base's document embeddings ONCE into <data root>/.embeddings_cache and list them in the
data root's MANIFEST.sha256 (tau3 retrieval = tau2's "alltools", the dense half asked through OpenRouter for
openai/text-embedding-3-large; about $0.02 of embeddings).

Runs under the tau3-bench venv (third_party/tau3-bench/.venv/bin/python, tau2 1.0.1):

  tau3_build_embeddings_cache.py --root R          compute (refuses when the cache entry already exists), verify,
                                                   rewrite MANIFEST.sha256 (the old one backed up first, next to R:
                                                   <R's parent>/<R's name>_MANIFEST.sha256.before_<stamp>)
  tau3_build_embeddings_cache.py --root R --check  verify only: the entry loads read-only and matches the KB

The vectors come from tau2's OWN indexing path (create_embedding_retrieval_pipeline -> embedding_indexer -> tau2's
EmbeddingsCache.put), with tau2's "openai" embedder swapped for scripts/tau3_cc/tau3_retrieval.py's OpenRouter
transport, so the cache file, its key and its document order are exactly what the bridge and the grader look up. The
OpenRouter key is read in-process (scripts/tau3_cc/openrouter_key.py: OPENROUTER_API_KEY from the environment, else
the repository's .env; never printed, never in argv or on disk; removed from the environment before tau2 loads). One
KB_search_dense-style query then runs against the read-only cache as an end-to-end check (one more request).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "scripts" / "tau3_cc")]
import openrouter_key as ORK  # noqa: E402
import tau3_retrieval as TR  # noqa: E402

MANIFEST = "MANIFEST.sha256"
PROBE_QUERY = "How do I dispute a transaction on my credit card?"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def credits(key: str) -> dict:
    """OpenRouter's account totals (numbers only)."""
    import httpx
    try:
        r = httpx.get("https://openrouter.ai/api/v1/credits", headers={"Authorization": f"Bearer {key}"}, timeout=30)
        d = (r.json() or {}).get("data") or {}
        return {"total_credits": d.get("total_credits"), "total_usage": d.get("total_usage")}
    except Exception as e:
        return {"error": type(e).__name__}


def rewrite_manifest(root: Path) -> dict:
    """MANIFEST.sha256 = its git rows (everything outside .embeddings_cache, unchanged) + one row per cache file,
    sorted by relative path (tau3_make_data_root.parse_manifest's order)."""
    mf = root / MANIFEST
    old = mf.read_text(encoding="utf-8")
    rows = {}
    for line in old.splitlines():
        h, rel = line.split("  ", 1)
        if not rel.startswith(TR.CACHE_DIRNAME + "/"):
            rows[rel] = h
    cache = TR.cache_files(root)
    rows.update(cache)
    text = "".join(f"{rows[k]}  {k}\n" for k in sorted(rows))
    backup_dir = root.parent                # outside the root: a file inside it would be unlisted
    bk = backup_dir / f"{root.name}_MANIFEST.sha256.before_{time.strftime('%Y%m%dT%H%M%S')}"
    if not any(backup_dir.glob(f"{root.name}_MANIFEST.sha256.before_*")):
        shutil.copy2(mf, bk)
    tmp = mf.with_name(MANIFEST + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.chmod(tmp, 0o644)
    os.replace(tmp, mf)
    return {"rows": len(rows), "cache_rows": len(cache), "manifest_sha256": sha256(mf), "backup": str(bk)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--root", type=Path, required=True, help="the data root (scripts/tau3_make_data_root.py --dest)")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    root = a.root.resolve()
    key = None
    if not a.check:
        try:
            key = ORK.openrouter_key()      # the environment, else the repository's .env; never printed
        except ORK.KeyMissing as e:
            sys.exit(f"REFUSED: {e}")
    TR.env_without_keys()
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["TAU2_DATA_DIR"] = str(root)
    from tau2.utils.utils import DATA_DIR
    if Path(DATA_DIR).resolve() != root:
        sys.exit(f"tau2 data root {DATA_DIR} != {root}")
    from tau2.domains.banking_knowledge import retrieval as R
    from tau2.domains.banking_knowledge.environment import get_knowledge_base

    kb = get_knowledge_base()
    docs = R.get_or_create_docs(kb)
    out = {"root": str(root), "documents": len(docs), "model": TR.OR_MODEL}
    if not a.check:
        if TR.cache_dir(root).is_dir() and any(TR.cache_dir(root).glob("*.pkl")):
            sys.exit(f"REFUSED: {TR.cache_dir(root)} already holds a cache entry (use --check)")
        out["credits_before"] = credits(key)
        TR.install(root, key=key, harden_shell=False, readonly_cache=False)
        t0 = time.time()
        R.create_embedding_retrieval_pipeline(kb, TR.EMBEDDER_TYPE, {"model": TR.TAU2_MODEL}, top_k=10)
        out["embed_s"] = round(time.time() - t0, 1)
    # verify read-only: the dense pipeline exactly as build_tools makes it, then one query
    TR.install(root, key=key, harden_shell=False, readonly_cache=True)
    spec = R.RETRIEVAL_VARIANTS[TR.VARIANT].kb_search_dense
    from tau2.knowledge import embeddings_cache as EC
    emb, ids = EC.get_embeddings_cache().get([{"id": d["id"], "text": d["text"]} for d in docs], TR.EMBEDDER_TYPE,
                                             {"model": TR.TAU2_MODEL})
    out["cached"] = {"shape": list(emb.shape), "dtype": str(emb.dtype), "ids_match": ids == [d["id"] for d in docs]}
    pipe = R._create_kb_pipeline(spec, kb)
    if key:
        hits = pipe.retrieve(PROBE_QUERY, top_k=3)
        out["probe_query"] = {"query": PROBE_QUERY, "top3": [[h[0], round(float(h[1]), 4)] for h in hits]}
        out["credits_after"] = credits(key)
    if not a.check:
        out["manifest"] = rewrite_manifest(root)
    out["cache_files"] = TR.cache_files(root)
    print("EMBCACHE " + json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
