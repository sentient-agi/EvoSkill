#!/usr/bin/env python3
"""Which tau3 banking_knowledge retrieval variants can we actually run, and at what cost?

    PYTHONPATH=$PWD/third_party/tau3-bench/src \
      third_party/tau2-bench/.venv/bin/python scripts/tau3_variant_probe.py
    ... --construct     # additionally prove the local-only ones build (still no spend)

WHY THIS EXISTS. `banking_knowledge` ships **21** retrieval variants and the choice
among them is an EXPERIMENTAL DESIGN DECISION, not a config knob — it decides which
tools the agent has and therefore what the task even is. `no_knowledge` gives it no
KB at all; `full_kb` puts the whole corpus in the prompt; `golden_retrieval` hands it
the required documents, i.e. an oracle ceiling; `terminal_use` gives it a shell.
Comparing a skill leg run under one against a base leg run under another would
measure the variant, not the skill.

Nothing in this repo currently PINS a variant for the tau3 run. The wiring smoke
deliberately used `grep_only` to prove tool attachment with zero spend and its
comment names `qwen_embeddings_grep` as "the production variant" — but a comment is
not a pin, and `DEFAULT_RETRIEVAL_VARIANT` upstream is `alltools`, which is a third
thing again. That silent three-way disagreement is exactly the kind of gap that
produces an unexplainable result three days later.

WHY STATIC INTROSPECTION RATHER THAN CONSTRUCTING ALL 21. Constructing a dense
variant embeds all 698 documents at construction and requires an embedding provider,
so a probe that "just tried them all" would spend money and could not run beside a
live experiment. `RetrievalVariant` is a declarative dataclass, so every requirement
is readable from the registry without building anything.

NO NETWORK, NO SPEND, NO DOCKER by default. Safe while another run holds the machine.
"""
from __future__ import annotations

import argparse
import os
import sys

from tau2.domains.banking_knowledge.retrieval import (  # noqa: E402
    DEFAULT_RETRIEVAL_VARIANT,
    RETRIEVAL_VARIANT_ALIASES,
    RETRIEVAL_VARIANTS,
)

# Embedder type -> the credential its provider needs. Read off the upstream CLI help
# ("Requires OPENROUTER_API_KEY: qwen_embeddings*, alltools-qwen") rather than guessed
# from the model name, since the same model string could be served by either provider.
EMBEDDER_CRED = {"openrouter": "OPENROUTER_API_KEY", "openai": "OPENAI_API_KEY"}

# Variants that are not a fair instrument for a SKILL experiment, and why. Kept here
# rather than silently filtered out: the point is to make the trade-off visible, not
# to make the choice.
DEGENERATE = {
    "no_knowledge": "no KB access at all — measures parametric memory, not retrieval",
    "full_kb": "entire corpus in the prompt — no retrieval to skill at",
    "golden_retrieval": "hands over the required documents — an ORACLE CEILING, "
                        "useful as an upper bound, invalid as a baseline",
}


def spec_kind(spec) -> str:
    if spec is None:
        return "-"
    t = getattr(spec, "type", None) or "?"
    m = getattr(spec, "embedder_model", None)
    et = getattr(spec, "embedder_type", None)
    return f"{t}" + (f"/{et}:{m}" if m else "")


def requirements(v) -> tuple[set[str], bool]:
    """(credentials needed, needs_sandbox). Derived from the declarative spec.

    THE RERANKER IS A SEPARATE CREDENTIAL, and missing it was a real bug in this
    script's first version: it reported `bm25_reranker` and `bm25_reranker_grep` as
    "nothing (LOCAL)" because `reranker` is a plain bool on `PipelineSpec` rather than
    a pipeline of its own, so the embedder-type walk never saw it. `--construct` then
    failed both with `OpenAIError: api_key must be set`. A variant advertised as free
    that actually needs a credential is the exact failure this probe exists to
    prevent, reproduced inside the probe — so the requirement below is recorded from
    the CONSTRUCTION result, not inferred from reading.
    """
    creds: set[str] = set()
    for spec in (v.kb_search, v.kb_search_bm25, v.kb_search_dense):
        if spec is None:
            continue
        if getattr(spec, "type", None) == "embedding":
            et = (getattr(spec, "embedder_type", None) or "").lower()
            creds.add(EMBEDDER_CRED.get(et, f"<embedder:{et or 'unknown'}>"))
        if getattr(spec, "reranker", False):
            creds.add("OPENAI_API_KEY(reranker)")
    return creds, v.shell is not None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--construct", action="store_true",
                    help="also build the local-only variants to prove they work "
                         "(no credentials, no docker, no spend)")
    a = ap.parse_args()

    print(f"tau2 source : {os.path.dirname(sys.modules['tau2'].__file__)}")
    print(f"upstream DEFAULT_RETRIEVAL_VARIANT = {DEFAULT_RETRIEVAL_VARIANT!r}")
    print(f"aliases: {RETRIEVAL_VARIANT_ALIASES}\n")

    rows, local = [], []
    for name in RETRIEVAL_VARIANTS:
        v = RETRIEVAL_VARIANTS[name]
        creds, sandbox = requirements(v)
        tools = []
        if v.kb_search is not None:
            tools.append("KB_search")
        if v.kb_search_bm25 is not None:
            tools.append("KB_search_bm25")
        if v.kb_search_dense is not None:
            tools.append("KB_search_dense")
        if v.grep is not None:
            tools.append("grep")
        if v.shell is not None:
            tools.append("shell")
        if not creds and not sandbox:
            local.append(name)
        rows.append((name, tools, creds, sandbox, v))

    print(f"  {'variant':<30} {'tools':<44} {'needs'}")
    for name, tools, creds, sandbox, v in rows:
        need = sorted(creds) + (["sandbox-runtime(docker)"] if sandbox else [])
        mark = "  <-- upstream default" if name == DEFAULT_RETRIEVAL_VARIANT else ""
        print(f"  {name:<30} {','.join(tools) or '(none)':<44} "
              f"{', '.join(need) or 'nothing (LOCAL)'}{mark}")

    print(f"\n  LOCAL-ONLY (runnable right now, zero spend): {len(local)}")
    for n in local:
        note = DEGENERATE.get(n)
        print(f"    {n}" + (f"   [{note}]" if note else ""))

    print("\n  NOT A FAIR SKILL INSTRUMENT (excluded from candidacy, stated not hidden):")
    for n, why in DEGENERATE.items():
        if n in RETRIEVAL_VARIANTS:
            print(f"    {n:<20} {why}")

    cands = [n for n, t, c, s, v in rows if n not in DEGENERATE and not s]
    print(f"\n  CANDIDATE instruments for a skill experiment (no shell/sandbox): {len(cands)}")
    print(f"    {', '.join(cands)}")

    print("\n  DECISION STILL OWED — three sources disagree and none is authoritative:")
    print(f"    upstream default        : {DEFAULT_RETRIEVAL_VARIANT}")
    print("    wiring-smoke comment    : qwen_embeddings_grep ('production')")
    print("    wiring-smoke actual use : grep_only (chosen only to avoid spend)")
    print("    Pin ONE in the runner and record it in the split/oracle metadata, the")
    print("    same way solver_image is recorded for SSB v2 — tree_hash cannot see it.")

    if a.construct:
        print("\n  --construct: building the local-only variants (no spend)")
        from tau2.registry import registry
        ctor = registry.get_env_constructor("banking_knowledge")
        for n in local:
            try:
                env = ctor(retrieval_variant=n)
                tk = getattr(env, "tools", None)
                # tau2 registers tools through a metaclass registry, so `_tools` is
                # not always the live mapping — the first version printed tools=0 for
                # every variant. Mirror tau3_wiring_smoke.tool_names()'s several-attr
                # walk instead of trusting one attribute name.
                names: set = set()
                for attr in ("_tools", "tools", "get_tools"):
                    val = getattr(tk, attr, None)
                    if val is None:
                        continue
                    if callable(val):
                        try:
                            val = val()
                        except Exception:
                            continue
                    if isinstance(val, dict):
                        names = set(val); break
                    if isinstance(val, (list, tuple, set)):
                        cand = {getattr(x, "name", None) or getattr(x, "__name__", None)
                                for x in val}
                        cand.discard(None)
                        if cand:
                            names = cand; break
                print(f"    {n:<20} OK   tools={len(names)}"
                      f"{'  skill=YES' if 'skill' in names else ''}")
            except Exception as e:
                print(f"    {n:<20} FAILED  {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
