#!/usr/bin/env python3
"""WHICH STEP does the Claude five-hour quota gate apply to?

    python3 -m src.orchestrator.quota_scope --model claude-opus-5   # exit 0 = gate applies
    python3 -m src.orchestrator.quota_scope --model deepseek/...    # exit 1 = does not apply
    python3 -m src.orchestrator.quota_scope --selftest

WHY THIS EXISTS. `scripts/quota_fresh_reading.py` answers "is the Claude five-hour window
clear?". It does NOT answer "does this step draw on that window at all", and until 2026-08-13
nothing did: `scripts/tau3_evolver.sh` ran the gate whenever a skip flag was unset, with no
reference to the evolver model, and `loop.py:_run_evolver` ran it unconditionally too. So a
DSV4F evolver — which reaches its model through OpenRouter and consumes exactly zero of the
Claude subscription — was blocked by a reading about a resource it never touches.

The same audit found the mirror-image hole, and it is the more serious of the two:
`loop.py:_run_author_prompt` runs the AO prompt author on `prompt_author_model` (opus[1m] in
every run to date, i.e. Claude) and was NEVER gated. The gate was therefore attached to the
step that might not spend the resource, and absent from the step that always does.

THE RULE THIS MODULE ENCODES. The gate follows the MODEL, not the run. A step is gated iff the
model executing it draws on the Claude subscription. On a DSV4F-evolver arm that means the AO
authoring call is gated and the evolver dispatch is not — which is not "a weaker gate", it is
the gate finally pointing at the thing it measures.

THIS IS NOT A LOOSENING, AND THE DISTINCTION IS THE WHOLE POINT. The 2026-08-01 incident this
gate exists to prevent was a CLAUDE evolver dying mid-run against the Claude five-hour limit,
having written nothing. That failure mode cannot occur for a model served by OpenRouter. What
a non-Claude evolver can exhaust is credit, which is a different resource with its own ceiling
(`ExperimentSpec.cost_ceiling_usd`). Removing a gate because the hazard is absent is not the
same act as removing one because the reading is inconvenient — the latter is what 2026-08-01
was, and `quota_fresh_reading.py` still refuses to re-roll an adverse reading.

CONSERVATIVE ON EVERYTHING IT DOES NOT RECOGNISE. An empty, missing or unfamiliar model
returns True and is therefore GATED. A new Claude alias that this matcher has not been taught
costs one needless gate check; a new Claude alias mistaken for a third-party model costs a
mid-run hard kill. Those are not symmetric, so the default is not symmetric either.

ONE DEFINITION, TWO CALLERS. `loop.py` imports `consumes_claude_quota`; `tau3_evolver.sh`
shells out to the CLI above. Neither carries its own copy of the rule. A second copy in the
shell is #147's exact shape, and the shell's copy would be the untestable one.
"""
from __future__ import annotations

import argparse
import re
import sys

# Claude-family markers, matched against the model string case-insensitively. `claude` covers
# the API ids (`claude-opus-5`, `claude-sonnet-5`); the bare aliases are what the CC CLI accepts
# and what `EVOSKILL_EVOLVER_MODEL` has actually carried in every run to date (`opus[1m]`).
_CLAUDE = re.compile(r"(claude|opus|sonnet|haiku)", re.I)

# Third-party model strings are vendor-prefixed OpenRouter ids. A prefix like `anthropic/` still
# means Claude weights, but served through OpenRouter and billed as credit — so it does NOT touch
# the subscription window. That is why the test below looks at the prefix before the family name:
# `anthropic/claude-sonnet-4-6` is NOT gated, `claude-opus-5` is.
_VENDOR_PREFIXED = re.compile(r"^[a-z0-9_.-]+/")


def consumes_claude_quota(model: str | None) -> bool:
    """True if a step running `model` draws on the Claude five-hour subscription window.

    True  -> the five-hour gate applies before launching this step.
    False -> it does not; the step is bounded by `cost_ceiling_usd` instead.

    Unknown input is True. See the module docstring on why the default is asymmetric.
    """
    m = (model or "").strip()
    if not m:
        return True                     # unnamed model: gate, and let the caller's own
        #                                 empty-model refusal produce the better error.
    if _VENDOR_PREFIXED.match(m):
        # Routed through a provider gateway (openrouter ids are always `vendor/model`), so it is
        # billed as credit even when the vendor is `anthropic`. The subscription is untouched.
        return False
    return bool(_CLAUDE.search(m))


# #307. WHAT MAKES `claude` LEAVE THE OAUTH SUBSCRIPTION. Any one of these in the environment and
# the CLI authenticates as an API client instead: it bills credit rather than the five-hour
# window, and — as #306 showed — it stops emitting `rate_limit_event` altogether.
#
# ANTHROPIC_API_KEY was missing from loop.py's scrub list until 2026-08-13, and the omission is
# measurable in the artifacts. Every AO-authoring and evolver session of ao5 and ssbv2_ao1
# recorded apiKeySource='none' (subscription); ao6's e1 authoring recorded 'ANTHROPIC_API_KEY' —
# the one session run after the operator's shell began exporting it. Same model, same policy,
# same prompt; different auth and billing path; nothing reported it.
#
# The second harm is worse than the cost: it makes the quota gate VACUOUS in exactly the case it
# covers. The gate reads the subscription window, then the step it guards spends something else.
CLAUDE_AUTH_OVERRIDES = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL")


def subscription_env(model: str | None, env: dict | None = None) -> dict:
    """Child environment for a `claude -p` call, with the auth path forced to match the model.

    Claude-family model  -> scrub the API-key overrides so it uses the OAuth subscription, which
                            is the path every completed run measured on and the path the quota
                            gate reads.
    Gateway model        -> BASE_URL and AUTH_TOKEN are kept, because that pair is exactly how it
                            reaches its endpoint — but ANTHROPIC_API_KEY is REMOVED, because
                            `claude` prefers it over AUTH_TOKEN and would then present an
                            Anthropic credential to a non-Anthropic gateway.

    #307. SCRUBBED AT THE POINT OF USE, NOT ASKED OF THE OPERATOR. The key arrives from the
    launch shell (`set -a; . .env; set +a`), not from anything in this repo, so the only fix that
    survives a forgetful launch is to remove it here. Same reasoning as #306's probe-side scrub.
    """
    import os
    e = dict(os.environ if env is None else env)
    if consumes_claude_quota(model):
        for k in CLAUDE_AUTH_OVERRIDES:
            e.pop(k, None)
    else:
        # #311. ANTHROPIC_API_KEY SHADOWS ANTHROPIC_AUTH_TOKEN, so it must go even on the path
        # that keeps the other two. `claude` resolves the API key first (it reports
        # apiKeySource='ANTHROPIC_API_KEY' when it does), sends it as `x-api-key`, and the
        # gateway then receives an ANTHROPIC credential — which OpenRouter answers with
        # 404 model_not_found, not 401, so it reads as "that model does not exist" rather than
        # "wrong key". Measured 2026-08-13: the AO ablation's dsv4f arm died in 2s with
        # `{"error":"model_not_found"}` for `deepseek/deepseek-v4-flash-0731` while a direct
        # curl of the SAME id with the SAME org key returned 200 from both /v1/messages and
        # /v1/chat/completions, and the id was present in a 412-model catalogue.
        #
        # The first version of THIS function returned a gateway model's environment unchanged,
        # which is the hole that produced that failure. scripts/dsv4fevo_launchers/* had it
        # right all along — they pop ANTHROPIC_API_KEY and authenticate via AUTH_TOKEN — and
        # the working recipe was three files away while I diagnosed from first principles.
        e.pop("ANTHROPIC_API_KEY", None)
    return e


def gate_reason(model: str | None) -> str:
    """One line for a log, saying which way it went and why. Logged at every skip: a gate that
    silently does not run is indistinguishable from one that ran and passed (#209's shape)."""
    m = (model or "").strip() or "<unset>"
    if consumes_claude_quota(model):
        return (f"five-hour quota gate APPLIES: {m} runs on the Claude subscription, whose "
                f"exhaustion is a hard kill with no overflow")
    return (f"five-hour quota gate does NOT apply: {m} is served through a provider gateway "
            f"and consumes credit, not the Claude subscription window (bounded by "
            f"cost_ceiling_usd instead)")


def selftest() -> int:
    """Every case asserted in BOTH directions, because a predicate that only ever returns True
    passes any one-sided test — the vacuous-guard shape this project has shipped twice (#154,
    #248)."""
    gated = [
        "claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5-20251001",
        "opus[1m]", "opus", "sonnet", "haiku",          # CC CLI aliases, as actually passed
        "Claude-Opus-5",                                 # case
        "", "   ", None,                                 # unknown -> conservative
        "some-unreleased-claude-thing",
    ]
    ungated = [
        "deepseek/deepseek-v4-flash-0731",               # the DSV4F arm
        "z-ai/glm-5.2:nitro",
        "openai/gpt-5-nano",
        "nex-agi/nex-n2-mini",
        "anthropic/claude-sonnet-4-6",                   # Claude WEIGHTS, but billed as credit
    ]
    bad = 0
    for m in gated:
        if not consumes_claude_quota(m):
            print(f"  [FAIL] {m!r} should be GATED but is not"); bad += 1
        else:
            print(f"  [ok] {m!r} -> gated")
    for m in ungated:
        if consumes_claude_quota(m):
            print(f"  [FAIL] {m!r} should NOT be gated but is"); bad += 1
        else:
            print(f"  [ok] {m!r} -> not gated")

    # The asymmetry itself is the invariant, so assert it rather than trusting the lists above.
    if not consumes_claude_quota(None) or not consumes_claude_quota(""):
        print("  [FAIL] unknown model must default to GATED"); bad += 1
    else:
        print("  [ok] unknown model defaults to gated")

    # #311. THE ROUTING, ASSERTED IN BOTH DIRECTIONS. Checking only that the Claude path is
    # scrubbed would have passed while the gateway path shipped the bug that cost two ablation
    # arms: ANTHROPIC_API_KEY must be gone in BOTH cases, for opposite reasons.
    fake = {"ANTHROPIC_API_KEY": "sk-ant-x", "ANTHROPIC_AUTH_TOKEN": "sk-or-y",
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:8100", "KEEP_ME": "1"}
    claude_env = subscription_env("opus[1m]", fake)
    gw_env = subscription_env("deepseek/deepseek-v4-flash-0731", fake)
    for label, env_, want_absent, want_present in (
        ("claude", claude_env, ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"), ("KEEP_ME",)),
        ("gateway", gw_env, ("ANTHROPIC_API_KEY",), ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "KEEP_ME")),
    ):
        miss = [k for k in want_absent if k in env_] + [k for k in want_present if k not in env_]
        if miss:
            print(f"  [FAIL] {label} env routing wrong: {miss}"); bad += 1
        else:
            print(f"  [ok] {label} env routing: "
                  f"{'subscription forced' if label == 'claude' else 'gateway creds kept, API key removed'}")

    print("ALL CASES BEHAVED" if not bad else f"{bad} CASE(S) WRONG")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", help="the model that will execute the step")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.model is None:
        ap.error("--model is required unless --selftest")
    applies = consumes_claude_quota(a.model)
    if not a.quiet:
        print(gate_reason(a.model))
    # Exit code IS the answer, so the shell needs no parsing: 0 = gate applies (run it),
    # 1 = it does not. Deliberately the same polarity as `quota_fresh_reading.py`'s "0 = go".
    return 0 if applies else 1


if __name__ == "__main__":
    sys.exit(main())
