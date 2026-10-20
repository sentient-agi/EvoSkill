#!/usr/bin/env python3
"""Is a model a first-party Claude model (Claude Code's own credential), or one reached through a gateway?

    python3 -m src.orchestrator.quota_scope --selftest

ONE DEFINITION, used by claude_pins (which pins apply), the loop's evolver dispatch and the prompt author (which
authentication variables a `claude -p` child keeps). A first-party Claude model runs on the run's ONE Claude Code
credential (src/orchestrator/cc_account.py): the subscription token file, or ANTHROPIC_API_KEY when no token file
is given. A vendor-prefixed model id (`deepseek/...`, `anthropic/claude-...`) is reached through a provider gateway
with ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN, and must never be shown an Anthropic credential.

CONSERVATIVE ON EVERYTHING IT DOES NOT RECOGNISE. An empty, missing or unfamiliar model counts as first-party
Claude, so its child gets the run's Claude credential and nothing a gateway would need.
"""
from __future__ import annotations

import argparse
import re
import sys

# Claude-family markers, matched against the model string case-insensitively. `claude` covers the API ids
# (`claude-opus-5`, `claude-sonnet-5`); the bare aliases are what the CC CLI accepts.
_CLAUDE = re.compile(r"(claude|opus|sonnet|haiku)", re.I)

# Third-party model strings are vendor-prefixed gateway ids. A prefix like `anthropic/` still means Claude weights,
# but served through the gateway, so the prefix is tested before the family name: `anthropic/claude-sonnet-4-6` is
# a gateway model, `claude-opus-5` is not.
_VENDOR_PREFIXED = re.compile(r"^[a-z0-9_.-]+/")


def consumes_claude_quota(model: str | None) -> bool:
    """True if a step running `model` runs on the run's Claude Code credential (a first-party Claude model);
    False for a gateway model. Unknown input is True (see the module docstring)."""
    m = (model or "").strip()
    if not m:
        return True
    if _VENDOR_PREFIXED.match(m):
        return False
    return bool(_CLAUDE.search(m))


# WHAT MAKES `claude` USE ANOTHER CREDENTIAL. Any one of these in the environment and the CLI authenticates with it:
# an API key left in the launch shell would silently move a token-file session onto API billing, and a gateway
# pair left over from solver work would send a first-party session to the gateway.
CLAUDE_AUTH_OVERRIDES = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL")


def subscription_env(model: str | None, env: dict | None = None) -> dict:
    """Child environment for a `claude -p` call, with the authentication variables matched to the model.

    Claude-family model -> ANTHROPIC_AUTH_TOKEN and ANTHROPIC_BASE_URL are removed. ANTHROPIC_API_KEY is removed
                           too when the run has a token file (cc_account.uses_api_key is False), and KEPT when no
                           token file is given, because then it is the run's one credential.
    Gateway model       -> BASE_URL and AUTH_TOKEN are kept (that pair is how it reaches its endpoint), and
                           ANTHROPIC_API_KEY is removed: `claude` prefers it over AUTH_TOKEN and would present an
                           Anthropic credential to a non-Anthropic gateway, which answers 404 model_not_found.

    Scrubbed at the point of use: the variables arrive from the launch shell, not from anything in this repo.
    """
    import os
    from .cc_account import uses_api_key
    e = dict(os.environ if env is None else env)
    if consumes_claude_quota(model):
        keep_key = uses_api_key(e)
        for k in CLAUDE_AUTH_OVERRIDES:
            if not (keep_key and k == "ANTHROPIC_API_KEY"):
                e.pop(k, None)
    else:
        e.pop("ANTHROPIC_API_KEY", None)
    return e


def selftest() -> int:
    """Every case asserted in BOTH directions, because a predicate that only ever returns True passes any one-sided
    test."""
    first_party = [
        "claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5-20251001",
        "opus[1m]", "opus", "sonnet", "haiku",          # CC CLI aliases, as actually passed
        "Claude-Opus-5",                                 # case
        "", "   ", None,                                 # unknown -> conservative
        "some-unreleased-claude-thing",
    ]
    gateway = [
        "deepseek/deepseek-v4-flash-0731",
        "z-ai/glm-5.2:nitro",
        "openai/gpt-5-nano",
        "nex-agi/nex-n2-mini",
        "anthropic/claude-sonnet-4-6",                   # Claude WEIGHTS, but through the gateway
    ]
    bad = 0
    for m in first_party:
        if not consumes_claude_quota(m):
            print(f"  [FAIL] {m!r} should be first-party Claude but is not"); bad += 1
        else:
            print(f"  [ok] {m!r} -> first-party Claude")
    for m in gateway:
        if consumes_claude_quota(m):
            print(f"  [FAIL] {m!r} should be a gateway model but is not"); bad += 1
        else:
            print(f"  [ok] {m!r} -> gateway")

    # THE ROUTING, ASSERTED IN EVERY DIRECTION: ANTHROPIC_API_KEY goes for a gateway model and for a token-file run,
    # and stays for an API-key run.
    tok = {"ANTHROPIC_API_KEY": "sk-ant-x", "ANTHROPIC_AUTH_TOKEN": "sk-or-y",
           "ANTHROPIC_BASE_URL": "http://127.0.0.1:8100", "KEEP_ME": "1",
           "EVOSKILL_CC_TOKEN_FILE": "/nonexistent/token"}
    key = {k: v for k, v in tok.items() if k != "EVOSKILL_CC_TOKEN_FILE"}
    for label, env_, want_absent, want_present in (
        ("claude, token file", subscription_env("opus[1m]", tok),
         ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"), ("KEEP_ME",)),
        ("claude, API key", subscription_env("opus[1m]", key),
         ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"), ("ANTHROPIC_API_KEY", "KEEP_ME")),
        ("gateway", subscription_env("deepseek/deepseek-v4-flash-0731", tok),
         ("ANTHROPIC_API_KEY",), ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "KEEP_ME")),
    ):
        miss = [k for k in want_absent if k in env_] + [k for k in want_present if k not in env_]
        if miss:
            print(f"  [FAIL] {label} env routing wrong: {miss}"); bad += 1
        else:
            print(f"  [ok] {label} env routing")

    print("ALL CASES BEHAVED" if not bad else f"{bad} CASE(S) WRONG")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", help="print whether this model is first-party Claude (exit 0) or a gateway model (1)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.model is None:
        ap.error("--model is required unless --selftest")
    first = consumes_claude_quota(a.model)
    print(f"{a.model!r}: {'first-party Claude (the run credential)' if first else 'gateway model'}")
    return 0 if first else 1


if __name__ == "__main__":
    sys.exit(main())
