#!/usr/bin/env python3
"""tau3 banking_knowledge rollout leg on the FROZEN split, N samples per task.

    EVOSKILL_TAU3_VARIANT=bm25_grep \
    EVOSKILL_SPLIT_KEY=train \
    EVOSKILL_RUN_TAG=tau3_base_train \
      python3 scripts/run_tau3_banking.py

Smoke the SAME script with env overrides rather than a separate toy script
(`EVOSKILL_N_TASKS=2 EVOSKILL_N_SAMPLES=1`), so what is smoked is what runs.

WHY THIS EXISTS, and what it fixes relative to `run_tau2_smoke_base.py`:

  1. FROZEN SPLIT, not `test[:N]`. The smoke took a prefix of whatever order the
     loader returned. This reads `outputs/tau3_banking/split.json` — stratified on
     binned hop depth, reproducible from seed 730 — and selects by task_id, so the
     leg cannot silently redraw its own task set between runs.

  2. N SAMPLES, with a real per-sample identity. banking_knowledge has ~32 tasks per
     split, and the split file's own SELECTION_WARNING records that a 40-task val with
     max-selection produced a tau2 skill reading +20pp at p~0.01 that then tested NULL.
     Repeats are the only cheap defence, so N defaults to 3.

  3. THE RETRIEVAL VARIANT IS REQUIRED AND RECORDED. There is no default on purpose.
     `banking_knowledge` ships 21 variants and the variant decides which TOOLS the
     agent has, i.e. what the task IS — `alltools` (upstream default) needs
     OPENAI_API_KEY plus a docker sandbox and is unavailable here; `grep_only` was only
     ever chosen to avoid spend. A run that inherits a default variant is a run whose
     instrument nobody chose.

  4. THE VARIANT NAMES THE CACHE DIR. Verified 2026-08-03 against a real record: the
     tau2 cache key is `effort|model|question|question_hash|sdk|tree_hash` and
     `question` is only `{"domain","task_id"}` — **the retrieval variant is not in the
     key.** 897 banking_knowledge records already exist from earlier variants, so a
     shared cache dir would replay a DIFFERENT experiment on a silent hit. Putting the
     variant in the DIRECTORY gives the cache the identity its key lacks, costs no key
     change, and evicts nothing. Same class of gap as `solver_image` on SSB v2, handled
     the same way: also written into the result JSON.

WHY N PASSES OVER `evaluate_agent_parallel` RATHER THAN A HAND-ROLLED GATHER.
`evaluate_agent_parallel` takes no `sample_idx`, so N copies of one item would share a
cache entry and collapse to a single episode. The alternative — reimplementing the fan-out
the way the SSB v2 runner does — would also reimplement its 3-attempt outer retry budget,
its `make_partial_error_trace` handling and its exception accounting, all of which were
hardened by the 2026-07-01 and 2026-07-11 audits. So instead each sample index gets its
OWN cache dir (`..._s0_cache`, `_s1_`, ...) and the proven path is called once per index.
The per-sample dir is deliberately explicit rather than clever: `ls .cache` shows exactly
which samples exist.
"""
from __future__ import annotations

import asyncio
import collections
import datetime as _dt
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILE = Path(os.environ.get(
    "EVOSKILL_TAU3_SPLIT", PROJECT_ROOT / "outputs/tau3_banking/split.json"))
TAU3_ROOT = os.environ.get("TAU2_BENCH_ROOT",
                           str(PROJECT_ROOT / "third_party/tau3-bench"))
# DOMAIN IS NOW A PARAMETER. It was hardcoded to "banking_knowledge", which made
# every other tau3 task type unrunnable through this script — the file name still says
# `banking` because launchers, logs and the outputs/ tree all reference it, and
# renaming it would orphan that history for no gain.
#
# RETRIEVAL IS A BANKING-ONLY CONCEPT. `retrieval_variant` exists on
# banking_knowledge's env constructor and nowhere else, so the variant is REQUIRED for
# banking_knowledge and must be ABSENT for the others. Both directions are enforced in
# preflight: silently ignoring a variant passed to `airline` would record
# `retrieval_variant: "bm25_grep"` in a result JSON for a run where retrieval played
# no part, which is a lie in the field that names the instrument.
_DOMAINS = ("banking_knowledge", "airline", "retail", "telecom")
DOMAIN = os.environ.get("EVOSKILL_TAU3_DOMAIN", "banking_knowledge")
_RETRIEVAL_DOMAIN = "banking_knowledge"

# ---- the one parameter with NO default (banking_knowledge only) -------------
VARIANT = os.environ.get("EVOSKILL_TAU3_VARIANT") or ""
# Cache-dir token. An empty VARIANT would render `tau3_airline__s0_cache`, whose
# double underscore is easy to misread as a truncation; and a bucket that LOOKS
# like a variant-less banking bucket is exactly the kind of ambiguity that made #148
# a contamination sink. Spell the absence.
_VTOK = VARIANT or "noretrieval"

SPLIT_KEY = os.environ.get("EVOSKILL_SPLIT_KEY", "train")
N_SAMPLES = int(os.environ.get("EVOSKILL_N_SAMPLES", "3"))
N_TASKS = int(os.environ.get("EVOSKILL_N_TASKS", "0"))     # 0 = the whole split
CONCURRENCY = int(os.environ.get("EVOSKILL_CONCURRENCY", "8"))
_RAW_SOLVER_MODEL = os.environ.get(
    "EVOSKILL_SOLVER_MODEL", "openrouter/deepseek/deepseek-v4-flash-0731")
# THE SPEC'S MODEL IS AN OPENROUTER ID; LITELLM ROUTES BY PREFIX. The spec stores the bare
# vendor form (deepseek/deepseek-v4-flash-0731 — spec.py's cross-family pin check requires
# that form), but this runner's calls go through litellm, where a bare `deepseek/` prefix
# selects the NATIVE DeepSeek provider. No credential for that exists here, so every episode
# of the first leg to combine the #320 spec-model export with this runner died with
# AuthenticationError — while the [PIN] banner claimed the OpenRouter pin was verified for a
# call that never went through OpenRouter. All prior runs used this default's `openrouter/`
# form, so the mismatch was invisible until the 2026-08-17 smoke. This runner routes
# exclusively via OpenRouter (it enforces the org key and the provider pin), so normalize.
SOLVER_MODEL = (_RAW_SOLVER_MODEL if _RAW_SOLVER_MODEL.startswith("openrouter/")
                else f"openrouter/{_RAW_SOLVER_MODEL}")
# ── USER-SIM MODELS: THE TWO TIERS ARE NOW SETTABLE INDEPENDENTLY ───────────
# tau2 picks the user simulator PER TASK: `user_model = reference if tier == "reference"
# else cheap` (solver.py:577, tier from user_tier_gate.get_tier). So there are two knobs,
# and this file exposed only ONE — `EVOSKILL_TAU2_USER_MODEL` was assigned to BOTH
# `reference_user_model` and `cheap_user_model`.
#
# WHAT THAT COST, 2026-08-06. Relaunching the e2 leg I reconstructed the env from the
# PARTIAL result JSON's `user_model_reference` field and exported the single knob. That
# silently collapsed `cheap` from the solver default `gpt-5-nano` to `gpt-4.1`:
#     e0 base, e1 v1     reference gpt-4.1 / cheap gpt-5-nano   (solver defaults)
#     e2, e3 (my launch) reference gpt-4.1 / cheap gpt-4.1      (collapsed)
# INERT on banking_knowledge — get_tier returns "reference" for ALL 97 tasks (verified
# across the frozen 32/32/33 split), so `cheap` is never invoked and no episode used a
# different user simulator. But it is a real config drift and on any split with cheap-tier
# tasks it would have changed the instrument mid-experiment. Collapsing the tiers also
# makes the tier gate inert, which the 0802 session already flagged as methodological.
#
# The recorded DECISION is `gpt-4.1` for the tau3 REFERENCE tier (0804b §E3, after
# retracting the cost case for nano in §K4: the user sim is 18.7% of episode cost, not
# 59%, so switching saves ~18%, not ~57% — not worth trading tau2's own reference-model
# comparability, `DEFAULT_LLM_AGENT = gpt-4.1-2025-04-14`). gpt-5-nano is the CHEAP tier.
# Neither default is changed here; only the ability to set them separately is added.
_USER_MODEL_BOTH = os.environ.get("EVOSKILL_TAU2_USER_MODEL")
USER_MODEL_REFERENCE = os.environ.get("EVOSKILL_TAU2_USER_MODEL_REFERENCE") or _USER_MODEL_BOTH
USER_MODEL_CHEAP = os.environ.get("EVOSKILL_TAU2_USER_MODEL_CHEAP") or _USER_MODEL_BOTH
if _USER_MODEL_BOTH and not (os.environ.get("EVOSKILL_TAU2_USER_MODEL_REFERENCE")
                             or os.environ.get("EVOSKILL_TAU2_USER_MODEL_CHEAP")):
    print(f"  WARNING    : EVOSKILL_TAU2_USER_MODEL={_USER_MODEL_BOTH} sets BOTH user-sim "
          f"tiers,\n               collapsing reference and cheap into one model and making "
          f"the tier\n               gate inert. Prefer EVOSKILL_TAU2_USER_MODEL_REFERENCE / "
          f"_CHEAP.", flush=True)
EPISODE_TIMEOUT = int(os.environ.get("EVOSKILL_TAU2_EPISODE_TIMEOUT", "460"))

# #182 DEFAULTS ON FOR TAU3, and this is the durable half of that fix. The guard in
# RunCache is opt-in and OFF by default -- correct for the repo at large, because most
# harnesses have no wall-limited termination and would pay a full re-execution for entries
# it refuses. But tau3 is precisely the harness that HAS one, and leaving activation to a
# hand-typed export failed within hours on 2026-08-06: e3_v3_val ran without it and wrote a
# 2821s wall-limited record into the live skills_v3 bucket, servable to any later leg.
# setdefault, not a hard set, so an operator can still switch it OFF deliberately.
os.environ.setdefault("EVOSKILL_CACHE_SKIP_WALL_LIMITED", "1")

# HARD PIN for DeepSeek, propagated to the worker subprocess through the environment.
# `pin_request_body_provider` only writes allow_fallbacks when this is set; unset means a
# preference-only pin that OpenRouter may still reroute -- indistinguishable from pinned
# in every log, which is why it is defaulted rather than left to the launcher.
os.environ.setdefault("EVOSKILL_SOLVER_ALLOW_FALLBACKS", "0")


def _solver_provider_pin(solver):
    """The provider block this leg's solver will ship, for the result JSON."""
    try:
        from src.agent_profiles.tau2_agent.solver import _provider_pin_for
        return _provider_pin_for(getattr(solver, "model", "") or SOLVER_MODEL)
    except Exception:
        return None


def _assert_provider_pin_in_force(solver):
    """Refuse to start a DeepSeek leg unless BOTH halves of the hard pin are present.

    #169 asked for the SERVED provider to be recorded so the pin could be verified after
    the fact. That needs plumbing through the shared client layer (6 harnesses, all-harness
    parity) and is NOT done here -- see the note in the result JSON. But the risk #169
    guards against is nearly closed from the other end: with `order=["deepseek"]` AND
    `allow_fallbacks=False`, OpenRouter can only serve DeepSeek or ERROR. It cannot silently
    substitute another provider. So the thing actually worth asserting is that both halves
    are really in force -- which is exactly what has failed before.

    It has failed before, twice, and silently:
      - 2026-07-30: options were built at TWO sites and the pin landed on the
        non-executing dict, so the pin was inert while every log claimed it was set.
      - 2026-08-01: several hand-rolled OpenRouter bodies set `order` and omitted
        `allow_fallbacks`, which makes the pin a mere PREFERENCE -- fallback still allowed.
    Both are invisible without an assertion, because a preference-only pin usually gets
    DeepSeek anyway and only diverges under load.

    DeepSeek family ONLY. GLM 5.2 on :nitro must keep multi-provider routing.
    """
    model = (getattr(solver, "model", "") or SOLVER_MODEL or "").lower()
    _explicit = [x.strip() for x in
                 os.environ.get("EVOSKILL_SOLVER_PROVIDER_ORDER", "").split(",") if x.strip()]
    if _explicit:
        # CROSS-FAMILY explicit pin. Asserted with the SAME strictness as the DeepSeek one:
        # an order without allow_fallbacks=False is a preference, not a pin, and reads as
        # pinned in every log. Checked here rather than trusted because that failure has
        # already happened twice (see this docstring).
        pin = _solver_provider_pin(solver)
        order = [str(x).lower() for x in ((pin or {}).get("order") or [])]
        fallbacks = (pin or {}).get("allow_fallbacks")
        if order != [x.lower() for x in _explicit] or fallbacks is not False:
            raise SystemExit(
                f"REFUSING: EVOSKILL_SOLVER_PROVIDER_ORDER={_explicit!r} was requested for "
                f"{model} but the pin as built is {pin!r}. Expected order="
                f"{[x.lower() for x in _explicit]} and allow_fallbacks=False; with fallbacks "
                f"unset or True another provider may serve and #169 means you could not tell "
                f"afterwards. Export EVOSKILL_SOLVER_ALLOW_FALLBACKS=0.")
        print(f"  [PIN] VERIFIED explicit hard pin for {model}: order={order} "
              f"allow_fallbacks=False (cross-family override).", flush=True)
        return
    if "deepseek" not in model:
        print(f"  [PIN] {model} is not DeepSeek family -- multi-provider routing left alone "
              f"(correct: pinning GLM/:nitro would defeat its routing).", flush=True)
        return
    pin = _solver_provider_pin(solver)
    if not pin:
        raise SystemExit(
            f"REFUSING: solver {model} is DeepSeek family but no provider pin was produced. "
            f"_solver_provider_pin() swallows exceptions and returns None, so this ALSO fires "
            f"when the import failed -- either way the pin is not verifiable and the standing "
            f"directive (hard-pin every DeepSeek model) is unmet.")
    order = [str(x).lower() for x in (pin.get("order") or [])]
    fallbacks = pin.get("allow_fallbacks")
    problems = []
    if order != ["deepseek"]:
        problems.append(f"order is {order!r}, expected ['deepseek']")
    if fallbacks is not False:
        problems.append(f"allow_fallbacks is {fallbacks!r}, expected False -- with it unset or "
                        f"True the pin is a PREFERENCE and another provider may serve")
    if problems:
        raise SystemExit("REFUSING: the DeepSeek hard pin is not fully in force:\n"
                         + "\n".join(f"    - {p}" for p in problems)
                         + f"\n  pin block as built: {pin!r}\n"
                         f"  Export EVOSKILL_SOLVER_ALLOW_FALLBACKS=0 and verify "
                         f"src/harness/direct/options.py:pin_request_body_provider.")
    print(f"  [PIN] VERIFIED hard pin for {model}: order={order} allow_fallbacks=False. "
          f"OpenRouter can serve DeepSeek or error, never substitute.", flush=True)


def _assert_cache_identity(cdir, solver):
    """Refuse to reuse a cache dir under an instrument that changed beneath it (#183).

    THE HOLE. The RunCache key is tree_hash | question_hash | sdk | model | effort, and
    `model` is the AGENT's model. The USER SIMULATOR's model is in neither the key nor the
    dir name -- yet the user simulator generates half of every episode. Swap gpt-4.1 for
    gpt-5-nano and every existing record replays as if nothing changed. This is the same
    class as the retrieval_variant note above and as #182's wall, and it is not
    hypothetical: on 2026-08-06 a single-knob launcher silently collapsed the `cheap` tier
    from gpt-5-nano to gpt-4.1 (inert only because every banking_knowledge task is
    reference tier), and a switch of the REFERENCE model is an open decision.

    WHY A SIDECAR AND NOT A LONGER DIR NAME. Putting the user models in the dir name is the
    obvious fix and the wrong one here: it renames the dir for the CURRENT, unchanged config
    too, orphaning every record on disk and forcing a full re-run to buy nothing.

    WHY THE WALL WARNS AND THE USER MODEL REFUSES -- the distinction is the point:
      * a user-model change invalidates EVERY record, because the conversation itself
        differs, so reuse is meaningless -> REFUSE;
      * a wall change invalidates only records the wall actually decided. A normally
        terminated episode is wall-INDEPENDENT and stays perfectly valid, and #182 stops
        the wall-limited ones from being stored or served -- BUT ONLY WHEN #182 IS ON, so
        this branch now verifies that instead of asserting it (see below). Refusing on
        every wall change would force a full re-run every time we widen a timeout.
    A guard that refused on both would look stricter and be worse: it would make the honest
    #175 wall widening cost a whole re-run, which is how a good fix gets reverted.

    THREE HOLES IN MY OWN FIRST VERSION, all found by adversarial review on 2026-08-06 and
    all confirmed against the code before being closed here:

    1. TRUST ON FIRST USE. The original wrote the sidecar whenever it was ABSENT. But an
       absent sidecar next to a POPULATED dir does not mean "fresh dir" -- it means
       "records built before this guard existed", which is exactly the population at risk.
       Blessing them stamps the CURRENT config on them as provenance and makes the mismatch
       unfalsifiable forever. Proven non-hypothetical: the retail probe dir holds cheap-tier
       episodes run on gpt-4.1 while the current default cheap model is gpt-5-nano.
       Now: interrogate the RECORDS (each payload carries user_sim_model + user_sim_tier)
       and refuse on a real mismatch. A dir whose records cannot answer is UNKNOWN, not
       matching -- and "unknown" refuses too, because a pass-on-miss check here would be
       the vacuous-guard pattern this project keeps hitting.
    2. THE WALL BRANCH ASSERTED #182's PROTECTION WITHOUT CHECKING IT. #182 is opt-in and
       OFF by default, so "reuse is sound" was an unchecked premise -- the same shape as
       #175's vacuous ordering comment: an invariant stated in prose beside code that does
       not derive it. Now the wall branch REFUSES when the guard is off.
    3. THE FLAG WAS INVISIBLE IN PROVENANCE. No result JSON recorded whether #182 was live,
       so a reader could not tell which legs were protected. Now it is in the sidecar and
       in the leg's own meta.
    """
    import json as _json
    import os as _os
    wall_guard_on = _os.environ.get(
        "EVOSKILL_CACHE_SKIP_WALL_LIMITED", "") not in ("", "0")
    ident = {
        "domain": DOMAIN, "variant": _VTOK,
        "reference_user_model": solver.reference_user_model,
        "cheap_user_model": solver.cheap_user_model,
        "solver_model": SOLVER_MODEL,
        # #185. The AGENT budget -- clean seconds and agent turns -- is the pair that
        # actually decides when an episode stops (the 5400s episode wall below is a
        # backstop that must never fire). Both were absent from this dict, which is the
        # exact hole #182 found for the wall and #183 for the user-sim model: a field that
        # changes the result but not the identity, so records from a DIFFERENT budget
        # replay silently into a new leg and are counted as if they were run under it.
        # Read from the env the worker itself reads (_tau2_worker.py:292-293), not from a
        # local copy, so the identity cannot drift from the value in force.
        "agent_clean_s": float(_os.environ.get("EVOSKILL_TAU2_AGENT_CLEAN_S", "0") or 0),
        "agent_turns": int(_os.environ.get("EVOSKILL_TAU2_AGENT_TURNS", "0") or 0),
        "inactivity_s": float(_os.environ.get("EVOSKILL_TAU2_INACTIVITY_S", "300") or 300),
        # #192. The PER-LLM-CALL budget belongs here for the same reason as the three above:
        # it changes what the solver is allowed to do, and it is not in the cache key. By
        # default it is DERIVED from the episode wall (`_tau2_protocol.llm_call_budget`:
        # max(60, wall/3)), so raising the backstop silently raises how long a single hung
        # call may block -- 2700s wall gave 900s (which is exactly #147's "900.04s" deaths),
        # and the 5400s wall this run uses gives 1800s. Recording it means a leg run under a
        # different per-call budget can never replay into one that is not.
        "llm_per_call_s": float(_os.environ.get(
            "EVOSKILL_TAU2_LLM_TIMEOUT_S", max(60.0, float(EPISODE_TIMEOUT) / 3.0))),
        "llm_retries": int(_os.environ.get("EVOSKILL_TAU2_LLM_RETRIES", "2")),
    }
    wall = {"episode_timeout_s": EPISODE_TIMEOUT}
    path = cdir.parent / f"{cdir.name}.IDENTITY.json"
    cdir.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        # HOLE 1: interrogate existing records rather than blessing them.
        files = sorted(cdir.glob("*/*.json"))
        rogue, n_read = [], 0
        for rp in files:
            try:
                rec = _json.loads(rp.read_text())
                _fa = ((rec.get("trace") or {}).get("output") or {}).get("final_answer")
                payload = _json.loads(str(_fa)) or {}
            except (ValueError, OSError, TypeError):
                continue
            m, t = payload.get("user_sim_model"), payload.get("user_sim_tier")
            if not m or not t:
                continue
            n_read += 1
            want = (solver.reference_user_model if t == "reference"
                    else solver.cheap_user_model)
            if m != want:
                rogue.append((payload.get("task_id"), t, m, want))
        if files and not n_read:
            raise SystemExit(
                f"REFUSING: {cdir.name} holds {len(files)} record(s), has no IDENTITY "
                f"sidecar, and none of them expose user_sim_model/user_sim_tier -- so the "
                f"instrument that produced them is UNKNOWN, which is not the same as "
                f"matching. Point this run at a fresh cache dir, or write the sidecar by "
                f"hand with \"blessed_retroactively\": true and the evidence you used.")
        if rogue:
            lines = "\n".join(f"    task {t!r} tier={ti}: cached={m!r} current={w!r}"
                              for t, ti, m, w in rogue[:8])
            raise SystemExit(
                f"REFUSING: {cdir.name} has no IDENTITY sidecar and {len(rogue)} of its "
                f"{n_read} readable record(s) were produced by a DIFFERENT user "
                f"simulator:\n{lines}\n"
                f"  Writing a sidecar now would assert the CURRENT config as their "
                f"provenance. Use a fresh cache dir, or bless them explicitly.")
        path.write_text(_json.dumps(
            {**ident, "wall_ADVISORY": wall,
             "cache_skip_wall_limited": wall_guard_on,
             "records_verified_at_bootstrap": n_read}, indent=1))
        return
    try:
        prev = _json.loads(path.read_text())
    except (ValueError, OSError) as e:
        raise SystemExit(f"REFUSING: {path} is unreadable ({e}). It records which "
                         f"instrument built {cdir.name}; delete it only if you are certain "
                         f"the records match the current config.")
    drift = {k: (prev.get(k), v) for k, v in ident.items() if prev.get(k) != v}
    if drift:
        lines = "\n".join(f"    {k}: cached={was!r} now={now!r}" for k, (was, now) in drift.items())
        raise SystemExit(
            f"REFUSING: {cdir.name} holds records built by a DIFFERENT instrument:\n{lines}\n"
            f"  These fields are not in the cache key, so reusing this dir would replay "
            f"episodes from the other configuration and report them as this one's.\n"
            f"  Either restore the config, or point this run at a fresh cache dir.")
    prev_wall = prev.get("wall_ADVISORY") or {}
    if prev_wall != wall:
        # HOLE 2: this branch used to ASSERT that #182 protects wall-limited records. It
        # only does when the guard is switched on, and the guard defaults OFF -- so verify.
        if not wall_guard_on:
            raise SystemExit(
                f"REFUSING: the wall changed since {cdir.name} was created "
                f"({prev_wall} -> {wall}) and EVOSKILL_CACHE_SKIP_WALL_LIMITED is NOT set, "
                f"so #182's store/serve refusal (src/cache/run_cache.py:_wall_skip_reasons) "
                f"is a no-op and records this dir's OLD wall decided WILL replay into a run "
                f"using the new one. That is exactly the straddle that corrupted the "
                f"epoch-2 val gate. Export EVOSKILL_CACHE_SKIP_WALL_LIMITED=1 and relaunch, "
                f"or point this run at a fresh cache dir.")
        print(f"  [CACHE-IDENTITY] wall changed since {cdir.name} was created "
              f"({prev_wall} -> {wall}). NOT refusing: normally-terminated records are "
              f"wall-independent, and #182 IS ACTIVE (verified: "
              f"EVOSKILL_CACHE_SKIP_WALL_LIMITED set) so wall-limited records are neither "
              f"stored nor served. Reuse is sound.", flush=True)
        path.write_text(_json.dumps(
            {**ident, "wall_ADVISORY": wall,
             "cache_skip_wall_limited": wall_guard_on}, indent=1))
SKILLS_DIR = os.environ.get("EVOSKILL_SKILLS_DIR") or ""
TAG = os.environ.get("EVOSKILL_RUN_TAG", f"tau3_{SPLIT_KEY}")
# #190: THE OUTPUT DIR IS PARAMETERISED, AND A CROSS-DOMAIN WRITE IS REFUSED.
# This was hardcoded to outputs/tau3_banking/ and ignored EVOSKILL_TAU3_RUN_DIR entirely,
# so the result path depended on TAG ALONE. On 2026-08-07 the retail base legs ran with
# tags e0_base_train / e0_base_val -- the same tags banking had used -- and OVERWROTE
# banking's base result JSONs with retail data. Nothing complained: both files were
# well-formed, both said n_tasks=20, and only `domain` inside them revealed the swap.
# Same class as #179 (RUN_DIR parameterised at three sites and missed at this one).
#
# TWO defences, because the path fix alone would not have caught it:
#   (a) the dir now follows EVOSKILL_TAU3_RUN_DIR, defaulting to the per-domain
#       outputs/tau3_<domain-ish> rather than always banking; and
#   (b) if a result JSON already exists at the target and names a DIFFERENT domain,
#       refuse rather than overwrite. A tag collision across domains is a mistake every
#       time, and the cost of being wrong is silent destruction of a measured baseline.
_RUN_DIR_DEFAULT = "outputs/tau3_banking" if DOMAIN == "banking_knowledge" else f"outputs/tau3_{DOMAIN}"
OUT_DIR = PROJECT_ROOT / os.environ.get("EVOSKILL_TAU3_RUN_DIR", _RUN_DIR_DEFAULT)
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / f"{TAG}.json"
if OUT.exists():
    try:
        _prior = json.loads(OUT.read_text()).get("domain")
    except Exception:
        _prior = None
    if _prior and _prior != DOMAIN:
        raise SystemExit(
            f"REFUSING: {OUT} already holds results for domain {_prior!r} but this run is "
            f"{DOMAIN!r}. Tag {TAG!r} collides across domains and overwriting would destroy "
            f"a measured baseline (this happened on 2026-08-07). Use a domain-specific tag "
            f"or set EVOSKILL_TAU3_RUN_DIR.")

# ORG key for all OpenRouter spend (standing directive). Only the last 4 chars are printed.
#
# AN EXPORTED KEY WINS. This used to read .env and OVERWRITE os.environ unconditionally,
# which broke three things at once: the standing "export the org key in every launcher"
# directive silently did nothing; `EVOSKILL_REQUIRE_KEY_SUFFIX` in tau3_leg_b2525.sh fenced
# the EXPORTED key while the run then spent the .env one, making the spend fence vacuous;
# and a missing .env (or one without the variable) died with an IndexError from split()
# rather than a message. It also contradicted .env.example's own first line, which promises
# this file is read only when the variables are not already exported.
#
# Mirrors run_ssbv2_base30_dsv4f_n3.py, which already had the correct shape -- same
# precedence, same _WHICH_KEY record (all-harness parity).
if os.environ.get("OPENROUTER_API_KEY"):
    _WHICH_KEY = "env"
else:
    _envfile = Path(os.environ.get("EVOSKILL_ENV_FILE", PROJECT_ROOT / ".env"))
    if not _envfile.is_file():
        raise SystemExit(
            f"set OPENROUTER_API_KEY, or create {_envfile} with an OPENROUTER_API_KEY= line "
            f"(override the location with EVOSKILL_ENV_FILE)")
    _lines = [ln for ln in _envfile.read_text().splitlines()
              if ln.strip().startswith("OPENROUTER_API_KEY=")]
    if not _lines:
        raise SystemExit(f"{_envfile} has no OPENROUTER_API_KEY= line")
    _key = _lines[-1].split("=", 1)[1].strip().strip('"').strip("'")
    if not _key:
        raise SystemExit(f"{_envfile} has an empty OPENROUTER_API_KEY=")
    os.environ["OPENROUTER_API_KEY"] = _key
    _WHICH_KEY = "env-file"
os.environ["EVOSKILL_SANDBOX"] = "none"

PHOENIX_PROJECT = os.environ.get(
    "PHOENIX_PROJECT_NAME",
    f"tau3-{TAG}-{_dt.datetime.now().strftime('%m%d-%H%M')}")


def refuse(msg: str) -> None:
    raise SystemExit(f"REFUSING: {msg}")


def preflight() -> tuple[dict, list[str]]:
    """Every check that can fail cheaply, before a single episode is spent."""
    if DOMAIN not in _DOMAINS:
        refuse(f"unknown EVOSKILL_TAU3_DOMAIN {DOMAIN!r}; known: {list(_DOMAINS)}")
    if DOMAIN == _RETRIEVAL_DOMAIN and not VARIANT:
        refuse(
            "EVOSKILL_TAU3_VARIANT is unset. This runner will not pick a retrieval\n"
            "  variant for you: banking_knowledge ships 21, they grant different TOOLS,\n"
            "  and the variant is NOT in the cache key, so an unpinned run is both an\n"
            "  unchosen instrument and a cache-collision risk. Pass one explicitly,\n"
            "  e.g. EVOSKILL_TAU3_VARIANT=bm25_grep (LOCAL, zero spend, deterministic).\n"
            "  `scripts/tau3_variant_probe.py` lists all 21 with their requirements.")
    if DOMAIN != _RETRIEVAL_DOMAIN and VARIANT:
        refuse(
            f"EVOSKILL_TAU3_VARIANT={VARIANT!r} was passed with domain {DOMAIN!r}, but\n"
            f"  retrieval variants exist ONLY for {_RETRIEVAL_DOMAIN}. Verified in\n"
            f"  tau2/runner/build.py: `env_kwargs['retrieval_variant']` is populated only\n"
            f"  when a retrieval_config is given, and {DOMAIN}'s constructor takes no such\n"
            f"  argument. Accepting it would record a retrieval_variant in the result JSON\n"
            f"  for a run in which retrieval did nothing. Unset it.")
    if not SPLIT_FILE.exists():
        refuse(f"no split at {SPLIT_FILE} — run scripts/make_tau3_banking_split.py")
    split = json.loads(SPLIT_FILE.read_text())
    # THE SPLIT FILE NAMES ITS OWN DOMAIN — CHECK IT. Every split this project builds
    # records `domain` (make_tau3_banking_split.py, make_tau3_domain_smoke_split.py), and
    # nothing read it. A mismatch is caught only INCIDENTALLY, by id-resolution failing,
    # which works while id spaces are disjoint — banking uses task_NNN. But AIRLINE ids are
    # 0..49 and RETAIL ids are 0..113, both BARE NUMERICS, so an airline split handed to a
    # retail run resolves cleanly against the wrong pool and produces a result JSON that
    # names retail while having run airline tasks. Nothing downstream could detect that.
    _sd = split.get("domain")
    if not _sd:
        refuse(f"{SPLIT_FILE.name} records no 'domain'. Every split this project builds "
               f"does, so an absent field means a hand-edited or foreign file — refusing "
               f"to guess which task pool it belongs to.")
    if _sd != DOMAIN:
        refuse(f"split {SPLIT_FILE.name} is for domain {_sd!r} but this run is {DOMAIN!r}. "
               f"airline ids are 0..49 and retail's are 0..113 — both bare numerics — so a "
               f"mismatched split can resolve ENTIRELY against the wrong pool and run the "
               f"wrong tasks under a result JSON that names this domain.")


    # Only these three are task-id lists. An `isinstance(list)` filter also matched
    # "bins" (a list of bin labels) and so offered it as a valid split key — a wrong
    # suggestion in an error message is how someone runs the wrong thing next.
    SPLIT_KEYS = ("train", "val", "test")
    keys = [k.strip() for k in SPLIT_KEY.split(",") if k.strip()]
    unknown = [k for k in keys if k not in SPLIT_KEYS or k not in split]
    if unknown:
        refuse(f"unknown split key(s) {unknown}; have {list(SPLIT_KEYS)}")
    want: list[str] = []
    for k in keys:
        want.extend(split[k])
    if len(set(want)) != len(want):
        refuse(f"split keys {keys} overlap — a task would be run twice")

    # THE VARIANT CHECK RUNS IN THE WORKER'S OWN INTERPRETER: `uv run --project
    # <TAU3_ROOT>` — the exact command Tau2Solver spawns for every episode. An earlier
    # version probed third_party/tau2-bench/.venv with PYTHONPATH=<TAU3_ROOT>/src: a
    # DIFFERENT interpreter with a DIFFERENT installed set. On 2026-08-17 that mismatch
    # false-passed a whole leg: rank_bm25 is a MAIN dependency of the v1.0.0 engine but
    # upstream later moved it into the optional `knowledge` extra, so the worker's venv
    # (implicitly synced from main deps only) lacked it while the probed venv had it —
    # env construction succeeded here and all 9 episodes then died at the first
    # retrieval tool call. A preflight that runs in a different interpreter than the
    # thing it preflights is measuring the wrong instrument.
    if not shutil.which("uv"):
        refuse("`uv` is not on PATH — the tau2 worker is launched via `uv run` and "
               "cannot start without it")
    # DOMAIN-PARAMETERISED. Previously both the RETRIEVAL_VARIANTS import and the
    # `get_env_constructor('banking_knowledge')` call were literals, so this preflight
    # would have "verified" banking's env while the leg then ran airline's.
    if DOMAIN == _RETRIEVAL_DOMAIN:
        probe = (
            "import json,sys\n"
            f"from tau2.domains.{DOMAIN}.retrieval import RETRIEVAL_VARIANTS\n"
            f"v={VARIANT!r}\n"
            "if v not in RETRIEVAL_VARIANTS:\n"
            "    print('UNKNOWN:'+','.join(sorted(RETRIEVAL_VARIANTS))); sys.exit(3)\n"
            "from tau2.registry import registry\n"
            "try:\n"
            f"    registry.get_env_constructor({DOMAIN!r})(retrieval_variant=v)\n"
            # Constructing the env does NOT import the retrieval stack — bm25's
            # rank_bm25 import is lazy, inside the first tool call (bm25_indexer.py),
            # which is exactly how the 2026-08-17 false-pass slipped through. Import it
            # eagerly here so a missing optional dep refuses before any spend.
            "    if 'bm25' in v:\n"
            "        from tau2.knowledge.document_preprocessors import bm25_indexer  # noqa: F401\n"
            "except Exception as e:\n"
            "    print('NOCONSTRUCT:%s: %s' % (type(e).__name__, e)); sys.exit(4)\n"
            "print('OK')\n")
    else:
        # No retrieval knob to check; what matters is that the env AND its tools
        # construct. `get_tools()` is included deliberately — the domain audit found
        # that constructing an env can succeed while tool construction fails, and the
        # solver needs the tools.
        probe = (
            "import json,sys\n"
            "from tau2.registry import registry\n"
            "try:\n"
            f"    e = registry.get_env_constructor({DOMAIN!r})()\n"
            "    n = len(e.tools.get_tools())\n"
            "except Exception as ex:\n"
            "    print('NOCONSTRUCT:%s: %s' % (type(ex).__name__, ex)); sys.exit(4)\n"
            "if n == 0:\n"
            "    print('NOCONSTRUCT:ValueError: env built but exposes 0 tools'); sys.exit(4)\n"
            "print('OK')\n")
    # No PYTHONPATH override: the worker gets none either — its `uv run --project`
    # venv has the tau2 package installed from <TAU3_ROOT> itself.
    r = subprocess.run(["uv", "run", "--project", str(TAU3_ROOT), "python", "-c", probe],
                       capture_output=True, text=True, env=dict(os.environ), timeout=300)
    out = (r.stdout or "").strip().splitlines()
    tail = out[-1] if out else ""
    if r.returncode == 3:
        refuse(f"unknown retrieval variant {VARIANT!r}; known: "
               f"{tail.split(':', 1)[1] if ':' in tail else '?'}")
    if r.returncode == 4:
        refuse(f"variant {VARIANT!r} will not CONSTRUCT here — "
               f"{tail.split(':', 1)[1].strip() if ':' in tail else tail}\n"
               f"  (banking_knowledge: 6 of 21 variants need OPENAI_API_KEY and/or a docker\n"
               f"  sandbox; this machine has no OPENAI_API_KEY. See tau3_variant_probe.py.\n"
               f"  A ModuleNotFoundError here means the worker venv is missing an optional\n"
               f"  extra: `uv sync --project {TAU3_ROOT} --extra knowledge` installs\n"
               f"  rank_bm25 for the bm25 retrieval variants.)")
    if r.returncode != 0 or tail != "OK":
        refuse(f"variant preflight failed (rc={r.returncode}): "
               f"{(r.stderr or '')[-400:]}")
    print(f"  preflight  : {DOMAIN} env"
          + (f" + variant {VARIANT!r}" if VARIANT else " (no retrieval)")
          + "  constructs in the tau2 venv  [OK]")
    return split, want


async def main() -> int:
    split, want = preflight()

    from src.tracing import init_tracing
    init_tracing(PHOENIX_PROJECT)

    from src.harness.sdk_config import set_sdk
    from src.evaluation.evaluate import evaluate_agent_parallel
    from src.cache import RunCache, CacheConfig
    from src.agent_profiles.tau2_agent.dataset import load_tau2
    from src.agent_profiles.tau2_agent.solver import Tau2Solver
    from src.harness.rollout_ledger import failure_ledger, summarise

    set_sdk("direct")

    workspace = PROJECT_ROOT / ".cache" / f"evo_workspace_{TAG}"
    workspace.mkdir(parents=True, exist_ok=True)

    # THE SOLVER READS SKILLS FROM THE WORKSPACE, NOT FROM EVOSKILL_SKILLS_DIR.
    # `Tau2Solver._load_skills` globs `<workspace>/.claude/skills/**/*.md` — the env var
    # reaches nothing on its own. The first version of this runner passed SKILLS_DIR only
    # to `CacheConfig(live_skills_dir=...)`, so a skill leg would have been a SILENT BASE
    # LEG, and the "refuse if empty" check above it validated a directory the solver never
    # opens — false confidence, which is worse than no check. Exactly the failure class
    # tau3_wiring_smoke.py exists to catch, reproduced here. So: stage the skills INTO the
    # workspace and then verify what the solver will actually load.
    skills_dir = workspace / ".claude" / "skills"
    if skills_dir.exists():
        # A base leg must PROVE it has no skills. The workspace is per-tag and persists
        # between runs, so re-running a tag that once carried skills would otherwise
        # inherit them and silently stop being a base leg.
        shutil.rmtree(skills_dir)
    skills_dir.mkdir(parents=True, exist_ok=True)
    if SKILLS_DIR:
        src = Path(SKILLS_DIR)
        if not src.is_dir() or not any(src.rglob("*.md")):
            refuse(f"EVOSKILL_SKILLS_DIR={SKILLS_DIR} is missing or has no *.md. An "
                   f"empty skills dir silently makes a skill leg identical to base.")
        for child in sorted(src.iterdir()):
            dst = skills_dir / child.name
            shutil.copytree(child, dst) if child.is_dir() else shutil.copy2(child, dst)

    # Ask the SOLVER'S OWN loader what it sees. Anything less is trusting the copy.
    from src.agent_profiles.tau2_agent.solver import _load_skills
    loaded = _load_skills(workspace)
    if SKILLS_DIR and not loaded:
        refuse(f"staged {SKILLS_DIR} into {skills_dir} but the solver's own "
               f"_load_skills() found ZERO skills there. A skill leg would score as "
               f"base. Expected layout: <dir>/<skill-name>/SKILL.md")
    if not SKILLS_DIR and loaded:
        refuse(f"this is a BASE leg but _load_skills() found {len(loaded)}: "
               f"{sorted(loaded)}")

    # Select the frozen split by task_id, preserving split-file order so sample_idx
    # alignment with any earlier leg is stable.
    tr, va, te = load_tau2(domain=DOMAIN, tau2_root=TAU3_ROOT)
    by_id = {}
    for q, gt, cat in list(tr) + list(va) + list(te):
        by_id[json.loads(q)["task_id"]] = (q, gt, cat)
    missing = [t for t in want if t not in by_id]
    if missing:
        refuse(f"{len(missing)} split id(s) absent from the loaded pool: {missing[:5]}")
    items_all = [by_id[t] for t in want]
    if N_TASKS:
        items_all = items_all[:N_TASKS]
    items = [(q, gt) for q, gt, _c in items_all]
    tids = [json.loads(q)["task_id"] for q, _gt in items]

    solver = Tau2Solver(
        model=SOLVER_MODEL, workspace=workspace, domain=DOMAIN,
        retrieval_config=VARIANT, tau2_root=TAU3_ROOT,
        episode_timeout=EPISODE_TIMEOUT,
        # Each tier passed ONLY if explicitly set, so an unset tier keeps the solver's
        # own default rather than inheriting the other tier's value.
        **({"reference_user_model": USER_MODEL_REFERENCE} if USER_MODEL_REFERENCE else {}),
        **({"cheap_user_model": USER_MODEL_CHEAP} if USER_MODEL_CHEAP else {}))

    print(f"TAU3 {TAG} | domain={DOMAIN} variant={VARIANT}")
    print(f"  split      : {SPLIT_FILE.name} key={SPLIT_KEY} "
          f"-> {len(items)} tasks x N={N_SAMPLES} = {len(items)*N_SAMPLES} episodes")
    print(f"  tau3 root  : {Path(TAU3_ROOT).name}  (version "
          f"{split.get('tau_version', '?')} per the split file)")
    print(f"  solver     : {SOLVER_MODEL}")
    print(f"  user-sim   : {solver.reference_user_model} / {solver.cheap_user_model}")
    print(f"  judge      : {solver.judge_model}")
    print(f"  self-play  : {solver.self_play_flags or '(none)'}")
    print(f"  skills     : {len(loaded)} loaded by the SOLVER'S loader"
          + (f" -> {sorted(loaded)}" if loaded else "  (BASE leg — verified empty)"))
    if SKILLS_DIR:
        print(f"               staged from {SKILLS_DIR}")
    # Read from the ENVIRONMENT, not from a local that only one branch binds. The
    # exported-key path sets _WHICH_KEY without ever assigning _key, so this line
    # raised NameError on exactly the path the fix above was written to enable.
    # Mirrors run_ssbv2_base30_dsv4f_n3.py:420, which prints the served key and its source.
    print(f"  [KEY] OpenRouter key ...{os.environ['OPENROUTER_API_KEY'][-4:]} ({_WHICH_KEY})")
    # Pre-flight, BEFORE any episode is paid for: a pin that turns out to be inert or
    # preference-only after a 150-episode leg is an unusable leg. Raises SystemExit.
    _assert_provider_pin_in_force(solver)
    print(f"  phoenix    : {PHOENIX_PROJECT}")
    print(f"  concurrency: {CONCURRENCY}  episode_timeout={EPISODE_TIMEOUT}s", flush=True)
    # PRINT THE WHOLE WALL STACK, not just the inner knob. #175: the inner wall is
    # only observable at a turn boundary, so `episode_timeout=1800s` alone told a
    # reader nothing about whether it could ever fire — and for three episodes it
    # could not. Anyone reading this banner should be able to check the ordering
    # without opening solver.py.
    _wb = solver._wall_budget
    print(f"  walls      : inner {_wb['inner_episode_timeout_s']}s (checked at TURN "
          f"BOUNDARIES only) | worst-case turn {_wb['worst_case_turn_s']:.0f}s "
          f"({_wb['llm_retries']}+1 x {_wb['llm_per_call_s']:.0f}s LLM) "
          f"| outer {_wb['outer_wall_s']}s = inner + worst-turn + "
          f"{_wb['post_sim_margin_s']}s post-sim", flush=True)

    # scores[task_id] = [per-sample score or None]
    scores: dict[str, list] = {t: [None] * N_SAMPLES for t in tids}
    errors: dict[str, list] = {t: [None] * N_SAMPLES for t in tids}
    cost = 0.0
    cost_unpriced = 0      # episodes whose trace carried no finite cost
    # PER-ROLE COST, accumulated separately (user-flagged 2026-08-06). The leg total alone
    # cannot answer "what is the user simulator costing us?", which is precisely the
    # question the gpt-4.1-vs-gpt-5-nano decision turns on — and the question a previous
    # session got wrong by 3x (0804b K3: claimed 59%, measured 18.7%) because it scaled one
    # episode's ratio to a whole leg instead of summing every episode.
    cost_agent = cost_user = cost_judge = 0.0
    cost_split_seen = 0
    cost_paid = 0.0
    cost_replayed = 0.0
    cost_split_missing = 0
    cost_split_errs: list[str] = []
    # Counter, not dict: every episode must land in SOME bucket so the buckets
    # sum to `episodes`. A wall-kill rate computed over a partial denominator
    # understates itself.
    term_reasons: collections.Counter[str] = collections.Counter()
    # The (task_id, sample) of every episode dropped for INFRASTRUCTURE_ERROR — the only
    # reason tau2's own metric excludes (agent_metrics.py:139). Kept as a LIST of
    # identities, not just a count, so the result JSON names exactly which episodes left
    # the denominator; a bare count would make the exclusion unauditable.
    infra_excluded: list[tuple[str, int]] = []
    t0 = time.monotonic()

    # ── #184: POOLED DISPATCH — every sample of every task in ONE concurrency pool ──
    # Was: N sequential `await evaluate_agent_parallel(...)` passes, one per sample index.
    # At any concurrency >= len(items) each pass runs all its tasks at once, so a pass
    # costs its SLOWEST episode rather than a sum — and three sequential passes pay that
    # worst case three times. Measured on the 60 base episodes: median 10.7 min, p90 28.4,
    # max 39.0. So sequential ~= 117 min worst case against ~39 pooled: a ~3x saving, not
    # the tail-trim it looks like at low concurrency.
    #
    # WHY THIS IS SAFE, both checked in source before changing anything:
    #   1. get_task_gate() in dynamic mode returns ONE process-wide DynamicAdmission
    #      shared by every wave (its own docstring says the runner's gate and
    #      evaluate_agent_parallel's per-call gate "must draw from the same pool"), so
    #      N concurrent passes do NOT multiply concurrency by N — they share the target.
    #   2. `workspace`/`skills_dir` are staged ONCE above this point and only read here,
    #      so overlapping passes cannot race on the staged skills.
    # The cache keys cannot collide either: each sample keeps its OWN dir, which is the
    # very reason the sequential shape existed (evaluate_agent_parallel takes no
    # sample_idx, so N copies of one item in ONE call would collapse to 1 episode + N-1
    # replays). Pooling by dir sidesteps that without touching the shared fan-out and its
    # hardened retry/exception accounting.
    #
    # NOT AN INSTRUMENT CHANGE: what decides an episode is agent clean time, agent turns
    # and the inactivity watchdog — all contention-immune by construction, which is why
    # that metric was adopted. Scheduling moves wall-clock, not verdicts.
    _caches = []
    # #194: RECORD THE DIRS THAT WERE ACTUALLY USED, never a re-derivation of the name.
    # The result JSON's `cache_dirs` field used to rebuild the name from scratch at write
    # time, and that copy of the expression was never updated when the leg token landed
    # (the #148 fix). So every leg -- base, v1, v2, v3 -- recorded the OLD SHARED bucket
    # name, i.e. the exact collision the leg token exists to prevent, while the real dirs
    # were correctly separate. An auditor reading the JSONs would have concluded all four
    # generations shared one cache. Same shape as the vacuous #148 fix and the "0 flips"
    # count: computing from a re-derivation instead of from the thing itself.
    _cdirs_used = []
    for s in range(N_SAMPLES):
        # ONE CACHE DIR PER SAMPLE — see the module docstring. The variant is in the
        # name because it is not in the key.
        # OPTIONAL RUN TOKEN IN THE CACHE DIR NAME. The #183 identity guard refuses a
        # dir whose records were built by a different instrument and tells the operator to
        # "point this run at a fresh cache dir" -- which, until this existed, the code gave
        # no way to do: the name was a pure function of domain+variant+sample. So the
        # guard's own remedy required editing source. It is OPTIONAL and empty by default,
        # so every existing dir name is unchanged and no record is orphaned; set it when
        # the instrument genuinely changed (e.g. the 2026-08-06 user-sim switch to
        # gpt-5-nano invalidated every banking record at once).
        _ctag = os.environ.get("EVOSKILL_TAU3_CACHE_TAG", "").strip()
        _ctok = f"_{_ctag}" if _ctag else ""
        # LEG TOKEN IN THE DIR NAME (#148). Until 2026-08-07 every leg of a run -- base,
        # skills_v1, skills_v2 ... -- shared ONE dir per sample and was separated only by
        # `tree_hash[:12]` BUCKET inside it. That separation is real but fragile in one
        # specific direction: an EMPTY or MISSING skills dir hashes to exactly the BASE
        # leg's tree_hash, so a mislaunched skill leg writes silently INTO base's bucket.
        # That is not hypothetical -- on 2026-08-04 the SSB v2 audit found 8 skills_v4
        # rollouts inside base's bucket, counted in base's published 78/360, erasing base's
        # entire ALL@3 lead. And the obvious guard cannot catch it: the bucket dir IS
        # tree_hash[:12] and each record carries that same truncated value, so "one
        # tree_hash per bucket" is true by construction (#154). Record COUNT does not catch
        # it either -- a foreign write lands on a slot the leg never ran, so the count
        # staying right is exactly what makes it invisible.
        #
        # Putting the leg in the DIRECTORY makes the collision structurally impossible
        # instead of merely absent, which is the only kind of guarantee worth having for a
        # base leg -- it is the arm everything is measured against.
        #
        # Cost: this renames the dirs, so records under the old shared name are not found
        # and a fresh base leg must be paid for. ~$53 for banking 50 tasks x 3. Accepted
        # deliberately: #190 destroyed a baseline four days ago for exactly this class of
        # shortcut, and recovering it was luck, not design.
        # MUST derive from SKILLS_DIR (the CONFIGURED source, line ~356), never from the local
        # `skills_dir`. The first version of this used `skills_dir` and the smoke test caught it
        # immediately: `skills_dir` is `workspace/.claude/skills` (line ~551), the STAGING path
        # the solver's loader reads, which exists and is named "skills" on EVERY leg including
        # base. So every leg hashed to the token "skills" and the fix separated nothing -- a
        # guard that looks like isolation and delivers none, i.e. the #154 vacuous shape this
        # very change was written to eliminate. Verified after the fix: base -> "base",
        # skills_v1 -> "skills-v1".
        _leg = "base" if not SKILLS_DIR else Path(str(SKILLS_DIR)).name
        _leg = "".join(c if c.isalnum() else "-" for c in _leg).strip("-").lower() or "base"
        if SKILLS_DIR and _leg == "base":
            raise SystemExit(f"REFUSING: EVOSKILL_SKILLS_DIR={SKILLS_DIR!r} reduced to the leg "
                             f"token 'base', which would put a SKILL leg in the base leg's cache "
                             f"dir -- the exact #148 collision. Rename the skills dir.")
        cdir = PROJECT_ROOT / ".cache" / f"tau3_{DOMAIN}_{_VTOK}{_ctok}_{_leg}_s{s}_cache"
        _cdirs_used.append(cdir.name)        # #194 — the real name, captured here
        _assert_cache_identity(cdir, solver)
        # TRANSCRIPT BUDGET (#149). `cached_transcript` is the ONLY usable
        # record of an episode after the write — `messages` does not survive
        # Pydantic round-tripping and is stripped anyway — and it is the entire
        # evidence surface the evolver reasons from. banking_knowledge's
        # retrieval tool returns TEN full policy documents per call, so at the
        # library defaults (20K per tool result, 200K per transcript) 26 of the
        # 55 rollouts already on disk lost their ENDING: median 40% of turns
        # gone, worst case 79 of 86 turns on a rollout that scored 1.0. The
        # library now truncates head+tail instead of head-only, but for this
        # benchmark the right answer is not to truncate at all: the cache is
        # the raw evidence and `tau3_build_digests.py` is the curated view, so
        # the compression belongs in the digest (which can collapse a retrieval
        # result to its ranked doc list) and not in the store, which cannot.
        # Cost of storing it all, measured from the existing records: ~330 KB
        # per episode on average, ~31 MB for a 32x3 leg. Cheap.
        cache = RunCache(CacheConfig(cache_dir=cdir, enabled=True, cwd=workspace,
                                     live_skills_dir=skills_dir,
                                     transcript_tool_result_max_chars=60_000,
                                     transcript_max_chars=3_000_000))
        print(f"  --- sample {s+1}/{N_SAMPLES}  cache={cdir.name} ---", flush=True)
        _caches.append(cache)

    print(f"\n  --- dispatching {len(items)} tasks x {N_SAMPLES} samples = "
          f"{len(items)*N_SAMPLES} episodes into ONE pool (#184) ---", flush=True)
    _all_results = await asyncio.gather(*[
        evaluate_agent_parallel(agent=solver, items=items, max_concurrent=CONCURRENCY,
                                cache=_caches[s], tag_prefix=f"{TAG}-s{s+1}")
        for s in range(N_SAMPLES)])

    # Accounting stays SEQUENTIAL and byte-identical to before: only the dispatch above
    # changed. Every hardened path below (NaN-safe cost, paid-vs-replayed split, the
    # per-role breakdown, the failure ledger) runs exactly as it did.
    for s in range(N_SAMPLES):
        results = _all_results[s]
        for tid, r in zip(tids, results):
            tr_ = getattr(r, "trace", None)
            sc = getattr(tr_, "inline_score", None) if tr_ else None
            is_err = bool(getattr(tr_, "is_error", False)) if tr_ else True
            c = getattr(tr_, "total_cost_usd", None) if tr_ else None
            # `float(c or 0.0)` propagates NaN, because NaN is TRUTHY — one trace with
            # an unpriced model turned the whole leg's cost into `$nan` on the first
            # smoke. Cost is a reported number (standing directive: snapshot spend per
            # experiment), so a single unpriced episode must not destroy the total.
            if c is not None and math.isfinite(float(c)):
                cost += float(c)
                # PAID vs REPLAYED (cost-b). `cost_usd` sums EVERY trace, including ones
                # served from cache, so a relaunch re-counts spend that was already paid
                # and any total across legs double-counts. Measured 2026-08-06: the e3
                # train re-run executed exactly ONE fresh episode of 15 yet reported
                # $2.0461 against the original's $1.9386 -- 89.8% of its "cost" was replay.
                # RunCache marks replays (`trace.from_cache`, run_cache.py:705), so the
                # split is free to compute and it is the number that answers "what did
                # this leg actually spend".
                if getattr(tr_, "from_cache", False):
                    cost_replayed += float(c)
                else:
                    cost_paid += float(c)
            else:
                cost_unpriced += 1
            # The per-role split lives in the episode payload's `cost_usd` dict, which the
            # solver already builds from tau2's own call names. Same NaN discipline as the
            # total above: a single unpriced role must not poison the leg's split.
            # TWO WRONG SHAPE ASSUMPTIONS IN ONE EXPRESSION, hidden by a bare except.
            # `trace.output` is a pydantic AgentResponse, not a dict, on EVERY path this
            # runner sees (fresh, and replayed too -- evaluate.py passes response_model to
            # cache.get, which model_validates it back). AgentResponse is truthy, so the
            # `or {}` never fired, and `.get` raised AttributeError straight into
            # `except Exception: pass`. Measured over 64 real cache records: the old
            # expression scored 0 hits, this one scores 64.
            #
            # The right idiom was already 18 lines below this block (line ~508 uses
            # getattr(getattr(tr_, "output", None), "final_answer", None)) -- the file held
            # the correct and the broken form side by side, and only the broken one was
            # unexercised, because its sole failure signal was a counter reading 0 and the
            # stdout report was gated on `if cost_split_seen:` -- so total failure printed
            # NOTHING. That is why the miss path below is now loud.
            try:
                _out_c = getattr(tr_, "output", None)
                _fa = (_out_c.get("final_answer") if isinstance(_out_c, dict)
                       else getattr(_out_c, "final_answer", None))
                _cs = (json.loads(str(_fa)) or {}).get("cost_usd") or {}
                _a, _u, _j = (_cs.get("agent"), _cs.get("user"), _cs.get("judge"))
                if all(x is not None and math.isfinite(float(x)) for x in (_a, _u, _j)):
                    cost_agent += float(_a); cost_user += float(_u); cost_judge += float(_j)
                    cost_split_seen += 1
                else:
                    cost_split_missing += 1
                    if len(cost_split_errs) < 5:
                        cost_split_errs.append(f"{tid} s{s}: no finite agent/user/judge in {_cs!r}")
            except Exception as _e:
                cost_split_missing += 1
                if len(cost_split_errs) < 5:
                    cost_split_errs.append(f"{tid} s{s}: {type(_e).__name__}: {_e}")
            # TERMINATION REASON — the infra-vs-real discriminator, and the one
            # number this leg most needs. Measured 2026-08-04: at the 460s
            # default wall, 3 of 7 episodes died AT the wall (462/463/466s) and
            # every one scored 0.0. Pooling those with wrong answers reports the
            # CLOCK as if it were capability, and it silently feeds the evolver
            # timeout traces to theorise about. tau2 puts the reason inside the
            # episode result JSON that becomes `final_answer`; read it from
            # there. Best-effort by design — a parse failure must never cost the
            # leg its scores, so it lands in an `unparsed` bucket instead of
            # raising, and an absent reason is counted as `absent` rather than
            # quietly dropped (an uncounted episode would understate the
            # wall-kill rate, which is the exact error this guards against).
            _fa = getattr(getattr(tr_, "output", None), "final_answer", None)
            _reason = None
            if _fa is None:
                _reason = "no_trace" if tr_ is None else "absent"
                term_reasons[_reason] += 1
            else:
                try:
                    _reason = json.loads(str(_fa)).get("termination_reason") or "absent"
                    term_reasons[_reason] += 1
                except (ValueError, TypeError, AttributeError):
                    _reason = "unparsed"
                    term_reasons["unparsed"] += 1
            # ---- exclude ONLY what tau2 itself excludes ------------------------
            # RETRACTION, 2026-08-06. I first wrote this branch to exclude
            # `termination_reason == "timeout"`, on the reasoning that a wall-kill measures
            # the clock rather than capability. Reading tau2's own code refuted that:
            #
            #   evaluator/evaluator.py:119 — `evaluate()` returns reward=0.0 for ANY
            #   termination_reason outside {AGENT_STOP, USER_STOP}, TIMEOUT included, with
            #   the note "Simulation terminated prematurely".
            #   metrics/agent_metrics.py:139 — tau2's own metric drops exactly ONE reason
            #   from the denominator: INFRASTRUCTURE_ERROR. TIMEOUT stays in.
            #
            # So a timed-out episode scoring 0.0 is the BENCHMARK'S RULE, not a defect
            # here. Excluding it would have made every tau3 number we report incomparable
            # with published tau2/tau3 results, and it would have moved them UPWARD — a
            # non-standard adjustment in our own favour. #152's concern is real but it
            # belongs at the DESIGN level: `episode_timeout` is a value WE choose, so the
            # remedy is to set it generously and report the rate, never to silently rescore.
            #
            # What IS owed is parity with tau2's one exclusion. INFRASTRUCTURE_ERROR means
            # the harness broke (API disconnect), which is not an answer at all.
            # WHY A LEDGER FAILURE AND NOT A SILENT EXCLUSION. tau2 excludes these from its
            # metric and moves on, but this runner already holds a stricter line: "a leg
            # with failed episodes is not a measurement" — it writes a PARTIAL, refuses the
            # canonical JSON, and relaunching replays every completed episode from cache and
            # retries only the broken ones. For an API disconnect, RETRYING is strictly
            # better than dropping the episode, and it costs one relaunch. Recording it in
            # `errors` (not just `scores`) is also what stops the ledger's SOURCE 2 from
            # mislabelling the None as "logic hole in the rollout coroutine".
            if _reason == "infrastructure_error":
                infra_excluded.append((tid, s))
                scores[tid][s] = None
                errors[tid][s] = ("infrastructure_error: tau2 terminated the simulation on "
                                  "an infrastructure fault (e.g. API disconnect), which is "
                                  "not an answer. tau2's own metric drops these "
                                  "(agent_metrics.py:139); this leg goes further and "
                                  "refuses to report — relaunch to retry just this episode.")
            elif not is_err:
                scores[tid][s] = float(sc or 0.0)
            else:
                # CARRY THE REASON, NOT JUST THE LABEL.
                #
                # `Tau2Solver._error_trace` puts a real diagnosis into `result` and
                # `parse_error` — "Worker subprocess exited N with no output. stderr
                # tail: …", "Worker produced non-JSON stdout: …", "Episode timed out
                # after Ns (OUTER wall …)" — and the first version of this line
                # recorded the string "is_error" and dropped all of it.
                #
                # That is not a cosmetic loss. On 2026-08-04 this leg reported
                # `3/96 EPISODES FAILED … 3 x RuntimeError: is_error` and the trace
                # carrying the cause was ALSO refused by the cache (correctly — a
                # transient must not be cached), so the only copy of the reason was
                # the one this line discarded. The leg made it structurally
                # impossible to learn why it failed, which is the same class of
                # defect as an evidence surface that lies: a failure I cannot
                # diagnose I can only guess about, and guessing is how a transient
                # gets relaunched into the same wall three times.
                if tr_ is None:
                    errors[tid][s] = "no_trace"
                else:
                    why = (getattr(tr_, "result", None)
                           or getattr(tr_, "parse_error", None) or "")
                    why = " ".join(str(why).split())      # collapse the stderr tail
                    errors[tid][s] = (f"is_error: {why[:4000]}" if why
                                      else "is_error (the trace carried no message — "
                                           "check Tau2Solver._error_trace call sites)")

    elapsed = time.monotonic() - t0

    # ---- ledger: refuse to emit a result JSON with holes in it ----------------
    idx = [(i, s) for i in range(len(tids)) for s in range(N_SAMPLES)]
    flat_scores = {i: scores[t] for i, t in enumerate(tids)}
    res_for_ledger = [
        RuntimeError(errors[tids[i]][s]) if errors[tids[i]][s] else None
        for i, s in idx
    ]
    failures = failure_ledger(idx, res_for_ledger, flat_scores)

    # The DENOMINATOR is the number of episodes that produced a score, not the number
    # dispatched: `n = len(tids) * N_SAMPLES` was a constant, so an excluded episode would
    # still have counted as 0/1. `n_dispatched` is reported so any drop is visible rather
    # than merely absent. NOTE this changes nothing for a leg with no infrastructure errors,
    # which is the normal case — and TIMEOUT is deliberately NOT excluded (see the scoring
    # site for why: it is tau2's own rule).
    n_dispatched = len(tids) * N_SAMPLES
    flat = [x for v in scores.values() for x in v if x is not None]
    n = len(flat)
    npass = sum(1 for x in flat if x >= 0.999)
    # ALL@N is "every VALID sample passed, and at least one was valid". Requiring all N to
    # be present would let a single excluded episode turn a 3-for-3 task into a failure.
    all_n = sum(1 for v in scores.values()
                if any(x is not None for x in v)
                and all(x >= 0.999 for x in v if x is not None))
    pass_n = sum(1 for v in scores.values()
                 if any(x is not None and x >= 0.999 for x in v))
    # A task with EVERY sample wall-killed contributes nothing and must not masquerade as
    # a failed task in a task-paired comparison.
    tasks_unscored = [t for t, v in scores.items() if all(x is None for x in v)]

    meta = {
        "tag": TAG, "domain": DOMAIN,
        # RECORDED because tree_hash cannot see it, exactly like solver_image on SSB v2.
        "retrieval_variant": VARIANT,
        "tau_version": split.get("tau_version"),
        "tau3_root": TAU3_ROOT,
        "split_file": str(SPLIT_FILE), "split_key": SPLIT_KEY,
        "split_seed": split.get("seed"), "stratified_on": split.get("stratified_on"),
        # `episodes` means DISPATCHED — the intuitive reading, and what termination_reasons
        # sums to. `episodes_scored` is the denominator of pass/total. Before this split the
        # single `episodes` field silently changed meaning to "scored" when the denominator
        # was fixed, so a leg with one excluded episode reported "episodes: 14" for a 15-episode
        # run: the same provenance trap as recording the derived wall instead of the one in force.
        "n_tasks": len(tids), "n_samples": N_SAMPLES,
        "episodes": n_dispatched, "episodes_scored": n,
        "solver_model": SOLVER_MODEL,
        "user_model_reference": solver.reference_user_model,
        "user_model_cheap": solver.cheap_user_model,
        "judge_model": solver.judge_model,
        "episode_timeout_s": EPISODE_TIMEOUT, "concurrency": CONCURRENCY,
        # THE FULL WALL STACK (#175). `episode_timeout_s` alone is not the
        # instrument: the inner wall is only checked at turn boundaries, so what
        # actually bounds an episode is this triple. Recorded per leg so two legs
        # can be compared on their walls rather than on an assumption, and so the
        # 2026-08-06 legs (outer 2100s, ordering broken) are distinguishable from
        # later ones without reading the code that produced them.
        "wall_budget": solver._wall_budget,
        # PROVENANCE for #182: without this a reader cannot tell which legs were protected.
        "cache_skip_wall_limited": os.environ.get(
            "EVOSKILL_CACHE_SKIP_WALL_LIMITED", "") not in ("", "0"),
        # ROUTING IS PART OF THE INSTRUMENT. Before 2026-08-06 the tau3 episode path sent
        # NO provider block, so DeepSeek rode whichever of ~23 endpoints OpenRouter chose;
        # now it is hard-pinned. That is a real change in what produced the numbers, and
        # #182 taught the lesson the expensive way: an instrument field absent from the
        # artifact cannot be used to date a leg afterwards.
        "provider_pin": _solver_provider_pin(solver),
        "provider_pin_verified_preflight": True,
        "PROVIDER_PIN_CAVEAT": (
            "This is the pin REQUESTED, asserted in force pre-flight (order=['deepseek'] AND "
            "allow_fallbacks=False). The SERVED provider is still NOT recorded per episode "
            "(#169 open): that needs plumbing through the shared client layer used by all 6 "
            "harnesses. The residual risk is small because allow_fallbacks=False makes "
            "OpenRouter error rather than substitute -- but it is not zero and is not "
            "verified from the response."),
        "COST_CAVEAT": (
            "#158 OPEN. Every cost_* figure here is COMPUTED from token counts x list price, "
            "not metered from the provider. It was measured to OVER-REPORT by roughly 11x on "
            "tau3. Use these numbers for RELATIVE comparison between legs of this run only; do "
            "not quote them as spend, and do not reconcile them against the OpenRouter "
            "dashboard. The real fix is `usage:{include:true}` as the source of truth, which "
            "is a shared-client change and was deliberately NOT made immediately before a "
            "paid multi-hour run."),
        "skills_dir": str(SKILLS_DIR) if SKILLS_DIR else None,
        "skills_loaded": sorted(loaded),
        "cache_tag": os.environ.get("EVOSKILL_TAU3_CACHE_TAG", "") or None,
        # #194 — the dirs that were actually opened, captured in the loop above. NOT a
        # second copy of the naming expression: that copy silently omitted the leg token
        # and mislabelled every leg of the 25/25 run as sharing the base bucket.
        "cache_dirs": list(_cdirs_used),
        "elapsed_s": round(elapsed, 1), "cost_usd": round(cost, 4),
        # PER-ROLE SPLIT. `cost_split_episodes` is the denominator the shares are computed
        # over — reported so a partial split is visible as partial rather than read as the
        # whole leg (the 0804b K3 error was exactly a share taken over the wrong denominator).
        "cost_agent_usd": round(cost_agent, 4),
        "cost_user_sim_usd": round(cost_user, 4),
        "cost_judge_usd": round(cost_judge, 4),
        # Rounded to the SAME precision as `cost_usd` (round(cost,4) above) so the three
        # reconcile exactly. Mixed precision made paid+replayed differ from the total
        # by 4.6e-05 -- harmless here, but a reconciliation check is only useful if it
        # can be written as an equality rather than a tolerance someone has to pick.
        "cost_paid_usd": round(cost_paid, 4),
        "cost_replayed_usd": round(cost_replayed, 4),
        "cost_split_episodes": cost_split_seen,
        "cost_split_missing_episodes": cost_split_missing,
        "cost_split_errors": cost_split_errs,
        "cost_user_sim_pct": (round(100.0 * cost_user / (cost_agent + cost_user + cost_judge), 1)
                              if (cost_agent + cost_user + cost_judge) > 0 else None),
        "user_sim_model_reference": solver.reference_user_model,
        "user_sim_model_cheap": solver.cheap_user_model,
        # Reported so a low cost cannot be mistaken for a cheap run when it is
        # really an unpriced one.
        "episodes_without_cost": cost_unpriced,
        # Buckets SUM to `episodes` by construction — see the comment at
        # the extraction site. `scripts/tau3_scoreboard.py` reads this to
        # separate wall-kills from wrong answers; without it that split is
        # UNKNOWN, not zero.
        "termination_reasons": dict(sorted(term_reasons.items())),
        # Audit trail for the denominator. `total` below is the SCORED count, so these
        # fields are what let a reader reconstruct it instead of assuming episodes==total.
        "episodes_dispatched": n_dispatched,
        "episodes_excluded_infra": len(infra_excluded),
        "infra_excluded_episodes": [{"task_id": t, "sample": s} for t, s in infra_excluded],
        # Recorded, NOT excluded — tau2 scores these 0.0 itself. Reported so a leg whose
        # score is depressed by our own `episode_timeout` choice is visible as such.
        "episodes_timeout_scored_zero": int(term_reasons.get("timeout", 0)),
        "tasks_fully_unscored": tasks_unscored,
        "task_ids": tids, "per_task": scores,
    }

    print("\n" + "=" * 68)
    if failures:
        print(f"{len(failures)}/{n} EPISODES FAILED — this leg is INCOMPLETE.")
        for kind, c in sorted(summarise(failures).items(), key=lambda kv: -kv[1]):
            print(f"  {c:4d} x {kind}")
        for i, s, kind, msg in failures[:15]:
            print(f"    {tids[i]} sample {s+1}: {kind}: {msg}")
        part = OUT.with_suffix(".PARTIAL.json")
        part.parent.mkdir(parents=True, exist_ok=True)
        part.write_text(json.dumps({**meta, "INCOMPLETE": True,
                                    "n_failed": len(failures),
                                    "failed_by_kind": summarise(failures),
                                    # The PER-EPISODE reasons, durably. `failed_by_kind`
                                    # is a histogram of exception TYPES, which for this
                                    # runner is always RuntimeError and therefore tells
                                    # nobody anything. The reason strings are the only
                                    # record that survives, because an errored trace is
                                    # deliberately not cached.
                                    "failed_episodes": [
                                        {"task_id": tids[i], "sample": s,
                                         "kind": kind, "reason": msg}
                                        for i, s, kind, msg in failures],
                                    }, indent=1))
        print(f"\nwrote {part}")
        print("NOT writing the canonical result JSON: a leg with failed episodes is not")
        print("a measurement. Relaunch with identical env — the per-sample caches replay")
        print("every completed episode and retry only these.")
        print("=" * 68, flush=True)
        return 1

    # n can now be 0 (every episode wall-killed). Refuse to divide rather than crash after
    # a leg's worth of spend.
    if n == 0:
        print(f"TAU3 {TAG} | {DOMAIN}/{_VTOK} | NO SCORED EPISODES of {n_dispatched}. "
              f"This is an infra outcome, not a 0% score.")
        return 1
    print(f"TAU3 {TAG} | {DOMAIN}/{_VTOK} | {npass}/{n} = {npass/n:.1%}")
    if infra_excluded:
        print(f"  {len(infra_excluded)} of {n_dispatched} episodes EXCLUDED "
              f"(infrastructure_error, matching tau2's own metric) — denominator is {n}, "
              f"not {n_dispatched}.")
    _n_to = int(term_reasons.get("timeout", 0))
    if _n_to:
        print(f"  {_n_to} of {n_dispatched} episodes hit the {EPISODE_TIMEOUT}s "
              f"episode_timeout and are SCORED 0.0 — tau2's own rule "
              f"(evaluator.py:119), NOT excluded. If that share is large, raise "
              f"episode_timeout; do not rescore.")
    if infra_excluded or _n_to:
        if tasks_unscored:
            print(f"  {len(tasks_unscored)} task(s) have NO scored sample at all: "
                  f"{tasks_unscored} — they are absent from ALL@N/Pass@N, not counted as fails.")
    print(f"  ALL@{N_SAMPLES}={all_n}/{len(tids)}  Pass@{N_SAMPLES}={pass_n}/{len(tids)}")
    print(f"  {elapsed:.0f}s  ${cost:.2f}"
          + (f"  ({cost_unpriced}/{n} episodes reported NO cost)" if cost_unpriced else ""))
    print(f"  cost       : ${cost:.4f} total = ${cost_paid:.4f} PAID (fresh inference) "
          f"+ ${cost_replayed:.4f} replayed from cache. Only the PAID figure is new spend; "
          f"summing `cost_usd` across legs double-counts every replay.", flush=True)
    if cost_split_missing:
        print(f"  COST SPLIT UNAVAILABLE for {cost_split_missing}/{n_dispatched} episode(s) "
              f"— the per-role numbers below cover only {cost_split_seen}. First errors: "
              + "; ".join(cost_split_errs), flush=True)
    if cost_split_seen:
        _sp = cost_agent + cost_user + cost_judge
        print(f"  cost split : agent ${cost_agent:.4f} ({100*cost_agent/_sp:.1f}%) | "
              f"user-sim ${cost_user:.4f} ({100*cost_user/_sp:.1f}%) | "
              f"judge ${cost_judge:.4f} ({100*cost_judge/_sp:.1f}%)   "
              f"over {cost_split_seen}/{n_dispatched} episodes")
        print(f"               user-sim ${cost_user/cost_split_seen:.4f}/episode "
              f"(reference={solver.reference_user_model}, cheap={solver.cheap_user_model})")
    print("=" * 68)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({**meta, "pass": npass, "total": n,
                               "all_n": all_n, "pass_n": pass_n}, indent=1))
    print(f"written {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
