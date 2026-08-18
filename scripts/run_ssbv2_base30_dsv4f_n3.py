#!/usr/bin/env python3
"""SSB v2 BASE eval: 30 randomly-sampled tasks x N=3, DSV4F, 40-turn cap.

Design notes that matter:

  KEY SELECTION. OPENROUTER_API_KEY is read from the environment; if unset, it
  is read from the file named by EVOSKILL_OR_KEY_FILE. The key value is never
  printed; the [KEY] line shows the last 4 chars so first-tick monitoring can
  confirm which key is live.

  NATIVE-PINNED. Standing directive: EVERY DeepSeek model (flash included) needs
  BOTH provider_order=["deepseek"] AND allow_fallbacks=False. Without the pin,
  OpenRouter silently routes to a third-party host whose outputs differ.

  FRESH CACHE DIR. RunCache keys on question|sdk|model|effort|sample — the turn
  BUDGET is not in the key. `.cache/ssbv2_run_cache` already holds DSV4F base
  entries in the same tree_hash bucket, so pointing at it would risk replaying
  rollouts taken under a different cap and reporting them as budget-40 results.
  Caching stays ON (this run's rollouts are stored and reusable); it just does
  not read a bucket whose budget provenance is unverified.

  N=3 via sample_idx. evaluate_agent_parallel is one-shot per item, so the
  sample loop is explicit here, mirroring run_ssb1_test_nex_n3.py: cache.get_n
  returns a list POSITIONALLY aligned with sample_idx — never re-derive the
  index by enumerating hits.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Key selection. Never echoed — only the last 4 chars, in the [KEY] line below.
if os.environ.get("OPENROUTER_API_KEY"):
    _WHICH_KEY = "env"
else:
    _KEYFILE = os.environ.get("EVOSKILL_OR_KEY_FILE")
    if not _KEYFILE:
        raise SystemExit("set OPENROUTER_API_KEY (or point EVOSKILL_OR_KEY_FILE at a key file)")
    _KEY = Path(_KEYFILE)
    if not _KEY.exists():
        raise SystemExit(f"key file not found at {_KEY}")
    os.environ["OPENROUTER_API_KEY"] = _KEY.read_text().strip()
    _WHICH_KEY = "file"

os.environ["EVOSKILL_SANDBOX"] = "docker"
# SOLVER IMAGE — selectable, and RECORDED (header line + result JSON).
#
# This was a hardcoded "ssb-solver". Two reasons it is now both:
#
#   Selectable, because 2026-08-02 added `ssb-solver-lo` (identical baseline plus
#   LibreOffice Calc + python3-uno). Forking this script for the second arm would
#   fork every other design note in it too.
#
#   Recorded, because the RunCache tree_hash is skills-dir + solver system prompt
#   and DOES NOT include the image. Swapping the image changes what the solver can
#   do while leaving every cache key identical, so records from two different
#   environments can share a bucket and aggregate into one number with nothing
#   raising. The image identity is therefore written into the run header and the
#   result JSON, which is the only place a later reader can recover it from.
#   Verify a dir before reusing it: scripts/ssbv2_solver_env_report.py --cache ...
SOLVER_IMAGE = os.environ.get("EVOSKILL_DOCKER_IMAGE", "ssb-solver")
os.environ["EVOSKILL_DOCKER_IMAGE"] = SOLVER_IMAGE
os.environ.setdefault("EVOSKILL_BASH_TIMEOUT", "600")
os.environ["EVOSKILL_STREAM"] = "1"
os.environ.setdefault("EVOSKILL_STREAM_SILENCE_TIMEOUT", "45")
# --- #138: THE WALL IS NOW DERIVED FROM THE TURN CAP, NOT HARDCODED ----------
# The comment below used to say the wall "must scale with the turn budget" and
# then hardcoded 3600. Stating a rule beside a constant that violates it is how it
# stayed broken through a 40->45 cap raise.
#
# WHY THE WALL MUST NOT BIND. There are two independent stop conditions and only
# one of them is the experiment:
#   * TURN CAP — the measured budget. Sets `budget_exhausted`, scored on whatever
#     the solver produced. Legitimate and visible in every result.
#   * WALL CLOCK — a hang guard. Kills the rollout mid-work; the runner then
#     scores whatever happened to be on disk.
# Wall-clock duration depends on API contention, i.e. on CONCURRENCY. So a binding
# wall makes the measured score a function of how many rollouts run in parallel —
# it promotes a throughput knob into a scientific axis, which is the one thing that
# must never happen. Base was measured at concurrency 10 and later legs at 30.
#
# THE DERIVATION. Hangs are already caught by the liveness watchdog below
# (EVOSKILL_INACTIVITY_TIMEOUT / STREAM_SILENCE_TIMEOUT), which is progress-based
# and therefore cannot mistake a slow-but-healthy rollout for a dead one. The wall
# is a backstop for the case where liveness itself fails, so it should sit ABOVE
# the worst case a healthy rollout can reach: cap turns, each taking at most the
# inactivity timeout. 45 x 120s = 5400s, against observed maxima of 2010s at
# concurrency 10 and 2592s at 30.
# Liveness FIRST, because the wall is derived from it. Reading
# EVOSKILL_INACTIVITY_TIMEOUT with its own "120" fallback before the setdefault
# below would put the same constant in two places, so raising one would silently
# leave the other — the identical two-copies defect as #147's four walls.
os.environ.setdefault("EVOSKILL_LIVENESS", "1")
os.environ.setdefault("EVOSKILL_INACTIVITY_TIMEOUT", "120")
_CAP = int(os.environ.get("EVOSKILL_TURN_BUDGET", "40"))
_INACT = int(os.environ["EVOSKILL_INACTIVITY_TIMEOUT"])   # no second default
_WALL = max(3600, _CAP * _INACT)
os.environ.setdefault("EVOSKILL_SOLVER_TIMEOUT_SECONDS", str(_WALL))
# THE WALL IN FORCE, which is NOT necessarily `_WALL`. `_WALL` is the derived DEFAULT and
# the line above installs it with setdefault, so any launcher that exports
# EVOSKILL_SOLVER_TIMEOUT_SECONDS overrides it -- the sealed-test launcher exports 10800
# against a derived 5400. Every place that CHECKS, PRINTS or RECORDS the wall must read this
# value, not `_WALL`: on 2026-08-06 all three read `_WALL`, so the legs announced "wall=5400s"
# while running under 10800s, the result JSONs recorded solver_wall_s=5400 (a 2x false
# provenance claim about a one-shot sealed test), and the cached-replay guard flagged a
# 5572s SUCCESS as a wall kill, which alone blocked a leg from reporting.
_EFF_WALL = int(os.environ["EVOSKILL_SOLVER_TIMEOUT_SECONDS"])
os.environ.setdefault("EVOSKILL_CLEAN_BUDGET", "1800")
# Same derivation, same reason — the liveness total wall must not undercut the
# solver wall it is meant to backstop.
os.environ.setdefault("EVOSKILL_TOTAL_WALL", str(_WALL))
os.environ.setdefault("EVOSKILL_MAX_CLEAN_GAP", "120")

# Phoenix. Setting PHOENIX_PROJECT_NAME alone does NOTHING — nothing reads it
# unless init_tracing() actually registers the OTEL provider, and this script
# used to only set the env var. Result: the 2026-07-30 base30 run was launched
# and reported as "phoenix on" while emitting ZERO spans (no such project exists
# in Phoenix). Registration must also happen HERE, before the agent/SDK imports
# below, because instrumentation patches at import time. Fail-soft: init_tracing
# swallows its own errors, so a dead Phoenix server never blocks a run — which
# is exactly why the silent-no-op was survivable and went unnoticed.
_TAG_EARLY = os.environ.get("EVOSKILL_RUN_TAG", "base30")
os.environ.setdefault("PHOENIX_PROJECT_NAME", f"ssbv2-{_TAG_EARLY}-dsv4f")
from src.tracing import init_tracing                                   # noqa: E402
init_tracing(os.environ["PHOENIX_PROJECT_NAME"])

from src.agent_profiles.ssbv2_agent.dataset import load_ssbv2          # noqa: E402
from src.agent_profiles.ssbv2_agent.ssb_v2_agent import SSBv2SolverAgent  # noqa: E402
from src.cache.run_cache import RunCache, CacheConfig                  # noqa: E402
from src.harness.seats import SeatPool                                 # noqa: E402
from src.harness.rollout_ledger import failure_ledger, summarise        # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if "SSBV2_ROOT" not in os.environ:
    raise SystemExit("SSBV2_ROOT must point to your SpreadsheetBench v2 checkout")
SSBV2_ROOT = Path(os.environ["SSBV2_ROOT"])

# USER DIRECTIVE 2026-08-01: "DSV4F" means the DATED 0731 SNAPSHOT by default,
# not the rolling deepseek-v4-flash. Still env-driven so another snapshot is a
# launch-flag change rather than a fork of this script. Dated snapshots are
# separate model IDs on OpenRouter, and the model IS part of the RunCache file
# key — so two snapshots coexist in one cache dir without colliding, and equally
# never replay for each other.
MODEL = os.environ.get("EVOSKILL_SOLVER_MODEL", "deepseek/deepseek-v4-flash-0731")
PROVIDER_ORDER = ["deepseek"]      # native pin (standing directive)
ALLOW_FALLBACKS = False
# The agent has no allow_fallbacks kwarg; it reads this env var (parity fix
# added to ssb_v2_agent.py 2026-07-30 — SSB v2 had NO way to hard-pin before).
os.environ["EVOSKILL_SOLVER_ALLOW_FALLBACKS"] = "0"
# TURN BUDGET. Overridable because the 40-turn cap turned out to be BINDING and
# category-specific: in the epoch-2 phase 65/360 rollouts died at exactly 40 turns
# — Financial_Model 40/120, Debugging 25/120, Template 0/120 — while
# non-exhausted rollouts averaged 21.4. A cap that only bites two of three
# categories silently handicaps them, and it penalises verbose/verification-heavy
# skills, which is the opposite of what a skill-evolution experiment wants to
# measure. Raising it invalidates cache reuse (the budget is NOT in the cache key),
# so a raised-cap run needs its own EVOSKILL_CACHE_DIR — see the header.
BUDGET = int(os.environ.get("EVOSKILL_TURN_BUDGET", "40"))
N_SAMPLES = int(os.environ.get("EVOSKILL_N_SAMPLES", "3"))
N_TASKS = int(os.environ.get("EVOSKILL_N_TASKS", "30"))
SEED = int(os.environ.get("EVOSKILL_SAMPLE_SEED", "730"))
CONCURRENCY = int(os.environ.get("EVOSKILL_CONCURRENCY", "10"))
THINKING_EFFORT = ""

TAG = _TAG_EARLY   # already resolved above so Phoenix could register pre-import
# One cache dir can safely serve MANY phases: tree_hash keys on the skills dir
# + solver prompt, so each skill version occupies its own bucket and no phase
# can replay another's rollouts. Sharing it means a crashed or killed phase
# resumes for free instead of re-paying. (Safe only since the 0731 RunCache
# write-race fix, #116 — do not share a cache dir on older checkouts.)
CACHE_DIR = Path(os.environ.get("EVOSKILL_CACHE_DIR",
                                PROJECT_ROOT / ".cache" / f"ssbv2_{TAG}_cache"))
# Base legs point at an EMPTY dir (asserted empty below). A skill leg points at
# a populated one via EVOSKILL_SKILLS_DIR. The skills dir is part of the RunCache
# tree_hash, so a skill leg lands in a different bucket than base and the two
# can never replay each other's rollouts even in a shared cache dir.
_SKILLS_OVERRIDE = os.environ.get("EVOSKILL_SKILLS_DIR")
EMPTY_SKILLS = (Path(_SKILLS_OVERRIDE) if _SKILLS_OVERRIDE
                else PROJECT_ROOT / ".cache" / f"ssbv2_{TAG}_skills")
TASKS_CACHE = PROJECT_ROOT / ".cache" / f"ssbv2_{TAG}_tasks"
OUT = PROJECT_ROOT / "outputs" / "ssbv2_base30" / f"{TAG}.json"


def _probe_solver_image() -> dict:
    """Resolve the solver image's id and whether it carries a formula engine.

    Runs BEFORE any rollout, and hard-fails on a missing image. Discovering that
    at rollout time instead means N tasks of bash tool calls all failing on
    container creation, each burning turns and money on an environment error that
    reads, in the aggregate, like a hard benchmark.

    The soffice flag is the fact this whole run turns on: without it the solver
    cannot evaluate the formulas it is graded on (openpyxl returns None for any
    uncached formula), which is the condition every SSB v2 number before
    2026-08-02 was produced under.
    """
    import subprocess
    idq = subprocess.run(["docker", "images", "--format", "{{.ID}}", SOLVER_IMAGE],
                         capture_output=True, text=True)
    ids = [x for x in (idq.stdout or "").strip().splitlines() if x]
    if not ids:
        raise SystemExit(
            f"solver image {SOLVER_IMAGE!r} is not on this docker daemon. "
            f"Build it first (scripts/build_ssb_solver_lo.sh for the LibreOffice "
            f"arm, docker/ssb-solver/Dockerfile for the baseline).")
    probe = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "python3", SOLVER_IMAGE, "-c",
         "import shutil;print(shutil.which('soffice') or '')"],
        capture_output=True, text=True, timeout=300)
    if probe.returncode != 0:
        raise SystemExit(f"solver image {SOLVER_IMAGE!r} will not run python3: "
                         f"{probe.stderr[-300:]}")
    return {"image": SOLVER_IMAGE, "image_id": ids[0],
            "soffice": (probe.stdout or "").strip() or None}


async def main() -> int:
    from src.harness.task_workspace import acquire_task_lock
    for d in (EMPTY_SKILLS, TASKS_CACHE, OUT.parent):
        d.mkdir(parents=True, exist_ok=True)
    leftover = list(EMPTY_SKILLS.iterdir())
    if _SKILLS_OVERRIDE:
        # Skill leg: the dir MUST be populated, and each skill must actually
        # parse, or the leg silently degenerates into a second base run whose
        # null result would be read as "the skill did not help".
        skills = sorted(p.name for p in EMPTY_SKILLS.glob("*/SKILL.md"))
        if not skills:
            raise SystemExit(f"EVOSKILL_SKILLS_DIR has no */SKILL.md: {EMPTY_SKILLS}")
        from src.harness.direct.executor import _discover_skills
        found = _discover_skills(EMPTY_SKILLS)
        if len(found) != len(skills):
            raise SystemExit(f"skills failed to parse: files={skills} parsed={list(found)}")
        for n, meta in found.items():
            print(f"  [SKILL] {n}: {meta.get('description', '')[:100]}", flush=True)
    elif leftover:
        raise SystemExit(f"base skills dir not empty: {leftover}")
    acquire_task_lock(TASKS_CACHE)

    train, val, test = load_ssbv2(SSBV2_ROOT, n_train=80, n_val=60)
    pool = list(train) + list(val) + list(test)

    # Two modes. SPLIT_FILE reads a FROZEN split (the 0801 evolution: the split
    # is committed once and every phase reads it, so a run cannot silently
    # redraw its own train set between epochs). Otherwise fall back to the
    # ad-hoc seeded sample the base30 runs used.
    split_file = os.environ.get("EVOSKILL_SPLIT_FILE")
    if split_file:
        split = json.loads(Path(split_file).read_text())
        # Comma-separated keys POOL several splits into ONE run ("train,val").
        # Two sequential 60-task phases each pay their own ramp-down tail: on the
        # 0801 bootstrap the last single rollout of the train phase ran ~25 min
        # with 29 of 30 concurrency slots idle. Pooling lets a straggler overlap
        # with real work, so an epoch pays one tail instead of two.
        #
        # This does NOT weaken the val firewall. The firewall governs what the
        # EVOLVER READS, not what executes: rollouts are keyed per task and the
        # digest builder filters to `train` and asserts no other split survives.
        # Pooling also cannot redraw the split — keys only ever index the frozen
        # file, and duplicates would be a config error, so reject them.
        keys = [k.strip() for k in os.environ["EVOSKILL_SPLIT_KEY"].split(",") if k.strip()]
        unknown = [k for k in keys if k not in split]
        if unknown:
            raise SystemExit(f"unknown split key(s) {unknown}; have {sorted(split)}")
        want: list[str] = []
        for k in keys:
            want.extend(split[k])
        if len(set(want)) != len(want):
            raise SystemExit(f"split keys {keys} overlap — refusing to run a task twice")
        by_id = {json.loads(q)["task_id"]: (q, gt, c) for q, gt, c in pool}
        missing = [t for t in want if t not in by_id]
        if missing:
            raise SystemExit(f"split ids absent from the loaded pool: {missing[:5]}")
        items = [by_id[t] for t in want]                # frozen file order
        # OPTIONAL SUBSET, for targeted diagnostics on an identified set of tasks
        # (e.g. only those that hit the turn cap). Order is preserved from the
        # split file so sample_idx alignment with a previous phase is unchanged;
        # the subset is asserted to be a real subset so a typo cannot silently
        # produce a smaller-but-different task set.
        _subset_file = os.environ.get("EVOSKILL_TASK_SUBSET")
        if _subset_file:
            _keep = set(json.loads(Path(_subset_file).read_text()))
            _have = {json.loads(q)["task_id"] for q, _, _ in items}
            _unknown = sorted(_keep - _have)
            if _unknown:
                raise SystemExit(f"EVOSKILL_TASK_SUBSET names tasks absent from the "
                                 f"selected split: {_unknown[:5]}")
            items = [it for it in items if json.loads(it[0])["task_id"] in _keep]
            if len(items) != len(_keep):
                raise SystemExit(f"subset filter kept {len(items)} of {len(_keep)} requested")
            print(f"SUBSET: {len(items)} of {len(_have)} tasks, from "
                  f"{Path(_subset_file).name}", flush=True)
        _per_key = " + ".join(f"{k}={len(split[k])}" for k in keys)
        print(f"SPLIT {'+'.join(keys)} from {Path(split_file).name}: "
              f"{len(items)} tasks ({_per_key})", flush=True)
    else:
        rng = random.Random(SEED)
        items = rng.sample(pool, min(N_TASKS, len(pool)))
    # Report which split each selected task came from — against the split SOURCE
    # ACTUALLY IN USE.
    #
    # This used to classify every task against `load_ssbv2`'s own default
    # partition (n_train=80, n_val=60) even when a FROZEN split file was driving
    # the run. Those two partitions are different, so a task legitimately in
    # split.json's train/val but outside the dataset's first 140 was printed as
    # "test". Every phase of the 0801 evolution therefore logged
    # `origin={'train': 35, 'val': 21, 'test': 64}` while running strictly on
    # train+val — a false contamination alarm sitting in the permanent record of
    # a run whose test split is under seal. Verified 2026-08-02: split.json's
    # test list and those phases' task sets are disjoint.
    origin = {"train": 0, "val": 0, "test": 0}
    if split_file:
        _by_split = {}
        for _k in ("train", "val", "test"):
            for _tid in split.get(_k, []):
                _by_split[_tid] = _k
        for q, _, _ in items:
            _tid = json.loads(q)["task_id"]
            origin[_by_split.get(_tid, "test")] += 1
        # A task the frozen split does not place at all must not be silently
        # bucketed as "test" — that is how the mislabel read as contamination.
        _unplaced = [json.loads(q)["task_id"] for q, _, _ in items
                     if json.loads(q)["task_id"] not in _by_split]
        if _unplaced:
            raise SystemExit(f"{len(_unplaced)} selected task(s) are absent from "
                             f"{Path(split_file).name}: {_unplaced[:5]}")
    else:
        tr_q = {q for q, _, _ in train}
        va_q = {q for q, _, _ in val}
        for q, _, _ in items:
            origin["train" if q in tr_q else ("val" if q in va_q else "test")] += 1
    if origin["test"] and not split_file:
        pass    # ad-hoc seeded sampling legitimately spans all three

    print(f"SSB v2 BASE | {len(items)} tasks x N={N_SAMPLES} = {len(items)*N_SAMPLES} rollouts",
          flush=True)
    print(f"  model={MODEL} pin={PROVIDER_ORDER} fallbacks={ALLOW_FALLBACKS} budget={BUDGET}",
          flush=True)
    print(f"  seed={SEED} origin={origin} concurrency={CONCURRENCY}", flush=True)
    print(f"  cache={CACHE_DIR.name} (fresh; budget is NOT in the cache key)", flush=True)
    # RECORD THE WALL, for the same reason solver_image is recorded: it is an
    # axis absent from the cache key that can change what gets measured. A
    # binding wall kills rollouts and they score 0.0, so a reader comparing two
    # legs needs to know each leg's wall. Derived from the turn cap (#138).
    print(f"  wall={_EFF_WALL}s IN FORCE"
          + (f" (derived default was {_WALL}s = cap {_CAP} x inactivity {_INACT}s, "
             f"OVERRIDDEN by EVOSKILL_SOLVER_TIMEOUT_SECONDS)" if _EFF_WALL != _WALL
             else f" (derived: cap {_CAP} x inactivity {_INACT}s)")
          + " — a hang guard, and it must NEVER bind before the turn cap", flush=True)
    # Image + tree_hash on adjacent lines, deliberately: the second does not
    # cover the first, and putting them side by side is what makes that visible
    # in a log someone reads six weeks from now.
    _img = _probe_solver_image()
    print(f"  image={_img['image']} ({_img['image_id']}) "
          f"formula_engine={_img['soffice'] or 'NONE — solver cannot evaluate what it is graded on'}",
          flush=True)
    # Report the ACTUAL provider class, not just the project name. init_tracing
    # is fail-soft, so "phoenix=<name>" on its own is compatible with emitting
    # nothing at all — the failure mode that made the 0730 run traceless. A real
    # SDK TracerProvider means spans will export; Proxy/NoOp means they will not.
    from opentelemetry import trace as _ot
    _tp = type(_ot.get_tracer_provider()).__name__
    _live = _tp not in ("ProxyTracerProvider", "NoOpTracerProvider", "DefaultTracerProvider")
    print(f"  phoenix={os.environ['PHOENIX_PROJECT_NAME']} "
          f"[{'LIVE' if _live else 'NOT REGISTERED'} — provider={_tp}]", flush=True)
    # Label must be derived, not hardcoded: the whole point of this line is that
    # tick-1 babysitting can tell which account is being billed, and a label that
    # says "personal" while the org key is live is worse than no label at all.
    print(f"  [KEY] OpenRouter key ...{os.environ['OPENROUTER_API_KEY'][-4:]} ({_WHICH_KEY})",
          flush=True)

    solver = SSBv2SolverAgent(
        model=MODEL, budget=BUDGET, shared_skills_dir=EMPTY_SKILLS,
        provider_order=PROVIDER_ORDER,
        thinking_effort=THINKING_EFFORT, name="solver",
        timeout_seconds=int(os.environ["EVOSKILL_SOLVER_TIMEOUT_SECONDS"]),
        tasks_cache=TASKS_CACHE,
    )
    solver.sdk_override = "direct"

    cache = RunCache(CacheConfig(
        cache_dir=CACHE_DIR, enabled=True, store_messages=True,
        cwd=PROJECT_ROOT, live_skills_dir=EMPTY_SKILLS,
        project_source_root=PROJECT_ROOT,
    ))

    # THREE AXES THAT CHANGE A ROLLOUT AND ARE NOT IN THE CACHE KEY.
    #
    # Found 2026-08-02, the hard way. `cache.set`/`get_n` below pass sdk+model
    # only, never `system_prompt` — and NONE of the 15 SSB v2 runners ever did —
    # so `_get_tree_hash` takes its FALLBACK branch and hashes
    # `solver/prompt.txt` + `officeqa_agent/prompt.md`, two files unrelated to
    # this benchmark. Consequence: every SSB v2 cache dir in the repo lives in
    # the same bucket, `bfb5bd90d1e0`, whatever the SSB v2 solver prompt says.
    # The docker image and the turn budget were never in the key either.
    #
    # So a run that swapped the solver image (no formula engine -> LibreOffice)
    # and pointed at an existing cache dir would replay the old environment's
    # rollouts and report them as the new one's. Nothing would error. On
    # 2026-08-02 only an unrelated choice of a fresh dir prevented exactly that.
    #
    # `system_prompt` is now passed to every cache call (below), which makes the
    # documented mechanism real. The marker covers what a cache key structurally
    # cannot: image identity and turn budget.
    from src.cache.cache_env import assert_cache_env, sha12
    from src.agent_profiles.ssbv2_agent import get_solver_prompt
    CACHE_SYS = get_solver_prompt()
    assert_cache_env(CACHE_DIR, {
        "solver_image": _img["image"],
        "solver_image_id": _img["image_id"],
        "solver_formula_engine": _img["soffice"],
        "turn_budget": BUDGET,
        "model": MODEL,
        "solver_prompt_sha256_12": sha12(CACHE_SYS),
        # THE WALL IS DELIBERATELY *NOT* AN AXIS HERE. assert_cache_env exists for
        # differences that are INVISIBLE in the records — image, engine, prompt: two
        # runs with different images produce indistinguishable cache keys, so only a
        # marker can catch it. A wall difference is VISIBLE: a wall kill carries
        # is_error plus a parse_error naming the timeout, and the ledger now refuses
        # the leg on it. Listing it here would lock every existing cache dir out of
        # every future run the moment the turn cap changes — including the base
        # records this arm's own comparison depends on — while adding nothing the
        # ledger does not already catch. It IS recorded in the result JSON.
        #
        # The precise residual concern is narrower, and is checked on the cache-hit
        # path instead: a cached record whose duration EXCEEDS this run's wall is one
        # this config would have killed, so replaying it as valid overstates the
        # solver. That check can actually fire; a blanket axis would fire on every
        # cap change and get switched off.
        # THE SCORER IS AN AXIS TOO, and it is the one that bit hardest.
        # `inline_score` is computed at rollout time and CACHED with the trace, so
        # a cached record carries a verdict produced by whatever scorer was live
        # then. On 2026-08-03 `_force_iterative_calc` was changed from an
        # openpyxl round-trip (which blanked the cached values of 143 of 169
        # rollouts, forcing LibreOffice to re-solve whole models) to a zip-level
        # patch that preserves them. Same rollouts, different verdicts, and
        # nothing in the key could tell the two apart.
        "scorer_sha256_12": sha12(
            (PROJECT_ROOT / "src/agent_profiles/ssbv2_agent/scorer.py").read_text()),
    }, label=TAG)

    scores: dict[int, list] = {i: [None] * N_SAMPLES for i in range(len(items))}
    # Track budget exhaustion: at a 40-turn cap on SSB v2 this is BINDING
    # (both pilot rollouts hit it), so the base rate partly measures the cap.
    # Reporting the count keeps that visible instead of buried.
    exhausted: dict[int, list] = {i: [False] * N_SAMPLES for i in range(len(items))}

    # ROLLOUT ADMISSION. Two gates, composed, and both must be held:
    #
    #   local  — this process's own ceiling, EVOSKILL_CONCURRENCY. Unchanged.
    #   seats  — a FLEET-WIDE pool shared with every other participating process,
    #            with strict priority (src/harness/seats.py). Opt-in.
    #
    # Why both rather than replacing one with the other: the local ceiling is what
    # every historical run was measured under, and dropping it would silently let a
    # single process take the whole fleet pool. Composing them means an absent
    # EVOSKILL_SEAT_TOTAL reproduces the previous behaviour EXACTLY, and setting the
    # seat total equal to the local concurrency makes the fleet gate the binding one.
    #
    # The ordering matters: take the scarce cross-process resource FIRST, then the
    # local slot. Reversed, a process could sit on a local slot while blocked on a
    # seat, and its own later rollouts would queue behind a coroutine that is itself
    # waiting — self-inflicted head-of-line blocking.
    _seats = SeatPool.from_env(
        member=os.environ.get("EVOSKILL_SEAT_MEMBER", TAG))
    _local = asyncio.Semaphore(CONCURRENCY)

    class _Gate:
        """asyncio.Semaphore-compatible composition of the two gates above."""

        async def __aenter__(self):
            if _seats is not None:
                await _seats.acquire()
            await _local.acquire()
            return self

        async def __aexit__(self, *exc):
            _local.release()
            if _seats is not None:
                _seats.release()

    sem = _Gate()
    if _seats is not None:
        s = _seats.snapshot()
        print(f"  [SEATS] fleet pool total={s['total']} used={s['used']} "
              f"member={_seats.member} priority={_seats.priority} "
              f"(local cap {CONCURRENCY})", flush=True)
    else:
        print(f"  [SEATS] disabled — local cap {CONCURRENCY} only "
              f"(set EVOSKILL_SEAT_TOTAL to share a fleet pool)", flush=True)
    # system_prompt is what makes the SOLVER PROMPT part of tree_hash. Without
    # it RunCache silently falls back to hashing two unrelated agents' prompt
    # files — see the assert_cache_env comment above. Passing it means a prompt
    # edit now rotates the bucket, as the docs have always claimed it did.
    ck = {"sdk": "direct", "model": MODEL, "system_prompt": CACHE_SYS}
    # #138: rollouts the WALL killed rather than the turn cap. Collected here
    # because the trace is only in scope at the scoring site — the ledger cannot
    # see it (the coroutine returns None). A leg with any entry is not a
    # measurement: the wall is a hang guard and must not decide an outcome.
    wall_killed: list[tuple[int, int, str]] = []

    async def one(i, q, gt, s):
        async with sem:
            hit = (cache.get_n(q, n=N_SAMPLES, **ck) or [None] * N_SAMPLES)[s]
            if hit is not None:
                scores[i][s] = hit.inline_score if hit.inline_score is not None else 0.0
                # Cached traces carry num_turns too. Reading it here matters on a
                # RESUMED run: without it every replayed rollout reports "not
                # exhausted", so a resume would under-count the turn cap and make
                # the base rate look less cap-bound than it is.
                exhausted[i][s] = bool(getattr(hit, "budget_exhausted", False))
                # A wall kill CACHED by an earlier leg replays as a normal 0.0.
                # That is how all 5 census records survived into reported numbers,
                # so the cache path needs the same check as the fresh path.
                _cw = str(getattr(hit, "parse_error", "") or "")
                if getattr(hit, "is_error", False) and "timed out" in _cw.lower():
                    wall_killed.append((i, s, f"CACHED wall kill: {_cw}"))
                # A cached record LONGER than this run's wall is one this config
                # would have killed, and replaying it as valid overstates the
                # solver. This is the only way a wall change can corrupt a replay,
                # and it is narrow enough to actually fire.
                # COMPARE AGAINST THE WALL ACTUALLY IN FORCE, not the derived default.
                # `_WALL` is only the DEFAULT (max(3600, cap * inactivity) = 5400 at cap 45);
                # line ~109 installs it with setdefault, so a launcher that exports
                # EVOSKILL_SOLVER_TIMEOUT_SECONDS -- as the sealed-test launcher does, with
                # 10800 -- wins and the solver runs under 10800s. Checking `_hd > _WALL`
                # therefore flagged a rollout that ran 5572s and SUCCEEDED, comfortably
                # inside its real 10800s wall, as a WallClockKill; that false failure alone
                # blocked the fm base leg of a one-shot sealed test from reporting. It is
                # the same two-copies defect the comment above the derivation warns about,
                # committed one screen below the warning.
                _hd = getattr(hit, "duration_ms", 0) or 0
                if _hd > _EFF_WALL * 1000:
                    wall_killed.append((
                        i, s, f"CACHED record ran {_hd // 1000}s, longer than this "
                              f"run's {_EFF_WALL}s wall — this config would have killed it"))
                return
            # SSBv2SolverAgent.run resolves ground truth internally from the
            # question JSON (eval_run_ground_truth contextvar).
            # sample_idx is REQUIRED here: this gather runs all N samples of a
            # task concurrently, and without it they share one workspace and one
            # output file (see task_ws_name). The 2026-07-30 and first 0731 runs
            # predate this and are corrupt.
            tr = await solver.run(q, tag=f"{TAG} {i+1}/{len(items)} [s{s+1}]", sample_idx=s)
            # Exact, not heuristic: Agent.run converts its wall-clock TimeoutError
            # into a partial trace carrying is_error plus a parse_error that names
            # the timeout. Matching on duration_ms would be a proxy, and a proxy
            # breaks the moment the wall changes.
            _why = str(getattr(tr, "parse_error", "") or "")
            if getattr(tr, "is_error", False) and "timed out" in _why.lower():
                wall_killed.append((i, s, _why))
            scores[i][s] = tr.inline_score if tr.inline_score is not None else 0.0
            # Use the harness's own flag, NOT a num_turns >= BUDGET proxy. A
            # rollout that ANSWERS on turn 40 has num_turns == 40 but was never
            # truncated, so the proxy over-counts exhaustion (22 vs 20 on the
            # 0731b base leg, 8 vs 7 on the skill leg). Same >=-vs-> boundary
            # error as #110: at equality the two readings disagree, so the
            # comparison operator IS the measurement.
            exhausted[i][s] = bool(getattr(tr, "budget_exhausted", False))
            cache.set(q, tr, sample_idx=s, **ck)

    # ONE ROLLOUT'S EXCEPTION MUST NOT DESTROY THE OTHER 359. (#140)
    #
    # Measured 2026-08-03: `test_skills_v4` died at 64/360 when a provider 503
    # survived the retry budget and propagated out of `one()` -> gather -> main.
    # A bare gather cancels every sibling on the first exception, so ~30 in-flight
    # rollouts were lost and NO result JSON was written. The 64 completed ones were
    # recoverable only because each cached record carries `trace.inline_score`.
    #
    # `return_exceptions=True` is necessary but NOT sufficient, and the naive
    # version is a trap in both directions:
    #   * default the failed score to 0.0 and a provider outage is silently
    #     recorded as a WRONG ANSWER — corrupting the very number being measured,
    #     unrecoverably and invisibly. This is strictly worse than the crash.
    #   * leave it None (as it is initialised) and the aggregation below raises
    #     `TypeError: '>=' not supported between NoneType and float`, i.e. the same
    #     lost leg with a stack trace pointing far away from the cause.
    #
    # So: keep going, keep the failures DISTINGUISHABLE from zeros, and refuse to
    # emit a result JSON that would read as a complete leg. A failed rollout is
    # never cached (`cache.set` is unreachable on raise), so a relaunch retries
    # exactly the failures and replays everything else from cache.
    idx = [(i, s) for i in range(len(items)) for s in range(N_SAMPLES)]
    try:
        results = await asyncio.gather(
            *[one(i, q, gt, s)
              for i, (q, gt, _c) in enumerate(items)
              for s in range(N_SAMPLES)],
            return_exceptions=True)
    finally:
        # Surrender the fleet membership so a higher-priority phase does not have to
        # wait out the TTL to reclaim these seats. The `finally` covers the crash
        # path too; if even that is skipped (SIGKILL), the pool's PID-liveness reap
        # is the backstop — belt and braces, because a leaked membership silently
        # shrinks the pool for every later run.
        if _seats is not None:
            try:
                _seats.leave()
            except OSError as e:
                print(f"  [SEATS] leave() failed: {e} (PID reap will reclaim)",
                      flush=True)

    # ---- FAILURE LEDGER (#140) ------------------------------------------------
    # See src/harness/rollout_ledger.py for why a tolerant gather is only safe
    # when paired with this: failures must stay distinguishable from real zeros.
    tid = [json.loads(gt).get("task_id", "?") if gt.startswith("{") else "?"
           for _q, gt, _c in items]
    failures = failure_ledger(idx, results, scores, killed=wall_killed)

    if failures:
        print("\n" + "!" * 68, flush=True)
        print(f"{len(failures)}/{len(idx)} ROLLOUTS FAILED — this leg is INCOMPLETE.",
              flush=True)
        by_kind = summarise(failures)
        for kind, n in sorted(by_kind.items(), key=lambda x: -x[1]):
            print(f"  {n:4d} x {kind}", flush=True)
        for i, s, kind, msg in failures[:20]:
            print(f"    task[{i}] {tid[i]} sample {s+1}: {kind}: {msg}", flush=True)
        if len(failures) > 20:
            print(f"    ... and {len(failures) - 20} more", flush=True)
        # A PARTIAL file, at a DELIBERATELY DIFFERENT PATH. `val_selection.py`,
        # `SEALED_TEST_RUNBOOK.sh` and the heartbeat all treat the canonical
        # `OUT` path as proof of a complete leg, and a partial leg reads HIGH
        # (rollouts finish slowest-last and slow correlates with hard), so
        # writing OUT here would hand a biased number to selection. The absent
        # OUT also keeps the monitor's DEAD-NO-RESULT-JSON alert firing, which
        # is the correct signal: this leg needs a relaunch.
        part = OUT.with_suffix(".PARTIAL.json")
        part.write_text(json.dumps({
            "INCOMPLETE": True, "tag": TAG,
            "n_failed": len(failures), "n_rollouts": len(idx),
            "failed_by_kind": by_kind,
            "failed": [{"task_index": i, "task_id": tid[i], "sample": s,
                        "kind": kind, "message": msg}
                       for i, s, kind, msg in failures],
            # Scores that DID land, so the partial state is inspectable without
            # walking the cache. Not a result — every consumer must gate on
            # INCOMPLETE.
            "per_task_partial": {str(i): v for i, v in scores.items()},
        }, indent=1))
        print(f"\nwrote {part}", flush=True)
        print("NOT writing the canonical result JSON: a leg with failed rollouts is "
              "not a measurement.", flush=True)
        print("RELAUNCH with identical env — the cache replays every completed "
              "rollout and retries only these.", flush=True)
        print("!" * 68, flush=True)
        return 1

    flat = [x for v in scores.values() for x in v]
    npass = sum(1 for x in flat if x >= 0.999)
    all3 = sum(1 for v in scores.values() if all(x >= 0.999 for x in v))
    pass3 = sum(1 for v in scores.values() if any(x >= 0.999 for x in v))
    print("\n" + "=" * 68)
    n_ex = sum(1 for v in exhausted.values() for x in v if x)
    print(f"BASE {npass}/{len(flat)} = {npass/len(flat):.1%} | ALL@{N_SAMPLES}={all3}/{len(items)}"
          f" | Pass@{N_SAMPLES}={pass3}/{len(items)}")
    print(f"TURN CAP: {n_ex}/{len(flat)} rollouts exhausted the {BUDGET}-turn budget"
          f" ({n_ex/len(flat):.0%}) — the base rate partly measures this cap")
    print("=" * 68, flush=True)
    OUT.write_text(json.dumps({
        "model": MODEL, "provider_order": PROVIDER_ORDER, "budget": BUDGET,
        "n_samples": N_SAMPLES, "seed": SEED, "origin": origin,
        "sample_pass": npass, "sample_total": len(flat), "all3": all3,
        "pass3": pass3, "n": len(items),
        "turn_exhausted": n_ex,
        # Solver ENVIRONMENT, recorded because tree_hash cannot: two runs with
        # different images and the same prompt are indistinguishable from their
        # cache keys alone, so this is the only durable record of which
        # environment produced these scores.
        "solver_image": _img["image"], "solver_image_id": _img["image_id"],
        "solver_formula_engine": _img["soffice"],
        "solver_prompt_sha256_12": sha12(CACHE_SYS),
        # #304: the orchestrator's _v_leg_loaded_promoted proves delivery by reading
        # this field; without it a leg can run against the wrong library and still
        # report a clean pass rate. None (not ".") when no override was given.
        "skills_dir": _SKILLS_OVERRIDE or None,
        # The wall IN FORCE, plus the default it may have overridden. Recording only the
        # derived value made every sealed-test JSON understate its own wall by 2x.
        "solver_wall_s": _EFF_WALL,
        "solver_wall_derived_default_s": _WALL,
        "solver_wall_derivation": (
            f"max(3600, cap {_CAP} * inactivity {_INACT}) = {_WALL}"
            + (f"; OVERRIDDEN to {_EFF_WALL} via EVOSKILL_SOLVER_TIMEOUT_SECONDS"
               if _EFF_WALL != _WALL else "")),
        "concurrency": CONCURRENCY, "tag": TAG,
        # The dirs this leg ACTUALLY used, so downstream steps read them instead of
        # re-deriving names (#194; the tau3 runner records cache_dirs for the same reason).
        "cache_dir": CACHE_DIR.name, "tasks_cache": str(TASKS_CACHE),
        "per_task": {str(i): v for i, v in scores.items()},
        # Same list the failure ledger labels with, computed once, so a PARTIAL
        # file and its eventual canonical result can never disagree about which
        # task index is which task id.
        "task_ids": tid,
    }, indent=1))
    print(f"written {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
