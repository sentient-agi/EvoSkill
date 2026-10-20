#!/usr/bin/env python3
"""The SSB v2 leg library on Daytona x86 sandboxes. The Claude Code legs (cc_pool_host.py, cc_leg_driver.py) reuse
its machine class (Shard: creation, identity / egress probes, cgroup delegation, the shim and grading self-tests), the
grading bundle, the image pins (EXPECTED), the run's declared split, the leg knobs and the data overlay. Its own `leg`
command runs a leg whose solver is the harness-free runner (scripts/run_ssbv2_base30_dsv4f_n3.py) through OpenRouter.

Built on ssb_leg_daytona.py, so a leg here and an ssb_leg_daytona.py leg are the same comparability class: the image recipe (Dockerfile.ssb_leg, sha256 pinned below), the sandbox scripts
(sandbox_entry.py, sandbox_launch.sh, sandbox_sampler.py, docker_shim.py -- uploaded unchanged), the
leg's scientific settings and throughput knobs (LEG_SETTINGS / KNOBS / THREAD_ENV) and the probe /
self-test commands (IDENTITY_CMD, EGRESS_CMD, SELFTEST_CMD, GRADE_SELFTEST), all imported read-only
from ssb_leg_daytona. What is NEW here:

  * the LOOP CONTRACT: one process per leg whose command line carries the runner's name
    (`--runner run_ssbv2_base30_dsv4f_n3.py`, so loop._pid_is_runner / _live_runners see exactly one
    pid per leg), the result lands at <run_dir>/<tag>.json ONLY when every shard succeeded with zero
    provider violations (else <run_dir>/<tag>.FAILED.json and exit 5), written atomically AFTER the
    merges, and the process exits last;
  * the PROMOTED SKILLS DIR is uploaded into every sandbox at its own absolute path and handed to the
    runner as EVOSKILL_SKILLS_DIR (a base leg gets none -- the runner's own per-tag empty dir), so the result's `skills_dir` equals the host path the loop checks;
  * the PER-EPOCH CACHE DIR (EVOSKILL_CACHE_DIR, one per epoch, shared by that epoch's train and val
    legs) is the source of truth on the host: each shard is SEEDED with the records of its own tasks
    (plus the CACHE_ENV marker), and every shard's cache is MERGED BACK (new files copied, identical
    skipped, a differing record for the same file = conflict = leg refused), so reruns replay;
  * the TASKS DIR (snapshots + DIFF.txt) of every shard is merged into <repo>/.cache/ssbv2_<tag>_tasks;
  * sandboxes are named evoskill-loop-<leg>-<shard>-<HHMMSS>, ledgered (create/readback/delete)
    to daytona_lib_loop.LEDGER, and deleted BY ID in a finally in every shard thread (SIGTERM/SIGINT
    abort -> kill runners, sync, delete);
  * a SPEND GUARD per leg: OpenRouter key-usage delta vs this leg's start (--leg cap), vs the RUN's
    origin (first leg of the run records it; --run cap), and an account-balance floor. Any breach
    aborts every shard;
  * the result JSON carries a `daytona` block: image id, Dockerfile sha, Daytona snapshot ref,
    LibreOffice build, grading openpyxl, per-shard sandbox ids/walls, spend, merges, skills sha;
  * THE GRADER TIMEOUTS FROM <run_dir>/INFRA.json: when the run has an INFRA.json (src/orchestrator/infra.py
    validates it), its grader.recalc_timeout_s and grader.datatable_uno_timeout_s go into every sandbox's
    leg.env as EVOSKILL_RECALC_TIMEOUT_SECONDS / EVOSKILL_DATATABLE_UNO_TIMEOUT_SECONDS and into
    runtime_env.knobs. These two are the ONLY knobs allowed to differ from the held-out baseline's
    KNOBS (see knob_drift). A run without INFRA.json runs the baseline's KNOBS unchanged -- EXCEPT a
    run whose RUNTIME.json declares prompt_mode "template": there INFRA.json is the one operator-owned
    infra config, so a leg without it is REFUSED rather than silently graded at the baseline's 180/300 s
    (runtime_prompt_mode + LegConfig);
  * THE DATA OVERLAY FROM <run_dir>/INFRA.json: its block "data" declares the repaired-file overlay
    (src/orchestrator/data_overlay.py). With an overlay declared, this process's
    EVOSKILL_SSBV2_DATA_OVERLAY is set from it when unset and the leg is REFUSED when it is set to
    another file, or when PATCHES.json no longer hashes to the declared sha or an entry fails to
    verify; with none declared (null) the variable set at all is refused. The path is passed to
    data_overlay explicitly and recorded (dry run, check-only line, result `daytona.data_overlay`).
    A run without INFRA.json keeps the bare variable (data_overlay_from_infra).
  * POOL MODE: RUNTIME daytona_leg.pool true (default false = everything above, unchanged) makes this process
    the leg's FOLLOWER in ONE machine pool per RUN that serves every epoch's legs (scripts/daytona/ssb_leg_pool.py:
    global priority queue of one-task units -- newest epoch's train first, then val oldest epoch first, then the
    held-out test -- fixed or ramped per-machine cap). The loop contract, the result path, merge_results/_finish
    and the FAILED path are the same; each leg's result is written as soon as ITS units have ended (the machine
    may still serve other legs: LegRun.unit_end_check / listing_ids replace the "sandbox deleted" checks for a
    pool leg); POOL_DEFAULTS lists the keys.

Subcommands (run with the repo's venv and its `daytona` extra: the Daytona SDK + python-dotenv + httpx;
`--dry-run`/`--check-only`/`plan` never import the SDK or read keys):

  leg     --runner run_ssbv2_base30_dsv4f_n3.py --tag T --key train|val --run-dir D --split F
          --cache-dir C [--skills-dir S] [--n-samples 3] [--concurrency 30] [--dry-run|--check-only]
  launch-base --run-dir D [--dry-run]     # the two e0 base legs, via the leg script, from SPEC+RUNTIME
  status  --run-dir D                     # per-tag: driver alive? result / FAILED? shard progress
  cleanup --run-dir D [--force]           # delete the run's loop sandboxes the ledger shows created, not deleted

Keys (DAYTONA_*, OPENROUTER_API_KEY) are read from the environment, else from the repo's .env, inside this
process only; the OpenRouter key reaches a sandbox only as the env of ONE SDK exec call (sandbox_launch.sh).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if str(HERE.parents[1]) not in sys.path:                # the repo, for src.orchestrator.infra
    sys.path.insert(1, str(HERE.parents[1]))
import daytona_lib_loop as dlx  # noqa: E402
import ssb_leg_daytona as base  # noqa: E402  (READ-ONLY reuse: constants, commands, helpers)
from src.orchestrator import infra as infra_cfg  # noqa: E402  (stdlib only)
from src.orchestrator import data_overlay  # noqa: E402  (stdlib only; repaired files, byte-wise)

REPO = base.REPO
DATA = base.DATA
DOCKERFILE = base.DOCKERFILE
RUNNER_REL = base.RUNNER_REL
RUNNER_NAME = Path(RUNNER_REL).name                     # run_ssbv2_base30_dsv4f_n3.py
CATEGORIES = base.CATEGORIES
RESOURCES = dict(base.RESOURCES)
ALLOW_DOMAINS = base.ALLOW_DOMAINS
POLL_S = int(os.environ.get("EVOSKILL_DAYTONA_POLL_S", "30"))
SANDBOX_SCRIPTS = ("sandbox_entry.py", "sandbox_launch.sh", "sandbox_sampler.py", "docker_shim.py")
DAYTONA_USD_PER_SANDBOX_HOUR = 0.926                    # list price, 8 vCPU / 32 GiB / 40 GB

# The baseline's image. A leg on any other image is a different comparability class from the held-out
# baseline, so each is REQUIRED by default (RUNTIME daytona_leg may relax require_snapshot_ref / require_libreoffice
# explicitly). The image's content identity is dockerfile_sha256 (the file's bytes, re-hashed before every
# create). snapshot_ref is the ref Daytona derives when it builds that file; it cannot be computed offline, so
# it is None here (= not pinned): every create accepts the ref Daytona reports and records it (the leg summary's
# readback.snapshot_ref, runtime_env image.daytona_snapshot_ref). Set it (here, or RUNTIME daytona_leg.
# require_snapshot_ref + INFRA leg_sandbox.snapshot_ref) to require that exact build.
EXPECTED = {
    "dockerfile_sha256": "681a617ef3ce70ab8056dcd78dffcf9531ddb3c55fd4d2925ba6d0dc9d06e65d",
    "snapshot_ref": None,
    "image_id": "x86daytona-681a617ef3ce",
    "libreoffice": "LibreOffice 25.2.3.2 520(Build:2)",
    "grading_openpyxl_prefix": "3.1.5",
    # The held-out baseline's code (its runtime_env). Each shard bundles the LIVE repo at its start, so an edit to
    # any of these files would silently enter later legs; a leg refuses before creating a sandbox, and the merge
    # refuses a shard whose recorded hash differs.
    "baseline_runtime": {
        # the runner: a 1800 s per-attempt / 3600 s per-call model-call deadline (EVOSKILL_LLM_CALL_DEADLINE / _MAX),
        # a cut attempt discarded and resampled as infra, a call TimeoutError re-raised as LLMCallInfraError, a wall
        # kill never cached
        "runner_sha256": ("scripts/run_ssbv2_base30_dsv4f_n3.py",
                          "eb72755cc396b6d12b701f04404845a0fae795aded34ffdba2272bd98d31c0c5"),
        # the executor: EVOSKILL_LLM_MAX_CONCURRENT for the model-call limiter (default 40, the constructor
        # default); the skill tool prefixes "Base directory for this skill: <dir>" (skill legs only); a solver
        # turn that hits the output-token cap is kept with LENGTH_CAP_NOTE appended and counts as one turn, never
        # retried. Compare only legs run under this sha.
        "executor_sha256": ("src/harness/direct/executor.py",
                            "13492f07739fae21dc3f25424426484548d2e4f4a6c2c6fd0d1f4a2b24f96e69"),
        "scorer_sha256": ("src/agent_profiles/ssbv2_agent/scorer.py",
                          "2442e5aa0e6172dbf62be5849a00fedf1b53e390d093d65ac7bb19da9b807f81"),
        # the shim binds a skill leg's DECLARED skills dir back read-only (skill_exposures) under the hidden
        # host dirs, so the skills' support files reach the solver's shell; a base leg's and every grading run's
        # inner script is unchanged by that.
        "shim_sha256": ("scripts/daytona/docker_shim.py",
                        "c0fe476bda68af0abf2778dcab076b503c2398e0ef489f1544ec14ef23af1e8d"),
        "entry_sha256": ("scripts/daytona/sandbox_entry.py",
                         "6d80727a66c49510d4d8b14fdbb77b7d451c76b400fbf6d1d503e0bc8dd2f851"),
    },
    "baseline_solver_prompt_sha256": "a06fb758253da19994c730d393ee550fc3992682b4697d5e48ef7befb21e08b1",
}
# THE SPLIT IS THE RUN'S DECLARATION, NOT A CODE PIN. A leg runs on the split its run declares (SPEC.json
# spec.split_path + split_sha256; the SSB split files carry no self-attested digest, so the declared sha is the
# whole-file sha256): LegConfig._check_spec refuses a --split that is not that file or does not hash to that sha, and
# the merge refuses a shard whose runtime_env.split_sha256 is not the leg's. PROJECT_SPLIT: the reported runs' split,
# 57/57/104 (unused 10), with 48 tasks dropped whose own reference workbook fails self-grading under the grader (key
# "dropped"). It is only a DEFAULT (for a script with no run to ask); nothing here refuses a split because of its sha.
PROJECT_SPLIT = ("examples/ssbv2/split.json",
                 "5be714178186e14406b70efd8d00c51f2b135b0c61038ca8b2467d0f87a1da5d")
DEFAULTS = {
    "max_sandboxes": 6, "concurrency": 30, "max_heavy_per_shard": 2,
    "leg_spend_cap_usd": 30.0, "run_spend_cap_usd": 150.0, "min_account_balance_usd": 100.0,
    "run_key_usage_start": None, "spend_every_s": 300,
    "ttl_minutes": 300, "shard_wall_min": 240,
    "ramp_first": 6, "ramp_wait_s": 300, "max_429_rate": 0.02,
    "require_snapshot_ref": EXPECTED["snapshot_ref"],
    "require_libreoffice": EXPECTED["libreoffice"],
    "store_root": str(REPO / ".cache" / "ssbv2_daytona"),
    # Where the merged tasks dir goes: <tasks_root>/ssbv2_<tag>_tasks, <repo>/.cache for a real run;
    # overridable only so tests never write into the repo.
    "tasks_root": str(REPO / ".cache"),
}
# POOL MODE keys (scripts/daytona/ssb_leg_pool.py). Accepted in RUNTIME daytona_leg next to DEFAULTS but
# NOT merged into a leg's `rt` unless declared, so a run that declares none reads exactly the dict it read before.
#   pool                 the switch (false = today's per-leg shards)      pool_keys   leg keys that pool (a test
#   pool_sandboxes       machines in the run's pool (every leg)                       leg keeps today's layout)
#   ramp_start/_step/_interval_s/_max/_min, ramp_down_min_s    the per-machine cap on in-flight attempts
#   stress_cores / stress_throttle_ratio / stress_mem_gib   what counts as a stressed machine, with any solver bash
#                        timeout (stress_on_timeout, default true); bash_watch_s > 0 also counts a command running that
#                        long (0 = off)
#   pool_max_heavy_units / pool_max_units_per_machine / unit_wall_min / pool_idle_release_s / pool_gen_check_n
#   pool_ttl_margin_min  a machine older than ttl_minutes - this takes no new unit (it retires when idle and a fresh
#                        machine replaces it if work remains), so no unit starts on a sandbox about to hit its TTL.
#                        Must be >= unit_wall_min + POOL_SYNC_MARGIN_MIN (a unit dispatched at the last moment still
#                        ends, is killed at its wall and synced before the TTL), and with the pool on ttl_minutes must
#                        leave >= POOL_MIN_USABLE_MIN on top of it (defaults: 240 + 15 = 255 -> ttl_minutes >= 315;
#                        600 leaves every machine 345 min in which it takes units)
#   pool_sandboxes       <= POOL_MAX_SANDBOXES (15 x 32 GiB = 480 GiB, under the Daytona org's 500 GiB memory cap,
#                        hit once at 12 leg sandboxes + others); a retiring machine still counts until it is deleted
#   pool_machine_recalc_budget   MACHINE-WIDE grader cap. The scorer's limits are PER PROCESS
#                        (EVOSKILL_RECALC_CONCURRENCY 4, EVOSKILL_DATATABLE_UNO_CONCURRENCY 2, sandbox_entry's
#                        EVOSKILL_HEAVY_COMPARE_CONCURRENCY 2: the leg's knobs, pinned) and every unit is its own
#                        runner process, so a machine never runs more than K = budget // recalc-per-process units
#                        at once. null = 2 x the leg's recalc-per-process (8 -> K = 2: at most 2 x today's one-runner
#                        sandbox for recalc, UNO and heavy compares alike). NOTE: with N = 3 samples per unit that is
#                        <= 6 attempts per machine, below ramp_start 8, so the attempt ramp only binds once the
#                        budget is raised (e.g. 40 -> K = 10); grader failures are then a stress signal
#                        (stress_on_grader_failure, default true)
#   smoke_stress_after_s SMOKE ONLY (a run dir whose name contains "smoke"): every machine counts as stressed from this
#                        many seconds after it became ready, so the smoke sees a raise and then a hold/down on a
#                        schedule instead of on whatever the real signals happen to do. null (default) = off
# TRAIN-FIRST / FIXED LAYOUT: one machine, train and val rollouts in one pool, a val rollout admitted only when every
# train rollout has finished or already has a spot in the pool:
#   pool_order           "interleave" (the pool's first design: heavy units first, then the legs' light units
#                        interleaved) or "train_first": the train leg's units (heavy first, then split order) come
#                        first, and a unit of any OTHER leg is dispatched only while NO train unit is still waiting
#                        (every train unit ended or already assigned/running on a machine). A train unit that goes
#                        back to the queue (lost machine) blocks the other legs again until it has a spot.
#   pool_train_first_wait_s  train_first only: a val leg that joined BEFORE its train sibling (<run>_vN_train /
#                        _base_train: same tag, "_train" for "_val") waits this long for that sibling to join, unless
#                        the sibling's result JSON already exists (then train is finished); after it, val goes ahead
#                        and the host logs why. 0 = never wait.
#   pool_fixed_inflight  null = the ramp above. An integer N = a FIXED per-machine cap of N in-flight attempts: no ramp
#                        up or down, ever; every stress signal is still computed and RECORDED (stress_log,
#                        stressed_polls), never acted on. The grader cap K must allow it (K x n_samples >= N: refused
#                        at the check-only line and by the host otherwise). <= POOL_MAX_FIXED_INFLIGHT.
#   pool_ttl_minutes     null = ttl_minutes. The Daytona TTL of POOL machines only (the shards of a non-pooled leg --
#                        the held-out test leg -- keep ttl_minutes, and so does INFRA.json leg_sandbox.ttl_minutes).
#                        The TTL rule above applies to it: pool_ttl_minutes - pool_ttl_margin_min >= 60.
# ONE LONG-LIVED POOL PER RUN: the evolver starts as soon as the training rollouts are done, validation rollouts
# overlap with it, and new training rollouts take a slot as soon as one opens (less blocking, the machine always
# busy). The pool is
# keyed by the RUN, so every epoch's legs (and the held-out test leg when "test" is in pool_keys) share one host and
# its machine(s); each leg's result is written as soon as ITS units have ended (ssb_leg_pool.py).
#   pool_keep_idle_s     when the pool has no unit queued or running and no join pending (e.g. while the next evolver
#                        runs), keep the last machine (and the host) this long for the next leg; then release it and
#                        close. 0 = release after pool_idle_release_s, as a per-epoch pool did. Cost of keeping: one
#                        idle 8 vCPU / 32 GiB machine at DAYTONA_USD_PER_SANDBOX_HOUR ($0.926/h list) for at most this
#                        long per gap; what it saves: a new machine's create + image + upload + self-tests on the
#                        critical path of the next train leg.
#   pool_idle_rollover_min  null = off. When the pool is idle and the kept machine has fewer than this many minutes left
#                        in which it takes units (ttl - pool_ttl_margin_min - its age), it is released NOW and a
#                        replacement is created at once (the old one deleted first: never two alive), so the next
#                        legs start on a fresh machine instead of draining the old one mid-leg at its retirement.
POOL_DEFAULTS = {
    "pool": False, "pool_keys": ["train", "val"], "pool_sandboxes": 12,
    "ramp_start": 8, "ramp_step": 4, "ramp_interval_s": 180, "ramp_max": 40, "ramp_min": 4,
    "ramp_down_min_s": 60, "stress_cores": 6.0, "stress_throttle_ratio": 0.25, "stress_mem_gib": 24.0,
    "bash_watch_s": 0, "pool_max_heavy_units": 1, "pool_max_units_per_machine": 16,
    "unit_wall_min": 240, "pool_idle_release_s": 120, "pool_gen_check_n": None, "pool_ttl_margin_min": 255,
    "stress_on_timeout": True, "stress_on_grader_failure": True, "pool_machine_recalc_budget": None,
    "smoke_stress_after_s": None,
    "pool_order": "interleave", "pool_train_first_wait_s": 300, "pool_fixed_inflight": None,
    "pool_ttl_minutes": None,
    "pool_keep_idle_s": 3600, "pool_idle_rollover_min": None,
}
POOL_ORDERS = ("interleave", "train_first")
POOL_MAX_FIXED_INFLIGHT = 256
POOL_MAX_SANDBOXES = 15
POOL_SYNC_MARGIN_MIN = 15
POOL_MIN_USABLE_MIN = 60
# The loop's env must carry these, equal to the baseline's LEG_SETTINGS (two sources must agree).
SOLVER_PIN_KEYS = ("EVOSKILL_SOLVER_MODEL", "EVOSKILL_SOLVER_PROVIDER_ORDER",
                   "EVOSKILL_SOLVER_PROVIDER_ONLY", "EVOSKILL_SOLVER_ALLOW_FALLBACKS",
                   "EVOSKILL_SOLVER_REQUIRE_PARAMETERS", "EVOSKILL_SOLVER_PROVIDER_QUANTIZATIONS",
                   "EVOSKILL_SOLVER_EXPECT_PROVIDER", "EVOSKILL_SOLVER_EXPECT_MODEL")
# Fields every shard's runner result must agree on (else the leg is not one condition).
SAME_RESULT = ("model", "provider_order", "budget", "provider_extra", "allow_fallbacks",
               "expect_provider", "expect_model", "n_samples", "seed", "solver_image",
               "solver_image_id", "solver_formula_engine", "solver_prompt_sha256_12", "skills_dir",
               "solver_wall_s", "solver_wall_derived_default_s", "solver_wall_derivation",
               "concurrency")
SAME_RUNTIME = ("image", "libreoffice", "grading_openpyxl", "runner_python", "solver_stack",
                "split_sha256", "runner_sha256", "executor_sha256", "scorer_sha256",
                "solver_prompt_sha256", "shim_sha256", "entry_sha256", "knobs", "thread_env",
                "arch")

EXIT_OK, EXIT_REFUSED, EXIT_LEG_FAILED, EXIT_DRY_RUN = 0, 4, 5, 7

# THE ONLY KNOBS A LEG MAY SET DIFFERENTLY FROM THE HELD-OUT BASELINE (base.KNOBS), and why: the scorer's
# LibreOffice recalculation timeout and its data-table (UNO)
# solve timeout, raised to 900 s from the defaults 180 / 300. A timeout only acts when a recalculation
# runs longer than it; the held-out baseline and this run's e0 legs graded with the defaults and
# recorded 0 timeouts, so every verdict they hold is one a 900 s grader produces too. Raising them can
# only turn a would-be timeout (a grader failure) into a completed grade; nothing that finished
# changes. Every OTHER knob (concurrency, heavy-compare, cgroup, cpu-count) stays the baseline's.
ALLOWED_KNOB_DELTA = infra_cfg.GRADER_TIMEOUT_KNOBS


class Refused(Exception):
    """A configuration refusal: nothing was created, nothing was spent."""


class Abort(Exception):
    pass


def knob_drift(knobs) -> dict:
    """{knob: (baseline, this leg)} for every knob that differs from the held-out baseline's KNOBS,
    except ALLOWED_KNOB_DELTA. {} = the leg is the baseline's configuration up to the two timeouts."""
    knobs = dict(knobs or {})
    return {k: (base.KNOBS.get(k), knobs.get(k)) for k in sorted(set(knobs) | set(base.KNOBS))
            if k not in ALLOWED_KNOB_DELTA and knobs.get(k) != base.KNOBS.get(k)}


def load_infra(run_dir: Path) -> tuple:
    """(INFRA dict or None, record or None) for <run_dir>/INFRA.json. Invalid = Refused."""
    f = Path(run_dir) / infra_cfg.INFRA_FILE
    if not f.exists():
        return None, None
    try:
        inf = infra_cfg.load(f)
    except infra_cfg.InfraError as e:
        raise Refused(f"{f} does not validate: {e}")
    return inf, {"file": str(f), "sha256": sha256_file(f),
                 "grader_env": infra_cfg.grader_env(inf)}


# The loop's RUNTIME prompt_mode values (src/orchestrator/loop.py PROMPT_MODES; a test pins the two equal).
PROMPT_MODES = ("ao", "template")


def runtime_prompt_mode(run_dir: Path) -> str:
    """RUNTIME.json `prompt_mode` of the run: absent (or no RUNTIME.json) = "ao"; "template" = the loop
    renders the evolver prompt and INFRA.json is REQUIRED (see LegConfig). Any other value -- a typo, a
    boolean -- is Refused, never read as either mode (the loop refuses it the same way)."""
    f = Path(run_dir) / "RUNTIME.json"
    if not f.is_file():
        return "ao"
    try:
        rt = json.loads(f.read_text())
    except json.JSONDecodeError as e:
        raise Refused(f"{f} is not readable JSON: {e}")
    if not isinstance(rt, dict):
        raise Refused(f"{f} must hold a JSON object")
    mode = rt.get("prompt_mode", "ao")
    if mode not in PROMPT_MODES:
        raise Refused(f"RUNTIME.json prompt_mode={mode!r} is not one of {PROMPT_MODES} (absent = 'ao')")
    return mode


def declared_split(run_dir: Path) -> tuple:
    """(split file, declared sha256 or None) from <run_dir>/SPEC.json spec.split_path / split_sha256 --
    the run's split, the only one its legs may run on (see PROJECT_SPLIT). (None, None) without a
    SPEC.json or a split_path in it."""
    f = Path(run_dir) / "SPEC.json"
    if not f.is_file():
        return None, None
    try:
        sp = json.loads(f.read_text()).get("spec", {})
    except (json.JSONDecodeError, AttributeError) as e:
        raise Refused(f"{f} is not a readable spec: {e}")
    if not sp.get("split_path"):
        return None, None
    return Path(str(sp["split_path"])).resolve(), (sp.get("split_sha256") or None)


def default_split(run_dir: Path | None = None) -> Path:
    """The split a command uses when none is passed: the run's declared split when `run_dir` has a
    SPEC.json naming one, else PROJECT_SPLIT."""
    if run_dir is not None:
        p, _sha = declared_split(Path(run_dir))
        if p is not None:
            return p
    return REPO / PROJECT_SPLIT[0]


def leg_knobs(inf, env: dict) -> dict:
    """The knobs every sandbox of this leg gets: the baseline's KNOBS plus, when the run has an
    INFRA.json, its two grader timeouts. A different value for either already in the caller's
    environment is REFUSED, never silently overwritten."""
    knobs = dict(base.KNOBS)
    if inf is None:
        return knobs
    g = infra_cfg.grader_env(inf)
    clash = {k: (env[k], v) for k, v in g.items()
             if k in env and not infra_cfg.same_env_value(k, v, env[k])}
    if clash:
        raise Refused(f"the environment sets grader timeout(s) that disagree with INFRA.json "
                      f"(env value, INFRA value): {clash} — unset them or change INFRA.json")
    knobs.update(g)
    drift = knob_drift(knobs)
    if drift:
        raise Refused(f"leg knobs differ from the held-out baseline's beyond "
                      f"{list(ALLOWED_KNOB_DELTA)}: {drift}")
    return knobs


def data_overlay_from_infra(inf, env: dict, *, set_process_env: bool = True):
    """The PATCHES.json whose repaired bytes this process uploads / hashes, or None.

    No INFRA.json (AO mode): None, and data_overlay keeps reading the bare EVOSKILL_SSBV2_DATA_OVERLAY
    of this process. With INFRA.json its block "data" decides:
      * overlay declared: the variable in `env` (the caller's) or in os.environ (the one data_overlay
        falls back to) naming ANOTHER file is refused; PATCHES.json must hash to overlay_patches_sha256
        and every entry verify now (data_overlay.verify); os.environ gets the path when unset
        (set_process_env) and it is returned;
      * none declared (null): the variable set (non-empty) in either is refused; None.
    Refused, never silently overwritten (the leg_knobs rule)."""
    name = data_overlay.ENV
    if inf is None:
        return None
    d = inf["data"]
    want = d["overlay_patches_json"]
    have = {w: v for w, v in (("caller env", env.get(name)), ("process env", os.environ.get(name))) if v}
    if want is None:
        if have:
            raise Refused(f"INFRA.json declares no data overlay (data.overlay_patches_json null) but "
                          f"{name} is set {have} — unset it or declare the overlay in INFRA.json")
        return None
    clash = {w: v for w, v in have.items() if not data_overlay.same_path(v, want)}
    if clash:
        raise Refused(f"{name} {clash} differs from INFRA.json data.overlay_patches_json {want!r} — "
                      f"unset it or change INFRA.json")
    try:
        sha = sha256_file(want)
    except OSError as e:
        raise Refused(f"INFRA.json data.overlay_patches_json {want}: {e}")
    if sha != d["overlay_patches_sha256"]:
        raise Refused(f"{want} sha256 {sha[:12]} != INFRA.json data.overlay_patches_sha256 "
                      f"{d['overlay_patches_sha256'][:12]}: the overlay changed since it was declared")
    try:
        data_overlay.verify(want)
    except data_overlay.OverlayError as e:
        raise Refused(f"the data overlay INFRA.json declares does not verify: {e}")
    if set_process_env and not os.environ.get(name):
        os.environ[name] = want
    return want


def overlay_note(cfg) -> str:
    """One line: which overlay this leg uses and who declared it (check-only line, logs)."""
    if cfg.infra is not None:
        return f"data overlay {cfg.data_overlay or 'none'} (INFRA.json data)"
    return f"data overlay {os.environ.get(data_overlay.ENV) or 'none'} ({data_overlay.ENV}, no INFRA.json)"


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def log_line(msg: str, fh=None) -> None:
    line = f"{dlx.now()} {msg}"
    print(line, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


# ───────────────────────────────────────────────────────────────────────────── configuration
def leg_code(tag: str) -> str:
    """Short, DNS-safe leg label for sandbox names: v1tr / v2va / b0tr / h<sha8>."""
    m = re.search(r"_v(\d+)_(train|val|test)$", tag)
    if m:
        return f"v{m.group(1)}{m.group(2)[:2]}"
    m = re.search(r"_base_(train|val|test)$", tag)
    if m:
        return f"b0{m.group(1)[:2]}"
    return "h" + hashlib.sha256(tag.encode()).hexdigest()[:8]


def load_runtime_block(run_dir: Path) -> dict:
    rt = {}
    f = run_dir / "RUNTIME.json"
    if f.is_file():
        try:
            rt = json.loads(f.read_text())
        except json.JSONDecodeError as e:
            raise Refused(f"{f} is not readable JSON: {e}")
    blk = rt.get("daytona_leg") or {}
    if not isinstance(blk, dict):
        raise Refused("RUNTIME.json daytona_leg must be an object")
    unknown = sorted(k for k in blk if not k.startswith("_") and k not in DEFAULTS and k not in POOL_DEFAULTS)
    if unknown:
        raise Refused(f"RUNTIME.json daytona_leg has unknown key(s) {unknown}; known "
                      f"{sorted(DEFAULTS)} and pool keys {sorted(POOL_DEFAULTS)}")
    out = dict(DEFAULTS)
    out.update({k: v for k, v in blk.items() if not k.startswith("_")})
    return out


def pool_settings(rt: dict) -> dict:
    """The pool block of a leg's `rt` (POOL_DEFAULTS + what RUNTIME declares), validated; Refused on a bad value
    (checked on every leg, so a typo'd pool block refuses before anything is created, pool on or off)."""
    p = dict(POOL_DEFAULTS)
    p.update({k: rt[k] for k in POOL_DEFAULTS if k in rt})
    bad = []
    for k in ("pool", "stress_on_timeout", "stress_on_grader_failure"):
        if not isinstance(p[k], bool):
            bad.append(f"{k}={p[k]!r} (true/false)")
    if not (isinstance(p["pool_keys"], list) and p["pool_keys"]
            and set(p["pool_keys"]) <= {"train", "val", "test"}):
        bad.append(f"pool_keys={p['pool_keys']!r}")
    ints = ("pool_sandboxes", "ramp_start", "ramp_step", "ramp_max", "ramp_min", "pool_max_heavy_units",
            "pool_max_units_per_machine")
    for k in ints:
        if not isinstance(p[k], int) or isinstance(p[k], bool) or p[k] < 1:
            bad.append(f"{k}={p[k]!r} (an integer >= 1)")
    for k in ("ramp_interval_s", "ramp_down_min_s", "stress_cores", "stress_throttle_ratio", "stress_mem_gib",
              "bash_watch_s", "unit_wall_min", "pool_idle_release_s", "pool_ttl_margin_min",
              "pool_train_first_wait_s", "pool_keep_idle_s"):
        if not isinstance(p[k], (int, float)) or isinstance(p[k], bool) or p[k] < 0:
            bad.append(f"{k}={p[k]!r} (a number >= 0)")
    if p["pool_order"] not in POOL_ORDERS:
        bad.append(f"pool_order={p['pool_order']!r} (one of {list(POOL_ORDERS)})")
    fx = p["pool_fixed_inflight"]
    if fx is not None and (not isinstance(fx, int) or isinstance(fx, bool) or not 1 <= fx <= POOL_MAX_FIXED_INFLIGHT):
        bad.append(f"pool_fixed_inflight={fx!r} (null or an integer in [1, {POOL_MAX_FIXED_INFLIGHT}])")
    pt = p["pool_ttl_minutes"]
    if pt is not None and (not isinstance(pt, int) or isinstance(pt, bool) or not 10 <= pt <= 1440):
        bad.append(f"pool_ttl_minutes={pt!r} (null or an integer in [10, 1440], the TTL range INFRA.json allows)")
    if not bad:
        if not p["ramp_min"] <= p["ramp_start"] <= p["ramp_max"] <= 64:
            bad.append(f"need ramp_min <= ramp_start <= ramp_max <= 64 (got {p['ramp_min']}, {p['ramp_start']}, "
                       f"{p['ramp_max']})")
        if p["pool_sandboxes"] > POOL_MAX_SANDBOXES:
            bad.append(f"pool_sandboxes={p['pool_sandboxes']} (> {POOL_MAX_SANDBOXES}: {p['pool_sandboxes']} x "
                       f"{RESOURCES['memory']} GiB would pass the Daytona org's 500 GiB memory cap)")
        if not 0 < p["stress_cores"] <= RESOURCES["cpu"]:
            bad.append(f"stress_cores={p['stress_cores']} (0 < x <= {RESOURCES['cpu']} vCPU)")
        need = float(p["unit_wall_min"]) + POOL_SYNC_MARGIN_MIN
        if p["pool_ttl_margin_min"] < need:
            bad.append(f"pool_ttl_margin_min={p['pool_ttl_margin_min']} < unit_wall_min {p['unit_wall_min']} + "
                       f"{POOL_SYNC_MARGIN_MIN} min sync margin: a unit dispatched just before its machine retires "
                       f"could outlive the machine's TTL")
        ttl, ttl_key = pool_ttl(rt, p)
        roll = p["pool_idle_rollover_min"]
        if roll is not None and isinstance(roll, int) and isinstance(ttl, (int, float)) \
                and roll >= ttl - p["pool_ttl_margin_min"]:
            bad.append(f"pool_idle_rollover_min={roll} >= the {ttl - p['pool_ttl_margin_min']} min a machine takes "
                       f"units ({ttl_key} {ttl} - pool_ttl_margin_min {p['pool_ttl_margin_min']}): every machine "
                       f"would be rolled over")
        if p["pool"] and isinstance(ttl, (int, float)) and ttl - p["pool_ttl_margin_min"] < POOL_MIN_USABLE_MIN:
            bad.append(f"{ttl_key}={ttl} - pool_ttl_margin_min={p['pool_ttl_margin_min']} leaves "
                       f"{ttl - p['pool_ttl_margin_min']} min in which a machine takes units "
                       f"(< {POOL_MIN_USABLE_MIN}); raise {ttl_key} to >= "
                       f"{p['pool_ttl_margin_min'] + POOL_MIN_USABLE_MIN} (e.g. 600) or lower unit_wall_min")
    for k in ("pool_gen_check_n", "pool_machine_recalc_budget", "pool_idle_rollover_min"):
        if p[k] is not None and (not isinstance(p[k], int) or isinstance(p[k], bool) or p[k] < 1):
            bad.append(f"{k}={p[k]!r} (null or an integer >= 1)")
    s = p["smoke_stress_after_s"]
    if s is not None and (not isinstance(s, (int, float)) or isinstance(s, bool) or s <= 0):
        bad.append(f"smoke_stress_after_s={s!r} (null or a number > 0)")
    if bad:
        raise Refused("RUNTIME.json daytona_leg pool settings invalid: " + "; ".join(bad))
    return p


def pool_ttl(rt: dict, p: dict) -> tuple:
    """(TTL minutes of a POOL machine, the key it comes from): pool_ttl_minutes when declared, else ttl_minutes (the
    shards' TTL, which INFRA.json leg_sandbox.ttl_minutes restates)."""
    if p.get("pool_ttl_minutes") is not None:
        return p["pool_ttl_minutes"], "pool_ttl_minutes"
    return rt.get("ttl_minutes", DEFAULTS["ttl_minutes"]), "ttl_minutes"


def pool_enabled(cfg) -> bool:
    """Does this leg run in the run's machine pool (RUNTIME daytona_leg.pool and its key in pool_keys)?"""
    p = pool_settings(cfg.rt)
    return bool(p["pool"]) and cfg.key in p["pool_keys"]


class LegConfig:
    def __init__(self, *, tag, key, run_dir, split_file, cache_dir, skills_dir, n_samples,
                 concurrency, env=None, allow_test=False):
        process_env = env is None               # only then is the data overlay set in os.environ
        env = dict(os.environ if env is None else env)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", tag or ""):
            raise Refused(f"bad tag {tag!r}")
        if key not in ("train", "val") and not (key == "test" and allow_test):
            raise Refused(f"key {key!r}: a loop leg is train or val (test needs --allow-test)")
        self.tag, self.key = tag, key
        self.run_dir = Path(run_dir).resolve()
        if not self.run_dir.is_dir():
            raise Refused(f"no run dir {self.run_dir}")
        self.rt = load_runtime_block(self.run_dir)
        self.split_file = Path(split_file).resolve()
        if not self.split_file.is_file():
            raise Refused(f"no split file {self.split_file}")
        split = json.loads(self.split_file.read_text())
        if not isinstance(split.get(key), list) or not split[key]:
            raise Refused(f"split {self.split_file.name} has no non-empty {key!r} list")
        self.tasks = list(split[key])
        self.split_sha256 = sha256_file(self.split_file)
        self.split_status = split.get("status")
        self.n_samples = int(n_samples)
        if self.n_samples < 1:
            raise Refused("n_samples must be >= 1")
        self.concurrency = int(concurrency)
        # EVOSKILL_LEG_MAX_CONCURRENCY (default 30): raised only to measure a one-machine layout above 30.
        conc_cap = int(os.environ.get("EVOSKILL_LEG_MAX_CONCURRENCY", "30"))
        if not 1 <= self.concurrency <= conc_cap:
            raise Refused(f"concurrency {self.concurrency}: per-sandbox concurrency is capped at "
                          f"{conc_cap} (Daytona 8/32/40 sizing)")
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.is_absolute():
            self.cache_dir = REPO / self.cache_dir
        self.cache_dir = Path(os.path.abspath(self.cache_dir))
        if (self.cache_dir / "PROVIDER_VIOLATION").exists():
            raise Refused(f"{self.cache_dir}/PROVIDER_VIOLATION exists — this cache dir is "
                          f"quarantined (a leg in it was served by an unexpected provider/model)")
        self.skills_dir = None
        if skills_dir:
            sd = Path(skills_dir).resolve()
            if not sd.is_dir() or sd.is_symlink():
                raise Refused(f"skills dir {sd} is absent (or a symlink) — a skill leg needs its "
                              f"promoted library")
            names = sorted(p.parent.name for p in sd.glob("*/SKILL.md"))
            if not names:
                raise Refused(f"skills dir {sd} has no */SKILL.md — refusing to run a skill leg "
                              f"that would silently degenerate into a base leg")
            self.skills_dir = sd
        # Two sources for the solver's pins must agree: the loop's env (RUNTIME solver_env via
        # loop._export_solver_env) and the baseline's LEG_SETTINGS that every Daytona leg runs.
        missing = [k for k in SOLVER_PIN_KEYS if not env.get(k)]
        if missing:
            raise Refused(f"the leg env lacks solver pin(s) {missing} — declare them (RUNTIME "
                          f"solver_env) so the record shows what the leg was told")
        diff = {k: (env[k], base.LEG_SETTINGS.get(k)) for k in SOLVER_PIN_KEYS
                if env[k] != base.LEG_SETTINGS.get(k)}
        if diff:
            raise Refused(f"solver pins disagree with the baseline's LEG_SETTINGS (a Daytona leg "
                          f"runs exactly those): {diff}")
        self.leg_settings = dict(base.LEG_SETTINGS)
        self.env_pins = {k: env[k] for k in SOLVER_PIN_KEYS}   # recorded: what the loop sent
        # The operator's INFRA.json, when the run has one: its solver pins are a THIRD statement
        # of the same facts and must agree too; its grader timeouts join the knobs.
        self.infra, self.infra_record = load_infra(self.run_dir)
        # In a template-mode run INFRA.json is REQUIRED. Without it this
        # leg would grade at the baseline's 180/300 s timeouts while every other part of the run reads
        # 900/900 from the operator's declaration -- a silent second configuration.
        self.prompt_mode = runtime_prompt_mode(self.run_dir)
        if self.prompt_mode == "template" and self.infra is None:
            raise Refused(f"RUNTIME.json prompt_mode is 'template' but {self.run_dir / infra_cfg.INFRA_FILE} "
                          f"does not exist: a template-mode run takes its grader timeouts (and every "
                          f"other infra fact) from the operator's INFRA.json, so a leg without it "
                          f"is refused rather than graded at the baseline's 180/300 s")
        if self.infra is not None:
            want = infra_cfg.solver_env(self.infra)
            bad = {k: (want.get(k), base.LEG_SETTINGS.get(k)) for k in SOLVER_PIN_KEYS
                   if not infra_cfg.same_env_value(k, want.get(k), base.LEG_SETTINGS.get(k))}
            if bad:
                raise Refused(f"INFRA.json solver pins disagree with the baseline's LEG_SETTINGS "
                              f"(INFRA, LEG_SETTINGS): {bad}")
        self.knobs = leg_knobs(self.infra, env)
        # The data overlay: INFRA.json "data" when the run has one, else the bare variable.
        self.data_overlay = data_overlay_from_infra(self.infra, env, set_process_env=process_env)
        self.spec = self._check_spec()
        df_sha = sha256_file(DOCKERFILE)
        if df_sha != EXPECTED["dockerfile_sha256"]:
            raise Refused(f"{DOCKERFILE.name} sha256 {df_sha[:12]} != the baseline's "
                          f"{EXPECTED['dockerfile_sha256'][:12]}: a different image is a different "
                          f"comparability class")
        self.dockerfile_sha256 = df_sha
        self.image_id = "x86daytona-" + df_sha[:12]
        drift = {k: (rel, sha256_file(REPO / rel)[:12], want[:12])
                 for k, (rel, want) in EXPECTED["baseline_runtime"].items()
                 if sha256_file(REPO / rel) != want}
        if drift:
            raise Refused(f"solver code differs from the held-out baseline's (a leg on other code is "
                          f"another condition): {drift}")
        self.store = Path(self.rt["store_root"]) / self.run_dir.name / tag
        self.code = leg_code(tag)

    def _check_spec(self) -> dict:
        f = self.run_dir / "SPEC.json"
        if not f.is_file():
            raise Refused(f"no frozen SPEC.json in {self.run_dir} — a loop leg runs under a spec")
        sp = json.loads(f.read_text()).get("spec", {})
        probs = []
        if sp.get("solver_model") != self.leg_settings["EVOSKILL_SOLVER_MODEL"]:
            probs.append(f"solver_model {sp.get('solver_model')!r}")
        order = ",".join(sp.get("solver_provider_order") or [])
        if order != self.leg_settings["EVOSKILL_SOLVER_PROVIDER_ORDER"]:
            probs.append(f"solver_provider_order {order!r}")
        if sp.get("solver_allow_fallbacks"):
            probs.append("solver_allow_fallbacks True")
        if int(sp.get("agent_turns") or 0) != int(self.leg_settings["EVOSKILL_TURN_BUDGET"]):
            probs.append(f"agent_turns {sp.get('agent_turns')}")
        if int(sp.get("n_samples") or 0) != self.n_samples:
            probs.append(f"n_samples {sp.get('n_samples')} vs leg {self.n_samples}")
        if Path(str(sp.get("split_path") or "")).resolve() != self.split_file:
            probs.append(f"split_path {sp.get('split_path')}")
        # The run's split is a comparability fact (see PROJECT_SPLIT): the file must still be the one
        # the spec hashed. The merge holds every shard to this same sha.
        if sp.get("split_sha256") and sp["split_sha256"] != self.split_sha256:
            probs.append(f"split_sha256 {str(sp['split_sha256'])[:12]} but {self.split_file.name} "
                         f"hashes to {self.split_sha256[:12]}")
        if probs:
            raise Refused("the frozen spec disagrees with this Daytona leg: " + "; ".join(probs))
        return sp

    @property
    def tasks_dir_local(self) -> Path:
        return Path(self.rt["tasks_root"]) / f"ssbv2_{self.tag}_tasks"

    @property
    def result_path(self) -> Path:
        return self.run_dir / f"{self.tag}.json"

    @property
    def failed_path(self) -> Path:
        return self.run_dir / f"{self.tag}.FAILED.json"


# ───────────────────────────────────────────────────────────────────────────── planning
def task_index() -> dict:
    return base.load_index()


def make_plan(tasks: list, idx: dict, *, n_samples: int, concurrency: int, max_sandboxes: int,
              max_heavy: int, heavy_bytes: int = base.HEAVY_BYTES) -> dict:
    """Shards over <= max_sandboxes sandboxes, ~concurrency episodes each; heavy tasks spread
    round-robin with <= max_heavy per shard; light tasks balanced by (count, bytes)."""
    size = {t: max(idx[t]["input"].stat().st_size, idx[t]["golden"].stat().st_size) for t in tasks}
    weight = {t: idx[t]["input"].stat().st_size + idx[t]["golden"].stat().st_size for t in tasks}
    heavy = [t for t in tasks if size[t] >= heavy_bytes]
    light = [t for t in tasks if t not in set(heavy)]
    want = max(1, math.ceil(len(tasks) * n_samples / max(1, concurrency)))
    need_h = math.ceil(len(heavy) / max_heavy) if heavy else 1
    n = max(1, min(len(tasks), max(want, need_h)))
    if n > max_sandboxes:
        if need_h > max_sandboxes:
            raise Refused(f"{len(heavy)} heavy tasks need >= {need_h} sandboxes for <= {max_heavy} "
                          f"per sandbox; max_sandboxes is {max_sandboxes}")
        n = max_sandboxes
    shards = [[] for _ in range(n)]
    for i, t in enumerate(heavy):
        shards[i % n].append(t)
    for t in sorted(light, key=lambda x: -weight[x]):
        j = min(range(n), key=lambda k: (len(shards[k]), sum(weight[x] for x in shards[k])))
        shards[j].append(t)
    order = {t: i for i, t in enumerate(tasks)}
    return {"shards": {f"s{i + 1:02d}": sorted(s, key=lambda t: order[t])
                       for i, s in enumerate(shards) if s},
            "heavy": heavy, "n_tasks": len(tasks), "n_samples": n_samples,
            "episodes_per_shard_max": max(len(s) for s in shards) * n_samples,
            "concurrency": concurrency, "max_heavy_per_shard": max_heavy}


# ───────────────────────────────────────────────────────────────────────────── cache seeding
def _record_task(p: Path) -> str | None:
    try:
        d = json.loads(p.read_text())
        return json.loads(d["cache_key"]["question"]).get("task_id")
    except Exception:  # noqa: BLE001 — not a rollout record
        return None


def seed_files(cache_dir: Path, tasks: list) -> list:
    """Local cache files a shard is seeded with: the CACHE_ENV marker and every rollout record of
    ITS tasks (so a relaunch replays them). Sidecars and other tasks' records stay home."""
    if not cache_dir.is_dir():
        return []
    want = set(tasks)
    out = []
    marker = cache_dir / "CACHE_ENV"
    if marker.is_file():
        out.append(marker)
    for p in sorted(cache_dir.rglob("*.json")):
        if p.parent == cache_dir:
            continue
        if _record_task(p) in want:
            out.append(p)
    return out


# ───────────────────────────────────────────────────────────────────────────── bundle
def _add_file(tf, src: Path, arc: str) -> int:
    tf.add(str(src), arcname=arc, recursive=False)
    return src.stat().st_size


def _add_bytes(tf, arc: str, data: bytes) -> None:
    ti = tarfile.TarInfo(arc)
    ti.size, ti.mtime, ti.mode = len(data), int(time.time()), 0o644
    tf.addfile(ti, io.BytesIO(data))


def _mirror(p: Path) -> str:
    return str(Path(p).resolve() if not Path(p).is_absolute() else Path(p)).lstrip("/")


def build_bundle(cfg: LegConfig, tasks: list, leg_env: str, out: Path, seeds: list,
                 idx: dict | None = None, extra=()) -> dict:
    """Repo src/ + runner + the split file + this shard's dataset files + the SKILLS DIR + seeded
    cache files, all at their Mac absolute paths; the unchanged sandbox scripts under opt/leg.
    `extra` = [(source file, archive name)] added after them (the pool's machine scripts; empty for a shard)."""
    idx = idx or task_index()
    n, nbytes, skills = 0, 0, []
    with tarfile.open(out, "w:gz") as tf:
        for p in sorted((REPO / "src").rglob("*")):
            if "__pycache__" in p.parts or p.suffix == ".pyc" or not p.is_file():
                continue
            nbytes += _add_file(tf, p, _mirror(p))
            n += 1
        for p in (REPO / RUNNER_REL, cfg.split_file):
            nbytes += _add_file(tf, p, _mirror(p))
            n += 1
        for cat in CATEGORIES:
            _add_file(tf, DATA / cat / "dataset.json", _mirror(DATA / cat / "dataset.json"))
            n += 1
        for t in tasks:
            for k in ("input", "golden"):
                # bytes from the data overlay (repaired files), path unchanged: prompt, cache keys and
                # the sandbox's view stay the canonical ones (src/orchestrator/data_overlay.py)
                nbytes += _add_file(tf, data_overlay.source(idx[t][k], cfg.data_overlay),
                                    _mirror(idx[t][k]))
                n += 1
        if cfg.skills_dir is not None:
            for p in sorted(cfg.skills_dir.rglob("*")):
                if p.is_symlink():
                    raise Refused(f"skills dir holds a symlink {p} — refusing (it would resolve "
                                  f"differently inside the sandbox)")
                if p.is_file():
                    nbytes += _add_file(tf, p, _mirror(p))
                    skills.append(str(p.relative_to(cfg.skills_dir)))
                    n += 1
        for p in seeds:
            _add_file(tf, p, _mirror(p))
            n += 1
        for f in SANDBOX_SCRIPTS:
            _add_file(tf, HERE / f, "opt/leg/" + f)
            n += 1
        for src, arc in extra:
            _add_file(tf, Path(src), arc)
            n += 1
        _add_bytes(tf, "opt/leg/leg.env", leg_env.encode())
        _add_bytes(tf, "opt/leg/subset.json", json.dumps(tasks).encode())
        n += 2
    return {"files": n, "payload_bytes": nbytes, "bundle_bytes": out.stat().st_size,
            "bundle_sha256": sha256_file(out), "seeded_cache_files": len(seeds),
            "skills_files": skills}


def leg_env_text(cfg: LegConfig) -> str:
    env = dict(cfg.leg_settings)
    env.update(cfg.knobs)                   # base.KNOBS (+ INFRA's grader timeouts, see leg_knobs)
    env.update(base.THREAD_ENV)
    if cfg.skills_dir is not None:
        env.pop("EVOSKILL_BASE_LEG", None)
        env["EVOSKILL_SKILLS_DIR"] = str(cfg.skills_dir)
    env.update({
        "EVOSKILL_OR_KEY": "env",
        "EVOSKILL_N_SAMPLES": str(cfg.n_samples),
        "EVOSKILL_CONCURRENCY": str(cfg.concurrency),
        "EVOSKILL_SPLIT_FILE": str(cfg.split_file),
        "EVOSKILL_SPLIT_KEY": cfg.key,
        "EVOSKILL_TASK_SUBSET": "/opt/leg/subset.json",
        "EVOSKILL_RUN_TAG": cfg.tag,
        "EVOSKILL_CACHE_DIR": str(cfg.cache_dir),
        "EVOSKILL_RUNTIME_ENV_FILE": "/opt/leg/runtime_env.json",
        "EVOSHIM_IMAGE_ID": cfg.image_id,
        "EVOSHIM_EXTRA_ENV": json.dumps(base.THREAD_ENV, sort_keys=True),
        "EVOSKILL_LLM_STATS_FILE": "/var/log/evoshim/llm_stats.json",
    })
    # The model-call limiter follows the rollout concurrency only when it exceeds the executor's default 40
    # (a one-machine layout above 30); every normal leg (<= 30) leaves it unset, i.e. unchanged.
    if cfg.concurrency > 40:
        env["EVOSKILL_LLM_MAX_CONCURRENT"] = str(cfg.concurrency)
    assert "OPENROUTER_API_KEY" not in env
    return "".join(f"export {k}={shlex.quote(str(v))}\n" for k, v in sorted(env.items()))


def skill_mds(sd: Path) -> list:
    """<sd>/<name>/SKILL.md for each <name> NOT starting with '.': the rule the sandbox side of the
    upload check applies (`ls <sd>/*/SKILL.md`: a shell glob skips dot-dirs). Python's Path.glob
    matches dot-dirs, so without this rule a .x/SKILL.md is counted locally and never in the
    sandbox -- a false "skills upload incomplete"."""
    return sorted(p for p in sd.glob("*/SKILL.md") if not p.parent.name.startswith("."))


def skills_fingerprint(sd: Path | None) -> dict | None:
    if sd is None:
        return None
    h = hashlib.sha256()
    files = sorted(p for p in sd.rglob("*") if p.is_file())
    for p in files:
        h.update(str(p.relative_to(sd)).encode() + b"\0" + p.read_bytes() + b"\0")
    return {"dir": str(sd), "sha256": h.hexdigest(),
            "skills": sorted(p.parent.name for p in sd.glob("*/SKILL.md")), "files": len(files)}


# ───────────────────────────────────────────────────────────────────────────── spend guard
def openrouter_usage(key: str) -> dict:
    import httpx
    h = {"Authorization": f"Bearer {key}"}
    k = httpx.get("https://openrouter.ai/api/v1/key", headers=h, timeout=30).json()["data"]
    out = {"ts": dlx.now(), "key_usage": float(k.get("usage") or 0.0),
           "limit_remaining": k.get("limit_remaining"), "balance": None}
    try:
        c = httpx.get("https://openrouter.ai/api/v1/credits", headers=h, timeout=30).json()["data"]
        out["balance"] = round(float(c["total_credits"]) - float(c["total_usage"]), 4)
    except Exception:  # noqa: BLE001
        pass
    return out


class SpendGuard:
    """Key-usage delta for THIS leg, the RUN's delta since its origin, and the account floor."""

    def __init__(self, fetch, *, leg_cap: float, run_cap: float, min_balance: float,
                 origin_file: Path, origin_usage=None, log=print):
        self.fetch, self.log = fetch, log
        self.leg_cap, self.run_cap, self.min_balance = float(leg_cap), float(run_cap), float(min_balance)
        self.start = fetch()
        self.last = dict(self.start)
        self.breached = threading.Event()
        self.reason = ""
        if origin_usage is None:
            origin_file.parent.mkdir(parents=True, exist_ok=True)
            try:                                  # first leg of the run records the origin
                fd = os.open(str(origin_file), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
                with os.fdopen(fd, "w") as fh:
                    json.dump({"key_usage": self.start["key_usage"], "ts": self.start["ts"],
                               "note": "run spend origin; recorded by the first Daytona leg"}, fh)
            except FileExistsError:
                pass
            origin_usage = float(json.loads(origin_file.read_text())["key_usage"])
        self.origin = float(origin_usage)
        self._judge("start")

    def leg_delta(self) -> float:
        return round(self.last["key_usage"] - self.start["key_usage"], 6)

    def run_delta(self) -> float:
        return round(self.last["key_usage"] - self.origin, 6)

    def _judge(self, where: str) -> None:
        bal = self.last.get("balance")
        why = []
        if self.leg_delta() > self.leg_cap:
            why.append(f"leg delta ${self.leg_delta():.4f} > cap ${self.leg_cap}")
        if self.run_delta() > self.run_cap:
            why.append(f"run delta ${self.run_delta():.4f} > cap ${self.run_cap}")
        if bal is not None and bal < self.min_balance:
            why.append(f"account balance ${bal} < floor ${self.min_balance}")
        self.log(f"[SPEND] {where} leg=${self.leg_delta():.4f}/{self.leg_cap} "
                 f"run=${self.run_delta():.4f}/{self.run_cap} balance={bal}")
        if why and not self.breached.is_set():
            self.reason = "; ".join(why)
            self.log(f"[SPEND] BREACH: {self.reason} — aborting every shard")
            self.breached.set()

    def check(self, where: str = "") -> None:
        try:
            self.last = self.fetch()
        except Exception as e:  # noqa: BLE001 — a failed reading is logged; the next decides
            self.log(f"[SPEND] check failed ({where}): {type(e).__name__}: {str(e)[:120]}")
            return
        self._judge(where)

    def record(self) -> dict:
        return {"leg_delta_usd": self.leg_delta(), "run_delta_usd": self.run_delta(),
                "leg_cap_usd": self.leg_cap, "run_cap_usd": self.run_cap,
                "min_balance_usd": self.min_balance, "run_origin_key_usage": self.origin,
                "start": self.start, "end": self.last, "breach": self.reason or None}


# ───────────────────────────────────────────────────────────────────────────── one shard
def _tagged(step: str, cmd: str) -> str:
    """A no-op marker line first (`:` builtin), so logs and test doubles can tell steps apart."""
    return f": EVOLEG={step}\n{cmd}"


class Shard:
    def __init__(self, leg: "LegRun", name: str, tasks: list):
        self.leg, self.cfg, self.name, self.tasks = leg, leg.cfg, name, tasks
        self.dir = leg.store / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.logf = open(self.dir / "driver.log", "a")
        self.summary: dict = {"shard": name, "tag": self.cfg.tag, "tasks": tasks,
                              "n_rollouts": len(tasks) * self.cfg.n_samples}
        self.sid = None
        self.sb = None
        self.log_offset = 0
        self.n429 = 0
        self.stats: dict = {}
        self.done = threading.Event()
        self.launched = threading.Event()

    def log(self, msg):
        log_line(f"[{self.cfg.code}/{self.name}] {msg}", self.logf)

    def sh(self, step, cmd, timeout=600, env=None, check=False):
        rc, out, _dt = dlx.run(self.sb, _tagged(step, cmd), timeout=timeout, env=env)
        if check and rc != 0:
            raise Abort(f"{step} failed rc={rc}: {out[-1500:]}")
        return rc, out

    def save(self):
        (self.dir / "summary.json").write_text(json.dumps(self.summary, indent=1, default=str))

    # -- steps
    def create(self):
        for attempt in range(10):
            try:
                return self._create_once()
            except Abort as e:
                msg = str(e).lower()
                transient = any(w in msg for w in ("quota", "limit", "exceed", "capacity",
                                                   "insufficient", "no available", "429",
                                                   "timeout", "timed out"))
                if (not transient or attempt == 9 or self.leg.abort_all.is_set()
                        or "per sandbox" in msg):
                    raise
                self.log(f"create refused (attempt {attempt + 1}/10), retry in 60s: {str(e)[:300]}")
                if self.sid:
                    dlx.delete_by_id(self.leg.client, self.sid, self.cfg.tag)
                    self.sid = None
                time.sleep(60)

    def allow_list(self) -> str:
        """The Daytona domain allow list this sandbox is created with: a shard's is the runner's ALLOW_DOMAINS
        (openrouter.ai). POOL MODE: ssb_leg_pool.Machine overrides it (api.anthropic.com added only when the run's
        evolver runs in a container on the pool machine, RUNTIME evolver_daytona.placement "pool_container")."""
        return ALLOW_DOMAINS

    def _create_once(self):
        from daytona import CreateSandboxFromImageParams, Image, Resources
        name = f"{dlx.PREFIX}{self.cfg.code}-{self.name}-{time.strftime('%H%M%S')}"
        ttl = int(self.cfg.rt["ttl_minutes"])
        allow = self.allow_list()
        params = CreateSandboxFromImageParams(
            name=name, image=Image.from_dockerfile(str(DOCKERFILE)), resources=Resources(**RESOURCES),
            ephemeral=True, auto_stop_interval=0, ttl_minutes=ttl, public=False,
            domain_allow_list=allow, labels={"shard": self.name, "leg": self.cfg.code})
        self.log(f"creating {name} ({RESOURCES}, ttl {ttl} min, egress {allow})")
        blog = open(self.dir / "image_build.log", "a")
        sb, sid, err, dt = dlx.create(self.leg.client, params, RESOURCES, tag=self.cfg.tag,
                                      timeout=3600,
                                      on_logs=lambda c: (blog.write(c + "\n"), blog.flush()))
        self.sid = sid
        self.summary.update(sandbox_id=sid, sandbox_name=name, create_s=dt)
        if err is not None:
            raise Abort(f"create failed after {dt}s: {type(err).__name__}: {str(err)[:600]}")
        self.sb = sb
        rb = dlx.readback(sb)
        self.summary["readback"] = rb
        mism = {k: (RESOURCES[k], rb.get(k)) for k in RESOURCES if rb.get(k) != RESOURCES[k]}
        self.log(f"created {sid} in {dt}s readback cpu={rb['cpu']} mem={rb['memory']} "
                 f"disk={rb['disk']} snapshot_ref={rb.get('snapshot_ref')}")
        if mism:
            raise Abort(f"resource readback mismatch {mism}")
        want = self.cfg.rt.get("require_snapshot_ref")
        if want and rb.get("snapshot_ref") != want:
            raise Abort(f"Daytona snapshot {rb.get('snapshot_ref')} != the baseline's {want} — a "
                        f"rebuilt image is a different comparability class (relax RUNTIME "
                        f"daytona_leg.require_snapshot_ref only deliberately)")

    def identity(self):
        rc, out = self.sh("identity", base.IDENTITY_CMD, timeout=300)
        ident = base.parse_kv(out)
        self.summary["identity"] = ident
        self.log("identity " + json.dumps(ident))
        if ident.get("arch") != "x86_64":
            raise Abort("identity probe failed: " + out[-800:])
        if not ident.get("runner_openpyxl", "").startswith(EXPECTED["grading_openpyxl_prefix"]):
            raise Abort(f"runner openpyxl is not {EXPECTED['grading_openpyxl_prefix']}: "
                        f"{ident.get('runner_openpyxl')}")
        if ident.get("claude_binaries") != "0":
            raise Abort(f"claude binary present in image: {ident.get('claude_binaries')}")
        want = self.cfg.rt.get("require_libreoffice")
        if want and ident.get("libreoffice") != want:
            raise Abort(f"LibreOffice {ident.get('libreoffice')!r} != the baseline's {want!r}")

    def egress(self):
        rc, out = self.sh("egress", base.EGRESS_CMD, timeout=180)
        m = re.search(r"^EGRESS (.*)$", out, re.M)
        if not m:
            raise Abort("egress probe failed: " + out[-800:])
        eg = json.loads(m.group(1))
        self.summary["egress"] = {"domain_allow_list": ALLOW_DOMAINS, "probe": eg}
        self.log("egress " + json.dumps(eg))
        if not eg.get("openrouter", {}).get("ok"):
            raise Abort("openrouter.ai NOT reachable from the sandbox")
        leaked = [k for k in eg if k != "openrouter" and eg[k].get("ok")]
        if leaked:
            raise Abort(f"egress allow list NOT enforced: reachable {leaked}")

    def upload(self, leg_env: str):
        b = self.dir / "bundle.tgz"
        seeds = seed_files(self.cfg.cache_dir, self.tasks)
        man = build_bundle(self.cfg, self.tasks, leg_env, b, seeds, idx=self.leg.idx)
        self.summary["bundle"] = man
        t = time.time()
        self.sb.fs.upload_file(str(b), "/tmp/bundle.tgz", timeout=1800)
        self.log(f"uploaded bundle {man['bundle_bytes'] / 1e6:.1f} MB ({man['files']} files, "
                 f"{man['seeded_cache_files']} seeded cache file(s), {len(man['skills_files'])} "
                 f"skill file(s)) in {time.time() - t:.0f}s")
        b.unlink()
        self.sh("unpack", "tar xzf /tmp/bundle.tgz -C / && rm /tmp/bundle.tgz && "
                "install -m 0755 /opt/leg/docker_shim.py /usr/local/bin/docker && "
                "mkdir -p /opt/leg/out /var/log/evoshim /var/lib/evoshim/state /var/lib/evoshim/priv && "
                "which docker && sha256sum /usr/local/bin/docker", timeout=600, check=True)
        if self.cfg.skills_dir is not None:
            rc, out = self.sh("skills", f"ls {shlex.quote(str(self.cfg.skills_dir))}/*/SKILL.md "
                              f"| wc -l", timeout=60)
            got = (out.strip().splitlines() or ["0"])[-1].strip()
            want = str(len(skill_mds(self.cfg.skills_dir)))        # same dot-dir rule as `ls`
            self.summary["skills_in_sandbox"] = got
            if got != want:
                raise Abort(f"skills upload incomplete: {got} SKILL.md in the sandbox, {want} local")

    def cgroups_and_selftest(self):
        cg = ("CG=/sys/fs/cgroup; echo '+memory +cpu +pids' > $CG/cgroup.subtree_control 2>&1 || "
              "{ mkdir -p $CG/sandbox_init; for p in $(cat $CG/cgroup.procs); do echo $p > "
              "$CG/sandbox_init/cgroup.procs 2>/dev/null; done; echo '+memory +cpu +pids' > "
              "$CG/cgroup.subtree_control; }; mkdir -p $CG/evoshim $CG/evoshim_runner; "
              "echo '+memory +cpu +pids' > $CG/evoshim/cgroup.subtree_control; "
              "echo root=$(cat $CG/cgroup.subtree_control) evoshim=$(cat $CG/evoshim/cgroup.subtree_control)")
        rc, out = self.sh("cgroups", cg, timeout=60)
        self.summary["cgroup_setup"] = out.strip()[-400:]
        if "memory" not in out.split("evoshim=")[-1]:
            raise Abort("memory controller not delegated to evoshim/ -- per-episode caps impossible")
        rc, out = self.sh("selftest", "bash -s " + shlex.quote(self.cfg.image_id)
                          + " <<'SELFTEST'\n" + base.SELFTEST_CMD + "\nSELFTEST", timeout=300)
        st = base.parse_kv(out)
        self.summary["shim_selftest"] = {"parsed": st, "raw_tail": out[-2500:]}
        need = {"DATASET": "HIDDEN", "UMOUNT": "DENIED", "NET": "NONE",
                "CGROUP_WRITE": "DENIED", "WS_WRITE": "OK", "RUNNING": "true",
                "PROBE": "/usr/bin/soffice", "IMAGES": self.cfg.image_id, "USERS_ENTRIES": "0",
                "OPT_ENTRIES": "0", "HOME_ENTRIES": "0", "SHIMSTATE_ENTRIES": "0",
                "CGROUPS_LEFT": "0", "EXEC_RC": "0", "KEY_IN_ENV": "0"}
        bad = {k: st.get(k) for k, v in need.items() if (st.get(k) or "").strip() != v}
        if bad:
            raise Abort(f"shim self-test failed: {bad}")
        rc, out = self.sh("grade", "bash -s " + shlex.quote(self.cfg.image_id) + " "
                          + shlex.quote(self.tasks[0]) + " <<'GRADE'\n" + base.GRADE_SELFTEST
                          + "\nGRADE", timeout=900)
        m = re.search(r"^GRADE_SELFTEST (.*)$", out, re.M)
        if not m:
            raise Abort("grading self-test crashed: " + out[-1500:])
        g = json.loads(m.group(1))
        self.summary["grade_selftest"] = g
        if re.search(r"recalculation failed|_solve_datatables_with_uno failed", out):
            raise Abort("grading self-test: LibreOffice recalculation FAILED through the shim")
        if not g.get("golden_vs_golden_passed"):
            self.log(f"NOTE: golden-vs-golden did not PASS for {g.get('task')} "
                     f"({g.get('matched')}/{g.get('total')})")

    def runtime_env_record(self, cfg=None, *, shard=None, tasks=None, seeded=None) -> dict:
        """The runtime_env the runner embeds in its result, for THIS sandbox running `cfg`'s leg (a pool unit
        passes its own leg cfg, unit name, task and seed count; a shard passes nothing)."""
        cfg = self.cfg if cfg is None else cfg
        ident = self.summary.get("identity", {})
        rb = self.summary.get("readback") or {}
        rt = {
            "platform": "daytona",
            "comparability_class": "x86_64 Daytona sandbox (NOT the Mac aarch64 docker legs)",
            "driver": "scripts/daytona/ssb_leg_daytona_loop.py",
            "sandbox_id": self.sid, "sandbox_name": self.summary.get("sandbox_name"),
            "readback": rb, "shard": self.name if shard is None else shard,
            "task_subset": self.tasks if tasks is None else tasks, "leg_tag": cfg.tag,
            "image": {"dockerfile": "scripts/daytona/Dockerfile.ssb_leg",
                      "dockerfile_sha256": cfg.dockerfile_sha256,
                      "daytona_snapshot_ref": rb.get("snapshot_ref"),
                      "image_id_reported_to_runner": cfg.image_id},
            "arch": ident.get("arch"), "cpu_model": ident.get("cpu_model"),
            "nproc_reported": ident.get("nproc"), "cgroup_cpu_max": ident.get("cgroup_cpu_max"),
            "cgroup_memory_max": ident.get("cgroup_memory_max"),
            "libreoffice": ident.get("libreoffice"), "grading_openpyxl": ident.get("runner_openpyxl"),
            "runner_python": ident.get("runner_python"), "solver_python": ident.get("solver_python"),
            "solver_stack": ident.get("solver_stack"),
            "runner_freeze_sha256": ident.get("runner_freeze_sha256"),
            "shim_sha256": sha256_file(HERE / "docker_shim.py"),
            "entry_sha256": sha256_file(HERE / "sandbox_entry.py"),
            "runner_sha256": sha256_file(REPO / RUNNER_REL),
            "executor_sha256": sha256_file(REPO / "src/harness/direct/executor.py"),
            "scorer_sha256": sha256_file(REPO / "src/agent_profiles/ssbv2_agent/scorer.py"),
            "solver_prompt_sha256": sha256_file(REPO / "src/agent_profiles/ssbv2_agent/prompt.md"),
            "split_file": str(cfg.split_file), "split_sha256": cfg.split_sha256,
            "split_status": cfg.split_status,
            "skills": skills_fingerprint(cfg.skills_dir),
            "cache_dir": str(cfg.cache_dir),
            "seeded_cache_files": ((self.summary.get("bundle") or {}).get("seeded_cache_files")
                                   if seeded is None else seeded),
            "knobs": cfg.knobs, "thread_env": base.THREAD_ENV,
            "infra": cfg.infra_record,
            "egress": self.summary.get("egress"), "grade_selftest": self.summary.get("grade_selftest"),
            "reasoning_param": "none sent (thinking_effort='', enable_thinking=True)",
            "written_utc": dlx.now(),
        }
        return rt

    def runtime_env(self):
        rt = self.runtime_env_record()
        (self.dir / "runtime_env.json").write_text(json.dumps(rt, indent=1, default=str))
        self.sb.fs.upload_file(json.dumps(rt, default=str).encode(), "/opt/leg/runtime_env.json")

    def launch(self, key: str):
        rc, out = self.sh("launch", "(setsid nohup bash /opt/leg/sandbox_launch.sh /opt/leg/leg.env "
                          f"{shlex.quote(str(REPO))} "
                          "> /opt/leg/out/launch.log 2>&1 < /dev/null &) ; sleep 8; "
                          "tail -5 /opt/leg/out/runner.log 2>/dev/null; ls /opt/leg/out",
                          timeout=120, env={"OPENROUTER_API_KEY": key})
        self.summary["launched_utc"] = dlx.now()
        self.launched.set()
        self.log("launched; first lines: " + out.strip()[-600:].replace("\n", " | "))

    def _sh_retry(self, step, cmd, timeout, tries=4, wait_s=15):
        for i in range(tries):
            try:
                return self.sh(step, cmd, timeout=timeout)
            except Exception as e:                              # noqa: BLE001
                if i == tries - 1:
                    raise
                self.log(f"{step} failed (attempt {i + 1}/{tries}), retry in {wait_s}s: "
                         f"{type(e).__name__}: {str(e)[:200]}")
                time.sleep(wait_s)

    def poll(self) -> bool:
        cache = shlex.quote(str(self.cfg.cache_dir))
        rc, out = self._sh_retry(
            "poll",
            f"cat /opt/leg/out/rc 2>/dev/null || echo RUNNING; "
            f"echo SIZE=$(stat -c %s /opt/leg/out/runner.log 2>/dev/null || echo 0); "
            f"echo DONE=$(find {cache} -name '*.json' 2>/dev/null | wc -l); "
            f"echo \"SAMPLE=$(tail -n1 /opt/leg/out/samples.jsonl 2>/dev/null)\"; "
            f"echo \"STATS=$(cat /var/log/evoshim/llm_stats.json 2>/dev/null)\"", timeout=120)
        lines = out.strip().splitlines()
        state = lines[0].strip() if lines else "?"
        kv = {}
        for ln in lines[1:]:
            if "=" in ln:
                k, v = ln.split("=", 1)
                kv[k] = v
        size = int(kv.get("SIZE", "0") or 0)
        if size > self.log_offset:
            _rc, chunk = self._sh_retry("logtail", f"tail -c +{self.log_offset + 1} /opt/leg/out/runner.log "
                                        f"| head -c 4000000", timeout=300)
            with open(self.dir / "runner.log", "a") as fh:
                fh.write(chunk)
            self.log_offset += len(chunk.encode())
            self.n429 += len(re.findall(r"API 429", chunk))
        try:
            self.stats = json.loads(kv.get("STATS") or "{}")
        except ValueError:
            pass
        self.summary["progress"] = {"cache_json_files": int(kv.get("DONE", "0") or 0),
                                    "of": self.summary["n_rollouts"], "n429_log": self.n429,
                                    "fresh_responses": self.stats.get("fresh_responses")}
        self.log(f"progress {kv.get('DONE')}/{self.summary['n_rollouts']} | fresh "
                 f"{self.stats.get('fresh_responses')} | 429s {self.n429} | state {state}")
        if state != "RUNNING":
            self.summary["runner_rc"] = state
            return True
        return False

    def kill_runner(self):
        self.sh("kill", "pkill -TERM -f '[s]andbox_entry.py'; sleep 10; pkill -KILL -f "
                "'[s]andbox_entry.py'; for c in /var/lib/evoshim/state/*.json; do [ -e \"$c\" ] && "
                "docker rm -f $(basename $c .json); done; echo killed", timeout=120)

    def sync_back(self):
        tag = self.cfg.tag
        items = [(f"{base.REPO_ARC}/outputs/ssbv2_base30", "result"),
                 (str(self.cfg.cache_dir).lstrip("/"), "cache"),
                 (f"{base.REPO_ARC}/.cache/ssbv2_{tag}_tasks", "tasks"),
                 ("var/log/evoshim", "evoshim"), ("opt/leg/out", "out"),
                 ("opt/leg/runtime_env.json", "runtime_env.sandbox.json"),
                 ("opt/leg/leg.env", "leg.env")]
        xf = " ".join(f"--transform {shlex.quote('s,^' + a + ',' + b + ',')}" for a, b in items)
        xf += " --exclude=.claude --exclude='*.lock'"
        srcs = " ".join(shlex.quote(a) for a, _ in items)
        spec = json.dumps({"out": "/tmp/sync.tgz", "items": items})
        rc, out = self.sh("pack", f"# {spec}\ncd / && for p in {srcs}; do [ -e \"$p\" ] || echo "
                          f"MISSING $p; done; tar czf /tmp/sync.tgz {xf} $(for p in {srcs}; do "
                          f"[ -e \"$p\" ] && echo $p; done) && ls -la /tmp/sync.tgz", timeout=900)
        self.log("sync pack: " + out.strip()[-400:].replace("\n", " | "))
        local = self.dir / "sync.tgz"
        self.sb.fs.download_file("/tmp/sync.tgz", str(local))
        skipped = 0
        with tarfile.open(local) as tf:
            for m in tf.getmembers():
                if not (m.isfile() or m.isdir()):
                    skipped += 1
                    continue
                tf.extract(m, self.dir, filter="data")
        local.unlink()
        self.summary.update(sync_skipped_members=skipped, synced_utc=dlx.now())

    def secret_scan(self, key: str):
        kb = key.encode()
        hits = []
        for p in self.dir.rglob("*"):
            if p.is_file():
                try:
                    if kb in p.read_bytes():
                        hits.append(str(p))
                except OSError:
                    pass
        self.summary["secret_hits"] = hits

    def cost_from_cache(self):
        tot, n = 0.0, 0
        c = self.dir / "cache"
        for p in (c.rglob("*.json") if c.exists() else []):
            try:
                tr = json.loads(p.read_text()).get("trace") or {}
                tot += float(tr.get("total_cost_usd") or 0)
                n += 1
            except (ValueError, OSError):
                pass
        self.summary["cache_records"] = n
        self.summary["cache_cost_usd"] = round(tot, 5)

    def verify_generations(self, key: str):
        r = self.dir / "result" / f"{self.cfg.tag}.json"
        ids = []
        if r.is_file():
            try:
                ids = json.loads(r.read_text()).get("generation_ids_sample") or []
            except ValueError:
                ids = []
        pick = list(dict.fromkeys([ids[0], ids[len(ids) // 2], ids[-1]])) if ids else []
        self.summary["mac_generation_lookups"] = [self.leg.gen_lookup(key, g) for g in pick]

    # -- orchestration
    def run(self, key: str):
        with self.leg.slots:
            return self._run(key)

    def _run(self, key: str):
        t0 = time.time()
        spend = self.leg.spend
        try:
            if self.leg.abort_all.is_set() or spend.breached.is_set():
                raise Abort("aborted before start")
            self.create()
            self.identity()
            self.egress()
            self.upload(leg_env_text(self.cfg))
            self.cgroups_and_selftest()
            self.runtime_env()
            spend.check(f"{self.name} pre-launch")
            if spend.breached.is_set():
                raise Abort("spend guard already breached")
            self.launch(key)
            deadline = t0 + int(self.cfg.rt["shard_wall_min"]) * 60
            while True:
                if self.leg.abort_all.is_set() or spend.breached.is_set():
                    self.log("ABORT requested -- killing the runner")
                    self.kill_runner()
                    self.summary["aborted"] = ("spend guard: " + spend.reason
                                               if spend.breached.is_set() else "abort_all")
                    break
                if time.time() > deadline:
                    self.log("shard wall reached -- killing the runner")
                    self.kill_runner()
                    self.summary["aborted"] = "shard wall"
                    break
                if self.poll():
                    break
                self.save()
                time.sleep(POLL_S)
            try:
                self.poll()
            except Exception:  # noqa: BLE001
                pass
            self.sync_back()
        except Abort as e:
            self.summary["error"] = str(e)
            self.log(f"ABORT: {e}")
            if self.sb is not None:
                try:
                    self.sync_back()
                except Exception as e2:  # noqa: BLE001
                    self.log(f"sync after abort failed: {e2}")
        except Exception as e:  # noqa: BLE001
            self.summary["error"] = f"{type(e).__name__}: {e}"
            self.log("CRASH: " + traceback.format_exc()[-2000:])
            if self.sb is not None:
                try:
                    self.sync_back()
                except Exception as e2:  # noqa: BLE001
                    self.log(f"sync after crash failed: {e2}")
        finally:
            if self.sid:
                ok = dlx.delete_by_id(self.leg.client, self.sid, self.cfg.tag)
                self.summary["deleted"] = ok
                self.log(f"DELETED {self.sid}: {ok}")
            self.summary["wall_s"] = round(time.time() - t0, 1)
            try:
                self.secret_scan(key)
                self.cost_from_cache()
                self.verify_generations(key)
            except Exception as e:  # noqa: BLE001
                self.log(f"post-processing failed: {e}")
            self.save()
            self.done.set()
            try:
                self.logf.close()
            except Exception:  # noqa: BLE001
                pass
        return self.summary


# ───────────────────────────────────────────────────────────────────────────── per-run liveness
LEGS_LOCK = Path(".machine") / "legs.lock"        # under the run dir
_HELD_LOCKS: list = []                             # held for the life of the process (a dropped file unlocks)


class RunLock:
    """A per-run flock file: SHARED by every live holder of one kind (several legs of a run run at once), tried
    EXCLUSIVE (non-blocking) by a command that must not overlap them (cleanup, status). It sees this run only, and
    the kernel drops a holder's share when its process exits, however it exits."""

    def __init__(self, path: Path):
        self.path, self.fh = Path(path), None

    def hold_shared(self) -> "RunLock":
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a")
        fcntl.flock(self.fh, fcntl.LOCK_SH)
        _HELD_LOCKS.append(self)
        return self

    def held_by_others(self) -> bool:
        """True while any other process holds the lock (a live holder); False when it is free (or never made)."""
        import fcntl
        if not self.path.is_file():
            return False
        with open(self.path, "a") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return True
            fcntl.flock(fh, fcntl.LOCK_UN)
        return False


def live_legs(run_dir: Path) -> bool:
    """A leg driver of THIS run is live (it holds <run>/.machine/legs.lock shared)."""
    return RunLock(Path(run_dir) / LEGS_LOCK).held_by_others()


# ───────────────────────────────────────────────────────────────────────────── merges
class _Lock:
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self):
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a")
        fcntl.flock(self.fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, *a):
        import fcntl
        fcntl.flock(self.fh, fcntl.LOCK_UN)
        self.fh.close()


def _marker_axes(p: Path) -> dict:
    d = json.loads(p.read_text())
    return {k: v for k, v in d.items() if not k.startswith("_")}


def merge_cache(shard_caches: list, local: Path) -> dict:
    """Copy every shard's cache back into the local per-epoch cache dir. New files copied,
    identical ones skipped; a DIFFERENT file at the same path is a conflict (the leg is refused).
    CACHE_ENV: the local marker wins, and every shard's axes must equal it. PROVIDER_VIOLATION:
    copied home (quarantine), and reported."""
    rep = {"copied": 0, "identical": 0, "conflicts": [], "marker": None,
           "provider_violation": False}
    local.mkdir(parents=True, exist_ok=True)
    with _Lock(local.parent / f".{local.name}.merge.lock"):
        for sc in shard_caches:
            sc = Path(sc)
            if not sc.is_dir():
                continue
            for p in sorted(sc.rglob("*")):
                if not p.is_file() or p.suffix == ".lock":
                    continue
                rel = p.relative_to(sc)
                dst = local / rel
                if rel == Path("CACHE_ENV"):
                    if dst.is_file():
                        if _marker_axes(dst) != _marker_axes(p):
                            rep["conflicts"].append(f"CACHE_ENV axes differ: shard {sc.parent.name}")
                    else:
                        shutil.copy2(p, dst)
                        rep["copied"] += 1
                    rep["marker"] = str(dst)
                    continue
                if rel == Path("PROVIDER_VIOLATION"):
                    rep["provider_violation"] = True
                    if not dst.exists():
                        shutil.copy2(p, dst)
                    continue
                if dst.is_file():
                    if dst.read_bytes() == p.read_bytes():
                        rep["identical"] += 1
                    else:
                        rep["conflicts"].append(str(rel))
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
                shutil.copy2(p, tmp)
                os.replace(tmp, dst)
                rep["copied"] += 1
    return rep


def merge_tasks(shard_tasks: list, local: Path) -> dict:
    """Every shard's tasks dir (per-sample workspaces + _snapshots/<task>/iter_*_s*/DIFF.txt) into
    <repo>/.cache/ssbv2_<tag>_tasks. A replayed rollout writes no snapshot, so files from an EARLIER
    attempt are kept; the same path written again is overwritten by the newer attempt (counted)."""
    rep = {"copied": 0, "overwritten": 0, "snapshot_samples": 0}
    for st in shard_tasks:
        st = Path(st)
        if not st.is_dir():
            continue
        for p in sorted(st.rglob("*")):
            if not p.is_file() or p.suffix == ".lock":
                continue
            dst = local / p.relative_to(st)
            if dst.is_file():
                rep["overwritten"] += 1
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
            rep["copied"] += 1
    snaps = local / "_snapshots"
    if snaps.is_dir():
        rep["snapshot_samples"] = sum(1 for _ in snaps.glob("*/iter_*_s*/DIFF.txt"))
    return rep


def _sum_counters(dicts) -> dict:
    c = Counter()
    for d in dicts:
        c.update(d or {})
    return dict(c)


def merge_call_deadline(rows: dict) -> tuple:
    """(merged `llm_call_deadline` block or None, problems): the runner records the executor's per-call deadline
    bounds and how often they fired (a cut attempt is resampled: an intervention on the measured solver). Every shard must have run under the same bounds;
    the counters add up. None = no shard recorded one (a runner older than the deadline), so such a leg merges
    exactly as before."""
    blocks = {s: r.get("llm_call_deadline") for s, r in rows.items()}
    have = {s: b for s, b in blocks.items() if isinstance(b, dict)}
    if not have:
        return None, []
    probs = []
    missing = sorted(s for s in blocks if s not in have)
    if missing:
        probs.append(f"llm_call_deadline recorded by some shards but not by {missing[:5]}")
    out = {}
    for k in ("attempt_s", "call_s"):
        vals = {json.dumps(b.get(k)) for b in have.values()}
        if len(vals) != 1:
            probs.append(f"llm_call_deadline.{k} differs across shards: {sorted(vals)[:3]}")
        out[k] = next(iter(have.values())).get(k)
    for k in sorted({k for b in have.values() for k in b} - {"attempt_s", "call_s"}):
        vals = [b.get(k) for b in have.values()]
        if all(v is None or (isinstance(v, (int, float)) and not isinstance(v, bool)) for v in vals):
            out[k] = sum(v or 0 for v in vals)
        else:
            probs.append(f"llm_call_deadline.{k} is not a counter: {vals[:3]}")
    return out, probs


def merge_results(cfg: LegConfig, plan: dict, rows: dict) -> tuple:
    """(merged runner-schema result, problems). `rows` = {shard: result dict or None}."""
    problems = []
    for shard, tasks in plan["shards"].items():
        r = rows.get(shard)
        if r is None:
            problems.append(f"{shard}: no canonical result")
            continue
        if r.get("INVALID") or r.get("provider_violations"):
            problems.append(f"{shard}: provider violations {r.get('provider_violations')}")
        if r.get("tag") != cfg.tag:
            problems.append(f"{shard}: result tag {r.get('tag')!r} != {cfg.tag!r}")
        if list(r.get("task_ids") or []) != list(tasks):
            problems.append(f"{shard}: task_ids differ from the plan")
        want_sd = str(cfg.skills_dir) if cfg.skills_dir is not None else None
        if r.get("skills_dir") != want_sd:
            problems.append(f"{shard}: skills_dir {r.get('skills_dir')!r} != {want_sd!r}")
        gc = r.get("generation_check") or {}
        if gc.get("providers") and set(gc["providers"]) != {cfg.leg_settings[
                "EVOSKILL_SOLVER_EXPECT_PROVIDER"]}:
            problems.append(f"{shard}: /generation providers {gc['providers']}")
        if gc.get("models") and set(gc["models"]) != {cfg.leg_settings[
                "EVOSKILL_SOLVER_EXPECT_MODEL"]}:
            problems.append(f"{shard}: /generation models {gc['models']}")
    if problems:
        return None, problems
    first = rows[next(iter(plan["shards"]))]
    for k in SAME_RESULT:
        vals = {json.dumps(r.get(k), sort_keys=True) for r in rows.values()}
        if len(vals) != 1:
            problems.append(f"result field {k} differs across shards: {sorted(vals)[:3]}")
    for k in SAME_RUNTIME:
        vals = {json.dumps((r.get("runtime_env") or {}).get(k), sort_keys=True)
                for r in rows.values()}
        if len(vals) != 1:
            problems.append(f"runtime_env.{k} differs across shards: {sorted(vals)[:3]}")
    base_rt = dict({k: want for k, (_rel, want) in EXPECTED["baseline_runtime"].items()},
                   solver_prompt_sha256=EXPECTED["baseline_solver_prompt_sha256"])
    for shard, r in rows.items():
        rte = r.get("runtime_env") or {}
        for k, want in base_rt.items():
            if rte.get(k) is not None and rte.get(k) != want:
                problems.append(f"{shard}: runtime_env.{k} {str(rte.get(k))[:12]} != baseline {want[:12]}")
        # The split the shard RECORDS: this leg's, i.e. the run's declared split (_check_spec).
        if rte.get("split_sha256") is not None and rte["split_sha256"] != cfg.split_sha256:
            problems.append(f"{shard}: runtime_env.split_sha256 {str(rte['split_sha256'])[:12]} != this "
                            f"leg's split {cfg.split_sha256[:12]} ({cfg.split_file.name})")
        # The knobs the shard RECORDS it ran with: this leg's (incl. INFRA's grader timeouts), and
        # never anything beyond the two timeouts the baseline comparison accepts (knob_drift).
        if "knobs" in rte:
            if rte["knobs"] != cfg.knobs:
                problems.append(f"{shard}: runtime_env.knobs {rte['knobs']} != this leg's {cfg.knobs}")
            drift = knob_drift(rte["knobs"])
            if drift:
                problems.append(f"{shard}: knobs differ from the held-out baseline's beyond "
                                f"{list(ALLOWED_KNOB_DELTA)}: {drift}")
    if first.get("n_samples") != cfg.n_samples:
        problems.append(f"n_samples {first.get('n_samples')} != leg {cfg.n_samples}")
    deadline, deadline_problems = merge_call_deadline(rows)
    problems.extend(deadline_problems)
    per_task_by_id = {}
    for shard, r in rows.items():
        for i, tid in enumerate(r["task_ids"]):
            sc = (r.get("per_task") or {}).get(str(i))
            if (tid in per_task_by_id or not isinstance(sc, list) or len(sc) != cfg.n_samples
                    or any(x is None for x in sc)):
                problems.append(f"{shard}: task {tid} duplicated or incomplete ({sc})")
                continue
            per_task_by_id[tid] = sc
    if sorted(per_task_by_id) != sorted(cfg.tasks):
        problems.append(f"merged task set != split[{cfg.key}] "
                        f"({len(per_task_by_id)} vs {len(cfg.tasks)})")
    if problems:
        return None, problems
    order = list(cfg.tasks)                     # the frozen split order, as a Mac leg indexes it
    per_task = {str(i): per_task_by_id[t] for i, t in enumerate(order)}
    flat = [x for v in per_task.values() for x in v]
    npass = sum(1 for x in flat if x >= 0.999)
    gcs = [r.get("generation_check") or {} for r in rows.values()]
    ids_first = [g for r in rows.values() for g in (r.get("generation_ids_sample") or [])]
    merged = {k: first.get(k) for k in SAME_RESULT}
    merged.update({
        "served_providers": _sum_counters(r.get("served_providers") for r in rows.values()),
        "served_models": _sum_counters(r.get("served_models") for r in rows.values()),
        "generation_check": {
            "n_fresh_ids": sum(int(g.get("n_fresh_ids") or 0) for g in gcs),
            "n_sampled": sum(int(g.get("n_sampled") or 0) for g in gcs),
            "n_ok": sum(int(g.get("n_ok") or 0) for g in gcs),
            "models": _sum_counters(g.get("models") for g in gcs),
            "providers": _sum_counters(g.get("providers") for g in gcs),
            "mismatches": [m for g in gcs for m in (g.get("mismatches") or [])],
            "failed_ids": [m for g in gcs for m in (g.get("failed_ids") or [])],
            "n_failed": sum(int(g.get("n_failed") or 0) for g in gcs)},
        "fresh_responses": sum(int(r.get("fresh_responses") or 0) for r in rows.values()),
        "fresh_responses_without_id": sum(int(r.get("fresh_responses_without_id") or 0)
                                          for r in rows.values()),
        "generation_ids_sample": ids_first[:25] + ids_first[25:][-25:],
        "provider_violations": [],
        **({"llm_call_deadline": deadline} if deadline is not None else {}),
        "origin": _sum_counters(r.get("origin") for r in rows.values()),
        "sample_pass": npass, "sample_total": len(flat),
        "all3": sum(1 for v in per_task.values() if all(x >= 0.999 for x in v)),
        "pass3": sum(1 for v in per_task.values() if any(x >= 0.999 for x in v)),
        "n": len(order),
        "turn_exhausted": sum(int(r.get("turn_exhausted") or 0) for r in rows.values()),
        "tag": cfg.tag, "per_task": per_task, "task_ids": order,
    })
    rt0 = first.get("runtime_env") or {}
    merged["runtime_env"] = {
        **{k: rt0.get(k) for k in SAME_RUNTIME},
        "platform": "daytona", "comparability_class": rt0.get("comparability_class"),
        "sandbox_ids": {s: (r.get("runtime_env") or {}).get("sandbox_id") for s, r in rows.items()},
        "daytona_snapshot_refs": sorted({json.dumps(((r.get("runtime_env") or {}).get("image") or {})
                                                    .get("daytona_snapshot_ref"))
                                         for r in rows.values()}),
    }
    return merged, []


# ───────────────────────────────────────────────────────────────────────────── the leg
class LegRun:
    def __init__(self, cfg: LegConfig, *, client=None, key=None, fetch_usage=None, gen_lookup=None,
                 idx=None):
        self.cfg = cfg
        self.store = cfg.store
        self.client, self.key = client, key
        self.fetch_usage = fetch_usage
        self.gen_lookup = gen_lookup or base.generation_lookup
        self.idx = idx
        self.abort_all = threading.Event()
        self.slots = None
        self.spend = None
        self.logf = None
        # POOL MODE only (ssb_leg_pool.py): merged into the daytona block / the result (and FAILED record) by
        # _finish. Empty for a shard leg, so its records are unchanged.
        self.extra_daytona: dict = {}
        self.extra_result: dict = {}
        # POOL MODE only (the LONG-LIVED pool): a pool leg's result is written while its machine may still
        # serve other legs, so "every sandbox deleted before the result" cannot hold. unit_end_check(unit) -> problem
        # or None REPLACES the per-shard "NOT verified deleted" check (the pool's: the unit ended, its outputs are
        # synced home and verified); listing_ids = the sandbox ids that must no longer be listed (the machines this
        # leg used that are already deleted) instead of every shard's. None (a shard leg) = the original checks.
        self.unit_end_check = None
        self.listing_ids = None

    def log(self, msg):
        log_line(f"[{self.cfg.code}] {msg}", self.logf)

    def _prepare_store(self):
        root = self.store.parent
        root.mkdir(parents=True, exist_ok=True)
        if self.store.exists() and any(self.store.iterdir()):
            prev = root / f"{self.cfg.tag}._prev_{time.strftime('%Y%m%dT%H%M%S')}"
            self.store.rename(prev)
        self.store.mkdir(parents=True, exist_ok=True)
        if self.cfg.failed_path.exists():
            self.cfg.failed_path.rename(self.cfg.failed_path.with_name(
                f"{self.cfg.tag}.FAILED.{time.strftime('%Y%m%dT%H%M%S')}.json"))

    def run(self) -> int:
        cfg = self.cfg
        if cfg.result_path.exists():
            raise Refused(f"{cfg.result_path} already exists — that leg is measured")
        self._prepare_store()
        self.logf = open(self.store / "leg.log", "a")
        self.idx = self.idx or task_index()
        plan = make_plan(cfg.tasks, self.idx, n_samples=cfg.n_samples, concurrency=cfg.concurrency,
                         max_sandboxes=int(cfg.rt["max_sandboxes"]),
                         max_heavy=int(cfg.rt["max_heavy_per_shard"]))
        plan.update({"tag": cfg.tag, "key": cfg.key, "split_file": str(cfg.split_file),
                     "split_sha256": cfg.split_sha256, "created": dlx.now()})
        (self.store / "plan.json").write_text(json.dumps(plan, indent=1))
        self.log(f"leg {cfg.tag}: {len(cfg.tasks)} tasks x {cfg.n_samples} over "
                 f"{len(plan['shards'])} sandbox(es) at conc {cfg.concurrency}; skills "
                 f"{cfg.skills_dir or 'NONE (base leg)'}; cache {cfg.cache_dir}")
        if self.client is None:
            self.client = dlx.client()
        if self.key is None:
            self.key = dlx.openrouter_key()
        fetch = self.fetch_usage or (lambda: openrouter_usage(self.key))
        self.spend = SpendGuard(fetch, leg_cap=cfg.rt["leg_spend_cap_usd"],
                                run_cap=cfg.rt["run_spend_cap_usd"],
                                min_balance=cfg.rt["min_account_balance_usd"],
                                origin_file=Path(cfg.rt["store_root"]) / cfg.run_dir.name
                                / "spend_origin.json",
                                origin_usage=cfg.rt.get("run_key_usage_start"), log=self.log)
        shards = [Shard(self, n, t) for n, t in plan["shards"].items()]
        self.slots = threading.BoundedSemaphore(len(shards))
        pre_abort = None
        try:
            if self.spend.breached.is_set():
                raise Abort("spend guard breached before any sandbox: " + self.spend.reason)
            self._supervise(shards)
        except Abort as e:
            self.log(f"ABORT: {e}")
            pre_abort = str(e)
            self.abort_all.set()
        except BaseException as e:  # noqa: BLE001 — a supervisor bug must still end in a record
            pre_abort = f"supervisor crashed: {type(e).__name__}: {e}"
            self.log(pre_abort + "\n" + traceback.format_exc()[-1500:])
            self.abort_all.set()
            for s in shards:                        # shard threads delete their own sandboxes
                s.done.wait(timeout=1800)
        return self._finish(plan, shards, pre_abort)

    def _supervise(self, shards):
        cfg = self.cfg
        threads = []

        def start(s):
            th = threading.Thread(target=s.run, args=(self.key,), daemon=False, name=s.name)
            th.start()
            threads.append(th)

        def on_sig(*_a):
            self.log("SIGINT/SIGTERM -- aborting all shards (sandboxes are deleted in finally)")
            self.abort_all.set()
        try:
            signal.signal(signal.SIGINT, on_sig)
            signal.signal(signal.SIGTERM, on_sig)
        except ValueError:
            pass                                   # not the main thread (tests)
        first = shards[: max(1, int(cfg.rt["ramp_first"]))]
        rest = shards[len(first):]
        for s in first:
            start(s)
            time.sleep(float(os.environ.get("EVOSKILL_DAYTONA_START_STAGGER_S", "3")))
        t_ramp, last_spend = None, time.time()
        tick = float(os.environ.get("EVOSKILL_DAYTONA_SUPERVISE_S", "30"))
        while any(th.is_alive() for th in threads) or (rest and not self.abort_all.is_set()
                                                       and not self.spend.breached.is_set()):
            time.sleep(tick)
            if time.time() - last_spend >= float(cfg.rt["spend_every_s"]):
                self.spend.check("watchdog")
                last_spend = time.time()
            if self.spend.breached.is_set():
                self.abort_all.set()
            if not rest:
                continue
            if self.abort_all.is_set():
                self.log(f"not starting {len(rest)} held shard(s): abort")
                rest = []
                continue
            if not any(s.launched.is_set() for s in first):
                if all(s.done.is_set() for s in first):
                    self.log("first wave ended before launching; not starting the rest")
                    rest = []
                continue
            t_ramp = t_ramp or time.time()
            if time.time() - t_ramp < float(cfg.rt["ramp_wait_s"]) and not all(
                    s.done.is_set() for s in first):
                continue
            calls = sum(int(s.stats.get("fresh_responses") or 0) for s in first)
            n429 = sum(int((s.stats.get("status_warnings") or {}).get("429", 0)) for s in first)
            rate = n429 / max(1, calls + n429)
            if calls > 0 and rate <= float(cfg.rt["max_429_rate"]):
                self.log(f"ramp OK (429 rate {rate:.1%}) -- starting {len(rest)} shard(s)")
                for s in rest:
                    start(s)
                rest = []
            elif all(s.done.is_set() for s in first):
                self.log(f"first wave over without a passing ramp check ({rate:.1%}); "
                         f"NOT starting {len(rest)} shard(s)")
                rest = []
            else:
                t_ramp = time.time()
        self.spend.check("final")

    def _finish(self, plan, shards, pre_abort) -> int:
        cfg = self.cfg
        rows, notes = {}, []
        for s in shards:
            r = s.dir / "result" / f"{cfg.tag}.json"
            if r.is_file():
                try:
                    rows[s.name] = json.loads(r.read_text())
                except ValueError as e:
                    notes.append(f"{s.name}: unreadable result ({e})")
            alt = sorted(p.name for p in (s.dir / "result").glob(f"{cfg.tag}.*.json")) \
                if (s.dir / "result").is_dir() else []
            if alt:
                notes.append(f"{s.name}: {alt}")
        cache_rep = merge_cache([s.dir / "cache" for s in shards], cfg.cache_dir)
        tasks_rep = merge_tasks([s.dir / "tasks" for s in shards], cfg.tasks_dir_local)
        merged, problems = merge_results(cfg, plan, rows)
        problems = list(problems)
        if pre_abort:
            problems.insert(0, f"aborted: {pre_abort}")
        for s in shards:
            if s.summary.get("error"):
                problems.append(f"{s.name}: {s.summary['error'][:300]}")
            if s.summary.get("aborted"):
                problems.append(f"{s.name}: aborted ({s.summary['aborted']})")
            if s.summary.get("secret_hits"):
                problems.append(f"{s.name}: KEY FOUND in synced files {s.summary['secret_hits']}")
            if self.unit_end_check is not None:             # pool leg: see __init__
                p_end = self.unit_end_check(s)
                if p_end:
                    problems.append(p_end)
            elif s.sid and s.summary.get("deleted") is not True:
                problems.append(f"{s.name}: sandbox {s.sid} NOT verified deleted")
        if cache_rep["conflicts"]:
            problems.append(f"cache merge conflicts: {cache_rep['conflicts'][:5]}")
        if cache_rep["provider_violation"]:
            problems.append(f"PROVIDER_VIOLATION marker merged into {cfg.cache_dir} (quarantined)")
        hours = sum(float(s.summary.get("wall_s") or 0) for s in shards) / 3600.0
        daytona = {
            "driver": "scripts/daytona/ssb_leg_daytona_loop.py",
            "driver_sha256": sha256_file(Path(__file__)),
            "image_id": cfg.image_id, "dockerfile_sha256": cfg.dockerfile_sha256,
            "snapshot_ref_required": cfg.rt.get("require_snapshot_ref"),
            "snapshot_refs": sorted({str((s.summary.get("readback") or {}).get("snapshot_ref"))
                                     for s in shards}),
            "libreoffice": sorted({str((s.summary.get("identity") or {}).get("libreoffice"))
                                   for s in shards}),
            "grading_openpyxl": sorted({str((s.summary.get("identity") or {})
                                            .get("runner_openpyxl")) for s in shards}),
            "solver_stack": sorted({str((s.summary.get("identity") or {}).get("solver_stack"))
                                    for s in shards}),
            "shards": {s.name: dict({k: s.summary.get(k) for k in (
                "sandbox_id", "sandbox_name", "tasks", "wall_s", "deleted", "runner_rc", "error",
                "aborted", "cache_records", "cache_cost_usd", "mac_generation_lookups")},
                seeded_cache_files=(s.summary.get("bundle") or {}).get("seeded_cache_files"))
                for s in shards},
            "plan": {k: plan[k] for k in ("n_tasks", "n_samples", "concurrency",
                                          "episodes_per_shard_max", "heavy")},
            "store": str(self.store), "cache_dir": str(cfg.cache_dir),
            "tasks_dir": str(cfg.tasks_dir_local),
            "cache_merge": cache_rep, "tasks_merge": tasks_rep,
            "skills": skills_fingerprint(cfg.skills_dir),
            "solver_pins_from_caller_env": cfg.env_pins,
            "knobs": cfg.knobs, "infra": cfg.infra_record, "prompt_mode": cfg.prompt_mode,
            "data_overlay": data_overlay.record(cfg.data_overlay),
            "data_overlay_from": "INFRA.json" if cfg.infra is not None else "env",
            "spend": self.spend.record() if self.spend else None,
            "sandbox_hours": round(hours, 3),
            "daytona_list_cost_usd": round(hours * DAYTONA_USD_PER_SANDBOX_HOUR, 3),
            "notes": notes,
        }
        daytona.update(self.extra_daytona)
        try:
            ours = dlx.list_ours(self.client)
            gone = self.listing_ids if self.listing_ids is not None else {s.sid for s in shards}
            daytona["loop_sandboxes_still_listed"] = [o for o in ours if o["id"] in gone]
            if daytona["loop_sandboxes_still_listed"]:
                problems.append(f"sandboxes still listed: {daytona['loop_sandboxes_still_listed']}")
        except Exception as e:  # noqa: BLE001
            daytona["final_listing_error"] = f"{type(e).__name__}: {e}"
        if problems or merged is None:
            rec = {"INCOMPLETE": True, "tag": cfg.tag, "key": cfg.key, "problems": problems,
                   "daytona": daytona, "data_overlay": data_overlay.record(cfg.data_overlay),
                   "written_utc": dlx.now(), **self.extra_result}
            _atomic_write(cfg.failed_path, rec)
            self.log("LEG FAILED — wrote " + cfg.failed_path.name + ":\n  " + "\n  ".join(problems))
            return EXIT_LEG_FAILED
        merged["daytona"] = daytona
        merged.update(self.extra_result)
        (self.store / f"{cfg.tag}.merged.json").write_text(json.dumps(merged, indent=1))
        _atomic_write(cfg.result_path, merged)
        self.log(f"LEG OK {merged['sample_pass']}/{merged['sample_total']} "
                 f"(pass@{cfg.n_samples} {merged['pass3']}/{merged['n']}) -> {cfg.result_path}")
        return EXIT_OK


def _atomic_write(path: Path, obj: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


# ───────────────────────────────────────────────────────────────────────────── commands
def _cfg_from_args(a) -> LegConfig:
    if a.runner != RUNNER_NAME:
        raise Refused(f"--runner must be {RUNNER_NAME} (the loop's liveness probe matches it), "
                      f"got {a.runner!r}")
    cfg = LegConfig(tag=a.tag, key=a.key, run_dir=a.run_dir, split_file=a.split,
                    cache_dir=a.cache_dir, skills_dir=a.skills_dir or None,
                    n_samples=a.n_samples, concurrency=a.concurrency, allow_test=a.allow_test)
    if getattr(a, "max_sandboxes", None):
        # e.g. the held-out leg: 12 sandboxes x ~10 tasks = the held-out baseline's own layout
        cfg.rt["max_sandboxes"] = int(a.max_sandboxes)
    return cfg


def cmd_leg(a) -> int:
    try:
        cfg = _cfg_from_args(a)
        if cfg.result_path.exists():
            raise Refused(f"{cfg.result_path} already exists — that leg is measured")
        pooled = pool_enabled(cfg)                  # validates the pool keys whether on or off
        if a.check_only:
            gt = {k: cfg.knobs.get(k, "scorer default") for k in ALLOWED_KNOB_DELTA}
            ps = pool_settings(cfg.rt) if pooled else None
            pool_note = ""
            if pooled:
                import ssb_leg_pool
                gc = ssb_leg_pool.grader_cap(cfg, ps)
                bad = ssb_leg_pool.layout_problem(ps, gc)
                if bad:
                    raise Refused(bad)
                ntsib = ssb_leg_pool.no_train_sibling(cfg)          # validates the val-only flag
                placement = ssb_leg_pool.evolver_placement(cfg.run_dir)   # validates evolver_daytona.placement
                pool_note = (f"; POOL on: {ps['pool_sandboxes']} machines shared with every leg of the run, "
                             f"{ssb_leg_pool.cap_note(ps)}, "
                             f"<= {gc['units_per_machine_cap']} units per machine (grader cap: "
                             f"{gc['machine_recalc_budget']} recalcs // {gc['recalc_per_process']} per runner) = <= "
                             f"{gc['attempts_per_machine_max_at_n']} attempts per machine; order {ps['pool_order']}; "
                             f"machine TTL {pool_ttl(cfg.rt, ps)[0]} min ({pool_ttl(cfg.rt, ps)[1]}); one pool per "
                             f"run, keep idle {ps['pool_keep_idle_s']}s; machine egress "
                             f"{','.join(ssb_leg_pool.allow_domains(placement))} (evolver_daytona.placement "
                             f"{placement})"
                             + ("; VAL-ONLY epoch (no train leg of this epoch will join)" if ntsib else ""))
            print(f"CHECK OK {cfg.tag}: {len(cfg.tasks)} {cfg.key} tasks, skills "
                  f"{cfg.skills_dir or 'none (base leg)'}, cache {cfg.cache_dir}; grader timeouts "
                  f"{gt} from {(cfg.infra_record or {}).get('file') or 'no INFRA.json'}; prompt_mode "
                  f"{cfg.prompt_mode}; {overlay_note(cfg)}{pool_note}")
            return EXIT_OK
        if a.dry_run:
            idx = task_index()
            if pooled:
                import ssb_leg_pool
                dp = ssb_leg_pool.dry_run_pool(cfg, idx)
                plan, seeds = dp["plan"], dp["seed_files"]
            else:
                plan = make_plan(cfg.tasks, idx, n_samples=cfg.n_samples, concurrency=cfg.concurrency,
                                 max_sandboxes=int(cfg.rt["max_sandboxes"]),
                                 max_heavy=int(cfg.rt["max_heavy_per_shard"]))
                seeds = {s: [str(p) for p in seed_files(cfg.cache_dir, t)]
                         for s, t in plan["shards"].items()}
            rec = {"DRY_RUN": True, "tag": cfg.tag, "key": cfg.key, "run_dir": str(cfg.run_dir),
                   "split_file": str(cfg.split_file), "split_sha256": cfg.split_sha256,
                   "cache_dir": str(cfg.cache_dir), "tasks_dir": str(cfg.tasks_dir_local),
                   "skills": skills_fingerprint(cfg.skills_dir), "plan": plan,
                   "seed_files": seeds, "leg_env": leg_env_text(cfg).splitlines(),
                   "solver_pins_from_caller_env": cfg.env_pins,
                   "knobs": cfg.knobs, "infra": cfg.infra_record, "prompt_mode": cfg.prompt_mode,
                   "data_overlay": data_overlay.record(cfg.data_overlay),
                   "data_overlay_from": "INFRA.json" if cfg.infra is not None else "env",
                   "image_id": cfg.image_id, "sandbox_name_example":
                   f"{dlx.PREFIX}{cfg.code}-s01-HHMMSS", "daytona_leg": cfg.rt,
                   "store": str(cfg.store), "runner": RUNNER_NAME,
                   "note": "nothing was created, uploaded, spent or launched"}
            if pooled:
                rec["pool"] = dp["pool"]
            out = cfg.run_dir / f"{cfg.tag}.DRYRUN.json"
            out.write_text(json.dumps(rec, indent=1, default=str))
            if pooled:
                print(f"DRY RUN {cfg.tag}: POOL mode, {len(plan['shards'])} unit(s) (one per task) on up to "
                      f"{dp['pool']['machines_max']} shared machine(s); record {out}")
            else:
                print(f"DRY RUN {cfg.tag}: {len(plan['shards'])} shard(s) "
                      f"{ {s: len(t) for s, t in plan['shards'].items()} }; record {out}")
            return EXIT_DRY_RUN
        RunLock(cfg.run_dir / LEGS_LOCK).hold_shared()      # until this process exits: cleanup sees a live leg
        if pooled:
            import ssb_leg_pool
            ssb_leg_pool.no_train_sibling(cfg)                  # a bad val-only flag refuses before any join
            return ssb_leg_pool.follow(cfg, a)
        leg = LegRun(cfg)
        return leg.run()
    except Refused as e:
        print(f"REFUSING: {e}", file=sys.stderr, flush=True)
        return EXIT_REFUSED


def base_leg_env(run_dir: Path, key: str) -> tuple:
    """(tag, env, argv) for an e0 base leg of this run, from its SPEC.json + RUNTIME.json."""
    sp = json.loads((run_dir / "SPEC.json").read_text())["spec"]
    rt = json.loads((run_dir / "RUNTIME.json").read_text())
    tag = f"{run_dir.name}_base_{key}"
    declared = rt.get("base_leg" if key == "train" else "base_leg_val")
    if not declared or Path(declared).resolve() != (run_dir / f"{tag}.json").resolve():
        raise Refused(f"RUNTIME {'base_leg' if key == 'train' else 'base_leg_val'}={declared!r} "
                      f"must be {run_dir / (tag + '.json')}")
    caches = rt.get("digest_caches") or []
    if len(caches) != 1:
        raise Refused("RUNTIME digest_caches must name exactly one e0 cache dir")
    env = {k: v for k, v in os.environ.items() if k not in ("EVOSKILL_SKILLS_DIR",)}
    for k in SOLVER_PIN_KEYS:
        env.pop(k, None)
    try:
        rel = str(run_dir.resolve().relative_to(REPO))
    except ValueError:
        rel = str(run_dir.resolve())
    env.update({"EVOSKILL_TAU3_RUN_DIR": rel, "EVOSKILL_TAU3_SPLIT": str(sp["split_path"]),
                "EVOSKILL_N_SAMPLES": str(sp["n_samples"]),
                "EVOSKILL_SOLVER_MODEL": sp["solver_model"],
                "EVOSKILL_CACHE_DIR": str(caches[0]), "EVO_CONC": "30"})
    env.update({str(k): str(v) for k, v in (rt.get("solver_env") or {}).items()})
    leg_script = rt.get("leg_script") or str(REPO / "scripts/ssbv2_leg_daytona.sh")
    return tag, env, ["bash", leg_script, tag, key, ""]


def cmd_launch_base(a) -> int:
    rd = Path(a.run_dir).resolve()
    rc_all = 0
    for key in ("train", "val"):
        try:
            tag, env, argv = base_leg_env(rd, key)
        except Refused as e:
            print(f"REFUSING: {e}", file=sys.stderr)
            return EXIT_REFUSED
        if (rd / f"{tag}.json").exists():
            print(f"{tag}: already measured ({rd / (tag + '.json')}) — skipped")
            continue
        if a.dry_run:
            env["EVOSKILL_LEG_DRY_RUN"] = "1"
        r = subprocess.run(argv, env=env, cwd=str(REPO), capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        if r.returncode not in (0, EXIT_DRY_RUN):
            rc_all = r.returncode
    return rc_all


def _alive(pid: int, needle: str) -> bool:
    q = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True)
    return needle in q.stdout


def cmd_status(a) -> int:
    rd = Path(a.run_dir).resolve()
    rt = load_runtime_block(rd)
    root = Path(rt["store_root"]) / rd.name
    tags = sorted({p.name for p in root.iterdir() if p.is_dir() and "._prev_" not in p.name
                   and p.name != "pool"}) if root.is_dir() else []
    print(f"live leg drivers of this run: {'yes' if live_legs(rd) else 'none'}")
    hosts = _pool_hosts(rd)
    if hosts is None:
        hosts = ["unknown (registry unreadable)"]
    if hosts or (root / "pool").is_dir():
        print(f"live pool hosts: {hosts or 'none'} (details: scripts/daytona/ssb_leg_pool.py status)")
    for t in tags:
        res = rd / f"{t}.json"
        fail = rd / f"{t}.FAILED.json"
        st = "RESULT" if res.exists() else ("FAILED" if fail.exists() else "pending/running")
        prog = []
        for s in sorted((root / t).glob("s*/summary.json")):
            try:
                d = json.loads(s.read_text())
            except ValueError:
                continue
            prog.append(f"{d.get('shard')}:{(d.get('progress') or {}).get('cache_json_files')}"
                        f"/{d.get('n_rollouts')} rc={d.get('runner_rc')} del={d.get('deleted')}")
        print(f"{t}: {st}  {' '.join(prog)}")
    return 0


def _pool_hosts(run_dir: Path) -> list | None:
    """The pid of THIS run's live POOL host (its registry's host, alive and still that program): it owns and deletes
    the pool's sandboxes. [] when the run has none; None when that cannot be told (RUNTIME or registry unreadable)."""
    import ssb_leg_pool
    try:
        rt = load_runtime_block(Path(run_dir))
        f = Path(rt["store_root"]) / Path(run_dir).name / "pool" / ssb_leg_pool.POOL_SCOPE / "registry.json"
    except (OSError, ValueError, KeyError, Refused):
        return None
    if not f.is_file():
        return []
    try:
        reg = json.loads(f.read_text())
    except (OSError, ValueError):
        return None
    return [str(reg.get("host_pid"))] if isinstance(reg, dict) and ssb_leg_pool.host_alive(reg) else []


def _run_tag(tag: str, run_name: str) -> bool:
    """A sandbox's ledger/label tag belongs to the run: the leg tags <run>_..., the pool's pool_<run>."""
    tag = str(tag or "")
    return tag == run_name or tag.startswith(run_name + "_") or tag == f"pool_{run_name}"


def cmd_cleanup(a) -> int:
    """Delete any sandbox of OURS (loop prefix) of THIS run that the ledger shows created and not deleted."""
    if not a.run_dir:
        print("cleanup needs --run-dir: it deletes that run's loop sandboxes only", file=sys.stderr)
        return EXIT_REFUSED
    rd = Path(a.run_dir).resolve()
    if live_legs(rd) and not a.force:
        print(f"a leg driver of {rd.name} is still running; it deletes its own sandboxes. "
              f"Use --force only if it is wedged.", file=sys.stderr)
        return EXIT_REFUSED
    hosts = _pool_hosts(rd)
    if (hosts is None or hosts) and not a.force:
        print(f"the pool host of {rd.name} is still running ({hosts}), or its registry cannot be read (None); it "
              f"deletes its own sandboxes. Use --force only if it is wedged.", file=sys.stderr)
        return EXIT_REFUSED
    created, deleted = {}, set()
    lines = Path(dlx.LEDGER).read_text().splitlines() if Path(dlx.LEDGER).is_file() else []
    for ln in lines:
        try:
            e = json.loads(ln)
        except ValueError:
            continue
        name = str(e.get("name") or "")
        if e.get("event") in ("created", "id_seen_during_create", "create_failed") and e.get("id") \
                and name.startswith(dlx.PREFIX) and _run_tag(e.get("tag"), rd.name):
            created[e["id"]] = name
        if e.get("event") == "deleted":
            deleted.add(e.get("id"))
    todo = {i: n for i, n in created.items() if i not in deleted}
    d = dlx.client()
    # A sandbox whose id never reached the ledger is invisible to the ledger: always list by prefix (and this run's
    # tag label) too. Safe because cleanup refuses above while a driver of the run is live.
    for sb in d.list():
        if str(sb.name).startswith(dlx.PREFIX) and _run_tag((getattr(sb, "labels", None) or {}).get("tag"), rd.name):
            todo.setdefault(sb.id, sb.name)
    if not todo:
        print(f"no loop sandbox of {rd.name} is outstanding in the ledger or listed with our prefix")
        return 0
    for sid, name in todo.items():
        ok = dlx.delete_by_id(d, sid, "cleanup")
        print(f"{sid} {name}: deleted={ok}")
    print(f"still listed with our prefix: {dlx.list_ours(d)}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("leg")
    p.add_argument("--runner", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--skills-dir", default="")
    p.add_argument("--n-samples", type=int, default=3)
    p.add_argument("--concurrency", type=int, default=30)
    p.add_argument("--allow-test", action="store_true")
    p.add_argument("--max-sandboxes", type=int, default=None,
                   help="override RUNTIME daytona_leg.max_sandboxes for this leg (recorded)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--check-only", action="store_true")
    p = sub.add_parser("launch-base")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("status")
    p.add_argument("--run-dir", required=True)
    p = sub.add_parser("cleanup")
    p.add_argument("--run-dir", default="")
    p.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    return {"leg": cmd_leg, "launch-base": cmd_launch_base, "status": cmd_status,
            "cleanup": cmd_cleanup}[a.cmd](a)


if __name__ == "__main__":
    # ssb_leg_pool imports this module by name: make that the SAME module object as __main__, so a Refused raised
    # there is the Refused cmd_leg catches here.
    sys.modules.setdefault("ssb_leg_daytona_loop", sys.modules["__main__"])
    sys.exit(main())
