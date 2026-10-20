#!/usr/bin/env python3
"""THE RUN'S MACHINE POOL: its registry, leases and machine rules, and a pool host for SSB v2 legs.

The Claude Code legs' machine owner (cc_pool_host.py) and the stateful evolver's driver (pool_container_evolver.py) use
this module's pool registry -- <store>/<run>/pool/run/ with registry.json (host pid, pool dir, state), machine.json
(MACHINES_SCHEMA: the current pool machine -- sandbox id and name, state, ttl_expires_ts, retire_at_ts, egress allow
list -- and every machine of the host) and leases/ -- plus host_alive, pool_code, pool_root, reap_pool_sandboxes and
Machine's egress rule and TTL grant check. The rules they rely on:

  * NEVER A SECOND MACHINE: every sandbox of the run's pool (<PREFIX><pool code>-*) that Daytona still lists is
    deleted by id (ledgered) BEFORE a machine is created, and the host refuses while any stays listed; a replacement
    is created only once the current machine is deleted. A host that is closing writes the registry state "closing"
    first and "closed" only once every machine thread has ended; attach() WAITS (bounded, EVOSKILL_POOL_CLOSING_WAIT_S)
    while the registered host is alive and not accepting, instead of spawning a new host beside it.
  * A valid LEASE (<pool root>/leases/*.json, see Registry) keeps the machine; the machine retires before its TTL (no
    new unit after ttl - pool_ttl_margin_min; units are never killed by the TTL), and Daytona's readback must grant
    the TTL that was requested (Machine.check_granted_ttl).
  * EGRESS: with RUNTIME evolver_daytona.placement "pool_container" (the evolver's container on the pool machine)
    every pool machine is created with openrouter.ai + api.anthropic.com, and its egress probe requires both
    reachable and everything else refused; otherwise openrouter.ai only (api.anthropic.com must be refused).

The rest of this file is a pool HOST for legs whose solver is the harness-free runner (scripts/run_ssbv2_base30_dsv4f_n3.py
through OpenRouter; RUNTIME daytona_leg.pool true, keys and defaults in ssb_leg_daytona_loop.POOL_DEFAULTS): a
global queue of one-task units (train of the newest epoch first, then val oldest epoch first, then the held-out
test), a per-machine concurrency ramp or a fixed per-machine cap (Ramp), stress signals, per-leg spend guards, and
per-leg results written as soon as that leg's units have ended.

Subcommands:
  host   --pool-dir D     the pool host (spawned by the first follower of a run; not for hand use)
  status --run-dir R      pool registry + live status of a run's pool(s)

Nothing here reads a key except the host (DAYTONA_* / the OpenRouter key, in-process, as the leg driver does).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
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
from collections import Counter, deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(1, str(HERE.parents[1]))
if __name__ == "__main__":                       # one module object, whichever way this file is entered
    sys.modules.setdefault("ssb_leg_pool", sys.modules["__main__"])
import daytona_lib_loop as dlx  # noqa: E402
import ssb_leg_daytona_loop as L  # noqa: E402
from src.orchestrator import data_overlay  # noqa: E402
from src.orchestrator import host_tools  # noqa: E402
from src.orchestrator import infra as infra_cfg  # noqa: E402
from src.orchestrator.outcomes import GRADER_FAILURE_PATTERNS  # noqa: E402

base = L.base
# Where the repo subset lives on a machine: build_bundle mirrors it at the host repo's own absolute path.
SANDBOX_REPO = str(L.REPO)
POOL_SANDBOX_DIR = "/opt/pool"
UNIT_LAUNCH = "pool_unit_launch.sh"
POLL_SCRIPT = "pool_poll.py"
WATCH_SCRIPT = "pool_cmdwatch.py"
POOL_SCRIPTS = (UNIT_LAUNCH, POLL_SCRIPT, WATCH_SCRIPT)
CMDWATCH_EVERY_S = 2.0
CMDWATCH_STALE_S = 3 * CMDWATCH_EVERY_S          # an older watcher state = the watcher is not running
SAMPLER_STALE_S = 60.0                           # no sampler row this recent = no CPU signal = no cap raise
RUNNER_PATTERN = "run_ssbv2"                     # loop.RUNNER_PATTERN["ssbv2"]; never in the host's argv
TIMEOUT_TEXT = "Error: command timed out after"  # the executor's tool result for a solver bash timeout
# the grader-failure warnings (src/orchestrator/outcomes.py, the `grader_failed` outcome)
GRADER_FAIL_RE = re.compile("|".join(re.escape(p) for p in GRADER_FAILURE_PATTERNS))
LOST_REQUEUES_MAX = 1                            # a unit of a LOST machine is run again at most once
SIGNAL_KEYS = ("cores_1min", "throttle_ratio", "mem_gib", "live_timeouts", "exact_timeouts", "long_cmds",
               "long_max_s", "cmds_ge60", "solver_cmds", "watcher", "watcher_age_s", "sampler_age_s",
               "grader_failures", "smoke_stress", "disk_free_gib")
# ── the LONG-LIVED pool ──
POOL_SCOPE = "run"                               # pool_root() = <store>/<run>/pool/<POOL_SCOPE>: one pool per RUN
MACHINES_FILE = "machine.json"                   # the current pool machine(s), for other tools (Registry)
MACHINES_SCHEMA = "evoskill.pool_machines/1"
LEASES_DIR = "leases"
LEG_CLASS = {"train": 0, "val": 1, "test": 2}    # admission priority class (lower first); see priority_key
NO_TRAIN_SIBLING_ENV = "EVOSKILL_POOL_NO_TRAIN_SIBLING"   # "1" = a VAL-only epoch (the loop's skip_last_train)
LOG_SYNC_TIMEOUT_S = 1800                        # a leg's result waits this long at most for its machines' log sync
LOG_SYNC_TRIES = 3
DISK_RETIRE_GIB = 3.0                            # a machine with less free disk takes no new unit (it retires)
DECISIONS_MAX = 5000
RESULT_GUARANTEE = (
    "LONG-LIVED POOL: this result was written when every unit of this leg had ENDED and its outputs were SYNCED home "
    "and VERIFIED (runner rc read; its cache, tasks dir, result and logs packed, downloaded and extracted; the key "
    "scan of the unit's files and of the logs of every machine it ran on, synced AFTER the leg's last unit ended, "
    "clean; records counted and costed). The pool machine(s) may still be serving other legs, so they are NOT deleted "
    "at this point: the shard guarantee 'every sandbox verified deleted before the result' is replaced by this one; "
    "only machines already deleted must be absent from the Daytona listing (checked), and each machine's deletion is "
    "verified in the pool dir's pool_record.json when the pool closes.")
# ── the final-integration fixes (see the header) ──
PLACEMENT_OWN, PLACEMENT_POOL_CONTAINER = "own_sandbox", "pool_container"   # pool_container_evolver.PLACEMENTS
PLACEMENTS = (PLACEMENT_OWN, PLACEMENT_POOL_CONTAINER)
EVOLVER_DOMAIN = "api.anthropic.com"             # the evolver container's upstream (pce.MACHINE_ALLOW)
CLOSING_WAIT_S = 3600.0                          # attach(): the longest wait for a CLOSING host to exit (env override)
REAP_TRIES, REAP_PAUSE_S = 3, 10.0               # _init_pool: list -> delete -> list again, this many rounds
GUARD_EVERY_S = 30.0                             # a blocked machine creation re-lists this often
GUARD_ABORT_S = 1800.0                           # ... and aborts the pool after this long with work and no machine
# The code the HOST process runs (imported) or uploads to its machines (the pool scripts), repo-relative: hashed at
# the host's start; a join is refused once any file differs on disk
HOST_CODE = tuple(f"scripts/daytona/{f}" for f in ("ssb_leg_pool.py", "ssb_leg_daytona_loop.py", "ssb_leg_daytona.py",
                                                    "daytona_lib_loop.py") + POOL_SCRIPTS) + (
    "src/orchestrator/data_overlay.py", "src/orchestrator/infra.py", "src/orchestrator/outcomes.py")
# The runtime_env fields a unit's result carries for files its MACHINE holds (bundled at the machine's creation)
MACHINE_CODE = {"shim_sha256": "scripts/daytona/docker_shim.py", "entry_sha256": "scripts/daytona/sandbox_entry.py",
                "runner_sha256": L.RUNNER_REL, "executor_sha256": "src/harness/direct/executor.py",
                "scorer_sha256": "src/agent_profiles/ssbv2_agent/scorer.py",
                "solver_prompt_sha256": "src/agent_profiles/ssbv2_agent/prompt.md"}
# A POOL machine's egress probe: base.EGRESS_CMD's hosts + the evolver's upstream (reachable iff it is allowed)
POOL_EGRESS_CMD = r"""
/opt/evoskill/venv/bin/python - <<'EOF'
import json, urllib.request, urllib.error, socket
socket.setdefaulttimeout(12)
out = {}
for name, url in (("openrouter", "https://openrouter.ai/api/v1/models"),
                  ("anthropic", "https://api.anthropic.com/v1/models"),
                  ("example_com", "https://example.com/"),
                  ("pypi", "https://pypi.org/simple/"),
                  ("github", "https://github.com/"),
                  ("google_ip", "https://8.8.8.8/")):
    try:
        with urllib.request.urlopen(url) as r:
            out[name] = {"ok": True, "status": r.status}
    except urllib.error.HTTPError as e:
        out[name] = {"ok": True, "status": e.code}
    except Exception as e:
        out[name] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:120]}"}
print("EGRESS " + json.dumps(out))
EOF
"""
EGRESS_HOST = {"openrouter": "openrouter.ai", "anthropic": EVOLVER_DOMAIN}
# The same tree hash as ssb_leg_daytona_loop.skills_fingerprint, run INSIDE the machine on its copy of a skills dir
SKILLS_TREE_PY = ("import hashlib,sys,pathlib\n"
                  "sd=pathlib.Path(sys.argv[1]);h=hashlib.sha256()\n"
                  "for p in sorted(x for x in sd.rglob('*') if x.is_file()):\n"
                  "    h.update(str(p.relative_to(sd)).encode()+b'\\0'+p.read_bytes()+b'\\0')\n"
                  "print('SKILLS_SHA='+h.hexdigest())\n")


def count_timeouts(raw: bytes) -> int:
    """Solver bash commands that ran into the executor's timeout, in ONE cache record: tool-role messages whose
    content STARTS with the executor's timeout text. Counting the text anywhere would count every
    timeout twice (trace.cached_transcript repeats it)."""
    try:
        msgs = (json.loads(raw).get("trace") or {}).get("messages") or []
    except (ValueError, AttributeError):
        return 0
    n = 0
    for m in msgs:
        if not isinstance(m, dict) or m.get("role") != "tool":
            continue
        c = m.get("content")
        if isinstance(c, list):                          # content blocks: the first text block
            c = next((b.get("text") for b in c if isinstance(b, dict) and isinstance(b.get("text"), str)), "")
        if isinstance(c, str) and c.startswith(TIMEOUT_TEXT):
            n += 1
    return n


def grader_cap(cfg, p: dict) -> dict:
    """MACHINE-WIDE grader cap without touching the pinned scorer / sandbox_entry: their limits
    are per PROCESS (the leg's knobs) and every unit is one runner process, so at most K units run on a machine at
    once, K = pool_machine_recalc_budget // recalc-per-process (budget null = 2 x recalc-per-process, i.e. twice
    today's one-runner sandbox). Machine-wide worst cases are K x each per-process limit."""
    kn = dict(getattr(cfg, "knobs", None) or {})

    def per(k, dflt):
        try:
            return max(1, int(kn.get(k) or dflt))
        except (TypeError, ValueError):
            return dflt
    recalc = per("EVOSKILL_RECALC_CONCURRENCY", 6)           # scorer defaults when a knob is absent
    uno = per("EVOSKILL_DATATABLE_UNO_CONCURRENCY", 3)
    heavy = per("EVOSKILL_HEAVY_COMPARE_CONCURRENCY", 2)
    budget = p.get("pool_machine_recalc_budget")
    src = "RUNTIME daytona_leg.pool_machine_recalc_budget"
    if budget is None:
        budget, src = 2 * recalc, "default: 2 x the leg's EVOSKILL_RECALC_CONCURRENCY (today's one-runner sandbox x 2)"
    k = max(1, int(budget) // recalc)
    eff = min(k, int(p["pool_max_units_per_machine"]))
    n = int(getattr(cfg, "n_samples", 0) or 0)
    hv = min(eff, int(p["pool_max_heavy_units"]))
    # what can REALLY run at once at N samples per unit: a unit grades at most N samples at a time, so its
    # per-process semaphore binds only when it is below N; a fixed cap bounds every sample-level count too
    top = int(p["pool_fixed_inflight"]) if p.get("pool_fixed_inflight") else None

    def at_n(per, units):
        v = units * (min(per, n) if n else per)
        return min(v, top) if top else v
    return {"recalc_per_process": recalc, "uno_per_process": uno, "heavy_compare_per_process": heavy,
            "machine_recalc_budget": int(budget), "budget_from": src, "units_per_machine_k": k,
            "units_per_machine_cap": eff,
            "machine_max": {"recalc": eff * recalc, "uno": eff * uno,
                            # a heavy compare needs a heavy task: at most pool_max_heavy_units heavy units per machine
                            "heavy_compare": hv * heavy},
            "machine_max_at_n": {"recalc": at_n(recalc, eff), "uno": at_n(uno, eff), "heavy_compare": at_n(heavy, hv),
                                 "note": "per-process limit capped at N samples per unit, and at the fixed "
                                         "in-flight cap when one is set"},
            "attempts_per_machine_max_at_n": eff * n}


def cap_note(p: dict) -> str:
    """One phrase: the per-machine cap policy (check-only line, follower, host.log)."""
    if p.get("pool_fixed_inflight"):
        return (f"FIXED cap {p['pool_fixed_inflight']} in-flight attempts per machine (no ramp; stress recorded, "
                f"not acted on)")
    return f"ramp {p['ramp_start']} +{p['ramp_step']}/{p['ramp_interval_s']}s max {p['ramp_max']}"


def layout_problem(p: dict, gc: dict):
    """Why this pool layout cannot do what it declares (None = it can): a FIXED cap the grader cap K cannot reach
    (K units x n_samples < N) would silently run fewer attempts than declared (fixed mode)."""
    fx = p.get("pool_fixed_inflight")
    if fx and gc["attempts_per_machine_max_at_n"] < int(fx):
        return (f"pool_fixed_inflight={fx} but the machine-wide grader cap allows only "
                f"{gc['units_per_machine_cap']} units x n_samples = {gc['attempts_per_machine_max_at_n']} attempts per "
                f"machine (K = min(pool_machine_recalc_budget {gc['machine_recalc_budget']} // "
                f"{gc['recalc_per_process']} recalcs per runner, pool_max_units_per_machine)); raise "
                f"pool_machine_recalc_budget and pool_max_units_per_machine, or lower pool_fixed_inflight")
    return None


# What a follower hands the host about its leg: the arguments the leg script passed and the env values LegConfig
# reads. No key, no token: solver pins, the data-overlay path and the grader timeouts only.
JOIN_ENV_KEYS = tuple(L.SOLVER_PIN_KEYS) + (data_overlay.ENV,) + tuple(infra_cfg.GRADER_TIMEOUT_KNOBS)


def _host_script() -> Path:
    """The program a follower starts as the pool host. EVOSKILL_POOL_HOST_SCRIPT exists for the loop-contract tests
    only (a stand-in host that needs no Daytona); a real run leaves it unset and the registry records the path."""
    return Path(os.environ.get("EVOSKILL_POOL_HOST_SCRIPT") or (HERE / "ssb_leg_pool.py"))


def _poll_s() -> float:
    return float(L.POLL_S)                         # read at call time (tests set it to 0)


def leg_bash_timeout(cfg) -> tuple:
    """(seconds, where it comes from): the solver's bash timeout a unit of this leg runs under. The leg env decides
    when it sets EVOSKILL_BASH_TIMEOUT; else the runner's own os.environ.setdefault (600 on run_ssbv2_...: every
    recorded SSB v2 timeout reads "after 600s"); else the executor's default (120). Read, never assumed."""
    env = {}
    for ln in L.leg_env_text(cfg).splitlines():
        if ln.startswith("export ") and "=" in ln:
            k, v = ln[len("export "):].split("=", 1)
            env[k] = (shlex.split(v) or [""])[0]
    if env.get("EVOSKILL_BASH_TIMEOUT"):
        return int(env["EVOSKILL_BASH_TIMEOUT"]), "leg env EVOSKILL_BASH_TIMEOUT"
    m = re.search(r'os\.environ\.setdefault\("EVOSKILL_BASH_TIMEOUT",\s*"(\d+)"\)',
                  (L.REPO / L.RUNNER_REL).read_text())
    if m:
        return int(m.group(1)), f"{L.RUNNER_NAME} os.environ.setdefault"
    m = re.search(r'BASH_TIMEOUT = int\(os\.environ\.get\("EVOSKILL_BASH_TIMEOUT", "(\d+)"\)\)',
                  (L.REPO / "src/harness/direct/executor.py").read_text())
    return (int(m.group(1)) if m else 120), "executor default"


def log_line(msg, fh=None):
    L.log_line(msg, fh)


# ───────────────────────────────────────────────────────────────────────────── the ramp (pure)
def stress_reasons(sig: dict, p: dict) -> list:
    """Why a machine counts as STRESSED at this poll ([] = not stressed). `sig` = Machine.signals()."""
    why = []
    c = sig.get("cores_1min")
    if c is not None and c >= float(p["stress_cores"]):
        why.append(f"cores_busy 1-min mean {c:.2f} >= {p['stress_cores']}")
    t = sig.get("throttle_ratio")
    if t is not None and t >= float(p["stress_throttle_ratio"]):
        why.append(f"CFS throttled {t:.0%} of periods >= {float(p['stress_throttle_ratio']):.0%}")
    m = sig.get("mem_gib")
    if m is not None and m >= float(p["stress_mem_gib"]):
        why.append(f"memory {m:.1f} GiB >= {p['stress_mem_gib']}")
    if p.get("stress_on_timeout", True) and sig.get("live_timeouts"):
        why.append(f"{sig['live_timeouts']} solver bash command(s) ran into the bash timeout (live watcher)")
    if p.get("stress_on_timeout", True) and sig.get("exact_timeouts"):
        why.append(f"{sig['exact_timeouts']} solver bash timeout(s) in records of units that finished here")
    if float(p.get("bash_watch_s") or 0) > 0 and sig.get("long_cmds"):
        why.append(f"{sig['long_cmds']} solver bash command(s) running >= {p['bash_watch_s']}s "
                   f"(longest {sig.get('long_max_s')}s)")
    if p.get("stress_on_grader_failure", True) and sig.get("grader_failures"):
        why.append(f"{sig['grader_failures']} grader failure warning(s) (the grader_failed outcome: LibreOffice "
                   f"recalculation / data-table solve / inline scoring failed)")
    if sig.get("smoke_stress"):
        why.append(f"SMOKE synthetic stress (smoke_stress_after_s={p.get('smoke_stress_after_s')}; machine ready "
                   f"{sig['smoke_stress']}s)")
    return why


def blind_reason(sig: dict):
    """Why a machine may NOT raise its cap at this poll although nothing reads as stress (None = it may): no CPU
    signal (a dead or stale sampler must not read as 'unstressed')."""
    age = sig.get("sampler_age_s")
    if sig.get("cores_1min") is None or age is None or float(age) > SAMPLER_STALE_S:
        return f"no CPU signal (cores_1min {sig.get('cores_1min')}, sampler_age_s {age})"
    return None


class Ramp:
    """One machine's cap on in-flight attempts.

    * start `ramp_start`; every `ramp_interval_s` since the last change, +`ramp_step` up to the ceiling (initially
      `ramp_max`) -- only if the cap was BINDING in that window (note_binding) and the pool is not in an API-429 hold;
    * a STRESSED poll freezes the ceiling at the current cap (a stressed machine never raises again) and lowers the
      cap by `ramp_step` (floor `ramp_min`) -- at most once per `ramp_down_min_s`, and only once the previous
      decrease took effect (in-flight <= cap) or a full `ramp_interval_s` passed, so a slow drain cannot ratchet the
      cap to the floor. The cap only gates NEW dispatches; nothing running is ever stopped.
    * a BLIND poll (no CPU signal: blind_reason) never raises -- a dead sampler is not "unstressed".
    Every change (and a ceiling freeze that could not lower yet: kind "hold") is appended to `changes`.

    FIXED mode (pool_fixed_inflight N): cap = ceiling = N for
    the machine's whole life; `changes` stays empty. Every poll's stress reasons are still computed and counted
    (stressed_polls, blind_polls), and each START of a stressed stretch is appended to `stress_log` (kind
    "stress_recorded", record_only True; at most STRESS_LOG_MAX entries) and returned so the machine logs it."""
    STRESS_LOG_MAX = 500

    def __init__(self, p: dict, now: float):
        self.p = p
        fx = p.get("pool_fixed_inflight")
        self.fixed = int(fx) if fx else None
        self.cap = self.fixed or int(p["ramp_start"])
        self.ceiling = self.fixed or int(p["ramp_max"])
        self.last_change = float(now)
        self.binding = False
        self.changes: list = []
        self.polls = 0
        self.stressed_polls = 0
        self.blind_polls = 0
        self.raises_blocked_blind = 0
        self.last_blind = None
        self.stress_log: list = []       # FIXED mode: stressed stretches, recorded only
        self.stress_stretches = 0
        self._stressed_now = False

    def note_binding(self) -> None:
        self.binding = True

    def observe(self, now: float, sig: dict, inflight: int, hold: bool = False):
        p = self.p
        self.polls += 1
        why = stress_reasons(sig, p)
        if self.fixed is not None:
            return self._observe_fixed(now, sig, inflight, why)
        since = now - self.last_change
        if why:
            self.stressed_polls += 1
            froze = self.ceiling > self.cap
            self.ceiling = min(self.ceiling, self.cap)
            if (self.cap > int(p["ramp_min"]) and since >= float(p["ramp_down_min_s"])
                    and (inflight <= self.cap or since >= float(p["ramp_interval_s"]))):
                return self._change(now, max(int(p["ramp_min"]), self.cap - int(p["ramp_step"])), "down",
                                    why, sig, inflight)
            if froze:
                return self._change(now, self.cap, "hold", why, sig, inflight)
            return None
        blind = blind_reason(sig)
        if blind:
            self.blind_polls += 1
            self.last_blind = blind
        if (not hold and self.binding and self.cap < self.ceiling
                and since >= float(p["ramp_interval_s"])):
            if blind:
                self.raises_blocked_blind += 1
                return None
            return self._change(now, min(self.ceiling, self.cap + int(p["ramp_step"])), "up", [], sig, inflight)
        return None

    def _observe_fixed(self, now, sig, inflight, why):
        """FIXED mode: never a cap change. A stressed poll is counted; the first of a stretch is recorded."""
        blind = blind_reason(sig)
        if blind:
            self.blind_polls += 1
            self.last_blind = blind
        if not why:
            self._stressed_now = False
            return None
        self.stressed_polls += 1
        if self._stressed_now:
            return None
        self._stressed_now = True
        self.stress_stretches += 1
        ev = {"ts": round(now, 1), "utc": dlx.now(), "kind": "stress_recorded", "record_only": True,
              "cap": self.cap, "inflight": int(inflight), "reasons": list(why),
              "signals": {k: sig.get(k) for k in SIGNAL_KEYS if k in sig}}
        if len(self.stress_log) < self.STRESS_LOG_MAX:
            self.stress_log.append(ev)
        return ev

    def _change(self, now, new, kind, why, sig, inflight) -> dict:
        ch = {"ts": round(now, 1), "utc": dlx.now(), "kind": kind, "from": self.cap, "to": int(new),
              "inflight": int(inflight), "reasons": list(why),
              "signals": {k: sig.get(k) for k in SIGNAL_KEYS if k in sig}}
        if kind != "hold":
            self.cap = int(new)
            if kind == "down":
                self.ceiling = min(self.ceiling, self.cap)
            self.last_change = now
            self.binding = False
        ch["ceiling"] = self.ceiling
        self.changes.append(ch)
        return ch

    def record(self) -> dict:
        rec = {"cap": self.cap, "ceiling": self.ceiling, "polls": self.polls,
               "stressed_polls": self.stressed_polls, "blind_polls": self.blind_polls,
               "raises_blocked_blind": self.raises_blocked_blind, "last_blind": self.last_blind,
               "cap_changes": list(self.changes), "cap_mode": "fixed" if self.fixed is not None else "ramp"}
        if self.fixed is not None:
            rec.update(fixed_inflight=self.fixed, stress_stretches=self.stress_stretches,
                       stress_log=list(self.stress_log))
        return rec


# ───────────────────────────────────────────────────────────────────────────── dispatch (pure)
def pick_machine(unit, machines: list, p: dict, units_cap=None):
    """(machine or None, [machines refused ONLY because of their cap]). Eligible: ready, not a machine this unit
    was lost on, fewer than `units_cap` active units (the machine-wide grader cap K, grader_cap; default
    pool_max_units_per_machine), the heavy rule (<= pool_max_heavy_units heavy units in flight), and
    in-flight + need <= cap -- an IDLE machine (in-flight 0) always takes one unit, so a cap below n_samples can
    never stall the pool. Best = most headroom (cap - in-flight), then fewest units, then name."""
    ok, capped = [], []
    ucap = int(p["pool_max_units_per_machine"]) if units_cap is None else int(units_cap)
    avoid = getattr(unit, "avoid", None) or ()
    for m in machines:
        if m.name in avoid:
            continue
        if m.active_units() >= ucap:
            continue
        if unit.heavy and m.heavy_units() >= int(p["pool_max_heavy_units"]):
            continue
        infl = m.inflight()
        if infl > 0 and infl + unit.need > m.ramp.cap:
            capped.append(m)
            continue
        ok.append(m)
    if not ok:
        return None, capped
    best = max(ok, key=lambda m: (m.ramp.cap - m.inflight(), -m.active_units(), [-ord(c) for c in m.name]))
    return best, capped


def leg_key(lg) -> str:
    """The leg key (train / val / test) of a PoolLeg (or a stand-in with .cfg.key); '' when unknown."""
    return str(getattr(getattr(lg, "cfg", None), "key", "") or "")


def leg_epoch(tag):
    """The epoch of a loop leg tag: <run>_v<N>_<key> -> N, <run>_base_<key> -> 0 (the loop's EpochPaths naming);
    None for any other tag (it sorts as the OLDEST: -1)."""
    m = re.search(r"_v(\d+)_(train|val|test)$", str(tag))
    if m:
        return int(m.group(1))
    return 0 if re.search(r"_base_(train|val|test)$", str(tag)) else None


def leg_class(lg) -> int:
    """Admission class of a leg: train 0, val 1, test 2 (anything else 3)."""
    return LEG_CLASS.get(leg_key(lg), 3)


def priority_key(lg, pos: int = 0) -> tuple:
    """Dispatch priority of a LEG (lower first): TRAIN of the NEWEST epoch first (after each evolver call the new
    training rollouts take each slot as it opens), then VAL of the OLDEST epoch first (the unrun val work of earlier epochs is deprioritized
    behind train, not behind newer val), then the held-out TEST last. `pos` = join order (tie-break)."""
    e = leg_epoch(getattr(getattr(lg, "cfg", None), "tag", ""))
    e = -1 if e is None else e
    c = leg_class(lg)
    return (c, -e if c == 0 else e, pos)


def order_queue(legs: list, order: str = "interleave") -> list:
    """Queued units in dispatch order.
    "interleave": every heavy unit first (spread by the heavy rule), then the legs' light units interleaved (leg 1,
    leg 2, leg 1, ...) each in its frozen split order.
    "train_first": leg by leg in priority_key order (newest train, then val oldest epoch first, then test), each
    leg's units heavy first, then its split order. Which of them may actually START is admission_block's decision."""
    if order == "train_first":
        ranked = sorted(enumerate(legs), key=lambda il: priority_key(il[1], il[0]))
        return [u for _i, lg in ranked for u in _interleave([lg])]
    return _interleave(legs)


def train_sibling_tag(tag: str):
    """The train leg of the same epoch as a val leg (<run>_vN_val -> <run>_vN_train, <run>_base_val ->
    <run>_base_train: the loop's EpochPaths naming); None for any other tag."""
    return tag[: -len("_val")] + "_train" if str(tag).endswith("_val") else None


HOLD_TRAIN_WAITING, HOLD_SIBLING_PENDING, HOLD_SIBLING_ABSENT = "train_waiting", "train_join_pending", "train_not_joined"
HOLD_HIGHER_WAITING = "train_or_val_waiting"


def no_train_sibling(cfg, env=None) -> bool:
    """The VAL-ONLY-epoch flag of a leg (EVOSKILL_POOL_NO_TRAIN_SIBLING, set by the loop for the val leg of an epoch
    whose train leg is skipped by design, RUNTIME skip_last_train): True = this val leg never waits for a train
    sibling. Unset / "0" = False; "1" on a train or test leg, or any other value, is Refused."""
    env = os.environ if env is None else env
    v = str(env.get(NO_TRAIN_SIBLING_ENV) or "").strip()
    if v in ("", "0"):
        return False
    if v != "1":
        raise L.Refused(f"{NO_TRAIN_SIBLING_ENV}={v!r}: '1' (a val-only epoch: no train leg of this epoch will "
                        f"join the pool) or unset")
    if getattr(cfg, "key", None) != "val":
        raise L.Refused(f"{NO_TRAIN_SIBLING_ENV}=1 declares a VAL-only epoch; {cfg.tag} is a {cfg.key} leg")
    return True


def queued_by_class(legs: dict) -> Counter:
    """{leg class: queued units} over every leg of the pool (admission_block's input; dispatch keeps it current)."""
    return Counter(leg_class(lg) for lg in legs.values() for x in lg.units if x.state == "queued")


def admission_block(u, legs: dict, p: dict, now: float, pending_tags=(), queued=None):
    """TRAIN FIRST (a val rollout is admitted only when every train rollout has finished or already has a spot in
    the pool): (kind, reason) why unit `u` may NOT be dispatched now, or None.
    Only pool_order "train_first" blocks anything, and never a train unit:
      * a VAL unit is held while any unit of a train leg in this pool (any epoch) is still QUEUED (not ended, not
        assigned/running); a TEST unit while any train or val unit is (the held-out test runs last);
      * a val unit whose train sibling has a join pending, or has not joined and has no result JSON yet, waits up to
        pool_train_first_wait_s from its own leg's acceptance (the loop dispatches train first, so this is the rare
        race of val's join landing first) -- unless its leg declared a VAL-ONLY epoch (no_train_sibling).
    `queued` = queued_by_class(legs), when the caller keeps it (dispatch: one count per pass, not one per unit)."""
    if p.get("pool_order") != "train_first":
        return None
    cls = leg_class(u.leg)
    if cls == 0:
        return None
    q = queued_by_class(legs) if queued is None else queued
    waiting = sum(n for c, n in q.items() if c < cls)
    if waiting > 0:
        if cls == 1:
            return HOLD_TRAIN_WAITING, f"{waiting} train unit(s) still waiting for a slot"
        return HOLD_HIGHER_WAITING, (f"{q.get(0, 0)} train + {q.get(1, 0)} val unit(s) still waiting for a slot "
                                     f"(the held-out test runs last)")
    if leg_key(u.leg) == "val" and not getattr(u.leg, "no_train_sibling", False):
        sib = train_sibling_tag(u.leg.cfg.tag)
        if sib and sib in pending_tags:
            return HOLD_SIBLING_PENDING, f"train leg {sib} has a join pending"
        if sib and sib not in legs and not (Path(u.leg.cfg.run_dir) / f"{sib}.json").exists():
            lim = float(p.get("pool_train_first_wait_s") or 0)
            waited = now - float(getattr(u.leg, "t_accepted", None) or now)
            if waited < lim:
                return HOLD_SIBLING_ABSENT, (f"train leg {sib} has not joined and has no result yet (waiting "
                                             f"{waited:.0f}/{lim:.0f}s for it)")
    return None


def _interleave(legs: list) -> list:
    pending = [[u for u in lg.units if u.state == "queued"] for lg in legs]
    heavy = [u for lst in pending for u in lst if u.heavy]
    light = [[u for u in lst if not u.heavy] for lst in pending]
    out, i = list(heavy), 0
    while any(light):
        for lst in light:
            if i < len(lst):
                out.append(lst[i])
        i += 1
        if all(i >= len(lst) for lst in light):
            break
    return out


# ───────────────────────────────────────────────────────────────────────────── registry / joins
def pool_root(cfg) -> Path:
    """<store_root>/<run>/pool/run: the RUN's pool (one for every leg of the run)."""
    return Path(cfg.rt["store_root"]) / cfg.run_dir.name / "pool" / POOL_SCOPE


def pool_code(run_name: str) -> str:
    """DNS-safe sandbox-name code of a run's pool machines: rp<sha5 of the run name> (leg_code's shape)."""
    return "rp" + hashlib.sha256(str(run_name).encode()).hexdigest()[:5]


class Registry:
    """<store>/<run>/pool/run/ -- the run's pool, as other processes find it:
      registry.json  {"state": "accepting"|"closing"|"closed", "host_pid", "pool_dir", "host_script", "scope": "run",
                     ...} (under registry.lock: followers join the live accepting host; WAIT while the registered host
                     is alive and not accepting -- "closing" = its machines are being released / deleted, "closed" =
                     they are gone and it is writing its records; else spawn a new one, whose _init_pool reaps any
                     machine a dead host left behind)
      machine.json   MACHINES_SCHEMA, rewritten by the host whenever a machine changes: "current" = the machine
                     that takes units now (or null), "machines": {name: {sandbox_id, sandbox_name, state (starting |
                     ready | retiring | released | deleted | failed), created_ts, ready_ts, ttl_minutes,
                     ttl_expires_ts, retire_at_ts (no new unit after it), domain_allow_list, units_running,
                     inflight, cap, leased_by}}, host_pid, pool_dir, generation (+1 per change)
      leases/<holder>.json  written by ANOTHER tool that runs work on a pool machine (e.g. an evolver container):
                     {"holder", "sandbox_id", "until_ts", "pid" (optional: the local process holding it)}. While
                     valid (until_ts in the future, pid alive when given) the host never releases that machine for
                     idleness nor for its retirement -- but never past ttl_expires_ts - POOL_SYNC_MARGIN_MIN: the
                     holder must check retire_at_ts before starting work that would outlive it (a lease past it
                     delays the replacement, and queued units wait)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.file = self.root / "registry.json"
        self.machines_file = self.root / MACHINES_FILE
        self.leases_dir = self.root / LEASES_DIR

    def lock(self):
        return L._Lock(self.root / "registry.lock")

    def read(self) -> dict:
        try:
            return json.loads(self.file.read_text())
        except (OSError, ValueError):
            return {}

    def write(self, d: dict) -> None:
        L._atomic_write(self.file, d)

    def leases(self, now: float) -> dict:
        """{sandbox_id: [holder, ...]} of the VALID leases now (unreadable / expired / dead-pid ones ignored)."""
        out: dict = {}
        if not self.leases_dir.is_dir():
            return out
        for f in sorted(self.leases_dir.glob("*.json")):
            try:
                d = json.loads(f.read_text())
                until = float(d.get("until_ts") or 0)
            except (OSError, ValueError, TypeError, AttributeError):
                continue
            if until <= now or not d.get("sandbox_id"):
                continue
            if d.get("pid") and not _pid_alive(d["pid"]):
                continue
            out.setdefault(str(d["sandbox_id"]), []).append(str(d.get("holder") or f.stem))
        return out


def evolver_placement(run_dir) -> str:
    """RUNTIME.json evolver_daytona.placement of the run: absent -> "own_sandbox" (the evolver's own VM; the pool
    machine's egress is openrouter.ai only), "pool_container" -> the evolver runs in a container ON the pool machine
    (its egress adds api.anthropic.com). Anything else is Refused (src/orchestrator/pool_container_evolver.placement
    raises on it too)."""
    f = Path(run_dir) / "RUNTIME.json"
    try:
        rt = json.loads(f.read_text()) if f.is_file() else {}
    except (OSError, ValueError) as e:
        raise L.Refused(f"{f} is not readable JSON: {e}")
    blk = rt.get("evolver_daytona") or {}
    if not isinstance(blk, dict):
        raise L.Refused("RUNTIME.json evolver_daytona must be an object")
    v = blk.get("placement", PLACEMENT_OWN)
    if v not in PLACEMENTS:
        raise L.Refused(f"RUNTIME.json evolver_daytona.placement={v!r} is not one of {list(PLACEMENTS)}")
    return v


def allow_domains(placement: str = PLACEMENT_OWN) -> list:
    """The egress allow list a pool machine is created with (Machine.allow_list -> Shard._create_once), as a list:
    the runner's ALLOW_DOMAINS (openrouter.ai), plus api.anthropic.com ONLY for placement "pool_container"."""
    d = L.ALLOW_DOMAINS
    out = [x.strip() for x in d.split(",") if x.strip()] if isinstance(d, str) else list(d)
    if placement == PLACEMENT_POOL_CONTAINER and EVOLVER_DOMAIN not in out:
        out.append(EVOLVER_DOMAIN)
    return out


def code_manifest(rels) -> dict:
    """{repo-relative path: sha256 (None = unreadable)} of these files as they are on disk NOW."""
    out = {}
    for rel in rels:
        try:
            out[rel] = L.sha256_file(L.REPO / rel)
        except OSError:
            out[rel] = None
    return out


def pool_sandboxes_listed(client, code: str) -> list:
    """The run's POOL sandboxes Daytona lists now ({id, name, state}; name <PREFIX><pool code>-*), destroyed ones
    excluded. Raises when the listing fails (callers fail CLOSED: no machine is created without it)."""
    pre = f"{dlx.PREFIX}{code}-"
    return [o for o in dlx.list_ours(client) if str(o.get("name") or "").startswith(pre)
            and "destroy" not in str(o.get("state") or "").lower()]


def reap_pool_sandboxes(client, code: str, tag: str, log, why: str, tries=None, pause_s=None, found=None) -> list:
    """Delete BY ID (ledgered: daytona_lib_loop.delete_by_id + a pool_orphan_found row) every sandbox of this run's
    pool that Daytona lists, list again, up to `tries` rounds. Returns what is STILL listed ([] = none). Raises when
    the listing itself failed every round. Only for a host that owns no machine yet (_init_pool): the registry lock and
    attach()'s wait make every such sandbox an orphan of a dead host (or of a host that could not delete it)."""
    tries = max(1, REAP_TRIES if tries is None else int(tries))
    pause_s = REAP_PAUSE_S if pause_s is None else float(pause_s)
    for i in range(tries):
        try:
            rows = pool_sandboxes_listed(client, code)
        except Exception as e:  # noqa: BLE001
            log(f"listing the run's pool sandboxes failed ({type(e).__name__}: {str(e)[:200]}); retry {i + 1}/{tries}")
            time.sleep(pause_s)
            continue
        if not rows:
            return []
        unverified = 0
        for o in rows:
            if found is not None:
                found.append({**o, "utc": dlx.now()})
            dlx.ledger(event="pool_orphan_found", tag=tag, id=o["id"], name=o["name"], state=o["state"], why=why)
            ok = dlx.delete_by_id(client, o["id"], tag)
            unverified += 0 if ok else 1
            log(f"ORPHAN pool sandbox {o['name']} ({o['id']}, {o['state']}): {why} -> deleted by id: "
                f"{'verified' if ok else 'NOT verified'}")
        if unverified:
            time.sleep(pause_s)
    return pool_sandboxes_listed(client, code)          # a failing final listing raises: callers fail closed


def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, ValueError, TypeError):
        return False
    return True


def src_manifest() -> dict:
    """{repo-relative path: sha256} of the repo's src/ tree, the files build_bundle ships to a machine (same filter)."""
    out = {}
    for p in sorted((L.REPO / "src").rglob("*")):
        if "__pycache__" in p.parts or p.suffix == ".pyc" or not p.is_file():
            continue
        out[p.relative_to(L.REPO).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def manifest_sha(man) -> str | None:
    if man is None:
        return None
    return hashlib.sha256(json.dumps(sorted(man.items())).encode()).hexdigest()


def _cmdline(pid: int) -> str:
    q = subprocess.run(["ps", "-p", str(int(pid)), "-o", "command="], capture_output=True, text=True)
    return q.stdout.strip()


def host_alive(reg: dict) -> bool:
    """The registered host is alive AND still that host (identity from its own command line: program + pool dir)."""
    pid = int(reg.get("host_pid") or 0)
    if pid <= 0:
        return False
    cmd = _cmdline(pid)
    return bool(cmd) and Path(str(reg.get("host_script") or "")).name in cmd and \
        str(reg.get("pool_dir") or "\0") in cmd


def env_subset(env) -> dict:
    return {k: str(env[k]) for k in JOIN_ENV_KEYS if env.get(k)}


def _ack_timeout_s() -> float:
    return float(os.environ.get("EVOSKILL_POOL_ACK_TIMEOUT_S", "240"))


def join_request(cfg, allow_test: bool = False, env=None, ack_timeout_s=None) -> dict:
    env = os.environ if env is None else env
    req = {"tag": cfg.tag, "key": cfg.key, "run_dir": str(cfg.run_dir), "split": str(cfg.split_file),
           "cache_dir": str(cfg.cache_dir), "skills_dir": str(cfg.skills_dir) if cfg.skills_dir else "",
           "n_samples": cfg.n_samples, "concurrency": cfg.concurrency, "allow_test": bool(allow_test),
           "no_train_sibling": no_train_sibling(cfg, env),
           "env": env_subset(env), "follower_pid": os.getpid(),
           "requested_utc": dlx.now(), "requested_ts": time.time(),
           "ack_timeout_s": float(_ack_timeout_s() if ack_timeout_s is None else ack_timeout_s),
           "nonce": f"{cfg.tag}.{os.getpid()}.{time.time_ns()}"}
    assert not any("OPENROUTER" in k or "DAYTONA" in k for k in req["env"])
    return req


def write_join(pdir: Path, req: dict) -> Path:
    j = Path(pdir) / "joins"
    j.mkdir(parents=True, exist_ok=True)
    f = j / f"{req['nonce']}.join.json"
    L._atomic_write(f, req)
    return f


def _stem(join_file: Path) -> str:
    return join_file.name[: -len(".join.json")]


def _ack_path(join_file: Path) -> Path:
    return join_file.with_name(_stem(join_file) + ".ack.json")


# JOIN HAND-OVER. A join is claimed by the host (<nonce>.claim) or cancelled by its follower
# (<nonce>.cancel.json), never both: each side decides under the join's own flock (<nonce>.lock). The host refuses a
# cancelled or stale join (older than the follower's ack timeout), re-checks the cancel marker before it acks, and
# once more when it finishes the leg, so a follower that recorded FAILED is never run -- or never written OK -- later.
def _claim_path(join_file: Path) -> Path:
    return join_file.with_name(_stem(join_file) + ".claim")


def _cancel_path(join_file: Path) -> Path:
    return join_file.with_name(_stem(join_file) + ".cancel.json")


def _join_lock(join_file: Path):
    return L._Lock(join_file.with_name(_stem(join_file) + ".lock"))


def spawn_host(pdir: Path) -> int:
    """Start the pool host DETACHED (own session, own log). Its argv must not carry the runner pattern: the loop's
    seat guard counts `run_ssbv2` processes as legs and would count the host as a third."""
    script = _host_script()
    argv = [host_tools.python_executable(), str(script), "host", "--pool-dir", str(pdir)]
    if any(RUNNER_PATTERN in a for a in argv):
        raise L.Refused(f"the pool host's command line would contain {RUNNER_PATTERN!r} ({argv}); the loop's seat "
                        f"guard would count it as a leg")
    lf = open(Path(pdir) / "host.out", "ab")
    p = subprocess.Popen(argv, cwd=str(L.REPO), stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
    threading.Thread(target=p.wait, daemon=True, name="reap-host").start()
    return p.pid


def _closing_wait_s() -> float:
    return float(os.environ.get("EVOSKILL_POOL_CLOSING_WAIT_S", str(CLOSING_WAIT_S)))


def attach(cfg, allow_test: bool = False, *, spawn=spawn_host, ack_timeout_s=None, closing_wait_s=None,
           note=None, poll_s: float = 2.0) -> tuple:
    """(pool dir, join file, host pid, spawned?) -- under the registry flock: join the RUN's live, accepting
    host (whatever epoch its other legs belong to), else write the join into a NEW pool dir and spawn its host (join
    first, so the host finds it).
    While the registered host is ALIVE but no longer accepting (closing: its machines are being released
    and deleted; closed: it is writing its records), a new host would create a second machine beside the first one --
    so the follower WAITS for that host to exit (the lock is released while waiting), at most `closing_wait_s`
    (EVOSKILL_POOL_CLOSING_WAIT_S, default 3600 s); past it the leg is REFUSED, never a second host spawned."""
    reg = Registry(pool_root(cfg))
    join_request(cfg, allow_test, ack_timeout_s=ack_timeout_s)       # a bad leg refuses before any wait
    lim = _closing_wait_s() if closing_wait_s is None else float(closing_wait_s)
    t0, noted = time.time(), False
    while True:
        with reg.lock():
            r = reg.read()
            alive = bool(r) and host_alive(r)
            if r.get("state") == "accepting" and alive:
                pdir = Path(r["pool_dir"])
                req = join_request(cfg, allow_test, ack_timeout_s=ack_timeout_s)
                return pdir, write_join(pdir, req), int(r["host_pid"]), False
            if not alive:
                pdir = reg.root / f"{time.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}"
                pdir.mkdir(parents=True)
                jf = write_join(pdir, join_request(cfg, allow_test, ack_timeout_s=ack_timeout_s))
                pid = spawn(pdir)
                prev = dict(r) if r else None
                if prev:
                    prev.pop("previous", None)           # one level of history, never a chain across a run's hosts
                reg.write({"state": "accepting", "host_pid": pid, "pool_dir": str(pdir), "scope": POOL_SCOPE,
                           "run_dir": str(cfg.run_dir), "host_script": str(_host_script()), "spawned_utc": dlx.now(),
                           "spawned_by": {"tag": cfg.tag, "pid": os.getpid()}, "previous": prev,
                           "waited_for_closing_host_s": round(time.time() - t0, 1) if noted else 0.0,
                           "machines_file": str(reg.machines_file), "leases_dir": str(reg.leases_dir)})
                return pdir, jf, pid, True
            waited = time.time() - t0
            if waited > lim:
                raise L.Refused(f"the run's pool host pid {r.get('host_pid')} (pool dir {r.get('pool_dir')}) is still "
                                f"alive in state {r.get('state')!r} after {waited:.0f}s: its machine(s) may still be "
                                f"alive, and a new host now would be a SECOND machine (a run uses one machine). "
                                f"See its host.log / `ssb_leg_pool.py status --run-dir {cfg.run_dir}`; relaunch the "
                                f"leg once that host exited (EVOSKILL_POOL_CLOSING_WAIT_S={lim:.0f})")
        if not noted and note is not None:
            note(f"the run's pool host pid {r.get('host_pid')} is {r.get('state')!r} (its machines are being "
                 f"released); waiting up to {lim:.0f}s for it to exit before starting a new host")
        noted = True
        time.sleep(poll_s)


# ───────────────────────────────────────────────────────────────────────────── the follower (one per leg)
def follow(cfg, a=None, *, out=print, spawn=spawn_host, poll_s=None, ack_timeout_s=None) -> int:
    """The leg's process in pool mode: join (or spawn) the run's pool, then wait for this leg's DONE file (written as
    soon as this leg's own units have ended, whatever else the pool still runs).
    0 = <tag>.json written; 5 = <tag>.FAILED.json written (by the host, or here if the host died); 4 = refused."""
    poll_s = float(os.environ.get("EVOSKILL_POOL_FOLLOW_POLL_S", "3")) if poll_s is None else poll_s
    ack_timeout_s = _ack_timeout_s() if ack_timeout_s is None else float(ack_timeout_s)

    def say(msg):
        out(f"{dlx.now()} [pool-follower {cfg.tag}] {msg}", flush=True)

    p = L.pool_settings(cfg.rt)
    pdir, jf, hpid, spawned = attach(cfg, bool(getattr(a, "allow_test", False)), spawn=spawn,
                                     ack_timeout_s=ack_timeout_s, note=say)
    say(f"{'spawned pool host' if spawned else 'joined the pool of host'} pid {hpid} (pool dir {pdir}); "
        f"pool_sandboxes={p['pool_sandboxes']} {cap_note(p)}; order {p['pool_order']}")
    reg_view = {"host_pid": hpid, "pool_dir": str(pdir), "host_script": str(_host_script())}
    ack_f = _ack_path(jf)
    abort = {"requested": None}

    def on_sig(signum, _frame):
        if abort["requested"] is None:
            abort["requested"] = time.time()
            say(f"signal {signum}: asking the pool host to ABORT this leg (the other leg continues)")

    prev = {}
    try:
        for sg in (signal.SIGTERM, signal.SIGINT):
            prev[sg] = signal.getsignal(sg)
            signal.signal(sg, on_sig)
    except ValueError:
        pass                                            # not the main thread (tests)
    try:
        return _follow_wait(cfg, pdir, jf, hpid, reg_view, ack_f, abort, say, poll_s, ack_timeout_s)
    finally:
        for sg, h in prev.items():
            try:
                signal.signal(sg, h)
            except (ValueError, TypeError):
                pass


def _follow_wait(cfg, pdir, jf, hpid, reg_view, ack_f, abort, say, poll_s, ack_timeout_s) -> int:
    t0 = time.time()
    grace = float(os.environ.get("EVOSKILL_POOL_CLAIM_GRACE_S", "1800"))
    claimed_noted = False
    while not ack_f.exists():
        if not host_alive(reg_view):
            time.sleep(0.5)
            if not ack_f.exists():
                return _host_died(cfg, reg_view, say, "host gone before acknowledging the join", jf=jf)
        waited = time.time() - t0
        if waited > ack_timeout_s:
            with _join_lock(jf):
                if ack_f.exists():
                    break
                if _claim_path(jf).exists() and waited <= ack_timeout_s + grace:
                    if not claimed_noted:                   # the host took the join: it decides, we wait
                        say(f"pool host {hpid} claimed the join but has not answered in {waited:.0f}s; waiting "
                            f"up to {grace:.0f}s more")
                        claimed_noted = True
                else:
                    say(f"no answer from pool host {hpid} in {waited:.0f}s")
                    return _give_up_locked(cfg, jf, reg_view, say, f"no ack within {waited:.0f}s")
        time.sleep(min(poll_s, 1.0))
    ack = json.loads(ack_f.read_text())
    if not ack.get("accepted"):
        say(f"REFUSED by the pool host: {ack.get('reason')}")
        print(f"REFUSING: {ack.get('reason')}", file=sys.stderr, flush=True)
        return L.EXIT_REFUSED
    done_f = Path(ack["done_file"])
    say(f"accepted as attempt {ack.get('attempt')}; waiting for {done_f.name}")
    last_note, abort_sent = 0.0, False
    while True:
        if done_f.exists():
            d = json.loads(done_f.read_text())
            say(f"LEG {'OK' if d.get('rc') == L.EXIT_OK else 'FAILED'} (rc {d.get('rc')}): "
                f"{d.get('result') or d.get('failed')}")
            for ln in d.get("problems") or []:
                say(f"  problem: {ln}")
            return int(d.get("rc", L.EXIT_LEG_FAILED))
        if abort["requested"] and not abort_sent:
            (pdir / "joins" / f"{cfg.tag}.{ack.get('attempt')}.abort").write_text(
                json.dumps({"tag": cfg.tag, "attempt": ack.get("attempt"), "pid": os.getpid(),
                            "utc": dlx.now()}))
            abort_sent = True
        if not host_alive(reg_view):
            time.sleep(1.0)
            if not done_f.exists():
                return _host_died(cfg, reg_view, say, "host gone before finishing this leg", jf=jf)
        if abort["requested"] and time.time() - abort["requested"] > 2700:
            return _host_died(cfg, reg_view, say, "no DONE 45 min after the abort request", jf=jf)
        if time.time() - last_note > 300:
            last_note = time.time()
            try:
                st = json.loads((pdir / "status.json").read_text())
                lg = (st.get("legs") or {}).get(cfg.tag) or {}
                say(f"units {lg.get('ended')}/{lg.get('units')} ended ({lg.get('running')} running, "
                    f"{lg.get('queued')} queued); machines {st.get('machines_ready')} ready, caps "
                    f"{[m.get('cap') for m in (st.get('machines') or {}).values()]}")
            except (OSError, ValueError):
                pass
        time.sleep(poll_s)


def _host_died(cfg, reg_view, say, when, jf=None) -> int:
    """The follower gives up on its leg. With its join file: under the join's lock, so the host can neither accept
    the join nor finish the leg OK afterwards (the cancel marker; see JOIN HAND-OVER)."""
    if jf is None:
        return _give_up_locked(cfg, None, reg_view, say, when)
    with _join_lock(jf):
        return _give_up_locked(cfg, jf, reg_view, say, when)


def _give_up_locked(cfg, jf, reg_view, say, when) -> int:
    """(caller holds the join lock) cancel marker, then FAILED unless the leg's result already exists."""
    msg = (f"pool host pid {reg_view['host_pid']}: the follower gave up on this leg ({when}; pool dir "
           f"{reg_view['pool_dir']}); see its host.log / host.out. The join is cancelled, so that host can no longer "
           f"run or finish this leg. A dead host's machine is deleted by the run's NEXT pool host before it creates "
           f"one (or by `ssb_leg_daytona_loop.py cleanup --run-dir {cfg.run_dir}`, or its TTL).")
    if cfg.result_path.exists():
        say(f"giving up ({when}), but {cfg.result_path.name} already exists: LEG OK")
        return L.EXIT_OK
    if jf is not None:
        L._atomic_write(_cancel_path(jf), {"tag": cfg.tag, "follower_pid": os.getpid(), "why": when,
                                           "utc": dlx.now()})
    say("FAILED: " + msg)
    if not cfg.failed_path.exists():
        L._atomic_write(cfg.failed_path, {"INCOMPLETE": True, "tag": cfg.tag, "key": cfg.key,
                                          "problems": [msg], "pool_dir": reg_view["pool_dir"],
                                          "cancelled_join": str(_cancel_path(jf)) if jf is not None else None,
                                          "written_utc": dlx.now()})
    return L.EXIT_LEG_FAILED


# ───────────────────────────────────────────────────────────────────────────── planning a pool leg
def seed_index(cache_dir: Path) -> tuple:
    """(CACHE_ENV marker or None, {task: [record paths]}) in ONE pass over the per-epoch cache dir -- the same
    selection as seed_files (records below the root, by their question's task_id), without re-reading every
    record once per task."""
    if not Path(cache_dir).is_dir():
        return None, {}
    marker = Path(cache_dir) / "CACHE_ENV"
    idx: dict = {}
    for p in sorted(Path(cache_dir).rglob("*.json")):
        if p.parent == Path(cache_dir):
            continue
        t = L._record_task(p)
        if t is not None:
            idx.setdefault(t, []).append(p)
    return (marker if marker.is_file() else None), idx


def plan_units(cfg, idx: dict, p: dict) -> tuple:
    """(plan dict in LegRun's plan schema with shards = units, [(name, task, need, heavy, seeds)]).
    One unit per task, named sNN in frozen split order, so <store>/<tag>/sNN is a shard dir to every reader
    (outcomes.discover_attempts globs s[0-9]*)."""
    marker, sidx = seed_index(cfg.cache_dir)
    rows = []
    for i, t in enumerate(cfg.tasks):
        recs = sidx.get(t, [])
        seeds = ([marker] if marker else []) + recs
        size = max(idx[t]["input"].stat().st_size, idx[t]["golden"].stat().st_size)
        rows.append((f"s{i + 1:02d}", t, max(0, cfg.n_samples - len(recs)), size >= base.HEAVY_BYTES, seeds))
    plan = {"shards": {n: [t] for n, t, *_ in rows}, "heavy": [t for _n, t, _need, h, _s in rows if h],
            "n_tasks": len(cfg.tasks), "n_samples": cfg.n_samples, "episodes_per_shard_max": cfg.n_samples,
            "concurrency": cfg.concurrency, "max_heavy_per_shard": int(p["pool_max_heavy_units"]),
            "pool": {"units": len(rows), "attempts_to_run": sum(r[2] for r in rows),
                     "seeded_records": sum(len(r[4]) - (1 if marker else 0) for r in rows),
                     "settings": dict(p)}}
    return plan, rows


def dry_run_pool(cfg, idx: dict) -> dict:
    """The pool part of a DRYRUN record: the unit plan, the per-unit seeds and the machine layout."""
    p = L.pool_settings(cfg.rt)
    gc = grader_cap(cfg, p)
    bad = layout_problem(p, gc)
    if bad:
        raise L.Refused(bad)
    plan, rows = plan_units(cfg, idx, p)
    pl = evolver_placement(cfg.run_dir)
    return {"plan": plan, "seed_files": {n: [str(s) for s in seeds] for n, _t, _need, _h, seeds in rows},
            "pool": {"enabled": True, "settings": p, "pool_root": str(pool_root(cfg)),
                     "evolver_placement": pl, "machine_domain_allow_list": allow_domains(pl),
                     "machines_max": min(int(p["pool_sandboxes"]), len(rows)),
                     "cap_policy": cap_note(p), "admission_policy": p["pool_order"],
                     "machine_ttl_minutes": L.pool_ttl(cfg.rt, p)[0],
                     "grader_cap": gc, "leg_epoch": leg_epoch(cfg.tag), "no_train_sibling": no_train_sibling(cfg),
                     "keep_idle_s": p["pool_keep_idle_s"],
                     "note": "ONE long-lived pool per RUN serves this leg and every other leg of the run (every "
                             "epoch; newest train first, then val oldest epoch first, then test); nothing was "
                             "created, uploaded, spent or launched"}}


# ───────────────────────────────────────────────────────────────────────────── units, legs
class Unit:
    """One task of one leg: all its samples, one runner invocation. Quacks like a Shard for LegRun._finish
    (.name .dir .summary .sid)."""

    def __init__(self, leg: "PoolLeg", name: str, task: str, need: int, heavy: bool, seeds: list):
        self.leg, self.name, self.task, self.need, self.heavy, self.seeds = leg, name, task, int(need), heavy, seeds
        self.tag = leg.cfg.tag
        self.dir = leg.cfg.store / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.extra_seeds: list = []    # [(file, path relative to the cache root)]: records salvaged from a lost try
        self.seeded_records = sum(1 for s in seeds if s.name != "CACHE_ENV")
        self.state = "queued"          # queued -> assigned -> running -> syncing -> ended
        self.machine = None
        self.records = 0               # NEW records this unit's runner wrote (seeded ones excluded)
        self.rc = None
        self.pid = None
        self.dead_polls = 0
        self.t_launch = None
        self.kill_reason = None
        self.killed = False
        self.cap_at_dispatch = None
        self.stats: dict = {}
        self.tries = 1                 # this unit's launch number in this leg attempt (a lost machine -> +1)
        self.lost_requeues = 0
        self.avoid: set = set()        # machines this unit was lost on
        self.log_off = 0               # bytes of its runner.log the polls have scanned for grader failures
        self.gfail_seen = 0            # grader-failure lines those polls counted (live)
        self.t_dispatch = None         # when the host gave it a spot on a machine (admission)
        self.t_end = None              # when it ENDED here (outputs synced home, or never launched / lost)
        self.summary: dict = {"shard": name, "tag": self.tag, "tasks": [task],
                              "n_rollouts": leg.cfg.n_samples, "pool_unit": True, "heavy": heavy,
                              "need": self.need, "bundle": {"seeded_cache_files": len(seeds)}}

    @property
    def key(self) -> str:
        return f"{self.tag}/{self.name}"

    @property
    def sid(self):
        return self.machine.sid if self.machine is not None else None

    @property
    def sandbox_dir(self) -> str:
        """A FRESH dir per launch: leg attempt + try in the name, so a relaunched leg, or a unit
        requeued after a lost machine, can never sync back an earlier launch's result."""
        return f"{POOL_SANDBOX_DIR}/units/{self.tag}/a{self.leg.attempt}/{self.name}.t{self.tries}"

    def seed_pairs(self) -> list:
        """[(local file, path relative to the cache root)] this launch is seeded with."""
        return ([(p, p.relative_to(self.leg.cfg.cache_dir).as_posix()) for p in self.seeds]
                + list(self.extra_seeds))

    def cache_private(self) -> str:
        c = str(self.leg.cfg.cache_dir)
        pre = SANDBOX_REPO + "/.cache/"
        return f"{self.sandbox_dir}/dotcache/{c[len(pre):]}" if c.startswith(pre) else f"{self.sandbox_dir}/cache_ext"

    def tasks_private(self) -> str:
        return f"{self.sandbox_dir}/dotcache/ssbv2_{self.tag}_tasks"

    def binds(self) -> list:
        """[(canonical path, private dir)] -- see pool_unit_launch.sh."""
        U = self.sandbox_dir
        b = [(f"{SANDBOX_REPO}/.cache", f"{U}/dotcache"),
             (f"{SANDBOX_REPO}/outputs/ssbv2_base30", f"{U}/result")]
        c = str(self.leg.cfg.cache_dir)
        if not c.startswith(SANDBOX_REPO + "/.cache/"):
            b.append((c, f"{U}/cache_ext"))
        return b

    def remaining(self) -> int:
        return max(0, self.need - self.records) if self.state in ("assigned", "running") else 0

    def save(self) -> None:
        try:
            (self.dir / "summary.json").write_text(json.dumps(self.summary, indent=1, default=str))
        except OSError:
            pass


class PoolLeg:
    def __init__(self, host: "PoolHost", cfg, attempt: int, req: dict):
        self.host, self.cfg, self.attempt, self.req = host, cfg, attempt, req
        # active -> units_done (every unit ended; waiting for its machines' log sync) -> finished (result or FAILED
        # written, DONE file for its follower). aborting: its follower asked to abort (its units are killed).
        self.state = "active"
        self.abort_reason = None
        self.fail_reason = None        # e.g. its spend cap: no unit admitted any more, running ones FINISH, then FAILED
        self.units: list = []
        self.plan = None
        self.legrun = None
        self.rc = None
        self.done_file = host.pdir / "legs" / f"{cfg.tag}.{attempt}.done.json"
        self.t_accepted = time.time()  # train_first: a val leg's wait for its train sibling runs from here
        self.t_queued = None           # when its units entered the pool's queue
        self.epoch = leg_epoch(cfg.tag)
        self.no_train_sibling = bool((req or {}).get("no_train_sibling"))
        self.skills_sha = None         # content sha of its skills dir at acceptance (machines key skills by it)
        self.src_manifest = None       # the Mac's src/ tree at acceptance (vs each machine's snapshot)
        self.t_units_done = None
        self.log_sync_wait: dict = {}  # machine name -> the log-sync request this leg's result waits for
        self.log_sync_problems: dict = {}   # machine name -> why its logs could not be synced / scanned
        self.spend_attributed = 0.0    # its share of the key usage (by fresh responses), accumulated per spend check

    def prepare(self, idx: dict, p: dict) -> None:
        cfg = self.cfg
        lr = L.LegRun(cfg, client=self.host.client, key=self.host.key, gen_lookup=self.host.gen_lookup, idx=idx)
        lr._prepare_store()                         # earlier attempt -> <store>._prev_<stamp>, FAILED aside
        lr.logf = open(cfg.store / "leg.log", "a")
        lr.spend = self.host.spend
        self.legrun = lr
        plan, rows = plan_units(cfg, idx, p)
        plan.update({"tag": cfg.tag, "key": cfg.key, "split_file": str(cfg.split_file),
                     "split_sha256": cfg.split_sha256, "created": dlx.now(),
                     "pool_dir": str(self.host.pdir), "attempt": self.attempt})
        (cfg.store / "plan.json").write_text(json.dumps(plan, indent=1, default=str))
        self.plan = plan
        self.units = [Unit(self, n, t, need, h, seeds) for n, t, need, h, seeds in rows]
        lr.log(f"leg {cfg.tag}: {len(cfg.tasks)} tasks x {cfg.n_samples} as {len(self.units)} POOL units "
               f"({plan['pool']['attempts_to_run']} attempts to run, {plan['pool']['seeded_records']} seeded) in "
               f"pool {self.host.pdir.name}; skills {cfg.skills_dir or 'NONE (base leg)'}; cache {cfg.cache_dir}")

    def counts(self) -> dict:
        c = Counter(u.state for u in self.units)
        return {"units": len(self.units), "ended": c["ended"], "running": c["running"] + c["syncing"],
                "assigned": c["assigned"], "queued": c["queued"], "state": self.state,
                "attempt": self.attempt, "epoch": self.epoch, "key": self.cfg.key,
                "fail_reason": self.fail_reason, "rc": self.rc}

    def metered_usd(self) -> float:
        """total_cost_usd of this leg's NEW cache records so far (units that ended here, and lost tries)."""
        tot = sum(float(u.summary.get("fresh_cost_usd") or 0) for u in self.units)
        return tot + sum(float(a.get("fresh_cost_usd") or 0) for u in self.units
                         for a in u.summary.get("lost_attempts") or [])


# ───────────────────────────────────────────────────────────────────────────── a pool machine
class Machine(L.Shard):
    """One pool sandbox. Creation, identity/egress probes, cgroup delegation, shim + grading self-tests are the
    Shard's own methods (same commands, same refusals); upload, launch, poll and sync are per UNIT."""

    def __init__(self, pool: "PoolHost", name: str, task0: str):
        super().__init__(pool, name, [task0])
        self.pool = pool
        self.ramp = None
        self.units: dict = {}
        self.inbox: list = []
        self.ready = threading.Event()
        self.release = threading.Event()
        self.wake = threading.Event()
        self.failed = None
        self.deleted = None
        self.tasks_on: set = set()
        self.skills_on: set = set()
        self.prev_cpu = None
        self.prev_watch = None
        self.live_timeouts_total = 0
        self.sig: dict = {}
        self.sig_max: dict = {}
        self.exact_timeouts_pending = 0
        self.exact_timeouts_total = 0
        self.grader_fail_pending = 0          # grader-failure lines not yet fed to the ramp
        self.grader_fail_total = 0
        self.watcher_stale_polls = 0
        self.long_cmd_polls = 0
        self.max_inflight = 0
        self.max_units = 0
        self.max_same_leg_units = 0         # most units of ONE leg running here at once (isolation evidence)
        self.units_run: list = []
        self.units_lost: list = []
        self.secret_hits: list = []
        self.t0 = time.time()
        self.t_created = None
        self.t_ready = None
        self.t_end = None
        self.poll_failures = 0
        # the LONG-LIVED pool
        self.skills_sha: dict = {}          # skills dir path on this machine -> the content sha it holds there
        self.src_manifest = None            # the src/ tree bundled at creation (src_manifest())
        self.log_sync_req = 0               # log syncs asked for by the host (a leg's result waits for one)
        self.log_sync_done = 0              # requests served (successfully or given up)
        self.log_sync_ok_upto = 0           # requests served successfully
        self.log_sync_tries = 0
        self.log_sync_utc = None
        self.log_sync_error = None
        self.disk_free_gib = None
        self.units_removed = 0              # unit dirs removed from the machine once their outputs were home
        # the egress allow list this machine is CREATED with (the host's, fixed at its init), and
        # the hashes of the files its bundle carried (MACHINE_CODE; a unit's runtime_env names these)
        self.allow_csv = str(getattr(pool, "allow_csv", None) or L.ALLOW_DOMAINS)
        self.code_snapshot: dict = {}

    def allow_list(self) -> str:
        """The Daytona domain allow list this machine is created with (Shard._create_once reads it): openrouter.ai,
        plus api.anthropic.com when the run's evolver runs in a container here (evolver_daytona.placement
        "pool_container"; PoolHost.allow_csv)."""
        return self.allow_csv

    def egress(self):
        """The egress probe of a POOL machine (replaces the shard's): openrouter.ai reachable; api.anthropic.com
        reachable iff it is on this machine's allow list, REFUSED otherwise; every other host refused."""
        rc, out = self.sh("egress", POOL_EGRESS_CMD, timeout=180)
        m = re.search(r"^EGRESS (.*)$", out, re.M)
        if not m:
            raise L.Abort("egress probe failed: " + out[-800:])
        eg = json.loads(m.group(1))
        allowed = {x.strip() for x in self.allow_csv.split(",") if x.strip()}
        must = sorted(k for k, host in EGRESS_HOST.items() if host in allowed)
        self.summary["egress"] = {"domain_allow_list": self.allow_csv, "probe": eg, "must_reach": must,
                                  "evolver_placement": getattr(self.pool, "placement", None)}
        self.log("egress " + json.dumps(eg))
        for k in EGRESS_HOST:
            if k not in eg:
                raise L.Abort(f"the egress probe did not test {EGRESS_HOST[k]}: {sorted(eg)}")
        down = [EGRESS_HOST[k] for k in must if not (eg.get(k) or {}).get("ok")]
        if down:
            raise L.Abort(f"{', '.join(down)} NOT reachable from the sandbox (allow list {self.allow_csv})")
        leaked = [k for k in eg if k not in must and (eg.get(k) or {}).get("ok")]
        if leaked:
            raise L.Abort(f"egress allow list NOT enforced: reachable {leaked} (allow list {self.allow_csv})")

    # -- accounting (callers hold pool.lock)
    def active(self) -> list:
        return [u for u in self.units.values() if u.state in ("assigned", "running", "syncing")]

    def active_units(self) -> int:
        return len(self.active())

    def heavy_units(self) -> int:
        return sum(1 for u in self.active() if u.heavy)

    def inflight(self) -> int:
        return sum(u.remaining() for u in self.units.values())

    def check_granted_ttl(self) -> None:
        """Daytona caps a sandbox's TTL at the org's maximum sandbox lifespan, and retire_at / ttl_expires are planned
        from the REQUESTED TTL: the machine's readback (created_at -> auto_destroy_at) must grant it, or the machine
        FAILS here, before any unit or lease can rely on it. Readback without those fields (an SDK or fake that omits them) is recorded
        as unverified, not refused."""
        import datetime as _dt
        rb = self.summary.get("readback") or {}
        want = float(self.cfg.rt.get("ttl_minutes") or 0)
        ca, da = rb.get("created_at"), rb.get("auto_destroy_at")
        if ca is None and da is None:
            self.summary["granted_ttl_min"] = None
            self.log(f"WARNING: Daytona readback has no created_at/auto_destroy_at: the {want:.0f}-min TTL is unverified")
            return
        try:
            ts = lambda v: _dt.datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
            granted = (ts(da) - ts(ca)) / 60
        except (TypeError, ValueError) as e:
            raise L.Abort(f"Daytona readback created_at={ca!r} auto_destroy_at={da!r} is unreadable ({e}): cannot "
                          f"confirm the {want:.0f}-min TTL the pool plans with")
        self.summary["granted_ttl_min"] = round(granted, 2)
        if want and granted < want - 1:
            raise L.Abort(f"Daytona granted a {granted:.1f}-min TTL (auto_destroy_at {da}), less than the {want:.0f} "
                          f"min requested (the org's maximum sandbox lifespan?): retire_at / ttl_expires would be "
                          f"past the machine's destruction")
        self.log(f"TTL granted {granted:.1f} min (requested {want:.0f}; auto_destroy_at {da})")

    def retire_at(self):
        """When this machine stops taking units: creation + TTL - pool_ttl_margin_min (None before creation)."""
        ttl = float(self.cfg.rt.get("ttl_minutes") or 0) * 60
        if not (self.t_created and ttl):
            return None
        return self.t_created + ttl - float(self.pool.p["pool_ttl_margin_min"]) * 60

    def ttl_expires(self):
        ttl = float(self.cfg.rt.get("ttl_minutes") or 0) * 60
        return self.t_created + ttl if (self.t_created and ttl) else None

    def retiring(self) -> bool:
        """Older than ttl_minutes - pool_ttl_margin_min: takes no new unit (see POOL_DEFAULTS). Also when its disk ran
        low (a long-lived machine: < DISK_RETIRE_GIB free)."""
        ra = self.retire_at()
        if ra is not None and time.time() > ra:
            return True
        return self.disk_free_gib is not None and self.disk_free_gib < DISK_RETIRE_GIB

    def usable(self) -> bool:
        return (self.ready.is_set() and not self.release.is_set() and not self.done.is_set() and not self.failed
                and not self.retiring())

    def state_name(self) -> str:
        if self.done.is_set():
            return "deleted" if self.deleted else ("failed" if self.failed else "gone")
        if self.failed:
            return "failed"
        if self.release.is_set():
            return "released"
        if not self.ready.is_set():
            return "starting"
        return "retiring" if self.retiring() else "ready"

    def public_record(self, leased_by=()) -> dict:
        """This machine as machine.json shows it (Registry)."""
        return {"sandbox_id": self.sid, "sandbox_name": self.summary.get("sandbox_name"), "state": self.state_name(),
                "created_ts": self.t_created, "ready_ts": self.t_ready,
                "ttl_minutes": self.cfg.rt.get("ttl_minutes"), "ttl_expires_ts": self.ttl_expires(),
                "retire_at_ts": self.retire_at(),
                "domain_allow_list": [x.strip() for x in self.allow_csv.split(",") if x.strip()],
                "domain_allow_list_readback": (self.summary.get("readback") or {}).get("domain_allow_list"),
                "units_running": len(self.active()), "inflight": self.inflight(),
                "cap": self.ramp.cap if self.ramp else None, "disk_free_gib": self.disk_free_gib,
                "leased_by": list(leased_by)}

    def request_log_sync(self) -> int:
        """(caller holds pool.lock) Ask this machine's thread to sync its logs home and scan them for the key; returns
        the request number a leg's result waits for (log_sync_done >= it)."""
        self.log_sync_req += 1
        self.wake.set()
        return self.log_sync_req

    # -- setup
    def upload_machine(self):
        b = self.dir / "bundle.tgz"
        # the pool scripts as the HOST loaded them at its start (a machine created later -- a TTL or a
        # lost-machine replacement -- never gets pool scripts the running host was not started with)
        sdir = self.dir / "pool_scripts"
        sdir.mkdir(parents=True, exist_ok=True)
        extra = []
        for f in POOL_SCRIPTS:
            data = (getattr(self.pool, "pool_script_bytes", None) or {}).get(f)
            if data is None:
                data = (HERE / f).read_bytes()
            (sdir / f).write_bytes(data)
            extra.append((sdir / f, f"opt/pool/{f}"))
        # the LONG-LIVED pool: a machine outlives the leg whose cfg the pool view copies, so it gets NO skills dir here
        # (a later replacement must not depend on an old epoch's skills dir still being on disk); every skill leg's
        # first unit brings its own, keyed by content (build_unit_tar)
        mcfg = copy.copy(self.cfg)
        mcfg.skills_dir = None
        self.src_manifest = src_manifest()
        code0 = code_manifest(MACHINE_CODE.values())
        man = L.build_bundle(mcfg, self.tasks, L.leg_env_text(mcfg), b, [], idx=self.pool.idx, extra=extra)
        # what the bundle carries = these hashes (a unit's runtime_env names them): a MACHINE_CODE file edited WHILE
        # the bundle was built would make them lie -> the machine fails (a replacement is built from the files as they
        # are then). Any other src/ file changed meanwhile is only named in the snapshot record.
        drift = sorted(k for k, v in code_manifest(MACHINE_CODE.values()).items() if code0.get(k) != v)
        if drift:
            raise L.Abort(f"file(s) changed while this machine's bundle was built: {drift}")
        src_after = src_manifest()
        src_moved = sorted(k for k in set(src_after) | set(self.src_manifest)
                           if src_after.get(k) != self.src_manifest.get(k))
        self.code_snapshot = {**code0, **{f"scripts/daytona/{f}": hashlib.sha256((sdir / f).read_bytes()).hexdigest()
                                          for f in POOL_SCRIPTS}}
        man["src_tree_sha256"] = manifest_sha(self.src_manifest)
        man["code_sha256"] = dict(self.code_snapshot)
        self.summary["bundle"] = man
        self.summary["src_snapshot"] = {"tree_sha256": man["src_tree_sha256"], "files": len(self.src_manifest),
                                        "bundled_utc": dlx.now(), "changed_while_bundling": src_moved[:40]}
        t = time.time()
        self.sb.fs.upload_file(str(b), "/tmp/bundle.tgz", timeout=1800)
        self.log(f"uploaded machine bundle {man['bundle_bytes'] / 1e6:.1f} MB ({man['files']} files) in "
                 f"{time.time() - t:.0f}s")
        b.unlink()
        self.sh("unpack", "tar xzf /tmp/bundle.tgz -C / && rm /tmp/bundle.tgz && "
                "install -m 0755 /opt/leg/docker_shim.py /usr/local/bin/docker && "
                "mkdir -p /opt/leg/out /var/log/evoshim /var/lib/evoshim/state /var/lib/evoshim/priv "
                "/opt/pool/units /opt/pool/out && "
                "which docker && sha256sum /usr/local/bin/docker", timeout=600, check=True)
        self.tasks_on.update(self.tasks)

    def _check_skills(self, sd: Path, want_sha=None):
        """The skills dir as the machine holds it: the SKILL.md count (as a shard checks) and, with `want_sha`, the
        whole tree's content sha computed IN the machine (SKILLS_TREE_PY = skills_fingerprint's hash) -- a long-lived
        machine may hold an older content at the same path (a redone epoch)."""
        rc, out = self.sh("skills", f"ls {shlex.quote(str(sd))}/*/SKILL.md | wc -l", timeout=60)
        got = (out.strip().splitlines() or ["0"])[-1].strip()
        want = str(len(L.skill_mds(sd)))
        self.summary.setdefault("skills_in_sandbox", {})[str(sd)] = got
        if got != want:
            raise L.Abort(f"skills upload incomplete for {sd}: {got} SKILL.md in the sandbox, {want} local")
        if want_sha:
            rc, out = self.sh("skills_sha", f"/opt/evoskill/venv/bin/python -c {shlex.quote(SKILLS_TREE_PY)} "
                              f"{shlex.quote(str(sd))}", timeout=300)
            m = re.search(r"SKILLS_SHA=([0-9a-f]{64})", out)
            have = m.group(1) if m else None
            self.summary.setdefault("skills_sha_in_sandbox", {})[str(sd)] = have
            if have != want_sha:
                raise L.Abort(f"skills dir {sd} on the machine hashes to {str(have)[:12]}, the leg's promoted set to "
                              f"{want_sha[:12]} (changed since the leg joined, or an incomplete upload)")

    def start_sampler(self):
        """The machine's resource sampler (the shards' sandbox_sampler.py, unchanged) and the solver-command
        watcher (pool_cmdwatch.py, live bash timeouts), one each per machine; neither holds the key."""
        py = "/opt/evoskill/venv/bin/python"
        rc, out = self.sh("sampler", f"mkdir -p {POOL_SANDBOX_DIR}/out && (setsid nohup env -u OPENROUTER_API_KEY "
                          f"{py} /opt/leg/sandbox_sampler.py {POOL_SANDBOX_DIR}/out/samples.jsonl 5 "
                          f"> {POOL_SANDBOX_DIR}/out/sampler.log 2>&1 < /dev/null &) ; (setsid nohup env -u "
                          f"OPENROUTER_API_KEY {py} {POOL_SANDBOX_DIR}/{WATCH_SCRIPT} {POOL_SANDBOX_DIR}/out/cmdwatch.json "
                          f"{POOL_SANDBOX_DIR}/out/cmdwatch.jsonl {CMDWATCH_EVERY_S} {self.pool.bash_timeout} > "
                          f"{POOL_SANDBOX_DIR}/out/cmdwatch.log 2>&1 < /dev/null &) ; sleep 1; echo sampler-started",
                          timeout=60)
        self.summary["sampler"] = out.strip()[-200:]

    # -- one unit
    def unit_runtime_env(self, u: Unit) -> dict:
        rt = self.runtime_env_record(u.leg.cfg, shard=u.name, tasks=[u.task], seeded=len(u.seed_pairs()))
        rt["driver"] = "scripts/daytona/ssb_leg_pool.py (pool unit) via scripts/daytona/ssb_leg_daytona_loop.py"
        rt["pool"] = {"machine": self.name, "unit": u.key, "pool_dir": str(self.pool.pdir),
                      "try": u.tries, "salvaged_seeds": len(u.extra_seeds),
                      "cap_at_dispatch": self.ramp.cap if self.ramp else None,
                      "unit_launcher_sha256": L.sha256_file(HERE / UNIT_LAUNCH),
                      "private_binds": u.binds(),
                      # the machine's src/ is the snapshot bundled at ITS creation, which on a long-lived machine
                      # can predate this leg: both trees are named (the pinned files are held to their pins anyway)
                      "machine_src_tree_sha256": manifest_sha(self.src_manifest),
                      "leg_src_tree_sha256_at_join": manifest_sha(getattr(u.leg, "src_manifest", None))}
        # runtime_env_record hashed the MAC's files at this launch; the runner runs the MACHINE's copy
        # (bundled at the machine's creation). Name what runs; the merge holds these to the pins (a machine built
        # from a drifted pinned file fails its legs there), and list any Mac file that differs now.
        snap = self.code_snapshot or {}
        mac_now = {}
        for field, rel in MACHINE_CODE.items():
            if snap.get(rel):
                if rt.get(field) != snap[rel]:
                    mac_now[rel] = rt.get(field)
                rt[field] = snap[rel]
        if snap.get(f"scripts/daytona/{UNIT_LAUNCH}"):
            rt["pool"]["unit_launcher_sha256"] = snap[f"scripts/daytona/{UNIT_LAUNCH}"]
        rt["pool"]["code_hashes_from"] = ("the machine's bundle at its creation" if snap else
                                          "the Mac's files at this launch (no machine snapshot)")
        rt["pool"]["mac_files_differing_now"] = mac_now
        return rt

    def unit_env_text(self, u: Unit) -> str:
        U = u.sandbox_dir
        add = {"EVOSKILL_TASK_SUBSET": f"{U}/subset.json", "EVOSKILL_RUNTIME_ENV_FILE": f"{U}/runtime_env.json",
               "EVOSKILL_LLM_STATS_FILE": f"{U}/llm_stats.json"}
        n = self.pool.p.get("pool_gen_check_n")
        if n:
            add["EVOSKILL_GEN_CHECK_N"] = str(int(n))
        return (L.leg_env_text(u.leg.cfg)
                + "# POOL UNIT (ssb_leg_pool.py): this unit's own files; every line above is the leg's leg.env\n"
                + "".join(f"export {k}={shlex.quote(v)}\n" for k, v in add.items()))

    def skills_needed(self, u: Unit):
        """(skills dir to upload with this unit or None, True if the machine holds OTHER content at that path)."""
        sd = u.leg.cfg.skills_dir
        if sd is None:
            return None, False
        want = getattr(u.leg, "skills_sha", None)
        have = self.skills_sha.get(str(sd))
        if str(sd) in self.skills_on and (want is None or have == want):
            return None, False
        return sd, str(sd) in self.skills_on

    def build_unit_tar(self, u: Unit, out: Path) -> dict:
        cfg = u.leg.cfg
        U = u.sandbox_dir.lstrip("/")
        n, nbytes, new_task, new_skills = 0, 0, False, None
        up_skills, replace = self.skills_needed(u)
        with tarfile.open(out, "w:gz") as tf:
            if u.task not in self.tasks_on:
                for k in ("input", "golden"):
                    nbytes += L._add_file(tf, data_overlay.source(self.pool.idx[u.task][k], cfg.data_overlay),
                                          L._mirror(self.pool.idx[u.task][k]))
                    n += 1
                new_task = True
            if up_skills is not None:
                for p in sorted(cfg.skills_dir.rglob("*")):
                    if p.is_symlink():
                        raise L.Abort(f"skills dir holds a symlink {p} -- refusing (it would resolve differently "
                                      f"inside the sandbox)")
                    if p.is_file():
                        nbytes += L._add_file(tf, p, L._mirror(p))
                        n += 1
                new_skills = cfg.skills_dir
            cpriv = u.cache_private().lstrip("/")
            for p, rel in u.seed_pairs():
                L._add_file(tf, p, f"{cpriv}/{rel}")
                n += 1
            rt = self.unit_runtime_env(u)
            (u.dir / "runtime_env.json").write_text(json.dumps(rt, indent=1, default=str))
            for name, data in (("unit.env", self.unit_env_text(u)), ("subset.json", json.dumps([u.task])),
                               ("runtime_env.json", json.dumps(rt, default=str)),
                               ("binds.tsv", "".join(f"{c}\t{p}\n" for c, p in u.binds()))):
                L._add_bytes(tf, f"{U}/{name}", data.encode())
                n += 1
        assert b"OPENROUTER_API_KEY=" not in self.unit_env_text(u).encode()
        return {"files": n, "payload_bytes": nbytes, "bundle_bytes": out.stat().st_size,
                "seeded_cache_files": len(u.seed_pairs()), "salvaged_seeds": len(u.extra_seeds),
                "new_task_files": new_task, "new_skills_dir": str(new_skills) if new_skills else None,
                "replaced_skills_dir": bool(new_skills and replace)}

    def launch_unit(self, u: Unit) -> None:
        q = shlex.quote
        U = u.sandbox_dir
        b = self.dir / f"unit_{u.tag}_{u.name}.tgz"
        man = self.build_unit_tar(u, b)
        remote = f"/tmp/unit_{u.tag}_{u.name}.t{u.tries}.tgz"
        self.sb.fs.upload_file(str(b), remote, timeout=900)
        b.unlink()
        # the machine holds ANOTHER content at this skills path (a redone epoch): it goes first, so nothing of the
        # old set can survive next to the new one (no unit of a leg with the old content can still be active: the
        # host refuses such a join, PoolHost._accept)
        wipe = f"rm -rf {q(man['new_skills_dir'])} && " if man.get("replaced_skills_dir") else ""
        # the unit dir is fresh by name (leg attempt + try); wiped anyway before unpacking
        self.sh("unit_unpack", f"{wipe}rm -rf {q(U)} && tar xzf {q(remote)} -C / && rm -f {q(remote)} && mkdir -p "
                f"{q(U)}/out {q(U)}/result {q(U)}/dotcache {q(u.cache_private())} && echo unpacked", timeout=600,
                check=True)
        self.tasks_on.add(u.task)
        if man["new_skills_dir"]:
            if man.get("replaced_skills_dir"):
                self.log(f"skills dir {man['new_skills_dir']} REPLACED on this machine (content "
                         f"{str(self.skills_sha.get(man['new_skills_dir']))[:12]} -> "
                         f"{str(getattr(u.leg, 'skills_sha', None))[:12]}) for {u.tag}")
            self.skills_on.discard(man["new_skills_dir"])
            self.skills_sha.pop(man["new_skills_dir"], None)
            self._check_skills(Path(man["new_skills_dir"]), getattr(u.leg, "skills_sha", None))
            self.skills_on.add(man["new_skills_dir"])
            self.skills_sha[man["new_skills_dir"]] = getattr(u.leg, "skills_sha", None)
        rc, out = self.sh("unit_launch", f"(setsid nohup bash {POOL_SANDBOX_DIR}/{UNIT_LAUNCH} {q(U)} > "
                          f"{q(U)}/out/launch.log 2>&1 < /dev/null &) ; sleep 2; echo PID=$(cat {q(U)}/out/pid "
                          f"2>/dev/null)", timeout=120, env={"OPENROUTER_API_KEY": self.pool.key})
        m = re.search(r"PID=(\d+)", out)
        with self.pool.lock:
            u.pid = m.group(1) if m else None
            u.t_launch = time.time()
            if u.state == "assigned":
                u.state = "running"
            u.summary.update(sandbox_id=self.sid, sandbox_name=self.summary.get("sandbox_name"), machine=self.name,
                             launched_utc=dlx.now(), unit_bundle=man, readback=self.summary.get("readback"),
                             identity=self.summary.get("identity"), egress=self.summary.get("egress"),
                             grade_selftest=self.summary.get("grade_selftest"),
                             cap_at_dispatch=getattr(u, "cap_at_dispatch", None), pid=u.pid)
            u.summary["bundle"] = {"seeded_cache_files": len(u.seed_pairs()), **man}
            running = [x for x in self.units.values() if x.state in ("running", "syncing")]
            same = Counter(x.tag for x in running)
            self.max_units = max(self.max_units, len(running))
            self.max_same_leg_units = max(self.max_same_leg_units, max(same.values() or [0]))
        (u.dir / "leg.env").write_text(self.unit_env_text(u))
        u.save()
        self.log(f"launched {u.key} ({u.task}, need {u.need}, seeded {u.seeded_records}, try {u.tries}) pid {u.pid} "
                 f"cap {self.ramp.cap} in-flight {self.inflight()} running units {len(running)} "
                 f"(same leg {same[u.tag]})")

    def kill_unit(self, u: Unit) -> None:
        q = shlex.quote
        U = u.sandbox_dir
        self.sh("unit_kill", f"P=$(cat {q(U)}/out/pid 2>/dev/null); if [ -n \"$P\" ]; then kill -TERM \"$P\" "
                f"2>/dev/null; sleep 10; kill -KILL \"$P\" 2>/dev/null; for c in /var/lib/evoshim/state/*.json; do "
                f"[ -e \"$c\" ] && grep -q \"evoskill.owner_pid=$P\\\"\" \"$c\" && docker rm -f "
                f"\"$(basename \"$c\" .json)\" >/dev/null 2>&1; done; fi; echo killed", timeout=180)
        with self.pool.lock:
            u.killed = True
            u.summary["aborted"] = u.kill_reason
        self.log(f"KILLED {u.key}: {u.kill_reason}")

    def signals(self, data: dict) -> dict:
        rows = data.get("samples") or []
        cores, age = None, None
        if rows:
            last = float(rows[-1].get("ts") or 0)
            age = round(float(data.get("ts") or time.time()) - last, 1)
            recent = [float(r["cores_busy"]) for r in rows
                      if r.get("cores_busy") is not None and float(r.get("ts") or 0) >= last - 60]
            if len(recent) >= 3 and age <= 60:
                cores = round(sum(recent) / len(recent), 2)
        cs = data.get("cpu_stat") or {}
        thr = None
        if self.prev_cpu and "nr_periods" in cs and "nr_periods" in self.prev_cpu:
            dp = cs["nr_periods"] - self.prev_cpu["nr_periods"]
            if dp > 0:
                thr = round((cs.get("nr_throttled", 0) - self.prev_cpu.get("nr_throttled", 0)) / dp, 3)
        if cs:
            self.prev_cpu = cs
        mem = data.get("mem_bytes")
        cmds = data.get("cmds") or {}
        ages = cmds.get("long") or []                    # running solver commands >= the poll's watch_s
        watch = float(self.pool.p.get("bash_watch_s") or 0)
        long_ = [a for a in ages if watch > 0 and a >= watch]
        cw = data.get("cmdwatch") if isinstance(data.get("cmdwatch"), dict) else None
        live, wage = None, None
        if cw is not None:
            # a watcher state older than CMDWATCH_STALE_S = the watcher is dead or wedged: it
            # counts as NO watcher, so the finished records' exact count becomes the timeout stress signal
            try:
                wage = round(float(data.get("ts") or time.time()) - float(cw.get("ts")), 1)
            except (TypeError, ValueError):
                wage = None
            if wage is None or wage > CMDWATCH_STALE_S:
                self.watcher_stale_polls += 1
                cw = None
        if cw is not None:
            tot = int(cw.get("timeouts") or 0)
            live = max(0, tot - (self.prev_watch if self.prev_watch is not None else 0))
            self.prev_watch = tot
            self.live_timeouts_total += live
        smoke = None
        after = self.pool.p.get("smoke_stress_after_s")
        if after and self.t_ready and time.time() - self.t_ready >= float(after):
            smoke = round(time.time() - self.t_ready, 1)
        disk = data.get("disk") if isinstance(data.get("disk"), dict) else {}
        try:
            dfree = round(float(disk["free_gib"]), 2) if disk.get("free_gib") is not None else None
        except (TypeError, ValueError):
            dfree = None
        if dfree is not None:
            if dfree < DISK_RETIRE_GIB and (self.disk_free_gib is None or self.disk_free_gib >= DISK_RETIRE_GIB):
                self.log(f"DISK LOW: {dfree} GiB free < {DISK_RETIRE_GIB}: this machine takes no new unit (retires)")
            self.disk_free_gib = dfree
        return {"cores_1min": cores, "throttle_ratio": thr,
                "mem_gib": round(mem / 2 ** 30, 2) if isinstance(mem, (int, float)) else None,
                "live_timeouts": live, "watcher": cw is not None, "watcher_age_s": wage,
                "long_cmds": len(long_), "long_max_s": max(long_) if long_ else None,
                "cmds_ge60": sum(1 for a in ages if a >= 60), "solver_cmds": cmds.get("n"), "sampler_age_s": age,
                "smoke_stress": smoke, "disk_free_gib": dfree}

    def poll_once(self) -> list:
        """One poll; updates units + signals + the ramp. Returns the units whose runner has exited."""
        pool = self.pool
        with pool.lock:
            watch = {u.key: {"dir": u.sandbox_dir, "cache": u.cache_private(), "log_off": u.log_off}
                     for u in self.units.values() if u.state == "running"}
        spec = {"units": watch, "watch_s": min(60.0, float(pool.p["bash_watch_s"]) or 60.0),
                "samples": f"{POOL_SANDBOX_DIR}/out/samples.jsonl", "tail": 24,
                "cmdwatch": f"{POOL_SANDBOX_DIR}/out/cmdwatch.json",
                "grader_fail_patterns": list(GRADER_FAILURE_PATTERNS)}
        rc, out = self._sh_retry("pool_poll", f"/opt/evoskill/venv/bin/python {POOL_SANDBOX_DIR}/{POLL_SCRIPT} "
                                 f"{shlex.quote(json.dumps(spec, separators=(',', ':')))}", timeout=120)
        m = re.search(r"^POOLPOLL (.*)$", out, re.M)
        if not m:
            raise L.Abort("pool poll printed no POOLPOLL line: " + out[-600:])
        data = json.loads(m.group(1))
        now = time.time()
        ended = []
        wall = float(pool.p["unit_wall_min"]) * 60
        with pool.lock:
            for key, ud in (data.get("units") or {}).items():
                u = self.units.get(key)
                if u is None or u.state != "running":
                    continue
                u.records = max(0, int(ud.get("records") or 0) - u.seeded_records)
                if isinstance(ud.get("stats"), dict):
                    u.stats = ud["stats"]
                g = int(ud.get("gfail") or 0)                # new grader-failure lines in its runner.log
                if isinstance(ud.get("log_off"), int) and ud["log_off"] >= u.log_off:
                    u.log_off = ud["log_off"]
                if g:
                    u.gfail_seen += g
                    self.grader_fail_pending += g
                    self.grader_fail_total += g
                u.summary["progress"] = {"cache_json_files": u.records + u.seeded_records,
                                         "of": u.leg.cfg.n_samples, "new_records": u.records,
                                         "fresh_responses": (u.stats or {}).get("fresh_responses")}
                if ud.get("rc") is not None:
                    u.rc = str(ud["rc"])
                    u.state = "syncing"
                    ended.append(u)
                    continue
                if u.pid and not ud.get("pid_alive"):
                    u.dead_polls += 1
                    if u.dead_polls >= 2:
                        u.rc = "dead (runner gone, no rc)"
                        u.state = "syncing"
                        ended.append(u)
                        continue
                if u.t_launch and now - u.t_launch > wall and u.kill_reason is None:
                    u.kill_reason = f"unit wall {pool.p['unit_wall_min']} min"
            sig = self.signals(data)
            # the records' exact count is the STRESS signal only where the live watcher is not running (it
            # would count the same timeouts a second time, minutes later); it is always recorded
            if sig.get("watcher"):
                sig["exact_timeouts_seen"] = self.exact_timeouts_pending
            else:
                sig["exact_timeouts"] = self.exact_timeouts_pending
            self.exact_timeouts_pending = 0
            sig["grader_failures"] = self.grader_fail_pending    # live from the polls + the rest at unit end
            self.grader_fail_pending = 0
            if sig.get("cmds_ge60"):
                self.long_cmd_polls += 1
            for k in ("cores_1min", "throttle_ratio", "mem_gib", "long_max_s", "cmds_ge60"):
                if sig.get(k) is not None:
                    self.sig_max[k] = max(self.sig_max.get(k) or 0, sig[k])
            infl = self.inflight()
            self.max_inflight = max(self.max_inflight, infl)
            self.sig = sig
            ch = self.ramp.observe(now, sig, infl, pool.hold_429)
            if ch and ch.get("record_only"):             # FIXED cap: recorded, never acted on
                self.log(f"STRESS RECORDED (fixed cap {ch['cap']} unchanged, in-flight {ch['inflight']}): "
                         f"{'; '.join(ch['reasons'])} | signals {json.dumps(ch['signals'])}")
                pool.log(f"[{self.name}] stress recorded, fixed cap {ch['cap']} unchanged: {'; '.join(ch['reasons'])}")
            elif ch:
                self.log(f"CAP {ch['kind'].upper()} {ch['from']} -> {ch['to']} (ceiling {ch['ceiling']}, in-flight "
                         f"{ch['inflight']}) {'; '.join(ch['reasons']) or 'binding + unstressed for the interval'} | "
                         f"signals {json.dumps(ch['signals'])}")
                pool.log(f"[{self.name}] cap {ch['kind']} {ch['from']}->{ch['to']} ceiling {ch['ceiling']}: "
                         f"{'; '.join(ch['reasons']) or 'binding, unstressed'}")
        return ended

    def sync_unit(self, u: Unit) -> None:
        U = u.sandbox_dir
        items = [(u.cache_private().lstrip("/"), "cache"), (u.tasks_private().lstrip("/"), "tasks"),
                 (f"{U}/result".lstrip("/"), "result"), (f"{U}/out".lstrip("/"), "out"),
                 (f"{U}/runtime_env.json".lstrip("/"), "runtime_env.sandbox.json"),
                 (f"{U}/unit.env".lstrip("/"), "leg.env"), (f"{U}/llm_stats.json".lstrip("/"), "llm_stats.json")]
        synced = True
        try:
            self._pack_download(items, u.dir, f"/tmp/sync_{u.tag}_{u.name}.tgz")
        except Exception as e:  # noqa: BLE001
            synced = False
            u.summary["error"] = f"sync failed: {type(e).__name__}: {e}"
            self.log(f"sync of {u.key} FAILED: {e}")
        self.post_unit(u)
        if synced and not u.summary.get("error"):
            # a LONG-LIVED machine runs hundreds of units: once a unit's outputs are home and verified, its dir (its
            # private cache, tasks dir with every workspace, result, logs) leaves the machine's disk. A unit with any
            # error keeps it (evidence, until the machine goes).
            try:
                self.sh("unit_rm", f"rm -rf {shlex.quote(U)}", timeout=300)
                with self.pool.lock:
                    self.units_removed += 1
            except Exception as e:  # noqa: BLE001
                self.log(f"removing {U} failed (disk only): {e}")

    def _pack_download(self, items: list, dest: Path, remote: str) -> None:
        """Shard.sync_back's pack -> download -> extract, for any item list and destination."""
        xf = " ".join(f"--transform {shlex.quote('s,^' + a + ',' + b + ',')}" for a, b in items)
        xf += " --exclude=.claude --exclude='*.lock'"
        srcs = " ".join(shlex.quote(a) for a, _ in items)
        spec = json.dumps({"out": remote, "items": items})
        self.sh("pack", f"# {spec}\ncd / && tar czf {shlex.quote(remote)} {xf} $(for p in {srcs}; do "
                f"[ -e \"$p\" ] && echo $p; done) && ls -la {shlex.quote(remote)}", timeout=900)
        dest.mkdir(parents=True, exist_ok=True)
        local = dest / "sync.tgz"
        self.sb.fs.download_file(remote, str(local))
        with tarfile.open(local) as tf:
            for mem in tf.getmembers():
                if mem.isfile() or mem.isdir():
                    tf.extract(mem, dest, filter="data")
        local.unlink()
        try:
            self.sh("rmsync", f"rm -f {shlex.quote(remote)}", timeout=60)
        except Exception:  # noqa: BLE001
            pass

    def post_unit(self, u: Unit) -> None:
        key = self.pool.key or ""
        hits, tot, fresh, n, n_to = [], 0.0, 0.0, 0, 0
        seeded = {rel for _p, rel in u.seed_pairs()}
        for p in u.dir.rglob("*"):
            if p.is_file() and key:
                try:
                    if key.encode() in p.read_bytes():
                        hits.append(str(p))
                except OSError:
                    pass
        c = u.dir / "cache"
        for p in (c.rglob("*.json") if c.is_dir() else []):
            if p.parent == c:
                continue
            try:
                raw = p.read_bytes()
                cost = float((json.loads(raw).get("trace") or {}).get("total_cost_usd") or 0)
            except (ValueError, OSError):
                continue
            n += 1
            tot += cost
            if p.relative_to(c).as_posix() not in seeded:
                fresh += cost
                n_to += count_timeouts(raw)
        gfail = 0
        rl = u.dir / "out" / "runner.log"
        if rl.is_file():
            gfail = sum(1 for ln in rl.read_text(errors="replace").splitlines() if GRADER_FAIL_RE.search(ln))
        g_rest = max(0, gfail - u.gfail_seen)            # written after the last poll read its log
        looks = []
        r = u.dir / "result" / f"{u.tag}.json"
        if r.is_file():
            try:
                ids = json.loads(r.read_text()).get("generation_ids_sample") or []
            except ValueError:
                ids = []
            pick = list(dict.fromkeys([ids[0], ids[len(ids) // 2], ids[-1]])) if ids else []
            try:
                looks = [self.pool.gen_lookup(key, g) for g in pick]
            except Exception as e:  # noqa: BLE001
                looks = [{"error": f"{type(e).__name__}: {e}"}]
        with self.pool.lock:
            u.summary.update(secret_hits=hits, cache_records=n, cache_cost_usd=round(tot, 5),
                             fresh_cost_usd=round(fresh, 5), bash_timeouts_exact=n_to,
                             grader_failure_lines=gfail, mac_generation_lookups=looks, runner_rc=u.rc,
                             wall_s=round(time.time() - (u.t_launch or time.time()), 1), synced_utc=dlx.now())
            if u.rc not in ("0",):
                u.summary.setdefault("error", f"runner rc {u.rc}")
            self.exact_timeouts_pending += n_to
            self.exact_timeouts_total += n_to
            self.grader_fail_pending += g_rest
            self.grader_fail_total += g_rest
            u.state, u.t_end = "ended", time.time()
            self.units_run.append(u.key)
        u.save()
        self.log(f"ended {u.key} rc {u.rc}: {n} records ({u.records} new, ${fresh:.4f}), bash timeouts {n_to}, "
                 f"grader-failure lines {gfail} ({g_rest} after the last poll)")

    def sync_machine_logs(self) -> None:
        """The machine's evoshim logs + sampler/watcher output home, then the same KEY SCAN a shard ran over its dir:
        a hit is a problem of every leg whose units ran here (PoolHost.finish_leg)."""
        items = [("var/log/evoshim", "evoshim"), ("opt/pool/out", "out"), ("opt/leg/leg.env", "leg.env")]
        try:
            self._pack_download(items, self.dir, "/tmp/sync_machine.tgz")
        finally:
            self.scan_machine_dir()

    def maybe_sync_logs(self) -> None:
        """(machine thread) Serve the host's log-sync requests: logs home + key scan (sync_machine_logs). A failure
        is retried at the next pass, LOG_SYNC_TRIES times in all; then the request counts as served WITHOUT success
        (log_sync_ok_upto stays below it) and the waiting leg records why."""
        with self.pool.lock:
            want = self.log_sync_req
            if want <= self.log_sync_done:
                return
        try:
            self.sync_machine_logs()
            err = None
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {str(e)[:300]}"
        with self.pool.lock:
            if err is None:
                self.log_sync_done = self.log_sync_ok_upto = want
                self.log_sync_tries, self.log_sync_error, self.log_sync_utc = 0, None, dlx.now()
            else:
                self.log_sync_tries += 1
                self.log_sync_error = err
                if self.log_sync_tries >= LOG_SYNC_TRIES:
                    self.log_sync_done, self.log_sync_tries = want, 0
        self.log(f"log sync #{want}: " + ("OK, key scan " + ("CLEAN" if not self.secret_hits else
                                                              f"{len(self.secret_hits)} HIT(S)")
                                          if err is None else f"FAILED ({err})"))

    def scan_machine_dir(self) -> list:
        key = (self.pool.key or "").encode()
        hits = []
        if key:
            for p in self.dir.rglob("*"):
                if p.is_file():
                    try:
                        if key in p.read_bytes():
                            hits.append(str(p))
                    except OSError:
                        pass
        with self.pool.lock:
            self.secret_hits = hits
            self.summary["secret_hits"] = hits
        if hits:
            self.log(f"KEY FOUND in {len(hits)} synced machine file(s): {hits[:5]}")
        return hits

    # -- the machine's life
    def run(self, key=None):
        pool = self.pool
        try:
            if pool.abort_all.is_set():
                raise L.Abort("aborted before start")
            t_req = time.time()                 # the TTL runs from creation: count from the request (earlier)
            self.create()
            self.t_created = t_req
            self.check_granted_ttl()
            self.identity()
            self.egress()
            self.upload_machine()
            self.cgroups_and_selftest()
            self.start_sampler()
            with pool.lock:
                self.ramp = Ramp(pool.p, time.time())
                self.t_ready = time.time()
                self.summary["ready_utc"] = dlx.now()
                self.ready.set()
            self.log(f"READY: cap {self.ramp.cap}, ceiling {self.ramp.ceiling}")
            pool.log(f"[{self.name}] ready ({self.sid})")
            self.serve()
        except L.Abort as e:
            self.failed = str(e)
            self.summary["error"] = str(e)
            self.log(f"ABORT: {e}")
        except Exception as e:  # noqa: BLE001
            self.failed = f"{type(e).__name__}: {e}"
            self.summary["error"] = self.failed
            self.log("CRASH: " + traceback.format_exc()[-2000:])
        finally:
            self._wind_down()

    def serve(self):
        pool = self.pool
        while True:
            if pool.abort_all.is_set():
                self.log("ABORT requested -- killing every runner on this machine")
                with pool.lock:
                    for u in self.units.values():
                        if u.state in ("assigned", "running") and not u.kill_reason:
                            u.kill_reason = "pool abort: " + (pool.abort_reason or "abort_all")
                try:
                    self.kill_runner()
                except Exception as e:  # noqa: BLE001
                    self.log(f"kill_runner failed: {e}")
                self._drain()
                return
            with pool.lock:
                todo, self.inbox = self.inbox, []
                kills = [u for u in self.units.values()
                         if u.kill_reason and not u.killed and u.state == "running"]
            for u in todo:
                if u.kill_reason:                       # aborted before it ever started
                    with pool.lock:
                        u.state, u.t_end, u.rc = "ended", time.time(), "never launched"
                        u.summary["aborted"] = u.kill_reason
                    u.save()
                    continue
                try:
                    self.launch_unit(u)
                except Exception as e:  # noqa: BLE001
                    with pool.lock:
                        u.state, u.t_end, u.rc = "ended", time.time(), "launch failed"
                        u.summary["error"] = f"launch failed: {type(e).__name__}: {e}"
                    u.save()
                    self.log(f"launch of {u.key} FAILED: {e}")
            for u in kills:
                try:
                    self.kill_unit(u)
                except Exception as e:  # noqa: BLE001
                    self.log(f"kill of {u.key} failed: {e}")
            try:
                ended = self.poll_once()
                self.poll_failures = 0
            except Exception as e:  # noqa: BLE001
                self.poll_failures += 1
                self.log(f"poll failed ({self.poll_failures}): {type(e).__name__}: {str(e)[:300]}")
                if self.poll_failures >= 3:
                    raise L.Abort(f"machine unreachable: {self.poll_failures} polls failed in a row")
                ended = []
            for u in ended:
                self.sync_unit(u)
            self.maybe_sync_logs()                     # a leg ended: its result waits for this machine's logs
            with pool.lock:
                idle = not self.active() and not self.inbox
            if self.release.is_set() and idle:
                return
            self.wake.wait(_poll_s())
            self.wake.clear()

    def _drain(self):
        """After a kill: poll until every running unit has exited (or ~3 min), then sync each one home."""
        t0 = time.time()
        while time.time() - t0 < 180:
            try:
                ended = self.poll_once()
            except Exception as e:  # noqa: BLE001
                self.log(f"drain poll failed: {e}")
                break
            for u in ended:
                self.sync_unit(u)
            with self.pool.lock:
                if not any(u.state == "running" for u in self.units.values()):
                    break
            time.sleep(max(1.0, min(_poll_s(), 10.0)))

    def _salvage(self, u: Unit):
        """Best effort, from a LOST machine that still answers: the unit's private cache, tasks dir, result and logs
        into <unit>/_lost_<machine>_t<try>/ (evidence; its finished records may seed the re-run). None if nothing
        could be fetched."""
        stage = u.dir / f"_lost_{self.name}_t{u.tries}"
        U = u.sandbox_dir
        items = [(u.cache_private().lstrip("/"), "cache"), (u.tasks_private().lstrip("/"), "tasks"),
                 (f"{U}/result".lstrip("/"), "result"), (f"{U}/out".lstrip("/"), "out"),
                 (f"{U}/llm_stats.json".lstrip("/"), "llm_stats.json")]
        try:
            self._pack_download(items, stage, f"/tmp/salvage_{u.tag}_{u.name}.tgz")
            self.log(f"salvaged {u.key} from the lost machine into {stage.name}")
            return stage
        except Exception as e:  # noqa: BLE001
            self.log(f"salvage of {u.key} failed ({type(e).__name__}: {str(e)[:200]}): it re-runs from its seeds")
            return None

    def _wind_down(self):
        pool = self.pool
        with pool.lock:
            # never launched here: back to the global queue (another machine runs them), unless aborting
            back = [u for u in self.units.values() if u.state == "assigned" and not u.kill_reason
                    and not pool.abort_all.is_set()]
            for u in back:
                del self.units[u.key]
                u.state, u.machine = "queued", None
            self.inbox = []
            if back:
                pool._requeue()
                pool.log(f"[{self.name}] {len(back)} unlaunched unit(s) back to the queue: {self.failed or 'stopped'}")
            lost = [u for u in self.units.values() if u.state in ("assigned", "running", "syncing")]
            # A LOST machine: its units run again ONCE, on another machine, instead of failing
            # both legs -- unless the pool or the unit's leg is aborting, or the unit was being killed anyway
            requeue = [u for u in lost if self.failed and not pool.abort_all.is_set() and not u.kill_reason
                       and u.leg.state == "active" and u.lost_requeues < LOST_REQUEUES_MAX]
        salvaged = {}
        for u in lost:
            if u in requeue:
                salvaged[u.key] = self._salvage(u) if (u.state != "assigned" and self.sb is not None) else None
                continue
            if u.state == "running" and self.sb is not None:
                try:
                    self.sync_unit(u)               # whatever reached its cache comes home
                except Exception:  # noqa: BLE001
                    pass
            with pool.lock:
                if u.state != "ended":
                    u.state = "ended"
                    u.t_end = u.t_end or time.time()
                    u.rc = u.rc or "machine lost"
                    u.summary.setdefault("error", f"machine {self.name} lost: {self.failed or 'stopped'}")
            u.save()
        if self.sb is not None:
            try:
                self.sync_machine_logs()
            except Exception as e:  # noqa: BLE001
                self.log(f"machine log sync failed: {e}")
        if self.sid:
            ok = dlx.delete_by_id(pool.client, self.sid, pool.cfg.tag)
            self.deleted = ok
            self.summary["deleted"] = ok
            self.log(f"DELETED {self.sid}: {ok}")
        for u in requeue:
            if self.sid and not self.deleted:
                # its runner may still be spending on a sandbox we could not delete: never run it twice
                with pool.lock:
                    u.state, u.t_end, u.rc = "ended", time.time(), u.rc or "machine lost"
                    u.summary.setdefault("error", f"machine {self.name} lost ({self.failed}) and NOT verified "
                                                  f"deleted: not requeued")
                u.save()
                continue
            pool.requeue_lost(u, self, salvaged.get(u.key))
        self.t_end = time.time()
        self.summary.update(wall_s=round(self.t_end - self.t0, 1), failed=self.failed,
                            ramp=self.ramp.record() if self.ramp else None, units_run=self.units_run)
        with pool.lock:
            for u in self.units.values():
                u.summary["deleted"] = self.deleted if self.sid else None
        for u in list(self.units.values()):
            u.save()
        self.save()
        self.done.set()
        try:
            self.logf.close()
        except Exception:  # noqa: BLE001
            pass

    def record(self) -> dict:
        return {"sandbox_id": self.sid, "sandbox_name": self.summary.get("sandbox_name"),
                "ready_utc": self.summary.get("ready_utc"), "failed": self.failed, "deleted": self.deleted,
                "wall_s": round((self.t_end or time.time()) - self.t0, 1),
                "ready_after_s": round(self.t_ready - self.t0, 1) if self.t_ready else None,
                "max_inflight": self.max_inflight, "units_run": len(self.units_run),
                "max_units_running": self.max_units, "max_same_leg_units": self.max_same_leg_units,
                "units_lost_requeued": list(self.units_lost), "secret_hits": list(self.secret_hits),
                "bash_timeouts_exact": self.exact_timeouts_total, "bash_timeouts_live": self.live_timeouts_total,
                "grader_failure_lines": self.grader_fail_total, "watcher_stale_polls": self.watcher_stale_polls,
                "polls_with_cmd_ge60s": self.long_cmd_polls,
                "created_ts": self.t_created, "retire_at_ts": self.retire_at(), "ttl_expires_ts": self.ttl_expires(),
                "src_snapshot": self.summary.get("src_snapshot"), "skills_sha": dict(self.skills_sha),
                "log_syncs": {"requested": self.log_sync_req, "served": self.log_sync_done,
                              "ok_upto": self.log_sync_ok_upto, "last_ok_utc": self.log_sync_utc,
                              "last_error": self.log_sync_error},
                "disk_free_gib": self.disk_free_gib, "unit_dirs_removed": self.units_removed,
                "signals_max": dict(self.sig_max), "last_signals": dict(self.sig),
                **(self.ramp.record() if self.ramp else {"cap": None, "ceiling": None, "cap_changes": []})}


# ───────────────────────────────────────────────────────────────────────────── spend (the run's pool)
class RunSpendGuard(L.SpendGuard):
    """The RUN cap (org key usage since the run origin) and the account-balance floor, exactly as the leg driver's
    SpendGuard reads them. The per-LEG cap is NOT a key delta here: in a pool that spans the run, the key delta since
    the host started covers every leg of every epoch (a $40 'leg' cap on it would add up
    across epochs). Each leg is measured on its own (PoolHost.leg_spend) and held to leg_cap_usd there; `leg_cap`
    is kept for the record only."""

    def _judge(self, where: str) -> None:
        bal = self.last.get("balance")
        why = []
        if self.run_delta() > self.run_cap:
            why.append(f"run delta ${self.run_delta():.4f} > cap ${self.run_cap}")
        if bal is not None and bal < self.min_balance:
            why.append(f"account balance ${bal} < floor ${self.min_balance}")
        self.log(f"[SPEND] {where} pool key delta=${self.leg_delta():.4f} (every leg of the pool; no cap on it) "
                 f"run=${self.run_delta():.4f}/{self.run_cap} balance={bal}")
        if why and not self.breached.is_set():
            self.reason = "; ".join(why)
            self.log(f"[SPEND] BREACH: {self.reason} -- no unit is admitted any more; running units finish")
            self.breached.set()

    def record(self) -> dict:
        rec = super().record()
        rec["pool_key_delta_usd"] = rec.pop("leg_delta_usd")
        return rec


# ───────────────────────────────────────────────────────────────────────────── the host
class PoolHost:
    """The RUN's pool (one long-lived host serves every leg of the run): joins, the global priority queue, machines,
    the per-machine cap, spend (per leg + run), each leg's result as soon as its units ended, machine.json."""

    def __init__(self, pdir, *, client=None, key=None, fetch_usage=None, gen_lookup=None, idx=None):
        self.pdir = Path(pdir)
        self.reg = Registry(self.pdir.parent)
        self.store = self.pdir / "machines"
        self.store.mkdir(parents=True, exist_ok=True)
        (self.pdir / "legs").mkdir(parents=True, exist_ok=True)
        (self.pdir / "joins").mkdir(parents=True, exist_ok=True)
        self.client, self.key, self.fetch_usage = client, key, fetch_usage
        self.gen_lookup = gen_lookup or base.generation_lookup
        self.idx = idx
        self.lock = threading.RLock()
        self.abort_all = threading.Event()
        self.abort_reason = None
        self.admit_stop = None           # a RUN spend breach: no unit is admitted any more (running ones finish)
        self.legs: dict = {}             # tag -> PoolLeg (its current attempt)
        self.leg_order: list = []
        self.queue: list = []
        self.machines: list = []
        self.threads: list = []
        self.cfg = None                  # the pool's cfg view (first leg's cfg, pool tag + code)
        self.first_cfg = None
        self.p: dict = {}
        self.spend = None
        self.hold_429 = False
        self.api_window: deque = deque()
        self.api_429 = {"holds": 0, "rate_max": 0.0}
        self.problems: list = []
        self.machine_failures: list = []
        self.t0 = time.time()
        self.t_queue_empty = None
        self.t_idle = None               # since when nothing is queued, pending or on any machine (the keep window)
        self.created = 0
        self.retired = 0                 # machines released because they neared their TTL (not failures)
        self.rollovers = 0               # idle machines replaced early (pool_idle_rollover_min)
        self.warm_until = None           # after an idle rollover: a replacement machine is wanted until then
        self.selftest_task = None        # the task a machine created while the queue is empty self-tests with
        self.lost_requeues: list = []
        self.grader = None               # grader_cap(...) once initialised
        self.units_cap = None
        self.bash_timeout, self.bash_timeout_source = 600, "not yet derived"
        self.machine_ttl = None          # (minutes, key) of the pool machines' Daytona TTL
        self.unattributed_usd = 0.0      # key usage in intervals where no unit reported a fresh response
        self._attr_prev_usage = None
        self._resp_seen: dict = {}
        self._train_waiting = False
        self._train_logged = False
        self._machines_pub = (None, 0.0)
        self._machines_gen = 0
        # the code this host LOADED (hashed now; a join is refused once
        # any file differs on disk), the pool scripts as those bytes (every machine gets them), the egress placement
        # and allow list (fixed at _init_pool), the machine-creation guard's state
        self.started_utc = dlx.now()
        self.pool_script_bytes = {f: (HERE / f).read_bytes() for f in POOL_SCRIPTS}
        self.code_at_start = code_manifest(HOST_CODE)
        self.code_at_start.update({f"scripts/daytona/{f}": hashlib.sha256(b).hexdigest()
                                   for f, b in self.pool_script_bytes.items()})
        self.placement = PLACEMENT_OWN
        self.allow_csv = ",".join(allow_domains(PLACEMENT_OWN))
        self.guard: dict = {"block": None, "since": None, "next_ts": 0.0, "checks": 0, "blocked_checks": 0,
                            "deleted": [], "last_logged": None}
        self.reaped: list = []           # sandboxes of this run's pool _init_pool found listed (and deleted)
        # ADMISSION (pool_order): every dispatch decision (with its leg's epoch), the hold reasons of non-train units,
        # the moments the train side was fully dispatched / fully ended, and per leg its join / first dispatch /
        # all dispatched / units done / result times (pool_ramp.admission in every leg's result)
        self.admission: dict = {"policy": None, "decisions": [], "holds": [], "hold_now": None,
                                "first_dispatch_utc": {}, "train_all_dispatched_utc": None,
                                "train_all_ended_utc": None, "train_requeued_after_all_dispatched": 0,
                                "non_train_admitted_while_train_queued": 0,
                                "test_admitted_while_train_or_val_queued": 0, "legs": {}}
        self.logf = open(self.pdir / "host.log", "a")

    def log(self, msg):
        log_line(f"[pool {self.pdir.name}] {msg}", self.logf)

    # -- joins
    def _pending_joins(self) -> list:
        return [f for f in sorted((self.pdir / "joins").glob("*.join.json")) if not _ack_path(f).exists()]

    def process_joins(self) -> None:
        for f in self._pending_joins():
            try:
                with _join_lock(f):                      # JOIN HAND-OVER: claim it, unless cancelled or stale
                    if _ack_path(f).exists():
                        continue
                    if _cancel_path(f).exists():
                        raise L.Refused("its follower cancelled this join (it gave up and recorded the leg FAILED)")
                    req = json.loads(f.read_text())
                    lim = float(req.get("ack_timeout_s") or _ack_timeout_s())
                    age = time.time() - float(req.get("requested_ts") or time.time())
                    if age > lim:
                        raise L.Refused(f"stale join: requested {age:.0f}s ago, past its follower's ack timeout "
                                        f"{lim:.0f}s (the follower may have given up)")
                    _claim_path(f).write_text(json.dumps({"host_pid": os.getpid(), "utc": dlx.now()}))
                known = self.legs.get(req["tag"])
                leg = self._accept(req)
                with _join_lock(f):
                    if _cancel_path(f).exists():         # the follower gave up while this host prepared the leg
                        if leg is not known:
                            self.abort_leg(leg, "its follower cancelled the join before the ack")
                        raise L.Refused("its follower cancelled this join before the ack")
                    ack = {"accepted": True, "attempt": leg.attempt, "done_file": str(leg.done_file),
                           "host_pid": os.getpid(), "pool_dir": str(self.pdir), "utc": dlx.now()}
                    L._atomic_write(_ack_path(f), ack)
                self.log(f"JOIN accepted: {req['tag']} attempt {leg.attempt} ({len(leg.units)} units, epoch "
                         f"{leg.epoch}, {leg.cfg.key}{', VAL-ONLY epoch' if leg.no_train_sibling else ''})")
                continue
            except L.Refused as e:
                ack = {"accepted": False, "reason": str(e), "utc": dlx.now()}
                self.log(f"JOIN refused: {f.name}: {e}")
            except Exception as e:  # noqa: BLE001
                ack = {"accepted": False, "reason": f"{type(e).__name__}: {e}", "utc": dlx.now()}
                self.log(f"JOIN failed: {f.name}: {traceback.format_exc()[-1500:]}")
            L._atomic_write(_ack_path(f), ack)

    def _accept(self, req: dict) -> PoolLeg:
        tag = req["tag"]
        cur = self.legs.get(tag)
        if cur is not None and (cur.state == "aborting" or (cur.state == "units_done" and cur.abort_reason)):
            # a new attempt now would reuse the aborting attempt's unit keys (<tag>/sNN: the machine's
            # dict would drop its still-running units, never killed or synced) and its store dirs
            raise L.Refused(f"attempt {cur.attempt} of {tag} is still being ABORTED ({cur.abort_reason}): its units "
                            f"are being killed and synced home. Relaunch the leg once that attempt is finished (its "
                            f"DONE file {cur.done_file.name}; usually within minutes)")
        if cur is not None and cur.state in ("active", "units_done"):
            return cur                                   # a second follower of a running leg attaches to it
        if self.abort_all.is_set():
            raise L.Refused("the pool is aborting")
        if self.admit_stop:
            raise L.Refused(f"the pool admits no more work: {self.admit_stop}")
        drift = self.code_drift()
        if drift:
            raise L.Refused(
                f"this pool host (pid {os.getpid()}, started {self.started_utc}) RUNS code that has changed on disk "
                f"since it started: {sorted(drift)}. A leg joined now would be recorded against files the host does "
                f"not run. RESTART THE HOST: `ssb_leg_pool.py status --run-dir {req.get('run_dir')}`; once no leg is "
                f"in it (stopping it earlier ends the legs still in it: their followers record them FAILED), "
                f"`kill -TERM {os.getpid()}` (its machine is deleted, the registry closes), then relaunch this leg -- "
                f"the next host loads the new code")
        cfg = L.LegConfig(tag=tag, key=req["key"], run_dir=req["run_dir"], split_file=req["split"],
                          cache_dir=req["cache_dir"], skills_dir=req.get("skills_dir") or None,
                          n_samples=int(req["n_samples"]), concurrency=int(req["concurrency"]),
                          env=dict(req.get("env") or {}), allow_test=bool(req.get("allow_test")))
        p = L.pool_settings(cfg.rt)
        if not (p["pool"] and cfg.key in p["pool_keys"]):
            raise L.Refused(f"pool mode is not on for a {cfg.key} leg of this run (RUNTIME daytona_leg.pool / "
                            f"pool_keys)")
        if cfg.result_path.exists():
            raise L.Refused(f"{cfg.result_path} already exists -- that leg is measured")
        hidden = (f"{SANDBOX_REPO}/.cache", f"{SANDBOX_REPO}/outputs/ssbv2_base30")
        for what, path in (("skills dir", cfg.skills_dir), ("split file", cfg.split_file)):
            if path is not None and any(str(path) == h or str(path).startswith(h + "/") for h in hidden):
                raise L.Refused(f"the {what} {path} lies under a dir each pool unit binds privately ({hidden}) -- "
                                f"it would be hidden from the runner")
        if p.get("smoke_stress_after_s") and "smoke" not in cfg.run_dir.name:
            raise L.Refused(f"smoke_stress_after_s={p['smoke_stress_after_s']} is a SMOKE knob (synthetic stress) and "
                            f"{cfg.run_dir.name} is not a smoke run dir")
        ntsib = bool(req.get("no_train_sibling"))
        if ntsib and cfg.key != "val":
            raise L.Refused(f"no_train_sibling (a VAL-only epoch) on a {cfg.key} leg")
        if cfg.key == "test":
            self._check_test_isolation(cfg)
        placement = evolver_placement(cfg.run_dir)
        if self.cfg is None:
            self._init_pool(cfg, p, placement)
        else:
            self._check_compatible(cfg, p)
            if placement != self.placement:
                raise L.Refused(f"RUNTIME evolver_daytona.placement is now {placement!r}; this pool host started with "
                                f"{self.placement!r} and its machine's egress allow list ({self.allow_csv}) is fixed "
                                f"at creation. It takes effect with a NEW pool host: once no leg is in this one, stop "
                                f"it (`ssb_leg_pool.py status`, then SIGTERM its host pid {os.getpid()}) and relaunch")
        if self.spend is None:                           # no leg is ever accepted without the guard
            raise L.Refused("the pool has no spend guard (its first setup failed); refusing to run without leg cap, "
                            "run cap and balance floor")
        sk = L.skills_fingerprint(cfg.skills_dir)
        sk_sha = sk["sha256"] if sk else None
        with self.lock:
            clash = sorted(lg.cfg.tag for lg in self.legs.values()
                           if lg.state != "finished" and lg.cfg.tag != tag and lg.cfg.skills_dir is not None
                           and cfg.skills_dir is not None and str(lg.cfg.skills_dir) == str(cfg.skills_dir)
                           and lg.skills_sha != sk_sha)
        if clash:
            raise L.Refused(f"skills dir {cfg.skills_dir} now holds other content ({str(sk_sha)[:12]}) than leg(s) "
                            f"{clash}, still in this pool, loaded: a machine holds ONE content per path; relaunch "
                            f"once they have ended")
        attempt = (cur.attempt + 1) if cur is not None else 1
        leg = PoolLeg(self, cfg, attempt, req)
        leg.skills_sha = sk_sha
        leg.src_manifest = src_manifest()
        leg.prepare(self.idx, self.p)
        with self.lock:
            self.legs[tag] = leg
            leg.t_queued = time.time()               # its units are in the queue from now on
            if tag not in self.leg_order:
                self.leg_order.append(tag)
            if self.selftest_task is None and leg.units:
                self.selftest_task = leg.units[0].task
            self.admission["legs"][tag] = {"epoch": leg.epoch, "key": cfg.key, "attempt": attempt,
                                           "units": len(leg.units), "no_train_sibling": ntsib,
                                           "joined_utc": dlx.now(), "joined_ts": round(time.time(), 3)}
            self._requeue()
        self.log(f"leg {tag}: epoch {leg.epoch}, {cfg.key}, {len(leg.units)} unit(s), skills "
                 f"{str(sk_sha)[:12] if sk_sha else 'none'}, src tree {str(manifest_sha(leg.src_manifest))[:12]}, "
                 f"cache {cfg.cache_dir.name}; queue now {len(self.queue)} unit(s)")
        return leg

    def _check_test_isolation(self, cfg) -> None:
        """A held-out TEST leg in the pool must not write into an epoch's evidence cache (the per-epoch cache dir
        that train/val legs share and the next evolver's digests are built from): neither a cache dir a train or
        val leg of this pool uses, nor one named like the loop's per-epoch dirs (<spec>_e<N>_cache)."""
        ev = {str(lg.cfg.cache_dir): lg.cfg.tag for lg in self.legs.values() if leg_key(lg) in ("train", "val")}
        c = str(cfg.cache_dir)
        if c in ev:
            raise L.Refused(f"a held-out TEST leg must use its own cache dir: {cfg.cache_dir} is the cache of "
                            f"{ev[c]} (an epoch's evidence cache)")
        if re.search(r"_e\d+_cache$", cfg.cache_dir.name):
            raise L.Refused(f"a held-out TEST leg must use its own cache dir: {cfg.cache_dir.name} is named like the "
                            f"loop's per-epoch evidence caches (<spec>_e<N>_cache); pass a new dir")

    def code_drift(self) -> dict:
        """{repo path: (sha at this host's start, sha on disk now)} of every HOST_CODE file that differs."""
        now = code_manifest(tuple(self.code_at_start))
        return {rel: (sha, now.get(rel)) for rel, sha in self.code_at_start.items() if now.get(rel) != sha}

    def _init_pool(self, cfg, p: dict, placement: str = PLACEMENT_OWN) -> None:
        """Everything is built into locals and assigned only when ALL of it succeeded --
        client, key, the spend guard's first /key reading, the pool view. A failure leaves the host exactly
        uninitialised (self.cfg None, self.spend None), so the next join retries the whole setup instead of being
        accepted as if the pool were initialised.
        Before this host creates any machine, every sandbox of this run's pool
        that Daytona still lists (<PREFIX><pool code>-*: a crashed host's machine, or one a host could not verify
        deleted) is deleted by id, and the join is REFUSED while any is still listed -- never two machines."""
        view = copy.copy(cfg)
        view.tag = f"pool_{cfg.run_dir.name}"
        view.code = pool_code(cfg.run_dir.name)
        # the pool machines' own TTL (pool_ttl_minutes; else ttl_minutes): a NEW dict, so the legs' own rt (their
        # compatibility fingerprint) is untouched; Shard._create_once and Machine.retiring read the view's
        mttl = L.pool_ttl(cfg.rt, p)
        view.rt = dict(cfg.rt, ttl_minutes=int(mttl[0]))
        bt = leg_bash_timeout(cfg)
        gcap = grader_cap(cfg, p)
        bad = layout_problem(p, gcap)
        if bad:
            raise L.Refused(bad)
        idx = self.idx or L.task_index()
        client = self.client if self.client is not None else dlx.client()
        key = self.key if self.key is not None else dlx.openrouter_key()
        fetch = self.fetch_usage or (lambda: L.openrouter_usage(key))
        spend = RunSpendGuard(fetch, leg_cap=cfg.rt["leg_spend_cap_usd"], run_cap=cfg.rt["run_spend_cap_usd"],
                              min_balance=cfg.rt["min_account_balance_usd"],
                              origin_file=Path(cfg.rt["store_root"]) / cfg.run_dir.name / "spend_origin.json",
                              origin_usage=cfg.rt.get("run_key_usage_start"), log=self.log)
        why = (f"a sandbox of run {cfg.run_dir.name}'s pool listed before pool host pid {os.getpid()} created any "
               f"machine: a previous host's leftover (only use 1 machine)")
        try:
            left = reap_pool_sandboxes(client, view.code, view.tag, self.log, why, found=self.reaped)
        except Exception as e:  # noqa: BLE001
            raise L.Refused(f"the Daytona listing of this run's pool sandboxes ({dlx.PREFIX}{view.code}-*) failed "
                            f"({type(e).__name__}: {str(e)[:300]}): no machine is created without it (a crashed "
                            f"host's machine would be a second one alive); relaunch the leg")
        if left:
            raise L.Refused(f"sandbox(es) of this run's pool are still listed after delete-by-id: "
                            f"{[(o['name'], o['id'], o['state']) for o in left]} -- a new machine now would be a "
                            f"second one alive (a run uses one machine). Delete them (`ssb_leg_daytona_loop.py "
                            f"cleanup --run-dir {cfg.run_dir}`) and relaunch the leg")
        # -- all of it succeeded
        self.placement, self.allow_csv = placement, ",".join(allow_domains(placement))
        self.first_cfg, self.p, self.cfg = cfg, dict(p), view
        self.bash_timeout, self.bash_timeout_source = bt
        self.grader, self.units_cap = gcap, gcap["units_per_machine_cap"]
        self.machine_ttl = (int(mttl[0]), mttl[1])
        self.admission["policy"] = self.p["pool_order"]
        self.admission["train_first_wait_s"] = self.p["pool_train_first_wait_s"]
        self.idx, self.client, self.key, self.spend = idx, client, key, spend
        self._attr_prev_usage = float(spend.start.get("key_usage") or 0)
        self.last_spend = time.time()
        if self.spend.breached.is_set():
            self._abort("spend guard breached before any sandbox: " + self.spend.reason)
        self.log(f"pool initialised by {cfg.tag} (ONE pool for every leg of run {cfg.run_dir.name}): "
                 f"{json.dumps(self.p, sort_keys=True)}")
        self.log(f"layout: {self.p['pool_sandboxes']} machine(s), {cap_note(self.p)}; machine TTL {mttl[0]} min "
                 f"({mttl[1]}), a machine takes units for {mttl[0] - self.p['pool_ttl_margin_min']} min; queue order "
                 f"{self.p['pool_order']}" + (" (newest epoch's train first, then val oldest epoch first, then test; "
                                              "a val unit starts only while no train unit waits, a test unit only "
                                              "while no train or val unit waits)"
                                              if self.p["pool_order"] == "train_first" else "")
                 + f"; idle machine kept {self.p['pool_keep_idle_s']}s for the next leg"
                 + (f"; idle rollover below {self.p['pool_idle_rollover_min']} min left"
                    if self.p.get("pool_idle_rollover_min") else ""))
        self.log(f"machine egress allow list {self.allow_csv} (RUNTIME evolver_daytona.placement {placement}); "
                 f"{len(self.reaped)} leftover sandbox(es) of this run's pool deleted before any machine; host code "
                 f"at start {manifest_sha(self.code_at_start)[:12]}")
        self.log(f"machine-wide grader cap: <= {gcap['units_per_machine_cap']} unit(s) per machine "
                 f"(budget {gcap['machine_recalc_budget']} recalcs // {gcap['recalc_per_process']} per runner; "
                 f"{gcap['budget_from']}) -> machine max {gcap['machine_max']}, <= "
                 f"{gcap['attempts_per_machine_max_at_n']} attempts per machine at N={cfg.n_samples}")
        self.log(f"spend: per LEG max(its metered records, its units' share of the key usage) vs "
                 f"${cfg.rt['leg_spend_cap_usd']}; run ${spend.run_delta():.4f}/${spend.run_cap} since the origin; "
                 f"balance floor ${spend.min_balance}")

    @staticmethod
    def _fingerprint(cfg) -> dict:
        """What every leg of ONE pool must share. Not the cache dir (each epoch has its own; the long-lived pool
        serves them all) and not the tag / key / skills dir (per leg by design)."""
        return {"run_dir": str(cfg.run_dir), "split_file": str(cfg.split_file),
                "split_sha256": cfg.split_sha256, "n_samples": cfg.n_samples, "concurrency": cfg.concurrency,
                "knobs": cfg.knobs, "image_id": cfg.image_id, "dockerfile_sha256": cfg.dockerfile_sha256,
                "infra": (cfg.infra_record or {}).get("sha256"), "data_overlay": cfg.data_overlay,
                "leg_settings": cfg.leg_settings, "prompt_mode": cfg.prompt_mode,
                "daytona_leg": {k: v for k, v in cfg.rt.items() if not str(k).startswith("_")}}

    def _check_compatible(self, cfg, p: dict) -> None:
        a, b = self._fingerprint(self.first_cfg), self._fingerprint(cfg)
        diff = sorted(k for k in a if a[k] != b[k])
        if diff:
            raise L.Refused(f"{cfg.tag} cannot share the pool of {self.first_cfg.tag}: they differ in {diff} (a "
                            f"RUNTIME daytona_leg change takes effect with a NEW pool host: once no leg is in this "
                            f"one, stop it -- `ssb_leg_pool.py status`, then SIGTERM its host pid -- and relaunch)")

    def _requeue(self) -> None:
        self.queue = order_queue([self.legs[t] for t in self.leg_order if t in self.legs],
                                 (self.p or {}).get("pool_order", "interleave"))

    # -- aborts / stops
    def _abort(self, why: str) -> None:
        if not self.abort_all.is_set():
            self.abort_reason = why
            self.log(f"ABORT ALL: {why}")
            self.abort_all.set()
            for m in self.machines:
                m.wake.set()

    def process_aborts(self) -> None:
        for f in sorted((self.pdir / "joins").glob("*.abort")):
            done = f.with_suffix(".abort_seen")
            if done.exists():
                continue
            try:
                d = json.loads(f.read_text())
            except (OSError, ValueError):
                d = {}
            leg = self.legs.get(d.get("tag"))
            # units_done too: a leg whose units all ended may still wait for its machines' log sync; an abort
            # there must fail it now, not let it finish OK after its follower gave up
            if leg is not None and leg.attempt == d.get("attempt") and leg.state in ("active", "units_done"):
                self.abort_leg(leg, f"leg abort requested by its follower (pid {d.get('pid')})")
            done.write_text(dlx.now())

    def abort_leg(self, leg: PoolLeg, why: str) -> None:
        with self.lock:
            leg.state, leg.abort_reason = "aborting", why
            for u in leg.units:
                if u.state == "queued":
                    u.state, u.t_end, u.rc = "ended", time.time(), "never launched"
                    u.summary["aborted"] = why
                    u.save()
                elif u.state in ("assigned", "running"):
                    u.kill_reason = why
                    if u.machine is not None:
                        u.machine.wake.set()
            self._requeue()
        self.log(f"ABORT LEG {leg.cfg.tag}: {why}")

    def _stop_leg(self, leg, why: str) -> int:
        """(caller holds the lock) No unit of `leg` is admitted any more; its running units FINISH (never killed);
        it is written FAILED with `why` once they ended. Returns how many units will never launch: the queued ones
        and the ones already ASSIGNED to a machine but still in its inbox -- its thread has not taken
        them yet, so they are withdrawn under the lock and can never start. A unit its machine thread already took
        (launching now) runs to its end like the running ones."""
        leg.fail_reason = leg.fail_reason or why
        n = 0
        for u in leg.units:
            m = u.machine
            if u.state == "assigned" and m is not None and u in m.inbox:
                m.inbox.remove(u)
                if m.units.get(u.key) is u:
                    del m.units[u.key]
                u.machine, u.kill_reason = None, why
            elif u.state != "queued":
                continue
            u.state, u.t_end, u.rc = "ended", time.time(), "never launched"
            u.summary["aborted"] = why
            u.save()
            n += 1
        return n

    def fail_leg(self, leg, why: str) -> None:
        """A LEG's spend cap: stop admitting its units, let its running units finish, write it FAILED."""
        with self.lock:
            if leg.fail_reason or leg.state == "finished":
                return
            n = self._stop_leg(leg, why)
            self._requeue()
            running = sum(1 for u in leg.units if u.state in ("assigned", "running", "syncing"))
        self.log(f"LEG STOPPED {leg.cfg.tag}: {why} -- {n} queued / not yet launched unit(s) will never launch, "
                 f"{running} running unit(s) finish, then the leg is written FAILED; the other legs go on")

    def drain_all(self, why: str) -> None:
        """A RUN spend breach (run cap / balance floor): nothing is admitted any more, running units FINISH, every
        active leg is written FAILED once its units ended, new joins are refused, machines go once idle."""
        with self.lock:
            if self.admit_stop:
                return
            self.admit_stop = why
            n = sum(self._stop_leg(lg, why) for lg in self.legs.values() if lg.state != "finished")
            self._requeue()
            running = sum(1 for lg in self.legs.values() for u in lg.units
                          if u.state in ("assigned", "running", "syncing"))
        self.log(f"DRAIN ALL ({why}): {n} queued / not yet launched unit(s) will never launch, {running} running "
                 f"unit(s) finish; every active leg will be written FAILED; new joins are refused")

    def requeue_lost(self, u: Unit, m: "Machine", stage) -> None:
        """A unit of a LOST machine, deleted-verified, goes back to the queue ONCE, for another
        machine, in a fresh unit dir (try + 1). If its private cache could be salvaged (`stage`), the samples it
        finished there seed the re-run (they REPLAY, no model call) and its tasks dir (their snapshots) joins the
        unit's -- unless that attempt's runner log holds a grader-failure warning, which the final attempt's
        log would not carry: then nothing is reused. The earlier try's local files move into its _lost_ dir."""
        cfg = u.leg.cfg
        stage = Path(stage) if stage else u.dir / f"_lost_{m.name}_t{u.tries}"
        stage.mkdir(parents=True, exist_ok=True)
        for name in ("result", "out", "llm_stats.json", "runtime_env.sandbox.json", "sync.tgz"):
            src = u.dir / name
            if src.exists():
                dst = stage / f"prelost_{name}"
                shutil.move(str(src), str(dst))
        base_rel = {p.relative_to(cfg.cache_dir).as_posix() for p in u.seeds}
        extra, fresh_cost, note = [], 0.0, None
        sc = stage / "cache"
        if sc.is_dir():
            recs = []
            for p in sorted(sc.rglob("*.json")):
                if p.parent == sc or L._record_task(p) != u.task:
                    continue
                rel = p.relative_to(sc).as_posix()
                if rel in base_rel:
                    continue
                recs.append((p, rel))
                try:
                    fresh_cost += float((json.loads(p.read_text()).get("trace") or {}).get("total_cost_usd") or 0)
                except (ValueError, OSError):
                    pass
            rl = stage / "out" / "runner.log"
            gl = sum(1 for ln in rl.read_text(errors="replace").splitlines() if GRADER_FAIL_RE.search(ln)) \
                if rl.is_file() else 0
            if recs and not rl.is_file():
                note = (f"{len(recs)} salvaged record(s) NOT reused: that try's runner log was not salvaged (its "
                        f"grader-failure warnings cannot be ruled out)")
            elif recs and gl:
                note = f"{len(recs)} salvaged record(s) NOT reused: {gl} grader-failure warning(s) in that try's log"
            elif recs:
                extra = recs
                st = stage / "tasks"
                if st.is_dir():
                    shutil.copytree(st, u.dir / "tasks", dirs_exist_ok=True)
        with self.lock:
            if (u.leg.state != "active" or u.kill_reason or self.abort_all.is_set() or u.leg.fail_reason
                    or self.admit_stop):
                # the leg / pool started aborting (or stopped admitting: spend) while this machine was wound down:
                # never run it again
                u.state, u.t_end, u.rc = "ended", time.time(), u.rc or "machine lost"
                u.summary.setdefault("error", f"machine {m.name} lost: {m.failed} (not requeued: "
                                              f"{u.kill_reason or self.abort_reason or u.leg.fail_reason or self.admit_stop or 'leg ' + u.leg.state})")
                u.save()
                return
            m.units.pop(u.key, None)
            m.units_lost.append(u.key)
            entry = {"machine": m.name, "sandbox_id": m.sid, "deleted": m.deleted, "error": m.failed,
                     "try": u.tries, "runner_rc": u.rc, "salvaged": bool(sc.is_dir()), "stage": str(stage),
                     "records_reused": len(extra), "fresh_cost_usd": round(fresh_cost, 5), "note": note,
                     "requeued_utc": dlx.now()}
            u.summary.setdefault("lost_attempts", []).append(entry)
            for k in ("error", "runner_rc", "progress", "secret_hits", "cache_records", "cache_cost_usd",
                      "fresh_cost_usd", "bash_timeouts_exact", "grader_failure_lines", "synced_utc"):
                u.summary.pop(k, None)
            u.extra_seeds = extra
            u.seeded_records = sum(1 for s in u.seeds if s.name != "CACHE_ENV") + len(extra)
            n_have = sum(1 for s in u.seeds if s.name != "CACHE_ENV") + len(extra)
            u.need = max(0, cfg.n_samples - n_have)
            u.summary["need"] = u.need
            u.tries += 1
            u.lost_requeues += 1
            u.avoid.add(m.name)
            u.state, u.machine, u.pid, u.rc = "queued", None, None, None
            u.records, u.dead_polls, u.t_launch, u.stats = 0, 0, None, {}
            u.kill_reason, u.killed, u.log_off, u.gfail_seen = None, False, 0, 0
            u.t_dispatch = u.t_end = None
            self.lost_requeues.append({"unit": u.key, **entry})
            if leg_key(u.leg) == "train" and self.admission["train_all_dispatched_utc"]:
                # train first: this train unit waits again, so the other legs' units are held until it has a spot
                self.admission["train_requeued_after_all_dispatched"] += 1
            self._requeue()
        u.save()
        self.log(f"REQUEUED {u.key} (lost with machine {m.name}: {m.failed}); try {u.tries}, need {u.need}, "
                 f"{len(extra)} salvaged record(s) reused{'; ' + note if note else ''}")

    # -- machines
    def _may_create(self) -> bool:
        """Creation budget: machine FAILURES since the last machine that became ready (a long-lived
        host must not end the run over two failures hours apart) -- at most max(2, pool_sandboxes) of them; a
        machine retired near its TTL (or rolled over when idle) is no failure; an absolute bound stops any runaway
        (a run's pool lives long: the bound grows with the machines retired so far)."""
        n = int(self.p["pool_sandboxes"])
        return len(self._failures_since_ready()) < max(2, n) and self.created < 6 * n + self.retired + self.rollovers

    def _failures_since_ready(self) -> list:
        """The failed machines whose failure came after the most recent moment a machine became READY (a machine
        that failed after being ready counts until a later one is ready)."""
        last_ready = max((float(getattr(m, "t_ready", None) or 0.0) for m in self.machines), default=0.0)
        now = time.time()
        return [m for m in self.machines if getattr(m, "failed", None)
                and float(getattr(m, "t_end", None) or now) >= last_ready]

    def _warm_wanted(self) -> bool:
        return (bool(self.warm_until) and time.time() < float(self.warm_until) and not self.admit_stop
                and not self.abort_all.is_set())

    def machines_to_create(self) -> int:
        """(callers hold the lock) How many machines to start now. EVERY machine not yet deleted counts toward
        pool_sandboxes (the org memory cap) -- a retiring or released one included, so its
        replacement waits until it is gone (never two alive with pool_sandboxes 1); only a machine that takes units
        counts as serving. With an empty queue, one machine only as the warm replacement of an idle rollover."""
        existing = [m for m in self.machines if not m.done.is_set()]
        serving = [m for m in existing if not m.release.is_set() and not m.retiring() and not m.failed]
        open_units = sum(1 for lg in self.legs.values() for u in lg.units
                         if u.state in ("queued", "assigned", "running"))
        cap = int(self.p["pool_sandboxes"])
        if self.queue:
            want = min(cap, open_units)
        elif self._warm_wanted():
            want = 1
        else:
            want = 0
        n_new = min(want - len(serving), cap - len(existing)) if want else 0
        return max(0, n_new) if self._may_create() else 0

    def _creation_guard(self) -> tuple:
        """(may a machine be created now?, why not): every sandbox of this run's pool that
        Daytona lists must be a live machine of THIS host (or one this host is creating: its id is not known yet).
        Anything else -- a machine of this host whose deletion could not be verified, a sandbox whose id never
        reached it -- is deleted by id (ledgered) when no creation of this host is in flight, and blocks the creation
        while it is still listed. A failing listing blocks too (fail closed). Re-checked every GUARD_EVERY_S while
        blocked; ensure_machines aborts the pool after GUARD_ABORT_S with work queued and no machine."""
        now = time.time()
        g = self.guard
        if g["block"] and now < g["next_ts"]:
            return False, g["block"]
        g["checks"] += 1
        why = None
        try:
            rows = pool_sandboxes_listed(self.client, self.cfg.code)
            with self.lock:
                live = {m.sid for m in self.machines if m.sid and not m.done.is_set()}
                pending = sum(1 for m in self.machines if not m.done.is_set() and not m.sid)
            other = [o for o in rows if o["id"] not in live]
            if other and not pending:
                for o in other:
                    dlx.ledger(event="pool_orphan_found", tag=self.cfg.tag, id=o["id"], name=o["name"],
                               state=o["state"], why="listed, not a live machine of this host, before a creation")
                    ok = dlx.delete_by_id(self.client, o["id"], self.cfg.tag)
                    g["deleted"].append({"id": o["id"], "name": o["name"], "verified": ok, "utc": dlx.now()})
                    self.log(f"GUARD: pool sandbox {o['name']} ({o['id']}) is listed but no live machine of this "
                             f"host -> deleted by id: {'verified' if ok else 'NOT verified'}")
                rows = pool_sandboxes_listed(self.client, self.cfg.code)
                other = [o for o in rows if o["id"] not in live]
            if len(other) > pending:
                why = (f"{len(other)} sandbox(es) of this run's pool listed that are not a live machine of this host: "
                       f"{[(o['name'], o['id'], o['state']) for o in other][:4]}")
        except Exception as e:  # noqa: BLE001
            why = f"the Daytona listing of this run's pool sandboxes failed ({type(e).__name__}: {str(e)[:200]})"
        if why is None:
            g["block"] = g["since"] = None
            return True, None
        g["block"], g["next_ts"] = why, now + GUARD_EVERY_S
        g["since"] = g["since"] or now
        g["blocked_checks"] += 1
        return False, why

    def ensure_machines(self) -> None:
        with self.lock:
            n_new = self.machines_to_create()
            task0 = self.queue[0].task if self.queue else self.selftest_task
        if n_new:
            ok, why = self._creation_guard()
            if not ok:
                n_new = 0
                if why != self.guard.get("last_logged"):
                    self.guard["last_logged"] = why
                    self.log(f"NO MACHINE CREATED (never two alive; re-checked every {GUARD_EVERY_S:.0f}s): {why}")
                with self.lock:
                    live = [m for m in self.machines if not m.done.is_set()]
                    stuck = bool(self.queue) and not live and \
                        time.time() - float(self.guard["since"] or time.time()) > GUARD_ABORT_S
                if stuck:
                    self._abort(f"no machine could be created for {GUARD_ABORT_S / 60:.0f} min with work queued: "
                                f"{why}")
        with self.lock:
            if n_new and not self.queue:
                self.warm_until = None                   # the warm replacement is on its way
        stagger = float(os.environ.get("EVOSKILL_DAYTONA_START_STAGGER_S", "3"))
        for _ in range(max(0, n_new)):
            self.created += 1
            m = Machine(self, f"m{self.created:02d}", task0)
            th = threading.Thread(target=m.run, daemon=False, name=m.name)
            with self.lock:
                self.machines.append(m)
            th.start()
            self.threads.append(th)
            self.log(f"machine {m.name} starting (selftest task {task0})")
            time.sleep(stagger)
        with self.lock:
            failed_now = [m for m in self.machines if m.done.is_set() and m.failed and
                          m.name not in {x['machine'] for x in self.machine_failures}]
            for m in failed_now:
                self.machine_failures.append({"machine": m.name, "sandbox_id": m.sid, "error": m.failed})
                self.log(f"machine {m.name} FAILED: {m.failed}")
            live_or_coming = [m for m in self.machines if not m.done.is_set()]
            if self.queue and not live_or_coming and not self._may_create():
                self._abort("no machine left to run the queue (every machine failed)")

    def _pending_join_tags(self) -> set:
        out = set()
        for f in self._pending_joins():
            try:
                out.add(json.loads(f.read_text()).get("tag"))
            except (OSError, ValueError):
                pass
        return out

    def _train_counts(self) -> dict:
        c = Counter(u.state for lg in self.legs.values() if leg_key(lg) == "train" for u in lg.units)
        return {"queued": c["queued"], "on_machine": c["assigned"] + c["running"] + c["syncing"],
                "ended": c["ended"], "legs": sum(1 for lg in self.legs.values() if leg_key(lg) == "train")}

    def dispatch(self) -> None:
        with self.lock:
            if self.admit_stop:
                return
            ready = [m for m in self.machines if m.usable()]
            if not ready or not self.queue:
                return
            now = time.time()
            train_first = self.p.get("pool_order") == "train_first"
            pend = self._pending_join_tags() if train_first else set()
            qc = queued_by_class(self.legs)              # kept current below, as units get their spots
            left, held = [], None
            for u in self.queue:
                if u.state != "queued":
                    continue
                if getattr(u.leg, "fail_reason", None):  # a stopped leg's units never start (normally ended at
                    u.state, u.t_end, u.rc = "ended", time.time(), "never launched"   # the stop already)
                    u.summary["aborted"] = u.leg.fail_reason
                    qc[leg_class(u.leg)] -= 1
                    continue
                blk = admission_block(u, self.legs, self.p, now, pend, qc)
                if blk is not None:                     # TRAIN FIRST: this non-train unit waits (logged below)
                    held = held or (u, blk)
                    left.append(u)
                    continue
                m, capped = pick_machine(u, ready, self.p, self.units_cap)
                if m is None:
                    for c in capped:
                        c.ramp.note_binding()
                    left.append(u)
                    continue
                u.state, u.machine, u.cap_at_dispatch = "assigned", m, m.ramp.cap
                qc[leg_class(u.leg)] -= 1
                m.units[u.key] = u
                m.inbox.append(u)
                m.wake.set()
                self._note_admission(u, m, now)
            self.queue = left
            self._note_hold(held)
            tc = self._train_counts()
            if tc["legs"] and not tc["queued"]:
                if self.admission["train_all_dispatched_utc"] is None:
                    self.admission["train_all_dispatched_utc"] = dlx.now()
                    self.admission["train_all_dispatched_ts"] = round(now, 3)
                if self._train_waiting or not self._train_logged:
                    self._train_logged = True
                    self.log(f"ADMISSION: every train unit has a spot or ended ({tc['on_machine']} on a machine, "
                             f"{tc['ended']} ended)" + (" -- other legs' units may start now" if train_first else ""))
            self._train_waiting = bool(tc["queued"])

    def _note_admission(self, u, m, now: float) -> None:
        """(lock held) One admission decision: its reason in host.log (with the leg's epoch), the unit's summary,
        pool_ramp.admission (decisions + the leg's own record)."""
        key = leg_key(u.leg)
        ep = getattr(u.leg, "epoch", leg_epoch(u.tag))
        tc = self._train_counts()
        if self.p.get("pool_order") != "train_first":
            why = f"order {self.p.get('pool_order')}: next in the queue that fits"
        elif key == "train":
            why = "train unit (train first)"
        elif key == "val":
            why = (f"no train unit waiting (train units: {tc['ended']} ended, {tc['on_machine']} on a machine, "
                   f"{tc['queued']} queued)")
            if tc["queued"]:
                self.admission["non_train_admitted_while_train_queued"] += 1   # never, by admission_block
        else:
            vq = sum(1 for lg in self.legs.values() if leg_key(lg) == "val" for x in lg.units if x.state == "queued")
            why = f"no train or val unit waiting (train queued {tc['queued']}, val queued {vq}; the test runs last)"
            if tc["queued"]:
                self.admission["non_train_admitted_while_train_queued"] += 1   # never, by admission_block
            if tc["queued"] or vq:
                self.admission["test_admitted_while_train_or_val_queued"] += 1  # never, by admission_block
        u.t_dispatch = now
        u.summary.update(dispatched_utc=dlx.now(), admission=why)
        a = self.admission
        a["first_dispatch_utc"].setdefault(key or u.tag, dlx.now())
        a.setdefault("first_dispatch_ts", {}).setdefault(key or u.tag, round(now, 3))
        lr = a["legs"].setdefault(u.tag, {"epoch": ep, "key": key})
        lr.setdefault("first_dispatch_utc", dlx.now())
        lr.setdefault("first_dispatch_ts", round(now, 3))
        lr["admitted"] = lr.get("admitted", 0) + 1
        if not any(x.state == "queued" for x in u.leg.units) and not lr.get("all_dispatched_utc"):
            lr["all_dispatched_utc"], lr["all_dispatched_ts"] = dlx.now(), round(now, 3)
            self.log(f"ADMISSION: every unit of {key} leg {u.tag} (epoch {ep}) has a spot or ended")
        if len(a["decisions"]) < DECISIONS_MAX:
            a["decisions"].append({"unit": u.key, "leg": key, "tag": u.tag, "epoch": ep, "machine": m.name,
                                   "try": u.tries, "utc": dlx.now(), "ts": round(now, 3), "cap": m.ramp.cap,
                                   "inflight_before": m.inflight() - u.remaining(), "train_queued": tc["queued"],
                                   "reason": why})
        self.log(f"ADMIT {u.key} ({key}) -> {m.name}: {why} [epoch {ep}]")

    def _note_hold(self, held) -> None:
        """(lock held) Log a change of the TRAIN-FIRST hold (its kind), never the same one every tick."""
        kind = held[1][0] if held else None
        if kind == self.admission["hold_now"]:
            return
        self.admission["hold_now"] = kind
        if held is None:
            return
        u, (kind, why) = held
        n = sum(1 for x in self.queue if x.state == "queued" and leg_key(x.leg) != "train")
        rec = {"utc": dlx.now(), "ts": round(time.time(), 3), "kind": kind, "reason": why,
               "blocked_units": n, "first_blocked": u.key, "epoch": getattr(u.leg, "epoch", None)}
        if len(self.admission["holds"]) < 500:
            self.admission["holds"].append(rec)
        self.log(f"HOLD {n} non-train unit(s) (first {u.key}, epoch {rec['epoch']}): {why} -- train first")

    def update_429(self) -> None:
        with self.lock:
            calls = sum(int((u.stats or {}).get("fresh_responses") or 0)
                        for lg in self.legs.values() for u in lg.units)
            n429 = sum(int(((u.stats or {}).get("status_warnings") or {}).get("429", 0))
                       for lg in self.legs.values() for u in lg.units)
        now = time.time()
        self.api_window.append((now, calls, n429))
        win = float(self.p.get("ramp_interval_s") or 180)
        while len(self.api_window) > 1 and now - self.api_window[0][0] > win:
            self.api_window.popleft()
        t0, c0, e0 = self.api_window[0]
        dc, de = max(0, calls - c0), max(0, n429 - e0)
        rate = de / max(1, dc + de)
        hold = de > 0 and rate > float(self.first_cfg.rt.get("max_429_rate", 0.02))
        if hold and not self.hold_429:
            self.api_429["holds"] += 1
            self.log(f"API 429 HOLD: {de} 429s vs {dc} fresh calls in the last {win:.0f}s ({rate:.1%}) -- no cap "
                     f"raises until it clears")
        self.api_429["rate_max"] = max(self.api_429["rate_max"], round(rate, 4))
        self.hold_429 = hold

    # -- spend, per leg
    def leg_spend(self, leg) -> dict:
        """A LEG's spend, measured on its own: max(its metered NEW cache records -- total_cost_usd of every record its
        units wrote, known once a unit ended here --, its units' share of the org key usage, attributed at every
        spend check in proportion to the fresh responses each leg's units reported in that interval, which covers
        rollouts still running). The cap is RUNTIME leg_spend_cap_usd."""
        met = float(leg.metered_usd())
        att = float(leg.spend_attributed)
        return {"leg_cap_usd": float(leg.cfg.rt["leg_spend_cap_usd"]), "metered_usd": round(met, 5),
                "key_attributed_usd": round(att, 5), "measured_usd": round(max(met, att), 5)}

    def attribute_spend(self) -> None:
        """After a key reading: the usage since the previous reading goes to the legs by their units' new fresh
        responses in that interval (u.stats, per leg attempt and unit try); none reported = unattributed."""
        if self.spend is None:
            return
        usage = float(self.spend.last.get("key_usage") or 0)
        prev = usage if self._attr_prev_usage is None else float(self._attr_prev_usage)
        d_usd = max(0.0, usage - prev)
        self._attr_prev_usage = usage
        with self.lock:
            per: dict = {}
            for lg in self.legs.values():
                d = 0
                for u in lg.units:
                    k = (lg.cfg.tag, lg.attempt, u.name, u.tries)
                    n = int((u.stats or {}).get("fresh_responses") or 0)
                    if n > self._resp_seen.get(k, 0):
                        d += n - self._resp_seen.get(k, 0)
                        self._resp_seen[k] = n
                if d:
                    per[lg] = d
            tot = sum(per.values())
            if d_usd <= 0:
                return
            if tot <= 0:
                self.unattributed_usd += d_usd
                return
            for lg, d in per.items():
                lg.spend_attributed += d_usd * d / tot

    def check_leg_caps(self) -> None:
        for lg in list(self.legs.values()):
            if lg.state == "finished" or lg.fail_reason:
                continue
            s = self.leg_spend(lg)
            if s["measured_usd"] > s["leg_cap_usd"]:
                self.fail_leg(lg, f"leg spend cap: ${s['measured_usd']:.4f} > ${s['leg_cap_usd']} (metered records "
                                  f"${s['metered_usd']:.4f}, key usage attributed ${s['key_attributed_usd']:.4f})")

    # -- machines: keep, lease, retire, roll over
    def _leased(self, m, leases: dict, now: float) -> list:
        """The holders of a valid lease on machine `m` (Registry) -- never honoured past its TTL minus the sync
        margin, when the machine must go anyway."""
        who = leases.get(str(m.sid)) if m.sid else None
        if not who:
            return []
        te = m.ttl_expires()
        if te is not None and now >= te - float(L.POOL_SYNC_MARGIN_MIN) * 60:
            return []
        return list(who)

    def release_idle(self) -> None:
        now = time.time()
        leases = self.reg.leases(now)
        with self.lock:
            for m in self.machines:                  # a retiring machine goes as soon as it is idle (and unleased)
                if m.retiring() and not m.release.is_set() and not m.done.is_set() and not m.active() \
                        and not m.inbox and not self._leased(m, leases, now):
                    m.release.set()
                    m.wake.set()
                    self.retired += 1
                    self.log(f"retiring idle machine {m.name} (older than its TTL - pool_ttl_margin_min, or disk "
                             f"low); its replacement starts once it is deleted")
            if self.queue or self._pending_joins():
                self.t_queue_empty = self.t_idle = None
                return
            self.t_queue_empty = self.t_queue_empty or now
            quiet = now - self.t_queue_empty
            live = [m for m in self.machines if not m.release.is_set() and not m.done.is_set()]
            # busy: any machine not yet deleted with a unit on it (a released / retiring one still draining included)
            # a machine with a valid lease (the evolver's container) is busy too: the keep window restarts when the
            # lease ends (an evolver turn longer than pool_keep_idle_s must not lose its machine)
            busy = [m for m in self.machines if not m.done.is_set()
                    and (m.active() or m.inbox or self._leased(m, leases, now))]
            idle = [m for m in live if m not in busy]
            # the POOL is idle (the keep window's clock): nothing queued, no join pending, no unit on any machine
            self.t_idle = None if busy else (self.t_idle or now)
            idle_for = now - self.t_idle if self.t_idle else 0.0
            keep_s = 0.0 if (self.admit_stop or self.abort_all.is_set()) else float(self.p.get("pool_keep_idle_s") or 0)
            # the newest usable idle machine is KEPT for keep_s once nothing runs (the next epoch's legs come after
            # its evolver); every other idle machine goes after pool_idle_release_s, as in a per-epoch pool
            keeper = None
            if not busy and keep_s > 0:
                cands = [m for m in idle if m.usable()]
                keeper = max(cands, key=lambda m: m.t_created or 0) if cands else None
            roll = self.p.get("pool_idle_rollover_min")
            debounce = float(self.p["pool_idle_release_s"])
            for m in idle:
                if not m.ready.is_set() or self._leased(m, leases, now):
                    continue                             # still starting (e.g. a warm replacement), or leased
                if m is keeper:
                    ra = m.retire_at()
                    # only a machine that already served units (a fresh warm replacement is never rolled over)
                    if roll and ra is not None and ra - now < float(roll) * 60 and idle_for >= debounce \
                            and idle_for < keep_s and m.units:
                        m.release.set()
                        m.wake.set()
                        self.rollovers += 1
                        self.warm_until = self.t_idle + keep_s
                        self.log(f"ROLLOVER idle machine {m.name}: {max(0.0, (ra - now) / 60):.0f} min left before it "
                                 f"retires (< pool_idle_rollover_min {roll}); a replacement is created once it is "
                                 f"deleted, while the pool is idle")
                        continue
                    if idle_for < keep_s:
                        continue
                    m.release.set()
                    m.wake.set()
                    self.log(f"releasing idle machine {m.name}: the pool had no work for {idle_for:.0f}s "
                             f"(pool_keep_idle_s {keep_s:.0f})")
                    continue
                if quiet >= debounce:
                    m.release.set()
                    m.wake.set()
                    self.log(f"releasing idle machine {m.name} (queue empty)")

    # -- legs
    def _machine(self, name):
        return next((m for m in self.machines if m.name == name), None)

    def _machines_of(self, leg) -> list:
        """Every machine a unit of `leg` ran (or was lost) on."""
        names = {u.machine.name for u in leg.units if u.machine is not None}
        names |= {a.get("machine") for u in leg.units for a in u.summary.get("lost_attempts") or []}
        return [m for m in self.machines if m.name in names]

    def check_legs(self) -> None:
        """A leg whose units have ALL ended: ask every machine they ran on (still alive) to sync its logs for the key
        scan; once they did (or were deleted, their wind-down synced them), write the leg's result NOW."""
        for lg in list(self.legs.values()):
            with self.lock:
                open_ = [u for u in lg.units if u.state != "ended"]
                if lg.state in ("active", "aborting") and not open_:
                    lg.state = "units_done"
                    lg.t_units_done = time.time()
                    lg.log_sync_wait = {m.name: m.request_log_sync() for m in self._machines_of(lg)
                                        if not m.done.is_set()}
                    a = self.admission["legs"].setdefault(lg.cfg.tag, {"epoch": lg.epoch, "key": leg_key(lg)})
                    a["units_done_utc"], a["units_done_ts"] = dlx.now(), round(time.time(), 3)
                    if leg_key(lg) == "train" and self.admission["train_all_ended_utc"] is None:
                        self.admission["train_all_ended_utc"] = dlx.now()
                        self.admission["train_all_ended_ts"] = round(time.time(), 3)
                    self.log(f"leg {lg.cfg.tag}: every unit ended (epoch {lg.epoch}); its result is written once "
                             f"{sorted(lg.log_sync_wait) or 'no live machine'} synced its logs")
            if lg.state != "units_done":
                continue
            try:
                done, probs = self._log_sync_state(lg)
            except Exception as e:  # noqa: BLE001 -- this leg only
                self.log(f"log-sync check of {lg.cfg.tag} crashed: {traceback.format_exc()[-800:]}")
                self.finish_leg(lg, pre_abort=f"the check of its machines' log sync crashed ({type(e).__name__}: {e}): "
                                              f"the key scan of the machine logs is unverified")
                continue
            if done:
                lg.log_sync_problems = probs
                self.finish_leg(lg, pre_abort=lg.abort_reason or lg.fail_reason)

    def _log_sync_state(self, lg) -> tuple:
        """(may the result be written now?, {machine: problem}) -- every machine this leg ran on has served the log
        sync requested after the leg's last unit ended (or is gone: its wind-down synced and scanned)."""
        probs = {}
        for name, n in (lg.log_sync_wait or {}).items():
            m = self._machine(name)
            if m is None or m.done.is_set():
                continue
            if m.log_sync_done < n:
                if time.time() - float(lg.t_units_done or time.time()) > LOG_SYNC_TIMEOUT_S:
                    probs[name] = (f"machine {name}: its logs were not synced within {LOG_SYNC_TIMEOUT_S}s after the "
                                   f"leg's last unit ended: the key scan of the machine's logs did not run")
                    continue
                return False, {}
            if m.log_sync_ok_upto < n:
                probs[name] = (f"machine {name}: the log sync for the key scan FAILED {LOG_SYNC_TRIES}x "
                               f"({m.log_sync_error})")
        return True, probs

    def _unit_end_check(self, leg, u) -> str | None:
        """LegRun.unit_end_check for a pool leg (replaces 'sandbox verified deleted'): the unit ended here, its
        outputs are synced home, and the machine it ran on synced its logs for the key scan after the leg ended."""
        if u.state != "ended":
            return f"{u.name}: not ended ({u.state}) when the leg's result was written"
        if u.rc not in ("never launched", "launch failed") and not u.summary.get("synced_utc"):
            return f"{u.name}: its outputs were never synced home"
        probs = getattr(leg, "log_sync_problems", None) or {}
        mname = u.machine.name if u.machine is not None else None
        if mname in probs:
            first = next((x for x in leg.units if x.machine is not None and x.machine.name == mname), None)
            if first is u:
                return f"{u.name}: {probs[mname]}"
        return None

    def all_done(self) -> bool:
        """Every leg's units ended and no machine has a unit (the per-epoch pool's close condition; kept for
        readers -- the long-lived host closes on _should_close)."""
        with self.lock:
            if not self.legs or self.queue:
                return False
            if any(lg.state in ("active", "aborting") for lg in self.legs.values()):
                return False
            return not any(m.active() or m.inbox for m in self.machines)

    def _should_close(self) -> bool:
        """The host closes when every leg is FINISHED (result or FAILED written), nothing is queued, no warm
        replacement is wanted and every machine is gone (released after pool_keep_idle_s, or failed)."""
        with self.lock:
            if not self.legs or self.queue:
                return False
            if any(lg.state != "finished" for lg in self.legs.values()):
                return False
            if self._warm_wanted():
                return False
            return all(m.done.is_set() for m in self.machines)

    # -- status + records
    def status(self) -> dict:
        with self.lock:
            return {"ts": dlx.now(), "host_pid": os.getpid(), "pool_dir": str(self.pdir), "scope": POOL_SCOPE,
                    "legs": {t: lg.counts() for t, lg in self.legs.items()}, "queue": len(self.queue),
                    "machines_ready": sum(1 for m in self.machines if m.usable()),
                    "hold_429": self.hold_429, "abort": self.abort_reason, "admit_stop": self.admit_stop,
                    "units_cap": self.units_cap, "keep_idle_s": self.p.get("pool_keep_idle_s"),
                    "idle_since": self.t_idle, "queue_empty_since": self.t_queue_empty,
                    "warm_until": self.warm_until,
                    "admission": {"policy": self.admission.get("policy"), "hold_now": self.admission.get("hold_now"),
                                  "train_all_dispatched_utc": self.admission.get("train_all_dispatched_utc"),
                                  "train_all_ended_utc": self.admission.get("train_all_ended_utc")},
                    "machines": {m.name: {"cap": m.ramp.cap if m.ramp else None,
                                          "ceiling": m.ramp.ceiling if m.ramp else None,
                                          "inflight": m.inflight(), "units": m.active_units(),
                                          "ready": m.ready.is_set(), "failed": m.failed,
                                          "released": m.release.is_set(), "state": m.state_name(),
                                          "retire_at_ts": m.retire_at(), "signals": m.sig}
                                 for m in self.machines}}

    def write_status(self) -> None:
        try:
            L._atomic_write(self.pdir / "status.json", self.status())
        except OSError:
            pass

    def machines_record(self, closed: bool = False) -> dict:
        now = time.time()
        leases = self.reg.leases(now)
        with self.lock:
            ms = {m.name: m.public_record(self._leased(m, leases, now)) for m in self.machines}
            cur = next((m.name for m in self.machines if m.usable()), None)
        return {"schema": MACHINES_SCHEMA, "run": self.first_cfg.run_dir.name if self.first_cfg else None,
                "pool_root": str(self.reg.root), "pool_dir": str(self.pdir), "host_pid": os.getpid(),
                "host_state": "closed" if closed else "open", "current": cur, "machines": ms,
                "evolver_placement": self.placement,
                "domain_allow_list": [x for x in self.allow_csv.split(",") if x],
                "leases_dir": str(self.reg.leases_dir),
                "lease_rule": ("a valid lease (until_ts in the future; pid alive when given) keeps its machine from "
                               "being released for idleness or retirement, never past ttl_expires_ts - "
                               f"{L.POOL_SYNC_MARGIN_MIN} min; check retire_at_ts before starting work")}

    def publish_machines(self, closed: bool = False) -> None:
        """machine.json (Registry): rewritten when a machine's identity/state/lease changes, else once a minute."""
        try:
            rec = self.machines_record(closed)
            vol = ("units_running", "inflight", "disk_free_gib", "cap")
            key = json.dumps({"c": rec["current"], "s": rec["host_state"],
                              "m": {n: {k: v for k, v in d.items() if k not in vol}
                                    for n, d in rec["machines"].items()}}, sort_keys=True, default=str)
            last_key, last_t = self._machines_pub
            if key == last_key and time.time() - last_t < 60:
                return
            if key != last_key:
                self._machines_gen += 1
            rec.update(generation=self._machines_gen, updated_utc=dlx.now(), updated_ts=round(time.time(), 1))
            L._atomic_write(self.reg.machines_file, rec)
            self._machines_pub = (key, time.time())
        except OSError:
            pass

    def pool_ramp_block(self) -> dict:
        with self.lock:
            adm = copy.deepcopy(self.admission)
            return {"pool_dir": str(self.pdir), "host_pid": os.getpid(), "scope": POOL_SCOPE,
                    "settings": {k: self.p.get(k) for k in L.POOL_DEFAULTS},
                    "solver_bash_timeout_s": self.bash_timeout, "solver_bash_timeout_from": self.bash_timeout_source,
                    "cmdwatch": {"every_s": CMDWATCH_EVERY_S,
                                 "counts_as_timeout_at_s": self.bash_timeout - CMDWATCH_EVERY_S - 1.0},
                    "legs": list(self.leg_order), "api_429": dict(self.api_429),
                    "cap_policy": cap_note(self.p) if self.p else None,
                    "machine_ttl_minutes": (self.machine_ttl or (None, None))[0],
                    "machine_ttl_from": (self.machine_ttl or (None, None))[1],
                    "admission": adm,
                    "grader_cap": self.grader, "machines_created": self.created, "machines_retired": self.retired,
                    "machines_rolled_over": self.rollovers,
                    "host_code_tree_sha256_at_start": manifest_sha(self.code_at_start),
                    "evolver_placement": self.placement, "domain_allow_list": self.allow_csv,
                    "orphans_deleted_at_init": list(self.reaped),
                    "creation_guard": {k: v for k, v in self.guard.items() if k != "last_logged"},
                    "lost_unit_requeues": list(self.lost_requeues),
                    "machine_failures": list(self.machine_failures),
                    "spend_unattributed_usd": round(self.unattributed_usd, 5),
                    "machines": {m.name: m.record() for m in self.machines}}

    def machine_hours(self) -> float:
        return sum(((m.t_end or time.time()) - m.t0) for m in self.machines) / 3600.0

    def leg_extra_daytona(self, leg: PoolLeg) -> dict:
        with self.lock:
            now = time.time()

            def slot(u):                                  # attempt-seconds a unit held its spot on a machine
                if not u.t_launch:
                    return 0.0
                return max(0.0, float(u.t_end or now) - float(u.t_launch)) * max(1, int(u.need))
            mine = sum(slot(u) for u in leg.units)
            every = sum(slot(u) for lg in self.legs.values() for u in lg.units)
            share = mine / every if every > 0 else 0.0
            hours = self.machine_hours()
            fresh = leg.metered_usd()
            rec = self.spend.record() if self.spend else {}
            ls = self.leg_spend(leg)
            touched = self._machines_of(leg)
            at_result = {m.name: {"sandbox_id": m.sid, "state": m.state_name(), "alive": not m.done.is_set(),
                                  "deleted": m.deleted, "failed": m.failed,
                                  "units_of_this_leg": sum(1 for u in leg.units if u.machine is m),
                                  "log_sync_waited_for": (leg.log_sync_wait or {}).get(m.name),
                                  "log_sync_ok_upto": m.log_sync_ok_upto, "logs_synced_utc": m.log_sync_utc}
                         for m in touched}
            lman = leg.src_manifest or {}
            src = {}
            for m in touched:
                mman = m.src_manifest or {}
                diff = sorted(k for k in set(lman) | set(mman) if lman.get(k) != mman.get(k))
                src[m.name] = {"machine_src_tree_sha256": manifest_sha(m.src_manifest),
                               "leg_src_tree_sha256_at_join": manifest_sha(leg.src_manifest),
                               "same": not diff, "files_differing": diff[:40], "n_differing": len(diff)}
        return {
            "pool": {"pool_dir": str(self.pdir), "attempt": leg.attempt, "legs_in_pool": list(self.leg_order),
                     "settings": dict(self.p), "admission_policy": self.p.get("pool_order"),
                     "cap_policy": cap_note(self.p),
                     "machine_ttl_minutes": (self.machine_ttl or (None, None))[0],
                     "machine_hours": round(hours, 3),
                     "machine_list_cost_usd": round(hours * L.DAYTONA_USD_PER_SANDBOX_HOUR, 3),
                     "machines": {m.name: {"sandbox_id": m.sid, "deleted": m.deleted, "failed": m.failed}
                                  for m in self.machines},
                     "machine_failures": list(self.machine_failures),
                     # the code this host LOADED at its start (it refuses a join once any of it
                     # differs on disk), not the files on disk when this result is written
                     "pool_driver_sha256": self.code_at_start.get("scripts/daytona/ssb_leg_pool.py"),
                     "unit_launcher_sha256": self.code_at_start.get(f"scripts/daytona/{UNIT_LAUNCH}"),
                     "pool_poll_sha256": self.code_at_start.get(f"scripts/daytona/{POLL_SCRIPT}"),
                     "pool_cmdwatch_sha256": self.code_at_start.get(f"scripts/daytona/{WATCH_SCRIPT}"),
                     "host_code": {"started_utc": self.started_utc, "host_pid": os.getpid(),
                                   "sha256_at_start": dict(self.code_at_start),
                                   "tree_sha256": manifest_sha(self.code_at_start),
                                   "drift_now": {k: list(v) for k, v in self.code_drift().items()},
                                   "note": "hashed when this host started; a join is refused once any differs"},
                     "machine_code": {m.name: dict(m.code_snapshot) for m in touched},
                     "evolver_placement": self.placement, "domain_allow_list": self.allow_csv,
                     # the LONG-LIVED pool
                     "scope": POOL_SCOPE, "leg_epoch": leg.epoch, "leg_key": leg.cfg.key,
                     "no_train_sibling": leg.no_train_sibling, "result_guarantee": RESULT_GUARANTEE,
                     "machines_at_result": at_result, "machine_src": src, "skills_sha256": leg.skills_sha,
                     "leg_admission": copy.deepcopy(self.admission["legs"].get(leg.cfg.tag)),
                     "keep_idle_s": self.p.get("pool_keep_idle_s")},
            "spend": {**rec, "pool_legs": list(self.leg_order), "attributed_leg_cost_usd": round(fresh, 5),
                      "leg": {**ls, "stopped": leg.fail_reason},
                      "measures": ("the LEG cap (leg_cap_usd) holds leg.measured_usd = max(leg.metered_usd: "
                                   "total_cost_usd of this leg's NEW cache records (= attributed_leg_cost_usd), "
                                   "leg.key_attributed_usd: its units' share of the org key usage by fresh responses "
                                   "per spend check); a breach stops admitting its units, running ones finish, the "
                                   "leg is FAILED. run_delta_usd = the key usage since the run origin (run_cap_usd), "
                                   "and the balance floor: a breach stops admitting every unit. pool_key_delta_usd = "
                                   "the key usage since THIS host started (every leg of the pool; no cap on it)")},
            # replaces LegRun._finish's on-disk hash of the loop driver: the one this host process loaded
            "driver_sha256": self.code_at_start.get("scripts/daytona/ssb_leg_daytona_loop.py"),
            "sandbox_hours": round(hours * share, 3),
            "daytona_list_cost_usd": round(hours * share * L.DAYTONA_USD_PER_SANDBOX_HOUR, 3),
            "sandbox_hours_basis": (f"pool machine-hours {hours:.3f} so far x this leg's share {share:.3f} of the "
                                    f"attempt-seconds (need x wall) the pool's units held so far"),
        }

    def _machine_hits_into_units(self, leg: PoolLeg) -> None:
        """A key found in a MACHINE's synced logs is a problem of every unit that ran there (the
        shards' secret_hits check in LegRun._finish then fails the leg)."""
        with self.lock:
            hits = {m.name: list(m.secret_hits) for m in self.machines if m.secret_hits}
            if not hits:
                return
            for u in leg.units:
                ran = {u.machine.name} if u.machine is not None else set()
                ran |= {a.get("machine") for a in u.summary.get("lost_attempts") or []}
                add = [h for n in sorted(ran) for h in hits.get(n, [])]
                if add:
                    u.summary["secret_hits"] = sorted(set(u.summary.get("secret_hits") or []) | set(add))

    def finish_leg(self, leg: PoolLeg, pre_abort=None) -> int:
        """Write this leg's result (or FAILED) and its DONE file. Checker fix 1: whatever fails while finishing ONE leg
        (its records, the pool blocks, the join lock) fails THAT leg only -- never the run-wide host and its other
        legs -- and the leg is always left finished with a DONE file (its follower exits)."""
        try:
            nonce = (leg.req or {}).get("nonce")
            jf = self.pdir / "joins" / f"{nonce}.join.json" if nonce else None
            if jf is None:
                return self._finish_leg_locked(leg, pre_abort, None)
            with _join_lock(jf):              # JOIN HAND-OVER: never finish OK a leg whose follower cancelled
                return self._finish_leg_locked(leg, pre_abort, jf)
        except Exception as e:  # noqa: BLE001
            self.log(f"finish of {leg.cfg.tag} crashed outside its record: {traceback.format_exc()[-1500:]}")
            return self._leg_failed_done(leg, f"pool finish crashed: {type(e).__name__}: {e}")

    def _finish_leg_locked(self, leg: PoolLeg, pre_abort, jf) -> int:
        try:
            if jf is not None and _cancel_path(jf).exists():
                why0 = "its follower gave up and cancelled the join (it recorded the leg FAILED)"
                pre_abort = why0 + (f"; {pre_abort}" if pre_abort else "")
            self._machine_hits_into_units(leg)
            lr = leg.legrun
            touched = self._machines_of(leg)
            # the LONG-LIVED pool: the result is written while the machine(s) may still serve other legs
            lr.unit_end_check = lambda s: self._unit_end_check(leg, s)
            lr.listing_ids = {m.sid for m in touched if m.sid and m.done.is_set()}
            lr.extra_daytona = self.leg_extra_daytona(leg)
            lr.extra_result = {"pool_ramp": self.pool_ramp_block()}
            why = pre_abort
            if why is None and self.abort_reason:
                why = self.abort_reason
            rc = lr._finish(leg.plan, leg.units, why)
        except Exception as e:  # noqa: BLE001 -- a finish bug must still end in a FAILED record + a DONE file
            self.log(f"finish of {leg.cfg.tag} crashed: {traceback.format_exc()[-1500:]}")
            if leg.cfg.result_path.exists() and not leg.cfg.failed_path.exists():
                rc = L.EXIT_OK                           # written (atomically, last): only a step after it failed
            else:
                return self._leg_failed_done(leg, f"pool finish crashed: {type(e).__name__}: {e}")
        return self._leg_done(leg, rc)

    def _leg_failed_done(self, leg: PoolLeg, why: str) -> int:
        """A FAILED record (unless a result or FAILED record exists already) and the DONE file."""
        try:
            if not leg.cfg.failed_path.exists() and not leg.cfg.result_path.exists():
                L._atomic_write(leg.cfg.failed_path, {"INCOMPLETE": True, "tag": leg.cfg.tag, "key": leg.cfg.key,
                                                      "problems": [why], "pool_dir": str(self.pdir),
                                                      "written_utc": dlx.now()})
        except Exception as e:  # noqa: BLE001
            self.log(f"FAILED record of {leg.cfg.tag} could not be written: {e}")
        return self._leg_done(leg, L.EXIT_OK if (leg.cfg.result_path.exists() and not leg.cfg.failed_path.exists())
                              else L.EXIT_LEG_FAILED)

    def _leg_done(self, leg: PoolLeg, rc: int) -> int:
        """The leg's DONE file (its follower exits on it) and state finished -- the latter ALWAYS, so neither the
        host's next tick nor _close finishes the leg a second time."""
        problems = []
        try:
            if rc != L.EXIT_OK and leg.cfg.failed_path.exists():
                problems = json.loads(leg.cfg.failed_path.read_text()).get("problems") or []
        except (OSError, ValueError, AttributeError):
            pass
        try:
            L._atomic_write(leg.done_file, {
                "tag": leg.cfg.tag, "attempt": leg.attempt, "rc": rc,
                "status": "ok" if rc == L.EXIT_OK else "failed",
                "result": str(leg.cfg.result_path) if rc == L.EXIT_OK else None,
                "failed": str(leg.cfg.failed_path) if rc != L.EXIT_OK else None,
                "problems": problems[:20], "utc": dlx.now()})
        except Exception as e:  # noqa: BLE001
            self.log(f"DONE file of {leg.cfg.tag} could not be written ({e}): its follower records the leg FAILED "
                     f"once this host is gone")
        finally:
            with self.lock:
                leg.state, leg.rc = "finished", rc
                a = self.admission["legs"].setdefault(leg.cfg.tag, {"epoch": leg.epoch, "key": leg_key(leg)})
                a["result_utc"], a["result_ts"], a["rc"] = dlx.now(), round(time.time(), 3), rc
                still = sum(1 for lg in self.legs.values() if lg.state != "finished")
        self.log(f"leg {leg.cfg.tag} FINISHED rc {rc} (epoch {leg.epoch}; its follower exits now; {still} other "
                 f"leg(s) still in the pool)")
        return rc

    # -- the host's life
    def run(self) -> int:
        r = self.reg.read()
        if int(r.get("host_pid") or 0) not in (0, os.getpid()) or \
                (r.get("pool_dir") and Path(r["pool_dir"]) != self.pdir):
            self.log(f"registry names another host ({r.get('host_pid')}, {r.get('pool_dir')}); handling only "
                     f"this dir's joins")
        host_tools.keep_awake(os.getpid())          # best effort: the host computer stays awake while this lives

        def on_sig(*_a):
            self._abort("SIGINT/SIGTERM to the pool host")
        try:
            signal.signal(signal.SIGINT, on_sig)
            signal.signal(signal.SIGTERM, on_sig)
        except ValueError:
            pass
        self.log(f"host pid {os.getpid()} started (scope: every leg of the run; registry {self.reg.root})")
        try:
            self._loop()
        except BaseException as e:  # noqa: BLE001
            self.problems.append(f"host crashed: {type(e).__name__}: {e}")
            self.log("HOST CRASH: " + traceback.format_exc()[-2000:])
            self._abort(f"host crashed: {type(e).__name__}: {e}")
        finally:
            self._close()
        return 0

    def _loop(self) -> None:
        tick = float(os.environ.get("EVOSKILL_POOL_TICK_S", "5"))
        while True:
            self.process_joins()
            self.process_aborts()
            if self.legs and self.spend is None:         # fail CLOSED, never run a leg unguarded
                self._abort("a leg is in the pool but the pool has no spend guard")
            if self.spend is not None:
                if time.time() - self.last_spend >= float(self.first_cfg.rt["spend_every_s"]):
                    self.spend.check("watchdog")
                    self.attribute_spend()
                    self.last_spend = time.time()
                if self.spend.breached.is_set() and not self.admit_stop:
                    self.drain_all("spend guard: " + self.spend.reason)
                self.check_leg_caps()
            if self.abort_all.is_set():
                with self.lock:
                    for lg in self.legs.values():
                        for u in lg.units:
                            if u.state == "queued":
                                u.state, u.t_end, u.rc = "ended", time.time(), "never launched"
                                u.summary["aborted"] = self.abort_reason
                    self.queue = []
                return
            if not self.legs:
                if time.time() - self.t0 > 300:
                    self.log("no leg joined within 300 s -- closing")
                    return
                time.sleep(tick)
                continue
            self.ensure_machines()
            self.dispatch()
            self.update_429()
            self.check_legs()
            self.release_idle()
            self.write_status()
            self.publish_machines()
            if self._should_close():
                with self.reg.lock():
                    if self._pending_joins():
                        continue
                    r = self.reg.read()
                    if Path(str(r.get("pool_dir") or "")) == self.pdir:
                        r.update(state="closed", closed_utc=dlx.now())
                        self.reg.write(r)
                self.log("every leg finished, no join pending, no machine left -- closing the pool")
                return
            time.sleep(tick)

    def _close(self) -> None:
        """Release every machine, wait for their threads (kill / drain / sync / delete), finish every open leg, write
        pool_record.json and machine.json. The registry says "closing" while machines may
        still be alive and "closed" only once every machine thread has ended -- attach() waits while the host lives,
        so no new host (and machine) starts beside this one's. Each step is guarded: a failing one is recorded, the
        rest still runs."""
        with self.reg.lock():
            r = self.reg.read()
            if Path(str(r.get("pool_dir") or "")) == self.pdir and r.get("state") not in ("closing", "closed"):
                r.update(state="closing", closing_utc=dlx.now())
                self.reg.write(r)
            late = self._pending_joins()
        for f in late:                               # joined after the close decision: refuse, it relaunches
            L._atomic_write(_ack_path(f), {"accepted": False, "reason": "the pool closed before this join was "
                                           "read; relaunch the leg", "utc": dlx.now()})
        for m in self.machines:
            m.release.set()
            m.wake.set()
        for th in self.threads:
            th.join(timeout=4 * 3600)
        alive = [th.name for th in self.threads if th.is_alive()]
        undeleted = [m.name for m in self.machines if m.sid and m.deleted is not True]
        with self.reg.lock():
            r = self.reg.read()
            if Path(str(r.get("pool_dir") or "")) == self.pdir:
                if alive:                            # a machine may still be alive: stay "closing" (attach waits)
                    r.update(state="closing", closing_problem=f"machine thread(s) {alive} still running after the "
                                                              f"join timeout")
                else:
                    r.update(state="closed", closed_utc=dlx.now(), machines_not_verified_deleted=undeleted)
                self.reg.write(r)
        if alive or undeleted:
            self.problems.append(f"at close: machine thread(s) still running {alive}; machine(s) not verified deleted "
                                 f"{undeleted} (the run's next pool host deletes whatever is still listed before it "
                                 f"creates a machine)")
        if self.spend is not None:
            try:
                self.spend.check("final")
                self.attribute_spend()
            except Exception as e:  # noqa: BLE001
                self.problems.append(f"final spend check failed: {type(e).__name__}: {e}")
        self.write_status()
        for lg in list(self.legs.values()):
            if lg.state != "finished":
                why = None
                if self.abort_reason:
                    why = self.abort_reason
                elif lg.abort_reason or lg.fail_reason:
                    why = lg.abort_reason or lg.fail_reason
                self.finish_leg(lg, pre_abort=why)

        def safe(fn, default=None):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001
                self.problems.append(f"pool record: {getattr(fn, '__name__', fn)} failed: {type(e).__name__}: {e}")
                return default
        rec = {"pool_dir": str(self.pdir), "closed_utc": dlx.now(), "abort": self.abort_reason,
               "admit_stop": self.admit_stop, "pool_ramp": safe(self.pool_ramp_block),
               "legs": {t: {"attempt": lg.attempt, "state": lg.state, "rc": lg.rc, "epoch": lg.epoch,
                            "key": leg_key(lg)} for t, lg in self.legs.items()},
               "spend": safe(lambda: self.spend.record() if self.spend else None),
               "legs_spend": safe(lambda: {t: self.leg_spend(lg) for t, lg in self.legs.items()}),
               "machine_hours": safe(lambda: round(self.machine_hours(), 3)),
               "machine_threads_alive_at_close": alive, "machines_not_verified_deleted": undeleted,
               "orphans_deleted_at_init": list(self.reaped), "creation_guard": dict(self.guard)}
        try:
            if self.client is not None and self.machines:
                ours = dlx.list_ours(self.client)
                rec["machines_still_listed"] = [o for o in ours if o["id"] in {m.sid for m in self.machines}]
        except Exception as e:  # noqa: BLE001
            rec["final_listing_error"] = f"{type(e).__name__}: {e}"
        rec["problems"] = self.problems
        L._atomic_write(self.pdir / "pool_record.json", rec)
        self.publish_machines(closed=True)
        self.log(f"pool closed: {json.dumps(rec['legs'])}")


# ───────────────────────────────────────────────────────────────────────────── commands
def cmd_host(a) -> int:
    return PoolHost(Path(a.pool_dir)).run()


def cmd_status(a) -> int:
    rd = Path(a.run_dir).resolve()
    rt = L.load_runtime_block(rd)
    root = Path(rt["store_root"]) / rd.name / "pool"
    if not root.is_dir():
        print(f"no pool dir under {root}")
        return 0
    for ep in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (ep / "registry.json").is_file():
            continue
        reg = Registry(ep)
        r = reg.read()
        scope = "RUN pool (every leg of the run)" if ep.name == POOL_SCOPE else "per-epoch pool"
        print(f"{ep.name} [{scope}]: state {r.get('state')} host {r.get('host_pid')} alive {host_alive(r)} dir "
              f"{r.get('pool_dir')}")
        try:
            mf = json.loads(reg.machines_file.read_text())
            print(f"  machine.json: current {mf.get('current')} (generation {mf.get('generation')}, host "
                  f"{mf.get('host_state')}); " + "; ".join(
                      f"{n} {d.get('sandbox_id')} {d.get('state')} retire_at "
                      f"{time.strftime('%H:%M', time.localtime(d['retire_at_ts'])) if d.get('retire_at_ts') else '-'}"
                      f"{' leased by ' + str(d.get('leased_by')) if d.get('leased_by') else ''}"
                      for n, d in (mf.get("machines") or {}).items()))
        except (OSError, ValueError, TypeError):
            pass
        try:
            st = json.loads((Path(r.get("pool_dir") or "") / "status.json").read_text())
        except (OSError, ValueError, TypeError):
            continue
        if st.get("admission"):
            print(f"  admission: {st['admission']}")
        if st.get("admit_stop"):
            print(f"  ADMITS NOTHING MORE: {st['admit_stop']}")
        print(f"  queue {st.get('queue')} unit(s); idle since {st.get('idle_since')}; keep idle "
              f"{st.get('keep_idle_s')}s")
        for t, lg in (st.get("legs") or {}).items():
            print(f"  {t}: {lg}")
        for n, m in (st.get("machines") or {}).items():
            print(f"  {n} [{m.get('state')}]: cap {m.get('cap')}/{m.get('ceiling')} in-flight {m.get('inflight')} "
                  f"units {m.get('units')} signals {m.get('signals')}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("host")
    p.add_argument("--pool-dir", required=True)
    p = sub.add_parser("status")
    p.add_argument("--run-dir", required=True)
    a = ap.parse_args(argv)
    return {"host": cmd_host, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
