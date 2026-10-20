#!/usr/bin/env python3
"""The evolver in a CONTAINER on the run's pool machine -- the shared half (RUNTIME evolver_host = "daytona" with
evolver_daytona.placement = "pool_container"; the driver is scripts/daytona/evolver_container.py, the machine side
scripts/daytona/evolver_container_ctl.py).

The run uses ONE machine: the evolver runs in an isolated container on the leg machine (whose disk holds the held-out
answers) rather than on a machine of its own, and the rollouts keep their concurrency.

WHAT STAYS THE SAME AS THE OWN-SANDBOX EVOLVER. The upload manifest, the receipt e{N}_EVOLVER_CONTAINMENT.json
(containment "daytona", same keys), the VM listing e{N}_EVOLVER_VM_LISTING.json, the log, the transcript dir, the
trace and the exit codes. daytona_evolver.check_upload_manifest() runs UNCHANGED: the listing is taken INSIDE the
container, as the unprivileged user, over /Users, /evo and /etc/claude-code, and must equal the manifest exactly.

WHAT CHANGES, and is checked here instead of daytona_evolver.check_daytona_receipt(): the machine is the pool's
(never created, never deleted by the evolver: resources and the create/delete ledger rows are the pool's), its
Daytona allow list is openrouter.ai + api.anthropic.com (the runner's and the evolver's), and containment is
proven from INSIDE the container: namespaces, credentials, capabilities, the exact mount table and top-level
listing, no machine path visible (canaries), no machine process visible, writes only where granted, the network
(only `lo`; api.anthropic.com through the allowlist proxy; openrouter.ai, the metadata IP, public DNS, the
machine's own addresses, loopback ports and abstract sockets unreachable), the proxy's own log, and a verified
teardown (no process left in the container's namespaces, cgroup removed, proxy gone, state wiped, no token prefix
in the container's writable state or tmpfs).

WHERE THE MACHINE COMES FROM (the run-wide long-lived pool): the pool host's own
<store_root>/<run>/pool/run/machine.json names its CURRENT machine; the driver takes it only when it is "ready",
its allow list carries api.anthropic.com and retire_at_ts leaves the evolver timeout + 30 min (pick_pool_machine,
waiting up to RUNTIME evolver_daytona.machine_wait_s for a replacement that is still starting), then writes
leases/<holder>.json in the pool's format, waits until the host lists it in leased_by (lease_acknowledged),
renews it while the container lives and removes it at the end (PoolLease). MACHINE_REGISTRY.json /
EVOLVER_LEASE.json (a stand-alone smoke registry, which no shipped command writes) are never a dispatch's machine
source: a receipt that names one fails the check.

STDLIB ONLY: imported by the loop's interpreter and by the driver.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PLACEMENT = "pool_container"
OWN_SANDBOX = "own_sandbox"
PLACEMENTS = (OWN_SANDBOX, PLACEMENT)
MACHINE_ALLOW = frozenset({"openrouter.ai", "api.anthropic.com"})
MACHINE_ALLOW_STR = "openrouter.ai,api.anthropic.com"
POOL_PREFIX = "evoskill-loop-"                # daytona_lib_loop.PREFIX: the pool's machines
SMOKE_PREFIX = "evoskill-evo-"                # daytona_lib_evo.PREFIX: a stand-alone smoke machine
API_HOST = "api.anthropic.com"
# A stand-alone smoke registry (no pool host exists there; no shipped command writes it). Never a dispatch's machine
# source: a dispatch reads the POOL's machine.json below.
REGISTRY_NAME = "MACHINE_REGISTRY.json"
REGISTRY_SCHEMA = "evoskill-pool-machines/1"
LEASE_NAME = "EVOLVER_LEASE.json"
# THE RUN-WIDE POOL'S registry (scripts/daytona/ssb_leg_pool.py Registry; mirrored here, the loop's interpreter
# must not import the pool -- tests/test_evolver_container.py asserts these equal the pool's own constants):
#   <store_root>/<run>/pool/run/machine.json  {"schema": POOL_MACHINES_SCHEMA, "current": <name>|null, "machines":
#       {<name>: {sandbox_id, sandbox_name, state (starting|ready|retiring|released|deleted|failed), ttl_expires_ts,
#       retire_at_ts, domain_allow_list, leased_by, ...}}, "host_pid", "host_state" ("open"|"closed"), "generation"}
#   <store_root>/<run>/pool/run/leases/<holder>.json  {"holder", "sandbox_id", "until_ts", "pid"}: while until_ts is
#       in the future and pid alive, the pool host never releases that machine for idleness or retirement (never
#       past ttl_expires_ts - POOL_SYNC_MARGIN_MIN); the holder checks retire_at_ts before starting.
POOL_SCOPE = "run"
POOL_MACHINES_FILE = "machine.json"
POOL_MACHINES_SCHEMA = "evoskill.pool_machines/1"
POOL_LEASES_DIR = "leases"
POOL_SYNC_MARGIN_MIN = 15                     # ssb_leg_daytona_loop.POOL_SYNC_MARGIN_MIN
EVOLVER_MARGIN_S = 1800                       # the machine must outlive the evolver's timeout by 30 min
MACHINE_WAIT_S = 900                          # RUNTIME evolver_daytona.machine_wait_s default (0 = refuse at once)
MACHINE_POLL_S = 15.0
LEASE_ACK_S = 120.0                           # the pool host lists the lease in machine.json within a tick (5 s)
LEASE_WINDOW_S = 900.0                        # until_ts = now + this, renewed every LEASE_RENEW_S while held
LEASE_RENEW_S = 60.0
DRIVER = "evolver_container"
CTL_FILE = REPO / "scripts/daytona/evolver_container_ctl.py"
ZERO_CAPS = "0000000000000000"
DEV_ALLOWED = ("fd", "full", "null", "random", "shm", "stderr", "stdin", "stdout", "tty", "urandom", "zero")


def load_ctl():
    """scripts/daytona/evolver_container_ctl.py as a module (one definition of the plan, for driver and check)."""
    name = "evolver_container_ctl"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, CTL_FILE)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# ───────────────────────────────────────────────────────────────────────────── RUNTIME
def placement(cfg: dict | None) -> str:
    """RUNTIME.json evolver_daytona.placement: absent -> "own_sandbox" (today's own Daytona VM); "pool_container" ->
    this module. Anything else raises ValueError (the loop HALTs on it, as on a bad evolver_host)."""
    v = (cfg or {}).get("placement", OWN_SANDBOX)
    if v not in PLACEMENTS:
        raise ValueError(f"RUNTIME.json evolver_daytona.placement={v!r} is not one of {PLACEMENTS}")
    return v


def container_limits_cfg(cfg: dict | None) -> dict:
    """RUNTIME evolver_daytona.container = {cpus, mem_max_gib, pids_max, tmp_gib, rw_max_gib, cpu_weight,
    mem_high_ratio} (all optional) -> the controller's limit names. Unknown keys raise ValueError. cpu_weight (1..10000)
    is the evolver cgroup's share against its sibling rollout cgroups (100 each): the evolver may take the whole
    machine (all its CPUs and memory) with a much higher weight than the rollouts beside it."""
    c = (cfg or {}).get("container") or {}
    if not isinstance(c, dict):
        raise ValueError("RUNTIME.json evolver_daytona.container must be an object")
    known = {"cpus", "mem_max_gib", "pids_max", "tmp_gib", "rw_max_gib", "cpu_weight", "mem_high_ratio"}
    bad = sorted(set(c) - known - {k for k in c if k.startswith("_")})
    if bad:
        raise ValueError(f"RUNTIME.json evolver_daytona.container: unknown keys {bad} (known {sorted(known)})")
    out = {}
    if c.get("cpus") is not None:
        out["cpus"] = float(c["cpus"])
    if c.get("mem_max_gib") is not None:
        out["mem_max_bytes"] = int(float(c["mem_max_gib"]) * (1 << 30))
    if c.get("pids_max") is not None:
        out["pids_max"] = int(c["pids_max"])
    if c.get("tmp_gib") is not None:
        out["tmp_bytes"] = int(float(c["tmp_gib"]) * (1 << 30))
    if c.get("rw_max_gib") is not None:
        out["rw_max_bytes"] = int(float(c["rw_max_gib"]) * (1 << 30))
    if c.get("cpu_weight") is not None:
        w = c["cpu_weight"]
        if isinstance(w, bool) or not isinstance(w, int) or not 1 <= w <= 10000:
            raise ValueError(f"RUNTIME.json evolver_daytona.container.cpu_weight must be an integer 1..10000, got {w!r}")
        out["cpu_weight"] = w
    if c.get("mem_high_ratio") is not None:
        r = c["mem_high_ratio"]
        if isinstance(r, bool) or not isinstance(r, (int, float)) or not 0.5 <= float(r) <= 1.0:
            raise ValueError(f"RUNTIME.json evolver_daytona.container.mem_high_ratio must be in 0.5..1.0, got {r!r}")
        out["mem_high_ratio"] = float(r)
    return out


def machine_wait_s(cfg: dict | None) -> float:
    """RUNTIME evolver_daytona.machine_wait_s (absent = MACHINE_WAIT_S): how long the driver waits for the pool's
    current machine to pass pick_pool_machine() before it REFUSES. The pool replaces an idle machine that is short
    of time (pool_idle_rollover_min) right when a train leg ends -- exactly when the evolver starts -- so the new
    machine is `starting` for a few minutes; 0 = refuse at once. ValueError outside 0..3600 s."""
    v = (cfg or {}).get("machine_wait_s", MACHINE_WAIT_S)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or not 0 <= v <= 3600:
        raise ValueError(f"RUNTIME.json evolver_daytona.machine_wait_s must be a number of seconds in 0..3600, "
                         f"got {v!r}")
    return float(v)


# ───────────────────────────────────────────────────────────────────────────── the pool's machine registry
def registry_dir(run_dir, runtime: dict | None = None) -> Path:
    """<daytona_leg.store_root or .cache/ssbv2_daytona>/<run>/pool (the leg driver's store root default)."""
    run_dir = Path(run_dir)
    if runtime is None:
        try:
            runtime = json.loads((run_dir / "RUNTIME.json").read_text())
        except (OSError, ValueError):
            runtime = {}
    store = ((runtime or {}).get("daytona_leg") or {}).get("store_root") or str(REPO / ".cache" / "ssbv2_daytona")
    return Path(store) / run_dir.name / "pool"


def pool_run_dir(run_dir, runtime: dict | None = None) -> Path:
    """<store_root>/<run>/pool/run: the RUN's pool (ssb_leg_pool.pool_root)."""
    return registry_dir(run_dir, runtime) / POOL_SCOPE


def pool_machines_file(run_dir, runtime: dict | None = None) -> Path:
    """The pool host's machine.json -- where a production dispatch finds its machine."""
    return pool_run_dir(run_dir, runtime) / POOL_MACHINES_FILE


def _num(v):
    """A finite float, or None (a registry value that is absent, null, non-numeric or inf)."""
    if isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def pid_alive(pid) -> bool:
    """ssb_leg_pool._pid_alive: the pool host and the driver run on the same host."""
    try:
        p = int(pid)
        if p <= 0:
            return False
        os.kill(p, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, ValueError, TypeError, OverflowError):
        return False
    return True


def pick_pool_machine(rec, *, now: float, need_s: float, sandbox_id: str | None = None,
                      alive=pid_alive) -> tuple:
    """(row, why) from the pool's machine.json: its CURRENT machine, or None and every reason it may not host the
    container. Refused: another schema; a closed or dead pool host (nobody would honour the lease); no current
    machine; state not "ready"; a domain_allow_list without api.anthropic.com; retire_at_ts - now < need_s (the
    evolver timeout + 30 min: a lease past retire_at_ts holds up the machine's replacement, and queued rollouts
    wait) or ttl_expires_ts - POOL_SYNC_MARGIN_MIN - now < need_s (the pool honours no lease past that). The row
    comes back with its machine.json name as "name"."""
    if not isinstance(rec, dict) or rec.get("schema") != POOL_MACHINES_SCHEMA:
        return None, (f"machine.json schema {rec.get('schema') if isinstance(rec, dict) else None!r} is not "
                      f"{POOL_MACHINES_SCHEMA!r}")
    if rec.get("host_state") != "open":
        return None, f"the pool host is {rec.get('host_state')!r}, not 'open'"
    if not alive(rec.get("host_pid")):
        return None, f"the pool host (pid {rec.get('host_pid')!r}) is not alive: nobody would honour a lease"
    ms = rec.get("machines")
    if not isinstance(ms, dict):
        return None, "machine.json 'machines' is not an object"
    cur = rec.get("current")
    if not cur:
        states = {str(n): (r or {}).get("state") for n, r in ms.items() if isinstance(r, dict)}
        return None, f"the pool has no current machine (machines {states or 'none'})"
    row = ms.get(cur)
    if not isinstance(row, dict) or not row.get("sandbox_id"):
        return None, f"the current machine {cur!r} has no row with a sandbox_id"
    sid = str(row["sandbox_id"])
    if sandbox_id and sid != sandbox_id:
        return None, f"the current machine {cur} is {sid[:8]}, not the requested {str(sandbox_id)[:8]}"
    why = []
    if row.get("state") != "ready":
        why.append(f"state {row.get('state')!r}, not 'ready'")
    if API_HOST not in allow_list_set(row.get("domain_allow_list")):
        why.append(f"domain_allow_list {row.get('domain_allow_list')!r} lacks {API_HOST}")
    ra, te = _num(row.get("retire_at_ts")), _num(row.get("ttl_expires_ts"))
    if ra is None:
        why.append(f"retire_at_ts {row.get('retire_at_ts')!r} (the machine's working life is unknown)")
    elif ra - now < need_s:
        why.append(f"{int(ra - now)} s to retire_at_ts < {int(need_s)} s (evolver timeout + 30 min)")
    if te is None:
        why.append(f"ttl_expires_ts {row.get('ttl_expires_ts')!r}")
    elif te - POOL_SYNC_MARGIN_MIN * 60 - now < need_s:
        why.append(f"{int(te - POOL_SYNC_MARGIN_MIN * 60 - now)} s of honoured lease left < {int(need_s)} s")
    if why:
        return None, f"current machine {cur} ({sid[:8]}): " + "; ".join(why)
    return dict(row, name=str(cur)), (f"current machine {cur} ({sid[:8]}): ready, {int((ra - now) / 60)} min to "
                                      f"retire_at, allow {row.get('domain_allow_list')!r}, generation "
                                      f"{rec.get('generation')!r}")


def lease_holder(cid: str) -> str:
    """leases/<holder>.json: one holder per dispatch (the container id is unique)."""
    h = f"evolver-{cid}"
    if not h.replace("-", "").replace("_", "").isalnum():
        raise ValueError(f"lease holder {h!r} is not a plain file name")
    return h


def lease_acknowledged(rec, *, name: str, sandbox_id: str, holder: str) -> tuple:
    """(verdict, why) from a machine.json read AFTER the lease was written: True = the pool host lists `holder`
    in that machine's leased_by and the machine is still "ready" (every later release_idle reads the lease);
    False = the machine is no longer usable (released / retiring / gone / another sandbox / host closed);
    None = the host has not published the lease yet."""
    if not isinstance(rec, dict) or rec.get("schema") != POOL_MACHINES_SCHEMA:
        return False, "machine.json is unreadable or of another schema"
    if rec.get("host_state") != "open":
        return False, f"the pool host is {rec.get('host_state')!r}"
    row = (rec.get("machines") or {}).get(name) if isinstance(rec.get("machines"), dict) else None
    if not isinstance(row, dict) or str(row.get("sandbox_id")) != str(sandbox_id):
        return False, f"machine {name} ({str(sandbox_id)[:8]}) is no longer in machine.json"
    if row.get("state") != "ready":
        return False, f"machine {name} is now {row.get('state')!r}"
    if holder in (row.get("leased_by") or []):
        return True, f"machine {name} leased by {holder} (generation {rec.get('generation')!r})"
    return None, f"machine {name}: leased_by {row.get('leased_by')!r} does not list {holder} yet"


def write_json_atomic(path: Path, obj: dict) -> None:
    """tmp beside it + os.replace: the pool host reads leases/*.json while we write (the tmp name is no *.json)."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as fh:
        fh.write(json.dumps(obj, indent=1))
        fh.flush()
        os.fsync(fh.fileno())
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


class PoolLease:
    """leases/<holder>.json in the POOL's format ({holder, sandbox_id, until_ts, pid} + informational keys the pool
    ignores), renewed from a daemon thread every LEASE_RENEW_S to now + LEASE_WINDOW_S (never past cap_ts), and
    removed by release() -- only while it is still ours. The pid makes a crashed driver's lease void at once."""

    def __init__(self, leases_dir, *, holder: str, sandbox_id: str, cap_ts: float, info: dict | None = None,
                 pid: int | None = None, window_s: float | None = None, every_s: float | None = None):
        self.dir = Path(leases_dir)
        self.path = self.dir / f"{holder}.json"
        self.holder, self.sandbox_id, self.cap_ts = holder, str(sandbox_id), float(cap_ts)
        self.pid = int(pid if pid is not None else os.getpid())
        self.window_s = float(LEASE_WINDOW_S if window_s is None else window_s)
        self.every_s = float(LEASE_RENEW_S if every_s is None else every_s)
        self.info = dict(info or {})
        self.writes, self.errors, self.until_ts = 0, [], None
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    def record(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        until = math.floor(min(now + self.window_s, self.cap_ts) * 1000) / 1000      # never past the cap
        return {**self.info, "holder": self.holder, "sandbox_id": self.sandbox_id, "until_ts": until,
                "pid": self.pid, "driver": DRIVER, "written_ts": round(now, 3)}

    def write(self) -> dict:
        with self._lock:
            if self._stop.is_set() and self.writes:
                return {}
            rec = self.record()
            self.dir.mkdir(exist_ok=True)
            write_json_atomic(self.path, rec)
            self.writes += 1
            self.until_ts = rec["until_ts"]
            return rec

    def _renew(self) -> None:
        while not self._stop.wait(self.every_s):
            try:
                self.write()
            except OSError as e:
                self.errors.append(f"{type(e).__name__}: {e}"[:200])

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._renew, name=f"lease-{self.holder}", daemon=True)
            self._thread.start()

    def release(self) -> bool:
        """Stop renewing and remove the lease file if it is still this holder's. True = gone."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        with self._lock:
            try:
                d = json.loads(self.path.read_text())
            except FileNotFoundError:
                return True
            except (OSError, ValueError):
                return False
            if d.get("holder") != self.holder or int(d.get("pid") or 0) != self.pid:
                return False
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
            return not self.path.exists()

    def summary(self) -> dict:
        return {"path": str(self.path), "holder": self.holder, "sandbox_id": self.sandbox_id, "pid": self.pid,
                "until_ts": self.until_ts, "cap_ts": self.cap_ts, "window_s": self.window_s,
                "renew_every_s": self.every_s, "writes": self.writes, "errors": self.errors[:5]}


# ───────────────────────────────────────────────────────────────────────────── the operator smoke's registry
def parse_registry(obj) -> tuple:
    """(machines, problems) of the SMOKE registry (REGISTRY_SCHEMA; `smoke-machine up`). A machine row:
    sandbox_id, sandbox_name, state, created_ts, ttl_minutes, domain_allow_list, evolver_container (bool),
    snapshot_ref."""
    probs = []
    if not isinstance(obj, dict) or obj.get("schema") != REGISTRY_SCHEMA:
        return [], [f"registry schema is not {REGISTRY_SCHEMA!r}"]
    rows = obj.get("machines")
    if not isinstance(rows, list):
        return [], ["registry 'machines' is not a list"]
    out = []
    for r in rows:
        if not isinstance(r, dict) or not r.get("sandbox_id"):
            probs.append(f"registry row without a sandbox_id: {str(r)[:80]}")
            continue
        out.append(r)
    return out, probs


def allow_list_set(s) -> frozenset:
    """A Daytona allow list as a set: the SDK's comma string, or machine.json's list (ssb_leg_pool.allow_domains)."""
    items = s if isinstance(s, (list, tuple)) else str(s or "").split(",")
    return frozenset(str(x).strip().lower() for x in items if str(x).strip())


_allow_set = allow_list_set


def pick_machine(reg, *, now: float, need_s: float, sandbox_id: str | None = None) -> tuple:
    """(row, why) from the SMOKE registry. The ONE machine to host the container: state "ready", evolver_container
    true, the two-domain allow list, and at least need_s of its Daytona TTL left. With sandbox_id, that machine only.
    Several candidates: the one with the most TTL left (every candidate is listed in `why`)."""
    rows, probs = parse_registry(reg)
    if probs and not rows:
        return None, "; ".join(probs)
    cands, notes = [], []
    for r in rows:
        if sandbox_id and r.get("sandbox_id") != sandbox_id:
            continue
        left = (_num(r.get("created_ts")) or 0.0) + (_num(r.get("ttl_minutes")) or 0.0) * 60 - now
        why = []
        if r.get("state") != "ready":
            why.append(f"state {r.get('state')!r}")
        if r.get("evolver_container") is not True:
            why.append("evolver_container not true")
        if _allow_set(r.get("domain_allow_list")) != MACHINE_ALLOW:
            why.append(f"allow list {r.get('domain_allow_list')!r}")
        if left < need_s:
            why.append(f"{int(left)} s of TTL left < {int(need_s)}")
        if why:
            notes.append(f"{str(r.get('sandbox_id'))[:8]}: " + ", ".join(why))
        else:
            cands.append((left, r))
    if not cands:
        return None, ("no usable pool machine" + (f" ({'; '.join(notes)})" if notes else " (registry lists none)"))
    cands.sort(key=lambda x: -x[0])
    return cands[0][1], (f"{len(cands)} candidate(s); chose {str(cands[0][1]['sandbox_id'])[:8]}"
                         + (f"; skipped {'; '.join(notes)}" if notes else ""))


def machine_problems(rb: dict, *, name: str, state: str, row: dict | None, require_snapshot_ref: str | None,
                     prefixes=(POOL_PREFIX,)) -> list:
    """A FRESH readback of the pool machine (daytona_lib_evo.readback + name/state) against what the container
    needs: the pool's prefix, started, the two-domain allow list and nothing else, the leg snapshot."""
    probs = []
    if not str(name).startswith(tuple(prefixes)):
        probs.append(f"machine name {name!r} has none of the prefixes {list(prefixes)}")
    if "started" not in str(state).lower():
        probs.append(f"machine state {state!r} is not started")
    if not rb.get("refreshed"):
        probs.append("readback was not a fresh read")
    if _allow_set(rb.get("domain_allow_list")) != MACHINE_ALLOW:
        probs.append(f"domain_allow_list {rb.get('domain_allow_list')!r} != {MACHINE_ALLOW_STR!r}")
    if rb.get("network_allow_list") not in (None, ""):
        probs.append(f"network_allow_list set ({rb.get('network_allow_list')!r})")
    if rb.get("network_block_all"):
        probs.append("network_block_all is on")
    if require_snapshot_ref and rb.get("snapshot_ref") != require_snapshot_ref:
        probs.append(f"snapshot {rb.get('snapshot_ref')} != the leg image's {require_snapshot_ref}")
    if row is not None and row.get("snapshot_ref") and rb.get("snapshot_ref") != row.get("snapshot_ref"):
        probs.append("snapshot differs from the registry's")
    return probs


def lease_record(*, sandbox_id: str, cid: str, run: str, epoch: int, pid: int, expires_ts: float) -> dict:
    """The SMOKE machine's EVOLVER_LEASE.json (no pool host reads it; a production dispatch writes PoolLease)."""
    return {"schema": "evoskill-evolver-lease/1", "sandbox_id": sandbox_id, "cid": cid, "run": run,
            "epoch": epoch, "driver_pid": pid, "driver": DRIVER, "written_ts": time.time(),
            "expires_ts": expires_ts}


def lease_live(lease: dict, *, now: float, pid_alive) -> bool:
    """A SMOKE lease record is live: the driver's pid is alive and the lease has not expired."""
    try:
        return bool(lease.get("sandbox_id")) and float(lease.get("expires_ts") or 0) > now and \
            bool(pid_alive(int(lease.get("driver_pid") or 0)))
    except (TypeError, ValueError):
        return False


# ───────────────────────────────────────────────────────────────────────────── the probe verdicts
NET_KEYS = ("net",)


# the kernel's per-namespace fallback tunnel devices (container_probe_problems, network)
FALLBACK_TUNNEL_IFACES = frozenset({"gre0", "gretap0", "erspan0", "ip6gre0", "ip6tnl0", "tunl0", "sit0", "ip_vti0",
                                    "ip6_vti0"})


def container_probe_problems(pre: dict, *, expected_top, expected_mounts, rw_paths, ro_rebind=(),
                             require_libreoffice: str | None = None) -> list:
    """Every containment claim the PRE probe (run inside the container, as evo, before the upload) must show.
    Returns problems (empty = contained). Also the loop's re-check of the recorded probe."""
    from src.orchestrator import daytona_evolver as de
    ctl = load_ctl()
    P = []
    I, V, W, X, N = (pre.get(k) or {} for k in ("identity", "view", "writes", "priv", "net"))
    H = pre.get("host") or {}
    # who, with what
    if (I.get("uid"), I.get("euid"), I.get("gid")) != (ctl.EVO_UID, ctl.EVO_UID, ctl.EVO_GID) or \
            I.get("groups") not in ([], [ctl.EVO_GID]):
        P.append(f"identity: uid/euid/gid/groups {I.get('uid')}/{I.get('euid')}/{I.get('gid')}/{I.get('groups')}")
    caps = I.get("caps") or {}
    if [k for k in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb") if caps.get(k) != ZERO_CAPS]:
        P.append(f"identity: capabilities not all zero {caps}")
    if str(I.get("no_new_privs")) != "1":
        P.append("identity: no_new_privs is not set")
    if I.get("cc_sha256") != ctl.CC_SHA256:          # the pool pin (evolver_container_ctl; EVOSKILL_CC_VERSION, default 2.1.281)
        P.append(f"identity: Claude Code sha {str(I.get('cc_sha256'))[:12]} != pinned")
    if not str(I.get("cc_version") or "").startswith(ctl.CC_VERSION):
        P.append(f"identity: claude --version {I.get('cc_version')!r}")
    if I.get("claude_on_path") != ctl.CC_CONTAINER or I.get("files_named_claude") != [ctl.CC_CONTAINER]:
        P.append(f"identity: claude on PATH {I.get('claude_on_path')!r}, files named claude "
                 f"{I.get('files_named_claude')!r}")
    if I.get("docker_on_path"):
        P.append(f"identity: docker is on PATH ({I.get('docker_on_path')})")
    if not str(I.get("solver_stack") or "").startswith("3.1.3 2.2.0 1.26.4 3.11."):
        P.append(f"identity: solver stack {I.get('solver_stack')!r}")
    if I.get("soffice_convert_ok") is not True:
        P.append("identity: LibreOffice could not convert a workbook in the container")
    if require_libreoffice and I.get("libreoffice") != require_libreoffice:
        P.append(f"identity: LibreOffice {I.get('libreoffice')!r} != {require_libreoffice!r}")
    if I.get("opt_entries") != ["claude-code"]:
        P.append(f"identity: /opt holds {I.get('opt_entries')!r} (the grader's tooling must be absent)")
    if I.get("users_files") != 0:
        P.append(f"identity: /Users held {I.get('users_files')!r} file(s) before the upload")
    if I.get("secret_env_names"):
        P.append(f"identity: secret-shaped env names {I.get('secret_env_names')}")
    want_px = {k: ctl.PROXY_URL for k in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy")}
    want_px.update({"NO_PROXY": "", "no_proxy": ""})
    if I.get("proxy_env") != want_px:
        P.append(f"identity: proxy env {I.get('proxy_env')!r}")
    if I.get("evo_lo_profile_owner") != "evo":
        P.append(f"identity: LibreOffice profile owner {I.get('evo_lo_profile_owner')!r}")
    if I.get("policy_dir_entries") != 0:
        P.append(f"identity: /etc/claude-code held {I.get('policy_dir_entries')!r} before the upload")
    # what is visible
    if V.get("top") != list(expected_top):
        P.append(f"view: / = {V.get('top')} != {list(expected_top)}")
    for k, want in (("home", ["evo"]), ("opt", ["claude-code"]), ("var", ["tmp"]), ("run", [])):
        if V.get(k) != want:
            P.append(f"view: /{k} = {V.get(k)!r} != {want!r}")
    if sorted(V.get("dev") or []) != sorted(DEV_ALLOWED):
        P.append(f"view: /dev = {V.get('dev')!r}")
    top_mount = {}
    for m in V.get("mounts") or []:
        top_mount[m.get("point")] = m
    want_pts = {p for p, _ in expected_mounts} | {"/"}
    extra = sorted(set(top_mount) - want_pts - set(ctl.PROC_MASKS))
    missing = sorted(want_pts - set(top_mount))
    if extra or missing:
        P.append(f"view: mount table differs from the plan (extra {extra[:4]}, missing {missing[:4]})")
    wrong_ro = [p for p, ro in list(expected_mounts) + [("/", True)] if p in top_mount and top_mount[p].get("ro") != ro]
    if wrong_ro:
        P.append(f"view: read-only flag differs from the plan at {wrong_ro[:4]}")
    if not V.get("canaries_checked") or V.get("canaries_visible_n") != 0:
        P.append(f"view: {V.get('canaries_visible_n')!r} machine path(s) visible of {V.get('canaries_checked')!r} "
                 f"checked: {(V.get('canaries_visible') or [])[:3]}")
    if V.get("host_pid_matches"):
        P.append(f"view: machine processes visible {V.get('host_pid_matches')[:3]}")
    procs = V.get("procs") or []
    if len(procs) > 16 or [p for p in procs if p.get("uid") not in (ctl.EVO_UID, ctl.REAPER_UID)]:
        P.append(f"view: {len(procs)} process(es) visible, uids {sorted({p.get('uid') for p in procs})}")
    mns, cns = H.get("machine_ns") or {}, V.get("ns") or {}
    shared = [n for n in ("mnt", "pid", "net", "ipc", "uts") if not cns.get(n) or cns.get(n) == mns.get(n)]
    if shared:
        P.append(f"view: namespaces shared with the machine (or unknown): {shared}")
    # writes
    want_rw = {"/tmp", "/var/tmp", ctl.EVO_HOME, ctl.VM_WORK, ctl.VM_OUT, *rw_paths}
    if set((W.get("rw") or {})) != want_rw or [d for d, v in (W.get("rw") or {}).items() if v != "WRITABLE"]:
        P.append(f"writes: granted dirs not exactly writable: {W.get('rw')}")
    if not W.get("ro") or [d for d, v in (W.get("ro") or {}).items() if v == "WRITABLE"]:
        P.append(f"writes: a read-only dir was writable: {W.get('ro')}")
    for d in ro_rebind:
        if (W.get("ro") or {}).get(d) in (None, "WRITABLE"):
            P.append(f"writes: re-bound read-only root {d} not proven read-only")
    # privilege
    for k in ("setuid0", "chroot", "kill_pid1", "ptrace_pid1", "proc_sys_write"):
        if X.get(k) in (None, "SUCCEEDED"):
            P.append(f"priv: {k} {X.get(k)!r}")
    for k in ("mount_rc", "umount_users_rc"):
        if X.get(k) in (None, 0):
            P.append(f"priv: {k} {X.get(k)!r}")
    for k in ("pid1_environ", "kcore"):
        v = str(X.get(k) or "")
        if v.startswith("READ") and v != "READ 0 B":          # the /dev/null mask reads as 0 bytes
            P.append(f"priv: {k} readable ({v})")
    if str(X.get("pid1_environ") or "").startswith("READ"):
        P.append("priv: PID 1's environment is readable")
    if X.get("userns_rc") == 0 and X.get("userns_view_equal") is not True:
        P.append("priv: a user namespace sees a different filesystem")
    # network. A fresh network namespace also holds the kernel's fallback tunnel devices whenever the host
    # has the tunnel modules loaded (ip_gre, ip6_gre, ip6_tunnel, ipip, sit, ip_vti, ip6_vti; sysctl
    # net.core.fb_tunnels_only_for_init_net 0): some Daytona machines show all nine, others none. They are accepted by name only, with no route in the namespace and no IPv6 address on them; the reach
    # checks below (proxy-less, raw TCP / UDP, IPv6, metadata, DNS) are unchanged.
    ifaces = list(N.get("ifaces") or [])
    extra = [i for i in ifaces if i != "lo"]
    v6_off_lo = [i for i in (N.get("inet6") or []) if i != "lo"]
    if "lo" not in ifaces or [i for i in extra if i not in FALLBACK_TUNNEL_IFACES] or N.get("routes") != 0 \
            or (extra and v6_off_lo):
        P.append(f"net: interfaces {N.get('ifaces')!r}, routes {N.get('routes')!r}"
                 + (f", IPv6 addresses on {v6_off_lo}" if extra and v6_off_lo else ""))
    reach = {r.get("url"): r.get("reachable") for r in N.get("http") or []}
    for u in de.MUST_REACH:
        if reach.get(u) is not True:
            P.append(f"net: {u} not reachable through the proxy")
    leaked = [u for u in de.MUST_REFUSE if reach.get(u) is not False]
    if leaked:
        P.append(f"net: not refused (or not probed): {leaked[:4]}")
    if not N.get("direct") or [r for r in N["direct"] if r.get("reached") is not False]:
        P.append(f"net: a proxy-less request reached {[r.get('target') for r in N.get('direct') or [] if r.get('reached') is not False][:3]}")
    raw = N.get("raw") or []
    need_raw = {"1.1.1.1:443", "8.8.8.8:443", "169.254.169.254:80", "example.com:22",
                "[2606:4700:4700::1111]:443", "udp 1.1.1.1:53", "udp 8.8.8.8:53"}
    if not need_raw <= {r.get("target") for r in raw}:
        P.append(f"net: raw targets not probed {sorted(need_raw - {r.get('target') for r in raw})}")
    open_ = [r.get("target") for r in raw if r.get("tcp_connected") is True or r.get("answered") is True]
    if open_:
        P.append(f"net: reachable without the proxy: {open_[:4]}")
    if [a for a in N.get("abstract") or [] if a.get("connected")]:
        P.append("net: a machine abstract socket was reachable")
    if [h for h, v in (N.get("dns") or {}).items() if not str(v).startswith("FAILED")]:
        P.append(f"net: DNS resolved inside the container {N.get('dns')}")
    px = H.get("proxy") or {}
    if px.get("listen_netns") != cns.get("net") or px.get("allow") != [ctl.ALLOW]:
        P.append(f"net: proxy not listening in the container's namespace with allow [{ctl.ALLOW}] ({px})")
    if not str(H.get("here_string_fd") or "").startswith("pipe:"):
        P.append(f"token: bash here-strings are not pipes on this machine ({H.get('here_string_fd')!r})")
    ev = H.get("episode_view")
    if ev and (ev.get("opt_entries") != "0" or ev.get("evolver_dir_visible") != "0"):
        P.append(f"episode view: a solver episode sees /opt entries {ev}")
    return P


def post_probe_problems(pre: dict, post: dict) -> list:
    """The END of the run: the same mount table (points + ro), the same top level, still only `lo`."""
    P = []
    a = {m["point"]: m.get("ro") for m in (pre.get("view") or {}).get("mounts") or []}
    b = {m["point"]: m.get("ro") for m in (post.get("view") or {}).get("mounts") or []}
    if a != b:
        P.append(f"post: mount table changed during the run (+{sorted(set(b) - set(a))[:3]} "
                 f"-{sorted(set(a) - set(b))[:3]} ro {[p for p in a if p in b and a[p] != b[p]][:3]})")
    if (pre.get("view") or {}).get("top") != (post.get("view") or {}).get("top"):
        P.append("post: the top-level listing changed during the run")
    pi = list((post.get("view") or {}).get("ifaces") or [])
    pre_i = list((pre.get("net") or {}).get("ifaces") or ["lo"])
    if "lo" not in pi or [i for i in pi if i != "lo" and i not in FALLBACK_TUNNEL_IFACES] or sorted(pi) != sorted(pre_i):
        P.append(f"post: interfaces {pi!r} (before the run {pre_i!r})")   # only lo + the kernel's fallback tunnels,
    return P                                                               # and none added during the run


# ───────────────────────────────────────────────────────────────────────────── the loop's receipt check
def ledger_container_down(ledger_path, *, sandbox_id: str, cid: str) -> bool:
    try:
        for line in open(ledger_path):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("event") == "container_down" and r.get("driver") == DRIVER and r.get("id") == sandbox_id \
                    and r.get("cid") == cid and r.get("verified") is True:
                return True
    except OSError:
        return False
    return False


def check_pool_container_receipt(receipt: dict, *, log, ledger_path=None, require_libreoffice: str | None = None,
                                 require_snapshot_ref: str | None = None, staging=None,
                                 prefixes=(POOL_PREFIX,), session: str = "per_epoch", rw_paths=None,
                                 teardown: dict | None = None, probe_pre_sha256: str | None = None) -> tuple:
    """(ok, detail): the #294 half for an evolver that ran in a container on the pool machine. Replaces
    daytona_evolver.check_daytona_receipt for placement "pool_container"; check_upload_manifest runs unchanged.

    STATEFUL SESSION (`session="stateful"`): the container is kept
    across the run's epochs, so per epoch the receipt must show `container.session` (mode stateful, this cid,
    `daytona.session_id` == its sid), the spec's writable dirs == `rw_paths` (the loop passes
    [/evo/staging, /evo/evolver_workspace]; default when None), the SAME creation-time PRE probe (`probe_pre_sha256`
    from the state record, stateful_layout.probe_sha256), the lease acknowledged AND released, and an
    `epoch_end` record (kept, frozen, thawed, init_alive; empty tmpfs + state scans; egress from its log).
    The down / wiped / ledger rules apply only when `teardown` (the finish record) is given: a verified
    wipe with a ledgered container_down, or `machine_absent` with machine.json showing the sandbox gone.
    Container-keyed manifest rows are mapped through the receipt's recorded `path_view` for the INDEX.json
    cross-check. Default arguments = today's per-epoch behaviour exactly."""
    from src.orchestrator import daytona_evolver as de
    if session not in ("per_epoch", "stateful"):
        raise ValueError(f"session={session!r} is not one of ('per_epoch', 'stateful')")
    stateful = session == "stateful"
    ctl = load_ctl()
    d = receipt.get("daytona") or {}
    c = d.get("container") or {}
    probs = []
    if receipt.get("containment") != "daytona" or not receipt.get("sandboxed") or d.get("placement") != PLACEMENT:
        probs.append("receipt is not a sandboxed daytona receipt with placement pool_container")
    # WHICH MACHINE (the long-lived pool): the run's pool machine.json named it current + ready with the
    # headroom, and its host listed this dispatch's lease before anything ran. A smoke registry is never accepted.
    pm, le = d.get("pool_machine") or {}, d.get("lease") or {}
    if pm.get("format") == "pool":
        if (pm.get("row") or {}).get("sandbox_id") != d.get("sandbox_id") or (pm.get("row") or {}).get("state") != "ready":
            probs.append("pool machine: the receipt's machine is not the ready machine.json row it was picked from")
        if le.get("acknowledged") is not True or le.get("sandbox_id") != d.get("sandbox_id"):
            probs.append("pool machine: the pool host never listed this dispatch's lease (leased_by)")
        if stateful and le.get("released") is not True:
            probs.append("pool machine: the lease was not released after this epoch (the machine must be free "
                         "for the next train leg while the container idles)")
    elif SMOKE_PREFIX not in tuple(prefixes):
        probs.append(f"pool machine: taken from {pm.get('format')!r}, not the run pool's {POOL_MACHINES_FILE}")
    for k in ("readback_create", "readback_end"):
        rb = d.get(k) or {}
        mp = machine_problems(rb, name=d.get("sandbox_name") or "", state=rb.get("state") or "", row=None,
                              require_snapshot_ref=require_snapshot_ref if k == "readback_create" else None,
                              prefixes=tuple(prefixes))
        if mp:
            probs.append(f"{k}: " + "; ".join(mp[:3]))
    # An unpinned leg snapshot (require_snapshot_ref None) is accepted, but the receipt must record the ref
    # Daytona reported at attach (it is named in the detail below).
    snap_seen = (d.get("readback_create") or {}).get("snapshot_ref")
    if not require_snapshot_ref and not (isinstance(snap_seen, str) and snap_seen.strip()):
        probs.append("readback_create records no snapshot_ref (an unpinned leg snapshot must be recorded)")
    ok_df, det_df, rec_df = de.dockerfile_check()
    if not ok_df or rec_df["base_sha256"] != de.BASE_DOCKERFILE_SHA or \
            (c.get("leg_dockerfile_sha256") != de.BASE_DOCKERFILE_SHA):
        probs.append("image: the machine's recipe is not the frozen leg Dockerfile")
    spec, facts, limits = c.get("spec") or {}, c.get("facts") or {}, c.get("limits") or {}
    try:
        pl = ctl.plan(spec, facts, limits)
        if pl != c.get("plan"):
            probs.append("container: the recorded plan does not re-derive from its spec/facts/limits")
        import hashlib
        if hashlib.sha256(ctl.render_init(spec, facts, limits, pl).encode()).hexdigest() != c.get("init_sha256"):
            probs.append("container: the recorded init script does not re-render")
    except Exception as e:                                  # noqa: BLE001
        probs.append(f"container: plan does not re-derive ({type(e).__name__}: {e})")
        pl = []
    vs = ctl.validate_spec(spec)
    if vs:
        probs.append(f"container: spec problems {vs[:3]}")
    spec_rw = [r.get("path") for r in spec.get("rw") or []]
    if staging is not None and spec_rw != [str(Path(staging).absolute())]:
        probs.append(f"container: writable dirs {spec_rw} != the staging dir only")
    if stateful or rw_paths is not None:
        if rw_paths is None:
            from src.orchestrator import stateful_layout as SL
            rw_paths = [r["path"] for r in SL.STATEFUL_RW]
        want_rw = sorted(str(p) for p in rw_paths)
        if sorted(str(p) for p in spec_rw) != want_rw:
            probs.append(f"container: writable dirs {spec_rw} != {want_rw}")
    if stateful:
        from src.orchestrator import stateful_layout as SL
        ses = c.get("session") or {}
        if ses.get("mode") != "stateful" or not c.get("cid") or ses.get("cid") != c.get("cid"):
            probs.append(f"session: the receipt records no stateful session on container {c.get('cid')!r} "
                         f"(session.mode {ses.get('mode')!r}, session.cid {ses.get('cid')!r})")
        if not ses.get("sid") or d.get("session_id") != ses.get("sid"):
            probs.append(f"session: daytona.session_id {str(d.get('session_id'))[:8]!r} != the session's sid "
                         f"{str(ses.get('sid'))[:8]!r} (one claude session for the whole run)")
        if probe_pre_sha256 is not None or c.get("probe_pre_sha256") is not None:
            h = SL.probe_sha256(c.get("probe_pre") or {})
            if probe_pre_sha256 is not None and h != probe_pre_sha256:
                probs.append("container probe: the recorded probe_pre is not the creation-time probe (its sha "
                             "differs from the state record's probe_pre_sha256)")
            if c.get("probe_pre_sha256") is not None and c.get("probe_pre_sha256") != h:
                probs.append("container probe: the receipt's probe_pre_sha256 does not hash its recorded probe_pre")
    if pl:
        pp = container_probe_problems(c.get("probe_pre") or {}, expected_top=ctl.expected_top(pl),
                                      expected_mounts=ctl.expected_mount_points(pl), rw_paths=spec_rw,
                                      ro_rebind=spec.get("ro_rebind") or [], require_libreoffice=require_libreoffice)
        probs += [f"container probe: {p}" for p in pp[:6]]
    probs += post_probe_problems(c.get("probe_pre") or {}, c.get("probe_post") or {})
    # The evolver may get the full machine (all its CPUs and memory, with a much higher cpu.weight than the rollouts
    # beside it): a cap is required and may be the whole machine, never more.
    lim_machine_cpu = int(limits.get("machine_ncpu") or 0)
    if not limits.get("cpus") or (lim_machine_cpu and float(limits["cpus"]) > lim_machine_cpu):
        probs.append(f"limits: cpus {limits.get('cpus')} of {lim_machine_cpu} (more than the machine)")
    mm = int(limits.get("machine_mem_bytes") or 0)
    if not limits.get("mem_max_bytes") or (mm and int(limits["mem_max_bytes"]) > mm):
        probs.append("limits: memory cap missing or more than the machine")
    if stateful:
        eg = (c.get("epoch_end") or {}).get("egress_log") or {}
    else:
        eg = (c.get("down") or {}).get("egress_log") or c.get("egress_log") or {}
    allowed_hosts = sorted({k.rsplit(":", 1)[0] for k in (eg.get("by_host") or {}) if k.endswith(":allowed")})
    if allowed_hosts not in ([], [ctl.ALLOW]):
        probs.append(f"egress log: tunnels to {allowed_hosts}")
    before = receipt.get("evidence_sha256_before") or {}
    if stateful:
        # A stateful receipt keys `before` by the host evo_e{K} dirs and `after_vm` by the container roots
        # (stateful_layout.after_hashes_vm, informational); the loop judges the host bytes before/after itself
        # (stateful_evolver.v_session_contained). Both records must exist; the evo layout is raw logs with no
        # INDEX.json, so there is no index cross-check.
        if not before or not receipt.get("evidence_sha256_after_vm"):
            probs.append("evidence hashes: the receipt records no evidence_sha256_before (host) or "
                         "evidence_sha256_after_vm (container)")
    elif (receipt.get("evidence_sha256_after_vm") or {}) != before or not before:
        probs.append("evidence changed in the container during the run (or no after-hash)")
    # Per-epoch runs: the rollout evidence (full traces + grades) went in exactly as each dir's
    # INDEX.json lists it (daytona_evolver.evidence_index_problems; check_upload_manifest applies it too).
    if receipt.get("role") != "ao" and not stateful:
        probs += [f"rollout evidence: {p}" for p in
                  de.evidence_index_problems(receipt.get("upload_manifest") or [], list(before))]
    _mroles = {r.get("role") for r in receipt.get("upload_manifest") or []}
    for _role, _key, _what in (("inputs", "inputs", "inputs"), ("train_reference", "references", "train references"),
                               ("grader", "grader", "clean grader")):
        if receipt.get("role") == "ao":
            break
        if _role in _mroles or receipt.get(f"{_key}_dir") or receipt.get(f"{_key}_sha256_before"):
            if receipt.get(f"{_key}_sha256_after_vm") != receipt.get(f"{_key}_sha256_before") \
                    or not receipt.get(f"{_key}_sha256_before"):
                probs.append(f"{_what} changed in the container during the run (or no hash)")
    ts = receipt.get("token_scan") or {}
    if ts.get("hits") != [] or not ts.get("scanned_files"):
        probs.append(f"token scan: hits {ts.get('hits')!r}, scanned {ts.get('scanned_files')}")
    _ledger = ledger_path or os.environ.get("EVOSKILL_DAYTONA_LEDGER",
                                            str(REPO / ".cache" / "daytona" / "ledger.jsonl"))
    if not stateful:
        dn = c.get("down") or {}
        if dn.get("verified") is not True or dn.get("wiped") is not True:
            probs.append(f"teardown not verified (verified {dn.get('verified')!r}, wiped {dn.get('wiped')!r})")
        scan_hits = [h for part in (dn.get("scan") or c.get("scan") or {}).values() for h in part.get("hits") or []]
        tmp_hits = (c.get("down_before_pack") or {}).get("tmpfs_scan", {}).get("hits") or []
        if scan_hits or tmp_hits or not (dn.get("scan") or c.get("scan")):
            probs.append(f"container state scan: {len(scan_hits)} + {len(tmp_hits)} token/key-prefix hit(s) (or no scan)")
        if not d.get("sandbox_id") or not ledger_container_down(_ledger, sandbox_id=d.get("sandbox_id"),
                                                                cid=c.get("cid") or ""):
            probs.append("no ledgered, verified 'container_down' row for this container")
        _end = "torn down (ledgered)"
    else:
        ee = c.get("epoch_end") or {}
        bad = [k for k in ("kept", "frozen", "thawed", "init_alive") if ee.get(k) is not True]
        if bad:
            probs.append(f"epoch-end: {', '.join(bad)} not true ({({k: ee.get(k) for k in bad})}); the container must "
                         f"be frozen for the scan, thawed, and alive for the next epoch")
        tmp_hits = (ee.get("tmpfs_scan") or {}).get("hits")
        scan = ee.get("scan") or {}
        scan_hits = [h for part in scan.values() for h in (part or {}).get("hits") or []]
        if scan_hits or tmp_hits != [] or not scan:
            probs.append(f"container state scan (epoch-end): {len(scan_hits)} + {len(tmp_hits or [])} token/key-prefix "
                         f"hit(s) (or no scan)")
        if c.get("down") is not None:
            probs.append("container went down inside a kept epoch ('down' recorded; a stateful epoch ends with "
                         "epoch-end and the container stays)")
        _end = "kept for the next epoch (epoch-end: frozen, scanned, thawed; lease released)"
        if teardown is not None:
            td = teardown if isinstance(teardown, dict) else {}
            if td.get("cid") != c.get("cid") or td.get("sandbox_id") != d.get("sandbox_id"):
                probs.append(f"teardown record names container {td.get('cid')!r} on {str(td.get('sandbox_id'))[:8]!r}, "
                             f"not this receipt's {c.get('cid')!r} on {str(d.get('sandbox_id'))[:8]!r}")
            ma = td.get("machine_absent")
            if ma is not None:
                # `finish --machine-absent-ok` (evolver_session.py): the pool's machine.json no longer names the
                # session's sandbox as a live machine (absent, or deleted / released / failed by the pool host)
                row = ma.get("machine_json") if isinstance(ma, dict) else {}
                gone = row is None or (isinstance(row, dict) and (row.get("state") in ("deleted", "released", "failed")
                                                                  or str(row.get("sandbox_id")) != str(d.get("sandbox_id"))))
                if not isinstance(ma, dict) or "machine_json" not in ma or not gone:
                    probs.append("teardown: machine_absent claimed, but machine.json still shows the sandbox (or the "
                                 "record carries no machine.json read)")
                _end = "machine absent at the run's end (recorded; the container cannot outlive its machine)"
            else:
                dn = td.get("down") or {}
                if dn.get("verified") is not True or dn.get("wiped") is not True:
                    probs.append(f"teardown not verified (verified {dn.get('verified')!r}, wiped {dn.get('wiped')!r})")
                if not d.get("sandbox_id") or not ledger_container_down(_ledger, sandbox_id=d.get("sandbox_id"),
                                                                        cid=c.get("cid") or ""):
                    probs.append("no ledgered, verified 'container_down' row for this container")
                _end = "torn down (ledgered)"
    lrt = d.get("log_round_trip") or {}
    _lb = Path(str(log)).read_bytes() if Path(str(log)).is_file() else None
    if _lb is not None and receipt.get("role") == "ao" and _lb.startswith(b"=== prompt author:"):
        _lb = _lb.split(b"\n", 1)[1] if b"\n" in _lb else b""
    if not lrt.get("equal") or _lb is None or de.sha256_bytes(_lb) != lrt.get("streamed_sha256"):
        probs.append("the audited log is not the byte-exact copy of the container's log recorded in the receipt")
    sa = d.get("session_audit") or {}
    if not sa.get("files"):
        probs.append("no session audit recorded")
    elif sa.get("denied_tool_uses") or sa.get("nested_claude") or sa.get("api_or_token_refs"):
        probs.append(f"session audit: denied tools {sa.get('denied_tool_uses')[:3]}, nested claude "
                     f"{sa.get('nested_claude')[:3]}, api/token refs {sa.get('api_or_token_refs')[:3]}")
    li = de.log_init_and_tools(log)
    from src.orchestrator.cc_account import allowed_api_key_sources
    if li["api_key_source"] not in allowed_api_key_sources():
        probs.append(f"log init apiKeySource {li['api_key_source']!r}: not the run's credential")
    if li["mcp_servers"] is None:
        probs.append("log carries no init record (mcp_servers unknown)")
    elif li["mcp_servers"] != []:
        probs.append(f"log init lists MCP servers {li['mcp_servers']!r}")
    if li["remote_trigger_calls"]:
        probs.append(f"{li['remote_trigger_calls']} RemoteTrigger call(s) (a RemoteTrigger call fails the check)")
    if probs:
        return False, "; ".join(probs[:6]) + (f" (+{len(probs)-6} more)" if len(probs) > 6 else "")
    return True, (f"container {c.get('cid')} on pool machine {str(d.get('sandbox_id'))[:8]} (snapshot {snap_seen}"
                  f"{'' if require_snapshot_ref else ', not pinned'}; allow list "
                  f"{MACHINE_ALLOW_STR}; container egress {ctl.ALLOW} only via the proxy), uid {ctl.EVO_UID}, no "
                  f"capabilities, own mnt/pid/net/ipc/uts namespaces, mount table = plan, 0 of "
                  f"{(c.get('probe_pre') or {}).get('view', {}).get('canaries_checked')} machine paths visible, "
                  f"limits cpus {limits.get('cpus')} mem {int(limits.get('mem_max_bytes') or 0) >> 20} MiB, "
                  f"evidence and granted dirs unchanged, 0 token hits, {_end}, "
                  f"{'API key' if li['api_key_source'] == 'ANTHROPIC_API_KEY' else 'OAuth token'} auth, "
                  f"no MCP, no RemoteTrigger"
                  + (f"; stateful session {str(d.get('session_id'))[:8]} epoch {(c.get('session') or {}).get('epoch')}"
                     if stateful else ""))
