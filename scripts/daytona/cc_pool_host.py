#!/usr/bin/env python3
"""THE MACHINE OWNER of a Claude Code solver run: ONE Daytona machine for the whole run.

Every rollout, grade and evolver container of a run shares one machine. This headless process creates, keeps and
finally deletes the run's single pool machine and publishes it in the pool's own registry format
(<store_root>/<run>/pool/run/{registry.json, machine.json, leases/}, schema evoskill.pool_machines/1), so the stateful
evolver's driver (evolver_container.attach_machine -> pool_container_evolver.pick_pool_machine + PoolLease) and the
Claude Code leg driver (cc_leg_driver.py, `join` below) find and lease it the same way.

THE MACHINE (CCMachine, a subclass of ssb_leg_daytona_loop.Shard: the same creation, identity/egress probes, cgroup
delegation, shim + grading self-tests as every leg machine; pool egress rule + TTL grant check reused from
ssb_leg_pool.Machine by unbound call, never edited): the LEG image, 8/32/40, TTL pool_ttl_minutes (1440), allow list
openrouter.ai + api.anthropic.com (pce.MACHINE_ALLOW_STR), the grading bundle (src/, dataset, every train+val
task's input+golden through the data overlay, /opt/leg/leg.env = grader knobs only, /opt/grade/grade_worker.py),
then evoctl + the egress proxy + the pinned Claude Code (sha-verified, skip-if-present). Its summary.json
(<pool dir>/machine/<name>/) carries identity, egress, grade_selftest, bundle, readback, install: the leg driver
copies them into runtime_env and refuses a machine whose identity differs from the row it leased. Each CC leg's
InPlaceGrader.install() adds the plug-in grade runner (/opt/grade/plugin_grade_runner.py), the leg's pinned plug-in
files and the user's material linked under /data.

Rollouts come before grading: every machine's setup writes the top-level cgroups' cpu.weight itself (evoevolver 10000
= solver + evolver containers, evoshim 1 + evoshim_runner 1 = grading; read back into summary['cgroup_weights'], a
mismatch logged, never fatal). A TAU3 run (SPEC spec.benchmark "tau3") gets its machine from INFRA leg_sandbox =
scripts/daytona/Dockerfile.tau3_leg (the SSB image's bytes + the /opt/tau3 venv; context by
tau3_cc_smoke.write_build_dir) and, in place of the SSB bundle and self-tests: the data root mirrored AT ITS HOST PATH
(MANIFEST-verified there), grade_worker.py + leg.env + the repo mirror dir (what InPlaceGrader needs), the cgroups plus
/sys/fs/cgroup/evobridge (weight 10000: the legs' host-side bridges), the tau3 venv against its pin and tau2 / data /
openrouter checked under /opt/tau3/venv (tau3_cc_smoke's machine checks). Its legs grade under that interpreter
(InPlaceGrader(runner_py=...)).

NEVER TWO ALIVE: every sandbox of this run's pool that Daytona still lists is deleted by id BEFORE a machine is
created (ssb_leg_pool.reap_pool_sandboxes), and the host refuses while any stays listed. A replacement (m02) is
created only after the current machine is deleted, and only on the operator's <pool root>/ROLLOVER file.
NO IDLE RELEASE: the machine lives for the run (the evolver's lease-less idle between epochs must not lose it).
STOP: <pool root>/STOP (`stop --run-dir R` writes it) or SIGTERM -> host_state "closing" (no new lease is
acknowledged), wait while any valid lease exists (bounded by pool_ttl_margin_min), `evoctl list` -> `down --wipe`
every remaining cc-* container (never an e*-/ao- one: those are the evolver's), delete the machine by id, verified by
re-listing, machine.json (deleted / closed), registry.json closed, pool_record.json.

SHARED-MACHINE CONTRACT with the stateful evolver: cgroup parent /sys/fs/cgroup/evoevolver is shared, cids are unique
(cc-<unit>-<stamp>-<salt> vs e<N>-...); the CC side touches only cc-* containers; /opt/cc/<run>/<tag>/ (unit outputs,
grading) is invisible to every container; both use the run's one Claude credential, read by the leg driver and the
evolver's driver, never by this host (nothing here reads a credential or a model key; OPENROUTER_API_KEY is scrubbed
at start). While it serves, the host keeps the computer awake (host_tools.keep_awake: caffeinate on macOS,
systemd-inhibit on Linux when installed, else nothing).

Subcommands:
  host   --run-dir R [--pool-dir D]   the owner process (spawned by the first CC leg driver, or started by hand)
  status --run-dir R                  registry + machine.json + leases of the run's pool
  stop   --run-dir R [--wait-s N]     write STOP, wait for the registry to say closed (the machine deleted)
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(1, str(HERE.parents[1]))
if __name__ == "__main__":                       # one module object, whichever way this file is entered
    sys.modules.setdefault("cc_pool_host", sys.modules["__main__"])
import daytona_lib_loop as dlx  # noqa: E402  (READ-ONLY reuse: create/delete/list/ledger/readback)
import evolver_container_ctl as ctl  # noqa: E402  (READ-ONLY reuse: the machine-side paths and pins)
import plugin_grade_runner as PGR  # noqa: E402  (READ-ONLY reuse: score_of, validate_job, the job/result schema)
import ssb_grade_daytona as G  # noqa: E402  (READ-ONLY reuse: grade_env_text)
import ssb_leg_daytona_loop as L  # noqa: E402  (READ-ONLY reuse: Shard, build_bundle, EXPECTED, settings)
import ssb_leg_pool as P  # noqa: E402  (READ-ONLY reuse: Registry, host_alive, pool_code, reap, egress rule)
from src.orchestrator import data_overlay  # noqa: E402
from src.orchestrator import host_tools  # noqa: E402  (keep_awake: the one platform-specific host helper)
from src.orchestrator import pool_container_evolver as pce  # noqa: E402

DRIVER = "cc_pool_host"
OWNER = "cc_pool_host"                           # registry.json / machine.json "owner"
HOST_SCRIPT_ENV = "EVOSKILL_CC_POOL_HOST_SCRIPT"  # a stand-in host for the contract tests only
STOP_FILE, ROLLOVER_FILE = "STOP", "ROLLOVER"    # under <pool root> (= the registry dir)
CC_CID_PREFIX = "cc-"                            # the CC legs' containers (cc_solver_lib.make_cid)
CC_LEG_BLOCK = "cc_leg"                          # RUNTIME.json block read only by the CC leg driver and this host
CC_LEG_DEFAULTS = {
    "concurrency": 30, "mem_max_gib_per_container": 1, "wall_min": 90, "machine_wait_s": 1800,
    "grade_max_procs": 4, "grade_max_heavy": 1, "skills_placement": "project_symlink", "cc_version": ctl.CC_VERSION,
    "thinking_display": "summarized",
}
# keys of the usage-threshold era: there is no threshold any more (a leg waits only after a session was rejected for a
# usage limit, cc_leg_lib.rate_limit_wait), so a RUNTIME that still sets one is refused rather than silently ignored
CC_LEG_RETIRED = ("quota_stop", "max_pause_s")
POOL_TTL_MINUTES_DEFAULT = 1440                  # when RUNTIME daytona_leg declares no pool_ttl_minutes
DELETE_VERIFY_S = 180                            # re-list this long after a delete (cc_solver.Machine.delete's rule)
DISK_REFRESH_S = 60.0
LEASE_WAIT_LOG_S = 300.0
GRADE_WORKER_SRC = HERE / "grade_worker.py"
GRADE_WORKER_VM = "/opt/grade/grade_worker.py"
# The legs grade with the run's frozen grade(task, snapshot) through this runner, which
# reuses the bundled grade_worker's Runner from its own directory. InPlaceGrader.install() puts it there, sha-verified.
RUNNER_SRC = HERE / "plugin_grade_runner.py"
RUNNER_VM = "/opt/grade/plugin_grade_runner.py"
RUNNER_PY = "/opt/evoskill/venv/bin/python"      # the grader's setup interpreter (the leg image's runner venv)
SNAPSHOT_ROOT = "evo/work"                       # the solver's working directory inside a container's out.tgz
TASK_FILES_PER_CALL = 100                        # sha256sum arguments per exec in install()'s task-file check
CC_LOCAL = L.REPO / ".cache" / "claude_code" / ctl.CC_VERSION / "linux-x64" / "claude"
CC_SHA256_PIN, CC_BYTES_PIN = ctl.CC_SHA256, ctl.CC_BYTES   # module names: a class's `ctl` method shadows the module
CTL_SRC = HERE / "evolver_container_ctl.py"
PROXY_SRC = L.REPO / "scripts" / "evolver_egress_proxy.py"
CC_ROOT_VM = "/opt/cc"                           # /opt/cc/<run>/<tag>/{units,grade}: invisible to every container
GRADE_WORK_VM = f"{P.SANDBOX_REPO}/.cache/_cc_grade_work"
SUMMARY_KEYS = ("identity", "egress", "grade_selftest", "bundle", "readback", "install", "sandbox_id",
                "sandbox_name", "granted_ttl_min")
# the Shard's unpack line (ssb_leg_daytona_loop.Shard.upload), verbatim
UNPACK_CMD = ("tar xzf /tmp/bundle.tgz -C / && rm /tmp/bundle.tgz && "
              "install -m 0755 /opt/leg/docker_shim.py /usr/local/bin/docker && "
              "mkdir -p /opt/leg/out /var/log/evoshim /var/lib/evoshim/state /var/lib/evoshim/priv && "
              "which docker && sha256sum /usr/local/bin/docker")
LEASE_RULE = ("a valid lease (until_ts in the future; pid alive when given) keeps its machine from being released "
              f"for retirement or at STOP, never past ttl_expires_ts - {L.POOL_SYNC_MARGIN_MIN} min; check "
              "retire_at_ts before starting work; no lease is acknowledged while the host is closing")
HOLDER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
# The machine setup writes these top-level cgroups' cpu.weight (rollouts over grading); evoctl's
# cg_ensure later only makes evoevolver if absent and keeps its weight
CPU_WEIGHTS = {"evoevolver": 10000, "evoshim": 1, "evoshim_runner": 1}
# the cgroup-delegation half of ssb_leg_daytona_loop.Shard.cgroups_and_selftest, verbatim (a test pins the two equal)
CGROUP_CMD = ("CG=/sys/fs/cgroup; echo '+memory +cpu +pids' > $CG/cgroup.subtree_control 2>&1 || "
              "{ mkdir -p $CG/sandbox_init; for p in $(cat $CG/cgroup.procs); do echo $p > "
              "$CG/sandbox_init/cgroup.procs 2>/dev/null; done; echo '+memory +cpu +pids' > "
              "$CG/cgroup.subtree_control; }; mkdir -p $CG/evoshim $CG/evoshim_runner; "
              "echo '+memory +cpu +pids' > $CG/evoshim/cgroup.subtree_control; "
              "echo root=$(cat $CG/cgroup.subtree_control) evoshim=$(cat $CG/evoshim/cgroup.subtree_control)")
# tau3 (SPEC spec.benchmark "tau3"): the machine tau3_cc_smoke builds and proves
TAU3_BENCHMARK = "tau3"
BRIDGE_CGROUP = "evobridge"                      # the legs' host-side bridges (tau2 + litellm) run in this cgroup
BRIDGE_CPU_WEIGHT = 10000                        # a bridge waits on its rollout: as urgent as the rollout itself
TAU3_DATA_TGZ_VM = "/tmp/tau3_data.tgz"
TAU3_MANIFEST = "MANIFEST.sha256"                # the data root's own manifest (scripts/tau3_make_data_root.py)
TAU3_VENV_EQUAL = "equal: 76 packages, CPython 3.12.13"   # tau3_check_venv.py on the image's /opt/tau3/venv vs its pin
TAU3_TAU2_VERSION, TAU3_N_TASKS = "1.0.1", 97


def q(s) -> str:
    return shlex.quote(str(s))


def parse_json_tail(out: str) -> dict:
    """The last JSON object line of a command's output (evoctl prints one)."""
    for ln in reversed((out or "").strip().splitlines()):
        ln = ln.strip()
        if ln.startswith("{"):
            try:
                return json.loads(ln)
            except ValueError:
                continue
    return {}


def _atomic_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def _sha_map(out: str) -> dict:
    """{path: sha256} from `sha256sum a b` output. A path may hold spaces (the frozen task files do); a
    name sha256sum has to escape (a backslash or a newline in it) is left out, so it reads as missing."""
    got = {}
    for ln in (out or "").splitlines():
        m = re.fullmatch(r"([0-9a-f]{64}) [ *](.+)", ln)
        if m:
            got[m.group(2)] = m.group(1)
    return got


class HandleError(RuntimeError):
    """A pool-machine command failed for a leg (the handle's sh/ctl)."""


class GradeError(RuntimeError):
    """The in-place grader refused or a grading command failed."""


# ───────────────────────────────────────────────────────────────────────────── settings
def cc_leg_settings(run_dir) -> dict:
    """RUNTIME.json `cc_leg` (CC_LEG_DEFAULTS + what the run declares; `_`-keys ignored). Unknown keys and
    non-numeric numbers are Refused (L.Refused), as load_runtime_block refuses a typo'd daytona_leg."""
    f = Path(run_dir) / "RUNTIME.json"
    rt = {}
    if f.is_file():
        try:
            rt = json.loads(f.read_text())
        except ValueError as e:
            raise L.Refused(f"{f} is not readable JSON: {e}")
    blk = rt.get(CC_LEG_BLOCK) or {}
    if not isinstance(blk, dict):
        raise L.Refused(f"RUNTIME.json {CC_LEG_BLOCK} must be an object")
    retired = sorted(k for k in blk if k in CC_LEG_RETIRED)
    if retired:
        raise L.Refused(f"RUNTIME.json {CC_LEG_BLOCK} sets {retired}: no usage threshold exists any more (a leg stops "
                        f"admitting only when a session is rejected for a usage limit, then waits for the reset it "
                        f"reported); remove the key(s)")
    unknown = sorted(k for k in blk if not str(k).startswith("_") and k not in CC_LEG_DEFAULTS)
    if unknown:
        raise L.Refused(f"RUNTIME.json {CC_LEG_BLOCK} has unknown key(s) {unknown}; known {sorted(CC_LEG_DEFAULTS)}")
    out = dict(CC_LEG_DEFAULTS)
    out.update({k: v for k, v in blk.items() if not str(k).startswith("_")})
    bad = []
    for k in ("concurrency", "mem_max_gib_per_container", "wall_min", "machine_wait_s", "grade_max_procs",
              "grade_max_heavy"):
        v = out[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
            bad.append(f"{k}={v!r} (a number > 0)")
    if int(out["concurrency"]) > 30:
        bad.append(f"concurrency={out['concurrency']} (at most 30 rollouts on one machine)")
    for k in ("skills_placement", "cc_version", "thinking_display"):
        if not isinstance(out[k], str) or not out[k]:
            bad.append(f"{k}={out[k]!r} (a non-empty string)")
    if bad:
        raise L.Refused(f"RUNTIME.json {CC_LEG_BLOCK} invalid: " + "; ".join(bad))
    return out


def pool_root(run_dir) -> Path:
    """<store_root>/<run>/pool/run: the run's pool (ssb_leg_pool.pool_root == pce.pool_run_dir)."""
    rd = Path(run_dir).resolve()
    rt = L.load_runtime_block(rd)
    return P.pool_root(_NS(rt=rt, run_dir=rd))


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class CCMachineCfg:
    """The plain object ssb_leg_daytona_loop.Shard / build_bundle / runtime_env_record read (a LegConfig's
    attributes without a solver: the DeepSeek pins do not exist for this run). Built by machine_cfg()."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    @property
    def tasks_dir_local(self) -> Path:
        return Path(self.rt["tasks_root"]) / f"ssbv2_{self.tag}_tasks"

    @property
    def tau3(self) -> bool:
        return self.benchmark == TAU3_BENCHMARK


def _tau3():
    """scripts/daytona/tau3_cc_smoke.py (READ-ONLY reuse: the tau3 image's build context, the data root's tgz and
    check, the machine checks), imported only on a tau3 run's path."""
    import tau3_cc_smoke as S
    return S


def tau3_image(ls: dict) -> tuple:
    """(image_id, dockerfile sha) of a tau3 run's machine: INFRA leg_sandbox.dockerfile must be Dockerfile.tau3_leg,
    its bytes on disk INFRA's dockerfile_sha256, image_id 'x86daytona-' + sha[:12] == INFRA's (Refused otherwise)."""
    S = _tau3()
    df = L.REPO / str(ls.get("dockerfile") or "")
    if df.resolve() != S.DOCKERFILE.resolve():
        raise L.Refused(f"INFRA leg_sandbox.dockerfile {ls.get('dockerfile')!r} is not "
                        f"{S.DOCKERFILE.relative_to(L.REPO)}: a tau3 run's machine is the tau3 leg image")
    df_sha = L.sha256_file(df)
    if df_sha != ls.get("dockerfile_sha256"):
        raise L.Refused(f"{df.name} sha256 {df_sha[:12]} != INFRA {str(ls.get('dockerfile_sha256'))[:12]}: a different "
                        f"image is a different comparability class")
    image_id = "x86daytona-" + df_sha[:12]
    if image_id != ls.get("image_id"):
        raise L.Refused(f"image_id {image_id} (the Dockerfile's) != INFRA's {ls.get('image_id')}")
    return image_id, df_sha


def machine_cfg(run_dir, *, env=None) -> CCMachineCfg:
    """Everything decided before the machine exists, from the run's SPEC / INFRA / RUNTIME (Refused = nothing
    created): tag pool_<run>, code rp<sha5>, the pool TTL (RUNTIME daytona_leg.pool_ttl_minutes, default 1440),
    require_snapshot_ref / require_libreoffice = INFRA leg_sandbox's (a RUNTIME value that disagrees is refused),
    image_id = EXPECTED's (must equal INFRA's and the Dockerfile on disk), the split (SPEC's file and sha),
    tasks = split train ∪ val (the machine bundles every one's input + golden), knobs, data overlay. A
    tau3 run (SPEC benchmark "tau3") takes its image from INFRA leg_sandbox instead (tau3_image) and its data root
    (INFRA data.root) is verified against its own MANIFEST here, before any machine (tau3_cc_smoke.verify_data_root)."""
    process_env = env is None
    env = dict(os.environ if env is None else env)
    rd = Path(run_dir).resolve()
    if not rd.is_dir():
        raise L.Refused(f"no run dir {rd}")
    if not NAME_RE.match(rd.name):
        raise L.Refused(f"run dir name {rd.name!r} is not [A-Za-z0-9_.-]+ (it names machine paths)")
    rt = L.load_runtime_block(rd)
    p = L.pool_settings(rt)
    inf, infra_record = L.load_infra(rd)
    if inf is None:
        raise L.Refused(f"no {L.infra_cfg.INFRA_FILE} in {rd}: the pool machine's image, timeouts and data overlay "
                        f"come from the operator's INFRA.json")
    ls = inf["leg_sandbox"]
    ttl = int(p["pool_ttl_minutes"] if p.get("pool_ttl_minutes") is not None else POOL_TTL_MINUTES_DEFAULT)
    if not 10 <= ttl <= 1440:
        raise L.Refused(f"pool TTL {ttl} min outside [10, 1440]")
    if ttl - int(p["pool_ttl_margin_min"]) < L.POOL_MIN_USABLE_MIN:
        raise L.Refused(f"pool TTL {ttl} - pool_ttl_margin_min {p['pool_ttl_margin_min']} leaves under "
                        f"{L.POOL_MIN_USABLE_MIN} min of service")
    for key, want in (("require_snapshot_ref", ls.get("snapshot_ref")), ("require_libreoffice", ls.get("libreoffice"))):
        have = rt.get(key)
        if have and want and have != want:
            raise L.Refused(f"RUNTIME daytona_leg.{key} {have!r} != INFRA leg_sandbox {want!r}")
    mrt = dict(rt, ttl_minutes=ttl, require_snapshot_ref=ls.get("snapshot_ref") or rt.get("require_snapshot_ref"),
               require_libreoffice=ls.get("libreoffice") or rt.get("require_libreoffice"))
    sf = rd / "SPEC.json"
    if not sf.is_file():
        raise L.Refused(f"no frozen SPEC.json in {rd}")
    spec = json.loads(sf.read_text()).get("spec") or {}
    benchmark, data = spec.get("benchmark"), None
    if benchmark == TAU3_BENCHMARK:
        image_id, df_sha = tau3_image(ls)
        S = _tau3()
        try:
            data = S.verify_data_root(Path(inf["data"]["root"]))
        except S.Refuse as e:
            raise L.Refused(f"INFRA data.root: {e}")
    else:
        df_sha = L.sha256_file(L.DOCKERFILE)
        if df_sha != L.EXPECTED["dockerfile_sha256"] or df_sha != ls.get("dockerfile_sha256"):
            raise L.Refused(f"{L.DOCKERFILE.name} sha256 {df_sha[:12]} != EXPECTED "
                            f"{L.EXPECTED['dockerfile_sha256'][:12]} / INFRA {str(ls.get('dockerfile_sha256'))[:12]}: "
                            f"a different image is a different comparability class")
        image_id = L.EXPECTED["image_id"]
        if image_id != "x86daytona-" + df_sha[:12] or image_id != ls.get("image_id"):
            raise L.Refused(f"image_id {image_id} != the Dockerfile's / INFRA's {ls.get('image_id')}")
    split_file = Path(str(spec.get("split_path") or ""))
    if not split_file.is_file():
        raise L.Refused(f"SPEC split_path {split_file} is not a file")
    split_sha = L.sha256_file(split_file)
    if spec.get("split_sha256") and spec["split_sha256"] != split_sha:
        raise L.Refused(f"SPEC split_sha256 {str(spec['split_sha256'])[:12]} but {split_file.name} hashes to "
                        f"{split_sha[:12]}")
    split = json.loads(split_file.read_text())
    train, val = list(split.get("train") or []), list(split.get("val") or [])
    if not train or not val:
        raise L.Refused(f"split {split_file.name} has no non-empty train / val list")
    tasks = train + [t for t in val if t not in set(train)]
    # The held-out TEST leg joins the same pool when RUNTIME daytona_leg.pool_keys
    # lists "test"; its tasks' input and reference files are bundled onto the machine only then
    if "test" in (p.get("pool_keys") or []):
        seen = set(tasks)
        tasks += [t for t in (split.get("test") or []) if t not in seen]
    n_samples = int(spec.get("n_samples") or 0)
    if n_samples < 1:
        raise L.Refused("SPEC n_samples must be >= 1")
    knobs = L.leg_knobs(inf, env)
    overlay = L.data_overlay_from_infra(inf, env, set_process_env=process_env)
    name = rd.name
    return CCMachineCfg(
        tag=f"pool_{name}", code=P.pool_code(name), run_dir=rd, run_name=name, rt=mrt, rt_declared=rt, pool=p,
        image_id=image_id, dockerfile_sha256=df_sha, split_file=split_file, split_sha256=split_sha,
        split_status=split.get("status"), tasks=tasks, train=train, val=val, n_samples=n_samples, key="pool",
        skills_dir=None, cache_dir=None, knobs=knobs, data_overlay=overlay, infra=inf, infra_record=infra_record,
        env_pins={}, leg_settings={}, prompt_mode=L.runtime_prompt_mode(rd), spec=spec,
        require_snapshot_ref=mrt["require_snapshot_ref"], require_libreoffice=mrt["require_libreoffice"],
        cc=cc_leg_settings(rd), store=None, benchmark=benchmark, data=data)


# ───────────────────────────────────────────────────────────────────────────── the installs (shared)
def install_ctl(run, upload) -> dict:
    """evoctl + the egress proxy on the machine, sha-verified; SKIPPED when the machine already holds these bytes
    (idempotent: the host, the evolver and the leg driver may each call it). `run(cmd, timeout, what) -> (rc,
    out)`, `upload(src, dst, timeout)`. Raises L.Abort when the upload does not verify."""
    want = {ctl.CTL: L.sha256_file(CTL_SRC), ctl.PROXY_FILE: L.sha256_file(PROXY_SRC)}
    rc, out = run(f"sha256sum {ctl.CTL} {ctl.PROXY_FILE} 2>/dev/null", 60, "ctl sha")
    if _sha_map(out) == want:
        return {"ctl": "already on the machine (sha verified)", "sha256": want}
    run(f"mkdir -p {ctl.BIN} && chmod 0700 {ctl.BASE}", 60, "mkdir bin")
    for src, dst in ((CTL_SRC, ctl.CTL), (PROXY_SRC, ctl.PROXY_FILE)):
        upload(str(src), dst, 300)
    rc, out = run(f"sha256sum {ctl.CTL} {ctl.PROXY_FILE}", 60, "ctl sha")
    got = _sha_map(out)
    if got != want:
        raise L.Abort(f"controller upload does not verify ({got} != {want})")
    return {"ctl": "uploaded, sha verified", "sha256": want}


def install_cc(run, upload, local=None, *, sha256=CC_SHA256_PIN, nbytes=CC_BYTES_PIN) -> dict:
    """The pinned Claude Code linux-x64 binary at ctl.CC_HOST, sha + size verified; SKIPPED when the machine already
    holds it. The local file must be the pinned binary (sha, size) before anything is uploaded."""
    local = Path(CC_LOCAL if local is None else local)
    if not local.is_file() or L.sha256_file(local) != sha256 or local.stat().st_size != int(nbytes):
        raise L.Abort(f"{local}: not the pinned Claude Code {ctl.CC_VERSION} linux-x64 binary (sha {sha256[:12]}, "
                      f"{nbytes} B)")
    rc, out = run(f"sha256sum {ctl.CC_HOST} 2>/dev/null | cut -d' ' -f1; stat -c %s {ctl.CC_HOST} 2>/dev/null",
                  120, "cc sha")
    lines = (out or "").split()
    if lines[:1] == [sha256] and lines[1:2] == [str(nbytes)]:
        return {"cc": "already on the machine (sha verified)", "cc_sha256": sha256, "path": ctl.CC_HOST}
    part = ctl.CC_HOST + ".part"
    run(f"mkdir -p {os.path.dirname(ctl.CC_HOST)} && rm -f {part}", 60, "cc dir")
    t = time.time()
    upload(str(local), part, 3600)
    rc, out = run(f"test \"$(sha256sum {part} | cut -d' ' -f1)\" = {sha256} && "
                  f"test \"$(stat -c %s {part})\" = {nbytes} && chmod 0755 {part} && "
                  f"mv {part} {ctl.CC_HOST} && echo CC_OK", 300, "cc verify")
    if "CC_OK" not in (out or ""):
        raise L.Abort("uploaded Claude Code binary does not verify on the machine")
    return {"cc": f"uploaded {nbytes} B in {time.time() - t:.0f}s, sha verified", "cc_sha256": sha256,
            "path": ctl.CC_HOST}


def cc_cids(listing: dict) -> list:
    """The CC legs' container ids of an `evoctl list` answer (cc-*); the evolver's (e<N>-, ao-) are never touched."""
    return sorted(str(c) for c in (listing or {}).get("containers") or [] if str(c).startswith(CC_CID_PREFIX))


def other_cids(listing: dict) -> list:
    return sorted(str(c) for c in (listing or {}).get("containers") or [] if not str(c).startswith(CC_CID_PREFIX))


# ───────────────────────────────────────────────────────────────────────────── the machine
class CCMachine(L.Shard):
    """The run's one pool machine: a Shard (creation, identity, cgroups, shim + grading self-tests, all unchanged)
    whose egress rule and TTL grant check are ssb_leg_pool.Machine's (called unbound), plus the grading bundle,
    evoctl / proxy / Claude Code installs, the registry row, cc-* cleanup and the verified delete. The cgroups'
    cpu.weight is written on every machine; a tau3 run's machine is created from the tau3 leg image
    and proven by the tau3 steps instead of the SSB bundle and self-tests (prove)."""

    def __init__(self, owner: "PoolMachineOwner", name: str, tasks: list):
        super().__init__(owner, name, tasks)
        self.pool = owner
        self.allow_csv = pce.MACHINE_ALLOW_STR
        self.t_created = self.t_ready = self.t_end = None
        self.failed = None
        self.deleted = None
        self.released = False
        self.disk_free_gib = None
        self.units_running = 0
        self.summary.update(role="cc_pool_machine", owner=OWNER, machine=name)

    # -- the Shard's hooks
    def allow_list(self) -> str:
        return self.allow_csv

    def egress(self):
        """The POOL rule (ssb_leg_pool.Machine.egress): openrouter.ai AND api.anthropic.com reachable, everything
        else refused."""
        return P.Machine.egress(self)

    def check_granted_ttl(self):
        return P.Machine.check_granted_ttl(self)

    def run_sh(self, cmd: str, timeout: int, what: str) -> tuple:
        return self.sh(what, cmd, timeout=timeout)

    def ctl(self, args: str, timeout: int = 600) -> tuple:
        rc, out = self.sh("evoctl", f"{ctl.PY} {ctl.CTL} {args}", timeout=timeout)
        return rc, out, parse_json_tail(out)

    def _create_once(self):
        """tau3: ssb_leg_daytona_loop.Shard._create_once with the tau3 leg image -- Dockerfile.tau3_leg next to the
        clean build context tau3_cc_smoke.write_build_dir writes under this machine's dir (again on a retried
        create); the name, resources, TTL, allow list, labels, build log and the readback's resource and
        require_snapshot_ref checks are the Shard's. An SSB machine is the Shard's own."""
        if not self.cfg.tau3:
            return super()._create_once()
        from daytona import CreateSandboxFromImageParams, Image, Resources
        S = _tau3()
        build = self.dir / "build"
        shutil.rmtree(build, ignore_errors=True)
        try:
            meta = S.write_build_dir(build)
        except S.Refuse as e:
            raise L.Abort(f"the tau3 image's build context: {e}")
        self.summary["image"] = meta
        name = f"{dlx.PREFIX}{self.cfg.code}-{self.name}-{time.strftime('%H%M%S')}"
        ttl = int(self.cfg.rt["ttl_minutes"])
        allow = self.allow_list()
        params = CreateSandboxFromImageParams(
            name=name, image=Image.from_dockerfile(str(meta["build_dockerfile"])), resources=Resources(**L.RESOURCES),
            ephemeral=True, auto_stop_interval=0, ttl_minutes=ttl, public=False,
            domain_allow_list=allow, labels={"shard": self.name, "leg": self.cfg.code})
        self.log(f"creating {name} (Dockerfile.tau3_leg {meta['dockerfile_sha256'][:12]}, context "
                 f"{meta['context_sha256'][:12]}, {L.RESOURCES}, ttl {ttl} min, egress {allow})")
        blog = open(self.dir / "image_build.log", "a")
        sb, sid, err, dt = dlx.create(self.leg.client, params, L.RESOURCES, tag=self.cfg.tag,
                                      timeout=3600,
                                      on_logs=lambda c: (blog.write(c + "\n"), blog.flush()))
        self.sid = sid
        self.summary.update(sandbox_id=sid, sandbox_name=name, create_s=dt)
        if err is not None:
            raise L.Abort(f"create failed after {dt}s: {type(err).__name__}: {str(err)[:600]}")
        self.sb = sb
        rb = dlx.readback(sb)
        self.summary["readback"] = rb
        mism = {k: (L.RESOURCES[k], rb.get(k)) for k in L.RESOURCES if rb.get(k) != L.RESOURCES[k]}
        self.log(f"created {sid} in {dt}s readback cpu={rb['cpu']} mem={rb['memory']} "
                 f"disk={rb['disk']} snapshot_ref={rb.get('snapshot_ref')}")
        if mism:
            raise L.Abort(f"resource readback mismatch {mism}")
        want = self.cfg.rt.get("require_snapshot_ref")
        if want and rb.get("snapshot_ref") != want:
            raise L.Abort(f"Daytona snapshot {rb.get('snapshot_ref')} != the baseline's {want} — a "
                          f"rebuilt image is a different comparability class (relax RUNTIME "
                          f"daytona_leg.require_snapshot_ref only deliberately)")

    # -- the steps, in the Shard's order
    def prove(self) -> None:
        self.t_created = time.time()
        self.create()
        self.check_granted_ttl()
        self.identity()                      # claude_binaries == 0 BEFORE the CC install
        self.egress()
        if self.cfg.tau3:
            self.upload_tau3()
            self.cgroups()
            self.set_cpu_weights()
            self.tau3_check()
        else:
            self.upload_machine()
            self.cgroups_and_selftest()
            self.set_cpu_weights()
        self.install()
        self.t_ready = time.time()
        self.summary.update(ready_utc=dlx.now(), created_ts=self.t_created, ready_ts=self.t_ready,
                            retire_at_ts=self.retire_at(), ttl_expires_ts=self.ttl_expires(),
                            allow_list=self.allow_csv)
        self.save()
        self.log(f"READY {self.sid} ({self.summary.get('sandbox_name')}): retire_at "
                 f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(self.retire_at()))}, ttl {self.cfg.rt['ttl_minutes']} min")

    def upload_machine(self) -> None:
        """The grading bundle: src/, the runner, the split, the 3 dataset.json, every train+val task's input + golden
        (overlay bytes, canonical paths), /opt/leg/leg.env = the grader knobs only (no solver settings, no key:
        grade_env_text asserts it), the sandbox scripts, /opt/grade/grade_worker.py; the Shard's unpack line."""
        b = self.dir / "bundle.tgz"
        leg_env = G.grade_env_text(self.cfg.knobs, self.cfg.image_id)
        man = L.build_bundle(self.cfg, self.tasks, leg_env, b, [], idx=self.leg.idx,
                             extra=[(GRADE_WORKER_SRC, GRADE_WORKER_VM.lstrip("/"))])
        man.update(grade_worker_sha256=L.sha256_file(GRADE_WORKER_SRC), shim_sha256=L.sha256_file(HERE / "docker_shim.py"),
                   leg_env_keys=sorted(ln.split("=", 1)[0][len("export "):] for ln in leg_env.splitlines()),
                   tasks=len(self.tasks))
        self.summary["bundle"] = man
        t = time.time()
        self.sb.fs.upload_file(str(b), "/tmp/bundle.tgz", timeout=1800)
        b.unlink()
        self.log(f"uploaded bundle {man['bundle_bytes'] / 1e6:.1f} MB ({man['files']} files, {len(self.tasks)} "
                 f"tasks' workbooks) in {time.time() - t:.0f}s")
        self.sh("unpack", UNPACK_CMD, timeout=600, check=True)
        rc, out = self.sh("verify", f"sha256sum /usr/local/bin/docker {GRADE_WORKER_VM} && test -s /opt/leg/leg.env && "
                          f"! grep -q -E 'OPENROUTER|TOKEN|EVOSKILL_SOLVER' /opt/leg/leg.env && echo LEGENV_OK",
                          timeout=120)
        got = _sha_map(out)
        want = {"/usr/local/bin/docker": man["shim_sha256"], GRADE_WORKER_VM: man["grade_worker_sha256"]}
        if got != want or "LEGENV_OK" not in out:
            raise L.Abort(f"machine bundle does not verify: shas {got} != {want}, leg.env "
                          f"{'ok' if 'LEGENV_OK' in out else 'NOT ok'}: {out[-400:]}")
        self.summary["bundle"]["verified_on_machine"] = want

    def upload_tau3(self) -> None:
        """tau3, in upload_machine's place: the data root (INFRA data.root, verified on the Mac by machine_cfg) AT ITS
        MAC PATH -- the frozen tasks' answer files, the bridge's task_file and TAU2_DATA_DIR name it -- uploaded and
        checked as tau3_cc_smoke does (a tgz extracted with no owner; MANIFEST.sha256 re-checked there, no link,
        nothing beyond it); /opt/grade/grade_worker.py (the plug-in runner imports it; InPlaceGrader requires it),
        /opt/leg/leg.env = the grader knobs only (runner_shell sources it), the repo mirror dir runner_shell cds
        into."""
        S, d = _tau3(), self.cfg.data
        root = str(d["root"])
        tgz, envf = self.dir / "data_root.tgz", self.dir / "leg.env"
        tgz.write_bytes(S.data_tgz(Path(root)))
        leg_env = G.grade_env_text(self.cfg.knobs, self.cfg.image_id)
        envf.write_text(leg_env)
        want = {GRADE_WORKER_VM: L.sha256_file(GRADE_WORKER_SRC), f"{root}/{TAU3_MANIFEST}": d["manifest_sha256"]}
        self.summary["bundle"] = {"data_root": root, "manifest_sha256": d["manifest_sha256"], "files": d["files"],
                                  "tasks": d["tasks"], "data_tgz_bytes": tgz.stat().st_size,
                                  "grade_worker_sha256": want[GRADE_WORKER_VM],
                                  "leg_env_keys": sorted(ln.split("=", 1)[0][len("export "):]
                                                         for ln in leg_env.splitlines())}
        t = time.time()
        try:
            self.sb.fs.upload_file(str(tgz), TAU3_DATA_TGZ_VM, timeout=900)
        finally:
            tgz.unlink()
        r = q(root)
        rc, out = self.sh("tau3 data",
                          f"set -e; R={r}; test ! -e \"$R\"; mkdir -p \"$R\"; tar -xzf {TAU3_DATA_TGZ_VM} -C \"$R\" "
                          f"--no-same-owner; rm -f {TAU3_DATA_TGZ_VM}; cd \"$R\"; sha256sum -c --quiet {TAU3_MANIFEST}; "
                          f"test -z \"$(find . -type l)\"; "
                          f"test \"$(find . -type f | wc -l)\" = \"$(( $(wc -l < {TAU3_MANIFEST}) + 1 ))\"; echo DATA_OK",
                          timeout=600)
        if "DATA_OK" not in out:
            raise L.Abort(f"the data root does not verify on the machine: {out[-600:]}")
        self.sh("tau3 dirs", f"mkdir -p {os.path.dirname(GRADE_WORKER_VM)} /opt/leg {q(P.SANDBOX_REPO)}", timeout=60,
                check=True)
        self.sb.fs.upload_file(str(GRADE_WORKER_SRC), GRADE_WORKER_VM, timeout=300)
        self.sb.fs.upload_file(str(envf), "/opt/leg/leg.env", timeout=120)
        rc, out = self.sh("tau3 grade files",
                          f"sha256sum {GRADE_WORKER_VM} {q(root + '/' + TAU3_MANIFEST)} && test -s /opt/leg/leg.env && "
                          f"! grep -q -E 'OPENROUTER|TOKEN|EVOSKILL_SOLVER' /opt/leg/leg.env && "
                          f"test -d {q(P.SANDBOX_REPO)} && echo LEGENV_OK", timeout=120)
        got = _sha_map(out)
        if got != want or "LEGENV_OK" not in out:
            raise L.Abort(f"tau3 machine files do not verify: shas {got} != {want}, leg.env / repo dir "
                          f"{'ok' if 'LEGENV_OK' in out else 'NOT ok'}: {out[-400:]}")
        self.summary["bundle"]["verified_on_machine"] = want
        self.log(f"data root {root} ({d['files']} files, MANIFEST {d['manifest_sha256'][:12]}) mirrored and verified, "
                 f"grade_worker.py + leg.env + {P.SANDBOX_REPO} in place, in {time.time() - t:.0f}s")

    def cgroups(self) -> None:
        """tau3, in cgroups_and_selftest's place: its cgroup delegation (CGROUP_CMD) plus the bridges' cgroup
        /sys/fs/cgroup/evobridge; no shim or SSB grading self-test (a tau3 machine grades nothing through the shim)."""
        rc, out = self.sh("cgroups", f"{CGROUP_CMD}; mkdir -p $CG/{BRIDGE_CGROUP}", timeout=60)
        self.summary["cgroup_setup"] = out.strip()[-400:]
        if "memory" not in out.split("evoshim=")[-1]:
            raise L.Abort("memory controller not delegated to evoshim/ -- per-episode caps impossible")

    def set_cpu_weights(self) -> dict:
        """Rollouts come before grading: cpu.weight of the top-level cgroups, each
        made first if absent -- evoevolver 10000 (solver + evolver containers), evoshim 1 and evoshim_runner 1 (the
        grader), evobridge 10000 on a tau3 machine -- then read back into summary['cgroup_weights']. A readback that
        differs is logged and recorded, never fatal."""
        want = dict(CPU_WEIGHTS, **({BRIDGE_CGROUP: BRIDGE_CPU_WEIGHT} if self.cfg.tau3 else {}))
        cg = ctl.CG_ROOT
        put = "; ".join(f"mkdir -p {cg}/{g} && echo {int(w)} > {cg}/{g}/cpu.weight" for g, w in want.items())
        read = "; ".join(f"echo WEIGHT {g} $(cat {cg}/{g}/cpu.weight 2>/dev/null)" for g in want)
        rc, out = self.sh("cpu weights", f"{put}; {read}", timeout=60)
        got = {}
        for ln in out.splitlines():
            w = ln.split()
            if len(w) == 3 and w[0] == "WEIGHT" and w[2].isdigit():
                got[w[1]] = int(w[2])
        rec = {"want": want, "readback": got, "equal": got == want}
        if not rec["equal"]:
            rec["out_tail"] = out[-400:]
            self.log(f"NOTE: cpu.weight readback {got} != {want} (recorded; the machine is kept): {out[-300:]}")
        else:
            self.log(f"cpu.weight {got}")
        self.summary["cgroup_weights"] = rec
        return rec

    def tau3_check(self) -> None:
        """tau3, in the SSB grading self-test's place (tau3_cc_smoke's machine checks, refused as the smoke refuses):
        the image's /opt/tau3/venv equals its pin; under it, with TAU2_DATA_DIR = the data root and no key in the
        environment, tau2 1.0.1 imports, reads the root's 97 banking tasks, and a key-less GET of openrouter.ai's
        /models answers 200 (the customer's path). -> summary['grade_selftest']."""
        S = _tau3()
        root = str(self.cfg.data["root"])
        rc, venv = self.sh("tau3 venv", f"{ctl.PY} {S.VM_CHECK} --python {S.VM_TAU3_PY} --pin {S.VM_PIN}", timeout=300)
        rc, out = self.sh("tau3 check",
                          f"cd / && env {' '.join('-u ' + k for k in S.KEY_ENV_NAMES)} PYTHON_DOTENV_DISABLED=1 "
                          f"LITELLM_LOCAL_MODEL_COST_MAP=True TAU2_DATA_DIR={q(root)} "
                          f"{S.VM_TAU3_PY} -c {q(S.TAU3_CHECK_PY)}", timeout=600)
        res = S.parse_tail(out, "TAU3_CHECK")
        self.summary["grade_selftest"] = {
            "interpreter": S.VM_TAU3_PY, "data_root": root, "venv_equals_pin": TAU3_VENV_EQUAL in venv,
            "venv_check": [ln for ln in venv.splitlines() if "tau3_check_venv" in ln][-1:], "tau3_check": res}
        probs = []
        if TAU3_VENV_EQUAL not in venv:
            probs.append(f"the image's tau3 venv differs from its pin: {venv[-600:]}")
        if res.get("tau2_version") != TAU3_TAU2_VERSION:
            probs.append(f"tau2 {res.get('tau2_version')!r} ({res.get('import_error')})")
        if res.get("data_dir") != root or res.get("n_tasks") != TAU3_N_TASKS:
            probs.append(f"data dir {res.get('data_dir')!r} with {res.get('n_tasks')} tasks")
        if res.get("openrouter_models_status") != 200:
            probs.append(f"openrouter /models {res.get('openrouter_models_status')} {res.get('openrouter_error')}")
        if probs:
            raise L.Abort("tau3 on the machine: " + "; ".join(probs))
        self.log(f"tau3 venv equals its pin; tau2 {res['tau2_version']}, {res['n_tasks']} tasks under {root}; "
                 f"openrouter /models 200")

    def install(self) -> None:
        up = lambda src, dst, timeout: self.sb.fs.upload_file(src, dst, timeout=timeout)  # noqa: E731
        rec = {"ctl": install_ctl(self.run_sh, up),
               "cc": install_cc(self.run_sh, up, self.pool.cc_local, sha256=self.pool.cc_sha256,
                                nbytes=self.pool.cc_bytes)}
        self.summary["install"] = rec
        self.save()
        self.log(f"evoctl + proxy: {rec['ctl']['ctl']}; Claude Code {ctl.CC_VERSION}: {rec['cc']['cc']}")

    # -- the registry row
    def retire_at(self):
        """No new work after creation + TTL - pool_ttl_margin_min (None before creation)."""
        ttl = float(self.cfg.rt.get("ttl_minutes") or 0) * 60
        if not (self.t_created and ttl):
            return None
        return self.t_created + ttl - float(self.pool.p["pool_ttl_margin_min"]) * 60

    def ttl_expires(self):
        ttl = float(self.cfg.rt.get("ttl_minutes") or 0) * 60
        return self.t_created + ttl if (self.t_created and ttl) else None

    def retiring(self) -> bool:
        ra = self.retire_at()
        if ra is not None and time.time() > ra:
            return True
        return self.disk_free_gib is not None and self.disk_free_gib < P.DISK_RETIRE_GIB

    def usable(self) -> bool:
        return bool(self.t_ready) and not self.released and not self.done.is_set() and not self.failed \
            and not self.retiring()

    def state_name(self) -> str:
        if self.done.is_set():
            return "deleted" if self.deleted else ("failed" if self.failed else "gone")
        if self.failed:
            return "failed"
        if self.released:
            return "released"
        if not self.t_ready:
            return "starting"
        return "retiring" if self.retiring() else "ready"

    def public_record(self, leased_by=()) -> dict:
        """This machine as machine.json shows it: ssb_leg_pool.Machine.public_record's keys, exactly."""
        return {"sandbox_id": self.sid, "sandbox_name": self.summary.get("sandbox_name"), "state": self.state_name(),
                "created_ts": self.t_created, "ready_ts": self.t_ready,
                "ttl_minutes": self.cfg.rt.get("ttl_minutes"), "ttl_expires_ts": self.ttl_expires(),
                "retire_at_ts": self.retire_at(),
                "domain_allow_list": [x.strip() for x in self.allow_csv.split(",") if x.strip()],
                "domain_allow_list_readback": (self.summary.get("readback") or {}).get("domain_allow_list"),
                "units_running": self.units_running, "inflight": self.units_running,
                "cap": int(self.pool.cc["concurrency"]), "disk_free_gib": self.disk_free_gib,
                "leased_by": list(leased_by)}

    def refresh(self) -> None:
        """disk_free_gib (`df /`) and the cc-* containers on the machine (units_running); failures are logged."""
        try:
            rc, out = self.sh("df", "df -k / | tail -1 | awk '{print $4}'", timeout=60)
            kib = int((out.strip().splitlines() or ["x"])[-1])
            self.disk_free_gib = round(kib / (1 << 20), 2)
        except (ValueError, L.Abort, Exception) as e:  # noqa: BLE001 (a transient exec failure)
            self.log(f"df failed: {type(e).__name__}: {str(e)[:120]}")
        try:
            rc, out, lst = self.ctl("list", timeout=120)
            self.units_running = len(cc_cids(lst))
        except Exception as e:  # noqa: BLE001
            self.log(f"evoctl list failed: {type(e).__name__}: {str(e)[:120]}")

    # -- the end
    def wipe_cc_containers(self) -> dict:
        """`evoctl list` -> `down --wipe` every cc-* container; e*-/ao- ones (the evolver's) are left alone."""
        rc, out, lst = self.ctl("list", timeout=120)
        res = {"downed": {}, "left_alone": other_cids(lst)}
        for c in cc_cids(lst):
            rc, out, r = self.ctl(f"down --cid {q(c)} --wipe", timeout=300)
            ok = bool(r.get("verified") and r.get("wiped"))
            res["downed"][c] = {"verified": r.get("verified"), "wiped": r.get("wiped")}
            dlx.ledger(event="container_down", driver=DRIVER, tag=self.cfg.tag, id=self.sid, cid=c, verified=ok)
            self.log(f"container {c}: down --wipe verified={r.get('verified')} wiped={r.get('wiped')}")
        self.summary["cleanup"] = res
        return res

    def delete(self) -> dict:
        """Delete by id (ledgered; dlx refuses any name without the loop prefix), then re-list for up to
        DELETE_VERIFY_S until the listing no longer shows it (the listing lags the delete by seconds)."""
        res = {"sandbox_id": self.sid, "deleted": None, "still_listed": [], "listing_error": None}
        if self.sid:
            res["deleted"] = dlx.delete_by_id(self.leg.client, self.sid, self.cfg.tag)
            t_end = time.time() + DELETE_VERIFY_S
            while True:
                try:
                    rows = dlx.list_ours(self.leg.client)
                    err = None
                except Exception as e:  # noqa: BLE001
                    rows, err = [], f"{type(e).__name__}: {str(e)[:200]}"
                still = [o for o in rows if o.get("id") == self.sid and "destroy" not in str(o.get("state")).lower()]
                if (not still and err is None) or time.time() > t_end:
                    break
                time.sleep(min(10.0, max(0.01, self.pool.tick)))
            res.update(still_listed=still, listing_error=err, listing_utc=dlx.now())
            self.deleted = bool(res["deleted"]) and not still and err is None
        else:
            self.deleted = True
        res["verified_gone"] = self.deleted
        self.t_end = time.time()
        self.summary["delete"] = res
        self.done.set()
        self.save()
        self.log(f"machine {self.sid} deleted={res['deleted']} verified_gone={self.deleted}")
        return res


# ───────────────────────────────────────────────────────────────────────────── the owner (host process)
def registry_record(*, host_pid: int, pool_dir, run_dir, reg: P.Registry, spawned_by=None, previous=None,
                    waited_s: float = 0.0, state: str = "accepting") -> dict:
    """registry.json in the pool's own shape (+ owner): ssb_leg_pool.host_alive reads host_pid/host_script/pool_dir."""
    return {"state": state, "host_pid": int(host_pid), "pool_dir": str(pool_dir), "host_script": str(_host_script()),
            "scope": P.POOL_SCOPE, "run_dir": str(run_dir), "spawned_utc": dlx.now(),
            "spawned_by": dict(spawned_by or {"tag": "", "pid": os.getpid()}), "previous": previous,
            "waited_for_closing_host_s": waited_s, "machines_file": str(reg.machines_file),
            "leases_dir": str(reg.leases_dir), "owner": OWNER}


class PoolMachineOwner:
    """The host: creates and proves the machine, publishes registry.json + machine.json, honours leases (a valid
    lease keeps the machine through STOP), serves until STOP / SIGTERM, then wipes cc-* containers and deletes the
    machine. Also the object a CCMachine calls `leg`: .cfg .store .client .idx .abort_all .spend .slots."""

    placement = pce.PLACEMENT

    def __init__(self, run_dir, *, pool_dir=None, client=None, idx=None, env=None, tick_s=None, cc_local=None,
                 keep_awake=None):
        self.cfg = machine_cfg(run_dir, env=env)
        self.run_dir = self.cfg.run_dir
        self.reg = P.Registry(P.pool_root(self.cfg))
        self.pdir = Path(pool_dir) if pool_dir else self.reg.root / f"{time.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}"
        self.pdir.mkdir(parents=True, exist_ok=True)
        self.store = self.pdir / "machine"
        self.store.mkdir(parents=True, exist_ok=True)
        self.cfg.cache_dir = self.pdir / "empty_cache"     # seed_files finds nothing there
        self.cfg.store = self.store
        self.p = self.cfg.pool
        self.cc = self.cfg.cc
        self._client, self._idx = client, idx
        self.cc_local = Path(cc_local) if cc_local else CC_LOCAL
        self.cc_sha256, self.cc_bytes = CC_SHA256_PIN, CC_BYTES_PIN     # tests substitute a small fake binary
        self.tick = float(os.environ.get("EVOSKILL_CC_HOST_TICK_S", "5") if tick_s is None else tick_s)
        self.keep_awake = (os.environ.get("EVOSKILL_KEEP_AWAKE", "1") == "1") if keep_awake is None \
            else bool(keep_awake)
        self.abort_all = threading.Event()
        self.abort_reason = None
        self.spend = None
        self.slots = None
        self.machines: list = []
        self.machine = None
        self.closing = False
        self.problems: list = []
        self.reaped: list = []
        self.t0 = time.time()
        self.started_utc = dlx.now()
        self._pub = (None, 0.0)
        self._gen = 0
        self.lease_waits: list = []
        self.logf = open(self.pdir / "host.log", "a")

    # -- plumbing
    @property
    def client(self):
        if self._client is None:
            self._client = dlx.client()
        return self._client

    @property
    def idx(self) -> dict:
        if self._idx is None:
            self._idx = L.task_index()
        return self._idx

    def log(self, msg: str) -> None:
        L.log_line(f"[cc_pool {self.pdir.name}] {msg}", self.logf)

    def _abort(self, why: str) -> None:
        if not self.abort_all.is_set():
            self.abort_reason = why
            self.log(f"ABORT: {why}")
            self.abort_all.set()

    def stop_file(self) -> Path:
        return self.reg.root / STOP_FILE

    def rollover_file(self) -> Path:
        return self.reg.root / ROLLOVER_FILE

    def next_name(self) -> str:
        return f"m{len(self.machines) + 1:02d}"

    # -- registry
    def register(self) -> dict:
        """Under the registry lock: the record the spawner wrote for THIS pid + pool dir is kept; a live host of
        another pid is never displaced (never two); a dead / absent one is replaced by this host's record."""
        with self.reg.lock():
            r = self.reg.read()
            if int(r.get("host_pid") or 0) == os.getpid() and Path(str(r.get("pool_dir") or "")) == self.pdir:
                return r
            if r and P.host_alive(r) and r.get("state") in ("accepting", "closing"):
                raise L.Refused(f"the registry names a live host pid {r.get('host_pid')} ({r.get('pool_dir')}, "
                                f"state {r.get('state')!r}); a second host would be a second machine")
            prev = dict(r) if r else None
            if prev:
                prev.pop("previous", None)
            rec = registry_record(host_pid=os.getpid(), pool_dir=self.pdir, run_dir=self.run_dir, reg=self.reg,
                                  spawned_by={"tag": "by hand", "pid": os.getpid()}, previous=prev)
            self.reg.write(rec)
            return rec

    def _registry_update(self, **fields) -> None:
        with self.reg.lock():
            r = self.reg.read()
            if Path(str(r.get("pool_dir") or "")) == self.pdir:
                r.update(fields)
                self.reg.write(r)

    def reap(self) -> list:
        """B1: delete by id every sandbox of this run's pool Daytona still lists; refuse while any stays listed."""
        why = (f"a sandbox of run {self.cfg.run_name}'s pool listed before cc_pool_host pid {os.getpid()} created any "
               f"machine: a previous host's leftover (only use 1 machine)")
        try:
            left = P.reap_pool_sandboxes(self.client, self.cfg.code, self.cfg.tag, self.log, why, found=self.reaped,
                                         pause_s=min(P.REAP_PAUSE_S, max(0.01, self.tick)))
        except Exception as e:  # noqa: BLE001
            raise L.Refused(f"the Daytona listing of this run's pool sandboxes ({dlx.PREFIX}{self.cfg.code}-*) failed "
                            f"({type(e).__name__}: {str(e)[:300]}): no machine is created without it (a crashed "
                            f"host's machine would be a second one alive)")
        if left:
            raise L.Refused(f"sandbox(es) of this run's pool are still listed after delete-by-id: "
                            f"{[(o['name'], o['id'], o['state']) for o in left]} -- a new machine now would be a "
                            f"second one alive (a run uses one machine). Delete them and relaunch")
        self.log(f"{len(self.reaped)} leftover sandbox(es) of this run's pool deleted before any machine")
        return list(self.reaped)

    # -- machines
    def create_machine(self, name: str | None = None) -> CCMachine:
        name = name or self.next_name()
        if any(not m.done.is_set() for m in self.machines):
            raise L.Refused("a machine of this host is still alive: never two")
        m = CCMachine(self, name, list(self.cfg.tasks))
        self.machines.append(m)
        self.machine = m
        self.publish(force=True)                     # state "starting"
        try:
            m.prove()
        except BaseException as e:
            m.failed = f"{type(e).__name__}: {str(e)[:600]}"
            m.save()
            self.problems.append(f"machine {name} failed: {m.failed}")
            self.log(f"MACHINE {name} FAILED: {m.failed}")
            self.publish(force=True)
            if m.sid:                                # never leave a failed machine alive
                try:
                    m.delete()
                except Exception as e2:  # noqa: BLE001
                    self.problems.append(f"delete of failed machine {name}: {type(e2).__name__}: {e2}")
            m.done.set()
            self.publish(force=True)
            raise
        dlx.ledger(event="cc_pool_machine_ready", driver=DRIVER, tag=self.cfg.tag, id=m.sid,
                   name=m.summary.get("sandbox_name"), retire_at_ts=m.retire_at())
        self.publish(force=True)                     # state "ready", current = name
        return m

    def leased(self, m: CCMachine, leases: dict, now: float) -> list:
        """The holders of a valid lease on `m` (ssb_leg_pool.PoolHost._leased's rule: never honoured past the TTL
        minus the sync margin, when the machine must go anyway)."""
        who = leases.get(str(m.sid)) if m.sid else None
        if not who:
            return []
        te = m.ttl_expires()
        if te is not None and now >= te - float(L.POOL_SYNC_MARGIN_MIN) * 60:
            return []
        return list(who)

    def machines_record(self, closed: bool = False) -> dict:
        now = time.time()
        leases = self.reg.leases(now)
        ms = {m.name: m.public_record(self.leased(m, leases, now)) for m in self.machines}
        cur = next((m.name for m in self.machines if m.usable()), None) if not (closed or self.closing) else None
        return {"schema": P.MACHINES_SCHEMA, "run": self.cfg.run_name, "pool_root": str(self.reg.root),
                "pool_dir": str(self.pdir), "host_pid": os.getpid(),
                "host_state": "closed" if closed else ("closing" if self.closing else "open"),
                "current": cur, "machines": ms, "evolver_placement": self.placement,
                "domain_allow_list": sorted(pce.MACHINE_ALLOW), "leases_dir": str(self.reg.leases_dir),
                "lease_rule": LEASE_RULE, "owner": OWNER}

    def publish(self, force: bool = False, closed: bool = False) -> None:
        """machine.json: rewritten when the machine's identity / state / leases change, else once a minute
        (ssb_leg_pool.PoolHost.publish_machines' rule)."""
        try:
            rec = self.machines_record(closed)
            vol = ("units_running", "inflight", "disk_free_gib", "cap")
            key = json.dumps({"c": rec["current"], "s": rec["host_state"],
                              "m": {n: {k: v for k, v in d.items() if k not in vol} for n, d in rec["machines"].items()}},
                             sort_keys=True, default=str)
            last_key, last_t = self._pub
            if not force and key == last_key and time.time() - last_t < 60:
                return
            if key != last_key:
                self._gen += 1
            rec.update(generation=self._gen, updated_utc=dlx.now(), updated_ts=round(time.time(), 1))
            L._atomic_write(self.reg.machines_file, rec)
            self._pub = (key, time.time())
        except OSError as e:
            self.problems.append(f"publish: {type(e).__name__}: {e}")

    def status(self) -> dict:
        now = time.time()
        leases = self.reg.leases(now)
        return {"utc": dlx.now(), "host_pid": os.getpid(), "pool_dir": str(self.pdir), "run": self.cfg.run_name,
                "closing": self.closing, "abort": self.abort_reason, "stop_file": self.stop_file().exists(),
                "machines": {m.name: {"sandbox_id": m.sid, "state": m.state_name(), "retire_at_ts": m.retire_at(),
                                      "leased_by": self.leased(m, leases, now), "disk_free_gib": m.disk_free_gib,
                                      "cc_containers": m.units_running} for m in self.machines},
                "leases": leases, "problems": self.problems, "uptime_s": round(now - self.t0, 1)}

    def write_status(self) -> None:
        try:
            L._atomic_write(self.pdir / "status.json", self.status())
        except OSError:
            pass

    # -- serve
    def serve(self) -> str:
        """Every tick: leases re-read and machine.json rewritten (leased_by), disk / cc-* count refreshed every
        DISK_REFRESH_S; STOP or SIGTERM ends it ('stop' / 'abort'); ROLLOVER (STOP absent) replaces the machine."""
        last_refresh = 0.0
        while True:
            if self.abort_all.is_set():
                return "abort"
            if self.stop_file().exists():
                self.log(f"STOP file {self.stop_file()} present: closing")
                return "stop"
            m = self.machine
            if m is not None and not m.done.is_set() and time.time() - last_refresh >= DISK_REFRESH_S:
                m.refresh()
                last_refresh = time.time()
            self.publish()
            self.write_status()
            if m is not None and not m.done.is_set() and self.rollover_file().exists():
                self.rollover()
                continue
            time.sleep(self.tick)

    def wait_leases(self, m: CCMachine) -> dict:
        """Block while any valid lease holds `m`, at most pool_ttl_margin_min (machine.json keeps showing the
        holders); the wait is recorded."""
        t0, bound = time.time(), float(self.p["pool_ttl_margin_min"]) * 60
        last_log, holders = 0.0, []
        while True:
            now = time.time()
            holders = self.leased(m, self.reg.leases(now), now)
            if not holders:
                break
            if now - t0 > bound:
                self.problems.append(f"{m.name}: lease(s) {holders} still valid after {bound:.0f}s; proceeding")
                self.log(f"{m.name}: leases {holders} still valid after the {bound:.0f}s bound: proceeding")
                break
            if now - last_log > LEASE_WAIT_LOG_S:
                self.log(f"{m.name}: waiting for lease(s) {holders} to end")
                last_log = now
            self.publish()
            time.sleep(self.tick)
        rec = {"machine": m.name, "waited_s": round(time.time() - t0, 1), "holders_at_end": holders}
        self.lease_waits.append(rec)
        return rec

    def retire_machine(self, m: CCMachine) -> None:
        m.released = True
        self.publish(force=True)
        self.wait_leases(m)
        try:
            m.wipe_cc_containers()
        except Exception as e:  # noqa: BLE001
            self.problems.append(f"{m.name}: cc-* cleanup failed: {type(e).__name__}: {str(e)[:200]}")
            self.log(f"{m.name}: cc-* cleanup failed: {type(e).__name__}: {str(e)[:200]}")
        try:
            m.delete()
        except Exception as e:  # noqa: BLE001
            self.problems.append(f"{m.name}: delete failed: {type(e).__name__}: {str(e)[:200]}")
            self.log(f"{m.name}: delete failed: {type(e).__name__}: {str(e)[:200]}")
            m.done.set()
        self.publish(force=True)

    def rollover(self) -> None:
        """Operator's ROLLOVER file: retire + delete the current machine (leases honoured), then create the next
        one -- never two alive. The file is removed once consumed."""
        m = self.machine
        self.log(f"ROLLOVER requested: retiring {m.name}")
        self.retire_machine(m)
        try:
            self.rollover_file().unlink()
        except OSError:
            pass
        if self.stop_file().exists() or self.abort_all.is_set():
            return
        self.create_machine()

    def close(self) -> None:
        """registry closing -> for the live machine: wait leases, wipe cc-*, delete (verified) -> machine.json
        (deleted, host closed) -> registry closed -> pool_record.json. Each step guarded."""
        self.closing = True
        self._registry_update(state="closing", closing_utc=dlx.now())
        self.publish(force=True)
        for m in [m for m in self.machines if not m.done.is_set()]:
            self.retire_machine(m)
        undeleted = [m.name for m in self.machines if m.sid and m.deleted is not True]
        rec = {"pool_dir": str(self.pdir), "owner": OWNER, "run": self.cfg.run_name, "closed_utc": dlx.now(),
               "abort": self.abort_reason, "stop_file": self.stop_file().exists(),
               "machines": {m.name: {k: m.summary.get(k) for k in ("sandbox_id", "sandbox_name", "created_ts",
                                                                  "ready_ts", "delete", "cleanup", "readback")}
                            | {"state": m.state_name(), "failed": m.failed} for m in self.machines},
               "machines_not_verified_deleted": undeleted, "orphans_deleted_at_init": list(self.reaped),
               "lease_waits": self.lease_waits, "started_utc": self.started_utc}
        try:
            if self.machines:
                ours = dlx.list_ours(self.client)
                rec["machines_still_listed"] = [o for o in ours if o.get("id") in {m.sid for m in self.machines}]
        except Exception as e:  # noqa: BLE001
            rec["final_listing_error"] = f"{type(e).__name__}: {e}"
        if undeleted:
            self.problems.append(f"at close: machine(s) not verified deleted {undeleted} (the run's next host "
                                 f"deletes whatever is still listed before it creates a machine)")
        rec["problems"] = self.problems
        try:
            L._atomic_write(self.pdir / "pool_record.json", rec)
        except OSError as e:
            self.log(f"pool_record.json: {e}")
        self.publish(force=True, closed=True)
        self._registry_update(state="closed", closed_utc=dlx.now(), machines_not_verified_deleted=undeleted)
        self.write_status()
        self.log(f"closed: {json.dumps({n: d['state'] for n, d in rec['machines'].items()})}; problems {self.problems}")

    def run(self) -> int:
        """0 = served and closed cleanly; 4 = refused before any machine; 1 = crashed (closed as far as possible)."""
        os.environ.pop("OPENROUTER_API_KEY", None)     # nothing here needs a model key; the machine never gets one
        if self.keep_awake:
            host_tools.keep_awake(os.getpid())         # best effort; a host that cannot be kept awake runs the same

        def on_sig(signum, _frame):
            self._abort(f"signal {signum} to the host")
        try:
            signal.signal(signal.SIGINT, on_sig)
            signal.signal(signal.SIGTERM, on_sig)
        except ValueError:
            pass                                       # not the main thread (tests)
        self.log(f"host pid {os.getpid()} started for run {self.cfg.run_name} (registry {self.reg.root}; machine "
                 f"TTL {self.cfg.rt['ttl_minutes']} min, service until TTL - {self.p['pool_ttl_margin_min']} min; "
                 f"egress {pce.MACHINE_ALLOW_STR}; {len(self.cfg.tasks)} train+val tasks bundled)")
        rc = 0
        try:
            self.register()
            self.reap()
            self.create_machine()
            self.serve()
        except L.Refused as e:
            self.problems.append(f"refused: {e}")
            self.log(f"REFUSED: {e}")
            rc = 4
        except BaseException as e:  # noqa: BLE001
            self.problems.append(f"host crashed: {type(e).__name__}: {e}")
            self.log("HOST CRASH: " + traceback.format_exc()[-2000:])
            self._abort(f"host crashed: {type(e).__name__}: {e}")
            rc = 1
        finally:
            try:
                self.close()
            except BaseException as e:  # noqa: BLE001
                self.log("CLOSE CRASH: " + traceback.format_exc()[-2000:])
                self.problems.append(f"close crashed: {type(e).__name__}: {e}")
                rc = rc or 1
        return rc


# ───────────────────────────────────────────────────────────────────────────── spawn / attach / join
def _host_script() -> Path:
    """This program (EVOSKILL_CC_POOL_HOST_SCRIPT: a stand-in for the contract tests only)."""
    return Path(os.environ.get(HOST_SCRIPT_ENV) or (HERE / "cc_pool_host.py"))


def host_argv(pdir, run_dir) -> list:
    return [sys.executable, str(_host_script()), "host", "--run-dir", str(run_dir), "--pool-dir", str(pdir)]


def spawn_host(pdir, run_dir) -> int:
    """Start the host DETACHED (own session; log <pool dir>/host.out)."""
    argv = host_argv(pdir, run_dir)
    lf = open(Path(pdir) / "host.out", "ab")
    p = subprocess.Popen(argv, cwd=str(L.REPO), stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
    threading.Thread(target=p.wait, daemon=True, name="reap-cc-host").start()
    return p.pid


def _same_dir(a, b) -> bool:
    try:
        return Path(str(a)).resolve() == Path(str(b)).resolve()
    except OSError:
        return str(a) == str(b)


def attach_or_spawn(run_dir, *, tag: str = "", spawn=None, closing_wait_s=None, note=None, poll_s: float = 2.0,
                    require_owner: str | None = OWNER) -> tuple:
    """(registry record, spawned?) under the registry flock (ssb_leg_pool.attach's rule without a join file): the
    run's live, accepting host is joined; a dead / absent one is replaced (a new pool dir, `spawn(pool_dir,
    run_dir) -> pid`, the record written BEFORE the host proves anything so a second driver joins the same host);
    a live host of ANOTHER run dir is refused, so is one that is not a cc_pool_host (require_owner; the DeepSeek
    pool's machine has no grade worker at /opt/grade and a different allow list: pass None to accept it anyway);
    a live host that is closing / closed is WAITED for (bounded by closing_wait_s / EVOSKILL_POOL_CLOSING_WAIT_S,
    default 3600 s) -- never a second host beside it."""
    rd = Path(run_dir).resolve()
    reg = P.Registry(pool_root(rd))
    spawn = spawn_host if spawn is None else spawn
    lim = P._closing_wait_s() if closing_wait_s is None else float(closing_wait_s)
    t0, noted = time.time(), False
    while True:
        with reg.lock():
            r = reg.read()
            alive = bool(r) and P.host_alive(r)
            if r.get("state") == "accepting" and alive:
                if r.get("run_dir") and not _same_dir(r["run_dir"], rd):
                    raise L.Refused(f"the registered pool host pid {r.get('host_pid')} serves run dir {r['run_dir']}, "
                                    f"not {rd}: refusing to lease its machine")
                owner = r.get("owner") or Path(str(r.get("host_script") or "")).stem
                if require_owner and owner != require_owner:
                    raise L.Refused(f"the registered pool host pid {r.get('host_pid')} is {owner!r} "
                                    f"({r.get('host_script')}), not {require_owner!r}: its machine was not prepared "
                                    f"for Claude Code legs (no /opt/grade worker, DeepSeek egress)")
                return dict(r), False
            if not alive:
                for f in (STOP_FILE, ROLLOVER_FILE):   # a STOP / ROLLOVER addressed to the
                    try:                               # dead host would close the NEW host at its first serve-loop
                        (reg.root / f).unlink()        # check, right after it created a machine (the host never
                    except FileNotFoundError:          # clears them itself; `stop` writes STOP and leaves it)
                        pass
                pdir = reg.root / f"{time.strftime('%Y%m%dT%H%M%S')}_{os.getpid()}"
                pdir.mkdir(parents=True, exist_ok=True)
                pid = spawn(pdir, rd)
                prev = dict(r) if r else None
                if prev:
                    prev.pop("previous", None)
                rec = registry_record(host_pid=pid, pool_dir=pdir, run_dir=rd, reg=reg,
                                      spawned_by={"tag": tag, "pid": os.getpid()}, previous=prev,
                                      waited_s=round(time.time() - t0, 1) if noted else 0.0)
                reg.write(rec)
                return rec, True
            waited = time.time() - t0
            if waited > lim:
                raise L.Refused(f"the run's pool host pid {r.get('host_pid')} (pool dir {r.get('pool_dir')}) is still "
                                f"alive in state {r.get('state')!r} after {waited:.0f}s: its machine may still be "
                                f"alive, and a new host now would be a SECOND machine (a run uses one machine). "
                                f"See its host.log / `cc_pool_host.py status --run-dir {rd}`; relaunch once that host "
                                f"exited (EVOSKILL_POOL_CLOSING_WAIT_S={lim:.0f})")
        if not noted and note is not None:
            note(f"the run's pool host pid {r.get('host_pid')} is {r.get('state')!r} (its machine is being released); "
                 f"waiting up to {lim:.0f}s for it to exit before starting a new host")
        noted = True
        time.sleep(poll_s)


def await_lease_ack(regf: Path, row: dict, holder: str, ack_s: float | None = None, poll_s: float | None = None) -> tuple:
    """(True|False, why): the host lists `holder` in leased_by of the picked machine while it is still ready."""
    ack_s = pce.LEASE_ACK_S if ack_s is None else float(ack_s)
    t_end = time.time() + ack_s
    while True:
        try:
            rec = json.loads(Path(regf).read_text())
        except (OSError, ValueError) as e:
            v, why = None, f"machine.json unreadable ({type(e).__name__})"
        else:
            v, why = pce.lease_acknowledged(rec, name=row["name"], sandbox_id=row["sandbox_id"], holder=holder)
        if v is not None:
            return v, why
        if time.time() >= t_end:
            return False, f"{why} after {int(ack_s)} s (is the pool host alive and reading leases/?)"
        time.sleep(max(0.0, min(2.0, ack_s / 20) if poll_s is None else poll_s))


def read_host_summary(reg: dict, name: str) -> dict:
    """<pool dir>/machine/<name>/summary.json of the registered host (what the leg driver copies into runtime_env)."""
    f = Path(str(reg.get("pool_dir") or "")) / "machine" / name / "summary.json"
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError) as e:
        raise L.Refused(f"the host's machine summary {f} is unreadable ({type(e).__name__}: {e})")


def identity_problems(summary: dict, row: dict, rb: dict, inf: dict) -> list:
    """Machine identity at join == the host's summary (sandbox id/name, snapshot, LibreOffice, solver stack)."""
    probs = []
    if summary.get("sandbox_id") != row.get("sandbox_id"):
        probs.append(f"summary sandbox_id {str(summary.get('sandbox_id'))[:8]} != row {str(row.get('sandbox_id'))[:8]}")
    if row.get("sandbox_name") and summary.get("sandbox_name") != row.get("sandbox_name"):
        probs.append("summary sandbox_name != row")
    srb = summary.get("readback") or {}
    if srb.get("snapshot_ref") != rb.get("snapshot_ref"):
        probs.append(f"snapshot at join {rb.get('snapshot_ref')} != the host's {srb.get('snapshot_ref')}")
    ident = summary.get("identity") or {}
    want_lo = (inf.get("leg_sandbox") or {}).get("libreoffice")
    if want_lo and ident.get("libreoffice") != want_lo:
        probs.append(f"LibreOffice {ident.get('libreoffice')!r} != INFRA {want_lo!r}")
    if len(str(ident.get("solver_stack") or "").split()) != 3:
        probs.append(f"solver_stack {ident.get('solver_stack')!r} is not 3 tokens")
    for k in ("egress", "grade_selftest", "install", "bundle"):
        if not summary.get(k):
            probs.append(f"the host's summary lacks {k}")
    return probs


def join(run_dir, holder: str, need_s: float, *, tag: str = "", epoch=None, key=None, info=None, store_dir=None,
         spawn=None, client=None, machine_wait_s=None, ack_s=None, poll_s=None, note=None, closing_wait_s=None,
         dle=None, install: bool = True, cc_local=None, cc_sha256: str = CC_SHA256_PIN,
         cc_bytes: int = CC_BYTES_PIN) -> tuple:
    """(PoolMachineHandle, PoolLease): the leg driver's join (evolver_container.attach_machine's shape). attach_or_spawn -> pick_pool_machine every poll up to cc_leg.machine_wait_s (a fresh host takes ~6-8
    min to create + prove; a host that died meanwhile is respawned) -> PoolLease(holder, cap now + need_s + 1 h)
    written, acknowledged (<= LEASE_ACK_S), renewed from its thread -> fresh readback gated by
    pce.machine_problems (prefix, started, the two-domain allow list, the leg snapshot) -> identity == the host's
    summary -> idempotent installs. The lease is released on every failure after it was written; the caller
    releases it in its `finally` after the last container is wiped, never before."""
    rd = Path(run_dir).resolve()
    if not HOLDER_RE.match(holder or ""):
        raise L.Refused(f"lease holder {holder!r} is not a plain file name")
    inf, _rec = L.load_infra(rd)
    if inf is None:
        raise L.Refused(f"no {L.infra_cfg.INFRA_FILE} in {rd}")
    want_snap = inf["leg_sandbox"].get("snapshot_ref")
    cc = cc_leg_settings(rd)
    wait_s = float(cc["machine_wait_s"] if machine_wait_s is None else machine_wait_s)
    poll = pce.MACHINE_POLL_S if poll_s is None else float(poll_s)
    regf = pce.pool_machines_file(rd)
    reg, spawned = attach_or_spawn(rd, tag=tag, spawn=spawn, note=note, closing_wait_s=closing_wait_s)
    t0, deadline, attempts, last_why, lease, rec = time.time(), time.time() + wait_s, [], None, None, None
    respawns = 0
    while True:
        try:
            rec = json.loads(regf.read_text())
            row, why = pce.pick_pool_machine(rec, now=time.time(), need_s=float(need_s))
        except (OSError, ValueError) as e:
            rec, row, why = None, None, f"no readable pool machine.json at {regf} ({type(e).__name__})"
        if row is not None:
            picked = time.time()
            lease = pce.PoolLease(regf.parent / pce.POOL_LEASES_DIR, holder=holder, sandbox_id=row["sandbox_id"],
                                  cap_ts=picked + float(need_s) + 3600,
                                  info={"tag": tag, "run": rd.name, "epoch": epoch, "key": key, "role": "cc_leg",
                                        **(info or {})})
            lease.write()
            ack, awhy = await_lease_ack(regf, row, holder, ack_s, poll_s)
            if ack:
                lease.start()
                break
            lease.release()
            lease = None
            why = f"lease on {row['name']} not acknowledged: {awhy}"
        attempts.append({"after_s": round(time.time() - t0, 1), "why": why[:300]})
        cur = P.Registry(regf.parent).read()
        if not (cur and P.host_alive(cur)):           # unlike the evolver, a CC leg may start the host: respawn
            reg, sp = attach_or_spawn(rd, tag=tag, spawn=spawn, note=note, closing_wait_s=closing_wait_s)
            spawned = spawned or sp
            respawns += 1 if sp else 0
        if time.time() >= deadline:
            raise L.Refused(f"pool machine: {why} (after {int(time.time() - t0)} s; {CC_LEG_BLOCK}.machine_wait_s "
                            f"{int(wait_s)}; {len(attempts)} attempt(s), {respawns} host respawn(s))")
        if why != last_why and note is not None:
            note(f"pool machine not usable yet: {why}; re-reading for up to {int(deadline - time.time())} s")
            last_why = why
        time.sleep(max(0.0, min(poll, deadline - time.time())))
    try:
        client = dlx.client() if client is None else client
        dle = _dle() if dle is None else dle
        sb = client.get(row["sandbox_id"])
        rb = dle.readback(sb)                        # the FRESH read machine_problems requires (`refreshed`)
        mp = pce.machine_problems(rb, name=sb.name, state=rb.get("state"), row=row, require_snapshot_ref=want_snap,
                                  prefixes=(pce.POOL_PREFIX,))
        if row.get("sandbox_name") and row["sandbox_name"] != sb.name:
            mp.append(f"sandbox name {sb.name!r} != the registry's {row['sandbox_name']!r}")
        if mp:
            raise L.Refused("pool machine: " + "; ".join(mp))
        summary = read_host_summary(reg, row["name"])
        ip = identity_problems(summary, row, rb, inf)
        if ip:
            raise L.Refused("pool machine identity != the host's summary: " + "; ".join(ip))
        handle = PoolMachineHandle(sb, row=row, registry=reg, machine_rec=rec, summary=summary,
                                   dir=store_dir or (Path(reg.get("pool_dir") or ".") / "handles" / holder),
                                   inf=inf, dle=dle, lease=lease, readback=rb,
                                   join_info={"spawned": spawned, "respawns": respawns, "waited_s": round(picked - t0, 1),
                                              "attempts": attempts[-20:], "need_s": float(need_s), "why": why})
        if install:
            handle.install(cc_local, sha256=cc_sha256, nbytes=cc_bytes)
    except BaseException:
        if lease is not None:
            lease.release()
        raise
    dlx.ledger(event="cc_leg_attach", driver=DRIVER, tag=tag or holder, id=sb.id, holder=holder, name=sb.name)
    return handle, lease


# ───────────────────────────────────────────────────────────────────────────── the handle (a leg's view)
def _dle():
    import daytona_lib_evo as dle
    return dle


class PoolMachineHandle:
    """The pool machine as a CC leg's Box / Queue see it: cc_solver.Machine's surface (.sb .sid .name .row .dir
    .lock .rec .inf .dle .ttl .t_created .token_name .sh() .ctl() .save() .identity) WITHOUT creation or deletion
    (delete() refuses: the machine is the host's). Idempotent installs, cc-* listing and an explicit wipe of the
    leg's own containers (never another tag's, never the evolver's)."""

    def __init__(self, sb, *, row: dict, registry: dict, machine_rec: dict | None, summary: dict, dir, inf: dict,
                 dle=None, lease=None, readback=None, join_info=None):
        self.sb = sb
        self.sid = str(row["sandbox_id"])
        self.name = str(row.get("sandbox_name") or getattr(sb, "name", ""))
        self.machine_name = str(row.get("name") or "")
        self.row = dict(row)
        self.registry = dict(registry or {})
        self.summary = dict(summary or {})
        self.identity = dict(self.summary.get("identity") or {})
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.inf = inf
        self.dle = dle if dle is not None else _dle()
        self.d = None                                  # no client of its own: it never creates / deletes
        self.ttl = int(row.get("ttl_minutes") or 0)
        self.t_created = row.get("created_ts")
        self.retire_at_ts = row.get("retire_at_ts")
        self.ttl_expires_ts = row.get("ttl_expires_ts")
        self.token_name = None
        self.lease = lease
        self.rec: dict = {"role": "pool_machine_handle", "owner": OWNER, "sandbox_id": self.sid, "name": self.name,
                          "machine": self.row,
                          "registry": {k: self.registry.get(k) for k in ("host_pid", "pool_dir", "host_script",
                                                                          "run_dir", "owner", "spawned_utc")},
                          "machine_json": {"generation": (machine_rec or {}).get("generation"),
                                           "host_pid": (machine_rec or {}).get("host_pid"),
                                           "updated_utc": (machine_rec or {}).get("updated_utc")},
                          "lease": lease.summary() if lease is not None else None,
                          "readback_join": readback, "identity": self.identity,
                          "host_summary": {k: self.summary.get(k) for k in SUMMARY_KEYS},
                          "join": join_info, "joined_utc": dlx.now()}
        self.save()

    def save(self) -> None:
        if self.lease is not None:
            self.rec["lease"] = {**(self.rec.get("lease") or {}), **self.lease.summary()}
        _atomic_json(self.dir / "MACHINE.json", self.rec)

    def sh(self, cmd: str, *, timeout: int = 600, env=None, what: str = "", retries: int = 3) -> tuple:
        last = None
        for _ in range(max(1, retries)):
            try:
                code, out, _s = self.dle.run(self.sb, cmd, timeout=timeout, env=env)
                return code, out
            except Exception as e:  # noqa: BLE001 -- transient toolbox errors
                last = e
                time.sleep(min(10.0, float(os.environ.get("EVOSKILL_CC_HANDLE_RETRY_S", "10"))))
        raise HandleError(f"exec {what or cmd[:40]!r} failed {retries}x: {type(last).__name__}: {last}")

    def ctl(self, args: str, *, timeout: int = 600, what: str = "", retries: int = 2) -> tuple:
        code, out = self.sh(f"{ctl.PY} {ctl.CTL} {args}", timeout=timeout, what=what or args.split()[0], retries=retries)
        return code, out, parse_json_tail(out)

    def run_sh(self, cmd: str, timeout: int, what: str) -> tuple:
        return self.sh(cmd, timeout=timeout, what=what)

    def install(self, cc_local=None, *, sha256: str = CC_SHA256_PIN, nbytes: int = CC_BYTES_PIN) -> dict:
        """evoctl / proxy / Claude Code, skip-if-sha-matches (the host installed them; the evolver may too)."""
        up = lambda src, dst, timeout: self.sb.fs.upload_file(src, dst, timeout=timeout)  # noqa: E731
        with self.lock:
            rec = {"ctl": install_ctl(self.run_sh, up),
                   "cc": install_cc(self.run_sh, up, cc_local, sha256=sha256, nbytes=nbytes)}
        self.rec["install"] = rec
        self.save()
        return rec

    def containers(self) -> dict:
        """{"cc": [cc-* cids], "other": [the evolver's]} from `evoctl list`."""
        code, out, lst = self.ctl("list", timeout=120, what="list")
        return {"cc": cc_cids(lst), "other": other_cids(lst)}

    def wipe(self, cids, *, tag: str = "") -> dict:
        """`down --wipe` exactly these cc-* containers (a relaunch's leftovers of ITS OWN tag, from its STATE.json);
        any id not starting with cc- is refused. Ledgered per container."""
        bad = [c for c in cids if not str(c).startswith(CC_CID_PREFIX)]
        if bad:
            raise HandleError(f"refusing to wipe non-CC container(s) {bad}: the evolver's containers are not ours")
        res = {}
        for c in cids:
            code, out, r = self.ctl(f"down --cid {q(c)} --wipe", timeout=300, what="down")
            ok = bool(r.get("verified") and r.get("wiped"))
            res[str(c)] = {"verified": r.get("verified"), "wiped": r.get("wiped")}
            self.dle.ledger(event="container_down", driver=DRIVER, tag=tag, id=self.sid, cid=str(c), verified=ok)
        self.rec.setdefault("wipes", []).append({"utc": dlx.now(), "tag": tag, "result": res})
        self.save()
        return res

    def release(self) -> bool | None:
        """Release the lease (after the last container of the leg is wiped, never before)."""
        if self.lease is None:
            return None
        ok = self.lease.release()
        self.rec["lease"] = {**(self.rec.get("lease") or {}), **self.lease.summary(), "released": ok,
                             "released_utc": dlx.now()}
        self.lease = None
        _atomic_json(self.dir / "MACHINE.json", self.rec)
        return ok

    def delete(self):
        raise HandleError("a CC leg never deletes the pool machine: it is the host's (cc_pool_host.py stop)")

    def up(self):
        raise HandleError("the pool machine exists already: PoolMachineHandle never creates one")


# ───────────────────────────────────────────────────────────────────────────── in-place grading
def job_id_for(task_id: str, k: int) -> str:
    """<task_>__s<k> with k the 0-based sample index (the record stem and the kept unit dir use it)."""
    return f"{task_id.replace('/', '_')}__s{int(k)}"


# THE decision on one results.jsonl line is the runner's own: ('ok', float, None) |
# ('grader_failed', None, reason) | ('refused', None, reason); a malformed line is refused (a loop fault).
score_of = PGR.score_of


def runner_argv(*, jobs: str, results: str, work: str, plugin_dir: str, expect: dict, max_procs: int, max_heavy: int,
                heavy_bytes: int, job_timeout: int) -> list:
    """The plug-in grade runner's `run` arguments. `expect` names every file of the plug-in dir (each file PLUGIN.json
    pins, plus tasks.json) with its sha256; the runner refuses (rc 3, no line) a dir that holds anything else."""
    argv = ["run", "--jobs", jobs, "--results", results, "--work", work, "--plugin-dir", plugin_dir]
    for name, sha256 in sorted(expect.items()):
        argv += ["--expect", f"{name}={sha256}"]
    return argv + ["--max-procs", str(int(max_procs)), "--max-heavy", str(int(max_heavy)),
                   "--heavy-bytes", str(int(heavy_bytes)), "--job-timeout", str(int(job_timeout))]


def runner_shell(argv: list, py: str = RUNNER_PY) -> str:
    """THE way the loop grades on the pool machine, in the grader's setup (the SSB grade worker's prefix byte for
    byte, running the plug-in grade runner): the leg's knobs sourced, the runner cgroup joined, `env -u
    OPENROUTER_API_KEY`, the runner's venv, from the repo mirror. The legs' InPlaceGrader runs exactly this. `py` =
    the interpreter the run's grader needs (a tau3 leg's: /opt/tau3/venv/bin/python)."""
    return (f"set -uo pipefail; . /opt/leg/leg.env; echo $$ > /sys/fs/cgroup/evoshim_runner/cgroup.procs "
            f"2>/dev/null; cd {P.SANDBOX_REPO} && env -u OPENROUTER_API_KEY {py} -u "
            f"{RUNNER_VM} " + " ".join(argv))


# install()'s /data step: a link is made only when absent or pointing elsewhere, by a rename (a concurrent train and
# val leg never see it missing); a real directory or file there is reported, never replaced.
LINK_FN = ('link() { if [ -L "$2" ] && [ "$(readlink "$2")" = "$1" ]; then echo "KEPT $2"; '
           'elif [ -e "$2" ] && [ ! -L "$2" ]; then echo "NOT_A_LINK $2"; '
           'else ln -s "$1" "$2.tmp.$$" && mv -Tf "$2.tmp.$$" "$2" && echo "MADE $2"; fi; '
           'echo "READLINK $2 -> $(readlink "$2")"; }')


class InPlaceGrader:
    """Grading on the pool machine, incremental, by the run's frozen grade(task, snapshot)
    (scripts/daytona/plugin_grade_runner.py, in the SSB grade worker's setup).
    install() once per leg, before any rollout: the runner and the pinned plug-in files + tasks.json on the machine
    (sha-verified), the user's material linked at its /data paths (plugin_runner.material_layout), every task file of
    the leg == the Mac's overlay bytes. Per collected unit: keep_snapshot (the container's whole out.tgz kept under
    /opt/cc/<run>/<tag>/units/<job>/, sha == the Mac's container_out.tgz, else re-uploaded from it), job() (the 6-key
    line), enqueue() (one line appended to jobs.jsonl; the runner launched when none runs -- one per leg, RESUME by
    job_id), poll() (results.jsonl downloaded; the NEW lines, decided by score_of), regrade() (ONCE per unit, job_id
    <job>#regrade1, the same kept tgz), cleanup() (units + work dir; results and the plug-in dir stay).
    runner_py = the interpreter the runner (and so the grader's child) runs under; RUNNER_PY unless the leg names
    another (a tau3 leg: /opt/tau3/venv/bin/python, where tau2 is)."""

    def __init__(self, handle: PoolMachineHandle, *, run_name: str, tag: str, knobs: dict, plugin: dict,
                 task_rows: dict, layout: dict, overlay=None, max_procs: int = 4, max_heavy: int = 1, heavy_bytes=None,
                 job_timeout: int = 5400, store=None, log=None, runner_py: str = RUNNER_PY):
        if not NAME_RE.match(run_name) or not NAME_RE.match(tag):
            raise GradeError(f"run {run_name!r} / tag {tag!r} must be [A-Za-z0-9_.-]+ (they name machine paths)")
        files = dict(plugin.get("files") or {})
        if not files or PGR.TASKS_JSON in files or not all(NAME_RE.match(n) for n in files):
            raise GradeError(f"the frozen plug-in's files {sorted(files)} must be plain [A-Za-z0-9_.-]+ names "
                             f"(they name machine paths), without {PGR.TASKS_JSON}")
        self.h, self.run_name, self.tag = handle, run_name, tag
        self.base = f"{CC_ROOT_VM}/{run_name}/{tag}"
        self.units_dir = f"{self.base}/units"
        self.grade_dir = f"{self.base}/grade"
        self.plugin_dir = f"{self.base}/plugin"
        self.jobs_file = f"{self.grade_dir}/jobs.jsonl"
        self.results_file = f"{self.grade_dir}/results.jsonl"
        self.rc_file = f"{self.grade_dir}/rc"
        self.worker_log = f"{self.grade_dir}/worker.log"
        self.meta_file = f"{self.grade_dir}/worker_meta.json"
        self.work = f"{GRADE_WORK_VM}/{tag}"
        # the upload sources: the pinned setup files, and the frozen setup/tasks.json beside them
        self.plugin_src = {n: Path(f["path"]) for n, f in files.items()}
        self.plugin_src[PGR.TASKS_JSON] = Path(next(iter(files.values()))["path"]).parent / PGR.TASKS_JSON
        self.expect = {n: str(f["sha256"]) for n, f in files.items()}
        self.expect[PGR.TASKS_JSON] = str(plugin.get("tasks_json_sha256"))
        self.task_rows = dict(task_rows or {})
        self.layout = dict(layout or {})
        self.knobs, self.overlay = dict(knobs or {}), overlay
        self.max_procs, self.max_heavy = int(max_procs), int(max_heavy)
        self.heavy_bytes = int(heavy_bytes or self.knobs.get("EVOSKILL_HEAVY_COMPARE_BYTES") or 5_000_000)
        self.job_timeout = int(job_timeout)
        self.runner_py = str(runner_py)
        self.runner_sha256 = L.sha256_file(RUNNER_SRC)
        self.store = Path(store) if store else handle.dir / "grade"
        self.store.mkdir(parents=True, exist_ok=True)
        self.log = log or (lambda m: None)
        self.setup = None
        self.jobs_sent: dict = {}
        self.seen: dict = {}
        self.regrades: dict = {}
        self.launches = 0
        self.statuses: Counter = Counter()
        self.kept: dict = {}

    # -- install (once per leg, before any rollout)
    def install(self, task_ids) -> dict:
        """The grader's setup for this leg, proven before any rollout (the driver FAILS the leg on a GradeError): the
        runner (+ the bundle's grade_worker.py it imports), the plug-in dir, the /data links, the leg's task files.
        Idempotent: what the machine already holds with the right sha is not uploaded again. -> self.setup (the leg
        JSON's daytona.grade.setup)."""
        rec = {"runner": self._install_runner(), "plugin": self._install_plugin(), "links": self._link_material(),
               "task_files": self._check_task_files(task_ids), "utc": dlx.now()}
        self.setup = rec
        self.log(f"grader installed: runner {'uploaded' if rec['runner']['uploaded'] else 'already there'}, plug-in "
                 f"{len(rec['plugin']['uploaded'])} file(s) uploaded, {len(rec['links'])} /data link(s), "
                 f"{rec['task_files']['n']} task file(s) == the Mac's bytes")
        return rec

    def _code_shas(self) -> dict:
        rc, out = self.h.sh(f"sha256sum {RUNNER_VM} {GRADE_WORKER_VM} 2>/dev/null", timeout=120, what="grader code sha")
        return _sha_map(out)

    def _install_runner(self) -> dict:
        """The runner at RUNNER_VM, uploaded unless the machine holds these bytes and put in place by a rename (a leg
        launching it meanwhile never reads a partial file); it and the bundle's grade_worker.py must equal the Mac's."""
        want = {RUNNER_VM: self.runner_sha256, GRADE_WORKER_VM: L.sha256_file(GRADE_WORKER_SRC)}
        got = self._code_shas()
        uploaded = got.get(RUNNER_VM) != want[RUNNER_VM]
        if uploaded:
            part = f"{RUNNER_VM}.{self.tag}.part"
            self.h.sb.fs.upload_file(str(RUNNER_SRC), part, timeout=300)
            self.h.sh(f"mv -f {q(part)} {q(RUNNER_VM)}", timeout=60, what="install runner")
            got = self._code_shas()
        if got != want:
            raise GradeError(f"the grading code on the machine does not verify ({got} != {want}): the runner, or the "
                             f"bundle's grade_worker.py it imports (a machine made from other code)")
        return {"uploaded": uploaded, "sha256": want}

    def _plugin_state(self) -> tuple:
        """(the names in the plug-in dir, {name: sha256} of the expected ones) on the machine."""
        d = q(self.plugin_dir)
        paths = " ".join(q(f"{self.plugin_dir}/{n}") for n in sorted(self.expect))
        rc, out = self.h.sh(f"mkdir -p {d}; echo \"FILES=$(ls -A {d} | tr '\\n' '/')\"; sha256sum {paths} 2>/dev/null",
                            timeout=120, what="plug-in state")
        names = next((sorted(x for x in ln[len("FILES="):].split("/") if x) for ln in out.splitlines()
                      if ln.startswith("FILES=")), [])
        return names, {Path(p).name: s for p, s in _sha_map(out).items()}

    def _install_plugin(self) -> dict:
        """The leg's plug-in dir holds exactly the pinned files and tasks.json (the runner refuses any other file): a
        file whose sha differs is uploaded, a name the frozen plug-in does not pin is removed."""
        names, got = self._plugin_state()
        extra = [n for n in names if n not in self.expect]
        todo = [n for n in sorted(self.expect) if got.get(n) != self.expect[n]]
        if extra:
            self.h.sh("rm -rf -- " + " ".join(q(f"{self.plugin_dir}/{n}") for n in extra), timeout=60,
                      what="plug-in extras")
        for n in todo:
            self.h.sb.fs.upload_file(str(self.plugin_src[n]), f"{self.plugin_dir}/{n}", timeout=300)
        if extra or todo:
            names, got = self._plugin_state()
        if names != sorted(self.expect) or got != self.expect:
            short = {n: s[:12] for n, s in got.items()}
            raise GradeError(f"the plug-in dir {self.plugin_dir} does not verify: it holds {names} with shas {short}, "
                             f"not the frozen {sorted(self.expect)}")
        return {"dir": self.plugin_dir, "uploaded": todo, "removed": extra, "sha256": dict(self.expect)}

    def _link_material(self) -> dict:
        """/data/<name> -> <Mac path> for the layout (plugin_runner.material_layout: the adapter's view of the user's
        material, so a grader written against it runs unchanged); the targets are the machine's mirrors of the Mac
        paths. A real directory or file at /data/<name> is a GradeError, and every link is read back."""
        if not self.layout:
            return {}
        parents = sorted({os.path.dirname(vm) for vm in self.layout.values()})
        rc, out = self.h.sh(f"set -u; mkdir -p {' '.join(q(d) for d in parents)} || exit 1; {LINK_FN}; " +
                            "; ".join(f"link {q(mac)} {q(vm)}" for mac, vm in sorted(self.layout.items())),
                            timeout=120, what="data links")
        seen = {}
        for ln in out.splitlines():
            for word in ("KEPT", "MADE", "NOT_A_LINK"):
                if ln.startswith(word + " "):
                    seen.setdefault(ln[len(word) + 1:], {})["action"] = word.lower()
            if ln.startswith("READLINK "):
                vm, _, tgt = ln[len("READLINK "):].partition(" -> ")
                seen.setdefault(vm, {})["readlink"] = tgt
        res = {}
        for mac, vm in sorted(self.layout.items()):
            r = seen.get(vm) or {}
            if r.get("action") == "not_a_link":
                raise GradeError(f"{vm} on the machine is a real directory or file, not a link: the grader's /data view "
                                 f"({vm} -> {mac}) cannot be made")
            if r.get("readlink") != mac:
                raise GradeError(f"{vm} on the machine reads back {r.get('readlink')!r}, not {mac} (rc {rc}): "
                                 f"{out[-300:]}")
            res[vm] = {"target": mac, "action": r.get("action")}
        return res

    def _check_task_files(self, task_ids) -> dict:
        """Every input and reference file of the leg's tasks, on the machine at its frozen path, == the Mac's bytes
        through the data overlay (the bundle put them there; this proves it once per leg). GradeError on the first
        mismatch: the grader would read other bytes than the task was frozen with."""
        unknown = sorted(set(task_ids) - set(self.task_rows))
        if unknown:
            raise GradeError(f"task(s) {unknown[:5]} are not in the frozen tasks.json")
        paths = sorted({str(p) for t in task_ids
                        for p in list(self.task_rows[t]["inputs"]) + list(self.task_rows[t]["answer_files"])})
        want = {}
        for p in paths:
            try:
                want[p] = L.sha256_file(data_overlay.source(p, self.overlay))
            except OSError as e:
                raise GradeError(f"task file {p}: the Mac's copy is unreadable ({type(e).__name__}: {e})")
        got = {}
        for i in range(0, len(paths), TASK_FILES_PER_CALL):
            rc, out = self.h.sh("sha256sum -- " + " ".join(q(p) for p in paths[i:i + TASK_FILES_PER_CALL]) +
                                " 2>/dev/null", timeout=600, what="task files sha")
            got.update(_sha_map(out))
        for p in paths:
            if got.get(p) != want[p]:
                raise GradeError(f"task file {p} on the machine ({(got.get(p) or 'missing')[:12]}) is not the Mac's "
                                 f"bytes through the data overlay ({want[p][:12]})")
        return {"n": len(paths), "sha256": want}

    # -- per unit
    def tgz_path(self, job_id: str) -> str:
        """The kept out.tgz of a unit (a #regrade1 job grades the same one)."""
        return f"{self.units_dir}/{job_id.split('#regrade', 1)[0]}/out.tgz"

    def keep_cmd(self, job_id: str, cdir: str) -> str:
        d = os.path.dirname(self.tgz_path(job_id))
        return (f"set -e; mkdir -p {q(d)} && cp {q(cdir + '/state/out.tgz')} {q(self.tgz_path(job_id))} && "
                f"sha256sum {q(self.tgz_path(job_id))} | cut -d' ' -f1")

    def _upload_tgz(self, mac_tgz, path: str) -> str:
        self.h.sh(f"mkdir -p {q(os.path.dirname(path))}", timeout=60, what="mkdir unit")
        self.h.sb.fs.upload_file(str(mac_tgz), path, timeout=900)
        rc, out = self.h.sh(f"sha256sum {q(path)} | cut -d' ' -f1", timeout=120, what="rehash")
        return (out.strip().splitlines() or [""])[-1].strip()

    def keep_snapshot(self, job_id: str, cdir, expect_sha256: str, mac_tgz=None) -> dict:
        """The container's whole out.tgz kept on the machine BEFORE the container is wiped; its sha must equal the
        Mac's container_out.tgz (the same bytes, downloaded at collect). A mismatch is recorded and the Mac copy
        uploaded (mac_tgz); still wrong -> GradeError. cdir=None uploads the Mac copy directly (regrade_leg.py)."""
        path = self.tgz_path(job_id)
        res = {"path": path, "sha256": None, "expected": expect_sha256, "matches": False, "reuploaded": False,
               "rc": None, "from": "container" if cdir is not None else "mac"}
        if cdir is not None:
            rc, out = self.h.sh(self.keep_cmd(job_id, cdir), timeout=300, what="keep snapshot")
            got = (out.strip().splitlines() or [""])[-1].strip()
            res.update(sha256=got, matches=got == expect_sha256, rc=rc)
        if not res["matches"] and mac_tgz is not None:
            if cdir is not None:
                self.log(f"{job_id}: kept snapshot sha {str(res['sha256'])[:12]} != the Mac's {expect_sha256[:12]}: "
                         f"re-uploading the Mac copy")
            got = self._upload_tgz(mac_tgz, path)
            res.update(sha256=got, matches=got == expect_sha256, reuploaded=cdir is not None)
        if not res["matches"]:
            raise GradeError(f"{job_id}: the snapshot on the machine ({str(res['sha256'])[:12]}) is not the Mac's "
                             f"container_out.tgz ({expect_sha256[:12]})")
        self.kept[job_id] = res
        return res

    def job(self, *, job_id: str, task_id: str, snapshot_sha256: str, cid=None, collected_utc=None, unit=None) -> dict:
        """The grade job (plugin_grade_runner's schema): the task by its id in the frozen tasks.json, the kept tgz with
        the Mac's sha, the solver's working directory inside it. Nothing benchmark-specific."""
        if task_id not in self.task_rows:
            raise GradeError(f"{job_id}: task {task_id!r} is not in the frozen tasks.json")
        j = {"job_id": job_id, "task_id": task_id, "snapshot_tgz": self.tgz_path(job_id),
             "snapshot_sha256": snapshot_sha256, "snapshot_root": SNAPSHOT_ROOT,
             "meta": {"unit": unit or job_id, "tag": self.tag, "cid": cid, "collected_utc": collected_utc or dlx.now()}}
        try:
            PGR.validate_job(j)
        except PGR.JobError as e:
            raise GradeError(f"{job_id}: {e}")
        return j

    def enqueue(self, job: dict) -> dict:
        """Append one job line on the machine (and to the store's copy); launch the worker if none runs."""
        jid = job["job_id"]
        if jid in self.jobs_sent:
            raise GradeError(f"{jid}: already enqueued")
        line = json.dumps(job, sort_keys=True)
        b64 = base64.b64encode((line + "\n").encode()).decode()
        rc, out = self.h.sh(f"set -e; mkdir -p {q(self.grade_dir)}; echo {b64} | base64 -d >> {q(self.jobs_file)}; "
                            f"wc -l < {q(self.jobs_file)}", timeout=120, what="enqueue job")
        if rc != 0:
            raise GradeError(f"{jid}: enqueue failed rc={rc}: {out[-300:]}")
        self.jobs_sent[jid] = job
        with open(self.store / "jobs.jsonl", "a") as fh:
            fh.write(line + "\n")
        return self.ensure_worker()

    def worker_argv(self) -> list:
        return runner_argv(jobs=self.jobs_file, results=self.results_file, work=self.work, plugin_dir=self.plugin_dir,
                           expect=self.expect, max_procs=self.max_procs, max_heavy=self.max_heavy,
                           heavy_bytes=self.heavy_bytes, job_timeout=self.job_timeout)

    def launch_cmd(self) -> str:
        """runner_shell, detached: the leg's knobs sourced, the runner cgroup joined, `env -u
        OPENROUTER_API_KEY`, rc file at exit."""
        inner = runner_shell(self.worker_argv(), self.runner_py) + f" >> {self.worker_log} 2>&1; echo $? > {self.rc_file}"
        if "'" in inner:
            raise GradeError("a single quote in the worker command line")
        return f"(setsid nohup bash -c '{inner}' </dev/null >/dev/null 2>&1 &); sleep 2"   # the sleep: ssb_grade_daytona's
        # launch pattern (its `sleep 5`) -- the exec must not return before the detached worker has started

    def worker_pattern(self) -> str:
        # `pgrep -f <pattern>` would match the probe's OWN shell (the SDK exec's argv carries the
        # pattern), so worker_state would read RUNNING while no worker exists and never launch one. Bracketing
        # the first character keeps the regex from matching its own text (the standing pgrep rule).
        return f"[/]{RUNNER_VM[1:]} run --jobs {self.jobs_file}"

    def worker_state(self) -> dict:
        rc, out = self.h.sh(f"if pgrep -f {q(self.worker_pattern())} >/dev/null 2>&1; then echo WORKER=RUNNING; else "
                            f"echo WORKER=IDLE; fi; echo RC=$(cat {q(self.rc_file)} 2>/dev/null); "
                            f"echo RESULTS=$(test -f {q(self.results_file)} && wc -l < {q(self.results_file)} || echo 0)",
                            timeout=120, what="worker state")
        kv = {}
        for ln in out.splitlines():
            if "=" in ln:
                k, v = ln.split("=", 1)
                kv[k.strip()] = v.strip()
        rcv = kv.get("RC") or None
        return {"running": kv.get("WORKER") == "RUNNING", "rc": rcv, "results_lines": int(kv.get("RESULTS") or 0)}

    def ensure_worker(self) -> dict:
        """Launch the worker when none runs and jobs remain without a result (a worker exits when its list drains;
        RESUME skips done ids)."""
        st = self.worker_state()
        pending = [j for j in self.jobs_sent if j not in self.seen]
        if st["running"] or not pending:
            return {"launched": False, **st, "pending": len(pending)}
        rc, out = self.h.sh(f"rm -f {q(self.rc_file)}; mkdir -p {q(self.grade_dir)}; " + self.launch_cmd(), timeout=60,
                            what="launch grade worker")
        self.launches += 1
        self.log(f"grade worker launched (#{self.launches}; {len(pending)} job(s) pending; max_procs {self.max_procs}, "
                 f"max_heavy {self.max_heavy})")
        return {"launched": True, "launch_n": self.launches, **st, "pending": len(pending)}

    def poll(self) -> list:
        """Download results.jsonl (+ worker_meta.json, worker.log) into the store; return the lines of THIS leg's
        jobs not seen before, in file order (a torn last line is left for the worker to repair)."""
        local = self.store / "results.jsonl"
        try:
            self.h.sb.fs.download_file(self.results_file, str(local))
        except Exception:  # noqa: BLE001 -- no results yet
            return []
        new = []
        try:
            text = local.read_text()
        except OSError:
            return []
        for ln in text.splitlines():
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            jid = r.get("job_id")
            if jid in self.seen or jid not in self.jobs_sent:
                continue
            self.seen[jid] = r
            self.statuses[str(r.get("status"))] += 1
            new.append(r)
        for name, src in (("worker_meta.json", self.meta_file), ("worker.log", self.worker_log)):
            try:
                self.h.sb.fs.download_file(src, str(self.store / name))
            except Exception:  # noqa: BLE001
                pass
        return new

    def download(self) -> dict:
        self.poll()
        return self.summary()

    def regrade(self, job_id: str) -> dict:
        """ONE re-grade of a unit from the same on-machine copy (job_id <unit>#regrade1); a second is refused."""
        base_id = job_id.split("#regrade", 1)[0]
        if base_id in self.regrades:
            raise GradeError(f"{base_id}: already re-graded once ({self.regrades[base_id]}): the leg is FAILED")
        src = self.jobs_sent.get(job_id) or self.jobs_sent.get(base_id)
        if src is None:
            raise GradeError(f"{job_id}: no job was sent for it")
        j = dict(src, job_id=f"{base_id}#regrade1", meta={**(src.get("meta") or {}), "regrade_of": job_id})
        self.regrades[base_id] = j["job_id"]
        self.enqueue(j)
        return j

    def enqueue_unit(self, *, job_id: str, task_id: str, cdir, snapshot_sha256: str, mac_tgz=None, cid=None,
                     unit=None, collected_utc=None) -> dict:
        """One collected unit, end to end: keep_snapshot (BEFORE its container is wiped) + job + enqueue -- the
        driver's `on_collected`; every completed unit gets one (the grader decides what a rollout that left nothing
        scores). Returns {"kept", "job", "worker"}."""
        kept = self.keep_snapshot(job_id, cdir, snapshot_sha256, mac_tgz=mac_tgz)
        j = self.job(job_id=job_id, task_id=task_id, snapshot_sha256=snapshot_sha256, cid=cid,
                     collected_utc=collected_utc, unit=unit)
        return {"kept": kept, "job": j, "worker": self.enqueue(j)}

    def pending(self) -> list:
        return [j for j in self.jobs_sent if j not in self.seen]

    def cleanup(self) -> dict:
        """After the leg's result: units (the kept snapshots) + the runner's work dir go; results.jsonl and the
        plug-in dir stay until the host closes."""
        rc, out = self.h.sh(f"rm -rf {q(self.units_dir)} {q(self.work)}; echo CLEAN_RC=$?", timeout=300, what="cleanup")
        return {"units_dir": self.units_dir, "work": self.work, "ok": "CLEAN_RC=0" in out}

    def plugin_record(self) -> dict:
        return {"files": {n: s for n, s in sorted(self.expect.items()) if n != PGR.TASKS_JSON},
                "tasks_json_sha256": self.expect[PGR.TASKS_JSON]}

    def grader_record(self) -> dict:
        """runtime_env['grader']: what graded the leg's rollouts."""
        return {"worker": PGR.WORKER_VERSION, "runner_sha256": self.runner_sha256, "plugin": self.plugin_record(),
                "interpreter": self.runner_py}

    def summary(self) -> dict:
        """The leg JSON's `daytona.grade` block."""
        return {"worker": PGR.WORKER_VERSION, "runner_sha256": self.runner_sha256, "plugin": self.plugin_record(),
                "layout": dict(self.layout), "setup": self.setup,
                "results": str(self.store / "results.jsonl"), "jobs": len(self.jobs_sent),
                "statuses": dict(self.statuses), "in_place": True, "max_procs": self.max_procs,
                "max_heavy": self.max_heavy, "heavy_bytes": self.heavy_bytes, "job_timeout_s": self.job_timeout,
                "launches": self.launches, "regrades": dict(self.regrades), "pending": len(self.pending()),
                "machine": {"base": self.base, "jobs": self.jobs_file, "results": self.results_file, "work": self.work,
                            "plugin_dir": self.plugin_dir}}


# the owner class exposes these too; the functions above are the definitions
PoolMachineOwner.attach_or_spawn = staticmethod(attach_or_spawn)
PoolMachineOwner.join = staticmethod(join)


# ───────────────────────────────────────────────────────────────────────────── commands
def cmd_host(a) -> int:
    return PoolMachineOwner(a.run_dir, pool_dir=a.pool_dir).run()


def cmd_status(a) -> int:
    rd = Path(a.run_dir).resolve()
    root = pool_root(rd)
    reg = P.Registry(root)
    r = reg.read()
    if not r:
        print(f"no registry under {root}")
        return 0
    print(f"{root}: state {r.get('state')} host {r.get('host_pid')} alive {P.host_alive(r)} owner {r.get('owner')} "
          f"dir {r.get('pool_dir')}")
    try:
        mf = json.loads(reg.machines_file.read_text())
        print(f"  machine.json: current {mf.get('current')} host {mf.get('host_state')} generation "
              f"{mf.get('generation')} updated {mf.get('updated_utc')}")
        for n, d in (mf.get("machines") or {}).items():
            ra = d.get("retire_at_ts")
            print(f"  {n}: {d.get('sandbox_id')} {d.get('sandbox_name')} {d.get('state')} retire_at "
                  f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(ra)) if ra else '-'} disk_free "
                  f"{d.get('disk_free_gib')} GiB cc-containers {d.get('units_running')} leased_by {d.get('leased_by')}")
    except (OSError, ValueError, TypeError):
        print("  machine.json: none")
    print(f"  leases now: {reg.leases(time.time())}")
    for f in (STOP_FILE, ROLLOVER_FILE):
        if (root / f).exists():
            print(f"  {f} present")
    try:
        st = json.loads((Path(r.get("pool_dir") or "") / "status.json").read_text())
        print(f"  status: {json.dumps({k: st.get(k) for k in ('utc', 'closing', 'abort', 'problems')})}")
    except (OSError, ValueError, TypeError):
        pass
    return 0


def cmd_stop(a) -> int:
    """Write <pool root>/STOP; wait (--wait-s) for the registry to say closed = the machine deleted (verified in
    the pool dir's pool_record.json)."""
    rd = Path(a.run_dir).resolve()
    root = pool_root(rd)
    reg = P.Registry(root)
    r = reg.read()
    if not r or not P.host_alive(r):
        print(f"no live host registered under {root} (state {r.get('state') if r else None}); nothing to stop")
        return 0
    (root / STOP_FILE).write_text(json.dumps({"utc": dlx.now(), "by_pid": os.getpid()}) + "\n")
    print(f"STOP written for host pid {r.get('host_pid')}; waiting up to {a.wait_s:.0f}s for it to close")
    t_end = time.time() + float(a.wait_s)
    while time.time() < t_end:
        r = reg.read()
        if r.get("state") == "closed":
            rec = {}
            try:
                rec = json.loads((Path(r.get("pool_dir") or "") / "pool_record.json").read_text())
            except (OSError, ValueError):
                pass
            print(f"closed at {r.get('closed_utc')}; machines {json.dumps({n: d.get('state') for n, d in (rec.get('machines') or {}).items()})}; "
                  f"still listed {rec.get('machines_still_listed')}; problems {rec.get('problems')}")
            return 0 if not r.get("machines_not_verified_deleted") else 5
        if not P.host_alive(r):
            print(f"the host pid {r.get('host_pid')} exited without closing the registry (state {r.get('state')})")
            return 5
        time.sleep(min(5.0, max(0.2, a.wait_s / 100)))
    print(f"still {r.get('state')} after {a.wait_s:.0f}s (leases keep the machine until they end)")
    return 5


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("host")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--pool-dir", default=None)
    p = sub.add_parser("status")
    p.add_argument("--run-dir", required=True)
    p = sub.add_parser("stop")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--wait-s", type=float, default=1800.0)
    a = ap.parse_args(argv)
    return {"host": cmd_host, "status": cmd_status, "stop": cmd_stop}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
