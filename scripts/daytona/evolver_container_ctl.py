#!/usr/bin/env python3
"""THE EVOLVER CONTAINER on the run's pool machine -- the MACHINE side (stdlib only, python >= 3.9).

The run uses ONE machine: the evolver runs in an isolated container on the leg machine, next to the rollouts,
instead of on a machine of its own. The held-out answers sit on that machine's disk, so the container sees only what
its upload set and mount plan give it.

Uploaded to /opt/evoskill-evolver/bin/evoctl.py (with scripts/evolver_egress_proxy.py beside it) and run as root
through Daytona SDK exec calls by scripts/daytona/evolver_container.py (the host driver). The host driver also
imports this file for the constants, the mount plan and the generated scripts, so there is ONE definition of what
the container sees; tests/test_evolver_container.py replays the generated script in a namespace simulator.

WHERE THINGS LIVE ON THE MACHINE (all under /opt, which every solver episode sees as an EMPTY tmpfs: the pinned
docker shim's HIDE list is /Users, /opt, /home, /var/lib/evoshim, /var/log/evoshim):
  /opt/evoskill-evolver/                      root 0700
    bin/evoctl.py, bin/evolver_egress_proxy.py
    cc/2.1.281/claude-code.bin                the Claude Code linux-x64 binary uploaded from the host (sha256 = INFRA
                                              claude_code.linux_x64_sha256). Never named `claude` on the machine: the
                                              leg identity probe counts files named claude.
    c/<cid>/                                  one container, root 0700
      root/     mount point of the container's tmpfs root
      etc/      the machine's /etc copied, minus secrets and host identity (ETC_EXCLUDE), plus user `evo`
      fs/       THE UPLOAD SET, extracted at its manifest paths: fs/<mirrored host paths>, fs/evo/..., fs/etc/claude-code/...
      rw/home/  evo's HOME (a copy of the solver's warm LibreOffice profile)
      state/    spec/facts/limits/plan/init script, init.log, proxy log, state.json, upload.tgz, out.tgz

WHAT THE CONTAINER SEES (plan(); a fresh mount namespace, pivot_root onto a tmpfs, the old root DETACHED):
  /usr                          the machine image's /usr, read-only (python 3.11 + the solver stack, LibreOffice)
                                with /usr/local/bin/docker (the leg's docker shim) masked by /dev/null
  /bin /sbin /lib /lib64 ...    the image's own symlinks into /usr (or read-only binds where they are dirs)
  /etc                          c/<cid>/etc, read-only;  /etc/claude-code = fs/etc/claude-code, read-only
  /Users                        fs/Users, read-only: the prompt, the evidence, the grader, the references (as
                                granted), the staging dir; NOTHING else of the machine's /Users (dataset mirror,
                                repo, runner caches, skills dirs of legs, other episodes' workspaces)
  <staging>                     fs/<staging>, read-write (evo-owned)
  /evo  (/evo/work, /evo/out)   fs/evo read-only; work and out read-write (evo-owned)
  /home/evo                     rw/home, read-write;  /tmp, /var/tmp, /dev/shm: private tmpfs
  /opt/claude-code/2.1.281/claude   the uploaded binary, read-only; nothing else under /opt
  /dev                          tmpfs with null zero full random urandom tty;  /run: tmpfs
  /proc                         the container's OWN pid namespace (/proc/sys read-only, kcore/keys/... masked)
  NO /sys, no /root, no /var/lib, no /opt/evoskill (grader tooling), no /workspace, no host /tmp.

PROCESSES: own pid/ipc/uts/net/mount namespaces; PID 1 is a reaper running as uid 65534 with no capabilities;
every command enters through `nsenter` + `setpriv` as uid 1000 (evo) with an empty bounding set, no ambient or
inheritable capabilities and no_new_privs, and a clean environment. The container's cgroup
(/sys/fs/cgroup/evoevolver/<cid>) caps memory, CPU and pids so the 120 rollouts are not starved.

NETWORK: the container's network namespace has only `lo`. This controller's `proxy` process creates its listening
socket INSIDE that namespace (setns) on 127.0.0.1:3128 and then returns to the machine's namespace, where it
tunnels CONNECT api.anthropic.com:443 only (scripts/evolver_egress_proxy.py's allowlist, request-line decision,
logging) -- directly or through the machine's own HTTPS proxy (Daytona's egress proxy) when one is set. DNS is the
proxy's; the container has no resolver, no route, no other listener.

THE TOKEN never touches this machine's filesystem: it reaches `enter` as fd 3 (a pipe from the SDK exec's
here-string), passes through nsenter into claude (CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR=3), and every other fd
is closed before nsenter. `down` scans the container's writable state (and its tmpfs mounts, before the kill) for
the token prefix and never prints a match's bytes.

Subcommands (every one prints ONE json line last):
  facts | prepare --spec-b64 B | up --cid C | proxy --cid C --init-pid P | enter --cid C [--cwd D] [--keep-fd3]
  [--log F] -- ARGV... | probe --cid C --phase pre|post | extract --cid C | tail --cid C --offset N --max M [--du]
  | stop-session --cid C | pack --cid C --part P... | scan --cid C | down --cid C [--wipe] | status --cid C | list
  STATEFUL SESSION (one container for the whole run):
  extract --cid C --incremental | epoch-begin --cid C [--clear P]... | epoch-end --cid C --part P... | restore --cid C
"""
from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tarfile
import threading
import time

SCHEMA = "evoskill-evolver-container/1"
BASE = "/opt/evoskill-evolver"
BIN = f"{BASE}/bin"
CTL = f"{BIN}/evoctl.py"
PROXY_FILE = f"{BIN}/evolver_egress_proxy.py"
PY = "/usr/local/bin/python3"                  # the image's python (3.11): runs this file on the machine
# The pinned Claude Code linux-x64 release (sha256 and size from downloads.claude.ai/claude-code-releases/<v>/manifest.json).
# Default 2.1.281. EVOSKILL_CC_VERSION selects another verified release (2.1.293, the newer one, for new runs; the solver
# and the evolver of a run use the same release).
CC_RELEASES = {
    "2.1.281": ("56fe3da88458465fb27d7e9299dddb3fead55750fb9c2de795f233b5eea6dce1", 237375560),
    "2.1.293": ("8968405e26db478af44eabc4635ab5ca557057b702a54460a59c13e1b253e978", 252755128),
}
# On the host the env var decides; on a machine (this file runs there without our env) the one pinned binary the host installed
# under {BASE}/cc/ decides (machines are fresh per run, so exactly one is present); otherwise the default.
_CC_ON_MACHINE = [v for v in CC_RELEASES if os.path.isfile(f"{BASE}/cc/{v}/claude-code.bin")]
CC_VERSION = (os.environ.get("EVOSKILL_CC_VERSION")
              or (_CC_ON_MACHINE[0] if len(_CC_ON_MACHINE) == 1 else "") or "2.1.281")
if CC_VERSION not in CC_RELEASES:
    raise SystemExit(f"EVOSKILL_CC_VERSION={CC_VERSION!r}: not a verified release ({sorted(CC_RELEASES)})")
CC_SHA256, CC_BYTES = CC_RELEASES[CC_VERSION]
# Extra flags a release needs so the solver surface matches 2.1.281's (placed right after `claude`): 2.1.293 ships a built-in
# plugin whose skill `plugin-authoring` survives CLAUDE_CODE_DISABLE_BUNDLED_SKILLS (measured with a skills probe);
# `/plugin disable <name>@builtin` is the settings key enabledPlugins.
CC_EXTRA_ARGS = {
    "2.1.281": (),
    "2.1.293": ("--settings", '{"enabledPlugins":{"cc-plugin-plugin-authoring@builtin":false}}'),
}[CC_VERSION]
CC_HOST = f"{BASE}/cc/{CC_VERSION}/claude-code.bin"
CC_DIR_CONTAINER = f"/opt/claude-code/{CC_VERSION}"
CC_CONTAINER = f"{CC_DIR_CONTAINER}/claude"
CG_ROOT = "/sys/fs/cgroup"
CG_PARENT = f"{CG_ROOT}/evoevolver"
EVO_UID = EVO_GID = 1000
REAPER_UID = REAPER_GID = 65534
EVO_HOME = "/home/evo"
VM_ROOT = "/evo"
VM_WORK = f"{VM_ROOT}/work"
VM_OUT = f"{VM_ROOT}/out"
VM_HOOKS = f"{VM_ROOT}/hooks"                  # never a rw path (the hook + settings the launch reads)
POLICY_DIR = "/etc/claude-code"
LISTING_ROOTS = ("/Users", VM_ROOT, POLICY_DIR)
UPLOAD_TOPS = ("Users/", "evo/", "etc/claude-code/")
PROXY_HOST, PROXY_PORT = "127.0.0.1", 3128
PROXY_URL = f"http://{PROXY_HOST}:{PROXY_PORT}"
ALLOW = "api.anthropic.com"
CONTAINER_PATH = f"{CC_DIR_CONTAINER}:/usr/local/bin:/usr/bin:/bin"
EVO_ENV = (f"HOME={EVO_HOME}", "USER=evo", "LOGNAME=evo", f"PATH={CONTAINER_PATH}", "LANG=C.UTF-8",
           "TMPDIR=/tmp", f"HTTPS_PROXY={PROXY_URL}", f"HTTP_PROXY={PROXY_URL}", f"https_proxy={PROXY_URL}",
           f"http_proxy={PROXY_URL}", "NO_PROXY=", "no_proxy=")
SYSDIR_NAMES = ("bin", "sbin", "lib", "lib32", "lib64", "libx32")
DEV_NODES = ("null", "zero", "full", "random", "urandom", "tty")
# In the machine's /usr but in neither the solver's nor the evolver's image: the leg bundle installs the docker
# shim as /usr/local/bin/docker (ssb_leg_daytona_loop Shard.upload). Masked by /dev/null (not executable: bash's
# PATH lookup skips it, so `docker` is "command not found", as on the evolver image).
MASKS = ("/usr/local/bin/docker",)
PROC_MASKS = ("/proc/kcore", "/proc/keys", "/proc/timer_list", "/proc/sysrq-trigger")
# /etc entries NOT copied into the container's /etc (secrets, host identity, and what this file writes itself).
ETC_EXCLUDE = frozenset({"shadow", "shadow-", "gshadow", "gshadow-", "passwd-", "group-", "subuid", "subuid-",
                         "subgid", "subgid-", "sudoers", "sudoers.d", "ssh", "machine-id", "hostname", "hosts",
                         "resolv.conf", "environment", "profile.d", "claude-code", ".pwd.lock", "crontab",
                         "cron.d", "cron.daily", "cron.hourly", "cron.weekly", "cron.monthly", "security"})
ETC_EXCLUDE_SUB = frozenset({"ssl/private"})
# The container's DNS: a resolver address with nothing behind it (every lookup fails at once; the proxy resolves).
RESOLV_CONF = "nameserver 127.0.0.1\noptions timeout:1 attempts:1\n"
HOSTS = "127.0.0.1\tlocalhost\n::1\tlocalhost ip6-localhost ip6-loopback\n"
HOSTNAME = "evolver"
# Spelled in pieces so that no scanner (this file's own, the pool's key scan, the driver's token scan) ever matches
# this source file itself.
TOKEN_PREFIX = b"sk-ant-" + b"oat01-"
OPENROUTER_PREFIX = b"sk-or-" + b"v1-"
SECRET_MARKS = (b"sk-" + b"ant-", OPENROUTER_PREFIX, b"OPENROUTER_API_KEY" + b"=", b"DAYTONA_API_KEY" + b"=",
                b"PRIVATE KEY" + b"-----")
SAFE_PATH = re.compile(r"^/[A-Za-z0-9._@+=,-]+(?:/[A-Za-z0-9._@+=,-]+)*$")
CID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{2,63}$")
LINK_RE = re.compile(r"^/?[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*$")
UNSHARE = ["unshare", "--mount", "--pid", "--net", "--ipc", "--uts", "--fork", "--kill-child"]
DROP = ("--inh-caps=-all", "--ambient-caps=-all", "--bounding-set=-all", "--no-new-privs")
# PID 1 of the container: reaps orphans (the shim's INIT, verbatim).
INIT = ("import os,signal,time\n"
        "signal.signal(signal.SIGTERM,lambda *a:os._exit(0))\n"
        "while True:\n"
        "    try: os.wait()\n"
        "    except ChildProcessError: time.sleep(0.5)\n")
# `lo` up inside the new network namespace (the shim's LO_UP, verbatim: no `ip` in the image).
LO_UP = ("/usr/local/bin/python3 -c 'import socket,fcntl,struct;s=socket.socket();"
         "ifr=struct.pack(\"16sh\",b\"lo\",0);fl=struct.unpack(\"16sh\",fcntl.ioctl(s,0x8913,ifr))[1];"
         "fcntl.ioctl(s,0x8914,struct.pack(\"16sh\",b\"lo\",fl|1))'")
SETHOSTNAME = f"/usr/local/bin/python3 -c 'import socket;socket.sethostname(\"{HOSTNAME}\")'"
# pivot_root(2) has no glibc wrapper and util-linux's binary is not guaranteed: the raw syscall.
PIVOT = ("/usr/local/bin/python3 -c 'import ctypes,os,sys;n={\"x86_64\":155,\"aarch64\":41}[os.uname().machine];"
         "l=ctypes.CDLL(None,use_errno=True);r=l.syscall(n,b\".\",b\".old_root\");e=ctypes.get_errno();"
         "sys.exit(0 if r==0 else \"pivot_root failed: errno %d\" % e)'")
DEFAULT_LIMITS = {"cpus": None, "mem_max_bytes": None, "mem_high_ratio": 0.85, "pids_max": 2048, "cpu_weight": 50,
                  "tmp_bytes": None, "rw_max_bytes": 5 << 30}
GIB = 1 << 30


def jprint(obj) -> None:
    sys.stdout.write(json.dumps(obj, default=str) + "\n")
    sys.stdout.flush()


def q(s) -> str:
    return shlex.quote(str(s))


def cdir(cid: str) -> str:
    if not CID_RE.match(cid or ""):
        raise ValueError(f"bad container id {cid!r}")
    return f"{BASE}/c/{cid}"


def _under(p: str, d: str) -> bool:
    d = d.rstrip("/")
    return p == d or p.startswith(d + "/")


# ───────────────────────────────────────────────────────────────────────────── spec / facts / limits
def validate_spec(spec: dict) -> list:
    """Problems with a spec from the driver (empty = fine). Paths: absolute, normalised, conservative charset."""
    probs = []
    if spec.get("schema") != SCHEMA:
        probs.append(f"schema {spec.get('schema')!r} != {SCHEMA!r}")
    if not CID_RE.match(str(spec.get("cid") or "")):
        probs.append(f"bad cid {spec.get('cid')!r}")
    rws = spec.get("rw") or []
    if not isinstance(rws, list):
        probs.append("rw must be a list")
        rws = []
    seen = []
    for r in rws:
        p = str((r or {}).get("path") or "")
        if not SAFE_PATH.match(p) or os.path.normpath(p) != p or "/../" in p + "/":
            probs.append(f"rw path {p!r} is not a plain absolute path")
            continue
        # strictly below /Users (the mirrored upload tree, per-epoch mode) OR strictly below /evo outside the
        # work / out / hooks trees (the stateful session's /evo/staging and /evo/evolver_workspace)
        below_users = _under(p, "/Users") and p != "/Users"
        below_evo = (_under(p, VM_ROOT) and p != VM_ROOT
                     and not any(_under(p, x) for x in (VM_WORK, VM_OUT, VM_HOOKS)))
        if not (below_users or below_evo):
            probs.append(f"rw path {p} is not strictly below /Users or /evo (outside work/out/hooks)")
        if (r.get("owner") not in ("evo", "root")) or (str(r.get("mode")) not in ("0755", "1777")):
            probs.append(f"rw {p}: owner/mode must be evo/0755 or root/1777")
        if any(_under(p, s) or _under(s, p) for s in seen):
            probs.append(f"rw path {p} nests with another rw path")
        seen.append(p)
    for p in spec.get("ro_rebind") or []:
        if not SAFE_PATH.match(str(p)) or os.path.normpath(str(p)) != p:
            probs.append(f"ro_rebind path {p!r} is not a plain absolute path")
        elif not any(_under(p, s) and p != s for s in seen):
            probs.append(f"ro_rebind path {p} is not below a rw path")
    for p in spec.get("readonly_roots") or []:
        if any(_under(s, p) for s in seen):
            probs.append(f"read-only root {p} contains a rw path")
        if any(_under(p, s) for s in seen) and p not in (spec.get("ro_rebind") or []):
            probs.append(f"read-only root {p} lies below a rw path but is not re-bound read-only")
    lim = spec.get("limits") or {}
    for k in lim:
        if k not in DEFAULT_LIMITS:
            probs.append(f"unknown limit {k!r}")
    cc = spec.get("cc") or {}
    if cc.get("sha256") != CC_SHA256 or cc.get("version") != CC_VERSION:
        probs.append("cc pin differs from this controller's")
    return probs


def host_facts() -> dict:
    """What the plan depends on, read from THIS machine (recorded in state/facts.json at prepare, reused by up)."""
    sysdirs = {}
    for n in SYSDIR_NAMES:
        p = "/" + n
        if os.path.islink(p):
            sysdirs[n] = "symlink:" + os.readlink(p)
        elif os.path.isdir(p):
            sysdirs[n] = "dir"
    mem_total = 0
    try:
        for ln in open("/proc/meminfo"):
            if ln.startswith("MemTotal:"):
                mem_total = int(ln.split()[1]) * 1024
    except OSError:
        pass
    cg_mem = _read(f"{CG_ROOT}/memory.max").strip()
    cg_cpu = _read(f"{CG_ROOT}/cpu.max").strip()     # e.g. "800000 100000" on an 8-vCPU Daytona sandbox
    return {"sysdirs": sysdirs, "masks_present": [m for m in MASKS if os.path.lexists(m)],
            "arch": os.uname().machine, "kernel": os.uname().release, "ncpu": os.cpu_count() or 1,
            "mem_total": mem_total, "cg_root_memory_max": cg_mem, "cg_root_cpu_max": cg_cpu,
            "userns_max": _read("/proc/sys/user/max_user_namespaces").strip(),
            "unprivileged_userns_clone": _read("/proc/sys/kernel/unprivileged_userns_clone").strip()}


def resolve_limits(facts: dict, want: dict | None) -> dict:
    """The container's cgroup limits. Defaults sized for the pool machine carrying 120 rollouts: at most a quarter
    of the vCPUs (1..4) as a hard cpu.max, cpu.weight 50 (a rollout episode's cgroup has 100, so under contention
    the rollouts get twice the evolver's share), memory.max = min(8 GiB, 20% of the machine) with memory.high at
    85% of it, swap off, 2048 pids, /tmp + /var/tmp + /dev/shm tmpfs within the memory cap."""
    lim = dict(DEFAULT_LIMITS)
    lim.update({k: v for k, v in (want or {}).items() if v is not None})
    total = int(facts.get("mem_total") or 0)
    cgm = str(facts.get("cg_root_memory_max") or "")
    if cgm.isdigit():
        total = min(total, int(cgm)) if total else int(cgm)
    ncpu = int(facts.get("ncpu") or 1)
    # Measured: os.cpu_count() is the HOST's (48 on Daytona), not the sandbox's quota; size from cpu.max.
    cq = str(facts.get("cg_root_cpu_max") or "").split()
    if len(cq) == 2 and cq[0].isdigit() and cq[1].isdigit() and int(cq[1]) > 0:
        ncpu = max(1, min(ncpu, int(cq[0]) // int(cq[1])))
    if lim["cpus"] is None:
        lim["cpus"] = float(max(1, min(4, ncpu // 4)))
    if lim["mem_max_bytes"] is None:
        lim["mem_max_bytes"] = int(min(8 * GIB, max(2 * GIB, total * 0.20))) if total else 4 * GIB
    if lim["tmp_bytes"] is None:
        lim["tmp_bytes"] = int(min(2 * GIB, lim["mem_max_bytes"] // 3))
    # Measured: the kernel rounds memory.max/high DOWN to a page (4096 B), so an unaligned value never
    # reads back equal; write page-aligned values.
    lim["mem_max_bytes"] = int(lim["mem_max_bytes"]) // 4096 * 4096
    lim["mem_high_bytes"] = int(lim["mem_max_bytes"] * float(lim["mem_high_ratio"])) // 4096 * 4096
    lim["cpu_max"] = f"{int(float(lim['cpus']) * 100000)} 100000"
    lim["machine_mem_bytes"], lim["machine_ncpu"] = total, ncpu
    return lim


# ───────────────────────────────────────────────────────────────────────────── the mount plan
def plan(spec: dict, facts: dict, limits: dict) -> list:
    """Ordered mounts of the container, container paths (dst) and host paths (src). Parents before children.
    kind: bind | tmpfs | proc | symlink | rebind_ro | mask (bind /dev/null, only if the path exists)
    mk:   root   = create the mount point (dir, or file for node=file) on a writable parent (the root tmpfs, /dev)
          exists = the mount point must already exist inside the parent mount's source (never created there)."""
    c = cdir(spec["cid"])
    P: list = []

    def add(kind, dst, src=None, ro=False, mk="root", node="dir", flags="nosuid,nodev", opts=None):
        P.append({"kind": kind, "dst": dst, "src": src, "ro": ro, "mk": mk, "node": node, "flags": flags,
                  "opts": opts})
    add("bind", "/usr", "/usr", ro=True)
    for n, what in sorted((facts.get("sysdirs") or {}).items()):
        if n not in SYSDIR_NAMES:
            raise ValueError(f"unexpected system dir {n!r}")
        if what.startswith("symlink:"):
            tgt = what[len("symlink:"):]
            if not LINK_RE.match(tgt) or ".." in tgt.split("/"):
                raise ValueError(f"/{n} -> {tgt!r}: not a plain link target")
            add("symlink", "/" + n, tgt, mk="root", node="link", flags=None)
        elif what == "dir":
            add("bind", "/" + n, "/" + n, ro=True)
    for m in facts.get("masks_present") or []:
        if m not in MASKS:
            raise ValueError(f"unexpected mask {m!r}")
        add("mask", m, "/dev/null", ro=True, mk="exists", node="file", flags="nosuid,nodev,noexec")
    add("bind", "/etc", f"{c}/etc", ro=True)
    add("bind", POLICY_DIR, f"{c}/fs{POLICY_DIR}", ro=True, mk="exists")
    add("bind", "/Users", f"{c}/fs/Users", ro=True)
    add("bind", VM_ROOT, f"{c}/fs{VM_ROOT}", ro=True)
    add("bind", VM_WORK, f"{c}/fs{VM_WORK}", mk="exists")
    add("bind", VM_OUT, f"{c}/fs{VM_OUT}", mk="exists")
    for r in sorted(spec.get("rw") or [], key=lambda r: (r["path"].count("/"), r["path"])):
        add("bind", r["path"], f"{c}/fs{r['path']}", mk="exists")
    for p in sorted(spec.get("ro_rebind") or [], key=lambda p: (p.count("/"), p)):
        add("bind", p, f"{c}/fs{p}", ro=True, mk="exists")
    add("bind", EVO_HOME, f"{c}/rw/home")
    add("tmpfs", "/tmp", "evotmp", opts=f"size={int(limits['tmp_bytes'])},mode=1777")
    add("tmpfs", "/var/tmp", "evovartmp", opts=f"size={int(min(limits['tmp_bytes'], GIB // 2))},mode=1777")
    add("bind", CC_CONTAINER, CC_HOST, ro=True, node="file")
    add("tmpfs", "/dev", "evodev", opts="size=1m,mode=0755", flags="nosuid,noexec")
    for n in DEV_NODES:
        add("bind", f"/dev/{n}", f"/dev/{n}", node="file", flags="nosuid,noexec")
    for name, tgt in (("fd", "/proc/self/fd"), ("stdin", "/proc/self/fd/0"), ("stdout", "/proc/self/fd/1"),
                      ("stderr", "/proc/self/fd/2")):
        add("symlink", f"/dev/{name}", tgt, node="link", flags=None)
    add("tmpfs", "/dev/shm", "evoshm", opts=f"size={int(min(limits['tmp_bytes'], GIB // 2))},mode=1777")
    add("tmpfs", "/run", "evorun", opts="size=1m,mode=0755", flags="nosuid,nodev,noexec")
    add("proc", "/proc", "proc", flags="nosuid,nodev,noexec")
    add("rebind_ro", "/proc/sys", None, ro=True, mk="exists", flags=None)
    for m in PROC_MASKS:
        add("mask", m, "/dev/null", ro=True, mk="exists", node="file", flags=None)
    for m in P:
        if not SAFE_PATH.match(m["dst"]) or os.path.normpath(m["dst"]) != m["dst"]:
            raise ValueError(f"mount point {m['dst']!r} is not a plain absolute path")
    return P


def expected_top(pl: list) -> list:
    """The entries of the container's / after the pivot: first component of every planned path (+ nothing else)."""
    return sorted({m["dst"].split("/")[1] for m in pl})


def expected_mount_points(pl: list) -> list:
    """[(mount point, ro)] the container's mountinfo must show beyond `/` (symlinks are not mounts; a PROC_MASKS
    mask shows only where the file exists, so masks are reported separately)."""
    out = []
    for m in pl:
        if m["kind"] in ("symlink",) or (m["kind"] == "mask" and m["dst"].startswith("/proc/")):
            continue
        out.append((m["dst"], bool(m["ro"])))
    return out


def host_path(pl: list, container_path: str, cid: str) -> str | None:
    """The HOST path behind a container path, through the bind that serves it (None for tmpfs/proc/dev paths).
    Callers must still refuse symlinks below the returned path (safe_host_path)."""
    cp = os.path.normpath(container_path)
    best = None
    for m in pl:
        if m["kind"] in ("bind",) and _under(cp, m["dst"]) and (best is None or len(m["dst"]) > len(best["dst"])):
            best = m
        elif m["kind"] in ("tmpfs", "proc", "mask", "rebind_ro") and _under(cp, m["dst"]) and \
                (best is None or len(m["dst"]) > len(best["dst"])):
            best = m
    if best is None or best["kind"] != "bind":
        return None
    rest = cp[len(best["dst"]):]
    src = best["src"]
    if not (src.startswith(cdir(cid) + "/")):
        return None                              # only the container's own state is ever read back
    return src + rest


def safe_host_path(pl: list, container_path: str, cid: str) -> tuple:
    """(host path or None, why). Every component below the serving mount's source must be a real directory on
    the HOST (lstat, no symlink): the evolver controls these trees, and a host-side read through a symlink it
    planted would resolve against the MACHINE's filesystem."""
    hp = host_path(pl, container_path, cid)
    if hp is None:
        return None, "not served by one of the container's own binds"
    cp = os.path.normpath(container_path)
    best = max((m for m in pl if m["kind"] == "bind" and _under(cp, m["dst"])), key=lambda m: len(m["dst"]))
    cur = best["src"]
    rel = [x for x in cp[len(best["dst"]):].split("/") if x]
    for part in rel:
        cur = os.path.join(cur, part)
        try:
            st = os.lstat(cur)
        except FileNotFoundError:
            return None, f"{cp}: absent"
        if stat.S_ISLNK(st.st_mode):
            return None, f"{cp}: a symlink on the way ({part})"
    return hp, ""


# ───────────────────────────────────────────────────────────────────────────── the generated init script
def _flags(m, ro: bool) -> str:
    fl = [x for x in ("ro" if ro else "rw", m.get("flags")) if x]
    return ",".join(fl)


def render_init(spec: dict, facts: dict, limits: dict, pl: list | None = None) -> str:
    """The script PID 1 of the new namespaces runs (bash -c): build the root, pivot, detach the old root, drop to
    the reaper. Every mount point below a read-only bind must already exist in its source (checked, never made)."""
    pl = plan(spec, facts, limits) if pl is None else pl
    c = cdir(spec["cid"])
    R = f"{c}/root"
    # `mount -n` everywhere: no mtab/utab is ever written (the machine's /run/mount before the pivot, the
    # container's /run after it).
    L = ["set -eu", f"R={q(R)}", "mount -n --make-rprivate /",
         'mount -n -t tmpfs -o size=64m,mode=0755,nosuid,nodev evoroot "$R"']
    # 1. mount points on the root tmpfs (all of them before anything is bound over their parents)
    for m in pl:
        if m["mk"] != "root" or m["dst"].startswith("/dev/"):
            continue
        d = m["dst"]
        if m["kind"] == "symlink":
            L.append(f'ln -s {q(m["src"])} "$R"{q(d)}')
        elif m["node"] == "file":
            L.append(f'mkdir -p "$R"{q(os.path.dirname(d))}; touch "$R"{q(d)}')
        else:
            L.append(f'mkdir -p "$R"{q(d)}')
    # 2. the mounts, in plan order
    for m in pl:
        d, k = m["dst"], m["kind"]
        if k == "symlink":
            if d.startswith("/dev/"):
                L.append(f'ln -s {q(m["src"])} "$R"{q(d)}')
            continue
        if m["mk"] == "root" and d.startswith("/dev/"):
            L.append(f'touch "$R"{q(d)}' if m["node"] == "file" else f'mkdir -p "$R"{q(d)}')
        if m["mk"] == "exists":
            test = "-f" if m["node"] == "file" and k != "mask" else ("-e" if k == "mask" else "-d")
            if k == "mask" and d.startswith("/proc/"):
                L.append(f'if [ -e "$R"{q(d)} ]; then mount -n --bind /dev/null "$R"{q(d)}; fi')
                continue
            L.append(f'[ {test} "$R"{q(d)} ] || {{ echo "EVOCTL_FAIL missing mount point {d}" >&2; exit 3; }}')
        if k in ("bind", "mask"):
            L.append(f'mount -n --bind {q(m["src"])} "$R"{q(d)}')
            if m["ro"] or m.get("flags"):
                L.append(f'mount -n -o remount,bind,{_flags(m, m["ro"])} "$R"{q(d)}')
        elif k == "tmpfs":
            L.append(f'mount -n -t tmpfs -o {m["opts"]}{"," + m["flags"] if m.get("flags") else ""} '
                     f'{q(m["src"])} "$R"{q(d)}')
        elif k == "proc":
            L.append(f'mount -n -t proc -o {m["flags"]} proc "$R"{q(d)}')
        elif k == "rebind_ro":
            L.append(f'mount -n --bind "$R"{q(d)} "$R"{q(d)}')
            L.append(f'mount -n -o remount,bind,ro "$R"{q(d)}')
        else:
            raise ValueError(f"unknown mount kind {k!r}")
    # 3. the network namespace's loopback, the hostname, then the pivot and the old root DETACHED
    L += [LO_UP, SETHOSTNAME,
          'cd "$R"', "mkdir .old_root", PIVOT, "cd /", "umount -n -l /.old_root", "rmdir /.old_root",
          # BIND remount (per-mount flags only). A plain remount reconfigures the tmpfs superblock with the options
          # mountinfo shows, and in a Daytona sandbox (user-namespaced) that fails: "tmpfs: Invalid uid".
          "mount -n -o remount,bind,ro,nosuid,nodev /",
          "echo EVOCTL_READY",
          "exec /usr/bin/setpriv " + " ".join(q(x) for x in (f"--reuid={REAPER_UID}", f"--regid={REAPER_GID}",
                                                            "--clear-groups", *DROP, "--"))
          + " /usr/bin/env -i PATH=/usr/bin:/bin /usr/local/bin/python3 -c " + q(INIT)
          + " </dev/null >/dev/null 2>&1"]
    return "\n".join(L)


def enter_argv(init_pid: int, argv: list, cwd: str = VM_WORK) -> list:
    """nsenter into ALL of the container's namespaces, then setpriv down to evo (no capabilities anywhere,
    no_new_privs) with a clean environment. The last bash `cd`s and execs ARGV."""
    if not SAFE_PATH.match(cwd) and cwd != "/":
        raise ValueError(f"bad cwd {cwd!r}")
    return (["nsenter", f"--target={int(init_pid)}", "--mount", "--pid", "--net", "--ipc", "--uts", "--root",
             "--wd", "--",
             "/usr/bin/setpriv", f"--reuid={EVO_UID}", f"--regid={EVO_GID}", "--clear-groups", *DROP, "--",
             "/usr/bin/env", "-i", *EVO_ENV,
             "/bin/bash", "-c", 'cd "$1" || exit 97; shift; exec "$@"', "evoenter", cwd] + list(argv))


# ───────────────────────────────────────────────────────────────────────────── small helpers
def _read(p: str) -> str:
    try:
        with open(p) as fh:
            return fh.read()
    except OSError:
        return ""


def _write(p: str, s: str) -> None:
    with open(p, "w") as fh:
        fh.write(s)


def _jload(p: str) -> dict:
    try:
        with open(p) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _jsave(p: str, obj) -> None:
    tmp = f"{p}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    os.replace(tmp, p)


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def pid_start(pid: int) -> int | None:
    st = _read(f"/proc/{pid}/stat")
    try:
        return int(st.rsplit(")", 1)[1].split()[19])
    except (IndexError, ValueError):
        return None


def alive(pid, start=None) -> bool:
    if not pid:
        return False
    st = _read(f"/proc/{int(pid)}/stat")
    if not st or st.rsplit(")", 1)[-1].split()[:1] in (["Z"], ["X"]):
        return False
    return start is None or pid_start(int(pid)) == start


def child_of(ppid: int) -> int | None:
    for d in os.listdir("/proc"):
        if d.isdigit():
            rest = _read(f"/proc/{d}/stat").rsplit(")", 1)[-1].split()
            if len(rest) > 1 and rest[1] == str(ppid):
                return int(d)
    return None


def ns_ids(pid) -> dict:
    out = {}
    for n in ("mnt", "pid", "net", "ipc", "uts", "user", "cgroup"):
        try:
            out[n] = os.readlink(f"/proc/{pid}/ns/{n}")
        except OSError:
            out[n] = None
    return out


def paths(cid: str) -> dict:
    c = cdir(cid)
    return {"c": c, "root": f"{c}/root", "etc": f"{c}/etc", "fs": f"{c}/fs", "rw": f"{c}/rw",
            "home": f"{c}/rw/home", "state": f"{c}/state", "cg": f"{CG_PARENT}/{cid}"}


def load_all(cid: str) -> tuple:
    P = paths(cid)
    spec, facts, limits = (_jload(f"{P['state']}/{n}.json") for n in ("spec", "facts", "limits"))
    if not spec:
        raise SystemExit(f"no prepared container {cid}")
    return P, spec, facts, limits, plan(spec, facts, limits), _jload(f"{P['state']}/state.json")


def scan_tree(root: str, marks=(TOKEN_PREFIX,), cap: int = 64 << 20) -> tuple:
    """(hits [(relpath, mark index)], files scanned). No symlink is followed; only regular files are opened
    (O_NOFOLLOW); a match's bytes are never returned."""
    hits, n = [], 0
    if not os.path.isdir(root) or os.path.islink(root):
        return hits, n
    for dp, dns, fns in os.walk(root, followlinks=False):
        for f in fns:
            p = os.path.join(dp, f)
            try:
                st = os.lstat(p)
                if not stat.S_ISREG(st.st_mode):
                    continue
                fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0))
            except OSError:
                continue
            try:
                n += 1
                tail, left = b"", cap
                while left > 0:
                    b = os.read(fd, min(1 << 20, left))
                    if not b:
                        break
                    left -= len(b)
                    win = tail + b
                    for i, mk in enumerate(marks):
                        if mk in win:
                            hits.append((os.path.relpath(p, root), i))
                    tail = win[-64:]
            finally:
                os.close(fd)
    return sorted(set(hits)), n


# ───────────────────────────────────────────────────────────────────────────── /etc
def build_etc(dst: str, src: str = "/etc") -> dict:
    """Copy the machine's /etc (the image's own) into dst: regular files, dirs and symlinks only, minus
    ETC_EXCLUDE / ETC_EXCLUDE_SUB; then write passwd/group (+ evo), hosts, resolv.conf, hostname and the
    claude-code mount point. Refuses (raises) if any copied file carries a secret-shaped string."""
    skipped, n = [], 0
    for dp, dns, fns in os.walk(src, followlinks=False):
        rel = os.path.relpath(dp, src)
        keep = []
        for d in dns:
            r = d if rel == "." else f"{rel}/{d}"
            if (rel == "." and d in ETC_EXCLUDE) or r in ETC_EXCLUDE_SUB:
                skipped.append(r)
            else:
                keep.append(d)
        dns[:] = keep
        tdir = dst if rel == "." else os.path.join(dst, rel)
        os.makedirs(tdir, exist_ok=True)
        st = os.lstat(dp)
        os.chmod(tdir, stat.S_IMODE(st.st_mode))
        for f in fns + [d for d in keep if os.path.islink(os.path.join(dp, d))]:
            r = f if rel == "." else f"{rel}/{f}"
            if rel == "." and f in ETC_EXCLUDE:
                skipped.append(r)
                continue
            s = os.path.join(dp, f)
            t = os.path.join(tdir, f)
            ls = os.lstat(s)
            if stat.S_ISLNK(ls.st_mode):
                if not os.path.lexists(t):
                    os.symlink(os.readlink(s), t)
            elif stat.S_ISREG(ls.st_mode):
                shutil.copy2(s, t, follow_symlinks=False)
                n += 1
            else:
                skipped.append(r)
    pw = [ln for ln in _read(f"{src}/passwd").splitlines()
          if ln and ln.split(":")[0] != "evo" and ln.split(":")[2:3] != [str(EVO_UID)]]
    gr = [ln for ln in _read(f"{src}/group").splitlines()
          if ln and ln.split(":")[0] != "evo" and ln.split(":")[2:3] != [str(EVO_GID)]]
    _write(f"{dst}/passwd", "\n".join(pw + [f"evo:x:{EVO_UID}:{EVO_GID}::{EVO_HOME}:/bin/bash"]) + "\n")
    _write(f"{dst}/group", "\n".join(gr + [f"evo:x:{EVO_GID}:"]) + "\n")
    _write(f"{dst}/hosts", HOSTS)
    _write(f"{dst}/resolv.conf", RESOLV_CONF)
    _write(f"{dst}/hostname", HOSTNAME + "\n")
    for f in ("passwd", "group", "hosts", "resolv.conf", "hostname"):
        os.chmod(f"{dst}/{f}", 0o644)
    os.makedirs(f"{dst}/claude-code", exist_ok=True)
    os.chmod(f"{dst}/claude-code", 0o755)
    hits, scanned = scan_tree(dst, SECRET_MARKS)
    if hits:
        raise RuntimeError(f"secret-shaped strings in the copied /etc: {[h[0] for h in hits][:5]}")
    listing = []
    for dp, dns, fns in os.walk(dst, followlinks=False):
        for f in sorted(dns + fns):
            listing.append(os.path.relpath(os.path.join(dp, f), dst))
    return {"files_copied": n, "skipped": sorted(skipped), "entries": len(listing),
            "entries_sha256": hashlib.sha256("\n".join(sorted(listing)).encode()).hexdigest(),
            "secret_scan_files": scanned}


# ───────────────────────────────────────────────────────────────────────────── cgroup
def cg_ensure(cid: str, limits: dict, init_root: bool = False) -> dict:
    need = {"memory", "cpu", "pids"}
    have = set(_read(f"{CG_ROOT}/cgroup.subtree_control").split())
    if not need <= have:
        try:
            _write(f"{CG_ROOT}/cgroup.subtree_control", "+memory +cpu +pids")
        except OSError:
            if not init_root:
                raise RuntimeError("root cgroup does not delegate memory/cpu/pids (the pool machine's "
                                   "cgroups_and_selftest does); refusing to move the machine's processes")
            os.makedirs(f"{CG_ROOT}/sandbox_init", exist_ok=True)
            for p in _read(f"{CG_ROOT}/cgroup.procs").split():
                try:
                    _write(f"{CG_ROOT}/sandbox_init/cgroup.procs", p)
                except OSError:
                    pass
            _write(f"{CG_ROOT}/cgroup.subtree_control", "+memory +cpu +pids")
    os.makedirs(CG_PARENT, exist_ok=True)
    if not need <= set(_read(f"{CG_PARENT}/cgroup.subtree_control").split()):
        _write(f"{CG_PARENT}/cgroup.subtree_control", "+memory +cpu +pids")
    cg = f"{CG_PARENT}/{cid}"
    os.makedirs(cg, exist_ok=True)
    _write(f"{cg}/memory.max", str(int(limits["mem_max_bytes"])))
    _write(f"{cg}/memory.high", str(int(limits["mem_high_bytes"])))
    if os.path.exists(f"{cg}/memory.swap.max"):
        _write(f"{cg}/memory.swap.max", "0")
    _write(f"{cg}/pids.max", str(int(limits["pids_max"])))
    _write(f"{cg}/cpu.max", limits["cpu_max"])
    _write(f"{cg}/cpu.weight", str(int(limits["cpu_weight"])))
    rb = {k: _read(f"{cg}/{k}").strip() for k in ("memory.max", "memory.high", "memory.swap.max", "pids.max",
                                                  "cpu.max", "cpu.weight")}
    if rb["memory.max"] != str(int(limits["mem_max_bytes"])) or rb["cpu.max"] != limits["cpu_max"]:
        raise RuntimeError(f"cgroup limits did not read back: {rb}")
    return {"path": cg, "readback": rb}


# ───────────────────────────────────────────────────────────────────────────── the egress proxy
def _import_eep():
    sys.path.insert(0, BIN if os.path.isfile(PROXY_FILE) else
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ""))
    import evolver_egress_proxy as eep   # noqa: E402  (scripts/evolver_egress_proxy.py)
    return eep


def upstream_from_env(env=None) -> dict | None:
    """The machine's own HTTPS proxy (Daytona's egress proxy), if the environment names one: {host, port, auth}.
    Only http:// proxies are chained; credentials are kept for the Proxy-Authorization header, never logged."""
    from urllib.parse import unquote, urlsplit
    env = os.environ if env is None else env
    raw = env.get("HTTPS_PROXY") or env.get("https_proxy") or ""
    if not raw:
        return None
    u = urlsplit(raw if "://" in raw else "http://" + raw)
    if u.scheme != "http" or not u.hostname:
        raise ValueError(f"unsupported upstream proxy scheme {u.scheme!r} (http only)")
    auth = None
    if u.username:
        auth = base64.b64encode(f"{unquote(u.username)}:{unquote(u.password or '')}".encode()).decode()
    return {"host": u.hostname, "port": u.port or 80, "auth": auth}


def make_chain_proxy(eep):
    class ChainProxy(eep.Proxy):
        """scripts/evolver_egress_proxy.Proxy with ONE change: the upstream TCP connect goes through the machine's
        own HTTP proxy when there is one (CONNECT host:443 there, 200 required). The allowlist decision is the
        parent's, made on the request line before any lookup or connect."""
        upstream = None

        def connect_upstream(self, host: str, port: int):
            if not self.upstream:
                return socket.create_connection((host, port), timeout=eep.CONNECT_TIMEOUT_S)
            up = self.upstream
            s = socket.create_connection((up["host"], up["port"]), timeout=eep.CONNECT_TIMEOUT_S)
            try:
                hdr = f"CONNECT {host}:{port} HTTP/1.1\r\nHost: {host}:{port}\r\n"
                if up.get("auth"):
                    hdr += f"Proxy-Authorization: Basic {up['auth']}\r\n"
                s.sendall((hdr + "\r\n").encode())
                buf, deadline = b"", time.monotonic() + eep.CONNECT_TIMEOUT_S
                while b"\r\n\r\n" not in buf:
                    if len(buf) > eep.MAX_HEADER or time.monotonic() > deadline:
                        raise OSError("upstream proxy: no complete response head")
                    s.settimeout(max(0.1, deadline - time.monotonic()))
                    b = s.recv(4096)
                    if not b:
                        raise OSError("upstream proxy closed before answering")
                    buf += b
                head, _, extra = buf.partition(b"\r\n\r\n")
                first = head.split(b"\r\n", 1)[0]
                parts = first.split()
                if len(parts) < 2 or parts[1] != b"200":
                    shown = first[:80].decode("latin-1")
                    raise OSError(f"upstream proxy refused CONNECT {host}:{port}: {shown}")
                if extra:
                    raise OSError("upstream proxy sent bytes after its CONNECT answer")
                s.settimeout(None)
                return s
            except Exception:
                s.close()
                raise

        def _handle(self, c, peer) -> None:        # the parent's _handle with connect_upstream() as the connect
            t0 = time.time()
            method = target = host = ""
            port = None
            try:
                head, extra = self._read_head(c)
                if head is None:
                    self.log(method="", target="", host="", port=None, allowed=False, reason=extra, bytes_up=0,
                             bytes_down=0, duration_s=round(time.time() - t0, 3), client_port=peer[1])
                    return
                line = head.split(b"\r\n", 1)[0].decode("latin-1", "replace")
                parts = line.split(" ")
                method = parts[0] if parts else ""
                target = parts[1] if len(parts) > 1 else ""
                why = ""
                if len(parts) != 3 or not parts[2].startswith("HTTP/1."):
                    code, why = 400, "malformed request line"
                elif method != "CONNECT":
                    code, why = 403, "only CONNECT to an allowlisted host:443 is permitted"
                else:
                    hp = eep.parse_connect_target(target)
                    if hp is None:
                        code, why = 400, "malformed CONNECT target"
                    else:
                        host, port = hp
                        if not eep.host_allowed(host, self.exact, self.suffixes):
                            code, why = 403, "host not on the allowlist"
                        elif port != self.allowed_port:
                            code, why = 403, f"port is not {self.allowed_port}"
                if why:
                    self._refuse(c, code, why)
                    self.log(method=method[:32], target=target[:256], host=eep._norm_host(host)[:256] if host
                             else "", port=port, allowed=False, reason=why, bytes_up=0, bytes_down=0,
                             duration_s=round(time.time() - t0, 3), client_port=peer[1])
                    return
                try:
                    u = self.connect_upstream(eep._norm_host(host), port)
                except OSError as e:
                    self._refuse(c, 502, f"upstream connect failed: {e}")
                    self.log(method=method, target=target, host=eep._norm_host(host), port=port, allowed=True,
                             reason=f"upstream connect failed: {e}", bytes_up=0, bytes_down=0,
                             duration_s=round(time.time() - t0, 3), client_port=peer[1])
                    return
                rec = {"method": method, "target": target, "host": eep._norm_host(host), "port": port,
                       "upstream_ip": "", "t0": t0, "client_port": peer[1], "up": 0, "down": 0, "logged": False}
                with self._active_lock:
                    self._active[id(rec)] = rec
                try:
                    rec["upstream_ip"] = u.getpeername()[0]
                    c.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                    if extra:
                        u.sendall(extra)
                        rec["up"] += len(extra)
                    self._tunnel(c, u, rec)
                except Exception as e:                  # noqa: BLE001
                    self._finish(rec, f"proxy error in tunnel: {type(e).__name__}: {e}")
                finally:
                    try:
                        u.close()
                    except OSError:
                        pass
                    self._finish(rec, "")
            except Exception as e:                      # noqa: BLE001
                self.log(method=method[:32], target=target[:256], host=str(host)[:256], port=port, allowed=False,
                         reason=f"proxy error: {type(e).__name__}: {e}", bytes_up=0, bytes_down=0,
                         duration_s=round(time.time() - t0, 3), client_port=peer[1] if peer else None)
            finally:
                try:
                    c.close()
                except OSError:
                    pass
    return ChainProxy


CLONE_NEWNET = 0x40000000


def _setns(fd: int, nstype: int) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.setns(fd, nstype) != 0:
        e = ctypes.get_errno()
        raise OSError(e, f"setns: {os.strerror(e)}")


def cmd_proxy(a) -> int:
    """The container's ONLY way out: listen in the container's network namespace, tunnel from the machine's."""
    P, spec, facts, limits, pl, _st = load_all(a.cid)
    eep = _import_eep()
    ChainProxy = make_chain_proxy(eep)
    init_pid, init_start = int(a.init_pid), pid_start(int(a.init_pid))
    own_id = os.readlink("/proc/thread-self/ns/net")
    own = os.open("/proc/thread-self/ns/net", os.O_RDONLY)
    tgt = os.open(f"/proc/{init_pid}/ns/net", os.O_RDONLY)
    tgt_id = os.readlink(f"/proc/{init_pid}/ns/net")
    if tgt_id == own_id:
        raise SystemExit("the container shares the machine's network namespace: refusing")
    _setns(tgt, CLONE_NEWNET)
    try:
        px = ChainProxy(spec.get("egress", {}).get("allow") or ALLOW, f"{P['state']}/egress.jsonl",
                        float(spec.get("egress", {}).get("idle_timeout_s") or 1800), PROXY_PORT, 443)
    finally:
        _setns(own, CLONE_NEWNET)
    back = os.readlink("/proc/thread-self/ns/net")
    if back != own_id:
        raise SystemExit("could not return to the machine's network namespace")
    up = upstream_from_env()
    px.upstream = up
    max_s = float(spec.get("egress", {}).get("max_seconds") or 6 * 3600)
    done = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: done.set())

    def watch():
        t_end = time.time() + max_s
        while not done.is_set():
            if not alive(init_pid, init_start) or time.time() > t_end:
                done.set()
                return
            done.wait(2.0)
    threading.Thread(target=watch, daemon=True).start()
    threading.Thread(target=px.serve_forever, daemon=True).start()
    _jsave(f"{P['state']}/proxy.ready", {
        "pid": os.getpid(), "listen": f"{PROXY_HOST}:{PROXY_PORT}", "listen_netns": tgt_id,
        "machine_netns": own_id, "allow": sorted(px.exact) + list(px.suffixes),
        "upstream": ({"kind": "http-proxy", "host": up["host"], "port": up["port"], "auth": bool(up.get("auth"))}
                     if up else {"kind": "direct"}), "max_tunnels": px.max_tunnels})
    while not done.wait(1.0):
        pass
    px.stop()
    px.log_open_tunnels("container gone or proxy stopped")
    os._exit(0)


# ───────────────────────────────────────────────────────────────────────────── prepare / up / enter
def cmd_facts(a) -> int:
    jprint(host_facts())
    return 0


def cmd_prepare(a) -> int:
    spec = json.loads(base64.b64decode(a.spec_b64).decode())
    probs = validate_spec(spec)
    if probs:
        jprint({"ok": False, "problems": probs})
        return 2
    P = paths(spec["cid"])
    if os.path.exists(P["c"]):
        jprint({"ok": False, "problems": [f"{P['c']} exists (a container id is used once)"]})
        return 2
    if not (os.path.isfile(CC_HOST) and os.path.getsize(CC_HOST) == CC_BYTES and sha256_file(CC_HOST) == CC_SHA256):
        jprint({"ok": False, "problems": [f"{CC_HOST} missing or not the pinned Claude Code {CC_VERSION}"]})
        return 2
    os.makedirs(BASE, exist_ok=True)
    os.chmod(BASE, 0o700)
    os.makedirs(f"{BASE}/c", exist_ok=True)
    os.chmod(f"{BASE}/c", 0o700)
    os.mkdir(P["c"], 0o700)
    for d, mode in ((P["root"], 0o755), (P["state"], 0o700), (P["fs"], 0o755), (f"{P['fs']}/Users", 0o755),
                    (f"{P['fs']}{VM_ROOT}", 0o755), (f"{P['fs']}{POLICY_DIR}", 0o755), (P["rw"], 0o755)):
        os.makedirs(d, exist_ok=True)
        os.chmod(d, mode)
    for d in (VM_WORK, VM_OUT):
        os.makedirs(f"{P['fs']}{d}", exist_ok=True)
        os.chmod(f"{P['fs']}{d}", 0o755)
        os.chown(f"{P['fs']}{d}", EVO_UID, EVO_GID)
    for r in spec.get("rw") or []:
        hp = f"{P['fs']}{r['path']}"
        os.makedirs(hp, exist_ok=True)
        os.chmod(hp, int(str(r["mode"]), 8))
        os.chown(hp, *((EVO_UID, EVO_GID) if r["owner"] == "evo" else (0, 0)))
    for p in spec.get("ro_rebind") or []:
        os.makedirs(f"{P['fs']}{p}", exist_ok=True)
    os.makedirs(P["home"], exist_ok=True)
    lo = "/root/.config/libreoffice"
    if os.path.isdir(lo):
        shutil.copytree(lo, f"{P['home']}/.config/libreoffice", symlinks=True)
    for dp, dns, fns in os.walk(P["home"], followlinks=False):
        for n in [dp] + [os.path.join(dp, x) for x in dns + fns]:
            os.lchown(n, EVO_UID, EVO_GID)
    os.chmod(P["home"], 0o755)
    try:
        etc = build_etc(P["etc"])
    except RuntimeError as e:
        shutil.rmtree(P["c"], ignore_errors=True)
        jprint({"ok": False, "problems": [str(e)]})
        return 2
    facts = host_facts()
    limits = resolve_limits(facts, spec.get("limits"))
    pl = plan(spec, facts, limits)
    for n, obj in (("spec", spec), ("facts", facts), ("limits", limits), ("plan", pl)):
        _jsave(f"{P['state']}/{n}.json", obj)
    _write(f"{P['state']}/init.sh", render_init(spec, facts, limits, pl))
    jprint({"ok": True, "cdir": P["c"], "facts": facts, "limits": limits, "etc": etc, "plan": pl,
            "expected_top": expected_top(pl), "cc_host_sha256": CC_SHA256})
    return 0


def cmd_up(a) -> int:
    P, spec, facts, limits, pl, st = load_all(a.cid)
    if st.get("init_pid") and alive(st["init_pid"], st.get("init_start")):
        jprint({"ok": False, "problems": ["container already up"]})
        return 2
    try:
        cg = cg_ensure(a.cid, limits, init_root=bool(a.init_root_cgroup))
    except (OSError, RuntimeError) as e:
        jprint({"ok": False, "problems": [f"cgroup: {e}"]})
        return 2
    script = _read(f"{P['state']}/init.sh")
    logf = open(f"{P['state']}/init.log", "ab")

    def pre():
        os.setsid()
        with open(f"{cg['path']}/cgroup.procs", "w") as fh:
            fh.write(str(os.getpid()))
    holder = subprocess.Popen(UNSHARE + ["/bin/bash", "-c", script], stdin=subprocess.DEVNULL, stdout=logf,
                              stderr=logf, preexec_fn=pre, close_fds=True)
    t0, ready = time.time(), False
    while time.time() - t0 < 90:
        if "EVOCTL_READY" in _read(f"{P['state']}/init.log"):
            ready = True
            break
        if holder.poll() is not None:
            break
        time.sleep(0.1)
    init_pid = child_of(holder.pid) if ready else None
    ok_uid = False
    for _ in range(100):
        if init_pid and "\nUid:\t%d\t" % REAPER_UID in "\n" + _read(f"/proc/{init_pid}/status"):
            ok_uid = True
            break
        time.sleep(0.1)
    if not (ready and init_pid and ok_uid):
        tail = _read(f"{P['state']}/init.log")[-1500:]
        try:
            _write(f"{cg['path']}/cgroup.kill", "1")
        except OSError:
            holder.kill()
        jprint({"ok": False, "problems": [f"container did not come up (ready={ready}, init={init_pid}, "
                                          f"reaper_uid={ok_uid})"], "init_log_tail": tail})
        return 3
    cns, hns = ns_ids(init_pid), ns_ids("self")
    shared = [n for n in ("mnt", "pid", "net", "ipc", "uts") if cns[n] == hns[n]]
    state = {"cid": a.cid, "holder_pid": holder.pid, "holder_start": pid_start(holder.pid), "init_pid": init_pid,
             "init_start": pid_start(init_pid), "ns": cns, "machine_ns": hns, "cgroup": cg, "t_up": time.time()}
    _jsave(f"{P['state']}/state.json", state)
    if shared:
        jprint({"ok": False, "problems": [f"namespaces shared with the machine: {shared}"], "state": state})
        return 3
    ready_p = _start_proxy(P, a.cid, init_pid, state)
    if not ready_p or ready_p.get("listen_netns") != cns["net"]:
        jprint({"ok": False, "problems": ["egress proxy not listening in the container's network namespace"],
                "proxy_out_tail": _read(f"{P['state']}/proxy.out")[-800:], "state": state})
        return 3
    jprint({"ok": True, "state": state})
    return 0


def _start_proxy(P: dict, cid: str, init_pid: int, state: dict) -> dict:
    """Start the egress proxy for the container whose PID 1 is init_pid, wait for its ready record (proxy.ready)
    and record pid/start/ready in state.json. Returns the ready record ({} when it never came up); the CALLER
    judges `listen_netns` against the container's network namespace. Extracted from `up` (byte-identical
    behaviour there); `epoch-begin` reuses it to restart a proxy that died between epochs (stateful session)."""
    pout = open(f"{P['state']}/proxy.out", "ab")
    prx = subprocess.Popen([sys.executable, os.path.abspath(__file__), "proxy", "--cid", cid, "--init-pid",
                            str(init_pid)], stdin=subprocess.DEVNULL, stdout=pout, stderr=pout, close_fds=True,
                           start_new_session=True)
    ready_p = None
    for _ in range(200):
        ready_p = _jload(f"{P['state']}/proxy.ready")
        if ready_p or prx.poll() is not None:
            break
        time.sleep(0.1)
    state.update(proxy_pid=prx.pid, proxy_start=pid_start(prx.pid), proxy=ready_p)
    _jsave(f"{P['state']}/state.json", state)
    return ready_p


def _close_fds(keep: int) -> None:
    try:
        import resource
        hi = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    except Exception:                                   # noqa: BLE001
        hi = 65536
    os.closerange(keep + 1, min(hi, 1 << 20) if hi > 0 else 65536)


def cmd_enter(a) -> int:
    """exec nsenter into the container as evo. fds: 0,1,2 (and 3 with --keep-fd3: the token pipe) survive, every
    other descriptor is closed first -- nothing of the SDK exec's plumbing reaches the container."""
    P, spec, facts, limits, pl, st = load_all(a.cid)
    if not (st.get("init_pid") and alive(st["init_pid"], st.get("init_start"))):
        sys.stderr.write(f"evoctl: container {a.cid} is not up\n")
        return 4
    argv = list(a.argv)
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if not argv:
        sys.stderr.write("evoctl enter: no command\n")
        return 2
    if a.log:
        fd = os.open(a.log, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        os.dup2(fd, 2)
        os.close(fd)
    with open(f"{st['cgroup']['path']}/cgroup.procs", "w") as fh:
        fh.write(str(os.getpid()))
    _close_fds(3 if a.keep_fd3 else 2)
    os.execvp("nsenter", enter_argv(st["init_pid"], argv, a.cwd))
    return 127


def _enter(cid: str, argv: list, *, cwd: str = "/tmp", inp: bytes | None = None, timeout: float = 900) -> tuple:
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "enter", "--cid", cid, "--cwd", cwd, "--"]
                       + list(argv), input=inp, capture_output=True, timeout=timeout)
    return r.returncode, r.stdout.decode("utf-8", "replace"), r.stderr.decode("utf-8", "replace")


# ───────────────────────────────────────────────────────────────────────────── probes
PROBE_PY = r'''
import ctypes, errno, hashlib, json, os, pwd, re, shutil, socket, subprocess, sys, urllib.error, urllib.request
inp = json.loads(sys.stdin.read() or "{}")
def rd(p, n=None):
    try:
        with open(p, "rb") as f:
            return f.read() if n is None else f.read(n)
    except OSError:
        return None
def run(argv, timeout=60, env=None):
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
        return r.returncode, (r.stdout + r.stderr)
    except Exception as e:
        return None, "%s: %s" % (type(e).__name__, str(e)[:200])
def ename(e):
    return errno.errorcode.get(getattr(e, "errno", None) or 0, type(e).__name__)
out = {"phase": inp.get("phase")}
st = (rd("/proc/self/status") or b"").decode()
kv = dict((l.split(":", 1)[0], l.split(":", 1)[1].strip()) for l in st.splitlines() if ":" in l)
I = {"uid": os.getuid(), "euid": os.geteuid(), "gid": os.getgid(), "groups": sorted(os.getgroups()),
     "caps": dict((k, kv.get(k)) for k in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")),
     "no_new_privs": kv.get("NoNewPrivs"), "seccomp": kv.get("Seccomp"), "arch": os.uname().machine,
     "hostname": socket.gethostname(), "user": pwd.getpwuid(os.getuid()).pw_name}
lo = rd("/etc/evoskill_libreoffice_version")
I["libreoffice"] = lo.decode().splitlines()[0].strip() if lo else None
h = hashlib.sha256()
try:
    with open(inp["cc_container"], "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    I["cc_sha256"] = h.hexdigest()
except OSError as e:
    I["cc_sha256"] = "unreadable: " + ename(e)
I["claude_on_path"] = shutil.which("claude")
I["docker_on_path"] = shutil.which("docker")
rc, o = run(["claude", "--version"], 90, env=dict(os.environ, HOME="/tmp/ccv", DISABLE_AUTOUPDATER="1",
                                                    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1", IS_SANDBOX="1"))
I["cc_version"] = (o.strip().splitlines() or [""])[0]
rc, o = run(["/usr/local/bin/python3", "-W", "ignore", "-c", "import openpyxl,pandas,numpy,sys;"
             "print(openpyxl.__version__,pandas.__version__,numpy.__version__,sys.version.split()[0])"], 180)
I["solver_stack"] = (o.strip().splitlines() or [""])[-1]
lo_dir = "/tmp/evoprobe_lo"
os.makedirs(lo_dir, exist_ok=True)
with open(lo_dir + "/x.csv", "w") as f:
    f.write("a,b\n1,=A2*2\n")
rc, o = run(["soffice", "--headless", "--calc", "--convert-to", "xlsx", "--outdir", lo_dir, lo_dir + "/x.csv"], 240)
I["soffice_convert_ok"] = os.path.isfile(lo_dir + "/x.xlsx")
I["soffice_convert_tail"] = o[-300:]
shutil.rmtree(lo_dir, ignore_errors=True)
I["opt_entries"] = sorted(os.listdir("/opt")) if os.path.isdir("/opt") else None
I["users_entries"] = len(os.listdir("/Users")) if os.path.isdir("/Users") else None
uf = 0
for dp, dns, fns in os.walk("/Users", followlinks=False):
    uf += len(fns) + sum(1 for d in dns if os.path.islink(os.path.join(dp, d)))
I["users_files"] = uf
I["secret_env_names"] = sorted(k for k in os.environ if re.search("(?i)api_key|token|secret|password", k))
I["proxy_env"] = dict((k, os.environ[k]) for k in sorted(os.environ) if "proxy" in k.lower())
I["env_names"] = sorted(os.environ)
try:
    I["evo_lo_profile_owner"] = pwd.getpwuid(os.stat(os.path.expanduser("~/.config/libreoffice")).st_uid).pw_name
except OSError as e:
    I["evo_lo_profile_owner"] = ename(e)
I["policy_dir_entries"] = len(os.listdir("/etc/claude-code")) if os.path.isdir("/etc/claude-code") else "absent"
named = []
for dp, dns, fns in os.walk("/", followlinks=False):
    if dp in ("/proc", "/dev", "/sys"):
        dns[:] = []
        continue
    named += [os.path.join(dp, f) for f in fns if f == "claude"]
I["files_named_claude"] = sorted(named)[:10]
out["identity"] = I
V = {"top": sorted(os.listdir("/")), "home": sorted(os.listdir("/home")) if os.path.isdir("/home") else None,
     "var": sorted(os.listdir("/var")) if os.path.isdir("/var") else None, "run": sorted(os.listdir("/run")),
     "dev": sorted(os.listdir("/dev")), "opt": sorted(os.listdir("/opt")) if os.path.isdir("/opt") else None}
mounts = []
for ln in (rd("/proc/self/mountinfo") or b"").decode().splitlines():
    f = ln.split(" ")
    try:
        sep = f.index("-")
    except ValueError:
        continue
    mounts.append({"point": f[4].replace("\\040", " "), "root": f[3], "opts": f[5], "fstype": f[sep + 1],
                   "source": f[sep + 2] if len(f) > sep + 2 else "", "ro": "ro" in f[5].split(",")})
V["mounts"] = mounts
vis = []
cans = inp.get("canaries") or []
for p in cans:
    try:
        os.lstat(p)
        vis.append(p)
    except OSError:
        pass
V["canaries_checked"], V["canaries_visible_n"], V["canaries_visible"] = len(cans), len(vis), vis[:50]
procs = []
for d in os.listdir("/proc"):
    if d.isdigit():
        s = (rd("/proc/%s/stat" % d) or b"").decode(errors="replace")
        try:
            procs.append({"pid": int(d), "comm": s.split("(", 1)[1].rsplit(")", 1)[0],
                          "start": int(s.rsplit(")", 1)[1].split()[19]),
                          "uid": int([l for l in (rd("/proc/%s/status" % d) or b"").decode().splitlines()
                                      if l.startswith("Uid:")][0].split()[1])})
        except Exception:
            pass
V["procs"] = sorted(procs, key=lambda x: x["pid"])
V["host_pid_matches"] = [hp for hp in inp.get("host_pids") or []
                         if any(p["pid"] == hp["pid"] and p["start"] == hp["start"] and p["comm"] == hp["comm"]
                                for p in procs)]
V["ns"] = dict((n, os.readlink("/proc/self/ns/" + n)) for n in ("mnt", "pid", "net", "ipc", "uts", "user", "cgroup"))
out["view"] = V
W = {"rw": {}, "ro": {}}
for kind in ("rw", "ro"):
    for d in inp.get("expect_" + kind) or []:
        p = os.path.join(d, ".evoprobe_%d" % os.getpid())
        try:
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            os.close(fd)
            os.unlink(p)
            W[kind][d] = "WRITABLE"
        except OSError as e:
            W[kind][d] = ename(e)
out["writes"] = W
X = {}
try:
    os.setuid(0)
    X["setuid0"] = "SUCCEEDED"
except OSError as e:
    X["setuid0"] = ename(e)
try:
    os.chroot("/tmp")
    X["chroot"] = "SUCCEEDED"
except OSError as e:
    X["chroot"] = ename(e)
os.makedirs("/tmp/evoprobe_mnt", exist_ok=True)
rc, o = run(["mount", "-t", "tmpfs", "none", "/tmp/evoprobe_mnt"], 20)
X["mount_rc"] = rc
rc, o = run(["umount", "-l", "/Users"], 20)
X["umount_users_rc"] = rc
os.rmdir("/tmp/evoprobe_mnt")
try:
    os.kill(1, 0)
    X["kill_pid1"] = "SUCCEEDED"
except OSError as e:
    X["kill_pid1"] = ename(e)
libc = ctypes.CDLL(None, use_errno=True)
r = libc.ptrace(16, 1, None, None)
X["ptrace_pid1"] = "SUCCEEDED" if r == 0 else errno.errorcode.get(ctypes.get_errno(), str(ctypes.get_errno()))
for k, p in (("pid1_environ", "/proc/1/environ"), ("kcore", "/proc/kcore"), ("sysrq", "/proc/sysrq-trigger")):
    try:
        with open(p, "rb") as f:
            X[k] = "READ %d B" % len(f.read(64))
    except OSError as e:
        X[k] = ename(e)
try:
    with open("/proc/sys/kernel/hostname", "w") as f:
        f.write("x")
    X["proc_sys_write"] = "SUCCEEDED"
except OSError as e:
    X["proc_sys_write"] = ename(e)
rc, o = run(["unshare", "-r", "true"], 20)
X["userns_rc"] = rc
if rc == 0:
    rc2, o2 = run(["unshare", "-rm", "/bin/sh", "-c", "ls -A /"], 20)
    X["userns_view_equal"] = sorted(o2.split()) == V["top"]
out["priv"] = X
N = {"ifaces": [l.split(":")[0].strip() for l in (rd("/proc/net/dev") or b"").decode().splitlines()[2:]],
     "routes": max(0, len((rd("/proc/net/route") or b"").decode().splitlines()) - 1),
     "inet6": [l.split()[-1] for l in (rd("/proc/net/if_inet6") or b"").decode().splitlines()]}
http = []
for u in list(inp.get("must_reach") or []) + list(inp.get("must_refuse") or []):
    try:
        urllib.request.urlopen(urllib.request.Request(u, method="GET"), timeout=15)
        http.append({"url": u, "reachable": True, "detail": "HTTP 2xx/3xx"})
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read(300).decode("utf-8", "replace")
        except Exception:
            pass
        denied = u.startswith("http://") and e.code == 403 and not (body.startswith("fl=") or "\nh=1.1.1.1" in body)
        http.append({"url": u, "reachable": not denied, "detail": "HTTP %d" % e.code,
                     "server": e.headers.get("Server", ""), "body": body[:160]})
    except Exception as e:
        http.append({"url": u, "reachable": False, "detail": ("%s: %s" % (type(e).__name__, e))[:160]})
N["http"] = http
direct = []
np_ = urllib.request.build_opener(urllib.request.ProxyHandler({}))
for t in ("https://example.com", "https://1.1.1.1", "https://github.com", "https://openrouter.ai",
          "https://api.anthropic.com"):
    try:
        np_.open(t, timeout=10)
        direct.append({"target": t, "reached": True, "detail": "HTTP 2xx/3xx"})
    except urllib.error.HTTPError as e:
        direct.append({"target": t, "reached": True, "detail": "HTTP %d" % e.code})
    except Exception as e:
        direct.append({"target": t, "reached": False, "detail": ("%s: %s" % (type(e).__name__, e))[:120]})
N["direct"] = direct
raw = []
def tcp(host, port, fam=socket.AF_INET, label=None):
    s = socket.socket(fam, socket.SOCK_STREAM)
    s.settimeout(6)
    try:
        s.connect((host, port))
        raw.append({"target": label or "%s:%d" % (host, port), "tcp_connected": True})
    except Exception as e:
        raw.append({"target": label or "%s:%d" % (host, port), "tcp_connected": False, "detail": ename(e)})
    finally:
        s.close()
for host, port in (("1.1.1.1", 443), ("8.8.8.8", 443), ("169.254.169.254", 80)):
    tcp(host, port)
tcp("example.com", 22)
tcp("2606:4700:4700::1111", 443, socket.AF_INET6, "[2606:4700:4700::1111]:443")
for ip in inp.get("openrouter_ips") or []:
    tcp(ip, 443, socket.AF_INET6 if ":" in ip else socket.AF_INET, "openrouter %s:443" % ip)
for ip in inp.get("anthropic_ips") or []:
    tcp(ip, 443, socket.AF_INET6 if ":" in ip else socket.AF_INET, "anthropic-direct %s:443" % ip)
for ip in inp.get("host_ips") or []:
    for port in (inp.get("host_ports") or [])[:12] or [22]:
        tcp(ip, port, socket.AF_INET, "machine %s:%d" % (ip, port))
for port in inp.get("host_ports") or []:
    if port != inp.get("proxy_port"):
        tcp("127.0.0.1", port, socket.AF_INET, "loopback 127.0.0.1:%d" % port)
for dst in ("1.1.1.1", "8.8.8.8"):
    try:
        u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        u.settimeout(4)
        u.sendto(b"\x12\x34\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x07example\x03com\x00\x00\x01\x00\x01", (dst, 53))
        u.recvfrom(512)
        raw.append({"target": "udp %s:53" % dst, "answered": True})
    except Exception as e:
        raw.append({"target": "udp %s:53" % dst, "answered": False, "detail": ename(e)})
N["raw"] = raw
ab = []
for name in inp.get("abstract_sockets") or []:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(3)
    try:
        s.connect("\0" + name)
        ab.append({"name": name, "connected": True})
    except Exception as e:
        ab.append({"name": name, "connected": False, "detail": ename(e)})
    finally:
        s.close()
N["abstract"] = ab
dns = {}
for hname in ("example.com", "api.anthropic.com", "openrouter.ai"):
    try:
        dns[hname] = sorted({a[4][0] for a in socket.getaddrinfo(hname, 443)})
    except Exception as e:
        dns[hname] = "FAILED: " + type(e).__name__
N["dns"] = dns
out["net"] = N
print(json.dumps(out))
'''

POST_PROBE_PY = r'''
import json, os
def rd(p):
    try:
        with open(p, "rb") as f:
            return f.read()
    except OSError:
        return b""
mounts = []
for ln in rd("/proc/self/mountinfo").decode().splitlines():
    f = ln.split(" ")
    if "-" in f:
        s = f.index("-")
        mounts.append({"point": f[4].replace("\\040", " "), "fstype": f[s + 1], "ro": "ro" in f[5].split(",")})
procs = []
for d in os.listdir("/proc"):
    if d.isdigit():
        try:
            s = rd("/proc/%s/stat" % d).decode(errors="replace")
            procs.append({"pid": int(d), "comm": s.split("(", 1)[1].rsplit(")", 1)[0]})
        except Exception:
            pass
print(json.dumps({"phase": "post", "view": {"top": sorted(os.listdir("/")), "mounts": mounts,
                                            "procs": sorted(procs, key=lambda x: x["pid"]),
                                            "ifaces": [l.split(":")[0].strip() for l in
                                                       rd("/proc/net/dev").decode().splitlines()[2:]]}}))
'''


def host_canaries(limit: int = 150_000, keep=()) -> tuple:
    """Paths that exist on the MACHINE and must not exist in the container: the machine's /Users tree (dataset
    mirror incl. every golden, repo, runner caches, skills dirs, episode workspaces), /opt (grader tooling, leg and
    pool scripts, this controller's own state), /root, /home, the shim's state and logs, /tmp, /workspace, and the
    secrets /etc keeps. Bounded by `limit`; the dataset workbooks are listed first."""
    first, rest = [], []
    for root, depth in (("/Users", 12), ("/opt", 4), ("/root", 3), ("/home", 3), ("/var/lib/evoshim", 4),
                        ("/var/log/evoshim", 2), ("/tmp", 2), ("/workspace", 2), ("/srv", 2), ("/mnt", 2)):
        if not os.path.isdir(root) or os.path.islink(root):
            continue
        base_depth = root.rstrip("/").count("/")
        for dp, dns, fns in os.walk(root, followlinks=False):
            if dp.startswith(BASE) or dp.count("/") - base_depth >= depth:
                dns[:] = []
            for n in dns + fns:
                p = os.path.join(dp, n)
                if p.startswith(BASE + "/"):
                    continue
                (first if n.lower().endswith((".xlsx", ".xlsm", ".xls", ".csv", ".json")) else rest).append(p)
            if len(first) + len(rest) > 4 * limit:
                break
    extra = [p for p in ("/etc/shadow", "/etc/gshadow", "/etc/ssh", "/etc/ssl/private", "/etc/machine-id",
                         "/var/lib/evoshim", "/var/log/evoshim", "/root", "/workspace", "/opt/evoskill",
                         "/opt/leg", "/opt/pool", BASE, "/sys/fs/cgroup") if os.path.lexists(p)]
    allp = [p for p in extra + first + rest if not any(_under(k, p) for k in keep)]
    return allp[:limit], len(allp) > limit


def host_net_facts() -> dict:
    """The machine's own listeners, addresses and abstract sockets (read in ITS network namespace), and the
    resolved addresses of the two model endpoints: every one must be unreachable from the container."""
    ports = set()
    for f in ("/proc/net/tcp", "/proc/net/tcp6"):
        for ln in _read(f).splitlines()[1:]:
            p = ln.split()
            if len(p) > 3 and p[3] == "0A":
                ports.add(int(p[1].rsplit(":", 1)[1], 16))
    ips = set()
    lines = _read("/proc/net/fib_trie").splitlines()
    for i, ln in enumerate(lines):
        if "/32 host LOCAL" in ln and i > 0:
            ip = lines[i - 1].split()[-1]
            if not ip.startswith("127."):
                ips.add(ip)
    abstract = set()
    for ln in _read("/proc/net/unix").splitlines()[1:]:
        p = ln.split()
        if len(p) >= 8 and p[7].startswith("@"):
            abstract.add(p[7][1:])
    res = {}
    for h in ("openrouter.ai", "api.anthropic.com"):
        try:
            res[h] = sorted({a[4][0] for a in socket.getaddrinfo(h, 443, proto=socket.IPPROTO_TCP)})[:4]
        except OSError:
            res[h] = []
    return {"ports": sorted(ports)[:40], "ips": sorted(ips)[:8], "abstract": sorted(abstract)[:40], "resolved": res}


def host_pids(exclude_ns: str) -> list:
    """Every machine process NOT in the container's pid namespace: (pid, start, comm) -- the container must see none."""
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            if os.readlink(f"/proc/{d}/ns/pid") == exclude_ns:
                continue
            s = _read(f"/proc/{d}/stat")
            out.append({"pid": int(d), "start": int(s.rsplit(")", 1)[1].split()[19]),
                        "comm": s.split("(", 1)[1].rsplit(")", 1)[0]})
        except (OSError, IndexError, ValueError):
            continue
    return out


def probe_input(P: dict, spec: dict, pl: list, st: dict, must_reach, must_refuse) -> dict:
    # a granted path's own ancestors (every directory from the mirror root down to the staging dir) exist in the container as
    # empty mount-point dirs by construction: never canaries
    keep = [r["path"] for r in spec.get("rw") or []] + list(spec.get("ro_rebind") or []) + \
        list(spec.get("readonly_roots") or [])
    cans, capped = host_canaries(keep=keep)
    nf = host_net_facts()
    rw = ["/tmp", "/var/tmp", EVO_HOME, VM_WORK, VM_OUT] + [r["path"] for r in spec.get("rw") or []]
    ro = ["/", "/usr", "/usr/local/bin", "/etc", POLICY_DIR, "/Users", VM_ROOT, "/opt", CC_DIR_CONTAINER, "/dev",
          "/run", "/var", "/home", "/proc/sys"] + list(spec.get("ro_rebind") or [])
    return {"canaries": cans, "canaries_capped": capped, "host_pids": host_pids(st["ns"]["pid"]),
            "host_ports": nf["ports"], "host_ips": nf["ips"], "abstract_sockets": nf["abstract"],
            "openrouter_ips": nf["resolved"].get("openrouter.ai") or [],
            "anthropic_ips": nf["resolved"].get("api.anthropic.com") or [], "proxy_port": PROXY_PORT,
            "expect_rw": rw, "expect_ro": ro, "cc_container": CC_CONTAINER,
            "must_reach": list(must_reach), "must_refuse": list(must_refuse)}


def cmd_probe(a) -> int:
    P, spec, facts, limits, pl, st = load_all(a.cid)
    if a.phase == "post":
        rc, out, err = _enter(a.cid, [PY, "-c", POST_PROBE_PY], timeout=300)
        res = json.loads(out.strip().splitlines()[-1]) if out.strip() else {"error": err[-800:]}
        jprint({"ok": rc == 0, "probe": res})
        return 0 if rc == 0 else 1
    spec_eg = spec.get("egress") or {}
    inp = probe_input(P, spec, pl, st, spec_eg.get("must_reach") or [], spec_eg.get("must_refuse") or [])
    rc, out, err = _enter(a.cid, [PY, "-c", PROBE_PY], inp=json.dumps(inp).encode(), timeout=1500)
    try:
        res = json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        jprint({"ok": False, "problems": ["probe output unparseable"], "stdout_tail": out[-800:],
                "stderr_tail": err[-800:]})
        return 1
    res["host"] = {"canaries_checked": len(inp["canaries"]), "canaries_capped": inp["canaries_capped"],
                   "host_pids_checked": len(inp["host_pids"]), "host_ports": inp["host_ports"],
                   "host_ips": inp["host_ips"], "abstract_sockets_n": len(inp["abstract_sockets"]),
                   "openrouter_ips": inp["openrouter_ips"], "anthropic_ips": inp["anthropic_ips"],
                   "machine_ns": st.get("machine_ns"), "container_ns": st.get("ns"),
                   "expected_top": expected_top(pl), "expected_mounts": expected_mount_points(pl),
                   "proxy": _jload(f"{P['state']}/proxy.ready"), "limits": limits, "facts": facts,
                   "here_string_fd": subprocess.run(["/bin/bash", "-c", "readlink /proc/self/fd/3 3<<<x"],
                                                    capture_output=True, text=True).stdout.strip()}
    if a.episode_check and os.path.exists("/usr/local/bin/docker"):
        r = subprocess.run(["/usr/local/bin/docker", "run", "--rm", "--entrypoint", "/bin/sh", "ssb-solver-lo:latest",
                            "-c", f"ls -A /opt | wc -l; ls -d {BASE} 2>/dev/null | wc -l; "
                                  f"ls /proc | grep -c '^[0-9]'"], capture_output=True, text=True, timeout=120)
        v = r.stdout.split()
        res["host"]["episode_view"] = {"rc": r.returncode, "opt_entries": v[0] if v else None,
                                       "evolver_dir_visible": v[1] if len(v) > 1 else None,
                                       "procs_visible": v[2] if len(v) > 2 else None}
    jprint({"ok": rc == 0, "probe": res})
    return 0 if rc == 0 else 1


# ───────────────────────────────────────────────────────────────────────────── upload / tail / pack / down
def _safe_makedirs(root: str, rel_dir: str, created: list | None = None) -> str:
    """root/rel_dir, created component by component; every existing component must be a real directory (never a
    symlink: a writable dir of the container could hold one, and this runs as root on the machine). Every
    directory this call CREATES is appended to `created` (the incremental extract / restore own only those)."""
    cur = root
    for part in [x for x in rel_dir.split("/") if x]:
        if part in (".", ".."):
            raise RuntimeError(f"bad path component {part!r}")
        cur = os.path.join(cur, part)
        try:
            st = os.lstat(cur)
        except FileNotFoundError:
            os.mkdir(cur, 0o755)
            if created is not None:
                created.append(cur)
            continue
        if not stat.S_ISDIR(st.st_mode):
            raise RuntimeError(f"{cur} is not a plain directory")
    return cur


def cmd_extract(a) -> int:
    """state/upload.tgz -> fs/: regular files only, below Users/ evo/ etc/claude-code/ only; then everything
    root-owned and world-readable, the rw dirs handed back to their owners.
    --incremental (stateful session, epoch N >= 2 delta or a rebuild's cumulative re-upload): create-only into a
    container that already holds files; ownership is set on the written members and created dirs ONLY."""
    P, spec, facts, limits, pl, st = load_all(a.cid)
    tgz = f"{P['state']}/upload.tgz"
    try:
        return _extract(P, spec, tgz, incremental=bool(getattr(a, "incremental", False)))
    except (RuntimeError, OSError) as e:
        jprint({"ok": False, "problems": [f"extract: {e}"]})
        return 2


def _evo_rw_paths(spec: dict) -> list:
    """The spec's rw paths handed to evo (container paths)."""
    return [r["path"] for r in spec.get("rw") or [] if r.get("owner") == "evo"]


def _extract(P: dict, spec: dict, tgz: str, *, incremental: bool = False) -> int:
    bad, n = [], 0
    created: list = []
    written: list = []
    with tarfile.open(tgz, "r:gz") as tf:
        members = tf.getmembers()
        for m in members:
            nm = m.name
            if nm.startswith("/") or ".." in nm.split("/") or not nm.startswith(UPLOAD_TOPS) or not m.isfile():
                bad.append(nm)
        if bad:
            jprint({"ok": False, "problems": [f"refused tar members {bad[:5]}"]})
            return 2
        if incremental:
            # create-only, judged BEFORE anything is written: a row whose path the container already holds
            # (an earlier epoch's upload, or something the evolver left) is a refusal, not a partial extract
            present = [m.name for m in members if os.path.lexists(os.path.join(P["fs"], m.name))]
            if present:
                jprint({"ok": False, "problems": [f"extract --incremental: members already present {present[:5]}"]})
                return 2
        for m in members:
            parent = _safe_makedirs(P["fs"], os.path.dirname(m.name), created if incremental else None)
            dst = os.path.join(parent, os.path.basename(m.name))
            fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
            with tf.extractfile(m) as src, os.fdopen(fd, "wb") as out:
                shutil.copyfileobj(src, out)
            written.append(dst)
            n += 1
    os.unlink(tgz)
    if incremental:
        # ONLY what this call made: a member / created dir below an evo-owned rw path (the /evo/staging seed) goes
        # to evo (editable in place), everything else to root; nothing pre-existing is touched
        evo_rw = _evo_rw_paths(spec)
        fs = P["fs"]
        for x, is_dir in [(d, True) for d in created] + [(f, False) for f in written]:
            cp = "/" + os.path.relpath(x, fs)
            own = (EVO_UID, EVO_GID) if any(_under(cp, w) for w in evo_rw) else (0, 0)
            os.lchown(x, *own)
            os.chmod(x, 0o755 if is_dir else 0o644)
        jprint({"ok": True, "files": n, "incremental": True, "dirs_created": ["/" + os.path.relpath(d, fs)
                                                                                 for d in created]})
        return 0
    for top in ("Users", VM_ROOT.lstrip("/"), POLICY_DIR.lstrip("/")):
        for dp, dns, fns in os.walk(os.path.join(P["fs"], top), followlinks=False):
            for x in [dp] + [os.path.join(dp, y) for y in dns + fns]:
                os.lchown(x, 0, 0)
                if not os.path.islink(x):
                    os.chmod(x, 0o755 if os.path.isdir(x) else 0o644)
    for d in (VM_WORK, VM_OUT):
        os.chown(f"{P['fs']}{d}", EVO_UID, EVO_GID)
    for r in spec.get("rw") or []:
        hp = f"{P['fs']}{r['path']}"
        os.chmod(hp, int(str(r["mode"]), 8))
        os.chown(hp, *((EVO_UID, EVO_GID) if r["owner"] == "evo" else (0, 0)))
    jprint({"ok": True, "files": n})
    return 0


def _read_nofollow(p: str, offset: int = 0, maxb: int = 64) -> tuple:
    """(size, bytes) of a REGULAR file, never through a symlink; (0, b'') otherwise."""
    try:
        fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0))
    except OSError:
        return 0, b""
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            return 0, b""
        os.lseek(fd, offset, 0)
        buf, left = [], maxb
        while left > 0:
            b = os.read(fd, min(left, 1 << 20))
            if not b:
                break
            buf.append(b)
            left -= len(b)
        return st.st_size, b"".join(buf)
    finally:
        os.close(fd)


def rw_usage(P: dict, spec: dict) -> int:
    tot = 0
    roots = [P["home"], f"{P['fs']}{VM_WORK}", f"{P['fs']}{VM_OUT}"] + [f"{P['fs']}{r['path']}" for r in spec.get("rw") or []]
    for r in roots:
        for dp, dns, fns in os.walk(r, followlinks=False):
            for f in fns:
                try:
                    tot += os.lstat(os.path.join(dp, f)).st_blocks * 512
                except OSError:
                    pass
    return tot


def cmd_tail(a) -> int:
    P, spec, facts, limits, pl, st = load_all(a.cid)
    base = f"{P['fs']}{VM_OUT}"
    size, data = _read_nofollow(f"{base}/{a.name}", int(a.offset), int(a.max))
    _, rcb = _read_nofollow(f"{base}/rc", 0, 32)
    rc = rcb.decode("ascii", "replace").strip() or "RUNNING"
    if not re.fullmatch(r"-?\d{1,4}|RUNNING", rc):
        rc = "-1"
    du = rw_usage(P, spec) if a.du else -1
    alive_ = int(bool(st.get("init_pid") and alive(st["init_pid"], st.get("init_start"))))
    sys.stdout.write(f"{size} {rc} {du} {alive_}\n@@B64@@{base64.b64encode(data).decode()}\n")
    return 0


STOP_SH = (f'P=$(cat {VM_OUT}/pid 2>/dev/null); if [ -n "$P" ]; then kill -TERM "$P" 2>/dev/null; sleep 20; '
           f'kill -KILL "$P" 2>/dev/null; fi; sleep 3; [ -s {VM_OUT}/rc ] || '
           f'{{ echo 124 > {VM_OUT}/rc.tmp && mv {VM_OUT}/rc.tmp {VM_OUT}/rc; }}; echo STOPPED')


def cmd_stop_session(a) -> int:
    rc, out, err = _enter(a.cid, ["/bin/bash", "-c", STOP_SH], cwd="/tmp", timeout=120)
    jprint({"ok": "STOPPED" in out, "rc": rc})
    return 0


def cmd_pack(a) -> int:
    """state/out.tgz of the given CONTAINER paths, read from their host dirs with no symlink followed anywhere
    (a planted link is stored as a link, and the driver refuses it). Run after `down`: nothing writes any more."""
    P, spec, facts, limits, pl, st = load_all(a.cid)
    if st.get("init_pid") and alive(st["init_pid"], st.get("init_start")):
        jprint({"ok": False, "problems": ["pack before down: the container is still up"]})
        return 2
    out = f"{P['state']}/out.tgz"
    packed, refused = _pack_parts(P, pl, a.cid, list(a.part))
    jprint({"ok": True, "tgz": out, "bytes": os.path.getsize(out), "packed": packed, "refused": refused})
    return 0


def _pack_parts(P: dict, pl: list, cid: str, parts: list) -> tuple:
    """state/out.tgz of the given CONTAINER paths, read from their host dirs with no symlink followed anywhere
    (a planted link is stored as a link, and the driver refuses it). (packed, refused). Extracted from `pack`;
    `epoch-end` calls it while the container is FROZEN (stateful session)."""
    out, refused, packed = f"{P['state']}/out.tgz", [], []

    def clean(ti):
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = "root"
        return ti
    with tarfile.open(out, "w:gz", format=tarfile.PAX_FORMAT, dereference=False) as tf:
        for part in parts:
            hp, why = safe_host_path(pl, part, cid)
            if hp is None:
                refused.append({"part": part, "why": why})
                continue
            tf.add(hp, arcname=part.lstrip("/"), recursive=True, filter=clean)
            packed.append(part)
    return packed, refused


def _scan_container(P: dict) -> dict:
    res = {}
    for label, root in (("fs", P["fs"]), ("rw", P["rw"]), ("state", P["state"])):
        hits, n = scan_tree(root, (TOKEN_PREFIX, OPENROUTER_PREFIX))
        res[label] = {"files": n, "hits": [f"{label}/{h[0]}:{'token-prefix' if h[1] == 0 else 'openrouter-key'}"
                                           for h in hits]}
    return res


def cmd_scan(a) -> int:
    P = paths(a.cid)
    jprint({"ok": True, "scan": _scan_container(P)})
    return 0


def cmd_down(a) -> int:
    """Stop the container and prove it: freeze, scan its tmpfs mounts, kill everything in its cgroup, stop the
    proxy, remove the cgroup, check no process remains in its namespaces, scan its writable state for the token
    prefix; with --wipe remove c/<cid>. The machine and every other process on it are left alone."""
    P = paths(a.cid)
    st = _jload(f"{P['state']}/state.json")
    res: dict = {"cid": a.cid}
    cg = P["cg"]
    init_pid, init_start = st.get("init_pid"), st.get("init_start")
    if os.path.isdir(cg) and os.path.exists(f"{cg}/cgroup.freeze"):
        try:
            _write(f"{cg}/cgroup.freeze", "1")
            res["frozen"] = True
        except OSError as e:
            res["frozen"] = f"failed: {e}"
    if init_pid and alive(init_pid, init_start):
        tmp_hits, tmp_n = [], 0
        for sub in ("tmp", "var/tmp", "dev/shm", "run"):
            h, n = scan_tree(f"/proc/{init_pid}/root/{sub}", (TOKEN_PREFIX,))
            tmp_hits += [f"{sub}/{x[0]}" for x in h]
            tmp_n += n
        res["tmpfs_scan"] = {"files": tmp_n, "hits": tmp_hits}
    if os.path.isdir(cg):
        try:
            _write(f"{cg}/cgroup.kill", "1")
        except OSError as e:
            res["cgroup_kill_error"] = str(e)
    for k in ("holder", "init"):
        pid, start = st.get(f"{k}_pid"), st.get(f"{k}_start")
        if pid and alive(pid, start):
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
    if os.path.isdir(cg) and os.path.exists(f"{cg}/cgroup.freeze"):
        try:
            _write(f"{cg}/cgroup.freeze", "0")
        except OSError:
            pass
    ppid, pstart = st.get("proxy_pid"), st.get("proxy_start")
    if ppid and alive(ppid, pstart):
        try:
            os.kill(ppid, signal.SIGTERM)
        except OSError:
            pass
        for _ in range(50):
            if not alive(ppid, pstart):
                break
            time.sleep(0.1)
        if alive(ppid, pstart):
            os.kill(ppid, signal.SIGKILL)
    for _ in range(100):
        if not os.path.isdir(cg) or not _read(f"{cg}/cgroup.procs").strip():
            break
        time.sleep(0.1)
    if os.path.isdir(cg):
        try:
            os.rmdir(cg)
        except OSError as e:
            res["cgroup_rmdir_error"] = str(e)
    left = []
    mnt = (st.get("ns") or {}).get("mnt")
    for d in os.listdir("/proc"):
        if d.isdigit() and mnt:
            try:
                if os.readlink(f"/proc/{d}/ns/mnt") == mnt:
                    left.append(int(d))
            except OSError:
                pass
    res.update(procs_in_container_ns=left, init_alive=bool(init_pid and alive(init_pid, init_start)),
               holder_alive=bool(st.get("holder_pid") and alive(st["holder_pid"], st.get("holder_start"))),
               proxy_alive=bool(ppid and alive(ppid, pstart)), cgroup_removed=not os.path.isdir(cg))
    if os.path.isdir(P["c"]):
        res["scan"] = _scan_container(P)
        res["egress_log"] = egress_summary(f"{P['state']}/egress.jsonl")
    if a.wipe and os.path.isdir(P["c"]):
        shutil.rmtree(P["c"], ignore_errors=True)
    res["wiped"] = not os.path.exists(P["c"])
    res["verified"] = (not left and not res["init_alive"] and not res["holder_alive"] and not res["proxy_alive"]
                       and res["cgroup_removed"])
    if os.path.isdir(P["state"]):
        _jsave(f"{P['state']}/down.json", res)
    jprint(res)
    return 0 if res["verified"] else 1


def egress_summary(p: str) -> dict:
    by, n = {}, 0
    for ln in _read(p).splitlines():
        try:
            r = json.loads(ln)
        except ValueError:
            continue
        n += 1
        k = f"{r.get('host') or r.get('method') or '?'}:{'allowed' if r.get('allowed') else 'refused'}"
        by[k] = by.get(k, 0) + 1
    return {"requests": n, "by_host": by}


# ───────────────────────────────────────────────────────────────────────────── the stateful session
# ONE container for the whole run: between epochs it idles (init reaper + proxy, no session); `epoch-begin` readies it for the
# next `claude --resume` turn, `epoch-end` collects without `down`, `restore` refills a REBUILT container's HOME
# and workspace from the driver's snapshot before `up`. `down [--wipe]` is untouched (the run-level `finish` calls it).
OUT_RESET_NAMES = ("pid", "rc", "rc.tmp", "evolver.log", "evolver.stderr")
RESTORE_HOME_PREFIX = f"{EVO_HOME.lstrip('/')}/.claude/projects/"     # home/evo/.claude/projects/
RESTORE_NOTES_PREFIX = f"{VM_ROOT.lstrip('/')}/evolver_workspace/"   # evo/evolver_workspace/
RESTORE_PREFIXES = (RESTORE_HOME_PREFIX, RESTORE_NOTES_PREFIX)
TMPFS_SUBS = ("tmp", "var/tmp", "dev/shm", "run")


def _init_alive(st: dict) -> bool:
    return bool(st.get("init_pid") and alive(st["init_pid"], st.get("init_start")))


def _session_pids(P: dict, st: dict) -> list:
    """Pids in the container's cgroup other than the holder and PID 1: a session (claude, its children, an
    `enter`) is still running. Zombies never appear in cgroup.procs."""
    own = {int(st.get(k) or 0) for k in ("holder_pid", "init_pid")}
    out = []
    for tok in _read(f"{P['cg']}/cgroup.procs").split():
        try:
            pid = int(tok)
        except ValueError:
            continue
        if pid not in own:
            out.append(pid)
    return sorted(set(out))


def _freeze(cg: str, on: bool):
    """cgroup.freeze 1 / 0 -> True, or the failure text. A freeze settles asynchronously: wait (<= 5 s) for
    cgroup.events to report it, so nothing in the container writes while the pack reads."""
    f = f"{cg}/cgroup.freeze"
    if not (os.path.isdir(cg) and os.path.exists(f)):
        return "failed: no cgroup.freeze"
    try:
        _write(f, "1" if on else "0")
    except OSError as e:
        return f"failed: {e}"
    ev, want = f"{cg}/cgroup.events", ("frozen 1" if on else "frozen 0")
    for _ in range(50):
        if not os.path.exists(ev) or want in _read(ev):
            break
        time.sleep(0.1)
    return True


def _tmpfs_scan(init_pid: int) -> dict:
    """The token-prefix scan of the container's private tmpfs mounts through PID 1's root (as `down` does)."""
    tmp_hits, tmp_n = [], 0
    for sub in TMPFS_SUBS:
        h, n = scan_tree(f"/proc/{int(init_pid)}/root/{sub}", (TOKEN_PREFIX,))
        tmp_hits += [f"{sub}/{x[0]}" for x in h]
        tmp_n += n
    return {"files": tmp_n, "hits": tmp_hits}


def _clear_tree(top: str) -> int:
    """Empty `top` (a real directory) bottom-up, following no symlink: files and symlinks are unlinked, dirs
    removed, the top dir itself kept. Returns the number of entries removed."""
    if not stat.S_ISDIR(os.lstat(top).st_mode):
        raise RuntimeError(f"{top} is not a plain directory")
    n = 0
    for dp, dns, fns in os.walk(top, topdown=False, followlinks=False):
        for name in fns + dns:            # a symlink to a dir sits in dns (never descended): lstat decides
            p = os.path.join(dp, name)
            if stat.S_ISDIR(os.lstat(p).st_mode):
                os.rmdir(p)
            else:
                os.unlink(p)
            n += 1
    return n


def cmd_epoch_begin(a) -> int:
    """Stateful session, epoch N >= 2 (and after a rebuild): make the kept container ready for the next session
    turn. init alive; no session running; thawed; the proxy alive (restarted if it died); /evo/out reset (the
    stale rc/pid/logs of the last epoch, so RUNNING holds until THIS epoch's rc); each --clear rw dir emptied
    (the staging is re-seeded by the upload that follows)."""
    P, spec, facts, limits, pl, st = load_all(a.cid)
    if not _init_alive(st):
        jprint({"ok": False, "problems": ["container not up"]})
        return 2
    procs = _session_pids(P, st)
    if procs:
        jprint({"ok": False, "problems": ["a session is still running in the container; stop-session first"],
                "procs": procs[:10]})
        return 2
    res: dict = {"ok": True, "init_alive": True, "thawed": False, "proxy_restarted": False}
    cg = P["cg"]
    if os.path.exists(f"{cg}/cgroup.freeze") and _read(f"{cg}/cgroup.freeze").strip() == "1":
        _write(f"{cg}/cgroup.freeze", "0")
        res["thawed"] = True
    if not (st.get("proxy_pid") and alive(st["proxy_pid"], st.get("proxy_start"))):
        try:
            os.unlink(f"{P['state']}/proxy.ready")       # the dead proxy's record must not pass for the new one
        except FileNotFoundError:
            pass
        _start_proxy(P, a.cid, int(st["init_pid"]), st)
        res["proxy_restarted"] = True
    ready = st.get("proxy") or {}
    res["proxy"] = ready
    if not ready or ready.get("listen_netns") != (st.get("ns") or {}).get("net"):
        res.update(ok=False, problems=["egress proxy not listening in the container's network namespace"],
                   proxy_out_tail=_read(f"{P['state']}/proxy.out")[-800:])
        jprint(res)
        return 3
    base, removed = f"{P['fs']}{VM_OUT}", []
    for name in OUT_RESET_NAMES:
        p = f"{base}/{name}"
        try:
            s = os.lstat(p)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(s.st_mode):
            res.update(ok=False, problems=[f"symlink in {VM_OUT}: {name}"])
            jprint(res)
            return 2
        if not stat.S_ISREG(s.st_mode):
            res.update(ok=False, problems=[f"{VM_OUT}/{name} is not a regular file"])
            jprint(res)
            return 2
        os.unlink(p)
        removed.append(name)
    res["out_reset"] = removed
    evo_rw, cleared = _evo_rw_paths(spec), {}
    for cp in a.clear or []:
        if cp not in evo_rw:
            res.update(ok=False, problems=[f"--clear {cp}: not an evo-owned rw path of the spec ({evo_rw})"])
            jprint(res)
            return 2
        top = f"{P['fs']}{cp}"
        try:
            cleared[cp] = _clear_tree(top)
        except (RuntimeError, OSError) as e:
            res.update(ok=False, problems=[f"--clear {cp}: {e}"])
            jprint(res)
            return 2
        os.chown(top, EVO_UID, EVO_GID)
        os.chmod(top, 0o755)
    res["cleared"] = cleared
    jprint(res)
    return 0


def cmd_epoch_end(a) -> int:
    """Stateful session: collect an epoch's outputs WITHOUT `down`. The container is FROZEN (cgroup.freeze) so
    nothing writes while its tmpfs mounts are scanned for the token prefix and the parts are packed, its
    writable state is scanned, then it is THAWED and must still be alive. Saved as state/epoch_end.json."""
    P, spec, facts, limits, pl, st = load_all(a.cid)
    res: dict = {"ok": False, "cid": a.cid, "kept": True, "frozen": False, "thawed": False, "init_alive": False}
    if not _init_alive(st):
        res["problems"] = ["container not up"]
        jprint(res)
        return 2
    cg, out = P["cg"], f"{P['state']}/out.tgz"
    res["session_procs"] = _session_pids(P, st)
    res["frozen"] = _freeze(cg, True)
    try:
        res["tmpfs_scan"] = _tmpfs_scan(int(st["init_pid"]))
        packed, refused = _pack_parts(P, pl, a.cid, list(a.part))
        res.update(tgz=out, bytes=os.path.getsize(out), packed=packed, refused=refused)
        res["scan"] = _scan_container(P)
        res["egress_log"] = egress_summary(f"{P['state']}/egress.jsonl")
    except (OSError, RuntimeError, tarfile.TarError) as e:
        res["problems"] = [f"epoch-end: {e}"]
    finally:
        res["thawed"] = _freeze(cg, False)
    res["init_alive"] = _init_alive(st)
    res["ok"] = bool(res["frozen"] is True and res["thawed"] is True and res["init_alive"] and "problems" not in res)
    if os.path.isdir(P["state"]):
        _jsave(f"{P['state']}/epoch_end.json", res)
    jprint(res)
    return 0 if res["ok"] else 1


def _restore_dest(P: dict, member: str) -> tuple:
    """(root, rel) on the host for a restore member: home/evo/... -> rw/home/..., evo/evolver_workspace/... ->
    fs/evo/evolver_workspace/..."""
    home_top = EVO_HOME.lstrip("/") + "/"
    if member.startswith(home_top):
        return P["home"], member[len(home_top):]
    return P["fs"], member


def cmd_restore(a) -> int:
    """Stateful session, REBUILD (a replaced machine or a lost container): state/restore.tgz -> the newest driver-side
    snapshot's session transcripts (home/evo/.claude/projects/**) into rw/home/.claude/projects/ and the evolver's
    workspace (evo/evolver_workspace/**) into fs/evo/evolver_workspace/, regular files only, create-only, evo-owned
    (dirs 0755, files 0644).
    Only BEFORE `up` (state.json must not exist): HOME is never written from outside a running container."""
    P, spec, facts, limits, pl, st = load_all(a.cid)
    if os.path.exists(f"{P['state']}/state.json"):
        jprint({"ok": False, "problems": ["container already up (state.json exists); restore runs before up"]})
        return 2
    tgz = f"{P['state']}/restore.tgz"
    bad, created, n = [], [], 0
    try:
        with tarfile.open(tgz, "r:gz") as tf:
            members = tf.getmembers()
            for m in members:
                nm = m.name
                if nm.startswith("/") or ".." in nm.split("/") or not nm.startswith(RESTORE_PREFIXES) \
                        or not m.isfile():
                    bad.append(nm)
            if bad:
                jprint({"ok": False, "problems": [f"refused tar members {bad[:5]}"]})
                return 2
            notes_cp = "/" + RESTORE_NOTES_PREFIX.rstrip("/")
            if any(m.name.startswith(RESTORE_NOTES_PREFIX) for m in members) and notes_cp not in _evo_rw_paths(spec):
                jprint({"ok": False, "problems": [f"restore: {notes_cp} is not an evo-owned rw path of the spec"]})
                return 2
            present = [m.name for m in members if os.path.lexists(os.path.join(*_restore_dest(P, m.name)))]
            if present:
                jprint({"ok": False, "problems": [f"restore: members already present {present[:5]}"]})
                return 2
            for m in members:
                root, rel = _restore_dest(P, m.name)
                parent = _safe_makedirs(root, os.path.dirname(rel), created)
                dst = os.path.join(parent, os.path.basename(rel))
                fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
                with tf.extractfile(m) as src, os.fdopen(fd, "wb") as fh:
                    shutil.copyfileobj(src, fh)
                os.lchown(dst, EVO_UID, EVO_GID)
                os.chmod(dst, 0o644)
                n += 1
        for d in created:
            os.lchown(d, EVO_UID, EVO_GID)
            os.chmod(d, 0o755)
        os.unlink(tgz)
    except (RuntimeError, OSError, tarfile.TarError) as e:
        jprint({"ok": False, "problems": [f"restore: {e}"], "files": n, "dirs": len(created)})
        return 2
    jprint({"ok": True, "files": n, "dirs": len(created)})
    return 0


def cmd_status(a) -> int:
    P = paths(a.cid)
    st = _jload(f"{P['state']}/state.json")
    jprint({"cid": a.cid, "exists": os.path.isdir(P["c"]), "state": st,
            "init_alive": bool(st.get("init_pid") and alive(st["init_pid"], st.get("init_start"))),
            "proxy_alive": bool(st.get("proxy_pid") and alive(st["proxy_pid"], st.get("proxy_start")))})
    return 0


def cmd_list(a) -> int:
    d = f"{BASE}/c"
    cids = sorted(os.listdir(d)) if os.path.isdir(d) else []
    jprint({"containers": cids, "cgroups": sorted(os.listdir(CG_PARENT)) if os.path.isdir(CG_PARENT) else []})
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="evoctl")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("facts")
    s = sub.add_parser("prepare")
    s.add_argument("--spec-b64", required=True)
    for name in ("up", "probe", "extract", "tail", "stop-session", "pack", "scan", "down", "status", "proxy", "enter",
                 "epoch-begin", "epoch-end", "restore"):
        s = sub.add_parser(name)
        s.add_argument("--cid", required=True)
        if name == "up":
            s.add_argument("--init-root-cgroup", action="store_true")
        if name == "extract":
            s.add_argument("--incremental", action="store_true")
        if name == "epoch-begin":
            s.add_argument("--clear", action="append", default=[])
        if name == "epoch-end":
            s.add_argument("--part", action="append", required=True)
        if name == "probe":
            s.add_argument("--phase", choices=("pre", "post"), default="pre")
            s.add_argument("--episode-check", action="store_true")
        if name == "tail":
            s.add_argument("--offset", default="0")
            s.add_argument("--max", default="3000000")
            s.add_argument("--name", default="evolver.log", choices=("evolver.log", "evolver.stderr"))
            s.add_argument("--du", action="store_true")
        if name == "pack":
            s.add_argument("--part", action="append", required=True)
        if name == "down":
            s.add_argument("--wipe", action="store_true")
        if name == "proxy":
            s.add_argument("--init-pid", required=True)
        if name == "enter":
            s.add_argument("--cwd", default=VM_WORK)
            s.add_argument("--keep-fd3", action="store_true")
            s.add_argument("--log", default="")
            s.add_argument("argv", nargs=argparse.REMAINDER)
    sub.add_parser("list")
    a = ap.parse_args(argv)
    fn = {"facts": cmd_facts, "prepare": cmd_prepare, "up": cmd_up, "proxy": cmd_proxy, "enter": cmd_enter,
          "probe": cmd_probe, "extract": cmd_extract, "tail": cmd_tail, "stop-session": cmd_stop_session,
          "pack": cmd_pack, "scan": cmd_scan, "down": cmd_down, "status": cmd_status, "list": cmd_list,
          "epoch-begin": cmd_epoch_begin, "epoch-end": cmd_epoch_end, "restore": cmd_restore}[a.cmd]
    return fn(a)


if __name__ == "__main__":
    sys.exit(main())
