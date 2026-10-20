#!/usr/local/bin/python3
"""`docker` stand-in for an SSB v2 leg INSIDE a Daytona sandbox (installed as /usr/local/bin/docker).

WHY. The sandbox image IS the solver image (scripts/daytona/Dockerfile.ssb_leg), and nested
dockerd was never exercised at production shape, so each "container" the
harness asks for becomes a namespaced process tree on this one image, keeping what a container
gave the episode:

  * private mount / net / pid / ipc / uts namespaces (unshare), /proc remounted;
  * `-v SRC:DST[:ro]` bind mounts (the episode workspace at /workspace), `-w`, `-e`;
  * private disk-backed /tmp and /var/tmp, and a private COPY of the LibreOffice profile
    (the solver image's warm profile for ssb-solver-lo; the recalc-always grading profile for
    ssb-verifier) -- soffice locks its profile and names its pipe in /tmp;
  * no network at all (the SSB solver never needs it; the Mac containers had DNS blackholed);
  * docker's DEFAULT capability set (setpriv): no CAP_SYS_ADMIN, so the process cannot unmount
    the tmpfs that hides /Users (dataset incl. golden answers, repo, caches), /opt (runner venv),
    /home and this shim's state; cgroupfs and /proc/sys are read-only inside;
  * ONE exception under the hidden dirs: a skill leg's declared skills dir, which
    the workspace's .claude/skills link names, is bound back READ-ONLY at its own path so the
    link resolves (see skill_exposures). A base leg's episode is unchanged;
  * a clean environment built from the solver image's ENV (+ `-e`, + explicit thread counts),
    NEVER the caller's -- the runner holds OPENROUTER_API_KEY and must not hand it to a solver;
  * a per-container child cgroup with memory.max (EVOSHIM_MEM_MAX, default 4G) and swap off:
    the audit's per-episode memory cap. Stats (memory.peak, cpu usage, oom kills) are appended
    to /var/log/evoshim/containers.jsonl when a container is removed / a run ends.

SUPPORTED, i.e. exactly what src/harness/direct/executor.py, the SSB v2 scorer and the runner's
image probe issue (anything else exits 125 and is logged):
  docker images --format {{.ID}} IMAGE
  docker image inspect IMAGE --format {{.Config.WorkingDir}}
  docker run -d ... --name N ... IMAGE CMD...        (long-lived episode container)
  docker run --rm [--entrypoint E] [-v ...] [-w D] IMAGE CMD...   (probe, recalc, UNO fix)
  docker exec [-i] [-t] [-w D] [-e K=V] N CMD...
  docker inspect [-f|--format FMT] N [--format FMT]   (State.Running, Mounts formats used)
  docker rm [-f] N...
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid

STATE = os.environ.get("EVOSHIM_STATE", "/var/lib/evoshim")
LOGDIR = os.environ.get("EVOSHIM_LOGDIR", "/var/log/evoshim")
CG_ROOT = os.environ.get("EVOSHIM_CGROUP_ROOT", "/sys/fs/cgroup/evoshim")
MEM_MAX = os.environ.get("EVOSHIM_MEM_MAX", "4G")
REQUIRE_CGROUP = os.environ.get("EVOSHIM_REQUIRE_CGROUP", "1") == "1"
IMAGE_ID = os.environ.get("EVOSHIM_IMAGE_ID", "x86daytona-unknown")
CONTAINER_ENV_FILE = "/etc/evoskill_container_env"
SOLVER_PROFILE = "/root/.config/libreoffice"
VERIFIER_PROFILE = "/opt/evoskill/verifier-lo-profile"
# Directories the episode must not see. Each gets an empty tmpfs in the episode's mount ns.
HIDE = ("/Users", "/opt", "/home", STATE, LOGDIR)
# The episode workspace's link to the leg's skills dir (task_workspace.setup_task_workspace), and
# the one env var that declares which dir a leg's skills live in (ssb_leg_daytona_loop leg.env).
SKILLS_LINK = os.path.join(".claude", "skills")
SKILLS_DECL = "EVOSKILL_SKILLS_DIR"
# Docker's default capability set (moby oci/caps/defaults.go).
DOCKER_CAPS = ("chown", "dac_override", "fsetid", "fowner", "mknod", "net_raw", "setgid",
               "setuid", "setfcap", "setpcap", "net_bind_service", "sys_chroot", "kill",
               "audit_write")
SETPRIV = ["/usr/bin/setpriv", "--bounding-set=-all," + ",".join("+" + c for c in DOCKER_CAPS),
           "--inh-caps=-all", "--"]
# PID 1 of a long-lived episode container: reaps orphans (docker ran these with --init).
INIT = ("import os,signal,time\n"
        "signal.signal(signal.SIGTERM,lambda *a:os._exit(0))\n"
        "while True:\n"
        "    try: os.wait()\n"
        "    except ChildProcessError: time.sleep(0.5)\n")
LO_UP = ("/usr/local/bin/python3 -c 'import socket,fcntl,struct;s=socket.socket();"
         "ifr=struct.pack(\"16sh\",b\"lo\",0);fl=struct.unpack(\"16sh\",fcntl.ioctl(s,0x8913,ifr))[1];"
         "fcntl.ioctl(s,0x8914,struct.pack(\"16sh\",b\"lo\",fl|1))'")


def _images(kind: str) -> set[str]:
    default = {"solver": "ssb-solver-lo:latest,ssb-solver-lo,ssb-solver:latest,ssb-solver",
               "verifier": "ssb-verifier:latest,ssb-verifier,ssb-verifier:recalc"}[kind]
    return {x.strip() for x in os.environ.get(f"EVOSHIM_{kind.upper()}_IMAGES", default).split(",")
            if x.strip()}


def image_kind(image: str) -> str | None:
    if image in _images("solver"):
        return "solver"
    if image in _images("verifier"):
        return "verifier"
    return None


# ----------------------------------------------------------------------------------- parsing
class Usage(Exception):
    pass


_RUN_VALUE_FLAGS = {"--name", "--network", "--net", "--dns", "--cpus", "--memory", "-m", "-e",
                    "--env", "--entrypoint", "-v", "--volume", "-w", "--workdir", "--label", "-l",
                    "--user", "-u", "--hostname", "--memory-swap", "--pids-limit", "--shm-size"}
_RUN_BOOL_FLAGS = {"--rm", "-d", "--detach", "--init", "-i", "-t", "-it", "-dit", "--interactive",
                   "--tty", "--read-only"}


def parse_run(argv: list[str]) -> dict:
    """`docker run` flags -> dict. Supports `--flag value` and `--flag=value`."""
    o = {"detach": False, "rm": False, "name": None, "volumes": [], "env": [], "workdir": None,
         "entrypoint": None, "labels": [], "image": None, "cmd": [], "ignored": []}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--") and "=" in a:
            a, v = a.split("=", 1)
            has_inline = True
        else:
            v, has_inline = None, False
        if a in _RUN_BOOL_FLAGS:
            if a in ("-d", "--detach", "-dit"):
                o["detach"] = True
            if a == "--rm":
                o["rm"] = True
            i += 1
            continue
        if a in _RUN_VALUE_FLAGS:
            if not has_inline:
                if i + 1 >= len(argv):
                    raise Usage(f"flag {a} needs a value")
                v = argv[i + 1]
                i += 2
            else:
                i += 1
            if a == "--name":
                o["name"] = v
            elif a in ("-v", "--volume"):
                o["volumes"].append(parse_volume(v))
            elif a in ("-e", "--env"):
                o["env"].append(v)
            elif a in ("-w", "--workdir"):
                o["workdir"] = v
            elif a == "--entrypoint":
                o["entrypoint"] = v
            elif a in ("--label", "-l"):
                o["labels"].append(v)
            else:
                o["ignored"].append(f"{a}={v}")
            continue
        if a.startswith("-"):
            raise Usage(f"unsupported docker run flag {a}")
        o["image"] = a
        o["cmd"] = argv[i + 1:]
        break
    if not o["image"]:
        raise Usage("docker run: no image")
    if o["entrypoint"]:
        o["cmd"] = [o["entrypoint"]] + o["cmd"]
    if not o["cmd"]:
        o["cmd"] = ["python3"]          # python:3.11-slim's CMD
    return o


def parse_volume(spec: str) -> tuple[str, str, bool]:
    parts = spec.split(":")
    if len(parts) < 2 or not parts[0].startswith("/") or not parts[1].startswith("/"):
        raise Usage(f"unsupported volume spec {spec!r} (need /abs/src:/abs/dst[:ro])")
    return parts[0], parts[1], (len(parts) > 2 and "ro" in parts[2].split(","))


def parse_exec(argv: list[str]) -> dict:
    o = {"workdir": None, "env": [], "name": None, "cmd": []}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("-i", "-t", "-it", "--interactive", "--tty", "-d"):
            i += 1
        elif a in ("-w", "--workdir"):
            o["workdir"] = argv[i + 1]
            i += 2
        elif a in ("-e", "--env"):
            o["env"].append(argv[i + 1])
            i += 2
        elif a.startswith("-"):
            raise Usage(f"unsupported docker exec flag {a}")
        else:
            o["name"], o["cmd"] = a, argv[i + 1:]
            break
    if not o["name"] or not o["cmd"]:
        raise Usage("docker exec: need CONTAINER CMD")
    return o


def parse_inspect(argv: list[str]) -> tuple[str | None, list[str]]:
    fmt, names, i = None, [], 0
    while i < len(argv):
        a = argv[i]
        if a in ("-f", "--format"):
            fmt = argv[i + 1]
            i += 2
        elif a.startswith("--format="):
            fmt = a.split("=", 1)[1]
            i += 1
        elif a.startswith("-"):
            raise Usage(f"unsupported docker inspect flag {a}")
        else:
            names.append(a)
            i += 1
    return fmt, names


def render_inspect(fmt: str | None, st: dict, running: bool) -> str:
    """The three Go-template formats the executor uses, rendered from shim state."""
    mounts = st.get("mounts", [])
    if fmt is None:
        return json.dumps([{"Name": "/" + st["name"], "Id": st["cid"],
                            "State": {"Running": running},
                            "Mounts": [{"Source": s, "Destination": d, "RW": not ro}
                                       for s, d, ro in mounts],
                            "Config": {"WorkingDir": st.get("workdir") or "/workspace",
                                       "Image": st.get("image")}}], indent=1) + "\n"
    f = fmt.replace("\\n", "\n")
    if f.strip() == "{{.State.Running}}":
        return ("true" if running else "false") + "\n"
    if f.strip() == '{{range .Mounts}}{{if eq .Destination "/workspace"}}{{.Source}}{{end}}{{end}}':
        return "".join(s for s, d, _ in mounts if d == "/workspace") + "\n"
    if f == "{{range .Mounts}}{{.Destination}}\n{{end}}":
        return "".join(d + "\n" for _, d, _ in mounts)
    if f.strip() == "{{.Config.WorkingDir}}":
        return (st.get("workdir") or "/workspace") + "\n"
    raise Usage(f"unsupported inspect format {fmt!r}")


# ----------------------------------------------------------------------------------- helpers
def log(event: str, **kw) -> None:
    try:
        os.makedirs(LOGDIR, exist_ok=True)
        with open(os.path.join(LOGDIR, "shim.log"), "a") as fh:
            fh.write(json.dumps({"ts": round(time.time(), 3), "pid": os.getpid(), "event": event, **kw})
                     + "\n")
    except OSError:
        pass


def container_env(name: str, extra: list[str]) -> list[str]:
    env: dict[str, str] = {}
    try:
        for ln in open(CONTAINER_ENV_FILE):
            ln = ln.rstrip("\n")
            if "=" in ln:
                k, v = ln.split("=", 1)
                env[k] = v
    except OSError:
        env["PATH"] = "/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        env["LANG"] = "C.UTF-8"
    env["HOME"] = "/root"
    env["HOSTNAME"] = name[:12]
    try:
        env.update(json.loads(os.environ.get("EVOSHIM_EXTRA_ENV", "{}") or "{}"))
    except ValueError:
        pass
    for e in extra:
        if "=" in e:
            k, v = e.split("=", 1)
            env[k] = v
        elif e in os.environ and e not in ("OPENROUTER_API_KEY",):
            env[e] = os.environ[e]
    env.pop("OPENROUTER_API_KEY", None)
    return [f"{k}={v}" for k, v in env.items()]


def cg_path(name: str) -> str:
    return os.path.join(CG_ROOT, name)


def cg_create(name: str) -> str | None:
    p = cg_path(name)
    try:
        os.makedirs(p, exist_ok=True)
        with open(os.path.join(p, "memory.max"), "w") as fh:
            fh.write(MEM_MAX)
        sw = os.path.join(p, "memory.swap.max")
        if os.path.exists(sw):
            with open(sw, "w") as fh:
                fh.write("0")
        return p
    except OSError as e:
        log("cgroup_create_failed", name=name, error=str(e))
        if REQUIRE_CGROUP:
            sys.stderr.write(f"docker-shim: cannot create cgroup {p}: {e}\n")
            sys.exit(125)
        return None


def cg_join(p: str | None, pid: int) -> None:
    if p:
        with open(os.path.join(p, "cgroup.procs"), "w") as fh:
            fh.write(str(pid))


def _read(p: str) -> str:
    try:
        with open(p) as fh:
            return fh.read()
    except OSError:
        return ""


def cg_stats(p: str | None) -> dict:
    if not p:
        return {}
    out: dict = {}
    pk = _read(os.path.join(p, "memory.peak")).strip()
    if pk.isdigit():
        out["mem_peak_mb"] = round(int(pk) / 2 ** 20, 1)
    for ln in _read(os.path.join(p, "cpu.stat")).splitlines():
        if ln.startswith("usage_usec"):
            out["cpu_s"] = round(int(ln.split()[1]) / 1e6, 2)
    for ln in _read(os.path.join(p, "memory.events")).splitlines():
        k, _, v = ln.partition(" ")
        if k in ("oom", "oom_kill", "max") and v.strip().isdigit() and int(v) > 0:
            out[f"mem_events_{k}"] = int(v)
    return out


def cg_kill_and_remove(p: str | None, pids: list[int]) -> None:
    if p and os.path.exists(os.path.join(p, "cgroup.kill")):
        try:
            with open(os.path.join(p, "cgroup.kill"), "w") as fh:
                fh.write("1")
        except OSError:
            pass
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, TypeError):
            pass
    if not p:
        return
    for _ in range(100):
        if not _read(os.path.join(p, "cgroup.procs")).strip():
            break
        time.sleep(0.05)
    try:
        os.rmdir(p)
    except OSError as e:
        log("cgroup_rmdir_failed", cg=p, error=str(e))


def make_priv(name: str, kind: str) -> str:
    priv = os.path.join(STATE, "priv", name)
    shutil.rmtree(priv, ignore_errors=True)
    os.makedirs(os.path.join(priv, "tmp"))
    os.makedirs(os.path.join(priv, "vartmp"))
    os.chmod(os.path.join(priv, "tmp"), 0o1777)
    os.chmod(os.path.join(priv, "vartmp"), 0o1777)
    src = SOLVER_PROFILE if kind == "solver" else VERIFIER_PROFILE
    if os.path.isdir(src):
        shutil.copytree(src, os.path.join(priv, "lo"), symlinks=True)
    else:
        os.makedirs(os.path.join(priv, "lo"))
    return priv


def _under(path: str, d: str) -> bool:
    """`path` is `d` or lies below it (absolute, normalised paths)."""
    d = d.rstrip("/")
    return not d or path == d or path.startswith(d + "/")


def _hide_index(path: str, hide: tuple) -> int | None:
    """Index of the first HIDE dir that `path` lies strictly below, else None."""
    for i, d in enumerate(hide):
        if path != d.rstrip("/") and _under(path, d):
            return i
    return None


def skill_exposures(mounts: list, priv: str, hide: tuple | None = None) -> list[tuple[str, str]]:
    """[(target, staging)]: the leg's skills dir, to be bound back READ-ONLY at its own path.

    WHY. The SSB runner's task workspace (setup_task_workspace)
    makes <workspace>/.claude/skills a symlink to the leg's shared skills dir, an ABSOLUTE path under
    /Users (the Mac mirror the leg bundle unpacks to). HIDE's tmpfs over /Users would leave that link
    DANGLING inside the episode: SKILL.md text would still reach the model through the skill tool (the
    runner reads it, outside the namespace), but no support file a skill ships (scripts, references)
    could be opened from the solver's shell.

    WHAT MAY COME BACK: per bind source, at most the one directory its .claude/skills link names, and
    only when ALL of these hold (else nothing; a skipped link to a non-empty dir is logged):
      * the link text is absolute and canonical (readlink == realpath), so no intermediate symlink
        can re-route the path the episode walks;
      * the target is a directory strictly below a HIDE dir that is hidden BEFORE the one holding
        `priv` (the staging bind lives in priv and is hidden again with it); a target outside HIDE
        needs nothing -- the link resolves already;
      * the target IS the leg's declared $EVOSKILL_SKILLS_DIR. A base leg declares none (its link
        names the runner's own empty per-tag dir), so a base leg's episode is byte-for-byte the
        pre-fix one -- the held-out baseline's condition;
      * the target holds no bind source, no HIDE dir and not `priv`.
    staging = <priv>/skills/<n>.
    """
    hide = HIDE if hide is None else hide
    declared = os.environ.get(SKILLS_DECL) or ""
    declared = os.path.realpath(declared) if declared else ""
    p_idx = _hide_index(priv, hide)
    others = [s for s, _, _ in mounts] + list(hide) + [priv]
    out: list[tuple[str, str]] = []
    for src, _dst, _ro in mounts:
        link = os.path.join(src, SKILLS_LINK)
        if not os.path.islink(link):
            continue
        text, target = os.readlink(link), os.path.realpath(link)
        t_idx = _hide_index(target, hide)
        if t_idx is None:
            continue
        why = None
        if text != target:
            why = f"link text {text!r} is not its canonical target {target!r}"
        elif not os.path.isdir(target):
            why = "target is not a directory"
        elif target != declared:
            why = f"target is not the declared ${SKILLS_DECL} ({declared or 'unset'})"
        elif p_idx is None or t_idx >= p_idx:
            why = f"no HIDE dir after the target's hides the staging dir under {priv}"
        elif any(_under(x, target) for x in others):
            why = "target contains a bind source, a HIDE dir or the shim's private dir"
        if why:
            try:
                has = bool(os.listdir(target))
            except OSError:
                has = False
            if has:
                log("skills_not_exposed", link=link, target=target, why=why)
            continue
        if target not in [t for t, _ in out]:
            out.append((target, os.path.join(priv, "skills", str(len(out)))))
    return out


def inner_script(mounts: list, priv: str, workdir: str, env: list[str], cmd: list[str],
                 ready_file: str | None, expose: list | tuple = ()) -> str:
    q = shlex.quote
    for target, stage in expose:
        ti, si = _hide_index(target, HIDE), _hide_index(stage, HIDE)
        if ti is None or si is None or ti >= si:
            raise ValueError(f"cannot expose {target} via {stage}: the target must be hidden "
                             f"before the staging dir is")
    L = ["set -e", "mount --make-rprivate /"]
    for src, dst, ro in mounts:
        L.append(f"mkdir -p {q(dst)}; mount --bind {q(src)} {q(dst)}")
        if ro:
            L.append(f"mount -o remount,bind,ro {q(dst)}")
    L += [f"mount --bind {q(priv)}/tmp /tmp",
          f"mount --bind {q(priv)}/vartmp /var/tmp",
          f"mkdir -p {q(SOLVER_PROFILE)}; mount --bind {q(priv)}/lo {q(SOLVER_PROFILE)}",
          "mount --bind /sys/fs/cgroup /sys/fs/cgroup && mount -o remount,bind,ro /sys/fs/cgroup || true",
          "mount --bind /proc/sys /proc/sys && mount -o remount,bind,ro /proc/sys || true",
          LO_UP]
    # skill_exposures: stage the skills dir in priv while it is still visible, ...
    for target, stage in expose:
        L += [f"mkdir -p {q(stage)}; mount --bind {q(target)} {q(stage)}",
              f"mount -o remount,bind,ro {q(stage)}"]
    if ready_file:
        L.append(f"echo ok > {q(ready_file)}")
    for i, d in enumerate(HIDE):
        # ... and bind it back at its own path once its HIDE dir is a tmpfs, just before the tmpfs
        # that hides the staging dir again.
        for target, stage in expose:
            if _hide_index(stage, HIDE) == i:
                L += [f"mkdir -p {q(target)}; mount --bind {q(stage)} {q(target)}",
                      f"mount -o remount,bind,ro {q(target)}"]
        L.append(f"if [ -d {q(d)} ]; then mount -t tmpfs -o size=4m,mode=0755 tmpfs {q(d)}; fi")
    L.append(f"cd {q(workdir)} 2>/dev/null || cd /")
    L.append("exec /usr/bin/env -i " + " ".join(q(e) for e in env) + " "
             + " ".join(q(x) for x in SETPRIV) + " " + " ".join(q(c) for c in cmd))
    return "\n".join(L)


UNSHARE = ["unshare", "--mount", "--net", "--pid", "--ipc", "--uts", "--fork", "--kill-child",
           "--mount-proc"]


def _state_file(name: str) -> str:
    return os.path.join(STATE, "state", name + ".json")


def load_state(name: str) -> dict | None:
    try:
        with open(_state_file(name)) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    st = _read(f"/proc/{pid}/stat")
    return bool(st) and st.split(") ", 1)[-1][:1] not in ("Z", "X")


def child_of(ppid: int) -> int | None:
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        st = _read(f"/proc/{d}/stat")
        if not st:
            continue
        rest = st.rsplit(")", 1)[-1].split()
        if len(rest) > 1 and rest[1] == str(ppid):
            return int(d)
    return None


def _pdeathsig():
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)   # PR_SET_PDEATHSIG
    except OSError:
        pass


def gc_stale(max_age_s: float = 120.0) -> None:
    """Remove leftovers of foreground runs whose shim was SIGKILLed before cleanup."""
    try:
        now = time.time()
        for n in os.listdir(CG_ROOT):
            if not n.startswith("run-"):
                continue
            p = os.path.join(CG_ROOT, n)
            if now - os.stat(p).st_mtime > max_age_s and not _read(os.path.join(p, "cgroup.procs")).strip():
                try:
                    os.rmdir(p)
                    shutil.rmtree(os.path.join(STATE, "priv", n), ignore_errors=True)
                except OSError:
                    pass
    except OSError:
        pass


def record(kind: str, name: str, image: str, t0: float, stats: dict, **kw) -> None:
    try:
        os.makedirs(LOGDIR, exist_ok=True)
        with open(os.path.join(LOGDIR, "containers.jsonl"), "a") as fh:
            fh.write(json.dumps({"kind": kind, "name": name, "image": image,
                                 "wall_s": round(time.time() - t0, 2), **stats, **kw}) + "\n")
    except OSError:
        pass


# ----------------------------------------------------------------------------------- commands
def cmd_images(argv: list[str]) -> int:
    fmt, imgs, i = None, [], 0
    while i < len(argv):
        if argv[i] in ("--format", "-f"):
            fmt = argv[i + 1]
            i += 2
        elif argv[i].startswith("-"):
            i += 1
        else:
            imgs.append(argv[i])
            i += 1
    for im in imgs or []:
        if image_kind(im):
            sys.stdout.write((IMAGE_ID if fmt in (None, "{{.ID}}") else IMAGE_ID) + "\n")
    return 0


def cmd_image(argv: list[str]) -> int:
    if not argv or argv[0] != "inspect":
        raise Usage(f"unsupported: docker image {argv[:1]}")
    fmt, names = parse_inspect(argv[1:])
    rc = 0
    for n in names:
        if not image_kind(n):
            sys.stderr.write(f"Error: No such image: {n}\n")
            rc = 1
            continue
        sys.stdout.write(render_inspect(fmt, {"name": n, "cid": IMAGE_ID, "mounts": [],
                                              "workdir": "/workspace", "image": n}, False))
    return rc


def cmd_run(argv: list[str]) -> int:
    o = parse_run(argv)
    kind = image_kind(o["image"])
    if kind is None:
        sys.stderr.write(f"Unable to find image '{o['image']}' locally (docker-shim)\n")
        return 125
    for src, _, _ in o["volumes"]:
        if not os.path.exists(src):
            os.makedirs(src, exist_ok=True)       # docker creates missing bind sources
    if o["detach"]:
        return run_detached(o, kind)
    return run_foreground(o, kind)


def run_detached(o: dict, kind: str) -> int:
    name = o["name"] or f"evoshim-{uuid.uuid4().hex[:8]}"
    if load_state(name) is not None:
        sys.stderr.write(f"docker: Error response from daemon: Conflict. The container name "
                         f"\"/{name}\" is already in use (docker-shim)\n")
        return 125
    t0 = time.time()
    os.makedirs(os.path.join(STATE, "state"), exist_ok=True)
    cid = hashlib.sha256(f"{name}{t0}{os.getpid()}".encode()).hexdigest()
    priv = make_priv(name, kind)
    cg = cg_create(name)
    ready = os.path.join(STATE, "state", name + ".ready")
    workdir = o["workdir"] or "/workspace"
    env = container_env(name, o["env"])
    expose = skill_exposures(o["volumes"], priv)
    # The episode's PID 1 is a reaping init (docker ran these with --init); the requested CMD
    # (the executor asks for `sleep 86400`) only keeps a container alive, which INIT does too.
    script = inner_script(o["volumes"], priv, workdir, env,
                          ["/usr/local/bin/python3", "-c", INIT], ready, expose)
    # skills_exposed stays OUT of "mounts": the executor compares inspect's mount list to its own.
    st = {"name": name, "cid": cid, "image": o["image"], "kind": kind, "created": t0,
          "mounts": o["volumes"], "workdir": workdir, "env_extra": o["env"], "cg": cg,
          "priv": priv, "requested_cmd": o["cmd"], "labels": o["labels"],
          "skills_exposed": [t for t, _ in expose]}
    with open(_state_file(name), "w") as fh:
        json.dump(st, fh)
    hlog = open(os.path.join(priv, "holder.log"), "ab")

    def _pre():
        os.setsid()
        cg_join(cg, os.getpid())

    p = subprocess.Popen(UNSHARE + ["bash", "-c", script], stdin=subprocess.DEVNULL,
                         stdout=hlog, stderr=hlog, preexec_fn=_pre, close_fds=True)
    deadline = time.time() + 30
    while not os.path.exists(ready):
        if p.poll() is not None or time.time() > deadline:
            err = _read(os.path.join(priv, "holder.log"))[-800:]
            log("create_failed", name=name, rc=p.poll(), err=err)
            cg_kill_and_remove(cg, [p.pid])
            shutil.rmtree(priv, ignore_errors=True)
            try:
                os.unlink(_state_file(name))
            except OSError:
                pass
            sys.stderr.write(f"docker-shim: container {name} failed to start: {err}\n")
            return 125
        time.sleep(0.02)
    init_pid = None
    for _ in range(250):
        init_pid = child_of(p.pid)
        if init_pid:
            break
        time.sleep(0.02)
    if not init_pid:
        cg_kill_and_remove(cg, [p.pid])
        sys.stderr.write(f"docker-shim: container {name}: no init process found\n")
        return 125
    st.update(holder_pid=p.pid, init_pid=init_pid)
    with open(_state_file(name), "w") as fh:
        json.dump(st, fh)
    log("created", name=name, holder=p.pid, init=init_pid, image=o["image"],
        ms=round((time.time() - t0) * 1000), **({"skills": st["skills_exposed"]} if expose else {}))
    sys.stdout.write(cid + "\n")
    return 0


def run_foreground(o: dict, kind: str) -> int:
    gc_stale()
    name = f"run-{uuid.uuid4().hex[:10]}"
    t0 = time.time()
    priv = make_priv(name, kind)
    cg = cg_create(name)
    env = container_env(name, o["env"])
    script = inner_script(o["volumes"], priv, o["workdir"] or "/", env, o["cmd"], None,
                          skill_exposures(o["volumes"], priv))

    def _pre():
        _pdeathsig()             # the scorer's timeout SIGKILLs this shim -> the tree dies too
        cg_join(cg, os.getpid())

    p = subprocess.Popen(UNSHARE + ["bash", "-c", script], preexec_fn=_pre)
    try:
        rc = p.wait()
    except KeyboardInterrupt:
        p.kill()
        rc = p.wait()
    stats = cg_stats(cg)
    cg_kill_and_remove(cg, [])
    shutil.rmtree(priv, ignore_errors=True)
    record("run", name, o["image"], t0, stats, rc=rc,
           cmd=" ".join(o["cmd"])[:160])
    return rc


def cmd_exec(argv: list[str]) -> int:
    o = parse_exec(argv)
    st = load_state(o["name"])
    if st is None or not alive(st.get("init_pid")):
        sys.stderr.write(f"Error response from daemon: No such container: {o['name']}\n"
                         if st is None else
                         f"Error response from daemon: container {o['name']} is not running\n")
        return 1
    env = container_env(st["name"], list(st.get("env_extra", [])) + o["env"])
    wd = o["workdir"] or st.get("workdir") or "/"
    if st.get("cg"):
        try:
            cg_join(st["cg"], os.getpid())
        except OSError as e:
            log("exec_cgroup_join_failed", name=st["name"], error=str(e))
    argv2 = (["nsenter", "-t", str(st["init_pid"]), "-m", "-n", "-p", "-i", "-u", "--",
              "/usr/bin/env", "-i", *env, *SETPRIV,
              "/bin/bash", "-c", 'cd "$1" 2>/dev/null || cd /; shift; exec "$@"', "_", wd]
             + o["cmd"])
    os.execvp("nsenter", argv2)
    return 127  # not reached


def cmd_inspect(argv: list[str]) -> int:
    fmt, names = parse_inspect(argv)
    rc = 0
    for n in names:
        st = load_state(n)
        if st is None:
            if image_kind(n):
                sys.stdout.write(render_inspect(fmt, {"name": n, "cid": IMAGE_ID, "mounts": [],
                                                      "workdir": "/workspace", "image": n}, False))
                continue
            sys.stderr.write(f"Error: No such object: {n}\n")
            rc = 1
            continue
        sys.stdout.write(render_inspect(fmt, st, alive(st.get("init_pid"))))
    return rc


def cmd_rm(argv: list[str]) -> int:
    names = [a for a in argv if not a.startswith("-")]
    rc = 0
    for n in names:
        st = load_state(n)
        if st is None:
            sys.stderr.write(f"Error response from daemon: No such container: {n}\n")
            rc = 1
            continue
        stats = cg_stats(st.get("cg"))
        cg_kill_and_remove(st.get("cg"), [st.get("holder_pid"), st.get("init_pid")])
        shutil.rmtree(st.get("priv") or "/nonexistent", ignore_errors=True)
        for f in (_state_file(n), os.path.join(STATE, "state", n + ".ready")):
            try:
                os.unlink(f)
            except OSError:
                pass
        record("container", n, st.get("image"), st.get("created", time.time()), stats)
        sys.stdout.write(n + "\n")
    return rc


def main(argv: list[str]) -> int:
    if not argv:
        sys.stderr.write("docker-shim: no command\n")
        return 125
    sub, rest = argv[0], argv[1:]
    table = {"images": cmd_images, "image": cmd_image, "run": cmd_run, "exec": cmd_exec,
             "inspect": cmd_inspect, "rm": cmd_rm}
    fn = table.get(sub)
    try:
        if fn is None:
            raise Usage(f"unsupported docker subcommand {sub!r}")
        return fn(rest)
    except Usage as e:
        log("unsupported", argv=argv[:12], error=str(e))
        sys.stderr.write(f"docker-shim: {e}\n")
        return 125


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
