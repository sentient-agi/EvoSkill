#!/usr/bin/env python3
"""Run a command in its own SESSION so a signal aimed at my shell's process group
cannot reach it. macOS has no `setsid(1)`, hence this shim.

WHY THIS EXISTS. `nohup ... & disown` is NOT enough. nohup ignores SIGHUP only, and
disown just drops the job from the shell's table — the child stays in the shell's
PROCESS GROUP. When a tool call in that shell is killed (a timeout, an interrupt), the
signal goes to the group and takes the "background" run with it. Measured 2026-08-06:
a tau3 leg relaunch (pid 67077) died the moment the Bash call that launched it hit its
2-minute cap, because I had appended a polling loop to the same call. It had already
printed its banner and started sample 1.

This is very likely the mechanism behind the recurring "all bg Bash tasks die at once"
symptom, which I had previously filed as environmental and worked around by relaunching.

Usage: detach.py <logfile> <pidfile> <cmd> [args...]
Prints the detached pid on stdout. Also attaches `caffeinate -i -w <pid>` (standing
directive) from INSIDE the new session, so the keep-awake process shares the child's
fate rather than the launching shell's.
"""
import os
import shutil
import subprocess
import sys

log, pidfile, cmd = sys.argv[1], sys.argv[2], sys.argv[3:]
if not cmd:
    sys.exit("usage: detach.py <logfile> <pidfile> <cmd> [args...]")

pid = os.fork()
if pid:
    # Parent: wait only for the intermediate child, which exits immediately.
    os.waitpid(pid, 0)
    with open(pidfile) as f:
        print(f.read().strip())
    raise SystemExit(0)

# Intermediate child: become a session leader, then spawn the real work and exit so
# the work is reparented to init and orphaned from my shell entirely.
os.setsid()
fh = open(log, "ab", buffering=0)
p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=False)
with open(pidfile, "w") as f:
    f.write(str(p.pid) + "\n")
# GUARDED, like the other three call sites. README's platform notes say caffeinate is
# "used when present, skipped on Linux" and loop.py:2021 checks shutil.which -- this one
# did not, and it is the shim EVERY tau3 leg launch goes through. On any host without
# caffeinate the Popen raises FileNotFoundError AFTER the child is already running and the
# pidfile is written, so the launch dies here while the work it detached carries on
# unsupervised and unrecorded.
if shutil.which("caffeinate"):
    subprocess.Popen(["caffeinate", "-i", "-w", str(p.pid)],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     stdin=subprocess.DEVNULL)
os._exit(0)
