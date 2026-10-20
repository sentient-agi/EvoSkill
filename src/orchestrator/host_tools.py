#!/usr/bin/env python3
"""The few host-side helpers whose implementation differs between macOS and Linux, in one place.

Everything that must be Linux runs on the run's machine; the host only drives it. So the host needs one Python
interpreter (the venv the loop runs under) and, optionally, a way to keep the computer from sleeping while a long
child process works. Nothing else here branches on the platform.

  python_executable()   the interpreter every child process runs under: $EVOSKILL_PYTHON when set, else the one
                        running this code. The Daytona SDK is the `daytona` extra of the same venv, so no second
                        interpreter is needed.
  keep_awake(pid)       best effort: hold off idle sleep while `pid` lives. caffeinate on macOS, systemd-inhibit on
                        Linux when it is installed, otherwise nothing. Never raises and never waits; a host that
                        cannot be kept awake runs exactly the same code.

STDLIB ONLY.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

PYTHON_ENV = "EVOSKILL_PYTHON"


def python_executable() -> str:
    """The interpreter for child processes ($EVOSKILL_PYTHON, else sys.executable)."""
    return (os.environ.get(PYTHON_ENV) or "").strip() or sys.executable


def keep_awake_argv(pid: int, platform: str | None = None, which=shutil.which) -> list | None:
    """The command that keeps the host awake while `pid` lives, or None when this host has none."""
    plat = platform or sys.platform
    pid = int(pid)
    if plat == "darwin" and which("caffeinate"):
        return ["caffeinate", "-i", "-w", str(pid)]
    if plat.startswith("linux") and which("systemd-inhibit"):
        # systemd-inhibit holds its lock for as long as the command it runs; this one exits when `pid` does.
        return ["systemd-inhibit", "--what=idle:sleep", "--who=evoskill", "--why=an EvoSkill run is working",
                "--mode=block", "sh", "-c", f"while kill -0 {pid} 2>/dev/null; do sleep 30; done"]
    return None


def keep_awake(pid: int):
    """Start the keep-awake helper for `pid` (detached, output discarded). Returns its Popen, or None."""
    argv = keep_awake_argv(pid)
    if not argv:
        return None
    try:
        return subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, start_new_session=True)
    except OSError:
        return None
