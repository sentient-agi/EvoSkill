"""Generic per-task workspace lifecycle helpers, shared across every
benchmark whose solver/evolver uses per-task (or per-run) directories under a
common cache root.

This exists because the same bug was independently discovered twice under two
different names: SSB v2's ".cache/ssbv2_tasks/ never wiped between runs" (M19)
and Polyglot's ".cache/polyglot_tasks/<task_name>/ raced by concurrent
scripts" (M20) are the identical failure mode — a shared directory gets
wiped/recreated by one script while another live process's container is still
reading/writing it — rediscovered separately because the locking logic lived
in two near-identical, benchmark-specific task_workspace.py copies instead of
one shared place. Put new per-benchmark workspace/runtime logic here, not in
a new agent_profiles/*_agent/task_workspace.py.
"""

from __future__ import annotations

import os
from pathlib import Path


def acquire_task_lock(path: Path) -> None:
    """Claim `path` for this process; raise if a different live process
    already owns it.

    Works identically at either granularity a caller needs:
      - a whole cache root, claimed once at script startup before wiping
        everything under it (e.g. SSB v2's .cache/ssbv2_tasks/)
      - a single per-task subdirectory, claimed on demand each time that one
        task's directory is about to be wiped and recreated (e.g. Polyglot's
        .cache/polyglot_tasks/<task_name>/)

    Re-entrant for the same process (a script legitimately re-locks its own
    root/dir across iterations or repeated task runs within one run). Stale
    locks from a dead process (checked via os.kill(pid, 0)) are silently
    reclaimed rather than blocking forever.
    """
    lock_path = path.parent / f"{path.name}.lock"
    if lock_path.exists():
        try:
            owner_pid = int(lock_path.read_text().strip())
        except (ValueError, OSError):
            owner_pid = None
        if owner_pid is not None and owner_pid != os.getpid():
            try:
                os.kill(owner_pid, 0)
                alive = True
            except (OSError, ProcessLookupError):
                alive = False
            if alive:
                raise RuntimeError(
                    f"{path} is owned by live process {owner_pid} "
                    f"(lock file: {lock_path}). Running a concurrent script "
                    f"against the same path would race on it — one script's "
                    f"wipe/teardown could delete a directory the other's live "
                    f"container is still using. Wait for it to finish, or "
                    f"remove the lock file if it's stale."
                )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(str(os.getpid()))
