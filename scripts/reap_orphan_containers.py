#!/usr/bin/env python3
"""Remove solver containers whose creating process is gone. (#117A)

THE LEAK THIS CLOSES. Sandboxes are created with `docker run -d --rm ... sleep 86400`
and tracked in an in-process set. `--rm` reaps a container when it STOPS; the set is
cleaned in a `finally`. A SIGKILL of the runner satisfies neither: the container is
still RUNNING its sleep, so `--rm` never fires, and the set died with the process.
Measured 2026-07-30: one host reached 983 running / 2828 total containers with 9.2 GB
of 13.9 GB VM memory consumed. Background tasks in this project die in batches, so
this is not a rare path.

WHY IT WAS NOT FIXABLE BEFORE. The only handle was the name prefix
`evoskill-sandbox-…`, which cannot distinguish an orphan from a container a LIVE run
is using — and killing the latter corrupts a measured rollout. Containers now carry
`evoskill.owner_pid`, so the question becomes decidable: is that PID alive? Same
liveness test fleet.py uses for budget shares and seats.py for rollout seats.

SAFETY, in order of importance:
  1. Never touch a container whose owner PID is ALIVE. A false positive corrupts a
     rollout, which is far worse than leaving a container running.
  2. Never touch an unlabelled container. Anything without `evoskill.owner_pid`
     predates this change or belongs to something else; it is not ours to judge.
  3. Dry-run by default. `--apply` is required to remove anything.
  4. A minimum age (`--min-age-s`) guards the startup race: a container created
     microseconds ago by a process still forking could otherwise look ownerless.

    python3 scripts/reap_orphan_containers.py             # report only
    python3 scripts/reap_orphan_containers.py --apply     # actually remove
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time


def _alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError, TypeError):
        return False
    return True


def containers() -> list[dict]:
    """Every running container, with its evoskill labels parsed out."""
    r = subprocess.run(
        ["docker", "ps", "--format", "{{.ID}}\t{{.Names}}\t{{.Labels}}\t{{.RunningFor}}"],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"docker ps failed: {r.stderr.strip()}")
    out = []
    for line in r.stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        cid, name, labels = parts[0], parts[1], (parts[2] if len(parts) > 2 else "")
        lab = {}
        for kv in labels.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                lab[k.strip()] = v.strip()
        out.append({"id": cid, "name": name, "labels": lab,
                    "running_for": parts[3] if len(parts) > 3 else "?"})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="actually remove; default is report-only")
    ap.add_argument("--min-age-s", type=float, default=120.0,
                    help="ignore containers younger than this (startup race guard)")
    # SECOND, INDEPENDENT REAP CRITERION — safe even when the owner is ALIVE.
    #
    # The original reaper can only remove containers whose owner PID is dead, because
    # `evoskill.owner_pid` is the only ownership label and a live PID cannot tell a
    # finished rollout's container from an in-flight one. That leaves the WITHIN-PROCESS
    # accumulation measured on 2026-08-03 (a leg holding 21 seats owned 36 containers)
    # entirely uncovered: those containers belong to a live process by construction.
    #
    # But a rollout CANNOT outlive the solver wall. Verified 2026-08-03 that scoring
    # does not extend a sandbox's life: `_recalculate_with_libreoffice` runs
    # `docker run --rm ... ssb-verifier:latest` — a DIFFERENT image, its own ephemeral
    # container, one blocking command — so the sandbox is not reused for recalc, and its
    # lifetime is bounded by EVOSKILL_SOLVER_TIMEOUT_SECONDS / EVOSKILL_TOTAL_WALL.
    #
    # So a LABELLED sandbox older than the wall provably cannot be serving in-flight
    # work, whatever its owner is doing. Default 0 = disabled, because the bound depends
    # on the wall the run was launched with and guessing it wrong would remove a live
    # rollout's container. Pass the run's actual wall plus a margin.
    ap.add_argument("--max-age-s", type=float, default=0.0,
                    help="also remove LABELLED containers older than this even if the "
                         "owner PID is alive — they cannot be serving an in-flight "
                         "rollout, since a rollout cannot outlive the solver wall. Set "
                         "to the run's wall + margin (e.g. 4200 for a 3600s wall). "
                         "0 disables.")
    a = ap.parse_args()
    if a.max_age_s and a.max_age_s <= a.min_age_s:
        raise SystemExit(f"--max-age-s ({a.max_age_s}) must exceed --min-age-s "
                         f"({a.min_age_s}); otherwise the two guards contradict.")

    now = time.time()
    # FOUR BUCKETS, not three. The first version merged "owner is alive" with "too
    # young to judge" into one counter labelled "owned by a live process" — which was
    # FALSE for the young ones: their owner was dead, they were spared by the age
    # guard. Verified live during the self-test, where a just-created container with a
    # dead owner was reported as live-owned. A spare-reason that misreports itself is
    # how a future reader concludes the guard is unnecessary and removes it.
    orphans, owner_alive, too_young, unlabelled = [], [], [], []
    past_wall = []          # live owner, but older than the wall -> rollout is over
    for c in containers():
        pid = c["labels"].get("evoskill.owner_pid")
        if pid is None:
            unlabelled.append(c)
            continue
        created = c["labels"].get("evoskill.created")
        age = (now - float(created)) if created else None
        if age is not None and age < a.min_age_s:
            too_young.append((c, age))
            continue
        if _alive(int(pid)):
            # Age can only clear a live owner when we KNOW the age. An unparseable
            # `created` label means the wall argument cannot be applied, so such a
            # container stays with its live owner rather than being guessed about.
            if a.max_age_s and age is not None and age > a.max_age_s:
                past_wall.append((c, pid, age))
            else:
                owner_alive.append((c, pid))
        else:
            orphans.append((c, pid, age))

    total = (len(orphans) + len(owner_alive) + len(too_young) + len(unlabelled)
             + len(past_wall))
    print(f"running containers: {total}")
    print(f"  owner pid ALIVE          : {len(owner_alive)}")
    print(f"  too young to judge       : {len(too_young)}"
          f"{'  <- spared by --min-age-s, owner liveness NOT asserted' if too_young else ''}")
    print(f"  unlabelled (not ours)    : {len(unlabelled)}"
          f"{'  <- includes ephemeral ssb-verifier recalc containers' if unlabelled else ''}")
    print(f"  ORPHANED (owner gone)    : {len(orphans)}")
    for c, pid, age in orphans[:15]:
        tag = c["labels"].get("evoskill.run_tag", "?")
        agestr = f"{age/60:.0f}min" if age else "?"
        print(f"    {c['name']}  pid={pid} (dead)  run_tag={tag}  age={agestr}")
    if a.max_age_s:
        print(f"  PAST WALL (live owner)   : {len(past_wall)}"
              f"   [--max-age-s={a.max_age_s:.0f}s]")
        for c, pid, age in past_wall[:15]:
            tag = c["labels"].get("evoskill.run_tag", "?")
            print(f"    {c['name']}  pid={pid} (ALIVE)  run_tag={tag} "
                  f"age={age/60:.0f}min > wall — its rollout cannot still be running")

    reap = orphans + past_wall
    if not reap:
        print("nothing to reap")
        return 0
    if not a.apply:
        print(f"\nDRY RUN — pass --apply to remove these {len(reap)} "
              f"({len(orphans)} orphaned"
              f"{f', {len(past_wall)} past-wall' if past_wall else ''}).")
        return 0

    names = [c["name"] for c, _, _ in reap]
    # Batched: one `docker rm -f` per call forks a CLI process and must stop the
    # container first; under load that is what could not keep up in the first place.
    removed = 0
    for i in range(0, len(names), 20):
        batch = names[i:i + 20]
        r = subprocess.run(["docker", "rm", "-f", *batch],
                           capture_output=True, text=True, timeout=180)
        removed += len([x for x in r.stdout.splitlines() if x.strip()])
        if r.returncode != 0:
            print(f"  batch error: {r.stderr.strip()[:200]}")
    print(f"\nremoved {removed}/{len(names)} orphaned containers")
    print(json.dumps({"reaped": removed, "ts": int(now)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
