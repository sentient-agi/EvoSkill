#!/bin/bash
# Launch one SSB v2 leg shard INSIDE a Daytona sandbox (run as root, detached, by
# scripts/daytona/ssb_leg_daytona.py via the SDK's process.exec with OPENROUTER_API_KEY in the
# ENVIRONMENT of this process only -- it is never written to a file, echoed or put on a command
# line; the runner prints only its last 4 chars).
#
#   bash /opt/leg/sandbox_launch.sh /opt/leg/leg.env REPO_DIR
#
# REPO_DIR is the repo mirror on the machine (the host repo's own absolute path, where the bundle unpacks it).
#
# Writes /opt/leg/out/{runner.log,samples.jsonl,cgroup_setup.txt,rc}. `rc` appears last: its
# presence means the runner has exited (the driver polls for it).
set -uo pipefail
LEGENV=${1:?usage: sandbox_launch.sh LEG_ENV REPO_DIR}
REPO_DIR=${2:?usage: sandbox_launch.sh LEG_ENV REPO_DIR}
LEG=/opt/leg
OUT=$LEG/out
mkdir -p "$OUT" /var/log/evoshim /var/lib/evoshim/state /var/lib/evoshim/priv
rm -f "$OUT/rc"
# shellcheck disable=SC1090
. "$LEGENV"

# ---- cgroups: per-episode memory caps need the memory controller delegated to evoshim/ --------
CG=/sys/fs/cgroup
{
  echo "cgroup.controllers: $(cat $CG/cgroup.controllers 2>&1)"
  if ! echo "+memory +cpu +pids" > $CG/cgroup.subtree_control 2>/dev/null; then
    # no-internal-process rule: park every process of the ns root in a leaf, then retry
    mkdir -p $CG/sandbox_init
    for p in $(cat $CG/cgroup.procs); do echo "$p" > $CG/sandbox_init/cgroup.procs 2>/dev/null; done
    echo "+memory +cpu +pids" > $CG/cgroup.subtree_control 2>&1 || echo "ROOT_SUBTREE_FAILED"
  fi
  mkdir -p $CG/evoshim $CG/evoshim_runner
  echo "+memory +cpu +pids" > $CG/evoshim/cgroup.subtree_control 2>&1 || echo "EVOSHIM_SUBTREE_FAILED"
  echo "root subtree_control: $(cat $CG/cgroup.subtree_control)"
  echo "evoshim subtree_control: $(cat $CG/evoshim/cgroup.subtree_control)"
  echo "memory.max(root): $(cat $CG/memory.max 2>&1)  cpu.max(root): $(cat $CG/cpu.max 2>&1)"
} > "$OUT/cgroup_setup.txt" 2>&1
if ! grep -q memory $CG/evoshim/cgroup.subtree_control 2>/dev/null; then
  if [ "${EVOSHIM_REQUIRE_CGROUP:-1}" = 1 ]; then
    echo "cgroup memory delegation unavailable; refusing (EVOSHIM_REQUIRE_CGROUP=1)" >> "$OUT/runner.log"
    echo 3 > "$OUT/rc"; exit 3
  fi
fi

# ---- resource sampler (no key in its environment) ----------------------------------------------
env -u OPENROUTER_API_KEY nohup /opt/evoskill/venv/bin/python "$LEG/sandbox_sampler.py" \
    "$OUT/samples.jsonl" "${EVOSKILL_SAMPLE_EVERY_S:-5}" > "$OUT/sampler.log" 2>&1 &
SAMPLER=$!

# ---- the runner ---------------------------------------------------------------------------------
echo $$ > $CG/evoshim_runner/cgroup.procs 2>/dev/null || echo "runner cgroup join failed" >> "$OUT/cgroup_setup.txt"
cd "$REPO_DIR" || { echo 3 > "$OUT/rc"; exit 3; }
/opt/evoskill/venv/bin/python -u "$LEG/sandbox_entry.py" > "$OUT/runner.log" 2>&1
RC=$?
kill "$SAMPLER" 2>/dev/null
sleep 1
echo "$RC" > "$OUT/rc"
exit "$RC"
