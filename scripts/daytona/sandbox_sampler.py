"""Resource sampler INSIDE a leg sandbox (stdlib only). One JSON line every EVERY seconds:

  mem_gib / cpu_s        the sandbox cgroup (ns root): memory.current, cpu.stat usage
  runner_mem_gib / runner_cpu_s     the runner process's own cgroup (in-process grading lives here)
  containers / containers_mem_gib   live episode + grading cgroups under evoshim/
  cores_busy             cpu-seconds per wall-second since the previous sample (the audit's
                         headroom rule: <= ~6.5 of 8 sustained)

usage: sandbox_sampler.py OUT.jsonl [EVERY_S]
"""
import json
import os
import signal
import sys
import time

CG = "/sys/fs/cgroup"


def rd(p):
    try:
        with open(p) as fh:
            return fh.read()
    except OSError:
        return ""


def cpu_usec(d):
    for ln in rd(os.path.join(d, "cpu.stat")).splitlines():
        if ln.startswith("usage_usec"):
            return int(ln.split()[1])
    return None


def mem(d):
    v = rd(os.path.join(d, "memory.current")).strip()
    return int(v) if v.isdigit() else None


def main():
    out, every = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    stop = {"x": False}
    signal.signal(signal.SIGTERM, lambda *a: stop.update(x=True))
    prev = None
    with open(out, "a") as fh:
        while not stop["x"]:
            t = time.time()
            c = cpu_usec(CG)
            row = {"ts": round(t, 1), "mem_gib": None, "cpu_s": None}
            m = mem(CG)
            if m is not None:
                row["mem_gib"] = round(m / 2 ** 30, 3)
            if c is not None:
                row["cpu_s"] = round(c / 1e6, 1)
                if prev:
                    row["cores_busy"] = round((c - prev[1]) / 1e6 / max(t - prev[0], 1e-3), 2)
                prev = (t, c)
            rm, rc = mem(os.path.join(CG, "evoshim_runner")), cpu_usec(os.path.join(CG, "evoshim_runner"))
            row["runner_mem_gib"] = round(rm / 2 ** 30, 3) if rm is not None else None
            row["runner_cpu_s"] = round(rc / 1e6, 1) if rc is not None else None
            n, tot = 0, 0
            try:
                for name in os.listdir(os.path.join(CG, "evoshim")):
                    p = os.path.join(CG, "evoshim", name)
                    if os.path.isdir(p):
                        n += 1
                        tot += mem(p) or 0
            except OSError:
                pass
            row["containers"] = n
            row["containers_mem_gib"] = round(tot / 2 ** 30, 3)
            row["load1"] = round(os.getloadavg()[0], 2)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            time.sleep(every)


if __name__ == "__main__":
    main()
