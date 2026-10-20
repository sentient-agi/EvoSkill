"""Runner entry point INSIDE a Daytona sandbox: two process-local adjustments, then the unchanged
SSB v2 runner (scripts/run_ssbv2_base30_dsv4f_n3.py) via runpy, as __main__.

Kept out of the repo's source on purpose: scorer.py's sha is a CACHE_ENV axis and the Mac legs
import the same modules, so anything specific to the sandbox lives here and is recorded in the
leg's runtime_env instead.

1. os.cpu_count(). In the sandbox it reports the HOST's CPUs (48-64) while the cgroup grants 8.
   asyncio's default executor -- the only thread pool on the SSB path, used for in-process
   scoring via asyncio.to_thread -- sizes itself min(32, cpu_count+4). Pinned to
   EVOSKILL_RUNNER_CPU_COUNT (default 14 = the Mac host the production legs ran on, pool 18),
   so the scoring pool is what it was there rather than a function of which runner we landed on.

2. At most EVOSKILL_HEAVY_COMPARE_CONCURRENCY (default 2) HEAVY grades at once. A heavy
   workbook's grade holds 5.0-5.8 GB RSS in openpyxl inside THIS process plus a ~1 GB soffice
   child; four at once measured 23.2 GiB alone, so 30 live solvers on top could OOM a
   32 GiB sandbox. "Heavy" = golden or output workbook >=
   EVOSKILL_HEAVY_COMPARE_BYTES (default 5,000,000: only the Debugging/10_x family in SSB v2,
   whose golden is 7.2 MB; the next largest is 1.8 MB). The gate wraps the `score_and_diff` name
   inside ssb_v2_agent (both scoring call sites resolve it there at call time); it only queues,
   never changes a verdict. Each gated grade is logged to /var/log/evoshim/heavy_compare.jsonl.

The gate is installed by a one-shot import hook, NOT by importing ssb_v2_agent here: the runner
sets EVOSKILL_STREAM etc. before its own imports and the executor reads them at import time, so
importing it early would silently change the transport.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import json
import os
import runpy
import sys
import threading
import time

# sandbox_launch.sh starts this file from the repo mirror (the host repo's own absolute path on the machine)
RUNNER = os.path.join(os.getcwd(), "scripts", "run_ssbv2_base30_dsv4f_n3.py")
TARGET = "src.agent_profiles.ssbv2_agent.ssb_v2_agent"
LOG = "/var/log/evoshim/heavy_compare.jsonl"

_CPU = int(os.environ.get("EVOSKILL_RUNNER_CPU_COUNT", "14"))
os.cpu_count = lambda: _CPU  # noqa: E731

_HEAVY_N = int(os.environ.get("EVOSKILL_HEAVY_COMPARE_CONCURRENCY", "2"))
_HEAVY_BYTES = int(os.environ.get("EVOSKILL_HEAVY_COMPARE_BYTES", "5000000"))


def _log(**kw) -> None:
    try:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with open(LOG, "a") as fh:
            fh.write(json.dumps({"ts": round(time.time(), 2), **kw}) + "\n")
    except OSError:
        pass


def _size(p) -> int:
    try:
        return os.path.getsize(p)
    except (OSError, TypeError):
        return 0


def install_gate(module) -> None:
    orig = module.score_and_diff
    sem = threading.BoundedSemaphore(max(1, _HEAVY_N))

    def score_and_diff(golden_path, proc_file, answer_position, *a, **kw):
        size = max(_size(golden_path), _size(proc_file))
        if size < _HEAVY_BYTES:
            return orig(golden_path, proc_file, answer_position, *a, **kw)
        t0 = time.time()
        with sem:
            t1 = time.time()
            try:
                return orig(golden_path, proc_file, answer_position, *a, **kw)
            finally:
                _log(event="heavy_grade", file=str(proc_file)[-120:], bytes=size,
                     waited_s=round(t1 - t0, 2), grade_s=round(time.time() - t1, 2))

    score_and_diff.__wrapped__ = orig
    module.score_and_diff = score_and_diff
    _log(event="gate_installed", concurrency=_HEAVY_N, threshold_bytes=_HEAVY_BYTES)


class _PatchOnImport(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != TARGET:
            return None
        sys.meta_path.remove(self)                      # one-shot
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"sandbox_entry: cannot locate {fullname}")
        real_exec = spec.loader.exec_module

        def exec_module(module, _real=real_exec):
            _real(module)
            install_gate(module)

        spec.loader.exec_module = exec_module
        return spec


if _HEAVY_N > 0:
    sys.meta_path.insert(0, _PatchOnImport())


# 3. LIVE LLM STATS for the driver's ramp / 429 check (EVOSKILL_LLM_STATS_FILE). Read-only view of
# the executor's own audits (_FRESH_RESPONSES, _PROVIDER_AUDIT, _MODEL_AUDIT), plus a count of the
# executor's "API <status> ..." retry warnings. The counting handler re-emits every record to
# stderr exactly as logging's lastResort would (it would otherwise be bypassed once a handler
# exists), so runner.log is unchanged. Attached only once the executor module is loaded -- this
# thread never imports it.
_STATS_FILE = os.environ.get("EVOSKILL_LLM_STATS_FILE", "")


def _stats_loop() -> None:
    import logging
    import re
    from collections import Counter

    class _Counting(logging.Handler):
        def __init__(self):
            super().__init__(logging.WARNING)
            self.status, self.reduced, self.silence = Counter(), 0, 0

        def emit(self, record):
            try:
                msg = record.getMessage()
                m = re.search(r"API (\d{3})", msg)
                if m:
                    self.status[m.group(1)] += 1
                if "Concurrency reduced" in msg:
                    self.reduced += 1
                if "silence timeout" in msg:
                    self.silence += 1
                sys.stderr.write(msg + "\n")
            except Exception:  # noqa: BLE001
                pass

    h = None
    while True:
        time.sleep(15)
        ex = sys.modules.get("src.harness.direct.executor")
        if ex is None:
            continue
        if h is None:
            h = _Counting()
            logging.getLogger("src.harness.direct.executor").addHandler(h)
        try:
            fr = list(getattr(ex, "_FRESH_RESPONSES", []) or [])
            data = {"ts": round(time.time(), 1), "fresh_responses": len(fr),
                    "with_id": sum(1 for g in fr if g), "recent_ids": [g for g in fr[-10:] if g][-5:],
                    "status_warnings": dict(h.status), "concurrency_reduced": h.reduced,
                    "stream_silence": h.silence,
                    "providers": dict(getattr(ex, "_PROVIDER_AUDIT", {}) or {}),
                    "models": dict(getattr(ex, "_MODEL_AUDIT", {}) or {})}
            tmp = _STATS_FILE + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(data, fh)
            os.replace(tmp, _STATS_FILE)
        except Exception:  # noqa: BLE001 — monitoring must never touch the leg
            pass


if _STATS_FILE:
    threading.Thread(target=_stats_loop, daemon=True, name="llm-stats").start()

if __name__ == "__main__":
    sys.argv = [RUNNER]
    runpy.run_path(RUNNER, run_name="__main__")
