#!/usr/bin/env python3
"""Run an SSB v2 leg (base or skill) on Daytona x86 sandboxes, one shard per sandbox.

Run with the repo's venv and its `daytona` extra (the Daytona SDK, python-dotenv, httpx).

  ssb_leg_daytona.py probe-cap                       # ONE 16 vCPU / 64 GiB request (refusal creates nothing)
  ssb_leg_daytona.py plan  --run-dir D --n-shards 12 --split F [--key test] [--tasks A,B]
  ssb_leg_daytona.py run   --run-dir D [--only s01,s02] [--concurrency 30] [--max-parallel 12]
                           [--ramp-first 4 --ramp-wait 300] [--spend-cap 60] [--n-samples 3] [--resume]
  aggregate_shards.py --run-dir D          # one leg result from every shard, or a refusal
  ssb_leg_daytona.py smoke --run-dir D --tasks A,B [--n-samples 1] [--spend-cap 3]

WHAT ONE SHARD DOES (run_shard), each step recorded in <run-dir>/<shard>/driver.log:
  1. create an 8 vCPU / 32 GiB / 40 GB sandbox from scripts/daytona/Dockerfile.ssb_leg (built
     server-side, cached by content), named evoskill-base-<shard>-<stamp>, ledgered; READ
     BACK cpu/memory/disk and refuse a mismatch; auto_stop_interval=0, ttl > leg, ephemeral,
     egress restricted to openrouter.ai (domain_allow_list);
  2. identity probe (arch, CPU, LibreOffice, both openpyxl versions, pythons, cgroup quota);
  3. egress check: openrouter.ai reachable, other hosts refused -- the leg refuses otherwise;
  4. upload the repo subset + the shard's dataset files (mirrored at their host paths),
     install the docker shim, self-test it (isolation: /Users hidden, no CAP_SYS_ADMIN, no key in
     the episode env, no network; grading path: golden-vs-golden scores PASS via LibreOffice);
  5. write runtime_env.json (image sha/ref, versions, egress, knobs) for the runner's result JSON;
  6. launch the UNCHANGED runner (scripts/run_ssbv2_base30_dsv4f_n3.py via sandbox_entry.py) with
     the leg settings; the OpenRouter key is passed ONLY as an env var of that one SDK exec call;
  7. every POLL_S: stream new runner.log bytes to the local log, count finished rollouts (cache
     records), read the runner's live LLM stats (fresh calls, 429s) and resource samples; every
     SPEND_EVERY_S: GET /api/v1/key -- hard abort of ALL shards if the usage delta > --spend-cap;
  8. on exit (or abort): sync back result JSON(s), the fresh cache dir, the tasks dir (snapshots +
     DIFF.txt), shim/grading/LLM logs and samples into <run-dir>/<shard>/; secret-scan them;
     look up >=2 generation ids on /api/v1/generation (free) from the Mac;
  9. delete the sandbox BY ID in a finally, verify it is gone, ledger it.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shlex
import signal
import sys
import tarfile
import threading
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import daytona_lib as dl  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def _ssb_data_root() -> Path | None:
    """The SpreadsheetBench v2 data tree (<checkout>/data/spreadsheetbench-v2): $EVOSKILL_SSBV2_DATA, else
    $SSBV2_ROOT/data/spreadsheetbench-v2 (the variables daytona_evolver.data_root() reads); None when neither is set
    (no SSB data on this host: a tau3 run)."""
    if os.environ.get("EVOSKILL_SSBV2_DATA"):
        return Path(os.environ["EVOSKILL_SSBV2_DATA"])
    if os.environ.get("SSBV2_ROOT"):
        return Path(os.environ["SSBV2_ROOT"]) / "data" / "spreadsheetbench-v2"
    return None


DATA = _ssb_data_root()
# Machines get the repo subset and the dataset files mirrored at their HOST paths (archive names without the
# leading "/"); the self-test scripts below name the same paths.
REPO_ARC = str(REPO).lstrip("/")
# (A placeholder when no SSB data is configured: only an SSB machine runs the self-tests that name it.)
SSB_CHECKOUT = str(DATA.parents[1]) if DATA is not None else "/nonexistent-ssbv2-checkout"
DOCKERFILE = HERE / "Dockerfile.ssb_leg"
RUNNER_REL = "scripts/run_ssbv2_base30_dsv4f_n3.py"
# NO DEFAULT SPLIT. A plan must name the split it measures (--split), and the bundle carries THAT file (a fixed
# default would hand the sandbox runner an EVOSKILL_SPLIT_FILE that does not exist there for any other split).
CATEGORIES = ("Debugging", "Financial_Model", "Template")
HEAVY_BYTES = 5_000_000
RESOURCES = {"cpu": 8, "memory": 32, "disk": 40}
ALLOW_DOMAINS = "openrouter.ai"
POLL_S = 30

# The leg's scientific settings. Everything else is a throughput knob, recorded.
LEG_SETTINGS = {
    "EVOSKILL_SOLVER_MODEL": "deepseek/deepseek-v4.1-flash",
    "EVOSKILL_SOLVER_PROVIDER_ORDER": "deepinfra",
    "EVOSKILL_SOLVER_PROVIDER_ONLY": "deepinfra",
    "EVOSKILL_SOLVER_ALLOW_FALLBACKS": "0",
    "EVOSKILL_SOLVER_REQUIRE_PARAMETERS": "1",
    "EVOSKILL_SOLVER_PROVIDER_QUANTIZATIONS": "fp8",
    "EVOSKILL_SOLVER_EXPECT_PROVIDER": "DeepInfra",
    "EVOSKILL_SOLVER_EXPECT_MODEL": "deepseek/deepseek-v4.1-flash-20260910",
    "EVOSKILL_TURN_BUDGET": "45",
    "EVOSKILL_SAMPLE_SEED": "730",
    "EVOSKILL_DOCKER_IMAGE": "ssb-solver-lo:latest",
    "EVOSKILL_SOLVER_TIMEOUT_SECONDS": "10800",       # as scripts/ssbv2_leg.sh and the 0806+ legs
    "EVOSKILL_BASE_LEG": "1",
}
THREAD_ENV = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
              "NUMEXPR_MAX_THREADS": "1", "MAX_CONCURRENCY": "2"}
KNOBS = {
    "EVOSKILL_RECALC_CONCURRENCY": "4",            # scorer default 6 was sized for a 13-CPU VM
    "EVOSKILL_DATATABLE_UNO_CONCURRENCY": "2",
    "EVOSKILL_HEAVY_COMPARE_CONCURRENCY": "2",     # audit: <=2 heavy compares per sandbox
    "EVOSKILL_HEAVY_COMPARE_BYTES": str(HEAVY_BYTES),
    "EVOSKILL_RUNNER_CPU_COUNT": "14",             # the Mac production host's os.cpu_count()
    "EVOSHIM_MEM_MAX": "4G",                       # per-episode (and per-grading-call) memory cap
    "EVOSHIM_REQUIRE_CGROUP": "1",
    "DOCKER_HOST_IP": "127.0.0.1",
    "PYTHONUNBUFFERED": "1",
}


# ------------------------------------------------------------------------------------ utilities
def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_index() -> dict:
    if DATA is None:
        raise SystemExit("no SSB data root: set EVOSKILL_SSBV2_DATA (the run's INFRA data.root) or SSBV2_ROOT")
    idx = {}
    for cat in CATEGORIES:
        for t in json.loads((DATA / cat / "dataset.json").read_text()):
            idx[f"{cat}/{t['id']}"] = {"input": DATA / cat / t["spreadsheet_path"],
                                       "golden": DATA / cat / t["golden_response_path"],
                                       "answer_position": t["answer_position"]}
    return idx


def is_heavy(info: dict) -> bool:
    return max(info["input"].stat().st_size, info["golden"].stat().st_size) >= HEAVY_BYTES


class Log:
    def __init__(self, path: Path, prefix: str = ""):
        self.path, self.prefix = path, prefix
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        line = f"{dl.now()} {self.prefix}{msg}"
        print(line, flush=True)
        with open(self.path, "a") as fh:
            fh.write(line + "\n")


# ------------------------------------------------------------------------------------ spend
class Spend:
    """OpenRouter key usage delta vs the start of this invocation (GET /api/v1/key, /credits)."""

    def __init__(self, key: str, cap: float, log: Log):
        self.key, self.cap, self.log = key, cap, log
        self.start = self._get()
        self.last = dict(self.start)
        self.breached = threading.Event()
        log(f"[SPEND] start key_usage={self.start['key_usage']:.4f} "
            f"limit_remaining={self.start['limit_remaining']} account_balance="
            f"{self.start['balance']} cap_delta=${cap}")

    def _get(self) -> dict:
        import httpx
        h = {"Authorization": f"Bearer {self.key}"}
        k = httpx.get("https://openrouter.ai/api/v1/key", headers=h, timeout=30).json()["data"]
        out = {"ts": dl.now(), "key_usage": float(k.get("usage") or 0.0),
               "limit_remaining": k.get("limit_remaining"), "balance": None}
        try:
            c = httpx.get("https://openrouter.ai/api/v1/credits", headers=h, timeout=30).json()["data"]
            out["balance"] = round(float(c["total_credits"]) - float(c["total_usage"]), 4)
        except Exception:
            pass
        return out

    def check(self, where: str = "") -> float:
        try:
            self.last = self._get()
        except Exception as e:  # a failed check is logged; the next one decides
            self.log(f"[SPEND] check failed ({where}): {type(e).__name__}: {str(e)[:120]}")
            return self.delta()
        d = self.delta()
        self.log(f"[SPEND] {where} delta=${d:.4f} (cap ${self.cap}) limit_remaining="
                 f"{self.last['limit_remaining']} balance={self.last['balance']}")
        if d > self.cap:
            self.log(f"[SPEND] CAP BREACHED: ${d:.4f} > ${self.cap} -- aborting every shard")
            self.breached.set()
        return d

    def delta(self) -> float:
        return round(self.last["key_usage"] - self.start["key_usage"], 6)


def generation_lookup(key: str, gid: str) -> dict:
    import httpx
    last = "no attempt"
    for attempt in range(6):
        try:
            r = httpx.get("https://openrouter.ai/api/v1/generation", params={"id": gid},
                          headers={"Authorization": f"Bearer {key}"}, timeout=30)
        except Exception as e:  # noqa: BLE001 — a lookup failure is recorded, never raised
            last = f"{type(e).__name__}"
            time.sleep(2 + 2 * attempt)
            continue
        last = f"HTTP {r.status_code}"
        if r.status_code == 200:
            d = r.json()["data"]
            return {k: d.get(k) for k in ("id", "model", "provider_name", "total_cost", "latency",
                                          "generation_time", "native_tokens_prompt",
                                          "native_tokens_completion", "native_tokens_reasoning",
                                          "native_tokens_cached", "finish_reason")}
        time.sleep(2 + 2 * attempt)
    return {"id": gid, "error": last}


# ------------------------------------------------------------------------------------ planning
def plan(args) -> dict:
    if not getattr(args, "split", None):
        raise SystemExit("--split is required: the split file this leg measures (the run's declared split; "
                         "there is no default)")
    split_file = Path(args.split).resolve()              # absolute: the sandbox mirrors it at this path
    split = json.loads(split_file.read_text())
    ids = list(split[args.key])
    idx = load_index()
    if args.tasks:
        want = [t.strip() for t in args.tasks.split(",") if t.strip()]
        bad = [t for t in want if t not in ids]
        if bad:
            raise SystemExit(f"--tasks not in split[{args.key}]: {bad}")
        ids = [t for t in ids if t in set(want)]
    heavy = [t for t in ids if is_heavy(idx[t])]
    light = [t for t in ids if t not in set(heavy)]
    n = max(1, min(args.n_shards, len(ids)))
    if heavy and -(-len(heavy) // n) > args.max_heavy:
        raise SystemExit(f"{len(heavy)} heavy tasks need >= {-(-len(heavy) // args.max_heavy)} shards "
                         f"for <= {args.max_heavy} per sandbox; got --n-shards {n}")
    shards: list[list[str]] = [[] for _ in range(n)]
    for i, t in enumerate(heavy):
        shards[i % n].append(t)
    weight = {t: idx[t]["input"].stat().st_size + idx[t]["golden"].stat().st_size for t in ids}
    for t in sorted(light, key=lambda x: -weight[x]):
        j = min(range(n), key=lambda k: (len(shards[k]), sum(weight[x] for x in shards[k])))
        shards[j].append(t)
    order = {t: i for i, t in enumerate(split[args.key])}
    names = [args.shard_name] if (n == 1 and args.shard_name) else [f"s{i + 1:02d}" for i in range(n)]
    p = {"split_file": str(split_file), "split_sha256": sha256_file(split_file),
         "split_status": split.get("status"), "key": args.key, "n_tasks": len(ids),
         "heavy_bytes": HEAVY_BYTES, "max_heavy_per_shard": args.max_heavy,
         "shards": {nm: sorted(s, key=lambda t: order[t]) for nm, s in zip(names, shards)},
         "heavy": heavy, "created": dl.now()}
    rd = Path(args.run_dir)
    rd.mkdir(parents=True, exist_ok=True)
    pf = rd / ("plan_smoke.json" if getattr(args, "cmd", "") == "smoke" else "plan.json")
    pf.write_text(json.dumps(p, indent=1))
    print(json.dumps({nm: {"n": len(s), "heavy": [t for t in s if t in set(heavy)]}
                      for nm, s in p["shards"].items()}, indent=1))
    print(f"plan -> {pf}  split sha256 {p['split_sha256']}")
    return p


# ------------------------------------------------------------------------------------ bundle
def _add(tf: tarfile.TarFile, src: Path, arc: str) -> None:
    tf.add(str(src), arcname=arc, recursive=False)


def build_bundle(shard: str, tasks: list[str], leg_env: str, out: Path,
                 seed_cache: Path | None = None, cache_dir_sandbox: str | None = None,
                 split_file: str | None = None) -> dict:
    """Repo subset + this shard's dataset files, mirrored to their Mac absolute paths, plus the
    sandbox scripts under opt/leg. Returns a manifest (file count, bytes, sha256 of key files).
    `split_file` is the plan's split (EVOSKILL_SPLIT_FILE in leg.env), mirrored at its absolute path."""
    if not split_file:
        raise ValueError("build_bundle needs the plan's split_file: the runner reads EVOSKILL_SPLIT_FILE")
    split_path = Path(split_file).resolve()
    idx = load_index()
    n, nbytes = 0, 0
    with tarfile.open(out, "w:gz") as tf:
        def add_tree(root: Path, arc_root: str):
            nonlocal n, nbytes
            for p in sorted(root.rglob("*")):
                if "__pycache__" in p.parts or p.suffix == ".pyc" or not p.is_file():
                    continue
                _add(tf, p, arc_root + "/" + str(p.relative_to(root)))
                n += 1
                nbytes += p.stat().st_size
        add_tree(REPO / "src", REPO_ARC + "/src")
        _add(tf, REPO / RUNNER_REL, REPO_ARC + "/" + RUNNER_REL)
        _add(tf, split_path, str(split_path).lstrip("/"))
        n += 2
        for cat in CATEGORIES:
            _add(tf, DATA / cat / "dataset.json", str(DATA / cat / "dataset.json").lstrip("/"))
            n += 1
        for t in tasks:
            for k in ("input", "golden"):
                p = idx[t][k]
                _add(tf, p, str(DATA / p.relative_to(DATA)).lstrip("/"))
                n += 1
                nbytes += p.stat().st_size
        seeded = 0
        if seed_cache is not None and cache_dir_sandbox:
            for p in sorted(seed_cache.rglob("*")):
                if p.is_file():
                    _add(tf, p, cache_dir_sandbox.lstrip("/") + "/" + str(p.relative_to(seed_cache)))
                    seeded += 1
        for f in ("sandbox_entry.py", "sandbox_launch.sh", "sandbox_sampler.py", "docker_shim.py"):
            _add(tf, HERE / f, "opt/leg/" + f)
            n += 1
        for name, content in (("opt/leg/leg.env", leg_env),
                              ("opt/leg/subset.json", json.dumps(tasks))):
            b = content.encode()
            ti = tarfile.TarInfo(name)
            ti.size, ti.mtime, ti.mode = len(b), int(time.time()), 0o644
            tf.addfile(ti, io.BytesIO(b))
            n += 1
    return {"files": n, "payload_bytes": nbytes, "bundle_bytes": out.stat().st_size,
            "bundle_sha256": sha256_file(out), "seeded_cache_files": seeded,
            "seeded_from": str(seed_cache) if seed_cache else None,
            "split_file": str(split_path), "split_sha256": sha256_file(split_path)}


def leg_env_text(shard: str, tag: str, cache_dir: str, n_samples: int, concurrency: int,
                 key: str, image_id: str, split_file: str) -> str:
    env = dict(LEG_SETTINGS)
    env.update(KNOBS)
    env.update(THREAD_ENV)
    env.update({
        "EVOSKILL_OR_KEY": "env",
        "EVOSKILL_N_SAMPLES": str(n_samples),
        "EVOSKILL_CONCURRENCY": str(concurrency),
        "EVOSKILL_SPLIT_FILE": split_file,
        "EVOSKILL_SPLIT_KEY": key,
        "EVOSKILL_TASK_SUBSET": "/opt/leg/subset.json",
        "EVOSKILL_RUN_TAG": tag,
        "EVOSKILL_CACHE_DIR": cache_dir,
        "EVOSKILL_RUNTIME_ENV_FILE": "/opt/leg/runtime_env.json",
        "EVOSHIM_IMAGE_ID": image_id,
        "EVOSHIM_EXTRA_ENV": json.dumps(THREAD_ENV, sort_keys=True),
        "EVOSKILL_LLM_STATS_FILE": "/var/log/evoshim/llm_stats.json",
    })
    assert "OPENROUTER_API_KEY" not in env
    return "".join(f"export {k}={shlex.quote(v)}\n" for k, v in sorted(env.items()))


# ------------------------------------------------------------------------------------ sandbox steps
IDENTITY_CMD = r"""
set +e
echo "whoami=$(whoami)"
echo "arch=$(uname -m)"
echo "kernel=$(uname -r)"
echo "cpu_model=$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')"
echo "nproc=$(nproc)"
echo "cgroup_cpu_max=$(cat /sys/fs/cgroup/cpu.max)"
echo "cgroup_memory_max=$(cat /sys/fs/cgroup/memory.max)"
echo "df_root=$(df -h / | tail -1 | awk '{print $2" "$4}')"
echo "libreoffice=$(cat /etc/evoskill_libreoffice_version | head -1)"
echo "solver_python=$(/usr/local/bin/python3 -V 2>&1)"
echo "solver_stack=$(/usr/local/bin/python3 -c 'import openpyxl,pandas,numpy;print(openpyxl.__version__,pandas.__version__,numpy.__version__)' 2>/dev/null)"
echo "runner_python=$(/opt/evoskill/venv/bin/python -V 2>&1)"
echo "runner_openpyxl=$(/opt/evoskill/venv/bin/python -c 'import openpyxl,lxml,openpyxl.xml as x;print(openpyxl.__version__, "lxml", lxml.__version__, "LXML", x.LXML, "DEFUSEDXML", x.DEFUSEDXML)')"
echo "uv_used_at_build=$(cat /opt/evoskill/uv_version)"
echo "runner_freeze_sha256=$(sha256sum /opt/evoskill/runner-freeze.txt | cut -c1-64)"
echo "claude_binaries=$(find / -xdev -name claude -type f 2>/dev/null | wc -l)"
echo "unshare=$(unshare --version | head -1)"
"""

EGRESS_CMD = r"""
/opt/evoskill/venv/bin/python - <<'EOF'
import json, urllib.request, urllib.error, socket
socket.setdefaulttimeout(12)
out = {}
for name, url in (("openrouter", "https://openrouter.ai/api/v1/models"),
                  ("example_com", "https://example.com/"),
                  ("pypi", "https://pypi.org/simple/"),
                  ("github", "https://github.com/"),
                  ("google_ip", "https://8.8.8.8/")):
    try:
        with urllib.request.urlopen(url) as r:
            out[name] = {"ok": True, "status": r.status}
    except urllib.error.HTTPError as e:
        out[name] = {"ok": True, "status": e.code}
    except Exception as e:
        out[name] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:120]}"}
print("EGRESS " + json.dumps(out))
EOF
"""

SELFTEST_CMD = r"""
set +e
export EVOSHIM_IMAGE_ID="$1"
. /opt/leg/leg.env
S=__REPO__/.cache/_shim_selftest_ws
mkdir -p $S/spreadsheets
echo "IMAGES=$(docker images --format '{{.ID}}' ssb-solver-lo:latest)"
echo "PROBE=$(docker run --rm --entrypoint python3 ssb-solver-lo:latest -c "import shutil;print(shutil.which('soffice') or '')")"
docker run -d --rm --init --dns 127.0.0.1 --name selftest --label t=1 -v $S:/workspace -e SEARCH_URL=http://127.0.0.1:8099 -w /workspace ssb-solver-lo:latest sleep 86400 >/dev/null
echo "RUNNING=$(docker inspect -f '{{.State.Running}}' selftest)"
echo "MOUNT=$(docker inspect selftest --format '{{range .Mounts}}{{if eq .Destination "/workspace"}}{{.Source}}{{end}}{{end}}')"
OPENROUTER_API_KEY=should-not-leak docker exec selftest bash -c '
echo "PID1=$(tr "\0" " " < /proc/1/cmdline | cut -c1-40)"
echo "NPROC_VISIBLE=$(ls /proc | grep -c "^[0-9]")"
echo "CWD=$(pwd)"
echo "USERS_ENTRIES=$(ls -A /Users | wc -l)"
echo "OPT_ENTRIES=$(ls -A /opt | wc -l)"
echo "HOME_ENTRIES=$(ls -A /home 2>/dev/null | wc -l)"
echo "SHIMSTATE_ENTRIES=$(ls -A /var/lib/evoshim 2>/dev/null | wc -l)"
test -e __SSB_CHECKOUT__ && echo "DATASET=VISIBLE" || echo "DATASET=HIDDEN"
umount /Users 2>/dev/null && echo "UMOUNT=ALLOWED" || echo "UMOUNT=DENIED"
echo "CAPEFF=$(grep CapEff /proc/self/status | cut -f2)"
echo "KEY_IN_ENV=$(env | grep -c OPENROUTER)"
echo "SEARCH_URL=$SEARCH_URL"
echo "OMP=$OMP_NUM_THREADS MAXCONC=$MAX_CONCURRENCY"
python3 -c "import openpyxl,sys;print(\"SOLVER_PY\", sys.version.split()[0], openpyxl.__version__)"
(exec 3<>/dev/tcp/1.1.1.1/443) 2>/dev/null && echo "NET=OPEN" || echo "NET=NONE"
echo x > /sys/fs/cgroup/evoshim/probe 2>/dev/null && echo "CGROUP_WRITE=ALLOWED" || echo "CGROUP_WRITE=DENIED"
touch /workspace/written_inside && echo "WS_WRITE=OK"
echo "TMP=$(ls -A /tmp | wc -l)"
soffice --version | head -1
'
echo "EXEC_RC=$?"
ls $S/written_inside >/dev/null 2>&1 && echo "WS_VISIBLE_ON_HOST=YES"
docker rm -f selftest >/dev/null
echo "AFTER_RM=$(docker inspect -f '{{.State.Running}}' selftest 2>&1)"
echo "CGROUPS_LEFT=$(find /sys/fs/cgroup/evoshim -mindepth 1 -maxdepth 1 -type d | wc -l)"
tail -2 /var/log/evoshim/containers.jsonl
rm -rf $S
""".replace("__REPO__", str(REPO)).replace("__SSB_CHECKOUT__", SSB_CHECKOUT)

GRADE_SELFTEST = r"""
cd __REPO__ && . /opt/leg/leg.env && export EVOSHIM_IMAGE_ID="$1" && \
/opt/evoskill/venv/bin/python - "$2" <<'EOF'
import json, shutil, sys, tempfile, time, os
sys.path.insert(0, "__REPO__")
from src.agent_profiles.ssbv2_agent.dataset import load_ssbv2, TASK_INDEX
from src.agent_profiles.ssbv2_agent.scorer import score_and_diff
load_ssbv2("__SSB_CHECKOUT__", n_train=80, n_val=60)
tid = sys.argv[1]
info = TASK_INDEX[tid]
d = tempfile.mkdtemp(prefix="gradeself_", dir="__REPO__/.cache")
out = os.path.join(d, "output"); os.makedirs(out)
cand = os.path.join(out, os.path.basename(info["spreadsheet_path"]))
shutil.copy2(info["golden_path"], cand)
t = time.time()
passed, m, n, diff = score_and_diff(info["golden_path"], cand, info["answer_position"],
                                    orig_input_file=info["spreadsheet_path"])
print("GRADE_SELFTEST " + json.dumps({"task": tid, "golden_vs_golden_passed": passed,
      "matched": m, "total": n, "seconds": round(time.time() - t, 1)}))
shutil.rmtree(d, ignore_errors=True)
EOF
""".replace("__REPO__", str(REPO)).replace("__SSB_CHECKOUT__", SSB_CHECKOUT)


def parse_kv(text: str) -> dict:
    out = {}
    for ln in text.splitlines():
        if "=" in ln and not ln.startswith(" "):
            k, v = ln.split("=", 1)
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", k):
                out[k] = v.strip()
    return out


class Abort(Exception):
    pass


class ShardRun:
    def __init__(self, ctx: dict, shard: str, tasks: list[str], concurrency: int):
        self.ctx, self.shard, self.tasks, self.conc = ctx, shard, tasks, concurrency
        self.dir = Path(ctx["run_dir"]) / shard
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = Log(self.dir / "driver.log", prefix=f"[{shard}] ")
        self.tag = f"{ctx['tag_prefix']}_{shard}"
        self.summary: dict = {"shard": shard, "tag": self.tag, "tasks": tasks,
                              "n_rollouts": len(tasks) * ctx["n_samples"], "concurrency": concurrency}
        self.sid = None
        self.sb = None
        self.seed_cache = None
        prev_cache = self.dir / "cache"
        if prev_cache.is_dir() or (self.dir / "summary.json").exists():
            prev = self.dir / f"_prev_{time.strftime('%Y%m%dT%H%M%S')}"
            prev.mkdir()
            for child in list(self.dir.iterdir()):
                if not child.name.startswith("_prev_"):
                    child.rename(prev / child.name)
            if ctx.get("resume") and (prev / "cache").is_dir():
                self.seed_cache = prev / "cache"
        self.log_offset = 0
        self.n429 = 0
        self.stats: dict = {}
        self.done = threading.Event()

    # -- helpers
    def sh(self, cmd: str, timeout: int = 600, env: dict | None = None, check: bool = False):
        rc, out, dt = dl.run(self.sb, cmd, timeout=timeout, env=env)
        if check and rc != 0:
            raise Abort(f"command failed rc={rc}: {cmd[:120]} :: {out[-1500:]}")
        return rc, out

    def save(self):
        (self.dir / "summary.json").write_text(json.dumps(self.summary, indent=1, default=str))

    # -- steps
    def create(self):
        """Create with retries on quota/capacity refusals (Daytona's org-level concurrency limit is
        unknown): up to 10 tries, 60 s apart, abandoned on abort. Anything else fails at once."""
        for attempt in range(10):
            try:
                return self._create_once()
            except Abort as e:
                msg = str(e).lower()
                transient = any(w in msg for w in ("quota", "limit", "exceed", "capacity",
                                                   "insufficient", "no available", "429"))
                if (not transient or attempt == 9 or self.ctx["abort_all"].is_set()
                        or "per sandbox" in msg):
                    raise
                self.log(f"create refused (attempt {attempt + 1}/10), retrying in 60s: {str(e)[:300]}")
                if self.sid:
                    dl.delete_by_id(self.ctx["d"], self.sid, self.tag)
                    self.sid = None
                time.sleep(60)

    def _create_once(self):
        from daytona import CreateSandboxFromImageParams, Image, Resources
        stamp = time.strftime("%H%M%S")
        name = f"{dl.PREFIX}{self.shard}-{stamp}"
        ttl = self.ctx["ttl_minutes"]
        params = CreateSandboxFromImageParams(
            name=name, image=Image.from_dockerfile(str(DOCKERFILE)), resources=Resources(**RESOURCES),
            ephemeral=True, auto_stop_interval=0, ttl_minutes=ttl, public=False,
            domain_allow_list=ALLOW_DOMAINS, labels={"shard": self.shard})
        blog = open(self.dir / "image_build.log", "a")
        self.log(f"creating {name} ({RESOURCES}, ttl {ttl} min, egress {ALLOW_DOMAINS})")
        sb, sid, err, dt = dl.create(self.ctx["d"], params, RESOURCES, tag=self.tag, timeout=3600,
                                     on_logs=lambda c: (blog.write(c + "\n"), blog.flush()))
        self.sid = sid
        self.summary.update(sandbox_id=sid, sandbox_name=name, create_s=dt)
        if err is not None:
            raise Abort(f"create failed after {dt}s: {type(err).__name__}: {str(err)[:600]}")
        self.sb = sb
        rb = dl.readback(sb)
        self.summary["readback"] = rb
        mism = {k: (RESOURCES[k], rb.get(k)) for k in RESOURCES if rb.get(k) != RESOURCES[k]}
        self.log(f"created {sid} in {dt}s readback cpu={rb['cpu']} mem={rb['memory']} "
                 f"disk={rb['disk']} target={rb['target']} snapshot_ref={rb.get('snapshot_ref')}")
        if mism:
            raise Abort(f"resource readback mismatch {mism}")

    def identity(self):
        rc, out = self.sh(IDENTITY_CMD, timeout=300)
        ident = parse_kv(out)
        self.summary["identity"] = ident
        self.log("identity " + json.dumps(ident))
        if ident.get("arch") != "x86_64":
            raise Abort("identity probe failed: " + out[-800:])
        if not ident.get("runner_openpyxl", "").startswith("3.1.5"):
            raise Abort(f"runner openpyxl is not 3.1.5: {ident.get('runner_openpyxl')}")
        if ident.get("claude_binaries") != "0":
            raise Abort(f"claude binary present in image: {ident.get('claude_binaries')}")

    def egress(self):
        rc, out = self.sh(EGRESS_CMD, timeout=180)
        m = re.search(r"^EGRESS (.*)$", out, re.M)
        if not m:
            raise Abort("egress probe failed: " + out[-800:])
        eg = json.loads(m.group(1))
        self.summary["egress"] = {"domain_allow_list": ALLOW_DOMAINS, "probe": eg}
        self.log("egress " + json.dumps(eg))
        others = [k for k in eg if k != "openrouter"]
        if not eg["openrouter"]["ok"]:
            raise Abort("openrouter.ai NOT reachable from the sandbox")
        leaked = [k for k in others if eg[k]["ok"]]
        if leaked:
            raise Abort(f"egress allow list NOT enforced: reachable {leaked}")

    def upload(self, leg_env: str):
        b = self.dir / "bundle.tgz"
        man = build_bundle(self.shard, self.tasks, leg_env, b, seed_cache=self.seed_cache,
                           cache_dir_sandbox=self.ctx["cache_dir_sandbox"],
                           split_file=self.ctx["split_file"])
        if man["split_sha256"] != self.ctx["split_sha256"]:
            raise Abort(f"the split file changed since the plan: {man['split_file']} is sha256 "
                        f"{man['split_sha256'][:12]}, the plan recorded {self.ctx['split_sha256'][:12]}")
        self.summary["bundle"] = man
        t = time.time()
        self.sb.fs.upload_file(str(b), "/tmp/bundle.tgz", timeout=1800)
        self.log(f"uploaded bundle {man['bundle_bytes'] / 1e6:.1f} MB ({man['files']} files) in "
                 f"{time.time() - t:.0f}s")
        b.unlink()
        self.sh("tar xzf /tmp/bundle.tgz -C / && rm /tmp/bundle.tgz && "
                "install -m 0755 /opt/leg/docker_shim.py /usr/local/bin/docker && "
                "mkdir -p /opt/leg/out /var/log/evoshim /var/lib/evoshim/state /var/lib/evoshim/priv && "
                "which docker && sha256sum /usr/local/bin/docker", timeout=600, check=True)

    def cgroups_and_selftest(self, image_id: str):
        # cgroup delegation first (the shim requires it), reusing the launcher's own logic
        cg = ("CG=/sys/fs/cgroup; echo '+memory +cpu +pids' > $CG/cgroup.subtree_control 2>&1 || "
              "{ mkdir -p $CG/sandbox_init; for p in $(cat $CG/cgroup.procs); do echo $p > "
              "$CG/sandbox_init/cgroup.procs 2>/dev/null; done; echo '+memory +cpu +pids' > "
              "$CG/cgroup.subtree_control; }; mkdir -p $CG/evoshim $CG/evoshim_runner; "
              "echo '+memory +cpu +pids' > $CG/evoshim/cgroup.subtree_control; "
              "echo root=$(cat $CG/cgroup.subtree_control) evoshim=$(cat $CG/evoshim/cgroup.subtree_control)")
        rc, out = self.sh(cg, timeout=60)
        self.summary["cgroup_setup"] = out.strip()[-400:]
        self.log("cgroups " + out.strip()[-300:])
        if "memory" not in out.split("evoshim=")[-1]:
            raise Abort("memory controller not delegated to evoshim/ -- per-episode caps impossible")
        rc, out = self.sh("bash -s " + shlex.quote(image_id) + " <<'SELFTEST'\n" + SELFTEST_CMD
                          + "\nSELFTEST", timeout=300)
        st = parse_kv(out)
        self.summary["shim_selftest"] = {"parsed": st, "raw_tail": out[-2500:]}
        self.log("shim selftest " + json.dumps(st))
        need = {"DATASET": "HIDDEN", "UMOUNT": "DENIED", "NET": "NONE",
                "CGROUP_WRITE": "DENIED", "WS_WRITE": "OK", "RUNNING": "true",
                "PROBE": "/usr/bin/soffice", "IMAGES": image_id, "USERS_ENTRIES": "0",
                "OPT_ENTRIES": "0", "HOME_ENTRIES": "0", "SHIMSTATE_ENTRIES": "0",
                "CGROUPS_LEFT": "0", "EXEC_RC": "0", "KEY_IN_ENV": "0"}
        bad = {k: st.get(k) for k, v in need.items() if (st.get(k) or "").strip() != v}
        if bad:
            raise Abort(f"shim self-test failed: {bad}")
        probe_task = self.tasks[0] if not self.ctx.get("grade_probe_task") else self.ctx["grade_probe_task"]
        rc, out = self.sh("bash -s " + shlex.quote(image_id) + " " + shlex.quote(probe_task)
                          + " <<'GRADE'\n" + GRADE_SELFTEST + "\nGRADE", timeout=900)
        m = re.search(r"^GRADE_SELFTEST (.*)$", out, re.M)
        if not m:
            raise Abort("grading self-test crashed: " + out[-1500:])
        g = json.loads(m.group(1))
        self.summary["grade_selftest"] = g
        self.log("grade selftest " + json.dumps(g))
        if re.search(r"recalculation failed|_solve_datatables_with_uno failed", out):
            raise Abort("grading self-test: LibreOffice recalculation FAILED through the shim: "
                        + out[-1500:])
        if not g["golden_vs_golden_passed"]:
            # Not fatal: a golden recalculated by LibreOffice can legitimately differ from its
            # Excel-cached values; the Mac grader has the same property. Recorded for the report.
            self.log(f"NOTE: golden-vs-golden did not PASS for {g['task']} ({g['matched']}/{g['total']})")

    def runtime_env(self, image_id: str):
        ident = self.summary.get("identity", {})
        rt = {
            "platform": "daytona", "comparability_class": "x86_64 Daytona sandbox (NOT the Mac aarch64 docker legs)",
            "sandbox_id": self.sid, "sandbox_name": self.summary.get("sandbox_name"),
            "readback": self.summary.get("readback"), "shard": self.shard, "task_subset": self.tasks,
            "image": {"dockerfile": "scripts/daytona/Dockerfile.ssb_leg",
                      "dockerfile_sha256": self.ctx["dockerfile_sha256"],
                      "daytona_snapshot_ref": (self.summary.get("readback") or {}).get("snapshot_ref"),
                      "image_id_reported_to_runner": image_id},
            "arch": ident.get("arch"), "cpu_model": ident.get("cpu_model"),
            "nproc_reported": ident.get("nproc"), "cgroup_cpu_max": ident.get("cgroup_cpu_max"),
            "cgroup_memory_max": ident.get("cgroup_memory_max"), "libreoffice": ident.get("libreoffice"),
            "grading_openpyxl": ident.get("runner_openpyxl"), "runner_python": ident.get("runner_python"),
            "solver_python": ident.get("solver_python"), "solver_stack": ident.get("solver_stack"),
            "runner_freeze_sha256": ident.get("runner_freeze_sha256"),
            "episode_isolation": "namespaced process tree per episode via scripts/daytona/docker_shim.py "
                                 "(mount/net/pid/ipc/uts ns, docker default caps, no network, "
                                 "/Users,/opt,/home hidden, per-episode cgroup memory.max)",
            "shim_sha256": sha256_file(HERE / "docker_shim.py"),
            "entry_sha256": sha256_file(HERE / "sandbox_entry.py"),
            "runner_sha256": sha256_file(REPO / RUNNER_REL),
            "executor_sha256": sha256_file(REPO / "src/harness/direct/executor.py"),
            "scorer_sha256": sha256_file(REPO / "src/agent_profiles/ssbv2_agent/scorer.py"),
            "solver_prompt_sha256": sha256_file(REPO / "src/agent_profiles/ssbv2_agent/prompt.md"),
            "split_file": self.ctx["split_file"], "split_sha256": self.ctx["split_sha256"],
            "split_status": self.ctx["split_status"],
            "knobs": KNOBS, "thread_env": THREAD_ENV,
            "egress": self.summary.get("egress"), "grade_selftest": self.summary.get("grade_selftest"),
            "reasoning_param": "none sent (thinking_effort='', enable_thinking=True)",
            "written_utc": dl.now(),
        }
        (self.dir / "runtime_env.json").write_text(json.dumps(rt, indent=1, default=str))
        self.sb.fs.upload_file(json.dumps(rt, default=str).encode(), "/opt/leg/runtime_env.json")
        self.summary["runtime_env_written"] = True

    def launch(self, key: str):
        rc, out = self.sh("(setsid nohup bash /opt/leg/sandbox_launch.sh /opt/leg/leg.env "
                          f"{shlex.quote(str(REPO))} "
                          "> /opt/leg/out/launch.log 2>&1 < /dev/null &) ; sleep 8; "
                          "tail -5 /opt/leg/out/runner.log 2>/dev/null; ls /opt/leg/out",
                          timeout=120, env={"OPENROUTER_API_KEY": key})
        self.summary["launched_utc"] = dl.now()
        self.log("launched; first lines: " + out.strip()[-600:].replace("\n", " | "))

    def poll(self) -> bool:
        """One monitoring tick. Returns True once the runner has exited."""
        cache = shlex.quote(self.ctx["cache_dir_sandbox"])
        rc, out = self.sh(
            f"cat /opt/leg/out/rc 2>/dev/null || echo RUNNING; "
            f"echo SIZE=$(stat -c %s /opt/leg/out/runner.log 2>/dev/null || echo 0); "
            f"echo DONE=$(find {cache} -name '*.json' 2>/dev/null | wc -l); "
            f"echo \"SAMPLE=$(tail -n1 /opt/leg/out/samples.jsonl 2>/dev/null)\"; "
            f"echo \"STATS=$(cat /var/log/evoshim/llm_stats.json 2>/dev/null)\"", timeout=120)
        lines = out.strip().splitlines()
        state = lines[0].strip() if lines else "?"
        kv = {}
        for ln in lines[1:]:
            if "=" in ln:
                k, v = ln.split("=", 1)
                kv[k] = v
        size = int(kv.get("SIZE", "0") or 0)
        if size > self.log_offset:
            rc2, chunk = self.sh(f"tail -c +{self.log_offset + 1} /opt/leg/out/runner.log | head -c 4000000",
                                 timeout=300)
            with open(self.dir / "runner.log", "a") as fh:
                fh.write(chunk)
            self.log_offset += len(chunk.encode())
            self.n429 += len(re.findall(r"API 429", chunk))
        try:
            self.stats = json.loads(kv.get("STATS") or "{}")
        except ValueError:
            pass
        try:
            samp = json.loads(kv.get("SAMPLE") or "{}")
        except ValueError:
            samp = {}
        self.summary["progress"] = {"done": int(kv.get("DONE", "0") or 0),
                                    "of": self.summary["n_rollouts"], "n429_log": self.n429,
                                    "llm": {k: self.stats.get(k) for k in ("fresh_responses", "status_warnings",
                                                                           "providers", "models")},
                                    "sample": samp}
        self.log(f"progress {kv.get('DONE')}/{self.summary['n_rollouts']} rollouts | fresh calls "
                 f"{self.stats.get('fresh_responses')} | 429s {self.n429} | "
                 f"mem {samp.get('mem_gib')} GiB cores {samp.get('cores_busy')} "
                 f"containers {samp.get('containers')} | state {state}")
        if state != "RUNNING":
            self.summary["runner_rc"] = state
            return True
        return False

    def kill_runner(self):
        # '[s]andbox' so pkill cannot match this very command line (it contains the pattern)
        self.sh("pkill -TERM -f '[s]andbox_entry.py'; sleep 10; pkill -KILL -f '[s]andbox_entry.py'; "
                "for c in /var/lib/evoshim/state/*.json; do [ -e \"$c\" ] && docker rm -f "
                "$(basename $c .json); done; echo killed", timeout=120)

    def sync_back(self):
        tag = self.tag
        cache = self.ctx["cache_dir_sandbox"].lstrip("/")
        tasks = f"{REPO_ARC}/.cache/ssbv2_{tag}_tasks"
        tr = [(f"{REPO_ARC}/outputs/ssbv2_base30", "result"), (cache, "cache"),
              (tasks, "tasks"), ("var/log/evoshim", "evoshim"), ("opt/leg/out", "out"),
              ("opt/leg/runtime_env.json", "runtime_env.sandbox.json"), ("opt/leg/leg.env", "leg.env")]
        xf = " ".join(f"--transform {shlex.quote('s,^' + a + ',' + b + ',')}" for a, b in tr)
        xf += " --exclude=.claude --exclude='*.lock'"
        srcs = " ".join(shlex.quote(a) for a, _ in tr)
        rc, out = self.sh(f"cd / && for p in {srcs}; do [ -e \"$p\" ] || echo MISSING $p; done; "
                          f"tar czf /tmp/sync.tgz {xf} $(for p in {srcs}; do [ -e \"$p\" ] && echo $p; done) "
                          f"&& ls -la /tmp/sync.tgz", timeout=900)
        self.log("sync pack: " + out.strip()[-400:].replace("\n", " | "))
        local = self.dir / "sync.tgz"
        self.sb.fs.download_file("/tmp/sync.tgz", str(local))
        skipped = 0
        with tarfile.open(local) as tf:
            for m in tf.getmembers():
                if not (m.isfile() or m.isdir()):
                    skipped += 1            # symlinks/devices: meaningless off the sandbox
                    continue
                tf.extract(m, self.dir, filter="data")
        self.summary["sync_skipped_members"] = skipped
        local.unlink()
        self.summary["synced_utc"] = dl.now()
        self.log(f"synced back into {self.dir}")

    def secret_scan(self, key: str):
        kb = key.encode()
        hits = []
        for p in self.dir.rglob("*"):
            if p.is_file():
                try:
                    if kb in p.read_bytes():
                        hits.append(str(p))
                except OSError:
                    pass
        self.summary["secret_hits"] = hits
        self.log(f"secret scan: {len(hits)} file(s) contain the key")

    def verify_generations(self, key: str):
        res = sorted((self.dir / "result").glob(f"{self.tag}*.json")) if (self.dir / "result").exists() else []
        res = [p for p in res if "generation_ids" not in p.name]
        ids = []
        for p in res:
            try:
                ids = json.loads(p.read_text()).get("generation_ids_sample") or []
                if ids:
                    break
            except ValueError:
                pass
        pick = list(dict.fromkeys([ids[0], ids[len(ids) // 2], ids[-1]])) if ids else []
        looks = [generation_lookup(key, g) for g in pick]
        self.summary["mac_generation_lookups"] = looks
        self.log("generation lookups (Mac): " + json.dumps(looks))

    def cost_from_cache(self):
        tot, n, turns = 0.0, 0, 0
        for p in (self.dir / "cache").rglob("*.json") if (self.dir / "cache").exists() else []:
            try:
                tr = json.loads(p.read_text()).get("trace") or {}
                tot += float(tr.get("total_cost_usd") or 0)
                turns += int(tr.get("num_turns") or 0)
                n += 1
            except (ValueError, OSError):
                pass
        self.summary["cache_records"] = n
        self.summary["cache_cost_usd"] = round(tot, 5)
        self.summary["cache_turns"] = turns

    # -- orchestration
    def run(self, key: str, spend: Spend) -> dict:
        slot = self.ctx["slots"]
        slot.acquire()
        try:
            return self._run(key, spend)
        finally:
            slot.release()

    def _run(self, key: str, spend: Spend) -> dict:
        t0 = time.time()
        try:
            if self.ctx["abort_all"].is_set() or spend.breached.is_set():
                raise Abort("aborted before start")
            self.create()
            self.identity()
            self.egress()
            leg_env = leg_env_text(self.shard, self.tag, self.ctx["cache_dir_sandbox"],
                                   self.ctx["n_samples"], self.conc, self.ctx["key"],
                                   self.ctx["image_id"], self.ctx["split_file"])
            self.upload(leg_env)
            self.cgroups_and_selftest(self.ctx["image_id"])
            self.runtime_env(self.ctx["image_id"])
            spend.check(f"{self.shard} pre-launch")
            if spend.breached.is_set():
                raise Abort("spend cap already breached")
            self.launch(key)
            deadline = t0 + self.ctx["shard_wall_s"]
            while True:
                if self.ctx["abort_all"].is_set() or spend.breached.is_set():
                    self.log("ABORT requested -- killing the runner")
                    self.kill_runner()
                    self.summary["aborted"] = "spend cap" if spend.breached.is_set() else "abort_all"
                    break
                if time.time() > deadline:
                    self.log("shard wall limit reached -- killing the runner")
                    self.kill_runner()
                    self.summary["aborted"] = "shard wall"
                    break
                if self.poll():
                    break
                self.save()
                time.sleep(POLL_S)
            try:
                self.poll()
            except Exception:
                pass
            self.sync_back()
        except Abort as e:
            self.summary["error"] = str(e)
            self.log(f"ABORT: {e}")
            if self.sb is not None:
                try:
                    self.sync_back()
                except Exception as e2:
                    self.log(f"sync after abort failed: {e2}")
        except Exception as e:  # noqa: BLE001
            self.summary["error"] = f"{type(e).__name__}: {e}"
            self.log("CRASH: " + traceback.format_exc()[-2000:])
            if self.sb is not None:
                try:
                    self.sync_back()
                except Exception as e2:
                    self.log(f"sync after crash failed: {e2}")
        finally:
            if self.sid:
                ok = dl.delete_by_id(self.ctx["d"], self.sid, self.tag)
                self.summary["deleted"] = ok
                self.log(f"DELETED {self.sid}: {ok}")
            self.summary["wall_s"] = round(time.time() - t0, 1)
            try:
                self.secret_scan(key)
                self.cost_from_cache()
                self.verify_generations(key)
            except Exception as e:
                self.log(f"post-processing failed: {e}")
            self.save()
            self.done.set()
        return self.summary


# ------------------------------------------------------------------------------------ commands
def cmd_probe_cap(args):
    from daytona import CreateSandboxFromImageParams, Image, Resources
    d = dl.client()
    req = {"cpu": 16, "memory": 64, "disk": 40}
    p = CreateSandboxFromImageParams(name=dl.PREFIX + "cap-16-64-40", image=Image.base("daytonaio/sandbox:0.9.0"),
                                     resources=Resources(**req), ephemeral=True, auto_stop_interval=5,
                                     ttl_minutes=10)
    sb, sid, err, dt = dl.create(d, p, req, tag="cap-probe", timeout=300)
    rec = {"requested": req, "seconds": dt, "id": sid,
           "error": None if err is None else f"{type(err).__name__}: {str(err)[:400]}"}
    if sb is not None:
        rec["readback"] = dl.readback(sb)
    if sid:
        rec["deleted"] = dl.delete_by_id(d, sid, "cap-probe")
    out = Path(args.run_dir) / "cap_probe_16_64.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps(rec, indent=1, default=str))


def _ctx(args, p: dict) -> dict:
    d = dl.client()
    df_sha = sha256_file(DOCKERFILE)
    return {"d": d, "run_dir": args.run_dir, "tag_prefix": args.tag_prefix, "n_samples": args.n_samples,
            "key": p["key"], "split_file": p["split_file"], "split_sha256": p["split_sha256"],
            "split_status": p["split_status"], "heavy": set(p["heavy"]),
            "cache_dir_sandbox": f"{REPO}/.cache/ssbv2_{args.tag_prefix}_cache",
            "dockerfile_sha256": df_sha, "image_id": "x86daytona-" + df_sha[:12],
            "ttl_minutes": args.ttl_minutes, "shard_wall_s": args.shard_wall_min * 60,
            "spend_every_s": args.spend_every_s, "abort_all": threading.Event(),
            "grade_probe_task": args.grade_probe_task,
            "slots": threading.BoundedSemaphore(max(1, getattr(args, "max_parallel", 12) or 12)),
            "resume": bool(getattr(args, "resume", False))}


def cmd_run(args, p: dict | None = None):
    p = p or json.loads((Path(args.run_dir) / "plan.json").read_text())
    shards = p["shards"]
    names = [s.strip() for s in args.only.split(",")] if args.only else list(shards)
    key = dl.openrouter_key()
    runlog = Log(Path(args.run_dir) / "run.log")
    ctx = _ctx(args, p)
    runlog(f"run: shards {names} concurrency {args.concurrency} n_samples {args.n_samples} "
           f"spend cap ${args.spend_cap} image {ctx['image_id']} split sha256 {p['split_sha256']}")
    try:
        before = dl.list_all_ids(ctx["d"])
        (Path(args.run_dir) / f"sandboxes_before_{time.strftime('%H%M%S')}.txt").write_text("\n".join(before) + "\n")
    except Exception as e:
        runlog(f"listing before failed: {e}")
    spend = Spend(key, args.spend_cap, runlog)
    runs = [ShardRun(ctx, n, shards[n], args.concurrency) for n in names]
    threads: list[threading.Thread] = []

    def start(r):
        th = threading.Thread(target=r.run, args=(key, spend), daemon=False, name=r.shard)
        th.start()
        threads.append(th)

    stop = {"sig": False}

    def on_sig(*a):
        runlog("SIGINT/SIGTERM received -- aborting all shards (sandboxes are deleted in finally)")
        stop["sig"] = True
        ctx["abort_all"].set()
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    first = runs[: max(1, args.ramp_first)]
    rest = runs[len(first):]
    for r in first:
        start(r)
        time.sleep(3)
    if rest:
        runlog(f"ramp: {len(first)} shard(s) started; checking the 429 rate after {args.ramp_wait}s "
               f"of LLM traffic")
    t_ramp = None
    last_spend = time.time()
    # ONE supervisor loop: ramp decisions + the spend watchdog (every --spend-every-s), until all
    # shard threads have finished (each deletes its own sandbox in a finally).
    while any(th.is_alive() for th in threads) or (rest and not ctx["abort_all"].is_set()
                                                    and not spend.breached.is_set()):
        time.sleep(30)
        if time.time() - last_spend >= args.spend_every_s:
            spend.check("watchdog")
            last_spend = time.time()
        if not rest or ctx["abort_all"].is_set() or spend.breached.is_set():
            if rest and (ctx["abort_all"].is_set() or spend.breached.is_set()):
                runlog(f"not starting {len(rest)} held shard(s): abort")
                rest = []
            continue
        launched = [r for r in first if r.summary.get("launched_utc")]
        first_over = all(r.done.is_set() for r in first)
        if not launched:
            if first_over:
                runlog("every first-wave shard ended before launching its runner; not starting the rest")
                rest = []
            continue
        if t_ramp is None:
            t_ramp = time.time()
        if time.time() - t_ramp < args.ramp_wait and not first_over:
            continue
        calls = sum(int(r.stats.get("fresh_responses") or 0) for r in first)
        n429 = sum(int((r.stats.get("status_warnings") or {}).get("429", 0)) for r in first)
        rate = n429 / max(1, calls + n429)
        ttft = []
        for r in first:
            for g in (r.stats.get("recent_ids") or [])[-2:]:
                lk = generation_lookup(key, g)
                if lk.get("latency") is not None:
                    ttft.append(lk["latency"])
        runlog(f"ramp check: fresh calls {calls}, 429 warnings {n429} ({rate:.1%}), "
               f"recent /generation latency (ms, time to first token) {ttft}")
        if calls > 0 and rate <= args.max_429_rate:
            runlog(f"ramp OK -- starting the remaining {len(rest)} shard(s)")
            for r in rest:
                start(r)
                time.sleep(3)
            rest = []
        elif first_over:
            # nothing left running to re-measure: a held rest would otherwise wait forever
            runlog(f"first wave over without a passing ramp check (429 rate {rate:.1%}, calls "
                   f"{calls}); NOT starting the remaining {len(rest)} shard(s) -- rerun them with --only")
            rest = []
        else:
            runlog(f"holding the remaining shards (429 rate {rate:.1%}, calls {calls}); "
                   f"re-check in {args.ramp_wait}s")
            t_ramp = time.time()
    spend.check("final")
    summ = {r.shard: r.summary for r in runs}
    total = {"spend_delta_usd": spend.delta(), "spend_start": spend.start, "spend_end": spend.last,
             "shards": {k: {kk: v.get(kk) for kk in ("sandbox_id", "deleted", "runner_rc", "error",
                                                    "cache_records", "cache_cost_usd", "wall_s",
                                                    "secret_hits", "aborted")}
                        for k, v in summ.items()}}
    (Path(args.run_dir) / f"run_summary_{time.strftime('%H%M%S')}.json").write_text(
        json.dumps(total, indent=1, default=str))
    try:
        ours = dl.list_ours(ctx["d"])
        runlog(f"sandboxes with our prefix still present: {ours}")
        total["ours_left"] = ours
    except Exception as e:
        runlog(f"final listing failed: {e}")
    runlog("run done: " + json.dumps(total["shards"], default=str))
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--run-dir", required=True)
        p.add_argument("--split", default=None,
                       help="the split file this leg measures (required by plan/smoke; run reads plan.json)")
        p.add_argument("--key", default="test")
        p.add_argument("--tag-prefix", default="testbase")
        p.add_argument("--n-samples", type=int, default=3)
        p.add_argument("--concurrency", type=int, default=30)
        p.add_argument("--spend-cap", type=float, default=60.0)
        p.add_argument("--spend-every-s", type=int, default=600)
        p.add_argument("--ttl-minutes", type=int, default=300)
        p.add_argument("--shard-wall-min", type=int, default=240)
        p.add_argument("--ramp-first", type=int, default=4)
        p.add_argument("--ramp-wait", type=int, default=300)
        p.add_argument("--max-429-rate", type=float, default=0.02)
        p.add_argument("--grade-probe-task", default=None)

    p = sub.add_parser("probe-cap")
    p.add_argument("--run-dir", required=True)
    p = sub.add_parser("plan")
    common(p)
    p.add_argument("--n-shards", type=int, default=12)
    p.add_argument("--max-heavy", type=int, default=2)
    p.add_argument("--tasks", default=None)
    p.add_argument("--shard-name", default=None)
    p = sub.add_parser("run")
    common(p)
    p.add_argument("--only", default=None)
    p.add_argument("--max-parallel", type=int, default=12,
                   help="at most this many live sandboxes at once (shards queue for a slot)")
    p.add_argument("--resume", action="store_true",
                   help="seed each shard's sandbox cache with <run-dir>/<shard>/cache from an "
                        "earlier attempt (moved to <shard>/_prev_<stamp>/), so finished rollouts replay")
    p = sub.add_parser("smoke")
    common(p)
    p.add_argument("--tasks", required=True)
    args = ap.parse_args()
    if args.cmd == "probe-cap":
        return cmd_probe_cap(args)
    if args.cmd == "plan":
        args.shard_name = getattr(args, "shard_name", None)
        return plan(args)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "smoke":
        args.n_shards, args.max_heavy, args.shard_name, args.only = 1, 2, "smoke", None
        pl = plan(args)
        return cmd_run(args, pl)


if __name__ == "__main__":
    main()
