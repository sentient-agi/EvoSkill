"""Claude Code as the solver, in isolated containers on ONE Daytona machine: the pure half (no Daytona SDK, no network,
no credential). scripts/cc_solver/cc_solver.py holds the machine-side classes (Box, Queue) and a standalone CLI;
scripts/daytona/cc_leg_driver.py runs the loop's legs with them.

A rollout UNIT is one sample of one task: unit id <task>#s<k>, run dir and workspace <task>__s<k>. Units run in a
QUEUE of at most `concurrency` containers at once on one machine (limits_for), with --thinking-display summarized: the
evolver reads the solver's thinking, and without the flag the thinking blocks carry no text (measured on Sonnet 5:
117 blocks, 0 chars).

WHAT IS HELD EQUAL TO THE DIRECT SOLVER (the harness-free solver the SSB legs ran before Claude Code):
  * the WORKSPACE: spreadsheets/<input> + an empty output/, the input bytes through the data overlay INFRA.json
    declares (src/orchestrator/data_overlay.py): the upload set is exactly ONE file, the task's input workbook;
  * the TURN BUDGET 45 (--max-turns) and the per-command bash timeout 600 s (BASH_DEFAULT_TIMEOUT_MS);
  * the machine image: the LEG image (Dockerfile.ssb_leg): the container's /usr is the solver environment of the
    direct legs.
WHAT DIFFERS BY DESIGN: the agent (Claude Code, the run's model, default thinking effort: no --effort, no
CLAUDE_CODE_EFFORT_LEVEL), its tools (Claude Code's own, minus CC_DISALLOWED) and the prompt: Claude Code keeps its own
system prompt and the first user message is the task's own text plus its input and output paths (PROMPT_DELIVERY).

--bare IS NOT USED: CC 2.1.281's own help says "--bare ... Anthropic auth is strictly ANTHROPIC_API_KEY or
apiKeyHelper via --settings (OAuth and keychain are never read)", and it sets CLAUDE_CODE_SIMPLE=1, which cuts the
tool set to a 'simple mode' subset (the binary's sI(): `if(a.CLAUDE_CODE_SIMPLE&&!r?.skipSimpleModeFilter)`). A
subscription token cannot run under --bare. What --bare would strip is stripped instead by construction (fresh HOME,
no CLAUDE.md/AGENTS.md/settings/plugins/MCP anywhere in the container) and by the binary's own switches in CC_ENV; the
init event of every run is recorded and checked (init_problems).

THE CREDENTIAL: one per run (src/orchestrator/cc_account.py decides which): a subscription token file, or
ANTHROPIC_API_KEY when no token file is given. It reaches Claude Code only on fd 3, named by CRED_FD_VARS[kind];
never on an argv, in an environment record or in a log. USAGE LIMITS: nothing is read ahead of time. A session that
is REJECTED for a usage limit (rejection_reason) stops admission, and the caller waits for the reset that session
reported (rejection_reset).
"""
from __future__ import annotations

import base64
import datetime as _dt
import hashlib
import io
import json
import os
import random
import re
import shlex
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(REPO), str(REPO / "scripts" / "daytona")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import evolver_container_ctl as ctl  # noqa: E402  (stdlib only: the container's plan, paths, pins)
from src.orchestrator import cc_account as ACC  # noqa: E402  (stdlib only: which credential a run uses)

SCHEMA = "evoskill-cc-solver/1"
SPLIT_KEY = "test"
SEED, N_SAMPLE = 926, 10


def infra_of(run_dir) -> dict:
    """<run_dir>/INFRA.json, parsed (the leg snapshot, LibreOffice build, data overlay, Claude Code pins)."""
    return json.loads((Path(run_dir) / "INFRA.json").read_text())


TOKEN_PREFIX = "sk-ant-" + "oat01-"          # a subscription token; spelled in pieces: no scanner matches this file
API_KEY_PREFIX = "sk-ant-" + "api"           # an Anthropic API key
TRAIN_KEY, TRAIN_SAMPLES = "train", 3
SAMPLE_FILE, SELECTION_FILE = "sample10.json", "selection_train.json"
VAL_SELECTION_FILE = "selection_val.json"      # the val selection: train_selection(key="val")
SPLIT_SETS = ("train", "val", "test", "unused")          # the split's disjoint task sets
THINKING_DISPLAY = "summarized"
TTL_MIN_DEFAULT = 240                        # machine TTL of a queue run; admission stops wall + 20 min before it

MODEL = "claude-sonnet-5"
MAX_TURNS = 45
WALL_MIN = 90
BASH_TIMEOUT_MS = 600_000                    # the direct solver's EVOSKILL_BASH_TIMEOUT=600

WS_CONTAINER = ctl.VM_WORK                   # /evo/work: the container's writable work dir = the task workspace
OUT_CONTAINER = ctl.VM_OUT                   # /evo/out: the stream, stderr, pid, rc
STREAM_NAME, STDERR_NAME = "evolver.log", "evolver.stderr"   # evoctl `tail --name` accepts only these two

# The requested four, plus the tools that reach beyond this one session and container: another Claude session or
# cloud agent of the SAME account (RemoteTrigger, SendMessage, ListAgents), a subagent workflow (Workflow, as
# Task/Agent), or work scheduled for later (Cron*, ScheduleWakeup). Listed in the init event of a default session
# of CC 2.1.281 in this container.
CC_DISALLOWED_REQUESTED = ("WebFetch", "WebSearch", "Task", "Agent")
CC_DISALLOWED_EXTRA = ("RemoteTrigger", "SendMessage", "ListAgents", "Workflow", "CronCreate", "CronDelete",
                       "CronList", "ScheduleWakeup")
CC_DISALLOWED = CC_DISALLOWED_REQUESTED + CC_DISALLOWED_EXTRA
# Every name verified in the strings of CC 2.1.281 (the Mac build of the same version). None changes
# the model, its thinking effort or its default tools; each removes something that is not the task.
CC_ENV = {
    "DISABLE_AUTOUPDATER": "1",                              # claude_pins.PIN_ENV_ALWAYS
    "CLAUDE_CODE_NO_MODEL_FALLBACK": "1",                    # claude_pins: no model substitution
    "CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK": "1",             # claude_pins
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",         # as the evolver (daytona_evolver.vm_env)
    "ENABLE_CLAUDEAI_MCP_SERVERS": "false",                  # no claude.ai connectors (MCP)
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",                  # no auto-memory (memory_paths)
    "CLAUDE_CODE_DISABLE_ORG_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",                   # no CLAUDE.md (none exists in the container anyway)
    "CLAUDE_CODE_DISABLE_BUNDLED_SKILLS": "1",               # no skills
    "CLAUDE_CODE_DISABLE_CLAUDE_API_SKILL": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_CODE_SKILL": "1",
    "CLAUDE_CODE_DISABLE_POLICY_SKILLS": "1",
    # /doctor is the one bundled skill that survives CLAUDE_CODE_DISABLE_BUNDLED_SKILLS (survivesBundledKillSwitch);
    # without this a session lists skills ['doctor']
    "DISABLE_DOCTOR_COMMAND": "1",
    "CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL": "1",   # no plugins
    "BASH_DEFAULT_TIMEOUT_MS": str(BASH_TIMEOUT_MS),         # parity: the direct solver's 600 s per command
}
FORBIDDEN_ENV = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_EFFORT_LEVEL",
                 "CLAUDE_CODE_SIMPLE", "CLAUDE_CODE_OAUTH_TOKEN")
# THE CREDENTIAL'S KIND decides two things inside a container: the variable that tells Claude Code which fd holds the
# credential, and what the session's init event must report as apiKeySource. Claude Code reports a key read from
# CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR as "ANTHROPIC_API_KEY" and a subscription token as "none"; any other source means
# it found a credential we did not hand it (init_problems refuses that). Every variable in FORBIDDEN_ENV is unset in
# the container either way: the credential travels on fd 3 only.
CRED_FD_VARS = {ACC.TOKEN_FILE: "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR",
                ACC.API_KEY: "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR"}
CRED_API_KEY_SOURCES = {ACC.TOKEN_FILE: ("none", None), ACC.API_KEY: ("ANTHROPIC_API_KEY",)}
_CRED = {"kind": ACC.TOKEN_FILE}             # this process's one credential (use_credential_kind)


def use_credential_kind(kind: str) -> None:
    """Record the kind of THIS process's one credential (cc_solver.read_credential calls it): every in-container
    command (cred_fd_var) and the init check (init_problems) follow it."""
    if kind not in CRED_FD_VARS:
        raise ValueError(f"credential kind {kind!r}: one of {sorted(CRED_FD_VARS)}")
    _CRED["kind"] = kind


def credential_kind() -> str:
    return _CRED["kind"]


def cred_fd_var(kind: str | None = None) -> str:
    """The variable naming the credential's fd for Claude Code (fd 3 in a container)."""
    return CRED_FD_VARS[kind or _CRED["kind"]]


GIB = 1 << 30
# cgroup limits per container, sized for 10 concurrent sessions on an 8 vCPU / 32 GiB machine: memory.max 3 GiB each
# (10 x 3 = 30 GiB <= the machine's 32 GiB, so the sum of hard caps can never OOM the machine; memory.high at 85%),
# a hard cap of 2 cores each with equal weights (8 cores shared in proportion when all are busy), 2048 pids,
# 1 GiB /tmp (tmpfs, inside the memory cap), 5 GiB writable disk (the driver stops a session above it).
LIMITS = {"cpus": 2.0, "mem_max_bytes": 3 * GIB, "mem_high_ratio": 0.85, "pids_max": 2048, "cpu_weight": 100,
          "tmp_bytes": 1 * GIB, "rw_max_bytes": 5 * GIB}
MAX_CONCURRENCY = 30
# a container that holds no task file (the skills probe): its 1 GiB is outside the 30 GiB the rollout containers
# share, so every hard cap together stays <= 31 of the machine's 32 GiB
PROBE_LIMITS = {"cpus": 1.0, "mem_max_bytes": 1 * GIB, "mem_high_ratio": 0.85, "pids_max": 1024, "cpu_weight": 100,
                "tmp_bytes": 256 << 20, "rw_max_bytes": 256 << 20}


def limits_for(concurrency: int) -> dict:
    """Per-container limits for `concurrency` rollouts in flight on the 8 vCPU / 32 GiB machine. Up to 10: LIMITS
    (the test probe's sizing). Above: memory.max = 30 GiB / concurrency (floored to 256 MiB; 20 -> 1.5 GiB, the
    probe's measured peak was 278-434 MiB), memory.high 85%, still 2 cores each (the 8 cores are oversubscribed and
    shared by equal weights), 2048 pids, /tmp = min(1 GiB, 2/3 of the cap) inside the cap, and 1 GiB writable (the
    driver stops a session above it: 20 x 1 GiB stays well inside the 40 GB disk)."""
    c = int(concurrency)
    if not 1 <= c <= MAX_CONCURRENCY:
        raise ValueError(f"concurrency {c}: 1..{MAX_CONCURRENCY} on one 8 vCPU / 32 GiB machine")
    if c <= 10:
        return dict(LIMITS)
    mem = (30 * GIB // c) // (256 << 20) * (256 << 20)
    return {"cpus": 2.0, "mem_max_bytes": mem, "mem_high_ratio": 0.85, "pids_max": 2048, "cpu_weight": 100,
            "tmp_bytes": min(1 * GIB, mem * 2 // 3 // 4096 * 4096), "rw_max_bytes": 1 * GIB}


# ───────────────────────────────────────────────────────────────────────────── small helpers
def now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def iso(ts) -> str | None:
    try:
        return _dt.datetime.fromtimestamp(float(ts), _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jdump(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, path)


def safe_id(task_id: str) -> str:
    """The run dir name of a task: task_workspace.task_ws_name(task_id) for sample_idx None."""
    return task_id.replace("/", "_")


UNIT_RE = re.compile(r"^(?P<task>[A-Za-z0-9_]+/[A-Za-z0-9_.-]+)(?:#s(?P<k>[1-9][0-9]{0,2}))?$")


def unit_id(task_id: str, sample_idx: int | None) -> str:
    """A rollout unit: the task itself (one sample, the test probe) or <task>#s<k>."""
    return task_id if sample_idx is None else f"{task_id}#s{int(sample_idx)}"


def parse_unit(unit: str) -> tuple:
    m = UNIT_RE.match(unit or "")
    if not m:
        raise ValueError(f"bad unit id {unit!r} (want <Category>/<id> or <Category>/<id>#s<k>)")
    return m["task"], (int(m["k"]) if m["k"] else None)


def unit_dir_name(unit: str) -> str:
    """The unit's run dir and workspace name = task_workspace.task_ws_name(task, sample_idx) (<task>__s<k>)."""
    t, k = parse_unit(unit)
    return safe_id(t) + ("" if k is None else f"__s{k}")


def run_dir(root: Path, unit: str) -> Path:
    return Path(root) / "runs" / unit_dir_name(unit)


# ───────────────────────────────────────────────────────────────────────────── 1. the sample
def sample_tasks(split_path, *, n: int = N_SAMPLE, seed: int = SEED, key: str = SPLIT_KEY,
                 expect_sha256: str | None = None) -> dict:
    """n task ids drawn uniformly without replacement from split[key] (file order) by random.Random(seed).sample."""
    raw = Path(split_path).read_bytes()
    sha = sha256_bytes(raw)
    if expect_sha256 and sha != expect_sha256:
        raise ValueError(f"{split_path}: sha256 {sha[:12]} != the pinned {expect_sha256[:12]}")
    split = json.loads(raw)
    pool = list(split[key])
    if len(set(pool)) != len(pool):
        raise ValueError(f"split[{key!r}] has duplicate ids")
    dropped = set((split.get("dropped") or {}).get("task_ids") or []) if isinstance(split.get("dropped"), dict) else set()
    ids = random.Random(seed).sample(pool, n)
    bad = [t for t in ids if t in dropped]
    if bad:
        raise ValueError(f"sampled dropped task(s) {bad}")
    return {"schema": SCHEMA + ":sample", "split": str(split_path), "split_sha256": sha, "key": key,
            "pool_size": len(pool), "seed": seed, "n": n, "task_ids": ids,
            "method": f"random.Random({seed}).sample(split[{key!r}] in file order, {n}) -- CPython Mersenne "
                      f"Twister, uniform without replacement",
            "python": sys.version.split()[0], "created_utc": now_utc()}


def _split_sets(split: dict) -> dict:
    sets = {k: list(split.get(k) or []) for k in SPLIT_SETS}
    dropped = split.get("dropped")
    sets["dropped"] = list((dropped or {}).get("task_ids") or []) if isinstance(dropped, dict) else []
    return sets


def train_selection(split_path, *, samples: int = TRAIN_SAMPLES, key: str = TRAIN_KEY,
                    expect_sha256: str | None = None) -> dict:
    """EVERY task of split[key] (file order) x samples 1..N. Units are SAMPLE-major (all s1, then all s2, ...): a
    run stopped early has covered the most tasks. Refused if the key's tasks meet any other set of the split (val /
    test / unused / dropped) or repeat. `expect_sha256` (when given) pins the split file's bytes."""
    raw = Path(split_path).read_bytes()
    sha = sha256_bytes(raw)
    if expect_sha256 and sha != expect_sha256:
        raise ValueError(f"{split_path}: sha256 {sha[:12]} != the pinned {expect_sha256[:12]}")
    sets = _split_sets(json.loads(raw))
    if key not in SPLIT_SETS or key == "unused":
        raise ValueError(f"key {key!r}: one of train / val / test")
    pool = sets[key]
    if not pool or len(set(pool)) != len(pool):
        raise ValueError(f"split[{key!r}] is empty or has duplicate ids")
    meet = {k: sorted(set(pool) & set(v)) for k, v in sets.items() if k != key and set(pool) & set(v)}
    if meet:
        raise ValueError(f"split[{key!r}] shares tasks with {meet}")
    if not 1 <= int(samples) <= 9:
        raise ValueError(f"samples {samples}: 1..9")
    units = [{"unit": unit_id(t, k), "task": t, "sample_idx": k} for k in range(1, int(samples) + 1) for t in pool]
    for u in units:
        parse_unit(u["unit"])
    return {"schema": SCHEMA + ":selection", "kind": f"{key}_all", "split": str(split_path), "split_sha256": sha,
            "key": key, "pool_size": len(pool), "n_tasks": len(pool), "samples": int(samples),
            "n_units": len(units), "task_ids": pool, "units": units,
            "set_sizes": {k: len(v) for k, v in sets.items()},
            "method": f"every task of split[{key!r}] in file order x samples 1..{samples}, sample-major; unit id "
                      f"<task>#s<k>; run dir and workspace <task>__s<k> (task_ws_name(task, k))",
            "python": sys.version.split()[0], "created_utc": now_utc()}


def selection_units(sel: dict) -> list:
    """[{unit, task, sample_idx}] of a selection file; the test probe's sample10.json is one unit per task."""
    if sel.get("units"):
        return [dict(u) for u in sel["units"]]
    return [{"unit": t, "task": t, "sample_idx": None} for t in sel["task_ids"]]


def units_split_problems(units: list, split_path, key: str, expect_sha256: str | None = None) -> list:
    """Re-checked at run and grade time: every unit's task is in split[key] and in no other set of the split."""
    raw = Path(split_path).read_bytes()
    if expect_sha256 and sha256_bytes(raw) != expect_sha256:
        return [f"{split_path}: sha256 {sha256_bytes(raw)[:12]} != the pinned {expect_sha256[:12]}"]
    sets = _split_sets(json.loads(raw))
    P = []
    for u in units:
        t, k = parse_unit(u["unit"])
        if t != u["task"] or k != u.get("sample_idx"):
            P.append(f"{u['unit']}: unit id != task {u['task']!r} / sample {u.get('sample_idx')!r}")
        if t not in sets.get(key, []):
            P.append(f"{u['unit']}: not in split[{key!r}]")
        other = [k2 for k2, v in sets.items() if k2 != key and t in v]
        if other:
            P.append(f"{u['unit']}: also in {other}")
    return P


# ───────────────────────────────────────────────────────────────────────────── 2. the solver's first message
# The solver's prompt holds bare-metal minimal information; adding instructions is the evolver's business, through
# skills. A harness-free solver's system prompt has no place under Claude Code. Claude Code keeps its own system prompt
# (nothing appended), and
# the first user message is the benchmark's own task text plus where its input is and where its output must be saved --
# nothing else (no role line, library list, how-to, grading description or reply format; prompt.md is the
# harness-free solver's and is not sent). PROMPT_DELIVERY is part of the cache key, so no earlier rollout is reused.
PROMPT_DELIVERY = "stock system prompt; the first user message is the task's own text and its input and output paths"
SOLVER_PREAMBLE = ""                         # nothing precedes the task's message (keys the cache bucket)


def task_user_message(instruction: str, spreadsheet_filename: str) -> str:
    """The solver's whole first user message: the task's instruction exactly as the data gives it, then the input
    workbook's path and the path the grader reads the output from."""
    return (f"{instruction}\n\n"
            f"Input spreadsheet: ./spreadsheets/{spreadsheet_filename}\n"
            f"Save your modified file to: ./output/{spreadsheet_filename}\n")


# ───────────────────────────────────────────────────────────────────────────── 2b. the upload set
def upload_rows(ws: Path) -> list:
    """Every regular file of the local workspace, as it will sit in the container (below WS_CONTAINER)."""
    ws = Path(ws)
    rows = []
    for p in sorted(ws.rglob("*")):
        if p.is_symlink():
            raise ValueError(f"{p}: a symlink in the workspace")
        if p.is_file():
            rel = p.relative_to(ws).as_posix()
            rows.append({"rel": rel, "vm_path": f"{WS_CONTAINER}/{rel}", "source": str(p),
                         "sha256": sha256_file(p), "bytes": p.stat().st_size})
    return rows


# every reference workbook of the dataset is named <stem>_golden.<ext> (297/297; inputs <stem>_input.<ext>): a
# plain word like "reference" is a legitimate input name (Debugging/04_06 "Incorrect Cross Sheet References_input")
REFERENCE_NAME_RE = re.compile(r"golden", re.I)


def check_upload_set(rows: list, *, spreadsheet_filename: str, input_sha256: str, golden_shas: dict,
                     golden_paths=()) -> list:
    """Problems (empty = fine): the upload set must be EXACTLY the task's input workbook at spreadsheets/<name>,
    with the bytes the direct solver gets (overlay applied), and no file may be a reference workbook -- by name,
    by source path, or by content (sha of ANY dataset golden)."""
    P = []
    want = f"spreadsheets/{spreadsheet_filename}"
    if [r["rel"] for r in rows] != [want]:
        P.append(f"upload set {[r['rel'] for r in rows]} != [{want!r}]")
    gp = {os.path.normpath(str(g)) for g in golden_paths}
    for r in rows:
        if r["sha256"] in golden_shas:
            P.append(f"{r['rel']}: bytes equal the reference workbook of {golden_shas[r['sha256']]}")
        if os.path.normpath(r["source"]) in gp:
            P.append(f"{r['rel']}: its source is a reference workbook path")
        if REFERENCE_NAME_RE.search(Path(r["rel"]).name):
            P.append(f"{r['rel']}: a reference-like file name")
        if not r["vm_path"].startswith(WS_CONTAINER + "/"):
            P.append(f"{r['rel']}: outside the workspace")
        if r["rel"] == want and r["sha256"] != input_sha256:
            P.append(f"{r['rel']}: sha {r['sha256'][:12]} != the input's {input_sha256[:12]}")
    return P


def build_upload_tgz(rows: list) -> bytes:
    """The container's upload (evoctl extract takes regular files below evo/ only): arc = the in-container path."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", format=tarfile.PAX_FORMAT) as tf:
        for r in rows:
            data = Path(r["source"]).read_bytes()
            if sha256_bytes(data) != r["sha256"]:
                raise ValueError(f"{r['source']} changed since the manifest")
            ti = tarfile.TarInfo(r["vm_path"].lstrip("/"))
            ti.size, ti.mode, ti.mtime = len(data), 0o644, int(time.time())
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = "root"
            tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


def listing_problems(listing: list, rows: list, *, expect_links: dict | None = None, extra_dirs=()) -> list:
    """The in-container listing of /Users, /evo, /etc/claude-code after the upload (daytona_evolver.VM_LISTING_PY
    rows [path, type, sha|target, bytes]) must hold exactly the manifest's files, the four workspace dirs, no link.

    For Claude Code legs: `expect_links` = {link path: target} that must be present EXACTLY
    (type "L", readlink == target; any other link is refused as before); `extra_dirs` = dirs allowed beyond the
    workspace ones (the skill set's dirs). The default call is unchanged."""
    P = []
    files = {p: h for p, t, h, _b in listing if t == "F"}
    links = {p: h for p, t, h, _b in listing if t == "L"}
    dirs = {p for p, t, _h, _b in listing if t == "D"}
    man = {r["vm_path"]: r["sha256"] for r in rows}
    if files != man:
        P.append(f"files {sorted(files)[:4]} != manifest {sorted(man)[:4]}")
    want_links = dict(expect_links or {})
    if links != want_links:
        extra = sorted(set(links) - set(want_links))
        missing = sorted(set(want_links) - set(links))
        wrong = sorted(p for p in set(links) & set(want_links) if links[p] != want_links[p])
        if extra or not want_links:
            P.append(f"symlinks {extra[:3] or sorted(links)[:3]}")
        if missing:
            P.append(f"expected links missing {missing[:3]}")
        if wrong:
            P.append(f"links with another target {[(p, links[p]) for p in wrong][:3]}")
    allowed = {WS_CONTAINER, OUT_CONTAINER, f"{WS_CONTAINER}/output"} | {os.path.dirname(v) for v in man} \
        | {str(d) for d in extra_dirs}
    if dirs - allowed:
        P.append(f"unexpected dirs {sorted(dirs - allowed)[:4]}")
    if f"{WS_CONTAINER}/output" not in dirs:
        P.append("output/ missing")
    return P


# ───────────────────────────────────────────────────────────────────────────── 3. the container
def container_spec(cid: str, *, wall_s: int, limits: dict | None = None) -> dict:
    """evoctl's spec: nothing below /Users (the workspace is /evo/work, evo-owned), egress api.anthropic.com only."""
    from src.orchestrator import daytona_evolver as de
    spec = {"schema": ctl.SCHEMA, "cid": cid, "rw": [], "ro_rebind": [], "readonly_roots": [],
            "limits": dict(LIMITS if limits is None else limits),
            "cc": {"version": ctl.CC_VERSION, "sha256": ctl.CC_SHA256},
            "egress": {"allow": ctl.ALLOW, "must_reach": list(de.MUST_REACH), "must_refuse": list(de.MUST_REFUSE),
                       "max_seconds": int(wall_s) + 3 * 3600, "idle_timeout_s": 1800}}
    probs = ctl.validate_spec(spec)
    if probs:
        raise ValueError(f"container spec: {probs}")
    return spec


def make_cid(unit: str, stamp: str, salt: str) -> str:
    base = re.sub(r"[^a-z0-9_.-]", "-", unit_dir_name(unit).lower())[:40]
    cid = f"cc-{base}-{stamp}-{salt}"[:64]
    if not ctl.CID_RE.match(cid):
        raise ValueError(f"bad container id {cid!r}")
    return cid


def cc_env() -> dict:
    bad = [k for k in CC_ENV if k in FORBIDDEN_ENV or re.search(r"TOKEN|KEY|SECRET", k)]
    if bad:
        raise ValueError(f"CC_ENV carries {bad}")
    return dict(CC_ENV)


def claude_argv(user_message: str, *, model: str = MODEL, max_turns: int = MAX_TURNS) -> list:
    """The solver's argv (the text as given). The prompt is the FIRST positional right after -p, so no variadic
    flag can swallow it; no --append-system-prompt (PROMPT_DELIVERY: Claude Code's own system prompt stays as it
    is); --disallowedTools takes ONE comma-joined value. No --effort, no --bare (module doc).
    --thinking-display summarized: the thinking blocks carry their (summarized) text; it changes what the stream
    SHOWS, not the thinking effort (module doc, TRAIN BASE ROLLOUTS)."""
    return ["claude", *ctl.CC_EXTRA_ARGS, "-p", user_message, "--model", model,
            "--max-turns", str(int(max_turns)), "--output-format", "stream-json", "--verbose",
            "--permission-mode", "bypassPermissions", "--strict-mcp-config",
            "--disallowedTools", ",".join(CC_DISALLOWED), "--thinking-display", THINKING_DISPLAY]


def has_thinking_display(argv: list) -> bool:
    return any(a == "--thinking-display" and argv[i + 1:i + 2] == [THINKING_DISPLAY] for i, a in enumerate(argv))


def _b64_var(name: str, text: str) -> str:
    """`NAME=<text>` in bash, the bytes exact (a command substitution would eat trailing newlines)."""
    b = base64.b64encode(text.encode()).decode()
    return f'{name}="$(printf %s {b} | base64 -d; printf x)"; {name}="${{{name}%x}}"'


def shell_argv(argv: list, texts: dict) -> str:
    """argv as a bash command line; an element equal to a value of `texts` becomes "$NAME" (set by _b64_var)."""
    rev = {v: k for k, v in texts.items()}
    return " ".join(f'"${rev[a]}"' if a in rev else shlex.quote(a) for a in argv)


def env_exports() -> str:
    return "export " + " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(cc_env().items())) + "; unset " + \
        " ".join(FORBIDDEN_ENV)


def launch_inner(user_message: str, *, model: str = MODEL, max_turns: int = MAX_TURNS) -> str:
    """The bash run INSIDE the container as evo (evoctl enter --keep-fd3): claude detached-safe, its stream in
    /evo/out/evolver.log, its pid and exit code in /evo/out/{pid,rc}; the credential only as fd 3 (cred_fd_var)."""
    texts = {"CC_USER_MESSAGE": user_message}
    argv = claude_argv(user_message, model=model, max_turns=max_turns)
    return "\n".join([
        f"exec >{OUT_CONTAINER}/{STREAM_NAME} 2>{OUT_CONTAINER}/{STDERR_NAME} </dev/null",
        env_exports(), f"cd {WS_CONTAINER}",
        _b64_var("CC_USER_MESSAGE", user_message),
        f"date +%s > {OUT_CONTAINER}/t_start",
        f"{cred_fd_var()}=3 {shell_argv(argv, texts)} &",
        f"echo $! > {OUT_CONTAINER}/pid", "wait $!",
        f"echo $? > {OUT_CONTAINER}/rc.tmp && date +%s > {OUT_CONTAINER}/t_end && "
        f"mv {OUT_CONTAINER}/rc.tmp {OUT_CONTAINER}/rc"])


# The toolchain probe, run as evo inside the container before the upload (cwd /tmp): the solver's tools, and the
# system prompt's OWN recalculation recipe on a formula openpyxl wrote.
TOOLCHAIN_PY = r'''
import json, os, shutil, subprocess, sys, tempfile
def run(argv, timeout=180):
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return {"rc": r.returncode, "out": (r.stdout + r.stderr).strip()[-300:]}
    except Exception as e:
        return {"rc": None, "out": ("%s: %s" % (type(e).__name__, e))[:300]}
o = {"id": run(["id"])["out"], "home": os.environ.get("HOME"), "path": os.environ.get("PATH")}
o["which"] = dict((b, shutil.which(b)) for b in ("bash", "python3", "libreoffice", "soffice", "claude", "base64", "timeout"))
o["bash"] = run(["bash", "--version"])["out"].splitlines()[0] if shutil.which("bash") else None
o["python_stack"] = run(["python3", "-W", "ignore", "-c", "import sys,openpyxl,pandas,numpy;print(sys.executable,sys.version.split()[0],openpyxl.__version__,pandas.__version__,numpy.__version__)"])
o["libreoffice"] = run(["libreoffice", "--version"])
o["soffice"] = run(["soffice", "--version"])
o["claude"] = run(["env", "HOME=/tmp/tc_home", "DISABLE_AUTOUPDATER=1", "claude", "--version"], 90)  # evo's HOME untouched
shutil.rmtree("/tmp/tc_home", ignore_errors=True)
d = tempfile.mkdtemp(prefix="tc_", dir="/tmp")
import openpyxl
wb = openpyxl.Workbook(); ws = wb.active; ws["A1"] = 21; ws["A2"] = "=A1*2"; ws["A3"] = '=_xlfn.XLOOKUP(21,A1:A1,A2:A2)'
wb.save(d + "/t.xlsx")
rc = run(["bash", "-c", 'cd %s && F=t.xlsx && mkdir -p /tmp/v_$$ && cp "$F" /tmp/v_$$/ && libreoffice -env:UserInstallation=file:///tmp/lo_$$ --headless --calc --convert-to "xlsx:Calc MS Excel 2007 XML" --outdir %s/out "/tmp/v_$$/$F"; r=$?; rm -rf /tmp/lo_$$ /tmp/v_$$; exit $r' % (d, d)], 300)
vals = None
try:
    w2 = openpyxl.load_workbook(d + "/out/t.xlsx", data_only=True).active
    vals = [w2["A2"].value, w2["A3"].value]
except Exception as e:
    vals = "unreadable: %s" % type(e).__name__
o["recalc_recipe"] = {"rc": rc["rc"], "A2_A3_after": vals, "ok": isinstance(vals, list) and vals[0] == 42}
shutil.rmtree(d, ignore_errors=True)
o["tmp_bytes_free"] = shutil.disk_usage("/tmp").free
o["cwd_listing"] = sorted(os.listdir(os.getcwd()))
print(json.dumps(o))
'''


def toolchain_problems(t: dict) -> list:
    P = []
    w = t.get("which") or {}
    for b in ("bash", "python3", "libreoffice", "soffice", "claude", "base64"):
        if not w.get(b):
            P.append(f"{b} not on PATH")
    if w.get("python3") != "/usr/local/bin/python3":
        P.append(f"python3 is {w.get('python3')!r}, not the solver image's /usr/local/bin/python3")
    stack = str((t.get("python_stack") or {}).get("out") or "").split()
    if stack[2:5] != ["3.1.3", "2.2.0", "1.26.4"]:
        P.append(f"python stack {stack}")
    if not (t.get("recalc_recipe") or {}).get("ok"):
        P.append(f"the system prompt's recalculation recipe failed: {t.get('recalc_recipe')}")
    if not str((t.get("claude") or {}).get("out") or "").startswith(ctl.CC_VERSION):
        P.append(f"claude --version {(t.get('claude') or {}).get('out')!r}")
    if not str(t.get("id") or "").startswith("uid=1000(evo)"):
        P.append(f"id {t.get('id')!r}")
    return P


# ───────────────────────────────────────────────────────────────────────────── 4. the stream
def events(text: str):
    """(events, unparsed line count) of a stream-json log (a torn last line is counted, not fatal)."""
    out, bad = [], 0
    for line in (text or "").splitlines():
        s = line.strip()
        if not s.startswith("{"):
            continue
        try:
            e = json.loads(s)
        except ValueError:
            bad += 1
            continue
        if isinstance(e, dict):
            out.append(e)
    return out, bad


def find_rate_limit_info(o):
    """The rate_limit_info object inside one event, wherever it is nested."""
    if isinstance(o, dict):
        if isinstance(o.get("rate_limit_info"), dict):
            return o["rate_limit_info"]
        for v in o.values():
            r = find_rate_limit_info(v)
            if r:
                return r
    elif isinstance(o, list):
        for v in o:
            r = find_rate_limit_info(v)
            if r:
                return r
    return None


def rl_brief(info: dict | None) -> dict | None:
    if not info:
        return None
    uw = info.get("unifiedWindows") or {}

    def win(k):
        w = uw.get(k) or {}
        return {"utilization": w.get("utilization"), "resetsAt": w.get("resetsAt"), "resetsAt_utc": iso(w.get("resetsAt"))}
    return {"status": info.get("status"), "rateLimitType": info.get("rateLimitType"),
            "utilization": info.get("utilization"),          # the warning window's own figure (present from its threshold up)
            "resetsAt": info.get("resetsAt"), "resetsAt_utc": iso(info.get("resetsAt")),
            "five_hour": win("five_hour"), "seven_day": win("seven_day"),
            "overageStatus": info.get("overageStatus"), "isUsingOverage": info.get("isUsingOverage")}


INIT_KEEP_OUT = ("uuid", "session_id", "messaging_socket_path")


def parse_stream(text: str) -> dict:
    evs, bad = events(text)
    init = result = None
    rl, models, tools = [], Counter(), Counter()
    think = {"blocks": 0, "with_text": 0, "chars": 0, "redacted": 0}
    for e in evs:
        t = e.get("type")
        if t == "system" and e.get("subtype") == "init" and init is None:
            init = {k: v for k, v in e.items() if k not in INIT_KEEP_OUT}
        elif t == "assistant":
            msg = e.get("message") or {}
            if msg.get("model"):
                models[str(msg["model"])] += 1
            for c in msg.get("content") or []:
                if isinstance(c, dict) and c.get("type") == "tool_use":
                    tools[str(c.get("name"))] += 1
                elif isinstance(c, dict) and c.get("type") == "thinking":
                    txt = str(c.get("thinking") or "")
                    think["blocks"] += 1
                    think["chars"] += len(txt)
                    think["with_text"] += bool(txt.strip())
                elif isinstance(c, dict) and c.get("type") == "redacted_thinking":
                    think["redacted"] += 1
        elif t == "result":
            result = e
        info = find_rate_limit_info(e)
        if info:
            rl.append(info)
    res = None
    if result is not None:
        res = {k: result.get(k) for k in ("subtype", "is_error", "num_turns", "duration_ms", "duration_api_ms",
                                          "total_cost_usd", "usage", "modelUsage", "stop_reason", "terminal_reason",
                                          "api_error_status")}
        res["permission_denials_n"] = len(result.get("permission_denials") or [])
        res["result_text_head"] = str(result.get("result") or "")[:300]
    return {"n_events": len(evs), "n_unparsed_lines": bad, "init": init, "result": res,
            "rate_limit": {"n": len(rl), "first": rl_brief(rl[0]) if rl else None,
                           "last": rl_brief(rl[-1]) if rl else None},
            "assistant_models": dict(models), "tool_uses": dict(tools), "thinking": think}


def thinking_problems(thinking: dict | None, argv: list | None = None) -> list:
    """The thinking half of the init check: the solver was given --thinking-display summarized, and the thinking
    blocks it wrote carry text (a session without any thinking block is no evidence either way)."""
    P = []
    if argv is not None and not has_thinking_display(argv):
        P.append(f"argv lacks --thinking-display {THINKING_DISPLAY}")
    th = thinking or {}
    if th.get("blocks") and not th.get("with_text"):
        P.append(f"{th['blocks']} thinking block(s), none carries text (--thinking-display not in effect?)")
    return P


def rejection_reset(brief: dict | None) -> tuple:
    """(reset in epoch seconds | None, window) that a REJECTED session reported: its rate_limit_info's own resetsAt
    (the rejected window's), else the resetsAt of the window it names (rateLimitType), else the five-hour window's.
    (None, None) when the reading carries none (an HTTP 429 that came with no reading)."""
    if not brief:
        return None, None
    kind = brief.get("rateLimitType")
    if isinstance(brief.get("resetsAt"), (int, float)):
        return float(brief["resetsAt"]), kind or "reported"
    for k in ([kind] if kind else []) + ["five_hour"]:
        r = (brief.get(k) or {}).get("resetsAt")
        if isinstance(r, (int, float)):
            return float(r), k
    return None, None


def rejection_reason(brief: dict | None) -> str | None:
    """Why admission must stop after this reading (None = go on): the session was REJECTED for a usage limit (status
    'rejected'). No utilization level and no warning status stops anything; only an actual rejection does. The
    reason starts with "quota " (Queue.stop_kind and the tau3 judge key on it)."""
    if not brief or brief.get("status") != "rejected":
        return None
    until, window = rejection_reset(brief)
    return f"quota rejected ({window or 'window not named'}; resets {iso(until) if until else 'not reported'})"


ANY_SKILLS = "*"          # init_problems: a skill leg's session may list whatever Claude Code loads


def init_problems(init: dict | None, *, model: str | None = MODEL, cwd: str | None = WS_CONTAINER,
                  expect_skills: list | str | None = None) -> list:
    """What must NOT reach the solver, read from the session's own init event.

    `expect_skills` (Claude Code legs): None = any listed skill is a problem (the base rollouts);
    a list = the session must list EXACTLY these names (the skills probe's known skill); [] = an empty list;
    ANY_SKILLS = no check (a loop skill leg: Claude Code's own loader decides what loads; its implementation of skills
    is canonical)."""
    if not init:
        return ["no init event"]
    P = []
    if init.get("mcp_servers"):
        P.append(f"mcp servers {init['mcp_servers']}")
    if expect_skills is None:
        if init.get("skills"):
            P.append(f"skills {init['skills'][:6]}")
    elif expect_skills == ANY_SKILLS:
        pass
    else:
        got, want = sorted(str(s) for s in (init.get("skills") or [])), sorted(str(s) for s in expect_skills)
        if got != want:
            P.append(f"skills {got[:8]} != {want[:8]}")
    if init.get("memory_paths"):
        P.append(f"memory paths {init['memory_paths']}")
    plugins = [p for p in init.get("plugins") or [] if (p or {}).get("path") != "builtin"]
    if plugins:
        P.append(f"plugins {plugins[:4]}")
    tools = set(init.get("tools") or [])
    if tools & set(CC_DISALLOWED):
        P.append(f"disallowed tools present {sorted(tools & set(CC_DISALLOWED))}")
    if "Bash" not in tools:
        P.append("no Bash tool")
    if init.get("apiKeySource") not in CRED_API_KEY_SOURCES[credential_kind()]:
        P.append(f"apiKeySource {init.get('apiKeySource')!r} (not the run's {credential_kind()} credential)")
    if init.get("permissionMode") != "bypassPermissions":
        P.append(f"permissionMode {init.get('permissionMode')!r}")
    if init.get("claude_code_version") != ctl.CC_VERSION:
        P.append(f"claude_code_version {init.get('claude_code_version')!r}")
    if model and not str(init.get("model") or "").startswith(model):
        P.append(f"init model {init.get('model')!r} != {model!r}")
    if cwd and init.get("cwd") != cwd:
        P.append(f"cwd {init.get('cwd')!r} != {cwd!r}")
    return P


# ───────────────────────────────────────────────────────────────────────────── 5-6. outputs, grades, table
def resolve_output(out_dir: Path, filename: str) -> Path | None:
    """ssb_v2_agent._resolve_output_file: the expected name, else the first *.xlsx in sorted order."""
    out_dir = Path(out_dir)
    exp = out_dir / filename
    if exp.is_file() and not exp.is_symlink():
        return exp
    xs = sorted(p for p in out_dir.glob("*.xlsx") if p.is_file() and not p.is_symlink())
    return xs[0] if xs else None


def grade_brief(rec: dict | None) -> dict:
    """One grade result (ssb_grade_daytona results.jsonl line) -> verdict + counts + the scorer's header line ONLY
    (the diff body lists reference values: never printed)."""
    if rec is None:
        return {"verdict": "not_eligible_no_output"}
    st = rec.get("status")
    head = str(rec.get("diff_file_text") or "").splitlines()[:1]
    if st != "ok":
        return {"verdict": f"grader_{st}", "status": st, "outcome": (rec.get("host") or {}).get("outcome")}
    return {"verdict": "pass" if rec.get("passed") else "graded_wrong", "passed": bool(rec.get("passed")),
            "n_match": rec.get("n_match"), "n_total": rec.get("n_total"),
            "outcome": (rec.get("host") or {}).get("outcome"), "scorer_header": head[0] if head else None,
            "warnings_n": len(rec.get("warnings") or [])}


def _render(lines: list) -> str:
    w = [max(len(x[i]) for x in lines) for i in range(len(lines[0]))]
    return "\n".join("  ".join(c.ljust(w[i]) for i, c in enumerate(x)).rstrip() for x in lines)


def table(rows: list) -> str:
    """One line per rollout unit ('unit', else 'task')."""
    hdr = ["unit", "result", "turns", "min", "cost$", "think", "grade", "cells"]
    lines = [hdr]
    for r in rows:
        g = r.get("grade") or {}
        cells = f"{g.get('n_match')}/{g.get('n_total')}" if g.get("n_total") is not None else "-"
        mins = r.get("minutes")
        sub = str(r.get("subtype") or "-")
        if r.get("is_error") and r.get("subtype") != "error_max_turns":
            sub += f" ({r.get('terminal_reason') or 'error'})"      # an api_error is not a 'success'
        th = r.get("thinking") or {}
        lines.append([r.get("unit") or r.get("task", "?"), sub, str(r.get("num_turns") or "-"),
                      f"{mins:.1f}" if isinstance(mins, (int, float)) else "-",
                      f"{r['cost']:.2f}" if isinstance(r.get("cost"), (int, float)) else "-",
                      f"{th.get('with_text', 0)}/{th.get('blocks', 0)}:{th.get('chars', 0)}" if th else "-",
                      str(g.get("verdict") or "-"), cells])
    return _render(lines)


MEASURED = ("pass", "graded_wrong", "not_eligible_no_output")   # a graded outcome (no output = a fail)


def per_task(rows: list) -> list:
    """Per task: samples run, samples with a measured outcome, samples passed, pass@k over the measured ones
    (k = that count; pass@3 when all 3 are measured = at least one sample passed)."""
    by: dict = {}
    for r in rows:
        by.setdefault(r["task"], []).append(r)
    out = []
    for t, rs in by.items():
        v = [((r.get("grade") or {}).get("verdict")) for r in rs]
        meas = [x for x in v if x in MEASURED]
        n_pass = sum(1 for x in meas if x == "pass")
        out.append({"task": t, "units": len(rs), "measured": len(meas), "passed": n_pass,
                    "pass_at_k": (1 if n_pass else 0) if meas else None, "k": len(meas),
                    "verdicts": v})
    return out


def pass_summary(rows: list, samples: int) -> dict:
    """pass@1 = passed / measured rollouts; pass@N over the tasks with all N samples measured."""
    pt = per_task(rows)
    meas = [r for r in rows if (r.get("grade") or {}).get("verdict") in MEASURED]
    n_pass = sum(1 for r in meas if r["grade"]["verdict"] == "pass")
    full = [t for t in pt if t["measured"] == samples]
    return {"rollouts": len(rows), "measured": len(meas), "passed": n_pass,
            "pass_at_1": round(n_pass / len(meas), 4) if meas else None,
            f"tasks_with_{samples}_measured": len(full),
            f"pass_at_{samples}": round(sum(t["pass_at_k"] for t in full) / len(full), 4) if full else None,
            "tasks_any_pass": sum(1 for t in pt if t["passed"]), "tasks": len(pt)}


def task_table(pt: list) -> str:
    lines = [["task", "units", "measured", "passed", "pass@k"]]
    for t in pt:
        lines.append([t["task"], str(t["units"]), str(t["measured"]), str(t["passed"]),
                      "-" if t["pass_at_k"] is None else f"{t['pass_at_k']} (k={t['k']})"])
    return _render(lines)


# ───────────────────────────────────────────────────────────────────────────── 7. loop legs
# Pure helpers of the Claude Code LEG DRIVER (scripts/cc_solver/cc_leg_lib.py + scripts/daytona/cc_leg_driver.py):
# the cache bucket id, and the stream -> direct-harness message conversion that the loop's evidence builder, digest
# builder and outcomes reader consume.
def cc_bucket(cc_version: str, model: str, instruction_sha256: str, skills_tree_sha256: str) -> str:
    """The cache bucket (RunCache's tree_hash slot, 12 hex) of a Claude Code leg: one per (prompt delivery, Claude
    Code version, model, solver instruction text, skill set). The two train/val legs of an epoch share it; a base
    leg's skills_tree_sha256 is sha256(b''). PROMPT_DELIVERY is part of the key so a record of the earlier
    delivery (the instruction appended to the system prompt) can never be served under this one."""
    key = json.dumps(["claude_code", PROMPT_DELIVERY, str(cc_version), str(model), str(instruction_sha256),
                      str(skills_tree_sha256)])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def to_messages(stream_text: str, user_msg: str) -> list:
    """Claude Code stream-json -> the direct harness's OpenAI-format message list (user, assistant with reasoning +
    tool_calls, tool results). No system row: the solver's system prompt is Claude Code's own and is not recorded
    (PROMPT_DELIVERY). Blocks of one API message (same message.id) are merged."""
    msgs = [{"role": "user", "content": user_msg}]
    cur_id = None
    for ln in stream_text.split("\n"):
        if not ln.strip():
            continue
        if not ln.lstrip().startswith("{"):    # a driver marker in a captured stream (CC_PROBE_RC=, STDERR: ...)
            continue                            # is not an event (events() skips the same lines); a torn JSON line still raises
        e = json.loads(ln)
        t = e.get("type")
        m = e.get("message") if isinstance(e.get("message"), dict) else None
        if t == "assistant" and m:
            if m.get("id") != cur_id:
                cur_id = m.get("id")
                msgs.append({"role": "assistant", "content": ""})
            a = msgs[-1]
            for b in m.get("content") or []:
                if b.get("type") == "text":
                    a["content"] = (a["content"] + b.get("text", "")) if a["content"] else b.get("text", "")
                elif b.get("type") == "thinking":
                    a["reasoning"] = a.get("reasoning", "") + (b.get("thinking") or "")
                elif b.get("type") == "tool_use":
                    a.setdefault("tool_calls", []).append({
                        "id": b["id"], "type": "function",
                        "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False)}})
        elif t == "user" and m and isinstance(m.get("content"), list):
            cur_id = None
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    c = b.get("content")
                    if isinstance(c, list):
                        c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text")
                    msgs.append({"role": "tool", "tool_call_id": b.get("tool_use_id"), "content": c or ""})
                elif b.get("type") == "text" and str(b.get("text") or "").strip():
                    # A text block in a user event is text the CLI itself put in
                    # front of the model -- a loaded skill's body ("Base directory for this skill: ..." after the
                    # Skill tool's "Launching skill: X" result), a reminder, a compaction summary. Dropping it would
                    # leave the evolver's trace without the skill text the solver actually read. Kept verbatim, flagged.
                    msgs.append({"role": "user", "content": b.get("text"), "from_harness": True})
    for a in msgs:
        if a["role"] == "assistant" and not a.get("content") and a.get("tool_calls"):
            a["content"] = None                 # the direct harness's shape for a tool-calling turn
    return msgs


def model_replies(stream_text: str) -> int:
    """Model replies = distinct assistant message ids (what the direct harness counts as turns; Claude Code's own
    num_turns reads one more on a capped run: 46 for --max-turns 45)."""
    ids = set()
    for ln in stream_text.split("\n"):
        if not ln.strip():
            continue
        if not ln.lstrip().startswith("{"):    # a driver marker in a captured stream (CC_PROBE_RC=, STDERR: ...)
            continue                            # is not an event (events() skips the same lines); a torn JSON line still raises
        e = json.loads(ln)
        if e.get("type") == "assistant" and isinstance(e.get("message"), dict):
            ids.add(e["message"].get("id"))
    return len(ids)
