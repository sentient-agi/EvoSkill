#!/usr/bin/env python3
"""Claude Code as the solver on ONE Daytona machine, one isolated container per rollout unit, a QUEUE of
--concurrency in flight. The pure half (prompts, parity, upload set, stream parsing, selection, limits, the rejection
rule) is scripts/cc_solver/cc_solver_lib.py; read its docstring for what is held equal to the direct solver and why
--bare is not used. The loop's legs use Box, Queue, judge, poll_batch and read_credential from here
(scripts/daytona/cc_leg_driver.py); the commands below are the standalone runner they grew out of.

    python cc_solver.py sample   --out S --split F [--kind test10|train|val --samples 3]
    python cc_solver.py prepare  --out S --split F --run-dir R [--tasks all]   # workspaces, prompts (SSBV2_ROOT set)
    python cc_solver.py run      --out S --split F --run-dir R --tasks all|rest|UNIT,TASK,.. [--concurrency 20]
                                 [--token-file F] [--wall-min 90] [--ttl-minutes 240] [--force]
    python cc_solver.py grade    --out S --split F --tasks all|ungraded|...   # ssb_grade_daytona.py
    python cc_solver.py report   --out S --tasks ...   # per unit, per task pass@k
    python cc_solver.py cleanup                        # delete leftover cc-solver machines

(python = the venv with the `daytona` extra.) The credential is the run's one (read_credential): --token-file or
EVOSKILL_CC_TOKEN_FILE (a subscription token file, mode 0600), else ANTHROPIC_API_KEY.

`run`, in order (every step recorded under S/machine/<stamp>/ and S/runs/<unit>/):
  1. the machine: Dockerfile.ssb_leg (sha = INFRA leg_sandbox), 8 vCPU / 32 GiB / 40 GB, domain allow list
     api.anthropic.com ONLY, ephemeral, ttl --ttl-minutes (default 240); snapshot_ref read back = the LEG snapshot;
     ledgered (daytona_lib_evo);
  2. evoctl (scripts/daytona/evolver_container_ctl.py) + the egress proxy + the pinned Claude Code linux binary,
     each sha-verified on the machine;
  3. the QUEUE: at most --concurrency units hold a container at once. Admitting a unit = `prepare` (spec: nothing
     writable below /Users, the workspace = /evo/work, cgroup limits_for(concurrency)), `up` (own mnt/pid/net/ipc/uts
     namespaces, pivot_root, uid 1000, no capabilities, proxy in the container's netns, api.anthropic.com only), the
     PRE probe (pool_container_evolver.container_probe_problems), the upload (exactly the task's input workbook), the
     workspace handed to evo (+ an empty output/), the listing INSIDE = the manifest, the toolchain probe, then claude
     launched detached (the credential on fd 3 only), streamed back byte-exactly, --wall-min hard wall. When a
     rollout ends: its stream's LAST rate_limit_info, init event and thinking blocks are checked (the GUARD), then
     stderr, cgroup peaks, POST probe, down, pack, download, log round trip, credential scan, output resolved as the
     direct solver resolves it, `down --wipe` -- and only then is its slot refilled with the next unit;
  4. the GUARD stops ADMITTING (in-flight rollouts finish; never killed) on: a session REJECTED for a usage limit
     (status 'rejected' or an HTTP 429), an init / thinking problem, 3 setup failures, a credential hit, or the
     machine TTL (no admission later than TTL - wall - 20 min). The units never started stay for `run --tasks rest`;
     the stop is recorded in MACHINE.json `guard`;
  5. the machine DELETED and the deletion verified with a listing (finally, also on SIGTERM / crash, after every
     container is wiped).
Exit 0 = every selected unit ran to a result and the machine is gone; 9 = the guard stopped admission (the rest is
left for `run --tasks rest`).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cc_solver_lib as L  # noqa: E402

REPO = L.REPO
ctl = L.ctl
PY, CTL = ctl.PY, ctl.CTL
q = shlex.quote
TAG = "cc-solver"
DRIVER = "cc_solver"
POLL_S = 15
CHUNK_B = 1 << 20
GRADE = REPO / "scripts/daytona/ssb_grade_daytona.py"
GRADE_LABEL = "ccsolver"
JOB_LABEL = "ccsonnet5"
CC_LOCAL = REPO / ".cache" / "claude_code" / ctl.CC_VERSION / "linux-x64" / "claude"
PROXY_SRC = REPO / "scripts/evolver_egress_proxy.py"
CTL_SRC = REPO / "scripts/daytona/evolver_container_ctl.py"


class Refuse(RuntimeError):
    pass


class TokenHit(Refuse):
    """The credential (or a token's prefix) in a rollout's files: never admit another rollout on it."""


_LOG = {"fh": None}
_LOCK = threading.Lock()


def log(msg: str) -> None:
    line = f"{L.now_utc()} [cc-solver] {msg}"
    with _LOCK:
        print(line, flush=True)
        if _LOG["fh"]:
            _LOG["fh"].write(line + "\n")
            _LOG["fh"].flush()


def parse_json_tail(out: str) -> dict:
    for ln in reversed((out or "").strip().splitlines()):
        ln = ln.strip()
        if ln.startswith("{"):
            try:
                return json.loads(ln)
            except ValueError:
                continue
    return {}


def load_selection(root: Path) -> dict:
    """The out root's selection: sample10.json (the test probe) or selection_train.json (the train rollouts) --
    exactly one of them. Adds `units` ([{unit, task, sample_idx}]) and `sel_file`."""
    # A val selection, selection_val.json, is the train one with key "val" (every split check below reads sel["key"]).
    have = [n for n in (L.SAMPLE_FILE, L.SELECTION_FILE, L.VAL_SELECTION_FILE) if (root / n).is_file()]
    if len(have) != 1:
        raise Refuse(f"{root}: {have or 'no selection file'} -- want exactly one of {L.SAMPLE_FILE} / "
                     f"{L.SELECTION_FILE} / {L.VAL_SELECTION_FILE} (run `cc_solver.py sample` first)")
    sel = json.loads((root / have[0]).read_text())
    sel["units"] = L.selection_units(sel)
    sel["sel_file"] = have[0]
    sel.setdefault("key", L.SPLIT_KEY)
    return sel


def _completed(root: Path, unit: str) -> bool:
    p = L.run_dir(root, unit) / "RESULT.json"
    try:
        return p.is_file() and bool(json.loads(p.read_text()).get("completed"))
    except ValueError:
        return False


def _graded(root: Path, unit: str) -> bool:
    rd = L.run_dir(root, unit)
    g, r = rd / "GRADE.json", rd / "RESULT.json"
    return g.is_file() and r.is_file() and g.stat().st_mtime >= r.stat().st_mtime


def select_units(spec: str, sel: dict, root: Path) -> list:
    """all | rest (no completed RESULT.json) | ungraded (completed, no GRADE.json newer than it) | a comma list of
    unit ids and/or task ids (a task id = every unit of that task). Order = the selection's."""
    units = [u["unit"] for u in sel["units"]]
    if spec == "all":
        return units
    if spec == "rest":
        return [u for u in units if not _completed(root, u)]
    if spec == "ungraded":
        return [u for u in units if _completed(root, u) and not _graded(root, u)]
    want = [t.strip() for t in spec.split(",") if t.strip()]
    by_task: dict = {}
    for u in sel["units"]:
        by_task.setdefault(u["task"], []).append(u["unit"])
    out, off = [], []
    for w in want:
        got = [w] if w in units else by_task.get(w)
        if not got:
            off.append(w)
        out += [u for u in got or [] if u not in out]
    if off:
        raise Refuse(f"not in {sel['sel_file']}: {off}")
    return [u for u in units if u in out]


def unit_rows(sel: dict) -> dict:
    return {u["unit"]: u for u in sel["units"]}


def infra(run_dir) -> dict:
    return L.infra_of(run_dir)


# ───────────────────────────────────────────────────────────────────────────── sample / prepare (no Daytona)
def cmd_sample(a) -> int:
    root = Path(a.out)
    names = {"train": L.SELECTION_FILE, "val": L.VAL_SELECTION_FILE, "test10": L.SAMPLE_FILE}
    if a.kind in ("train", "val"):
        p = root / names[a.kind]
        s = L.train_selection(a.split, samples=a.samples, key=a.kind)
        same = ("units", "split_sha256", "key", "samples")
    else:
        p = root / L.SAMPLE_FILE
        s = L.sample_tasks(a.split)
        same = ("task_ids", "split_sha256")
    others = [root / n for k, n in names.items() if k != a.kind and (root / n).is_file()]
    if others:
        raise Refuse(f"{others[0]} exists: one selection per out root (use another --out)")
    if p.is_file():
        old = json.loads(p.read_text())
        if any(old.get(k) != s.get(k) for k in same):
            raise Refuse(f"{p} exists with a different selection")
        s = old
    else:
        L.jdump(p, s)
    brief = {k: v for k, v in s.items() if k not in ("units",)}
    print(json.dumps(brief, indent=1))
    if s.get("units"):
        print(f"{len(s['units'])} units: {s['units'][0]['unit']} ... {s['units'][-1]['unit']}")
    return 0


def cmd_prepare(a) -> int:
    """Workspaces by the repo's own builder, the data overlay's bytes, and the solver's user message per task (the
    instruction text + the direct solver's task message; nothing is appended to Claude Code's system prompt).
    The dataset is read from $SSBV2_ROOT."""
    from src.agent_profiles.ssbv2_agent.dataset import TASK_INDEX, load_ssbv2
    from src.agent_profiles.ssbv2_agent.task_workspace import setup_task_workspace, task_ws_name
    from src.orchestrator import data_overlay
    root = Path(a.out)
    sel = load_selection(root)
    units = select_units(a.tasks, sel, root)
    probs = L.units_split_problems([unit_rows(sel)[u] for u in units], a.split, sel["key"])
    if probs:
        raise Refuse(f"selection vs split: {probs[:5]}")
    ssbv2_root = os.environ.get("SSBV2_ROOT")
    if not ssbv2_root:
        raise Refuse("SSBV2_ROOT is not set (the SpreadsheetBench v2 checkout the dataset is read from)")
    inf = infra(a.run_dir)
    pj = inf["data"]["overlay_patches_json"]
    if L.sha256_file(pj) != inf["data"]["overlay_patches_sha256"]:
        raise Refuse(f"{pj} does not hash to INFRA.json data.overlay_patches_sha256")
    overlay = data_overlay.verify(pj)
    train, val, test = load_ssbv2(ssbv2_root)
    questions = {json.loads(qq)["task_id"]: json.loads(qq) for qq, _gt, _c in list(train) + list(val) + list(test)}
    golden_paths = [info["golden_path"] for info in TASK_INDEX.values()]
    golden_shas = {}
    for tid, info in TASK_INDEX.items():
        for p in {info["golden_path"], str(data_overlay.source(info["golden_path"], pj))}:
            golden_shas[L.sha256_file(p)] = tid
    instruction = L.SOLVER_PREAMBLE                   # bare metal: the task's message is the whole message
    ws_root = root / "ws"
    skills = root / "empty_skills"
    skills.mkdir(parents=True, exist_ok=True)
    if any(skills.iterdir()):
        raise Refuse(f"{skills} must be empty")
    rows_by_unit = unit_rows(sel)
    for unit in units:
        t, k = rows_by_unit[unit]["task"], rows_by_unit[unit]["sample_idx"]
        qd = questions[t]
        canonical = qd["spreadsheet_path"]
        src = data_overlay.source(canonical, pj)
        ws = ws_root / task_ws_name(t, k)             # one fresh workspace per unit (<task>__s<k>)
        if ws.name != L.unit_dir_name(unit):
            raise Refuse(f"{unit}: workspace {ws.name} != run dir {L.unit_dir_name(unit)}")
        if ws.exists():
            shutil.rmtree(ws)
        if not Path(canonical).is_file():
            raise Refuse(f"{t}: input {canonical} missing")
        setup_task_workspace(task_id=t, spreadsheet_path=canonical, tasks_cache_root=ws_root,
                             shared_skills_dir=skills, sample_idx=k)
        link = ws / ".claude" / "skills"
        if not link.is_symlink():
            raise Refuse(f"{t}: setup_task_workspace made no skills link (builder changed?)")
        link.unlink()                                 # the CC solver gets NO skills dir (the direct one's is empty)
        (ws / ".claude").rmdir()
        dst = ws / "spreadsheets" / Path(canonical).name
        shutil.copyfile(src, dst)                     # the overlay's bytes, as the leg bundle uploads them
        rows = L.upload_rows(ws)
        in_sha = L.sha256_file(src)
        probs = L.check_upload_set(rows, spreadsheet_filename=qd["spreadsheet_filename"], input_sha256=in_sha,
                                   golden_shas=golden_shas, golden_paths=golden_paths)
        if probs:
            raise Refuse(f"{t}: upload set: {probs}")
        um = L.task_user_message(qd["instruction"], qd["spreadsheet_filename"])
        rd = L.run_dir(root, unit)
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "user_message.txt").write_text(um)
        prep = {"schema": L.SCHEMA + ":prep", "unit": unit, "task_id": t, "sample_idx": k,
                "safe_id": L.safe_id(t), "unit_dir": L.unit_dir_name(unit), "category": qd["category"],
                "spreadsheet_filename": qd["spreadsheet_filename"], "canonical_input": canonical,
                "input_source": str(src), "overlay_applied": str(src) != os.path.normpath(canonical),
                "input_sha256": in_sha, "workspace": str(ws), "workspace_container": L.WS_CONTAINER,
                "output_dir_container": f"{L.WS_CONTAINER}/output", "upload_rows": rows, "upload_problems": [],
                "golden_shas_checked": len(golden_shas),
                "user_message_sha256": L.sha256_bytes(um.encode()),
                "user_message_source": "the task's instruction (dataset) + its input and output paths "
                                       "(cc_solver_lib.task_user_message)",
                "instruction_sha256": L.sha256_bytes(instruction.encode()),
                "prompt_delivery": L.PROMPT_DELIVERY,
                "question": qd, "data_overlay": {"patches_json": pj, "entries": len(overlay)},
                "created_utc": L.now_utc()}
        L.jdump(rd / "PREP.json", prep)
        print(f"{unit}: ws {ws.name} input {qd['spreadsheet_filename']} ({rows[0]['bytes']} B, overlay "
              f"{prep['overlay_applied']}), user message {len(um)} chars (instruction {len(instruction)})")
    print(f"prepared {len(units)} unit(s) of {len({rows_by_unit[u]['task'] for u in units})} task(s) "
          f"({sel['sel_file']}, split[{sel['key']!r}]); {L.PROMPT_DELIVERY}")
    return 0


# ───────────────────────────────────────────────────────────────────────────── the machine
class Machine:
    def __init__(self, root: Path, stamp: str, ttl_minutes: int, run_dir):
        import daytona_lib_evo as dle
        from src.orchestrator import daytona_evolver as de
        self.dle, self.de = dle, de
        self.dir = root / "machine" / stamp
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ttl = int(ttl_minutes)
        self.d = self.sb = self.sid = None
        self.rec: dict = {"stamp": stamp, "ttl_minutes": self.ttl}
        self.inf = infra(run_dir)
        self.lock = threading.Lock()          # evoctl prepare / up one at a time (the root cgroup's subtree)
        self.token_name = None                # where the credential came from (a token file's path or a variable's
        #                                       name; never its content)
        self.t_created = None                 # time.time() just before the create call: the TTL clock

    def save(self):
        L.jdump(self.dir / "MACHINE.json", self.rec)

    def sh(self, cmd: str, *, timeout: int = 600, env=None, what: str = "", retries: int = 3) -> tuple:
        last = None
        for _ in range(max(1, retries)):
            try:
                code, out, _s = self.dle.run(self.sb, cmd, timeout=timeout, env=env)
                return code, out
            except Exception as e:  # noqa: BLE001 -- transient toolbox errors
                last = e
                time.sleep(10)
        raise Refuse(f"exec {what or cmd[:40]!r} failed {retries}x: {type(last).__name__}: {last}")

    def ctl(self, args: str, *, timeout: int = 600, what: str = "", retries: int = 2) -> tuple:
        code, out = self.sh(f"{PY} {CTL} {args}", timeout=timeout, what=what or args.split()[0], retries=retries)
        return code, out, parse_json_tail(out)

    def up(self):
        from daytona import CreateSandboxFromImageParams, Image, Resources
        from src.orchestrator import pool_container_evolver as pce
        ls = self.inf["leg_sandbox"]
        df = self.de.BASE_DOCKERFILE
        if L.sha256_file(df) != ls["dockerfile_sha256"] or self.de.BASE_DOCKERFILE_SHA != ls["dockerfile_sha256"]:
            raise Refuse(f"{df} sha != INFRA leg_sandbox.dockerfile_sha256")
        if not CC_LOCAL.is_file() or L.sha256_file(CC_LOCAL) != ctl.CC_SHA256 or \
                CC_LOCAL.stat().st_size != ctl.CC_BYTES:
            raise Refuse(f"{CC_LOCAL}: not the pinned Claude Code {ctl.CC_VERSION} linux-x64 binary")
        self.d = self.dle.client()
        name = f"{self.dle.PREFIX}ccsolver-{time.strftime('%H%M%S')}"
        params = CreateSandboxFromImageParams(
            name=name, image=Image.from_dockerfile(str(df)), resources=Resources(**self.de.RESOURCES),
            ephemeral=True, auto_stop_interval=0, ttl_minutes=self.ttl, public=False,
            domain_allow_list=ctl.ALLOW, labels={"role": TAG})
        log(f"creating {name} (leg image, {self.de.RESOURCES}, allow {ctl.ALLOW}, ttl {self.ttl} min)")
        self.t_created = time.time()
        sb, sid, err, secs = self.dle.create(self.d, params, self.de.RESOURCES, tag=TAG, timeout=3600)
        self.sid = sid
        self.rec.update(name=name, sandbox_id=sid, create_s=secs)
        self.save()
        if err is not None:
            raise Refuse(f"create failed after {secs}s: {type(err).__name__}: {str(err)[:500]}")
        self.sb = sb
        rb = self.dle.readback(sb)
        self.rec["readback_create"] = rb
        self.save()
        probs = []
        if pce.allow_list_set(rb.get("domain_allow_list")) != frozenset({ctl.ALLOW}):
            probs.append(f"domain_allow_list {rb.get('domain_allow_list')!r}")
        if ls.get("snapshot_ref") and rb.get("snapshot_ref") != ls["snapshot_ref"]:
            probs.append(f"snapshot_ref {rb.get('snapshot_ref')} != the leg snapshot {ls['snapshot_ref']}")
        for k, v in self.de.RESOURCES.items():
            if rb.get(k) != v:
                probs.append(f"{k} {rb.get(k)} != {v}")
        if rb.get("network_block_all"):
            probs.append("network_block_all is set (api.anthropic.com would be unreachable)")
        self.rec["readback_problems"] = probs
        self.save()
        log(f"machine {sid} up in {secs}s: allow {rb.get('domain_allow_list')!r}, snapshot "
            f"{rb.get('snapshot_ref')}, {rb.get('cpu')} cpu / {rb.get('memory')} GiB; problems {probs}")
        if probs:
            raise Refuse("machine readback: " + "; ".join(probs))
        self.dle.ledger(event="cc_solver_machine_ready", driver=DRIVER, tag=TAG, id=sid, name=name)

    def install(self):
        self.sh(f"mkdir -p {ctl.BIN} && chmod 0700 {ctl.BASE}", timeout=60, what="mkdir")
        for src, dst in ((CTL_SRC, ctl.CTL), (PROXY_SRC, ctl.PROXY_FILE)):
            self.sb.fs.upload_file(str(src), dst, timeout=300)
        code, out = self.sh(f"sha256sum {ctl.CTL} {ctl.PROXY_FILE}", timeout=60, what="ctl sha")
        got = {ln.split()[1]: ln.split()[0] for ln in out.splitlines() if len(ln.split()) == 2}
        want = {ctl.CTL: L.sha256_file(CTL_SRC), ctl.PROXY_FILE: L.sha256_file(PROXY_SRC)}
        if got != want:
            raise Refuse(f"controller upload does not verify ({got} != {want})")
        part = ctl.CC_HOST + ".part"
        self.sh(f"mkdir -p {os.path.dirname(ctl.CC_HOST)} && rm -f {part}", timeout=60, what="cc dir")
        t = time.time()
        self.sb.fs.upload_file(str(CC_LOCAL), part, timeout=3600)
        code, out = self.sh(f"test \"$(sha256sum {part} | cut -d' ' -f1)\" = {ctl.CC_SHA256} && "
                            f"test \"$(stat -c %s {part})\" = {ctl.CC_BYTES} && chmod 0755 {part} && "
                            f"mv {part} {ctl.CC_HOST} && echo CC_OK", timeout=300, what="cc verify")
        if "CC_OK" not in out:
            raise Refuse("uploaded Claude Code binary does not verify on the machine")
        self.rec["install"] = {"ctl_sha256": want, "cc": f"{ctl.CC_BYTES} B in {time.time() - t:.0f}s, sha verified",
                               "cc_sha256": ctl.CC_SHA256}
        self.save()
        log(f"evoctl + proxy + Claude Code {ctl.CC_VERSION} installed and verified")

    def delete(self) -> dict:
        res = {"sandbox_id": self.sid}
        if not self.sid:
            return res
        if self.d is None:
            self.d = self.dle.client()
        res["deleted"] = self.dle.delete_by_id(self.d, self.sid, TAG)
        # the listing lags the delete by seconds (deleted=True and get() "not found" while the listing still shows
        # the machine STARTED for up to a minute): re-list for up to 3 min
        t_end = time.time() + 180
        while True:
            try:
                listing = self.dle.list_all_ids(self.d)
            except Exception as e:  # noqa: BLE001
                listing = [f"listing failed: {type(e).__name__}: {e}"]
            still = [r for r in listing if r.startswith(self.sid[:8])]
            live = [r for r in still if "destroy" not in r.lower()] or \
                [r for r in listing if r.startswith("listing failed")]
            if not live or time.time() > t_end:
                break
            time.sleep(10)
        res.update(listing_after=listing, listing_utc=L.now_utc(), still_listed=still,
                   verified_gone=bool(res["deleted"]) and not live)
        self.rec["delete"] = res
        self.save()
        log(f"machine {self.sid} deleted={res['deleted']}; listing now {listing}")
        return res


# ───────────────────────────────────────────────────────────────────────────── one task's container
class Box:
    # A loop leg's LegBox (cc_leg_driver.py) overrides these for a SKILL leg -- the
    # promoted set's rows join the upload tgz, the workspace gets exactly one link (.claude/skills -> /evo/skills),
    # and the in-container listing must show that link and the skill dirs. The defaults are cc_solver's behaviour.
    expect_link_count = 0
    expect_links = None
    extra_dirs = ()
    skill_rows: list = []
    model = L.MODEL                       # a loop leg's LegBox sets the run's SPEC solver_model here

    def __init__(self, m: Machine, root: Path, unit: str, stamp: str, wall_s: int, limits: dict):
        self.m, self.unit = m, unit
        self.task = unit                      # the log / message name of this box (the unit id)
        self.rd = L.run_dir(root, unit)
        if not (self.rd / "PREP.json").is_file():
            raise Refuse(f"{unit}: not prepared (cc_solver.py prepare)")
        self.prep = json.loads((self.rd / "PREP.json").read_text())
        if self.prep.get("unit", self.prep.get("task_id")) != unit:
            raise Refuse(f"{unit}: PREP.json is {self.prep.get('unit')!r}")
        self.um = (self.rd / "user_message.txt").read_text()
        if L.sha256_bytes(self.um.encode()) != self.prep["user_message_sha256"]:
            raise Refuse(f"{unit}: user_message.txt differs from PREP.json")
        self.rows = self.prep["upload_rows"]
        for r in self.rows:
            if L.sha256_file(r["source"]) != r["sha256"]:
                raise Refuse(f"{unit}: {r['source']} changed since prepare")
        self.cid = L.make_cid(unit, stamp, uuid.uuid4().hex[:4])
        self.cdir = ctl.cdir(self.cid)
        self.wall_s = int(wall_s)
        self.spec = L.container_spec(self.cid, wall_s=wall_s, limits=limits)
        self.rec: dict = {"unit": unit, "task": self.prep.get("task_id", unit),
                          "sample_idx": self.prep.get("sample_idx"), "cid": self.cid, "machine_run": m.dir.name,
                          "sandbox_id": m.sid, "spec": self.spec, "argv": L.claude_argv(self.um, model=self.model),
                          "env": L.cc_env()}
        self.off, self.fail, self.stopped, self.rc = 0, 0, None, None
        self.t_launch = self.t_done = None
        self.status = "new"
        self.pl = None
        self.stream_fh = None
        self.guard = None
        self.wiped = self.result_written = False
        self.res = None

    def save(self):
        L.jdump(self.rd / "RECEIPT.json", self.rec)

    def fail_on(self, cond, msg):
        if cond:
            self.status = "refused"
            self.rec["refused"] = msg
            self.save()
            raise Refuse(f"{self.task}: {msg}")

    def prepare(self):
        b64 = base64.b64encode(json.dumps(self.spec).encode()).decode()
        code, out, res = self.m.ctl(f"prepare --spec-b64 {b64}", timeout=900, what="prepare")
        self.rec["prepare"] = {k: res.get(k) for k in ("ok", "problems", "cdir", "etc", "expected_top",
                                                       "cc_host_sha256", "limits", "facts")}
        self.save()
        self.fail_on(not res.get("ok"), f"prepare refused: {res.get('problems') or out[-400:]}")
        self.pl = ctl.plan(self.spec, res["facts"], res["limits"])
        self.fail_on(self.pl != res.get("plan"), "the machine's plan differs from the one derived here")
        self.status = "prepared"

    def up(self, init_root: bool):
        code, out, res = self.m.ctl(f"up --cid {self.cid}" + (" --init-root-cgroup" if init_root else ""),
                                    timeout=300, what="up")
        self.rec["up"] = res
        self.save()
        self.fail_on(not res.get("ok"), f"container did not come up: {res.get('problems') or out[-600:]}")
        self.m.dle.ledger(event="container_up", driver=DRIVER, tag=TAG, id=self.m.sid, cid=self.cid, task=self.task)
        self.status = "up"

    def enter(self, argv: list, *, cwd: str = "/tmp", timeout: int = 900, what: str = "enter") -> tuple:
        cmd = f"{PY} {CTL} enter --cid {self.cid} --cwd {cwd} -- " + " ".join(q(x) for x in argv) + " </dev/null"
        return self.m.sh(cmd, timeout=timeout, what=what, retries=2)

    def probe(self):
        from src.orchestrator import pool_container_evolver as pce
        code, out, res = self.m.ctl(f"probe --cid {self.cid} --phase pre", timeout=1800, what="probe")
        pre = res.get("probe") or {}
        probs = pce.container_probe_problems(
            pre, expected_top=ctl.expected_top(self.pl), expected_mounts=ctl.expected_mount_points(self.pl),
            rw_paths=[], ro_rebind=[], require_libreoffice=self.m.inf["leg_sandbox"]["libreoffice"])
        self.rec["probe_pre"] = pre
        self.rec["probe_pre_problems"] = probs
        self.save()
        v = pre.get("view") or {}
        log(f"{self.task}: PRE probe {len(probs)} problem(s); {v.get('canaries_visible_n')} of "
            f"{v.get('canaries_checked')} machine paths visible; / = {v.get('top')}")
        self.fail_on(bool(probs), "PRE probe: " + "; ".join(probs[:5]))

    def after_extract(self):
        """Hook between `evoctl extract` and the workspace handover: a loop skill
        leg's LegBox (cc_leg_driver.py) makes the workspace's one link .claude/skills -> /evo/skills here, so the
        handover's link count and the listing inside see it. cc_solver's own boxes do nothing."""
        return None

    def upload(self):
        data = L.build_upload_tgz(self.rows + list(self.skill_rows))
        tmp = self.m.dir / f"upload_{self.cid}.tgz"
        tmp.write_bytes(data)
        self.m.sb.fs.upload_file(str(tmp), f"{self.cdir}/state/upload.tgz", timeout=900)
        tmp.unlink()
        code, out, res = self.m.ctl(f"extract --cid {self.cid}", timeout=600, what="extract")
        self.fail_on(not res.get("ok"), f"extract refused: {res.get('problems') or out[-400:]}")
        self.after_extract()
        w = f"{self.cdir}/fs{L.WS_CONTAINER}"
        # the workspace handed to evo (the direct solver's episode owns its whole /workspace) + an empty output/,
        # as setup_task_workspace leaves it; nothing in it can be a link (evoctl extract writes regular files only)
        code, out = self.m.sh(f"set -e; W={q(w)}; test -d \"$W\"; test \"$(find \"$W\" -mindepth 1 -type l | wc -l)\" "
                              f"= {int(self.expect_link_count)}; chown -R {ctl.EVO_UID}:{ctl.EVO_GID} \"$W\"; "
                              f"install -d -o {ctl.EVO_UID} -g "
                              f"{ctl.EVO_GID} -m 0755 \"$W/output\"; find \"$W\" -printf '%y %u %m %s %P\\n'; "
                              f"echo WS_OK", timeout=120, what="workspace owner")
        self.rec["workspace_host_view"] = out.strip().splitlines()[-12:]
        self.fail_on("WS_OK" not in out, f"workspace handover failed: {out[-400:]}")
        from src.orchestrator import daytona_evolver as de
        code, out = self.enter([PY, "-c", de.VM_LISTING_PY, "/Users", ctl.VM_ROOT, ctl.POLICY_DIR], what="listing")
        try:
            listing = json.loads(out.strip().splitlines()[-1])
        except (ValueError, IndexError):
            self.fail_on(True, f"listing inside failed: {out[-400:]}")
        # the listing inside must equal the manifest + the skill rows + the one link
        probs = L.listing_problems(listing, self.rows + list(self.skill_rows), expect_links=self.expect_links,
                                   extra_dirs=self.extra_dirs)
        self.rec["listing_after_upload"] = listing
        self.rec["listing_problems"] = probs
        self.save()
        log(f"{self.task}: upload {len(self.rows)} file(s) ({sum(r['bytes'] for r in self.rows)} B); listing "
            f"inside {len(listing)} entries, problems {probs}")
        self.fail_on(bool(probs), "listing inside != manifest: " + "; ".join(probs))

    def toolchain(self):
        code, out = self.enter([PY, "-c", L.TOOLCHAIN_PY], cwd=L.WS_CONTAINER, timeout=600, what="toolchain")
        t = parse_json_tail(out)
        probs = L.toolchain_problems(t) if t else [f"toolchain probe unparseable: {out[-300:]}"]
        self.rec["toolchain"] = t
        self.rec["toolchain_problems"] = probs
        self.save()
        log(f"{self.task}: toolchain {len(probs)} problem(s): python {(t.get('python_stack') or {}).get('out')}; "
            f"{(t.get('libreoffice') or {}).get('out')}; recalc {t.get('recalc_recipe')}")
        self.fail_on(bool(probs), "toolchain: " + "; ".join(probs))
        self.status = "ready"

    def token_cmd(self, inner: str, *, cwd: str, detached: bool, tok: str) -> str:
        """evolver_container.ContainerMixin.enter_cmd's shape: the credential (a token or an API key, carried in the
        exec's env as CLAUDE_CODE_OAUTH_TOKEN whatever its kind) leaves that env before any process of ours starts and
        reaches the container only as fd 3 (a here-string pipe); `inner` names the fd for claude (cred_fd_var)."""
        base = (f"env -u CLAUDE_CODE_OAUTH_TOKEN {PY} {CTL} enter --cid {self.cid} --cwd {cwd} --keep-fd3"
                + (f" --log {self.cdir}/state/launch.err" if detached else "") + f" -- /bin/bash -c {q(inner)}")
        redir = ' 3<<<"$CLAUDE_CODE_OAUTH_TOKEN"'
        cmd = (f"(setsid nohup {base} </dev/null >/dev/null 2>&1{redir} &)" if detached
               else base + " </dev/null" + redir)
        if tok in cmd:
            raise Refuse("the credential would appear in a command string")
        return cmd

    def launch(self, tok: str):
        self.tok_used = tok                   # the credential this session starts on (its collect scans for it)
        inner = L.launch_inner(self.um, model=self.model)
        cmd = self.token_cmd(inner, cwd=L.WS_CONTAINER, detached=True, tok=tok) + \
            f" ; sleep 4; {PY} {CTL} status --cid {self.cid} | tail -c 300"
        self.rec["launched_utc"] = L.now_utc()
        self.save()
        # never retried: a second claude, a truncated log
        code, out = self.m.sh(cmd, timeout=180, env={"CLAUDE_CODE_OAUTH_TOKEN": tok}, what="launch", retries=1)
        self.stream_fh = open(self.rd / "stream.jsonl", "wb")
        self.t_launch = time.time()
        self.status = "running"               # last: the poller picks the box up only now
        log(f"{self.task}: launched in {self.cid}")

    def stop(self, why: str):
        self.stopped = why
        self.rec["stopped"] = {"why": why, "utc": L.now_utc()}
        log(f"{self.task}: STOP ({why})")
        try:
            self.m.ctl(f"stop-session --cid {self.cid}", timeout=180, what="stop")
        except Refuse as e:
            log(f"{self.task}: stop-session failed: {e}")

    def collect(self, tok: str):
        rec = self.rec
        try:
            self.stream_fh.close()
        except Exception:  # noqa: BLE001
            pass
        code, out = self.m.sh(f"{PY} {CTL} tail --cid {self.cid} --name {L.STDERR_NAME} --offset 0 --max 400000",
                              timeout=120, what="stderr")
        head, sep, b64 = out.partition("@@B64@@")
        (self.rd / "stderr.txt").write_bytes(base64.b64decode(b64.strip()) if sep and b64.strip() else b"")
        cg = f"{ctl.CG_PARENT}/{self.cid}"
        code, out = self.m.sh(f"for f in memory.peak memory.events cpu.stat pids.peak; do echo \"== $f\"; "
                              f"cat {cg}/$f 2>/dev/null; done", timeout=60, what="cgroup")
        rec["cgroup_end"] = out.strip()[-1500:]
        from src.orchestrator import pool_container_evolver as pce
        code, out, res = self.m.ctl(f"probe --cid {self.cid} --phase post", timeout=300, what="post probe")
        rec["probe_post_problems"] = pce.post_probe_problems(rec.get("probe_pre") or {}, res.get("probe") or {})
        code, out, res = self.m.ctl(f"down --cid {self.cid}", timeout=300, what="down")
        rec["down_before_pack"] = res
        parts = [L.WS_CONTAINER, L.OUT_CONTAINER, f"{ctl.EVO_HOME}/.claude"]
        code, out, res = self.m.ctl(f"pack --cid {self.cid} " + " ".join(f"--part {q(p)}" for p in parts),
                                    timeout=900, what="pack")
        rec["pack"] = {k: res.get(k) for k in ("ok", "bytes", "packed", "refused")}
        self.save()
        if not res.get("ok"):
            raise Refuse(f"{self.task}: pack refused: {res.get('problems') or out[-300:]}")
        local = self.rd / "container_out.tgz"
        self.m.sb.fs.download_file(f"{self.cdir}/state/out.tgz", str(local))
        ex = self.rd / "container"
        if ex.exists():
            shutil.rmtree(ex)
        ex.mkdir()
        refused = []
        with tarfile.open(local, "r:gz") as tf:
            for mbr in tf.getmembers():
                if mbr.name.startswith("/") or ".." in Path(mbr.name).parts or not (mbr.isfile() or mbr.isdir()):
                    refused.append(mbr.name)
                    continue
                tf.extract(mbr, ex, filter="data")
        rec["download_refused_members"] = refused
        vm_log = ex / L.OUT_CONTAINER.lstrip("/") / L.STREAM_NAME
        streamed = self.rd / "stream.jsonl"
        rec["log_round_trip"] = {"streamed_sha256": L.sha256_file(streamed) if streamed.is_file() else None,
                                 "container_sha256": L.sha256_file(vm_log) if vm_log.is_file() else None}
        rec["log_round_trip"]["equal"] = (rec["log_round_trip"]["streamed_sha256"] is not None and
                                          rec["log_round_trip"]["streamed_sha256"] ==
                                          rec["log_round_trip"]["container_sha256"])
        outp = L.resolve_output(ex / L.WS_CONTAINER.lstrip("/") / "output", self.prep["spreadsheet_filename"])
        od = self.rd / "output"
        if od.exists():
            shutil.rmtree(od)
        if outp is not None:
            od.mkdir()
            shutil.copy2(outp, od / outp.name)
            rec["output"] = {"path": str(od / outp.name), "name": outp.name,
                             "expected_name": outp.name == self.prep["spreadsheet_filename"],
                             "sha256": L.sha256_file(od / outp.name), "bytes": (od / outp.name).stat().st_size}
        else:
            rec["output"] = None
        ts = {}
        for n in ("t_start", "t_end", "rc"):
            p = ex / L.OUT_CONTAINER.lstrip("/") / n
            ts[n] = p.read_text().strip() if p.is_file() else None
        rec["container_times"] = ts
        from src.orchestrator import daytona_evolver as de
        hits, n = de.token_scan([self.rd], tok)
        rec["token_scan"] = {"hits": hits, "scanned_files": n}
        scan = (rec["down_before_pack"] or {}).get("scan") or {}
        mhits = [h for part in scan.values() for h in (part or {}).get("hits") or []] + \
            list(((rec["down_before_pack"] or {}).get("tmpfs_scan") or {}).get("hits") or [])
        rec["container_token_hits"] = mhits
        self.save()
        if hits or mhits:
            raise TokenHit(f"{self.task}: CREDENTIAL OR TOKEN PREFIX FOUND ({len(hits)} local, {len(mhits)} in the "
                           f"container) -- rotate the credential ({self.m.token_name})")
        self.status = "collected"

    def wipe(self):
        try:
            code, out, res = self.m.ctl(f"down --cid {self.cid} --wipe", timeout=300, what="down wipe", retries=3)
            self.rec["down_wipe"] = res
            ok = bool(res.get("verified") and res.get("wiped"))
        except Refuse as e:
            self.rec["down_wipe"] = {"error": str(e)}
            ok = False
        self.m.dle.ledger(event="container_down", driver=DRIVER, tag=TAG, id=self.m.sid, cid=self.cid, verified=ok)
        self.save()
        return ok

    def result(self) -> dict:
        s = L.parse_stream((self.rd / "stream.jsonl").read_text(errors="replace")
                           if (self.rd / "stream.jsonl").is_file() else "")
        r = s["result"] or {}
        ts = self.rec.get("container_times") or {}
        mins = None
        try:
            mins = (int(ts["t_end"]) - int(ts["t_start"])) / 60.0
        except (TypeError, ValueError, KeyError):
            if r.get("duration_ms"):
                mins = r["duration_ms"] / 60000.0
        from src.orchestrator.claude_pins import check_session
        out = {"schema": L.SCHEMA + ":result", "unit": self.unit, "task": self.rec["task"],
               "sample_idx": self.rec["sample_idx"], "cid": self.cid, "machine_run": self.m.dir.name,
               "credential": self.m.token_name, "status": self.status, "claude_rc": self.rc, "stopped": self.stopped,
               # a measured outcome (a result event, or the wall) vs an infrastructure failure (re-run by `rest`)
               # CC ends an API / auth / rate-limit failure with subtype 'success' + is_error true +
               # terminal_reason 'api_error' -- infrastructure, never a measured outcome (re-run by `rest`)
               "infra_error": bool(r.get("is_error")) and (r.get("terminal_reason") == "api_error"
                                                           or r.get("api_error_status") is not None),
               "completed": self.status == "collected" and self.rc is not None
               and not (bool(r.get("is_error")) and (r.get("terminal_reason") == "api_error"
                                                     or r.get("api_error_status") is not None)),
               "subtype": r.get("subtype"), "is_error": r.get("is_error"), "num_turns": r.get("num_turns"),
               "duration_ms": r.get("duration_ms"), "minutes": mins, "cost": r.get("total_cost_usd"),
               "usage": r.get("usage"), "modelUsage": r.get("modelUsage"), "terminal_reason": r.get("terminal_reason"),
               "api_error_status": r.get("api_error_status"),
               "result_text_head": r.get("result_text_head"), "rate_limit": s["rate_limit"],
               "init": s["init"], "init_problems": L.init_problems(s["init"], model=self.model),
               # the init check's thinking half: --thinking-display summarized given, the thinking blocks carry text
               "thinking": s["thinking"], "thinking_problems": L.thinking_problems(s["thinking"], self.rec["argv"]),
               "guard": self.guard, "cgroup_end": self.rec.get("cgroup_end"),
               "session_check": check_session(str(self.rd / "stream.jsonl"), self.model)
               if (self.rd / "stream.jsonl").is_file() else None,
               "tool_uses": s["tool_uses"], "n_events": s["n_events"], "n_unparsed_lines": s["n_unparsed_lines"],
               "output": self.rec.get("output"), "log_round_trip": self.rec.get("log_round_trip"),
               "token_scan": self.rec.get("token_scan"), "probe_post_problems": self.rec.get("probe_post_problems"),
               "down_wipe_verified": bool((self.rec.get("down_wipe") or {}).get("verified")),
               "argv_flags": [a for a in self.rec["argv"] if a.startswith("-")], "env": self.rec["env"]}
        L.jdump(self.rd / "RESULT.json", out)
        return out


# ───────────────────────────────────────────────────────────────────────────── run
def read_token(path) -> str:
    """The subscription token in the file `path`: a regular file, mode 0600-like, one sk-ant-oat01- token.
    Never printed, never in a command string (Box.token_cmd); it reaches a container only as fd 3."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise Refuse(f"{path} is not a regular file")
    st = path.stat()
    if st.st_mode & 0o077:
        raise Refuse(f"{path} is readable by others (mode {oct(st.st_mode & 0o777)})")
    tok = path.read_text().strip()
    if not tok.startswith(L.TOKEN_PREFIX) or len(tok) < 40 or any(c.isspace() for c in tok):
        raise Refuse(f"{path} does not hold one OAuth token")
    return tok


def read_credential(declared=None, env=None) -> tuple:
    """(secret, cc_account.Credential, where) of the run's ONE credential (src/orchestrator/cc_account.py): the
    declared token file, else $EVOSKILL_CC_TOKEN_FILE (read_token), else $ANTHROPIC_API_KEY (an sk-ant- key, one
    word). `where` names its source for records (a path or a variable's name, never the value). Also records the
    kind for this process (cc_solver_lib.use_credential_kind), which every in-container command follows. Refused
    when there is none or it is malformed; the secret is never printed."""
    e = os.environ if env is None else env
    try:
        c = L.ACC.resolve(declared, e)
    except ValueError as err:
        raise Refuse(str(err)) from None
    if c.kind == L.ACC.TOKEN_FILE:
        secret, where = read_token(c.token_file), str(c.token_file)
    else:
        secret, where = (e.get(L.ACC.API_KEY_ENV) or "").strip(), L.ACC.API_KEY_ENV
        if not secret.startswith("sk-ant-") or len(secret) < 40 or any(ch.isspace() for ch in secret):
            raise Refuse(f"{L.ACC.API_KEY_ENV} does not hold one Anthropic API key")
    L.use_credential_kind(c.kind)
    return secret, c, where


def poll_batch(m: Machine, boxes: list) -> dict:
    items = " ".join(q(f"{b.cid}:{b.off}") for b in boxes)
    cmd = (f'for s in {items}; do c="${{s%%:*}}"; o="${{s##*:}}"; echo "@@CID@@ $c"; '
           f'{PY} {CTL} tail --cid "$c" --offset "$o" --max {CHUNK_B} --du 2>&1; done')
    code, out = m.sh(cmd, timeout=240, what="poll", retries=2)
    res = {}
    for sec in out.split("@@CID@@ ")[1:]:
        first, _, rest = sec.partition("\n")
        head, sep, b64 = rest.partition("@@B64@@")
        try:
            size, rc, du, alive = head.strip().splitlines()[-1].split()
            data = base64.b64decode(b64.strip()) if sep and b64.strip() else b""
            res[first.strip()] = (int(size), rc, int(du), alive, data)
        except (ValueError, IndexError):
            res[first.strip()] = None
    return res


def poll_once(m, active: list, wall_s: int, rw_max: int, poll=poll_batch) -> tuple:
    """One poll of the running boxes: stream bytes appended locally, the wall / disk stops applied. Returns
    (the boxes that ENDED this round, whether any bytes arrived)."""
    try:
        res = poll(m, active)
    except Refuse as e:
        log(f"poll failed: {e}")
        res = {}
    ended, got = [], False
    for b in active:
        r = res.get(b.cid)
        if r is None:
            b.fail += 1
            if b.fail > 40:
                b.status = "poll_failed"
                ended.append(b)
            continue
        b.fail = 0
        size, rc, du, alive, data = r
        if data:
            b.stream_fh.write(data)
            b.stream_fh.flush()
            b.off += len(data)
            got = True
        if rc != "RUNNING" and b.off >= size:
            b.rc, b.t_done, b.status = int(rc), time.time(), "finished"
            ended.append(b)
            log(f"{b.task}: claude exited rc {rc} after {(b.t_done - b.t_launch) / 60:.1f} min, {b.off} B stream")
        elif alive == "0" and rc == "RUNNING":
            b.status = "container_died"
            ended.append(b)
            log(f"{b.task}: the container died while claude was running")
        elif not b.stopped and time.time() - b.t_launch > wall_s:
            b.stop(f"wall {wall_s // 60} min")
        elif b.stopped and rc == "RUNNING":
            # a stopped box whose rc never appears must not hold the run forever: 20 more polls
            # (~5 min at the 15 s poll interval), then it is given up as stop_timeout
            b.after_stop = getattr(b, "after_stop", 0) + 1
            if b.after_stop > 20:
                b.status = "stop_timeout"
                ended.append(b)
                log(f"{b.task}: no exit code ~5 min after the stop -- marked stop_timeout")
        elif not b.stopped and du > rw_max:
            b.stop(f"writable dirs {du >> 20} MiB > {rw_max >> 20} MiB")
    return ended, got


def judge(b, threshold=None, expect_skills: list | None = None) -> dict:
    """The GUARD on an ended rollout, from its own stream: a usage-limit REJECTION (the LAST rate_limit_info says
    status 'rejected', cc_solver_lib.rejection_reason; or an HTTP 429), the init event (a claude that exited without
    one failed at start) and the thinking blocks. `expect_skills` (a loop skill leg): the names the init event must
    list (cc_solver_lib.init_problems). `threshold` is not read (no utilization level stops admission); the slot stays
    so callers that pass it positionally keep working."""
    p = b.rd / "stream.jsonl"
    s = L.parse_stream(p.read_text(errors="replace") if p.is_file() else "")
    last, r = s["rate_limit"]["last"], s["result"] or {}
    reasons = []
    why = L.rejection_reason(last)
    if why:
        reasons.append(why)
    if str(r.get("api_error_status")) == "429":
        reasons.append("HTTP 429 api_error")
    ip = L.init_problems(s["init"], model=getattr(b, "model", L.MODEL), expect_skills=expect_skills) \
        if (s["init"] or b.status == "finished") else []
    tp = L.thinking_problems(s["thinking"], b.rec["argv"])
    if ip:
        reasons.append(f"init {ip}")
    if tp:
        reasons.append(f"thinking {tp}")
    return {"utc": L.now_utc(), "unit": b.unit, "status": b.status, "rc": b.rc,
            "last_rate_limit": last, "n_rate_limit": s["rate_limit"]["n"], "subtype": r.get("subtype"),
            "api_error_status": r.get("api_error_status"), "thinking": s["thinking"], "init_problems": ip,
            "thinking_problems": tp, "stop_reasons": reasons}


def archive_previous(rd: Path, stamp: str) -> list:
    """A unit run again (rest / --force): its earlier RESULT / GRADE are kept aside, never mistaken for this run's."""
    moved = []
    for n in ("RESULT.json", "GRADE.json"):
        p = rd / n
        if p.is_file():
            dst = rd / f"{p.stem}.prev-{stamp}.json"
            os.replace(p, dst)
            moved.append(dst.name)
    return moved


WORKERS = 8                        # setups + collects in parallel (the main thread polls)
TTL_MARGIN_S = 20 * 60             # no admission unless wall + this still fits in the machine's TTL
SETUP_FAIL_STOP = 3


class Queue:
    """At most `conc` units hold a container at once; a slot is freed only after its rollout is collected and its
    container wiped: when a rollout ends, its container is collected and torn down before the next unit's starts.
    Admission stops for good on the first GUARD reason; in-flight rollouts always finish."""

    def __init__(self, m, root: Path, units: list, stamp: str, wall_s: int, limits: dict, tok: str, conc: int, *,
                 box_cls=None, poll=poll_batch, ttl_s: int | None = None, poll_s: float = POLL_S,
                 expect_skills: list | None = None, deadline_ts: float | None = None, on_idle_closed=None):
        """For the loop's leg driver: `expect_skills` reaches judge() (a skill leg's init events must list the
        promoted set); `deadline_ts` replaces the machine's t_created + ttl as the admission deadline (a pool
        machine's retire_at_ts); `on_idle_closed()` is called when nothing is in flight, admission is closed and units
        are pending -- True = it reopened admission (a rate-limit wait ended), False/None = return. Every default
        keeps the standalone runner's behaviour."""
        self.m, self.root, self.stamp = m, root, stamp
        self.pending = deque(units)
        self.wall_s, self.limits, self.tok = int(wall_s), limits, tok
        self.conc = int(conc)
        self.box_cls = box_cls or Box
        self.poll, self.poll_s = poll, poll_s
        self.ttl_s = int(ttl_s if ttl_s is not None else m.ttl * 60)
        self.expect_skills = (expect_skills if expect_skills is None or expect_skills == L.ANY_SKILLS
                              else list(expect_skills))          # L.ANY_SKILLS: whatever Claude Code loads
        self.deadline_ts = float(deadline_ts) if deadline_ts is not None else None
        self.on_idle_closed = on_idle_closed
        self.inflight: list = []
        self.boxes: list = []
        self.futs: dict = {}
        self.guard = {"admitting": True, "reason": None, "at_utc": None, "by": None, "pending_at_stop": None}
        self.readings: list = []
        self.errors: list = []
        self.setup_failures = 0
        self.max_inflight = 0
        self.stopping = False                 # shutdown(): no launch any more (the guard record is left as it is)
        self.ex = ThreadPoolExecutor(max_workers=WORKERS)

    # -- admission -----------------------------------------------------------------------------------------------
    def close(self, reason: str, by: str | None = None, kind: str | None = None):
        """`kind` (the leg driver's wait/resume): "quota" for a usage-limit rejection or an HTTP 429 (a wait for the
        reported reset can end it), "defect" for an init / thinking / credential-hit / setup-failure reason (the
        environment is wrong), "ttl" for the machine deadline. None = not classified (cc_solver's own callers)."""
        if self.guard["admitting"]:
            self.guard.update(admitting=False, reason=reason, at_utc=L.now_utc(), by=by,
                              pending_at_stop=len(self.pending))
            if kind is not None:
                self.guard["kind"] = kind
            log(f"GUARD: stop admitting ({reason}); {len(self.pending)} unit(s) left for `run --tasks rest`, "
                f"{len(self.inflight)} in flight will finish")

    def reopen(self, note: str):
        """Admission resumes after a rate-limit wait (the driver waited until the reset the rejected session reported)."""
        self.guard.update(admitting=True, reason=None, by=None, at_utc=None)
        self.guard.pop("kind", None)
        self.guard.setdefault("reopenings", []).append({"reopened_utc": L.now_utc(), "note": note})
        log(f"GUARD: admitting again ({note}); {len(self.pending)} unit(s) pending")

    def ttl_ok(self) -> bool:
        if self.deadline_ts is not None:
            return time.time() + self.wall_s + TTL_MARGIN_S <= self.deadline_ts
        t0 = self.m.t_created or time.time()
        return time.time() - t0 + self.wall_s + TTL_MARGIN_S <= self.ttl_s

    def admit(self):
        while self.guard["admitting"] and self.pending and len(self.inflight) < self.conc:
            if not self.ttl_ok():
                self.close(f"machine TTL {self.ttl_s // 60} min: a rollout admitted now could outlive the machine",
                           kind="ttl")
                return
            u = self.pending.popleft()
            try:
                b = self.box_cls(self.m, self.root, u, self.stamp[4:].lower(), self.wall_s, self.limits)
            except Refuse as e:
                self.errors.append(f"{u}: not admitted: {e}")
                log(f"{u}: not admitted: {e}")
                self.setup_failures += 1
                if self.setup_failures >= SETUP_FAIL_STOP:
                    self.close(f"{self.setup_failures} setup failures", by=u, kind="defect")
                continue
            b.rec["previous_archived"] = archive_previous(b.rd, self.stamp)
            b.wiped = b.result_written = False
            self.boxes.append(b)
            self.inflight.append(b)
            self.max_inflight = max(self.max_inflight, len(self.inflight))
            self.futs[self.ex.submit(self._setup, b)] = (b, "setup")

    # -- the workers ---------------------------------------------------------------------------------------------
    def _setup(self, b):
        with self.m.lock:
            b.prepare()
            b.up(init_root=False)
        b.probe()
        b.upload()
        b.toolchain()
        if not self.guard["admitting"] or self.stopping:
            b.status = "not_launched"
            b.rec["not_launched"] = self.guard["reason"] or "shutdown"
            b.save()
            log(f"{b.unit}: set up but NOT launched (the guard closed admission meanwhile)")
            return
        b.launch(self.token_for(b))

    def token_for(self, b) -> str:
        """The credential a unit's session starts on: the run's one credential."""
        return self.tok

    def _finish(self, b):
        try:
            b.collect(getattr(b, "tok_used", None) or self.tok)
        finally:
            b.wiped = b.wipe()
            b.res = b.result()
            b.result_written = True

    def _teardown(self, b):
        if b.status != "new":
            b.wiped = b.wipe()

    # -- the loop ------------------------------------------------------------------------------------------------
    def harvest(self):
        for f in [f for f in self.futs if f.done()]:
            b, kind = self.futs.pop(f)
            e = f.exception()
            if kind == "setup":
                if e is not None:
                    self.setup_failures += 1
                    b.rec["setup_error"] = f"{type(e).__name__}: {e}"
                    b.save()
                    self.errors.append(f"{b.unit}: setup: {e}")
                    log(f"{b.unit}: setup failed ({type(e).__name__}: {str(e)[:300]}); it stays for `rest`")
                    if self.setup_failures >= SETUP_FAIL_STOP:
                        self.close(f"{self.setup_failures} setup failures", by=b.unit, kind="defect")
                    self.futs[self.ex.submit(self._teardown, b)] = (b, "teardown")
                elif b.status == "not_launched":
                    self.futs[self.ex.submit(self._teardown, b)] = (b, "teardown")
                continue
            if e is not None:
                self.errors.append(f"{b.unit}: {kind}: {e}")
                log(f"{b.unit}: {kind}: {type(e).__name__}: {str(e)[:300]}")
                if isinstance(e, TokenHit):
                    self.close(f"token hit: {e}", by=b.unit, kind="defect")
            if b in self.inflight:
                self.inflight.remove(b)           # the slot is free only now: collected (or given up) and wiped

    @staticmethod
    def stop_kind(reasons: list) -> str:
        """"quota" when every reason is a usage-limit rejection / 429 one (a wait can end it), else "defect"."""
        return "quota" if reasons and all(r.startswith(("quota ", "HTTP 429")) for r in reasons) else "defect"

    def on_ended(self, b):
        g = judge(b, None, self.expect_skills)
        b.guard = g
        lr = g["last_rate_limit"] or {}
        self.readings.append({"utc": g["utc"], "unit": b.unit, "status": lr.get("status"),
                              "five_hour": (lr.get("five_hour") or {}).get("utilization"),
                              "seven_day": (lr.get("seven_day") or {}).get("utilization"),
                              "thinking_chars": g["thinking"]["chars"], "stop_reasons": g["stop_reasons"]})
        log(f"{b.unit}: guard reading 5h {self.readings[-1]['five_hour']} 7d {self.readings[-1]['seven_day']} "
            f"status {lr.get('status')}; thinking {g['thinking']['with_text']}/{g['thinking']['blocks']} blocks "
            f"with text, {g['thinking']['chars']} chars; reasons {g['stop_reasons']}")
        if g["stop_reasons"]:
            self.close("; ".join(g["stop_reasons"]), by=b.unit, kind=self.stop_kind(g["stop_reasons"]))
        b.status_ended = b.status
        self.futs[self.ex.submit(self._finish, b)] = (b, "collect")

    def run(self):
        last_note = 0.0
        while self.pending or self.inflight:
            self.harvest()
            self.admit()
            if not self.inflight:
                if not self.guard["admitting"] or not self.pending:
                    # the leg driver's hook: a rate-limit wait that ends with `reopen` keeps the queue going
                    if self.pending and not self.guard["admitting"] and self.on_idle_closed is not None \
                            and not self.stopping and self.on_idle_closed():
                        continue
                    break
                continue
            running = [b for b in self.inflight if b.status == "running"]
            got = False
            if running:
                ended, got = poll_once(self.m, running, self.wall_s, int(self.limits["rw_max_bytes"]), self.poll)
                for b in ended:
                    self.on_ended(b)
                self.m.rec["guard"], self.m.rec["guard_readings"] = self.guard, self.readings
                self.m.save()
            if time.time() - last_note > 300:
                last_note = time.time()
                st: dict = {}
                for b in self.inflight:
                    st[b.status] = st.get(b.status, 0) + 1
                done = sum(1 for b in self.boxes if b not in self.inflight)
                log(f"progress: in flight {len(self.inflight)} {st}; done {done}; pending {len(self.pending)}; "
                    f"admitting {self.guard['admitting']}; last reading "
                    f"{(self.readings[-1] if self.readings else {}).get('five_hour')}")
            if not got:
                time.sleep(self.poll_s if running else min(2.0, self.poll_s))

    def shutdown(self):
        """No more launches, the queued setups cancelled, the running workers waited for (each step is bounded).
        The guard record keeps what the GUARD decided: a queue that simply ran out of units stays 'admitting'."""
        self.stopping = True
        self.ex.shutdown(wait=True, cancel_futures=True)


def cmd_run(a) -> int:
    root = Path(a.out)
    sel = load_selection(root)
    units = select_units(a.tasks, sel, root)
    if not units:
        print("nothing to run")
        return 0
    rows = unit_rows(sel)
    probs = L.units_split_problems([rows[u] for u in units], a.split, sel["key"])
    if probs:
        raise Refuse(f"selection vs split: {probs[:5]}")
    for u in units:
        if not (L.run_dir(root, u) / "PREP.json").is_file():
            raise Refuse(f"{u}: not prepared (cc_solver.py prepare)")
        if _completed(root, u) and not a.force:
            raise Refuse(f"{u}: already has a completed RESULT.json (pass --force to run it again)")
    conc = int(a.concurrency)
    try:
        limits = L.limits_for(conc)
    except ValueError as e:
        raise Refuse(str(e))
    wall_s = int(a.wall_min) * 60
    ttl = int(a.ttl_minutes or L.TTL_MIN_DEFAULT)
    if ttl * 60 < wall_s + TTL_MARGIN_S + 600:
        raise Refuse(f"--ttl-minutes {ttl} leaves no room for one {a.wall_min}-min rollout")
    tok, cred, where = read_credential(a.token_file or None)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    m = Machine(root, stamp, ttl_minutes=ttl, run_dir=a.run_dir)
    m.token_name = where
    _LOG["fh"] = open(m.dir / "driver.log", "a")
    m.rec.update(selection=sel["sel_file"], split_sha256=sel.get("split_sha256"), key=sel["key"], units=units,
                 n_units=len(units), concurrency=conc, wall_min=a.wall_min, limits=limits,
                 credential={"kind": cred.kind, "from": where},
                 model=L.MODEL, max_turns=L.MAX_TURNS, thinking_display=L.THINKING_DISPLAY,
                 argv_flags=[x for x in L.claude_argv("U") if x.startswith("-")], env=L.cc_env(),
                 disallowed=list(L.CC_DISALLOWED))
    log(f"RUN {stamp}: {len(units)} unit(s) of {sel['sel_file']} (split[{sel['key']!r}]), concurrency {conc}, "
        f"limits {limits}, wall {a.wall_min} min, ttl {ttl} min, credential {cred.kind} ({where})")
    qu = None
    rc = 0
    try:
        m.up()
        m.install()
        qu = Queue(m, root, units, stamp, wall_s, limits, tok, conc)
        m.rec["queue_started_utc"] = L.now_utc()
        m.save()
        qu.run()
        m.rec["all_done_utc"] = L.now_utc()
        m.save()
        if qu.errors:
            rc = 5
        if not qu.guard["admitting"] or qu.pending:
            rc = rc or 9
    except Refuse as e:
        log(f"REFUSED: {e}")
        m.rec["refused"] = str(e)
        rc = 4
    except BaseException as e:  # noqa: BLE001 -- SIGTERM / Ctrl-C / crash: still tear everything down
        log(f"CRASH: {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
        m.rec["crash"] = f"{type(e).__name__}: {e}"
        rc = 6
    finally:
        boxes = qu.boxes if qu is not None else []
        if qu is not None:
            qu.shutdown()
        for b in boxes:
            if b.status != "new" and not b.wiped:
                b.wiped = b.wipe()
                if not b.wiped:
                    rc = rc or 7
            if b.t_launch and not b.result_written:
                try:
                    b.res = b.result()
                    b.result_written = True
                except Exception as e:  # noqa: BLE001
                    log(f"{b.unit}: result: {e}")
        if m.sb is not None or m.sid:
            d = m.delete()
            if not d.get("verified_gone"):
                rc = rc or 8
        m.rec["results"] = [{k: (getattr(b, "res", None) or {}).get(k) for k in
                             ("unit", "status", "completed", "subtype", "num_turns", "minutes", "cost",
                              "init_problems", "thinking_problems")} | {"unit": b.unit}
                            for b in boxes]
        m.rec["thinking_chars"] = {b.unit: ((getattr(b, "res", None) or {}).get("thinking") or {}).get("chars")
                                   for b in boxes}
        if qu is not None:
            m.rec.update(guard=qu.guard, guard_readings=qu.readings, queue_errors=qu.errors,
                         max_inflight=qu.max_inflight, not_started=list(qu.pending),
                         setup_failures=qu.setup_failures)
        m.save()
        log(f"RUN {stamp} rc {rc}: {sum(1 for r in m.rec['results'] if r.get('completed'))}/{len(units)} unit(s) "
            f"completed; guard {json.dumps(m.rec.get('guard'), default=str)[:600]}")
    return rc


# ───────────────────────────────────────────────────────────────────────────── grade
def job_label(sample_idx) -> str:
    """The grade job's label: one per sample, so the N samples of a task get N distinct job ids
    (<task>/saved/<label>/grade)."""
    return JOB_LABEL if sample_idx is None else f"{JOB_LABEL}-s{int(sample_idx)}"


def _check_only_estimate(jobs: Path, gdir: Path) -> dict:
    """ssb_grade_daytona's own estimate for these jobs (--check-only: nothing created, no key read)."""
    r = subprocess.run([sys.executable, str(GRADE), "run", "--jobs", str(jobs), "--out-dir", str(gdir / "check_only"),
                        "--label", GRADE_LABEL, "--check-only"], cwd=str(REPO), capture_output=True, text=True)
    (gdir / "check_only.out").write_text(r.stdout + r.stderr)
    body = r.stdout.split("CHECK-ONLY:")[0]
    try:
        return json.loads(body[body.index("{"):]).get("estimate") or {}
    except ValueError:
        return {}


def cmd_grade(a) -> int:
    root = Path(a.out)
    sel = load_selection(root)
    rows_by = unit_rows(sel)
    units = select_units(a.tasks, sel, root)
    probs = L.units_split_problems([rows_by[u] for u in units], a.split, sel["key"])
    if probs:
        raise Refuse(f"selection vs split: {probs[:5]}")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    gdir = root / "grade" / stamp
    gdir.mkdir(parents=True, exist_ok=True)
    saved, none, not_run, incomplete, jid_unit = [], [], [], [], {}
    for u in units:
        rp = L.run_dir(root, u) / "RESULT.json"
        if not rp.is_file():
            if a.tasks in ("all", "ungraded"):
                not_run.append(u)
                continue
            raise Refuse(f"{u}: no RESULT.json (not run)")
        res = json.loads(rp.read_text())
        if not res.get("completed"):
            incomplete.append(u)          # an infrastructure failure / a dead container: `run --tasks rest` redoes it
            continue
        t, k = rows_by[u]["task"], rows_by[u]["sample_idx"]
        o = res.get("output")
        if o and Path(o["path"]).is_file() and L.sha256_file(o["path"]) == o["sha256"]:
            lab = job_label(k)
            saved.append({"task": t, "output": o["path"], "label": lab, "modes": ["grade"],
                          "meta": {"source": "cc_solver", "unit": u, "machine_run": res.get("machine_run"),
                                   "output_sha256": o["sha256"]}})
            jid_unit[f"{t}/saved/{lab}/grade"] = u
        else:
            none.append(u)
    L.jdump(gdir / "saved.json", saved)
    L.jdump(gdir / "not_eligible.json", none)
    L.jdump(gdir / "skipped.json", {"not_run": not_run, "incomplete": incomplete})
    results = {}
    if saved:
        jobs = gdir / "jobs.jsonl"
        r = subprocess.run([sys.executable, str(GRADE), "make-jobs", "--kinds", "", "--saved-json",
                            str(gdir / "saved.json"), "--split", str(a.split), "--key", sel["key"],
                            "--out", str(jobs)], cwd=str(REPO), capture_output=True, text=True)
        (gdir / "make_jobs.out").write_text(r.stdout + r.stderr)
        if r.returncode != 0:
            raise Refuse(f"make-jobs rc {r.returncode}: {(r.stdout + r.stderr)[-600:]}")
        est = _check_only_estimate(jobs, gdir)
        hours = float(est.get("sandbox_hours_est") or 1.0)
        wall = int(a.wall_min) or max(90, int(hours * 60 * 1.5) + 15)
        usd = float(a.max_usd) or max(3.0, round(float(est.get("usd_est") or 1.0) * 2, 2))
        print(f"{len(saved)} grade job(s); estimate {est.get('sandbox_hours_est')} h / ${est.get('usd_est')} "
              f"({est.get('light')} light, {est.get('heavy')} heavy) -> --wall-min {wall} --max-usd {usd}")
        with open(gdir / "grade_run.out", "w") as fh:
            r = subprocess.run([sys.executable, str(GRADE), "run", "--jobs", str(jobs), "--out-dir", str(gdir),
                                "--label", GRADE_LABEL, "--wall-min", str(wall), "--max-usd", str(usd)],
                               cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT)
        print(f"grade run rc {r.returncode} (log {gdir / 'grade_run.out'})")
        for ln in (gdir / "results.jsonl").read_text().splitlines() if (gdir / "results.jsonl").is_file() else []:
            try:
                rec = json.loads(ln)
            except ValueError:
                continue
            u = jid_unit.get(str(rec.get("job_id") or ""))
            if rec.get("mode") == "grade" and u:
                results[u] = rec
    rows = []
    for u in units:
        if u in not_run:
            rows.append({"unit": u, "verdict": "not_run"})
            continue
        if u in incomplete:
            g = {"verdict": "not_graded_incomplete"}
        elif u in none:
            g = L.grade_brief(None)
        elif u in results:
            g = L.grade_brief(results[u])
        else:
            g = {"verdict": "not_graded"}
        g.update(unit=u, grade_dir=str(gdir), job_id=(results.get(u) or {}).get("job_id"))
        L.jdump(L.run_dir(root, u) / "GRADE.json", g)
        rows.append(g)
    L.jdump(gdir / "GRADES.json", rows)
    for r in rows:
        print(f"{r['unit']}: {r.get('verdict')} {r.get('n_match')}/{r.get('n_total')} {r.get('scorer_header') or ''}")
    bad = [r["unit"] for r in rows if r.get("verdict") not in L.MEASURED + ("not_run", "not_graded_incomplete")]
    if bad:
        print(f"NOT GRADED: {bad} (re-run `grade --tasks ungraded`)")
    return 5 if bad else 0


# ───────────────────────────────────────────────────────────────────────────── report / cleanup
def _machine_of(mrec: dict) -> dict:
    return {"credential": mrec.get("credential"), "concurrency": mrec.get("concurrency"),
            "guard": {k: (mrec.get("guard") or {}).get(k) for k in ("admitting", "reason", "at_utc", "by",
                                                                   "pending_at_stop")},
            "machine_deleted": (mrec.get("delete") or {}).get("verified_gone")}


def cmd_report(a) -> int:
    root = Path(a.out)
    sel = load_selection(root)
    rows_by = unit_rows(sel)
    units = select_units(a.tasks, sel, root)
    rows, runs = [], {}
    for u in units:
        rd = L.run_dir(root, u)
        res = json.loads((rd / "RESULT.json").read_text()) if (rd / "RESULT.json").is_file() else {}
        g = json.loads((rd / "GRADE.json").read_text()) if (rd / "GRADE.json").is_file() else {}
        rows.append({"unit": u, "task": rows_by[u]["task"], "sample_idx": rows_by[u]["sample_idx"],
                     "completed": res.get("completed"), "status": res.get("status"), "subtype": res.get("subtype"),
                     "is_error": res.get("is_error"), "terminal_reason": res.get("terminal_reason"),
                     "num_turns": res.get("num_turns"), "minutes": res.get("minutes"), "cost": res.get("cost"),
                     "thinking": res.get("thinking"), "grade": g,
                     "rl_first": (res.get("rate_limit") or {}).get("first"),
                     "rl_last": (res.get("rate_limit") or {}).get("last"), "init_problems": res.get("init_problems"),
                     "thinking_problems": res.get("thinking_problems"), "machine_run": res.get("machine_run")})
        if res.get("machine_run"):
            runs.setdefault(res["machine_run"], []).append(u)
    print(L.table(rows))
    samples = int(sel.get("samples") or 1)
    pt = L.per_task(rows)
    summ = L.pass_summary(rows, samples)
    if samples > 1:
        print("\n" + L.task_table(pt))
    costs = [r["cost"] for r in rows if isinstance(r.get("cost"), (int, float))]
    mins = [r["minutes"] for r in rows if isinstance(r.get("minutes"), (int, float))]
    summ.update(completed=sum(1 for r in rows if r.get("completed")), not_run=sum(1 for r in rows if not r["status"]),
                cost_usd_notional=round(sum(costs), 2), rollout_minutes=round(sum(mins), 1),
                thinking_chars=sum(((r.get("thinking") or {}).get("chars") or 0) for r in rows))
    print(f"\n{json.dumps(summ)}")
    machines = {}
    for run in sorted(runs):
        mp = root / "machine" / run / "MACHINE.json"
        machines[run] = dict(_machine_of(json.loads(mp.read_text()) if mp.is_file() else {}), units=len(runs[run]))
        mm = machines[run]
        print(f"run {run} ({mm['units']} unit(s), credential {mm['credential']}, concurrency {mm['concurrency']}): "
              f"guard {'open' if (mm['guard'] or {}).get('admitting') in (True, None) else mm['guard']}; machine "
              f"deleted {mm['machine_deleted']}")
    L.jdump(root / "REPORT.json", {"selection": sel["sel_file"], "split_sha256": sel.get("split_sha256"),
                                   "key": sel["key"], "rows": rows, "per_task": pt, "summary": summ,
                                   "machines": machines, "created_utc": L.now_utc()})
    return 0


def cmd_cleanup(a) -> int:
    import daytona_lib_evo as dle
    d = dle.client()
    mine = [r for r in dle.list_ours(d) if "-ccsolver-" in r["name"]]
    for r in mine:
        print(r["id"], r["name"], dle.delete_by_id(d, r["id"], TAG + "-cleanup"))
    print(json.dumps({"deleted_or_checked": mine, "listing_after": dle.list_all_ids(d)}, indent=1))
    return 0


def _on_sigterm(signum, frame):
    """SIGTERM -> SystemExit, so cmd_run's `finally` still wipes the containers and deletes the machine."""
    raise SystemExit(128 + signum)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sample", "prepare", "run", "grade", "report", "cleanup"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=name != "cleanup", help="the out root of one selection")
        if name in ("sample", "prepare", "run", "grade"):
            s.add_argument("--split", required=True, help="the split file the selection is drawn from")
        if name in ("prepare", "run"):
            s.add_argument("--run-dir", required=True,
                           help="a run dir whose INFRA.json names the leg image, data overlay and Claude Code pins")
        if name == "sample":
            s.add_argument("--kind", choices=("test10", "train", "val"), default="test10",
                           help="test10: sample10.json (seed 926, split test); train: selection_train.json (every "
                                "split['train'] task x --samples)")
            s.add_argument("--samples", type=int, default=L.TRAIN_SAMPLES)
        if name in ("prepare", "run", "grade", "report"):
            s.add_argument("--tasks", default="all", help="all | rest | ungraded | comma-separated unit ids "
                                                          "(<task>#s<k>) and/or task ids (= all their units)")
        if name == "run":
            s.add_argument("--concurrency", type=int, default=20, help="rollouts holding a container at once")
            s.add_argument("--token-file", default="",
                           help="a subscription token file (mode 0600); default $EVOSKILL_CC_TOKEN_FILE, else "
                                "$ANTHROPIC_API_KEY (never printed; reaches containers as fd 3 only)")
            s.add_argument("--wall-min", type=int, default=L.WALL_MIN)
            s.add_argument("--ttl-minutes", type=int, default=0, help=f"0 = {L.TTL_MIN_DEFAULT}")
            s.add_argument("--force", action="store_true")
        if name == "grade":
            s.add_argument("--wall-min", type=int, default=0, help="0 = from the grader's own estimate")
            s.add_argument("--max-usd", type=float, default=0.0, help="0 = from the grader's own estimate")
    a = ap.parse_args(argv)
    signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        return {"sample": cmd_sample, "prepare": cmd_prepare, "run": cmd_run, "grade": cmd_grade,
                "report": cmd_report, "cleanup": cmd_cleanup}[a.cmd](a)
    except Refuse as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
