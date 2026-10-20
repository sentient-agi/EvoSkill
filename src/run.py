#!/usr/bin/env python3
"""`evoskill`: one command that runs EvoSkill v2 end to end on the Daytona backend.

    evoskill run --instruction FILE --data DIR --split FILE --grader FILE --run-dir DIR --task-type {workbook,tau3}
                 [--solver-model M] [--evolver-model M] [--adaptor-model M]
                           (defaults: workbook claude-sonnet-5-5 for all three; tau3 solver claude-haiku-5-5,
                            evolver and adaptor claude-sonnet-5-5)
                 [--token-file FILE]        (else $EVOSKILL_CC_TOKEN_FILE, else $ANTHROPIC_API_KEY)
                 [--concurrency N]          (attempts in flight on the run's machine; default 30, at most 30)
                 [--stop-after {init,adapt,base,evolve,test}] [--dry-run]
    evoskill status --run-dir DIR
    evoskill stop   --run-dir DIR [--machine]

THE STEPS. Each is skipped when its recorded output is complete, so running the same command again resumes a stopped
run. A step that ran and left its output incomplete stops the run with the reason.

  1 init    The run dir and RUN.json (every input, hashed), INFRA.json, RUNTIME.json and the SPEC fields from
            src/run_templates/<task type>/ (the report's run configuration), the instruction and the split copied
            under given/, the user's grader as setup/benchmark_grader.py (where the benchmark adaptor and the plug-in
            freeze expect it). n_epochs and n_samples come from the instruction's "Run size" section
            ("3 epochs · 3 samples per task"); an instruction without one is refused. The SPEC fields are validated
            against the user's split here, before anything is spent.
  2 adapt   The benchmark adaptor: scripts/daytona/setup_ao_probe.py, one Claude Code turn on a throwaway Daytona
            sandbox (--message = the instruction, --grader, --guidance = src/orchestrator/prompts/
            BA_GUIDANCE.template.md, the full data view), which writes setup/benchmark_plugin.py and reference/. Then,
            as for the reported runs: `plugin_runner freeze` (setup/tasks.json, setup/split.json, setup/PLUGIN.json)
            and `plugin_runner verify`; the frozen split checked to hold the user's train, val and test tasks;
            REFERENCE_CHECK.json (reference/ holds no byte copy of a validation or held-out task's answer file, of
            the split or of the grader); SPEC.json frozen on setup/split.json; `infra check` (INFRA.json agrees with
            SPEC, RUNTIME, the drivers, the Dockerfiles, the data and the environment).
            SPEC.json is frozen here and not at init because the loop and the leg driver read the split through
            SPEC split_path, and that must be the frozen plug-in's split (its ids are the plug-in's task ids).
  3 base    The pinned Claude Code linux-x64 binary on this host (fetched once, sha-checked), the skills probe
            (`cc_leg_driver.py probe-skills`, which creates the run's one Daytona machine), then the no-skill legs on
            train and val (launched back to back; val waits while train runs) and on test, each through
            scripts/ssbv2_leg_cc.sh, waiting for each leg's <tag>.json (a <tag>.FAILED.json stops the run with
            its text).
  4 evolve  The epoch loop in the foreground: python -m src.orchestrator.loop --run-dir R --epoch 1 --step --drive
            --until-epoch N. It runs every epoch (evolver turn, skills_vK, train and val legs; the last epoch has no
            train leg) and ends the evolver session at DRIVE DONE. Resume = run again.
  5 test    The held-out leg with skills_vN, the summary (no skill vs skills_vN on the held-out tasks), then the
            run's Daytona machine is deleted (`cc_pool_host.py stop`).

A CHANGED INPUT IS REFUSED, never silently reused: the instruction, data, split and grader (by content), the task
type, the three models, the credential and the concurrency are fixed when the run dir is made. Start a new --run-dir
to change one.

THE ENVIRONMENT every child gets: the caller's, without any EVOSKILL_* variable (except EVOSKILL_PYTHON,
EVOSKILL_ENV_FILE and EVOSKILL_RATE_LIMIT_MAX_WAIT_S), ANTHROPIC_AUTH_TOKEN, ANTHROPIC_BASE_URL and SSBV2_ROOT; then
EVOSKILL_CLAUDE_EFFORT=default (Claude Code's own default effort), EVOSKILL_CC_VERSION and
EVOSKILL_EXPECT_CLAUDE_CODE_VERSION = the report's Claude Code release, DISABLE_AUTOUPDATER=1, and the credential:
EVOSKILL_CC_TOKEN_FILE (ANTHROPIC_API_KEY is then removed, as a token-file run never uses it) or ANTHROPIC_API_KEY.
A workbook run gets EVOSKILL_SSBV2_DATA = --data and loses OPENROUTER_API_KEY (nothing in it uses one); a tau3 run
keeps OPENROUTER_API_KEY for the simulated customer. DAYTONA_API_KEY and OPENROUTER_API_KEY are read by the drivers
from the environment, else from the repository's .env; this file never reads either value.

--dry-run prints every check, command and environment variable (a secret's value never) and runs nothing: no file
is written, no process started. `status` is offline. `stop` holds new legs (HOLD_LEGS), stops a running
`evoskill run` and its drive, and stops every live leg (their containers are wiped); it never ends the evolver
session or deletes the machine, which a later `evoskill run` resumes with. `stop --machine` also deletes the
machine, and is refused while an evolver session is open.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.orchestrator import cc_account, host_tools, infra, plugin_runner  # noqa: E402
from src.orchestrator.spec import ExperimentSpec, GateSpec, HeadroomGate, SpecError, split_digest  # noqa: E402

TEMPLATES = Path(__file__).resolve().parent / "run_templates"
TASK_TYPES = ("workbook", "tau3")
STEPS = ("init", "adapt", "base", "evolve", "test")
DEFAULT_MODEL = "claude-sonnet-5-5"
# The report's headline configuration per task type (solver, evolver, adaptor). tau3: the Haiku 5.5 solver evolved by
# a Sonnet 5.5 evolver; workbook: Sonnet 5.5 for all three.
DEFAULT_MODELS = {
    "workbook": {"solver": DEFAULT_MODEL, "evolver": DEFAULT_MODEL, "adaptor": DEFAULT_MODEL},
    "tau3": {"solver": "claude-haiku-5-5", "evolver": DEFAULT_MODEL, "adaptor": DEFAULT_MODEL},
}
DEFAULT_CONCURRENCY = 30                       # the report's: 30 attempts in flight on the run's one machine
MAX_CONCURRENCY = 30                           # scripts/ssbv2_leg_cc.sh refuses more
CC_VERSION = "2.1.293"                         # the report's Claude Code release (solver and evolver)

RUN_RECORD = "RUN.json"
RUN_SCHEMA = "evoskill.run/1"
STATE_DIR = ".evoskill"                        # <run>/.evoskill/: this command's lock and process record
GIVEN_DIR = "given"                            # <run>/given/: the user's instruction and split, as handed over
REFERENCE_CHECK = "REFERENCE_CHECK.json"
HOLD_FILE = "HOLD_LEGS"                        # scripts/ssbv2_leg_cc.sh launches no leg while it exists
HOLD_MARK = "evoskill stop"                    # the first words of a HOLD_LEGS this command wrote

LEG_SCRIPT = REPO / "scripts" / "ssbv2_leg_cc.sh"
LEG_DRIVER = REPO / "scripts" / "daytona" / "cc_leg_driver.py"
POOL_HOST = REPO / "scripts" / "daytona" / "cc_pool_host.py"
SETUP_AO = REPO / "scripts" / "daytona" / "setup_ao_probe.py"
BA_GUIDANCE = REPO / "src" / "orchestrator" / "prompts" / "BA_GUIDANCE.template.md"
OVERLAY = REPO / "examples" / "ssbv2" / "overlay" / "PATCHES.json"
TAU3_DOCKERFILE = "scripts/daytona/Dockerfile.tau3_leg"
CACHE_ROOT = REPO / ".cache"                   # cc_leg_lib.CACHE_ROOT_DEFAULT: <name>_e<N>_cache
STORE_ROOT = CACHE_ROOT / "ssbv2_daytona"      # cc_leg_lib.STORE_ROOT_DEFAULT: <name>/<tag>, <name>/pool/run
PROBE_RECORD = Path(".operator") / "CC_SKILLS_PROBE.json"
DISPATCH_LOCK = Path(".machine") / "evolver_dispatch.lock"    # tau3_evolver_daytona.hold_dispatch_lock
LEG_PID_DIR = Path(".machine") / "legs"                       # cc_leg_lib.LEG_PID_DIR
TEARDOWN_RECORD = "EVOLVER_SESSION_TEARDOWN.json"            # written when the drive ends the evolver session
# RUNTIME keys a template leaves null on purpose: the Daytona snapshot refs. Daytona derives a snapshot ref when it
# first builds a Dockerfile, so it cannot be known offline; null = not pinned (the Dockerfile's sha256 is the image's
# identity, and the ref Daytona reports is recorded in every receipt). INFRA's snapshot_ref fields are null with them.
UNPINNED_OK = ("evolver_daytona.require_snapshot_ref", "evolver_daytona.require_leg_snapshot_ref",
               "daytona_leg.require_snapshot_ref")

ADAPTOR_TIMEOUT_S = 3600                       # the reported adaptor runs' --timeout-s
MAX_DRIVE_STEPS = 600                          # the reported drives' --max-steps
LEG_POLL_S = 30
LEG_PROGRESS_S = 900
LEG_START_GRACE_S = 120                        # a just-launched leg may not hold its pid file yet
LEG_LAUNCH_GAP_S = 20                          # between the train and the val launch, as the reported launchers did
POOL_STOP_WAIT_S = 1800

KEEP_EVOSKILL = ("EVOSKILL_PYTHON", "EVOSKILL_ENV_FILE", "EVOSKILL_RATE_LIMIT_MAX_WAIT_S")
UNSET_ALWAYS = ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL", "SSBV2_ROOT", "EVO_CONC")
_SECRET_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD)", re.I)
DRIVE_EXIT = {1: "HALT", 2: "BLOCKED: the run needs an answer", 3: "an operator step", 4: "the step ceiling",
              64: "usage"}


class RunError(Exception):
    """A refusal or a failed step: the message says why; nothing after it runs."""


# ───────────────────────────────────────────────────────────────────────────── small helpers
def _sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _read_json(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def _write_json(p, obj) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, p)


def _strip_comments(x):
    if isinstance(x, dict):
        return {k: _strip_comments(v) for k, v in x.items() if not str(k).startswith("_")}
    return x


def template(task_type: str, name: str) -> dict:
    """src/run_templates/<task_type>/<name> without its "_" comment keys."""
    return _strip_comments(json.loads((TEMPLATES / task_type / name).read_text()))


def split_ids(sp: dict, key: str) -> list:
    """The ids under one split key: a list of ids or of {"id": ...} (loop._split_key_ids' shapes)."""
    v = sp.get(key)
    if isinstance(v, dict):
        v = v.get("task_ids")
    if not isinstance(v, list):
        return []
    return [x if isinstance(x, str) else x["id"] for x in v]


def tree_fingerprint(root: Path) -> dict | None:
    """(files, bytes, sha256 over every file's relative path and size) of a directory; None when it is missing.
    Cheap enough to take on every run, and it changes when a file is added, removed or resized."""
    root = Path(root)
    if not root.is_dir():
        return None
    h, n, total = hashlib.sha256(), 0, 0
    for p in sorted(root.rglob("*")):
        if p.is_file():
            size = p.stat().st_size
            h.update(f"{p.relative_to(root).as_posix()}\0{size}\n".encode())
            n, total = n + 1, total + size
    return {"files": n, "bytes": total, "sha256": h.hexdigest()}


def _lock_held(path: Path) -> bool:
    """True when another process holds a flock on `path` (the kernel drops it when that process ends)."""
    try:
        fh = open(path)
    except OSError:
        return False
    with fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError:
            return True
        fcntl.flock(fh, fcntl.LOCK_UN)
        return False


def _exclusive_free(path: Path) -> bool:
    """True when nobody holds `path` (a shared holder blocks an exclusive try). Never blocks."""
    try:
        fh = open(path)
    except OSError:
        return True
    with fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        fcntl.flock(fh, fcntl.LOCK_UN)
        return True


def _cmdline(pid: int) -> str:
    try:
        r = subprocess.run(["ps", "-p", str(int(pid)), "-o", "command="], capture_output=True, text=True)
    except OSError:
        return ""
    return r.stdout.strip()


def _env_file() -> Path:
    return Path(os.environ.get("EVOSKILL_ENV_FILE") or (REPO / ".env"))


def _secret_source(name: str, env: dict) -> str | None:
    """Where a driver would find `name`: "environment", "<.env path> (present; not read here)", or None. The
    .env file is never opened here."""
    if (env.get(name) or "").strip():
        return "the environment"
    f = _env_file()
    if f.is_file():
        return f"{f} (present; the driver reads it, this command does not)"
    return None


# ───────────────────────────────────────────────────────────────────────────── the instruction
_RUN_SIZE_HEAD = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]*(?:\d+[.)][ \t]*)?run size\b.*$", re.I | re.M)
_ANY_HEAD = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]", re.M)
_EPOCHS = re.compile(r"(\d+)\s*epochs?\b", re.I)
_SAMPLES = re.compile(r"(\d+)\s*(?:samples?|attempts?)\s+(?:per|for each|each)\s+task", re.I)


def parse_run_size(text: str) -> tuple:
    """(n_epochs, n_samples) from the instruction's "Run size" section, e.g. a heading "### 5. Run size" followed
    by "3 epochs · 3 samples per task". RunError when the section or either number is missing."""
    m = _RUN_SIZE_HEAD.search(text)
    if not m:
        raise RunError("the instruction has no 'Run size' section: add a heading such as '### 5. Run size' "
                       "followed by a line such as '3 epochs · 3 samples per task'")
    rest = text[m.end():]
    nxt = _ANY_HEAD.search(rest)
    body = rest[:nxt.start()] if nxt else rest
    e, s = _EPOCHS.search(body), _SAMPLES.search(body)
    if not e or not s:
        raise RunError(f"the instruction's Run size section must give both numbers, as in '3 epochs · 3 samples "
                       f"per task'; it says: {body.strip()[:200]!r}")
    n_epochs, n_samples = int(e.group(1)), int(s.group(1))
    if n_epochs < 1 or n_samples < 1:
        raise RunError(f"Run size: {n_epochs} epochs, {n_samples} samples per task (both must be at least 1)")
    return n_epochs, n_samples


_DATA_PATH = re.compile(r"(?<![\w/.])/data/([^\s/`'\"()\[\],;:<>]+)")


def instruction_path_problems(text: str, data: Path, split: Path, repo: Path = REPO) -> list:
    """The /data/<name> paths the instruction names that the adaptor's machine will not have. What the user hands
    over sits there by its own name (plugin_runner.material_layout): --data at /data/<its folder name>, --split at
    /data/<its file name>, the repository at /data/EvoSkill."""
    known = {data.name, split.name, plugin_runner.REPO_VIEW_NAME}
    named = sorted({n.rstrip(".") for n in _DATA_PATH.findall(text)} - {""})
    return [f"the instruction names /data/{n}, but the adaptor's machine has "
            f"{', '.join('/data/' + k for k in sorted(known))} (--data is uploaded as /data/<its folder name>, "
            f"--split as /data/<its file name>)" for n in named if n not in known]


# ───────────────────────────────────────────────────────────────────────────── the plan
@dataclass
class Plan:
    """Everything the command line decides, before anything is written."""
    run_dir: Path
    task_type: str
    instruction: Path
    data: Path
    split: Path
    grader: Path
    solver_model: str
    evolver_model: str
    adaptor_model: str
    concurrency: int
    credential: dict          # {"kind": "token_file" | "api_key", "token_file": path or None}
    n_epochs: int
    n_samples: int
    problems: tuple = ()      # what make_plan(strict=False) could not resolve (a dry run reports it)

    @property
    def name(self) -> str:
        return self.run_dir.name


def resolve_credential(token_file, env: dict) -> dict:
    """The run's one Claude Code credential (cc_account.resolve: --token-file, else $EVOSKILL_CC_TOKEN_FILE, else
    $ANTHROPIC_API_KEY). A token file is checked by stat (a regular file, mode 0600) and never opened here."""
    try:
        c = cc_account.resolve(str(token_file) if token_file else None, env)
    except ValueError as e:
        raise RunError(f"credential: {e}")
    return {"kind": c.kind, "token_file": str(Path(c.token_file).absolute()) if c.token_file else None}


def make_plan(a, env: dict, *, strict: bool = True) -> Plan:
    """The Plan of `evoskill run`'s arguments. strict=False (a dry run) records an unresolvable credential in
    Plan.problems instead of refusing."""
    for what, p in (("--instruction", a.instruction), ("--split", a.split), ("--grader", a.grader)):
        if not Path(p).is_file():
            raise RunError(f"{what} {p}: no such file")
    instruction = Path(a.instruction).resolve()
    n_epochs, n_samples = parse_run_size(instruction.read_text(encoding="utf-8"))
    problems = []
    try:
        cred = resolve_credential(a.token_file, env)
    except RunError as e:
        if strict:
            raise
        cred, problems = {"kind": None, "token_file": None}, [str(e)]
    return Plan(run_dir=Path(a.run_dir).resolve(), task_type=a.task_type, instruction=instruction,
                data=Path(a.data).resolve(), split=Path(a.split).resolve(), grader=Path(a.grader).resolve(),
                solver_model=a.solver_model or DEFAULT_MODELS[a.task_type]["solver"],
                evolver_model=a.evolver_model or DEFAULT_MODELS[a.task_type]["evolver"],
                adaptor_model=a.adaptor_model or DEFAULT_MODELS[a.task_type]["adaptor"],
                concurrency=int(a.concurrency), credential=cred, n_epochs=n_epochs, n_samples=n_samples,
                problems=tuple(problems))


def overlay_for(plan: Plan) -> Path | None:
    """The repaired-file overlay of the reported SpreadsheetBench v2 runs (examples/ssbv2/overlay/PATCHES.json) when
    --data holds every file it repairs; otherwise none. Declared in INFRA data and re-verified by `infra check`."""
    if plan.task_type != "workbook" or not OVERLAY.is_file() or not plan.data.is_dir():
        return None
    paths = [f["path"] for f in json.loads(OVERLAY.read_text())["files"]]
    return OVERLAY if paths and all((plan.data / p).is_file() for p in paths) else None


def inputs_of(plan: Plan) -> dict:
    """What a run is made from. Any later difference is refused (check_inputs)."""
    fp = tree_fingerprint(plan.data)
    return {"task_type": plan.task_type,
            "instruction_sha256": _sha256(plan.instruction), "split_sha256": _sha256(plan.split),
            "grader_sha256": _sha256(plan.grader),
            "data_root": str(plan.data), "data_fingerprint": fp["sha256"] if fp else None,
            "solver_model": plan.solver_model, "evolver_model": plan.evolver_model,
            "adaptor_model": plan.adaptor_model, "concurrency": plan.concurrency,
            "credential": dict(plan.credential)}


def check_inputs(recorded: dict, now: dict) -> None:
    changed = [k for k in sorted(set(recorded) | set(now)) if recorded.get(k) != now.get(k)]
    if changed:
        lines = []
        for k in changed:
            was, is_ = recorded.get(k), now.get(k)
            if k.endswith(("_sha256", "_fingerprint")):
                lines.append(f"{k}: the content changed")
            else:
                lines.append(f"{k}: was {was!r}, now {is_!r}")
        raise RunError("this run dir was made from other inputs; a run's inputs are fixed when it is made "
                       "(start a new --run-dir to change one):\n  " + "\n  ".join(lines))


def run_env(plan: Plan, base: dict) -> dict:
    """The environment of every child (module doc: THE ENVIRONMENT)."""
    env = {k: v for k, v in base.items()
           if not (k.startswith("EVOSKILL_") and k not in KEEP_EVOSKILL) and k not in UNSET_ALWAYS}
    env.update({"EVOSKILL_CLAUDE_EFFORT": "default", "EVOSKILL_CC_VERSION": CC_VERSION,
                "EVOSKILL_EXPECT_CLAUDE_CODE_VERSION": CC_VERSION, "DISABLE_AUTOUPDATER": "1"})
    # the interpreter of every child (the leg launcher reads it too): $EVOSKILL_PYTHON, else this one
    env["EVOSKILL_PYTHON"] = (base.get(host_tools.PYTHON_ENV) or "").strip() or sys.executable
    if plan.credential["kind"] == cc_account.TOKEN_FILE:
        env[cc_account.TOKEN_FILE_ENV] = plan.credential["token_file"]
        env.pop(cc_account.API_KEY_ENV, None)
    if plan.task_type == "workbook":
        env["EVOSKILL_SSBV2_DATA"] = str(plan.data)
        env.pop("OPENROUTER_API_KEY", None)
    return env


def preflight(plan: Plan, env: dict) -> list:
    """Every offline check of the plan; [] = nothing to refuse. Reads files, never opens a secret."""
    probs = list(plan.problems)
    if not plan.data.is_dir():
        probs.append(f"--data {plan.data}: no such directory")
    probs += instruction_path_problems(plan.instruction.read_text(encoding="utf-8"), plan.data, plan.split)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", plan.name):
        probs.append(f"--run-dir: its folder name {plan.name!r} names the run and must be letters, digits, _ or -")
    for what, m in (("--solver-model", plan.solver_model), ("--evolver-model", plan.evolver_model),
                    ("--adaptor-model", plan.adaptor_model)):
        if not re.fullmatch(r"claude-[a-z0-9.-]+", m or ""):
            probs.append(f"{what} {m!r}: an explicit claude-* model id")
    if not 1 <= plan.concurrency <= MAX_CONCURRENCY:
        probs.append(f"--concurrency {plan.concurrency}: between 1 and {MAX_CONCURRENCY}")
    g = plugin_runner.static_check_grader(plan.grader.read_text(encoding="utf-8"))
    probs += [f"grader {plan.grader.name}: {p}" for p in g["problems"]]
    try:
        probs += [f"spec: {e}" for e in spec_for(plan, plan.split).validate()]
    except (KeyError, ValueError, TypeError) as e:
        probs.append(f"spec: the split {plan.split.name} is not a split file ({type(e).__name__}: {e})")
    if not _secret_source("DAYTONA_API_KEY", env):
        probs.append(f"DAYTONA_API_KEY: set it in the environment or in {_env_file()}")
    if plan.task_type == "tau3" and not _secret_source("OPENROUTER_API_KEY", env):
        probs.append(f"OPENROUTER_API_KEY (the simulated customer): set it in the environment or in {_env_file()}")
    if not (plan.run_dir / RUN_RECORD).is_file():
        clash = sorted(p.name for p in (*CACHE_ROOT.glob(f"{plan.name}_e*_cache"), STORE_ROOT / plan.name)
                       if p.exists())
        if clash:
            probs.append(f"a run named {plan.name!r} already has caches under {CACHE_ROOT} ({clash[:3]}); the caches "
                         f"are keyed by the run dir's folder name, so pick another name")
    return probs


# ───────────────────────────────────────────────────────────────────────────── the run's files
def spec_for(plan: Plan, split_path: Path) -> ExperimentSpec:
    """The run's ExperimentSpec on `split_path`: the template's fields, the split's sizes, the instruction's run size
    and the models. gate.n_comparisons covers every epoch (spec.validate's rule)."""
    t = template(plan.task_type, "SPEC.json")
    sp = json.loads(Path(split_path).read_text())
    gate = dict(t.pop("gate"))
    gate["tests"] = tuple(gate["tests"])
    gate["n_comparisons"] = max(int(gate["n_comparisons"]), plan.n_epochs)
    headroom = t.pop("headroom")
    t["solver_provider_order"] = tuple(t.get("solver_provider_order") or ())
    return ExperimentSpec(name=plan.name, split_path=str(split_path), split_sha256=split_digest(split_path),
                          n_train=len(split_ids(sp, "train")), n_val=len(split_ids(sp, "val")),
                          n_test=len(split_ids(sp, "test")), n_samples=plan.n_samples, n_epochs=plan.n_epochs,
                          solver_model=plan.solver_model, evolver_model=plan.evolver_model,
                          gate=GateSpec(**gate), headroom=HeadroomGate(**headroom), **t)


def infra_for(plan: Plan) -> dict:
    """INFRA.json: the task type's template with this run's name, models, credential, data and concurrency."""
    inf = template(plan.task_type, "INFRA.json")
    tok = plan.credential["token_file"]
    inf["run"] = plan.name
    inf["solver"].update(model=plan.solver_model, expect_served_model=plan.solver_model, key_file=tok)
    inf["evolver"]["model"] = plan.evolver_model
    inf["ao"]["model"] = plan.adaptor_model
    inf["claude_code"]["token_file"] = tok
    ov = overlay_for(plan)
    inf["data"] = {"root": str(plan.data), "overlay_patches_json": str(ov) if ov else None,
                   "overlay_patches_sha256": _sha256(ov) if ov else None}
    inf["leg_sandbox"]["concurrency"] = plan.concurrency
    if plan.task_type == "tau3":
        sha = _sha256(REPO / TAU3_DOCKERFILE)
        inf["leg_sandbox"].update(dockerfile_sha256=sha, image_id="x86daytona-" + sha[:12])
    probs = infra.validate(inf)
    if probs:
        raise RunError(f"the INFRA.json made from src/run_templates/{plan.task_type} does not validate: {probs}")
    return {"_note": f"Written by `evoskill run` from src/run_templates/{plan.task_type}/INFRA.json.", **inf}


def runtime_for(plan: Plan) -> dict:
    """RUNTIME.json: the task type's template with this run's paths, models and concurrency."""
    rt = template(plan.task_type, "RUNTIME.json")
    rd, name = plan.run_dir, plan.name
    rt.update(leg_script=str(LEG_SCRIPT), shared_dir=str(rd), oracle=str(rd / "oracle"),
              prompt_author_model=plan.adaptor_model,
              base_leg=str(rd / f"{name}_base_train.json"), base_leg_val=str(rd / f"{name}_base_val.json"),
              heldout_base_leg=str(rd / f"{name}_base_test.json"))
    rt["claude"].update(ao_model=plan.adaptor_model, evolver_model_from_spec=plan.evolver_model)
    rt["cc_leg"]["concurrency"] = plan.concurrency
    rt["daytona_leg"]["concurrency"] = plan.concurrency
    left = [k for k, v in rt.items() if v is None] + [f"{b}.{k}" for b, blk in rt.items() if isinstance(blk, dict)
                                                       for k, v in blk.items() if v is None]
    left = [k for k in left if k not in UNPINNED_OK]
    if left:
        raise RunError(f"src/run_templates/{plan.task_type}/RUNTIME.json leaves {left} unfilled")
    return {"_note": f"Written by `evoskill run` from src/run_templates/{plan.task_type}/RUNTIME.json.", **rt}


def split_problems(frozen: dict, user: dict) -> list:
    """The frozen plug-in's train / val / test against the user's split. An id matches when it is the user's id, or
    the user's id after the plug-in's domain prefix (tau3: banking_knowledge/task_004 vs task_004)."""
    every = {t for k in ("train", "val", "test") for t in split_ids(user, k)}
    probs = []
    for k in ("train", "val", "test"):
        got = {t if t in every else t.rsplit("/", 1)[-1] for t in split_ids(frozen, k)}
        want = set(split_ids(user, k))
        if got != want:
            probs.append(f"{k}: the adaptor's plug-in puts {len(got)} tasks there, the given split {len(want)}; "
                         f"{sorted(got - want)[:3]} not in the split, {sorted(want - got)[:3]} missing")
    return probs


def reference_check(run_dir: Path, grader: Path, split: Path) -> dict:
    """reference/ (what the evolver will read) against the frozen task list: no byte copy of a validation or held-out
    task's answer file, of the split file or of the grader. The record REFERENCE_CHECK.json holds."""
    ref = run_dir / "reference"
    tasks = json.loads((run_dir / "setup" / plugin_runner.TASKS_JSON).read_text())
    forbidden = {_sha256(split): "the split file", _sha256(grader): "the grader"}
    for t in tasks:
        if t.get("split") in ("validation", "test"):
            for f in t.get("answer_files") or []:
                if Path(f).is_file():
                    forbidden.setdefault(_sha256(f), f"an answer file of {t['split']} task {t['id']}")
    rec = {"reference_dir": str(ref), "files": 0, "top_level": [], "forbidden_hashes": len(forbidden),
           "problems": [], "checked_utc": _now()}
    if not ref.is_dir():
        rec["problems"].append("reference/ is missing (the adaptor writes it)")
    else:
        files = sorted(p for p in ref.rglob("*") if p.is_file() and not p.is_symlink())
        rec["files"] = len(files)
        rec["top_level"] = sorted({p.relative_to(ref).parts[0] for p in files})
        for p in files:
            what = forbidden.get(_sha256(p))
            if what:
                rec["problems"].append(f"reference/{p.relative_to(ref).as_posix()} is {what}")
    rec["ok"] = not rec["problems"]
    return rec


# ───────────────────────────────────────────────────────────────────────────── commands
@dataclass
class Cmd:
    argv: list
    extra: dict               # variables on top of the run's environment
    what: str
    log: Path | None = None   # also written here (appended)

    def text(self) -> str:
        return " ".join(shlex.quote(str(x)) for x in self.argv)


def _shown(name: str, value) -> str:
    if value is None:
        return f"{name} (unset)"
    if _SECRET_NAME.search(name) and not name.endswith("_FILE"):
        return f"{name}=(set; not shown)"
    return f"{name}={value}"


def env_lines(base: dict, env: dict) -> list:
    """The difference between the caller's environment and the run's, one variable per line; secrets by name only."""
    out = [_shown(k, env[k]) for k in sorted(env) if base.get(k) != env[k]]
    out += [_shown(k, None) for k in sorted(base) if k not in env]
    return out


# ───────────────────────────────────────────────────────────────────────────── the run
class Run:
    """The five steps of one run dir. `out` prints, `sleep` waits; both are replaced in the tests."""

    def __init__(self, plan: Plan, env: dict, *, out=print, sleep=time.sleep):
        self.plan, self.env, self.out, self.sleep = plan, env, out, sleep
        self.rd = plan.run_dir
        self.py = env["EVOSKILL_PYTHON"]
        self.drive_proc = None

    # ── paths
    def leg_path(self, tag: str, suffix: str = ".json") -> Path:
        return self.rd / f"{tag}{suffix}"

    def tag(self, which: str, key: str) -> str:
        return f"{self.plan.name}_{which}_{key}"

    @property
    def frozen_split(self) -> Path:
        return self.rd / "setup" / plugin_runner.SPLIT_JSON

    def given(self, p: Path) -> Path:
        return self.rd / GIVEN_DIR / p.name

    # ── the steps: (check, run, describe) per step
    def steps(self) -> list:
        return [(s, getattr(self, f"check_{s}"), getattr(self, f"run_{s}"), getattr(self, f"describe_{s}"))
                for s in STEPS]

    # 1 init ─────────────────────────────────────────────────────────────────
    def init_outputs(self) -> dict:
        return {"RUN.json": self.rd / RUN_RECORD, "INFRA.json": self.rd / "INFRA.json",
                "RUNTIME.json": self.rd / "RUNTIME.json",
                "setup/benchmark_grader.py": self.rd / "setup" / "benchmark_grader.py",
                f"{GIVEN_DIR}/{self.plan.instruction.name}": self.given(self.plan.instruction),
                f"{GIVEN_DIR}/{self.plan.split.name}": self.given(self.plan.split)}

    def check_init(self):
        rec = _read_json(self.rd / RUN_RECORD)
        if rec is None:
            return False, "no RUN.json"
        check_inputs(rec.get("inputs") or {}, inputs_of(self.plan))
        missing = [k for k, p in self.init_outputs().items() if not p.is_file()]
        if missing:
            return False, f"missing {missing}"
        try:
            infra.load(self.rd / "INFRA.json")
        except infra.InfraError as e:
            raise RunError(f"INFRA.json does not validate: {e}")
        return True, f"{self.plan.n_epochs} epochs x {self.plan.n_samples} samples, task type {self.plan.task_type}"

    def run_init(self):
        p = self.plan
        if self.rd.exists() and any(x.name != STATE_DIR for x in self.rd.iterdir()) \
                and not (self.rd / RUN_RECORD).is_file():
            raise RunError(f"{self.rd} is not empty and holds no {RUN_RECORD}: not a run dir made by `evoskill run`")
        probs = preflight(p, self.env)
        if probs:
            raise RunError("refusing to start:\n  " + "\n  ".join(probs))
        if p.instruction.name == p.split.name:
            raise RunError(f"--instruction and --split share the file name {p.split.name!r}")
        self.rd.mkdir(parents=True, exist_ok=True)
        if not (self.rd / RUN_RECORD).is_file():
            ov = overlay_for(p)
            _write_json(self.rd / RUN_RECORD, {
                "schema": RUN_SCHEMA, "made_utc": _now(), "inputs": inputs_of(p),
                "sources": {"instruction": str(p.instruction), "data": str(p.data), "split": str(p.split),
                            "grader": str(p.grader)},
                "run_size": {"n_epochs": p.n_epochs, "n_samples": p.n_samples},
                "data_overlay": str(ov) if ov else None, "claude_code": CC_VERSION})
        for src in (p.instruction, p.split):
            if not self.given(src).is_file():
                self.given(src).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, self.given(src))
        g = self.rd / "setup" / "benchmark_grader.py"
        if not g.is_file():
            g.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p.grader, g)
        for name, make in (("INFRA.json", infra_for), ("RUNTIME.json", runtime_for)):
            if not (self.rd / name).is_file():
                _write_json(self.rd / name, make(p))
        self.out(f"  run dir {self.rd}: {p.n_epochs} epochs x {p.n_samples} samples; overlay "
                 f"{overlay_for(p) or 'none'}; credential {p.credential['kind']}")

    def describe_init(self):
        p = self.plan
        lines = [f"create {self.rd} with RUN.json (the inputs, hashed), INFRA.json and RUNTIME.json from "
                 f"src/run_templates/{p.task_type}/, {GIVEN_DIR}/{p.instruction.name}, {GIVEN_DIR}/{p.split.name} "
                 f"and setup/benchmark_grader.py (= {p.grader})",
                 f"run size from the instruction: {p.n_epochs} epochs x {p.n_samples} samples per task",
                 f"data overlay: {overlay_for(p) or 'none'}",
                 f"credential: {p.credential['kind']}" + (f" {p.credential['token_file']} (mode checked by stat)"
                                                          if p.credential["token_file"] else "")]
        if p.task_type == "tau3":
            lines.append(f"leg image: {TAU3_DOCKERFILE} sha256 {_sha256(REPO / TAU3_DOCKERFILE)[:12]} (on disk)")
        return lines

    # 2 adapt ────────────────────────────────────────────────────────────────
    def adaptor_cmd(self) -> Cmd:
        # the adaptor's sandbox settings: the run's evolver_daytona block without what only the evolver's container
        # reads (as the reported adaptor launchers passed it)
        rt = _read_json(self.rd / "RUNTIME.json") or template(self.plan.task_type, "RUNTIME.json")
        ed = {k: v for k, v in rt["evolver_daytona"].items() if not k.startswith("_")
              and k not in ("placement", "container", "machine_wait_s", "placement_from_epoch")}
        # The adaptor runs on the Claude sandbox image, whose Claude Code is daytona_evolver.CC_VERSION.
        vm_cc = infra._constants(REPO / infra.EVOLVER_MODULE, ("CC_VERSION",)).get("CC_VERSION") or CC_VERSION
        return Cmd([self.py, str(SETUP_AO), "--run-dir", str(self.rd),
                    "--message", str(self.given(self.plan.instruction)), "--guidance", str(BA_GUIDANCE),
                    "--grader", str(self.rd / "setup" / "benchmark_grader.py"),
                    "--split", str(self.given(self.plan.split)), "--data-view", "full",
                    "--model", self.plan.adaptor_model, "--timeout-s", str(ADAPTOR_TIMEOUT_S)],
                   {"EVOSKILL_EVOLVER_DAYTONA_JSON": json.dumps(ed, sort_keys=True),
                    "EVOSKILL_EXPECT_CLAUDE_CODE_VERSION": vm_cc},
                   "the benchmark adaptor (one Claude Code turn on a Daytona sandbox)",
                   log=self.rd / "setup_probe" / "launch_ao.out")

    def plugin_cmd(self, verb: str) -> Cmd:
        return Cmd([self.py, "-m", "src.orchestrator.plugin_runner", verb, "--run-dir", str(self.rd)], {},
                   f"plug-in {verb}")

    def infra_cmd(self) -> Cmd:
        return Cmd([self.py, "-m", "src.orchestrator.infra", "check", "--run-dir", str(self.rd)], {},
                   "INFRA.json against everything that states the same facts")

    def check_adapt(self):
        try:
            plugin_runner.verify_frozen(self.rd)
        except plugin_runner.PluginDrift as e:
            return False, f"plug-in not frozen: {str(e)[:200]}"
        rc = _read_json(self.rd / REFERENCE_CHECK)
        if not (rc or {}).get("ok"):
            return False, f"no passing {REFERENCE_CHECK}"
        try:
            ExperimentSpec.assert_unmutated(self.rd)
            spec, _ = ExperimentSpec.load(self.rd)
        except SpecError as e:
            return False, f"SPEC.json: {e}"
        if Path(spec.split_path).resolve() != self.frozen_split.resolve():
            raise RunError(f"SPEC.json split_path {spec.split_path} is not the frozen plug-in's {self.frozen_split}")
        return True, (f"plug-in frozen, {spec.n_train}/{spec.n_val}/{spec.n_test} train/val/test tasks, "
                      f"reference/ {rc.get('files')} files")

    def run_adapt(self):
        setup = self.rd / "setup"
        if not (setup / plugin_runner.PLUGIN_JSON).is_file():
            have_plugin = (setup / "benchmark_plugin.py").is_file()
            have_ref = (self.rd / "reference").exists()
            if have_plugin != have_ref:
                part = "setup/benchmark_plugin.py" if have_plugin else "reference/"
                raise RunError(f"the adaptor's output is partial (only {part} exists); remove it to run the adaptor "
                               f"again")
            if not have_plugin:
                self.exec(self.adaptor_cmd(), "the benchmark adaptor")
                if not (setup / "benchmark_plugin.py").is_file() or not (self.rd / "reference").is_dir():
                    raise RunError(f"the adaptor wrote no setup/benchmark_plugin.py or reference/; see "
                                   f"{self.rd / 'setup_probe' / 'setup_ao.log'}")
            self.exec(self.plugin_cmd("freeze"), "plugin_runner freeze")
        self.exec(self.plugin_cmd("verify"), "plugin_runner verify", quiet=True)
        probs = split_problems(json.loads(self.frozen_split.read_text()),
                               json.loads(self.given(self.plan.split).read_text()))
        if probs:
            raise RunError("the frozen plug-in's split is not the given split:\n  " + "\n  ".join(probs))
        rec = reference_check(self.rd, self.rd / "setup" / "benchmark_grader.py", self.given(self.plan.split))
        _write_json(self.rd / REFERENCE_CHECK, rec)
        if not rec["ok"]:
            raise RunError("reference/ holds material the evolver must not see:\n  " + "\n  ".join(rec["problems"][:20]))
        spec = spec_for(self.plan, self.frozen_split)
        errs = spec.validate()
        if errs:
            raise RunError(f"the run's spec does not validate on the frozen split: {errs}")
        try:
            spec.freeze(self.rd)
        except SpecError as e:
            raise RunError(str(e))
        self.exec(self.infra_cmd(), "infra check")

    def describe_adapt(self):
        return [self.adaptor_cmd(), self.plugin_cmd("freeze"), self.plugin_cmd("verify"),
                "(here) the frozen setup/split.json must hold the given split's train, val and test tasks",
                f"(here) {REFERENCE_CHECK}: reference/ holds no byte copy of a val/test answer file, the split or "
                f"the grader",
                "(here) SPEC.json frozen on setup/split.json", self.infra_cmd()]

    # 3 base ─────────────────────────────────────────────────────────────────
    def leg_env(self, epoch: int) -> dict:
        return {"EVOSKILL_TAU3_RUN_DIR": str(self.rd), "EVOSKILL_TAU3_SPLIT": str(self.frozen_split),
                "EVOSKILL_CACHE_DIR": str(CACHE_ROOT / f"{self.plan.name}_e{int(epoch)}_cache"),
                "EVO_CONC": str(self.plan.concurrency), "EVOSKILL_SOLVER_MODEL": self.plan.solver_model,
                "EVOSKILL_N_SAMPLES": str(self.plan.n_samples)}

    def leg_cmd(self, tag: str, key: str, skills: str, epoch: int) -> Cmd:
        argv = ["bash", str(LEG_SCRIPT), tag, key] + ([skills] if skills else [])
        return Cmd(argv, self.leg_env(epoch), f"the {'no-skill' if not skills else skills} {key} leg {tag}")

    def cc_binary_cmd(self) -> Cmd:
        code = ("import sys; from pathlib import Path; sys.path.insert(0, 'scripts/daytona'); "
                "import evolver_container as E; print(E.ensure_cc_binary({}, Path(sys.argv[1])))")
        return Cmd([self.py, "-c", code, str(self.rd)], {},
                   f"the pinned Claude Code {CC_VERSION} linux-x64 binary on this host (fetched once, sha-checked)")

    def probe_cmd(self) -> Cmd:
        return Cmd([self.py, str(LEG_DRIVER), "probe-skills", "--run-dir", str(self.rd)], {},
                   "the skills probe (creates the run's Daytona machine)")

    def base_legs(self) -> list:
        """[(group of (tag, key, skills, epoch))]: train and val launched together, then test."""
        return [[(self.tag("base", "train"), "train", "", 0), (self.tag("base", "val"), "val", "", 0)],
                [(self.tag("base", "test"), "test", "", 0)]]

    def probe_ok(self) -> bool:
        rec = _read_json(self.rd / PROBE_RECORD) or {}
        return bool((rec.get("primary") or {}).get("ok"))

    def check_base(self):
        missing = [t for g in self.base_legs() for t, *_ in g if not self.leg_path(t).is_file()]
        return (not missing), (f"missing {missing}" if missing else "no-skill train, val and test legs measured")

    def run_base(self):
        self.release_hold()
        self.exec(self.cc_binary_cmd(), "the Claude Code binary", quiet=True)
        if self.probe_ok():
            self.out(f"  skills probe record present ({PROBE_RECORD})")
        else:
            self.exec(self.probe_cmd(), "probe-skills")
        for group in self.base_legs():
            self.run_legs(group)

    def describe_base(self):
        lines = [self.cc_binary_cmd(), self.probe_cmd() if not self.probe_ok() else "skills probe: record present"]
        for group in self.base_legs():
            lines += [self.leg_cmd(*leg) for leg in group]
            lines.append(f"(here) wait for {', '.join(t + '.json' for t, *_ in group)}")
        return lines

    # 4 evolve ───────────────────────────────────────────────────────────────
    def drive_cmd(self) -> Cmd:
        n = self.plan.n_epochs
        return Cmd([self.py, "-m", "src.orchestrator.loop", "--run-dir", str(self.rd), "--epoch", "1", "--step",
                    "--drive", "--until-epoch", str(n), "--max-steps", str(MAX_DRIVE_STEPS),
                    "--note", f"evoskill run: epochs 1..{n}"], {},
                   f"the epoch loop, epochs 1..{n}", log=self.rd / "drive.out")

    def check_evolve(self):
        n, name = self.plan.n_epochs, self.plan.name
        need = {f"skills_v{n}/": (self.rd / f"skills_v{n}").is_dir(),
                f"{name}_v{n}_val.json": self.leg_path(self.tag(f"v{n}", "val")).is_file(),
                TEARDOWN_RECORD: (self.rd / TEARDOWN_RECORD).is_file()}
        missing = [k for k, ok in need.items() if not ok]
        return (not missing), (f"missing {missing}" if missing else f"epochs 1..{n} complete, evolver session ended")

    def run_evolve(self):
        self.release_hold()
        if not _exclusive_free(self.rd / DISPATCH_LOCK):
            raise RunError("an evolver or adaptor turn of this run is still running (it holds "
                           f"{DISPATCH_LOCK}); wait for it to end, then run again")
        rec = _read_json(self.rd / STATE_DIR / "run.json") or {}
        dp = rec.get("drive_pid")
        if dp and "src.orchestrator.loop" in _cmdline(dp) and str(self.rd) in _cmdline(dp):
            raise RunError(f"a drive of this run is still running (pid {dp}); `evoskill stop --run-dir {self.rd}` "
                           f"or wait for it")
        rc = self.exec(self.drive_cmd(), "the drive", check=False, drive=True)
        if rc != 0:
            raise RunError(f"the drive stopped with exit {rc} ({DRIVE_EXIT.get(rc, 'see its output')}); its output "
                           f"is in {self.rd / 'drive.out'}. Act on what it printed, then run the same command to "
                           f"resume")

    def describe_evolve(self):
        return [self.drive_cmd()]

    # 5 test ─────────────────────────────────────────────────────────────────
    def test_leg(self) -> tuple:
        n = self.plan.n_epochs
        return (self.tag(f"v{n}", "test"), "test", f"skills_v{n}", n)

    def pool_stop_cmd(self) -> Cmd:
        return Cmd([self.py, str(POOL_HOST), "stop", "--run-dir", str(self.rd), "--wait-s", str(POOL_STOP_WAIT_S)],
                   {}, "delete the run's Daytona machine")

    def machine_up(self) -> bool:
        """The run's pool host registry says a host is serving and its pid lives (cc_pool_host's registry)."""
        reg = _read_json(STORE_ROOT / self.plan.name / "pool" / "run" / "registry.json")
        if not reg or reg.get("state") == "closed":
            return False
        try:
            os.kill(int(reg.get("host_pid") or 0), 0)
        except (OSError, ValueError):
            return False
        return int(reg.get("host_pid") or 0) > 0

    def check_test(self):
        tag = self.test_leg()[0]
        if not self.leg_path(tag).is_file():
            return False, f"missing {tag}.json"
        if self.machine_up():
            return False, "the held-out leg is measured; the run's machine is still up"
        return True, "held-out leg measured, machine deleted"

    def run_test(self):
        tag, key, skills, epoch = self.test_leg()
        if not self.leg_path(tag).is_file():
            done, why = self.check_evolve()
            if not done:
                raise RunError(f"the held-out leg needs the finished drive ({why})")
            self.release_hold()
            self.run_legs([self.test_leg()])
        if self.machine_up():
            self.exec(self.pool_stop_cmd(), "cc_pool_host stop")

    def describe_test(self):
        return [self.leg_cmd(*self.test_leg()), f"(here) wait for {self.test_leg()[0]}.json",
                self.pool_stop_cmd(), "(here) the summary: no skill vs the last epoch's skills, validation and held-out"]

    # ── legs
    def release_hold(self) -> None:
        h = self.rd / HOLD_FILE
        if not h.exists():
            return
        if h.read_text(errors="replace").startswith(HOLD_MARK):
            h.unlink()
            self.out(f"  removed {HOLD_FILE} (written by `evoskill stop`)")
        else:
            raise RunError(f"{h} is present and was not written by `evoskill stop`: legs are on hold. Remove it to "
                           f"continue")

    def leg_alive(self, tag: str) -> bool:
        return _lock_held(self.rd / LEG_PID_DIR / f"{tag}.pid")

    def run_legs(self, group: list) -> None:
        launched = {}
        for i, (tag, key, skills, epoch) in enumerate(group):
            if self.leg_path(tag).is_file():
                self.out(f"  {tag}: measured already")
                continue
            if self.leg_alive(tag):
                self.out(f"  {tag}: running already; waiting for it")
                launched[tag] = 0.0
                continue
            failed = self.leg_path(tag, ".FAILED.json")
            if failed.is_file():
                kept = failed.with_name(f"{tag}.FAILED.{time.strftime('%Y%m%dT%H%M%S')}.json")
                failed.rename(kept)
                self.out(f"  {tag}: the last attempt failed (kept as {kept.name}); launching it again")
            if launched and i:
                self.sleep(LEG_LAUNCH_GAP_S)
            self.exec(self.leg_cmd(tag, key, skills, epoch), f"the leg launcher ({tag})")
            launched[tag] = time.time()
        self.wait_legs([t for t, *_ in group], launched)

    def wait_legs(self, tags: list, launched: dict) -> None:
        pending, last = set(tags), time.time()
        while True:
            for tag in sorted(pending):
                res = self.leg_path(tag)
                if res.is_file():
                    pending.discard(tag)
                    self.out(f"  {tag}: {leg_line(res)}")
                    continue
                failed = self.leg_path(tag, ".FAILED.json")
                if failed.is_file():
                    raise RunError(f"leg {tag} FAILED:\n{failed.read_text(errors='replace')[:1500]}")
                started = launched.get(tag, 0.0)
                if not self.leg_alive(tag) and time.time() - started > LEG_START_GRACE_S and not res.is_file():
                    raise RunError(f"leg {tag} is not running and wrote no result; see {self.leg_path(tag, '.log')}"
                                   f":\n{_tail(self.leg_path(tag, '.log'))}")
            if not pending:
                return
            if time.time() - last >= LEG_PROGRESS_S:
                last = time.time()
                for tag in sorted(pending):
                    self.out(f"  ... {tag}: {_last_progress(self.leg_path(tag, '.log'))}")
            self.sleep(LEG_POLL_S)

    # ── processes
    def exec(self, cmd: Cmd, what: str, *, check: bool = True, quiet: bool = False, drive: bool = False) -> int:
        """Run `cmd` under the run's environment from the repository root, its output to the terminal (and to
        cmd.log). A nonzero exit is a RunError naming `what` (check=True). The drive is started in its own session,
        so a Ctrl-C here reaches only this process, which then stops the drive itself."""
        env = {**self.env, **cmd.extra}
        self.out(f"  $ {cmd.text()}" if not quiet else f"  ({what})")
        fh = None
        if cmd.log is not None:
            cmd.log.parent.mkdir(parents=True, exist_ok=True)
            fh = open(cmd.log, "a")
            fh.write(f"\n=== {_now()} {cmd.text()}\n")
        try:
            p = subprocess.Popen(cmd.argv, cwd=str(REPO), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, text=True, errors="replace", start_new_session=drive)
            if drive:
                self.drive_proc = p
                self.record_process(drive_pid=p.pid)
            tail = []
            for line in p.stdout:
                if fh:
                    fh.write(line)
                tail = (tail + [line])[-40:]
                if not quiet:
                    sys.stdout.write(line)
                    sys.stdout.flush()
            rc = p.wait()
        finally:
            if fh:
                fh.close()
            if drive:
                self.drive_proc = None
                self.record_process(drive_pid=None)
        if check and rc != 0:
            raise RunError(f"{what} exited {rc}:\n{''.join(tail)[-3000:]}")
        return rc

    def record_process(self, **kw) -> None:
        p = self.rd / STATE_DIR / "run.json"
        rec = _read_json(p) or {}
        rec.update(kw)
        _write_json(p, rec)


def _tail(p: Path, n: int = 20) -> str:
    try:
        return "".join(Path(p).read_text(errors="replace").splitlines(True)[-n:])
    except OSError:
        return "(no log)"


def _last_progress(p: Path) -> str:
    try:
        lines = [ln for ln in Path(p).read_text(errors="replace").splitlines() if "progress:" in ln]
    except OSError:
        return "(no log yet)"
    return lines[-1][:170] if lines else "(no progress line yet)"


def leg_counts(p: Path) -> dict | None:
    d = _read_json(p)
    if not isinstance(d, dict):
        return None
    return {k: d.get(k) for k in ("sample_pass", "sample_total", "pass3", "all3", "grader_failed")}


def leg_line(p: Path) -> str:
    c = leg_counts(p)
    if c is None:
        return "unreadable"
    return (f"{c['sample_pass']}/{c['sample_total']} attempts passed, pass@N {c['pass3']}, all@N {c['all3']}, "
            f"grader_failed {c['grader_failed']}")


def summary_lines(rd: Path, name: str, n_epochs: int) -> list:
    out = ["", f"SUMMARY {name}"]
    rows = [("validation, no skill", f"{name}_base_val")]
    rows += [(f"validation, skills_v{k}", f"{name}_v{k}_val") for k in range(1, n_epochs + 1)]
    rows += [("held-out, no skill", f"{name}_base_test"), (f"held-out, skills_v{n_epochs}", f"{name}_v{n_epochs}_test")]
    for label, tag in rows:
        p = rd / f"{tag}.json"
        out.append(f"  {label:<24} {leg_line(p) if p.is_file() else '(not measured)'}")
    b, s = leg_counts(rd / f"{name}_base_test.json"), leg_counts(rd / f"{name}_v{n_epochs}_test.json")
    if b and s:
        out.append(f"  held-out pass count: {b['sample_pass']} -> {s['sample_pass']} of {s['sample_total']}")
    return out


# ───────────────────────────────────────────────────────────────────────────── driving the steps
def run_steps(steps: list, stop_after: str | None, out=print) -> None:
    """Each (name, check, run[, describe]) in order: a complete step is skipped, an incomplete one runs and must then
    check complete. Stops after `stop_after`. RunError from any of them ends the sequence."""
    for name, check, run, *_ in steps:
        done, detail = check()
        if done:
            out(f"[{name}] complete, skipped: {detail}")
        else:
            out(f"[{name}] running ({detail})")
            run()
            done, detail = check()
            if not done:
                raise RunError(f"step {name} ran but its output is not complete: {detail}")
            out(f"[{name}] done: {detail}")
        if name == stop_after:
            out(f"stopped after {name} (--stop-after); run again without it to continue")
            return


def dry_run(r: Run, base_env: dict, stop_after: str | None, out=print) -> None:
    """Print the checks, every command and its environment; run nothing, write nothing."""
    p = r.plan
    out(f"DRY RUN: evoskill run, task type {p.task_type}, run dir {p.run_dir} (nothing is written or started)")
    out(f"  run size: {p.n_epochs} epochs x {p.n_samples} samples per task; models: solver {p.solver_model}, "
        f"evolver {p.evolver_model}, adaptor {p.adaptor_model}; concurrency {p.concurrency}")
    out(f"  DAYTONA_API_KEY: {_secret_source('DAYTONA_API_KEY', base_env) or 'MISSING'}")
    if p.task_type == "tau3":
        out(f"  OPENROUTER_API_KEY: {_secret_source('OPENROUTER_API_KEY', base_env) or 'MISSING'}")
    probs = preflight(p, base_env)
    out("CHECKS: " + ("all pass" if not probs else "a real run would refuse:"))
    for pr in probs:
        out(f"  - {pr}")
    out("ENVIRONMENT of every command (the difference from this shell):")
    for ln in env_lines(base_env, r.env):
        out(f"  {ln}")
    exists = (p.run_dir / RUN_RECORD).is_file()
    for name, check, _run, describe in r.steps():
        state = ""
        if exists:
            try:
                done, detail = check()
                state = f" -- complete, would be skipped ({detail})" if done else f" -- would run ({detail})"
            except RunError as e:
                state = f" -- would refuse: {e}"
        out(f"[{name}]{state}")
        for item in describe():
            if isinstance(item, Cmd):
                out(f"  {item.what}:")
                out(f"    $ {item.text()}")
                for ln in env_lines({}, item.extra):
                    out(f"      + {ln}")
            else:
                out(f"  {item}")
        if name == stop_after:
            out(f"(--stop-after {name})")
            break


# ───────────────────────────────────────────────────────────────────────────── status / stop
def _record_plan(rd: Path) -> dict:
    rec = _read_json(rd / RUN_RECORD)
    if rec is None:
        raise RunError(f"{rd} holds no {RUN_RECORD}: not a run dir made by `evoskill run`")
    return rec


def _live_legs(rd: Path) -> list:
    out = []
    for f in sorted((rd / LEG_PID_DIR).glob("*.pid")):
        if _lock_held(f):
            rec = _read_json(f) or {}
            out.append((f.stem, rec.get("pid")))
    return out


def _session_open(rd: Path) -> bool:
    started = any(rd.glob("e*_EVOLVER_DAYTONA.out")) or any(rd.glob("e*_evolver_session"))
    return started and not (rd / TEARDOWN_RECORD).is_file()


def cmd_status(a, out=print) -> int:
    rd = Path(a.run_dir).resolve()
    rec = _record_plan(rd)
    inp, size = rec.get("inputs") or {}, rec.get("run_size") or {}
    name, n = rd.name, int(size.get("n_epochs") or 0)
    out(f"{rd}: task type {inp.get('task_type')}, {n} epochs x {size.get('n_samples')} samples; solver "
        f"{inp.get('solver_model')}, evolver {inp.get('evolver_model')}, adaptor {inp.get('adaptor_model')}")
    me = _read_json(rd / STATE_DIR / "run.json") or {}
    out(f"  evoskill run: {'running, pid ' + str(me.get('pid')) if _lock_held(rd / STATE_DIR / 'run.lock') else 'not running'}"
        + (f"; drive pid {me['drive_pid']}" if me.get("drive_pid") else ""))
    rows = [("init", (rd / "INFRA.json").is_file() and (rd / "RUNTIME.json").is_file()),
            ("adapt", (rd / "SPEC.json").is_file() and (rd / "setup" / plugin_runner.PLUGIN_JSON).is_file()),
            ("base", all((rd / f"{name}_base_{k}.json").is_file() for k in ("train", "val", "test"))),
            ("evolve", n > 0 and (rd / TEARDOWN_RECORD).is_file() and (rd / f"{name}_v{n}_val.json").is_file()),
            ("test", n > 0 and (rd / f"{name}_v{n}_test.json").is_file())]
    for step, done in rows:
        out(f"  [{step}] {'complete' if done else 'not complete'}")
    tags = [f"{name}_base_{k}" for k in ("train", "val", "test")]
    for k in range(1, n + 1):
        tags += [f"{name}_v{k}_train", f"{name}_v{k}_val"] if k < n else [f"{name}_v{k}_val"]
    tags.append(f"{name}_v{n}_test")
    live = dict(_live_legs(rd))
    for t in tags:
        p = rd / f"{t}.json"
        if p.is_file():
            out(f"  {t}: {leg_line(p)}")
        elif t in live:
            out(f"  {t}: running (pid {live[t]}) {_last_progress(rd / f'{t}.log')}")
        elif (rd / f"{t}.FAILED.json").is_file():
            out(f"  {t}: FAILED (see {t}.FAILED.json)")
    if not _exclusive_free(rd / DISPATCH_LOCK):
        out("  an evolver or adaptor turn is running")
    out(f"  evolver session: {'open' if _session_open(rd) else 'ended' if (rd / TEARDOWN_RECORD).is_file() else 'not started'}")
    if (rd / HOLD_FILE).exists():
        out(f"  {HOLD_FILE} present: no new leg launches")
    reg = _read_json(STORE_ROOT / name / "pool" / "run" / "registry.json")
    out(f"  machine: {'registry state ' + str(reg.get('state')) if reg else 'none registered'}")
    return 0


def cmd_stop(a, out=print, kill=os.kill) -> int:
    rd = Path(a.run_dir).resolve()
    _record_plan(rd)
    if a.machine and _session_open(rd):
        raise RunError("the evolver session of this run is open (its container lives on the run's machine); deleting "
                       "the machine now would end it. `evoskill stop` without --machine keeps both")
    (rd / HOLD_FILE).write_text(f"{HOLD_MARK} {_now()}: no new legs until `evoskill run` continues the run\n")
    out(f"wrote {rd / HOLD_FILE}: no new leg launches")
    me = _read_json(rd / STATE_DIR / "run.json") or {}
    if _lock_held(rd / STATE_DIR / "run.lock") and me.get("pid"):
        try:
            kill(int(me["pid"]), signal.SIGTERM)
            out(f"stopped evoskill run (pid {me['pid']}); it stops its drive")
        except OSError as e:
            out(f"could not signal evoskill run pid {me['pid']}: {e}")
    elif me.get("drive_pid") and "src.orchestrator.loop" in _cmdline(me["drive_pid"]) \
            and str(rd) in _cmdline(me["drive_pid"]):
        kill(int(me["drive_pid"]), signal.SIGTERM)
        out(f"stopped the drive (pid {me['drive_pid']})")
    for tag, pid in _live_legs(rd):
        if pid:
            try:
                kill(int(pid), signal.SIGTERM)
                out(f"stopped leg {tag} (pid {pid}); it wipes its containers and releases the machine")
            except OSError as e:
                out(f"could not signal leg {tag} pid {pid}: {e}")
    if not _exclusive_free(rd / DISPATCH_LOCK):
        out("an evolver turn is running; it is left to finish (it is part of the evolver session), and the next "
            "`evoskill run` continues after it")
    if a.machine:
        code = subprocess.run([host_tools.python_executable(), str(POOL_HOST), "stop", "--run-dir", str(rd),
                               "--wait-s", str(POOL_STOP_WAIT_S)], cwd=str(REPO)).returncode
        return code
    out("the evolver session and the run's machine are kept; `evoskill run` with the same arguments resumes")
    return 0


# ───────────────────────────────────────────────────────────────────────────── CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="evoskill", description="EvoSkill v2: run the whole system on Daytona from "
                                 "your instruction, data, split and grader.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="make or resume a run (every step; each skipped when complete)")
    r.add_argument("--instruction", required=True, help="the prose instruction for the benchmark adaptor (with a "
                   "'Run size' section)")
    r.add_argument("--data", required=True, help="the benchmark's data root (uploaded to the adaptor as /data/<its "
                   "folder name>)")
    r.add_argument("--split", required=True, help="the split file: train / val / test task ids")
    r.add_argument("--grader", required=True, help="your grader: a .py defining grade(task, snapshot)")
    r.add_argument("--run-dir", required=True, help="the run's folder; its name names the run")
    r.add_argument("--task-type", required=True, choices=TASK_TYPES)
    for role in ("solver", "evolver", "adaptor"):
        r.add_argument(f"--{role}-model", default=None,
                       help="default: " + "; ".join(f"{tt} {DEFAULT_MODELS[tt][role]}" for tt in TASK_TYPES))
    r.add_argument("--token-file", default=None, help="a Claude subscription token file (mode 0600); else "
                   "$EVOSKILL_CC_TOKEN_FILE, else $ANTHROPIC_API_KEY")
    r.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                   help=f"solver attempts in flight on the run's machine (default {DEFAULT_CONCURRENCY})")
    r.add_argument("--stop-after", choices=STEPS, default=None)
    r.add_argument("--dry-run", action="store_true", help="print every check, command and variable; run nothing")
    s = sub.add_parser("status", help="where a run is (offline)")
    s.add_argument("--run-dir", required=True)
    t = sub.add_parser("stop", help="hold new legs, stop the drive and the live legs; keep the evolver session")
    t.add_argument("--run-dir", required=True)
    t.add_argument("--machine", action="store_true", help="also delete the run's Daytona machine (refused while "
                   "the evolver session is open)")
    return ap


def hold_run_lock(rd: Path):
    """<run>/.evoskill/run.lock held for as long as this `evoskill run` lives (one per run dir at a time)."""
    d = rd / STATE_DIR
    d.mkdir(parents=True, exist_ok=True)
    fh = open(d / "run.lock", "a")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        raise RunError(f"another `evoskill run` of this run dir is running (it holds {d / 'run.lock'})")
    return fh


def cmd_run(a, out=print) -> int:
    base = dict(os.environ)
    plan = make_plan(a, base, strict=not a.dry_run)
    r = Run(plan, run_env(plan, base), out=out)
    if a.dry_run:
        dry_run(r, base, a.stop_after, out=out)
        return 0
    if plan.run_dir.exists() and not plan.run_dir.is_dir():
        raise RunError(f"--run-dir {plan.run_dir} is not a directory")
    if not (plan.run_dir / RUN_RECORD).is_file():
        probs = preflight(plan, r.env)          # before anything is written
        if probs:
            raise RunError("refusing to start:\n  " + "\n  ".join(probs))
    holder = hold_run_lock(plan.run_dir)

    def stop_handler(signum, _frame):
        p = r.drive_proc
        if p is not None and p.poll() is None:
            p.send_signal(signal.SIGTERM)
            out(f"\nstopping the drive (pid {p.pid}); live legs keep running (`evoskill stop` stops them)")
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    try:
        r.record_process(pid=os.getpid(), started_utc=_now(), drive_pid=None)
        host_tools.keep_awake(os.getpid())
        run_steps(r.steps(), a.stop_after, out=out)
        if r.leg_path(r.test_leg()[0]).is_file():
            for line in summary_lines(r.rd, plan.name, plan.n_epochs):
                out(line)
    finally:
        holder.close()
    return 0


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        if a.cmd == "run":
            return cmd_run(a)
        if a.cmd == "status":
            return cmd_status(a)
        return cmd_stop(a)
    except RunError as e:
        print(f"evoskill {a.cmd}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
