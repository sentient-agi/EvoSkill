#!/usr/bin/env python3
"""The evolver on a Daytona sandbox (RUNTIME evolver_host = "daytona") -- the shared half.

The evolver runs in the solver's own environment (the solver image), on a machine that holds only the current run's
material, with egress limited to api.anthropic.com. That makes the containment argument one about the UPLOAD
MANIFEST: nothing but what this run uploads exists on the box, so "what could it read?" is answered by the manifest.
Every file on the box is one of this run's own train-side files, each re-derivable from its host source by sha256,
and the in-VM listing after upload equals the manifest exactly. That is check_upload_manifest(), which replaces the
local #213 grep (reads of another run's skills) in daytona mode; the #186 held-out id scan of the log stays as a
second check. check_daytona_receipt() is the #294 half: resources, the domain allow list at create AND at end,
egress probes, the image recipe, the delete, the token scan.

WHAT THE EVOLVER MAY HOLD beyond the prompt / evidence / oracle / seed: the TRAINING tasks' reference workbooks;
training input workbooks are OFF by default (flags); no validation/test reference workbook or non-train input ever
reaches the evolver. Two operator grants, read from RUNTIME.json by evolver_grants() and never from the AO:
  evolver_daytona.train_inputs      -> evolver_inputs_train/      (role "inputs", absent = off)
  evolver_daytona.train_references  -> evolver_references_train/  (role "train_reference", absent = off)
forbidden_shas() covers every golden, train goldens included: a byte stream the evolver must never hold. The ONE
exception is a train golden that arrives as a "train_reference" row whose source is the verified references dir, in
a run that grants train_references; a train golden in any other row (evidence, seed, prompt ...) is refused.

A THIRD grant, the clean grader (the loop-rendered evolver prompt names <run>/grader_clean, and a prompt must not
name a directory the VM does not hold):
  evolver_daytona.grader_clean      -> grader_clean/              (role "grader", absent = off)
It is read by evolver_grader_grant(), NOT by evolver_grants(): GRANT_KEYS and the two-key grant records in receipts
(compared by grants_equal) do not include it, so it carries its own receipt fields (grader_granted, grader_dir,
grader_sha256_before / _after_vm). verify_grader_dir() proves the dir is code only: every file listed in its
MANIFEST.json at its clean sha, no workbook, no held-out task id, no byte stream of any dataset workbook (and, when
the MANIFEST pins it, the README at its readme_sha256, so the plug-in freeze that pins the MANIFEST pins the README
too).

NO ORACLE README IN TEMPLATE MODE. The run's oracle/README.md is written for the AO world: it says the reference
workbooks are withheld, which contradicts what the evolver gets (the TRAINING reference workbooks). With RUNTIME
prompt_mode="template" (evolver_template_mode()) the upload therefore carries NO oracle row: the clean grader dir
(which must be granted) and, when granted, the references dir replace it. check_upload_manifest() fails an oracle
row in template mode, a template run without the grader grant, and a receipt whose recorded template_mode differs
from RUNTIME's. AO mode uploads exactly oracle/README.md.

NO GRADING-VALIDATION MATERIAL. setup/GRADING_VALIDATION.json (the pinned copy of plugin_validate's
PLUGIN_VALIDATION.json) holds held-out aggregate figures, and the validation out dirs hold per-job held-out records.
The upload is a whitelist, so none of it is uploaded; in depth, validation_source() names it (VALIDATION_FILES by
name, anything under a *.operator dir or a validation out dir), _refuse_forbidden_sources() refuses it as the source
of any row (evolver and AO), check_upload_manifest() and check_ao_upload() fail a receipt that holds it, and the
machine-wide policy on the VM denies the Read tool on those names and on *.operator paths (DENIED_READS).

STDLIB ONLY: imported by the loop's interpreter and by the driver (the Daytona SDK venv).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Machines this module names (the benchmark adaptor's and an own-sandbox evolver's): every machine a run creates
# starts with "evoskill-"; this prefix marks the Claude sandboxes among them.
EVO_PREFIX = "evoskill-evo-"
ALLOW_DOMAINS = "api.anthropic.com"
RESOURCES = {"cpu": 8, "memory": 32, "disk": 40}
BASE_DOCKERFILE = REPO / "scripts/daytona/Dockerfile.ssb_leg"
BASE_DOCKERFILE_SHA = "681a617ef3ce70ab8056dcd78dffcf9531ddb3c55fd4d2925ba6d0dc9d06e65d"
EVO_DOCKERFILE = REPO / "scripts/daytona/Dockerfile.ssb_evolver"
CC_VERSION = "2.1.281"
CC_LINUX_X64_SHA = "56fe3da88458465fb27d7e9299dddb3fead55750fb9c2de795f233b5eea6dce1"
CC_PATH = f"/usr/local/lib/claude-code/{CC_VERSION}/claude"
# The Claude sessions run as the UNPRIVILEGED image user `evo` (uid 1000), never root:
# every uploaded file stays root-owned and read-only to it, and it cannot read root's
# /proc/<pid>/environ. Its work dir, out dir and staging are the only things handed to it.
VM_USER = "evo"
VM_HOME = "/home/evo"
VM_ROOT = "/evo"
VM_WORK = f"{VM_ROOT}/work"
VM_OUT = f"{VM_ROOT}/out"
VM_HOOKS = f"{VM_ROOT}/hooks"
VM_ENV_FILE = f"{VM_ROOT}/evo.env"
VM_PROJECTS = f"{VM_HOME}/.claude/projects"
VM_SESSION_DIR = f"{VM_PROJECTS}/-evo-work"
# Machine-wide Claude Code policy: every session on the box, nested ones included, is denied the
# server-side WebSearch, WebFetch and RemoteTrigger (all of which reach the internet THROUGH the one
# allowed host, api.anthropic.com).
VM_POLICY = "/etc/claude-code/managed-settings.json"
DENIED_TOOLS = ("WebSearch", "WebFetch", "RemoteTrigger")
# The grading validation's files and every *.operator path are denied to
# the Read tool too (`//` anchors a Claude Code rule at the filesystem root, `**` spans directories).
# A Read rule binds the Read tool (and, best effort, Grep and Glob), not a Bash command: the control
# is that no such file is on the box (validation_source(), check_upload_manifest); this is the second.
VALIDATION_FILES = ("GRADING_VALIDATION.json", "PLUGIN_VALIDATION.json", "HELDOUT_DETAIL.operator.json")
DENIED_READS = (*(f"Read(//**/{f})" for f in VALIDATION_FILES), "Read(//**/*.operator.json)",
                "Read(//**/*.operator/**)")
INPUTS_DIRNAME = "evolver_inputs_train"
REFERENCES_DIRNAME = "evolver_references_train"
GRADER_DIRNAME = "grader_clean"          # scripts/make_clean_grader.py writes <run>/grader_clean
GRADER_GRANT = "grader_clean"            # RUNTIME evolver_daytona key; deliberately not in GRANT_KEYS
GRADER_DOC_FILES = ("MANIFEST.json", "README.md")
# RUNTIME prompt_mode values (loop.PROMPT_MODES; a test pins the two equal). Absent means "ao".
PROMPT_MODES = ("ao", "template")
TEMPLATE_MODE = "template"
# The split roles whose verdicts this run REPORTS: no byte stream of theirs may reach the evolver.
HELDOUT_ROLES = ("val", "test")
# Split key "dropped": tasks whose own reference
# workbook fails self-grading under the grader of record. In no set, never measured, NEVER
# evolver-visible: split_roles() gives them this role whatever other list names them (fail-closed: a
# dropped task is never train), so every "non-train" rule below covers them (forbidden_shas: all goldens
# + every non-train input; inputs/references dirs: train only), reference_label_verdict() refuses a train
# golden that byte-equals a dropped task's golden, and heldout_id_hits() flags
# their ids like val/test ids.
DROPPED_ROLE = "dropped"
# Grading-validation material: scripts/plugin_validate.py writes held-out
# AGGREGATE figures into PLUGIN_VALIDATION.json (V4 pass counts, V2.passed_heldout, Stage B's held-out
# lines), which `freeze --grading-validation` copies to the run's setup/GRADING_VALIDATION.json, and
# held-out rows into HELDOUT_DETAIL.operator.json; its out dir, the grading driver's
# (scripts/daytona/ssb_grade_daytona.py) and the loop's re-grade dir hold per-job held-out records.
# validation_source() names all of it; no upload row may come from it. A validation OUT dir is known
# by what its writers leave in it (a dir holding every name of one tuple):
OPERATOR_SUFFIX = ".operator"            # loop.operator_dir(): <run>.operator/
VALIDATION_DIR_MARKS = (("BUILD.json", "jobs.jsonl"),         # plugin_validate build (evaluate: same --out)
                        ("HELDOUT_DETAIL.operator.json",),    # plugin_validate evaluate
                        ("jobs_normalized.jsonl",),           # ssb_grade_daytona.py run --out-dir
                        ("REGRADE.json",))                    # the loop's re-grade (loop.REGRADE_RECORD)
GRANT_KEYS = ("train_inputs", "train_references")
SPREADSHEET_SUFFIXES = (".xlsx", ".xlsm", ".xls", ".csv")
MAX_PROMPT_BYTES = 120_000              # Linux MAX_ARG_STRLEN is 128 KiB for ONE argv string
SYNC_HOOK = REPO / "scripts/evolver_sync_agent_hook.py"
DRIVER_NAME = "tau3_evolver_daytona"
# Hosts that must be UNREACHABLE from the evolver VM (probe list; any HTTP answer = reachable).
MUST_REFUSE = ("https://openrouter.ai", "https://example.com", "https://github.com",
               "https://pypi.org", "https://raw.githubusercontent.com", "https://claude.ai",
               "https://platform.claude.com", "https://downloads.claude.ai",
               "https://registry.npmjs.org", "https://app.daytona.io", "https://huggingface.co",
               "https://storage.googleapis.com", "https://8.8.8.8", "http://1.1.1.1/cdn-cgi/trace")
MUST_REACH = ("https://api.anthropic.com/v1/models",)
TOKEN_PREFIX = "sk-ant-oat01-"


DATA_ROOT_ENV = "EVOSKILL_SSBV2_DATA"
SSBV2_ROOT_ENV = "SSBV2_ROOT"
SSBV2_DATA_SUFFIX = "data/spreadsheetbench-v2"


def data_root_or_none() -> Path | None:
    """The SSB dataset tree from the environment: $EVOSKILL_SSBV2_DATA, else $SSBV2_ROOT/data/spreadsheetbench-v2,
    else None (no SSB tree on this host; the loop exports INFRA data.root as EVOSKILL_SSBV2_DATA for an SSB run)."""
    if os.environ.get(DATA_ROOT_ENV):
        return Path(os.environ[DATA_ROOT_ENV])
    if os.environ.get(SSBV2_ROOT_ENV):
        return Path(os.environ[SSBV2_ROOT_ENV]) / SSBV2_DATA_SUFFIX
    return None


def data_root() -> Path:
    """The SSB dataset tree (data_root_or_none). ValueError when the environment names none."""
    root = data_root_or_none()
    if root is None:
        raise ValueError(f"no SSB dataset root: set {DATA_ROOT_ENV} (the run's INFRA data.root) or {SSBV2_ROOT_ENV}")
    return root


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ─────────────────────────────────────────────────────────────── dataset / split / train inputs
def dataset_index(root: Path | None = None) -> dict:
    """{task_id: {input, golden, cat}} from every <root>/<cat>/dataset.json (the solver's own
    mapping: dataset.py / ssb_leg_daytona.load_index)."""
    root = Path(root or data_root())
    idx = {}
    for cat in sorted(os.listdir(root)):
        dj = root / cat / "dataset.json"
        if not dj.is_file():
            continue
        for t in json.loads(dj.read_text()):
            idx[f"{cat}/{t['id']}"] = {"cat": cat, "input": str(root / cat / t["spreadsheet_path"]),
                                       "golden": str(root / cat / t["golden_response_path"])}
    return idx


def _listed_ids(block) -> list:
    """The task ids of a split block that is a list, or a dict carrying "task_ids"."""
    ids = (block.get("task_ids") if isinstance(block, dict) else block) or []
    return [t if isinstance(t, str) else t["id"] for t in ids]


def split_roles(split) -> dict:
    """{task_id: role}: train / val / test / unused from those lists, then excluded_prior_run for ids no
    list names, then DROPPED_ROLE for every id of the "dropped" block, OVERRIDING any other role (a
    dropped task is never train; plugin_runner.validate reports such a split as malformed). A task the
    split does not name has no role (callers label it 'other'); anything but "train" is non-train."""
    sp = json.loads(Path(split).read_text())
    roles = {}
    for k in ("train", "val", "test", "unused"):
        for t in sp.get(k) or []:
            roles[t if isinstance(t, str) else t["id"]] = k
    for t in _listed_ids(sp.get("excluded_prior_run")):
        roles.setdefault(t, "excluded_prior_run")
    for t in _listed_ids(sp.get(DROPPED_ROLE)):
        roles[t] = DROPPED_ROLE
    return roles


def forbidden_shas(idx: dict, roles: dict, root: Path | None = None) -> dict:
    """{sha256: label} for every byte stream the evolver must never hold: every task's golden
    (train goldens too), every non-train input, and every other spreadsheet file anywhere in the
    dataset tree. The operator grants do NOT shrink this set: a train golden may reach the VM only as a
    "train_reference" row of the verified references dir in a run that grants train_references
    (check_upload_manifest), never through any other row. One label per sha (first seen); for
    every label a sha carries, use dataset_sha_labels()."""
    root = Path(root or data_root())
    train_inputs = {v["input"] for t, v in idx.items() if roles.get(t) == "train"}
    bad = {}
    for t, v in idx.items():
        if os.path.isfile(v["golden"]):
            bad.setdefault(sha256_file(v["golden"]), f"golden:{roles.get(t, 'other')}:{t}")
        if roles.get(t) != "train" and os.path.isfile(v["input"]):
            bad.setdefault(sha256_file(v["input"]), f"input:{roles.get(t, 'other')}:{t}")
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SPREADSHEET_SUFFIXES \
                and str(p) not in train_inputs:
            bad.setdefault(sha256_file(p), f"dataset-file:{p.relative_to(root)}")
    return bad


def inputs_dir(run_dir) -> Path:
    return Path(run_dir).absolute() / INPUTS_DIRNAME


def build_inputs_dir(run_dir, split, root: Path | None = None) -> dict:
    """<run>/evolver_inputs_train/<safe_id>/<input basename> for every TRAIN id, plus MANIFEST.json
    ([relpath, bytes, sha256] only -- no dataset path, no golden). Idempotent: an existing dir
    that verifies is left alone; one that does not verify is refused, never overwritten."""
    run_dir, root = Path(run_dir), Path(root or data_root())
    d = inputs_dir(run_dir)
    if d.exists():
        c = verify_inputs_dir(run_dir, split, root)
        if c.status != "PASS":
            raise ValueError(f"REFUSING: {d} exists and does not verify: {c.detail}")
        return {"dir": str(d), "status": "exists-verified", "detail": c.detail}
    idx, roles = dataset_index(root), split_roles(split)
    train = [t for t, r in roles.items() if r == "train"]
    tmp = d.with_name(d.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    rows = []
    for tid in sorted(train):
        src = Path(idx[tid]["input"])
        dst = tmp / tid.replace("/", "_") / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        rows.append([str(dst.relative_to(tmp)), dst.stat().st_size, sha256_file(dst)])
    (tmp / "MANIFEST.json").write_text(json.dumps(
        {"what": "the INPUT workbook of every train task (never a reference/golden workbook)",
         "layout": "<dir>/<train evidence file name without .md>/<input file basename>",
         "files": rows}, indent=1))
    tmp.rename(d)
    c = verify_inputs_dir(run_dir, split, root)
    if c.status != "PASS":
        raise ValueError(f"BUILT BUT DOES NOT VERIFY: {c.detail}")
    return {"dir": str(d), "status": "built", "detail": c.detail}


def verify_inputs_dir(run_dir, split, root: Path | None = None):
    """Re-derive the inputs dir from the dataset and the split. FAIL on: a subdir that is not a train
    stem, a missing train task, more than one file per subdir, a wrong basename, a sha that differs
    from the dataset input, a sha equal to any golden / non-train input / other dataset file, a
    symlink, an extra file, a MANIFEST that disagrees with the bytes."""
    from src.orchestrator.evidence_surface import symlinks_under
    from src.orchestrator.invariants import _fail, _ok, _skip
    n = "evolver inputs are train-only"
    run_dir, root = Path(run_dir), Path(root or data_root())
    d = inputs_dir(run_dir)
    if not d.is_dir():
        return _skip(n, f"no inputs dir at {d}", "#186")
    links = symlinks_under(d)
    if links:
        return _fail(n, f"{len(links)} symlink(s) under {d}: {links[:3]}", "#186")
    idx, roles = dataset_index(root), split_roles(split)
    train = sorted(t for t, r in roles.items() if r == "train")
    want = {t.replace("/", "_"): t for t in train}
    bad_sha = forbidden_shas(idx, roles, root)
    problems, rows = [], {}
    for p in sorted(d.iterdir()):
        if p.name == "MANIFEST.json":
            continue
        if not p.is_dir() or p.name not in want:
            problems.append(f"{p.name}: not a train task subdir")
            continue
        files = [f for f in p.iterdir()]
        if len(files) != 1 or not files[0].is_file():
            problems.append(f"{p.name}: holds {len(files)} entries, want exactly one input file")
            continue
        f, src = files[0], Path(idx[want[p.name]]["input"])
        if f.name != src.name:
            problems.append(f"{p.name}: file {f.name!r} is not the task's input {src.name!r}")
        h = sha256_file(f)
        if h != sha256_file(src):
            problems.append(f"{p.name}: bytes differ from the dataset input")
        if h in bad_sha:
            problems.append(f"{p.name}: sha equals a forbidden file ({bad_sha[h]})")
        rows[str(f.relative_to(d))] = h
    missing = sorted(set(want) - {p.name for p in d.iterdir() if p.is_dir()})
    if missing:
        problems.append(f"{len(missing)} train task(s) without an input: {missing[:3]}")
    mf = d / "MANIFEST.json"
    if not mf.is_file():
        problems.append("MANIFEST.json missing")
    else:
        mj = json.loads(mf.read_text())
        m = {r[0]: r[2] for r in mj.get("files", [])}
        if m != rows:
            problems.append("MANIFEST.json disagrees with the files on disk")
        if set(mj) != {"what", "layout", "files"}:
            problems.append(f"MANIFEST.json carries unexpected keys {sorted(set(mj) - {'what', 'layout', 'files'})}")
    if problems:
        return _fail(n, "; ".join(problems[:6]) + (f" (+{len(problems)-6} more)" if len(problems) > 6 else ""),
                     "#186", problems=problems)
    return _ok(n, f"{len(rows)} input workbooks = the {len(want)} train tasks, one each, byte-equal to "
                  f"the dataset inputs; 0 equal any of {len(bad_sha)} forbidden files (goldens, "
                  f"non-train inputs, other dataset files); no symlinks", "#186",
               n_files=len(rows), n_forbidden_checked=len(bad_sha))


# ─────────────────────────────────────────────────────────────── operator grants
def evolver_grants(run_dir=None, *, runtime: dict | None = None) -> dict:
    """{"train_inputs": bool, "train_references": bool, "source": str}: what the operator grants the
    evolver VM beyond prompt / evidence / oracle / seed. Read from RUNTIME.json
    `evolver_daytona` -- the operator-owned sandbox block the driver already receives as
    EVOSKILL_EVOLVER_DAYTONA_JSON -- never from anything the AO wrote. A key, when present, must be
    a JSON boolean (the string "false" is refused, not read as true); an ABSENT key is OFF.
    A run dir with no RUNTIME.json at all (offline callers, unit fixtures; never a loop run -- the
    driver refuses this source) keeps the legacy grant: inputs on, references off."""
    if runtime is None:
        rt_file = Path(run_dir) / "RUNTIME.json" if run_dir is not None else None
        if rt_file is None or not rt_file.is_file():
            return {"train_inputs": True, "train_references": False,
                    "source": "no RUNTIME.json: legacy default (inputs on, references off)"}
        try:
            runtime = json.loads(rt_file.read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"{rt_file} unreadable ({type(e).__name__}): refusing to guess the grants")
        src = f"{rt_file} evolver_daytona"
    else:
        src = "RUNTIME evolver_daytona (passed in)"
    if not isinstance(runtime, dict):
        raise ValueError("RUNTIME.json must be an object")
    cfg = runtime.get("evolver_daytona", {})
    if not isinstance(cfg, dict):
        raise ValueError("RUNTIME.json evolver_daytona must be an object")
    out = {}
    for k in GRANT_KEYS:
        v = cfg.get(k, False)
        if not isinstance(v, bool):
            raise ValueError(f"RUNTIME.json evolver_daytona.{k} must be true or false, got {v!r}")
        out[k] = v
    out["source"] = src
    return out


def grants_equal(a: dict | None, b: dict | None) -> bool:
    """The two grant records name the same grants (the 'source' field is ignored)."""
    return bool(a) and bool(b) and all(a.get(k) is b.get(k) for k in GRANT_KEYS)


# ─────────────────────────────────────────────────────────────── train references
def references_dir(run_dir) -> Path:
    return Path(run_dir).absolute() / REFERENCES_DIRNAME


def _label_parts(label: str) -> tuple:
    """'golden:val:Cat/01_02' -> ('golden', 'val', 'Cat/01_02'); 'dataset-file:rel' -> ('dataset-file', None, 'rel')."""
    kind, _, rest = label.partition(":")
    if kind in ("golden", "input"):
        role, _, tid = rest.partition(":")
        return kind, role, tid
    return kind, None, rest


def dataset_sha_labels(idx: dict, roles: dict, root: Path | None = None) -> dict:
    """{sha256: set of EVERY label that byte stream carries}: golden:<role>:<task>,
    input:<role>:<task> (role 'other' = a task the split does not list), and
    dataset-file:<relpath> for any other spreadsheet file under the dataset root. forbidden_shas()
    keeps one label per sha; deciding whether a train golden is ALSO some other task's file needs
    all of them."""
    root = Path(root or data_root())
    cache, lab, task_paths = {}, {}, set()

    def h(p: str) -> str:
        if p not in cache:
            cache[p] = sha256_file(p)
        return cache[p]
    for t, v in idx.items():
        r = roles.get(t, "other")
        for kind in ("golden", "input"):
            p = os.path.normpath(v[kind])
            task_paths.add(p)
            if os.path.isfile(p):
                lab.setdefault(h(p), set()).add(f"{kind}:{r}:{t}")
    for p in root.rglob("*"):
        sp = os.path.normpath(str(p))
        if p.is_file() and p.suffix.lower() in SPREADSHEET_SUFFIXES and sp not in task_paths:
            lab.setdefault(h(sp), set()).add(f"dataset-file:{p.relative_to(root)}")
    return lab


def reference_label_verdict(labels, *, strict: bool = False) -> tuple:
    """(refused, tolerated): the labels a TRAIN golden's bytes carry besides golden:train:*.
    REFUSED always (no validation/test reference workbook or non-train input ever reaches the
    evolver): the golden of a held-out task (val/test), the golden of a DROPPED task (never
    evolver-visible, see DROPPED_ROLE), and ANY task's input -- a non-train input by that rule, a train input
    too, so the references dir can never carry an input while train_inputs is off. TOLERATED unless
    strict: the golden of a task this run never grades (unused / excluded_prior_run / not in the
    split) and a stray dataset file -- both hold the train golden's own bytes, so they tell the
    evolver nothing about any held-out task. On the SSB v2 split some train goldens carry such a
    label: answer-workbook FILES that an excluded_prior_run task also names (same path), some with a
    '.fresh_datatable_scratch' copy left in the dataset tree; none carries a refused label, and none
    equals a dropped task's golden.
    strict=True refuses the tolerated class as well."""
    refused, tolerated = [], []
    for lb in sorted(labels):
        kind, role, _ = _label_parts(lb)
        if kind == "golden" and role == "train":
            continue
        if (kind == "golden" and role in (*HELDOUT_ROLES, DROPPED_ROLE)) or kind == "input":
            refused.append(lb)
        else:
            (refused if strict else tolerated).append(lb)
    return refused, tolerated


def heldout_reference_collisions(idx: dict, roles: dict, root: Path | None = None,
                                 labels: dict | None = None) -> list:
    """['<train id> = <held-out label>', ...]: train goldens whose bytes equal a held-out (val/test)
    golden or input. The project-level split (no answer workbook shared across train/val/test)
    makes this impossible; the builder and the verifier assert it."""
    labels = labels if labels is not None else dataset_sha_labels(idx, roles, root)
    out = []
    for t in sorted(t for t, r in roles.items() if r == "train"):
        g = (idx.get(t) or {}).get("golden")
        if not g or not os.path.isfile(g):
            continue
        for lb in sorted(labels.get(sha256_file(g), ())):
            kind, role, _ = _label_parts(lb)
            if kind in ("golden", "input") and role in HELDOUT_ROLES:
                out.append(f"{t} = {lb}")
    return out


def build_references_dir(run_dir, split, root: Path | None = None, *, strict: bool = False) -> dict:
    """<run>/evolver_references_train/<safe_id>/<golden basename> for every TRAIN id, plus
    MANIFEST.json ([relpath, bytes, sha256] only -- no dataset path, no other task's id). Refuses
    BEFORE copying anything if any train golden's bytes equal a held-out golden/input or any input
    (reference_label_verdict). Idempotent like build_inputs_dir: an existing dir that verifies is
    left alone; one that does not verify is refused, never overwritten."""
    run_dir, root = Path(run_dir), Path(root or data_root())
    d = references_dir(run_dir)
    if d.exists():
        c = verify_references_dir(run_dir, split, root, strict=strict)
        if c.status != "PASS":
            raise ValueError(f"REFUSING: {d} exists and does not verify: {c.detail}")
        return {"dir": str(d), "status": "exists-verified", "detail": c.detail,
                "tolerated": c.evidence.get("tolerated", {})}
    idx, roles = dataset_index(root), split_roles(split)
    train = sorted(t for t, r in roles.items() if r == "train")
    unknown = [t for t in train if t not in idx]
    if unknown:
        raise ValueError(f"REFUSING: {len(unknown)} train id(s) not in the dataset: {unknown[:3]}")
    labels = dataset_sha_labels(idx, roles, root)
    coll = heldout_reference_collisions(idx, roles, root, labels)
    if coll:
        raise ValueError(f"REFUSING: {len(coll)} train golden(s) byte-equal a held-out file -- the "
                         f"split is not project-level: {coll[:3]}")
    bad = {}
    for tid in train:
        refused, _tol = reference_label_verdict(labels.get(sha256_file(idx[tid]["golden"]), ()),
                                                strict=strict)
        if refused:
            bad[tid] = refused
    if bad:
        raise ValueError(f"REFUSING: {len(bad)} train golden(s) byte-equal a forbidden file: "
                         f"{sorted(bad.items())[:3]}")
    tmp = d.with_name(d.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    rows = []
    for tid in train:
        src = Path(idx[tid]["golden"])
        dst = tmp / tid.replace("/", "_") / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        rows.append([str(dst.relative_to(tmp)), dst.stat().st_size, sha256_file(dst)])
    (tmp / "MANIFEST.json").write_text(json.dumps(
        {"what": "the REFERENCE (golden) workbook of every TRAIN task -- never a validation/test "
                 "task's, never an input workbook",
         "layout": "<dir>/<train evidence file name without .md>/<reference file basename>",
         "files": rows}, indent=1))
    tmp.rename(d)
    c = verify_references_dir(run_dir, split, root, strict=strict)
    if c.status != "PASS":
        raise ValueError(f"BUILT BUT DOES NOT VERIFY: {c.detail}")
    return {"dir": str(d), "status": "built", "detail": c.detail,
            "tolerated": c.evidence.get("tolerated", {})}


def verify_references_dir(run_dir, split, root: Path | None = None, *, strict: bool = False):
    """Re-derive the references dir from the dataset and the split. FAIL on: any train golden
    byte-equal to a held-out (val/test) golden or input (asserted impossible under the project
    split); a subdir that is not a train stem; a missing train task; more than one file per subdir;
    a basename that is not the task's golden; bytes that differ from the task's golden; bytes that
    ALSO equal a held-out golden or any input (reference_label_verdict; with strict, any other
    task's golden or dataset file too); a symlink; an extra file; a MANIFEST that disagrees with the
    bytes or carries other keys."""
    from src.orchestrator.evidence_surface import symlinks_under
    from src.orchestrator.invariants import _fail, _ok, _skip
    n = "evolver references are train goldens only"
    run_dir, root = Path(run_dir), Path(root or data_root())
    d = references_dir(run_dir)
    if not d.is_dir():
        return _skip(n, f"no references dir at {d}", "#186")
    links = symlinks_under(d)
    if links:
        return _fail(n, f"{len(links)} symlink(s) under {d}: {links[:3]}", "#186")
    idx, roles = dataset_index(root), split_roles(split)
    train = sorted(t for t, r in roles.items() if r == "train" and t in idx)
    want = {t.replace("/", "_"): t for t in train}
    labels = dataset_sha_labels(idx, roles, root)
    problems, rows, tolerated = [], {}, {}
    coll = heldout_reference_collisions(idx, roles, root, labels)
    if coll:
        problems.append(f"{len(coll)} train golden(s) byte-equal a held-out file: {coll[:2]}")
    unknown = sorted(t for t, r in roles.items() if r == "train" and t not in idx)
    if unknown:
        problems.append(f"{len(unknown)} train id(s) not in the dataset: {unknown[:3]}")
    for p in sorted(d.iterdir()):
        if p.name == "MANIFEST.json":
            continue
        if not p.is_dir() or p.name not in want:
            problems.append(f"{p.name}: not a train task subdir")
            continue
        files = [f for f in p.iterdir()]
        if len(files) != 1 or not files[0].is_file():
            problems.append(f"{p.name}: holds {len(files)} entries, want exactly one reference file")
            continue
        f, src = files[0], Path(idx[want[p.name]]["golden"])
        if f.name != src.name:
            problems.append(f"{p.name}: file {f.name!r} is not the task's reference {src.name!r}")
        h = sha256_file(f)
        if h != sha256_file(src):
            problems.append(f"{p.name}: bytes differ from the task's reference workbook")
        refused, tol = reference_label_verdict(labels.get(h, ()), strict=strict)
        if refused:
            problems.append(f"{p.name}: bytes equal a forbidden file ({refused[:2]})")
        if tol:
            tolerated[p.name] = tol
        rows[str(f.relative_to(d))] = h
    missing = sorted(set(want) - {p.name for p in d.iterdir() if p.is_dir()})
    if missing:
        problems.append(f"{len(missing)} train task(s) without a reference: {missing[:3]}")
    mf = d / "MANIFEST.json"
    if not mf.is_file():
        problems.append("MANIFEST.json missing")
    else:
        try:
            mj = json.loads(mf.read_text())
        except json.JSONDecodeError:
            mj = {}
            problems.append("MANIFEST.json is not JSON")
        m = {r[0]: r[2] for r in mj.get("files", [])}
        if m != rows:
            problems.append("MANIFEST.json disagrees with the files on disk")
        if set(mj) != {"what", "layout", "files"}:
            problems.append(f"MANIFEST.json carries keys {sorted(mj)} != ['files', 'layout', 'what']")
    if problems:
        return _fail(n, "; ".join(problems[:6]) + (f" (+{len(problems)-6} more)" if len(problems) > 6 else ""),
                     "#186", problems=problems, tolerated=tolerated)
    return _ok(n, f"{len(rows)} reference workbooks = the {len(want)} train tasks, one each, byte-equal "
                  f"to the task's golden; 0 byte-equal a val/test golden or any input workbook; "
                  f"{len(tolerated)} also byte-equal a never-graded task's golden or a stray dataset "
                  f"copy ({'refused: strict' if strict else 'tolerated: same bytes, no held-out file'}); "
                  f"no symlinks", "#186", n_files=len(rows), tolerated=tolerated)


# ─────────────────────────────────────────────────────────────── the clean grader
def grader_dir(run_dir) -> Path:
    return Path(run_dir).absolute() / GRADER_DIRNAME


def evolver_grader_grant(run_dir=None, *, runtime: dict | None = None) -> bool:
    """RUNTIME.json evolver_daytona.grader_clean: upload the run's grader_clean/ (the grader's code
    without comments). A JSON boolean or absent (= off); anything else raises ValueError, like
    evolver_grants(). A run dir with no RUNTIME.json -> off: no upload before this grant carried it."""
    if runtime is None:
        rt_file = Path(run_dir) / "RUNTIME.json" if run_dir is not None else None
        if rt_file is None or not rt_file.is_file():
            return False
        try:
            runtime = json.loads(rt_file.read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"{rt_file} unreadable ({type(e).__name__}): refusing to guess the grants")
    if not isinstance(runtime, dict):
        raise ValueError("RUNTIME.json must be an object")
    cfg = runtime.get("evolver_daytona", {})
    if not isinstance(cfg, dict):
        raise ValueError("RUNTIME.json evolver_daytona must be an object")
    v = cfg.get(GRADER_GRANT, False)
    if not isinstance(v, bool):
        raise ValueError(f"RUNTIME.json evolver_daytona.{GRADER_GRANT} must be true or false, got {v!r}")
    return v


def evolver_template_mode(run_dir=None, *, runtime: dict | None = None) -> bool:
    """RUNTIME.json prompt_mode == "template" (the loop renders the evolver prompt). Absent or
    "ao" -> False; any other value raises ValueError, as loop._prompt_mode halts on it. A run dir
    with no RUNTIME.json -> False (the AO world: the oracle README travels as before).
    In template mode the upload carries no oracle README (module docstring)."""
    if runtime is None:
        rt_file = Path(run_dir) / "RUNTIME.json" if run_dir is not None else None
        if rt_file is None or not rt_file.is_file():
            return False
        try:
            runtime = json.loads(rt_file.read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"{rt_file} unreadable ({type(e).__name__}): refusing to guess the prompt mode")
    if not isinstance(runtime, dict):
        raise ValueError("RUNTIME.json must be an object")
    mode = runtime.get("prompt_mode", "ao")
    if mode not in PROMPT_MODES:
        raise ValueError(f"RUNTIME.json prompt_mode={mode!r} is not one of {PROMPT_MODES}")
    return mode == TEMPLATE_MODE


def _resolve_template_mode(run_dir, template_mode) -> bool:
    """template_mode=None -> the run's RUNTIME.json. An explicit value must agree with a RUNTIME.json
    that exists (ValueError otherwise); with no RUNTIME.json the explicit value is used."""
    declared = evolver_template_mode(run_dir)
    if template_mode is None:
        return declared
    if bool(template_mode) != declared and (Path(run_dir) / "RUNTIME.json").is_file():
        raise ValueError(f"REFUSING: template_mode={bool(template_mode)} but the run's RUNTIME.json "
                         f"declares prompt_mode={'template' if declared else 'ao'}")
    return bool(template_mode)


def verify_grader_dir(run_dir, split=None, root: Path | None = None, *,
                      require_readme_pin: bool = False):
    """The clean grader dir is CODE ONLY and is what its MANIFEST.json says. FAIL on: a symlink; a
    MANIFEST.json that is missing, unreadable or lists no file; a listed path that is not plain and
    relative; a listed file that is absent or whose bytes differ from its clean_sha256; a MANIFEST
    entry recording task-id hits; any file that is neither listed nor MANIFEST.json / README.md; any
    workbook-type file; a README.md that differs from a MANIFEST readme_sha256. With `split`: a
    MANIFEST checked against another split, any held-out task id (either SSB spelling) in any file,
    and any file byte-equal to a dataset workbook (forbidden_shas, when the dataset root exists).
    `require_readme_pin` (the template-mode upload): also FAIL a MANIFEST that pins no README or does
    not record the evolver image as free of the grader's tooling -- an older grader_clean whose README
    does not say the code is for reading."""
    from src.orchestrator.evidence_surface import symlinks_under
    from src.orchestrator.invariants import _fail, _ok, _skip
    n = "evolver clean grader is code only, as its MANIFEST says"
    d = grader_dir(run_dir)
    if not d.is_dir():
        return _skip(n, f"no grader dir at {d}", "#186")
    links = symlinks_under(d)
    if links:
        return _fail(n, f"{len(links)} symlink(s) under {d}: {links[:3]}", "#186")
    problems = []
    mj = {}
    try:
        mj = json.loads((d / "MANIFEST.json").read_text())
    except FileNotFoundError:
        problems.append("MANIFEST.json missing")
    except (OSError, json.JSONDecodeError) as e:
        problems.append(f"MANIFEST.json unreadable ({type(e).__name__})")
    if not isinstance(mj, dict):
        problems.append("MANIFEST.json is not an object")
        mj = {}
    listed = {}
    files = mj.get("files")
    if not isinstance(files, list) or not files:
        problems.append("MANIFEST.json lists no files")
        files = []
    for e in files:
        e = e if isinstance(e, dict) else {}
        rel = str(e.get("file") or "")
        if not rel or rel.startswith("/") or ".." in Path(rel).parts:
            problems.append(f"MANIFEST entry {rel!r} is not a plain relative path")
            continue
        if e.get("task_id_hits"):
            problems.append(f"{rel}: MANIFEST.json records task-id hits {e['task_id_hits'][:3]}")
        listed[rel] = e.get("clean_sha256")
    on_disk = {p.relative_to(d).as_posix(): p for p in d.rglob("*") if p.is_file()}
    for rel, want in sorted(listed.items()):
        p = on_disk.get(rel)
        if p is None:
            problems.append(f"{rel}: listed in MANIFEST.json but absent")
        elif sha256_file(p) != want:
            problems.append(f"{rel}: bytes differ from its MANIFEST clean_sha256")
    extra = sorted(set(on_disk) - set(listed) - set(GRADER_DOC_FILES))
    if extra:
        problems.append(f"{len(extra)} file(s) not in MANIFEST.json: {extra[:3]}")
    # The README is what the evolver reads about the grader; a current MANIFEST pins it
    # (readme_sha256), so the plug-in freeze, which pins the MANIFEST, also pins the README.
    readme_pin = mj.get("readme_sha256")
    if require_readme_pin:
        if readme_pin is None:
            problems.append("MANIFEST.json pins no README (readme_sha256): an older grader_clean, "
                            "whose README does not say the code is for reading; rebuild it with "
                            "scripts/make_clean_grader.py and re-freeze the plug-in")
        if (mj.get("evolver_image") or {}).get("grader_tooling_absent") is not True:
            problems.append("MANIFEST.json does not record the evolver image as free of the grader's "
                            "tooling (evolver_image.grader_tooling_absent)")
    if readme_pin is not None:
        rp = on_disk.get("README.md")
        if rp is None:
            problems.append("MANIFEST.json pins README.md (readme_sha256) but README.md is absent")
        elif sha256_file(rp) != readme_pin:
            problems.append("README.md bytes differ from its MANIFEST readme_sha256")
    books = sorted(r for r in on_disk if Path(r).suffix.lower() in SPREADSHEET_SUFFIXES)
    if books:
        problems.append(f"workbook-type file(s) in the grader dir: {books[:3]}")
    n_forbidden = None
    if split is not None:
        ms = mj.get("split")
        # A repo-relative MANIFEST split (make_clean_grader --split outputs/...) is judged from the repo,
        # not from whatever directory this process runs in.
        if ms and os.path.realpath(str(ms if os.path.isabs(str(ms)) else REPO / str(ms))) \
                != os.path.realpath(str(split)):
            problems.append(f"MANIFEST.json was checked against split {ms}, not {split}")
        hits = heldout_id_hits([d], split)
        if hits:
            problems.append(f"held-out task ids in the grader dir: {hits[:3]}")
        rt = Path(root or data_root())
        if rt.is_dir():
            bad = forbidden_shas(dataset_index(rt), split_roles(split), rt)
            n_forbidden = len(bad)
            clash = sorted(r for r, p in on_disk.items() if sha256_file(p) in bad)
            if clash:
                problems.append(f"file(s) byte-equal a dataset workbook: {clash[:3]}")
    if problems:
        return _fail(n, "; ".join(problems[:6]) + (f" (+{len(problems)-6} more)" if len(problems) > 6 else ""),
                     "#186", problems=problems)
    return _ok(n, f"{len(listed)} code file(s) at their MANIFEST clean sha + {len(on_disk) - len(listed)} "
                  f"doc file(s); no workbook"
                  + ("" if split is None else "; 0 held-out task ids")
                  + ("" if n_forbidden is None else f"; 0 of {n_forbidden} forbidden dataset hashes")
                  + "; no symlinks", "#186", n_files=len(on_disk))


# ─────────────────────────────────────────────────────────────── image
def dockerfile_check() -> tuple:
    """(ok, detail, record). The evolver Dockerfile must START with the frozen solver Dockerfile's
    exact bytes, carry no COPY/ADD, and pin the Claude Code binary's sha."""
    base = BASE_DOCKERFILE.read_bytes()
    evo = EVO_DOCKERFILE.read_bytes()
    rec = {"base": str(BASE_DOCKERFILE.relative_to(REPO)), "base_sha256": sha256_bytes(base),
           "evolver": str(EVO_DOCKERFILE.relative_to(REPO)), "evolver_sha256": sha256_bytes(evo)}
    probs = []
    if rec["base_sha256"] != BASE_DOCKERFILE_SHA:
        probs.append(f"frozen solver Dockerfile changed (sha {rec['base_sha256'][:12]})")
    if not evo.startswith(base):
        probs.append("evolver Dockerfile does not start with the solver Dockerfile's bytes")
    copies = [ln for ln in evo.decode().splitlines() if ln.strip().upper().startswith(("COPY", "ADD"))]
    if copies:
        probs.append(f"COPY/ADD lines present: {copies[:2]}")
    if CC_LINUX_X64_SHA not in evo.decode():
        probs.append("Claude Code sha pin missing")
    return (not probs), ("; ".join(probs) or "solver recipe + evolver layer, no COPY/ADD, CC pinned"), rec


# The GRADER's tooling in the leg image: its runner venv (/opt/evoskill/venv, openpyxl 3.1.5) and the
# recalculate-on-load LibreOffice profile (/opt/evoskill/verifier-lo-profile) that the leg's docker
# shim mounts when scorer.py runs `docker run ssb-verifier:latest`.
GRADER_TOOLING_DIR = "/opt/evoskill"
_GRADER_TOOLING_MARKS = ("/opt/evoskill", "verifier-lo-profile", "ssb-verifier")


def _dockerfile_instructions(text: str) -> list:
    """Logical Dockerfile instructions: comment and blank lines dropped, backslash continuations
    joined (a comment line inside a continuation is dropped, as Docker does)."""
    out, cur = [], ""
    for ln in text.splitlines():
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if s.endswith("\\"):
            cur += s[:-1] + " "
            continue
        out.append((cur + s).strip())
        cur = ""
    if cur.strip():
        out.append(cur.strip())
    return out


def evolver_image_grader_tooling(evo_dockerfile: Path | None = None,
                                 base_dockerfile: Path | None = None) -> tuple:
    """(absent, detail, record): is the grader's tooling ABSENT from the evolver's image? True only
    if dockerfile_check() passes (for the real recipe), the evolver layer (the bytes after the leg
    Dockerfile) has a RUN instruction that removes /opt/evoskill, no instruction after it mentions
    /opt/evoskill, verifier-lo-profile or ssb-verifier, and no RUN instruction anywhere in the recipe
    mentions docker (scorer.py recalculates through `docker run ssb-verifier:latest`; without docker
    that step fails and the workbook is graded un-recalculated). grader_clean/README.md
    (scripts/make_clean_grader.py) and the loop-rendered prompt (evolver_template.py) say "for
    reading; not runnable on your machine" only when this holds."""
    evo_p = Path(evo_dockerfile or EVO_DOCKERFILE)
    base_p = Path(base_dockerfile or BASE_DOCKERFILE)
    evo, base = evo_p.read_bytes(), base_p.read_bytes()
    rec = {"evolver_dockerfile": str(evo_p.relative_to(REPO)) if evo_p.is_relative_to(REPO) else str(evo_p),
           "evolver_dockerfile_sha256": sha256_bytes(evo), "grader_tooling_dir": GRADER_TOOLING_DIR}
    probs = []
    if evo_dockerfile is None and base_dockerfile is None:
        ok_df, det_df, _rec = dockerfile_check()
        if not ok_df:
            probs.append(f"evolver recipe does not check: {det_df}")
    if not evo.startswith(base):
        probs.append("evolver Dockerfile does not start with the leg Dockerfile's bytes")
        layer = []
    else:
        layer = _dockerfile_instructions(evo[len(base):].decode())
    rm_at = next((i for i, ins in enumerate(layer)
                  if ins.upper().startswith("RUN ")
                  and re.search(r"\brm\s+-(?:rf|fr|r)\s+/opt/evoskill/?(?:\s|$|&|;)", ins)), None)
    if rm_at is None:
        probs.append("the evolver layer never removes /opt/evoskill")
    else:
        rec["removed_by"] = layer[rm_at]
        later = [ins for ins in layer[rm_at + 1:] if any(m in ins for m in _GRADER_TOOLING_MARKS)]
        if later:
            probs.append(f"an instruction after the removal brings grader tooling back: {later[0][:120]}")
    dock = [ins for ins in _dockerfile_instructions(evo.decode())
            if ins.upper().startswith("RUN ") and re.search(r"(?i)\bdocker\b", ins)]
    rec["run_instructions_mentioning_docker"] = len(dock)
    if dock:
        probs.append(f"a RUN instruction mentions docker: {dock[0][:120]}")
    absent = not probs
    return absent, ("; ".join(probs) if probs else
                    f"the evolver image removes {GRADER_TOOLING_DIR} (the grader's venv and "
                    f"recalculate-on-load LibreOffice profile) and installs no docker, so scorer.py's "
                    f"`docker run ssb-verifier:latest` recalculation cannot run there"), rec


# ─────────────────────────────────────────────────────────────── VM env, hooks
def vm_env(model: str, env: dict | None = None) -> dict:
    """Every NON-secret variable the VM claude gets (written to VM_ENV_FILE). The OAuth token is
    NOT here: it travels only in the launch exec's env parameter."""
    from src.orchestrator.claude_pins import pin_record
    e = dict(pin_record(model, env=env)["env"])
    e.update({"IS_SANDBOX": "1", "HOME": VM_HOME,
              "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
              "ENABLE_CLAUDEAI_MCP_SERVERS": "false", "DISABLE_AUTOUPDATER": "1"})
    bad = [k for k in e if re.search(r"TOKEN|KEY|SECRET|ANTHROPIC_", k)]
    if bad:
        raise ValueError(f"vm_env would carry secret-shaped keys {bad}")
    return e


def render_env_file(e: dict, *, keep_api_key: bool | None = None) -> str:
    """VM_ENV_FILE's text. It unsets every variable that would make the session use another credential than the
    run's: on a token-file run that includes ANTHROPIC_API_KEY; on an API-key run (keep_api_key, default
    cc_account.uses_api_key()) the key is the run's credential and is not unset."""
    import shlex
    from src.orchestrator.cc_account import uses_api_key
    keep = uses_api_key() if keep_api_key is None else keep_api_key
    unset = ("ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL" if keep
             else "ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL")
    return "".join(f"{k}={shlex.quote(str(v))}\n" for k, v in sorted(e.items())) + f"unset {unset}\n"


def policy_settings_text() -> str:
    """/etc/claude-code/managed-settings.json: highest-precedence settings for EVERY Claude Code session
    on the VM -- a nested `claude` started from Bash cannot re-enable them. Denies DENIED_TOOLS and
    the Read rules DENIED_READS."""
    return json.dumps({"permissions": {"deny": [*DENIED_TOOLS, *DENIED_READS]}}, indent=1)


def hook_settings_text() -> str:
    """The Agent matcher only: it forces subagents to run synchronously (scripts/evolver_sync_agent_hook.py).
    No leak or peer hook: nothing on the VM needs guarding, and the leak
    hook would need the split, which must never be uploaded."""
    return json.dumps({"hooks": {"PreToolUse": [{"matcher": "Agent", "hooks": [
        {"type": "command",
         "command": f"/usr/local/bin/python3 {VM_HOOKS}/evolver_sync_agent_hook.py"}]}]}}, indent=1)


# ─────────────────────────────────────────────────────────────── upload manifest
def _row(vm_path: str, role: str, source: str | None = None, data: bytes | None = None) -> dict:
    if data is None:
        data_sha, nbytes = sha256_file(source), os.path.getsize(source)
    else:
        data_sha, nbytes = sha256_bytes(data), len(data)
    return {"vm_path": vm_path, "role": role, "source": source or "generated",
            "sha256": data_sha, "bytes": nbytes}


def build_upload_manifest(*, run_dir, epoch: int, prompt, evidence_dirs, oracle, staging,
                          prev_library, inputs=None, env_text: str, hooks_text: str, split=None,
                          references=None, grants: dict | None = None, grader=None,
                          grader_granted: bool | None = None,
                          template_mode: bool | None = None) -> tuple:
    """(rows, generated {vm_path: bytes}). Every file the VM will hold, at its VM path. Host paths
    are mirrored (the prompt names them); the seed library goes in AT THE STAGING PATH only.

    The operator grants: `inputs` (the run's evolver_inputs_train, role "inputs") and `references` (the run's
    evolver_references_train, role "train_reference") are each uploaded whole, and only when
    `grants` (evolver_grants()) grants them -- passing a dir the grants do not cover, or omitting
    one they do, is refused. grants=None is the legacy call: inputs as passed, no references.
    The references dir must verify (verify_references_dir, which needs `split`) at build time.
    `grader` (the run's grader_clean, role "grader") likewise travels whole and only when
    `grader_granted` (evolver_grader_grant()) is true; grader_granted=None means "as passed".
    `template_mode` (None = the run's RUNTIME.json prompt_mode, evolver_template_mode()): in template
    mode NO oracle row is built -- the AO-era oracle README says the reference workbooks are withheld,
    which contradicts what the evolver gets -- and the clean grader, which replaces it, must be passed (refused
    otherwise). An explicit value that disagrees with an existing RUNTIME.json is refused."""
    run_dir = Path(run_dir).absolute()
    tm = _resolve_template_mode(run_dir, template_mode)
    g = grants if grants is not None else {"train_inputs": inputs is not None, "train_references": False}
    for k, dirv, want_dir in (("train_inputs", inputs, inputs_dir(run_dir)),
                              ("train_references", references, references_dir(run_dir))):
        if bool(g.get(k)) != (dirv is not None):
            raise ValueError(f"REFUSING: grant {k}={bool(g.get(k))} but the "
                             f"{want_dir.name} dir was {'omitted' if dirv is None else 'passed'}")
        if dirv is not None and Path(dirv).absolute() != want_dir:
            raise ValueError(f"REFUSING: {k} dir {dirv} is not this run's {want_dir}")
    gg = (grader is not None) if grader_granted is None else bool(grader_granted)
    if gg != (grader is not None):
        raise ValueError(f"REFUSING: grant {GRADER_GRANT}={gg} but the {GRADER_DIRNAME} dir was "
                         f"{'omitted' if grader is None else 'passed'}")
    if tm and grader is None:
        raise ValueError(f"REFUSING: prompt_mode=template uploads no oracle README; the clean grader "
                         f"({GRADER_DIRNAME}) replaces it, and it is not granted/passed")
    if grader is not None:
        if Path(grader).absolute() != grader_dir(run_dir):
            raise ValueError(f"REFUSING: grader dir {grader} is not this run's {grader_dir(run_dir)}")
        gc = verify_grader_dir(run_dir, split, require_readme_pin=tm)
        if gc.status != "PASS":
            raise ValueError(f"REFUSING: grader dir does not verify: {gc.detail[:300]}")
    if references is not None:
        if not split:
            raise ValueError("REFUSING: train references need the split to verify against")
        vc = verify_references_dir(run_dir, split)
        if vc.status != "PASS":
            raise ValueError(f"REFUSING: references dir does not verify: {vc.detail[:300]}")
    rows, gen = [], {}
    p = Path(prompt).absolute()
    rows.append(_row(str(p), "prompt", str(p)))
    rows.append(_row(f"{VM_WORK}/PROMPT.md", "prompt", str(p)))
    for d in evidence_dirs:
        d = Path(os.path.realpath(str(d)))
        for f in sorted(d.rglob("*")):
            if f.is_symlink():
                raise SystemExit(f"REFUSING: symlink in evidence {f}")
            if f.is_file():
                rows.append(_row(str(f), "evidence", str(f)))
    # The rollout evidence goes up exactly as its INDEX.json lists it, refused here --
    # before any machine is touched -- rather than found by check_upload_manifest afterwards.
    _ix = evidence_index_problems(rows, evidence_dirs)
    if _ix:
        raise ValueError("REFUSING: the rollout evidence does not match its INDEX.json: " + "; ".join(_ix[:4]))
    if not tm:
        readme = Path(oracle).absolute() / "README.md"
        rows.append(_row(str(readme), "oracle", str(readme)))
    st = Path(staging).absolute()
    if epoch > 1:
        pl = Path(prev_library).absolute()
        for f in sorted(pl.rglob("*")):
            if f.is_symlink():
                raise SystemExit(f"REFUSING: symlink in previous library {f}")
            if f.is_file():
                rows.append(_row(str(st / f.relative_to(pl)), "seed", str(f)))
    for dirv, role in ((inputs, "inputs"), (references, "train_reference"), (grader, "grader")):
        if dirv is None:
            continue
        rd_ = Path(dirv).absolute()
        for f in sorted(rd_.rglob("*")):
            if f.is_symlink():
                raise ValueError(f"REFUSING: symlink in {rd_.name}: {f}")
            if f.is_file():
                rows.append(_row(str(f), role, str(f)))
    rows.append(_row(f"{VM_HOOKS}/evolver_sync_agent_hook.py", "hook", str(SYNC_HOOK)))
    gen[f"{VM_HOOKS}/settings.json"] = hooks_text.encode()
    rows.append(_row(f"{VM_HOOKS}/settings.json", "hook", data=gen[f"{VM_HOOKS}/settings.json"]))
    gen[VM_ENV_FILE] = env_text.encode()
    rows.append(_row(VM_ENV_FILE, "env", data=gen[VM_ENV_FILE]))
    gen[VM_POLICY] = policy_settings_text().encode()
    rows.append(_row(VM_POLICY, "policy", data=gen[VM_POLICY]))
    _refuse_forbidden_sources(rows, split)
    return rows, gen


def evidence_index_problems(rows, evidence_dirs) -> list:
    """The ROLLOUT EVIDENCE (src/orchestrator/rollout_evidence.py, RUNTIME evolver_evidence_layout
    "rollouts") travels whole and exactly as its INDEX.json lists it. For every granted
    evidence dir that holds an INDEX.json: INDEX.json and INDEX.md are evidence rows at their bytes on
    disk now; every trace, grade and solver-context file the INDEX lists is an evidence row at the
    sha256 the INDEX records (so the evolver received the very transcript and grade each row's score
    line describes); and no evidence row under rollouts/ or solver_context/ is unlisted. A dir with no
    INDEX.json may hold no such rows either. `rows` = manifest rows ({vm_path, role, sha256}); a row's
    vm_path is its host path (evidence is mirrored). [] = consistent."""
    ev = {r["vm_path"]: r.get("sha256") for r in rows if r.get("role") == "evidence"}
    probs = []
    for d in evidence_dirs:
        d = os.path.realpath(str(d))
        sub = [p for p in ev if p.startswith(d + "/rollouts/") or p.startswith(d + "/solver_context/")]
        ip = os.path.join(d, "INDEX.json")
        name = os.path.basename(d)
        if not os.path.isfile(ip):
            if sub:
                probs.append(f"{name}: {len(sub)} rollout-evidence row(s) but no INDEX.json lists them")
            continue
        for f in ("INDEX.json", "INDEX.md"):
            fp = os.path.join(d, f)
            if not os.path.isfile(fp) or ev.get(fp) != sha256_file(fp):
                probs.append(f"{name}/{f} is not an evidence row at its bytes on disk")
        try:
            idx = json.loads(Path(ip).read_text())
        except (OSError, ValueError) as e:
            probs.append(f"{name}/INDEX.json unreadable: {e}")
            continue
        listed = {}
        for r in idx.get("rows") or []:
            for key in ("trace", "grade"):
                if r.get(key):
                    listed[os.path.join(d, r[key])] = r.get(f"{key}_sha256")
        for m in idx.get("solver_context") or []:
            listed[os.path.join(d, str(m.get("path")))] = m.get("sha256")
        bad = sorted(p for p, h in listed.items() if ev.get(p) != h)
        extra = sorted(set(sub) - set(listed))
        if bad:
            probs.append(f"{name}: {len(bad)} file(s) INDEX.json lists are not evidence rows at the sha it "
                         f"records, e.g. {[os.path.relpath(p, d) for p in bad[:3]]}")
        if extra:
            probs.append(f"{name}: {len(extra)} rollout-evidence row(s) INDEX.json does not list, e.g. "
                         f"{[os.path.relpath(p, d) for p in extra[:3]]}")
    return probs


def _refuse_forbidden_sources(rows, split=None, root=None) -> None:
    """No row may come from the dataset tree (goldens and every input live there -- the train inputs
    travel only as their verified copies in the run's inputs dir), be the split file, or be a
    dataset.json. Judged on real paths, not on name substrings (a skill may legitimately be called
    'golden-rules'). A file of an evolver_references_train dir travels ONLY as a
    "train_reference" row and a "train_reference" row comes ONLY from such a dir (likewise
    evolver_inputs_train and "inputs"), so a train golden cannot ride in as evidence, seed or a
    literal -- for the AO too, which calls this with its own rows. The dataset tree is `root`, else the
    environment's (data_root_or_none); with neither (another benchmark on a host with no SSB tree) the
    dataset-tree rule has no tree to apply to."""
    _root = Path(root) if root else data_root_or_none()
    droot = os.path.realpath(str(_root)) if _root is not None else None
    sp = os.path.realpath(str(split)) if split else None
    for r in rows:
        src = r.get("source") or ""
        role = r.get("role")
        if src == "generated":
            if role in ("train_reference", "inputs", "grader"):
                raise ValueError(f"REFUSING manifest row {r['vm_path']}: a generated {role!r} row")
            continue
        rp = os.path.realpath(src)
        parts = Path(rp).parts
        for dname, drole in ((REFERENCES_DIRNAME, "train_reference"), (INPUTS_DIRNAME, "inputs"),
                             (GRADER_DIRNAME, "grader")):
            if (dname in parts) != (role == drole):
                raise ValueError(f"REFUSING manifest row {r['vm_path']}: role {role!r} "
                                 f"{'from' if dname in parts else 'not from'} a {dname} dir")
        if droot and (rp == droot or rp.startswith(droot + os.sep)):
            raise ValueError(f"REFUSING manifest row {r['vm_path']}: source under the dataset root")
        if sp and rp == sp:
            raise ValueError(f"REFUSING manifest row {r['vm_path']}: the split file")
        if os.path.basename(rp) == "dataset.json":
            raise ValueError(f"REFUSING manifest row {r['vm_path']}: a dataset.json")
        why = validation_source(rp) or validation_name(r.get("vm_path") or "")
        if why:
            raise ValueError(f"REFUSING manifest row {r['vm_path']}: {why} (grading-validation "
                             f"material holds held-out figures; no agent receives it)")


def validation_name(path) -> str:
    """'' unless the file NAME of `path` is one of VALIDATION_FILES or any *.operator.json (case folded:
    the host's filesystem may be case-insensitive, as macOS's is). Else the reason."""
    name = os.path.basename(str(path)).lower()
    if name in {f.lower() for f in VALIDATION_FILES} or name.endswith(OPERATOR_SUFFIX + ".json"):
        return f"a grading-validation file ({os.path.basename(str(path))})"
    return ""


def validation_source(path) -> str:
    """'' unless `path`, judged on its REAL path, is grading-validation material:
    a validation_name(); anything under a *.operator dir (loop.operator_dir: plugin_validate's --out,
    the loop's re-grades, the operator conversation); anything under a validation OUT dir, i.e. a dir
    holding every name of one VALIDATION_DIR_MARKS tuple (plugin_validate refuses an --out inside a run
    dir, so such a dir can sit anywhere). Else the reason. Every ancestor is looked at, to the root."""
    rp = Path(os.path.realpath(str(path)))
    why = validation_name(rp)
    if why:
        return why
    for d in rp.parents:
        if d.name.lower().endswith(OPERATOR_SUFFIX):
            return f"under the operator dir {d}"
        for marks in VALIDATION_DIR_MARKS:
            if all((d / m).is_file() for m in marks):
                return f"under a validation out dir {d} (it holds {' + '.join(marks)})"
    return ""


# Run on the VM through `python3 -c` (no file written): lists every entry under the given roots as
# [path, type, sha256|link target, bytes]. Types: F file, L symlink, D dir.
VM_LISTING_PY = r'''
import hashlib, json, os, sys
out = []
for root in sys.argv[1:]:
    if not os.path.lexists(root):
        continue
    for dp, dns, fns in os.walk(root, followlinks=False):
        for n in sorted(dns + fns):
            p = os.path.join(dp, n)
            if os.path.islink(p):
                out.append([p, "L", os.readlink(p), 0])
            elif os.path.isdir(p):
                out.append([p, "D", "", 0])
            elif os.path.isfile(p):
                h = hashlib.sha256()
                with open(p, "rb") as f:
                    for b in iter(lambda: f.read(1 << 20), b""):
                        h.update(b)
                out.append([p, "F", h.hexdigest(), os.path.getsize(p)])
print(json.dumps(out))
'''


def dir_sha256_from_rows(rows, root: str) -> str:
    """evidence_surface.dir_sha256 recomputed from a listing ([path, type, sha, bytes] rows)."""
    root = root.rstrip("/")
    lines = []
    for p, t, h, _b in rows:
        if not p.startswith(root + "/"):
            continue
        rel = p[len(root) + 1:]
        if t == "L":
            lines.append(f"L\t{rel}\t{h}")
        elif t == "F":
            lines.append(f"F\t{rel}\t{h}")
    hh = hashlib.sha256()
    for r in sorted(lines):
        hh.update(r.encode("utf-8", "surrogateescape"))
        hh.update(b"\n")
    return hh.hexdigest()


def listing_roots(manifest_rows=None) -> list:
    """The roots the VM listing must cover: /Users (every mirrored host path), VM_ROOT, and the Claude
    Code policy dir. /home/evo holds only the image's LibreOffice profile copy before launch."""
    return ["/Users", VM_ROOT, "/etc/claude-code"]


# ─────────────────────────────────────────────────────────────── the checks
def check_upload_manifest(*, run_dir, epoch: int, split, evidence_dirs, prev_library, prompt,
                          oracle, staging, receipt: dict | None = None, listing: dict | None = None,
                          root: Path | None = None, grants: dict | None = None,
                          grader_grant: bool | None = None, template_mode: bool | None = None):
    """REPLACES the #213 log grep in daytona mode. Re-derives every manifest row from the host bytes
    NOW and proves the VM held nothing else: (a) the prompt, twice, at its current sha; (b) every
    file of every granted evidence dir, and nothing else under them (dir sha from the rows equals the
    pre-dispatch hash AND the dir now); (c) the oracle README only; (d) the seed = skills_v{N-1}
    exactly (none at N=1); (e) the inputs dir, whole, verified train-only -- only if the run grants
    train_inputs, else not one inputs row; (e') the references dir, whole (dir sha from the rows =
    the pre-dispatch hash = the dir now), every file verified a train golden that is no held-out file
    -- only if the run grants train_references, else not one train_reference row; (f) the sync hook
    and the two generated files, re-rendered. Then: no sha of any OTHER row is a golden / non-train
    input / other dataset file (train goldens included: a train golden rides only as a
    train_reference row), and the in-VM listing after upload equals the manifest file-for-file with
    no symlinks. `grants` defaults to evolver_grants(run_dir), i.e. the run's RUNTIME.json -- the
    operator's declaration, not the receipt's; a receipt that records other grants FAILS.
    (e'') the clean grader dir, whole (rows = pre-dispatch hash = the dir now), verify_grader_dir
    PASS -- only if `grader_grant` (default evolver_grader_grant(run_dir)) is true, else not one
    grader row; a receipt whose grader_granted differs from the run's FAILS.
    (c') `template_mode` (default: RUNTIME.json prompt_mode, evolver_template_mode()): in template
    mode (c) becomes NOT ONE oracle row (the AO-era README contradicts what the evolver gets; the grader and references
    dirs replace it) and the grader must be granted; a receipt whose template_mode differs FAILS."""
    from src.orchestrator.evidence_surface import dir_sha256
    from src.orchestrator.invariants import _fail, _ok, _skip
    n = "evolver VM held only this run's train-side files"
    run_dir = Path(run_dir).absolute()
    rc = receipt if receipt is not None else _load(run_dir / f"e{epoch}_EVOLVER_CONTAINMENT.json")
    if not rc or rc.get("containment") != "daytona":
        return _skip(n, "no daytona receipt (containment is not 'daytona')", "#213")
    rows = rc.get("upload_manifest") or []
    if not rows:
        return _skip(n, "receipt carries an empty upload_manifest", "#213")
    try:
        g = grants if grants is not None else evolver_grants(run_dir)
        gg = bool(grader_grant) if grader_grant is not None else evolver_grader_grant(run_dir)
        tm = _resolve_template_mode(run_dir, template_mode)
    except ValueError as e:
        return _fail(n, f"the run's grants cannot be read: {e}", "#213")
    lst = listing if listing is not None else _load(run_dir / f"e{epoch}_EVOLVER_VM_LISTING.json")
    probs = []
    if rc.get("grants") is not None and not grants_equal(rc.get("grants"), g):
        probs.append(f"receipt records grants {({k: rc['grants'].get(k) for k in GRANT_KEYS})} != the "
                     f"run's {({k: g.get(k) for k in GRANT_KEYS})} ({g.get('source')})")
    if rc.get("grader_granted") is not None and bool(rc.get("grader_granted")) != gg:
        probs.append(f"receipt records {GRADER_GRANT}={rc.get('grader_granted')!r} != the run's {gg!r}")
    if rc.get("template_mode") is not None and bool(rc.get("template_mode")) != tm:
        probs.append(f"receipt records template_mode={rc.get('template_mode')!r} != the run's {tm!r}")
    root = Path(root or data_root())
    bad_sha = forbidden_shas(dataset_index(root), split_roles(split), root)
    prompt_sha = sha256_file(prompt)
    ev_dirs = [os.path.realpath(str(d)) for d in evidence_dirs]
    st = str(Path(staging).absolute())
    idir, rdir = str(inputs_dir(run_dir)), str(references_dir(run_dir))
    gdir = str(grader_dir(run_dir))

    def _under(p, d):
        return bool(p) and (p == d or p.startswith(d + "/"))
    by_role = {}
    for r in rows:
        role = r.get("role")
        by_role.setdefault(role, []).append(r)
        src = r.get("source")
        if src and src != "generated":
            if not os.path.isfile(src):
                probs.append(f"{r['vm_path']}: source {src} missing now")
                continue
            if sha256_file(src) != r.get("sha256"):
                probs.append(f"{r['vm_path']}: source bytes changed since upload")
        vp = r["vm_path"]
        # Grading-validation material, whatever the row's role.
        why = validation_name(vp) or (validation_source(src) if src and src != "generated" else "")
        if why:
            probs.append(f"{vp}: {why}")
        if role == "train_reference":
            # A train golden IS in bad_sha (and its name says golden): these rows are judged, whole,
            # against the verified references dir below -- and only if the run grants them.
            if not (_under(vp, rdir) and src == vp):
                probs.append(f"{vp}: train_reference row not mirrored from {rdir}")
            continue
        if _under(vp, rdir) or _under(src, rdir):
            probs.append(f"{vp}: a {role!r} row from the references dir")
        if role != "inputs" and (_under(vp, idir) or _under(src, idir)):
            probs.append(f"{vp}: a {role!r} row from the inputs dir")
        if role == "grader":
            if not (_under(vp, gdir) and src == vp):
                probs.append(f"{vp}: grader row not mirrored from {gdir}")
        elif _under(vp, gdir) or _under(src, gdir):
            probs.append(f"{vp}: a {role!r} row from the grader dir")
        if r.get("sha256") in bad_sha:
            probs.append(f"{vp}: sha equals a forbidden file ({bad_sha[r['sha256']]})")
        s = (vp + " " + (src or "")).lower()
        if "golden" in s or "dataset.json" in s or os.path.realpath(str(split)) == os.path.realpath(src or "/nonexistent"):
            probs.append(f"{vp}: forbidden name")
    known = {"prompt", "evidence", "oracle", "seed", "inputs", "train_reference", "grader", "hook",
             "env", "policy"}
    for role in set(by_role) - known:
        probs.append(f"unknown manifest role {role!r}")
    pr = by_role.get("prompt", [])
    if sorted(r["vm_path"] for r in pr) != sorted([str(Path(prompt).absolute()), f"{VM_WORK}/PROMPT.md"]) \
            or any(r["sha256"] != prompt_sha for r in pr):
        probs.append("prompt rows are not exactly the current ITER prompt at its path and PROMPT.md")
    ev = by_role.get("evidence", [])
    before = rc.get("evidence_sha256_before") or {}
    for d in ev_dirs:
        mine = [[r["vm_path"], "F", r["sha256"], r["bytes"]] for r in ev if r["vm_path"].startswith(d + "/")]
        got = dir_sha256_from_rows(mine, d)
        if got != before.get(d):
            probs.append(f"{os.path.basename(d)}: manifest rows != pre-dispatch hash")
        if got != dir_sha256(d):
            probs.append(f"{os.path.basename(d)}: manifest rows != the dir on disk now")
    stray = [r["vm_path"] for r in ev if not any(r["vm_path"].startswith(d + "/") for d in ev_dirs)]
    if stray:
        probs.append(f"{len(stray)} evidence row(s) outside the granted evidence dirs: {stray[:2]}")
    # The rollout evidence travelled exactly as its INDEX.json lists it (traces + grades).
    probs += evidence_index_problems(rows, ev_dirs)
    n_traces = sum(1 for r in ev if r["vm_path"].endswith(".trace.jsonl") and "/rollouts/" in r["vm_path"])
    orc = by_role.get("oracle", [])
    _readme = str(Path(oracle).absolute() / "README.md")
    if tm:
        if orc:
            probs.append(f"{len(orc)} oracle row(s), but prompt_mode=template uploads no oracle README "
                         f"(it says the reference workbooks are withheld, which the evolver is not; the grader and "
                         f"references dirs replace it)")
        if not gg:
            probs.append(f"prompt_mode=template but {GRADER_GRANT} is not granted: with no oracle "
                         f"README, nothing on the VM states how a verdict is produced")
    elif [r["vm_path"] for r in orc] != [_readme] or [r["source"] for r in orc] != [_readme]:
        probs.append("oracle rows are not exactly oracle/README.md")
    seed = by_role.get("seed", [])
    if epoch <= 1 and seed:
        probs.append(f"{len(seed)} seed row(s) at epoch 1")
    if epoch > 1:
        pl = Path(prev_library).absolute()
        want = {str(Path(st) / f.relative_to(pl)): sha256_file(f) for f in pl.rglob("*") if f.is_file()}
        if {r["vm_path"]: r["sha256"] for r in seed} != want:
            probs.append(f"seed rows != skills_v{epoch - 1} file-for-file")
    inp = by_role.get("inputs", [])
    if g.get("train_inputs"):
        want_in = {str(f): sha256_file(f) for f in Path(idir).rglob("*") if f.is_file()} if Path(idir).is_dir() else {}
        if {r["vm_path"]: r["sha256"] for r in inp} != want_in or not want_in:
            probs.append("inputs rows != the whole evolver_inputs_train dir")
        vc = verify_inputs_dir(run_dir, split, root)
        if vc.status != "PASS":
            probs.append(f"inputs dir does not verify: {vc.detail[:200]}")
    elif inp:
        probs.append(f"{len(inp)} inputs row(s), but the run does not grant train_inputs ({g.get('source')})")
    ref = by_role.get("train_reference", [])
    if g.get("train_references"):
        have = Path(rdir).is_dir()
        want_ref = {str(f): sha256_file(f) for f in Path(rdir).rglob("*") if f.is_file()} if have else {}
        if {r["vm_path"]: r["sha256"] for r in ref} != want_ref or not want_ref or len(ref) != len(want_ref):
            probs.append(f"train_reference rows != the whole {REFERENCES_DIRNAME} dir")
        got = dir_sha256_from_rows([[r["vm_path"], "F", r["sha256"], r["bytes"]] for r in ref], rdir)
        if got != rc.get("references_sha256_before"):
            probs.append(f"{REFERENCES_DIRNAME}: manifest rows != pre-dispatch hash")
        if not have or got != dir_sha256(rdir):
            probs.append(f"{REFERENCES_DIRNAME}: manifest rows != the dir on disk now")
        vr = verify_references_dir(run_dir, split, root)
        if vr.status != "PASS":
            probs.append(f"references dir does not verify: {vr.detail[:300]}")
    elif ref:
        probs.append(f"{len(ref)} train_reference row(s), but the run does not grant train_references "
                     f"({g.get('source')})")
    grd = by_role.get("grader", [])
    if gg:
        have = Path(gdir).is_dir()
        want_g = {str(f): sha256_file(f) for f in Path(gdir).rglob("*") if f.is_file()} if have else {}
        if {r["vm_path"]: r["sha256"] for r in grd} != want_g or not want_g or len(grd) != len(want_g):
            probs.append(f"grader rows != the whole {GRADER_DIRNAME} dir")
        got = dir_sha256_from_rows([[r["vm_path"], "F", r["sha256"], r["bytes"]] for r in grd], gdir)
        if got != rc.get("grader_sha256_before"):
            probs.append(f"{GRADER_DIRNAME}: manifest rows != pre-dispatch hash")
        if not have or got != dir_sha256(gdir):
            probs.append(f"{GRADER_DIRNAME}: manifest rows != the dir on disk now")
        vg = verify_grader_dir(run_dir, split, root, require_readme_pin=tm)
        if vg.status != "PASS":
            probs.append(f"grader dir does not verify: {vg.detail[:300]}")
    elif grd:
        probs.append(f"{len(grd)} grader row(s), but the run does not grant {GRADER_GRANT}")
    hk = {r["vm_path"]: r for r in by_role.get("hook", [])}
    if set(hk) != {f"{VM_HOOKS}/evolver_sync_agent_hook.py", f"{VM_HOOKS}/settings.json"}:
        probs.append("hook rows are not exactly the sync hook + its settings")
    else:
        if hk[f"{VM_HOOKS}/settings.json"]["sha256"] != sha256_bytes(hook_settings_text().encode()):
            probs.append("hook settings differ from the rendered #212-only settings")
        if hk[f"{VM_HOOKS}/evolver_sync_agent_hook.py"]["source"] != str(SYNC_HOOK):
            probs.append("hook script row does not come from scripts/evolver_sync_agent_hook.py")
    envr = by_role.get("env", [])
    venv = (rc.get("daytona") or {}).get("vm_env") or {}
    if [r["vm_path"] for r in envr] != [VM_ENV_FILE]:
        probs.append("env rows are not exactly the one env file")
    elif not venv or envr[0]["sha256"] != sha256_bytes(render_env_file(venv).encode()):
        probs.append("env file does not re-render from the receipt's recorded vm_env")
    elif [k for k in venv if re.search(r"TOKEN|KEY|SECRET|ANTHROPIC_", k)]:
        probs.append("recorded vm_env carries secret-shaped keys")
    pol = by_role.get("policy", [])
    if [r["vm_path"] for r in pol] != [VM_POLICY] or \
            pol[0]["sha256"] != sha256_bytes(policy_settings_text().encode()):
        probs.append("policy rows are not exactly the rendered managed-settings.json")
    # listing
    if not lst or "after_upload" not in lst:
        probs.append(f"no VM listing (e{epoch}_EVOLVER_VM_LISTING.json after_upload)")
    else:
        if (lst.get("pre_upload") or {}).get("users_exists"):
            probs.append("/Users existed on the VM before upload (the image carries data?)")
        files = {p: h for p, t, h, _b in lst["after_upload"] if t == "F"}
        links = [p for p, t, _h, _b in lst["after_upload"] if t == "L"]
        man = {r["vm_path"]: r["sha256"] for r in rows}
        if links:
            probs.append(f"{len(links)} symlink(s) on the VM after upload: {links[:2]}")
        extra = sorted(set(files) - set(man))
        missing = sorted(set(man) - set(files))
        diff = sorted(p for p in set(files) & set(man) if files[p] != man[p])
        if extra:
            probs.append(f"{len(extra)} file(s) on the VM not in the manifest: {extra[:3]}")
        if missing:
            probs.append(f"{len(missing)} manifest file(s) absent on the VM: {missing[:3]}")
        if diff:
            probs.append(f"{len(diff)} file(s) on the VM differ from the manifest: {diff[:3]}")
    if probs:
        return _fail(n, "; ".join(probs[:6]) + (f" (+{len(probs)-6} more)" if len(probs) > 6 else ""),
                     "#213", problems=probs)
    counts = {k: len(v) for k, v in sorted(by_role.items())}
    gr = {k: bool(g.get(k)) for k in GRANT_KEYS}
    return _ok(n, f"{len(rows)} files, every one re-derived from this run's own bytes {counts}; grants "
                  f"{gr}, {GRADER_GRANT}={gg} ({g.get('source')}); "
                  + ("prompt_mode=template: no oracle README; " if tm else "")
                  + (f"{n_traces} rollout trace(s) + grades exactly as their INDEX.json lists them; "
                     if n_traces else "")
                  + f"the VM listing after upload equals "
                  f"the manifest exactly; 0 of {len(bad_sha)} forbidden hashes outside the granted "
                  f"train references; no other run's file can have been read (none was on the box)",
               "#213", counts=counts, grants=gr, grader_granted=gg, template_mode=tm)


def check_ao_upload(*, run_dir, epoch: int, receipt: dict, message, evidence_dirs, oracle,
                    listing: dict | None = None):
    """The AO's upload = exactly: the author message (as sent, at VM_WORK/MESSAGE.md), every file
    of every granted evidence dir, the oracle README, the AO's three literals (SPEC.json, the
    e{N} RUNTIME projection, the previous prompt) at their own paths and current bytes, and the env
    file. The in-VM listing after upload equals it file for file, with no symlinks."""
    from src.orchestrator.evidence_surface import dir_sha256
    from src.orchestrator.invariants import _fail, _ok, _skip
    n = "AO VM held only the author message, the evidence, the oracle README and its literals"
    run_dir = Path(run_dir).absolute()
    if receipt.get("containment") != "daytona" or receipt.get("role") != "ao":
        return _skip(n, "not an AO daytona receipt")
    rows = receipt.get("upload_manifest") or []
    if not rows:
        return _skip(n, "empty upload_manifest")
    lst = listing if listing is not None else _load(run_dir / f"e{epoch}_AO_VM_LISTING.json")
    probs, by = [], {}
    for r in rows:
        by.setdefault(r.get("role"), []).append(r)
        src = r.get("source")
        if src and src != "generated":
            if not os.path.isfile(src):
                probs.append(f"{r['vm_path']}: source missing now")
            elif sha256_file(src) != r.get("sha256"):
                probs.append(f"{r['vm_path']}: source bytes changed since upload")
        s = (r["vm_path"] + " " + (src or "")).lower()
        if "golden" in s or "dataset.json" in s or "split" in os.path.basename(src or "").lower():
            probs.append(f"{r['vm_path']}: forbidden name")
        why = validation_name(r["vm_path"]) or (validation_source(src) if src and src != "generated" else "")
        if why:
            probs.append(f"{r['vm_path']}: {why}")
    for role in set(by) - {"message", "evidence", "oracle", "literal", "env", "policy"}:
        probs.append(f"unknown role {role!r}")
    pol = by.get("policy", [])
    if [r["vm_path"] for r in pol] != [VM_POLICY] or pol[0]["sha256"] != sha256_bytes(policy_settings_text().encode()):
        probs.append("policy row is not exactly the rendered managed-settings.json")
    msg = by.get("message", [])
    if [r["vm_path"] for r in msg] != [f"{VM_WORK}/MESSAGE.md"] or \
            any(r["sha256"] != sha256_file(message) for r in msg):
        probs.append("message row is not exactly the sent author message")
    ev_dirs = [os.path.realpath(str(d)) for d in evidence_dirs]
    ev = by.get("evidence", [])
    before = receipt.get("evidence_sha256_before") or {}
    for d in ev_dirs:
        mine = [[r["vm_path"], "F", r["sha256"], r["bytes"]] for r in ev if r["vm_path"].startswith(d + "/")]
        got = dir_sha256_from_rows(mine, d)
        if got != before.get(d) or got != dir_sha256(d):
            probs.append(f"{os.path.basename(d)}: manifest rows != the evidence dir")
    if any(not any(r["vm_path"].startswith(d + "/") for d in ev_dirs) for r in ev):
        probs.append("an evidence row lies outside the granted evidence dirs")
    if [r["vm_path"] for r in by.get("oracle", [])] != [str(Path(oracle).absolute() / "README.md")]:
        probs.append("oracle rows are not exactly oracle/README.md")
    allowed = {str(run_dir / "SPEC.json"), str(run_dir / f"e{epoch}_AO_RUNTIME_VIEW.json"),
               str(run_dir / f"ITER{epoch - 1}_PROMPT.md")}
    for r in by.get("literal", []):
        if r["vm_path"] not in allowed or r["vm_path"] != r.get("source"):
            probs.append(f"literal {r['vm_path']} is not one of the AO's three grants")
    if [r["vm_path"] for r in by.get("env", [])] != [VM_ENV_FILE]:
        probs.append("env rows are not exactly the one env file")
    if not lst or "after_upload" not in lst:
        probs.append(f"no VM listing e{epoch}_AO_VM_LISTING.json")
    else:
        if (lst.get("pre_upload") or {}).get("users_exists"):
            probs.append("/Users existed on the VM before upload")
        files = {p: h for p, t, h, _b in lst["after_upload"] if t == "F"}
        links = [p for p, t, _h, _b in lst["after_upload"] if t == "L"]
        man = {r["vm_path"]: r["sha256"] for r in rows}
        if links or set(files) != set(man) or any(files[p] != man[p] for p in man if p in files):
            probs.append(f"VM listing != manifest (extra {sorted(set(files) - set(man))[:2]}, missing "
                         f"{sorted(set(man) - set(files))[:2]}, links {links[:2]})")
    if probs:
        return _fail(n, "; ".join(probs[:6]), "#253", problems=probs)
    return _ok(n, f"{len(rows)} files {({k: len(v) for k, v in sorted(by.items())})}; VM listing equals "
                  f"the manifest", "#253")


def _load(p) -> dict:
    try:
        return json.loads(Path(p).read_text())
    except Exception:  # noqa: BLE001
        return {}


def log_init_and_tools(log) -> dict:
    """{api_key_source, mcp_servers, tools, remote_trigger_calls, cc_version, model} from the log."""
    out = {"api_key_source": None, "mcp_servers": None, "tools": None, "remote_trigger_calls": 0,
           "cc_version": None, "model": None}
    try:
        text = Path(log).read_text(errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") == "system" and e.get("subtype") == "init" and out["api_key_source"] is None:
            out.update(api_key_source=e.get("apiKeySource"), mcp_servers=e.get("mcp_servers"),
                       tools=e.get("tools"), cc_version=e.get("claude_code_version"), model=e.get("model"))
        if e.get("type") == "assistant":
            for c in (e.get("message") or {}).get("content") or []:
                if isinstance(c, dict) and c.get("type") == "tool_use" and c.get("name") == "RemoteTrigger":
                    out["remote_trigger_calls"] += 1
    return out


_NESTED_CLAUDE = re.compile(r"(?:^|[\s;&|(`$])claude(?:\s|$)")


def audit_sessions(paths) -> dict:
    """Every Claude session log on the VM (the parent's stream-json AND every *.jsonl transcript under
    the downloaded projects tree -- nested sessions and subagents included): tool_use counts, any use
    of a denied tool, any Bash command that starts a nested `claude`, names the API host, or touches
    the OAuth token variable. Values are never returned, only file names and short tags."""
    out = {"files": 0, "tool_uses": {}, "denied_tool_uses": [], "nested_claude": [],
           "api_or_token_refs": []}
    files = []
    for base in paths:
        b = Path(base)
        if b.is_file():
            files.append(b)
        elif b.is_dir():
            files += sorted(p for p in b.rglob("*.jsonl") if p.is_file())
    for f in files:
        out["files"] += 1
        for line in f.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = e.get("message") if isinstance(e.get("message"), dict) else {}
            for c in (msg.get("content") or []) if isinstance(msg.get("content"), list) else []:
                if not (isinstance(c, dict) and c.get("type") == "tool_use"):
                    continue
                name = str(c.get("name"))
                out["tool_uses"][name] = out["tool_uses"].get(name, 0) + 1
                if name in DENIED_TOOLS:
                    out["denied_tool_uses"].append(f"{f.name}:{name}")
                cmd = str((c.get("input") or {}).get("command") or "")
                if name == "Bash" and _NESTED_CLAUDE.search(cmd):
                    out["nested_claude"].append(f.name)
                if name == "Bash" and ("api.anthropic.com" in cmd or "OAUTH_TOKEN" in cmd):
                    out["api_or_token_refs"].append(f.name)
    return out


def heldout_id_hits(paths, split) -> list:
    """Held-out (val/test) and DROPPED task ids anywhere in the given logs/transcripts, in EITHER
    spelling the SSB v2 surface uses: canonical 'Cat/NN_MM' (escape-tolerant: no quote context
    required) and the file-safe 'Cat_NN_MM'. invariants.check_no_heldout_reachable matches only quoted
    whole-string forms, which stream-json escaping and the safe spelling both defeat.
    Dropped ids (split key "dropped") count too: a dropped task is never evolver-visible (DROPPED_ROLE)."""
    sp = json.loads(Path(split).read_text())
    held = [t if isinstance(t, str) else t["id"] for k in ("val", "test") for t in (sp.get(k) or [])]
    held += sorted(set(_listed_ids(sp.get(DROPPED_ROLE))) - set(held))
    pats = []
    for t in held:
        pats.append((t, re.compile(r"(?<![\w/])" + re.escape(t) + r"(?![\w])")))
        safe = t.replace("/", "_")
        pats.append((t, re.compile(r"(?<![\w])" + re.escape(safe) + r"(?![\w])")))
    hits = []
    for base in paths:
        b = Path(base)
        fl = [b] if b.is_file() else (sorted(p for p in b.rglob("*") if p.is_file()) if b.is_dir() else [])
        for f in fl:
            text = f.read_text(errors="replace")
            for t, rx in pats:
                if rx.search(text):
                    hits.append(f"{f.name}:{t}")
    return sorted(set(hits))


def ledger_deleted(ledger_path, sandbox_id: str) -> bool:
    try:
        for line in open(ledger_path):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("event") == "deleted" and r.get("id") == sandbox_id and r.get("driver") == DRIVER_NAME:
                return True
    except OSError:
        return False
    return False


def check_daytona_receipt(receipt: dict, *, log, ledger_path=None,
                          require_libreoffice: str | None = None) -> tuple:
    """(ok, detail). The #294 half for a daytona receipt: where the evolver ran and how it was held."""
    d = receipt.get("daytona") or {}
    probs = []
    if receipt.get("containment") != "daytona" or not receipt.get("sandboxed"):
        probs.append("receipt is not a sandboxed daytona receipt")
    for k in ("readback_create", "readback_end"):
        rb = d.get(k) or {}
        if {x: rb.get(x) for x in RESOURCES} != RESOURCES:
            probs.append(f"{k}: resources {({x: rb.get(x) for x in RESOURCES})} != {RESOURCES}")
        if rb.get("domain_allow_list") != ALLOW_DOMAINS:
            probs.append(f"{k}: domain_allow_list {rb.get('domain_allow_list')!r} != {ALLOW_DOMAINS!r}")
        if rb.get("network_allow_list") not in (None, ""):
            probs.append(f"{k}: network_allow_list set ({rb.get('network_allow_list')!r})")
        if rb.get("network_block_all"):
            probs.append(f"{k}: network_block_all is on (then nothing reaches the model API)")
    if not (d.get("readback_end") or {}).get("refreshed"):
        probs.append("readback_end was not a fresh read (refresh failed or not recorded)")
    eg = d.get("egress_probe") or []
    reach = {r["url"]: r.get("reachable") for r in eg if isinstance(r, dict) and "url" in r}
    for u in MUST_REACH:
        if reach.get(u) is not True:
            probs.append(f"egress: {u} not reachable")
    leaked = [u for u in MUST_REFUSE if reach.get(u) is not False]
    if leaked:
        probs.append(f"egress: not refused (or not probed): {leaked[:4]}")
    direct = d.get("egress_direct") or []
    if not direct or any(r.get("reached") is not False for r in direct):
        probs.append(f"egress: a proxy-less HTTPS request was not refused (or not probed): "
                     f"{[r.get('target') for r in direct if r.get('reached') is not False][:3] or 'none probed'}")
    raw = {r.get("target"): r for r in (d.get("egress_raw") or [])}
    dns_ok = (d.get("egress_dns") or {}).get("accepted") is True   # DNS to the sandbox's own resolver
    for t, key in (("example.com:22", "tcp_connected"), ("udp 1.1.1.1:53", "answered"),
                   ("[2606:4700:4700::1111]:443", "tcp_connected"), ("169.254.169.254:80", "tcp_connected")):
        if t == "udp 1.1.1.1:53" and dns_ok:
            continue   # DNS resolution gives no web content (tau3_evolver_daytona egress_dns)
        if t not in raw or raw[t].get(key) is not False:
            probs.append(f"egress: {t} not refused (or not probed)")
    ok_df, det_df, rec_df = dockerfile_check()
    if not ok_df:
        probs.append(f"image: {det_df}")
    if (d.get("dockerfile") or {}).get("evolver_sha256") != rec_df["evolver_sha256"]:
        probs.append("image: receipt's evolver Dockerfile sha != the file on disk")
    ident = d.get("identity") or {}
    if ident.get("cc_sha256") != CC_LINUX_X64_SHA:
        probs.append(f"identity: Claude Code binary sha {str(ident.get('cc_sha256'))[:12]} != pinned")
    if require_libreoffice and ident.get("libreoffice") != require_libreoffice:
        probs.append(f"identity: LibreOffice {ident.get('libreoffice')!r} != {require_libreoffice!r}")
    before = receipt.get("evidence_sha256_before") or {}
    if (receipt.get("evidence_sha256_after_vm") or {}) != before or not before:
        probs.append("evidence changed on the VM during the run (or no after-hash)")
    # Each granted read-only dir (inputs, train references, clean grader) must hash the same before and after
    # the run. A dir is "granted" when the manifest carries its rows OR the receipt names it -- so a
    # receipt cannot drop the hashes of a dir that was on the box.
    _mroles = {r.get("role") for r in receipt.get("upload_manifest") or []}
    for _role, _key, _what in (("inputs", "inputs", "inputs"),
                               ("train_reference", "references", "train references"),
                               ("grader", "grader", "clean grader")):
        if receipt.get("role") == "ao":
            break
        if _role in _mroles or receipt.get(f"{_key}_dir") or receipt.get(f"{_key}_sha256_before"):
            if receipt.get(f"{_key}_sha256_after_vm") != receipt.get(f"{_key}_sha256_before") \
                    or not receipt.get(f"{_key}_sha256_before"):
                probs.append(f"{_what} changed on the VM during the run (or no hash)")
    ts = receipt.get("token_scan") or {}
    if ts.get("hits") != [] or not ts.get("scanned_files"):
        probs.append(f"token scan: hits {ts.get('hits')!r}, scanned {ts.get('scanned_files')}")
    if not d.get("sandbox_id") or not ledger_deleted(
            ledger_path or os.environ.get("EVOSKILL_DAYTONA_LEDGER",
                                          str(REPO / ".cache" / "daytona" / "ledger.jsonl")),
            d.get("sandbox_id")):
        probs.append("no ledgered 'deleted' row for this sandbox id")
    lrt = d.get("log_round_trip") or {}
    _lb = Path(str(log)).read_bytes() if Path(str(log)).is_file() else None
    if _lb is not None and receipt.get("role") == "ao" and _lb.startswith(b"=== prompt author:"):
        _lb = _lb.split(b"\n", 1)[1] if b"\n" in _lb else b""   # the one header line the AO log carries
    if not lrt.get("equal") or _lb is None or sha256_bytes(_lb) != lrt.get("streamed_sha256"):
        probs.append("the audited log is not the byte-exact copy of the VM log recorded in the receipt")
    sa = d.get("session_audit") or {}
    if not sa.get("files"):
        probs.append("no session audit recorded")
    elif sa.get("denied_tool_uses") or sa.get("nested_claude") or sa.get("api_or_token_refs"):
        probs.append(f"session audit: denied tools {sa.get('denied_tool_uses')[:3]}, nested claude "
                     f"{sa.get('nested_claude')[:3]}, api/token refs {sa.get('api_or_token_refs')[:3]}")
    li = log_init_and_tools(log)
    # apiKeySource names API-KEY sources only: a session authenticated by CLAUDE_CODE_OAUTH_TOKEN
    # (or by a keychain login) reports "none". On a token-file run the check is therefore that NO API key
    # was used; on an API-key run (no token file given) the key is the run's one credential.
    from src.orchestrator.cc_account import allowed_api_key_sources
    if li["api_key_source"] not in allowed_api_key_sources():
        probs.append(f"log init apiKeySource {li['api_key_source']!r}: not the run's credential")
    if li["mcp_servers"] is None:
        probs.append("log carries no init record (mcp_servers unknown)")
    elif li["mcp_servers"] != []:
        probs.append(f"log init lists MCP servers {li['mcp_servers']!r}")
    if li["remote_trigger_calls"]:
        probs.append(f"{li['remote_trigger_calls']} RemoteTrigger call(s) (a RemoteTrigger call fails the check)")
    if probs:
        return False, "; ".join(probs[:6]) + (f" (+{len(probs)-6} more)" if len(probs) > 6 else "")
    return True, (f"Daytona sandbox {str(d.get('sandbox_id'))[:8]} 8/32/40, allow list "
                  f"{ALLOW_DOMAINS} at create and at end, {len(MUST_REFUSE)} hosts refused, solver "
                  f"image + CC {CC_VERSION} (pinned sha), evidence and granted dirs unchanged on the VM, "
                  f"0 token hits, deleted (ledgered), "
                  f"{'API key' if li['api_key_source'] == 'ANTHROPIC_API_KEY' else 'OAuth token'} auth, no MCP, "
                  f"no RemoteTrigger")


def token_scan(paths, token: str) -> tuple:
    """(hit paths, files scanned). Raw-byte search for the token AND its prefix; never returns the
    token or any excerpt."""
    tok = token.encode()
    pre = TOKEN_PREFIX.encode()
    hits, n = [], 0
    for base in paths:
        for f in ([Path(base)] if Path(base).is_file() else Path(base).rglob("*")):
            if f.is_file() and not f.is_symlink():
                n += 1
                b = f.read_bytes()
                if tok in b or pre in b:
                    hits.append(str(f))
    return hits, n


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("build-inputs", "verify-inputs", "build-references", "verify-references"):
        s = sub.add_parser(c)
        s.add_argument("--run-dir", required=True)
        s.add_argument("--split", required=True)
        if c.endswith("references"):
            s.add_argument("--strict", action="store_true",
                           help="also refuse a train golden that equals a never-graded task's golden "
                                "or a stray dataset copy")
    s = sub.add_parser("grants", help="print the run's operator grants as resolved from its RUNTIME.json")
    s.add_argument("--run-dir", required=True)
    s = sub.add_parser("verify-grader", help="verify <run>/grader_clean is code only, as its MANIFEST says")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--split", required=True)
    sub.add_parser("grader-tooling", help="is the grader's tooling (/opt/evoskill, docker) absent from "
                                          "the evolver image? exit 0 = absent")
    sub.add_parser("dockerfile")
    a = ap.parse_args(argv)
    if a.cmd == "build-inputs":
        print(json.dumps(build_inputs_dir(a.run_dir, a.split), indent=1))
        return 0
    if a.cmd == "verify-inputs":
        c = verify_inputs_dir(a.run_dir, a.split)
        print(c.line())
        return 0 if c.status == "PASS" else 1
    if a.cmd == "build-references":
        print(json.dumps(build_references_dir(a.run_dir, a.split, strict=a.strict), indent=1))
        return 0
    if a.cmd == "verify-references":
        c = verify_references_dir(a.run_dir, a.split, strict=a.strict)
        print(c.line())
        return 0 if c.status == "PASS" else 1
    if a.cmd == "grants":
        tm = evolver_template_mode(a.run_dir)
        print(json.dumps({**evolver_grants(a.run_dir), GRADER_GRANT: evolver_grader_grant(a.run_dir),
                          "template_mode": tm, "oracle_readme_uploaded": not tm}, indent=1))
        return 0
    if a.cmd == "grader-tooling":
        absent, det, rec = evolver_image_grader_tooling()
        print(json.dumps({"absent_from_evolver_image": absent, "detail": det, **rec}, indent=1))
        return 0 if absent else 1
    if a.cmd == "verify-grader":
        c = verify_grader_dir(a.run_dir, a.split)
        print(c.line())
        return 0 if c.status == "PASS" else 1
    ok, det, rec = dockerfile_check()
    print(json.dumps({"ok": ok, "detail": det, **rec}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
