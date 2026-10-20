#!/usr/bin/env python3
"""The loop's side of the benchmark adapter's plug-in (plugin_api version 3): sandboxed run, validation, freeze, verify.

The plug-in is the SINGLE SOURCE of the benchmark's tasks and their sets: the loop does not read the dataset itself,
shared reference files are found by bytes here, and what the plug-in code may do is the sandbox-exec profile's rule
(read-only, offline, no process), not an import allow-list. There is no operator signature: the loop is automatic.
The run's grader, grade(task, snapshot), is the user's own file setup/benchmark_grader.py, handed to the adapter. Here it is only read (static_check_grader) and pinned (SETUP_FILES);
it is imported and called only in the grader's setup, by the leg's grading runner.

  static_check(source)            parses and defines list_tasks(root); read BEFORE the code ever runs
  static_check_grader(source)     parses and defines grade(task, snapshot); the grader file is never run here
  run_plugin(plugin, root)        list_tasks(root) in a child under sandbox-exec (build_profile) -> task dicts
  validate(tasks)                 ids unique; every listed file exists; no input is a reference file (by path or
                                  bytes); tasks sharing a reference file (by bytes) are in ONE set; a train set exists
  check(run_dir)                  the setup files, static (both), run, validate: one report
  freeze(run_dir)                 check, then setup/{PLUGIN_VALIDATION.json, tasks.json, split.json, PLUGIN.json}
  verify_frozen(run_dir)          re-hash everything PLUGIN.json pins; refuse on drift or an edited record
  material_layout(root, split)    the user's material at its paths under /data (the adapter's and the grader's view)

The legs run from setup/tasks.json (cc_leg_lib.tasks_json_questions); the loop's split readers take setup/split.json
(train / val / test / unused, from Task.split). Nothing here imports or calls grade().

CLI (from the repo root):
  python -m src.orchestrator.plugin_runner check  --run-dir outputs/<run> [--data-root R]
  python -m src.orchestrator.plugin_runner freeze --run-dir outputs/<run> [--force]
  python -m src.orchestrator.plugin_runner verify --run-dir outputs/<run> [--recheck]
Prints one JSON report; exit 0 = ok, 1 = refused / problems found, 2 = usage.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from src.orchestrator import plugin_api as A

REPO = Path(__file__).resolve().parents[2]
API_FILE = Path(__file__).resolve().parent / "plugin_api.py"
# the adapter's plug-in, the user's grader file and the canonical interface, all pinned at freeze
SETUP_FILES = ("benchmark_plugin.py", "benchmark_grader.py", "plugin_api.py")
PLUGIN_JSON = "PLUGIN.json"
VALIDATION_JSON = "PLUGIN_VALIDATION.json"
TASKS_JSON = "tasks.json"                  # the frozen task list (every task, its set included)
SPLIT_JSON = "split.json"                  # the loop's split file, materialised from Task.split
# A grading-sandbox validation is OPTIONAL: the loop does not require one at freeze. Its record format and gate table
# stay here for a tool that writes such a record (grading_validation_problems).
GRADING_VALIDATION_JSON = "GRADING_VALIDATION.json"
GRADING_VALIDATION_KIND = "grading_validation"
GRADING_VALIDATION_SCHEMA_PREFIX = "evoskill.plugin_grading_validation/"
GRADING_REQUIRED_HARD_GATES = ("complete", "plugin_valid", "V1_every_golden_passes",
                               "OPEN_references_readable", "V3_no_output_is_not_eligible", "V4_exact_parity",
                               "V5_deterministic", "no_grader_failure_after_regrade")
SPLIT_SETS = ("train", "val", "test")
SPLIT_FILE_KEY = {"train": "train", "validation": "val", "test": "test", None: "unused"}
DROPPED = "dropped"                        # a split-file key older splits carry: tasks in no set
TASK_KEYS = ("id", "category", "instruction", "inputs", "answer_files", "grading", "split")
MAX_RESULT_BYTES = 64 << 20
CHILD_FSIZE_LIMIT = 256 << 20              # RLIMIT_FSIZE inside the child (its temp dir is its only disk)
PROFILE_VERSION = "plugin-list-tasks/2"


class PluginRunError(RuntimeError):
    """list_tasks did not produce a well-formed task list inside the sandbox."""


class FreezeRefused(RuntimeError):
    """freeze() will not write PLUGIN.json."""


class PluginDrift(RuntimeError):
    """verify_frozen(): something PLUGIN.json pins has changed, or the record was edited."""

    def __init__(self, msg: str, report: dict):
        super().__init__(msg)
        self.report = report


# ─────────────────────────────────────────────────────────────────────────────── small helpers
def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=False).encode() + b"\n"


def default_data_root() -> Path:
    """Same resolution as daytona_evolver.data_root(): EVOSKILL_SSBV2_DATA, else $SSBV2_ROOT/data/spreadsheetbench-v2.
    ValueError when the environment names neither (pass the data root, or declare INFRA data.root)."""
    if os.environ.get("EVOSKILL_SSBV2_DATA"):
        return Path(os.environ["EVOSKILL_SSBV2_DATA"])
    if os.environ.get("SSBV2_ROOT"):
        return Path(os.environ["SSBV2_ROOT"]) / "data/spreadsheetbench-v2"
    raise ValueError("no data root: pass --data-root, declare INFRA.json data.root, or set EVOSKILL_SSBV2_DATA / "
                     "SSBV2_ROOT")


# What the user hands over sits under /data by its own name on the adapter's machine, so no path names the operator's
# own machine. The user's grader names those paths, so the grader's setup links the same view; this is its ONE
# definition (setup_ao_probe.vm_layout delegates to it).
MATERIAL_ROOT = "/data"
# The repository is always /data/EvoSkill, whatever the clone's folder is called: the shipped graders and
# instructions name files under that path.
REPO_VIEW_NAME = "EvoSkill"


def material_layout(data_root: Path, split: Path, repo: Path = REPO) -> dict:
    """Mac path -> its path under /data: the data root and the split file under /data by their own names, the
    repository the user instruction names files in at /data/EvoSkill. ValueError when two sources map to one path."""
    lay = {str(data_root): f"{MATERIAL_ROOT}/{data_root.name}", str(split): f"{MATERIAL_ROOT}/{split.name}",
           str(repo): f"{MATERIAL_ROOT}/{REPO_VIEW_NAME}"}
    if len(set(lay.values())) != len(lay):
        raise ValueError(f"two sources map to one path under {MATERIAL_ROOT}: {lay}")
    return lay


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ─────────────────────────────────────────────────────────────────────────────── static check
def _static(source: str, name: str, n_args: int, missing: str, arity: str) -> dict:
    """The source parses and defines a top-level `def <name>` with exactly n_args positional arguments and no
    *args, **kwargs or keyword-only ones. Returns {"ok", "problems", "bytes", "sha256", "top_level_names"}; never
    runs the code."""
    problems: list[str] = []
    rep = {"bytes": len(source.encode()), "sha256": sha256_bytes(source.encode())}
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return {**rep, "ok": False, "problems": [f"does not parse: {e}"], "top_level_names": []}
    top = sorted({n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
                 | {t.id for n in tree.body if isinstance(n, ast.Assign) for t in n.targets if isinstance(t, ast.Name)})
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name), None)
    if fn is None:
        problems.append(missing)
    else:
        a = fn.args
        if len(a.posonlyargs) + len(a.args) != n_args or a.vararg or a.kwarg or a.kwonlyargs:
            problems.append(arity)
    rep.update(top_level_names=top, problems=problems, ok=not problems)
    return rep


def static_check(source: str) -> dict:
    """benchmark_plugin.py: the source parses and defines a top-level `def list_tasks(root)` with exactly one
    argument (see _static). What the code may DO is the sandbox's business (build_profile), not an allow-list's."""
    return _static(source, "list_tasks", 1, "no top-level def list_tasks(root)",
                   "list_tasks must take exactly one argument (root)")


def static_check_grader(source: str) -> dict:
    """benchmark_grader.py (the user's grader): the source parses and defines a top-level
    `def grade(task, snapshot)` with exactly two positional arguments (see _static). AST only: the grader file is
    never imported or run on this machine, so its top level may import what the grader's setup provides."""
    return _static(source, "grade", 2, "no top-level def grade(task, snapshot) in benchmark_grader.py",
                   "grade must take exactly two arguments (task, snapshot)")


# ─────────────────────────────────────────────────────────────────────────────── sandboxed run
# The child: loads the CANONICAL plugin_api (a copy of src/orchestrator/plugin_api.py staged next to a copy of
# the plug-in), imports the plug-in, calls list_tasks(root), type-checks every object against the loop's own
# dataclass and writes plain JSON into its temp dir. It runs `python -I -S -B -X utf8`: no site, no PYTHON*
# variables, no bytecode writes.
CHILD = r'''
import json, resource, sys, traceback
stage, data_root, result_path, fsize = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
sys.path.insert(0, stage)
out = {"problems": [], "tasks": []}
class _Stop(Exception):
    pass
code = 0
try:
    import inspect
    from pathlib import Path
    import plugin_api as A
    if A.__file__ != stage + "/plugin_api.py":
        raise RuntimeError("plugin_api did not load from the staged canonical copy: %r" % A.__file__)
    import benchmark_plugin as P
    prob = out["problems"]
    out["api_version"] = A.API_VERSION
    fn = getattr(P, "list_tasks", None)
    if not callable(fn) or len(inspect.signature(fn).parameters) != 1:
        prob.append("list_tasks is not a one-argument callable")
        raise _Stop()
    tasks = fn(Path(data_root))
    if type(tasks) is not list:
        prob.append("list_tasks returned %s, not list" % type(tasks).__name__)
        raise _Stop()
    for i, t in enumerate(tasks):
        if type(t) is not A.Task:
            prob.append("item %d is %s, not plugin_api.Task" % (i, type(t).__name__))
            continue
        e = []
        for k in ("id", "category", "instruction"):
            if not isinstance(getattr(t, k), str):
                e.append(k)
        for k in ("inputs", "answer_files"):
            v = getattr(t, k)
            if type(v) is not tuple or not all(isinstance(p, Path) and p.is_absolute() for p in v):
                e.append(k + " (must be a tuple of absolute Paths)")
        try:
            items = list(t.grading.items())
        except Exception:
            items = None
        if items is None or not all(isinstance(a, str) and isinstance(b, str) for a, b in items):
            e.append("grading (must map str to str)")
        if t.split is not None and t.split not in A.SPLITS:
            e.append("split (must be one of %s or None)" % (A.SPLITS,))
        if e:
            prob.append("task %d (%r): bad %s" % (i, t.id, ", ".join(e)))
            continue
        out["tasks"].append({"id": t.id, "category": t.category, "instruction": t.instruction,
                             "inputs": [str(p) for p in t.inputs], "answer_files": [str(p) for p in t.answer_files],
                             "grading": dict(items), "split": t.split})
    code = 4 if prob else 0
except _Stop:
    code = 4
except BaseException:
    out["error"] = traceback.format_exc()[-4000:]
    code = 3
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(out, f)
sys.exit(code)
'''


def _sb_str(p) -> str:
    s = str(p)
    if '"' in s or "\\" in s or "\n" in s:
        raise PluginRunError(f"path not usable in a sandbox profile: {s!r}")
    return s


def build_profile(*, tmp, python, prefixes, data_root, setup_dir, repo=REPO, home=None) -> str:
    """The sandbox-exec profile for one list_tasks run. SBPL takes the LAST matching rule, so the order is:
    broad allow -> process/IPC denies -> write deny + temp-dir grant -> read denies on every user/scratch tree
    -> narrow read grants (interpreter, data root, temp dir, ancestor nodes as metadata only) -> named read
    denies (~/.claude, <repo>/outputs, <repo>/.cache) -> the plug-in's own setup dir -> secrets by name,
    anywhere (.env*, *cc_oauth_token*), LAST so no grant can re-open them.

    Measured on macOS 26.5: with these rules python -I -S starts and list_tasks runs; network,
    DNS, fork, exec, signals to other processes, and writes outside `tmp` all fail with EPERM."""
    home = Path(home or Path.home())
    tmp, python = _sb_str(os.path.realpath(tmp)), _sb_str(os.path.realpath(python))
    grants = [_sb_str(os.path.realpath(p)) for p in prefixes] + [_sb_str(os.path.realpath(data_root)), tmp]
    setup = _sb_str(os.path.realpath(setup_dir))
    repo_r, home_r = Path(os.path.realpath(repo)), Path(os.path.realpath(home))
    ancestors = sorted({str(a) for g in grants for a in Path(g).parents})
    L = ["(version 1)", f"; {PROFILE_VERSION}: benchmark plug-in list_tasks, deny-first",
         "(allow default)",
         "; no network, no sockets at all",
         "(deny network*)", "(deny system-socket)",
         "; no child processes; only the interpreter itself may be exec'd (by sandbox-exec)",
         "(deny process-fork)", "(deny process-exec)", f'(allow process-exec (literal "{python}"))',
         "; no signals to, or inspection of, any other process; no system services",
         "(deny signal)", "(allow signal (target self))",
         "(deny process-info*)", "(allow process-info* (target self))",
         "(deny mach-lookup)", "(deny ipc-posix*)", "(deny ipc-sysv*)", "(deny iokit-open)",
         "(deny sysctl-write)", "(deny nvram*)", "(deny user-preference-read)",
         "; writes: only the child's own temp dir",
         "(deny file-write*)", f'(allow file-write* (subpath "{tmp}"))',
         '(allow file-write-data (literal "/dev/null"))',
         "; reads: every user and scratch tree is closed ...",
         '(deny file-read* (subpath "/Users"))', '(deny file-read* (subpath "/Volumes"))',
         '(deny file-read* (subpath "/private/tmp"))', '(deny file-read* (subpath "/private/var/folders"))',
         f'(deny file-read* (subpath "{_sb_str(home_r)}"))',
         "; ... then re-opened for the interpreter, the data root and the temp dir"]
    L += [f'(allow file-read* (subpath "{g}"))' for g in grants]
    L += ["; ancestor directory NODES of those grants: stat only, no listing"]
    L += [f'(allow file-read-metadata (literal "{_sb_str(a)}"))' for a in ancestors]
    L += ["; named denies: win over every grant above",
          f'(deny file-read* (subpath "{_sb_str(home_r / ".claude")}"))',
          f'(deny file-read* (subpath "{_sb_str(repo_r / "outputs")}"))',
          f'(deny file-read* (subpath "{_sb_str(repo_r / ".cache")}"))',
          "; the plug-in's own setup dir (under outputs/) and the temp dir stay readable",
          f'(allow file-read* (subpath "{setup}"))', f'(allow file-read* (subpath "{tmp}"))',
          "; secrets by name, anywhere: LAST, so nothing re-opens them",
          r'(deny file-read* (regex #"/\.env[^/]*(/|$)"))',
          r'(deny file-read* (regex #"cc_oauth_token"))']
    return "\n".join(L) + "\n"


def _python_prefixes(python: str) -> list[str]:
    py = os.path.realpath(python)
    if py == os.path.realpath(sys.executable):
        base = sys.base_prefix
    else:
        r = subprocess.run([py, "-I", "-S", "-c", "import sys; print(sys.base_prefix)"],
                           capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            raise PluginRunError(f"cannot query interpreter {py}: {r.stderr[-300:]}")
        base = r.stdout.strip()
    return sorted({os.path.realpath(base), os.path.dirname(py)})


def _check_task_dict(d) -> str | None:
    if not isinstance(d, dict) or tuple(sorted(d)) != tuple(sorted(TASK_KEYS)):
        return "keys"
    if not all(isinstance(d[k], str) for k in ("id", "category", "instruction")):
        return "string fields"
    if not all(isinstance(d[k], list) and all(isinstance(x, str) for x in d[k]) for k in ("inputs", "answer_files")):
        return "path lists"
    if not isinstance(d["grading"], dict) or not all(isinstance(k, str) and isinstance(v, str)
                                                      for k, v in d["grading"].items()):
        return "grading"
    if d["split"] is not None and d["split"] not in A.SPLITS:
        return "split"
    return None


def run_plugin(plugin_path, data_root, timeout: float = 300, *, python: str | None = None,
               repo=REPO, home=None) -> dict:
    """Run the plug-in's list_tasks(data_root) under sandbox-exec (build_profile). Returns
    {"api_version", "tasks", "n_tasks", "tasks_sha256", "plugin_sha256", "python", "profile"}.
    Raises PluginRunError on a timeout, a crash, a type problem or a malformed result. Never runs unsandboxed."""
    plugin_path, data_root = Path(plugin_path).resolve(), Path(data_root)
    if not plugin_path.is_file():
        raise PluginRunError(f"no plug-in at {plugin_path}")
    if not os.path.exists("/usr/bin/sandbox-exec"):
        raise PluginRunError("sandbox-exec not available: refusing to run plug-in code unsandboxed")
    python = os.path.realpath(python or sys.executable)
    prefixes = _python_prefixes(python)
    tmp = os.path.realpath(tempfile.mkdtemp(prefix="plugin_list_tasks_"))
    try:
        stage = Path(tmp) / "stage"
        stage.mkdir()
        shutil.copyfile(API_FILE, stage / "plugin_api.py")
        shutil.copyfile(plugin_path, stage / "benchmark_plugin.py")
        plugin_sha = sha256_file(stage / "benchmark_plugin.py")
        result = Path(tmp) / "result.json"
        profile = build_profile(tmp=tmp, python=python, prefixes=prefixes, data_root=data_root,
                                setup_dir=plugin_path.parent, repo=repo, home=home)
        prof = Path(tmp) / "profile.sb"
        prof.write_text(profile)
        argv = ["/usr/bin/sandbox-exec", "-f", str(prof), python, "-I", "-S", "-B", "-X", "utf8", "-c", CHILD,
                str(stage), str(os.path.abspath(data_root)), str(result), str(CHILD_FSIZE_LIMIT)]
        env = {"PATH": "/usr/bin:/bin", "HOME": tmp, "TMPDIR": tmp, "LANG": "C", "LC_ALL": "C"}
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, cwd=tmp, env=env,
                               stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            raise PluginRunError(f"list_tasks did not finish within {timeout} s (killed)")
        if not result.is_file():
            raise PluginRunError(f"no result (rc {r.returncode}); stderr tail: {r.stderr[-1500:]}")
        if result.stat().st_size > MAX_RESULT_BYTES:
            raise PluginRunError(f"result is {result.stat().st_size} bytes (cap {MAX_RESULT_BYTES})")
        try:
            res = json.loads(result.read_text(encoding="utf-8"))
        except ValueError as e:
            raise PluginRunError(f"result is not JSON: {e}")
        if not isinstance(res, dict):
            raise PluginRunError("result is not a JSON object")
        if res.get("error"):
            raise PluginRunError(f"list_tasks raised:\n{res['error']}")
        if r.returncode != 0 or res.get("problems"):
            probs = res.get("problems") or []
            raise PluginRunError(f"rc {r.returncode}; {len(probs)} problem(s): {probs[:20]}; "
                                 f"stderr tail: {r.stderr[-800:]}")
        tasks = res.get("tasks")
        if not isinstance(tasks, list):
            raise PluginRunError("result has no task list")
        bad = [(i, why) for i, why in ((i, _check_task_dict(t)) for i, t in enumerate(tasks)) if why]
        if bad:
            raise PluginRunError(f"malformed task dicts: {bad[:10]}")
        return {"api_version": res.get("api_version"), "tasks": tasks, "n_tasks": len(tasks),
                "tasks_sha256": sha256_bytes(canonical_json(tasks)), "plugin_sha256": plugin_sha,
                "python": python, "profile": PROFILE_VERSION}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_list_tasks(plugin_path, data_root, timeout: float = 300, **kw) -> list[dict]:
    """list_tasks(data_root) of the plug-in at `plugin_path`, run sandboxed (see run_plugin). Returns one dict
    per task: {id, category, instruction, inputs: [str], answer_files: [str], grading: {str: str}, split}."""
    return run_plugin(plugin_path, data_root, timeout, **kw)["tasks"]


# ─────────────────────────────────────────────────────────────────────────────── validation
def _block_ids(block) -> list:
    """The task ids of a split block that is a list, or a dict carrying "task_ids"."""
    ids = (block.get("task_ids") if isinstance(block, dict) else block) or []
    return [t if isinstance(t, str) else t["id"] for t in ids]


def _load_split(split) -> tuple[dict, dict]:
    """(split dict, {task_id: role}) of a split FILE (setup/split.json, or an older hand-made split): the same
    roles as daytona_evolver.split_roles(); "dropped" overrides any other role."""
    sp = split if isinstance(split, dict) else json.loads(Path(split).read_text(encoding="utf-8"))
    roles = {}
    for k in ("train", "val", "test", "unused"):
        for t in sp.get(k) or []:
            roles[t if isinstance(t, str) else t["id"]] = k
    for t in _block_ids(sp.get("excluded_prior_run")):
        roles.setdefault(t, "excluded_prior_run")
    for t in _block_ids(sp.get(DROPPED)):
        roles[t] = DROPPED
    return sp, roles


def split_from_tasks(tasks) -> dict:
    """The loop's split file from the task list: {"train", "val", "test", "unused": [ids]} in list order."""
    out = {k: [] for k in (*SPLIT_SETS, "unused")}
    for t in tasks:
        out[SPLIT_FILE_KEY[t["split"]]].append(t["id"])
    return out


def validate(tasks, *, hash_files: bool = True) -> dict:
    """The plug-in's task list on its own terms (the plug-in is the single source of truth). Problems: duplicate
    ids; a split value outside plugin_api.SPLITS; no train task; a listed file missing; an input that is an answer
    file (by path, any task's; and by bytes, when hash_files); tasks that share a reference file (the same bytes,
    or the same real path when not hashed) in different sets. Deterministic output (sorted), so its sha can pin it."""
    problems: list[str] = []
    dups = sorted(k for k, n in Counter(t["id"] for t in tasks).items() if n > 1)
    if dups:
        problems.append(f"duplicate ids: {dups[:10]}")
    bad_split = sorted(t["id"] for t in tasks if t["split"] is not None and t["split"] not in A.SPLITS)
    if bad_split:
        problems.append(f"split not one of {A.SPLITS} / None: {bad_split[:5]}")
    counts = Counter(SPLIT_FILE_KEY.get(t["split"], "?") for t in tasks)
    if not counts.get("train"):
        problems.append("no task is in the train set")
    all_answers = {a for t in tasks for a in t["answer_files"]}
    missing = []
    for t in tasks:
        for p in t["inputs"] + t["answer_files"]:
            if not Path(p).is_file():
                missing.append(f"{t['id']}: {p}")
        clash = sorted(set(t["inputs"]) & all_answers)
        if clash:
            problems.append(f"{t['id']}: an input is an answer file: {clash[:2]}")
    if missing:
        problems.append(f"{len(missing)} missing file(s), e.g. {missing[:4]}")
    sha: dict[str, str] = {}
    if hash_files:
        for t in tasks:
            for p in t["inputs"] + t["answer_files"]:
                if p not in sha and Path(p).is_file():
                    sha[p] = sha256_file(p)
        answer_shas = {sha[a] for t in tasks for a in t["answer_files"] if a in sha}
        for t in tasks:
            same = [p for p in t["inputs"] if sha.get(p) in answer_shas]
            if same:
                problems.append(f"{t['id']}: an input has the same bytes as a reference file: {same[:2]}")
    groups: dict[str, set] = {}
    members: dict[str, set] = {}
    for t in tasks:
        key = "|".join(sorted(sha.get(a, os.path.realpath(a)) for a in t["answer_files"]))
        groups.setdefault(key, set()).add(t["split"])
        members.setdefault(key, set()).add(t["id"])
    spanning = {k: sorted(s - {None}) for k, s in groups.items() if len(s - {None}) > 1}
    for k in sorted(spanning, key=lambda x: sorted(members[x])):
        problems.append(f"tasks sharing a reference file are in different sets {spanning[k]}: {sorted(members[k])[:4]}")
    return {"ok": not problems, "n_tasks": len(tasks),
            "split_counts": {k: int(counts.get(k, 0)) for k in (*SPLIT_SETS, "unused")},
            "per_category": dict(sorted(Counter(t["category"] for t in tasks).items())),
            "hashed_files": len(sha), "n_answer_groups": len(groups), "answer_groups_spanning_sets": len(spanning),
            "n_problems": len(problems), "problems": problems[:200]}


# ─────────────────────────────────────────────────────────────────────────────── check / freeze / verify
def _resolve(run_dir, data_root=None) -> dict:
    """The data root: the argument, else INFRA.json's data.root, else the environment default."""
    run_dir = Path(run_dir).resolve()
    if data_root is None:
        inf = run_dir / "INFRA.json"
        try:
            data_root = (json.loads(inf.read_text()).get("data") or {}).get("root") if inf.is_file() else None
        except (OSError, ValueError):
            data_root = None
    return {"run_dir": run_dir, "setup": run_dir / "setup",
            "data_root": Path(data_root) if data_root else default_data_root()}


def check(run_dir, *, data_root=None, timeout: float = 300, python: str | None = None) -> dict:
    """Every front-door check on outputs/<run>/setup, as one deterministic report {"ok", "problems", ...} that also
    carries the task list ("tasks"). Runs the plug-in only if its source passes static_check; the grader file is
    only read (static_check_grader), never run. Writes nothing."""
    c = _resolve(run_dir, data_root)
    setup, problems = c["setup"], []
    rep = {"run_dir": str(c["run_dir"]), "data_root": str(c["data_root"]), "files": {}, "tasks": []}
    for name in SETUP_FILES:
        p = setup / name
        rep["files"][name] = sha256_file(p) if p.is_file() else None
        if rep["files"][name] is None:
            problems.append(f"setup/{name} missing")
    if rep["files"]["plugin_api.py"] and rep["files"]["plugin_api.py"] != sha256_file(API_FILE):
        problems.append("setup/plugin_api.py differs from src/orchestrator/plugin_api.py")
    src_path = setup / "benchmark_plugin.py"
    st = static_check(src_path.read_text(encoding="utf-8")) if src_path.is_file() else {"ok": False, "problems": []}
    rep["static"] = {k: st.get(k) for k in ("ok", "top_level_names", "bytes")}
    problems += [f"static: {p}" for p in st["problems"]]
    gr_path = setup / "benchmark_grader.py"            # the user's grader, AST only
    gst = (static_check_grader(gr_path.read_text(encoding="utf-8")) if gr_path.is_file()
           else {"ok": False, "problems": []})
    rep["grader_static"] = {k: gst.get(k) for k in ("ok", "top_level_names", "bytes")}
    problems += [f"grader static: {p}" for p in gst["problems"]]
    run = None
    if st["ok"]:
        try:
            run = run_plugin(src_path, c["data_root"], timeout, python=python)
        except PluginRunError as e:
            problems.append(f"run: {e}")
    elif not st["ok"]:
        problems.append("run: not attempted (static check failed)")
    if run is not None:
        rep["run"] = {k: run[k] for k in ("api_version", "n_tasks", "tasks_sha256", "plugin_sha256", "profile")}
        if run["plugin_sha256"] != rep["files"]["benchmark_plugin.py"]:
            problems.append("benchmark_plugin.py changed while it was being checked")
        val = validate(run["tasks"])
        rep["validation"] = {k: v for k, v in val.items() if k != "problems"}
        problems += [f"validate: {p}" for p in val["problems"]]
        rep["tasks"] = run["tasks"]
    rep["n_problems"] = len(problems)
    rep["problems"] = problems[:300]
    rep["ok"] = not problems
    return rep


def _record_sha(record: dict) -> str:
    body = {k: v for k, v in record.items() if k not in ("record_sha256", "operator_signature")}
    return sha256_bytes(canonical_json(body))


def grading_validation_problems(v, *, run_name: str, split_sha256: str | None, grader_sha256: str | None,
                                plugin_sha256: str | None, tasks_sha256: str | None) -> list[str]:
    """[] when `v` (a loaded grading-validation record) is a COMPLETE grading validation
    of THIS plug-in on THIS run: kind/schema, not a subset, freeze_allowed, every blocking gate present and passed,
    and the run / split / grader / plug-in / task list it validated are the ones asked about. Kept for the
    optional grading-validation tool; the loop does not require it at freeze."""
    if not isinstance(v, dict):
        return ["the grading validation is not a JSON object"]
    p = []
    if v.get("kind") != GRADING_VALIDATION_KIND:
        p.append(f"kind {v.get('kind')!r} is not {GRADING_VALIDATION_KIND!r}")
    if not str(v.get("schema") or "").startswith(GRADING_VALIDATION_SCHEMA_PREFIX):
        p.append(f"schema {v.get('schema')!r} is not {GRADING_VALIDATION_SCHEMA_PREFIX}*")
    if v.get("subset") is not False:
        p.append("it validated a SUBSET of the tasks (a smoke or --tasks run is never freeze-eligible)")
    if v.get("freeze_allowed") is not True:
        p.append(f"freeze_allowed is {v.get('freeze_allowed')!r}")
    gates = v.get("gates") if isinstance(v.get("gates"), dict) else {}
    for name in GRADING_REQUIRED_HARD_GATES:
        g = gates.get(name)
        if not isinstance(g, dict):
            p.append(f"blocking gate {name} is missing")
        elif g.get("hard") is not True or g.get("pass") is not True:
            p.append(f"blocking gate {name}: hard={g.get('hard')!r} pass={g.get('pass')!r} ({g.get('detail')!r})"[:300])
    for name, g in sorted(gates.items()):
        if isinstance(g, dict) and g.get("hard") is True and g.get("pass") is not True \
                and name not in GRADING_REQUIRED_HARD_GATES:
            p.append(f"blocking gate {name} did not pass ({g.get('detail')!r})"[:300])
    if v.get("run") != run_name:
        p.append(f"it validated run {v.get('run')!r}, not {run_name!r}")
    if not split_sha256 or v.get("split_sha256") != split_sha256:
        p.append(f"it validated split sha256 {str(v.get('split_sha256'))[:12]}, not {str(split_sha256)[:12]}")
    g = v.get("grader") if isinstance(v.get("grader"), dict) else {}
    if not grader_sha256 or g.get("sha256") != grader_sha256:
        p.append(f"it validated grader sha256 {str(g.get('sha256'))[:12]}, not {str(grader_sha256)[:12]}")
    pl = v.get("plugin") if isinstance(v.get("plugin"), dict) else {}
    if not plugin_sha256 or pl.get("plugin_sha256") != plugin_sha256:
        p.append(f"it validated benchmark_plugin.py sha256 {str(pl.get('plugin_sha256'))[:12]}, not "
                 f"{str(plugin_sha256)[:12]}")
    if not tasks_sha256 or pl.get("tasks_sha256") != tasks_sha256:
        p.append(f"it validated task list sha256 {str(pl.get('tasks_sha256'))[:12]}, not {str(tasks_sha256)[:12]}")
    return p


def freeze(run_dir, *, data_root=None, timeout: float = 300, python: str | None = None, force: bool = False) -> dict:
    """check(), and only if it passes: write setup/PLUGIN_VALIDATION.json (the report without the task list,
    canonical JSON), setup/tasks.json (the task list) and setup/split.json (split_from_tasks), then setup/PLUGIN.json
    pinning benchmark_plugin.py, benchmark_grader.py and plugin_api.py (SETUP_FILES), the canonical plugin_api, and
    the three files just written. Refuses (FreezeRefused) if check fails, or if PLUGIN.json exists and not force (a
    forced re-freeze keeps the old record as PLUGIN.json.superseded.<utc>). No signature: the loop is automatic."""
    c = _resolve(run_dir, data_root)
    setup = c["setup"]
    pj = setup / PLUGIN_JSON
    if pj.exists() and not force:
        raise FreezeRefused(f"{pj} exists (pass force=True / --force to re-freeze; the old record is kept)")
    rep = check(c["run_dir"], data_root=c["data_root"], timeout=timeout, python=python)
    if not rep["ok"]:
        raise FreezeRefused(f"check failed with {rep['n_problems']} problem(s): {rep['problems'][:10]}")
    tasks = rep.pop("tasks")
    vbytes = canonical_json(rep)
    (setup / VALIDATION_JSON).write_bytes(vbytes)
    tbytes = canonical_json(tasks)
    (setup / TASKS_JSON).write_bytes(tbytes)
    split = split_from_tasks(tasks)
    sbytes = canonical_json(split)
    (setup / SPLIT_JSON).write_bytes(sbytes)
    files = {n: {"path": str(setup / n), "sha256": rep["files"][n]} for n in SETUP_FILES}
    record = {
        "what": "frozen benchmark plug-in and the run's grader (plugin_runner.freeze, plugin_api 3); the loop refuses "
                "to use them unless verify_frozen passes: every sha below re-hashes equal and the record is unedited",
        "api_version": rep["run"]["api_version"],
        "run_dir": str(c["run_dir"]),
        "files": files,
        "canonical_plugin_api": {"path": str(API_FILE), "sha256": sha256_file(API_FILE)},
        "validation": {"path": str(setup / VALIDATION_JSON), "sha256": sha256_bytes(vbytes),
                       "n_tasks": rep["run"]["n_tasks"], "tasks_sha256": rep["run"]["tasks_sha256"],
                       "n_problems": rep["n_problems"]},
        "tasks_json": {"path": str(setup / TASKS_JSON), "sha256": sha256_bytes(tbytes), "n_tasks": len(tasks)},
        "split_json": {"path": str(setup / SPLIT_JSON), "sha256": sha256_bytes(sbytes),
                       "counts": {k: len(v) for k, v in split.items()}},
        "data_root": str(c["data_root"]),
        "checked_with": {"python": os.path.realpath(python or sys.executable), "profile": rep["run"]["profile"]},
        "frozen_utc": _now_utc(),
    }
    record["record_sha256"] = _record_sha(record)
    if pj.exists():
        shutil.copyfile(pj, setup / f"{PLUGIN_JSON}.superseded.{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    pj.write_bytes(canonical_json(record))
    return record


def verify_frozen(run_dir, *, recheck: bool = False, timeout: float = 300, python: str | None = None) -> dict:
    """Re-hash everything setup/PLUGIN.json pins. Returns the report; raises PluginDrift (with .report) on any
    drift or an edited record. recheck=True also re-runs check() and requires the same validation bytes (a fresh
    sandboxed run) and the same task list."""
    run_dir = Path(run_dir).resolve()
    pj = run_dir / "setup" / PLUGIN_JSON
    problems: list[str] = []
    rep = {"plugin_json": str(pj)}
    if not pj.is_file():
        rep.update(ok=False, problems=[f"{pj} missing (run freeze)"])
        raise PluginDrift(rep["problems"][0], rep)
    rec = json.loads(pj.read_text(encoding="utf-8"))
    rs = _record_sha(rec)
    rep["record_sha256"] = rs
    if rec.get("record_sha256") != rs:
        problems.append("PLUGIN.json was edited after freeze (record_sha256 does not match its content)")

    def pin(label, path, want):
        if not path or not Path(path).is_file():
            problems.append(f"{label}: {path} missing")
        elif sha256_file(path) != want:
            problems.append(f"{label}: {path} changed (sha256 {sha256_file(path)[:12]} != pinned {str(want)[:12]})")

    for name, f in (rec.get("files") or {}).items():
        pin(name, f.get("path"), f.get("sha256"))
    missing = sorted(set(SETUP_FILES) - set(rec.get("files") or {}))
    if missing:
        problems.append(f"PLUGIN.json pins {sorted(rec.get('files') or {})}, lacking {missing}")
    api = rec.get("canonical_plugin_api") or {}
    pin("canonical plugin_api", str(API_FILE), api.get("sha256"))
    if (rec.get("files") or {}).get("plugin_api.py", {}).get("sha256") != api.get("sha256"):
        problems.append("setup/plugin_api.py was not the canonical file at freeze")
    if rec.get("api_version") != A.API_VERSION:
        problems.append(f"PLUGIN.json records api_version {rec.get('api_version')!r}, not {A.API_VERSION} (re-freeze)")
    for label, key in (("validation", "validation"), ("tasks.json", "tasks_json"), ("split.json", "split_json")):
        d = rec.get(key) or {}
        pin(label, d.get("path"), d.get("sha256"))
    if recheck and not problems:
        try:
            again = check(run_dir, data_root=rec["data_root"], timeout=timeout, python=python)
            tasks = again.pop("tasks")
            if sha256_bytes(canonical_json(again)) != (rec.get("validation") or {}).get("sha256"):
                problems.append(f"recheck: a fresh check differs from PLUGIN_VALIDATION.json "
                                f"({again['n_problems']} problem(s) now)")
            if sha256_bytes(canonical_json(tasks)) != (rec.get("tasks_json") or {}).get("sha256"):
                problems.append("recheck: a fresh list_tasks differs from the frozen tasks.json")
        except (PluginRunError, FreezeRefused) as e:
            problems.append(f"recheck: {e}")
        rep["rechecked"] = True
    rep["problems"] = problems
    rep["ok"] = not problems
    if problems:
        raise PluginDrift(f"{len(problems)} problem(s): {problems[:5]}", rep)
    return rep


def frozen_tasks(run_dir) -> list[dict]:
    """The frozen task list (setup/tasks.json) after verify_frozen: what every leg runs from."""
    verify_frozen(run_dir)
    return json.loads((Path(run_dir).resolve() / "setup" / TASKS_JSON).read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────────────────────── CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.orchestrator.plugin_runner", description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("check", "freeze", "verify"))
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--data-root", default=None, help="default: the run's INFRA.json data.root, else the environment")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--python", default=None, help="interpreter for the sandboxed child (default: this one)")
    ap.add_argument("--force", action="store_true", help="freeze: replace an existing PLUGIN.json")
    ap.add_argument("--recheck", action="store_true", help="verify: also re-run check and compare")
    a = ap.parse_args(argv)
    kw = dict(data_root=a.data_root, timeout=a.timeout, python=a.python)
    try:
        if a.command == "check":
            out = check(a.run_dir, **kw)
            out.pop("tasks", None)
            code = 0 if out["ok"] else 1
        elif a.command == "freeze":
            rec = freeze(a.run_dir, force=a.force, **kw)
            out = {"ok": True, "plugin_json": str(Path(a.run_dir).resolve() / "setup" / PLUGIN_JSON),
                   "record_sha256": rec["record_sha256"], "n_tasks": rec["validation"]["n_tasks"],
                   "split_counts": rec["split_json"]["counts"]}
            code = 0
        else:
            out = verify_frozen(a.run_dir, recheck=a.recheck, timeout=a.timeout, python=a.python)
            code = 0
    except PluginDrift as e:
        out, code = e.report, 1
    except (FreezeRefused, PluginRunError) as e:
        out, code = {"ok": False, "refused": str(e)}, 1
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return code


if __name__ == "__main__":
    sys.exit(main())
