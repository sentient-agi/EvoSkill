"""The PURE half of the Claude Code LEG DRIVER for the loop: how a leg of Claude Code rollouts is recorded on disk for
the loop's stateful program (loop.py's leg checks, src/orchestrator/evo_layout.py). No Daytona SDK, no network, no
credential, no model call; stdlib + cc_solver_lib (which imports the stdlib-only container controller) +
src/orchestrator/plugin_runner (the frozen plug-in's pins). scripts/daytona/cc_leg_driver.py is the process. Also here:
skill delivery, the rate-limit wait plan (rate_limit_wait), the LegLayout interface and the per-run pid file
a live leg process holds (hold_leg_pid / leg_alive).

Every completed rollout is graded by the run's frozen grade(task, snapshot) (setup/benchmark_grader.py, the user's),
which returns a number from 0 to 1 or raises. ONE score chain: the grader's result line -> STATE.json -> the two
record copies -> the leg's per_task, where a rollout whose grade() raised twice is "grader_failed" in STATE and null
in the records and per_task (never a number).

WHAT A LEG IS ON DISK (RUN = the run dir, NAME its name, TAG = <NAME>_v<N>_<key> | <NAME>_base_<key>):
  RUN/<TAG>.json                                  the leg -- written LAST, atomically, only after verify()
  STORE = <store_root>/<NAME>/<TAG>/              STATE.json (every unit's score and snapshot sha), runs/<task_>__s<j>/
                                                  (a unit's session; container_out.tgz = the snapshot that was graded),
                                                  s01/grade/ (the grader's job and result lines, downloaded)
  STORE/s01/cache/<BUCKET>/<stem>.json            the record
  CACHE = .cache/<spec.name>_e<N>_cache/<BUCKET>/ the same records, byte-identical
The cache record is a "direct"-SDK record: cache_key.sdk="direct", model = spec.solver_model, effort "",
trace.messages in the direct harness's OpenAI format (cc_solver_lib.to_messages) + the raw stream kept in
trace.stream_jsonl, trace.inline_score = the grader's score (null for grader_failed). The record stem is
outcomes._record_path_stem(q, "direct", MODEL, "", k) with k = the 0-based sample index (cc unit sample_idx - 1).
"""
from __future__ import annotations

import datetime as _dt
import fcntl
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import cc_solver_lib as L  # noqa: E402

HARNESS = "claude_code"
SDK, EFFORT, RECORD_VERSION = "direct", "", "1.0"
CC_VERSION = L.ctl.CC_VERSION
CC_SHA256 = L.ctl.CC_SHA256
# /2: a unit row carries `score` (a number from 0 to 1 | GRADER_FAILED | None = not graded yet) and `snapshot_sha256`
# (runs/<ud>/container_out.tgz, the bytes graded); a /1 store (pass/fail verdicts, no score) is never resumed
STATE_SCHEMA = "evoskill.cc_leg_state/2"
FAILED_SUFFIX = ".FAILED.json"
DRYRUN_SUFFIX = ".DRYRUN.json"
DRIVER = "scripts/daytona/cc_leg_driver.py"
COMPARABILITY_CLASS = "x86_64 Daytona sandbox (Claude Code solver)"
STORE_ROOT_DEFAULT = REPO / ".cache" / "ssbv2_daytona"          # RUNTIME daytona_leg.store_root default
CACHE_ROOT_DEFAULT = REPO / ".cache"
GRADER_FAILED = "grader_failed"          # = plugin_api.GRADER_FAILED: the STATE score when grade() raised twice
PASS_THRESHOLD = 0.999                   # = outcomes.PASS_THRESHOLD: a pass is full marks; a fraction is a fail
SKILLS_VM = f"{L.ctl.VM_ROOT}/skills"                            # /evo/skills: the promoted set, read-only
SKILLS_LINK = f"{L.WS_CONTAINER}/.claude/skills"                  # /evo/work/.claude/skills -> /evo/skills
# Claude Code's implementation of skills is canonical: the loop has NO skill rules of its own -- no name pattern, reserved name, frontmatter
# key, description or listing limit, "!`" line, size cap, or SKILL.md-less dir / stray file refusal. Claude Code decides
# what loads (https://code.claude.com/docs/en/skills.md). The upload itself refuses only a symlink (skill_rows).
# the rate-limit wait (rate_limit_wait): woken this long after the reported reset; a 429 that reported no reset waits
# PAUSE_NO_READING_S; the pending units are assumed to need at least PAUSE_MIN_REMAINING_S after the wait
PAUSE_MARGIN_S = 120
PAUSE_NO_READING_S = 1800
PAUSE_MIN_REMAINING_S = 20 * 60
# the longest a leg waits for a reported reset (one five-hour window). The leg's machine lease reserves this much
# time (cc_leg_driver need_s), so a wait never outlives the lease; a reset further away (a weekly limit) FAILS the leg.
WAIT_MAX_S = 5 * 3600
LEG_PID_DIR = Path(".machine") / "legs"   # <run>/.machine/legs/<tag>.pid: held (flock) by the live leg process
STATE_UNIT_STATES = ("queued", "setup", "running", "collected", "graded", "failed")

# what a written leg JSON guarantees (recorded in it as daytona.pool.result_guarantee)
CC_RESULT_GUARANTEE = (
    "CLAUDE CODE LEG ON THE RUN'S ONE POOL MACHINE: this result was written when every unit of this leg had a "
    "completed session (a result event; api_error sessions were re-run), every completed unit's workspace snapshot "
    "was graded IN PLACE on the pool machine by the run's frozen grade(task, snapshot) (pinned by setup/PLUGIN.json) "
    "in the grader's setup, a rollout whose grade() failed twice is recorded and listed as grader_failed (never a "
    "number), every container of this leg was torn down and wiped (verified), every session's init event was clean "
    "for its leg kind, the token prefix was found nowhere, and STATE.json, both record copies and per_task agree "
    "sample by sample (re-read before this file was written). The pool machine stays (it serves the evolver "
    "container and the other leg); the machine's deletion is verified in the pool host's pool_record.json when the "
    "run ends.")

# re-exported: callers name them under cc_leg_lib as well as cc_solver_lib
cc_bucket = L.cc_bucket
to_messages = L.to_messages
model_replies = L.model_replies


# ───────────────────────────────────────────────────────────────────────────── the loop's readers, by path
_MODS: dict = {}


def _by_path(name: str):
    """src/orchestrator/<name>.py loaded by path (stdlib-only modules: outcomes, rollout_evidence, skill_library).
    Loaded by path, not imported as a package: the leg driver runs under the Daytona interpreter, which has no
    src/ package deps, and rollout_evidence loads its own siblings by path the same way."""
    if name not in _MODS:
        p = REPO / "src" / "orchestrator" / f"{name}.py"
        spec = importlib.util.spec_from_file_location(f"evoskill_{name}_for_cc_leg", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODS[name] = mod
    return _MODS[name]


def outcomes_mod():
    return _by_path("outcomes")


def rollout_evidence_mod():
    return _by_path("rollout_evidence")


def skill_library_mod():
    return _by_path("skill_library")


def plugin_runner_mod():
    """src/orchestrator/plugin_runner.py, imported as the package module (it imports plugin_api from the package; the
    standard library otherwise), so its PluginDrift is the one every caller catches."""
    if str(REPO) not in sys.path:
        sys.path.append(str(REPO))
    from src.orchestrator import plugin_runner
    return plugin_runner


def frozen_plugin(run_dir) -> dict:
    """The run's frozen benchmark plug-in, after plugin_runner.verify_frozen re-hashed every pin of setup/PLUGIN.json
    (legs execute plug-in code: the grader). {"files": PLUGIN.json's files {name:
    {path, sha256}} (benchmark_plugin.py, benchmark_grader.py, plugin_api.py), "tasks": the setup/tasks.json rows,
    "tasks_json_sha256"}. A drift or an edited record raises ValueError (re-freeze)."""
    PR = plugin_runner_mod()
    try:
        PR.verify_frozen(run_dir)
    except PR.PluginDrift as e:
        raise ValueError(f"the run's frozen plug-in does not verify ({e}); re-freeze it "
                         f"(python -m src.orchestrator.plugin_runner freeze)") from e
    setup = Path(run_dir).resolve() / "setup"
    rec = json.loads((setup / PR.PLUGIN_JSON).read_text(encoding="utf-8"))
    data = (setup / PR.TASKS_JSON).read_bytes()
    return {"files": {n: {"path": f["path"], "sha256": f["sha256"]} for n, f in rec["files"].items()},
            "tasks": json.loads(data), "tasks_json_sha256": hashlib.sha256(data).hexdigest()}


def tasks_json_questions(tasks: list) -> tuple:
    """(questions {task_id: json string}, idx {task_id: {input, golden, answer_position, category, split}}) from the
    frozen task list. The question string keeps the key set of src/agent_profiles/ssbv2_agent/dataset.py (task_id,
    category, instruction, spreadsheet_path, spreadsheet_filename), so every reader of a cache record finds them;
    one input file and one reference file per task (the leg image's grade worker takes exactly that).
    A task with NO input and one reference file (a tau3 banking task: its tau2
    task file) is a conversation, not a workbook: its question carries task_id, category and instruction only, its
    idx row input None and answer_position None."""
    questions, idx = {}, {}
    for t in tasks:
        if not t["inputs"] and len(t["answer_files"]) == 1:
            questions[t["id"]] = json.dumps({"task_id": t["id"], "category": t["category"],
                                             "instruction": t["instruction"]})
            idx[t["id"]] = {"input": None, "golden": Path(t["answer_files"][0]), "answer_position": None,
                            "category": t["category"], "split": t.get("split")}
            continue
        if len(t["inputs"]) != 1 or len(t["answer_files"]) != 1:
            raise ValueError(f"{t['id']}: {len(t['inputs'])} input(s) / {len(t['answer_files'])} reference file(s); "
                             f"the leg runs one of each")
        inp, gold = t["inputs"][0], t["answer_files"][0]
        questions[t["id"]] = json.dumps({"task_id": t["id"], "category": t["category"], "instruction": t["instruction"],
                                         "spreadsheet_path": inp, "spreadsheet_filename": Path(inp).name})
        idx[t["id"]] = {"input": Path(inp), "golden": Path(gold), "answer_position": t["grading"].get("answer_position"),
                        "category": t["category"], "split": t.get("split")}
    return questions, idx


# ───────────────────────────────────────────────────────────────────────────── small helpers
def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def naive_utc(when) -> str:
    """created_at as a Daytona runner writes it: a NAIVE UTC isoformat ("YYYY-MM-DDTHH:MM:SS.ffffff"). Accepts an
    epoch number, a datetime (aware or naive-UTC), or an ISO string ("...Z" / "+00:00" / naive = UTC)."""
    if isinstance(when, (int, float)):
        t = _dt.datetime.fromtimestamp(float(when), _dt.timezone.utc)
    elif isinstance(when, _dt.datetime):
        t = when if when.tzinfo is None else when.astimezone(_dt.timezone.utc)
    else:
        s = str(when).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        t = _dt.datetime.fromisoformat(s)
        if t.tzinfo is not None:
            t = t.astimezone(_dt.timezone.utc)
    return t.replace(tzinfo=None).isoformat()


def is_score(x) -> bool:
    """A grader's score as the loop records it: a finite number from 0 to 1 (not a bool)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0 <= x <= 1


def score_agrees(state_score, value) -> bool:
    """A STATE score (a number from 0 to 1, or GRADER_FAILED) and a record's inline_score / a per_task entry (that
    number as a float, or None for GRADER_FAILED) say the same thing."""
    if state_score == GRADER_FAILED:
        return value is None
    return is_score(state_score) and is_score(value) and float(value) == float(state_score)


def epoch_of(created_at: str) -> float:
    """The epoch seconds of a naive-UTC created_at (outcomes._created_epochs' UTC reading)."""
    return _dt.datetime.fromisoformat(created_at).replace(tzinfo=_dt.timezone.utc).timestamp()


def utc_str(ts: float) -> str:
    return _dt.datetime.fromtimestamp(float(ts), _dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_bytes_atomic(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_json_atomic(path: Path, obj, indent=1) -> None:
    write_bytes_atomic(path, (json.dumps(obj, indent=indent, ensure_ascii=False, default=str) + "\n").encode("utf-8"))


def spec_of(run_dir) -> dict:
    """SPEC.json's `spec` block (solver_model, n_samples, split_path, split_sha256, name, agent_turns)."""
    d = json.loads((Path(run_dir) / "SPEC.json").read_text())
    return d.get("spec") if isinstance(d.get("spec"), dict) else d


def split_ids(split_path, key: str) -> list:
    """split[key] of the split FILE in file order (loop.split_ids' rule: a list of ids, of {"id"} rows, or a record
    with "task_ids")."""
    sp = json.loads(Path(split_path).read_text())
    v = sp.get(key)
    if isinstance(v, dict):
        v = v.get("task_ids")
    if not isinstance(v, list):
        return []
    return [x if isinstance(x, str) else x["id"] for x in v]


def leg_tag(run_name: str, epoch: int, key: str) -> str:
    """The loop's leg name: <NAME>_v<N>_<key>, or <NAME>_base_<key> for epoch 0 (EpochPaths.leg / RUNTIME base_leg)."""
    return f"{run_name}_base_{key}" if int(epoch) == 0 else f"{run_name}_v{int(epoch)}_{key}"


def cache_dir_for(spec_name: str, epoch: int, cache_root=None) -> Path:
    """.cache/<spec.name>_e<N>_cache (loop.py's EVOSKILL_CACHE_DIR / digest_caches naming)."""
    return Path(cache_root or CACHE_ROOT_DEFAULT) / f"{spec_name}_e{int(epoch)}_cache"


def tasks_dir_for(tag: str, cache_root=None) -> Path:
    """.cache/ssbv2_<TAG>_tasks (the runner's TASKS_CACHE naming; digest_snapshot_roots = <this>/_snapshots)."""
    return Path(cache_root or CACHE_ROOT_DEFAULT) / f"ssbv2_{tag}_tasks"


def store_for(run_name: str, tag: str, store_root=None) -> Path:
    return Path(store_root or STORE_ROOT_DEFAULT) / run_name / tag


def skills_tree_sha256(skills_dir) -> str:
    """sha256(json.dumps(skill_library.tree(dir), sort_keys=True)) for a skill leg; sha256(b"") for a base leg."""
    if skills_dir is None:
        return hashlib.sha256(b"").hexdigest()
    tree = skill_library_mod().tree(skills_dir)
    return hashlib.sha256(json.dumps(tree, sort_keys=True).encode("utf-8")).hexdigest()


def bucket_for(model: str, instruction: str, skills_dir=None, cc_version: str = CC_VERSION) -> str:
    """BUCKET: cc_bucket(CC_VERSION, MODEL, sha256(the solver instruction text), skills_tree_sha)."""
    return cc_bucket(cc_version, model, sha256_text(instruction), skills_tree_sha256(skills_dir))


def sample_index(unit: str) -> tuple:
    """(task, k) with k the 0-based sample index of a <task>#s<j> unit (j = 1..N); the loop, RunCache and
    outcomes.py count samples from 0, cc_solver's units from 1."""
    task, j = L.parse_unit(unit)
    if j is None:
        raise ValueError(f"{unit!r}: a leg unit needs a sample index (<task>#s<k>)")
    return task, int(j) - 1


def record_stem(question: str, model: str, k: int) -> str:
    """outcomes._record_path_stem(q, "direct", MODEL, "", k) == RunCache._get_cache_path's file stem."""
    return outcomes_mod()._record_path_stem(question, SDK, model, EFFORT, int(k))


def question_hash(question: str, model: str) -> str:
    """cache_key.question_hash as RunCache writes it (run_cache.py:863-870): sha256 of the keyed string WITHOUT the
    sample, hash_length 12."""
    return hashlib.sha256(f"{question.strip()}|sdk={SDK}|model={model}".encode("utf-8")).hexdigest()[:12]


def budget_exhausted_text(subtype, max_turns: int = L.MAX_TURNS) -> str:
    """The direct harness's budget string for a turn-capped rollout (executor.py) iff Claude Code ended with
    subtype error_max_turns; "" otherwise. rollout_evidence._direct_termination reads exactly this prefix."""
    return f"Turn budget exhausted ({int(max_turns)}/{int(max_turns)} turns)" if subtype == "error_max_turns" else ""


def solver_harness_text(model: str = L.MODEL, cc_version: str = CC_VERSION) -> str:
    return f"claude_code {cc_version} ({model}, --thinking-display {L.THINKING_DISPLAY})"


# ───────────────────────────────────────────────────────────────────────────── the cache record (2.5)
def build_trace(*, stream_text: str, user_message: str, result: dict, score: float | None,
                output_saved: bool, model: str = L.MODEL, max_turns: int = L.MAX_TURNS, cc: dict | None = None,
                cc_version: str = CC_VERSION) -> dict:
    """The record's `trace` from a unit's stream + user message + RESULT.json (`result`). inline_score = the
    grader's `score` as a float, None for a grader_failed rollout. output.finished stays bool(RESULT.output), a descriptive record only."""
    init = result.get("init") or {}
    return {"messages": to_messages(stream_text, user_message),
            "num_turns": model_replies(stream_text),
            "num_turns_cc_reported": result.get("num_turns"),
            "budget_exhausted": budget_exhausted_text(result.get("subtype"), max_turns),
            "inline_score": None if score is None else float(score),
            "total_cost_usd": result.get("cost"), "duration_ms": result.get("duration_ms"),
            "model": model, "is_error": bool(result.get("is_error")),
            "result": result.get("result_text_head"), "output": {"finished": bool(output_saved)},
            "tools": list(init.get("tools") or []), "tool_schemas": None,
            "session_id": init.get("session_id"), "stream_jsonl": stream_text,
            "solver_harness": solver_harness_text(model, cc_version),
            "cc": dict(cc or {})}


def build_record(*, question: str, model: str, bucket: str, trace: dict, created_at) -> dict:
    """The record body: version, created_at (naive UTC), cache_key {tree_hash = BUCKET,
    question_hash, question, sdk direct, model, effort ""}, trace."""
    return {"version": RECORD_VERSION, "created_at": naive_utc(created_at),
            "cache_key": {"tree_hash": bucket, "question_hash": question_hash(question, model), "question": question,
                          "sdk": SDK, "model": model, "effort": EFFORT},
            "trace": trace}


def record_bytes(record: dict) -> bytes:
    return json.dumps(record, ensure_ascii=False).encode("utf-8")


# ───────────────────────────────────────────────────────────────────────────── the leg JSON (2.6)
def _scores_by_task(task_ids: list, n_samples: int, units_results: dict) -> tuple:
    """({task: [score | None x N]}, {task: [exhausted x N]}, problems); None = the rollout is grader_failed."""
    scores, exh, probs = {}, {}, []
    seen = set()
    for u, r in units_results.items():
        task, k = sample_index(u)
        if task not in task_ids:
            probs.append(f"{u}: not a task of this leg")
            continue
        if not 0 <= k < n_samples:
            probs.append(f"{u}: sample index {k} outside 0..{n_samples - 1}")
            continue
        seen.add((task, k))
        scores.setdefault(task, [None] * n_samples)[k] = None if r["score"] is None else float(r["score"])
        exh.setdefault(task, [False] * n_samples)[k] = bool(r.get("turn_exhausted"))
    missing = [f"{t}#s{k + 1}" for t in task_ids for k in range(n_samples) if (t, k) not in seen]
    if missing:
        probs.append(f"{len(missing)} unit(s) without a result: {missing[:4]}")
    return scores, exh, probs


def _passes(x) -> bool:
    return x is not None and x >= PASS_THRESHOLD


def leg_counts(per_task: dict, task_ids: list) -> dict:
    """The leg's counts from its per_task: a pass is a number >= PASS_THRESHOLD (full
    marks); pass3 = tasks with any pass; all3 = tasks whose N entries are all passes (a null fails it); grader_failed
    = the sorted "<task>#s<j>" of the null entries (never-graded, not a moved denominator)."""
    rows = [(t, per_task.get(str(i)) or []) for i, t in enumerate(task_ids)]
    return {"sample_pass": sum(1 for _t, v in rows for x in v if _passes(x)),
            "sample_total": sum(len(v) for _t, v in rows),
            "pass3": sum(1 for _t, v in rows if any(_passes(x) for x in v)),
            "all3": sum(1 for _t, v in rows if v and all(_passes(x) for x in v)),
            "grader_failed": sorted(f"{t}#s{k + 1}" for t, v in rows for k, x in enumerate(v) if x is None)}


def assemble_leg(*, tag: str, key: str, epoch: int, model: str, budget: int, n_samples: int, task_ids: list,
                 units_results: dict, skills_dir, runtime_env: dict, solver_harness: dict, daytona: dict,
                 extra: dict | None = None) -> dict:
    """The leg JSON. `units_results` = {unit: {"score": float | None, "turn_exhausted": bool}}, None
    = grader_failed; per_task keys "0".."n-1" in task_ids order (the split's file order), N entries each in sample
    order (the float, or null), counts per leg_counts, leg["grader_failed"] = the null units. Raises ValueError
    when a unit is missing or foreign: a leg with a hole is never assembled (the loop's _leg_on_split rule)."""
    n = len(task_ids)
    scores, exh, probs = _scores_by_task(list(task_ids), int(n_samples), units_results)
    if probs:
        raise ValueError("leg cannot be assembled: " + "; ".join(probs[:4]))
    per_task = {str(i): list(scores[t]) for i, t in enumerate(task_ids)}
    c = leg_counts(per_task, list(task_ids))
    leg = {"tag": tag, "key": key, "epoch": int(epoch), "model": model, "budget": int(budget),
           "n_samples": int(n_samples), "n": n, "task_ids": list(task_ids), "per_task": per_task,
           "sample_pass": c["sample_pass"], "sample_total": n * int(n_samples),
           "pass3": c["pass3"], "all3": c["all3"], "grader_failed": c["grader_failed"],
           "turn_exhausted": sum(1 for t in task_ids for x in exh[t] if x),
           "episodes_excluded_infra": 0, "episodes_timeout_scored_zero": 0,
           "skills_dir": str(skills_dir) if skills_dir else None,
           "origin": {"cc_solver": n * int(n_samples)}, "written_utc": L.now_utc(),
           "runtime_env": dict(runtime_env), "solver_harness": dict(solver_harness), "daytona": dict(daytona)}
    if extra:
        for k, v in extra.items():
            leg.setdefault(k, v)
    return leg


def solver_harness_record(results: list, *, model: str = L.MODEL, max_turns: int = L.MAX_TURNS,
                          cc_version: str = CC_VERSION, skills_delivery: str | None = None,
                          skills_delivery_facts: dict | None = None) -> dict:
    """`solver_harness` from the units' RESULT.json dicts: the tool union, argv flags, env keys.
    skills_delivery / _facts are ABSENT on a base leg (evolver_template reads the string when present)."""
    tools, flags, env_keys = set(), [], set()
    for r in results:
        tools |= set(((r.get("init") or {}).get("tools") or []))
        env_keys |= set((r.get("env") or {}).keys())
        if not flags and r.get("argv_flags"):
            flags = list(r["argv_flags"])
    h = {"kind": HARNESS, "version": cc_version, "model": model, "max_turns": int(max_turns),
         "thinking_display": L.THINKING_DISPLAY, "prompt_delivery": L.PROMPT_DELIVERY, "tools": sorted(tools),
         "argv_flags": flags, "env_keys": sorted(env_keys), "disallowed_tools": list(L.CC_DISALLOWED)}
    if skills_delivery:
        h["skills_delivery"] = skills_delivery
    if skills_delivery_facts:
        h["skills_delivery_facts"] = dict(skills_delivery_facts)
    return h


# "grader" = {worker, runner_sha256, plugin shas, interpreter}; scorer_sha256 stays a key (None on a leg graded by
# the run's grade())
RUNTIME_ENV_KEYS = ("platform", "comparability_class", "driver", "sandbox_id", "sandbox_name", "readback", "image",
                    "libreoffice", "solver_stack", "solver_python", "grading_openpyxl", "runner_python",
                    "scorer_sha256", "grader", "shim_sha256", "split_file", "split_sha256", "skills", "cache_dir",
                    "knobs", "infra", "egress", "grade_selftest", "cc_version", "cc_sha256", "thread_env")


def runtime_env_record(**kw) -> dict:
    """`runtime_env` with every key present (None when unknown). solver_stack (3 tokens) and
    libreoffice are REQUIRED on the base leg (evolver_template.machine_facts_for): refused when missing."""
    out = {k: None for k in RUNTIME_ENV_KEYS}
    out.update({"platform": "daytona", "comparability_class": COMPARABILITY_CLASS, "driver": DRIVER})
    unknown = sorted(set(kw) - set(RUNTIME_ENV_KEYS))
    if unknown:
        raise ValueError(f"runtime_env: unknown keys {unknown}")
    out.update(kw)
    stack = str(out.get("solver_stack") or "").split()
    if len(stack) != 3 or not out.get("libreoffice"):
        raise ValueError(f"runtime_env needs solver_stack (3 tokens) and libreoffice; got {out.get('solver_stack')!r} / "
                         f"{out.get('libreoffice')!r}")
    return out


def cache_env_record(*, model: str, bucket: str, skills_dir, tag: str, cc_version: str = CC_VERSION) -> dict:
    """CACHE/CACHE_ENV.json (the digest builder skips files without cache_key: build_rollout_digests.py:277)."""
    return {"solver_harness": HARNESS, "cc_version": cc_version, "model": model, "bucket": bucket,
            "skills_dir": str(skills_dir) if skills_dir else None, "tag": tag, "written_utc": L.now_utc()}


# ───────────────────────────────────────────────────────────────────────────── LegLayout (9)
class LegLayout:
    """Where one leg's records, store and cache are, and the writers for them (the leg driver's, at grade time).

      LegLayout(run_dir, tag, key, epoch, bucket, skills_dir, store, cache_dir, tasks_dir, *, questions)
        .write_sample(unit, record_trace, created_at, *, question, model)   the two record copies
        .kept_sample_problems(unit, question, *, score)                     a written sample against its STATE score
        .write_leg(units_results, runtime_env, solver_harness, daytona, problems) -> Path (the leg, or the FAILED file)
        .verify(leg_dict) -> problems (STATE.json <-> both records <-> per_task, and the counts)
    The model, budget and n_samples default to SPEC.json's (solver_model, agent_turns, n_samples). `questions` =
    {task_id: the record's question string} (tasks_json_questions): how verify() finds each unit's record.
    tasks_dir is only recorded (daytona.tasks_dir) and `scratch` is unused: no snapshot dirs are written."""

    def __init__(self, run_dir, tag: str, key: str, epoch: int, bucket: str, skills_dir, store, cache_dir, tasks_dir,
                 *, model: str | None = None, n_samples: int | None = None, budget: int | None = None,
                 task_ids: list | None = None, split_path=None, scratch=None, questions: dict | None = None):
        # every path ABSOLUTE: the leg JSON carries them (daytona.store is where STATE.json and the runs are, read
        # from whatever cwd the loop or a reader runs in)
        self.run_dir = Path(run_dir).resolve()
        self.tag, self.key, self.epoch, self.bucket = str(tag), str(key), int(epoch), str(bucket)
        if not re.fullmatch(r"[0-9a-f]{12}", self.bucket):
            raise ValueError(f"bucket {bucket!r}: 12 hex")
        if self.key not in ("train", "val", "test"):          # test: the held-out leg
            raise ValueError(f"key {key!r}: train | val | test")
        self.skills_dir = Path(skills_dir).resolve() if skills_dir else None
        self.store, self.cache_dir, self.tasks_dir = (Path(store).resolve(), Path(cache_dir).resolve(),
                                                      Path(tasks_dir).resolve())
        sp = {}
        if (self.run_dir / "SPEC.json").is_file():
            sp = spec_of(self.run_dir)
        self.model = model or sp.get("solver_model") or L.MODEL
        self.n_samples = int(n_samples if n_samples is not None else sp.get("n_samples") or L.TRAIN_SAMPLES)
        self.budget = int(budget if budget is not None else sp.get("agent_turns") or L.MAX_TURNS)
        self.split_path = Path(split_path) if split_path else (Path(sp["split_path"]) if sp.get("split_path") else None)
        self.task_ids = list(task_ids) if task_ids is not None else (
            split_ids(self.split_path, self.key) if self.split_path else [])
        self.scratch = Path(scratch) if scratch else None
        self.questions = dict(questions or {})
        self.written: dict = {}

    # -- paths ----------------------------------------------------------------------------------------------------
    @property
    def attempt(self) -> Path:
        return self.store / "s01"

    @property
    def leg_path(self) -> Path:
        return self.run_dir / f"{self.tag}.json"

    @property
    def failed_path(self) -> Path:
        return self.run_dir / f"{self.tag}{FAILED_SUFFIX}"

    @property
    def state_path(self) -> Path:
        return self.store / "STATE.json"

    def record_paths(self, stem: str) -> list:
        return [self.attempt / "cache" / self.bucket / f"{stem}.json", self.cache_dir / self.bucket / f"{stem}.json"]

    # -- writers --------------------------------------------------------------------------------------------------
    def write_sample(self, unit: str, record_trace: dict, created_at, *, question: str,
                     model: str | None = None) -> dict:
        """One graded sample = its record, written twice byte-identical (STORE/s01/cache/<BUCKET>/<stem>.json AND
        CACHE/<BUCKET>/<stem>.json), nothing else: the snapshot that was graded is the
        unit's runs/<ud>/container_out.tgz (STATE snapshot_sha256); no DIFF.txt, no snapshot dirs."""
        task, k = sample_index(unit)
        if json.loads(question).get("task_id") != task:
            raise ValueError(f"{unit}: the question names {json.loads(question).get('task_id')!r}")
        rec = build_record(question=question, model=model or self.model, bucket=self.bucket, trace=record_trace,
                           created_at=created_at)
        data = record_bytes(rec)
        stem = record_stem(question, model or self.model, k)
        for p in self.record_paths(stem):
            write_bytes_atomic(p, data)
        out = {"unit": unit, "task": task, "k": k, "stem": stem, "records": [str(p) for p in self.record_paths(stem)],
               "record_sha256": hashlib.sha256(data).hexdigest(), "created_at": rec["created_at"],
               "inline_score": record_trace.get("inline_score")}
        self.written[unit] = out
        return out

    def write_cache_env(self) -> Path:
        p = self.cache_dir / "CACHE_ENV.json"
        write_json_atomic(p, cache_env_record(model=self.model, bucket=self.bucket, skills_dir=self.skills_dir,
                                              tag=self.tag))
        return p

    def kept_sample_problems(self, unit: str, question: str, *, score) -> list:
        """A written sample against its STATE `score` (resume and verify()): both record copies
        present and byte-identical, and their inline_score is that score (null for GRADER_FAILED)."""
        _task, k = sample_index(unit)
        blobs = [p.read_bytes() if p.is_file() else None
                 for p in self.record_paths(record_stem(question, self.model, k))]
        if any(b is None for b in blobs):
            return [f"{unit}: record missing"]
        if blobs[0] != blobs[1]:
            return [f"{unit}: the two record copies differ"]
        try:
            inline = (json.loads(blobs[0]).get("trace") or {}).get("inline_score")
        except ValueError as e:
            return [f"{unit}: record unreadable ({e})"]
        if not score_agrees(score, inline):
            return [f"{unit}: record inline_score {inline!r} != STATE score {score!r}"]
        return []

    def assemble(self, units_results: dict, *, runtime_env: dict, solver_harness: dict, daytona: dict,
                 extra: dict | None = None) -> dict:
        dy = dict(daytona)
        dy.setdefault("store", str(self.store))
        dy.setdefault("cache_dir", str(self.cache_dir))
        dy.setdefault("tasks_dir", str(self.tasks_dir))
        dy.setdefault("bucket", self.bucket)
        return assemble_leg(tag=self.tag, key=self.key, epoch=self.epoch, model=self.model, budget=self.budget,
                            n_samples=self.n_samples, task_ids=self.task_ids, units_results=units_results,
                            skills_dir=self.skills_dir, runtime_env=runtime_env, solver_harness=solver_harness,
                            daytona=dy, extra=extra)

    def write_failed(self, problems: list, daytona: dict | None = None) -> Path:
        """RUN/<TAG>.FAILED.json {INCOMPLETE: true, tag, key, problems, daytona, written_utc}."""
        write_json_atomic(self.failed_path, {"INCOMPLETE": True, "tag": self.tag, "key": self.key, "epoch": self.epoch,
                                             "problems": list(problems), "daytona": dict(daytona or {}),
                                             "written_utc": L.now_utc()})
        return self.failed_path

    def write_leg(self, units_results: dict, runtime_env: dict, solver_harness: dict, daytona: dict,
                  problems: list, *, verify: bool = True, extra: dict | None = None) -> Path:
        """The leg JSON, written atomically ONLY when `problems` is empty and (verify=True) verify() finds nothing;
        otherwise RUN/<TAG>.FAILED.json. Refuses when RUN/<TAG>.json already exists (never overwrite a leg the loop
        read)."""
        if self.leg_path.exists():
            raise FileExistsError(f"{self.leg_path} exists: a leg is written once (move it aside to re-run)")
        probs = list(problems)
        leg = None
        if not probs:
            try:
                leg = self.assemble(units_results, runtime_env=runtime_env, solver_harness=solver_harness,
                                    daytona=daytona, extra=extra)
            except ValueError as e:
                probs.append(str(e))
        if probs:
            return self.write_failed(probs, daytona)
        if verify:
            probs = self.verify(leg)
            if probs:
                return self.write_failed(probs, daytona)
        write_json_atomic(self.leg_path, leg)
        return self.leg_path

    # -- verify ---------------------------------------------------------------------------------------------------
    def verify(self, leg: dict) -> list:
        """Why this leg must not be written (empty = it may): for every unit of task_ids x N, its STATE.json row is graded and its
        score (a number from 0 to 1, or GRADER_FAILED), the leg's per_task entry and both record copies'
        inline_score say the same (GRADER_FAILED <-> null), the copies byte-identical; and sample_pass /
        sample_total / pass3 / all3 / grader_failed re-derive from per_task (leg_counts). Reads only."""
        try:
            st = state_read(self.state_path)
        except (OSError, ValueError) as e:
            return [f"STATE.json unreadable: {e}"]
        if st is None:
            return [f"{self.state_path} missing"]
        rows = st.get("units") or {}
        per_task = leg.get("per_task") or {}
        P = []
        if list(leg.get("task_ids") or []) != self.task_ids or leg.get("n_samples") != self.n_samples:
            P.append(f"the leg's tasks x samples ({len(leg.get('task_ids') or [])} x {leg.get('n_samples')}) are not "
                     f"this layout's ({len(self.task_ids)} x {self.n_samples})")
        shape_ok = True
        for i, task in enumerate(self.task_ids):
            v = per_task.get(str(i))
            if not isinstance(v, list) or len(v) != self.n_samples:
                P.append(f"{task}: per_task[{i}] = {v!r} (want {self.n_samples} entries)")
                shape_ok = False
                continue
            q = self.questions.get(task)
            if q is None:
                P.append(f"{task}: no question to find its records by (LegLayout questions=)")
            for k in range(self.n_samples):
                unit = f"{task}#s{k + 1}"
                row = rows.get(unit) or {}
                s = row.get("score")
                if row.get("state") != "graded":
                    P.append(f"{unit}: STATE state {row.get('state')!r}, not graded")
                    continue
                if not (s == GRADER_FAILED or is_score(s)):
                    P.append(f"{unit}: STATE score {s!r} is not a number from 0 to 1 or {GRADER_FAILED!r}")
                    continue
                if not score_agrees(s, v[k]):
                    P.append(f"{unit}: per_task {v[k]!r} != STATE score {s!r}")
                if q is not None:
                    P += self.kept_sample_problems(unit, q, score=s)
        if shape_ok:
            c = leg_counts(per_task, self.task_ids)
            drift = {n: (leg.get(n), c[n]) for n in c if leg.get(n) != c[n]}
            if drift:
                P.append(f"counts do not re-derive from per_task {{name: (leg, derived)}}: {drift}")
        return P


# ───────────────────────────────────────────────────────────────────────────── skills (4.1, 4.3, 4.4)
def skill_rows(skills_dir) -> list:
    """Every regular file of the promoted set as an upload row {rel, vm_path /evo/skills/<rel>, source, sha256,
    bytes} (cc_solver_lib.upload_rows' shape), whatever its name. Refused: a symlink (the upload copies files and
    follows no link). An empty set gives no rows (the solver runs with no skills)."""
    root = Path(skills_dir)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"{root}: not a directory (or a symlink)")
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise ValueError(f"{p}: a symlink in the skill set")
        if p.is_file():
            rel = p.relative_to(root)
            rows.append({"rel": rel.as_posix(), "vm_path": f"{SKILLS_VM}/{rel.as_posix()}", "source": str(p),
                         "sha256": L.sha256_file(p), "bytes": p.stat().st_size})
    return rows


def skill_names(rows: list) -> list:
    """The top-level dirs that hold a SKILL.md: the skills Claude Code can find (a record, not a rule)."""
    return sorted({Path(r["rel"]).parts[0] for r in rows if Path(r["rel"]).parts[1:] == ("SKILL.md",)})


def check_skill_rows(rows: list, skills_dir=None) -> list:
    """Problems with a skill-row list (empty = fine): every vm_path under /evo/skills/, a sha256 each, no row twice;
    with `skills_dir`, the rows must be exactly skill_rows(skills_dir) by (rel, sha256) -- the promoted dir may not
    change under a running leg."""
    P = []
    seen = set()
    for r in rows:
        rel = str(r.get("rel") or "")
        if not rel or r.get("vm_path") != f"{SKILLS_VM}/{rel}":
            P.append(f"{rel}: vm_path {r.get('vm_path')!r} is not under {SKILLS_VM}/")
        if not re.fullmatch(r"[0-9a-f]{64}", str(r.get("sha256") or "")):
            P.append(f"{rel}: no sha256")
        if rel in seen:
            P.append(f"{rel}: listed twice")
        seen.add(rel)
    if skills_dir is not None:
        try:
            want = {(r["rel"], r["sha256"]) for r in skill_rows(skills_dir)}
        except ValueError as e:
            return P + [f"{skills_dir}: {e}"]
        got = {(r.get("rel"), r.get("sha256")) for r in rows}
        if got != want:
            P.append(f"rows differ from {Path(skills_dir).name} on disk: {sorted(got ^ want)[:3]}")
    return P


def expect_links_for(skill_leg: bool) -> dict | None:
    """The ONE link a skill leg's workspace carries (Box.expect_links); None on a base leg."""
    return {SKILLS_LINK: SKILLS_VM} if skill_leg else None


def extra_dirs_for(rows: list) -> set:
    """Dirs the in-container listing may show beyond the workspace ones: /evo/skills, /evo/work/.claude and every
    dir under /evo/skills the rows imply (Box.extra_dirs)."""
    if not rows:
        return set()
    out = {SKILLS_VM, os.path.dirname(SKILLS_LINK)}
    for r in rows:
        d = os.path.dirname(r["vm_path"])
        while d.startswith(SKILLS_VM + "/"):
            out.add(d)
            d = os.path.dirname(d)
    return out


def parse_frontmatter(text: str) -> tuple:
    """({key: value} of the top-level `key: value` lines of a leading --- block, body) ; ({}, text) without one."""
    if not text.startswith("---"):
        return {}, text
    lines = text.split("\n")
    if lines[0].strip() != "---":
        return {}, text
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return {}, text
    fm = {}
    for ln in lines[1:end]:
        m = re.match(r"^([A-Za-z0-9_-]+):\s*(.*)$", ln)
        if m:
            fm[m.group(1)] = m.group(2).strip().strip("'\"")
    return fm, "\n".join(lines[end + 1:])


def skills_delivery_text(label: str, tree12: str) -> str:
    """solver_harness.skills_delivery: the sentence evolver_template appends to the solver
    description. Carries the base-directory line the Skill tool reply begins with."""
    return (f"The solver carried `{label}` (sha256 {tree12}), installed read-only at "
            f"`{SKILLS_LINK}/<skill-name>/` (a symlink to `{SKILLS_VM}`, Claude Code's project skills directory for "
            f"its working directory `{L.WS_CONTAINER}`; the direct harness's own `.claude/skills` convention). At the "
            f"start of every session Claude Code listed the skills to the solver as `- <name>: <description>` lines "
            f"(Claude Code's own rules: name = the frontmatter `name` or the directory name, description = the "
            f"frontmatter `description` or the body's first line); the solver loads one "
            f"with its `Skill` tool, whose reply begins with the line `Base directory for this skill: "
            f"{SKILLS_LINK}/<name>` followed by the SKILL.md body; files beside SKILL.md are readable from that "
            f"directory by the solver's bash and are run through their interpreter (they are not marked executable).")


def skills_delivery_facts(skills_dir, *, init_skills_seen: dict | None = None, sessions: int = 0,
                          probe: str | None = None, cc_version: str = CC_VERSION) -> dict:
    SL = skill_library_mod()
    d = Path(skills_dir)
    return {"how": "claude_code_project_skills_symlink", "path": SKILLS_LINK, "target": SKILLS_VM, "readonly": True,
            "owner": "root:root 0644/0755, read as uid 1000 (evo)", "library": d.name,
            "library_sha256": SL.library_sha256(d), "tree_sha256": skills_tree_sha256(d),
            "names": sorted(p.parent.name for p in d.glob("*/SKILL.md")),
            "init_skills_seen": dict(init_skills_seen or {}), "sessions": int(sessions),
            "rules": "Claude Code's own (the loop adds none)", "cc_version": cc_version, "probe": probe}


# ───────────────────────────────────────────────────────────────────────────── the rate-limit wait
def rate_limit_wait(rejections: list, now: float, *, leg_deadline_ts: float, pending: int, conc: int,
                    median_rollout_s: float | None) -> dict:
    """What a leg does after sessions were REJECTED for a usage limit (admission closed, nothing in flight).
    `rejections` = [{"brief": the session's last rate_limit_info (cc_solver_lib.rl_brief) or None, ...}]. until = the
    LATEST reset they reported (cc_solver_lib.rejection_reset; none reported, e.g. a bare HTTP 429: now +
    PAUSE_NO_READING_S), never before `now`. The leg sleeps until until + PAUSE_MARGIN_S and admits again, with no call
    to read the limit first; a session rejected again closes admission again. It FAILS instead (`reason` set) when
    the reset is more than WAIT_MAX_S away, or when the pending units would not fit before the leg's deadline after it
    (ceil(pending / conc) x the median rollout so far, at least PAUSE_MIN_REMAINING_S).
    {"until_ts", "until_utc", "window", "wait_s", "remaining_estimate_s", "beyond_deadline", "reason"}."""
    resets = [L.rejection_reset((r or {}).get("brief")) for r in rejections or []]
    known = [(ts, w) for ts, w in resets if ts is not None]
    if known:
        until, window = max(known)
    else:
        until, window = float(now) + PAUSE_NO_READING_S, "no_reset_reported"
    until = max(float(until), float(now))
    wait = until + PAUSE_MARGIN_S - float(now)
    per = float(median_rollout_s) if median_rollout_s else 0.0
    rem = max(PAUSE_MIN_REMAINING_S, math.ceil(max(0, int(pending)) / max(1, int(conc))) * per)
    beyond = until + PAUSE_MARGIN_S > float(leg_deadline_ts) - rem
    reason = None
    if until - float(now) > WAIT_MAX_S:
        reason = (f"usage limit: the rejected session's limit ({window}) resets at {utc_str(until)}, "
                  f"{(until - float(now)) / 3600:.1f} h away; a leg waits at most {WAIT_MAX_S // 3600} h")
    elif beyond:
        reason = (f"usage limit: the limit resets at {utc_str(until)}, beyond this leg's deadline "
                  f"{utc_str(leg_deadline_ts)} (with {int(rem)} s of rollouts still to run)")
    return {"until_ts": until, "until_utc": utc_str(until), "window": window, "wait_s": round(wait, 1),
            "remaining_estimate_s": rem, "beyond_deadline": beyond, "reason": reason}


# ───────────────────────────────────────────────────────────────────────────── the leg's pid file
def leg_pid_path(run_dir, tag: str) -> Path:
    return Path(run_dir) / LEG_PID_DIR / f"{tag}.pid"


def hold_leg_pid(run_dir, tag: str, **info):
    """<run>/.machine/legs/<tag>.pid = {pid, tag, started_utc, **info}, with an exclusive flock held on it for as long
    as the returned file stays open (the kernel drops it when the process ends, however it ends). A per-run record of
    the live leg in place of a machine-wide process search. RuntimeError when another live process holds it: that tag
    is running already."""
    path = leg_pid_path(run_dir, tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a+")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        held = leg_alive(run_dir, tag) or {}
        raise RuntimeError(f"leg {tag} is already running (pid {held.get('pid')}, {path})") from None
    fh.seek(0)
    fh.truncate()
    fh.write(json.dumps({"pid": os.getpid(), "tag": tag, "started_utc": L.now_utc(), **info}) + "\n")
    fh.flush()
    return fh


def leg_alive(run_dir, tag: str) -> dict | None:
    """The pid file's record when a live process holds its lock (the leg runs), else None (no file, or its process
    has ended). Never blocks."""
    path = leg_pid_path(run_dir, tag)
    try:
        fh = open(path)
    except OSError:
        return None
    with fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError:
            try:
                return json.loads(path.read_text() or "{}") or {"pid": None}
            except (OSError, ValueError):
                return {"pid": None}
        fcntl.flock(fh, fcntl.LOCK_UN)
        return None


# ───────────────────────────────────────────────────────────────────────────── STATE.json (5.4)
def unit_row() -> dict:
    """A unit's STATE row before anything happened to it (schema /2): score None = not graded yet, then the grader's
    number from 0 to 1 or GRADER_FAILED; snapshot_sha256 = the sha of runs/<ud>/container_out.tgz once collected."""
    return {"state": "queued", "cid": None, "result": False, "grade_job_id": None, "graded_utc": None, "score": None,
            "snapshot_sha256": None}


def state_new(*, tag: str, key: str, epoch: int, run_dir, bucket: str, skills_dir, stamp: str, units: list,
              machine: dict | None = None, lease: dict | None = None) -> dict:
    return {"schema": STATE_SCHEMA, "tag": tag, "key": key, "epoch": int(epoch), "run_dir": str(run_dir),
            "bucket": bucket, "skills_dir": str(skills_dir) if skills_dir else None, "started_utc": L.now_utc(),
            "stamp": stamp, "machine": dict(machine or {}), "lease": dict(lease or {}),
            "units": {u: unit_row() for u in units},
            "guard": {}, "pauses": [], "attempt": 1}


def state_write(path, st: dict) -> None:
    st["updated_utc"] = L.now_utc()
    write_json_atomic(path, st)


def state_read(path) -> dict | None:
    p = Path(path)
    if not p.is_file():
        return None
    st = json.loads(p.read_text())
    if st.get("schema") != STATE_SCHEMA:
        raise ValueError(f"{p}: schema {st.get('schema')!r} != {STATE_SCHEMA}")
    return st


def state_set_unit(st: dict, unit: str, **fields) -> None:
    u = st["units"].setdefault(unit, unit_row())
    if "state" in fields and fields["state"] not in STATE_UNIT_STATES:
        raise ValueError(f"{unit}: state {fields['state']!r}")
    u.update(fields)


def resume_units(st: dict, units: list) -> tuple:
    """(kept, requeue): a relaunch keeps the units the earlier attempt GRADED, grader_failed ones included (their
    records were written at grade time; the driver re-verifies them with LegLayout.kept_sample_problems against the
    STATE score) and re-queues every other unit in the leg's order."""
    kept, requeue = [], []
    for u in units:
        if (st.get("units") or {}).get(u, {}).get("state") == "graded":
            kept.append(u)
        else:
            requeue.append(u)
    return kept, requeue
