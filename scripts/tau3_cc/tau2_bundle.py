"""The tau2 bundle of one tau3 rollout: what the rollout harness writes into the solver's working directory after the
conversation, dest/tau2/.

tau2's own Orchestrator runs the conversation (build_text_orchestrator, then run_conversation below). Then
write_bundle(orch, sim, dest, meta) writes three files:

  - simulation.json: tau2's SimulationRun, SimulationRun.model_dump(mode="json") with the raw_data key removed from
    every message. raw_data is the model provider's raw reply (AssistantMessage and UserMessage only,
    tau2/data_model/message.py:218); nothing in tau2's grading reads it. The key is removed from the message dicts
    only, never searched for deeper, so a tool argument that happens to be named raw_data is left alone. Every other
    field is kept: tool-call ids, names, arguments and requestor, ToolMessage content and error flag, turn_idx,
    timestamps, the termination reason, the seed. There is no reward_info, because the harness never grades. This is
    the only file the grader reads (grade(task, snapshot), examples/tau3_banking/grader.py).
  - run.json: how the conversation was run. From tau2 itself: the tau2 version (get_tau2_version()) and the pinned
    commit 363133a, the data root tau2 actually read (tau2.utils.utils.DATA_DIR), and from the orchestrator the
    domain, the task id, the simulation id, the seed, max_steps, max_errors, the wall limit (orch.timeout, in
    seconds, tau2's own timeout), the steps taken and the tool errors counted. From the SimulationRun: the duration and
    the end reason (termination_reason). From meta, the facts only the harness knows: task_file (its sha256 is
    computed here), retrieval_variant, retrieval_kwargs (default {}) and end_detail (default None; what
    run_conversation returned). Any other meta key is written as given under "harness" (the harness's name and sha,
    the user simulator's model, Claude Code's session id and exit code, ...).
  - final_db.json: the environment's databases when the conversation ended, {"agent": {"hash", "db"},
    "user": {"hash", "db"}}. hash is Environment.get_db_hash() / get_user_db_hash() (tau2/environment/
    environment.py:275-291), db is that toolkit's db.model_dump(mode="json"). This is evidence for the evolver (did
    the refund happen?); the grader never reads it. In banking_knowledge the agent's and the user's toolkits share one
    database object (domains/banking_knowledge/environment.py:71-83), so the two dumps are the same.

write_bundle creates dest/tau2 and refuses (FileExistsError) when it already exists, a symlink included, so it never
writes through anything planted there. The harness writes into a directory of its own on the host and ships the files
into the container as the container user. bundle() returns the same three
files as bytes, for a harness that builds the tgz in memory.

ENDING A CONVERSATION FROM INSIDE THE AGENT (run_conversation, HarnessEnd). tau2's Orchestrator.run() has no except
clause: an exception raised in the agent's generate_next_message propagates, and run()'s `finally` only calls
_cleanup() (tau2/orchestrator/orchestrator.py:260-291). So the harness's agent raises HarnessEnd, and
run_conversation catches it, sets orch.termination_reason (agent_error by default) and orch.done, and calls
orch._finalize() itself. _finalize accepts that state, because the orchestrator's next recipient is the agent
(to_role == AGENT), and it builds the SimulationRun from the trajectory so far (orchestrator.py:752-821). _finalize is
a PRIVATE method: this relies on tau2 at the pinned commit 363133a. _cleanup and then _finalize both call the agent's
and the user's stop(), so a harness agent's stop() must tolerate being called twice. Any other exception propagates
unchanged. The reasons a harness end may give: agent_error (Claude ended a turn
with empty text, hit its turn limit or called EndConversation), timeout (the wall ran out while the agent was
answering), infrastructure_error (an API, quota or process failure; the loop does not grade it and re-runs the unit).
tau2 scores all three 0 (tau2/evaluator/evaluator.py:118-128).

IMPORT: standard library at import time. tau2 is imported inside the functions, because a process must set
TAU2_DATA_DIR, PYTHON_DOTENV_DISABLED=1 and LITELLM_LOCAL_MODEL_COST_MAP=True before tau2's first import (tau2 calls
load_dotenv() and reads the data root once, at import, tau2/utils/utils.py:13-28), and importing this module must never
be that first import. Runs only under the tau3-bench venv (tau2 1.0.1, commit 363133a). tests/test_tau2_bundle.py
checks it on every banking task but task_102 with the scripted golden pair of tests/fixtures_tau2_scripted.py.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

TAU2_COMMIT = "363133a"          # the tau3-bench commit the rollouts and the grader are pinned to
BUNDLE_DIR = "tau2"              # <dest>/tau2/, in the solver's working directory
FILES = ("simulation.json", "run.json", "final_db.json")
HARNESS_ENDS = ("agent_error", "timeout", "infrastructure_error")    # TerminationReason values a harness end may give
META_KEYS = ("task_file", "retrieval_variant", "retrieval_kwargs", "end_detail")   # meta keys read here; the rest -> "harness"


class HarnessEnd(Exception):
    """Raised by the harness's agent inside generate_next_message to end the conversation; run_conversation turns it
    into tau2's termination `reason` and a finished SimulationRun. detail says what happened, for run.json."""

    def __init__(self, detail: str, reason: str = "agent_error"):
        if reason not in HARNESS_ENDS:
            raise ValueError(f"a harness end is one of {HARNESS_ENDS}, not {reason!r}")
        super().__init__(detail)
        self.detail = detail
        self.reason = reason


def run_conversation(orch):
    """orch.run(), or, when the agent raises HarnessEnd, the SimulationRun of the conversation so far, ended with the
    exception's reason (a private tau2 method, _finalize, at the pinned commit). Returns (sim, end_detail): end_detail
    is None when tau2 itself ended the conversation, else the HarnessEnd's detail."""
    from tau2.data_model.simulation import TerminationReason

    try:
        return orch.run(), None
    except HarnessEnd as e:
        orch.termination_reason = TerminationReason(e.reason)
        orch.done = True
        return orch._finalize(), e.detail


def _json(obj) -> bytes:
    return (json.dumps(obj, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def _simulation(sim) -> dict:
    d = sim.model_dump(mode="json")
    for m in d.get("messages") or []:
        m.pop("raw_data", None)
    return d


def _run(orch, sim, meta: dict) -> dict:
    from tau2.utils.utils import DATA_DIR, get_tau2_version

    task_file = Path(meta["task_file"])
    return {
        "tau2": {"version": get_tau2_version(), "commit": TAU2_COMMIT},
        "data_dir": str(DATA_DIR),
        "domain": orch.domain,
        "retrieval_variant": meta["retrieval_variant"],
        "retrieval_kwargs": dict(meta.get("retrieval_kwargs") or {}),
        "task_id": orch.task.id,
        "task_file": str(task_file),
        "task_file_sha256": hashlib.sha256(task_file.read_bytes()).hexdigest(),
        "simulation_id": sim.id,
        "seed": orch.seed,
        "max_steps": orch.max_steps,
        "max_errors": orch.max_errors,
        "wall_s": orch.timeout,
        "duration_s": sim.duration,
        "steps": orch.step_count,
        "tool_errors": orch.num_errors,
        "end": {"reason": sim.termination_reason.value, "detail": meta.get("end_detail")},
        "harness": {k: v for k, v in meta.items() if k not in META_KEYS},
    }


def _final_db(env) -> dict:
    def dump(toolkit):
        return None if toolkit is None or toolkit.db is None else toolkit.db.model_dump(mode="json")

    return {"agent": {"hash": env.get_db_hash(), "db": dump(env.tools)},
            "user": {"hash": env.get_user_db_hash(), "db": dump(env.user_tools)}}


def bundle(orch, sim, meta: dict) -> dict:
    """The three files of the bundle, {name: bytes}. sim must be the SimulationRun this orchestrator produced."""
    if sim.id != orch.simulation_id:
        raise ValueError(f"the SimulationRun {sim.id!r} is not the one orchestrator {orch.simulation_id!r} produced")
    return {"simulation.json": _json(_simulation(sim)), "run.json": _json(_run(orch, sim, meta)),
            "final_db.json": _json(_final_db(orch.environment))}


def write_bundle(orch, sim, dest, meta: dict) -> Path:
    """Write <dest>/tau2/{simulation.json, run.json, final_db.json} and return <dest>/tau2. dest is created when
    missing; <dest>/tau2 must not exist (FileExistsError)."""
    files = bundle(orch, sim, meta)
    out = Path(dest) / BUNDLE_DIR
    out.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir()
    for name in FILES:
        with open(out / name, "xb") as fh:
            fh.write(files[name])
    return out
