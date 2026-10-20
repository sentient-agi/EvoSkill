"""Golden tau2 conversations for tau3 banking_knowledge, recorded the way the rollout harness records one.

The reference a tau3 grader test grades. For one task it builds a tau2 SimulationRun from the task's own expected
actions and writes it where the harness writes the rollout's conversation, <work>/tau2/simulation.json. Reused by
tests/test_tau3_grader.py (the user's grader), the bundle builder's tests
and the verifier, so the reference is built one way everywhere.

IMPORT: standard library only (the repo's .venv has no tau2), so a test module can import this file for its paths and
rows. Every function that needs tau2 imports it inside and must run in the tau3-bench venv (TAU3_PY) with child_env():
PYTHON_DOTENV_DISABLED=1 (tau2 calls load_dotenv() at import), LITELLM_LOCAL_MODEL_COST_MAP=True, TAU2_DATA_DIR set to
the data root, and nothing else of the caller's environment, so no key can reach tau2. Nothing here calls a model: the
domain tools are local code, and no user simulator, agent LLM or judge is ever built.

THE GOLDEN CONVERSATION (golden_run), one message per step as tau2's Orchestrator records a half-duplex run:
  - tau2's greeting "Hi! How can I help you today?" (tau2/orchestrator/orchestrator.py:47-49), then a customer line;
  - each expected action of the task as ONE ToolCall in its own message (AssistantMessage for the agent's,
    UserMessage for the customer's; Claude Code runs a tool without readOnlyHint one at a time), with a toolu_-style id
    for the agent (Claude Code's ids) and a call_-style id for the customer (the gpt-5-nano simulator's), followed by
    the ToolMessage the environment returns for it. The environment is built as the rollout builds it
    (build_environment with _build_env_kwargs(TextRunConfig(domain, retrieval_config=variant), task), tau2/runner/
    build.py:337-353 and 356-425) and the task's initial state is set as Orchestrator.initialize sets it;
  - a text hand-over whenever the turn passes between agent and customer, so the list is a valid half-duplex history
    (Orchestrator.validate_message_history checks it);
  - a closing agent line and the customer's "###STOP###": termination user_stop.
  Timestamps, turn_idx and ids are fixed, so two builds of one task write the same bytes.

VARIANTS: stop_run (the greeting, then "###STOP###": a do-nothing agent), ended(run, reason) (the same messages with
another termination reason), tampered(run, task) (the first ToolMessage of a state-changing call carries other content;
None when the task has no such call), for_task(run, task_id) (the run claims another task).

CLI, under TAU3_PY (sets the three variables itself, TAU2_DATA_DIR from the task file):
  tau3_golden.py bundle --task-file F --kind golden|stop|agent_error|max_steps|tampered --dest D [--variant V]
writes D/tau2/simulation.json and prints one JSON line {"task_id", "kind", "path", "sha256"}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TAU3 = REPO / "third_party" / "tau3-bench"
TAU3_PY = TAU3 / ".venv" / "bin" / "python"               # CPython 3.12.13, tau2 1.0.1 (tau3-bench 363133a)
DATA_ROOT_ENV = "EVOSKILL_TAU3_DATA_ROOT"                   # another data root (a copy of <tau3-bench>/data)
DEFAULT_DATA_ROOT = TAU3 / "data"
DOMAIN = "banking_knowledge"
VARIANT = "bm25_grep"
BUNDLE = Path("tau2") / "simulation.json"
GREETING = "Hi! How can I help you today?"                  # tau2's DEFAULT_FIRST_AGENT_MESSAGE
STOP = "###STOP###"                                         # tau2's user stop token (user_simulator_base.py:51)
OPENING = "Hi, I need some help with my bank account."
TO_USER = "Please do this step on your side and tell me when it is done."
TO_AGENT = "Done."
CLOSING = "Everything is taken care of. Is there anything else I can help you with?"
KINDS = ("golden", "stop", "agent_error", "max_steps", "tampered")
T0 = datetime(2026, 9, 30, 0, 0, 0)
_B62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


# ───────────────────────────────────────────────────────────────────────────── standard library only
def tool_path(names=("srt", "rg", "node", "bwrap", "socat")) -> str:
    """A PATH for a child that runs tau2's KB shell: the directories this process's PATH finds `names` in (alltools'
    shell sandbox needs srt and rg; Linux also bwrap and socat), then /usr/local/bin, /usr/bin and /bin."""
    dirs: list[str] = []
    for n in names:
        w = shutil.which(n)
        if w and str(Path(w).parent) not in dirs:
            dirs.append(str(Path(w).parent))
    for d in ("/usr/local/bin", "/usr/bin", "/bin"):
        if d not in dirs:
            dirs.append(d)
    return os.pathsep.join(dirs)


def data_root() -> Path:
    """The data root the tests use: $EVOSKILL_TAU3_DATA_ROOT, else the checkout's <tau3-bench>/data."""
    return Path(os.environ.get(DATA_ROOT_ENV) or DEFAULT_DATA_ROOT)


def root_of(task_file) -> Path:
    """<root>/tau2/domains/<domain>/tasks/<file> -> <root> (the grader's rule)."""
    return Path(task_file).parents[4]


def task_files(root=None, domain: str = DOMAIN) -> list[Path]:
    return sorted((Path(root or data_root()) / "tau2" / "domains" / domain / "tasks").glob("task_*.json"))


def needs_judge(task_file) -> bool:
    """The task's reward basis holds NL_ASSERTION: tau2 grades it with an LLM judge, which no test here may call."""
    crit = json.loads(Path(task_file).read_text(encoding="utf-8")).get("evaluation_criteria") or {}
    return "NL_ASSERTION" in (crit.get("reward_basis") or [])


def graded_task_files(root=None, domain: str = DOMAIN) -> list[Path]:
    """Every task file the grader can score offline (banking: all but task_102)."""
    return [p for p in task_files(root, domain) if not needs_judge(p)]


def plugin_row(task_file, *, split: str = "train", domain: str = DOMAIN, variant: str = VARIANT,
               instruction: str = "") -> dict:
    """The tasks.json row a tau3 plug-in lists for one task (the grader reads answer_files[0] and grading)."""
    p = Path(task_file)
    return {"id": f"{domain}/{p.stem}", "category": domain, "instruction": instruction, "inputs": [],
            "answer_files": [str(p)], "grading": {"domain": domain, "retrieval_variant": variant}, "split": split}


def child_env(root=None, **extra) -> dict:
    """The whole environment of a process that imports tau2: nothing of the caller's is inherited."""
    env = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp"), "LC_ALL": "C.UTF-8",
           "PYTHON_DOTENV_DISABLED": "1", "LITELLM_LOCAL_MODEL_COST_MAP": "True", "PYTHONDONTWRITEBYTECODE": "1"}
    if root is not None:
        env["TAU2_DATA_DIR"] = str(root)
    env.update({k: str(v) for k, v in extra.items()})
    return env


def call_id(task_id: str, i: int, requestor: str) -> str:
    """A fixed id per (task, action): toolu_01 + 22 base62 characters for the agent, call_ + 24 for the customer."""
    n = int(hashlib.sha256(f"{task_id}:{i}".encode()).hexdigest(), 16)
    chars = []
    while len(chars) < 24:
        n, r = divmod(n, 62)
        chars.append(_B62[r])
    s = "".join(chars)
    return "toolu_01" + s[:22] if requestor == "assistant" else "call_" + s


def stamp(i: int) -> str:
    return (T0 + timedelta(seconds=i)).isoformat(timespec="microseconds")


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ───────────────────────────────────────────────────────────────────────────── tau2 (tau3 venv only)
def load_task(task_file):
    """The tau2 Task, read as the banking loader reads it (Task.model_validate on the file's JSON)."""
    from tau2.data_model.tasks import Task
    return Task.model_validate_json(Path(task_file).read_text(encoding="utf-8"))


def fresh_env(task, *, domain: str = DOMAIN, variant: str = VARIANT):
    """The environment the rollout's Orchestrator talks to, in the task's initial state."""
    from tau2.data_model.simulation import TextRunConfig
    from tau2.runner.build import _build_env_kwargs, build_environment
    ist = task.initial_state
    if ist is not None and ist.message_history:
        raise ValueError(f"{task.id}: a task with a message history is not built here (no banking task has one)")
    env = build_environment(domain, solo_mode=False,
                            env_kwargs=_build_env_kwargs(TextRunConfig(domain=domain, retrieval_config=variant), task))
    env.set_state(initialization_data=ist.initialization_data if ist else None,
                  initialization_actions=ist.initialization_actions if ist else None, message_history=[])
    return env


def _run(task_id: str, messages: list, termination: str):
    from tau2.data_model.simulation import SimulationRun, TerminationReason
    from tau2.orchestrator.orchestrator import Orchestrator
    Orchestrator.validate_message_history(messages)
    for i, m in enumerate(messages):
        m.timestamp = stamp(i)
        m.turn_idx = i
    return SimulationRun(id=f"golden-{task_id}", task_id=task_id, timestamp=stamp(0), start_time=stamp(0),
                         end_time=stamp(len(messages)), duration=float(len(messages)),
                         termination_reason=TerminationReason(termination), messages=messages,
                         mode="half_duplex")


def golden_messages(task, env) -> list:
    """The golden conversation's messages; env is fresh_env(task) and is advanced by the task's expected actions."""
    from tau2.data_model.message import AssistantMessage, ToolCall, UserMessage

    def says(role: str, text: str):
        return (AssistantMessage if role == "assistant" else UserMessage)(role=role, content=text)

    msgs = [says("assistant", GREETING), says("user", OPENING)]
    turn = "assistant"
    for i, a in enumerate(task.evaluation_criteria.actions or []):
        if a.requestor != turn:
            msgs.append(says(turn, TO_USER if turn == "assistant" else TO_AGENT))
            turn = a.requestor
        tc = ToolCall(id=call_id(task.id, i, a.requestor), name=a.name, arguments=dict(a.arguments or {}),
                      requestor=a.requestor)
        msgs.append((AssistantMessage if a.requestor == "assistant" else UserMessage)(
            role=a.requestor, content=None, tool_calls=[tc]))
        msgs.append(env.get_response(tc))
    if turn == "user":
        msgs.append(says("user", TO_AGENT))
    msgs += [says("assistant", CLOSING), says("user", STOP)]
    return msgs


def golden_run(task, *, domain: str = DOMAIN, variant: str = VARIANT):
    """The golden conversation as tau2's SimulationRun, ended by the customer (user_stop)."""
    return _run(task.id, golden_messages(task, fresh_env(task, domain=domain, variant=variant)), "user_stop")


def stop_run(task):
    """A do-nothing agent: the greeting, then the customer's ###STOP###."""
    from tau2.data_model.message import AssistantMessage, UserMessage
    return _run(task.id, [AssistantMessage(role="assistant", content=GREETING),
                          UserMessage(role="user", content=STOP)], "user_stop")


def ended(run, reason: str):
    """The same messages with another termination reason (agent_error, max_steps, ...)."""
    from tau2.data_model.simulation import TerminationReason
    return run.model_copy(deep=True, update={"termination_reason": TerminationReason(reason)})


def for_task(run, task_id: str):
    """The same conversation claiming to be for task_id."""
    return run.model_copy(deep=True, update={"task_id": task_id})


def tampered(run, task, *, domain: str = DOMAIN, variant: str = VARIANT):
    """The run with the first state-changing call's ToolMessage content replaced (strict replay must refuse it);
    None when no recorded call changes state. 'State-changing' is tau2's own test, Environment._is_mutating_tool,
    the one replay applies (tau2/environment/environment.py:388)."""
    from tau2.data_model.message import ToolMessage
    env = fresh_env(task, domain=domain, variant=variant)
    out = run.model_copy(deep=True)
    calls = {tc.id: tc.name for m in out.messages for tc in (getattr(m, "tool_calls", None) or [])}
    for m in out.messages:
        if isinstance(m, ToolMessage) and env._is_mutating_tool(calls[m.id]):
            m.content = json.dumps({"tampered": m.content})
            return out
    return None


def write_bundle(run, work) -> Path:
    """<work>/tau2/simulation.json: SimulationRun.model_dump(mode='json'), as the harness writes it."""
    p = Path(work) / BUNDLE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(run.model_dump(mode="json"), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def build(task_file, kind: str, *, variant: str = VARIANT):
    """The run of one KIND for one task file (None: a tampered run of a task with no state-changing call)."""
    task = load_task(task_file)
    if kind == "stop":
        return stop_run(task)
    gold = golden_run(task, variant=variant)
    if kind == "golden":
        return gold
    if kind in ("agent_error", "max_steps"):
        return ended(gold, kind)
    if kind == "tampered":
        return tampered(gold, task, variant=variant)
    raise ValueError(f"unknown kind {kind!r}; one of {KINDS}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("bundle")
    p.add_argument("--task-file", required=True)
    p.add_argument("--kind", required=True, choices=KINDS)
    p.add_argument("--dest", required=True)
    p.add_argument("--variant", default=VARIANT)
    args = ap.parse_args(argv)
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    task_file = Path(args.task_file).absolute()
    os.environ["TAU2_DATA_DIR"] = str(root_of(task_file))
    run = build(task_file, args.kind, variant=args.variant)
    if run is None:
        print(json.dumps({"task_file": args.task_file, "kind": args.kind, "path": None,
                          "why": "no recorded call changes state"}))
        return 1
    path = write_bundle(run, args.dest)
    print(json.dumps({"task_id": run.task_id, "kind": args.kind, "path": str(path), "sha256": sha256_file(path)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
