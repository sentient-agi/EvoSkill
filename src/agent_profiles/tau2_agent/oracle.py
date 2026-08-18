"""Oracle spec for tau2 — evaluator source + full golden task definitions.

Mirrors SSB v2's and SWE-rebench's convention (src/loop/oracle.py): the
evolver — and ONLY the evolver — gets `./oracle/` in its workspace with
the EXACT scoring code the harness runs plus golden task records. tau2
traces already embed each episode's graded RewardInfo (golden actions
with compare_args, reward_basis, per-component breakdown), so this adds
the two things traces cannot carry:

1. The evaluator SOURCE — how components combine into the final reward,
   argument-comparison semantics, action-set matching, DB-hash replay.
2. The full task definitions — complete `evaluation_criteria` and the
   `user_scenario` (persona + instructions) that drives the user
   simulator — for exactly the task ids the caller passed (TRAIN ONLY when a
   val firewall is enforced; see scripts/make_tau3_oracle.py).

Task meta is one JSON file per domain (`tasks/<domain>.json`, an object
keyed by task_id), NOT one file per task: telecom task_ids are 150+ char
composite strings ("[mms_issue]airplane_mode_on|...[PERSONA:Easy]") that
make terrible filenames, and telecom alone has 2,285 tasks — callers
pass the id subset their run actually uses so the oracle stays sized to
what the evolver's traces can ever reference.
"""

from __future__ import annotations

import json
from pathlib import Path

from .dataset import _default_tau2_root, load_raw_tasks

_EVALUATOR_SOURCES = (
    "evaluator.py",
    "evaluator_base.py",
    "evaluator_action.py",
    "evaluator_env.py",
    "evaluator_communicate.py",
    "evaluator_nl_assertions.py",
)

_SCORING_NOTE = """
Final reward = PRODUCT of the component rewards named in each task's
`evaluation_criteria.reward_basis` (this harness runs
`EvaluationType.ALL_WITH_NL_ASSERTIONS` — see `evaluator.py`'s own
docstring; `_tau2_worker.py` documents why not `ALL`). Components:

- ENV / "DB" (`evaluator_env.py`): the final database state must
  hash-match the state produced by replaying the task's golden write
  actions against a fresh environment. Binary.
- ENV_ASSERTION (`evaluator_env.py`, telecom tasks): each
  `evaluation_criteria.env_assertions` entry is an environment-tool
  assertion run against the final environment state; reward is the
  fraction/product per the evaluator's own combination — read the
  source for exact semantics.
- ACTION (`evaluator_action.py`): every golden action in
  `evaluation_criteria.actions` must appear in the transcript with
  matching arguments — ONLY the arguments listed in that action's
  `compare_args` are compared; others are free.
- COMMUNICATE (`evaluator_communicate.py`): each `communicate_info`
  item must have been conveyed to the user.
- NL_ASSERTIONS (`evaluator_nl_assertions.py`): LLM-judged
  natural-language assertions, included when in `reward_basis`.

Premature termination (too many errors / max steps / invalid agent
message) is reward 0.0 regardless of components. Since the reward is a
product of 0/1 component scores, any single failed component zeroes the
whole episode.

Task definitions live in `tasks/<domain>.json` — one JSON object per
domain, keyed by task_id, covering exactly the task ids this oracle was BUILT
with — see `Files` above for the count. A run that enforces a val firewall
passes TRAIN ONLY, and then no val or test task appears here at all. Each record is the COMPLETE golden spec the evaluator grades
against: `evaluation_criteria` (actions with compare_args,
communicate_info, nl_assertions, reward_basis), the `user_scenario`
(persona + instructions driving the user simulator), and
`initial_state`. The solver NEVER sees any of this — visible to YOU
only.

Trace files already embed each episode's graded RewardInfo. Use the
sources together: the trace for WHAT failed, `tasks/<domain>.json` for
the full golden spec, the evaluator source for exactly HOW the
comparison works.

KNOWLEDGE-RETRIEVAL DOMAINS (`banking_knowledge`): each task record also
carries `required_documents` — the exact set of knowledge-base document
ids that task needed, out of a corpus of ~698. This is the most directly
diagnosable signal in the whole oracle, and it is specific to this
domain:

  * Compare `required_documents` against the documents the episode
    ACTUALLY opened (the agent's `KB_search_*` / `grep` / document-read
    tool calls in the trace). A task that failed while never opening a
    required document failed at RETRIEVAL, not at reasoning — a
    different root cause with a different remedy.
  * The reverse is also informative: an episode that opened every
    required document and still failed did not have a search problem, so
    a rule about searching harder would not have helped it.
  * Counting how many required documents were opened before the agent
    committed to an answer separates "stopped searching too early" from
    "searched wrongly". Those need different rules.

Do NOT put document ids, document contents, or `required_documents`
themselves into a skill — that is training-data leakage of exactly the
kind the hard rules forbid. What generalises is the STRATEGY (how to
search, when to stop, what a sufficient answer needs), never the answer
key.
""".strip()

_DIFF_RECIPE = """
Given a failing trace whose Question is {"domain": "%(domain)s", "task_id": "%(tid)s"}:

```
python3 - <<'EOF'
import json
tasks = json.load(open("oracle/tasks/%(domain)s.json"))
t = tasks["%(tid)s"]
ec = t["evaluation_criteria"]
print("reward_basis:", ec.get("reward_basis"))
print("golden actions:", json.dumps(ec.get("actions"), indent=1))
print("communicate_info:", ec.get("communicate_info"))
print("nl_assertions:", ec.get("nl_assertions"))
print("user_scenario:", json.dumps(t.get("user_scenario"), indent=1)[:2000])
EOF
```

Then compare against the trace's own `action_checks` / `reward_breakdown`
header to see which specific golden item the episode missed.
""".strip()


def write_all_task_meta(
    tasks_dir: Path,
    task_ids: dict[str, set[str]],
    tau2_root: str | Path | None = None,
) -> int:
    """Write tasks/<domain>.json for each domain in `task_ids`.

    task_ids: {domain: set of task_ids to include} — pass the union of a
    run's train/val pool ids. Raises if any requested id is missing from
    the domain's raw task set (a silent drop here would starve the
    evolver of exactly the task it is diagnosing).
    """
    n = 0
    for domain, ids in sorted(task_ids.items()):
        raw = load_raw_tasks(domain, tau2_root)
        missing = set(ids) - raw.keys()
        if missing:
            raise KeyError(f"tau2 oracle: {len(missing)} unknown task_id(s) for {domain}: {sorted(missing)[:3]}...")
        subset = {tid: raw[tid] for tid in sorted(ids)}
        (Path(tasks_dir) / f"{domain}.json").write_text(json.dumps(subset, indent=1))
        n += len(subset)
    return n


def build_oracle_spec(task_ids: dict[str, set[str]], tau2_root: str | Path | None = None):
    """OracleSpec giving the tau2 evolver the evaluator source + golden tasks."""
    from src.loop.oracle import OracleSpec

    root = Path(tau2_root) if tau2_root else _default_tau2_root()
    ev_dir = root / "src" / "tau2" / "evaluator"
    scoring_code = [ev_dir / name for name in _EVALUATOR_SOURCES]
    for p in scoring_code:
        if not p.exists():
            raise FileNotFoundError(f"tau2 evaluator source missing: {p}")
    # RENDER THE RECIPE AGAINST THIS RUN'S OWN IDS. It used to hardcode
    # {"domain": "retail", "task_id": "84"} and `oracle/tasks/retail.json`. On a
    # banking_knowledge run that file does not exist, so an evolver following the
    # recipe — which the prompt tells it to copy pinned paths from VERBATIM rather
    # than construct — hit FileNotFoundError on its first diagnostic step.
    ex_domain = sorted(task_ids)[0]
    ex_ids = sorted(task_ids[ex_domain])
    if not ex_ids:
        raise ValueError(f"tau2 oracle: no task ids for domain {ex_domain!r}")
    recipe = _DIFF_RECIPE % {"domain": ex_domain, "tid": ex_ids[0]}
    return OracleSpec(
        scoring_note=_SCORING_NOTE,
        scoring_code=scoring_code,
        diff_recipe=recipe,
        write_task_meta=lambda tasks_dir: write_all_task_meta(tasks_dir, task_ids, tau2_root),
    )
