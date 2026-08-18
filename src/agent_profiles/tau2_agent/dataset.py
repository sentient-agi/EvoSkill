"""Load tau2-bench tasks (retail, airline, telecom, banking_knowledge) and
build EvoSkill's train/val/test split.

Reads the raw task JSON directly from the vendored tau2-bench clone
(`third_party/tau2-bench/data/tau2/domains/<domain>/tasks.json` +
`split_tasks.json`) rather than importing the `tau2` package -- this keeps
EvoSkill's own venv free of any tau2 dependency (see `solver.py`'s module
docstring for why). The task JSON schema matches tau2's own `Task` Pydantic
model (`id`, `description`, `user_scenario`, `ticket`, `initial_state`,
`evaluation_criteria`, `issues`, `required_documents`, `user_tools`) closely
enough for our purposes (split assembly + category bucketing + the
user-tier-gate classifier's input fields) that re-parsing with our own
lightweight dict access, rather than pulling in pydantic validation via
tau2 itself, is a deliberate and low-risk simplification.

Split, retail/airline/telecom (native `split_tasks.json` exists for all
three, confirmed sizes: retail train=74/test=40, airline train=30/test=20,
telecom train=74/test=40):
  - test = tau2's own native "test" split, untouched, so scores stay
    comparable to anything reported against tau2-bench's own test set.
  - train/val = tau2's native "train" pool evenly halved via a deterministic
    sha256-hash sort keyed by task_id, mirroring tb2_agent/dataset.py's
    `_sort_key` pattern (retail: 37/37, airline: 15/15, telecom: 37/37).
  Judgment call (flagged in the design doc, retail-specific but the same
  rationale extends to airline/telecom): the resulting train/val split departs
  slightly from this repo's usual ~40/40/rest convention, driven by the fixed
  native-train pool size -- reusing tau2's own test split (rather than
  re-deriving a from-scratch split that would touch it) was judged more
  valuable for external comparability.

Split, banking_knowledge (NO native `split_tasks.json` -- confirmed via
directory listing; only `db.json`, `tasks.json`, `tasks_voice.json`, plus
`documents/`, `prompts/`, `tasks/` subdirs exist): there is no "native test
split" to preserve for external comparability here (unlike retail/airline/
telecom), so this build synthesizes its own fixed, deterministic 30/30/37
split (train/val/test) across all 97 tasks, via the SAME sha256-hash sort
(`_sort_key`, seed=42) used to halve the other domains' native train pools --
just applied to the full 97-task population instead of a 74/30/74-task
native "train" subset. 30/30/37 was chosen (rather than e.g. an even 32/32/33)
because it keeps train/val equal in size (so the evolver's train-batch and
the val-gate see comparably-sized pools, matching this repo's usual
train==val-size convention elsewhere) while giving test a few extra tasks
as the fixed remainder -- this is a reasonable, but not the only defensible,
split of a small 97-task pool; there is no tau2-upstream precedent to match
against here, so any deterministic, seed-documented split is equally valid.

CRITICAL data-quality note (found during the 4-domain extension, NOT a
hypothesis): banking_knowledge's single-file `tasks.json` aggregate is STALE
relative to the authoritative per-task files tau2's own registered loader
(`banking_knowledge/environment.py:get_tasks()`) actually reads at episode
run-time -- a byte-level diff found 12/97 task IDs differ in content
(`required_documents`/evaluation-relevant fields) between `tasks.json` and
the corresponding `tasks/task_NNN.json`. `load_raw_tasks()` below therefore
enumerates `tasks/task_*.json` directly for banking_knowledge (mirroring
`environment.py`'s own `sorted(tasks_dir.glob("task_*.json"))`), NOT the
stale `tasks.json`, so EvoSkill's split-building/tier-gating machinery never
disagrees with what the real episode run actually evaluates against.

Category axis for `train_pools`: NOT reward_basis (112/114 retail tasks
share the identical ("DB","NL_ASSERTION") tuple -- verified, near-degenerate
within retail). Instead buckets by golden-action-count into
{"0-2_actions", "3-4_actions", "5+_actions"} as a task-complexity proxy.
This bucketing function (`_action_count_bucket`) is domain-agnostic and was
confirmed to work unmodified across all 4 domains (every domain's raw task
dict has an `evaluation_criteria.actions` list, possibly empty). Callers
that want a domain-based category instead (e.g. a multi-domain self-evolve
launcher spanning all 4 domains in one run) should simply re-tag the
`(question, ground_truth, category)` tuples this module returns with their
own category string -- this module's own category field is left as the
action-count bucket unconditionally, for backward compatibility with any
existing single-domain caller.

Domain scope (explicit design decision, recorded here because a prior
version of this build never wrote it down anywhere -- a later reader could
only infer "airline was considered and dropped" from the absence of an
`airline.json` tier-gate manifest, which is not an acceptable substitute for
a stated rationale):

  Originally only `retail` was integrated (see git history for the full
  original rationale on why retail was chosen first over airline/telecom/
  banking_knowledge). This module has since been extended to cover all 4
  domains `tau2` registers for text-mode use: `retail`, `airline`,
  `telecom`, `banking_knowledge`. `telecom-workflow` (a second registered
  domain name sharing the same task pool, differing only in which
  tech-support policy variant is used) and `mock` are still NOT covered --
  out of scope for this extension, not evaluated.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _default_tau2_root() -> Path:
    env = os.environ.get("TAU2_BENCH_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[3] / "third_party" / "tau2-bench"


def _tasks_path(tau2_root: Path, domain: str) -> Path:
    return tau2_root / "data" / "tau2" / "domains" / domain / "tasks.json"


def _split_path(tau2_root: Path, domain: str) -> Path:
    return tau2_root / "data" / "tau2" / "domains" / domain / "split_tasks.json"


def _banking_knowledge_tasks_dir(tau2_root: Path) -> Path:
    # Mirrors tau2's own `KNOWLEDGE_TASK_SET_PATH`
    # (`tau2/domains/banking_knowledge/utils.py`) exactly -- the directory
    # of individual `task_NNN.json` files, NOT the stale single-file
    # `tasks.json` aggregate (see module docstring's "CRITICAL data-quality
    # note").
    return tau2_root / "data" / "tau2" / "domains" / "banking_knowledge" / "tasks"


def load_raw_tasks(domain: str = "retail", tau2_root: str | Path | None = None) -> dict[str, dict]:
    """Return {task_id: raw_task_dict} for every task in `domain`."""
    root = Path(tau2_root) if tau2_root else _default_tau2_root()
    if domain == "banking_knowledge":
        # NOT `_tasks_path()` / the single-file `tasks.json` -- that
        # aggregate is stale for 12/97 tasks (confirmed byte-level diff, see
        # module docstring). Enumerate `tasks/task_*.json` directly instead,
        # mirroring `banking_knowledge/environment.py:get_tasks()`'s own
        # `sorted(tasks_dir.glob("task_*.json"))`, so this dict always
        # agrees with what the real episode run (via `get_tasks()` inside
        # `_tau2_worker.py`) actually evaluates against.
        tasks_dir = _banking_knowledge_tasks_dir(root)
        tasks = [json.loads(p.read_text()) for p in sorted(tasks_dir.glob("task_*.json"))]
        return {t["id"]: t for t in tasks}
    tasks = json.loads(_tasks_path(root, domain).read_text())
    return {t["id"]: t for t in tasks}


def _action_count_bucket(task: dict) -> str:
    n = len((task.get("evaluation_criteria") or {}).get("actions") or [])
    if n <= 2:
        return "0-2_actions"
    if n <= 4:
        return "3-4_actions"
    return "5+_actions"


def _sort_key(task_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()


def load_tau2(
    domain: str = "retail",
    tau2_root: str | Path | None = None,
    split_seed: int = 42,
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Load tau2-bench tasks and split into train/val/test.

    Returns (train, val, test) where each element is
    (question_json, "PASS", category).

    question_json is `{"domain": ..., "task_id": ...}` -- a pointer, not the
    full task payload (the full `Task` is resolved inside the worker
    subprocess, which has real tau2 `Task` parsing available).
    ground_truth is always the literal "PASS" (ALFWorld/TB2 convention --
    real correctness data lives in the task's own evaluation_criteria,
    resolved live by tau2's own evaluator, not in a `ground_truth` string).
    category is the action-count bucket (see module docstring).

    Works for all 4 integrated domains:
      - retail / airline / telecom: native `split_tasks.json` "train"/"test"
        keys exist for all three (confirmed sizes: retail 74/40, airline
        30/20, telecom 74/40); "train" is evenly halved (deterministic
        sha256-hash sort keyed by task_id, seed=`split_seed`) into
        train/val, "test" is tau2's own native test split, untouched.
      - banking_knowledge: NO native `split_tasks.json` exists (confirmed).
        This build synthesizes its own fixed, deterministic 30/30/37
        train/val/test split across all 97 tasks, via the SAME hash-sort
        mechanism applied to the full task-id population instead of a
        native "train" subset (see module docstring for the exact
        rationale).
    """
    root = Path(tau2_root) if tau2_root else _default_tau2_root()
    raw_tasks = load_raw_tasks(domain, root)

    def _make(task_id: str) -> tuple[str, str, str]:
        task = raw_tasks[task_id]
        question = json.dumps({"domain": domain, "task_id": task_id})
        return question, "PASS", _action_count_bucket(task)

    if domain == "banking_knowledge":
        # No native split_tasks.json (confirmed via directory listing) --
        # build our own fixed 30/30/37 split across all 97 tasks, same
        # deterministic sha256-hash-sort-by-task_id mechanism as the other
        # domains use to halve their native train pool (see module
        # docstring's banking_knowledge split section for the exact
        # rationale on why 30/30/37 was chosen).
        all_ids = sorted(raw_tasks.keys(), key=lambda tid: _sort_key(tid, split_seed))
        train_ids = all_ids[:30]
        val_ids = all_ids[30:60]
        test_ids = all_ids[60:]
    else:
        native_split = json.loads(_split_path(root, domain).read_text())
        native_train_ids = sorted(native_split["train"], key=lambda tid: _sort_key(tid, split_seed))
        test_ids = sorted(native_split["test"], key=lambda tid: _sort_key(tid, split_seed))

        half = len(native_train_ids) // 2
        train_ids = native_train_ids[:half]
        val_ids = native_train_ids[half:]

    train = [_make(tid) for tid in train_ids]
    val = [_make(tid) for tid in val_ids]
    test = [_make(tid) for tid in test_ids]

    return train, val, test


def load_tau2_compact(
    domain: str,
    n_train: int,
    n_val: int,
    tau2_root: str | Path | None = None,
    split_seed: int = 42,
) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    """Like `load_tau2`, but train/val are a smaller, caller-sized SUBSET of
    the same native/synthesized pool, taken from the front of the SAME
    deterministic sha256-hash-sorted order `load_tau2` itself uses (so a
    `load_tau2_compact(..., n_train=37, n_val=37)` call reproduces
    `load_tau2`'s own train/val exactly for retail/telecom).

    `test` is completely untouched -- identical to `load_tau2`'s own test
    split (tau2's native test for retail/airline/telecom, the same
    30/30/37-synthesized test slice for banking_knowledge) -- this function
    only shrinks how much of the TRAIN pool gets used for train+val, never
    touches or reduces what counts as held-out test data.

    Built for constructing a smaller, cross-domain-stratified combined
    train/val split (e.g. 40/40 total across all 4 domains, matching this
    repo's usual train==val-size convention) while preserving each domain's
    share of the full population -- see
    scripts/run_tau2_dsv4f_dsv4p_compact40.py for the stratified allocation
    this feeds into.
    """
    root = Path(tau2_root) if tau2_root else _default_tau2_root()
    raw_tasks = load_raw_tasks(domain, root)

    def _make(task_id: str) -> tuple[str, str, str]:
        task = raw_tasks[task_id]
        question = json.dumps({"domain": domain, "task_id": task_id})
        return question, "PASS", _action_count_bucket(task)

    if domain == "banking_knowledge":
        all_ids = sorted(raw_tasks.keys(), key=lambda tid: _sort_key(tid, split_seed))
        pool_ids = all_ids[:60]  # same 60-task train+val pool load_tau2 uses (30+30)
        test_ids = all_ids[60:]
    else:
        native_split = json.loads(_split_path(root, domain).read_text())
        pool_ids = sorted(native_split["train"], key=lambda tid: _sort_key(tid, split_seed))
        test_ids = sorted(native_split["test"], key=lambda tid: _sort_key(tid, split_seed))

    if n_train + n_val > len(pool_ids):
        raise ValueError(
            f"{domain}: requested n_train={n_train} + n_val={n_val} = {n_train + n_val} "
            f"exceeds the available {len(pool_ids)}-task train+val pool."
        )
    train_ids = pool_ids[:n_train]
    val_ids = pool_ids[n_train:n_train + n_val]

    train = [_make(tid) for tid in train_ids]
    val = [_make(tid) for tid in val_ids]
    test = [_make(tid) for tid in test_ids]

    return train, val, test
