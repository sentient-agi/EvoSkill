"""Refuse to share a cache directory between runs whose ENVIRONMENT differs.

WHY THIS EXISTS. `RunCache`'s key is
``tree_hash | question_hash | question | sdk | model | effort``, where
``tree_hash`` mixes the live ``.claude/skills/`` content and — *only if the caller
passes it* — the agent's system prompt. Three things that change what a rollout
IS are therefore invisible to the key:

1. **The docker image.** Adding LibreOffice to the SSB v2 solver on 2026-08-02
   changed whether the solver can evaluate the formulas it is graded on, while
   every cache key stayed byte-identical.
2. **The turn budget.** Documented in the SSB v2 runner's own header: raising the
   cap from 40 to 45 needs a fresh cache dir, because a budget-40 rollout will
   otherwise replay and be reported as a budget-45 result.
3. **The system prompt, whenever the caller forgets to pass it.** Of the 9 SSB v2
   scripts that instantiate ``RunCache``, the one that hand-rolled its cache calls
   omitted ``system_prompt`` (the rest delegate to ``evaluate_agent_parallel``,
   which forwards it). ``_get_tree_hash`` then took its fallback branch and hashed
   two unrelated files
   (``solver/prompt.txt``, ``officeqa_agent/prompt.md``). Every SSB v2 cache dir
   in the repo therefore sits in the SAME bucket, ``bfb5bd90d1e0``, regardless of
   what the SSB v2 solver prompt actually said.

Each of those is silent in the same way: the records replay, the scores
aggregate, nothing errors, and the resulting number describes a mixture of two
conditions while claiming to describe one. The only thing that prevented it on
2026-08-02 was a fresh cache dir chosen for an unrelated reason.

So: on first use a run stamps a ``CACHE_ENV`` marker into its cache dir. Every later
run stamps its own environment and REFUSES if they disagree. A miss is cheap; a
false hit is a wrong result you cannot detect afterwards.

Deliberately generic — callers name their own axes — so each harness can adopt it
without this module having to know what matters for that benchmark.

    from src.cache.cache_env import assert_cache_env
    assert_cache_env(CACHE_DIR, {
        "solver_image": SOLVER_IMAGE,
        "solver_image_id": img_id,
        "turn_budget": BUDGET,
        "solver_prompt_sha256_12": sha12(get_solver_prompt()),
    })

There is NO override parameter. The 2026-07-24 factorial measured an offered
escape hatch being taken 72 times out of 72. To reuse a dir under a different
environment, move the dir aside or point at a new one — both of which leave a
trace, which is the point.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# NO `.json` EXTENSION, deliberately — the content IS json.
#
# The first version was `CACHE_ENV.json`, which broke `build_rollout_digests.py`
# within minutes: it sweeps a cache dir with `rglob("*.json")` and every file it
# finds is assumed to be a record, so the marker raised `KeyError: 'cache_key'`.
# Two stress-test scripts also count `rglob("*.json")` and would have silently
# reported one extra rollout. Patching each consumer would leave the next one to
# rediscover it; a name outside the record namespace fixes all of them, including
# the ones not written yet.
MARKER = "CACHE_ENV"


def sha12(text: str) -> str:
    """Short content hash, for putting a prompt's identity in a marker."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


class CacheEnvMismatch(RuntimeError):
    """The cache dir was built under a different environment than this run."""


def assert_cache_env(cache_dir: Path | str, env: dict, *, label: str = "") -> Path:
    """Stamp `env` into `cache_dir`, or raise if a different one is already there.

    Returns the marker path. Creates `cache_dir` if needed.

    Comparison is on the full dict: an axis added later counts as a difference,
    which is the safe direction — a run that has started tracking something new
    should not silently reuse records from before it was tracked.
    """
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    marker = d / MARKER
    # Sort keys so a reordered literal is not read as a changed environment.
    mine = json.loads(json.dumps(env, sort_keys=True, default=str))

    if not marker.exists():
        # An EXISTING dir with records but no marker predates this guard. Stamp
        # it rather than refusing: the records were made under whatever the
        # caller is running now in the common case (a resumed run), and refusing
        # would make the guard's introduction a breaking change for every
        # in-flight cache. The stamp means the NEXT run is protected.
        payload = dict(mine)
        payload["_note"] = (
            "Environment axes NOT in the RunCache key. A run whose axes differ "
            "from these must not reuse this directory. Stamped on first use; if "
            "this dir already held records at that moment, they are assumed to "
            "match and only later runs are checked."
        )
        if label:
            payload["_label"] = label
        payload["_preexisting_records"] = sum(
            1 for _ in d.glob("*/*.json"))
        marker.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        return marker

    have = json.loads(marker.read_text())
    theirs = {k: v for k, v in have.items() if not k.startswith("_")}
    if theirs == mine:
        return marker

    diffs = []
    for k in sorted(set(theirs) | set(mine)):
        a, b = theirs.get(k, "<absent>"), mine.get(k, "<absent>")
        if a != b:
            diffs.append(f"    {k}: cache has {a!r}, this run has {b!r}")
    raise CacheEnvMismatch(
        f"cache dir {d} was built under a DIFFERENT environment:\n"
        + "\n".join(diffs)
        + "\n\n  These axes are NOT part of the RunCache key, so reusing this "
        "directory\n  would replay records from the other environment and report "
        "them as this\n  run's results — silently, with no error and no missing "
        "data.\n\n  Point EVOSKILL_CACHE_DIR at a fresh path, or move this one "
        f"aside.\n  (marker: {marker})"
    )
