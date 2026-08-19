#!/usr/bin/env python3
"""Layer 2 of the orchestrator — the FROZEN EXPERIMENT SPEC.

WHAT THIS IS FOR

An evolution run has ~30 knobs that decide what its numbers mean: which split, which seed,
how many samples, which solver, which user simulator, which wall, which gate. Today those
knobs live in a shell launcher's environment block, and the record of what a number meant
is "whatever the launcher happened to export that day". That is how #182/#183 happened
(two knobs outside the cache identity, so a config change silently replayed stale
episodes), and it is how a gate can be chosen after seeing the data (#121).

This module makes the config a first-class artifact:

    spec = ExperimentSpec(...)          # declare
    spec.validate()                     # structural + cross-field rules, each able to FAIL
    spec.freeze(run_dir)                # canonical JSON + sha256, written ONCE
    ...
    ExperimentSpec.assert_unmutated(run_dir)   # before every leg, and at report time

The freeze is the pre-registration. `assert_unmutated` is what makes it mean anything: a
spec you can quietly edit after seeing epoch 2 is not a pre-registration, it is a diary.

TWO RULES THIS ENCODES THAT COST REAL RUNS TO LEARN

1. The gate metric is declared BEFORE the comparison leg runs. Task-paired vs episode-level
   pairing changed the discordant count from 1-3 to 12-13 on identical rollouts; choosing
   between them after seeing either is a selection move (#121, handoff 0807 K9).

2. The cache tag is DERIVED from the config hash, never hand-typed. Two knobs
   (llm_args_agent, the seed clamp) are outside the cache identity, so any config change
   that is not reflected in the tag replays another config's episodes (#182/#183). A human
   remembering to bump a string is not a mechanism.

NOT IMPLEMENTED HERE: the natural-language front end. `from_natural_language` documents its
contract and raises. It is deliberately not a stub that returns a plausible object — a spec
that looks filled-in but was guessed is worse than no spec, because freeze() would stamp a
sha256 on a fiction.

USAGE
    python3 -m src.orchestrator.spec --selftest
    python3 -m src.orchestrator.spec --check outputs/tau3_dsv4fevo

EXIT CODES: 0 ok, 1 validation/mutation FAIL, 2 could not evaluate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass, field
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPEC_FILENAME = "SPEC.json"

# The exact-binomial floor: with fewer than 6 discordant pairs, no two-sided p can reach
# 0.05 (2 * 0.5**5 = 0.0625). A gate that cannot fire is not a gate (#154).
DISCORDANT_FLOOR = 6

VALID_PAIRING = ("episode_level", "task_paired")
VALID_BASES = ("DB", "ACTION", "ENV_ASSERTION", "COMMUNICATE", "NL_ASSERTION")


class SpecError(Exception):
    """Raised for any violation. Never caught inside this module."""


def split_digest(path) -> str:
    """The split's OWN attested digest, re-derived — not a second, competing hash.

    make_tau3_balanced_split.py writes the file, hashes `json.dumps(out, indent=1)` of
    everything EXCEPT the digest field (insertion order, NOT sort_keys), then rewrites the
    file with the digest appended. So a whole-file hash disagrees with the recorded value by
    construction, and quoting the whole-file hash next to the pre-registered one would
    invent a conflict where none exists. Reproduce the generator's canonicalization instead;
    a pre-registration digest nobody can recompute is decorative.

    Falls back to the whole-file hash for splits with no self-attested digest.
    """
    p = Path(path)
    d = json.loads(p.read_text())
    if isinstance(d, dict) and "self_sha256" in d:
        body = {k: v for k, v in d.items() if k != "self_sha256"}
        return hashlib.sha256(json.dumps(body, indent=1).encode()).hexdigest()
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ── the spec ─────────────────────────────────────────────────────────────────────────────
@dataclass
class GateSpec:
    """The pre-registered decision rule. Declared before the comparison leg exists."""
    pairing: str = "episode_level"
    tests: tuple = ("episode_paired_sign", "cluster_pass_count_sign")
    alpha: float = 0.05
    n_comparisons: int = 4                  # epochs; alpha is Bonferroni-divided by this
    require_all_tests: bool = True          # BOTH tests must clear, not either
    discordant_floor: int = DISCORDANT_FLOOR
    report_per_basis: bool = True           # #156: never ship the pooled number alone

    @property
    def alpha_corrected(self) -> float:
        return self.alpha / max(1, self.n_comparisons)


@dataclass
class HeadroomGate:
    """#180 + #206. A domain can be unusable for two opposite reasons, and one gate that
    only checks the ceiling misses the floor entirely."""
    min_train_pass_rate: float = 0.08       # below this the evolver has nothing to learn from
    max_train_task_paired: float = 0.90     # at/above this the ceiling sets the result
    min_exploration_rate: float = 0.0       # #206: floored instrumentation looks like a hard domain


# THE SSB v2 SOLVER ALLOW-LIST, mirroring scripts/ssbv2_leg.sh's `case "$MODEL" in`.
# Two dated snapshots, extended only for one an operator named explicitly.
SSBV2_SOLVER_SNAPSHOTS = ("-0731", "-0813")


@dataclass
class ExperimentSpec:
    # identity
    name: str
    benchmark: str                           # e.g. "tau3"
    domain: str                              # e.g. "banking_knowledge"
    variant: str = ""                        # retrieval variant, "" when the domain has none

    # data
    split_path: str = ""
    split_sha256: str = ""                   # of the split file itself; verified at freeze
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0
    test_sealed: bool = True                 # the test split is never run during evolution

    # instrument — every field here changes what a number means
    n_samples: int = 3
    solver_model: str = ""
    solver_provider_order: tuple = ()
    solver_allow_fallbacks: bool = False
    user_sim_model: str = ""
    judge_model: str = ""
    agent_clean_s: int = 1200
    agent_turns: int = 100
    episode_timeout_s: int = 5400
    # #249. WHICH HARNESS EXECUTES A SOLVER EPISODE. This is an INSTRUMENT field and is in
    # _INSTRUMENT below: a harness decides the tool schemas the model sees, the shape of the
    # agent loop, and how the prompt is assembled, so the same model on two harnesses does not
    # produce the same episode. Leaving it out of the cache identity would be #182/#183 exactly
    # — the class of defect where a config that changes an episode is invisible to the key, and
    # records from one instrument replay silently into a leg measured under another.
    # Default "direct" is what every tau3 episode to date has actually run on.
    solver_harness: str = "direct"

    # optimizer
    evolver_model: str = ""
    # #249. WHICH HARNESS RUNS THE EVOLVER. DELIBERATELY NOT IN _INSTRUMENT, for the same reason
    # evolver_model is not: an evolver is not present when a solver rollout runs, so it cannot
    # change what an episode does. Keying on it is safe and never free — it can only turn a
    # cache hit into a miss, and base legs are the most reusable artifact in the system.
    #   "claude_cli" — today's path, `bash scripts/tau3_evolver.sh` -> `claude -p`
    #   "direct"     — src/orchestrator/evolver_direct_runner.py, confined by the deny-first
    #                  sandbox profile in evolver_dispatch.py (direct has NO PreToolUse hooks)
    evolver_harness: str = "claude_cli"
    n_epochs: int = 4

    # decision rules
    gate: GateSpec = field(default_factory=GateSpec)
    headroom: HeadroomGate = field(default_factory=HeadroomGate)

    # budget
    cost_ceiling_usd: float = 0.0            # 0 = no ceiling declared (validate() rejects)

    # ── canonical form ───────────────────────────────────────────────────────────────────
    def _canonical(self) -> str:
        """Stable JSON. Tuples -> lists, keys sorted, no whitespace drift. This string IS
        the identity of the experiment; anything not in it cannot be claimed as controlled."""
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=list)

    @property
    def config_sha256(self) -> str:
        return hashlib.sha256(self._canonical().encode()).hexdigest()

    # The ONLY fields that can change the bytes an episode produces AND are absent from the
    # RunCache record key. The record key is {tree_hash, question={domain,task_id}, model,
    # effort}; the cache DIR name adds sample index and leg token. Everything below is what
    # neither of those captures, so the tag has to carry it — that is the whole reason
    # #182 (episode wall) and #183 (user simulator) mandated a derived tag in the first place.
    #
    # ALLOWLIST, NOT DENYLIST. Adding a field that matters is a one-line fix; forgetting one is
    # a silent replay of stale episodes into a changed instrument, which is exactly #182/#183.
    # So a new spec field is EXCLUDED by default and must be argued INTO this tuple.
    _INSTRUMENT = ("benchmark", "domain", "variant", "solver_model", "solver_provider_order",
                   "solver_allow_fallbacks", "user_sim_model", "judge_model",
                   "agent_clean_s", "agent_turns", "episode_timeout_s",
                   # #249: the harness IS the instrument. See the field's own comment.
                   "solver_harness")

    @property
    def instrument_sha256(self) -> str:
        """Hash of the measuring apparatus alone."""
        d = {k: getattr(self, k) for k in self._INSTRUMENT}
        return hashlib.sha256(
            json.dumps(d, sort_keys=True, separators=(",", ":"), default=list).encode()
        ).hexdigest()

    @property
    def cache_tag(self) -> str:
        """DERIVED, never hand-typed (#182/#183) — but from the INSTRUMENT, not the whole config.

        It hashed `config_sha256` until 2026-08-09, i.e. all 24 spec fields, ten of which cannot
        touch an episode: evolver_model, n_epochs, cost_ceiling_usd, name, gate, headroom,
        test_sealed and the three split sizes. An evolver is not present when a solver rollout
        runs; a cost ceiling is a budget; a gate is applied to results afterwards. Keying on them
        is safe and never free — it can only ever turn a hit into a miss.

        Measured on the first agentic run: `hits=0 fresh=30` on every sample, $1.45 spent, for a
        BASE leg whose 22 overlapping tasks had already been computed byte-identically under a
        different arm's tag. Base legs carry no skills and are the most reusable artifact in the
        system, and every new arm was re-buying them.

        The split is deliberately NOT here either: which tasks a split SELECTS cannot change what
        an episode for task 63 does, and `question` already carries the task id inside the record.
        Two arms that draw different splits from one pool now share every task they have in common.

        NOT n_samples: the sample index lives in the cache DIR name (`_s0_`), so a 3-sample and a
        5-sample run legitimately share s0..s2.
        """
        return f"{self.name}_{self.instrument_sha256[:12]}"

    # ── validation ───────────────────────────────────────────────────────────────────────
    def validate(self) -> list:
        """Returns a list of failure strings. Empty list = valid. Each rule below can
        actually fail on a reachable input — see selftest()."""
        errs = []

        if not self.name or not self.name.replace("_", "").replace("-", "").isalnum():
            errs.append(f"name must be alphanumeric/_/-, got {self.name!r}")
        for f in ("benchmark", "domain", "solver_model", "evolver_model", "split_path"):
            if not getattr(self, f):
                errs.append(f"{f} is required and empty")
        # A user simulator and a judge are not universal. tau3 has both; SSB v2 has NEITHER —
        # its tasks are scored by a script against a golden workbook, with nobody to talk to.
        # Requiring a non-empty model here made the spec unable to express a whole class of
        # benchmark, which surfaced the first time the intake CLI was pointed at a toy dataset
        # with no conversation in it. The field must still be ANSWERED — "none" is a deliberate
        # statement that this benchmark has no such component, and an empty string is still an
        # omission, because the difference between "no judge" and "I forgot" is the whole point.
        for f in ("user_sim_model", "judge_model"):
            if not getattr(self, f):
                errs.append(f"{f} is required — give a model, or 'none' if this benchmark "
                            f"has no such component")

        # ── SSB v2 REFUSES A ROLLING SOLVER TAG, AND THE FREEZE IS WHERE TO SAY SO ─────
        # scripts/ssbv2_leg.sh:86 accepts only dated snapshots (*-0731, *-0813), because
        # `deepseek/deepseek-v4-flash` and `deepseek/deepseek-v4-flash-0731` are different
        # models and the RunCache keys on the model STRING: the rolling tag does not corrupt
        # an existing bucket, it silently opens a new one and re-buys every episode under a
        # model nobody chose. That rule lived ONLY in the leg launcher, which is the LAST
        # step -- an operator could brief the AO with any solver, get a frozen spec, build
        # the oracle, and only then find that no leg will ever run. The constraint belongs
        # where the choice is made.
        if (self.benchmark or "").lower() == "ssbv2" and self.solver_model:
            # THE SAME LIST THE LEG REFUSES ON, not merely "looks dated". My first version of
            # this check accepted any -NNNN suffix, so a spec naming `…-0901` froze cleanly and
            # scripts/ssbv2_leg.sh then refused it with exit 4 — reproducing at the freeze the
            # very "find out after paying for the AO run and the oracle" this check exists to
            # prevent. tests/test_ssbv2_solver_snapshot.py reads the shell case statement and
            # asserts the two stay identical, because extending one and forgetting the other
            # is how they drifted in the first place.
            if not self.solver_model.endswith(SSBV2_SOLVER_SNAPSHOTS):
                errs.append(
                    f"benchmark=ssbv2 accepts only an allow-listed dated solver snapshot "
                    f"{list(SSBV2_SOLVER_SNAPSHOTS)}; got {self.solver_model!r}. A rolling tag "
                    f"re-buys every episode under a different model than the one named, and "
                    f"scripts/ssbv2_leg.sh refuses anything else with exit 4 — so freezing it "
                    f"would buy an AO run and an oracle for legs that can never run.")

        # data
        if self.n_train <= 0 or self.n_val <= 0:
            errs.append(f"n_train and n_val must be > 0, got {self.n_train}/{self.n_val}")
        if not self.test_sealed:
            errs.append("test_sealed=False: this module will not describe a run that "
                        "touches the held-out test split during evolution")
        if self.split_path:
            p = Path(self.split_path)
            if not p.is_absolute():
                p = REPO / p
            if not p.exists():
                errs.append(f"split_path does not exist: {p}")
            else:
                actual = split_digest(p)
                if not self.split_sha256:
                    errs.append("split_sha256 is empty — a split you did not hash is a "
                                "split you cannot prove you used")
                elif actual != self.split_sha256:
                    errs.append(f"split_sha256 mismatch: spec {self.split_sha256[:16]} "
                                f"vs file {actual[:16]}")
                else:
                    sp = json.loads(p.read_text())
                    for key, want in (("train", self.n_train), ("val", self.n_val),
                                      ("test", self.n_test)):
                        got = sp.get(key)
                        got_n = len(got) if isinstance(got, list) else got
                        if want and got_n != want:
                            errs.append(f"split {key}: spec says {want}, file says {got_n}")

        # instrument
        if self.n_samples < 1:
            errs.append(f"n_samples must be >= 1, got {self.n_samples}")
        if self.episode_timeout_s <= self.agent_clean_s:
            errs.append(f"episode_timeout_s ({self.episode_timeout_s}) must exceed "
                        f"agent_clean_s ({self.agent_clean_s}) — an outer wall at or below "
                        f"the inner one makes the inner unreachable (#175)")
        if self.solver_provider_order and self.solver_allow_fallbacks:
            errs.append("a provider order with allow_fallbacks=True is not a pin — both "
                        "halves are required (cross-family pin)")
        # the pin must not reach outside the model's own family
        if self.solver_provider_order and self.solver_model:
            fam = self.solver_model.split("/")[0].lower()
            for prov in self.solver_provider_order:
                if fam and fam not in prov.lower() and prov.lower() not in fam:
                    errs.append(f"provider pin {prov!r} is outside the family of "
                                f"{self.solver_model!r} — a pin must not cross families")

        # gate
        g = self.gate
        if g.pairing not in VALID_PAIRING:
            errs.append(f"gate.pairing must be one of {VALID_PAIRING}, got {g.pairing!r}")
        if not g.tests:
            errs.append("gate.tests is empty — an undeclared decision rule is not a gate")
        if not (0 < g.alpha < 1):
            errs.append(f"gate.alpha must be in (0,1), got {g.alpha}")
        if g.n_comparisons < 1:
            errs.append(f"gate.n_comparisons must be >= 1, got {g.n_comparisons}")
        if g.discordant_floor < DISCORDANT_FLOOR:
            errs.append(f"gate.discordant_floor {g.discordant_floor} is below the exact-"
                        f"binomial floor {DISCORDANT_FLOOR}: with fewer discordant pairs no "
                        f"two-sided p can reach 0.05, so the gate could never fire (#154)")
        # power sanity: the maximum available pairs must at least be able to reach the floor
        max_pairs = self.n_val * (self.n_samples if g.pairing == "episode_level" else 1)
        if max_pairs < g.discordant_floor:
            errs.append(f"gate is unreachable: pairing={g.pairing} on n_val={self.n_val} x "
                        f"n_samples={self.n_samples} yields at most {max_pairs} pairs, "
                        f"below the floor of {g.discordant_floor}")

        # headroom
        h = self.headroom
        if not (0 <= h.min_train_pass_rate < h.max_train_task_paired <= 1):
            errs.append(f"headroom window is empty or inverted: "
                        f"[{h.min_train_pass_rate}, {h.max_train_task_paired}]")

        # budget
        if self.cost_ceiling_usd <= 0:
            errs.append("cost_ceiling_usd must be declared (> 0) — an unbounded run has no "
                        "stopping condition that is not a human noticing")

        if self.n_epochs < 1:
            errs.append(f"n_epochs must be >= 1, got {self.n_epochs}")
        if g.n_comparisons < self.n_epochs:
            errs.append(f"gate.n_comparisons ({g.n_comparisons}) < n_epochs "
                        f"({self.n_epochs}): the correction must cover every epoch gated")

        return errs

    # ── freeze / verify ──────────────────────────────────────────────────────────────────
    def freeze(self, run_dir) -> Path:
        """Write SPEC.json once. Refuses to overwrite a DIFFERENT frozen spec."""
        errs = self.validate()
        if errs:
            raise SpecError("refusing to freeze an invalid spec:\n  - " + "\n  - ".join(errs))
        rd = Path(run_dir)
        rd.mkdir(parents=True, exist_ok=True)
        out = rd / SPEC_FILENAME
        payload = {
            "spec": json.loads(self._canonical()),
            "config_sha256": self.config_sha256,
            "cache_tag": self.cache_tag,
        }
        if out.exists():
            prev = json.loads(out.read_text())
            if prev.get("config_sha256") != self.config_sha256:
                raise SpecError(
                    f"{out} already holds a DIFFERENT frozen spec "
                    f"({prev.get('config_sha256','?')[:12]} vs {self.config_sha256[:12]}). "
                    f"A frozen spec is a pre-registration; editing it after the fact is the "
                    f"selection move it exists to prevent. Start a new run dir.")
            return out
        out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
        return out

    @classmethod
    def load(cls, run_dir):
        p = Path(run_dir) / SPEC_FILENAME
        if not p.exists():
            raise SpecError(f"no frozen spec at {p}")
        payload = json.loads(p.read_text())
        d = dict(payload["spec"])
        d["gate"] = GateSpec(**{**d.get("gate", {}),
                                "tests": tuple(d.get("gate", {}).get("tests", ()))})
        d["headroom"] = HeadroomGate(**d.get("headroom", {}))
        d["solver_provider_order"] = tuple(d.get("solver_provider_order", ()))
        return cls(**d), payload

    @staticmethod
    def pin_cache_tag(run_dir, reason: str) -> str:
        """Record that this run's episodes live under the STORED tag, and why.

        Written beside the spec, never inside it: documenting a pin must not alter the frozen
        pre-registration's digest. Returns the pinned tag.
        """
        p = Path(run_dir) / SPEC_FILENAME
        payload = json.loads(p.read_text())
        payload["cache_tag_pin"] = reason
        p.write_text(json.dumps(payload, indent=1))
        return payload["cache_tag"]

    @classmethod
    def assert_unmutated(cls, run_dir) -> str:
        """Re-derive the hash from the stored fields and compare to the stored hash.
        Catches an edit to SPEC.json that forgot to (or could not) update the digest."""
        spec, payload = cls.load(run_dir)
        stored = payload.get("config_sha256")
        actual = spec.config_sha256
        if stored != actual:
            raise SpecError(
                f"SPEC.json has been MUTATED since freeze: stored digest {stored[:16]} but "
                f"its own fields hash to {actual[:16]}. Every number produced under this "
                f"run dir is now of unknown provenance.")
        # THE STORED TAG IS THE RECORD OF WHERE THIS RUN'S EPISODES ACTUALLY LIVE.
        # Drift here does not mean the spec was edited (config_sha256 already proved it was
        # not) — it means the DERIVATION CHANGED under a run whose episodes are already on
        # disk. That happened on 2026-08-09 when the tag stopped hashing all 24 config fields
        # and started hashing the 11-field instrument. Silently accepting it would let a later
        # leg write into a different bucket than the base legs it is compared against; silently
        # refusing would strand every run that predates the improvement.
        #
        # So: accept the stored tag, but ONLY against a written acknowledgement. `cache_tag_pin`
        # lives beside the spec dict, not inside it, so recording it cannot change
        # config_sha256 — the pre-registration is untouched by the act of documenting a pin.
        if payload.get("cache_tag") != spec.cache_tag:
            pin = payload.get("cache_tag_pin")
            if not pin:
                raise SpecError(
                    f"cache_tag drifted: stored {payload.get('cache_tag')} vs derived "
                    f"{spec.cache_tag} (#182/#183 — the tag is the only thing separating this "
                    f"config's episodes from another's).\n"
                    f"  If the DERIVATION changed under a run already on disk, the stored tag "
                    f"is where its episodes are. Record why with `pin_cache_tag(run_dir, "
                    f"reason)` and the stored tag becomes authoritative for this run.")
        return actual


# ── layer 1 boundary ─────────────────────────────────────────────────────────────────────
def from_natural_language(description: str, *, model: str = "") -> ExperimentSpec:
    """NOT IMPLEMENTED — deliberately.

    The intended contract: an LLM reads `description`, fills every field above, and returns
    a spec that a human confirms before freeze(). Two properties are non-negotiable when it
    is built:

      * It must NOT default silently. A field the description did not determine must surface
        as a question, not as this module's dataclass default. The defaults here exist so a
        spec can be written in code, not so a guess can be laundered into a sha256.
      * freeze() must remain human-gated. The model proposes; the person running the
        experiment pre-registers.

    Returning a plausible-looking object from a guess is worse than raising: freeze() would
    stamp a digest on it and assert_unmutated() would then defend the fiction.
    """
    raise NotImplementedError(
        "from_natural_language is the layer-1 front end and is not built yet. "
        "Construct ExperimentSpec(...) explicitly and call .validate() / .freeze().")


# ── the tau3 banking spec, as actually run ───────────────────────────────────────────────
def tau3_banking_2525() -> ExperimentSpec:
    """The live configuration, expressed as a spec. Useful as a worked example and as the
    fixture the selftest mutates."""
    sp = REPO / "outputs/tau3_banking_2525/split_252547.json"
    sha = split_digest(sp) if sp.exists() else ""
    return ExperimentSpec(
        name="tau3_banking_2525",
        benchmark="tau3",
        domain="banking_knowledge",
        variant="bm25_grep",
        split_path=str(sp),
        split_sha256=sha,
        n_train=25, n_val=25, n_test=47, test_sealed=True,
        n_samples=3,
        solver_model="deepseek/deepseek-v4-flash-0731",
        solver_provider_order=("deepseek",),
        solver_allow_fallbacks=False,
        user_sim_model="openai/gpt-5-nano",
        judge_model="openai/gpt-4.1",
        agent_clean_s=1200, agent_turns=100, episode_timeout_s=5400,
        evolver_model="claude-opus-5",
        n_epochs=4,
        gate=GateSpec(n_comparisons=4),
        headroom=HeadroomGate(min_train_pass_rate=0.08, max_train_task_paired=0.90),
        cost_ceiling_usd=400.0,
    )


# ── selftest: every rule must FAIL on a known-bad input ──────────────────────────────────
def _mut(base: ExperimentSpec, **kw) -> ExperimentSpec:
    d = asdict(base)
    d["gate"] = GateSpec(**{**d["gate"], "tests": tuple(d["gate"]["tests"])})
    d["headroom"] = HeadroomGate(**d["headroom"])
    d["solver_provider_order"] = tuple(d["solver_provider_order"])
    d.update(kw)
    return ExperimentSpec(**d)


def selftest() -> int:
    print("SELFTEST — each rule must FAIL on a known-bad input\n")
    base = tau3_banking_2525()
    # SELF-CONTAINED, because a selftest that only runs on one machine is not a test.
    # This used to bail with "live split file absent" whenever
    # outputs/tau3_banking_2525/split_252547.json was missing — i.e. on every clone but the
    # one this was written in — and return 2 having exercised 0 of the 12 rules below. The
    # same defect was fixed in loop.py --selftest; this is its twin. Synthesize a split with
    # the fixture's own shape and re-point the fixture at it, so all 12 rules run anywhere.
    _tmp = None
    if not (REPO / base.split_path).exists() and not Path(base.split_path).exists():
        _tmp = tempfile.mkdtemp(prefix="evoskill_spec_selftest_")
        _sp = Path(_tmp) / "split_252547.json"
        _sp.write_text(json.dumps({
            "train": [f"task_{i:03d}" for i in range(1, base.n_train + 1)],
            "val": [f"task_{i:03d}" for i in range(base.n_train + 1,
                                                   base.n_train + base.n_val + 1)],
            "test": [f"task_{i:03d}" for i in range(base.n_train + base.n_val + 1,
                                                    base.n_train + base.n_val
                                                    + base.n_test + 1)],
        }, indent=1))
        # Hash it the same way the real generator would be hashed, or the very first rule
        # ("valid baseline PASSES") would fail on a sha mismatch of our own making.
        base = _mut(base, split_path=str(_sp), split_sha256=split_digest(_sp))
        print(f"  [note] live split absent — running against a synthesized split at {_sp}\n")

    bad = [
        ("valid baseline PASSES", base, None),
        ("outer wall <= inner wall (#175)",
         _mut(base, episode_timeout_s=1200), "must exceed"),
        ("provider order without allow_fallbacks=False",
         _mut(base, solver_allow_fallbacks=True), "not a pin"),
        ("cross-family provider pin",
         _mut(base, solver_provider_order=("novita",)), "outside the family"),
        ("discordant floor below the exact-binomial floor (#154)",
         _mut(base, gate=GateSpec(discordant_floor=3, n_comparisons=4)), "could never fire"),
        ("gate unreachable at this n (power)",
         _mut(base, n_val=1, gate=GateSpec(pairing="task_paired", n_comparisons=4)),
         "unreachable"),
        ("Bonferroni does not cover every epoch",
         _mut(base, n_epochs=8), "must cover every epoch"),
        ("split sha mismatch",
         _mut(base, split_sha256="0" * 64), "mismatch"),
        ("split size disagrees with the file",
         _mut(base, n_train=99), "spec says 99"),
        ("test split not sealed",
         _mut(base, test_sealed=False), "will not describe a run"),
        ("no cost ceiling",
         _mut(base, cost_ceiling_usd=0.0), "cost_ceiling_usd must be declared"),
        ("empty headroom window (#180/#206)",
         _mut(base, headroom=HeadroomGate(min_train_pass_rate=0.9,
                                          max_train_task_paired=0.5)), "inverted"),
    ]
    fails = 0
    for label, spec, want in bad:
        errs = spec.validate()
        if want is None:
            ok = not errs
            detail = "no errors" if ok else f"UNEXPECTED: {errs}"
        else:
            ok = any(want in e for e in errs)
            detail = next((e for e in errs if want in e), f"NOT CAUGHT (errors={errs})")
        print(f"  [{'ok' if ok else 'BAD'}] {label}\n        {detail[:150]}")
        fails += 0 if ok else 1

    # cache tag must change when the INSTRUMENT changes (#182/#183). This asserted
    # `n_samples=5` changed the bucket while the tag was config-derived; it no longer does, and
    # that is correct — the sample index lives in the cache DIR name (`_s0_`), so a 3-sample and
    # a 5-sample run legitimately share s0..s2. Re-pointed at a field that genuinely alters what
    # an episode does. The exhaustive per-field sweep is below.
    t0 = base.cache_tag
    t1 = _mut(base, user_sim_model="someone-else").cache_tag
    ok = t0 != t1
    print(f"  [{'ok' if ok else 'BAD'}] cache tag changes with the instrument (#183)\n"
          f"        {t0} -> {t1} ({'changed' if ok else 'DID NOT CHANGE'})")
    fails += 0 if ok else 1

    # from_natural_language must refuse rather than guess
    try:
        from_natural_language("evolve a banking agent")
        print("  [BAD] from_natural_language returned a spec instead of raising")
        fails += 1
    except NotImplementedError:
        print("  [ok] from_natural_language refuses rather than guessing")

    # ── the cache tag must key on the INSTRUMENT and nothing else ────────────────────────
    # Exhaustive and per-field on purpose. A denylist regression here does not crash and does
    # not show up in any result — it either silently replays stale episodes into a changed
    # instrument (#182/#183) or silently re-buys episodes that were already correct. Both are
    # invisible without exactly this test, so every field is asserted individually.
    from dataclasses import replace as _replace
    _b = tau3_banking_2525()
    print("\n  cache tag keys on the instrument, and only the instrument")
    for _f, _v in (("evolver_model", "something-else"), ("n_epochs", 99),
                   ("cost_ceiling_usd", 999.0), ("n_train", 1), ("n_val", 1), ("n_test", 1),
                   ("test_sealed", False), ("split_path", "/tmp/elsewhere.json"),
                   ("split_sha256", "f" * 64), ("n_samples", 7), ("name", "renamed"),
                   # #249: the evolver's harness, like the evolver's model, is not present when
                   # a solver episode runs, so it must not fork the bucket.
                   ("evolver_harness", "direct")):
        _o = _replace(_b, **{_f: _v})
        # `name` is a label and legitimately prefixes the tag; compare the HASH for that one.
        _same = (_o.instrument_sha256 == _b.instrument_sha256 if _f == "name"
                 else _o.cache_tag == _b.cache_tag)
        print(f"  [{'ok' if _same else 'BAD'}] {_f} does NOT change the cache bucket "
              f"({_f}={_v!r} cannot affect an episode)")
        fails += 0 if _same else 1
    for _f, _v in (("benchmark", "other"), ("domain", "retail"), ("variant", "none"),
                   ("solver_model", "other"), ("solver_provider_order", ("elsewhere",)),
                   ("solver_allow_fallbacks", True), ("user_sim_model", "other"),
                   ("judge_model", "other"), ("agent_clean_s", 1), ("agent_turns", 1),
                   ("episode_timeout_s", 1),
                   # #249: the SOLVER's harness decides tool schemas, loop shape and prompt
                   # assembly, so the same model on two harnesses is two instruments.
                   ("solver_harness", "claude")):
        _o = _replace(_b, **{_f: _v})
        _ch = _o.cache_tag != _b.cache_tag
        print(f"  [{'ok' if _ch else 'BAD'}] {_f} DOES change the cache bucket "
              f"({_f}={_v!r} changes what an episode does)")
        fails += 0 if _ch else 1

    if _tmp:
        shutil.rmtree(_tmp, ignore_errors=True)
    print(f"\n{'ALL CHECKS BEHAVED' if not fails else str(fails) + ' CHECK(S) MISBEHAVED'}")
    return 0 if not fails else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check", metavar="RUN_DIR",
                    help="verify a frozen SPEC.json has not been mutated")
    ap.add_argument("--emit", metavar="RUN_DIR",
                    help="freeze the tau3 banking spec into RUN_DIR (will not overwrite)")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.check:
        try:
            d = ExperimentSpec.assert_unmutated(a.check)
        except SpecError as e:
            print(f"FAIL {e}")
            return 1
        print(f"ok  SPEC.json intact, config_sha256={d[:16]}")
        return 0
    if a.emit:
        s = tau3_banking_2525()
        try:
            p = s.freeze(a.emit)
        except SpecError as e:
            print(f"FAIL {e}")
            return 1
        print(f"ok  froze {p}\n    config_sha256={s.config_sha256[:16]}\n    cache_tag={s.cache_tag}")
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
