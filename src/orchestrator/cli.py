#!/usr/bin/env python3
"""EvoSkill v2 — the front door.

    python3 -m src.orchestrator.cli new                     # interactive
    python3 -m src.orchestrator.cli new --script answers.json --run-dir outputs/foo
    python3 -m src.orchestrator.cli selftest

WHAT THIS IS. The thing a user meets first. It asks what their benchmark is, INTERROGATES the
answer until every part of it is checkable, goes and checks it against their files, asks for
train/val/test counts, builds the split, and freezes an ExperimentSpec. From there
`src.orchestrator.loop` drives the epochs.

THE ONE RULE THAT SHAPES EVERYTHING. It never writes a field it has not verified. When the
dataset disagrees with what it was told, it ASSUMES IT MISUNDERSTOOD THE USER, asks one specific
question, and re-checks (the maintainer, 2026-08-09). It does not correct the user silently, and it does
not record the mismatch and continue — an unverified field flowing into a frozen spec is a guess
wearing a digest.

WHY --script EXISTS. The acceptance test is "role-play a user, define a benchmark you have
already evolved by hand, and check the numbers land in the same place". That test has to be
repeatable and diffable, so every answer a human would type can come from a JSON file instead.
Same code path, same verification, same refusals — only the input stream differs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.orchestrator.benchmark import (BenchmarkClaim, adopt_split,   # noqa: E402
                                        build_split, render, unresolved, verify)


class Answers:
    """The user's side of the conversation — a live human or a recorded script."""

    def __init__(self, scripted=None):
        self.scripted = scripted or {}
        self.transcript = []

    def ask(self, key: str, prompt: str, default: str = "") -> str:
        if key in self.scripted:
            v = str(self.scripted[key])
            print(f"{prompt}\n  > {v}")
        elif self.scripted and not sys.stdin.isatty():
            # SCRIPTED MODE ON A PIPE MUST NEVER FALL THROUGH TO input(). stdin is usually a pipe
            # rather than a closed handle, so input() BLOCKS FOREVER instead of raising EOFError —
            # the first real-dataset run of this CLI hung for two minutes on exactly that, waiting
            # for a human who was never coming. A missing answer is a missing answer; say so.
            # With a LIVE TTY the hang cannot happen, so a partial script (or an accepted --brief
            # proposal, which fills scripted the same way) falls through and asks the human —
            # #315: the brief render promises "WILL ASK YOU" and this is where that promise is kept.
            raise SystemExit(
                f"\nSCRIPT INCOMPLETE — no answer for {key!r}.\n"
                f"  I asked: {prompt.strip()}\n"
                f"  Add {key!r} to the script (or, with --brief, state it in the brief) and "
                f"re-run. No experiment definition has been written; intake artifacts already "
                f"in the run dir remain for audit.")
        else:
            try:
                v = input(f"{prompt}\n  > ").strip() or default
            except EOFError:
                raise SystemExit(f"\nno answer available for {key!r} and stdin is closed. "
                                 f"Add it to the --script file.")
        self.transcript.append({"key": key, "prompt": prompt, "answer": v})
        return v

    def ask_int(self, key: str, prompt: str) -> int:
        while True:
            raw = self.ask(key, prompt)
            try:
                return int(raw)
            except ValueError:
                print(f"  '{raw}' is not a whole number.")
                if key in self.scripted:
                    raise SystemExit(f"scripted answer for {key!r} is not an integer")


def build_claim(a: Answers) -> BenchmarkClaim:
    print("\n" + "=" * 76)
    print("Tell me about your benchmark. I will check everything you say against the files,")
    print("so approximate answers are fine — I will come back and ask about anything that")
    print("does not line up.")
    print("=" * 76)

    name = a.ask("name", "\nWhat is this benchmark called?")
    # Free prose FIRST. It is never parsed into behaviour — it is kept verbatim as the record of
    # what the user believed they were setting up, which is the thing a later reader needs when a
    # number looks wrong. The structured questions below are what actually drive anything.
    desc = a.ask("description", "\nDescribe it in your own words — what does one task ask the "
                                "agent to do, and how is a response judged?")
    domain = a.ask("domain", "\nWhat does the harness call this task pool / domain?\n"
                             "  (the exact string it expects, e.g. 'retail')")
    root = a.ask("root", "\nWhere does the data live? (path, relative to the repo root)")
    task_glob = a.ask("task_glob", "\nWhat pattern picks out the tasks under that path?\n"
                                   "  e.g. 'data/*' if each task is a folder, or "
                                   "'tasks/all.json' if they are records inside one file")
    records_in = a.ask("records_in", "\nAre the tasks records INSIDE the matched file(s)?\n"
                                     "  'array' | 'object' | 'jsonl' | blank if one entry per task",
                       default="")
    id_from = a.ask("id_from", "\nHow is a task's id determined?\n"
                               "  'stem' | 'name' | 'parent' | 'json:<key>'", default="stem")
    input_glob = a.ask("input_glob", "\nWithin a task, what file is the agent GIVEN? "
                                     "(blank if the task entry itself is the input)", default="")
    answer_glob = a.ask("answer_glob", "\nAnd where is the ground truth? "
                                       "(blank if it lives inside the task record)", default="")
    scoring_desc = a.ask("scoring_desc", "\nHow does a response become a score?")
    scoring_script = a.ask("scoring_script", "\nIs there a scoring script? "
                                             "(path relative to the data root, or blank)",
                           default="")
    return BenchmarkClaim(
        name=name, root=root, task_glob=task_glob, records_in=records_in, id_from=id_from,
        input_glob=input_glob, answer_glob=answer_glob, scoring_desc=scoring_desc,
        scoring_script=scoring_script, domain=domain, notes=desc)


# ── the offline test double ──────────────────────────────────────────────────────────────
# TEST ONLY, and reachable ONLY via EVOSKILL_CLI_NO_AO=1. This is the eight-entry table that used
# to BE the mechanism, demoted to a stub so `selftest` stays offline, free and deterministic.
# It is not a fallback: production never reaches it, and when a check is missing from it the stub
# raises ClarifyUnavailable — which exercises the degraded path rather than papering over it.
# The whole point of gap 1 is that a closed table cannot phrase a question about evidence it
# never read; leaving one here as the real path would undo the change.
_TEST_ONLY_FIELD_FOR_CHECK = {
    "dataset root exists": "root",
    "the task pattern enumerates tasks": "task_glob",
    "one entry per task": "records_in",
    "every task has an id, and ids are unique": "id_from",
    "every task has its input": "input_glob",
    "every task has its answer": "answer_glob",
    "the scoring script exists": "scoring_script",
    "the scoring script exposes a callable": "scoring_script",
}


def _stub_compose(claim, finding, *, round_no=1, model="", timeout_s=0, log_path=None):
    from src.orchestrator.clarify import ClarifyUnavailable
    fld = _TEST_ONLY_FIELD_FOR_CHECK.get(getattr(finding, "check", ""))
    if not fld:
        raise ClarifyUnavailable(f"[stub] no mapping for {getattr(finding, 'check', '')!r}")
    return {"field": fld, "question": finding.question or finding.detail,
            "why": "[stub composer: EVOSKILL_CLI_NO_AO=1]", "model": "stub", "round": round_no}


def _ask_which_field(a: Answers, claim: BenchmarkClaim, f, rnd: int, why: str) -> str:
    """THE DEGRADED PATH: the composer could not be used, so the USER picks the field.

    Note what this is NOT: a fallback to guessing. The old eight-entry table guessed — silently
    and correctly-by-luck — which of the user's answers to revise. When the AO cannot be reached
    the honest move is to say so and ask, because the one party who certainly knows which of their
    own statements was loose is the person who made them.
    """
    from src.orchestrator.clarify import REVISABLE
    fields = [x for x in REVISABLE if hasattr(claim, x)]
    print(f"\n  (I could not have the orchestrator phrase this for you: {why})")
    print("  So I will ask you plainly, and you tell me which of your answers to revise.")
    print("     " + " | ".join(fields))
    while True:
        which = a.ask(f"which_field#{rnd}",
                      f"\n{f.question or f.detail}\n"
                      f"  Which of your earlier answers should I revise? "
                      f"(one of: {', '.join(fields)})").strip()
        if which in fields:
            return which
        print(f"  '{which}' is not one of {fields}.")
        if a.scripted:
            raise SystemExit(f"scripted field choice {which!r} is not revisable")


def clarify_loop(claim: BenchmarkClaim, a: Answers, max_rounds: int = 6, *, compose=None,
                 log_dir=None, model: str = "opus[1m]"):
    """Verify, and on any failure have the AO ASK ONE QUESTION, then re-verify.

    Returns (claim, findings, conversation) where `conversation` records, per round, what failed,
    which field the AO chose, why, and the question it put — so the setup is auditable afterwards
    rather than only its outcome.

    THE DIVISION OF LABOUR, restated here because it is easy to erode:
      verify()  decides what is and is not satisfied. Always. The AO never sees a vote.
      the AO    chooses which field to re-ask and phrases the question, having read the evidence.
      the user  answers. Nothing else may supply a value for a claim field.

    BOUNDED. If the same check keeps failing after a few rounds the loop stops and hands the
    problem back whole, rather than asking the same question forever — a clarification loop that
    cannot terminate is worse than a refusal, because it looks like progress.
    """
    from src.orchestrator.clarify import ClarifyUnavailable
    from src.orchestrator.clarify import compose as _default_compose
    compose = compose or _default_compose
    convo = []
    for rnd in range(1, max_rounds + 1):
        findings = verify(claim)
        bad = unresolved(findings)
        print("\n" + render(findings))
        if not bad:
            return claim, findings, convo
        f = bad[0]
        print(f"\n  --- round {rnd}: I need to understand this better "
              f"(I am assuming I misread you, not that your data is wrong) ---")
        try:
            plan = compose(claim, f, round_no=rnd, model=model,
                           log_path=(Path(log_dir) / f"clarify_r{rnd}.log") if log_dir else None)
            field_name, question, why = plan["field"], plan["question"], plan.get("why", "")
            if why:
                print(f"  (asking about '{field_name}' because: {why})")
        except ClarifyUnavailable as e:
            field_name, question, why = _ask_which_field(a, claim, f, rnd, str(e)), "", f"DEGRADED: {e}"
            new = getattr(claim, field_name)
            convo.append({"round": rnd, "check": f.check, "detail": f.detail,
                          "field": field_name, "why": why, "question": f.question or f.detail,
                          "answer": new, "composer": "unavailable"})
            new = a.ask(f"{field_name}#{rnd}", "\n  New value for that answer:")
            setattr(claim, field_name, new)
            convo[-1]["answer"] = new
            continue
        new = a.ask(f"{field_name}#{rnd}", f"\n{question}")
        setattr(claim, field_name, new)
        convo.append({"round": rnd, "check": f.check, "detail": f.detail, "field": field_name,
                      "why": why, "question": question, "answer": new,
                      "composer": plan.get("model", model)})
    raise SystemExit(f"\nstill unresolved after {max_rounds} rounds. Nothing has been written. "
                     f"The outstanding check is: {unresolved(verify(claim))[0].detail}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["new", "selftest"])
    ap.add_argument("--script", default="", help="JSON of pre-recorded answers")
    ap.add_argument("--run-dir", default="", help="where this experiment's artifacts go")
    ap.add_argument("--split-out", default="", help="path for the split file")
    ap.add_argument("--split", default="",
                    help="ADOPT this existing split instead of building a new one. Use this for "
                         "any follow-up run on a benchmark you have already split: re-partitioning "
                         "the same pool with a fresh seed puts the PREVIOUS run's sealed test "
                         "tasks into this run's train, and nothing would report an error.")
    # ── gap 2: prose in, setup out ───────────────────────────────────────────────────────────
    ap.add_argument("--brief", default="",
                    help="a file holding your instruction in prose ('-' for stdin). The AO reads "
                         "your repo and PROPOSES every answer; you countersign; the front door "
                         "then verifies each one against your files exactly as if you had typed "
                         "them.")
    ap.add_argument("--root-hint", default="", help="with --brief: where the data probably lives")
    ap.add_argument("--yes", action="store_true",
                    help="with --brief: accept the proposal without the interactive "
                         "countersignature (recorded as such in BENCHMARK.json)")
    ap.add_argument("--phoenix-project", default="",
                    help="emit the operator<->AO setup conversation to this Phoenix project")
    ap.add_argument("--benchmark", default="tau3",
                    help="the harness family this run belongs to (tau3 | ssbv2 | ...). It is in "
                         "ExperimentSpec._INSTRUMENT and therefore in the CACHE TAG, so a wrong "
                         "value makes one benchmark read another's episodes out of cache.")
    ap.add_argument("--spec-extra", default="",
                    help="JSON of additional ExperimentSpec fields the questions do not cover — "
                         "instrument pins like variant, solver_provider_order, episode_timeout_s. "
                         "OPERATOR-SUPPLIED ONLY: these are never proposed by the AO, because a "
                         "wrong provider pin or retrieval variant changes what every number means "
                         "and would be invisible in the answers.")
    args = ap.parse_args()
    if args.cmd == "selftest":
        return selftest()

    benchmark = args.benchmark
    scripted = json.loads(Path(args.script).read_text()) if args.script else {}

    # ── the brief, if there is one ───────────────────────────────────────────────────────────
    # It fills `scripted`, so from here down NOTHING changes: the same `Answers` object, the same
    # verify(), the same clarify_loop, the same refusals. An explicit --script entry always wins
    # over the AO's proposal — a value the operator typed is not up for reinterpretation.
    brief_text, brief_meta = "", {}
    if args.brief:
        from src.orchestrator import brief as _brief
        brief_text = (sys.stdin.read() if args.brief == "-"
                      else Path(args.brief).read_text())
        if not brief_text.strip():
            raise SystemExit("the --brief file is empty; nothing has been written")
        rd_early = Path(args.run_dir) if args.run_dir else None
        if rd_early:
            rd_early.mkdir(parents=True, exist_ok=True)
        print("\n  Reading your brief and your repository. This takes a few minutes — the AO is "
              "\n  opening your data files, not pattern-matching your sentence.\n")
        try:
            proposed = _brief.propose(
                brief_text, root_hint=args.root_hint,
                log_path=(rd_early / "BRIEF_INTAKE.log") if rd_early else None)
        except _brief.BriefUnavailable as e:
            raise SystemExit(
                f"\nTHE AO COULD NOT TURN THIS BRIEF INTO A SETUP:\n  {e}\n"
                f"\n  Nothing has been written. Re-run without --brief to answer the questions "
                f"yourself — the twenty-question path is unaffected by this failure.")
        print(_brief.render(proposed, brief_text))
        # #316: `--benchmark` is the one cache-keying value the AO never sees and no question
        # ever asks. If it is wrong, this run reads and writes another benchmark family's cached
        # episodes (the comment on ExperimentSpec below tells that story). The countersigner must
        # see it, so it renders here — labelled as the operator's own input, not a proposal.
        print(f"\n  FROM YOUR COMMAND LINE, NOT FROM THE AO: --benchmark {benchmark!r} "
              f"(default 'tau3').\n"
              f"  This keys the episode cache. If {benchmark!r} is not the harness family for "
              f"the data above, stop and re-run with the right --benchmark.")
        brief_meta = {"brief": brief_text,
                      "benchmark_flag": benchmark,
                      "reasoning": proposed.get("_reasoning", ""),
                      "unsure": proposed.get("_unsure", []),
                      "accepted_without_review": bool(args.yes)}
        if not args.yes:
            # THE COUNTERSIGNATURE. Not a formality: the AO just wrote most of an experiment's
            # definition from one paragraph, and the only party who knows what the paragraph meant
            # is the person who wrote it. `--yes` exists for scripted runs and is RECORDED, so a
            # later reader can tell a reviewed setup from an unreviewed one.
            # #317: the intake log was already written at propose time (BRIEF_INTAKE.log, when a
            # run dir was given). The refusal messages below must not claim otherwise — in this
            # codebase a message that is not literally true is the same defect class as #228/#239.
            _kept = (f" The AO's intake log at {rd_early / 'BRIEF_INTAKE.log'} remains for audit."
                     if rd_early and (rd_early / "BRIEF_INTAKE.log").exists() else "")
            try:
                said = input("\n  Accept this proposal? (yes / no) > ").strip().lower()
            except EOFError:
                raise SystemExit("\n  stdin is closed and --yes was not given; no experiment "
                                 "definition has been written." + _kept)
            if said not in ("y", "yes"):
                raise SystemExit("\n  Not accepted. No experiment definition has been written."
                                 + _kept + " Re-run with a sharper brief, or without --brief to "
                                 "answer the questions yourself.")
        for k, v in proposed.items():
            if not k.startswith("_") and k not in scripted:
                scripted[k] = v
        if rd_early:
            (rd_early / "BRIEF_PROPOSAL.json").write_text(json.dumps(
                {"brief": brief_text, "proposal": proposed}, indent=1))
            # #301: the setup is part of the conversation, and it is the part where the AO makes
            # the most consequential guesses. Recorded before any of it is verified.
            from src.orchestrator import conversation as _conv
            _conv.record(rd_early, speaker="operator", channel="intake", content=brief_text,
                         meta={"root_hint": args.root_hint, "source": args.brief})
            _conv.record(rd_early, speaker="ao", channel="intake",
                         content=_brief.render(proposed, brief_text),
                         meta={"accepted_without_review": bool(args.yes),
                               "unsure": ",".join(proposed.get("_unsure", []))})

    a = Answers(scripted)

    claim = build_claim(a)
    run_dir_early = Path(args.run_dir) if args.run_dir else None
    if run_dir_early:
        run_dir_early.mkdir(parents=True, exist_ok=True)
    claim, findings, clarifications = clarify_loop(
        claim, a, log_dir=run_dir_early,
        compose=_stub_compose if os.environ.get("EVOSKILL_CLI_NO_AO") == "1" else None)

    pool = [i for i in (claim.record_id(p, r) for p, r in claim.records()) if i]
    print(f"\n  The pool holds {len(pool)} task(s).")

    if args.split:
        # ADOPTED, NOT REBUILT. The counts and the seed come from the file, and the three
        # questions below are not asked: asking them and then ignoring the answers is worse than
        # not asking. See benchmark.adopt_split for why re-partitioning breaks the older run's
        # seal without anything reporting an error.
        split_out = Path(args.split).resolve()
        split = adopt_split(claim, split_out)
        n_train, n_val, n_test = (split["n"]["train"], split["n"]["val"], split["n"]["test"])
        seed = split.get("seed", -1)
        print(f"\n  ADOPTED the split at {split_out}\n"
              f"    train {n_train} | val {n_val} | test {n_test} SEALED   seed {seed}\n"
              f"    Every id checked against this benchmark's pool of {len(pool)}; the three "
              f"sets are disjoint.")
        run_dir = Path(args.run_dir or a.ask("run_dir",
                                             "\nWhere should this run's artifacts go?"))
    else:
        while True:
            n_train = a.ask_int("n_train", "\nHow many tasks in TRAIN?")
            n_val = a.ask_int("n_val", "How many in VAL?")
            n_test = a.ask_int("n_test",
                               "How many in TEST (sealed — never run during evolution)?")
            if n_train + n_val + n_test <= len(pool):
                break
            print(f"  That is {n_train + n_val + n_test} tasks but the pool holds {len(pool)}.")
            if scripted:
                raise SystemExit("scripted counts exceed the pool; nothing written")

        seed = a.ask_int("seed", "\nSplit seed? (any integer; it goes into the split file)")
        run_dir = Path(args.run_dir or a.ask("run_dir",
                                             "\nWhere should this run's artifacts go?"))
        split_out = Path(args.split_out or (run_dir / f"split_{n_train}{n_val}{n_test}.json"))
        split = build_split(claim, n_train=n_train, n_val=n_val, n_test=n_test,
                            seed=seed, out_path=split_out)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "BENCHMARK.json").write_text(json.dumps({
        "claim": claim.__dict__,
        "findings": [{"check": f.check, "ok": f.ok, "detail": f.detail} for f in findings],
        "conversation": a.transcript,
        # Gap 1. Every clarification round, verbatim: what failed, which field the AO chose to
        # re-ask and why, the question it composed, and what the user said. Without this the
        # setup records only its outcome, and "why was I asked that?" is unanswerable later.
        "clarifications": clarifications,
        # Gap 2. Empty when the twenty questions were answered by hand, which is itself the
        # record that no model proposed anything here.
        "brief": brief_meta,
    }, indent=1))
    # #301. Every clarification round as a two-sided turn, so the setup dialogue reads in the
    # same place as the epoch dialogue instead of living only inside a JSON field.
    from src.orchestrator import conversation as _conv
    for c in clarifications:
        _conv.record_once(run_dir, key=f"clarify:{c['round']}:{c['field']}", speaker="ao",
                          channel="clarify",
                          content=f"{c['question']}\n\n(asking about '{c['field']}' because: "
                                  f"{c.get('why', '')})",
                          meta={"round": c["round"], "check": c["check"], "field": c["field"]})
        _conv.record_once(run_dir, key=f"clarify-a:{c['round']}:{c['field']}", speaker="operator",
                          channel="clarify", content=str(c["answer"]),
                          meta={"round": c["round"], "field": c["field"]})

    # ── freeze the spec, so the loop can drive this without a human re-typing anything ──
    # This is the join between the front door and the engine. Everything above was about
    # WHAT the benchmark is; the fields below are HOW it will be measured, and they are asked
    # rather than assumed because each one changes what a number means.
    from src.orchestrator.spec import ExperimentSpec
    inst = {k: a.ask(k, q, default=d) for k, q, d in (
        ("solver_model", "\nWhich model is the SOLVER (the agent being improved)?", ""),
        ("evolver_model", "Which model writes the skills (the EVOLVER)?", "opus[1m]"),
        ("user_sim_model", "Which model simulates the user? ('none' if this benchmark has no simulated user)", "none"),
        ("judge_model", "Which model judges? ('none' if scoring is entirely mechanical)", "none"),
    )}
    n_samples = a.ask_int("n_samples", "How many samples per task?")
    n_epochs = a.ask_int("n_epochs", "How many epochs?")
    ceiling = a.ask_int("cost_ceiling_usd", "Cost ceiling for the whole run, in whole dollars?")

    # OPERATOR-SUPPLIED INSTRUMENT PINS. Validated against the dataclass by NAME before they are
    # passed, for the clarify.py reason: an unvalidated string reaching a constructor kwarg is how
    # a typo becomes a silently-ignored pin. `test_sealed` is refused outright — a flag that
    # unseals the test set must never be settable from a JSON blob on a command line.
    extra = {}
    if args.spec_extra:
        import dataclasses as _dc
        try:
            extra = json.loads(args.spec_extra if args.spec_extra.strip().startswith("{")
                               else Path(args.spec_extra).read_text())
        except (json.JSONDecodeError, OSError) as e:
            raise SystemExit(f"--spec-extra is not readable JSON: {e}")
        if not isinstance(extra, dict):
            raise SystemExit("--spec-extra must be a JSON object")
        known = {f.name for f in _dc.fields(ExperimentSpec)}
        bad = sorted(set(extra) - known)
        if bad:
            raise SystemExit(f"--spec-extra names field(s) ExperimentSpec does not have: {bad}\n"
                             f"  known: {sorted(known)}")
        clash = sorted(set(extra) & {"name", "split_path", "split_sha256", "n_train", "n_val",
                                     "n_test", "test_sealed", "solver_model", "evolver_model",
                                     "user_sim_model", "judge_model", "n_samples", "n_epochs",
                                     "cost_ceiling_usd"})
        if clash:
            raise SystemExit(
                f"--spec-extra tries to set field(s) the setup has already determined: {clash}. "
                f"Those come from the split and from your answers; two sources for one field is "
                f"how they come to disagree.")
        # gate/headroom are nested dataclasses, not dicts. Rebuild them rather than letting a
        # plain dict through: asdict() would still serialise it and config_sha256 would still
        # compute, so the spec would look frozen and correct while every gate lookup did
        # `dict.alpha` and raised at the one moment the gate is consulted.
        from src.orchestrator.spec import GateSpec, HeadroomGate
        for k, cls in (("gate", GateSpec), ("headroom", HeadroomGate)):
            if isinstance(extra.get(k), dict):
                try:
                    extra[k] = cls(**{kk: (tuple(vv) if isinstance(vv, list) else vv)
                                      for kk, vv in extra[k].items()})
                except TypeError as e:
                    raise SystemExit(f"--spec-extra {k!r}: {e}")
        extra = {k: (tuple(v) if isinstance(v, list) else v) for k, v in extra.items()}

    # #316: the strongest wrongness check available without inventing a registry — this repo's
    # own frozen specs. If this DOMAIN was ever frozen under a DIFFERENT benchmark family, one of
    # the two runs is reading the other's episodes out of cache, so this refuses rather than
    # warns. It covers both the --brief path and the interactive path, because `claim.domain` is
    # settled either way by this point.
    _bench_conflicts = {}
    for _prior in sorted(Path("outputs").glob("*/SPEC.json")):
        try:
            _ps = json.loads(_prior.read_text()).get("spec", {})
        except (json.JSONDecodeError, OSError):
            continue
        if _ps.get("domain") == claim.domain and _ps.get("benchmark") not in (None, benchmark):
            _bench_conflicts[str(_prior)] = _ps.get("benchmark")
    if _bench_conflicts:
        raise SystemExit(
            f"\n--benchmark {benchmark!r} CONFLICTS with this repo's own record: domain "
            f"{claim.domain!r} is already frozen under benchmark "
            f"{sorted(set(_bench_conflicts.values()))} in:\n  "
            + "\n  ".join(sorted(_bench_conflicts)) + "\n"
            "  `benchmark` is in the cache tag; the same domain under two families makes one "
            "benchmark read the other's cached episodes (#148's shape). Re-run with the "
            "matching --benchmark, or rename the domain if this is genuinely a new family.\n"
            "  The split IS written; no spec has been frozen.")

    spec = ExperimentSpec(
        # BENCHMARK COMES FROM THE SETUP, NOT FROM A LITERAL. It was hardcoded "tau3", which was
        # invisible while tau3 was the only benchmark and is a silent contamination sink the
        # moment it is not: `benchmark` is in ExperimentSpec._INSTRUMENT, so it is part of the
        # CACHE TAG. An SSB v2 run frozen through this front door would have written its episodes
        # into a tau3-tagged bucket and read tau3's back out — #148's shape exactly, arriving by
        # a different road. Default stays "tau3" so every existing invocation is unchanged.
        name=claim.name.replace(" ", "_"), benchmark=benchmark, domain=claim.domain,
        split_path=str(split_out), split_sha256=split["file_sha256"],
        n_train=n_train, n_val=n_val, n_test=n_test, test_sealed=True,
        n_samples=n_samples, solver_model=inst["solver_model"],
        user_sim_model=inst["user_sim_model"], judge_model=inst["judge_model"],
        evolver_model=inst["evolver_model"], n_epochs=n_epochs,
        cost_ceiling_usd=float(ceiling), **extra)
    errs = spec.validate()
    if errs:
        raise SystemExit("\nI cannot freeze this spec:\n  - " + "\n  - ".join(errs) +
                         "\nThe split IS written; fix the above and re-run to freeze.")
    spec_path = spec.freeze(run_dir)

    print(f"\n{'=' * 76}")
    print(f"  split      {split_out}  (file sha {split['file_sha256'][:12]})")
    print(f"  spec       {spec_path}  (config {spec.config_sha256[:12]}, "
          f"cache tag {spec.cache_tag})")
    print(f"             train {len(split['train'])} | val {len(split['val'])} | "
          f"test {len(split['test'])} SEALED   seed {seed}")
    print(f"  claim      {run_dir / 'BENCHMARK.json'} — every answer you gave, and what I checked")
    print("\n  Verified against your files, not just recorded. Drive it with:")
    print(f"    python3 -m src.orchestrator.loop --plan --run-dir {run_dir} --epoch 1")
    print("=" * 76)

    receipt = (f"Setup frozen.\n"
               f"  split {split_out} (sha {split['file_sha256'][:12]}) — "
               f"train {len(split['train'])} / val {len(split['val'])} / "
               f"test {len(split['test'])} SEALED, seed {seed}\n"
               f"  spec  {spec_path} (config {spec.config_sha256[:12]}, "
               f"cache tag {spec.cache_tag})\n"
               f"  {spec.n_epochs} epoch(s), {spec.n_samples} sample(s)/task, solver "
               f"{spec.solver_model}, evolver {spec.evolver_model}, "
               f"ceiling ${spec.cost_ceiling_usd:.0f}")
    _conv.record_once(run_dir, key="intake:frozen", speaker="ao", channel="intake",
                      content=receipt, meta={"spec_sha": spec.config_sha256})
    if args.phoenix_project:
        try:
            n_x, n_t = _conv.emit(run_dir, args.phoenix_project)
            print(f"\n  phoenix    {n_x} exchange(s) / {n_t} turn(s) -> "
                  f"{args.phoenix_project}   (Sessions tab, session {run_dir.name!r})")
        except Exception as e:                                   # noqa: BLE001
            print(f"\n  phoenix    emission FAILED ({type(e).__name__}: {e}) — the transcript on "
                  f"disk is unaffected: {_conv.path_for(run_dir)}", file=sys.stderr)
    return 0


def selftest() -> int:
    """A toy benchmark on disk, driven end-to-end by a scripted user — including a WRONG first
    answer, so the clarification loop is exercised rather than assumed."""
    import shutil
    import tempfile
    td = Path(tempfile.mkdtemp())
    fails = 0
    try:
        # toy dataset: 12 tasks as records inside one JSON array
        (td / "data").mkdir()
        (td / "data" / "tasks.json").write_text(json.dumps(
            [{"id": f"t{i:02d}", "question": f"q{i}", "answer": str(i * 2)} for i in range(12)]))
        (td / "data" / "score.py").write_text("def score(got, want):\n    return got == want\n")

        # The scripted user gets the SHAPE WRONG first — exactly as a real one would.
        script = {
            "name": "toy", "domain": "toy",
            "description": "answer a question with a number",
            "root": str(td / "data"), "task_glob": "*.json", "records_in": "",
            "id_from": "json:id", "input_glob": "", "answer_glob": "",
            "scoring_desc": "exact string match", "scoring_script": "score.py",
            "records_in#1": "array",                       # the correction
            "n_train": 5, "n_val": 4, "n_test": 3, "seed": 7,
            "solver_model": "toy/solver", "evolver_model": "toy/evolver",
            "user_sim_model": "none", "judge_model": "none", "n_samples": 3, "n_epochs": 2,
            "cost_ceiling_usd": 5,
        }
        sp = td / "answers.json"
        sp.write_text(json.dumps(script))
        rd = td / "run"
        rc = _run_scripted(sp, rd, td / "split.json")
        ok = rc == 0
        print(f"  [{'ok' if ok else 'BAD'}] scripted run completed (rc={rc})")
        fails += 0 if ok else 1

        s = json.loads((td / "split.json").read_text())
        for label, cond, detail in (
            ("split has the asked-for sizes",
             (len(s["train"]), len(s["val"]), len(s["test"])) == (5, 4, 3),
             f"{len(s['train'])}/{len(s['val'])}/{len(s['test'])}"),
            ("the three sets are disjoint",
             not (set(s["train"]) & set(s["val"])) and not (set(s["train"]) & set(s["test"]))
             and not (set(s["val"]) & set(s["test"])), "no overlap"),
            ("every id came from the pool",
             all(i.startswith("t") for i in s["train"] + s["val"] + s["test"]), "all t*"),
            ("the split file is NOT claiming a hash of itself",
             "self_sha256" not in s, "digest is held by the spec, which hashes these bytes"),
            ("a spec was frozen, hashed, and carries the split digest",
             (rd / "SPEC.json").is_file()
             and json.loads((rd / "SPEC.json").read_text())["spec"]["split_sha256"]
             == __import__("hashlib").sha256(
                 (rd.parent / "split.json").read_bytes()).hexdigest(),
             "SPEC.json binds the split it was built from"),
            ("the clarification round was actually needed",
             any(t["key"] == "records_in#1" for t in
                 json.loads((rd / "BENCHMARK.json").read_text())["conversation"]),
             "records_in was re-asked"),
        ):
            print(f"  [{'ok' if cond else 'BAD'}] {label}\n        {detail}")
            fails += 0 if cond else 1

        # asking for more than exists must REFUSE, not shrink
        try:
            build_split(BenchmarkClaim(name="toy", root=str(td / "data"),
                                       task_glob="tasks.json", records_in="array",
                                       id_from="json:id"),
                        n_train=10, n_val=10, n_test=10, seed=1, out_path=td / "x.json")
            print("  [BAD] over-sized split -> should have refused")
            fails += 1
        except ValueError as e:
            print(f"  [ok] over-sized split refuses\n        {str(e)[:90]}")
    finally:
        shutil.rmtree(td, ignore_errors=True)
    print(f"\n{'ALL PASS' if not fails else str(fails) + ' FAILED'}")
    return 0 if not fails else 1


def _run_scripted(script_path: Path, run_dir: Path, split_out: Path) -> int:
    import subprocess
    env = dict(os.environ, EVOSKILL_CLI_NO_AO="1")   # selftest is offline and deterministic
    r = subprocess.run([sys.executable, "-m", "src.orchestrator.cli", "new",
                        "--script", str(script_path), "--run-dir", str(run_dir),
                        "--split-out", str(split_out)],
                       cwd=REPO, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print(r.stdout[-1500:], r.stderr[-800:])
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
