#!/usr/bin/env python3
"""Layer 3 of the orchestrator — the DETERMINISTIC LOOP.

WHAT THIS IS, AND THE ONE THING IT MUST NOT BE

It executes a frozen ExperimentSpec: base leg, then N epochs of
(build evidence -> evolve -> gate the artifact -> promote -> roll out -> gate the result).
It decides NOTHING. Every threshold, metric and stopping rule comes from the spec, which
was frozen and hashed before any leg ran. The loop's job is to do the same thing every
time and to refuse when reality disagrees with the plan.

The thing it must not be is a happy-path DAG driver. A step's own report can be green while
its artifact is wrong:

    evolver "wrote 5 skills"   -> the loader reads 1; four bodies silently discarded
    evolver run "succeeded"    -> it read the comparator arm's answers
    process "launched"         -> dead in 200ms on `400 No models provided`
    split "built, 45 tasks"    -> most ids do not exist in the harness's pool

A driver that advances on each step's own report ships all four. So every step here has a
`verify` that re-derives its answer from the artifact THROUGH THE ARTIFACT'S REAL CONSUMER,
and the loop advances on the verify, never on the exit code.

THREE RULES

1. HALT, DO NOT RETRY. A failed verify means the world is not what the plan assumed. Retrying
   an evolver whose output was contaminated just produces contaminated output again — the
   correct response to a contaminated evolver is to restart the epoch, and to lost skills to
   fix the prompt. Both are human calls. `HALT` is the only failure mode; there is no backoff.

2. ISOLATION IS FILESYSTEM-LEVEL, NOT TEXTUAL. A subagent can walk past a regex hook by
   splitting a quoted literal so the word never appears whole. Any guard that
   pattern-matches command strings is a tripwire that makes accidents loud, not a boundary.
   `EpochPaths.grants()` returns the ONLY directories an evolver may see, and the peer arms'
   skill directories are not among them. The hook stays armed as a tripwire; it is not the
   mechanism. With RUNTIME `evolver_evidence: "all_epochs"` that set is THIS run's
   digests_e1..eK (`EpochPaths.evidence_dirs()`, an explicit bounded range, never a glob) plus
   staging and the oracle; without it, the current epoch's digests only.

3. THE RUN IS AUTOMATIC. Nothing waits for a signature: every check is a verify the loop runs
   itself, and a frozen file that changes still halts the run. A Step may still be marked
   operator_gate (advance() then raises OperatorGate, exit 3 under --drive); the epoch program
   marks none.

WHAT THIS MODULE DOES NOT DO
  - It does not compose evolver prompts. The loop takes a prompt path and verifies the prompt's
    numbers re-derive from the result JSONs, refusing a prompt whose headline figures
    disagree with the data behind them.
    ONE DECLARED EXCEPTION: with RUNTIME prompt_mode="template" the loop RENDERS
    the prompt deterministically (src/orchestrator/evolver_template.py: fixed text + the setup
    AO's frozen one-paragraph template + slots filled from the artifacts). It composes no
    prose of its own and calls no model; every prompt check still runs on the rendered file.
    See Ctx.prompt_mode and _template_steps().
  - It does not judge skill quality. Only infra failures halt a run; bad reasoning from sound
    evidence is the optimizer's output, and filtering on taste makes the operator a hidden
    part of the optimizer.

USAGE
    python3 -m src.orchestrator.loop --plan     --run-dir <run> --epoch N
    python3 -m src.orchestrator.loop --step     --run-dir <run> --epoch N [--drive --until-epoch M]
    python3 -m src.orchestrator.loop --selftest
    python3 -m src.orchestrator.loop --val-check --step --run-dir R --epoch N   (RUNTIME pipeline_val only)

EXIT CODES, and 2 is the one with a contract:
    0   the step completed, or every step already verified and there was nothing to do
    1   HALT — something disagrees with the frozen plan. A person decides what to fix.
    2   BLOCKED — the run needs an ANSWER from you, and ONLY that. A caller may key on this to
        tell "answer me" from "go fix something" without parsing prose, which is the entire
        reason the code is distinct; nothing else in this module may return it.
   64   usage error (EX_USAGE) — missing --run-dir/--epoch.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import re
import subprocess
from subprocess import DEVNULL
import shutil
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# THERE IS DELIBERATELY NO STATE FILE. Every step's position is re-derived from the artifacts on disk each time it is asked, so there is no
# second record of where the run is that could disagree with the run itself — and a stale
# checkpoint claiming a step was done is precisely the failure this module exists to refuse.
# It also makes the loop resumable for free: `--step` after any halt, crash or reboot simply
# re-reads the world. If a state file is ever added, it must be a CACHE of the artifacts, never
# an authority over them.


class Halt(Exception):
    """A verify failed. The run stops and a human is told what disagreed with the plan."""


class OperatorGate(Halt):
    """A step that is deliberately not automatable has been reached — a human must act.

    A SUBCLASS of Halt, so every existing `except Halt` keeps behaving exactly as before and the
    default single-step path is byte-identical. It exists so `--drive` can report "your turn"
    with its own exit code instead of "something is broken": a caller that cannot tell a gate
    from a defect will eventually route one through the handler for the other, and the handler
    for a defect is the one that says "go fix something". The gate is not a defect. It is the
    design working.
    """


class Blocked(BaseException):
    """A step needs an ANSWER from the user before the run can continue.

    IT INHERITS BaseException, NOT Exception (and therefore NOT Halt). As an Exception it would
    be swallowable by any `except Exception` between ask() and _safe_verify's re-raise. A merely
    defensive verify --
        try: a = ask(ctx, "basis", "Which basis?")
        except Exception: a = "assume DB-state"
    -- would turn "I need a human answer" into a silent default AND leave BLOCKED_e{N}.json listing
    the question as outstanding for a step that had already passed: the artifact and the
    behaviour disagreeing. Several verifies in this file use bare `except Exception`.
    BaseException is the standard idiom for a control-flow signal that must not be caught by
    ordinary error handling -- the same reason KeyboardInterrupt and SystemExit sit there.
    Callers that mean to handle a block must name it.

    Distinct from a plain Halt, and the distinction is the point. A Halt says "something
    disagreed with the plan" — a defect, and the operator's job is to go fix it. A Blocked says
    "I cannot tell whether this is wrong or whether you meant something I did not parse, and
    guessing would make me a hidden author of your experiment."

    Without this channel every step could only refuse, and a person would have to turn the
    refusal into a question by hand.

    IT BLOCKS rather than queueing the question and carrying on with independent work: every
    artifact downstream of an unanswered question inherits an assumption nobody confirmed.
    """

    def __init__(self, question: str, *, step: str = "", detail: str = "", qid: str = ""):
        self.question, self.step, self.detail, self.qid = question, step, detail, qid
        super().__init__(f"BLOCKED — {step}: {question}" if step else f"BLOCKED — {question}")



# ── the answer channel ───────────────────────────────────────────────────────────────────
def operator_dir(run_dir) -> "Path":
    """The one place the operator channel lives. See answers_path for why it is a SIBLING."""
    from pathlib import Path
    rd = Path(run_dir)
    return rd.parent / f"{rd.name}.operator"


def answers_path(run_dir, epoch) -> "Path":
    """Where the user's answers live. One file per epoch, so an answer cannot silently carry
    across an epoch boundary into a question it was never given for.

    OUTSIDE THE RUN DIRECTORY, in a sibling `<run>.operator/`. Defence in depth on top of the
    agents' write containment, not a replacement for it. The sandbox denies an agent write access
    to the run dir, and that is the primary control -- but it is an ALLOWLIST, and
    one forgotten grant re-opens the boundary silently. Putting the answer channel somewhere no
    agent job ever declares as a write target means a future grant mistake cannot reach it.

    Sibling rather than a subdirectory, because a subdirectory of the run dir inherits any
    future subpath grant on its parent (a log's parent grant can BE the run dir). A sibling has
    no such parent relationship.
    """
    return operator_dir(run_dir) / f"ANSWERS_e{epoch}.json"


def blocked_path(run_dir, epoch) -> "Path":
    """The outstanding-question record. Sibling to the run dir alongside ANSWERS, so the
    whole operator channel lives in one place that no agent can write."""
    return operator_dir(run_dir) / f"BLOCKED_e{epoch}.json"


# Code points that are neither whitespace by str.isspace() nor visible. An "answer" made only of
# these is not an answer -- see _is_blank.
_INVISIBLE = "​‌‍⁠﻿­᠎⠀ㅤﾠᅟᅠ"


def _is_blank(s: str) -> bool:
    """`s.strip()` removes only characters where str.isspace() is True, so an answer consisting
    entirely of ZERO-WIDTH SPACE, a BOM, WORD JOINER, SOFT HYPHEN, BRAILLE BLANK or a Hangul filler
    would pass an emptiness check and be returned verbatim as "a genuine non-empty answer". The
    realistic trigger is not an attack: it is a copy-paste out of a terminal or a web
    page that carried an invisible character, after which the run proceeds on an answer no human
    can see -- the exact "the artifact and the behaviour disagree" shape.
    """
    import unicodedata
    for ch in s:
        if ch.isspace() or ch in _INVISIBLE:
            continue
        if unicodedata.category(ch) in ("Cf", "Cc"):        # format / control
            continue
        return False
    return True


def _assert_epoch_consistent(ctx) -> None:
    """answers_path() resolves ctx.epoch LIVE on every ask(), while Ctx.__post_init__ snapshots
    `self.paths = EpochPaths(run_dir, epoch)` ONCE. Unpinned, a verify that reassigned ctx.epoch
    could be answered out of another epoch's ANSWERS file while every other artifact path in the
    same run still referred to the original epoch. This pins the two together, so the per-epoch
    answer firewall ("an answer cannot silently carry across an epoch boundary") holds.
    """
    paths = getattr(ctx, "paths", None)
    pe = getattr(paths, "epoch", None)
    if pe is not None and pe != ctx.epoch:
        raise Halt(f"ctx.epoch is {ctx.epoch} but every artifact path in this context was frozen "
                   f"at epoch {pe}. The per-epoch answer firewall cannot be enforced against an "
                   f"epoch that moves under it; build a new Ctx rather than reassigning .epoch.")


def ask(ctx, qid: str, question: str) -> str:
    """Return the user's answer to `qid`, or BLOCK the run until there is one.

    A verify calls this instead of guessing. Contract, and every clause is load-bearing:

      * ABSENT answer  -> Blocked. NOT a default, NOT a skip, NOT a pass ("that fallback IS the
        defect", applied to human input): a step that proceeds on an unanswered question has
        invented the answer.
      * PRESENT but empty/whitespace -> Blocked, with the same question. An empty string is not
        an answer, and treating it as one is how "the user declined to say" becomes "the user
        said nothing matters".
      * PRESENT and non-empty -> returned verbatim, never parsed or coerced here. The caller
        decides what it means, because only the caller knows what it asked.
      * PRESENT, non-empty, but given for a DIFFERENT question under this qid -> Blocked.

    The question is also written to BLOCKED_e{N}.json so it survives the process that asked it:
    a record written only on success is not a record.
    """
    import json as _json
    _assert_epoch_consistent(ctx)
    ap = answers_path(ctx.run_dir, ctx.epoch)
    if ap.is_file():
        try:
            doc = _json.loads(ap.read_text())
        except Exception as e:                                  # noqa: BLE001
            # A malformed answers file BLOCKS. It does not fall back to "no answers" — that would
            # turn a typo in the file into a silent re-ask, and on a long run the operator would
            # never learn the file had been unreadable the whole time. It RECORDS the question
            # too, like every other block: the cause here is an operator typo, the block most
            # likely to be hit repeatedly.
            note = _try_record(ctx, qid, question)
            raise Blocked(f"{ap.name} is not readable JSON ({type(e).__name__}: {e}). "
                          f"Fix the file; I will not proceed on the assumption it is empty."
                          + note, step="answer channel", qid=qid)
        if doc is not None and not isinstance(doc, dict):
            note = _try_record(ctx, qid, question)
            raise Blocked(f"{ap.name} holds a {type(doc).__name__}, not a JSON object. It must map "
                          f'question ids to answers, e.g. {{"{qid}": "your answer"}}.' + note,
                          step="answer channel", qid=qid)
        got = (doc or {}).get(qid)
        if isinstance(got, str) and not _is_blank(got):
            # THE ANSWER IS BOUND TO THE QUESTION, NOT ONLY TO THE KEY. A check of `.get(qid)` alone
            # would let a step asking a DIFFERENT question under the same qid be silently satisfied
            # by the earlier answer -- "a step that proceeds on an unanswered question has invented
            # the answer", arrived at by a different road. An answer must not outlive an edit to the
            # question it was given for.
            prev = _recorded_question(ctx, qid)
            if prev is not None and prev != question:
                _try_record(ctx, qid, question)
                raise Blocked(
                    f"the question under {qid!r} has CHANGED since that answer was written, so the "
                    f"answer on disk was given for something else.\n"
                    f"      previously asked: {prev}\n"
                    f"      now asking:       {question}\n"
                    f"    Replace the value under {qid!r} with an answer to the question above.",
                    step=getattr(ctx, "_asking_step", ""), qid=qid)
            # Establish (or refresh) the binding, and stop presenting this question as outstanding
            # (otherwise BLOCKED_e{N}.json would list every question ever asked as still open).
            _try_record(ctx, qid, question, answered=True)
            # The transcript records this exchange: ANSWERS_e{N}.json is a
            # dict, so it records the answer but destroys the ORDER and the pairing — re-reading
            # it later cannot tell you which question came first or which answer unblocked what.
            # Recorded on the CONSUMING read (here), not on the write, because that is the moment
            # the answer actually enters the run. `_conv_once` keeps a re-run from duplicating it.
            _conv_once(ctx, f"answer:{qid}:{question}", speaker="operator", channel="answer",
                       content=got, meta={"qid": qid, "question": question})
            return got
    # RECORD, THEN BLOCK -- and block even if the record could not be written. An unguarded write
    # (read-only run dir, full disk, a directory where the file belongs) would raise OSError
    # INSTEAD of Blocked; _safe_verify would turn that into an ordinary failed verify, and advance()
    # would fall through and EXECUTE the step's runner -- dispatching a paid evolver, or promoting
    # skills, on a question no human ever saw. Losing the audit record is tolerable; proceeding
    # past an unanswered question is not.
    raise Blocked(question + _try_record(ctx, qid, question),
                  step=getattr(ctx, "_asking_step", ""), qid=qid)


def _conv_once(ctx, key: str, *, speaker: str, channel: str, content: str, meta=None) -> None:
    """Append one operator<->AO turn to the run's transcript, at most once per `key`.

    READ-ONLY CONTEXTS DO NOT WRITE, same rule as `_record_block`: `--plan` promises to change
    nothing, and a transcript is a file.
    """
    if getattr(ctx, "readonly", False):
        return
    from src.orchestrator import conversation
    conversation.record_once(ctx.run_dir, key=key, speaker=speaker, channel=channel,
                             content=content, epoch=ctx.epoch, meta=meta)


def _try_record(ctx, qid: str, question: str, *, answered: bool = False) -> str:
    """_record_block, but a write failure is a NOTE rather than an exception (see ask())."""
    try:
        _record_block(ctx, qid, question, answered=answered)
        if not answered:
            # The AO's side of the ask/answer channel. Only on the ASKING record: `answered=True`
            # is a refresh of an existing binding, not a new utterance, and recording it would
            # put the question into the transcript a second time AFTER the answer it precedes.
            _conv_once(ctx, f"question:{qid}:{question}", speaker="ao", channel="ask",
                       content=question,
                       meta={"qid": qid, "step": getattr(ctx, "_asking_step", "")})
        return ""
    except Exception as e:                                      # noqa: BLE001
        return (f" [WARNING: could not write {blocked_path(ctx.run_dir, ctx.epoch).name} "
                f"({type(e).__name__}: {e}) -- the block still stands, but this question is "
                f"not recorded on disk]")


def _read_blocked(ctx) -> dict:
    """The outstanding-question record, or {} — never raises. Corruption is PRESERVED, not
    silently replaced; see _record_block."""
    import json as _json
    bp = blocked_path(ctx.run_dir, ctx.epoch)
    if not bp.is_file():
        return {}
    try:
        cur = _json.loads(bp.read_text())
    except Exception:                                           # noqa: BLE001
        return {}
    return cur if isinstance(cur, dict) else {}


def _recorded_question(ctx, qid: str):
    """The question this qid was last asked, or None if there is no record of one."""
    e = _read_blocked(ctx).get(qid)
    return e.get("question") if isinstance(e, dict) else None


def outstanding_questions(ctx) -> dict:
    """Questions still awaiting an answer. An ANSWERED entry is kept (it carries the
    question<->answer binding ask() checks) but is no longer outstanding."""
    return {k: v for k, v in _read_blocked(ctx).items()
            if isinstance(v, dict) and not v.get("answered")}


def _record_block(ctx, qid: str, question: str, *, answered: bool = False) -> None:
    """Append the outstanding question to BLOCKED_e{N}.json. Idempotent on qid.

    READ-ONLY CONTEXTS DO NOT WRITE. `--plan` promises to "report every verify and change
    nothing", and once a verify can ask a question, planning would otherwise leave files behind.

    A CORRUPT RECORD IS PRESERVED, NEVER DISCARDED. Overwriting an unreadable file would replace a
    truncated record -- e.g. from a process killed mid-write, the exact durability scenario this
    file exists for -- with one holding only the newest question, and every other outstanding
    question would vanish with no error. It is moved aside to a .corrupt file instead.
    """
    import json as _json
    if getattr(ctx, "readonly", False):
        return
    bp = blocked_path(ctx.run_dir, ctx.epoch)
    cur, salvage = {}, None
    if bp.is_file():
        try:
            cur = _json.loads(bp.read_text())
            if not isinstance(cur, dict):
                raise ValueError(f"holds a {type(cur).__name__}, not an object")
        except Exception as e:                                  # noqa: BLE001
            cur, salvage = {}, f"{type(e).__name__}: {e}"
    if salvage is not None:
        keep = bp.with_name(bp.name + ".corrupt")
        n = 0
        while keep.exists():
            n += 1
            keep = bp.with_name(f"{bp.name}.corrupt.{n}")
        try:
            bp.replace(keep)
        except OSError:
            pass
    prev = cur.get(qid) if isinstance(cur.get(qid), dict) else {}
    cur[qid] = {
        "question": question,
        "step": getattr(ctx, "_asking_step", "") or prev.get("step", ""),
        "answered": bool(answered),
        "answer_file": str(answers_path(ctx.run_dir, ctx.epoch)),
        "how_to_answer": f'add {{"{qid}": "your answer"}} to '
                         f"{answers_path(ctx.run_dir, ctx.epoch)}, then re-run the step",
    }
    if salvage is not None:
        cur["_SALVAGE_NOTE"] = {
            "question": f"the previous {bp.name} was unreadable ({salvage}) and was MOVED ASIDE "
                        f"rather than overwritten; any questions it held are in the .corrupt file "
                        f"next to this one",
            "answered": True, "step": "", "answer_file": "", "how_to_answer": "",
        }
    bp.parent.mkdir(parents=True, exist_ok=True)
    # ATOMIC: a plain write_text lets a concurrent reader
    # -- the operator tailing the file, or a second `--plan` -- observe a truncated record and
    # conclude the outstanding questions are gone. Unique temp name per caller, then os.replace.
    import uuid as _uuid
    _tmp = bp.parent / f".{bp.name}.{os.getpid()}.{_uuid.uuid4().hex[:8]}.rw"
    _tmp.write_text(_json.dumps(cur, indent=1))   # may raise; _try_record guards the call
    os.replace(str(_tmp), str(bp))

# ── paths + grants ───────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class EpochPaths:
    run_dir: Path
    epoch: int

    @property
    def staging(self) -> Path:
        return self.run_dir / f"skills_v{self.epoch}_staging"

    @property
    def promoted(self) -> Path:
        return self.run_dir / f"skills_v{self.epoch}"

    @property
    def digests(self) -> Path:
        return self.run_dir / f"digests_e{self.epoch}_train"

    @property
    def evo(self) -> Path:
        """The material the epoch-N evolver reads = epoch N-1's rollouts,
        <run>/evo_e{N-1}/ (evo_layout: traces/, solver_workspaces/, scores.json; N = 1 -> the base leg)."""
        from src.orchestrator import evo_layout as _el
        return _el.epoch_dir(self.run_dir, self.epoch - 1)

    def evo_dirs(self) -> list[Path]:
        """evo_e0 .. evo_e{N-1}: every epoch's rollouts up to this one (the cumulative container view)."""
        return [EpochPaths(self.run_dir, j).evo.resolve() for j in range(1, self.epoch + 1)]

    @property
    def prompt(self) -> Path:
        return self.run_dir / f"ITER{self.epoch}_PROMPT.md"

    def leg(self, key: str) -> Path:
        return self.run_dir / f"{self.run_dir.name}_v{self.epoch}_{key}.json"

    def evidence_dirs(self) -> list[Path]:
        """Every epoch's train evidence up to and including this one: digests_e1..e{epoch}.

        RUNTIME `evolver_evidence: "all_epochs"` grants exactly this list (the epoch-K evolver
        reads the solver rollouts of ALL earlier epochs, stored on disk and read on demand). An EXPLICIT BOUNDED RANGE, NEVER A GLOB, and never derived from
        spec.name / shared_dir / the oracle's or base leg's parent:
          * a glob would grant stale siblings (digests_e2_train.062721.bak, *.as_given,
            digests_probe_*) and, when an earlier epoch is re-run after a later surface exists,
            digests_e{K+1}_train — rollouts of the very version being evolved;
          * spec.name can name ANOTHER run (nothing checks it for uniqueness), whose digests
            carry that arm's skills.
        Resolved, because the launcher compares realpaths and the evolver's cwd is not the repo.
        """
        return [EpochPaths(self.run_dir, j).digests.resolve() for j in range(1, self.epoch + 1)]

    def grants(self, oracle: Path, all_epochs: bool = False, template_mode: bool = False) -> list[Path]:
        """The COMPLETE set of directories the evolver may see. This list, not a regex
        over command strings, is the isolation boundary. Anything absent here is absent from
        the filesystem the evolver runs against — in particular every peer arm's skills_v*,
        which live under a different run dir and are therefore simply not granted.

        Note what is NOT here: the run dir itself. Granting the parent would hand over every
        sibling epoch's promoted skills and every peer arm reachable by `cd ..`: a digests grant
        inside a comparator's run dir puts its answers one `cd ..` from a legitimate path.

        `all_epochs` (RUNTIME evolver_evidence) swaps the one digest dir for evidence_dirs().
        Earlier epochs' digests of THIS run are legitimate evidence in that mode; its own
        skills_v* are still not granted (the producing library travels inside each digest dir).

        `template_mode` (RUNTIME prompt_mode=template) drops the oracle: the rendered prompt names no
        oracle README and the Daytona upload carries none (daytona_evolver: no oracle row in template
        mode), so a grant would hand the evolver a dir its prompt never describes.
        """
        ev = self.evidence_dirs() if all_epochs else [self.digests]
        return [*ev, self.staging] if template_mode else [*ev, self.staging, oracle]


# ── steps ────────────────────────────────────────────────────────────────────────────────
@dataclass
class Step:
    """One unit of work. `run` may be None for steps a human or another process performs;
    the loop then only verifies. `verify` MUST re-derive from the artifact, never read a
    status field, and must be able to fail on a reachable input."""
    name: str
    verify: object                      # callable(ctx) -> (ok: bool, detail: str)
    run: object = None                  # callable(ctx) -> None
    defect: str = ""                    # the failure class this step's verify catches
    operator_gate: bool = False         # a person must act; never auto-satisfied (no epoch step sets it)


@dataclass
class Ctx:
    run_dir: Path
    epoch: int
    spec: object                        # ExperimentSpec
    oracle: Path
    # The per-sample cache dirs the digest builder reads. Explicit rather than derived: the
    # dir name encodes domain, variant, cache tag, leg token and sample index, and rebuilding
    # that string here would be a second source of truth for a name that must have exactly one
    # (the leg runner owns it).
    digest_caches: tuple = ()
    # Runtime plumbing, NOT experiment identity — deliberately on Ctx and not on the frozen
    # ExperimentSpec, whose hash defines the cache tag. Which launcher script drives the
    # rollouts, and how long to wait for one, cannot change what an epoch measures.
    leg_script: str = "scripts/ssbv2_leg_cc.sh"
    shared_dir: str = ""            # split/oracle/base live here; "" means same as run_dir
    # The no-skills comparator. Declared, not derived: a base leg need not follow the loop's naming
    # convention (`<run>_base_train.json`, not `<run>_v0_train.json`), and a guessed name would make
    # the epoch-1 prompt check vacuous. "" means undeclared, and the epoch-1 prompt verify FAILS
    # rather than passes.
    base_leg: str = ""
    # The VAL comparator, declared for the same reason base_leg is. Deriving it (substituting
    # "_base_" for "_vN_" in the skill leg's filename) resolves only inside a self-contained run dir,
    # not when the base legs live in another run's directory; a fallback to base_leg -- the TRAIN
    # base -- would compare the VAL skill leg against it and fail "different task sets" on a good
    # run. A guard that fails spuriously gets disabled, so it is worse than none. Declared, never
    # guessed; "" means val pairing is reported UNMEASURED rather than checked against the wrong file.
    base_leg_val: str = ""
    # The HELD-OUT comparator (RUNTIME `heldout_base_leg`): the no-skill leg on the run's
    # split's TEST key, declared for the same reason as the two above. It may be a measured leg or a
    # record derive_restricted_leg() wrote (a measured leg restricted to the split's test key, with a
    # "derived_from" block the held-out step re-derives). "" = undeclared; the held-out step HALTs.
    heldout_base_leg: str = ""
    # prompt_mode "ao": THE AO AUTHORS THE EVOLVER PROMPT, and this is the how-to it is initialised with.
    # Supplied ONCE at init (RUNTIME.json "prompt_policy"), never per epoch — that is the point:
    # a policy is a config, so prompt assembly becomes an ablatable variable instead of a thing a
    # human re-tunes each epoch. "" falls back to src/orchestrator/PROMPT_POLICY.md.
    prompt_policy: str = ""
    prompt_author_model: str = "opus[1m]"
    ao_continuous_session: bool = False        # one AO conversation per run; see RUNTIME wiring
    # An arm already in flight was launched with a hand-chosen cache tag; the spec-derived tag
    # would send later epochs to a DIFFERENT bucket than epoch 1 wrote. Safe (a distinct tag
    # never replays stale records) but it silently splits one arm across two cache
    # lineages, and the next epoch's digest builder would look in the wrong place. "" = use the
    # spec's derived tag, which is correct for any run the loop starts itself.
    cache_tag: str = ""
    # `--plan` promises to "report every verify and change nothing". Before questions existed that
    # was free: every verify was a pure read. ask() writes BLOCKED_e{N}.json, so planning would
    # now leave files behind unless the read-only promise is made mechanical rather than assumed.
    readonly: bool = False
    # Wrap a LOCAL evolver's `claude` child in sandbox-exec. DEFAULT ON, with `"evolver_sandbox":
    # false` in RUNTIME.json as the recorded opt-out: if containment is not the default, the default
    # is UNCONTAINED, and the AO likewise refuses to dispatch uncontained rather than degrading.
    #
    # What "uncontained" costs: the only thing then standing between the evolver and the split
    # file, the held-out specs and peer arms' skills is a set of string-matching hooks, which
    # can false-positive, be partially fixed, or examine nothing and report "passed". The result is
    # not a crash, it is that a positive gate result becomes unfalsifiable after the fact.
    #
    # DO NOT FLIP THIS MID-RUN: containment should be a no-op on every number, but "should be" is
    # not "measured", so a change of world belongs at a run boundary.
    evolver_sandbox: bool = True
    # SSB v2 only: which skill-version bucket of a multi-bucket cache to digest.
    digest_bucket: str = ""
    # SSB v2, EPOCH 1 only: every `<tag>_tasks/_snapshots` root that holds the base leg's workbook
    # snapshots. A cache-REPLAYED leg writes none, so each replayed rollout's snapshot sits under the
    # tag that first scored it, and the root derived from the cache name can hold only part of the
    # train tasks. Declared, never derived;
    # () keeps the old single derived root, and the builder's REQUIRE mode refuses any gap.
    digest_snapshot_roots: tuple = ()
    # WHICH EVIDENCE THE EVOLVER IS GRANTED (RUNTIME.json `evolver_evidence`).
    # "current_epoch" — the default when the key is absent: digests_eK only.
    # "all_epochs"    — digests_e1..eK of THIS run (EpochPaths.evidence_dirs()), read on demand.
    # Runtime plumbing like evolver_sandbox, deliberately NOT an ExperimentSpec field: load() fills
    # defaults and _canonical hashes every field, so a new spec field would change config_sha256
    # and make assert_unmutated call every frozen SPEC.json MUTATED. It changes what the evolver
    # sees, so runs under it are a DIFFERENT CONDITION from current_epoch runs and the receipt
    # records it. REFUSED with the evolver sandbox off: --add-dir does not confine Bash, so "all
    # earlier rollouts" would silently become "the whole run dir".
    evolver_evidence: str = "current_epoch"
    # The evidence the evolver reads (RUNTIME `evolver_evidence_layout`): high-fidelity rollout traces,
    # each epoch's skills, and each rollout's score. "digests" (absent) = the per-task digests only;
    # "rollouts" = every evidence dir ALSO holds
    # rollouts/<task>/s<k>.trace.jsonl + s<k>.grade.txt, solver_context/ and INDEX.json/.md
    # (src/orchestrator/rollout_evidence.py): built by the digest step's runner for the current epoch,
    # re-derived from the bytes by the train-only and completeness steps for every granted dir.
    # RUNTIME-only for the config-hash reason of evolver_evidence; SSB v2 only. "evo" = the stateful
    # session's layout, <run>/evo_e{K}/ (src/orchestrator/evo_layout.py).
    evidence_layout: str = "digests"
    # THE EVOLVER SESSION (RUNTIME `evolver_session`): "per_epoch" (absent) = one container per epoch;
    # "stateful" = ONE long-lived container on the
    # leg pool's machine and ONE claude session for the whole run (epoch 1 `--session-id`, later epochs
    # `--resume`), the loop-rendered epoch message + an appended system prompt, container-native paths
    # (/evo/traces/epoch_K, /evo/skills/epoch_K, /evo/staging, /evo/notes), the score table with epoch 0.
    # Admitted only by _evolver_session (stateful_evolver.validate_session's rules). RUNTIME-only for the
    # config-hash reason of evolver_evidence.
    evolver_session: str = "per_epoch"
    leg_timeout_s: int = 3600
    evolver_timeout_s: int = 5400
    # THE SOLVER'S PROVIDER PINS, DECLARED (RUNTIME.json `solver_env`), exported into every leg's
    # env by _leg_argv_env. {} (absent): the leg inherits whatever the operator's shell exports.
    # When declared, every OTHER EVOSKILL_SOLVER_* routing
    # variable inherited from the shell is REMOVED, so a leftover pin cannot reach a leg unrecorded.
    solver_env: dict = field(default_factory=dict)
    # WHERE THE EVOLVER RUNS (RUNTIME `evolver_host`; RUNTIME-only for the same config-hash reason as
    # evolver_evidence). "local" (absent) = a local launcher under sandbox-exec (not shipped).
    # "daytona" = on Daytona, from the SOLVER's image (+ Claude Code), so evolver and solver see the
    # same environment; only this run's train-side files are uploaded, egress api.anthropic.com only.
    # A different condition from every local-host run -- the receipt records it and
    # _v_evolver_contained refuses a receipt from the other world.
    evolver_host: str = "local"
    # daytona only: {oauth_token_file, ttl_minutes, require_libreoffice, require_snapshot_ref}. The
    # token file PATH travels to the driver; the token itself never enters this process's records.
    # evolver_daytona.placement says WHERE on Daytona -- absent / "own_sandbox" = tau3_evolver_daytona.py (its own VM, unchanged);
    # "pool_container" = scripts/daytona/evolver_container.py, an isolated container on the leg pool's one machine
    # (src/orchestrator/pool_container_evolver.py; optional evolver_daytona.container = its cgroup limits).
    evolver_daytona: dict = field(default_factory=dict)
    # The evolver run log, the only record of what the evolver actually read.
    # "" means the conventional per-epoch name; an ABSENT log is a HALT, never a skip.
    evolver_log: str = ""
    # WHO WRITES ITER{N}_PROMPT.md (RUNTIME `prompt_mode`; RUNTIME-only for the same config-hash
    # reason as evolver_evidence). "ao" (absent) = the AO prompt author writes it every epoch.
    # "template" = the benchmark adaptor worked ONCE at setup (setup/benchmark_plugin.py, frozen in
    # setup/PLUGIN.json) and the LOOP renders the prompt every epoch: no model call. The epoch then
    # also starts with the infra agreement check and the frozen-plug-in check, re-grades a "grader
    # failed" sample ONCE in the grading sandbox and halts if that fails too, and uploads what the
    # run's evolver grants say. See _template_steps() for exactly which steps differ; every prompt
    # check runs on the rendered file unchanged.
    prompt_mode: str = "ao"
    # Val rollouts never block the loop. RUNTIME `pipeline_val`: true = the PIPELINED epoch order (see the PIPELINED VAL section
    # below): epoch N+1 starts as soon as epoch N's TRAIN leg is in and checked, every val leg is
    # checked when it ends and a failed val check never halts evolution, and only the end awaits them.
    # Absent/false = the serial order. RUNTIME-only for the config-hash reason of evolver_evidence:
    # the order changes when a val leg is looked at, not what any leg measures.
    pipeline_val: bool = False
    # RUNTIME daytona_leg.pool is true: ONE leg pool on ONE machine admits the rollouts, so (in pipeline
    # mode only) more than two leg processes may be alive. False = the loop's two-leg seat ceiling.
    leg_pool: bool = False
    # RUNTIME `skip_last_train`, valid only with pipeline_val: the run's LAST epoch (spec n_epochs)
    # launches ONLY its val leg, because no later evolver reads its train leg; its train-leg checks are
    # recorded "skipped by design (no later evolver reads it)". Every other epoch is unchanged.
    # ABSENT with pipeline_val true means true (_pipeline_runtime); an explicit false is honoured.
    # The Ctx default stays False.
    skip_last_train: bool = False
    paths: EpochPaths = field(init=False)

    def __post_init__(self):
        self.paths = EpochPaths(self.run_dir, self.epoch)


EVIDENCE_MODES = ("current_epoch", "all_epochs")


def _evidence_mode(ctx: Ctx) -> str:
    """The run's declared evidence mode, or a HALT. Checked, never assumed.

    all_epochs is REFUSED when the evolver would run uncontained (claude_cli with
    evolver_sandbox false): without the kernel profile, --add-dir does not confine Bash and
    `--hide <run dir>` does not exist, so the list of granted digest dirs would describe nothing —
    the whole run dir, val legs and gate results included, is reachable either way. The direct
    harness is contained unconditionally.
    """
    mode = getattr(ctx, "evolver_evidence", "current_epoch")
    if mode not in EVIDENCE_MODES:
        raise Halt(f"RUNTIME.json evolver_evidence={mode!r} is not one of {EVIDENCE_MODES}. "
                   f"Absent means current_epoch; a typo must not silently mean either.")
    if mode == "all_epochs":
        harness = getattr(ctx.spec, "evolver_harness", "claude_cli")
        if harness != "direct" and not bool(getattr(ctx, "evolver_sandbox", True)):
            raise Halt("RUNTIME.json declares evolver_evidence=all_epochs AND evolver_sandbox="
                       "false. Refusing: the all-epochs grant is a list of subpaths under a run "
                       "dir the kernel profile hides by name; with no profile there is no hide, "
                       "and --add-dir does not confine Bash, so val legs, gate results "
                       "and sealed-test files would be reachable. Turn the sandbox on, or drop "
                       "the all-epochs key.")
    return mode


def _evidence_layout(ctx: Ctx) -> str:
    """The run's declared evidence LAYOUT (RUNTIME evolver_evidence_layout), or a HALT. "rollouts" is
    SSB v2 only: tau3's builder writes no traces of this kind, and a declaration nothing builds must
    not read as one that holds."""
    from src.orchestrator import rollout_evidence as _re
    v = getattr(ctx, "evidence_layout", "digests")
    if v == "evo":
        # The stateful session's layout -- <run>/evo_e{K}/ (raw session logs,
        # solver workspaces, scores.json; src/orchestrator/evo_layout.py) uploaded as /evo/traces/epoch_K etc.
        if getattr(ctx, "evolver_session", "per_epoch") != "stateful":
            raise Halt(f"RUNTIME.json {_re.LAYOUT_KEY}='evo' needs evolver_session=stateful (the container layout "
                       f"for the stateful session); the per-epoch driver reads digests or rollouts")
        return v
    if v not in _re.LAYOUTS:
        raise Halt(f"RUNTIME.json {_re.LAYOUT_KEY}={v!r} is not one of {_re.LAYOUTS}. Absent means "
                   f"'digests'; a typo must not silently mean either.")
    if v == "rollouts" and getattr(ctx.spec, "benchmark", "") != "ssbv2":
        raise Halt(f"RUNTIME.json {_re.LAYOUT_KEY}='rollouts' but benchmark "
                   f"{getattr(ctx.spec, 'benchmark', '')!r}: only the SSB v2 surface builds rollout evidence")
    return v


def _evolver_session(ctx: Ctx) -> str:
    """The run's declared evolver SESSION (RUNTIME `evolver_session`), or a HALT. "per_epoch" (absent)
    = one container per epoch, nothing else checked. "stateful" = ONE container and ONE claude session
    for the whole run, admitted by stateful_evolver.validate_session only when its rules hold
    (prompt_mode template; evolver_host daytona with the pool container from epoch 1, no
    placement_from_epoch; evolver_evidence all_epochs; evolver_evidence_layout evo; claude_cli on
    an explicit claude-* model; EVOSKILL_CLAUDE_EFFORT opted out = Anthropic's default effort; RUNTIME
    claude.effort "default"; daytona_leg.pool true with pool_ttl_minutes). A typo HALTs; a Ctx that disagrees
    with the run's RUNTIME.json HALTs (the driver and the upload check read that file). Checked, never
    assumed: called in main()'s preflight, and before the render/evolver steps spend."""
    from src.orchestrator import stateful_evolver as _se
    _se.validate_session(ctx)
    return getattr(ctx, "evolver_session", "per_epoch")


def _evidence_sources(ctx: Ctx, j: int) -> tuple:
    """(leg json, producing library) behind digests_e<j>_train: epoch 1 = the declared base leg and no
    library; epoch j = skills_v{j-1}'s train leg and skills_v{j-1} (the _batch_leg mapping, per j)."""
    rd = Path(ctx.run_dir)
    if j <= 1:
        return (Path(ctx.base_leg) if getattr(ctx, "base_leg", "") else None), None
    return EpochPaths(rd, j - 1).leg("train"), EpochPaths(rd, j - 1).promoted


_EVIDENCE_DIR_RE = re.compile(r"^digests_e(\d+)_train$")


def _rollout_evidence_build_cmd(ctx: Ctx, j: int) -> str:
    """The exact CLI that builds digests_e<j>_train's rollout evidence (named by every HALT that needs it)."""
    leg, lib = _evidence_sources(ctx, j)
    cache = (str(ctx.digest_caches[0]) if (j == 1 and ctx.digest_caches)
             else str(REPO / ".cache" / f"{ctx.spec.name}_e{j - 1}_cache"))
    return (f"{sys.executable} src/orchestrator/rollout_evidence.py build --out "
            f"{EpochPaths(Path(ctx.run_dir), j).digests} --run-dir {ctx.run_dir} --leg {leg} "
            f"--split {ctx.spec.split_path} --n-samples {int(ctx.spec.n_samples)} --epoch {j}"
            + (f" --library {lib}" if lib else "") + f" --cache {cache}"
            + (f" --bucket {ctx.digest_bucket}" if (j == 1 and ctx.digest_bucket) else "")
            + f" --model {ctx.spec.solver_model}")


def _rollout_evidence_problems(ctx: Ctx, dirs) -> list:
    """rollout_evidence.check_dir over every dir, each against the leg and library it describes."""
    from src.orchestrator import rollout_evidence as _re
    out = []
    for d in dirs:
        d = Path(d)
        m = _EVIDENCE_DIR_RE.match(d.name)
        if not m:
            out.append(f"{d.name}: not a digests_e<j>_train dir, so its leg cannot be named")
            continue
        j = int(m.group(1))
        leg, lib = _evidence_sources(ctx, j)
        if leg is None or not leg.is_file():
            out.append(f"{d.name}: the leg it describes ({leg}) is not on disk")
            continue
        probs = _re.check_dir(d, split_path=ctx.spec.split_path, n_samples=int(ctx.spec.n_samples),
                              library=lib, leg_json=leg, run_dir=ctx.run_dir)
        if probs:
            fix = ("move the dir OUT of the run dir (never delete evidence an evolver read) and rebuild "
                   "it" if (d / _re.INDEX_JSON).is_file() else "build it")
            out.append(f"{d.name}: " + "; ".join(probs[:4]) + (f" (+{len(probs) - 4} more)" if len(probs) > 4 else "")
                       + f". REMEDY: {fix}: {_rollout_evidence_build_cmd(ctx, j)}")
    return out


def _ensure_rollout_evidence(ctx: Ctx, out: Path, cache, bucket: str = "") -> None:
    """Rollouts layout: build the CURRENT epoch's rollout evidence into `out` when it holds none. An
    existing INDEX.json is never rebuilt here (the steps re-derive it; a bad one HALTs with the remedy)."""
    from src.orchestrator import rollout_evidence as _re
    if _evidence_layout(ctx) != "rollouts" or (Path(out) / _re.INDEX_JSON).is_file():
        return
    leg, lib = _evidence_sources(ctx, ctx.epoch)
    if leg is None or not leg.is_file():
        raise Halt(f"the rollout evidence for {Path(out).name} describes {leg}, which is not on disk")
    try:
        s = _re.build(out, run_dir=ctx.run_dir, leg_json=leg, split_path=ctx.spec.split_path,
                      n_samples=int(ctx.spec.n_samples), library=lib, epoch=ctx.epoch,
                      cross_check_cache=(str(cache) if cache else None), bucket=bucket or "",
                      expect_model=ctx.spec.solver_model)
    except (_re.EvidenceError, ValueError, OSError) as e:
        raise Halt(f"the rollout evidence for {Path(out).name} could not be built: {e}")
    c, z = s["counts"], s["sizes"]
    print(f"  [rollout evidence] {Path(out).name}: {c['rollouts']} rollouts ({c['passed']} passed, "
          f"{c['graded_wrong']} graded_wrong, {c['not_eligible']} not_eligible, {c['grader_failed']} "
          f"grader_failed; turn cap hit {c['turn_cap_hit']}; loaded a skill {c['rollouts_that_loaded_a_skill']}; "
          f"redactions {c['redactions']}); traces {z['traces']['bytes'] / 1e6:.1f} MB, grades "
          f"{z['grades']['bytes'] / 1e6:.2f} MB, dir {s['dir_total_bytes'] / 1e6:.1f} MB")


EVOLVER_HOSTS = ("local", "daytona")


def _evolver_host(ctx: Ctx) -> str:
    """The run's declared evolver host, or a HALT. daytona needs the claude_cli harness, the sandbox
    flag on (its receipts are containment receipts) and the run's one Claude Code credential
    (cc_account.resolve: evolver_daytona.oauth_token_file, else $EVOSKILL_CC_TOKEN_FILE -- a token file
    readable by its owner only -- else ANTHROPIC_API_KEY). Any benchmark: the pool machine runs the run's
    own leg image, and the stateful session reads only the evo layout."""
    host = getattr(ctx, "evolver_host", "local")
    if host not in EVOLVER_HOSTS:
        raise Halt(f"RUNTIME.json evolver_host={host!r} is not one of {EVOLVER_HOSTS}.")
    if host == "daytona":
        if getattr(ctx.spec, "evolver_harness", "claude_cli") != "claude_cli":
            raise Halt("evolver_host=daytona needs evolver_harness=claude_cli")
        if not bool(getattr(ctx, "evolver_sandbox", True)):
            raise Halt("evolver_host=daytona with evolver_sandbox=false: refusing a contradiction")
        cfg = getattr(ctx, "evolver_daytona", {}) or {}
        if not isinstance(cfg, dict):
            raise Halt("RUNTIME.json evolver_daytona must be an object")
        if getattr(ctx, "evolver_log", ""):
            raise Halt("evolver_log with evolver_host=daytona: the driver installs the log at the "
                       "conventional path, so a declared override would audit a different file")
        from .cc_account import resolve as _credential
        try:
            _credential(cfg.get("oauth_token_file"))
        except ValueError as e:
            raise Halt(f"the evolver's Claude Code credential: {e}")
        if _evolver_placement(ctx, configured=True) == "pool_container":
            if not bool(getattr(ctx, "leg_pool", False)):
                raise Halt("evolver_daytona.placement=pool_container needs the leg pool (RUNTIME daytona_leg.pool: "
                           "true): the evolver's container runs on the pool's machine")
            # its cgroup limits and the machine wait, refused HERE (before any spend) rather than by the driver
            from src.orchestrator.pool_container_evolver import container_limits_cfg, machine_wait_s
            try:
                container_limits_cfg(cfg)
                machine_wait_s(cfg)
            except ValueError as e:
                raise Halt(str(e))
            # No declared LEG snapshot (daytona_leg.require_snapshot_ref / evolver_daytona.require_leg_snapshot_ref
            # null) = not pinned: the pool machine's image is the leg Dockerfile by its sha256 (checked before the
            # machine is created), and the snapshot ref Daytona reports is accepted and recorded in each receipt.
    return host


def _evolver_placement(ctx: Ctx, *, configured: bool = False) -> str:
    """RUNTIME evolver_daytona.placement ("own_sandbox" when absent), or a HALT on any other value.
    evolver_daytona.placement_from_epoch N (a switch to the pool container mid-run): an epoch before N
    is "own_sandbox" -- its evolver already ran on its own VM, and its receipt is checked as such; configured=True
    gives the declared value whatever the epoch (the pool's machine egress and the config checks follow it)."""
    from src.orchestrator.pool_container_evolver import placement
    cfg = getattr(ctx, "evolver_daytona", {}) or {}
    try:
        pl = placement(cfg)
    except ValueError as e:
        raise Halt(str(e))
    fe = cfg.get("placement_from_epoch")
    if fe is not None and (not isinstance(fe, int) or isinstance(fe, bool) or fe < 1):
        raise Halt(f"RUNTIME evolver_daytona.placement_from_epoch={fe!r} (an integer >= 1, or absent)")
    if configured or fe is None or pl == "own_sandbox":
        return pl
    return pl if int(ctx.epoch) >= fe else "own_sandbox"


def _pool_container_check_kw(ctx: Ctx) -> dict:
    """What check_pool_container_receipt asserts from the run's declarations (the evolver's and the AO's receipt
    alike): the LibreOffice pin inside the container, and the LEG snapshot of the pool machine -- evolver_daytona.
    require_leg_snapshot_ref, else daytona_leg.require_snapshot_ref (the driver derives it the same way;
    evolver_daytona.require_snapshot_ref names the OWN evolver VM's snapshot and does not apply here)."""
    cfg = getattr(ctx, "evolver_daytona", {}) or {}
    rt = {}
    try:
        rt = json.loads((Path(ctx.run_dir) / "RUNTIME.json").read_text())
    except (OSError, ValueError):
        pass
    return {"require_libreoffice": cfg.get("require_libreoffice"),
            "require_snapshot_ref": cfg.get("require_leg_snapshot_ref")
            or ((rt.get("daytona_leg") or {}) if isinstance(rt.get("daytona_leg"), dict) else {})
            .get("require_snapshot_ref")}


def _evolver_env_facts(ctx: Ctx) -> dict:
    """FACTS for the daytona evolver's environment, every value from a file: the image recipe
    re-hashed from disk, the solver versions from the base train leg's own runtime_env (the legs
    run the SAME recipe), the inputs dir from its verified MANIFEST. Raises if any is missing."""
    from src.orchestrator.daytona_evolver import (dockerfile_check, inputs_dir,
                                                  verify_inputs_dir)
    ok, det, rec = dockerfile_check()
    if not ok:
        raise Halt(f"evolver image recipe does not check: {det}")
    c = verify_inputs_dir(ctx.run_dir, Path(ctx.spec.split_path))
    if c.status != "PASS":
        raise Halt(f"inputs dir does not verify: {c.detail}")
    base = Path(ctx.base_leg) if ctx.base_leg else None
    rt = (json.loads(base.read_text()).get("runtime_env") or {}) if base and base.is_file() else {}
    stack = str(rt.get("solver_stack") or "").split()
    if len(stack) < 3 or not rt.get("libreoffice"):
        raise Halt(f"base leg {base} records no solver_stack/libreoffice in runtime_env")
    d = inputs_dir(ctx.run_dir)
    files = [f for f in d.rglob("*") if f.is_file() and f.name != "MANIFEST.json"]
    return {"image_recipe": rec["base"], "image_sha12": rec["base_sha256"][:12], "python": "3.11",
            "openpyxl": stack[0], "pandas": stack[1], "numpy": stack[2],
            "libreoffice": rt["libreoffice"], "inputs_dir": str(d), "n_inputs": len(files),
            "inputs_mb": round(sum(f.stat().st_size for f in files) / 1e6, 1)}


PROMPT_MODES = ("ao", "template")
TEMPLATE_PARAGRAPH = "EVOLVER_TEMPLATE.md"
RENDER_RECORD = "e{epoch}_PROMPT_RENDER.json"


def _prompt_mode(ctx: Ctx) -> str:
    """The run's declared prompt mode (RUNTIME `prompt_mode`), or a HALT. Checked, never assumed.

    "template" needs the Daytona evolver host: the rendered prompt names the clean grader and, when
    granted, the training reference workbooks, and only the Daytona upload carries those, under
    grants (RUNTIME evolver_daytona) that check_upload_manifest re-derives. A local-host evolver
    would be told about directories its sandbox-exec profile does not grant."""
    mode = getattr(ctx, "prompt_mode", "ao")
    if mode not in PROMPT_MODES:
        raise Halt(f"RUNTIME.json prompt_mode={mode!r} is not one of {PROMPT_MODES}. Absent means "
                   f"'ao'; a typo must not silently mean either.")
    if mode == "template" and getattr(ctx, "evolver_host", "local") != "daytona":
        raise Halt("RUNTIME.json prompt_mode=template needs evolver_host=daytona: the rendered prompt "
                   "names the clean grader (and, when granted, the training reference workbooks), "
                   "and only the Daytona upload carries them under grants the loop re-derives.")
    return mode


def _grants(ctx: Ctx) -> dict:
    """The operator's grants for the evolver VM, from RUNTIME.json evolver_daytona (never from the
    AO): train_inputs, train_references (daytona_evolver.evolver_grants) and grader_clean
    (daytona_evolver.evolver_grader_grant). The Daytona driver and check_upload_manifest read the
    same file, so the loop, the upload and the audit cannot disagree about what was granted."""
    from src.orchestrator.daytona_evolver import evolver_grader_grant, evolver_grants
    try:
        g = dict(evolver_grants(ctx.run_dir))
        g["grader_clean"] = evolver_grader_grant(ctx.run_dir)
    except ValueError as e:
        raise Halt(f"the evolver's grants cannot be read: {e}")
    g["changed_cells"] = _changed_cells_grant(ctx)
    return g


CHANGED_CELLS_GRANT = "changed_cells"


def _changed_cells_grant(ctx: Ctx) -> bool:
    """RUNTIME.json evolver_daytona.changed_cells: the digests' "What the SOLVER changed" sections.
    A JSON boolean or absent (= OFF: their lines carry training INPUT cell values); anything else HALTs,
    like the other grants. Read from the file, as evolver_grants reads the other grants."""
    f = Path(ctx.run_dir) / "RUNTIME.json"
    if not f.is_file():
        return False
    try:
        rt = json.loads(f.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise Halt(f"{f} is unreadable ({type(e).__name__}): refusing to guess the changed-cells grant")
    cfg = rt.get("evolver_daytona", {}) if isinstance(rt, dict) else None
    if not isinstance(cfg, dict):
        raise Halt("RUNTIME.json evolver_daytona must be an object")
    v = cfg.get(CHANGED_CELLS_GRANT, False)
    if not isinstance(v, bool):
        raise Halt(f"RUNTIME.json evolver_daytona.{CHANGED_CELLS_GRANT} must be true or false, got {v!r}")
    return v


def _ao_mode_grant_problem(ctx: Ctx) -> str:
    """AO mode, daytona host: the AO's policy addendum describes the inputs dir (FACTS carry it) and
    nothing else, so the upload must carry exactly that. The driver uploads only what
    RUNTIME grants (inputs absent = OFF), so an AO-mode run without `train_inputs: true` would name a
    dir its evolver never receives, and a granted references/grader dir would reach the evolver
    undescribed. "" when consistent (or when no RUNTIME.json exists: inputs on)."""
    g = _grants(ctx)
    probs = []
    if not g["train_inputs"]:
        probs.append("train_inputs is not granted, but the AO-authored prompt names the inputs dir")
    for k in ("train_references", "grader_clean"):
        if g.get(k):
            probs.append(f"{k} is granted, but the AO's policy does not describe that dir")
    if not probs:
        return ""
    return ("prompt_mode=ao on the daytona host: " + "; ".join(probs) + f" ({g.get('source')}). "
            "REMEDY: set RUNTIME.json evolver_daytona.train_inputs=true (and drop train_references / "
            "grader_clean) to keep the AO world, or declare prompt_mode=template.")


def _v_inputs_train_only(ctx: Ctx):
    """daytona host: the evolver's input-workbook dir re-derived train-only (daytona_evolver).

    Template mode follows the operator's grant: with train_inputs off the dir is neither uploaded
    nor named, so there is nothing to verify. AO mode (daytona) additionally requires the grants to
    match what the AO describes (_ao_mode_grant_problem)."""
    if getattr(ctx, "evolver_host", "local") != "daytona":
        return True, "local evolver host: no input workbooks are granted"
    if _prompt_mode(ctx) == "template":
        if not _grants(ctx)["train_inputs"]:
            return True, ("template mode, train_inputs not granted (RUNTIME evolver_daytona): the "
                          "inputs dir is neither uploaded nor named")
    else:
        prob = _ao_mode_grant_problem(ctx)
        if prob:
            return False, prob
    from src.orchestrator.daytona_evolver import verify_inputs_dir
    c = verify_inputs_dir(ctx.run_dir, Path(ctx.spec.split_path))
    return c.status == "PASS", c.detail


def _run_build_inputs(ctx: Ctx) -> None:
    if getattr(ctx, "evolver_host", "local") != "daytona":
        return
    if _prompt_mode(ctx) == "template":
        if not _grants(ctx)["train_inputs"]:
            return
    else:
        prob = _ao_mode_grant_problem(ctx)
        if prob:
            raise Halt(prob)
    from src.orchestrator.daytona_evolver import build_inputs_dir
    build_inputs_dir(ctx.run_dir, Path(ctx.spec.split_path))


# ── template mode (prompt_mode = "template"): steps that exist only there ─────────────────
def _v_infra_agrees(ctx: Ctx):
    """Models, providers, pins, sandboxes and timeouts are declared ONCE, in <run>/INFRA.json, and
    every other statement of the same facts (SPEC, SPEC_EXTRA, RUNTIME, the drivers' constants, both
    Dockerfiles, the token/key files by stat only, this process's environment) must agree with it
    (src/orchestrator/infra.py agree()). Placed FIRST in every template-mode epoch. No runner: the
    only fix is a person re-declaring one side."""
    from src.orchestrator import infra as _inf
    try:
        items = _inf.agree(ctx.run_dir)
    except _inf.InfraError as e:
        return False, (f"INFRA.json is missing or invalid ({'; '.join(e.problems[:4])}). The "
                       f"operator-owned infra config must exist and validate before an epoch starts.")
    if items:
        return False, (f"{len(items)} statement(s) disagree with INFRA.json: "
                       + "; ".join(str(d) for d in items[:6])
                       + (f" (+{len(items) - 6} more)" if len(items) > 6 else "")
                       + ". Fix the other statement or re-declare INFRA.json; `python -m "
                         "src.orchestrator.infra check --run-dir <run>` lists them all.")
    sha = hashlib.sha256((Path(ctx.run_dir) / _inf.INFRA_FILE).read_bytes()).hexdigest()
    return True, (f"INFRA.json sha256:{sha[:12]} agrees with SPEC, SPEC_EXTRA, RUNTIME, the drivers' "
                  f"constants, both Dockerfiles, the secret files (stat only) and this environment")


def _v_plugin_frozen(ctx: Ctx):
    """The benchmark adapter's plug-in (benchmark_plugin.py; an older record may also pin PLAN.json and
    EVOLVER_TEMPLATE.md) is frozen in setup/PLUGIN.json, and nothing it pins has changed since
    (plugin_runner.verify_frozen: every pinned sha re-hashed, the record unedited). Required before
    epoch 1 and re-checked every epoch. No runner: freezing is a separate command. EvoSkill is
    automatic, so nothing is signed; the pins are what hold."""
    from src.orchestrator import plugin_runner as _pr
    try:
        rep = _pr.verify_frozen(ctx.run_dir)
    except _pr.PluginDrift as e:
        probs = (getattr(e, "report", None) or {}).get("problems") or [str(e)]
        return False, ("the benchmark plug-in is not frozen: " + "; ".join(probs[:4])
                       + f". REMEDY: `python -m src.orchestrator.plugin_runner freeze --run-dir {ctx.run_dir}`.")
    return True, (f"setup/PLUGIN.json record {rep['record_sha256'][:12]} re-hashes unchanged (benchmark_plugin.py, "
                  f"plugin_api.py, PLUGIN_VALIDATION.json, tasks.json, split.json)")


def _batch_leg(ctx: Ctx):
    """The leg the newest digest dir was built from: epoch 1 -> the declared base leg, epoch N ->
    skills_v{N-1}'s train leg. The same mapping _run_author_prompt and _v_prompt_numbers use."""
    if ctx.epoch == 1:
        return Path(ctx.base_leg) if ctx.base_leg else None
    return EpochPaths(ctx.run_dir, ctx.epoch - 1).leg("train")


# ── the re-grade: a "grader failed" sample is re-graded ONCE from its saved workbook ────────────────
# Grader failed (the grader raised, or a recalculation / data-table failure warning for that sample)
# -> re-grade from the saved output; halt if it still fails. The re-grade runs
# in the grading sandbox (scripts/daytona/ssb_grade_daytona.py: the leg image, score_and_diff sha-pinned,
# INFRA.json's 900/900 timeouts, egress blocked, no model key, one child process per job). Its work dir
# is the OPERATOR dir (<run>.operator/regrade/<leg stem>.<leg sha12>/): the results quote reference
# values (a val leg's are val goldens) and no agent job ever declares that dir.
REGRADE_DIRNAME = "regrade"
REGRADE_RECORD = "REGRADE.json"
REGRADE_SCHEMA = "evoskill.loop_regrade/1"
GRADE_DRIVER = REPO / "scripts" / "daytona" / "ssb_grade_daytona.py"
GRADE_DRIVER_EXITS = {0: "ok", 4: "refused (nothing created)", 5: "incomplete", 7: "check-only"}
REGRADE_WALL_MIN = 120                                      # the driver's own defaults, stated here
REGRADE_MAX_USD = 3.0
_KEYLIKE_ENV = re.compile(r"(API_KEY|_TOKEN|SECRET|PASSWORD|OAUTH)", re.I)
REGRADABLE = ("passed", "graded_wrong")
# PROVENANCE a re-grade line must carry before its verdict counts: graded by the pinned scorer
# (EXPECTED["baseline_runtime"]["scorer_sha256"] in ssb_leg_daytona_loop.py), by the grading sandbox's
# worker, merged by the driver from a named sandbox, against the reference and input the job names.
# Without it a results line written by another scorer, or on the Mac, would read as a re-grade.
GRADE_SCORER_SHA256 = "2442e5aa0e6172dbf62be5849a00fedf1b53e390d093d65ac7bb19da9b807f81"
GRADE_WORKER = "grade_worker/1"                             # scripts/daytona/grade_worker.py WORKER_VERSION
REGRADE_LOST = "dispatched, no result: a lost re-grade may not be retried silently"


def _regrade_attempts() -> int:
    """evolver_template.REGRADE_ATTEMPTS: the number the prompt and grader_clean/README.md state. The
    loop implements exactly one re-grade per sample (job id .../regrade1), so any other value HALTs."""
    from src.orchestrator.evolver_template import REGRADE_ATTEMPTS
    if REGRADE_ATTEMPTS != 1:
        raise Halt(f"evolver_template.REGRADE_ATTEMPTS is {REGRADE_ATTEMPTS}, but the loop re-grades a "
                   f"grader-failed sample exactly once; the prompt would state a policy the loop does not run")
    return REGRADE_ATTEMPTS


def _regrade_dir(ctx: Ctx, leg) -> Path:
    """<run>.operator/regrade/<leg stem>.<sha12 of the leg JSON>: a re-run leg (new bytes) gets its own dir,
    so a result can never be credited to a different leg's sample."""
    leg = Path(leg)
    return (operator_dir(ctx.run_dir) / REGRADE_DIRNAME
            / f"{leg.stem}.{hashlib.sha256(leg.read_bytes()).hexdigest()[:12]}")


def _regrade_job_id(task: str, sample: int) -> str:
    return f"{task}/s{int(sample)}/regrade{_regrade_attempts()}"


def _regrade_label(leg) -> str:
    """The grading sandbox's label ([a-z0-9-], <= 24): rg-v1tr / rg-v2va / rg-b0tr / rg-h<sha8>."""
    stem = Path(leg).stem
    m = re.search(r"_v(\d+)_(train|val)$", stem)
    if m:
        return f"rg-v{m.group(1)}{m.group(2)[:2]}"
    m = re.search(r"_base_(train|val)$", stem)
    if m:
        return f"rg-b0{m.group(1)[:2]}"
    return "rg-h" + hashlib.sha256(stem.encode()).hexdigest()[:8]


def _saved_output(sample: dict):
    """(path, "") of the ONE workbook this sample saved (the snapshot's workspace/output_raw, the copy
    kept before the first grade rewrote output/), or (None, why)."""
    snap = sample.get("snapshot")
    if not snap:
        return None, "no snapshot of this sample exists, so there is no saved workbook to re-grade"
    raw = Path(snap) / "workspace" / "output_raw"
    files = sorted(p for p in raw.iterdir() if p.is_file()) if raw.is_dir() else []
    if len(files) != 1:
        return None, f"{raw} holds {len(files)} file(s), not exactly one saved workbook"
    return files[0], ""


def _regrade_sources(task: str, run_dir=None) -> dict:
    """{gt_file, gt_sha256, input_file, input_sha256}: the reference and input a re-grade of `task` must
    be graded against -- the dataset's own mapping (daytona_evolver.dataset_index), hashed through the
    benchmark data overlay (data_overlay.source: a repaired file's BYTES, its path unchanged), which is
    how ssb_grade_daytona.normalize_jobs states the job's shas. Root and overlay by the grade driver's
    rule: the run's INFRA.json block "data" when it has one (overlay_patches_json null = no overlay, the
    PATCHES.json must hash to overlay_patches_sha256), else daytona_evolver.data_root() and the bare
    EVOSKILL_SSBV2_DATA_OVERLAY. ValueError when they cannot be stated (INFRA.json invalid, not a dataset
    task, a file missing, an overlay entry that no longer hashes as declared)."""
    from src.orchestrator import data_overlay
    from src.orchestrator import infra as _inf
    from src.orchestrator.daytona_evolver import data_root, dataset_index, sha256_file
    root, patches, declared = data_root(), None, False
    try:
        inf = _inf.load_run(run_dir) if run_dir else None
    except _inf.InfraError as e:
        raise ValueError(f"INFRA.json is invalid ({'; '.join(e.problems[:2])})")
    blk = (inf or {}).get("data")
    if isinstance(blk, dict):
        root, patches, declared = Path(blk.get("root") or root), blk.get("overlay_patches_json"), True
    info = dataset_index(root).get(task)
    if not info:
        raise ValueError(f"{task} is not a task of the dataset at {root}")
    out = {}
    try:
        if patches and sha256_file(patches) != blk.get("overlay_patches_sha256"):
            raise ValueError(f"{patches} does not hash to INFRA.json data.overlay_patches_sha256")
        for lab, key in (("gt", "golden"), ("input", "input")):
            out[f"{lab}_file"] = info[key]
            out[f"{lab}_sha256"] = sha256_file(Path(info[key]) if declared and not patches
                                               else data_overlay.source(info[key], patches))
    except (OSError, data_overlay.OverlayError) as e:
        raise ValueError(f"{task}: {type(e).__name__}: {e}")
    return out


def _dispatched(rec) -> dict:
    """{job_id: dispatch entry} of a REGRADE.json: every re-grade the loop handed to the driver. A record
    written before the dispatch list existed names its jobs in `samples` (written before its launch)."""
    if not isinstance(rec, dict):
        return {}
    got = rec.get("dispatched")
    if isinstance(got, dict):
        return got
    if rec.get("invocations"):
        return {s.get("job_id"): {"invocation": None} for s in rec.get("samples") or [] if s.get("job_id")}
    return {}


def _write_regrade_record(rp: Path, rec: dict) -> None:
    """REGRADE.json, written whole (tmp, fsync, rename): the dispatch list must be on disk BEFORE the
    driver starts, and a torn write must read as unreadable (a HALT), never as "nothing dispatched"."""
    tmp = rp.with_name(rp.name + ".tmp")
    with open(tmp, "w") as fh:
        fh.write(json.dumps(rec, indent=1, default=str))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, rp)


def _read_regrade(d: Path) -> tuple:
    """(REGRADE.json or None, {job_id: the FIRST merged result line}) of a re-grade dir. The driver never
    rewrites a line and skips a job id it already has, so the first line is the one re-grade."""
    rec, res = None, {}
    rp = d / REGRADE_RECORD
    if rp.is_file():
        try:
            rec = json.loads(rp.read_text())
        except (OSError, json.JSONDecodeError):
            rec = {"unreadable": True}
    rf = d / "results.jsonl"
    if rf.is_file():
        for ln in rf.read_text(errors="replace").splitlines():
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if isinstance(r, dict) and r.get("job_id") and r["job_id"] not in res:
                res[r["job_id"]] = r
    return rec, res


def _regrade_outcome(task: str, out_sha: str, r, want: dict | None = None) -> tuple:
    """(outcome, why) of one re-grade result line (grade_worker.py + the driver's host block):
    passed / graded_wrong = a completed grade with no failure warning; grader_failed = it raised, timed
    out, crashed, gave no verdict, or logged a failure warning; invalid = not a grade of THIS saved
    workbook by the grading sandbox (refused, other bytes, another task, a source changed, or no
    provenance: a worker other than GRADE_WORKER, no host.sandbox_id, a reference / input that does not
    hash to the job's `want`, a scorer other than GRADE_SCORER_SHA256); missing = no line yet.
    A timed-out or crashed line comes from the worker's parent (the child never reported), so it carries
    the worker and the sandbox but no child fields; any line whose child ran must carry them all."""
    if r is None:
        return "missing", "no result line"
    if r.get("mode") != "grade" or r.get("task_id") != task:
        return "invalid", f"the result is a {r.get('mode')!r} of {r.get('task_id')!r}, not a grade of {task}"
    if r.get("worker") != GRADE_WORKER:
        return "invalid", f"worker {r.get('worker')!r}, not the grading sandbox's {GRADE_WORKER}"
    h = r.get("host") if isinstance(r.get("host"), dict) else {}
    if not h.get("sandbox_id"):
        return "invalid", "no host.sandbox_id: the driver did not merge this line from a grading sandbox"
    st = r.get("status")
    if st == "refused":
        return "invalid", f"refused: {str(r.get('refused'))[:200]}"
    if st in ("timeout", "crashed"):            # the worker's own line: the child never reported back
        return "grader_failed", str(st)
    if r.get("output_sha256_before") != out_sha:
        return "invalid", (f"it graded bytes {str(r.get('output_sha256_before'))[:12]}, not the saved "
                           f"workbook's {out_sha[:12]}")
    for lab in ("gt", "input"):
        got, exp = r.get(f"{lab}_sha256_before"), (want or {}).get(f"{lab}_sha256")
        if not exp or got != exp:
            return "invalid", (f"it graded against a {'reference' if lab == 'gt' else 'input'} hashing to "
                               f"{str(got)[:12]}, not the job's {str(exp)[:12]}")
    if r.get("originals_unchanged") is False:
        return "invalid", "a source file changed during the re-grade"
    sc = r.get("scorer_sha256")
    if (sc or st == "ok") and sc != GRADE_SCORER_SHA256:    # no sha = the scorer never imported (raised)
        return "invalid", f"scorer sha256 {str(sc)[:12]}, not the pinned {GRADE_SCORER_SHA256[:12]}"
    if st == "exception":
        exc = r.get("exception") if isinstance(r.get("exception"), dict) else {}
        return "grader_failed", f"exception{': ' + str(exc.get('type', '')) if exc else ''}"
    if st != "ok" or not isinstance(r.get("passed"), bool):
        return "grader_failed", f"no verdict (status {st!r})"
    if r.get("d4_failure_warnings"):
        return "grader_failed", "failure_warning: " + str(r["d4_failure_warnings"][0])[:160]
    oc = h.get("outcome")
    if oc == "grader_failed":
        return "grader_failed", str(h.get("outcome_reason"))
    if oc not in REGRADABLE or (oc == "passed") != r["passed"]:
        return "invalid", f"host outcome {oc!r} ({h.get('outcome_reason')}) with passed={r['passed']}"
    return oc, str(h.get("outcome_reason") or "")


def _counts_text(c: dict) -> str:
    return (f"passed {c['passed']}, graded and wrong {c['graded_wrong']}, not eligible to grade "
            f"{c['not_eligible']}, grader failed {c['grader_failed']} (n={c['n_episodes']})")


def _leg_outcomes_after_regrade(ctx: Ctx, leg, what: str) -> dict:
    """One leg's four outcomes from its own records (outcomes.leg_outcome_counts), with every
    "grader failed" sample replaced by its ONE re-grade from the saved workbook (the grading sandbox's
    results in _regrade_dir). Returns {"ok", "state", "detail", "counts", "rows", "dir"}; state is
    ok | no_leg | unreadable | unreportable | needs_regrade | cannot_regrade | regrade_failed |
    regrade_invalid | regrade_record_unreadable | regrade_lost | regrade_flip.

    WHY A RE-GRADE THAT CHANGES THE RECORDED VERDICT IS A HALT. The leg JSON's per_task is the record every
    downstream number reads (pairing, reconcile, the prompt figures, the gate) and the loop never rewrites
    a recorded verdict. A re-grade that agrees resolves the sample; one that disagrees means the recorded
    number is wrong, and only a person may decide what replaces it.

    WHY A DISPATCHED RE-GRADE WITH NO RESULT IS A HALT. _run_regrade records every job
    as dispatched in REGRADE.json before it launches the driver. If the driver dies after the sandbox
    graded a job but before the results were merged, that job has no line; grading it again would let a
    lost FAILING re-grade be retried until one passes. So only a job never dispatched is "missing"."""
    from src.orchestrator import outcomes as _oc
    out = {"ok": False, "state": "", "detail": "", "counts": None, "rows": [], "dir": None}

    def done(state, detail, ok=False, **kw):
        out.update(state=state, detail=detail, ok=ok, **kw)
        return out
    if leg is None:
        return done("no_leg", f"{what}: no leg declared (RUNTIME.json base_leg) to read the outcomes of")
    leg = Path(leg)
    if not leg.is_file():
        return done("no_leg", f"{what}: no leg result at {leg}")
    try:
        r = _oc.leg_outcome_counts(ctx.run_dir, leg)
    except Exception as e:                                      # noqa: BLE001
        return done("unreadable", f"{what} {leg.name}: the four outcomes could not be read from its "
                                  f"records ({type(e).__name__}: {e}); an unreadable record is not a zero")
    counts = _counts_text(r)
    unrep = (f"{what} {leg.name}: outcomes are not reportable (status {r['status']!r}; "
             f"{len(r['mismatches'])} mismatch(es), {len(r['unattributed_warnings'])} unattributed "
             f"failure warning(s), {len(r['problems'])} problem(s); pass matches recorded: "
             f"{r['pass_matches_recorded']}) ({counts}). A re-grade cannot fix this: run `python "
             f"src/orchestrator/outcomes.py {ctx.run_dir} {leg}` for the samples; a person decides.")
    if r["mismatches"] or r["problems"] or r["unattributed_warnings"]:
        return done("unreportable", unrep)
    base = {k: r[k] for k in _oc.OUTCOMES}
    base.update(n_episodes=r["n_episodes"], status="ok", reportable=True, regraded=0,
                source="outcomes.leg_outcome_counts")
    failed = [s for s in r["samples"] if s["outcome"] == "grader_failed"]
    if not failed:
        if not r["reportable"] or not r["pass_matches_recorded"]:
            return done("unreportable", unrep)
        return done("ok", f"{what} {leg.name}: {counts}; every sample classified, status ok", ok=True,
                    counts=base)

    d = _regrade_dir(ctx, leg)
    rec, res = _read_regrade(d)
    head = f"{what} {leg.name}: {len(failed)} sample(s) were GRADER FAILED ({counts})"
    if isinstance(rec, dict) and rec.get("unreadable"):
        return done("regrade_record_unreadable",
                    f"{head}. {d / REGRADE_RECORD} is unreadable, so which re-grades were dispatched cannot "
                    f"be told and none is started. HALT: a person reads {d}.", dir=str(d))
    sent = _dispatched(rec)
    rows = []
    for s in failed:
        path, why = _saved_output(s)
        row = {"task": s["task"], "sample": s["sample"], "first": s["reason"],
               "recorded": s["recorded"], "job_id": _regrade_job_id(s["task"], s["sample"]),
               "output": str(path) if path else None, "snapshot": s.get("snapshot")}
        line = res.get(row["job_id"])
        if path is None:
            row.update(outcome="cannot", why=why)
        elif line is None:
            row["output_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            row["outcome"], row["why"] = ("lost", REGRADE_LOST) if row["job_id"] in sent \
                else ("missing", "no result line")
        else:
            row["output_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            try:
                want = _regrade_sources(s["task"], ctx.run_dir)
            except ValueError as e:
                row.update(outcome="invalid", why=f"the job's reference / input cannot be stated: {e}")
            else:
                row["outcome"], row["why"] = _regrade_outcome(s["task"], row["output_sha256"], line, want)
            # The scorer's caught-exception warnings are not counted as grader failed (only the failure
            # warnings are); they are recorded per re-grade so a person can see them.
            row["scorer_exception_warnings"] = len(line.get("scorer_exception_warnings") or [])
        rows.append(row)
    out.update(rows=rows, dir=str(d))
    n = len(rows)

    def ids(xs):
        return ", ".join(f"{x['task']} s{x['sample']} ({x.get('why') or x['first']})" for x in xs[:4]) + \
            (f" (+{len(xs) - 4} more)" if len(xs) > 4 else "")
    by = {k: [x for x in rows if x["outcome"] == k] for k in
          ("cannot", "grader_failed", "invalid", "lost", "missing", *REGRADABLE)}
    if by["cannot"]:
        return done("cannot_regrade", f"{head}, and {len(by['cannot'])} of them cannot be re-graded: "
                                      f"{ids(by['cannot'])}. The re-grade reads the saved workbook; with "
                                      f"none there is nothing to re-grade. HALT: a person decides.")
    if by["grader_failed"]:
        return done("regrade_failed", f"{head}. RE-GRADE FAILED AGAIN for {len(by['grader_failed'])}: "
                                      f"{ids(by['grader_failed'])} (results in {d}). The loop re-grades "
                                      f"once and halts if the re-grade fails too; no number is reported.")
    if by["invalid"]:
        return done("regrade_invalid", f"{head}. {len(by['invalid'])} re-grade result(s) in {d} are not a "
                                       f"grade of the saved workbook by the grading sandbox: "
                                       f"{ids(by['invalid'])}. HALT: a person reads {d} before anything "
                                       f"is re-run.")
    if by["lost"]:
        inv = (rec.get("invocations") if isinstance(rec, dict) else None) or []
        rcs = ", ".join(f"invocation {i + 1}: driver exit {x.get('rc')} "
                        f"({x.get('rc_meaning', 'no exit recorded')})"
                        for i, x in enumerate(inv)) or "no invocation recorded"
        return done("regrade_lost", f"{head}. {len(by['lost'])} re-grade(s) were DISPATCHED and have NO "
                                    f"RESULT: " + ", ".join(f"{x['task']} s{x['sample']} ({x['job_id']})"
                                                            for x in by["lost"])
                    + f" -- {REGRADE_LOST} (each sample is re-graded ONCE; a lost failing re-grade graded "
                      f"again could come back passing). The loop recorded them as dispatched in "
                      f"{d / REGRADE_RECORD} before launching the driver ({rcs}). HALT: a person reads the "
                      f"driver log(s) and run_*/summary.json in {d} and decides; the loop never "
                      f"dispatches a job twice.")
    if by["missing"]:
        return done("needs_regrade", f"{head}. RE-GRADE NEEDED: {len(by['missing'])} of them have no "
                                     f"re-grade yet; this step's runner re-grades each ONCE from its "
                                     f"saved workbook in the grading sandbox (scripts/daytona/"
                                     f"ssb_grade_daytona.py, work dir {d}) and halts if it fails again.")

    def rec_pass(x):
        return x["recorded"] is not None and x["recorded"] >= _oc.PASS_THRESHOLD
    flips = [x for x in rows if (x["outcome"] == "passed") != rec_pass(x)]
    if flips:
        return done("regrade_flip", (
            f"{head}. The ONE re-grade from the saved workbook CHANGES the recorded verdict of "
            f"{len(flips)} sample(s): " + "; ".join(
                f"{x['task']} sample {x['sample']}: recorded verdict {'PASS' if rec_pass(x) else 'FAIL'} "
                f"(recorded score {x['recorded']}; first grade {x['first']}), re-grade verdict "
                f"{'PASS' if x['outcome'] == 'passed' else 'FAIL'} ({x['outcome']}, job {x['job_id']})"
                for x in flips)
            + f". Evidence: the re-grade lines in {d / 'results.jsonl'}, the first grade in each sample's "
              f"snapshot. The leg JSON is the record every downstream number reads (pairing, reconcile, "
              f"the prompt figures, the gate) and the loop never rewrites a recorded verdict. HALT -- THE "
              f"OPERATOR DECIDES, for each sample named: EITHER (a) a correction of the recorded verdict "
              f"written by a person (the loop writes none), OR (b) a re-run of "
              f"the leg {leg.name} (a re-run leg has new bytes and gets a fresh re-grade dir). Until one "
              f"is done this step halts at every invocation and no second re-grade is started."))
    c = dict(base)
    c["grader_failed"] -= n
    c["passed"] += len(by["passed"])
    c["graded_wrong"] += len(by["graded_wrong"])
    c.update(regraded=n, regrade_dir=str(d),
             source="outcomes.leg_outcome_counts + the loop's re-grade (scripts/daytona/ssb_grade_daytona.py)")
    if c["grader_failed"] != 0 or c["passed"] != r["recorded_pass"] or \
            (r["recorded_sample_pass"] is not None and r["recorded_sample_pass"] != r["recorded_pass"]):
        return done("unreportable", f"{head}; after the re-grade the counts ({_counts_text(c)}) still do not "
                                    f"match the leg's recorded {r['recorded_pass']} passes. HALT.")
    openq = sum(1 for x in rows if x.get("scorer_exception_warnings"))
    return done("ok", (f"{what} {leg.name}: {_counts_text(c)} after the re-grade: {n} grader-failed "
                       f"sample(s) re-graded ONCE from their saved workbooks ({len(by['passed'])} passed, "
                       f"{len(by['graded_wrong'])} graded and wrong, each the recorded verdict; {openq} of the "
                       f"re-grades logged scorer exception warnings, which are NOT counted as grader failed "
                       f"(recorded for a person to read); results in {d}); status ok"), ok=True, counts=c)


def _outcomes_verdict(ctx: Ctx, leg, what: str):
    """One leg's outcomes, as a verify: _leg_outcomes_after_regrade's (ok, detail)."""
    st = _leg_outcomes_after_regrade(ctx, leg, what)
    return st["ok"], st["detail"]


def _grade_driver_python() -> str:
    from .host_tools import python_executable
    return python_executable()


def _run_regrade(ctx: Ctx, leg, what: str) -> None:
    """THE RE-GRADE. Every grader-failed sample of `leg` that has no re-grade yet is graded
    once more, from its saved workbook, by scripts/daytona/ssb_grade_daytona.py (one grading sandbox,
    deleted by id when it ends; ledgered as driver ssb_grade_daytona). Job ids are .../regrade1.

    ONCE, ALSO ACROSS A CRASH. Every job is recorded as DISPATCHED in REGRADE.json
    (written whole, fsynced) BEFORE the driver is launched, and only a job never dispatched is handed to
    it. A dispatched job with no result line afterwards is "dispatched, no result": the verify HALTs on it
    and this runner never sends it again, so a lost failing re-grade cannot be retried until it passes.
    Any other state (a re-grade that failed again, a flip, an invalid line, an unreadable or unreportable
    leg, a sample with no saved workbook) is a HALT here: "halt if it still fails"."""
    st = _leg_outcomes_after_regrade(ctx, leg, what)
    if st["ok"]:
        return
    if st["state"] != "needs_regrade":
        raise Halt(st["detail"])
    if not GRADE_DRIVER.is_file():
        raise Halt(f"{what}: the grading-sandbox driver {GRADE_DRIVER} does not exist; cannot re-grade")
    d = Path(st["dir"])
    d.mkdir(parents=True, exist_ok=True)
    rows = [x for x in st["rows"] if x["outcome"] == "missing"]
    leg = Path(leg)
    for x in rows:
        try:
            x.update(_regrade_sources(x["task"], ctx.run_dir))
        except ValueError as e:
            raise Halt(f"{what}: the reference / input a re-grade of {x['task']} s{x['sample']} must use "
                       f"cannot be stated ({e}); nothing dispatched")
    # gt_sha256 / input_sha256 state the job's sources: the driver recomputes its own (normalize_jobs, the
    # same overlay) and the worker grades against those; the verify checks the result's against these.
    jobs = [{"job_id": x["job_id"], "mode": "grade", "task_id": x["task"], "output": x["output"],
             "output_sha256": x["output_sha256"], "gt_sha256": x["gt_sha256"],
             "input_sha256": x["input_sha256"],
             "meta": {"kind": "loop_d4_regrade", "leg": str(leg), "sample": x["sample"],
                      "first_grade": x["first"], "recorded": x["recorded"]}} for x in rows]
    jf = d / "jobs.jsonl"
    jtext = "".join(json.dumps(j, sort_keys=True) + "\n" for j in jobs)
    jf.write_text(jtext)
    rp = d / REGRADE_RECORD
    rec, _res = _read_regrade(d)
    if not isinstance(rec, dict):
        rec = {"schema": REGRADE_SCHEMA, "run_dir": str(Path(ctx.run_dir).resolve()), "epoch": ctx.epoch,
               "what": what, "leg_json": str(leg.resolve()),
               "leg_sha256": hashlib.sha256(leg.read_bytes()).hexdigest(),
               "policy": "re-grade a grader-failed sample ONCE from its saved workbook; halt if it "
                         "fails again, and halt on a dispatched re-grade with no result",
               "attempts_per_sample": _regrade_attempts(), "invocations": []}
    sent = dict(_dispatched(rec))
    smp = {s.get("job_id"): s for s in rec.get("samples") or []}
    smp.update({x["job_id"]: {k: x.get(k) for k in ("task", "sample", "first", "recorded", "job_id", "output",
                                                    "output_sha256", "gt_sha256", "input_sha256")}
                for x in rows})
    rec["samples"] = list(smp.values())
    argv = [_grade_driver_python(), str(GRADE_DRIVER), "run", "--jobs", str(jf), "--out-dir", str(d),
            "--run-dir", str(Path(ctx.run_dir).resolve()), "--label", _regrade_label(leg),
            "--wall-min", str(REGRADE_WALL_MIN), "--max-usd", str(REGRADE_MAX_USD)]
    inv = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "argv": argv,
           "jobs_sha256": hashlib.sha256(jtext.encode()).hexdigest(), "n_jobs": len(jobs),
           "n_missing": len(rows)}
    rec["invocations"].append(inv)
    sent.update({j["job_id"]: {"invocation": len(rec["invocations"]), "utc": inv["started_utc"],
                               "output_sha256": j["output_sha256"]} for j in jobs})
    rec["dispatched"] = sent
    _write_regrade_record(rp, rec)                      # DISPATCHED, on disk, before the driver starts
    # No model key travels from here (the driver refuses one); only the Daytona variables, which the driver
    # reads from the environment or the repo's .env.
    env = {k: v for k, v in os.environ.items() if k.startswith("DAYTONA_") or not _KEYLIKE_ENV.search(k)}
    log = d / f"driver.{inv['started_utc'].replace(':', '')}.out"
    print(f"[regrade] {what}: {len(jobs)} job(s) -> {' '.join(argv[1:3])} ... (log {log})", file=sys.stderr)
    try:
        with open(log, "w") as fh:
            rc = subprocess.run(argv, cwd=str(REPO), env=env, stdout=fh, stderr=subprocess.STDOUT,
                                timeout=(REGRADE_WALL_MIN + 45) * 60).returncode
    except subprocess.TimeoutExpired:
        rc = None
    except OSError as e:                                # the interpreter could not be started at all
        inv.update(ended_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), rc=None,
                   rc_meaning=f"not started: {type(e).__name__}: {e}", log=str(log))
        _write_regrade_record(rp, rec)
        raise Halt(f"{what}: the grading-sandbox driver could not be started ({type(e).__name__}: {e}). "
                   f"Its {len(jobs)} job(s) stay recorded as dispatched in {rp}: a person confirms no "
                   f"driver ran before anything is re-graded.")
    inv.update(ended_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), rc=rc,
               rc_meaning=GRADE_DRIVER_EXITS.get(rc, "timeout" if rc is None else "unknown"), log=str(log))
    _write_regrade_record(rp, rec)
    if rc != 0:
        tail = log.read_text(errors="replace")[-1200:] if log.is_file() else ""
        raise Halt(f"{what}: the grading-sandbox re-grade exited {rc} ({inv['rc_meaning']}); results so far "
                   f"are kept in {d}. Every job it was given is recorded as dispatched, so one with no "
                   f"result now HALTs ({REGRADE_LOST}) instead of being graded again.\n{tail}")


def _v_batch_outcomes(ctx: Ctx):
    """Before the prompt is rendered: the batch the newest evidence describes has no grader
    failure left after its re-grade, and its four outcomes are what the prompt will state."""
    return _outcomes_verdict(ctx, _batch_leg(ctx), "newest batch")


def _run_batch_regrade(ctx: Ctx) -> None:
    _run_regrade(ctx, _batch_leg(ctx), "newest batch")


def _v_leg_outcomes(ctx: Ctx, key: str):
    """After this epoch's leg ran and BEFORE any number of it is read (pairing, reconcile, the gate):
    no grader failure left after its re-grade."""
    return _outcomes_verdict(ctx, ctx.paths.leg(key), f"{key} leg")


def _run_leg_regrade(ctx: Ctx, key: str) -> None:
    _run_regrade(ctx, ctx.paths.leg(key), f"{key} leg")


def _v_operator_turns_recorded(ctx: Ctx):
    """Template mode's replacement for "operator<->AO conversation is recorded".

    WHY IT IS REPLACED, NOT KEPT. That check asks for a two-sided exchange THIS epoch. A template-mode
    epoch calls no AO (the AO worked once, at setup), so it could only pass for the wrong reason: it
    counts the epoch-less intake turn and the loop's own report turns, which conversation.record files
    under speaker "ao". What IS true and checkable here: the operator's side of this epoch is recorded
    (main() writes the invocation before any verify under --step), the transcript is uncorrupt, and no
    AO AUTHORING turn exists for this epoch (one would mean an AO wrote a prompt in a run that declared
    none). The loop's report/ask turns are not an AO speaking and are not counted as one."""
    from src.orchestrator import conversation
    p = conversation.path_for(ctx.run_dir)
    ro = getattr(ctx, "readonly", False)
    if not p.is_file():
        if ro:
            return True, (f"{UNMEASURED}no transcript at {p.name} and this is a read-only plan, which "
                          f"records nothing by design. Re-check under --step.")
        return False, (f"NO TRANSCRIPT at {p}. This invocation recorded the operator's turn before running "
                       f"any check, so the file should exist; its absence means the write failed.")
    turns = conversation.read(ctx.run_dir)
    bad = [t for t in turns if (t.get("meta") or {}).get("unparseable")]
    if bad:
        return False, (f"{len(bad)} unparseable line(s) in {p.name}: the transcript is corrupt, first: "
                       f"{bad[0]['content'][:160]}")
    mine = [t for t in turns if t.get("epoch") == ctx.epoch]
    # A run that switched from prompt_mode "ao" to "template" keeps its AO-era turns as history:
    # the operator records the switch as a "note" turn with meta.mode_switch == "template", and only AO
    # authoring turns AFTER the latest such switch count here. Earlier ones are reported, not hidden.
    switches = [int(t.get("ts_ns") or 0) for t in turns if t.get("channel") == "note"
                and (t.get("meta") or {}).get("mode_switch") == "template"]
    since = max(switches) if switches else 0
    all_authored = [t for t in mine if t.get("speaker") == "ao" and t.get("channel") == "authoring"]
    authored = [t for t in all_authored if int(t.get("ts_ns") or 0) > since]
    superseded = len(all_authored) - len(authored)
    if authored:
        return False, (f"{len(authored)} AO AUTHORING turn(s) are recorded for epoch {ctx.epoch}, but RUNTIME "
                       f"prompt_mode=template has no per-epoch AO: something other than the loop wrote "
                       f"this epoch's prompt. Read {p} before continuing.")
    ops = [t for t in mine if t.get("speaker") == "operator"]
    if not ops:
        # A single `--drive --until-epoch K` invocation is recorded once, under its starting epoch, and then
        # runs epochs up to K: that operator turn covers this epoch too.
        for t in turns:
            m = t.get("meta") or {}
            if t.get("speaker") != "operator" or t.get("channel") != "invocation" or m.get("mode") != "drive":
                continue
            argv = str(m.get("argv") or "").split()
            until = argv[argv.index("--until-epoch") + 1] if "--until-epoch" in argv[:-1] else ""
            if isinstance(t.get("epoch"), int) and t["epoch"] <= ctx.epoch and until.isdigit() \
                    and int(until) >= ctx.epoch:
                ops.append(t)
    if not ops:
        if ro:
            return True, (f"{UNMEASURED}no operator turn for epoch {ctx.epoch} yet, and a read-only plan "
                          f"records none. Re-check under --step.")
        return False, (f"no operator turn is recorded for epoch {ctx.epoch} in {p.name}, though this "
                       f"invocation records one before any check: the write failed.")
    chans = sorted({t.get("channel") for t in ops})
    return True, (f"prompt_mode=template: no AO is called per epoch, so the two-sided operator<->AO "
                  f"check does not apply (the AO worked once, at setup). {len(ops)} operator turn(s) for "
                  f"epoch {ctx.epoch} are recorded (channels {chans}); 0 AO authoring turns"
                  + (f" since the recorded switch to template mode ({superseded} earlier AO-era authoring "
                     f"turn(s) kept as history)" if superseded else "")
                  + "; the loop's own report/ask turns are not counted as an AO")


def _v_references_train_only(ctx: Ctx):
    """With RUNTIME evolver_daytona.train_references on, <run>/evolver_references_train holds the
    reference workbook of every TRAIN task and nothing else, re-derived from the dataset and the split
    (daytona_evolver.verify_references_dir: no val/test golden, no input, byte-equal to each train
    task's golden). Off: nothing is built, uploaded or named."""
    if not _grants(ctx)["train_references"]:
        return True, ("train_references not granted (RUNTIME evolver_daytona): no reference workbook "
                      "is uploaded or named")
    from src.orchestrator.daytona_evolver import verify_references_dir
    c = verify_references_dir(ctx.run_dir, Path(ctx.spec.split_path))
    return c.status == "PASS", c.detail


def _run_build_references(ctx: Ctx) -> None:
    if not _grants(ctx)["train_references"]:
        return
    from src.orchestrator.daytona_evolver import build_references_dir
    build_references_dir(ctx.run_dir, Path(ctx.spec.split_path))


def _v_grader_code_only(ctx: Ctx):
    """The scoring rule + clean grader. Every rendered prompt names <run>/grader_clean, so the
    run must grant it (RUNTIME evolver_daytona.grader_clean) and it must be code only, as its
    MANIFEST.json says (daytona_evolver.verify_grader_dir: clean shas, no workbook, no held-out id,
    no dataset byte stream), and that MANIFEST must be the one the frozen plug-in pins. No runner:
    scripts/make_clean_grader.py builds the dir, before the freeze."""
    if not _grants(ctx)["grader_clean"]:
        return False, ("prompt_mode=template names the clean grader in every evolver prompt, so "
                       "RUNTIME.json evolver_daytona.grader_clean must be true, and it is not: the "
                       "evolver would be told about a dir it never receives.")
    from src.orchestrator.daytona_evolver import grader_dir, sha256_file, verify_grader_dir
    c = verify_grader_dir(ctx.run_dir, Path(ctx.spec.split_path))
    if c.status != "PASS":
        return False, c.detail
    man = grader_dir(ctx.run_dir) / "MANIFEST.json"
    pj = Path(ctx.run_dir) / "setup" / "PLUGIN.json"
    try:
        pin = json.loads(pj.read_text()).get("grader_clean_manifest") or {}
    except (OSError, json.JSONDecodeError) as e:
        return False, f"cannot read the frozen plug-in record {pj} ({type(e).__name__})"
    if str(Path(str(pin.get("path") or "")).resolve()) != str(man.resolve()) \
            or pin.get("sha256") != sha256_file(man):
        return False, (f"{man} is not the grader_clean MANIFEST the frozen plug-in pins "
                       f"({pin.get('path')} sha256:{str(pin.get('sha256'))[:12]}); re-freeze after "
                       f"rebuilding the clean grader")
    return True, c.detail + "; its MANIFEST is the one setup/PLUGIN.json pins"


def _render_prompt_parts(ctx: Ctx) -> tuple:
    """(parts, record) for this epoch's template-mode prompt. Everything the render reads is resolved
    HERE, once, so the runner and the verify cannot render from different inputs:
      the AO paragraph = the EVOLVER_TEMPLATE.md the frozen setup/PLUGIN.json pins (verify_frozen), if any;
      the facts       = prompt_author.derive_facts over the newest digest dir, cross-checked against
                        the leg it digests, with the EVIDENCE INDEX in all-epochs mode (the deriver
                        the prompt-figures check and the launcher's prompt gate re-derive from);
      the outcomes    = outcomes.leg_outcome_counts of that leg (the renderer refuses grader failed);
      the machine     = the image recipe re-hashed + the base leg's runtime_env;
      the flags       = the operator's grants (RUNTIME evolver_daytona), the upload's own source.
    Raises Halt (or evolver_template.RenderError) when the prompt cannot be rendered truthfully."""
    from src.orchestrator import evolver_template as _et
    from src.orchestrator import plugin_runner as _pr
    rd = Path(ctx.run_dir)
    try:
        frozen = _pr.verify_frozen(rd)
    except _pr.PluginDrift as e:
        raise Halt(f"the benchmark plug-in is not frozen: {e}")
    rec = json.loads((rd.resolve() / "setup" / "PLUGIN.json").read_text())
    # The benchmark adaptor writes no EVOLVER_TEMPLATE.md; a record that pins none renders the verdict section
    # from the loop's own text. A record that pins one must pin this run's file.
    tpl_pin = ((rec.get("files") or {}).get(TEMPLATE_PARAGRAPH) or {}).get("path")
    tpl = Path(str(tpl_pin)) if tpl_pin else None
    if tpl is not None and tpl.resolve() != (rd.resolve() / "setup" / TEMPLATE_PARAGRAPH):
        raise Halt(f"setup/PLUGIN.json pins the template paragraph at {tpl}, not this run's "
                   f"setup/{TEMPLATE_PARAGRAPH}")
    g = _grants(ctx)
    if not g["grader_clean"]:
        raise Halt("RUNTIME.json evolver_daytona.grader_clean is not true, and every rendered prompt "
                   "names the clean grader")
    if not ctx.base_leg:
        raise Halt("no base_leg declared (RUNTIME.json base_leg): the epoch-1 facts and the EVIDENCE "
                   "INDEX need it")
    leg = _batch_leg(ctx)
    if leg is None or not Path(leg).is_file():
        raise Halt(f"the leg the newest evidence digests is absent: {leg}")
    # The facts, the EVIDENCE INDEX and the outcomes are read from the base leg and the batch leg; both
    # must be the run's declared split's train tasks (a leg stating no task_ids is not judged here).
    for what, p in (("base_leg", Path(ctx.base_leg)), ("the newest batch's leg", Path(leg))):
        if p.is_file():
            st, det = _leg_on_split(p, ctx.spec.split_path, "train", getattr(ctx.spec, "n_samples", None))
            if st == "bad":
                raise Halt(f"{what} is not on the run's split, so no figure of it may reach the prompt: {det}")
    # The four outcomes AFTER the re-grade of any grader-failed sample. The renderer refuses
    # any grader failure left in them, so a failed / missing / verdict-changing re-grade stops here.
    oc = _leg_outcomes_after_regrade(ctx, leg, "newest batch")
    if not oc["ok"]:
        raise Halt(oc["detail"])
    all_epochs = _evidence_mode(ctx) == "all_epochs"
    facts = _et.facts_for_epoch(rd, ctx.epoch, base_leg=Path(ctx.base_leg), all_epochs=all_epochs,
                                variant=getattr(ctx.spec, "variant", "") or "")
    mf = _et.machine_facts_for(rd, split_path=ctx.spec.split_path, base_leg=Path(ctx.base_leg),
                               train_inputs=g["train_inputs"])
    prior = EpochPaths(rd, ctx.epoch - 1).promoted if ctx.epoch > 1 else None
    # With RUNTIME evolver_evidence_layout=rollouts the prompt carries the skill-set TABLE (not the skills'
    # text), each epoch's skill snapshot, the newest batch's per-rollout scores from its INDEX.json (the
    # prompt-figures check re-derives them) and the trace layout. The re-grade rows travel with the
    # counts so a re-graded sample shows its re-grade's outcome.
    layout = _evidence_layout(ctx)
    parts = _et.render_parts(rd, ctx.epoch, evidence_dirs=_evidence_dirs(ctx),
                             staging_dir=ctx.paths.staging, prior_library=prior, facts=facts,
                             template_paragraph_path=tpl, machine_facts=mf,
                             train_references=g["train_references"], train_inputs=g["train_inputs"],
                             leg_json=leg, changed_cells=g["changed_cells"],
                             require_full_grader_output=True, outcomes=oc["counts"],
                             evidence_layout=layout,
                             regraded=([{k: x.get(k) for k in ("task", "sample", "outcome")}
                                        for x in oc.get("rows") or []] if layout == "rollouts" else None))
    block = facts.as_block()
    inf = rd / "INFRA.json"
    record = {
        "mode": "template", "epoch": ctx.epoch, "model_call": False,
        "what": ("ITER{N}_PROMPT.md rendered by the loop (src/orchestrator/evolver_template.py) from "
                 "fixed loop text + the frozen AO paragraph + slots; no model was called"),
        "renderer": {"module": "src/orchestrator/evolver_template.py",
                     "sha256": hashlib.sha256(Path(_et.__file__).read_bytes()).hexdigest()},
        "template_paragraph": ({"path": str(tpl),
                                "file_sha256": hashlib.sha256(tpl.read_bytes()).hexdigest(),
                                "paragraph_sha256": parts["template_paragraph_sha256"]}
                               if tpl is not None else None),
        "plugin_record_sha256": frozen.get("record_sha256"),
        "infra_sha256": hashlib.sha256(inf.read_bytes()).hexdigest() if inf.is_file() else None,
        "facts_sha256": hashlib.sha256(block.encode("utf-8")).hexdigest(),
        "facts": block,
        "leg_json": str(leg),
        "outcome_counts": parts["outcome_counts"], "outcome_source": parts["outcome_source"],
        "grants": g,
        "evidence": parts["evidence"], "grader_dir": parts["grader_dir"],
        "references": parts["references"], "inputs": parts["inputs"],
        "staging": parts["staging"], "prior_library": parts["prior_library"],
        "changed_cells_dirs": parts.get("changed_cells_dirs"),
        # Every evidence dir's verified DIGEST_INFO.json (path, header sha, full/clipped grader
        # output, changed cells on/off, tool clips) and the renderer's flags, as the render used them.
        "digest_info": parts.get("digest_info"), "flags": parts.get("flags"),
        "grader_tooling_absent": parts.get("grader_tooling_absent"),
        # The rollouts layout's inputs (absent in the digests layout, so a digests-layout
        # record is unchanged): every granted dir's INDEX.json sha, the per-rollout table's size.
        **({"evidence_layout": layout, "rollout_index": parts.get("rollout_index"),
            "rollout_table": parts.get("rollout_table"), "trace_stats": parts.get("trace_stats")}
           if layout == "rollouts" else {}),
        # Which samples of the batch were re-graded, and where the grading sandbox's results are.
        "regrade": {"n": (oc["counts"] or {}).get("regraded", 0), "dir": oc.get("dir"),
                    "samples": [{k: x.get(k) for k in ("task", "sample", "first", "outcome", "job_id",
                                                       "output_sha256", "scorer_exception_warnings")}
                                for x in oc.get("rows") or []]},
        "prompt": {"path": str(ctx.paths.prompt), "sha256": parts["sha256"], "bytes": parts["bytes"]},
    }
    return parts, record


def _run_render_prompt(ctx: Ctx) -> None:
    """THE LOOP RENDERS THE EVOLVER'S PROMPT. No model call. Refuses rather
    than overwrites: an existing prompt or render record is evidence a person may need."""
    from src.orchestrator.evolver_template import RenderError
    p = ctx.paths.prompt
    rec_p = Path(ctx.run_dir) / RENDER_RECORD.format(epoch=ctx.epoch)
    for f in (p, rec_p, Path(ctx.run_dir) / f"e{ctx.epoch}_PROMPT_AUTHOR.json"):
        if f.exists():
            raise Halt(f"refusing to render over {f.name}, which exists. Move {p.name}, {rec_p.name} "
                       f"and any e{ctx.epoch}_PROMPT_AUTHOR.json aside (keep them) and re-run --step.")
    try:
        parts, record = _render_prompt_parts(ctx)
    except RenderError as e:
        raise Halt(f"the evolver prompt cannot be rendered truthfully: {e}")
    record["rendered_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(p, "x", encoding="utf-8") as fh:
        fh.write(parts["text"])
    tmp = rec_p.with_name(f".{rec_p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(record, indent=1, default=str))
    os.replace(str(tmp), str(rec_p))


def _v_prompt_rendered(ctx: Ctx):
    """The prompt exists, the loop's render record is beside it, the prompt is byte-for-byte what
    the record says, and RE-RENDERING IT NOW from what is on disk gives the same bytes. So an edit
    after the render, a changed grant, a changed paragraph or plug-in, or a changed batch all fail
    here, before any downstream check reads the file. An AO-authored record for
    this epoch is refused: the run declared a world with no per-epoch AO."""
    from src.orchestrator.evolver_template import RenderError
    p = ctx.paths.prompt
    rd = Path(ctx.run_dir)
    ao = rd / f"e{ctx.epoch}_PROMPT_AUTHOR.json"
    if ao.is_file():
        return False, (f"{ao.name} exists: this epoch's prompt was authored by the AO, but RUNTIME "
                       f"prompt_mode=template. Move {p.name} and {ao.name} aside (keep them) and "
                       f"re-run --step.")
    if not p.is_file():
        return False, f"no prompt at {p.name} — the loop has not rendered it yet"
    rec_p = rd / RENDER_RECORD.format(epoch=ctx.epoch)
    if not rec_p.is_file():
        return False, (f"{p.name} exists but there is no {rec_p.name}: a prompt the loop cannot "
                       f"attribute to a render. Move it aside and re-run --step.")
    try:
        rec = json.loads(rec_p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return False, f"{rec_p.name} is unreadable ({type(e).__name__})"
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    if rec.get("mode") != "template" or rec.get("model_call") is not False:
        return False, f"{rec_p.name} is not a template render record (mode {rec.get('mode')!r})"
    if (rec.get("prompt") or {}).get("sha256") != actual:
        return False, (f"{p.name} hashes to {actual[:12]} but {rec_p.name} recorded "
                       f"{str((rec.get('prompt') or {}).get('sha256'))[:12]}: edited after the render")
    try:
        parts, now = _render_prompt_parts(ctx)
    except (Halt, RenderError) as e:
        return False, f"the prompt no longer re-renders from what is on disk: {e}"
    if parts["sha256"] != actual:
        changed = [k for k in ("facts_sha256", "template_paragraph", "grants", "outcome_counts",
                               "evidence", "references", "inputs", "prior_library", "plugin_record_sha256",
                               "digest_info", "regrade", "evidence_layout", "rollout_index",
                               "rollout_table", "trace_stats")
                   if rec.get(k) != now.get(k)]
        return False, (f"re-rendering now gives {parts['sha256'][:12]}, but {p.name} is {actual[:12]}: "
                       f"an input changed since the render ({changed or 'the fixed loop text'}). Move "
                       f"the prompt and {rec_p.name} aside and re-run --step.")
    tp = now.get("template_paragraph")
    return True, ("rendered by the loop (no model call) from "
                  + (f"{TEMPLATE_PARAGRAPH} sha256:{tp['paragraph_sha256'][:12]}" if tp
                     else "the loop's own text (no adapter paragraph)")
                  + f" + facts sha256:{now['facts_sha256'][:12]}, {parts['bytes']} B; re-renders byte-identically "
                  f"now; outcomes {({k: now['outcome_counts'].get(k) for k in ('passed', 'graded_wrong', 'not_eligible', 'grader_failed')})}")


def _dir_mention_problems(text: str, dirname: str, absd: str, label: str, *, subpaths: bool) -> list:
    """Every mention of `dirname` must be this run's absolute `absd`, never a sibling sharing its name
    prefix, and, with subpaths=True, never a concrete subpath below it (a task id)."""
    bad = []
    for m in re.finditer(re.escape(dirname), text):
        if not text[:m.end()].endswith(absd):
            bad.append(f"names the {label} without this run's absolute path")
        nxt = text[m.end():m.end() + 2]
        if nxt[:1] and (nxt[:1].isalnum() or nxt[:1] in "_-"):
            bad.append(f"names a sibling that shares the {label}'s name prefix")
        if subpaths and len(nxt) == 2 and nxt[0] == "/" and not nxt[1].isspace() \
                and nxt[1] not in "<*`'\")":
            bad.append(f"names a subpath below the {label} (a task)")
    return bad


def _v_prompt_names_granted_dirs(ctx: Ctx):
    """Template mode, for every granted dir: the prompt names exactly what the upload carries:
    each GRANTED dir (inputs, training references, clean grader) by this run's absolute path, never
    a per-task subpath of the inputs or references dir; no dir the run does not grant; no dataset or
    golden path. The AO-mode inputs check (_v_prompt_names_inputs) assumed inputs always travel."""
    from src.orchestrator.daytona_evolver import (GRADER_DIRNAME, INPUTS_DIRNAME, REFERENCES_DIRNAME,
                                                  data_root, grader_dir, inputs_dir, references_dir)
    pr = ctx.paths.prompt
    if not pr.is_file():
        return False, f"no prompt at {pr.name}"
    text = pr.read_text(errors="replace")
    g = _grants(ctx)
    bad, named = [], []
    for key, dirname, absd, label, subs in (
            ("train_inputs", INPUTS_DIRNAME, str(inputs_dir(ctx.run_dir)), "inputs dir", True),
            ("train_references", REFERENCES_DIRNAME, str(references_dir(ctx.run_dir)),
             "references dir", True),
            ("grader_clean", GRADER_DIRNAME, str(grader_dir(ctx.run_dir)), "clean grader dir", False)):
        if g.get(key):
            if absd not in text:
                bad.append(f"never names the granted {label} {absd}")
            else:
                named.append(dirname)
            bad += _dir_mention_problems(text, dirname, absd, label, subpaths=subs)
        elif dirname in text:
            bad.append(f"names the {label} ({dirname}), which RUNTIME does not grant: the evolver "
                       f"will not have it")
    low = text.lower()
    if "spreadsheetbench" in low or "_golden." in low or str(data_root()).lower() in low:
        bad.append("names a dataset/golden path")
    if bad:
        return False, ("; ".join(sorted(set(bad))) + f". REMEDY: move {pr.name} and "
                       f"{RENDER_RECORD.format(epoch=ctx.epoch)} aside (keep them), fix the grant or "
                       f"the dir, and re-run --step.")
    return True, (f"prompt names exactly the granted dirs {named} by absolute path, no task subpath, "
                  f"no ungranted dir, no dataset path")


def _v_prompt_names_inputs(ctx: Ctx):
    """daytona host: the prompt names the inputs dir, only by this run's absolute path, never a
    subpath below it (that would be a task id), and no dataset path at all."""
    if getattr(ctx, "evolver_host", "local") != "daytona":
        return True, "local evolver host: no inputs dir to name"
    from src.orchestrator.daytona_evolver import INPUTS_DIRNAME, inputs_dir
    pr = ctx.paths.prompt
    if not pr.is_file():
        return False, f"no prompt at {pr.name}"
    text = pr.read_text()
    absd = str(inputs_dir(ctx.run_dir))
    bad = []
    if absd not in text:
        bad.append(f"never names {absd}")
    from src.orchestrator.daytona_evolver import data_root
    for m in re.finditer(re.escape(INPUTS_DIRNAME), text):
        if not text[:m.end()].endswith(absd):
            bad.append("names the inputs dir without this run's absolute path")
        nxt = text[m.end():m.end() + 2]
        if nxt[:1] and (nxt[:1].isalnum() or nxt[:1] in "_-"):   # '.' ends sentences; not a sibling
            bad.append("names a sibling that shares the inputs dir's name prefix")
        if len(nxt) == 2 and nxt[0] == "/" and not nxt[1].isspace() and nxt[1] not in "<*`'\")":
            bad.append("names a subpath below the inputs dir (a task)")
    low = text.lower()
    if "spreadsheetbench" in low or "_golden." in low or str(data_root()).lower() in low:
        bad.append("names a dataset/golden path")
    if bad:
        return False, "; ".join(sorted(set(bad)))
    return True, f"prompt names {INPUTS_DIRNAME} by absolute path only, no subpath, no dataset path"


def _evidence_dirs(ctx: Ctx) -> list:
    """The evidence dirs THIS epoch's evolver is granted: [digests_eK] or digests_e1..eK; under the evo layout
    [evo_e{K-1}] or evo_e0..evo_e{K-1}."""
    if _evidence_layout(ctx) == "evo":
        return ctx.paths.evo_dirs() if _evidence_mode(ctx) == "all_epochs" else [ctx.paths.evo]
    if _evidence_mode(ctx) == "all_epochs":
        return ctx.paths.evidence_dirs()
    return [ctx.paths.digests]


def _v_evidence_train_only(ctx: Ctx):
    """All-epochs mode: EVERY granted digest dir must be this run's own, real (no symlink, no
    symlinked file), non-empty and train-only. An earlier dir was verified at its own epoch
    only; anything added since — a hand rebuild, a copy, a retrofit — would otherwise reach
    every later evolver unchecked. Rules: src/orchestrator/evidence_surface.py (the launcher
    applies the same function to the list it is handed)."""
    from src.orchestrator import evidence_surface as _es
    rd = Path(ctx.run_dir)
    bad, lines = [], []
    for j in range(1, ctx.epoch + 1):
        d = EpochPaths(rd, j).digests
        probs = _es.surface_problems(d, ctx.spec.split_path)
        if not probs and not _es._inside(os.path.realpath(d), os.path.realpath(rd)):
            probs = [f"{d.name} resolves OUTSIDE the run dir"]
        if not probs:
            probs = _layout_missing(ctx, d, j)
        if probs:
            bad.append(f"{d.name}: " + "; ".join(probs))
        else:
            lines.append(f"{d.name} {_es.n_task_files(d)}")
    if bad:
        return False, "all-epochs evidence surface: " + " | ".join(bad)
    return True, (f"all-epochs evidence, {len(lines)} dir(s), every one train-only with no "
                  f"symlinks: " + ", ".join(lines))


# ── verifies (each re-derives from the artifact) ─────────────────────────────────────────
def _v_digests_train_only(ctx: Ctx):
    """The evidence surface must contain train ids and nothing else. Re-derived by
    intersecting the digest filenames with the split's own val/test id lists.

    In all-epochs mode (RUNTIME evolver_evidence) it checks every granted dir; see
    _v_evidence_train_only. The single-dir path below is unchanged."""
    if _evidence_mode(ctx) == "all_epochs":
        return _v_evidence_train_only(ctx)
    d = ctx.paths.digests
    if not d.is_dir():
        return False, f"no digest dir at {d}"
    sp = json.loads(Path(ctx.spec.split_path).read_text())
    def ids(k):
        v = sp.get(k) or []
        return set(v if not v or isinstance(v[0], str) else [x["id"] for x in v])
    # Naming-agnostic: a benchmark may name its digests `task_*.md` or by bare task id (`5.md`), and
    # a glob for one convention would find an EMPTY set on the other and block a correct epoch.
    # Sidecars the digest builder writes alongside the per-task files are excluded by name.
    _SIDECAR = {"REGRESSIONS", "README", "INDEX"}
    have = {f.stem for f in d.glob("*.md")} - _SIDECAR

    # COMPARE IN FILENAME SPACE, NOT ID SPACE. SSB v2 ids carry a category ("Debugging/01_02")
    # and the builder writes "Debugging_01_02.md", so a raw set intersection against the split
    # matched NOTHING: all 60 digests of a PERFECTLY CLEAN train surface came back "stray", and
    # the message showed `sorted(stray)[:6]` — six names that are all genuinely train tasks,
    # which reads like a leak report and is the opposite of one. Fourth site of the naming
    # assumption this function's own comment warns about.
    #
    # The map runs id -> stem, never stem -> id. That direction is a FUNCTION; the reverse is
    # ambiguous, because category names contain underscores ("Financial_Model_01_02" could split
    # as Financial/Model_01_02 or Financial_Model/01_02) and guessing it would silently
    # mis-attribute a held-out task to train — the exact failure this guard exists to catch.
    # Collisions would break the reverse lookup the strictness depends on, so they are refused
    # rather than tolerated (0 across all 240 ids of the SSB v2 split; tau3 ids are unaffected
    # because they contain no "/").
    def stem(i: str) -> str:
        return i.replace("/", "_")
    every = [i for k in ("train", "val", "test") for i in ids(k)]
    dup = {stem(i) for i in every if sum(stem(j) == stem(i) for j in every) > 1}
    if dup:
        return False, (f"split ids collide when mapped to digest filenames: {sorted(dup)[:6]} — "
                       "cannot decide train-vs-held-out by filename")
    tr = {stem(i) for i in ids("train")}
    va = {stem(i) for i in ids("val")}
    te = {stem(i) for i in ids("test")}
    bad = (have & va) | (have & te)
    if bad:
        return False, f"HELD-OUT MATERIAL IN THE EVIDENCE SURFACE: {sorted(bad)[:6]}"
    if not have:
        return False, f"digest dir holds no per-task .md files (sidecars excluded: {sorted(_SIDECAR)})"
    # STRICTLY STRONGER than a val/test intersection: every digest must BE a train task. An id in
    # neither train nor val nor test — a stale file from another split, say — would pass an
    # intersection silently because it intersects nothing.
    stray = have - tr
    if stray:
        return False, f"digests present that are NOT train tasks: {sorted(stray)[:6]}"
    # The rollout evidence's subtree, and non-train task ids in ANY file's text (the full
    # traces carry the solver's reasoning, which the digests never did) -- evidence_surface's rules,
    # the same the all-epochs path applies through surface_problems.
    from src.orchestrator import evidence_surface as _es
    extra = _es.rollouts_problems(d, ctx.spec.split_path) + _es.content_problems(d, ctx.spec.split_path)
    extra = extra or _layout_missing(ctx, d, ctx.epoch)
    if extra:
        return False, "; ".join(extra)
    return True, (f"{len(have)} digests, all {len(have & tr)} train, 0 val, 0 test, 0 stray; 0 non-train "
                  f"ids in any file" + ("; rollout evidence present" if (d / "INDEX.json").is_file() else ""))


def _layout_missing(ctx: Ctx, d: Path, j: int) -> list:
    """Rollouts layout: [why dir d is not yet in it] -- no INDEX.json. For the current epoch the digest
    step's runner then builds it; an earlier epoch's dir needs the named command (an operator decision:
    that dir was granted to an earlier evolver, and adding to it changes its recorded hash)."""
    if _evidence_layout(ctx) != "rollouts" or (Path(d) / "INDEX.json").is_file():
        return []
    how = ("the digest step builds it" if j == ctx.epoch
           else f"build it by hand: {_rollout_evidence_build_cmd(ctx, j)}")
    return [f"{Path(d).name} holds no rollout evidence (INDEX.json) although RUNTIME declares "
            f"evolver_evidence_layout=rollouts; {how}"]


def _v_digests_complete(ctx: Ctx, dirs=None, rollouts: bool = True):
    """The evidence surface must be COMPLETE, re-derived from its bytes: every
    train task of the split, every one of spec.n_samples samples, each with its grade evidence (a
    scorer diff for SSB v2, a structured verdict for tau3). Rules: src/orchestrator/digest_coverage.py.

    WHY A SEPARATE STEP. `_v_digests_train_only` checks MEMBERSHIP, and a surface can pass it with
    half its samples carrying no scorer diff. Folding completeness
    into it would change what its selftest case means (five one-byte train digests pass). And it
    has NO runner: the only real fixes are upstream (declare the snapshot roots, finish the leg), so
    a runner could only re-run the builder, whose idempotent path would re-read the same bytes and
    report "RAN, and its verify still fails". The HALT names the remedy instead.

    `dirs` IS A LIST. By default it is every dir the evolver will be granted: [this epoch's
    digests], or digests_e1..eK in all-epochs mode (RUNTIME evolver_evidence) — an incomplete
    earlier surface would otherwise be re-served to every later epoch unchecked.

    It reads only what a rebuild would rewrite, so it is also the one guard a shell export cannot
    switch off: the builder's REQUIRE mode is an env var, this is not.
    """
    from src.orchestrator import digest_coverage as _cov
    dirs = _evidence_dirs(ctx) if dirs is None else list(dirs)
    ok, detail = _cov.check(dirs, _cov.train_stems(ctx.spec.split_path), int(ctx.spec.n_samples),
                            getattr(ctx.spec, "benchmark", ""))
    # Rollouts layout (RUNTIME evolver_evidence_layout): every granted dir's rollout
    # evidence re-derived from its bytes and its leg (rollout_evidence.check_dir): one trace + one grade
    # per train task x sample and nothing else, every INDEX row's outcome / cells / turns / cap /
    # termination / skills re-derived, the library byte-identical, 0 non-train ids, the legacy digest's
    # verdicts (and full grader text) agreeing. `rollouts=False`: the digest runner's own coverage gate,
    # which runs before it builds the rollout evidence.
    if not rollouts or _evidence_layout(ctx) != "rollouts":
        return ok, detail
    rp = _rollout_evidence_problems(ctx, dirs)
    if rp:
        return False, detail + "\n  ROLLOUT EVIDENCE: " + "\n  ROLLOUT EVIDENCE: ".join(rp)
    return ok, detail + f"; rollout evidence re-derived from the bytes and the leg in {len(dirs)} dir(s)"


def _v_evidence_libraries(ctx: Ctx):
    """All-epochs mode only. Every granted digest dir must say truthfully which library
    produced it, re-derived from its bytes (rules: src/orchestrator/skill_library.py).

    A skill-load marker saying the body "is the file you were handed in your staging dir" is true
    only of the newest surface. On digests_e2..e{K-1} the solver ran under v1..v{K-2}; staging holds
    a same-named skill whose text may differ, or none at all. So in all-epochs mode:
    digests_e1_train (the base leg) must hold no _library/ and no marker. Each digests_e{j}_train
    (j >= 2) must hold a byte-identical copy of skills_v{j-1} under _library/, and every task file
    and marker in it must name that version and hash.

    No runner, like the coverage step: a surface built before the mode was declared (or by a hand
    pipeline) cannot be fixed in place, and the builder refuses to overwrite one. The HALT names
    the remedy. In current_epoch mode the evolver reads only the newest surface, whose markers
    point at staging truthfully, so there is nothing to check.
    """
    if _evidence_mode(ctx) != "all_epochs":
        return True, ("current_epoch mode: the evolver reads only the newest surface, whose "
                      "skill-load markers point at its staging dir truthfully; nothing to check")
    from . import skill_library as _sl
    rd = Path(ctx.run_dir)
    bad, good = [], []
    for j in range(1, ctx.epoch + 1):
        d = EpochPaths(rd, j).digests
        lib = EpochPaths(rd, j - 1).promoted if j > 1 else None
        probs = _sl.problems(d, lib, _sl.label_for_epoch(j) if j > 1 else "")
        if probs:
            bad.append("; ".join(probs))
        else:
            good.append(f"{d.name} <- " + (f"{lib.name} sha256:{_sl.library_sha256(lib)[:12]}"
                                           if lib else "no skills"))
    if bad:
        return False, ("an evidence dir does not carry its producing library: " + " | ".join(bad)
                       + ". REMEDY: a surface built before evolver_evidence=all_epochs was "
                         "declared, or by a hand pipeline, names no version. Move that "
                         "digests_e<j>_train OUT of the run dir (never delete evidence an evolver "
                         "read), rebuild it with `--epoch <j> --step` under the all-epochs "
                         "RUNTIME, and record the retrofit.")
    return True, f"{len(good)} evidence dir(s), each names its producing library: " + ", ".join(good)


def _v_skills_load(ctx: Ctx):
    """Ask the skills' real consumer what the set holds -- not the filesystem, not the evolver.

    The consumer is Claude Code, the solver of every leg, and its own loader is the canonical judge of a
    skill set: the invariants layer re-implements none of its rules and records the count of
    <name>/SKILL.md it will read. A consumer that cannot be asked is a HALT, because an audit that cannot
    run must not read as one that passed.

    An EMPTY staging (the solver runs with no skills) and a staging byte-identical to the previous set
    (the solver carries the same set again, as the evolver's system prompt says) are both results. On the
    Daytona host the staging dir exists only once a turn is installed, and a turn that failed exits
    non-zero, which the evolver step turns into a HALT.
    """
    from src.orchestrator.invariants import CONSUMER_CLAUDE_CODE, check_skills_load_through_consumer
    st = ctx.paths.staging
    if not st.is_dir():
        return False, f"no staging dir at {st}"
    c = check_skills_load_through_consumer(st, consumer=CONSUMER_CLAUDE_CODE)
    if c.status == "SKIP":
        return False, (f"could not ask the real loader: {c.detail}. An audit that cannot run "
                       f"must not read as one that passed.")
    if c.status != "PASS":
        return False, c.detail
    return True, c.detail


def _v_promoted_matches_staging(ctx: Ctx):
    """Promotion is a copy, and a copy is a place a set can silently diverge. Compare content
    digests, not names or counts."""
    st, pr = ctx.paths.staging, ctx.paths.promoted
    if not pr.is_dir():
        return False, f"nothing promoted to {pr}"
    a, b = _skill_fingerprint(st), _skill_fingerprint(pr)
    if a != b:
        only_s = sorted(set(a) - set(b)); only_p = sorted(set(b) - set(a))
        diff = sorted(k for k in set(a) & set(b) if a[k] != b[k])
        return False, (f"promoted set differs from staging: staging-only={only_s} "
                       f"promoted-only={only_p} content-differs={diff}")
    return True, f"{len(a)} skill(s), content-identical to staging"


def _v_conversation_recorded(ctx: Ctx):
    """The operator<->AO exchange for this epoch is on disk, two-sided, and uncorrupt.

    WHY THIS IS A STEP AND NOT JUST A WRITE. `conversation.record` deliberately does not raise on
    a write failure — a full disk must not kill a paid epoch. But an observability channel that
    fails silently is believed captured without ever being checked against the artifact. So the
    write is best-effort and the CHECK is mandatory, the same division this file uses for
    BLOCKED_e{N}.json ("losing the audit record is tolerable; proceeding past an unanswered
    question is not" — with the difference that here the record IS the deliverable, so it gets a
    gate of its own).

    ABSENT IS NOT AUTOMATICALLY A FAILURE. Under `--step`,
    main() records the operator's turn BEFORE any verify runs, so the file must exist and its
    absence means the write failed — HALT. Under `--plan` nothing is recorded by design
    (ctx.readonly), so absence is UNMEASURED and says so.

    Placed after the authoring step so BOTH sides have necessarily spoken: the loop's composed message
    and the AO's reply are recorded by `author_prompt` itself. Checking two-sidedness before that
    would fail on the first invocation of every run, which is a check that fires on correct
    behaviour.
    """
    from src.orchestrator import conversation
    p = conversation.path_for(ctx.run_dir)
    if not p.is_file():
        if getattr(ctx, "readonly", False):
            return True, (f"UNMEASURED — no transcript at {p.name} and this is a read-only plan, "
                          f"which records nothing by design. Re-check under --step.")
        return False, (f"NO TRANSCRIPT at {p}. This invocation recorded the operator's turn "
                       f"before running any check, so the file should exist; its absence means "
                       f"the write failed (permissions, full disk, or a directory in the way). "
                       f"Fix that before continuing — the run is currently producing an "
                       f"unauditable conversation.")
    return conversation.verify_covers(ctx.run_dir, ctx.epoch)


def _v_prompt_authored(ctx: Ctx):
    """The prompt exists and the AO's own authoring record is beside it.

    Deliberately weak on its own — it asserts an artifact, not a quality. The claims INSIDE the
    prompt are checked by the prompt-figures check (numbers re-derive from the leg json) and the
    leakage check (no ids, no held-out figure). Splitting it that way is the point: authorship
    moved to the AO, verification did not move anywhere.

    IT DOES CHECK ONE THING ABOUT HOW THE AO RAN: whether the conversation the run declared it
    wanted is the conversation it got. Without that comparison, a run that declared
    `ao_continuous_session` and then authored unrelated single-shot epochs would pass every gate
    and every report line identically to one that worked. A feature whose absence is invisible is
    not shipped, it is claimed.
    """
    p = ctx.paths.prompt
    if not p.is_file():
        return False, f"no prompt at {p.name} — the AO has not authored it yet"
    rec = p.parent / f"e{ctx.epoch}_PROMPT_AUTHOR.json"
    if not rec.is_file():
        return False, (f"{p.name} exists but there is no {rec.name} — a prompt nobody can attribute "
                       f"is not an authored prompt. If it was hand-written, say so in that file.")
    d = json.loads(rec.read_text())
    base = (f"authored by {d.get('model')} in {d.get('seconds')}s, {d.get('bytes')} B, "
            f"policy {Path(str(d.get('policy'))).name}")
    # Daytona host: the addendum that describes the evolver's own machine is either wired with its
    # exact bytes (daytona) or absent (local) -- requested vs wired.
    _add = Path(__file__).resolve().parent / "PROMPT_POLICY_ADDENDUM_EVOLVER_DAYTONA.md"
    if getattr(ctx, "evolver_host", "local") == "daytona":
        _want = hashlib.sha256(_add.read_bytes()).hexdigest() if _add.is_file() else None
        if not _want or d.get("policy_addendum_sha256") != _want:
            return False, (f"evolver_host=daytona but {rec.name} records policy_addendum_sha256="
                           f"{str(d.get('policy_addendum_sha256'))[:12]!r}, not the addendum's "
                           f"{str(_want)[:12]!r}: the prompt was not authored for the Daytona "
                           f"evolver (move the prompt aside and re-author)")
        base += f" + addendum {_add.name}"
        # The AO itself ran on its own Daytona sandbox: same containment half as the evolver's
        # (resources, allow list at create and end, egress, image, delete, token scan, log), and
        # its upload = exactly the author message, the evidence dirs, the oracle README and the
        # three literals the Mac AO is granted.
        from src.orchestrator.daytona_evolver import check_ao_upload, check_daytona_receipt
        _ar = p.parent / f"e{ctx.epoch}_AO_DAYTONA.json"
        if d.get("ao_host") != "daytona" or not _ar.is_file():
            return False, f"evolver_host=daytona but the AO did not run on Daytona (no {_ar.name})"
        _ad = json.loads(_ar.read_text())
        if (_ad.get("daytona") or {}).get("heldout_hits"):
            return False, f"held-out ids in the AO's session: {_ad['daytona']['heldout_hits'][:4]}"
        # WHERE ON DAYTONA the AO ran, refused both ways (as _v_evolver_contained): placement
        # "pool_container" = scripts/daytona/evolver_container.py `ao` in the pool machine's evolver container, whose
        # receipt has no own VM (no resources, no ledgered delete) -- check_daytona_receipt would refuse every one.
        _want_pl = _evolver_placement(ctx)
        _got_pl = (_ad.get("daytona") or {}).get("placement") or "own_sandbox"
        if _got_pl != _want_pl:
            return False, (f"AO placement MISMATCH: RUNTIME evolver_daytona.placement={_want_pl!r} but {_ar.name} "
                           f"records {_got_pl!r}.")
        if _want_pl == "pool_container":
            from src.orchestrator.pool_container_evolver import check_pool_container_receipt
            _kw = _pool_container_check_kw(ctx)    # require_snapshot_ref None = not pinned: recorded, not compared
            # the AO's one writable dir is the run dir it writes ITER{N}_PROMPT.md into (AORun.writable_dir)
            _ok, _det = check_pool_container_receipt(_ad, log=Path(str(d.get("log") or "")),
                                                     staging=p.parent, **_kw)
            if not _ok:
                return False, f"AO pool-container containment: {_det}"
        else:
            _ok, _det = check_daytona_receipt(_ad, log=Path(str(d.get("log") or "")))
            if not _ok:
                return False, f"AO daytona containment: {_det}"
        _c = check_ao_upload(run_dir=p.parent, epoch=ctx.epoch, receipt=_ad,
                             message=p.parent / f"e{ctx.epoch}_AUTHOR_MESSAGE.md",
                             evidence_dirs=_evidence_dirs(ctx), oracle=Path(ctx.oracle))
        if _c.status != "PASS":
            return False, f"AO upload: {_c.detail}"
        base += f"; AO on Daytona ({_det[:80]})"
    elif d.get("policy_addendum_sha256"):
        return False, (f"{rec.name} carries a daytona policy addendum but this run's evolver_host is "
                       f"local")
    # An EXPLICIT claude-* author model is asserted against the session's own
    # stream (re-derived from the log, never the record's status field). Alias requests and
    # records that predate the check are unchanged.
    from .claude_pins import check_session, expected_model
    if expected_model(d.get("model")) and ("model_check" in d or d.get("pins")):
        _log = Path(str(d.get("log") or ""))
        if not _log.is_file():
            return False, f"{rec.name} names author log {_log} which is absent — model unprovable"
        _mc = check_session(_log, d.get("model"))
        if not _mc["ok"]:
            return False, (f"the AO session did not run on {_mc['expected_model']}: "
                           + "; ".join(_mc["violations"]))
        base += (f"; session ran on {_mc['resolved_model']} (claude_code "
                 f"{_mc['claude_code_version']})")

    want = bool(getattr(ctx, "ao_continuous_session", False))
    mode, sid = d.get("session_mode"), d.get("session_id")
    argv = d.get("session_argv")
    if "session_mode" not in d:
        return True, (f"{UNMEASURED}{base}; this record predates session accounting, so whether the "
                      f"AO held one conversation across the run is NOT established by it")
    if want:
        if mode == "single-shot" or not sid:
            return False, (f"RUNTIME.json declares ao_continuous_session, but {rec.name} records "
                           f"session_mode={mode!r} / session_id={sid!r}. This epoch's AO had no "
                           f"memory of the previous one; the run is not measuring what it declared.")
        if isinstance(argv, list) and not any(str(x) in ("--resume", "--session-id") for x in argv):
            return False, (f"{rec.name} claims session_mode={mode!r} with id {sid} but records NO "
                           f"session flag in the dispatched argv ({argv!r}) — the record describes "
                           f"a conversation the CLI was never told about.")
        if ctx.epoch > 1 and str(mode).startswith("new"):
            return False, (f"epoch {ctx.epoch} started a NEW AO conversation ({mode!r}) instead of "
                           f"resuming the run's. Everything the AO settled in epochs 1.."
                           f"{ctx.epoch - 1} is absent from it. Check "
                           f"AO_SESSION.json.")
        return True, f"{base}; AO session {mode} ({str(sid)[:8]})"
    if mode != "single-shot":
        return False, (f"{rec.name} records session_mode={mode!r}, but this run did NOT declare "
                       f"ao_continuous_session — the epochs are sharing a conversation the spec "
                       f"does not describe.")
    return True, f"{base}; single-shot AO (no cross-epoch session declared)"


def _prompt_policy(ctx: Ctx):
    """The policy the AO is initialised with, or a HALT when it contradicts the evidence mode.

    current_epoch mode returns exactly what it always did (RUNTIME `prompt_policy`, or None for
    prompt_author's default). all_epochs mode defaults to PROMPT_POLICY_ALL_EPOCHS.md. It
    refuses the single-batch PROMPT_POLICY.md, whose section 5 names one directory and whose
    section 6 says every earlier skill version is unopenable, which _library/ now makes false.
    A RUNTIME.json may name that file explicitly, so a run switched to all-epochs would otherwise
    keep it silently. The reverse, all-epochs policy under
    current_epoch, is refused too: it asks for an EVIDENCE INDEX that FACTS will not contain.
    A custom policy is trusted as declared; the prompt it produces is still checked
    (_v_prompt_names_evidence).
    """
    from .prompt_author import ALL_EPOCHS_POLICY, DEFAULT_POLICY
    declared = Path(ctx.prompt_policy) if ctx.prompt_policy else None

    def same(a, b):
        try:
            return a.resolve() == b.resolve()
        except OSError:
            return False
    if _evidence_mode(ctx) != "all_epochs":
        if declared is not None and same(declared, ALL_EPOCHS_POLICY):
            raise Halt(f"RUNTIME.json prompt_policy is {ALL_EPOCHS_POLICY.name}, but the run does not "
                       f"declare evolver_evidence=all_epochs. That policy tells the AO to name every "
                       f"epoch's evidence dir from an index FACTS will not contain. Declare the mode "
                       f"or name another policy.")
        return declared
    if declared is None:
        return ALL_EPOCHS_POLICY
    if same(declared, DEFAULT_POLICY):
        raise Halt(f"RUNTIME.json declares evolver_evidence=all_epochs but prompt_policy is the "
                   f"single-batch {DEFAULT_POLICY.name}, which tells the AO to name ONE evidence dir "
                   f"and that no earlier skill version can be opened. Set prompt_policy to "
                   f"{ALL_EPOCHS_POLICY} (or remove it; that is the all-epochs default).")
    return declared


def _run_author_prompt(ctx: Ctx) -> None:
    """THE AO WRITES THE EVOLVER'S USER PROMPT. Facts derived deterministically, prose by a model."""
    from .prompt_author import author_prompt, derive_facts

    # Before any spend: a policy that contradicts the evidence mode is a HALT.
    policy = _prompt_policy(ctx)
    dg = ctx.paths.digests
    # The leg the digests were BUILT from: epoch 1 -> the declared base, epoch N -> v(N-1)'s train.
    # Derived from EpochPaths rather than a guessed filename, and cross-checked inside derive_facts
    # so a mismatch RAISES instead of authoring a prompt against the wrong evidence (0809c C-4.1).
    if ctx.epoch == 1:
        leg = Path(ctx.base_leg) if ctx.base_leg else None
    else:
        leg = EpochPaths(ctx.run_dir, ctx.epoch - 1).leg("train")
    prior = EpochPaths(ctx.run_dir, ctx.epoch - 1).promoted if ctx.epoch > 1 else None
    corpus = None
    for cand in (REPO / "third_party/tau3-bench/data/tau2/domains" / ctx.spec.domain / "documents",):
        if cand.is_dir():
            corpus = cand
    # All-epochs mode: the AO is granted, and told about, every dir the evolver will be granted.
    # None keeps the single-dir AO exactly.
    _ev = ctx.paths.evidence_dirs() if _evidence_mode(ctx) == "all_epochs" else None
    if _ev is None:
        facts = derive_facts(dg, Path(leg) if leg else None, corpus,
                             variant=getattr(ctx.spec, "variant", "") or "")
    else:
        # The EVIDENCE INDEX. Each dir paired with the leg it digests and the library that
        # produced it, by the SAME mapping _v_prompt_numbers' known legs use: e1 is the
        # base leg and no library, e_j is skills_v{j-1}'s train leg and skills_v{j-1}.
        _rd = ctx.run_dir
        evidence = [(j, d,
                     (Path(ctx.base_leg) if ctx.base_leg else None) if j == 1
                     else EpochPaths(_rd, j - 1).leg("train"),
                     None if j == 1 else EpochPaths(_rd, j - 1).promoted)
                    for j, d in enumerate(_ev, start=1)]
        facts = derive_facts(dg, Path(leg) if leg else None, corpus,
                             variant=getattr(ctx.spec, "variant", "") or "", evidence=evidence)
    _host = _evolver_host(ctx)
    _addendum = None
    if _host == "daytona":
        # The AO must describe the evolver's OWN machine truthfully (the solver's image,
        # its versions, the model-API-only network, the train input workbooks), and may state
        # only what FACTS gives -- so the facts go in FACTS, and the how-to in an addendum file.
        facts.evolver_env = _evolver_env_facts(ctx)
        _addendum = Path(__file__).resolve().parent / "PROMPT_POLICY_ADDENDUM_EVOLVER_DAYTONA.md"
    info = author_prompt(facts=facts, out=ctx.paths.prompt, epoch=ctx.epoch, digests=dg,
                         evidence_dirs=_ev, policy_addendum=_addendum, host=_host,
                         daytona_cfg=(getattr(ctx, "evolver_daytona", {}) or {}) if _host == "daytona" else None,
                         staging=ctx.paths.staging,
                         oracle=Path(ctx.oracle), log=ctx.run_dir / f"e{ctx.epoch}_prompt_author.log",
                         model=getattr(ctx.spec, "prompt_author_model", None)
                         or ctx.prompt_author_model,
                         policy=policy,
                         prior_skills=prior,
                         # OPT-IN, default OFF. Turning it on changes what the AO IS: one
                         # conversation across the run instead of one stranger per epoch.
                         # Defaulting off keeps every completed run byte-reproducible;
                         # RUNTIME.json opts a new run in.
                         continuous_session=bool(getattr(ctx, "ao_continuous_session", False)))
    info["facts"] = facts.as_block()
    (ctx.run_dir / f"e{ctx.epoch}_PROMPT_AUTHOR.json").write_text(json.dumps(info, indent=1))
    # RECORDED FIRST, THEN HALT. A prompt authored by a substituted model (alias
    # flip, refusal fallback) is not the AO the run declared; _v_prompt_authored re-derives the
    # same verdict from the log on every later invocation, so the HALT cannot be stepped past.
    mc = info.get("model_check") or {}
    if mc.get("checked") and not mc.get("ok"):
        raise Halt(f"the AO prompt-author session did NOT run on {mc.get('expected_model')}: "
                   + "; ".join(mc.get("violations") or []) + f". Recorded in "
                   f"e{ctx.epoch}_PROMPT_AUTHOR.json; {ctx.paths.prompt.name} must not be used.")


def run_id(ctx: Ctx) -> str:
    """THE RUN'S RUNTIME IDENTITY IS ITS RUN DIRECTORY, NOT `spec.name`.

    `spec.name` is operator-typed and NOTHING checks it for uniqueness, so two runs sharing a name
    would share every runtime identity derived from it -- the evolver's isolated work dir and the
    Claude Code session dir that follows it (~/.claude/projects/<slug-of-cwd>): one run's prompt
    staged into the other's work dir, and its sandbox granting read on the other run's transcript.
    EpochPaths.leg() already keys on the run dir; so does this.

    DELIBERATELY NOT APPLIED to the two SSB v2 cache/snapshot names (`{name}_e{N}_cache` and the
    `ssbv2_{name}_v{N}_train_tasks` snapshot root): renaming them would orphan existing cache
    directories.
    """
    return Path(ctx.run_dir).resolve().name


def _non_train_extra_mentions(body: str, split) -> list:
    """The ids under the split's dropped / unused / excluded_prior_run keys (NON_TRAIN_EXTRA_KEYS) that
    `body` names, in either spelling; a train id's spelling never counts (invariants.id_mentions). `split`
    is the split dict or the split file's path. Shared with the stateful evolver's message + system prompt
    check (stateful_evolver.v_message_no_leakage), so both apply the one rule."""
    from .invariants import id_mentions
    sp = split if isinstance(split, dict) else json.loads(Path(split).read_text())
    train = _split_key_ids(sp, "train")
    return id_mentions(body, sorted({str(i) for k in NON_TRAIN_EXTRA_KEYS for i in _split_key_ids(sp, k)}
                                    - set(train)), allowed=train)


def _v_prompt_no_leakage(ctx: Ctx):
    """Mechanical, because a model writes the prompt.

    The prompt MAY name TRAIN task ids (the rollouts layout's per-rollout lines name every train task).
    It refuses every
    val/test id and every id under the split's dropped / unused / excluded_prior_run keys, in both
    spellings (canonical and the file-safe `/`->`_` form of trace paths), and any held-out figure."""
    from .invariants import check_prompt_no_leakage
    c = check_prompt_no_leakage(ctx.paths.prompt, Path(ctx.spec.split_path))
    if c.status == "SKIP":
        return True, f"{UNMEASURED}{c.detail}"
    if c.status != "PASS":
        return False, c.detail
    # The check above enumerates val/test ids only; the split's other keys
    # ("dropped", "unused", "excluded_prior_run") are non-train too, and no prompt may name one.
    try:
        sp = json.loads(Path(ctx.spec.split_path).read_text())
        body = ctx.paths.prompt.read_text()
    except (OSError, json.JSONDecodeError) as e:
        return False, f"{c.detail}; the split's other keys could not be scanned ({type(e).__name__})"
    extra = _non_train_extra_mentions(body, sp)
    if extra:
        return False, (f"prompt names {len(extra)} task id(s) from the split's "
                       f"{'/'.join(NON_TRAIN_EXTRA_KEYS)} keys — {extra[:6]}. A train task may be named "
                       f"; a non-train task never is.")
    n_extra = sum(len(_split_key_ids(sp, k)) for k in NON_TRAIN_EXTRA_KEYS)
    return True, f"{c.detail}; also 0 of {n_extra} dropped/unused/excluded ids present"


def _v_prompt_not_analytical(ctx: Ctx):
    """The AO prepares raw information and is not part of the optimizer. Mechanical, because
    the policy prose demonstrably did not hold it — see check_prompt_not_analytical."""
    from .invariants import check_prompt_not_analytical
    c = check_prompt_not_analytical(ctx.paths.prompt)
    if c.status == "SKIP":
        return True, f"{UNMEASURED}{c.detail}"
    return c.status == "PASS", c.detail


def _v_legs_comparable(ctx: Ctx, key: str):
    """The skill leg and its BASE comparator must be the same comparison — same tasks,
    same samples-per-task, and both matching the spec. See invariants.check_legs_paired_comparably
    for why a config whose spec.n_samples equals the script default of 3 hides a wiring defect."""
    from .invariants import check_legs_paired_comparably
    skill = ctx.paths.leg(key)
    # DECLARED PER KEY, NEVER SUBSTITUTED ACROSS KEYS. Comparing a val leg against the train base
    # is not a weaker check, it is a guaranteed spurious FAIL (different task sets by definition).
    declared = {"train": getattr(ctx, "base_leg", ""),
                "val": getattr(ctx, "base_leg_val", "")}.get(key, "")
    base = Path(declared) if declared else None
    if base is None or not base.is_file():
        # Last resort, and only inside the run dir: a self-contained run whose base legs sit
        # beside its skill legs under the same stem. Never reaches for the other key's file.
        cand = skill.parent / skill.name.replace(f"_v{ctx.epoch}_", "_base_")
        base = cand if cand.is_file() else None
    if base is None:
        return True, (f"{UNMEASURED}{key}: no {key} base comparator declared "
                      f"(RUNTIME.json '{'base_leg' if key == 'train' else 'base_leg_val'}') "
                      f"and none beside the skill leg — pairing NOT checked")
    c = check_legs_paired_comparably(base, skill, want_samples=ctx.spec.n_samples)
    if c.status == "SKIP":
        return True, f"{UNMEASURED}{key}: {c.detail}"
    if c.status != "PASS":
        return False, f"{key}: {c.detail}"
    # check_legs_paired_comparably compares per_task KEY sets, which for SSB v2 are indices: two
    # equal-size legs on different task sets compare equal there. Legs that state task_ids are
    # compared by id, and each against the run's declared split[key], so a base leg left on another
    # split cannot pair with a skill leg on this one.
    probs = []
    for side, p in (("base", base), ("skill", skill)):
        st, det = _leg_on_split(p, ctx.spec.split_path, key, ctx.spec.n_samples)
        if st == "bad":
            probs.append(f"{side}: {det}")
    if probs:
        return False, f"{key}: " + "; ".join(probs)
    return True, f"{key}: {c.detail}"


def _v_leg_loaded_promoted(ctx: Ctx, key: str):
    """A leg can run against the WRONG skills dir and still report a
    clean pass rate. Read the dir the leg says it loaded and compare to what was promoted."""
    p = ctx.paths.leg(key)
    if not p.exists():
        return False, f"no result json at {p.name}"
    d = json.loads(p.read_text())
    want = ctx.paths.promoted
    raw = d.get("skills_dir")
    if not raw:
        # Absence is decided on the raw value, before Path() can invent a cwd:
        # Path("") == Path(".") is truthy, so a runner that never wrote the field would
        # read as "loaded .".
        # Fallback: the runner's startup banner lists every loaded skill as an
        # "[SKILL] <name>:" line, and it hard-exits if EVOSKILL_SKILLS_DIR has no
        # */SKILL.md — so an exact name-set match against the promoted dir is proof
        # of delivery even when the JSON predates the skills_dir field.
        log = p.with_suffix(".log")
        if not log.exists():
            return False, (f"{p.name} records no skills_dir and {log.name} is absent "
                           f"— cannot prove what it loaded")
        loaded = set(re.findall(r"^\s*\[SKILL\]\s+([A-Za-z0-9._-]+):",
                                log.read_text(), re.M))
        promoted = {q.parent.name for q in want.glob("*/SKILL.md")}
        if not promoted:
            return False, f"promoted dir {want} has no */SKILL.md to compare against"
        if loaded == promoted:
            return True, (f"{key}: JSON lacks skills_dir — log evidence: "
                          f"[SKILL] lines match the promoted set exactly "
                          f"({len(promoted)} skills)")
        return False, (f"{key}: JSON lacks skills_dir and log [SKILL] set "
                       f"{sorted(loaded)} != promoted {sorted(promoted)}")
    got = Path(raw)
    try:
        same = got.resolve() == want.resolve()
    except OSError:
        same = str(got) == str(want)
    if not same:
        return False, f"leg loaded {got}, promoted was {want}"
    return True, f"{key}: loaded {want.name}"


def _v_no_infra_exclusions(ctx: Ctx, key: str):
    """An excluded episode is a silently smaller denominator. Neither exclusion nor
    timeout-zeroing may be nonzero without a human deciding what it means."""
    p = ctx.paths.leg(key)
    if not p.exists():
        return False, f"no result json at {p.name}"
    d = json.loads(p.read_text())
    ex = d.get("episodes_excluded_infra", 0) or 0
    tz = d.get("episodes_timeout_scored_zero", 0) or 0
    if ex or tz:
        return False, f"{key}: {ex} infra-excluded, {tz} timeout-zeroed — denominator moved"
    return True, f"{key}: 0 excluded, 0 timeout-zeroed"


def _v_skills_were_consulted(ctx: Ctx):
    """Did the solver ever OPEN the library it was given?

    A run can score normally, gate normally, and produce a null that reads as "skills did not
    help" while its solver consulted the library in a few percent of rollouts: the intervention
    was never delivered, and the legs measured a closed channel.

    This runs on the NEXT epoch's digests, because those are built from THIS version's rollouts.
    At epoch N there is no epoch N+1 surface yet, so it SKIPS rather than inventing a number —
    and a skip says so out loud instead of reading as a pass.
    """
    nxt = ctx.run_dir / f"digests_e{ctx.epoch + 1}_train"
    skills = ctx.paths.promoted
    if not skills.is_dir():
        return False, f"no promoted skills at {skills.name}"
    if not nxt.is_dir():
        return True, (UNMEASURED + f"{nxt.name} does not exist yet, so this version's uptake "
                      f"is not measurable until the next epoch's surface is built.")
    # Rollouts layout: measured from the TRACES (rollout_evidence INDEX.json, each row's skills_loaded
    # re-derived from its full trace by the completeness step), not from the digests' [SKILL LOADED:
    # markers. The skill tool's result starts "Base directory for this skill:", so the digest builder's
    # collapse (it wants the text to START with the frontmatter) does not fire on a skill-tool load, and
    # the digest markers would undercount loads the traces show.
    if _evidence_layout(ctx) == "rollouts" and (nxt / "INDEX.json").is_file():
        from src.orchestrator import rollout_evidence as _re
        from src.orchestrator.skill_delivery_shim import UPTAKE_FLOOR
        try:
            u = _re.uptake(_re.read_index(nxt))
        except (_re.EvidenceError, ValueError, OSError) as e:
            return False, f"{nxt.name}/INDEX.json cannot be read: {e}"
        if u["library"] != skills.name:
            return False, (f"{nxt.name}/INDEX.json was produced by {u['library']!r}, not {skills.name}: it "
                           f"does not describe this version's rollouts")
        detail = (f"{u['consulted']}/{u['rollouts']} rollouts loaded a skill = {u['uptake']:.0%} [rollout "
                  f"traces, {nxt.name}/INDEX.json]; per skill {u['per_skill']}"
                  + (f"; never loaded {u['never_loaded']}" if u["never_loaded"] else ""))
        if u["uptake"] < UPTAKE_FLOOR:
            return False, (detail + " — the library was NOT DELIVERED. Any gate verdict on this version is "
                           "about the delivery channel, not about the skills.")
        return True, detail
    assess_delivery = _load_delivery_assessor()
    if assess_delivery is None:
        return True, UNMEASURED + "the skill-delivery assessor (src/orchestrator/skill_delivery_shim.py) is not importable."
    r = assess_delivery(skills, nxt)
    # A reader that finds none of its OWN markers has not measured 0% — it has failed to read,
    # and those must not share a return value: a reader pointed at another benchmark's surface
    # matches none of its notation and would report "0/N consulted — NOT DELIVERED" with a correct
    # `rollouts` count, which is what makes it convincing. Absent key => a reader that predates
    # this distinction; treat as readable. See src/orchestrator/skill_delivery_shim.py.
    if r.get("unreadable"):
        return False, (f"{r['rollouts']} rollout section(s) present but NOT ONE carries a "
                       f"skill-load marker this reader knows. That is a READER failure, not 0% "
                       f"uptake, and must not be read as 'not delivered'. Add the surface's "
                       f"dialect to src/orchestrator/skill_delivery_shim.py.")
    detail = (f"{r['consulted']}/{r['rollouts']} rollouts consulted a skill = {r['uptake']:.0%}"
              + (f" [{r['dialect']}]" if r.get("dialect") else "")
              + (f"; INVALID names {dict(**r['typos'], **r['category_errors'])}"
                 if (r["typos"] or r["category_errors"]) else "")
              + (f"; never loaded {r['never_loaded']}" if r["never_loaded"] else ""))
    if not r["delivered"]:
        return False, (detail + " — the library was NOT DELIVERED. Any gate verdict on this "
                       "version is about the delivery channel, not about the skills.")
    return True, detail


def _v_outcomes_reconcile(ctx: Ctx, key: str):
    """pass + graded-fail + never-graded must equal the total, and never-graded must
    equal the capped count. A 0 that no check produced is not a wrong answer, and reporting
    raw pass deltas without this split over-claims."""
    p = ctx.paths.leg(key)
    if not p.exists():
        return False, f"no result json at {p.name}"
    d = json.loads(p.read_text())
    try:
        passed, total = _leg_pass_total(d, p)
    except KeyError as e:
        return False, str(e)
    if "termination_reasons" in d:
        # tau3 schema: an episode capped at max_steps / context_overflow was never
        # judged at all, so the cap count IS the never-graded channel.
        term = d.get("termination_reasons") or {}
        never = int(term.get("max_steps", 0)) + int(term.get("context_overflow", 0))
        capped_note = ""
    elif "turn_exhausted" in d:
        # SSB v2 schema: a turn-capped episode is STILL graded on whatever it
        # saved, so turn_exhausted is a separate, non-exclusion category. What CAN
        # silently move here is the episode grid, so reconcile total against the
        # declared n x n_samples instead. A rollout whose grade(task, snapshot)
        # raised twice has no score; the Claude Code
        # leg lists it in d["grader_failed"], and that list IS the never-graded
        # channel. A leg without the key (a direct leg) has none.
        never = len(d.get("grader_failed") or [])
        n, ns = int(d.get("n", 0)), int(d.get("n_samples", 0))
        if n * ns != total:
            return False, (f"{key}: n({n}) x n_samples({ns}) != sample_total({total}) "
                           f"— episode grid does not reconcile")
        capped_note = f", capped-but-graded {int(d.get('turn_exhausted', 0))}"
    else:
        return False, (f"{p.name} has neither termination_reasons (tau3) nor "
                       f"turn_exhausted (ssbv2) — cannot attribute the outcome channels")
    graded_fail = total - passed - never
    if graded_fail < 0:
        return False, (f"{key}: pass({passed}) + never-graded({never}) exceeds total({total})")
    return True, (f"{key}: pass {passed}, graded-fail {graded_fail}, never-graded {never} "
                  f"= {total}{capped_note}")


def _leg_pass_total(d: dict, src) -> tuple:
    """(passed, total) episodes from a leg JSON, whichever benchmark wrote it.

    tau3 writes `pass`/`total`; SSB v2 writes `sample_pass`/`sample_total`, where the episode
    unit is a task x sample draw (60 tasks x 3 samples = 180). Reading `d["pass"]` unconditionally
    would raise KeyError on every SSB v2 leg: a crashing guard is worse than a failing one, because
    it carries no verdict.

    REFUSES rather than defaulting to (0, 0). A zero total would make every fraction in every
    prompt trivially unmatchable, turning the prompt-figures check off silently.
    """
    for pk, tk in (("pass", "total"), ("sample_pass", "sample_total")):
        if pk in d and tk in d:
            return int(d[pk]), int(d[tk])
    raise KeyError(
        f"{getattr(src, 'name', src)} has neither pass/total nor sample_pass/sample_total "
        f"(keys: {sorted(d)[:12]}) — cannot re-derive the prompt's figures")


# ── the run's DECLARED split, read from the file — never a count in code ──────────────────────────
# A split may carry a "dropped" key (tasks whose own reference fails self-grading) besides train / val /
# test / unused. Nothing below knows any set of numbers: every count is the length of a key of the file
# ctx.spec.split_path names, and a leg is ON the split when its task ids ARE
# that key. Dropped tasks are NON-TRAIN like val/test/unused: no rule that forbids non-train material to
# the evolver may stop covering them, and no prompt may name one.
NON_TRAIN_EXTRA_KEYS = ("dropped", "unused", "excluded_prior_run")


def _split_key_ids(sp: dict, key: str) -> list:
    """Ids under one split key, in file order. A key may be a list of ids, a list of {"id": ...}, or a
    record carrying a "task_ids" list (the shape of "dropped" and "excluded_prior_run")."""
    v = sp.get(key)
    if isinstance(v, dict):
        v = v.get("task_ids")
    if not isinstance(v, list):
        return []
    return [x if isinstance(x, str) else x["id"] for x in v]


def split_ids(split_path, key: str) -> list:
    """split[key] of the split FILE, in file order ([] when the key is absent)."""
    return _split_key_ids(json.loads(Path(split_path).read_text()), key)


def _leg_task_ids(d: dict):
    """The leg's task ids in per_task order, or None when the leg does not state them.

    SSB v2 legs key per_task by the INDEX into task_ids ("0".."n-1"): the key SET of per_task says how
    MANY tasks ran and nothing about WHICH, so two legs on different task sets of equal size have equal
    per_task key sets (invariants.check_legs_paired_comparably compares exactly that set). A leg with
    task_ids is read through it; one whose per_task keys are themselves ids (tau3, the selftest's
    synthetic legs) states no task_ids and is None here, so every caller keeps its old behaviour for it.
    Raises ValueError when task_ids and per_task disagree (a leg that contradicts itself)."""
    tids = d.get("task_ids")
    if not isinstance(tids, list) or not tids:
        return None
    pt = d.get("per_task")
    if not isinstance(pt, dict) or set(pt) != {str(i) for i in range(len(tids))}:
        raise ValueError(f"per_task keys ({len(pt) if isinstance(pt, dict) else 'none'}) are not the "
                         f"indices 0..{len(tids) - 1} of its {len(tids)} task_ids")
    if len(set(tids)) != len(tids):
        raise ValueError("task_ids lists a task more than once")
    return [str(t) for t in tids]


def _leg_on_split(leg, split_path, key: str, n_samples=None) -> tuple:
    """("ok" | "bad" | "unknown", detail): is this leg JSON exactly split[key] — the same task SET, each
    task once — at n_samples per task? "unknown" = the leg states no task ids (see _leg_task_ids); every
    caller treats that as before this check existed. Counts come from the split file only."""
    leg = Path(leg)
    try:
        d = json.loads(leg.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return "bad", f"{leg.name} is unreadable ({type(e).__name__})"
    try:
        tids = _leg_task_ids(d)
    except ValueError as e:
        return "bad", f"{leg.name}: {e}"
    if tids is None:
        return "unknown", f"{leg.name} states no task_ids — its task set cannot be compared to the split"
    if not split_path:
        return "unknown", f"no split declared to compare {leg.name} with"
    try:
        want = split_ids(split_path, key)
    except (OSError, json.JSONDecodeError) as e:
        return "bad", f"the split {split_path} is unreadable ({type(e).__name__})"
    if not want:
        return "bad", f"the split {Path(split_path).name} has no {key!r} ids to compare {leg.name} with"
    have, need = set(tids), set(want)
    if have != need:
        extra, missing = sorted(have - need), sorted(need - have)
        return "bad", (f"{leg.name} ran {len(tids)} tasks but the run's split "
                       f"{Path(split_path).name} (sha256:{_sha12(split_path)}) has {len(want)} under "
                       f"{key!r}: {len(extra)} not in it {extra[:4]}, {len(missing)} of it not run "
                       f"{missing[:4]}. A leg on another task set is not this run's {key} measurement")
    ns = d.get("n_samples")
    if n_samples is not None and ns is not None and int(ns) != int(n_samples):
        return "bad", f"{leg.name} ran N={ns}, the spec asks for n_samples={n_samples}"
    bad_n = [tids[int(i)] for i, v in d["per_task"].items()
             if not isinstance(v, list) or (ns is not None and len(v) != int(ns))]
    if bad_n:
        return "bad", f"{leg.name}: {len(bad_n)} task(s) without exactly N={ns} scores {bad_n[:4]}"
    return "ok", (f"{leg.name}: exactly the split's {len(want)} {key} tasks x N={ns} "
                  f"(split sha256:{_sha12(split_path)})")


def _sha12(p) -> str:
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]
    except OSError:
        return "unreadable"


def _v_prompt_numbers(ctx: Ctx):
    """The evolver reasons from the headline figures in its prompt. If those disagree
    with the result JSONs they came from, it is reasoning about a different experiment.
    Checks the episode fractions that appear as `N of M` or `N/M`."""
    pr = ctx.paths.prompt
    prev = ctx.epoch - 1
    if not pr.exists():
        return False, f"no prompt at {pr.name}"

    # WHERE THE PRIOR NUMBERS LIVE. For epoch >= 2 it is the previous epoch's train leg, named
    # by the same expression _run_leg writes. For epoch 1 it is the BASE leg, which does NOT
    # follow that convention (`<run>_base_train.json`, not `<run>_v0_train.json`) and so must be
    # declared. Passing epoch 1 with "no prior leg to re-derive against" would be VACUOUS at the
    # one epoch that matters most: every epoch-1 prompt quotes a base pass rate, and the evolver
    # reasons from exactly those figures.
    if prev >= 1:
        # Via EpochPaths.leg(), NEVER a second copy of the naming expression: an inline copy is
        # blind to any run whose legs are named differently.
        src = EpochPaths(ctx.run_dir, prev).leg("train")
        if not src.exists():
            return False, (f"epoch {ctx.epoch} prompt must re-derive against {src.name}, which "
                           f"does not exist — the previous epoch's train leg has not been run")
    else:
        if not ctx.base_leg:
            return False, ("epoch 1 prompt quotes figures that came from the BASE leg, but no "
                           "base_leg is declared (RUNTIME.json 'base_leg' / --base-leg), so "
                           "there is nothing to check them against. Declare it rather than "
                           "letting an unchecked number reach the evolver.")
        src = Path(ctx.base_leg)
        if not src.exists():
            return False, f"declared base_leg does not exist: {src}"

    # The figure must come from a leg ON THE RUN'S SPLIT: a base leg left on another split would
    # re-derive "its" figure perfectly and the prompt would quote a number about a different task set.
    # Legs stating no task_ids are not judged here.
    st, det = _leg_on_split(src, ctx.spec.split_path, "train", getattr(ctx.spec, "n_samples", None))
    if st == "bad":
        return False, f"the leg the prompt's figures come from is not on the run's split: {det}"
    text = pr.read_text(errors="replace")
    # The rollouts layout tells the evolver each rollout's score in the prompt. The per-rollout lines state "<matched>/<graded> cells" per
    # sample: they are taken OUT of the headline-fraction scan below (a task with 171 graded cells would
    # read as a /171 batch figure) and re-derived instead, every line, from the newest batch's INDEX.json
    # and this very leg JSON (_rollout_table_problems). The digests layout is scanned exactly as before.
    try:
        layout = _evidence_layout(ctx)
    except Halt as e:
        return False, str(e)
    table_note = ""
    if layout == "rollouts":
        from src.orchestrator import evolver_template as _et
        tab = _et.parse_rollout_table(text)
        tprobs, table_note = _rollout_table_problems(ctx, tab, src)
        if tprobs:
            return False, ("the per-rollout lines do not re-derive from the evidence INDEX and the leg: "
                           + "; ".join(tprobs[:6])
                           + (f" (+{len(tprobs) - 6} more)" if len(tprobs) > 6 else ""))
        text = tab["residual"]
    probs, note = _headline_fraction_problems(ctx, text, src)
    if probs:
        return False, probs[0]
    return True, note + table_note


def _headline_fraction_problems(ctx: Ctx, text: str, src: Path) -> tuple:
    """(problems, note): the headline-fraction rule on `text` against the leg `src` (the newest batch's).
    The CURRENT leg's `P of E episodes` / `P/E` must appear, and every other figure on that denominator must
    be a leg that really exists (the base leg or an earlier epoch's train leg); an unsourced number is a
    problem. `note` is the success detail (empty when there are problems). Shared with the stateful
    evolver's message check (stateful_evolver.v_message_numbers), which applies the one definition to
    its score table's residual."""
    src = Path(src)
    d = json.loads(src.read_text())
    want = _leg_pass_total(d, src)
    found = set()
    # The `N of M episodes` arm does not require bold markers to wrap the phrase exactly: an author
    # may write
    #     - **passed: 1 of 3 episodes (33.3%)**
    # where the bold span is wider than the phrase. A guard that refuses valid input gets switched
    # off, so being too strict is a defect of the same family as being vacuous. What must not match
    # is `2 of 3 tasks` when the denominator happens to equal the leg total; the literal word
    # `episodes` does that job directly, so anchor on the noun and let the markup be whatever prose
    # wants.
    for m in re.finditer(r"(\d+)\s+of\s+(\d+)\s+episodes?\b|\b(\d+)\s*/\s*(\d+)\b", text):
        a, b, c, e = m.groups()
        pair = (int(a), int(b)) if a else (int(c), int(e))
        if pair[1] == want[1]:
            found.add(pair)
    if not found:
        return [f"prompt states no /{want[1]} episode fraction to check against {src.name}"], ""

    # Demanding that EVERY fraction on the denominator equal the current leg's would make an honest
    # multi-epoch trajectory unstateable: "no skills scored 59 of 90, round one scored 64" is exactly
    # the comparison the evolver needs. The author's only recourse would be to phrase prior rates so
    # the regex cannot see them — a check that is satisfied by hiding evidence from it is worse than
    # no check.
    #
    # So: the CURRENT leg's figure must be present, and every other figure on that denominator
    # must match some OTHER leg that really exists (the base, or an earlier epoch's train leg).
    # A number that matches no leg at all is still a refusal, so an invented figure is told apart
    # from a correctly-quoted prior one.
    known = {}
    base = Path(ctx.base_leg) if ctx.base_leg else None
    if base and base.is_file():
        bj = json.loads(base.read_text())
        known[_leg_pass_total(bj, base)] = "the base leg"
    for e in range(1, ctx.epoch):
        q = EpochPaths(ctx.run_dir, e).leg("train")
        if q.is_file():
            qj = json.loads(q.read_text())
            known[_leg_pass_total(qj, q)] = f"epoch {e}'s train leg"
    if want not in found:
        return [f"prompt states {sorted(found)} on /{want[1]} but never the CURRENT leg's "
                f"{want[0]}/{want[1]} from {src.name}"], ""
    unknown = {f for f in found if f != want and f not in known}
    if unknown:
        return [f"prompt states {sorted(unknown)} on /{want[1]}, which matches neither "
                f"{src.name} ({want[0]}/{want[1]}) nor any prior leg "
                f"({sorted(known)}) — an unsourced number"], ""
    quoted = {known[f] for f in found if f != want}
    return [], (f"prompt's {want[0]}/{want[1]} re-derives from {src.name}"
                + (f"; also correctly quotes {', '.join(sorted(quoted))}" if quoted else ""))


def _rollout_table_problems(ctx: Ctx, tab: dict, leg_path: Path) -> tuple:
    """(problems, note): the prompt-figures check for the per-rollout lines of a rollouts-layout prompt. `tab` is
    evolver_template.parse_rollout_table(prompt). Each sample's line is re-derived on its own, never
    by re-rendering: pass/fail from the LEG JSON's recorded score (the record every downstream number
    reads), and outcome, cells, turns, turn-cap mark and skills loaded (through the legend, which must
    be exactly the batch's library) from the newest batch's INDEX.json, whose rows the evidence steps
    re-derived from the trace and grade bytes. The lines must cover exactly the leg's samples, and
    their (cap) marks must add up to the leg's own turn_exhausted."""
    from src.orchestrator import evolver_template as _et
    from src.orchestrator import rollout_evidence as _re
    from src.orchestrator.outcomes import PASS_THRESHOLD
    probs = [f"a per-rollout line does not parse: {b[:120]!r}" for b in tab["bad"]]
    d = ctx.paths.digests
    try:
        idx = _re.read_index(d)
    except (_re.EvidenceError, OSError, ValueError) as e:
        return probs + [f"{d.name}/INDEX.json cannot be read ({e})"], ""
    if (idx.get("leg") or {}).get("sha256") != hashlib.sha256(Path(leg_path).read_bytes()).hexdigest():
        probs.append(f"{d.name}/INDEX.json was built from {(idx.get('leg') or {}).get('file')} "
                     f"sha256:{str((idx.get('leg') or {}).get('sha256'))[:12]}, not the current bytes of "
                     f"{leg_path.name}")
    if not tab["tasks"]:
        return probs + [f"the prompt carries no per-rollout line; the rollouts layout states one per train "
                        f"task of {leg_path.name}"], ""
    want = {(str(t).replace("/", "_"), k): v for (t, k), v in _et.leg_sample_scores(leg_path).items()}
    rows = {(str(r.get("task_file")), int(r.get("sample"))): r for r in idx.get("rows") or []}
    have = {(stem, k) for stem, ss in tab["tasks"].items() for k in ss}
    if have != set(want):
        miss, extra = sorted(set(want) - have), sorted(have - set(want))
        probs.append(f"the lines state {len(have)} samples, {leg_path.name} has {len(want)}: {len(miss)} "
                     f"missing (e.g. {miss[:2]}), {len(extra)} not in the leg (e.g. {extra[:2]})")
    names = sorted(str(x.get("name")) for x in (idx.get("producing_library") or {}).get("skills") or [])
    legend = tab["legend"]
    if names and (legend is None or sorted(legend.values()) != names):
        probs.append(f"the skills legend {sorted((legend or {}).values())} is not the batch's library {names}")
    if not names and legend:
        probs.append(f"a skills legend {sorted(legend.values())}, but the batch's solver carried no skill")
    caps = 0
    for stem, k in sorted(have & set(want)):
        p, tag, score = tab["tasks"][stem][k], f"{stem} s{k}", want[(stem, k)]
        oc = _et.TOKEN_OUTCOME[p["outcome"]]
        if (oc == "passed") != (score is not None and score >= PASS_THRESHOLD):
            probs.append(f"{tag}: the prompt says {p['outcome']}, {leg_path.name} records score {score}")
        caps += bool(p["cap"])
        r = rows.get((stem, k))
        if r is None:
            probs.append(f"{tag}: no row in {d.name}/INDEX.json")
            continue
        if r.get("score") != score:
            probs.append(f"{tag}: {d.name}/INDEX.json records score {r.get('score')}, {leg_path.name} {score}")
        if p["regraded"]:
            if r.get("outcome") != "grader_failed" or oc not in ("passed", "graded_wrong"):
                probs.append(f"{tag}: marked re-graded as {p['outcome']}, but its INDEX outcome is "
                             f"{r.get('outcome')}")
        elif oc != r.get("outcome"):
            probs.append(f"{tag}: the prompt says {p['outcome']}, {d.name}/INDEX.json says {r.get('outcome')}")
        stated = (int(p["m"]), int(p["g"])) if p["m"] is not None else None
        rec = (None if (p["regraded"] or r.get("outcome") == "not_eligible" or r.get("cells_graded") is None)
               else (r.get("cells_matched"), r.get("cells_graded")))
        if stated != rec:
            probs.append(f"{tag}: the prompt states cells {stated}, {d.name}/INDEX.json {rec}")
        turns = None if p["turns"] == "?" else int(p["turns"])
        if turns != r.get("turns"):
            probs.append(f"{tag}: the prompt states {p['turns']} turns, {d.name}/INDEX.json {r.get('turns')}")
        if bool(p["cap"]) != bool(r.get("turn_cap_hit")):
            probs.append(f"{tag}: the (cap) mark disagrees with {d.name}/INDEX.json turn_cap_hit="
                         f"{r.get('turn_cap_hit')}")
        if names:
            got = (None if p["loaded"] is None else
                   [] if p["loaded"] == "none" else [(legend or {}).get(x) for x in p["loaded"].split()])
            if got is None or None in got or sorted(got) != sorted(r.get("skills_loaded") or []):
                probs.append(f"{tag}: the prompt states skills loaded {p['loaded']!r}, {d.name}/INDEX.json "
                             f"{r.get('skills_loaded')}")
        elif p["loaded"] is not None:
            probs.append(f"{tag}: states skills loaded, but the batch's solver carried no skill")
    legd = json.loads(leg_path.read_text())
    if "turn_exhausted" in legd and int(legd["turn_exhausted"]) != caps:
        probs.append(f"the lines mark {caps} samples (cap), {leg_path.name} records turn_exhausted="
                     f"{legd['turn_exhausted']}")
    return probs, (f"; its {len(tab['tasks'])} per-rollout lines ({len(have)} samples) re-derive from "
                   f"{d.name}/INDEX.json and {leg_path.name}")


# A mention of an evidence dir: its name plus whatever path-ish characters follow it. A suffix
# of dots only is sentence punctuation; anything else (".bak", ".as_given", "_old") is a sibling.
_DIGEST_MENTION = re.compile(r"digests_e(\d+)_train([A-Za-z0-9_.\-]*)")


def _v_prompt_names_evidence(ctx: Ctx):
    """All-epochs mode only. The prompt must name EVERY granted evidence dir by its absolute
    path, and no digest dir that is not granted.

    A dir granted and never mentioned is evidence the evolver does not know it has, and a dir named
    but not granted (a later epoch, a stray sibling, another run's, or a RELATIVE path, which
    resolves against the evolver's isolated work dir) is an instruction it cannot follow.

    Rules, re-derived from the prompt bytes and EpochPaths.evidence_dirs():
      * each granted dir appears as an absolute path: the resolved form the author message gives,
        or the run dir's unresolved absolute form;
      * every `digests_e<j>_train` mention has 1 <= j <= epoch (never a later epoch's surface),
        sits directly under THIS run's absolute path (never relative, bare or another run's), and
        is not a sibling such as digests_e1_train.bak.
    Placed before the evolver dispatch, so no evolver runs on a prompt that failed it.
    """
    if _evidence_mode(ctx) != "all_epochs":
        try:
            if _evidence_layout(ctx) == "rollouts" and ctx.paths.prompt.is_file():
                sub = _evidence_subpath_problems(ctx.paths.prompt.read_text(errors="replace"),
                                                 [ctx.paths.digests], ctx.run_dir, "rollouts")
                if sub:
                    return False, "the prompt does not name the granted evidence: " + "; ".join(sub[:6])
        except Halt as e:
            return False, str(e)
        return True, ("current_epoch mode: the prompt describes one digest dir; this check applies "
                      "to all-epochs evidence only")
    pr = ctx.paths.prompt
    if not pr.is_file():
        return False, f"no prompt at {pr.name}"
    text = pr.read_text(errors="replace")
    roots = sorted({str(Path(ctx.run_dir).resolve()), str(Path(ctx.run_dir).absolute())})
    missing = []
    for j, d in enumerate(ctx.paths.evidence_dirs(), start=1):
        forms = {str(d), str(Path(ctx.run_dir).absolute() / d.name)}
        if not any(re.search(re.escape(f) + r"(?![A-Za-z0-9_\-])(?!\.[A-Za-z0-9_\-])", text)
                   for f in forms):
            missing.append(d.name)
    bad = []
    for m in _DIGEST_MENTION.finditer(text):
        j, suffix = int(m.group(1)), m.group(2)
        name = f"digests_e{j}_train"
        if suffix.strip("."):
            bad.append(f"names {name}{suffix}, a sibling that is not granted")
        elif not 1 <= j <= ctx.epoch:
            bad.append(f"names {name}, which is not granted at epoch {ctx.epoch}"
                       + (" (a LATER epoch's surface: future evidence)" if j > ctx.epoch else ""))
        elif not any(text[:m.start()].endswith(r + "/") for r in roots):
            bad.append(f"names {name} without this run's absolute path (relative, bare, or "
                       f"another run's), which the evolver cannot resolve from its work dir")
    try:
        layout = _evidence_layout(ctx)
    except Halt as e:
        return False, str(e)
    bad += _evidence_subpath_problems(text, ctx.paths.evidence_dirs(), ctx.run_dir, layout)
    if missing or bad:
        # This is a no-runner Step, so without the remedy the HALT reads only "This step has
        # no runner". author_prompt refuses to overwrite and the AO step's verify already passes,
        # so the one way forward is to move the prompt aside and let the AO author it again.
        return False, ("the prompt does not name exactly the granted evidence: "
                       + (f"never names {missing} by absolute path; " if missing else "")
                       + "; ".join(sorted(set(bad))[:6])
                       + f". REMEDY: move {Path(ctx.run_dir).absolute() / pr.name} aside (keep "
                       f"it) and re-run --step; the AO re-authors against the EVIDENCE INDEX "
                       f"paths.")
    return True, (f"prompt names all {ctx.epoch} granted evidence dir(s) by absolute path, and no "
                  f"other digest dir")


_SUBPATH_CHARS = r"[^\s`'\"<>()\[\]]*"


def _evidence_subpath_problems(text: str, dirs, run_dir, layout: str) -> list:
    """The naming rule for paths INSIDE a granted evidence dir (the rollouts layout names the snapshot of
    each epoch's available skills). A prompt that names `<dir>/<something>` sends the evolver there, so that
    something must exist in the granted dir, as a real file or dir (never a symlink, never a `..` climb):
    an evolver must not be told about something unreal. A path with a <placeholder>
    is judged up to the placeholder. In the rollouts layout the prompt must also name every granted dir's
    INDEX.md (the older epochs' per-rollout scores live there), the _library/ of every dir produced by a
    library (the skills that epoch's solver HAD, not only the ones it loaded), and the newest dir's rollouts/."""
    bad = []
    dirs = [Path(d) for d in dirs]
    for j, d in enumerate(dirs, start=1):
        forms = sorted({str(d), str(Path(run_dir).absolute() / d.name)}, key=len, reverse=True)
        named = set()
        for f in forms:
            for m in re.finditer(re.escape(f) + "/(" + _SUBPATH_CHARS + ")", text):
                rest = m.group(1).rstrip(".,;:")
                if not rest:
                    continue
                named.add(rest.rstrip("/"))
                parts = [x for x in rest.split("/") if x]
                target = d.joinpath(*parts) if parts else d
                if ".." in parts or target.is_symlink() or not target.exists():
                    bad.append(f"names {d.name}/{rest}, which is not in the granted dir")
        if layout != "rollouts":
            continue
        if "INDEX.md" not in named:
            bad.append(f"never names {d.name}/INDEX.md (rollouts layout: every granted dir's per-rollout rows)")
        if (d / "_library").is_dir() and "_library" not in named:
            bad.append(f"never names {d.name}/_library/, the skills that epoch's solver carried")
        if j == len(dirs) and not any(x == "rollouts" or x.startswith("rollouts/") for x in named):
            bad.append(f"never names {d.name}/rollouts/, where the newest batch's traces are")
    return bad


def _v_evolver_contained(ctx: Ctx):
    """If the run declared kernel containment for the evolver, prove the dispatch had it.

    Requested vs wired: `EVOSKILL_EVOLVER_SANDBOX` is read by a launcher several layers away; if the plumbing broke — a typo in the env key, a launcher that ignored it, an
    `sb` array that expanded to nothing — every artifact and every other verify would look
    identical to a properly contained run. So the launcher writes a receipt BEFORE dispatch and
    this compares it against what the run said it wanted.

    UNMEASURED, never a pass, when the receipt predates this check: an old run cannot be
    retroactively declared contained, and reporting it as ok would be the vacuous-guard shape
    this file keeps having to remove.
    """
    # THE HARNESSES HAVE DIFFERENT DEFAULTS, AND THAT IS NOT A WART. `direct` runs a python
    # module and is contained unconditionally -- there is no opt-out and none is wanted.
    # `claude_cli` runs the CLI, whose local containment carries a recorded escape hatch. Expected
    # containment is therefore a function of the harness, not a single global flag.
    harness = getattr(ctx.spec, "evolver_harness", "claude_cli")
    want = True if harness == "direct" else bool(getattr(ctx, "evolver_sandbox", True))
    rec = ctx.run_dir / f"e{ctx.epoch}_EVOLVER_CONTAINMENT.json"
    if not rec.is_file():
        # A NEW GUARD MUST NOT FAIL ON HISTORY IT CANNOT APPLY TO. An epoch whose launcher predates
        # the receipt can never produce one; a check that a finished, correct run cannot pass is
        # not a check, it is noise that trains the operator to ignore the board.
        #
        # Absence is therefore UNMEASURED in both directions, with the reason stated, and the
        # only FAIL is a receipt that CONTRADICTS what the run declared. That still catches the
        # thing worth catching -- broken plumbing on a live dispatch, since the launcher now
        # always writes a receipt -- without inventing a verdict about the past.
        ran = (ctx.run_dir / f"tau3_evolver_e{ctx.epoch}.log").exists()
        if not want:
            return True, (f"{UNMEASURED}no {rec.name}; containment was explicitly disabled for "
                          f"this run, so the evolver had hooks + --add-dir only")
        if ran:
            return True, (f"{UNMEASURED}the evolver for this epoch ALREADY RAN under a launcher "
                          f"that predates containment accounting (no {rec.name}). Whether it was "
                          f"contained is NOT established and cannot be established after the "
                          f"fact — treat this epoch as uncontained when reasoning about contamination.")
        return True, (f"{UNMEASURED}no {rec.name} yet — the evolver has not been dispatched for "
                      f"this epoch. The dispatch step ahead of this one owns that ordering.")
    try:
        d = json.loads(rec.read_text())
    except Exception as e:                                      # noqa: BLE001
        return False, f"{rec.name} is unreadable ({type(e).__name__}: {e})"
    got = bool(d.get("sandboxed"))
    if got != want:
        return False, (f"containment MISMATCH: harness={harness!r} with evolver_sandbox="
                       f"{getattr(ctx, 'evolver_sandbox', True)} expects sandboxed={want}, but "
                       f"{rec.name} records sandboxed={got} ({d.get('containment')!r}). The "
                       f"dispatch did not run in the world this run declared.")
    # WHICH WORLD. A daytona receipt under a local-host run (or the reverse) is an
    # evolver that ran somewhere other than where the run declared: refused both ways.
    _host = getattr(ctx, "evolver_host", "local")
    _is_day = d.get("containment") == "daytona"
    if _is_day != (_host == "daytona"):
        return False, (f"world MISMATCH: RUNTIME evolver_host={_host!r} but {rec.name} records "
                       f"containment={d.get('containment')!r}.")
    day_note = ""
    if _is_day:
        # WHERE ON DAYTONA, refused both ways like the world check above: a pool-container
        # receipt under an own-sandbox run (or the reverse) ran somewhere other than declared.
        _want_pl = _evolver_placement(ctx)
        _got_pl = (d.get("daytona") or {}).get("placement") or "own_sandbox"
        if _got_pl != _want_pl:
            return False, (f"placement MISMATCH: RUNTIME evolver_daytona.placement={_want_pl!r} but {rec.name} "
                           f"records {_got_pl!r}.")
        _dcfg = getattr(ctx, "evolver_daytona", {}) or {}
        if _want_pl == "pool_container":
            # The container half (src/orchestrator/pool_container_evolver): the pool machine's readback (its
            # two-domain allow list, the leg snapshot) at attach AND at end, the plan re-derived, the in-container
            # probe (namespaces, credentials, mount table = plan, no machine path/process visible, writes only
            # where granted, egress api.anthropic.com only), the POST probe, evidence + granted dirs unchanged,
            # the token scans, the verified + ledgered teardown, OAuth auth / no MCP / no RemoteTrigger.
            from src.orchestrator.pool_container_evolver import check_pool_container_receipt
            _kw = _pool_container_check_kw(ctx)    # require_snapshot_ref None = not pinned: recorded, not compared
            _ok, _det = check_pool_container_receipt(d, log=_evolver_log_path(ctx), staging=ctx.paths.staging,
                                                     **_kw)
            if not _ok:
                return False, f"pool-container containment: {_det}"
            day_note = f"; {_det}"
        else:
            # The Daytona half (src/orchestrator/daytona_evolver.check_daytona_receipt): resources and
            # the domain allow list at create AND at end, egress probes, the image recipe re-hashed
            # from disk, the ledgered delete, evidence + inputs unchanged ON THE VM, the token scan,
            # OAuth auth / no MCP / no RemoteTrigger from the log. There is no sandbox-exec profile
            # and no by-name write deny on the Mac run dir to check: nothing of the Mac was there.
            from src.orchestrator.daytona_evolver import check_daytona_receipt
            _ok, _det = check_daytona_receipt(
                d, log=_evolver_log_path(ctx), require_libreoffice=_dcfg.get("require_libreoffice"))
            if not _ok:
                return False, f"daytona containment: {_det}"
            day_note = f"; {_det}"
    if want and not _is_day and not Path(str(d.get("profile") or "")).is_file():
        return False, (f"{rec.name} claims sandbox-exec but its profile "
                       f"{d.get('profile')!r} is not on disk — the receipt describes containment "
                       f"nothing can confirm.")
    if want and not _is_day and "sandbox_no_writes" in d:
        # The by-name WRITE denies on the run dir and the operator channel are checked, so a
        # launcher edit that dropped them cannot pass. A receipt that carries the field must name
        # both (in either spelling of the operator dir: from the resolved run dir, or resolved from
        # the declared one). Receipts written before the field existed keep passing.
        _nw = {str(Path(str(p)).resolve()) for p in (d.get("sandbox_no_writes") or [])}
        _rd_r = Path(ctx.run_dir).resolve()
        _need = {"the run dir": {str(_rd_r)},
                 "the operator channel": {str(operator_dir(_rd_r)),
                                          str(operator_dir(ctx.run_dir).resolve())}}
        _lost = [k for k, alts in _need.items() if not alts & _nw]
        if _lost:
            return False, (f"{rec.name} records no by-name write deny on {' or '.join(_lost)} "
                           f"(sandbox_no_writes={sorted(_nw)}); the evolver could write there "
                           f"wherever the run dir sits — the split, the evidence, the answers.")
    ev_note = ""
    if "evidence_dirs" in d:
        # THE EVIDENCE HALF. Receipts written before this field existed skip it (a new guard must
        # not fail on history it cannot apply to). A receipt that has it
        # must describe the evidence set this run's mode declares — a launcher that dropped the
        # prior-digest list, or granted a stray dir, is a different world than the one declared —
        # and the evidence must be byte-identical before and after the evolver ran: every later
        # epoch, the uptake check and the stale-surface guard read those same files.
        want_ev = sorted(str(Path(p).resolve()) for p in _evidence_dirs(ctx))
        got_ev = sorted(str(Path(str(p)).resolve()) for p in (d.get("evidence_dirs") or []))
        if got_ev != want_ev:
            return False, (f"{rec.name} records evidence dirs {[Path(p).name for p in got_ev]} "
                           f"but evolver_evidence={getattr(ctx, 'evolver_evidence', '')!r} at "
                           f"epoch {ctx.epoch} declares {[Path(p).name for p in want_ev]}. The "
                           f"dispatch did not grant the evidence this run declared.")
        # ...and nothing the kernel was told to grant may undo the run-dir hide: not the run dir
        # or a parent of it (that would expose peer runs), not a digest dir outside the declared set.
        _rd = str(Path(ctx.run_dir).resolve())
        for g in d.get("sandbox_grants") or []:
            gp = str(Path(str(g)).resolve())
            if gp == _rd or _rd.startswith(gp.rstrip("/") + "/"):
                return False, (f"{rec.name} records a kernel grant on {gp}, which is the run dir "
                               f"or a parent of it — every sibling of the evidence was readable")
            if Path(gp).name.startswith("digests_") and gp not in want_ev:
                return False, (f"{rec.name} records a kernel grant on {Path(gp).name}, a digest "
                               f"dir outside the declared evidence set {[Path(p).name for p in want_ev]}")
        before = d.get("evidence_sha256_before")
        if not isinstance(before, dict) or not before:
            # The launcher refuses an ALL-EPOCHS dispatch without a before-hash
            # (nothing spent), so here a missing one fails only in that mode. In current-epoch
            # mode the launcher warns and dispatches, and the epoch is not thrown away after a
            # full evolver run for a record it was never required to carry: a NOTE, stated.
            if _evidence_mode(ctx) == "all_epochs":
                return False, (f"{rec.name} names evidence dirs but records no pre-dispatch hash "
                               f"({d.get('evidence_sha256_error', 'absent')}) — whether the "
                               f"evolver changed its own evidence cannot be established")
            ev_note = (f"; NOTE: no pre-dispatch evidence hash recorded "
                       f"({d.get('evidence_sha256_error', 'absent')}); current_epoch mode, so "
                       f"whether the evolver changed its evidence is NOT established")
        else:
            after = d.get("evidence_sha256_after")
            src = "recorded after the evolver exited"
            if not isinstance(after, dict) or not after:
                from src.orchestrator.evidence_surface import hashes as _hashes
                after, src = _hashes(list(before)), "computed now (no post-run hash recorded)"
            changed = sorted(Path(p).name for p in before if after.get(p) != before[p])
            if changed:
                return False, (f"the evidence CHANGED between dispatch and {src}: {changed}. A "
                               f"granted digest dir is read-only evidence for this and every later "
                               f"epoch; find what wrote it before trusting anything built on it.")
            ev_note = (f"; evidence {d.get('evidence_mode', '?')}: {len(before)} dir(s), "
                       f"unchanged ({src})")
    return True, (f"evolver ran under {d.get('containment')}"
                  + (f" ({Path(str(d.get('profile'))).name})" if want and d.get("profile") else "")
                  + day_note + ev_note)


def _v_evolver_read_nothing_forbidden(ctx: Ctx):
    """Peer-artifact and held-out reads, audited from the EVOLVER'S OWN LOG — the only record of what
    it touched.

    The loop DISPATCHES the evolver, so without this it could launch one, take its skills, and promote
    them without ever asking whether it had read a peer arm's answers to the same train tasks or
    reached a held-out task spec.
    Those are the two contamination modes that invalidate an arm outright, and they are invisible
    in the artifact — a contaminated skill set looks exactly like a clean one.

    CALLED, NOT REIMPLEMENTED. A second copy of a guard is how two guards become one real one.

    A SKIP IS A FAILURE HERE. The invariants layer returns SKIP when the log is absent, which is
    honest per-check and dishonest in aggregate — it converts "I could not check" into "nothing to
    report". So an unreadable or missing log HALTS; if the evolver was driven by hand and its log
    lives elsewhere, declare the path rather than letting the audit quietly not happen.
    """
    from src.orchestrator.invariants import (check_no_heldout_reachable,
                                             check_no_peer_artifacts_read)
    log = Path(ctx.evolver_log) if ctx.evolver_log else \
        ctx.run_dir / f"tau3_evolver_e{ctx.epoch}.log"
    if not log.is_file():
        return False, (f"no evolver log at {log.name}, so neither the peer-artifact nor "
                       f"the held-out audit can run. Declare evolver_log if it lives "
                       f"elsewhere — an audit that cannot run must not read as one that passed.")
    out = []
    if getattr(ctx, "evolver_host", "local") == "daytona":
        # On the VM there is no other run to read, so the peer-artifact question is answered by
        # what was ON THE BOX, not by grepping commands: every file there re-derived from this run's own bytes,
        # and the in-VM listing equal to that manifest. The log grep would HALT on paths that do
        # not exist there (the solver transcripts name this run's own skills-dir symlink target).
        # The held-out check (held-out ids in the log) stays as the second check.
        from src.orchestrator.daytona_evolver import check_upload_manifest
        _first = check_upload_manifest(
            run_dir=ctx.run_dir, epoch=ctx.epoch, split=Path(ctx.spec.split_path),
            evidence_dirs=_evidence_dirs(ctx), prev_library=ctx.run_dir / f"skills_v{ctx.epoch - 1}",
            prompt=ctx.paths.prompt, oracle=Path(ctx.oracle), staging=ctx.paths.staging)
        # Held-out ids in either SSB spelling, re-derived from the installed log AND every downloaded
        # session transcript (subagents, nested sessions); the matcher below misses the file-safe form.
        from src.orchestrator.daytona_evolver import heldout_id_hits
        _hh = heldout_id_hits([log, ctx.run_dir / f"e{ctx.epoch}_evolver_session"], Path(ctx.spec.split_path))
        if _hh:
            return False, f"held-out task ids in the evolver's logs/transcripts: {_hh[:4]}"
    else:
        _first = check_no_peer_artifacts_read(log, ctx.run_dir)
    for c in (_first, check_no_heldout_reachable(log, Path(ctx.spec.split_path))):
        if c.status == "FAIL":
            return False, f"{c.name}: {c.detail}"
        if c.status == "SKIP":
            return False, f"{c.name} could not run: {c.detail}"
        out.append(c.name)
    return True, f"{len(out)} contamination audits passed against {log.name}"


def _skill_fingerprint(root: Path) -> dict:
    """Content digest of a skill set, keyed by skill name. ONE definition, used by both the
    promotion check and the pipeline's promoted-set record — two implementations of "has this set
    changed" is how a record ends up covering something other than what was promoted.

    That one definition lives in skill_library.py, because the digest builders hash a surface's
    producing library with it too, and they import nothing from this module."""
    from . import skill_library as _sl
    return _sl.fingerprint(root)


def staging_sha256(root: Path) -> str:
    """A single stable hash over a whole skill set (the pipeline records it per promoted set)."""
    from . import skill_library as _sl
    return _sl.library_sha256(root)


# ── the epoch program ────────────────────────────────────────────────────────────────────
# ── runners ──────────────────────────────────────────────────────────────────────────────
# A step with a runner can be DRIVEN, not only checked.
#
# WHAT IS DRIVABLE. A runner is written where the action is MECHANICAL: a deterministic
# transformation of artifacts already on disk, a leg launch, or the evolver dispatch.
# Every runner is idempotent and refuses rather than overwrites, because a re-run after a HALT
# must not silently destroy the evidence that caused the halt.

def _run_promote(ctx: Ctx) -> None:
    """staging -> skills_vN. Refuses to overwrite a promoted set that differs from staging."""
    src, dst = ctx.paths.staging, ctx.paths.promoted
    if not src.is_dir():
        raise Halt(f"nothing to promote: {src} does not exist")
    if dst.exists():                                 # an EMPTY staging is promoted too (no skills)
        same = {p.relative_to(src): p.read_bytes() for p in src.rglob("SKILL.md")} == \
               {p.relative_to(dst): p.read_bytes() for p in dst.rglob("SKILL.md")}
        if same:
            return                                   # idempotent: already promoted, byte-identical
        raise Halt(f"{dst} exists and DIFFERS from staging — refusing to overwrite a promoted "
                   f"skill set; a leg may already have been measured against it")
    shutil.copytree(src, dst)


def _load_delivery_assessor():
    """The skill-uptake assessor, or None if it genuinely cannot be loaded.

    ONE loader, two callers (the stale-surface guard at build time and the uptake verify
    downstream). A bare `try: import ... except: return` would let a missing module silently no-op
    the guard while reporting nothing. Callers decide what None means; neither may treat it as a
    pass.
    """
    try:
        from .skill_delivery_shim import assess_delivery                   # noqa: PLC0415
        return assess_delivery
    except Exception:                                                     # noqa: BLE001
        return None


def _digest_caches_for_epoch(ctx: Ctx) -> tuple:
    """WHICH ROLLOUTS does epoch N's evidence surface describe?

    Epoch N's surface must describe version N-1's rollouts: base at epoch 1, skills_v1 at epoch 2,
    and so on. `ctx.digest_caches` names the BASE caches — it is an init-time declaration, one list
    for the whole run — so using it at every epoch rebuilds the epoch-1 surface forever.

    Then `digests_e2_train` comes out BYTE-IDENTICAL to `digests_e1_train`, and the epoch-2 evolver
    is shown the no-skills rollouts as if v1 had never run. Nothing downstream catches it, because
    everything downstream is consistent with it: the prompt-figures check agrees with a base surface,
    and the uptake check reads this surface and reports "NOT DELIVERED" for a version whose own
    transcripts hold skill-tool calls. Agreeing signals, all wrong, because all read the same wrong
    artifact.

    So this reads what the runner RECORDED rather than deriving the name a second time: every leg
    json carries `cache_dirs`, written by the process that opened them. A re-derivation can drop the
    per-leg token and name the shared bucket instead.
    """
    if ctx.epoch <= 1:
        return tuple(ctx.digest_caches)
    prev = EpochPaths(ctx.run_dir, ctx.epoch - 1).leg("train")
    if not prev.is_file():
        raise Halt(f"epoch {ctx.epoch}'s surface must describe v{ctx.epoch - 1}'s rollouts, but "
                   f"{prev.name} does not exist. Run the previous epoch's train leg first.")
    dirs = json.loads(prev.read_text()).get("cache_dirs") or []
    if not dirs:
        raise Halt(f"{prev.name} records no cache_dirs, so the caches v{ctx.epoch - 1} actually "
                   f"wrote cannot be identified. Refusing to fall back to the base caches: that "
                   f"fallback is the stale surface itself.")
    root = Path(".cache")
    out = []
    for d in dirs:
        p = Path(d)
        p = p if p.is_absolute() else root / p
        if not p.is_dir():
            raise Halt(f"{prev.name} names cache dir {p} which is not on disk. The surface would "
                       f"be built from an incomplete set of samples.")
        out.append(p)
    return tuple(out)


def _assert_surface_is_current(ctx: Ctx, caches: tuple) -> None:
    """The stale-surface guard, placed where the cause is. Two independent ways for the surface to be stale,
    each checkable without trusting the other:

    1. It is byte-identical to the previous epoch's surface. Two epochs of rollouts producing the
       same bytes is not a coincidence any real run can produce.
    2. It shows no trace of the skills version it is supposed to describe. From epoch 2 on, the
       rollouts behind the surface ran WITH skills_v{N-1}, so a surface that contains zero
       skill-tool calls naming that library is describing different rollouts.

    Check 2 is the same assessor the uptake verify uses downstream. Running it here too is deliberate: at this
    point a failure means "the surface is wrong"; downstream the identical reading means "the
    intervention was not delivered". Same measurement, opposite conclusions — so it must be settled
    here, before an evolver reads the surface and before a gate verdict is attributed to skills.
    """
    if ctx.epoch <= 1:
        return
    out, prev_dir = ctx.paths.digests, EpochPaths(ctx.run_dir, ctx.epoch - 1).digests
    if prev_dir.is_dir():
        cur = {p.name: p.read_bytes() for p in out.glob("*.md")}
        old = {p.name: p.read_bytes() for p in prev_dir.glob("*.md")}
        if cur and cur == old:
            raise Halt(f"{out.name} is BYTE-IDENTICAL to {prev_dir.name} — it describes the same "
                       f"rollouts, so it is stale. caches used: "
                       f"{[c.name for c in caches]}")
    lib_dir = EpochPaths(ctx.run_dir, ctx.epoch - 1).promoted
    if not lib_dir.is_dir():
        return
    assess_delivery = _load_delivery_assessor()
    if assess_delivery is None:
        raise Halt("cannot assess whether the surface shows the previous version's skills: no "
                   "delivery assessor is importable. Refusing to pass a check that did not run "
                   "— agreeing signals can all be unread.")
    r = assess_delivery(lib_dir, out)
    if r["rollouts"] and r["consulted"] == 0:
        raise Halt(
            f"{out.name} contains ZERO skill-tool calls across {r['rollouts']} rollouts, but the "
            f"rollouts it claims to describe ran with {lib_dir.name} loaded. Either the surface is "
            f"built from the wrong caches or the library truly was not delivered — "
            f"and those have opposite meanings, so resolve it against the raw transcripts in "
            f"{[c.name for c in caches]} before spending an evolver on this surface.")


def _run_build_digests_ssbv2(ctx: Ctx) -> None:
    """SSB v2's evidence surface, via scripts/build_rollout_digests.py.

    WHY A SECOND BRANCH RATHER THAN A GENERALISED ONE. The two benchmarks put their evidence in
    genuinely different places: tau3's builder reads PER-SAMPLE cache dirs plus an oracle tasks
    file, SSB v2's reads ONE cache dir plus a `_snapshots` tree holding the produced workbooks —
    because for SSB v2 the artifact under review is a spreadsheet, not a conversation. Forcing one
    signature over both would mean a function whose parameters are half-ignored depending on a
    flag, which is how one benchmark's assumptions get baked in invisibly.

    COVERAGE. An epoch-1 surface can have half its samples without a scorer diff and pass every
    membership gate. This runner:
      * forces EVOSKILL_N_SAMPLES (the spec's) and EVOSKILL_DIGEST_REQUIRE_SCORER_DIFF=1 with
        env.update, so the builder refuses any gap and leaves no half-surface behind;
      * refuses an exported EVOSKILL_DIGEST_CACHE / _BUCKET / _SKILLS that disagrees with the
        declared value (an export would otherwise REPLACE it silently) and scrubs them from the
        builder's env;
      * builds epoch 1 from declared `digest_snapshot_roots` when given (a train-only view over
        all of them), because a replayed base leg's snapshots are scattered across the tags that
        first scored each rollout;
      * does not trust an existing *.md set: the idempotent path re-derives coverage from the
        bytes and runs `_assert_surface_is_current`.
    """
    out = ctx.paths.digests
    if ctx.epoch >= 2:
        # Epoch N's surface digests epoch N-1's rollouts, and the loop itself
        # CREATED that cache (one dir per epoch, exactly one skill-version bucket —
        # the reason per-epoch dirs exist). Derive it; the RUNTIME `digest_caches`
        # declaration names the BASE legs' shared 17-bucket dir and is correct only
        # for epoch 1, so trusting it here re-digests the wrong version's evidence.
        cache = REPO / ".cache" / f"{ctx.spec.name}_e{ctx.epoch - 1}_cache"
    else:
        cache = ctx.digest_caches[0] if ctx.digest_caches else ""
    if out.is_dir() and any(out.glob("*.md")):
        # An existing surface is NOT trusted as built: a hand build, a surface left by an older
        # builder, or one copied in, would otherwise reach the evolver unchecked.
        _halt_on_digest_gaps(ctx, [out])
        _assert_surface_is_current(ctx, (Path(cache),) if cache else ())
        # Rollouts layout: an existing digest surface of THIS epoch without its rollout evidence gets it
        # added (never rebuilt: an INDEX.json already there is re-derived by the steps instead).
        _ensure_rollout_evidence(ctx, out, cache, ctx.digest_bucket if ctx.epoch == 1 else "")
        return                                       # idempotent
    # THE BUCKET MUST BE NAMED WHEN THE CACHE HOLDS MORE THAN ONE. The builder refuses otherwise
    # — "refusing to mix skill versions" — and it is right to: a surface built across two skill
    # versions is evidence about no single agent. The shared base cache holds 17 buckets, so
    # epoch 1 declares which. Epochs 2+ read a PER-EPOCH cache dir written by that epoch's own
    # skill leg, which therefore holds exactly one bucket and needs no declaration.
    bucket = ctx.digest_bucket if ctx.epoch == 1 else ""
    skills = str(EpochPaths(ctx.run_dir, ctx.epoch - 1).promoted) if ctx.epoch > 1 else ""
    library = _producing_library(ctx)
    opts = _digest_options(ctx)
    _refuse_inherited_digest_env({"EVOSKILL_DIGEST_CACHE": str(cache) if cache else "",
                                  "EVOSKILL_DIGEST_BUCKET": bucket,
                                  "EVOSKILL_DIGEST_SKILLS": skills,
                                  "EVOSKILL_DIGEST_LIBRARY": library,
                                  **{k: opts.get(k, "") for k in _DIGEST_OPTION_KEYS}})
    if ctx.epoch >= 2:
        if not cache.is_dir():
            raise Halt(f"expected the previous epoch's cache at {cache} and it is absent — "
                       f"cannot build the e{ctx.epoch} surface from nothing")
        # Must mirror the runner's TASKS_CACHE = .cache/ssbv2_{TAG}_tasks
        # (the SSB v2 runner) under the leg tag scheme {name}_v{N}_{key};
        # the surface is train-only, so only the train leg's workbooks are needed.
        snaps = (REPO / ".cache" /
                 f"ssbv2_{ctx.spec.name}_v{ctx.epoch - 1}_train_tasks" / "_snapshots")
    else:
        if not cache:
            raise Halt(
                "no digest cache declared for the SSB v2 evidence surface. Set `digest_caches` in "
                "RUNTIME.json to the ONE run-cache dir the base/skill leg wrote into (SSB v2 uses a "
                "single dir, not tau3's per-sample dirs). Naming it is not optional: the default is a "
                "per-tag dir that is EMPTY for a new tag, and an empty cache builds an empty surface.")
        roots = tuple(getattr(ctx, "digest_snapshot_roots", ()) or ())
        snaps = (_ssb_snapshot_view(ctx, roots) if roots
                 else Path(str(cache).replace("_cache", "_tasks")) / "_snapshots")
    if not snaps.is_dir():
        raise Halt(f"no snapshots at {snaps}. SSB v2 digests need the PRODUCED WORKBOOKS, which "
                   f"live beside the run cache in <tag>_tasks/_snapshots — the cache holds "
                   f"transcripts only. Point digest_caches at a leg whose task dirs survive, or "
                   f"(epoch 1) declare every root that holds them in RUNTIME.json "
                   f"`digest_snapshot_roots`.")
    env = {k: v for k, v in os.environ.items() if k not in _DIGEST_ENV_OVERRIDES}
    env.update({
        "EVOSKILL_DIGEST_CACHE": str(cache),
        "EVOSKILL_DIGEST_SNAPS": str(snaps),
        "EVOSKILL_DIGEST_OUT": str(out),
        "EVOSKILL_DIGEST_SPLIT_FILE": str(ctx.spec.split_path),
        "EVOSKILL_DIGEST_SPLIT_KEY": "train",        # train-only surface, by construction
        "EVOSKILL_SOLVER_MODEL": ctx.spec.solver_model,
        "EVOSKILL_DIGEST_LEG": "base" if ctx.epoch == 1 else "skill",
        # Forced, never inherited: N decides which records resolve to a sample at all, and
        # REQUIRE is what turns a gap into a refusal instead of a silently thinner surface.
        "EVOSKILL_N_SAMPLES": str(int(ctx.spec.n_samples)),
        "EVOSKILL_DIGEST_REQUIRE_SCORER_DIFF": "1",
    })
    if bucket:
        env["EVOSKILL_DIGEST_BUCKET"] = bucket
    if skills:
        env["EVOSKILL_DIGEST_SKILLS"] = skills
    if library:                          # all-epochs mode, epoch >= 2 only
        env["EVOSKILL_DIGEST_LIBRARY"] = library
    env.update(opts)                     # always stated (AO mode: AO_DIGEST_OPTIONS)
    r = subprocess.run([sys.executable, "scripts/build_rollout_digests.py"],
                       cwd=str(REPO), env=env, capture_output=True, text=True)
    if r.returncode != 0:
        hint = ""
        if "COVERAGE REFUSED" in (r.stderr or ""):
            hint = ("\n  REMEDY: the builder refused a surface with holes and wrote nothing. For "
                    "missing scorer diffs at epoch 1, declare every snapshot root that holds the "
                    "replayed base leg's workbooks in RUNTIME.json `digest_snapshot_roots`; for "
                    "missing samples or tasks, finish the leg first. Then re-run --step.")
        raise Halt(f"the SSB v2 digest builder exited {r.returncode}: "
                   f"{(r.stderr or r.stdout)[-1200:]}{hint}")
    if not any(out.glob("*.md")):
        raise Halt(f"the digest builder exited 0 but wrote no *.md into {out}. Exit 0 is not "
                   f"evidence of an artifact.\n{(r.stdout or '')[-400:]}")
    # The builder's census is printed, not discarded with the rest of its stdout on exit 0.
    for line in (r.stdout or "").splitlines():
        if line.startswith(("COVERAGE:", "REQUIRE mode:", "WARNING", "DIGEST OPTIONS:", "DIGEST INFO:")):
            print(f"  [digest builder] {line}")
    _halt_on_digest_gaps(ctx, [out])
    _assert_surface_is_current(ctx, (Path(cache),))
    # Rollouts layout: the full traces + grades + INDEX, AFTER the digests (the digest
    # builder's REQUIRE mode refuses a non-empty OUT), from the same leg, cross-checked against the
    # same cache the digests were built from.
    _ensure_rollout_evidence(ctx, out, cache, bucket)


# Read directly by the builder, so a leftover export in the operator's shell would change WHICH
# rollouts the loop's surface describes: CACHE would replace the declared epoch-1 cache (and the
# snapshot root derived from it), BUCKET the bucket at every epoch, and SKILLS would reach the base
# build. The builder's content options: the loop sets them in template mode (full grader output on;
# the changed-cells list from RUNTIME evolver_daytona.changed_cells, absent = off) and in AO mode
# (AO_DIGEST_OPTIONS); an export would otherwise change what the evolver is shown in a run that never
# declared it.
_DIGEST_OPTION_KEYS = ("EVOSKILL_DIGEST_GRADER_OUTPUT_FULL", "EVOSKILL_DIGEST_CHANGED_CELLS",
                       "EVOSKILL_DIGEST_CHANGED_CELLS_MAX")
_DIGEST_ENV_OVERRIDES = ("EVOSKILL_DIGEST_CACHE", "EVOSKILL_DIGEST_BUCKET", "EVOSKILL_DIGEST_SKILLS",
                         # The producing library. Set by the loop in all-epochs mode only; an
                         # export would otherwise turn the snapshot and anchored markers on (or
                         # point them at another set) in a run that never declared them.
                         "EVOSKILL_DIGEST_LIBRARY", *_DIGEST_OPTION_KEYS)


# AO mode's digest options, stated explicitly: the builder's DEFAULTS are changed cells off and grader
# output in full, so leaving AO mode on the defaults would change its evidence mid-run -- an AO prompt
# describing "every cell the solver changed" (ITER1_PROMPT.md) would name a section no longer there.
AO_DIGEST_OPTIONS = {"EVOSKILL_DIGEST_GRADER_OUTPUT_FULL": "0",      # the 2500+800 cut, now marked
                     "EVOSKILL_DIGEST_CHANGED_CELLS": "1",           # "What the SOLVER changed" on
                     "EVOSKILL_DIGEST_CHANGED_CELLS_MAX": "60"}      # _solver_edit_section's old max_cells


def _digest_options(ctx: Ctx) -> dict:
    """The digest builder's content options this run declares. prompt_mode=template: the grader's
    output IN FULL (the renderer refuses a clipped one) and the changed-cells list only under its grant
    (RUNTIME evolver_daytona.changed_cells; absent = off); the cap is the builder's. AO mode:
    AO_DIGEST_OPTIONS, set explicitly, so an AO-mode epoch is built exactly as its earlier epochs were."""
    if getattr(ctx, "prompt_mode", "ao") != "template":
        return dict(AO_DIGEST_OPTIONS)
    return {"EVOSKILL_DIGEST_GRADER_OUTPUT_FULL": "1",
            "EVOSKILL_DIGEST_CHANGED_CELLS": "1" if _changed_cells_grant(ctx) else "0"}


def _producing_library(ctx: Ctx) -> str:
    """The promoted set this epoch's surface was produced by, as the builders' LIBRARY input: the
    absolute skills_v{N-1}, in all-epochs mode from epoch 2 on. "" otherwise, and then neither
    builder snapshots a library or changes a byte (only a surface an evolver reads after a
    later one exists needs its markers anchored to a version)."""
    if ctx.epoch < 2 or _evidence_mode(ctx) != "all_epochs":
        return ""
    return str(EpochPaths(ctx.run_dir, ctx.epoch - 1).promoted.resolve())
# Where epoch-1 snapshot views are assembled. Under .cache, NEVER the run dir: the view is a tree
# of symlinks into roots that also hold val snapshots, and the run dir is what the evolver's
# sandbox grants subpaths of. Module-level so fixtures can point it at a temp dir.
_SNAPVIEW_BASE = REPO / ".cache"


def _refuse_inherited_digest_env(want: dict, keys: tuple = _DIGEST_ENV_OVERRIDES) -> None:
    """HALT if the shell exports a digest input that disagrees with what the loop will pass.

    Equal values are harmless and are noted; the builder's env is scrubbed of these keys either
    way and only the declared values are set. Paths compare resolved, so a relative export of the
    same dir is not a disagreement.
    """
    def same(k, a, b):
        if k == "EVOSKILL_DIGEST_BUCKET" or k in _DIGEST_OPTION_KEYS or not (a and b):
            return a == b
        return Path(a).expanduser().resolve() == Path(b).expanduser().resolve()
    for k in keys:
        got = os.environ.get(k) or ""
        if not got:
            continue
        exp = want.get(k, "")
        if not same(k, got, exp):
            raise Halt(
                f"{k}={got!r} is exported in this shell, but the loop's declared value is "
                f"{exp or '(none)'!r}. The builder reads it directly, so it would REPLACE the "
                f"declaration silently and change which rollouts the evidence surface describes. "
                f"Unset it (`unset {k}`), or declare the value in RUNTIME.json "
                f"(`digest_caches` / `digest_bucket`) so it is on the record, then re-run.")
        print(f"  note: {k} is exported and equals the declared value; ignored — the loop sets "
              f"the builder's env itself")


def _ssb_snapshot_view(ctx: Ctx, roots) -> Path:
    """Epoch 1 with declared `digest_snapshot_roots`: ONE train-only view over all of them.

    scripts/ssbv2_make_snapshot_view.py does the hard part: it links each train task from the one
    root that holds it, refuses a task found under two roots (two runs scored it, so the diff may
    not belong to the cached rollout), refuses a linked sample dir without DIFF.txt, and refuses
    any out-of-split task. The builder then requires one diff per SAMPLE, which the view does not.
    """
    view = (_SNAPVIEW_BASE / f"ssbv2_{Path(ctx.run_dir).resolve().name}_e{ctx.epoch}_snapview"
            / "_snapshots_train")
    env = dict(os.environ)
    env.update({"EVOSKILL_VIEW_SPLIT_FILE": str(ctx.spec.split_path),
                "EVOSKILL_VIEW_SPLIT_KEY": "train",
                "EVOSKILL_VIEW_OUT": str(view)})
    r = subprocess.run([sys.executable, "scripts/ssbv2_make_snapshot_view.py",
                        *[str(x) for x in roots]],
                       cwd=str(REPO), env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise Halt(f"the declared digest_snapshot_roots do not assemble into one complete, "
                   f"train-only snapshot view (exit {r.returncode}):\n"
                   f"{((r.stdout or '') + (r.stderr or ''))[-1200:]}\n"
                   f"  Fix the declaration (a missing root, or a task under two roots), then "
                   f"re-run --step.")
    return view


def _halt_on_digest_gaps(ctx: Ctx, dirs) -> None:
    ok, detail = _v_digests_complete(ctx, dirs, rollouts=False)
    if not ok:
        raise Halt(f"the evidence surface is INCOMPLETE: {detail}")


def _run_build_digests(ctx: Ctx) -> None:
    """Build the train-only evidence surface by invoking the SAME script the operator runs.

    Deliberately a subprocess call to scripts/tau3_build_digests.py rather than a reimplementation:
    that script carries the split firewall, the id redaction rule and the leg-identity assertion,
    and a second copy of that logic would drift from it (two guards, one real).

    DISPATCHES ON BENCHMARK. SSB v2's surface is built from a different script reading different
    inputs; see _run_build_digests_ssbv2.
    """
    _evidence_mode(ctx)          # a bad evolver_evidence declaration HALTs before anything builds
    if getattr(ctx.spec, "benchmark", "") == "ssbv2":
        return _run_build_digests_ssbv2(ctx)
    out = ctx.paths.digests
    if out.is_dir() and any(out.glob("*.md")):
        _assert_surface_is_current(ctx, _digest_caches_for_epoch(ctx))
        return                                       # idempotent
    spec = ctx.spec
    caches = _digest_caches_for_epoch(ctx)           # per EPOCH, not per run
    # The producing library, all-epochs mode from epoch 2 only. Never inherited: an exported
    # value that disagrees with the declaration is refused, and the key is scrubbed either way.
    library = _producing_library(ctx)
    _refuse_inherited_digest_env({"EVOSKILL_TAU3_DIGEST_LIBRARY": library},
                                 keys=("EVOSKILL_TAU3_DIGEST_LIBRARY",))
    env = {k: v for k, v in os.environ.items() if k != "EVOSKILL_TAU3_DIGEST_LIBRARY"}
    if library:
        env["EVOSKILL_TAU3_DIGEST_LIBRARY"] = library
    env.update({
        "EVOSKILL_TAU3_DOMAIN": spec.domain,
        "EVOSKILL_TAU3_VARIANT": getattr(spec, "variant", "") or "",
        "EVOSKILL_TAU3_DIGEST_OUT": str(out),
        "EVOSKILL_TAU3_DIGEST_SPLIT_FILE": str(spec.split_path),
        "EVOSKILL_TAU3_DIGEST_SPLIT_KEY": "train",
        "EVOSKILL_TAU3_ORACLE_TASKS": str(ctx.oracle / "tasks" / f"{spec.domain}.json"),
        "EVOSKILL_TAU3_DIGEST_LEG": "base" if ctx.epoch == 1 else "skill",
        "EVOSKILL_TAU3_DIGEST_CACHES": ",".join(str(c) for c in caches),
        "EVOSKILL_TAU3_CACHE_TAG": spec.cache_tag,
    })
    # With no explicit caches the builder falls back to DERIVING the cache dir name, and the
    # tag it would derive from is the spec hash — which names this arm's identity, not the
    # directory the rollouts actually wrote. It would then refuse on a missing dir (loudly, so
    # nothing silently builds from the wrong place), but the message would blame the cache
    # rather than the caller. Refuse here instead, where the cause is.
    if not caches:
        raise Halt("no digest_caches declared. Naming them is not optional: the dir name is "
                   "owned by the runner, and a second derivation here is the two-copies defect "
                   "the builder's own comment records twice over.")
    r = subprocess.run([sys.executable, "scripts/tau3_build_digests.py"],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise Halt(f"digest build exited {r.returncode}: {(r.stderr or r.stdout)[-400:]}")
    _assert_surface_is_current(ctx, caches)


EVOLVER_LOCK = "EVOLVER.lock"            # <run>.operator/: held (flock) for the whole evolver dispatch
EVOLVER_PID = "EVOLVER.pid.json"         # <run>.operator/: the holder's pid and start time, for a person to read


def _run_evolver(ctx: Ctx) -> None:
    """Dispatch the evolver for this epoch through its driver (evolver_host / placement).

    THE LOOP DOES NOT JUDGE THE OUTPUT. What happens here is purely mechanical: hand an already
    checked prompt to an already-chosen model, under an isolation boundary the driver owns.

    ONE EVOLVER PER RUN AT A TIME. Two dispatches of one run would contend for its one session, its
    staging and its machine. The rule is held as an exclusive flock on <run>.operator/EVOLVER.lock
    for the whole dispatch (the kernel releases it if this process dies), with the holder's pid and
    start time in EVOLVER.pid.json; a second dispatch of the same run HALTs naming the holder.
    Nothing on the host is matched by process name, so other runs and other programs never count.

    NEVER RETRIED. A contaminated evolver run re-run is contaminated again. `advance()` calls a
    runner once and then re-verifies; if the skills that come back do not load, that is a HALT for
    a person, not a loop to spin. Only INFRA failures may ever justify a rerun, never quality.
    """
    st = ctx.paths.staging
    if st.is_dir() and _skill_fingerprint(st):
        return                                       # idempotent: this epoch already evolved
    _mode = _evidence_mode(ctx)                      # HALT on all_epochs + uncontained, before spend
    _session = _evolver_session(ctx)                 # HALT on a stateful run the loop cannot serve, before spend
    od = operator_dir(ctx.run_dir)
    pidf = od / EVOLVER_PID
    with _flock(od / EVOLVER_LOCK, blocking=False) as got:
        if not got:
            try:
                holder = json.loads(pidf.read_text())
            except (OSError, ValueError):
                holder = {}
            raise Halt(f"another evolver dispatch of this run is live (pid {holder.get('pid', '?')}, epoch "
                       f"{holder.get('epoch', '?')}, since {holder.get('at', '?')}; it holds "
                       f"{od.name}/{EVOLVER_LOCK}). One evolver per run at a time: wait for it to end.")
        rec = {"pid": os.getpid(), "lstart": _proc_lstart(os.getpid()), "epoch": ctx.epoch, "at": _stamp()}
        tmp = pidf.with_name(f".{pidf.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(rec))
        os.replace(str(tmp), str(pidf))
        try:
            _dispatch_evolver(ctx, _mode, _session)
        finally:
            try:
                pidf.unlink()
            except OSError:
                pass


def _dispatch_evolver(ctx: Ctx, _mode: str, _session: str) -> None:
    """_run_evolver's body, run while this process holds the run's evolver lock."""
    st = ctx.paths.staging
    _host = _evolver_host(ctx)                       # HALT on a bad host declaration, before spend

    prompt = ctx.paths.prompt
    if not prompt.is_file() or not prompt.read_text().strip():
        raise Halt(f"no usable prompt at {prompt}")

    from .quota_scope import consumes_claude_quota, subscription_env

    env = dict(os.environ)
    env.pop("EVOSKILL_TAU3_GATE_ONLY", None)         # the launcher must never inherit gate-only mode
    # A first-party Claude evolver runs on the run's one credential and must not inherit an
    # OpenRouter/proxy base URL left over from solver work; a non-Claude evolver reaches its model
    # THROUGH that proxy and must keep it (quota_scope.consumes_claude_quota: one rule, both sites).
    _on_subscription = consumes_claude_quota(ctx.spec.evolver_model)

    # CALLED UNCONDITIONALLY, BECAUSE IT BRANCHES INTERNALLY: its gateway half strips
    # ANTHROPIC_API_KEY so `claude` cannot present an Anthropic credential to a gateway, and its
    # Claude half keeps the key only on an API-key run (no token file given).
    env = subscription_env(ctx.spec.evolver_model, env)
    # Nothing that identifies or reaches THIS (the operator's) Claude Code session
    # travels to the evolver: messaging socket/token, session and bridge ids, job dir and kind,
    # pid. One list (evolver_dispatch.PARENT_SESSION_ENV); the launcher scrubs again and records
    # the NAMES in its receipt, told here which ones this copy already lost.
    from .evolver_dispatch import scrub_parent_session_env
    env, _parent_scrubbed = scrub_parent_session_env(env)
    env["EVOSKILL_PARENT_ENV_SCRUBBED"] = ",".join(_parent_scrubbed)
    if _on_subscription:
        for k in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT"):
            env.pop(k, None)
    # No model/refusal fallback, no auto-update, effort in env, subagents forced
    # onto an explicit claude-* evolver model. The launcher applies the same pins itself (same
    # module, `claude_pins --shell`), so a hand launch is pinned too; this copy is belt and braces.
    from .claude_pins import ModelPinError, pinned_env
    try:
        env = pinned_env(ctx.spec.evolver_model, env)
    except ModelPinError as e:
        raise Halt(f"evolver pins refused: {e}")
    env.update({
        "EVOSKILL_TAU3_DOMAIN": ctx.spec.domain,
        "EVOSKILL_EVOLVER_MODEL": ctx.spec.evolver_model,
        # keyed on the RUN DIR, not spec.name: two runs sharing a name would share this dir.
        "EVOSKILL_TAU3_WORK_TAG": run_id(ctx),       # per-RUN isolated cwd
        # ABSOLUTE, and deliberately the OPPOSITE of what _run_leg passes: a leg launcher builds
        # `RUNDIR=$R/${EVOSKILL_TAU3_RUN_DIR}` so it needs a repo-RELATIVE path, while the evolver
        # side forms its log path from the value inside a `cd "$WORK"` subshell, where a relative
        # path would resolve against the ISOLATED WORK DIR and the output stream would land nowhere.
        "EVOSKILL_TAU3_RUN_DIR": str(Path(ctx.run_dir).resolve()),
        # KERNEL CONTAINMENT FOR THE EVOLVER'S `claude` CHILD, DEFAULT ON (Ctx.evolver_sandbox),
        # with RUNTIME.json `"evolver_sandbox": false` as the recorded opt-out. The launcher builds
        # no profile of its own — it asks evolver_dispatch for the bytes — and it sandboxes only the
        # child, because the launcher itself must read the split file for its oracle firewall and
        # that file lists every held-out id.
        "EVOSKILL_EVOLVER_SANDBOX": "1" if getattr(ctx, "evolver_sandbox", True) else "0",
        "EVOSKILL_TAU3_ORACLE": str(Path(ctx.oracle).resolve()),
        # SSB v2 grades against a reference WORKBOOK. Exposing it would hand the evolver the
        # train answers, so there is no oracle to validate and the launcher's tau3-shaped
        # preflight (evaluator.py, per-domain task meta, split firewall) has nothing to check.
        # DECLARED, never inferred: if this were derived from "the oracle dir looks empty",
        # a tau3 run whose oracle build silently failed would skip its own firewall too.
        **({"EVOSKILL_ORACLE_ABSENT": "1"}
           if getattr(ctx.spec, "benchmark", "") == "ssbv2" else {}),
        "EVOSKILL_TAU3_EVOLVER_SPLIT": str(Path(ctx.spec.split_path).resolve()),
        # ABSOLUTE ROOTS THE LEAK HOOK MUST DENY. SSB v2 grades against a reference workbook,
        # so the dataset root holds the answers for every task, held-out ones included. Taken from
        # the environment only (SSBV2_ROOT); there is no default path.
        **({"EVOSKILL_LEAK_DENY_PATHS": os.environ["SSBV2_ROOT"]}
           if getattr(ctx.spec, "benchmark", "") == "ssbv2" and os.environ.get("SSBV2_ROOT") else {}),
    })
    # ALL-EPOCHS EVIDENCE CHANNEL (RUNTIME evolver_evidence). $3 stays THIS epoch's dir, so the
    # launcher's derived facts and prompt gate still describe the current batch; the EARLIER dirs
    # travel separately, absolute and resolved (the launcher's cwd is the isolated WORK dir), and
    # only in all-epochs mode. NEVER INHERITED: an exported EVOSKILL_PRIOR_DIGESTS in the
    # operator's shell would otherwise switch a current-epoch run into a different condition.
    for _k in ("EVOSKILL_PRIOR_DIGESTS", "EVOSKILL_EVOLVER_EVIDENCE"):
        env.pop(_k, None)
    if _mode == "all_epochs":
        env["EVOSKILL_EVOLVER_EVIDENCE"] = "all_epochs"
        env["EVOSKILL_PRIOR_DIGESTS"] = os.pathsep.join(
            str(p) for p in _evidence_dirs(ctx)[:-1])
    # the newest evidence dir ($3 of the Daytona drivers): the evo layout's evo_e{N-1}
    _newest_ev = ctx.paths.evo if _evidence_layout(ctx) == "evo" else ctx.paths.digests

    _harness = getattr(ctx.spec, "evolver_harness", "claude_cli")
    if _host == "daytona":
        # Positional argv: epoch, prompt, newest evidence dir, staging. The driver runs the launcher's own gates first
        # (EVOSKILL_TAU3_GATE_ONLY), then the sandbox; it returns only after the staging download
        # and the sandbox delete, so there is no detached pid to wait on.
        # placement pool_container: the same argv, receipts and exit codes from scripts/daytona/evolver_container.py.
        # The interpreter is this venv's (host_tools.python_executable: $EVOSKILL_PYTHON, else sys.executable).
        from .host_tools import python_executable
        _drv = ("scripts/daytona/evolver_container.py" if _evolver_placement(ctx) == "pool_container"
                else "scripts/daytona/tau3_evolver_daytona.py")
        argv = [python_executable(), _drv, str(ctx.epoch), str(prompt), str(_newest_ev), str(st)]
        env["EVOSKILL_EVOLVER_DAYTONA_JSON"] = json.dumps(getattr(ctx, "evolver_daytona", {}) or {})
        env["EVOSKILL_EVOLVER_TIMEOUT_S"] = str(ctx.evolver_timeout_s)
        # STATEFUL SESSION: the driver (evolver_container.py routes to evolver_session.py on RUNTIME
        # evolver_session) is told the session mode and where the epoch-1 system prompt is; both are refused
        # by it when they disagree with the run's files. NEVER INHERITED: a stale export in the operator's
        # shell must not switch a per-epoch run into a stateful dispatch.
        from src.orchestrator import stateful_layout as _sl
        for _k in (_sl.ENV_SESSION, _sl.ENV_SYSTEM_PROMPT):
            env.pop(_k, None)
        if _session == _sl.STATEFUL:
            env[_sl.ENV_SESSION] = _sl.STATEFUL
            env[_sl.ENV_SYSTEM_PROMPT] = str(Path(ctx.run_dir).resolve() / _sl.SYSTEM_PROMPT_FILE)
    else:
        # The local launchers (a claude_cli shell launcher on this host, the direct harness's runner)
        # are not part of this release: the evolver runs on the Daytona host only.
        raise Halt(f"evolver_host {_host!r} with evolver_harness {_harness!r}: only evolver_host "
                   f"\"daytona\" ships; declare it in RUNTIME.json")
    # Redirected to a FILE, never a pipe: a pipe held open by a short-lived foreground reader can
    # kill an evolver launch. The driver's own output goes to e{N}_EVOLVER_DAYTONA.out and the DRIVER
    # writes the evolver log (byte-identical to the machine's copy, sha recorded in the receipt), so
    # the log holds stream-json only. The wait allows for create + upload + collect around the
    # evolver's own timeout.
    _out = ctx.run_dir / f"e{ctx.epoch}_EVOLVER_DAYTONA.out"
    _wait_s = ctx.evolver_timeout_s + 3600
    if _evolver_placement(ctx) == "pool_container":
        # + the driver's bounded wait for the pool's current machine (evolver_daytona.machine_wait_s)
        from src.orchestrator.pool_container_evolver import machine_wait_s as _mw
        _wait_s += int(_mw(getattr(ctx, "evolver_daytona", {}) or {}))
    with open(_out, "wb") as fh:
        p = subprocess.Popen(argv, env=env, stdout=fh, stderr=subprocess.STDOUT, stdin=DEVNULL)
        from .host_tools import keep_awake
        keep_awake(p.pid)                            # best effort: the host does not sleep under it
        try:
            rc = p.wait(timeout=_wait_s)
        except subprocess.TimeoutExpired:
            raise Halt(f"evolver still running after {_wait_s}s (pid {p.pid}). It "
                       f"has NOT been killed — inspect {_out.name} and decide. Killing a live "
                       f"evolver mid-write leaves a half-written skill set that looks complete.")
    if rc != 0:
        tail = _out.read_text(errors="replace")[-500:]
        raise Halt(f"evolver exited {rc}. This is a refusal or a crash, not a result:\n  {tail}")
    _record_evidence_after(ctx)                      # the driver is synchronous
    _record_evolver_model(ctx, halt=True)


def _evolver_log_path(ctx: Ctx) -> Path:
    return Path(ctx.evolver_log) if getattr(ctx, "evolver_log", "") else \
        ctx.run_dir / f"tau3_evolver_e{ctx.epoch}.log"


def _record_evolver_model(ctx: Ctx, halt: bool = False) -> dict:
    """Which model the evolver session reported, written to
    e{N}_EVOLVER_MODEL.json with the pins it was launched under; HALT on a mismatch when asked.
    The step `evolver ran on the declared model` re-derives the same verdict from the log."""
    from .claude_pins import check_session, pin_record
    model = getattr(ctx.spec, "evolver_model", "")
    chk = check_session(_evolver_log_path(ctx), model)
    rec = {"epoch": ctx.epoch, "log": str(_evolver_log_path(ctx)),
           "harness": getattr(ctx.spec, "evolver_harness", "claude_cli"),
           "pins": pin_record(model), **{k: chk[k] for k in (
               "requested_model", "expected_model", "checked", "ok", "violations", "notes",
               "resolved_model", "claude_code_version", "claude_code_versions", "init_models",
               "assistant_models", "subagent_models", "model_usage")}}
    try:
        (ctx.run_dir / f"e{ctx.epoch}_EVOLVER_MODEL.json").write_text(json.dumps(rec, indent=1))
    except OSError as e:
        print(f"[evolver] could not write e{ctx.epoch}_EVOLVER_MODEL.json: {e}", file=sys.stderr)
    if halt and rec["checked"] and not rec["ok"] and rec["harness"] == "claude_cli":
        raise Halt(f"the evolver session did NOT run on {rec['expected_model']}: "
                   + "; ".join(rec["violations"]) + f". Its staging must not be promoted; "
                   f"recorded in e{ctx.epoch}_EVOLVER_MODEL.json.")
    return rec


def _v_evolver_model(ctx: Ctx):
    """The evolver session reported exactly the explicit model the spec names
    (every init and assistant event, subagents included), re-derived from its stream log.

    Alias models (`opus[1m]`) and the direct harness (no CC stream) are UNMEASURED, not passed:
    an alias can resolve to either Opus in CC 2.1.281, which is the defect this step exists for.
    """
    from .claude_pins import check_session, expected_model
    model = getattr(ctx.spec, "evolver_model", "")
    if getattr(ctx.spec, "evolver_harness", "claude_cli") != "claude_cli":
        return True, f"{UNMEASURED}evolver harness is not claude_cli; no CC stream to read"
    if not expected_model(model):
        return True, (f"{UNMEASURED}evolver model {model!r} is not an explicit claude-* id, so the "
                      f"resolved model is not asserted (an alias may resolve to either Opus)")
    log = _evolver_log_path(ctx)
    if not log.is_file():
        return False, f"no evolver log at {log.name} — the model it ran on cannot be shown"
    chk = check_session(log, model)
    if not chk["ok"]:
        return False, (f"the evolver session did not run on {chk['expected_model']}: "
                       + "; ".join(chk["violations"]))
    n = sum(chk["assistant_models"].values()) + sum(chk["subagent_models"].values())
    return True, (f"evolver ran on {chk['resolved_model']} (claude_code "
                  f"{chk['claude_code_version']}); {n} assistant message(s), all {model}")


def _record_evidence_after(ctx: Ctx) -> None:
    """Hash every evidence dir the receipt names, AFTER the evolver exited, into the receipt.

    The launcher recorded `evidence_sha256_before` at dispatch; _v_evolver_contained compares the
    two. A receipt without the before-hash (an older launcher) is left untouched. Best-effort
    write: a failure is recorded in the receipt, and the verify then recomputes from disk.
    """
    rec = ctx.run_dir / f"e{ctx.epoch}_EVOLVER_CONTAINMENT.json"
    try:
        d = json.loads(rec.read_text())
    except Exception:                                           # noqa: BLE001
        return
    before = d.get("evidence_sha256_before")
    if not isinstance(before, dict) or not before:
        return
    extra = {}
    try:
        from src.orchestrator.evidence_surface import hashes as _hashes
        extra["evidence_sha256_after"] = _hashes(list(before))
    except Exception as e:                                      # noqa: BLE001
        extra["evidence_sha256_after_error"] = f"{type(e).__name__}: {e}"
    # the receipt, and the launcher's trace beside it when it carries the pre-dispatch hash too
    for p in (rec, ctx.run_dir / f"tau3_evolver_e{ctx.epoch}.trace.json"):
        try:
            cur = json.loads(p.read_text())
        except Exception:                                       # noqa: BLE001
            continue
        if not isinstance(cur, dict) or "evidence_sha256_before" not in cur:
            continue
        cur.update(extra)
        tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(cur, indent=1))
        os.replace(str(tmp), str(p))


LEG_KEYS = ("train", "val")


def _dispatch_path(ctx: Ctx, key: str) -> Path:
    """Where a launched-but-unfinished leg records its pid, so a LATER PROCESS can find it.

    Not `<tag>.pid` — the launcher writes that and it survives the run that wrote it, so a stale
    one from an earlier attempt at the same tag would have us waiting on a dead stranger. This
    file is written by the dispatcher at launch, carries the pid AND the epoch it belongs to, and
    is deleted once the leg's artifact exists.
    """
    return operator_dir(ctx.run_dir) / f"{ctx.paths.leg(key).stem}.dispatch.json"


def _dispatch_record(ctx: Ctx, key: str) -> dict:
    """The dispatch record of this epoch's `key` leg ({} when there is none or it is unreadable)."""
    try:
        rec = json.loads(_dispatch_path(ctx, key).read_text())
    except (OSError, ValueError):
        return {}
    return rec if isinstance(rec, dict) else {}


def _leg_alive(pid, ident, tag: str) -> bool:
    """Is the leg process `pid` alive AND still the process that was dispatched?

    IDENTITY, NOT A NAME PATTERN. A pid outlives its owner and gets reused, so liveness alone would
    have us wait on a stranger. The dispatch records the process's start time (and whether its
    command line names the leg's tag); both are re-derived from the process itself on every call
    (`ps -p <pid> -o lstart=,command=`, the same on macOS and Linux procps). A record with no start
    time (the process had already ended when it was recorded) counts as alive only while the pid's
    command line names the leg's tag."""
    if not pid:
        return False
    ident = ident or {}
    if not ident.get("lstart"):
        return _proc_is(pid, None, (tag,))
    return _proc_is(pid, ident, (tag,) if ident.get("cmd_has_tag") else ())


def _live_runners(ctx=None) -> list:
    """Pids of THIS RUN's rollout legs that are alive now, one per leg: every dispatch record in the
    run's operator dir whose process is still the one dispatched (_leg_alive). Per run by
    construction: nothing on the host is matched by name, so another run's legs never count."""
    od = operator_dir(ctx.run_dir) if ctx is not None else None
    if od is None or not od.is_dir():
        return []
    pids = []
    for f in sorted(od.glob("*.dispatch.json")):
        try:
            rec = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        pid = int(rec.get("pid") or 0) if isinstance(rec, dict) else 0
        tag = str(rec.get("tag") or f.name[:-len(".dispatch.json")]) if pid else ""
        if pid and pid not in pids and _leg_alive(pid, rec.get("identity"), tag):
            pids.append(pid)
    return pids


def _dispatch_leg(ctx: Ctx, key: str) -> int:
    """Launch one leg and return its pid WITHOUT waiting. Idempotent."""
    out = ctx.paths.leg(key)
    if out.is_file():
        return 0                                     # already measured
    dp = _dispatch_path(ctx, key)
    if dp.is_file():
        try:
            rec = json.loads(dp.read_text())
        except (json.JSONDecodeError, OSError):
            rec = {}
        pid = int(rec.get("pid") or 0)
        if pid and rec.get("epoch") == ctx.epoch and _leg_alive(pid, rec.get("identity"), out.stem):
            return pid                               # already in flight, same epoch
    argv, env = _leg_argv_env(ctx, key)
    r = subprocess.run(argv, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise Halt(f"leg launcher exited {r.returncode} for {key}: "
                   f"{(r.stderr or r.stdout)[-400:]}")
    m = re.search(r"\bpid=(\d+)", r.stdout)
    if not m:
        raise Halt(f"launcher printed no `pid=` line for {key}, so this leg cannot be waited on "
                   f"by identity: {(r.stdout or r.stderr)[-300:]}")
    pid = int(m.group(1))
    try:
        dp.parent.mkdir(parents=True, exist_ok=True)
        dp.write_text(json.dumps({"pid": pid, "epoch": ctx.epoch, "key": key, "tag": out.stem,
                                  "identity": _proc_identity(pid, out.stem),
                                  "at": time.strftime("%Y-%m-%dT%H:%M:%S")}))
    except OSError as e:
        print(f"[leg] could not record the dispatch of {key} ({e}); it IS running as pid {pid}",
              file=sys.stderr)
    return pid


def _run_leg(ctx: Ctx, key: str) -> None:
    """Roll out one leg by invoking the SAME launcher script an operator would run.

    BOTH LEGS ARE LAUNCHED TOGETHER, AND THIS CALL WAITS ONLY FOR ITS OWN. Train and val depend on
    nothing but the promoted skill set; running them one after the other would pay both legs'
    wall-clock for no reason. `_run_author_prompt` reads `EpochPaths(run_dir, epoch-1).leg("train")`
    and never touches val, so val is a LEAF — a gate measurement with nothing downstream of it.

    The seat pool is built for this overlap: train and val draw from the same seats, with train
    admitted first.

    WHY THIS DOES NOT CHANGE WHAT AN EPOCH MEASURES. Concurrency is not an ExperimentSpec field, so
    it is in neither the cache identity nor the frozen config. The one way load COULD reach a score
    is by pushing episodes into the episode wall; `_v_no_infra_exclusions` and the outcome-reconcile
    steps check every epoch that no episode was excluded or timed out, so if load ever does start
    ending episodes the run HALTs rather than quietly measuring something else.

    THE SEAT GUARD IS A CEILING, NOT A MUTEX: it refuses a third live leg, the number the pool
    cannot absorb (unless the leg pool admits rollouts itself; see _leg_argv_env).
    """
    return _await_leg_of(ctx, key)


def _await_leg_of(ctx: Ctx, key: str) -> None:
    out = ctx.paths.leg(key)
    if out.is_file():
        return                                       # idempotent: already measured

    # Launch whatever is not yet running — MINE FIRST, so a refusal below can never leave this
    # step waiting on a sibling it launched instead of on itself.
    others = [k for k in LEG_KEYS if k != key]
    mine = _dispatch_leg(ctx, key)
    for k in others:
        try:
            _dispatch_leg(ctx, k)
        except Halt as e:
            # A sibling that will not launch is NOT this step's failure. Say so and carry on;
            # that leg's own step will meet the same refusal and halt there, where it belongs.
            print(f"[leg] sibling {k} did not launch ({e}); {key} continues alone",
                  file=sys.stderr)
    if not mine:
        return
    _await_leg(out, mine, ctx.leg_timeout_s, key, ident=_dispatch_record(ctx, key).get("identity"),
               tag=out.stem)
    try:
        _dispatch_path(ctx, key).unlink()
    except OSError:
        pass


def _leg_argv_env(ctx: Ctx, key: str):
    """(argv, env) for one leg. Split out of _run_leg so dispatch and wait are separable.

    A subprocess call, not a reimplementation, for the reason given at _run_build_digests: the
    launcher owns the org-key refusal, the provider-pin assertion, the seat pool and the cache
    identity, and a second copy of any of that would be two guards, one real.

    THE TAG IS THE CONTRACT. The launcher writes its result to `<run_dir>/<TAG>.json`, and
    EpochPaths.leg() looks for `<run_dir>/<run_dir.name>_v<epoch>_<key>.json`. Deriving the tag
    from the same expression is what makes the runner and the verify agree; a hand-picked tag
    is exactly the drift that makes checks silently SKIP on a run they were not written for.

    IT DOES NOT WAIT. `_await_leg_of` does, once both legs are in flight; see _run_leg for why
    they are launched together.
    """
    out = ctx.paths.leg(key)

    # SEAT SAFETY, AS A CEILING RATHER THAN A MUTEX: this run's live legs (_live_runners). Two legs
    # at conc 30 is the designed case (train || val); a THIRD would oversubscribe the machine, and a
    # leg finishing under contention has a different wall-clock profile from its comparator, which
    # is an instrument difference nobody records. So: two is fine, three is refused.
    live = _live_runners(ctx)
    if len(live) >= 2 and _pool_admits(ctx):
        # pipeline_val: older val legs may still run when epoch N+1's train and val
        # launch. With RUNTIME daytona_leg.pool=true every leg process is a follower of ONE pool on
        # ONE machine and the POOL admits rollouts (newest train first), so the count of leg
        # processes is not a count of seats. Without the pool the ceiling below stands unchanged.
        print(f"[leg] {len(live)} leg processes are live (pids {live}); launching {key} as one more: "
              f"RUNTIME pipeline_val + daytona_leg.pool=true, the pool admits rollouts, not this "
              f"two-leg ceiling", file=sys.stderr)
    elif len(live) >= 2:
        raise Halt(f"{len(live)} of this run's legs are already live (pids {live}); refusing to "
                   f"launch {key} as a third — two legs at conc 30 fill the machine and a "
                   f"third oversubscribes it. Wait for one to finish."
                   + (" (pipeline_val is on but RUNTIME daytona_leg.pool is not true, so the loop "
                      "keeps its two-leg ceiling.)" if _pipeline_on(ctx) else ""))

    tag = out.stem                                   # == f"{run_dir.name}_v{epoch}_{key}"
    script = ctx.leg_script
    if not Path(script).is_file():
        raise Halt(f"no leg launcher at {script}")
    # REPO-RELATIVE, NOT ABSOLUTE. A leg launcher may build `SHARED=$R/${EVOSKILL_TAU3_SHARED_DIR}`
    # and `RUNDIR=$R/${EVOSKILL_TAU3_RUN_DIR}`, so an absolute path would double-prefix into
    # `<repo>/<repo>/outputs/...`. The loop should not hand a launcher a path it cannot use.
    def _rel(x):
        q = Path(x)
        try:
            return str(q.resolve().relative_to(REPO))
        except ValueError:
            return str(q)                    # outside the repo: pass through and let it refuse

    env = dict(os.environ)
    env.update({
        "EVOSKILL_TAU3_RUN_DIR": _rel(ctx.run_dir),
        "EVOSKILL_TAU3_SHARED_DIR": _rel(ctx.shared_dir or ctx.run_dir),
        "EVOSKILL_SKILLS_DIR": str(ctx.paths.promoted),
        "EVOSKILL_TAU3_CACHE_TAG": ctx.cache_tag or ctx.spec.cache_tag,
        "EVOSKILL_TAU3_SPLIT": str(ctx.spec.split_path),
        # THE SPEC'S SAMPLE COUNT MUST REACH THE LEG, or spec.n_samples is decorative. Every leg
        # script defaults to `${EVOSKILL_N_SAMPLES:-3}`, which agrees with a spec of 3 by
        # coincidence only. An unmatched n on the two sides of a paired gate is not a smaller
        # effect, it is a different comparison. Pass it explicitly.
        "EVOSKILL_N_SAMPLES": str(ctx.spec.n_samples),
        # THE SPEC'S SOLVER MODEL MUST REACH THE LEG, or spec.solver_model is decorative: a leg
        # script's default model would silently roll under a spec that names another one, into
        # that spec's cache bucket.
        "EVOSKILL_SOLVER_MODEL": ctx.spec.solver_model,
        "EVO_CONC": "30", "EVO_CONC_MAX": "30", "EVO_PROBE_MAX": "30", "EVO_SEATS": "64",
    })
    # The interpreter the launcher runs its driver with: this venv's (host_tools.python_executable).
    from .host_tools import PYTHON_ENV, python_executable
    env.setdefault(PYTHON_ENV, python_executable())
    # RUNTIME skip_last_train: this epoch's VAL leg has no train sibling -- the run's leg pool must not hold it
    # pool_train_first_wait_s for one (scripts/daytona/ssb_leg_pool.py NO_TRAIN_SIBLING_ENV; '0' = the default).
    env["EVOSKILL_POOL_NO_TRAIN_SIBLING"] = "1" if (key == "val" and _skip_train_here(ctx)) else "0"
    _export_solver_env(ctx, env)
    if (getattr(ctx.spec, "benchmark", "") == "ssbv2"
            or getattr(ctx.spec, "solver_harness", "") == "claude_code"):
        # ONE CACHE DIR PER EPOCH. A per-TAG dir is empty for a new tag and silently re-buys every
        # episode. A per-EPOCH dir shared by that epoch's train and val legs gets the replay AND
        # leaves the digest builder exactly one skill-version bucket to read. Every claude_code leg
        # too, whatever the benchmark: scripts/ssbv2_leg_cc.sh refuses a leg without it and
        # cc_leg_driver checks the name (cc_leg_lib.cache_dir_for).
        env["EVOSKILL_CACHE_DIR"] = str(REPO / ".cache" /
                                        f"{ctx.spec.name}_e{ctx.epoch}_cache")
    # The launcher DETACHES and echoes `launched <tag> pid=<P> ...`; _dispatch_leg reads the pid
    # from that line rather than from <tag>.pid, which may be stale from an earlier attempt at
    # the same tag and would have the waiter asserting against a dead stranger.
    return (["bash", script, tag, key, ctx.paths.promoted.name], env)


# The routing variables a declared `solver_env` may carry, and the ones it scrubs when declared.
SOLVER_ENV_KEYS = ("EVOSKILL_SOLVER_PROVIDER_ORDER", "EVOSKILL_SOLVER_PROVIDER_ONLY",
                   "EVOSKILL_SOLVER_ALLOW_FALLBACKS", "EVOSKILL_SOLVER_REQUIRE_PARAMETERS",
                   "EVOSKILL_SOLVER_PROVIDER_QUANTIZATIONS", "EVOSKILL_SOLVER_EXPECT_PROVIDER",
                   "EVOSKILL_SOLVER_EXPECT_MODEL")


def _export_solver_env(ctx: Ctx, env: dict) -> None:
    """Put the DECLARED solver pins into a leg's env. No declaration: no change.

    Refuses (HALT) a declaration that disagrees with the frozen spec — the spec's solver_model
    and solver_provider_order are the pre-registration; RUNTIME may only restate them — and any
    key outside SOLVER_ENV_KEYS, because a typo'd pin that is silently ignored is a pin that never
    reaches a query.
    """
    decl = dict(getattr(ctx, "solver_env", {}) or {})
    if not decl:
        return
    bad = sorted(k for k in decl if k not in SOLVER_ENV_KEYS)
    if bad:
        raise Halt(f"RUNTIME.json solver_env names key(s) the leg does not read: {bad}. Known: "
                   f"{list(SOLVER_ENV_KEYS)}. The solver model comes from the frozen spec.")
    order = tuple(x.strip() for x in decl.get("EVOSKILL_SOLVER_PROVIDER_ORDER", "").split(",")
                  if x.strip())
    spec_order = tuple(getattr(ctx.spec, "solver_provider_order", ()) or ())
    if spec_order and order != spec_order:
        raise Halt(f"RUNTIME.json solver_env EVOSKILL_SOLVER_PROVIDER_ORDER={order} disagrees with "
                   f"the frozen spec's solver_provider_order={spec_order}. The spec is the "
                   f"pre-registration; fix the RUNTIME declaration.")
    if "EVOSKILL_SOLVER_ALLOW_FALLBACKS" in decl and bool(
            getattr(ctx.spec, "solver_allow_fallbacks", False)) != (
            decl["EVOSKILL_SOLVER_ALLOW_FALLBACKS"].strip() not in ("0", "", "false", "False")):
        raise Halt("RUNTIME.json solver_env EVOSKILL_SOLVER_ALLOW_FALLBACKS disagrees with the "
                   "frozen spec's solver_allow_fallbacks.")
    for k in SOLVER_ENV_KEYS:
        env.pop(k, None)
    env.update({k: str(v) for k, v in decl.items()})


def _await_leg(out: Path, pid: int, timeout_s: int, key: str, poll_s: float = 15.0, *,
               ident: dict | None = None, tag: str = "", on_tick=None) -> None:
    """Block until the leg is BOTH written and fully exited.

    WHY EXIT, AND NOT JUST THE ARTIFACT. A leg driver writes its result JSON shortly before it
    returns, so there is a window in which the artifact exists and the process is still alive.
    Returning inside that window is harmless on its own, but the very next step of the same epoch
    is the other leg, whose seat guard counts live legs — the loop would refuse to run val because
    train had not finished exiting, on a correct epoch, nondeterministically. Waiting on identity
    (`ident`, `tag`: _leg_alive) removes the class rather than budgeting a margin against a window
    whose width varies with teardown time.

    Ordering is deliberate: wait for the process, THEN judge the artifact. A runner that exited
    without writing has FAILED, and that is a different message from one still running.

    `on_tick` (pipeline_val only; None = unchanged): called once per poll while the leg runs, so the
    loop checks val legs that end while it waits on a train leg.
    """
    # The pid is tested BEFORE the deadline, so an already-finished leg is never reported as
    # "still running" just because the clock had run out — the artifact, not the timer, decides.
    deadline = time.time() + timeout_s
    tag = tag or out.stem
    while _leg_alive(pid, ident, tag):
        if time.time() >= deadline:
            raise Halt(f"{key} leg (pid {pid}) still running after {timeout_s}s. Do NOT relaunch "
                       f"— read failures from the PARTIAL json's `failed_episodes` key, never "
                       f"the log (it prints only the first 15 and truncates provider messages).")
        if on_tick is not None:
            on_tick()
        time.sleep(max(0.0, min(poll_s, deadline - time.time())))
    if not out.is_file():
        raise Halt(f"{key} leg (pid {pid}) EXITED WITHOUT WRITING {out.name}. This is a failed "
                   f"leg, not a slow one; relaunching without a diagnosis repeats it.")


# A verify that CANNOT be evaluated yet is not a pass. It returns True (so the loop advances —
# an unmeasurable check must not block a finished epoch) but prefixes its detail with this, and
# report() counts and renders it separately. Without this, `_v_skills_were_consulted` at the last
# epoch — where no next-epoch digest surface exists — would print as a pass and read exactly like
# the epochs that really did measure uptake. A check that reports success without having checked
# is worse than none; this keeps the (bool, str) contract every other verify already uses.
UNMEASURED = "UNMEASURED — "


def epoch_steps() -> list:
    return [
        Step("evidence surface is train-only", _v_digests_train_only, run=_run_build_digests,
             defect="held-out leak"),
        # Membership is not coverage: a surface can pass the step above with samples missing their
        # scorer diff. No runner — the remedy is
        # upstream (snapshot roots, an unfinished leg), and the HALT names it. Before the AO, so
        # no prompt is ever authored over a surface with holes.
        Step("evidence surface is complete", _v_digests_complete, defect="evidence gap"),
        # All-epochs mode only (passes trivially otherwise): every granted dir names the
        # library that produced it and carries that library's exact files, so a skill-load in an
        # OLD rollout is never credited to the current staging text of a same-named skill. No
        # runner: a surface built without its library cannot be fixed in place.
        Step("evidence names its producing library", _v_evidence_libraries,
             defect="stale skill-load marker"),
        # Daytona host only (passes trivially otherwise). The evolver on the VM also
        # gets every TRAIN task's input workbook (never a golden): proven train-only against the
        # dataset and the split BEFORE the AO writes a prompt that names the dir.
        Step("evolver inputs are train-only", _v_inputs_train_only, run=_run_build_inputs,
             defect="held-out leak"),
        # THE AO AUTHORS THE PROMPT. Placed after the evidence surface (it needs the digests)
        # and before every prompt CHECK, so the checks guard a model author exactly as they guarded
        # a human one. Authorship moved; verification did not.
        Step("evolver prompt authored by the AO", _v_prompt_authored,
             run=_run_author_prompt, defect="AO authorship"),
        # Immediately after authoring, because that is the first moment both sides have
        # necessarily spoken. No runner: the turns are written by the channels themselves, and a
        # step that could "fix" a transcript by writing one would be fabricating a conversation.
        Step("operator<->AO conversation is recorded", _v_conversation_recorded, defect="operator turns"),
        Step("evolver prompt numbers re-derive", _v_prompt_numbers, defect="prompt figures"),
        # Train ids may be named, held-out ids never.
        Step("prompt names no held-out task id, no held-out figure", _v_prompt_no_leakage, defect="held-out leak in prompt"),
        Step("prompt is raw information, not analysis", _v_prompt_not_analytical, defect="analysis in prompt"),
        # All-epochs mode only (passes trivially otherwise). Before the evolver dispatch, so it
        # runs on a prompt that names exactly the evidence it is granted.
        Step("prompt names every granted evidence dir", _v_prompt_names_evidence, defect="dir naming"),
        # Daytona host only: the inputs dir by this run's absolute path, never a
        # per-task subpath (a task id), never a dataset path.
        Step("prompt names the granted inputs dir", _v_prompt_names_inputs, defect="dir naming"),
        Step("skills load through the real consumer", _v_skills_load, run=_run_evolver,
             defect="skills not loaded"),
        # No runner: a session on the wrong model cannot be fixed in place, and
        # its staging must not be promoted. Right after the evolver step, before promotion.
        Step("evolver ran on the declared model", _v_evolver_model, defect="model alias/fallback"),
        Step("evolver ran under declared containment", _v_evolver_contained, defect="containment"),
        Step("evolver read nothing forbidden", _v_evolver_read_nothing_forbidden,
             defect="forbidden read"),
        Step("promoted set matches staging", _v_promoted_matches_staging, run=_run_promote),
        Step("train leg loaded what was promoted",
             lambda c: _v_leg_loaded_promoted(c, "train"),
             run=lambda c: _run_leg(c, "train"), defect="skills not loaded"),
        Step("val leg loaded what was promoted",
             lambda c: _v_leg_loaded_promoted(c, "val"),
             run=lambda c: _run_leg(c, "val"), defect="skills not loaded"),
        # BEFORE any outcome is read, establish that the two sides of the gate are the same
        # comparison. Placed here deliberately: a pass-rate delta computed across unequal n is
        # not a weaker result, it is a meaningless one, so this must gate the reconcile steps
        # rather than sit beside them.
        Step("train pairs comparably with base", lambda c: _v_legs_comparable(c, "train"),
             defect="unpaired comparison"),
        Step("val pairs comparably with base", lambda c: _v_legs_comparable(c, "val"),
             defect="unpaired comparison"),
        Step("train: no infra exclusions", lambda c: _v_no_infra_exclusions(c, "train")),
        Step("val: no infra exclusions", lambda c: _v_no_infra_exclusions(c, "val")),
        Step("the agent actually consulted the skills", _v_skills_were_consulted, defect="skills unused"),
        Step("train outcomes reconcile", lambda c: _v_outcomes_reconcile(c, "train"),
             defect="outcome buckets"),
        Step("val outcomes reconcile", lambda c: _v_outcomes_reconcile(c, "val"),
             defect="outcome buckets"),
    ]


# The AO-mode step each template-mode step replaces, by NAME (the list above is the one source of
# truth for everything else, so a step added there reaches template mode unchanged).
AO_AUTHOR_STEP = "evolver prompt authored by the AO"
AO_CONVERSATION_STEP = "operator<->AO conversation is recorded"
AO_INPUTS_NAMING_STEP = "prompt names the granted inputs dir"
INPUTS_STEP = "evolver inputs are train-only"
PAIRING_STEP = "{key} pairs comparably with base"
RECONCILE_STEP = "{key} outcomes reconcile"
TEMPLATE_CONVERSATION_STEP = "operator turns are recorded (no per-epoch AO)"
LEG_D4_STEP = "{key} leg has no grader failure"


def _template_steps(base: list) -> list:
    """prompt_mode=template: `base` (epoch_steps(), or whatever a caller installed in its place) with
    exactly these differences, each by step name:
      + FIRST   "infra declaration agrees"               (infra.agree; HALT on any disagreement)
      + SECOND  "benchmark plug-in is frozen and signed"  (plugin_runner.verify_frozen: every pin unchanged)
      + before the inputs step:  "newest batch has no grader failure" (outcomes.py; its runner
                                 re-grades a grader-failed sample ONCE in the grading sandbox)
      + after the inputs step:   "evolver references are train-only" (the grant; builds the dir)
                                 "clean grader is granted and code only"
      ~ "evolver prompt authored by the AO" -> "evolver prompt rendered from the template" (the
        loop renders ITER{N}_PROMPT.md, records e{N}_PROMPT_RENDER.json; no model call)
      ~ "operator<->AO conversation is recorded" -> "operator turns are recorded (no per-epoch AO)"
        (with no AO call per epoch the two-sided check could pass only for the wrong reason)
      ~ "prompt names the granted inputs dir" -> "prompt names exactly the granted dirs"
      + before the FIRST pairing / reconcile step of the legs: "<key> leg has no grader failure" for
        every leg key the list pairs or reconciles (no number of a leg -- the pairing, the reconcile
        line, the gate -- is read before its grader failures are re-graded; the runner re-grades
        once and a re-grade that fails again HALTs)
    Everything else -- every prompt check, the evolver,
    containment, promotion, the legs -- is the same Step object, in the same order. Steps the base
    list does not contain are simply not anchored (a caller's reduced list stays reduced)."""
    out = [Step("infra declaration agrees", _v_infra_agrees, defect="infra disagreement"),
           Step("benchmark plug-in is frozen and signed", _v_plugin_frozen,
                defect="drifted plug-in")]
    names = {s.name for s in base}
    anchors = {PAIRING_STEP.format(key=k) for k in LEG_KEYS} | {RECONCILE_STEP.format(key=k) for k in LEG_KEYS}
    d4_keys = [k for k in LEG_KEYS
               if PAIRING_STEP.format(key=k) in names or RECONCILE_STEP.format(key=k) in names]
    d4_placed = False
    for s in base:
        if s.name in anchors and not d4_placed:
            for k in d4_keys:
                out.append(Step(LEG_D4_STEP.format(key=k), lambda c, k=k: _v_leg_outcomes(c, k),
                                run=lambda c, k=k: _run_leg_regrade(c, k), defect="grader failed"))
            d4_placed = True
        if s.name == INPUTS_STEP:
            out.append(Step("newest batch has no grader failure", _v_batch_outcomes,
                            run=_run_batch_regrade, defect="grader failed"))
            out.append(s)
            out.append(Step("evolver references are train-only", _v_references_train_only,
                            run=_run_build_references, defect="held-out leak"))
            out.append(Step("clean grader is granted and code only", _v_grader_code_only,
                            defect="grader grant"))
        elif s.name == AO_AUTHOR_STEP:
            out.append(Step("evolver prompt rendered from the template", _v_prompt_rendered,
                            run=_run_render_prompt, defect="loop-rendered prompt"))
        elif s.name == AO_CONVERSATION_STEP:
            out.append(Step(TEMPLATE_CONVERSATION_STEP, _v_operator_turns_recorded,
                            defect="operator turns"))
        elif s.name == AO_INPUTS_NAMING_STEP:
            out.append(Step("prompt names exactly the granted dirs", _v_prompt_names_granted_dirs,
                            defect="dir naming"))
        else:
            out.append(s)
    return out


STATEFUL_LAYOUT_STEP = "evolver material laid out (raw session logs, solver workspaces, scores)"
STATEFUL_RENDER_STEP = "evolver message + system prompt rendered"
STATEFUL_CONTAINED_STEP = "evolver ran in the run's container session"
STATEFUL_READ_STEP = "evolver read nothing forbidden (session)"
STATEFUL_CONTINUED_STEP = "evolver session continued"
STATEFUL_SNAPSHOT_STEP = "evolver state snapshotted to the Mac"
TEMPLATE_RENDER_STEP = "evolver prompt rendered from the template"
TEMPLATE_GRANTED_DIRS_STEP = "prompt names exactly the granted dirs"
CONTAINED_STEP = "evolver ran under declared containment"
READ_NOTHING_STEP = "evolver read nothing forbidden"
PROMOTE_STEP = "promoted set matches staging"


def _stateful_steps(steps: list, ctx=None) -> list:
    """RUNTIME evolver_session=stateful: `steps` (the template-mode program) with these changes BY NAME (the
    `_template_steps` pattern); every other Step is the same object in the same order:
      ~ "evidence surface is train-only" -> STATEFUL_LAYOUT_STEP (stateful_evolver.run_build_evo_layout / v_evo_layout:
        <run>/evo_e{N-1}/ = the newest train leg's raw session logs, solver workspaces and scores, from the leg store)
      ~ "evolver prompt rendered from the template" -> STATEFUL_RENDER_STEP (run_render_session_prompt /
        v_session_prompt_rendered: ITER{N}_PROMPT.md and, once at epoch 1, EVOLVER_SYSTEM.md; two fixed templates)
      ~ "evolver ran under declared containment" -> "evolver ran in the run's container session"
      ~ "evolver read nothing forbidden" -> "evolver read nothing forbidden (session)"
      + AFTER it: "evolver session continued" (the same sid, --resume, the transcript grew as a prefix), "evolver
        state snapshotted to the Mac" (what the session wrote is not scanned for held-out ids)
      - GONE: the digest surface steps, the inputs / references / clean-grader grants, the score-table
        re-derivation, the prompt leakage / analysis / path-naming checks, and the three re-grade steps "newest
        batch has no grader failure" and "<key> leg has no grader failure": the leg driver grades every rollout
        with the run's frozen grade(task, snapshot) and does its own single re-grade, so no loop step re-grades
        here (loop._run_regrade -> ssb_grade_daytona.py).
    Unchanged: the evolver step (run=_run_evolver), the model step, promotion, the legs, the uptake check,
    the reconciles (a grader_failed rollout counts as never-graded).
    `ctx` is unused here (the caller keys on _evolver_session)."""
    from src.orchestrator import stateful_evolver as _se
    # The loop computes NOTHING for the evolver. The digest surface, its completeness / library steps, the
    # inputs / references / clean-grader grants, the score-table re-derivation, the prompt leakage / analysis /
    # path-naming checks are GONE for a stateful session: the evolver's material is <run>/evo_e{K}/ (evo_layout:
    # raw logs, workspaces, scores) and its two texts are fixed templates. Nor is what the session WRITES
    # scanned for held-out ids.
    rep = {
        "evidence surface is train-only": Step(STATEFUL_LAYOUT_STEP, _se.v_evo_layout,
                                               run=_se.run_build_evo_layout, defect="evo layout"),
        TEMPLATE_RENDER_STEP: Step(STATEFUL_RENDER_STEP, _se.v_session_prompt_rendered,
                                   run=_se.run_render_session_prompt, defect="loop-rendered prompt"),
        CONTAINED_STEP: Step(STATEFUL_CONTAINED_STEP, _se.v_session_contained, defect="containment"),
        READ_NOTHING_STEP: Step(STATEFUL_READ_STEP, _se.v_session_read_nothing_forbidden, defect="forbidden read"),
    }
    # Nor does the loop re-grade. The leg driver calls the run's frozen
    # grade(task, snapshot) for every rollout, re-grades a failure once and records a second one as grader_failed
    # in the leg, so STATE is final when the leg JSON exists and evo_e{K}/scores.json is never built ahead of a
    # re-grade. The re-grade steps and their runners stay in the template, per-epoch and held-out programs.
    gone = {"evidence surface is complete", "evidence names its producing library", INPUTS_STEP,
            "evolver references are train-only", "clean grader is granted and code only",
            "evolver prompt numbers re-derive", "prompt names no held-out task id, no held-out figure",
            "prompt is raw information, not analysis", "prompt names every granted evidence dir",
            TEMPLATE_GRANTED_DIRS_STEP,
            "newest batch has no grader failure", LEG_D4_STEP.format(key="train"), LEG_D4_STEP.format(key="val")}
    inserted = [Step(STATEFUL_CONTINUED_STEP, _se.v_session_continued, defect="resume"),
                Step(STATEFUL_SNAPSHOT_STEP, _se.v_session_snapshotted, defect="resume material")]
    names = {s.name for s in steps}
    anchor = READ_NOTHING_STEP if READ_NOTHING_STEP in names else PROMOTE_STEP
    out = []
    for s in steps:
        if s.name in gone:
            continue
        if s.name == PROMOTE_STEP and anchor == PROMOTE_STEP:
            out += inserted
        out.append(rep.get(s.name, s))
        if s.name == READ_NOTHING_STEP and anchor == READ_NOTHING_STEP:
            out += inserted
    return out


def _serial_steps_for(ctx: Ctx) -> list:
    """The epoch program in the serial order: epoch_steps(), transformed for prompt_mode=template, and
    again for RUNTIME evolver_session=stateful (_stateful_steps; _evolver_session HALTs on a run the stateful
    session cannot serve)."""
    base = epoch_steps()
    if getattr(ctx, "prompt_mode", "ao") == "template":
        _prompt_mode(ctx)                       # HALT on a template run the loop cannot serve
        steps = _template_steps(base)
        return _stateful_steps(steps, ctx) if _evolver_session(ctx) == "stateful" else steps
    return base


def _steps_for(ctx: Ctx) -> list:
    """The epoch program for THIS run. epoch_steps() is looked up at call time (callers and fixtures
    replace it with a zero-argument lambda); prompt_mode=template transforms whatever it returns, and
    RUNTIME pipeline_val then moves the val leg's checks out of the blocking list (_pipeline_steps)."""
    steps = _serial_steps_for(ctx)
    return _pipeline_steps(steps, ctx) if _pipeline_on(ctx) else steps


# ── PIPELINED VAL (RUNTIME pipeline_val) ──────────────────────────────────────────────────────────────
# Training rollouts are prioritised; val rollouts enter the pool only when every train rollout has finished
# or holds a slot. The next evolver starts as soon as the training rollouts are done, val rollouts overlap
# with it, and val rollouts never block the loop. The POOL (scripts/daytona, RUNTIME daytona_leg.pool)
# decides which rollout runs; this is the loop's half:
#
#   epoch N: promote v_N -> launch train_N AND val_N -> await TRAIN_N only -> every train_N check ->
#   epoch N+1: digests from train_N, render ITER{N+1}, prompt checks, evolver N+1, its checks, promote
#   v_{N+1} -> launch train_{N+1} + val_{N+1} AT ONCE even while val_N (or older) still runs -> ...
#
# WHY THIS IS SAFE. Val is a LEAF (see _run_leg): the next epoch's evidence is built from the TRAIN leg
# only (_digest_caches_for_epoch / _run_build_digests_ssbv2, split key "train"), the prompt's facts read
# train legs only, and the evolver never sees val. Val legs serve reporting and the held-out rule.
#
# WHAT HAPPENS TO THE VAL CHECKS. They are exactly the serial order's val-leg verifies (loaded what was
# promoted, grader failure incl. its re-grade runner, pairing with the val base, infra exclusions, the
# outcome reconcile), taken BY NAME from the serial program (_val_bundle), so a
# step changed there changes here. At EVERY step boundary (and every PIPELINE_POLL_S while a train leg is
# awaited) the loop polls the val legs it tracks and checks each one that has ENDED, in order, stopping at
# the first failure like advance(). A FAILED VAL CHECK NEVER HALTS EVOLUTION: it is recorded in
# <run>.operator/PIPELINE_VAL.json and printed as one line naming the epoch, the check and the remedy. A
# failure whose step has a runner (only the re-grade) gets that runner ONCE, in a background
# `--val-check --step` process, so a grading sandbox never holds up the loop either; it counts as attempted
# only once that process has STARTED (a failed start is retried at the next boundary, a bounded number of
# times). VAL NEVER BLOCKS: a val leg that DIED without a result, or outlived its own
# leg_timeout_s (counted from ITS dispatch), is RECORDED as a val failure with the exact relaunch command,
# never a HALT; any other error while polling is recorded, never a traceback. Only the END waits: after the
# last epoch the drive awaits every tracked val leg (each bounded by its own timeout) and background
# re-grade, checks it, and reports done listing every val failure. The held-out program selects only an
# epoch whose train checks AND val checks passed (re-verified).
#
# A RECORD MUST BE CURRENT. Each record names the promoted set it was dispatched for
# (promoted_sha256 = staging_sha256 of skills_v<N>). It counts only while that set is still on disk AND the
# leg is running / ended awaiting its check, or checked with its result file byte-identical to what was
# checked (or a recorded failure that has no result by nature). Anything else (an epoch redone, a result
# moved aside) is SUPERSEDED with the reason, never announced as passed, and the epoch's val leg is
# launched again when the epoch is re-run. A superseded leg that is still running, or whose result is the
# one on disk, is never adopted for the new skill set: the epoch's entry is then a recorded failure naming it.
#
# RESTART. PIPELINE_VAL.json holds every tracked val leg with its pid identity (the start time, and the leg's
# tag when the command line held it at dispatch; _leg_alive), so a restarted
# loop re-attaches to a running leg, checks a finished one from its result file, and records a dead one.
#
# SKIP LAST TRAIN (RUNTIME skip_last_train, pipeline_val only). The run's last epoch (spec n_epochs) has no
# later evolver to read its train leg, so it launches ONLY its val leg; its train-leg checks are recorded
# "skipped by design (no later evolver reads it)", and the held-out rule accepts that epoch on its val leg
# (plus every other requirement: promoted set, skills load, the evolver's checks). ON by default
# with pipeline_val (an absent key = true); RUNTIME "skip_last_train": false keeps the last epoch's train leg.
PIPELINE_VAL_FILE = "PIPELINE_VAL.json"
PIPELINE_VAL_SCHEMA = "evoskill.pipeline_val/1"
PIPELINE_POLL_S = 60.0
PIPELINE_REMEDY_SPAWN_TRIES = 5        # step boundaries at which a background re-grade that will not START is retried
TRAIN_LEG_STEP = "train leg loaded what was promoted"
VAL_LEG_STEP = "val leg loaded what was promoted"
VAL_TRACKED_STEP = "val leg launched and tracked (pipeline: checked when it ends, never awaited here)"
VAL_DIED_STEP = "val leg wrote its result (it EXITED WITHOUT WRITING it)"
VAL_TIMEOUT_STEP = "val leg ended within leg_timeout_s"
PIPELINE_VAL_STEPS = (VAL_LEG_STEP, LEG_D4_STEP.format(key="val"), PAIRING_STEP.format(key="val"),
                      "val: no infra exclusions", RECONCILE_STEP.format(key="val"))
# The train leg's own checks: at a skip_last_train epoch each is recorded SKIPPED_BY_DESIGN instead.
PIPELINE_TRAIN_STEPS = (TRAIN_LEG_STEP, LEG_D4_STEP.format(key="train"), PAIRING_STEP.format(key="train"),
                        "train: no infra exclusions", RECONCILE_STEP.format(key="train"))
SKIPPED_BY_DESIGN = "skipped by design (no later evolver reads it)"
# A tracked val leg's states that are a RECORDED VAL FAILURE (label for the drive line); the last three
# have no result by nature, so their record stays current without one.
VAL_FAILURE_STATES = {"failed": "FAILED", "died": "DIED WITHOUT A RESULT", "timed_out": "TIMED OUT",
                      "blocked": "NOT LAUNCHED"}
VAL_NO_RESULT_FAILURES = ("died", "timed_out", "blocked")
_REMEDY_CHILDREN: list = []            # Popen objects of background --val-check runs, reaped by the poll


def _pipeline_on(ctx) -> bool:
    return bool(getattr(ctx, "pipeline_val", False))


def _pool_admits(ctx) -> bool:
    """More than two leg processes may be alive only in pipeline mode AND with the leg pool declared."""
    return _pipeline_on(ctx) and bool(getattr(ctx, "leg_pool", False))


def _skip_train_epoch(ctx):
    """RUNTIME skip_last_train (pipeline_val only): the run's LAST epoch, spec n_epochs, which launches only
    its val leg. None when the key is off, pipeline_val is off, or the spec states no n_epochs."""
    if not (_pipeline_on(ctx) and bool(getattr(ctx, "skip_last_train", False))):
        return None
    n = int(getattr(getattr(ctx, "spec", None), "n_epochs", 0) or 0)
    return n or None


def _skip_train_here(ctx) -> bool:
    e = _skip_train_epoch(ctx)
    return e is not None and int(getattr(ctx, "epoch", 0) or 0) == e


def _pipeline_runtime(rt: dict) -> dict:
    """Ctx kwargs from RUNTIME pipeline_val / skip_last_train / daytona_leg.pool. Presence semantics, and a
    non-boolean is refused so a typo ("true" as a string) cannot silently mean either order. ValueError =
    the HALT main() prints. skip_last_train=true is valid only with pipeline_val=true: in the serial order
    the next epoch's evidence and the val checks both sit behind the train leg. With pipeline_val true and
    NO skip_last_train key, skip_last_train is true; an explicit value is honoured."""
    kw = {}
    if "pipeline_val" in rt:
        if not isinstance(rt["pipeline_val"], bool):
            raise ValueError(f"RUNTIME.json pipeline_val must be true or false, got {rt['pipeline_val']!r}")
        kw["pipeline_val"] = rt["pipeline_val"]
    if "skip_last_train" in rt:
        v = rt["skip_last_train"]
        if not isinstance(v, bool):
            raise ValueError(f"RUNTIME.json skip_last_train must be true or false, got {v!r}")
        if v and kw.get("pipeline_val") is not True:
            raise ValueError("RUNTIME.json skip_last_train=true is valid only with pipeline_val=true (the "
                             "pipelined order checks the last epoch's val leg on its own); set pipeline_val "
                             "to true or remove skip_last_train.")
        kw["skip_last_train"] = v
    elif kw.get("pipeline_val") is True:
        # The last epoch's train leg feeds no later
        # evolver, so the pipelined order launches only that epoch's val leg unless RUNTIME says skip_last_train false.
        kw["skip_last_train"] = True
    # The leg pool (RUNTIME daytona_leg.pool, read by the leg driver): only a literal true lifts the loop's
    # two-leg ceiling, and only in pipeline mode (Ctx.leg_pool / _pool_admits).
    if isinstance(rt.get("daytona_leg"), dict) and rt["daytona_leg"].get("pool") is True:
        kw["leg_pool"] = True
    return kw


def _proc_cmd(pid) -> str:
    """The process's command line, or "" when it is gone (a zombie's reads as its bare name)."""
    if not pid:
        return ""
    r = subprocess.run(["ps", "-p", str(int(pid)), "-o", "command="], capture_output=True, text=True)
    return r.stdout.strip()


def _proc_lstart(pid) -> str:
    if not pid:
        return ""
    r = subprocess.run(["ps", "-p", str(int(pid)), "-o", "lstart="], capture_output=True, text=True)
    return " ".join(r.stdout.split())


def _proc_identity(pid, tag: str = "") -> dict | None:
    """What a later process needs to tell THIS process from a recycled pid: its start time, and whether
    its command line names the leg's tag. The command line itself is not stored."""
    cmd = _proc_cmd(pid)
    if not cmd:
        return None
    return {"lstart": _proc_lstart(pid), "cmd_has_tag": bool(tag) and tag in cmd}


def _proc_is(pid, ident, needles) -> bool:
    """Alive AND the same process: every needle in its command line and, when recorded, the same start."""
    cmd = _proc_cmd(pid)
    if not cmd or not all(n in cmd for n in needles if n):
        return False
    want = (ident or {}).get("lstart")
    return not want or _proc_lstart(pid) == want


def _stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _file_sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _ctx_for_epoch(ctx, e: int):
    """A context for another epoch of the same run: a NEW Ctx (never a moved .epoch; see _assert_epoch_consistent)."""
    import dataclasses
    return dataclasses.replace(ctx, epoch=int(e))


def _pipeline_path(run_dir) -> Path:
    return operator_dir(run_dir) / PIPELINE_VAL_FILE


def _pipeline_read(run_dir) -> dict:
    """The tracked val legs + per-epoch train records. Absent = nothing tracked. Unreadable = HALT: it is
    the only record of which val legs are outstanding, so it is never silently replaced."""
    p = _pipeline_path(run_dir)
    if not p.is_file():
        return {"schema": PIPELINE_VAL_SCHEMA, "run_dir": str(Path(run_dir).resolve()), "legs": {},
                "train": {}, "superseded": []}
    try:
        d = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise Halt(f"{p} is unreadable ({type(e).__name__}: {e}). It is the only record of which val legs "
                   f"are outstanding (pids) and how their checks ended; a person reads and repairs it, "
                   f"the loop never replaces it.")
    if not isinstance(d, dict) or d.get("schema") != PIPELINE_VAL_SCHEMA:
        raise Halt(f"{p} is not a {PIPELINE_VAL_SCHEMA} record; a person reads it.")
    for k, v in (("legs", {}), ("train", {}), ("superseded", [])):
        d.setdefault(k, v)
    return d


@contextlib.contextmanager
def _flock(path: Path, blocking: bool = True):
    """An exclusive flock on `path`; yields whether it was acquired (always True when blocking)."""
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(path, "a")
    got = False
    try:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
            got = True
        except OSError:
            got = False
        yield got
    finally:
        if got:
            fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


def _pipeline_update(run_dir, fn):
    """Read-modify-write PIPELINE_VAL.json under its lock (the drive and a background --val-check both
    write it), atomically. `fn(d)` mutates the freshly read record; its return value is returned."""
    p = _pipeline_path(run_dir)
    with _flock(p.with_name(p.name + ".lock")):
        d = _pipeline_read(run_dir)
        r = fn(d)
        import uuid as _uuid
        tmp = p.with_name(f".{p.name}.{os.getpid()}.{_uuid.uuid4().hex[:8]}.tmp")
        tmp.write_text(json.dumps(d, indent=1, sort_keys=True, default=str))
        os.replace(str(tmp), str(p))
        return r


def _val_record(run_dir, e) -> dict | None:
    return _pipeline_read(run_dir)["legs"].get(str(int(e)))


def _val_check_cmd(c) -> str:
    return (f"python -m src.orchestrator.loop --run-dir {c.run_dir} --epoch {c.epoch} --val-check --step")


def _pipeline_steps(steps: list, ctx=None) -> list:
    """pipeline_val's epoch program: `steps` (the serial program) minus the val leg's checks, which move
    to _val_bundle and run when the val leg ends; the train leg step's runner launches train AND val and
    awaits TRAIN only; right after it, a step makes sure the val leg is launched and tracked.
    Every other step is the same object, in the same order. Matched BY NAME, like _template_steps.

    At the skip_last_train epoch (`ctx` given, _skip_train_here): every train-leg check (PIPELINE_TRAIN_STEPS)
    keeps its name and is recorded SKIPPED_BY_DESIGN (no runner: no train leg is launched), and the tracker
    step launches the val leg."""
    skip = ctx is not None and _skip_train_here(ctx)
    out = []
    for s in steps:
        if s.name in PIPELINE_VAL_STEPS:
            continue
        if skip and s.name in PIPELINE_TRAIN_STEPS:
            out.append(Step(s.name, _v_train_skipped_by_design, defect=s.defect))
            if s.name == TRAIN_LEG_STEP:
                out.append(Step(VAL_TRACKED_STEP, _v_val_tracked, run=_run_track_val, defect="pipeline_val"))
        elif s.name == TRAIN_LEG_STEP:
            out.append(Step(s.name, s.verify, run=_run_train_leg_pipelined, defect=s.defect,
                            operator_gate=s.operator_gate))
            out.append(Step(VAL_TRACKED_STEP, _v_val_tracked, run=_run_track_val, defect="pipeline_val"))
        else:
            out.append(s)
    return out


def _v_train_skipped_by_design(ctx: Ctx):
    """A train-leg check at the skip_last_train epoch: not a pass (UNMEASURED renders it SKIP on the board)."""
    return True, (f"{UNMEASURED}{SKIPPED_BY_DESIGN}: RUNTIME skip_last_train, epoch {ctx.epoch} is the run's last "
                  f"epoch (spec n_epochs); no train leg is launched and none is checked. Its val leg is "
                  f"launched and checked, and the held-out rule accepts this epoch on its val leg.")


def _val_bundle(ctx) -> list:
    """The val leg's checks, in the serial program's order, taken from it BY NAME (template mode's
    grader-failure step and its re-grade runner included). The val-loaded step loses its runner: in pipeline mode that
    runner (launch + await) is the train step's and the tracker's."""
    return [Step(s.name, s.verify, run=None if s.name == VAL_LEG_STEP else s.run, defect=s.defect)
            for s in _serial_steps_for(ctx) if s.name in PIPELINE_VAL_STEPS]


def _promoted_sha(c) -> str | None:
    """staging_sha256 of epoch c.epoch's PROMOTED set, or None when it is absent (an EMPTY set has a hash too:
    the solver runs with no skills)."""
    p = c.paths.promoted
    if not p.is_dir():
        return None
    return staging_sha256(p)


def _val_status(c, rec) -> tuple:
    """(current?, why, kind) of epoch c.epoch's val record. Read-only (a --plan verify calls it).

    CURRENT = dispatched for the promoted set NOW on disk (promoted_sha256), and still a live record of
    that leg: running (alive; or ended and awaiting its check; or ended WITHOUT a result, which the next
    boundary records as a val failure), a recorded failure that has no result by nature (died / timed out
    / not launched), or checked (passed / failed / re-grading) with its result file present and
    byte-identical to what was checked. `kind` says why not: none | tag | promoted | result_gone |
    result_changed."""
    out = c.paths.leg("val")
    if rec is None:
        return False, f"no epoch-{c.epoch} entry in {PIPELINE_VAL_FILE}", "none"
    if rec.get("tag") != out.stem:
        return False, f"its entry names {rec.get('tag')}, not {out.stem}", "tag"
    have = _promoted_sha(c)
    if have is None:
        return False, (f"the promoted set {c.paths.promoted.name} it was dispatched for is gone (the epoch "
                       f"was redone or moved aside)"), "promoted"
    want = rec.get("promoted_sha256")
    if want != have:
        return False, (f"it was dispatched for promoted set {str(want)[:12]}, the promoted set on disk is "
                       f"{have[:12]} (the epoch was redone)"), "promoted"
    st = rec.get("state")
    if st == "running":
        if _val_alive(c, rec):
            return True, "running", ""
        return True, ("ended; its checks run at the next step boundary" if out.is_file() else
                      "ended WITHOUT A RESULT; the next step boundary records it as a val failure"), ""
    if st in VAL_NO_RESULT_FAILURES:
        return True, f"{st}: a recorded val failure for this promoted set", ""
    if not out.is_file():
        return False, f"it is recorded {st}, but its result {out.name} is gone", "result_gone"
    if _file_sha(out) != rec.get("leg_sha256"):
        return False, f"its result {out.name} changed since it was checked ({st})", "result_changed"
    return True, str(st), ""


def _v_val_tracked(ctx: Ctx):
    """The val leg of this epoch is launched and in PIPELINE_VAL.json (with its pid identity) for the
    promoted set on disk (_val_status), or its result already exists and is tracked. Its CHECKS are not
    this step: they run when it ends. A recorded val failure (died / timed out / not launched) is tracked:
    relaunching it is the operator's call, with the command the record names."""
    out = ctx.paths.leg("val")
    rec = _val_record(ctx.run_dir, ctx.epoch)
    ok, why, _kind = _val_status(ctx, rec)
    if not ok:
        return False, (f"val leg {out.stem} is not launched and tracked for the promoted set on disk: {why} "
                       f"({operator_dir(ctx.run_dir).name}/{PIPELINE_VAL_FILE})")
    st = rec.get("state")
    f = rec.get("failure") or {}
    return True, (f"val leg {out.stem} tracked: {st}"
                  + (f" (pid {rec.get('pid')})" if st == "running" and rec.get("pid") else "")
                  + (f" — {why}" if st == "running" and why != "running" else "")
                  + (f"; a RECORDED val failure at '{f.get('step')}' (remedy: {f.get('remedy')})"
                     if st in VAL_FAILURE_STATES else "")
                  + "; its checks run when it ends and never halt evolution")


def _pipeline_supersede(c, rec: dict, why: str, readopt: dict | None = None) -> None:
    """Move epoch c.epoch's val record to `superseded` with the reason and what was true at that moment
    (was the promoted set changed? was its process alive? which result bytes were on disk?), and put
    `readopt` in its place if given, in ONE locked write. _stale_val_guard reads these notes."""
    out = c.paths.leg("val")
    have = _promoted_sha(c)
    note = {"superseded_at": _stamp(), "superseded_t": time.time(), "why": str(why),
            "promoted_changed": rec.get("promoted_sha256") != have,
            "result_sha256_at_supersede": _file_sha(out) if out.is_file() else None,
            "pid_alive_at_supersede": bool(rec.get("pid")) and _val_alive(c, rec)}

    def put(d):
        old = d["legs"].pop(str(int(c.epoch)), None)
        if old is not None:
            d["superseded"].append({**old, **note})
        if readopt is not None:
            d["legs"][str(int(c.epoch))] = readopt
    _pipeline_update(c.run_dir, put)


def _stale_val_guard(c) -> tuple | None:
    """(detail, remedy) when epoch c.epoch's val leg must NOT be launched or adopted now, else None: a
    superseded val leg of this tag, dispatched for a DIFFERENT promoted set, is still running (it would
    write this very result file, and the launcher would re-attach to it), or the result on disk is, or may
    be, that superseded leg's. Read-only."""
    reg = _pipeline_read(c.run_dir)
    out = c.paths.leg("val")
    have = _promoted_sha(c)
    cmd = _val_check_cmd(c)
    res_sha = _file_sha(out) if out.is_file() else None
    for s in reversed(reg.get("superseded") or []):
        if str(s.get("epoch")) != str(int(c.epoch)) or s.get("tag") != out.stem:
            continue
        if s.get("promoted_sha256") == have:
            continue                                # the same skill set: nothing stale about it
        if s.get("pid") and _val_alive(c, s):
            return ((f"the superseded val leg {out.stem} (pid {s['pid']}, dispatched {s.get('dispatched_at')} "
                     f"for promoted set {str(s.get('promoted_sha256'))[:12]}) is STILL RUNNING; it would write "
                     f"{out.name} from the OLD skill set, and the launcher would re-attach to it"),
                    f"stop pid {s['pid']}, then relaunch with: {cmd}")
        if res_sha and (res_sha in (s.get("result_sha256_at_supersede"), s.get("leg_sha256"))
                        or (s.get("pid_alive_at_supersede")
                            and out.stat().st_mtime >= float(s.get("superseded_t") or 0))):
            return ((f"{out.name} on disk (sha256 {res_sha[:12]}) is, or may be, the result of the superseded "
                     f"val leg (promoted set {str(s.get('promoted_sha256'))[:12]}), not of the promoted set on "
                     f"disk ({str(have)[:12]})"),
                    f"move {out.name} aside (keep it), then relaunch with: {cmd}")
    return None


def _pipeline_put_blocked(c, base: dict, detail: str, remedy: str) -> dict:
    """Epoch c.epoch's entry as a RECORDED val failure 'not launched' (a stale leg or result is in the
    way). Evolution continues; the held-out rule skips the epoch; the end lists it."""
    entry = {**base, "pid": 0, "identity": None, "state": "blocked",
             "failure": {"step": VAL_TRACKED_STEP, "detail": str(detail)[:4000], "remedy": remedy},
             "failed_at": _stamp(), "failed_t": time.time()}

    def put(d):
        d["legs"][str(int(c.epoch))] = entry
    _pipeline_update(c.run_dir, put)
    return entry


def _pipeline_track_val(ctx: Ctx) -> dict:
    """Launch this epoch's val leg unless a CURRENT record tracks it (_val_status), and record it DURABLY
    (pid identity, dispatch time, the promoted set it loads) before anything waits. A record that is not
    current (epoch redone, result gone or changed) is superseded with the reason and a new val leg is
    dispatched, unless a superseded leg or its result is in the way (_stale_val_guard): then the entry is a
    recorded failure naming it. A recorded val failure of THIS promoted set (died, timed out) is current:
    relaunching it is the operator's call (--val-check --step), never an automatic retry."""
    out = ctx.paths.leg("val")
    rec = _val_record(ctx.run_dir, ctx.epoch)
    ok, why, _kind = _val_status(ctx, rec)
    if ok:
        return rec
    have = _promoted_sha(ctx)
    if have is None:
        raise Halt(f"no promoted skill set at {ctx.paths.promoted.name}: the val leg of epoch {ctx.epoch} has "
                   f"nothing to load")
    if rec is not None:
        _pipeline_supersede(ctx, rec, why)
        print(f"[pipeline] VAL e{ctx.epoch} ({rec.get('tag')}): its {rec.get('state')} record is SUPERSEDED "
              f"({why}); it is not counted, and the val leg is launched for the promoted set on disk")
    base = {"epoch": int(ctx.epoch), "key": "val", "tag": out.stem, "result": str(out),
            "promoted_sha256": have, "dispatched_at": _stamp(),
            "dispatched_t": time.time()}
    guard = _stale_val_guard(ctx)
    if guard is not None:
        entry = _pipeline_put_blocked(ctx, base, *guard)
        _pipeline_announce(ctx, getattr(ctx, "_pipeline_say", None))
        return entry
    pid = int(_dispatch_leg(ctx, "val") or 0)          # 0 = its result is already on disk
    entry = {**base, "pid": pid, "identity": _proc_identity(pid, out.stem) if pid else None,
             "state": "running"}

    def put(d):
        d["legs"][str(int(ctx.epoch))] = entry
    _pipeline_update(ctx.run_dir, put)
    print(f"[pipeline] val leg {out.stem} " + (f"launched (pid {pid})" if pid else "already has its result")
          + f"; tracked in {PIPELINE_VAL_FILE}, checked when it ends, never awaited before the end",
          file=sys.stderr)
    return entry


def _run_track_val(ctx: Ctx) -> None:
    _pipeline_track_val(ctx)


def _pipeline_ticker(ctx: Ctx):
    """_await_leg's on_tick while a TRAIN leg is awaited: poll the val legs every PIPELINE_POLL_S."""
    last = [time.time()]

    def tick():
        if time.time() - last[0] >= PIPELINE_POLL_S:
            last[0] = time.time()
            _pipeline_poll(ctx)
    return tick


def _run_train_leg_pipelined(ctx: Ctx) -> None:
    """pipeline_val's train-leg runner: launch train (FIRST, so a two-leg ceiling can never leave train
    waiting on its sibling), launch + track val, then await TRAIN ONLY, polling val legs meanwhile."""
    out = ctx.paths.leg("train")
    if out.is_file():
        return                                       # idempotent: already measured
    mine = _dispatch_leg(ctx, "train")
    try:
        _pipeline_track_val(ctx)
    except Exception as e:                           # noqa: BLE001 (Halt included: train must not wait on val)
        print(f"[leg] val did not launch ({type(e).__name__}: {e}); train continues alone. The step "
              f"'{VAL_TRACKED_STEP}' launches it after train, and HALTs there if it still cannot.",
              file=sys.stderr)
    if not mine:
        return
    _await_leg(out, mine, ctx.leg_timeout_s, "train", ident=_dispatch_record(ctx, "train").get("identity"),
               tag=out.stem, on_tick=_pipeline_ticker(ctx))
    try:
        _dispatch_path(ctx, "train").unlink()
    except OSError:
        pass


def _val_alive(c, rec) -> bool:
    """The recorded val leg process is alive AND the same process (_leg_alive: the recorded start time,
    and the tag when it was in the command line at dispatch). A leg whose start time was NEVER recorded
    counts as alive only while its result file is absent AND the pid's command line names that leg's tag."""
    pid = rec.get("pid")
    if not pid:
        return False
    ident = rec.get("identity") or {}
    if not ident.get("lstart") and c.paths.leg("val").is_file():
        return False
    return _leg_alive(pid, ident, rec.get("tag") or c.paths.leg("val").stem)


def _pipeline_check_val(c, *, run_remedies: bool = False) -> dict:
    """Run the val bundle on epoch c.epoch's val leg, in order, stopping at the first failure (as
    advance() does). With run_remedies, a failing step's runner (the re-grade) runs ONCE and the step is
    re-verified. NEVER raises for a failed check: it returns {"state": "passed"|"failed", "rows", ...}."""
    rows = []
    for s in _val_bundle(c):
        def verify():
            try:
                ok, detail, q = _safe_verify(s, c)
            except Blocked as e:
                return False, f"needs an answer ({e.question})"
            return (False, f"{detail} (asks: {q})") if q else (ok, detail)
        ok, detail = verify()
        ran = None
        if not ok and run_remedies and s.run is not None:
            try:
                s.run(c)
                ran = "ran"
            except Blocked as e:
                ran = f"needs an answer: {e.question}"
            except Exception as e:                  # noqa: BLE001 (Halt included: recorded, not raised)
                ran = f"{type(e).__name__}: {e}"
            ok, detail = verify()
        row = {"step": s.name, "ok": bool(ok), "detail": str(detail)[:2000]}
        if ran is not None:
            row["runner"] = str(ran)[:2000]
        rows.append(row)
        if not ok:
            # The re-grade runner acts only on "RE-GRADE NEEDED"; every other state (failed again, a flip, a
            # lost or invalid re-grade) is a person's decision, so no background run is started for it.
            runnable = s.run is not None and (s.name != LEG_D4_STEP.format(key="val")
                                              or "RE-GRADE NEEDED" in str(detail))
            return {"state": "failed", "rows": rows, "failed_step": s.name, "detail": str(detail),
                    "remedy_runnable": runnable}
    return {"state": "passed", "rows": rows}


def _pipeline_remedy_text(c, res: dict) -> str:
    step, detail = res.get("failed_step", ""), res.get("detail", "")
    cmd = _val_check_cmd(c)
    if step == LEG_D4_STEP.format(key="val"):
        if "RE-GRADE NEEDED" in detail:
            return (f"the re-grade (once, grading sandbox): the loop runs it in the background; by "
                    f"hand: {cmd}")
        return (f"a person reads the detail above and the re-grade record under "
                f"{operator_dir(c.run_dir)}/{REGRADE_DIRNAME}/, decides (a written correction or a re-run "
                f"of the val leg), then re-checks with: {cmd}")
    if res.get("remedy_runnable"):
        return f"this check's runner: {cmd}"
    return f"a person fixes what the check names, then re-checks with: {cmd}"


def _pipeline_line(c, rec: dict) -> str:
    st, tag = rec.get("state"), rec.get("tag")
    where = f"{operator_dir(c.run_dir).name}/{PIPELINE_VAL_FILE}"
    if st == "passed":
        return (f"[pipeline] VAL e{c.epoch} ({tag}): all {len(rec.get('checks') or [])} val checks PASSED "
                f"(recorded in {where})")
    if st in VAL_FAILURE_STATES:
        f = rec.get("failure") or {}
        return (f"[pipeline] VAL e{c.epoch} ({tag}): {VAL_FAILURE_STATES[st]} '{f.get('step')}' — "
                f"{str(f.get('detail'))[:700]}\n"
                f"    REMEDY: {f.get('remedy')}\n"
                f"    Recorded in {where}. Evolution CONTINUES (a val leg never halts the loop); the "
                f"held-out program skips epoch {c.epoch} until its val checks pass.")
    if st == "remedy_running":
        r = rec.get("remedy") or {}
        return (f"[pipeline] VAL e{c.epoch} ({tag}): '{(rec.get('failure') or {}).get('step')}' failed and "
                f"its runner (the re-grade) is running ONCE in the background (pid {r.get('pid')}, log "
                f"{r.get('log')}); evolution continues")
    return f"[pipeline] VAL e{c.epoch} ({tag}): {st}"


def _pipeline_announce(c, say=None, mark: bool = True) -> None:
    """Print (and put in the transcript) a val leg's state once per (state, leg bytes). A --val-check
    process prints without marking (mark=False), so the DRIVE still prints the line in its own output."""
    rec = _val_record(c.run_dir, c.epoch) or {}
    st = rec.get("state")
    if st not in ("passed", "remedy_running") and st not in VAL_FAILURE_STATES:
        return
    key = f"{st}:{str(rec.get('leg_sha256') or '')[:12]}"
    if mark and rec.get("announced") == key:
        return
    line = _pipeline_line(c, rec)
    print(line)
    if not mark:
        return
    if say is not None:
        try:
            say(line, f"pipeline-val:{c.epoch}:{key}", c=c)
        except Exception as e:                      # noqa: BLE001 (the transcript is best-effort here)
            print(f"[pipeline] could not record the line in the transcript ({type(e).__name__}: {e})",
                  file=sys.stderr)

    def put(d):
        if str(c.epoch) in d["legs"]:
            d["legs"][str(c.epoch)]["announced"] = key
    _pipeline_update(c.run_dir, put)


def _pipeline_spawn_remedy(c) -> dict:
    """The failing val check's runner (the re-grade), ONCE, in a detached `--val-check --step` process,
    so a grading sandbox never holds up the loop. Its output goes to a log in the operator dir."""
    od = operator_dir(c.run_dir)
    od.mkdir(parents=True, exist_ok=True)
    log = od / f"pipeline_val_e{c.epoch}.remedy.{time.strftime('%Y%m%dT%H%M%S')}.log"
    argv = [sys.executable, "-m", "src.orchestrator.loop", "--run-dir", str(Path(c.run_dir).resolve()),
            "--epoch", str(c.epoch), "--val-check", "--step"]
    with open(log, "wb") as fh:
        p = subprocess.Popen(argv, cwd=str(REPO), stdout=fh, stderr=subprocess.STDOUT, stdin=DEVNULL,
                             start_new_session=True)
    _REMEDY_CHILDREN.append(p)
    return {"pid": p.pid, "identity": _proc_identity(p.pid), "log": str(log), "argv": argv,
            "started_at": _stamp()}


def _record_val_result(c, res: dict, *, extra: dict | None = None, remedy_text: str | None = None) -> None:
    out = c.paths.leg("val")
    sha = _file_sha(out) if out.is_file() else None
    have = _promoted_sha(c)

    def put(d):
        rec = d["legs"].setdefault(str(c.epoch), {"epoch": int(c.epoch), "key": "val", "tag": out.stem,
                                                    "result": str(out), "pid": 0, "identity": None,
                                                    "promoted_sha256": have})
        rec.update(state=res["state"], leg_sha256=sha, checked_at=_stamp(), checked_t=time.time(),
                   checks=res["rows"])
        rec.pop("detail", None)                      # a 'died' record's note, when a late result is checked
        if res["state"] == "failed":
            rec["failure"] = {"step": res.get("failed_step"), "detail": str(res.get("detail"))[:4000],
                              "remedy": remedy_text or _pipeline_remedy_text(c, res),
                              "remedy_runnable": bool(res.get("remedy_runnable"))}
        else:
            rec.pop("failure", None)
        rec.update(extra or {})
    _pipeline_update(c.run_dir, put)


def _remedy_tries(rem: dict, sha) -> int:
    """How many times the background re-grade of THESE leg bytes failed to start."""
    return int(rem.get("spawn_failures") or 0) if rem.get("spawn_failures_for") == sha else 0


def _pipeline_check_and_record(c, *, say=None, remedies: bool = True) -> str:
    """An ENDED val leg: verify its bundle now (verify-only, fast); record passed / failed. A failure
    whose step has a runner gets it ONCE per leg bytes, in the background, and is marked attempted only
    once that process has STARTED; a start that fails is recorded (the failure stays 'failed', with the
    error) and retried at the next boundary, at most PIPELINE_REMEDY_SPAWN_TRIES times. The spawn happens
    under this epoch's check lock, which the child waits on, so the record says remedy_running before the
    child can read it. Skipped (checked at the next boundary) while another process holds the lock."""
    out = c.paths.leg("val")
    lockp = operator_dir(c.run_dir) / f".pipeline_val_e{int(c.epoch)}.check.lock"
    with _flock(lockp, blocking=False) as got:
        if not got:
            return "locked"
        res = _pipeline_check_val(c, run_remedies=False)
        sha = _file_sha(out) if out.is_file() else None
        prev = (_val_record(c.run_dir, c.epoch) or {}).get("remedy") or {}
        extra, remedy_text = {}, None
        tries = _remedy_tries(prev, sha)
        if res["state"] == "failed" and remedies and res.get("remedy_runnable") \
                and prev.get("attempted_for") != sha and tries < PIPELINE_REMEDY_SPAWN_TRIES:
            try:
                child = _pipeline_spawn_remedy(c)
                err = "" if (child or {}).get("pid") else f"the start returned no pid ({child!r})"
            except Exception as e:                  # noqa: BLE001 (recorded; the loop continues)
                child, err = None, f"{type(e).__name__}: {e}"
            if not err:
                extra = {"state": "remedy_running",
                         "remedy": {**child, "attempted_for": sha, "started_t": time.time()}}
            else:
                tries += 1
                extra = {"remedy": {**prev, "error": err[:2000], "spawn_failures": tries,
                                    "spawn_failures_for": sha, "last_try_at": _stamp()}}
                more = tries < PIPELINE_REMEDY_SPAWN_TRIES
                remedy_text = (f"the re-grade (once, grading sandbox): its background start FAILED "
                               f"({err[:300]}; try {tries} of {PIPELINE_REMEDY_SPAWN_TRIES})"
                               + ("; the loop retries at the next step boundary" if more else
                                  "; the loop does not try again")
                               + f"; by hand: {_val_check_cmd(c)}")
                print(f"[pipeline] VAL e{c.epoch} ({out.stem}): the background re-grade did NOT start ({err[:300]}); "
                      f"try {tries} of {PIPELINE_REMEDY_SPAWN_TRIES}"
                      + (", retried at the next step boundary" if more else ", not tried again")
                      + f". Evolution continues. By hand: {_val_check_cmd(c)}")
        _record_val_result(c, res, extra=extra, remedy_text=remedy_text)
    try:
        _dispatch_path(c, "val").unlink()
    except OSError:
        pass
    _pipeline_announce(c, say)
    return (_val_record(c.run_dir, c.epoch) or {}).get("state", "")


def _record_val_failure(c, state: str, *, step: str, detail: str, remedy: str,
                        extra: dict | None = None) -> None:
    """A val failure that has no check result (died / timed out): recorded, announced by the caller, and
    never a HALT. The held-out rule skips the epoch; the end lists it with `remedy`."""
    def put(d):
        rec = d["legs"].get(str(int(c.epoch)))
        if rec is None:
            return
        rec.update(state=state, failure={"step": step, "detail": str(detail)[:4000], "remedy": remedy},
                   failed_at=_stamp(), failed_t=time.time())
        rec.update(extra or {})
    _pipeline_update(c.run_dir, put)


def _pipeline_poll(ctx, *, say=None, remedies: bool = True) -> dict:
    """AT A STEP BOUNDARY: every tracked val leg is looked at, never awaited (_pipeline_poll_one): an ended
    one is checked now; a running one is left alone until it outlives its own leg_timeout_s; one that died
    without a result, or timed out, is RECORDED as a val failure with its relaunch command; a record that is
    no longer current is superseded and never announced as passed. NOTHING HERE HALTS EXCEPT AN
    UNREADABLE PIPELINE_VAL.json (the only record of the outstanding legs); any other error is recorded in
    it and printed as one line, and the leg is polled again at the next boundary. Returns the fresh record.
    Read-only contexts (--plan) never poll."""
    if getattr(ctx, "readonly", False):
        return _pipeline_read(ctx.run_dir)
    try:
        for p in list(_REMEDY_CHILDREN):
            if p.poll() is not None:
                _REMEDY_CHILDREN.remove(p)
    except Exception:                               # noqa: BLE001 (reaping is housekeeping)
        pass
    say = say if say is not None else getattr(ctx, "_pipeline_say", None)
    reg = _pipeline_read(ctx.run_dir)               # unreadable = HALT, the one HALT a poll raises
    for ek in sorted(reg["legs"], key=lambda k: int(k) if str(k).isdigit() else 10 ** 9):
        try:
            _pipeline_poll_one(_ctx_for_epoch(ctx, int(ek)), reg["legs"][ek], say=say, remedies=remedies)
        except Halt:
            raise
        except Exception as e:                      # noqa: BLE001 (recorded, never a traceback)
            _pipeline_poll_error(ctx, ek, e)
    return _pipeline_read(ctx.run_dir)


def _pipeline_poll_error(ctx, ek, e) -> None:
    msg = f"{type(e).__name__}: {e}"
    print(f"[pipeline] VAL e{ek}: the poll could not process this val leg ({msg[:400]}); recorded in "
          f"{PIPELINE_VAL_FILE} (poll_errors). Evolution continues; it is polled again at the next step boundary.")

    def put(d):
        errs = d.setdefault("poll_errors", [])
        errs.append({"epoch": str(ek), "at": _stamp(), "error": msg[:2000]})
        del errs[:-50]
    try:
        _pipeline_update(ctx.run_dir, put)
    except Halt:
        raise
    except Exception as e2:                         # noqa: BLE001
        print(f"[pipeline] VAL e{ek}: the error could not be recorded either ({type(e2).__name__}: {e2})")


def _pipeline_poll_one(c, rec: dict, *, say, remedies: bool) -> None:
    """One tracked val leg of epoch c.epoch at a step boundary. Never raises for a val failure."""
    out = c.paths.leg("val")
    cmd = _val_check_cmd(c)
    ok, why, kind = _val_status(c, rec)
    if not ok:
        if kind == "result_changed":
            # new bytes on disk for the same promoted set (a re-run): superseded, then checked afresh
            fresh = {"epoch": int(c.epoch), "key": "val", "tag": out.stem, "result": str(out), "pid": 0,
                     "identity": None, "promoted_sha256": _promoted_sha(c),
                     "dispatched_at": _stamp(), "dispatched_t": time.time(), "state": "running",
                     "adopted": "a changed result on disk, for the same promoted set"}
            _pipeline_supersede(c, rec, why, readopt=fresh)
            print(f"[pipeline] VAL e{c.epoch} ({out.stem}): {why}; the old record is superseded and the new "
                  f"result is checked now")
            _pipeline_check_and_record(c, say=say, remedies=remedies)
            return
        alive = bool(rec.get("pid")) and _val_alive(c, rec)
        _pipeline_supersede(c, rec, why)
        print(f"[pipeline] VAL e{c.epoch} ({rec.get('tag')}): its {rec.get('state')} record is SUPERSEDED and "
              f"NOT counted: {why}."
              + (f" Its process (pid {rec.get('pid')}) is STILL RUNNING the old skill set: stop it." if alive else "")
              + f" Re-running epoch {c.epoch} launches its val leg again; by hand: {cmd}")
        return
    st = rec.get("state")
    now = time.time()
    if st == "running":
        if _val_alive(c, rec):
            t0 = rec.get("dispatched_t")
            if t0 is None:                          # never recorded: its own clock starts now
                def put(d, e=str(int(c.epoch))):
                    if e in d["legs"]:
                        d["legs"][e]["dispatched_t"] = now
                _pipeline_update(c.run_dir, put)
                return
            limit = int(getattr(c, "leg_timeout_s", 0) or 0)
            if limit and now - float(t0) >= limit:
                _record_val_failure(
                    c, "timed_out", step=VAL_TIMEOUT_STEP,
                    detail=(f"val leg {out.stem} (pid {rec.get('pid')}) is still running {int(now - float(t0))}s "
                            f"after its dispatch at {rec.get('dispatched_at')}, past leg_timeout_s={limit}. It is "
                            f"NOT relaunched and NOT awaited; a result it writes later is checked at the next "
                            f"step boundary"),
                    remedy=(f"to wait: nothing (a later result is checked at the next step boundary, or with: "
                            f"{cmd}); to give up on it: stop pid {rec.get('pid')}, diagnose (read the PARTIAL "
                            f"json's failed_episodes), then relaunch with: {cmd}"))
                _pipeline_announce(c, say)
            return
        if out.is_file():
            _pipeline_check_and_record(c, say=say, remedies=remedies)
            return
        failed = out.with_name(out.stem + ".FAILED.json")
        _record_val_failure(
            c, "died", step=VAL_DIED_STEP,
            detail=(f"val leg {rec.get('tag')} (epoch {c.epoch}, pid {rec.get('pid')}) EXITED WITHOUT WRITING "
                    f"{out.name}" + (f"; {failed.name} is there" if failed.is_file() else "")
                    + ". This is a failed leg, not a slow one; relaunching without a diagnosis repeats it"),
            remedy=("diagnose" + (f" (read {failed.name})" if failed.is_file() else "")
                    + f", then relaunch it with: {cmd}"))
        _pipeline_announce(c, say)
        return
    if st == "remedy_running":
        r = rec.get("remedy") or {}
        if r.get("pid") and _proc_is(r["pid"], r.get("identity"), ("--val-check",)):
            limit = int(getattr(c, "leg_timeout_s", 0) or 0)
            t0 = r.get("started_t")
            if limit and t0 is not None and now - float(t0) >= limit:
                # the re-grade outlived its own timeout: the val failure it was fixing stands, never awaited
                def put(d, e=str(int(c.epoch))):
                    x = d["legs"].get(e)
                    if x is None:
                        return
                    x["state"] = "failed"
                    x.setdefault("failure", {})["remedy"] = (
                        f"the background re-grade (pid {r['pid']}, log {r.get('log')}) is still running "
                        f"{int(now - float(t0))}s after its start, past leg_timeout_s={limit}; if it finishes it "
                        f"records its result itself. To give up on it: stop pid {r['pid']}; by hand: {cmd}")
                _pipeline_update(c.run_dir, put)
                _pipeline_announce(c, say)
            return
        # the background run ended without writing a final state: re-check, never a second runner
        _pipeline_check_and_record(c, say=say, remedies=False)
        return
    if st in ("died", "timed_out"):
        if out.is_file() and not _val_alive(c, rec):
            _pipeline_check_and_record(c, say=say, remedies=remedies)      # a late (or hand-run) result
            return
    if st == "failed":
        f, r = rec.get("failure") or {}, rec.get("remedy") or {}
        sha = rec.get("leg_sha256")
        if remedies and f.get("remedy_runnable") and r.get("attempted_for") != sha \
                and _remedy_tries(r, sha) < PIPELINE_REMEDY_SPAWN_TRIES:
            _pipeline_check_and_record(c, say=say, remedies=True)     # the background start is retried
            return
    _pipeline_announce(c, say)                      # passed / failed / blocked / a written-elsewhere state


def _pipeline_record_train(ctx) -> None:
    """Epoch N's blocking program verified end to end: its TRAIN checks passed. The held-out rule reads it."""
    if getattr(ctx, "readonly", False):
        return
    tl = ctx.paths.leg("train")
    if _skip_train_here(ctx):
        try:
            names = [s.name for s in _serial_steps_for(ctx) if s.name in PIPELINE_TRAIN_STEPS]
        except Exception:                            # noqa: BLE001 (the names are a record, not a check)
            names = list(PIPELINE_TRAIN_STEPS)
        entry = {"state": "passed", "at": _stamp(), "t": time.time(), "leg_sha256": None,
                 "skipped_by_design": True, "train_leg": SKIPPED_BY_DESIGN,
                 "train_checks": {n: SKIPPED_BY_DESIGN for n in names},
                 "what": ("every blocking step of the epoch verified (promoted set, skills load, the evolver's "
                          "checks); its train leg and train-leg checks were skipped by design: RUNTIME "
                          "skip_last_train, the run's last epoch")}
    else:
        entry = {"state": "passed", "at": _stamp(), "t": time.time(),
                 "leg_sha256": _file_sha(tl) if tl.is_file() else None,
                 "what": "every blocking step of the epoch verified (the train leg and its checks)"}

    def put(d):
        d["train"][str(int(ctx.epoch))] = entry
    _pipeline_update(ctx.run_dir, put)


def _pipeline_waiting(reg: dict) -> list:
    return [(int(k), r) for k, r in sorted(reg["legs"].items(), key=lambda kv: int(kv[0]))
            if r.get("state") in ("running", "remedy_running")]


def _pipeline_final(ctx, *, say=None) -> list:
    """THE END (after the last epoch): await every tracked val leg and background re-grade, check each, and
    return the failures [(epoch, step, detail)]. The only place val waits, and it is bounded by each leg's
    OWN leg_timeout_s (the poll records an overdue leg as a val failure and stops waiting for it), so the
    end never HALTs on a val leg: it lists it with its remedy. Every epoch 1..this one is listed; one with
    no current val record (never tracked, or superseded) is listed as NOT TRACKED, never as PASSED."""
    reg = _pipeline_poll(ctx, say=say)
    waiting = _pipeline_waiting(reg)
    if waiting:
        print(f"\nPIPELINE END — the last epoch ({ctx.epoch}) is complete; awaiting every outstanding val leg "
              f"(the end is the only place val waits, each leg bounded by its own leg_timeout_s): "
              + ", ".join(f"e{e} {r.get('state')}" for e, r in waiting))
    t0 = time.time()
    bound = int(ctx.leg_timeout_s) + 2 * PIPELINE_POLL_S      # defensive: the per-leg timeouts end it first
    while waiting:
        if time.time() - t0 >= bound:
            for e, r in waiting:
                c = _ctx_for_epoch(ctx, e)
                _record_val_failure(
                    c, "timed_out", step=VAL_TIMEOUT_STEP,
                    detail=(f"still {r.get('state')} (pid {r.get('pid')}) {int(time.time() - t0)}s into the end's "
                            f"wait; not awaited further"),
                    remedy=(f"a later result is checked by: {_val_check_cmd(c)}; to give up on it: stop pid "
                            f"{r.get('pid')}, then relaunch with: {_val_check_cmd(c)}"))
            reg = _pipeline_read(ctx.run_dir)
            break
        time.sleep(max(0.0, min(PIPELINE_POLL_S, bound - (time.time() - t0))))
        reg = _pipeline_poll(ctx, say=say)
        waiting = _pipeline_waiting(reg)
    return _pipeline_list(ctx, reg)


def _pipeline_list(ctx, reg: dict) -> list:
    """Print every epoch's val leg as it stands and return the ones that are not a current PASS."""
    fails = []
    print("\nVAL LEGS (pipeline_val):")
    epochs = sorted(set(range(1, int(ctx.epoch) + 1))
                    | {int(k) for k in reg["legs"] if str(k).isdigit()})
    for e in epochs:
        c = _ctx_for_epoch(ctx, e)
        tag = c.paths.leg("val").stem
        r = reg["legs"].get(str(e))
        try:
            if r is None:
                sup = [s for s in reg.get("superseded") or [] if str(s.get("epoch")) == str(e)]
                why = f"superseded: {sup[-1].get('why')}" if sup else f"no val record in {PIPELINE_VAL_FILE}"
                print(f"  e{e} {tag}: NOT TRACKED — {why}; launch or check it with: {_val_check_cmd(c)}")
                fails.append((e, "no current val record", why[:300]))
                continue
            ok, why, _kind = _val_status(c, r)
            st = r.get("state")
            if not ok:
                print(f"  e{e} {tag}: NOT COUNTED ({st} record no longer current: {why}); re-check with: "
                      f"{_val_check_cmd(c)}")
                fails.append((e, "val record no longer current", why[:300]))
                continue
            f = r.get("failure") or {}
            label = VAL_FAILURE_STATES.get(st, str(st).upper())
            print(f"  e{e} {r.get('tag')}: {label}"
                  + (f" at '{f.get('step')}' — {str(f.get('detail'))[:300]}\n      REMEDY: {f.get('remedy')}"
                     if st in VAL_FAILURE_STATES else ""))
            if st != "passed":
                fails.append((e, f.get("step") or st, str(f.get("detail") or st)[:300]))
        except Exception as ex:                     # noqa: BLE001 (the end reports; it never raises)
            print(f"  e{e} {tag}: UNREADABLE ({type(ex).__name__}: {ex})")
            fails.append((e, "val record unreadable", f"{type(ex).__name__}: {ex}"[:300]))
    return fails


def _pipeline_heldout_gate(ctx):
    """val_gate for last_successful_epoch in pipeline mode: epoch e qualifies only when its train checks
    are recorded as passed (train leg unchanged since), and its val leg is on disk, recorded as PASSED for
    exactly these bytes and for the promoted set on disk, and passes every val check again NOW (re-derived,
    never read as a status). skip_last_train: the run's LAST epoch qualifies with its train leg skipped by
    design (its blocking program, i.e. promoted set, skills load and the evolver's checks, recorded as
    passed) and its val leg complete and passing; a train leg recorded as skipped at any other epoch does not."""
    skip_e = _skip_train_epoch(ctx)

    def gate(e):
        c = _ctx_for_epoch(ctx, e)
        reg = _pipeline_read(ctx.run_dir)
        tr = reg["train"].get(str(int(e))) or {}
        tl, vl = c.paths.leg("train"), c.paths.leg("val")
        if tr.get("state") != "passed":
            return False, (f"epoch {e}'s train checks are not recorded as passed in {PIPELINE_VAL_FILE} "
                           f"(the epoch's blocking program did not complete)")
        skipped = bool(tr.get("skipped_by_design"))
        if skipped and e != skip_e:
            return False, (f"epoch {e}'s train leg is recorded as skipped by design, but only the run's last "
                           f"epoch may skip it, under RUNTIME skip_last_train"
                           + (f" (that is epoch {skip_e})" if skip_e else " (which is off)"))
        if not skipped and tl.is_file() and tr.get("leg_sha256") and _file_sha(tl) != tr["leg_sha256"]:
            return False, f"{tl.name} changed since its checks passed"
        rec = reg["legs"].get(str(int(e)))
        if rec is None:
            return False, (f"val leg {vl.name} is UNCHECKED (no entry in {PIPELINE_VAL_FILE}); check it "
                           f"with: {_val_check_cmd(c)}")
        st = rec.get("state")
        if st != "passed":
            f = rec.get("failure") or {}
            return False, (f"val leg {vl.name} is {str(st).upper()}"
                           + (f" at '{f.get('step')}': {str(f.get('detail'))[:200]}" if f else ""))
        if not vl.is_file():
            return False, f"val leg {vl.name} is MISSING"
        if _file_sha(vl) != rec.get("leg_sha256"):
            return False, f"{vl.name} changed since its val checks passed; re-check with: {_val_check_cmd(c)}"
        if rec.get("promoted_sha256") != _promoted_sha(c):
            return False, (f"val leg {vl.name} was dispatched for promoted set "
                           f"{str(rec.get('promoted_sha256'))[:12]}, not the one on disk; re-check with: "
                           f"{_val_check_cmd(c)}")
        res = _pipeline_check_val(c, run_remedies=False)
        if res["state"] != "passed":
            return False, (f"val leg {vl.name} FAILS its re-check now at '{res.get('failed_step')}': "
                           f"{str(res.get('detail'))[:200]}")
        if skipped:
            return True, (f"blocking program passed with its train leg {SKIPPED_BY_DESIGN} (RUNTIME "
                          f"skip_last_train, the last epoch); val checks passed (val re-verified now)")
        return True, "train and val checks passed (val re-verified now)"
    return gate


def _pipeline_plan_text(ctx) -> str:
    """--plan's PIPELINE section: the order, the val checks and when they run, every tracked val leg (read
    only: its pid is looked at, nothing is written), and what the held-out rule would select now."""
    n = getattr(ctx.spec, "n_epochs", "?")
    L_ = ["", "PIPELINE (RUNTIME pipeline_val=true; "
          + ("daytona_leg.pool=true: the pool admits rollouts, more than two leg processes may be alive)"
             if getattr(ctx, "leg_pool", False) else
             "daytona_leg.pool is NOT true: the loop keeps its two-leg seat ceiling)"),
          "  order, epoch N: promote v_N -> launch train_N AND val_N -> await TRAIN_N only -> train_N checks ->",
          "    epoch N+1: digests from train_N -> render ITER{N+1} -> prompt checks -> evolver N+1 -> promote",
          "    v_{N+1} -> launch train_{N+1} + val_{N+1} at once (older val legs may still run) -> ...",
          "  val legs: polled at every step boundary (and every "
          f"{int(PIPELINE_POLL_S)}s while a train leg is awaited); each is checked the moment it has ended;",
          "    a failed val check is RECORDED (operator dir + one drive line with the remedy) and never halts",
          "    evolution; the re-grade runs once in a background --val-check; a val leg that dies without a",
          f"    result, or runs past its own leg_timeout_s={getattr(ctx, 'leg_timeout_s', '?')}s, is RECORDED as a val",
          "    failure with its relaunch command, never a HALT; a record whose promoted set changed is superseded.",
          f"  the end: after the last epoch (--until-epoch; spec n_epochs={n}) the drive awaits every tracked",
          "    val leg (each bounded by its own timeout), checks it, and reports done listing any val failure.",
          "  val checks, in order, when the leg ends: "
          + "; ".join(s.name for s in _val_bundle(ctx)),
          f"  record: {_pipeline_path(ctx.run_dir)}"]
    skip_e = _skip_train_epoch(ctx)
    if skip_e:
        L_.append(f"  skip_last_train (RUNTIME): epoch {skip_e}, the last, launches ONLY its val leg; its train-leg "
                  f"checks are recorded '{SKIPPED_BY_DESIGN}'; the held-out rule accepts it on its val leg")
    try:
        reg = _pipeline_read(ctx.run_dir)
    except Halt as e:
        return "\n".join(L_ + [f"  RECORD UNREADABLE: {e}"])
    if not reg["legs"]:
        L_.append("  tracked val legs: none yet")
    for ek in sorted(reg["legs"], key=lambda k: int(k) if str(k).isdigit() else 10 ** 9):
        r = reg["legs"][ek]
        c = _ctx_for_epoch(ctx, int(ek))
        st = r.get("state")
        ok, why, _kind = _val_status(c, r)
        if not ok:
            L_.append(f"  e{ek} {r.get('tag')}: {st} record NO LONGER CURRENT ({why}): superseded at the next "
                      f"step boundary, never counted")
        elif st == "running":
            alive = _val_alive(c, r)
            done = c.paths.leg("val").is_file()
            now = ("ALIVE — its checks run at the first step boundary after it ends" if alive else
                   "ENDED — checked at the next step boundary" if done else
                   "DEAD WITHOUT A RESULT — the next step boundary records it as a val failure")
            L_.append(f"  e{ek} {r.get('tag')}: running, pid {r.get('pid')}: {now}")
        else:
            f = r.get("failure") or {}
            L_.append(f"  e{ek} {r.get('tag')}: {st}"
                      + (f" at '{f.get('step')}' — remedy: {f.get('remedy')}" if st in VAL_FAILURE_STATES else ""))
    if reg["train"]:
        L_.append("  train checks recorded passed: "
                  + ", ".join(f"e{k}" + (" (train leg skipped by design)" if reg["train"][k].get("skipped_by_design")
                                         else "")
                              for k in sorted(reg["train"], key=int)
                              if reg["train"][k].get("state") == "passed"))
    try:
        e, why = last_successful_epoch(ctx.run_dir, int(getattr(ctx.spec, "n_epochs", 0) or 0),
                                       ctx.spec.split_path, getattr(ctx.spec, "n_samples", None),
                                       val_gate=_pipeline_heldout_gate(ctx), skip_train_epoch=skip_e)
        L_.append("  held-out rule now: " + (f"epoch {e}" if e else "no epoch qualifies yet")
                  + ("" if not why else " (" + "; ".join(f"e{k}: {w}" for k, w in why)[:900] + ")"))
    except Exception as ex:                          # noqa: BLE001 (a plan reports; it never raises)
        L_.append(f"  held-out rule now: not computable ({type(ex).__name__}: {ex})")
    return "\n".join(L_)


def _val_check_program(ctx, *, step: bool, note: str = "") -> int:
    """`--val-check` (pipeline_val): the val leg of --epoch. --plan: its checks, read-only. --step: its
    checks WITH the failing step's runner (the re-grade, once), recorded in PIPELINE_VAL.json; a val leg
    with no result and no live process is (re)launched: the operator's --step is that decision (this is
    the relaunch command a died / timed-out record names). A record that is not current is superseded
    first. It never relaunches over a live process of the leg, and never checks a result that is, or may
    be, a superseded leg's (_stale_val_guard). Waits for the epoch's check lock (a background re-grade of
    the same leg finishes first). Exit 0 passed or nothing to check yet, 1 failed / refused / HALT."""
    out = ctx.paths.leg("val")
    if not step:
        ctx.readonly = True
        rec = _val_record(ctx.run_dir, ctx.epoch)
        print(f"VAL CHECK — {ctx.run_dir.name} epoch {ctx.epoch} ({out.name}); record: "
              + (json.dumps({k: rec.get(k) for k in ('state', 'pid', 'failure')}, default=str)[:600]
                 if rec else "none")
              + (f"; {'current' if _val_status(ctx, rec)[0] else 'NOT CURRENT'}: {_val_status(ctx, rec)[1]}"
                 if rec else ""))
        if not out.is_file():
            print("  no val result yet: nothing to check")
            return 0
        res = _pipeline_check_val(ctx, run_remedies=False)
        for r in res["rows"]:
            print(f"  {'ok  ' if r['ok'] else 'FAIL'}  {r['step']}\n        {r['detail'][:600]}")
        print(f"\n  {res['state'].upper()} (read-only; --step also runs a failing check's runner and records)")
        return 0
    if note:
        _conv_once(ctx, f"invocation:{ctx.epoch}:val-check:{note}", speaker="operator", channel="invocation",
                   content=note, meta={"mode": "val-check"})
    lockp = operator_dir(ctx.run_dir) / f".pipeline_val_e{int(ctx.epoch)}.check.lock"
    with _flock(lockp, blocking=True):
        rec = _val_record(ctx.run_dir, ctx.epoch)
        if _promoted_sha(ctx) is None:
            print(f"no promoted skill set at {ctx.paths.promoted.name}: epoch {ctx.epoch}'s val leg has nothing "
                  f"to load, so nothing is checked or launched")
            return 1
        ok, why, _kind = _val_status(ctx, rec)
        if not out.is_file():
            if rec is not None and rec.get("pid") and _val_alive(ctx, rec):
                if rec.get("state") == "running":
                    print(f"val leg {out.stem} is still running (pid {rec.get('pid')}): nothing to check yet")
                    return 0
                print(f"val leg {out.stem} (recorded {rec.get('state')}) is STILL RUNNING as pid {rec.get('pid')}; "
                      f"it is not relaunched over a live process. Stop it first, then re-run this command.")
                return 1
            if rec is not None:
                _pipeline_supersede(ctx, rec, why if not ok else
                                    f"relaunched by --val-check --step (operator; it was {rec.get('state')})")
            entry = _pipeline_track_val(ctx)
            if entry.get("state") == "blocked":
                print(_pipeline_line(ctx, entry))
                return 1
            print(f"val leg {out.stem} (re)launched and tracked; its checks run when it ends")
            return 0
        if rec is not None and (not ok or rec.get("state") == "blocked"):
            _pipeline_supersede(ctx, rec, why if not ok else "re-checked by --val-check --step (operator)")
            rec = None
        if rec is None:
            guard = _stale_val_guard(ctx)
            if guard is not None:
                base = {"epoch": int(ctx.epoch), "key": "val", "tag": out.stem, "result": str(out),
                        "promoted_sha256": _promoted_sha(ctx),
                        "dispatched_at": _stamp(), "dispatched_t": time.time()}
                print(_pipeline_line(ctx, _pipeline_put_blocked(ctx, base, *guard)))
                return 1
        res = _pipeline_check_val(ctx, run_remedies=True)
        _record_val_result(ctx, res)
    _pipeline_announce(ctx, mark=False)
    return 0 if res["state"] == "passed" else 1


# ── THE HELD-OUT TEST, after the last epoch (`--heldout`) ─────────────────────────────────────────────
# The rule: after the last epoch, run the held-out test set ONCE on the skill set of the LAST SUCCESSFUL
# epoch: the last epoch whose skill set was promoted and whose train and val legs completed. Validation
# legs are reported and do not choose the skill set.
#
# LEG_KEYS is train/val; this program is the held-out step, built so the comparison and every number
# reported from it use ONE task set, the run's DECLARED split's test key, on both sides:
#   * the skill leg is EpochPaths(run, E).leg("test") (= <run>_v<E>_test.json), E = the rule's epoch;
#   * the baseline is RUNTIME `heldout_base_leg`: a measured no-skill leg on that key, or a record
#     derive_restricted_leg() wrote from a measured leg on a SUPERSET of it. Restriction is exact: the N
#     samples of a task are drawn and graded independently of every other task (the runner's cache key
#     is per task and sample; a replay returns the stored score without re-grading), so dropping tasks
#     drops their samples and changes nothing else. A cache-replayed leg would reproduce the same scores
#     at the cost of sandboxes;
#   * HELDOUT_COMPARISON.json is written from those two only after both verify on the split's test key,
#     and its own verify re-derives it. It lives in the OPERATOR dir: it carries per-task held-out detail.
# `--heldout` is invoked by the operator once evolution is over; it is never part of an epoch.
HELDOUT_KEY = "test"
HELDOUT_COMPARISON = "HELDOUT_COMPARISON.json"
HELDOUT_COMPARISON_SCHEMA = "evoskill.heldout_comparison/1"
RESTRICTED_LEG_SCHEMA = "evoskill.restricted_leg/1"
# Leg fields that describe the INSTRUMENT, identical for any subset of its tasks: a restricted record
# restates them. Every other field the source carries that is not recomputed below (per-call and
# per-shard aggregates: served providers, generation checks, shard ledgers, fresh-call counts) describes
# ALL the source's tasks and is named, not restated, in derived_from.source_fields_not_restated.
_RESTATED_FIELDS = ("model", "provider_order", "budget", "provider_extra", "allow_fallbacks",
                    "expect_provider", "expect_model", "runtime_env", "n_samples", "seed",
                    "solver_image", "solver_image_id", "solver_formula_engine", "solver_prompt_sha256_12",
                    "skills_dir", "solver_wall_s", "solver_wall_derived_default_s", "solver_wall_derivation",
                    "concurrency", "image_digest")
_RECOMPUTED_FIELDS = ("task_ids", "per_task", "n", "sample_pass", "sample_total", "pass3", "all3",
                      "per_category", "episode_status", "origin", "turn_exhausted", "tag", "split_file",
                      "split_key", "split_sha256", "split_status", "provider_violations")


def _pass_threshold() -> float:
    from src.orchestrator.outcomes import PASS_THRESHOLD
    return PASS_THRESHOLD


def _task_scores(d: dict, src) -> dict:
    """{task: [score per sample]} of a leg that states task_ids (see _leg_task_ids)."""
    tids = _leg_task_ids(d)
    if tids is None:
        raise ValueError(f"{getattr(src, 'name', src)} states no task_ids; a held-out number cannot be tied "
                         f"to its tasks")
    return {t: list(d["per_task"][str(i)]) for i, t in enumerate(tids)}


def _summary(scores: dict, ns: int) -> dict:
    """sample_pass / sample_total / n / pass@N / all@N / per_category of {task: scores}, by the runner's
    own pass test. ONE definition, used for the restricted record, its re-derivation and the comparison."""
    thr = _pass_threshold()
    per_cat: dict = {}
    tot = {"n": 0, "sample_pass": 0, "sample_total": 0, "pass3": 0, "all3": 0}
    for t, v in scores.items():
        k = sum(1 for x in v if x is not None and x >= thr)
        cat = t.split("/")[0]
        c = per_cat.setdefault(cat, {"n_tasks": 0, "sample_pass": 0, "sample_total": 0, "pass3": 0,
                                     "all3": 0})
        for acc in (tot, c):
            acc["n" if acc is tot else "n_tasks"] += 1
            acc["sample_pass"] += k
            acc["sample_total"] += len(v)
            acc["pass3"] += int(k > 0)
            acc["all3"] += int(k == ns == len(v))
    for c in per_cat.values():
        c["episode_status"] = {"graded_fail": c["sample_total"] - c["sample_pass"], "pass": c["sample_pass"]}
    tot["per_category"] = dict(sorted(per_cat.items()))
    return tot


def derive_restricted_leg(src, split_path, key: str = HELDOUT_KEY, *, exhausted=None,
                          exhausted_from=None) -> dict:
    """A measured leg RESTRICTED to split[key] (option (b) of the drop48 held-out plan), as a leg record
    the held-out step reads exactly like a measured one, with a `derived_from` block saying what it is.

    Refuses (ValueError) when split[key] is not a subset of the source's tasks, when the source's own
    figures do not re-derive from its per_task (then this module's pass arithmetic is not the runner's),
    or when a source episode_status line cannot be attributed to a task. `exhausted` =
    {task: [turn-exhausted? per sample]} makes the record carry turn_exhausted for the kept tasks; without it the
    field is absent and derived_from says so (it is a count in the source and cannot be restricted)."""
    src = Path(src)
    d = json.loads(src.read_text())
    scores = _task_scores(d, src)
    ns = int(d["n_samples"])
    want = split_ids(split_path, key)
    if not want:
        raise ValueError(f"the split {split_path} has no {key!r} ids")
    missing = [t for t in want if t not in scores]
    if missing:
        raise ValueError(f"{len(missing)} of split[{key!r}]'s {len(want)} tasks are not in {src.name} "
                         f"(first {missing[:4]}): a restriction cannot add tasks")
    full = _summary(scores, ns)
    for f in ("sample_pass", "sample_total", "pass3", "all3", "n"):
        if f in d and int(d[f]) != int(full[f]):
            raise ValueError(f"{src.name} records {f}={d[f]} but its per_task gives {full[f]}: the "
                             f"source does not re-derive, so no restriction of it is trustworthy")
    keep = set(want)
    kept = {t: scores[t] for t in want}                               # the split's order
    s = _summary(kept, ns)
    out = {k: d[k] for k in _RESTATED_FIELDS if k in d}
    removed = [t for t in scores if t not in keep]
    thr = _pass_threshold()
    out.update({
        "tag": f"{d.get('tag') or src.stem}.restricted_{_sha12(split_path)}",
        "task_ids": list(want),
        "per_task": {str(i): kept[t] for i, t in enumerate(want)},
        "n": s["n"], "sample_pass": s["sample_pass"], "sample_total": s["sample_total"],
        "pass3": s["pass3"], "all3": s["all3"], "per_category": s["per_category"],
        "origin": {key: s["n"]},
        "split_file": str(Path(split_path).resolve()), "split_key": key,
        "split_sha256": hashlib.sha256(Path(split_path).read_bytes()).hexdigest(),
        "split_status": json.loads(Path(split_path).read_text()).get("status"),
    })
    if d.get("provider_violations") == []:
        out["provider_violations"] = []                      # none in the whole source => none in a subset
    es = d.get("episode_status")
    if isinstance(es, dict):
        nes = {"counts": {"graded_fail": s["sample_total"] - s["sample_pass"], "pass": s["sample_pass"]}}
        for k2, v in es.items():
            if isinstance(v, list):
                bad = [x for x in v if not (isinstance(x, str) and " s" in x
                                            and x.rsplit(" s", 1)[0] in scores)]
                if bad:
                    raise ValueError(f"{src.name} episode_status.{k2} has entries naming no task: {bad[:3]}")
                nes[k2] = [x for x in v if x.rsplit(" s", 1)[0] in keep]
            elif k2 not in ("counts",):
                nes[k2] = v
        out["episode_status"] = nes
    if exhausted is not None:
        out["turn_exhausted"] = sum(sum(1 for x in exhausted[t] if x) for t in want)
    not_restated = sorted(k for k in d if (k not in _RESTATED_FIELDS and k not in _RECOMPUTED_FIELDS)
                          or (k == "provider_violations" and d[k] != []))
    out["derived_from"] = {
        "schema": RESTRICTED_LEG_SCHEMA,
        "kind": f"restriction of a measured leg to the run's split[{key!r}]",
        "source": str(src.resolve()),
        "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "source_tag": d.get("tag"),
        "source_split_file": d.get("split_file"), "source_split_sha256": d.get("split_sha256"),
        "source_counts": {f: full[f] for f in ("n", "sample_pass", "sample_total", "pass3", "all3")},
        "split_file": out["split_file"], "split_sha256": out["split_sha256"], "split_key": key,
        "kept_tasks": len(want),
        "removed_task_ids": removed,
        "removed_samples": sum(len(scores[t]) for t in removed),
        "removed_sample_passes": sum(1 for t in removed for x in scores[t] if x is not None and x >= thr),
        "why_exact": ("each task's N samples are drawn and graded independently of every other task (the "
                      "runner's cache key is per task and sample, and a replay returns the stored score "
                      "without re-grading), so restricting to a subset of tasks removes their samples and "
                      "changes no other sample's score"),
        "turn_exhausted": (exhausted_from or "from the per-sample map the caller gave")
        if exhausted is not None else "ABSENT: the source records only a total, and no per-sample map was given",
        "source_fields_not_restated": not_restated,
        "derived_by": "src/orchestrator/loop.py derive_restricted_leg",
    }
    return out


def _restricted_leg_problems(d: dict) -> list:
    """Why a record carrying derived_from is NOT what derive_restricted_leg() would write from its source
    now. [] = the source re-hashes as recorded and re-derives to the same tasks, scores and figures."""
    df = d.get("derived_from") or {}
    if df.get("schema") != RESTRICTED_LEG_SCHEMA:
        return [f"derived_from.schema is {df.get('schema')!r}, not {RESTRICTED_LEG_SCHEMA}"]
    src, sp = Path(str(df.get("source") or "")), Path(str(df.get("split_file") or ""))
    if not src.is_file():
        return [f"its source {src} is gone, so the restriction cannot be re-derived"]
    if hashlib.sha256(src.read_bytes()).hexdigest() != df.get("source_sha256"):
        return [f"its source {src.name} no longer hashes to the recorded {str(df.get('source_sha256'))[:12]}"]
    if not sp.is_file() or hashlib.sha256(sp.read_bytes()).hexdigest() != df.get("split_sha256"):
        return [f"the split it was restricted to ({sp}) no longer hashes to {str(df.get('split_sha256'))[:12]}"]
    try:
        again = derive_restricted_leg(src, sp, df.get("split_key") or HELDOUT_KEY)
    except (OSError, ValueError, KeyError) as e:
        return [f"re-deriving from {src.name} fails: {e}"]
    fields = ("task_ids", "per_task", "n", "sample_pass", "sample_total", "pass3", "all3", "per_category",
              "episode_status", "split_sha256", "split_key", "n_samples", "model", "skills_dir")
    return [f"{f}: record {str(d.get(f))[:60]} != re-derived {str(again.get(f))[:60]}"
            for f in fields if d.get(f) != again.get(f)]


def last_successful_epoch(run_dir, n_epochs: int, split_path="", n_samples=None, val_gate=None,
                          skip_train_epoch=None) -> tuple:
    """(E or None, [(epoch, why not)]) under the rule: the LAST epoch whose skill set was promoted and
    whose train and val legs completed. Scans n_epochs..1. An epoch qualifies when
      * skills_v<E> holds at least one */SKILL.md and is content-identical to skills_v<E>_staging (the
        promotion the epoch's "promoted set matches staging" step verifies), and
      * <run>_v<E>_train.json and <run>_v<E>_val.json exist (the leg drivers write them only when every
        shard succeeded) and neither is on another task set than the run's split (a leg stating no
        task_ids is not judged). Validation figures are NOT read: they do not choose the skill set.
      * pipeline_val only (`val_gate`, a callable epoch -> (ok, why); None = unchanged): the epoch's train
        checks are recorded as passed and its val leg is complete, checked and PASSED (re-verified now).
        In that order a val check failure never halts evolution, so an epoch can have both legs on disk
        and a failed val check; the held-out program must not select it.
      * skip_last_train only (`skip_train_epoch`, that run's last epoch; None = unchanged): that epoch
        launched no train leg by design, so its train leg is neither required nor judged here; `val_gate`
        then decides it on its val leg and its recorded blocking program."""
    run_dir = Path(run_dir)
    why = []
    for e in range(int(n_epochs), 0, -1):
        ep = EpochPaths(run_dir, e)
        if not ep.promoted.is_dir() or not any(ep.promoted.glob("*/SKILL.md")):
            why.append((e, f"no promoted skill set at {ep.promoted.name}"))
            continue
        if not ep.staging.is_dir() or _skill_fingerprint(ep.staging) != _skill_fingerprint(ep.promoted):
            why.append((e, f"{ep.promoted.name} is not content-identical to {ep.staging.name}"))
            continue
        bad = []
        for k in LEG_KEYS:
            if k == "train" and skip_train_epoch is not None and e == int(skip_train_epoch):
                continue                    # skip_last_train: no train leg by design at the last epoch
            p = ep.leg(k)
            if not p.is_file():
                bad.append(f"no {k} leg ({p.name})")
                continue
            st, det = _leg_on_split(p, split_path, k, n_samples) if split_path else ("unknown", "")
            if st == "bad":
                bad.append(det)
        if bad:
            why.append((e, "; ".join(bad)))
            continue
        if val_gate is not None:
            ok, det = val_gate(e)
            if not ok:
                why.append((e, det))
                continue
        return e, why
    return None, why


def _heldout_side(leg, split_path, n_samples) -> dict:
    """{task: scores} of one side of the held-out comparison, refused unless the leg is EXACTLY the split's
    test key at n_samples and its recorded figures re-derive from its per_task."""
    st, det = _leg_on_split(leg, split_path, HELDOUT_KEY, n_samples)
    if st != "ok":
        raise ValueError(det)
    d = json.loads(Path(leg).read_text())
    scores = _task_scores(d, Path(leg))
    s = _summary(scores, int(d["n_samples"]))
    for f in ("sample_pass", "sample_total", "pass3", "all3", "n"):
        if f in d and int(d[f]) != int(s[f]):
            raise ValueError(f"{Path(leg).name} records {f}={d[f]} but its per_task gives {s[f]}")
    return scores


def heldout_comparison(base_leg, skill_leg, split_path, n_samples, *, epoch=None, skills_dir="") -> dict:
    """The held-out comparison, both sides on the SAME tasks: the split's test key, in the split's order.
    Descriptive counts only (no test statistic): pass counts, pass@N / all@N, per category, and the task-
    and sample-paired tallies a report states. Raises ValueError when either side is not on those tasks."""
    thr = _pass_threshold()
    b = _heldout_side(base_leg, split_path, n_samples)
    s = _heldout_side(skill_leg, split_path, n_samples)
    tasks = split_ids(split_path, HELDOUT_KEY)
    ns = int(n_samples)
    per_task, more, fewer, same, b_only, s_only = {}, 0, 0, 0, 0, 0
    for t in tasks:
        bp = [x is not None and x >= thr for x in b[t]]
        sp_ = [x is not None and x >= thr for x in s[t]]
        kb, ks = sum(bp), sum(sp_)
        more, fewer, same = more + (ks > kb), fewer + (ks < kb), same + (ks == kb)
        b_only += sum(1 for i in range(ns) if bp[i] and not sp_[i])
        s_only += sum(1 for i in range(ns) if sp_[i] and not bp[i])
        per_task[t] = {"base": b[t], "skill": s[t], "base_pass": kb, "skill_pass": ks}
    sb, ss = _summary({t: b[t] for t in tasks}, ns), _summary({t: s[t] for t in tasks}, ns)
    return {
        "schema": HELDOUT_COMPARISON_SCHEMA,
        "rule": "held-out test ONCE on the last successful epoch's skill set (selection rule v2)",
        "epoch": epoch, "skills_dir": str(skills_dir),
        "split": {"file": str(Path(split_path).resolve()), "key": HELDOUT_KEY,
                  "sha256": hashlib.sha256(Path(split_path).read_bytes()).hexdigest()},
        "n_tasks": len(tasks), "n_samples": ns, "same_tasks": True,
        "base": {"leg": str(Path(base_leg).resolve()),
                 "leg_sha256": hashlib.sha256(Path(base_leg).read_bytes()).hexdigest(),
                 "derived": "derived_from" in json.loads(Path(base_leg).read_text()), **sb},
        "skill": {"leg": str(Path(skill_leg).resolve()),
                  "leg_sha256": hashlib.sha256(Path(skill_leg).read_bytes()).hexdigest(), **ss},
        "delta_sample_pass": ss["sample_pass"] - sb["sample_pass"],
        "tasks_skill_more_passes": more, "tasks_skill_fewer_passes": fewer, "tasks_same_passes": same,
        "samples_base_pass_skill_fail": b_only, "samples_skill_pass_base_fail": s_only,
        "per_task": per_task,
    }


def _heldout_comparison_path(ctx: Ctx) -> Path:
    return operator_dir(ctx.run_dir) / HELDOUT_COMPARISON


def _instrument_problems(ctx: Ctx, leg) -> list:
    """A held-out leg must be the spec's instrument: the solver model and provider order it declares."""
    d = json.loads(Path(leg).read_text())
    out = []
    want_m = getattr(ctx.spec, "solver_model", "")
    if want_m and d.get("model") not in (None, want_m):
        out.append(f"{Path(leg).name} ran model {d.get('model')!r}, the spec's solver is {want_m!r}")
    want_o = list(getattr(ctx.spec, "solver_provider_order", ()) or ())
    if want_o and d.get("provider_order") is not None and list(d["provider_order"]) != want_o:
        out.append(f"{Path(leg).name} ran provider order {d['provider_order']}, the spec pins {want_o}")
    return out


def _v_heldout_epoch(ctx: Ctx):
    """The rule's epoch, re-derived from the artifacts; this invocation must be it. pipeline_val: an
    epoch whose val leg is missing, unchecked, running or failed does not qualify (_pipeline_heldout_gate)."""
    gate = _pipeline_heldout_gate(ctx) if _pipeline_on(ctx) else None
    skip_e = _skip_train_epoch(ctx)                 # None unless pipeline_val + skip_last_train
    e, why = last_successful_epoch(ctx.run_dir, int(getattr(ctx.spec, "n_epochs", 0) or 0),
                                   ctx.spec.split_path, getattr(ctx.spec, "n_samples", None),
                                   val_gate=gate, skip_train_epoch=skip_e)
    later = "; ".join(f"epoch {k}: {w}" for k, w in why) or "none"
    if e is None:
        return False, (f"no epoch of {getattr(ctx.spec, 'n_epochs', '?')} qualifies under the held-out rule "
                       f"(promoted skill set + train and val legs on the split"
                       + (" + train checks and val checks PASSED (pipeline_val)" if gate else "")
                       + (f"; epoch {skip_e}'s train leg is skipped by design (skip_last_train)" if skip_e else "")
                       + f"): {later}")
    if e != ctx.epoch:
        return False, (f"the rule selects epoch {e} (later epochs: {later}), but this invocation is epoch "
                       f"{ctx.epoch}. Re-run --heldout without --epoch, or with --epoch {e}.")
    if skip_e is not None and e == skip_e:
        return True, (f"epoch {e} is the last successful epoch: {EpochPaths(ctx.run_dir, e).promoted.name} "
                      f"promoted, val leg on the split, its train leg {SKIPPED_BY_DESIGN} (skip_last_train), "
                      f"blocking program and val checks passed (val re-verified now) (later epochs: {later})")
    return True, (f"epoch {e} is the last successful epoch: {EpochPaths(ctx.run_dir, e).promoted.name} "
                  f"promoted, train and val legs on the split"
                  + (", train and val checks passed (val re-verified now)" if gate else "")
                  + f" (later epochs: {later})")


def _v_heldout_base(ctx: Ctx):
    """The held-out comparator: declared, a no-skill leg, EXACTLY the split's test key at n_samples, the
    spec's instrument, and — when it is a restricted record — re-derivable from its source now."""
    p = getattr(ctx, "heldout_base_leg", "") or ""
    if not p:
        return False, ("no held-out baseline declared (RUNTIME.json 'heldout_base_leg'). Declare the no-skill "
                       "leg on the split's test key, or a record written by `python -m src.orchestrator.loop "
                       "--derive-restricted-leg <measured leg> --split <the run's split> --out <file>`.")
    p = Path(p)
    if not p.is_file():
        return False, f"the declared heldout_base_leg does not exist: {p}"
    st, det = _leg_on_split(p, ctx.spec.split_path, HELDOUT_KEY, getattr(ctx.spec, "n_samples", None))
    if st != "ok":
        return False, f"held-out baseline: {det}"
    d = json.loads(p.read_text())
    if d.get("skills_dir"):
        return False, f"{p.name} loaded skills ({d['skills_dir']}); the held-out baseline is the no-skill leg"
    probs = _instrument_problems(ctx, p)
    if "derived_from" in d:
        probs += _restricted_leg_problems(d)
    if probs:
        return False, "held-out baseline: " + "; ".join(probs[:4])
    how = (f"; restricted from {Path(d['derived_from']['source']).name} "
           f"(sha256:{d['derived_from']['source_sha256'][:12]}) and re-derives identically"
           if "derived_from" in d else "")
    return True, f"held-out baseline {det}; {d.get('sample_pass')}/{d.get('sample_total')}{how}"


def _v_heldout_leg_loaded(ctx: Ctx):
    """The held-out skill leg exists and loaded the selected epoch's promoted set. NO RUNNER: the loop's
    leg launcher (scripts/ssbv2_leg_daytona.sh, frozen) accepts train|val only."""
    p = ctx.paths.leg(HELDOUT_KEY)
    if not p.is_file():
        return False, (f"no held-out leg at {p.name}. The loop does not launch it: launch the leg with "
                       f"tag {p.stem} on the split's test key ({ctx.spec.split_path}) and the skills dir "
                       f"{ctx.paths.promoted} through the run's leg launcher (a person decides; it spends), "
                       f"then re-run --heldout.")
    return _v_leg_loaded_promoted(ctx, HELDOUT_KEY)


def _v_heldout_leg_on_split(ctx: Ctx):
    p = ctx.paths.leg(HELDOUT_KEY)
    if not p.is_file():
        return False, f"no held-out leg at {p.name}"
    st, det = _leg_on_split(p, ctx.spec.split_path, HELDOUT_KEY, getattr(ctx.spec, "n_samples", None))
    probs = ([] if st == "ok" else [det]) + _instrument_problems(ctx, p)
    if probs:
        return False, "held-out leg: " + "; ".join(probs)
    return True, det


def _run_heldout_comparison(ctx: Ctx) -> None:
    p = _heldout_comparison_path(ctx)
    try:
        c = heldout_comparison(ctx.heldout_base_leg, ctx.paths.leg(HELDOUT_KEY), ctx.spec.split_path,
                               ctx.spec.n_samples, epoch=ctx.epoch, skills_dir=ctx.paths.promoted)
    except (OSError, ValueError, KeyError) as e:
        raise Halt(f"the held-out comparison cannot be computed on one task set: {e}")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(c, indent=1))
    os.replace(str(tmp), str(p))


def _v_heldout_comparison(ctx: Ctx):
    """HELDOUT_COMPARISON.json exists and is EXACTLY what the two verified legs give now."""
    p = _heldout_comparison_path(ctx)
    if not p.is_file():
        return False, f"no {p.name} in {p.parent.name} yet"
    try:
        have = json.loads(p.read_text())
        now = heldout_comparison(ctx.heldout_base_leg, ctx.paths.leg(HELDOUT_KEY), ctx.spec.split_path,
                                 ctx.spec.n_samples, epoch=ctx.epoch, skills_dir=ctx.paths.promoted)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        return False, f"{p.name} cannot be re-derived: {e}"
    if have != now:
        diff = sorted(k for k in set(have) | set(now) if have.get(k) != now.get(k))
        return False, (f"{p.name} disagrees with what the legs give now ({diff[:6]}). Move it aside (keep "
                       f"it) and re-run --heldout.")
    return True, (f"{p.name}: {now['n_tasks']} tasks x N={now['n_samples']} on BOTH sides (split "
                  f"sha256:{now['split']['sha256'][:12]}): base {now['base']['sample_pass']}/"
                  f"{now['base']['sample_total']}, skills_v{ctx.epoch} {now['skill']['sample_pass']}/"
                  f"{now['skill']['sample_total']}")


def heldout_steps(ctx=None) -> list:
    """The held-out program. prompt_mode=template adds the grader-failure check on the held-out leg (no number of it is read
    before its grader failures are re-graded), exactly as each epoch does for train/val."""
    steps = [Step("held-out epoch is the last successful one", _v_heldout_epoch, defect="held-out epoch rule"),
             Step("held-out baseline is the split's test tasks", _v_heldout_base, defect="held-out task set"),
             Step("held-out leg loaded what was promoted", _v_heldout_leg_loaded, defect="skills not loaded"),
             Step("held-out leg is the split's test tasks", _v_heldout_leg_on_split, defect="unpaired comparison")]
    if getattr(ctx, "prompt_mode", "ao") == "template":
        steps.append(Step(LEG_D4_STEP.format(key=HELDOUT_KEY), lambda c: _v_leg_outcomes(c, HELDOUT_KEY),
                          run=lambda c: _run_leg_regrade(c, HELDOUT_KEY), defect="grader failed"))
    steps.append(Step("held-out comparison uses the same tasks", _v_heldout_comparison,
                      run=_run_heldout_comparison, defect="held-out task set"))
    return steps


def _safe_verify(step: Step, ctx: Ctx):
    """A verify that CRASHES is a HALT, never a pass. Shared by evaluate() and advance() so the
    two cannot drift into disagreeing about what a broken verify means.

    RETURNS A 3-TUPLE (ok, detail, question). A verify may return either 2 or 3 elements, so all
    ~19 pre-existing steps are untouched and their behaviour is byte-identical; `question` is ""
    for every one of them. Normalising HERE rather than at each call site is deliberate: the two
    callers (evaluate/advance) previously destructured a 2-tuple in three places, and a widening
    done per-site is how they would drift.

    A Blocked raised inside a verify is NOT converted to a failed verify — it propagates, because
    "I need an answer" must not be reportable as "this check failed".
    """
    ctx._asking_step = step.name
    try:
        r = step.verify(ctx)
    except Blocked:
        raise
    except Exception as e:                          # noqa: BLE001
        return False, f"verify raised {type(e).__name__}: {e}", ""
    return _normalise_verify(step, r)


def _normalise_verify(step: Step, r):
    """Coerce whatever a verify returned into (ok, detail, question), refusing anything else.

    A verify that returns None, or a tuple of the wrong arity, must not raise TypeError/ValueError
    straight past evaluate(), advance() and main() ("a verify that CRASHES is a HALT, never a pass")
    and reach the operator as a raw traceback. Forgetting a `return` is a common mistake.
    """
    if isinstance(r, tuple) and len(r) in (2, 3):
        ok = bool(r[0])
        detail = r[1] if isinstance(r[1], str) else str(r[1])
        question = (r[2] if len(r) == 3 else "") or ""
        question = question if isinstance(question, str) else str(question)
        if ok and question:
            # A verify returning (True, detail, question) would pass with the question thrown away
            # -- no block, no record, no trace anywhere. A check that passes has nothing to ask, so
            # the combination is a contradiction and is reported as one.
            return (False, f"verify returned ok=True together with a question ({question!r}). A "
                           f"check that passes has nothing to ask: return (False, detail, "
                           f"question) if it needs an answer, or drop the question.", "")
        return ok, detail, question
    return (False, f"verify returned {type(r).__name__} ({r!r}), not a 2- or 3-tuple. A verify "
                   f"must return (ok, detail) or (ok, detail, question); a missing `return` "
                   f"reads as None here and is treated as a failed check, never a pass.", "")


def evaluate(ctx: Ctx, steps=None):
    """Run every verify and return (results, first_halt). Never raises on a failed verify —
    the caller decides; `--step` halts, `--plan` reports.

    IT NEVER RAISES Blocked EITHER. evaluate() is the BOARD, not the actor: if Blocked propagated,
    the first step to ask a question would abort the whole report, and the operator could not see
    what else was already true. The board's rows are advisory, and advance() — which is where a
    block actually stops work — still raises. A blocked row is rendered as ASK, distinct from both
    ok and HALT.

    Each row is (step, ok, detail, question); `report()` and the selftest read rows positionally
    and tolerate both widths.
    """
    out = []
    halt = None
    for s in steps or _steps_for(ctx):
        try:
            ok, detail, question = _safe_verify(s, ctx)
        except Blocked as e:
            ok, detail, question = False, (e.detail or "needs an answer before this can be "
                                           "checked"), e.question
        if question:
            detail = f"{detail}\n        ASKS: {question}"
        out.append((s, ok, detail, question))
        if not ok and halt is None:
            halt = (s, detail)
    return out, halt


def _resolve_question(ctx: Ctx, s: Step, detail: str, question: str, rounds: int = 4):
    """A verify RETURNED a question. Route it through ask() so the answer channel is real.

    Raising `Blocked(qid=s.name)` directly for a returned question would make main()'s printed
    instruction ("put {"<step name>": "your answer"} in ANSWERS_e{N}.json, then re-run") FALSE: only
    ask() reads the answers file, and a verify that merely RETURNS a question never calls ask(). The
    operator would write the answer, re-run, and get the identical block forever — a clarification
    loop that cannot terminate, which is worse than a refusal.

    Routing through ask() makes the printed procedure true: ask() reads ANSWERS_e{N}.json under
    the SAME key main() tells the operator to write, blocks when it is absent, enforces the
    question<->answer binding, and clears the outstanding record once answered. The verify is then
    re-run — which is the only party that can decide whether the answer settles anything.

    IT IS BOUNDED, AND A VERIFY THAT CANNOT CONSUME ITS OWN ANSWER SAYS SO. If the same question
    comes back after it has been answered, the verify is not reading the answer; that is a coding
    error in the verify, and it HALTs with the fix rather than re-blocking. A verify that asks a
    DIFFERENT question is making progress, so it may ask again, up to `rounds`.
    """
    seen = []
    for _ in range(rounds):
        if question in seen:
            raise Halt(
                f"{s.name}: asked the same question again after it was answered — "
                f"{question!r}\n"
                f"  The answer is on disk at {answers_path(ctx.run_dir, ctx.epoch)} under the key "
                f"{s.name!r}, so this verify is not reading it. A verify that asks must consume "
                f"the answer through ask(ctx, {s.name!r}, ...); returning the question as a third "
                f"tuple element only DELIVERS it. Blocking again would loop forever.")
        seen.append(question)
        ask(ctx, s.name, question)          # Blocked when unanswered — that is the normal path
        ok, detail, question = _safe_verify(s, ctx)
        if not question:
            return ok, detail
    raise Halt(f"{s.name}: still asking after {rounds} answered questions ({seen!r}). A "
               f"clarification loop that cannot terminate is worse than a refusal.")


def advance(ctx: Ctx, steps=None):
    """Execute AT MOST ONE step: the first whose verify does not already pass.

    Returns (step_or_None, detail). None means every step already verifies — the epoch is done
    and there was nothing to do, which is the correct answer on a re-invocation and the reason
    every runner is idempotent.

    THREE PROPERTIES:

    1. IT ADVANCES ON THE VERIFY, NEVER ON THE RUN. `s.run(ctx)` returning without raising
       proves only that a subprocess exited 0. An evolver that "wrote 5 skills" the loader reads
       as 1, a run that "succeeded" having read the comparator arm's answers, a process that
       "launched" and was dead in 200ms, a split "built with 45 tasks" most of which do not exist:
       every one of them passes an exit-code check. So the run's return value is discarded and the verify is asked
       again, from the artifact, through the artifact's real consumer.

    2. A STEP THAT RAN AND STILL FAILS IS A HALT, NOT A RETRY. Re-running an evolver whose
       evidence was contaminated just produces contaminated output again.

    3. ONE STEP PER INVOCATION. The operator sees each transition. A driver that runs an epoch
       end-to-end unattended catches only DECLARED failures, and the defects that matter are the
       undeclared ones. (`--drive` takes these same single steps in a row, and stops at the first
       that does not verify.)
    """
    for s in steps or _steps_for(ctx):
        ok, detail, question = _safe_verify(s, ctx)
        if question:
            ok, detail = _resolve_question(ctx, s, detail, question)
        if ok:
            continue
        if s.operator_gate:
            raise OperatorGate(f"{s.name}: {detail}\n"
                       f"  This gate is deliberately not automatable: a person must act before the "
                       f"run continues. Auto-satisfying it would turn 'a person did this' into 'a "
                       f"person was assumed to have done this'.")
        if s.run is None:
            raise Halt(f"{s.name}: {detail}\n"
                       f"  This step has no runner — it is performed elsewhere (by a human or "
                       f"another process) and the loop only checks it.")
        # Guarded: main() catches only Blocked and Halt, so an operator-facing refusal a runner
        # raises as a plain exception (the prompt author raises ValueError for its own) would reach
        # the operator as a raw traceback instead of the HALT block that explains what to do.
        # Blocked is a BaseException and passes straight through, which is why it is one.
        try:
            s.run(ctx)
        except Halt:
            raise
        except Exception as e:                      # noqa: BLE001
            raise Halt(f"{s.name} RAISED {type(e).__name__}: {e}\n"
                       f"  The runner refused rather than returning. Nothing was verified, so no "
                       f"artifact from this step should be trusted; read the message above, fix "
                       f"the cause, and re-run.") from e
        ok, detail, _q = _safe_verify(s, ctx)
        if not ok:
            raise Halt(f"{s.name} RAN, and its verify still fails: {detail}\n"
                       f"  The step reported no error, so this is a disagreement between what "
                       f"the step did and what the artifact says. Do not re-run it.")
        return s, detail
    return None, "every step verifies — nothing to do"


def report(ctx: Ctx, results) -> None:
    """Render the board. A row is (step, ok, detail[, question]).

    ASK IS ITS OWN MARK, NOT A HALT. Rendered as a plain HALT with the question dropped, "I need to
    know something" would be indistinguishable from "something is broken" on `--plan`, the precise
    confusion Blocked exists to remove.
    """
    print(f"LOOP — {ctx.run_dir.name} epoch {ctx.epoch}\n")
    n_ask = n_skip = n_ok = 0
    for row in results:
        s, ok, detail = row[0], row[1], row[2]
        question = row[3] if len(row) > 3 else ""
        if question:
            mark, n_ask = "ASK ", n_ask + 1
        elif ok and detail.startswith(UNMEASURED):
            mark, n_skip = "SKIP", n_skip + 1
        elif ok:
            mark, n_ok = "ok  ", n_ok + 1
        else:
            mark = "HALT"
        tag = f" [{s.defect}]" if s.defect else (" [operator gate]" if s.operator_gate else "")
        print(f"  {mark}  {s.name}{tag}\n        {detail}")
    n_halt = len(results) - n_ok - n_skip - n_ask
    print(f"\n  {n_ok} pass, {n_halt} HALT"
          + (f", {n_ask} ASK (needs an answer from you — not a defect)" if n_ask else "")
          + (f", {n_skip} UNMEASURED (not a pass — see above)" if n_skip else ""))
    if n_ask:
        print(f"  Answer in {answers_path(ctx.run_dir, ctx.epoch)}; outstanding questions are "
              f"listed in {blocked_path(ctx.run_dir, ctx.epoch).name}.")


# ── selftest: every verify must fail on a known-bad input ────────────────────────────────
def _selftest_split_and_heldout(check, td: Path) -> None:
    """The declared-split checks and the held-out program, on synthetic SSB-shaped
    legs (per_task keyed by index, task_ids beside it) and a split with a "dropped" key."""
    from types import SimpleNamespace as _NS
    print("\n  declared split + held-out")
    rd = td / "hrun"
    rd.mkdir()
    sp = td / "hsplit.json"
    sp.write_text(json.dumps({"train": ["A/01", "A/02"], "val": ["A/03", "A/04"],
                              "test": ["B/01", "B/02", "C/01"], "unused": ["A/09"],
                              "dropped": {"task_ids": ["B/09"], "reason": "reference fails its grader"}}))
    spec = _NS(benchmark="ssbv2", split_path=str(sp), n_samples=3, n_epochs=3, name="hrun",
               solver_model="m", solver_provider_order=("p",))

    def leg(p, tasks, scores, **extra):
        d = {"task_ids": tasks, "per_task": {str(i): s for i, s in enumerate(scores)}, "n_samples": 3,
             "model": "m", "provider_order": ["p"], "skills_dir": None, "n": len(tasks),
             "sample_pass": sum(1 for s in scores for x in s if x >= 0.999),
             "sample_total": 3 * len(tasks), "turn_exhausted": 0, **extra}
        Path(p).write_text(json.dumps(d))
        return Path(p)
    one, zero = [1.0, 0.0, 1.0], [0.0, 0.0, 0.0]
    st, d = _leg_on_split(leg(rd / "t_ok.json", ["A/01", "A/02"], [one, zero]), sp, "train", 3)
    check("a leg on exactly split[train] -> ok", st == "ok", True, d)
    st, d = _leg_on_split(leg(rd / "t_bad.json", ["A/01", "A/03"], [one, zero]), sp, "train", 3)
    check("a leg of the same SIZE on other tasks -> bad (per_task index keys hide it)", st == "bad", True, d)
    # equal per_task key sets, different tasks
    ctx = Ctx(rd, 1, spec, rd / "oracle", base_leg=str(rd / "t_ok.json"))
    leg(ctx.paths.leg("train"), ["A/01", "A/03"], [one, zero], skills_dir=str(ctx.paths.promoted))
    ok, d = _v_legs_comparable(ctx, "train")
    check("base and skill legs, same size, different task ids -> HALT", ok, False, d)
    leg(ctx.paths.leg("train"), ["A/02", "A/01"], [zero, one], skills_dir=str(ctx.paths.promoted))
    ok, d = _v_legs_comparable(ctx, "train")
    check("same task set in another order -> pass", ok, True, d)
    # a dropped task's id in a prompt
    ctx.paths.prompt.write_text("Train scored 2/6 episodes. See B/09 for a hard one.\n")
    ok, d = _v_prompt_no_leakage(ctx)
    check("prompt names a DROPPED task id -> HALT", ok, False, d)
    ctx.paths.prompt.write_text("Train scored 2/6 episodes.\n")
    ok, d = _v_prompt_no_leakage(ctx)
    check("prompt names no id of any key -> pass", ok, True, d)
    # the held-out baseline: a superset leg (4 test-ish tasks) restricted to split[test]
    src = leg(rd / "tb_full.json", ["B/01", "B/02", "B/09", "C/01"], [one, zero, zero, one],
              split_sha256="old", episode_status={"counts": {}, "no_output_file": ["B/09 s0", "C/01 s1"]})
    rec = derive_restricted_leg(src, sp, "test")
    check("restriction keeps split[test] in split order with re-derived figures",
          (rec["task_ids"], rec["sample_pass"], rec["sample_total"], rec["episode_status"]["no_output_file"])
          == (["B/01", "B/02", "C/01"], 4, 9, ["C/01 s1"]), True, str(rec["task_ids"]))
    base = rd / "tb_restricted.json"
    base.write_text(json.dumps(rec))
    hctx = Ctx(rd, 2, spec, rd / "oracle")
    ok, d = _v_heldout_base(hctx)
    check("no heldout_base_leg declared -> HALT", ok, False, d)
    hctx.heldout_base_leg = str(src)
    ok, d = _v_heldout_base(hctx)
    check("held-out baseline on a SUPERSET of split[test] -> HALT", ok, False, d)
    hctx.heldout_base_leg = str(base)
    ok, d = _v_heldout_base(hctx)
    check("restricted baseline, re-derives from its source -> pass", ok, True, d)
    bad = dict(rec, per_task=dict(rec["per_task"], **{"0": [1.0, 1.0, 1.0]}))
    bad["sample_pass"] = 6
    base.write_text(json.dumps(bad))
    ok, d = _v_heldout_base(hctx)
    check("restricted record edited after derivation -> HALT", ok, False, d)
    base.write_text(json.dumps(rec))
    # the rule's epoch: e3 lacks its val leg, e2 qualifies
    for e in (2, 3):
        ep = EpochPaths(rd, e)
        for dd in (ep.staging, ep.promoted):
            (dd / "s").mkdir(parents=True, exist_ok=True)
            (dd / "s" / "SKILL.md").write_text("---\nname: s\ndescription: d\n---\nb\n")
        leg(ep.leg("train"), ["A/01", "A/02"], [one, zero], skills_dir=str(ep.promoted))
        if e == 2:
            leg(ep.leg("val"), ["A/03", "A/04"], [one, zero], skills_dir=str(ep.promoted))
    e, why = last_successful_epoch(rd, 3, sp, 3)
    check("rule: e3 has no val leg, e2 is complete -> epoch 2", e == 2, True, str(why))
    ok, d = _v_heldout_epoch(Ctx(rd, 3, spec, rd / "oracle"))
    check("held-out invoked at a non-selected epoch -> HALT", ok, False, d)
    # the comparison: one task set on both sides
    hl = hctx.paths.leg(HELDOUT_KEY)
    leg(hl, ["C/01", "B/01", "B/02"], [one, one, one], skills_dir=str(hctx.paths.promoted))
    ok, d = _v_heldout_comparison(hctx)
    check("no comparison written yet -> HALT", ok, False, d)
    _run_heldout_comparison(hctx)
    ok, d = _v_heldout_comparison(hctx)
    check("comparison on the split's 3 test tasks, both sides -> pass", ok, True, d)
    leg(hl, ["C/01", "B/01", "A/01"], [one, one, one], skills_dir=str(hctx.paths.promoted))
    ok, d = _v_heldout_comparison(hctx)
    check("held-out skill leg on other tasks -> HALT", ok, False, d)
    ok, d = _v_heldout_leg_on_split(hctx)
    check("held-out leg not the split's test key -> HALT", ok, False, d)


def selftest() -> int:
    import shutil
    import tempfile
    print("SELFTEST — each verify must FAIL on a known-bad input\n")
    import dataclasses
    from src.orchestrator.spec import split_digest, tau3_banking_2525
    fails = 0

    def check(label, got_ok, want_ok, detail):
        nonlocal fails
        ok = got_ok is want_ok
        print(f"  [{'ok' if ok else 'BAD'}] {label}\n        {detail[:130]}")
        fails += 0 if ok else 1

    with tempfile.TemporaryDirectory() as td:
        rd = Path(td) / "run"
        rd.mkdir()
        # the spec's own split file is replaced by a synthetic one, so the selftest needs no run on disk
        sp_f = Path(td) / "split.json"
        sp_f.write_text(json.dumps({"train": [f"task_{i:03d}" for i in range(1, 26)],
                                    "val": [f"task_{i:03d}" for i in range(26, 51)],
                                    "test": [f"task_{i:03d}" for i in range(51, 98)]}))
        spec = dataclasses.replace(tau3_banking_2525(), split_path=str(sp_f), split_sha256=split_digest(sp_f))
        ctx = Ctx(rd, 2, spec, rd / "oracle")

        # skills_load: no staging must FAIL; Claude Code is the judge of a staged set, so it passes
        st = ctx.paths.staging
        ok, d = _v_skills_load(ctx)
        check("no staging dir -> HALT", ok, False, d)
        st.mkdir(parents=True)
        for n in ("a", "b"):
            (st / n).mkdir()
            (st / n / "SKILL.md").write_text(f"---\nname: {n}\ndescription: when x\n---\nbody\n")
        ok, d = _v_skills_load(ctx)
        check("proper <name>/SKILL.md dirs -> pass", ok, True, d)

        # promotion divergence must FAIL
        shutil.copytree(st, ctx.paths.promoted)
        ok, d = _v_promoted_matches_staging(ctx)
        check("promoted == staging -> pass", ok, True, d)
        (ctx.paths.promoted / "a" / "SKILL.md").write_text("---\nname: a\ndescription: EDITED\n---\n")
        ok, d = _v_promoted_matches_staging(ctx)
        check("promoted content drifted from staging -> HALT", ok, False, d)

        # leg loaded the wrong skills dir must FAIL
        ctx.paths.leg("train").write_text(json.dumps(
            {"skills_dir": str(rd / "skills_v1"), "pass": 25, "total": 75,
             "termination_reasons": {"max_steps": 10}}))
        ok, d = _v_leg_loaded_promoted(ctx, "train")
        check("leg loaded a DIFFERENT skills dir -> HALT", ok, False, d)

        # a JSON with NO skills_dir must never read as "loaded ." — Path("") is
        # truthy and equals Path("."). Absence falls back to the leg log's [SKILL]
        # banner, or HALTs when there is no log either.
        ctx.paths.leg("train").write_text(json.dumps({"pass": 25, "total": 75}))
        logf = ctx.paths.leg("train").with_suffix(".log")
        if logf.exists():
            logf.unlink()
        ok, d = _v_leg_loaded_promoted(ctx, "train")
        check("no skills_dir and no leg log -> HALT", ok, False, d)
        logf.write_text("  [SKILL] a: how to do a\n  [SKILL] b: how to do b\n")
        ok, d = _v_leg_loaded_promoted(ctx, "train")
        check("no skills_dir, log [SKILL] set == promoted -> pass", ok, True, d)
        logf.write_text("  [SKILL] a: how to do a\n")
        ok, d = _v_leg_loaded_promoted(ctx, "train")
        check("no skills_dir, log missing a promoted skill -> HALT", ok, False, d)
        ctx.paths.leg("train").write_text(json.dumps(
            {"skills_dir": str(ctx.paths.promoted), "pass": 25, "total": 75,
             "termination_reasons": {"max_steps": 10},
             "episodes_excluded_infra": 0, "episodes_timeout_scored_zero": 0}))
        ok, d = _v_leg_loaded_promoted(ctx, "train")
        check("leg loaded the promoted dir -> pass", ok, True, d)

        # infra exclusions must FAIL
        ok, d = _v_no_infra_exclusions(ctx, "train")
        check("0 excluded / 0 timeout-zeroed -> pass", ok, True, d)
        j = json.loads(ctx.paths.leg("train").read_text())
        j["episodes_excluded_infra"] = 3
        ctx.paths.leg("train").write_text(json.dumps(j))
        ok, d = _v_no_infra_exclusions(ctx, "train")
        check("3 episodes infra-excluded -> HALT", ok, False, d)

        # outcome reconciliation must FAIL when the buckets cannot add up
        j["episodes_excluded_infra"] = 0
        j["pass"] = 70; j["termination_reasons"] = {"max_steps": 10}
        ctx.paths.leg("train").write_text(json.dumps(j))
        ok, d = _v_outcomes_reconcile(ctx, "train")
        check("pass + never-graded > total -> HALT", ok, False, d)
        j["pass"] = 25
        ctx.paths.leg("train").write_text(json.dumps(j))
        ok, d = _v_outcomes_reconcile(ctx, "train")
        check("buckets reconcile -> pass", ok, True, d)

        # held-out material in the digests must FAIL
        dg = ctx.paths.digests
        dg.mkdir(parents=True)
        sp = json.loads(Path(spec.split_path).read_text())
        tr = sp["train"] if isinstance(sp["train"][0], str) else [x["id"] for x in sp["train"]]
        va = sp["val"] if isinstance(sp["val"][0], str) else [x["id"] for x in sp["val"]]
        for t in tr[:5]:
            (dg / f"{t}.md").write_text("x")
        ok, d = _v_digests_train_only(ctx)
        check("train-only digests -> pass", ok, True, d)
        (dg / f"{va[0]}.md").write_text("x")
        ok, d = _v_digests_train_only(ctx)
        check("a VAL task in the evidence surface -> HALT", ok, False, d)
        (dg / f"{va[0]}.md").unlink()

        # fix (2): the surface above is train-only AND hollow — it must not pass as complete
        ok, d = _v_digests_complete(ctx)
        check("five one-byte train digests -> HALT (membership is not coverage)", ok, False, d)
        cov = rd / "coverage_probe"
        cov.mkdir()

        def _t3_digest(tid, verdicts):
            body = [f"# {tid}  (banking, variant x)", "", "**Result: 0/3 samples passed.**", ""]
            for i, v in enumerate(verdicts):
                body += [f"## Rollout s{i} — FAIL [kind=fail] (score=0.0, 9 turns, 1.0s)", "",
                         "### Grade (authoritative — this is what produced the score)", ""]
                body += (["_No structured verdict on this trace — none._", ""] if v is None
                         else ["- reward **0.0**", ""])
                body += ["### Transcript", "", "### Grade (authoritative — this is what "
                         "produced the score)", "pasted by a tool, not a grade", ""]
            return "\n".join(body)
        for t in tr:
            (cov / f"{t.replace('/', '_')}.md").write_text(_t3_digest(t, [1, 1, 1]))
        ok, d = _v_digests_complete(ctx, [cov])
        check("every train task x 3 samples with a grade -> pass", ok, True, d)
        (cov / f"{tr[0]}.md").write_text(_t3_digest(tr[0], [1, None, 1]))
        ok, d = _v_digests_complete(ctx, [cov])
        check("one sample with no structured verdict -> HALT", ok, False, d)
        (cov / f"{tr[0]}.md").write_text(_t3_digest(tr[0], [1, 1]))
        ok, d = _v_digests_complete(ctx, [cov])
        check("one task with N-1 samples -> HALT", ok, False, d)
        (cov / f"{tr[0]}.md").write_text(_t3_digest(tr[0], [1, 1, 1]))
        ok, d = _v_digests_complete(ctx, [cov, dg])
        check("a complete dir listed with a hollow one -> HALT (every dir is checked)",
              ok, False, d)
        names = [s.name for s in epoch_steps()]
        pos = names.index("evidence surface is complete") if (
            "evidence surface is complete" in names) else -1
        check("coverage step follows train-only and precedes the AO",
              pos > 0 and names[pos - 1] == "evidence surface is train-only"
              and pos < names.index("evolver prompt authored by the AO"), True, str(names[:4]))
        # ...and an EXISTING SSB v2 surface is re-verified, not trusted by the idempotent return
        from types import SimpleNamespace as _NS
        (rd / "ssb_split.json").write_text(json.dumps(
            {"train": ["Debugging/01_01"], "val": [], "test": []}))
        sctx = Ctx(rd / "ssbrun", 1, _NS(benchmark="ssbv2", n_samples=3, name="ssbrun",
                                         split_path=str(rd / "ssb_split.json"),
                                         solver_model="m"),
                   rd / "oracle", digest_caches=(rd / "x_cache",))
        sctx.paths.digests.mkdir(parents=True)
        (sctx.paths.digests / "Debugging_01_01.md").write_text("\n".join(
            f"## Rollout s{k} — FAIL (1 turns)" for k in range(3)))
        try:
            _run_build_digests_ssbv2(sctx)
            got = (True, "returned: an existing surface with no scorer diff was trusted")
        except Halt as e:
            got = (False, str(e))
        check("existing SSB surface without scorer diffs -> HALT, not trusted", got[0], False,
              got[1])

        # KEY ROUTING. Deriving the val comparator by filename substitution and, when that file
        # does not exist (base legs living in ANOTHER run's dir), falling back to base_leg — the
        # TRAIN base — would trip the "different task sets" FAIL on a perfectly good run, and a
        # guard that cries wolf gets switched off. So: with only base_leg declared, VAL must come
        # back UNMEASURED, never FAIL.
        def _mkleg(p, nsamp, tasks, eps):
            p.write_text(json.dumps({"n_samples": nsamp, "episodes": eps,
                                     "per_task": {t: {} for t in tasks}}))
        _mkleg(rd / "btrain.json", 3, ["t1", "t2"], 6)
        _mkleg(ctx.paths.leg("val"), 3, ["v1", "v2"], 6)
        ctx.base_leg = str(rd / "btrain.json")
        ctx.base_leg_val = ""
        ok, d = _v_legs_comparable(ctx, "val")
        check("val pairing with ONLY a train base declared -> UNMEASURED, not HALT",
              ok and d.startswith(UNMEASURED), True, d)
        _mkleg(rd / "bval.json", 3, ["v1", "v2"], 6)
        ctx.base_leg_val = str(rd / "bval.json")
        ok, d = _v_legs_comparable(ctx, "val")
        check("val pairing against the declared VAL base -> pass", ok, True, d)
        _mkleg(rd / "bval.json", 1, ["v1", "v2"], 2)
        ok, d = _v_legs_comparable(ctx, "val")
        check("val base N=1 vs val skill N=3 -> HALT", ok, False, d)
        ctx.base_leg = ctx.base_leg_val = ""

        ctx.paths.prompt.write_text("# Iteration 2\n\nIt scored **25 of 75 episodes**.\n")

        # prompt numbers that disagree with the prior leg must FAIL
        (rd / f"{rd.name}_v1_train.json").write_text(json.dumps({"pass": 23, "total": 75}))
        ok, d = _v_prompt_numbers(ctx)
        check("prompt says 25/75, prior leg says 23/75 -> HALT", ok, False, d)
        ctx.paths.prompt.write_text("# Iteration 2\n\nIt scored **23 of 75 episodes**.\n")
        ok, d = _v_prompt_numbers(ctx)
        check("prompt re-derives from the prior leg -> pass", ok, True, d)

        # a verify that raises must be a HALT, not a pass
        boom = Step("deliberately broken verify",
                    lambda c: (_ for _ in ()).throw(RuntimeError("boom")))
        res, halt = evaluate(ctx, [boom])
        check("a verify that RAISES -> HALT", res[0][1], False, res[0][2])

        # grants must never include the run dir or a peer arm
        g = ctx.paths.grants(rd / "oracle")
        bad = [p for p in g if p == rd]
        check("grants exclude the run dir itself", not bad, True,
              f"grants = {[p.name for p in g]}")

        # ALL-EPOCHS EVIDENCE: an explicit bounded range, checked as EXACT sets.
        for K in (1, 2, 4):
            got_ev = EpochPaths(rd, K).evidence_dirs()
            want_ev = [(rd / f"digests_e{j}_train").resolve() for j in range(1, K + 1)]
            check(f"evidence_dirs at epoch {K} is exactly digests_e1..e{K}", got_ev == want_ev,
                  True, str([p.name for p in got_ev]))
        # a LATER epoch's surface and stale siblings sit on disk; none may be granted
        for n in ("digests_e5_train", "digests_e3_train.062721.bak", "digests_e0_train",
                  "digests_probe_x", "digests_e1_train"):
            (rd / n).mkdir(exist_ok=True)
        ep4 = EpochPaths(rd, 4)
        g4 = ep4.grants(rd / "oracle", all_epochs=True)
        want4 = [(rd / f"digests_e{j}_train").resolve() for j in range(1, 5)] + [ep4.staging,
                                                                                 rd / "oracle"]
        check("all-epochs grants at K=4 == digests_e1..e4 + staging + oracle, exactly",
              g4 == want4, True, str([p.name for p in g4]))
        check("digests_e5_train ON DISK is NOT granted at K=4 (no future evidence)",
              any("digests_e5" in str(p) for p in g4), False, str([p.name for p in g4]))
        check("current-epoch grants are unchanged: [digests_e4, staging, oracle]",
              ep4.grants(rd / "oracle") == [ep4.digests, ep4.staging, rd / "oracle"], True,
              str([p.name for p in ep4.grants(rd / "oracle")]))
        try:
            _evidence_mode(Ctx(rd, 2, spec, rd / "oracle", evolver_evidence="all_epochs",
                               evolver_sandbox=False))
            got = (True, "accepted")
        except Halt as e:
            got = (False, str(e))
        check("all_epochs with the evolver sandbox OFF -> HALT", got[0], False, got[1])
        try:
            _evidence_mode(Ctx(rd, 2, spec, rd / "oracle", evolver_evidence="all"))
            got = (True, "accepted")
        except Halt as e:
            got = (False, str(e))
        check("an unknown evolver_evidence value -> HALT, never a guessed mode", got[0], False,
              got[1])
        check("absent key -> current_epoch, one digest dir",
              _evidence_dirs(Ctx(rd, 2, spec, rd / "oracle")) == [EpochPaths(rd, 2).digests],
              True, "")
        # the train-only verify covers EVERY granted dir in all-epochs mode
        actx = Ctx(rd, 2, spec, rd / "oracle", evolver_evidence="all_epochs")
        for t in tr[:3]:
            (rd / "digests_e1_train" / f"{t}.md").write_text("x")
        ok, d = _v_digests_train_only(actx)
        check("all-epochs: e1 and e2 both train-only -> pass", ok, True, d)
        (rd / "digests_e1_train" / f"{va[0]}.md").write_text("x")
        ok, d = _v_digests_train_only(actx)
        check("all-epochs: a VAL digest in an EARLIER epoch's dir -> HALT", ok, False, d)
        (rd / "digests_e1_train" / f"{va[0]}.md").unlink()
        (rd / "digests_e1_train" / f"{tr[5]}.md").symlink_to(rd / "digests_e2_train" /
                                                              f"{tr[0]}.md")
        ok, d = _v_digests_train_only(actx)
        check("all-epochs: a SYMLINKED file in an earlier dir -> HALT", ok, False, d)
        (rd / "digests_e1_train" / f"{tr[5]}.md").unlink()
        ok, d = _v_digests_complete(actx)
        check("all-epochs coverage checks e1 too (hollow e1 -> HALT)",
              ok is False and "digests_e1_train" in d, True, d)

        # Each granted dir names the library that produced it, and the
        # prompt names every granted dir by absolute path, before the evolver is dispatched.
        from . import skill_library as _sl
        names = [s.name for s in epoch_steps()]
        placed = (all(n in names for n in ("evidence surface is complete",
                                           "evidence names its producing library",
                                           "prompt names every granted evidence dir",
                                           "skills load through the real consumer"))
                  and names.index("evidence names its producing library")
                  == names.index("evidence surface is complete") + 1
                  and names.index("prompt names every granted evidence dir")
                  < names.index("skills load through the real consumer"))
        check("library step follows coverage; names-evidence step precedes the evolver dispatch",
              placed, True, str(names[:6]))
        lrd = (rd / "d6run").resolve()
        (lrd / "digests_e1_train").mkdir(parents=True)
        (lrd / "digests_e1_train" / "x.md").write_text("# base rollouts, no skill loaded\n")
        (lrd / "skills_v1" / "a").mkdir(parents=True)
        (lrd / "skills_v1" / "a" / "SKILL.md").write_text("---\nname: a\n---\nv1 body\n")
        e2 = lrd / "digests_e2_train"
        _lab, _sha = _sl.snapshot(lrd / "skills_v1", e2)
        (e2 / "x.md").write_text(_sl.header_line(_lab, _sha) + "\n" + _sl.marker(
            "a", len((lrd / "skills_v1" / "a" / "SKILL.md").read_bytes()), _lab, _sha,
            "a/SKILL.md") + "\n")
        lctx = Ctx(lrd, 2, spec, lrd / "oracle", evolver_evidence="all_epochs")
        ok, d = _v_evidence_libraries(lctx)
        check("all-epochs: e2 carries skills_v1 byte for byte, markers anchored -> pass",
              ok, True, d)
        (lrd / "skills_v1" / "a" / "SKILL.md").write_text("---\nname: a\n---\nEDITED\n")
        ok, d = _v_evidence_libraries(lctx)
        check("all-epochs: _library/ no longer matches skills_v1 -> HALT", ok, False, d)
        ok, d = _v_evidence_libraries(Ctx(lrd, 2, spec, lrd / "oracle"))
        check("current_epoch: the library step does not apply", ok, True, d)
        named = f"Evidence: {lrd}/digests_e1_train and {lrd}/digests_e2_train/_library/.\n"
        lctx.paths.prompt.write_text(named)
        ok, d = _v_prompt_names_evidence(lctx)
        check("prompt names both granted dirs by absolute path -> pass", ok, True, d)
        lctx.paths.prompt.write_text(named + "see outputs/d6run/digests_e2_train\n")
        ok, d = _v_prompt_names_evidence(lctx)
        check("a RELATIVE digest path -> HALT", ok, False, d)
        lctx.paths.prompt.write_text(named + f"and {lrd}/digests_e3_train\n")
        ok, d = _v_prompt_names_evidence(lctx)
        check("a LATER epoch's dir named -> HALT", ok, False, d)

        _selftest_split_and_heldout(check, Path(td))

    # ── advance(): the property that a run's own report is never believed ─────────────────
    # These use synthetic Steps rather than the real epoch, because the point under test is the
    # CONTROL FLOW, and it must be exercised on inputs where each branch is reachable.
    print("\n  advance() — one step, and it advances on the verify")
    with tempfile.TemporaryDirectory() as td:
        rd = Path(td) / "run"
        rd.mkdir()
        ctx = Ctx(rd, 1, spec, rd / "oracle")
        flag = {"ran": 0}

        def passing(_c):
            return True, "already true"

        def failing(_c):
            return False, "not true"

        def liar(_c):                       # reports success, changes nothing
            flag["ran"] += 1

        def fixer(_c):
            flag["ran"] += 1
            flag["fixed"] = True

        def conditional(_c):
            return bool(flag.get("fixed")), "fixed" if flag.get("fixed") else "not yet"

        s, d = advance(ctx, [Step("a", passing), Step("b", passing)])
        check("all steps verify -> nothing to do", s is None, True, d)

        try:
            advance(ctx, [Step("gate", failing, operator_gate=True)])
            check("unsatisfied operator gate -> HALT", False, True, "returned instead")
        except Halt as e:
            check("unsatisfied operator gate -> HALT", "not automatable" in str(e), True, str(e)[:80])

        try:
            advance(ctx, [Step("verify-only", failing)])
            check("failing step with no runner -> HALT", False, True, "returned instead")
        except Halt as e:
            check("failing step with no runner -> HALT", "no runner" in str(e), True, str(e)[:80])

        # THE ONE THAT MATTERS. The runner returns cleanly and the artifact still disagrees.
        # An exit-code driver ships this; advance() must stop, and must NOT re-run.
        flag["ran"] = 0
        try:
            advance(ctx, [Step("lying step", failing, run=liar)])
            check("run() 'succeeds' but verify still fails -> HALT", False, True, "returned")
        except Halt as e:
            check("run() 'succeeds' but verify still fails -> HALT",
                  "still fails" in str(e) and flag["ran"] == 1, True,
                  f"halted after exactly {flag['ran']} run, no retry")

        flag["ran"] = 0
        s, d = advance(ctx, [Step("real step", conditional, run=fixer)])
        check("run() actually fixes it -> step returned", s is not None and flag["ran"] == 1,
              True, f"{s.name if s else None}: {d}")

        # Idempotence: a second invocation finds it satisfied and does NOT run again.
        s2, d2 = advance(ctx, [Step("real step", conditional, run=fixer)])
        check("re-invocation is a no-op, runner not called twice",
              s2 is None and flag["ran"] == 1, True, f"ran={flag['ran']} ({d2})")

        # Order: the FIRST unsatisfied step is the one that runs, not any later one.
        flag.clear(); flag["ran"] = 0
        order = []
        def mark(name):
            def _r(_c):
                order.append(name); flag["fixed"] = True
            return _r
        advance(ctx, [Step("first", conditional, run=mark("first")),
                      Step("second", conditional, run=mark("second"))])
        check("the FIRST unsatisfied step runs", order == ["first"], True, f"ran {order}")

    print(f"\n{'ALL VERIFIES BEHAVED' if not fails else str(fails) + ' MISBEHAVED'}")
    return 0 if not fails else 1


def _drive(ctx: Ctx, a, rd: Path, spec, oracle, kw: dict, say_ao, board_of) -> int:
    """Take steps back to back until something requires a person. Returns the exit code.

    WHY A DRIVER CAN BE TRUSTED HERE. A driver catches only DECLARED failures, and the defects
    that matter tend to be undeclared. So `advance()` runs ONE step and then re-asks that step's
    verify FROM THE ARTIFACT, THROUGH THE ARTIFACT'S REAL CONSUMER; a step that ran and still fails
    is a HALT, not a retry. This driver removes the typing between steps. It does not remove the
    checking: it stops dead at the first HALT or the first question, and it prints WHY.

    Concretely it stops on:
      * HALT           — exit 1. Something disagrees with the frozen plan.
      * Blocked        — exit 2. The AO is asking the operator a question.
      * operator gate  — exit 3, its own code, for a Step marked operator_gate (the epoch program
                         marks none): "a human must act now" is neither a defect nor a question.
      * epoch complete — roll to `--until-epoch`, or exit 0.
      * --max-steps    — exit 4. A ceiling that is REPORTED rather than silent; an unbounded
                         `while True` around a paid dispatch is how a bug becomes a bill.

    THE STEP COUNTER IS NOT A PROGRESS BAR. Two invocations can take the same step count and do
    entirely different work, so every step taken is printed with its name as it happens.
    """
    # `first_epoch` is captured NOW because ctx is REBUILT on every epoch roll below, so
    # `ctx.epoch` stops meaning "where this invocation started" the moment it advances. Without
    # it the closing line read "epochs 11..11 complete in 3 step(s)" after driving 9, 10 and 11 —
    # a summary that contradicts the three epoch banners printed directly above it.
    steps_taken, epoch_now, rc = 0, ctx.epoch, 0
    first_epoch = ctx.epoch
    last_epoch = max(a.until_epoch or ctx.epoch, ctx.epoch)
    # RUNTIME pipeline_val: val legs are polled at every step boundary and checked when they
    # end, never awaited before the end; see the PIPELINED VAL section. Off = this loop byte for byte.
    pipe = _pipeline_on(ctx)
    print(f"\nDRIVE — epoch {epoch_now}"
          + (f" through {last_epoch}" if last_epoch != epoch_now else "")
          + f", ceiling {a.max_steps} step(s). Stops at the first HALT, question, or "
            f"operator gate.\n")
    skip_e = _skip_train_epoch(ctx) if pipe else None
    if skip_e is not None and last_epoch > skip_e:
        msg = (f"RUNTIME skip_last_train: epoch {skip_e} (spec n_epochs) launches no train leg, so no epoch "
               f"after it can be built, and --until-epoch {last_epoch} goes past it. Drive to --until-epoch "
               f"{skip_e}, or turn skip_last_train off.")
        print(f"\nHALT before any step\n  {msg}\n  The driver stops here. This is not a retry condition.")
        say_ao(f"HALT\n\n{msg}", f"halt:{epoch_now}:0:skip_last_train:{last_epoch}", c=ctx)
        return 1
    if pipe:
        print("PIPELINE (RUNTIME pipeline_val): each epoch awaits its TRAIN leg only; val legs are checked "
              "when they end, a failed or dead val leg is recorded and never halts evolution, and the end "
              "awaits every val leg (each bounded by its own leg_timeout_s)."
              + (f" skip_last_train: epoch {skip_e} launches ONLY its val leg." if skip_e is not None else "")
              + "\n")
    while True:
        if steps_taken >= a.max_steps:
            msg = (f"STOPPED at the --max-steps ceiling ({a.max_steps}) with epoch {epoch_now} "
                   f"still unfinished. Nothing is wrong; raise the ceiling and re-run.")
            print(f"\n{msg}")
            say_ao(msg, f"drive-ceiling:{epoch_now}:{steps_taken}", c=ctx)
            return 4
        try:
            if pipe:
                ctx._pipeline_say = say_ao
                _pipeline_poll(ctx, say=say_ao)     # the step boundary: check every val leg that ended
            board = board_of(ctx)
            step, detail = advance(ctx)
        except Blocked as e:
            apth = answers_path(ctx.run_dir, ctx.epoch)
            try: apth.parent.mkdir(parents=True, exist_ok=True)
            except OSError: pass
            print(f"\nBLOCKED after {steps_taken} step(s) — epoch {epoch_now} needs an answer.\n"
                  f"\n  step:     {e.step or '(unknown)'}\n  question: {e.question}\n")
            if e.detail:
                print(f"  what the check saw: {e.detail}\n")
            print(f"  TO ANSWER: put this in {apth}\n"
                  f'      {{"{e.qid}": "your answer"}}\n'
                  f"  then re-run the same --drive command; it resumes where it stopped.")
            say_ao(f"BLOCKED at step {e.step or '(unknown)'}\n\n{e.question}",
                   f"blocked:{epoch_now}:{e.qid}:{e.question}", c=ctx)
            return 2
        except OperatorGate as e:
            # Checked BEFORE `except Halt`, because it IS a Halt. Ordering the other way would
            # make the subclass unreachable — a silent no-op of the kind this file has been bitten
            # by before, and one that no test of the gate ITSELF would catch.
            print(f"\nYOUR TURN after {steps_taken} step(s) in epoch {epoch_now}\n  {e}\n"
                  f"  The driver has done everything it can without you. Read the artifacts,\n"
                  f"  act on them, and re-run the same --drive command to continue.")
            say_ao(f"OPERATOR GATE\n\n{e}", f"gate:{epoch_now}:{steps_taken}:{e}", c=ctx)
            return 3
        except Halt as e:
            print(f"\nHALT after {steps_taken} step(s) in epoch {epoch_now}\n  {e}\n"
                  f"  The driver stops here. This is not a retry condition.")
            say_ao(f"HALT\n\n{e}", f"halt:{epoch_now}:{steps_taken}:{e}", c=ctx)
            return 1
        if step is not None:
            steps_taken += 1
            print(f"\n[{steps_taken}] RAN + VERIFIED: {step.name}\n      {detail}")
            say_ao(f"{board}\nRAN + VERIFIED: {step.name}\n  {detail}",
                   f"report:{epoch_now}:{steps_taken}:{step.name}", c=ctx)
            continue
        # Epoch complete.
        if pipe:
            try:
                _pipeline_record_train(ctx)         # the held-out rule reads it
            except Exception as e:                  # noqa: BLE001 (Halt included: a HALT block, never a traceback)
                if not isinstance(e, Halt):
                    e = Halt(f"could not record epoch {epoch_now}'s train checks in {PIPELINE_VAL_FILE} "
                             f"({type(e).__name__}: {e}); the held-out rule reads that record")
                print(f"\nHALT after {steps_taken} step(s) in epoch {epoch_now}\n  {e}\n"
                      f"  The driver stops here. This is not a retry condition.")
                say_ao(f"HALT\n\n{e}", f"halt:{epoch_now}:{steps_taken}:{e}", c=ctx)
                return 1
        print(f"\nepoch {epoch_now} is COMPLETE — {detail}")
        say_ao(f"{board}\nepoch {epoch_now} is COMPLETE — {detail}",
               f"report:{epoch_now}:complete", c=ctx)
        if epoch_now >= last_epoch:
            val_fails = []
            if pipe:
                # THE END: the one place val blocks. Await every tracked val leg, check it, list failures.
                try:
                    val_fails = _pipeline_final(ctx, say=say_ao)
                except Halt as e:
                    print(f"\nHALT after {steps_taken} step(s), awaiting the val legs after epoch "
                          f"{epoch_now}\n  {e}\n  The driver stops here. This is not a retry condition.")
                    say_ao(f"HALT\n\n{e}", f"halt:{epoch_now}:{steps_taken}:final:{e}", c=ctx)
                    return 1
            # STATEFUL SESSION: the run's one evolver container is
            # torn down at DRIVE DONE (evolver_session.py finish, verified + ledgered, then the last epoch's
            # receipt re-checked with the teardown record). A HALT here is a re-run point: the container survives
            # it, and `--drive` again (or `finish` by hand) completes the teardown. Per-epoch runs never reach it.
            try:
                if _evolver_session(ctx) == "stateful":
                    from src.orchestrator import stateful_evolver as _se
                    _se.run_session_finish(ctx)
            except Halt as e:
                print(f"\nHALT after {steps_taken} step(s), tearing down the evolver session after epoch "
                      f"{epoch_now}\n  {e}\n  The driver stops here. This is not a retry condition.")
                say_ao(f"HALT\n\n{e}", f"halt:{epoch_now}:{steps_taken}:finish:{e}", c=ctx)
                return 1
            print(f"\nDRIVE DONE — epochs {first_epoch}..{epoch_now} complete in "
                  f"{steps_taken} step(s)."
                  + ("" if not pipe else
                     " Every val leg is checked; no val failure." if not val_fails else
                     " VAL FAILURES (recorded; the held-out program skips these epochs): "
                     + "; ".join(f"e{e} '{s}': {d}" for e, s, d in val_fails)))
            if pipe:
                say_ao(f"DRIVE DONE — epochs {first_epoch}..{epoch_now}; val failures: "
                       f"{val_fails or 'none'}", f"report:{epoch_now}:drive-done:{len(val_fails)}", c=ctx)
            return rc
        epoch_now += 1
        # A FRESH Ctx PER EPOCH, never a mutated one. Ctx carries per-epoch caches and the
        # `_asking_step` cursor, and `_assert_epoch_consistent` exists precisely because a
        # stale epoch on a live context silently reads the previous epoch's answer file. Rebuilding
        # is cheap and removes the whole class.
        ctx = Ctx(rd, epoch_now, spec, oracle, **kw)
        print(f"\n{'=' * 76}\nEPOCH {epoch_now}\n{'=' * 76}")
    # unreachable


def _derive_restricted_cli(a) -> int:
    """`--derive-restricted-leg LEG --split F [--key test] --out NEW`. Offline: reads LEG and F, writes ONE
    new file (refuses to overwrite), prints its figures."""
    if not a.split or not a.out:
        print("--derive-restricted-leg needs --split (the run's declared split) and --out (a new file)")
        return 64
    out = Path(a.out)
    if out.exists():
        print(f"REFUSING: {out} exists; a derived record is written once. Move it aside (keep it).")
        return 1
    try:
        rec = derive_restricted_leg(a.derive_restricted_leg, a.split, a.key)
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as e:
        print(f"REFUSING: {type(e).__name__}: {e}")
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, indent=1))
    df = rec["derived_from"]
    print(f"wrote {out} (sha256 {hashlib.sha256(out.read_bytes()).hexdigest()})\n"
          f"  {rec['n']} tasks of split[{a.key!r}] sha256:{rec['split_sha256'][:12]}: "
          f"{rec['sample_pass']}/{rec['sample_total']} samples passed, pass@N {rec['pass3']}, all@N "
          f"{rec['all3']}, turn_exhausted {rec.get('turn_exhausted', 'ABSENT')}\n"
          f"  from {df['source']} ({df['source_counts']['sample_pass']}/{df['source_counts']['sample_total']});"
          f" removed {len(df['removed_task_ids'])} tasks / {df['removed_samples']} samples holding "
          f"{df['removed_sample_passes']} passes")
    return 0


def _export_run_env(run_dir, benchmark: str) -> None:
    """Route the run's declared credential and data root through the environment, for this process and every
    child it starts (leg launchers, evolver drivers, checks), so nothing reads a path built into the code:
      * INFRA claude_code.token_file -> EVOSKILL_CC_TOKEN_FILE (src/orchestrator/cc_account.py; a null token file
        leaves ANTHROPIC_API_KEY as the run's one credential);
      * INFRA data.root -> EVOSKILL_SSBV2_DATA on an SSB run, when neither it nor SSBV2_ROOT is set
        (daytona_evolver.data_root()).
    A value already in the environment is never overwritten: infra.agree() names one that disagrees with INFRA."""
    try:
        inf = json.loads((Path(run_dir) / "INFRA.json").read_text())
    except (OSError, ValueError):
        return
    if not isinstance(inf, dict):
        return
    from .cc_account import TOKEN_FILE_ENV
    tf = (inf.get("claude_code") or {}).get("token_file")
    if isinstance(tf, str) and tf.strip():
        os.environ.setdefault(TOKEN_FILE_ENV, tf.strip())
    root = (inf.get("data") or {}).get("root")
    if benchmark == "ssbv2" and isinstance(root, str) and root.strip() \
            and not os.environ.get("EVOSKILL_SSBV2_DATA") and not os.environ.get("SSBV2_ROOT"):
        os.environ["EVOSKILL_SSBV2_DATA"] = root.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir")
    ap.add_argument("--epoch", type=int)
    ap.add_argument("--oracle", default="")
    ap.add_argument("--plan", action="store_true",
                    help="report every verify and change nothing; exit 0 even on failures")
    ap.add_argument("--step", action="store_true",
                    help="EXECUTE the first unsatisfied step, then re-verify it")
    ap.add_argument("--selftest", action="store_true")
    # Runtime plumbing. Deliberately CLI/config and not ExperimentSpec fields: the spec's hash is
    # the cache tag, and which launcher drives a rollout cannot change what an epoch measures.
    ap.add_argument("--leg-script", default="")
    ap.add_argument("--shared-dir", default="", help="where split/oracle/base live, if not run-dir")
    ap.add_argument("--base-leg", default="", help="no-skills comparator train leg (epoch 1)")
    ap.add_argument("--cache-tag", default="", help="pinned cache tag; overrides the spec's")
    ap.add_argument("--evolver-log", default="", help="evolver log to audit, if not the default")
    ap.add_argument("--digest-cache", action="append", default=[],
                    help="per-sample cache dir the digest builder reads; repeatable")
    ap.add_argument("--leg-timeout-s", type=int, default=0)
    # ── the operator's side of the conversation ───────────────────────────────────────
    ap.add_argument("--note", default="",
                    help="what you are asking for and why; recorded as YOUR turn in the run's "
                         "operator<->AO transcript")
    # ── driving ──────────────────────────────────────────────────────────────────────────────
    ap.add_argument("--drive", action="store_true",
                    help="keep taking steps until the epoch completes, HALTs, BLOCKs, or an "
                         "operator gate is reached")
    ap.add_argument("--until-epoch", type=int, default=0,
                    help="with --drive: on completing an epoch, roll to the next, up to this one")
    ap.add_argument("--max-steps", type=int, default=200,
                    help="with --drive: hard ceiling on steps taken in one invocation")
    # ── the held-out test ───────────────────────────────────────────────────────────
    ap.add_argument("--heldout", action="store_true",
                    help="the HELD-OUT program instead of an epoch: the last successful epoch's skill set "
                         "on the split's test key vs RUNTIME heldout_base_leg, one task set on both "
                         "sides. --epoch defaults to the rule's epoch; with --plan or --step as usual")
    ap.add_argument("--derive-restricted-leg", default="", metavar="LEG_JSON",
                    help="OFFLINE: write LEG_JSON restricted to --split[--key] (with a derived_from block) "
                         "to --out; no run dir needed")
    ap.add_argument("--split", default="", help="with --derive-restricted-leg: the split file")
    ap.add_argument("--key", default=HELDOUT_KEY, help="with --derive-restricted-leg: the split key")
    ap.add_argument("--out", default="", help="with --derive-restricted-leg: the record to write (new)")
    # ── pipelined val (RUNTIME pipeline_val) ─────────────────────────────────────────────
    ap.add_argument("--val-check", action="store_true",
                    help="RUNTIME pipeline_val only: the VAL leg of --epoch instead of an epoch. --plan: its "
                         "checks, read-only; --step: its checks with a failing check's runner (the "
                         "re-grade, once), recorded in <run>.operator/PIPELINE_VAL.json; a val leg with no "
                         "result and no live process is (re)launched")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.derive_restricted_leg:
        return _derive_restricted_cli(a)
    # EXIT 2 MEANS BLOCKED AND NOTHING ELSE, so a caller can tell "answer me" from "go fix
    # something" without parsing prose. A wrapper keying on 2 must never see a corrupt config as a
    # question waiting for the user: usage errors use EX_USAGE (64); anything that prints HALT
    # exits 1, like every other HALT.
    if not a.run_dir or (a.epoch is None and not a.heldout):
        ap.print_help()
        return 64
    if a.heldout and (a.drive or a.until_epoch):
        print("--heldout takes ONE step per --step invocation; --drive/--until-epoch drive epochs.")
        return 64
    if a.val_check and (a.drive or a.until_epoch or a.heldout or a.epoch is None):
        print("--val-check checks the val leg of ONE --epoch (with --plan or --step); it does not drive.")
        return 64
    from src.orchestrator.spec import ExperimentSpec, SpecError
    rd = Path(a.run_dir)
    try:
        ExperimentSpec.assert_unmutated(rd)
        spec, _ = ExperimentSpec.load(rd)
    except SpecError as e:
        print(f"HALT  frozen spec: {e}")
        return 1
    _export_run_env(rd, getattr(spec, "benchmark", "") or "")

    # RUNTIME.json supplies the same plumbing without a wall of flags on every invocation.
    # Explicit flags win, so a one-off override never requires editing the file.
    rt = {}
    rt_file = rd / "RUNTIME.json"
    if rt_file.is_file():
        try:
            rt = json.loads(rt_file.read_text())
        except json.JSONDecodeError as e:
            print(f"HALT  {rt_file.name} is not readable JSON: {e}")
            return 1                                   # prints HALT, so exits like a HALT

    oracle = Path(a.oracle or rt.get("oracle") or (rd / "oracle"))
    kw = {}
    if a.leg_script or rt.get("leg_script"):
        kw["leg_script"] = a.leg_script or rt["leg_script"]
    if a.shared_dir or rt.get("shared_dir"):
        kw["shared_dir"] = a.shared_dir or rt["shared_dir"]
    if rt.get("base_leg_val"):                       # declared, see Ctx.base_leg_val
        kw["base_leg_val"] = rt["base_leg_val"]
    # The AO's prompt-assembly how-to (prompt_mode "ao"), supplied ONCE at init.
    if rt.get("prompt_policy"):
        kw["prompt_policy"] = rt["prompt_policy"]
    if rt.get("prompt_author_model"):
        kw["prompt_author_model"] = rt["prompt_author_model"]
    # ONE AO CONVERSATION PER RUN, opt-in. Off by default because turning it on changes what the
    # AO is (one conversation instead of single-shot authorings with no memory of each other). A
    # completed run must stay reproducible, so this is never retro-applied — a NEW run declares it.
    if rt.get("ao_continuous_session"):
        kw["ao_continuous_session"] = bool(rt["ao_continuous_session"])
    if rt.get("digest_bucket"):
        kw["digest_bucket"] = rt["digest_bucket"]
    # SSB v2 epoch-1 snapshot roots. Absent keeps the single derived root.
    if rt.get("digest_snapshot_roots"):
        _roots = rt["digest_snapshot_roots"]
        kw["digest_snapshot_roots"] = ((str(_roots),) if isinstance(_roots, str)
                                       else tuple(str(x) for x in _roots))
    # evolver_sandbox defaults ON, so the key is read when PRESENT (including an explicit false) rather
    # than only when truthy — `if rt.get(...)` could never turn it off.
    if "evolver_sandbox" in rt:
        kw["evolver_sandbox"] = bool(rt["evolver_sandbox"])
    # All-epochs evidence. PRESENCE semantics like evolver_sandbox, and the value is
    # passed through UNCOERCED so a typo or a boolean is refused below rather than read as either
    # mode. Absent keeps "current_epoch", i.e. every existing run is unchanged.
    if "evolver_evidence" in rt:
        kw["evolver_evidence"] = rt["evolver_evidence"]
    # The rollout evidence (Ctx.evidence_layout). Presence semantics, value UNCOERCED so a
    # typo is refused by _evidence_layout below. Absent keeps "digests": every existing run unchanged.
    if "evolver_evidence_layout" in rt:
        kw["evidence_layout"] = rt["evolver_evidence_layout"]
    # The stateful evolver session (Ctx.evolver_session). Presence semantics, value UNCOERCED so
    # a typo is refused by _evolver_session below. Absent keeps "per_epoch": every existing run unchanged.
    if "evolver_session" in rt:
        kw["evolver_session"] = rt["evolver_session"]
    # The pinned cache tag. Ctx.cache_tag existed and drive_a2.py passed it, but main() never
    # read it from RUNTIME.json — so the CLI path silently fell back to spec.cache_tag and the
    # first epoch-1 leg wrote into a DIFFERENT bucket than the pinned base. An added parameter
    # that only one of two callers passes is indistinguishable from a parameter that works.
    if a.cache_tag or rt.get("cache_tag"):
        kw["cache_tag"] = a.cache_tag or rt["cache_tag"]
    if a.base_leg or rt.get("base_leg"):
        kw["base_leg"] = a.base_leg or rt["base_leg"]
    if rt.get("heldout_base_leg"):                  # declared, see Ctx.heldout_base_leg
        kw["heldout_base_leg"] = rt["heldout_base_leg"]
    if a.evolver_log or rt.get("evolver_log"):
        kw["evolver_log"] = a.evolver_log or rt["evolver_log"]
    caches = a.digest_cache or rt.get("digest_caches") or []
    if caches:
        kw["digest_caches"] = tuple(Path(c) for c in caches)
    if a.leg_timeout_s or rt.get("leg_timeout_s"):
        kw["leg_timeout_s"] = a.leg_timeout_s or rt["leg_timeout_s"]
    # Declared solver provider pins (see Ctx.solver_env).
    if rt.get("solver_env"):
        if not isinstance(rt["solver_env"], dict):
            print(f"HALT  RUNTIME.json solver_env must be an object, got "
                  f"{type(rt['solver_env']).__name__}")
            return 1
        kw["solver_env"] = {str(k): str(v) for k, v in rt["solver_env"].items()}
    # The evolver host (Ctx.evolver_host). Presence semantics, value UNCOERCED so a
    # typo is refused by _evolver_host below. evolver_timeout_s was listed for the AO's runtime
    # view but never read here; it matters now that a Daytona epoch has a declared wall.
    if "evolver_host" in rt:
        kw["evolver_host"] = rt["evolver_host"]
    if "evolver_daytona" in rt:
        if not isinstance(rt["evolver_daytona"], dict):
            print("HALT  RUNTIME.json evolver_daytona must be an object")
            return 1
        kw["evolver_daytona"] = dict(rt["evolver_daytona"])
    if rt.get("evolver_timeout_s") and rt.get("evolver_host") == "daytona":
        # daytona only: a local-host run keeps the 5400 s default it has always run under.
        kw["evolver_timeout_s"] = int(rt["evolver_timeout_s"])
    # Who writes the evolver prompt. Presence semantics, value UNCOERCED so a typo
    # is refused by _prompt_mode below. Absent keeps "ao", i.e. every existing run is unchanged.
    if "prompt_mode" in rt:
        kw["prompt_mode"] = rt["prompt_mode"]
    # The pipelined val order (Ctx.pipeline_val), skip_last_train (Ctx.skip_last_train, valid
    # only with pipeline_val) and the leg pool (Ctx.leg_pool). Presence semantics; a non-boolean is refused
    # so a typo ("true" as a string) cannot silently mean either order. Absent = the serial order, except that
    # skip_last_train absent with pipeline_val true is true.
    try:
        kw.update(_pipeline_runtime(rt))
    except ValueError as e:
        print(f"HALT  {e}")
        return 1
    # The evolver's grants live in evolver_daytona (train_inputs / train_references / grader_clean), where
    # the Daytona driver and check_upload_manifest read them. The same names declared at the top
    # level would be read by NOTHING, so a grant the operator meant would silently be off.
    _misplaced = [k for k in ("evolver_train_inputs", "evolver_train_references",
                              "evolver_grader_clean", "train_inputs", "train_references",
                              "grader_clean") if k in rt]
    if _misplaced:
        print(f"HALT  RUNTIME.json declares {_misplaced} at the top level; the evolver's grants are "
              f"read ONLY from evolver_daytona.train_inputs / .train_references / .grader_clean. "
              f"Move them there.")
        return 1
    epoch = a.epoch
    if a.heldout and epoch is None:
        # The rule's epoch, re-derived; the first held-out step re-derives it again and must agree.
        # pipeline_val: an epoch whose val leg is missing, unchecked or failed does not qualify.
        _gate = (_pipeline_heldout_gate(Ctx(rd, 1, spec, oracle, **kw)) if kw.get("pipeline_val")
                 else None)
        try:
            epoch, _why = last_successful_epoch(rd, spec.n_epochs, spec.split_path, spec.n_samples,
                                                val_gate=_gate,
                                                skip_train_epoch=_skip_train_epoch(Ctx(rd, 1, spec, oracle, **kw)))
        except Halt as e:
            print(f"HALT  {e}")
            return 1
        if epoch is None:
            print("HALT  no epoch qualifies under the held-out rule (promoted skill set + train and val "
                  "legs on the split): " + "; ".join(f"epoch {k}: {w}" for k, w in _why))
            return 1
    ctx = Ctx(rd, epoch, spec, oracle, **kw)
    steps = heldout_steps(ctx) if a.heldout else None
    # A bad evidence declaration is a config error like an unreadable RUNTIME.json: refuse before
    # anything is planned against it. (Every verify and runner that depends on it checks again.)
    try:
        _evidence_mode(ctx)
        _evidence_layout(ctx)
        _evolver_host(ctx)
        _prompt_mode(ctx)
        _evolver_session(ctx)
    except Halt as e:
        print(f"HALT  {e}")
        return 1

    if a.step and a.plan:
        print("--step and --plan are contradictory: one changes the world, the other promises "
              "not to. Pick one.")
        return 64
    if a.drive and not a.step:
        print("--drive is a way of taking STEPS; pass --step as well, so that a command that "
              "changes the world always says so.")
        return 64
    if a.until_epoch and not a.drive:
        print("--until-epoch only means something with --drive.")
        return 64
    if a.val_check:
        # Before the operator-turn record below: a background --val-check is started by the drive, not
        # by the operator; one run by hand records its --note (if any) as the operator's turn itself.
        if not _pipeline_on(ctx):
            print("--val-check belongs to RUNTIME pipeline_val=true; this run declares the serial order, "
                  "where the val leg's checks are ordinary epoch steps.")
            return 64
        try:
            return _val_check_program(ctx, step=a.step, note=a.note)
        except Halt as e:
            print(f"\nHALT\n  {e}")
            return 1

    # THE OPERATOR'S TURN, recorded BEFORE anything runs (a record written only on success is not a
    # record) and BEFORE the readonly flag is set, because --plan is also a thing the operator said
    # and a transcript must not omit the operator's questions.
    # Deduped on (epoch, argv, note): re-running an identical command is one instruction repeated,
    # not two.
    _argv = " ".join(sys.argv[1:])
    if a.note or a.step:
        _conv_once(ctx, f"invocation:{a.epoch}:{_argv}:{a.note}", speaker="operator",
                   channel="invocation",
                   content=(a.note or "(no note given)") + f"\n\n$ python3 -m "
                           f"src.orchestrator.loop {_argv}",
                   meta={"argv": _argv, "mode": "drive" if a.drive
                         else "step" if a.step else "plan"})

    def _say_ao(text: str, key: str, *, c=None) -> None:
        """Record the AO's reply to this invocation."""
        _conv_once(c or ctx, key, speaker="ao", channel="report", content=text)

    def _board(c: Ctx) -> str:
        """report(), captured verbatim so the transcript and the terminal cannot disagree."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            results, _ = evaluate(c, steps)
            report(c, results)
        out = buf.getvalue()
        sys.stdout.write(out)
        return out

    # EVERY path that can reach a verify is inside this guard, including the two evaluate()
    # calls below.
    #
    # A Blocked raised by a verify outside this guard would reach the operator as a raw
    # `src.orchestrator.loop.Blocked` traceback with exit 1, never the operator instructions, never
    # exit 2 -- and a wrapper keying on the exit code would read "answer me" as "go fix something",
    # the precise confusion Blocked was built to remove.
    try:
        if not a.step:
            # `--plan` says it "changes nothing"; make that mechanical, not a promise. ask() writes
            # BLOCKED_e{N}.json, so without this a plan would leave files behind.
            ctx.readonly = True
            results, _ = evaluate(ctx, steps)
            report(ctx, results)
            if _pipeline_on(ctx) and not a.heldout:
                print(_pipeline_plan_text(ctx))     # the order, the val legs and when their checks run
            return 0

        if a.drive:
            return _drive(ctx, a, rd, spec, oracle, kw, _say_ao, _board)

        # --step CHANGES THE WORLD, so say what is already true before touching anything. An
        # operator who sees the same board twice knows the step was a no-op.
        if _pipeline_on(ctx) and not a.heldout:
            _pipeline_poll(ctx, say=_say_ao)          # a step boundary: check every val leg that ended
        board = _board(ctx)
        step, detail = advance(ctx, steps)
        if step is None and _pipeline_on(ctx) and not a.heldout:
            _pipeline_record_train(ctx)
    except Blocked as e:
        # A BLOCK is not a defect report. Exit 2, distinct from HALT's 1, so a caller can tell
        # "answer me" from "go fix something" without parsing prose.
        ap = answers_path(ctx.run_dir, ctx.epoch)
        # Create the operator directory so the printed instruction is literally followable.
        # It sits outside the run dir; without the mkdir the user's first action after
        # reading "put this in <path>" is a No such file or directory.
        try: ap.parent.mkdir(parents=True, exist_ok=True)
        except OSError: pass
        print(f"\nBLOCKED — the run needs an answer from you, and will not proceed without one.\n"
              f"\n  step:     {e.step or '(unknown)'}\n"
              f"  question: {e.question}\n")
        if e.detail:
            print(f"  what the check saw: {e.detail}\n")
        print(f"  TO ANSWER: put this in {ap}\n"
              f'      {{"{e.qid}": "your answer"}}\n'
              f"  then re-run the same command. The outstanding question is also recorded in\n"
              f"  {blocked_path(ctx.run_dir, ctx.epoch).name}, so it survives this process.\n"
              f"\n  It BLOCKS rather than assuming, by design: every artifact downstream of an\n"
              f"  unanswered question would inherit an assumption nobody confirmed.")
        _say_ao(f"BLOCKED at step {e.step or '(unknown)'}\n\n{e.question}",
                f"blocked:{a.epoch}:{e.qid}:{e.question}")
        return 2
    except Halt as e:
        print(f"\nHALT\n  {e}\n"
              f"  This is not a retry condition. The world disagrees with the frozen plan;\n"
              f"  a person decides whether to fix the step or restart the epoch.")
        _say_ao(f"HALT\n\n{e}", f"halt:{a.epoch}:{_argv}:{e}")
        return 1
    if step is None:
        print(f"\nepoch {a.epoch} is COMPLETE — {detail}")
        _say_ao(f"{board}\nepoch {a.epoch} is COMPLETE — {detail}",
                f"report:{a.epoch}:{_argv}:complete")
        return 0
    print(f"\nRAN + VERIFIED: {step.name}\n  {detail}\n"
          f"  One step per invocation, by design. Run again for the next.")
    _say_ao(f"{board}\nRAN + VERIFIED: {step.name}\n  {detail}",
            f"report:{a.epoch}:{_argv}:{step.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
