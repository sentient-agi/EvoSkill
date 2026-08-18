#!/usr/bin/env python3
"""Layer 3 of the orchestrator — the DETERMINISTIC LOOP.

WHAT THIS IS, AND THE ONE THING IT MUST NOT BE

It executes a frozen ExperimentSpec: base leg, then N epochs of
(build evidence -> evolve -> gate the artifact -> promote -> roll out -> gate the result).
It decides NOTHING. Every threshold, metric and stopping rule comes from the spec, which
was frozen and hashed before any leg ran. The loop's job is to do the same thing every
time and to refuse when reality disagrees with the plan.

The thing it must not be is a happy-path DAG driver. On 2026-08-08 four defects hit one
evolution arm and EVERY ONE reported success:

    #214  evolver "wrote 5 skills"   -> the loader read 1; four bodies silently discarded
    #213  evolver run "succeeded"    -> it had read the comparator arm's answers
    --    process "launched"         -> dead in 200ms on `400 No models provided`
    --    split "built, 45 tasks"    -> 36 of 45 ids did not exist in the harness's pool

A driver that advances on each step's own report ships all four. So every step here has a
`verify` that re-derives its answer from the artifact THROUGH THE ARTIFACT'S REAL CONSUMER,
and the loop advances on the verify, never on the exit code.

THREE RULES THAT COST REAL RUNS TO LEARN

1. HALT, DO NOT RETRY. A failed verify means the world is not what the plan assumed. Retrying
   an evolver whose output was contaminated just produces contaminated output again — the
   correct response to #213 was to restart the epoch from iteration 1, and to #214 to fix the
   prompt. Both are human calls. `HALT` is the only failure mode; there is no backoff.

2. ISOLATION IS FILESYSTEM-LEVEL, NOT TEXTUAL. #215: a subagent walked past the peer-skill
   regex hook by splitting a quoted literal so the word never appeared whole. Any guard that
   pattern-matches command strings is a tripwire that makes accidents loud, not a boundary.
   `EpochPaths.grants()` returns the ONLY directories an evolver may see, and the peer arms'
   skill directories are not among them. The hook stays armed as a tripwire; it is not the
   mechanism.

3. AN UNSATISFIED OPERATOR GATE IS A HALT, NOT A WARNING. The cognitive proofread of each
   evolver prompt, and of each evolved skill set, is not automatable and this module does not
   pretend otherwise. It requires a countersignature artifact naming what was checked. With no
   signature the loop stops. Silently skipping it would convert "a human read this" into "a
   human was assumed to have read this", which is how an optimizer acquires a hidden author.

WHAT THIS MODULE DOES NOT DO
  - It does not compose evolver prompts. That is layer 1 plus a human proofread; the loop
    takes a prompt path and verifies the prompt's numbers re-derive from the result JSONs
    (#208), refusing a prompt whose headline figures disagree with the data behind them.
  - It does not judge skill quality. Only infra failures halt a run; bad reasoning from sound
    evidence is the optimizer's output, and filtering on taste makes the operator a hidden
    part of the optimizer.

USAGE
    python3 -m src.orchestrator.loop --plan     --run-dir outputs/tau3_dsv4fevo
    python3 -m src.orchestrator.loop --step     --run-dir outputs/tau3_dsv4fevo
    python3 -m src.orchestrator.loop --selftest

EXIT CODES, and 2 is the one with a contract (#281):
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
SIGN_FILENAME = "PROOFREAD.json"

# THERE IS DELIBERATELY NO STATE FILE. A `STATE_FILENAME = "LOOP_STATE.json"` constant sat here
# unused, promising a progress file this loop has never written and should not write. Every
# step's position is re-derived from the artifacts on disk each time it is asked, so there is no
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

    IT INHERITS BaseException, NOT Exception (and therefore NOT Halt).
    #264, found 2026-08-11 by adversarial audit. It was a Halt, hence an Exception, hence
    swallowable by any `except Exception` between ask() and _safe_verify's re-raise. A merely
    defensive verify --
        try: a = ask(ctx, "basis", "Which basis?")
        except Exception: a = "assume DB-state"
    -- turned "I need a human answer" into a silent default AND left BLOCKED_e{N}.json listing
    the question as outstanding for a step that had already passed: the artifact and the
    behaviour disagreeing, which is the shape of every defect that has cost this programme a run.
    Three verifies in this very file already use bare `except Exception` (_v_skills_were_consulted,
    and twice in _load_delivery_assessor), so the first verify to combine the two idioms would
    have shipped it. BaseException is the standard idiom for a control-flow signal that must not
    be caught by ordinary error handling -- the same reason KeyboardInterrupt and SystemExit sit
    there. Callers that mean to handle a block must name it.

    Distinct from a plain Halt, and the distinction is the point. A Halt says "something
    disagreed with the plan" — a defect, and the operator's job is to go fix it. A Blocked says
    "I cannot tell whether this is wrong or whether you meant something I did not parse, and
    guessing would make me a hidden author of your experiment."

    THIS IDIOM ALREADY EXISTED ONE LAYER DOWN AND NEVER PROPAGATED UP. benchmark.py (L0) has
    `Finding.question` — "what to ask the user when it fails, never a correction the module
    invents, because the user may simply have described it differently than it parsed" — and
    `verify()` there "never raises on a mismatch ... because the caller's job is to ASK, and a
    traceback is not a question." L3 had no such channel: all ~19 steps could only refuse, and a
    human (me) read the refusal and turned it into a question by hand. That hand step is the one
    the AO is supposed to own.

    IT BLOCKS, by explicit decision. The alternative — queue the question
    and carry on with independent work — was considered and rejected. Rationale: every artifact
    downstream of an unanswered question inherits an assumption nobody confirmed, and this
    programme has already lost a run that way (ao3: the AO asserted corpus structure the
    measurement then contradicted, and four epochs were built on it).
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

    #268: OUTSIDE THE RUN DIRECTORY, in a sibling `<run>.operator/`. Defence in depth on top of
    the #266/#267 write containment, not a replacement for it. The sandbox now denies an agent
    write access to the run dir, and that is the primary control -- but it is an ALLOWLIST, and
    one forgotten grant re-opens the boundary silently. Putting the answer channel somewhere no
    agent job ever declares as a write target means a future grant mistake cannot reach it.

    Sibling rather than a subdirectory, because a subdirectory of the run dir inherits any
    future subpath grant on its parent -- exactly the #266 shape (the log's parent grant WAS the
    run dir). A sibling has no such parent relationship.
    """
    return operator_dir(run_dir) / f"ANSWERS_e{epoch}.json"


def blocked_path(run_dir, epoch) -> "Path":
    """The outstanding-question record. Sibling to the run dir alongside ANSWERS (#268), so the
    whole operator channel lives in one place that no agent can write."""
    return operator_dir(run_dir) / f"BLOCKED_e{epoch}.json"


# Code points that are neither whitespace by str.isspace() nor visible. An "answer" made only of
# these is not an answer -- see _is_blank.
_INVISIBLE = "​‌‍⁠﻿­᠎⠀ㅤﾠᅟᅠ"


def _is_blank(s: str) -> bool:
    """#269. `s.strip()` removes only characters where str.isspace() is True, so an answer
    consisting entirely of ZERO-WIDTH SPACE, a BOM, WORD JOINER, SOFT HYPHEN, BRAILLE BLANK or a
    Hangul filler passed ask()'s emptiness check and was returned verbatim as "a genuine non-empty
    answer". The realistic trigger is not an attack: it is a copy-paste out of a terminal or a web
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
    """#270. answers_path() resolves ctx.epoch LIVE on every ask(), while Ctx.__post_init__
    snapshots `self.paths = EpochPaths(run_dir, epoch)` ONCE. Nothing pinned the two together, so
    a verify that reassigned ctx.epoch could be answered out of another epoch's ANSWERS file while
    every other artifact path in the same run still referred to the original epoch. The per-epoch
    answer firewall this module promises ("an answer cannot silently carry across an epoch
    boundary") was enforced only against a value nothing held still.
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

      * ABSENT answer  -> Blocked. NOT a default, NOT a skip, NOT a pass. This is the #257 rule
        ("that fallback IS the defect") applied to human input: a step that proceeds on an
        unanswered question has invented the answer.
      * PRESENT but empty/whitespace -> Blocked, with the same question. An empty string is not
        an answer, and treating it as one is how "the user declined to say" becomes "the user
        said nothing matters".
      * PRESENT and non-empty -> returned verbatim, never parsed or coerced here. The caller
        decides what it means, because only the caller knows what it asked.
      * PRESENT, non-empty, but given for a DIFFERENT question under this qid -> Blocked (#271).

    The question is also written to BLOCKED_e{N}.json so it survives the process that asked it.
    #244's lesson: a record written only on success is not a record.
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
            # never learn the file had been unreadable the whole time.
            #
            # #272: this branch used to raise WITHOUT recording, contradicting the contract three
            # paragraphs up ("the question is also written to BLOCKED_e{N}.json so it survives the
            # process that asked it"). The one block that leaves no trace was the one whose cause
            # is an operator typo -- i.e. the one most likely to be hit repeatedly.
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
            # #271. THE ANSWER IS BOUND TO THE QUESTION, NOT ONLY TO THE KEY. Until now the check
            # was `.get(qid)` and nothing compared the stored answer against the question text it
            # was given for, so a step asking a DIFFERENT question under the same qid was silently
            # satisfied by the earlier answer -- "a step that proceeds on an unanswered question
            # has invented the answer", arrived at by a different road. _v_operator_signature
            # already hashes the artifact it signs for exactly this reason: "a signature cannot
            # silently outlive an edit to the thing it signed".
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
            # (#273: nothing ever marked a question answered, so BLOCKED_e{N}.json listed every
            # question ever asked as still open, with live how_to_answer instructions).
            _try_record(ctx, qid, question, answered=True)
            # #301, and this is the exchange the transcript most needed: ANSWERS_e{N}.json is a
            # dict, so it records the answer but destroys the ORDER and the pairing — re-reading
            # it later cannot tell you which question came first or which answer unblocked what.
            # Recorded on the CONSUMING read (here), not on the write, because that is the moment
            # the answer actually enters the run. `_conv_once` keeps a re-run from duplicating it.
            _conv_once(ctx, f"answer:{qid}:{question}", speaker="operator", channel="answer",
                       content=got, meta={"qid": qid, "question": question})
            return got
    # #265: RECORD, THEN BLOCK -- and block even if the record could not be written. This call
    # used to be unguarded, so a read-only run dir / full disk / a directory sitting where the
    # file belongs raised OSError INSTEAD of Blocked; _safe_verify turned that into an ordinary
    # failed verify with question=="", and advance() then fell through and EXECUTED the step's
    # runner -- dispatching a paid evolver, or promoting skills, on a question no human ever saw.
    # Losing the audit record is tolerable; proceeding past an unanswered question is not.
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
    """_record_block, but a write failure is a NOTE rather than an exception. See #265."""
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
    """Questions still awaiting an answer. #273: an ANSWERED entry is kept (it carries the
    question<->answer binding #271 needs) but is no longer outstanding."""
    return {k: v for k, v in _read_blocked(ctx).items()
            if isinstance(v, dict) and not v.get("answered")}


def _record_block(ctx, qid: str, question: str, *, answered: bool = False) -> None:
    """Append the outstanding question to BLOCKED_e{N}.json. Idempotent on qid.

    READ-ONLY CONTEXTS DO NOT WRITE. `--plan` promises to "report every verify and change
    nothing", and once a verify can ask a question, planning would otherwise leave files behind.

    A CORRUPT RECORD IS PRESERVED, NEVER DISCARDED (#274). The read used to sit inside a bare
    `except Exception: cur = {}` followed by an unconditional overwrite, so a truncated file --
    e.g. from a process killed mid-write, which is the exact durability scenario this file exists
    for -- was replaced by a record holding only the newest question, and every other outstanding
    question vanished with no error. That is the opposite of the #244 property the file is for.
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
    # ATOMIC (#290, same lesson as AO_SESSION.json): a plain write_text lets a concurrent reader
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
    def prompt(self) -> Path:
        return self.run_dir / f"ITER{self.epoch}_PROMPT.md"

    def leg(self, key: str) -> Path:
        return self.run_dir / f"{self.run_dir.name}_v{self.epoch}_{key}.json"

    def grants(self, oracle: Path) -> list[Path]:
        """The COMPLETE set of directories the evolver may see. #215: this list, not a regex
        over command strings, is the isolation boundary. Anything absent here is absent from
        the filesystem the evolver runs against — in particular every peer arm's skills_v*,
        which live under a different run dir and are therefore simply not granted.

        Note what is NOT here: the run dir itself. Granting the parent would hand over every
        sibling epoch's promoted skills and every peer arm reachable by `cd ..`, which is
        exactly how #213 happened — the digests grant sat inside the comparator's run dir, so
        its answers were one `cd ..` from a legitimate path.
        """
        return [self.digests, self.staging, oracle]


# ── steps ────────────────────────────────────────────────────────────────────────────────
@dataclass
class Step:
    """One unit of work. `run` may be None for steps a human or another process performs;
    the loop then only verifies. `verify` MUST re-derive from the artifact, never read a
    status field, and must be able to fail on a reachable input."""
    name: str
    verify: object                      # callable(ctx) -> (ok: bool, detail: str)
    run: object = None                  # callable(ctx) -> None
    defect: str = ""                    # the historical defect this step's verify catches
    operator_gate: bool = False         # requires a countersignature; never auto-satisfied


@dataclass
class Ctx:
    run_dir: Path
    epoch: int
    spec: object                        # ExperimentSpec
    oracle: Path
    # The per-sample cache dirs the digest builder reads. Explicit rather than derived: the
    # dir name encodes domain, variant, cache tag, leg token and sample index, and rebuilding
    # that string here would be a second source of truth for the one thing #148/#154 showed
    # must have exactly one (run_tau3_banking.py:772 owns it).
    digest_caches: tuple = ()
    # Runtime plumbing, NOT experiment identity — deliberately on Ctx and not on the frozen
    # ExperimentSpec, whose hash defines the cache tag. Which launcher script drives the
    # rollouts, and how long to wait for one, cannot change what an epoch measures.
    leg_script: str = "scripts/tau3_leg_b2525.sh"
    shared_dir: str = ""            # split/oracle/base live here; "" means same as run_dir
    # The no-skills comparator. Declared, not derived: base legs predate the loop's naming
    # convention (`retail_base_train.json`), and guessing a name is what made the epoch-1 prompt
    # check vacuous. "" means undeclared, and the epoch-1 prompt verify FAILS rather than passes.
    base_leg: str = ""
    # #251's second half. The VAL comparator, declared for the same reason base_leg is: my first
    # cut of _v_legs_comparable DERIVED it by substituting "_base_" for "_vN_" in the skill leg's
    # filename. That happens to resolve inside a self-contained run dir (the smoke), and does NOT
    # resolve when the base legs live in another run's directory (tau3_banking_ao2 reuses
    # tau3_banking_2525/e0_base_*.json). In that case the code fell back to base_leg — the TRAIN
    # base — and would have compared the VAL skill leg against it, tripping my own
    # "different task sets" FAIL on a perfectly good run. A guard that fails spuriously gets
    # disabled, so it is worse than none. Declared, never guessed; "" means val pairing is
    # reported UNMEASURED rather than checked against the wrong file.
    base_leg_val: str = ""
    # #253. THE AO AUTHORS THE EVOLVER PROMPT, and this is the how-to it is initialised with.
    # Supplied ONCE at init (RUNTIME.json "prompt_policy"), never per epoch — that is the point:
    # a policy is a config, so prompt assembly becomes an ablatable variable instead of a thing a
    # human re-tunes each epoch. "" falls back to src/orchestrator/PROMPT_POLICY.md.
    # Before this, the operator both WROTE and COUNTERSIGNED the prompt, so half the operator gate
    # was self-review; now the gate's prompt half is adversarial like its skills half already was.
    prompt_policy: str = ""
    prompt_author_model: str = "opus[1m]"
    ao_continuous_session: bool = False        # one AO conversation per run; see RUNTIME wiring
    # An arm already in flight was launched with a hand-chosen cache tag; the spec-derived tag
    # would send later epochs to a DIFFERENT bucket than epoch 1 wrote. Safe (a distinct tag
    # never replays stale records, #182/#183) but it silently splits one arm across two cache
    # lineages, and the next epoch's digest builder would look in the wrong place. "" = use the
    # spec's derived tag, which is correct for any run the loop starts itself.
    cache_tag: str = ""
    # `--plan` promises to "report every verify and change nothing". Before questions existed that
    # was free: every verify was a pure read. ask() writes BLOCKED_e{N}.json, so planning would
    # now leave files behind unless the read-only promise is made mechanical rather than assumed.
    readonly: bool = False
    # #294. Wrap the evolver's `claude` child in sandbox-exec. DEFAULT ON,
    # with `"evolver_sandbox": false` in RUNTIME.json as the recorded opt-out.
    #
    # IT SHIPPED OPT-IN FOR HALF A DAY AND THAT WAS THE WRONG CALL. the maintainer's question settled it:
    # "if it's not the default, then what is?" -- the answer was UNCONTAINED, which is precisely
    # the state I had just called the largest remaining gap. It was also inconsistent with the
    # AO, where #286 refuses to dispatch uncontained rather than degrading. Same question, and I
    # had answered it two different ways in the same afternoon, with the weaker answer given to
    # the component with the larger blast radius.
    #
    # What "uncontained" costs: the only thing then standing between the evolver and the split
    # file, the held-out specs and peer arms' skills is a set of string-matching hooks that have
    # demonstrably failed three times (#215 false positive, #217 partial fix, #248 examined ZERO
    # records and reported "passed"). The result is not a crash, it is that a positive gate
    # result becomes unfalsifiable after the fact.
    #
    # DO NOT FLIP THIS MID-ARM. Arms A-C ran uncontained; containment should be a no-op on every
    # number, but "should be" is not "measured", so a change of world belongs at an arm boundary.
    evolver_sandbox: bool = True
    # SSB v2 only: which skill-version bucket of a multi-bucket cache to digest.
    digest_bucket: str = ""
    leg_timeout_s: int = 3600
    evolver_timeout_s: int = 5400
    # The evolver run log, the only record of what the evolver actually read.
    # "" means the conventional per-epoch name; an ABSENT log is a HALT, never a skip.
    evolver_log: str = ""
    paths: EpochPaths = field(init=False)

    def __post_init__(self):
        self.paths = EpochPaths(self.run_dir, self.epoch)


# ── verifies (each re-derives from the artifact) ─────────────────────────────────────────
def _v_digests_train_only(ctx: Ctx):
    """#186. The evidence surface must contain train ids and nothing else. Re-derived by
    intersecting the digest filenames with the split's own val/test id lists."""
    d = ctx.paths.digests
    if not d.is_dir():
        return False, f"no digest dir at {d}"
    sp = json.loads(Path(ctx.spec.split_path).read_text())
    def ids(k):
        v = sp.get(k) or []
        return set(v if not v or isinstance(v[0], str) else [x["id"] for x in v])
    # Naming-agnostic. This globbed `task_*.md` until 2026-08-09 — banking's convention. Retail
    # names its digests by bare task id (`5.md`), so on the first real drive `have` came back
    # EMPTY and the step failed with "digest dir has no task_*.md" against a directory holding
    # five perfectly good digests. Third site of the same assumption after run_epoch()'s leg
    # filenames and check_no_heldout_reachable()'s id matching.
    #
    # It failed LOUDLY, which is the right direction and why this was cheap to find — but it
    # would have blocked a correct epoch. Sidecars the digest builder writes alongside the
    # per-task files are excluded by name.
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
    # STRICTLY STRONGER than the old val/test intersection: every digest must BE a train task.
    # An id in neither train nor val nor test — a stale file from another split, say — used to
    # pass silently because it intersected nothing.
    stray = have - tr
    if stray:
        return False, f"digests present that are NOT train tasks: {sorted(stray)[:6]}"
    return True, f"{len(have)} digests, all {len(have & tr)} train, 0 val, 0 test, 0 stray"


def _v_skills_load(ctx: Ctx):
    """#214. Ask the REAL loader how many skills exist — not the filesystem, not the evolver.

    THIS FUNCTION USED TO SAY THAT AND NOT DO IT. Until 2026-08-09 it globbed `*.md`, counted
    `*/SKILL.md` directories and grepped the first 400 bytes for "description:", which is a
    RE-IMPLEMENTATION of the consumer's rules rather than the consumer. That is the single
    recurring defect shape in this programme — checking a proxy and reporting it as the thing —
    and it sat inside the check written to stop exactly that. The proxy and the loader agree
    today; nothing made them agree tomorrow, and #214 was originally missed precisely because
    the producer's count and the consumer's count were assumed equal.

    It now calls the invariants layer, which copies the staging dir into a throwaway workspace
    and asks `_load_skills` what it returns. One implementation, and it is the real one.

    A LOADER THAT CANNOT BE IMPORTED IS A HALT. The invariants layer returns SKIP there, which
    is right for a coverage report and wrong for a gate — and a stray `attr.py` on sys.path
    silently produced that SKIP on 2026-08-09 while overall coverage still read PASS.
    """
    from src.orchestrator.invariants import check_skills_load_through_consumer
    st = ctx.paths.staging
    if not st.is_dir():
        return False, f"no staging dir at {st}"
    c = check_skills_load_through_consumer(st)
    if c.status == "SKIP":
        return False, (f"could not ask the real loader: {c.detail}. An audit that cannot run "
                       f"must not read as one that passed.")
    if c.status != "PASS":
        return False, c.detail

    # #120 + the residual half of #230. tau3_evolver.sh SEEDS staging by copying the previous
    # iteration's promoted skills into it before the evolver runs. So a staging dir that loads
    # cleanly proves the SEED is well-formed and says nothing about whether the evolver produced
    # anything. That is how this step reported "RAN + VERIFIED: 4 skills load" 45 seconds after
    # dispatch while the evolver was still writing.
    #
    # An epoch whose library is byte-identical to the previous one is not automatically wrong —
    # "the evidence supports no change" is a legitimate finding — but it is never something to
    # record silently, because the identical set will be re-measured and re-gated as though it
    # were new. So it stops here for a person.
    # #237. THE EPOCH-1 HOLE IN THE CHECK BELOW. The byte-identical guard needs a previous
    # promoted set to compare against, so on epoch 1 it does not run at all — and an EMPTY
    # staging dir loads perfectly cleanly (zero skills, zero errors). Measured 2026-08-09: the
    # evolver REFUSED to launch (exit 10, prompt/evidence mismatch) but had already created its
    # staging dir, so on the very next invocation this step reported ok, `advance()` treated the
    # dispatch as satisfied, and THE EVOLVER WAS NEVER RE-RUN. A refusal became a pass, silently,
    # using nothing but the debris the refusal left behind. The fixed prompt was never exercised.
    #
    # An empty library IS a legitimate evolver outcome — the prompt says so explicitly, and a
    # manufactured rule is worse than none. But "it decided to ship nothing" and "it never ran"
    # are not the same event and must not present identically. So: stop, and make a person say
    # which one happened. Same shape, and same reason, as the byte-identical guard below.
    if not _skill_fingerprint(st):
        return False, (f"{st.name} exists but is EMPTY — zero skills load. That is either an "
                       f"evolver that deliberately shipped nothing (legitimate) or one that "
                       f"never ran / refused and left this directory behind (not). Check "
                       f"tau3_evolver_e{ctx.epoch}.log for a completion before continuing; if "
                       f"the evolver did not complete, delete this directory so the dispatch "
                       f"step is unsatisfied again and can re-run.")
    if ctx.epoch > 1:
        seed = EpochPaths(ctx.run_dir, ctx.epoch - 1).promoted
        if seed.is_dir() and _skill_fingerprint(seed) == _skill_fingerprint(st):
            return False, (f"{st.name} is byte-identical to {seed.name} — the evolver changed "
                           f"nothing, or has not finished writing. Rolling this out would "
                           f"re-measure the previous epoch's library under a new version "
                           f"number. Confirm which it is before continuing.")
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
    """#301. The operator<->AO exchange for this epoch is on disk, two-sided, and uncorrupt.

    WHY THIS IS A STEP AND NOT JUST A WRITE. `conversation.record` deliberately does not raise on
    a write failure — a full disk must not kill a paid epoch. But an observability channel that
    fails silently is how #301 happened in the first place: I believed the conversation was
    captured, and the belief was never checked against the artifact. So the write is best-effort
    and the CHECK is mandatory, which is the same division this file already uses for
    BLOCKED_e{N}.json (#265: "losing the audit record is tolerable; proceeding past an unanswered
    question is not" — with the difference that here the record IS the deliverable, so it gets a
    gate of its own).

    ABSENT IS NOT AUTOMATICALLY A FAILURE, AND THIS IS THE #298 LESSON APPLIED. Flipping a default
    to ON and then HALTing on every run that predates it turned four healthy historical epochs
    into spurious failures. The distinction that makes absence readable here: under `--step`,
    main() records the operator's turn BEFORE any verify runs, so the file must exist and its
    absence means the write failed — HALT. Under `--plan` nothing is recorded by design
    (ctx.readonly), so absence is UNMEASURED and says so.

    Placed after the authoring step so BOTH sides have necessarily spoken: my composed message and
    the AO's reply are recorded by `author_prompt` itself. Checking two-sidedness before that
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
    """#253. The prompt exists and the AO's own authoring record is beside it.

    Deliberately weak on its own — it asserts an artifact, not a quality. The claims INSIDE the
    prompt are checked by #208 (numbers re-derive from the leg json) and #253 (no ids, no held-out
    figure), and its judgement by the operator gate. Splitting it that way is the point: authorship
    moved to the AO, verification did not move anywhere.

    IT DOES CHECK ONE THING ABOUT HOW THE AO RAN: whether the conversation the run declared it
    wanted is the conversation it got (#288). Before this, loop.py contained ZERO occurrences of
    session_mode / session_id / AO_SESSION, so a run that declared `ao_continuous_session` and
    then authored four unrelated single-shot epochs — the exact ao4 state gap 3 exists to close —
    passed every gate and every report line identically to one that worked. Mutation testing
    found FOUR separate ways to sever the feature (hardcode the kwarg False, drop *sess_argv from
    argv, neuter the RUNTIME.json read, neuter the `if continuous_session:`) with all 45 fixtures
    green, because nothing anywhere compared the request against the result. A feature whose
    absence is invisible is not shipped, it is claimed.
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


def _claude_quota_gate(step: str, model: str | None) -> None:
    """Refuse to start `step` unless the Claude five-hour window is clear — WHEN it applies.

    #304. THE GATE FOLLOWS THE MODEL, NOT THE RUN. `src.orchestrator.quota_scope` owns that rule
    and is the only place it is written down; see its docstring for why a provider-gateway model
    is not gated and why anything unrecognised is. Before this, the evolver was gated regardless
    of model (so a DSV4F evolver was blocked by a reading about a window it never touches) while
    the AO prompt author — Claude in every run to date — was not gated at all.

    #305. THE VERDICT IS THE EXIT CODE, NEVER A SUBSTRING. This function previously tested
    `"CLEAR" not in q.stdout.upper()`. The UNKNOWN branch of `quota_fresh_reading.classify()`
    returns "... Cannot tell whether the window is clear; treat as blocked." — which CONTAINS
    "clear", so the test passed and the loop launched. That is fail-open on the one gate whose
    failure killed an evolver mid-run with nothing written (2026-08-01), and it was live on
    exactly the reading ao6 hit twelve times on 2026-08-13: the shell launcher's own exit-code
    check is the only thing that stopped it. Exit codes are 0=CLEAR, 1=BLOCK, 2=UNKNOWN, and
    UNKNOWN blocks like BLOCK — so anything non-zero halts.
    """
    from .quota_scope import consumes_claude_quota, gate_reason

    if not consumes_claude_quota(model):
        # SAY SO OUT LOUD. A gate that silently does not run reads, in a log, exactly like a
        # gate that ran and passed — and "it must have been checked" is how a skipped guard
        # survives review.
        print(f"  [{step}] {gate_reason(model)}")
        return

    q = subprocess.run([sys.executable, "scripts/quota_fresh_reading.py"],
                       capture_output=True, text=True)
    if q.returncode != 0:
        raise Halt(f"five-hour quota gate did not return CLEAR (rc={q.returncode}), so "
                   f"{step} may not start:\n  {(q.stdout or q.stderr).strip()[:300]}")


def _run_author_prompt(ctx: Ctx) -> None:
    """THE AO WRITES THE EVOLVER'S USER PROMPT. Facts derived deterministically, prose by a model.

    #304: GATED, because this step is Claude-powered on every run to date (`prompt_author_model`
    defaults to opus[1m]) and therefore spends the same five-hour window the evolver gate guards.
    It was ungated until 2026-08-13 — the gate was attached to the step that might not spend the
    resource and missing from the step that always does.
    """
    from .prompt_author import author_prompt, derive_facts

    _claude_quota_gate("prompt author",
                       getattr(ctx.spec, "prompt_author_model", None) or ctx.prompt_author_model)
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
    facts = derive_facts(dg, Path(leg) if leg else None, corpus,
                         variant=getattr(ctx.spec, "variant", "") or "")
    info = author_prompt(facts=facts, out=ctx.paths.prompt, epoch=ctx.epoch, digests=dg,
                         staging=ctx.paths.staging,
                         oracle=Path(ctx.oracle), log=ctx.run_dir / f"e{ctx.epoch}_prompt_author.log",
                         model=getattr(ctx.spec, "prompt_author_model", None)
                         or ctx.prompt_author_model,
                         policy=Path(ctx.prompt_policy) if ctx.prompt_policy else None,
                         prior_skills=prior,
                         # OPT-IN, default OFF. Turning it on changes what the AO IS: one
                         # conversation across the run instead of four strangers, which is the
                         # intended V2 shape but is NOT what ao4 measured. Defaulting off keeps
                         # every completed run byte-reproducible; RUNTIME.json opts a new run in.
                         continuous_session=bool(getattr(ctx, "ao_continuous_session", False)))
    info["facts"] = facts.as_block()
    info["phoenix_project"] = ao_project(ctx)
    info["phoenix_replay"] = _replay_to_phoenix(
        role="ao", log=ctx.run_dir / f"e{ctx.epoch}_prompt_author.log", project=ao_project(ctx),
        prompt=ctx.run_dir / f"e{ctx.epoch}_AUTHOR_MESSAGE.md")
    (ctx.run_dir / f"e{ctx.epoch}_PROMPT_AUTHOR.json").write_text(json.dumps(info, indent=1))


def run_id(ctx: Ctx) -> str:
    """#309. THE RUN'S RUNTIME IDENTITY IS ITS RUN DIRECTORY, NOT `spec.name`.

    `spec.name` is operator-typed at intake (cli.py) and NOTHING checks it for uniqueness. Four
    run dirs on disk currently carry another run's name: tau3_banking_ao6 -> "tau3_banking_ao5",
    tau3_sbcheck -> "tau3_banking_ao4", tau3_retail_agentic -> "tau3_retail", tau3_smoke_orch ->
    "smokeorch". The ao6 case is the one that was caught in flight and it was not one collision
    but five, because five separate runtime identities were derived from that single string:

        the evolver's isolated work dir      (EVOSKILL_TAU3_WORK_TAG -> tau3_evolver.sh)
        the CC session dir that follows it   (~/.claude/projects/<slug-of-cwd>)
        the evolver's Phoenix project        (EVOSKILL_TAU3_PHOENIX_PROJECT)
        the AO's Phoenix project             (ao_project)
        each leg's Phoenix project           (PHOENIX_PROJECT_NAME)

    MEASURED CONSEQUENCE, not a hypothetical: ao6's epoch-1 prompt was staged into ao5's e1 work
    dir (overwriting ao5's), and `_v_traces_in_phoenix` (#301) returned a false PASS on all five
    dispatch attempts because it found ao5's spans and could not tell them from ao6's. The
    sandbox profile built for that work dir also granted kernel read on ao5's 957 KB evolver
    transcript, which holds ao5's finished SKILL.md bodies verbatim — on ao6's own train tasks —
    with both PreToolUse hooks and the #213 audit blind to that path shape.

    EpochPaths.leg() and EVOSKILL_SEAT_MEMBER already keyed on the run dir. These five did not.

    DELIBERATELY NOT APPLIED to the two SSB v2 cache/snapshot names (`{name}_e{N}_cache` and the
    `ssbv2_{name}_v{N}_train_tasks` snapshot root). Those sit behind `benchmark == "ssbv2"`, so
    they cannot execute on a tau3 run at all, and renaming them would orphan ssbv2_ao1's existing
    cache directories — trading a latent collision for a certain one. Filed separately.
    """
    return Path(ctx.run_dir).resolve().name


def ao_project(ctx: Ctx) -> str:
    """Phoenix project for the AO's authoring session. Sibling to the evolver's `{id}-e{N}` and
    the legs' `{id}-e{N}-{key}`, so one epoch's three agents sort together in the project list."""
    return f"{run_id(ctx)}-e{ctx.epoch}-AO"


def _replay_to_phoenix(*, role: str, log, project: str, prompt=None, trace_json=None) -> dict:
    """Replay a stream-json log into Phoenix, right after the session that wrote it. Best-effort.

    WHY AUTOMATIC AND NOT A MANUAL STEP. Until now this was something I remembered to run, and
    #301 is what that produced: four AO sessions whose replay was done by hand, into a project
    whose root span had no input attached, while the conversation was claimed to be in Phoenix.
    A trace that depends on the operator remembering is a trace that is missing exactly when the
    operator is busy.

    BEST-EFFORT, AND CHECKED SEPARATELY. Losing observability must not fail a paid epoch, so a
    failure here is recorded and printed rather than raised — and `_v_traces_in_phoenix` then
    turns a missing project into a DECLARED failure at the end of the epoch. Same division as the
    conversation channel: the write is soft, the check is hard.
    """
    log = Path(log)
    out = {"role": role, "project": project, "log": str(log), "ok": False, "detail": ""}
    if not log.is_file():
        out["detail"] = f"no stream log at {log} — nothing to replay"
        print(f"[phoenix] {out['detail']}", file=sys.stderr)
        return out
    argv = [sys.executable, str(REPO / "scripts/phoenix_replay_evolver.py"),
            "--log", str(log), "--project", project, "--role", role]
    if prompt and Path(prompt).is_file():
        argv += ["--prompt", str(prompt)]
    if trace_json and Path(trace_json).is_file():
        argv += ["--trace-json", str(trace_json)]
    try:
        r = subprocess.run(argv, cwd=str(REPO), capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as e:
        out["detail"] = f"{type(e).__name__}: {e}"
        print(f"[phoenix] replay of {log.name} failed: {out['detail']}", file=sys.stderr)
        return out
    out["ok"] = r.returncode == 0
    out["detail"] = (r.stdout or r.stderr or "").strip()[-600:]
    print(f"[phoenix] {'replayed' if out['ok'] else 'REPLAY FAILED for'} {log.name} -> {project}",
          file=sys.stderr)
    return out


def _phoenix_projects(endpoint: str = "") -> tuple:
    """(reachable, {name: id}) from Phoenix's REST API.

    REST, NOT GraphQL: the standing rule for this project after GraphQL returned false negatives
    for projects that demonstrably had spans ([[feedback-check-phoenix-yourself]]).
    """
    import urllib.error
    import urllib.request
    ep = endpoint or os.environ.get("PHOENIX_ENDPOINT", "http://localhost:6006")
    try:
        raw = urllib.request.urlopen(f"{ep}/v1/projects?limit=1000", timeout=8).read()
        return True, {p["name"]: p["id"] for p in json.loads(raw).get("data", [])}
    except (urllib.error.URLError, OSError, json.JSONDecodeError, KeyError, TypeError):
        return False, {}


def _phoenix_span_count(project: str, endpoint: str = "") -> int:
    """Spans in `project`, or -1 meaning COULD NOT READ. Callers must not conflate the two.

    -1 IS NOT A COUNT, and treating it as one was a live defect in this file. `_v_traces_in_phoenix`
    tested `== 0` for emptiness, so a project whose spans could not be read (-1) sailed through as
    non-empty and the step printed `(e1=-1, AO=-1, train=-1, val=-1)` while reporting "all 4
    projects present with spans". A check that passes on a failed measurement is the #154 shape:
    it cannot fail for the reason it exists.

    WHY IT FAILS HERE, measured 2026-08-12: Phoenix's /spans endpoint times out (>30s) on these
    projects — a solver leg is ~30k spans and the endpoint materialises them all. Raising the
    timeout is not the fix, because the caller does not need the exact count, only whether the
    project is non-empty. limit=1 answers that in milliseconds.
    """
    import urllib.error
    import urllib.request
    ep = endpoint or os.environ.get("PHOENIX_ENDPOINT", "http://localhost:6006")
    try:
        raw = urllib.request.urlopen(f"{ep}/v1/projects/{project}/spans?limit=1",
                                     timeout=20).read()
        return len(json.loads(raw).get("data", []))
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return -1


def _v_traces_in_phoenix(ctx: Ctx):
    """Every agent that ran this epoch left a trace in Phoenix — checked, not assumed.

    Requirement: every trace (evolver+solver) is logged to Phoenix. Four projects per
    epoch, one per party that consumed tokens:

        {name}-e{N}          the evolver's turn tree, streamed live by the launcher
        {name}-e{N}-AO       the AO's authoring session, replayed by _run_author_prompt
        {name}-e{N}-train    the train solver leg
        {name}-e{N}-val      the val solver leg

    UNREACHABLE IS NOT MISSING. If Phoenix is down the step reports UNMEASURED and says so, rather
    than failing an epoch because a local observability service was restarted. That distinction is
    the whole content of #298: absence of a reading is not a negative reading. But an EMPTY project
    on a REACHABLE Phoenix is a real failure, because the run said it wrote there and it did not.

    Placed last in the epoch. Three of the four projects do not exist until their leg has run, so
    anywhere earlier this fires on correct behaviour.
    """
    up, projects = _phoenix_projects()
    want = [f"{run_id(ctx)}-e{ctx.epoch}", ao_project(ctx),
            f"{run_id(ctx)}-e{ctx.epoch}-train", f"{run_id(ctx)}-e{ctx.epoch}-val"]
    if not up:
        return True, (f"{UNMEASURED}Phoenix is not reachable, so I cannot confirm the epoch's "
                      f"traces arrived. Expected projects: {', '.join(want)}. This is a missing "
                      f"READING, not a missing trace — re-run this check with Phoenix up.")
    missing = [p for p in want if p not in projects]
    counts = {p: _phoenix_span_count(p) for p in want if p in projects}
    empty = [p for p, n in counts.items() if n == 0]
    # -1 IS "COULD NOT READ", NOT "HAS SPANS". Kept separate from `empty` because the two demand
    # opposite responses: an empty project means the trace never arrived and the epoch should
    # HALT; an unreadable one means Phoenix answered the project list but not the span query, and
    # HALTing a paid epoch on a slow observability endpoint would be the #298 error again. So it
    # is reported as UNMEASURED for those projects, loudly, and never silently folded into a pass.
    unread = [p for p, n in counts.items() if n < 0]
    if missing or empty:
        parts = []
        if missing:
            parts.append(f"no project named {missing}")
        if empty:
            parts.append(f"project(s) present but holding ZERO spans: {empty}")
        return False, ("; ".join(parts) + f". Phoenix IS up and lists {len(projects)} project(s), "
                       f"so this is an absent trace, not an absent reading. The run's own logs "
                       f"are unaffected — re-replay with scripts/phoenix_replay_evolver.py.")
    if unread:
        return True, (f"{UNMEASURED}all 4 projects exist, but Phoenix would not answer a span "
                      f"query for {unread} (its /spans endpoint times out on large projects). "
                      f"Existence is confirmed; non-emptiness is NOT. Check by hand at "
                      f"{os.environ.get('PHOENIX_ENDPOINT', 'http://localhost:6006')}/projects.")
    shown = ", ".join(f"{p.split('-')[-1]}{'≥' if counts[p] else ''}{counts[p]}" for p in want)
    return True, f"all 4 projects present and non-empty ({shown}; probed with limit=1)"


def _v_prompt_no_leakage(ctx: Ctx):
    """#253. Mechanical, because a model now writes the prompt."""
    from .invariants import check_prompt_no_leakage
    c = check_prompt_no_leakage(ctx.paths.prompt, Path(ctx.spec.split_path))
    if c.status == "SKIP":
        return True, f"{UNMEASURED}{c.detail}"
    return c.status == "PASS", c.detail


def _v_prompt_not_analytical(ctx: Ctx):
    """#262. The AO prepares raw information and is not part of the optimizer. Mechanical, because
    the policy prose demonstrably did not hold it — see check_prompt_not_analytical."""
    from .invariants import check_prompt_not_analytical
    c = check_prompt_not_analytical(ctx.paths.prompt)
    if c.status == "SKIP":
        return True, f"{UNMEASURED}{c.detail}"
    return c.status == "PASS", c.detail


def _v_legs_comparable(ctx: Ctx, key: str):
    """#251. The skill leg and its BASE comparator must be the same comparison — same tasks,
    same samples-per-task, and both matching the spec. See invariants.check_legs_paired_comparably
    for why the full banking config hides this (there spec.n_samples == the script default of 3,
    so the wiring defect and the correct behaviour are indistinguishable)."""
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
    return c.status == "PASS", f"{key}: {c.detail}"


def _v_leg_loaded_promoted(ctx: Ctx, key: str):
    """#214's second half. A leg can run against the WRONG skills dir and still report a
    clean pass rate. Read the dir the leg says it loaded and compare to what was promoted."""
    p = ctx.paths.leg(key)
    if not p.exists():
        return False, f"no result json at {p.name}"
    d = json.loads(p.read_text())
    want = ctx.paths.promoted
    raw = d.get("skills_dir")
    if not raw:
        # #305: the old guard built Path(raw or "") FIRST, and Path("") == Path(".")
        # is truthy — so a runner that never wrote the field (#304, the SSB v2 runner)
        # read as "loaded ." instead of reaching this branch. Absence must be decided
        # on the raw value, before Path() can invent a cwd.
        # #304 fallback: the runner's startup banner lists every loaded skill as an
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
            return True, (f"{key}: JSON lacks skills_dir (#304) — log evidence: "
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
    """#234. Did the solver ever OPEN the library it was given?

    tau3 Arm C ran four legs, scored normally, gated normally, and produced a null that read as
    "skills did not help ling-3.0-flash". Its solver consulted the library in 2 of 75 rollouts —
    3%, against 85-100% in every other arm in the project. The intervention was never delivered;
    nothing anywhere noticed; and four legs were spent measuring a closed channel.

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
    try:
        from .skill_delivery_shim import assess_delivery
    except Exception:                                                    # noqa: BLE001
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "tau3_skill_delivery", Path("scripts/tau3_skill_delivery.py"))
        if spec is None or spec.loader is None:
            return True, UNMEASURED + "scripts/tau3_skill_delivery.py is not importable."
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assess_delivery = lambda s, d: mod.assess(mod.library(s), d)     # noqa: E731
    r = assess_delivery(skills, nxt)
    # A reader that finds none of its OWN markers has not measured 0% — it has failed to read,
    # and those must not share a return value. On 2026-08-12 the tau3-only fallback reader was
    # pointed at an SSB v2 surface, matched none of its `(tool.skill)` notation, and reported
    # "0/180 consulted — NOT DELIVERED" for a version whose digests hold 405 skill-load lines
    # (100% of rollouts). `rollouts` was correct, which is what made it convincing: only the
    # numerator was blind. Absent key => a reader that predates this distinction; treat as
    # readable, exactly as before. See src/orchestrator/skill_delivery_shim.py.
    if r.get("unreadable"):
        return False, (f"{r['rollouts']} rollout section(s) present but NOT ONE carries a "
                       f"skill-load marker this reader knows. That is a READER failure, not 0% "
                       f"uptake, and must not be read as 'not delivered'. Add the surface's "
                       f"dialect to src/orchestrator/skill_delivery_shim.py (#234).")
    detail = (f"{r['consulted']}/{r['rollouts']} rollouts consulted a skill = {r['uptake']:.0%}"
              + (f" [{r['dialect']}]" if r.get("dialect") else "")
              + (f"; INVALID names {dict(**r['typos'], **r['category_errors'])}"
                 if (r["typos"] or r["category_errors"]) else "")
              + (f"; never loaded {r['never_loaded']}" if r["never_loaded"] else ""))
    if not r["delivered"]:
        return False, (detail + " — the library was NOT DELIVERED. Any gate verdict on this "
                       "version is about the delivery channel, not about the skills (#234).")
    return True, detail


def _v_outcomes_reconcile(ctx: Ctx, key: str):
    """#209. pass + graded-fail + never-graded must equal the total, and never-graded must
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
        # SSB v2 schema: a turn-capped episode is STILL graded on whatever file it
        # saved, so never-graded is structurally 0 and turn_exhausted is a separate,
        # non-exclusion category. What CAN silently move here is the episode grid,
        # so reconcile total against the declared n x n_samples instead.
        never = 0
        n, ns = int(d.get("n", 0)), int(d.get("n_samples", 0))
        if n * ns != total:
            return False, (f"{key}: n({n}) x n_samples({ns}) != sample_total({total}) "
                           f"— episode grid does not reconcile")
        capped_note = f", capped-but-graded {int(d.get('turn_exhausted', 0))}"
    else:
        return False, (f"{p.name} has neither termination_reasons (tau3) nor "
                       f"turn_exhausted (ssbv2) — cannot attribute the #209 channels")
    graded_fail = total - passed - never
    if graded_fail < 0:
        return False, (f"{key}: pass({passed}) + never-graded({never}) exceeds total({total})")
    return True, (f"{key}: pass {passed}, graded-fail {graded_fail}, never-graded {never} "
                  f"= {total}{capped_note}")


def _leg_pass_total(d: dict, src) -> tuple:
    """(passed, total) episodes from a leg JSON, whichever benchmark wrote it.

    tau3 writes `pass`/`total`; SSB v2 writes `sample_pass`/`sample_total`, where the episode
    unit is a task x sample draw (60 tasks x 3 samples = 180). _v_prompt_numbers read `d["pass"]`
    unconditionally and so raised KeyError on every SSB v2 leg — the gate neither passed nor
    failed, it CRASHED, which the driver reported as "verify raised KeyError: 'pass'". A crashing
    guard is worse than a failing one: it carries no verdict, and the prompt it was meant to check
    was in fact correct (39 of 180, 21.7%, exactly the measured base).

    REFUSES rather than defaulting to (0, 0). A zero total would make every fraction in every
    prompt trivially unmatchable, turning #208 off silently — the failure mode #154 is named for.
    """
    for pk, tk in (("pass", "total"), ("sample_pass", "sample_total")):
        if pk in d and tk in d:
            return int(d[pk]), int(d[tk])
    raise KeyError(
        f"{getattr(src, 'name', src)} has neither pass/total nor sample_pass/sample_total "
        f"(keys: {sorted(d)[:12]}) — cannot re-derive the prompt's figures")


def _v_prompt_numbers(ctx: Ctx):
    """#208. The evolver reasons from the headline figures in its prompt. If those disagree
    with the result JSONs they came from, it is reasoning about a different experiment.
    Checks the episode fractions that appear as `N of M` or `N/M`."""
    pr = ctx.paths.prompt
    prev = ctx.epoch - 1
    if not pr.exists():
        return False, f"no prompt at {pr.name}"

    # WHERE THE PRIOR NUMBERS LIVE. For epoch >= 2 it is the previous epoch's train leg, named
    # by the same expression _run_leg writes. For epoch 1 it is the BASE leg, which does NOT
    # follow that convention (`retail_base_train.json`, not `<run>_v0_train.json`) and so must be
    # declared. This used to `return True` for epoch 1 with "no prior leg to re-derive against",
    # which was VACUOUS at the one epoch that matters most: every epoch-1 prompt in this
    # programme quotes a base pass rate, and #208 exists because the evolver reasons from exactly
    # those figures. A check that cannot fail on the epoch it most needs to guard is #154's shape.
    if prev >= 1:
        # Via EpochPaths.leg(), NEVER a second copy of the naming expression. This rebuilt the
        # name inline until 2026-08-09 and was therefore blind to any run whose legs are named
        # differently — the FIFTH site of that one assumption, after the digest glob, run_epoch's
        # leg filenames, check_no_heldout_reachable's id matching, and _run_leg's tag. It failed
        # loudly on Arm 2, which is the only reason it was cheap to find.
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

    d = json.loads(src.read_text())
    text = pr.read_text(errors="replace")
    want = _leg_pass_total(d, src)
    found = set()
    # #254. The `N of M episodes` arm used to REQUIRE the bold markers to wrap the phrase exactly
    # (`\*\*(\d+) of (\d+) episodes?\*\*`). The first AO-authored prompt wrote
    #     - **passed: 1 of 3 episodes (33.3%)**
    # where the bold span is wider than the phrase, so the pattern missed it and #208 HALTed a
    # perfectly correct prompt. A guard that refuses valid input gets switched off, so being too
    # strict is a defect of the same family as being vacuous — see #251b.
    # The bold requirement was load-bearing for one reason only: to stop `2 of 3 tasks` matching
    # when the denominator happens to equal the leg total. The literal word `episodes` does that
    # job directly, so anchor on the noun and let the markup be whatever prose wants.
    for m in re.finditer(r"(\d+)\s+of\s+(\d+)\s+episodes?\b|\b(\d+)\s*/\s*(\d+)\b", text):
        a, b, c, e = m.groups()
        pair = (int(a), int(b)) if a else (int(c), int(e))
        if pair[1] == want[1]:
            found.add(pair)
    if not found:
        return False, f"prompt states no /{want[1]} episode fraction to check against {src.name}"

    # #223. This used to demand that EVERY fraction on the denominator equal the current leg's,
    # which made an honest multi-epoch trajectory unstateable: "no skills scored 59 of 90, round
    # one scored 64" is exactly the comparison the evolver needs, and the gate rejected it. The
    # author's only recourse was to phrase prior rates so the regex could not see them — a check
    # that is satisfied by hiding evidence from it is worse than no check.
    #
    # So: the CURRENT leg's figure must be present, and every other figure on that denominator
    # must match some OTHER leg that really exists (the base, or an earlier epoch's train leg).
    # A number that matches no leg at all is still a refusal. Strictly stronger than before —
    # the old rule could not tell an invented figure from a correctly-quoted prior one.
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
        return False, (f"prompt states {sorted(found)} on /{want[1]} but never the CURRENT leg's "
                       f"{want[0]}/{want[1]} from {src.name}")
    unknown = {f for f in found if f != want and f not in known}
    if unknown:
        return False, (f"prompt states {sorted(unknown)} on /{want[1]}, which matches neither "
                       f"{src.name} ({want[0]}/{want[1]}) nor any prior leg "
                       f"({sorted(known)}) — an unsourced number")
    quoted = {known[f] for f in found if f != want}
    return True, (f"prompt's {want[0]}/{want[1]} re-derives from {src.name}"
                  + (f"; also correctly quotes {', '.join(sorted(quoted))}" if quoted else ""))


def _v_evolver_contained(ctx: Ctx):
    """#294. If the run declared kernel containment for the evolver, prove the dispatch had it.

    Same shape and same reason as #288. `EVOSKILL_EVOLVER_SANDBOX` is read by a shell script four
    layers away; if the plumbing broke — a typo in the env key, a launcher that ignored it, an
    `sb` array that expanded to nothing — every artifact and every other verify would look
    identical to a properly contained run. So the launcher writes a receipt BEFORE dispatch and
    this compares it against what the run said it wanted.

    UNMEASURED, never a pass, when the receipt predates this check: an old run cannot be
    retroactively declared contained, and reporting it as ok would be the vacuous-guard shape
    this file keeps having to remove.
    """
    # THE HARNESSES HAVE DIFFERENT DEFAULTS, AND THAT IS NOT A WART. `direct` runs a python
    # module and has been contained unconditionally since #250 -- there is no opt-out and none is
    # wanted. `claude_cli` runs the CLI, whose containment was only proved to START a real process
    # on 2026-08-11 (#291), so it carries a recorded escape hatch. Expected containment is
    # therefore a function of the harness, not a single global flag.
    harness = getattr(ctx.spec, "evolver_harness", "claude_cli")
    want = True if harness == "direct" else bool(getattr(ctx, "evolver_sandbox", True))
    rec = ctx.run_dir / f"e{ctx.epoch}_EVOLVER_CONTAINMENT.json"
    if not rec.is_file():
        # #298. A NEW GUARD MUST NOT FAIL ON HISTORY IT CANNOT APPLY TO. Flipping #294's default
        # to ON immediately turned every completed epoch into a HALT: ao4 epoch 4 went from
        # 0 HALT to 1, because its launcher predates the receipt and no artifact could ever
        # appear. A check that a finished, correct run cannot pass is not a check, it is noise
        # that trains the operator to ignore the board.
        #
        # Absence is therefore UNMEASURED in both directions, with the reason stated, and the
        # only FAIL is a receipt that CONTRADICTS what the run declared. That still catches the
        # thing worth catching -- broken plumbing on a live dispatch, since the launcher now
        # always writes a receipt -- without inventing a verdict about the past.
        ran = (ctx.run_dir / f"tau3_evolver_e{ctx.epoch}.log").exists()
        if not want:
            return True, (f"{UNMEASURED}no {rec.name}; containment was explicitly disabled for "
                          f"this run, so the evolver had hooks + --add-dir only (#186/#294)")
        if ran:
            return True, (f"{UNMEASURED}the evolver for this epoch ALREADY RAN under a launcher "
                          f"that predates containment accounting (no {rec.name}). Whether it was "
                          f"contained is NOT established and cannot be established after the "
                          f"fact — treat this epoch as uncontained when reasoning about #213/#186.")
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
    if want and not Path(str(d.get("profile") or "")).is_file():
        return False, (f"{rec.name} claims sandbox-exec but its profile "
                       f"{d.get('profile')!r} is not on disk — the receipt describes containment "
                       f"nothing can confirm.")
    return True, (f"evolver ran under {d.get('containment')}"
                  + (f" ({Path(str(d['profile'])).name})" if want else ""))


def _v_evolver_read_nothing_forbidden(ctx: Ctx):
    """#213 + #186, audited from the EVOLVER'S OWN LOG — the only record of what it touched.

    These two checks existed in the invariants layer and were never reachable from here, which
    stopped mattering the moment this module gained the ability to DISPATCH an evolver: the loop
    could launch one, take its skills, and promote them without ever asking whether it had read a
    peer arm's answers to the same train tasks (#213) or reached a held-out task spec (#186).
    Those are the two contamination modes that invalidate an arm outright, and they are invisible
    in the artifact — a contaminated skill set looks exactly like a clean one.

    CALLED, NOT REIMPLEMENTED. A second copy of a guard is how two guards become one real one.

    A SKIP IS A FAILURE HERE. The invariants layer returns SKIP when the log is absent, which is
    honest per-check and dishonest in aggregate — that exact conversion of "I could not check"
    into "nothing to report" is what let six of ten invariants silently no-op on the first real
    run. So an unreadable or missing log HALTS; if the evolver was driven by hand and its log
    lives elsewhere, declare the path rather than letting the audit quietly not happen.
    """
    from src.orchestrator.invariants import (check_no_heldout_reachable,
                                             check_no_peer_artifacts_read)
    log = Path(ctx.evolver_log) if ctx.evolver_log else \
        ctx.run_dir / f"tau3_evolver_e{ctx.epoch}.log"
    if not log.is_file():
        return False, (f"no evolver log at {log.name}, so neither the peer-artifact (#213) nor "
                       f"the held-out (#186) audit can run. Declare evolver_log if it lives "
                       f"elsewhere — an audit that cannot run must not read as one that passed.")
    out = []
    for c in (check_no_peer_artifacts_read(log, ctx.run_dir),
              check_no_heldout_reachable(log, Path(ctx.spec.split_path))):
        if c.status == "FAIL":
            return False, f"{c.name}: {c.detail}"
        if c.status == "SKIP":
            return False, f"{c.name} could not run: {c.detail}"
        out.append(c.name)
    return True, f"{len(out)} contamination audits passed against {log.name}"


def _skill_fingerprint(root: Path) -> dict:
    """Content digest of a skill set, keyed by skill name. ONE definition, used by both the
    promotion check and the proofread signature — two implementations of "has this set changed"
    is how a signature ends up covering something other than what was promoted."""
    return {f.parent.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16]
            for f in sorted(root.glob("*/SKILL.md"))}


def staging_sha256(root: Path) -> str:
    """A single stable hash over a whole skill set, for the countersignature to name."""
    fp = _skill_fingerprint(root)
    return hashlib.sha256(json.dumps(fp, sort_keys=True).encode()).hexdigest()


def _v_operator_signature(ctx: Ctx):
    """The proofread. Not automatable, and not assumed. Requires PROOFREAD.json in the epoch's
    run dir naming what was checked and the sha256 of the artifact checked, so a signature
    cannot silently outlive an edit to the thing it signed."""
    sig = ctx.run_dir / f"e{ctx.epoch}_{SIGN_FILENAME}"
    if not sig.exists():
        return False, (f"no countersignature at {sig.name}. The evolver prompt and the "
                       f"evolved skill set must be read by a person before dispatch/promotion. "
                       f"Write {sig.name} with keys: checked[], prompt_sha256, staging_sha256.")
    s = json.loads(sig.read_text())
    pr = ctx.paths.prompt
    if pr.exists():
        actual = hashlib.sha256(pr.read_bytes()).hexdigest()
        if s.get("prompt_sha256") != actual:
            return False, (f"signature covers prompt {str(s.get('prompt_sha256'))[:12]} but "
                           f"{pr.name} now hashes to {actual[:12]} — it was edited after the "
                           f"proofread")
    # THE SECOND HALF OF THE GATE. Its own message says the prompt AND the evolved skill set must
    # be read, but only the prompt hash was ever enforced. That gap is invisible in the ordinary
    # sequence: the operator signs after proofreading the PROMPT, at which point staging does not
    # exist yet, and the evolver then writes a skill set the signature silently appears to cover.
    # So: staging is checked exactly when it exists, and a signature written before it must be
    # renewed once it does. A signature that outlives the artifact it names is not a signature.
    st = ctx.paths.staging
    if st.is_dir() and _skill_fingerprint(st):
        actual = staging_sha256(st)
        if not s.get("staging_sha256"):
            return False, (f"{sig.name} covers the prompt but names no staging_sha256, and "
                           f"{st.name} now holds a skill set. Re-read the evolved skills and "
                           f"re-sign; the existing signature predates them.")
        if s["staging_sha256"] != actual:
            return False, (f"signature covers staging {str(s['staging_sha256'])[:12]} but "
                           f"{st.name} now hashes to {actual[:12]} — the skill set changed "
                           f"after the proofread")
    if not s.get("checked"):
        return False, f"{sig.name} lists nothing in `checked`"
    scope = "prompt+skills" if s.get("staging_sha256") else "prompt only (no staging yet)"
    return True, f"signed [{scope}]: {', '.join(s['checked'][:4])}"


# ── the epoch program ────────────────────────────────────────────────────────────────────
# ── runners ──────────────────────────────────────────────────────────────────────────────
# Until 2026-08-09 EVERY step was verify-only: `Step.run` existed in the dataclass and no step
# ever set it, so this layer could check an epoch an operator had already driven but could not
# drive one itself. The gap was invisible from the outside — the module reported 11 green steps
# either way — and it is why "the orchestrator is built" was true and "the orchestrator runs
# experiments" was not.
#
# WHAT IS DRIVABLE AND WHAT IS DELIBERATELY NOT. A runner is written only where the action is
# MECHANICAL: a deterministic transformation of artifacts already on disk. Two steps keep no
# runner on purpose:
#   * the operator gate — a countersignature a machine produces is not a countersignature;
#   * evolver dispatch — it spends real money against a live quota and, per the standing
#     directive, an evolver is launched only after a fresh quota reading and a human-proofread
#     prompt. Automating it would route around both.
# Every runner is idempotent and refuses rather than overwrites, because a re-run after a HALT
# must not silently destroy the evidence that caused the halt.

def _run_promote(ctx: Ctx) -> None:
    """staging -> skills_vN. Refuses to overwrite a promoted set that differs from staging."""
    src, dst = ctx.paths.staging, ctx.paths.promoted
    if not src.is_dir():
        raise Halt(f"nothing to promote: {src} does not exist")
    if not any(src.glob("*/SKILL.md")):
        raise Halt(f"refusing to promote an empty staging dir: {src}")
    if dst.exists():
        same = {p.relative_to(src): p.read_bytes() for p in src.rglob("SKILL.md")} == \
               {p.relative_to(dst): p.read_bytes() for p in dst.rglob("SKILL.md")}
        if same:
            return                                   # idempotent: already promoted, byte-identical
        raise Halt(f"{dst} exists and DIFFERS from staging — refusing to overwrite a promoted "
                   f"skill set; a leg may already have been measured against it")
    shutil.copytree(src, dst)


def _load_delivery_assessor():
    """The #234 uptake assessor, or None if it genuinely cannot be loaded.

    ONE loader, two callers (#257's build-time guard and #234's downstream verify). It exists as a
    function because the first cut of the build-time guard did `try: from .skill_delivery_shim
    import assess_delivery / except: return` — and that module is NOT present in this tree, so the
    guard silently no-opped while reporting nothing. A guard that cannot fail is the #154 shape,
    and I had just written one into the fix for a defect whose whole lesson is that agreeing
    signals may all be unread. Callers decide what None means; neither may treat it as a pass.
    """
    try:
        from .skill_delivery_shim import assess_delivery                   # noqa: PLC0415
        return assess_delivery
    except Exception:                                                     # noqa: BLE001
        pass
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "tau3_skill_delivery", Path("scripts/tau3_skill_delivery.py"))
    if spec is None or spec.loader is None:
        return None
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception:                                                     # noqa: BLE001
        return None
    return lambda s, d: mod.assess(mod.library(s), d)


def _digest_caches_for_epoch(ctx: Ctx) -> tuple:
    """#257. WHICH ROLLOUTS does epoch N's evidence surface describe?

    Epoch N's surface must describe version N-1's rollouts: base at epoch 1, skills_v1 at epoch 2,
    and so on. `ctx.digest_caches` names the BASE caches — it is an init-time declaration, one list
    for the whole run — so using it at every epoch rebuilds the epoch-1 surface forever.

    That is exactly what happened on tau3_banking_ao3: `digests_e2_train` came out BYTE-IDENTICAL to
    `digests_e1_train`, and the epoch-2 evolver would have been shown the no-skills rollouts as if
    v1 had never run. Nothing downstream could catch it, because everything downstream was
    consistent with it: #208 re-derives the prompt's headline against `base_leg`, which agrees with
    a base surface; the #234 uptake check reads this surface, so it reported 0/75 consulted and
    "NOT DELIVERED" for a version whose own transcripts contain 98 skill-tool calls. Three
    agreeing signals, all wrong, because all three read the same wrong artifact.

    The fix reads what the runner RECORDED rather than deriving the name a second time: every leg
    json carries `cache_dirs`, written by the process that opened them. #194 is the warning against
    the alternative — that field was once populated by a re-derivation that dropped the per-leg
    token, and recorded the shared-bucket name for four generations of rollouts.
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
                   f"fallback is #257 itself.")
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
    """#257's guard, placed where the cause is. Two independent ways for the surface to be stale,
    each checkable without trusting the other:

    1. It is byte-identical to the previous epoch's surface. Two epochs of rollouts producing the
       same bytes is not a coincidence any real run can produce.
    2. It shows no trace of the skills version it is supposed to describe. From epoch 2 on, the
       rollouts behind the surface ran WITH skills_v{N-1}, so a surface that contains zero
       skill-tool calls naming that library is describing different rollouts.

    Check 2 is the same assessor #234 uses downstream. Running it here too is deliberate: at this
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
                       f"rollouts, so it is stale [#257]. caches used: "
                       f"{[c.name for c in caches]}")
    lib_dir = EpochPaths(ctx.run_dir, ctx.epoch - 1).promoted
    if not lib_dir.is_dir():
        return
    assess_delivery = _load_delivery_assessor()
    if assess_delivery is None:
        raise Halt("cannot assess whether the surface shows the previous version's skills: no "
                   "delivery assessor is importable. Refusing to pass a check that did not run "
                   "— the whole point of #257 is that three agreeing signals were all unread.")
    r = assess_delivery(lib_dir, out)
    if r["rollouts"] and r["consulted"] == 0:
        raise Halt(
            f"{out.name} contains ZERO skill-tool calls across {r['rollouts']} rollouts, but the "
            f"rollouts it claims to describe ran with {lib_dir.name} loaded. Either the surface is "
            f"built from the wrong caches [#257] or the library truly was not delivered [#234] — "
            f"and those have opposite meanings, so resolve it against the raw transcripts in "
            f"{[c.name for c in caches]} before spending an evolver on this surface.")


def _run_build_digests_ssbv2(ctx: Ctx) -> None:
    """SSB v2's evidence surface, via scripts/build_rollout_digests.py.

    WHY A SECOND BRANCH RATHER THAN A GENERALISED ONE. The two benchmarks put their evidence in
    genuinely different places: tau3's builder reads PER-SAMPLE cache dirs plus an oracle tasks
    file, SSB v2's reads ONE cache dir plus a `_snapshots` tree holding the produced workbooks —
    because for SSB v2 the artifact under review is a spreadsheet, not a conversation. Forcing one
    signature over both would mean a function whose parameters are half-ignored depending on a
    flag, which is how the tau3 assumptions got baked in invisibly the first time.

    THIS IS THE FOURTH PIECE OF tau3-SHAPING IN THIS FILE, after `benchmark` (hardcoded into the
    frozen spec), `_pid_is_runner` and `_live_runners` (a literal process name). It was the one
    that actually cost time: the loop HALTed at step 0 with "no digest_caches declared", which
    reads like a missing config line and is in fact a missing implementation.
    """
    out = ctx.paths.digests
    if out.is_dir() and any(out.glob("*.md")):
        return                                       # idempotent
    if ctx.epoch >= 2:
        # #307: epoch N's surface digests epoch N-1's rollouts, and the loop itself
        # CREATED that cache (one dir per epoch, exactly one skill-version bucket —
        # the reason per-epoch dirs exist). Derive it; the RUNTIME `digest_caches`
        # declaration names the BASE legs' shared 17-bucket dir and is correct only
        # for epoch 1, so trusting it here re-digests the wrong version's evidence.
        cache = REPO / ".cache" / f"{ctx.spec.name}_e{ctx.epoch - 1}_cache"
        if not cache.is_dir():
            raise Halt(f"expected the previous epoch's cache at {cache} and it is absent — "
                       f"cannot build the e{ctx.epoch} surface from nothing")
        # Must mirror the runner's TASKS_CACHE = .cache/ssbv2_{TAG}_tasks
        # (run_ssbv2_base30_dsv4f_n3.py) under the leg tag scheme {name}_v{N}_{key};
        # the surface is train-only, so only the train leg's workbooks are needed.
        snaps = (REPO / ".cache" /
                 f"ssbv2_{ctx.spec.name}_v{ctx.epoch - 1}_train_tasks" / "_snapshots")
    else:
        cache = os.environ.get("EVOSKILL_DIGEST_CACHE") or (ctx.digest_caches[0]
                                                            if ctx.digest_caches else "")
        if not cache:
            raise Halt(
                "no digest cache declared for the SSB v2 evidence surface. Set `digest_caches` in "
                "RUNTIME.json to the ONE run-cache dir the base/skill leg wrote into (SSB v2 uses a "
                "single dir, not tau3's per-sample dirs). Naming it is not optional: the default is a "
                "per-tag dir that is EMPTY for a new tag, and an empty cache builds an empty surface.")
        snaps = Path(str(cache).replace("_cache", "_tasks")) / "_snapshots"
    if not snaps.is_dir():
        raise Halt(f"no snapshots at {snaps}. SSB v2 digests need the PRODUCED WORKBOOKS, which "
                   f"live beside the run cache in <tag>_tasks/_snapshots — the cache holds "
                   f"transcripts only. Point digest_caches at a leg whose task dirs survive.")
    env = dict(os.environ)
    env.update({
        "EVOSKILL_DIGEST_CACHE": str(cache),
        "EVOSKILL_DIGEST_SNAPS": str(snaps),
        "EVOSKILL_DIGEST_OUT": str(out),
        "EVOSKILL_DIGEST_SPLIT_FILE": str(ctx.spec.split_path),
        "EVOSKILL_DIGEST_SPLIT_KEY": "train",        # train-only surface, by construction
        "EVOSKILL_SOLVER_MODEL": ctx.spec.solver_model,
        "EVOSKILL_DIGEST_LEG": "base" if ctx.epoch == 1 else "skill",
    })
    # THE BUCKET MUST BE NAMED WHEN THE CACHE HOLDS MORE THAN ONE. The builder refuses otherwise
    # — "refusing to mix skill versions" — and it is right to: a surface built across two skill
    # versions is evidence about no single agent. The shared base cache holds 17 buckets, so
    # epoch 1 declares which. Epochs 2+ read a PER-EPOCH cache dir written by that epoch's own
    # skill leg, which therefore holds exactly one bucket and needs no declaration.
    bucket = os.environ.get("EVOSKILL_DIGEST_BUCKET") or (
        ctx.digest_bucket if ctx.epoch == 1 else "")
    if bucket:
        env["EVOSKILL_DIGEST_BUCKET"] = bucket
    if ctx.epoch > 1:
        env["EVOSKILL_DIGEST_SKILLS"] = str(EpochPaths(ctx.run_dir, ctx.epoch - 1).promoted)
    r = subprocess.run([sys.executable, "scripts/build_rollout_digests.py"],
                       cwd=str(REPO), env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise Halt(f"the SSB v2 digest builder exited {r.returncode}: "
                   f"{(r.stderr or r.stdout)[-600:]}")
    if not any(out.glob("*.md")):
        raise Halt(f"the digest builder exited 0 but wrote no *.md into {out}. Exit 0 is not "
                   f"evidence of an artifact.\n{(r.stdout or '')[-400:]}")


def _run_build_digests(ctx: Ctx) -> None:
    """Build the train-only evidence surface by invoking the SAME script the operator runs.

    Deliberately a subprocess call to scripts/tau3_build_digests.py rather than a reimplementation:
    that script carries the split firewall, the #219 redaction rule and the leg-identity assertion,
    and a second copy of that logic would drift from it (the #154 shape — two guards, one real).

    DISPATCHES ON BENCHMARK. SSB v2's surface is built from a different script reading different
    inputs; see _run_build_digests_ssbv2.
    """
    if getattr(ctx.spec, "benchmark", "") == "ssbv2":
        return _run_build_digests_ssbv2(ctx)
    out = ctx.paths.digests
    if out.is_dir() and any(out.glob("*.md")):
        _assert_surface_is_current(ctx, _digest_caches_for_epoch(ctx))
        return                                       # idempotent
    spec = ctx.spec
    caches = _digest_caches_for_epoch(ctx)           # #257: per EPOCH, not per run
    # THE VARIANT DESCRIBES THE ROLLOUTS BEING DIGESTED, NOT THIS RUN'S DECLARATION. It names
    # the cache dir (it is NOT in the cache key), and the intake does not yet capture it — the
    # leg script pins its own — so a spec frozen from a brief that never said "bm25_grep"
    # carries variant="" and this env var reached the builder empty (refused, 2026-08-17 smoke,
    # epoch 1 step 1). Read what the runner RECORDED in the leg result json instead (#194: never
    # re-derive what the producing process wrote). Spec, when set, still wins — but only if it
    # AGREES with the record; a disagreement means the surface would describe a different
    # instrument than declared, which is a HALT, not a preference.
    variant = getattr(spec, "variant", "") or ""
    src_leg = (Path(ctx.base_leg) if ctx.epoch <= 1 and ctx.base_leg
               else EpochPaths(ctx.run_dir, ctx.epoch - 1).leg("train") if ctx.epoch > 1
               else None)
    if src_leg is not None and src_leg.is_file():
        recorded = json.loads(src_leg.read_text()).get("retrieval_variant", "") or ""
        if variant and recorded and variant != recorded:
            raise Halt(f"spec.variant={variant!r} but {src_leg.name} records "
                       f"retrieval_variant={recorded!r} — refusing to build a surface that "
                       f"describes a different instrument than the spec declares")
        variant = variant or recorded
    env = dict(os.environ)
    env.update({
        "EVOSKILL_TAU3_DOMAIN": spec.domain,
        "EVOSKILL_TAU3_VARIANT": variant,
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


def _run_evolver(ctx: Ctx) -> None:
    """Dispatch the evolver for this epoch by invoking scripts/tau3_evolver.sh.

    THE LOOP DOES NOT COMPOSE THE PROMPT AND DOES NOT JUDGE THE OUTPUT. The prompt is written by
    a person and countersigned at the operator gate, which sits BEFORE this step and therefore
    cannot be reached without it. What happens here is purely mechanical: hand an already-approved
    prompt to an already-chosen model, under an isolation boundary the script owns.

    THE QUOTA GATE IS NOT OPTIONAL AND IS NOT INHERITED. Launching a new evolver at or above 0.96
    five-hour utilisation risks a HARD KILL with no overflow, and the hand launchers set
    EVOSKILL_TAU3_SKIP_QUOTA_GATE=1 precisely because a human ran the gate first. The loop IS
    that human here, so it runs the gate itself and refuses on anything but a fresh CLEAR — and
    it deliberately does NOT set the skip flag, leaving the script's own gate armed behind it.

    ONE EVOLVER AT A TIME. Evolvers are strictly serialised across arms; two share one five-hour
    window and neither reading would then describe the machine either of them is running on.

    NEVER RETRIED. A contaminated evolver run re-run is contaminated again. `advance()` calls a
    runner once and then re-verifies; if the skills that come back do not load, that is a HALT for
    a person, not a loop to spin. Only INFRA failures may ever justify a rerun, never quality.
    """
    st = ctx.paths.staging
    if st.is_dir() and _skill_fingerprint(st):
        return                                       # idempotent: this epoch already evolved

    # SERIALISATION HOLDS FOR EVERY EVOLVER, INCLUDING THE UNGATED ONES. Its original stated
    # reason — "they share one five-hour quota window" — is only true of Claude-powered evolvers,
    # and #304 makes that visible. The check stays for the reasons that survive: two evolvers
    # contend for the same machine and the same seat pool, and #213 is a peer-artifact hazard
    # between concurrent arms, neither of which cares which provider serves the model.
    live = subprocess.run(["pgrep", "-f", "tau3_evolve[r]"], capture_output=True, text=True)
    if live.stdout.strip():
        raise Halt(f"another evolver is live (pids {live.stdout.split()}); evolvers are strictly "
                   f"serialised — they contend for one machine, one seat pool, and (when both "
                   f"are Claude-powered) one five-hour window")

    _claude_quota_gate("a NEW evolver", getattr(ctx.spec, "evolver_model", ""))

    prompt = ctx.paths.prompt
    if not prompt.is_file() or not prompt.read_text().strip():
        raise Halt(f"no usable prompt at {prompt}")

    from .quota_scope import consumes_claude_quota, subscription_env

    env = dict(os.environ)
    # A first-party Claude evolver must not inherit an OpenRouter/proxy base URL left over from
    # solver work; a non-Claude evolver reaches its model THROUGH that proxy and must keep it.
    #
    # #304: this predicate used to be an inline regex here and a nonexistent one in the launcher.
    # It is the SAME question the quota gate asks — "does this model run on the subscription or
    # through a gateway" — so it is now asked once, in quota_scope, and used at both sites. Two
    # copies of one rule is #147's shape and they drift silently.
    _on_subscription = consumes_claude_quota(ctx.spec.evolver_model)

    # #313. CALLED UNCONDITIONALLY, BECAUSE IT BRANCHES INTERNALLY AND THIS DID NOT.
    #
    # This line stood inside `if _on_subscription:` for about ninety minutes, which meant the
    # gateway half of subscription_env() — the half added by #311 specifically to strip
    # ANTHROPIC_API_KEY so `claude` cannot present an Anthropic credential to OpenRouter — was
    # UNREACHABLE FROM HERE. The DSV4F evolver dispatch then failed exactly as #311 had, with
    # OpenRouter's 404 rendered by the CLI as "There's an issue with the selected model
    # (deepseek/deepseek-v4-flash-0731). It may not exist or you may not have access to it."
    #
    # WHY THE SELFTEST DID NOT CATCH IT, which is the part worth keeping. quota_scope's selftest
    # asserts subscription_env() in BOTH directions and passes — the function was always right.
    # What nothing asserted was that this CALLER reaches it on both paths. Testing the guard
    # instead of the call site is #154 and #248's shape, and I reproduced it inside the fix for
    # a bug of the same family, one hour after writing the comment that names the pattern.
    #
    # There is now a call-site test (selftest below, "evolver env routing") that builds the env
    # for a gateway model through THIS function and asserts the key is gone.
    env = subscription_env(ctx.spec.evolver_model, env)
    if _on_subscription:
        for k in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT"):
            env.pop(k, None)
    env.update({
        # #304. The launcher runs its OWN quota gate behind this one and must reach the same
        # verdict. It is told the answer rather than recomputing it, so the shell carries no
        # copy of the model-family rule; absent, the shell defaults to GATING (see the script).
        "EVOSKILL_EVOLVER_ON_CLAUDE_QUOTA": "1" if _on_subscription else "0",
        "EVOSKILL_TAU3_DOMAIN": ctx.spec.domain,
        "EVOSKILL_EVOLVER_MODEL": ctx.spec.evolver_model,
        # #309: keyed on the RUN DIR, not spec.name. Two runs sharing a name shared this dir.
        "EVOSKILL_TAU3_WORK_TAG": run_id(ctx),       # per-RUN isolated cwd
        # ABSOLUTE, and deliberately the OPPOSITE of what _run_leg passes. tau3_leg_retail.sh
        # builds `RUNDIR=$R/${EVOSKILL_TAU3_RUN_DIR}` so it needs a repo-RELATIVE path;
        # tau3_evolver.sh uses the value directly to form $LOG and then redirects inside a
        # `( cd "$WORK" && ... > "$LOG" )` subshell, so a relative path resolves against the
        # ISOLATED WORK DIR and the evolver's entire output stream lands nowhere. Two scripts,
        # two opposite contracts, one env var name. Fixing the leg side on 2026-08-09 left this
        # side taking whatever Ctx.run_dir happened to be — absolute in the hand-built drives,
        # relative once the CLI built the run dir, which is how it finally surfaced.
        "EVOSKILL_TAU3_RUN_DIR": str(Path(ctx.run_dir).resolve()),
        # #294. KERNEL CONTAINMENT FOR THE EVOLVER'S `claude` CHILD, opt-in, default OFF.
        # The launcher builds no profile of its own — it asks evolver_dispatch for the bytes —
        # and it sandboxes only the child, because the launcher itself must read the split file
        # for its oracle firewall and that file lists every held-out id. Default OFF because
        # this profile broke the AO in two unpredicted ways the first time anything real ran
        # under it (#291), and the evolver does strictly more than the AO.
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
        # so the dataset root holds the answers for all 240 tasks including every held-out
        # one. The hook's patterns knew tau3 paths and tau3 ids only, so on this run it
        # matched nothing and protected nothing; the launcher's deny-probe refused dispatch
        # (exit 13) rather than proceeding, which is why this was found before an evolver ran.
        **({"EVOSKILL_LEAK_DENY_PATHS": os.environ.get("SSBV2_ROOT", "")}
           if getattr(ctx.spec, "benchmark", "") == "ssbv2" else {}),
        "EVOSKILL_TAU3_PHOENIX_PROJECT": f"{run_id(ctx)}-e{ctx.epoch}",
    })

    log = ctx.run_dir / f"tau3_evolver_e{ctx.epoch}.log"
    # ── #249: THE EVOLVER HARNESS IS SELECTABLE ─────────────────────────────────────────────
    # `claude_cli` is byte-identical to what four epochs of this programme already ran, and it
    # DELEGATES to the script rather than reimplementing it — the script owns the org-key
    # refusal, the quota gate, the leak hooks, the seeded staging and the Phoenix tee, and a
    # second copy of any of that is #154's shape.
    #
    # `direct` routes to evolver_direct_runner.py under the deny-first sandbox profile, because
    # `src/harness/direct` has NO PreToolUse hook mechanism and would otherwise run with none of
    # the containment the CC path gets for free. That runner's selftest asserts a peer-arm read
    # in ITS OWN log format makes the real #213 audit FAIL — building the guard before the thing
    # it guards, which is the order #186 and #213 were not built in.
    _harness = getattr(ctx.spec, "evolver_harness", "claude_cli")
    if _harness == "claude_cli":
        argv = ["bash", "scripts/tau3_evolver.sh", str(ctx.epoch),
                str(prompt), str(ctx.paths.digests), str(st)]
    elif _harness == "direct":
        from src.orchestrator.evolver_dispatch import (EvolverJob,
                                                       build_evolver_sandbox_profile,
                                                       preflight)
        _job = EvolverJob(epoch=ctx.epoch, work_dir=ctx.run_dir / f"evolver_work_e{ctx.epoch}",
                          prompt="", model=ctx.spec.evolver_model, log_path=log,
                          prompt_path=Path(prompt),
                          # #250c is handled INSIDE build_evolver_sandbox_profile as a LITERAL
                          # grant on the log's parent, NOT as a subpath grant here. A subpath grant
                          # on the run dir would make its split file readable, and the split file
                          # LISTS ALL 47 TEST IDS -- widening containment to fix a plumbing bug.
                          grants=[Path(ctx.paths.digests), Path(st), Path(ctx.oracle)],
                          # #297. STAGING MUST BE WRITABLE, AND IT WAS NOT. It was declared as a
                          # read grant only, so `_wr_sub` held just the work dir and the repo-wide
                          # `(deny file-write* (subpath <repo>))` caught the one directory the
                          # evolver's entire deliverable goes into. Probed 2026-08-11: write to
                          # staging DENIED, write to work dir allowed, peer skills correctly
                          # denied. Latent only because this harness has never run in production
                          # (#229 held its spec wiring); the first real dispatch would have burned
                          # a full evolver and produced an empty staging dir, which #237 says is
                          # indistinguishable from "it decided to ship nothing" without a check.
                          writes=[Path(st), ctx.run_dir / f"evolver_work_e{ctx.epoch}"],
                          # #295 parity: deny the run dir BY NAME so containment does not depend
                          # on the run dir happening to sit under /Users.
                          hide=[Path(ctx.run_dir)],
                          no_write=[Path(ctx.run_dir), operator_dir(ctx.run_dir)],
                          timeout_s=ctx.evolver_timeout_s)
        _job.work_dir.mkdir(parents=True, exist_ok=True)
        # PREFLIGHT BEFORE SPENDING A DISPATCH. Four consecutive attempts each died on a different
        # missing grant, one per launch, because nothing enumerated the requirements up front.
        # preflight() probes exec, import, every declared read, and the log dir's is_dir() under
        # the REAL profile and reports ALL failures together. It must never widen the profile: a
        # denial means a narrow grant decided by a person.
        _gaps = preflight(_job)
        if _gaps:
            raise Halt("evolver preflight found " + str(len(_gaps)) + " containment gap(s) — "
                       "NOT dispatching, nothing spent:\n  - " + "\n  - ".join(_gaps) +
                       "\n  Grant each path NARROWLY (a literal for a file or a directory node, "
                       "a subpath only for a tree the evolver is entitled to). Do NOT widen to "
                       "the repo: its split file lists every held-out id.")
        _prof = ctx.run_dir / f"evolver_e{ctx.epoch}.sb"
        _prof.write_text(build_evolver_sandbox_profile(_job))
        argv = ["sandbox-exec", "-f", str(_prof),
                sys.executable, "-m", "src.orchestrator.evolver_direct_runner",
                "--epoch", str(ctx.epoch), "--work-dir", str(_job.work_dir),
                "--model", ctx.spec.evolver_model, "--log", str(log),
                "--prompt", str(prompt), "--timeout-s", str(ctx.evolver_timeout_s)]
        for g in _job.grants:
            argv += ["--grant", str(Path(g).resolve())]
        # The same receipt the CC launcher writes (#294), so _v_evolver_contained can see this
        # harness too rather than reporting UNMEASURED on a path that is always contained.
        (ctx.run_dir / f"e{ctx.epoch}_EVOLVER_CONTAINMENT.json").write_text(json.dumps(
            {"sandboxed": True, "profile": str(_prof), "work_dir": str(_job.work_dir),
             "harness": "direct", "containment": "sandbox-exec"}, indent=1))
    else:
        raise Halt(f"unknown evolver_harness {_harness!r}; known: claude_cli, direct")
    # Redirected to a FILE, never a pipe: caffeinate inherits stdout, and a pipe held open by a
    # short-lived foreground reader has killed an evolver launch before.
    with open(log, "wb") as fh:
        p = subprocess.Popen(argv, env=env, stdout=fh, stderr=subprocess.STDOUT, stdin=DEVNULL)
        if shutil.which("caffeinate"):  # macOS sleep inhibitor; absent on Linux
            subprocess.Popen(["caffeinate", "-i", "-w", str(p.pid)],
                             stdout=DEVNULL, stderr=DEVNULL)
        try:
            rc = p.wait(timeout=ctx.evolver_timeout_s)
        except subprocess.TimeoutExpired:
            raise Halt(f"evolver still running after {ctx.evolver_timeout_s}s (pid {p.pid}). It "
                       f"has NOT been killed — inspect {log.name} and decide. Killing a live "
                       f"evolver mid-write leaves a half-written skill set that looks complete.")
    if rc != 0:
        tail = log.read_text(errors="replace")[-500:]
        raise Halt(f"evolver exited {rc}. This is a refusal or a crash, not a result:\n  {tail}")

    # THE LAUNCHER IS NOT THE EVOLVER. tau3_evolver.sh ends with
    # `( cd "$WORK" && exec nohup claude -p ... ) &` — it DETACHES and returns in seconds, so the
    # wait above proves only that dispatch succeeded. On 2026-08-09 that made advance() report
    # "RAN + VERIFIED: skills load through the real consumer" 45 seconds after launch, while the
    # evolver was still writing. The verify passed because the script SEEDS staging with the
    # previous iteration's skills (`cp -R "$PREV_SKILLS"/. "$STAGING"/`) — so it was reading the
    # seed and calling it the output. Checking a proxy instead of the thing, inside the loop
    # written to stop exactly that, and it took driving a real arm to expose it.
    #
    # So wait on the evolver's OWN pid, by identity, exactly as _await_leg does for a rollout.
    pidf = ctx.run_dir / f"tau3_evolver_e{ctx.epoch}.pid"
    deadline = time.time() + ctx.evolver_timeout_s
    while not pidf.is_file() and time.time() < deadline:
        time.sleep(2)
    if not pidf.is_file():
        raise Halt(f"launcher exited 0 but wrote no {pidf.name}; there is no evolver to wait on "
                   f"and no way to tell a finished run from one that never started.")
    epid = int(pidf.read_text().strip())
    while _pid_is(epid, "claude"):
        if time.time() >= deadline:
            raise Halt(f"evolver (pid {epid}) still running after {ctx.evolver_timeout_s}s. It "
                       f"has NOT been killed — inspect {log.name} and decide. Killing a live "
                       f"evolver mid-write leaves a half-written skill set that looks complete.")
        time.sleep(15)


LEG_KEYS = ("train", "val")


def _dispatch_path(ctx: Ctx, key: str) -> Path:
    """Where a launched-but-unfinished leg records its pid, so a LATER PROCESS can find it.

    Not `<tag>.pid` — the launcher writes that and it survives the run that wrote it, so a stale
    one from an earlier attempt at the same tag would have us waiting on a dead stranger. This
    file is written by the dispatcher at launch, carries the pid AND the epoch it belongs to, and
    is deleted once the leg's artifact exists.
    """
    return operator_dir(ctx.run_dir) / f"{ctx.paths.leg(key).stem}.dispatch.json"


# THE ROLLOUT RUNNER'S NAME, PER BENCHMARK. It was the literal "run_tau3_banking" in two places
# (the liveness probe and the seat census), which is invisible while tau3 is the only benchmark
# and silently wrong the moment it is not: an SSB v2 leg would never match, so `_pid_is_runner`
# would call a healthy running leg dead and `_await_leg` would report it as having exited without
# writing. Keyed on ExperimentSpec.benchmark, which is already the field that distinguishes them.
RUNNER_PATTERN = {"tau3": "run_tau3_banking", "ssbv2": "run_ssbv2"}
DEFAULT_RUNNER_PATTERN = "run_tau3_banking"


def _runner_pattern(ctx=None) -> str:
    bm = getattr(getattr(ctx, "spec", None), "benchmark", "") or ""
    return RUNNER_PATTERN.get(bm, DEFAULT_RUNNER_PATTERN)


def _bracketed(pat: str) -> str:
    """`pgrep -f` matches its OWN command line, so an unbracketed pattern finds the pgrep itself.
    Bracketing the last character keeps the regex equivalent while excluding the pgrep."""
    return f"{pat[:-1]}[{pat[-1]}]" if pat else pat


def _live_runners(ctx=None) -> list:
    """Pids of rollout LEGS currently alive, for THIS benchmark — one pid per leg.

    COUNT LEGS, NOT PROCESSES. ssbv2_leg.sh backgrounds the runner inside a `bash -c` wrapper
    whose own command line necessarily contains the runner pattern, because _pid_is_runner_pat
    re-derives identity from `ps -o command=` and would otherwise declare a healthy leg dead. The
    consequence, hit live on 2026-08-12: ONE train leg presented as TWO runners, the seat guard
    concluded "two legs already fill 60 of 64 seats" and refused to launch val, and epoch 1
    silently serialised — the exact cost the concurrent-legs work existed to remove.

    It failed SAFE (a leg too few, never too many) and it said so in the log, which is why this
    took one tick to find rather than showing up as an unexplained slowdown.

    A wrapper is the PARENT of its runner, so dropping any matched pid that is also the parent of
    another matched pid leaves exactly one pid per leg. Parentage comes from `ps -o ppid=`, which
    needs no env access — unlike matching on EVOSKILL_RUN_TAG, which `pgrep -f` cannot see at all.
    """
    q = subprocess.run(["pgrep", "-f", _bracketed(_runner_pattern(ctx))],
                       capture_output=True, text=True)
    pids = [int(x) for x in q.stdout.split() if x.isdigit()]
    if len(pids) < 2:
        return pids
    parents = set()
    for pid in pids:
        r = subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True)
        pp = r.stdout.strip()
        if pp.isdigit():
            parents.add(int(pp))
    return [pid for pid in pids if pid not in parents] or pids


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
        if pid and rec.get("epoch") == ctx.epoch and _pid_is_runner(pid, ctx):
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
        dp.write_text(json.dumps({"pid": pid, "epoch": ctx.epoch, "key": key,
                                  "tag": out.stem, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}))
    except OSError as e:
        print(f"[leg] could not record the dispatch of {key} ({e}); it IS running as pid {pid}",
              file=sys.stderr)
    return pid


def _run_leg(ctx: Ctx, key: str) -> None:
    """Roll out one leg by invoking the SAME launcher script an operator would run.

    BOTH LEGS ARE LAUNCHED TOGETHER, AND THIS CALL WAITS ONLY FOR ITS OWN (2026-08-12, the maintainer).
    Until now the train step ran train to completion and the val step then ran val, so an epoch
    paid train+val ≈ 78 min of wall-clock for two rollouts that depend on nothing but the
    promoted skill set. The serialization was an artifact of `epoch_steps()` being a linear list,
    not of the data: `_run_author_prompt` reads `EpochPaths(run_dir, epoch-1).leg("train")` and
    never touches val, so val is a LEAF — a gate measurement with nothing downstream of it.

    The seat pool was built for exactly this overlap and its own comment says so: 64 seats "so
    train and val can run CONCURRENTLY drawing from the same 64 rather than taking 64 each", with
    train priority 10 above val's 5. Two legs at conc 30 is 60 of 64. The loop was defeating its
    own admission policy.

    WHY THIS DOES NOT CHANGE WHAT AN EPOCH MEASURES, checked rather than assumed. Concurrency is
    not an ExperimentSpec field and not in `_INSTRUMENT`, so it is in neither the cache identity
    nor the frozen config. The one way load COULD reach a score is by pushing episodes into the
    5400 s wall — and across every banking leg to date `episodes_timeout_scored_zero` is 0,
    `episodes_excluded_infra` is 0, and every termination is `user_stop` or `max_steps` (the
    agent-turn cap). The wall has never been what ended an episode. `_v_no_infra_exclusions` and
    the outcome-reconcile steps check that again every epoch, so if load ever does start ending
    episodes the run HALTs rather than quietly measuring something else.

    THE SEAT GUARD IS NOW A CEILING, NOT A MUTEX. It used to refuse if ANY runner was live, which
    is what made concurrency impossible; it now refuses at three, which is the number the 64-seat
    pool cannot absorb.
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
    _await_leg(out, mine, ctx.leg_timeout_s, key, pattern=_runner_pattern(ctx))
    try:
        _dispatch_path(ctx, key).unlink()
    except OSError:
        pass


def _leg_argv_env(ctx: Ctx, key: str):
    """(argv, env) for one leg. Split out of _run_leg so dispatch and wait are separable.

    A subprocess call, not a reimplementation, for the reason given at _run_build_digests: the
    launcher owns the org-key refusal, the provider-pin assertion, the seat pool and the cache
    identity, and a second copy of any of that is the #154 shape (two guards, one real).

    THE TAG IS THE CONTRACT. The launcher writes its result to `<run_dir>/<TAG>.json`, and
    EpochPaths.leg() looks for `<run_dir>/<run_dir.name>_v<epoch>_<key>.json`. Deriving the tag
    from the same expression is what makes the runner and the verify agree; a hand-picked tag
    (a1v1_train, dsv4fevo_v1_train, ...) is exactly the drift that made run_epoch() silently
    SKIP six of ten invariants the first time this module met a run it had not been written for.

    IT DOES NOT WAIT. `_await_leg_of` does, once both legs are in flight; see _run_leg for why
    they are launched together.
    """
    out = ctx.paths.leg(key)

    # SEAT SAFETY, AS A CEILING RATHER THAN A MUTEX. This used to refuse if ANY runner was live,
    # which made the intended train||val overlap impossible — the loop enforcing a stricter rule
    # than the seat pool it delegates admission to. Two legs at conc 30 is 60 of 64 seats and is
    # the designed case; a THIRD would oversubscribe the pool, and a leg finishing under
    # contention has a different wall-clock profile from its comparator, which is an instrument
    # difference nobody records. So: two is fine, three is refused.
    live = _live_runners(ctx)
    if len(live) >= 2:
        raise Halt(f"{len(live)} rollout runners are already live (pids {live}); refusing to "
                   f"launch {key} as a third — two legs at conc 30 fill 60 of the 64 seats and a "
                   f"third oversubscribes the pool. Wait for one to finish.")

    tag = out.stem                                   # == f"{run_dir.name}_v{epoch}_{key}"
    script = ctx.leg_script
    if not Path(script).is_file():
        raise Halt(f"no leg launcher at {script}")
    # REPO-RELATIVE, NOT ABSOLUTE. tau3_leg_retail.sh builds `SHARED=$R/${EVOSKILL_TAU3_SHARED_DIR}`
    # and `RUNDIR=$R/${EVOSKILL_TAU3_RUN_DIR}`, so an absolute path double-prefixes into
    # `/Users/.../EvoSkill//Users/.../EvoSkill/outputs/...`. The launcher's own `[ -d "$SHARED" ]`
    # guard caught it and exited 4 before spending anything, which is the guard earning its keep —
    # but the loop should not be handing it a path it cannot use.
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
        # #251. THE SPEC'S SAMPLE COUNT MUST REACH THE LEG, or spec.n_samples is decorative.
        # Every leg script defaults to `${EVOSKILL_N_SAMPLES:-3}`, and the loop never set it, so
        # an orchestrator-driven leg ran N=3 whatever the spec said. On the full banking run
        # spec.n_samples is ALSO 3, so the two agree by coincidence and the defect is invisible
        # there — it surfaced only on a smoke spec asking for N=1, whose v1 train leg dispatched
        # 9 episodes against a 3-episode base. An unmatched n on the two sides of a paired gate is
        # not a smaller effect, it is a different comparison. Pass it explicitly.
        "EVOSKILL_N_SAMPLES": str(ctx.spec.n_samples),
        # #320: THE SPEC'S SOLVER MODEL MUST REACH THE LEG, or spec.solver_model is decorative —
        # the leg script defaults to deepseek-v4-flash-0731, and every run to date used exactly
        # that model, so the missing export was invisible (#251's shape, second occurrence).
        # Caught pre-launch on the first run whose spec names a DIFFERENT model
        # (ssbv2_selfevo_pro): every leg would have silently rolled the Flash solver under a
        # Pro-named spec, into a Pro-named cache bucket.
        "EVOSKILL_SOLVER_MODEL": ctx.spec.solver_model,
        "EVO_CONC": "30", "EVO_CONC_MAX": "30", "EVO_PROBE_MAX": "30", "EVO_SEATS": "64",
        # #236: name the SOLVER leg's Phoenix project, so one epoch's solver and evolver spans
        # are findable together. The evolver already gets `<spec>-e<N>` (see _run_evolver); this
        # is the same stem plus the leg key, giving `<spec>-e<N>-train` / `-val` beside it.
        # Until now the loop set only the evolver's, and each leg script picked its own
        # timestamped name via a BARE export — so the solver half of an epoch was reachable only
        # by guessing a minute, and on retail it was additionally stamped with the BANKING
        # launcher's prefix. Both leg-script defects are fixed; this is the naming side.
        "PHOENIX_PROJECT_NAME": f"{run_id(ctx)}-e{ctx.epoch}-{key}",
    })
    if getattr(ctx.spec, "benchmark", "") == "ssbv2":
        # ONE CACHE DIR PER EPOCH. The SSB v2 runner defaults to a per-TAG dir, which is empty for
        # a new tag and silently re-buys every episode (caught 2026-08-12 after two from-scratch
        # legs). A per-EPOCH dir shared by that epoch's train and val legs gets the replay AND
        # leaves the digest builder exactly one skill-version bucket to read.
        env["EVOSKILL_CACHE_DIR"] = str(REPO / ".cache" /
                                        f"{ctx.spec.name}_e{ctx.epoch}_cache")
    # The launcher DETACHES and echoes `launched <tag> pid=<P> ...`; _dispatch_leg reads the pid
    # from that line rather than from <tag>.pid, which may be stale from an earlier attempt at
    # the same tag and would have the waiter asserting against a dead stranger.
    return (["bash", script, tag, key, ctx.paths.promoted.name], env)


def _pid_is(pid: int, needle: str) -> bool:
    """Is `pid` alive AND still the process we think it is? Identity re-derived from its own
    command line every call, because a pid outlives its owner and gets reused."""
    q = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True)
    return needle in q.stdout


def _pid_is_runner_pat(pid: int, pattern: str) -> bool:
    """Liveness+identity against an EXPLICIT pattern, for callers that hold no ctx."""
    q = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True)
    return pattern in q.stdout


def _pid_is_runner(pid: int, ctx=None) -> bool:
    """Is `pid` alive AND still the rollout runner we launched?

    THE PATTERN IS PER-BENCHMARK (see RUNNER_PATTERN). It was the literal
    "run_tau3_banking", which would call a healthy SSB v2 leg dead — and `_await_leg` reports a
    process it cannot see as one that "exited without writing", so the epoch would HALT on a leg
    that was running perfectly.

    Both halves are load-bearing. `kill(pid, 0)` alone answers "alive", and over a leg that runs
    for an hour the pid can be recycled onto an unrelated process — we would then wait forever on
    a stranger. Identity is re-derived from the process's own command line each time, which is
    the same lesson as verifying a KILL by pid rather than by a pgrep pattern.
    """
    q = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True)
    return _runner_pattern(ctx) in q.stdout


def _await_leg(out: Path, pid: int, timeout_s: int, key: str, poll_s: float = 15.0,
               pattern: str = DEFAULT_RUNNER_PATTERN) -> None:
    """Block until the leg is BOTH written and fully exited.

    WHY EXIT, AND NOT JUST THE ARTIFACT. run_tau3_banking.py writes its result JSON seven lines
    before it returns: write -> print -> return -> asyncio teardown -> atexit span flush. So
    there is a window in which the artifact exists and the process is still in `pgrep` output.
    Returning inside that window is harmless on its own, but the very next step of the same epoch
    is the other leg, whose seat guard pgreps for exactly that pattern — the loop would refuse to
    run val because train had not finished exiting, on a correct epoch, nondeterministically.
    MEASURED, NOT ASSUMED: 0.53s on the Arm-2 epoch-1 val leg (pid 17415, 2026-08-09), by polling
    artifact-existence and process-liveness at 0.25s against the live run. Small, and entirely
    sufficient — the seat guard fires the instant the next step starts, so a val leg launched on
    artifact-existence would have seen that pid and refused a correct epoch. The fix does not
    depend on the number: waiting on identity removes the class rather than budgeting a margin
    against a window whose width varies with span-flush and teardown time.

    Ordering is deliberate: wait for the process, THEN judge the artifact. A runner that exited
    without writing has FAILED, and that is a different message from one still running.
    """
    # The pid is tested BEFORE the deadline, so an already-finished leg is never reported as
    # "still running" just because the clock had run out — the artifact, not the timer, decides.
    deadline = time.time() + timeout_s
    while _pid_is_runner_pat(pid, pattern):
        if time.time() >= deadline:
            raise Halt(f"{key} leg (pid {pid}) still running after {timeout_s}s. Do NOT relaunch "
                       f"— read failures from the PARTIAL json's `failed_episodes` key, never "
                       f"the log (it prints only the first 15 and truncates provider messages).")
        time.sleep(max(0.0, min(poll_s, deadline - time.time())))
    if not out.is_file():
        raise Halt(f"{key} leg (pid {pid}) EXITED WITHOUT WRITING {out.name}. This is a failed "
                   f"leg, not a slow one; relaunching without a diagnosis repeats it.")


# A verify that CANNOT be evaluated yet is not a pass. It returns True (so the loop advances —
# an unmeasurable check must not block a finished epoch) but prefixes its detail with this, and
# report() counts and renders it separately. Without this, `_v_skills_were_consulted` at the last
# epoch — where no next-epoch digest surface exists — printed under "13 pass, 0 HALT" and read
# exactly like the three epochs that really did measure 93% uptake. A check that reports success
# without having checked is the #154 shape; this is the smallest honest fix that keeps the
# (bool, str) contract every other verify already uses.
UNMEASURED = "UNMEASURED — "


def epoch_steps() -> list:
    return [
        Step("evidence surface is train-only", _v_digests_train_only, run=_run_build_digests,
             defect="#186"),
        # #253. THE AO AUTHORS THE PROMPT. Placed after the evidence surface (it needs the digests)
        # and before every prompt CHECK, so the checks guard a model author exactly as they guarded
        # a human one. Authorship moved; verification did not.
        Step("evolver prompt authored by the AO", _v_prompt_authored,
             run=_run_author_prompt, defect="#253"),
        # #301. Immediately after authoring, because that is the first moment both sides have
        # necessarily spoken. No runner: the turns are written by the channels themselves, and a
        # step that could "fix" a transcript by writing one would be fabricating a conversation.
        Step("operator<->AO conversation is recorded", _v_conversation_recorded, defect="#301"),
        Step("evolver prompt numbers re-derive", _v_prompt_numbers, defect="#208"),
        Step("prompt names no task id, no held-out figure", _v_prompt_no_leakage, defect="#253"),
        Step("prompt is raw information, not analysis", _v_prompt_not_analytical, defect="#262"),
        Step("prompt + skills proofread by a person", _v_operator_signature,
             operator_gate=True),
        Step("skills load through the real consumer", _v_skills_load, run=_run_evolver,
             defect="#214"),
        Step("evolver ran under declared containment", _v_evolver_contained, defect="#294"),
        Step("evolver read nothing forbidden", _v_evolver_read_nothing_forbidden,
             defect="#213/#186"),
        Step("promoted set matches staging", _v_promoted_matches_staging, run=_run_promote),
        Step("train leg loaded what was promoted",
             lambda c: _v_leg_loaded_promoted(c, "train"),
             run=lambda c: _run_leg(c, "train"), defect="#214"),
        Step("val leg loaded what was promoted",
             lambda c: _v_leg_loaded_promoted(c, "val"),
             run=lambda c: _run_leg(c, "val"), defect="#214"),
        # #251. BEFORE any outcome is read, establish that the two sides of the gate are the same
        # comparison. Placed here deliberately: a pass-rate delta computed across unequal n is
        # not a weaker result, it is a meaningless one, so this must gate the reconcile steps
        # rather than sit beside them.
        Step("train pairs comparably with base", lambda c: _v_legs_comparable(c, "train"),
             defect="#251"),
        Step("val pairs comparably with base", lambda c: _v_legs_comparable(c, "val"),
             defect="#251"),
        Step("train: no infra exclusions", lambda c: _v_no_infra_exclusions(c, "train")),
        Step("val: no infra exclusions", lambda c: _v_no_infra_exclusions(c, "val")),
        Step("the agent actually consulted the skills", _v_skills_were_consulted, defect="#234"),
        Step("train outcomes reconcile", lambda c: _v_outcomes_reconcile(c, "train"),
             defect="#209"),
        Step("val outcomes reconcile", lambda c: _v_outcomes_reconcile(c, "val"),
             defect="#209"),
        # LAST, because three of the four projects it looks for do not exist until their leg has
        # run. No runner: the traces are written by the parties that produced them, and a step
        # that could "fix" this by emitting something would be manufacturing the evidence.
        Step("every agent's trace reached Phoenix", _v_traces_in_phoenix, defect="#301"),
    ]


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

    #275. The unpack `ok, detail = r` used to sit OUTSIDE _safe_verify's try/except, so a verify
    that returned None, or a tuple of the wrong arity, raised TypeError/ValueError straight past
    evaluate(), advance() and main() -- contradicting the docstring one line above it ("a verify
    that CRASHES is a HALT, never a pass") and reaching the operator as a raw traceback. The
    selftest covered only a verify that RAISES, never one that returns garbage, and forgetting a
    `return` is the more common mistake of the two.
    """
    if isinstance(r, tuple) and len(r) in (2, 3):
        ok = bool(r[0])
        detail = r[1] if isinstance(r[1], str) else str(r[1])
        question = (r[2] if len(r) == 3 else "") or ""
        question = question if isinstance(question, str) else str(question)
        if ok and question:
            # #276. `if ok: continue` was evaluated before `if question:` in BOTH advance() and
            # evaluate(), so a verify returning (True, detail, question) passed with the question
            # thrown away -- no block, no record, no trace anywhere. A check that passes has
            # nothing to ask, so the combination is a contradiction and is reported as one.
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

    IT NEVER RAISES Blocked EITHER (#277). It used to let Blocked propagate, on the reasoning
    that "I need an answer" must not be reportable as "this check failed" — but evaluate() is the
    BOARD, not the actor, and the effect was that the first step to ask a question aborted the
    whole report, so the operator could not see what else was already true. Nothing was gained:
    the board's rows are advisory, and advance() — which is where a block actually stops work —
    still raises. A blocked row is rendered as ASK, distinct from both ok and HALT.

    Each row is (step, ok, detail, question). The 4th element is new; `report()` and the selftest
    read rows positionally and tolerate both widths.
    """
    out = []
    halt = None
    for s in steps or epoch_steps():
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

    #279, and it is the defect the whole block channel was one step away from shipping with.
    advance() used to raise `Blocked(qid=s.name)` directly for a returned question, and main()
    then printed "put {"<step name>": "your answer"} in ANSWERS_e{N}.json, then re-run". THAT
    INSTRUCTION WAS FALSE: only ask() reads the answers file, and a verify that merely RETURNS a
    question never calls ask(). The operator would write the answer, re-run, and get the identical
    block forever — a clarification loop that cannot terminate, which this codebase already knows
    is worse than a refusal (cli.clarify_loop bounds its rounds for exactly that reason).

    Routing through ask() makes the printed procedure true: ask() reads ANSWERS_e{N}.json under
    the SAME key main() tells the operator to write, blocks when it is absent, enforces the #271
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

    THREE PROPERTIES, EACH PAID FOR BY A REAL FAILURE:

    1. IT ADVANCES ON THE VERIFY, NEVER ON THE RUN. `s.run(ctx)` returning without raising
       proves only that a subprocess exited 0. On 2026-08-08 four separate defects each reported
       success — an evolver that "wrote 5 skills" the loader read as 1, a run that "succeeded"
       having read the comparator arm's answers, a process that "launched" and was dead in
       200ms, a split "built with 45 tasks" of which 36 did not exist. Every one of them would
       pass an exit-code check. So the run's return value is discarded and the verify is asked
       again, from the artifact, through the artifact's real consumer.

    2. A STEP THAT RAN AND STILL FAILS IS A HALT, NOT A RETRY. Re-running an evolver whose
       evidence was contaminated just produces contaminated output again.

    3. ONE STEP PER INVOCATION. The operator sees each transition. A driver that runs an epoch
       end-to-end unattended catches only DECLARED failures, and every defect that has mattered
       in this programme was undeclared.
    """
    for s in steps or epoch_steps():
        ok, detail, question = _safe_verify(s, ctx)
        if question:
            ok, detail = _resolve_question(ctx, s, detail, question)
        if ok:
            continue
        if s.operator_gate:
            raise OperatorGate(f"{s.name}: {detail}\n"
                       f"  This gate is deliberately not automatable. It wants a human to have "
                       f"read the prompt and the skills, and to countersign what they checked. "
                       f"Auto-satisfying it would turn 'a person read this' into 'a person was "
                       f"assumed to have read this', which is how an optimizer acquires a hidden "
                       f"author.")
        if s.run is None:
            raise Halt(f"{s.name}: {detail}\n"
                       f"  This step has no runner — it is performed elsewhere (by a human or "
                       f"another process) and the loop only checks it.")
        # #278. `s.run(ctx)` used to be unguarded while main() caught only Blocked and Halt, so
        # every operator-facing refusal a runner raises as a plain exception -- and the prompt
        # author raises ValueError for all six of its own ("AO_SESSION.json is unreadable, fix or
        # delete it deliberately", "no prompt policy", "refusing to overwrite an authored prompt",
        # a non-zero exit, exit 0 with no file, a too-short prompt) -- reached the operator as a
        # raw traceback with exit 1 instead of the HALT block that explains what to do. Blocked is
        # a BaseException and passes straight through, which is the point of #264.
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

    ASK IS ITS OWN MARK, NOT A HALT (#280). A step that needs an answer used to render as a plain
    HALT with the question dropped entirely, because main() discarded evaluate()'s halt tuple and
    that tuple was the only carrier — so on `--plan` "I need to know something" was
    indistinguishable from "something is broken", which is the precise confusion Blocked exists to
    remove, reappearing in the renderer.
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
def selftest() -> int:
    import shutil
    import tempfile
    print("SELFTEST — each verify must FAIL on a known-bad input\n")
    from src.orchestrator.spec import tau3_banking_2525
    spec = tau3_banking_2525()
    fails = 0

    def check(label, got_ok, want_ok, detail):
        nonlocal fails
        ok = got_ok is want_ok
        print(f"  [{'ok' if ok else 'BAD'}] {label}\n        {detail[:130]}")
        fails += 0 if ok else 1

    with tempfile.TemporaryDirectory() as td:
        rd = Path(td) / "run"
        rd.mkdir()
        ctx = Ctx(rd, 2, spec, rd / "oracle")

        # skills_load: flat files must FAIL, dirs must PASS
        st = ctx.paths.staging
        st.mkdir(parents=True)
        for n in ("a", "b", "c"):
            (st / f"{n}.md").write_text("---\nname: x\ndescription: y\n---\nbody\n")
        ok, d = _v_skills_load(ctx)
        check("flat .md files in staging root -> HALT", ok, False, d)
        for f in st.glob("*.md"):
            f.unlink()
        for n in ("a", "b"):
            (st / n).mkdir()
            (st / n / "SKILL.md").write_text(f"---\nname: {n}\ndescription: when x\n---\nbody\n")
        ok, d = _v_skills_load(ctx)
        check("proper <name>/SKILL.md dirs -> pass", ok, True, d)

        # missing description frontmatter must FAIL
        (st / "b" / "SKILL.md").write_text("---\nname: b\n---\nbody\n")
        ok, d = _v_skills_load(ctx)
        check("SKILL.md with no description -> HALT", ok, False, d)
        (st / "b" / "SKILL.md").write_text("---\nname: b\ndescription: when x\n---\nbody\n")

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

        # #304/#305: a JSON with NO skills_dir must never read as "loaded ." —
        # Path("") is truthy and equals Path("."), which is how the old guard
        # converted an absent field into a fake cwd. Absence now falls back to
        # the leg log's [SKILL] banner, or HALTs when there is no log either.
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

        # #251 KEY ROUTING. The regression this exists to catch: an earlier cut of
        # _v_legs_comparable derived the val comparator by filename substitution and, when that
        # file did not exist (base legs living in ANOTHER run's dir, which is exactly the
        # tau3_banking_ao2 case), fell back to base_leg — the TRAIN base. Comparing a val leg to
        # the train base trips the "different task sets" FAIL on a perfectly good run, and a
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

        # operator gate: absent, then stale, then valid
        ctx.paths.prompt.write_text("# Iteration 2\n\nIt scored **25 of 75 episodes**.\n")
        ok, d = _v_operator_signature(ctx)
        check("no countersignature -> HALT", ok, False, d)
        sig = rd / f"e2_{SIGN_FILENAME}"
        sig.write_text(json.dumps({"checked": ["prompt", "skills"],
                                   "prompt_sha256": "0" * 64}))
        ok, d = _v_operator_signature(ctx)
        check("signature covers a DIFFERENT prompt version -> HALT", ok, False, d)
        sig.write_text(json.dumps({
            "checked": ["prompt numbers", "no benchmark leakage", "descriptions are triggers"],
            "prompt_sha256": hashlib.sha256(ctx.paths.prompt.read_bytes()).hexdigest()}))
        # A staging dir exists here, so a prompt-only signature must NOT satisfy the gate. Its
        # own message promises the skill set was read too, and until 2026-08-09 nothing enforced
        # that: the operator signs after proofreading the PROMPT, before the evolver has written
        # anything, and the signature then silently appeared to cover skills nobody had seen.
        ok, d = _v_operator_signature(ctx)
        check("prompt-only signature, staging exists -> HALT", ok, False, d)
        good_sig = {
            "checked": ["prompt numbers", "no benchmark leakage", "descriptions are triggers"],
            "prompt_sha256": hashlib.sha256(ctx.paths.prompt.read_bytes()).hexdigest(),
            "staging_sha256": staging_sha256(ctx.paths.staging)}
        sig.write_text(json.dumps(good_sig))
        ok, d = _v_operator_signature(ctx)
        check("signature covering prompt AND skills -> pass", ok, True, d)

        # And it must not outlive an edit to the skills either.
        (ctx.paths.staging / "a" / "SKILL.md").write_text(
            "---\nname: a\ndescription: EDITED AFTER SIGNING\n---\nbody\n")
        ok, d = _v_operator_signature(ctx)
        check("skills edited after signing -> HALT", ok, False, d)
        sig.write_text(json.dumps({**good_sig,
                                   "staging_sha256": staging_sha256(ctx.paths.staging)}))
        ok, d = _v_operator_signature(ctx)
        check("re-signed after the edit -> pass", ok, True, d)

        # A signature with an EMPTY staging dir is legitimate: shipping nothing is a real
        # deliverable, and the gate must not force a skill to exist.
        for sk in list(ctx.paths.staging.glob("*/")):
            shutil.rmtree(sk)
        sig.write_text(json.dumps({"checked": ["prompt", "empty library is intended"],
                                   "prompt_sha256": good_sig["prompt_sha256"]}))
        ok, d = _v_operator_signature(ctx)
        check("empty staging + signed prompt -> pass", ok, True, d)

        # #208: prompt numbers that disagree with the prior leg must FAIL
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
        check("grants exclude the run dir itself (#213)", not bad, True,
              f"grants = {[p.name for p in g]}")

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

        def liar(_c):                       # the 2026-08-08 shape: reports success, changes nothing
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

    WHY THIS IS NOT AN "UNATTENDED DRIVER". The reason stands: *a driver catches only DECLARED
    failures, and every defect that has mattered in this programme was undeclared.* What changed is not the rule but
    the loop it drives. `advance()` runs ONE step and then re-asks that step's verify FROM THE
    ARTIFACT, THROUGH THE ARTIFACT'S REAL CONSUMER; a step that ran and still fails is a HALT, not
    a retry. So "declared failure" here means all twenty verifies, plus an operator gate that
    cannot be automated at all. This driver removes the typing between steps. It does not remove
    the reading: it stops dead at the first HALT, the first question, and the proofread gate every
    single epoch, and it prints WHY.

    Concretely it stops on:
      * HALT           — exit 1. Something disagrees with the frozen plan.
      * Blocked        — exit 2. The AO is asking the operator a question.
      * operator gate  — exit 3, its own code, because "a human must read this now" is neither a
                         defect nor a question and a caller that cannot tell them apart will
                         eventually paper over one with the other.
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
    print(f"\nDRIVE — epoch {epoch_now}"
          + (f" through {last_epoch}" if last_epoch != epoch_now else "")
          + f", ceiling {a.max_steps} step(s). Stops at the first HALT, question, or "
            f"operator gate.\n")
    while True:
        if steps_taken >= a.max_steps:
            msg = (f"STOPPED at the --max-steps ceiling ({a.max_steps}) with epoch {epoch_now} "
                   f"still unfinished. Nothing is wrong; raise the ceiling and re-run.")
            print(f"\n{msg}")
            say_ao(msg, f"drive-ceiling:{epoch_now}:{steps_taken}", c=ctx)
            return 4
        try:
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
                  f"  countersign, and re-run the same --drive command to continue.")
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
        print(f"\nepoch {epoch_now} is COMPLETE — {detail}")
        say_ao(f"{board}\nepoch {epoch_now} is COMPLETE — {detail}",
               f"report:{epoch_now}:complete", c=ctx)
        if epoch_now >= last_epoch:
            print(f"\nDRIVE DONE — epochs {first_epoch}..{epoch_now} complete in "
                  f"{steps_taken} step(s).")
            return rc
        epoch_now += 1
        # A FRESH Ctx PER EPOCH, never a mutated one. Ctx carries per-epoch caches and the
        # `_asking_step` cursor, and `_assert_epoch_consistent` exists (#270) precisely because a
        # stale epoch on a live context silently reads the previous epoch's answer file. Rebuilding
        # is cheap and removes the whole class.
        ctx = Ctx(rd, epoch_now, spec, oracle, **kw)
        print(f"\n{'=' * 76}\nEPOCH {epoch_now}\n{'=' * 76}")
    # unreachable


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
    # ── the operator's side of the conversation (#301) ───────────────────────────────────────
    ap.add_argument("--note", default="",
                    help="what you are asking for and why; recorded as YOUR turn in the run's "
                         "operator<->AO transcript and emitted to Phoenix")
    ap.add_argument("--phoenix-project", default="",
                    help="emit the transcript to this Phoenix project after the invocation")
    # ── driving ──────────────────────────────────────────────────────────────────────────────
    ap.add_argument("--drive", action="store_true",
                    help="keep taking steps until the epoch completes, HALTs, BLOCKs, or an "
                         "operator gate is reached")
    ap.add_argument("--until-epoch", type=int, default=0,
                    help="with --drive: on completing an epoch, roll to the next, up to this one")
    ap.add_argument("--max-steps", type=int, default=200,
                    help="with --drive: hard ceiling on steps taken in one invocation")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    # #281. EXIT 2 MEANS BLOCKED AND NOTHING ELSE. The Blocked handler's comment makes 2
    # load-bearing ("so a caller can tell 'answer me' from 'go fix something' without parsing
    # prose") — but main() also returned 2 for a usage error and for an unreadable RUNTIME.json,
    # the latter while printing the word HALT. A wrapper keying on 2 would have treated a corrupt
    # config as a question waiting for the user. Usage errors now use EX_USAGE (64); anything
    # that prints HALT exits 1, like every other HALT.
    if not a.run_dir or a.epoch is None:
        ap.print_help()
        return 64
    from src.orchestrator.spec import ExperimentSpec, SpecError
    rd = Path(a.run_dir)
    try:
        ExperimentSpec.assert_unmutated(rd)
        spec, _ = ExperimentSpec.load(rd)
    except SpecError as e:
        print(f"HALT  frozen spec: {e}")
        return 1

    # RUNTIME.json supplies the same plumbing without a wall of flags on every invocation.
    # Explicit flags win, so a one-off override never requires editing the file.
    rt = {}
    rt_file = rd / "RUNTIME.json"
    if rt_file.is_file():
        try:
            rt = json.loads(rt_file.read_text())
        except json.JSONDecodeError as e:
            print(f"HALT  {rt_file.name} is not readable JSON: {e}")
            return 1                                   # #281: prints HALT, so exits like a HALT

    oracle = Path(a.oracle or rt.get("oracle") or (rd / "oracle"))
    kw = {}
    if a.leg_script or rt.get("leg_script"):
        kw["leg_script"] = a.leg_script or rt["leg_script"]
    if a.shared_dir or rt.get("shared_dir"):
        kw["shared_dir"] = a.shared_dir or rt["shared_dir"]
    if rt.get("base_leg_val"):                       # #251: declared, see Ctx.base_leg_val
        kw["base_leg_val"] = rt["base_leg_val"]
    # #253: the AO's prompt-assembly how-to, supplied ONCE at init.
    if rt.get("prompt_policy"):
        kw["prompt_policy"] = rt["prompt_policy"]
    if rt.get("prompt_author_model"):
        kw["prompt_author_model"] = rt["prompt_author_model"]
    # ONE AO CONVERSATION PER RUN, opt-in. Off by default because turning it on changes what the
    # AO is: ao4 ran four SINGLE-SHOT authorings with no memory of each other, and re-deriving the
    # benchmark from harness source each time was a direct consequence. A completed run must stay
    # reproducible, so this is never retro-applied — a NEW run declares it.
    if rt.get("ao_continuous_session"):
        kw["ao_continuous_session"] = bool(rt["ao_continuous_session"])
    # #294: default ON, so read the key when PRESENT (including an explicit false) rather than
    # only when truthy — `if rt.get(...)` could never turn it off, which is the shape of a flag
    # that silently cannot be set.
    if rt.get("digest_bucket"):
        kw["digest_bucket"] = rt["digest_bucket"]
    if "evolver_sandbox" in rt:
        kw["evolver_sandbox"] = bool(rt["evolver_sandbox"])
    # The pinned cache tag. Ctx.cache_tag existed and drive_a2.py passed it, but main() never
    # read it from RUNTIME.json — so the CLI path silently fell back to spec.cache_tag and the
    # first epoch-1 leg wrote into a DIFFERENT bucket than the pinned base. An added parameter
    # that only one of two callers passes is indistinguishable from a parameter that works.
    if a.cache_tag or rt.get("cache_tag"):
        kw["cache_tag"] = a.cache_tag or rt["cache_tag"]
    if a.base_leg or rt.get("base_leg"):
        kw["base_leg"] = a.base_leg or rt["base_leg"]
    if a.evolver_log or rt.get("evolver_log"):
        kw["evolver_log"] = a.evolver_log or rt["evolver_log"]
    caches = a.digest_cache or rt.get("digest_caches") or []
    if caches:
        kw["digest_caches"] = tuple(Path(c) for c in caches)
    if a.leg_timeout_s or rt.get("leg_timeout_s"):
        kw["leg_timeout_s"] = a.leg_timeout_s or rt["leg_timeout_s"]
    ctx = Ctx(rd, a.epoch, spec, oracle, **kw)

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

    # THE OPERATOR'S TURN, recorded BEFORE anything runs (#244: a record written only on success
    # is not a record) and BEFORE the readonly flag is set, because --plan is also a thing the
    # operator said and a transcript that omits the questions I asked is the #301 defect again.
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

    def _emit_transcript() -> None:
        """Push the transcript to Phoenix. Best-effort: losing observability must not change the
        exit code of a run, and the JSONL on disk is the source of truth either way."""
        if not a.phoenix_project:
            return
        try:
            from src.orchestrator import conversation as _conv
            n_x, n_t = _conv.emit(rd, a.phoenix_project)
            print(f"[phoenix] {n_x} exchange(s) / {n_t} turn(s) -> {a.phoenix_project}",
                  file=sys.stderr)
        except Exception as e:                                  # noqa: BLE001
            print(f"[phoenix] transcript emission failed ({type(e).__name__}: {e}); "
                  f"the run and {rd.name}.operator/CONVERSATION.jsonl are unaffected",
                  file=sys.stderr)

    def _say_ao(text: str, key: str, *, c=None) -> None:
        """Record the AO's reply to this invocation."""
        _conv_once(c or ctx, key, speaker="ao", channel="report", content=text)

    def _board(c: Ctx) -> str:
        """report(), captured verbatim so the transcript and the terminal cannot disagree."""
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            results, _ = evaluate(c)
            report(c, results)
        out = buf.getvalue()
        sys.stdout.write(out)
        return out

    # EVERY path that can reach a verify is inside this guard, including the two evaluate()
    # calls below.
    #
    # #263 (found 2026-08-11 by adversarial audit, independently by two lenses). This function
    # USED to call evaluate() OUTSIDE the try, and evaluate() deliberately re-raises Blocked so
    # that "I need an answer" is never reportable as "this check failed". The two facts compose
    # into a dead handler: the documented pattern -- "a verify calls ask() instead of guessing" --
    # produced a raw `src.orchestrator.loop.Blocked` traceback with exit 1, never the operator
    # instructions, never exit 2. --plan crashed identically. A wrapper keying on the exit code
    # read "answer me" as "go fix something", which is the precise confusion Blocked was built to
    # remove.
    #
    # It survived 26 green fixtures because every one of them exercised ask() / _safe_verify /
    # evaluate() / advance() DIRECTLY and not one called main(). Testing a mechanism's parts is
    # not testing the mechanism: the defect lived in the seam between two tested pieces.
    try:
        if not a.step:
            # `--plan` says it "changes nothing"; make that mechanical, not a promise. ask() writes
            # BLOCKED_e{N}.json, so without this a plan would leave files behind.
            ctx.readonly = True
            results, _ = evaluate(ctx)
            report(ctx, results)
            _emit_transcript()
            return 0

        if a.drive:
            rc = _drive(ctx, a, rd, spec, oracle, kw, _say_ao, _board)
            _emit_transcript()
            return rc

        # --step CHANGES THE WORLD, so say what is already true before touching anything. An
        # operator who sees the same board twice knows the step was a no-op.
        board = _board(ctx)
        step, detail = advance(ctx)
    except Blocked as e:
        # A BLOCK is not a defect report. Exit 2, distinct from HALT's 1, so a caller can tell
        # "answer me" from "go fix something" without parsing prose.
        ap = answers_path(ctx.run_dir, ctx.epoch)
        # Create the operator directory so the printed instruction is literally followable.
        # #268 moved this outside the run dir; without the mkdir the user's first action after
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
        _emit_transcript()
        return 2
    except Halt as e:
        print(f"\nHALT\n  {e}\n"
              f"  This is not a retry condition. The world disagrees with the frozen plan;\n"
              f"  a person decides whether to fix the step or restart the epoch.")
        _say_ao(f"HALT\n\n{e}", f"halt:{a.epoch}:{_argv}:{e}")
        _emit_transcript()
        return 1
    if step is None:
        print(f"\nepoch {a.epoch} is COMPLETE — {detail}")
        _say_ao(f"{board}\nepoch {a.epoch} is COMPLETE — {detail}",
                f"report:{a.epoch}:{_argv}:complete")
        _emit_transcript()
        return 0
    print(f"\nRAN + VERIFIED: {step.name}\n  {detail}\n"
          f"  One step per invocation, by design. Run again for the next.")
    _say_ao(f"{board}\nRAN + VERIFIED: {step.name}\n  {detail}",
            f"report:{a.epoch}:{_argv}:{step.name}")
    _emit_transcript()
    return 0


if __name__ == "__main__":
    sys.exit(main())
