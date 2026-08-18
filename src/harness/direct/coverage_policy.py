"""Coverage reminder policies — the 2x2 ablation, and the production knob.

The deployed reminder (executor.coverage_gap_message) bundles five separable
things, and two of them quietly hand the stop decision to the evolver:

  (a) count        "nothing in your bash history referenced N of the reads..."
  (b) enumeration  "  - <what>: `<path>`"
  (c) escape #1    "...OR an explicit shortfall stated in writing"
  (d) coaching     "Reading is not the point - DECIDING is: ..."
  (e) escape #2    "This check sees only paths your commands mention - if you
                    covered one another way, say so and move on."

Used as-is it is therefore NOT a clean "harness decides + fully informed" cell:
comparing it against a self-stop variant would compare a thing to itself, and
comparing it against a no-list variant would confound the list with (d).

So the four cells below vary exactly two factors and hold (d) byte-identical:

                  |  harness DISCLOSES its measurement  |  harness stays SILENT
  ----------------+-------------------------------------+----------------------
  harness decides |  A1  disclose_gated                 |  A2  silent_gated
  evolver decides |  A3  disclose_selfstop              |  A4  silent_selfstop

  information axis = (a)+(b) present or absent
  authority axis   = escape hatch present or absent
  A4 is the "just ask the model" design in its pure form.

(e) IS DROPPED FROM ALL FOUR -- but NOT because the estimator got good enough to
assert a file was unread. It did not: measured against owed reads whose path
appears in a command carrying a read verb, the content check MISSES 52% (59% of
those are head/tail/sed -n partial reads whose window excludes the fingerprint,
19% files with no unique line at all, 14% full `cat` reads that should have
matched). Naming rescues that particular set, and the residual -- read by an
unnamed glob/loop idiom AND missed by content -- can be bounded but not measured,
since an unnamed read leaves no ground-truth handle.

What removes the need for a hedge is the FRAMING, not the estimator. Every cell
states only what the harness can see ("neither your commands nor their output
show N of the reads"), never what the evolver did ("you have not read these").
An evidence claim cannot be false; an agent claim can, and roughly half the time
would be. Keeping (e) on top of evidence framing would be redundant AND would
leak stop-authority into the two `_gated` cells, collapsing the second factor.

CODA is verbatim from the deployed (d), with one wording change: "for each one"
became "for each read you have not yet made", so the identical string is
grammatical in the silent cells where no list precedes it.
"""

from __future__ import annotations

from typing import Any

# Held byte-identical across all four cells. Any edit here changes every arm
# equally and so cannot confound either factor.
#
# Rewritten from the deployed version (~595 -> ~215 chars). Three defects, two of
# them substantive:
#
#  1. It opened "Reading is not the point — DECIDING is". In the one message whose
#     purpose is to obtain more reading, that is an escape hatch. Held constant it
#     does not cancel out -- it puts a weak escape in ALL four cells and so
#     compresses the authority contrast the 2x2 exists to measure.
#  2. "for each read you have not yet made, say what it changes about the edit you
#     already wrote" asks for a verdict on a file BEFORE it is opened. Followed
#     literally it instructs confabulation, and the confabulation lands in the
#     skill edit -- the artifact under measurement. With content coverage at
#     0.073 (full batch) / 0.327 (minibatch) that covers most of the owed list.
#     Fixed by making the read a precondition: "read it, then say".
#  3. ~50% of it was coaching about skill-edit epistemics (the "be suspicious of
#     'the skill never loaded here'" block), not coverage instruction. In A2 that
#     made the message 82% coaching, 18% reminder, so every cell was dominated by
#     content unrelated to its own factor.
#
# The suspicion block's right home is the evolver's MAIN prompt, where it governs
# all reasoning rather than only post-reminder turns. It cannot move there for
# this ablation: Option-A anchoring replays each cached session's history
# verbatim, so the main prompt must stay byte-identical to what those 12 sessions
# saw. Relocating it is a follow-up for live runs.
#
# "For each read you have not yet made" names a CATEGORY, not a list, so it stays
# grammatical in the silent cells -- the antecedent trap that killed the deployed
# "Each still owes a full read".
CODA = (
    "For each read you have not yet made: read it, then say what it changes "
    "about the edit you already wrote — and if nothing, why. That edit is a "
    "hypothesis until this evidence tests it. Then give your final answer, "
    "edited or not."
)

# The escape hatch. Present in the two `_selfstop` cells, absent in the two
# `_gated` cells. This single string IS the authority factor.
#
# It carries an EMITTABLE TOKEN because the cells fire repeatedly (same cap and
# same stop rules as the gated cells -- dosage differences between the families
# are a MEDIATOR of the escape, not a confound to be capped away). A behavioural
# proxy such as "stopped making progress" will not do: ESCAPE promises that saying
# so ends the checks, so if the harness instead infers intent from the next
# round's behaviour it (a) breaks that promise, (b) spends a round detecting it,
# and (c) leaves the harness deciding after all -- which is the gated condition,
# so the authority factor would not be implemented at all.
#
# The token is introduced HERE rather than in the schema or the system prompt.
# Neither may change: Option-A anchoring replays each cached session verbatim, so
# the evolver must never be asked for a field it was not told about before the
# anchor. Prose alongside the artifact is already handled --
# executor._extract_json_from_text collects every candidate JSON blob and takes
# the one ending latest -- so a sentinel line costs nothing.
SELFSTOP_TOKEN = "COVERAGE: SUFFICIENT"

ESCAPE = (
    "If you judge that reading the remainder would not change your analysis, "
    f"reply with the single line {SELFSTOP_TOKEN} and then give your final "
    "answer — no further coverage checks will be sent."
)


def declined(assistant_content: str) -> bool:
    """Did the evolver emit the stop token? Scan the ASSISTANT MESSAGE ONLY.

    Never scan the transcript or the message list: the reminder we sent contains
    SELFSTOP_TOKEN as an instruction, so a whole-history search finds the
    harness's own words and credits them to the evolver. That is the exact failure
    mode that produced every coverage-estimator bug found on 0727 -- a shared
    string credited to the wrong author -- and it would make every selfstop arm
    terminate after one round regardless of what the evolver actually said.

    Honoured only in the `_selfstop` cells. If the token shows up in a `_gated`
    cell the harness MUST ignore it (that is what "harness decides" means), but
    log it -- a spontaneous emission of a token the cell never offered would be
    worth knowing about.
    """
    return SELFSTOP_TOKEN in (assistant_content or "")

# There is deliberately no separate demand sentence. The deployed message had
# "Each still owes a full read", which cannot be held constant across the
# information factor: in the silent cells no list precedes it, so "Each" dangles
# with no antecedent (and the first draft of this module silently omitted it from
# A4 only, which would have made the "constant" block vary by cell). CODA already
# carries the demand -- "for each read you have not yet made, say what it changes"
# -- so the sentence was redundant as well as ungrammatical.


def _enumerate(missing: list[dict[str, Any]]) -> list[str]:
    """(b) — every outstanding group, never truncated.

    An earlier deployed version capped this at 12 and told the evolver "and N
    more not listed here", which is unactionable: it cannot read what it is not
    told it owes. Cost is negligible (the largest observed shortfall was ~2.5KB
    against a 141K-token context).
    """
    out = []
    for g in missing:
        paths = g.get("paths") or []
        alt = "" if len(paths) <= 1 else f" (or any one of {len(paths)})"
        out.append(f"  - {g.get('what')}: `{paths[0] if paths else '?'}`{alt}")
    return out


def disclose_gated(missing: list[dict[str, Any]]) -> str:
    """A1 — count + paths, no escape hatch. Harness re-checks and re-nudges."""
    lines = [
        "[HARNESS] Coverage check — neither your commands nor their output show "
        f"{len(missing)} of the reads this batch owes (all of them are listed):"
    ]
    lines += _enumerate(missing)
    lines.append(CODA)
    return "\n".join(lines)


def silent_gated(missing: list[dict[str, Any]]) -> str:
    """A2 — existence of a shortfall only: no count, no paths, no escape hatch.

    Says "do not appear in your commands or their output", NOT "you have not
    read" (first draft said the latter). The estimator misses 52% of reads it can
    be checked against, so the agent claim would be false about half the time it
    was made; the evidence claim is true by construction. This also keeps A2
    consistent with A1/A3, which were already evidence-framed -- otherwise the
    information factor would be confounded with a shift in what is being asserted.

    Deliberately says "some", not a count: the count is the information factor.

    `missing` is accepted and unused: the harness still gates on it internally,
    so the signature matches the other cells and no caller can leak it by
    accident.
    """
    return (
        "[HARNESS] Coverage check — some of the reads this batch owes do not "
        "appear in your commands or their output. Continue your analysis. "
        f"{CODA}"
    )


def disclose_selfstop(missing: list[dict[str, Any]]) -> str:
    """A3 — count + paths, and the evolver may decline. Information vs authority.

    ESCAPE is a pure SUFFIX, after CODA. Ordered the other way (as the first draft
    had it) the cell said "you may skip these" and then "read it, then say what it
    changes", which reads as self-contradictory. As a suffix the authority factor
    is exactly one appended sentence and nothing else about the message shifts.
    """
    lines = [
        "[HARNESS] Coverage check — neither your commands nor their output show "
        f"{len(missing)} of the reads this batch owes (all of them are listed):"
    ]
    lines += _enumerate(missing)
    lines.append(f"{CODA} {ESCAPE}")
    return "\n".join(lines)


def silent_selfstop(missing: list[dict[str, Any]]) -> str:
    """A4 — "just ask the model", pure. No count, no paths, evolver decides."""
    return (
        "[HARNESS] Coverage check — is your coverage of the reads this batch "
        f"owes complete or incomplete? If incomplete, continue your analysis. "
        f"{CODA} {ESCAPE}"
    )


def deployed_verbatim(missing: list[dict[str, Any]]) -> str:
    """REFERENCE ARM, outside the 2x2 — production's message, unmodified.

    A1 is not the deployed closed form: four things were changed (the "OR an
    explicit shortfall" clause, the estimator disclaimer, the demand sentence, and
    CODA). Without this arm the ablation compares four variants of one new design
    and merely ASSUMES the cleanup helped. It also supplies the only arm
    commensurable with the 9 nudged sessions already on disk, since those were
    produced by exactly this message.

    Reads as a bundle: `A1 != deployed` shows the cleanup mattered without
    identifying which of the four edits did it. Decomposing needs four more arms.

    Imported lazily -- executor imports this module, so a top-level import cycles.
    """
    from .executor import coverage_gap_message
    return coverage_gap_message(missing)


# ---------------------------------------------------------------------------
# REFERENCE ARM, outside the 2x2 (like deployed_verbatim): a DEPTH demand.
#
# Motivation from the completed 2x2. The information factor was null on content
# coverage (+0.002, p=0.81) while large on the naming-inclusive metric (+0.202),
# because handing the evolver a 226-item list makes it skim all of it: observed
# commands batch up to 74 owed paths through one `head` loop, with the evolver's
# own comment "get just the FAIL/PASS line from each remaining DIFF". That is
# genuine reading, but breadth-first and a few lines deep. The harness's own gauge
# caused it -- the coverage check counts files touched, not depth read, so a
# uniform shallow scan maximises exactly what is measured.
#
# This cell changes the DEMAND rather than the disclosure: it asks for evidence a
# first-lines skim cannot produce. Not held constant with the four cells (its coda
# differs), so it is a reference point, never a factorial cell.
#
# It needs its own measurement: a quoted line is not the estimator's fingerprint
# line, so content coverage would score this null by construction. Score it with
# verify_quotes() below, which checks each quoted string against the owed file --
# turning coverage from a heuristic into a proof, and incidentally sidestepping the
# ~52% miss rate the content channel has against partial-window reads.
DEPTH_CODA = (
    "For each read you have not yet made: open it and quote, verbatim and in "
    "backticks, the one line that most determines the outcome for that task -- the "
    "line where the solver diverged, or the line carrying the scorer's verdict. "
    "Then say what it changes about the edit you already wrote, and if nothing, "
    "why. A quote from the opening lines of a file is not evidence that you found "
    "the decisive one. Then give your final answer, edited or not."
)


def depth_quote(missing: list[dict[str, Any]]) -> str:
    """REFERENCE ARM -- disclosure plus a demand for a decisive verbatim quote."""
    lines = [
        "[HARNESS] Coverage check — neither your commands nor their output show "
        f"{len(missing)} of the reads this batch owes (all of them are listed):"
    ]
    lines += _enumerate(missing)
    lines.append(DEPTH_CODA)
    return "\n".join(lines)


def verify_quotes(assistant_text: str, owed_paths_to_text: dict) -> dict:
    """Score a depth arm: how many owed files were quoted VERIFIABLY?

    Extracts backticked spans from the evolver's message and checks each against
    the owed files' contents. A quote that appears in no owed file is reported
    separately -- fabrication is the obvious failure mode of demanding quotes, and
    the deployed coda once instructed exactly that by asking for a verdict on a
    file before opening it, so it is measured rather than assumed absent.
    """
    import re as _re
    quotes = [q.strip() for q in _re.findall(r"`([^`\n]{16,})`", assistant_text or "")]
    verified, unmatched = set(), []
    for q in quotes:
        hit = next((p for p, t in owed_paths_to_text.items() if q in (t or "")), None)
        if hit:
            verified.add(hit)
        else:
            unmatched.append(q[:60])
    return {"quotes": len(quotes), "files_verified": len(verified),
            "unverifiable": len(unmatched), "examples": unmatched[:3]}


POLICIES = {
    "off": None,
    "disclose_gated": disclose_gated,
    "silent_gated": silent_gated,
    "disclose_selfstop": disclose_selfstop,
    "silent_selfstop": silent_selfstop,
    "deployed_verbatim": deployed_verbatim,
    "depth_quote": depth_quote,
}

# The 2x2 proper. `deployed_verbatim` is a reference point, not a cell, and is
# excluded from any factorial analysis of the two factors.
FACTORIAL = ("disclose_gated", "silent_gated", "disclose_selfstop", "silent_selfstop")

# Which cells let the evolver end the round by declaring done. For the two
# `_gated` cells the harness ignores any such declaration and re-checks its own
# criterion; for the two `_selfstop` cells the first non-tool-use message after
# the reminder ends the arm.
SELFSTOP = {"disclose_selfstop", "silent_selfstop"}
