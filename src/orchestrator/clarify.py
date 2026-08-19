#!/usr/bin/env python3
"""GAP 1 — THE AO ASKS THE CLARIFICATION QUESTION, IN ITS OWN WORDS.

WHAT THIS REPLACES, AND WHY IT HAD TO GO.

`cli.clarify_loop` used to resolve a failed check through an eight-entry literal:

    REASK = {"dataset root exists": ("root", "Where does the data live?"), ...}

Two defects, and they are different in kind.

  1. IT IS A CLOSED SET OVER AN OPEN ONE. `benchmark.verify()` decides what can fail; the table
     decides what can be asked. Nothing keeps them in step, and the failure mode is not a wrong
     question but `SystemExit: no re-ask is defined for the check ...` — the front door dying on
     the user, at the exact moment the user was about to tell it what it needed. Today the table
     happens to be complete. It stopped being complete the moment anyone added a check, and the
     person adding the check gets no signal.

  2. THE QUESTION IS THE PART THAT MUST NOT BE CANNED. the maintainer's V2 target is "an interactive
     session where the user describes the benchmark and the AO asks clarification questions if
     needed". A fixed string cannot cite what was actually found. "What pattern picks out the
     tasks?" is a form field; "`*.json` matched one file, `tasks.json`, and that file parses as a
     114-element array — are those 114 the tasks, or is the file one task?" is a question, and it
     is a question only something that read the evidence can ask.

WHERE THE TRUST BOUNDARY SITS, WHICH IS THE WHOLE DESIGN.

    verify()   REMAINS THE SOLE ARBITER. The AO never decides whether a claim is satisfied, never
               sees a check pass or fail by its own judgement, and its output is not consulted by
               anything except the phrasing of a question and the choice of which field the USER
               will revise. Everything the user then says is re-checked against the files.

    the AO     composes. It may READ the dataset (read-only, contained) so its question can cite
               the actual bytes. It may NOT write anything, may NOT edit the claim, and may NOT
               supply the answer — the two together would make it a hidden author of the
               experiment, which is the standing rule this programme keeps re-learning (#218: an
               optimizer with an undeclared human, or model, in the loop is not the optimizer you
               reported).

    the user   answers. Always. There is no branch on which a missing or unusable AO response
               causes the loop to guess a value: it degrades to asking the user which field to
               revise, using verify()'s own question text, and says why.

WHAT COMES BACK. Exactly one JSON object: {"field": <a revisable BenchmarkClaim field>,
"question": <prose, addressed to the user>, "why": <one line, for the transcript>}. The field is
validated against the dataclass — an invented name is rejected rather than setattr'd, because
setattr on an unvalidated string is how a model gets write access to a spec by accident.
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess

from src.orchestrator.quota_scope import subscription_env
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Fields the USER may be asked to revise. `name`/`domain`/`notes` are excluded deliberately: they
# are labels and free prose, never the cause of a check failing against the filesystem, and
# offering them as a target invites a re-ask that cannot possibly resolve anything.
REVISABLE = ("root", "task_glob", "records_in", "id_from", "input_glob", "answer_glob",
             "scoring_desc", "scoring_script")


def _claim_fields(claim) -> tuple:
    """The revisable fields that actually exist on this claim — never a hardcoded echo."""
    have = {f.name for f in dataclasses.fields(claim)}
    return tuple(f for f in REVISABLE if f in have)


def _listing(root: Path, limit: int = 40) -> str:
    """A short, TRUE picture of what is on disk under the dataset root.

    Included so the AO's question can cite bytes rather than restate the user's description back
    to them. Bounded, and it never opens a file: this runs before anything is verified, and a
    front door that reads user data to compose a question would be doing more than it said.
    """
    out = []
    try:
        for p in sorted(root.rglob("*"))[:limit]:
            try:
                rel = p.relative_to(root)
            except ValueError:
                continue
            kind = "dir " if p.is_dir() else f"{p.stat().st_size:>9} B"
            out.append(f"  {kind}  {rel}")
    except OSError as e:
        return f"  (could not list {root}: {e})"
    if not out:
        return f"  ({root} is empty or does not exist)"
    return "\n".join(out) + ("\n  ... (truncated)" if len(out) >= limit else "")


def _message(claim, finding, fields: tuple, round_no: int) -> str:
    root = Path(getattr(claim, "root", "") or ".")
    told = "\n".join(f"  {f:<15} {getattr(claim, f, '')!r}" for f in fields)
    ev = json.dumps(getattr(finding, "evidence", {}) or {}, sort_keys=True)
    return "\n".join([
        "You are the orchestrator of a skill-evolution experiment, talking to the DEVELOPER who "
        "owns the benchmark being set up. This is round " + str(round_no) + " of setting it up.",
        "",
        "They have described their benchmark. I checked that description against their files and "
        "one check did not come out the way the description implies. I am ASSUMING I MISREAD THEM, "
        "not that their data is wrong — they know this dataset and I learned about it a minute ago.",
        "",
        "WHAT THEY TOLD ME:",
        told,
        "",
        "THE CHECK THAT DID NOT COME OUT:",
        f"  check:    {getattr(finding, 'check', '')}",
        f"  what I saw: {getattr(finding, 'detail', '')}",
        f"  evidence: {ev}",
        (f"  a default phrasing exists, for reference only: {finding.question}"
         if getattr(finding, "question", "") else ""),
        "",
        f"WHAT IS ACTUALLY ON DISK under {root}:",
        _listing(root),
        "",
        "YOUR JOB — two decisions, and nothing else:",
        "",
        "  1. WHICH ONE of their answers should they revise? Pick exactly one of: "
        + ", ".join(fields) + ".",
        "     Pick the field whose value, if it were different, would make this check come out. "
        "     Re-asking the wrong field is how a clarification loop becomes an infinite one.",
        "",
        "  2. HOW should I ask them? Write the question I will put to them.",
        "",
        "WRITING THE QUESTION — this is the part that matters:",
        "  - They know their benchmark inside out. They know NOTHING about this tool's internals: "
        "not its field names, not its class names, not its check names. Never make them decode a "
        "term that only exists in here.",
        "  - CITE WHAT YOU FOUND. Name the actual files, counts and sizes above. A question that "
        "could have been written without looking at their data is a form field, not a question.",
        "  - Ask about their DATASET, in their world's terms — what one task is, where its input "
        "lives, how it is identified — not about how to fill in a config.",
        "  - One question. If you need two things, ask for the one that unblocks the other.",
        "  - Do not tell them their data is wrong, do not propose a value and ask them to confirm "
        "it, and do not answer it yourself. You are asking, they are deciding.",
        "",
        "OUTPUT: one JSON object and nothing else, no prose around it, no code fence:",
        '  {"field": "<one of the fields above>", "question": "<what I should ask them>", '
        '"why": "<one line: why that field, for the setup transcript>"}',
    ])


def _sandbox():
    """WRITE containment for the composer. Reads are NOT restricted — say so plainly.

    This took a `root` argument it never used, and both this docstring and brief.py's said the
    composer "may read the dataset root and the repo", which reads as a confinement. The profile
    below has no file-read rule at all: under `(allow default)` every read on the host is
    permitted, and the only thing it constrains is writing. The parameter is gone rather than
    left as evidence for a claim the code does not make.

    Write containment is the property that actually matters here, and it is real: the composer
    runs BEFORE any spec is frozen, so an uncontained one could write the very files the front
    door is about to verify. Read confinement is deliberately not attempted — the operator's
    dataset can be anywhere, nothing is sealed yet (no split exists to leak), and the evolver
    profile next door took four attempts to grant a CLI enough to start. Guessing at a read
    policy here would buy a claim, not a guarantee.
    REFUSES rather than degrading when sandbox-exec is unavailable — a containment layer that
    quietly no-ops is worse than none, and that exact silent `return ([], None)` was a live defect
    in prompt_author until today.
    """
    import tempfile
    from .evolver_dispatch import _sandbox_exec_available
    if not _sandbox_exec_available():
        if os.environ.get("EVOSKILL_AO_ALLOW_UNCONTAINED") == "1":
            return ([], None)
        raise ClarifyUnavailable(
            "sandbox-exec is unavailable, so the clarification composer cannot be contained. "
            "Set EVOSKILL_AO_ALLOW_UNCONTAINED=1 to run it uncontained as a recorded decision.")
    prof = "\n".join([
        "(version 1)",
        "(allow default)",
        '(deny file-write*)',
        '(allow file-write* (subpath "/dev"))',
        '(allow file-write* (subpath "/private/var/folders"))',
        '(allow file-write* (subpath "/private/tmp"))',
        f'(allow file-write* (subpath "{Path.home() / ".claude"}"))',
        "",
    ])
    fh = tempfile.NamedTemporaryFile("w", suffix=".clarify.sb", delete=False)
    fh.write(prof)
    fh.close()
    return (["sandbox-exec", "-f", fh.name], fh.name)


class ClarifyUnavailable(Exception):
    """The composer could not be reached or could not be trusted. NEVER a reason to guess a value
    — the caller falls back to asking the user which field to revise."""


def _extract_json(text: str) -> dict:
    """The first complete JSON object in the text. Models fence, preface and apologise; the
    contract is one object, and this recovers it without accepting a second one."""
    depth, start = 0, -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                try:
                    d = json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    continue
                if isinstance(d, dict):
                    return d
    raise ClarifyUnavailable("no JSON object in the composer's reply")


def compose(claim, finding, *, round_no: int = 1, model: str = "opus[1m]",
            timeout_s: int = 240, log_path=None) -> dict:
    """Ask the AO which field to re-ask and how to phrase it. Raises ClarifyUnavailable, never
    returns a guess. See the module docstring for what it is and is not permitted to decide."""
    fields = _claim_fields(claim)
    if not fields:
        raise ClarifyUnavailable("the claim exposes no revisable field")
    msg = _message(claim, finding, fields, round_no)
    sb_argv, sb_prof = _sandbox()
    argv = [*sb_argv, "claude", "-p", msg, "--model", model,
            "--output-format", "json", "--dangerously-skip-permissions"]
    try:
        # SCRUB THE API-KEY OVERRIDES, like prompt_author.py and loop.py already do.
        # README.md and .env.example both state that quota_scope.subscription_env()
        # removes ANTHROPIC_API_KEY/AUTH_TOKEN/BASE_URL from "every `claude` child that
        # runs a Claude-family model". This call site passed no env= at all, so the AO
        # front door -- the FIRST command in "Start an evolution" -- inherited all three
        # and billed API credit behind the subscription quota gate the loop reads (#307).
        # The documentation was true of two of the four dispatch sites.
        r = subprocess.run(argv, cwd=str(REPO), capture_output=True, text=True,
                           timeout=timeout_s, env=subscription_env(model))
    except FileNotFoundError:
        raise ClarifyUnavailable("`claude` is not on PATH")
    except subprocess.TimeoutExpired:
        raise ClarifyUnavailable(f"the composer did not answer within {timeout_s}s")
    finally:
        if sb_prof:
            try:
                os.unlink(sb_prof)
            except OSError:
                pass
    if log_path:
        try:
            Path(log_path).write_text(
                f"=== clarify round {round_no} model {model} rc {r.returncode}\n"
                f"--- SENT\n{msg}\n--- STDOUT\n{r.stdout}\n--- STDERR\n{r.stderr[-4000:]}\n")
        except OSError:
            pass
    if r.returncode != 0:
        raise ClarifyUnavailable(f"the composer exited {r.returncode}: {r.stderr.strip()[-300:]}")

    # `--output-format json` wraps the turn; the payload we want is in `result`.
    body = r.stdout
    try:
        env = json.loads(body)
        if isinstance(env, dict) and isinstance(env.get("result"), str):
            body = env["result"]
    except json.JSONDecodeError:
        pass
    d = _extract_json(body)

    field, question = d.get("field"), d.get("question")
    # VALIDATE THE FIELD AGAINST THE DATACLASS. The caller setattr()s this string; an unvalidated
    # one is a model with write access to arbitrary attributes of the object the whole front door
    # exists to keep honest.
    if field not in fields:
        raise ClarifyUnavailable(f"the composer chose {field!r}, which is not one of {fields}")
    if not (isinstance(question, str) and question.strip()):
        raise ClarifyUnavailable("the composer returned no question")
    return {"field": field, "question": question.strip(),
            "why": str(d.get("why", "")).strip(), "model": model, "round": round_no}
