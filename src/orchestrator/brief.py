#!/usr/bin/env python3
"""GAP 2 — THE AO TURNS A SENTENCE INTO A SETUP.

    python3 -m src.orchestrator.brief --brief-file BRIEF.md --run-dir outputs/foo
    python3 -m src.orchestrator.cli new --brief BRIEF.md --run-dir outputs/foo

WHAT THIS IS. `cli new` asks twenty structured questions. That is the right shape for a careful
setup and the wrong shape for the thing the maintainer actually asked for: *"AO can take your instruction
(you role play a user who wants to benchmax tau3 banking) and launch a 4-epoch evolution"*. A
person with that intent writes a paragraph, not twenty form fields. This module is the translator:
free prose in, a complete set of ANSWERS out, which then flow through the existing front door
unchanged.

THE TRUST BOUNDARY, WHICH IS DIFFERENT FROM clarify.py's AND DELIBERATELY SO.

`clarify.compose` may never supply a value — the user answers, always — because there the AO is
resolving a disagreement between the user's description and their files, and a model that both
spots the disagreement and settles it has authored the experiment. Here the AO is doing something
else: PARSING AN INSTRUCTION THE OPERATOR JUST GAVE IT. Refusing to let it fill fields would make
the brief pointless. So it may propose values, under three constraints that keep it from becoming
a hidden author:

  1. IT PROPOSES ANSWERS, NOT FACTS. Every value it returns is fed to `cli new` as if the operator
     had typed it, and `benchmark.verify()` then checks each one against the actual files. The AO
     gets no vote on whether its own proposal is correct. A proposal that disagrees with the disk
     lands in `clarify_loop`, where the rule reverts to the strict one: the AO asks, the OPERATOR
     answers.
  2. THE KEY SET IS CLOSED. Only the keys `cli new` actually asks for are accepted; anything else
     is a refusal, not a warning. An open key set is how `test_sealed: false` arrives in a spec
     without anyone deciding it.
  3. THE OPERATOR COUNTERSIGNS BEFORE ANYTHING FREEZES. The proposal is written to disk and
     printed field by field, with the brief beside it. `--yes` exists for scripted runs and says
     so in the receipt.

WHAT IT MAY WRITE — nothing outside /dev, the temp dirs and ~/.claude, contained by `sandbox-exec`,
the same containment `clarify.compose` uses and for the same reason (#267/#286): this runs BEFORE
any spec is frozen, so an uncontained composer could write the very files the front door is about
to verify. It REFUSES rather than degrading if containment is unavailable. READS ARE NOT
RESTRICTED: the profile carries no file-read rule, so this is write containment and nothing more.
An earlier version of this paragraph said "the dataset root and the repo, read-only", which
described a confinement the profile has never had.

WHAT IT MUST NOT DO. Choose the split sizes silently when the brief did not say. A benchmark's
train/val/test counts are the experiment's power budget, and a model picking them from thin air
would be picking the study's sensitivity. If the brief is silent, it must return `null` for that
key and SAY SO, and the operator is asked. Same for the cost ceiling.
"""
from __future__ import annotations

import json
import os
import subprocess

from src.orchestrator.quota_scope import subscription_env
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.orchestrator.clarify import (ClarifyUnavailable, _extract_json,  # noqa: E402
                                      _listing, _sandbox)

# ── the closed key set ───────────────────────────────────────────────────────────────────────
# EXACTLY the keys `cli.main` asks for, in the order it asks them. Kept as one table with a
# per-key note so the prompt below is GENERATED from it: a second hand-written list of keys in the
# prompt would drift from this one the first time a question changed, and the failure would be a
# missing answer at the worst moment (`SCRIPT INCOMPLETE — no answer for ...`).
KEYS: tuple[tuple[str, str], ...] = (
    ("name", "short identifier for the benchmark; becomes the run's name (no spaces)"),
    ("description", "what one task asks the agent to do and how a response is judged, in prose"),
    ("domain", "the exact task-pool string the harness expects, e.g. 'banking'"),
    ("root", "path to the data, relative to the repo root"),
    ("task_glob", "pattern under root that picks out the tasks"),
    ("records_in", "'array' | 'object' | 'jsonl' | '' if one file per task"),
    ("id_from", "'stem' | 'name' | 'parent' | 'json:<key>'"),
    ("input_glob", "file the agent is GIVEN inside a task, or '' if the record IS the input"),
    ("answer_glob", "where the ground truth lives, or '' if it is inside the task record"),
    ("scoring_desc", "how a response becomes a score"),
    ("scoring_script", "scoring script path relative to root, or ''"),
    ("n_train", "int — tasks in TRAIN. null unless the brief says."),
    ("n_val", "int — tasks in VAL. null unless the brief says."),
    ("n_test", "int — tasks in TEST, sealed and never run. null unless the brief says."),
    ("seed", "int — split seed. null unless the brief says."),
    ("solver_model", "the agent being improved"),
    ("evolver_model", "the model that writes skills"),
    ("user_sim_model", "the model simulating the user, or 'none'"),
    ("judge_model", "the judging model, or 'none'"),
    ("n_samples", "int — samples per task. null unless the brief says."),
    ("n_epochs", "int — epochs. null unless the brief says."),
    ("cost_ceiling_usd", "int — whole dollars for the entire run. null unless the brief says."),
)
KEY_NAMES = tuple(k for k, _ in KEYS)
# Keys the AO must NOT invent when the brief is silent — see the module docstring's last
# paragraph. Returning null here is the CORRECT answer, and the operator is then asked.
MUST_NOT_GUESS = ("n_train", "n_val", "n_test", "seed", "n_samples", "n_epochs",
                  "cost_ceiling_usd")


class BriefUnavailable(Exception):
    """The AO could not be reached, or its proposal could not be trusted. Never a reason to
    proceed with a guessed setup — the caller falls back to the twenty questions."""


def _message(brief: str, root_hint: str, repo: Path) -> str:
    hint = Path(root_hint) if root_hint else None
    listing = _listing(hint) if hint else "  (no data root named in the brief — go and find it)"
    keys = "\n".join(f'    "{k}": ...   # {note}' for k, note in KEYS)
    return "\n".join([
        "You are the orchestrator (AO) of a skill-evolution experiment. An engineer has just told "
        "you, in their own words, what they want to run. Your job is to turn that into the exact "
        "set of setup answers the pipeline needs — by READING THEIR REPOSITORY, not by guessing.",
        "",
        "=" * 78, "WHAT THEY SAID", "=" * 78, "",
        brief.strip(),
        "", "=" * 78, "WHERE YOU ARE", "=" * 78, "",
        f"  repo root: {repo}",
        f"  data root hint from the brief: {root_hint or '(none given)'}",
        "",
        listing,
        "",
        "=" * 78, "HOW TO WORK", "=" * 78, "",
        "  * READ THE REPOSITORY. Find the dataset, open a task file, look at how ids are formed, "
        "find the scorer. Grep for the harness's own name for this task pool. Every field below "
        "is a claim about files that exist, and you are about to be checked against them: the "
        "value you give is verified against the disk immediately after you return it, and a value "
        "that does not check out costs a round trip to the engineer.",
        "  * PREFER WHAT THE REPO ALREADY DOES. If a split builder, a runner or a previous run "
        "directory for this benchmark exists, read it — the conventions in there are the right "
        "answers, and inventing new ones creates a second way to do the same thing.",
        "  * DO NOT WRITE ANYTHING. You are contained read-only. You have no deliverable file; "
        "your answer is the JSON below, on stdout.",
        "",
        "=" * 78, "WHAT YOU MUST NOT INVENT", "=" * 78, "",
        "  These keys — " + ", ".join(MUST_NOT_GUESS) + " — set how much evidence the experiment "
        "will have and how much it may cost. If the brief did not state a value, return null and "
        "the engineer will be asked. A number you chose because it seemed reasonable would be you "
        "picking the study's sensitivity, which is not yours to pick. Returning null is the "
        "CORRECT answer here, not a failure.",
        "",
        "=" * 78, "OUTPUT", "=" * 78, "",
        "One JSON object, nothing else — no prose around it, no code fence:",
        "",
        "  {",
        keys,
        '    "reasoning": "<8-15 lines: for each non-obvious field, the file or line you read '
        'that settled it. This is read by a person deciding whether to trust the setup.>",',
        '    "unsure": ["<field names you are least confident about, if any>"]',
        "  }",
        "",
        "Every key above must be present. Use null only where the rules permit it; use \"\" for a "
        "field that is genuinely empty (e.g. no scoring script).",
    ])


def propose(brief: str, *, root_hint: str = "", model: str = "opus[1m]", timeout_s: int = 900,
            log_path=None, repo: Path = REPO) -> dict:
    """Ask the AO to turn `brief` into setup answers. Raises BriefUnavailable, never guesses."""
    msg = _message(brief, root_hint, repo)
    try:
        sb_argv, sb_prof = _sandbox()
    except ClarifyUnavailable as e:
        raise BriefUnavailable(str(e)) from e
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
        r = subprocess.run(argv, cwd=str(repo), capture_output=True, text=True,
                           timeout=timeout_s, env=subscription_env(model))
    except FileNotFoundError as e:
        raise BriefUnavailable("`claude` is not on PATH") from e
    except subprocess.TimeoutExpired as e:
        raise BriefUnavailable(f"the AO did not answer within {timeout_s}s") from e
    finally:
        if sb_prof:
            try:
                os.unlink(sb_prof)
            except OSError:
                pass
    if log_path:
        try:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            Path(log_path).write_text(f"=== brief intake, model {model}, rc {r.returncode}\n"
                                      f"--- SENT\n{msg}\n--- STDOUT\n{r.stdout}\n"
                                      f"--- STDERR\n{r.stderr[-4000:]}\n")
        except OSError:
            pass
    if r.returncode != 0:
        raise BriefUnavailable(f"the AO exited {r.returncode}: {r.stderr.strip()[-400:]}")

    body = r.stdout
    try:
        env = json.loads(body)
        if isinstance(env, dict) and isinstance(env.get("result"), str):
            body = env["result"]
    except json.JSONDecodeError:
        pass
    try:
        d = _extract_json(body)
    except ClarifyUnavailable as e:
        raise BriefUnavailable(str(e)) from e
    return validate(d, msg)


def validate(d: dict, sent: str = "") -> dict:
    """Refuse a proposal that is not exactly the closed key set. Returns the cleaned answers.

    STRICT IN BOTH DIRECTIONS, and the second direction is the one that matters. A MISSING key is
    caught later anyway (`Answers.ask` raises SCRIPT INCOMPLETE), but an EXTRA key is not caught
    by anything: `cli new` reads the script by key, so an invented `test_sealed` or `n_epochs_real`
    would sit in the file looking authoritative and influencing nobody, or worse, would be picked
    up by a later reader as a decision someone made. A model's output is not a config format.
    """
    if not isinstance(d, dict):
        raise BriefUnavailable(f"the AO returned a {type(d).__name__}, not an object")
    extra = sorted(set(d) - set(KEY_NAMES) - {"reasoning", "unsure"})
    if extra:
        raise BriefUnavailable(
            f"the AO returned key(s) the setup does not have: {extra}. The key set is closed; a "
            f"value nobody asked for cannot be verified and must not reach a frozen spec.")
    missing = [k for k in KEY_NAMES if k not in d]
    if missing:
        raise BriefUnavailable(f"the AO omitted required key(s): {missing}")
    out = {}
    for k in KEY_NAMES:
        v = d[k]
        if v is None:
            if k not in MUST_NOT_GUESS:
                raise BriefUnavailable(
                    f"{k!r} came back null, but null is only permitted for {MUST_NOT_GUESS} — "
                    f"every other field is a claim about files the AO was able to read.")
            continue                      # left out entirely -> cli asks the operator
        if k in MUST_NOT_GUESS:
            # int() TRUNCATES. `int(25.5) == 25` would turn an ambiguous answer into a confident
            # split size that is neither of the two readings, silently — caught by the fixture,
            # and exactly the class of failure this whole validator exists for. A bool is an int
            # in Python, so it is excluded by name or `n_train: true` becomes a 1-task split.
            if isinstance(v, bool) or isinstance(v, float) or (
                    not isinstance(v, int) and not (isinstance(v, str) and v.strip().lstrip("-")
                                                    .isdigit())):
                raise BriefUnavailable(
                    f"{k!r} came back {v!r}, which is not a whole number. Rounding it would be me "
                    f"choosing the value, which is the one thing this key may not have done to "
                    f"it.")
            out[k] = int(v)
        else:
            out[k] = str(v)
    for k in ("name", "domain", "root", "task_glob"):
        if not out.get(k, "").strip():
            raise BriefUnavailable(f"{k!r} is empty — the setup cannot proceed without it")
    if " " in out["name"]:
        # `cli` does `claim.name.replace(" ", "_")` for the spec name but the RUN DIR and every
        # Phoenix project stem are built from it too. Refuse rather than silently rewriting: a
        # name the operator sees in the receipt must be the name on the artifacts.
        raise BriefUnavailable(f"name {out['name']!r} contains a space; give one without.")
    out["_reasoning"] = str(d.get("reasoning", "")).strip()
    out["_unsure"] = [str(x) for x in (d.get("unsure") or []) if isinstance(x, (str, int))]
    out["_sent_chars"] = len(sent)
    return out


def render(answers: dict, brief: str) -> str:
    """The proposal, laid out for a human to countersign. Nulls are shown as ASK YOU."""
    L = ["=" * 78, "THE AO READ YOUR BRIEF AND PROPOSES THIS SETUP", "=" * 78, "",
         "YOUR BRIEF, verbatim:", ""]
    L += ["  " + ln for ln in brief.strip().splitlines()]
    L += ["", "ITS PROPOSAL:", ""]
    for k, note in KEYS:
        if k in answers:
            L.append(f"  {k:<18} {answers[k]!r}")
        else:
            L.append(f"  {k:<18} — WILL ASK YOU ({note})")
    if answers.get("_reasoning"):
        L += ["", "WHY (its words):", ""]
        L += ["  " + ln for ln in answers["_reasoning"].splitlines()]
    if answers.get("_unsure"):
        L += ["", f"IT FLAGGED AS LEAST CERTAIN: {', '.join(answers['_unsure'])}"]
    L += ["", "Nothing above is trusted yet. Every field is re-checked against your files by the",
          "front door, and anything that disagrees comes back to you as a question.", "=" * 78]
    return "\n".join(L)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--brief", help="the instruction, inline")
    src.add_argument("--brief-file", help="the instruction, from a file ('-' for stdin)")
    ap.add_argument("--root-hint", default="", help="where the data probably lives")
    ap.add_argument("--model", default="opus[1m]")
    ap.add_argument("--out", default="", help="write the proposal here as a --script JSON")
    ap.add_argument("--log", default="", help="write the full AO exchange here")
    ap.add_argument("--timeout-s", type=int, default=900)
    a = ap.parse_args()
    brief = (a.brief if a.brief else
             sys.stdin.read() if a.brief_file == "-" else Path(a.brief_file).read_text())
    if not brief.strip():
        print("the brief is empty", file=sys.stderr)
        return 64
    try:
        ans = propose(brief, root_hint=a.root_hint, model=a.model, timeout_s=a.timeout_s,
                      log_path=a.log or None)
    except BriefUnavailable as e:
        print(f"\nTHE AO COULD NOT TURN THIS BRIEF INTO A SETUP:\n  {e}\n"
              f"\n  Nothing has been written. Run `cli new` without --brief to answer the "
              f"questions yourself.", file=sys.stderr)
        return 1
    print(render(ans, brief))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(
            {k: v for k, v in ans.items() if not k.startswith("_")}, indent=1))
        print(f"\nwrote {a.out} — pass it as `cli new --script {a.out}`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
