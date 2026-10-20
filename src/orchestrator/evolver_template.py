#!/usr/bin/env python3
"""THE EVOLVER PROMPT, RENDERED BY THE LOOP EVERY EPOCH.

No model writes the per-epoch prompt: every epoch the LOOP renders ITER{N}_PROMPT.md from

    fixed loop text      (this file: the header, what the solver can observe, the cost of a skill,
                          the constraints, the format, the report contract)
  + the AO paragraph     (setup/EVOLVER_TEMPLATE.md, read VERBATIM, never edited; only runs whose
                          setup wrote such a paragraph carry one -- otherwise the verdict section is
                          the loop's own text)
  + slots                (the newest train batch's facts, the evidence list, the prior skill set,
                          the staging dir, the prior-verdict report line).

WHAT DOES NOT MOVE. The numbers come from prompt_author.derive_facts, the SAME deriver the gates
re-derive from (`_v_prompt_numbers` and any launcher-side prompt gate), so a rendered prompt passes
every prompt check unchanged. The batch line keeps the exact shape the gates match:

    This batch is **<T> train tasks** × <S> samples = <E> episodes, of which **<P> of <E> episodes** passed.

The four outcomes come from src/orchestrator/outcomes.py leg_outcome_counts(run_dir, leg_json).
While that module is absent a local fallback returns passed only, and the prompt says the split of
the rest is NOT COMPUTED rather than printing zeros (an empty parse must never render as a zero
measurement).

WHAT THE EVOLVER IS TOLD IT HAS: every epoch's train rollouts, the grader's output in full, the
clean grader, and, when `train_references` is on (the default), the TRAINING tasks' reference
workbooks. Training input workbooks are off by default (`train_inputs`). Every directory the prompt
names is re-checked here before it is named: a prompt must never tell the evolver about a directory
it will not have.

WHAT A DIGEST HOLDS IS READ, NEVER ASSUMED. A digest builder may clip scorer diffs, cap the
changed-cell lists and cut long tool outputs, so the prompt must not claim "in full" by default.
Each evidence dir carries DIGEST_INFO.json; src/orchestrator/digest_info.py checks it against the
task files' bytes and hashes, and the evidence section states only what it records: "in full" only
when no diff is clipped, the changed-cells list only when it is present (and it is refused unless
granted: `changed_cells`, default off), and tool outputs as shortened at marked cuts with the
counts. A dir with no header, or one whose bytes contradict it, is refused. A clipped grader output
is refused unless `require_full_grader_output=False` (the evolver sees the grader's output in full).

THE CLEAN GRADER IS FOR READING. scorer.py recalculates through `docker run ssb-verifier:latest`,
which exists only in the grading image's LibreOffice setup; the evolver's image removes /opt/evoskill
and has no docker. The prompt says the code is there to read and, on the solver's image, that the
setup is not on the evolver's machine (the machine section restores "the grader's own tooling is not
in it") -- both only after daytona_evolver.evolver_image_grader_tooling() confirms it from the
evolver's Dockerfile. GRADER_FAILED_POLICY is the loop's behaviour for a grader-failed episode
(re-grade once, halt if it fails again), one sentence shared with the clean grader's README.

THE ROLLOUTS LAYOUT. The evolver should see (1) high-fidelity rollout traces, (2) the path to the
snapshot of each epoch's available skills, so it knows which skills were NOT loaded as well as which
were, without the skills' full text in its prompt, and (3) each rollout's score in the prompt itself,
so it need not open every trace to find it. With RUNTIME evolver_evidence_layout = "rollouts"
(render_parts(evidence_layout=...)), every evidence dir also holds rollouts/<stem>/s<k>.trace.jsonl +
.grade.txt, solver_context/ and INDEX.json/.md (src/orchestrator/rollout_evidence.py), and the prompt
changes in four places only:
  (a) the skill set being revised is a TABLE (name, the SKILL.md description line, its files and sizes,
      and where they are: the staging dir and the newest dir's _library/), never the files' text (a
      pasted skill set can run to ~90 KB);
  (b) every evidence dir's row names the skill snapshot its solver carried and the skills in it, so a
      skill a rollout did not load is visible as available-and-unused;
  (c) the newest batch's per-rollout scores, one compact line per train task (ROLLOUT_* below), read
      from its INDEX.json and cross-checked against the leg JSON and the outcome counts; the older
      epochs point to their INDEX.md;
  (d) the evidence layout in plain words (traces, grade files, INDEX; how to read a long trace), with
      every number read from the INDEX / the files, and the old per-task digests described as the
      shorter summaries DIGEST_INFO.json says they are.
Every other sentence is the digests-layout text, unchanged. Absent / "digests" renders byte for byte
what it rendered before, so the prompts already rendered for a run keep re-rendering (_v_prompt_rendered).
The per-rollout lines are re-derived line by line from the INDEX and the leg JSON by
loop._v_prompt_numbers, which parses them with parse_rollout_table below.

This module renders text and writes nothing, except through the preview CLI, which refuses to
write the run's own ITER{N}_PROMPT.md (the loop owns that path) and never overwrites a file.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TEMPLATE_NAME = "EVOLVER_TEMPLATE.md"
GRADER_DIRNAME = "grader_clean"
# The run's directory of TRAINING reference workbooks. Built and uploaded by the evolver-upload
# builder; this module only names it, after checking it. Never shares a prefix with
# daytona_evolver.INPUTS_DIRNAME, which the inputs gate would read as a sibling.
REFERENCES_DIRNAME = "evolver_references_train"
MAX_PARAGRAPH_CHARS = 8000
OUTCOME_KEYS = ("passed", "graded_wrong", "not_eligible", "grader_failed")
SPREADSHEET_SUFFIXES = (".xlsx", ".xlsm", ".xls")
# The one batch-identity line. The prompt gates match it literally.
BATCH_LINE = ("This batch is **{T} train tasks** × {S} samples = {E} episodes, "
              "of which **{P} of {E} episodes** passed.")
MACHINE_KEYS = ("image_recipe", "image_sha12", "python", "openpyxl", "pandas", "numpy",
                "libreoffice")
INPUT_KEYS = ("inputs_dir", "n_inputs", "inputs_mb")
# The grader-failed policy: the loop re-grades a grader-failed episode ONCE from its saved workbook and
# halts the run if that re-grade fails too. ONE sentence, used by the prompt and by the clean grader's
# README, so the two cannot drift.
REGRADE_ATTEMPTS = 1
GRADER_FAILED_POLICY = ("The loop re-grades such an episode once, from the saved workbook, and halts "
                        "the run if the re-grade fails too.")
# The clean grader is for READING: scorer.py recalculates through
# `docker run ssb-verifier:latest`, which exists only inside the grading image's LibreOffice setup.
GRADER_READ_ONLY = (" It is there to read: the recorded verdicts come from running it inside the "
                    "grading image's LibreOffice setup, which recalculates the saved workbook with a "
                    "profile that recalculates every formula when a file is loaded.")
# "is not guaranteed to", never "does not": run un-recalculated, the
# grader can still agree with the recorded verdict (a workbook with no formulas, say).
GRADER_NOT_HERE = (" That setup is not on your machine: run here, the grader cannot recalculate the "
                   "workbook, so it is not guaranteed to reproduce a recorded verdict.")
# What a digest shows of a tool call, exactly as the SSB v2 digest builder writes it (both the
# Anthropic tool_use and the OpenAI tool_calls branch): the command alone when the call has one (so
# the bash tool's timeout_sec is dropped), else the JSON arguments whole. "each tool call's arguments"
# would be false: a call with a command does not get them.
TOOL_CALL_SHOWN = ("each tool call's command (or its JSON arguments when it has no command; a call "
                   "that has a command is shown without its other arguments, such as `timeout_sec`)")
_SIDECARS = frozenset({"REGRESSIONS", "README", "INDEX"})
# The evidence layouts (RUNTIME evolver_evidence_layout; rollout_evidence.LAYOUTS, a test pins them equal).
LAYOUTS = ("digests", "rollouts")
# (c) THE PER-ROLLOUT LINES. One line per train task of the newest batch, inside one fenced block:
#     <task file stem> | s0 passed 4457/4457 cells, 44 turns, loaded C D | s1 wrong 4438/4457 cells, 45 turns (cap), loaded C D | ...
# preceded, when the batch's solver carried skills, by the legend line "skills: A = <name>, B = <name>, ...".
# Skills are letters so no skill name sits beside a cell fraction: a name holding "test" or "validation"
# would otherwise make #253's held-out-figure rule read the line as a held-out score. The cells are
# "<matched>/<graded> cells", never "... episodes|rollouts", so the launcher's pass-rate gate never reads a
# line as a batch claim; #208 takes these lines out of its fraction scan and re-derives each one instead.
ROLLOUT_SECTION = "## Per-rollout scores of the newest batch"
OUTCOME_TOKEN = {"passed": "passed", "graded_wrong": "wrong", "not_eligible": "no-output",
                 "grader_failed": "grader-failed"}
TOKEN_OUTCOME = {v: k for k, v in OUTCOME_TOKEN.items()}
REGRADED_MARK = " (re-graded)"
LEGEND_PREFIX = "skills: "
ROLLOUT_LINE_RE = re.compile(r"^(?P<stem>[A-Za-z0-9][A-Za-z0-9_.\-]*) \| (?P<rest>s\d+ .*)$")
ROLLOUT_SAMPLE_RE = re.compile(
    r"^s(?P<k>\d+) (?P<outcome>passed|wrong|no-output|grader-failed)(?P<regraded> \(re-graded\))?"
    r"(?: (?P<m>\d+)/(?P<g>\d+) cells)?, (?P<turns>\d+|\?) turns(?P<cap> \(cap\))?"
    r"(?:, loaded (?P<loaded>none|[A-Z]+(?: [A-Z]+)*))?$")
LEGEND_RE = re.compile(r"^skills: (?P<body>[A-Z]+ = [^,\s]+(?:, [A-Z]+ = [^,\s]+)*)$")
# A trace line carrying a non-empty `reasoning` string, as rollout_evidence.py writes it (json.dumps with
# ensure_ascii=False, default separators). Only gates the words "its reasoning" in the prompt.
_REASONING_BYTES = re.compile(rb'"reasoning": "[^"]')


class RenderError(ValueError):
    """The prompt cannot be rendered truthfully from what is on disk. Never caught here."""


# ─────────────────────────────────────────────────────────────── inputs to the render
def read_template_paragraph(path) -> str:
    """The AO's paragraph, VERBATIM (surrounding whitespace only is dropped). Refused unless it is
    exactly one paragraph of plain prose: no blank line, no heading, no slot marker, bounded size.
    The AO writes it once at setup; the loop never edits it."""
    p = Path(path)
    if not p.is_file():
        raise RenderError(f"no AO template paragraph at {p}")
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        raise RenderError(f"{p} is empty")
    if re.search(r"\n[ \t]*\n", text):
        raise RenderError(f"{p} holds more than one paragraph; the template allows exactly one")
    if any(ln.lstrip().startswith("#") for ln in text.splitlines()):
        raise RenderError(f"{p} holds a markdown heading; the template allows one paragraph of prose")
    if re.search(r"\{[A-Z_]+\}", text):
        raise RenderError(f"{p} holds an unfilled slot marker ({{NAME}})")
    if len(text) > MAX_PARAGRAPH_CHARS:
        raise RenderError(f"{p} is {len(text)} chars; the cap is {MAX_PARAGRAPH_CHARS}")
    return text


def default_leg_json(run_dir, epoch: int) -> Path:
    """The leg the newest digest dir was built from, by the loop's own mapping: epoch 1 = the
    declared base leg (RUNTIME.json base_leg), epoch N = skills_v{N-1}'s train leg (EpochPaths)."""
    from src.orchestrator.loop import EpochPaths
    rd = Path(run_dir)
    if epoch >= 2:
        return EpochPaths(rd, epoch - 1).leg("train")
    rt = rd / "RUNTIME.json"
    base = json.loads(rt.read_text()).get("base_leg") if rt.is_file() else None
    if not base:
        raise RenderError(f"epoch 1 needs the base leg and {rt} declares no base_leg")
    return Path(base)


def _fallback_outcome_counts(run_dir, leg_json) -> dict:
    """Used ONLY while src/orchestrator/outcomes.py does not exist. Passed from the leg json (the
    same sum derive_facts cross-checks); the other three are None = NOT COMPUTED, never 0."""
    from src.orchestrator.prompt_author import _leg_totals
    lp, _le = _leg_totals(Path(leg_json))
    return {"passed": int(lp), "graded_wrong": None, "not_eligible": None, "grader_failed": None}


def outcome_counts(run_dir, leg_json) -> tuple:
    """(counts, source). counts = {passed, graded_wrong, not_eligible, grader_failed}.

    The module's presence is tested, not its import: if outcomes.py exists and fails to import,
    that error propagates rather than silently selecting the fallback."""
    if importlib.util.find_spec("src.orchestrator.outcomes") is None:
        return _fallback_outcome_counts(run_dir, leg_json), "fallback (outcomes.py absent)"
    from src.orchestrator.outcomes import leg_outcome_counts
    got = leg_outcome_counts(Path(run_dir), Path(leg_json))
    counts = {k: got.get(k) for k in OUTCOME_KEYS}
    for k in ("status", "reportable"):
        if k in got:
            counts[k] = got[k]
    return counts, "outcomes.leg_outcome_counts"


def facts_for_epoch(run_dir, epoch: int, *, base_leg=None, all_epochs: bool = True,
                    variant: str = ""):
    """The Facts the loop's gates re-derive from, built exactly as loop._run_author_prompt builds
    them: derive_facts over the newest digest dir, cross-checked against the leg it digests, and in
    all-epochs mode the EVIDENCE INDEX pairing e1 with the base leg and no library, e_j with
    skills_v{j-1}'s train leg and skills_v{j-1}."""
    from src.orchestrator.loop import EpochPaths
    from src.orchestrator.prompt_author import derive_facts
    rd = Path(run_dir)
    base = Path(base_leg) if base_leg else default_leg_json(rd, 1)
    paths = EpochPaths(rd, epoch)
    leg = base if epoch == 1 else EpochPaths(rd, epoch - 1).leg("train")
    if not all_epochs:
        return derive_facts(paths.digests, leg, None, variant=variant)
    evidence = [(j, d, base if j == 1 else EpochPaths(rd, j - 1).leg("train"),
                 None if j == 1 else EpochPaths(rd, j - 1).promoted)
                for j, d in enumerate(paths.evidence_dirs(), start=1)]
    return derive_facts(paths.digests, leg, None, variant=variant, evidence=evidence)


def machine_facts_for(run_dir, *, split_path, base_leg, train_inputs: bool = False) -> dict:
    """The machine facts for a Daytona evolver. Delegates to loop._evolver_env_facts (image recipe
    re-hashed, solver versions from the base leg's runtime_env, inputs dir verified train-only)
    whenever the inputs dir exists; drops the inputs keys when `train_inputs` is off. With no
    inputs dir at all (inputs off, never built) the image half is read from the same two sources."""
    from types import SimpleNamespace
    from src.orchestrator.daytona_evolver import dockerfile_check, inputs_dir
    rd = Path(run_dir)
    if inputs_dir(rd).is_dir():
        from src.orchestrator.loop import _evolver_env_facts
        ns = SimpleNamespace(run_dir=rd, spec=SimpleNamespace(split_path=str(split_path)),
                             base_leg=str(base_leg))
        mf = dict(_evolver_env_facts(ns))
    else:
        if train_inputs:
            raise RenderError(f"train_inputs is on but there is no inputs dir at {inputs_dir(rd)}")
        ok, det, rec = dockerfile_check()
        if not ok:
            raise RenderError(f"evolver image recipe does not check: {det}")
        rt = json.loads(Path(base_leg).read_text()).get("runtime_env") or {}
        stack = str(rt.get("solver_stack") or "").split()
        if len(stack) < 3 or not rt.get("libreoffice"):
            raise RenderError(f"base leg {base_leg} records no solver_stack/libreoffice")
        mf = {"image_recipe": rec["base"], "image_sha12": rec["base_sha256"][:12],
              "python": "3.11", "openpyxl": stack[0], "pandas": stack[1], "numpy": stack[2],
              "libreoffice": rt["libreoffice"]}
    if not train_inputs:
        for k in INPUT_KEYS:
            mf.pop(k, None)
    return mf


# ─────────────────────────────────────────────────────────────── checks on what gets named
def _task_files(d: Path) -> list:
    return sorted(p for p in Path(d).glob("*.md") if p.stem not in _SIDECARS)


def _evidence_rows(run_dir, epoch: int, evidence_dirs, facts, prior_library) -> list:
    """The evidence rows to name, verified against `evidence_dirs`, the facts and the loop's paths.
    All-epochs facts carry the EVIDENCE INDEX (already checked by derive_facts); current-epoch
    facts carry none, and then exactly one dir, the newest, is allowed."""
    from src.orchestrator.loop import EpochPaths
    from src.orchestrator.prompt_author import EvidenceRow
    from src.orchestrator.skill_library import library_sha256
    want = [Path(os.path.realpath(str(d))) for d in evidence_dirs]
    if not want:
        raise RenderError("no evidence dirs given")
    newest = Path(os.path.realpath(str(EpochPaths(Path(run_dir), epoch).digests)))
    if want[-1] != newest:
        raise RenderError(f"the newest evidence dir must be {newest}, got {want[-1]}")
    if facts.evidence_index:
        rows = list(facts.evidence_index)
        if [Path(r.path) for r in rows] != want:
            raise RenderError(f"evidence_dirs {[str(p) for p in want]} differ from the facts' "
                              f"EVIDENCE INDEX {[r.path for r in rows]}")
        if not rows[-1].current:
            raise RenderError("the facts' EVIDENCE INDEX does not mark its last row current")
    else:
        if len(want) != 1:
            raise RenderError("several evidence dirs but the facts carry no EVIDENCE INDEX; "
                              "derive the facts with `evidence=` (all-epochs mode)")
        rows = [EvidenceRow(epoch=epoch, path=str(newest),
                            library="" if epoch == 1 else f"skills_v{epoch - 1}",
                            library_sha12=(library_sha256(prior_library)[:12]
                                           if prior_library else ""),
                            task_files=len(_task_files(newest)), passed=facts.passed,
                            episodes=facts.n_episodes, rate_checked=True, current=True)]
    cur = rows[-1]
    if (cur.passed, cur.episodes) != (facts.passed, facts.n_episodes):
        raise RenderError(f"the newest dir re-derives {cur.passed}/{cur.episodes} but the facts "
                          f"say {facts.passed}/{facts.n_episodes}")
    return rows


def _fence(body: str) -> str:
    """A code fence longer than any backtick run inside `body`, so a skill's own fences stay inside."""
    runs = [len(m.group(0)) for m in re.finditer(r"`+", body)]
    return "`" * max(3, max(runs, default=0) + 1)


def _prior_skill_set(epoch: int, prior_library, newest: Path, staging: Path,
                     earlier_libraries: bool = False) -> tuple:
    """(section text, record) for epoch >= 2; ("", {}) for epoch 1. The prior set is skills_v{N-1};
    when the newest digest dir carries _library/ it must be byte-identical to it (skill_library.tree),
    because the prompt says so."""
    from src.orchestrator import skill_library as sl
    if epoch == 1:
        if prior_library is not None:
            raise RenderError("epoch 1 has no prior skill set, but prior_library was given")
        return "", {}
    if prior_library is None:
        raise RenderError(f"epoch {epoch} needs prior_library (skills_v{epoch - 1})")
    lib = Path(prior_library)
    label = f"skills_v{epoch - 1}"
    if lib.name != label or lib.is_symlink() or not lib.is_dir():
        raise RenderError(f"prior_library must be the real directory {label}, got {lib}")
    if sl.symlinks(lib):
        raise RenderError(f"{lib} holds symlinks")
    copy = newest / sl.LIBRARY_DIR
    has_copy = copy.is_dir()
    if has_copy and sl.tree(copy) != sl.tree(lib):
        raise RenderError(f"{copy} is not byte-identical to {lib}; the prompt would misstate it")
    skills = sorted(d for d in lib.iterdir() if d.is_dir())
    out = ["## The skill set you are revising", ""]
    first = (f"Your staging directory, `{staging}`, starts with `{label}`: its files are seeded "
             f"there before you start")
    first += (f", byte-identical to `{copy}/`." if has_copy else ".")
    first += " No other previous or peer skill version is on your filesystem."
    if earlier_libraries:
        first += (" Each earlier evidence directory from epoch 2 on holds, in its own `_library/`, "
                  "the library that produced it.")
    out.append(first)
    out.append("")
    files = sorted(p for p in lib.rglob("*") if p.is_file())
    if not files:
        out.append(f"`{label}` holds no skills, so your staging directory starts empty.")
    else:
        # EVERY file the staging seed copies, not only <skill>/SKILL.md: the countersigned prompt
        # must cover exactly what the evolver starts with.
        out.append(f"`{label}` holds these files, reproduced verbatim:")
        for f in files:
            rel = f.relative_to(lib).as_posix()
            raw = f.read_bytes()
            try:
                body = raw.decode("utf-8")
            except UnicodeDecodeError:
                out += ["", f"`{rel}`: a binary file of {len(raw)} B, not reproduced."]
                continue
            fence = _fence(body)
            out += ["", f"`{rel}`:", "", fence, body.rstrip("\n"), fence]
    out += ["", "You may keep, rewrite, merge, split or delete any of them. Deleting a rule the "
                "evidence does not support is not a regression."]
    rec = {"label": label, "path": str(lib), "sha256": sl.library_sha256(lib),
           "skills": [d.name for d in skills], "library_copy_checked": has_copy}
    return "\n".join(out), rec


def _grader_dir(run_dir, machine_facts: dict) -> tuple:
    """(absolute path, same_program claim allowed?). No workbook may sit in it."""
    g = Path(machine_facts.get("grader_dir") or Path(run_dir).absolute() / GRADER_DIRNAME)
    g = g.absolute()
    if not g.is_dir():
        raise RenderError(f"no grader directory at {g}")
    books = [p for p in g.rglob("*") if p.suffix.lower() in SPREADSHEET_SUFFIXES + (".csv",)]
    if books:
        raise RenderError(f"the grader directory {g} holds workbooks {books[:2]}")
    same = False
    man = g / "MANIFEST.json"
    if man.is_file():
        files = json.loads(man.read_text()).get("files") or []
        same = bool(files) and all(f.get("same_program") is True for f in files)
    return g, same


def _grader_tooling_absent(machine_facts: dict):
    """None for an evolver not on the solver's image (no machine facts: nothing is claimed about its
    machine). On the solver's image: True once daytona_evolver.evolver_image_grader_tooling()
    confirms from the evolver's Dockerfile that the grader's tooling (/opt/evoskill, docker) is not in
    it; RenderError otherwise, since the prompt says it is not there (and it must not be)."""
    if not machine_facts:
        return None
    from src.orchestrator.daytona_evolver import evolver_image_grader_tooling
    absent, det, _rec = evolver_image_grader_tooling()
    if not absent:
        raise RenderError(f"the evolver image is not confirmed free of the grader's tooling: {det}")
    return True


def grader_sentence(grader, same_program: bool, tooling_absent) -> str:
    """The prompt's sentence about the clean grader: where it is, that it is the same program (when
    its MANIFEST proves it), that it is there to READ (always), and, when tooling_absent is True (the
    evolver's image, checked by _grader_tooling_absent), that the grading setup is not on the
    evolver's machine."""
    return (f"The grader's code is in `{grader}`"
            + (": the grader's own implementation with comments and docstrings removed, "
               "the same program." if same_program else ".")
            + GRADER_READ_ONLY + (GRADER_NOT_HERE if tooling_absent is True else ""))


def _references_dir(run_dir, machine_facts: dict, newest: Path) -> dict:
    """The TRAINING reference workbooks dir, checked before it is named: it exists, holds
    workbooks, and every first-level subdir is the stem of a task file in the newest (train-only)
    evidence dir, so no held-out task's reference can be described by this prompt."""
    d = Path(machine_facts.get("references_dir") or Path(run_dir).absolute() / REFERENCES_DIRNAME)
    d = d.absolute()
    if not d.is_dir():
        raise RenderError(f"train_references is on but there is no references dir at {d}")
    low = str(d).lower()
    if "spreadsheetbench" in low or "evolver_inputs_train" in low:
        raise RenderError(f"references dir {d} must not live in the dataset tree or the inputs dir")
    stems = {p.stem for p in _task_files(newest)}
    subdirs = sorted(p.name for p in d.iterdir() if p.is_dir())
    stray = [s for s in subdirs if s not in stems]
    if stray:
        raise RenderError(f"references dir {d} holds {len(stray)} subdir(s) with no train task "
                          f"file in {newest.name}: {stray[:3]}")
    books = [p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in SPREADSHEET_SUFFIXES]
    if not books:
        raise RenderError(f"references dir {d} holds no workbook")
    return {"dir": str(d), "n": len(books),
            "mb": round(sum(p.stat().st_size for p in books) / 1e6, 1)}


def _inputs(run_dir, machine_facts: dict) -> dict:
    from src.orchestrator.daytona_evolver import inputs_dir
    miss = [k for k in INPUT_KEYS if k not in machine_facts]
    if miss:
        raise RenderError(f"train_inputs is on but the machine facts lack {miss}")
    if str(machine_facts["inputs_dir"]) != str(inputs_dir(run_dir)):
        raise RenderError(f"machine facts name inputs dir {machine_facts['inputs_dir']}, not this "
                          f"run's {inputs_dir(run_dir)}")
    return {"dir": str(machine_facts["inputs_dir"]), "n": machine_facts["n_inputs"],
            "mb": machine_facts["inputs_mb"]}


def _check_outcomes(counts: dict, facts) -> bool:
    """True when all four are computed. Refuses counts that disagree with the facts, and any
    remaining grader failure (no number is reported while one remains)."""
    if counts.get("reportable") is False:
        raise RenderError(f"the leg's outcomes are not reportable (status {counts.get('status')!r}). "
                          f"No number is reported while that holds, so no prompt is rendered")
    if counts.get("passed") != facts.passed:
        raise RenderError(f"outcome counts say passed={counts.get('passed')} but the facts say "
                          f"{facts.passed}")
    rest = [counts.get(k) for k in OUTCOME_KEYS[1:]]
    if all(v is None for v in rest):
        return False
    if any(not isinstance(v, int) or v < 0 for v in rest):
        raise RenderError(f"outcome counts are partial or malformed: {counts}")
    if sum(rest) + counts["passed"] != facts.n_episodes:
        raise RenderError(f"outcome counts {counts} do not sum to the facts' {facts.n_episodes} "
                          f"episodes")
    if counts["grader_failed"]:
        raise RenderError(f"{counts['grader_failed']} episode(s) are 'grader failed'. Re-grade "
                          f"from the saved output and halt if it still fails; no prompt reports a "
                          f"number while one remains")
    return True


def _exhaustion_fields_seen(newest: Path) -> int:
    from src.orchestrator.prompt_author import _EXH
    n = 0
    for p in _task_files(newest):
        head = "\n".join(p.read_text(errors="replace").splitlines()[:40])
        n += bool(_EXH.search(head))
    return n


# ─────────────────────────────────────────────────────────────── the text
def _facts_section(facts, counts: dict, computed: bool, newest: Path) -> str:
    T, S, E, P = facts.n_tasks, facts.n_samples, facts.n_episodes, facts.passed
    if T * S != E:
        raise RenderError(f"{T} tasks x {S} samples != {E} episodes: the batch line would be false")
    out = ["## The facts", "", BATCH_LINE.format(T=T, S=S, E=E, P=P), ""]
    if computed:
        out.append(f"Outcomes of the {E} episodes: {P} passed, {counts['graded_wrong']} graded and "
                   f"wrong, {counts['not_eligible']} not eligible to grade, and "
                   f"{counts['grader_failed']} grader failed.")
    else:
        out.append(f"Outcomes of the {E} episodes: {P} passed. How the other {E - P} divide "
                   f"between graded and wrong, not eligible to grade and grader failed was not "
                   f"computed for this batch.")
    out.append("")
    if S > 1:
        out.append(f"Each task has {S} samples, so a task's outcome is a rate, not a single draw. "
                   f"{facts.reliable} tasks passed every sample, {facts.unstable} passed some but "
                   f"not all (the same task and the same solver gave different outcomes), and "
                   f"{facts.never_passes} passed no sample.")
    else:
        out.append("Each task has one sample, so a task's outcome is a single draw.")
    seen = _exhaustion_fields_seen(newest)
    if seen and seen != facts.result_headers_parsed:
        raise RenderError(f"{seen} of {facts.result_headers_parsed} task files record the turn "
                          f"ceiling field; a partial parse cannot be stated as a count")
    cut = []
    if not seen:
        cut.append("The task files of this batch do not record whether an episode was cut off at "
                   "the turn ceiling, so no count is stated.")
    elif facts.exhausted_episodes:
        reasons = sorted(facts.termination.items())
        if len(reasons) == 1 and reasons[0][1] == facts.exhausted_episodes:
            why = f"every one with the termination reason `{reasons[0][0]}`"
        else:
            why = "termination reasons " + ", ".join(f"`{k}` ({v})" for k, v in reasons)
        cut.append(f"{facts.exhausted_episodes} episodes were cut off at the turn ceiling, {why}. "
                   f"A cut-off episode is graded on whatever workbook it had saved by then, or is "
                   f"not eligible to grade if it saved none, so it is also counted in the outcomes "
                   f"above. {facts.tasks_with_exhausted} tasks lost at least one sample this way, "
                   f"and {facts.tasks_all_exhausted} lost every sample this way.")
    else:
        cut.append("No episode was cut off at the turn ceiling.")
    if facts.turns_max:
        cut.append(f"Episodes used {facts.turns_min}..{facts.turns_max} agent turns.")
    out += ["", " ".join(cut)]
    extra = []
    if facts.variant:
        extra.append(f"Retrieval variant: `{facts.variant}`.")
    if facts.basis_counts:
        extra.append("Scoring basis, tasks per basis: "
                     + ", ".join(f"`{k}` {v}" for k, v in sorted(facts.basis_counts.items())) + ".")
    if extra:
        out += ["", " ".join(extra)]
    gf = (GRADER_FAILED_POLICY[:-1] + ", so none remains in this batch." if computed
          else GRADER_FAILED_POLICY)
    out += ["", "Each episode ends in one of four outcomes:",
            "- **passed**: the grader ran on the saved workbook and every graded cell matched.",
            "- **graded and wrong**: the grader ran on the saved workbook and at least one graded "
            "cell did not match.",
            "- **not eligible to grade**: the solver saved no output workbook, so there was nothing "
            "to grade; the grader's recorded output reads `Output file not found`. Its 0 is the "
            "absence of a verdict, not a wrong verdict.",
            "- **grader failed**: a workbook was saved but the grader did not complete on it: it "
            "raised, or its recalculation or data-table step failed. " + gf]
    return "\n".join(out)


def _evidence_section(rows, newest: Path, *, changed_cells: bool = False,
                      require_full_grader_output: bool = True) -> tuple:
    many = len(rows) > 1
    out = ["## Where the evidence is", "",
           ("There is one evidence directory per epoch so far, oldest first:" if many
            else "There is one evidence directory:"), ""]
    for r in rows:
        if r.library:
            s = (f"- `{r.path}`: epoch {r.epoch}, produced by the solver carrying `{r.library}` "
                 f"(sha256:{r.library_sha12}), whose exact files are in `{r.path}/_library/`. "
                 f"{r.task_files} task files.")
        else:
            s = (f"- `{r.path}`: epoch {r.epoch}, produced by the solver carrying no skills (the "
                 f"base leg). {r.task_files} task files.")
        if r.current:
            s += " This is the batch every figure above describes."
        elif r.rate_checked:
            s += f" {r.passed} of {r.episodes} episodes passed."
        else:
            s += " No episode count is stated for it: no leg result file is declared for it."
        out.append(s)
    infos = _digest_infos(rows, changed_cells=changed_cells,
                          require_full_grader_output=require_full_grader_output)
    with_changed = [p for p, i in infos if i["changed_cells"]["on"]]
    out += ["", _holds_text(infos)]
    if with_changed and len(with_changed) != len(rows):
        out += ["", "Task files in " + ", ".join(f"`{p}`" for p in with_changed)
                + " also list, for each sample, " + _changed_text([i for p, i in infos
                                                                     if p in with_changed]) + "."]
    tail = "Every directory named in this prompt is train-only by construction."
    if many:
        tail += " The earlier directories are there to read on demand."
    out += ["", tail]
    return "\n".join(out), {"changed_cells_dirs": with_changed,
                            "digest_info": [_info_record(p, i) for p, i in infos]}


def _digest_infos(rows, *, changed_cells: bool, require_full_grader_output: bool) -> list:
    """[(path, header)] per evidence row, each DIGEST_INFO.json verified against its dir's bytes
    (digest_info.read), then held to the grants: the changed-cells list only when granted (default
    off; its lines carry training input values), the grader's output only in full unless
    the caller waives it."""
    from src.orchestrator import digest_info as DI
    infos = []
    for r in rows:
        try:
            infos.append((r.path, DI.read(r.path)))
        except DI.DigestInfoError as e:
            raise RenderError(f"evidence dir {e}") from None
    cc = [p for p, i in infos if i["changed_cells"]["on"]]
    if cc and not changed_cells:
        raise RenderError(f"{len(cc)} evidence dir(s) carry the changed-cells list ({cc[:2]}) but it "
                          f"is not granted (default off: its lines carry training input cell "
                          f"values). Rebuild them without EVOSKILL_DIGEST_CHANGED_CELLS=1, or grant it")
    clipped = [(p, i["grader_output"]["clipped"]) for p, i in infos
               if not i["grader_output"]["full"] or i["grader_output"]["clipped"]]
    if clipped and require_full_grader_output:
        raise RenderError(f"the grader's output is clipped in {clipped[:2]} (path, diffs clipped); "
                          f"the evolver sees it in full. Rebuild with "
                          f"EVOSKILL_DIGEST_GRADER_OUTPUT_FULL=1 (the builder default)")
    return infos


def _info_record(path, i: dict) -> dict:
    to = i["tool_output"]
    return {"path": str(path),
            "sha256": hashlib.sha256((Path(path) / "DIGEST_INFO.json").read_bytes()).hexdigest(),
            "grader_output_full": i["grader_output"]["full"],
            "grader_clipped": i["grader_output"]["clipped"],
            "changed_cells": i["changed_cells"]["on"],
            "changed_lists_capped": i["changed_cells"]["lists_capped"],
            "tool_clip": [to["clip_head"], to["clip_tail"]],
            "tool_results_clipped": to["results"]["clipped"],
            "tool_calls_clipped": to["calls"]["clipped"],
            "clipped_items_total": i["clipped_items_total"]}


def _changed_text(infos: list) -> str:
    """The changed-cells clause, from the headers of the dirs that carry the list."""
    n = sum(i["changed_cells"]["sections"] for i in infos)
    capped = sum(i["changed_cells"]["lists_capped"] for i in infos)
    s = "the cells the solver changed, comparing its output with the input"
    if capped:
        return s + (f" ({capped} of the {n} lists are cut short, each ending in a line starting "
                    f"`[DIGEST CAP —` that says how many changed cells it does not list)")
    return s + " (every changed cell is listed)"


def _holds_text(infos: list) -> str:
    """What each task file holds, stated only as the verified headers record it."""
    n_sec = sum(i["grader_output"]["sections"] for p, i in infos)
    n_clip = sum(i["grader_output"]["clipped"] for p, i in infos)
    what = ("whether it passed, how many graded cells matched, and each mismatched cell with its "
            "expected and produced value (a sample that saved no workbook reads `Output file not "
            "found`)")
    if n_clip == 0:
        s = (f"Each task file holds the instruction and the name of its input file; for each "
             f"sample, the grader's own output in full, exactly as the grader wrote it: {what}; ")
    else:
        s = (f"Each task file holds the instruction and the name of its input file; for each "
             f"sample, the grader's own output: {what}. {n_clip} of the {n_sec} grader outputs are "
             f"shortened in the middle, at a line starting `[DIGEST CLIP —` that says how many "
             f"characters are not shown; ")
    on = [i for p, i in infos if i["changed_cells"]["on"]]
    if on and len(on) == len(infos):
        s += "for each sample, " + _changed_text(on) + "; "
    s += (f"and each sample's transcript: the solver's messages, {TOOL_CALL_SHOWN} and each tool's "
          f"output, without the solver's system prompt.")
    tos = [i["tool_output"] for p, i in infos]
    rc = sum(t["results"]["clipped"] for t in tos)
    ri = sum(t["results"]["items"] for t in tos)
    cc = sum(t["calls"]["clipped"] for t in tos)
    ci = sum(t["calls"]["items"] for t in tos)
    if rc or cc:
        lims = sorted({(t["clip_head"], t["clip_tail"]) for t in tos})
        if len(lims) == 1:
            h, t = lims[0]
            s += (f" A tool call or tool output longer than {h + t:,} characters is shortened to its "
                  f"first {h:,} and last {t:,} characters")
        else:
            s += (" A long tool call or tool output is shortened to its first and last characters "
                  "(the limits differ between directories)")
        s += (f", with a line starting `[DIGEST CLIP —` at the cut that says how many characters "
              f"are not shown: {rc:,} of the {ri:,} tool outputs and {cc:,} of the {ci:,} tool calls "
              f"are shortened this way.")
    else:
        s += " No tool call or tool output is shortened."
    if any(str(i.get("transcript", {}).get("skill_loads", "")).startswith("replaced")
           for p, i in infos):
        s += (" A tool output that is a skill body the solver loaded is replaced by a one-line "
              "`[SKILL LOADED: ...]` line naming the skill.")
    return s


# ─────────────────────────────────────────────────────────────── the rollouts layout
def _rollout_module():
    from src.orchestrator import rollout_evidence as RE
    return RE


def _n(k: int, noun: str) -> str:
    return f"{k:,} {noun}" + ("" if k == 1 else "s")


def _kb(b) -> str:
    return f"{b / 1024:,.0f} KB"


def skill_labels(names) -> dict:
    """{skill name: letter}: A..Z, then AA, AB, ... in sorted-name order (the per-rollout lines' legend)."""
    out = {}
    for i, name in enumerate(sorted(names)):
        s, j = "", i + 1
        while j:
            j, r = divmod(j - 1, 26)
            s = chr(65 + r) + s
        out[name] = s
    return out


def _label_key(label: str) -> tuple:
    return (len(label), label)


def skill_description(text: str):
    """The `description` of a SKILL.md's YAML frontmatter as one line (a block scalar's lines joined,
    surrounding quotes dropped), or None when the file has no frontmatter or no description key."""
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    lines = (text[3:end] if end != -1 else "").splitlines()
    for i, ln in enumerate(lines):
        m = re.match(r"^description:[ \t]*(.*)$", ln)
        if not m:
            continue
        v = m.group(1).strip()
        if v in ("", "|", ">", "|-", ">-", "|+", ">+"):
            cont = []
            for x in lines[i + 1:]:
                if x.strip() and not x[:1].isspace():
                    break
                cont.append(x.strip())
            v = " ".join(c for c in cont if c)
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        return " ".join(v.split()) or None
    return None


def _regraded_map(regraded) -> dict:
    """{(task id, sample): outcome after the loop's re-grade} from the loop's re-grade rows."""
    out = {}
    for x in regraded or []:
        oc = x.get("outcome")
        if oc not in ("passed", "graded_wrong"):
            raise RenderError(f"a re-graded sample must end passed or graded_wrong, got {x}")
        out[(str(x["task"]), int(x["sample"]))] = oc
    return out


def leg_sample_scores(leg_path) -> dict:
    """{(task id, sample): recorded score} from a leg JSON: per_task indexed by task_ids (SSB v2) or
    keyed by task id. The record every downstream number reads."""
    d = json.loads(Path(leg_path).read_text())
    pt = d.get("per_task") or {}
    tids = d.get("task_ids")
    out = {}
    if isinstance(tids, list) and tids:
        for i, t in enumerate(tids):
            for k, v in enumerate(pt.get(str(i)) or []):
                out[(str(t), k)] = v
    else:
        for t, vs in pt.items():
            for k, v in enumerate(vs if isinstance(vs, list) else []):
                out[(str(t), k)] = v
    return out


def _rollout_evidence(rows, *, facts, counts: dict, computed: bool, leg, regraded: dict) -> tuple:
    """([(EvidenceRow, INDEX)], {(task, sample): final outcome}) for the rollouts layout.

    Every granted dir's INDEX.json is read with rollout_evidence.read_index (every listed file at its
    recorded sha256; the loop's evidence steps re-derive every row from the bytes before this runs)
    and held to what the prompt says of it: its dir and epoch, its producing library (label, sha,
    the _library/ on disk holding exactly the INDEX's skills), its pass count where one is stated.
    The NEWEST dir's INDEX must be built from this very leg file, hold one row per train task x
    sample of it, agree with the leg's recorded score on every row's pass/fail, and give the outcome
    counts the facts section states (after the loop's re-grade, `regraded`)."""
    RE = _rollout_module()
    from src.orchestrator.outcomes import PASS_THRESHOLD
    pairs = []
    for r in rows:
        d = Path(r.path)
        try:
            idx = RE.read_index(d)
        except (RE.EvidenceError, OSError, ValueError) as e:
            raise RenderError(f"evidence dir {d.name} has no readable rollout evidence ({e}); the rollouts "
                              f"layout names every granted dir's INDEX (src/orchestrator/rollout_evidence.py "
                              f"build)") from None
        if idx.get("evidence_dir") != d.name or idx.get("epoch") not in (None, r.epoch):
            raise RenderError(f"{d.name}/INDEX.json describes {idx.get('evidence_dir')!r} epoch "
                              f"{idx.get('epoch')!r}, not {d.name} epoch {r.epoch}")
        if (idx.get("fidelity") or {}).get("clipped") != 0:
            raise RenderError(f"{d.name}/INDEX.json does not record its traces as unclipped; the prompt "
                              f"says nothing in them is shortened by the loop")
        lib = idx.get("producing_library")
        if bool(lib) != bool(r.library):
            raise RenderError(f"{d.name}/INDEX.json names producing library "
                              f"{(lib or {}).get('label')!r}, the EVIDENCE INDEX says {r.library or None!r}")
        if lib:
            if lib.get("label") != r.library or lib.get("sha256_12") != r.library_sha12:
                raise RenderError(f"{d.name}/INDEX.json names {lib.get('label')} sha256:{lib.get('sha256_12')}, "
                                  f"the EVIDENCE INDEX {r.library} sha256:{r.library_sha12}")
            ld = d / str(lib.get("path") or "")
            if ld.is_symlink() or not ld.is_dir() or ld.name != RE.LIBRARY_DIR:
                raise RenderError(f"{d.name}: no {RE.LIBRARY_DIR}/ although its INDEX names {lib.get('label')}")
            on_disk = sorted(p.parent.name for p in ld.glob("*/SKILL.md"))
            if on_disk != sorted(s.get("name") for s in lib.get("skills") or []):
                raise RenderError(f"{d.name}/{RE.LIBRARY_DIR}/ holds skills {on_disk}, its INDEX names "
                                  f"{[s.get('name') for s in lib.get('skills') or []]}")
        c = idx.get("counts") or {}
        if r.rate_checked and (c.get("passed"), c.get("rollouts")) != (r.passed, r.episodes):
            raise RenderError(f"{d.name}/INDEX.json counts {c.get('passed')}/{c.get('rollouts')} passed, "
                              f"its leg {r.passed}/{r.episodes}")
        pairs.append((r, idx))
    r, idx = pairs[-1]
    d = Path(r.path)
    want = hashlib.sha256(Path(leg).read_bytes()).hexdigest()
    got = (idx.get("leg") or {}).get("sha256")
    if got != want:
        raise RenderError(f"{d.name}/INDEX.json was built from {(idx.get('leg') or {}).get('file')} "
                          f"sha256:{str(got)[:12]}, not the leg this prompt describes ({Path(leg).name} "
                          f"sha256:{want[:12]})")
    rws = idx.get("rows") or []
    keys = [(str(x.get("task")), int(x.get("sample"))) for x in rws]
    if len(set(keys)) != len(keys) or len(keys) != facts.n_episodes \
            or len({t for t, _ in keys}) != facts.n_tasks:
        raise RenderError(f"{d.name}/INDEX.json holds {len(keys)} rows over {len({t for t, _ in keys})} "
                          f"tasks; the batch is {facts.n_tasks} tasks x {facts.n_samples} samples")
    scores = leg_sample_scores(leg)
    if set(scores) != set(keys):
        raise RenderError(f"{d.name}/INDEX.json rows are not the samples of {Path(leg).name}")
    stray = sorted(set(regraded) - set(keys))
    if stray:
        raise RenderError(f"re-graded samples {stray[:3]} are not rows of {d.name}/INDEX.json")
    final = {}
    for x, key in zip(rws, keys):
        oc = x.get("outcome")
        if oc not in OUTCOME_TOKEN:
            raise RenderError(f"{key[0]} s{key[1]}: outcome {oc!r} in {d.name}/INDEX.json")
        if key in regraded:
            if oc != "grader_failed":
                raise RenderError(f"{key[0]} s{key[1]} was re-graded, but its INDEX row is {oc}")
            oc = regraded[key]
        s = scores[key]
        if (s is not None and s >= PASS_THRESHOLD) != (oc == "passed"):
            raise RenderError(f"{key[0]} s{key[1]}: {d.name}/INDEX.json says {oc}, the leg records score {s}")
        if x.get("score") != s:
            raise RenderError(f"{key[0]} s{key[1]}: {d.name}/INDEX.json records score {x.get('score')}, the "
                              f"leg {s}")
        final[key] = oc
    tally = collections.Counter(final.values())
    if tally.get("passed", 0) != facts.passed:
        raise RenderError(f"{d.name}/INDEX.json gives {tally.get('passed', 0)} passed, the facts {facts.passed}")
    if computed and any(tally.get(k, 0) != counts.get(k) for k in OUTCOME_KEYS):
        raise RenderError(f"{d.name}/INDEX.json gives outcomes {dict(tally)}, the facts state "
                          f"{ {k: counts.get(k) for k in OUTCOME_KEYS} }")
    return pairs, final


def _trace_stats(pairs) -> dict:
    """Read from the trace bytes of every granted dir: sizes, the longest line, and whether any
    message carries reasoning text (the only thing that lets the prompt say "its reasoning")."""
    sizes, longest, reasoning = [], 0, False
    for r, idx in pairs:
        for x in idx.get("rows") or []:
            b = (Path(r.path) / x["trace"]).read_bytes()
            sizes.append(len(b))
            longest = max(longest, max((len(ln) for ln in b.split(b"\n")), default=0))
            reasoning = reasoning or bool(_REASONING_BYTES.search(b))
    if not sizes:
        raise RenderError("the granted rollout evidence holds no trace")
    return {"traces": len(sizes), "avg_bytes": round(sum(sizes) / len(sizes)), "max_bytes": max(sizes),
            "longest_line_bytes": longest, "reasoning": reasoning}


def table_cells(row: dict, regraded_outcome=None):
    """(matched, graded) as a per-rollout line states them, or None: never for a re-graded sample (its
    grade file is the failed first grading), never for `no-output` (the grader's "0/0" there counts no
    workbook), and not when the grader's output gave no count. #208 applies the same rule."""
    if regraded_outcome or row.get("outcome") == "not_eligible" or row.get("cells_graded") is None:
        return None
    return int(row["cells_matched"]), int(row["cells_graded"])


def rollout_table_lines(index: dict, labels: dict, regraded: dict) -> list:
    """The per-rollout lines (ROLLOUT_* format) of one INDEX: the legend when `labels`, then one line per
    task file stem, sorted, its samples in order. Every line is parsed back before it is returned."""
    by = collections.defaultdict(dict)
    for x in index.get("rows") or []:
        by[str(x["task_file"])][int(x["sample"])] = x
    out = []
    if labels:
        out.append(LEGEND_PREFIX + ", ".join(f"{labels[n]} = {n}"
                                             for n in sorted(labels, key=lambda n: _label_key(labels[n]))))
    for stem in sorted(by):
        cells = []
        for k in sorted(by[stem]):
            x = by[stem][k]
            rg = regraded.get((str(x["task"]), k))
            s = f"s{k} " + (OUTCOME_TOKEN[rg] + REGRADED_MARK if rg else OUTCOME_TOKEN[x["outcome"]])
            if table_cells(x, rg) is not None:
                s += f" {int(x['cells_matched'])}/{int(x['cells_graded'])} cells"
            t = x.get("turns")
            s += f", {t if isinstance(t, int) and not isinstance(t, bool) else '?'} turns"
            if x.get("turn_cap_hit"):
                s += " (cap)"
            loaded = list(x.get("skills_loaded") or [])
            if labels:
                miss = [n for n in loaded if n not in labels]
                if miss:
                    raise RenderError(f"{stem} s{k} loaded {miss}, which are not in the batch's library")
                s += ", loaded " + (" ".join(sorted((labels[n] for n in loaded), key=_label_key)) or "none")
            elif loaded:
                raise RenderError(f"{stem} s{k} loaded {loaded}, but no skill was available to the batch")
            cells.append(s)
        out.append(f"{stem} | " + " | ".join(cells))
    back = parse_rollout_table("\n".join(out))
    if back["bad"] or len(back["tasks"]) != len(by) or bool(labels) != (back["legend"] is not None):
        raise RenderError(f"a per-rollout line does not parse back: {back['bad'][:2]}")
    return out


def parse_rollout_table(text: str) -> dict:
    """The per-rollout lines of a prompt, parsed (the ROLLOUT_* format; #208 re-derives each one).
    {"legend": {letter: skill name} or None, "tasks": {stem: {k: fields}}, "bad": [lines],
    "residual": the text without the table and legend lines}. A line starting "<stem> | s<digit>" IS a
    table line: one whose samples do not all parse, or a stem listed twice, is "bad", never skipped. A
    line matching LEGEND_RE is the legend; a second one is "bad"."""
    legend, tasks, bad, keep = None, {}, [], []
    for ln in text.split("\n"):
        m = ROLLOUT_LINE_RE.match(ln)
        if m:
            got = {}
            for part in m.group("rest").split(" | "):
                sm = ROLLOUT_SAMPLE_RE.match(part)
                if not sm or int(sm.group("k")) in got:
                    got = None
                    break
                got[int(sm.group("k"))] = sm.groupdict()
            if got is None or m.group("stem") in tasks:
                bad.append(ln)
            else:
                tasks[m.group("stem")] = got
            continue
        lm = LEGEND_RE.match(ln)
        if lm:
            items = [x.split(" = ", 1) for x in lm.group("body").split(", ")]
            lg = {a: b for a, b in items}
            if legend is not None or len(lg) != len(items) or len(set(lg.values())) != len(items):
                bad.append(ln)
            else:
                legend = lg
            continue
        keep.append(ln)
    return {"legend": legend, "tasks": tasks, "bad": bad, "residual": "\n".join(keep)}


def _skill_set_table(epoch: int, prior_library, newest: Path, staging: Path,
                     earlier_libraries: bool = False) -> tuple:
    """(a) The skill set being revised as a TABLE, never its text: per skill the `description` of its
    SKILL.md and its files with sizes, and where they are (the staging dir, the newest dir's
    _library/). The same checks as _prior_skill_set; ("", {}) for epoch 1."""
    from src.orchestrator import skill_library as sl
    if epoch == 1:
        if prior_library is not None:
            raise RenderError("epoch 1 has no prior skill set, but prior_library was given")
        return "", {}
    if prior_library is None:
        raise RenderError(f"epoch {epoch} needs prior_library (skills_v{epoch - 1})")
    lib = Path(prior_library)
    label = f"skills_v{epoch - 1}"
    if lib.name != label or lib.is_symlink() or not lib.is_dir():
        raise RenderError(f"prior_library must be the real directory {label}, got {lib}")
    if sl.symlinks(lib):
        raise RenderError(f"{lib} holds symlinks")
    copy = newest / sl.LIBRARY_DIR
    has_copy = copy.is_dir()
    if has_copy and sl.tree(copy) != sl.tree(lib):
        raise RenderError(f"{copy} is not byte-identical to {lib}; the prompt would misstate it")
    skills = sorted(d for d in lib.iterdir() if d.is_dir())
    files = sorted(p for p in lib.rglob("*") if p.is_file())
    out = ["## The skill set you are revising", ""]
    first = (f"Your staging directory, `{staging}`, starts with `{label}`: its files are seeded "
             f"there before you start")
    first += (f", byte-identical to `{copy}/`." if has_copy else ".")
    first += " No other previous or peer skill version is on your filesystem."
    if earlier_libraries:
        first += (" Each earlier evidence directory from epoch 2 on holds, in its own `_library/`, "
                  "the library that produced it.")
    out += [first, ""]
    total = sum(p.stat().st_size for p in files)
    if not files:
        out.append(f"`{label}` holds no skills, so your staging directory starts empty.")
    else:
        lines = []
        for sd in skills:
            fs = sorted(p for p in sd.rglob("*") if p.is_file())
            sk = sd / "SKILL.md"
            if sk.is_file():
                desc = (skill_description(sk.read_text(errors="replace"))
                        or "(its SKILL.md frontmatter has no description)")
            else:
                desc = "(no SKILL.md)"
            lines += [sd.name, f"  description: {desc}",
                      "  files: " + (", ".join(f"{p.relative_to(sd).as_posix()} ({p.stat().st_size:,} B)"
                                               for p in fs) or "none")]
        loose = [p for p in files if p.relative_to(lib).parts[0] not in {d.name for d in skills}]
        if loose:
            lines += ["(outside any skill directory)",
                      "  files: " + ", ".join(f"{p.relative_to(lib).as_posix()} ({p.stat().st_size:,} B)"
                                              for p in loose)]
        body = "\n".join(lines)
        fence = _fence(body)
        where = f"`{staging}/`" + (f" or `{copy}/`" if has_copy else "")
        out += [f"This prompt does not reproduce the skills' text; read their files in {where}. "
                f"`{label}` holds {_n(len(skills), 'skill')} in {_n(len(files), 'file')} ({total:,} bytes). "
                f"For each skill: the `description` of its `SKILL.md`, the only part of it the solver sees "
                f"until it loads the skill, and its files, relative to the skill's directory:",
                "", fence, body, fence]
    out += ["", "You may keep, rewrite, merge, split or delete any of them. Deleting a rule the "
                "evidence does not support is not a regression."]
    rec = {"label": label, "path": str(lib), "sha256": sl.library_sha256(lib),
           "skills": [d.name for d in skills], "library_copy_checked": has_copy,
           "rendered": "table", "files": len(files), "bytes": total}
    return "\n".join(out), rec


def _rollouts_layout_text(pairs, stats: dict) -> str:
    """(d) What each evidence dir holds in the rollouts layout, every number read from the INDEX / the
    trace bytes (what the evidence holds is read, never assumed)."""
    RE = _rollout_module()

    def tot(key):
        return sum(int((idx.get("counts") or {}).get(key) or 0) for _, idx in pairs)
    kinds = {(idx.get("solver") or {}).get("kind") for _, idx in pairs}
    ctx_kinds = [{m.get("kind") for m in idx.get("solver_context") or []} for _, idx in pairs]
    no_grade = any(not x.get("grade") for _, idx in pairs for x in idx.get("rows") or [])
    trace = ("- `rollouts/<task file stem>/s<k>.trace.jsonl`: the rollout's whole conversation, one JSON "
             "object per line: a `header` line (task, sample, solver, model), then every message in order "
             "(`n` is its position), then an `end` line (the agent turns it used, whether the turn ceiling "
             "cut it off, and how it ended). The loop shortens nothing in it: the task message, the "
             "solver's replies")
    if stats["reasoning"]:
        trace += (" with their thinking as Claude Code displayed it (`reasoning`, summarized)"
                  if kinds == {"claude_code"} else " with the reasoning text its model API returned")
    trace += (", every tool call with its full arguments, every tool result as the solver received it, "
              "and the full text of every skill it loaded.")
    h = tot("harness_truncated_results")
    if h:
        trace += (f" The solver's own harness shortened {_n(h, 'tool result')} that exceeded its limit before "
                  f"the solver saw {'it' if h == 1 else 'them'}")
        trace += (f", marked inside the result by `{RE.HARNESS_TRUNCATION.rstrip()}`." if kinds == {"direct"}
                  else ".")
    if all("system_prompt" in k for k in ctx_kinds):
        trace += (" The solver's system prompt" + (" and tool definitions are" if all("tools" in k for k in ctx_kinds)
                                                  else " is")
                  + " stored once per directory in `solver_context/`, and message 0 of each trace names its file.")
    red = tot("redactions")
    if red:
        trace += (f" {_n(red, 'occurrence')} of a task name outside the training set "
                  f"{'was' if red == 1 else 'were'} replaced by `{RE.REDACTION}`.")
    grade = ("- `rollouts/<task file stem>/s<k>.grade.txt`: the grader's own output for that rollout, as "
             "the grader wrote it: whether it passed, how many graded cells matched, and each mismatched "
             "cell with its expected and produced value (a sample that saved no workbook reads `Output "
             "file not found`)" + ("; a rollout whose grading failed may have none." if no_grade else "."))
    index = ("- `INDEX.md`, and the same rows as JSON in `INDEX.json`: one row per rollout with its outcome, "
             "cells matched and graded, turns, whether the turn ceiling cut it off, how it ended, the skills "
             "it loaded and the available skills it did not load, and its files' paths and sizes "
             "(`INDEX.json` adds their hashes).")
    reader = all("python3 -c" in (Path(r.path) / RE.INDEX_MD).read_text(errors="replace") for r, _ in pairs)
    read = (f"A trace is {_kb(stats['avg_bytes'])} on average and {_kb(stats['max_bytes'])} at most, and one "
            f"line of it (one message) reaches {_kb(stats['longest_line_bytes'])}, so a viewer that shortens "
            f"long lines shows only part of it. Read traces with Python (`python3`): parse each line with "
            f"`json.loads` and print only the messages you need"
            + ("; `INDEX.md` shows a one-line reader." if reader else "."))
    return "\n".join(["Each evidence directory holds every rollout of its batch (one train task, one sample) "
                      "under `rollouts/`, where `<task file stem>` is the task's id with `/` replaced by "
                      "`_`:", trace, grade, index, "", read])


def _digests_summary_text(infos: list) -> str:
    """The old per-task digests in the rollouts layout: what DIGEST_INFO.json records, nothing more."""
    n_sec = sum(i["grader_output"]["sections"] for p, i in infos)
    n_clip = sum(i["grader_output"]["clipped"] for p, i in infos)
    s = ("Each directory also holds one `<task file stem>.md` per task, the task files counted above: an "
         "older, shorter summary of the same rollouts that the loop's own checks read. It holds, for each "
         "sample, the grader's output" + (" in full" if n_clip == 0 else
                                          f" ({n_clip} of the {n_sec} shortened in the middle)")
         + " and the transcript")
    tos = [i["tool_output"] for p, i in infos]
    rc = sum(t["results"]["clipped"] for t in tos)
    ri = sum(t["results"]["items"] for t in tos)
    cc = sum(t["calls"]["clipped"] for t in tos)
    ci = sum(t["calls"]["items"] for t in tos)
    if rc or cc:
        lims = sorted({(t["clip_head"], t["clip_tail"]) for t in tos})
        if len(lims) == 1:
            hd, tl = lims[0]
            s += (f", but a tool call or tool output longer than {hd + tl:,} characters is cut to its first "
                  f"{hd:,} and last {tl:,} characters")
        else:
            s += ", but a long tool call or tool output is cut to its first and last characters"
        s += f" ({rc:,} of the {ri:,} tool outputs and {cc:,} of the {ci:,} tool calls)"
    if any(str(i.get("transcript", {}).get("skill_loads", "")).startswith("replaced") for p, i in infos):
        s += ", and the text of a skill the solver loaded is replaced by a one-line `[SKILL LOADED: ...]` line"
    s += "."
    on = [(p, i) for p, i in infos if i["changed_cells"]["on"]]
    if on:
        s += (" The summaries in " + ", ".join(f"`{p}`" for p, _ in on) + " also list, for each sample, "
              + _changed_text([i for _, i in on]) + ".")
    return s + " The traces and grade files are the full record."


def _evidence_section_rollouts(pairs, stats: dict, *, changed_cells: bool = False,
                               require_full_grader_output: bool = True) -> tuple:
    """(b) + (d): every evidence dir with the skill snapshot its solver carried, then the layout."""
    many = len(pairs) > 1
    out = ["## Where the evidence is", "",
           ("There is one evidence directory per epoch so far, oldest first:" if many
            else "There is one evidence directory:"), ""]
    for r, idx in pairs:
        lib = idx.get("producing_library")
        if lib:
            names = [s["name"] for s in lib.get("skills") or []]
            s = (f"- `{r.path}`: epoch {r.epoch}, produced by the solver carrying `{r.library}` "
                 f"(sha256:{r.library_sha12}), whose exact files are in `{r.path}/_library/`. ")
            if names:
                s += (f"{'Its' if len(names) == 1 else 'All'} {_n(len(names), 'skill')} "
                      f"{'was' if len(names) == 1 else 'were'} available to every one of its rollouts: "
                      + ", ".join(f"`{n}`" for n in names) + ".")
            else:
                s += "That library holds no skill, so none was available to its rollouts."
        else:
            s = (f"- `{r.path}`: epoch {r.epoch}, produced by the solver carrying no skills (the base "
                 f"leg): no skill was available to its rollouts.")
        s += f" {r.task_files} task files."
        index_md = f"`{r.path}/INDEX.md`"
        if r.current:
            s += (f" This is the batch every figure above describes; its per-rollout scores are below "
                  f"and in {index_md}.")
        elif r.rate_checked:
            s += f" {r.passed} of {r.episodes} episodes passed; its per-rollout scores are in {index_md}."
        else:
            s += (f" No episode count is stated for it: no leg result file is declared for it. Its "
                  f"per-rollout scores are in {index_md}.")
        out.append(s)
    out += ["", _rollouts_layout_text(pairs, stats)]
    infos = _digest_infos([r for r, _ in pairs], changed_cells=changed_cells,
                          require_full_grader_output=require_full_grader_output)
    out += ["", _digests_summary_text(infos)]
    tail = "Every directory named in this prompt is train-only by construction."
    if many:
        tail += " The earlier directories are there to read on demand."
    out += ["", tail]
    RE = _rollout_module()
    return "\n".join(out), {
        "changed_cells_dirs": [p for p, i in infos if i["changed_cells"]["on"]],
        "digest_info": [_info_record(p, i) for p, i in infos],
        "rollout_index": [{"epoch": r.epoch, "path": str(Path(r.path) / RE.INDEX_JSON),
                           "sha256": hashlib.sha256((Path(r.path) / RE.INDEX_JSON).read_bytes()).hexdigest()}
                          for r, _ in pairs],
        "trace_stats": stats}


def _rollout_table_section(r, idx: dict, regraded: dict) -> tuple:
    """(c) The newest batch's per-rollout scores: an explanation, then the ROLLOUT_* lines in one fence."""
    lib = idx.get("producing_library")
    labels = skill_labels([s["name"] for s in (lib or {}).get("skills") or []])
    lines = rollout_table_lines(idx, labels, regraded)
    body = "\n".join(lines)
    fence = _fence(body)
    intro = (f"One line per train task of `{r.path}`, from its `INDEX.json`: the task file stem, then for "
             f"each sample `s<k>` its outcome (`passed`; `wrong` = graded and wrong; `no-output` = not "
             f"eligible to grade; `grader-failed`), the graded cells it matched out of the cells graded "
             f"(absent for `no-output`, and when the grader's output gives no count), the agent turns it used, `(cap)` when the turn "
             f"ceiling cut it off")
    if labels:
        intro += (", and the skills it loaded, by the letters of the first line (`none` = it loaded no "
                  "skill; a skill not listed was available and not loaded)")
    intro += "."
    if not labels:
        intro += " No skill was available to these rollouts."
    if regraded:
        intro += (" `(re-graded)` marks a sample whose grading failed and which the loop re-graded once from "
                  "its saved workbook: the outcome shown is the re-grade's, and its grade file holds the "
                  "failed first grading.")
    intro += (f" Sample k's full trace is `{r.path}/rollouts/<task file stem>/s<k>.trace.jsonl` and its "
              f"grader output `{r.path}/rollouts/<task file stem>/s<k>.grade.txt`.")
    text = "\n".join([ROLLOUT_SECTION, "", intro, "", fence, body, fence])
    return text, {"lines": len(lines), "bytes": len(body.encode("utf-8")),
                  "labels": {v: k for k, v in labels.items()}}


def _machine_section(mf: dict, refs, inputs, *, rollouts: bool = False) -> str:
    miss = [k for k in MACHINE_KEYS if k not in mf]
    if miss:
        raise RenderError(f"machine facts lack {miss}")
    out = ["## The machine you are on", "",
           f"You run in the solver's own image (recipe `{mf['image_recipe']} "
           f"sha256:{mf['image_sha12']}`) plus the Claude Code CLI; the grader's own tooling is not "
           f"in it. It has Python {mf['python']} "
           f"with openpyxl {mf['openpyxl']}, pandas {mf['pandas']} and numpy {mf['numpy']}, and "
           f"{mf['libreoffice']} with `soffice` on PATH: the versions the solver's episodes use. "
           f"The network reaches the model API only; no other host is reachable."]
    if inputs and rollouts:
        out += ["", f"The inputs dir is `{inputs['dir']}`. It holds the input workbook of every "
                    f"task that has an evidence file, and never a reference workbook: "
                    f"{inputs['n']} input workbooks, {inputs['mb']} MB in total. Its layout is "
                    f"`{inputs['dir']}/<task file stem>/<the file named on the Input file line of "
                    f"that task's `<task file stem>.md`>`."]
    elif inputs:
        out += ["", f"The inputs dir is `{inputs['dir']}`. It holds the input workbook of every "
                    f"task that has an evidence file, and never a reference workbook: "
                    f"{inputs['n']} input workbooks, {inputs['mb']} MB in total. Its layout is "
                    f"`{inputs['dir']}/<evidence file name without .md>/<the file named on that "
                    f"evidence file's Input file line>`."]
    have = ["the evidence directories", "the grader directory"]
    if refs:
        have.append("the reference workbooks directory")
    if inputs:
        have.append("the inputs dir")
    have.append("your staging directory")
    out += ["", "Nothing from any other run is on your filesystem. Paths in the "
                + ("traces" if rollouts else "transcripts") + ", such "
                "as `/workspace/...`, `/tmp/...` and the target of the solver's `.claude/skills` "
                "link, belong to the solver's own episodes and do not exist on your machine. The "
                "paths that do exist are " + ", ".join(have[:-1]) + " and " + have[-1] + "."]
    return "\n".join(out)


# The solver section. A leg JSON without `solver_harness` is the direct harness (bash + skill tool): its text is
# unchanged, so every existing prompt re-renders byte-identically. A Claude Code leg (solver_harness.kind
# "claude_code") says that the solver is a Claude Code agent and names the tools its sessions actually had (recorded from the sessions' init events, not assumed).
_DIRECT_OBSERVE = (
    "The solver has bash and, when it carries a skill set, a skill tool that loads an "
    "installed skill; a solver with no skills has bash only. Files placed beside a skill's "
    "SKILL.md are installed with it, read-only, where the solver's bash can read and run them; "
    "when the solver loads a skill, the tool's reply begins with the line `Base directory for "
    "this skill: <absolute path>`, the directory those files are in.")
_WORKDIR_OBSERVE = (
    " It works in a directory holding "
    "the input workbook and is told to save its modified file to `./output/<input file name>`.")
_BUDGET_OBSERVE = (
    " It is never shown its turn budget while it works, so it cannot follow a rule phrased in "
    "terms of turns, steps, time or budget. Do not write one.")
_COMMON_OBSERVE = _WORKDIR_OBSERVE + _BUDGET_OBSERVE


def solver_harness(leg_path) -> dict:
    """The newest batch's solver harness as its leg JSON records it ({} = the direct harness)."""
    try:
        h = json.loads(Path(leg_path).read_text()).get("solver_harness")
    except (OSError, ValueError):
        return {}
    return h if isinstance(h, dict) else {}


def solver_observe_text(leg_path, *, budget_rule: bool = True) -> str:
    """budget_rule False (the rollouts layout, where the evolver decides whether a skill is turn-budget aware): no
    sentence about the turn budget and no rule against budget-aware
    skills; the facts section still states the cap and every capped episode."""
    h = solver_harness(leg_path)
    if h.get("kind") != "claude_code":
        return _DIRECT_OBSERVE + (_COMMON_OBSERVE if budget_rule else _WORKDIR_OBSERVE)
    tools = [t for t in (h.get("tools") or []) if isinstance(t, str)]
    tool_text = (f"; the tools its sessions had were {', '.join(f'`{t}`' for t in tools)}" if tools else "")
    text = (f"The solver is a Claude Code agent: `claude -p` (Claude Code {h.get('version') or '?'}) running "
            f"`{h.get('model') or '?'}`, with Claude Code's own system prompt and tools and this task's instructions "
            f"appended to that system prompt{tool_text}. Web access and subagents are disabled.")
    # how a skill set reaches a Claude Code solver is stated only when the leg records it (not built yet)
    sd = h.get("skills_delivery")
    if isinstance(sd, str) and sd.strip():
        text += f" {sd.strip()}"
    text += _WORKDIR_OBSERVE
    if budget_rule:
        # The cap is on the `claude -p` command line, not in the conversation, and the solver's bash can list that
        # process (a solver can print it with ps), so the direct harness's "never shown" would be false here.
        cap = f" (`--max-turns {h['max_turns']}`)" if h.get("max_turns") else ""
        text += (f" Its turn cap is set on the `claude -p` command line{cap}, not in the conversation; a solver that "
                 f"lists its own processes can see that command line, but nothing tells it how many turns it has "
                 f"used. It cannot follow a rule phrased in terms of turns, steps, time or budget. Do not write one.")
    return text


def render_parts(run_dir, epoch: int, *, evidence_dirs, staging_dir, prior_library, facts,
                 template_paragraph_path=None, machine_facts=None, train_references: bool = True,
                 train_inputs: bool = False, leg_json=None, changed_cells: bool = False,
                 require_full_grader_output: bool = True, outcomes=None,
                 evidence_layout: str = "digests", regraded=None) -> dict:
    """Render ITER{N}_PROMPT.md's content. Returns {"text", "sha256", "bytes", ...provenance}.

    run_dir         the run directory (its absolute form is what the evolver's VM mirrors)
    epoch           N >= 1
    evidence_dirs   the digest dirs granted this epoch, oldest first; the last is the newest batch
    staging_dir     must be EpochPaths(run_dir, N).staging
    prior_library   skills_v{N-1} for N >= 2; None for N == 1
    facts           prompt_author.derive_facts(...) for the newest dir (facts_for_epoch builds it)
    template_paragraph_path   setup/EVOLVER_TEMPLATE.md (the AO's one paragraph, used verbatim) for a run
                    whose setup wrote one; None (the default): the benchmark adapter writes no paragraph
                    and the verdict section is the loop's own text
    machine_facts   loop._evolver_env_facts(ctx) (+ optional grader_dir, references_dir); {} or
                    None for an evolver not on the solver's image (no machine section)
    train_references  name the TRAINING reference workbooks dir (default on)
    train_inputs      name the TRAINING input workbooks dir (default off)
    leg_json        the leg the newest dir digests; default = default_leg_json(run_dir, N)
    changed_cells   the changed-cells list is granted (default off): an evidence dir carrying it
                    is refused unless this is on; the prompt describes it only where present
    require_full_grader_output  refuse an evidence dir whose DIGEST_INFO.json records any clipped
                    grader output (default on); off, the clipped count is stated instead
    outcomes        the leg's four outcomes AFTER the loop's re-grade ({passed, graded_wrong,
                    not_eligible, grader_failed, status, reportable, source}; loop.
                    _leg_outcomes_after_regrade). None = outcomes.leg_outcome_counts(leg_json) as
                    recorded. Either way _check_outcomes refuses any grader failure left in it.
    evidence_layout  RUNTIME evolver_evidence_layout: "digests" (absent; the prompt as before, byte for
                    byte) or "rollouts" (the skill-set table, each epoch's skill snapshot,
                    the newest batch's per-rollout scores from its INDEX.json, the trace layout; every
                    granted dir must hold rollout evidence whose INDEX re-derives, see _rollout_evidence)
    regraded        rollouts layout: the loop's re-grade rows ({task, sample, outcome}) of the
                    newest batch, so a re-graded sample shows its re-grade's outcome
    Every evidence dir must carry a DIGEST_INFO.json that its bytes confirm (digest_info.read).
    """
    from src.orchestrator.daytona_evolver import MAX_PROMPT_BYTES
    from src.orchestrator.loop import EpochPaths
    if not isinstance(epoch, int) or epoch < 1:
        raise RenderError(f"epoch must be an int >= 1, got {epoch!r}")
    if evidence_layout not in LAYOUTS:
        raise RenderError(f"evidence_layout {evidence_layout!r} is not one of {LAYOUTS}")
    rollouts = evidence_layout == "rollouts"
    if regraded and not rollouts:
        raise RenderError("regraded rows are read in the rollouts layout only")
    rd = Path(run_dir)
    mf = dict(machine_facts or {})
    # The benchmark adapter writes no EVOLVER_TEMPLATE.md; a stateful evolver reads the grader itself. None = no
    # adapter paragraph and the verdict section holds the loop's own sentences only. A path given must still be a
    # valid paragraph (runs whose setup wrote one re-render byte for byte).
    paragraph = read_template_paragraph(template_paragraph_path) if template_paragraph_path else None
    rows = _evidence_rows(rd, epoch, evidence_dirs, facts, prior_library)
    newest = Path(rows[-1].path)
    staging = Path(staging_dir).absolute()
    want_staging = EpochPaths(rd, epoch).staging.absolute()
    if staging != want_staging:
        raise RenderError(f"staging_dir must be {want_staging} (the loop collects from there), "
                          f"got {staging}")
    leg = Path(leg_json) if leg_json else default_leg_json(rd, epoch)
    if outcomes is None:
        counts, source = outcome_counts(rd, leg)
    else:
        if not isinstance(outcomes, dict):
            raise RenderError(f"outcomes must be a dict of the four counts, got {type(outcomes).__name__}")
        counts = {k: outcomes.get(k) for k in OUTCOME_KEYS}
        for k in ("status", "reportable"):
            if k in outcomes:
                counts[k] = outcomes[k]
        source = str(outcomes.get("source") or "the loop (outcomes.py + re-grade)")
    computed = _check_outcomes(counts, facts)
    grader, same_program = _grader_dir(rd, mf)
    tooling_absent = _grader_tooling_absent(mf)
    refs = _references_dir(rd, mf, newest) if train_references else None
    inputs = _inputs(rd, mf) if train_inputs else None
    earlier_libs = any(r.library for r in rows[:-1])
    table_text, table_rec = "", None
    if rollouts:
        rg = _regraded_map(regraded)
        pairs, _final = _rollout_evidence(rows, facts=facts, counts=counts, computed=computed, leg=leg,
                                          regraded=rg)
        prior_text, prior_rec = _skill_set_table(epoch, prior_library, newest, staging, earlier_libs)
    else:
        prior_text, prior_rec = _prior_skill_set(epoch, prior_library, newest, staging, earlier_libs)

    if epoch == 1:
        opening = ("This is the first iteration. No skill set exists yet. Everything you are shown "
                   "comes from a solver that carried no skills, and you are writing the first set, "
                   "`skills_v1`.")
    else:
        opening = (f"This is iteration {epoch}. The newest batch comes from a solver carrying "
                   f"`skills_v{epoch - 1}`, and you are revising that set into `skills_v{epoch}`.")
    verdict = [paragraph, ""] if paragraph is not None else []
    verdict.append(grader_sentence(grader, same_program, tooling_absent))
    if refs:
        verdict[-1] += (f" The reference workbook of every training task that has an evidence file "
                        f"is in `{refs['dir']}`: {refs['n']} reference workbooks, {refs['mb']} MB "
                        f"in total, laid out as `{refs['dir']}/"
                        + ("<task file stem>/" if rollouts else "<evidence file name without .md>/")
                        + "<reference workbook file>`.")
    if rollouts:
        evidence_text, ev_rec = _evidence_section_rollouts(
            pairs, _trace_stats(pairs), changed_cells=changed_cells,
            require_full_grader_output=require_full_grader_output)
        table_text, table_rec = _rollout_table_section(*pairs[-1], rg)
    else:
        evidence_text, ev_rec = _evidence_section(
            rows, newest, changed_cells=changed_cells,
            require_full_grader_output=require_full_grader_output)
    parts = [
        f"# Iteration {epoch}: write skill set v{epoch}",
        opening,
        _facts_section(facts, counts, computed, newest),
        "## How a verdict is produced\n\n" + "\n".join(verdict),
        "## What the solver can and cannot observe\n\n" + solver_observe_text(leg, budget_rule=not rollouts),
        evidence_text,
    ]
    if table_text:
        parts.append(table_text)
    if mf:
        parts.append(_machine_section(mf, refs, inputs, rollouts=rollouts))
    if prior_text:
        parts.append(prior_text)
    parts += [
        "## The cost of a skill\n\n"
        "Every skill costs context on every episode that loads it. Until a skill is invoked, only "
        "its name and description are visible, so a description must say when the skill applies. "
        "An empty or small skill set is a valid result.",
        "## Constraints on the skills you write\n\n"
        "- Nothing specific to this benchmark, its grader, or any task, file, sheet or other "
        "identifier.\n"
        "- No mention of samples, episodes, scores, graders or digests inside a skill. Name your "
        "evidence in your final message instead.\n"
        # rollouts layout: the evolver decides whether a skill is turn-budget aware
        + ("" if rollouts else "- No rule about turns, steps, time or budget.\n")
        + "- No mention of your own machine or of any directory named in this prompt; the solver "
        "will not have them.\n"
        "- No rule written only to fill space.",
        "## Format and where to write\n\n"
        f"Write your skill set into exactly this directory: `{staging}`"
        + (f". It holds `skills_v{epoch - 1}` when you start; what it holds when you finish is "
           f"`skills_v{epoch}`." if epoch >= 2 else ".")
        + f" Write each skill to `{staging}/<skill-name>/SKILL.md`, starting with YAML frontmatter "
          "holding `name` (the same as the directory name) and `description`. A skill written "
          "anywhere else, including any `.claude/skills/` directory, is lost.",
        "## What to report in your final message\n\n"
        "For each skill: what it does, the evidence behind it, and when it applies. "
        + (f"For each skill of `skills_v{epoch - 1}`, a verdict: kept, rewritten, merged, split or "
           f"deleted, and why. " if epoch >= 2 else "")
        + "Then name the rule you judge most at risk of describing these transcripts rather than "
          "being durable, and what you did about it.",
    ]
    text = "\n\n".join(parts) + "\n"
    data = text.encode("utf-8")
    if len(data) > MAX_PROMPT_BYTES:
        raise RenderError(f"the rendered prompt is {len(data)} B; the evolver VM's argv cap is "
                          f"{MAX_PROMPT_BYTES} B")
    return {
        "text": text, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
        "epoch": epoch, "run_dir": str(rd.absolute()),
        "template_paragraph": str(Path(template_paragraph_path).absolute()) if template_paragraph_path else None,
        "template_paragraph_sha256": (hashlib.sha256(paragraph.encode("utf-8")).hexdigest()
                                      if paragraph is not None else None),
        "leg_json": str(leg), "outcome_counts": counts, "outcome_source": source,
        "outcomes_computed": computed,
        "evidence": [{"epoch": r.epoch, "path": r.path, "library": r.library,
                      "task_files": r.task_files, "current": r.current} for r in rows],
        "grader_dir": str(grader), "grader_tooling_absent": tooling_absent,
        "references": refs, "inputs": inputs,
        "staging": str(staging), "prior_library": prior_rec, **ev_rec,
        **({"evidence_layout": evidence_layout, "rollout_table": table_rec} if rollouts else {}),
        "flags": {"train_references": bool(train_references), "train_inputs": bool(train_inputs),
                  "changed_cells": bool(changed_cells),
                  "require_full_grader_output": bool(require_full_grader_output)},
    }


def render(run_dir, epoch: int, *, evidence_dirs, staging_dir, prior_library, facts,
           template_paragraph_path=None, machine_facts=None, train_references: bool = True,
           train_inputs: bool = False, leg_json=None, changed_cells: bool = False,
           require_full_grader_output: bool = True, outcomes=None, evidence_layout: str = "digests",
           regraded=None) -> str:
    """ITER{N}_PROMPT.md's content. See render_parts for the arguments and the checks."""
    return render_parts(run_dir, epoch, evidence_dirs=evidence_dirs, staging_dir=staging_dir,
                        prior_library=prior_library, facts=facts,
                        template_paragraph_path=template_paragraph_path,
                        machine_facts=machine_facts, train_references=train_references,
                        train_inputs=train_inputs, leg_json=leg_json,
                        changed_cells=changed_cells,
                        require_full_grader_output=require_full_grader_output,
                        outcomes=outcomes, evidence_layout=evidence_layout,
                        regraded=regraded)["text"]


# ─────────────────────────────────────────────────────────────── preview CLI
def main(argv=None) -> int:
    """Preview: render epoch N of a run into --out (never the run's ITER{N}_PROMPT.md, never an
    existing file) and print the provenance record."""
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-leg", default="")
    ap.add_argument("--train-inputs", action="store_true")
    ap.add_argument("--no-train-references", action="store_true")
    ap.add_argument("--changed-cells", action="store_true",
                    help="the changed-cells list is granted (default off)")
    a = ap.parse_args(argv)
    from src.orchestrator.loop import EpochPaths
    rd = Path(a.run_dir)
    out = Path(a.out).absolute()
    if out == EpochPaths(rd, a.epoch).prompt.absolute():
        print(f"REFUSING: {out} is the run's own prompt; the loop writes it, a preview does not")
        return 2
    if out.exists():
        print(f"REFUSING: {out} exists; a preview never overwrites")
        return 2
    rt = json.loads((rd / "RUNTIME.json").read_text()) if (rd / "RUNTIME.json").is_file() else {}
    base = Path(a.base_leg) if a.base_leg else default_leg_json(rd, 1)
    all_epochs = rt.get("evolver_evidence") == "all_epochs"
    facts = facts_for_epoch(rd, a.epoch, base_leg=base, all_epochs=all_epochs)
    spec = json.loads((rd / "SPEC.json").read_text()).get("spec") or {}
    mf = (machine_facts_for(rd, split_path=spec.get("split_path"), base_leg=base,
                            train_inputs=a.train_inputs)
          if rt.get("evolver_host") == "daytona" else {})
    ep = EpochPaths(rd, a.epoch)
    evidence = ep.evidence_dirs() if all_epochs else [ep.digests]
    layout = _rollout_module().layout_of(rt)          # a typo refuses, as the loop's _evidence_layout
    parts = render_parts(rd, a.epoch, evidence_dirs=evidence, staging_dir=ep.staging,
                         prior_library=EpochPaths(rd, a.epoch - 1).promoted if a.epoch > 1 else None,
                         facts=facts, template_paragraph_path=rd / "setup" / TEMPLATE_NAME,
                         machine_facts=mf, train_references=not a.no_train_references,
                         train_inputs=a.train_inputs, changed_cells=a.changed_cells,
                         evidence_layout=layout)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8") as fh:
        fh.write(parts["text"])
    print(json.dumps({k: v for k, v in parts.items() if k != "text"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
