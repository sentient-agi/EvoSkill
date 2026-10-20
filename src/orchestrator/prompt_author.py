#!/usr/bin/env python3
"""AO-AUTHORED EVOLVER PROMPT — the orchestrator writes the evolver's user prompt itself.

WHY THIS EXISTS.

A prompt written by hand each epoch has two defects, and the second is worse than the first:

  1. SELF-REVIEW. When the person who writes the prompt also checks it, that check examines its
     own work: it looks like verification while examining what it produced.

  2. PROMPT ASSEMBLY IS NOT AN ABLATABLE VARIABLE. The prompt is the largest lever in the system,
     but you cannot ablate a term a human re-tunes by hand every epoch. Making assembly a POLICY
     turns it into a config, and two policies are an experiment.

THE DESIGN, AND WHERE THE TRUST BOUNDARY SITS.

    derive_facts()   DETERMINISTIC. Reads the digests, the base/prior leg result json, the oracle,
                     the split and the corpus, and computes EVERY number the prompt may state.
                     No model involved. This is the ground truth the prompt is allowed to assert.

    PROMPT_POLICY    Supplied AT INITIALISATION (RUNTIME.json "prompt_policy"), not per epoch. It
                     says HOW to turn a benchmark's format plus solver trace logs into an evolver
                     prompt: which sections, which constraints, what must never appear. It is
                     TASK-UNIVERSAL — it says "state the scoring-basis distribution", never
                     "say 23/25 are DB-state" (standing rule: no benchmark-specific content in
                     optimizer-facing prompts).

    author_prompt()  Dispatches a model with the POLICY + the FACTS BLOCK + read access to the
                     train-only digests under the same containment as the evolver. It writes prose;
                     it does not invent numbers.

    THEN THE GATES BITE. The prompt-numbers check re-derives the prompt's batch rate from the leg
    json, and check_prompt_no_leakage asserts zero held-out ids and no val figure. They guard a
    model author exactly as they would a human one. If the author fabricates a number, the loop
    HALTs and nothing is dispatched.

WHAT THIS DELIBERATELY DOES NOT DO. It does not touch the evolver itself, the solver, the gate, or
the skills. It replaces one input — who writes the prompt — and leaves every verification in place.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.orchestrator import conversation
from src.orchestrator.quota_scope import subscription_env

REPO = Path(__file__).resolve().parents[2]

# The digest header the builder writes, e.g.
#   **Result: 3/3 samples passed.**  scores=[1.0, 1.0, 1.0]  kind=['pass','pass','pass'] turns=[...]
# PARSE THE HEADER, NOT THE BODY. A regex over a whole digest body matches transaction amounts and
# tool output, not the pass rate. The header is the artifact's own authoritative field. Look at the
# format before writing the matcher.
_RESULT = re.compile(r"\*\*Result:\s*(\d+)/(\d+)\s+samples passed\.\*\*")
_KIND = re.compile(r"kind=\[([^\]]*)\]")
# SSB v2 records step-budget exhaustion in its OWN header field. A reader that knows only tau3's
# `kind=[...max_steps...]` and `termination=[...]` misses every match on an SSB v2 surface and
# reports "tasks losing >=1 sample to the step budget   0" and "termination reasons {}" even when
# many samples were TRUNCATED at the turn ceiling, and the author would transcribe those zeros into
# the evolver prompt faithfully. An empty parse must never render as a zero MEASUREMENT.
_EXH = re.compile(r"budget_exhausted=\[([^\]]*)\]")
_TERM = re.compile(r"termination=\[([^\]]*)\]")
_TURNS = re.compile(r"turns=\[([^\]]*)\]")
_BASIS = re.compile(r"\*\*reward_basis:\*\*\s*\[([^\]]*)\]")
_GOLD = re.compile(r"\*\*Golden actions \((\d+)\)")
_REQ = re.compile(r"\*\*required_documents \((\d+)\)\*\*")
# A real retrieval miss is written MISSED [...]; a clean sample says "missed none". The distinction
# is CASE: a case-insensitive filter would score every genuine miss as clean.
_MISSED = re.compile(r"\bMISSED\b")
_SAMPLE_LINE = re.compile(r"\s*-\s+\*\*s\d")


def _split_list(s: str) -> list:
    return [x.strip().strip("'\"") for x in s.split(",") if x.strip()]


@dataclass
class Facts:
    """Every number the prompt is permitted to state, each derived from an artifact on disk."""
    n_tasks: int = 0
    n_samples: int = 0
    n_episodes: int = 0
    passed: int = 0
    graded_fail: int = 0
    never_graded: int = 0
    reliable: int = 0            # passed every sample
    unstable: int = 0            # passed some but not all
    never_passes: int = 0
    termination: dict = field(default_factory=dict)
    basis_counts: dict = field(default_factory=dict)
    golden_min: int = 0
    golden_max: int = 0
    corpus_docs: int = 0
    variant: str = ""
    # DISTINCT FROM never_graded, and the distinction is not pedantic. In tau3 a max_steps
    # episode produces NO answer, so exhaustion and never-graded are the same episodes. In
    # SSB v2 the grader scores whatever workbook is in ./output/ at cutoff, so a truncated
    # episode IS graded — it lands in passed or graded-WRONG. Folding it into never_graded
    # would misreport the outcome triple; dropping it loses the single largest addressable
    # failure mode. It is its own line.
    exhausted_episodes: int = 0
    tasks_with_exhausted: int = 0
    tasks_all_exhausted: int = 0
    tasks_with_ungraded: int = 0
    tasks_all_ungraded: int = 0
    retrieval_samples_missing: int = 0
    retrieval_tasks_all_missing: int = 0
    turns_min: int = 0
    turns_max: int = 0
    # PROOF-OF-COVERAGE counters. A facts block computed from zero files must never look like a
    # facts block of all-zeros.
    digests_seen: int = 0
    result_headers_parsed: int = 0
    retrieval_lines_parsed: int = 0
    # ALL-EPOCHS MODE ONLY (RUNTIME evolver_evidence): one row per evidence dir the
    # evolver is granted, oldest first. Empty in every other run, and then as_block() is
    # byte-identical to what it was before this field existed.
    evidence_index: list = field(default_factory=list)
    # DAYTONA EVOLVER HOST ONLY (RUNTIME evolver_host): the facts of the evolver's OWN
    # environment -- the solver's image and versions, the network, the input-workbook dir. Empty in
    # every other run, and then as_block() is byte-identical to before. Appended AFTER the index,
    # so the selftest's "batch block, then EVIDENCE INDEX" prefix still holds.
    evolver_env: dict = field(default_factory=dict)

    def as_block(self) -> str:
        """The verbatim FACTS the author may assert. Anything not here is not established."""
        block = self._batch_block()
        if self.evidence_index:
            block = block + "\n\n" + _index_block(self.evidence_index)
        if self.evolver_env:
            block = block + "\n\n" + _env_block(self.evolver_env)
        return block

    def _batch_block(self) -> str:
        gr = self.passed + self.graded_fail
        pct = (100.0 * self.passed / self.n_episodes) if self.n_episodes else 0.0
        return "\n".join([
            "## VERIFIED FACTS — every number below was computed from the artifacts on disk.",
            "## You may state these. You may NOT state any other quantity, and you may not round",
            "## them into a different claim.",
            "",
            f"tasks                        {self.n_tasks}",
            f"samples per task             {self.n_samples}",
            f"episodes                     {self.n_episodes}",
            f"episodes passed              {self.passed}   ({pct:.1f}%)",
            f"episodes graded and WRONG    {self.graded_fail}",
            f"episodes NEVER GRADED        {self.never_graded}   (ran out of step budget; no answer to grade)",
            f"episodes that reached a verdict  {gr}",
            "",
            f"tasks passing EVERY sample   {self.reliable}",
            f"tasks passing SOME not all   {self.unstable}   (same task, same agent, different outcome)",
            f"tasks passing NO sample      {self.never_passes}",
            "",
            # THE ANNOTATION IS BENCHMARK-SPECIFIC, so it must not be emitted unconditionally.
            # "still graded on whatever was saved, so they also appear above" is true of SSB v2,
            # where the grader scores whatever workbook sits in ./output/ at cutoff. It is FALSE
            # of tau3, where a max_steps episode produces no answer at all -- which is why this
            # class's own `exhausted_episodes` comment says exhaustion and never-graded are the
            # SAME episodes there. On a tau3 batch the line would contradict the "NEVER GRADED
            # ... no answer to grade" line two rows above it, inside one block the author is
            # told it may not question.
            #
            # Authors given this identical block may or may not carry such a claim, and every
            # prompt check passes either way: none of them asks whether a TRANSCRIBED CLAIM IS
            # TRUE -- they ask whether a number re-derives, whether the register is analytic, and
            # whether an id leaked. A false premise faithfully copied is invisible to all three.
            #
            # DERIVED FROM THE DATA, NOT FROM A BENCHMARK FLAG. Facts has no `benchmark` field,
            # and adding one to render a sentence would put benchmark-specific content in a
            # task-universal component. The counts already answer the question: if every
            # exhausted episode is also never-graded, exhaustion PRODUCED no verdict here.
            f"episodes CUT OFF at the turn ceiling  {self.exhausted_episodes}"
            + ("   (these produced no answer, so they ARE the never-graded episodes above)"
               if self.exhausted_episodes and self.exhausted_episodes == self.never_graded
               else "   (still graded on whatever was saved, so they also appear above)"
               if self.exhausted_episodes else ""),
            f"tasks losing >=1 sample to the step budget   {self.tasks_with_exhausted}",
            f"tasks losing EVERY sample that way           {self.tasks_all_exhausted}",
            "",
            f"termination reasons          {self.termination}",
            f"scoring basis, tasks per basis  {self.basis_counts}",
            f"required actions per task    {self.golden_min}..{self.golden_max}",
            f"agent turns observed         {self.turns_min}..{self.turns_max}",
            f"retrieval corpus documents   {self.corpus_docs}",
            f"retrieval variant            {self.variant}",
            f"samples that missed >=1 required document   {self.retrieval_samples_missing}",
            f"tasks missing >=1 required doc in EVERY sample  {self.retrieval_tasks_all_missing}",
            "",
            f"[coverage proof: {self.digests_seen} digest files read, "
            f"{self.result_headers_parsed} result headers parsed, "
            f"{self.retrieval_lines_parsed} retrieval lines parsed]",
        ])


@dataclass
class EvidenceRow:
    """One evidence directory in all-epochs mode: where it is, what produced it, what it holds."""
    epoch: int
    path: str                    # absolute and resolved: exactly what the evolver is granted
    library: str                 # "" for the base surface (digests_e1_train)
    library_sha12: str
    task_files: int
    passed: int
    episodes: int
    rate_checked: bool           # re-derived from this directory's own leg result file
    current: bool                # the batch every other figure in FACTS describes


def _index_block(rows) -> str:
    """The EVIDENCE INDEX, appended to FACTS in all-epochs mode only.

    Phrased for the two gates that read the prompt the AO writes from it. An earlier directory's
    count is `<P> of <E> episodes passed`, the one form loop._v_prompt_numbers and the
    launcher's gate both accept when it equals a granted earlier directory's leg. The file count
    is `<N> task files`, never "<N> train tasks" or "tasks in this batch", which the launcher
    reads as a claim about this batch. Coverage is not stated at all: the "evidence surface is
    complete" step already guarantees every directory is whole, and a coverage fraction near
    "scor..." reads as a pass claim to the launcher.
    """
    out = ["## EVIDENCE INDEX — every directory the evolver is granted, one per epoch so far, "
           "oldest first.",
           "## Name each one by the absolute path below, with the library that produced it. "
           "The LAST row is",
           "## this batch: every figure above describes it, and nothing else.",
           ""]
    for r in rows:
        out.append(f"e{r.epoch}  {r.path}" + ("   (THIS BATCH)" if r.current else ""))
        if r.library:
            out.append(f"      produced by   the solver carrying {r.library} "
                       f"sha256:{r.library_sha12}; that library's exact files are in "
                       f"{r.path}/_library/")
        else:
            out.append("      produced by   the solver carrying NO skills (the base leg)")
        out.append(f"      task files    {r.task_files}")
        if r.rate_checked:
            out.append(f"      episodes      {r.passed} of {r.episodes} episodes passed")
        else:
            out.append("      episodes      NOT STATED: no leg result file is declared for this "
                       "directory, so no gate could re-derive a count. State none.")
    unchecked = sum(1 for r in rows if not r.rate_checked)
    out += ["", f"[index proof: {len(rows)} evidence director{'y' if len(rows) == 1 else 'ies'}, "
                f"{sum(r.task_files for r in rows)} task files read; every stated count "
                f"re-derived from its own leg's result file"
                + (f"; {unchecked} count(s) withheld because no leg is declared" if unchecked
                   else "") + "]"]
    return "\n".join(out)


def _env_block(e: dict) -> str:
    """EVOLVER ENVIRONMENT, appended to FACTS in daytona-host mode only. Phrased for the prompt
    gates: counts are "<N> input workbooks" (never "<N> train tasks", never a fraction), no
    held-out keyword anywhere, the layout is generic (placeholders, never a task's subdir), and no
    sentence says what to DO with any of it (the not-analytical check)."""
    return "\n".join([
        "## EVOLVER ENVIRONMENT — where the evolver itself runs in this run (evolver host: daytona).",
        "## State these as facts about the evolver's own environment. Do not suggest what to do "
        "with them.",
        "",
        f"image         the solver's own image recipe ({e['image_recipe']} sha256:{e['image_sha12']}) "
        f"plus the Claude Code CLI; the grader's own tooling is not in it",
        f"python        {e['python']} with openpyxl {e['openpyxl']}, pandas {e['pandas']}, numpy "
        f"{e['numpy']}: the versions the solver's episodes use",
        f"libreoffice   {e['libreoffice']} (soffice on PATH): the build the solver's episodes use",
        "network       the model API only; no other host is reachable",
        f"inputs dir    {e['inputs_dir']}",
        f"              the input workbook (never a reference workbook) of every task that has an "
        f"evidence file: {e['n_inputs']} input workbooks, {e['inputs_mb']} MB in total",
        f"              layout: {e['inputs_dir']}/<evidence file name without .md>/<the file named "
        f"on that evidence file's Input file line>",
        "other runs    nothing of any other run is on the evolver's filesystem",
        "",
        f"[environment proof: image recipe re-hashed from disk; inputs dir verified against the "
        f"dataset and the split ({e['n_inputs']} files, each byte-equal to its task's input, none "
        f"equal to any reference workbook)]",
    ])


_SIDECARS = frozenset({"REGRESSIONS", "README", "INDEX"})


def _surface_counts(d: Path) -> tuple:
    """(task files, result headers parsed, episodes passed, episodes) of ONE digest dir, from
    each top-level task file's Result header — the same header derive_facts parses."""
    files = sorted(p for p in Path(d).glob("*.md") if p.stem not in _SIDECARS)
    parsed = passed = eps = 0
    for p in files:
        m = _RESULT.search("\n".join(p.read_text(errors="replace").splitlines()[:40]))
        if m:
            parsed += 1
            passed += int(m.group(1))
            eps += int(m.group(2))
    return len(files), parsed, passed, eps


def _leg_totals(leg_json: Path) -> tuple:
    """(passed, episodes) a leg's result json records — the cross-check derive_facts applies."""
    d = json.loads(Path(leg_json).read_text())
    pt = d.get("per_task") or {}
    lp = sum(sum(x for x in v if x is not None) for v in pt.values())
    le = sum(len([x for x in v if x is not None]) for v in pt.values())
    return lp, le


def _evidence_index(evidence, current_dir: Path, f: Facts) -> list:
    """Build and VERIFY the EVIDENCE INDEX. Raises rather than describing a directory wrongly.

    `evidence` is [(j, digest dir, leg json or None, producing library or None)] for j = 1..K, as
    loop._run_author_prompt builds it from EpochPaths: epoch j's surface digests the base leg
    (j == 1) or skills_v{j-1}'s train leg (j >= 2), and was produced by no library (j == 1) or by
    skills_v{j-1}. Every row is checked against the artifacts before the AO is shown it:
      * the directory is the canonical digests_e{j}_train, a real dir, and the last one is the
        batch the rest of FACTS describes;
      * it carries its producing library honestly (skill_library.problems: _library/ byte-identical
        to skills_v{j-1}, every marker anchored to it; none of that on the base surface);
      * every task file has a Result header, and the pass count they sum to equals what the
        directory's own leg json records (the 0809c C-4.1 check, per directory).
    """
    from src.orchestrator import skill_library as _sl
    ev = sorted(((int(j), Path(d), leg, lib) for j, d, leg, lib in evidence), key=lambda t: t[0])
    if [t[0] for t in ev] != list(range(1, len(ev) + 1)):
        raise ValueError(f"EVIDENCE INDEX: the evidence list must be epochs 1..K with none "
                         f"missing, got {[t[0] for t in ev]}")
    cur = Path(current_dir).resolve()
    rows = []
    for j, d, leg, lib in ev:
        if d.name != f"digests_e{j}_train" or d.is_symlink() or not d.is_dir():
            raise ValueError(f"EVIDENCE INDEX: epoch {j}'s evidence must be a real directory named "
                             f"digests_e{j}_train, got {d}")
        is_cur = d.resolve() == cur
        if is_cur != (j == len(ev)):
            raise ValueError(f"EVIDENCE INDEX: the newest directory must be the batch these facts "
                             f"describe ({cur}); {d} is epoch {j} of {len(ev)}")
        if j >= 2 and lib is None:
            raise ValueError(f"EVIDENCE INDEX: no producing library given for {d.name}")
        label = _sl.label_for_epoch(j) if j >= 2 else ""
        probs = _sl.problems(d, lib, label)
        if probs:
            raise ValueError(f"EVIDENCE INDEX: {d.name} does not carry its producing library "
                             f"honestly, so the index cannot say what produced it: "
                             + "; ".join(probs))
        n_files, parsed, p, e = _surface_counts(d)
        if n_files == 0 or parsed != n_files:
            raise ValueError(f"EVIDENCE INDEX: {d.name} has {n_files} task file(s) and "
                             f"{parsed} Result header(s); every task file must carry one")
        checked = False
        if leg is not None:
            if not Path(leg).is_file():
                raise ValueError(f"EVIDENCE INDEX: {d.name}'s leg {leg} does not exist")
            lp, le = _leg_totals(Path(leg))
            if (lp, le) != (p, e):
                raise ValueError(
                    f"EVIDENCE MISMATCH: {d.name} re-derives {p}/{e} but {Path(leg).name} says "
                    f"{lp:.0f}/{le}. That directory is not its leg's surface; refusing to index it.")
            checked = True
        if is_cur:
            if (p, e) != (f.passed, f.n_episodes):
                raise ValueError(f"EVIDENCE INDEX: {d.name} re-derives {p}/{e} but the batch "
                                 f"facts say {f.passed}/{f.n_episodes}")
            checked = True           # the batch block above states this figure regardless
        rows.append(EvidenceRow(epoch=j, path=str(d.resolve()), library=label,
                                library_sha12=_sl.library_sha256(lib)[:12] if lib else "",
                                task_files=n_files, passed=p, episodes=e,
                                rate_checked=checked, current=is_cur))
    return rows


def derive_facts(digest_dir: Path, leg_json: Path, corpus_dir: Path | None = None,
                 variant: str = "", evidence: list | None = None) -> Facts:
    """Compute the facts block. Raises rather than returning zeros on an empty surface.

    `evidence` (all-epochs mode only): [(j, digest dir, leg json, producing library)] for every
    evidence dir the evolver is granted. It adds the EVIDENCE INDEX; None leaves FACTS exactly as
    it was.
    """
    f = Facts(variant=variant)
    # Sidecars excluded: a rollout-evidence dir carries INDEX.md next to the task files, and
    # the coverage proof must keep counting task digests only (the same set _task_files names).
    files = sorted(p for p in digest_dir.glob("*.md") if p.stem not in _SIDECARS)
    if not files:
        raise ValueError(f"no digests at {digest_dir} — cannot author a prompt with no evidence")
    turns_all: list = []
    golds: list = []
    for p in files:
        t = p.read_text(errors="replace")
        head = "\n".join(t.splitlines()[:40])
        f.digests_seen += 1
        m = _RESULT.search(head)
        if not m:
            continue
        f.result_headers_parsed += 1
        a, b = int(m.group(1)), int(m.group(2))
        f.passed += a
        f.n_episodes += b
        f.n_tasks += 1
        f.n_samples = max(f.n_samples, b)
        if a == b:
            f.reliable += 1
        elif a > 0:
            f.unstable += 1
        else:
            f.never_passes += 1
        mk = _KIND.search(head)
        ung = 0
        if mk:
            kinds = _split_list(mk.group(1))
            ung = sum(1 for k in kinds if "max_steps" in k or "ungraded" in k)
        f.never_graded += ung
        if ung:
            f.tasks_with_ungraded += 1
        me = _EXH.search(head)
        if me:
            flags = _split_list(me.group(1))
            hit = [x for x in flags if x.strip().strip("'\"")]
            f.exhausted_episodes += len(hit)
            if hit:
                f.tasks_with_exhausted += 1
                if len(hit) == len(flags):
                    f.tasks_all_exhausted += 1
                for x in hit:
                    k = x.strip().strip("'\"")
                    f.termination[k] = f.termination.get(k, 0) + 1
        elif ung:
            f.exhausted_episodes += ung          # tau3: max_steps IS the exhaustion signal
            f.tasks_with_exhausted += 1
            if mk and ung == len(_split_list(mk.group(1))):
                f.tasks_all_exhausted += 1
        if ung and mk and ung == len(_split_list(mk.group(1))):
            f.tasks_all_ungraded += 1
        mt = _TERM.search(head)
        if mt:
            for x in _split_list(mt.group(1)):
                f.termination[x] = f.termination.get(x, 0) + 1
        mu = _TURNS.search(head)
        if mu:
            turns_all += [int(float(x)) for x in _split_list(mu.group(1)) if x.replace(".", "").isdigit()]
        mb = _BASIS.search(t)
        if mb:
            key = "+".join(sorted(_split_list(mb.group(1))))
            f.basis_counts[key] = f.basis_counts.get(key, 0) + 1
        mg = _GOLD.search(t)
        if mg:
            golds.append(int(mg.group(1)))
        # retrieval section, if the digest has one
        sec = re.search(r"## Did the agent retrieve.*?(?=\n## )", t, re.S)
        if sec:
            lines = [l for l in sec.group(0).splitlines() if _SAMPLE_LINE.match(l)]
            f.retrieval_lines_parsed += len(lines)
            bad = [l for l in lines if _MISSED.search(l)]
            f.retrieval_samples_missing += len(bad)
            if lines and len(bad) == len(lines):
                f.retrieval_tasks_all_missing += 1
    f.graded_fail = f.n_episodes - f.passed - f.never_graded
    if golds:
        f.golden_min, f.golden_max = min(golds), max(golds)
    if turns_all:
        f.turns_min, f.turns_max = min(turns_all), max(turns_all)
    if corpus_dir and corpus_dir.is_dir():
        f.corpus_docs = len(list(corpus_dir.iterdir()))

    # CROSS-CHECK AGAINST THE LEG JSON. The digests are BUILT from the leg's cache, so the two must
    # agree; if they do not, the evidence surface is not this leg's and the 0809c C-4.1 failure mode
    # is live (the evolver would revise against results its library never produced).
    if leg_json and leg_json.is_file():
        d = json.loads(leg_json.read_text())
        pt = d.get("per_task") or {}
        lp = sum(sum(x for x in v if x is not None) for v in pt.values())
        le = sum(len([x for x in v if x is not None]) for v in pt.values())
        if (lp, le) != (f.passed, f.n_episodes):
            raise ValueError(
                f"EVIDENCE MISMATCH: digests re-derive {f.passed}/{f.n_episodes} but "
                f"{leg_json.name} says {lp:.0f}/{le}. The digest surface is not this leg's. "
                f"Refusing to author a prompt against evidence the library never produced.")
    if f.result_headers_parsed == 0:
        raise ValueError(f"read {f.digests_seen} digest file(s) and parsed 0 result headers — the "
                         f"extractor did not understand the format; refusing to emit zeros as facts")
    if evidence is not None:
        f.evidence_index = _evidence_index(evidence, digest_dir, f)
    return f



DEFAULT_POLICY = Path(__file__).resolve().parent / "PROMPT_POLICY.md"
# RUNTIME evolver_evidence=all_epochs. A SEPARATE FILE, never an edit of the one above: earlier
# runs' RUNTIME.json name PROMPT_POLICY.md by path, so editing it in place would silently change the
# AO's instructions for any re-author of a completed epoch. author_prompt records its sha256 too.
ALL_EPOCHS_POLICY = Path(__file__).resolve().parent / "PROMPT_POLICY_ALL_EPOCHS.md"


def _prior_note(prior: Path | None, all_epochs: bool = False, host: str = "local") -> str:
    """THE EVOLVER CANNOT OPEN THE PRIOR SKILL SET, so the prompt must CONTAIN it.

    A prompt that says "There is a prior skill set of 3 skills at outputs/<run>/skills_v1 … Read
    them before you decide." is unfollowable: the evolver's grants are exactly {digests, staging,
    oracle}, and the peer-skill hook denies every `skills_v*` read, including this run's OWN
    previous version. Granting the path would mean loosening the one guard that stops an evolver
    copying a peer arm's answers to the same train tasks, so the predecessor travels through the
    PROMPT instead.

    That is also the better artifact: the countersigned prompt hash then covers exactly what the
    evolver was shown about its predecessor, instead of a directory that could change underneath
    it. The ORCHESTRATOR reads the bodies here (this function runs outside the AO's sandbox, which
    does not grant the run dir) and the author is told to reproduce them; nothing
    downstream has to be relaxed.

    `all_epochs` (RUNTIME evolver_evidence): the same verbatim carry, plus the one thing that
    changes. Every evidence dir from e2 on stores the exact library that produced it under
    `_library/`, so the prompt names those instead of reproducing v1..v{K-2}. The run's own
    skills_v* stay unopenable. False keeps the text byte-identical to before.
    """
    if not (prior and prior.is_dir()):
        return "NO skill set exists yet. This is the first iteration; say so in the prompt."
    files = sorted(prior.glob("*/SKILL.md"))
    if not files:
        return "NO skill set exists yet. This is the first iteration; say so in the prompt."
    if all_epochs:
        out = [
            f"A PRIOR SKILL SET OF {len(files)} SKILL(S) EXISTS, and the evolver is revising it — so "
            "the prompt must say what changed since it was written, and must judge each prior rule "
            "against this batch's evidence of that rule IN USE.",
            "",
            "*** REPRODUCE THE PRIOR SKILL SET VERBATIM INSIDE THE PROMPT. *** The evolver CANNOT "
            "OPEN it where it lives: it is granted the evidence directories, its staging dir and "
            "the oracle, and " + ("no skills_v* directory of this or any other run is on its "
                                  "filesystem" if host == "daytona" else
                                  "a hook denies every skills_v* read — including this run's own "
                                  "previous version") + ". It is the set being revised: it is "
            "seeded into the staging dir, and "
            "the countersigned prompt must cover exactly what the evolver is shown of it. Quote "
            "each file's full text in a fenced block, under its name, and name no skills_v* path.",
            "",
            "What the evolver CAN open about libraries, and the prompt says so once: every evidence "
            "directory from e2 on stores the exact library that produced its rollouts in its own "
            "`_library/` subdirectory (the newest directory's is a byte-identical copy of the set "
            "below). Name those `_library/` directories by their absolute paths. Do NOT reproduce "
            "the earlier libraries; like the earlier rollouts, they are read on demand.",
            "",
            "THE PRIOR SET, which you must carry into the prompt:",
        ]
        for f in files:
            out += ["", f"--- {f.parent.name}/SKILL.md ---", f.read_text(errors="replace").rstrip()]
        return "\n".join(out)
    out = [
        f"A PRIOR SKILL SET OF {len(files)} SKILL(S) EXISTS, and the evolver is revising it — so the "
        "prompt must say what changed since it was written, and must judge each prior rule against "
        "this batch's evidence of that rule IN USE.",
        "",
        "*** REPRODUCE THE PRIOR SKILL SET VERBATIM INSIDE THE PROMPT. *** The evolver CANNOT OPEN "
        "it: the evolver is granted only the digests, its staging dir and the oracle, and a hook "
        "denies every skills_v* read — including this run's own previous version. A prompt that "
        f"says 'read them at `{prior}`' is telling the evolver to do something it cannot do, and "
        "no gate catches that. Quote each file's full text in a fenced block, under its name, so "
        "the prompt is self-contained. Do not name the path at all.",
        "",
        "THE PRIOR SET, which you must carry into the prompt:",
    ]
    for f in files:
        out += ["", f"--- {f.parent.name}/SKILL.md ---", f.read_text(errors="replace").rstrip()]
    return "\n".join(out)


def _ao_sandbox(out, digests, oracle, log, epoch: int, evidence_dirs=None, cwd=None):
    """(argv_prefix, profile_path_or_None) putting the AO under the SAME containment as the evolver.

    The AO runs with `--dangerously-skip-permissions`, so without a sandbox it could write any
    file the loop later trusts: ask() reads ANSWERS_e{N}.json, and the AO runs BEFORE any step
    that reads a person's answers, so an unsandboxed AO could pre-answer its own clarification
    questions. Containment by prose in the author message ("Write that file and nothing else")
    is a tripwire, not a boundary.

    WRITES are the narrow part and are granted as LITERALS: exactly ITER{N}_PROMPT.md and the
    log. Not their parent, which is the run dir.

    READS follow what the AO demonstrably needs, taken from its own tool calls rather than
    guessed: the evidence surface, the train-only oracle, the run's SPEC and a PROJECTION of its
    RUNTIME (never the raw file: its notes quote val results — see _ao_runtime_view). The prior
    epoch's prompt is granted so continuity-by-inspection keeps working. The repo's src/ and
    scripts/ come free from the shared builder, which is what let the AO verify the retriever
    against source — a capability worth keeping.

    IT REFUSES RATHER THAN DEGRADING. Returning `([], None)` and recording `containment: "NONE"`
    in a JSON field nobody reads would dispatch the AO completely uncontained on a host without
    sandbox-exec (any Linux, a stripped macOS image): able to write ANSWERS_e{N}.json, whose whole
    purpose is to carry a HUMAN's answer. A prose claim of loudness attached to a silent code path
    is a tripwire, not a boundary.

    An explicit opt-out exists — EVOSKILL_AO_ALLOW_UNCONTAINED=1 — because a deliberate,
    recorded decision to run without containment is legitimate and an unrecorded one is not.
    """
    from .evolver_dispatch import EvolverJob, build_evolver_sandbox_profile, _sandbox_exec_available
    import tempfile as _tf
    if not _sandbox_exec_available():
        if os.environ.get("EVOSKILL_AO_ALLOW_UNCONTAINED") == "1":
            return ([], "UNCONTAINED-BY-EXPLICIT-OPT-OUT")
        raise ValueError(
            "sandbox-exec is unavailable, so the AO cannot be contained. It would run "
            "with write access to the run directory, which holds the operator's answer channel "
            "and the countersignature the AO's own work is gated by — and it runs BEFORE that "
            "gate. Refusing to dispatch.\n"
            "  To run anyway, as a recorded decision: EVOSKILL_AO_ALLOW_UNCONTAINED=1")
    rd = out.parent
    # THE AO MUST NOT READ THE WHOLE RUN DIR. `work_dir=rd` would make the builder emit
    # `(allow file-read* (subpath <run dir>))` AFTER every deny, so the AO could open val legs,
    # E*_GATE_RESULT.json, SEALED_TEST_RESULT.json, <run>_test_*.json, drive logs and earlier
    # evolver transcripts -- and each sentence it writes is another chance to carry a held-out
    # figure past the held-out-figure regex. The profile's work dir is a scratch dir OUTSIDE the
    # run dir (the CLI's own private temp root, below), the run dir is hidden BY NAME (so a run dir
    # outside the home tree is covered too), and exactly what the AO needs is granted back after
    # the hide: the evidence dir(s), the oracle, SPEC.json, a PROJECTION of RUNTIME.json, the
    # previous iteration's prompt, its own artifact's name family and its own log. The CLI's cwd
    # stays the run dir (--resume is cwd-bound), granted as directory NODES only, as the shared
    # builder does for every ancestor.
    ev = [digests] if evidence_dirs is None else list(evidence_dirs)
    grants = [p for p in (*ev, oracle) if p and Path(p).is_dir()]
    # NOT THE RAW RUNTIME.json. Real ones can carry free-text '_' notes that quote val-leg results
    # ('val 36/36') and a base_leg_val key -- the held-out exposure the hide closes, re-opened by
    # one literal grant. The AO gets a projection of the operational keys,
    # written beside it by the orchestrator (e{N}_AO_RUNTIME_VIEW.json: listable where the AO
    # already looks, and a record of exactly what it could read).
    view = _ao_runtime_view(rd, epoch)
    lits = [p for p in (rd / "SPEC.json", view,
                        rd / f"ITER{epoch - 1}_PROMPT.md") if p is not None and p.exists()]
    # ...and nothing granted on that basis may carry a held-out figure (the leakage check's own rule).
    from .invariants import held_out_figures
    for _lit in (rd / "SPEC.json", view):
        if _lit is not None and _lit.is_file():
            _figs = held_out_figures(_lit.read_text(errors="replace"))
            if _figs:
                raise ValueError(f"{_lit.name} states a held-out figure {_figs[:2]} and the AO "
                                 f"would be granted it; refusing to dispatch (held-out figure). "
                                 f"Remove the figure from {_lit.name}.")
    # The CLI keeps every session's scratch and Bash task outputs under
    # /tmp/claude-<uid>, which the shared profile now denies. The AO's CLI gets a private root of
    # its own there instead (CLAUDE_CODE_TMPDIR, carried by the argv below so every consumer of it
    # — author_prompt, the live fixtures — hands it to `claude`). It doubles as the profile's
    # work dir: outside the run dir, read+write for this AO only. NOT created here — the CLI
    # creates it inside the grant — so a refusal or a caller that never launches leaves nothing.
    from .evolver_dispatch import mint_cli_tmpdir
    cli_tmp = mint_cli_tmpdir("ao")
    scratch = Path(cli_tmp).resolve()

    def _refuse(msg: str):
        """Refuse to dispatch. Nothing has been created yet: the profile file is written last."""
        raise ValueError(msg)
    # DENY THE THINGS BEING PROTECTED, NOT A PREFIX THEY HAPPEN TO SIT UNDER. The shared builder
    # denies writes by LOCATION (repo, home). A run dir anywhere else -- an external volume, /tmp,
    # a scratch mount -- matches neither prefix, `(allow default)` applies, and the AO could write
    # the operator's ANSWERS file, its own countersignature and every skills_v*: the whole
    # containment boundary would depend on where the operator put the run. So the run dir and the
    # operator dir are denied by name.
    job = EvolverJob(epoch=epoch, work_dir=scratch, prompt="", model="", log_path=log,
                     grants=[Path(g).resolve() for g in grants], writes=[],
                     hide=[rd.resolve()],
                     no_write=[rd.resolve(), operator_dir_for(rd)])
    prof = build_evolver_sandbox_profile(job)
    # CHECK THE ARTIFACT, NOT THE RECIPE. Asserting the emitted bytes carry both denies is
    # stronger than asserting the builder was called correctly: if the builder ever stops
    # honouring `no_write`, this refuses to dispatch instead of quietly running uncontained.
    for _must in (rd.resolve(), operator_dir_for(rd)):
        if f'(deny file-write* (subpath "{_must}"))' not in prof:
            _refuse(f"the emitted AO sandbox profile does not deny writes to {_must}; "
                    f"refusing to dispatch with containment that cannot be confirmed")
    # ...and the read half: the run dir is hidden by name and never granted as a tree.
    if f'(deny file-read* (subpath "{rd.resolve()}"))' not in prof or any(
            f'(allow file-{m}* (subpath "{rd.resolve()}"))' in prof for m in ("read", "write")):
        _refuse(f"the emitted AO sandbox profile does not hide the run dir {rd} (or re-grants it "
                f"as a tree); refusing to dispatch an AO that could read val legs and gate "
                f"results (the run dir must stay hidden)")
    # The literal grant admits only a direct open() of the exact path, but Claude Code's
    # Write tool writes ATOMICALLY -- a temp file BESIDE the target (NAME.tmp.<x>, .NAME.tmp)
    # then rename -- so with the literal alone the AO could never materialise its own
    # artifact (a direct shell write passes while both temp shapes EPERM). Admit exactly the
    # artifact's own NAME FAMILY in
    # its own directory; answers/countersignature/staging stay denied by name, and no
    # consumer reads any of the extra names. Appended AFTER the denies because the last
    # matching rule wins in sandbox profiles.
    import re as _re
    _family = ('^' + _re.escape(str(out.parent.resolve()))
               + r'/\.?' + _re.escape(out.name) + r'(\.[^/]*)?$')
    prof += '\n(allow file-write* (regex #"' + _family + '"))\n'
    prof += f'(allow file-write* (literal "{out.resolve()}"))\n'
    # The run dir is hidden, so the AO's own artifact must be granted back for READ as
    # well — the Write/Edit tools re-read what they wrote.
    prof += '(allow file-read* (regex #"' + _family + '"))\n'
    prof += "\n" + "\n".join(f'(allow file-read* (literal "{x.resolve()}"))' for x in lits) + "\n"
    # The CLI's cwd (the run dir, unless the caller moved it) and its ancestors as directory
    # NODES (a denied ancestor of cwd is a fatal EPERM for the CLI), plus the config names the CLI
    # probes in its cwd. None exists in a run dir and the AO cannot create them (writes denied);
    # granting the names keeps the startup the AO had before the hide.
    _cwd = Path(cwd or rd).resolve()
    _nodes, _cur = [], _cwd
    while True:
        _nodes.append(_cur)
        if _cur == _cur.parent or _cur == Path.home() or len(_cur.parts) <= 2:
            break
        _cur = _cur.parent
    # The PARENT nodes are only those under HOME, as the evolver builder does. HOME is where the
    # location deny makes an ancestor fatal for the CLI; outside
    # it the broad system allow already covers an ancestor, unless something denies it on
    # purpose — and re-opening a denied ancestor is exactly the leak: a run dir under
    # /tmp/claude-<uid> made that per-uid root (every session's task outputs) listable. The cwd
    # node itself is always kept: it is the AO's own run dir.
    _homes = {Path.home(), Path.home().resolve()}
    _nodes = _nodes[:1] + [n for n in _nodes[1:]
                           if any(n == h or h in n.parents for h in _homes)]
    # "under HOME" is only as good as $HOME. With HOME at or above the
    # per-uid CLI temp root (HOME=/tmp, say), that root passed the filter above and was granted
    # again (its listing names every session's dir on this host). So, whatever HOME says: no
    # parent node that IS a per-uid root or lies above one. The cwd node stays.
    from .evolver_dispatch import cli_uid_roots as _uid_roots
    _roots = [r.resolve() for r in _uid_roots()]
    _nodes = _nodes[:1] + [n for n in _nodes[1:] if not any(n == r or n in r.parents
                                                             for r in _roots)]
    prof += "; the AO's cwd and its ancestors, as NODES only\n"
    prof += "".join(f'(allow file-read* (literal "{n}"))\n' for n in _nodes)
    prof += "".join(f'(allow file-read* (literal "{_cwd / n}"))\n'
                    for n in ("CLAUDE.md", "CLAUDE.local.md", ".mcp.json"))
    prof += f'(allow file-read* (subpath "{_cwd / ".claude"}"))\n'
    prof += _own_transcript_grant(rd)
    # THE PROFILE FILE IS CREATED LAST: an exception or a refusal above leaves nothing behind.
    # Removed by ao_sandbox_cleanup().
    fh = _tf.NamedTemporaryFile("w", suffix=".ao.sb", delete=False)
    fh.write(prof); fh.close()
    return (["/usr/bin/env", f"CLAUDE_CODE_TMPDIR={cli_tmp}", "sandbox-exec", "-f", fh.name],
            fh.name)


# The RUNTIME.json keys the AO may read — operational plumbing only.
# Never a '_' note (free text that quotes val/sealed results), never a key naming a held-out
# split (base_leg_val), and only plain values (a string, number, bool, or a list of those).
AO_RUNTIME_KEYS = ("leg_script", "shared_dir", "oracle", "prompt_policy", "prompt_author_model",
                   "ao_continuous_session", "evolver_sandbox", "evolver_evidence", "evolver_host",
                   "evolver_timeout_s", "leg_timeout_s", "digest_bucket", "digest_caches",
                   "digest_snapshot_roots", "cache_tag", "base_leg", "evolver_log")


def ao_runtime_view(runtime: dict) -> dict:
    """The projection of a RUNTIME.json the AO is granted. Pure; see AO_RUNTIME_KEYS."""
    from .invariants import held_out_figures

    def plain(v):
        if isinstance(v, bool) or v is None or isinstance(v, (int, float)):
            return True
        if isinstance(v, str):
            return len(v) <= 1024 and not held_out_figures(v)
        if isinstance(v, list):
            return len(v) <= 64 and all(not isinstance(x, list) and plain(x) for x in v)
        return False
    out = {"_about": ("Operational RUNTIME.json keys only, projected for the prompt author. "
                      "Free-text notes and any key naming a held-out split are withheld.")}
    if isinstance(runtime, dict):
        for k in AO_RUNTIME_KEYS:
            if k in runtime and "val" not in k and "test" not in k and plain(runtime[k]):
                out[k] = runtime[k]
    return out


def _ao_runtime_view(rd: Path, epoch: int) -> "Path | None":
    """Write e{epoch}_AO_RUNTIME_VIEW.json beside RUNTIME.json; None when there is no RUNTIME.
    An unreadable RUNTIME yields a view with no keys rather than the raw file."""
    src = Path(rd) / "RUNTIME.json"
    if not src.is_file():
        return None
    try:
        rt = json.loads(src.read_text())
    except Exception:                                           # noqa: BLE001
        rt = None
    dst = Path(rd) / f"e{epoch}_AO_RUNTIME_VIEW.json"
    dst.write_text(json.dumps(ao_runtime_view(rt), indent=1))
    return dst


def ao_cli_tmpdir(argv) -> "Path | None":
    """The CLAUDE_CODE_TMPDIR an _ao_sandbox argv hands the AO's CLI, resolved; None if none."""
    for a in argv or []:
        if isinstance(a, str) and a.startswith("CLAUDE_CODE_TMPDIR="):
            return Path(a.split("=", 1)[1]).resolve()
    return None


def ao_sandbox_cleanup(argv, prof) -> None:
    """Remove what one _ao_sandbox call and the AO it contained left behind: the profile file and
    the AO CLI's private temp root. Only ever a path mint_cli_tmpdir could have produced (an
    'ao.' dir directly under a per-uid CLI temp root) is removed."""
    import shutil as _sh
    from .evolver_dispatch import cli_uid_roots
    if prof and prof != "UNCONTAINED-BY-EXPLICIT-OPT-OUT":
        try:
            os.unlink(prof)
        except OSError:
            pass
    d = ao_cli_tmpdir(argv)
    if d is not None and d.name.startswith("ao.") and d.parent in cli_uid_roots() \
            and not d.is_symlink():
        _sh.rmtree(d, ignore_errors=True)


def operator_dir_for(run_dir):
    """The operator channel's directory for a run. Imported from loop rather than re-derived: two
    definitions of where the answer channel lives is exactly one too many (the profile denies it by
    name, and a second spelling of the name would deny the wrong directory)."""
    from .loop import operator_dir
    return operator_dir(run_dir).resolve()


def _own_transcript_grant(run_dir) -> str:
    """Read access to THIS RUN'S conversation, and no other.

    The shared profile denies `~/.claude/projects` wholesale, because that tree holds every
    session transcript on the machine. That deny is correct and on its own it breaks `--resume`: the
    CLI resolves a session by READING `~/.claude/projects/<slug-of-realpath(cwd)>/<sid>.jsonl`, so
    epoch 2 would get `No conversation found with session ID: ...` while the file sits on disk,
    present and unreadable. Epoch 1 passes (it only WRITES), which is why a one-shot probe would
    call the containment good.

    THE SLUG IS FOUND, NOT DERIVED. The slugging rule is the CLI's ('/'->'-', '_'->'-', dots
    kept, as observed) and a wrong guess here would silently re-break resume — the same reasoning
    as _transcript_exists. So: glob for the file named after the session id we already hold, and
    grant the DIRECTORY THAT CONTAINS IT. On the first dispatch there is no transcript yet and
    nothing is granted, which is correct: epoch 1 does not resume anything.
    """
    import json as _json
    from pathlib import Path as _P
    sp = ao_session_path(run_dir)
    try:
        sid = (_json.loads(sp.read_text()) or {}).get("session_id")
    except Exception:                                           # noqa: BLE001
        return ""
    if not isinstance(sid, str) or not sid.strip():
        return ""
    hits = list(_P.home().glob(f".claude/projects/*/{sid}.jsonl"))
    if not hits:
        return ""
    return ("\n; THIS RUN'S OWN conversation only, found by session id rather than by\n"
            "; reconstructing the CLI's slugging rule. Every other project stays denied.\n"
            + "\n".join(f'(allow file-read* (subpath "{h.parent}"))' for h in hits) + "\n")


def ao_session_path(run_dir) -> "Path":
    """Where the AO's session id lives, ONE PER RUN, not per epoch.

    Per-run is the point. A per-epoch id would give unrelated sessions: `claude -p` with no session
    flags means epoch N has no memory of epoch N-1, so the AO re-derives the benchmark from harness
    source every time (most of an epoch's tool calls can go to locating settings in src/ and
    third_party/). A run-level session lets the settled understanding survive an epoch boundary.
    """
    from pathlib import Path
    return Path(run_dir) / "AO_SESSION.json"


def _transcript_exists(sid: str) -> bool:
    """Has the CLI ever actually opened a conversation under this id?

    Claude Code writes `~/.claude/projects/<slug-of-realpath(cwd)>/<sid>.jsonl` once it accepts
    argv and starts. Globbing across all project slugs rather than reconstructing the slug is
    deliberate: the slugging rule is the CLI's, not ours, and a wrong guess here would silently
    report "no conversation" for a live session and re-mint over it — strictly worse than the
    defect this answers.
    """
    from pathlib import Path as _P
    try:
        return any(_P.home().glob(f".claude/projects/*/{sid}.jsonl"))
    except OSError:
        return False


def _mint_session(sp, epoch: int, cwd: str) -> tuple:
    """Create AO_SESSION.json exclusively, or lose the race and read the winner's id.

    `if sp.is_file()` then `sp.write_text(...)` is check-then-write. Two concurrent first uses
    would each see no file, each mint a distinct uuid, each return mode='new' -- and the file keeps
    one, so the other id goes to a real `claude --session-id` whose conversation is then
    unreachable forever. O_EXCL makes the create the arbiter, and the loser reads the id the winner
    actually recorded.
    """
    import json as _json
    import os as _os
    import uuid as _uuid
    sid = str(_uuid.uuid4())
    payload = _json.dumps({
        "session_id": sid, "first_epoch": epoch, "cwd": cwd, "confirmed": False,
        "minted_pid": _os.getpid(),
        "note": "ONE AO conversation per run. --session-id on first use, --resume thereafter. "
                "`confirmed` flips to true once the CLI has actually opened this conversation; "
                "until then the id may be re-minted, because an id no transcript exists for "
                "cannot be resumed.",
    }, indent=1)
    sp.parent.mkdir(parents=True, exist_ok=True)
    # EXCLUSIVE **AND** ATOMIC. O_EXCL alone closes the double-mint but opens a second race:
    # create-then-write is two steps, so the loser can read the winner's file in the window
    # between them, get an EMPTY file, and raise "AO_SESSION.json is unreadable -- refusing to
    # lose the existing conversation" on a conversation that is being created correctly. Write the content into a temp file first and publish it with
    # os.link(), which is atomic and still fails if the target exists: readers only ever see a
    # complete record, and only one caller can win.
    # UNIQUE PER CALLER, not per process. A temp name keyed on the pid alone is shared by two
    # THREADS in one process: thread A writes its payload, thread B overwrites the same temp file,
    # and A's os.link publishes B's id while A returns its own as "new" -- two live ids again.
    tmp = sp.parent / f".{sp.name}.{_os.getpid()}.{_uuid.uuid4().hex[:8]}.tmp"
    tmp.write_text(payload)
    try:
        _os.link(str(tmp), str(sp))
    except FileExistsError:
        tmp.unlink(missing_ok=True)
        return (None, None)                       # someone else won; caller re-reads the file
    finally:
        tmp.unlink(missing_ok=True)
    return (["--session-id", sid], sid)


def _atomic_write_json(path, obj) -> None:
    """Replace `path` with `obj` such that no reader can observe a partial file. See _mint_session."""
    import json as _json
    import uuid as _uuid
    tmp = Path(path).parent / f".{Path(path).name}.{os.getpid()}.{_uuid.uuid4().hex[:8]}.rw"
    tmp.write_text(_json.dumps(obj, indent=1))
    os.replace(str(tmp), str(path))


def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _is_burned(d: dict, sid: str) -> bool:
    """Is this a recorded id that no conversation exists for AND nobody is about to create one?

    Re-minting on `not confirmed and not _transcript_exists(sid)` alone would DEFEAT THE O_EXCL
    MINT: the loser of a concurrent first-use reads the winner's brand-new record, sees an id with
    no transcript (the winner has not dispatched yet), deletes the file and mints its own --
    reproducing the orphaned-conversation race the exclusive create closes.

    So "no transcript" is necessary and not sufficient. A record is burned only if ALSO either the
    dispatch is known to have failed (`burned`, written by author_prompt when the CLI exits
    non-zero having created nothing), or the process that minted it is GONE — a live minter is
    either dispatching now or about to.

    Residual, stated rather than hidden: a recycled pid can make a genuinely burned id look live.
    The cost is one confusing "No conversation found" from the CLI and a deliberate delete, which
    is strictly better than silently orphaning a real conversation.
    """
    if d.get("confirmed"):
        return False
    if _transcript_exists(sid):
        return False
    return bool(d.get("burned")) or not _pid_alive(d.get("minted_pid"))


def _burn_session(run_dir, sid: str) -> None:
    """Record that this id's dispatch failed before any conversation existed. See _is_burned."""
    import json as _json
    sp = ao_session_path(run_dir)
    try:
        d = _json.loads(sp.read_text())
        if isinstance(d, dict) and d.get("session_id") == sid and not d.get("confirmed") \
                and not _transcript_exists(sid):
            d["burned"] = True
            _atomic_write_json(sp, d)
    except Exception:                                           # noqa: BLE001
        pass


def _confirm_session(run_dir, sid: str) -> None:
    """Mark the session id as one a conversation demonstrably exists for. Best-effort: failing to
    record confirmation costs a transcript probe next time, never correctness."""
    import json as _json
    sp = ao_session_path(run_dir)
    try:
        d = _json.loads(sp.read_text())
        if isinstance(d, dict) and d.get("session_id") == sid and not d.get("confirmed"):
            d["confirmed"] = True
            _atomic_write_json(sp, d)
    except Exception:                                           # noqa: BLE001
        pass


def _session_argv(run_dir, epoch: int, cwd=None) -> tuple:
    """Return (argv_fragment, session_id, mode) for a CONTINUOUS AO session.

    First use mints a uuid and passes --session-id; every later use passes --resume, which the
    CLI supports together with --print. The id is persisted immediately, BEFORE the model runs:
    if the process dies mid-authoring, the next invocation must resume that session rather than
    silently start a fresh one and lose the conversation (a record written only on success is
    not a record).

    THREE THINGS THAT PERSIST-BEFORE-DISPATCH GETS WRONG ON ITS OWN:

      A BURNED ID WEDGES THE RUN. The CLI creates the transcript only once it has accepted argv
           and started. Any failure before that -- a bad flag, `claude` not on PATH, the binary
           refusing -- leaves an id no conversation exists for: `claude -p ... --session-id U
           --nonexistent-flag-zzz` exits 1 and creates ZERO files under ~/.claude/projects.
           Without a re-mint branch, every later epoch would pass `--resume <dead id>` and fail
           identically, forever. So the record carries `confirmed`, and an UNCONFIRMED id with
           no transcript on disk is re-minted rather than resumed. A CONFIRMED id is never
           re-minted -- losing a real conversation is the worse error, and that is the whole
           reason the record is written early.

      --resume IS CWD-BOUND. Transcripts are stored per realpath(cwd); `--resume <sid>` from any
           other directory exits 1 "No conversation found". Moving or renaming the run dir would
           turn continuity into a hard failure with a misleading message, so the mint cwd is
           recorded and a mismatch REFUSES with the actual directory to run from.

      the mint race -- see _mint_session.
    """
    import json as _json
    from pathlib import Path as _P
    cwd = str(_P(cwd or _P.cwd()).resolve())
    sp = ao_session_path(run_dir)
    if not sp.is_file():
        argv, sid = _mint_session(sp, epoch, cwd)
        if argv is not None:
            return (argv, sid, "new")
        # lost the O_EXCL race — fall through and resume whatever the winner recorded
    try:
        d = _json.loads(sp.read_text())
    except Exception as e:                                      # noqa: BLE001
        # A corrupt session file is NOT a reason to start fresh. Starting fresh would look
        # like success while silently discarding the conversation the run depends on.
        raise ValueError(f"{sp.name} is unreadable ({type(e).__name__}: {e}). Refusing to "
                         f"start a new AO session and lose the existing conversation. "
                         f"Fix or delete the file deliberately.")
    if not isinstance(d, dict):
        # The try/except above wraps only json.loads, so `[]`, `"abc"` or `3` -- all valid JSON --
        # would reach `d.get(...)` and raise AttributeError, escaping BOTH guarded refusals.
        raise ValueError(f"{sp.name} holds a {type(d).__name__}, not a JSON object. It must be "
                         f'{{"session_id": "<uuid>", ...}}. Fix or delete the file deliberately.')
    sid = d.get("session_id")
    if not (isinstance(sid, str) and sid.strip()):
        raise ValueError(f"{sp.name} has no usable session_id: {d!r}")
    rec_cwd = d.get("cwd")
    if isinstance(rec_cwd, str) and rec_cwd and str(_P(rec_cwd).resolve()) != cwd:
        raise ValueError(
            f"{sp.name} was minted with cwd={rec_cwd} and `claude --resume` only resolves a "
            f"session from the directory it was created in; this call is from {cwd}. Run from "
            f"{rec_cwd}, or delete {sp.name} deliberately to start a new conversation and accept "
            f"that the earlier epochs' reasoning is not carried forward.")
    if _is_burned(d, sid):
        try:
            sp.unlink()
        except OSError:
            pass
        argv, new_sid = _mint_session(sp, epoch, cwd)
        if argv is not None:
            return (argv, new_sid, "new (re-minted: no conversation exists for the recorded id)")
        d = _json.loads(sp.read_text())
        sid = d.get("session_id")
    return (["--resume", sid], sid, "resume")


class _author_lock:
    """Mutual exclusion for AO dispatch within one run dir. See the call site in author_prompt.

    O_EXCL create, pid recorded, STOLEN if the recorded pid is dead. Stealing matters: a lock that
    survives a killed orchestrator would wedge the run forever, and "the run cannot proceed and
    nobody knows why" is the failure mode this programme keeps paying for.
    """

    def __init__(self, run_dir, epoch: int):
        self.path = Path(run_dir) / ".ao_author.lock"
        self.epoch = epoch
        self.fd = None

    def _owner(self):
        import json as _json
        try:
            return _json.loads(self.path.read_text())
        except Exception:                                       # noqa: BLE001
            return {}

    @staticmethod
    def _alive(pid) -> bool:
        try:
            os.kill(int(pid), 0)
            return True
        except (OSError, TypeError, ValueError):
            return False

    def __enter__(self):
        import json as _json
        for attempt in (1, 2):
            try:
                self.fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                os.write(self.fd, _json.dumps({"pid": os.getpid(), "epoch": self.epoch}).encode())
                return self
            except FileExistsError:
                own = self._owner()
                if attempt == 1 and not self._alive(own.get("pid")):
                    try:
                        self.path.unlink()          # stale: the holder is gone
                    except OSError:
                        pass
                    continue
                raise ValueError(
                    f"another AO authoring dispatch is already running in this run dir "
                    f"(pid {own.get('pid')}, epoch {own.get('epoch')}). Two dispatches would "
                    f"--resume the same conversation concurrently and FORK it, and the next epoch "
                    f"would see only one branch. Wait for it, or remove {self.path.name} if you "
                    f"are certain that process is gone.")
        raise ValueError(f"could not acquire {self.path.name}")   # unreachable; explicit anyway

    def __exit__(self, *exc):
        if self.fd is not None:
            try:
                os.close(self.fd)
            except OSError:
                pass
        try:
            self.path.unlink()
        except OSError:
            pass
        return False


_EVIDENCE_NAME = re.compile(r"^digests_e(\d+)_train$")


def _evidence_lines(evidence_dirs) -> list:
    """One line per granted evidence dir: absolute path and the library that produced it.
    The epoch is read from the canonical name EpochPaths gives the dir; anything else refuses."""
    dirs = [Path(d).resolve() for d in evidence_dirs]
    js = []
    for d in dirs:
        m = _EVIDENCE_NAME.match(d.name)
        if not m:
            raise ValueError(f"{d} is not a digests_e<j>_train dir; the evidence list is "
                             f"EpochPaths.evidence_dirs() and nothing else")
        js.append(int(m.group(1)))
    if js != list(range(1, len(dirs) + 1)):
        raise ValueError(f"the evidence list must be digests_e1..e{len(dirs)}_train in order, "
                         f"got epochs {js}")
    out = []
    for j, d in zip(js, dirs):
        if j == 1:
            src = "rollouts of the solver carrying NO skills (the base leg)"
        else:
            src = (f"rollouts of the solver carrying skills_v{j - 1}; that library's exact files "
                   f"are in {d}/_library/")
        if j == len(dirs):
            src += " — THIS BATCH, the one every headline figure in FACTS describes"
        out.append(f"    e{j}  {d}  — {src}")
    return out


def _author_user_prompt(policy_text: str, facts: Facts, out: Path, epoch: int,
                        digests: Path, oracle: Path, prior: Path | None, staging: Path,
                        evidence_dirs: list | None = None, host: str = "local") -> str:
    """The message the AUTHORING model receives. The policy is the how; the facts are the what.

    `evidence_dirs` (all-epochs mode, RUNTIME evolver_evidence): every digest dir the evolver is
    granted, as EpochPaths.evidence_dirs() gives them. The message then names each by absolute
    path with its producing library, says the earlier ones are read on demand, and drops the old
    "describe the failure modes QUALITATIVELY" sentence, which contradicted the policy. None keeps
    the message byte-identical to before.
    """
    if evidence_dirs is not None:
        return _author_user_prompt_all_epochs(policy_text, facts, out, epoch, evidence_dirs,
                                              oracle, prior, staging, host=host)
    prior_note = _prior_note(prior)
    return "\n".join([
        "You are assembling the evolver's user prompt for iteration " + str(epoch) + ".",
        "",
        "Follow the POLICY below exactly. Ground every quantity in the FACTS block, verbatim.",
        f"Read the evidence at `{digests}` and the golden specifications at `{oracle}` to describe the",
        "failure modes QUALITATIVELY — that is what the traces are for, and it is the part no number",
        "can supply. Those directories are TRAIN-only by construction; do not look for anything else.",
        "",
        prior_note,
        "",
        "THE EVOLVER MUST BE TOLD TO WRITE ITS SKILL SET INTO THIS EXACT DIRECTORY:",
        f"    {staging}",
        "State that literal path in the prompt. Do NOT tell it to write where skills live at solver",
        "RUNTIME (a .claude/skills/ or similar) — the pipeline collects the deliverable from the",
        "staging path above, and a skill set written anywhere else is invisible to every later step.",
        "",
        f"WRITE THE FINISHED PROMPT TO `{out}` using the Write tool. Write that file and nothing else.",
        "Do not create scratch files. Your final message should be one short paragraph saying what you",
        "emphasised and why — it is read by an operator, not by the evolver.",
        "",
        "=" * 78, "POLICY", "=" * 78, "",
        policy_text,
        "", "=" * 78, "FACTS", "=" * 78, "",
        facts.as_block(),
    ])


def _author_user_prompt_all_epochs(policy_text: str, facts: Facts, out: Path, epoch: int,
                                   evidence_dirs: list, oracle: Path, prior: Path | None,
                                   staging: Path, host: str = "local") -> str:
    """All-epochs mode's author message. Every path is absolute and resolved: the evolver's cwd
    is its isolated work dir, so a relative path there resolves to nothing."""
    return "\n".join([
        "You are assembling the evolver's user prompt for iteration " + str(epoch) + ".",
        "",
        "Follow the POLICY below exactly. Ground every quantity in the FACTS block, verbatim.",
        "",
        "THE EVOLVER IS GRANTED THE EVIDENCE OF EVERY EPOCH SO FAR, one directory per epoch, each",
        "TRAIN-only by construction and each an absolute path it can open:",
        *_evidence_lines(evidence_dirs),
        "Name EVERY one of them in the prompt by exactly that absolute path, with the library that",
        "produced it (POLICY section 5). Never write a relative digest path or a bare directory name,",
        "and never name a digest directory that is not in this list: a check refuses the prompt",
        "otherwise, before the operator reads it.",
        "The earlier directories are read by the evolver ON DEMAND. Do not paste, summarise or",
        "characterise their contents, and do not compare the batches; about each directory, state",
        "only what the EVIDENCE INDEX at the end of FACTS gives.",
        f"The golden specifications are at `{Path(oracle).resolve()}`, also train-only. You may open",
        "these directories and the oracle only to state what the harness IS; do not look for",
        "anything else.",
        "",
        _prior_note(prior, all_epochs=True, host=host),
        "",
        "THE EVOLVER MUST BE TOLD TO WRITE ITS SKILL SET INTO THIS EXACT DIRECTORY:",
        f"    {Path(staging).resolve()}",
        "State that literal path in the prompt. Do NOT tell it to write where skills live at solver",
        "RUNTIME (a .claude/skills/ or similar) — the pipeline collects the deliverable from the",
        "staging path above, and a skill set written anywhere else is invisible to every later step.",
        "",
        f"WRITE THE FINISHED PROMPT TO `{out}` using the Write tool. Write that file and nothing else.",
        "Do not create scratch files. Your final message should be one short paragraph saying what you",
        "emphasised and why — it is read by an operator, not by the evolver.",
        "",
        "=" * 78, "POLICY", "=" * 78, "",
        policy_text,
        "", "=" * 78, "FACTS", "=" * 78, "",
        facts.as_block(),
    ])


def final_reply(log: Path) -> tuple[str, str]:
    """The AO's closing message from a stream-json log — (text, where_it_came_from).

    Three sources, in descending fidelity, and the SECOND return value names which was used so a
    reader never has to guess whether they are looking at the model's words or at a description
    of its absence. A run that was killed has no `result` event at all, and reporting silence as
    an empty string would make "it said nothing" indistinguishable from "it never got to speak" —
    the same class of ambiguity as session_requested-vs-session_mode on the receipt.
    """
    if not Path(log).is_file():
        return f"(no stream log at {log} — the AO's reply was never captured)", "missing-log"
    last_text, result = "", None
    for line in Path(log).read_text(errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue                       # a partially-written final line on a live log
        if e.get("type") == "result":
            result = e
        elif e.get("type") == "assistant" and not e.get("parent_tool_use_id"):
            txt = "\n".join(c.get("text", "") for c in
                            (e.get("message", {}) or {}).get("content", []) or []
                            if isinstance(c, dict) and c.get("type") == "text")
            if txt.strip():
                last_text = txt
    if result and result.get("result"):
        return str(result["result"]), "result-event"
    if last_text:
        return last_text, "last-assistant-text (no result event — killed or still running)"
    return (f"(the AO produced no closing text in {Path(log).name}; it may have been killed "
            f"before replying)"), "no-text"


def author_prompt(*, facts: Facts, out: Path, epoch: int, digests: Path, oracle: Path,
                  log: Path, staging: Path, model: str = "opus[1m]", policy: Path | None = None,
                  prior_skills: Path | None = None, timeout_s: int = 3600,
                  work_dir: Path | None = None, continuous_session: bool = False,
                  evidence_dirs: list | None = None, policy_addendum: Path | None = None,
                  host: str = "local", daytona_cfg: dict | None = None) -> dict:
    """Dispatch the authoring model and RETURN WITHOUT TRUSTING IT.

    The contract is deliberately narrow: the model's only permitted side effect is creating `out`.
    We verify that the file appeared and is substantial; every claim INSIDE it is then checked by
    the ordinary epoch gates (numbers, leakage, format), which is where a fabrication dies.
    """
    import subprocess
    import time
    # An all-epochs dispatch with no policy named gets the all-epochs policy, never the
    # single-batch one, whose section 5 describes one directory.
    pol = (policy or (ALL_EPOCHS_POLICY if evidence_dirs is not None else DEFAULT_POLICY))
    if not pol.is_file():
        raise ValueError(f"no prompt policy at {pol} — the AO cannot author a prompt without the "
                         f"how-to it is initialised with (RUNTIME.json 'prompt_policy')")
    if out.exists():
        raise ValueError(f"{out.name} already exists — refusing to overwrite an authored prompt. "
                         f"Move it aside if a re-author is intended.")
    # WHICH BYTES OF WHICH POLICY. The file at a policy path is shared by every run that names it,
    # so recording the path alone would let an edit change what an earlier record appears to
    # describe. The hash pins the instructions this AO was actually given.
    pol_sha256 = hashlib.sha256(pol.read_bytes()).hexdigest()
    pol_text = pol.read_text()
    add_sha256 = None
    if policy_addendum is not None:
        # Daytona host: an ADDENDUM file appended after the policy; neither existing policy file is
        # edited, and its sha is recorded next to the policy's (what was requested and what was
        # wired are both on the receipt).
        if not Path(policy_addendum).is_file():
            raise ValueError(f"no policy addendum at {policy_addendum}")
        add_sha256 = hashlib.sha256(Path(policy_addendum).read_bytes()).hexdigest()
        pol_text = (pol_text.rstrip() + "\n\n=== POLICY ADDENDUM (evolver host: " + host + ") ===\n\n"
                    + Path(policy_addendum).read_text())
    if host == "daytona":
        # The loop's paths may be relative; on the VM the AO's cwd is its work dir,
        # so the WRITE line and every grant must name the absolute (mirrored) path.
        out = Path(out).absolute()
    msg = _author_user_prompt(pol_text, facts, out, epoch, digests, oracle, prior_skills,
                              staging, evidence_dirs=evidence_dirs, host=host)
    work = work_dir or out.parent
    # PERSIST WHAT WE SEND. As argv alone, the ONE thing the operator hands the AO would be the
    # one thing no artifact records -- the AO's own stream-json log contains only tool_result
    # `user` events, never the prompt. Written before dispatch, so a crashed
    # authoring run still leaves the question behind.
    sent = out.parent / f"e{epoch}_AUTHOR_MESSAGE.md"
    sent.write_text(msg)
    # And record it as a TURN, not only as a file. The file answers "what did we send?"; the
    # transcript answers "what was the conversation?", which a per-artifact file cannot answer
    # because it has no ordering with respect to the AO's replies, its questions, or the
    # operator's answers. Recorded BEFORE dispatch: a run that dies mid-authoring must still show
    # what it was asked.
    conversation.record(out.parent, speaker="operator", channel="authoring", content=msg,
                        epoch=epoch,
                        meta={"model": model, "policy": pol.name, "sent_file": sent.name,
                              "chars": len(msg)})

    # ONE AUTHORING DISPATCH PER RUN AT A TIME. `if out.exists()` above is the only other guard
    # and it is False for the whole time the first `claude -p` is authoring, so a second --step
    # inside that window would run `--resume <same id>` against a LIVE session. Two concurrent
    # --resume of one id FORK the transcript, and the next epoch
    # sees one branch — the other epoch's entire reasoning silently absent from the conversation
    # the run is supposed to be accumulating. The lock is held by the ORCHESTRATOR (the AO cannot
    # write here), and a lock whose owner is dead is stolen rather than wedging the run.
    if host == "daytona":
        return _author_on_daytona(facts=facts, out=out, epoch=epoch, digests=digests, oracle=oracle,
                                  log=log, model=model, pol=pol, pol_sha256=pol_sha256, msg=msg,
                                  sent=sent, evidence_dirs=evidence_dirs, timeout_s=timeout_s,
                                  continuous_session=continuous_session, policy_addendum=policy_addendum,
                                  add_sha256=add_sha256, daytona_cfg=daytona_cfg or {})
    with _author_lock(out.parent, epoch):
        sess_argv, sid, sess_mode = ([], None, "single-shot")
        if continuous_session:
            sess_argv, sid, sess_mode = _session_argv(out.parent, epoch, cwd=work)
        # `evidence_dirs` (all-epochs mode, RUNTIME evolver_evidence): every digest dir the evolver
        # will be granted, absolute. None = today's single dir. The sandbox grants exactly these.
        sb_argv, sb_prof = _ao_sandbox(out, digests, oracle, log, epoch,
                                       evidence_dirs=evidence_dirs, cwd=work)
        _more = [x for d in (evidence_dirs or []) if Path(d).resolve() != Path(digests).resolve()
                 for x in ("--add-dir", str(d))]
        # --effort, for a subscription model only; see claude_pins.
        from .claude_pins import effort_argv as _effort_argv, pinned_env as _pinned_env
        from .claude_pins import pin_record as _pin_record
        argv = [*sb_argv, "claude", "-p", msg, "--model", model, *_effort_argv(model), *sess_argv,
                *_more,
                "--add-dir", str(digests), "--add-dir", str(oracle), "--add-dir", str(out.parent),
                # UNDOCUMENTED flag (absent from --help; argparse rejects bad values, so it
                # IS registered). Verified working on 2.1.220 — but ONLY once the reasoning
                # is long enough to summarize: short thoughts return the block with empty
                # text, which reads as "flag inert" if probed with easy prompts (measured:
                # 973-output-token probe -> 1,064 chars; trivial probes -> 0). Do NOT gate
                # it behind a help-grep capability check; that check fails forever.
                "--thinking-display", "summarized",
                "--dangerously-skip-permissions", "--output-format", "stream-json", "--verbose"]
        # FORCE THE AUTH PATH TO MATCH THE MODEL, rather than inheriting whatever the
        # launch shell had. An inherited env lets an exported ANTHROPIC_API_KEY silently move the
        # AO off the OAuth subscription and onto API billing, a drift between runs meant to be
        # comparable. For a gateway authoring model the env is returned untouched,
        # because such a model reaches its endpoint through precisely those variables.
        _env = subscription_env(model)
        # The AO's `claude` never inherits what identifies or reaches the
        # operator's own session (messaging socket/token, session/bridge ids, job dir and kind,
        # pid). Names only are recorded in the return value (e{N}_PROMPT_AUTHOR.json).
        from .evolver_dispatch import scrub_parent_session_env
        _env, _parent_scrubbed = scrub_parent_session_env(_env)
        # No model/refusal fallback, no auto-update, effort pinned in env too, and
        # subagents forced onto the same explicit model (claude_pins module doc).
        _env = _pinned_env(model, _env)
        _pins = _pin_record(model, _env)
        t0 = time.time()
        # The profile and the AO CLI's private temp root live exactly as long as the dispatch:
        # removed in a `finally`, so a timeout or a failed run leaves nothing behind.
        try:
            with log.open("w") as fh:
                fh.write(f"=== prompt author: epoch {epoch} model {model} policy {pol.name}\n")
                fh.write(f"=== facts sha: {abs(hash(facts.as_block())) % (10 ** 12)}\n")
                fh.flush()
                r = subprocess.run(argv, cwd=str(work), stdout=fh, stderr=subprocess.STDOUT,
                                   timeout=timeout_s, env=_env)
        finally:
            ao_sandbox_cleanup(sb_argv, sb_prof)
        dt = time.time() - t0
        if sid:
            # Record which of the two happened, at the only moment either is knowable.
            (_confirm_session if r.returncode == 0 else _burn_session)(out.parent, sid)
    # THE AO'S SIDE OF THE EXCHANGE, recorded before the exit-code checks below so a FAILED
    # authoring run still leaves what the AO said, ordered against the message that prompted it.
    reply, why = final_reply(log)
    # WHICH MODEL ACTUALLY ANSWERED, from the stream's own init/assistant events. The
    # caller (loop._run_author_prompt) HALTs on a mismatch after recording it; _v_prompt_authored
    # re-derives the same check from the log.
    from .claude_pins import check_session as _check_session
    _mchk = _check_session(log, model)
    conversation.record(out.parent, speaker="ao", channel="authoring", content=reply, epoch=epoch,
                        meta={"exit_code": r.returncode, "seconds": round(dt, 1),
                              "stream_log": str(log), "session_id": sid or "",
                              "source": why, "wrote": out.name if out.is_file() else "(nothing)"})
    if r.returncode != 0:
        raise ValueError(f"authoring model exited {r.returncode} after {dt:.0f}s; see {log.name}")
    if not out.is_file():
        raise ValueError(f"authoring model exited 0 but wrote NO {out.name} after {dt:.0f}s. "
                         f"Exit 0 is not evidence of an artifact — see {log.name}.")
    body = out.read_text()
    if len(body) < 1500:
        raise ValueError(f"{out.name} is only {len(body)} bytes — too short to carry the sections the "
                         f"policy requires; treating as a failed authoring run")
    extra = {}
    if evidence_dirs is not None:
        extra["evidence_dirs"] = [str(Path(d).resolve()) for d in evidence_dirs]
    if policy_addendum is not None:
        extra.update(policy_addendum=str(policy_addendum), policy_addendum_sha256=add_sha256,
                     evolver_host=host)
    return {**extra, "bytes": len(body.encode()), "seconds": round(dt, 1), "model": model,
            "policy": str(pol), "policy_sha256": pol_sha256, "log": str(log),
            "sandboxed": bool(sb_argv),
            "containment": "sandbox-exec" if sb_argv else "NONE (explicit opt-out)",
            "parent_env_scrubbed": _parent_scrubbed,
            "sent_message": str(sent), "sent_message_chars": len(msg),
            # `continuous_session` is recorded as ASKED-FOR, separately from what was actually
            # WIRED. Without both, a receipt saying session_mode='single-shot' is ambiguous
            # between "this run does not use a continuous session" and "it does, and the wiring
            # silently dropped it". _v_prompt_authored compares the two.
            "session_requested": bool(continuous_session),
            "session_mode": sess_mode, "session_id": sid,
            "session_argv": list(sess_argv), "session_cwd": str(Path(work).resolve()),
            # The pins sent and the model/CLI version the session reported.
            "pins": _pins, "resolved_model": _mchk["resolved_model"],
            "claude_code_version": _mchk["claude_code_version"],
            "model_check": {k: _mchk[k] for k in ("requested_model", "expected_model", "checked",
                                                   "ok", "violations", "notes", "init_models",
                                                   "assistant_models", "subagent_models",
                                                   "model_usage", "claude_code_versions")}}


def _author_on_daytona(*, facts, out, epoch, digests, oracle, log, model, pol, pol_sha256, msg, sent,
                       evidence_dirs, timeout_s, continuous_session, policy_addendum, add_sha256,
                       daytona_cfg) -> dict:
    """The AO on its own Daytona sandbox (loop RUNTIME evolver_host='daytona'), so it runs on the same
    kind of machine as the evolver. Same inputs as the Mac AO's sandbox-exec grants --
    evidence dirs, oracle, SPEC.json, the RUNTIME projection, the previous prompt -- uploaded by
    scripts/daytona/tau3_evolver_daytona.py `ao`, which brings back exactly ITER{N}_PROMPT.md, the
    stream-json log and the session transcript, with web tools off and egress api.anthropic.com
    only. Everything after the dispatch (final reply, model check, conversation record, the
    artifact checks, the receipt) is the same code path as the local AO."""
    import subprocess
    import time
    if continuous_session:
        raise ValueError("ao_continuous_session is not supported with evolver_host=daytona "
                         "(a --resume session cannot cross fresh sandboxes)")
    rd = out.parent
    view = _ao_runtime_view(rd, epoch)
    lits = [p for p in (rd / "SPEC.json", view, rd / f"ITER{epoch - 1}_PROMPT.md")
            if p is not None and Path(p).exists()]
    from .invariants import held_out_figures
    for _lit in (rd / "SPEC.json", view):
        if _lit is not None and Path(_lit).is_file():
            _figs = held_out_figures(Path(_lit).read_text(errors="replace"))
            if _figs:
                raise ValueError(f"{Path(_lit).name} states a held-out figure {_figs[:2]}; refusing (#253)")
    ev = [digests] if evidence_dirs is None else list(evidence_dirs)
    repo = Path(__file__).resolve().parents[2]
    # RUNTIME evolver_daytona.placement "pool_container": the AO runs in the pool machine's evolver
    # container too -- same `ao` flags, receipt e{N}_AO_DAYTONA.json and listing (scripts/daytona/evolver_container.py).
    _drv = "evolver_container.py" if (daytona_cfg or {}).get("placement") == "pool_container" \
        else "tau3_evolver_daytona.py"
    _wait_s = timeout_s + 1800
    if _drv == "evolver_container.py":
        # + the driver's bounded wait for the pool's current machine (evolver_daytona.machine_wait_s); on SIGTERM it
        # tears its container down and removes its pool lease (evolver_container._on_sigterm)
        from .pool_container_evolver import machine_wait_s as _mw
        _wait_s += int(_mw(daytona_cfg))
    from .host_tools import python_executable
    argv = [python_executable(),
            str(repo / "scripts/daytona" / _drv), "ao", "--run-dir", str(rd),
            "--epoch", str(epoch), "--message", str(sent), "--out", str(out), "--log", str(log),
            "--model", model, "--oracle", str(Path(oracle).absolute()), "--timeout-s", str(timeout_s)]
    for d in ev:
        argv += ["--evidence", str(Path(d).resolve())]
    for p in lits:
        argv += ["--literal", str(Path(p).absolute())]
    try:
        _sp = json.loads((rd / "SPEC.json").read_text()).get("spec", {}).get("split_path")
        if _sp:
            argv += ["--split", str(Path(_sp).absolute())]
    except Exception:  # noqa: BLE001 -- the held-out scan then records itself as not run
        pass
    from .quota_scope import subscription_env
    from .evolver_dispatch import scrub_parent_session_env
    from .claude_pins import pinned_env as _pinned_env, pin_record as _pin_record
    _env, _parent_scrubbed = scrub_parent_session_env(subscription_env(model))
    _env = _pinned_env(model, _env)
    _env["EVOSKILL_EVOLVER_DAYTONA_JSON"] = json.dumps(daytona_cfg)
    _pins = _pin_record(model, _env)
    t0 = time.time()
    dout = rd / f"e{epoch}_AO_DAYTONA.out"
    with _author_lock(rd, epoch):
        with open(dout, "wb") as fh:
            # Popen + wait, and SIGTERM on timeout (never SIGKILL): the driver's `finally` then still
            # deletes its sandbox.
            proc = subprocess.Popen(argv, cwd=str(repo), stdout=fh, stderr=subprocess.STDOUT, env=_env)
            try:
                proc.wait(timeout=_wait_s)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    pass
                raise ValueError(f"Daytona AO still running after {_wait_s}s; sent SIGTERM "
                                 f"(its sandbox is deleted by the driver); see {dout.name}")
            r = proc
    dt = time.time() - t0
    reply, why = final_reply(log) if Path(log).is_file() else ("", "no log")
    from .claude_pins import check_session as _check_session
    _mchk = _check_session(log, model) if Path(log).is_file() else {
        "resolved_model": None, "claude_code_version": None, "requested_model": model,
        "expected_model": model, "checked": False, "ok": False, "violations": ["no log"], "notes": [],
        "init_models": [], "assistant_models": {}, "subagent_models": {}, "model_usage": {},
        "claude_code_versions": []}
    conversation.record(rd, speaker="ao", channel="authoring", content=reply, epoch=epoch,
                        meta={"exit_code": r.returncode, "seconds": round(dt, 1), "stream_log": str(log),
                              "session_id": "", "source": why, "host": "daytona",
                              "wrote": out.name if out.is_file() else "(nothing)"})
    if r.returncode != 0:
        tail = dout.read_text(errors="replace")[-600:] if dout.is_file() else ""
        raise ValueError(f"Daytona AO driver exited {r.returncode} after {dt:.0f}s; see {dout.name}:\n{tail}")
    if not out.is_file():
        raise ValueError(f"Daytona AO exited 0 but no {out.name} was installed; see {dout.name}")
    body = out.read_text()
    if len(body) < 1500:
        raise ValueError(f"{out.name} is only {len(body)} bytes — treating as a failed authoring run")
    extra = {}
    if evidence_dirs is not None:
        extra["evidence_dirs"] = [str(Path(d).resolve()) for d in evidence_dirs]
    if policy_addendum is not None:
        extra.update(policy_addendum=str(policy_addendum), policy_addendum_sha256=add_sha256,
                     evolver_host="daytona")
    return {**extra, "bytes": len(body.encode()), "seconds": round(dt, 1), "model": model,
            "policy": str(pol), "policy_sha256": pol_sha256, "log": str(log),
            "sandboxed": True, "containment": "daytona", "ao_host": "daytona",
            "ao_daytona_receipt": str(rd / f"e{epoch}_AO_DAYTONA.json"), "ao_driver_out": str(dout),
            "parent_env_scrubbed": _parent_scrubbed,
            "sent_message": str(sent), "sent_message_chars": len(msg),
            "session_requested": False, "session_mode": "single-shot", "session_id": None,
            "session_argv": [], "session_cwd": "/root/evo/work",
            "pins": _pins, "resolved_model": _mchk["resolved_model"],
            "claude_code_version": _mchk["claude_code_version"],
            "model_check": {k: _mchk.get(k) for k in ("requested_model", "expected_model", "checked",
                                                       "ok", "violations", "notes", "init_models",
                                                       "assistant_models", "subagent_models",
                                                       "model_usage", "claude_code_versions")}}


def selftest() -> int:
    """The facts deriver must FAIL on a surface it cannot read, and must not emit zeros."""
    import shutil
    import tempfile
    print("SELFTEST — prompt_author\n")
    rc = 0
    td = Path(tempfile.mkdtemp())
    try:
        dg = td / "digests"
        dg.mkdir()

        def mk(name, a, b, kinds, terms, basis="DB", gold=3, miss=0):
            lines = [f"# {name}", "",
                     f"**Result: {a}/{b} samples passed.**  scores=[]  kind={kinds}  "
                     f"turns=[7, 9, 11]  duration_s=[]  termination={terms}", "",
                     "## What the task IS", "",
                     f"**reward_basis:** ['{basis}']", "",
                     f"**Golden actions ({gold}):**", "",
                     "## Did the agent retrieve the documents the answer depends on?", ""]
            for i in range(b):
                tail = "MISSED ['doc_x']" if i < miss else "missed none"
                lines.append(f"- **s{i}** (PASS): retrieved 1/1 required; 9 distinct docs seen; {tail}")
            lines += ["", "## Rollout s0", ""]
            (dg / f"{name}.md").write_text("\n".join(lines))

        mk("t1", 3, 3, "['pass','pass','pass']", "['user_stop','user_stop','user_stop']")
        mk("t2", 1, 3, "['pass','fail','ungraded(max_steps)']",
           "['user_stop','user_stop','max_steps']", miss=3)
        mk("t3", 0, 3, "['ungraded(max_steps)','ungraded(max_steps)','ungraded(max_steps)']",
           "['max_steps','max_steps','max_steps']", basis="ACTION", gold=9, miss=3)
        leg = td / "leg.json"
        leg.write_text(json.dumps({"per_task": {"t1": [1, 1, 1], "t2": [1, 0, 0], "t3": [0, 0, 0]}}))
        f = derive_facts(dg, leg)

        def chk(label, got, want):
            nonlocal rc
            ok = got == want
            print(f"  [{'ok' if ok else 'BAD'}] {label}: {got}" + ("" if ok else f" (want {want})"))
            rc |= 0 if ok else 1

        chk("episodes", f.n_episodes, 9)
        chk("passed", f.passed, 4)
        chk("never_graded", f.never_graded, 4)
        chk("graded_fail", f.graded_fail, 1)
        chk("reliable/unstable/never", (f.reliable, f.unstable, f.never_passes), (1, 1, 1))
        chk("tasks_with_ungraded", f.tasks_with_ungraded, 2)
        chk("tasks_all_ungraded", f.tasks_all_ungraded, 1)
        chk("retrieval samples missing", f.retrieval_samples_missing, 6)
        chk("retrieval tasks all missing", f.retrieval_tasks_all_missing, 2)
        chk("golden range", (f.golden_min, f.golden_max), (3, 9))
        chk("basis counts", f.basis_counts, {"DB": 2, "ACTION": 1})
        chk("coverage proof non-zero", (f.digests_seen, f.result_headers_parsed) > (0, 0), True)
        chk("triple reconciles", f.passed + f.graded_fail + f.never_graded, f.n_episodes)

        # A surface the extractor cannot read must RAISE, never return a zeroed Facts.
        bad = td / "bad"
        bad.mkdir()
        (bad / "x.md").write_text("# nothing the parser understands\n")
        try:
            derive_facts(bad, None)
            print("  [BAD] unreadable digests -> returned facts instead of raising")
            rc |= 1
        except ValueError as e:
            print(f"  [ok] unreadable digests -> raised ({str(e)[:60]}...)")

        # An empty surface must RAISE.
        try:
            derive_facts(td / "does_not_exist", None)
            print("  [BAD] missing digest dir -> no raise")
            rc |= 1
        except ValueError:
            print("  [ok] missing digest dir -> raised")

        # A leg json that disagrees with the digests must RAISE (0809c C-4.1).
        leg2 = td / "leg2.json"
        leg2.write_text(json.dumps({"per_task": {"t1": [1, 1, 1]}}))
        try:
            derive_facts(dg, leg2)
            print("  [BAD] digest/leg MISMATCH -> no raise")
            rc |= 1
        except ValueError as e:
            print(f"  [ok] digest/leg MISMATCH -> raised ({str(e)[:52]}...)")

        # The EVIDENCE INDEX (all-epochs mode only). Without it the block is unchanged.
        chk("no evidence -> no index, the block exactly as before",
            (f.evidence_index, f.as_block() == f._batch_block()), ([], True))
        from src.orchestrator import skill_library as _sl
        run = td / "run"
        e1, e2 = run / "digests_e1_train", run / "digests_e2_train"
        shutil.copytree(dg, e1)
        (run / "skills_v1" / "a").mkdir(parents=True)
        (run / "skills_v1" / "a" / "SKILL.md").write_text("---\nname: a\n---\nbody\n")
        lab, sha = _sl.snapshot(run / "skills_v1", e2)
        for p in sorted(dg.glob("*.md")):
            t = p.read_text().split("\n")
            (e2 / p.name).write_text("\n".join(t[:4] + [_sl.header_line(lab, sha), ""] + t[4:]))
        ev = [(1, e1, leg, None), (2, e2, leg, run / "skills_v1")]
        fi = derive_facts(e2, leg, evidence=ev)
        chk("index: base row then skills_v1 row",
            [(r.epoch, r.library) for r in fi.evidence_index], [(1, ""), (2, "skills_v1")])
        chk("index: each row's count re-derived from its own leg",
            [(r.passed, r.episodes, r.rate_checked) for r in fi.evidence_index],
            [(4, 9, True), (4, 9, True)])
        chk("index: rendered after the unchanged batch block",
            fi.as_block().startswith(fi._batch_block() + "\n\n## EVIDENCE INDEX"), True)
        shutil.rmtree(e2 / "_library")
        try:
            derive_facts(e2, leg, evidence=ev)
            print("  [BAD] an earlier surface without its library -> indexed anyway")
            rc |= 1
        except ValueError as e:
            print(f"  [ok] an earlier surface without its library -> raised ({str(e)[:52]}...)")
    finally:
        shutil.rmtree(td, ignore_errors=True)
    print("\n" + ("ALL FACTS CHECKS BEHAVED" if rc == 0 else "SELFTEST FAILED"))
    return rc


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
