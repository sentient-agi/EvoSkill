#!/usr/bin/env python3
"""AO-AUTHORED EVOLVER PROMPT — the orchestrator writes the evolver's user prompt itself.

WHY THIS EXISTS.

Until 2026-08-10 the operator (a human, or me) hand-wrote `ITER{N}_PROMPT.md` each epoch and then
countersigned it. That has two defects, and the second is worse than the first:

  1. THE OPERATOR GATE WAS HALF SELF-REVIEW. The step is named "prompt + skills proofread by a
     person". For the SKILLS that is a genuine adversarial check — the evolver wrote them, the
     operator verifies them. For the PROMPT the author and the countersigner were the same party,
     so that half of the gate reviewed its own work. Same shape as #248: a check that looks like
     verification while examining what it produced.

  2. PROMPT ASSEMBLY WAS NOT AN ABLATABLE VARIABLE. #218's finding is that 1 of 25+ pre-registered
     gate evaluations passed, i.e. roughly chance, and the conclusion owed is "change what is being
     optimised". The prompt is the largest lever in the system — but you cannot ablate a term a
     human re-tunes by hand every epoch. Making assembly a POLICY turns it into a config, and two
     policies are an experiment.

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

    THEN THE EXISTING GATES BITE. #208 re-derives the prompt's batch rate from the leg json, and
    check_prompt_no_leakage asserts zero held-out ids and no val figure. Those ran against a human
    author before; they now guard a model author, which is exactly what they were built for. If the
    author fabricates a number, #208 HALTs and nothing is dispatched.

WHAT THIS DELIBERATELY DOES NOT DO. It does not touch the evolver itself, the solver, the gate, or
the skills. It replaces one input — who writes the prompt — and leaves every verification in place.
"""
from __future__ import annotations

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
# PARSE THE HEADER, NOT THE BODY. On 2026-08-10 I re-derived a pass rate with a regex over the whole
# 52 KB body and got 12590/12946 — it was matching transaction amounts and tool output. The header is
# the artifact's own authoritative field. Look at the format before writing the matcher.
_RESULT = re.compile(r"\*\*Result:\s*(\d+)/(\d+)\s+samples passed\.\*\*")
_KIND = re.compile(r"kind=\[([^\]]*)\]")
# SSB v2 records step-budget exhaustion in its OWN header field. The reader knew tau3's
# `kind=[...max_steps...]` and `termination=[...]` only, so on an SSB v2 surface every match
# missed and the stats block reported "tasks losing >=1 sample to the step budget   0" and
# "termination reasons {}" — when 30 of 180 samples (16.7%) had in fact been TRUNCATED at the
# 45-turn ceiling. The AO transcribed those zeros into the evolver prompt faithfully; it was
# handed a false batch. Sixth site of this one assumption, after the digest glob, run_epoch's
# leg filenames, check_no_heldout_reachable's id matching, _run_leg's tag and the train-only
# guard. An empty parse must never render as a zero MEASUREMENT.
_EXH = re.compile(r"budget_exhausted=\[([^\]]*)\]")
_TERM = re.compile(r"termination=\[([^\]]*)\]")
_TURNS = re.compile(r"turns=\[([^\]]*)\]")
_BASIS = re.compile(r"\*\*reward_basis:\*\*\s*\[([^\]]*)\]")
_GOLD = re.compile(r"\*\*Golden actions \((\d+)\)")
_REQ = re.compile(r"\*\*required_documents \((\d+)\)\*\*")
# A real retrieval miss is written MISSED [...]; a clean sample says "missed none". The distinction
# is CASE. On 2026-08-10 a case-sensitive filter scored every genuine miss as clean and reported
# 0 affected tasks when the truth was 16 of 25.
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
    # facts block of all-zeros (#248's lesson, applied to my own extraction).
    digests_seen: int = 0
    result_headers_parsed: int = 0
    retrieval_lines_parsed: int = 0

    def as_block(self) -> str:
        """The verbatim FACTS the author may assert. Anything not here is not established."""
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
            # #312. THE ANNOTATION IS BENCHMARK-SPECIFIC AND WAS EMITTED UNCONDITIONALLY.
            # "still graded on whatever was saved, so they also appear above" is true of SSB v2,
            # where the grader scores whatever workbook sits in ./output/ at cutoff. It is FALSE
            # of tau3, where a max_steps episode produces no answer at all — which is why this
            # class's own `exhausted_episodes` comment says exhaustion and never-graded are the
            # SAME episodes there. On a tau3 batch the line therefore contradicted the
            # "NEVER GRADED ... no answer to grade" line two rows above it, inside one block the
            # author is told it may not question.
            #
            # FOUND BY ABLATION, 2026-08-13, and the way it surfaced is the lesson. Three authors
            # were given this identical block: Opus's prompt did not carry the claim; Sonnet's and
            # DSV4F's did. All three passed #208, #262 and the leakage check, because none of
            # those asks whether a TRANSCRIBED CLAIM IS TRUE — they ask whether a number
            # re-derives, whether the register is analytic, and whether an id leaked. A false
            # premise faithfully copied is invisible to all three.
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


def derive_facts(digest_dir: Path, leg_json: Path, corpus_dir: Path | None = None,
                 variant: str = "") -> Facts:
    """Compute the facts block. Raises rather than returning zeros on an empty surface."""
    f = Facts(variant=variant)
    files = sorted(digest_dir.glob("*.md"))
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
    return f



DEFAULT_POLICY = Path(__file__).resolve().parent / "PROMPT_POLICY.md"


def _prior_note(prior: Path | None) -> str:
    """#258. THE EVOLVER CANNOT OPEN THE PRIOR SKILL SET, so the prompt must CONTAIN it.

    The epoch-2 prompt said "There is a prior skill set of 3 skills at outputs/<run>/skills_v1 …
    Read them before you decide." That instruction is unfollowable: the evolver's grants are
    exactly {digests, staging, oracle}, and the #213 peer-skill hook denies every `skills_v*`
    read — including this run's OWN previous version, which I probed and confirmed. Granting the
    path would mean loosening the one guard that stops an evolver copying a peer arm's answers to
    the same train tasks, so the predecessor travels through the PROMPT instead.

    That is also the better artifact: the countersigned prompt hash then covers exactly what the
    evolver was shown about its predecessor, instead of a directory that could change underneath
    it. The author reads the bodies here (its own grants include the run dir) and is told to
    reproduce them; nothing downstream has to be relaxed.
    """
    if not (prior and prior.is_dir()):
        return "NO skill set exists yet. This is the first iteration; say so in the prompt."
    files = sorted(prior.glob("*/SKILL.md"))
    if not files:
        return "NO skill set exists yet. This is the first iteration; say so in the prompt."
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


def _ao_sandbox(out, digests, oracle, log, epoch: int):
    """(argv_prefix, profile_path_or_None) putting the AO under the SAME containment as the evolver.

    #267. The AO ran with `--dangerously-skip-permissions` and `--add-dir <run dir>` and NO
    sandbox, while the direct evolver has had sandbox-exec since #250. That asymmetry was
    invisible for as long as nothing trusted a file the AO could author. It stopped being
    invisible when ask() began reading ANSWERS_e{N}.json and _v_operator_signature reads
    e{N}_PROOFREAD.json: the AO runs at step index 1, BEFORE the operator gate at index 5, so an
    unsandboxed AO could pre-answer its own clarification questions and countersign its own
    prompt. Containment by prose in the author message ("Write that file and nothing else") is
    the #215 tripwire-not-boundary shape.

    WRITES are the narrow part and are granted as LITERALS: exactly ITER{N}_PROMPT.md and the
    log. Not their parent, which is the run dir — that mistake is #266, made and caught earlier
    today on this very profile.

    READS follow what the AO demonstrably needs, taken from ao4's own tool calls rather than
    guessed: the evidence surface, the train-only oracle, and the run's SPEC/RUNTIME. The prior
    epoch's prompt is granted so continuity-by-inspection keeps working. The repo's src/ and
    scripts/ come free from the shared builder, which is what let the AO verify the retriever
    against source — a capability worth keeping.

    IT REFUSES RATHER THAN DEGRADING (#286). This docstring used to END with the sentence
    "Degrades LOUDLY, never silently: if sandbox-exec is unavailable the caller is told and
    decides" — and the code then `return ([], None)`, recorded `containment: "NONE"` in a JSON
    field nobody reads, and dispatched the AO completely uncontained. On a host without
    sandbox-exec (any Linux, a stripped macOS image) the AO would run exactly as it did before
    #267: able to write ANSWERS_e{N}.json, whose whole purpose is to carry a HUMAN's answer, and
    e{N}_PROOFREAD.json, the operator countersignature — at step index 1, before the gate at
    index 5. A prose claim of loudness attached to a silent code path is the #215 shape
    (tripwire-not-boundary) inside the very function written to fix it.

    An explicit opt-out exists — EVOSKILL_AO_ALLOW_UNCONTAINED=1 — because a deliberate,
    recorded decision to run without containment is legitimate and an unrecorded one is not.
    """
    from .evolver_dispatch import EvolverJob, build_evolver_sandbox_profile, _sandbox_exec_available
    import tempfile as _tf
    if not _sandbox_exec_available():
        if os.environ.get("EVOSKILL_AO_ALLOW_UNCONTAINED") == "1":
            return ([], "UNCONTAINED-BY-EXPLICIT-OPT-OUT")
        raise ValueError(
            "sandbox-exec is unavailable, so the AO cannot be contained (#267/#286). It would run "
            "with write access to the run directory, which holds the operator's answer channel "
            "and the countersignature the AO's own work is gated by — and it runs BEFORE that "
            "gate. Refusing to dispatch.\n"
            "  To run anyway, as a recorded decision: EVOSKILL_AO_ALLOW_UNCONTAINED=1")
    rd = out.parent
    grants = [p for p in (digests, oracle) if p and Path(p).is_dir()]
    lits = [p for p in (rd / "SPEC.json", rd / "RUNTIME.json",
                        rd / f"ITER{epoch - 1}_PROMPT.md") if p.exists()]
    # #293. DENY THE THINGS BEING PROTECTED, NOT A PREFIX THEY HAPPEN TO SIT UNDER. The shared
    # builder denies writes by LOCATION (repo, home) and every real run dir lives under the repo,
    # so the AO's write containment looked total. A run dir anywhere else -- an external volume,
    # /tmp, a scratch mount -- matched neither prefix, `(allow default)` applied, and the AO could
    # write the operator's ANSWERS file, its own countersignature and every skills_v*. The whole
    # #267/#268 boundary evaporated silently, purely because of where the operator put the run.
    # Found by the first fixture that used a tempdir; every earlier probe had used a run dir under
    # the repo, so the positional assumption held by luck and the guard read as verified (#215's
    # shape: a check that passes for a reason other than the one it claims).
    job = EvolverJob(epoch=epoch, work_dir=rd, prompt="", model="", log_path=log,
                     grants=[Path(g).resolve() for g in grants], writes=[],
                     no_write=[rd.resolve(), operator_dir_for(rd)])
    prof = build_evolver_sandbox_profile(job)
    prof = prof.replace(f'(allow file-write* (subpath "{rd.resolve()}"))',
                        "\n".join([f'(allow file-write* (literal "{out.resolve()}"))']))
    # CHECK THE ARTIFACT, NOT THE RECIPE. Asserting the emitted bytes carry both denies is
    # stronger than asserting the builder was called correctly: if the builder ever stops
    # honouring `no_write`, this refuses to dispatch instead of quietly running uncontained.
    for _must in (rd.resolve(), operator_dir_for(rd)):
        if f'(deny file-write* (subpath "{_must}"))' not in prof:
            raise ValueError(f"the emitted AO sandbox profile does not deny writes to {_must}; "
                             f"refusing to dispatch with containment I cannot confirm (#293)")
    # #308. The literal grant admits only a direct open() of the exact path, but Claude
    # Code's Write tool writes ATOMICALLY — a temp file BESIDE the target (NAME.tmp.<x>,
    # .NAME.tmp) then rename — so the AO could never materialise its own artifact: probed
    # live under this exact profile (e2, 2026-08-12), a direct shell write to the literal
    # passed while both temp shapes EPERM'd. e1 only "worked" because the pre-#267 AO
    # escaped through unsandboxed bash. Admit exactly the artifact's own NAME FAMILY in
    # its own directory; answers/countersignature/staging stay denied by name, and no
    # consumer reads any of the extra names. Appended AFTER the denies because the last
    # matching rule wins in sandbox profiles.
    import re as _re
    prof += ('\n(allow file-write* (regex #"^'
             + _re.escape(str(out.parent.resolve()))
             + r'/\.?' + _re.escape(out.name) + r'(\.[^/]*)?$"))' + "\n")
    prof += "\n" + "\n".join(f'(allow file-read* (literal "{x.resolve()}"))' for x in lits) + "\n"
    prof += _own_transcript_grant(rd)
    fh = _tf.NamedTemporaryFile("w", suffix=".ao.sb", delete=False)
    fh.write(prof); fh.close()
    return (["sandbox-exec", "-f", fh.name], fh.name)


def operator_dir_for(run_dir):
    """The operator channel's directory for a run. Imported from loop rather than re-derived: two
    definitions of where the answer channel lives is exactly one too many (#293 denies it by name,
    and a second spelling of the name would deny the wrong directory)."""
    from .loop import operator_dir
    return operator_dir(run_dir).resolve()


def _own_transcript_grant(run_dir) -> str:
    """Read access to THIS RUN'S conversation, and no other — the #292 half of #291.

    The shared profile denies `~/.claude/projects` wholesale, because that tree holds every
    session transcript on the machine. That deny is correct and it broke `--resume`: the CLI
    resolves a session by READING `~/.claude/projects/<slug-of-realpath(cwd)>/<sid>.jsonl`, so
    epoch 2 got `No conversation found with session ID: ...` while the file sat on disk, present
    and unreadable. Epoch 1 passed (it only WRITES), which is why a one-shot probe would have
    called the containment good.

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
    return ("\n; #292 THIS RUN'S OWN conversation only, found by session id rather than by\n"
            "; reconstructing the CLI's slugging rule. Every other project stays denied.\n"
            + "\n".join(f'(allow file-read* (subpath "{h.parent}"))' for h in hits) + "\n")


def ao_session_path(run_dir) -> "Path":
    """Where the AO's session id lives, ONE PER RUN, not per epoch.

    Per-run is the point. A per-epoch id would give four unrelated sessions, which is exactly the
    state ao4 shipped in: `claude -p` with no session flags meant epoch N had no memory of epoch
    N-1, so the AO re-derived the benchmark from harness source every time (epoch 2 spent 18 of
    its 31 tool calls inside src/ and third_party/ locating top_k and the step budget). A run-level
    session lets the settled understanding survive an epoch boundary.
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

    #282: `if sp.is_file()` then `sp.write_text(...)` is check-then-write. Two concurrent first
    uses each saw no file, each minted a distinct uuid, each returned mode='new' — and the file
    kept one, so the other id went to a real `claude --session-id` whose conversation was then
    unreachable forever. Measured at 199/200 barrier-synchronised trials. O_EXCL makes the create
    the arbiter, and the loser reads the id the winner actually recorded.
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
                "cannot be resumed (#283).",
    }, indent=1)
    sp.parent.mkdir(parents=True, exist_ok=True)
    # #290. EXCLUSIVE **AND** ATOMIC. O_EXCL alone closed the double-mint but opened a second race
    # the same fixture found: create-then-write is two steps, so the loser read the winner's file
    # in the window between them, got an EMPTY file, and raised "AO_SESSION.json is unreadable —
    # refusing to lose the existing conversation" on a conversation that was being created
    # correctly right then. Write the content into a temp file first and publish it with
    # os.link(), which is atomic and still fails if the target exists: readers only ever see a
    # complete record, and only one caller can win.
    # UNIQUE PER CALLER, not per process. The first cut keyed the temp name on the pid alone, and
    # two THREADS in one process therefore shared it: thread A wrote its payload, thread B
    # overwrote the same temp file, and A's os.link published B's id while A returned its own as
    # "new" -- two live ids again, the very failure O_EXCL was added to prevent, reintroduced by
    # the scratch file. Caught by the #282 fixture at 5 of 6 trials.
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
    """Replace `path` with `obj` such that no reader can observe a partial file. See #290."""
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

    #289, and it is a bug I introduced with the #283 fix and caught with the #282 fixture. The
    first cut re-minted on `not confirmed and not _transcript_exists(sid)`, which DEFEATED THE
    O_EXCL MINT: the loser of a concurrent first-use read the winner's brand-new record, saw an id
    with no transcript (the winner had not dispatched yet), deleted the file and minted its own —
    reproducing the exact orphaned-conversation race the exclusive create had just closed.
    Measured: 1/1 barrier trials produced two ids again.

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
    silently start a fresh one and lose the conversation (#244 — a record written only on success
    is not a record).

    THREE THINGS THAT PERSIST-BEFORE-DISPATCH GETS WRONG ON ITS OWN, all found 2026-08-11:

      #283 A BURNED ID WEDGES THE RUN. The CLI creates the transcript only once it has accepted
           argv and started. Any failure before that — a bad flag, `claude` not on PATH, the
           binary refusing — leaves an id no conversation exists for, and there was no branch
           that ever minted again: every later epoch passed `--resume <dead id>` and failed
           identically, forever. Proved: `claude -p ... --session-id U --nonexistent-flag-zzz`
           exits 1 and creates ZERO files under ~/.claude/projects. So the record now carries
           `confirmed`, and an UNCONFIRMED id with no transcript on disk is re-minted rather
           than resumed. A CONFIRMED id is never re-minted — losing a real conversation is the
           worse error, and that is the whole reason the record is written early.

      #284 --resume IS CWD-BOUND AND THE RECORD DID NOT SAY SO. Transcripts are stored per
           realpath(cwd); `--resume <sid>` from any other directory exits 1 "No conversation
           found". The record held only {session_id, first_epoch} and was handed back regardless,
           so moving or renaming the run dir turned continuity into a hard failure with a
           misleading message. The mint cwd is recorded and a mismatch REFUSES with the actual
           directory to run from.

      #282 the mint race — see _mint_session.
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
        # #285: the try/except above wrapped only json.loads, so `[]`, `"abc"` or `3` — all valid
        # JSON — reached `d.get(...)` and raised AttributeError, escaping BOTH guarded refusals
        # and reaching the operator as a traceback from inside advance().
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
    """Mutual exclusion for AO dispatch within one run dir. See #287 at the call site.

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


def _author_user_prompt(policy_text: str, facts: Facts, out: Path, epoch: int,
                        digests: Path, oracle: Path, prior: Path | None, staging: Path) -> str:
    """The message the AUTHORING model receives. The policy is the how; the facts are the what."""
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


def final_reply(log: Path) -> tuple[str, str]:
    """The AO's closing message from a stream-json log — (text, where_it_came_from).

    Three sources, in descending fidelity, and the SECOND return value names which was used so a
    reader never has to guess whether they are looking at the model's words or at a description
    of its absence. A run that was killed has no `result` event at all, and reporting silence as
    an empty string would make "it said nothing" indistinguishable from "it never got to speak" —
    the same class of ambiguity as #288's session_requested-vs-session_mode.
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
                  prior_skills: Path | None = None, timeout_s: int = 1200,
                  work_dir: Path | None = None, continuous_session: bool = False) -> dict:
    """Dispatch the authoring model and RETURN WITHOUT TRUSTING IT.

    The contract is deliberately narrow: the model's only permitted side effect is creating `out`.
    We verify that the file appeared and is substantial; every claim INSIDE it is then checked by
    the ordinary epoch gates (#208 numbers, leakage, format), which is where a fabrication dies.
    """
    import subprocess
    import time
    pol = (policy or DEFAULT_POLICY)
    if not pol.is_file():
        raise ValueError(f"no prompt policy at {pol} — the AO cannot author a prompt without the "
                         f"how-to it is initialised with (RUNTIME.json 'prompt_policy')")
    if out.exists():
        raise ValueError(f"{out.name} already exists — refusing to overwrite an authored prompt. "
                         f"Move it aside if a re-author is intended.")
    msg = _author_user_prompt(pol.read_text(), facts, out, epoch, digests, oracle, prior_skills,
                              staging)
    work = work_dir or out.parent
    # PERSIST WHAT WE SEND. Until now the composed message existed only as argv, so the ONE
    # thing the operator hands the AO was the one thing no artifact recorded — the AO's own
    # stream-json log contains only tool_result `user` events, never the prompt. Auditing "what
    # did we ask it?" required re-deriving from source. Written before dispatch, so a crashed
    # authoring run still leaves the question behind.
    sent = out.parent / f"e{epoch}_AUTHOR_MESSAGE.md"
    sent.write_text(msg)
    # #301. And record it as a TURN, not only as a file. The file answers "what did we send?";
    # the transcript answers "what was the conversation?", which is the question the maintainer actually
    # asked and the one a per-artifact file cannot answer because it has no ordering with respect
    # to the AO's replies, its questions, or my answers. Recorded BEFORE dispatch for the #244
    # reason: a run that dies mid-authoring must still show what it was asked.
    conversation.record(out.parent, speaker="operator", channel="authoring", content=msg,
                        epoch=epoch,
                        meta={"model": model, "policy": pol.name, "sent_file": sent.name,
                              "chars": len(msg)})

    # #287. ONE AUTHORING DISPATCH PER RUN AT A TIME. `if out.exists()` above is the only other
    # guard and it is False for the entire ~1200s the first `claude -p` is authoring, so a second
    # --step inside that window ran `--resume <same id>` against a LIVE session. Verified against
    # the installed CLI: two concurrent --resume of one id FORK the transcript, and the next epoch
    # sees one branch — the other epoch's entire reasoning silently absent from the conversation
    # the run is supposed to be accumulating. The lock is held by the ORCHESTRATOR (the AO cannot
    # write here), and a lock whose owner is dead is stolen rather than wedging the run.
    with _author_lock(out.parent, epoch):
        sess_argv, sid, sess_mode = ([], None, "single-shot")
        if continuous_session:
            sess_argv, sid, sess_mode = _session_argv(out.parent, epoch, cwd=work)
        sb_argv, sb_prof = _ao_sandbox(out, digests, oracle, log, epoch)
        argv = [*sb_argv, "claude", "-p", msg, "--model", model, *sess_argv,
                "--add-dir", str(digests), "--add-dir", str(oracle), "--add-dir", str(out.parent),
                # UNDOCUMENTED flag (absent from --help; argparse rejects bad values, so it
                # IS registered). Verified working on 2.1.220 — but ONLY once the reasoning
                # is long enough to summarize: short thoughts return the block with empty
                # text, which reads as "flag inert" if probed with easy prompts (measured:
                # 973-output-token probe -> 1,064 chars; trivial probes -> 0). Do NOT gate
                # it behind a help-grep capability check; that check fails forever.
                "--thinking-display", "summarized",
                "--dangerously-skip-permissions", "--output-format", "stream-json", "--verbose"]
        # #307. FORCE THE AUTH PATH TO MATCH THE MODEL, rather than inheriting whatever the
        # launch shell had. Until 2026-08-13 this call passed no env at all, so an exported
        # ANTHROPIC_API_KEY silently moved the AO off the OAuth subscription and onto API
        # billing. The artifacts show exactly one session where that happened — ao6 e1
        # (apiKeySource='ANTHROPIC_API_KEY') against 'none' for every ao5 and ssbv2_ao1
        # authoring — so this is a real, observed drift between runs meant to be comparable,
        # not a hypothetical. For a gateway authoring model the env is returned untouched,
        # because such a model reaches its endpoint through precisely those variables.
        _env = subscription_env(model)
        t0 = time.time()
        with log.open("w") as fh:
            fh.write(f"=== prompt author: epoch {epoch} model {model} policy {pol.name}\n")
            fh.write(f"=== facts sha: {abs(hash(facts.as_block())) % (10 ** 12)}\n")
            fh.flush()
            r = subprocess.run(argv, cwd=str(work), stdout=fh, stderr=subprocess.STDOUT,
                               timeout=timeout_s, env=_env)
        dt = time.time() - t0
        if sid:
            # #283. Record which of the two happened, at the only moment either is knowable.
            (_confirm_session if r.returncode == 0 else _burn_session)(out.parent, sid)
    # THE AO'S SIDE OF THE EXCHANGE, recorded before the exit-code checks below so a FAILED
    # authoring run still leaves what the AO said. That is the half of #301 that was already
    # reaching Phoenix via the stream replay — but only for runs that completed, and never
    # ordered against the message that prompted it.
    reply, why = final_reply(log)
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
    if sb_prof:
        try: os.unlink(sb_prof)
        except OSError: pass
    return {"bytes": len(body.encode()), "seconds": round(dt, 1), "model": model,
            "policy": str(pol), "log": str(log),
            "sandboxed": bool(sb_argv),
            "containment": "sandbox-exec" if sb_argv else "NONE (explicit opt-out, #286)",
            "sent_message": str(sent), "sent_message_chars": len(msg),
            # #288: `continuous_session` is recorded as ASKED-FOR, separately from what was
            # actually WIRED. Without both, a receipt saying session_mode='single-shot' is
            # ambiguous between "this run does not use a continuous session" and "it does, and
            # the wiring silently dropped it" — and three surviving mutants lived in exactly that
            # ambiguity. _v_prompt_authored compares the two.
            "session_requested": bool(continuous_session),
            "session_mode": sess_mode, "session_id": sid,
            "session_argv": list(sess_argv), "session_cwd": str(Path(work).resolve())}


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
    finally:
        shutil.rmtree(td, ignore_errors=True)
    print("\n" + ("ALL FACTS CHECKS BEHAVED" if rc == 0 else "SELFTEST FAILED"))
    return rc


if __name__ == "__main__":
    import sys
    sys.exit(selftest())
