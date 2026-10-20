#!/usr/bin/env python3
"""Epoch invariants — the layer that decides whether an epoch MEASURED WHAT IT CLAIMS.

WHY THIS EXISTS, AND WHY IT IS THE CENTRE OF THE ORCHESTRATOR RATHER THAN A SIDE CHECK.

A step's own report can be green while its artifact is wrong:

    evolver "wrote 5 skills"        -> the loader reads 1; 4 bodies silently discarded
    evolver run "succeeded"         -> it read the comparator arm's answers
    process "launched"              -> `API Error: 400 No models provided`, 200ms
    split "built, 45 tasks"         -> most ids do not exist in the harness's pool

A happy-path DAG driver ships all four, because each step's own report is green. The only
thing that catches this class is checking the artifact THROUGH ITS REAL CONSUMER instead of
through the producer's claim. "The evolver says it wrote 5 files" and "the solver loads 5
skills" are different propositions, and only the second one decides what the experiment
measured.

So: every check here re-derives its answer from the thing itself. None of them read a
status field. A check that cannot fail is worse than no check, so each one is paired with a
known-bad fixture in `selftest()` and must fail on it.

USAGE
    python3 -m src.orchestrator.invariants --run-dir <run> --epoch 1 --split <split.json> \
        --base-train <run>/<run>_base_train.json
    python3 -m src.orchestrator.invariants --selftest

EXIT CODES: 0 all pass, 1 one or more FAIL, 2 could not evaluate (missing inputs).
A FAIL means STOP AND ESCALATE — it does not mean "retry".
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


# ── result type ──────────────────────────────────────────────────────────────────────
@dataclass
class Check:
    name: str
    status: str                      # PASS | FAIL | SKIP
    detail: str
    defect: str = ""                 # the failure class this check exists to catch
    evidence: dict = field(default_factory=dict)

    def line(self) -> str:
        mark = {"PASS": "ok  ", "FAIL": "FAIL", "SKIP": "skip"}[self.status]
        tag = f" [{self.defect}]" if self.defect else ""
        return f"  {mark}  {self.name}{tag}\n        {self.detail}"


def _ok(n, d, defect="", **ev):
    return Check(n, "PASS", d, defect, ev)


def _fail(n, d, defect="", **ev):
    return Check(n, "FAIL", d, defect, ev)


def _skip(n, d, defect=""):
    return Check(n, "SKIP", d, defect)


# ── the checks ───────────────────────────────────────────────────────────────────────
CONSUMER_CLAUDE_CODE = "claude_code"


def check_skills_load_through_consumer(staging: Path, consumer: str = CONSUMER_CLAUDE_CODE) -> Check:
    """Ask the skills' real consumer what the set holds, not the evolver's claim about it.

    The consumer is Claude Code, the solver EvoSkill ships. Claude Code's own skill loader is the
    canonical judge of a skill set, so this check re-implements none of its rules (frontmatter keys,
    descriptions, companion files, flat .md files in the root): it PASSes and records how many
    <name>/SKILL.md the set holds. A consumer without a shipped loader SKIPs, which a gate must read as
    "could not check", never as a pass.
    """
    n = "skills load through the real consumer"
    if not staging.is_dir():
        return _skip(n, f"no staging dir at {staging}", "skills not loaded")
    dirs = sorted(staging.glob("*/SKILL.md"))
    if consumer != CONSUMER_CLAUDE_CODE:
        return _skip(n, f"no skill loader for consumer {consumer!r} ships with EvoSkill (only "
                        f"{CONSUMER_CLAUDE_CODE!r})", "skills not loaded")
    return _ok(n, f"{len(dirs)} <name>/SKILL.md for Claude Code's own loader to read (the loop adds no rule)",
               "skills not loaded", written=[d.parent.name for d in dirs], consumer=consumer)


def check_leg_loaded_what_was_promoted(result_json: Path, staging: Path) -> Check:
    """The leg must have SEEN the skills, not merely been pointed at them.

    A leg that silently loads 1 of 6 still writes a complete-looking result JSON and reports
    a pass rate. That number then becomes the evolver's next-epoch batch rate.
    """
    n = "the leg loaded what was promoted"
    if not result_json.is_file():
        return _skip(n, f"no result json at {result_json.name}", "skills not loaded")
    d = json.loads(result_json.read_text())
    loaded = d.get("skills_loaded")
    if loaded is None:
        return _skip(n, "result json records no skills_loaded", "skills not loaded")
    want = len(list(staging.glob("*/SKILL.md"))) if staging.is_dir() else None
    if want is None:
        return _skip(n, "no staging dir to compare against", "skills not loaded")
    if len(loaded) != want:
        return _fail(n, f"{result_json.name} loaded {len(loaded)} skill(s) but {want} were "
                        f"promoted: {loaded}", "skills not loaded", loaded=loaded, promoted=want)
    return _ok(n, f"{result_json.name} loaded all {want}", "skills not loaded")


def _ci(pat: str):
    """Compile case-insensitively. NOT inline (?i) per alternative: Python 3.11+ raises
    'global flags not at the start of the expression', and this list is built by alternation.
    The fixtures caught it before it reached a live epoch — which is why the fixtures exist."""
    return re.compile(pat, re.IGNORECASE)


_ANALYTIC = [
    # (label, pattern) — each is a REGISTER the AO must not write in, not a banned word.
    ("dataset-structure claim", _ci(
        r"\b(?:misses|failures|required docs?|required documents?)\b[^.\n]{0,60}"
        r"\b(?:cluster|tend to|are (?:usually|typically|mostly))\b"
        r"|\btitle conventions?\b|\bnumbered famil(?:y|ies)\b"
        r"|\bthe corpus is (?:organis|organiz|arrang|structur)")),
    ("importance ranking", _ci(
        r"\b(?:highest[- ]value|most actionable|the single most (?:important|useful)"
        r"|worth more per|weight (?:your effort |them )?accordingly|matters? most"
        r"|the real finding|where the (?:real )?opportunity)\b")),
    ("prescribed remedy", _ci(
        r"\bwhat (?:such )?a rule in that position needs\b|\bthe (?:fix|remedy) is\b"
        r"|\bwhat it needs instead\b|\bsharper trigger\b")),
    ("diagnosis of a failure", _ci(
        r"\b(?:they|it|the agent) failed on\b|\bnone of them failed on\b"
        r"|\bthe (?:root )?cause (?:was|is)\b|\bthis explains\b"
        r"|\bwhat those .{0,30}have in common\b")),
    ("agenda-setting question", _ci(
        r"\bquestions? (?:a score cannot answer|the digests answer|to ask)\b"
        r"|\bthree questions\b|\bgo to the transcripts with questions\b")),
    ("evidence-strength verdict", _ci(
        r"\bthere is (?:enough|not enough) here\b|\benough to see (?:a )?patterns?\b"
        r"|\bis a real signal\b|\bdo not hedge\b|\bcalibrate honestly\b")),
]


def check_prompt_not_analytical(prompt: Path) -> Check:
    """THE AO PREPARES RAW INFORMATION; IT IS NOT PART OF THE OPTIMIZER.

    Prose in the policy does not hold this. A policy that invites the AO to reason about which prior
    rules failed yields prompts that assert theories about the benchmark corpus; even hedged as "test
    this yourself", such a theory sets the evolver's agenda, and its framing survives into skill
    bodies.

    So this is mechanical, in the same spirit as the prompt-figures check. It matches REGISTERS, not topics:
    an author may state that a never-graded episode produced no answer to grade (data semantics), and
    may not state that this bucket is where the opportunity lies (a ranking), that a group of tasks
    failed on when-to-stop (a diagnosis), or that misses cluster anywhere (a dataset theory).

    WHY THIS CAN FAIL LOUDLY RATHER THAN SILENTLY: an over-eager matcher here is cheap — it refuses
    an authoring run, costing ~6 minutes and no evolver spend, and the message names the offending
    sentence so the operator can judge. A missed analytical claim is expensive: it enters the
    optimizer and cannot be told apart from the intervention working.
    """
    n = "prompt is raw information, not analysis"
    if not prompt.is_file():
        return _skip(n, f"no prompt at {prompt.name}")
    body = prompt.read_text(errors="replace")
    # The prior skill set is reproduced VERBATIM inside the prompt and is not the author's prose;
    # judging it would flag the evolver's own past wording. Strip fenced blocks first.
    prose = re.sub(r"```.*?```", "", body, flags=re.S)
    hits = []
    for label, pat in _ANALYTIC:
        for m in pat.finditer(prose):
            s = prose[max(0, m.start() - 60):m.end() + 60].replace("\n", " ")
            hits.append(f"{label}: …{s.strip()}…")
    if hits:
        return _fail(n, f"{len(hits)} analytical passage(s) — the AO must hand over facts, not "
                        f"findings: " + " | ".join(hits[:4]), "analysis in prompt",
                     analytical_hits=len(hits))
    return _ok(n, f"no analytical register found ({len(prose)} chars of author prose scanned, "
                  f"{len(_ANALYTIC)} registers checked)", "analysis in prompt")


def held_out_figures(text: str) -> list:
    """The held-out FIGURE rule: a val/test/held-out keyword and an `N/M` or `N of M` within
    40 characters of each other on one line, IN EITHER ORDER: keyword-then-number ("val: 14/75")
    and number-then-keyword ("14 of 75 on val", the phrasing a fluent author is more likely to
    produce) both count. ONE definition: the prompt check below, and the AO sandbox's refusal to
    grant a file carrying such a figure (prompt_author._ao_sandbox), both call this."""
    _KW = r"(?:val|validation|test|held.?out)"
    _NUM = r"\d+\s*(?:/|\s+of\s+)\s*\d+"
    return (re.findall(rf"(?i)\b{_KW}\b[^.\n]{{0,40}}?\b{_NUM}", text)
            + re.findall(rf"(?i)\b{_NUM}\b[^.\n]{{0,40}}?\b{_KW}\b", text))


def id_mentions(body: str, ids, allowed=()) -> list:
    """The ids of `ids` that `body` names, in either spelling the SSB v2 surface uses (canonical
    'Cat/NN_MM' and the file-safe 'Cat_NN_MM' of every trace path and rollouts/ dir), as a whole token:
    not after and not before a word character. A spelling that is also an `allowed` id's spelling (a
    train id) is never looked for, so the file-safe form cannot turn a train id into a hit."""
    ok = {f for a in allowed for f in (str(a), str(a).replace("/", "_"))}
    hits = set()
    for i in ids:
        for f in {str(i), str(i).replace("/", "_")} - ok:
            if re.search(r"(?<![\w])" + re.escape(f) + r"(?![\w])", body):
                hits.add(str(i))
                break
    return sorted(hits)


def _split_ids(sp: dict, key: str) -> list:
    v = sp.get(key)
    if isinstance(v, dict):
        v = v.get("task_ids")
    return [x if isinstance(x, str) else x.get("id") for x in (v if isinstance(v, list) else [])]


def check_prompt_no_leakage(prompt: Path, split_path: Path) -> Check:
    """The evolver prompt must name NO held-out (val/test) task id, and no held-out figure.

    When a model authors the prompt, the check has to be mechanical: a model that has read the
    digests can very naturally write "task_027 shows..." and that id is train-visible but the habit
    is one keystroke from a held-out one.

    TWO THINGS, and the second is the one that actually protects the experiment:
      * NO held-out id, in either spelling (the canonical id and its file-safe `/`->`_` form, the one
        every trace path and rollouts/ dir uses). The split's dropped / unused / excluded_prior_run ids
        are refused by loop._v_prompt_no_leakage on top of this.
      * NO val/test count. Validation is the selection signal. An evolver told the val rate is being
        handed the very thing val exists to protect, and no later gate can undo that.

    TRAIN ids MAY appear: the rollouts layout tells the evolver each rollout's score in the prompt, one
    line per train task naming the task. Train ids are not a leak; held-out ids and held-out figures
    are.
    """
    n = "prompt names no held-out task id and no held-out figure"
    if not prompt.is_file():
        return _skip(n, f"no prompt at {prompt.name}", "held-out leak in prompt")
    if not split_path.is_file():
        return _skip(n, f"no split at {split_path} — cannot enumerate ids to look for", "held-out leak in prompt")
    sp = json.loads(split_path.read_text())
    train = _split_ids(sp, "train")
    held = sorted({i for k in ("val", "test") for i in _split_ids(sp, k)} - set(train))
    if not held:
        return _skip(n, f"split {split_path.name} lists no val/test ids", "held-out leak in prompt")
    body = prompt.read_text()
    hits = id_mentions(body, held, allowed=train)
    heldout = held_out_figures(body)
    if hits:
        return _fail(n, f"prompt names {len(hits)} held-out (val/test) task id(s) — {hits[:6]}. Train "
                        f"ids may be named; a held-out one never.", "held-out leak in prompt", ids=hits)
    if heldout:
        return _fail(n, f"prompt states a held-out figure: {heldout[:3]}. Validation is the "
                        f"selection signal; an evolver shown it is handed what val protects.",
                     "held-out leak in prompt", figures=heldout[:3])
    return _ok(n, f"0 of {len(held)} val/test ids present (either spelling; train ids may be named); no "
                  f"held-out figure ({len(body)} chars scanned)", "held-out leak in prompt", scanned=len(body),
               ids_checked=len(held))


def check_legs_paired_comparably(base_json: Path, skill_json: Path, want_samples=None) -> Check:
    """The two sides of a paired gate must be the SAME comparison.

    The gate pairs per episode and per task, so it is only meaningful if both legs ran the same
    tasks at the same samples-per-task. Every leg script defaults to `${EVOSKILL_N_SAMPLES:-3}`,
    so a leg the spec's value never reached runs N=3 whatever the spec says.

    WHY THIS CAN BE INVISIBLE. When spec.n_samples is ALSO 3, the spec and the script's default
    AGREE by coincidence, not by construction, and a check that only ran on that config could never
    fail. This one re-derives n from BOTH result JSONs and compares them to each other and to the
    spec, so the coincidence cannot hide the wiring.
    """
    n = "both legs are the same comparison"
    for p in (base_json, skill_json):
        if not p.is_file():
            return _skip(n, f"no result json at {p.name}", "unpaired comparison")
    b, s = (json.loads(p.read_text()) for p in (base_json, skill_json))
    bn, sn = b.get("n_samples"), s.get("n_samples")
    if bn is None or sn is None:
        return _skip(n, "a result json records no n_samples", "unpaired comparison")
    if bn != sn:
        return _fail(n, f"{base_json.name} ran N={bn} but {skill_json.name} ran N={sn} — the gate "
                        f"would pair {b.get('episodes')} base episodes against "
                        f"{s.get('episodes')} skill episodes. That is a different comparison, not "
                        f"a smaller one.", "unpaired comparison", base_n=bn, skill_n=sn)
    if want_samples is not None and bn != want_samples:
        return _fail(n, f"both legs ran N={bn} but the spec asks for n_samples={want_samples} — "
                        f"the spec's value is not reaching the leg script, whose own default won.",
                     "unpaired comparison", ran=bn, spec=want_samples)
    bt, st = set((b.get("per_task") or {})), set((s.get("per_task") or {}))
    if bt != st:
        only_b, only_s = sorted(bt - st)[:4], sorted(st - bt)[:4]
        return _fail(n, f"the two legs ran different task sets — {len(bt-st)} only in "
                        f"{base_json.name} ({only_b}), {len(st-bt)} only in {skill_json.name} "
                        f"({only_s})", "unpaired comparison")
    return _ok(n, f"both legs: {len(bt)} tasks x N={bn} = {b.get('episodes')} vs "
                  f"{s.get('episodes')} episodes, same task set", "unpaired comparison",
               n_samples=bn, tasks=len(bt))


def check_no_peer_artifacts_read(evolver_log: Path, own_run_dir: Path) -> Check:
    """A peer arm's evolved skills are that arm's ANSWERS to the same train tasks.

    Not held-out data — worse: reading them measures copying rather than evolution. The
    digests grant sits INSIDE the reference run's directory, so the peer skills are one
    `cd ..` from a legitimately granted path.
    """
    n = "no peer arm's skills were read"
    if not evolver_log.is_file():
        return _skip(n, f"no evolver log at {evolver_log.name}", "peer skills read")
    own = own_run_dir.name
    # "Is this path the run's OWN?" is not a SUBSTRING test (`own in p`): with own='ssbv2_selfevo' a
    # peer's 'outputs/ssbv2_selfevo_f2/skills_v1/...' would read as own. An absolute path is own only
    # when it is at or under the run dir's absolute path; a relative one only when `own` is a whole
    # path COMPONENT. Legitimate cross-epoch reads (all-epochs mode) widen the traffic a blind spot
    # like that would hide in.
    _own_abs = {str(own_run_dir.absolute()), str(own_run_dir.resolve())}

    def _is_own(p: str) -> bool:
        # Spelling is not ownership: '~' is expanded and '..' collapsed BEFORE
        # the comparison, so '<own>/digests_e1_train/../../<peer>/skills_v1' (a natural spelling
        # once every earlier digests dir is granted) is the peer's path, and
        # '~/.../outputs/<own>/skills_v1' is ours. A relative path that still climbs out
        # ('../...') after normalising has left the run, whatever it names next.
        p = os.path.normpath(os.path.expanduser(p))
        if p == ".." or p.startswith("../"):
            return False
        if p.startswith("/"):
            return any(p == a or p.startswith(a.rstrip("/") + "/") for a in _own_abs)
        return own in [c for c in p.split("/") if c]

    # Where peer runs live: the run dir's PARENT, and every sibling
    # under it. A `cd` that lands there (or climbs, or guesses with a glob) has left this run
    # whether or not it spells `outputs/`.
    _parents = {str(Path(a).parent) for a in _own_abs}

    def _peer_side(p: str) -> bool:
        return any(p == a or p.startswith(a.rstrip("/") + "/") for a in _parents)

    def _cd_outside(blob: str) -> list:
        """The `cd` targets in one command that leave this run, each resolved against the cd
        before it (so `cd <own>/digests_e1_train && cd ../../<peer>` is judged at <peer>). A
        relative target with no absolute cd before it is relative to an unknown cwd and is judged
        by its spelling, as before. Leaving = not own AND (under `outputs/`, at or under the run
        dir's parent, spelled with `..`, or spelled with a glob)."""
        out, base = [], None
        for t in re.findall(r"\bcd\s+[\"']?([\w./~*?\[\]-]+)", blob):
            te = os.path.expanduser(t)
            tgt = os.path.normpath(te if te.startswith("/") or base is None
                                   else os.path.join(base, te))
            base = tgt
            if _is_own(tgt):
                continue
            if ("outputs/" in tgt + "/" or _peer_side(tgt) or ".." in t.split("/")
                    or re.search(r"[*?\[]", t)):
                out.append(t)
        return out
    hits, denials = [], 0
    # ── THE SESSION GATE MUST NOT MAKE THE AUDIT VACUOUS ─────────────────────────────────────
    # Records are judged per session: `cur` binds on a `{"type":"system","subtype":"init"}` record
    # and every record of another session is skipped. Some launchers emit NO init record (their
    # only `system` subtype is `thinking_tokens`); with `cur` bound only by init, every record would
    # be skipped and the function would return "no attempt to read another run's skills" having
    # examined nothing.
    #
    # Defaulting `cur` alone would keep the same silent-degradation shape for the next format
    # change. An audit that cannot establish which session it is judging has NOT run, and must say
    # so. So: bind to the first session_id seen if no init record ever appears, AND record that the
    # fallback was used, so a format drift is visible instead of silent.
    _sessions_seen: list = []
    _init_seen = False
    _examined = 0
    cur = None
    for line in evolver_log.read_text(errors="ignore").splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        sid = d.get("session_id")
        if sid is not None and sid not in _sessions_seen:
            _sessions_seen.append(sid)
        if d.get("type") == "system" and d.get("subtype") == "init":
            _init_seen = True
            cur = sid
            hits, denials = [], 0          # only judge the LAST session in an appending log
        # FALLBACK: no init record has been seen yet, but records are arriving. Bind to the
        # LAST session id observed so far rather than dropping every record on the floor. In an
        # appending log this still converges on the newest session, because a later session's
        # first record rebinds it — the same intent the init branch had, without depending on a
        # record type that the actual launcher does not emit.
        if cur is None and sid is not None:
            cur = sid
            hits, denials = [], 0
        if sid != cur:
            continue
        # EVERY assistant record, INCLUDING SUBAGENTS (records carrying parent_tool_use_id). The
        # evolver dispatches several subagents concurrently, and a subagent reading a peer arm's
        # SKILL.md is precisely the contamination this check exists to catch.
        if d.get("type") == "assistant":
            for c in d.get("message", {}).get("content", []):
                if c.get("type") != "tool_use":
                    continue
                _examined += 1          # proof the audit actually looked
                i = c.get("input", {}) or {}
                blob = " ".join(str(i.get(k, "")) for k in
                                ("command", "file_path", "pattern", "path", "glob"))
                # ── JUDGE THE WHOLE COMMAND, NOT EACH TOKEN ─────────────────────────────────
                # A matcher that requires `outputs/` or `.claude/skills` INSIDE THE SAME TOKEN as
                # `skills` misses `cd <peer-run> && cat skills_v1/*/SKILL.md`: the token is the
                # bare `skills_v1/`, which carries neither. Those `cd`-then-relative forms are
                # exactly what a runtime hook blocks, and this audit is the backstop for that hook
                # regressing. So a bare `skills_v<N>` counts when the same command reaches outside
                # this run -- which is what a `cd` to a peer run dir, or any `outputs/<other-run>`
                # path, means.
                _outside = _cd_outside(blob) + [p for p in re.findall(r"outputs/[\w.-]+", blob)
                                                if not _is_own(p)]
                for m in re.findall(r"[\w./~-]*skills(?:_[\w.-]+)?[\w./~-]*", blob):
                    # OWNERSHIP IS JUDGED ON THE NORMALISED PATH, AND ONLY THERE. A `"staging" in m`
                    # exemption on the raw token would pass a peer's own `skills_v2_staging` and
                    # '<own>/skills_v3_staging/../../<peer>/skills_v1'. This run's own staging is own
                    # by path; nothing else is exempt.
                    if _is_own(m):
                        continue
                    if re.search(r"outputs/|\.claude/skills", m):
                        hits.append((c.get("name"), m))
                    elif os.path.expanduser(m).startswith("/") and not m.startswith("//"):
                        # An ABSOLUTE skills path that is not this run's, however it is spelled:
                        # `<root>/output?/<peer>/skills_v1` (a glob) leaves `outputs/` out of the
                        # token. ('//' is a URL remnant, not a path.)
                        hits.append((c.get("name"), m))
                    elif _outside and not m.startswith(("/", "~")):
                        # A RELATIVE skills path in a command that reached outside this run dir:
                        # `skills_v1/`, `skills_v1`, `../skills_v1/x/SKILL.md`, including
                        # `cd <peer>/digests_e2_train && cat ../skills_v1/...` -- one `..` past a
                        # legitimately granted path.
                        #
                        # This cannot over-block on the bare WORD "skills" in prose,
                        # because `_outside` only binds on a path under `outputs/` belonging to
                        # ANOTHER run, or a `cd` that leaves this run for where peer runs live
                        # (_cd_outside: its parent, a climbing `..`, a glob): the granted corpus
                        # lives under third_party/, and this run's own dirs are excluded by path.
                        # Reaching a peer run AND naming a skills path in one command is the
                        # contamination, in any spelling.
                        hits.append((c.get("name"), f"{m} (relative, after reaching {_outside[0]})"))
        if d.get("type") == "user":
            cc = d.get("message", {}).get("content")
            if isinstance(cc, list):
                for c in cc:
                    if c.get("type") == "tool_result" and "ANOTHER run" in str(c.get("content")):
                        denials += 1
    # A CLEAN VERDICT MUST PROVE IT LOOKED. An empty `hits` means either "examined every tool call
    # and found nothing" or "examined nothing at all". `_examined` distinguishes them, and a zero
    # count is a SKIP -- which this module's own contract converts into a HALT upstream.
    if _examined == 0:
        return _skip(n, f"parsed {len(_sessions_seen)} session(s) but examined 0 assistant "
                        f"tool-call records in {evolver_log.name} — the audit did not run. "
                        f"(init record seen: {_init_seen}.) A log this check cannot read must "
                        f"not report as a clean one.", "peer skills read")
    _how = "init record" if _init_seen else "first-session fallback (no init record emitted)"
    if hits and denials == 0:
        return _fail(n, f"{len(hits)} read(s) of another run's skills reached the tool with "
                        f"NO hook denial: {hits[:3]}", "peer skills read", hits=hits[:10])
    if hits:
        return _ok(n, f"{len(hits)} attempt(s) made, {denials} denial(s) fired — hook working",
                   "peer skills read", attempts=len(hits), denials=denials)
    return _ok(n, f"no attempt to read another run's skills "
                  f"({_examined} tool calls examined via {_how})",
               "peer skills read", denials=denials, examined=_examined)


def check_no_heldout_reachable(evolver_log: Path, split_file: Path) -> Check:
    """--add-dir does not confine Bash; the full task pool is one path away."""
    n = "no held-out task material reached the evolver"
    if not (evolver_log.is_file() and split_file.is_file()):
        return _skip(n, "missing evolver log or split", "held-out reachable")
    sp = json.loads(split_file.read_text())
    held = set(sp.get("val") or []) | set(sp.get("test") or [])
    if not held:
        return _skip(n, "split declares no held-out ids", "held-out reachable")
    txt = evolver_log.read_text(errors="ignore")
    # the FULL-POOL path is the high-signal one: a single read of it exposes every spec
    pool = re.findall(r"data/tau2/domains/[a-z_]+/tasks\.json", txt)
    if pool:
        return _fail(n, f"{len(pool)} reference(s) to the FULL task pool path — that file "
                        f"holds every held-out spec", "held-out reachable", pool=pool[:3])

    # BARE-INTEGER IDS. Flagging any id appearing as "{i}" / '{i}' / /{i}. anywhere in the transcript
    # fires, on a benchmark whose task ids are BARE INTEGERS, on the evolver's OWN todo list:
    #     "task":{"id":"1","subject":"Investigate regression: task 0 (exchange...)"}
    #     {"taskId":"3","status":"completed"}
    # The same small integers are TaskCreate handles and held-out benchmark tasks: two different
    # numbering systems, one regex.
    #
    # THE RULE, same as the digest builder and the oracle scrubber: only match where a match is
    # EVIDENCE OF A REFERENCE. A distinctive id (task_026) can only be a reference. A bare
    # integer cannot be distinguished from any other small number, so for those, match only the
    # shapes that MEAN a benchmark task -- a task-file path, or a task-labelled mention -- and
    # never a bare quoted number.
    distinctive = {i for i in held if len(i) >= 4 and not i.isdigit()}
    ambiguous = held - distinctive
    named = sorted(i for i in distinctive
                   if f'"{i}"' in txt or f"'{i}'" in txt or f"/{i}." in txt)
    for i in sorted(ambiguous):
        # `tasks/5.json`, `digests/5.md`, `task 5`, `task_5` -- shapes that name a BENCHMARK
        # task. Deliberately NOT a bare quoted "5", which is a false positive. Nor `tasks/` + digits
        # alone: `(?:tasks?[/_ ]|digests?/)18\b` fires on a PROSE FRACTION such as
        #     "`no-unprompted-extra-actions` never fires once across all 6 tasks/18 rollouts"
        # -- "6 tasks / 18 rollouts", a ratio, not a path.
        #
        # A reference to a benchmark task is either a FILE (tasks/18.json, digests/18.md) or a
        # LABEL (task 18, task_18). It is never `tasks/18 rollouts`. Require the extension for
        # the path form, and the singular `task` for the label form -- "6 tasks/18" cannot
        # match either.
        if re.search(rf"(?:tasks?/|digests?/){re.escape(i)}\.(?:json|md)\b", txt) or \
           re.search(rf"\btask[_ ]{re.escape(i)}\b", txt):
            named.append(i)
    if named:
        return _fail(n, f"{len(named)} held-out task id(s) appear in the evolver transcript: "
                        f"{sorted(named)[:5]}", "held-out reachable", ids=sorted(named)[:10])
    return _ok(n, f"0 of {len(held)} held-out ids present ({len(ambiguous)} bare-integer id(s) "
                  f"matched structurally, not as bare numbers); no full-pool reads", "held-out reachable")


def check_prompt_numbers_rederive(prompt: Path, result_json: Path,
                                  base_json: Path | None) -> Check:
    """Every 'N of M' in the prompt must come from the JSONs, not from an author.

    The existing shell gate matches prose patterns and cannot tell an UPTAKE claim from a
    PASS-RATE claim — it passes or fails on the word 'the'. This checks the pair that
    actually drives the evolver's reasoning: the batch rate and the base rate.
    """
    n = "prompt's headline numbers re-derive from the result JSONs"
    if not (prompt.is_file() and result_json.is_file()):
        return _skip(n, "missing prompt or result json", "prompt figures")
    txt = prompt.read_text()
    cur = json.loads(result_json.read_text())
    want_pass, want_tot = cur.get("pass"), cur.get("total")
    claims = set()
    for m in re.finditer(r"\*\*(\d+)\s+of\s+(\d+)\s+episodes\*\*", txt):
        claims.add((int(m.group(1)), int(m.group(2))))
    for m in re.finditer(r"scored\s+\*\*?(\d+)\s+of\s+(\d+)", txt):
        claims.add((int(m.group(1)), int(m.group(2))))
    if not claims:
        return _skip(n, "no headline 'N of M episodes' claim found to check", "prompt figures")
    if (want_pass, want_tot) not in claims:
        return _fail(n, f"prompt claims {sorted(claims)} but the leg scored "
                        f"{want_pass}/{want_tot}", "prompt figures",
                     claimed=sorted(claims), actual=[want_pass, want_tot])
    if base_json and base_json.is_file():
        b = json.loads(base_json.read_text())
        bp, bt = b.get("pass"), b.get("total")
        if re.search(r"scored\s+\d+\s+of\s+\d+\s+with no skills", txt):
            got = re.search(r"scored\s+(\d+)\s+of\s+(\d+)\s+with no skills", txt)
            if (int(got.group(1)), int(got.group(2))) != (bp, bt):
                return _fail(n, f"prompt's no-skills claim {got.group(1)}/{got.group(2)} != "
                                f"base {bp}/{bt}", "prompt figures")
    return _ok(n, f"headline {want_pass}/{want_tot} matches the leg", "prompt figures")


def check_outcome_kinds_sum(result_json: Path) -> Check:
    """pass + graded-fail + never-graded must equal the episode count.

    A never-graded episode carries a 0 that NO CHECK PRODUCED. Folding it into the failure
    count inflates the graded-failure rate and makes 'skills fixed wrong answers' look true
    when the gain actually came out of the never-graded bucket.
    """
    n = "pass / graded-fail / never-graded reconcile"
    if not result_json.is_file():
        return _skip(n, "no result json", "outcome buckets")
    d = json.loads(result_json.read_text())
    tot, ps = d.get("total"), d.get("pass")
    term = d.get("termination_reasons") or {}
    ungraded = int(term.get("max_steps", 0)) + int(term.get("context_overflow", 0))
    if tot is None or ps is None:
        return _skip(n, "result json lacks pass/total", "outcome buckets")
    gfail = tot - ps - ungraded
    if gfail < 0:
        return _fail(n, f"negative graded-fail: total={tot} pass={ps} ungraded={ungraded}",
                     "outcome buckets")
    return _ok(n, f"pass {ps} + graded-fail {gfail} + never-graded {ungraded} = {tot}",
               "outcome buckets", **{"pass": ps, "graded_fail": gfail, "never_graded": ungraded})


def check_no_infra_exclusions(result_json: Path) -> Check:
    """A leg that silently dropped episodes is not comparable to one that did not."""
    n = "no episodes were infra-excluded or timeout-zeroed"
    if not result_json.is_file():
        return _skip(n, "no result json")
    d = json.loads(result_json.read_text())
    ex = d.get("episodes_excluded_infra") or 0
    tz = d.get("episodes_timeout_scored_zero") or 0
    un = len(d.get("tasks_fully_unscored") or [])
    sc, di = d.get("episodes_scored"), d.get("episodes_dispatched")
    if ex or tz or un:
        return _fail(n, f"infra-excluded={ex} timeout-zeroed={tz} fully-unscored-tasks={un}",
                     evidence_scored=sc, dispatched=di)
    if sc is not None and di is not None and sc != di:
        return _fail(n, f"scored {sc} of {di} dispatched")
    return _ok(n, f"{sc}/{di} scored, 0 excluded, 0 timeout-zeroed")


# ── driver ───────────────────────────────────────────────────────────────────────────
def run_epoch(run_dir: Path, epoch: int, split_file: Path,
              base_train: Path | None, leg: str | None = None) -> list[Check]:
    """Every invariant for one epoch, re-derived from artifacts on disk.

    `leg` is the result-JSON filename pattern, with {epoch} and {key}; None = the loop's own naming,
    `<run dir name>_v{epoch}_{key}.json`. A pattern that does not match the run's files turns most
    checks into SKIPs -- "I could not check", never "nothing to check" -- and `assert_covered()` below
    turns a mostly-SKIPPED epoch into a hard error, so silence is never mistaken for a pass.
    """
    if leg is None:
        leg = f"{run_dir.name}_v{{epoch}}_{{key}}.json"
    v = f"v{epoch}"
    staging = run_dir / f"skills_{v}_staging"
    promoted = run_dir / f"skills_{v}"
    stag = promoted if promoted.is_dir() else staging
    tr = run_dir / leg.format(epoch=epoch, key="train")
    va = run_dir / leg.format(epoch=epoch, key="val")
    return [
        check_skills_load_through_consumer(stag),
        check_leg_loaded_what_was_promoted(tr, stag),
        check_leg_loaded_what_was_promoted(va, stag),
        check_no_peer_artifacts_read(run_dir / f"tau3_evolver_e{epoch}.log", run_dir),
        check_no_heldout_reachable(run_dir / f"tau3_evolver_e{epoch}.log", split_file),
        check_prompt_numbers_rederive(run_dir / f"ITER{epoch + 1}_PROMPT.md", tr, base_train),
        check_outcome_kinds_sum(tr),
        check_outcome_kinds_sum(va),
        check_no_infra_exclusions(tr),
        check_no_infra_exclusions(va),
    ]


def assert_covered(checks: list[Check], *, max_skip_frac: float = 0.5) -> Check:
    """Refuse to let a mostly-SKIPPED run read as clean.

    A SKIP is honest per-check ("this artifact is absent") and dishonest in aggregate: a
    caller scanning for FAIL sees none and concludes the epoch is sound. This converts that
    silence into a verdict. It is deliberately a CHECK and not an exception, so it appears in
    the same report as everything else and cannot be caught and swallowed upstream.
    """
    n = len(checks)
    skipped = [c.name for c in checks if c.status == "SKIP"]
    if n and len(skipped) / n > max_skip_frac:
        return _fail(
            "coverage: enough invariants actually ran",
            f"{len(skipped)}/{n} checks SKIPPED — this epoch is UNVERIFIED, not clean. "
            f"Usually a filename-pattern mismatch: pass leg=... to run_epoch(). "
            f"Skipped: {', '.join(sorted(set(skipped))[:4])}",
            "vacuous check")
    return _ok("coverage: enough invariants actually ran",
               f"{n - len(skipped)}/{n} checks executed against real artifacts", "vacuous check")


def selftest() -> int:
    """Every check must FAIL on a known-bad fixture. A guard that cannot fail is worse than none.

    Synthetic fixtures only, so coverage never depends on which run dirs exist on disk.
    """
    print("SELFTEST — each check must FAIL on a known-bad input\n")
    rc = 0
    import json as _json, shutil as _shutil, tempfile as _tempfile
    td = Path(_tempfile.mkdtemp())

    def _c(label, got, want):
        nonlocal_ok = got.status == want
        print(f"  [{'ok' if nonlocal_ok else 'BROKEN'}] {label} -> {got.status} (want {want})")
        print(f"        {got.detail[:150]}")
        return 0 if nonlocal_ok else 1

    try:
        sk = td / "skills_v1"; (sk / "a").mkdir(parents=True)
        (sk / "a" / "SKILL.md").write_text("---\nname: a\ndescription: when x\n---\nbody\n")
        leg = td / "leg.json"

        # The skills set, read by its consumer (Claude Code); a consumer with no shipped loader SKIPs.
        rc |= _c("skills set for Claude Code", check_skills_load_through_consumer(sk), "PASS")
        rc |= _c("consumer with no shipped loader", check_skills_load_through_consumer(sk, "other"), "SKIP")

        # Leg loaded what was promoted. NOTE the field: this check reads `skills_loaded` (the
        # LIST OF NAMES the solver's loader returned), whereas loop._v_leg_loaded_promoted reads
        # `skills_dir` (which directory was pointed at). Two same-named checks in two layers
        # asking different questions; both are worth asking, they are just not interchangeable.
        leg.write_text(_json.dumps({"skills_loaded": [], "pass": 1, "total": 3}))
        rc |= _c("leg loaded FEWER skills than promoted",
                 check_leg_loaded_what_was_promoted(leg, sk), "FAIL")
        leg.write_text(_json.dumps({"skills_loaded": ["a"], "pass": 1, "total": 3}))
        rc |= _c("leg loaded all promoted skills",
                 check_leg_loaded_what_was_promoted(leg, sk), "PASS")

        # outcome kinds must not exceed the total
        leg.write_text(_json.dumps({"pass": 3, "total": 3, "termination_reasons": {"max_steps": 2}}))
        rc |= _c("pass + never-graded exceeds total", check_outcome_kinds_sum(leg), "FAIL")
        leg.write_text(_json.dumps({"pass": 1, "total": 3, "termination_reasons": {"max_steps": 1}}))
        rc |= _c("buckets reconcile", check_outcome_kinds_sum(leg), "PASS")

        # infra exclusions
        leg.write_text(_json.dumps({"pass": 1, "total": 3, "episodes_excluded_infra": 2,
                                    "episodes_scored": 1}))
        rc |= _c("episodes infra-excluded", check_no_infra_exclusions(leg), "FAIL")
        leg.write_text(_json.dumps({"pass": 1, "total": 3, "episodes_excluded_infra": 0,
                                    "episodes_timeout_scored_zero": 0, "episodes_scored": 3}))
        rc |= _c("nothing excluded", check_no_infra_exclusions(leg), "PASS")

        # peer artifacts, INCLUDING the subagent path
        sp = td / "split.json"
        sp.write_text(_json.dumps({"train": ["1"], "val": ["9"], "test": []}))
        rd = td / "myrun"; rd.mkdir()
        log = rd / "e.log"
        init = {"type": "system", "subtype": "init", "session_id": "s"}
        def _read(path, parent=None):
            d = {"type": "assistant", "session_id": "s", "message": {"content": [
                {"type": "tool_use", "name": "Read", "input": {"file_path": path}}]}}
            if parent:
                d["parent_tool_use_id"] = parent
            return d
        peer = "/x/outputs/otherrun/skills_v3/s/SKILL.md"
        log.write_text("\n".join(_json.dumps(x) for x in [init, _read(peer)]))
        rc |= _c("MAIN-session read of a peer arm's skills",
                 check_no_peer_artifacts_read(log, rd), "FAIL")
        log.write_text("\n".join(_json.dumps(x) for x in [init, _read(peer, "toolu_sub")]))
        rc |= _c("SUBAGENT read of a peer arm's skills",
                 check_no_peer_artifacts_read(log, rd), "FAIL")
        log.write_text("\n".join(_json.dumps(x) for x in
                                 [init, _read(str(rd / "skills_v1_staging/a/SKILL.md"))]))
        rc |= _c("reading its OWN staging", check_no_peer_artifacts_read(log, rd), "PASS")

        # '..' is collapsed and '~' expanded BEFORE ownership is judged
        _ord = td / "outputs" / "ownrun"; _ord.mkdir(parents=True)
        _olog = _ord / "e.log"
        for _path, _want, _label in (
                (f"{_ord}/digests_e1_train/../../otherrun/skills_v1/a/SKILL.md", "FAIL",
                 "absolute '<own>/digests_e1_train/../../<peer>' is the PEER's"),
                ("outputs/ownrun/digests_e1_train/../../otherrun/skills_v1/a/SKILL.md", "FAIL",
                 "relative '<own>/digests_e1_train/../../<peer>' is the PEER's"),
                ("../outputs/ownrun/skills_v1/a/SKILL.md", "FAIL",
                 "a relative path that climbs out ('..') is not own"),
                ("~/outputs/ownrun/skills_v1/a/SKILL.md", "PASS",
                 "'~/.../outputs/<own>/skills_v1' IS own"),
                # 'staging' in the raw token does not exempt it
                (f"{_ord}/skills_v3_staging/../../otherrun/skills_v1/a/SKILL.md", "FAIL",
                 "'<own>/skills_v3_staging/../../<peer>' is the PEER's"),
                (f"{_ord.parent}/otherrun/skills_v2_staging/a/SKILL.md", "FAIL",
                 "a PEER's own staging is the peer's"),
                (f"{_ord}/skills_v3_staging/a/SKILL.md", "PASS", "this run's own staging"),
                (f"{td}/output?/otherrun/skills_v1/a/SKILL.md", "FAIL",
                 "an absolute peer path spelled with a glob")):
            _olog.write_text("\n".join(_json.dumps(x) for x in [init, _read(_path)]))
            _home = os.environ.get("HOME")
            os.environ["HOME"] = str(td)
            try:
                rc |= _c(_label, check_no_peer_artifacts_read(_olog, _ord), _want)
            finally:
                if _home is None:
                    os.environ.pop("HOME", None)
                else:
                    os.environ["HOME"] = _home

        # held-out reachability
        log.write_text('{"type":"x"}\n$ cat tasks/9.json\n')
        rc |= _c("held-out task spec read", check_no_heldout_reachable(log, sp), "FAIL")
        log.write_text('{"type":"x"}\n$ cat tasks/1.json\n')
        rc |= _c("only TRAIN specs read", check_no_heldout_reachable(log, sp), "PASS")

        # both legs must be the same comparison. FOUR cases: N differs, both legs agree with each
        # other but not with the spec (hidden whenever the spec's N equals the script default),
        # a differing task set, and the clean case.
        def _leg(p, nsamp, tasks, eps):
            p.write_text(_json.dumps({"n_samples": nsamp, "episodes": eps,
                                      "per_task": {t: {} for t in tasks}}))
            return p
        b1 = _leg(td / "b1.json", 1, ["t1", "t2", "t3"], 3)
        s3 = _leg(td / "s3.json", 3, ["t1", "t2", "t3"], 9)
        rc |= _c("base N=1 vs skill N=3",
                 check_legs_paired_comparably(b1, s3), "FAIL")
        s1 = _leg(td / "s1.json", 1, ["t1", "t2", "t3"], 3)
        rc |= _c("both N=1 but spec wants 3 (the wiring, hidden when 3==3)",
                 check_legs_paired_comparably(b1, s1, want_samples=3), "FAIL")
        sx = _leg(td / "sx.json", 1, ["t1", "t2", "t9"], 3)
        rc |= _c("same N, DIFFERENT task set", check_legs_paired_comparably(b1, sx), "FAIL")
        rc |= _c("same N, same tasks, matches spec",
                 check_legs_paired_comparably(b1, s1, want_samples=1), "PASS")

        # prompt leakage. The check must fire on an id AND on a held-out figure, and must not
        # false-positive on a prompt that only states counts — which is what a correct prompt does.
        spl = td / "split253.json"
        spl.write_text(_json.dumps({"train": ["task_004", "task_027"], "val": ["task_101"],
                                    "test": ["task_900"]}))
        pr = td / "P.md"
        pr.write_text("The agent ran 25 tasks x 3 samples = 75 episodes and scored 19 of 75.\n"
                      "4 tasks passed every sample; 18 episodes were never graded.\n")
        rc |= _c("prompt with COUNTS ONLY", check_prompt_no_leakage(pr, spl), "PASS")
        pr.write_text("Look at task_027, which failed all three times.\n")
        # a TRAIN id may be named; a held-out one never
        rc |= _c("prompt naming a TRAIN id", check_prompt_no_leakage(pr, spl), "PASS")
        pr.write_text("A held-out task like task_900 shows the same shape.\n")
        rc |= _c("prompt naming a TEST id", check_prompt_no_leakage(pr, spl), "FAIL")
        spl2 = td / "split253b.json"
        spl2.write_text(_json.dumps({"train": ["Cat/01_01"], "val": ["Cat/02_01"], "test": ["Cat/03_01"]}))
        pr.write_text("Trace: rollouts/Cat_01_01/s0.trace.jsonl and Cat/01_01.\n")
        rc |= _c("prompt naming a TRAIN id in both spellings", check_prompt_no_leakage(pr, spl2), "PASS")
        pr.write_text("Trace: rollouts/Cat_02_01/s0.trace.jsonl\n")
        rc |= _c("prompt naming a VAL id in its file-safe spelling", check_prompt_no_leakage(pr, spl2),
                 "FAIL")
        pr.write_text("It scored 19 of 75 on train and 14 of 75 on val.\n")
        rc |= _c("prompt stating a VAL figure", check_prompt_no_leakage(pr, spl), "FAIL")
        pr.write_text("It scored 19 of 75. Validation exists to protect against overfitting.\n")
        rc |= _c("prompt MENTIONING val with no figure", check_prompt_no_leakage(pr, spl), "PASS")

        # assert_covered: a mostly-skipped run must NOT read as clean
        skips = [_skip("a", "absent"), _skip("b", "absent"), _ok("c", "fine")]
        rc |= _c("2 of 3 checks SKIPPED", assert_covered(skips), "FAIL")
        rc |= _c("all checks executed", assert_covered([_ok("a", "x"), _ok("b", "y")]), "PASS")
    finally:
        _shutil.rmtree(td, ignore_errors=True)

    print(f"\n{'ALL CHECKS BEHAVED' if rc == 0 else 'SOMETHING MISBEHAVED'}")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir")
    ap.add_argument("--epoch", type=int)
    ap.add_argument("--split", help="the run's split file (required with --run-dir)")
    ap.add_argument("--base-train", help="the run's no-skills base train leg (required with --run-dir)")
    ap.add_argument("--leg", default=None,
                    help="result-JSON filename pattern with {epoch} and {key}; default <run>_v{epoch}_{key}.json")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not (a.run_dir and a.epoch and a.split and a.base_train):
        ap.error("--run-dir, --epoch, --split and --base-train are required unless --selftest")

    rd = Path(a.run_dir)
    if not rd.is_absolute():
        rd = REPO / rd
    sp = Path(a.split)
    if not sp.is_absolute():
        sp = REPO / sp
    bt = Path(a.base_train)
    if not bt.is_absolute():
        bt = REPO / bt

    checks = run_epoch(rd, a.epoch, sp, bt, leg=a.leg)
    print(f"EPOCH INVARIANTS — {rd.name} epoch {a.epoch}\n")
    for c in checks:
        print(c.line())
    n_fail = sum(1 for c in checks if c.status == "FAIL")
    n_skip = sum(1 for c in checks if c.status == "SKIP")
    n_pass = sum(1 for c in checks if c.status == "PASS")
    print(f"\n  {n_pass} pass, {n_fail} FAIL, {n_skip} skip")
    if n_fail:
        print("\n  STOP. A failing invariant means the epoch did not measure what it claims.")
        print("  Do not retry the step — diagnose it. Retrying re-runs the same wrong thing.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
