#!/usr/bin/env python3
"""Turn tau3 banking_knowledge rollouts into one readable digest per task.

    EVOSKILL_TAU3_VARIANT=bm25_grep \
    EVOSKILL_TAU3_DIGEST_SPLIT_KEY=train \
    EVOSKILL_TAU3_DIGEST_OUT=outputs/tau3_banking/digests_e0_train \
      python3 scripts/tau3_build_digests.py

This is the evolver's ENTIRE evidence surface. Everything it can know about how
the solver behaved, it knows from these files — so every decision here is a
decision about what the optimizer is allowed to see, and the guards below exist
because each one has already been violated once.

WHY A TAU3-SPECIFIC BUILDER RATHER THAN `build_rollout_digests.py`. Three
structural differences, none cosmetic:

  1. THE TRANSCRIPT IS A PRE-RENDERED STRING, NOT A MESSAGE LIST. tau3 traces
     arrive with `messages == []` and the turn-by-turn text in
     `cached_transcript` (written at cache.set() time; see #149). So there are no
     typed blocks to walk — the parsing happens on text.

  2. THE TOOL RESULTS ARE MOSTLY POLICY DOCUMENTS. The retrieval tool returns
     ten full documents per call. Measured over 55 rollouts: 1016 tool results,
     8.5 MB, p50 = 7.7 KB each. Handed over verbatim they are ~90% of the
     corpus and they crowd out the thing an analyst actually needs, which is the
     agent's reasoning. They are collapsed here to their RANKED LIST — rank,
     title, doc id, BM25 score — with the bodies elided and labelled as elided.
     This is the same trade as the SSB builder's skill-echo collapse and it is
     justified the same way: the solver saw the full bodies, the evolver can
     read the full bodies in the corpus, and what the evolver needs from the
     transcript is WHICH documents came back at WHAT rank.

  3. THE GRADE IS A STRUCTURED VERDICT, NOT A DIFF. tau2's evaluator returns
     per-action `action_match`, per-item communicate checks, nl-assertion
     judgements and a `reward_basis` naming which components multiply into the
     final reward. That breakdown is authoritative and free, exactly as the SSB
     scorer's DIFF.txt is, so it is rendered per sample rather than left for
     the reader to infer from the transcript.

THE `required_documents` DIAGNOSIS. Each task's golden spec names the documents
a correct answer depends on. Whether the agent ever retrieved them is the single
most useful discriminator this benchmark offers — "wrong because it never found
the policy" and "wrong although it read the policy" are different failures with
different remedies, and no other field distinguishes them. So the digest
computes it: required, retrieved, and the set difference, per sample.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# DOMAIN parameterised (2026-08-06), matching run_tau3_banking.py's own env var so a
# leg and its digests cannot be built for different task types. It was a literal, which
# made this builder — the evolver's ENTIRE evidence surface — banking-only.
_DOMAINS = ("banking_knowledge", "airline", "retail", "telecom")
DOMAIN = os.environ.get("EVOSKILL_TAU3_DOMAIN", "banking_knowledge")
if DOMAIN not in _DOMAINS:
    raise SystemExit(f"REFUSING: unknown EVOSKILL_TAU3_DOMAIN {DOMAIN!r}; "
                     f"known {list(_DOMAINS)}")
_RETRIEVAL_DOMAIN = "banking_knowledge"

VARIANT = os.environ.get("EVOSKILL_TAU3_VARIANT") or ""
# Must match run_tau3_banking.py's token EXACTLY or the derived cache-dir names point at
# buckets that do not exist and the builder reports an empty leg.
_VTOK = VARIANT or "noretrieval"
N_SAMPLES = int(os.environ.get("EVOSKILL_N_SAMPLES", "3"))
# REFUSES rather than defaulting (#216). Every other identity-determining input in this
# file refuses to guess — split file, split key, variant, bucket, leg — because guessing
# any of them silently produces a digest set that describes a DIFFERENT experiment. The
# output path was the one exception, defaulting to a hardcoded `outputs/tau3_banking/`
# regardless of which run dir the caller passed. That is exactly #190's shape (OUT
# hardcoded while RUN_DIR was honoured), which destroyed banking's base baselines when
# two domains wrote the same tag. An evidence surface built into the wrong run dir is
# worse than a crash: the epoch proceeds, on another arm's evidence.
_OUT_ENV = os.environ.get("EVOSKILL_TAU3_DIGEST_OUT", "").strip()
if not _OUT_ENV:
    raise SystemExit(
        "REFUSING: EVOSKILL_TAU3_DIGEST_OUT is unset.\n"
        "  This names the directory the evidence surface is written to, and it is not\n"
        "  derivable from the other inputs — a run dir can hold digests_e1_train,\n"
        "  digests_e2_train, ... and only the caller knows which epoch this is.\n"
        "  Defaulting would write one epoch's evidence over another's, or into a\n"
        "  different arm's run dir entirely (#190). Pass it explicitly, e.g.\n"
        "    EVOSKILL_TAU3_DIGEST_OUT=outputs/tau3_dsv4fevo/digests_e3_train")
OUT = Path(_OUT_ENV)
_SPLIT_FILE_ENV = os.environ.get("EVOSKILL_TAU3_DIGEST_SPLIT_FILE", "")
SPLIT_FILE = Path(_SPLIT_FILE_ENV) if _SPLIT_FILE_ENV else None
SPLIT_KEY = os.environ.get("EVOSKILL_TAU3_DIGEST_SPLIT_KEY", "")
# Per-domain oracle dir: make_tau3_oracle.py writes banking to `oracle/` and every other
# domain to `oracle_<domain>/`, so each evolver's evidence surface is exactly its own.
_ORC = "oracle" if DOMAIN == _RETRIEVAL_DOMAIN else f"oracle_{DOMAIN}"
ORACLE_TASKS = Path(os.environ.get(
    "EVOSKILL_TAU3_ORACLE_TASKS",
    ROOT / f"outputs/tau3_banking/{_ORC}/tasks/{DOMAIN}.json"))
BUCKET = os.environ.get("EVOSKILL_TAU3_DIGEST_BUCKET", "")
# LEG IDENTITY, asserted against evidence a record cannot fake. See the guard at
# the bottom of this file for why the obvious check (one tree_hash per bucket)
# is worthless.
LEG = os.environ.get("EVOSKILL_TAU3_DIGEST_LEG", "").strip().lower()
if LEG and LEG not in ("base", "skill"):
    raise SystemExit("REFUSING: EVOSKILL_TAU3_DIGEST_LEG must be 'base' or 'skill'")
# Explicit cache dirs override the derived per-sample names, for digesting a leg
# whose dirs were named by hand.
CACHE_DIRS_ENV = os.environ.get("EVOSKILL_TAU3_DIGEST_CACHES", "")

# --- PRIOR LEG, so the evolver can see what its OWN last version BROKE ----------------
# Until 2026-08-07 a digest carried exactly ONE Result line: the leg being digested. From
# epoch 2 onward that makes regressions structurally invisible -- the evolver reads
# "0/3 passed" and cannot tell "the skills I wrote last epoch BROKE this task" from "this
# task was already failing and my skills did not help". Those two call for opposite edits:
# the first means REMOVE or narrow a rule, the second means add one. Epoch 1 on the 20/20
# banking split broke task_064 and task_099 while fixing task_060, and none of that reached
# the epoch-2 surface.
#
# Cheap because every leg already writes `per_task` = {task_id: [score, ...]} into its
# result JSON, so this is a read of an artifact that exists, not new instrumentation.
_PRIOR_ENV = os.environ.get("EVOSKILL_TAU3_DIGEST_PRIOR", "")
PRIOR_PATH = Path(_PRIOR_ENV) if _PRIOR_ENV else None
PRIOR: dict[str, list] = {}
PRIOR_TAG = ""
if PRIOR_PATH:
    if not PRIOR_PATH.exists():
        raise SystemExit(f"REFUSING: EVOSKILL_TAU3_DIGEST_PRIOR={PRIOR_PATH} does not exist. "
                         f"A silently-absent prior would rebuild the regression-blind surface "
                         f"this flag exists to fix.")
    _pj = json.loads(PRIOR_PATH.read_text())
    PRIOR = _pj.get("per_task") or {}
    PRIOR_TAG = _pj.get("tag") or PRIOR_PATH.stem
    if not PRIOR:
        raise SystemExit(f"REFUSING: {PRIOR_PATH} has no per_task block, so no prior outcome "
                         f"can be rendered. Do not fall back to a prior-free surface silently.")
    # A prior from a different split or domain would produce meaningless deltas.
    if _pj.get("domain") and _pj["domain"] != DOMAIN:
        raise SystemExit(f"REFUSING: prior leg is domain {_pj['domain']!r}, this run is {DOMAIN!r}")

def prior_block(tid: str, cur_scores: list) -> list[str]:
    """Render the prior-vs-current comparison, naming a regression as a regression."""
    if not PRIOR:
        return []
    pv = PRIOR.get(tid)
    if pv is None:
        return [f"**Prior leg `{PRIOR_TAG}`: this task was NOT in that leg.** No delta is "
                f"computable; treat the result above as a first measurement.", ""]
    pn = sum(1 for s in pv if isinstance(s, (int, float)) and s >= 0.999)
    cn = sum(1 for s in cur_scores if isinstance(s, (int, float)) and s >= 0.999)
    if cn < pn:
        tag = ("**DELTA: REGRESSION.** The previously-active skills made this task WORSE: "
               f"{pn}/{len(pv)} samples passed before, {cn}/{len(cur_scores)} pass now. "
               "A rule that is currently firing on this task is HARMFUL here. Prefer "
               "narrowing its applicability condition or removing it over adding another rule.")
    elif cn > pn:
        tag = (f"**DELTA: IMPROVED** ({pn}/{len(pv)} -> {cn}/{len(cur_scores)}). Whatever is "
               "firing here is working; do not disturb it.")
    else:
        tag = (f"**DELTA: UNCHANGED** ({pn}/{len(pv)} both times). Note this is unchanged in "
               "COUNT; the per-sample pattern below may still differ.")
    return [f"**Prior leg `{PRIOR_TAG}`: {pn}/{len(pv)} samples passed.**  "
            f"prior_scores={pv}", tag, ""]

# Non-retrieval tool results (DB record dumps, tool acknowledgements) are the
# data the agent actually reasons over, so they are kept — but a
# `get_credit_card_transactions_by_user` on a busy account is thousands of
# lines, and past that point it stops informing and starts burying. Head+tail so
# a long dump never hides its own ending.
TOOL_HEAD, TOOL_TAIL = 2000, 800
# Retrieved document bodies are elided to this much, purely so the reader can
# tell WHICH document it is without scrolling. The full text is in the corpus.
# Measured on a real 27-turn rollout: the collapsed retrieval blocks are 71% of
# the digest even AFTER collapsing, because ~40 retrieval calls x 10 documents
# means 400 previews. At 200 chars each that is 80 KB per rollout of text whose
# job the TITLE already does. 120 keeps the "is this the right document" signal
# and puts a tau3 digest in the same size class as the SSB v2 digests the 0801
# evolvers read (mean 88 KB, max 194 KB).
DOC_BODY_KEEP = int(os.environ.get("EVOSKILL_TAU3_DOC_BODY_KEEP", "120"))


def refuse(msg: str) -> None:
    raise SystemExit(f"REFUSING: {msg}")


def clip(s: str, head: int = TOOL_HEAD, tail: int = TOOL_TAIL) -> str:
    s = s or ""
    if len(s) <= head + tail:
        return s
    return (f"{s[:head]}\n    ...[{len(s) - head - tail:,} chars elided from the "
            f"MIDDLE of this tool result]...\n{s[-tail:]}")


# --- the retrieval-result collapse -----------------------------------------
# Anchored on the ID+Score pair, NOT on the "N. " prefix alone. Document BODIES
# contain their own numbered lists ("1. Verify customer identity", "2. Explain
# that ..."), so a bare `^\d+\. ` parser reads a policy's steps as retrieval
# hits — it found 12 "documents" in a block that held 9. The three-line header
# shape is unambiguous.
# `^[ \t]*` and not a bare `^`: the renderer puts the FIRST document on the same
# line as the `(tool.result):` marker, so a body that begins " 1. Title" has its
# rank preceded by a space. A bare `^\d+\.` silently dropped rank 1 from EVERY
# retrieval result — the top hit, the one the agent is most likely to have acted
# on — and dropped it from the required-documents diagnosis too, which would have
# under-reported retrieval recall on every rollout. Caught by noticing a header
# that said "9 documents" above a list numbered 2-10.
_DOC_RE = re.compile(
    r"(?m)^[ \t]*(?P<rank>\d+)\. (?P<title>.*)\n"
    r"[ \t]+ID: (?P<id>\S+)\n"
    r"[ \t]+Score: (?P<score>[-\d.]+)\n"
    r"[ \t]+Content: ",
)
_TIMING_RE = re.compile(r"\[Timing:[^\]]*\]")
_SKILL_ECHO_RE = re.compile(r"\A\s*---\s*\nname:\s*(?P<name>[^\n]+)\n")
# The skill TOOL CALL, which is present whether or not the load succeeded. Used
# for leg identity because it cannot be produced by a base rollout at all: a base
# solver is never given the tool.
_SKILL_CALL_RE = re.compile(r"(?m)^\(tool\.skill\):")


def collapse_retrieval(body: str, required: set[str] | None = None) -> tuple[str, list[str]]:
    """(rendered, doc_ids). Returns ("", []) when this is not a ranked list.

    `required` marks the task's required_documents inline. That marker is the
    cheapest high-value annotation in the whole digest: it turns every retrieval
    result into a local answer to "did the document this task depends on come
    back, and at what rank" — without it the reader has to hold five doc ids in
    their head while scrolling forty retrieval calls.
    """
    required = required or set()
    hits = list(_DOC_RE.finditer(body))
    if not hits:
        return "", []
    lines: list[str] = []
    ids: list[str] = []
    ranks: list[int] = []
    for i, m in enumerate(hits):
        start = m.end()
        end = hits[i + 1].start() if i + 1 < len(hits) else len(body)
        content = body[start:end].rstrip()
        ids.append(m.group("id"))
        ranks.append(int(m.group("rank")))
        keep = content[:DOC_BODY_KEEP].replace("\n", " ").strip()
        flag = " **REQUIRED-BY-THIS-TASK**" if m.group("id") in required else ""
        lines.append(
            f"  {m.group('rank')}. {m.group('title').strip()}  "
            f"[{m.group('id')}  score={m.group('score')}]{flag}\n"
            f"       body {len(content):,} chars, first {min(len(content), DOC_BODY_KEEP)}: "
            f"{keep}…")
    timing = " ".join(_TIMING_RE.findall(body))
    total = len(body)
    head = (f"[RETRIEVAL RESULT — {len(hits)} document(s) returned, {total:,} chars. "
            f"THE SOLVER SAW EVERY BODY IN FULL; they are elided here only to keep "
            f"this digest readable. Read any of them in the corpus at "
            f"third_party/tau3-bench/data/tau2/domains/{DOMAIN}/documents/. "
            f"Ranks and scores below are the retriever's, verbatim.]")
    # A retriever returns ranks 1..k. Anything else means this parse lost a
    # document — either the parser is wrong (it was, once) or the stored result
    # was clipped mid-header. Say so IN the digest: a silently short ranked list
    # reads as "the retriever returned less", which is a different finding.
    if ranks != list(range(1, len(ranks) + 1)):
        head += (f"\n  [PARSE WARNING: recovered ranks {ranks} are not 1..{len(ranks)}. "
                 f"At least one document header was not parsed — do NOT read this "
                 f"list as the complete result set.]")
    tail = f"\n  {timing}" if timing else ""
    return head + "\n" + "\n".join(lines) + tail, ids


def collapse_skill_echo(body: str) -> str:
    """Replace a SKILL.md echo with a one-line marker, or "" if not one.

    Every skill load re-emits the whole SKILL.md as a tool result, so an evolver
    reading raw transcripts reads its OWN previous-epoch output back and can
    mistake it for evidence about the solver. Worse, clipped skill text shows a
    truncated mid-sentence version of a file the evolver has in full in its
    staging dir, which invites reasoning about text that was never in the
    solver's context in that form. Same call as the SSB builder's
    `_collapse_skill_echo`, same justification.

    WHAT IS PRESERVED: that the skill loaded, which one, and how big it was —
    which is the part needed to attribute behaviour to a rule.
    """
    m = _SKILL_ECHO_RE.match(body)
    if not m:
        return ""
    return (f"[SKILL LOADED: {m.group('name').strip()} — {len(body.strip()):,} B "
            f"echoed back here. Body omitted: it is the file in your staging dir, "
            f"so read it there rather than from this echo.]")


# A segment boundary in the rendered transcript. `### Turn N` / `### User Query`
# are anchored to a whole line so a policy body's own "### Something" heading
# cannot split a tool result.
_SEG_RE = re.compile(
    r"(?m)^(?:### Turn \d+\s*$|### User Query\s*$|text: |\(thinking\)|\(tool\.)")


# The shared renderer labels EVERY `role == "user"` message `### User Query`
# (src/harness/agent.py). In the single-agent arms there is exactly one such
# message — the task prompt — so the label is accurate there. In tau2/tau3 the
# "user" is a live CUSTOMER who speaks on nearly every turn, so that heading
# repeats down the file (864 times across this batch) and reads like a fresh task
# statement each time. Relabelled HERE, at digest time, rather than in the shared
# renderer: re-labelling upstream would change what every other arm's transcripts
# look like, and this is a tau3 presentation concern. Fixing the label is also
# strictly better than explaining it in the evolver prompt — the same reason the
# outcome `kind` is a computed field instead of a paragraph.
_CUSTOMER_HEADING = ("### Customer (the simulated user speaking — a conversational "
                     "turn, not a task statement)")


def relabel_speakers(ct: str) -> tuple[str, int]:
    """Rename the customer's heading so the two speakers are distinguishable."""
    if not ct:
        return ct, 0
    out, n = re.subn(r"(?m)^### User Query$", _CUSTOMER_HEADING, ct)
    return out, n


def rewrite_transcript(ct: str, required: set[str] | None = None) -> tuple[str, list[str], dict]:
    """Collapse every retrieval result in a rendered transcript.

    Returns (rewritten, retrieved_doc_ids, stats).
    """
    ct, _relabelled = relabel_speakers(ct)
    out: list[str] = []
    ids: list[str] = []
    stats = collections.Counter()
    pos = 0
    marker = "(tool.result):"
    while True:
        i = ct.find(marker, pos)
        if i < 0:
            out.append(ct[pos:])
            break
        out.append(ct[pos:i])
        body_start = i + len(marker)
        m = _SEG_RE.search(ct, body_start)
        body_end = m.start() if m else len(ct)
        body = ct[body_start:body_end]
        rendered, doc_ids = collapse_retrieval(body, required)
        if rendered:
            stats["retrieval_results"] += 1
            stats["retrieval_chars_elided"] += len(body) - len(rendered)
            ids.extend(doc_ids)
            out.append(marker + " " + rendered + "\n")
            pos = body_end
            continue
        # Collapse BEFORE clipping: clip() would cut the frontmatter marker this
        # matches on, for any skill whose body exceeds the head budget.
        echo = collapse_skill_echo(body)
        if echo:
            stats["skill_echoes"] += 1
            stats["skill_chars_elided"] += len(body.strip()) - len(echo)
            out.append(marker + " " + echo + "\n")
            pos = body_end
            continue
        stats["other_results"] += 1
        clipped = clip(body.strip())
        stats["other_chars_elided"] += len(body.strip()) - len(clipped)
        out.append(marker + " " + clipped + "\n")
        pos = body_end
    return "".join(out), ids, dict(stats)


# --- inputs ----------------------------------------------------------------
if DOMAIN == _RETRIEVAL_DOMAIN and not VARIANT:
    refuse("EVOSKILL_TAU3_VARIANT is unset. The variant names the cache dir "
           "(it is NOT in the cache key), so guessing it would digest a "
           "different experiment's rollouts. Pass the one the leg ran with.")
if not SPLIT_KEY:
    refuse(
        "EVOSKILL_TAU3_DIGEST_SPLIT_KEY is unset. Refusing to guess: the whole\n"
        "  point of the split is that the evolver edits from TRAIN ONLY, and an\n"
        "  unset key on a shared cache is a firewall breach waiting to happen.\n"
        "  Pass train (or val/test for an audit that is not feeding an evolver).")
if SPLIT_FILE is None:
    # SYMMETRY WITH THE SPLIT_KEY GUARD ABOVE. That guard refuses to guess WHICH ARM to
    # digest; this one refuses to guess WHICH SPLIT the arm belongs to. Only having the
    # first is worse than having neither, because "train" reads as explicit while the
    # pool it indexes is silently whatever file the default happens to name.
    #
    # Measured on the real repo 2026-08-06, defaulting to split.json (32/32/33) while the
    # live experiment was split_2020.json (20/20/57): 7 of the new VAL tasks and 17 of the
    # new TEST tasks land in the digested "train" surface, and 12 of the 20 real train
    # tasks are silently absent. The evolver would have edited from val+test rollouts and
    # never seen most of its own training set -- an invisible firewall breach, since every
    # log line still says "train".
    _avail = sorted(p.name for p in (ROOT / "outputs/tau3_banking").glob("split*.json"))
    refuse(
        "EVOSKILL_TAU3_DIGEST_SPLIT_FILE is unset. Refusing to default: the split file\n"
        "  NAMES the task pool, and a stale default silently digests another split's\n"
        "  val/test rollouts into a surface every log line still calls 'train'.\n"
        f"  Available under outputs/tau3_banking/: {', '.join(_avail) or '(none)'}")
if not SPLIT_FILE.exists():
    refuse(f"no split at {SPLIT_FILE}")

_split = json.loads(SPLIT_FILE.read_text())
# Same guard as run_tau3_banking.py: the split file NAMES ITS OWN DOMAIN and nothing read
# it. airline (0..49) and retail (0..113) share a bare-numeric id space, so a mismatched
# split builds a digest surface from the wrong domain's rollouts and the evolver reasons
# about tasks that were never run.
_sdom = _split.get("domain")
if not _sdom:
    refuse(f"{SPLIT_FILE.name} records no 'domain' — refusing to guess its task pool.")
if _sdom != DOMAIN:
    refuse(f"split {SPLIT_FILE.name} is for {_sdom!r} but this build is {DOMAIN!r}; "
           f"airline/retail ids are both bare numerics, so this can silently resolve "
           f"against the wrong pool.")

if SPLIT_KEY not in ("train", "val", "test") or SPLIT_KEY not in _split:
    refuse(f"unknown split key {SPLIT_KEY!r}; have train/val/test")
allowed: set[str] = set(_split[SPLIT_KEY])
forbidden: set[str] = set()
for k in ("train", "val", "test"):
    if k != SPLIT_KEY:
        forbidden |= set(_split.get(k, []))
# ALSO the held-out ids a non-banking split records under an UNSELECTABLE key.
# airline/retail/telecom ship their own train/test partition and no val, so
# make_tau3_domain_smoke_split.py writes `val: []` and `test: []` (the runner matches
# split keys against an explicit tuple, so it can never select them) and puts the shipped
# test ids under `shipped_test_ids_NOT_A_SPLIT_KEY`. Reading only val|test there leaves
# `forbidden` EMPTY, and this builder then prints "firewall OK: 0 of 0 out-of-split tasks
# present" — a check that cannot fail, reported as a pass. Observed on the first airline
# digest build, 2026-08-06; same shape as #154 and as the oracle builder's own
# forbidden-set bug fixed in the same hour.
forbidden |= set(_split.get("shipped_test_ids_NOT_A_SPLIT_KEY", []))
forbidden -= allowed        # a task cannot be both the split under digest and held out
if not forbidden:
    print("  NOTE: the forbidden set is EMPTY, so the leak check below cannot fail. "
          "That is only legitimate when the split genuinely has no held-out ids "
          "recorded anywhere; treat a 'firewall OK: 0 of 0' line as UNTESTED.")

# --- #219: WHICH held-out ids may be redacted BY SUBSTRING, and which may not -------------
# Found 2026-08-09 when retail's first evolver launch refused: its digests read
#     **Result: 0/<a held-out task, id redacted> samples passed.**
#     turns=[<a held-out task, id redacted><a held-out task, id redacted>, ...]
# because retail task ids are BARE INTEGERS ('0', '3', '11', '100', ...) and the #193
# redaction below did `body.replace(_fid, ...)` for every held-out id. "3" is a held-out
# task id AND a sample count AND a turn count AND a digit of a duration, so the pass rate,
# the turn counts and the durations were all destroyed -- on every retail digest.
#
# THE RULE, and it is the same one that fixed #218 in make_tau3_oracle.py an hour earlier:
# redact by substring ONLY where a match is EVIDENCE OF A REFERENCE. `task_026` in prose can
# only be a reference to task_026. `3` in prose is a digit. There is no regex, word boundary
# or look-around that recovers the distinction, because the information is not in the text.
# So the ambiguous ids are NOT redacted textually; the cross-reference scan below establishes
# structurally that there is nothing to redact, and refuses if there is.
#
# This is the second site of one defect. I fixed #218 in the oracle builder and did not check
# whether the same blind-replace existed elsewhere -- it did, ten lines of a different file.
_REDACTABLE = {i for i in forbidden if len(i) >= 4 and not i.isdigit()}
_AMBIGUOUS = forbidden - _REDACTABLE

oracle: dict = {}
if ORACLE_TASKS.exists():
    oracle = json.loads(ORACLE_TASKS.read_text())
else:
    refuse(
        f"no oracle task file at {ORACLE_TASKS}. The digests would then carry no\n"
        f"  golden actions, no reward_basis and no required_documents — i.e. the\n"
        f"  evolver would be told what the solver did and never what it was\n"
        f"  supposed to do. Two of the 0801 epochs went out on that footing and\n"
        f"  reached opposite wrong conclusions. Build it with "
        f"scripts/make_tau3_oracle.py.")

# --- #219 part 2: the ambiguous ids are handled STRUCTURALLY, never textually --------------
# The leak #193 guards is an INCLUDED spec whose prose NAMES a HELD-OUT task ("Adversarial
# variant of task_026"). A cross-reference has a shape: a referring word next to an id. So
# rather than mangle every digit on the chance one of them is an id, look for that shape --
# and if it is there, refuse to the operator instead of guessing.
if _AMBIGUOUS:
    _refwords = r"(?:variant|variation|same as|see|cf\.?|follow[- ]?up|builds? on|task|episode)"
    _hits: list[tuple[str, str]] = []
    for _tid, _spec in oracle.items():
        for _m in re.finditer(_refwords + r"[^A-Za-z0-9]{0,4}(\d{1,4})\b",
                              json.dumps(_spec), re.I):
            if _m.group(1) in _AMBIGUOUS:
                _hits.append((_tid, _m.group(0)[:60]))
    if _hits:
        refuse(
            f"#219: {len(_hits)} possible cross-reference(s) to a HELD-OUT task id were found "
            f"in INCLUDED specs, and this domain's ids are bare integers, so they cannot be "
            f"redacted by substring without destroying every number in the digests.\n"
            + "".join(f"    {t}: ...{s}...\n" for t, s in _hits[:8])
            + "  Decide per case and redact structurally in the oracle, not here.")
    print(f"  #219 {len(_AMBIGUOUS)} held-out id(s) are bare integers -> substring redaction "
          f"is UNSAFE and was NOT applied.\n"
          f"       Cross-reference scan over {len(oracle)} included spec(s): no reference to "
          f"any held-out id, so there is nothing to redact.\n"
          f"       ({len(_REDACTABLE)} distinctive id(s) remain substring-redactable.)")

# An oracle built for a DIFFERENT split silently answers "required_documents?"
# with the wrong task's documents. Cheap to check, so check.
_oracle_extra = sorted(set(oracle) & forbidden)
if _oracle_extra:
    refuse(f"the oracle contains {len(_oracle_extra)} task(s) from another split "
           f"({_oracle_extra[:5]}) — it was not built with the {SPLIT_KEY} "
           f"firewall and must not be handed to an evolver.")

if CACHE_DIRS_ENV:
    cache_dirs = [(i, Path(p)) for i, p in
                  enumerate(x.strip() for x in CACHE_DIRS_ENV.split(",") if x.strip())]
else:
    # MUST mirror run_tau3_banking.py's construction EXACTLY, including the cache TAG and the
    # LEG token. It previously omitted both, so once the runner started writing
    # `tau3_<domain>_<variant>_<tag>_<leg>_s0_cache` (tag for instrument changes, leg added
    # 2026-08-07 for #148) this derivation pointed at a directory that does not exist and the
    # builder refused with "missing cache dir(s)" — and the only workaround was to pass every
    # path by hand, which is how the two names drift apart in the first place. Same
    # two-copies defect as #179 (RUN_DIR parameterised at three sites, missed at a fourth) and
    # #190 (OUT hardcoded while RUN_DIR was honoured).
    _ctag = os.environ.get("EVOSKILL_TAU3_CACHE_TAG", "").strip()
    _ctok = f"_{_ctag}" if _ctag else ""
    _dleg = os.environ.get("EVOSKILL_TAU3_DIGEST_LEG_DIR", "").strip()
    if not _dleg:
        _sd = os.environ.get("EVOSKILL_SKILLS_DIR", "").strip()
        _dleg = Path(_sd).name if _sd else "base"
    _dleg = "".join(c if c.isalnum() else "-" for c in _dleg).strip("-").lower() or "base"
    cache_dirs = [(s, ROOT / ".cache" / f"tau3_{DOMAIN}_{_VTOK}{_ctok}_{_dleg}_s{s}_cache")
                  for s in range(N_SAMPLES)]
missing_dirs = [str(p) for _s, p in cache_dirs if not p.is_dir()]
if missing_dirs:
    refuse(f"missing cache dir(s): {missing_dirs}\n"
           f"  A digest set built from a subset of the samples reports a k/N that "
           f"is not the leg's k/N, and that number drives the evolver's batch "
           f"pass rate.")

print(f"tau3 digests | {DOMAIN}/{_VTOK} split={SPLIT_KEY} ({len(allowed)} tasks) "
      f"samples={[s for s, _ in cache_dirs]}")
print(f"  oracle : {ORACLE_TASKS.name} ({len(oracle)} tasks)")

# --- scan ------------------------------------------------------------------
# rollouts[task_id][sample_idx] = record
rollouts: dict[str, dict[int, dict]] = {}
skipped_other_split = 0
buckets_seen: set[str] = set()

for sample_idx, cdir in cache_dirs:
    subdirs = sorted(p.name for p in cdir.iterdir() if p.is_dir())
    if BUCKET:
        if BUCKET not in subdirs:
            refuse(f"bucket {BUCKET!r} not in {cdir.name} (has {subdirs})")
        scan_dirs = [cdir / BUCKET]
    else:
        # SKILL-VERSION FIREWALL, same as the SSB builder's. One bucket per
        # tree_hash, i.e. per skill version. Two buckets in one dir means two
        # skill versions' rollouts are sitting side by side keyed identically,
        # and an unfiltered scan silently labels one as the other.
        if len(subdirs) > 1:
            refuse(
                f"{cdir.name} holds {len(subdirs)} tree_hash buckets {subdirs} "
                f"but EVOSKILL_TAU3_DIGEST_BUCKET is unset — refusing to mix "
                f"skill versions. Set it to the bucket for the leg you are "
                f"digesting.")
        scan_dirs = [cdir / d for d in subdirs]
    for sd in scan_dirs:
        buckets_seen.add(sd.name)
        for f in sorted(sd.glob("*.json")):
            d = json.loads(f.read_text())
            if "cache_key" not in d:
                continue          # sidecars (CACHE_ENV.json et al) are not rollouts
            ck = d["cache_key"]
            tid = json.loads(ck["question"])["task_id"]
            if tid not in allowed:
                skipped_other_split += 1
                continue

            # SAMPLE IDENTITY COMES FROM THE DIRECTORY, AND ONLY FROM IT.
            # `run_tau3_banking.py` gives each sample index its own cache dir and
            # calls the single-trace path, so the filename hash carries no
            # `|sample=` component (verified against real records 2026-08-04).
            # That is fine — but it is fine only as long as it stays true. If a
            # future runner switches to `set_n`, the hash WOULD carry
            # `|sample=i` and every rollout here would be labelled with the
            # directory's index regardless of its real one. The SSB builder
            # mislabelled 53 of 90 rollouts exactly that way, by trusting an
            # ordering instead of checking it. So: recompute the no-sample hash
            # and refuse if the filename disagrees.
            base = f"{ck['question'].strip()}|sdk={ck['sdk']}|model={ck['model']}"
            if ck.get("effort"):
                base += f"|effort={ck['effort']}"
            expect = hashlib.sha256(base.encode("utf-8")).hexdigest()[:len(f.stem)]
            if expect != f.stem:
                alt = [i for i in range(max(N_SAMPLES, 8))
                       if hashlib.sha256((base + f"|sample={i}").encode()
                                         ).hexdigest()[:len(f.stem)] == f.stem]
                refuse(
                    f"{f} does not hash as a no-sample record"
                    + (f" — it is a sample={alt[0]} record living in "
                       f"{cdir.name}. The cache layout changed under this "
                       f"builder; sample labels here would be wrong."
                       if alt else
                       " and no |sample=i form matches either. The cache key "
                       "shape changed; refusing to guess which sample this is."))

            if sample_idx in rollouts.get(tid, {}):
                refuse(f"DUPLICATE (task, sample): {tid} s{sample_idx} seen twice. "
                       f"Two legs are being merged and the digest would "
                       f"misrepresent which skills produced which behaviour.")
            rollouts.setdefault(tid, {})[sample_idx] = d

if len(buckets_seen) > 1:
    refuse(f"scanned {len(buckets_seen)} tree_hash buckets {sorted(buckets_seen)} "
           f"across the per-sample dirs — the samples did not run under the same "
           f"skill version, so their k/N is not a measurement of anything.")

if not rollouts:
    refuse(f"no {SPLIT_KEY} rollouts found under {[str(p) for _s, p in cache_dirs]}")


# --- rendering -------------------------------------------------------------
def parse_final_answer(tr: dict) -> dict | None:
    """tau2's structured verdict, out of `output.final_answer`.

    The evaluator writes `NaN` for an unpriced model's cost, which is valid
    Python-repr JSON but not valid strict JSON, so `json.loads` rejects the
    WHOLE verdict over a cost field nobody reads here. Substituting null keeps
    the grade readable; it is only ever the cost that is affected.
    """
    fa = (tr.get("output") or {}).get("final_answer")
    if fa is None:
        return None
    for candidate in (str(fa), re.sub(r"\bNaN\b", "null", str(fa))):
        try:
            v = json.loads(candidate)
        except (ValueError, TypeError):
            continue
        return v if isinstance(v, dict) else None
    return None


def classify_outcome(fa: dict | None, score: float | None) -> str:
    """Label WHAT KIND of outcome a rollout is. Derived, never assumed.

    A reward of 0.0 has two completely different meanings in this benchmark and
    the score alone cannot tell them apart:

      * the evaluator ran its checks and they FAILED  — a wrong answer;
      * the evaluator ran NO checks at all           — the 0.0 is the ABSENCE
        of a grade, and says nothing about whether the agent was right.

    The discriminator is the evaluator's own `reward_basis`: the list of
    components it multiplied into the reward. An empty basis means the product
    was over nothing. In the epoch-0 train batch every one of the 6 `timeout`
    episodes has `reward_basis == []` and all 90 `user_stop` episodes have a
    non-empty one (81 `['DB']`, 9 `['ACTION']`), so the two axes agree there —
    but they are DIFFERENT FACTS and this classifies on the basis, not on the
    termination reason, because the basis is what the reward was actually
    computed from.

    Why this belongs in the digest rather than in the evolver prompt: a prompt
    can only ASSERT the distinction in prose, next to a census I retyped by
    hand. A label on every sample lets the evolver read it off the evidence.
    That retyping is not hypothetical — the epoch-0 prompt claimed one episode
    "ran 2033s and finished normally" when the 2033s episode is in fact one of
    the six ungraded timeouts, at reward 0.0.
    """
    if fa is None:
        return "no_verdict"
    basis = fa.get("reward_basis") or []
    if not basis:
        term = fa.get("termination_reason") or "unknown"
        return f"ungraded({term})"
    if score is None:
        return "unscored"
    return "pass" if score >= 0.999 else "fail"


def render_grade(fa: dict | None) -> list[str]:
    if fa is None:
        return ["_No structured verdict on this trace — the episode produced no "
                "evaluator output at all, so WHY it scored what it scored cannot "
                "be read off. Treat the score as unexplained, not as a graded "
                "failure._", ""]
    L: list[str] = []
    basis = fa.get("reward_basis") or []
    bd = fa.get("reward_breakdown") or {}
    L += [f"- reward **{fa.get('reward')}**  = PRODUCT of {basis} "
          f"-> {json.dumps(bd)}",
          f"- termination_reason: **{fa.get('termination_reason')}**",
          f"- user sim: {fa.get('user_sim_tier')} / {fa.get('user_sim_model')}"]
    if not basis:
        # `PRODUCT of [] -> {}` is the whole story and it is easy to skim past.
        # Say what it means, because the reward printed above is 0.0 and looks
        # exactly like a graded failure.
        L += ["",
              f"**NOT GRADED — no check ran.** `reward_basis` is empty, so the "
              f"0.0 above is a product over nothing, not a verdict. The episode "
              f"ended `{fa.get('termination_reason')}` before the evaluator "
              f"scored anything: whether this agent was on track is NOT knowable "
              f"from the reward, only from the transcript below.", ""]
    db = fa.get("db_check") or {}
    if db:
        L.append(f"- DB check: match={db.get('db_match')} reward={db.get('db_reward')}")
    acts = fa.get("action_checks") or []
    if acts:
        ok = sum(1 for a in acts if a.get("action_match"))
        L += ["", f"**Golden actions: {ok}/{len(acts)} matched.** "
                  f"Only the arguments in each action's `compare_args` are "
                  f"compared; a null `compare_args` compares ALL of them.", ""]
        for a in acts:
            act = a.get("action") or {}
            mark = "OK  " if a.get("action_match") else "MISS"
            L.append(f"- `{mark}` {act.get('action_id')} "
                     f"{act.get('requestor')}->{act.get('name')} "
                     f"(tool_type={a.get('tool_type')}, reward={a.get('action_reward')})")
            L.append(f"      args={json.dumps(act.get('arguments'))[:400]}")
            if act.get("compare_args") is not None:
                L.append(f"      compare_args={json.dumps(act.get('compare_args'))}")
    for key, label in (("communicate_checks", "Communicate checks"),
                       ("nl_assertions", "NL assertions")):
        items = fa.get(key)
        if items:
            L += ["", f"**{label}:**", ""]
            for it in items:
                L.append(f"- {json.dumps(it)[:500]}")
        elif items == []:
            L += ["", f"_{label}: none for this task._"]
    return L + [""]


def _quote(s: str) -> str:
    return "\n".join(f"> {ln}" if ln.strip() else ">" for ln in s.strip().splitlines())


def render_task_spec(tid: str) -> list[str]:
    t = oracle.get(tid)
    if not t:
        return [f"_This task is not in the oracle ({ORACLE_TASKS.name}), so its "
                f"golden spec and required documents are UNKNOWN here. Do not "
                f"infer them from the transcript._", ""]
    ec = t.get("evaluation_criteria") or {}
    us = t.get("user_scenario") or {}
    L = ["## What the task IS (golden spec, from the oracle)", ""]
    desc = t.get("description") or {}
    # `purpose` is often just "Task: task_015" — say so rather than presenting a
    # placeholder as if it were a description.
    purpose = (desc.get("purpose") or "").strip()
    if purpose and purpose != f"Task: {tid}":
        L += [f"**Purpose:** {purpose}", ""]
    for k in ("relevant_policies", "notes"):
        if desc.get(k):
            L += [f"**{k}:** {desc[k]}", ""]
    # Quoted, not inlined. The scenario text carries its OWN `##` headings, so
    # inlining it puts "## Your Referral Inquiries" at the same level as this
    # digest's own sections and a reader loses track of which document they are
    # in. Blockquoting marks it unambiguously as the golden spec's words.
    if us.get("persona"):
        L += ["**User persona:**", "", _quote(str(us["persona"])), ""]
    if us.get("instructions"):
        L += ["**What the simulated user was told to do** (the agent never sees "
              "this; it is the source of everything the user says):", "",
              _quote(str(us["instructions"])), ""]
    L += [f"**reward_basis:** {ec.get('reward_basis')}  — the final reward is the "
          f"PRODUCT of these components, so any one of them at 0 zeroes the "
          f"episode.", ""]
    acts = ec.get("actions") or []
    L += [f"**Golden actions ({len(acts)}):**", ""]
    for a in acts:
        L.append(f"- {a.get('action_id')} {a.get('requestor')}->{a.get('name')} "
                 f"args={json.dumps(a.get('arguments'))[:300]}"
                 + (f" compare_args={json.dumps(a.get('compare_args'))}"
                    if a.get("compare_args") is not None else ""))
    if ec.get("communicate_info"):
        L += ["", "**communicate_info (must be conveyed to the user):**", ""]
        for c in ec["communicate_info"]:
            L.append(f"- {json.dumps(c)[:300]}")
    if ec.get("nl_assertions"):
        L += ["", "**nl_assertions (LLM-judged):**", ""]
        for c in ec["nl_assertions"]:
            L.append(f"- {json.dumps(c)[:300]}")
    req = t.get("required_documents") or []
    # EMIT THIS SECTION ONLY WHERE THE CONCEPT EXISTS. `required_documents` is a
    # banking_knowledge field — that domain is a knowledge-retrieval task with a policy
    # corpus behind it. airline/retail/telecom golden specs have no such field, and
    # rendering "**required_documents (0)** — the policy documents a correct answer
    # depends on:" for them states a FACT ABOUT THE TASK ("zero documents are required")
    # where the truth is that the notion does not apply. The evolver reads this as
    # authoritative, so a misleading zero is worse than an absent section.
    if req:
        L += ["", f"**required_documents ({len(req)})** — the policy documents a "
                  f"correct answer depends on:", ""]
        for r in req:
            L.append(f"- `{r}`")
    elif DOMAIN == _RETRIEVAL_DOMAIN:
        # In banking an EMPTY list is a real property of the task, not an absent concept.
        L += ["", "**required_documents (0)** — this task's golden spec names no "
                  "required documents, so retrieval recall cannot explain its outcome.", ""]
    return L + [""]


index: list[tuple[str, int, int, int]] = []
doc_recall_rows: list[tuple[str, int, int, int, float | None]] = []
outcome_census: collections.Counter[str] = collections.Counter()
# (task_id, sample_idx, n_skill_calls, n_skill_loads) per rollout — the leg-identity
# evidence, gathered from the transcripts themselves.
skill_evidence: list[tuple[str, int, int, int]] = []
# (task_id, prior_pass, prior_n, now_pass, now_n) for tasks whose pass COUNT moved.
regressions: list[tuple[str, int, int, int, int]] = []
# (task_id, redacted_held_out_id) — #193; reported so a silent redaction is impossible
redactions: list[tuple[str, str]] = []
improvements: list[tuple[str, int, int, int, int]] = []

for tid in sorted(rollouts):
    samples = rollouts[tid]
    idxs = sorted(samples)
    traces = {i: samples[i]["trace"] for i in idxs}
    fas = {i: parse_final_answer(traces[i]) for i in idxs}
    scores = [traces[i].get("inline_score") for i in idxs]
    npass = sum(1 for s in scores if s is not None and s >= 0.999)
    req = set((oracle.get(tid) or {}).get("required_documents") or [])

    kinds = [classify_outcome(fas[i], traces[i].get("inline_score")) for i in idxs]
    for k in kinds:
        outcome_census[k] += 1

    L = [f"# {tid}  ({DOMAIN}, variant {VARIANT})", ""]
    # THE MARKER THE LAUNCHER GREPS. `tau3_evolver.sh` derives the batch pass
    # rate from these lines, so the wording is load-bearing: keep it byte-stable.
    # Fields may be APPENDED (the launcher's regexes are per-key), never renamed.
    L += [f"**Result: {npass}/{len(idxs)} samples passed.**  "
          f"scores={scores}  "
          f"kind={kinds}  "
          f"turns={[traces[i].get('num_turns') for i in idxs]}  "
          f"duration_s={[round((traces[i].get('duration_ms') or 0) / 1000, 1) for i in idxs]}  "
          f"termination={[(fas[i] or {}).get('termination_reason') for i in idxs]}",
          ""]
    # Appended as its OWN line, never folded into the Result line above: the launcher
    # greps that line per-key and its wording is load-bearing.
    L += prior_block(tid, scores)
    if PRIOR:
        _pv = PRIOR.get(tid)
        if _pv is not None:
            _pn = sum(1 for s in _pv if isinstance(s, (int, float)) and s >= 0.999)
            if npass < _pn:
                regressions.append((tid, _pn, len(_pv), npass, len(idxs)))
            elif npass > _pn:
                improvements.append((tid, _pn, len(_pv), npass, len(idxs)))
    if any(k.startswith(("ungraded", "no_verdict", "unscored")) for k in kinds):
        L += ["**Read the `kind=` field before the scores.** At least one sample "
              "here was never graded — its 0.0 is the absence of a verdict, not a "
              "wrong answer, so it is no evidence about what the agent CONCLUDED. "
              "Its termination reason is still a measurement, though: a rule about "
              "pace or search economy can rest on it, a rule about correctness "
              "cannot.", ""]

    L += render_task_spec(tid)

    per_sample_rewrites = {}
    for i in idxs:
        ct = traces[i].get("cached_transcript") or ""
        per_sample_rewrites[i] = rewrite_transcript(ct, req) if ct else ("", [], {})

    # Document recall FIRST, before the transcripts, because it is the one line
    # that most often answers "why did this fail".
    if req:
        L += ["## Did the agent retrieve the documents the answer depends on?", "",
              "Computed by matching the required doc ids against every id that "
              "appeared in a retrieval result. `missed` means the id never came "
              "back from the retriever at all — a retrieval failure, not a "
              "reasoning failure. Note it can also be retrieved and ignored, "
              "which this table cannot distinguish; the transcript can.", ""]
        for i in idxs:
            got = set(per_sample_rewrites[i][1])
            hit = req & got
            miss = sorted(req - got)
            L.append(f"- **s{i}** ({'PASS' if (scores[idxs.index(i)] or 0) >= 0.999 else 'FAIL'}): "
                     f"retrieved {len(hit)}/{len(req)} required; "
                     f"{len(got)} distinct docs seen in total"
                     + (f"; MISSED {miss}" if miss else "; missed none"))
            doc_recall_rows.append((tid, i, len(hit), len(req),
                                    scores[idxs.index(i)]))
        L.append("")

    for i in idxs:
        tr = traces[i]
        sc = tr.get("inline_score")
        verdict = "PASS" if (sc is not None and sc >= 0.999) else "FAIL"
        rewritten, _ids, stats = per_sample_rewrites[i]
        skill_evidence.append((
            tid, i,
            len(_SKILL_CALL_RE.findall(tr.get("cached_transcript") or "")),
            int(stats.get("skill_echoes", 0)),
        ))
        L += [f"## Rollout s{i} — {verdict} "
              f"[kind={classify_outcome(fas[i], sc)}] (score={sc}, "
              f"{tr.get('num_turns')} turns, "
              f"{round((tr.get('duration_ms') or 0) / 1000, 1)}s)", ""]
        if tr.get("budget_exhausted"):
            L += [f"**Harness budget exhausted:** {tr['budget_exhausted']}", ""]
        L += ["### Grade (authoritative — this is what produced the score)", ""]
        L += render_grade(fas[i])
        L += ["### Transcript", ""]
        if not rewritten:
            L += ["_No transcript stored for this rollout._", ""]
        else:
            if stats:
                L += [f"_{stats.get('retrieval_results', 0)} retrieval result(s) "
                      f"collapsed to ranked lists ({stats.get('retrieval_chars_elided', 0):,} "
                      f"chars of document bodies elided); "
                      f"{stats.get('other_results', 0)} other tool result(s), "
                      f"{stats.get('other_chars_elided', 0):,} chars elided from "
                      f"long ones. Nothing else was altered._", ""]
            if "TRANSCRIPT ELIDED IN THE MIDDLE" in rewritten:
                L += ["**NOTE: this transcript hit the cache's storage cap and has "
                      "a labelled gap in the middle.** The marker names the turn "
                      "range that is missing. Those turns DID happen.", ""]
            L += [rewritten, ""]

    safe = tid.replace("/", "_")
    p = OUT / f"{safe}.md"
    OUT.mkdir(parents=True, exist_ok=True)

    # --- #193: REDACT HELD-OUT IDS THAT THE DATASET'S OWN SPECS REFERENCE -----------------
    # Found 2026-08-07 by the post-hoc leak audit on a run whose every other gate was green.
    # The oracle gate verifies held-out task SPECS are ABSENT, and the firewall below verifies
    # no held-out task got its own DIGEST. Neither checks whether an INCLUDED train spec
    # MENTIONS a held-out id -- and banking's own golden notes do exactly that:
    #   task_027 (train): "Adversarial variant of task_026: ..."   <- task_026 is TEST
    #   task_029 (train): "Adversarial variant of task_028: ..."   <- task_028 is TEST
    # So the train-only firewall was defeated by the DATASET'S CROSS-REFERENCES, not by any
    # failure to confine the evolver. Two guards that both check membership can both pass while
    # the content still leaks; this is the third, on the bytes actually handed over.
    body = "\n".join(L)
    _red = 0
    # #219: _REDACTABLE, NOT `forbidden`. Substring-replacing a bare-integer id rewrites the
    # sample counts, turn counts and durations of every digest in the domain -- see the block
    # where _REDACTABLE is defined. The bare-integer ids are cleared structurally instead.
    for _fid in sorted(_REDACTABLE, key=len, reverse=True):  # longest first: task_1 vs task_12
        if _fid and _fid in body:
            body = body.replace(_fid, "<a held-out task, id redacted>")
            _red += 1
            redactions.append((tid, _fid))
    p.write_text(body)

    # Assert on the WRITTEN bytes, not on the intent. A redaction that silently missed a
    # variant spelling would otherwise look identical to one that had nothing to redact.
    # Scoped to _REDACTABLE for the same reason: "3" occurs in every digest ever written,
    # so asserting over `forbidden` here would refuse every retail build unconditionally.
    _still = sorted(f for f in _REDACTABLE if f and f in p.read_text(errors="replace"))
    if _still:
        refuse(f"REDACTION FAILED for {tid}: held-out id(s) {_still[:5]} survive in the written "
               f"digest. Do not hand this surface to an evolver.")
    index.append((tid, npass, len(idxs), p.stat().st_size))

# --- firewall assertions, after the fact rather than on trust --------------
leaked = sorted({tid for tid, *_ in index} & forbidden)
if leaked:
    refuse(f"SPLIT FIREWALL BREACH: {len(leaked)} task(s) from another split "
           f"reached the digests: {leaked[:5]}")

# --- LEG IDENTITY, THE NON-VACUOUS VERSION -------------------------------
# On 2026-08-04 the SSB v2 audit found 8 skills_v4 rollouts sitting in the BASE
# leg's cache bucket, counted in base's published 78/360 and erasing base's
# entire ALL@3 lead. The reason nobody caught it is worth stating in full,
# because the same trap exists here: the bucket DIRECTORY is `tree_hash[:12]`
# and the tree_hash STORED in each record is that same truncated value, so
# "every record in this bucket shares one tree_hash" is true BY CONSTRUCTION
# and can never fail for any input. It reads like a contamination check and is
# incapable of detecting contamination. Worse, "the bucket holds exactly N
# records" is not proof either — a foreign write lands on a slot the leg never
# ran instead of adding an N+1th record, so the count staying right is exactly
# what makes it invisible.
#
# The mechanism is that an EMPTY OR MISSING skills dir hashes to exactly the
# BASE leg's tree_hash. So contamination is structurally one-directional: it can
# only ever flow INTO base. That asymmetry is what the assertion below exploits.
# A base rollout is never given the `skill` tool at all, so a single
# `(tool.skill)` call in a base leg's transcript is proof of a foreign record —
# evidence the record cannot fake, unlike a hash it merely carries.
#
# The symmetric assertion would be WRONG and is deliberately not made: uptake on
# a skill leg is not 100% (SSB v1 loaded a skill body in 330 of 360), so
# "every skill-leg rollout loaded a skill" would fire on honest data. Uptake is
# therefore REPORTED, not asserted.
_calls = sum(c for _t, _s, c, _l in skill_evidence)
_loads = sum(l for _t, _s, _c, l in skill_evidence)
_with_call = [(t, s) for t, s, c, _l in skill_evidence if c]
if LEG == "base":
    if _with_call:
        refuse(
            f"BASE LEG CONTAMINATED: {len(_with_call)} of {len(skill_evidence)} "
            f"rollouts invoked the `skill` tool, which a base solver is never "
            f"given: {_with_call[:8]}.\n"
            f"  An empty/missing skills dir hashes to the base tree_hash, so a "
            f"skill leg's skills-missing rollouts land in THIS bucket and are "
            f"counted as base. That is what happened to SSB v2 (8 of 360). "
            f"Do not digest this as a base leg.")
    print(f"  leg identity: BASE asserted — 0/{len(skill_evidence)} rollouts "
          f"invoked `skill` [OK]")
elif LEG == "skill":
    print(f"  leg identity: SKILL leg — {len(_with_call)}/{len(skill_evidence)} "
          f"rollouts invoked `skill` ({_calls} calls, {_loads} bodies loaded). "
          f"Uptake is REPORTED, not asserted: honest uptake is below 100%.")
else:
    print(f"  leg identity: NOT ASSERTED (EVOSKILL_TAU3_DIGEST_LEG unset). "
          f"{len(_with_call)}/{len(skill_evidence)} rollouts invoked `skill`, "
          f"{_loads} bodies loaded. Set the var to make this a check.")

missing = sorted(allowed - {tid for tid, *_ in index})
n_ep = sum(n for _t, _p, n, _s in index)
tot_pass = sum(p for _t, p, _n, _s in index)

print(f"  bucket : {sorted(buckets_seen)[0]}")
print(f"  excluded (other splits): {skipped_other_split} rollouts")
print(f"  firewall OK: 0 of {len(forbidden)} out-of-split tasks present")
if redactions:
    import collections as _c
    _by=_c.Counter(t for t,_f in redactions)
    print(f"  #193 REDACTED held-out ids referenced by INCLUDED specs: "
          f"{len(redactions)} occurrence(s) across {len(_by)} digest(s)")
    for _t,_f in redactions:
        print(f"    {_t} referenced {_f} -> replaced with a neutral placeholder")
    print("    (The dataset's own notes cross-reference held-out tasks; membership checks\n"
          "     cannot catch that, so the digest bytes are scrubbed and re-asserted.)")
else:
    print("  #193 no held-out id was referenced by any included spec")
if missing:
    print(f"  WARNING: {len(missing)} {SPLIT_KEY} task(s) have NO rollouts: "
          f"{missing[:8]}")
print(f"  {len(index)} task digests -> {OUT}")
for tid, np_, n, sz in index:
    print(f"    {tid:<12} {np_}/{n} pass   {sz // 1024}KB")

# --- REGRESSION ROSTER ----------------------------------------------------------------
# A per-file DELTA line the evolver must notice 25 separate times is weaker evidence than
# one roster it reads once. Written INTO the digest dir so it is part of the surface, not
# just operator console output.
if PRIOR:
    _rl = [f"# What the previously-active skills CHANGED  (prior leg: `{PRIOR_TAG}`)", "",
           "Pass COUNTS per task, prior -> now. This is the only place in the surface that",
           "distinguishes a task the current rules BROKE from one they merely failed to fix.",
           "Those two call for opposite edits, so read this before proposing anything.", ""]
    if regressions:
        _rl += [f"## REGRESSIONS — {len(regressions)} task(s) got WORSE", "",
                "A rule now firing on these is actively harmful. Prefer NARROWING its",
                "applicability condition, or removing it, over stacking another rule on top.", ""]
        _rl += [f"- **{t}**: {pp}/{pn} -> {cp}/{cn} passed" for t, pp, pn, cp, cn in regressions]
        _rl += [""]
    else:
        _rl += ["## REGRESSIONS — none. No task's pass count fell.", ""]
    if improvements:
        _rl += [f"## IMPROVEMENTS — {len(improvements)} task(s) got better", ""]
        _rl += [f"- **{t}**: {pp}/{pn} -> {cp}/{cn} passed" for t, pp, pn, cp, cn in improvements]
        _rl += [""]
    else:
        _rl += ["## IMPROVEMENTS — none. No task's pass count rose.", ""]
    _n_moved = len(regressions) + len(improvements)
    _rl += ["## How much of this is NOISE", "",
            f"{_n_moved} of {len(index)} tasks moved at all. On {N_SAMPLES} samples a task's "
            f"pass count is a noisy statistic: pure sampling noise alone moves a large share of "
            f"task pairs, so a SHORT roster here is consistent with the skills having done "
            f"nothing. Do not treat a single task's movement as evidence about a rule. Treat a "
            f"regression as a REASON TO LOOK at that task's transcript, not as proof by itself.",
            ""]
    (OUT / "REGRESSIONS.md").write_text("\n".join(_rl))
    print(f"  PRIOR LEG `{PRIOR_TAG}`: {len(regressions)} regression(s), "
          f"{len(improvements)} improvement(s) -> {OUT}/REGRESSIONS.md")
    for t, pp, pn, cp, cn in regressions:
        print(f"    REGRESSED  {t:<12} {pp}/{pn} -> {cp}/{cn}")
print(f"  BATCH: {tot_pass}/{n_ep} episodes passed "
      f"({tot_pass / n_ep:.1%})" if n_ep else "  BATCH: no episodes")

# OUTCOME CENSUS. Printed separately from the termination census because they
# answer different questions: `termination` says how the episode ENDED, `kind`
# says whether the reward it carries MEANS anything. Conflating them is a real
# mistake with a real instance — see classify_outcome's docstring.
_ungraded = sum(v for k, v in outcome_census.items() if not k in ("pass", "fail"))
print("  OUTCOME KINDS: "
      + ", ".join(f"{k}={v}" for k, v in sorted(outcome_census.items())))
if _ungraded:
    print(f"    {_ungraded} of {n_ep} episodes carry a reward NO CHECK PRODUCED. "
          f"They are not evidence of a wrong answer; a graded-failure rate that "
          f"counts them is inflated by {_ungraded}.")
else:
    print("    every episode was graded (non-empty reward_basis) — a 0.0 in this "
          "batch always means the checks ran and failed.")

if doc_recall_rows:
    full = sum(1 for (_tid, _s, hit, need, _sc) in doc_recall_rows if hit == need)
    print(f"  doc recall: {full}/{len(doc_recall_rows)} rollouts retrieved ALL "
          f"required documents")
    for bucket, label in ((True, "retrieved ALL required"),
                          (False, "missed at least one")):
        rows = [r for r in doc_recall_rows if (r[2] == r[3]) is bucket]
        if not rows:
            continue
        passed = sum(1 for r in rows if (r[4] or 0) >= 0.999)
        print(f"    {label:<24} n={len(rows):<4} passed {passed} "
              f"({passed / len(rows):.0%})")
    print("    (Descriptive only — which rollouts retrieve everything is not "
          "randomised, so this split is selection, not an effect.)")
