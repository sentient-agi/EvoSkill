#!/usr/bin/env python3
"""Layer 0 of the orchestrator — WHAT IS THIS BENCHMARK, AND DOES THE DATA AGREE?

This sits in front of intake/spec/loop. The user describes their benchmark in their own words;
this module turns that description into a CLAIM with named parts, then goes and CHECKS THE CLAIM
AGAINST THE FILES. Only a claim that survives contact with the dataset may become a frozen spec.

WHY VERIFY AT ALL — the user is the domain expert, after all.

Because a description is a memory of a dataset, not the dataset. Everything that has gone wrong
in this programme has the same shape: a plausible account of the data that the data did not
support, believed because it was plausible. A split "built with 45 tasks" of which 36 ids did not
exist in the harness's pool. A scorer that read values a repair had never recalculated. A digest
set whose ids came from a different split. Each was describable in one confident sentence, and
each was refuted the moment someone read the actual files. The cost of checking is seconds; the
cost of not checking is a 90-episode leg measuring the wrong thing.

WHAT A FINDING DOES WHEN IT FAILS — the maintainer's decision, 2026-08-09.

It ASKS A CLARIFYING QUESTION AND RE-CHECKS. It assumes it misunderstood the user before it
assumes the dataset is wrong, because the user knows their benchmark and this module knows only
what it was told a minute ago. What it must NOT do is record the mismatch and continue: an
unverified field that flows into a frozen spec is a guess wearing a digest.

WHAT THIS MODULE DOES NOT DO
  - It does not choose the split. It reports how many tasks EXIST; the counts are the user's.
  - It does not judge whether the benchmark is a good one.
  - It does not read held-out data. It counts and shapes; it never opens a label it was told is
    ground truth beyond confirming the file is there and parses.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path


# ── the claim ────────────────────────────────────────────────────────────────────────────
@dataclass
class BenchmarkClaim:
    """What the user SAYS their benchmark is. Every field is their assertion, not a fact yet.

    The locators are what make the claim checkable. A description like "each task is a folder
    with a spreadsheet in it" is unfalsifiable prose; `task_glob="data/*"` plus
    `input_glob="input.xlsx"` is a claim that either matches the filesystem or does not.
    """
    name: str
    root: str                      # dataset root the globs are relative to
    task_glob: str                 # enumerates one entry PER TASK, e.g. "data/*" or "tasks/*.json"
    id_from: str = "stem"          # "stem" | "name" | "parent" | "json:<key>"
    input_glob: str = ""           # relative to a task entry; "" = the task entry IS the input
    answer_glob: str = ""          # ground truth / golden spec, relative to a task entry
    scoring_desc: str = ""         # prose: how a response becomes a score
    scoring_script: str = ""       # path to the scorer, relative to root; "" = none
    # What the HARNESS calls this task pool. Not cosmetic: the tau3 runner refuses a split with
    # no `domain`, and the first split this CLI ever produced was rejected by it on exactly that
    # — a split the front door built that the engine could not consume. Asked, never derived:
    # I had been deriving it as `name.split()[-1]`, which is a guess that happens to work for
    # "tau3 retail" and silently would not for "SSB v2" or "tau3 banking_knowledge".
    domain: str = ""
    notes: str = ""

    # When set, the matched file(s) CONTAIN the tasks rather than being one. "" = one entry per
    # task. This exists because the first real dataset this module met (tau3 retail) keeps all
    # 114 tasks as elements of a single JSON array, and without it the pool reads as size 1.
    records_in: str = ""           # "" | "array" | "object" | "jsonl"

    def task_entries(self) -> list:
        return sorted(Path(self.root).glob(self.task_glob))

    def records(self) -> list:
        """The task RECORDS, whether they are files or elements inside files."""
        if not self.records_in:
            return [(p, None) for p in self.task_entries()]
        out = []
        for p in self.task_entries():
            try:
                if self.records_in == "jsonl":
                    docs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
                else:
                    d = json.loads(p.read_text())
                    docs = d if isinstance(d, list) else list(d.values())
            except Exception:                                        # noqa: BLE001
                continue
            out.extend((p, r) for r in docs)
        return out

    def record_id(self, p, rec) -> str:
        """The task's id, as the HARNESS forms it — which is not always one field.

        `parent+json:<key>` composes the containing directory with a record key, giving
        `Debugging/01_02`. SpreadsheetBench v2 needs it: `ssbv2_agent/dataset.py` builds ids as
        f"{category}/{record['id']}", and the bare `id` REPEATS across categories (179 distinct
        across 297 records). Without this form the claim can only name `json:id`, so `verify()`
        raises a uniqueness finding and `adopt_split` rejects every id in the split as outside the
        pool — the front door refusing a correct setup because its vocabulary was too small.

        Found by the AO while reading the repo for the SSB v2 brief, BEFORE anything was frozen:
        it named the field, the line, the exact count of collisions, and what would fail. That is
        the clarification channel working as designed — it asked rather than guessing a value.
        """
        if rec is None:
            return self.task_id(p)
        if not isinstance(rec, dict):
            return ""
        spec = self.id_from
        prefix = ""
        if spec.startswith("parent+"):
            prefix, spec = f"{Path(p).parent.name}/", spec[len("parent+"):]
        key = spec.split(":", 1)[1] if spec.startswith("json:") else "id"
        v = rec.get(key, "")
        return f"{prefix}{v}" if v != "" else ""

    def task_id(self, p: Path) -> str:
        if self.id_from.startswith("json:"):
            key = self.id_from.split(":", 1)[1]
            try:
                d = json.loads(p.read_text())
            except Exception:                                        # noqa: BLE001
                return ""
            return str(d.get(key, "")) if isinstance(d, dict) else ""
        return {"stem": p.stem, "name": p.name, "parent": p.parent.name}.get(self.id_from, p.stem)


@dataclass
class Finding:
    """One checked claim. `question` is what to ask the user when it fails — never a correction
    the module invents, because the user may simply have described it differently than it parsed."""
    check: str
    ok: bool
    detail: str
    question: str = ""
    evidence: dict = field(default_factory=dict)

    def line(self) -> str:
        return f"  {'ok  ' if self.ok else 'ASK '}  {self.check}\n        {self.detail}"


# ── verification ─────────────────────────────────────────────────────────────────────────
def verify(claim: BenchmarkClaim) -> list:
    """Check the claim against the files. Never raises on a mismatch — returns Findings, because
    the caller's job is to ASK, and a traceback is not a question."""
    out = []
    root = Path(claim.root)

    if not root.is_dir():
        return [Finding("dataset root exists", False, f"no directory at {root}",
                        f"Where is {claim.name}'s data? I looked in {root} and found nothing.")]

    entries = claim.task_entries()
    out.append(Finding(
        "the task pattern enumerates tasks", bool(entries),
        f"{claim.task_glob!r} under {root} matched {len(entries)} entr(ies)"
        + (f"; first: {[p.name for p in entries[:3]]}" if entries else ""),
        "" if entries else
        "That pattern matched nothing. What does ONE task look like on disk — a folder, a "
        "single file, or rows inside one big file?",
        evidence={"matched": len(entries)}))
    if not entries:
        return out

    # DOES ONE MATCHED ENTRY ACTUALLY HOLD MANY TASKS? A glob that matches a handful of files,
    # each parsing as a long list of records, almost always means the tasks live INSIDE the file
    # and the pool is not `len(entries)`. This module made exactly that error the first time it
    # was pointed at a real dataset: it reported "1 task" for tau3 retail, whose 114 tasks are
    # elements of one JSON array, and called the claim verified. A pool size that is wrong by
    # two orders of magnitude silently invalidates every split and every denominator after it.
    if not claim.records_in and len(entries) <= 3:
        for p in entries:
            if p.is_dir() or p.suffix not in (".json", ".jsonl"):
                continue
            try:
                d = json.loads(p.read_text())
            except Exception:                                        # noqa: BLE001
                continue
            n = len(d) if isinstance(d, (list, dict)) else 0
            if n > 3:
                out.append(Finding(
                    "one entry per task", False,
                    f"{p.name} matched as a single task but parses as {n} records. If those "
                    f"records ARE the tasks, the pool is {n}, not {len(entries)}.",
                    f"Does {p.name} contain {n} separate tasks, or is it one task? If it holds "
                    f"them all, I should read the tasks out of it rather than treat the file as "
                    f"one task.",
                    evidence={"file": p.name, "records": n}))
                return out

    # ── ids ──────────────────────────────────────────────────────────────────────────────
    recs = claim.records()
    ids = [claim.record_id(p, r) for p, r in recs]
    blank = [p.name for p, i in zip(entries, ids) if not i]
    dupes = sorted({i for i in ids if i and ids.count(i) > 1})
    out.append(Finding(
        "every task has an id, and ids are unique", not blank and not dupes,
        f"{len(set(ids))} distinct id(s) from {len(ids)} task record(s)"
        + (f"; BLANK for {blank[:4]}" if blank else "")
        + (f"; DUPLICATED: {dupes[:4]}" if dupes else ""),
        "" if not blank and not dupes else
        f"I derived ids using {claim.id_from!r} and got collisions or blanks. How is a task's "
        f"id determined?",
        evidence={"n_ids": len(set(ids))}))

    # BARE-INTEGER IDS. Not a failure — plenty of real benchmarks use them — but it must be
    # SURFACED, because it silently breaks two things downstream that have both bitten already:
    # redaction (#219: blind substring replacement of "3" rewrote sample counts and durations
    # across every retail digest) and leak detection (#218: a bare id matches any number in any
    # prose). Anything relying on an id being distinctive must know before the run, not after.
    bare = [i for i in ids if i.isdigit()]
    if bare:
        out.append(Finding(
            "task ids are distinctive strings", True,
            f"{len(bare)} of {len(ids)} ids are BARE INTEGERS ({bare[:5]}). Not an error, but "
            f"downstream redaction and held-out-leak checks must match them STRUCTURALLY "
            f"(tasks/<id>.json, task_<id>) and never as loose numbers.",
            evidence={"bare_integer_ids": len(bare)}))

    # ── inputs / answers ─────────────────────────────────────────────────────────────────
    for label, glob, why in (
            ("input", claim.input_glob, "the thing the agent is given"),
            ("answer", claim.answer_glob, "the ground truth a response is scored against")):
        if not glob:
            continue
        missing = [claim.task_id(p) or p.name for p in entries
                   if not (list(p.glob(glob)) if p.is_dir() else
                           list(p.parent.glob(glob)))]
        out.append(Finding(
            f"every task has its {label}", not missing,
            f"{glob!r} ({why}) present for {len(entries)-len(missing)}/{len(entries)} task(s)"
            + (f"; MISSING for {missing[:5]}" if missing else ""),
            "" if not missing else
            f"{len(missing)} task(s) have no {label} matching {glob!r}. Is the {label} stored "
            f"somewhere else for those, or are they meant to be excluded?",
            evidence={"missing": len(missing)}))

    # ── scorer ───────────────────────────────────────────────────────────────────────────
    if claim.scoring_script:
        sp = root / claim.scoring_script
        exists = sp.is_file()
        out.append(Finding(
            "the scoring script exists", exists,
            f"{sp}" + ("" if exists else " — not found"),
            "" if exists else f"I could not find the scorer at {sp}. Where does scoring happen?"))
        if exists and sp.suffix == ".py":
            src = sp.read_text(errors="replace")
            fns = re.findall(r"^def\s+(\w+)", src, re.M)
            out.append(Finding(
                "the scoring script exposes a callable", bool(fns),
                f"top-level functions: {fns[:8] or 'NONE'}",
                "" if fns else
                f"{sp.name} defines no top-level function. How is it invoked — as a script?"))
    elif claim.scoring_desc:
        out.append(Finding(
            "scoring is described but has no script", True,
            f"scored by: {claim.scoring_desc[:110]}. No script named, so scoring is either "
            f"built into the harness or done by a judge model — that has to be true, because "
            f"nothing here can check it."))

    return out


def unresolved(findings: list) -> list:
    """The questions to put to the user, in order. Empty = the claim survived the dataset."""
    return [f for f in findings if not f.ok]


# ── splits ───────────────────────────────────────────────────────────────────────────────
def _per_basis_source(claim) -> str:
    """Where a per-task evaluation basis could be read from, or "" if nowhere.

    tau3 banking_knowledge ships a `tasks/` DIRECTORY of one file per task, and that is what the
    gate script reads each task's reward basis from. tau3 retail ships only `tasks.json`, so no
    per-task basis is derivable and #156 reporting is impossible on it. The split must say which
    of those two worlds it is in, at build time — discovering it later, from a scoring tool that
    refuses to run, is how the agentic retail run ended up ungatable by the canonical script.
    """
    root = Path(getattr(claim, "root", "") or "")
    if not root:
        return ""
    for cand in (root / "tasks", root.parent / "tasks"):
        try:
            if cand.is_dir() and any(cand.iterdir()):
                return str(cand)
        except OSError:
            continue
    return ""


def adopt_split(claim: BenchmarkClaim, split_path) -> dict:
    """Take an EXISTING split as given, after checking it against this claim's pool.

    WHY ADOPTING BEATS REBUILDING, and it is not convenience. The alternative for a follow-up run
    is `build_split` with a fresh seed, which re-partitions the same 97 tasks — and that quietly
    breaks the seal on the PREVIOUS run's test set, because the new run's TRAIN would contain
    tasks that the old run is still holding back. Nothing in either run would report an error; the
    old sealed test would simply have been shown to an evolver, once, in a different directory.
    That is the whole class of contamination the seal exists to prevent, arriving by the one route
    a per-run check cannot see.

    IT NEVER REPARTITIONS AND NEVER RENAMES. The file is used as-is; this only refuses the ways it
    could be wrong: unknown ids, overlapping sets, or a pool that has changed underneath it.
    """
    p = Path(split_path)
    if not p.is_file():
        raise ValueError(f"no split at {p}")
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"{p} is not readable JSON: {e}") from e
    for k in ("train", "val", "test"):
        if not isinstance(d.get(k), list):
            raise ValueError(f"{p} has no {k!r} list — this is not a split file")
    tr, va, te = d["train"], d["val"], d["test"]
    pool = {i for i in (claim.record_id(pp, r) for pp, r in claim.records()) if i}
    unknown = sorted(set(tr + va + te) - pool)
    if unknown:
        raise ValueError(
            f"{p} names {len(unknown)} id(s) that are not in this benchmark's pool of "
            f"{len(pool)} — first few: {unknown[:5]}. Either the split belongs to a different "
            f"dataset or the claim points somewhere else; adopting it would build an experiment "
            f"on tasks that cannot be run.")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        both = sorted(set(d[a]) & set(d[b]))
        if both:
            raise ValueError(f"{p} puts {len(both)} task(s) in BOTH {a} and {b}: {both[:5]}")
    d["n"] = {"train": len(tr), "val": len(va), "test": len(te)}
    # THE DIGEST MUST BE THE ONE spec.validate() WILL RECOMPUTE, not a second hash of the same
    # bytes. `split_digest` reproduces the GENERATOR's canonicalisation for a file carrying its
    # own `self_sha256` (the field is excluded before hashing, so a whole-file hash disagrees by
    # construction) and falls back to the whole-file hash otherwise. Writing the whole-file hash
    # here made freeze fail with "split_sha256 mismatch" on every split that self-attests — i.e.
    # on exactly the established splits adopt_split exists to reuse. spec.py does not import this
    # module, so there is no cycle.
    from src.orchestrator.spec import split_digest
    d["file_sha256"] = split_digest(p)
    d["adopted_from"] = str(p)
    return d


def build_split(claim: BenchmarkClaim, *, n_train: int, n_val: int, n_test: int,
                seed: int, out_path) -> dict:
    """Create train/val/test from the VERIFIED task pool.

    The seed and the resulting id lists are written into the file, and the file hashes ITSELF —
    so a split can prove its own provenance later without trusting whoever quotes it. Sampling is
    without replacement from one shuffled pool, so the three sets are disjoint by construction
    rather than by a check that could be skipped.

    REFUSES to silently shrink. Asking for more tasks than exist is a misunderstanding about the
    dataset, and quietly returning fewer would make every later denominator wrong.
    """
    ids = [i for i in (claim.record_id(p, r) for p, r in claim.records()) if i]
    want = n_train + n_val + n_test
    if want > len(ids):
        raise ValueError(f"asked for {want} tasks ({n_train}/{n_val}/{n_test}) but the pool holds "
                         f"{len(ids)}. Nothing is written; choose counts that fit, or point me at "
                         f"a larger pool.")
    if len(set(ids)) != len(ids):
        raise ValueError("the task pool contains duplicate ids; a split built on it could put the "
                         "same task in train and test")

    pool = sorted(ids)
    random.Random(seed).shuffle(pool)
    tr, va, te = pool[:n_train], pool[n_train:n_train + n_val], \
        pool[n_train + n_val:n_train + n_val + n_test]
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))

    # A split IS the pre-registration. Until 2026-08-09 this one recorded only ids and counts, so
    # the gate it would be judged by lived in my session notes rather than in the artifact — and
    # `scripts/tau3_gate_epoch.py` refused the whole benchmark for want of a REPORTING_RULE it
    # expects every split to carry (#235c). Declare both here, BEFORE any leg runs, which is the
    # only moment at which declaring them means anything.
    per_basis = _per_basis_source(claim)
    payload = {
        "benchmark": claim.name,
        "domain": claim.domain,
        "root": claim.root,
        "task_glob": claim.task_glob,
        "id_from": claim.id_from,
        "seed": seed,
        "pool_size": len(pool),
        "n": {"train": n_train, "val": n_val, "test": n_test},
        "train": tr, "val": va, "test": te,
        "TEST_IS_SEALED": ("The test ids are recorded so the split is reproducible. They must "
                           "never be rolled out during evolution and never shown to an evolver."),
        "POWER_NOTE": (
            f"n_val={n_val} tasks x N samples. The pre-registered gate is BOTH of: (1) an "
            f"EPISODE-LEVEL paired sign test over (task, sample slot) pairs, and (2) a "
            f"cluster-respecting pass-count sign test over tasks. Both must clear p<0.05, exact "
            f"two-sided binomial on DISCORDANT pairs only. Fewer than 6 discordant pairs cannot "
            f"reach p<0.05 at all, so a result with fewer is UNDERPOWERED, not negative. "
            f"Episode-level pairing alone is anti-conservative under within-task clustering, "
            f"which is why (2) is required alongside it."),
        "REPORTING_RULE": (
            "Report per-category deltas as well as the pooled number (#156) — an aggregate can "
            "average two opposite-sign effects into a null. Always report the outcome TRIPLE "
            "pass / graded-fail / never-graded (#209): a never-graded episode hit a step cap or "
            "context limit and was never scored, so counting it as a graded failure over-claims. "
            "Never call any epoch the 'best epoch' (#121)."),
        "PER_BASIS_SOURCE": per_basis or (
            "NONE FOUND. This benchmark exposes no per-task file from which an evaluation basis "
            "could be read, so per-category reporting under #156 is NOT AVAILABLE and any gate "
            "verdict on this split is pooled-only. That is a real limitation of the split, "
            "recorded here rather than discovered later by a tool that refuses to score it."),
    }
    # HASH THE FILE, NOT THE PAYLOAD-MINUS-ITSELF. The first version hashed the canonical body
    # before inserting the digest, so nobody re-reading the file could reproduce the number:
    # ExperimentSpec.validate() hashes the FILE BYTES and reported a mismatch against a split it
    # had just been handed. A digest two components compute differently is not a digest.
    p = Path(out_path)
    if p.exists():
        prev = json.loads(p.read_text())
        prev["file_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
        if json.loads(json.dumps(prev, sort_keys=True)).get("train") != tr or \
                prev.get("val") != va or prev.get("test") != te:
            raise ValueError(f"{p} already holds a DIFFERENT split. A split is a pre-registration; "
                             f"overwriting it after any leg has run silently redefines what was "
                             f"measured. Choose a new path.")
        return prev
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=1))
    # `file_sha256` is added to the RETURNED dict but is deliberately NOT written into the file:
    # a file cannot contain a hash of itself, and the earlier attempt to fake it (hash the body
    # minus the digest) produced a number nobody re-reading the file could reproduce. The digest
    # is held by the ExperimentSpec, which hashes these bytes — one definition, one holder.
    payload["file_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return payload


def render(findings: list) -> str:
    n_ask = len(unresolved(findings))
    body = "\n".join(f.line() for f in findings)
    tail = (f"\n\n  {n_ask} thing(s) I need to ask you about before this can be frozen."
            if n_ask else "\n\n  The dataset matches everything you told me.")
    return body + tail
