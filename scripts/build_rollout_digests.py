#!/usr/bin/env python3
"""Turn the 90 raw rollouts into one readable digest per task.

Why digests and not the raw cache: each rollout carries ~83 messages and the
cache dir is 50MB. An analyst reading raw JSON burns its context on tool-output
noise and never reaches the reasoning. So per rollout we keep the assistant's
own words in full (that is where the diagnosis lives), the exact bash commands,
and only a head/tail slice of each tool result.

The scorer's DIFF.txt rides along per sample. It is authoritative and free —
it is the same comparison that produced the score, so any claim about WHY a
rollout failed should be checkable against it rather than guessed from the
transcript.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Env-driven so each evolution phase points it at its own rollouts without
# editing the file (moved out of a disposable job tmp dir 2026-08-01).
CACHE = Path(os.environ.get("EVOSKILL_DIGEST_CACHE", ROOT / ".cache/ssbv2_base30_0731b_cache"))
SNAPS = Path(os.environ.get("EVOSKILL_DIGEST_SNAPS", ROOT / ".cache/ssbv2_base30_0731b_tasks/_snapshots"))
OUT = Path(os.environ.get("EVOSKILL_DIGEST_OUT", "/tmp/digests"))
OUT.mkdir(parents=True, exist_ok=True)

TOOL_HEAD, TOOL_TAIL = 1200, 600


def clip(s: str, head: int = TOOL_HEAD, tail: int = TOOL_TAIL) -> str:
    s = s or ""
    if len(s) <= head + tail:
        return s
    return f"{s[:head]}\n    ...[{len(s)-head-tail} chars elided]...\n{s[-tail:]}"


def content_text(c) -> str:
    """Messages carry either a string or a list of typed blocks."""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        out = []
        for b in c:
            if not isinstance(b, dict):
                out.append(str(b))
                continue
            t = b.get("type")
            if t == "text":
                out.append(b.get("text", ""))
            elif t == "thinking":
                out.append("[thinking] " + b.get("thinking", ""))
            elif t == "tool_use":
                inp = b.get("input") or {}
                out.append(f"[TOOL CALL {b.get('name')}]\n{inp.get('command', json.dumps(inp)[:2000])}")
            elif t == "tool_result":
                out.append("[TOOL RESULT]\n" + clip(content_text(b.get("content"))))
            else:
                out.append(f"[{t}]")
        return "\n".join(x for x in out if x)
    return str(c)


MODEL = os.environ.get("EVOSKILL_SOLVER_MODEL", "deepseek/deepseek-v4-flash-0731")
N_SAMPLES = 3


def true_sample_idx(question: str, stem: str) -> int | None:
    """Recover the REAL sample index of a cache record.

    sample_idx is not a field in the record, but it IS in the filename: RunCache
    keys each file as sha256(f"{question}|sdk={sdk}|model={model}|sample={i}")
    truncated to hash_length. So recompute the hash for each candidate i and see
    which one produced this filename.

    The first version of this script instead numbered rollouts by rglob() order.
    That is arbitrary, while the DIFF blocks come from correctly-named snapshot
    dirs (iter_0_s0/s1/s2) -- so transcripts and diffs were paired by a label
    that was wrong for 53 of 90 rollouts. Nine of ten downstream analysts caught
    the mismatch and re-derived pairing from cell-value fingerprints, which is
    the only reason the conclusions held. Never infer an index from iteration
    order when the index is recoverable.
    """
    for i in range(N_SAMPLES):
        keyed = f"{question.strip()}|sdk=direct|model={MODEL}|sample={i}"
        if hashlib.sha256(keyed.encode("utf-8")).hexdigest()[:12] == stem:
            return i
    return None


# --- VAL FIREWALL -----------------------------------------------------------
# The 0801 evolution shares ONE cache dir across every phase, and a given
# epoch's train and val legs run under the SAME skills dir — so they hash to the
# same tree_hash bucket and their records sit side by side. A bare
# CACHE.rglob("*.json") therefore sweeps val rollouts into the train analysis,
# which silently breaks the run's first design decision (edit from train only).
#
# So: when a split file is supplied, keep ONLY that split's task_ids and assert
# afterwards that nothing from another split survived. Refusing to guess is the
# point — an unset split on a shared cache is a firewall breach waiting to
# happen, so we make it loud rather than convenient.
SPLIT_FILE = os.environ.get("EVOSKILL_DIGEST_SPLIT_FILE")
SPLIT_KEY = os.environ.get("EVOSKILL_DIGEST_SPLIT_KEY")
allowed: set[str] | None = None
forbidden: set[str] = set()
if SPLIT_FILE:
    if not SPLIT_KEY:
        raise SystemExit("EVOSKILL_DIGEST_SPLIT_FILE set without EVOSKILL_DIGEST_SPLIT_KEY")
    _split = json.loads(Path(SPLIT_FILE).read_text())
    allowed = set(_split[SPLIT_KEY])
    for k in ("train", "val", "test"):
        if k != SPLIT_KEY:
            forbidden |= set(_split.get(k, []))
    print(f"SPLIT FILTER: {SPLIT_KEY} ({len(allowed)} tasks); "
          f"{len(forbidden)} tasks from other splits will be excluded")

# --- SKILL-VERSION FIREWALL -------------------------------------------------
# The shared cache holds one BUCKET per tree_hash, i.e. per skill version. By
# epoch 1 the same 60 train tasks exist in BOTH the no-skill bucket and the
# v1 bucket. Records are keyed by (task_id, sample_idx), which is identical
# across buckets — so an unfiltered rglob silently overwrites one version's
# rollouts with the other's, and the evolver receives a mix of pre- and
# post-edit traces labelled as post-edit. Caught 0801 before epoch 2 ran.
#
# EVOSKILL_DIGEST_BUCKET restricts the scan to one bucket. The collision
# assertion below is the general guard: it fires on ANY duplicate
# (task_id, sample_idx), so a future cache layout change cannot reintroduce
# this quietly.
BUCKET = os.environ.get("EVOSKILL_DIGEST_BUCKET")
scan_root = CACHE / BUCKET if BUCKET else CACHE
if BUCKET and not scan_root.is_dir():
    raise SystemExit(f"no such cache bucket: {scan_root}")
_buckets = sorted(p.name for p in CACHE.iterdir() if p.is_dir())
if not BUCKET and len(_buckets) > 1:
    raise SystemExit(
        f"cache has {len(_buckets)} skill-version buckets {_buckets} but "
        f"EVOSKILL_DIGEST_BUCKET is unset — refusing to mix skill versions. "
        f"Set it to the bucket for the phase you are digesting."
    )
print(f"BUCKET: {BUCKET or _buckets[0]} (of {len(_buckets)} in cache)")

rollouts: dict[str, dict[int, dict]] = {}
unresolved = 0
skipped_other_split = 0
for f in scan_root.rglob("*.json"):
    d = json.loads(f.read_text())
    # Not every .json under a cache dir is a rollout record. Skip anything
    # without a cache_key rather than dying on it: a sidecar dropped into the
    # cache dir must not be able to abort a digest build. (Cost 20 minutes on
    # 2026-08-03 when a CACHE_ENV.json marker did exactly that.)
    if "cache_key" not in d:
        continue
    raw_q = d["cache_key"]["question"]
    q = json.loads(raw_q)
    tid = q["task_id"]
    if allowed is not None and tid not in allowed:
        skipped_other_split += 1
        continue
    tr = d["trace"]
    idx = true_sample_idx(raw_q, f.stem)
    if idx is None:
        unresolved += 1
        continue
    if idx in rollouts.get(tid, {}):
        raise SystemExit(
            f"DUPLICATE (task, sample): {tid} s{idx} seen twice while scanning "
            f"{scan_root}. Two skill versions are being merged; the digests "
            f"would misrepresent which skills produced which behaviour."
        )
    rollouts.setdefault(tid, {})
    rollouts[tid][idx] = {
        "instruction": q["instruction"],
        "category": q.get("category"),
        "file": q.get("spreadsheet_filename"),
        "score": tr.get("inline_score"),
        "turns": tr.get("num_turns"),
        "exhausted": tr.get("budget_exhausted"),
        "cost": tr.get("total_cost_usd"),
        "messages": tr.get("messages") or [],
    }

SKILLS_DIR = os.environ.get("EVOSKILL_DIGEST_SKILLS")


def _load_skill_heads() -> list[tuple[str, int]]:
    """(skill-name, full body length) for the skills the solver was carrying.

    Used to collapse skill-load tool results — see _collapse_skill_echo.
    """
    if not SKILLS_DIR:
        return []
    out = []
    for f in sorted(Path(SKILLS_DIR).rglob("SKILL.md")):
        out.append((f.parent.name, len(f.read_text())))
    return out


_SKILL_HEADS = _load_skill_heads()


def _collapse_skill_echo(text: str) -> str:
    """Replace a skill-load tool result with a one-line marker.

    Every rollout re-emits the SKILL.md bodies it loaded as tool results, so the
    evolver reads its own previous-epoch output back — 341 blocks, 0.60 MB, ~10%
    of the corpus across 60 digests. It is already clipped by `clip()` to ~1.8 KB
    a block from ~8.8 KB bodies, so this is not the corpus hog it first appeared
    to be (a first pass mis-measured it at 47% by multiplying line-hits by full
    body length; the bodies are NOT contiguous in the digest). But clipped skill
    text is worse than none: the evolver sees a truncated, mid-sentence version of
    a file it can read in full in its staging dir, which invites reasoning about
    text that was never actually in the solver's context in that form.
    WHAT IS PRESERVED: that the skill loaded, and which one. That is the part the
    evolver needs to attribute behaviour to a rule. The body it already has.
    No-op unless EVOSKILL_DIGEST_SKILLS is set, so this cannot silently alter a
    digest build that did not ask for it.
    """
    if not _SKILL_HEADS:
        return text
    t = text.lstrip("\n")
    for name, size in _SKILL_HEADS:
        if t.startswith(f"---\nname: {name}\n"):
            return (f"[SKILL LOADED: {name} — {size} B. Body omitted here; it is "
                    f"the file you were handed in your staging dir, so read it "
                    f"there rather than from this truncated echo.]")
    return text


def _solver_edit_section(sd: Path, max_cells: int = 60) -> list[str]:
    """"What the solver itself changed" — precomputed from the RIGHT two files.

    The scorer diff answers *"what did the grader see?"* (golden vs the
    recalculated candidate). It does NOT answer *"what did the solver do?"*, and
    the two get conflated because the solver saves under the input's filename,
    so a rollout dir holds three same-named workbooks.

    On 2026-08-01 epoch 2 concluded "never re-save", epoch 3 concluded "re-saving
    is harmless, freeze everything", and I concluded "the solver overwrote the
    data table" — three different confident answers, all from mis-paired files.
    Computing it once, here, removes the opportunity.

    Requires ``workspace/output_raw/`` (rollouts from 2026-08-02 on). For older
    snapshots it says so plainly rather than silently diffing the wrong pair,
    because a section that quietly compares output-to-output is worse than no
    section at all.
    """
    ws = sd / "workspace"
    ins = sorted((ws / "spreadsheets").glob("*.xlsx"))
    raws = sorted((ws / "output_raw").glob("*.xlsx"))
    outs = sorted((ws / "output").glob("*.xlsx"))
    head = f"## What the SOLVER changed — {sd.name} (input vs the solver's own output)"

    if not ins:
        return [head, "", "_No input snapshot for this rollout._", ""]
    if not raws:
        return [
            head, "",
            "_NOT AVAILABLE for this rollout._ No `workspace/output_raw/` was "
            "captured, so the solver's pre-recalculation bytes do not exist and "
            "this question cannot be answered. Do NOT substitute "
            "`workspace/output/` — the scorer rewrites it in place before "
            "comparing (`_recalculate_with_libreoffice`, "
            "`_solve_datatables_with_uno`, …), so an input-vs-`output/` diff "
            "mixes the solver's edits with the grader's recalculation and cannot "
            "distinguish them."
            + ("" if outs else " (No output file either — the solver saved nothing.)"),
            "",
        ]

    try:
        import openpyxl
        wa = openpyxl.load_workbook(ins[0], data_only=True)
        wb = openpyxl.load_workbook(raws[0], data_only=True)
    except Exception as e:                       # noqa: BLE001
        return [head, "", f"_Could not open the workbooks: {e}_", ""]

    rows: list[str] = []
    n_diff = 0
    for sheet in wb.sheetnames:
        if sheet not in wa.sheetnames:
            rows.append(f"- sheet `{sheet}`: ADDED by the solver")
            n_diff += 1
            continue
        sa, sb = wa[sheet], wb[sheet]
        for row in sb.iter_rows():
            for cell in row:
                old = sa.cell(row=cell.row, column=cell.column).value
                if old == cell.value:
                    continue
                n_diff += 1
                if len(rows) < max_cells:
                    rows.append(f"- `'{sheet}'!{cell.coordinate}`: "
                                f"{old!r} -> {cell.value!r}")
    for sheet in wa.sheetnames:
        if sheet not in wb.sheetnames:
            rows.append(f"- sheet `{sheet}`: REMOVED by the solver")
            n_diff += 1
    wa.close(); wb.close()

    same = ins[0].stat().st_size == raws[0].stat().st_size and n_diff == 0
    out = [head, ""]
    if n_diff == 0:
        out += [f"**The solver changed NOTHING**"
                f"{' (byte sizes also match)' if same else ''}. "
                f"Any scorer mismatch above therefore comes from the grader's "
                f"recalculation, not from an edit.", ""]
    else:
        out += [f"**{n_diff} cell(s) changed by the solver."
                f"{f' First {max_cells} shown.' if n_diff > max_cells else ''}**", ""]
        out += rows + [""]
    return out


# --- LEG IDENTITY, THE NON-VACUOUS VERSION ---------------------------------
# HARNESS PARITY. `scripts/tau3_build_digests.py` has carried this assertion since
# 2026-08-04; the SSB v2 path — the very path whose contamination motivated it —
# did not. That is backwards, so it is ported here verbatim in intent.
#
# WHY THE OBVIOUS CHECK CANNOT WORK. The bucket DIRECTORY is `tree_hash[:12]` and
# the tree_hash STORED in each record is that same truncated string, so "every
# record in this bucket shares one tree_hash" is true BY CONSTRUCTION and can never
# fail for any input. It reads like a contamination check and is incapable of
# detecting contamination — it is why 8 skills_v4 rollouts sat in the SSB v2 BASE
# bucket, counted in base's published 78/360, erasing base's entire ALL@3 lead.
# Record COUNT is no proof either: a foreign write lands on a slot the leg never
# ran rather than adding an N+1th record, so the count staying right is exactly what
# makes it invisible.
#
# WHAT MAKES THIS ONE REAL. An empty or missing skills dir hashes to exactly the
# BASE leg's tree_hash, so contamination is structurally ONE-DIRECTIONAL — it can
# only flow INTO base. A base solver is never handed any skill, so skill
# frontmatter appearing in a base rollout's tool results is evidence the record
# cannot fake, unlike a hash it merely carries.
#
# THE SYMMETRIC ASSERTION WOULD BE WRONG and is deliberately not made: honest
# uptake on a skill leg is below 100% (SSB v1 loaded a body in 330 of 360), so
# "every skill-leg rollout loaded a skill" would fire on good data. Uptake is
# REPORTED, never asserted.
_FM = re.compile(r"^---\s*\nname:\s*(\S+)", re.M)
LEG = os.environ.get("EVOSKILL_DIGEST_LEG")
_evidence = []
for _tid, _samples in sorted(rollouts.items()):
    for _i, _r in sorted(_samples.items()):
        _txt = "\n".join(content_text(m.get("content")) for m in _r["messages"]
                         if m.get("role") == "tool")
        _found = sorted(set(_FM.findall(_txt)))
        _evidence.append((_tid, _i, _found))
_with_skill = [(t, i, f) for t, i, f in _evidence if f]

if LEG == "base":
    if _with_skill:
        raise SystemExit(
            f"BASE LEG CONTAMINATED: {len(_with_skill)} of {len(_evidence)} rollouts "
            f"carry SKILL.md frontmatter in a tool result, which a base solver is "
            f"never given: {_with_skill[:6]}.\n"
            f"  An empty/missing skills dir hashes to the base tree_hash, so a skill "
            f"leg's skills-missing rollouts land in THIS bucket and are counted as "
            f"base. That is what happened to SSB v2 (8 of 360). Do NOT digest this "
            f"as a base leg.")
    print(f"  leg identity: BASE asserted — 0/{len(_evidence)} rollouts loaded a "
          f"skill body [OK]")
elif LEG == "skill":
    _names = sorted({n for _t, _i, f in _evidence for n in f})
    print(f"  leg identity: SKILL leg — {len(_with_skill)}/{len(_evidence)} rollouts "
          f"loaded a skill body {_names}. Uptake is REPORTED, not asserted: honest "
          f"uptake is below 100%.")
else:
    print(f"  leg identity: NOT ASSERTED (EVOSKILL_DIGEST_LEG unset). "
          f"{len(_with_skill)}/{len(_evidence)} rollouts loaded a skill body. "
          f"Set it to 'base' or 'skill' to make this a check.")

index = []
for tid, samples in sorted(rollouts.items()):
    safe = tid.replace("/", "_")
    first = samples[0]
    scores = [samples[i]["score"] for i in sorted(samples)]
    npass = sum(1 for s in scores if s and s >= 0.999)

    L = [
        f"# {tid}  ({first['category']})",
        "",
        f"**Result: {npass}/{len(scores)} samples passed.**  "
        f"turns={[samples[i]['turns'] for i in sorted(samples)]}  "
        f"budget_exhausted={[samples[i]['exhausted'] for i in sorted(samples)]}",
        "",
        "## Instruction (what the solver was asked to do)",
        "",
        first["instruction"].strip(),
        "",
        f"Input file: `{first['file']}`",
        "",
    ]

    snapdir = SNAPS / safe
    if snapdir.is_dir():
        for sd in sorted(snapdir.iterdir()):
            # Iter dirs only. The old code was accidentally safe here — a stray
            # FILE just failed the `(sd/"DIFF.txt").exists()` test and was
            # skipped — but _solver_edit_section is called unconditionally below
            # and would emit a spurious "_No input snapshot_" block for it. A
            # single .DS_Store would put a phantom rollout in the digest, and a
            # phantom rollout in the evidence is exactly the class of defect this
            # whole pass exists to remove.
            if not sd.is_dir():
                continue
            diff = sd / "DIFF.txt"
            if diff.exists():
                L += [f"## Scorer diff — {sd.name} (authoritative)", "",
                      "```", clip(diff.read_text(), 2500, 800), "```", ""]
            # What the SOLVER itself changed — precomputed, so nobody has to
            # pair .xlsx files by hand. On 2026-08-01 two consecutive evolvers
            # and then the orchestrator all did that pairing wrong and reached
            # confident opposite conclusions from it. A reader who must work out
            # which of three identically-named workbooks is which will
            # eventually get it wrong; the fix is to stop asking.
            L += _solver_edit_section(sd)

    for i in sorted(samples):
        r = samples[i]
        verdict = "PASS" if (r["score"] or 0) >= 0.999 else "FAIL"
        L += [f"## Rollout s{i} — {verdict} ({r['turns']} turns"
              + (", HIT TURN CAP" if r["exhausted"] else "") + ")", ""]
        for m in r["messages"]:
            role = m.get("role")
            if role == "system":
                continue
            txt = content_text(m.get("content"))
            if not txt.strip():
                continue
            if role == "tool":
                # Collapse BEFORE clipping: clip() would otherwise cut the
                # frontmatter marker this matches on, for any skill whose body
                # exceeds the head budget.
                txt = _collapse_skill_echo(txt)
                txt = clip(txt)
            L += [f"### {role}", "", txt.strip(), ""]

    p = OUT / f"{safe}.md"
    p.write_text("\n".join(L))
    index.append((tid, npass, len(scores), p.stat().st_size))

# Assert the firewall held, rather than trusting the filter above. A digest set
# that quietly contains one val task would contaminate every downstream edit.
leaked = sorted({tid for tid, *_ in index} & forbidden)
if leaked:
    raise SystemExit(
        f"VAL FIREWALL BREACH: {len(leaked)} task(s) from another split reached "
        f"the digests: {leaked[:5]}"
    )

print(f"unresolved sample_idx: {unresolved}")
if allowed is not None:
    print(f"excluded (other splits): {skipped_other_split} rollouts")
    print(f"firewall OK: 0 of {len(forbidden)} out-of-split tasks present")
    missing = sorted(allowed - {tid for tid, *_ in index})
    if missing:
        print(f"WARNING: {len(missing)} split tasks have NO rollouts: {missing[:5]}")
print(f"{len(index)} task digests -> {OUT}")
for tid, np_, n, sz in index:
    print(f"  {tid:<28} {np_}/{n} pass   {sz//1024}KB")
