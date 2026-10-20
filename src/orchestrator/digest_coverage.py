"""Evidence-surface COVERAGE, re-derived from the digest bytes.

WHY THIS EXISTS. A digest surface can be half-evidenced and still look complete: when the base leg
is a cache REPLAY, no workbook snapshot is written, so a builder that skips missing snapshots
silently leaves many samples without a scorer diff. `_v_digests_train_only` checks MEMBERSHIP (no
val, no test, no stray), not COMPLETENESS, so such a surface would pass every other gate while the
prompt tells the evolver that every file holds "the scorer's own diff".

WHAT THIS CHECKS, per expected train task (split id with '/' -> '_', the map
`_v_digests_train_only` uses) and per sample K in 0..N-1, N = spec.n_samples:

  both benchmarks  exactly one rollout header `## Rollout sK — PASS|FAIL ...`;
  ssbv2            exactly one `## Scorer diff — iter_<L>[_midgate]_sK (authoritative)` heading in
                   the region BEFORE the first rollout header. The region matters: rollout
                   sections paste clipped tool output verbatim, so the heading text can appear
                   inside a rollout, and counting it there would let a transcript satisfy the
                   check. The builder's '## NO scorer diff — sK' marker is, by design, NOT a diff;
                   it only changes how the gap is described ("builder marker" vs "silent").
  tau3             exactly one `### Grade (authoritative — ...)` heading between the rollout
                   header and that rollout's `### Transcript`, and not the no-verdict marker.
                   `ungraded(<reason>)` is a legitimate OUTCOME (the evaluator ran no check,
                   e.g. a timeout), not an evidence gap, and is not counted.

WHY A SEPARATE MODULE WITH NO PACKAGE IMPORTS. The loop calls it for its `_v_digests_complete`
step; tests/test_digest_coverage.py loads it BY PATH, so a mutation harness can mutate a throwaway
copy and prove the tests go red without touching the live file (the loop is invoked between epochs
of real runs).

IT TAKES A LIST OF DIRS. By default the loop passes [the current epoch's digests]. When the evolver is
granted every earlier epoch's surface too, the same call takes all of them — an earlier surface
that was never complete must not be re-served to every later epoch unchecked.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

SIDECARS = frozenset({"REGRESSIONS", "README", "INDEX"})

_ROLLOUT = {
    # build_rollout_digests.py: f"## Rollout s{i} — {verdict} ({turns} turns..."
    "ssbv2": re.compile(r"(?m)^## Rollout s(\d+) — (?:PASS|FAIL) \("),
    # tau3_build_digests.py: f"## Rollout s{i} — {verdict} [kind=...] (score=..."
    "tau3": re.compile(r"(?m)^## Rollout s(\d+) — (?:PASS|FAIL) \[kind="),
}
# The builder's exact heading. `(?:_midgate)?` because snapshot_output() labels mid-gate re-evals
# iter_<N>_midgate_sK; a sample carrying BOTH is two diffs for one rollout and is ambiguous.
SSB_DIFF = re.compile(r"(?m)^## Scorer diff — iter_\d+(?:_midgate)?_s(\d+) \(authoritative\)$")
SSB_NO_DIFF_MARKER = re.compile(r"(?m)^## NO scorer diff — s(\d+) \(")
T3_GRADE = re.compile(r"(?m)^### Grade \(authoritative — this is what produced the score\)$")
T3_TRANSCRIPT = re.compile(r"(?m)^### Transcript$")
T3_NO_VERDICT = "_No structured verdict on this trace"

# What to do about each kind of gap. The loop's HALT prints the lines for the kinds it found.
REMEDY = {
    "no digest file": (
        "the cache holds no rollout for these train tasks, so the leg behind this surface is "
        "incomplete. Finish or resume that leg, move the digest dir aside, and rebuild."),
    "no rollout": (
        "the cache holds fewer than N samples for these tasks, so the leg behind this surface is "
        "incomplete. Finish or resume that leg, move the digest dir aside, and rebuild."),
    "duplicate rollout": (
        "one sample is rendered twice. The surface was edited or merged by hand; rebuild it from "
        "the cache."),
    "rollout beyond N": (
        "the surface holds more samples than the spec's n_samples. It was built with a different "
        "N; rebuild it with the spec's."),
    "no scorer diff": (
        "the produced-workbook snapshots for these samples were not found when the surface was "
        "built. A cache-REPLAYED leg writes none, so they live under the tag that first scored "
        "the rollout. Declare every snapshot root that holds them in RUNTIME.json "
        "`digest_snapshot_roots` (each a .cache/ssbv2_<tag>_tasks/_snapshots), move the digest "
        "dir OUT of the run dir (keep it, and record that an earlier evolver saw the partial "
        "surface), then re-run --step to rebuild."),
    "ambiguous scorer diff": (
        "one sample has two scorer diffs (e.g. iter_0_s0 and iter_0_midgate_s0), so which one "
        "produced the score is undecidable. Resolve it at the snapshot root, then rebuild."),
    "no grade block": (
        "a rollout has no grade section. The surface was not written by the current builder; "
        "rebuild it."),
    "duplicate grade block": (
        "a rollout has two grade sections. The surface was edited or merged by hand; rebuild "
        "it."),
    "no structured verdict": (
        "the evaluator produced no verdict for these episodes, so their scores are unexplained. "
        "Re-grade or re-run them, move the digest dir aside, and rebuild."),
}


def benchmark_rule(benchmark: str) -> str | None:
    """The coverage rule for a spec.benchmark, or None when there is none (the caller refuses)."""
    return benchmark if benchmark in _ROLLOUT else None


def train_stems(split_path) -> list[str]:
    """The expected task files: split['train'] in FILENAME form, the map the builders write."""
    sp = json.loads(Path(split_path).read_text())
    v = sp.get("train") or []
    ids = [x if isinstance(x, str) else x["id"] for x in v]
    return sorted({i.replace("/", "_") for i in ids})


def file_gaps(text: str, n: int, benchmark: str) -> list[tuple[str, int]]:
    """Every (gap kind, sample) in ONE digest's text. [] means complete."""
    ro = _ROLLOUT[benchmark]
    heads = list(ro.finditer(text))
    gaps: list[tuple[str, int]] = []
    seen: dict[int, int] = {}
    for m in heads:
        k = int(m.group(1))
        seen[k] = seen.get(k, 0) + 1
    for k in range(n):
        c = seen.get(k, 0)
        if c == 0:
            gaps.append(("no rollout", k))
        elif c > 1:
            gaps.append(("duplicate rollout", k))
    for k in sorted(seen):
        if k >= n:
            gaps.append(("rollout beyond N", k))

    if benchmark == "ssbv2":
        pre = text[: heads[0].start()] if heads else text
        diffs: dict[int, int] = {}
        for x in SSB_DIFF.findall(pre):
            diffs[int(x)] = diffs.get(int(x), 0) + 1
        for k in range(n):
            c = diffs.get(k, 0)
            if c == 0:
                gaps.append(("no scorer diff", k))
            elif c > 1:
                gaps.append(("ambiguous scorer diff", k))
    else:
        for j, m in enumerate(heads):
            k = int(m.group(1))
            end = heads[j + 1].start() if j + 1 < len(heads) else len(text)
            sec = text[m.start():end]
            t = T3_TRANSCRIPT.search(sec)
            head = sec[: t.start()] if t else sec
            g = len(T3_GRADE.findall(head))
            if g == 0:
                gaps.append(("no grade block", k))
            elif g > 1:
                gaps.append(("duplicate grade block", k))
            elif T3_NO_VERDICT in head:
                gaps.append(("no structured verdict", k))
    return gaps


def dir_coverage(d: Path, stems, n: int, benchmark: str) -> dict:
    """Coverage of one digest dir against the expected stems. Reads bytes, decodes leniently:
    a digest holding a stray control byte is still a digest (plain grep once called one binary
    and under-counted a complete surface)."""
    d = Path(d)
    gaps: list[tuple[str, str, int]] = []                  # (stem, kind, sample)
    marker: set[tuple[str, int]] = set()                   # samples the builder marked itself
    present = 0
    for stem in stems:
        f = d / f"{stem}.md"
        if not f.is_file():
            gaps += [(stem, "no digest file", k) for k in range(n)]
            continue
        present += 1
        text = f.read_bytes().decode("utf-8", errors="replace")
        fg = file_gaps(text, n, benchmark)
        gaps += [(stem, kind, k) for kind, k in fg]
        if benchmark == "ssbv2":
            ro = _ROLLOUT[benchmark].search(text)
            pre = text[: ro.start()] if ro else text
            marker |= {(stem, int(x)) for x in SSB_NO_DIFF_MARKER.findall(pre)}
    bad_samples = {(s, k) for s, _kind, k in gaps if k < n}
    return {
        "dir": str(d),
        "exists": d.is_dir(),
        "tasks_expected": len(stems),
        "tasks_present": present,
        "samples_expected": len(stems) * n,
        "samples_complete": len(stems) * n - len(bad_samples),
        "tasks_with_gaps": len({s for s, _k, _i in gaps}),
        "gaps": gaps,
        "marked_by_builder": marker,
    }


def _summary(cov: dict, n: int, benchmark: str) -> str:
    name = Path(cov["dir"]).name
    what = "rollout + scorer diff" if benchmark == "ssbv2" else "rollout + grade"
    if not cov["gaps"]:
        return (f"{name}: {cov['tasks_present']}/{cov['tasks_expected']} train task files, "
                f"{cov['samples_complete']}/{cov['samples_expected']} samples complete "
                f"(N={n}; {what} each)")
    kinds: dict[str, int] = {}
    for _s, kind, _k in cov["gaps"]:
        kinds[kind] = kinds.get(kind, 0) + 1
    first = []
    for s, kind, k in cov["gaps"][:4]:
        tag = ""
        if kind == "no scorer diff":
            tag = (" (builder marker)" if (s, k) in cov["marked_by_builder"]
                   else " (silent: no marker)")
        first.append(f"{s} s{k} {kind}{tag}")
    bad = cov["samples_expected"] - cov["samples_complete"]
    return (f"EVIDENCE GAPS in {name}: {bad} of {cov['samples_expected']} samples "
            f"({cov['tasks_with_gaps']} of {cov['tasks_expected']} task files) — "
            + ", ".join(f"{k} x{v}" for k, v in sorted(kinds.items()))
            + f". First: {'; '.join(first)}")


def check(dirs, stems, n: int, benchmark: str) -> tuple[bool, str]:
    """(ok, detail) over EVERY dir given. A missing dir, an unknown benchmark, an empty expected
    set or a non-positive N is a failure, never a vacuous pass."""
    dirs = [Path(d) for d in dirs]
    if not dirs:
        return False, "no evidence dir to check — coverage of nothing is not coverage"
    rule = benchmark_rule(benchmark)
    if rule is None:
        return False, (f"no coverage rule for benchmark {benchmark!r} (known: "
                       f"{sorted(_ROLLOUT)}). Refusing to pass a surface this check cannot read.")
    if n < 1:
        return False, f"n_samples={n}: nothing could be checked"
    stems = sorted(stems)
    if not stems:
        return False, "the split names no train task, so there is nothing to check against"
    lines, kinds, ok = [], set(), True
    for d in dirs:
        if not d.is_dir():
            ok = False
            lines.append(f"no digest dir at {d}")
            continue
        cov = dir_coverage(d, stems, n, rule)
        lines.append(_summary(cov, n, rule))
        if cov["gaps"]:
            ok = False
            kinds |= {kind for _s, kind, _k in cov["gaps"]}
    detail = "; ".join(lines)
    if not ok and kinds:
        detail += "\n  REMEDY: " + "\n  REMEDY: ".join(REMEDY[k] for k in sorted(kinds))
        detail += ("\n  A gap is a hole in the evidence the evolver reads, not a verdict. Fix it "
                   "at the source and rebuild; never edit a digest to satisfy this check.")
    return ok, detail
