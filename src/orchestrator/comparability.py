"""Refuse to answer "did A beat B?" when A and B were not measured the same way.

WHY THIS EXISTS
---------------
`ExperimentSpec._INSTRUMENT` already names every field that changes what an episode
produces, and `cache_tag` keeps records from one instrument out of a leg measured under
another. That protects the CACHE. Nothing protected the CLAIM: nothing stopped a report
saying "v4 26/75 vs base 14/75" when the two legs ran different solver models. #155 is
open for exactly that reason, and #129 (the pin that landed on the non-executing dict)
is what a claim-layer check would have caught before it shipped.

So this module answers a different question from the cache tag:

    cache_tag        "may I REUSE this recorded episode?"
    compare_runs()   "may I CLAIM one of these runs beat the other?"

The answer is a refusal, not a score. `compare_runs` raises `RunsNotComparable` rather
than returning a verdict the declarations do not license.

THREE BUCKETS, AND THE THIRD IS THE POINT
-----------------------------------------
PROTECTED    must match, or the comparison is refused.
REPORTED     may differ; returned so the receipt states what varied.
UNVERIFIABLE facts no artifact records, named explicitly.

The third bucket is why this is not just a dict diff. A check that stays silent about
what it cannot see is the vacuous guard that #154/#227/#228/#248 all were. Two rules
follow, and both are load-bearing:

  * A protected field ABSENT from either spec is NOT a match. Absence is reported as
    unverifiable and refuses by default. "Neither run recorded it" is the one case
    where a naive diff says "equal" and means "blind".
  * `_UNVERIFIABLE_ALWAYS` names facts that no run dir can settle no matter how new it
    is, so a caller cannot mistake a clean pass for a complete one.

TREATMENT
---------
`treatment=` names the fields the caller is deliberately varying. Those are exempted
from the protected check AND asserted to have actually differed — a declared treatment
that did not vary means the comparison measured nothing, which is #120 at the claim
layer. Declaring a protected field as treatment is allowed and is recorded in the
result, so the receipt shows the exemption instead of hiding it.

_INSTRUMENT is imported, never re-listed. A second copy of that tuple would be the
two-build-sites defect (the pin that was set on one dict and read from another).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.orchestrator.spec import ExperimentSpec


class RunsNotComparable(Exception):
    """The recorded declarations do not license a comparison between these two runs."""


class VacuousComparison(RunsNotComparable):
    """A field declared as `treatment` did not actually differ, so nothing was varied."""


class SpecUnavailable(RunsNotComparable):
    """A run directory carries no readable SPEC.json, so nothing about it is checkable."""


# ── what must match ──────────────────────────────────────────────────────────────────
# The instrument (what an episode IS) plus three fields that change what a COMPARISON
# means without changing any single episode. Each is argued, because an unargued entry
# here produces false refusals, and a gate that cries wolf gets bypassed — which is
# worse than no gate at all.
_PROTECTED_BEYOND_INSTRUMENT = (
    # Two runs that drew from different split files may not share a test set at all.
    "split_sha256",
    # A sealed test number and an unsealed one do not mean the same thing (§SEL-2).
    "test_sealed",
    # Our gate counts DISCORDANT PAIRS (standing rule: >=6 for any p<0.05). A 3-sample
    # and a 5-sample estimate have different power, so this is "effective budget".
    "n_samples",
)

PROTECTED: tuple[str, ...] = tuple(ExperimentSpec._INSTRUMENT) + _PROTECTED_BEYOND_INSTRUMENT

# Facts NO run directory records, at any version. Named so that a pass is never read as
# a clean bill of health on them. Extend this list rather than letting silence imply
# verification.
_UNVERIFIABLE_ALWAYS: tuple[tuple[str, str], ...] = (
    (
        "served_provider",
        "#169: only the REQUESTED pin (solver_provider_order / solver_allow_fallbacks) "
        "is recorded. What OpenRouter actually served is in no artifact, so a matching "
        "pin does not prove the two runs were served by the same provider.",
    ),
    (
        "scorer_version",
        "BENCHMARK.json records `scoring_desc` as prose naming a file:line, not a "
        "content hash. Two runs can name the same scorer and have run different bytes.",
    ),
)

_MISSING = object()


def _spec_path(run_dir: Path | str) -> Path:
    return Path(run_dir) / "SPEC.json"


def load_spec(run_dir: Path | str) -> Mapping[str, Any]:
    """The `spec` block of a run directory's SPEC.json.

    Raises SpecUnavailable rather than returning {} — an empty mapping would make every
    protected field 'absent on both sides', which is precisely the blind-equals-equal
    failure this module exists to prevent.
    """
    p = _spec_path(run_dir)
    if not p.is_file():
        raise SpecUnavailable(f"no SPEC.json at {p}")
    try:
        doc = json.loads(p.read_text())
    except (OSError, ValueError) as e:
        raise SpecUnavailable(f"{p} is unreadable: {type(e).__name__}: {e}") from e
    spec = doc.get("spec")
    if not isinstance(spec, dict) or not spec:
        raise SpecUnavailable(f"{p} has no non-empty 'spec' block (keys: {sorted(doc)})")
    return spec


def _norm(v: Any) -> Any:
    """Compare by value, not by JSON incidentals. Lists and tuples are the same order."""
    if isinstance(v, (list, tuple)):
        return tuple(_norm(x) for x in v)
    return v


@dataclass(frozen=True)
class Comparison:
    """A comparison the recorded declarations license. Only built when nothing refused."""

    a_name: str
    b_name: str
    matched: dict[str, Any] = field(default_factory=dict)
    treatment: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    reported: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    unverifiable: tuple[tuple[str, str], ...] = ()
    exempted: tuple[str, ...] = ()

    def render(self) -> str:
        L = [f"COMPARABLE: {self.a_name}  vs  {self.b_name}", ""]
        L.append(f"  protected fields matched ({len(self.matched)}):")
        for k in sorted(self.matched):
            L.append(f"    = {k:26} {self.matched[k]!r}")
        if self.treatment:
            L.append("")
            L.append("  declared treatment (verified to differ):")
            for k in sorted(self.treatment):
                a, b = self.treatment[k]
                L.append(f"    ~ {k:26} {a!r}  ->  {b!r}")
        if self.exempted:
            L.append("")
            L.append("  PROTECTED fields exempted by caller declaration:")
            for k in sorted(self.exempted):
                L.append(f"    ! {k}")
        if self.reported:
            L.append("")
            L.append(f"  differing, not protected ({len(self.reported)}):")
            for k in sorted(self.reported):
                a, b = self.reported[k]
                L.append(f"    · {k:26} {a!r}  ->  {b!r}")
        L.append("")
        L.append("  NOT VERIFIED — no artifact records these:")
        for name, why in self.unverifiable:
            L.append(f"    ? {name}")
            L.append(f"        {why}")
        return "\n".join(L)


def compare_runs(
    a_dir: Path | str,
    b_dir: Path | str,
    *,
    treatment: Iterable[str] = (),
    allow_unrecorded: bool = False,
) -> Comparison:
    """Refuse, or return the licensed comparison.

    treatment          fields deliberately varied. Exempt from the protected check and
                       REQUIRED to differ (VacuousComparison if they did not).
    allow_unrecorded   permit a protected field that neither spec records. Default False:
                       an unrecorded field is unverified, not equal.
    """
    treatment = tuple(dict.fromkeys(treatment))  # de-dup, keep order
    sa, sb = load_spec(a_dir), load_spec(b_dir)
    a_name = str(sa.get("name") or Path(a_dir).name)
    b_name = str(sb.get("name") or Path(b_dir).name)

    unknown = [t for t in treatment if t not in sa and t not in sb]
    if unknown:
        raise RunsNotComparable(
            f"treatment names {unknown} which appear in neither spec — a typo here would "
            f"silently exempt nothing and re-protect the field you meant to vary"
        )

    differing: list[str] = []
    unrecorded: list[str] = []
    matched: dict[str, Any] = {}

    for f in PROTECTED:
        if f in treatment:
            continue
        va, vb = sa.get(f, _MISSING), sb.get(f, _MISSING)
        if va is _MISSING or vb is _MISSING:
            where = ("a" if va is _MISSING else "") + ("b" if vb is _MISSING else "")
            unrecorded.append(f"{f} (absent from: {where})")
            continue
        if _norm(va) != _norm(vb):
            differing.append(f"{f}: {va!r} != {vb!r}")
        else:
            matched[f] = va

    problems: list[str] = []
    if differing:
        problems.append(
            "protected declarations differ, so a winner or tie would compare two "
            "different measurements:\n    " + "\n    ".join(differing)
        )
    if unrecorded and not allow_unrecorded:
        problems.append(
            "protected declarations NOT RECORDED — absence is not a match:\n    "
            + "\n    ".join(unrecorded)
            + "\n  (pass allow_unrecorded=True to accept this explicitly)"
        )
    if problems:
        raise RunsNotComparable(
            f"{a_name} vs {b_name}\n  " + "\n  ".join(problems)
        )

    # A declared treatment that did not vary means the comparison measured nothing.
    tre: dict[str, tuple[Any, Any]] = {}
    inert: list[str] = []
    for f in treatment:
        va, vb = sa.get(f, _MISSING), sb.get(f, _MISSING)
        if va is not _MISSING and vb is not _MISSING and _norm(va) == _norm(vb):
            inert.append(f"{f} (both {va!r})")
        else:
            tre[f] = (None if va is _MISSING else va, None if vb is _MISSING else vb)
    if inert:
        raise VacuousComparison(
            f"{a_name} vs {b_name}: declared treatment did not vary:\n    "
            + "\n    ".join(inert)
            + "\n  Nothing was manipulated, so the difference in outcome has no "
              "declared cause."
        )

    reported: dict[str, tuple[Any, Any]] = {}
    for f in sorted(set(sa) | set(sb)):
        # `name` differs for ANY two distinct runs, so listing it carries no information
        # and only dilutes a list whose whole job is to be worth reading. It is already
        # on the header line as a_name / b_name.
        if f in PROTECTED or f in treatment or f == "name":
            continue
        va, vb = sa.get(f, _MISSING), sb.get(f, _MISSING)
        if _norm(va) != _norm(vb):
            reported[f] = (None if va is _MISSING else va, None if vb is _MISSING else vb)

    unver = list(_UNVERIFIABLE_ALWAYS)
    if unrecorded:  # reached only under allow_unrecorded=True
        unver.append(("unrecorded_protected_fields", "; ".join(unrecorded)))

    return Comparison(
        a_name=a_name,
        b_name=b_name,
        matched=matched,
        treatment=tre,
        reported=reported,
        unverifiable=tuple(unver),
        exempted=tuple(f for f in treatment if f in PROTECTED),
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="comparability",
        description="Refuse or license a comparison between two run directories.",
    )
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument(
        "--treatment", default="",
        help="comma-separated fields deliberately varied (exempt from the protected "
             "check, and required to actually differ)",
    )
    ap.add_argument("--allow-unrecorded", action="store_true")
    ns = ap.parse_args(argv)

    tre = tuple(t.strip() for t in ns.treatment.split(",") if t.strip())
    try:
        c = compare_runs(ns.run_a, ns.run_b, treatment=tre,
                         allow_unrecorded=ns.allow_unrecorded)
    except RunsNotComparable as e:
        print(f"REFUSED — {type(e).__name__}\n  {e}")
        return 2
    print(c.render())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
