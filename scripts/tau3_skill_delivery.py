#!/usr/bin/env python3
"""Did the intervention actually reach the agent? Uptake + invalid-name accounting for a leg.

Arm C of the banking ablation ran four legs, scored normally, and produced a null that looked like
a result. It was not: the solver consulted its skill library in 2 of 75 rollouts (3%), against
85-96% in every other arm. Nothing in the pipeline noticed. The legs ran, the gate computed, and
"skills did not help ling" was the reading — when the truth was that the skills were never read.

    arm / version        skills    bytes     rollouts that consulted a skill
    A a_sonnet v1             5   10,662     72/75  =  96%
    B nexmini  v1             6   17,031     64/75  =  85%
    C lingt0   v1             6   27,108      2/75  =   3%    <- never delivered
    retail     v3             2   11,679     84/90  =  93%

It also counts skill-tool calls that named something that is not in the library. Those are silently
discarded today and nobody counts them:

    B nexmini v1   3x "write-only-what-asked"      a TYPO of write-only-what-was-asked
    C lingt0  v1   1x "open_bank_account_4821"     a DOMAIN TOOL name — a category error

A typo means the library was reaching the agent and one call missed. A domain-tool name means the
agent did not understand what the tool is. Reporting them separately is the point.

Run this after epoch 1 of every arm, BEFORE spending epochs 2-4. A low-uptake arm cannot produce an
interpretable result at any epoch, so the epochs are wasted whatever the skills say.

Usage:
    python3 scripts/tau3_skill_delivery.py --skills outputs/ARM/skills_v1 \\
                                           --digests outputs/ARM/digests_e2_train
    python3 scripts/tau3_skill_delivery.py --selftest
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

# a skill-tool call in a digest, with the name it asked for
CALL = re.compile(r'\(tool\.skill\)[^\n]*?"name"\s*:\s*"([^"]+)"')
# a skill-tool call whose name argument we could not parse — still evidence the tool was reached
BARE = re.compile(r"\(tool\.skill\)")
# tau3 domain tools carry a numeric suffix; a skill name never does
DOMAIN_TOOLish = re.compile(r"^[a-z_]+_\d{3,}$")

# Below this, the arm's result is about delivery, not about skills. Not a hard threshold — it is
# the line at which the report says so out loud.
UPTAKE_FLOOR = 0.50


def library(skills_dir: Path) -> dict[str, int]:
    return {p.parent.name: p.stat().st_size for p in skills_dir.rglob("SKILL.md")}


def _sections(digests_dir: Path):
    """Every rollout section across the digest set.

    Digest filenames differ by benchmark — banking uses `task_004.md`, retail uses `36.md`. Globbing
    `task_*.md` silently returns nothing on retail and reads as 0% uptake, which is how I produced a
    false '0/0' once. Glob `*.md` and drop the known non-task file instead.
    """
    for f in sorted(digests_dir.glob("*.md")):
        if f.stem == "REGRESSIONS":
            continue
        for sec in f.read_text(errors="replace").split("## Rollout ")[1:]:
            yield sec


def assess(lib: dict[str, int], digests_dir: Path) -> dict:
    rollouts = 0
    consulted = 0
    valid = 0
    per_skill = collections.Counter()
    typos: collections.Counter = collections.Counter()
    category_errors: collections.Counter = collections.Counter()

    for sec in _sections(digests_dir):
        rollouts += 1
        names = CALL.findall(sec)
        hit = False
        for n in names:
            if n in lib:
                valid += 1
                per_skill[n] += 1
                hit = True
            elif DOMAIN_TOOLish.match(n):
                category_errors[n] += 1
            else:
                typos[n] += 1
        if not names and BARE.search(sec):
            hit = True  # reached the tool, name unparseable — count the rollout, not a name
        if hit:
            consulted += 1

    rate = consulted / rollouts if rollouts else 0.0
    return {
        "skills": len(lib), "bytes": sum(lib.values()),
        "rollouts": rollouts, "consulted": consulted, "uptake": rate,
        "valid_calls": valid, "per_skill": dict(per_skill),
        "typos": dict(typos), "category_errors": dict(category_errors),
        "never_loaded": sorted(n for n in lib if n not in per_skill),
        "delivered": rate >= UPTAKE_FLOOR,
    }


def render(a: dict) -> str:
    L = ["SKILL DELIVERY", ""]
    L.append(f"  library     {a['skills']} skills, {a['bytes']:,} B")
    L.append(f"  uptake      {a['consulted']}/{a['rollouts']} rollouts consulted a skill "
             f"= {a['uptake']:.0%}")
    L.append(f"  valid calls {a['valid_calls']}")
    if a["typos"]:
        L.append(f"  NAME TYPOS  {a['typos']}   (library WAS reaching the agent; the call missed)")
    if a["category_errors"]:
        L.append(f"  CATEGORY ERRORS {a['category_errors']}   (a DOMAIN TOOL name was passed to the "
                 f"skill tool — the agent does not understand what this tool is)")
    if a["never_loaded"]:
        L.append(f"  never loaded: {a['never_loaded']}")
    L.append("")
    if a["delivered"]:
        L.append(f"  DELIVERED — uptake is at or above {UPTAKE_FLOOR:.0%}. A null from this arm is "
                 f"about the skills.")
    else:
        L.append(f"  *** NOT DELIVERED *** — uptake below {UPTAKE_FLOOR:.0%}. Any result from this "
                 f"arm is about the DELIVERY CHANNEL, not about the skills. Further epochs cannot "
                 f"be interpreted until this is fixed.")
    return "\n".join(L)


def selftest() -> int:
    """Fixtures with a known-POSITIVE and a known-NEGATIVE.

    A detector validated only against the broken case is untested: if the parser stops matching
    anything, an all-negative suite still passes. That is precisely how I misread `trace.tools` as
    'no skill tool offered' — the probe returned empty for the healthy arm too, and only a control
    revealed it. So both directions are asserted here.
    """
    import tempfile
    bad = 0
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        sk = root / "skills"
        for n in ("alpha-skill", "beta-skill"):
            (sk / n).mkdir(parents=True)
            (sk / n / "SKILL.md").write_text("---\nname: %s\ndescription: d\n---\nbody\n" % n)
        lib = library(sk)

        def digests(name, secs):
            d = root / name
            d.mkdir()
            (d / "task_001.md").write_text("".join(f"## Rollout {i}\n{s}\n" for i, s in enumerate(secs)))
            return d

        healthy = digests("healthy", ['(tool.skill): {"name": "alpha-skill"}'] * 9 + ["no tools here"])
        starved = digests("starved", ["no tools here"] * 9 + ['(tool.skill): {"name": "beta-skill"}'])
        typo = digests("typo", ['(tool.skill): {"name": "alpha-skil"}'] * 3 + ["x"] * 7)
        cat = digests("cat", ['(tool.skill): {"name": "open_bank_account_4821"}'] * 3 + ["x"] * 7)
        # retail-style filenames must be found too — this is the glob bug that produced a false 0%
        rd = root / "intnames"
        rd.mkdir()
        (rd / "36.md").write_text('## Rollout 0\n(tool.skill): {"name": "alpha-skill"}\n')

        cases = [
            ("healthy arm  -> delivered", healthy, True, 0.9),
            ("starved arm  -> NOT delivered", starved, False, 0.1),
        ]
        for label, d, want_del, want_rate in cases:
            r = assess(lib, d)
            ok = r["delivered"] == want_del and abs(r["uptake"] - want_rate) < 1e-9
            bad += (not ok)
            print(f"  {'ok  ' if ok else 'BAD '} {label:32s} uptake {r['uptake']:.0%} "
                  f"delivered={r['delivered']}")

        r = assess(lib, typo)
        ok = r["typos"] == {"alpha-skil": 3} and not r["category_errors"]
        bad += (not ok)
        print(f"  {'ok  ' if ok else 'BAD '} {'typo -> typos, not category':32s} {r['typos']}")

        r = assess(lib, cat)
        ok = r["category_errors"] == {"open_bank_account_4821": 3} and not r["typos"]
        bad += (not ok)
        print(f"  {'ok  ' if ok else 'BAD '} {'domain-tool name -> category':32s} {r['category_errors']}")

        r = assess(lib, rd)
        ok = r["rollouts"] == 1 and r["consulted"] == 1
        bad += (not ok)
        print(f"  {'ok  ' if ok else 'BAD '} {'<int>.md digests are found':32s} "
              f"{r['consulted']}/{r['rollouts']}")

        r = assess(lib, healthy)
        ok = r["never_loaded"] == ["beta-skill"]
        bad += (not ok)
        print(f"  {'ok  ' if ok else 'BAD '} {'never-loaded skills listed':32s} {r['never_loaded']}")

    print(f"\n  6 checks, {bad} bad")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skills", type=Path)
    ap.add_argument("--digests", type=Path)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--strict", action="store_true", help="exit 1 if the library was not delivered")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.skills or not a.digests:
        ap.error("--skills and --digests are both required")
    lib = library(a.skills)
    if not lib:
        ap.error(f"no SKILL.md under {a.skills}")
    res = assess(lib, a.digests)
    print(json.dumps(res, indent=1) if a.json else render(res))
    return 1 if (a.strict and not res["delivered"]) else 0


if __name__ == "__main__":
    sys.exit(main())
