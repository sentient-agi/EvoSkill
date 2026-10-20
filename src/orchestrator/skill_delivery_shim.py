"""Skill-uptake assessment that knows which DIALECT of transcript it is reading.

WHY THIS FILE EXISTS. `loop.py::_v_skills_were_consulted` imports `assess_delivery` from here. A
reader that knows only one transcript dialect recognises exactly one notation for "the agent opened
a skill":

    CALL = re.compile(r'\\(tool\\.skill\\)[^\\n]*?"name"\\s*:\\s*"([^"]+)"')

which is tau3's. SSB v2 digests record a skill load as a tool-result line:

    [SKILL LOADED: excel-edit-protocol — 6169 B. Body omitted here; ...]

So on an SSB v2 surface the tau3 patterns match nothing, `consulted` stays 0, and the check
reports **0/180 rollouts consulted a skill = 0% — the library was NOT DELIVERED**, on a surface
where every rollout loaded a skill.

Note what would make such a false verdict convincing: `rollouts` was RIGHT (180), because sections are
split on `## Rollout `, which both benchmarks use. Only the numerator was blind. A reader that
gets the denominator right looks like a reader that works.

THE RULE THIS ENCODES. A parser that finds none of ITS OWN markers on a surface has not measured
0% — it has failed to read. Those are different facts and they must not share a return value.
A reader applied to the wrong format must declare the branch and refuse rather than default, and a
check must be able both to pass and to fail. So: name the dialect, and when no dialect is
recognisable, say UNREADABLE rather than emitting a number.

Adding a benchmark means adding a DIALECT entry, and the selftest asserts each dialect's fixture
is parsed by exactly one entry, so a new pattern that accidentally swallows another benchmark's
notation fails loudly here rather than silently downstream.
"""
from __future__ import annotations

import collections
import re
from pathlib import Path

# Below this, the arm's result is about delivery, not about skills.
UPTAKE_FLOOR = 0.50

# tau3 domain tools carry a numeric suffix; a skill name never does.
DOMAIN_TOOLish = re.compile(r"^[a-z_]+_\d{3,}$")


class Dialect:
    """One transcript notation for 'the agent opened a skill'."""

    def __init__(self, name: str, call: str, bare: str):
        self.name = name
        self.call = re.compile(call)
        self.bare = re.compile(bare)


DIALECTS = (
    # tau3 / tau2: a tool call rendered with its JSON arguments.
    Dialect("tau3", r'\(tool\.skill\)[^\n]*?"name"\s*:\s*"([^"]+)"', r"\(tool\.skill\)"),
    # SSB v2: a tool RESULT line naming the skill and its byte size. The name is the first
    # token after the colon; the em dash and size follow. Anchored on the literal opening
    # bracket so prose mentioning a skill by name is never counted as a load.
    Dialect("ssbv2", r"\[SKILL LOADED:\s*([A-Za-z0-9][A-Za-z0-9._-]*)", r"\[SKILL LOADED:"),
)


def library(skills_dir: Path) -> dict[str, int]:
    return {p.parent.name: p.stat().st_size for p in Path(skills_dir).rglob("SKILL.md")}


def _sections(digests_dir: Path):
    """Every rollout section across the digest set.

    Digest filenames differ by benchmark (`task_004.md`, `36.md`, `Debugging_02_01.md`), so glob
    `*.md` and drop the known non-task files rather than guessing a stem pattern — globbing
    `task_*.md` once returned nothing on retail and read as 0% uptake.
    """
    for f in sorted(Path(digests_dir).glob("*.md")):
        if f.stem in ("REGRESSIONS", "README", "INDEX"):
            continue
        for sec in f.read_text(errors="replace").split("## Rollout ")[1:]:
            yield sec


def assess(lib: dict[str, int], digests_dir: Path) -> dict:
    rollouts = 0
    consulted = 0
    valid = 0
    per_skill: collections.Counter = collections.Counter()
    typos: collections.Counter = collections.Counter()
    category_errors: collections.Counter = collections.Counter()
    seen_dialects: collections.Counter = collections.Counter()

    for sec in _sections(digests_dir):
        rollouts += 1
        hit = False
        for d in DIALECTS:
            names = d.call.findall(sec)
            bare = bool(d.bare.search(sec))
            if names or bare:
                seen_dialects[d.name] += 1
            for n in names:
                if n in lib:
                    valid += 1
                    per_skill[n] += 1
                    hit = True
                elif DOMAIN_TOOLish.match(n):
                    category_errors[n] += 1
                else:
                    typos[n] += 1
            if not names and bare:
                # Reached the tool, name unparseable — count the rollout, not a name.
                hit = True
        if hit:
            consulted += 1

    # THE REFUSAL. Sections exist but not one of them carries any dialect's marker: this reader
    # cannot see loads on this surface. Emitting 0.0 here is what produced the false NOT
    # DELIVERED, so emit a fact about the READER instead and let the caller refuse.
    unreadable = rollouts > 0 and not seen_dialects

    rate = consulted / rollouts if rollouts else 0.0
    return {
        "skills": len(lib), "bytes": sum(lib.values()),
        "rollouts": rollouts, "consulted": consulted, "uptake": rate,
        "valid_calls": valid, "per_skill": dict(per_skill),
        "typos": dict(typos), "category_errors": dict(category_errors),
        "never_loaded": sorted(n for n in lib if n not in per_skill),
        "dialect": ",".join(sorted(seen_dialects)) or "none",
        "unreadable": unreadable,
        "delivered": (not unreadable) and rate >= UPTAKE_FLOOR,
    }


def assess_delivery(skills_dir, digests_dir) -> dict:
    return assess(library(Path(skills_dir)), Path(digests_dir))


# ── selftest ─────────────────────────────────────────────────────────────────────────────────
def _selftest() -> int:
    import tempfile

    bad = 0

    def check(label, cond):
        nonlocal bad
        if not cond:
            bad += 1
            print(f"  [BAD] {label}")
        else:
            print(f"  [ok]  {label}")

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        sk = root / "skills"
        for n in ("excel-edit-protocol", "excel-model-completion"):
            (sk / n).mkdir(parents=True)
            (sk / n / "SKILL.md").write_text(f"---\nname: {n}\ndescription: d\n---\nbody\n")
        lib = library(sk)
        check("library reads both skills", set(lib) == {"excel-edit-protocol",
                                                        "excel-model-completion"})

        # ssbv2 dialect: one rollout loads a skill, one does not.
        d1 = root / "d_ssbv2"
        d1.mkdir()
        (d1 / "Debugging_02_01.md").write_text(
            "header\n"
            "## Rollout s0 — FAIL (26 turns)\n"
            "[SKILL LOADED: excel-edit-protocol — 6169 B. Body omitted here; read it there.]\n"
            "## Rollout s1 — FAIL (25 turns)\n"
            "no skill here\n")
        a = assess(lib, d1)
        check("ssbv2: 2 rollouts counted", a["rollouts"] == 2)
        check("ssbv2: 1 consulted", a["consulted"] == 1)
        check("ssbv2: dialect named", a["dialect"] == "ssbv2")
        check("ssbv2: not unreadable", a["unreadable"] is False)
        check("ssbv2: per-skill attributed", a["per_skill"] == {"excel-edit-protocol": 1})

        # The version-anchored marker keeps the NAME as the first token, and the
        # producing library it sits beside (_library/) is a subdir, not a digest.
        d1b = root / "d_ssbv2_anchored"
        (d1b / "_library" / "excel-edit-protocol").mkdir(parents=True)
        (d1b / "_library" / "excel-edit-protocol" / "SKILL.md").write_text(
            "## Rollout s0 — FAIL (a SKILL.md in _library/ must never be read as a digest)\n")
        (d1b / "Debugging_02_01.md").write_text(
            "header\n"
            "## Rollout s0 — FAIL (26 turns)\n"
            "[SKILL LOADED: excel-edit-protocol — 6169 B; library skills_v1 sha256:0123456789ab; "
            "body in ./_library/excel-edit-protocol/SKILL.md of this digest dir]\n"
            "## Rollout s1 — FAIL (25 turns)\n"
            "no skill here\n")
        a2 = assess(lib, d1b)
        check("anchored marker: name parsed, version ignored",
              a2["per_skill"] == {"excel-edit-protocol": 1} and not a2["typos"])
        check("anchored marker: _library/ is not counted as rollouts",
              a2["rollouts"] == 2 and a2["consulted"] == 1)

        # tau3 dialect must still parse through this same reader.
        d2 = root / "d_tau3"
        d2.mkdir()
        (d2 / "task_004.md").write_text(
            "## Rollout s0\n(tool.skill) {\"name\": \"excel-model-completion\"}\n")
        b = assess(lib, d2)
        check("tau3: 1 consulted", b["consulted"] == 1 and b["rollouts"] == 1)
        check("tau3: dialect named", b["dialect"] == "tau3")

        # THE REGRESSION THIS FILE EXISTS FOR: an unrecognised notation must NOT read as 0%.
        d3 = root / "d_alien"
        d3.mkdir()
        (d3 / "t.md").write_text("## Rollout s0\n<<opened skill excel-edit-protocol>>\n")
        c = assess(lib, d3)
        check("alien dialect -> unreadable", c["unreadable"] is True)
        check("alien dialect -> NOT delivered-true", c["delivered"] is False)
        check("alien dialect -> uptake 0 but flagged", c["uptake"] == 0.0)

        # A genuinely empty surface (no rollout sections) is not 'unreadable' — it is empty.
        d4 = root / "d_empty"
        d4.mkdir()
        (d4 / "t.md").write_text("no sections at all\n")
        e = assess(lib, d4)
        check("no sections -> rollouts 0, not unreadable", e["rollouts"] == 0
              and e["unreadable"] is False)

        # Prose naming a skill is not a load.
        d5 = root / "d_prose"
        d5.mkdir()
        (d5 / "t.md").write_text("## Rollout s0\nI should use excel-edit-protocol next time.\n")
        f = assess(lib, d5)
        check("prose mention is not a load", f["consulted"] == 0 and f["unreadable"] is True)

    print("SELFTEST", "FAILED" if bad else "PASSED")
    return 1 if bad else 0


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
