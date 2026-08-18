#!/usr/bin/env python3
"""PreToolUse hook: DENY any evolver read of ANOTHER RUN'S skill set.

WHY THIS EXISTS (#213, found live 2026-08-08 during DSV4F epoch 1).
The evolver's whole job is to derive a skill set from the train evidence. The 0807 Opus
reference run's derived skill sets sit at `outputs/tau3_banking_2525/skills_v{1,2,3,4}/`,
built from THE SAME 25 TRAIN TASKS. Reading them does not leak held-out data — it leaks
the ANSWER, from the very run this arm is supposed to be compared against.

The DSV4F epoch-1 evolver did exactly that, 172 turns in, with three Bash calls:
    ls -la .../tau3_banking_2525/skills_v1 ... skills_v4
    cd .../tau3_banking_2525 && for d in skills_v1..v4; do ls -la $d; done
    cd .../tau3_banking_2525 && for f in skills_v1/*/SKILL.md; do cat "$f"; done
The third one read every accepted skill body. The run was killed and restarted from iter 1
(corrupted evidence => restart, not fix-and-continue).

WHY THE EXISTING GUARDS DID NOT CATCH IT.
  --add-dir            does not confine Bash at all (#186).
  evolver_leak_hook.py guards held-out TASK SPECS. A peer arm's skill file is neither a
                       task spec nor held-out data, so it matched nothing.
  the isolated cwd     only stops CLAUDE.md/skills auto-loading; `cd` anywhere still works.
And the digests grant is INSIDE `outputs/tau3_banking_2525/`, so the reference run's
directory is one `cd ..` from a legitimately granted path. This was reachable by accident,
not by trying.

RELATED, ALREADY ON THE RECORD: Arm A's Sonnet evolver made 3 accesses to
`outputs/ssbv2_evo_0801/skills_v1` (handoff H2) through the same Bash gap. That is a
DIFFERENT benchmark's skills, so it is weaker contamination than reading the same tasks'
answers — but it is the same hole, and this hook closes both.

WHAT IS ALLOWED. The evolver's OWN staging directory, named by EVOSKILL_EVOLVER_STAGING.
That is where it reads the skills it is currently revising and writes its output; blocking
it would make revision impossible.

FAILS CLOSED. With no staging path configured, every skill-directory access is denied
rather than allowed: this guards the validity of the comparison, and a hook that silently
permits is worth less than no hook (#154).

#215 (2026-08-08, DSV4F epoch 2) — TWO FINDINGS, ONE FIXED HERE.

(a) FALSE POSITIVES, fixed below. The matcher ran over every whitespace-token of a Bash
    command, so any occurrence of the WORD "skills" was treated as a path. Two legitimate
    reads of the run's OWN evidence were denied:
        echo "########## 097 skills ##########"        -> token "skills"
        a python heredoc using the identifier `skills_loaded` (the uptake metric)
    Neither is a path. The fix distinguishes the two token shapes that actually name a
    directory — anything containing "/" (checked broadly, as before), and the bare
    `skills_v<N>` convention every peer arm uses — from prose and identifiers. The bare
    form must stay covered without a "/": the original incident's second command was
    `cd <peer-run-dir> && for d in skills_v1 skills_v2; do ls -la $d; done`, where the
    directory names carry no slash and the `cd` is invisible to this hook (it runs as a
    subprocess with its own cwd). Requiring the token to exist on disk would therefore
    reopen the hole, which is why existence is NOT the test.

(b) NOT FIXED HERE, AND NOT FIXABLE HERE. A subagent got past this hook by splitting a
    quoted literal in two so the word never appears whole. A textual guard over command
    strings is a
    speed bump, not a boundary: any model that wants a peer arm's skills can concatenate
    past it, and no regex closes that. The real boundary is filesystem-level — do not
    grant the peer directories in the first place. This hook is worth keeping as a
    tripwire that makes accidental reads loud, but it must NOT be described as isolation.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# For tokens that contain a "/" — i.e. that are unambiguously path references. Broad on
# purpose: `skills`, `skills_v1`, `skills_v1_staging`, `skills_backup_20260505`, ...
_SKILL_DIR_RE = re.compile(r"(?:^|/)skills(?:_[A-Za-z0-9_.-]+)?(?:/|$)")

# For bare tokens with NO "/". Only the versioned convention every derived skill set uses
# (`skills_v1`, `skills_v12_staging`, ...). This is what a `for d in skills_v1 skills_v2`
# loop after a `cd` looks like — the original #213 command. Deliberately does NOT match the
# bare word "skills" or identifiers like `skills_loaded`, which are prose, not paths (#215).
_VERSIONED_SKILL_RE = re.compile(r"^skills_v\d+(?:[_.-][A-Za-z0-9_.-]*)?$")

_FIELDS = ("command", "file_path", "pattern", "path", "glob", "notebook_path", "url")


def _is_peer_skill_token(tok: str) -> bool:
    """True when `tok` names another run's skill directory.

    #217, second attempt. The first attempt asked "is this token anchored like a path?" and
    a property probe over 4000 generated prose paths immediately produced
    `docs/docs/read/grades/skills` — `docs/` exists in this repo, so the token anchored and
    the trailing `/skills` matched. Anchoring is not the discriminator.

    The discriminator is VERSIONING. Every derived skill set this project has ever produced
    is `skills_v<N>` (optionally `_staging`), in every real #213 command:
        cd <peer-run> && for f in skills_v1/*/SKILL.md; do cat "$f"; done
        cd <peer-run> && for d in skills_v1 skills_v2; do ls -la $d; done
        ls -la <peer-run>/skills_v1
    English prose never contains `skills_v1`. So require the version, as a whole path
    component or as the entire token, and the false-positive class disappears without
    dropping a single observed true positive.

    RESIDUAL, stated rather than hidden: a peer directory named plainly `skills/` with no
    version would not match. That is an accepted gap, not an oversight — this hook is a
    tripwire that makes accidental reads loud, and a subagent has already walked past it by
    splitting a string literal. The boundary is `EpochPaths.grants()`, which does not hand
    the evolver any peer run's directory in the first place.
    """
    for part in tok.split("/"):
        if _VERSIONED_SKILL_RE.match(part):
            return True
    return False


def _deny(msg: str) -> None:
    sys.stderr.write(
        "BLOCKED — this reads ANOTHER run's skill set.\n"
        f"{msg}\n"
        "Your staging directory is the only skill set you may read or write. Other runs' "
        "skills_v* directories hold a DIFFERENT optimizer's answers to these same train "
        "tasks; deriving your set from them would measure copying, not evolution. Work "
        "from ./digests/, ./oracle/ and the document corpus instead.\n"
    )
    sys.exit(2)


def _allowed_roots() -> list[str]:
    staging = os.environ.get("EVOSKILL_EVOLVER_STAGING", "").strip()
    if not staging:
        return []
    p = Path(staging)
    out = {str(p)}
    try:
        out.add(str(p.resolve()))
    except OSError:
        pass
    # The staging dir is usually referenced by its basename after a `cd` into its parent.
    out.add(p.name)
    return sorted(out)


def _owned(tok: str, allowed: list) -> bool:
    """True only if `tok` names THIS run's staging dir.

    Bare name (no separator) -> must equal the staging basename exactly.
    Anything with a path component -> must equal, or sit under, an ABSOLUTE allowed root.
    """
    abs_roots = [a for a in allowed if a.startswith("/")]
    names = [a for a in allowed if not a.startswith("/")]
    if "/" not in tok:
        return tok in names
    return any(tok == a or tok.startswith(a.rstrip("/") + "/") for a in abs_roots)


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0  # never wedge the evolver on a malformed payload

    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return 0

    allowed = _allowed_roots()
    for field in _FIELDS:
        val = tool_input.get(field)
        if not isinstance(val, str) or not val:
            continue
        for tok in re.split(r"[\s;|&()<>\"']+", val):
            if not tok or not _is_peer_skill_token(tok):
                continue
            # ITS OWN STAGING ONLY -- AND `a in tok` DID NOT MEAN THAT.
            # `allowed` carries the bare basename (legitimately: the dir is usually named
            # after a `cd` into its parent). Combined with a SUBSTRING test, that allowed
            # any path merely CONTAINING it -- and every arm names its staging dir
            # skills_v{N}_staging, so a PEER arm's same-epoch staging, another optimizer's
            # in-progress answers to these very train tasks, read as "its own". Measured:
            # /runs/OTHER_RUN/skills_v1_staging/a/SKILL.md returned 0 while the same run's
            # skills_v1 was correctly denied.
            # A token WITH a directory component must match an absolute allowed root by
            # prefix; only a bare basename may match by name.
            if _owned(tok, allowed):
                continue
            _deny(f"  tool={payload.get('tool_name')} field={field}\n  offending path: {tok}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
