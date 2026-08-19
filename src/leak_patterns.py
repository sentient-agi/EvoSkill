"""ONE definition of "this text names held-out benchmark material".

IT LIVES AT src/ TOP LEVEL, NOT UNDER src/harness/, AND THAT IS LOAD-BEARING.
Claude Code invokes the PreToolUse hook as `python3 <path>` -- the SYSTEM interpreter,
not the project venv. `src/harness/__init__.py` imports pydantic, so putting this module
there made the hook die on import with ModuleNotFoundError and exit 1, which does not
block: exactly the defect this whole file exists to close, reintroduced by the fix for
it. Caught by running the hook. This module and everything it imports must stay
stdlib-only, and `src/__init__.py` must stay free of eager imports.

Three enforcement points consume these: the live PreToolUse hook
(scripts/evolver_leak_hook.py), the post-hoc session audit
(scripts/evolver_leak_audit.py), and the orchestrator's backstop
(src.orchestrator.invariants.check_no_heldout_reachable). They had THREE separate
copies of the rule, and a copy is how two guards become one real one: on 2026-08-18
the backstop was found carrying invented regexes that missed SSB v2 entirely while
the audit's own patterns missed a whole directory.

THE DIRECTORY HOLE, measured. Every pattern named a spec FILE, so
    cp -r .../domains/banking_knowledge/tasks /tmp/specs
copied all 97 specs -- train, val and test -- and returned rc=0 from the hook, while
the same read naming one id returned rc=2. The backstop then certified the session
"no full-pool reads". The same shape on SSB v2:
    cp /.../spreadsheetbench-v2/*/*/*_answer.xlsx /tmp/
A guard that only recognises the singular case is not a guard against bulk copying,
which is the easier attack and the likelier accident.
"""
from __future__ import annotations

import posixpath
import re

# BOTH BENCHMARKS' ID SHAPES. tau3 ids are `task_004`; SSB v2 ids are `Debugging/01_02` on
# the split and `Debugging_01_02` as a filename. Both spellings are matched rather than
# normalising one to the other, because stem -> id is ambiguous (category names contain
# underscores) and a wrong normalisation would file a held-out task under train.
TASK_RE = re.compile(r"task_\d+|[A-Za-z][A-Za-z_]*[/_]\d+_\d+")


def id_forms(ids) -> set:
    """Every spelling an id can take in a command line."""
    out = set()
    for i in ids:
        out.add(i)
        out.add(i.replace("/", "_"))
    return out


# Paths that carry held-out material. Deliberately broad: a false positive costs one
# manual look, a false negative costs an epoch.
LEAK_PATH_RE = re.compile(
    r"""(?x)
    # ANY tasks-bearing json at the domain level, not two hardcoded names. tau2's own
    # domains ship tasks.json, tasks_voice.json, tasks_full.json, tasks_small.json and
    # split_tasks.json -- and only the first two were named, so `cat
    # domains/telecom/tasks_full.json` (the full pool) and split_tasks.json (356 KB of task
    # ids) read clean. `tasks\.json` cannot match `tasks_full.json`, and the directory
    # alternative below cannot either: it requires a non-word character after `tasks`.
    domains/[a-z_]+/[a-z_]*tasks[a-z_]*\.json
  | domains/[a-z_]+/tasks/task_\d+\.json   # one per-task spec
  | domains/[a-z_]+/tasks(?![\w.])         # ...and the DIRECTORY holding all of them
    # ...AND THE DOMAIN DIRECTORY NAMED BARE, which is how the specs leave without ever
    # being named: `cp -r .../domains/banking_knowledge /tmp/bk` then read /tmp/bk/tasks.json,
    # or `cd .../domains/banking_knowledge` then `cat tasks.json` relative. Neither command
    # contains any pattern above. On macOS the profile's --hide covers it; on Linux there is
    # no seatbelt and this line is the only layer. The optional trailing slash is NOT
    # cosmetic -- `cp -r <dir>/ /tmp/bk` is the same escape one keystroke away -- and the
    # lookahead forbids `/` so `<dir>/documents` (the corpus the evolver IS given) does not
    # match by backtracking the `/?` to empty.
  | tau2/domains/[a-z_]+/?(?![\w./])
  | test_domains?/                         # per-task pytest walkthroughs
  | tests?/.*test_task_\d+
    # ── SSB v2. Its ground truth IS the reference workbook, so the dataset root holds the
    # answers for every task including all 120 held out. The root is configurable
    # (SSBV2_ROOT) and supplied at runtime via EVOSKILL_LEAK_DENY_PATHS; these cover the
    # shapes that appear wherever it lives. resolve_base() supports BOTH <root>/data/
    # spreadsheetbench-v2 and <root>/spreadsheetbench-v2, so no pattern may require `data/`.
  | SpreadsheetBench-v2/                   # the dataset root, answers for all splits
  | spreadsheetbench-v2/                   # the corpus dir under either layout
  | _snapshots(_[a-z]+)?/                  # produced-workbook trees, all splits together
  | /(answer|ground_?truth|reference)[_-]?\w*\.xlsx
  | \d+_answer\.xlsx                       # the reference workbook, any path or glob
    """
)

# FULL-POOL reads, for the post-hoc backstop: a single one of these exposes every spec or
# every reference answer, so it is high-signal and reported separately from id matches.
POOL_RES = (
    r"data/tau2/domains/[a-z_]+/tasks\.json",
    r"domains/[a-z_]+/tasks(?![\w.])",
    r"spreadsheetbench-v2/[A-Za-z_]+/\d+_\d+",
    r"spreadsheetbench-v2/[^\s'\"]*\*",          # a glob across the corpus
    r"[A-Za-z_]+/\d+_\d+/\d+_answer\.xlsx",
    r"\d+_answer\.xlsx",
)


def normalised_paths(text: str) -> str:
    """`text`, plus a lexically-normalised copy of every path-like token in it.

    WHY. Every matcher in this module compares SPELLINGS, and a shell reaches the same
    bytes by several. Measured against the patterns above before this existed:

        cat .../domains/telecom/tasks.json                        -> BLOCKED
        cat .../domains/banking_knowledge/../telecom/tasks.json   -> allowed
        cat .../domains/./telecom/./tasks.json                    -> allowed
        cat .../domains//telecom//tasks.json                      -> allowed

    -- the full 97-task pool, three keystrokes away from a guard that says it denies it.
    `..` also defeats a prefix test, which is how the peer-skill hook let
    /runs/OWN/skills_v1_staging/../../OTHER/skills_v1/a/SKILL.md through as "its own".

    APPENDED, NEVER SUBSTITUTED. normpath() would mangle a URL's scheme
    (https://h/a/../b -> https:/h/b) and the hook matches URLs too, so the original text
    stays first and the normalised forms are added after it. Purely lexical: no
    filesystem access, because the hook runs inside a sandbox that cannot stat these
    paths, and resolving symlinks is not what a spelling check is for.
    """
    out = [text]
    for tok in re.split(r"[\s;|&()<>\"']+", text):
        if "/" not in tok:
            continue
        tok = tok.strip(",:")
        try:
            n = posixpath.normpath(tok)
        except (ValueError, TypeError):
            continue
        if n and n != tok:
            out.append(n)
    return "\n".join(out)
