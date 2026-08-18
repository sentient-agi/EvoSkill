"""Harness-selectable evolver dispatch: `claude_cli` today, `direct` as a peer.

NOTHING IMPORTS THIS YET, deliberately. It is built while a 4-epoch run is live, and wiring it
into `loop.py` would change the running instrument. `spec.py` gains `evolver_harness` only after
that run's last epoch lands.

── WHY THE HARNESS IS A CONTAINMENT QUESTION FIRST AND A PLUMBING QUESTION SECOND ──────────────

The CC evolver is confined by three PreToolUse hooks installed by `tau3_evolver.sh`:

    matcher "Bash|Read|Grep|Glob"              -> leak hook, held-out task specs   (#186)
    matcher "Bash|Read|Grep|Glob|Write|Edit"   -> peer-skill hook, other arms' skills (#213)
    matcher "Agent"                            -> synchronous subagent dispatch    (#212)

`src/harness/direct` HAS NO HOOK MECHANISM AT ALL. It has `_check_write_guard` (writes only) and
a sandbox profile. So porting the evolver to `direct` without building containment would SILENTLY
DELETE every one of those guards — the #186/#205/#213 defect class, reintroduced by a config
change, which is the worst way to reintroduce it.

── WHY WE DO NOT REUSE `direct._build_sandbox_profile` ─────────────────────────────────────────

It is deny-read-by-default and then, at executor.py:1285:

    (allow file-read* (subpath "{project_root}"))

i.e. it whitelists the WHOLE REPOSITORY. Every held-out task spec lives under the repository.
That profile is correct for a SOLVER, which is supposed to see the project, and useless for an
EVOLVER, whose entire containment problem is "this subtree and no other". Reusing it would have
produced a guard that cannot fail — #154's shape exactly.

── WHY SANDBOX-EXEC IS THE RIGHT PRIMITIVE, AND STRONGER THAN THE HOOKS IT REPLACES ────────────

#186 recorded that CC's `--add-dir` does NOT confine Bash: the evolver could reach held-out specs
through a shell command even though the directory was never added. The hooks exist to patch that,
and a hook is a PATTERN MATCH ON A COMMAND STRING — it must anticipate the phrasing. This session
alone produced five cases where a substring match was not a meaning; a hook is that same instrument
pointed at an adversary that can rephrase.

`sandbox-exec` denies at the kernel. A path that is not granted cannot be opened, by any command,
however phrased, including ones nobody thought of. So the direct path is not a downgrade to be
compensated for — done deny-first it is the stronger containment, and the CC path is the one
carrying the weaker guarantee.

── THE RULE THIS MODULE ENFORCES ───────────────────────────────────────────────────────────────

A harness may not be selected until it has DEMONSTRATED refusal on a real held-out path.
`containment_probe()` executes an actual read attempt and returns what happened. `selftest()`
asserts both that a forbidden read FAILS and that a granted read SUCCEEDS — the second half is
what stops this from being a guard that passes by refusing everything (a vacuous guard is #154).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

HARNESSES = ("claude_cli", "direct")


@dataclass
class EvolverJob:
    """Everything a dispatch backend needs, harness-independent."""

    epoch: int
    work_dir: Path                      # the evolver's isolated cwd
    prompt: str                         # the composed, proofread user prompt
    model: str
    log_path: Path
    grants: list[Path] = field(default_factory=list)   # readable subtrees (digests, staging, oracle)
    prompt_path: Path | None = None      # the composed prompt FILE — granted as a literal, not a tree
    writes: list[Path] = field(default_factory=list)   # #266: paths this job may WRITE (deny-all otherwise)
    # #295: subtrees to deny READ BY NAME, wherever they are. See the deny block in
    # build_evolver_sandbox_profile for why the four-prefix deny is not sufficient on its own.
    hide: list[Path] = field(default_factory=list)
    # #293: subtrees to deny WRITE BY NAME. Same reasoning as `hide`, other side of the profile.
    # A caller declares the run dir here and re-grants only what the job may produce via `writes`.
    no_write: list[Path] = field(default_factory=list)
    timeout_s: int = 5400
    env: dict = field(default_factory=dict)


# ── containment ─────────────────────────────────────────────────────────────────────────────

def _cli_install_allows() -> list[str]:
    """Read grants for wherever the `claude` CLI actually lives on THIS machine.

    Returns the resolved binary, its bin dir, and -- for a node install -- the package
    tree next to it, since the launcher script only re-execs into node_modules. Silent
    when the CLI is absent: preflight reports that separately, and a profile is still
    emittable on a machine that has not installed it yet.
    """
    out: list[str] = []
    exe = shutil.which("claude")
    if not exe:
        return out
    real = Path(exe).resolve()
    roots = {Path(exe).parent, real.parent}
    # npm -g layout: <prefix>/bin/claude -> <prefix>/lib/node_modules/@anthropic-ai/...
    for cand in (real.parent.parent, Path(exe).parent.parent):
        for sub in ("lib/node_modules", "libexec"):
            q = cand / sub
            if q.is_dir():
                roots.add(q)
    node = shutil.which("node")
    if node:
        # BOTH the symlink's directory and its target's: sandbox-exec resolves an exec
        # through every path component, so granting only the realpath tree still failed at
        # /opt/homebrew/bin/node with "execvp() ... Operation not permitted".
        roots.add(Path(node).parent)
        roots.add(Path(node).resolve().parent)
        roots.add(Path(node).resolve().parent.parent)

    # A PACKAGE-MANAGER PREFIX, GRANTED ONLY IF THE CLI ACTUALLY LIVES IN ONE.
    # Homebrew's node loads a web of sibling dylibs (@rpath/libnode.*, then
    # /opt/homebrew/opt/llhttp/lib/libllhttp.*, and onward), so granting the node package
    # alone gets past execvp and dies in dyld one symptom later. Enumerating those
    # libraries is unbounded and version-specific. The trade taken here: if -- and only
    # if -- the resolved CLI or its interpreter sits under a known prefix, grant that
    # prefix for READ. It is the operator's own toolchain directory; the containment that
    # matters is over the benchmark's held-out material, which lives under $HOME and the
    # repo and is denied by name elsewhere in this profile. On a machine where the CLI is
    # self-contained (the ~/.local install) no prefix is granted at all.
    for pref in (Path("/opt/homebrew"), Path("/usr/local"), Path("/opt/local")):
        if any(str(r).startswith(str(pref)) for r in roots):
            roots.add(pref)
    out.append("; the `claude` CLI as installed on THIS machine, derived not assumed (#291)")
    out.append(f'(allow file-read* (literal "{real}"))')
    for r in sorted({str(x) for x in roots}):
        out.append(f'(allow file-read* (subpath "{r}"))')
    return out


def build_evolver_sandbox_profile(job: EvolverJob) -> str:
    """Deny-read by default; grant ONLY what the evolver is entitled to.

    The inversion versus `direct._build_sandbox_profile` is the whole point: that one denies a
    fixed list of system subpaths and then allows the project root, so anything inside the repo
    is readable. This one allows a fixed list of SYSTEM paths needed to run a shell, denies
    everything else including the repository, and then grants exactly `job.grants`.

    `/Users` is denied as a subpath BEFORE the grants are allowed. sandbox-exec resolves by most
    specific match, so a later `(allow (subpath "<grant>"))` under /Users still wins for that
    subtree while its siblings stay denied. That ordering is the load-bearing detail and it is
    what `selftest()` proves, rather than assuming.
    """
    allows = "\n".join(f'(allow file-read* (subpath "{p.resolve()}"))' for p in job.grants)
    # #295. DENY THE RUN DIRECTORY BY NAME, exactly as #293 does for writes, and for exactly the
    # same reason: the block below denies reads by LOCATION -- /Users, /home, /opt, /Volumes --
    # and `(allow default)` covers everything else. A run directory outside those four prefixes
    # (a scratch mount, /var/folders, an external volume) is therefore FULLY READABLE, peer arms'
    # skills_v* and the split file with every held-out id included. The #213 and #186 properties
    # would hold in production only because run dirs happen to live under /Users.
    #
    # Caught by the FIRST fixture that put a peer arm in a tempdir (2026-08-11): the evolver read
    # it straight out, including through Bash, which is #186's measured failure reappearing at a
    # different address. Declared, never derived -- the caller names what to hide, because
    # guessing a parent from the grants is how a deny lands on the wrong directory.
    hides = "\n".join(f'(deny file-read* (subpath "{p.resolve()}"))' for p in (job.hide or []))
    # #293, as a first-class field rather than string surgery at one call site. The write deny
    # below is by LOCATION (repo, home); a run dir outside both matches neither and `(allow
    # default)` makes it fully writable -- the operator's answer channel, the countersignature
    # and every skills_v* included. Emitted BEFORE `write_allows`, so the job's declared outputs
    # are re-granted afterwards and win (last match).
    no_writes = "\n".join(f'(deny file-write* (subpath "{p.resolve()}"))'
                          for p in (job.no_write or []))
    work = job.work_dir.resolve()
    # #250c. THE LOG'S PARENT DIRECTORY NODE, AS A LITERAL, NEVER AS A SUBPATH.
    # LogWriter calls mkdir(exist_ok=True); CPython's exist_ok branch catches FileExistsError then
    # calls is_dir() to confirm, and a DENIED is_dir() makes it RE-RAISE. Writes were never
    # blocked (this profile only denies reads) -- the confirmation READ was.
    # My first fix added the run dir to `grants`, i.e. as a SUBPATH. That would have made the run
    # dir's split file readable, and the split file LISTS ALL 47 TEST IDS -- widening containment
    # to fix a plumbing bug, which is how #186 happens. A `literal` grant gives is_dir() and
    # listing on that one node while every file inside stays denied unless separately granted
    # (digests, staging and the train-only oracle already are).
    _logdir = job.log_path.resolve().parent
    # #300. THE LOG FILE ITSELF, not just its parent node. The child's stdout/stderr are
    # redirected INTO this file, and a process fstat()s its own stderr fd at startup. The parent
    # was granted as a directory node (#250c) but the file was not, and once #295 denied the run
    # dir by name the fd became unreadable: the CLI died with
    # `EPERM: operation not permitted, fstat` inside a Bun WriteStream — a JS stack trace naming
    # no path, which is why three probe generations missed it and only a real launch found it.
    # A LITERAL on the file, never a subpath on its directory: the run dir holds the split.
    # #300. THE FILES THE CHILD'S OWN fd's POINT AT — every one of them.
    # A process fstat()s its inherited stdout/stderr at startup (Bun does it in
    # internal:util/colors to detect colour support), and fstat on an fd whose PATH the sandbox
    # denies is EPERM. Once #295 denied the run dir by name, the log the launcher redirects into
    # became unreadable and the CLI died before its first turn, with a Bun stack trace naming no
    # path at all. Three probe generations and two "fixes" missed it; only a real launch found it.
    #
    # `.streamer.stderr` is the launcher's own convention for the Phoenix branch's stderr sink,
    # named here rather than passed in because getting it WRONG is silent (the child dies at
    # startup) while getting it wrong in the launcher would at least be visible in argv. If that
    # suffix ever changes, this must change with it.
    #
    # STRUCTURAL NOTE, recorded rather than fixed tonight: the evolver's log lives in the RUN DIR,
    # which is exactly the tree containment must hide (it holds the split). Logging into a denied
    # directory and then granting the log back file-by-file is a standing tension. The clean fix
    # is to write the log inside WORK -- already granted read+write -- and copy it out after. That
    # touches every consumer of the log path, so it is deliberately not done under time pressure.
    _lits = [_logdir, job.log_path.resolve(),
             job.log_path.resolve().with_name(job.log_path.name + ".streamer.stderr")]
    # The PROMPT FILE is the evolver's own input and is deliberately given to it. Granted as a
    # LITERAL FILE, so nothing else in the run dir (notably the split, which lists all 47 test ids)
    # becomes readable alongside it.
    if job.prompt_path:
        _lits.append(job.prompt_path.resolve())
    log_allow = "\n".join(f'(allow file-read* (literal "{x}"))' for x in _lits)
    # #250. THE INTERPRETER MUST BE GRANTED OR NOTHING RUNS AT ALL. `/Users` is denied as a
    # subpath, and on this host sys.executable lives under the user's home (a conda
    # install) — so the profile denied exec of the very binary the dispatch invokes:
    #     sandbox-exec: execvp() of '.../anaconda3/bin/python3' failed: Operation not permitted
    # Measured on the FIRST live dispatch, 2026-08-10. Zero cost (it died before any model call).
    #
    # Derived from sys.executable, NOT hardcoded: a hardcoded anaconda path would break the
    # moment the interpreter moved, and it would break silently-at-launch rather than loudly.
    # `direct._build_sandbox_profile` already grants the interpreter tree and the user's
    # ~/.local for exactly this reason — I read that line while deciding not to reuse the profile
    # and did not carry the requirement across. Granting the interpreter tree is safe: it is
    # outside the repository, so held-out specs stay denied.
    # ...AND THE UNRESOLVED PATH, WHICH IS THE ONE THAT GETS EXEC'D. resolve() maps
    # .venv/bin/python to its conda target, so only the conda tree was granted, while the
    # exec itself names the venv symlink inside the REPO -- denied by design. Result:
    #     execvp() of '<repo>/.venv/bin/python' failed: Operation not permitted
    # which is the #250 failure again, one indirection out. Caught 2026-08-18 by this
    # module's own selftest reporting "9 checks, 1 bad" while the launcher still dispatched.
    # A read grant on .venv/bin does not widen access to the repo's content: the src/ and
    # scripts/ content grants are separate and held-out material is denied by name.
    _exe = Path(sys.executable).resolve()
    _interp = sorted({str(_exe.parents[1]), str(Path.home() / ".local"),
                      str(Path(sys.executable).parent)})
    interp_allows = "\n".join(f'(allow file-read* (subpath "{d}"))' for d in _interp)

    # #250b. THE RUNNER MODULE LIVES IN THE REPO, AND THE REPO IS DENIED BY DESIGN. Granting the
    # interpreter got python to exec; it then died in `_path_importer_cache` with
    # `PermissionError: [Errno 1] Operation not permitted`, because it could not even STAT its
    # own sys.path entries — `python3 -m src.orchestrator.evolver_direct_runner` needs to read
    # the repo's CODE.
    #
    # The fix must not become "allow the repo", which is precisely the reuse trap this profile
    # exists to avoid (direct._build_sandbox_profile whitelists project_root and every held-out
    # spec lives under it). So the grant is split by WHAT IS NEEDED, not by convenience:
    #   * file-read-METADATA on the repo root -> stat/scandir succeed, so imports resolve, but
    #     no file CONTENT is readable.
    #   * file-read* on src/ and scripts/ only -> the code, which is not data.
    # Held-out material lives in third_party/ (the task pool) and outputs/*/oracle/ (other runs'
    # oracles). Neither is granted, so both remain unreadable — which selftest() proves by
    # probing a real forbidden path AFTER these grants exist, not before.
    _repo = Path(__file__).resolve().parents[2]
    repo_allows = "\n".join([
        # LITERAL, not subpath: grants the repo-root DIRECTORY NODE so python can enumerate it,
        # without granting anything inside. `file-read-metadata (subpath ...)` was tried first and
        # is NOT sufficient -- macOS requires file-read-DATA on a directory to list it, and python
        # lists every sys.path entry. Measured: metadata-only died in _path_importer_cache.
        f'(allow file-read* (literal "{_repo}"))',
        f'(allow file-read* (subpath "{_repo / "src"}"))',
        f'(allow file-read* (subpath "{_repo / "scripts"}"))',
    ])
    # #266 WRITE CONTAINMENT (2026-08-11). Until now this profile governed READS ONLY — there was
    # not one file-write rule in it, for the evolver or anyone else. That was survivable while
    # nothing downstream TRUSTED a file an agent could author. It stopped being survivable the
    # moment ask() started reading ANSWERS_e{N}.json and _v_operator_signature started reading
    # e{N}_PROOFREAD.json: both are trust boundaries, and an agent with write access to the run
    # dir could satisfy its own gate — ask itself a question and answer it, or countersign its own
    # prompt. Read-containment cannot express that; only write-containment can.
    #
    # Deny-then-grant, same shape as the reads, same most-specific-match resolution. The grants
    # are deliberately narrow: the job's own work dir, its log's directory, and whatever it
    # explicitly declared in `writes`. Everything else under /Users is denied, INCLUDING the run
    # directory the AO previously had wholesale.
    #
    # ~/.claude is granted because the CLI writes its own session/telemetry state there and dies
    # without it — verified by probe, not assumed. That is a real widening and it is why the
    # answer channel must ALSO live outside any agent-writable path rather than relying on this
    # list alone: defence in depth, because one forgotten grant here re-opens the boundary.
    # THE LOG IS GRANTED AS A LITERAL FILE, NEVER VIA ITS PARENT. My first version wrote
    # `_wr = [work, _logdir]`, and _logdir is log_path.parent — which IS THE RUN DIR. That
    # granted write access to the whole run dir while purporting to lock it down: the probe
    # showed ANSWERS/PROOFREAD/skills all still writable. The read path carries a comment
    # (#250c) saying exactly this — "the log's parent directory node, as a LITERAL, never as a
    # SUBPATH" — and I reproduced the mistake it warns about, on the other side of the profile.
    # A subpath grant for a file that lives in a sensitive directory grants the directory.
    _wr_sub = [work] + [Path(w).resolve() for w in (job.writes or [])]
    _wr_lit = [job.log_path.resolve()]
    _home_claude = Path.home() / ".claude"
    write_allows = "\n".join(
        [f'(allow file-write* (subpath "{d}"))' for d in sorted({str(x) for x in _wr_sub})]
        + [f'(allow file-write* (literal "{f}"))' for f in sorted({str(x) for x in _wr_lit})]
        + [f'(allow file-write* (subpath "{_home_claude}"))'])

    # ── #291. WHAT THE `claude` CLI ITSELF NEEDS TO RUN AT ALL ────────────────────────────
    #
    # This profile was written for, and only ever exercised against, the `direct` harness — a
    # python module. The `claude_cli` harness (every real evolver run to date) does NOT wrap its
    # dispatch in sandbox-exec at all; it shells out to scripts/tau3_evolver.sh unwrapped. So when
    # #267 put the AO under this profile, THAT WAS THE FIRST TIME A REAL `claude` LAUNCH HAD EVER
    # RUN UNDER IT — and it died instantly. Two distinct failures, both invisible to the existing
    # probes because those probe /bin/cat and /bin/sh, never the CLI:
    #
    #   1. AUTHENTICATION. `~/.claude` was granted WRITE ("the CLI writes its own state there and
    #      dies without it — verified by probe") but never READ, and OAuth credentials live in the
    #      login keychain under ~/Library/Keychains. Result: `Not logged in · Please run /login`,
    #      exit 1, in 4 seconds. The probe that "verified" the write grant never checked that the
    #      process could then authenticate.
    #
    #   2. DENIED != ABSENT ON THE CONFIG SEARCH PATH. The CLI walks from cwd up to $HOME looking
    #      for configuration. A MISSING directory is fine; a DENIED one raises EPERM and kills the
    #      process outright — `error: An internal error occurred (EPERM)`, with no indication of
    #      which path. Since `(deny file-read* (subpath "/Users"))` denies every ancestor, the CLI
    #      could not start regardless of what was granted below. Bisected: granting exactly the
    #      ancestor DIRECTORY NODES (literals — the node, never the tree) is sufficient.
    #
    # The grants below are the minimum found by bisection, and `~/.claude/projects` is re-denied
    # on top: that subtree holds every past session transcript on this machine, which is both the
    # largest read-widening available here and the one most likely to contain held-out material
    # from other runs. Verified: the CLI starts and authenticates with projects/ denied.
    _anc, _p = [], work
    for _p in (work, _repo):
        cur = Path(_p).resolve()
        while True:
            _anc.append(cur)
            if cur == cur.parent or cur == Path.home() or len(cur.parts) <= 2:
                break
            cur = cur.parent
    _anc = sorted({str(x) for x in _anc if str(x).startswith(str(Path.home()))})
    cli_allows = "\n".join(
        ["; #291 the ancestor directory NODES of cwd and the repo, as literals. The CLI walks up",
         "; from cwd looking for config; a DENIED ancestor is EPERM (fatal), unlike an absent one."]
        + [f'(allow file-read* (literal "{a}"))' for a in _anc]
        + ["; #291 the CLI's own installation, config and credentials. Without these it reports",
         "; `Not logged in` and exits 1 — the write grant on ~/.claude is not sufficient.",
         f'(allow file-read* (subpath "{Path.home() / ".local"}"))',
         f'(allow file-read* (subpath "{_home_claude}"))',
         # #291 FOURTH SITE. The grants above cover ~/.local, which is where the CLI lands
         # for the installer this repo's authors happen to use -- so the profile worked here
         # and nowhere else. README.md tells operators to `npm install -g`, which under
         # Homebrew on Apple Silicon puts the binary at /opt/homebrew/bin/claude with its
         # modules under /opt/homebrew/lib/node_modules, and /opt is DENIED a few lines
         # below. The child then dies at execvp while the launcher prints "containment:
         # sandbox-exec ...". Derive the location instead of assuming it, the same way #250
         # derives the interpreter from sys.executable rather than hardcoding a path.
         f'(allow file-read* (literal "{Path.home() / ".claude.json"}"))',
         f'(allow file-read* (subpath "{Path.home() / "Library" / "Keychains"}"))',
         "; ... but NOT other sessions' transcripts. Granted above by the ~/.claude subpath;",
         "; taken back here, because that tree holds every past run's conversation on this host.",
         f'(deny file-read* (subpath "{_home_claude / "projects"}"))']
        + _cli_install_allows())

    # #296. RULE ORDER IS THE WHOLE PROFILE, and it was inverted. SBPL takes the LAST matching
    # rule, and the broad "minimum needed to run a shell" allow -- which includes
    # (subpath "/var") and (subpath "/private") -- used to sit at the BOTTOM. It therefore
    # overrode every deny above it for those trees, silently. Proof: a #295 deny on a run dir in
    # /var/folders emitted correctly into the profile and the file was still readable, contents
    # and all, including through Bash.
    #
    # The order below is broad-allow FIRST, then denies, then narrow grants: each later rule
    # refines the one before it, which is the only arrangement in which "deny X, then grant X/y"
    # means what it reads like. The system paths are still allowed; they are simply no longer
    # allowed LAST.
    return f"""(version 1)
(allow default)
; the minimum needed to run a shell and a python. FIRST, so the denies and grants below can
; refine it -- see #296 for what happened when this sat at the bottom.
(allow file-read* (subpath "/usr") (subpath "/bin") (subpath "/sbin")
  (subpath "/System") (subpath "/Library") (subpath "/var") (subpath "/private")
  (subpath "/dev") (subpath "/etc"))
(deny file-write*
  (subpath "{_repo}")
  (subpath "{Path.home()}")
)
; #293 denied BY NAME, so write containment does not depend on where the run dir happens to sit
{no_writes}
; the ONLY writable paths: this job's work dir, its log dir, its declared outputs, CLI state
{write_allows}
(deny file-read*
  (subpath "/Users")
  (subpath "/home")
  (subpath "/opt")
  (subpath "/Volumes")
)
; #295 denied BY NAME, so containment does not depend on where the run dir happens to sit.
; The explicit grants below come later and win for their own leaves (last match wins).
{hides}
; the interpreter and its library tree — without these sandbox-exec cannot even exec python
{interp_allows}
; #291 what the `claude` CLI needs to start and authenticate — see the comment at the builder
{cli_allows}
; the repo: METADATA everywhere (so imports can stat sys.path), CONTENT only under src/ scripts/
{repo_allows}
; the evolver's own working directory — it must be able to read what it writes
(allow file-read* (subpath "{work}"))
; the log's parent DIRECTORY NODE only (#250c) — is_dir() must succeed; contents stay denied
{log_allow}
; explicit grants: the evidence surface, the staging dir, the train-only oracle. LAST, so they
; win over both the system allow above and the #295 deny-by-name (#296).
{allows}
"""


def containment_probe(job: EvolverJob, target: Path) -> tuple[bool, str]:
    """Actually attempt to read `target` under this job's profile. Returns (readable, detail).

    EXECUTES the read; it does not reason about the profile. A containment claim that was never
    exercised is the #154 shape, and this module's whole premise is that a harness must
    demonstrate refusal before it may be selected.
    """
    if not _sandbox_exec_available():
        return (True, "sandbox-exec unavailable — containment NOT enforceable on this host")
    with tempfile.NamedTemporaryFile("w", suffix=".sb", delete=False) as fh:
        fh.write(build_evolver_sandbox_profile(job))
        prof = fh.name
    try:
        r = subprocess.run(
            ["sandbox-exec", "-f", prof, "/bin/cat", str(target)],
            capture_output=True, text=True, timeout=30,
        )
        # rc 0 AND content => the read succeeded. A denied read exits non-zero.
        ok = r.returncode == 0
        return (ok, (r.stdout[:200] if ok else (r.stderr or "denied").strip()[:200]))
    finally:
        os.unlink(prof)


def write_containment_probe(job: EvolverJob, target: Path) -> tuple[bool, str]:
    """Actually ATTEMPT TO WRITE `target` under this job's profile. Returns (writable, detail).

    #266. Mirrors containment_probe deliberately, because the read version is the one that has
    been trusted for weeks and the failure it exists to prevent (#154) is precisely a check that
    could never fail. This one shells out and tries the write; it does not inspect the profile
    text. A profile that CONTAINS the right rule and a process that is actually refused are two
    different claims, and only the second one is worth anything.

    The target is written and then removed on success, so a probe never leaves an artifact that a
    later verify could mistake for a real one.
    """
    if not _sandbox_exec_available():
        return (True, "sandbox-exec unavailable — containment NOT enforceable on this host")
    with tempfile.NamedTemporaryFile("w", suffix=".sb", delete=False) as fh:
        fh.write(build_evolver_sandbox_profile(job))
        prof = fh.name
    try:
        r = subprocess.run(
            ["sandbox-exec", "-f", prof, "/bin/sh", "-c", f'echo probe > "{target}"'],
            capture_output=True, text=True, timeout=30,
        )
        ok = r.returncode == 0 and Path(target).exists()
        if ok:
            try: Path(target).unlink()
            except OSError: pass
        return (ok, "WRITE SUCCEEDED" if ok else (r.stderr or "denied").strip()[:200])
    finally:
        os.unlink(prof)


def preflight(job: EvolverJob) -> list[str]:
    """Probe every path the runner will touch, under the REAL profile. Returns ALL failures.

    WHY THIS EXISTS. Four consecutive dispatch attempts each died on a different missing grant —
    the interpreter (#250), the repo listing (#250b), the log dir's is_dir() (#250c), the prompt
    file. Each cost a launch to discover ONE gap, because nothing enumerated the requirements up
    front. Fixing them one at a time is the wrong shape: the failures were not independent bugs,
    they were one missing step. So: probe the whole set before spending a dispatch, and report
    every failure together.

    This is a PREFLIGHT, not a guard — it must never widen the profile. If a required path is
    denied, the answer is a narrow grant for that path, decided by a person, not a broader rule.
    """
    if not _sandbox_exec_available():
        return ["sandbox-exec unavailable — containment is NOT enforceable on this host"]
    bad: list[str] = []
    with tempfile.NamedTemporaryFile("w", suffix=".sb", delete=False) as fh:
        fh.write(build_evolver_sandbox_profile(job))
        prof = fh.name
    try:
        # 1. the interpreter must EXEC
        r = subprocess.run(["sandbox-exec", "-f", prof, sys.executable, "-c", "print(1)"],
                           capture_output=True, text=True, timeout=60)
        if r.returncode:
            bad.append(f"cannot EXEC {sys.executable}: {(r.stderr or '').strip()[:120]}")
        # 1b. THE `claude` CLI MUST EXEC. Preflight covered the interpreter and the runner
        # but never the binary the AO and the evolver actually run through, so the one gap
        # it could not see was the one that hit operators who installed the CLI where
        # README.md tells them to (npm -g -> /opt/homebrew, a DENIED tree). This probe is
        # the reason the grant above can stay narrow: if a machine puts the CLI somewhere
        # neither is derived, this says so before a dispatch is spent rather than after.
        cli = shutil.which("claude")
        if cli is None:
            bad.append("`claude` is not on PATH — the AO and the evolver both exec it "
                       "(README: npm install -g @anthropic-ai/claude-code)")
        else:
            r = subprocess.run(["sandbox-exec", "-f", prof, cli, "--version"],
                               capture_output=True, text=True, timeout=60)
            if r.returncode:
                bad.append(f"cannot EXEC the claude CLI at {cli} under the profile: "
                           f"{(r.stderr or '').strip()[:160]}")
        # 2. the runner module must IMPORT
        r = subprocess.run(["sandbox-exec", "-f", prof, sys.executable, "-c",
                            "import src.orchestrator.evolver_direct_runner"],
                           capture_output=True, text=True, timeout=90,
                           cwd=str(Path(__file__).resolve().parents[2]))
        if r.returncode:
            bad.append(f"cannot IMPORT the runner: {(r.stderr or '').strip().splitlines()[-1][:120]}")
        # 3. every declared READ must succeed
        reads = [job.prompt_path] + list(job.grants) if job.prompt_path else list(job.grants)
        for p in reads:
            if p is None or not Path(p).exists():
                continue
            tgt = p if Path(p).is_file() else next(iter(sorted(Path(p).rglob("*"))), None)
            if tgt is None or not Path(tgt).is_file():
                continue
            r = subprocess.run(["sandbox-exec", "-f", prof, "/bin/cat", str(tgt)],
                               capture_output=True, text=True, timeout=30)
            if r.returncode:
                bad.append(f"cannot READ a required input: {tgt}")
        # 4. the log's parent must satisfy is_dir()
        r = subprocess.run(["sandbox-exec", "-f", prof, sys.executable, "-c",
                            f"import pathlib,sys; sys.exit(0 if pathlib.Path({str(job.log_path.resolve().parent)!r}).is_dir() else 3)"],
                           capture_output=True, text=True, timeout=60)
        if r.returncode:
            bad.append("log parent fails is_dir() — mkdir(exist_ok=True) will re-raise (#250c)")
    finally:
        os.unlink(prof)
    return bad


def _sandbox_exec_available() -> bool:
    return shutil.which("sandbox-exec") is not None


# ── dispatch backends ───────────────────────────────────────────────────────────────────────

def dispatch(job: EvolverJob, harness: str) -> list[str]:
    """Return the argv for `harness`. DELEGATES for claude_cli — never reimplements it.

    `tau3_evolver.sh` owns the org-key refusal, the quota gate, the leak hooks, the seeded
    staging and the Phoenix tee. A second copy of any of that is #154, so the claude_cli backend
    stays a thin call into the script that four epochs of this programme have already exercised.
    """
    if harness not in HARNESSES:
        raise ValueError(f"unknown evolver harness {harness!r}; known: {HARNESSES}")
    if harness == "claude_cli":
        return ["bash", "scripts/tau3_evolver.sh", str(job.epoch), str(job.log_path)]
    return _direct_argv(job)


def _direct_argv(job: EvolverJob) -> list[str]:
    """Direct-harness evolver, run under the deny-first profile.

    NOT YET WIRED: the runner module this points at is the next piece of work. The argv shape is
    fixed here so the containment tests above can be written and run against the profile FIRST —
    building the guard before the thing it guards is deliberate, and the reverse order is how
    #186 and #213 happened.
    """
    return [
        "sandbox-exec", "-f", "<profile-written-at-dispatch>",
        "python3", "-m", "src.orchestrator.evolver_direct_runner",
        "--epoch", str(job.epoch),
        "--work-dir", str(job.work_dir),
        "--model", job.model,
        "--log", str(job.log_path),
    ]


# ── selftest ────────────────────────────────────────────────────────────────────────────────

def selftest() -> int:
    """Both halves matter: forbidden must FAIL and granted must SUCCEED.

    A profile that denies everything would pass a refusal-only test while making the evolver
    useless, and would look like security. That is why the granted-read assertion is here.
    """
    checks: list[tuple[str, bool]] = []
    # FIXTURES MUST LIVE WHERE THE REAL THING LIVES. The first version of this test used
    # `tempfile.mkdtemp()`, which on macOS returns a path under /var/folders — and the profile
    # legitimately allows /var so a shell can run. So the "forbidden" read SUCCEEDED, the test
    # went red, and the profile it was testing was in fact correct for production, where held-out
    # specs sit under /Users/<me>/dev/EvoSkill and are denied. A guard proved against the wrong
    # geometry is not proved. Fixtures therefore go under $HOME, the tree the profile denies.
    tmp = Path(tempfile.mkdtemp(prefix=".evodisp_", dir=str(Path.home())))
    try:
        granted = tmp / "granted"
        forbidden = tmp / "forbidden"
        granted.mkdir()
        forbidden.mkdir()
        (granted / "digest.json").write_text('{"task":"train_only"}')
        (forbidden / "held_out_spec.json").write_text('{"task":"task_026","SEALED":true}')

        job = EvolverJob(epoch=1, work_dir=tmp / "work", prompt="p", model="m",
                         log_path=tmp / "e.log", grants=[granted])
        (tmp / "work").mkdir()

        ok_g, d_g = containment_probe(job, granted / "digest.json")
        ok_f, d_f = containment_probe(job, forbidden / "held_out_spec.json")

        if not _sandbox_exec_available():
            checks.append(("sandbox-exec present (containment enforceable)", False))
        else:
            checks.append(("GRANTED path is readable (guard is not vacuous)", ok_g is True))
            checks.append(("FORBIDDEN path is refused (guard actually guards)", ok_f is False))

        # #250. THE CHECK WHOSE ABSENCE COST THE FIRST LIVE DISPATCH. Every other check here
        # exercises READS under the profile; none of them exercised EXEC of the interpreter the
        # dispatch actually invokes, so a profile that denied `sys.executable` passed 7/7 and
        # then died at launch. Proving a guard against the wrong operation is the same error as
        # proving it against the wrong geometry (the tempfile-under-/var case above).
        if _sandbox_exec_available():
            with tempfile.NamedTemporaryFile("w", suffix=".sb", delete=False) as fh:
                fh.write(build_evolver_sandbox_profile(job))
                _prof = fh.name
            try:
                r = subprocess.run(["sandbox-exec", "-f", _prof, sys.executable, "-c",
                                    "print('exec-ok')"], capture_output=True, text=True, timeout=60)
                checks.append(("sys.executable can EXEC under the profile (#250)",
                               r.returncode == 0 and "exec-ok" in r.stdout))
            finally:
                os.unlink(_prof)

        checks.append(("unknown harness raises", _raises(lambda: dispatch(job, "nope"))))
        checks.append(("claude_cli delegates to the script, does not reimplement",
                       dispatch(job, "claude_cli")[:2] == ["bash", "scripts/tau3_evolver.sh"]))
        checks.append(("profile denies /Users before granting",
                       '(subpath "/Users")' in build_evolver_sandbox_profile(job)))
        # The trap is `file-read* (subpath "<repo>")` -- content on the whole tree. A `literal`
        # grant on the root NODE is not the trap and is required for imports (#250b). Testing the
        # intent, not the substring: my first version of this check failed the moment a legitimate
        # repo grant appeared, which would have pressured me to weaken the check instead of the
        # profile. Also assert positively that the held-out subtree is NOT granted.
        _prof = build_evolver_sandbox_profile(job)
        _repo_s = str(Path(__file__).resolve().parents[2])
        checks.append(("profile does NOT grant CONTENT on the repo subtree (the reuse trap)",
                       f'(allow file-read* (subpath "{_repo_s}"))' not in _prof))
        checks.append(("third_party/ (the task pool) is NOT granted",
                       "third_party" not in _prof))
        checks.append(("grants appear in the profile",
                       str(granted.resolve()) in build_evolver_sandbox_profile(job)))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    bad = 0
    for name, good in checks:
        print(f"  [{'ok' if good else 'BAD'}] {name}")
        bad += 0 if good else 1
    print(f"\n{len(checks)} checks, {bad} bad")
    return 1 if bad else 0


def _raises(fn) -> bool:
    try:
        fn()
    except Exception:
        return True
    return False


def _cli() -> int:
    """Emit a profile / run a preflight for a caller that is NOT python — i.e. the shell launcher.

    #294. `scripts/tau3_evolver.sh` needs the same containment the direct harness has, and the
    one thing it must NOT do is build the profile itself. A second spelling of these rules is
    #154's shape exactly: two implementations that agree today, drift silently, and the one that
    matters is the one nobody tested. So the shell asks this module for the bytes.

    THE PROFILE WRAPS THE `claude` CHILD, NOT THE SCRIPT. The launcher READS THE SPLIT FILE to
    run its own oracle-vs-split firewall (tau3_evolver.sh's exit-8 gate), and that file lists
    every held-out id — the precise artifact the evolver's containment exists to hide. Sandboxing
    the whole script would therefore require granting the launcher the thing the sandbox is for.
    Launcher and evolver have opposite access needs; only the child gets confined.
    """
    import argparse
    ap = argparse.ArgumentParser(prog="evolver_dispatch")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--emit-profile", metavar="OUT", default="",
                    help="write the sandbox profile for the described job to OUT")
    ap.add_argument("--preflight", action="store_true",
                    help="probe every declared path under the real profile; exit 9 on any gap")
    ap.add_argument("--epoch", type=int, default=0)
    ap.add_argument("--work-dir", default="")
    ap.add_argument("--log", default="")
    ap.add_argument("--prompt-path", default="")
    ap.add_argument("--grant", action="append", default=[], help="readable subtree; repeatable")
    ap.add_argument("--write", action="append", default=[], help="writable subtree; repeatable")
    ap.add_argument("--hide", action="append", default=[],
                    help="#295: subtree to deny READ by name (e.g. the run dir); repeatable")
    ap.add_argument("--no-write", action="append", default=[],
                    help="#293: subtree to deny WRITE by name (e.g. the run dir); repeatable")
    a = ap.parse_args()
    if a.selftest or not (a.emit_profile or a.preflight):
        return selftest()
    if not (a.work_dir and a.log):
        print("--work-dir and --log are required", file=sys.stderr)
        return 64
    job = EvolverJob(
        epoch=a.epoch, work_dir=Path(a.work_dir), prompt="", model="",
        log_path=Path(a.log),
        prompt_path=Path(a.prompt_path) if a.prompt_path else None,
        # #299. DO NOT DROP A DECLARED GRANT BECAUSE THE PATH IS NOT THERE YET. The filter
        # `if Path(g).exists()` silently discarded the CLI's own session-transcript directory,
        # which the CLI CREATES at startup — so the grant that fixes #292 never reached the
        # profile and the evolver died with the identical error, identical log length, as if the
        # fix had not been written. A dropped grant is invisible: the only symptom was the
        # "4 grant(s)" line still reading 4. SBPL does not require a path to exist.
        grants=[Path(g).resolve() for g in a.grant],
        writes=[Path(w).resolve() for w in a.write],
        hide=[Path(h).resolve() for h in a.hide],
        no_write=[Path(w).resolve() for w in a.no_write])
    if not _sandbox_exec_available():
        # NEVER emit an empty profile: `sandbox-exec -f <empty>` is not "no sandbox", it is a
        # confusing failure, and returning success here would let the launcher believe it is
        # contained. Refuse, and let the caller decide (#286's rule, applied to this entry point).
        print("REFUSING: sandbox-exec is unavailable, so the evolver cannot be contained.",
              file=sys.stderr)
        return 9
    if a.preflight:
        gaps = preflight(job)
        if gaps:
            print(f"PREFLIGHT FOUND {len(gaps)} CONTAINMENT GAP(S) — not dispatching:",
                  file=sys.stderr)
            for g in gaps:
                print(f"  - {g}", file=sys.stderr)
            print("  Grant each path NARROWLY (a literal for a file or directory node, a subpath\n"
                  "  only for a tree the evolver is entitled to). Do NOT widen to the repo: its\n"
                  "  split file lists every held-out id.", file=sys.stderr)
            return 9
        print(f"preflight OK — {len(job.grants)} grant(s), {len(job.writes)} write target(s)")
    if a.emit_profile:
        out = Path(a.emit_profile)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(build_evolver_sandbox_profile(job))
        print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
