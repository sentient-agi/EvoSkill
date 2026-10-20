"""The macOS sandbox-exec containment for a LOCAL `claude` child (the evolver or the AO run on the host), and
the parent-session environment scrub every evolver dispatch applies.

The report's evolver runs in a container on the pool machine (scripts/daytona/evolver_container.py) and uses only
scrub_parent_session_env() from this module. The rest serves a local child: a deny-first sandbox-exec profile
(build_evolver_sandbox_profile), probes that EXECUTE reads, writes and connects under it (containment_probe,
write_containment_probe, preflight, egress_probes), and a CLI that emits the profile for a shell launcher.

── WHY SANDBOX-EXEC ────────────────────────────────────────────────────────────────────────────

Claude Code's `--add-dir` does NOT confine Bash: a child can reach any path through a shell command even though
the directory was never added. A PreToolUse hook is a PATTERN MATCH ON A COMMAND STRING and must anticipate every
phrasing. `sandbox-exec` denies at the kernel: a path that is not granted cannot be opened, by any command, however
phrased.

── WHY NOT A SOLVER'S PROFILE ──────────────────────────────────────────────────────────────────

A solver's profile whitelists the WHOLE REPOSITORY, and every held-out task spec lives under the repository. That is
right for a solver, which is supposed to see the project, and useless for an evolver, whose containment problem is
"this subtree and no other". Reusing it would produce a guard that cannot fail.

── THE RULE THIS MODULE ENFORCES ───────────────────────────────────────────────────────────────

A containment claim must be DEMONSTRATED. `containment_probe()` executes an actual read attempt and returns what
happened. `selftest()` asserts both that a forbidden read FAILS and that a granted read SUCCEEDS -- the second half
is what stops this from being a guard that passes by refusing everything.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

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
    writes: list[Path] = field(default_factory=list)   # paths this job may WRITE (deny-all otherwise)
    # Subtrees to deny READ BY NAME, wherever they are. See the deny block in
    # build_evolver_sandbox_profile for why the four-prefix deny is not sufficient on its own.
    hide: list[Path] = field(default_factory=list)
    # Subtrees to deny WRITE BY NAME. Same reasoning as `hide`, other side of the profile.
    # A caller declares the run dir here and re-grants only what the job may produce via `writes`.
    no_write: list[Path] = field(default_factory=list)
    timeout_s: int = 5400
    env: dict = field(default_factory=dict)
    # The `claude` child's OWN temp root, handed to it as
    # CLAUDE_CODE_TMPDIR (see mint_cli_tmpdir). Granted read+write AFTER the deny of the host's
    # shared per-uid temp root. None for a child that is not the CLI.
    cli_tmpdir: Path | None = None
    # A DRY RUN's preflight never probes the host's REAL per-uid CLI temp root
    # (every session's task outputs); step 5 probes a throwaway stand-in instead. See preflight().
    dry_run: bool = False
    # EGRESS. None = the profile carries NO network rule, so network-outbound falls under `(allow default)`.
    # A list = egress ENFORCED: every outbound connection is denied — TCP, UDP and AF_UNIX
    # connects alike (measured: an AF_UNIX connect under `(deny network-outbound)` is EPERM, so
    # mDNSResponder, i.e. DNS, is closed too) — and then exactly these LOOPBACK ports are allowed
    # again: the allowlist proxy's first, then e.g. a local model proxy's.
    egress_ports: list | None = None


def egress_section(ports) -> str:
    """The profile's network rules. LAST in the profile (see the rule-order note at the builder's
    return): SBPL takes the last matching rule, so nothing may follow the deny that could match
    network-outbound again. `ports` None -> a comment only (egress not restricted here)."""
    if ports is None:
        return ("; EGRESS: NOT restricted by this profile — network-outbound falls under "
                "(allow default)")
    bad = [p for p in ports if not (isinstance(p, int) and 1 <= p <= 65535)]
    if bad:
        raise ValueError(f"egress ports must be integers in 1..65535, got {bad!r}")
    lines = ["; EGRESS ENFORCED. Every outbound connection is denied — TCP, UDP and",
             "; AF_UNIX connects (DNS via mDNSResponder included) — except to these loopback",
             "; ports: the allowlist CONNECT proxy first, then any declared local model proxy.",
             "; LAST in the profile, so no later rule can re-open outbound (last match wins).",
             "(deny network-outbound)"]
    lines += [f'(allow network-outbound (remote ip "localhost:{p}"))'
              for p in dict.fromkeys(ports)]
    return "\n".join(lines)


# ── the PARENT session's environment ───────────────────────────────────────────────────────
# The orchestrator itself may run inside a Claude Code session, and every `claude` child (the evolver, the AO)
# would inherit that session's environment:
# its messaging socket and token (a live, non-file channel INTO the operator's session), its
# session and bridge ids, its background-job dir and kind, its pid and executable. None of it is
# the child's to use, and CLAUDE_JOB_DIR/CLAUDE_CODE_SESSION_KIND would make a child believe it
# is part of the operator's job. Names seen in os.environ of a real session:
# CLAUDECODE, CLAUDE_CODE_BRIDGE_SESSION_ID, CLAUDE_CODE_CHILD_SESSION, CLAUDE_CODE_ENTRYPOINT,
# CLAUDE_CODE_EXECPATH, CLAUDE_CODE_MESSAGING_SOCKET, CLAUDE_CODE_MESSAGING_TOKEN,
# CLAUDE_CODE_SESSION_ATTENDED, CLAUDE_CODE_SESSION_ID, CLAUDE_PID; CLAUDE_JOB_DIR and
# CLAUDE_CODE_SESSION_KIND are set for background jobs. Kept on purpose: auth
# (CLAUDE_CODE_OAUTH_TOKEN), the child's own CLAUDE_CODE_TMPDIR, and model/config variables.
# A new CLAUDE_CODE_* name about a session, messaging, a bridge, a parent or a job is caught by
# the pattern. Only NAMES are ever recorded; values are never printed.
#
# The IDE-integration channel too. Run from an IDE's terminal, the
# session exports CLAUDE_CODE_SSE_PORT (the IDE extension's MCP server) and may set
# ENABLE_IDE_INTEGRATION / CLAUDE_CODE_*IDE*; a child inheriting them attaches to the OPERATOR's
# IDE. Matched as whole `_`-separated tokens: a bare substring would also take *OVERRIDE* and
# *PROCESSES* names, which are configuration, not a channel.
PARENT_SESSION_ENV = (
    "CLAUDECODE", "CLAUDE_PID", "CLAUDE_JOB_DIR",
    "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_EXECPATH",
    "CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_SESSION_KIND", "CLAUDE_CODE_SESSION_ATTENDED",
    "CLAUDE_CODE_BRIDGE_SESSION_ID", "CLAUDE_CODE_CHILD_SESSION",
    "CLAUDE_CODE_MESSAGING_SOCKET", "CLAUDE_CODE_MESSAGING_TOKEN",
    "CLAUDE_CODE_SSE_PORT", "ENABLE_IDE_INTEGRATION",
)
_PARENT_SESSION_RE = re.compile(r"^CLAUDE_CODE_\w*(SESSION|MESSAGING|BRIDGE|PARENT|JOB)\w*$")
_PARENT_IDE_RE = re.compile(r"^CLAUDE_CODE_(?:\w+_)?(?:IDE|SSE)(?:_\w+)?$")


def parent_session_env_names(env=None) -> list[str]:
    """Sorted names in `env` (default os.environ) that identify or reach the parent session."""
    env = os.environ if env is None else env
    return sorted(k for k in env if k in PARENT_SESSION_ENV or _PARENT_SESSION_RE.match(k)
                  or _PARENT_IDE_RE.match(k))


def scrub_parent_session_env(env: dict) -> tuple[dict, list[str]]:
    """(a copy of `env` without the parent-session variables, the sorted names removed)."""
    names = parent_session_env_names(env)
    return ({k: v for k, v in env.items() if k not in names}, names)


# ── the CLI's per-uid temp root ─────────────────────────────────────────────────────────────
# `claude` keeps EVERY session's scratch under <root>/claude-<uid>,
# root = $CLAUDE_CODE_TMPDIR or /tmp: per-cwd-slug/<session>/tasks/*.output (the Bash output of
# every session on this host, the orchestrator's included), scratchpads, and shared entries
# (bundled-skills/, bash-edit-diff/, plugin-tool-staging/, fswatch probes). The profile's first
# rule allows all of /private, so all of it would be readable -- and writable -- by the evolver and
# the AO. The profile denies that tree outright, and the child is pointed at a PRIVATE root of
# its own instead (CLAUDE_CODE_TMPDIR, the CLI's documented override), granted back by name.
# Relocating rather than granting back "this session's slug dir" is deliberate: the CLI also
# uses the shared entries above, which a slug grant would silently break, and the slug rule is
# the CLI's to change.
#
# THE 44-BYTE BUDGET. The CLI gives child processes $TMPDIR = <root>/claude-<uid> only while
# that path is at most 44 bytes (AF_UNIX socket paths); beyond it the child's $TMPDIR falls
# back to /tmp/claude-<uid> — the very tree denied here. So the minted path is short, spelled
# via /tmp, and checked.
CLI_TMP_BUDGET = 44


def cli_uid_roots() -> list[Path]:
    """The host's per-uid CLI temp root(s) the profile denies: /tmp/claude-<uid> resolved (macOS
    resolves /tmp under /private), plus the one an inherited CLAUDE_CODE_TMPDIR would select."""
    uid = os.getuid()
    roots = {Path("/tmp").resolve() / f"claude-{uid}"}
    inherited = os.environ.get("CLAUDE_CODE_TMPDIR", "")
    if inherited:
        roots.add(Path(inherited).resolve() / f"claude-{uid}")
    return sorted(roots, key=str)


def mint_cli_tmpdir(tag: str) -> str:
    """A fresh, unused CLAUDE_CODE_TMPDIR for one child: /tmp/claude-<uid>/<tag>.<8 hex>.
    NOT created — the CLI creates it (and <it>/claude-<uid>) inside the grant. Raises when the
    CLI's own per-uid dir under it would exceed CLI_TMP_BUDGET."""
    import uuid
    if not tag or not tag.isalnum() or len(tag) > 4:
        raise ValueError(f"cli tmpdir tag {tag!r}: 1-4 alphanumerics")
    uid = os.getuid()
    p = f"/tmp/claude-{uid}/{tag}.{uuid.uuid4().hex[:8]}"
    if len(f"{p}/claude-{uid}".encode()) > CLI_TMP_BUDGET:
        raise ValueError(f"{p}/claude-{uid} exceeds the CLI's {CLI_TMP_BUDGET}-byte temp-dir "
                         f"budget; its child processes would fall back to the denied shared root")
    return p


# ── containment ─────────────────────────────────────────────────────────────────────────────

def build_evolver_sandbox_profile(job: EvolverJob, uid_roots=None) -> str:
    """Deny-read by default; grant ONLY what the evolver is entitled to.

    The inversion versus a solver's profile is the whole point: that one denies a
    fixed list of system subpaths and then allows the project root, so anything inside the repo
    is readable. This one allows a fixed list of SYSTEM paths needed to run a shell, denies
    everything else including the repository, and then grants exactly `job.grants`.

    `/Users` is denied as a subpath BEFORE the grants are allowed. sandbox-exec resolves by most
    specific match, so a later `(allow (subpath "<grant>"))` under /Users still wins for that
    subtree while its siblings stay denied. That ordering is the load-bearing detail and it is
    what `selftest()` proves, rather than assuming.

    `uid_roots` replaces the host's per-uid CLI temp root(s) (cli_uid_roots()) in the rules. Only
    a dry-run preflight passes it, to probe the rule shape on a throwaway stand-in.
    """
    allows = "\n".join(f'(allow file-read* (subpath "{p.resolve()}"))' for p in job.grants)
    # DENY THE RUN DIRECTORY BY NAME, exactly as no_write does for writes, and for exactly the
    # same reason: the block below denies reads by LOCATION -- /Users, /home, /opt, /Volumes --
    # and `(allow default)` covers everything else. A run directory outside those four prefixes
    # (a scratch mount, /var/folders, an external volume) is therefore FULLY READABLE, peer arms'
    # skills_v* and the split file with every held-out id included; the no-peer-read and
    # no-held-out-read properties would hold only because run dirs happen to live under /Users.
    # Declared, never derived -- the caller names what to hide, because
    # guessing a parent from the grants is how a deny lands on the wrong directory.
    hides = "\n".join(f'(deny file-read* (subpath "{p.resolve()}"))' for p in (job.hide or []))
    # The write deny by NAME, as a first-class field rather than string surgery at one call site. The write deny
    # below is by LOCATION (repo, home); a run dir outside both matches neither and `(allow
    # default)` makes it fully writable -- the operator's answer channel and every skills_v*
    # included. Emitted BEFORE `write_allows`, so the job's declared outputs
    # are re-granted afterwards and win (last match).
    no_writes = "\n".join(f'(deny file-write* (subpath "{p.resolve()}"))'
                          for p in (job.no_write or []))
    work = job.work_dir.resolve()
    # THE LOG'S PARENT DIRECTORY NODE, AS A LITERAL, NEVER AS A SUBPATH.
    # A log writer calls mkdir(exist_ok=True); CPython's exist_ok branch catches FileExistsError then
    # calls is_dir() to confirm, and a DENIED is_dir() makes it RE-RAISE.
    # Granting the run dir as a SUBPATH would make its split file readable, and the split file lists
    # every held-out id. A `literal` grant gives is_dir() and listing on that one node while every
    # file inside stays denied unless separately granted (digests, staging and the train-only oracle are).
    _logdir = job.log_path.resolve().parent
    # THE FILES THE CHILD'S OWN fds POINT AT -- every one of them, as LITERALS (the run dir holds the split).
    # A process fstat()s its inherited stdout/stderr at startup (Bun does it in internal:util/colors to detect
    # colour support), and fstat on an fd whose PATH the sandbox denies is EPERM: with the run dir denied by name,
    # the CLI would die before its first turn with `EPERM: operation not permitted, fstat`, a stack trace naming
    # no path.
    #
    # `.streamer.stderr` is the launcher's own convention for its stream branch's stderr sink,
    # named here rather than passed in because getting it WRONG is silent (the child dies at
    # startup) while getting it wrong in the launcher would at least be visible in argv. If that
    # suffix ever changes, this must change with it.
    #
    # STRUCTURAL NOTE: the child's log lives in the RUN DIR, which is exactly the tree containment must hide
    # (it holds the split), so the log is granted back file by file. Writing the log inside WORK (granted
    # read+write) and copying it out after would avoid that, at the cost of changing every consumer of the
    # log path.
    _lits = [_logdir, job.log_path.resolve(),
             job.log_path.resolve().with_name(job.log_path.name + ".streamer.stderr")]
    # The PROMPT FILE is the evolver's own input and is deliberately given to it. Granted as a
    # LITERAL FILE, so nothing else in the run dir (notably the split, which lists every held-out id)
    # becomes readable alongside it.
    if job.prompt_path:
        _lits.append(job.prompt_path.resolve())
    log_allow = "\n".join(f'(allow file-read* (literal "{x}"))' for x in _lits)
    # THE INTERPRETER MUST BE GRANTED OR NOTHING RUNS AT ALL. `/Users` is denied as a subpath, and an
    # interpreter under the home directory (a conda install, a venv) then cannot even be exec'd:
    #     sandbox-exec: execvp() of '.../bin/python3' failed: Operation not permitted
    #
    # Derived from sys.executable, NOT hardcoded: a hardcoded path would break the moment the
    # interpreter moved, and it would break silently-at-launch rather than loudly. Granting the
    # interpreter tree (and ~/.local) is safe: it is outside the repository, so held-out specs stay denied.
    _exe = Path(sys.executable).resolve()
    _interp = sorted({str(_exe.parents[1]), str(Path.home() / ".local")})
    interp_allows = "\n".join(f'(allow file-read* (subpath "{d}"))' for d in _interp)

    # THE REPO'S CODE IS NEEDED, AND THE REPO IS DENIED BY DESIGN. With only the interpreter granted, python
    # dies in `_path_importer_cache` with `PermissionError: [Errno 1] Operation not permitted`, because it
    # cannot even STAT its own sys.path entries; a python child needs to read the repo's CODE.
    #
    # The grant must not become "allow the repo", which is precisely the trap this profile
    # exists to avoid (every held-out spec lives under it).
    # So the grant is split by WHAT IS NEEDED, not by convenience:
    #   * file-read-METADATA on the repo root -> stat/scandir succeed, so imports resolve, but
    #     no file CONTENT is readable.
    #   * file-read* on src/ and scripts/ only -> the code, which is not data.
    # Held-out material lives in third_party/ (the task pool) and outputs/*/oracle/ (other runs'
    # oracles). Neither is granted, so both remain unreadable — which selftest() proves by
    # probing a real forbidden path AFTER these grants exist, not before.
    _repo = Path(__file__).resolve().parents[2]
    repo_allows = "\n".join([
        # LITERAL, not subpath: grants the repo-root DIRECTORY NODE so python can enumerate it,
        # without granting anything inside. `file-read-metadata (subpath ...)` is NOT sufficient --
        # macOS requires file-read-DATA on a directory to list it, and python
        # lists every sys.path entry. Measured: metadata-only died in _path_importer_cache.
        f'(allow file-read* (literal "{_repo}"))',
        f'(allow file-read* (subpath "{_repo / "src"}"))',
        f'(allow file-read* (subpath "{_repo / "scripts"}"))',
    ])
    # WRITE CONTAINMENT. Files the loop TRUSTS (the operator's answers, the run's records) live in the run
    # dir; an agent with write access to it could satisfy its own gate, e.g. ask itself a question and
    # answer it. Read-containment cannot express that; only write-containment can.
    #
    # Deny-then-grant, same shape as the reads, same most-specific-match resolution. The grants
    # are deliberately narrow: the job's own work dir, its log's directory, and whatever it
    # explicitly declared in `writes`. Everything else under /Users is denied, INCLUDING the run
    # directory.
    #
    # ~/.claude is granted because the CLI writes its own session/telemetry state there and dies
    # without it (verified by probe). That is a real widening and it is why the
    # answer channel must ALSO live outside any agent-writable path rather than relying on this
    # list alone: defence in depth, because one forgotten grant here re-opens the boundary.
    # THE LOG IS GRANTED AS A LITERAL FILE, NEVER VIA ITS PARENT: log_path.parent IS THE RUN DIR, and a
    # subpath grant for a file that lives in a sensitive directory grants the directory (the same rule as
    # the read side's log node above).
    _wr_sub = [work] + [Path(w).resolve() for w in (job.writes or [])]
    _wr_lit = [job.log_path.resolve()]
    _home_claude = Path.home() / ".claude"
    write_allows = "\n".join(
        [f'(allow file-write* (subpath "{d}"))' for d in sorted({str(x) for x in _wr_sub})]
        + [f'(allow file-write* (literal "{f}"))' for f in sorted({str(x) for x in _wr_lit})]
        + [f'(allow file-write* (subpath "{_home_claude}"))'])
    # ~/.claude, WRITE SIDE. The ~/.claude write allow above is a blanket, so every
    # evolver and AO could rewrite what the OPERATOR's own sessions load and run: settings.json
    # (hooks run unsandboxed in the operator's session, where the split and the val/sealed results
    # are readable, and could drop them where a later evolver reads them), statusline scripts,
    # CLAUDE.md, user-level skills/ commands/ agents/ (one arm contaminating every later session),
    # plugins/ (plugin hooks), the local install and the Chrome native-host wrapper — and other
    # jobs' work dirs under jobs/, whose READ is taken back below too. Denied by name AFTER the
    # allow; the job's own declared writes are then granted AGAIN below, because an evolver's WORK
    # lives under jobs/ and must still win (last match). The list follows the config the CLI's own
    # sandbox write-protects (skills, hooks, commands, agents, workflows, routines, rules,
    # output-styles, plugins, settings*, CLAUDE.md, launch/scheduled_tasks/daemon/policy-limits
    # json, local, jobs, chrome), minus what the CLI itself writes at run time (projects/ for its
    # transcript, shell-snapshots/, session-env/, sessions/, backups/, todos/, statsig state).
    _cfg_re = re.escape(str(_home_claude))
    _cli_roots_w = cli_uid_roots() if uid_roots is None else [Path(r) for r in uid_roots]
    write_denies = "\n".join(
        [f'(deny file-write* (subpath "{_home_claude / n}"))'
         for n in ("skills", "hooks", "commands", "agents", "workflows", "routines", "rules",
                   "output-styles", "plugins", "cowork_plugins", "local", "chrome", "jobs")]
        + [f'(deny file-write* (regex #"^{_cfg_re}/(settings[^/]*|statusline[^/]*|[^/]*\\.md'
           f'|scheduled_tasks\\.json|launch\\.json|daemon\\.json|policy-limits\\.json)$"))'])
    # The host's per-uid CLI temp root (see cli_uid_roots), denied BY NAME on both sides: other
    # sessions' Bash task outputs and scratchpads — the orchestrator's included, which the
    # operator's session reads back, so a write there is a channel INTO the operator's session.
    # Emitted right after the location-based denies, BEFORE every narrower allow, so that the
    # child's own private root (cli_tmpdir, granted back below) and any declared grant win.
    cli_root_write_denies = "\n".join(f'(deny file-write* (subpath "{r}"))' for r in _cli_roots_w)
    cli_root_read_denies = "\n".join(f'(deny file-read* (subpath "{r}"))' for r in _cli_roots_w)
    _cli_tmp = Path(job.cli_tmpdir).resolve() if job.cli_tmpdir else None
    write_reallows = "\n".join(
        [f'(allow file-write* (subpath "{d}"))'
         for d in sorted({str(x) for x in _wr_sub + ([_cli_tmp] if _cli_tmp else [])})]
        + [f'(allow file-write* (literal "{f}"))' for f in sorted({str(x) for x in _wr_lit})])
    # The child CLI's own temp root (CLAUDE_CODE_TMPDIR) for READ, after the per-uid root deny;
    # plus the per-uid root NODE for metadata only (the CLI realpaths and stats its temp dir's
    # ancestors), never for listing — the root's entries name other sessions.
    cli_tmp_allows = "\n".join(
        [f'(allow file-read-metadata (literal "{r}"))' for r in _cli_roots_w]
        + ([f'(allow file-read* (subpath "{_cli_tmp}"))'] if _cli_tmp else []))

    # ── WHAT THE `claude` CLI ITSELF NEEDS TO RUN AT ALL ─────────────────────────────────
    #
    # Two requirements a probe of /bin/cat or /bin/sh never shows:
    #
    #   1. AUTHENTICATION. The CLI must READ ~/.claude (not only write it), and OAuth credentials live
    #      in the login keychain under ~/Library/Keychains. Without them: `Not logged in · Please run
    #      /login`, exit 1.
    #
    #   2. DENIED != ABSENT ON THE CONFIG SEARCH PATH. The CLI walks from cwd up to $HOME looking
    #      for configuration. A MISSING directory is fine; a DENIED one raises EPERM and kills the
    #      process outright — `error: An internal error occurred (EPERM)`, with no indication of
    #      which path. Since `(deny file-read* (subpath "/Users"))` denies every ancestor, the CLI
    #      cannot start regardless of what is granted below. Granting exactly the ancestor DIRECTORY
    #      NODES (literals -- the node, never the tree) is sufficient (bisected).
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
    # "Under HOME" is only as good as $HOME. With HOME at or above the
    # per-uid CLI temp root (HOME=/tmp, resolved), a work dir beneath that root (the AO's private
    # CLI temp root is one) would put the root itself in this list -- a file-read* literal AFTER the
    # root's deny, so every session's slug dir would be listable again. Whatever HOME says: no node
    # that is a per-uid root or lies above one. The root keeps its metadata-only grant
    # (cli_tmp_allows) for the CLI's realpath; nodes INSIDE it stay, as the CLI needs them when
    # HOME itself lies there (the fixtures do).
    _rr = [Path(r).resolve() for r in _cli_roots_w]
    _anc = [a for a in _anc if not any(Path(a) == r or Path(a) in r.parents for r in _rr)]
    # THE REST OF ~/.claude THAT IS OTHER SESSIONS' MATERIAL, taken back like
    # projects/. The subpath read grant above re-denied only projects/, which left readable:
    #   jobs/          every other job's scratch — other runs' evolver WORK dirs with their
    #                  PROMPT.md (from epoch 2 on, another arm's prior skill set word for word) and
    #                  scratch/, and the orchestrator's own job dirs, which may quote val and
    #                  held-out test results;
    #   history.jsonl, file-history/, paste-cache/, debug/, sessions/, backups/
    #                  prompts, file snapshots and pastes from every session on this host.
    # NOT shell-snapshots/: the CLI writes its OWN snapshot there and every
    # Bash call `source`s it with the failure swallowed, so a deny silently strips the child's
    # PATH export and its find/grep/rg/pkill wrappers. The files hold aliases, functions and
    # PATH, not conversations; the CLI's own sandbox re-opens that dir for reads for this reason.
    # NOT a deny of all of ~/.claude: the CLI's startup needs (settings, credentials, plugins,
    # statsig state, ...) were found by bisection and cannot be re-bisected without a live launch.
    # Narrow grants that come LATER still win (last match): this job's WORK dir (under jobs/),
    # this run's own PROJDIR, the AO's own transcript dir. The ancestor NODES below are emitted
    # AFTER these denies for the same reason — the evolver's cwd lives under jobs/, and a denied
    # ancestor of cwd is a fatal EPERM -- so jobs/ and jobs/<id>/ stay listable (names only).
    _other_sessions = [f'(deny file-read* (subpath "{_home_claude / n}"))'
                       for n in ("jobs", "file-history", "paste-cache",
                                 "debug", "sessions", "backups")]
    _other_sessions.append(f'(deny file-read* (literal "{_home_claude / "history.jsonl"}"))')
    cli_allows = "\n".join(
        ["; the CLI's own installation, config and credentials. Without these it reports",
         "; `Not logged in` and exits 1 — the write grant on ~/.claude is not sufficient.",
         f'(allow file-read* (subpath "{Path.home() / ".local"}"))',
         f'(allow file-read* (subpath "{_home_claude}"))',
         f'(allow file-read* (literal "{Path.home() / ".claude.json"}"))',
         f'(allow file-read* (subpath "{Path.home() / "Library" / "Keychains"}"))',
         "; ... but NOT other sessions' transcripts. Granted above by the ~/.claude subpath;",
         "; taken back here, because that tree holds every past run's conversation on this host.",
         f'(deny file-read* (subpath "{_home_claude / "projects"}"))',
         "; nor other jobs' scratch, the prompt history, file snapshots, pastes, debug",
         "; logs, session state or backups of any session on this host."]
        + _other_sessions
        + ["; the ancestor directory NODES of cwd and the repo, as literals. The CLI walks up",
           "; from cwd looking for config; a DENIED ancestor is EPERM (fatal), unlike an absent one.",
           "; AFTER the denies above, so an ancestor under ~/.claude/jobs stays stat-able."]
        + [f'(allow file-read* (literal "{a}"))' for a in _anc])

    # RULE ORDER IS THE WHOLE PROFILE. SBPL takes the LAST matching rule, and the broad "minimum
    # needed to run a shell" allow includes (subpath "/var") and (subpath "/private"). At the BOTTOM it
    # would override every deny above it for those trees, silently (a deny-by-name on a run dir in
    # /var/folders would leave the file readable, including through Bash).
    #
    # The order below is broad-allow FIRST, then denies, then narrow grants: each later rule
    # refines the one before it, which is the only arrangement in which "deny X, then grant X/y"
    # means what it reads like. The system paths are allowed, just not LAST.
    return f"""(version 1)
(allow default)
; the minimum needed to run a shell and a python. FIRST, so the denies and grants below can
; refine it (last match wins: at the bottom it would override every deny).
(allow file-read* (subpath "/usr") (subpath "/bin") (subpath "/sbin")
  (subpath "/System") (subpath "/Library") (subpath "/var") (subpath "/private")
  (subpath "/dev") (subpath "/etc"))
(deny file-write*
  (subpath "{_repo}")
  (subpath "{Path.home()}")
)
; the host's per-uid CLI temp root (every session's task outputs), denied BY NAME
{cli_root_write_denies}
; no-write paths denied BY NAME, so write containment does not depend on where the run dir happens to sit
{no_writes}
; the ONLY writable paths: this job's work dir, its log dir, its declared outputs, CLI state
{write_allows}
; ...but not the operator's config and code under ~/.claude, or other jobs.
; Then this job's own writes AGAIN (and its CLI temp root), so they win (last match).
{write_denies}
{write_reallows}
(deny file-read*
  (subpath "/Users")
  (subpath "/home")
  (subpath "/opt")
  (subpath "/Volumes")
)
; the host's per-uid CLI temp root (every session's task outputs and scratchpads), BY NAME
{cli_root_read_denies}
; hidden paths denied BY NAME, so containment does not depend on where the run dir happens to sit.
; The explicit grants below come later and win for their own leaves (last match wins).
{hides}
; the interpreter and its library tree — without these sandbox-exec cannot even exec python
{interp_allows}
; what the `claude` CLI needs to start and authenticate — see the comment at the builder
{cli_allows}
; the repo: METADATA everywhere (so imports can stat sys.path), CONTENT only under src/ scripts/
{repo_allows}
; the evolver's own working directory — it must be able to read what it writes
(allow file-read* (subpath "{work}"))
; the per-uid CLI temp root as a NODE (metadata only), and the child's own CLI temp root
{cli_tmp_allows}
; the log's parent DIRECTORY NODE only -- is_dir() must succeed; contents stay denied
{log_allow}
; explicit grants: the evidence surface, the staging dir, the train-only oracle. LAST among the
; file rules, so they win over both the system allow above and the deny-by-name (last match).
{allows}
{egress_section(job.egress_ports)}
"""


def containment_probe(job: EvolverJob, target: Path) -> tuple[bool, str]:
    """Actually attempt to read `target` under this job's profile. Returns (readable, detail).

    EXECUTES the read; it does not reason about the profile. A containment claim that was never
    exercised is a guard that cannot fail, and this module's whole premise is that containment
    must demonstrate refusal.
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

    Mirrors containment_probe deliberately: the failure it exists to prevent is a check that
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

    WHY THIS EXISTS. A child can die on any one missing grant -- the interpreter, the repo listing,
    the log dir's is_dir(), the prompt file -- and a launch discovers only one gap at a time. So:
    probe the whole set before spending a dispatch, and report every failure together.

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
        # 2. (a python runner module would be probed for IMPORT here; the shipped children are the
        #    `claude` CLI, which imports nothing from the repo)
        # 3. every declared READ must succeed
        reads = [job.prompt_path] + list(job.grants) if job.prompt_path else list(job.grants)
        for p in reads:
            if p is None or not Path(p).exists():
                continue
            tgt = p if Path(p).is_file() else next(iter(sorted(Path(p).rglob("*"))), None)
            # An all-epochs digest dir holds its producing library in a
            # _library/ subdir, and "_" sorts before lowercase task names (task_004.md), so the
            # first entry is that DIRECTORY and the grant would go unprobed. Probe a task file.
            if tgt is not None and Path(tgt).is_dir() and Path(tgt).name == "_library":
                tgt = next(iter(sorted(Path(p).glob("*.md"))), None)
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
            bad.append("log parent fails is_dir() — mkdir(exist_ok=True) will re-raise")
        # 5. the host's per-uid CLI temp root — every session's Bash
        #    task outputs — must NOT be listable, while its NODE stays stat-able for the CLI's
        #    realpath of its own private root beneath it. Creates nothing.
        #    A DRY RUN never lists the REAL root (other sessions' material, the operator's
        #    included): it checks the real root's rules are in the profile TEXT, and probes the
        #    same rules, built around a throwaway stand-in root, by execution.
        if job.dry_run:
            bad += _probe_uid_root_standin(job, build_evolver_sandbox_profile(job))
        for root in ([] if job.dry_run else cli_uid_roots()):
            if not root.is_dir():
                continue
            r = subprocess.run(["sandbox-exec", "-f", prof, "/bin/ls", str(root)],
                               capture_output=True, text=True, timeout=30)
            if r.returncode == 0:
                bad.append(f"the per-uid CLI temp root {root} is LISTABLE — other sessions' task "
                           f"outputs and scratchpads would be readable")
            if job.cli_tmpdir:
                r = subprocess.run(["sandbox-exec", "-f", prof, "/usr/bin/stat", str(root)],
                                   capture_output=True, text=True, timeout=30)
                if r.returncode:
                    bad.append(f"the per-uid CLI temp root {root} cannot be stat()ed — the CLI "
                               f"could not realpath its private temp root beneath it")
        # 6. every declared NO-WRITE (the run dir, the operator
        #    channel) must refuse a fresh name under the REAL profile — the write allows and the
        #    grants come later in it and could undo the deny. An existing dir is probed with a new
        #    file inside it; an absent one with its own mkdir (the evolver could create it). A
        #    probe that holds creates nothing; one that got through is removed and reported.
        bad += _probe_no_writes(job, prof)
    finally:
        os.unlink(prof)
    return bad


def _probe_no_writes(job: EvolverJob, prof: str) -> list[str]:
    bad: list[str] = []
    for d in (Path(p) for p in (job.no_write or [])):
        existed = d.is_dir()
        if d.exists() and not existed:
            continue                      # a file: nothing to create inside it
        target = d / f".evodisp_nowrite_probe.{os.getpid()}" if existed else d
        script = ': > "$1"' if existed else 'mkdir "$1"'
        r = subprocess.run(["sandbox-exec", "-f", prof, "/bin/sh", "-c", script, "sh",
                            str(target)], capture_output=True, text=True, timeout=30)
        if r.returncode == 0:
            bad.append(f"a declared no-write path is WRITABLE under the profile: {d} (a probe "
                       f"{'file was created in it' if existed else 'mkdir created it'}; removed)")
            try:
                target.unlink() if existed else target.rmdir()
            except OSError:
                pass
    return bad


def _probe_uid_root_standin(job: EvolverJob, real_profile: str) -> list[str]:
    """Preflight step 5 for a DRY RUN. The real per-uid root is only READ as profile text;
    the listing and stat probes run against a stand-in root under $TMPDIR, denied by a profile
    built for this job with the stand-in in place of the real root. Removed afterwards."""
    bad: list[str] = []
    for root in cli_uid_roots():
        if f'(deny file-read* (subpath "{root}"))' not in real_profile:
            bad.append(f"the profile does not deny reads of the per-uid CLI temp root {root}")
    st = Path(tempfile.mkdtemp(prefix="evodisp_uidroot.")).resolve()
    try:
        (st / "-other-session" / "tasks").mkdir(parents=True)
        (st / "-other-session" / "tasks" / "t.output").write_text("x")
        with tempfile.NamedTemporaryFile("w", suffix=".sb", delete=False) as fh:
            fh.write(build_evolver_sandbox_profile(job, uid_roots=[st]))
            prof = fh.name
        try:
            r = subprocess.run(["sandbox-exec", "-f", prof, "/bin/ls", str(st)],
                               capture_output=True, text=True, timeout=30)
            if r.returncode == 0:
                bad.append("the per-uid CLI temp root rules leave the root LISTABLE (probed on a "
                           "dry-run stand-in) — other sessions' task outputs would be readable")
            r = subprocess.run(["sandbox-exec", "-f", prof, "/bin/cat",
                                str(st / "-other-session" / "tasks" / "t.output")],
                               capture_output=True, text=True, timeout=30)
            if r.returncode == 0:
                bad.append("another session's task output under the per-uid CLI temp root is "
                           "READABLE (probed on a dry-run stand-in)")
            if job.cli_tmpdir:
                r = subprocess.run(["sandbox-exec", "-f", prof, "/usr/bin/stat", str(st)],
                                   capture_output=True, text=True, timeout=30)
                if r.returncode:
                    bad.append("the per-uid CLI temp root cannot be stat()ed (probed on a dry-run "
                               "stand-in) — the CLI could not realpath its private temp root")
        finally:
            os.unlink(prof)
    finally:
        shutil.rmtree(st, ignore_errors=True)
    return bad


def _sandbox_exec_available() -> bool:
    return shutil.which("sandbox-exec") is not None


# ── egress probes ──────────────────────────────────────────────────────────────
# Executed under the EMITTED profile file, never reasoned from its text. The child is this
# module's own interpreter (granted by the profile: interp_allows), so it needs nothing else.
_EGRESS_CHILD = r"""
import errno, json, socket, sys
cfg = json.loads(sys.argv[1])
def err(e):
    return errno.errorcode.get(e.errno, str(e)) if getattr(e, "errno", None) else type(e).__name__
def connect(fam, addr):
    s = socket.socket(fam, socket.SOCK_STREAM); s.settimeout(3)
    try:
        s.connect(addr); return "connected"
    except OSError as e:
        return err(e)
    finally:
        s.close()
def via_proxy(target):
    try:
        s = socket.create_connection(("127.0.0.1", cfg["proxy"]), timeout=5)
    except OSError as e:
        return "no proxy: " + err(e)
    try:
        s.sendall(("CONNECT %s HTTP/1.1\r\nHost: %s\r\n\r\n" % (target, target)).encode())
        buf = b""
        while b"\r\n" not in buf and len(buf) < 4096:
            c = s.recv(4096)
            if not c: break
            buf += c
        return buf.split(b"\r\n", 1)[0].decode("latin-1") or "no response"
    except OSError as e:
        return "error: " + err(e)
    finally:
        s.close()
out = {
  "a_other_loopback_port": connect(socket.AF_INET, ("127.0.0.1", cfg["other"])),
  "b_proxy_port": connect(socket.AF_INET, ("127.0.0.1", cfg["proxy"])),
  "c_proxy_refuses_host": via_proxy("%s:443" % cfg["host"]),
  "c_proxy_refuses_ip_literal": via_proxy("127.0.0.1:%d" % cfg["other"]),
  "d_offhost_ipv4": connect(socket.AF_INET, (cfg["ip"], 443)),
  "d_offhost_ipv6": connect(socket.AF_INET6, (cfg["ip6"], 443)),
  "e_unix_socket": connect(socket.AF_UNIX, cfg["unix"]),
}
for p in cfg["extra"]:
    out["f_local_port_%d" % p] = connect(socket.AF_INET, ("127.0.0.1", p))
print(json.dumps(out))
"""


def egress_probes(profile: Path, proxy_port: int, extra_ports=(), probe_host: str = "example.com",
                  probe_ip: str = "192.0.2.1", probe_ip6: str = "2001:db8::1") -> list[dict]:
    """Run the egress probes under `profile` and return one dict per probe:
    {"probe", "expect", "got", "ok"}. Creates two throwaway listeners (a TCP port on 127.0.0.1
    that is NOT the proxy's, and an AF_UNIX socket under /tmp) and removes them afterwards.

      a  a TCP connect to the throwaway loopback port is refused by the kernel (EPERM)
      b  the proxy port accepts
      c  CONNECT through the proxy to a non-allowlisted host, and to an IP literal, gets 403 —
         and the throwaway listener the IP-literal CONNECT names received NO connection
      d  a raw connect to an off-host IP literal (v4 and v6; documentation ranges by default,
         so even a broken profile contacts no real host) is refused by the kernel (EPERM)
      e  an AF_UNIX connect is refused by the kernel (EPERM)
      f  each extra declared local port is NOT refused by the kernel (connected or ECONNREFUSED)
    """
    lst = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    lst.bind(("127.0.0.1", 0))
    lst.listen(16)
    other = lst.getsockname()[1]
    udir = tempfile.mkdtemp(prefix="egp", dir="/tmp")
    upath = os.path.join(udir, "u.sock")
    ulst = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    ulst.bind(upath)
    ulst.listen(4)
    cfg = {"proxy": int(proxy_port), "other": other, "host": probe_host, "ip": probe_ip,
           "ip6": probe_ip6, "unix": upath, "extra": [int(p) for p in extra_ports]}
    try:
        # the RESOLVED interpreter: a venv's bin/python reads pyvenv.cfg inside the (denied) repo,
        # its target lives in the tree interp_allows grants
        # -I and cwd "/": the child imports nothing from, and never stats, whatever cwd the caller
        # had (the repo is denied by the profile)
        r = subprocess.run(["sandbox-exec", "-f", str(profile), str(Path(sys.executable).resolve()),
                            "-I", "-c", _EGRESS_CHILD, json.dumps(cfg)],
                           capture_output=True, text=True, timeout=120, cwd="/")
        try:
            got = json.loads(r.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return [{"probe": "probe child", "expect": "JSON results", "ok": False,
                     "got": f"rc={r.returncode} {(r.stderr or r.stdout).strip()[-300:]}"}]
        accepted = []
        for s, name in ((lst, "tcp"), (ulst, "unix")):
            s.setblocking(False)
            try:
                c, _ = s.accept()
                c.close()
                accepted.append(name)
            except (BlockingIOError, OSError):
                pass
    finally:
        lst.close()
        ulst.close()
        shutil.rmtree(udir, ignore_errors=True)

    def row(probe, expect, ok):
        return {"probe": probe, "expect": expect, "got": got.get(probe), "ok": bool(ok)}
    rows = [
        row("a_other_loopback_port", "EPERM", got.get("a_other_loopback_port") == "EPERM"),
        row("b_proxy_port", "connected", got.get("b_proxy_port") == "connected"),
        row("c_proxy_refuses_host", "HTTP 403",
            str(got.get("c_proxy_refuses_host", "")).split(" ")[1:2] == ["403"]),
        row("c_proxy_refuses_ip_literal", "HTTP 403",
            str(got.get("c_proxy_refuses_ip_literal", "")).split(" ")[1:2] == ["403"]),
        row("d_offhost_ipv4", "EPERM", got.get("d_offhost_ipv4") == "EPERM"),
        row("d_offhost_ipv6", "EPERM", got.get("d_offhost_ipv6") == "EPERM"),
        row("e_unix_socket", "EPERM", got.get("e_unix_socket") == "EPERM"),
    ]
    rows.append({"probe": "c_no_upstream_connection", "expect": "throwaway listeners accepted "
                 "nothing", "got": accepted or "nothing accepted", "ok": not accepted})
    for p in cfg["extra"]:
        k = f"f_local_port_{p}"
        rows.append(row(k, "not EPERM (connected or ECONNREFUSED)",
                        got.get(k) in ("connected", "ECONNREFUSED")))
    return rows



# ── selftest ────────────────────────────────────────────────────────────────────────────────

def selftest() -> int:
    """Both halves matter: forbidden must FAIL and granted must SUCCEED.

    A profile that denies everything would pass a refusal-only test while making the evolver
    useless, and would look like security. That is why the granted-read assertion is here.
    """
    checks: list[tuple[str, bool]] = []
    # FIXTURES MUST LIVE WHERE THE REAL THING LIVES. `tempfile.mkdtemp()` on macOS returns a path
    # under /var/folders, which the profile legitimately allows so a shell can run; a "forbidden"
    # fixture there would be readable while the profile is correct for the repository under $HOME.
    # A guard proved against the wrong geometry is not proved. Fixtures therefore go under $HOME,
    # the tree the profile denies.
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

        # EXEC of the interpreter. Every other check here exercises READS under the profile; a
        # profile that denied `sys.executable` would pass them all and then die at launch. Proving
        # a guard against the wrong operation is the same error as proving it against the wrong
        # geometry (the tempfile-under-/var case above).
        if _sandbox_exec_available():
            with tempfile.NamedTemporaryFile("w", suffix=".sb", delete=False) as fh:
                fh.write(build_evolver_sandbox_profile(job))
                _prof = fh.name
            try:
                r = subprocess.run(["sandbox-exec", "-f", _prof, sys.executable, "-c",
                                    "print('exec-ok')"], capture_output=True, text=True, timeout=60)
                checks.append(("sys.executable can EXEC under the profile",
                               r.returncode == 0 and "exec-ok" in r.stdout))
            finally:
                os.unlink(_prof)

        checks.append(("profile denies /Users before granting",
                       '(subpath "/Users")' in build_evolver_sandbox_profile(job)))
        # The trap is `file-read* (subpath "<repo>")` -- content on the whole tree. A `literal`
        # grant on the root NODE is not the trap and is required for imports. Testing the intent,
        # not the substring, so a legitimate repo grant never pressures a weaker check. Also assert
        # positively that the held-out subtree is NOT granted.
        _prof = build_evolver_sandbox_profile(job)
        _repo_s = str(Path(__file__).resolve().parents[2])
        checks.append(("profile does NOT grant CONTENT on the repo subtree (the reuse trap)",
                       f'(allow file-read* (subpath "{_repo_s}"))' not in _prof))
        checks.append(("third_party/ (the task pool) is NOT granted",
                       "third_party" not in _prof))
        checks.append(("grants appear in the profile",
                       str(granted.resolve()) in build_evolver_sandbox_profile(job)))
        if _sandbox_exec_available():
            checks += _selftest_claude_dir(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    bad = 0
    for name, good in checks:
        print(f"  [{'ok' if good else 'BAD'}] {name}")
        bad += 0 if good else 1
    print(f"\n{len(checks)} checks, {bad} bad")
    return 1 if bad else 0


def _selftest_claude_dir(tmp: Path) -> list[tuple[str, bool]]:
    """EXECUTED: the operator's config under ~/.claude cannot be
    WRITTEN, the host's per-uid CLI temp root cannot be read or written, and the child's own
    private CLI temp root beneath it can be. HOME points at a fixture for the whole block, so a
    broken deny writes into the fixture, never into the real ~/.claude.

    The per-uid-root fixture is created under $EVOSKILL_DISPATCH_SELFTEST_TMP when set (it must
    lie under /tmp/claude-<uid>), else directly under that root; it is removed afterwards."""
    out: list[tuple[str, bool]] = []
    home = tmp / "home"
    cl = home / ".claude"
    (cl / "skills").mkdir(parents=True)
    (cl / "settings.json").write_text("{}")
    root = cli_uid_roots()[0]
    parent = Path(os.environ.get("EVOSKILL_DISPATCH_SELFTEST_TMP") or root)
    parent.mkdir(parents=True, exist_ok=True)
    fx = Path(tempfile.mkdtemp(prefix="evodisp.", dir=str(parent))).resolve()
    old_home = os.environ.get("HOME")
    try:
        other = fx / "-other-session" / "tasks"
        other.mkdir(parents=True)
        (other / "t.output").write_text("SEALED TEST 12/40")
        own = fx / "ev.selftest"
        os.environ["HOME"] = str(home)
        job = EvolverJob(epoch=1, work_dir=tmp / "work", prompt="p", model="m",
                         log_path=tmp / "e.log", cli_tmpdir=own)
        with tempfile.NamedTemporaryFile("w", suffix=".sb", delete=False) as fh:
            fh.write(build_evolver_sandbox_profile(job))
            prof = fh.name
        try:
            def sb(cmd: str) -> bool:
                return subprocess.run(["sandbox-exec", "-f", prof, "/bin/sh", "-c", cmd],
                                      capture_output=True, text=True, timeout=30).returncode == 0
            out.append(("~/.claude/settings.json is NOT writable (hooks run in operator sessions)",
                        not sb(f'echo x > "{cl / "settings.json"}"')
                        and (cl / "settings.json").read_text() == "{}"))
            out.append(("~/.claude/skills/ is NOT writable (user skills load in every session)",
                        not sb(f'mkdir -p "{cl / "skills" / "evil"}"')
                        and not (cl / "skills" / "evil").exists()))
            out.append(("another session's task output under the per-uid root is NOT readable",
                        not sb(f'cat "{other / "t.output"}"')))
            out.append(("...nor writable", not sb(f'echo x >> "{other / "t.output"}"')
                        and (other / "t.output").read_text() == "SEALED TEST 12/40"))
            # Created the way the CLI creates it — mkdir of the missing leaves only. NOT
            # `mkdir -p`: BSD mkdir -p re-mkdirs every EXISTING ancestor, and the kernel answers
            # a write-denied ancestor with EPERM before EEXIST.
            out.append(("the child's OWN CLI temp root is creatable, writable and readable",
                        sb(f'mkdir "{own}" && mkdir "{own}/claude-1" && echo ok > '
                           f'"{own}/claude-1/o" && cat "{own}/claude-1/o"')))
        finally:
            os.unlink(prof)
    finally:
        if old_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = old_home
        shutil.rmtree(fx, ignore_errors=True)
    return out


def _cli() -> int:
    """Emit a profile / run a preflight for a caller that is NOT python -- a shell launcher.

    A launcher must NOT build the profile itself: a second spelling of these rules would drift
    silently from this one. So the shell asks this module for the bytes.

    THE PROFILE WRAPS THE `claude` CHILD, NOT THE LAUNCHER. A launcher may READ THE SPLIT FILE for
    its own checks, and that file lists every held-out id -- the precise artifact the child's
    containment exists to hide. Launcher and child have opposite access needs; only the child gets
    confined.
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
                    help="subtree to deny READ by name (e.g. the run dir); repeatable")
    ap.add_argument("--no-write", action="append", default=[],
                    help="subtree to deny WRITE by name (e.g. the run dir); repeatable")
    ap.add_argument("--cli-tmpdir", default="",
                    help="the claude child's CLAUDE_CODE_TMPDIR, granted read+write after "
                         "the per-uid temp root is denied")
    ap.add_argument("--mint-cli-tmpdir", metavar="TAG", default="",
                    help="print a fresh CLAUDE_CODE_TMPDIR (/tmp/claude-<uid>/<TAG>.<hex>) and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="the preflight probes a stand-in for the per-uid CLI temp root, "
                         "never the real one")
    ap.add_argument("--parent-session-env", action="store_true",
                    help="print the NAMES (never values) of the parent-session variables in "
                         "this environment, one per line, and exit")
    # EGRESS. --egress-proxy-port switches the profile's egress section ON: deny all
    # network-outbound, allow that loopback port and each --egress-local-port.
    ap.add_argument("--egress-proxy-port", type=int, default=0,
                    help="egress ENFORCED: the allowlist proxy's loopback port")
    ap.add_argument("--egress-local-port", type=int, action="append", default=[],
                    help="another loopback port the child may connect to (e.g. a local model "
                         "proxy's); repeatable; needs --egress-proxy-port")
    ap.add_argument("--egress-probe", metavar="PROFILE", default="",
                    help="run the egress probes under this EMITTED profile and exit (9 on any "
                         "failed probe); needs --egress-proxy-port")
    ap.add_argument("--egress-probe-host", default="example.com",
                    help="a NON-allowlisted hostname the proxy must refuse (never resolved)")
    ap.add_argument("--egress-probe-ip", default="192.0.2.1",
                    help="an off-host IPv4 literal the kernel must refuse (default: TEST-NET-1, so "
                         "even a broken profile contacts no real host)")
    ap.add_argument("--egress-probe-ip6", default="2001:db8::1",
                    help="an off-host IPv6 literal the kernel must refuse (default: the "
                         "documentation prefix)")
    ap.add_argument("--egress-probe-out", default="", help="also write the probe rows as JSON")
    a = ap.parse_args()
    if a.parent_session_env:
        for n in parent_session_env_names():
            print(n)
        return 0
    if a.egress_local_port and not a.egress_proxy_port:
        print("REFUSING: --egress-local-port without --egress-proxy-port", file=sys.stderr)
        return 64
    if a.egress_probe:
        if not a.egress_proxy_port:
            print("--egress-probe needs --egress-proxy-port", file=sys.stderr)
            return 64
        if not _sandbox_exec_available():
            print("REFUSING: sandbox-exec is unavailable; egress cannot be probed.",
                  file=sys.stderr)
            return 9
        rows = egress_probes(Path(a.egress_probe), a.egress_proxy_port, a.egress_local_port,
                             probe_host=a.egress_probe_host, probe_ip=a.egress_probe_ip,
                             probe_ip6=a.egress_probe_ip6)
        for r in rows:
            print(f"  egress probe: [{'ok ' if r['ok'] else 'BAD'}] {r['probe']}: "
                  f"expect {r['expect']}, got {r['got']}")
        if a.egress_probe_out:
            Path(a.egress_probe_out).write_text(json.dumps(rows, indent=1))
        bad = [r for r in rows if not r["ok"]]
        if bad:
            print(f"EGRESS PROBES FAILED: {len(bad)} of {len(rows)} — egress is NOT enforced as "
                  f"declared", file=sys.stderr)
            return 9
        return 0
    if a.mint_cli_tmpdir:
        try:
            print(mint_cli_tmpdir(a.mint_cli_tmpdir))
        except ValueError as e:
            print(f"REFUSING: {e}", file=sys.stderr)
            return 9
        return 0
    if a.selftest or not (a.emit_profile or a.preflight):
        return selftest()
    if not (a.work_dir and a.log):
        print("--work-dir and --log are required", file=sys.stderr)
        return 64
    job = EvolverJob(
        epoch=a.epoch, work_dir=Path(a.work_dir), prompt="", model="",
        log_path=Path(a.log),
        prompt_path=Path(a.prompt_path) if a.prompt_path else None,
        # DO NOT DROP A DECLARED GRANT BECAUSE THE PATH IS NOT THERE YET: the CLI CREATES its own
        # session-transcript directory at startup, and a dropped grant is invisible. SBPL does not
        # require a path to exist.
        grants=[Path(g).resolve() for g in a.grant],
        writes=[Path(w).resolve() for w in a.write],
        hide=[Path(h).resolve() for h in a.hide],
        no_write=[Path(w).resolve() for w in a.no_write],
        cli_tmpdir=Path(a.cli_tmpdir) if a.cli_tmpdir else None,
        dry_run=a.dry_run,
        egress_ports=([a.egress_proxy_port] + list(a.egress_local_port)
                      if a.egress_proxy_port else None))
    if not _sandbox_exec_available():
        # NEVER emit an empty profile: `sandbox-exec -f <empty>` is not "no sandbox", it is a
        # confusing failure, and returning success here would let the launcher believe it is
        # contained. Refuse, and let the caller decide.
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
        print(f"preflight OK — {len(job.grants)} grant(s), {len(job.writes)} write target(s)"
              + (" (dry run: the per-uid CLI temp root was probed on a stand-in, never listed)"
                 if job.dry_run else ""))
    if a.emit_profile:
        out = Path(a.emit_profile)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(build_evolver_sandbox_profile(job))
        print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
