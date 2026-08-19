"""Direct API executor — multi-turn tool calling via httpx.

No SDK, no background server. Sends OpenAI-compatible chat completion
requests with function calling, executes tools locally, loops until the
model stops issuing tool calls (or hits max turns / timeout).

Dependencies: httpx, subprocess, pathlib, json, glob, re, os, asyncio.
"""

from __future__ import annotations

import asyncio
import contextvars
import fnmatch
from collections import Counter
import json
import logging
import hashlib
import os
import signal
import sys
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Type

import httpx
from opentelemetry import trace as otel_trace
from opentelemetry.trace import set_span_in_context
from pydantic import BaseModel, ValidationError

from ...skill_integrity import assert_text_clean
from ..errors import UngradableEpisode
from ..provider_auth import ensure_provider_api_key
from .._tracing import (
    LiveTranscriptEmitter,
    agent_root_span,
    estimate_cost,
    finalise_root_span,
    flush_spans,
    span_text,
    start_tool_span,
    start_turn_span,
)

_tracer = otel_trace.get_tracer("evoskill.harness.direct")

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Write guard — restricts file writes to allowed directories
# ---------------------------------------------------------------------------

_allowed_write_dirs: contextvars.ContextVar[list[Path]] = contextvars.ContextVar(
    "_allowed_write_dirs", default=[]
)
_bash_read_only: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_bash_read_only", default=False
)
_project_root: contextvars.ContextVar[str] = contextvars.ContextVar(
    "_project_root", default=""
)
_allowed_bash_write_dirs: contextvars.ContextVar[list[str]] = contextvars.ContextVar(
    "_allowed_bash_write_dirs", default=[]
)

_DANGEROUS_BASH_PATTERNS = [
    re.compile(r'\bmkfs\b'),
    re.compile(r'\bdd\s+.*of=/dev/'),
    re.compile(r'>\s*/dev/sd'),
    re.compile(r'\bchmod\s+-R\s+777\s+/'),
    re.compile(r'\bkill\s+-9\s+-1\b'),
    re.compile(r'\bsudo\b'),
]

# Patterns that modify the filesystem (write/delete/move).
# Used when bash_read_only=True to enforce read-only bash.
_BASH_WRITE_PATTERNS = [
    re.compile(r'\brm\s'),
    re.compile(r'\brmdir\s'),
    re.compile(r'\bmv\s'),
    re.compile(r'\bcp\s'),
    re.compile(r'\bmkdir\s'),
    re.compile(r'\btouch\s'),
    re.compile(r'\btee\s'),
    re.compile(r'\bchmod\s'),
    re.compile(r'\bchown\s'),
    re.compile(r'\bln\s'),
    re.compile(r'(?:^|[^|])>\s'),  # redirect > (but not | grep >)
    re.compile(r'>>\s'),          # append >>
    re.compile(r'\bcat\s.*>\s'),  # cat ... > file
]

def _check_write_guard(path: Path) -> str | None:
    """Return an error message if path is outside all allowed write dirs, or None if OK."""
    dirs = _allowed_write_dirs.get()
    if not dirs:
        return None
    resolved = path.resolve()
    for allowed in dirs:
        try:
            resolved.relative_to(allowed.resolve())
            return None
        except ValueError:
            continue
    allowed_list = ", ".join(str(d) for d in dirs)
    return f"Error: write blocked — {path} is outside allowed directories ({allowed_list})"


def _shell_quote(s: str) -> str:
    """Shell-quote a string for use inside bash -c '...'."""
    return "'" + s.replace("'", "'\\''") + "'"


_WRITE_VERBS = ('rm', 'rmdir', 'mv', 'cp', 'mkdir', 'touch', 'tee', 'chmod', 'chown', 'ln')


def _extract_write_targets(cmd: str) -> list[str]:
    """Best-effort extraction of the file/dir path arguments a bash write
    operation targets, so the write-guard can check the actual resolved path
    instead of scanning the whole command string for a directory-name
    substring — a check a command can trivially defeat by mentioning the
    allowed dir's name anywhere (a comment, an echoed string) while writing
    somewhere else entirely."""
    targets = []
    for m in re.finditer(r'>>?\s*([\'"]?)([^\s\'"|;&<>]+)\1', cmd):
        targets.append(m.group(2))
    for verb in _WRITE_VERBS:
        for m in re.finditer(rf'\b{verb}\s+([^\n;|&]*)', cmd):
            for tok in re.split(r'\s+', m.group(1).strip()):
                tok = re.split(r'[;|&]', tok)[0]
                if tok and not tok.startswith('-'):
                    targets.append(tok.strip('\'"'))
    return targets


def _resolve_write_target(target: str, project_root: str) -> Path | None:
    """Resolve a write-target path argument to an absolute host path.

    Bash commands run inside a container where /workspace is bind-mounted
    from project_root (the host cwd) — a relative path or a /workspace/-
    prefixed absolute path both map onto project_root on the host. Any other
    absolute path (e.g. /etc/passwd) is left as-is, which correctly fails the
    allowed-dir containment check since it won't resolve under any of them.
    """
    try:
        if target.startswith('/workspace/'):
            p = Path(project_root) / target[len('/workspace/'):]
        elif target == '/workspace':
            p = Path(project_root)
        elif target.startswith('/'):
            p = Path(target)
        else:
            p = Path(project_root) / target
        return p.resolve()
    except (ValueError, OSError):
        return None


# Matches `<<[-]['"]?DELIM['"]?[trailing] ... \nDELIM` heredoc blocks so their
# bodies can be excluded from bash-syntax write-pattern scanning. Embedded
# scripts (python3 << 'EOF' ... EOF) routinely contain tokens like "> 20" (a
# numeric comparison) or "mv"/"cp"/"ln" as substrings of identifiers/comments —
# none of which are bash writes, but the naive whole-string scan below can't
# tell. The trailing-content group captures anything between the delimiter and
# the newline (e.g. a real outer redirect like `<<EOF > file.txt` or a pipe
# like `<<EOF | grep`) so it's preserved in the replacement, not discarded —
# that content is genuine outer-command syntax, not heredoc body.
_HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)(\w+)\1([^\n]*)\n(.*?)\n\2(?=\s|$)", re.DOTALL)


def _strip_heredocs(cmd: str) -> str:
    """Blank out heredoc bodies, keeping markers and any trailing outer-command
    syntax on the opening line, so write-pattern regexes only see the outer
    bash command — not embedded interpreter source."""
    return _HEREDOC_RE.sub(
        lambda m: f"<<{m.group(1)}{m.group(2)}{m.group(1)}{m.group(3)}\nHEREDOC_BODY_STRIPPED\n{m.group(2)}",
        cmd,
    )


# `python3 -c "..."` / `node -e '...'` bodies are interpreter SOURCE, not shell.
# Same treatment as heredocs, for the same reason — and it is not optional:
# a real evolver turn was blocked outright because a COMMENT inside
# `python3 -c` read `TIMEVALUE(RIGHT(serial,8)) -> #VALUE!`. The `->` matched the
# redirect pattern, `#VALUE!` was extracted as the "write target", it resolved to
# nothing, and the fail-closed branch rejected a pure `print()` computation —
# killing exactly the independent derivation Stage 1 of the protocol demands.
# Note this opens no new hole: writes inside a heredoc body were already
# unchecked, so shell-level redirection has always been the guard's real scope.
_INLINE_SCRIPT_RE = re.compile(
    r"""(?P<interp>\b(?:python3?|perl|ruby|node|bash|sh|zsh)\s+
         (?:-\w+\s+)*                 # other flags (e.g. -u, -E)
         -(?:c|e)\s+)                 # the inline-source flag
        (?P<q>['"])                      # opening quote
        (?P<body>(?:\\.|(?!(?P=q)).)*)   # body, honouring backslash escapes
        (?P=q)""",
    re.VERBOSE | re.DOTALL,
)


def _strip_inline_scripts(cmd: str) -> str:
    """Blank out `-c`/`-e` inline interpreter source, keeping the shell around it."""
    return _INLINE_SCRIPT_RE.sub(
        lambda m: f"{m.group('interp')}{m.group('q')}INLINE_SCRIPT_STRIPPED{m.group('q')}",
        cmd,
    )


def _check_bash_safety(cmd: str, read_only: bool = False) -> str | None:
    """Return an error message if the bash command is unsafe, or None if OK.

    When read_only=True, blocks all filesystem-modifying commands.
    When read_only=False, only blocks catastrophically dangerous patterns.
    In both modes, allows read operations (curl, grep, cat, head, etc.).
    """
    for pattern in _DANGEROUS_BASH_PATTERNS:
        if pattern.search(cmd):
            return f"Error: blocked dangerous command: {cmd[:200]}"

    # Closed-corpus mode: block fetching any external URL. The only network
    # the solver may touch is the corpus search server ($SEARCH_URL — local
    # host or the docker host gateway). Catches domains AND numeric IPs, so
    # it also covers what the docker --dns blackhole can't (direct-IP fetch)
    # and the local (non-docker) bash path where DNS works normally.
    if os.environ.get("EVOSKILL_CORPUS_ONLY") == "1":
        _allowed_hosts = {
            "localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal",
            os.environ.get("SEARCH_HOST", ""),
        }

        def _is_blocked_host(host: str) -> bool:
            host = host.lower()
            is_private_ip = bool(re.match(
                r'(?:10\.|192\.168\.|172\.(?:1[6-9]|2\d|3[01])\.)', host))
            return host not in _allowed_hosts and not is_private_ip

        for m in re.finditer(r'https?://([^/\s"\'<>:]+)(?::\d+)?', cmd):
            host = m.group(1)
            if _is_blocked_host(host):
                # State the boundary fact only. Do NOT suggest the search
                # endpoint here — coaching search strategy from the guardrail
                # would confound the experiment (that's the evolver's job).
                return (
                    f"Error: external network access is blocked "
                    f"(closed-corpus benchmark). Blocked host: {host.lower()}"
                )
        # Scheme-less invocations (`curl example.com`, `wget example.com/x`)
        # bypass the scheme-based scan above entirely (bug M27) — this only
        # matters on EVOSKILL_SANDBOX=sandbox/none, which have no iptables
        # backstop; on docker mode the kernel-level firewall is load-bearing
        # regardless of this check.
        for m in re.finditer(r'\b(?:curl|wget|nc|ncat|telnet)\s+(?:-\S+\s+)*([a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}|\d{1,3}(?:\.\d{1,3}){3})(?::\d+)?(?:[/\s]|$)', cmd):
            host = m.group(1)
            if _is_blocked_host(host):
                return (
                    f"Error: external network access is blocked "
                    f"(closed-corpus benchmark). Blocked host: {host.lower()}"
                )

    # Block recursive grep — never terminates on broad dirs, eats 100% CPU.
    # Only allow grep -r/-R if it ONLY targets /workspace (not / or ..)
    if re.search(r'grep\b.*\s+-[a-zA-Z]*[rR]', cmd) or re.search(r'grep\b.*-[rR]', cmd):
        # Block if / or .. appears as a standalone path argument (not inside /workspace)
        has_root = bool(re.search(r'\s/\s|\s/\b(?!workspace)', cmd))
        has_parent = '..' in cmd
        only_workspace = '/workspace' in cmd and not has_root and not has_parent
        if not only_workspace:
            return "Error: recursive grep over the filesystem is blocked (it does not terminate on broad paths). Restrict the search to /workspace."

    # Write-pattern regexes assume bash syntax (rm, mv, >, etc.) — scan only
    # the outer command, not the body of any `python3 << 'EOF' ... EOF`
    # heredoc, where the same tokens routinely appear as ordinary code
    # (e.g. `if x > 20:`) with no bearing on the filesystem.
    cmd_no_heredoc = _strip_inline_scripts(_strip_heredocs(cmd))

    if read_only:
        for pattern in _BASH_WRITE_PATTERNS:
            if pattern.search(cmd_no_heredoc):
                return (
                    f"Error: write operation blocked (read-only mode). "
                    f"This agent cannot modify files. Command: {cmd[:200]}"
                )

    bash_dirs = _allowed_bash_write_dirs.get()
    proj_root = _project_root.get()
    if bash_dirs and not read_only and not _docker_runtime_switched.get():
        for pattern in _BASH_WRITE_PATTERNS:
            if pattern.search(cmd_no_heredoc):
                allowed_resolved = []
                for d in bash_dirs:
                    try:
                        allowed_resolved.append(Path(d).resolve())
                    except OSError:
                        continue
                targets = _extract_write_targets(cmd_no_heredoc)

                def _target_allowed(t: str) -> bool:
                    resolved = _resolve_write_target(t, proj_root or ".")
                    if resolved is None:
                        return False
                    # /dev/null is always harmless to write to (discards
                    # everything, never persists state) — without this, a
                    # completely unrelated `2>/dev/null` anywhere in a
                    # multi-line command (an extremely common idiom for
                    # suppressing stderr noise) gets misparsed by the ">"
                    # redirect regex in _extract_write_targets as a write
                    # target, and since a REAL write elsewhere in the same
                    # command is only allowed if ALL extracted targets pass,
                    # one stray "2>/dev/null" silently blocks an otherwise
                    # legitimate write to the agent's own allowed directory
                    # (found via a real evolver trace: repeated failed
                    # `rm -rf .claude/skills/X` attempts, each paired with a
                    # trailing `2>/dev/null` on an unrelated line).
                    if str(resolved) == "/dev/null":
                        return True
                    for allowed in allowed_resolved:
                        try:
                            resolved.relative_to(allowed)
                            return True
                        except ValueError:
                            continue
                    return False

                # Fail closed: if we couldn't confidently extract any target
                # path from a command matching a write pattern, don't allow it
                # through on the absence of evidence — require at least one
                # resolved target, and every extracted target must be allowed.
                if targets and all(_target_allowed(t) for t in targets):
                    return None
                return (
                    f"Error: write blocked — target not in allowed directories "
                    f"({', '.join(bash_dirs)}). Command: {cmd[:200]}"
                )

    return None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_TURNS = 200
# 60s was too short for reasoning models with long silent "thinking" phases
# before the first response byte (confirmed live on ARC-AGI-2/DSV4F: a
# non-streaming call can sit with zero bytes received for multiple minutes
# on a hard task, hitting this read-timeout and burning a retry+backoff
# cycle repeatedly rather than ever seeing the real response). Raised to
# 300s project-wide (shared by every direct-harness benchmark) rather than
# patched per-benchmark, since the same failure mode applies to any
# reasoning-heavy model on the non-streaming path (EVOSKILL_STREAM=0).
HTTP_TIMEOUT = int(os.environ.get("EVOSKILL_HTTP_TIMEOUT", "300"))  # seconds per API call
STREAM_ENABLED = os.environ.get("EVOSKILL_STREAM", "0") == "1"

# Per-process count of responses by serving provider.
# Populated in the turn loop; lets logs prove "N calls, all deepseek" and
# makes a silent reroute off a pinned provider loudly visible.
_PROVIDER_AUDIT: dict[str, int] = {}
STREAM_SILENCE_TIMEOUT = int(os.environ.get("EVOSKILL_STREAM_SILENCE_TIMEOUT", "45"))
MAX_RETRIES_PER_CALL = 5
RETRY_BACKOFF_BASE = 2.0  # seconds; doubles each retry
MAX_RETRY_BACKOFF = 30.0  # cap exponential backoff
CIRCUIT_BREAKER_ACQUIRE_TIMEOUT = float(os.environ.get("EVOSKILL_CIRCUIT_BREAKER_TIMEOUT", "180"))


# Liveness heartbeats (src/harness/liveness.py). Imported lazily inside a tiny
# wrapper so this module keeps working if liveness is absent, and so a
# heartbeat can never raise into the hot path.
#
# WHY every candidate path is instrumented rather than just the known one:
# span forensics eliminated EVERY instrumented explanation for
# 690-1452s of dead air inside hung rollouts — the SSE silence watchdog never
# fired (0 occurrences), admission starvation never fired (0), there were no
# unclosed child spans, no ReadTimeout at all, and LLM retries never advanced
# past attempt 2. So the time is burned in code that emits neither spans nor
# logs, and could not be named from the available evidence. Labelling each
# heartbeat with a `kind` makes the NEXT occurrence self-identifying: the
# watchdog's kill log reports last_kind, which names the culprit directly
# instead of requiring another forensic session.
def _liveness_touch(kind: str) -> None:
    try:
        from src.harness.liveness import touch as _t
        _t(kind)
    except Exception:
        pass


async def _await_beating(awaitable, kind: str, interval: float = 30.0):
    """Thin indirection onto liveness.await_beating.

    Kept local so the import stays lazy and optional, matching
    _liveness_touch: the harness must run even if the liveness module is
    unavailable, in which case beating degrades to a plain await.
    """
    try:
        from src.harness.liveness import await_beating
    except Exception:
        return await awaitable
    return await await_beating(awaitable, kind, interval)


# Totals from the most recent execute_query. See the assignment site for the
# concurrency caveat.
LAST_RUN_STATS: dict[str, Any] = {}


def _log_http_error_body(resp: Any) -> None:
    """Log the response BODY before raise_for_status() discards it.

    httpx's HTTPStatusError message carries the status line and a docs URL but
    not the body, and for a 4xx the body is the entire diagnosis: OpenRouter
    replies e.g. {"error":{"message":"reasoning.effort: Invalid option: expected
    one of ..."}}. Without this a 400 is a dead end -- the caller sees "Client
    error '400 Bad Request'" and has to reconstruct the payload by hand to find
    out which field the upstream objected to.
    """
    if resp.status_code < 400:
        return
    try:
        body = resp.text
    except Exception:                       # body already consumed/streamed
        return
    if body:
        logger.error("API %s body: %s", resp.status_code, body[:2000])


async def _log_http_error_body_stream(resp: Any) -> None:
    """Same, for a streaming response whose body has not been read yet."""
    if resp.status_code < 400:
        return
    try:
        raw = await resp.aread()
    except Exception:
        return
    if raw:
        logger.error("API %s body (stream): %s", resp.status_code,
                     raw.decode(errors="replace")[:2000])


# ---------------------------------------------------------------------------
# Adaptive concurrency limiter — AIMD-based throttling
# ---------------------------------------------------------------------------

class _AdaptiveConcurrencyLimiter:
    """AIMD (Additive Increase, Multiplicative Decrease) concurrency control.

    Replaces binary circuit breaker with smooth adaptive throttling:
      - On success: increase concurrency cap by 1 (additive)
      - On failure: halve concurrency cap (multiplicative)
      - In-flight requests above cap finish but no new ones start

    Same principle as TCP congestion control. Eliminates thundering herd
    by ramping up gradually after backoff instead of all-at-once retry.
    """
    def __init__(self, max_concurrent: int = 40, min_concurrent: int = 2):
        self._max = max_concurrent
        self._min = min_concurrent
        self._cap = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._in_flight = 0
        self._lock = asyncio.Lock()
        self._last_decrease = 0.0
        # Fleet-wide budget (2026-07-30, #109 first live wiring): when set,
        # the process's fair share CLAMPS the admission gate; AIMD state and
        # recovery dynamics are untouched. None = feature off (default).
        self._fleet = None

    async def before_call(self, timeout: float = CIRCUIT_BREAKER_ACQUIRE_TIMEOUT):
        """Wait until concurrency is below cap.

        Bounded by `timeout` — unbounded here meant a cap collapsed by a
        failure cascade elsewhere in the swarm could starve a task for the
        entire outer wall-clock budget with zero spans/events logged (found
        via a trace that hung ~29min with turns 1-7 done in <1min, then
        total silence until the outer 1800s timeout force-closed it).
        """
        deadline = time.monotonic() + timeout
        while True:
            # Fleet share is read OUTSIDE the lock (it does memoized flock'd
            # file I/O, at most once per refresh_s) and must never break the
            # call path — on any error the gate falls back to AIMD-only.
            fleet_cap = None
            if self._fleet is not None:
                try:
                    fleet_cap = self._fleet.share()
                except Exception:
                    fleet_cap = None
            async with self._lock:
                eff_cap = self._cap if fleet_cap is None else min(self._cap, fleet_cap)
                if self._in_flight < eff_cap:
                    self._in_flight += 1
                    return
                wait = 1.0 + (id(asyncio.current_task()) % 50) * 0.02
            if time.monotonic() + wait > deadline:
                raise asyncio.TimeoutError(
                    f"Concurrency limiter starved for {timeout}s "
                    f"(cap={self._cap}, in_flight={self._in_flight})"
                )
            await asyncio.sleep(wait)

    async def after_call(self):
        """Release concurrency slot."""
        async with self._lock:
            self._in_flight = max(0, self._in_flight - 1)

    async def record_success(self):
        async with self._lock:
            if self._cap < self._max:
                self._cap = min(self._cap + 1, self._max)

    async def record_failure(self):
        async with self._lock:
            now = time.monotonic()
            if now - self._last_decrease < 2.0:
                return
            old_cap = self._cap
            self._cap = max(self._cap // 2, self._min)
            self._last_decrease = now
            if self._cap < old_cap:
                logger.warning(
                    f"Concurrency reduced: {old_cap} → {self._cap} "
                    f"(in-flight: {self._in_flight})"
                )


_circuit_breaker = _AdaptiveConcurrencyLimiter()

# Attach the fleet-wide budget when enabled (EVOSKILL_FLEET_LLM_TOTAL > 0).
# from_env returns None when off, so default behavior is byte-identical.
try:
    from src.harness.fleet import FleetBudget as _FleetBudget
    _circuit_breaker._fleet = _FleetBudget.from_env(
        member=os.environ.get("EVOSKILL_FLEET_MEMBER", f"pid-{os.getpid()}"))
    if _circuit_breaker._fleet is not None:
        logger.warning(
            f"[FLEET] dynamic LLM budget ON: total={_circuit_breaker._fleet.total} "
            f"member={_circuit_breaker._fleet.member} "
            f"min_share={_circuit_breaker._fleet.min_share}")
except Exception as _fleet_err:  # fleet must never prevent a run from starting
    logger.warning(f"[FLEET] disabled (init error: {_fleet_err})")
    _circuit_breaker._fleet = None


# ---------------------------------------------------------------------------
# Shared httpx client — bounded connection pool
# ---------------------------------------------------------------------------

_shared_client: httpx.AsyncClient | None = None
_shared_client_lock = asyncio.Lock()


async def _get_shared_client() -> httpx.AsyncClient:
    """Lazily create a shared httpx client with bounded connections."""
    global _shared_client
    if _shared_client is not None and not _shared_client.is_closed:
        return _shared_client
    async with _shared_client_lock:
        if _shared_client is None or _shared_client.is_closed:
            _shared_client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=500,
                    max_keepalive_connections=300,
                ),
            )
    return _shared_client

_shared_stream_client: httpx.AsyncClient | None = None
_shared_stream_lock = asyncio.Lock()


async def _get_shared_stream_client() -> httpx.AsyncClient:
    """Shared httpx client for streaming with longer timeouts."""
    global _shared_stream_client
    if _shared_stream_client is not None and not _shared_stream_client.is_closed:
        return _shared_stream_client
    async with _shared_stream_lock:
        if _shared_stream_client is None or _shared_stream_client.is_closed:
            _shared_stream_client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=30.0, read=float(HTTP_TIMEOUT), write=30.0, pool=30.0,
                ),
                limits=httpx.Limits(
                    max_connections=500,
                    max_keepalive_connections=300,
                    # OpenRouter closes idle conns server-side; without an expiry
                    # the pool hands out dead sockets → spurious ConnectError +
                    # retry. But too-short expiry (20s) churns reconnects and the
                    # per-connection DNS lookups burst-fail under concurrency
                    # (macOS resolver pressure). 55s stays under typical 60s LB
                    # idle timeouts while cutting DNS churn ~3x.
                    keepalive_expiry=55.0,
                ),
            )
    return _shared_stream_client

# Seconds per bash command. Overridable because legitimate batch work can
# exceed 120s — e.g. an evolver diffing 68 solver outputs against goldens
# with openpyxl (~2 workbook loads/task) needs 3-6 min in one call (Jul 1).
BASH_TIMEOUT = int(os.environ.get("EVOSKILL_BASH_TIMEOUT", "120"))
MAX_TOOL_RESULT_CHARS = 100_000  # truncate huge tool outputs
# Sentinel for handlers that must NOT pre-truncate: _execute_tool applies the
# agent's own max_tool_output_chars afterwards. Pre-truncating at the module
# default made any cap ABOVE 100,000 silently a no-op.
NO_TRUNCATE = 1 << 40

# Mirrors scripts/run_tb2_evo.py's BASH_DEFAULT_TIMEOUT_SEC / BASH_HARD_CAP_SEC
# (same env var names, same literal defaults: 240s / 900s) -- applied to the
# evolver's OWN bash tool ONLY while it is switched into a TB2.1 task's
# runtime via switch_runtime (see _exec_switch_runtime / _exec_bash below),
# so its hands-on debugging experiences the same per-call budget TB2.1's own
# solver gets instead of this module's generic 120s default.
#
# These are a deliberate MIRROR, not an import of the real constants: the
# real ones live in scripts/run_tb2_evo.py, a launch script with import-time
# side effects (reads a hardcoded absolute-path secrets file, mutates
# os.environ, spins up a brand-new Phoenix tracing project) that must never
# fire just from importing this low-level, widely-shared executor module.
# Reading the SAME env var names means an operator overriding TB2.1's solver
# timeout via EVOSKILL_TB2_BASH_TIMEOUT/_HARD_CAP transparently overrides
# this mirror too, so the two stay in sync in the one way that matters
# operationally even though they're two physically separate constants.
TB2_SWITCHED_BASH_DEFAULT_SEC = int(os.environ.get("EVOSKILL_TB2_BASH_TIMEOUT", "240"))
TB2_SWITCHED_BASH_HARD_CAP_SEC = int(os.environ.get("EVOSKILL_TB2_BASH_HARD_CAP", "900"))

# Same rationale, generic path: switch_runtime called with an explicit
# docker_image (no task.toml to read benchmark-specific timeouts from —
# e.g. SWE-rebench, which has no local task_dir at all, only a docker_image
# string per task). Same literal defaults as the TB2 mirror above since
# "give hands-on debugging plenty of room to actually run tests" is the
# same need regardless of which benchmark's image is switched into.
GENERIC_SWITCHED_BASH_DEFAULT_SEC = int(os.environ.get("EVOSKILL_SWITCHED_BASH_TIMEOUT", "240"))
GENERIC_SWITCHED_BASH_HARD_CAP_SEC = int(os.environ.get("EVOSKILL_SWITCHED_BASH_HARD_CAP", "900"))
# Defaults for a docker_image-only switch (no task.toml [environment] section
# to read cpus/memory_mb from) -- matches SWE-rebench's own container spec
# (scripts/run_swe_rebench_evo.py's SWE_REBENCH_CPUS/MEMORY_MB), the first
# real consumer of this path; override per-call via the tool's own cpus/
# memory_mb arguments if a future benchmark needs something different.
GENERIC_SWITCHED_CPUS = int(os.environ.get("EVOSKILL_SWITCHED_CPUS", "4"))
GENERIC_SWITCHED_MEMORY_MB = int(os.environ.get("EVOSKILL_SWITCHED_MEMORY_MB", "8192"))

# Which backend switch_runtime's docker_image path launches on. "docker"
# (default) keeps the original local-container behavior, used by TB2.1-style
# benchmarks whose own solver also runs locally. "daytona" launches a cloud
# sandbox via the same primitive the SWE-rebench solver itself already uses
# (src/harness/daytona_sandbox.py), matching that benchmark's own execution
# backend and avoiding local Colima disk/memory pressure from pulling
# multi-GB images. Set per-launcher (e.g. run_swe_rebench_evo.py sets this to
# "daytona" at import time) rather than globally, since which backend is
# "correct" is a per-benchmark decision, not a repo-wide one -- TB2.1/ARC/tau2
# should keep the local-docker default.
_SWITCH_RUNTIME_BACKEND = os.environ.get("EVOSKILL_SWITCH_RUNTIME_BACKEND", "docker")

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

# Names of containers THIS process created (via _create_docker_container),
# populated as they're made. _cleanup_all_docker_containers below used to
# filter host-wide by `ancestor=$EVOSKILL_DOCKER_IMAGE` instead of using this
# set -- confirmed as an active bug: since dozens of scripts share the same
# image name (e.g. "ssb-solver"), any short-lived script exiting (success,
# error, or Ctrl-C) would force-remove EVERY container on the host matching
# that image, including a concurrently-running sibling process's entire
# in-flight batch. Tracking our own containers by name makes cleanup scoped
# to this process only.
_own_containers: set[str] = set()


def _cleanup_all_docker_containers():
    """Kill only the sandbox containers THIS process created, on exit.

    Batched: `docker rm -f` accepts many names per call, and one subprocess per
    container made shutdown scale linearly in daemon round-trips — unacceptable
    now that a failed removal keeps its container tracked, since the set at exit
    can hold hundreds after a contended run.
    """
    import subprocess as _sp
    names = list(_own_containers)
    for i in range(0, len(names), 50):
        batch = names[i:i + 50]
        try:
            _sp.run(["docker", "rm", "-f", *batch], capture_output=True, timeout=120)
        except Exception:
            pass


import atexit
atexit.register(_cleanup_all_docker_containers)


def _install_sigterm_cleanup() -> None:
    """Run container cleanup on SIGTERM/SIGHUP, not just on normal exit.

    atexit does NOT fire on SIGTERM — the default disposition terminates the
    process outright — so every `kill <pid>` of a runner leaked its entire
    in-flight container set. Measured live: a clean 20s SIGTERM exit of
    one arm dropped the host container count by only 20 of its ~500, and two
    killed judge legs left 545 behind, which is what pushed a 13-CPU host to
    1023 containers and 8.8GB of 13.9GB VM memory.

    Chains to any previously installed handler so this does not silently swallow
    another component's shutdown logic, and re-raises the default behaviour by
    restoring the handler and re-sending the signal to self.
    """
    import signal

    def _handler(signum, frame):
        try:
            _cleanup_all_docker_containers()
        except Exception:
            pass
        prev = _prev_handlers.get(signum)
        if callable(prev):
            prev(signum, frame)
            return
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    _prev_handlers: dict[int, Any] = {}
    for _sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            _prev_handlers[_sig] = signal.getsignal(_sig)
            signal.signal(_sig, _handler)
        except (ValueError, OSError):
            # Not the main thread, or the platform disallows it — atexit still
            # covers the normal-exit path.
            pass


_install_sigterm_cleanup()


# ---------------------------------------------------------------------------
# Skill discovery
# ---------------------------------------------------------------------------

def _discover_skills(skills_dir: str | Path) -> dict[str, dict[str, str]]:
    """Scan skills_dir for SKILL.md files and parse frontmatter.

    Returns {name: {"path": ..., "description": ..., "name": ...}}.
    """
    skills_dir = Path(skills_dir)
    found: dict[str, dict[str, str]] = {}
    if not skills_dir.is_dir():
        return found
    for skill_md in sorted(skills_dir.glob("*/SKILL.md")):
        text = skill_md.read_text(errors="replace")
        # A skill carrying merge-conflict markers was never authored by the
        # evolver; loading it silently corrupts the rollout (0726: 28 rollouts
        # read a skill that was 58% merge residue). The registry gates this at
        # snapshot/restore time — this is the backstop for any path that writes
        # skills without going through the registry.
        assert_text_clean(text, origin=str(skill_md))
        name = skill_md.parent.name
        description = ""
        if text.startswith("---"):
            end = text.find("---", 3)
            if end > 0:
                fm = text[3:end]
                for line in fm.splitlines():
                    line = line.strip()
                    if line.startswith("name:"):
                        name = line[5:].strip().strip("'\"")
                    elif line.startswith("description:"):
                        description = line[12:].strip().strip("'\"")
        found[name] = {
            "name": name,
            "path": str(skill_md),
            "description": description or f"Skill from {skill_md.parent.name}",
        }
    return found


def _build_skill_index(skills: dict[str, dict[str, str]]) -> str:
    """Build a system prompt section listing available skills."""
    lines = ["\n\n## Skills"]
    lines.append("")
    lines.append("Skills are detailed instruction sets. Only their names and trigger conditions are shown here to save context space. When a trigger condition matches your task, call the `skill` tool with `{\"name\": \"<skill-name>\"}` to load the full instructions. See available skills below.")
    lines.append("")
    if skills:
        for info in skills.values():
            lines.append(f"- {info['name']}: {info['description']}")
    else:
        lines.append("(No skills available yet.)")
    return "\n".join(lines)


# Per-agent skill state via ContextVar (concurrent agents see their own skills)
_active_skills: contextvars.ContextVar[dict[str, dict[str, str]]] = contextvars.ContextVar(
    "_active_skills", default={}
)

TOOL_DEFS: dict[str, dict[str, Any]] = {
    "bash": {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Run a bash command and return stdout+stderr. The working "
                "directory persists between calls (a `cd` in one call carries "
                "into the next). Shell variables and functions do NOT persist "
                f"— each call runs in a fresh shell process. Default timeout "
                f"{BASH_TIMEOUT}s. If you've switched into a task's own runtime "
                "via switch_runtime, that benchmark's own bash timeout/hard-cap "
                "applies instead (see switch_runtime's tool result for the "
                "exact numbers) and timeout_sec can raise it up to that cap; "
                "outside a switched runtime, timeout_sec has no effect."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute.",
                    },
                    "timeout_sec": {
                        "type": "integer",
                        "description": (
                            "Optional per-call timeout override in seconds. "
                            "Only takes effect while switched into a task's "
                            "runtime via switch_runtime (clamped to that "
                            "benchmark's hard cap); ignored otherwise."
                        ),
                    },
                },
                "required": ["command"],
            },
        },
    },
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative file path."},
                    "offset": {"type": "integer", "description": "Line number to start from (0-based). Optional."},
                    "limit": {"type": "integer", "description": "Max lines to read. Optional."},
                },
                "required": ["path"],
            },
        },
    },
    "write_file": {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file (creates or overwrites).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative file path."},
                    "content": {"type": "string", "description": "The full file content to write."},
                },
                "required": ["path", "content"],
            },
        },
    },
    "edit_file": {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact string in a file with a new string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path."},
                    "old_string": {"type": "string", "description": "Exact text to find."},
                    "new_string": {"type": "string", "description": "Replacement text."},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    "glob": {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern. Returns newline-separated paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern (e.g. '**/*.py')."},
                },
                "required": ["pattern"],
            },
        },
    },
    "grep": {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search files for a regex pattern. Returns matching lines with file:line: prefix.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for."},
                    "path": {"type": "string", "description": "Directory or file to search in. Defaults to cwd."},
                    "include": {"type": "string", "description": "Glob for filenames to include (e.g. '*.py'). Optional."},
                },
                "required": ["pattern"],
            },
        },
    },
    "skill": {
        "type": "function",
        "function": {
            "name": "skill",
            "description": "Load a skill's full instructions by name. See the Available Skills section in the system prompt for the list of skill names.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The skill name to load (from the Available Skills list).",
                    },
                },
                "required": ["name"],
            },
        },
    },
    "switch_runtime": {
        "type": "function",
        "function": {
            "name": "switch_runtime",
            "description": (
                "Launch a FRESH instance of a specific task's own runtime and "
                "redirect every subsequent bash call to it, so you can debug a "
                "past failure by actually running commands inside that task's "
                "environment instead of only reading static trace/diff text. "
                "This is a NEW instance, not a resume of the exact failed "
                "attempt -- the original container (and anything the solver "
                "itself wrote inside it) is already gone by the time you run; "
                "if you need the solver's actual final output, read it as text "
                "from the trace first, then re-apply/re-create it yourself "
                "once switched in. Pass EXACTLY ONE of task_dir or docker_image "
                "depending on what this benchmark's traces expose: "
                "task_dir (a directory containing task.toml, e.g. TB2.1) -- "
                "its own [environment] section supplies the docker_image, cpus "
                "and memory_mb, and if the task has a tests/ directory its "
                "test files are copied in under /tests so you can run "
                "/tests/test.sh yourself. "
                "docker_image (a plain image string/tag, e.g. a SWE-rebench "
                "task's own docker_image field shown in its trace) -- no "
                "task.toml is read, so optionally pass cpus/memory_mb too if "
                "this task needs more than the generic default; there is no "
                "tests/ copy-in on this path since there's no local task_dir "
                "to copy from -- reproduce the verifier yourself once inside "
                "(e.g. the actual test command from the trace). For some "
                "benchmarks this launches a cloud sandbox (matching that "
                "benchmark's own solver backend) rather than a local "
                "container -- functionally identical from your side, just "
                "note whether the tool result says 'Sandbox' or 'Container'. "
                "Once switched, the bash tool's timeout also changes to match "
                "this task's own solver (longer default and hard-cap than the "
                "standard bash timeout) -- see the tool result after calling "
                "this for the exact numbers. Call exit_runtime when done to "
                "return to your own environment; only one runtime can be "
                "switched into at a time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_dir": {
                        "type": "string",
                        "description": "Absolute path to the task's directory (containing task.toml) -- this is the same task_dir string that appears in the failure trace's question JSON. Mutually exclusive with docker_image.",
                    },
                    "docker_image": {
                        "type": "string",
                        "description": "A docker image string/tag to launch directly (no task.toml involved) -- e.g. the docker_image value shown for a task in this benchmark's trace index. Mutually exclusive with task_dir.",
                    },
                    "cpus": {
                        "type": "number",
                        "description": "Only used with docker_image (task_dir reads this from task.toml instead). Defaults to a generic value if omitted.",
                    },
                    "memory_mb": {
                        "type": "integer",
                        "description": "Only used with docker_image (task_dir reads this from task.toml instead). Defaults to a generic value if omitted.",
                    },
                },
                "required": [],
            },
        },
    },
    "exit_runtime": {
        "type": "function",
        "function": {
            "name": "exit_runtime",
            "description": "Tear down the task runtime opened by switch_runtime and return to your own environment. Safe to call even if nothing is currently switched (returns a no-op message).",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
}


def _get_enabled_tools(tool_flags: dict[str, bool]) -> list[dict[str, Any]]:
    """Return OpenAI tool definitions for enabled tools."""
    return [TOOL_DEFS[name] for name, on in tool_flags.items() if on and name in TOOL_DEFS]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def _resolve_path(path: str, cwd: str) -> Path:
    """Resolve a path relative to the working directory."""
    p = Path(path)
    if not p.is_absolute():
        p = Path(cwd) / p
    return p.resolve()


def _format_tool_call(tc: dict[str, Any]) -> str:
    fn = tc.get("function", {})
    name = fn.get("name", "?")
    args_raw = fn.get("arguments", "{}")
    try:
        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        if name == "bash" and "command" in args:
            return f"**{name}**\n```\n{args['command']}\n```"
        else:
            return f"**{name}**\n```json\n{json.dumps(args, indent=2)}\n```"
    except (json.JSONDecodeError, TypeError):
        return f"**{name}**\n```\n{args_raw}\n```"


def _emit_messages_on_span(span, prefix: str, messages: list[dict[str, Any]]) -> int:
    """Set llm.input_messages or llm.output_messages on a span.

    Logs the EXACT message content — no additional truncation.
    The messages array already contains the real content (including any
    _truncate markers from tool execution). What you see in Phoenix is
    exactly what the LLM received.

    Tool calls are rendered as separate 'tool_call' blocks for readability,
    but the content is faithful to the original arguments.
    """
    idx = 0
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "") or ""
        tool_calls = m.get("tool_calls")
        # Reasoning/thinking tokens (DeepSeek 'reasoning_content', OpenRouter 'reasoning').
        # With DeepSeek V4 reasoning passback, these fields are NOW preserved on
        # clean_msg (required for tool-calling turns), so they appear on BOTH
        # input_messages AND output_messages. This is honest: the reasoning IS
        # genuinely in the context we send to the LLM on subsequent turns.
        reasoning = m.get("reasoning_content") or m.get("reasoning") or ""

        if reasoning:
            span.set_attribute(f"{prefix}.{idx}.message.role", "reasoning")
            span.set_attribute(f"{prefix}.{idx}.message.content", reasoning)
            idx += 1

        if content:
            span.set_attribute(f"{prefix}.{idx}.message.role", role)
            span.set_attribute(f"{prefix}.{idx}.message.content", content)
            idx += 1

        if tool_calls:
            for tc in tool_calls:
                span.set_attribute(f"{prefix}.{idx}.message.role", "tool_call")
                span.set_attribute(f"{prefix}.{idx}.message.content", _format_tool_call(tc))
                idx += 1
        elif not content and not tool_calls and role:
            span.set_attribute(f"{prefix}.{idx}.message.role", role)
            span.set_attribute(f"{prefix}.{idx}.message.content", "(empty)")
            idx += 1

    return idx


# Where overflowed tool output is spilled. Inside the workspace so the sandboxed
# bash (cwd=/workspace) can read it back, and under .cache/ which the workspace
# .gitignore already excludes, so spills never enter a candidate commit.
SPILL_SUBDIR = ".cache/tool_output"
# Overflow bookkeeping, keyed by content hash, so a repeat of the SAME output is
# recognised rather than re-spilled. Reset per run by _reset_spill_state().
_spill_seen: dict[str, int] = {}


def _reset_spill_state() -> None:
    _spill_seen.clear()


def _spill_path(text: str, cwd: str | None) -> tuple[Path | None, str]:
    """Write `text` to a content-addressed file; return (path, sha12).

    Content-addressed on purpose. Claude Code's spill is path-random: cat'ing a
    spill file spills it AGAIN to a fresh path (verified -- two byte-identical
    621KB artifacts, sha 82ea103f, at different paths), so a model that re-reads
    the artifact loops forever while disk grows without bound. Hashing makes the
    write idempotent and gives _truncate a key to count repeats on.
    """
    sha = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]
    if not cwd:
        return None, sha
    try:
        d = Path(cwd) / SPILL_SUBDIR
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{sha}.txt"
        if not p.exists():                      # idempotent: never rewrite
            p.write_text(text, errors="replace")
        return p, sha
    except Exception:
        return None, sha                        # spilling is best-effort, never fatal


_SINGLE_FILE_CMD = re.compile(
    r"^\s*(?:cat|bat|nl|less|more)\s+(?:-\w+\s+)*(?P<f>[^\s|;&><]+)\s*$"
)


def _aliased_source(text: str, cwd: str | None, cmd: str | None) -> Path | None:
    """If `text` is just the content of one file, return that file.

    A plain `cat bigfile` produces output byte-identical to bigfile, so spilling
    it would create a second copy of a file the agent can already address -- and
    pointing at the copy is what makes the loop attractive. Point at the original
    instead: no duplicate on disk, and the path is one the agent already knows.
    """
    if not (cmd and cwd):
        return None
    m = _SINGLE_FILE_CMD.match(cmd)
    if not m:
        return None
    try:
        p = (Path(cwd) / m.group("f")).resolve()
        if p.is_file() and p.stat().st_size == len(text.encode("utf-8", errors="replace")):
            return p
    except Exception:
        return None
    return None


def _truncate(
    text: str,
    max_chars: int = MAX_TOOL_RESULT_CHARS,
    *,
    cwd: str | None = None,
    cmd: str | None = None,
) -> str:
    """Cap oversized tool output middle-out, spill the whole thing, break loops.

    Three changes over the head-only version, each answering a measured failure
    across the 12 cached evolver sessions + 144 replay arms (30 truncation events,
    3,468,485 chars dropped, 0 ever recovered):

    MIDDLE-OUT. Head-only truncation silently ate the TAIL of every bulk loop --
    one `for t in <11 tasks>; do cat DIFF; done` delivered task 1 and dropped 10
    with nothing naming them. Keeping both ends means a batch's last item is as
    visible as its first, and the marker names the missing LINE RANGE rather than
    a character count, so the gap is addressable instead of merely quantified.

    SPILL. The full text goes to a content-addressed file, so nothing is lost even
    when the omitted region is a synthesized stream that no range read of any
    single file could reconstruct (14 of the 30 events were exactly that).

    LOOP BREAK. Spilling alone loops: the artifact is as big as what overflowed,
    so re-reading it overflows again. Content addressing makes the repeat
    *detectable*, and on the second sighting of the same bytes we stop offering
    the whole artifact and hand over a concrete bounded command instead.
    """
    if len(text) <= max_chars:
        return text

    total = len(text)
    lines = text.count("\n") + 1
    head_budget = max_chars // 2
    tail_budget = max_chars - head_budget
    # Cut on line boundaries so neither fragment ends mid-record.
    head = text[:head_budget]
    head = head[: head.rfind("\n") + 1] or head
    tail = text[total - tail_budget:]
    tail = tail[tail.find("\n") + 1:] or tail
    omitted = total - len(head) - len(tail)
    head_lines = head.count("\n")
    tail_lines = tail.count("\n")
    gap_first = head_lines + 1
    gap_last = max(gap_first, lines - tail_lines)

    src = _aliased_source(text, cwd, cmd)
    if src is not None:
        path_note = str(src)
        sha = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:12]
    else:
        p, sha = _spill_path(text, cwd)
        path_note = str(p) if p is not None else ""

    _spill_seen[sha] = _spill_seen.get(sha, 0) + 1
    seen = _spill_seen[sha]

    if path_note:
        # Emit a path RELATIVE to cwd. bash runs inside the container, where this
        # same directory is bind-mounted at /workspace, so a host-absolute path
        # would not resolve there. A prefix test on the raw string is not enough:
        # _aliased_source resolve()s its result, and on macOS that rewrites
        # /var -> /private/var, so the prefix silently fails and we would hand the
        # container an unusable absolute path. relpath handles it; reject only a
        # result that escapes cwd.
        # Try cwd both as given and resolved: the spill file is built from the
        # UNRESOLVED cwd while _aliased_source returns a RESOLVED path, so each
        # form relativizes against a different base. Testing only one left the
        # spill case emitting an absolute host path under a symlinked cwd.
        rel = path_note
        for base in ([str(Path(cwd)), str(Path(cwd).resolve())] if cwd else []):
            try:
                cand = os.path.relpath(path_note, base)
            except Exception:
                continue
            if not cand.startswith(".."):
                rel = cand
                break
        if seen == 1:
            how = (
                f"Full output ({total:,} chars, {lines:,} lines) is at `{rel}`. "
                f"To read the omitted region: "
                f"`awk 'NR>={gap_first} && NR<={gap_last}' {rel}` "
                f"— or in slices if that is itself large."
            )
        else:
            # Second sighting of identical bytes: the agent is re-reading the
            # thing that just overflowed. Refuse the whole artifact and give it a
            # bounded command with the numbers already filled in.
            #
            # Size the slice in BYTES, not as a fraction of the gap. A fraction
            # (the first version used a quarter) still overflows whenever the gap
            # is large -- my own test caught it offering 1,194 lines against a
            # 2,000-char cap. Budget half the cap and derive the line count from
            # the measured average line width, so the suggested command is
            # guaranteed to fit and the loop provably terminates.
            bytes_per_line = max(1, total // max(1, lines))
            span = max(1, (max_chars // 2) // bytes_per_line)
            how = (
                f"You have now received this same output {seen}x — re-reading it "
                f"whole will keep overflowing. Read the gap in bounded slices "
                f"instead, starting with "
                f"`awk 'NR>={gap_first} && NR<={gap_first + span - 1}' {rel}`, "
                f"then advance the range. Lines {gap_first:,}-{gap_last:,} are "
                f"the only ones you have not seen."
            )
    else:
        how = (
            f"Full output ({total:,} chars) could not be spilled to disk; lines "
            f"{gap_first:,}-{gap_last:,} are unavailable. Re-run the command over "
            f"a narrower input (one file, or one loop item at a time)."
        )

    # With very long lines either budget can land mid-line and hold ZERO
    # complete lines, making the line ranges nonsense — caught live on MB-it5
    # (270,339 chars over 54 lines rendered as "showing lines 1-43 and 55-54"),
    # and the same degeneracy on the head side renders "lines 1-0". Describe
    # each fragment by what it actually holds.
    head_desc = (f"lines 1-{head_lines:,}" if head_lines >= 1
                 else "a partial first line")
    tail_desc = (f"lines {gap_last + 1:,}-{lines:,}" if gap_last < lines
                 else "a partial final line")
    return (
        f"{head}\n[OUTPUT TRUNCATED MIDDLE-OUT — {omitted:,} of {total:,} chars "
        f"omitted; showing {head_desc} and {tail_desc}, "
        f"omitting lines {gap_first:,}-{gap_last:,}. {how}]\n{tail}"
    )


def _build_sandbox_profile(cwd: str) -> str:
    """Build a macOS sandbox-exec profile that restricts filesystem reads.

    Strategy: allow-default, deny file-read* on all directories, then whitelist
    workspace + project + individual binaries. grep -r / completes in <1s.
    """
    project_root = str(Path(cwd).resolve().parents[1]) if "/.cache/" in cwd else cwd
    # The interpreter tree (often a conda or venv install under the user's home,
    # which the deny-list above covers) and ~/.local must stay readable or nothing
    # runs at all. Derived from sys.executable rather than hardcoded so the grant
    # follows whatever environment launched this process.
    interp_root = str(Path(sys.executable).resolve().parents[1])
    home_local = str(Path.home() / ".local")
    return f"""(version 1)
(allow default)
(deny file-read*
  (subpath "/Users")
  (subpath "/home")
  (subpath "/usr/share")
  (subpath "/usr/local")
  (subpath "/usr/bin")
  (subpath "/usr/sbin")
  (subpath "/usr/libexec")
  (subpath "/opt")
  (subpath "/System")
  (subpath "/Library")
  (subpath "/Applications")
  (subpath "/var")
  (subpath "/Volumes")
  (subpath "/sbin")
  (subpath "/bin")
)
(allow file-read* (subpath "{cwd}"))
(allow file-read* (subpath "{project_root}"))
(allow file-read* (subpath "{interp_root}"))
(allow file-read* (subpath "{home_local}"))
(allow file-read* (literal "/usr/bin/bash") (literal "/usr/bin/env")
  (literal "/usr/bin/curl") (literal "/usr/bin/grep") (literal "/usr/bin/cat")
  (literal "/usr/bin/head") (literal "/usr/bin/tail") (literal "/usr/bin/find")
  (literal "/usr/bin/wc") (literal "/usr/bin/sort") (literal "/usr/bin/uniq")
  (literal "/usr/bin/sed") (literal "/usr/bin/awk") (literal "/usr/bin/ls")
  (literal "/usr/bin/cut") (literal "/usr/bin/tr") (literal "/usr/bin/tee")
  (literal "/usr/bin/xargs") (literal "/usr/bin/printf") (literal "/usr/bin/dirname")
  (literal "/usr/bin/basename") (literal "/usr/bin/mktemp") (literal "/usr/bin/touch")
  (literal "/bin/bash") (literal "/bin/sh") (literal "/bin/ls") (literal "/bin/cat")
  (literal "/bin/echo") (literal "/bin/test") (literal "/bin/mkdir")
  (literal "/bin/rm") (literal "/bin/mv") (literal "/bin/cp"))
(allow file-read* (subpath "/usr/lib"))
(allow file-read* (subpath "/dev"))
(allow file-read* (subpath "/private/tmp"))
(allow file-read* (subpath "/private/etc"))
"""

# Sandbox mode: "docker" (default, strict), "sandbox" (macOS sandbox-exec), "none" (raw bash)
_VALID_SANDBOX_MODES = {"docker", "sandbox", "none"}
_sandbox_mode = os.environ.get("EVOSKILL_SANDBOX", "docker")
if _sandbox_mode not in _VALID_SANDBOX_MODES:
    logging.getLogger(__name__).warning(
        f"EVOSKILL_SANDBOX={_sandbox_mode!r} not in {_VALID_SANDBOX_MODES}, defaulting to 'docker'"
    )
    _sandbox_mode = "docker"

# Per-agent Docker container pool.
# Each concurrent agent gets its own container for write isolation.
# Containers are reused across agents via an asyncio.Queue pool.
_docker_pool: asyncio.Queue | None = None
_docker_pool_lock = asyncio.Lock()
_docker_agent_container: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_docker_agent_container", default=None
)
# CC-style stateful working directory for the bash tool: a `cd` in one call
# persists into the next call (matching Claude Code semantics and the
# stateful-shell training distribution of modern agentic LLMs), while env
# vars/shell functions still reset per call (each call is a fresh process,
# which keeps the per-command PID-targeted timeout-kill machinery intact).
# Tracked container-side for docker mode, host-side otherwise. Per-agent
# isolation via ContextVar; reset at the start of each run.
_bash_persist_cwd: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_bash_persist_cwd", default=None
)
# When True, _acquire_docker_container skips the mount check and uses the
# assigned container as-is. Set by external solvers (e.g. TB2) that manage
# their own containers with non-standard mount layouts.
_docker_external_container: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_docker_external_container", default=False
)
# Set while the evolver has used the switch_runtime tool to redirect its own
# bash execution into a freshly-launched instance of a SPECIFIC task's
# runtime (e.g. a TB2.1 task's docker_image), to debug a past failure in situ
# rather than only reading static trace/diff text. Distinct from
# _docker_external_container (which TB2Solver also sets for its OWN task
# execution) because that flag alone doesn't tell _check_bash_safety whether
# the write-guard's allowed-dirs containment check is meaningful: the
# evolver's allowed_write_dirs are paths under ITS OWN host workspace
# (.claude/skills, .cache/evolver_scratch) which have zero relationship to a
# switched-into container's filesystem (no /workspace bind mount at all for
# TB2.1 task images) -- enforcing that containment there would block every
# write inside the debug container, including harmless scratch files used to
# reproduce the failure. The debug container is disposable (torn down by
# exit_runtime), so relaxing containment while switched can't corrupt
# anything persistent; the separate dangerous-pattern check in
# _check_bash_safety still applies regardless of this flag.
_docker_runtime_switched: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_docker_runtime_switched", default=False
)
# Saved pre-switch container assignment, restored by exit_runtime so the
# evolver returns to its OWN container (not a freshly re-acquired one from
# the pool) after debugging -- avoids leaking the original container as
# permanently "checked out" from the pool's perspective.
_docker_pre_switch_container: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_docker_pre_switch_container", default=None
)
_docker_pre_switch_external: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_docker_pre_switch_external", default=False
)
# Set by _exec_switch_runtime while the evolver is inside a switched task
# runtime; holds (default_timeout_sec, hard_cap_sec) for _exec_bash to apply
# INSTEAD OF the flat BASH_TIMEOUT, so the evolver's own bash calls get the
# same per-call budget that benchmark's actual solver gets (not a generic
# 120s regardless of which environment it's now literally sitting inside).
# None when not switched; _exec_bash then falls back to BASH_TIMEOUT exactly
# as before switch_runtime existed. Reset to None by exit_runtime.
_switched_bash_timeout: contextvars.ContextVar[tuple[int, int] | None] = contextvars.ContextVar(
    "_switched_bash_timeout", default=None
)
# Set by _exec_switch_runtime instead of _docker_agent_container when the
# docker_image path is launched via the Daytona backend (_SWITCH_RUNTIME_
# BACKEND == "daytona"): holds the live DaytonaSandboxHandle. Kept as its own
# contextvar rather than overloading _docker_agent_container (a bare
# container-name string) to avoid any type confusion in _exec_bash's
# dispatch -- _exec_bash checks this FIRST, before the _sandbox_mode-based
# docker/sandbox/plain-bash chain, so a Daytona-backed switch never touches
# the local-docker machinery at all. None when not switched, or when
# switched via the local-docker path instead. Reset to None by
# _force_cleanup_switched_runtime / exit_runtime.
_daytona_switch_handle: contextvars.ContextVar[Any | None] = contextvars.ContextVar(
    "_daytona_switch_handle", default=None
)

def _get_docker_host_ip() -> str:
    """Get the host IP visible from inside colima/Docker VM."""
    import subprocess as _sp
    try:
        out = _sp.check_output(["colima", "ssh", "--", "ip", "route"], text=True, timeout=5)
        for line in out.splitlines():
            if "default" in line:
                return line.split()[2]
    except Exception:
        pass
    return os.environ.get("DOCKER_HOST_IP", "host.docker.internal")

_firewall_host_ip: str | None = None

def _ensure_corpus_only_firewall(host_ip: str) -> None:
    """Kernel-level egress lockdown for closed-corpus benchmarks.

    Installs iptables + ip6tables chains in the Colima VM (DOCKER-USER hook)
    that drop ALL container egress except a TCP connection to the host gateway
    on the single search port. This is the enforcement layer the in-process
    bash guard can't provide: it blocks direct-IP fetches, bash /dev/tcp, raw
    python sockets, non-http protocols, and DNS-tunnel exfil via the host
    resolver (:53 is not in the allow rule). The host allow is scoped to
    host_ip:search_port/tcp ONLY — not the whole host — so :22/:53 and any
    other host service are unreachable. IPv4 and IPv6 both get a default-DROP
    (the sandbox image has no IPv6 stack today, but the v6 DROP closes the
    classic "v4 firewall, v6 wide open" bypass if that ever changes).
    Idempotent; re-applied each process and after any `colima restart`
    (rules don't persist across VM restarts). Self-heals if the docker host
    gateway IP changes mid-process (e.g. after a VM restart) — the chain is
    flushed and rebuilt for the new IP, so the old IP is no longer whitelisted
    and the new search-server IP is reachable.
    """
    global _firewall_host_ip
    if os.environ.get("EVOSKILL_CORPUS_ONLY") != "1":
        return
    if _firewall_host_ip == host_ip:
        return  # chain already built for this exact gateway IP
    if not re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host_ip):
        # _get_docker_host_ip() fell back to a hostname (e.g. host.docker.internal);
        # iptables -d needs a literal IP or it builds a broken allow rule.
        raise RuntimeError(
            f"EVOSKILL_CORPUS_ONLY=1 needs a numeric docker host IP, got {host_ip!r}. "
            f"Set DOCKER_HOST_IP or check `colima ssh -- ip route`.")
    import subprocess as _sp
    search_port = os.environ.get("SEARCH_PORT", "8099")

    def _run(bin_, *args):
        return _sp.run(["colima", "ssh", "--", "sudo", bin_, *args],
                       capture_output=True, text=True, timeout=15)

    def _build(bin_, host_allow):
        _run(bin_, "-N", "EVOSKILL-EGRESS")
        _run(bin_, "-F", "EVOSKILL-EGRESS")
        _run(bin_, "-A", "EVOSKILL-EGRESS", "-m", "conntrack", "--ctstate",
             "ESTABLISHED,RELATED", "-j", "RETURN")
        if host_allow:
            _run(bin_, "-A", "EVOSKILL-EGRESS", "-d", host_allow,
                 "-p", "tcp", "--dport", search_port, "-j", "RETURN")
        _run(bin_, "-A", "EVOSKILL-EGRESS", "-j", "DROP")
        if _run(bin_, "-C", "DOCKER-USER", "-j", "EVOSKILL-EGRESS").returncode != 0:
            r = _run(bin_, "-I", "DOCKER-USER", "1", "-j", "EVOSKILL-EGRESS")
            if r.returncode != 0:
                raise RuntimeError(f"{bin_} hook failed: {r.stderr.strip()}")

    try:
        # IPv4: allow only host_ip:search_port. IPv6: default-DROP (no allow).
        _build("iptables", f"{host_ip}/32")
        _build("ip6tables", None)
        # Verify the DROP actually took: a fresh container must not reach out.
        probe = _sp.run(
            ["docker", "run", "--rm", "--init", "--dns", "127.0.0.1",
             "evoskill-solver-sandbox", "bash", "-c",
             "curl -s -m4 https://1.1.1.1/ -o /dev/null -w %{http_code} || true"],
            capture_output=True, text=True, timeout=30)
        if probe.stdout.strip() not in ("", "000"):
            raise RuntimeError(
                f"egress probe reached the internet (HTTP {probe.stdout.strip()}) "
                f"— firewall not effective")
        _firewall_host_ip = host_ip
        print("[SANDBOX] corpus-only egress firewall active (v4+v6, %s:%s/tcp only, "
              "verified)" % (host_ip, search_port), flush=True)
    except Exception as e:
        # Fail CLOSED: a closed-corpus run without the firewall risks silent
        # contamination, which is worse than not running.
        raise RuntimeError(
            f"EVOSKILL_CORPUS_ONLY=1 but egress firewall setup failed: {e}"
        ) from e


def _assert_dataset_mount_is_train_only(spec: str) -> None:
    """Refuse to mount a dataset that contains val/test tasks (0725 directive).

    EVERY evolver container is created through this function, so this is the one
    place a contamination guard cannot be bypassed by a launcher. 11 SSB
    launchers each build their own EXTRA_MOUNTS string; the enforcement belongs
    here, not in 11 copies that can drift.

    Contract: a launcher that has scoped its dataset mount exports
    EVOSKILL_TRAIN_TASK_IDS (comma-separated). We then verify the mount source
    contains ONLY those task dirs, and raise otherwise — so a mount that was
    widened by accident fails at container creation instead of quietly handing
    the evolver val and test golden answers.

    When the var is UNSET we cannot know the split, so we warn loudly rather
    than block: the other 10 launchers still mount the whole corpus and should
    be visibly unconverted, not silently trusted.
    """
    try:
        src, _, rest = spec.partition(":")
        target = rest.split(":", 1)[0]
    except Exception:
        return
    if not target.rstrip("/").endswith("/ssb_data"):
        return
    allowed = os.environ.get("EVOSKILL_TRAIN_TASK_IDS", "").strip()
    src_path = Path(src)
    if not allowed:
        logger.warning(
            "[CONTAMINATION] %s is mounted at %s with no EVOSKILL_TRAIN_TASK_IDS "
            "set — this launcher has not been scoped, so val/test golden answers "
            "are reachable by the evolver. See stage_split_dataset().",
            src, target,
        )
        return
    allow = {t for t in (x.strip() for x in allowed.split(",")) if t}
    if not src_path.is_dir():
        return
    leaked = sorted(
        d.name for d in src_path.iterdir()
        if d.is_dir() and not d.name.startswith(".") and d.name not in allow
    )
    if leaked:
        raise RuntimeError(
            f"[CONTAMINATION] refusing to start: the dataset mounted at {target} "
            f"contains {len(leaked)} task dir(s) outside the train split "
            f"(e.g. {leaked[:5]}). The evolver must not be able to reach val or "
            f"test golden answers. Mount a train-only mirror "
            f"(stage_split_dataset) instead of the dataset root."
        )


async def _create_docker_container(cwd: str) -> str:
    """Create a new Docker sandbox container, return its name."""
    name = f"evoskill-sandbox-{uuid.uuid4().hex[:8]}"
    host_ip = _get_docker_host_ip()
    _ensure_corpus_only_firewall(host_ip)
    search_port = os.environ.get("SEARCH_PORT", "8099")
    search_url = f"http://{host_ip}:{search_port}"
    # --dns 127.0.0.1 blackholes external DNS so the solver cannot reach the
    # open internet (closed-corpus benchmark integrity). The search server is
    # addressed by numeric host IP via $SEARCH_URL, so it is unaffected.
    # Opt out with EVOSKILL_ALLOW_EXTERNAL_NET=1 for benchmarks that need web.
    dns_args = [] if os.environ.get("EVOSKILL_ALLOW_EXTERNAL_NET") == "1" else ["--dns", "127.0.0.1"]
    extra_mount_args: list[str] = []
    _mounts_cwd = os.environ.get("EVOSKILL_DOCKER_EXTRA_MOUNTS_CWD", "")
    if not _mounts_cwd or os.path.realpath(cwd) == os.path.realpath(_mounts_cwd):
        for spec in filter(None, os.environ.get("EVOSKILL_DOCKER_EXTRA_MOUNTS", "").split(",")):
            _assert_dataset_mount_is_train_only(spec.strip())
            extra_mount_args.extend(["-v", spec.strip()])
    # --rm: let the daemon reap the container when it stops, instead of relying
    # solely on our explicit `docker rm -f`. Per-rollout removal cannot keep up
    # under load — each `docker rm -f` forks a docker CLI process and must stop
    # the container first, so with 35 concurrent rollouts finishing, removals hit
    # their 15s timeout and containers pile up faster than they are reaped
    # (measured 2026-07-30: one arm reached 741 live containers, and the host hit
    # 983 running / 2828 total with 9.2GB of 13.9GB VM memory gone). --rm also
    # stops "stopped but never removed" containers from accumulating on disk.
    # OWNERSHIP LABELS, so an orphan is identifiable after its creator is gone.
    #
    # `--rm` reaps a container when it STOPS, and `_own_containers` tracks them
    # in-process — neither survives a SIGKILL of the runner. The container keeps
    # running (`sleep 86400`), owned by nothing, and the only prior way to find it
    # was the name prefix, which cannot distinguish "orphan from a dead run" from
    # "in use by a live run". Recording the owner PID makes that distinction
    # decidable: liveness of the PID is the whole test (same pattern fleet.py and
    # seats.py use to reclaim budget shares and seats).
    labels = [
        "--label", f"evoskill.owner_pid={os.getpid()}",
        "--label", f"evoskill.run_tag={os.environ.get('EVOSKILL_RUN_TAG', 'untagged')}",
        "--label", f"evoskill.created={int(time.time())}",
    ]
    proc = await asyncio.create_subprocess_exec(
        "docker", "run", "-d", "--rm", "--init", *dns_args, "--name", name,
        *labels,
        "-v", f"{cwd}:/workspace",
        *extra_mount_args,
        "-e", f"SEARCH_URL={search_url}",
        "-e", f"SEARCH_HOST={host_ip}",
        "-e", f"SEARCH_PORT={search_port}",
        "-w", "/workspace",
        os.environ.get("EVOSKILL_DOCKER_IMAGE", "evoskill-solver-sandbox"), "sleep", "86400",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    # Unlike every other subprocess call in this file, this one had no
    # timeout — a `docker run` that hangs (daemon contention under heavy
    # concurrent load) blocked forever with zero logging until the outer
    # per-task wall-clock timeout ate the whole budget silently (root-caused
    # via a Phoenix trace: 4 fast turns, then ~26min of total silence with
    # no further spans, closing at exactly the 1800s outer cutoff).
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=60)
    except asyncio.TimeoutError:
        try:
            proc.kill()
            await asyncio.wait_for(proc.communicate(), timeout=5)
        except Exception:
            pass
        raise RuntimeError(f"docker run timed out after 60s creating container {name}")
    if proc.returncode != 0:
        err = (stderr_bytes or b"").decode(errors="replace").strip()
        raise RuntimeError(f"Docker container creation failed (rc={proc.returncode}): {err}")
    _own_containers.add(name)
    return name


async def _init_docker_pool():
    """Lazily initialize the global container pool."""
    global _docker_pool
    if _docker_pool is not None:
        return
    async with _docker_pool_lock:
        if _docker_pool is None:
            _docker_pool = asyncio.Queue()


_docker_shared_container: str | None = None
_docker_shared_lock = asyncio.Lock()


async def _is_container_alive(name: str) -> bool:
    """Check if a Docker container is running."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "inspect", "-f", "{{.State.Running}}", name,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        return out.strip() == b"true"
    except Exception:
        return False


async def _get_container_cwd(name: str) -> str | None:
    """Return the host path mounted at /workspace in the container, or None."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "inspect", name,
            "--format", "{{range .Mounts}}{{if eq .Destination \"/workspace\"}}{{.Source}}{{end}}{{end}}",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        return out.decode().strip() or None
    except Exception:
        return None


def _expected_extra_mount_dests(cwd: str) -> set[str]:
    """Container-side mount destinations _create_docker_container would add
    for this cwd (EVOSKILL_DOCKER_EXTRA_MOUNTS, gated on
    EVOSKILL_DOCKER_EXTRA_MOUNTS_CWD matching) — empty if this cwd doesn't
    get the extra mounts."""
    mounts_cwd = os.environ.get("EVOSKILL_DOCKER_EXTRA_MOUNTS_CWD", "")
    if mounts_cwd and os.path.realpath(cwd) != os.path.realpath(mounts_cwd):
        return set()
    dests = set()
    for spec in filter(None, os.environ.get("EVOSKILL_DOCKER_EXTRA_MOUNTS", "").split(",")):
        parts = spec.strip().split(":")
        if len(parts) >= 2:
            dests.add(parts[1])
    return dests


async def _get_container_mount_dests(name: str) -> set[str]:
    """Return the full set of mount destinations inside a container."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "inspect", name,
            "--format", "{{range .Mounts}}{{.Destination}}\n{{end}}",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        return {line for line in out.decode().splitlines() if line}
    except Exception:
        return set()


async def _container_fully_matches(name: str, cwd: str) -> bool:
    """True iff a candidate container's /workspace mount matches cwd AND it
    also has every extra mount this cwd is supposed to get.

    Without the extra-mounts check, the pool's reuse criterion was cwd-only
    (see _acquire_docker_container) — a container created for the SAME cwd
    but BEFORE extra mounts applied (or from a code path that never applied
    them) would silently satisfy the mount==cwd check and get reused, even
    though bind mounts are fixed at `docker run` time and can't be added
    retroactively. Found via a live evolver trace: solver_outputs/ (an
    extra mount) was completely absent for one iteration's evolver
    container while its /workspace mount matched correctly — the evolver
    couldn't see any task diffs for that entire session with no error.
    """
    if await _get_container_cwd(name) != cwd:
        return False
    expected_extra = _expected_extra_mount_dests(cwd)
    if not expected_extra:
        return True
    actual = await _get_container_mount_dests(name)
    return expected_extra.issubset(actual)


async def _acquire_docker_container(cwd: str) -> str:
    """Get a container for the current agent.

    Read-only agents share a single container (no write isolation needed).
    Write-enabled agents get per-agent containers from the pool.
    Containers are only reused if their /workspace mount matches `cwd` —
    a container created for a per-task workspace must not be handed to the
    evolver which needs the main workspace.
    """
    if _bash_read_only.get():
        global _docker_shared_container
        async with _docker_shared_lock:
            if _docker_shared_container is not None and await _is_container_alive(_docker_shared_container):
                # Same mount-match requirement as the write-enabled path below
                # (bug M26) — without it, a second read-only agent with a
                # different cwd would silently receive the first agent's
                # container and read the wrong task's files. Latent until any
                # profile actually sets bash_read_only=True with varying cwds.
                if await _container_fully_matches(_docker_shared_container, cwd):
                    return _docker_shared_container
                await _cleanup_docker_container(_docker_shared_container)
            _docker_shared_container = await _create_docker_container(cwd)
            return _docker_shared_container

    assigned = _docker_agent_container.get()
    if assigned is not None:
        if await _is_container_alive(assigned):
            if _docker_external_container.get():
                return assigned
            if await _container_fully_matches(assigned, cwd):
                return assigned
            # Wrong mount — destroy it, don't return to pool (accumulation
            # caused 124 orphaned containers and resource exhaustion).
            await _cleanup_docker_container(assigned)
        _docker_agent_container.set(None)
    await _init_docker_pool()
    # Drain containers from pool, use first alive one with matching mount.
    # Destroy ALL mismatched containers — don't leave them in the pool
    # where they accumulate across iterations (seen: 80+ orphaned containers).
    name = None
    while not name:
        try:
            candidate = _docker_pool.get_nowait()
            if not await _is_container_alive(candidate):
                continue
            if await _container_fully_matches(candidate, cwd):
                name = candidate
            else:
                await _cleanup_docker_container(candidate)
        except asyncio.QueueEmpty:
            # Pool exhausted with no match — flush any remaining stale
            # containers that other concurrent agents released while we
            # were draining. Without this, containers from prior phases
            # accumulate in the pool (80+ observed in a 4-iter run).
            while True:
                try:
                    stale = _docker_pool.get_nowait()
                    await _cleanup_docker_container(stale)
                except asyncio.QueueEmpty:
                    break
            name = await _create_docker_container(cwd)
    _docker_agent_container.set(name)
    return name


async def _cleanup_docker_container(name: str):
    """Force-remove a container. Best-effort, no exceptions.

    Third instance of tonight's unbounded-subprocess-wait bug class (see
    _create_docker_container and the AIMD limiter fixes) — a hung `docker rm`
    would otherwise block whichever task is acquiring/replacing a container
    silently, with no bound, despite this function's own "best-effort"
    docstring implying it can't block the caller.
    """
    removed = False
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "rm", "-f", name,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        )
        rc = await asyncio.wait_for(proc.wait(), timeout=15)
        removed = rc == 0
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    # Only forget the container if it is actually GONE. Discarding
    # unconditionally (the previous behaviour) made every failed or
    # timed-out `docker rm` orphan the container AND drop it from the atexit
    # safety net, so the leak fed itself: heavy load slowed the daemon, slow
    # removals hit the 15s timeout, each timeout added a permanent orphan, and
    # the orphans slowed the daemon further. Measured that night: 203 -> 460 ->
    # 653 -> 1023 containers on a 13-CPU host, ending with 8.8GB of 13.9GB VM
    # memory consumed and the SSB scorer starved past the liveness gate.
    if removed:
        _own_containers.discard(name)
    else:
        logger.warning(
            "docker rm failed or timed out for %s — keeping it tracked so exit "
            "cleanup retries it (host may be under daemon contention)", name,
        )


async def _release_docker_container():
    """Release the finished rollout's container: remove it, or pool it for reuse.

    Pooling only pays off when a later rollout runs in the SAME cwd, because a
    container hard-binds its workspace at creation (-v {cwd}:/workspace) and
    acquisition rejects any mismatch. Every artifact benchmark here gives each
    task+sample its own workspace, so a pooled container can never be reused —
    it just waits to be destroyed by whichever later acquisition happens to
    drain it. Under concurrency, creation outruns that lazy drain and the host
    container count grows with rollouts completed rather than staying near the
    concurrency limit. Measured 2026-07-30 with two arms resuming: 419 -> 636
    containers in 90s, ~3/s, +25MB/s of VM memory, on course to exhaust a 14GB
    VM in about four minutes.

    So the default is to remove on release, which bounds live containers to
    roughly the admission limit. Set EVOSKILL_DOCKER_POOL=1 for the genuinely
    shared-cwd case, where reuse avoids a create per rollout.
    """
    name = _docker_agent_container.get()
    if name is None:
        return
    _docker_agent_container.set(None)
    if os.environ.get("EVOSKILL_DOCKER_POOL", "0") == "1" and _docker_pool is not None:
        await _docker_pool.put(name)
    else:
        await _cleanup_docker_container(name)

_CWD_SENTINEL_PREFIX = "__EVOSKILL_CWD__"
_CWD_SENTINEL_SUFFIX = "__END__"
_CWD_SENTINEL_RE = re.compile(
    re.escape(_CWD_SENTINEL_PREFIX) + r"(.*?)" + re.escape(_CWD_SENTINEL_SUFFIX)
)


def _augment_cmd_for_cwd_tracking(cmd: str) -> str:
    """Append a sentinel that reports the shell's final $PWD on stdout.

    The user command runs first; its exit code is preserved via the
    trailing `exit`. If the command exits/execs before the sentinel runs
    (explicit `exit`, `exec`, parse error), the sentinel simply never
    prints and the caller keeps the previous working directory.
    """
    return (
        f"{cmd}\n"
        f"_evoskill_rc=$?; "
        f"printf '\\n{_CWD_SENTINEL_PREFIX}%s{_CWD_SENTINEL_SUFFIX}\\n' \"$PWD\"; "
        f"exit $_evoskill_rc"
    )


def _extract_and_strip_cwd(out: str) -> tuple[str, str | None]:
    """Pull the final-cwd sentinel out of command output.

    Uses the LAST match: the genuine sentinel prints at the end of stdout,
    so a command that happens to echo the marker text earlier can't spoof
    the tracked directory."""
    matches = list(_CWD_SENTINEL_RE.finditer(out))
    if not matches:
        return out, None
    m = matches[-1]
    new_cwd = m.group(1).strip()
    start, end = m.span()
    # Also swallow the newlines our printf added around the sentinel.
    while start > 0 and out[start - 1] == "\n":
        start -= 1
    while end < len(out) and out[end] == "\n":
        end += 1
    cleaned = out[:start] + ("\n" if start > 0 and end < len(out) else "") + out[end:]
    return cleaned, (new_cwd if new_cwd.startswith("/") else None)


def _effective_bash_timeout(args: dict[str, Any]) -> int:
    """Resolve this call's timeout: flat BASH_TIMEOUT normally, or the
    switched-in benchmark's own default/hard-cap (with an optional per-call
    `timeout_sec` override, clamped to that benchmark's hard cap) once
    switch_runtime has redirected bash into a task's runtime. Outside a
    switched runtime, `timeout_sec` is accepted by the schema but has no
    effect here -- every call still uses the single flat BASH_TIMEOUT,
    unchanged from before this override existed.
    """
    override = _switched_bash_timeout.get()
    if override is None:
        return BASH_TIMEOUT
    default, hard_cap = override
    requested = args.get("timeout_sec")
    try:
        requested = float(requested) if requested is not None else None
    except (TypeError, ValueError):
        requested = None
    base = default if requested is None or requested != requested or requested <= 0 else requested
    return max(1, int(min(base, hard_cap)))


# ---------------------------------------------------------------------------
# Coverage feedback: closing the loop on owed reads
#
# An analysis protocol can oblige an agent to read specific evidence in full,
# but a requirement with no return path is unenforceable — the only coverage
# measurement we had ran offline against Phoenix AFTER the episode, so the
# evolver never learned it had skipped task X while it still had turns to go
# read it. That is why a self-reported coverage ledger could stay complete and
# vacuous run after run (ARC-AGI-2: a filled row per task, derivation almost
# never done; three 2026-07-24 episodes: `full` claimed for capped reads).
#
# These are module-level and pure so they can be tested without a live model,
# and reused if another harness's tool loop gains parity. The per-episode state
# stays local to execute_query — never a module global or contextvar, because
# the 2026-07-10 switch_runtime contextvar leak crossed agents under
# concurrency, and one evolver's reads must never credit another's ledger.
# ---------------------------------------------------------------------------
def build_coverage_groups(manifest: Any) -> list[dict[str, Any]]:
    """Normalize a published owed-reads manifest into trackable groups.

    Each entry is DISJUNCTIVE: satisfied by reading any one of `any_of`, because
    the protocol owes ONE passing trace per task rather than all of its samples.
    """
    groups: list[dict[str, Any]] = []
    for g in (manifest or []):
        if not isinstance(g, dict):
            continue
        paths = [str(p).strip() for p in (g.get("any_of") or []) if str(p).strip()]
        if paths:
            groups.append({
                "what": str(g.get("what") or paths[0]),
                "paths": paths,
                "seen": False,
            })
    return groups


# Commands that move a file's CONTENT into the model's context. Naming a path is
# not reading it: Stage 1 mandates `wc -c`/`wc -l` over every pinned trace in ONE
# call before opening any, so a matcher that credited any mention would have
# marked nearly every owed read satisfied on turn one and the gate would never
# have fired. `ls`, `wc`, `stat`, `du`, `find` enumerate and measure; they are
# deliberately absent. head/tail/sed count here because they do deliver content —
# whether they delivered ALL of it is a depth question this reference-level check
# does not claim to answer, and its wording says so.
# Ceiling on reminders per episode. Each one already has to earn itself with
# progress, so this is a backstop against a pathological read-one-per-turn
# stall, not the primary control.
COVERAGE_NUDGE_CAP = int(os.environ.get("EVOSKILL_COVERAGE_NUDGE_CAP", "4"))

_READ_VERB_RE = re.compile(
    r"(?:^|[\s;|&(])(?:cat|bat|less|more|nl|head|tail|sed|awk|grep|egrep|fgrep|rg|"
    r"cut|jq|strings|diff|python|python3)(?:\s|$)"
)


def credit_coverage(groups: list[dict[str, Any]], cmd: str, output: str = "") -> None:
    """Credit, in place, any owed group whose CONTENT appears in tool output.

    NOT deliberately generous. The previous version matched command text and was
    documented as erring toward silence on the theory that "a false 'seen' costs
    one unspent reminder". That cost model was wrong: a false "seen" costs the
    run's premise. The 0727 audit measured the consequence — the gate reported
    226/226 and 224/224 (0 reminders) on full-batch epochs where the evolver had
    received 254KB of tool output against 7.7MB of owed evidence, i.e. ~3%. It
    then declined to edit, and that null was read as evidence about saturation
    when it was evidence about this function.

    Credit now requires a fingerprint from the file's interior/tail to appear in
    what the tool actually returned. Text-only matching survives solely as a
    fallback for files with no usable fingerprint, and marks itself `weak_credit`.
    """
    for g in groups:
        if g["seen"]:
            continue
        # CONTENT: a line unique to this file among the owed set appeared in what
        # the tool returned. Subsumes globs and shell loops for free -- however the
        # bytes were obtained, the bytes show up. This is the strong basis, but it
        # is a LOWER bound: measured against owed reads whose path appears in a
        # read-verb command, it misses 52% (59% of those are head/tail/sed -n
        # partial reads whose window excludes the fingerprint, 14% full `cat`
        # reads defeated by output truncation).
        if output and any(fp in output for fp in (g.get("fingerprints") or [])):
            g["seen"] = True
            continue
        # NAMING: a read-shaped command names the file discriminatingly. Needed
        # because content alone under-credits by half, and because ~11% of owed
        # files have no unique line at all. Marked weak_credit -- it is an UPPER
        # bound, since `wc -l <path>` and a path inside a loop over other tasks
        # both match.
        #
        # Matching is against `path_tokens` -- the SHORTEST path suffix unique
        # among the owed set, computed once by _assign_unique_path_tokens. Do not
        # substitute a fixed number of components here. Both obvious choices are
        # wrong on real data:
        #  - bare basename: every scorer DIFF is literally named DIFF.txt (74 of
        #    226 owed entries in one batch), so one mention credited the batch --
        #    asymmetrically, +0.108 on the minibatch arm versus +0.007 on full
        #    batch, in favour of the contrast being reported.
        #  - last TWO components: DIFF paths are <task>/iter_0_s<N>/DIFF.txt, so
        #    the last two are `iter_0_s0/DIFF.txt`, shared by every task at that
        #    sample index. A `for t in ...; do cat .../$t/iter_0_s0/DIFF.txt`
        #    header contains that literal and credited 27 of 74 DIFF groups in the
        #    ground-truth test. The discriminating component is third-from-last
        #    here and elsewhere it is not -- so derive it, never assume it.
        if not cmd or not _READ_VERB_RE.search(cmd):
            continue
        for tok in (g.get("path_tokens") or []):
            if tok in cmd:
                g["seen"] = True
                g["weak_credit"] = True
                break


def drop_unreadable_owed(groups: list[dict[str, Any]], cwd: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep only groups the evolver could actually read from `cwd`.

    Demanding a file that does not exist would burn a turn on OUR artifact-capture
    bug, and the evolver has no way to comply. Absence is judged three-state, the
    same trap `_diff_pin_status` exists for: in a real run `solver_outputs/` can be
    an empty docker MOUNT POINT on the host, so "I cannot see it from here" must not
    be read as "it is not there". A group is dropped only when every one of its
    paths is *definitely* absent — the file is missing while its parent directory
    is plainly visible. Returns (kept, dropped_labels) so the caller can report the
    drop instead of silently shrinking coverage.
    """
    kept: list[dict[str, Any]] = []
    dropped: list[str] = []
    root = Path(cwd)
    for g in groups:
        definitely_absent = True
        for p in g["paths"]:
            fp = root / p
            if fp.exists():
                definitely_absent = False
                break
            if not fp.parent.is_dir() or not any(fp.parent.iterdir()):
                definitely_absent = False   # unknown, not absent — keep it
                break
        if definitely_absent:
            dropped.append(f"{g['what']} ({g['paths'][0]})")
        else:
            g["bytes"] = _owed_bytes(root, g["paths"])
            kept.append(g)
    _assign_unique_fingerprints(root, kept)
    _assign_unique_path_tokens(kept)
    return kept, dropped


# Coverage is credited on EVIDENCE OF CONTENT, not on command text. The old
# text-matching version reported 226/226 and 224/224 while the evolver had
# actually read ~3% of the owed bytes (0727 audit): a bare-basename match
# credited a file whenever its name appeared anywhere in a command — including
# inside a `for task in 55965 414-20 ...` header enumerating OTHER tasks — and
# `head -50 trace.md` earned full credit for reading a header that holds the task
# prompt and none of the failure.
_FP_MIN_LEN = 24          # a fingerprint shorter than this risks colliding
_FP_PER_FILE = 3          # sample a few, so truncation cannot hide all of them


def _owed_bytes(root: Path, paths: list[str]) -> int:
    for p in paths:
        fp = root / p
        try:
            if fp.is_file():
                return fp.stat().st_size
        except OSError:
            pass
    return 0


def _assign_unique_path_tokens(groups: list[dict[str, Any]]) -> None:
    """Per group, the SHORTEST path suffix that no OTHER owed file also matches.

    The naming half of the trigger needs a substring that identifies one owed file
    and not its siblings. Neither obvious choice works on real layouts: the bare
    basename is `DIFF.txt` for every scorer diff, and the last two components are
    `iter_0_s<N>/DIFF.txt`, shared by every task at that sample index -- a loop
    header enumerating other tasks contains that literal and credited 27 of 74
    DIFF groups. The discriminating component happens to be third-from-last for
    this layout; on the trace files it is the basename alone. So derive it per
    path instead of picking a number, exactly as fingerprint uniqueness is derived
    rather than assumed.

    Suffixes are grown one component at a time and tested against every OTHER
    owed path, so the token is as short as it can be while still discriminating.
    Paths that are genuinely identical across groups (the same file owed under two
    descriptions) fall through to the full path, which is correct -- crediting
    both is right, they are one file.
    """
    norm: dict[int, list[str]] = {}
    for g in groups:
        norm[id(g)] = [(p[2:] if p.startswith("./") else p).rstrip("/")
                       for p in g["paths"] if p]

    for g in groups:
        mine = norm[id(g)]
        others = [q for h in groups if h is not g for q in norm[id(h)]]
        toks: list[str] = []
        for p in mine:
            parts = p.split("/")
            chosen = p
            for n in range(1, len(parts) + 1):
                cand = "/".join(parts[-n:])
                # unique iff no OTHER owed path ends with this same suffix
                if not any(q == cand or q.endswith("/" + cand) for q in others):
                    chosen = cand
                    break
            toks.append(chosen)
        g["path_tokens"] = toks


def _assign_unique_fingerprints(root: Path, groups: list[dict[str, Any]]) -> None:
    """Give each owed group lines that occur in EXACTLY ONE owed file.

    Replaces per-file tail-half sampling, which was measured broken on 0727: it
    picks boilerplate, so one file's content credited many. `text: {"finished":
    true}` occurred in 137 of 226 owed files; 82% of files had at least one
    non-unique fingerprint and only 41/226 had all-unique ones. Reading a single
    trace could therefore satisfy most of the batch.

    Uniqueness is not a property of a file, so it CANNOT be computed per file --
    hence one set-level pass here rather than a call inside the per-group loop.
    A line qualifies only if it appears in no other owed file's text; longest
    first, because a long line is the least likely to coincide.

    Tail-preference is deliberately DROPPED. It existed so a trace header (which
    the evolver already holds verbatim in its own instructions) could not earn
    credit, but uniqueness subsumes that: a line the evolver was already given is
    not unique to the file. Keeping both filters only shrank the candidate pool.

    Files left with no unique line get an empty list. They are then uncreditable
    by content and rely on the naming half of the trigger -- 128 of 1136 owed
    files in the 0727 corpus, ~11%, so this is a real and reportable blind spot,
    not a rounding error.
    """
    texts: dict[str, set[str]] = {}
    resolved: dict[int, str] = {}
    for g in groups:
        for p in g["paths"]:
            fp = root / p
            key = str(fp)
            if key in texts:
                resolved[id(g)] = key
                break
            try:
                if not fp.is_file():
                    continue
                text = fp.read_text(errors="replace")
            except OSError:
                continue
            texts[key] = {
                ln.strip() for ln in text.splitlines()
                if len(ln.strip()) >= _FP_MIN_LEN
            }
            resolved[id(g)] = key
            break

    counts: Counter[str] = Counter()
    for lines in texts.values():
        counts.update(lines)

    for g in groups:
        key = resolved.get(id(g))
        lines = texts.get(key) if key else None
        if not lines:
            g["fingerprints"] = []
            continue
        uniq = [ln for ln in lines if counts[ln] == 1]
        uniq.sort(key=len, reverse=True)
        g["fingerprints"] = uniq[:_FP_PER_FILE]


HARNESS_PREFIX = "[HARNESS]"


def _harness_user(content: str) -> dict[str, str]:
    """A user-role message the HARNESS wrote, tagged so it is attributable.

    Harness-authored messages ride role=user for API compatibility, and
    harness/agent.py renders any user message starting with HARNESS_PREFIX as
    "### Harness Message" instead of "### User Query". Untagged, they render as
    the task author speaking — and the evolver READS these transcripts, so it
    would reason about "what the user asked" from text the scaffold wrote.

    Before this helper only the coverage reminder and the turn-budget marker were
    tagged; the empty-reask, JSON-salvage, compaction and turn-limit messages were
    not. None of those four are reachable for the SSB solver profile (it disables
    the salvage turn — see ssb_agent.py:180 — and no SSB trace on disk contains
    their text, verified), so no existing result is affected. They ARE reachable in
    other harnesses, which is why this is a parity fix rather than a live bug fix.

    NOT for composite messages that embed the user's own query — the handover
    message is original_query_text + template, and tagging it whole would
    misattribute the user's text to the harness, which is the same error in the
    opposite direction. Leave those untagged.
    """
    return {"role": "user", "content": f"{HARNESS_PREFIX} {content}"}


def coverage_gap_message(missing: list[dict[str, Any]]) -> str:
    """The reminder itself. Names what went unreferenced and what to do about it.

    A declared shortfall is a legitimate response: the protocol asks for reads OR
    an explicit declaration, and this message must not imply otherwise — turning
    it into a demand would push the evolver to spend its last turns proving
    compliance instead of reporting honestly.

    EVERY outstanding group is listed. An earlier version capped the enumeration at
    12 on the theory that a long list costs more attention than it buys, but that
    reasoning was wrong in both directions (user-flagged 0725): the message becomes
    unactionable, because the evolver cannot read what it is not told it owes, and
    "… and N more not listed here" tells it that it is working blind. The cost it
    was trading against is negligible — the live 20x3 episode's largest shortfall
    was 26 groups, ~2.5KB, against a 141K-token context.
    """
    lines = [
        "[HARNESS] Coverage check — nothing in your bash history referenced "
        f"{len(missing)} of the reads this batch owes (all of them are listed):"
    ]
    for g in missing:
        alt = "" if len(g["paths"]) == 1 else f" (or any one of {len(g['paths'])})"
        lines.append(f"  - {g['what']}: `{g['paths'][0]}`{alt}")
    lines.append(
        # NOT "in your ledger" (0725 proofread, 5 independent lenses): the only
        # ledger the protocol defines is two columns over FAILING tasks, so it has
        # no row for the passing-trace items this very message lists — and by rule
        # it was already sent in an earlier message, since this fires only at the
        # final answer. Pointing there made the instruction unsatisfiable, inviting
        # either an invented column or a dropped declaration. The protocol's real
        # shortfall channel is prose ("declare the rest in writing").
        "Each still owes a full read OR an explicit shortfall stated in writing, "
        "in this message. "
        "Reading is not the point — DECIDING is: for each one, say what it changes "
        "about the edit you already wrote — and if nothing, why. That edit is a "
        "hypothesis until this evidence tests it, and revising it now costs less "
        "than shipping it wrong. Be suspicious of \"the skill never loaded here, "
        "so my edit cannot affect it\": that says the TRIGGER did not match, which "
        "is a thing you control, not evidence the rule is right — a task that "
        # "by doing the opposite" was categorical and arrived last, overriding the
        # protocol's decisive qualifier: a pass by LUCK on an unsound route is not
        # evidence against a rule ("treat the task as failing"). Unqualified, this
        # made the weakest draw in the batch retract correctly-derived rules.
        "passed by a SOUND route opposite to your rule is a counterexample whether "
        "or not the rule fired. Then give your final answer, edited or not. This "
        "check sees only paths your commands mention — if you covered one another "
        "way, say so and move on."
    )
    return "\n".join(lines)


async def _exec_bash(args: dict[str, Any], cwd: str) -> str:
    cmd = args.get("command", "")
    if not isinstance(cmd, str):
        cmd = str(cmd) if cmd else ""
    if not cmd:
        return "Error: empty command"
    safety_err = _check_bash_safety(cmd, read_only=_bash_read_only.get())
    if safety_err:
        return safety_err
    bash_timeout = _effective_bash_timeout(args)
    container = None
    proc = None
    # CC-style cwd persistence: start where the previous bash call left off.
    persist_cwd = _bash_persist_cwd.get()
    aug_cmd = _augment_cmd_for_cwd_tracking(cmd)

    _daytona_handle = _daytona_switch_handle.get()
    if _daytona_handle is not None:
        # switch_runtime redirected into a Daytona sandbox instead of a
        # local container -- short-circuit BEFORE the _sandbox_mode-based
        # docker/sandbox/plain-bash chain below entirely, since none of that
        # local-execution machinery (container acquisition, PID-targeted
        # kill-on-timeout) applies here. _check_bash_safety above still ran
        # unconditionally, so safety checking is unaffected by this branch.
        try:
            _cd_prefix = (
                f"cd {_shell_quote(persist_cwd)} 2>/dev/null ||:; " if persist_cwd else ""
            )
            wrapped_cmd = f"{_cd_prefix}{aug_cmd}"
            out, rc = await asyncio.to_thread(
                _daytona_handle.exec, wrapped_cmd, timeout=bash_timeout, max_output_chars=None,
            )
            out, _new_cwd = _extract_and_strip_cwd(out)
            if _new_cwd:
                _bash_persist_cwd.set(_new_cwd)
            if rc != 0 and f"[exit code: {rc}]" not in out:
                out += f"\n[exit code: {rc}]"
            return _truncate(out, NO_TRUNCATE) if out.strip() else "(no output)"
        except Exception as e:
            return f"Error: {e}"

    try:
        if _sandbox_mode == "docker":
            container = await _acquire_docker_container(cwd)
            # Wrap command to echo PID for targeted kill on timeout. The
            # `exec` keeps the printed PID == the command shell's PID. The
            # cd prefix (wrapper shell, inherited through exec) restores
            # the persisted working directory; `2>/dev/null ||:` makes a
            # vanished directory degrade to the container default rather
            # than failing the command.
            _cd_prefix = (
                f"cd {_shell_quote(persist_cwd)} 2>/dev/null ||:; " if persist_cwd else ""
            )
            wrapped = f'echo $$; {_cd_prefix}exec bash -c {_shell_quote(aug_cmd)}'
            proc = await asyncio.create_subprocess_exec(
                "docker", "exec", container, "bash", "-c", wrapped,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        elif _sandbox_mode == "sandbox" and sys.platform == "darwin":
            profile = _build_sandbox_profile(cwd)
            _start_dir = persist_cwd if (persist_cwd and os.path.isdir(persist_cwd)) else cwd
            proc = await asyncio.create_subprocess_exec(
                "sandbox-exec", "-p", profile, "bash", "-c", aug_cmd,
                cwd=_start_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        else:
            _start_dir = persist_cwd if (persist_cwd and os.path.isdir(persist_cwd)) else cwd
            proc = await asyncio.create_subprocess_exec(
                "bash", "-c", aug_cmd,
                cwd=_start_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        async def _kill_bash_process():
            # Kill only the SPECIFIC bash process, not all agents' processes
            if _sandbox_mode == "docker" and container and proc and proc.pid:
                # Read the PID from the first line of stdout (from our echo $$ wrapper)
                _inner_pid = None
                if proc.stdout:
                    try:
                        _first_line = await asyncio.wait_for(proc.stdout.readline(), timeout=0.5)
                        _inner_pid = int(_first_line.strip())
                    except Exception:
                        pass
                if _inner_pid:
                    _kill = await asyncio.create_subprocess_exec(
                        "docker", "exec", container, "kill", "-9", str(_inner_pid),
                        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                    )
                    try:
                        await asyncio.wait_for(_kill.wait(), timeout=3)
                    except Exception:
                        pass
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                if proc:
                    proc.kill()
            if proc:
                try:
                    await asyncio.wait_for(proc.wait(), timeout=3)
                except Exception:
                    pass

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=bash_timeout)
        except asyncio.TimeoutError:
            # wait_for's OWN timeout firing -- a genuine, expected, per-command
            # timeout. Recoverable: clean up this one process and hand the
            # agent an error string it can react to; the turn loop continues.
            await _kill_bash_process()
            return f"Error: command timed out after {bash_timeout}s"
        except asyncio.CancelledError:
            # This task was cancelled from OUTSIDE this await (e.g. the
            # enclosing Agent-level asyncio.timeout() wall-clock deadline, or
            # genuine external cancellation) -- NOT the same as wait_for's
            # own timeout above. Swallowing this and returning a normal
            # string here (as this function used to do) means the
            # CancelledError never reaches the outer asyncio.timeout() block,
            # so Timeout.__aexit__ never observes it and never raises
            # TimeoutError -- the wall-clock budget enforcement is silently
            # defeated and the turn loop just keeps running unbounded. Clean
            # up the process the same way, then re-raise so cancellation
            # propagates correctly.
            await _kill_bash_process()
            raise
        # Strip the PID line from docker output
        if _sandbox_mode == "docker" and stdout:
            lines = stdout.split(b"\n", 1)
            stdout = lines[1] if len(lines) > 1 else b""
        out = (stdout or b"").decode("utf-8", errors="replace") + (stderr or b"").decode("utf-8", errors="replace")
        # Persist the shell's final working directory for the next call
        # (strip the sentinel BEFORE truncation so it can't leak into the
        # model-visible output or get cut off by the length cap).
        out, _new_cwd = _extract_and_strip_cwd(out)
        if _new_cwd:
            _bash_persist_cwd.set(_new_cwd)
        if proc.returncode != 0:
            out += f"\n[exit code: {proc.returncode}]"
        return _truncate(out, NO_TRUNCATE) if out.strip() else "(no output)"
    except Exception as e:
        return f"Error: {e}"


def _exec_read_file(args: dict[str, Any], cwd: str) -> str:
    try:
        p = _resolve_path(args["path"], cwd)
        text = p.read_text(errors="replace")
        lines = text.splitlines(keepends=True)
        offset = args.get("offset", 0) or 0
        limit = args.get("limit")
        if limit:
            lines = lines[offset : offset + limit]
        elif offset:
            lines = lines[offset:]
        numbered = "".join(f"{i + offset + 1}\t{line}" for i, line in enumerate(lines))
        return _truncate(numbered)
    except Exception as e:
        return f"Error: {e}"


def _exec_write_file(args: dict[str, Any], cwd: str) -> str:
    try:
        p = _resolve_path(args["path"], cwd)
        guard_err = _check_write_guard(p)
        if guard_err:
            return guard_err
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(args.get("content", ""))
        return f"Wrote {p}"
    except Exception as e:
        return f"Error: {e}"


def _exec_edit_file(args: dict[str, Any], cwd: str) -> str:
    try:
        p = _resolve_path(args["path"], cwd)
        guard_err = _check_write_guard(p)
        if guard_err:
            return guard_err
        text = p.read_text()
        old = args.get("old_string", "")
        new = args.get("new_string", "")
        if old not in text:
            return f"Error: old_string not found in {p}"
        count = text.count(old)
        if count > 1:
            return f"Error: old_string found {count} times in {p} (must be unique)"
        text = text.replace(old, new, 1)
        p.write_text(text)
        return f"Edited {p}"
    except Exception as e:
        return f"Error: {e}"


def _exec_glob(args: dict[str, Any], cwd: str) -> str:
    try:
        pattern = args.get("pattern", "")
        base = Path(cwd)
        matches = sorted(str(p) for p in base.glob(pattern))
        if not matches:
            return "(no matches)"
        return _truncate("\n".join(matches))
    except Exception as e:
        return f"Error: {e}"


def _exec_grep(args: dict[str, Any], cwd: str) -> str:
    try:
        pattern = args.get("pattern", "")
        search_path = args.get("path", cwd) or cwd
        include = args.get("include", "")
        regex = re.compile(pattern)
        base = _resolve_path(search_path, cwd)
        results: list[str] = []
        max_results = 500

        if base.is_file():
            files = [base]
        else:
            files = sorted(base.rglob("*"))

        for fp in files:
            if not fp.is_file():
                continue
            if include and not fnmatch.fnmatch(fp.name, include):
                continue
            try:
                for lineno, line in enumerate(fp.open(errors="replace"), 1):
                    if regex.search(line):
                        results.append(f"{fp}:{lineno}: {line.rstrip()}")
                        if len(results) >= max_results:
                            results.append(f"... (truncated at {max_results} matches)")
                            return "\n".join(results)
            except Exception:
                continue

        return _truncate("\n".join(results)) if results else "(no matches)"
    except Exception as e:
        return f"Error: {e}"


def _exec_skill(args: dict[str, Any], cwd: str) -> str:
    name = args.get("name", "").strip()
    if not name:
        return "Error: skill name is required"
    info = _active_skills.get().get(name)
    if info is None:
        _skills = _active_skills.get()
        available = ", ".join(_skills.keys()) if _skills else "(none)"
        return f"Error: unknown skill '{name}'. Available skills: {available}"
    try:
        return Path(info["path"]).read_text(errors="replace")
    except Exception as e:
        return f"Error reading skill '{name}': {e}"


async def _exec_switch_runtime(args: dict[str, Any], cwd: str) -> str:
    """Launch a fresh instance of a task's own runtime and redirect bash to
    it. Two mutually exclusive input modes:

    task_dir: mirrors TB2DirectSolver's own container-launch recipe exactly
    (scripts/run_tb2_evo.py) -- same docker run flags, same WORKDIR-
    inspection fallback, same tests/ copy-in -- so debugging happens in a
    genuinely representative environment, not an approximation of one.

    docker_image: for benchmarks with no local task_dir/task.toml at all
    (e.g. SWE-rebench, whose only per-task environment handle is a plain
    docker_image string looked up from a side metadata dict, not a
    filesystem path -- see scripts/run_swe_rebench_evo.py's
    `meta.get("docker_image")`). Generalized 2026-07-11 (user-requested)
    from the original TB2.1-only version -- this path skips the task.toml
    read and the tests/ copy-in (nothing to copy from) and uses the
    GENERIC_SWITCHED_* constants/defaults instead of the TB2-specific ones.
    """
    task_dir_str = args.get("task_dir", "").strip() if args.get("task_dir") else ""
    docker_image_arg = args.get("docker_image", "").strip() if args.get("docker_image") else ""
    if task_dir_str and docker_image_arg:
        return "Error: pass only one of task_dir or docker_image, not both"
    if not task_dir_str and not docker_image_arg:
        return "Error: task_dir or docker_image is required"

    if _docker_runtime_switched.get():
        return (
            "Error: already switched into a runtime — call exit_runtime first. "
            "Only one runtime can be switched into at a time."
        )

    task_dir: Path | None = None
    tests_dir: Path | None = None
    if task_dir_str:
        task_dir = Path(task_dir_str)
        toml_path = task_dir / "task.toml"
        if not toml_path.exists():
            return f"Error: no task.toml found at {toml_path}"
        try:
            import tomllib
            with open(toml_path, "rb") as f:
                task_cfg = tomllib.load(f)
            docker_image = task_cfg["environment"]["docker_image"]
            cpus = task_cfg["environment"].get("cpus", 1)
            memory_mb = task_cfg["environment"].get("memory_mb", 2048)
        except Exception as e:
            return f"Error: could not read task.toml: {e}"
        tests_dir = task_dir / "tests"
        label = task_dir.name
        default_timeout, hard_cap = TB2_SWITCHED_BASH_DEFAULT_SEC, TB2_SWITCHED_BASH_HARD_CAP_SEC
    else:
        docker_image = docker_image_arg
        try:
            cpus = float(args["cpus"]) if args.get("cpus") is not None else GENERIC_SWITCHED_CPUS
            memory_mb = int(args["memory_mb"]) if args.get("memory_mb") is not None else GENERIC_SWITCHED_MEMORY_MB
        except (TypeError, ValueError) as e:
            return f"Error: invalid cpus/memory_mb: {e}"
        label = docker_image
        default_timeout, hard_cap = GENERIC_SWITCHED_BASH_DEFAULT_SEC, GENERIC_SWITCHED_BASH_HARD_CAP_SEC

    container_name: str | None = None
    daytona_handle = None
    use_daytona = (not task_dir_str) and _SWITCH_RUNTIME_BACKEND == "daytona"

    if use_daytona:
        # The Daytona cloud-sandbox backend is not included in this release
        # (src/harness/daytona_sandbox.py is not shipped).
        raise NotImplementedError(
            "switch_runtime backend 'daytona' is not available in this release; "
            "unset EVOSKILL_SWITCH_RUNTIME_BACKEND (or set it to 'docker') to use "
            "the local Docker backend."
        )
    else:
        container_name = f"evoskill-runtime-switch-{uuid.uuid4().hex[:8]}"
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "run", "-d", "--name", container_name,
                "--cpus", str(cpus), "--memory", f"{memory_mb}m",
                docker_image, "sleep", "infinity",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            # 300s, not the original 60s: that was fine for TB2.1's own (already-
            # local) task images, but a docker_image the local daemon hasn't
            # pulled yet -- the common case for SWE-rebench's multi-GB images --
            # needs real pull time. 2026-07-11, confirmed live: an evolver's
            # switch_runtime call hit the 60s ceiling, got told "Error: timed
            # out", but `docker run -d` succeeded moments later anyway --
            # leaving a live, untracked container (never reaches
            # _own_containers.add below on the timeout path) that the atexit
            # cleanup handler can never find.
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=300)
            if proc.returncode != 0:
                err = (stderr_b or b"").decode(errors="replace").strip()
                return f"Error: failed to start runtime container: {err}"
        except asyncio.TimeoutError:
            # Even at 300s this can still fire (a genuinely huge image, or a
            # contended daemon) -- `docker run -d` may have actually succeeded
            # moments after our wait gave up, same failure mode as above just
            # rarer. Force-remove by NAME (safe even if it doesn't exist) so
            # this timeout can never leak a live, untracked container rather
            # than trusting a best-effort background task to catch it later.
            try:
                _cleanup_proc = await asyncio.create_subprocess_exec(
                    "docker", "rm", "-f", container_name,
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(_cleanup_proc.wait(), timeout=15)
            except Exception as _cleanup_exc:
                # Best-effort -- the error message returned below is still
                # accurate either way, but don't swallow this silently: a
                # failed cleanup here is exactly how containers leak.
                print(f"    [switch_runtime] cleanup of {container_name} after timeout failed: {_cleanup_exc}", flush=True)
            return f"Error: docker run timed out starting {docker_image} (any container that landed anyway has been removed)"
        except Exception as e:
            return f"Error: {e}"
        _own_containers.add(container_name)

        workdir = "/app"
        try:
            insp = await asyncio.create_subprocess_exec(
                "docker", "inspect", docker_image, "--format", "{{.Config.WorkingDir}}",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            wd_out, _ = await asyncio.wait_for(insp.communicate(), timeout=15)
            workdir = wd_out.decode().strip() or "/app"
        except Exception:
            pass

    tests_note = "(no tests/ copy-in on the docker_image path -- reproduce the verifier yourself once inside)"
    if tests_dir is not None:
        tests_note = "(this task has no tests/ directory)"
        if tests_dir.exists():
            try:
                mkdir_proc = await asyncio.create_subprocess_exec(
                    "docker", "exec", container_name, "mkdir", "-p", "/tests",
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(mkdir_proc.communicate(), timeout=10)
                copy_proc = await asyncio.create_subprocess_exec(
                    "docker", "cp", str(tests_dir) + "/.", f"{container_name}:/tests/",
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                await asyncio.wait_for(copy_proc.communicate(), timeout=30)
                tests_note = "test files copied to /tests -- run 'bash /tests/test.sh' to reproduce the verifier"
            except Exception as e:
                tests_note = f"(tests/ exists but copy-in failed: {e})"

    # Save the pre-switch container assignment so exit_runtime can restore it
    # exactly, rather than letting the evolver's own container leak as
    # permanently checked-out from the pool's perspective. Meaningful even
    # for a Daytona-backed switch: the evolver's OWN resting environment
    # (this executor's _sandbox_mode) is unaffected by that -- it's still a
    # local-docker container -- only left as an identity save/restore since
    # this code path doesn't touch _docker_agent_container in that case.
    _docker_pre_switch_container.set(_docker_agent_container.get())
    _docker_pre_switch_external.set(_docker_external_container.get())

    if daytona_handle is not None:
        _daytona_switch_handle.set(daytona_handle)
    else:
        _docker_agent_container.set(container_name)
        _docker_external_container.set(True)
    _docker_runtime_switched.set(True)
    _bash_persist_cwd.set(None)  # stale cwd from the old environment is meaningless here

    # Apply this path's own bash-timeout constants (not this module's generic
    # 120s) to every bash call made while switched in, exactly matching what
    # that benchmark's actual solver experiences.
    _switched_bash_timeout.set((default_timeout, hard_cap))

    handle_line = (
        f"Sandbox: {daytona_handle.sandbox_id} (Daytona cloud sandbox)"
        if daytona_handle is not None
        else f"Container: {container_name}"
    )
    return (
        f"Switched into a fresh instance of {label}'s runtime.\n"
        f"{handle_line}\n"
        f"Image: {docker_image}\n"
        f"Working directory: {workdir}\n"
        f"{tests_note}\n"
        f"NOTE: this is a NEW instance, not the exact state the failed attempt left behind "
        f"(that {'sandbox' if daytona_handle is not None else 'container'} no longer exists) -- but it's the same environment type. "
        f"Every bash call now runs inside this {'sandbox' if daytona_handle is not None else 'container'} until you call exit_runtime.\n"
        f"Bash timeout now matches this task's own solver: default "
        f"{default_timeout}s, up to {hard_cap}s via the "
        f"bash tool's timeout_sec argument (instead of the standard {BASH_TIMEOUT}s)."
    )


async def _force_cleanup_switched_runtime() -> None:
    """Idempotent switch_runtime cleanup, callable from anywhere — not just
    the evolver's own `exit_runtime` tool call.

    2026-07-10 evolver-tool-parity adversarial review finding (CRITICAL,
    confirmed by an independent verify pass): `_docker_runtime_switched`,
    `_switched_bash_timeout`, `_docker_external_container`, and
    `_docker_pre_switch_container(_external)` were previously reset ONLY
    inside `_exec_exit_runtime` — but a whole TB2.1 evolution run executes
    as ONE asyncio Task (`asyncio.run(run_evo()) -> loop.run() -> a plain
    `for` loop over iterations, no `create_task`/`gather` between them), so
    if the evolver's call ends abnormally (wall-clock timeout, or any
    exception `Agent._run_with_retry` retries past) while still switched
    in — a very plausible outcome, since switch_runtime is used for exactly
    the "deep failure-analysis" sessions most likely to run long — these
    contextvars stayed stuck for every subsequent iteration of the SAME
    run: `switch_runtime` became permanently rejected
    ("already switched into a runtime"), every ordinary bash call silently
    kept the TB2-switched 240s/900s timeout instead of the flat 120s
    `BASH_TIMEOUT`, and — the most serious part — `_check_bash_safety`'s
    write-directory sandbox restriction is itself gated on
    `not _docker_runtime_switched.get()`, so it was silently DISABLED for
    the rest of the run too.

    Called unconditionally from two places that can each be the last thing
    to run for a given invocation: `execute_query`'s `finally` (covers
    normal-Task-boundary exits) and `Agent._run_with_retry`'s
    exception-retry path (covers a retry within the SAME Task/context,
    which never passes through `execute_query`'s finally in between
    attempts). A no-op when nothing is switched in (`_docker_runtime_switched`
    is False), so this is safe to call defensively everywhere.
    """
    if not _docker_runtime_switched.get():
        return
    _daytona_handle = _daytona_switch_handle.get()
    if _daytona_handle is not None:
        # Daytona-backed switch never touched _docker_agent_container (the
        # evolver's own, unrelated local-docker container) -- only tear down
        # the sandbox and clear this contextvar; _docker_pre_switch_container
        # is an identity save/restore here (see _exec_switch_runtime), so
        # skip re-assigning it to avoid a needless no-op container swap.
        await asyncio.to_thread(_daytona_handle.delete)
        _daytona_switch_handle.set(None)
    else:
        container_name = _docker_agent_container.get()
        if container_name:
            await _cleanup_docker_container(container_name)
        _docker_agent_container.set(_docker_pre_switch_container.get())
        _docker_external_container.set(_docker_pre_switch_external.get())
    _docker_pre_switch_container.set(None)
    _docker_pre_switch_external.set(False)
    _docker_runtime_switched.set(False)
    _bash_persist_cwd.set(None)
    _switched_bash_timeout.set(None)  # back to the flat generic BASH_TIMEOUT


async def _exec_exit_runtime(args: dict[str, Any], cwd: str) -> str:
    if not _docker_runtime_switched.get():
        return "No runtime is currently switched into — nothing to do."
    await _force_cleanup_switched_runtime()
    return "Exited the switched runtime and returned to your own environment."


_TOOL_DISPATCH: dict[str, Callable[[dict[str, Any], str], str]] = {
    "bash": _exec_bash,
    "read_file": _exec_read_file,
    "write_file": _exec_write_file,
    "edit_file": _exec_edit_file,
    "glob": _exec_glob,
    "grep": _exec_grep,
    "skill": _exec_skill,
    "switch_runtime": _exec_switch_runtime,
    "exit_runtime": _exec_exit_runtime,
}


async def _execute_tool(name: str, arguments: dict[str, Any], cwd: str, max_chars: int = MAX_TOOL_RESULT_CHARS) -> str:
    handler = _TOOL_DISPATCH.get(name)
    if handler is None:
        return f"Error: unknown tool '{name}'"
    result = handler(arguments, cwd)
    if asyncio.iscoroutine(result):
        result = await result
    # cwd/cmd let _truncate spill into the workspace and recognise a plain
    # single-file read (which must alias the source, not duplicate it).
    return _truncate(
        result, max_chars, cwd=cwd,
        cmd=arguments.get("command") if name == "bash" else None,
    )


# ---------------------------------------------------------------------------
# Plain-text tool call fallback (the ~11% bug)
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(
    r'<tool_call>\s*\{.*?"name"\s*:\s*"(\w+)".*?"arguments"\s*:\s*(\{.*?\})\s*\}?\s*</tool_call>',
    re.DOTALL,
)

_JSON_TOOL_RE = re.compile(
    r'\{\s*"(?:id|name)"\s*:.*?"name"\s*:\s*"(\w+)".*?"arguments"\s*:\s*(\{[^}]*\})',
    re.DOTALL,
)


def _extract_plaintext_tool_calls(content: str) -> list[dict[str, Any]]:
    """Regex fallback: extract tool calls embedded in plain text content."""
    calls: list[dict[str, Any]] = []
    for rx in (_TOOL_CALL_RE, _JSON_TOOL_RE):
        for m in rx.finditer(content):
            name = m.group(1)
            try:
                args = json.loads(m.group(2))
            except json.JSONDecodeError:
                continue
            calls.append({
                "id": f"fallback_{uuid.uuid4().hex[:12]}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            })
    return calls


# ---------------------------------------------------------------------------
# API interaction
# ---------------------------------------------------------------------------

async def _chat_completion(
    client: httpx.AsyncClient | None = None,
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Single chat completion with retry, circuit breaker, and shared client."""
    if client is None:
        client = await _get_shared_client()

    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if extra_body:
        body.update(extra_body)

    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES_PER_CALL):
        try:
            await _circuit_breaker.before_call()
        except asyncio.TimeoutError as e:
            wait = min(RETRY_BACKOFF_BASE * (2 ** attempt), MAX_RETRY_BACKOFF)
            logger.warning(f"{e} — retrying in {wait:.1f}s (attempt {attempt+1})")
            last_err = e
            await asyncio.sleep(wait)
            continue
        try:
            try:
                resp = await client.post(url, headers=headers, json=body, timeout=HTTP_TIMEOUT)
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except (json.JSONDecodeError, ValueError) as e:
                        await _circuit_breaker.record_failure()
                        wait = min(RETRY_BACKOFF_BASE * (2 ** attempt), MAX_RETRY_BACKOFF)
                        logger.warning(f"Invalid JSON in 200 response, retrying in {wait:.1f}s (attempt {attempt+1})")
                        last_err = e
                        await asyncio.sleep(wait)
                        continue
                    choices = data.get("choices", [])
                    if not choices or not choices[0].get("message"):
                        await _circuit_breaker.record_failure()
                        wait = min(RETRY_BACKOFF_BASE * (2 ** attempt), MAX_RETRY_BACKOFF)
                        logger.warning(f"Empty response from API (no choices/message), retrying in {wait:.1f}s (attempt {attempt+1})")
                        last_err = RuntimeError("Empty API response")
                        await asyncio.sleep(wait)
                        continue
                    await _circuit_breaker.record_success()
                    # Non-streaming path has no per-chunk signal, so the whole
                    # call is one heartbeat. This is why max_clean_gap must
                    # exceed a plausible full non-streaming call.
                    _liveness_touch("llm")
                    return data
                if resp.status_code in (404, 429, 500, 502, 503):
                    await _circuit_breaker.record_failure()
                    wait = min(RETRY_BACKOFF_BASE * (2 ** attempt), MAX_RETRY_BACKOFF)
                    logger.warning(f"API {resp.status_code}, retrying in {wait:.1f}s (attempt {attempt+1})")
                    last_err = httpx.HTTPStatusError(
                        f"{resp.status_code}", request=resp.request, response=resp
                    )
                    await asyncio.sleep(wait)
                    continue
                _log_http_error_body(resp)
                resp.raise_for_status()
            except httpx.HTTPStatusError:
                raise
            except httpx.TransportError as e:
                await _circuit_breaker.record_failure()
                wait = min(RETRY_BACKOFF_BASE * (2 ** attempt), MAX_RETRY_BACKOFF)
                logger.warning(f"{type(e).__name__}: {e}, retrying in {wait:.1f}s (attempt {attempt+1})")
                last_err = e
                await asyncio.sleep(wait)
        finally:
            # try/finally instead of scattered after_call() calls at each exit
            # point (bug H24) — a bare call site list misses asyncio.CancelledError
            # raised mid-await (e.g. from an outer wall-clock timeout), which
            # permanently leaked the in-flight slot and could eventually starve
            # all subsequent concurrent LLM calls process-wide.
            await _circuit_breaker.after_call()

    raise last_err or RuntimeError("Max retries exceeded")


def _contains_json_object(text: str) -> bool:
    """True if text contains at least one parseable JSON object.

    Structural guard for the salvage turn — deliberately not a content
    filter (no refusal-phrase matching; see the removed-blocklist lesson).
    """
    start = text.find("{")
    if start == -1:
        return False
    end = text.rfind("}")
    if end > start:
        try:
            json.loads(text[start:end + 1])
            return True
        except Exception:
            pass
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    json.loads(text[start:i + 1])
                    return True
                except Exception:
                    return False
    return False


class StreamSilenceTimeout(asyncio.TimeoutError):
    """No SSE chunk received within the silence window."""
    pass


async def _chat_completion_stream(
    client: httpx.AsyncClient | None = None,
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    extra_body: dict[str, Any] | None = None,
    silence_timeout: int = STREAM_SILENCE_TIMEOUT,
    on_partial: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Streaming chat completion with token-silence hang detection.

    Returns the same dict shape as _chat_completion (compatible drop-in).
    Raises StreamSilenceTimeout if no SSE chunk arrives within silence_timeout.

    on_partial(content, reasoning, tool_calls_accum) is invoked on each
    meaningful SSE delta so callers can push near-real-time Phoenix snapshots.
    """
    if client is None:
        client = await _get_shared_client()

    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if extra_body:
        body.update(extra_body)

    _sc = await _get_shared_stream_client()
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES_PER_CALL):
        try:
            await _circuit_breaker.before_call()
        except asyncio.TimeoutError as e:
            wait = min(RETRY_BACKOFF_BASE * (2 ** attempt), MAX_RETRY_BACKOFF)
            logger.warning(f"{e} — retrying in {wait:.1f}s (attempt {attempt+1})")
            last_err = e
            await asyncio.sleep(wait)
            continue
        try:
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            reasoning_content_parts: list[str] = []
            reasoning_details_accum: list[dict] = []
            tool_calls_accum: dict[int, dict[str, Any]] = {}
            usage_data: dict[str, Any] = {}
            finish_reason = ""
            provider_name = ""
            model_returned = ""

            resp = await _sc.send(
                _sc.build_request("POST", url, headers=headers, json=body),
                stream=True,
            )

            if resp.status_code != 200:
                await resp.aclose()
                if resp.status_code in (404, 429, 500, 502, 503):
                    await _circuit_breaker.record_failure()
                    wait = min(RETRY_BACKOFF_BASE * (2 ** attempt), MAX_RETRY_BACKOFF)
                    logger.warning(f"API {resp.status_code} (stream), retrying in {wait:.1f}s (attempt {attempt+1})")
                    last_err = httpx.HTTPStatusError(
                        f"{resp.status_code}", request=resp.request, response=resp
                    )
                    await asyncio.sleep(wait)
                    continue
                await _log_http_error_body_stream(resp)
                resp.raise_for_status()

            try:
                async for line in _iter_sse_lines(resp, silence_timeout):
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if not provider_name:
                        provider_name = chunk.get("provider", "")
                    if not model_returned:
                        model_returned = chunk.get("model", "")

                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if delta.get("content"):
                            content_parts.append(delta["content"])
                        # Reasoning deltas MUST be accumulated: the non-streaming
                        # path returns them on the assistant message, which gets
                        # appended to history (DeepSeek reasoning passback).
                        # Dropping them silently changes multi-turn behavior vs
                        # the non-streaming executor.
                        if delta.get("reasoning"):
                            reasoning_parts.append(delta["reasoning"])
                        if delta.get("reasoning_content"):
                            reasoning_content_parts.append(delta["reasoning_content"])
                        if delta.get("reasoning_details"):
                            for rd in delta["reasoning_details"]:
                                if reasoning_details_accum and reasoning_details_accum[-1].get("index") == rd.get("index"):
                                    reasoning_details_accum[-1]["text"] = reasoning_details_accum[-1].get("text", "") + rd.get("text", "")
                                else:
                                    reasoning_details_accum.append(dict(rd))
                        for tc in delta.get("tool_calls", []):
                            idx = tc.get("index", 0)
                            if idx not in tool_calls_accum:
                                tool_calls_accum[idx] = {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
                            if tc.get("id"):
                                tool_calls_accum[idx]["id"] = tc["id"]
                            fn = tc.get("function", {})
                            if fn.get("name"):
                                tool_calls_accum[idx]["function"]["name"] = fn["name"]
                            if fn.get("arguments"):
                                tool_calls_accum[idx]["function"]["arguments"] += fn["arguments"]
                        fr = choices[0].get("finish_reason")
                        if fr:
                            finish_reason = fr

                        if on_partial and (
                            delta.get("content")
                            or delta.get("reasoning")
                            or delta.get("reasoning_content")
                            or delta.get("tool_calls")
                        ):
                            try:
                                _reason = "".join(reasoning_parts) or "".join(reasoning_content_parts)
                                on_partial(
                                    "".join(content_parts),
                                    _reason,
                                    tool_calls_accum,
                                )
                            except Exception:
                                pass

                    if chunk.get("usage"):
                        usage_data = chunk["usage"]
            finally:
                await resp.aclose()

            assistant_msg: dict[str, Any] = {}
            full_content = "".join(content_parts)
            if full_content:
                assistant_msg["content"] = full_content
            if reasoning_parts:
                assistant_msg["reasoning"] = "".join(reasoning_parts)
            if reasoning_content_parts:
                assistant_msg["reasoning_content"] = "".join(reasoning_content_parts)
            if reasoning_details_accum:
                assistant_msg["reasoning_details"] = reasoning_details_accum
            if tool_calls_accum:
                _tcs = [tool_calls_accum[i] for i in sorted(tool_calls_accum)]
                for _tc in _tcs:
                    if not _tc.get("id"):
                        _tc["id"] = f"fallback_{uuid.uuid4().hex[:12]}"
                assistant_msg["tool_calls"] = _tcs
            assistant_msg["role"] = "assistant"

            if not assistant_msg.get("content") and not assistant_msg.get("tool_calls"):
                await _circuit_breaker.record_failure()
                wait = min(RETRY_BACKOFF_BASE * (2 ** attempt), MAX_RETRY_BACKOFF)
                logger.warning(f"Empty streamed response, retrying in {wait:.1f}s (attempt {attempt+1})")
                last_err = RuntimeError("Empty streamed response")
                await asyncio.sleep(wait)
                continue

            await _circuit_breaker.record_success()
            if not usage_data:
                logger.warning("Streamed response carried no usage data — cost/token accounting will undercount this turn")
            result: dict[str, Any] = {
                "choices": [{"message": assistant_msg, "finish_reason": finish_reason}],
                "usage": usage_data,
                "model": model_returned,
            }
            if provider_name:
                result["provider"] = provider_name
            return result

        except StreamSilenceTimeout:
            await _circuit_breaker.record_failure()
            wait = min(RETRY_BACKOFF_BASE * (2 ** attempt), MAX_RETRY_BACKOFF)
            logger.warning(f"Stream silence timeout ({silence_timeout}s), retrying in {wait:.1f}s (attempt {attempt+1})")
            last_err = StreamSilenceTimeout(f"No SSE chunk for {silence_timeout}s")
            await asyncio.sleep(wait)
        except httpx.HTTPStatusError:
            raise
        except (httpx.TransportError, asyncio.TimeoutError) as e:
            await _circuit_breaker.record_failure()
            wait = min(RETRY_BACKOFF_BASE * (2 ** attempt), MAX_RETRY_BACKOFF)
            logger.warning(f"{type(e).__name__}: {e} (stream), retrying in {wait:.1f}s (attempt {attempt+1})")
            last_err = e
            await asyncio.sleep(wait)
        finally:
            # try/except/finally as one compound statement — finally always
            # runs exactly once per before_call(), including when
            # asyncio.CancelledError propagates uncaught past every except
            # clause here (bug H24; see the non-streaming call site above for
            # the full rationale).
            await _circuit_breaker.after_call()

    raise last_err or RuntimeError("Max retries exceeded (stream)")


def _is_sse_progress(line: str) -> bool:
    """True only for lines that are evidence of GENERATION progress.

    An SSE stream carries three kinds of line: payload (``data: {...}``), frame
    separators (empty), and comments (``: OPENROUTER PROCESSING`` keep-alives).
    Only the first means the model produced something. The consumer above
    likewise skips anything that is not ``data: ``, so this predicate matches
    exactly what the stream actually delivers.
    """
    return line.startswith("data: ")


async def _iter_sse_lines(resp: httpx.Response, silence_timeout: int):
    """Yield SSE lines from a streaming response, raising StreamSilenceTimeout on silence.

    Payload lines are liveness heartbeats (src/harness/liveness.py): the
    finest-grained evidence that generation is genuinely progressing, and what
    lets the watchdog leave a 25-minute stream alone while killing a dead
    socket in ~120s.

    KEEP-ALIVES ARE NOT HEARTBEATS (bug M-10). ``Progress.touch()`` does double
    duty: it resets the stall detector AND charges the elapsed gap to the clean
    budget. Touching on keep-alives therefore made a hung-but-still-pinging
    provider invisible twice over -- idle never grew, so no INACTIVITY kill,
    while clean_s accrued at wall-clock rate until it hit CLEAN_BUDGET. And
    CLEAN_BUDGET is deliberately NOT infra, so the phantom failure was scored
    as a genuine 0 and written to the RunCache: a wrong answer made permanent.

    Not touching on keep-alives makes such a stall trip INACTIVITY_KILL
    instead, which IS infra -- excluded from means, never cached, retried. The
    cost is that a model silent past INACTIVITY_TIMEOUT before its first token
    is killed and retried rather than waited out. That trade is deliberate: a
    wasted retry is recoverable, a cached wrong score is not.

    The socket-level ``silence_timeout`` is unchanged and still resets on ANY
    line -- a keep-alive really is proof the connection is alive, which is a
    different question from whether the model is producing tokens.
    """
    line_iter = resp.aiter_lines().__aiter__()
    keepalives = 0
    progressed = False
    while True:
        try:
            line = await asyncio.wait_for(line_iter.__anext__(), timeout=silence_timeout)
            if _is_sse_progress(line):
                _liveness_touch("sse")
                progressed = True
            else:
                keepalives += 1
            yield line
        except StopAsyncIteration:
            if keepalives and not progressed:
                # Worth saying out loud: the provider held the socket open and
                # sent nothing but keep-alives for the whole call.
                logger.warning(
                    "SSE stream closed after %d keep-alive/blank lines and ZERO "
                    "payload lines — provider produced no tokens", keepalives,
                )
            return
        except asyncio.TimeoutError:
            raise StreamSilenceTimeout(f"No SSE data for {silence_timeout}s")


# ---------------------------------------------------------------------------
# Multi-turn execution loop
# ---------------------------------------------------------------------------

async def execute_query(options: dict[str, Any], query: str, *, agent_name: str = "direct_agent", tag: str | None = None, initial_messages: list[dict[str, Any]] | None = None, create_root_span: bool = True, parent_span: Any | None = None) -> list[Any]:
    """Run a multi-turn tool-calling loop and return the conversation history.

    Args:
        options: Dict from build_direct_options() with keys:
            system, format, tools, model_id, provider_id, cwd
        query: The question/task to send to the agent

    Returns:
        List of dicts: the full message history (OpenAI format) plus a
        final metadata dict with usage stats.
    """
    if not isinstance(options, dict):
        raise TypeError(f"Direct executor requires dict options, got {type(options)}")

    provider = options.get("provider_id", "openrouter")
    # Per-agent key override. Carries the ENV VAR NAME, never the secret, so a
    # key can't leak via any path that serialises `options` (spans, logs,
    # cache-key debugging). Used to run the evolver on one account and the
    # solver on another when a model is allowlisted on only one of them
    # (e.g. when a model is allowlisted on only one of the two accounts).
    # NOT part of the cache tree_hash (skills + system prompt only), so
    # switching keys does not invalidate cached rollouts.
    _key_env = options.get("api_key_env")
    if _key_env:
        # Check BEFORE resolving: resolving first would run
        # ensure_provider_api_key() needlessly and, if the default key is also
        # unset, surface its "OpenRouter API key not configured" error instead
        # of the accurate one below.
        api_key = os.environ.get(_key_env)
        if not api_key:
            raise RuntimeError(
                f"api_key_env={_key_env!r} was requested but that env var is unset — "
                f"refusing to silently fall back to the default key, which would run "
                f"this agent on the wrong account."
            )
    else:
        api_key = ensure_provider_api_key(provider)
    model = options.get("model_id", "deepseek/deepseek-v4-flash")
    cwd = options.get("cwd", os.getcwd())
    tool_flags = options.get("tools", {})
    tools_oai = _get_enabled_tools(tool_flags)

    base_url = os.environ.get("DIRECT_API_BASE_URL", "https://openrouter.ai/api/v1")

    # Skill discovery: scan cwd/.claude/skills/ for SKILL.md files
    # Only inject skills into system prompt for solver agents, not evolvers.
    inject_skills = options.get("inject_skills", True)
    skills_dir = options.get("skills_dir") or os.path.join(cwd, ".claude", "skills")
    if inject_skills:
        discovered = _discover_skills(skills_dir)
        _active_skills.set(discovered)
        skill_index = _build_skill_index(discovered)
        if discovered and "skill" not in tool_flags:
            tool_flags["skill"] = True
            tools_oai = _get_enabled_tools(tool_flags)
    else:
        _active_skills.set({})
        skill_index = ""

    max_tool_chars = options.get("max_tool_output_chars", MAX_TOOL_RESULT_CHARS)

    # Write guard: restrict file writes to allowed directories.
    # Uses ContextVars so concurrent agents don't contaminate each other.
    _project_root.set(cwd)
    raw_write_dirs = options.get("allowed_write_dirs", [])
    _allowed_write_dirs.set([Path(d) for d in raw_write_dirs] if raw_write_dirs else [])
    _allowed_bash_write_dirs.set(raw_write_dirs)

    # bash_read_only: if True, bash cannot modify files at all (solver mode)
    _bash_read_only.set(options.get("bash_read_only", False))

    # Fresh persistent-cwd per agent run: cd's persist across bash calls
    # WITHIN a run, never across runs (or across pooled-container reuse).
    _bash_persist_cwd.set(None)

    # Overflow bookkeeping is per-run: the loop-break escalation counts repeats
    # of identical output WITHIN one agent's session, and carrying counts across
    # runs would escalate on a first sighting.
    _reset_spill_state()

    # Create scratch dir if specified
    scratch_dir = options.get("scratch_dir")
    if scratch_dir:
        Path(scratch_dir).mkdir(parents=True, exist_ok=True)

    messages: list[dict[str, Any]] = []
    system_text = options.get("system", "")
    if system_text:
        tool_names = [t["function"]["name"] for t in (tools_oai or [])]
        _suffix_path = Path(__file__).parent / "system_suffix.md"
        tool_nudge = ""
        if tool_names and _suffix_path.exists():
            tool_nudge = "\n\n" + _suffix_path.read_text().strip().format(
                tool_names=", ".join(tool_names),
            )
        # State the mechanism, not just the limit. The previous note described a
        # bare character count, and across 12 evolver sessions + 144 replay arms
        # the omitted region was recovered 0 times out of 30 -- a recovery route
        # the model is not told about is a route it does not take. So name all
        # three properties it can act on: both ends survive, the whole output is
        # on disk, and the marker carries the exact missing line range.
        truncation_note = (
            f"\n\nTool output limit: each tool result is capped at {max_tool_chars:,} characters. "
            f"Oversized output is truncated MIDDLE-OUT — you keep the beginning and the end, "
            f"and lose a contiguous middle region — so the last item of a batched command "
            f"survives, but items in the middle do not. The marker states the exact line "
            f"range omitted and gives a filesystem path holding the COMPLETE output; read "
            f"that path over the stated line range to recover what was cut. Re-reading the "
            f"whole file will simply overflow again."
        )
        messages.append({"role": "system", "content": system_text + skill_index + tool_nudge + truncation_note})

    # Schema example goes into the system prompt so the user message is
    # the question only — cleaner for the evolver's trace analysis.
    # Artifact-contract profiles skip it (user directive 2026-07-23): their
    # profile prompt already states the {"finished": true} contract, so the
    # generic suffix ("Replace the placeholder values with your actual
    # answer") is redundant noise. NOTE: removing it changes the effective
    # system prompt => new cache keys for those profiles from this commit on.
    schema_info = options.get("format", {})
    if schema_info and schema_info.get("schema") and not options.get("artifact_contract"):
        schema = schema_info["schema"]
        props = schema.get("properties", {})
        example = {}
        for key, spec in props.items():
            t = spec.get("type", "string")
            if t == "number":
                example[key] = 85
            elif t == "integer":
                example[key] = 42
            elif t == "boolean":
                example[key] = True
            else:
                example[key] = f"<your {key.replace('_', ' ')} here>"
        schema_suffix = (
            "\n\nWhen you have your final answer, respond with ONLY a JSON object "
            "(no markdown fences, no extra text before or after) like this example:\n"
            f"{json.dumps(example, indent=2)}\n"
            "Replace the placeholder values with your actual answer."
        )
        messages[0]["content"] += schema_suffix

    messages.append({"role": "user", "content": query})

    # Support conversation continuation: replace built messages with
    # pre-existing history (e.g., for eval feedback retries that
    # preserve the solver's full conversation context).
    if initial_messages:
        messages = list(initial_messages)

    extra_body: dict[str, Any] = {}
    enable_thinking = options.get("enable_thinking", False)
    thinking_effort = options.get("thinking_effort", "")
    if "deepseek" in model.lower():
        if not enable_thinking:
            extra_body["thinking"] = {"type": "disabled"}
        elif thinking_effort:
            extra_body["reasoning"] = {"effort": thinking_effort}
    elif thinking_effort:
        extra_body["reasoning"] = {"effort": thinking_effort}

    _service_tier = options.get("service_tier")
    if _service_tier:
        extra_body["service_tier"] = _service_tier

    _provider_order = options.get("provider_order")
    if _provider_order:
        extra_body["provider"] = {"order": _provider_order}
        # `order` alone is a PREFERENCE: OpenRouter still falls back to any other
        # upstream when the listed ones are busy or erroring. That silently
        # breaks a run whose whole point is a specific provider -- e.g. the 0722
        # rule that DSV4 must ride NATIVE DeepSeek (no :nitro), where a quiet
        # reroute to another host produces numbers attributed to the wrong
        # endpoint. Opt in to a hard pin with provider_allow_fallbacks=False;
        # omitted, behaviour is unchanged (fallbacks allowed).
        # DEEPSEEK FAMILY ONLY (2026-08-01). The native-provider directive is
        # narrow; GLM 5.2 et al. are deliberately served by the multi-provider
        # :nitro pool and must keep their fallbacks. Callers set
        # provider_allow_fallbacks from a process-wide env var, so gate here too
        # rather than trusting every caller to have checked the model family.
        _allow_fb = options.get("provider_allow_fallbacks")
        if _allow_fb is not None and "deepseek" in model.lower():
            extra_body["provider"]["allow_fallbacks"] = bool(_allow_fb)

    budget_type = options.get("budget_type", "turns")
    budget_value = options.get("budget_value", 40)
    # MAX_TURNS is meant as an absolute safety ceiling for time/cost-budgeted
    # runs (which have no turn count to bound the loop by construction), NOT
    # a silent cap on turn-budgeted runs. Confirmed as an active bug: 20+
    # evolver scripts pass budget_type="turns", budget_value=300, and were
    # being cut off exactly 100 turns short with num_turns=200 looking like a
    # normal voluntary stop (budget_exhausted never gets set, since the
    # explicit check below never gets a turn beyond 199 to fire on).
    _turn_loop_limit = max(MAX_TURNS, int(budget_value)) if budget_type == "turns" else MAX_TURNS
    compaction_threshold = options.get("compaction_threshold_tokens", 0)
    compaction_triggered = False
    compaction_count = 0
    max_compactions = 3

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cost_usd = 0.0
    total_cached_tokens = 0
    turn = 0
    t0 = time.monotonic()
    turn = 0
    budget_exhausted = ""
    _last_provider = ""
    _salvage_used = False
    _empty_reask_used = False

    # -- Coverage feedback (owed reads) --------------------------------------
    # The analysis protocol obliges the evolver to read certain traces in full.
    # Measuring that offline after the episode (Phoenix bash spans) told US what
    # it skipped but told IT nothing, so the requirement had no return path and
    # its self-reported ledger could stay vacuous run after run. The manifest is
    # published per-batch by the renderer that pinned those paths
    # (SelfImprovingLoop._publish_owed_reads -> TraceDB.owed_reads); each entry
    # is a disjunctive group satisfied by reading ANY of its paths, because
    # Stage 2 owes one passing trace per task rather than all of them.
    #
    # State is LOCAL to this call, never a module global or contextvar: the
    # 2026-07-10 switch_runtime contextvar leak crossed agents under
    # concurrency, and a coverage tracker keyed globally would bleed one
    # evolver's reads into another's ledger.
    _owed_groups = build_coverage_groups(options.get("coverage_owed_reads"))
    if _owed_groups:
        _owed_groups, _owed_dropped = drop_unreadable_owed(_owed_groups, cwd)
        if _owed_dropped:
            # Report it: a shrinking owed set is an artifact-capture signal, and
            # silently reducing coverage would look like a clean run.
            print(f"  [COVERAGE] {len(_owed_dropped)} owed read(s) not on disk — "
                  f"excluded (nagging for them would spend a turn on our bug): "
                  f"{_owed_dropped[:3]}", flush=True)
    # Repeat until covered, but every repeat must EARN itself: fire again only
    # when the previous reminder produced progress (strictly fewer unread groups
    # than when it was last issued). That bounds the total at
    # min(cap, initial_unread) firings and makes the failure mode benign -- an
    # evolver that reads nothing after a reminder is declining, which the
    # protocol permits, and repeating at it would spend its remaining turns on
    # our insistence instead of on its answer. Partial compliance, which a
    # one-shot gate let through unchallenged, is exactly the case this catches.
    # Normally 0. A caller RESUMING a session may have already appended a
    # coverage reminder as the last message before handing the history over --
    # the 2x2 ablation must, because DeepSeek rejects a request whose last
    # message is an assistant turn while tools are enabled. Without this the
    # counter would start at 0 and the `if _cov_nudges and declined(...)` guard
    # below would IGNORE the evolver's reply to that reminder, send another one,
    # and only honour the second identical declaration. Verified on MB-it8: the
    # stop token was emitted, dropped, re-requested, and honoured one round late
    # -- which breaks ESCAPE's promise that saying so ends the checks, and is
    # exactly the "harness decides after all" collapse the authority factor
    # exists to avoid.
    _cov_nudges = int(options.get("coverage_nudges_already_sent") or 0)
    # Seed the progress baseline too, or the gate below
    # (`len(_cov_missing) < _cov_last_missing`) compares against 0, reads as "no
    # progress", and NO further reminder is ever sent -- which would silently
    # reduce the gated cells to a single reminder while the selfstop cells kept
    # theirs. The baseline is what was outstanding when that pre-appended
    # reminder was written, and coverage_owed_reads is already scoped to exactly
    # that set, so its length IS the baseline.
    _cov_last_missing = len(_owed_groups) if _cov_nudges else 0
    _cov_declared = False           # evolver emitted the stop token AND it was honoured
    _cov_token_unoffered = False    # ...emitted it in a cell that never offered it
    # Which reminder policy is live. The 2x2 ablation knob IS the production knob:
    # 'off' disables reminding entirely (the A0 control), the four cell names select
    # a cell, and anything else raises rather than silently falling back -- a typo'd
    # arm name that quietly ran the default would produce a plausible wrong result.
    _COV_POLICY_NAME = os.environ.get("EVOSKILL_COVERAGE_POLICY", "disclose_gated")
    try:
        from .coverage_policy import POLICIES as _COV_POLICIES, SELFSTOP as _COV_SS, declined as _COV_DECLINED
    except Exception:                                   # policy module absent
        _COV_POLICIES, _COV_SS = {"disclose_gated": coverage_gap_message}, set()
        def _COV_DECLINED(_c: str) -> bool:  # noqa: E306
            return False
    if _COV_POLICY_NAME not in _COV_POLICIES:
        raise ValueError(
            f"EVOSKILL_COVERAGE_POLICY={_COV_POLICY_NAME!r} is not one of "
            f"{sorted(_COV_POLICIES)}"
        )
    _COV_POLICY = _COV_POLICIES[_COV_POLICY_NAME]
    _COV_SELFSTOP = _COV_POLICY_NAME in _COV_SS

    from contextlib import contextmanager
    @contextmanager
    def _noop_span():
        from opentelemetry import trace as _otrace
        _span = parent_span if (parent_span is not None and hasattr(parent_span, 'is_recording') and parent_span.is_recording()) else _otrace.get_current_span()
        if not (_span and _span.is_recording()):
            _span = _otrace.NonRecordingSpan(_otrace.SpanContext(trace_id=0, span_id=0, is_remote=False))
        class _NoopCtx:
            label = agent_name
            iter_prefix = ""
            ground_truth = ""
            scorer = None
        yield (_span, _NoopCtx(), "")
    _span_ctx = agent_root_span(_tracer, harness_name="direct", agent_name=agent_name, query=query, run_tag=tag or "") if create_root_span else _noop_span()
    with _span_ctx as (run_span, eval_ctx, _session_id):
     try:
      # OpenRouter prompt-cache sticky routing: pin this agent run to one
      # provider endpoint from turn 1 (critical for Z.AI/GLM automatic
      # caching and for long evolver loops). Without session_id, sticky
      # routing only activates AFTER a cache hit is observed.
      # See https://openrouter.ai/docs/guides/best-practices/prompt-caching
      _or_session = options.get("session_id") or _session_id
      if _or_session:
          extra_body["session_id"] = str(_or_session)[:256]
      # Optional explicit cache_control (Anthropic/Qwen/etc.). Do NOT set
      # this for Z.AI/GLM — their caching is automatic, and a top-level
      # cache_control can restrict routing away from GLM endpoints.
      _cache_control = options.get("cache_control")
      if _cache_control:
          extra_body["cache_control"] = _cache_control

      # Don't clobber a container the caller pre-assigned via the
      # _docker_external_container escape hatch (e.g. TB2's Harbor-managed
      # per-task container) — only clear stale state when no external
      # assignment is active for this async context.
      if not _docker_external_container.get():
          _docker_agent_container.set(None)
      # Log tool definitions on the root span
      tool_names_sent = [t["function"]["name"] for t in (tools_oai or [])]
      run_span.set_attribute("tools.sent", json.dumps(tool_names_sent))
      if tools_oai:
          run_span.set_attribute("tools.definitions", json.dumps(tools_oai, indent=2))
      if extra_body.get("session_id"):
          run_span.set_attribute("openrouter.session_id", str(extra_body["session_id"]))

      # Phase span: only for agents with compaction (evolver), not solver
      phase_num = 1
      phase_label = eval_ctx.label or agent_name
      phase_tag = f" [{tag}]" if tag else ""
      phase_span = None
      if compaction_threshold > 0:
          phase_span = _tracer.start_span(
              f"{eval_ctx.iter_prefix}{phase_label}{phase_tag}/phase.{phase_num}",
              context=set_span_in_context(run_span),
          )
          phase_span.set_attribute("openinference.span.kind", "CHAIN")
          phase_span.set_attribute("phase", phase_num)
      active_parent = phase_span or run_span

      client = await _get_shared_client()
      for turn in range(_turn_loop_limit):
            if turn > 0:
                await asyncio.sleep(0.5)
                # Cooperative pause point, same semantics as the claude
                # executor's (src/harness/claude/executor.py) -- confirmed
                # as an active bug that this was never wired in here despite
                # direct being the dominant harness by script count in this
                # repo: touching the pause flag file had zero effect under
                # set_sdk("direct"), silently not extending the wall-clock
                # deadline for the pause duration either. Placed at the same
                # "after the previous turn, before the next API call"
                # boundary; the current turn's tokens are already billed by
                # this point.
                from src.harness.pause import wait_if_paused
                await wait_if_paused(reason=f"after turn {turn}")

            # Push partial state for timeout recovery
            try:
                from src.harness.agent import _push_partial_state
                _push_partial_state(turn + 1)
            except Exception:
                pass

            # -- Turn span: start BEFORE API call to capture full latency --
            turn_span = start_turn_span(
                _tracer, active_parent,
                turn_num=turn + 1, model=model,
                eval_ctx=eval_ctx, agent_name=agent_name,
                run_tag=tag or "",
            )

            # Grow this turn span's output in place while streaming (no
            # live.N.pK clone spans — one turn.N that accumulates text).
            _live_tx: LiveTranscriptEmitter | None = None
            _on_partial = None
            if STREAM_ENABLED and LiveTranscriptEmitter.enabled():
                _live_tx = LiveTranscriptEmitter(turn_span)
                _on_partial = _live_tx.update

            if STREAM_ENABLED:
                resp_data = await _chat_completion_stream(
                    client,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    messages=messages,
                    tools=tools_oai if tools_oai else None,
                    extra_body=extra_body,
                    on_partial=_on_partial,
                )
            else:
                resp_data = await _chat_completion(
                    client,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    messages=messages,
                    tools=tools_oai if tools_oai else None,
                    extra_body=extra_body,
                )

            _provider = resp_data.get("provider", "")
            if _provider:
                turn_span.set_attribute("inference.provider", _provider)
                _last_provider = _provider
                # Process-level provider audit (2026-07-30, v4 prep): Phoenix
                # spans carry per-call provider but aren't grep-able live.
                # Log first-seen and every CHANGE so a silent reroute off a
                # pinned provider is visible in the run log the babysit
                # monitors tail. Counts let the end-of-run summary state
                # "N calls, all <provider>" as a compliance line.
                global _PROVIDER_AUDIT
                try:
                    _PROVIDER_AUDIT[_provider] = _PROVIDER_AUDIT.get(_provider, 0) + 1
                    if len(_PROVIDER_AUDIT) == 1 and _PROVIDER_AUDIT[_provider] == 1:
                        logger.warning(f"[PROVIDER] first response served by: {_provider}")
                    elif _PROVIDER_AUDIT[_provider] == 1:
                        logger.warning(
                            f"[PROVIDER] ROUTE CHANGE: now served by {_provider} "
                            f"(seen so far: {dict(_PROVIDER_AUDIT)})")
                except Exception:
                    pass

            usage = resp_data.get("usage", {})
            turn_prompt = usage.get("prompt_tokens", 0)
            turn_completion = usage.get("completion_tokens", 0)
            turn_cost = float(usage.get("cost", 0) or 0)
            if turn_cost <= 0 and turn_prompt + turn_completion > 0:
                turn_cost = estimate_cost(model, turn_prompt, turn_completion)
            total_prompt_tokens += turn_prompt
            total_completion_tokens += turn_completion
            total_cost_usd += turn_cost
            # Surface prompt-cache hits (Z.AI/GLM, OpenAI, etc.)
            _ptd = usage.get("prompt_tokens_details") or {}
            _cached = int(_ptd.get("cached_tokens") or usage.get("cached_tokens") or 0)
            total_cached_tokens += _cached
            if _cached:
                # The OpenInference STANDARD key — this is the one Phoenix's cost
                # calculator reads to price cache reads at the cached rate. Emitting
                # only our own `llm.cached_tokens` left Phoenix assuming every prompt
                # token was fresh, which inflated the k/N 20x3 episode's header to
                # $3.92 against an actual $1.0971 (3.57x). By turn 40 that episode was
                # 200,960/201,872 = 99.5% cache reads, all billed by Phoenix at the
                # full input rate. Phoenix keeps its OWN costSummary and ignores the
                # `llm.cost.total` we already set, so the fix has to be the token
                # breakdown, not the cost.
                turn_span.set_attribute(
                    "llm.token_count.prompt_details.cache_read", _cached)
                turn_span.set_attribute("llm.cached_tokens", _cached)  # kept: our own dashboards
                if turn_prompt > 0:
                    turn_span.set_attribute(
                        "llm.cache_hit_ratio", round(_cached / turn_prompt, 4)
                    )

            choices = resp_data.get("choices", [])
            if not choices:
                logger.warning("Empty choices in API response")
                turn_span.end()
                break

            choice = choices[0]
            assistant_msg = choice.get("message", {})
            finish_reason = choice.get("finish_reason", "")
            # Input messages: full conversation (what the LLM actually receives)
            _emit_messages_on_span(turn_span, "llm.input_messages", messages)

            # Output: assistant response + tool calls as separate blocks
            content_text = assistant_msg.get("content", "") or ""
            raw_tool_calls = assistant_msg.get("tool_calls")
            _emit_messages_on_span(turn_span, "llm.output_messages", [assistant_msg])
            turn_span.set_attribute("llm.token_count.prompt", turn_prompt)
            turn_span.set_attribute("llm.token_count.completion", turn_completion)
            turn_span.set_attribute(
                "llm.token_count.total", turn_prompt + turn_completion)
            # Break out reasoning vs content tokens so Phoenix shows how much of the
            # completion was thinking (DeepSeek reasoning_content is billed here but is
            # otherwise invisible — the gap that made iter-3 look like "0 reasoning").
            _reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
            if _reasoning_tokens:
                turn_span.set_attribute("llm.token_count.completion_details.reasoning", _reasoning_tokens)
                turn_span.set_attribute("reasoning_tokens", _reasoning_tokens)
                turn_span.set_attribute("content_tokens", turn_completion - _reasoning_tokens)
            # Phoenix prices LLM turn spans via Settings → Models (requires
            # llm.model_name, set in start_turn_span). Also stamp the real
            # OpenRouter turn cost for Attributes; CHAIN roots are never priced.
            if turn_cost > 0:
                turn_span.set_attribute("turn.cost_usd", turn_cost)
                turn_span.set_attribute("llm.cost.total", turn_cost)
            if raw_tool_calls:
                try:
                    turn_span.set_attribute("tool_calls", json.dumps(raw_tool_calls, default=str))
                except Exception:
                    pass

            # Plaintext tool-call extraction BEFORE building clean_msg, so that
            # recovered tool_calls appear in the message history the LLM sees.
            # Without this ordering, the model would see tool results on the
            # next turn without the corresponding tool_calls in its own prior
            # assistant message — a broken conversation history.
            tool_calls = assistant_msg.get("tool_calls")
            if not tool_calls and assistant_msg.get("content"):
                tool_calls = _extract_plaintext_tool_calls(assistant_msg["content"])
                if tool_calls:
                    assistant_msg["tool_calls"] = tool_calls

            # Preserve reasoning on the assistant message we carry forward.
            # DeepSeek V4 (thinking mode) REQUIRES the reasoning_content of any
            # tool-calling turn to be passed back in all subsequent turns —
            # omitting it returns a 400. Through OpenRouter the carrier is
            # `reasoning_details` (preferred — keeps encrypted/summarized blocks
            # intact) or `reasoning` (plaintext); both must be echoed UNMODIFIED.
            clean_msg = {k: v for k, v in assistant_msg.items()
                         if k in ("role", "content", "tool_calls", "refusal",
                                  "reasoning_details", "reasoning", "reasoning_content")}
            messages.append(clean_msg)

            # Do NOT flush LiveTranscriptEmitter here — it used to rewrite
            # llm.output_messages into one flat assistant blob (reasoning +
            # text + tool_call), undoing _emit_messages_on_span's split and
            # leaving duplicate Phoenix blocks (turn.1 Jul 23). Final
            # structured emit above is authoritative.

            try:
                from src.harness.agent import _push_partial_message
                _push_partial_message(clean_msg)
            except Exception:
                pass

            if not tool_calls:
                turn_span.end()
                flush_spans()
                if "evolver" in (agent_name or ""):
                    print(f"  [turn {turn+1}] done | ctx {turn_prompt:,}+{turn_completion:,} tok | cumul {total_prompt_tokens:,} | ${total_cost_usd:.4f} | {content_text[:80]}", flush=True)
                content = clean_msg.get("content", "") or ""
                if not content.strip() and turn > 0 and not _empty_reask_used:
                    _empty_reask_used = True
                    messages.append(_harness_user(
                        "Based on the information you found, provide your final answer "
                        "now as JSON with final_answer, reasoning, and confidence fields."
                    ))
                    continue
                # Salvage turn: a schema was requested but the final message
                # carries no parseable JSON object — persona collapse into
                # chat boilerplate, malformed JSON, or leaked tool markup
                # (all observed with DeepSeek V4 Flash). One structural
                # re-ask; if the model collapses again we break for real.
                if (
                    schema_info and schema_info.get("schema")
                    and content.strip()
                    and not _contains_json_object(content)
                    and not _salvage_used
                    # Artifact benchmarks score from disk — a malformed final
                    # message can't change the outcome, so no salvage turn.
                    and not options.get("artifact_contract")
                ):
                    _salvage_used = True
                    logger.warning(f"[{agent_name}] Final message has no JSON object — issuing salvage turn")
                    messages.append(_harness_user(
                        "Your last message was not a valid JSON answer. Output ONLY the "
                        "JSON object matching the required schema now — no markdown "
                        "fences, no other text."
                    ))
                    continue
                # Coverage feedback: the one point where an unread owed trace is
                # still fixable. ONE reminder only (same one-shot discipline as
                # the salvage turn above) — a gate that keeps re-firing would
                # trade the evolver's remaining turns for our own insistence,
                # and "declared shortfall" is a legitimate answer to it.
                _cov_missing = [g for g in _owed_groups if not g["seen"]]
                # Progress since the last reminder is what licenses another one.
                _cov_progressed = (
                    _cov_nudges == 0 or len(_cov_missing) < _cov_last_missing
                )
                # The evolver may END the loop by SAYING so, but only where the
                # active policy offered it that affordance. In the `_gated` cells
                # the token is ignored-but-logged: honouring an unoffered token
                # would collapse the authority factor the 2x2 manipulates.
                # declined() scans the ASSISTANT message only -- the reminder we
                # sent contains the token as an instruction, so a history-wide
                # search finds our own words and stops every arm after one round.
                if _cov_nudges and _COV_DECLINED(content):
                    if _COV_SELFSTOP:
                        print(f"  [COVERAGE] evolver declared sufficiency "
                              f"({_COV_POLICY_NAME}) — honoured, "
                              f"{len(_cov_missing)}/{len(_owed_groups)} still "
                              f"unreferenced", flush=True)
                        _cov_declared = True
                        _cov_missing = []
                    else:
                        print(f"  [COVERAGE] evolver emitted the stop token under "
                              f"{_COV_POLICY_NAME}, which never offered it — "
                              f"IGNORED (harness decides)", flush=True)
                        _cov_token_unoffered = True
                if _COV_POLICY is None:
                    _cov_missing = []          # policy 'off': never remind
                if _cov_missing and _cov_progressed and _cov_nudges < COVERAGE_NUDGE_CAP:
                    # Needs turns left to act: reading takes one and answering
                    # takes another, so below that the reminder would only
                    # displace the final answer it is meant to improve.
                    # The floor matters because parse_response takes the LAST
                    # substantive assistant message: if a nudged evolver spent
                    # every remaining turn reading and never re-answered, the
                    # message parsed would be its post-nudge narration, not the
                    # JSON it had already written. The turn-budget path below
                    # still forces a final answer turn (the evolver sets no
                    # artifact_contract), so this trades turns, never the answer.
                    _turns_left = (
                        int(budget_value) - (turn + 1)
                        if budget_type == "turns" else MAX_TURNS
                    )
                    if _turns_left >= 3:
                        _cov_nudges += 1
                        _cov_last_missing = len(_cov_missing)
                        print(
                            f"  [COVERAGE] {len(_cov_missing)}/{len(_owed_groups)} owed "
                            f"read(s) unreferenced — reminding evolver "
                            f"(#{_cov_nudges}, {_turns_left} turns left)", flush=True)
                        messages.append({
                            "role": "user",
                            "content": _COV_POLICY(_cov_missing),
                        })
                        continue
                if _owed_groups:
                    _seen_n = sum(1 for g in _owed_groups if g["seen"])
                    _strong = sum(1 for g in _owed_groups
                                  if g["seen"] and not g.get("weak_credit"))
                    _stop = "criterion_met"
                    if _cov_declared:
                        _stop = "declared_sufficient"
                    elif _COV_POLICY is None:
                        _stop = "policy_off"
                    elif _cov_missing and _cov_nudges:
                        _stop = ("no_progress" if not _cov_progressed
                                 else "cap_reached" if _cov_nudges >= COVERAGE_NUDGE_CAP
                                 else "too_few_turns")
                    elif _cov_missing:
                        _stop = "never_reminded"
                    print(
                        f"  [COVERAGE] final: {_seen_n}/{len(_owed_groups)} owed read(s) "
                        f"credited ({_strong} by content, {_seen_n - _strong} by naming) "
                        f"| policy={_COV_POLICY_NAME} reminders={_cov_nudges} "
                        f"stop={_stop}"
                        f"{' token_unoffered' if _cov_token_unoffered else ''}", flush=True)
                break

            # -- Tool spans: one per tool call within this turn --
            turn_tool_names: list[str] = []
            for tc in tool_calls:
                func = tc.get("function", {})
                tc_name = func.get("name", "")
                tc_id = tc.get("id", f"tc_{uuid.uuid4().hex[:12]}")
                turn_tool_names.append(tc_name)

                try:
                    tc_args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    tc_args = {}

                tool_sp = start_tool_span(_tracer, turn_span, tool_name=tc_name, tool_input=tc_args)
                try:
                    # Heartbeat both sides of the tool call AND throughout it.
                    # Bracketing alone made a long-but-healthy tool call
                    # indistinguishable from a wedged one: a bash command in a
                    # container legitimately runs for minutes (its real bound is
                    # EVOSKILL_BASH_TIMEOUT), while the idle gate is 120s, so
                    # every slow command was killed as a stall — measured in v3
                    # on 2026-07-30, 4 of 13 kills had last_kind tool_start or
                    # tool_done. That censors exactly the tasks whose solutions
                    # need heavy computation, which is a selection effect on
                    # difficulty, not a random loss.
                    _liveness_touch("tool_start")
                    result_str = await _await_beating(
                        _execute_tool(tc_name, tc_args, cwd, max_chars=max_tool_chars),
                        "tool_running")
                    _liveness_touch("tool_done")
                except Exception as _tool_err:
                    result_str = f"Error: tool execution failed — {type(_tool_err).__name__}: {_tool_err}"
                if _owed_groups and tc_name == "bash":
                    # AFTER execution on purpose: coverage is credited on returned
                    # content, so this cannot run before the tool produces output.
                    credit_coverage(_owed_groups, str(tc_args.get("command") or ""), result_str)

                # Verbatim: result_str is ALREADY capped to
                # max_tool_output_chars by _execute_tool, so this is
                # exactly the string appended to messages below.
                _out = span_text(result_str)
                tool_sp.set_attribute("tool.output", _out)
                tool_sp.set_attribute("output.value", _out)
                tool_sp.end()

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result_str,
                })

            turn_span.end()
            flush_spans()
            if "evolver" in (agent_name or ""):
                print(f"  [turn {turn+1}] {', '.join(turn_tool_names)} | ctx {turn_prompt:,}+{turn_completion:,} tok | cumul {total_prompt_tokens:,} | ${total_cost_usd:.4f} | {content_text[:80]}", flush=True)

            if finish_reason == "stop" and not tool_calls:
                break

            # -- Context compaction check (use turn_prompt = actual context size, not cumulative) --
            if (compaction_threshold > 0
                    and not compaction_triggered
                    and compaction_count < max_compactions
                    and turn_prompt >= compaction_threshold):
                compaction_triggered = True
                print(f"  [COMPACTING at {turn_prompt:,} context tokens (cumulative: {total_prompt_tokens:,})]", flush=True)

                # End phase 1 span
                phase_span.set_attribute("compaction", "triggered")
                phase_span.set_attribute("num_turns", turn + 1)
                phase_span.end()
                flush_spans()

                # Compaction span — child of current phase, LLM kind for block UI
                compact_span = _tracer.start_span(
                    f"{eval_ctx.iter_prefix}{phase_label}{phase_tag}/compaction",
                    context=set_span_in_context(phase_span),
                )
                compact_span.set_attribute("openinference.span.kind", "LLM")
                compact_span.set_attribute("llm.model_name", model)
                compact_span.set_attribute("llm.provider", "openrouter")
                compact_span.set_attribute("model", model)
                compact_span.set_attribute("context_tokens_at_trigger", turn_prompt)
                compact_span.set_attribute("cumulative_prompt_tokens", total_prompt_tokens)

                _compaction_prompt_path = Path(__file__).parent / "compaction_prompt.md"
                compaction_msg = _compaction_prompt_path.read_text(encoding="utf-8").strip()
                messages.append(_harness_user(compaction_msg))

                # Log full conversation (all messages sent to the API) as input
                _emit_messages_on_span(compact_span, "llm.input_messages", messages)

                # NOT _chat_completion_stream: compaction wants one blocking
                # summary, and this call must stay non-streaming so `usage` is
                # available synchronously for the cost accounting just below.
                compact_resp = await _chat_completion(
                    client, base_url=base_url, api_key=api_key,
                    model=model, messages=messages,
                    tools=None,
                    extra_body=extra_body,
                )
                compact_usage = compact_resp.get("usage", {})
                compact_prompt_tok = compact_usage.get("prompt_tokens", 0)
                compact_comp_tok = compact_usage.get("completion_tokens", 0)
                total_prompt_tokens += compact_prompt_tok
                total_completion_tokens += compact_comp_tok
                total_cost_usd += float(compact_usage.get("cost", 0) or 0)

                compact_choices = compact_resp.get("choices", [])
                summary_text = ""
                thinking_text = ""
                if compact_choices:
                    msg = compact_choices[0].get("message", {})
                    summary_text = msg.get("content") or ""
                    thinking_text = msg.get("reasoning_content") or msg.get("reasoning") or ""
                    if not summary_text and thinking_text:
                        summary_text = thinking_text

                # Log thinking + content as separate output blocks
                _co_idx = 0
                if thinking_text and thinking_text != summary_text:
                    compact_span.set_attribute(f"llm.output_messages.{_co_idx}.message.role", "reasoning")
                    compact_span.set_attribute(f"llm.output_messages.{_co_idx}.message.content", thinking_text)
                    _co_idx += 1
                if summary_text:
                    compact_span.set_attribute(f"llm.output_messages.{_co_idx}.message.role", "assistant")
                    compact_span.set_attribute(f"llm.output_messages.{_co_idx}.message.content", summary_text)

                compact_span.set_attribute("llm.token_count.prompt", compact_prompt_tok)
                compact_span.set_attribute("llm.token_count.completion", compact_comp_tok)
                compact_span.set_attribute(
                    "llm.token_count.total", compact_prompt_tok + compact_comp_tok)
                compact_cost = float(compact_usage.get("cost", 0) or 0)
                if compact_cost <= 0 and (compact_prompt_tok + compact_comp_tok) > 0:
                    compact_cost = estimate_cost(model, compact_prompt_tok, compact_comp_tok)
                if compact_cost > 0:
                    compact_span.set_attribute("compaction.cost_usd", compact_cost)
                    compact_span.set_attribute("llm.cost.total", compact_cost)
                # Token breakdown: reasoning vs content
                _reasoning_tokens = compact_usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
                _content_tokens = compact_comp_tok - _reasoning_tokens if _reasoning_tokens else compact_comp_tok
                compact_span.set_attribute("reasoning_tokens", _reasoning_tokens)
                compact_span.set_attribute("content_tokens", _content_tokens)
                compact_span.set_attribute("summary_length_chars", len(summary_text))
                compact_span.set_attribute("thinking_length_chars", len(thinking_text))

                if not summary_text.strip() or len(summary_text.strip()) < 200:
                    reason = "empty" if not summary_text.strip() else f"too short ({len(summary_text)} chars)"
                    print(f"  [COMPACTION FAILED — {reason}, forcing budget exhaust]", flush=True)
                    compact_span.set_attribute("compaction_result", f"failed: {reason}")
                    compact_span.end()
                    # Don't continue with bloated context — it'll just hard-fail
                    # on the next API call (empty response, context too large).
                    # Force the agent to produce output with what it has so far.
                    budget_exhausted = f"Compaction failed ({reason}) — context too large to continue"
                    break
                else:
                    print(f"  [COMPACTED — {len(summary_text):,} chars summary, restarting context]", flush=True)
                    compact_span.set_attribute("compaction_result", "success")
                    compact_span.end()
                    flush_spans()

                    # Reset for next compaction cycle
                    compaction_triggered = False
                    compaction_count += 1
                    _pre_compact_tokens = total_prompt_tokens
                    total_prompt_tokens = 0

                    # Start new phase
                    phase_num += 1
                    phase_span = _tracer.start_span(
                        f"{eval_ctx.iter_prefix}{phase_label}{phase_tag}/phase.{phase_num}",
                        context=set_span_in_context(run_span),
                    )
                    phase_span.set_attribute("openinference.span.kind", "CHAIN")
                    phase_span.set_attribute("phase", phase_num)
                    phase_span.set_attribute("compacted_from_tokens", _pre_compact_tokens)
                    active_parent = phase_span

                    # Restart context: system + single user message with original task + findings
                    system_msg = messages[0]
                    original_query_text = messages[1].get("content", "")
                    _handover_prompt_path = Path(__file__).parent / "handover_prompt.md"
                    _handover_template = _handover_prompt_path.read_text(encoding="utf-8").strip()
                    handover_msg = (
                        f"{original_query_text}\n\n"
                        f"---\n\n"
                        f"{_handover_template.format(summary=summary_text)}"
                    )
                    messages = [
                        system_msg,
                        {"role": "user", "content": handover_msg},
                    ]
                    # Compaction discards any coverage reminder already issued —
                    # it is neither the system message nor the original query. The
                    # progress gate infers "it saw the reminder and declined" from
                    # a lack of progress, and that inference no longer holds once
                    # the reminder itself has been thrown away. Re-license exactly
                    # one firing so a compacted episode is still told what it owes.
                    if _cov_nudges:
                        _cov_last_missing = len(_owed_groups) + 1

            # -- Budget check after each turn --
            # When budget is exhausted but the model just made tool calls,
            # allow ONE more turn so the model sees the tool results and
            # can produce a final answer. Without this, the model's last
            # tool calls execute but results are never sent back.
            if budget_type == "turns" and (turn + 1) >= int(budget_value):
                if tool_calls and (turn + 1) == int(budget_value) and not options.get("artifact_contract"):
                    messages.append(_harness_user(
                        "You have reached your turn limit. Based on all the information "
                        "gathered so far, provide your final answer NOW as a JSON object "
                        "with final_answer, reasoning, and confidence fields."
                    ))
                    # Remove tools for the bonus turn so model MUST produce text
                    tools_oai = None
                    continue
                budget_exhausted = f"Turn budget exhausted ({turn + 1}/{int(budget_value)} turns)"
                if options.get("artifact_contract"):
                    # Artifact benchmarks: the score is determined by files on
                    # disk, so no bonus answer turn. Persist a terminal marker
                    # (appended to the transcript, NEVER sent to the model —
                    # zero API cost) so evolver-facing traces and the Phoenix
                    # span record that the harness, not the model, ended the
                    # episode.
                    messages.append({
                        "role": "user",
                        "content": f"[HARNESS] Reached turn budget ({turn + 1}/{int(budget_value)}). Episode terminated — no further turns granted.",
                    })
                break
            elif budget_type == "time":
                elapsed = time.monotonic() - t0
                if elapsed >= float(budget_value):
                    budget_exhausted = f"Time budget exhausted ({elapsed:.0f}s/{float(budget_value):.0f}s)"
                    break
            elif budget_type == "cost" and total_cost_usd >= float(budget_value):
                budget_exhausted = f"Cost budget exhausted (${total_cost_usd:.4f}/${float(budget_value):.4f})"
                break

     finally:
      # Force-run switch_runtime's exit-cleanup if this call is ending
      # (normally or abnormally) while still switched in — must run BEFORE
      # _release_docker_container so that call sees the restored pre-switch
      # container (if any), not the switched-in one, and pools it normally.
      # See _force_cleanup_switched_runtime's docstring for the full incident
      # this fixes (2026-07-10 evolver-tool-parity adversarial review).
      await _force_cleanup_switched_runtime()
      # Release this agent's Docker container back to the pool
      await _release_docker_container()

      # End the active phase span (only exists when compaction is enabled)
      if phase_span is not None:
          try:
              phase_span.set_attribute("num_turns", turn + 1)
              phase_span.end()
              flush_spans()
          except Exception:
              pass

      # Finalise root span — runs even on timeout/cancellation
      elapsed_ms = int((time.monotonic() - t0) * 1000)
      last_content = ""
      for msg in reversed(messages):
          if isinstance(msg, dict) and msg.get("role") == "assistant":
              last_content = msg.get("content", "") or ""
              break

      # Inline async scoring — call score_fn before finalising so the
      # span name can be updated with [OK 1.00] / [WRONG 0.00].
      _inline_score = None
      _score_fn = options.get("score_fn")
      _gt = eval_ctx.ground_truth
      # Artifact benchmarks: score_fn reads files from disk and ignores the
      # answer text, so a budget-terminated episode with no final assistant
      # message must still be scored in-span (else the trace title loses its
      # [OK/WRONG] verdict exactly on the runs that most need flagging).
      _artifact = bool(options.get("artifact_contract"))
      if callable(_score_fn) and _gt and (last_content or _artifact):
          _answer = last_content or "(artifact-scored: output on disk)"
          if "final_answer" in last_content:
              try:
                  _m = re.search(r'"final_answer"\s*:\s*"([^"]*)"', last_content)
                  if _m and _m.group(1).strip():
                      _answer = _m.group(1)
              except Exception:
                  pass
          if _answer:
              try:
                  # to_thread, NOT run_in_executor:
                  # run_in_executor does not propagate
                  # contextvars, so the scorer's liveness heartbeats
                  # (score_start/score_done) were silent no-ops and a slow
                  # Docker verify was invisible to the watchdog. to_thread
                  # copies the context (PEP 567), so those touches register.
                  #
                  # _await_beating on top of that (2026-07-30): the scorer is
                  # ONE blocking call with no interior instrumentation, so
                  # start/done touches alone still leave a gap the length of
                  # the whole scoring phase, and >INACTIVITY_TIMEOUT scoring
                  # was being killed as a stall. Scoring is CPU work on
                  # arbitrary user spreadsheets — its duration is legitimately
                  # unbounded, so it must be bounded by CLEAN_BUDGET, never by
                  # the idle gate.
                  if asyncio.iscoroutinefunction(_score_fn):
                      _inline_score = await _await_beating(
                          _score_fn(query, _answer, _gt), "score_running")
                  else:
                      _inline_score = await _await_beating(
                          asyncio.to_thread(_score_fn, query, _answer, _gt),
                          "score_running")
                  _inline_score = float(_inline_score)
              except UngradableEpisode:
                  # THE FIFTH HANDLER. src/harness/errors.py named four broad
                  # `except Exception` handlers that turned an ungradable episode back
                  # into the 0.0 it was written to prevent; this is the one in the file
                  # that actually INVOKES the scorer, so ssb_v2_agent's own
                  # `except UngradableEpisode: raise` was being discarded one frame up.
                  # Swallowing it here does not merely lose the ledger entry: it leaves
                  # trace.inline_score None, so ssb_v2_agent's fallback pass re-runs
                  # score_and_diff on a workbook the first pass already mutated in place
                  # (xlfn prefixes, LibreOffice recalc, UNO data tables), and a transient
                  # container fault is reported as a verdict computed on a
                  # doubly-recalculated artifact.
                  raise
              except Exception:
                  pass

      if create_root_span:
        finalise_root_span(
          _tracer, run_span, eval_ctx,
          agent_answer=last_content,
          result_text=last_content,
          model=model,
          num_turns=turn + 1,
          prompt_tokens=total_prompt_tokens,
          completion_tokens=total_completion_tokens,
          cost_usd=total_cost_usd if total_cost_usd > 0 else estimate_cost(model, total_prompt_tokens, total_completion_tokens),
          system_prompt="",
          messages=messages,
          query=query,
          inference_provider=_last_provider,
          inline_score=_inline_score,
          budget_exhausted=bool(budget_exhausted),
      )

    if budget_exhausted:
        print(f"  [{budget_exhausted}]", flush=True)

    # Append metadata as the final element (parse_response reads it)
    elapsed_ms = int((time.monotonic() - t0) * 1000)
    messages.append({
        "_meta": True,
        "duration_ms": elapsed_ms,
        "num_turns": turn + 1,
        "usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
        },
        "total_cost_usd": total_cost_usd,
        "model": model,
        "budget_type": budget_type,
        "budget_value": budget_value,
        "budget_exhausted": budget_exhausted,
        "inline_score": _inline_score,
    })

    # Per-run totals for callers that need them. execute_query returns only the
    # message list, and changing that signature would ripple through every
    # caller, so the numbers are published here instead. Written at the END of a
    # run and read immediately after by the caller that awaited it -- safe for
    # sequential use, and safe under process-per-run concurrency. NOT safe for
    # two concurrent execute_query calls in ONE process: the second would
    # overwrite the first before it was read.
    global LAST_RUN_STATS
    LAST_RUN_STATS = {
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "cached_tokens": total_cached_tokens,
        "cost_usd": total_cost_usd,
        "turns": turn,
        "cache_hit_rate": (round(total_cached_tokens / total_prompt_tokens, 4)
                           if total_prompt_tokens else 0.0),
    }
    return messages


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _extract_json_from_text(text: str) -> Any | None:
    """Try to extract a JSON object from text (handles markdown fences, etc.).

    Always prefers whichever valid candidate ends LATEST in the text, across
    BOTH fenced and unfenced candidates uniformly (not fenced-vs-unfenced as
    separate fallback tiers) -- models are instructed to end their response
    with ONLY the JSON object, unfenced, but on very long responses can
    quote the format's placeholder example earlier as a self-reminder (e.g.
    "the JSON format is ```\\n{\"action\": \"<your action here>\", ...}\\n```,
    let me fill this in") before actually answering, sometimes INSIDE a
    markdown fence for its own readability. That early quote is itself
    syntactically valid JSON, so treating "the last fenced block" and "the
    last brace-matched block" as separate sequential fallbacks can each
    independently return the wrong (earlier) one depending on which form the
    placeholder happened to take -- confirmed as a real failure mode (a
    226k-char TB2.1 evolver response fenced the placeholder at char ~14k
    for a "let me look at the format" note, then gave its real,
    fully-reasoned, UNFENCED answer at the very end; a fenced-first-tier
    extraction returned the fenced placeholder and discarded the real
    analysis). Collecting every candidate's END offset and picking strictly
    the rightmost one fixes this regardless of which form is used where.
    """
    # Try direct parse first (fast path for the common, clean case).
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    candidates: list[tuple[int, str]] = []  # (end_offset, blob)

    # Fenced code blocks -- record each one's end offset in the ORIGINAL text.
    for m in re.finditer(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL):
        candidates.append((m.end(), m.group(1)))

    # Every outermost { ... } pair, found by scanning backward from each '}'
    # and walking back to its matching '{'. Skips past any span already
    # claimed by a successful match to avoid re-scanning inside it.
    i = len(text) - 1
    while i >= 0:
        if text[i] != "}":
            i -= 1
            continue
        depth = 0
        j = i
        matched_start = None
        while j >= 0:
            if text[j] == "}":
                depth += 1
            elif text[j] == "{":
                depth -= 1
                if depth == 0:
                    matched_start = j
                    break
            j -= 1
        if matched_start is not None:
            candidates.append((i + 1, text[matched_start : i + 1]))
            i = matched_start - 1
        else:
            i -= 1

    for _end_offset, blob in sorted(candidates, key=lambda c: c[0], reverse=True):
        try:
            return json.loads(blob)
        except json.JSONDecodeError:
            continue
    return None


def _find_last_substantive_assistant(messages: list[dict[str, Any]]) -> str:
    """Walk backwards through messages to find the last assistant message that
    has visible content (not just reasoning/thinking, not just tool_calls).

    Returns the content string, or "" if the solver exhausted its budget
    without producing a substantive response.
    """
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = (msg.get("content") or "").strip()
        if content:
            return content
        # For reasoning models (GPT-5 Nano, o-series): the final answer
        # may be in the reasoning field when content is null/empty.
        reasoning = (msg.get("reasoning") or "").strip()
        if reasoning and "{" in reasoning and "final_answer" in reasoning:
            return reasoning
    return ""


def parse_response(
    messages: list[Any],
    response_model: Type[BaseModel],
    get_options: Callable[[], Any],
) -> dict[str, Any]:
    """Parse the conversation history into AgentTrace field values.

    Extraction logic:
    1. Find the last assistant message with non-empty content (skip
       thinking-only and tool-call-only turns).
    2. If no such message → budget exhausted, score 0.
    3. Parse JSON from that message via _extract_json_from_text.
    4. If JSON parse fails → score 0 (no bold-text heuristic fallback).
    """
    meta: dict[str, Any] = {}
    if messages and isinstance(messages[-1], dict) and messages[-1].get("_meta"):
        meta = messages.pop()

    last_content = _find_last_substantive_assistant(messages)

    output = None
    parse_error = None
    raw_structured_output = None

    if last_content:
        parsed = _extract_json_from_text(last_content)
        if parsed is not None:
            if isinstance(parsed, dict) and "properties" in parsed and len(parsed) == 1:
                parsed = parsed["properties"]
            raw_structured_output = parsed
            try:
                output = response_model.model_validate(parsed)
            except (ValidationError, TypeError) as e:
                parse_error = f"{type(e).__name__}: {e}"
        else:
            parse_error = "No JSON object found in final assistant message"
    else:
        parse_error = "No substantive assistant message (budget exhausted or no content)"

    options = get_options()
    model_name = options.get("model_id", "unknown") if isinstance(options, dict) else "unknown"
    tools = (
        list(options.get("tools", {}).keys())
        if isinstance(options, dict) and options.get("tools")
        else []
    )
    # Full OpenAI-schema tool defs (name + description + parameters) for the
    # SAME tools `tools` names above — derived via the same _get_enabled_tools
    # helper execute_query itself uses to build the real `tools_oai` sent to
    # the model, so this is never a hand-authored second copy of the schema.
    tool_schemas = (
        _get_enabled_tools(options.get("tools", {}))
        if isinstance(options, dict) and options.get("tools")
        else []
    )

    # Prefer the parsed final_answer as `result` when available — the raw
    # last_content blob includes reasoning/confidence which confuses the scorer.
    result_text = last_content
    if output is not None and hasattr(output, "final_answer"):
        fa = getattr(output, "final_answer", None)
        if fa:
            result_text = str(fa)

    return dict(
        uuid=str(uuid.uuid4()),
        session_id=str(uuid.uuid4()),
        model=model_name,
        tools=tools,
        tool_schemas=tool_schemas,
        duration_ms=meta.get("duration_ms", 0),
        total_cost_usd=meta.get("total_cost_usd", 0.0),
        num_turns=meta.get("num_turns", 0),
        usage=meta.get("usage", {}),
        result=result_text,
        is_error=parse_error is not None,
        output=output,
        parse_error=parse_error,
        raw_structured_output=raw_structured_output,
        messages=messages,
        budget_exhausted=meta.get("budget_exhausted", ""),
        inline_score=meta.get("inline_score"),
    )
