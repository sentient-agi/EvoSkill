#!/usr/bin/env python3
"""The tau3 Claude Code bridge on Daytona, end to end: ONE throwaway machine, N train tasks, one rollout each.

A real Claude Code (2.1.281, claude-sonnet-5-5, default effort, stock system prompt) speaks the bridge's stream-json
control protocol (scripts/tau3_cc/tau2_cc_harness.py), with tau2's own Orchestrator, the banking_knowledge domain
(retrieval alltools: KB_search_bm25 + KB_search_dense over text-embedding-3-large through OpenRouter + a read-only
`shell` through sandbox-runtime 0.0.23; scripts/tau3_cc/tau3_retrieval.py), the OpenRouter customer and the user's tau3
grader. The module also builds the tau3 image's context (build-context; cc_pool_host.py uses write_build_dir) and
holds the unit helpers a tau3 leg reuses (scripts/tau3_cc/tau3_leg.py).

    .venv/bin/python scripts/daytona/tau3_cc_smoke.py run --data-root R [--tasks task_004,task_005 | --n 2]
        [--sample 1] [--dry-run] [--ttl-minutes 150] [--small] [--hard-stop-min M] [--token-file F]
        [--model claude-sonnet-5-5] [--job-set KEY=JSON ...] [--split F] [--out DIR]
    .venv/bin/python scripts/daytona/tau3_cc_smoke.py build-context --dest DIR   # offline, no Daytona
    .venv/bin/python scripts/daytona/tau3_cc_smoke.py delete --sandbox-id ID    # a leftover smoke machine

(the venv with the `daytona` extra: the Daytona SDK + python-dotenv.)

`run --dry-run` prints the whole plan (tasks, units, container ids, job.json, claude's argv as the harness builds it,
the launch command with the secrets as placeholders, the image's Dockerfile + context hashes (the srt tree's too),
machine parameters, the retrieval (alltools), what the harness implements) and touches nothing: no Daytona client, no
token file or key read, nothing written (the job is validated by the harness's own load_job in a temporary directory
that is removed at once).

WHAT `run` DOES, in order (every step recorded under .cache/tau3_cc_smoke/<stamp>/; the machine is deleted in
`finally`, also on a refusal, a crash, Ctrl-C, SIGTERM or SIGHUP):
  0. Host preflight: the tasks are in split['train'] of --split (default examples/tau3_banking/split.json) and never a
     dropped one; the curated data root (--data-root, built by scripts/tau3_make_data_root.py) matches its MANIFEST
     (scripts/tau3_make_data_root.verify_root); the pinned Claude Code linux binary is the one evoctl pins; the
     harness validates every job (load_job). The ONE Claude credential (src/orchestrator/cc_account.py): the
     subscription token file of --token-file (else EVOSKILL_CC_TOKEN_FILE; a regular file, mode 0600), else the
     environment's ANTHROPIC_API_KEY, read by cc_solver.read_credential; the OpenRouter key: OPENROUTER_API_KEY from
     the environment, else the repository's .env (scripts/tau3_cc/openrouter_key.py). Both stay in this process's
     memory.
  1. The image: scripts/daytona/Dockerfile.tau3_leg (the SSB leg Dockerfile's bytes + the /opt/tau3 layer + the KB
     shell's sandbox) copied next to a clean build context written here (tau3_ctx/tau2-bench = `git archive 363133a` of
     pyproject.toml uv.lock README.md LICENSE .python-version src, uv.lock patched by uvlock_1.0.1.patch and
     sha-checked; tau3_ctx/check = scripts/tau3_check_venv.py + tau3_venv_freeze.txt; tau3_ctx/srt/sandbox-runtime =
     the host's installed @anthropic-ai/sandbox-runtime 0.0.23 (EVOSKILL_SRT_SRC, else the package the `srt` on PATH
     belongs to), file by file with its modes, its tree sha recorded). The SDK uploads each COPY source tree whole, so
     the context is never the checkout (data/, .venv, .git: ~1 GB).
  2. The machine: name <daytona_lib_evo.PREFIX>tau3smoke-HHMMSS (daytona_lib_evo: ledgered, its own prefix, so the
     loop's cleanups never take it and its delete refuses any other name), 8 vCPU / 32 GiB / 40 GB (--small: 4/16/40),
     ephemeral, TTL --ttl-minutes, domain allow list openrouter.ai + api.anthropic.com (the pool machine's,
     pce.MACHINE_ALLOW_STR: the bridge runs as root and calls OpenRouter; the containers reach api.anthropic.com only,
     through evoctl's proxy). Read back: allow list, network_block_all, resources. Build logs to build_logs.txt.
  3. Machine probes as root: ssb_leg_pool.POOL_EGRESS_CMD (openrouter.ai + api.anthropic.com reachable, everything
     else refused), bash >= 5.1 (a here-string is a pipe), the image's LibreOffice version (the containers' PRE probe
     compares with it), the tau3 venv against its pin.
  4. evoctl + the egress proxy + Claude Code 2.1.281, exactly as the SSB pool host installs them
     (cc_pool_host.install_ctl / install_cc, sha-verified); the bridge (BRIDGE_FILES: tau2_cc_harness.py,
     tau2_bundle.py, tau3_retrieval.py -- the harness imports the last from its own dir), the user's grader
     (grader.py) and the split to /opt/tau3smoke (root 0700), sha-checked; the data root (its .embeddings_cache
     included) as a tgz, extracted AT ITS HOST PATH (the harness's TAU2_DATA_DIR = the task file's parents[4], the
     grader's rule; no container sees the machine's copy), MANIFEST re-checked there. Then TAU3_CHECK_PY under
     /opt/tau3/venv (also the pool host's tau3 machine check): tau2 imports, reads the root (97 tasks), a key-less GET
     of https://openrouter.ai/api/v1/models answers (the customer's path); the KB shell's binaries (tau2's
     _required_binaries + node) and srt 0.0.23; the document embeddings load from <root>/.embeddings_cache through
     tau2's EmbeddingsCache.get (on a copy: never written); a KB shell probe with the hardened deny list lists the KB
     and reads nothing of the data root or /opt, while the CONTROL (tau2's own settings) does read the task file. Any
     of the last three failing prints TAU3_CHECK_FAIL and no TAU3_CHECK line (the setup is refused).
  5. No usage reading is taken: a session rejected for a usage limit ends as an infrastructure end (exit 75).
  6. Per task, one cc-* container exactly as the SSB solver's (cc_solver.Box.prepare: cc_solver_lib.container_spec,
     limits_for(1), egress api.anthropic.com only; `up`; the PRE probe with pce.container_probe_problems), no skills,
     nothing uploaded into it (evoctl prepare makes an empty evo-owned /evo/work). Then job.json (no secret in it) in
     /opt/tau3smoke/units/<task>_s<k>/ (root 0700, OUTSIDE the container): workdir = its snapshot/ (the bundle is
     written there as root; shipping it into /evo/work is not built), out_dir = out/, home /home/evo, token_fd 3,
     token_kind (oauth / api_key), key_fd 4, container = {evoctl: [python3, evoctl.py], cid, cwd /evo/work}: the
     harness's own container mode builds `evoctl enter --cid C --cwd /evo/work --keep-fd3 -- /usr/bin/env
     HOME=/home/evo <CC_ENV> <TAU2_ENV> CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR=3 claude <stream argv>` (an API key:
     CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR=3; evoctl starts ARGV with `env -i` + its EVO_ENV, so claude's variables
     travel in the argv). Every other job key is the harness's default (agent_time_s 1200, the customer's model,
     timeout and retries, max_errors).
  7. The bridge, launched detached in ONE SDK exec whose env carries the two secrets (PATH set explicitly to
     VM_PATH: srt, rg, bwrap, socat and node for tau2's KB shell):
       (setsid nohup env -u CLAUDE_CODE_OAUTH_TOKEN -u TAU3_OR_KEY PATH=... /bin/bash -c '<cd U; harness --job job.json
        >bridge.log 2>&1 & echo $! >bridge.pid; wait; echo $? >rc>' ...
        3<<<"$CLAUDE_CODE_OAUTH_TOKEN" 4<<<"$TAU3_OR_KEY" &)
     The Claude credential is fd 3 (evoctl --keep-fd3 passes exactly that fd into the container), the key fd 4 (the
     harness reads it into litellm.openrouter_key and closes it). Neither is in any argv, file, or the bridge's
     environment; the command string is refused if it holds either.
  8. Poll every 15 s (rc, stream / stdin bytes, the bridge's pid, bridge.log's tail -- redacted on the host). The
     harness's only limit is the agent clock (agent_time_s; the customer's time never counts), so the driver's hard
     stop is agent_time_s + 40 min (--hard-stop-min): SIGTERM to the bridge (the harness kills claude, writes
     harness.json, exits 75); after 2 more minutes without an rc, `evoctl down` and SIGKILL.
  9. Collect each ended unit: `evoctl down` (freeze, tmpfs scan, cgroup kill, proxy off, state scan for the token
     prefix and sk-or-v1-), `evoctl pack --part /evo/work --part /home/evo/.claude` (Claude Code's own transcript is
     /home/evo/.claude/projects/-evo-work/<session>.jsonl), the unit dir as a tgz (job.json, bridge.log, rc,
     out/{claude.stream.jsonl, claude.stdin.jsonl, claude.stderr, harness.json}, snapshot/tau2/*), the grade on the
     machine (the user's grader under /opt/tau3/venv, `env -u OPENROUTER_API_KEY`, offline), then `down --wipe`.
 10. On the host: the same grader under third_party/tau3-bench/.venv on the downloaded snapshot (a Linux-made
     conversation graded on the host: strict-replay parity), every downloaded file scanned for the credential, its
     prefix, the key and sk-or-v1-, then SMOKE_REPORT.json.
 11. finally: bridges killed, every container `down --wipe`d (verified), the machine deleted by id and re-listed for
     up to 3 min until it is gone; ledger rows.
Never: `cc_solver.py cleanup` / `tau3_evolver_daytona.py cleanup`, any sandbox but the one created here (a live pool
machine is a -loop- sandbox; daytona_lib_evo refuses it by name).

SMOKE_REPORT.json, per unit: exit code (the bridge's rc and harness.json's), termination, agent seconds (the sum of
tau2's generation_time_seconds over the agent's messages = Claude Code's own time, its tool calls included; also
Claude Code's result duration_ms sum), tool calls (the harness's count; tau2's; ids from _meta = toolu_; by name from
Claude Code's stream and from tau2's trajectory: KB_search_bm25 / KB_search_dense / shell / the domain tools -- counted,
never required: which retrieval tool the agent uses is its choice), harness.json's retrieval record, the
machine and host grades and their parity, the transcript's local path, the sha256 of the first message actually sent
(claude.stdin.jsonl) and whether it is the harness's FIRST_MESSAGE_TEMPLATE filled, the PASS checks (run.json
retrieval_variant alltools among them). Also `harness_settings`: what the uploaded harness does today (first
message, no --max-turns, empty customer llm args, agent-time limit, alltools through tau3_retrieval) --
informational; the smoke runs the harness as it is.

Exit: 0 = every unit passed every check and the machine is gone; 3 = a unit failed a check; 4 = refused (preflight,
machine, setup); 6 = crash / signal; 7 = a container wipe was not verified; 8 = the machine was not verified gone;
2 = usage.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(REPO), str(HERE), str(REPO / "scripts" / "cc_solver"), str(REPO / "scripts" / "tau3_cc")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cc_solver as CS  # noqa: E402  (READ-ONLY reuse: Machine.sh/ctl, Box.prepare/enter/token_cmd, read_token)
import cc_solver_lib as CL  # noqa: E402  (READ-ONLY reuse: container_spec, limits_for, parse_stream)
import evolver_container_ctl as ctl  # noqa: E402  (READ-ONLY reuse: the machine-side paths and pins)
import openrouter_key as ORK  # noqa: E402  (the OpenRouter key: environment, else the repository's .env; stdlib)
import tau2_cc_harness as H  # noqa: E402  (the bridge itself: CC_ENV, TAU2_ENV, JOB_DEFAULTS, load_job; stdlib)
import tau3_retrieval as TR  # noqa: E402  (the bridge's alltools module: VARIANT, shell deny list, cache; stdlib)
from src.orchestrator import cc_account  # noqa: E402  (READ-ONLY reuse: which ONE credential a session uses)

NAME = "tau3_cc_smoke"
TAG = "tau3-cc-smoke"
ROLE = "tau3-smoke"
SCHEMA = "evoskill.tau3_cc_smoke/1"
NAME_PART = "tau3smoke-"                     # sandbox name = daytona_lib_evo.PREFIX + this + HHMMSS

DOCKERFILE = HERE / "Dockerfile.tau3_leg"
BASE_DOCKERFILE = HERE / "Dockerfile.ssb_leg"
CONTEXT_DIR = "tau3_ctx"
TAU3_REPO = REPO / "third_party" / "tau3-bench"
TAU3_COMMIT = "363133ada1936491fb5bcec33cd62c3518a99f65"
ARCHIVE_PATHS = ("pyproject.toml", "uv.lock", "README.md", "LICENSE", ".python-version", "src")
ENV_DIR = REPO / "scripts" / "tau3_env"
UVLOCK_PATCH = ENV_DIR / "uvlock_1.0.1.patch"
FREEZE = ENV_DIR / "tau3_venv_freeze.txt"
CHECK_VENV = REPO / "scripts" / "tau3_check_venv.py"
UVLOCK_SHA_COMMIT = "62d3a8c4807b89e85703b3c03f2c21048a2da9736ca83d9af6f61adc74ac5314"
UVLOCK_SHA_PATCHED = "b7c738079f883646b50d93f72caaa256dee5cf7e130baa68cb460cfc43e18e14"

SPLIT = REPO / "examples" / "tau3_banking" / "split.json"
GRADER = REPO / "examples" / "tau3_banking" / "grader.py"   # the user's tau3 grader (the reported runs' bytes)
# the bridge = every file the harness imports from its own dir: tau2_cc_harness.py, tau2_bundle.py, tau3_retrieval.py
BRIDGE_FILES = (Path(H.__file__).resolve(), Path(H.B.__file__).resolve(), Path(TR.__file__).resolve())
OUT_ROOT = REPO / ".cache" / "tau3_cc_smoke"
TAU3_PY_MAC = TAU3_REPO / ".venv" / "bin" / "python"   # the host's tau3-bench venv (the host-side grade)
CC_LOCAL = REPO / ".cache" / "claude_code" / ctl.CC_VERSION / "linux-x64" / "claude"
DOMAIN, VARIANT = "banking_knowledge", TR.VARIANT    # alltools
MODEL = "claude-sonnet-5-5"                  # the harness's default too

VM_BASE = "/opt/tau3smoke"                   # root 0700: invisible to every container (their /opt = claude-code)
VM_BIN = f"{VM_BASE}/bin"
VM_UNITS = f"{VM_BASE}/units"
VM_HARNESS = f"{VM_BIN}/tau2_cc_harness.py"
VM_GRADER = f"{VM_BIN}/{GRADER.name}"
VM_SPLIT = f"{VM_BASE}/split.json"
VM_DATA_TGZ = f"{VM_BASE}/data.tgz"
VM_TAU3_PY = "/opt/tau3/venv/bin/python"
VM_CHECK = "/opt/tau3/check/tau3_check_venv.py"
VM_PIN = "/opt/tau3/check/tau3_venv_freeze.txt"
# the bridge's and the machine grader's PATH, set in their command lines (Debian's root default): srt (/usr/local/bin),
# rg, bwrap, socat and node (/usr/bin) for tau2's KB shell, whatever PATH the SDK exec carries
VM_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
HOST_TOOLS = ("srt", "rg", "node")           # what the host-side grade's PATH must reach (host_tool_path)
# the KB shell's sandbox in the image (Dockerfile.tau3_leg step 9b): tau2's pin of Anthropic's sandbox-runtime, the
# host's `npm install -g` of it copied into the build context file by file (default_srt_src)
SRT_SRC_ENV = "EVOSKILL_SRT_SRC"
SRT_NAME, SRT_VERSION = "@anthropic-ai/sandbox-runtime", "0.0.23"
SRT_CTX = "srt/sandbox-runtime"              # below tau3_ctx/
VM_SRT = "/usr/local/lib/node_modules/@anthropic-ai/sandbox-runtime"
COPY_SOURCES = (f"{CONTEXT_DIR}/tau2-bench", f"{CONTEXT_DIR}/check", f"{CONTEXT_DIR}/{SRT_CTX}")
CONTEXT_MAX_BYTES = 32 << 20                 # the three trees: ~3 MB of tau2-bench + ~16 MB of srt
TOKEN_FD, KEY_FD = 3, 4                      # evoctl --keep-fd3 keeps exactly fd 3: the credential must be there
TOK_ENV, KEY_ENV = "CLAUDE_CODE_OAUTH_TOKEN", "TAU3_OR_KEY"   # the launch exec's carriers (unset before any process)
# the ONE Claude credential: a subscription token file, else an Anthropic API key (src/orchestrator/cc_account.py)
TOKEN_FILE_ENV, API_KEY_ENV = cc_account.TOKEN_FILE_ENV, cc_account.API_KEY_ENV
TOKEN_PREFIX = "sk-ant-" + "oat01-"          # a subscription token; spelled in pieces: no scanner matches the source
API_KEY_PREFIX = "sk-ant-" + "api"           # an Anthropic API key
OR_PREFIX = "sk-or-" + "v1-"
KEY_ENV_NAMES = ("OPENROUTER_API_KEY", "OR_API_KEY", "OPENAI_API_KEY")
POLL_S = 15
CUSTOMER_ALLOWANCE_S = 2400                  # hard stop = agent_time_s + this (customer calls, starts, graces)
HARD_STOP_MARGIN_S = 600                     # (a harness with a whole-conversation wall_s: wall + this)
STOP_GRACE_S = 120                           # after the bridge's SIGTERM, it has this long to write its rc
DEAD_GRACE_S = 60                            # a bridge pid gone with no rc for this long: collected as it is
POLL_FAIL_STOP = 40                          # polls in a row without an answer (~10 min): the units are given up
DEFAULT_TTL_MIN = 150
SETUP_AND_COLLECT_S = 45 * 60                # the TTL must hold the hard stop + this
SMALL_RESOURCES = {"cpu": 4, "memory": 16, "disk": 40}
CONTAINER_PARTS = (ctl.VM_WORK, f"{ctl.EVO_HOME}/.claude")
# job keys the driver owns (a --job-set may not touch them): the paths, the fds, the container and claude's command
OWNED_JOB_KEYS = ("task_file", "workdir", "out_dir", "home", "task_id", "domain", "retrieval_variant", "container",
                  "claude_cmd", "token_fd", "token_kind", "key_fd")
q = shlex.quote


class Refuse(RuntimeError):
    """The smoke cannot go on as asked (rc 4)."""


# ───────────────────────────────────────────────────────────────────────────── secrets and host tools
def token_kind(kind: str | None = None) -> str:
    """The harness's job "token_kind" for a credential kind (cc_account.TOKEN_FILE / API_KEY; by default this
    process's ONE credential, as cc_solver.read_credential recorded it: cc_solver_lib.credential_kind): "oauth" for a
    subscription token, "api_key" for an Anthropic API key."""
    return "api_key" if (kind or CL.credential_kind()) == cc_account.API_KEY else "oauth"


def token_prefix(kind: str | None = None) -> str:
    """The well-known prefix of the credential's kind: a scan needle besides the credential itself."""
    return API_KEY_PREFIX if token_kind(kind) == "api_key" else TOKEN_PREFIX


def credential_source(token_file=None) -> dict:
    """Where the ONE Claude credential comes from, nothing read (cc_account.resolve: the token file given, else
    EVOSKILL_CC_TOKEN_FILE, else the environment's ANTHROPIC_API_KEY). No secret in the result; "kind" is the
    harness's token_kind."""
    try:
        c = cc_account.resolve(token_file, check=False)
    except ValueError as e:
        return {"kind": None, "source": None, "token_file": None, "error": str(e)}
    return {"kind": token_kind(c.kind), "source": c.source, "token_file": str(c.token_file) if c.token_file else None}


def openrouter_key() -> str:
    """The customer's OpenRouter key (scripts/tau3_cc/openrouter_key.py: the environment, else the repository's
    .env), into memory only."""
    try:
        return ORK.openrouter_key()
    except ORK.KeyMissing as e:
        raise Refuse(f"the customer's OpenRouter key: {e}") from None


def default_srt_src() -> Path | None:
    """The installed sandbox-runtime tree the image copies: EVOSKILL_SRT_SRC, else the npm package the `srt` on PATH
    belongs to (`npm install -g @anthropic-ai/sandbox-runtime@0.0.23` links <prefix>/bin/srt to the package's
    dist/cli.js). None when neither is there; srt_files checks the package's name and version."""
    env = (os.environ.get(SRT_SRC_ENV) or "").strip()
    if env:
        return Path(env).expanduser()
    w = shutil.which("srt")
    if w:
        for d in Path(w).resolve().parents:
            if (d / "package.json").is_file():
                return d
    return None


def host_tool_path() -> str:
    """The host-side grade's PATH: the directories where this process's PATH finds srt, rg and node (an alltools
    environment builds tau2's KB shell, whose constructor requires srt and rg), then /usr/bin:/bin."""
    dirs: list = []
    for b in HOST_TOOLS:
        w = shutil.which(b)
        if w and str(Path(w).parent) not in dirs:
            dirs.append(str(Path(w).parent))
    return os.pathsep.join(dirs + [d for d in ("/usr/bin", "/bin") if d not in dirs])


# ───────────────────────────────────────────────────────────────────────────── small helpers (pure)
def now_utc() -> str:
    return CL.now_utc()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p) -> str:
    return CL.sha256_file(p)


def jdump(path, obj) -> None:
    CL.jdump(path, obj)


def parse_tail(out: str, tag: str) -> dict:
    """The JSON after the last line starting with `tag ` ({} when absent or unreadable)."""
    for ln in reversed((out or "").splitlines()):
        if ln.startswith(tag + " "):
            try:
                v = json.loads(ln[len(tag) + 1:])
                return v if isinstance(v, dict) else {}
            except ValueError:
                return {}
    return {}


class Redactor:
    """Every text that came from the machine passes through this before it is printed or written by the driver."""

    PATTERNS = (re.compile(re.escape(TOKEN_PREFIX) + r"[A-Za-z0-9_\-]+"),
                re.compile(re.escape(API_KEY_PREFIX) + r"[A-Za-z0-9_\-]+"),
                re.compile(re.escape(OR_PREFIX) + r"[A-Za-z0-9]+"))

    def __init__(self):
        self.secrets: list = []

    def add(self, s: str | None, label: str) -> None:
        if s:
            self.secrets.append((s, f"[{label}]"))

    def __call__(self, text) -> str:
        s = text if isinstance(text, str) else str(text)
        for v, lab in self.secrets:
            s = s.replace(v, lab)
        for p in self.PATTERNS:
            s = p.sub("[secret-shaped]", s)
        return s


def secret_scan(paths, needles: dict) -> dict:
    """{hits: ['<file>:<label>'], files}: raw bytes of every regular file below `paths` (no link followed); a hit's
    bytes are never returned. `needles` = {label: str|bytes}."""
    nd = {k: (v.encode() if isinstance(v, str) else v) for k, v in needles.items() if v}
    hits, n = [], 0
    for base in paths:
        base = Path(base)
        if not base.exists():
            continue
        for p in sorted([base] if base.is_file() else base.rglob("*")):
            if p.is_symlink() or not p.is_file():
                continue
            n += 1
            try:
                data = p.read_bytes()
            except OSError:
                continue
            for label, v in nd.items():
                if v in data:
                    hits.append(f"{p}:{label}")
    return {"hits": hits, "files": n}


def safe_extract(tgz: Path, dest: Path) -> list:
    """Extract regular files and dirs only; an absolute name, a '..' or any other member type is refused (listed)."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    refused = []
    with tarfile.open(tgz, "r:gz") as tf:
        for m in tf.getmembers():
            if m.name.startswith("/") or ".." in Path(m.name).parts or not (m.isfile() or m.isdir()):
                refused.append(m.name)
                continue
            tf.extract(m, dest, filter="data")
    return refused


# ───────────────────────────────────────────────────────────────────────────── the image and its build context
def apply_patch(text: str, patch: str, target: str) -> str:
    """A unified diff for ONE file (`target`) applied to `text`: every hunk's old lines must be at its position."""
    lines = text.split("\n")
    heads = [ln for ln in patch.splitlines() if ln.startswith(("--- ", "+++ "))]
    if heads != [f"--- a/{target}", f"+++ b/{target}"]:
        raise Refuse(f"the patch is not one diff of {target}: {heads}")
    hunks, cur = [], None
    for ln in patch.splitlines():
        m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", ln)
        if m:
            cur = {"old_start": int(m.group(1)), "lines": []}
            hunks.append(cur)
        elif cur is not None and ln[:1] in (" ", "-", "+"):
            cur["lines"].append(ln)
        elif cur is not None and ln.startswith("\\"):
            raise Refuse("the patch touches a file end without a newline (not handled here)")
    if not hunks:
        raise Refuse("the patch has no hunk")
    for h in reversed(hunks):
        old = [x[1:] for x in h["lines"] if x[0] in " -"]
        new = [x[1:] for x in h["lines"] if x[0] in " +"]
        i = h["old_start"] - 1
        if lines[i:i + len(old)] != old:
            raise Refuse(f"the patch hunk at line {h['old_start']} does not apply")
        lines[i:i + len(old)] = new
    return "\n".join(lines)


def tree_sha256(files: dict, prefix: str = "") -> str:
    """sha256 over the sorted (path relative to `prefix`, octal mode, sha256 of the bytes) lines of `files`
    ({path: (bytes, mode)}, only the paths under `prefix`): the build context's identity, independent of mtimes."""
    h = hashlib.sha256()
    for rel in sorted(files):
        if prefix and not rel.startswith(prefix):
            continue
        data, mode = files[rel]
        h.update(f"{rel[len(prefix):]}\0{mode:o}\0{sha256_bytes(data)}\n".encode())
    return h.hexdigest()


def srt_files(src: Path | None = None) -> tuple:
    """({srt/sandbox-runtime/<rel>: (bytes, mode)}, meta): the host's installed sandbox-runtime (tau2's pin, 0.0.23;
    default_srt_src when src is None), file by file with its modes (0755 when any exec bit is set, else 0644:
    dist/cli.js and the vendored apply-seccomp binaries). Refused: a missing tree, any symlink in it (a dir or a
    file), anything but a regular file, a package other than @anthropic-ai/sandbox-runtime 0.0.23, dist/cli.js or an
    apply-seccomp without its exec bit (or no apply-seccomp at all), anything named claude (the image's step 10
    refuses it). Nothing is written."""
    src = src if src is not None else default_srt_src()
    if src is None:
        raise Refuse(f"no installed sandbox-runtime: set {SRT_SRC_ENV} to its package dir, or put srt on PATH "
                     f"(npm install -g {SRT_NAME}@{SRT_VERSION})")
    src = Path(src)
    if src.is_symlink() or not src.is_dir():
        raise Refuse(f"{src} is not the installed sandbox-runtime tree "
                     f"(npm install -g {SRT_NAME}@{SRT_VERSION} puts it there)")
    try:
        pkg = json.loads((src / "package.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise Refuse(f"{src}/package.json: {type(e).__name__}: {e}") from None
    if (pkg.get("name"), pkg.get("version")) != (SRT_NAME, SRT_VERSION):
        raise Refuse(f"{src} is {pkg.get('name')} {pkg.get('version')}, not {SRT_NAME} {SRT_VERSION} (tau2's pin: "
                     f"0.0.24+ leaves file stubs behind on Linux)")
    files: dict = {}
    for dirpath, dirnames, filenames in os.walk(src, followlinks=False):
        d = Path(dirpath)
        for n in sorted(dirnames + filenames):
            rel = (d / n).relative_to(src).as_posix()
            if (d / n).is_symlink():
                raise Refuse(f"{SRT_NAME}: {rel} is a symlink (the context copies regular files only)")
            if n == "claude":
                raise Refuse(f"{SRT_NAME}: {rel} is named claude (the image allows no such file)")
        for n in sorted(filenames):
            p = d / n
            rel = p.relative_to(src).as_posix()
            st = os.lstat(p)
            if not stat.S_ISREG(st.st_mode):
                raise Refuse(f"{SRT_NAME}: {rel} is not a regular file")
            files[f"{SRT_CTX}/{rel}"] = (p.read_bytes(), 0o755 if st.st_mode & 0o111 else 0o644)
    seccomp = sorted(k for k in files if k.endswith("/apply-seccomp"))
    if not seccomp:
        raise Refuse(f"{SRT_NAME}: no vendored apply-seccomp binary (srt's Linux sandbox needs it)")
    need = [f"{SRT_CTX}/dist/cli.js", *seccomp]
    bad = [k for k in need if k not in files or files[k][1] != 0o755]
    if bad:
        raise Refuse(f"{SRT_NAME}: not executable: {bad}")
    meta = {"source": str(src), "name": SRT_NAME, "version": SRT_VERSION, "n_files": len(files),
            "bytes": sum(len(b) for b, _ in files.values()), "tree_sha256": tree_sha256(files, SRT_CTX + "/"),
            "executables": sorted(k[len(SRT_CTX) + 1:] for k, (_, m) in files.items() if m == 0o755),
            "image_path": VM_SRT}
    return files, meta


def context_files(repo: Path = TAU3_REPO, commit: str = TAU3_COMMIT, srt: Path | None = None) -> tuple:
    """({path below tau3_ctx/: (bytes, mode)}, meta): the tau2-bench sources at the pinned commit (git archive, read
    only), uv.lock patched and sha-checked both ways, the venv check + its pin, and the KB shell's sandbox-runtime
    tree (srt_files). Nothing is written; the same inputs give the same context_sha256."""
    r = subprocess.run(["git", "--no-optional-locks", "-C", str(repo), "archive", "--format=tar", commit, "--",
                        *ARCHIVE_PATHS], capture_output=True)
    if r.returncode != 0:
        raise Refuse(f"git archive {commit[:7]} in {repo}: {r.stderr.decode(errors='replace').strip()[:400]}")
    files: dict = {}
    with tarfile.open(fileobj=io.BytesIO(r.stdout), mode="r:") as tf:
        for m in tf.getmembers():
            if m.isdir() or m.type in (tarfile.XGLTYPE, tarfile.XHDTYPE):
                continue
            if not m.isfile() or m.name.startswith("/") or ".." in Path(m.name).parts:
                raise Refuse(f"git archive member {m.name!r} is not a plain file")
            files[f"tau2-bench/{m.name}"] = (tf.extractfile(m).read(), 0o755 if m.mode & 0o111 else 0o644)
    lock_key = "tau2-bench/uv.lock"
    if lock_key not in files or "tau2-bench/src/tau2/__init__.py" not in files:
        raise Refuse("git archive lacks uv.lock or src/tau2")
    lock = files[lock_key][0]
    if sha256_bytes(lock) != UVLOCK_SHA_COMMIT:
        raise Refuse(f"uv.lock at {commit[:7]} hashes {sha256_bytes(lock)[:12]}, "
                     f"not the pinned {UVLOCK_SHA_COMMIT[:12]}")
    patched = apply_patch(lock.decode("utf-8"), UVLOCK_PATCH.read_text(encoding="utf-8"), "uv.lock").encode("utf-8")
    if sha256_bytes(patched) != UVLOCK_SHA_PATCHED:
        raise Refuse(f"patched uv.lock hashes {sha256_bytes(patched)[:12]}, not {UVLOCK_SHA_PATCHED[:12]} (scripts/tau3_env)")
    files[lock_key] = (patched, files[lock_key][1])
    files["check/tau3_check_venv.py"] = (CHECK_VENV.read_bytes(), 0o644)
    files["check/tau3_venv_freeze.txt"] = (FREEZE.read_bytes(), 0o644)
    sfiles, smeta = srt_files(srt)
    files.update(sfiles)
    meta = {"commit": commit, "archive_paths": list(ARCHIVE_PATHS), "n_files": len(files),
            "bytes": sum(len(d) for d, _ in files.values()), "context_sha256": tree_sha256(files),
            "uv_lock_sha256": {"commit": UVLOCK_SHA_COMMIT, "patched": sha256_bytes(patched)},
            "patch_sha256": sha256_file(UVLOCK_PATCH), "freeze_sha256": sha256_file(FREEZE),
            "check_venv_sha256": sha256_file(CHECK_VENV), "srt": smeta}
    if meta["bytes"] > CONTEXT_MAX_BYTES:
        raise Refuse(f"the build context is {meta['bytes']} bytes (> {CONTEXT_MAX_BYTES}): something extra came in")
    return files, meta


def dockerfile_meta() -> dict:
    base, full = BASE_DOCKERFILE.read_bytes(), DOCKERFILE.read_bytes()
    if not full.startswith(base):
        raise Refuse(f"{DOCKERFILE.name} does not start with the bytes of {BASE_DOCKERFILE.name}")
    copies = [ln.split()[1:] for ln in full.decode().splitlines() if re.match(r"^\s*COPY\s", ln, re.I)]
    srcs = [c[0] for c in copies]
    if srcs != list(COPY_SOURCES):
        raise Refuse(f"{DOCKERFILE.name}: COPY sources {srcs}, want {list(COPY_SOURCES)}")
    if copies[2][1:] != [VM_SRT]:
        raise Refuse(f"{DOCKERFILE.name}: sandbox-runtime is copied to {copies[2][1:]}, not {VM_SRT} (outside /opt: "
                     f"the KB shell denies reads of /opt and srt runs its apply-seccomp inside the sandbox)")
    return {"dockerfile": str(DOCKERFILE), "dockerfile_sha256": sha256_bytes(full),
            "base_dockerfile_sha256": sha256_bytes(base), "copy_sources": srcs}


def write_build_dir(dest: Path) -> dict:
    """dest/Dockerfile.tau3_leg (a copy) + dest/tau3_ctx/...: what Image.from_dockerfile(dest/Dockerfile.tau3_leg)
    uploads. dest must be absent or empty."""
    files, meta = context_files()
    meta.update(dockerfile_meta())
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        raise Refuse(f"{dest} is not empty")
    dest.mkdir(parents=True, exist_ok=True)
    df = dest / DOCKERFILE.name
    shutil.copyfile(DOCKERFILE, df)
    for rel, (data, mode) in sorted(files.items()):
        p = dest / CONTEXT_DIR / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        os.chmod(p, mode)
    meta.update(build_dir=str(dest), build_dockerfile=str(df))
    return meta


# ───────────────────────────────────────────────────────────────────────────── tasks, units, jobs, commands
def load_split(path: Path = SPLIT) -> dict:
    d = json.loads(Path(path).read_text())
    if d.get("domain") != DOMAIN or not d.get("train"):
        raise Refuse(f"{path}: not a {DOMAIN} split with a train list")
    return d


def task_file(task: str, root: Path) -> Path:
    return Path(root) / "tau2" / "domains" / DOMAIN / "tasks" / f"{task}.json"


def pick_tasks(split: dict, tasks: str | None, n: int, root: Path) -> list:
    """--tasks (comma list) or the first n of split['train'] in file order. Train tasks only; a dropped task, a task
    of another set, or a missing file is refused."""
    train = list(split.get("train") or [])
    others = {k: set(split.get(k) or []) for k in ("val", "test")}
    others["dropped"] = set(((split.get("dropped") or {}).get("task_ids")) or [])
    want = [t.strip() for t in tasks.split(",") if t.strip()] if tasks else train[:max(1, int(n))]
    if not want or len(set(want)) != len(want):
        raise Refuse(f"tasks {want}: empty or repeated")
    for t in want:
        if t not in train:
            raise Refuse(f"{t} is not in split['train'] (train tasks only)")
        hit = [k for k, s in others.items() if t in s]
        if hit:
            raise Refuse(f"{t} is also in {hit}")
        if not task_file(t, root).is_file():
            raise Refuse(f"{task_file(t, root)} is missing")
    return want


def unit_name(task: str, sample: int) -> str:
    return f"{task}_s{int(sample)}"


def make_cid(task: str, sample: int, stamp: str, salt: str | None = None) -> str:
    base = re.sub(r"[^a-z0-9_.-]", "-", f"{task}-s{int(sample)}".lower())[:30]
    cid = f"cc-tau3-{base}-{stamp.lower()[-6:]}-{salt or uuid.uuid4().hex[:4]}"[:64]
    if not ctl.CID_RE.match(cid):
        raise Refuse(f"bad container id {cid!r}")
    return cid


def container_field(cid: str, *, evoctl=(ctl.PY, ctl.CTL)) -> dict:
    """The job's "container": the harness itself builds `evoctl enter --cid C --cwd /evo/work --keep-fd3 --
    /usr/bin/env HOME=... <CC_ENV> <TAU2_ENV> <the credential's fd variable>=3 claude <stream argv>`
    (evoctl starts ARGV with `env -i` + its EVO_ENV, so claude's variables travel in the argv; one definition, the
    harness's)."""
    return {"evoctl": list(evoctl), "cid": cid, "cwd": ctl.VM_WORK}


def parse_job_sets(items) -> dict:
    """--job-set KEY=JSON (a value that is not JSON is taken as a string); only the harness's own job keys, never
    one the driver owns."""
    out = {}
    for it in items or []:
        k, sep, v = str(it).partition("=")
        if not sep or not k:
            raise Refuse(f"--job-set {it!r}: want KEY=VALUE")
        if k in OWNED_JOB_KEYS or k not in H.JOB_DEFAULTS:
            raise Refuse(f"--job-set {k}: not a harness job key this driver leaves open "
                         f"({sorted(set(H.JOB_DEFAULTS) - set(OWNED_JOB_KEYS))})")
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    return out


def unit_job(task: str, sample: int, cid: str, vm_unit: str, *, root: Path, model: str = MODEL, overrides=None,
             token_kind: str = "oauth") -> dict:
    """One rollout's job.json (no secret in it; token_kind names what fd 3 carries, token_kind() of the
    credential). Every key not set here is the harness's own default."""
    job = {"task_file": str(task_file(task, root)), "workdir": f"{vm_unit}/snapshot", "out_dir": f"{vm_unit}/out",
           "home": ctl.EVO_HOME, "task_id": f"{DOMAIN}/{task}", "domain": DOMAIN, "retrieval_variant": VARIANT,
           "sample": int(sample), "model": model, "container": container_field(cid), "token_fd": TOKEN_FD,
           "token_kind": token_kind, "key_fd": KEY_FD}
    job.update(dict(overrides or {}))
    unknown = sorted(set(job) - set(H.JOB_REQUIRED) - set(H.JOB_DEFAULTS))
    if unknown:
        raise Refuse(f"the harness does not know the job keys {unknown} (its file changed?)")
    return job


def hard_stop_s(job: dict, override_min=None) -> float:
    """When the driver stops a rollout that has not ended by itself: the harness's only limit is the agent clock
    (agent_time_s; the customer's time never counts), so the job's agent time + CUSTOMER_ALLOWANCE_S."""
    if override_min:
        return float(override_min) * 60.0
    if job.get("agent_time_s"):
        return float(job["agent_time_s"]) + CUSTOMER_ALLOWANCE_S
    if job.get("wall_s"):
        return float(job["wall_s"]) + HARD_STOP_MARGIN_S
    return 3600.0


def validate_job(job: dict) -> dict:
    """The harness's own load_job on the job's bytes (in a temporary directory, removed at once)."""
    with tempfile.TemporaryDirectory(prefix="tau3smoke_job_") as d:
        p = Path(d) / "job.json"
        p.write_text(json.dumps(job, indent=1))
        try:
            return H.load_job(p)
        except H.JobError as e:
            raise Refuse(f"the harness refuses the job: {e}") from None


def launch_cmd(vm_unit: str, tok: str, key: str, *, py: str = VM_TAU3_PY, harness: str = VM_HARNESS,
               detached: bool = True, path: str = VM_PATH) -> str:
    """The bridge's launch: the secrets leave the exec's environment before any process of ours starts and reach the
    bridge only as fd 3 (token) and fd 4 (key), here-string pipes (bash >= 5.1). Refused if either is in the string.
    PATH is set to `path` (VM_PATH: srt in /usr/local/bin, rg / bwrap / socat / node in /usr/bin, which tau2's KB
    shell looks up on PATH; a local test passes its own)."""
    inner = "\n".join([
        f"cd {q(vm_unit)} || exit 1",
        f"{q(py)} {q(harness)} --job job.json >bridge.log 2>&1 &",
        "echo $! >bridge.pid",
        "wait $!",
        "echo $? >rc.tmp && mv rc.tmp rc"])
    base = (f"env -u {TOK_ENV} -u {KEY_ENV} PATH={q(path)} PYTHON_DOTENV_DISABLED=1 LITELLM_LOCAL_MODEL_COST_MAP=True "
            f"/bin/bash -c {q(inner)}")
    redir = f' 3<<<"${TOK_ENV}" 4<<<"${KEY_ENV}"'
    cmd = (f"(setsid nohup {base} </dev/null >/dev/null 2>&1{redir} &)" if detached
           else f"{base} </dev/null >/dev/null 2>&1{redir}")
    for s in (tok, key):
        if not s or s in cmd:
            raise Refuse("a secret is empty or would appear in the launch command")
    return cmd


POLL_PY = r'''
import json, os, sys
out = {}
for d in sys.argv[1:]:
    r = {}
    try:
        r["rc"] = open(os.path.join(d, "rc")).read().strip()
    except OSError:
        r["rc"] = None
    try:
        pid = int(open(os.path.join(d, "bridge.pid")).read().strip())
    except (OSError, ValueError):
        pid = None
    r["pid"] = pid
    r["alive"] = bool(pid) and os.path.exists("/proc/%d" % pid)
    for k, n in (("stream_bytes", "out/claude.stream.jsonl"), ("stdin_bytes", "out/claude.stdin.jsonl"),
                 ("log_bytes", "bridge.log")):
        try:
            r[k] = os.path.getsize(os.path.join(d, n))
        except OSError:
            r[k] = None
    try:
        with open(os.path.join(d, "out", "claude.stdin.jsonl"), "rb") as fh:
            r["stdin_lines"] = fh.read().count(b"\n")
    except OSError:
        r["stdin_lines"] = None
    try:
        with open(os.path.join(d, "bridge.log"), "rb") as fh:
            fh.seek(max(0, (r["log_bytes"] or 0) - 600))
            r["log_tail"] = fh.read().decode("utf-8", "replace")
    except OSError:
        r["log_tail"] = ""
    out[os.path.basename(d)] = r
print("POLL " + json.dumps(out))
'''

GRADE_PY = f"DOMAIN, VARIANT = {DOMAIN!r}, {VARIANT!r}\n" + r'''
import importlib.util, json, sys
from pathlib import Path
from types import SimpleNamespace
g, task_file, snap = sys.argv[1:4]
spec = importlib.util.spec_from_file_location("benchmark_grader", g)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
task = SimpleNamespace(answer_files=[Path(task_file)], grading={"domain": DOMAIN, "retrieval_variant": VARIANT})
try:
    v = mod.grade(task, Path(snap))
    out = {"score": v, "type": type(v).__name__}
except Exception as e:
    out = {"raised": type(e).__name__, "message": str(e)[:600]}
print("GRADE_JSON " + json.dumps(out))
'''

# The tau3 machine check (this smoke's step 4 and cc_pool_host.CCMachine.tau3_check, run as root under /opt/tau3/venv
# with TAU2_DATA_DIR = the data root and no key in the environment). The bridge files are not on the machine yet when
# it runs, so tau3_retrieval's constants are written into the text here and its shell hardening is redone the same
# way (the srt settings JSON's filesystem.denyRead += the data root, /opt, /root, /home, /Users).
#   - always: tau2 version, data dir, task count, litellm / httpx versions, OpenRouter's key-less /models (the
#     callers judge these from the TAU3_CHECK line, as before);
#   - (a) sandbox: tau2's own _required_binaries() for this platform (linux: srt rg bwrap socat; darwin: srt rg) +
#     node (srt's interpreter) found on PATH, their versions, srt's package.json version == 0.0.23;
#   - (b) embeddings: tau2's EmbeddingsCache.get(("openai", {"model": "text-embedding-3-large"})) over the KB's
#     documents as tau2's dense pipeline hands them over (retrieval.get_or_create_docs -> {id, text}) loads an entry
#     of shape [n_docs, 3072] with the KB's document ids -- from a COPY of <root>/.embeddings_cache (tau2's get()
#     deletes an entry it finds stale; the data root is never written);
#   - (c) shell: a KB sandbox built as tau2 builds alltools' (retrieval._create_sandbox), hardened: `ls | head -2`
#     lists KB documents; `cd / && cat <root>/tau2/domains/banking_knowledge/tasks/task_004.json | head -c 80` and
#     `cd / && ls opt | head -3` print nothing; the CONTROL -- the same cat in a sandbox with tau2's own settings --
#     prints the task's JSON (so an empty answer above is the deny list's doing, not a broken sandbox);
#     `cd / && ls proc | head` recorded. Outputs are recorded by length / sha / a short head (the control's by
#     length and shape only: it is a task file).
# (a), (b) or (c) failing prints `TAU3_CHECK_FAIL {reasons, ...}` and NO TAU3_CHECK line, exit 1: a caller that
# reads TAU3_CHECK (cc_pool_host) then refuses the machine.
TAU3_CHECK_PY = (f"DOMAIN, VARIANT, PROBE_TASK = {DOMAIN!r}, {VARIANT!r}, 'task_004'\n"
                 f"DENY_FIXED, CACHE_DIRNAME = {list(TR.DENY_READ_FIXED)!r}, {TR.CACHE_DIRNAME!r}\n"
                 f"EMBEDDER_TYPE, EMBEDDER_MODEL, EMBED_DIM = "
                 f"{TR.EMBEDDER_TYPE!r}, {TR.TAU2_MODEL!r}, {TR.EMBED_DIM!r}\n"
                 f"SRT_VERSION = {SRT_VERSION!r}\n") + r"""
import hashlib, importlib.metadata as md, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
PATH_INHERITED = os.environ.get("PATH")    # what an exec without an explicit PATH (the in-place grader's) gets
_path = [x for x in (PATH_INHERITED or "").split(os.pathsep) if x]
os.environ["PATH"] = os.pathsep.join(_path + [d for d in ("/usr/local/bin", "/usr/bin", "/bin") if d not in _path])
out, fail = {}, []
try:
    import tau2, litellm, rank_bm25
    from tau2.utils.utils import DATA_DIR, get_tau2_version
    from tau2.domains.banking_knowledge.environment import get_tasks
    out.update(tau2_version=get_tau2_version(), data_dir=str(DATA_DIR), n_tasks=len(get_tasks()),
               litellm=md.version("litellm"), httpx=md.version("httpx"))
except Exception as e:
    out["import_error"] = ("%s: %s" % (type(e).__name__, e))[:600]
try:
    import httpx
    r = httpx.get("https://openrouter.ai/api/v1/models", timeout=30)
    out["openrouter_models_status"] = r.status_code
    try:
        ids = [m.get("id") for m in (r.json().get("data") or [])]
        out["openrouter_models_n"] = len(ids)
        out["openrouter_has_gpt_5_nano"] = "openai/gpt-5-nano" in ids
    except Exception as e:
        out["openrouter_models_parse"] = type(e).__name__
except Exception as e:
    out["openrouter_error"] = ("%s: %s" % (type(e).__name__, e))[:300]


def _err(e):
    return ("%s: %s" % (type(e).__name__, e))[:400]


def _first(argv):
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    except Exception as e:
        return {"error": _err(e)}
    lines = [x.strip() for x in ((r.stdout or "") + "\n" + (r.stderr or "")).splitlines() if x.strip()]
    pick = [x for x in lines if "version" in x.lower()] or lines
    return {"rc": r.returncode, "line": (pick[0] if pick else "")[:200]}


def _brief(text):
    t = text or ""
    return {"chars": len(t), "sha256": hashlib.sha256(t.encode()).hexdigest()[:16], "head": t[:120]}


# (a) the KB shell's binaries
try:
    from tau2.knowledge import sandbox_manager as SM
    req = list(SM._required_binaries())
    names = list(dict.fromkeys(req + ["node", "srt", "rg", "bwrap", "socat"]))
    which = {b: shutil.which(b) for b in names}
    argv = {"srt": ["--version"], "rg": ["--version"], "bwrap": ["--version"], "socat": ["-V"], "node": ["--version"]}
    vers = {b: _first([which[b]] + argv[b]) for b in names if which[b]}
    srt_pkg = None
    if which.get("srt"):
        pj = Path(os.path.realpath(which["srt"])).parent.parent / "package.json"
        try:
            srt_pkg = json.loads(pj.read_text()).get("version")
        except Exception as e:
            srt_pkg = "unreadable: " + _err(e)
    out["sandbox"] = {"platform": sys.platform, "required": req, "which": which, "versions": vers,
                      "srt_package_version": srt_pkg, "path": os.environ["PATH"], "path_inherited": PATH_INHERITED,
                      "inherited_path_finds": sorted(b for b in req + ["node"]
                                                     if shutil.which(b, path=PATH_INHERITED or "") == which.get(b))}
    missing = [b for b in req + ["node"] if not which.get(b)]
    if missing:
        fail.append("sandbox binaries missing on PATH: %s" % missing)
    if srt_pkg != SRT_VERSION:
        fail.append("srt package version %r, not %s" % (srt_pkg, SRT_VERSION))
except Exception as e:
    fail.append("sandbox binaries: " + _err(e))

# (b) the document embeddings, read-only
try:
    from tau2.domains.banking_knowledge import retrieval as R
    from tau2.domains.banking_knowledge.environment import get_knowledge_base
    from tau2.knowledge.embeddings_cache import EmbeddingsCache
    root = Path(DATA_DIR)
    kb = get_knowledge_base()
    docs = [{"id": d["id"], "text": d["text"]} for d in R.get_or_create_docs(kb)]
    src = root / CACHE_DIRNAME
    emb = {"dir": str(src), "documents": len(docs),
           "files": sorted(p.name for p in src.iterdir()) if src.is_dir() else None}
    with tempfile.TemporaryDirectory(prefix="tau3check_emb_") as tmp:
        cdir = Path(tmp) / CACHE_DIRNAME
        shutil.copytree(src, cdir)
        cache = EmbeddingsCache(str(cdir))
        cfg = {"model": EMBEDDER_MODEL}
        key = cache._get_cache_key(cache._compute_document_hash(docs), cache._compute_embedder_hash(EMBEDDER_TYPE, cfg))
        f = cdir / (key + ".pkl")
        got = None
        if f.is_file():                 # tau3_retrieval.in_kb_order's rule: the same ids in another order = re-ordered
            import pickle, numpy as np
            data = pickle.loads(f.read_bytes())
            ids, cached = [d["id"] for d in docs], list(data.get("doc_ids") or [])
            emb["cached_order_equals_kb_order"] = cached == ids
            if cached == ids:
                got = (data["embeddings"], cached)
            elif len(cached) == len(ids) and len(set(cached)) == len(cached) and set(cached) == set(ids):
                pos = {d: k for k, d in enumerate(cached)}
                got = (np.asarray(data["embeddings"])[[pos[d] for d in ids]], ids)
    emb.update(entry=key + ".pkl", entry_in_root=(src / (key + ".pkl")).is_file(), read_from_copy=True)
    if got is None:
        fail.append("no embeddings cache entry %s.pkl for %s/%s over the %d KB documents"
                    % (key, EMBEDDER_TYPE, EMBEDDER_MODEL, len(docs)))
    else:
        emb.update(shape=list(got[0].shape), dtype=str(got[0].dtype), ids_match=got[1] == [d["id"] for d in docs])
        if emb["shape"] != [len(docs), EMBED_DIM] or not emb["ids_match"]:
            fail.append("embeddings shape %s (want [%d, %d]), ids match %s"
                        % (emb["shape"], len(docs), EMBED_DIM, emb["ids_match"]))
    out["embeddings"] = emb
except Exception as e:
    fail.append("embeddings: " + _err(e))

# (c) the KB shell, hardened as tau3_retrieval hardens it, and its control
sbs = []
try:
    from tau2.domains.banking_knowledge import retrieval as R
    from tau2.domains.banking_knowledge.environment import get_knowledge_base
    from tau2.knowledge.sandbox_manager import SandboxManager
    root = Path(DATA_DIR)
    deny = list(dict.fromkeys([str(root.resolve()), str(root), *DENY_FIXED]))
    task_rel = "%s/tau2/domains/%s/tasks/%s.json" % (os.path.relpath(str(root.resolve()), "/"), DOMAIN, PROBE_TASK)
    spec = R.RETRIEVAL_VARIANTS[VARIANT].shell
    hard = R._create_sandbox(get_knowledge_base(), spec)
    sbs.append(hard)
    sp = Path(hard.settings_path)
    st = json.loads(sp.read_text())
    fs = st.setdefault("filesystem", {})
    fs["denyRead"] = list(dict.fromkeys([*fs.get("denyRead", []), *deny]))
    sp.write_text(json.dumps(st, indent=2))
    ctrl = SandboxManager(allow_writes=False)
    sbs.append(ctrl)

    def run(sb, cmd):
        rc, so, se = sb.run_command(cmd)
        return rc, so or "", se or ""
    kb_files = set(hard.list_files())
    rc, so, se = run(hard, "ls | head -2")
    ls_lines = so.split()
    pr = {"ls": {"rc": rc, "stdout": so[:200], "stderr": se[:200]}}
    rc, so, se = run(hard, "cd / && cat %s | head -c 80" % task_rel)
    pr["cat_task"] = {"rc": rc, "stdout": _brief(so), "stderr": se[:200]}
    cat_empty = so.strip() == ""
    rc, so, se = run(hard, "cd / && ls opt | head -3")
    pr["ls_opt"] = {"rc": rc, "stdout": _brief(so), "stderr": se[:200]}
    opt_empty = so.strip() == ""
    rc, so, se = run(hard, "cd / && ls proc | head")
    pr["ls_proc"] = {"rc": rc, "stdout": so[:200], "stderr": se[:200]}
    rc, so, se = run(ctrl, "cd / && cat %s | head -c 80" % task_rel)
    ctrl_json = so.lstrip().startswith("{") and ('"%s"' % PROBE_TASK) in so
    pr["control_cat_task"] = {"rc": rc, "stdout_chars": len(so), "starts_with_brace": so.lstrip().startswith("{"),
                              "names_the_task": ('"%s"' % PROBE_TASK) in so, "stderr": se[:200]}
    ok = {"ls_lists_kb_documents": bool(ls_lines) and all(x in kb_files for x in ls_lines),
          "task_file_unreadable": cat_empty, "opt_unreadable": opt_empty, "control_reads_the_task_file": ctrl_json}
    out["shell"] = {"variant": VARIANT, "deny_read_added": deny, "settings_deny_read": fs["denyRead"],
                    "kb_files": len(kb_files), "task_file": task_rel, "probes": pr, "ok": ok}
    bad = [k for k, v in ok.items() if not v]
    if bad:
        fail.append("KB shell probe: %s" % bad)
except Exception as e:
    fail.append("KB shell: " + _err(e))
finally:
    for sb in sbs:
        try:
            sb.cleanup()
        except Exception:
            pass
if fail:
    print("TAU3_CHECK_FAIL " + json.dumps(dict(out, reasons=fail)))
    sys.exit(1)
print("TAU3_CHECK " + json.dumps(out))
"""


def grade_vm_cmd(task_file_vm: str, snapshot_vm: str) -> str:
    return (f"cd / && env {' '.join('-u ' + k for k in KEY_ENV_NAMES)} PATH={q(VM_PATH)} PYTHON_DOTENV_DISABLED=1 "
            f"LITELLM_LOCAL_MODEL_COST_MAP=True {VM_TAU3_PY} -c {q(GRADE_PY)} {q(VM_GRADER)} {q(task_file_vm)} "
            f"{q(snapshot_vm)}")


def grade_local(task_file_local: Path, snapshot: Path, *, py: Path = TAU3_PY_MAC, grader: Path = GRADER,
                home: Path | None = None, timeout: int = 900) -> dict:
    """The user's grader on the host, in a CHILD under the tau3-bench venv with a clean environment (no key, no
    .env: PYTHON_DOTENV_DISABLED=1), one process per grade (tau2 reads its data root once). PATH holds the host's srt
    and rg (host_tool_path): an alltools environment builds tau2's KB shell, whose constructor requires them."""
    if not Path(py).is_file():
        return {"skipped": f"{py} is absent"}
    with tempfile.TemporaryDirectory(prefix="tau3smoke_grade_") as d:
        env = {"PATH": host_tool_path(), "HOME": str(home or d), "LC_ALL": "C.UTF-8",
               "PYTHON_DOTENV_DISABLED": "1",
               "LITELLM_LOCAL_MODEL_COST_MAP": "True", "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            r = subprocess.run([str(py), "-c", GRADE_PY, str(grader), str(task_file_local), str(snapshot)],
                               capture_output=True, text=True, env=env, cwd=d, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"raised": "TimeoutExpired", "message": f"no grade within {timeout} s"}
    res = parse_tail(r.stdout, "GRADE_JSON")
    if not res:
        res = {"raised": "NoOutput", "message": f"rc {r.returncode}: {(r.stderr or '')[-600:]}"}
    return res


def data_tgz(root: Path) -> bytes:
    """The data root as a gzip tar (sorted, root-owned, plain modes), MANIFEST.sha256 included."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", format=tarfile.PAX_FORMAT) as tf:
        for p in sorted(Path(root).rglob("*")):
            if p.is_symlink():
                raise Refuse(f"{p} is a symlink")
            ti = tf.gettarinfo(str(p), arcname=p.relative_to(root).as_posix())
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = "root"
            ti.mode = 0o755 if p.is_dir() else 0o644
            if p.is_file():
                with open(p, "rb") as fh:
                    tf.addfile(ti, fh)
            else:
                tf.addfile(ti)
    return buf.getvalue()


def verify_data_root(root: Path) -> dict:
    import importlib.util                      # READ-ONLY reuse of scripts/tau3_make_data_root.py: the root against
    spec = importlib.util.spec_from_file_location("tau3_make_data_root", REPO / "scripts" / "tau3_make_data_root.py")
    MDR = importlib.util.module_from_spec(spec)   # its own manifest (loaded by path: scripts/ never joins sys.path)
    spec.loader.exec_module(MDR)
    try:
        rows = MDR.verify_root(Path(root))
    except MDR.Refused as e:
        raise Refuse(f"data root: {e}") from None
    return {"root": str(root), "files": len(rows), "manifest_sha256": sha256_file(Path(root) / MDR.MANIFEST),
            "tasks": sum(1 for r in rows if r.startswith(f"tau2/domains/{DOMAIN}/tasks/"))}


# ───────────────────────────────────────────────────────────────────────────── reading a collected unit
def _jload(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def first_user_message(stdin_text: str):
    for ln in (stdin_text or "").splitlines():
        try:
            ev = json.loads(ln)
        except ValueError:
            continue
        if isinstance(ev, dict) and ev.get("type") == "user":
            c = (ev.get("message") or {}).get("content")
            if isinstance(c, list):
                c = "".join(x.get("text", "") for x in c if isinstance(x, dict))
            return c if isinstance(c, str) else None
    return None


def first_message_check(text, customer_first, template: str | None = None) -> dict:
    """Is the first message sent the first-message template (the harness's FIRST_MESSAGE_TEMPLATE by default) with
    {policy} and {customer} filled?"""
    if text is None:
        return {"sha256": None, "matches_template": None}
    tmpl = template if template is not None else H.FIRST_MESSAGE_TEMPLATE
    a, _, rest = tmpl.partition("{policy}")
    b, _, c = rest.partition("{customer}")
    t, c2 = text.rstrip("\n"), c.rstrip("\n")
    ok = t.startswith(a) and t.endswith(c2) and len(t) >= len(a) + len(c2)
    customer = None
    if ok:
        mid = t[len(a):len(t) - len(c2)] if c2 else t[len(a):]
        i = mid.rfind(b)
        ok = i >= 0
        customer = mid[i + len(b):] if ok else None
    return {"sha256": sha256_bytes(text.encode("utf-8")), "chars": len(text),
            "matches_template": bool(ok),
            "customer_part_is_the_customers_first_message": (customer == customer_first) if ok else None,
            "template_sha256": sha256_bytes(tmpl.encode("utf-8"))}


RETRIEVAL_TOOLS = ("KB_search_bm25", "KB_search_dense", "shell")    # alltools' three; the rest are the domain's
MCP_PREFIX = f"mcp__{H.SERVER}__"                                      # Claude Code's name for a tau2 tool


def tool_calls_by_name(stream_events: list, tau2_calls: list) -> dict:
    """Tool calls by name, from Claude Code's stream (its assistant events' tool_use blocks, once per id: the tau2
    tools under their tau2 names, Claude Code's own tools -- Skill, Bash, ... -- under theirs) and from tau2's
    trajectory; plus the retrieval split (KB_search_bm25 / KB_search_dense / shell / the domain's tools). Counted
    only: which retrieval tool the agent uses is its choice."""
    seen: dict = {}
    for e in stream_events:
        if e.get("type") != "assistant":
            continue
        for b in ((e.get("message") or {}).get("content") or []):
            if isinstance(b, dict) and b.get("type") == "tool_use":
                seen.setdefault(b.get("id") or f"noid{len(seen)}", str(b.get("name") or ""))
    cc: dict = {}
    for n in seen.values():
        k = n[len(MCP_PREFIX):] if n.startswith(MCP_PREFIX) else n
        cc[k] = cc.get(k, 0) + 1
    t2: dict = {}
    for tc in tau2_calls:
        k = str(tc.get("name") or "")
        t2[k] = t2.get(k, 0) + 1
    tau2_names = {n[len(MCP_PREFIX):] for n in seen.values() if n.startswith(MCP_PREFIX)} | set(t2)
    split = {k: t2.get(k, 0) for k in RETRIEVAL_TOOLS}
    split["domain_tools"] = sum(v for k, v in t2.items() if k not in RETRIEVAL_TOOLS)
    return {"stream": dict(sorted(cc.items())), "tau2": dict(sorted(t2.items())), "retrieval": split,
            "claude_code_own_tools": sorted(k for k in cc if k not in tau2_names)}


def summarize_unit(mdir: Path, *, container_dir: Path | None = None, template: str | None = None) -> dict:
    """The facts of one collected unit dir (<unit>/{rc, job.json, bridge.log, out/, snapshot/tau2/})."""
    mdir = Path(mdir)
    rc_text = (mdir / "rc").read_text().strip() if (mdir / "rc").is_file() else None
    hj = _jload(mdir / "out" / "harness.json") or {}
    tau2 = mdir / "snapshot" / "tau2"
    sim = _jload(tau2 / "simulation.json")
    infra_sim = _jload(tau2 / H.INFRA_SIMULATION)
    run = _jload(tau2 / "run.json") or {}
    s = sim or infra_sim or {}
    msgs = s.get("messages") or []
    gen_s = sum(float(m.get("generation_time_seconds") or 0) for m in msgs if m.get("role") == "assistant")
    calls = [tc for m in msgs if m.get("role") == "assistant" for tc in (m.get("tool_calls") or [])]
    paired = True
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            ids = [tc.get("id") for tc in m["tool_calls"]]
            nxt = [x.get("id") for x in msgs[i + 1:i + 1 + len(ids)] if x.get("role") == "tool"]
            paired = paired and nxt == ids
    customer_first = next((m.get("content") for m in msgs if m.get("role") == "user" and m.get("content")), None)
    stream = (mdir / "out" / "claude.stream.jsonl").read_text(errors="replace") \
        if (mdir / "out" / "claude.stream.jsonl").is_file() else ""
    evs, _bad = CL.events(stream)
    cc_ms = sum(int(e.get("duration_ms") or 0) for e in evs if e.get("type") == "result")
    stdin = (mdir / "out" / "claude.stdin.jsonl").read_text(errors="replace") \
        if (mdir / "out" / "claude.stdin.jsonl").is_file() else ""
    cc = hj.get("cc") or {}
    clock = hj.get("agent_time") or {}
    servers = {x.get("name"): x.get("status") for x in cc.get("mcp_servers") or [] if isinstance(x, dict)}
    transcript_vm = cc.get("transcript")
    transcript_local = None
    if transcript_vm and container_dir is not None:
        p = Path(container_dir) / transcript_vm.lstrip("/")
        transcript_local = str(p) if p.is_file() else None
    try:
        rc = int(rc_text) if rc_text is not None else None
    except ValueError:
        rc = None
    md = template if template is not None else H.FIRST_MESSAGE_TEMPLATE
    tmpl_sha = (run.get("harness") or {}).get("first_message_sha256")
    return {
        "exit_code": rc, "rc_text": rc_text, "harness_exit_code": hj.get("exit_code"),
        "termination": (hj.get("end") or {}).get("reason") or s.get("termination_reason"),
        "end_detail": (hj.get("end") or {}).get("detail"), "raised": hj.get("raised"), "signal": hj.get("signal"),
        "graded_bundle": hj.get("graded_bundle"), "steps": hj.get("steps"), "tool_errors": hj.get("tool_errors"),
        "seconds": hj.get("seconds"), "where": hj.get("where"),
        "agent_seconds": clock.get("used_s"), "agent_time_limit_s": clock.get("limit_s"),
        "agent_turns": clock.get("turns"),
        "agent_seconds_tau2_generation": round(gen_s, 3), "agent_seconds_cc_results": round(cc_ms / 1000.0, 3),
        "customer": hj.get("customer"),
        "tool_calls": cc.get("tool_calls"), "tool_calls_tau2": len(calls), "id_sources": cc.get("id_sources"),
        "tool_calls_by_name": tool_calls_by_name(evs, calls),
        "retrieval": hj.get("retrieval") if hj.get("retrieval") is not None else (run.get("harness") or {}).get("retrieval"),
        "late_ids": cc.get("late_ids"),
        "tool_call_ids_toolu": sum(1 for tc in calls if str(tc.get("id") or "").startswith("toolu_")),
        "calls_paired_with_results": paired, "messages": len(msgs),
        "session_id": cc.get("session_id"), "claude_code_version": cc.get("claude_code_version"),
        "model": cc.get("model"), "mcp_servers": servers,
        "tau2_tools_listed_n": len(cc.get("tau2_tools_listed") or []),
        "mcp_tools_listed": cc.get("mcp_tools_listed"), "mcp_methods": cc.get("mcp_methods"),
        "cc_turns": cc.get("turns"), "cc_results": cc.get("results"), "api_retries": cc.get("api_retries"),
        "rate_limit_last": CL.rl_brief(cc.get("rate_limit")) if isinstance(cc.get("rate_limit"), dict) else None,
        "transcript_vm": transcript_vm, "transcript": transcript_local,
        "first_message": first_message_check(first_user_message(stdin), customer_first, md),
        "first_message_template_sha256_harness": tmpl_sha,
        "first_message_template_is_the_harness": tmpl_sha == sha256_bytes(md.encode("utf-8")) if tmpl_sha else None,
        "run_json": {k: run.get(k) for k in ("tau2", "data_dir", "domain", "retrieval_variant", "task_id", "seed",
                                             "max_steps", "max_errors", "wall_s", "duration_s")},
        "bundle_files": sorted(p.name for p in tau2.iterdir()) if tau2.is_dir() else None,
        "env_names_bridge_popen": hj.get("env_names"), "claude_env_names": hj.get("claude_env_names"),
        "dropped_env": hj.get("dropped_env"), "user_sim_model": hj.get("user_sim_model"),
        "user_llm_args": hj.get("user_llm_args"), "user_sim_key": hj.get("user_sim_key")}


def pass_checks(s: dict, grade_vm: dict | None, grade_mac: dict | None, secret_hits: list, data_root: str) -> dict:
    """The smoke's PASS criteria for one unit (each True/False)."""
    meth = set(s.get("mcp_methods") or [])
    gv, gm = grade_vm or {}, grade_mac or {}
    num = (int, float)
    fm = s.get("first_message") or {}
    n_calls = s.get("tool_calls_tau2") or 0
    return {
        "exit_0": s.get("exit_code") == 0 and s.get("harness_exit_code") == 0,
        "graded_end": s.get("termination") in ("user_stop", "agent_stop"),
        "tau2_server_connected": (s.get("mcp_servers") or {}).get(H.SERVER) == "connected",
        "tau2_tools_listed": (s.get("tau2_tools_listed_n") or 0) > 0,
        "mcp_handshake_and_calls": {"initialize", "tools/list", "tools/call"} <= meth,
        "tool_call_ids_from_meta": n_calls > 0 and s.get("tool_call_ids_toolu") == n_calls
        and set((s.get("id_sources") or {"_meta": 1})) == {"_meta"},
        "calls_paired_with_results": bool(s.get("calls_paired_with_results")),
        "first_message_is_the_template": bool(fm.get("matches_template"))
        and fm.get("customer_part_is_the_customers_first_message") is True,
        "run_json_tau2_1_0_1": ((s.get("run_json") or {}).get("tau2") or {}).get("version") == "1.0.1"
        and (s.get("run_json") or {}).get("data_dir") == str(data_root),
        "run_json_variant_alltools": (s.get("run_json") or {}).get("retrieval_variant") == VARIANT,
        "agent_clock_recorded": isinstance(s.get("agent_seconds"), num),
        "graded_on_machine": isinstance(gv.get("score"), num),
        "graded_on_mac": isinstance(gm.get("score"), num),
        "grades_equal": isinstance(gv.get("score"), num) and gv.get("score") == gm.get("score"),
        "transcript_collected": bool(s.get("transcript")),
        "no_secret_in_files": not secret_hits}


def harness_settings() -> dict:
    """What the harness file does today: first message, turn and step limits, the customer, the retrieval
    (informational; the smoke runs it as is)."""
    out: dict = {"harness_sha256": sha256_file(H.__file__), "bundle_sha256": sha256_file(H.B.__file__),
                 "retrieval_sha256": sha256_file(TR.__file__)}

    def probe(key, fn):
        try:
            out[key] = fn()
        except Exception as e:  # noqa: BLE001
            out[key] = f"unknown ({type(e).__name__}: {str(e)[:80]})"
    probe("first_message_template_sha256", H.first_message_sha256)
    probe("first_message_fills_policy_and_customer",
          lambda: H.first_message("{policy}", "{customer}") == H.FIRST_MESSAGE_TEMPLATE)
    probe("no_claude_max_turns", lambda: "--max-turns" not in H.stream_argv())
    probe("agent_time_s_default", lambda: H.JOB_DEFAULTS.get("agent_time_s"))
    probe("agent_time_is_20_min", lambda: H.JOB_DEFAULTS.get("agent_time_s") == 1200.0)
    rc = None
    try:
        rc = H.run_config({**H.JOB_DEFAULTS, "domain": DOMAIN, "retrieval_variant": VARIANT}, 0)
    except Exception as e:  # noqa: BLE001
        out["run_config"] = f"unknown ({type(e).__name__})"
    if rc is not None:
        out["max_steps_is_tau2_default_200"] = rc.get("max_steps") == 200   # ends an endless conversation
        out["no_tau2_wall"] = rc.get("timeout") is None
        out["max_errors"] = rc.get("max_errors")
        out["empty_customer_llm_args"] = rc.get("llm_args_user") == {}
        out["customer_model_is_gpt_5_nano"] = rc.get("llm_user") == "openrouter/openai/gpt-5-nano"
    probe("customer_call_retries", lambda: {k: H.JOB_DEFAULTS.get(k) for k in
                                                ("customer_timeout_s", "customer_attempts", "customer_backoff_s")})
    probe("container_mode", lambda: "container" in H.JOB_DEFAULTS)
    probe("job_variant_is_alltools", lambda: VARIANT == "alltools")
    probe("harness_default_variant", lambda: H.JOB_DEFAULTS.get("retrieval_variant"))
    probe("harness_imports_tau3_retrieval",
          lambda: bool(re.search(r"^\s*(import tau3_retrieval|from tau3_retrieval import)", Path(H.__file__).read_text(),
                                 re.M)))
    return out


# ───────────────────────────────────────────────────────────────────────────── the outside world (replaceable)
class Deps:
    """Everything that reaches Daytona, a secret or another driver's code at run time; the tests replace it."""

    def __init__(self):
        import daytona_lib_evo as dle
        self.dle = dle

    def client(self):
        return self.dle.client()

    def sdk(self):
        from daytona import CreateSandboxFromImageParams, Image, Resources
        return CreateSandboxFromImageParams, Image, Resources

    def credential(self, token_file=None) -> tuple:
        """(secret, cc_account.Credential, where): cc_solver.read_credential, which also records the kind for this
        process (cc_solver_lib.credential_kind: token_kind() and every in-container command follow it)."""
        return CS.read_credential(token_file)

    def openrouter_key(self) -> str:
        return openrouter_key()

    def install_ctl(self, run, upload) -> dict:
        import cc_pool_host as CPH
        return CPH.install_ctl(run, upload)

    def install_cc(self, run, upload) -> dict:
        import cc_pool_host as CPH
        return CPH.install_cc(run, upload)

    def egress_cmd(self) -> tuple:
        import ssb_leg_pool as SLP
        return SLP.POOL_EGRESS_CMD, dict(SLP.EGRESS_HOST)

    def grade_local(self, task_file_local, snapshot, home=None) -> dict:
        return grade_local(task_file_local, snapshot, home=home)

    def sleep(self, s: float) -> None:
        time.sleep(s)


# ───────────────────────────────────────────────────────────────────────────── the machine and its containers
class SmokeMachine(CS.Machine):
    """cc_solver.Machine's sh / ctl / save (inherited), with this smoke's own create (the tau3 image, the pool allow
    list, no INFRA leg-snapshot pin) and delete (its own ledger tag)."""

    def __init__(self, deps: Deps, mdir: Path, ttl_minutes: int, resources: dict, log):
        self.deps, self.dle = deps, deps.dle
        self.dir = Path(mdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ttl = int(ttl_minutes)
        self.resources = dict(resources)
        self.d = self.sb = self.sid = self.name = None
        self.rec: dict = {"ttl_minutes": self.ttl, "resources": self.resources}
        self.inf: dict = {}
        self.lock = threading.Lock()
        self.token_name = None
        self.t_created = None
        self.libreoffice = None
        self.log = log

    def up(self, build_dockerfile: Path, meta: dict) -> None:
        from src.orchestrator import pool_container_evolver as pce
        Params, Image, Resources = self.deps.sdk()
        self.d = self.deps.client()
        self.name = f"{self.dle.PREFIX}{NAME_PART}{time.strftime('%H%M%S')}"
        logs = open(self.dir / "build_logs.txt", "a")

        def on_logs(chunk):
            logs.write(str(chunk).rstrip("\n") + "\n")
            logs.flush()
        params = Params(name=self.name, image=Image.from_dockerfile(str(build_dockerfile)),
                        resources=Resources(**self.resources), ephemeral=True, auto_stop_interval=0,
                        ttl_minutes=self.ttl, public=False, domain_allow_list=pce.MACHINE_ALLOW_STR,
                        labels={"role": ROLE})
        self.log(f"creating {self.name} (Dockerfile.tau3_leg {meta['dockerfile_sha256'][:12]}, context "
                 f"{meta['context_sha256'][:12]}, {self.resources}, allow {pce.MACHINE_ALLOW_STR}, ttl {self.ttl} min)")
        self.t_created = time.time()
        try:
            sb, sid, err, secs = self.dle.create(self.d, params, self.resources, tag=TAG, timeout=3600,
                                                 on_logs=on_logs)
        finally:
            logs.close()
        self.sid = sid
        self.rec.update(name=self.name, sandbox_id=sid, create_s=secs, image=meta)
        self.save()
        if err is not None:
            raise Refuse(f"create failed after {secs}s: {type(err).__name__}: {str(err)[:500]}")
        self.sb = sb
        rb = self.dle.readback(sb)
        self.rec["readback_create"] = rb
        probs = []
        if pce.allow_list_set(rb.get("domain_allow_list")) != pce.MACHINE_ALLOW:
            probs.append(f"domain_allow_list {rb.get('domain_allow_list')!r}")
        if rb.get("network_block_all"):
            probs.append("network_block_all is set")
        for k, v in self.resources.items():
            if rb.get(k) != v:
                probs.append(f"{k} {rb.get(k)} != {v}")
        self.rec["readback_problems"] = probs
        self.save()
        self.log(f"machine {sid} up in {secs}s: allow {rb.get('domain_allow_list')!r}, snapshot "
                 f"{rb.get('snapshot_ref')}, {rb.get('cpu')} cpu / {rb.get('memory')} GiB; problems {probs}")
        if probs:
            raise Refuse("machine readback: " + "; ".join(probs))
        self.dle.ledger(event="tau3_smoke_machine_ready", driver=NAME, tag=TAG, id=sid, name=self.name,
                        dockerfile_sha256=meta["dockerfile_sha256"], context_sha256=meta["context_sha256"])

    def upload(self, src, dst: str, timeout: int = 900) -> None:
        self.sb.fs.upload_file(str(src), dst, timeout=timeout)

    def download(self, src: str, dst) -> None:
        self.sb.fs.download_file(src, str(dst))

    def delete(self) -> dict:
        res = {"sandbox_id": self.sid, "name": self.name}
        if not (self.sid or self.name):
            return {**res, "verified_gone": True, "note": "nothing was created"}
        if self.d is None:
            self.d = self.deps.client()
        if not self.sid:                        # interrupted while the image built: the id may never have been seen
            try:
                self.sid = res["sandbox_id"] = self.d.get(self.name).id
            except Exception as e:  # noqa: BLE001
                gone = "not found" in str(e).lower() or "404" in str(e)
                self.log(f"looking up {self.name} to delete it: {type(e).__name__}; "
                         + ("no such sandbox" if gone else "unknown: the TTL is the backstop"))
                res.update(verified_gone=gone, lookup_error=f"{type(e).__name__}: {str(e)[:200]}")
                self.rec["delete"] = res
                self.save()
                return res
        res["deleted"] = self.dle.delete_by_id(self.d, self.sid, TAG)
        t_end = time.time() + 180               # the listing lags a delete by seconds (cc_solver.Machine.delete)
        still, listing, live = [], [], []
        while True:
            try:
                listing = self.dle.list_all_ids(self.d)
            except Exception as e:  # noqa: BLE001
                listing = [f"listing failed: {type(e).__name__}: {e}"]
            still = [r for r in listing if r.startswith(self.sid[:8])]
            live = [r for r in still if "destroy" not in r.lower()] or \
                [r for r in listing if r.startswith("listing failed")]
            if not live or time.time() > t_end:
                break
            self.deps.sleep(10)
        res.update(listing_utc=now_utc(), still_listed=still, verified_gone=bool(res["deleted"]) and not live)
        self.rec["delete"] = res
        self.save()
        self.log(f"machine {self.sid} deleted={res['deleted']}, verified gone {res['verified_gone']}")
        return res


class SmokeBox(CS.Box):
    """A cc-* container exactly as the SSB solver's (cc_solver.Box.prepare / enter / token_cmd, inherited), with no
    PREP.json (nothing is uploaded into it), this smoke's ledger tag and the PRE probe held to the image's OWN
    LibreOffice version (the tau3 image is a fresh build of an unpinned apt package)."""

    def __init__(self, m: SmokeMachine, rd: Path, cid: str, wall_s: int, limits: dict, role: str):
        self.m = m
        self.unit = self.task = role
        self.rd = Path(rd)
        self.rd.mkdir(parents=True, exist_ok=True)
        self.cid = cid
        self.cdir = ctl.cdir(cid)
        self.wall_s = int(wall_s)
        self.spec = CL.container_spec(cid, wall_s=self.wall_s, limits=limits)
        self.rec = {"role": role, "cid": cid, "sandbox_id": m.sid, "spec": self.spec}
        self.status, self.pl, self.stream_fh = "new", None, None
        self.wiped = self.result_written = False
        self.t_launch = None

    def up(self, init_root: bool):
        code, out, res = self.m.ctl(f"up --cid {self.cid}" + (" --init-root-cgroup" if init_root else ""),
                                    timeout=300, what="up")
        self.rec["up"] = res
        self.save()
        self.fail_on(not res.get("ok"), f"container did not come up: {res.get('problems') or out[-600:]}")
        self.m.dle.ledger(event="container_up", driver=NAME, tag=TAG, id=self.m.sid, cid=self.cid, task=self.task)
        self.status = "up"

    def probe(self):
        from src.orchestrator import pool_container_evolver as pce
        code, out, res = self.m.ctl(f"probe --cid {self.cid} --phase pre", timeout=1800, what="probe")
        pre = res.get("probe") or {}
        probs = pce.container_probe_problems(
            pre, expected_top=ctl.expected_top(self.pl), expected_mounts=ctl.expected_mount_points(self.pl),
            rw_paths=[], ro_rebind=[], require_libreoffice=self.m.libreoffice)
        self.rec["probe_pre"] = pre
        self.rec["probe_pre_problems"] = probs
        self.save()
        v = pre.get("view") or {}
        self.m.log(f"{self.task}: PRE probe {len(probs)} problem(s); {v.get('canaries_visible_n')} of "
                   f"{v.get('canaries_checked')} machine paths visible; / = {v.get('top')}")
        self.fail_on(bool(probs), "PRE probe: " + "; ".join(probs[:5]))
        self.status = "ready"

    def down(self) -> dict:
        code, out, res = self.m.ctl(f"down --cid {self.cid}", timeout=300, what="down", retries=2)
        return res

    def wipe(self):
        try:
            code, out, res = self.m.ctl(f"down --cid {self.cid} --wipe", timeout=300, what="down wipe", retries=3)
            self.rec["down_wipe"] = res
            ok = bool(res.get("verified") and res.get("wiped"))
        except CS.Refuse as e:
            self.rec["down_wipe"] = {"error": str(e)}
            ok = False
        self.m.dle.ledger(event="container_down", driver=NAME, tag=TAG, id=self.m.sid, cid=self.cid, verified=ok)
        self.wiped = ok
        self.save()
        return ok


class Unit:
    def __init__(self, task: str, sample: int, stamp: str, out: Path):
        self.task, self.sample = task, int(sample)
        self.name = unit_name(task, sample)
        self.cid = make_cid(task, sample, stamp)
        self.local = Path(out) / "units" / self.name
        self.vm = f"{VM_UNITS}/{self.name}"
        self.mdir = self.local / "machine" / self.name
        self.cont = self.local / "container"
        self.box = None
        self.job = None
        self.status = "new"
        self.t_launch = self.t_end = self.t_stop = self.t_dead = None
        self.hard_stop_s = None
        self.last_poll: dict = {}
        self.errors: list = []
        self.grade_vm = self.grade_mac = None
        self.collected = False
        self.down_before_pack: dict = {}


# ───────────────────────────────────────────────────────────────────────────── the plan (shared by --dry-run)
def make_plan(a, stamp: str) -> dict:
    split = load_split(Path(a.split))
    tasks = pick_tasks(split, a.tasks, a.n, Path(a.data_root))
    overrides = parse_job_sets(a.job_set)
    cred = credential_source(a.token_file)        # where the credential comes from; nothing is read
    units = []
    for t in tasks:
        cid = make_cid(t, a.sample, stamp, salt="xxxx")
        vm = f"{VM_UNITS}/{unit_name(t, a.sample)}"
        job = unit_job(t, a.sample, cid, vm, model=a.model, overrides=overrides, root=Path(a.data_root),
                       token_kind=cred.get("kind") or "oauth")
        eff = validate_job(job)
        lp = H.launch_plan(eff) if hasattr(H, "launch_plan") else {}
        units.append({"unit": unit_name(t, a.sample), "task": t, "sample": a.sample, "cid_example": cid,
                      "vm_dir": vm, "job": job, "agent_time_s": eff.get("agent_time_s"),
                      "hard_stop_s": hard_stop_s(eff, a.hard_stop_min),
                      "claude_argv": lp.get("argv"), "claude_env_names": lp.get("claude_env_names"),
                      "bridge_popen_env_names": sorted(lp.get("env") or {}), "where": lp.get("where")})
    from src.orchestrator import daytona_evolver as de       # READ-ONLY: the leg machine's 8 vCPU / 32 GiB / 40 GB
    resources = dict(SMALL_RESOURCES) if a.small else dict(de.RESOURCES)
    stop_max = max(u["hard_stop_s"] for u in units)
    if int(a.ttl_minutes) * 60 < stop_max + SETUP_AND_COLLECT_S:
        raise Refuse(f"--ttl-minutes {a.ttl_minutes} cannot hold a {stop_max / 60:.0f}-min rollout + "
                     f"{SETUP_AND_COLLECT_S // 60} min of setup and collection")
    import daytona_lib_evo as dle                            # READ-ONLY: the ledgered machine-name prefix (no client)
    files, ctx = context_files()
    ctx.update(dockerfile_meta())
    return {"schema": SCHEMA + ":plan", "stamp": stamp, "tasks": tasks, "units": units, "sample": a.sample,
            "model": a.model, "job_overrides": overrides, "resources": resources, "ttl_minutes": int(a.ttl_minutes),
            "machine_name": f"{dle.PREFIX}{NAME_PART}HHMMSS",
            "allow_list": "openrouter.ai,api.anthropic.com", "image": ctx, "credential": cred,
            "launch_cmd_shape": launch_cmd(units[0]["vm_dir"], "<TOKEN>", "<KEY>"),
            "claude_argv": units[0]["claude_argv"], "split": str(a.split), "data_root": str(a.data_root),
            "harness_settings": harness_settings(),
            "retrieval": {"variant": VARIANT, "module": str(Path(TR.__file__).resolve()),
                          "module_sha256": sha256_file(TR.__file__), "embedder": [TR.EMBEDDER_TYPE, TR.TAU2_MODEL],
                          "openrouter_model": TR.OR_MODEL, "cache_dir": str(TR.cache_dir(Path(a.data_root))),
                          "shell_deny_read": list(TR.deny_read_paths(Path(a.data_root)))},
            "bridge_path": VM_PATH,
            "upload": {"bridge": [str(p) for p in BRIDGE_FILES], "grader": str(GRADER), "split": str(a.split)}}


# ───────────────────────────────────────────────────────────────────────────── the run
class Smoke:
    def __init__(self, a, deps: Deps, out: Path, stamp: str, plan: dict):
        self.a, self.deps, self.out, self.stamp, self.plan = a, deps, Path(out), stamp, plan
        self.out.mkdir(parents=True, exist_ok=True)
        self.red = Redactor()
        self.logfh = open(self.out / "driver.log", "a")
        CS._LOG["fh"] = self.logfh            # cc_solver's own log lines land in driver.log too
        self.tok = self.key = None
        self.credential = None
        self.m = SmokeMachine(deps, self.out / "machine", plan["ttl_minutes"], plan["resources"], self.log)
        self.units = [Unit(u["task"], u["sample"], stamp, self.out) for u in plan["units"]]
        self.root_cgroup_done = False
        self.report: dict = {"schema": SCHEMA, "stamp": stamp, "started_utc": now_utc(), "argv": sys.argv[1:],
                             "plan": {k: plan[k] for k in ("tasks", "sample", "model", "job_overrides", "resources",
                                                            "ttl_minutes", "credential")},
                             "image": plan["image"], "harness_settings": plan["harness_settings"],
                             "errors": []}
        self.rc = 0

    # -- logging
    def log(self, msg: str) -> None:
        line = f"{now_utc()} [tau3-smoke] {self.red(msg)}"
        print(line, flush=True)
        try:
            self.logfh.write(line + "\n")
            self.logfh.flush()
        except ValueError:
            pass

    def err(self, msg: str) -> None:
        self.report["errors"].append(self.red(msg)[:2000])
        self.log("ERROR " + msg)

    def save(self) -> None:
        jdump(self.out / "SMOKE_REPORT.json", self.report)

    # -- 0. secrets (memory only)
    def secrets(self) -> None:
        self.tok, cred, where = self.deps.credential(self.a.token_file)
        self.credential = {"kind": token_kind(cred.kind), "source": cred.source, "from": where}
        self.m.token_name = where
        self.key = self.deps.openrouter_key()
        self.red.add(self.tok, "claude-token")
        self.red.add(self.key, "openrouter-key")
        self.report["credential"] = self.credential
        self.log(f"Claude credential: {self.credential['kind']} from {self.m.token_name} (read into memory only); "
                 f"OpenRouter key read into memory")

    # -- 1-2. image + machine
    def build_and_create(self) -> None:
        meta = write_build_dir(self.out / "build")
        self.report["image"] = meta
        self.save()
        self.m.up(Path(meta["build_dockerfile"]), meta)
        self.report["machine"] = {k: self.m.rec.get(k) for k in ("name", "sandbox_id", "create_s", "ttl_minutes",
                                                                 "resources", "readback_create")}
        self.save()

    def sh(self, cmd: str, *, timeout: int = 600, what: str = "", env=None, retries: int = 3) -> tuple:
        code, out = self.m.sh(cmd, timeout=timeout, what=what, env=env, retries=retries)
        return code, self.red(out)

    # -- 3. machine probes
    def machine_checks(self) -> None:
        cmd, hosts = self.deps.egress_cmd()
        code, out = self.sh(cmd, timeout=180, what="egress")
        m = re.search(r"^EGRESS (.*)$", out, re.M)
        if not m:
            raise Refuse("machine egress probe failed: " + out[-600:])
        eg = json.loads(m.group(1))
        must = sorted(hosts)                       # openrouter + anthropic: both on this machine's allow list
        down = [hosts[k] for k in must if not (eg.get(k) or {}).get("ok")]
        leaked = [k for k in eg if k not in must and (eg.get(k) or {}).get("ok")]
        self.report.setdefault("machine_checks", {})["egress"] = eg
        if down or leaked or any(k not in eg for k in must):
            raise Refuse(f"machine egress: unreachable {down}, reachable beyond the allow list {leaked}")
        code, out = self.sh("bash --version | head -1; head -1 /etc/evoskill_libreoffice_version; "
                            f"{ctl.PY} {VM_CHECK} --python {VM_TAU3_PY} --pin {VM_PIN}", timeout=300, what="checks")
        lines = out.strip().splitlines()
        bv = re.search(r"version (\d+)\.(\d+)", lines[0] if lines else "")
        self.m.libreoffice = lines[1].strip() if len(lines) > 1 and lines[1].startswith("LibreOffice") else None
        venv_ok = "equal: 76 packages, CPython 3.12.13" in out
        self.report["machine_checks"].update(bash=lines[0] if lines else None, libreoffice=self.m.libreoffice,
                                             tau3_venv_check=[ln for ln in lines if "tau3_check_venv" in ln][-1:])
        self.save()
        if not bv or (int(bv.group(1)), int(bv.group(2))) < (5, 1):
            raise Refuse(f"bash on the machine is {lines[:1]}: a here-string must be a pipe (5.1+)")
        if not self.m.libreoffice:
            raise Refuse(f"no LibreOffice version on the image: {lines[1:2]}")
        if not venv_ok:
            raise Refuse(f"the image's tau3 venv differs from its pin: {out[-800:]}")
        self.log(f"machine: egress {sorted(k for k, v in eg.items() if v.get('ok'))} reachable only; "
                 f"{lines[0]}; {self.m.libreoffice}; tau3 venv equals its pin")

    # -- 4. installs + uploads
    def install(self) -> None:
        def run(cmd, timeout, what):
            return self.m.sh(cmd, timeout=timeout, what=what)

        def upload(src, dst, timeout):
            self.m.upload(src, dst, timeout=timeout)
        try:
            rc_ctl = self.deps.install_ctl(run, upload)
            rc_cc = self.deps.install_cc(run, upload)
        except Exception as e:  # noqa: BLE001 -- cc_pool_host raises its leg driver's Abort
            raise Refuse(f"install (evoctl / proxy / Claude Code): {type(e).__name__}: {e}") from None
        self.report["install"] = {"ctl": rc_ctl, "cc": rc_cc}
        self.save()
        self.log(f"evoctl + proxy: {rc_ctl.get('ctl')}; Claude Code {ctl.CC_VERSION}: {rc_cc.get('cc')}")

    def upload_files(self) -> None:
        self.sh(f"install -d -m 0700 {VM_BASE} {VM_BIN} {VM_UNITS}", timeout=60, what="mkdir")
        want = {}
        for src in [*BRIDGE_FILES, GRADER]:
            dst = f"{VM_BIN}/{Path(src).name}"
            self.m.upload(src, dst, timeout=300)
            want[dst] = sha256_file(src)
        self.m.upload(Path(self.a.split), VM_SPLIT, timeout=300)
        want[VM_SPLIT] = sha256_file(self.a.split)
        code, out = self.sh("sha256sum " + " ".join(q(p) for p in want), timeout=60, what="bridge sha")
        got = {ln.split()[1]: ln.split()[0] for ln in out.splitlines() if len(ln.split()) == 2}
        if got != want:
            raise Refuse(f"bridge / grader / split upload does not verify ({got} != {want})")
        self.report["uploads"] = {"files": want}
        tgz = self.out / "data_upload.tgz"
        tgz.write_bytes(data_tgz(Path(self.a.data_root)))
        try:
            self.m.upload(tgz, VM_DATA_TGZ, timeout=900)
        finally:
            tgz.unlink()
        r = q(str(self.a.data_root))
        code, out = self.sh(
            f"set -e; R={r}; test ! -e \"$R\"; mkdir -p \"$R\"; tar -xzf {VM_DATA_TGZ} -C \"$R\" --no-same-owner; "
            f"rm -f {VM_DATA_TGZ}; cd \"$R\"; sha256sum -c --quiet MANIFEST.sha256; "
            f"test -z \"$(find . -type l)\"; "
            f"test \"$(find . -type f | wc -l)\" = \"$(( $(wc -l < MANIFEST.sha256) + 1 ))\"; echo DATA_OK",
            timeout=600, what="data root")
        if "DATA_OK" not in out:
            raise Refuse(f"the data root does not verify on the machine: {out[-600:]}")
        self.report["uploads"]["data_root"] = str(self.a.data_root)
        self.save()
        self.log(f"bridge + grader + split uploaded and verified; data root at {self.a.data_root}, MANIFEST verified")

    def tau3_check(self) -> None:
        code, out = self.sh(f"cd / && env {' '.join('-u ' + k for k in KEY_ENV_NAMES)} PYTHON_DOTENV_DISABLED=1 "
                            f"LITELLM_LOCAL_MODEL_COST_MAP=True TAU2_DATA_DIR={q(str(self.a.data_root))} "
                            f"{VM_TAU3_PY} -c {q(TAU3_CHECK_PY)}", timeout=600, what="tau3 check")
        res = parse_tail(out, "TAU3_CHECK")
        failed = parse_tail(out, "TAU3_CHECK_FAIL")
        self.report["tau3_check"] = res or {"failed": failed}
        self.save()
        if not res:
            raise Refuse(f"tau3 check on the machine: no TAU3_CHECK line; TAU3_CHECK_FAIL "
                         f"{failed.get('reasons') if failed else out[-800:]}")
        probs = []
        if res.get("tau2_version") != "1.0.1":
            probs.append(f"tau2 {res.get('tau2_version')!r} ({res.get('import_error')})")
        if res.get("data_dir") != str(self.a.data_root) or res.get("n_tasks") != 97:
            probs.append(f"data dir {res.get('data_dir')!r} with {res.get('n_tasks')} tasks")
        if res.get("openrouter_models_status") != 200:
            probs.append(f"openrouter /models {res.get('openrouter_models_status')} {res.get('openrouter_error')}")
        if probs:
            raise Refuse("tau3 on the machine: " + "; ".join(probs))
        sb, emb, sh = res.get("sandbox") or {}, res.get("embeddings") or {}, res.get("shell") or {}
        self.log(f"tau3 venv: tau2 {res['tau2_version']}, litellm {res.get('litellm')}, {res['n_tasks']} tasks; "
                 f"openrouter /models 200 (gpt-5-nano listed: {res.get('openrouter_has_gpt_5_nano')}); srt "
                 f"{sb.get('srt_package_version')} + {sorted(b for b, w in (sb.get('which') or {}).items() if w)}; "
                 f"embeddings {emb.get('shape')} from {emb.get('entry')}; KB shell {sh.get('ok')}")

    # -- 6-7. a unit's container, job and launch
    def setup_and_launch(self, u: Unit, stop_s: float) -> None:
        u.hard_stop_s = float(stop_s)
        u.box = SmokeBox(self.m, u.local, u.cid, int(stop_s), CL.limits_for(1), u.name)
        u.box.prepare()
        u.box.up(init_root=not self.root_cgroup_done)
        self.root_cgroup_done = True
        u.box.probe()
        u.job = unit_job(u.task, u.sample, u.cid, u.vm, model=self.a.model,
                         overrides=self.plan["job_overrides"], root=Path(self.a.data_root),
                         token_kind=self.credential["kind"])
        validate_job(u.job)
        u.local.mkdir(parents=True, exist_ok=True)
        jf = u.local / "job.json"
        jf.write_text(json.dumps(u.job, indent=1) + "\n")
        self.sh(f"install -d -m 0700 {q(u.vm)} {q(u.vm + '/snapshot')}", timeout=60, what="unit dir")
        self.m.upload(jf, f"{u.vm}/job.json", timeout=120)
        code, out = self.sh(f"sha256sum {q(u.vm + '/job.json')} | cut -d' ' -f1", timeout=60, what="job sha")
        if out.strip().splitlines()[-1:] != [sha256_file(jf)]:
            raise Refuse(f"{u.name}: job.json does not verify on the machine")
        cmd = launch_cmd(u.vm, self.tok, self.key) + f" ; sleep 3; cat {q(u.vm + '/bridge.pid')} 2>/dev/null"
        u.t_launch = time.time()
        u.status = "running"
        code, out = self.m.sh(cmd, timeout=180, env={TOK_ENV: self.tok, KEY_ENV: self.key}, what="launch",
                              retries=1)              # never retried: a second bridge
        self.dle_ledger("bridge_launched", cid=u.cid, unit=u.name)
        self.log(f"{u.name}: bridge launched in {u.cid} (pid {self.red(out).strip()[-12:]}); agent time "
                 f"{u.job.get('agent_time_s', H.JOB_DEFAULTS.get('agent_time_s'))} s; hard stop {stop_s:.0f} s")

    def dle_ledger(self, event: str, **kw) -> None:
        try:
            self.m.dle.ledger(event=event, driver=NAME, tag=TAG, id=self.m.sid, **kw)
        except Exception as e:  # noqa: BLE001
            self.log(f"ledger {event}: {e}")

    # -- 8. poll
    def poll(self, units: list) -> dict:
        code, out = self.sh(f"{ctl.PY} -c {q(POLL_PY)} " + " ".join(q(u.vm) for u in units), timeout=120,
                            what="poll", retries=2)
        return parse_tail(out, "POLL")

    def stop(self, u: Unit, why: str) -> None:
        """SIGTERM to the bridge: the harness kills claude's process group, clears the bundle path, writes
        harness.json and exits 75 (an infrastructure end), and the launch shell writes its rc."""
        u.t_stop = time.time()
        u.errors.append(f"stopped: {why}")
        pid = (u.last_poll or {}).get("pid")
        self.log(f"{u.name}: STOP ({why}): SIGTERM to the bridge (pid {pid})")
        if pid:
            try:
                self.sh(f"kill -TERM {int(pid)} 2>/dev/null; true", timeout=60, what="stop bridge", retries=1)
            except CS.Refuse as e:
                self.err(f"{u.name}: SIGTERM at stop: {e}")

    def kill_bridge(self, u: Unit) -> None:
        """The last resort: evoctl down (freeze + cgroup kill: claude goes), then SIGKILL to the bridge."""
        try:
            u.down_before_pack = u.box.down()
        except CS.Refuse as e:
            self.err(f"{u.name}: down: {e}")
        pid = (u.last_poll or {}).get("pid")
        if pid:
            self.sh(f"kill -TERM {int(pid)} 2>/dev/null; sleep 5; kill -KILL {int(pid)} 2>/dev/null; true",
                    timeout=60, what="kill bridge", retries=1)

    def wait_all(self) -> None:
        last_note = 0.0
        poll_failures = 0
        while True:
            running = [u for u in self.units if u.status == "running"]
            if not running:
                return
            try:
                res = self.poll(running)
                poll_failures = 0 if res else poll_failures + 1
            except CS.Refuse as e:
                self.log(f"poll failed: {e}")
                res, poll_failures = {}, poll_failures + 1
            now = time.time()
            for u in running:
                r = res.get(u.name)
                if r is not None:
                    u.last_poll = r
                    (u.local / "bridge_log_tail.txt").write_text(self.red(r.get("log_tail") or ""))
                if r is not None and r.get("rc") is not None:
                    u.status, u.t_end = "ended", now
                    self.log(f"{u.name}: bridge exited rc {r['rc']} after {(now - u.t_launch) / 60:.1f} min "
                             f"(stream {r.get('stream_bytes')} B, {r.get('stdin_lines')} stdin lines)")
                    self.collect(u)
                elif u.t_stop is None and now - u.t_launch > u.hard_stop_s:
                    self.stop(u, f"hard stop at {u.hard_stop_s:.0f} s")
                elif u.t_stop is not None and now - u.t_stop > STOP_GRACE_S:
                    self.log(f"{u.name}: no rc {STOP_GRACE_S} s after the stop: the bridge is killed")
                    self.kill_bridge(u)
                    u.status, u.t_end = "killed", now
                    self.collect(u)
                elif r is not None and not r.get("alive") and r.get("pid"):
                    u.t_dead = u.t_dead or now
                    if now - u.t_dead > DEAD_GRACE_S:
                        u.errors.append("the bridge is gone and wrote no rc")
                        u.status, u.t_end = "bridge_died", now
                        self.collect(u)
                elif poll_failures >= POLL_FAIL_STOP:
                    u.errors.append(f"{poll_failures} polls in a row failed")
                    self.err(f"{u.name}: {poll_failures} polls in a row failed: stopped and collected as it is")
                    self.kill_bridge(u)
                    u.status, u.t_end = "poll_failed", now
                    self.collect(u)
            if now - last_note > 60:
                last_note = now
                for u in [x for x in self.units if x.status == "running"]:
                    r = u.last_poll
                    self.log(f"{u.name}: running {(now - u.t_launch) / 60:.1f} min; stream {r.get('stream_bytes')} B; "
                             f"stdin {r.get('stdin_lines')} lines; bridge alive {r.get('alive')}")
            self.deps.sleep(POLL_S)

    # -- 9. collect
    def collect(self, u: Unit) -> None:
        try:
            if not u.down_before_pack:
                u.down_before_pack = u.box.down()
            code, out, res = self.m.ctl(f"pack --cid {u.cid} " + " ".join(f"--part {q(p)}" for p in CONTAINER_PARTS),
                                        timeout=900, what="pack")
            u.box.rec["pack"] = {k: res.get(k) for k in ("ok", "bytes", "packed", "refused")}
            if res.get("ok"):
                local = u.local / "container_out.tgz"
                self.m.download(f"{u.box.cdir}/state/out.tgz", local)
                u.box.rec["container_refused_members"] = safe_extract(local, u.cont)
            else:
                u.errors.append(f"pack refused: {res.get('problems') or out[-300:]}")
            code, out = self.sh(f"tar -czf {q(u.vm + '.tgz')} -C {VM_UNITS} {q(u.name)} && echo TAR_OK",
                                timeout=300, what="unit tar")
            if "TAR_OK" in out:
                local = u.local / "unit.tgz"
                self.m.download(f"{u.vm}.tgz", local)
                u.box.rec["unit_refused_members"] = safe_extract(local, u.local / "machine")
            else:
                u.errors.append(f"unit tar failed: {out[-300:]}")
            code, out = self.sh(grade_vm_cmd(u.job["task_file"], f"{u.vm}/snapshot"), timeout=900, what="grade")
            u.grade_vm = parse_tail(out, "GRADE_JSON") or {"raised": "NoOutput", "message": out[-400:]}
            jdump(u.local / "GRADE_machine.json", u.grade_vm)
            u.collected = True
            self.log(f"{u.name}: collected; machine grade {u.grade_vm}")
        except Exception as e:  # noqa: BLE001
            u.errors.append(f"collect: {type(e).__name__}: {e}")
            self.err(f"{u.name}: collect: {type(e).__name__}: {e}")
        finally:
            u.box.rec["down_before_pack"] = u.down_before_pack
            u.box.save()
            if not u.box.wiped:
                u.box.wipe()

    # -- 10. the host side
    def grade_mac_all(self) -> None:
        home = self.out / "mac_grade_home"
        home.mkdir(exist_ok=True)
        for u in self.units:
            snap = u.mdir / "snapshot"
            if not (snap / "tau2" / "simulation.json").is_file():
                u.grade_mac = {"skipped": "no tau2/simulation.json (an infrastructure end or nothing collected)"}
            else:
                u.grade_mac = self.deps.grade_local(task_file(u.task, Path(self.a.data_root)), snap, home=home)
            jdump(u.local / "GRADE_mac.json", u.grade_mac)
            self.log(f"{u.name}: host grade {u.grade_mac} (machine {u.grade_vm})")

    def unit_report(self, u: Unit, hits: list) -> dict:
        s = summarize_unit(u.mdir, container_dir=u.cont) if u.mdir.is_dir() else {}
        down = u.down_before_pack or {}
        scan = down.get("scan") or {}
        mhits = [h for part in scan.values() for h in (part or {}).get("hits") or []] + \
            list((down.get("tmpfs_scan") or {}).get("hits") or [])
        checks = pass_checks(s, u.grade_vm, u.grade_mac, hits + mhits, str(self.a.data_root)) if s \
            else {"collected": False}
        return {"unit": u.name, "task": u.task, "sample": u.sample, "cid": u.cid, "status": u.status,
                "minutes": round((u.t_end - u.t_launch) / 60, 2) if u.t_end and u.t_launch else None,
                "grade_machine": u.grade_vm, "grade_mac": u.grade_mac,
                "secret_hits_local": hits, "secret_hits_container": mhits,
                "container_wiped": bool(u.box and u.box.wiped), "errors": u.errors, **s,
                "checks": checks, "passed": bool(s) and all(checks.values())}

    # -- the whole run
    def run(self) -> int:
        self.log(f"SMOKE {self.stamp}: tasks {self.plan['tasks']}, sample {self.a.sample}, model {self.a.model}; "
                 f"out {self.out}")
        hv = self.plan["harness_settings"]
        pend = [k for k, v in hv.items() if k.startswith("d2") and v is False]
        if pend:
            self.log(f"NOTE: the harness does not yet implement {pend}; the smoke runs it as it is")
        try:
            self.secrets()
            self.build_and_create()
            self.machine_checks()
            self.install()
            self.upload_files()
            self.tau3_check()
            for u, pu in zip(self.units, self.plan["units"]):
                try:
                    self.setup_and_launch(u, pu["hard_stop_s"])
                except (Refuse, CS.Refuse) as e:
                    u.status = "setup_failed"
                    u.errors.append(f"setup: {e}")
                    self.err(f"{u.name}: setup refused: {e}")
                    self.rc = self.rc or 4
            self.wait_all()
            self.grade_mac_all()
        except (Refuse, CS.Refuse) as e:
            self.err(f"REFUSED: {e}")
            self.rc = 4
        except BaseException as e:  # noqa: BLE001 -- SIGTERM / Ctrl-C / crash: still tear everything down
            self.err(f"CRASH: {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
            self.rc = 6
        finally:
            self.teardown()
        return self.rc

    def teardown(self) -> None:
        prev = {}
        for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            try:
                prev[s] = signal.signal(s, lambda *_a: self.log("a signal during teardown: ignored, it goes on"))
            except (ValueError, OSError):
                pass
        try:
            self._teardown()
        finally:
            for s, h in prev.items():
                try:
                    signal.signal(s, h)
                except (ValueError, OSError, TypeError):
                    pass

    def _teardown(self) -> None:
        for u in self.units:
            if u.status == "running":
                self.log(f"{u.name}: still running at teardown: down + kill")
                try:
                    self.kill_bridge(u)
                except Exception as e:  # noqa: BLE001
                    self.err(f"{u.name}: stop at teardown: {e}")
                u.status = "torn_down"
            if u.box is not None and u.box.status != "new" and not u.box.wiped:
                if not u.box.wipe():
                    self.rc = self.rc or 7
        if self.m.sb is not None or self.m.sid or self.m.name:
            try:
                d = self.m.delete()
            except Exception as e:  # noqa: BLE001
                d = {"error": f"{type(e).__name__}: {e}"}
            self.report["machine_delete"] = d
            if not d.get("verified_gone"):
                self.rc = self.rc or 8
                self.err(f"the machine {self.m.sid} is NOT verified gone: delete it (delete --sandbox-id)")
        self.finish()

    def finish(self) -> None:
        needles = {"claude-token": self.tok, "token-prefix": token_prefix(), "openrouter-key": self.key,
                   "openrouter-prefix": OR_PREFIX}
        rows = []
        for u in self.units:
            hits = secret_scan([u.local], needles)["hits"]
            rows.append(self.unit_report(u, hits))
            jdump(u.local / "SUMMARY.json", rows[-1])
        other = secret_scan([self.out / "driver.log", self.out / "machine"], needles)
        self.report["machine"] = {k: self.m.rec.get(k) for k in ("name", "sandbox_id", "create_s", "ttl_minutes",
                                                                 "resources", "readback_create", "readback_problems",
                                                                 "delete")}
        self.report.update(units=rows, secret_scan_other=other, ended_utc=now_utc(),
                           all_units_passed=bool(rows) and all(r["passed"] for r in rows))
        if other["hits"]:
            self.err(f"secret-shaped bytes in driver files: {other['hits']}")
        if self.rc == 0 and not self.report["all_units_passed"]:
            self.rc = 3
        if self.rc == 0 and other["hits"]:
            self.rc = 3
        self.report["rc"] = self.rc
        self.save()
        for r in rows:
            self.log(f"{r['unit']}: passed {r['passed']}; exit {r.get('exit_code')} / harness "
                     f"{r.get('harness_exit_code')}; {r.get('termination')}; agent {r.get('agent_seconds')} s; "
                     f"tool calls {r.get('tool_calls')}; grade machine {r.get('grade_machine')} mac "
                     f"{r.get('grade_mac')}; failed checks {[k for k, v in (r.get('checks') or {}).items() if not v]}")
        self.log(f"SMOKE {self.stamp} rc {self.rc}; report {self.out / 'SMOKE_REPORT.json'}")
        try:
            self.logfh.close()
        except Exception:  # noqa: BLE001
            pass
        CS._LOG["fh"] = None


# ───────────────────────────────────────────────────────────────────────────── commands
def _raise_exit(signum, frame):
    raise SystemExit(128 + signum)


def cmd_run(a, deps: Deps | None = None) -> int:
    stamp = time.strftime("%Y%m%dT%H%M%S")
    plan = make_plan(a, stamp)
    if a.dry_run:
        print(json.dumps(plan, indent=1, default=str))
        print(f"DRY RUN: nothing created, no secret read, nothing written; {len(plan['units'])} unit(s), image "
              f"{plan['image']['dockerfile_sha256'][:12]} + context {plan['image']['context_sha256'][:12]}")
        return 0
    deps = deps or Deps()
    verify = verify_data_root(Path(a.data_root))
    if not CC_LOCAL.is_file() or CL.sha256_file(CC_LOCAL) != ctl.CC_SHA256 or CC_LOCAL.stat().st_size != ctl.CC_BYTES:
        raise Refuse(f"{CC_LOCAL}: not the pinned Claude Code {ctl.CC_VERSION} linux-x64 binary")
    out = Path(a.out or OUT_ROOT) / stamp
    s = Smoke(a, deps, out, stamp, plan)
    s.report["data_root_check"] = verify
    jdump(out / "PLAN.json", plan)
    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _raise_exit)
    return s.run()


def cmd_build_context(a) -> int:
    meta = write_build_dir(Path(a.dest))
    print(json.dumps(meta, indent=1))
    return 0


def cmd_delete(a, deps: Deps | None = None) -> int:
    """Delete ONE leftover smoke machine by id: its name must carry this smoke's prefix."""
    deps = deps or Deps()
    d = deps.client()
    sb = d.get(a.sandbox_id)
    if not str(sb.name).startswith(deps.dle.PREFIX + NAME_PART):
        print(f"REFUSED: {a.sandbox_id} is {sb.name!r}, not a {deps.dle.PREFIX}{NAME_PART}* machine", file=sys.stderr)
        return 4
    ok = deps.dle.delete_by_id(d, a.sandbox_id, TAG + "-delete")
    print(json.dumps({"sandbox_id": a.sandbox_id, "name": sb.name, "deleted": ok}))
    return 0 if ok else 8


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tasks", default=None, help="comma list of split['train'] task ids (default: the first --n)")
    r.add_argument("--n", type=int, default=2)
    r.add_argument("--sample", type=int, default=1)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--ttl-minutes", type=int, default=DEFAULT_TTL_MIN)
    r.add_argument("--small", action="store_true", help="4 vCPU / 16 GiB / 40 GB instead of 8/32/40")
    r.add_argument("--hard-stop-min", type=float, default=None,
                   help=f"stop a rollout after this long (default: agent_time_s + {CUSTOMER_ALLOWANCE_S // 60} min)")
    r.add_argument("--token-file", default=None,
                   help=f"a Claude subscription token file, mode 0600 (default: {TOKEN_FILE_ENV}; with neither, "
                        f"{API_KEY_ENV} from the environment)")
    r.add_argument("--model", default=MODEL)
    r.add_argument("--job-set", action="append", default=[], help="KEY=JSON: one harness job key (not a path/fd)")
    r.add_argument("--out", default=None, help=f"out root (default {OUT_ROOT})")
    r.add_argument("--split", default=str(SPLIT))
    r.add_argument("--data-root", required=True,
                   help="the curated tau3 data root (tau3_make_data_root.py --dest, with its .embeddings_cache)")
    b = sub.add_parser("build-context")
    b.add_argument("--dest", required=True)
    d = sub.add_parser("delete")
    d.add_argument("--sandbox-id", required=True)
    return ap


def main(argv=None, deps: Deps | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        if a.cmd == "run":
            return cmd_run(a, deps)
        if a.cmd == "build-context":
            return cmd_build_context(a)
        return cmd_delete(a, deps)
    except Refuse as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
