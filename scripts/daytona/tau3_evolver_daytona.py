#!/usr/bin/env python3
"""Dispatch ONE epoch's evolver on its own Daytona sandbox built from the evolver image (RUNTIME
evolver_host = "daytona", placement own_sandbox). Its classes are also the base of the container drivers
(evolver_container.py, evolver_session.py) and of the benchmark adapter's (setup_ao_probe.py).

    tau3_evolver_daytona.py <epoch> <prompt> <digests_dir> <staging_dir>     # the loop's dispatch
    tau3_evolver_daytona.py probe --run-dir outputs/<run>                    # live smoke test
    tau3_evolver_daytona.py cleanup --run-dir outputs/<run>                  # the run's evo-prefix sandboxes

Why, and what is proven where: src/orchestrator/daytona_evolver.py (the checks the loop runs).

DISPATCH, in order (exit codes in brackets; every non-zero HALTs the loop):
  1. the launcher's own pre-dispatch gates, via tau3_evolver.sh with EVOSKILL_TAU3_GATE_ONLY=1
     and a scratch staging dir (its exit code is passed through: 2/4/7/9/10/16/21 ...);
  2. the run's Claude Code credential (src/orchestrator/cc_account.py: a token file or ANTHROPIC_API_KEY; never
     printed), prompt size [2], the grants from RUNTIME.json evolver_daytona (train_inputs / train_references /
     grader_clean, absent = off) and each granted dir verifies [23];
  3. upload manifest + receipt + trace written BEFORE anything is created;
  4. create (solver image + CC, 8/32/40, domain allow list api.anthropic.com), readback [25];
     from here on everything runs inside try/finally delete_by_id [29 if the delete is unverified];
  5. identity [25] + LibreOffice pin [26] + egress probes [22]; pre-upload listing;
  6. upload one tgz; the VM listing must equal the manifest exactly [23];
  7. launch detached, the credential in the exec env ONLY (claude reads it once from fd 3); stream the log back
     byte-exactly (base64 chunks);
  8. collect: after-run hashes of the read-only roots, staging + session transcript + log into a
     QUARANTINE dir, log sha round-trip, token scan [24]; readback at end; delete;
  9. install log / staging / transcript into the run dir; claude rc != 0 -> [27], timeout -> [28].

RATE LIMITS. No usage is read before a launch. A session that the API REJECTED for a usage limit or with an HTTP 429
(its own stream says so) ends the turn [27]; the dispatch then waits until the reset time the session reported
(bounded by EVOSKILL_RATE_LIMIT_MAX_WAIT_S, one clear log line) and runs the turn again (run_resuming).
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import io
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import daytona_lib_evo as dle  # noqa: E402
from src.orchestrator import daytona_evolver as de  # noqa: E402
from src.orchestrator import evidence_surface as es  # noqa: E402

EXIT_USAGE, EXIT_EGRESS, EXIT_MANIFEST = 2, 22, 23
EXIT_TOKEN, EXIT_IDENTITY, EXIT_PIN, EXIT_CLAUDE_RC, EXIT_TIMEOUT, EXIT_DELETE = 24, 25, 26, 27, 28, 29
EXIT_LIVE = 4                            # a run-level command refused while a dispatch of the run is live
POLL_S = 15
CHUNK = 3_000_000


class Refuse(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


def log(msg: str) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def jdump(p: Path, obj) -> None:
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    os.replace(tmp, p)


def iso(ts) -> str | None:
    return dt.datetime.fromtimestamp(float(ts), dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") \
        if isinstance(ts, (int, float)) else None


# ── the Claude Code credential: ONE per run (src/orchestrator/cc_account.py), handed to claude on fd 3 only.
# The SDK exec's env carries it under SECRET_ENV for one shell; `env -u` drops it before any process of ours starts,
# and claude reads it once from the here-string pipe on fd 3 (the variable FD_ENV names tells it which kind).
SECRET_ENV = "EVOSKILL_CC_SECRET"
FD_ENV = {"token_file": "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR", "api_key": "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR"}


def load_credential(cfg: dict) -> tuple:
    """(secret, the claude variable naming its fd, record) of the run's credential: RUNTIME evolver_daytona's
    oauth_token_file when declared, else $EVOSKILL_CC_TOKEN_FILE, else $ANTHROPIC_API_KEY. A token file must be
    owner-only and hold a setup-token. The record names the kind and the file, never the secret."""
    from src.orchestrator import cc_account
    try:
        c = cc_account.resolve((cfg or {}).get("oauth_token_file"))
    except ValueError as e:
        raise Refuse(EXIT_USAGE, f"credential: {e}")
    if c.kind == cc_account.TOKEN_FILE:
        secret = c.token_file.read_text().strip()
        if not secret.startswith(de.TOKEN_PREFIX):
            raise Refuse(EXIT_USAGE, f"token file {c.token_file} does not hold a setup-token (prefix check)")
    else:
        secret = (os.environ.get(cc_account.API_KEY_ENV) or "").strip()
    if not secret or any(ch.isspace() for ch in secret):
        raise Refuse(EXIT_USAGE, f"the {c.kind} credential is empty or holds whitespace (it is not printed)")
    return secret, FD_ENV[c.kind], {"kind": c.kind, "source": c.source,
                                    "token_file": str(c.token_file) if c.token_file else None}


# ── rate limits: an actual rejection ends the turn; the dispatch waits for the reported reset and runs it again
RATE_LIMIT_MAX_WAIT_ENV = "EVOSKILL_RATE_LIMIT_MAX_WAIT_S"
RATE_LIMIT_MAX_WAIT_S = 6 * 3600          # the longest single wait (a five-hour window resets within it)
RATE_LIMIT_NO_RESET_WAIT_S = 900          # a rejection that names no reset time (a bare HTTP 429) waits this long
RATE_LIMIT_MARGIN_S = 60                  # past the reported reset before the turn runs again
RATE_LIMIT_MAX_RESUMES = 3                # rejections one dispatch waits out; the next one ends it [27]
_LIMIT_TEXT = ("usage limit", "rate limit", "hit your limit", "limit reached")


def _rate_limit_info(o):
    """The rate_limit_info object inside one stream event, wherever it is nested."""
    if isinstance(o, dict):
        if isinstance(o.get("rate_limit_info"), dict):
            return o["rate_limit_info"]
        for v in o.values():
            r = _rate_limit_info(v)
            if r:
                return r
    elif isinstance(o, list):
        for v in o:
            r = _rate_limit_info(v)
            if r:
                return r
    return None


def usage_limit_rejection(log_path) -> dict | None:
    """The session's own report, in its stream-json log, that the API REJECTED it (a usage limit, or an HTTP 429)
    and that it ended on that error: {"resets_at", "resets_at_utc", "rate_limit_type", "api_error_status",
    "detail"}. None otherwise: a utilization figure or a warning is not a rejection, nor is one the session got past."""
    info = result = None
    try:
        fh = open(log_path, errors="replace")
    except OSError:
        return None
    with fh:
        for line in fh:
            if not line.startswith("{"):
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if not isinstance(e, dict):
                continue
            ri = _rate_limit_info(e)
            if ri:
                info = ri
            if e.get("type") == "result":
                result = e
    if result is not None and not result.get("is_error"):
        return None
    text = str((result or {}).get("result") or "").strip()
    rejected = isinstance(info, dict) and info.get("status") == "rejected"
    http_429 = (result or {}).get("api_error_status") == 429
    if not (rejected or http_429 or (result is not None and any(k in text.lower() for k in _LIMIT_TEXT))):
        return None
    info = info or {}
    ra = info.get("resetsAt")
    if not isinstance(ra, (int, float)):
        w = ((info.get("unifiedWindows") or {}).get(info.get("rateLimitType") or "") or {})
        ra = w.get("resetsAt") if isinstance(w.get("resetsAt"), (int, float)) else None
    return {"resets_at": ra, "resets_at_utc": iso(ra), "rate_limit_type": info.get("rateLimitType"),
            "status": info.get("status"), "api_error_status": (result or {}).get("api_error_status"),
            "detail": text[:200]}


def wait_for_reset(rl: dict, *, what: str) -> bool:
    """Sleep until the reset `rl` reports (+ RATE_LIMIT_MARGIN_S), or RATE_LIMIT_NO_RESET_WAIT_S when it names none,
    and return True; return False at once (logged) when that is further away than the bound."""
    try:
        cap = float(os.environ.get(RATE_LIMIT_MAX_WAIT_ENV) or RATE_LIMIT_MAX_WAIT_S)
    except ValueError:
        cap = float(RATE_LIMIT_MAX_WAIT_S)
    ra = rl.get("resets_at")
    wait = (float(ra) - time.time() + RATE_LIMIT_MARGIN_S) if ra else float(RATE_LIMIT_NO_RESET_WAIT_S)
    wait = max(wait, float(RATE_LIMIT_MARGIN_S))
    why = rl.get("rate_limit_type") or (f"HTTP {rl['api_error_status']}" if rl.get("api_error_status") else
                                        "usage limit")
    src = f"the reset the session reported, {rl.get('resets_at_utc')}" if ra else "no reset time reported"
    if wait > cap:
        log(f"RATE LIMIT: {what} was rejected ({why}); {src} is {int(wait)} s away, beyond the {int(cap)} s bound "
            f"({RATE_LIMIT_MAX_WAIT_ENV}): not waiting")
        return False
    log(f"RATE LIMIT: {what} was rejected ({why}); waiting {int(wait)} s until {iso(time.time() + wait)} ({src}), "
        f"then running it again")
    time.sleep(wait)
    return True


def run_resuming(make_run, *, gate_argv=None, what: str = "the evolver turn") -> int:
    """make_run() builds a fresh run object; run it, and while its session ended on a rate-limit rejection, wait for
    the reported reset and run a fresh one (at most RATE_LIMIT_MAX_RESUMES times). A stateful session's re-run
    resumes the same conversation: its rejected epoch is recorded failed, not collected (evolver_session.py)."""
    rc = 0
    for attempt in range(RATE_LIMIT_MAX_RESUMES + 1):
        run = make_run()
        rc = run.run(gate_argv=gate_argv)
        rl = getattr(run, "rate_limited", None)
        if rc != EXIT_CLAUDE_RC or not rl:
            return rc
        if attempt == RATE_LIMIT_MAX_RESUMES:
            log(f"RATE LIMIT: {what} was rejected {attempt + 1} times in a row; giving up (exit {rc})")
            return rc
        if not wait_for_reset(rl, what=what):
            return rc
    return rc


# ── per-run liveness: a lock file under the run dir instead of a machine-wide process search
DISPATCH_LOCK = Path(".machine") / "evolver_dispatch.lock"
_HELD_LOCKS: list = []


def hold_dispatch_lock(run_dir) -> None:
    """A live dispatch (evolver or adaptor) of the run holds <run>/.machine/evolver_dispatch.lock SHARED until its
    process exits (dispatches never exclude each other); a run-level command that must not overlap one (cleanup,
    finish) takes it EXCLUSIVE. It sees this run only, and the kernel drops a holder's share when it dies."""
    import fcntl
    p = Path(run_dir) / DISPATCH_LOCK
    p.parent.mkdir(parents=True, exist_ok=True)
    fh = open(p, "a")
    fcntl.flock(fh, fcntl.LOCK_SH)
    _HELD_LOCKS.append(fh)


def exclusive_dispatch_lock(run_dir):
    """The run's dispatch lock taken EXCLUSIVE (an open file: keep it while no dispatch may start; closing it
    releases), or None while a dispatch of the run is live."""
    import fcntl
    p = Path(run_dir) / DISPATCH_LOCK
    p.parent.mkdir(parents=True, exist_ok=True)
    fh = open(p, "a")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


IDENTITY_CMD = r"""
echo "whoami=$(whoami)"; echo "arch=$(uname -m)"
echo "libreoffice=$(cat /etc/evoskill_libreoffice_version 2>/dev/null)"
echo "cc_file=$(cat /etc/evoskill_claude_code 2>/dev/null)"
echo "cc_sha256=$(sha256sum __CC__ 2>/dev/null | cut -d' ' -f1)"
echo "cc_binaries=$(find / -xdev -name claude -type f 2>/dev/null | wc -l)"
echo "cc_version=$(IS_SANDBOX=1 HOME=/tmp/ccv DISABLE_AUTOUPDATER=1 CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 timeout 60 claude --version 2>&1 | head -1)"
echo "solver_stack=$(/usr/local/bin/python3 -W ignore -c 'import openpyxl,pandas,numpy,sys;print(openpyxl.__version__,pandas.__version__,numpy.__version__,sys.version.split()[0])' 2>/dev/null)"
echo "opt_entries=$(ls -A /opt 2>/dev/null | wc -l)"
echo "users_exists=$([ -e /Users ] && echo yes || echo no)"
echo "secret_env_names=$(env | cut -d= -f1 | grep -iE 'api_key|token|secret|password' | sort | tr '\n' ',')"
echo "proxy_env_names=$(env | cut -d= -f1 | grep -i proxy | sort | tr '\n' ',')"
echo "evo_uid=$(id -u evo 2>/dev/null)"
echo "evo_lo_profile=$([ -d /home/evo/.config/libreoffice ] && stat -c %U /home/evo/.config/libreoffice)"
echo "policy_dir=$([ -e /etc/claude-code ] && ls -A /etc/claude-code | wc -l || echo absent)"
""".replace("__CC__", de.CC_PATH)

EGRESS_PY = r"""
import json, socket, sys, urllib.request, urllib.error
out = []
for u in sys.argv[1:]:
    try:
        urllib.request.urlopen(urllib.request.Request(u, method="GET"), timeout=10)
        out.append({"url": u, "reachable": True, "detail": "HTTP 2xx/3xx"})
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read(300).decode("utf-8", "replace")
        except Exception:
            pass
        # Plain HTTP goes through the sandbox's HTTP_PROXY, which answers a denied host ITSELF with
        # an HTTP 403; the origin never saw the request. Only an answer that carries the origin's
        # own content (Cloudflare's trace page: "h=1.1.1.1") proves the origin was reached. Over
        # HTTPS a denial is a failed CONNECT (URLError), so any HTTP status there IS the origin.
        denied_by_proxy = u.startswith("http://") and e.code == 403 and not (
            body.startswith("fl=") or "\nh=1.1.1.1" in body)
        out.append({"url": u, "reachable": not denied_by_proxy, "detail": f"HTTP {e.code}",
                    "server": e.headers.get("Server", ""), "body": body[:160]})
    except Exception as e:
        out.append({"url": u, "reachable": False, "detail": f"{type(e).__name__}: {str(e)[:120]}"})
direct = []
# PROXY-LESS: a client that ignores HTTP(S)_PROXY. Any HTTP answer from the origin = reached.
_np = urllib.request.build_opener(urllib.request.ProxyHandler({}))
for t in ("https://example.com", "https://1.1.1.1", "https://github.com"):
    try:
        _np.open(t, timeout=10)
        direct.append({"target": t, "reached": True, "detail": "HTTP 2xx/3xx"})
    except urllib.error.HTTPError as e:
        direct.append({"target": t, "reached": True, "detail": f"HTTP {e.code}"})
    except Exception as e:
        direct.append({"target": t, "reached": False, "detail": f"{type(e).__name__}: {str(e)[:100]}"})
raw = []
for host, port in (("1.1.1.1", 443), ("8.8.8.8", 443), ("example.com", 22), ("169.254.169.254", 80)):
    s = socket.socket(); s.settimeout(6)
    try:
        s.connect((host, port)); raw.append({"target": f"{host}:{port}", "tcp_connected": True})
    except Exception as e:
        raw.append({"target": f"{host}:{port}", "tcp_connected": False, "detail": type(e).__name__})
    finally:
        s.close()
try:
    u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); u.settimeout(4)
    u.sendto(b"\x12\x34\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x07example\x03com\x00\x00\x01\x00\x01", ("1.1.1.1", 53))
    u.recvfrom(512); raw.append({"target": "udp 1.1.1.1:53", "answered": True})
except Exception as e:
    raw.append({"target": "udp 1.1.1.1:53", "answered": False, "detail": type(e).__name__})
try:
    s6 = socket.socket(socket.AF_INET6, socket.SOCK_STREAM); s6.settimeout(6)
    s6.connect(("2606:4700:4700::1111", 443)); raw.append({"target": "[2606:4700:4700::1111]:443", "tcp_connected": True})
except Exception as e:
    raw.append({"target": "[2606:4700:4700::1111]:443", "tcp_connected": False, "detail": type(e).__name__})
print(json.dumps({"http": out, "direct": direct, "raw": raw}))
"""


def parse_kv(text: str) -> dict:
    out = {}
    for ln in text.splitlines():
        if "=" in ln and not ln.startswith(" "):
            k, v = ln.split("=", 1)
            out[k.strip()] = v.strip()
    return out


class EvolverRun:
    DEADLINE_TAKES_STAGING = True        # the evolver's staging is taken as it stands at the deadline
    # the roots the upload places files under besides /evo: each must be absent before the upload, is root-owned
    # after it and is covered by the listing (the benchmark adapter uses /data, not the host's paths)
    MIRROR_ROOTS = ("/Users",)

    def __init__(self, *, epoch: int, prompt: Path, digests: Path, staging: Path, run_dir: Path,
                 prior: list, model: str, oracle: Path, cfg: dict, timeout_s: int,
                 receipt_path: Path, trace_path: Path, listing_path: Path, log_path: Path,
                 quarantine: Path, max_turns: int | None = None, install: bool = True, mode: str = "all_epochs"):
        self.epoch, self.prompt, self.digests, self.staging = epoch, prompt, digests, staging
        self.run_dir, self.prior, self.model, self.oracle, self.cfg = run_dir, prior, model, oracle, cfg
        self.timeout_s, self.max_turns, self.install_outputs, self.mode = timeout_s, max_turns, install, mode
        self.receipt_path, self.trace_path, self.listing_path = receipt_path, trace_path, listing_path
        self.log_path, self.q = log_path, quarantine
        self.q.mkdir(parents=True, exist_ok=True)
        os.chmod(self.q, 0o700)
        self.sid_session = str(uuid.uuid4())
        self.client = None
        self.sb = None
        self.sandbox_id = None
        self.receipt: dict = {}
        self.token = ""                           # the credential's secret (load_credential), never printed
        self.fd_env = FD_ENV["token_file"]
        self.credential = None
        self.rate_limited = None
        self.stream_name = "evolver.log"          # the AO role streams to ao.log (dashboard path)

    # ── helpers
    def sh(self, cmd: str, timeout: int = 600, env: dict | None = None, what: str = "",
           retries: int = 4) -> tuple:
        last = None
        for attempt in range(retries):
            try:
                code, out, secs = dle.run(self.sb, cmd, timeout=timeout, env=env)
                return code, out
            except Exception as e:  # noqa: BLE001 -- transient toolbox errors
                last = e
                time.sleep(15)
        raise Refuse(EXIT_IDENTITY, f"exec {what or cmd[:40]!r} failed 4x: {type(last).__name__}: {last}")

    def save_receipt(self) -> None:
        jdump(self.receipt_path, self.receipt)

    @staticmethod
    def as_evo(inner: str, stdout: str, stderr: str) -> str:
        """Run `inner` as the unprivileged user with a CLEAN environment and the credential on fd 3
        only. The credential arrives in this exec's env (the SDK's env parameter, never a command string);
        `env -u` drops it before setpriv, `env -i` starts the child from nothing, and bash's here-string
        puts it in a pipe that claude drains once at startup (CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR=3, or
        CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR=3 for an API key; read once and cached). So no process running as
        `evo` -- claude, its Bash tool, a nested claude -- has it in its environment or can re-read it from the pipe."""
        proxies = " ".join(f'{v}="${{{v}:-}}"' for v in
                           ("HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy"))
        return (f"env -u {SECRET_ENV} setpriv --reuid=evo --regid=evo --init-groups "
                f"env -i HOME={de.VM_HOME} PATH=/usr/local/bin:/usr/bin:/bin {proxies} "
                f"bash -c {shlex.quote(inner)} > {stdout} 2> {stderr} < /dev/null "
                f'3<<<"${SECRET_ENV}"')

    # ── the two seams the STATEFUL session driver overrides (scripts/daytona/evolver_session.py): what the upload
    # manifest is, and which dirs claude gets --add-dir'ed. Here = the per-epoch behaviour (prepare() calls them).
    def build_manifest(self, prev_lib):
        try:
            return de.build_upload_manifest(
                run_dir=self.run_dir, epoch=self.epoch, prompt=self.prompt, evidence_dirs=self.evidence_dirs,
                oracle=self.oracle, staging=self.staging, prev_library=prev_lib,
                inputs=self.inputs, references=self.references, grants=self.grants,
                grader=self.grader, grader_granted=self.grader_grant, template_mode=self.template_mode,
                env_text=de.render_env_file(self.env), hooks_text=de.hook_settings_text(), split=self.split)
        except ValueError as e:
            raise Refuse(EXIT_MANIFEST, f"upload manifest: {e}")

    def add_dirs_for(self) -> list:
        return [*self.evidence_dirs, str(self.staging),
                *([] if self.template_mode else [str(self.oracle)]),
                *(d for d in (self.inputs, self.references, self.grader) if d)]

    # ── 2-3: everything decided on the host before any sandbox exists
    def prepare(self) -> None:
        self.token, self.fd_env, self.credential = load_credential(self.cfg)
        pb = self.prompt.read_bytes()
        if len(pb) > de.MAX_PROMPT_BYTES:
            raise Refuse(EXIT_USAGE, f"prompt is {len(pb)} B > {de.MAX_PROMPT_BYTES} (one argv string)")
        ok_df, det_df, rec_df = de.dockerfile_check()
        if not ok_df:
            raise Refuse(EXIT_IDENTITY, f"evolver Dockerfile: {det_df}")
        split = os.environ.get("EVOSKILL_TAU3_EVOLVER_SPLIT", "")
        if not split:
            raise Refuse(EXIT_USAGE, "EVOSKILL_TAU3_EVOLVER_SPLIT unset")
        self.split = Path(split)
        # Training inputs and training reference workbooks reach the VM only when the operator's RUNTIME.json
        # grants them -- the same file check_upload_manifest reads.
        try:
            self.grants = de.evolver_grants(self.run_dir)
        except ValueError as e:
            raise Refuse(EXIT_USAGE, f"grants: {e}")
        if self.grants["source"].startswith("no RUNTIME.json"):
            raise Refuse(EXIT_USAGE, f"{self.run_dir}/RUNTIME.json missing: the grants must be declared")
        for k in de.GRANT_KEYS:
            if k in self.cfg and self.cfg[k] is not self.grants[k]:
                raise Refuse(EXIT_USAGE, f"EVOSKILL_EVOLVER_DAYTONA_JSON {k}={self.cfg[k]!r} != "
                                         f"RUNTIME.json's {self.grants[k]!r}")
        self.inputs = self.references = None
        if self.grants["train_inputs"]:
            c = de.verify_inputs_dir(self.run_dir, self.split)
            if c.status != "PASS":
                raise Refuse(EXIT_MANIFEST, f"inputs dir: {c.detail}")
            self.inputs = str(de.inputs_dir(self.run_dir))
        if self.grants["train_references"]:
            c = de.verify_references_dir(self.run_dir, self.split)
            if c.status != "PASS":
                raise Refuse(EXIT_MANIFEST, f"references dir: {c.detail}")
            self.references = str(de.references_dir(self.run_dir))
        # The scoring rule + clean grader: <run>/grader_clean reaches the VM only when RUNTIME.json
        # evolver_daytona.grader_clean grants it (the loop-rendered prompt names that dir).
        try:
            self.grader_grant = de.evolver_grader_grant(self.run_dir)
        except ValueError as e:
            raise Refuse(EXIT_USAGE, f"grants: {e}")
        if de.GRADER_GRANT in self.cfg and self.cfg[de.GRADER_GRANT] is not self.grader_grant:
            raise Refuse(EXIT_USAGE, f"EVOSKILL_EVOLVER_DAYTONA_JSON {de.GRADER_GRANT}="
                                     f"{self.cfg[de.GRADER_GRANT]!r} != RUNTIME.json's {self.grader_grant!r}")
        self.grader = None
        if self.grader_grant:
            c = de.verify_grader_dir(self.run_dir, self.split)
            if c.status != "PASS":
                raise Refuse(EXIT_MANIFEST, f"grader dir: {c.detail}")
            self.grader = str(de.grader_dir(self.run_dir))
        # With RUNTIME prompt_mode=template the AO-era oracle README (which says the reference workbooks
        # are withheld, against the grants) is NOT uploaded, and the oracle dir is neither --add-dir'ed
        # nor hashed: the grader and references dirs replace it.
        try:
            self.template_mode = de.evolver_template_mode(self.run_dir)
        except ValueError as e:
            raise Refuse(EXIT_USAGE, f"prompt mode: {e}")
        self.evidence_dirs = [os.path.realpath(p) for p in self.prior] + [os.path.realpath(str(self.digests))]
        prev_lib = self.run_dir / f"skills_v{self.epoch - 1}"
        if self.epoch > 1 and not prev_lib.is_dir():
            raise Refuse(EXIT_USAGE, f"missing previous library {prev_lib}")
        self.prev_lib = prev_lib
        self.env = de.vm_env(self.model, env=os.environ)
        rows, gen = self.build_manifest(prev_lib)               # the seam (per-epoch: de.build_upload_manifest)
        self.rows, self.gen = rows, gen
        self.add_dirs = self.add_dirs_for()                     # the seam (per-epoch: the host-path list)
        add_argv = [x for d in self.add_dirs for x in ("--add-dir", d)]
        sb_argv = [x for d in self.add_dirs for x in ("--grant", d)]
        ev = es.record(self.mode, self.digests, self.prior, add_argv, sb_argv)
        counts = {}
        for r in rows:
            counts[r["role"]] = counts.get(r["role"], 0) + 1
        self.receipt = {
            "sandboxed": True, "containment": "daytona", "profile": None,
            "work_dir": de.VM_WORK, "evidence_mode": self.mode, **ev,
            "grants": dict(self.grants),
            "inputs_dir": self.inputs,
            "inputs_sha256_before": es.dir_sha256(self.inputs) if self.inputs else None,
            "references_dir": self.references,
            "references_sha256_before": es.dir_sha256(self.references) if self.references else None,
            "grader_granted": self.grader_grant,
            "grader_dir": self.grader,
            "grader_sha256_before": es.dir_sha256(self.grader) if self.grader else None,
            "template_mode": self.template_mode, "oracle_readme_uploaded": not self.template_mode,
            "upload_manifest": rows, "upload_counts": counts,
            "daytona": {"driver": str(Path(__file__).relative_to(REPO)), "prefix": de.EVO_PREFIX,
                        "resources_requested": de.RESOURCES, "domain_allow_list": de.ALLOW_DOMAINS,
                        "dockerfile": rec_df, "cc_version": de.CC_VERSION,
                        "cc_sha256_pinned": de.CC_LINUX_X64_SHA, "session_id": self.sid_session,
                        "config": {k: v for k, v in self.cfg.items() if k != "oauth_token_file"},
                        "credential": self.credential, "oauth_token_file": self.credential["token_file"],
                        "vm_env_names": sorted(self.env), "vm_env": self.env, "max_turns": self.max_turns},
            "written_utc": dle.now(),
        }
        if ev.get("evidence_sha256_error"):
            raise Refuse(21, f"evidence hash failed: {ev['evidence_sha256_error']}")
        self.save_receipt()
        trace = {"iteration": self.epoch, "session_id": self.sid_session, "cwd": de.VM_WORK,
                 "project_dir": de.VM_SESSION_DIR, "stream_json_log": str(self.log_path),
                 "prompt_file": str(self.prompt), "prompt_chars": len(pb.decode(errors="replace")),
                 "digests": str(self.digests), "staging": str(self.staging), "oracle": str(self.oracle),
                 "oracle_readme_uploaded": not self.template_mode,
                 "inputs_dir": self.inputs, "references_dir": self.references, "grader_dir": self.grader,
                 "n_digests": sum(1 for f in Path(self.digests).glob("*.md")),
                 "host": "daytona", **ev}
        jdump(self.trace_path, trace)
        log(f"manifest: {len(rows)} files {counts}; receipt + trace written before any sandbox exists")

    # ── 4: create
    def create(self) -> None:
        from daytona import CreateSandboxFromImageParams, Image, Resources
        self.client = dle.client()
        name = f"{de.EVO_PREFIX}e{self.epoch}-{time.strftime('%H%M%S')}"
        ttl = int(self.cfg.get("ttl_minutes") or (math.ceil(self.timeout_s / 60) + 60))
        params = CreateSandboxFromImageParams(
            name=name, image=Image.from_dockerfile(str(de.EVO_DOCKERFILE)),
            resources=Resources(**de.RESOURCES), ephemeral=True, auto_stop_interval=0,
            ttl_minutes=ttl, public=False, domain_allow_list=de.ALLOW_DOMAINS,
            labels={"role": "evolver", "run": self.run_dir.name, "epoch": str(self.epoch)})
        log(f"creating {name} ({de.RESOURCES}, ttl {ttl} min, egress {de.ALLOW_DOMAINS})")
        blog = open(self.q / "image_build.log", "a")
        sb = sid = err = None
        for attempt in range(10):
            sb, sid, err, secs = dle.create(self.client, params, de.RESOURCES, tag=self.run_dir.name,
                                            timeout=3600,
                                            on_logs=lambda c: (blog.write(c + "\n"), blog.flush()))
            self.sandbox_id = sid
            if err is None:
                break
            msg = str(err).lower()
            if sid:
                dle.delete_by_id(self.client, sid, self.run_dir.name)
                self.sandbox_id = None
            if "per sandbox" in msg or not any(k in msg for k in (
                    "quota", "limit", "exceed", "capacity", "insufficient", "no available",
                    "429", "timeout", "timed out")):
                raise Refuse(EXIT_IDENTITY, f"create failed: {type(err).__name__}: {str(err)[:400]}")
            log(f"create refused ({str(err)[:160]}); retry {attempt + 1}/10 in 60 s")
            time.sleep(60)
            params.name = f"{de.EVO_PREFIX}e{self.epoch}-{time.strftime('%H%M%S')}"
        if err is not None:
            raise Refuse(EXIT_IDENTITY, f"create failed 10x: {str(err)[:400]}")
        self.sb = sb
        rb = dle.readback(sb)
        self.receipt["daytona"].update(sandbox_id=sid, sandbox_name=params.name, readback_create=rb)
        self.save_receipt()
        log(f"created {sid} readback cpu={rb['cpu']} mem={rb['memory']} disk={rb['disk']} "
            f"allow={rb.get('domain_allow_list')!r} snapshot_ref={rb.get('snapshot_ref')}")
        mism = {k: (de.RESOURCES[k], rb.get(k)) for k in de.RESOURCES if rb.get(k) != de.RESOURCES[k]}
        if mism:
            raise Refuse(EXIT_IDENTITY, f"resource readback mismatch {mism}")
        if rb.get("domain_allow_list") != de.ALLOW_DOMAINS or rb.get("network_allow_list") \
                or rb.get("network_block_all"):
            raise Refuse(EXIT_EGRESS, f"network readback wrong: {rb.get('domain_allow_list')!r} / "
                                      f"{rb.get('network_allow_list')!r} / {rb.get('network_block_all')!r}")
        want = self.cfg.get("require_snapshot_ref")
        if want and rb.get("snapshot_ref") != want:
            raise Refuse(EXIT_PIN, f"snapshot {rb.get('snapshot_ref')} != pinned {want}")

    # ── 5: identity, egress, pre-upload listing
    def probes(self) -> None:
        code, out = self.sh(IDENTITY_CMD, timeout=300, what="identity")
        ident = parse_kv(out)
        self.receipt["daytona"]["identity"] = ident
        log("identity " + json.dumps(ident))
        bad = []
        if ident.get("whoami") != "root" or ident.get("arch") != "x86_64":
            bad.append("not root/x86_64")
        if ident.get("cc_sha256") != de.CC_LINUX_X64_SHA:
            bad.append(f"CC sha {ident.get('cc_sha256')}")
        if ident.get("cc_binaries") != "1":
            bad.append(f"{ident.get('cc_binaries')} claude binaries")
        if not str(ident.get("cc_version", "")).startswith(de.CC_VERSION):
            bad.append(f"claude --version {ident.get('cc_version')!r}")
        if not str(ident.get("solver_stack", "")).startswith("3.1.3 2.2.0 1.26.4 3.11."):
            bad.append(f"solver stack {ident.get('solver_stack')!r}")
        if ident.get("opt_entries") != "0":
            bad.append("/opt not empty")
        if ident.get("users_exists") != "no":
            bad.append("/Users exists in the image")
        if ident.get("secret_env_names"):
            bad.append(f"secret-shaped env names in the sandbox: {ident.get('secret_env_names')}")
        if ident.get("evo_uid") != "1000" or ident.get("evo_lo_profile") != "evo":
            bad.append(f"unprivileged user/profile missing (uid {ident.get('evo_uid')!r}, "
                       f"profile owner {ident.get('evo_lo_profile')!r})")
        if ident.get("policy_dir") not in ("absent", "0"):
            bad.append("/etc/claude-code already holds files in the image")
        if bad:
            self.save_receipt()
            raise Refuse(EXIT_IDENTITY, "identity: " + "; ".join(bad))
        want_lo = self.cfg.get("require_libreoffice")
        if want_lo and ident.get("libreoffice") != want_lo:
            self.save_receipt()
            raise Refuse(EXIT_PIN, f"LibreOffice {ident.get('libreoffice')!r} != pinned {want_lo!r}")
        urls = list(de.MUST_REACH) + list(de.MUST_REFUSE)
        code, out = self.sh("/usr/local/bin/python3 -c " + shlex.quote(EGRESS_PY) + " "
                            + " ".join(shlex.quote(u) for u in urls), timeout=400, what="egress")
        try:
            eg = json.loads(out.strip().splitlines()[-1])
        except Exception:  # noqa: BLE001
            raise Refuse(EXIT_EGRESS, f"egress probe output unparseable: {out[-400:]}")
        self.receipt["daytona"]["egress_probe"] = eg["http"]
        self.receipt["daytona"]["egress_direct"] = eg["direct"]
        self.receipt["daytona"]["egress_raw"] = eg["raw"]
        reach = {r["url"]: r["reachable"] for r in eg["http"]}
        raw = {r["target"]: r for r in eg["raw"]}
        # Daytona's domain_allow_list sandboxes resolve through public DNS (/etc/resolv.conf: 1.1.1.1, 1.0.0.1), so
        # UDP 53 answers. DNS resolution gives no web content (it cannot help an agent cheat), so it is accepted when
        # 1.1.1.1 is the sandbox's own configured resolver; recorded in the receipt. Every HTTP, TCP, IPv6 and
        # metadata check below stays: web egress is still api.anthropic.com only.
        code_r, resolv = self.sh("cat /etc/resolv.conf", timeout=60, what="resolv.conf")
        resolvers = {ln.split()[1] for ln in resolv.splitlines()
                     if ln.strip().startswith("nameserver") and len(ln.split()) > 1}
        dns_open = raw.get("udp 1.1.1.1:53", {}).get("answered") is True
        self.receipt["daytona"]["egress_dns"] = {"udp_1_1_1_1_53_answered": dns_open,
                                                 "configured_resolvers": sorted(resolvers),
                                                 "accepted": dns_open and "1.1.1.1" in resolvers}
        if dns_open and "1.1.1.1" in resolvers:
            raw["udp 1.1.1.1:53"] = dict(raw["udp 1.1.1.1:53"], answered=False, accepted_as="configured resolver")
        wrong = [u for u in de.MUST_REACH if not reach.get(u)] + \
                [u for u in de.MUST_REFUSE if reach.get(u) is not False] + \
                [r["target"] for r in eg["direct"] if r.get("reached") is not False] + \
                [t for t, k in (("example.com:22", "tcp_connected"), ("udp 1.1.1.1:53", "answered"),
                                ("[2606:4700:4700::1111]:443", "tcp_connected"),
                                ("169.254.169.254:80", "tcp_connected")) if raw.get(t, {}).get(k) is not False]
        log(f"egress: must-reach {[reach.get(u) for u in de.MUST_REACH]}, "
            f"{sum(1 for u in de.MUST_REFUSE if reach.get(u) is False)}/{len(de.MUST_REFUSE)} refused; "
            f"raw {eg['raw']}")
        self.save_receipt()
        if wrong:
            raise Refuse(EXIT_EGRESS, f"egress wrong for {wrong}")
        code, out = self.sh("for d in " + " ".join(shlex.quote(r) for r in self.MIRROR_ROOTS)
                            + '; do test -e "$d" && echo "EXISTS $d"; done; echo PRE_DONE', timeout=60,
                            what="pre-listing")
        if "PRE_DONE" not in out:
            raise Refuse(EXIT_MANIFEST, f"pre-upload check failed: {out[-300:]}")
        existing = [ln.split(" ", 1)[1] for ln in out.splitlines() if ln.startswith("EXISTS ")]
        self.listing = {"pre_upload": {"users_exists": "/Users" in existing, "mirror_roots_existing": existing}}

    # ── 6: upload + listing
    def upload(self) -> None:
        tgz = self.q / "upload.tgz"
        with tarfile.open(tgz, "w:gz", format=tarfile.PAX_FORMAT) as tf:
            for r in self.rows:
                arc = r["vm_path"].lstrip("/")
                if r["source"] == "generated":
                    data = self.gen[r["vm_path"]]
                    ti = tarfile.TarInfo(arc)
                    ti.size, ti.mode, ti.mtime = len(data), 0o644, int(time.time())
                    tf.addfile(ti, io.BytesIO(data))
                else:
                    ti = tf.gettarinfo(r["source"], arcname=arc)
                    ti.uid = ti.gid = 0
                    ti.uname = ti.gname = "root"
                    with open(r["source"], "rb") as fh:
                        tf.addfile(ti, fh)
        self.sb.fs.upload_file(str(tgz), "/tmp/evo.tgz", timeout=1800)
        st = shlex.quote(str(self.staging))
        extra_rw = ""
        # the AO / benchmark adapter: create-only in the run-dir mirror (writable_dir), or every dir of
        # writable_dirs (the setup dir and the adapter's reference dir, created here when the upload lacks it)
        rw = getattr(self, "writable_dirs", None) or \
            ([self.writable_dir] if getattr(self, "writable_dir", None) else [])
        for d in rw:
            q = shlex.quote(str(d))
            extra_rw += f" && mkdir -p {q} && chmod 1777 {q}"
        owned = " ".join(shlex.quote(r) for r in (*self.MIRROR_ROOTS, de.VM_ROOT, "/etc/claude-code"))
        code, out = self.sh(
            "tar xzf /tmp/evo.tgz -C / && rm -f /tmp/evo.tgz"
            f' && for d in {owned}; do [ -e "$d" ] || continue; chown -R root:root "$d"'
            ' && chmod -R a+rX,go-w "$d" || exit 1; done'
            f" && mkdir -p {st} {de.VM_WORK} && chown evo:evo {st} {de.VM_WORK}{extra_rw}"
            " && echo UPLOAD_OK", timeout=900, what="extract")
        if "UPLOAD_OK" not in out:
            raise Refuse(EXIT_MANIFEST, f"extract failed: {out[-400:]}")
        code, out = self.sh("/usr/local/bin/python3 -c " + shlex.quote(de.VM_LISTING_PY) + " "
                            + " ".join(shlex.quote(r) for r in self.listing_roots()), timeout=900, what="listing")
        self.listing["after_upload"] = json.loads(out.strip().splitlines()[-1])
        jdump(self.listing_path, self.listing)
        files = {p: h for p, t, h, _b in self.listing["after_upload"] if t == "F"}
        links = [p for p, t, _h, _b in self.listing["after_upload"] if t == "L"]
        man = {r["vm_path"]: r["sha256"] for r in self.rows}
        extra, missing = sorted(set(files) - set(man)), sorted(set(man) - set(files))
        diff = sorted(p for p in set(files) & set(man) if files[p] != man[p])
        log(f"upload: {len(files)} files on the VM; manifest {len(man)}; extra {len(extra)} missing "
            f"{len(missing)} differing {len(diff)} symlinks {len(links)}")
        pre = self.listing["pre_upload"]
        if extra or missing or diff or links or pre["users_exists"] or pre.get("mirror_roots_existing"):
            raise Refuse(EXIT_MANIFEST, f"VM listing != manifest (extra {extra[:3]}, missing "
                                        f"{missing[:3]}, diff {diff[:3]}, links {links[:3]}, existed before "
                                        f"the upload {pre.get('mirror_roots_existing')})")

    def listing_roots(self) -> list:
        return [*de.listing_roots(), *(r for r in self.MIRROR_ROOTS if r not in de.listing_roots())]

    # ── 7: launch + stream
    def claude_argv(self) -> list:
        from src.orchestrator.claude_pins import effort_argv
        argv = ["claude", "-p", "__PROMPT__", "--model", self.model, "--session-id", self.sid_session]
        for d in self.add_dirs:
            argv += ["--add-dir", d]
        argv += ["--disallowed-tools", "WebFetch", "WebSearch", *effort_argv(self.model, os.environ),
                 "--settings", f"{de.VM_HOOKS}/settings.json", "--thinking-display", "summarized",
                 "--dangerously-skip-permissions", "--output-format", "stream-json", "--verbose"]
        if self.max_turns:
            argv += ["--max-turns", str(self.max_turns)]
        return argv

    prompt_vm_file = f"{de.VM_WORK}/PROMPT.md"
    # A run whose argv carries "__SYSPROMPT__" (the setup AO: the benchmark-agnostic AO guidance appended to Claude
    # Code's default system prompt) gets it expanded from this uploaded VM file, like __PROMPT__.
    # None = no run uses it (every other argv is unchanged).
    sysprompt_vm_file = None

    def launch(self) -> None:
        argv = self.claude_argv()
        inner = " ".join(shlex.quote(a) for a in argv).replace(
            shlex.quote("__PROMPT__"), '"$(cat ' + self.prompt_vm_file + ')"')
        if self.sysprompt_vm_file:
            inner = inner.replace(shlex.quote("__SYSPROMPT__"), '"$(cat ' + self.sysprompt_vm_file + ')"')
        elif "__SYSPROMPT__" in argv:
            raise Refuse(EXIT_USAGE, "argv carries __SYSPROMPT__ but no system-prompt file is set")
        inner = (f"set -a; . {de.VM_ENV_FILE}; set +a; cd {de.VM_WORK}; "
                 f"{self.fd_env}=3 {inner} & "
                 f"echo $! > {de.VM_OUT}/pid; wait $!; "
                 f"echo $? > {de.VM_OUT}/rc.tmp && mv {de.VM_OUT}/rc.tmp {de.VM_OUT}/rc")
        cmd = (f"mkdir -p {de.VM_OUT} && chown evo:evo {de.VM_OUT} && (setsid nohup "
               + self.as_evo(inner, f"{de.VM_OUT}/evolver.log", f"{de.VM_OUT}/evolver.stderr")
               + f" &) ; sleep 8; head -c 1500 {de.VM_OUT}/evolver.log")
        if self.token in cmd:
            raise Refuse(EXIT_TOKEN, "token would appear in a command string")
        self.receipt["daytona"]["launch_argv"] = argv
        self.receipt["daytona"]["launched_utc"] = dle.now()
        self.save_receipt()
        code, out = self.sh(cmd, timeout=180, env={SECRET_ENV: self.token}, what="launch",
                            retries=1)                # never re-run: a second claude, a truncated log
        self.launched = True
        log(f"launched; first bytes: {out[:300]!r}")

    def stream(self) -> str:
        """Pull the VM log byte-exactly until claude exits. Returns 'done' or 'timeout'."""
        local = self.q / self.stream_name
        off, t0, status = 0, time.time(), "done"
        with open(local, "ab") as fh:
            while True:
                cmd = (f"s=$(stat -c %s {de.VM_OUT}/evolver.log 2>/dev/null || echo 0); "
                       f"r=$(cat {de.VM_OUT}/rc 2>/dev/null); [ -n \"$r\" ] || r=RUNNING; "
                       f"echo \"$s $r\"; echo @@B64@@; "
                       f"tail -c +{off + 1} {de.VM_OUT}/evolver.log 2>/dev/null | head -c {CHUNK} | base64 -w0")
                try:
                    code, out = self.sh(cmd, timeout=180, what="poll")
                    head, sep, b64 = out.partition("@@B64@@")
                    if not sep:
                        raise ValueError("no marker in poll output")
                    size_s, _, rc_s = head.strip().splitlines()[-1].partition(" ")
                    int(size_s)
                    data = base64.b64decode(b64.strip()) if b64.strip() else b""
                    self._poll_fail = 0
                except Exception as e:  # noqa: BLE001 -- a lost poll is retried, bounded
                    self._poll_fail = getattr(self, "_poll_fail", 0) + 1
                    if self._poll_fail > 40:
                        raise Refuse(EXIT_TIMEOUT, f"polling failed 40x in a row: {type(e).__name__}: {e}")
                    time.sleep(POLL_S)
                    continue
                if data:
                    fh.write(data)
                    fh.flush()
                    off += len(data)
                size = int(size_s or 0)
                if rc_s.strip() != "RUNNING" and off >= size:
                    self.claude_rc = int(rc_s.strip() or -1)
                    break
                if time.time() - t0 > self.timeout_s:
                    log(f"TIMEOUT after {self.timeout_s}s: stopping claude on the VM")
                    # by the recorded PID: a `pkill -f 'claude -p'` also matches its own shell on Linux
                    self.sh(f"P=$(cat {de.VM_OUT}/pid 2>/dev/null); if [ -n \"$P\" ]; then kill -TERM \"$P\" "
                            f"2>/dev/null; sleep 20; kill -KILL \"$P\" 2>/dev/null; fi; sleep 3; "
                            f"[ -s {de.VM_OUT}/rc ] || {{ echo 124 > {de.VM_OUT}/rc.tmp && mv {de.VM_OUT}/rc.tmp "
                            f"{de.VM_OUT}/rc; }}", timeout=120, what="kill")
                    status = "timeout"
                    t0 = float("inf")  # one kill only; keep draining until rc + offset settle
                    continue
                if not data:
                    time.sleep(POLL_S)
        log(f"stream finished ({status}): {off} bytes, claude rc {getattr(self, 'claude_rc', None)}")
        return status

    def note_rate_limit(self) -> None:
        """After the stream: was the session REJECTED for a usage limit / an HTTP 429 (usage_limit_rejection)? Recorded
        as receipt daytona.rate_limited; run_resuming waits for the reset and runs the turn again."""
        self.rate_limited = None
        if getattr(self, "claude_rc", 0) != 0 and not getattr(self, "deadline_stop", False):
            self.rate_limited = usage_limit_rejection(self.q / self.stream_name)
        if self.rate_limited:
            self.receipt.setdefault("daytona", {})["rate_limited"] = self.rate_limited
            self.save_receipt()
            log(f"the session was rejected for a usage limit / HTTP 429: {self.rate_limited}")

    # ── 8: collect
    def readonly_roots(self) -> list:
        return [*self.evidence_dirs, *(d for d in (self.inputs, self.references, self.grader) if d),
                *([] if getattr(self, "template_mode", False) else [str(self.oracle)])]

    def download_parts(self) -> list:
        return [str(self.staging).lstrip("/"), de.VM_PROJECTS.lstrip("/"), de.VM_OUT.lstrip("/")]

    def after_hashes(self, after) -> None:
        self.receipt["evidence_sha256_after_vm"] = {d: de.dir_sha256_from_rows(after, d) for d in self.evidence_dirs}
        if self.inputs:
            self.receipt["inputs_sha256_after_vm"] = de.dir_sha256_from_rows(after, self.inputs)
        if self.references:
            self.receipt["references_sha256_after_vm"] = de.dir_sha256_from_rows(after, self.references)
        if self.grader:
            self.receipt["grader_sha256_after_vm"] = de.dir_sha256_from_rows(after, self.grader)

    def collect(self) -> None:
        roots = self.readonly_roots()
        code, out = self.sh("/usr/local/bin/python3 -c " + shlex.quote(de.VM_LISTING_PY) + " "
                            + " ".join(shlex.quote(r) for r in roots), timeout=900, what="after-listing")
        after = json.loads(out.strip().splitlines()[-1])
        self.listing["after_run_readonly"] = after
        jdump(self.listing_path, self.listing)
        self.after_hashes(after)
        parts = self.download_parts()
        cmd = ("cd / && L=''; for p in " + " ".join(shlex.quote(p) for p in parts)
               + '; do [ -e "$p" ] && L="$L $p"; done; tar czf /tmp/out.tgz $L && ls -l /tmp/out.tgz')
        code, out = self.sh(cmd, timeout=900, what="pack")
        local = self.q / "out.tgz"
        self.sb.fs.download_file("/tmp/out.tgz", str(local))
        ex = self.q / "extract"
        if ex.exists():
            shutil.rmtree(ex)
        ex.mkdir()
        refused = []
        with tarfile.open(local, "r:gz") as tf:
            for m in tf.getmembers():
                name = m.name
                if name.startswith("/") or ".." in Path(name).parts or not (m.isfile() or m.isdir()):
                    refused.append(name)
                    continue
                tf.extract(m, ex, filter="data")
        self.receipt["daytona"]["download_refused_members"] = refused
        vm_log = ex / de.VM_OUT.lstrip("/") / "evolver.log"
        streamed = self.q / self.stream_name
        same = vm_log.is_file() and streamed.is_file() and \
            de.sha256_file(vm_log) == de.sha256_file(streamed)
        self.receipt["daytona"]["log_round_trip"] = {
            "streamed_sha256": de.sha256_file(streamed) if streamed.is_file() else None,
            "vm_sha256": de.sha256_file(vm_log) if vm_log.is_file() else None, "equal": same}
        proj = ex / de.VM_PROJECTS.lstrip("/")
        self.receipt["daytona"]["session_audit"] = de.audit_sessions([streamed, proj])
        self.receipt["daytona"]["heldout_hits"] = de.heldout_id_hits(
            [streamed, proj, ex / str(self.staging).lstrip("/")], self.split) if getattr(self, "split", None) else []
        hits, n = de.token_scan([ex, streamed], self.token)
        self.receipt["token_scan"] = {"hits": hits, "scanned_files": n}
        self.receipt["daytona"]["readback_end"] = dle.readback(self.sb)
        self.save_receipt()
        log(f"collected: log round-trip equal={same}; token scan {len(hits)} hit(s) in {n} files; "
            f"refused tar members {len(refused)}")
        if hits:
            raise Refuse(EXIT_TOKEN, f"TOKEN FOUND in {len(hits)} downloaded file(s): {hits[:3]} -- "
                                     f"nothing installed; rotate the credential")
        self.collected = True
        if not same:
            raise Refuse(EXIT_MANIFEST, "streamed log != VM log")

    def delete(self) -> bool:
        if not (self.client and self.sandbox_id):
            return True
        ok = dle.delete_by_id(self.client, self.sandbox_id, self.run_dir.name)
        self.receipt.setdefault("daytona", {})["deleted"] = ok
        self.receipt["daytona"]["deleted_utc"] = dle.now()
        self.save_receipt()
        log(f"DELETED {self.sandbox_id}: {ok}")
        return ok

    # ── 9: install
    def install(self) -> None:
        ex = self.q / "extract"
        os.replace(self.q / "evolver.log", self.log_path)
        err = ex / de.VM_OUT.lstrip("/") / "evolver.stderr"
        if err.is_file():
            shutil.copyfile(err, self.log_path.with_suffix(".stderr"))
        sess = ex / de.VM_PROJECTS.lstrip("/")
        if sess.is_dir():
            dst = self.run_dir / f"e{self.epoch}_evolver_session"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(sess, dst)
            self.receipt["daytona"]["session_transcript_dir"] = str(dst)
        src = ex / str(self.staging).lstrip("/")
        n_skills = len(list(src.rglob("SKILL.md"))) if src.is_dir() else 0
        links = es.symlinks_under(src) if src.is_dir() else []
        self.receipt["daytona"]["staging_downloaded_skills"] = n_skills
        # No refusal of a staged file byte-equal to a benchmark workbook (the evolver holds training workbooks only,
        # so no copy reveals a val/test answer), an EMPTY staging is installed (the solver then runs with no skills),
        # and a turn stopped at its DEADLINE is installed as it stands (evolver_container.stream sets deadline_stop).
        if (getattr(self, "claude_rc", -1) == 0 or getattr(self, "deadline_stop", False)) and src.is_dir() \
                and not links:
            st = self.staging
            if st.exists():
                if any(st.iterdir()):
                    raise Refuse(EXIT_MANIFEST, f"{st} is not empty; refusing to overwrite")
                st.rmdir()
            tmp = st.with_name(f".{st.name}.install")
            if tmp.exists():
                shutil.rmtree(tmp)
            shutil.copytree(src, tmp)
            os.rename(tmp, st)
            self.receipt["daytona"]["staging_installed"] = True
        else:
            self.receipt["daytona"]["staging_installed"] = False
        self.receipt["daytona"]["claude_rc"] = getattr(self, "claude_rc", None)
        self.save_receipt()
        log(f"installed: log -> {self.log_path.name}; staging installed="
            f"{self.receipt['daytona']['staging_installed']} ({n_skills} skill(s))")

    def run(self, gate_argv: list | None = None) -> int:
        if gate_argv:
            env = dict(os.environ, EVOSKILL_TAU3_GATE_ONLY="1")
            r = subprocess.run(gate_argv, cwd=str(REPO), env=env)
            if r.returncode != 0:
                log(f"pre-dispatch gate refused (exit {r.returncode})")
                return r.returncode
        try:
            self.prepare()
        except Refuse as e:
            log(f"REFUSED before any sandbox: {e}")
            return e.code
        rc, status = 0, "done"
        try:
            self.create()
            self.probes()
            self.upload()
            self.launch()
            status = self.stream()
            self.note_rate_limit()
            self.collect()
        except Refuse as e:
            log(f"REFUSED: {e}")
            rc = e.code
        except Exception as e:  # noqa: BLE001
            log(f"CRASH: {type(e).__name__}: {e}")
            rc = 1
        finally:
            # Salvage what exists before the sandbox goes: a failure after launch would otherwise lose the
            # staging, the transcript and the log together.
            if rc and getattr(self, "launched", False) and not getattr(self, "collected", False):
                try:
                    self.collect()
                    log("salvaged the staging, transcript and log into quarantine")
                except Exception as e2:  # noqa: BLE001
                    log(f"salvage collect failed: {type(e2).__name__}: {e2}")
            if not self.delete():
                rc = rc or EXIT_DELETE
        if rc:
            return rc
        if self.install_outputs:
            try:
                self.install()
            except Refuse as e:
                log(f"REFUSED at install: {e}")
                return e.code
        if getattr(self, "deadline_stop", False) and self.DEADLINE_TAKES_STAGING:
            log(f"the turn was stopped at its deadline (claude rc {getattr(self, 'claude_rc', None)}); its staging "
                f"was taken as it stands")
            return 0
        if status == "timeout":
            return EXIT_TIMEOUT
        if getattr(self, "claude_rc", -1) != 0:
            log(f"claude exited {getattr(self, 'claude_rc', None)} on the VM"
                + (" (rejected for a usage limit / HTTP 429)" if self.rate_limited else ""))
            return EXIT_CLAUDE_RC
        return 0


class AORun(EvolverRun):
    DEADLINE_TAKES_STAGING = False       # the adapter's output is not installed on a timeout: a deadline stop fails
    """The AO's prompt-authoring session on its own sandbox from the same image. Uploads ONLY what the AO
    is granted -- the author message, the evidence dirs, the oracle README, SPEC.json, the e{N} RUNTIME
    projection and the previous prompt -- and brings back exactly one artifact, ITER{N}_PROMPT.md, plus the
    stream-json log and the session transcript. Web tools off, egress api.anthropic.com only. The live
    stream sits at .cache/daytona_ao/<run>/e{N}_<ts>/ao.log while the AO runs."""

    prompt_vm_file = f"{de.VM_WORK}/MESSAGE.md"

    def __init__(self, *, message: Path, out: Path, literals: list, log_header: str, **kw):
        super().__init__(**kw)
        self.message, self.out, self.literals, self.log_header = message, out, literals, log_header
        self.stream_name = "ao.log"

    def prepare(self) -> None:
        self.token, self.fd_env, self.credential = load_credential(self.cfg)
        mb = self.message.read_bytes()
        if len(mb) > de.MAX_PROMPT_BYTES:
            raise Refuse(EXIT_USAGE, f"author message is {len(mb)} B > {de.MAX_PROMPT_BYTES}")
        if self.out.exists():
            raise Refuse(EXIT_USAGE, f"{self.out} already exists")
        ok_df, det_df, rec_df = de.dockerfile_check()
        if not ok_df:
            raise Refuse(EXIT_IDENTITY, f"image Dockerfile: {det_df}")
        self.evidence_dirs = [os.path.realpath(str(d)) for d in self.prior]
        self.env = de.vm_env(self.model, env=os.environ)
        env_text = de.render_env_file(self.env).encode()
        rows = [de._row(f"{de.VM_WORK}/MESSAGE.md", "message", str(self.message))]
        for d in self.evidence_dirs:
            for f in sorted(Path(d).rglob("*")):
                if f.is_symlink():
                    raise Refuse(EXIT_MANIFEST, f"symlink in evidence {f}")
                if f.is_file():
                    rows.append(de._row(str(f), "evidence", str(f)))
        readme = Path(self.oracle).absolute() / "README.md"
        rows.append(de._row(str(readme), "oracle", str(readme)))
        for lit in self.literals:
            rows.append(de._row(str(Path(lit).absolute()), "literal", str(Path(lit).absolute())))
        rows.append(de._row(de.VM_ENV_FILE, "env", data=env_text))
        pol = de.policy_settings_text().encode()
        rows.append(de._row(de.VM_POLICY, "policy", data=pol))
        de._refuse_forbidden_sources(rows, getattr(self, "split", None))
        self.rows, self.gen = rows, {de.VM_ENV_FILE: env_text, de.VM_POLICY: pol}
        self.add_dirs = [*self.evidence_dirs, str(self.oracle), str(self.out.parent)]
        counts = {}
        for r in rows:
            counts[r["role"]] = counts.get(r["role"], 0) + 1
        self.inputs = self.references = self.grader = None
        self.receipt = {
            "role": "ao", "sandboxed": True, "containment": "daytona", "profile": None,
            "work_dir": de.VM_WORK, "evidence_dirs": self.evidence_dirs,
            "evidence_sha256_before": es.hashes(self.evidence_dirs),
            "upload_manifest": rows, "upload_counts": counts, "out": str(self.out),
            "daytona": {"driver": str(Path(__file__).relative_to(REPO)), "prefix": de.EVO_PREFIX,
                        "resources_requested": de.RESOURCES, "domain_allow_list": de.ALLOW_DOMAINS,
                        "dockerfile": rec_df, "cc_version": de.CC_VERSION,
                        "cc_sha256_pinned": de.CC_LINUX_X64_SHA, "session_id": self.sid_session,
                        "config": {k: v for k, v in self.cfg.items() if k != "oauth_token_file"},
                        "credential": self.credential, "oauth_token_file": self.credential["token_file"],
                        "vm_env_names": sorted(self.env), "vm_env": self.env},
            "written_utc": dle.now()}
        self.save_receipt()
        # The AO writes exactly one new file into the mirrored run dir: that dir becomes root-owned
        # 1777 (create-only; the sticky bit stops it deleting or replacing the root-owned evidence and
        # literals), and `staging` (chowned to evo in upload) is just its work dir.
        self.writable_dir = self.out.parent
        self.staging = Path(de.VM_WORK)
        log(f"AO manifest: {len(rows)} files {counts}; receipt written before any sandbox exists")

    def claude_argv(self) -> list:
        from src.orchestrator.claude_pins import effort_argv
        argv = ["claude", "-p", "__PROMPT__", "--model", self.model, "--session-id", self.sid_session,
                *effort_argv(self.model, os.environ)]
        for d in self.add_dirs:
            argv += ["--add-dir", d]
        return argv + ["--disallowed-tools", "WebFetch", "WebSearch", "--thinking-display",
                       "summarized", "--dangerously-skip-permissions", "--output-format",
                       "stream-json", "--verbose"]

    def readonly_roots(self) -> list:
        return [*self.evidence_dirs, str(self.oracle)]

    def download_parts(self) -> list:
        return [str(self.out).lstrip("/"), de.VM_PROJECTS.lstrip("/"), de.VM_OUT.lstrip("/")]

    def after_hashes(self, after) -> None:
        self.receipt["evidence_sha256_after_vm"] = {d: de.dir_sha256_from_rows(after, d)
                                                    for d in self.evidence_dirs}

    def install(self) -> None:
        ex = self.q / "extract"
        body = (self.q / self.stream_name).read_bytes()
        tmp = self.log_path.with_name(f".{self.log_path.name}.tmp")
        tmp.write_bytes(self.log_header.encode() + body)
        os.replace(tmp, self.log_path)
        sess = ex / de.VM_PROJECTS.lstrip("/")
        if sess.is_dir():
            dst = self.run_dir / f"e{self.epoch}_ao_session"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(sess, dst)
            self.receipt["daytona"]["session_transcript_dir"] = str(dst)
        src = ex / str(self.out).lstrip("/")
        wrote = src.is_file() and not src.is_symlink()
        if getattr(self, "claude_rc", -1) == 0 and wrote:
            if self.out.exists():
                raise Refuse(EXIT_MANIFEST, f"{self.out} appeared on the host during the run")
            shutil.copyfile(src, self.out)
        self.receipt["daytona"].update(claude_rc=getattr(self, "claude_rc", None), out_written=wrote,
                                       out_sha256=de.sha256_file(src) if wrote else None)
        self.save_receipt()
        log(f"AO installed: log -> {self.log_path.name}; {self.out.name} written={wrote}")


def cmd_ao(a) -> int:
    run_dir = Path(a.run_dir).resolve()
    # EVERY path absolute: the loop runs with a relative --run-dir, and a relative --out would be packed from
    # `/outputs/...` and granted relative to the VM's cwd.
    a.message, a.out, a.log, a.oracle = (str(Path(x).absolute()) for x in (a.message, a.out, a.log, a.oracle))
    a.evidence = [str(Path(x).absolute()) for x in a.evidence]
    a.literal = [str(Path(x).absolute()) for x in (a.literal or [])]
    hold_dispatch_lock(run_dir)

    def make_run():
        q = REPO / ".cache/daytona_ao" / run_dir.name / f"e{a.epoch}_{time.strftime('%Y%m%dT%H%M%S')}"
        run = AORun(message=Path(a.message), out=Path(a.out), literals=[Path(x) for x in (a.literal or [])],
                    log_header=f"=== prompt author: epoch {a.epoch} model {a.model} host daytona\n",
                    epoch=int(a.epoch), prompt=Path(a.message), digests=Path(a.evidence[-1]),
                    staging=Path(a.out).parent, run_dir=run_dir, prior=list(a.evidence), model=a.model,
                    oracle=Path(a.oracle), cfg=_cfg(), timeout_s=int(a.timeout_s),
                    receipt_path=run_dir / f"e{a.epoch}_AO_DAYTONA.json",
                    trace_path=q / "unused_trace.json", listing_path=run_dir / f"e{a.epoch}_AO_VM_LISTING.json",
                    log_path=Path(a.log), quarantine=q, mode="all_epochs")
        if a.split:
            run.split = Path(a.split).absolute()
        log(f"DAYTONA AO e{a.epoch}: run {run_dir.name}, model {a.model}, {len(a.evidence)} evidence dir(s), "
            f"quarantine {q}")
        return run
    return run_resuming(make_run, gate_argv=None, what=f"the AO's epoch-{a.epoch} turn")


def _cfg() -> dict:
    raw = os.environ.get("EVOSKILL_EVOLVER_DAYTONA_JSON", "")
    try:
        cfg = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        raise SystemExit(f"EVOSKILL_EVOLVER_DAYTONA_JSON is not JSON: {raw[:80]!r}")
    if not isinstance(cfg, dict):
        raise SystemExit("EVOSKILL_EVOLVER_DAYTONA_JSON must be an object")
    return cfg


def cmd_dispatch(a) -> int:
    run_dir = Path(os.environ.get("EVOSKILL_TAU3_RUN_DIR") or "").resolve()
    if not run_dir.is_dir():
        log("EVOSKILL_TAU3_RUN_DIR unset or missing")
        return EXIT_USAGE
    epoch = int(a.epoch)
    prompt, digests, staging = (Path(p) if Path(p).is_absolute() else (REPO / p)
                                for p in (a.prompt, a.digests, a.staging))
    model = os.environ.get("EVOSKILL_EVOLVER_MODEL", "")
    if not model.startswith("claude-"):
        log(f"EVOSKILL_EVOLVER_MODEL must be an explicit claude-* id, got {model!r}")
        return EXIT_USAGE
    mode = os.environ.get("EVOSKILL_EVOLVER_EVIDENCE", "current_epoch")
    prior = [p for p in os.environ.get("EVOSKILL_PRIOR_DIGESTS", "").split(os.pathsep) if p] \
        if mode == "all_epochs" else []
    oracle = Path(os.environ.get("EVOSKILL_TAU3_ORACLE") or (run_dir / "oracle"))
    if not oracle.is_absolute():
        oracle = REPO / oracle
    timeout_s = int(os.environ.get("EVOSKILL_EVOLVER_TIMEOUT_S") or 7200)
    hold_dispatch_lock(run_dir)
    q0 = REPO / ".cache/daytona_evolver" / run_dir.name / f"e{epoch}_{time.strftime('%Y%m%dT%H%M%S')}"
    scratch = q0 / "gate_staging"

    def make_run():
        q = REPO / ".cache/daytona_evolver" / run_dir.name / f"e{epoch}_{time.strftime('%Y%m%dT%H%M%S')}"
        log(f"DAYTONA EVOLVER e{epoch}: run {run_dir.name}, model {model}, evidence {mode} "
            f"({len(prior) + 1} dir(s)), timeout {timeout_s}s, quarantine {q}")
        return EvolverRun(epoch=epoch, prompt=prompt, digests=digests, staging=staging, run_dir=run_dir,
                          prior=prior, model=model, oracle=oracle, cfg=_cfg(), timeout_s=timeout_s,
                          receipt_path=run_dir / f"e{epoch}_EVOLVER_CONTAINMENT.json",
                          trace_path=run_dir / f"tau3_evolver_e{epoch}.trace.json",
                          listing_path=run_dir / f"e{epoch}_EVOLVER_VM_LISTING.json",
                          log_path=run_dir / f"tau3_evolver_e{epoch}.log", quarantine=q, mode=mode)
    gate = ["bash", str(REPO / "scripts/tau3_evolver.sh"), str(epoch), str(prompt), str(digests), str(scratch)]
    return run_resuming(make_run, gate_argv=gate, what=f"the evolver's epoch-{epoch} turn")


def cmd_probe(a) -> int:
    """Live smoke test on the real e1 upload bundle with a tiny prompt; writes only under
    .cache/daytona_evolver/_probe_<ts>/ (+ the ledger). Never touches outputs/."""
    run_dir = Path(a.run_dir).resolve()
    hold_dispatch_lock(run_dir)
    q = REPO / ".cache/daytona_evolver" / f"_probe_{time.strftime('%Y%m%dT%H%M%S')}"
    q.mkdir(parents=True)
    staging = q / "probe_staging"
    prompt = q / "PROBE_PROMPT.md"
    g = de.evolver_grants(run_dir)
    granted = [str(d) for k, d in (("train_inputs", de.inputs_dir(run_dir)),
                                   ("train_references", de.references_dir(run_dir))) if g[k]]
    prompt.write_text(
        "This is an infrastructure probe, not an evolution task. Do exactly these steps and nothing else.\n"
        + (f"1. Run: ls {' '.join(granted)} | head -3\n" if granted else "1. Run: echo no-granted-dirs\n")
        + "2. Run: python3 -c 'import openpyxl,pandas;print(openpyxl.__version__,pandas.__version__)'\n"
        "3. Run: soffice --version\n"
        "4. Run: env | cut -d= -f1 | grep -iE 'token|oauth|api_key' ; echo names-done\n"
        "5. Run: grep -l sk-ant- /proc/*/environ 2>/dev/null | wc -l\n"
        f"6. Run: touch {run_dir}/digests_e1_train/_probe_write_test 2>&1 ; echo rc=$?\n"
        "7. Run: timeout 60 claude -p 'say hi' --model claude-haiku-4-5-20251001 2>&1 | head -3 ; echo rc=$?\n"
        f"8. Write the file {staging}/probe-skill/SKILL.md containing exactly:\n"
        "---\nname: probe-skill\ndescription: infrastructure probe\n---\nprobe\n"
        "9. Reply with the single word DONE.\n")
    # The run's declared split (SPEC.json split_path; the loop exports the same), else the SSB example's split.
    _spec = run_dir / "SPEC.json"
    _declared = json.loads(_spec.read_text()).get("spec", {}).get("split_path") if _spec.is_file() else None
    os.environ.setdefault("EVOSKILL_TAU3_EVOLVER_SPLIT", _declared or str(REPO / "examples/ssbv2/split.json"))
    cfg = _cfg()                          # the credential: cc_account.resolve (declared, env token file, or API key)
    run = EvolverRun(epoch=1, prompt=prompt, digests=run_dir / "digests_e1_train", staging=staging,
                     run_dir=run_dir, prior=[], model=a.model, oracle=run_dir / "oracle", cfg=cfg,
                     timeout_s=900, receipt_path=q / "receipt.json", trace_path=q / "trace.json",
                     listing_path=q / "listing.json", log_path=q / "probe_evolver.log", quarantine=q / "q",
                     max_turns=16, install=False, mode="all_epochs")
    rc = run.run(gate_argv=None)
    ok_r = ok_m = None
    if rc == 0:
        shutil.copyfile(q / "q" / "evolver.log", q / "probe_evolver.log")
        ok_r, det_r = de.check_daytona_receipt(run.receipt, log=q / "probe_evolver.log",
                                               require_libreoffice=cfg.get("require_libreoffice"))
        c = de.check_upload_manifest(run_dir=run_dir, epoch=1, split=run.split,
                                     evidence_dirs=run.evidence_dirs, prev_library=run.prev_lib,
                                     prompt=prompt, oracle=run_dir / "oracle", staging=staging,
                                     receipt=run.receipt, listing=run.listing)
        ok_m = c.status == "PASS"
        from src.orchestrator.claude_pins import check_session
        cs = check_session(q / "probe_evolver.log", a.model)
        li = de.log_init_and_tools(q / "probe_evolver.log")
        rep = {"rc": rc, "receipt_check": [ok_r, det_r], "manifest_check": [c.status, c.detail],
               "check_session": cs, "init": {k: v for k, v in li.items() if k != "tools"},
               "init_tools": li.get("tools"),
               "snapshot_ref": (run.receipt.get("daytona", {}).get("readback_create") or {}).get("snapshot_ref"),
               "libreoffice": (run.receipt.get("daytona", {}).get("identity") or {}).get("libreoffice")}
        jdump(q / "PROBE_REPORT.json", rep)
        log("PROBE REPORT: " + json.dumps({k: rep[k] for k in ("rc", "receipt_check", "manifest_check",
                                                              "snapshot_ref", "libreoffice")}, default=str))
        log(f"check_session: {json.dumps(cs, default=str)[:600]}")
    log(f"probe dir {q}; rc {rc}")
    return rc if rc else (0 if (ok_r and ok_m) else 1)


def cmd_cleanup(a) -> int:
    """Delete the run's evo-prefix sandboxes (created with its name as their `run` / `tag` label), refused while a
    dispatch of the run is live."""
    run_dir = Path(a.run_dir).resolve()
    lock = exclusive_dispatch_lock(run_dir)
    if lock is None:
        log(f"a dispatch of {run_dir.name} is live; refusing cleanup")
        return EXIT_LIVE
    d = dle.client()
    ours = []
    for sb in d.list():
        labels = getattr(sb, "labels", None) or {}
        if str(sb.name).startswith(dle.PREFIX) and run_dir.name in (labels.get("run"), labels.get("tag")):
            ours.append({"id": sb.id, "name": sb.name})
    for r in ours:
        ok = dle.delete_by_id(d, r["id"], "cleanup")
        log(f"deleted {r['id']} {r['name']}: {ok}")
    log(f"{len(ours)} evo-prefix sandbox(es) of {run_dir.name} found; none remain" if ours
        else f"no evo-prefix sandbox of {run_dir.name} listed")
    lock.close()
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "probe":
        ap = argparse.ArgumentParser(prog="tau3_evolver_daytona.py probe")
        ap.add_argument("--run-dir", required=True)
        ap.add_argument("--model", default="claude-opus-5-5")
        return cmd_probe(ap.parse_args(argv[1:]))
    if argv and argv[0] == "cleanup":
        ap = argparse.ArgumentParser(prog="tau3_evolver_daytona.py cleanup")
        ap.add_argument("--run-dir", required=True)
        return cmd_cleanup(ap.parse_args(argv[1:]))
    if argv and argv[0] == "ao":
        ap = argparse.ArgumentParser(prog="tau3_evolver_daytona.py ao")
        for k in ("--run-dir", "--epoch", "--message", "--out", "--log", "--model", "--oracle"):
            ap.add_argument(k, required=True)
        ap.add_argument("--evidence", action="append", required=True)
        ap.add_argument("--literal", action="append")
        ap.add_argument("--split")
        ap.add_argument("--timeout-s", default="3600")
        return cmd_ao(ap.parse_args(argv[1:]))
    ap = argparse.ArgumentParser(prog="tau3_evolver_daytona.py")
    for k in ("epoch", "prompt", "digests", "staging"):
        ap.add_argument(k)
    return cmd_dispatch(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
