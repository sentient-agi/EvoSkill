"""Egress allowlist for the evolver's `claude` child. Never launches `claude`.

Pinned, each by EXECUTION rather than by reading text wherever that is possible:
  * scripts/evolver_egress_proxy.py — the allowlist CONNECT proxy: an allowed host tunnels bytes
    both ways; a refused host, an IP literal, a wrong port and plain HTTP get 403 and the upstream
    listener sees NO connection; one JSON log line per request; a one-way stream outlives the idle
    timeout while a silent tunnel does not; many tunnels at once; exit when the watched pid dies,
    after the watch-pid file hands it a new pid, and at --max-seconds. Loaded and run BY PATH
    (EVOSKILL_TEST_EGRESS_PROXY) so a mutated copy can be tested.
  * evolver_dispatch.build_evolver_sandbox_profile's egress section and egress_probes — under a
    real sandbox-exec: a throwaway loopback port, an AF_UNIX socket and off-host IP literals are
    EPERM, the proxy port is not; the section is LAST, and a broad network-outbound allow placed
    AFTER it measurably re-opens outbound while the same rule before it does not
    (EVOSKILL_TEST_DISPATCH).

LOCAL ONLY. Every listener is on 127.0.0.1 or an AF_UNIX path. Off-host probes use 127.0.0.2 and
::ffff:127.0.0.2, which the kernel profile treats as NOT localhost (measured) and which never leave
this host even when a broken profile lets the connect through. Refused hostnames are `localhost`,
so a proxy mutant that "allows everything" resolves nothing off-host either.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXROOT = Path(tempfile.gettempdir()) / "fix-egress-fixtures"
PROXY = Path(os.environ.get("EVOSKILL_TEST_EGRESS_PROXY",
                            str(ROOT / "scripts/evolver_egress_proxy.py")))


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PX = _load(PROXY, "egress_proxy_under_test")
DISPATCH = _load(Path(os.environ.get("EVOSKILL_TEST_DISPATCH",
                                     str(ROOT / "src/orchestrator/evolver_dispatch.py"))),
                 "evolver_dispatch_egress_under_test")

SANDBOX = shutil.which("sandbox-exec") is not None
needs_sandbox = pytest.mark.skipif(not SANDBOX, reason="sandbox-exec unavailable on this host")
LOCAL_OFFHOST_V4 = "127.0.0.2"          # EPERM under the profile; stays on-host when not denied
LOCAL_OFFHOST_V6 = "::ffff:127.0.0.2"


@pytest.fixture
def fx():
    FIXROOT.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix="egr_", dir=str(FIXROOT))).resolve()
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ══ local plumbing ══════════════════════════════════════════════════════════════════════════════
class Upstream:
    """A TCP server on 127.0.0.1. Counts accepted connections. Default: echoes b"ECHO:" + data.
    trickle=(n, every_s): sends one byte every every_s seconds, n times, then closes."""

    def __init__(self, greet: bytes = b"", trickle=None):
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(64)
        self.port = self.srv.getsockname()[1]
        self.accepted = 0
        self.received = b""
        self.greet, self.trickle = greet, trickle
        self._lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            try:
                c, _ = self.srv.accept()
            except OSError:
                return
            with self._lock:
                self.accepted += 1
            threading.Thread(target=self._serve, args=(c,), daemon=True).start()

    def _serve(self, c):
        try:
            if self.greet:
                c.sendall(self.greet)
            if self.trickle:
                n, every = self.trickle
                for _ in range(n):
                    time.sleep(every)
                    c.sendall(b".")
                return
            while True:
                d = c.recv(65536)
                if not d:
                    return
                with self._lock:
                    self.received += d
                c.sendall(b"ECHO:" + d)
        except OSError:
            pass
        finally:
            c.close()

    def close(self):
        self.srv.close()


@contextmanager
def running_proxy(tmp: Path, allow: str, *, allowed_port=None, watch_pid=None, watch_file=None,
                  max_seconds=120, idle=1800.0, poll=0.2, expect_ready=True, preexec_fn=None):
    ready = tmp / f"ready.{time.time_ns()}"
    log = tmp / "egress.jsonl"
    argv = [sys.executable, str(PROXY), "--allow", allow, "--log", str(log),
            "--ready-file", str(ready), "--max-seconds", str(max_seconds),
            "--idle-timeout", str(idle), "--poll", str(poll)]
    if allowed_port:
        argv += ["--allowed-port", str(allowed_port)]
    if watch_pid is not False:
        argv += ["--watch-pid", str(watch_pid or os.getpid())]
    if watch_file:
        argv += ["--watch-pid-file", str(watch_file)]
    errf = open(tmp / f"proxy.{time.time_ns()}.err", "w+")
    p = subprocess.Popen(argv, stdout=errf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         preexec_fn=preexec_fn)
    try:
        port = None
        for _ in range(100):
            if ready.is_file() and ready.stat().st_size:
                port = json.loads(ready.read_text())["port"]
                break
            if p.poll() is not None:
                break
            time.sleep(0.05)
        if expect_ready:
            errf.seek(0)
            assert port, f"proxy never became ready: {errf.read()}"
        yield p, port, log, ready
    finally:
        if p.poll() is None:
            p.send_signal(signal.SIGTERM)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
        errf.close()


def via(port: int, request: bytes):
    """Send a raw request to the proxy; return (socket, status line)."""
    s = socket.create_connection(("127.0.0.1", port), timeout=10)
    s.sendall(request)
    buf = b""
    while b"\r\n\r\n" not in buf:
        c = s.recv(4096)
        if not c:
            break
        buf += c
    return s, buf.split(b"\r\n", 1)[0].decode("latin-1")


def connect_req(target: str) -> bytes:
    return f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode()


def log_lines(log: Path, n: int, timeout: float = 5.0) -> list:
    deadline = time.time() + timeout
    while time.time() < deadline:
        rows = [json.loads(x) for x in log.read_text().splitlines()] if log.is_file() else []
        if len(rows) >= n:
            return rows
        time.sleep(0.05)
    return [json.loads(x) for x in log.read_text().splitlines()] if log.is_file() else []


def recv_until(s: socket.socket, want: bytes, timeout: float = 5.0) -> bytes:
    s.settimeout(timeout)
    buf = b""
    while want not in buf:
        c = s.recv(4096)
        if not c:
            break
        buf += c
    return buf


# ══ the proxy ═══════════════════════════════════════════════════════════════════════════════════
def test_proxy_tunnels_an_allowed_host_both_ways(fx):
    up = Upstream(greet=b"HELLO\n")
    try:
        with running_proxy(fx, "localhost", allowed_port=up.port) as (_p, port, log, _r):
            s, status = via(port, connect_req(f"localhost:{up.port}"))
            assert status.startswith("HTTP/1.1 200"), status
            assert recv_until(s, b"HELLO\n") == b"HELLO\n"             # upstream -> client
            s.sendall(b"ping")
            assert recv_until(s, b"ECHO:ping").endswith(b"ECHO:ping")   # client -> upstream -> back
            s.close()
            rows = log_lines(log, 1)
            assert up.accepted == 1 and up.received == b"ping"
            r = rows[-1]
            assert r["allowed"] is True and r["method"] == "CONNECT"
            assert r["host"] == "localhost" and r["port"] == up.port
            assert r["bytes_up"] == 4 and r["bytes_down"] == len(b"HELLO\nECHO:ping")
    finally:
        up.close()


def test_proxy_refuses_a_host_not_on_the_list_without_connecting(fx):
    up = Upstream()
    try:
        # `localhost` resolves on-host, so a proxy that wrongly allowed it would CONNECT here
        with running_proxy(fx, "api.anthropic.com", allowed_port=up.port) as (_p, port, log, _r):
            for target in (f"localhost:{up.port}", f"LOCALHOST.:{up.port}",
                           f"sub.localhost:{up.port}"):
                s, status = via(port, connect_req(target))
                s.close()
                assert status.startswith("HTTP/1.1 403"), (target, status)
            rows = log_lines(log, 3)
            assert all(r["allowed"] is False and r["reason"] == "host not on the allowlist"
                       for r in rows[-3:]), rows
            time.sleep(0.2)
            assert up.accepted == 0, "a refused CONNECT reached the upstream"
    finally:
        up.close()


def test_proxy_refuses_an_ip_literal_and_a_wrong_port_without_connecting(fx):
    up = Upstream()
    try:
        with running_proxy(fx, "localhost") as (_p, port, log, _r):        # allowed port 443
            for target, why in ((f"127.0.0.1:{up.port}", "host not on the allowlist"),
                                (f"[::1]:{up.port}", "host not on the allowlist"),
                                (f"localhost:{up.port}", "port is not 443")):
                s, status = via(port, connect_req(target))
                s.close()
                assert status.startswith("HTTP/1.1 403"), (target, status)
            rows = log_lines(log, 3)
            assert [r["reason"] for r in rows[-3:]] == ["host not on the allowlist",
                                                        "host not on the allowlist",
                                                        "port is not 443"], rows
            time.sleep(0.2)
            assert up.accepted == 0
    finally:
        up.close()


def test_proxy_refuses_plain_http_and_other_methods(fx):
    up = Upstream()
    try:
        with running_proxy(fx, "localhost", allowed_port=up.port) as (_p, port, log, _r):
            for req in (f"GET http://localhost:{up.port}/ HTTP/1.1\r\nHost: localhost\r\n\r\n",
                        f"POST http://localhost:{up.port}/x HTTP/1.1\r\nContent-Length: 0\r\n\r\n",
                        f"connect localhost:{up.port} HTTP/1.1\r\n\r\n"):
                s, status = via(port, req.encode())
                s.close()
                assert status.startswith("HTTP/1.1 403"), (req, status)
            s, status = via(port, b"CONNECT nonsense\r\n\r\n")
            s.close()
            assert status.startswith("HTTP/1.1 400"), status
            rows = log_lines(log, 4)
            assert [r["reason"] for r in rows[-4:]] == ["not CONNECT"] * 3 + [
                "malformed request line"], rows
            time.sleep(0.2)
            assert up.accepted == 0
    finally:
        up.close()


def test_proxy_logs_one_json_line_per_request(fx):
    up = Upstream()
    try:
        with running_proxy(fx, "localhost", allowed_port=up.port) as (_p, port, log, _r):
            s, _ = via(port, connect_req(f"localhost:{up.port}"))
            s.sendall(b"x")
            recv_until(s, b"ECHO:x")
            s.close()
            assert len(log_lines(log, 1)) == 1          # a tunnel's line lands when it closes
            s, _ = via(port, connect_req(f"example.invalid:{up.port}"))
            s.close()
            rows = log_lines(log, 2)
            assert len(rows) == 2, rows
            for r in rows:
                for k in ("time", "ts", "method", "target", "host", "port", "allowed", "reason",
                          "bytes_up", "bytes_down", "duration_s", "proxy_pid"):
                    assert k in r, (k, r)
            assert [r["allowed"] for r in rows] == [True, False]
            assert rows[1]["host"] == "example.invalid" and "upstream_ip" not in rows[1]
    finally:
        up.close()


def test_proxy_keeps_a_one_way_stream_open_past_the_idle_timeout(fx):
    """A model stream is ONE-directional for minutes: idle is measured across both directions."""
    up = Upstream(trickle=(8, 0.3))                  # 2.4 s of server->client trickle
    try:
        with running_proxy(fx, "localhost", allowed_port=up.port, idle=1.0) as (_p, port, _l, _r):
            s, status = via(port, connect_req(f"localhost:{up.port}"))
            assert status.startswith("HTTP/1.1 200"), status
            got = recv_until(s, b"........", timeout=10)
            s.close()
            assert got == b"........", got
    finally:
        up.close()


def test_proxy_closes_a_silent_tunnel_after_the_idle_timeout(fx):
    up = Upstream()
    try:
        with running_proxy(fx, "localhost", allowed_port=up.port, idle=1.0) as (_p, port, log, _r):
            s, status = via(port, connect_req(f"localhost:{up.port}"))
            assert status.startswith("HTTP/1.1 200"), status
            s.settimeout(8)
            t0 = time.time()
            assert s.recv(10) == b"", "the tunnel should close on idle"
            assert time.time() - t0 < 6
            s.close()
            assert log_lines(log, 1)[-1]["allowed"] is True
    finally:
        up.close()


def test_proxy_handles_many_concurrent_tunnels(fx):
    up = Upstream()
    try:
        with running_proxy(fx, "localhost", allowed_port=up.port) as (_p, port, log, _r):
            socks = []
            for _ in range(24):
                s, status = via(port, connect_req(f"localhost:{up.port}"))
                assert status.startswith("HTTP/1.1 200"), status
                socks.append(s)
            for i, s in enumerate(socks):
                s.sendall(b"n%d" % i)
            for i, s in enumerate(socks):
                assert recv_until(s, b"ECHO:n%d" % i).endswith(b"ECHO:n%d" % i)
            for s in socks:
                s.close()
            assert up.accepted == 24
            assert sum(r["allowed"] for r in log_lines(log, 24)) == 24
    finally:
        up.close()


def _sleeper(seconds: int = 60) -> subprocess.Popen:
    return subprocess.Popen(["/bin/sleep", str(seconds)])


def test_proxy_exits_when_the_watched_pid_dies(fx):
    victim = _sleeper()
    try:
        with running_proxy(fx, "api.anthropic.com", watch_pid=victim.pid) as (p, _port, _l, ready):
            time.sleep(0.5)
            assert p.poll() is None, "exited while the watched pid was alive"
            victim.kill()
            victim.wait()
            p.wait(timeout=10)
            assert p.returncode == 0
            assert not ready.exists(), "the ready file must not outlive the proxy"
    finally:
        if victim.poll() is None:
            victim.kill()


def test_proxy_follows_the_watch_pid_file(fx):
    """The launcher hands claude's pid over AFTER starting it: from then on the launcher's own
    death must not stop the proxy, and claude's must."""
    launcher, evolver = _sleeper(), _sleeper()
    wf = fx / "watch.pid"
    try:
        with running_proxy(fx, "api.anthropic.com", watch_pid=launcher.pid,
                           watch_file=wf) as (p, _port, _l, _r):
            wf.write_text(f"{evolver.pid}\n")
            time.sleep(0.8)                          # several polls: it has switched by now
            launcher.kill()
            launcher.wait()
            time.sleep(1.0)
            assert p.poll() is None, "exited with the LAUNCHER after claude's pid was handed over"
            evolver.kill()
            evolver.wait()
            p.wait(timeout=10)
            assert not wf.exists()
    finally:
        for x in (launcher, evolver):
            if x.poll() is None:
                x.kill()


def test_proxy_logs_a_tunnel_that_is_still_open_when_it_exits(fx):
    """A tunnel's line is written when it closes; the proxy exits within a poll of the evolver,
    possibly before a long-lived tunnel closed. Its line must still land, never be dropped."""
    up = Upstream()
    victim = _sleeper()
    try:
        with running_proxy(fx, "localhost", allowed_port=up.port,
                           watch_pid=victim.pid) as (p, port, log, _r):
            s, status = via(port, connect_req(f"localhost:{up.port}"))
            assert status.startswith("HTTP/1.1 200"), status
            s.sendall(b"abc")
            recv_until(s, b"ECHO:abc")
            victim.kill()
            victim.wait()
            p.wait(timeout=10)
            s.close()
        rows = log_lines(log, 1)
        assert len(rows) == 1 and rows[0]["allowed"] is True, rows
        assert rows[0]["reason"].startswith("still open at proxy exit"), rows[0]
        assert rows[0]["bytes_up"] == 3 and rows[0]["bytes_down"] == len(b"ECHO:abc")
    finally:
        up.close()
        if victim.poll() is None:
            victim.kill()


def test_proxy_does_not_hold_a_half_closed_tunnel_open(fx, monkeypatch):
    """One side gone, the other silent: drained for HALF_CLOSE_GRACE_S, then closed — not kept
    for the whole idle timeout. In-process, on socket pairs, with the grace shortened."""
    monkeypatch.setattr(PX, "HALF_CLOSE_GRACE_S", 0.5)
    px = PX.Proxy("localhost", str(fx / "l.jsonl"), idle_timeout=60)
    try:
        client, c = socket.socketpair()
        upstream, u = socket.socketpair()
        rec = {"up": 0, "down": 0}
        t = threading.Thread(target=px._tunnel, args=(c, u, rec), daemon=True)
        t.start()
        client.sendall(b"hi")
        assert upstream.recv(10) == b"hi"
        client.close()                               # client side gone; upstream stays silent
        t.join(timeout=5)
        assert not t.is_alive(), "a half-closed tunnel was held open"
        assert rec == {"up": 2, "down": 0}
        for x in (upstream, c, u):
            x.close()
    finally:
        px.stop()


def test_proxy_exits_at_max_seconds(fx):
    t0 = time.time()
    with running_proxy(fx, "api.anthropic.com", max_seconds=1.5) as (p, _port, _l, _r):
        p.wait(timeout=15)
    assert time.time() - t0 < 12


def test_proxy_refuses_to_start_without_a_watch_or_with_a_bad_allowlist(fx):
    for allow in ("https://api.anthropic.com", "api.anthropic.com:443", "*", "*.anthropic.com",
                  "", "api.anthropic.com,,claude.ai", ".com", "api.anthropic.com/v1"):
        r = subprocess.run([sys.executable, str(PROXY), "--allow", allow, "--log", str(fx / "l"),
                            "--ready-file", str(fx / "r"), "--watch-pid", str(os.getpid())],
                           capture_output=True, text=True, timeout=30)
        assert r.returncode == 2 and "REFUSING" in r.stderr, (allow, r.returncode, r.stderr)
    r = subprocess.run([sys.executable, str(PROXY), "--allow", "claude.ai", "--log", str(fx / "l"),
                        "--ready-file", str(fx / "r")], capture_output=True, text=True, timeout=30)
    assert r.returncode == 2 and "watch" in r.stderr
    assert not (fx / "r").exists()


def test_allowlist_matching_rules():
    exact, suf = PX.parse_allowlist("API.Anthropic.com,claude.ai,.example.org,10.0.0.1")
    ok = lambda h: PX.host_allowed(h, exact, suf)   # noqa: E731
    assert ok("api.anthropic.com") and ok("API.ANTHROPIC.COM") and ok("api.anthropic.com.")
    assert ok("claude.ai") and ok("sub.example.org") and ok("a.b.example.org") and ok("10.0.0.1")
    for h in ("anthropic.com", "evil-api.anthropic.com", "api.anthropic.com.evil.com",
              "xclaude.ai", "claude.ai.evil", "example.org", "evilexample.org", "10.0.0.2",
              "1.1.1.1", "::1", "", "api.anthropic.com..", "a_b.claude.ai", "claude.ai:443"):
        assert not ok(h), h
    assert PX.parse_connect_target("api.anthropic.com:443") == ("api.anthropic.com", 443)
    assert PX.parse_connect_target("[::1]:443") == ("::1", 443)
    for t in ("api.anthropic.com", "a:b:443", "http://x:443", "x:99999", "user@x:443", "x:0"):
        assert PX.parse_connect_target(t) is None, t


@pytest.mark.parametrize("url,kind,extra", [
    ("http://127.0.0.1:8100", "local", {"port": 8100}),
    ("http://localhost:8100/v1", "local", {"port": 8100}),
    ("https://api.anthropic.com", "proxied", {"host": "api.anthropic.com"}),
    ("https://openrouter.ai/api", "refused", {}),
    ("http://api.anthropic.com", "refused", {}),
    ("https://api.anthropic.com:8443", "refused", {}),
    ("ftp://x", "refused", {}),
])
def test_classify_base_url(url, kind, extra):
    r = PX.classify_base_url(url, "api.anthropic.com,claude.ai")
    assert r["kind"] == kind, r
    for k, v in extra.items():
        assert r[k] == v, r
    p = subprocess.run([sys.executable, str(PROXY), "--classify-url", url, "--allow",
                        "api.anthropic.com,claude.ai"], capture_output=True, text=True, timeout=30)
    assert p.returncode == (3 if kind == "refused" else 0), p.stdout + p.stderr
    assert json.loads(p.stdout)["kind"] == kind


# ══ the profile ═════════════════════════════════════════════════════════════════════════════════
def _job(fx: Path, ports):
    (fx / "work").mkdir(exist_ok=True)
    (fx / "dg").mkdir(exist_ok=True)
    (fx / "dg" / "t.md").write_text("x")
    return DISPATCH.EvolverJob(epoch=1, work_dir=fx / "work", prompt="", model="",
                               log_path=fx / "e.log", grants=[fx / "dg"], hide=[fx],
                               egress_ports=ports)


def _rules(profile: str) -> list:
    return [x.strip() for x in profile.splitlines() if x.strip() and not x.strip().startswith(";")]


def test_profile_egress_rules_are_last_and_in_order(fx):
    body = DISPATCH.build_evolver_sandbox_profile(_job(fx, [40001, 8100]))
    rules = _rules(body)
    i = rules.index("(deny network-outbound)")
    assert rules[i:] == ['(deny network-outbound)',
                         '(allow network-outbound (remote ip "localhost:40001"))',
                         '(allow network-outbound (remote ip "localhost:8100"))'], rules[i:]
    # nothing after the deny may match network-outbound broadly, and nothing BEFORE it opens more
    # than (allow default) already does (which the deny overrides, being later)
    assert rules.index("(allow default)") < i
    assert sum("network" in r for r in rules) == 3, [r for r in rules if "network" in r]


def test_profile_allows_a_declared_local_model_port(fx):
    body = DISPATCH.build_evolver_sandbox_profile(_job(fx, [40001, 8100, 8100]))
    assert body.count('(allow network-outbound (remote ip "localhost:8100"))') == 1
    assert '(allow network-outbound (remote ip "localhost:40001"))' in body


def test_profile_off_mode_has_no_network_rule_and_says_so(fx):
    body = DISPATCH.build_evolver_sandbox_profile(_job(fx, None))
    assert not [r for r in _rules(body) if "network" in r]
    assert "EGRESS: NOT restricted by this profile" in body


def test_profile_rejects_bad_ports(fx):
    for bad in ([0], [65536], ["8100"], [-1]):
        with pytest.raises(ValueError):
            DISPATCH.build_evolver_sandbox_profile(_job(fx, bad))


def _probe(fx, prof: Path, proxy_port: int, extra=()):
    return {r["probe"]: r for r in DISPATCH.egress_probes(
        prof, proxy_port, extra, probe_host="localhost", probe_ip=LOCAL_OFFHOST_V4,
        probe_ip6=LOCAL_OFFHOST_V6)}


@needs_sandbox
def test_egress_probes_pass_under_the_emitted_profile(fx):
    with running_proxy(fx, "api.anthropic.com") as (_p, port, log, _r):
        prof = fx / "p.sb"
        prof.write_text(DISPATCH.build_evolver_sandbox_profile(_job(fx, [port, 1])))
        rows = _probe(fx, prof, port, extra=[1])
        assert all(r["ok"] for r in rows.values()), rows
        assert rows["a_other_loopback_port"]["got"] == "EPERM"
        assert rows["e_unix_socket"]["got"] == "EPERM"             # AF_UNIX is network-outbound
        assert rows["d_offhost_ipv4"]["got"] == "EPERM"
        assert rows["d_offhost_ipv6"]["got"] == "EPERM"
        assert rows["f_local_port_1"]["got"] == "ECONNREFUSED"      # allowed; nothing listens
        refused = [r for r in log_lines(log, 3) if r["method"] == "CONNECT"]
        assert {r["host"] for r in refused} == {"localhost", "127.0.0.1"}
        assert not any(r["allowed"] for r in refused)


@needs_sandbox
def test_egress_probes_fail_without_the_egress_section(fx):
    """Not vacuous: with no egress rule the same probes FAIL (each connect goes through)."""
    with running_proxy(fx, "api.anthropic.com") as (_p, port, _log, _r):
        prof = fx / "p.sb"
        prof.write_text(DISPATCH.build_evolver_sandbox_profile(_job(fx, None)))
        rows = _probe(fx, prof, port)
        for k in ("a_other_loopback_port", "e_unix_socket", "d_offhost_ipv4", "d_offhost_ipv6",
                  "c_no_upstream_connection"):
            assert rows[k]["ok"] is False, (k, rows[k])
        assert rows["b_proxy_port"]["ok"] and rows["c_proxy_refuses_host"]["ok"]


def _sb_connect(prof: Path, port: int) -> str:
    code = ("import socket,errno,sys\ns=socket.socket();s.settimeout(3)\n"
            "try:\n s.connect(('127.0.0.1',int(sys.argv[1])));print('connected')\n"
            "except OSError as e:\n print(errno.errorcode.get(e.errno,e))")
    r = subprocess.run(["sandbox-exec", "-f", str(prof), str(Path(sys.executable).resolve()),
                        "-I", "-c", code, str(port)], capture_output=True, text=True, timeout=60,
                       cwd="/")
    return (r.stdout.strip() or r.stderr.strip()[-200:])


@needs_sandbox
def test_rule_order_is_load_bearing_and_the_emitted_order_holds(fx):
    """SBPL takes the LAST matching rule for an operation (measured; `(allow default)`
    is the fallback, not an ordered rule, so it cannot re-open anything wherever it sits). The
    emitted profile puts the egress section last: a throwaway loopback port is EPERM. A broad
    `(allow network-outbound)` AFTER the section re-opens it; the same rule BEFORE the section
    does not; nor does a later allow for every localhost port — which is why the proxy port is
    named, never `localhost:*`."""
    up = Upstream()
    try:
        body = DISPATCH.build_evolver_sandbox_profile(_job(fx, [40001]))
        good = fx / "good.sb"
        good.write_text(body)
        assert _sb_connect(good, up.port) == "EPERM"
        lines = body.splitlines()
        first_eg = min(i for i, x in enumerate(lines) if "network-outbound" in x)
        cases = {
            "after": body + "\n(allow network-outbound)\n",
            "before": "\n".join(lines[:first_eg] + ["(allow network-outbound)"]
                                + lines[first_eg:]),
            "any_localhost_after": body + '\n(allow network-outbound (remote ip "localhost:*"))\n',
        }
        got = {}
        for k, text in cases.items():
            f = fx / f"{k}.sb"
            f.write_text(text)
            got[k] = _sb_connect(f, up.port)
        assert got == {"after": "connected", "before": "EPERM",
                       "any_localhost_after": "connected"}, got
    finally:
        up.close()


# ══ the proxy under load and slow clients ═══════════════════════════════════════════════════════
def test_proxy_drops_a_slow_header_on_an_absolute_deadline(fx, monkeypatch):
    """A client that sends one byte every 0.4 s never trips a per-recv timeout; the head deadline
    must be absolute (slowloris could otherwise hold every tunnel slot)."""
    monkeypatch.setenv("EVOSKILL_EGRESS_HEADER_TIMEOUT_S", "2")
    with running_proxy(fx, "localhost") as (_p, port, log, _r):
        s = socket.create_connection(("127.0.0.1", port), timeout=10)
        t0 = time.time()
        closed_at = None
        for ch in b"CONNECT localhost:443 HTTP/1.1\r\nHost: x\r\n":
            try:
                s.sendall(bytes([ch]))
            except OSError:
                closed_at = time.time()
                break
            time.sleep(0.4)
            if time.time() - t0 > 8:
                break
        if closed_at is None:
            s.settimeout(5)
            try:
                s.recv(1)                     # the proxy has closed its side -> b"" or reset
            except OSError:
                pass
            closed_at = time.time()
        s.close()
        rows = log_lines(log, 1, timeout=8)
        assert any(r.get("reason") == "request head timed out" for r in rows), rows
        assert closed_at - t0 < 7, f"head not dropped on the absolute 2 s deadline ({closed_at - t0:.1f}s)"


def test_proxy_serves_90_tunnels_under_a_256_fd_limit(fx):
    """kqueue selection allows only ~80 concurrent tunnels at macOS's default 256-fd limit (one kqueue fd per
    tunnel). With poll-based selection a tunnel holds 2 fds, so 90 must all work at a hard 256."""
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = 4096 if hard == resource.RLIM_INFINITY else min(4096, hard)
    if soft < want:
        resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))   # the TEST process holds 180+ fds

    def cap256():
        resource.setrlimit(resource.RLIMIT_NOFILE, (256, 256))

    up = Upstream()
    socks = []
    try:
        with running_proxy(fx, "localhost", allowed_port=up.port, preexec_fn=cap256) as (_p, port, log, ready):
            info = json.loads(ready.read_text())
            assert info["fd_limit"] == 256 and info["max_tunnels"] == 96, info
            for _ in range(90):
                s, status = via(port, connect_req(f"localhost:{up.port}"))
                assert status.startswith("HTTP/1.1 200"), status
                socks.append(s)
            for i, s in enumerate(socks):
                s.sendall(b"p%d" % i)
            for i, s in enumerate(socks):
                want_b = b"ECHO:p%d" % i
                assert recv_until(s, want_b).endswith(want_b)
            assert up.accepted == 90
    finally:
        for s in socks:
            try:
                s.close()
            except OSError:
                pass
        up.close()

