#!/usr/bin/env python3
"""Allowlist HTTPS CONNECT proxy for the evolver's `claude` child (egress containment).

    scripts/evolver_egress_proxy.py --allow api.anthropic.com,claude.ai,platform.claude.com \\
        --log RUN/tau3_evolver_e2.egress.jsonl --ready-file RUN/tau3_evolver_e2.egress.ready \\
        --watch-pid <launcher pid> --watch-pid-file RUN/tau3_evolver_e2.egress.watchpid \\
        --max-seconds 43200

WHY IT EXISTS. A `claude` session in an evolver or solver container has no network of its own: the
container's network namespace holds only `lo`. scripts/daytona/evolver_container_ctl.py (`proxy`) runs this
proxy's Proxy class with its listening socket inside that namespace and its upstream side outside it, so this
process is the child's ONLY way off the machine, and it tunnels only to hosts on the allowlist. Run standalone
(main), it serves a child on the same host whose own containment allows only this proxy's loopback port.

WHAT IT DOES
  * binds 127.0.0.1 on an ephemeral port and writes {"port", "pid", ...} to --ready-file with an
    atomic rename, so the launcher can wait for readiness and learn the port;
  * accepts `CONNECT host:443` only (443 = --allowed-port), where host is on the allowlist:
    exact hostname match, case insensitive, one trailing dot ignored. An entry with a leading '.' (".example.com") matches
    SUBDOMAINS of example.com only, never the bare domain; nothing else is a wildcard. An IP
    literal is refused unless that exact literal is listed;
  * answers everything else with 403 (plain HTTP, other methods, other ports, other hosts), and
    for a refused request NEVER resolves or connects anywhere: the allowlist decision is made on
    the request line alone, before any name lookup;
  * logs one JSON line per request (time, method, target, host, port, allowed, reason, bytes
    up/down, duration) to --log, which the launcher places outside the evolver's writable area;
  * tunnels with an idle timeout of --idle-timeout seconds (default 1800: model streams can be
    silent for minutes) measured across BOTH directions, one thread per tunnel;
  * exits when the watched pid dies (checked every --poll seconds). It starts out watching
    --watch-pid (the launcher) and, as soon as --watch-pid-file holds a pid (the launcher writes
    claude's pid there once it has started it), watches that pid instead, for good. It also exits
    after --max-seconds whatever happens, and on SIGTERM/SIGINT.

STDLIB ONLY and Python 3.9 compatible: the launcher runs it with whatever `python3` is on PATH.
"""
from __future__ import annotations

import argparse
import errno
import ipaddress
import json
import os
import re
import selectors
import signal
import socket
import sys
import threading
import time

MAX_HEADER = 16 * 1024          # a CONNECT request line + headers never needs more
HEADER_TIMEOUT_S = float(os.environ.get("EVOSKILL_EGRESS_HEADER_TIMEOUT_S", "30"))  # tests shorten it
CONNECT_TIMEOUT_S = 20.0        # upstream TCP connect, allowed hosts only
MAX_TUNNELS = 512               # concurrent connections; beyond this: 503, never queued


def _selector():
    """A selector that holds NO file descriptor of its own (DefaultSelector is kqueue on
    macOS, one extra fd per tunnel, which capped the proxy at ~80 tunnels under the default 256-fd
    limit). poll() is a plain syscall; fall back to select() where poll is missing."""
    return selectors.PollSelector() if hasattr(selectors, "PollSelector") else selectors.SelectSelector()


def raise_fd_limit(want: int = 4096) -> int:
    """Raise the soft RLIMIT_NOFILE toward `want` (never above the hard limit); return the soft limit."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = want if hard == resource.RLIM_INFINITY else min(want, hard)
        if soft < target:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
        return resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    except Exception:                                           # noqa: BLE001
        return 256
HALF_CLOSE_GRACE_S = 15.0       # after one side closes, the other may drain this long, no longer
_HOST_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)*$")


# ── the allowlist ───────────────────────────────────────────────────────────────────────────

def _norm_host(h: str) -> str:
    """Lower-case, strip ONE trailing dot, strip IPv6 brackets."""
    h = h.strip().lower()
    if h.startswith("[") and h.endswith("]"):
        h = h[1:-1]
    if h.endswith(".") and not h.endswith(".."):
        h = h[:-1]
    return h


def _is_ip(h: str) -> bool:
    try:
        ipaddress.ip_address(h)
        return True
    except ValueError:
        return False


def parse_allowlist(spec: str) -> tuple[frozenset, tuple]:
    """(exact names/IP literals, leading-dot suffixes). Raises ValueError on anything that is not
    a bare hostname, a '.suffix' or an IP literal: a scheme, a port, a path, '*' or an empty
    entry is a configuration error, loud at startup, never a silent widening."""
    exact, suffixes = set(), []
    entries = [e.strip() for e in (spec or "").split(",")]
    if not any(entries):
        raise ValueError("the allowlist is empty")
    for raw in entries:
        if not raw:
            raise ValueError(f"empty entry in allowlist {spec!r}")
        e = raw.lower()
        if e.startswith("."):
            base = _norm_host(e[1:])
            if not base or "." not in base or not _HOST_RE.match(base):
                raise ValueError(f"allowlist entry {raw!r}: a '.suffix' needs a dotted domain "
                                 f"after the dot (e.g. .example.com)")
            suffixes.append("." + base)
            continue
        h = _norm_host(e)
        if _is_ip(h):
            exact.add(str(ipaddress.ip_address(h)))
            continue
        if not _HOST_RE.match(h):
            raise ValueError(f"allowlist entry {raw!r} is not a bare hostname (no scheme, port, "
                             f"path or wildcard; use '.example.com' for subdomains)")
        exact.add(h)
    return frozenset(exact), tuple(suffixes)


def host_allowed(host: str, exact: frozenset, suffixes: tuple) -> bool:
    h = _norm_host(host)
    if not h:
        return False
    if _is_ip(h):
        return str(ipaddress.ip_address(h)) in exact
    if not _HOST_RE.match(h):
        return False
    if h in exact:
        return True
    return any(h.endswith(s) and len(h) > len(s) for s in suffixes)


def parse_connect_target(target: str):
    """'host:port' or '[v6]:port' -> (host, port), or None when malformed."""
    m = re.match(r"^\[([0-9A-Fa-f:.]+)\]:(\d{1,5})$", target) or \
        re.match(r"^([^\s:/\[\]@]+):(\d{1,5})$", target)
    if not m:
        return None
    port = int(m.group(2))
    if not 1 <= port <= 65535:
        return None
    return m.group(1), port


# ── the server ──────────────────────────────────────────────────────────────────────────────

class Proxy:
    def __init__(self, allow: str, log_path: str, idle_timeout: float, port: int = 0,
                 allowed_port: int = 443):
        self.exact, self.suffixes = parse_allowlist(allow)
        self.allowed_port = int(allowed_port)
        self.allow_spec = allow
        self.idle_timeout = float(idle_timeout)
        self.log_path = log_path
        self._log_lock = threading.Lock()
        d = os.path.dirname(os.path.abspath(log_path))
        os.makedirs(d, exist_ok=True)
        self._log = open(log_path, "a", buffering=1)
        # each tunnel holds 2 fds (client + upstream); keep 64 spare for the listener, log, etc.
        self.fd_limit = raise_fd_limit()
        self.max_tunnels = max(8, min(MAX_TUNNELS, (self.fd_limit - 64) // 2))
        self._sem = threading.BoundedSemaphore(self.max_tunnels)
        # tunnels in flight: a tunnel's line is written when it closes, so a proxy that exits
        # with the evolver would otherwise drop the lines of whatever was still open
        self._active: dict = {}
        self._active_lock = threading.Lock()
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", int(port)))
        self.srv.listen(256)
        self.port = self.srv.getsockname()[1]
        self._stopping = False

    # one JSON line per request; never raises
    def log(self, **rec) -> None:
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
               "ts": round(time.time(), 3), "proxy_pid": os.getpid(), **rec}
        try:
            with self._log_lock:
                self._log.write(json.dumps(rec, sort_keys=True) + "\n")
                self._log.flush()
        except Exception:                                       # noqa: BLE001
            pass

    def serve_forever(self) -> None:
        while not self._stopping:
            try:
                c, peer = self.srv.accept()
            except OSError:
                if self._stopping:
                    return
                continue
            if not self._sem.acquire(blocking=False):
                try:
                    c.sendall(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n"
                              b"Connection: close\r\n\r\n")
                finally:
                    c.close()
                self.log(method="", target="", host="", port=None, allowed=False,
                         reason="too many concurrent tunnels", bytes_up=0, bytes_down=0,
                         duration_s=0.0, client_port=peer[1])
                continue
            t = threading.Thread(target=self._handle_guarded, args=(c, peer), daemon=True)
            t.start()

    def stop(self) -> None:
        self._stopping = True
        try:
            self.srv.close()
        except OSError:
            pass

    def _handle_guarded(self, c: socket.socket, peer) -> None:
        try:
            self._handle(c, peer)
        finally:
            self._sem.release()

    def _read_head(self, c: socket.socket):
        """(head bytes, extra bytes after the blank line) or (None, reason)."""
        # ABSOLUTE deadline for the whole head: a per-recv timeout lets a client
        # that sends one byte every 29 s hold a tunnel slot for hours (slowloris).
        deadline = time.monotonic() + HEADER_TIMEOUT_S
        buf = b""
        while b"\r\n\r\n" not in buf:
            if len(buf) > MAX_HEADER:
                return None, "request head too large"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None, "request head timed out"
            c.settimeout(remaining)
            try:
                chunk = c.recv(4096)
            except socket.timeout:
                return None, "request head timed out"
            except OSError as e:
                return None, f"recv failed: {e}"
            if not chunk:
                return None, "client closed before a complete request"
            buf += chunk
        head, _, extra = buf.partition(b"\r\n\r\n")
        return head, extra

    def _refuse(self, c, code: int, why: str) -> None:
        msg = {400: b"Bad Request", 403: b"Forbidden", 502: b"Bad Gateway"}.get(code, b"Error")
        body = (why + "\n").encode("utf-8", "replace")
        try:
            c.sendall(b"HTTP/1.1 %d %s\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n"
                      b"Connection: close\r\n\r\n" % (code, msg, len(body)) + body)
        except OSError:
            pass

    def _handle(self, c: socket.socket, peer) -> None:
        t0 = time.time()
        method = target = host = ""
        port = None
        up = down = 0
        try:
            head, extra = self._read_head(c)
            if head is None:
                self.log(method="", target="", host="", port=None, allowed=False, reason=extra,
                         bytes_up=0, bytes_down=0, duration_s=round(time.time() - t0, 3),
                         client_port=peer[1])
                return
            line = head.split(b"\r\n", 1)[0].decode("latin-1", "replace")
            parts = line.split(" ")
            method = parts[0] if parts else ""
            target = parts[1] if len(parts) > 1 else ""
            if len(parts) != 3 or not parts[2].startswith("HTTP/1."):
                self._refuse(c, 400, "malformed request line")
                self.log(method=method[:32], target=target[:256], host="", port=None,
                         allowed=False, reason="malformed request line", bytes_up=0,
                         bytes_down=0, duration_s=round(time.time() - t0, 3), client_port=peer[1])
                return
            if method != "CONNECT":
                self._refuse(c, 403, "only CONNECT to an allowlisted host:443 is permitted")
                self.log(method=method[:32], target=target[:256], host="", port=None,
                         allowed=False, reason="not CONNECT", bytes_up=0, bytes_down=0,
                         duration_s=round(time.time() - t0, 3), client_port=peer[1])
                return
            hp = parse_connect_target(target)
            if hp is None:
                self._refuse(c, 400, "malformed CONNECT target")
                self.log(method=method, target=target[:256], host="", port=None, allowed=False,
                         reason="malformed CONNECT target", bytes_up=0, bytes_down=0,
                         duration_s=round(time.time() - t0, 3), client_port=peer[1])
                return
            host, port = hp
            # THE DECISION, made on the request line alone: no lookup, no connect before it.
            if not host_allowed(host, self.exact, self.suffixes):
                why = "host not on the allowlist"
            elif port != self.allowed_port:
                why = f"port is not {self.allowed_port}"
            else:
                why = ""
            if why:
                self._refuse(c, 403, why)
                self.log(method=method, target=target[:256], host=_norm_host(host)[:256],
                         port=port, allowed=False, reason=why, bytes_up=0, bytes_down=0,
                         duration_s=round(time.time() - t0, 3), client_port=peer[1])
                return
            try:
                u = socket.create_connection((_norm_host(host), port), timeout=CONNECT_TIMEOUT_S)
            except OSError as e:
                self._refuse(c, 502, f"upstream connect failed: {e}")
                self.log(method=method, target=target, host=_norm_host(host), port=port,
                         allowed=True, reason=f"upstream connect failed: {e}", bytes_up=0,
                         bytes_down=0, duration_s=round(time.time() - t0, 3), client_port=peer[1])
                return
            rec = {"method": method, "target": target, "host": _norm_host(host), "port": port,
                   "upstream_ip": "", "t0": t0, "client_port": peer[1], "up": 0, "down": 0,
                   "logged": False}
            with self._active_lock:
                self._active[id(rec)] = rec
            try:
                rec["upstream_ip"] = u.getpeername()[0]
                c.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                if extra:
                    u.sendall(extra)
                    rec["up"] += len(extra)
                self._tunnel(c, u, rec)
            except Exception as e:                              # noqa: BLE001
                self._finish(rec, f"proxy error in tunnel: {type(e).__name__}: {e}")
            finally:
                try:
                    u.close()
                except OSError:
                    pass
                self._finish(rec, "")                   # no-op when already logged
        except Exception as e:                                  # noqa: BLE001
            self.log(method=method[:32], target=target[:256], host=host[:256], port=port,
                     allowed=False, reason=f"proxy error: {type(e).__name__}: {e}",
                     bytes_up=up, bytes_down=down, duration_s=round(time.time() - t0, 3),
                     client_port=peer[1] if peer else None)
        finally:
            try:
                c.close()
            except OSError:
                pass

    def _finish(self, rec: dict, reason: str) -> None:
        """Write a tunnel's ONE line, exactly once (the tunnel thread and the exit path race)."""
        with self._active_lock:
            if rec["logged"]:
                return
            rec["logged"] = True
            self._active.pop(id(rec), None)
        self.log(method=rec["method"], target=rec["target"], host=rec["host"], port=rec["port"],
                 allowed=True, reason=reason, upstream_ip=rec["upstream_ip"],
                 bytes_up=rec["up"], bytes_down=rec["down"],
                 duration_s=round(time.time() - rec["t0"], 3), client_port=rec["client_port"])

    def log_open_tunnels(self, why: str) -> None:
        with self._active_lock:
            recs = list(self._active.values())
        for rec in recs:
            self._finish(rec, f"still open at proxy exit ({why})")

    def _tunnel(self, c: socket.socket, u: socket.socket, rec: dict) -> None:
        """Copy bytes both ways until both sides are done, or nothing moves for idle_timeout, or
        one side has been closed for HALF_CLOSE_GRACE_S. Byte counts go into rec as they move."""
        c.setblocking(False)
        u.setblocking(False)
        sel = _selector()
        sel.register(c, selectors.EVENT_READ, u)
        sel.register(u, selectors.EVENT_READ, c)
        key_of = {c: "up", u: "down"}
        open_readers = 2
        last = time.time()
        half_deadline = None
        try:
            while open_readers:
                wait = min(30.0, self.idle_timeout)
                if half_deadline is not None:
                    wait = max(0.0, min(wait, half_deadline - time.time()))
                events = sel.select(timeout=wait)
                if half_deadline is not None and time.time() >= half_deadline:
                    break
                if not events:
                    if time.time() - last >= self.idle_timeout:
                        break
                    continue
                for key, _ in events:
                    src, dst = key.fileobj, key.data
                    try:
                        data = src.recv(65536)
                    except (BlockingIOError, InterruptedError):
                        continue
                    except OSError:
                        data = b""
                    if not data:
                        sel.unregister(src)
                        open_readers -= 1
                        if half_deadline is None:
                            half_deadline = time.time() + HALF_CLOSE_GRACE_S
                        try:
                            dst.shutdown(socket.SHUT_WR)
                        except OSError:
                            pass
                        continue
                    last = time.time()
                    rec[key_of[src]] += len(data)
                    if not _send_all(dst, data, self.idle_timeout):
                        return
        finally:
            sel.close()


def _send_all(s: socket.socket, data: bytes, timeout: float) -> bool:
    view = memoryview(data)
    deadline = time.time() + timeout
    while view:
        try:
            n = s.send(view)
            view = view[n:]
        except (BlockingIOError, InterruptedError):
            if time.time() > deadline:
                return False
            _wait_writable(s, 1.0)
        except OSError:
            return False
    return True


def _wait_writable(s: socket.socket, t: float) -> None:
    sel = _selector()
    try:
        sel.register(s, selectors.EVENT_WRITE)
        sel.select(timeout=t)
    finally:
        sel.close()


# ── lifetime ────────────────────────────────────────────────────────────────────────────────

def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError as e:
        return e.errno == errno.EPERM                # exists, owned by someone else
    return True


def read_pid_file(path: str):
    try:
        with open(path) as fh:
            s = fh.read().strip()
    except OSError:
        return None
    return int(s) if s.isdigit() and int(s) > 0 else None


def _write_ready(path: str, payload: dict) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".{os.path.basename(path)}.{os.getpid()}.tmp")
    with open(tmp, "w") as fh:
        fh.write(json.dumps(payload) + "\n")
    os.replace(tmp, path)


def classify_base_url(url: str, allow: str) -> dict:
    """How the evolver's `claude` would reach ANTHROPIC_BASE_URL under this allowlist.
      {"kind": "local", "port": P}   a loopback URL: the profile must allow that port directly
                                     (NO_PROXY keeps it off the proxy) — e.g. a local model proxy;
      {"kind": "proxied", "host": H} an https URL whose host the proxy will tunnel;
      {"kind": "refused", "why": W}  anything else: the model endpoint would be unreachable."""
    from urllib.parse import urlsplit
    exact, suffixes = parse_allowlist(allow)
    try:
        u = urlsplit(url.strip())
        host, port = (u.hostname or ""), u.port
    except ValueError as e:
        return {"kind": "refused", "why": f"unparseable URL: {e}"}
    if u.scheme not in ("http", "https") or not host:
        return {"kind": "refused", "why": f"not an http(s) URL: {url!r}"}
    if _norm_host(host) in ("127.0.0.1", "localhost", "::1"):
        return {"kind": "local", "port": port or (443 if u.scheme == "https" else 80)}
    if u.scheme != "https":
        return {"kind": "refused", "why": f"plain-http model endpoint {host!r}: the proxy "
                                          f"tunnels TLS (CONNECT) only"}
    if (port or 443) != 443:
        return {"kind": "refused", "why": f"model endpoint port {port} is not 443"}
    if not host_allowed(host, exact, suffixes):
        return {"kind": "refused", "why": f"model endpoint host {host!r} is not on the egress "
                                          f"allowlist"}
    return {"kind": "proxied", "host": _norm_host(host)}


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] == ["--classify-url"]:
        # `--classify-url URL --allow SPEC`: for the launcher, which must not carry a second
        # copy of the host-matching rule. Prints JSON; exit 0 local/proxied, 3 refused, 2 usage.
        cp = argparse.ArgumentParser(prog="evolver_egress_proxy --classify-url")
        cp.add_argument("--classify-url", required=True)
        cp.add_argument("--allow", required=True)
        c = cp.parse_args(args)
        try:
            r = classify_base_url(c.classify_url, c.allow)
        except ValueError as e:
            print(f"REFUSING: {e}", file=sys.stderr)
            return 2
        print(json.dumps(r))
        return 3 if r["kind"] == "refused" else 0
    ap = argparse.ArgumentParser(prog="evolver_egress_proxy",
                                 description="allowlist HTTPS CONNECT proxy on 127.0.0.1")
    ap.add_argument("--allow", required=True,
                    help="comma-separated hostnames (exact), '.suffix' for subdomains, or IP literals")
    ap.add_argument("--log", required=True, help="JSONL request log (append)")
    ap.add_argument("--ready-file", required=True,
                    help="written atomically with {port, pid} once listening")
    ap.add_argument("--port", type=int, default=0, help="default 0 = ephemeral")
    ap.add_argument("--allowed-port", type=int, default=443,
                    help="the ONE upstream port a CONNECT may name (default 443; the tests use a "
                         "local upstream on an unprivileged port)")
    ap.add_argument("--watch-pid", type=int, default=0,
                    help="exit when this pid dies (until --watch-pid-file names another)")
    ap.add_argument("--watch-pid-file", default="",
                    help="once this file holds a pid, watch THAT pid instead, for good")
    ap.add_argument("--max-seconds", type=float, default=43200.0, help="hard lifetime cap")
    ap.add_argument("--idle-timeout", type=float, default=1800.0,
                    help="tunnel idle timeout across both directions (>= 900 s recommended)")
    ap.add_argument("--poll", type=float, default=2.0, help="watch interval, seconds")
    a = ap.parse_args(args)
    if a.max_seconds <= 0:
        print("REFUSING: --max-seconds must be > 0", file=sys.stderr)
        return 2
    if not (a.watch_pid or a.watch_pid_file):
        print("REFUSING: give --watch-pid and/or --watch-pid-file; a proxy that outlives its "
              "evolver keeps a hole open", file=sys.stderr)
        return 2
    try:
        px = Proxy(a.allow, a.log, a.idle_timeout, a.port, a.allowed_port)
    except ValueError as e:
        print(f"REFUSING: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"REFUSING: cannot listen/log: {e}", file=sys.stderr)
        return 3

    done = threading.Event()
    reason = {"why": ""}

    def finish(why: str) -> None:
        if not reason["why"]:
            reason["why"] = why
        done.set()

    signal.signal(signal.SIGTERM, lambda *_: finish("SIGTERM"))
    signal.signal(signal.SIGINT, lambda *_: finish("SIGINT"))

    def watcher() -> None:
        t_end = time.time() + a.max_seconds
        watched = a.watch_pid or 0
        switched = False
        while not done.is_set():
            if a.watch_pid_file and not switched:
                p = read_pid_file(a.watch_pid_file)       # the file first, then the pid
                if p:
                    watched, switched = p, True
                    print(f"[egress-proxy] now watching pid {p} (from {a.watch_pid_file})",
                          file=sys.stderr, flush=True)
            if watched and not pid_alive(watched):
                finish(f"watched pid {watched} exited")
                return
            if time.time() >= t_end:
                finish("max-seconds")
                return
            done.wait(a.poll)

    threading.Thread(target=watcher, daemon=True).start()
    threading.Thread(target=px.serve_forever, daemon=True).start()
    exact = sorted(px.exact) + list(px.suffixes)
    _write_ready(a.ready_file, {"port": px.port, "pid": os.getpid(), "host": "127.0.0.1",
                                "allow": exact, "allowed_port": px.allowed_port,
                                "log": os.path.abspath(a.log),
                                "fd_limit": px.fd_limit, "max_tunnels": px.max_tunnels})
    print(f"[egress-proxy] fd limit {px.fd_limit}, max concurrent tunnels {px.max_tunnels}",
          file=sys.stderr, flush=True)
    print(f"[egress-proxy] listening 127.0.0.1:{px.port} pid {os.getpid()} allow {exact}",
          file=sys.stderr, flush=True)
    while not done.wait(1.0):
        pass
    px.stop()
    px.log_open_tunnels(reason["why"])
    print(f"[egress-proxy] exiting: {reason['why']}", file=sys.stderr, flush=True)
    for f in (a.ready_file, a.watch_pid_file):
        if f:
            try:
                os.unlink(f)
            except OSError:
                pass
    sys.stderr.flush()
    os._exit(0)                                  # daemon tunnel threads must not hold us up


if __name__ == "__main__":
    raise SystemExit(main())
