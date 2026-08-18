#!/usr/bin/env python3
"""Transparent proxy that intercepts CC SDK → OpenRouter traffic.

Logs every request/response to Phoenix (separate project) AND stdout.
Set ANTHROPIC_BASE_URL=http://localhost:8100 to route CC through this proxy.
"""
import json
import time
import sys
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request
import urllib.error
import ssl

from opentelemetry import trace
# NOTE 2026-08-08: `SimpleSpanExporter` does not exist in the installed
# opentelemetry-sdk (the real names are SimpleSpanProcessor / ConsoleSpanExporter),
# so this module CRASHED ON IMPORT and the proxy could not start at all. Both this
# and the TracerProvider import were DEAD — pyflakes flags them as unused, and the
# tracer actually comes from `phoenix.otel.register` below. Removed rather than
# corrected, because importing a name nothing uses is how the breakage hid.
# TRACING IS OPTIONAL; THE PROXY IS NOT.
# These two packages live in the OPTIONAL `[phoenix]` extra, but this proxy is a
# MANDATORY step for DeepSeek-family models through the `claude` CLI -- README's
# "Model routing notes" says so and scripts/lib/deepseek_pin_guard.sh refuses to
# launch without it. Importing them at module scope meant that after the documented
# `uv sync` (no extras) the mandatory component could not start, so the pin guard
# demanded a proxy the environment could not run. Observability must not be able to
# take down routing. Absent, the proxy runs and pins exactly as before and only the
# Phoenix spans are missing -- which it says once, loudly, rather than degrading
# silently.
try:
    from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues
    from opentelemetry.trace import StatusCode
    _TRACING = True
except ImportError as _e:
    _TRACING = False
    _TRACING_ERR = _e

PHOENIX_PROJECT = os.environ.get("CC_PROXY_PHOENIX_PROJECT", "cc-proxy-raw")
if PHOENIX_PROJECT == "cc-proxy-raw":
    print("WARNING: Using default project name 'cc-proxy-raw'. Set CC_PROXY_PHOENIX_PROJECT for experiment-specific naming.", flush=True)
PHOENIX_ENDPOINT = os.environ.get("CC_PROXY_PHOENIX_ENDPOINT", "http://localhost:6006/v1/traces")
TARGET = "https://openrouter.ai/api"
PORT = int(os.environ.get("CC_PROXY_PORT", "8100"))

# Init Phoenix tracing, if the optional extra is installed.
if _TRACING:
    try:
        from phoenix.otel import register
        tracer_provider = register(
            project_name=PHOENIX_PROJECT,
            endpoint=PHOENIX_ENDPOINT,
        )
        _tracer = trace.get_tracer("cc-proxy")
    except ImportError as _e:
        _TRACING, _TRACING_ERR = False, _e

if not _TRACING:
    print(f"cc_proxy: Phoenix tracing DISABLED ({_TRACING_ERR}). The proxy still runs "
          f"and still pins the provider; only spans are missing. Install the extra with "
          f"`uv sync --extra phoenix` if you want traces.", flush=True)

    class _NoSpan:
        def set_attribute(self, *a, **k): pass
        def set_status(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False

    class _NoTracer:
        def start_as_current_span(self, *a, **k): return _NoSpan()

    class _NoAttrs:
        """Any SpanAttributes.X resolves to a harmless key; nothing consumes it."""
        def __getattr__(self, name): return f"noop.{name}"

    _tracer = _NoTracer()
    SpanAttributes = _NoAttrs()

    class _NoKind:
        LLM = type("V", (), {"value": "LLM"})()
    OpenInferenceSpanKindValues = _NoKind()

    class StatusCode:  # noqa: F811 - deliberate stand-in for the real enum
        ERROR = "ERROR"

_ctx = ssl.create_default_context()
_call_num = 0
_total_prompt = 0
_total_completion = 0
_total_cost = 0.0


def _messages_to_str(messages):
    """Render all messages as a readable string for Phoenix."""
    parts = []
    for i, m in enumerate(messages):
        role = m.get("role", "?")
        content = m.get("content", "")
        if isinstance(content, list):
            content = json.dumps(content)
        else:
            content = str(content)
        parts.append(f"[{i}] {role} ({len(content):,} chars):\n{content}")
    return "\n\n".join(parts)


class ProxyHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        global _call_num, _total_prompt, _total_completion, _total_cost
        _call_num += 1
        call_id = _call_num

        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len)

        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            req = {}

        model = req.get("model", "?")
        messages = req.get("messages", [])
        n_messages = len(messages)
        thinking = req.get("thinking", {})
        reasoning_cfg = req.get("reasoning", {})
        max_tokens = req.get("max_tokens", "?")
        stream = req.get("stream", False)

        # ── #314. PROVIDER PIN, INJECTED ON THE WIRE ────────────────────────────────────────
        #
        # OpenRouter routes a model to whichever upstream it likes unless the REQUEST BODY says
        # otherwise; `provider` is a body field, not a header or a URL parameter, so the only
        # place to set it for a CC-driven call is here. Without it the DSV4F evolver was served
        # by DigitalOcean (observed in the assistant records of the 2026-08-13 epoch-1 run),
        # which is a different deployment of the same weights, with its own quantisation,
        # context handling and failure modes.
        #
        # WHY IT MATTERS EVEN THOUGH evolver_model IS OUTSIDE _INSTRUMENT. The cache identity is
        # unaffected, so nothing silently replays. What is affected is what "the DSV4F evolver
        # arm" NAMES: across three epochs an unpinned model can be served by three different
        # providers, and the run would then describe an average over an uncontrolled variable.
        # the maintainer's standing rule is that every DeepSeek model carries BOTH order=["deepseek"]
        # AND allow_fallbacks=false — a preference is not a pin, and allow_fallbacks left true
        # silently reintroduces exactly what the order was meant to exclude.
        #
        # SCOPED BY MODEL FAMILY. Only vendor-prefixed DeepSeek ids are pinned. Pinning
        # unconditionally would 404 any other model routed through this proxy, which is the
        # failure the codex pin_proxy hit and had to learn to gate on.
        _pin_applied = None
        if isinstance(req, dict) and str(model).lower().startswith("deepseek/"):
            want = {"order": ["deepseek"], "allow_fallbacks": False}
            if req.get("provider") != want:
                req["provider"] = want
                body = json.dumps(req).encode()
                # Content-Length is recomputed below from this new body; the header dict is
                # rebuilt there, so no stale length can escape.
                _pin_applied = want
        if _pin_applied:
            print(f"  PIN: provider={_pin_applied} (#314 — native DeepSeek, no fallbacks)",
                  flush=True)

        # Stdout log
        print(f"\n{'='*70}", flush=True)
        print(f"[CALL {call_id}] POST {self.path}", flush=True)
        print(f"  Model: {model} | Messages: {n_messages} | max_tokens: {max_tokens} | stream: {stream}", flush=True)
        if thinking:
            print(f"  Thinking: {thinking}", flush=True)
        if reasoning_cfg:
            print(f"  Reasoning config: {reasoning_cfg}", flush=True)
        print(f"--- REQUEST MESSAGES ---", flush=True)
        for i, m in enumerate(messages):
            role = m.get("role", "?")
            content = m.get("content", "")
            if isinstance(content, list):
                content_str = json.dumps(content)
            else:
                content_str = str(content)
            print(f"  [{i}] {role} ({len(content_str):,} chars):", flush=True)
            print(content_str, flush=True)
        print(f"--- END REQUEST ---", flush=True)

        # Forward to OpenRouter
        target_url = TARGET + self.path
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in ("host", "content-length")}
        headers["Content-Length"] = str(len(body))

        t0 = time.monotonic()
        status = 0
        resp_body = b""
        resp_headers = {}
        error_msg = None
        try:
            fwd_req = urllib.request.Request(
                target_url, data=body, headers=headers, method="POST"
            )
            with urllib.request.urlopen(fwd_req, context=_ctx, timeout=120) as resp:
                resp_body = resp.read()
                status = resp.status
                resp_headers = dict(resp.headers)
        except urllib.error.HTTPError as e:
            resp_body = e.read()
            status = e.code
            resp_headers = dict(e.headers)
        except Exception as e:
            error_msg = str(e)
            print(f"  [ERROR] {e}", flush=True)
            self.send_error(502, str(e))

        elapsed = time.monotonic() - t0

        # ── #310. DECOMPRESS BEFORE PARSING, AND NEVER LET PARSING KILL THE PROXY ──────────
        #
        # This handler copies every client header through except host/content-length, so it
        # forwards the client's `Accept-Encoding: gzip` — and then could not read what it had
        # asked for. OpenRouter returned a gzip body, `json.loads` called `.decode()` on the
        # raw bytes, and that raises UnicodeDecodeError ("can't decode byte 0x8b in position 1"
        # — 1f 8b is the gzip magic). UnicodeDecodeError is NOT a subclass of JSONDecodeError,
        # so the narrow except below did not catch it, the exception escaped do_POST, the
        # handler thread died mid-request, and the CLIENT saw a broken connection.
        #
        # WHAT THAT COST, and why the failure mode matters more than the bug: the AO authoring
        # arm reported `is_error: true, num_turns: 1, input_tokens: 0, output_tokens: 0,
        # duration_api_ms: 0` after 187 seconds of wall time. Read quickly, that is a model
        # that produced nothing. It is in fact a proxy that never delivered the response. This
        # is the THIRD time on this project that a transport fault has been legible as a
        # capability ceiling (#210, #212 — both DSV4F), so the parse is now defensive and the
        # decode is explicit.
        #
        # PARSING IS FOR LOGGING AND TRACING ONLY. The bytes returned to the client further
        # down are the ORIGINAL `resp_body` with the ORIGINAL headers, so the client still gets
        # exactly what OpenRouter sent, Content-Encoding included. Decompressing here changes
        # what the proxy can SEE, never what it FORWARDS.
        _enc = str(resp_headers.get("Content-Encoding") or
                   resp_headers.get("content-encoding") or "").lower()
        _readable = resp_body
        if _enc == "gzip" and resp_body[:2] == b"\x1f\x8b":
            import gzip as _gzip
            try:
                _readable = _gzip.decompress(resp_body)
            except Exception as e:                                    # noqa: BLE001
                print(f"  [WARN] gzip body would not decompress: {e!r}", flush=True)
        elif _enc == "deflate":
            import zlib as _zlib
            try:
                _readable = _zlib.decompress(resp_body)
            except Exception:                                          # noqa: BLE001
                try:
                    _readable = _zlib.decompress(resp_body, -_zlib.MAX_WBITS)
                except Exception as e:                                 # noqa: BLE001
                    print(f"  [WARN] deflate body would not decompress: {e!r}", flush=True)

        # BROAD ON PURPOSE. A proxy that dies while parsing is strictly worse than a proxy that
        # logs nothing: the first loses the caller's request, the second loses only a span. The
        # narrow `except json.JSONDecodeError` here is exactly what let #310 escape.
        try:
            resp_json = json.loads(_readable)
        except Exception as e:                                         # noqa: BLE001
            print(f"  [WARN] response body not parseable as JSON ({e!r}); "
                  f"{len(resp_body)} B, encoding={_enc or 'none'}. Forwarding it untouched; "
                  f"this span will be empty.", flush=True)
            resp_json = {}
        if not isinstance(resp_json, dict):
            resp_json = {}

        usage = resp_json.get("usage", {})
        prompt_tok = usage.get("prompt_tokens", 0)
        completion_tok = usage.get("completion_tokens", 0)
        cache_read = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        cost = float(usage.get("cost", 0) or 0)

        _total_prompt += prompt_tok
        _total_completion += completion_tok
        _total_cost += cost

        choices = resp_json.get("choices", [{}])
        finish = choices[0].get("finish_reason", "?") if choices else "?"
        msg = choices[0].get("message", {}) if choices else {}
        resp_content = msg.get("content", "") or ""
        resp_reasoning = msg.get("reasoning_content", "") or msg.get("reasoning", "") or ""
        tool_calls = msg.get("tool_calls", [])

        # Stdout log response
        print(f"  [{elapsed:.1f}s] {status} | {prompt_tok:,} prompt ({cache_read:,} cached) + {completion_tok:,} completion", flush=True)
        print(f"  Cost: ${cost:.4f} | Finish: {finish}", flush=True)
        if resp_reasoning:
            print(f"  Reasoning ({len(resp_reasoning):,} chars):", flush=True)
            print(resp_reasoning, flush=True)
        if resp_content:
            print(f"  Content ({len(resp_content):,} chars):", flush=True)
            print(resp_content, flush=True)
        if tool_calls:
            for tc in tool_calls:
                fn = tc.get("function", {})
                print(f"  Tool: {fn.get('name', '?')} | Args:", flush=True)
                print(fn.get("arguments", ""), flush=True)
        print(f"  CUMULATIVE: {_total_prompt:,} prompt + {_total_completion:,} completion = ${_total_cost:.4f} ({call_id} calls)", flush=True)

        # Emit Phoenix span
        with _tracer.start_as_current_span(
            f"call-{call_id} | {model}",
            attributes={
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
                SpanAttributes.LLM_MODEL_NAME: model,
                SpanAttributes.LLM_TOKEN_COUNT_PROMPT: prompt_tok,
                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION: completion_tok,
            },
        ) as span:
            # Full request
            span.set_attribute(SpanAttributes.INPUT_VALUE, _messages_to_str(messages))
            span.set_attribute("request.n_messages", n_messages)
            span.set_attribute("request.max_tokens", str(max_tokens))
            span.set_attribute("request.stream", stream)
            if thinking:
                span.set_attribute("request.thinking", json.dumps(thinking))
            if reasoning_cfg:
                span.set_attribute("request.reasoning", json.dumps(reasoning_cfg))

            # Full response
            output_parts = []
            if resp_reasoning:
                output_parts.append(f"=== REASONING ({len(resp_reasoning):,} chars) ===\n{resp_reasoning}")
            if resp_content:
                output_parts.append(f"=== CONTENT ({len(resp_content):,} chars) ===\n{resp_content}")
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    output_parts.append(f"=== TOOL: {fn.get('name', '?')} ===\n{fn.get('arguments', '')}")
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, "\n\n".join(output_parts) or "(empty)")

            # Cost + tokens
            span.set_attribute("llm.cost.total", cost)
            span.set_attribute("llm.cost.cumulative", _total_cost)
            span.set_attribute("llm.tokens.cache_read", cache_read)
            span.set_attribute("response.finish_reason", finish)
            span.set_attribute("response.status", status)
            span.set_attribute("response.latency_s", round(elapsed, 2))
            span.set_attribute("call.id", call_id)

            if error_msg:
                span.set_status(StatusCode.ERROR, error_msg)
            elif status >= 400:
                span.set_status(StatusCode.ERROR, f"HTTP {status}")

        # Send response back to CC
        if not error_msg:
            self.send_response(status)
            for k, v in resp_headers.items():
                if k.lower() not in ("transfer-encoding", "content-encoding", "content-length"):
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(resp_body)))
            self.end_headers()
            self.wfile.write(resp_body)

    def do_GET(self):
        # #319: the pin CONTRACT, served by the proxy itself. The dispatch-side guard
        # (scripts/lib/deepseek_pin_guard.sh) refuses any DeepSeek-family CC dispatch whose
        # base URL does not answer this with deepseek_pin:true — so a live port that is not
        # a pinning proxy cannot satisfy it.
        if self.path == "/pin":
            body = json.dumps({
                "deepseek_pin": True,
                "rule": {"order": ["deepseek"], "allow_fallbacks": False},
                "scope": "models with prefix deepseek/ (#314)",
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        target_url = TARGET + self.path
        headers = {k: v for k, v in self.headers.items() if k.lower() != "host"}
        try:
            fwd_req = urllib.request.Request(target_url, headers=headers)
            with urllib.request.urlopen(fwd_req, context=_ctx, timeout=30) as resp:
                resp_body = resp.read()
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() not in ("transfer-encoding", "content-encoding", "content-length"):
                        self.send_header(k, v)
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
        except Exception as e:
            self.send_error(502, str(e))

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    print(f"CC Proxy listening on http://localhost:{PORT}", flush=True)
    print(f"Forwarding to {TARGET}", flush=True)
    print(f"Phoenix project: {PHOENIX_PROJECT} → {PHOENIX_ENDPOINT}", flush=True)
    print(f"Set ANTHROPIC_BASE_URL=http://localhost:{PORT}", flush=True)
    server = HTTPServer(("localhost", PORT), ProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nTotal: {_call_num} calls, {_total_prompt:,} prompt + {_total_completion:,} completion, ${_total_cost:.4f}")
