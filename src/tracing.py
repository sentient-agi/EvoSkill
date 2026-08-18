"""Phoenix observability for EvoSkill.

Call `init_tracing()` early in your entry point — before any LLM SDK imports —
so that Phoenix auto-instrumentation can patch the Anthropic client.
Failures are logged but never raised; the loop must run even if tracing is broken.
"""

import os
import sys

_initialized = False

# Daytona SDK @with_instrumentation spans (Process.exec, Sandbox.refresh_data, …)
# only carry {component, method} — no command/stdout. Our tool.bash spans already
# record input/output; drop the SDK noise so Phoenix stays readable.
_DROP_SPAN_NAMES = frozenset({
    "Process.exec",
    "Sandbox.refresh_data",
    "Sandbox.wait_for_sandbox_start",
})
_DROP_TRACER_PREFIXES = ("daytona-sdk",)


def _install_daytona_span_filter() -> None:
    """Drop low-info Daytona SDK spans before they reach Phoenix."""
    from opentelemetry import trace
    from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor

    provider = trace.get_tracer_provider()
    active = getattr(provider, "_active_span_processor", None)
    if active is None:
        return

    def _should_drop(span: ReadableSpan) -> bool:
        name = getattr(span, "name", "") or ""
        if name in _DROP_SPAN_NAMES:
            return True
        # Daytona auto-spans only set component/method — drop those leftovers
        instr = getattr(span, "instrumentation_scope", None) or getattr(
            span, "instrumentation_info", None
        )
        instr_name = (getattr(instr, "name", None) or "") if instr else ""
        if not any(instr_name.startswith(p) for p in _DROP_TRACER_PREFIXES):
            return False
        attrs = dict(getattr(span, "attributes", None) or {})
        keys = set(attrs) - {"component", "method"}
        return not keys  # only component/method → noise

    class _FilteringProcessor(SpanProcessor):
        def __init__(self, inner: SpanProcessor):
            self._inner = inner

        def on_start(self, span, parent_context=None):  # noqa: ANN001
            return self._inner.on_start(span, parent_context)

        def on_end(self, span: ReadableSpan) -> None:
            if _should_drop(span):
                return
            self._inner.on_end(span)

        def shutdown(self) -> None:
            return self._inner.shutdown()

        def force_flush(self, timeout_millis: int = 30000) -> bool:
            return self._inner.force_flush(timeout_millis)

    procs = getattr(active, "_span_processors", None)
    if procs is not None:
        provider._active_span_processor._span_processors = tuple(  # type: ignore[attr-defined]
            _FilteringProcessor(p) for p in procs
        )
        return
    if hasattr(provider, "_active_span_processor"):
        provider._active_span_processor = _FilteringProcessor(active)  # type: ignore[attr-defined]


def init_tracing(project_name: str = "evoskill") -> None:
    """Register Phoenix OTEL tracing (idempotent, non-fatal on failure)."""
    global _initialized
    if _initialized:
        return

    # Allow Phoenix to capture full base64 images (default 32KB is too small)
    os.environ.setdefault("OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH", "2000000")
    os.environ.setdefault("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "http/protobuf")
    # Raise the per-attribute size limit so full tool inputs / assistant text
    # are preserved (default ~12KB truncates long Bash scripts and Read results).
    # Use direct assignment rather than setdefault so we override any pre-existing
    # smaller limit from the environment.
    os.environ["OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT"] = "16777216"  # 16MB
    os.environ["OTEL_SPAN_ATTRIBUTE_VALUE_LENGTH_LIMIT"] = "16777216"

    # auto_instrument=False on purpose: openinference-instrumentation-anthropic
    # auto-creates one `messages.create` span per LLM round-trip, which
    # clutters the Phoenix trace list with hundreds of low-information rows
    # per agent run. The same per-turn information is preserved on our
    # manual `{agent_name}/turn.N` spans (input.value = query, output.value
    # = parsed response, model, cost, token counts), so the auto spans are
    # redundant noise. To re-enable for low-level Anthropic SDK debugging,
    # flip the flag.
    try:
        from phoenix.otel import register as phoenix_register
        phoenix_register(
            project_name=project_name,
            auto_instrument=False,
            protocol="http/protobuf",
        )
        try:
            _install_daytona_span_filter()
        except Exception as fe:
            print(
                f"[tracing] Daytona span filter not installed ({type(fe).__name__}: {fe})",
                file=sys.stderr,
            )
        _initialized = True
    except Exception as e:
        print(
            f"[tracing] Phoenix tracing fully disabled ({type(e).__name__}: {e}). "
            f"The loop will run without observability.",
            file=sys.stderr,
        )
        _initialized = True  # mark done so we don't retry


def get_tracer(name: str = "evoskill"):
    """Return an OpenTelemetry tracer (safe to call before or after init)."""
    from opentelemetry import trace

    return trace.get_tracer(name)
