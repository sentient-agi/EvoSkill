"""Phoenix observability for EvoSkill.

Call `init_tracing()` early in your entry point — before any LLM SDK imports —
so that Phoenix auto-instrumentation can patch the Anthropic client.
Failures are logged but never raised; the loop must run even if tracing is broken.
"""

import os
import sys

_initialized = False


def init_tracing(project_name: str = "evoskill") -> None:
    """Register Phoenix OTEL tracing (idempotent, non-fatal on failure)."""
    global _initialized
    if _initialized:
        return

    # Allow Phoenix to capture full base64 images (default 32KB is too small)
    os.environ.setdefault("OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH", "2000000")
    os.environ.setdefault("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "http/protobuf")

    try:
        from phoenix.otel import register as phoenix_register
        phoenix_register(
            project_name=project_name,
            auto_instrument=True,
            protocol="http/protobuf",
        )
        _initialized = True
    except Exception as e:
        # Try again with auto_instrument disabled — still gives us manual spans
        print(
            f"[tracing] Phoenix auto-instrumentation failed ({type(e).__name__}: {e}). "
            f"Falling back to manual-span-only mode.",
            file=sys.stderr,
        )
        try:
            from phoenix.otel import register as phoenix_register
            phoenix_register(
                project_name=project_name,
                auto_instrument=False,
                protocol="http/protobuf",
            )
            _initialized = True
        except Exception as e2:
            print(
                f"[tracing] Phoenix tracing fully disabled ({type(e2).__name__}: {e2}). "
                f"The loop will run without observability.",
                file=sys.stderr,
            )
            _initialized = True  # mark done so we don't retry


def get_tracer(name: str = "evoskill"):
    """Return an OpenTelemetry tracer (safe to call before or after init)."""
    from opentelemetry import trace

    return trace.get_tracer(name)
