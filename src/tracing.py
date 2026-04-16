"""Phoenix observability for EvoSkill.

Call `init_tracing()` early in your entry point — before any LLM SDK imports —
so that Phoenix auto-instrumentation can patch the Anthropic client.
"""

import os

_initialized = False


def init_tracing(project_name: str = "evoskill") -> None:
    """Register Phoenix OTEL tracing (idempotent)."""
    global _initialized
    if _initialized:
        return

    # Allow Phoenix to capture full base64 images (default 32KB is too small)
    os.environ.setdefault("OPENINFERENCE_BASE64_IMAGE_MAX_LENGTH", "2000000")
    os.environ.setdefault("OTEL_EXPORTER_OTLP_TRACES_PROTOCOL", "http/protobuf")

    from phoenix.otel import register as phoenix_register

    phoenix_register(
        project_name=project_name,
        auto_instrument=True,
        protocol="http/protobuf",
    )
    _initialized = True


def get_tracer(name: str = "evoskill"):
    """Return an OpenTelemetry tracer (safe to call before or after init)."""
    from opentelemetry import trace

    return trace.get_tracer(name)
