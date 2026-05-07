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
