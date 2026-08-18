"""Direct API harness — no SDK, no background server.

Calls OpenRouter (or any OpenAI-compatible endpoint) directly via httpx
with multi-turn tool calling. Self-contained: only depends on httpx,
subprocess, pathlib, and Python stdlib.
"""

from .options import (
    apply_provider_pin, build_direct_options, pin_request_body_provider,
)
from .executor import execute_query, parse_response

__all__ = [
    "apply_provider_pin",
    "pin_request_body_provider",
    "build_direct_options",
    "execute_query",
    "parse_response",
]
