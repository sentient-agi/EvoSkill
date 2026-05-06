"""Harbor harness — run the base agent inside Harbor sandboxes.

Unlike the other harness submodules (claude/, opencode/, ...) which are SDK
adapters that talk directly to LLMs, this one delegates execution to the
`harbor` CLI. The agent's "query" is a Harbor task id; the response is a JSON
envelope wrapping the verifier's reward.
"""

from .agent import HarborAgent, HarborRunError

__all__ = ["HarborAgent", "HarborRunError"]
