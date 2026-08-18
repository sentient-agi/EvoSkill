"""
Program-aware run caching module.

Cache keys on git tree hash of .claude/ directory + question hash.
Identical content (skills, prompts, config) = cache hit.
"""

from .run_cache import RunCache, CacheConfig

__all__ = ["RunCache", "CacheConfig"]
