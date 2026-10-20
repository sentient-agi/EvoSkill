"""The OpenRouter key of a tau3 run: OPENROUTER_API_KEY from the environment, else from the repository's .env.

The key pays for the tau3 customer (tau2's user simulator) and for KB_search_dense's query embeddings. It is read into
the calling process's memory only: never printed, never put in an argv, a file or a machine's environment (a tau3 leg
hands it to each bridge on fd 4). Standard library only, so the tau3-bench venv can import it too
(scripts/tau3_build_embeddings_cache.py).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ENV_FILE = REPO / ".env"
NAME = "OPENROUTER_API_KEY"


class KeyMissing(RuntimeError):
    """No usable OpenRouter key (the message never holds the key)."""


def env_file_value(name: str, path: Path = ENV_FILE) -> str | None:
    """`name`'s value in a dotenv file (KEY=VALUE lines, an optional `export `, optional quotes, # comments), or None
    when the file or the name is absent."""
    p = Path(path)
    if not p.is_file():
        return None
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("export "):
            s = s[len("export "):].lstrip()
        k, sep, v = s.partition("=")
        if not sep or k.strip() != name:
            continue
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            return v[1:-1]
        return v.split(" #", 1)[0].strip()
    return None


def openrouter_key(env_file: Path = ENV_FILE) -> str:
    """The key: the environment's OPENROUTER_API_KEY, else the one in `env_file` (the repository's .env)."""
    k = (os.environ.get(NAME) or "").strip() or (env_file_value(NAME, env_file) or "").strip()
    if not k:
        raise KeyMissing(f"{NAME} is set neither in the environment nor in {env_file}")
    if any(c.isspace() for c in k):
        raise KeyMissing(f"{NAME} holds whitespace (the key is not printed)")
    return k
