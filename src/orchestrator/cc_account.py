"""The ONE Claude Code credential every `claude -p` session of a run uses.

A run authenticates its solver attempts, its evolver and its benchmark adaptor with one credential, either:

  * a subscription token file: the run's declared token file (INFRA claude_code.token_file / RUNTIME
    evolver_daytona.oauth_token_file), else $EVOSKILL_CC_TOKEN_FILE. A regular file (not a symlink) readable by its
    owner only (mode 0600). Drivers read it in-process and hand the token to Claude Code on a file descriptor;
    it never appears on a command line, in an environment record or in a log; or
  * $ANTHROPIC_API_KEY, used only when no token file is given.

The token file wins when both are present, so an API key left in the launch shell never moves a token-file run onto
API billing. This module holds paths and the name of a variable only, never a secret: nothing here opens the token
file or reads the key's value beyond testing that it is non-empty.

    python -m src.orchestrator.cc_account          # which credential a session started now would use
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

TOKEN_FILE_ENV = "EVOSKILL_CC_TOKEN_FILE"
API_KEY_ENV = "ANTHROPIC_API_KEY"
TOKEN_FILE, API_KEY = "token_file", "api_key"
# What Claude Code reports as `apiKeySource` in its init record for each credential: a session on the OAuth token
# (CLAUDE_CODE_OAUTH_TOKEN, handed over on a file descriptor) reports "none" or the variable's name.
OAUTH_API_KEY_SOURCES = ("none", "CLAUDE_CODE_OAUTH_TOKEN")
API_KEY_SOURCES = (API_KEY_ENV,)


@dataclass(frozen=True)
class Credential:
    kind: str                       # TOKEN_FILE or API_KEY
    token_file: Path | None         # the token file for TOKEN_FILE, else None
    source: str                     # where it came from: "declared", TOKEN_FILE_ENV or API_KEY_ENV


def check_token_file(p) -> Path:
    """`p` as a Path when it is a usable token file (a regular file, not a symlink, owner-only). ValueError otherwise.
    Stat only: the file is never opened here."""
    p = Path(str(p))
    if p.is_symlink() or not p.is_file():
        raise ValueError(f"{p} is not a regular file")
    if p.stat().st_mode & 0o077:
        raise ValueError(f"{p} is readable or writable by group/other (want mode 0600)")
    return p


def resolve(declared=None, env=None, *, check: bool = True) -> Credential:
    """The credential a session starting now uses: the declared token file, else $EVOSKILL_CC_TOKEN_FILE, else
    $ANTHROPIC_API_KEY. ValueError when there is none, or (check=True) when the token file is unusable: a session
    must not start on a guess."""
    e = os.environ if env is None else env
    for path, source in ((declared, "declared"), (e.get(TOKEN_FILE_ENV), TOKEN_FILE_ENV)):
        if path and str(path).strip():
            p = Path(str(path).strip())
            return Credential(TOKEN_FILE, check_token_file(p) if check else p, source)
    if (e.get(API_KEY_ENV) or "").strip():
        return Credential(API_KEY, None, API_KEY_ENV)
    raise ValueError(f"no Claude Code credential: set {TOKEN_FILE_ENV} to a subscription token file (mode 0600) "
                     f"or {API_KEY_ENV}")


def uses_api_key(env=None, declared=None) -> bool:
    """True when the run authenticates with ANTHROPIC_API_KEY (no token file declared or in the environment).
    Never raises: no credential at all is False (the session that needs one refuses on its own)."""
    e = os.environ if env is None else env
    if (declared and str(declared).strip()) or (e.get(TOKEN_FILE_ENV) or "").strip():
        return False
    return bool((e.get(API_KEY_ENV) or "").strip())


def allowed_api_key_sources(env=None, declared=None) -> tuple:
    """The `apiKeySource` values a session of this run may report: the OAuth ones for a token-file run, plus
    ANTHROPIC_API_KEY for an API-key run."""
    return OAUTH_API_KEY_SOURCES + (API_KEY_SOURCES if uses_api_key(env, declared) else ())


def main(argv=None) -> int:
    try:
        c = resolve()
    except ValueError as e:
        print(str(e))
        return 1
    print(f"credential: {c.kind}" + (f" {c.token_file}" if c.token_file else "") + f" (from {c.source})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
