"""Benchmark data overlay: repaired copies of SpreadsheetBench v2 input files, substituted BYTE-WISE at upload/hash
time.

Every path stays the canonical one (<data root>/...): the solver's prompt, the cache keys (question text),
dataset.json, the plug-in's task list and every mirror path inside a sandbox are unchanged. Only the bytes of the
files PATCHES.json lists come from the patched copy. Declared by EVOSKILL_SSBV2_DATA_OVERLAY=<PATCHES.json>; unset =
no overlay (identity). Each entry is verified on load: the canonical file must still hash to sha256_before and the
patched one to sha256_after, or OverlayError is raised (a drifted file is never substituted silently).

PATCHES.json names two trees. `dst` (the patched copy) may be relative: it resolves against the directory that holds
PATCHES.json, so the overlay shipped in examples/ssbv2/overlay/ works from any checkout. `src` (the canonical data
root) may be an absolute path, or null: then it is the data root of the run, passed as `root` or, when none is
passed, read from the environment the way daytona_evolver.data_root() reads it ($EVOSKILL_SSBV2_DATA, else
$SSBV2_ROOT/data/spreadsheetbench-v2; the loop exports INFRA.json data.root as EVOSKILL_SSBV2_DATA).

OPERATOR-OWNED: a run with an INFRA.json declares the overlay there (block "data": root, overlay_patches_json,
overlay_patches_sha256; src/orchestrator/infra.py). The leg and grade drivers then take the variable FROM INFRA.json
(set when unset, refused when it differs, refused when set while INFRA declares none) and pass the path explicitly;
infra.agree() re-verifies every entry with verify() (uncached). A run without INFRA.json keeps the bare variable.

Users: scripts/daytona/cc_leg_driver.py, scripts/daytona/cc_pool_host.py, scripts/daytona/ssb_leg_daytona_loop.py
(bundle upload), scripts/daytona/ssb_grade_daytona.py (job shas + bundle upload). stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ENV = "EVOSKILL_SSBV2_DATA_OVERLAY"
DATA_ROOT_ENV, SSBV2_ROOT_ENV = "EVOSKILL_SSBV2_DATA", "SSBV2_ROOT"     # daytona_evolver.data_root()'s two variables
SSBV2_DATA_SUFFIX = "data/spreadsheetbench-v2"
_LOADED: dict = {}


class OverlayError(RuntimeError):
    pass


def _sha(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _key(p) -> str:
    return os.path.normpath(os.path.abspath(str(p)))


def patches_path(patches_json=None) -> str:
    return str(patches_json or os.environ.get(ENV) or "")


def same_path(a, b) -> bool:
    """Two spellings of one PATCHES.json path (normalised absolute), as load() keys them."""
    return bool(a) and bool(b) and _key(a) == _key(b)


def env_root():
    """The data root the environment names ($EVOSKILL_SSBV2_DATA, else $SSBV2_ROOT/data/spreadsheetbench-v2), or
    None."""
    if os.environ.get(DATA_ROOT_ENV):
        return Path(os.environ[DATA_ROOT_ENV])
    if os.environ.get(SSBV2_ROOT_ENV):
        return Path(os.environ[SSBV2_ROOT_ENV]) / SSBV2_DATA_SUFFIX
    return None


def trees(pj, rec: dict, root=None) -> tuple:
    """(canonical root, patched root) of a parsed PATCHES.json: `src` null = `root`, else the environment's data root;
    a relative `dst` resolves against the directory of PATCHES.json. OverlayError when `src` is null and no root is
    known."""
    src, dst = rec.get("src"), Path(rec["dst"])
    if src is None:
        src = root if root else env_root()
        if not src:
            raise OverlayError(f"overlay {pj}: src is null (the run's data root) and no root is known: declare "
                               f"INFRA.json data.root or set {DATA_ROOT_ENV}")
    if not dst.is_absolute():
        dst = Path(os.path.abspath(pj)).parent / dst
    return Path(src), dst


def _read(pj: str, root=None) -> dict:
    """Parse and verify one PATCHES.json, uncached. Every failure is an OverlayError."""
    try:
        rec = json.loads(Path(pj).read_text())
        src, dst = trees(pj, rec, root)
        entries = [(f["path"], f["sha256_before"], f["sha256_after"]) for f in rec["files"]]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise OverlayError(f"overlay {pj}: unreadable ({type(e).__name__}: {e})")
    m = {}
    for rel, before, after in entries:
        if not isinstance(rel, str) or rel.startswith("/") or ".." in Path(rel).parts:
            raise OverlayError(f"overlay {pj}: entry path {rel!r} must be relative, without '..'")
        a, b = src / rel, dst / rel
        if not a.is_file() or _sha(a) != before:
            raise OverlayError(f"overlay {pj}: canonical {a} no longer hashes to sha256_before")
        if not b.is_file() or _sha(b) != after:
            raise OverlayError(f"overlay {pj}: patched {b} does not hash to sha256_after")
        m[_key(a)] = _key(b)
    return m


def load(patches_json=None) -> dict:
    """{canonical path: patched path} (normalised absolute strings); {} when no overlay is declared.
    Verified once per process and path (cached)."""
    pj = patches_path(patches_json)
    if not pj:
        return {}
    pj = _key(pj)
    if pj not in _LOADED:
        _LOADED[pj] = _read(pj)
    return _LOADED[pj]


def verify(patches_json, root=None) -> dict:
    """load() without the cache: every entry re-hashed NOW (infra.agree, the drivers' config check). `root` is the
    canonical data root a null `src` stands for (default: the environment's). Refreshes the cache with the result."""
    pj = _key(patches_json)
    _LOADED[pj] = _read(pj, root)
    return _LOADED[pj]


def source(p, patches_json=None) -> Path:
    """The file whose BYTES stand for canonical path p (p itself unless the overlay lists it)."""
    return Path(load(patches_json).get(_key(p), str(p)))


def record(patches_json=None) -> dict | None:
    """What a result records about the overlay; None when none is declared."""
    pj = patches_path(patches_json)
    if not pj:
        return None
    m = load(pj)
    return {"env": ENV, "patches_json": _key(pj), "patches_json_sha256": _sha(pj),
            "files": [{"canonical": a, "patched": b, "sha256": _sha(b)} for a, b in sorted(m.items())]}
