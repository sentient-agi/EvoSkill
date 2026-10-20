"""The PRODUCING LIBRARY of a digest surface, and the skill-load markers that point at it.

WHY THIS EXISTS. A digest builder collapses every skill-load echo in a rollout into a one-line
marker. A marker that says the body "is the file you were handed in your staging dir" is true for
exactly one surface: the newest, digests_eK_train, whose rollouts ran under skills_v{K-1}, the set
seeded into staging. Once the epoch-K evolver can read EVERY earlier epoch's digests (RUNTIME
`evolver_evidence: "all_epochs"`), such a marker is false on digests_e2 .. digests_e{K-1}. Their
rollouts ran under v1 .. v{K-2}, and the staging dir holds a same-named skill whose text may differ
or may not exist at all (a skill can be loaded often in one epoch and deleted in the next). The
evolver would credit behaviour in an old rollout to the current text of a same-named skill, which
corrupts attribution, the main reason for showing it every epoch.

So a surface built in all-epochs mode carries the library that produced it:

  * a byte-identical copy of skills_v{j-1} under digests_e{j}_train/_library/. The name is
    never `skills_v<N>`: the peer-skill hook denies any path component with that shape, and the
    peer-artifact audit reads it as a peer's answers. digests_e1_train has none, because the base solver
    carried no skills.
  * a header line in every task file naming the library and its hash;
  * version-anchored markers:
        [SKILL LOADED: <name> — <bytes> B; library skills_v1 sha256:<12>; body in
         ./_library/<name>/SKILL.md of this digest dir]
    <name> stays the FIRST token after "SKILL LOADED:", because skill_delivery_shim's SSB dialect
    parses it from there. A marker that put the version first would read 'v1' as a
    mistyped skill name.

The run's own skills_v* directories are still NOT granted. The copy travels inside the digest dir
the evolver is already granted, so no kernel grant or hook changes.

ONE DEFINITION of the library hash. `library_sha256` is loop.staging_sha256 (loop delegates here):
the promotion check and the countersignature use it, and a second implementation is how a
signature ends up covering something other than what was promoted. The <12> in a marker is
therefore the same hash the proofread names for that skill set.

NO PACKAGE IMPORTS, and Python 3.9 compatible: both builders load this file by path under
whatever `python3` runs them, and the tests load it by path so it can be mutated in a copy.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

LIBRARY_DIR = "_library"
LABEL_RE = re.compile(r"^skills_v(\d+)$")
# Top-level files the builders may write that are not per-task digests (same set as
# loop._v_digests_train_only, evidence_surface.SIDECARS and digest_coverage).
SIDECARS = frozenset({"REGRESSIONS", "README", "INDEX"})
# The two legacy marker texts (written before markers named their library). Each tells the reader the body is in its staging dir,
# which is false on every surface but the newest.
LEGACY_MARKER_TEXTS = (
    "it is the file you were handed in your staging dir",       # build_rollout_digests.py
    "Body omitted: it is the file in your staging dir",         # tau3_build_digests.py
)
MARKER_RE = re.compile(r"\[SKILL LOADED:[^\]\n]*\]")
_FRONTMATTER_NAME = re.compile(r"\A\s*---\s*\n(?:[^\n]*\n)*?name:\s*([^\n]+)\n")


# ── the hash (one definition; loop._skill_fingerprint / staging_sha256 delegate here) ──────────
def fingerprint(root) -> dict:
    """Content digest of a skill set, keyed by skill name: {dir name: sha256(SKILL.md)[:16]}."""
    return {f.parent.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16]
            for f in sorted(Path(root).glob("*/SKILL.md"))}


def library_sha256(root) -> str:
    """A single stable hash over a whole skill set (== loop.staging_sha256)."""
    return hashlib.sha256(json.dumps(fingerprint(root), sort_keys=True).encode()).hexdigest()


def symlinks(root) -> list:
    """Every symlink under root, files and directories, never followed."""
    out = []
    for base, dirs, files in os.walk(str(root), followlinks=False):
        for n in list(dirs) + list(files):
            p = os.path.join(base, n)
            if os.path.islink(p):
                out.append(p)
    return sorted(out)


def tree(root) -> dict:
    """{posix relative path: sha256 of the bytes} for every regular file under root.

    Unlike `fingerprint`, which covers the SKILL.md files the countersignature names, this covers
    every file, so a snapshot can be proven byte-identical to its source.
    """
    root = Path(root)
    out = {}
    for base, _dirs, files in os.walk(str(root), followlinks=False):
        for n in files:
            p = Path(base) / n
            if p.is_symlink() or not p.is_file():
                continue
            out[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


# ── naming ──────────────────────────────────────────────────────────────────────────────────────
def label_of(library) -> str:
    """The library's label, which is its directory name. It must be a promoted set, skills_v<N>:
    a staging dir or anything else is not a library that produced a batch."""
    name = Path(str(library).rstrip("/")).name
    if not LABEL_RE.match(name):
        raise ValueError(f"{library} is not a promoted skill set (skills_v<N>); a surface's "
                         f"producing library is the set its solver ran with")
    return name


def label_for_epoch(j: int) -> str:
    """digests_e{j}_train was produced by skills_v{j-1}; j == 1 is the base leg (no library)."""
    if j < 2:
        raise ValueError(f"epoch {j}'s surface is the base leg's; it has no producing library")
    return f"skills_v{j - 1}"


def _norm(name: str) -> str:
    return name.strip().strip("'\"").strip()


def skills(root) -> dict:
    """{name: (posix path relative to root, size in bytes)} for every SKILL.md under root.

    Keyed by the skill's directory name AND by its frontmatter `name:` when they differ. The SSB
    builder matches echoes by directory name, the tau3 builder by the echoed frontmatter name.
    """
    root = Path(root)
    out = {}
    for f in sorted(root.rglob("SKILL.md")):
        if f.is_symlink():
            continue
        rel = f.relative_to(root).as_posix()
        size = len(f.read_bytes())
        out.setdefault(f.parent.name, (rel, size))
        m = _FRONTMATTER_NAME.match(f.read_text(errors="replace"))
        if m and _norm(m.group(1)):
            out.setdefault(_norm(m.group(1)), (rel, size))
    return out


def lookup(lib_skills: dict, name: str):
    return lib_skills.get(name) or lib_skills.get(_norm(name))


# ── the text the builders write ─────────────────────────────────────────────────────────────────
def marker(name: str, nbytes: int, label: str, sha12: str, rel: str) -> str:
    """The version-anchored skill-load marker. `name` stays the first token (skill_delivery_shim)."""
    return (f"[SKILL LOADED: {name} — {nbytes} B; library {label} sha256:{sha12}; "
            f"body in ./{LIBRARY_DIR}/{rel} of this digest dir]")


def not_found_marker(name: str, nbytes: int, label: str, sha12: str) -> str:
    """A load whose frontmatter names no skill of the producing library. Says so, and carries no
    `body in` pointer, so the loop's library check refuses the surface rather than let a marker
    point at a body that is not there."""
    return (f"[SKILL LOADED: {name} — {nbytes} B echoed back here; NOT FOUND in the producing "
            f"library {label} sha256:{sha12}, so no body is attached]")


def header_line(label: str, sha12: str) -> str:
    """One line under the Result line of every task file. On its own line, never folded into the
    Result line: that line is parsed per key by the launcher, derive_facts and #208."""
    return (f"**Producing library:** `{label}` sha256:{sha12}. The exact files this solver was "
            f"given are in ./{LIBRARY_DIR}/ of this digest dir; every skill-load marker below "
            f"names this library.")


def anchor(label: str, sha12: str) -> str:
    """The substring every marker of a surface produced by (label, sha12) must carry."""
    return f"; library {label} sha256:{sha12}; body in ./{LIBRARY_DIR}/"


# ── writing the snapshot ────────────────────────────────────────────────────────────────────────
def snapshot(library, digest_dir) -> tuple:
    """Copy the producing library into <digest_dir>/_library/. Returns (label, sha12).

    Refuses (ValueError) on a missing library, a library that is not skills_v<N>, any symlink in
    it (the evidence dir must hold none: the kernel would resolve it and a reader outside the
    sandbox would follow it), or an existing _library/ with different bytes. The copy is
    compared file by file with its source afterwards. That checks what was written, not only the
    call that wrote it.
    """
    lib = Path(library)
    if lib.is_symlink() or not lib.is_dir():
        raise ValueError(f"no producing library at {lib} (or it is a symlink)")
    label = label_of(lib)
    links = symlinks(lib)
    if links:
        raise ValueError(f"the producing library {lib} holds {len(links)} symlink(s) "
                         f"{links[:3]}; refusing to copy links into an evidence dir")
    want = tree(lib)
    dest = Path(digest_dir) / LIBRARY_DIR
    if dest.is_symlink():
        raise ValueError(f"{dest} is a symlink")
    if dest.exists():
        if tree(dest) != want or symlinks(dest):
            raise ValueError(f"{dest} already exists and is NOT byte-identical to {lib}. "
                             f"Move the digest dir aside and rebuild; never overwrite evidence "
                             f"an evolver may have read.")
    else:
        Path(digest_dir).mkdir(parents=True, exist_ok=True)
        shutil.copytree(str(lib), str(dest), symlinks=True)
    if tree(dest) != want or symlinks(dest):
        raise ValueError(f"the copy at {dest} does not match {lib} byte for byte")
    return label, library_sha256(lib)[:12]


# ── checking a surface ──────────────────────────────────────────────────────────────────────────
def problems(digest_dir, library=None, label: str = "") -> list:
    """Why `digest_dir` does NOT honestly carry its producing library. [] means it does.

    library None: a BASE surface (digests_e1_train). It must hold no _library/ and no skill-load
    marker at all, because a base solver is never given a skill.
    library given: the surface must hold _library/ (a real dir, no symlinks) byte-identical to
    `library`, every per-task file must name (label, sha) in its header, every skill-load marker
    must carry that anchor and point at a file that exists under _library/, and no marker may be
    one of the legacy "it is in your staging dir" texts.
    """
    d = Path(digest_dir)
    if not d.is_dir():
        return [f"no digest dir at {d}"]
    lib_copy = d / LIBRARY_DIR
    files = sorted(p for p in d.glob("*.md") if p.stem not in SIDECARS)
    probs = []
    if library is None:
        if lib_copy.exists() or lib_copy.is_symlink():
            probs.append(f"{d.name} is a base surface but holds a {LIBRARY_DIR}/")
        n = sum(len(MARKER_RE.findall(p.read_text(errors="replace"))) for p in files)
        if n:
            probs.append(f"{d.name} is a base surface but carries {n} skill-load marker(s)")
        return probs
    lib = Path(library)
    label = label or label_of(lib)
    if lib.is_symlink() or not lib.is_dir():
        return [f"{d.name}: the producing library {label} is not on disk at {lib}, so what "
                f"{LIBRARY_DIR}/ claims to be cannot be checked"]
    if lib_copy.is_symlink() or not lib_copy.is_dir():
        return [f"{d.name} holds no {LIBRARY_DIR}/ (it was built without its producing "
                f"library {label}, so its skill-load markers name no version)"]
    links = symlinks(lib_copy)
    if links:
        probs.append(f"{d.name}/{LIBRARY_DIR} holds {len(links)} symlink(s)")
    if tree(lib_copy) != tree(lib):
        probs.append(f"{d.name}/{LIBRARY_DIR} is NOT byte-identical to {label}")
    sha12 = library_sha256(lib)[:12]
    head = header_line(label, sha12)
    no_head, legacy, unanchored, dangling = [], [], 0, []
    anc = anchor(label, sha12)
    for p in files:
        t = p.read_text(errors="replace")
        if head not in t:
            no_head.append(p.name)
        if any(x in t for x in LEGACY_MARKER_TEXTS):
            legacy.append(p.name)
        for mk in MARKER_RE.findall(t):
            if anc not in mk:
                unanchored += 1
                continue
            rel = mk.split(f"./{LIBRARY_DIR}/", 1)[1].split(" of this digest dir", 1)[0]
            if not (lib_copy / rel).is_file():
                dangling.append(rel)
    if no_head:
        probs.append(f"{len(no_head)} task file(s) in {d.name} do not name the producing "
                     f"library {label} sha256:{sha12} in their header, e.g. {no_head[:3]}")
    if legacy:
        probs.append(f"{len(legacy)} task file(s) in {d.name} still carry the legacy marker "
                     f"that says the body is in the staging dir, e.g. {legacy[:3]}")
    if unanchored:
        probs.append(f"{unanchored} skill-load marker(s) in {d.name} do not name {label} "
                     f"sha256:{sha12} with a body under {LIBRARY_DIR}/")
    if dangling:
        probs.append(f"{len(dangling)} marker(s) in {d.name} point at a body missing from "
                     f"{LIBRARY_DIR}/: {sorted(set(dangling))[:3]}")
    return probs
