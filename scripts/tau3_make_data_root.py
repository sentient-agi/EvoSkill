#!/usr/bin/env python3
"""Build the curated tau3 data root: the banking_knowledge data tau2 reads, and nothing else, with a sha manifest.

The user of a tau3 run hands the adapter one data root, and the same copy serves the rollouts (TAU2_DATA_DIR), the
grading (the grader derives TAU2_DATA_DIR from a task's answer file) and the adapter's /data view. It is copied from
the tau3-bench checkout at commit 363133a (tau2-bench 1.0.1):

  tau2/domains/banking_knowledge/db.json      the bank's database               (banking_knowledge/utils.py:43)
  tau2/domains/banking_knowledge/documents/   the knowledge-base articles       (utils.py:39, environment.py:31)
  tau2/domains/banking_knowledge/prompts/     the agent policy per retrieval setup, with components/ (retrieval.py:55)
  tau2/domains/banking_knowledge/tasks/       one task_NNN.json per task        (utils.py:40, environment.py:94-124)
  tau2/user_simulator/                        the simulated customer's guidelines (user/user_simulator.py:29-46)
  MANIFEST.sha256                             `sha256sum` lines of every file above, sorted by relative path

DERIVED ROWS (tau3 retrieval = tau2's "alltools"). scripts/tau3_build_embeddings_cache.py computes the knowledge
base's document embeddings once into <root>/.embeddings_cache/ (tau2's EmbeddingsCache format: one <document
hash>_<embedder hash>.pkl and metadata.json) and adds a row per file to MANIFEST.sha256. Those rows are derived, not
tracked by git, so this script never writes them; it accepts them:
  - a fresh build writes the git rows only (the precompute adds the cache afterwards);
  - --check: every row outside .embeddings_cache/ must equal the checkout's exactly as before (same paths, same shas,
    none missing, none extra); rows under .embeddings_cache/ are verified by sha against the files like every other
    row (verify_root), and their number is reported;
  - building onto an existing root whose rows outside .embeddings_cache/ equal the checkout's and whose files match
    its manifest is "unchanged" (never refused, never touched), cache rows or not.
A row anywhere else that git does not track, a cache file the manifest does not list, or a sha that differs is
refused as before.

Left out: banking_knowledge/tasks.json and tasks_voice.json. They are stale aggregates of every task, validation and
test included; tau2 reads neither for this domain (get_tasks globs tasks/task_*.json), and a copy of them would leak
held-out tasks into the adapter's reference/. The other domains, voice/ and results/ are left out too: importing
tau2 only builds their paths.

The files copied are exactly the ones git tracks at 363133a under the kept paths, and each one's bytes must hash to
its blob at that commit (the checkout is read, never changed: read-only git with --no-optional-locks). A file on disk
that the commit does not track, or a byte that differs, refuses the build.

  tau3_make_data_root.py --dest D             build D (e.g. <somewhere>/tau3_data_363133a; INFRA data.root)
  tau3_make_data_root.py --dest D --check     verify D against the checkout; writes nothing

Building is idempotent: a root whose MANIFEST.sha256 rows outside .embeddings_cache/ equal the ones computed from the
checkout, and whose files all match it with nothing extra, is left untouched ("unchanged"). Any other existing path
at D (a different manifest, no manifest, a changed or extra file) is never overwritten: rc 3. A new root is written
into a temporary sibling, verified against its manifest, then renamed into place, so D is either absent or complete.
rc 0: created, unchanged or checked. rc 3: refused (the reason is on stderr). Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
TAU3 = REPO / "third_party" / "tau3-bench"
COMMIT = "363133ada1936491fb5bcec33cd62c3518a99f65"      # tau3-bench at tau2-bench 1.0.1
SRC_PREFIX = "data/"                                     # the checkout's data dir, relative to TAU3
MANIFEST = "MANIFEST.sha256"

DOMAIN = "tau2/domains/banking_knowledge"
KEEP = (                                                 # relative to the data dir: files, or dirs taken whole
    f"{DOMAIN}/db.json",
    f"{DOMAIN}/documents",
    f"{DOMAIN}/prompts",
    f"{DOMAIN}/tasks",
    "tau2/user_simulator",
)
LEFT_OUT = (f"{DOMAIN}/tasks.json", f"{DOMAIN}/tasks_voice.json")   # the stale aggregates; never copied
CACHE_DIR = ".embeddings_cache"                          # = scripts/tau3_cc/tau3_retrieval.CACHE_DIRNAME (derived rows)


class Refused(RuntimeError):
    """The root cannot be built or checked as asked; nothing was written."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _kept(rel: str) -> bool:
    return rel not in LEFT_OUT and any(rel == k or rel.startswith(k + "/") for k in KEEP)


def _git(*args: str) -> bytes:
    r = subprocess.run(["git", "--no-optional-locks", "-C", str(TAU3), *args], capture_output=True)
    if r.returncode != 0:
        raise Refused(f"git {' '.join(args)} failed in {TAU3}: {r.stderr.decode(errors='replace').strip()}")
    return r.stdout


def plan() -> list[tuple[str, str, int]]:
    """(relative path, sha256, bytes) of every file the root holds besides the manifest, sorted by path. Each file
    is a regular file tracked at COMMIT whose checkout bytes equal its blob; the kept dirs hold no other file."""
    if _git("rev-parse", "--verify", f"{COMMIT}^{{commit}}").decode().strip() != COMMIT:
        raise Refused(f"{TAU3} does not resolve commit {COMMIT}")
    tree = _git("ls-tree", "-r", "-z", "--full-tree", COMMIT, "--", *(SRC_PREFIX + k for k in KEEP))
    tracked: dict[str, str] = {}
    for rec in tree.split(b"\0"):
        if not rec:
            continue
        meta, path = rec.split(b"\t", 1)
        mode, kind, oid = meta.decode().split()
        full = path.decode()
        rel = full[len(SRC_PREFIX):]
        if rel in LEFT_OUT and full.startswith(SRC_PREFIX):
            continue
        if not full.startswith(SRC_PREFIX) or not _kept(rel):
            raise Refused(f"git listed a path outside the kept set: {full}")
        if (mode, kind) != ("100644", "blob"):
            raise Refused(f"{rel}: mode {mode} {kind} at {COMMIT[:7]}, expected a plain file (100644 blob)")
        if "\\" in rel or "\n" in rel:
            raise Refused(f"{rel!r}: a name sha256sum would escape")
        tracked[rel] = oid
    if not tracked:
        raise Refused(f"no file tracked at {COMMIT[:7]} under {KEEP}")
    for k in KEEP:
        if not any(r == k or r.startswith(k + "/") for r in tracked):
            raise Refused(f"{k}: nothing tracked at {COMMIT[:7]}")

    src = TAU3 / SRC_PREFIX
    on_disk = set()
    for k in KEEP:
        p = src / k
        if p.is_symlink():
            raise Refused(f"{p} is a symlink")
        if p.is_file():
            on_disk.add(k)
            continue
        for q in p.rglob("*"):
            if q.is_symlink() or not (q.is_file() or q.is_dir()):
                raise Refused(f"{q} is not a plain file or dir")
            if q.is_file():
                on_disk.add(q.relative_to(src).as_posix())
    on_disk -= set(LEFT_OUT)
    if on_disk != set(tracked):
        extra, missing = sorted(on_disk - set(tracked)), sorted(set(tracked) - on_disk)
        raise Refused(f"checkout differs from {COMMIT[:7]}: untracked {extra[:5]} missing {missing[:5]}")

    out = []
    for rel in sorted(tracked):
        data = (src / rel).read_bytes()
        if _git_blob(data) != tracked[rel]:
            raise Refused(f"{rel}: the checkout's bytes differ from {COMMIT[:7]}")
        out.append((rel, hashlib.sha256(data).hexdigest(), len(data)))
    return out


def manifest_text(rows: list[tuple[str, str, int]]) -> str:
    return "".join(f"{sha}  {rel}\n" for rel, sha, _ in rows)


def parse_manifest(text: str) -> dict[str, str]:
    """{relative path: sha256} from sha256sum lines; raises Refused on a malformed, unsorted or repeated line."""
    rows, prev = {}, None
    for n, line in enumerate(text.splitlines(), 1):
        sha, sep, rel = line.partition("  ")
        if not sep or len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha) or not rel:
            raise Refused(f"{MANIFEST} line {n} is not a sha256sum line")
        if rel in rows or (prev is not None and rel < prev):
            raise Refused(f"{MANIFEST} line {n}: {rel} repeated or out of order")
        rows[rel], prev = sha, rel
    return rows


def is_cache_row(rel: str) -> bool:
    """A derived row: a file under <root>/.embeddings_cache/ (written by scripts/tau3_build_embeddings_cache.py)."""
    return rel.startswith(CACHE_DIR + "/")


def checkout_rows_match(rows: dict[str, str], planned: list[tuple[str, str, int]]) -> None:
    """The manifest's rows outside .embeddings_cache/ are exactly the checkout's (paths and shas); raises Refused."""
    git = {rel: sha for rel, sha, _ in planned}
    mine = {rel: sha for rel, sha in rows.items() if not is_cache_row(rel)}
    if mine != git:
        extra, missing = sorted(set(mine) - set(git)), sorted(set(git) - set(mine))
        changed = sorted(r for r in set(mine) & set(git) if mine[r] != git[r])
        raise Refused(f"{MANIFEST} differs from the one {COMMIT[:7]} gives outside {CACHE_DIR}/: extra {extra[:5]} "
                      f"missing {missing[:5]} changed {changed[:5]}")


def verify_root(root: Path) -> dict[str, str]:
    """The root's files match its own manifest exactly: every listed file present with its sha, nothing else
    (MANIFEST.sha256 aside), no symlink. Returns the manifest rows."""
    mf = root / MANIFEST
    if not mf.is_file() or mf.is_symlink():
        raise Refused(f"{root} has no {MANIFEST}")
    rows = parse_manifest(mf.read_text(encoding="utf-8"))
    present = set()
    for q in root.rglob("*"):
        if q.is_symlink() or not (q.is_file() or q.is_dir()):
            raise Refused(f"{q} is not a plain file or dir")
        if q.is_file():
            present.add(q.relative_to(root).as_posix())
    present.discard(MANIFEST)
    if present != set(rows):
        extra, missing = sorted(present - set(rows)), sorted(set(rows) - present)
        raise Refused(f"{root} differs from its manifest: extra {extra[:5]} missing {missing[:5]}")
    for rel, sha in rows.items():
        if _sha256(root / rel) != sha:
            raise Refused(f"{root / rel}: sha differs from its manifest")
    return rows


def _write(rows: list[tuple[str, str, int]], text: str, tmp: Path) -> None:
    src = TAU3 / SRC_PREFIX
    for rel, sha, _ in rows:
        dst = tmp / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src / rel, dst)
        os.chmod(dst, 0o644)
        if _sha256(dst) != sha:
            raise Refused(f"{rel}: the copy's sha differs (the checkout changed while copying?)")
    (tmp / MANIFEST).write_text(text, encoding="utf-8")
    os.chmod(tmp / MANIFEST, 0o644)
    for d in [tmp, *(p for p in tmp.rglob("*") if p.is_dir())]:
        os.chmod(d, 0o755)


def build(dest: Path) -> str:
    """Create dest from the checkout, or confirm it already is that root (its derived .embeddings_cache rows aside).
    Returns 'created' or 'unchanged'."""
    rows = plan()
    text = manifest_text(rows)
    if dest.exists() or dest.is_symlink():
        mf = dest / MANIFEST
        if dest.is_symlink() or not dest.is_dir():
            raise Refused(f"{dest} exists and is not a directory")
        if not mf.is_file() or mf.is_symlink():
            raise Refused(f"{dest} exists and has no {MANIFEST}; not overwriting")
        try:
            checkout_rows_match(parse_manifest(mf.read_text(encoding="utf-8")), rows)
        except Refused as e:
            raise Refused(f"{dest} exists and {e}; not overwriting") from None
        verify_root(dest)
        return "unchanged"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{dest.name}.tmp-", dir=dest.parent))
    try:
        _write(rows, text, tmp)
        verify_root(tmp)
        if dest.exists() or dest.is_symlink():
            raise Refused(f"{dest} appeared while building; not overwriting")
        os.rename(tmp, dest)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return "created"


def check(dest: Path) -> dict[str, str]:
    """dest is the root this checkout builds: its rows outside .embeddings_cache/ are the checkout's, and every file
    (cache files included) matches its row, nothing unlisted. Writes nothing. Returns the manifest rows."""
    planned = plan()
    rows = verify_root(dest)
    try:
        checkout_rows_match(rows, planned)
    except Refused as e:
        raise Refused(f"{dest}: {e}") from None
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--dest", type=Path, required=True, help="the data root to build or check")
    ap.add_argument("--check", action="store_true", help="verify --dest against the checkout; write nothing")
    a = ap.parse_args(argv)
    dest = a.dest.expanduser().absolute()
    try:
        if a.check:
            check(dest)
            status = "checked"
        else:
            status = build(dest)
    except (Refused, UnicodeDecodeError) as e:           # a manifest that is not text is not ours either
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    rows = parse_manifest((dest / MANIFEST).read_text(encoding="utf-8"))
    n_bytes = sum((dest / rel).stat().st_size for rel in rows) + (dest / MANIFEST).stat().st_size
    n_cache = sum(map(is_cache_row, rows))
    print(f"{status}: {dest} ({len(rows)} data files + {MANIFEST}: {len(rows) - n_cache} from git, {n_cache} under "
          f"{CACHE_DIR}/; {n_bytes} bytes; manifest sha256 {_sha256(dest / MANIFEST)}; tau3-bench {COMMIT[:7]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
