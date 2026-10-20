"""The evolver's EVIDENCE SURFACE when it spans several epochs (RUNTIME `evolver_evidence`).

By default the epoch-K evolver is granted exactly one digest dir, digests_eK_train, and every other
epoch's rollouts sit behind `--hide <run dir>`. A run that sets RUNTIME.json
`"evolver_evidence": "all_epochs"` grants digests_e1_train .. digests_eK_train of THIS run instead,
stored on disk and read on demand. Absent (or "current_epoch") keeps the single-dir behaviour.

This module is the ONE implementation of three things that would otherwise exist in several
places and drift:

  * which directories count as one run's evidence at iteration K, and whether a directory someone
    hands a launcher really is one of them (`validate`);
  * whether one of those directories is train-only and free of symlinks (`surface_problems`, used
    by the launcher AND by the loop's all-epochs train-only verify);
  * the content hash of an evidence dir (`dir_sha256`), recorded in the containment receipt BEFORE
    the evolver runs and again AFTER, so "the evolver was granted exactly these bytes and did not
    change them" is a recorded fact rather than an assumption.

THE LIST IS AN EXPLICIT, BOUNDED RANGE, NEVER A GLOB. Older run dirs hold siblings such as
digests_e2_train.062721.bak, digests_e0_train.pre_calcpr_fix, *.as_given and digests_probe_*; a
glob would grant them. Re-running an earlier epoch after a later surface exists would grant
digests_e{K+1}_train, i.e. rollouts of the very version being evolved. And any derivation from
spec.name, shared_dir, the oracle's parent or the base leg's parent can name ANOTHER run (a run
may reuse another run's spec name or base leg), which would hand the evolver another arm's
rollouts under another arm's skills. So a directory is evidence of run R at iteration K only if
it IS R/digests_e{j}_train for some 1 <= j <= K.

NO PACKAGE IMPORTS, and Python 3.9 compatible: the launcher runs it with whatever `python3` is on
PATH, and the tests load it by path so the mutation harness can mutate a throwaway copy.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

# Sidecars the digest builders may write next to the per-task files. Same set as
# loop._v_digests_train_only and digest_coverage.SIDECARS.
SIDECARS = frozenset({"REGRESSIONS", "README", "INDEX"})
# The rollout evidence (src/orchestrator/rollout_evidence.py, RUNTIME evolver_evidence_layout
# "rollouts"): full traces and grades per rollout under rollouts/<train stem>/, the
# solver's system prompt and tools under solver_context/. Judged here structurally and by content
# for EVERY dir, whatever the layout, so a hand-copied subtree cannot slip past the stem check.
ROLLOUTS_DIR = "rollouts"
CONTEXT_DIR = "solver_context"
_ROLLOUT_FILE = re.compile(r"^s\d+\.(?:trace\.jsonl|grade\.txt)$")
_CONTEXT_FILE = re.compile(r"^(?:system_prompt\.[0-9a-f]{12}\.txt|tools\.[0-9a-f]{12}\.json)$")
# Every split key that is NOT train. "dropped" and "excluded_prior_run" are records with task_ids.
NON_TRAIN_KEYS = ("val", "test", "dropped", "unused", "excluded_prior_run")
MODES = ("current_epoch", "all_epochs")
DIGEST_DIR = "digests_e{j}_train"
# Exit code the launcher uses for every evidence-list refusal.
REFUSE_RC = 21


def canonical_dir(run_dir, j: int) -> Path:
    """The ONLY name epoch j's evidence may have: <run_dir>/digests_e<j>_train (loop.EpochPaths)."""
    return Path(run_dir) / DIGEST_DIR.format(j=j)


def _inside(child: str, parent: str) -> bool:
    """True when realpath `child` is strictly inside realpath `parent` (path components, not a
    string prefix: /x/run_f2 is NOT inside /x/run)."""
    try:
        return os.path.commonpath([child, parent]) == parent and child != parent
    except ValueError:
        return False


def _split_stems(split_path):
    """(train, held_out, collisions) in FILENAME form ('/' -> '_'), the map the builders write."""
    sp = json.loads(Path(split_path).read_text())

    def ids(k):
        v = sp.get(k) or []
        return [x if isinstance(x, str) else x["id"] for x in v]

    every = [i for k in ("train", "val", "test") for i in ids(k)]
    stems = [i.replace("/", "_") for i in every]
    collide = sorted({s for s in stems if stems.count(s) > 1})
    train = {i.replace("/", "_") for i in ids("train")}
    held = {i.replace("/", "_") for k in ("val", "test") for i in ids(k)}
    return train, held, collide


def symlinks_under(d) -> list:
    """Every symlink anywhere under d (files AND directories), never followed. A hand-copied
    `<train_stem>.md -> <elsewhere>` passes a stem check; under the kernel profile the target is
    resolved and denied, but with the sandbox off, or to a reader outside it, it is arbitrary
    content wearing a train task's name."""
    out = []
    for root, dirs, files in os.walk(str(d), followlinks=False):
        for n in list(dirs) + list(files):
            p = os.path.join(root, n)
            if os.path.islink(p):
                out.append(p)
    return sorted(out)


def surface_problems(d, split_path) -> list:
    """Why directory `d` may NOT be served to an evolver as train evidence. [] means it may.

    Checks: it is a real directory (not a symlink), it contains no symlink anywhere, it holds at
    least one per-task *.md at top level, and every top-level *.md stem (sidecars excluded) is a
    TRAIN task of the split. Subdirectories (e.g. a producing library under _library/) and
    non-.md files (COVERAGE.json) are not task digests and are not judged as such.
    """
    d = Path(d)
    if os.path.islink(str(d)):
        return [f"{d} is a SYMLINK — the kernel profile would grant its target, not this path"]
    if not d.is_dir():
        return [f"no digest dir at {d}"]
    probs = []
    links = symlinks_under(d)
    if links:
        probs.append(f"{len(links)} symlink(s) inside {d.name}: {links[:3]} — refused, a link "
                     f"serves whatever it points at under a train task's name")
    try:
        train, held, collide = _split_stems(split_path)
    except Exception as e:                                      # noqa: BLE001
        return probs + [f"cannot read the split {split_path}: {type(e).__name__}: {e}"]
    if collide:
        return probs + [f"split ids collide when mapped to digest filenames: {collide[:6]} — "
                        f"cannot decide train-vs-held-out by filename"]
    have = {f.stem for f in d.glob("*.md")} - SIDECARS
    bad = sorted(have & held)
    if bad:
        probs.append(f"HELD-OUT MATERIAL IN {d.name}: {bad[:6]}")
    if not have:
        probs.append(f"{d.name} holds no per-task .md files (sidecars excluded: {sorted(SIDECARS)})")
    stray = sorted(have - train - held)
    if stray:
        probs.append(f"{d.name} holds digests that are NOT train tasks: {stray[:6]}")
    probs += rollouts_problems(d, split_path)
    probs += content_problems(d, split_path)
    return probs


# ── non-train task ids in TEXT (the rollout evidence) ───────────────────────────────────────────
# The per-task digests never carried the solver's reasoning; the full traces do, and a solver
# sometimes speculates about sibling task names there ("maybe there are tasks Debugging/01_01,
# ..."). A name is not held-out CONTENT, but no non-train id may reach an
# evolver: the loop's post-run audit (daytona_evolver.heldout_id_hits over the evolver's log) would
# then fail on an evolver that merely quoted a trace. rollout_evidence.py redacts them at build time;
# this is the check, over every file of an evidence dir.
def split_key_ids(sp: dict, key: str) -> list:
    """Ids under one split key: a list of ids, a list of {"id": ...}, or a record with "task_ids"."""
    v = sp.get(key)
    if isinstance(v, dict):
        v = v.get("task_ids")
    if not isinstance(v, list):
        return []
    return [x if isinstance(x, str) else x["id"] for x in v]


def split_sets(split_path) -> tuple:
    """(train ids, non-train ids): non-train = every NON_TRAIN_KEYS id that is not also train."""
    sp = json.loads(Path(split_path).read_text())
    train = set(split_key_ids(sp, "train"))
    non = {i for k in NON_TRAIN_KEYS for i in split_key_ids(sp, k)} - train
    return train, non


class IdScanner:
    """Finds and redacts non-train task ids in text, in BOTH SSB v2 spellings ('Cat/NN_MM' and the
    file-safe 'Cat_NN_MM'). The boundary rule is at least as broad as daytona_evolver.heldout_id_hits
    (canonical form: not after a word char or '/'; here: not after a word char, a superset) and as
    invariants.check_no_heldout_reachable (quoted whole strings). One compiled pattern keyed on the
    category prefixes, then set membership: ~1 s for a 60 MB epoch. Ids without a '/' (tau3's bare
    numbers) cannot be told apart from other text and are not scanned (`unscannable`)."""

    def __init__(self, train, non_train):
        self.non = set(non_train) - set(train)
        prefixes = sorted({i.split("/", 1)[0] for i in self.non if "/" in i}, key=len, reverse=True)
        self.unscannable = sorted(i for i in self.non if "/" not in i)
        self.rx = (re.compile(r"(?<![\w])(" + "|".join(re.escape(p) for p in prefixes)
                              + r")[/_]([A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)(?![\w])") if prefixes else None)

    @classmethod
    def for_split(cls, split_path):
        return cls(*split_sets(split_path))

    @staticmethod
    def _canon(m) -> str:
        return m.group(1) + "/" + m.group(2)

    def find(self, text: str) -> list:
        if not self.rx or not text:
            return []
        return sorted({self._canon(m) for m in self.rx.finditer(text) if self._canon(m) in self.non})

    def redact(self, text: str, marker: str) -> tuple:
        """(text with every non-train id replaced by `marker`, count)."""
        if not self.rx or not text:
            return text, 0
        n = [0]

        def sub(m):
            if self._canon(m) in self.non:
                n[0] += 1
                return marker
            return m.group(0)
        return self.rx.sub(sub, text), n[0]


def text_id_hits(d, scanner, unreadable=None) -> list:
    """[(path relative to d, id)] for every non-train id in any file under d (no symlink followed).
    A file that cannot be read is appended to `unreadable` when a list is given, else the error raises:
    an unscanned file is never silently treated as clean."""
    d = Path(d)
    hits = []
    for root, _dirs, files in os.walk(str(d), followlinks=False):
        for n in sorted(files):
            p = Path(root) / n
            if p.is_symlink() or not p.is_file():
                continue
            try:
                b = p.read_bytes()
            except OSError:
                if unreadable is None:
                    raise
                unreadable.append(p.relative_to(d).as_posix())
                continue
            for i in scanner.find(b.decode("utf-8", errors="replace")):
                hits.append((p.relative_to(d).as_posix(), i))
    return sorted(hits)


def content_problems(d, split_path) -> list:
    """A non-train task id anywhere in any file of the evidence dir (digests, _library, rollouts, ...)."""
    try:
        sc = IdScanner.for_split(split_path)
    except Exception as e:                                      # noqa: BLE001
        return [f"cannot read the split {split_path}: {type(e).__name__}: {e}"]
    unreadable: list = []
    hits = text_id_hits(d, sc, unreadable)
    probs = []
    if unreadable:
        probs.append(f"{len(unreadable)} file(s) in {Path(d).name} cannot be read, so they cannot be shown "
                     f"free of non-train task ids: {unreadable[:3]}")
    if hits:
        files = sorted({p for p, _i in hits})
        probs.append(f"NON-TRAIN TASK IDS in {Path(d).name}: {len(hits)} (file, id) pair(s) in {len(files)} "
                     f"file(s), e.g. {hits[:3]}")
    return probs


def rollouts_problems(d, split_path) -> list:
    """rollouts/ may hold only real dirs named by a TRAIN task stem, each holding only s<k>.trace.jsonl /
    s<k>.grade.txt; solver_context/ only system_prompt.<sha12>.txt / tools.<sha12>.json. Absent = []."""
    d = Path(d)
    probs = []
    try:
        train, non = split_sets(split_path)
    except Exception as e:                                      # noqa: BLE001
        return [f"cannot read the split {split_path}: {type(e).__name__}: {e}"]
    tstems = {i.replace("/", "_") for i in train}
    nstems = {i.replace("/", "_") for i in non}
    ro = d / ROLLOUTS_DIR
    if os.path.lexists(str(ro)):
        if ro.is_symlink() or not ro.is_dir():
            return [f"{d.name}/{ROLLOUTS_DIR} is not a real directory"]
        held, stray, bad = [], [], []
        for c in sorted(ro.iterdir()):
            if c.name in nstems:
                held.append(c.name)
            elif c.name not in tstems or c.is_symlink() or not c.is_dir():
                stray.append(c.name)
            else:
                bad += [f"{c.name}/{f.name}" for f in sorted(c.iterdir())
                        if f.is_symlink() or not f.is_file() or not _ROLLOUT_FILE.match(f.name)]
        if held:
            probs.append(f"HELD-OUT MATERIAL IN {d.name}/{ROLLOUTS_DIR}: {held[:6]}")
        if stray:
            probs.append(f"{d.name}/{ROLLOUTS_DIR} holds entries that are NOT train task dirs: {stray[:6]}")
        if bad:
            probs.append(f"{d.name}/{ROLLOUTS_DIR} holds files other than s<k>.trace.jsonl / s<k>.grade.txt: "
                         f"{bad[:4]}")
    sc = d / CONTEXT_DIR
    if os.path.lexists(str(sc)):
        if sc.is_symlink() or not sc.is_dir():
            probs.append(f"{d.name}/{CONTEXT_DIR} is not a real directory")
        else:
            bad = [f.name for f in sorted(sc.iterdir())
                   if f.is_symlink() or not f.is_file() or not _CONTEXT_FILE.match(f.name)]
            if bad:
                probs.append(f"{d.name}/{CONTEXT_DIR} holds unexpected entries: {bad[:4]}")
    return probs


def n_task_files(d) -> int:
    """Per-task digest files at TOP LEVEL only (a _library/ subtree is not a digest)."""
    return len({f.stem for f in Path(d).glob("*.md")} - SIDECARS)


def validate(run_dir, iteration, current, prior: str, split_path):
    """Validate an evidence list handed to the launcher. Returns (prior_ok, errors).

    `prior` is the os.pathsep-joined list from EVOSKILL_PRIOR_DIGESTS. `prior_ok` is
    [(j, realpath)] sorted by j. Every entry must be ABSOLUTE, must not be a symlink, must resolve
    INSIDE realpath(run_dir), and must be EXACTLY <run_dir>/digests_e<j>_train for some
    1 <= j < iteration (the explicit list the loop passes; epoch numbers are never re-derived from
    free-form names, because hand launches used digests_e0_train for iteration 1). The current
    dir must be <run_dir>/digests_e<iteration>_train. Every dir must pass `surface_problems`.
    A non-empty list must also be COMPLETE: every j in 1..iteration-1, none missing.
    """
    errors = []
    try:
        k = int(iteration)
    except (TypeError, ValueError):
        return [], [f"iteration {iteration!r} is not an integer"]
    split_ok = bool(split_path) and Path(split_path).is_file()
    if not split_ok:
        errors.append(f"no split file at {split_path!r} — the train-only firewall has nothing "
                      f"to check against, and a default split is exactly the one that is wrong")
    rd = os.path.realpath(str(run_dir))
    if not os.path.isdir(rd):
        return [], errors + [f"no run dir at {run_dir}"]
    expected = {}
    for j in range(1, k):
        rp = os.path.realpath(str(canonical_dir(run_dir, j)))
        if rp in expected:
            errors.append(f"digests_e{j}_train and digests_e{expected[rp]}_train resolve to the "
                          f"same directory {rp}")
        expected.setdefault(rp, j)
    ok, seen = [], set()
    entries = prior.split(os.pathsep) if prior else []
    for e in entries:
        if not e:
            errors.append("an EMPTY entry in the prior-digest list (a doubled or trailing "
                          "separator); refusing to guess what was meant")
            continue
        if not os.path.isabs(e):
            errors.append(f"{e!r} is not absolute — it would resolve against the isolated work dir")
            continue
        if os.path.islink(e):
            errors.append(f"{e} is a SYMLINK — the profile builder resolves it and would grant "
                          f"its target")
            continue
        rp = os.path.realpath(e)
        if not _inside(rp, rd):
            errors.append(f"{e} is OUTSIDE this run dir ({rd}); another run's digests carry "
                          f"another arm's skills (#213)")
            continue
        if rp not in expected:
            errors.append(f"{e} is not digests_e<j>_train of this run for any 1 <= j < {k} "
                          f"(a later epoch, the current one, or a stray sibling such as a .bak)")
            continue
        j = expected[rp]
        if os.path.islink(str(canonical_dir(run_dir, j))):
            errors.append(f"{canonical_dir(run_dir, j)} is a SYMLINK")
            continue
        if j in seen:
            errors.append(f"digests_e{j}_train is listed twice")
            continue
        seen.add(j)
        if split_ok:
            errors += [f"digests_e{j}_train: {p}" for p in surface_problems(e, split_path)]
        ok.append((j, rp))
    # THE LIST MUST BE COMPLETE. Every entry above can be valid while one
    # earlier epoch is simply ABSENT, and the launcher would then dispatch with a partial surface —
    # banner, grants and prompt gate all describing it — which only _v_evolver_contained noticed,
    # after the evolver had spent its run. A non-empty list is all-epochs mode, which grants EVERY
    # earlier epoch; the empty list at iteration > 1 is refused by the launcher itself.
    if entries:
        missing = sorted(set(range(1, k)) - seen)
        if missing:
            errors.append("the prior-digest list omits " + ", ".join(
                f"digests_e{j}_train" for j in missing) + " — all-epochs mode grants EVERY "
                f"earlier epoch (1..{k - 1}); refusing a partial evidence surface")
    if current is not None:
        cur_c = os.path.realpath(str(canonical_dir(run_dir, k)))
        if os.path.islink(str(current)):
            errors.append(f"the current digest dir {current} is a SYMLINK")
        elif os.path.realpath(str(current)) != cur_c:
            errors.append(f"the current digest dir {current} is not {canonical_dir(run_dir, k)} "
                          f"— in all-epochs mode every granted surface is this run's own")
        elif split_ok:
            errors += [f"digests_e{k}_train: {p}" for p in surface_problems(current, split_path)]
    return sorted(ok), errors


def dir_sha256(d) -> str:
    """Content hash of a whole evidence dir: every entry's relative path and bytes, in sorted order.
    Symlinks are recorded by their target text and never followed. Stable across machines."""
    d = Path(d)
    h = hashlib.sha256()
    rows = []
    for root, dirs, files in os.walk(str(d), followlinks=False):
        for n in sorted(list(dirs) + list(files)):
            p = os.path.join(root, n)
            rel = os.path.relpath(p, str(d))
            if os.path.islink(p):
                rows.append(f"L\t{rel}\t{os.readlink(p)}")
            elif os.path.isfile(p):
                fh = hashlib.sha256()
                with open(p, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        fh.update(chunk)
                rows.append(f"F\t{rel}\t{fh.hexdigest()}")
    for r in sorted(rows):
        h.update(r.encode("utf-8", "surrogateescape"))
        h.update(b"\n")
    return h.hexdigest()


def hashes(dirs) -> dict:
    return {os.path.realpath(str(p)): dir_sha256(p) for p in dirs}


def _pairs(tokens, flags) -> dict:
    """{flag: [values]} for `--flag value` pairs in an argv-shaped list."""
    out = {f: [] for f in flags}
    i = 0
    while i < len(tokens):
        if tokens[i] in out and i + 1 < len(tokens):
            out[tokens[i]].append(tokens[i + 1])
            i += 2
        else:
            i += 1
    return out


def record(mode, current, prior, add_dirs_argv, sb_argv) -> dict:
    """The evidence half of the containment receipt and the trace: what was granted, what was
    hidden, and the hash of every evidence dir BEFORE dispatch."""
    prior = [os.path.realpath(p) for p in prior if p]
    ev = prior + [os.path.realpath(str(current))]
    sb = _pairs(list(sb_argv), ("--grant", "--write", "--hide", "--no-write"))
    rec = {
        "evidence_mode": mode,
        "prior_digests": prior,
        "evidence_dirs": ev,
        "add_dirs": _pairs(list(add_dirs_argv), ("--add-dir",))["--add-dir"],
        "sandbox_grants": sb["--grant"],
        "sandbox_writes": sb["--write"],
        "sandbox_hides": sb["--hide"],
        "sandbox_no_writes": sb["--no-write"],
    }
    try:
        rec["evidence_sha256_before"] = hashes(ev)
    except Exception as e:                                      # noqa: BLE001
        rec["evidence_sha256_before"] = {}
        rec["evidence_sha256_error"] = f"{type(e).__name__}: {e}"
    return rec


def _before_hash_gap(receipt: str, evidence_dirs) -> str:
    """'' when the receipt on disk records a pre-dispatch hash for every evidence dir; otherwise
    what is missing."""
    try:
        before = json.loads(Path(receipt).read_text()).get("evidence_sha256_before")
    except Exception as e:                                      # noqa: BLE001
        return f"the receipt cannot be read back ({type(e).__name__}: {e})"
    if not isinstance(before, dict) or not before:
        return "the before-hash is empty"
    missing = [d for d in evidence_dirs if not before.get(d)]
    return f"no before-hash for {missing}" if missing else ""


def _annotate(receipt: str, trace: str, fields: dict) -> None:
    for path in (receipt, trace):
        if not path:
            continue
        p = Path(path)
        try:
            d = json.loads(p.read_text()) if p.is_file() else {}
        except Exception:                                       # noqa: BLE001
            d = {"_unreadable_before_annotation": True}
        d.update(fields)
        tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(d, indent=1))
        os.replace(str(tmp), str(p))


def main(argv=None) -> int:
    """CLI for the shell launcher.

      --validate  --run-dir R --iter K --current D --split S [--prior P]
            prints each validated prior dir's realpath, one per line (sorted by epoch);
            errors to stderr, exit 21.
      --hash D [D ...]
            prints {realpath: sha256} as JSON.
      --annotate RECEIPT TRACE MODE CURRENT PRIOR -- <GRANTS argv> -- <sandbox argv>
            merges `record()` into the receipt and the trace JSON.
    """
    a = list(sys.argv[1:] if argv is None else argv)
    if a[:1] == ["--hash"]:
        print(json.dumps(hashes(a[1:]), indent=1))
        return 0
    if a[:1] == ["--annotate"]:
        receipt, trace, mode, current, prior = a[1:6]
        rest = a[6:]
        if rest[:1] == ["--"]:
            rest = rest[1:]
        cut = rest.index("--") if "--" in rest else len(rest)
        grants, sb = rest[:cut], rest[cut + 1:]
        fields = record(mode, current, prior.split(os.pathsep) if prior else [], grants, sb)
        _annotate(receipt, trace, fields)
        # The launcher refuses an all-epochs dispatch on a non-zero exit, and the
        # loop fails such a receipt afterwards. So "recorded" means the RECEIPT ON DISK carries a
        # before-hash for every evidence dir — not merely that hashing raised nothing.
        gap = _before_hash_gap(receipt, fields["evidence_dirs"])
        if "evidence_sha256_error" in fields or gap:
            print(f"  evidence rec: NO usable pre-dispatch hash in {Path(receipt).name}: "
                  f"{fields.get('evidence_sha256_error') or gap}", file=sys.stderr)
            return 3
        print(f"  evidence rec: {len(fields['evidence_dirs'])} dir(s) hashed into "
              f"{Path(receipt).name}" + (f" and {Path(trace).name}" if trace else ""))
        return 0
    if a[:1] == ["--validate"]:
        import argparse
        ap = argparse.ArgumentParser(prog="evidence_surface --validate")
        ap.add_argument("--validate", action="store_true")
        ap.add_argument("--run-dir", required=True)
        ap.add_argument("--iter", required=True)
        ap.add_argument("--current", required=True)
        ap.add_argument("--split", required=True)
        ap.add_argument("--prior", default="")
        o = ap.parse_args(a)
        ok, errs = validate(o.run_dir, o.iter, o.current, o.prior, o.split)
        if errs:
            print("EVIDENCE LIST REFUSED — not dispatching:", file=sys.stderr)
            for e in errs:
                print(f"  * {e}", file=sys.stderr)
            return REFUSE_RC
        for _j, p in ok:
            print(p)
        return 0
    print(main.__doc__, file=sys.stderr)
    return 64


if __name__ == "__main__":
    raise SystemExit(main())
