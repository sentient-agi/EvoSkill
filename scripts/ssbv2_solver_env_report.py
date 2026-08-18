#!/usr/bin/env python3
"""Report the SSB v2 solver's effective environment AND its cache identity together.

WHY THIS EXISTS. `RunCache._get_tree_hash` mixes exactly two ingredients — the
live `.claude/skills/` content and the solver's system prompt. The **docker image
is not one of them**. So changing what the solver can DO (adding a formula engine)
leaves every cache key byte-identical, and a run seeded from older records
silently mixes two environments while reporting one. There is no downstream check
that would notice: the traces replay, the scores aggregate, nothing errors.

That is the same silent-drift shape as the two-build-sites provider pin, where a
pin landed on a non-executing options dict and 100% of traffic kept going
somewhere else with no symptom.

This script puts the two facts on one screen: what the image can do, and what the
cache thinks the run is. Use it to satisfy "verify the tree_hash actually
rotated" with a number rather than an inference, and to check a cache directory
for records made under a DIFFERENT solver environment before reusing them.

    python3 scripts/ssbv2_solver_env_report.py
    python3 scripts/ssbv2_solver_env_report.py --image ssb-solver-lo:latest
    python3 scripts/ssbv2_solver_env_report.py --expect-not bfb5bd90d1e0
    python3 scripts/ssbv2_solver_env_report.py --cache .cache/ssbv2_evo0801_cap45_cache

Exit codes: 0 ok · 3 an --expect assertion failed · 4 the image is unusable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def probe_image(image: str) -> dict:
    """What can the solver actually do inside this image?

    Runs one container and asks it, rather than reading a Dockerfile — a
    Dockerfile records intent, and the whole failure being guarded here is a gap
    between intent and the running container.
    """
    out: dict = {"image": image}
    idq = subprocess.run(["docker", "images", "--format", "{{.ID}}", image],
                         capture_output=True, text=True)
    out["id"] = (idq.stdout or "").strip().splitlines()[0] if idq.stdout.strip() else None
    if not out["id"]:
        out["error"] = "image not present on this daemon"
        return out

    script = r"""
import importlib.util as u, json, shutil, subprocess, sys
d = {
  "soffice": shutil.which("soffice"),
  "libreoffice": shutil.which("libreoffice"),
  "python3": shutil.which("python3"),
  "py": "%d.%d" % sys.version_info[:2],
}
for m in ("openpyxl", "pandas", "numpy"):
    try:
        d[m] = __import__(m).__version__
    except Exception as e:
        d[m] = f"MISSING ({e.__class__.__name__})"
d["pure_python_engines"] = [m for m in ("formulas", "pycel", "xlcalculator", "schedula")
                            if u.find_spec(m)]
try:
    subprocess.run(["/usr/bin/python3", "-c", "import uno"], check=True,
                   capture_output=True, timeout=60)
    d["uno"] = True
except Exception:
    d["uno"] = False
print(json.dumps(d))
"""
    r = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "python3", image, "-c", script],
        capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        out["error"] = f"probe failed rc={r.returncode}: {r.stderr[-400:]}"
        return out
    out.update(json.loads(r.stdout))
    return out


def cache_identity(live_skills: Path) -> dict:
    """tree_hash exactly as the runner will compute it.

    Calls the REAL RunCache with the REAL solver prompt. Re-deriving the hash
    from its documented recipe would be computing from a proxy — the error shape
    behind every retraction in this project's ledger.
    """
    from src.cache.run_cache import CacheConfig, RunCache
    from src.agent_profiles.ssbv2_agent import get_solver_prompt

    prompt = get_solver_prompt()
    cache = RunCache(CacheConfig(cache_dir=ROOT / ".cache/_treehash_probe",
                                 cwd=ROOT, live_skills_dir=live_skills))
    return {
        "live_skills_dir": str(live_skills),
        "skills_files": sorted(p.name for p in live_skills.glob("**/*") if p.is_file())
                        if live_skills.exists() else [],
        "prompt_chars": len(prompt),
        "prompt_sha256_12": hashlib.sha256(prompt.encode()).hexdigest()[:12],
        "tree_hash": cache._get_tree_hash(prompt),
    }


def scan_cache(cache_dir: Path) -> dict:
    """Bucket a cache dir by tree_hash, and count records that MENTION LibreOffice.

    A record's tree_hash tells you which prompt+skills produced it. It cannot
    tell you which image — that is the gap. The LibreOffice mention count is the
    available proxy: a solver that probed for soffice and found nothing usually
    says so in its transcript.
    """
    res: dict = {"cache_dir": str(cache_dir), "exists": cache_dir.exists()}
    if not cache_dir.exists():
        return res
    files = list(cache_dir.glob("*/*.json"))
    buckets: Counter = Counter()
    absent = present = mentions = 0
    for f in files:
        try:
            rec = json.loads(f.read_text())
        except Exception:
            continue
        buckets[(rec.get("cache_key") or {}).get("tree_hash", "?")] += 1
        blob = json.dumps(rec).lower()
        if "soffice" in blob or "libreoffice" in blob:
            mentions += 1
            # These two phrasings are what the solver actually printed in the
            # traces the maintainer caught. Treated as WEAK evidence and reported as
            # such — never as a substitute for knowing the image.
            if "no formula engine" in blob or "not found" in blob:
                absent += 1
            elif "/usr/bin/soffice" in blob:
                present += 1
    res.update(records=len(files), tree_hash_buckets=dict(buckets),
               mentioning_libreoffice=mentions,
               weak_evidence_absent=absent, weak_evidence_present=present)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="ssb-solver",
                    help="solver image to probe (default: the runner's default)")
    ap.add_argument("--live-skills", default=str(ROOT / ".claude/skills"))
    ap.add_argument("--expect-not", action="append", default=[],
                    metavar="TREEHASH", help="fail if the tree_hash equals this")
    ap.add_argument("--expect", default=None, metavar="TREEHASH",
                    help="fail unless the tree_hash equals this")
    ap.add_argument("--cache", default=None, help="also bucket this cache dir")
    a = ap.parse_args()

    rc = 0
    img = probe_image(a.image)
    print("=== solver image ===")
    for k in ("image", "id", "error"):
        if img.get(k) is not None:
            print(f"  {k:<22} {img[k]}")
    if "error" in img:
        rc = 4
    else:
        for k in ("soffice", "libreoffice", "uno", "python3", "py",
                  "openpyxl", "pandas", "numpy", "pure_python_engines"):
            print(f"  {k:<22} {img.get(k)}")
        verdict = ("HAS a formula engine (can self-verify computed values)"
                   if img.get("soffice") else
                   "NO formula engine — openpyxl returns None for every "
                   "uncached formula, so the solver CANNOT evaluate what it is graded on")
        print(f"  {'verdict':<22} {verdict}")

    ident = cache_identity(Path(a.live_skills))
    print("\n=== cache identity (skills + solver prompt ONLY — image excluded) ===")
    for k, v in ident.items():
        print(f"  {k:<22} {v}")
    print("  NOTE                   the image above is NOT in this hash. Two runs "
          "with different\n"
          "                         solver images and the same prompt share a "
          "cache bucket.")

    # COMPARE ON THE PREFIX, not the full digest. The cache stores and buckets on
    # `tree_hash[:hash_length]` (run_cache.py:694 and _get_cache_dir_for_tree), so
    # a cache dir names 12 characters while _get_tree_hash returns 64. The first
    # version of this gate compared the full digest against a 12-char argument,
    # which can never be equal — an assertion that always passes, on the one fact
    # the run's validity depends on. Same shape as every retraction in this
    # project's ledger: checking a proxy instead of the thing.
    def _same(a_: str, b_: str) -> bool:
        n = min(len(a_), len(b_))
        return n >= 8 and a_[:n] == b_[:n]

    for bad in a.expect_not:
        if _same(ident["tree_hash"], bad):
            print(f"\nFAILED: tree_hash still starts with {bad} — it did NOT "
                  f"rotate. Records made under the previous solver environment "
                  f"would keep hitting.", file=sys.stderr)
            rc = 3
        else:
            print(f"  rotated OK             {ident['tree_hash'][:12]} != {bad}")
    if a.expect and not _same(ident["tree_hash"], a.expect):
        print(f"\nFAILED: tree_hash is {ident['tree_hash'][:12]}, expected {a.expect}",
              file=sys.stderr)
        rc = 3

    if a.cache:
        sc = scan_cache(Path(a.cache))
        print("\n=== cache dir scan ===")
        for k, v in sc.items():
            print(f"  {k:<22} {v}")
        if sc.get("records") and ident["tree_hash"] not in (sc.get("tree_hash_buckets") or {}):
            print("  -> this run's tree_hash is absent from the dir: nothing here "
                  "will be reused (safe).")
        elif sc.get("records"):
            print("  -> this run's tree_hash IS present here. Reuse only if the "
                  "solver IMAGE also matches;\n"
                  "     the hash cannot tell you that.")

    shutil.rmtree(ROOT / ".cache/_treehash_probe", ignore_errors=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
