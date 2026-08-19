#!/usr/bin/env python3
"""The SSB v2 AO-run epoch gate — ONE reader for every epoch, all pre-registered statistics.

    python3 scripts/ssbv2_gate_epoch.py --run-dir outputs/my_run --version v2
    python3 scripts/ssbv2_gate_epoch.py --run-dir outputs/my_run --version v3 \
        --out outputs/my_run/E3_GATE_RESULT.json

WHY A SCRIPT AND NOT A SNIPPET PER EPOCH. This is the same argument
scripts/tau3_gate_epoch.py makes, and it is transplanted here deliberately: on
the tau3 25/25 run, three of the errors were READING errors rather than data
errors (a pass rate read off a rollout count, a per_task block assumed to be
keyed by task_id, a "0 cached timeouts" false zero from guessing a field shape).
One reader used at every epoch is the fix. Epochs 1 and 2 of THIS run were read
by inline snippets; from here they are read by this file.

THE GATE IS NOT CHOSEN HERE (#121). It is read out of the run's own SPEC.json
(`spec.gate`), which was written before any leg ran, and this script REFUSES
rather than defaults if that block is missing or names a test it does not
implement. As of the ssbv2_ao1 run that block reads:

    pairing            episode_level
    tests              episode_paired_sign, cluster_pass_count_sign
    alpha              0.05
    n_comparisons      4        -> alpha_corrected = 0.0125 (Bonferroni over epochs)
    require_all_tests  true     -> BOTH must clear, not either
    discordant_floor   6        -> below this, no p can be believed (n<6 cannot reach 0.05)
    report_per_basis   true     -> #156, never ship the pooled number alone

  * primary  : EPISODE-LEVEL paired sign test on the VAL arm. Pairing is
               (task, sample-slot) -> (task, sample-slot), 180 units.
  * companion: the CLUSTER-RESPECTING pass-count sign test on the VAL arm.
               Episode-level pairing treats a task's 3 samples as independent
               when they are not, which is ANTI-CONSERVATIVE; this companion is
               what catches that. Both must clear alpha_corrected.
  * TRAIN IS NOT THE GATE. The skills were derived from train; it is reported
    because a train-vs-val divergence is diagnostic, not because it decides.

Everything else printed is DESCRIPTIVE and pre-declared as such: the #209 triple
(pass / graded-fail / never-graded), the cap decomposition, and the per-category
breakdown that #156 requires — an aggregate can average two opposite-sign
effects into a null, or (as at e2 here) hide that the whole pooled delta sits in
one category that started near the floor.

SCHEMA NOTE. The SSB v2 runner writes `per_task` keyed by the STRING INDEX of
the task within `task_ids`, not by task id, with a list of n_samples float
scores. That is the #306 dual-schema trap in miniature; this reader asserts the
shape rather than tolerating it.
"""
from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

IMPLEMENTED_TESTS = {"episode_paired_sign", "cluster_pass_count_sign"}
CATEGORY_OF = lambda tid: tid.split("/")[0]  # noqa: E731 — "Debugging/02_02" -> "Debugging"


def sign_test_two_sided(up: int, down: int) -> float:
    """Exact two-sided binomial on DISCORDANT pairs only.

    Concordant pairs carry no information about direction and are excluded — the
    mistake this project made twice (0804d K9, 0807 K8). n<6 discordant cannot
    reach p<0.05 no matter how lopsided, which is why discordant_floor exists.
    """
    n = up + down
    if n == 0:
        return 1.0
    k = min(up, down)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load_leg(path: Path) -> dict:
    d = json.loads(path.read_text())
    for key in ("per_task", "task_ids", "sample_pass", "sample_total", "n", "n_samples"):
        if key not in d:
            raise SystemExit(f"REFUSING: {path} has no '{key}' — schema is not what this reader expects")
    n, ns = d["n"], d["n_samples"]
    if len(d["task_ids"]) != n:
        raise SystemExit(f"REFUSING: {path} task_ids={len(d['task_ids'])} but n={n}")
    if len(d["per_task"]) != n:
        raise SystemExit(f"REFUSING: {path} per_task has {len(d['per_task'])} entries but n={n}")
    if n * ns != d["sample_total"]:
        raise SystemExit(f"REFUSING: {path} grid {n}x{ns} != sample_total {d['sample_total']}")
    for i in range(n):
        v = d["per_task"].get(str(i))
        if v is None or len(v) != ns:
            raise SystemExit(f"REFUSING: {path} per_task['{i}'] is not a list of {ns} scores")
    return d


def vectors(d: dict) -> tuple[list[str], list[list[float]]]:
    return d["task_ids"], [d["per_task"][str(i)] for i in range(d["n"])]


def compare(a: dict, b: dict, label: str, gate: dict, is_gate: bool) -> dict:
    ta, va = vectors(a)
    tb, vb = vectors(b)
    if ta != tb:
        raise SystemExit(f"REFUSING: {label} — the two legs did not run the same task set")

    # SAME TASKS IS NOT SAME COMPARISON -- THE SAMPLE COUNT HAS TO MATCH TOO.
    # load_leg validates each leg against its OWN n_samples and never against the
    # other's, and both loops below use zip(), which truncates in silence. With a
    # base leg at N=3 and a skill leg at N=1 the episode-paired test then compares
    # slot 0 only (10 pairs out of a declared 30-episode leg) while the cluster test
    # sums 3 samples against 1 -- so the two PRE-REGISTERED tests are computed over
    # different pairings of the same pair of legs, and the reader still emits a
    # verdict. Measured: episode test cleared at p=0.0020 while the cluster test went
    # the other way, from the same two files.
    na, nb = a.get("n_samples"), b.get("n_samples")
    if na != nb:
        raise SystemExit(
            f"REFUSING: {label} — the legs ran different sample counts "
            f"(base n_samples={na}, skill n_samples={nb}). The paired tests would be "
            f"computed over different pairings of the same legs.")
    for i, (ra, rb) in enumerate(zip(va, vb)):
        if len(ra) != len(rb):
            raise SystemExit(
                f"REFUSING: {label} — task {ta[i]!r} has {len(ra)} base samples and "
                f"{len(rb)} skill samples; zip() would drop the excess in silence.")

    ep_up = ep_down = 0
    ep_pairs = 0
    for ra, rb in zip(va, vb):
        for x, y in zip(ra, rb):
            ep_pairs += 1
            if y > x:
                ep_up += 1
            elif y < x:
                ep_down += 1
    cl_up = cl_down = 0
    for ra, rb in zip(va, vb):
        ca, cb = sum(ra), sum(rb)
        if cb > ca:
            cl_up += 1
        elif cb < ca:
            cl_down += 1

    alpha = gate["alpha_corrected"]
    floor = gate["discordant_floor"]
    res = {
        "label": label,
        "is_gate": is_gate,
        "alpha_corrected": alpha,
        # `paired` is REPORTED, not just computed: it is the denominator each test
        # actually ran over. When zip() could still truncate (before the n_samples
        # refusal above) the two tests silently used different pairings of the same
        # two legs, and nothing in the output said so.
        "episode_paired_sign": {
            "up": ep_up, "down": ep_down, "discordant": ep_up + ep_down,
            "paired": ep_pairs,
            "p": sign_test_two_sided(ep_up, ep_down),
        },
        "cluster_pass_count_sign": {
            "up": cl_up, "down": cl_down, "discordant": cl_up + cl_down,
            "paired": len(va),
            "p": sign_test_two_sided(cl_up, cl_down),
        },
    }
    for name in ("episode_paired_sign", "cluster_pass_count_sign"):
        s = res[name]
        s["clears"] = bool(s["p"] < alpha and s["discordant"] >= floor and s["up"] > s["down"])
        s["floor_met"] = bool(s["discordant"] >= floor)

    tests = gate["tests"]
    verdict = (all(res[t]["clears"] for t in tests) if gate["require_all_tests"]
               else any(res[t]["clears"] for t in tests))
    res["verdict"] = "PASS" if verdict else "FAIL"

    cats: dict[str, list[int]] = {}
    for tid, ra, rb in zip(ta, va, vb):
        c = cats.setdefault(CATEGORY_OF(tid), [0, 0, 0, 0, 0])
        c[0] += int(sum(ra)); c[1] += int(sum(rb)); c[2] += 1
        c[3] += 1 if sum(ra) > 0 else 0
        c[4] += 1 if sum(rb) > 0 else 0
    res["per_category"] = {
        c: {"episodes_from": v[0], "episodes_to": v[1], "episodes_of": v[2] * a["n_samples"],
            "pass_at_n_from": v[3], "pass_at_n_to": v[4], "tasks": v[2]}
        for c, v in sorted(cats.items())
    }
    return res


def triple(d: dict) -> dict:
    """#209: pass / graded-fail / never-graded, and they must sum to the grid.

    SSB v2 has NO never-graded bucket by construction — a turn-capped episode is
    still graded on whatever file is at the output path — so this is 0 and the
    capped count is reported SEPARATELY. Reporting a raw pass delta without this
    triple is how #209 over-claimed on tau3.
    """
    total = d["sample_total"]
    passed = int(d["sample_pass"])
    return {"pass": passed, "graded_fail": total - passed, "never_graded": 0, "total": total,
            "capped_but_graded": d.get("turn_exhausted"),
            "pass_at_n": d.get("pass3"), "all_n": d.get("all3"),
            "skills_dir": d.get("skills_dir"), "solver_harness": d.get("solver_harness")}


def fmt_block(r: dict) -> str:
    out = [f"\n=== {r['label']} ===   [{'GATE' if r['is_gate'] else 'descriptive only'}]"]
    for name in ("episode_paired_sign", "cluster_pass_count_sign"):
        s = r[name]
        mark = "CLEARS" if s["clears"] else "does not clear"
        note = "" if s["floor_met"] else f"  (BELOW discordant floor — p is not interpretable)"
        # ...AND `paired` GOES IN THE PRINTED LINE, not just the JSON. The comment at the
        # results dict says it "is REPORTED", and it was computed and stored and then never
        # rendered -- so the human-readable output, which is what an operator reads, still
        # said nothing about the denominator each test ran over.
        out.append(f"  {name:24s} +{s['up']:3d} / -{s['down']:3d}  discordant={s['discordant']:3d}"
                   f"  paired={s['paired']:3d}"
                   f"  p={s['p']:.4f}  vs alpha={r['alpha_corrected']:.4f}  -> {mark}{note}")
    out.append(f"  VERDICT: {r['verdict']}")
    for c, v in r["per_category"].items():
        out.append(f"    {c:16s} episodes {v['episodes_from']:3d} -> {v['episodes_to']:3d}"
                   f" /{v['episodes_of']:3d}    pass@n {v['pass_at_n_from']:2d} -> {v['pass_at_n_to']:2d}"
                   f" /{v['tasks']}")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--version", required=True, help="e.g. v2, v3 — the CURRENT skill version")
    ap.add_argument("--base", default="e0_base")
    ap.add_argument("--prev", default=None, help="optional previous version for a v(N-1)->v(N) readout")
    ap.add_argument("--out", default=None)
    # THE SEALED TEST IS NOT AN EPOCH. Epoch legs are written as
    # {run}_{version}_{train|val}.json; the test leg is {run}_{version}.json with no
    # split suffix, because there is one split and it is run once. Without this flag the
    # reader dies on a missing _train file — which is the correct failure (it refuses
    # rather than guessing a path), but it needs the mode to exist.
    ap.add_argument("--single-split", default=None, metavar="NAME",
                    help="Leg files carry NO _train/_val suffix (the sealed test). "
                         "Runs exactly ONE comparison: base -> version.")
    a = ap.parse_args(argv)

    rd = Path(a.run_dir)
    # A MISSING SPEC IS A REFUSAL, NOT A TRACEBACK. This read was unguarded, so pointing the
    # gate at a run dir without a frozen spec -- the ordinary mistake -- printed a
    # FileNotFoundError stack through pathlib internals. Operator-facing scripts in this repo
    # say what is wrong and what to do; this one just crashed.
    _spec_path = rd / "SPEC.json"
    if not _spec_path.is_file():
        raise SystemExit(
            f"REFUSING: no SPEC.json in {rd}. The gate reads the PRE-REGISTERED decision rule "
            f"from the frozen spec; without it there is nothing to gate against. Point "
            f"--run-dir at a run created by `evoskill new`, or freeze a spec there first.")
    try:
        spec = json.loads(_spec_path.read_text())["spec"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise SystemExit(f"REFUSING: {_spec_path} is not a readable spec ({type(e).__name__}: {e})")
    g = spec.get("gate")
    if not g:
        raise SystemExit("REFUSING: SPEC.json carries no spec.gate — the decision rule is not pre-registered")
    unknown = set(g["tests"]) - IMPLEMENTED_TESTS
    if unknown:
        raise SystemExit(f"REFUSING: SPEC.json names test(s) this reader does not implement: {sorted(unknown)}")
    gate = {
        "tests": list(g["tests"]),
        "alpha": g["alpha"],
        "n_comparisons": g["n_comparisons"],
        "alpha_corrected": g["alpha"] / max(1, g["n_comparisons"]),
        "require_all_tests": g["require_all_tests"],
        "discordant_floor": g["discordant_floor"],
        "pairing": g["pairing"],
    }
    if gate["pairing"] != "episode_level":
        raise SystemExit(f"REFUSING: pairing '{gate['pairing']}' is not what this reader implements")

    def leg(v, s):
        return load_leg(rd / f"{rd.name}_{v}_{s}.json")

    if a.single_split:
        S = a.single_split
        legs = {(v, S): load_leg(rd / f"{rd.name}_{v}.json")
                for v in filter(None, (a.base, a.prev, a.version))}
    else:
        legs = {(v, s): leg(v, s) for v in filter(None, (a.base, a.prev, a.version)) for s in ("train", "val")}

    print(f"GATE as declared in {rd/'SPEC.json'}: tests={gate['tests']} alpha={gate['alpha']}"
          f" /{gate['n_comparisons']} = {gate['alpha_corrected']:.4f}"
          f"  require_all={gate['require_all_tests']}  discordant_floor={gate['discordant_floor']}")
    print("\n--- #209 triple per leg (pass / graded-fail / never-graded = total) ---")
    for (v, s), d in legs.items():
        t = triple(d)
        # ABSENT IS NOT ZERO, and it must not be a traceback either. These come from
        # `.get()` on the leg JSON, so a leg written by a runner that does not record a
        # field arrives as None and the int format raised
        # `TypeError: unsupported format string passed to NoneType.__format__`,
        # killing the gate mid-report. Rendering None as 0 would be worse: this block
        # exists because a missing count must never read as a measured zero (#209).
        def _n(x, w=3):
            return f"{x:{w}d}" if isinstance(x, int) else "?".rjust(w)
        print(f"  {v:8s} {s:5s}  {_n(t['pass'])} / {_n(t['graded_fail'])} / {_n(t['never_graded'])}"
              f" = {_n(t['total'])}   pass@3={_n(t['pass_at_n'], 2)}  all3={_n(t['all_n'], 2)}"
              f"  capped-but-graded={_n(t['capped_but_graded'])}"
              f"  skills={str(t['skills_dir']).split('/')[-1]}")

    results = []
    if a.single_split:
        S = a.single_split
        results.append(compare(legs[(a.base, S)], legs[(a.version, S)],
                               f"{S.upper()}  {a.base} -> {a.version}", gate, is_gate=True))
        # THE SEALED TEST IS NOT ONE OF THE PRE-REGISTERED COMPARISONS. SPEC.json declared
        # n_comparisons=4 for the EPOCH val gates; this is a separate, single, confirmatory
        # comparison run once after the run was sealed. Reporting it against the corrected
        # alpha alone would silently import a correction it was never part of, and
        # reporting it against 0.05 alone would drop the run's own standard. So both are
        # printed, labelled, and the reader does NOT choose between them.
        r = results[0]
        print("\n*** SEALED TEST — READ THIS BEFORE THE VERDICT LINE ***")
        print("  This comparison was NOT among the 4 pre-registered epoch gates. The verdict")
        print(f"  below applies the run's corrected alpha ({gate['alpha_corrected']:.4f}).")
        print("  Against an uncorrected alpha=0.05 for a single confirmatory test:")
        for name in ("episode_paired_sign", "cluster_pass_count_sign"):
            s = r[name]
            at05 = bool(s["p"] < 0.05 and s["discordant"] >= gate["discordant_floor"]
                        and s["up"] > s["down"])
            print(f"    {name:24s} p={s['p']:.4f}  ->  {'clears' if at05 else 'does not clear'} at 0.05")
    else:
        results.append(compare(legs[(a.base, "val")], legs[(a.version, "val")],
                               f"VAL   {a.base} -> {a.version}", gate, is_gate=True))
        results.append(compare(legs[(a.base, "train")], legs[(a.version, "train")],
                               f"TRAIN {a.base} -> {a.version}", gate, is_gate=False))
        if a.prev:
            results.append(compare(legs[(a.prev, "val")], legs[(a.version, "val")],
                                   f"VAL   {a.prev} -> {a.version}", gate, is_gate=False))
    for r in results:
        print(fmt_block(r))

    gate_r = results[0]
    print(f"\nPRE-REGISTERED VERDICT ({gate_r['label']}): {gate_r['verdict']}")
    if gate_r["verdict"] == "FAIL":
        print("  A raw pass-count delta is NOT the result. Report the verdict, the discordant")
        print("  counts and the per-category split together, per #209/#156.")

    if a.out:
        payload = {"gate": gate, "version": a.version, "base": a.base,
                   "triples": {f"{v}_{s}": triple(d) for (v, s), d in legs.items()},
                   "comparisons": results}
        Path(a.out).write_text(json.dumps(payload, indent=1))
        print(f"\nwritten {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
