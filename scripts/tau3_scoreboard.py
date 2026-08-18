#!/usr/bin/env python3
"""Compare tau3 banking_knowledge legs — the analysis tool tau3 was missing.

    python3 scripts/tau3_scoreboard.py <base.json> <skill.json> [<skill2.json> ...]

Every guard in here was earned by a specific mistake made on SSB v2 or tau3 on
2026-08-03/04. They are not defensive boilerplate; each one is a thing that
already produced a wrong number once.

 1. REFUSES a headline while any leg is incomplete. A partial leg reads HIGH:
    rollouts do not finish in random order, the slow ones land last, and slow
    correlates with hard. On SSB v2 a leg at 18/30 read 34.4% against a true
    25.6% and would have won a one-shot selection on biased data.

 2. REPORTS EVERY comparison, never the best one. Reporting whichever leg
    happens to look good is selection; the pre-commitment is all of them.

 3. STATES POWER at the observed split, and says plainly when a null is
    uninformative. The paired sign test discards concordant pairs, so n_eff =
    W+L sets the power — not the task count. Two SSB v2 nulls ran at 2% power,
    where "no significant difference" carries almost no information.

 4. SPLITS FAILURES INTO INFRA vs REAL. This is the tau3-specific one and the
    reason this file exists. Measured 2026-08-04: at the 460s default wall,
    3 of 7 episodes died AT the wall (462/463/466s) and every one scored 0.0.
    A wall-kill is not a wrong answer, and a scoreboard that pools them
    reports the clock as if it were capability. `termination_reason` is the
    discriminator; the census is printed unconditionally.

 5. NEVER prints a bare $0.00. `episodes_without_cost` is reported alongside
    cost, because an unpriced model yields NaN/0 and a trustworthy-looking
    zero is worse than a blank.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def sign_p(w: int, l: int) -> float:
    """Exact two-sided binomial sign test at p=0.5. Ties already dropped."""
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def sign_power(n: int, p_win: float, alpha: float = 0.05) -> float:
    """P(reject | true per-pair win prob p_win), exact, two-sided.

    Sums BOTH rejection tails, so it is symmetric in p_win by construction:
    verified `sign_power(13, 10/13) == sign_power(13, 3/13) == 0.394`, matching
    the textbook 39.4%. The caller still passes max(p, 1-p) — belt and braces,
    and it documents intent at the call site.

    Written this way deliberately. `val_selection.sign_power` sums only the
    upper tail, so feeding it the LOSING side's probability returns ~0: on
    2026-08-03 that reported 0.0% power at n=13, p_win=0.231 where the true
    value is 39.4%. Symmetric here means the bug cannot recur through a
    forgotten max().
    """
    if n <= 0:
        return 0.0
    crit = None
    for k in range(n + 1):
        if sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n) > alpha / 2:
            crit = k - 1
            break
    if crit is None or crit < 0:
        return 0.0
    pw = 0.0
    for i in range(n + 1):
        if i <= crit or i >= n - crit:
            pw += math.comb(n, i) * (p_win ** i) * ((1 - p_win) ** (n - i))
    return min(1.0, pw)


def load(p: Path) -> dict:
    d = json.loads(p.read_text())
    d["_path"] = str(p)
    d["_name"] = d.get("tag") or p.stem
    return d


def complete(d: dict) -> tuple[bool, str]:
    exp = int(d.get("n_tasks", 0)) * int(d.get("n_samples", 0))
    got = int(d.get("episodes", 0))
    if exp and got < exp:
        return False, f"{got}/{exp} episodes"
    per = d.get("per_task") or {}
    missing = [t for t, v in per.items()
               if any(s is None for s in (v if isinstance(v, list) else [v]))]
    if missing:
        return False, f"{len(missing)} task(s) hold a None sample"
    return True, f"{got} episodes"


def mvb(d: dict) -> float:
    per = d.get("per_task") or {}
    tot = sum(len(v) for v in per.values())
    if not tot:
        return float("nan")
    return sum(1 for v in per.values() for s in v if s is not None and s >= 0.999) / tot


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    legs = [load(Path(a)) for a in sys.argv[1:]]
    base, skills = legs[0], legs[1:]

    # Comparability is a precondition, not a footnote: a differing wall or
    # variant means the legs are not measuring the same thing (the
    # episode-timeout parity defect found 2026-08-04 made exactly this
    # possible), so refuse rather than quietly compare.
    for f in ("retrieval_variant", "episode_timeout_s", "solver_model",
              "split_key", "n_samples", "judge_model"):
        vals = {leg["_name"]: leg.get(f) for leg in legs}
        if len(set(map(str, vals.values()))) > 1:
            print(f"  REFUSING: legs disagree on {f!r} -> {vals}")
            print("  These legs are not comparable. Re-run with identical config.")
            return 3

    print(f"\ntau3 {base.get('domain')} | variant={base.get('retrieval_variant')} "
          f"| split={base.get('split_key')} | wall={base.get('episode_timeout_s')}s "
          f"| N={base.get('n_samples')}")
    print(f"\n  {'leg':22s} {'state':22s} {'MV-B':>8s} {'ALL@N':>7s} {'cost':>10s} "
          f"{'no-cost eps':>12s}")
    incomplete = []
    for leg in legs:
        ok, why = complete(leg)
        if not ok:
            incomplete.append((leg["_name"], why))
        c = leg.get("cost_usd")
        cost = "n/a" if c is None or (isinstance(c, float) and math.isnan(c)) \
            else f"${float(c):.4f}"
        print(f"  {leg['_name']:22s} {('COMPLETE' if ok else 'PARTIAL — ' + why):22s} "
              f"{mvb(leg):7.1%} {leg.get('all_n', '?'):>7} {cost:>10s} "
              f"{leg.get('episodes_without_cost', '?'):>12}")
    if any(leg.get("episodes_without_cost") for leg in legs):
        print("  NOTE: some episodes reported NO cost — the total is a FLOOR, not the "
              "spend. Check for an unpriced model in cost_tracking._PRICING_USD_PER_TOKEN.")

    if incomplete:
        print("\n  REFUSING to report a headline: " +
              ", ".join(f"{n} ({w})" for n, w in incomplete))
        print("  A partial leg reads HIGH — the slow (hard) episodes land last.")
        return 1

    # ---- termination census: infra vs real failure -------------------------
    print("\n  TERMINATION CENSUS (a wall-kill is NOT a wrong answer):")
    any_term = False
    for leg in legs:
        term = leg.get("termination_reasons") or {}
        if not term:
            continue
        any_term = True
        n = sum(term.values()) or 1
        killed = int(term.get("timeout", 0))
        print(f"    {leg['_name']:22s} " +
              "  ".join(f"{k}={v}" for k, v in sorted(term.items())) +
              f"   -> {killed}/{n} = {killed / n:.0%} WALL-KILLED")
        if killed:
            print(f"      {killed} episode(s) hit the {leg.get('episode_timeout_s')}s "
                  f"wall. Those are infra outcomes; excluding them changes the "
                  f"denominator, so state which convention you used.")
    if not any_term:
        print("    (absent from these result JSONs — run_tau3_banking.py must record "
              "`termination_reasons` for this census to work. Until it does, the "
              "infra/real split is UNKNOWN, not zero.)")

    # ---- paired sign tests, all of them -----------------------------------
    print(f"\n  PAIRED SIGN TESTS vs {base['_name']} (all reported):")
    print(f"  {'comparison':28s} {'W':>4s} {'L':>4s} {'T':>4s} {'n_eff':>6s} "
          f"{'p':>8s} {'power':>7s}  favours")
    bp = base.get("per_task") or {}
    for leg in skills:
        sp = leg.get("per_task") or {}
        shared = sorted(set(bp) & set(sp))
        w = l = t = 0
        for tid in shared:
            a = sum(1 for s in bp[tid] if s is not None and s >= 0.999)
            b = sum(1 for s in sp[tid] if s is not None and s >= 0.999)
            if b > a:
                w += 1
            elif a > b:
                l += 1
            else:
                t += 1
        n_eff = w + l
        p = sign_p(w, l)
        obs = (w / n_eff) if n_eff else 0.5
        pw = sign_power(n_eff, max(obs, 1 - obs))
        fav = "—" if w == l else (leg["_name"] if w > l else base["_name"])
        print(f"  {leg['_name']:28s} {w:>4} {l:>4} {t:>4} {n_eff:>6} "
              f"{p:>8.3f} {pw:>6.0%}  {fav}")
        if len(shared) != len(bp):
            print(f"      NOTE: compared on {len(shared)} shared tasks; base has "
                  f"{len(bp)}. A leg missing tasks is not a fair pairing.")
        if p >= 0.05:
            verdict = "NOT significant"
            if pw < 0.50:
                verdict += (f" — and at {pw:.0%} power this does NOT distinguish "
                            f"'no effect' from 'underpowered'. Say so.")
            print(f"      {verdict}")
    if len(skills) > 1:
        print(f"\n  FWER: {len(skills)} comparisons against one base at alpha=0.05 "
              f"gives family-wise error ~{1 - 0.95 ** len(skills):.2f}. Bonferroni "
              f"to {0.05 / len(skills):.4f} is the honest correction; state which "
              f"alpha you used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
