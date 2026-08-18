#!/usr/bin/env python3
"""Take a FRESH five-hour-quota reading, or parse one already captured.

    python3 scripts/quota_fresh_reading.py              # probe now
    python3 scripts/quota_fresh_reading.py --file X.json # parse a captured stream

Exit 0 = CLEAR to launch, 1 = BLOCKED, 2 = UNKNOWN. UNKNOWN blocks like BLOCKED
does; the caller decides whether to wait or obtain a better reading.

WHY THIS EXISTS ALONGSIDE `scripts/quota_gate.py`. That one reads the last
`rate_limit_event` out of the SSB run's evolver LOGS. It is right for what it was
built for and useless for anything else: for a tau3 launch every such reading is
both stale and about a different run. Its own docstring explains why staleness is
the danger — utilization only RISES within a window, so a stale LOW reading proves
nothing about now, and treating it as clear is fail-open on the one gate whose
failure kills a run mid-flight having written nothing (2026-08-01).

So this takes its own reading. One word on haiku costs a rounding error against
the window it measures, and a reading this process just took cannot be stale by
construction — which lets the absent-utilization branch be SAFE here in a way it
is not in `quota_gate.py` (see #139: that branch bypassed the staleness check
entirely and cleared a launch on a 79-minute-old reading).

ABSENT utilization means BELOW 90%, not "the window reset". The
`session-usage-check` skill documents the ladder: `utilization` is present ONLY above
90%, so its absence under `status: allowed` is the normal below-90% payload and not the
probe failing. The first version of this file inferred "absent = just reset" from a
single observation taken minutes after a real reset — the same verdict for the wrong
reason, and the wrong reason produced a fail-open branch for
`allowed_warning` + absent (see `classify`). Read the skill before reimplementing it;
this file exists to automate that skill, not to rediscover it.

It also exists so the launcher does not carry its own copy of this parsing. Two
copies of one rule is defect #147's exact shape — the launcher's copy would be
the untestable one, and the untestable copy is the one that rots.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# the maintainer's rule, 2026-08-03: never launch a NEW evolver at or above this.
THRESHOLD = 0.96


# #306. KEYS WHOSE PRESENCE MAKES THIS PROBE MEASURE THE WRONG THING. `claude` prefers a
# resolved API key over the OAuth subscription, and the five-hour window is a property of the
# SUBSCRIPTION. On the API-key path the CLI's rate-limit handler returns before any
# `rate_limit_event` is enqueued, so the stream is not "sparse" — it is STRUCTURALLY SILENT,
# and every retry against it is guaranteed to fail in exactly the same way.
_AUTH_OVERRIDES = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL")


def probe(out: Path) -> None:
    """Emit a rate_limit_event by making the cheapest possible real request.

    #306. THE PROBE MUST RUN ON THE PATH IT CLAIMS TO MEASURE. Measured 2026-08-13 across 143
    preserved probe files, cross-tabbed on the stream's own `system/init.apiKeySource`:

        apiKeySource='none'              -> rate_limit_info present  101 / 101
        apiKeySource='ANTHROPIC_API_KEY' -> rate_limit_info present    0 /  42

    Zero counter-examples in either direction, with the two classes interleaved inside fifteen
    seconds on 08-13 — so this is not time-of-day, window position or account state. A
    three-probe intervention reproduces the flip on demand: same binary, same cwd, same command,
    the only difference being whether an operator `.env` exporting ANTHROPIC_API_KEY had been sourced.

    Nothing in this repo exports that key on the tau3 path; the operator's launch idiom
    (`set -a; . .env; set +a`) does, and it is then inherited all the way down. Scrubbing it
    HERE rather than relying on a launch convention is deliberate: a convention that must be
    remembered at every launch site is a convention that will be missed at one of them, and the
    failure is silent.
    """
    env = {k: v for k, v in os.environ.items() if k not in _AUTH_OVERRIDES}
    with out.open("w") as fh:
        subprocess.run(
            ["claude", "-p", "ok", "--model", "haiku",
             "--output-format", "stream-json", "--verbose",
             "--dangerously-skip-permissions"],
            stdout=fh, stderr=subprocess.DEVNULL, timeout=300, check=False, env=env)


def auth_source(path: Path) -> str | None:
    """What the CLI says it authenticated with, read from the stream's own init record.

    #306. ASSERTED, NEVER ASSUMED. The scrub above should guarantee the subscription path, but
    a guard that cannot observe its own precondition is the vacuous kind this project keeps
    shipping (#154, #248). If a future CLI resolves a key by some route the scrub does not
    cover, this is what turns a silent 'no reading' into a named, actionable failure.
    """
    for line in path.open(errors="replace"):
        if '"apiKeySource"' not in line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("type") == "system" and "apiKeySource" in d:
            return d.get("apiKeySource")
        for v in d.values():
            if isinstance(v, dict) and "apiKeySource" in v:
                return v["apiKeySource"]
    return None


def find_info(path: Path) -> dict | None:
    """Last rate_limit_info in the stream, wherever it is nested."""
    def walk(o):
        if isinstance(o, dict):
            if "rate_limit_info" in o:
                return o["rate_limit_info"]
            for v in o.values():
                r = walk(v)
                if r:
                    return r
        elif isinstance(o, list):
            for v in o:
                r = walk(v)
                if r:
                    return r
        return None

    info = None
    for line in path.open(errors="replace"):
        if '"rate_limit_info"' not in line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        info = walk(d) or info
    return info


def classify(info: dict | None, fresh: bool) -> tuple[int, str]:
    if info is None:
        return 2, ("UNKNOWN — no rate_limit_event in the stream. Cannot tell "
                   "whether the window is clear; treat as blocked.")
    util, status = info.get("utilization"), info.get("status")
    extra = ""
    # Overage is what would otherwise absorb an overrun. When it is unavailable
    # the five-hour limit is a HARD kill with no overflow, which is exactly the
    # condition under which the gate matters most — so say so rather than leave
    # it in the JSON for someone to notice.
    if info.get("overageStatus") == "rejected" or info.get("overageDisabledReason"):
        extra = (f"  [no overage available: {info.get('overageDisabledReason') or 'rejected'}"
                 f" — the limit is a HARD stop]")
    if util is None:
        if not fresh:
            return 2, ("UNKNOWN — utilization absent and the reading is not fresh. "
                       "Absent normally means below 90%, but a stale stream may simply "
                       "never have carried the field; only a fresh probe distinguishes "
                       "them, and guessing here is #139." + extra)
        # ABSENT + `allowed` = BELOW 90%. Per the session-usage-check skill's ladder,
        # `utilization` exists ONLY above 90% — its absence is not the probe failing and
        # it is not evidence of a reset. The first version of this function said "absent
        # means the window RESET", inferred from one observation taken minutes after a
        # real reset. Same verdict, wrong reason, and the wrong reason left the hole below.
        if status == "allowed":
            return 0, ("CLEAR — utilization absent under status=allowed, which means "
                       "below 90% (the field only exists above 90%)." + extra)
        # ABSENT + `allowed_warning` IS ANOMALOUS AND MUST NOT CLEAR. That status always
        # carries 0.90-0.99, so a missing field here means the 0.96 test cannot be
        # applied — and 0.90-0.99 straddles the threshold. The first version returned
        # CLEAR for this, i.e. fail-open on the one gate whose failure killed evolver 3
        # on 2026-08-01 with nothing written.
        if status == "allowed_warning":
            return 2, ("UNKNOWN — status=allowed_warning but NO utilization reported. "
                       "That status always carries 0.90-0.99, which straddles the 0.96 "
                       "threshold, so the gate cannot be evaluated. Treat as blocked and "
                       "re-probe." + extra)
        return 1, f"BLOCK — status={status} with no utilization reported.{extra}"
    util = float(util)
    if util >= THRESHOLD:
        return 1, (f"BLOCK — utilization {util:.3f} >= {THRESHOLD}. Do not start a "
                   f"NEW evolver; one already running is left alone." + extra)
    return 0, f"CLEAR — utilization {util:.3f} < {THRESHOLD} (status={status}).{extra}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="parse this captured stream-json instead of probing")
    ap.add_argument("--json", action="store_true", help="also print the raw rate_limit_info")
    a = ap.parse_args()

    if a.file:
        path, fresh = Path(a.file), False
        info = find_info(path)
    else:
        # #240. RETRY A MISSING MEASUREMENT — and ONLY a missing one.
        #
        # A probe that returns no rate_limit event is not a quota signal, it is a FAILED
        # READING. Observed once on 2026-08-09: a probe came back is_error=false,
        # stop_reason=end_turn, num_turns=1 -- a completely successful request -- with no
        # rate_limit event anywhere in 14 lines of stream, while probes minutes either side
        # each carried one. One occurrence, not a measured rate.
        #
        # CORRECTION, kept because the wrong version of this comment shipped for twenty
        # minutes: I originally wrote that this absence "blocked four consecutive evolver
        # dispatches". It did not. It blocked ONE. The other three never reached this gate at
        # all -- the first refusal left tau3_evolver.sh's SEEDED staging behind, which made
        # the operator gate fail earlier in the sequence, so advance() halted before the
        # dispatch step (that is #237's mechanism, in the seeded form). I read a stale
        # "REFUSING" line out of an APPENDING log three times and built a fault theory on it,
        # while the log's byte count sat unchanged at 1900B and told me so. The retry below
        # is still correct on its own terms; its original justification was not.
        #
        # THIS IS NOT FAIL-OPEN, and the distinction is the whole point. Only the
        # info-is-None case is retried. A reading that ARRIVES and says utilization is high
        # still returns BLOCK on the first attempt and is never re-rolled -- re-rolling an
        # adverse reading until it turns favourable is exactly the 2026-08-01 incident this
        # gate exists to prevent. Retrying a measurement that did not happen and retrying one
        # you dislike are opposite acts.
        # ATTEMPT COUNT MEASURED, NOT GUESSED (2026-08-13). Three was far too few. The CLI
        # emits `rate_limit_event` on only SOME otherwise-identical successful probes:
        # 12 consecutive preserved probe files, same host, same minute-scale window, showed
        #   00:06 x4 no event | 00:10 x2 EVENT | 00:11 x6 no event
        # i.e. ~2/12 ~ 17% hit rate, with every probe completing cleanly (assistant + result
        # present, no error). At 17%, three attempts miss all three about 0.83^3 = 57% of
        # the time — so the gate blocked a CLEAR window more often than not, and did it to
        # the ao6 epoch-1 dispatch three times running.
        #
        # This is NOT a weakening of the gate, and the distinction is the same one the
        # comment above draws: the number of attempts to OBTAIN a reading is independent of
        # what we do with the reading once it arrives. classify() is untouched, the
        # threshold is untouched, and an adverse reading still returns BLOCK on the first
        # attempt and is never re-rolled. More draws of a sparse signal is measurement;
        # re-rolling until you like the answer is the 2026-08-01 incident.
        #
        # 12 attempts puts the miss rate at 0.83^12 ~ 11%; the sleep is short because the
        # probes are seconds each and the failure is not rate-related.
        # RETRACTED 2026-08-13, IN PLACE, because the wrong version shipped and was quoted as
        # justification: the "~2/12 ~ 17% hit rate" above is a CONFOUND, not a measurement. Those
        # twelve files were 2 subscription probes (both of which emitted) and 10 API-key probes
        # (none of which structurally could) — the 00:06 x4, 00:11 x6 silent bursts are exactly
        # the apiKeySource='ANTHROPIC_API_KEY' class. On the path this probe now forces, measured
        # emission is 101/101 = 100% over 143 files. So ATTEMPTS is NOT sized against a sparse
        # signal; sparseness was never real, and no attempt count would have rescued a
        # structurally silent stream — which is why ao6's 12/12 miss was not bad luck.
        #
        # The retry is KEPT, small, for a different reason: one genuinely unexplained absence
        # (2026-08-09) predates the oldest preserved probe file and is therefore neither
        # confirmed nor refuted by this dataset. Lowering 12 -> 3 is defensible now that the
        # structural case exits immediately below; RAISING it never is.
        ATTEMPTS = 3
        info, path = None, None
        for attempt in range(ATTEMPTS):
            path = Path(tempfile.mkstemp(prefix="quota_probe_", suffix=".json")[1])
            probe(path)
            # #306. A PROBE THAT RAN ON THE WRONG AUTH PATH MEASURED NOTHING, and must not be
            # retried — every further attempt is silent for the same structural reason. Fail
            # LOUD and specific rather than exhausting the loop into a generic UNKNOWN, which is
            # what cost ao6 five dispatch attempts and hid the cause for a day.
            src = auth_source(path)
            if src not in (None, "none"):
                print(f"UNKNOWN — the probe authenticated with {src!r}, not the OAuth "
                      f"subscription. The five-hour window is a property of the SUBSCRIPTION, "
                      f"and on the API-key path the CLI emits no rate_limit_event at all "
                      f"(measured 0/42). This reading is structurally impossible, not absent — "
                      f"retrying cannot help. Unset {'/'.join(_AUTH_OVERRIDES)} for the probe.",
                      file=sys.stderr)
                return 2
            info = find_info(path)
            if info is not None:
                if attempt:
                    print(f"  (reading obtained on probe {attempt + 1}/{ATTEMPTS} — the "
                          f"event is emitted on only a minority of probes)", file=sys.stderr)
                break
            if attempt < ATTEMPTS - 1:
                print(f"  (probe {attempt + 1}/{ATTEMPTS} returned no rate_limit event — "
                      f"re-probing; a missing reading is not a quota signal)", file=sys.stderr)
                time.sleep(2)
        fresh = True

    code, msg = classify(info, fresh)
    print(msg)
    if a.json:
        print(json.dumps(info))
    if info and info.get("resetsAt"):
        import datetime
        print("  window resets at "
              + datetime.datetime.fromtimestamp(info["resetsAt"]).strftime("%H:%M:%S"))
    return code


if __name__ == "__main__":
    sys.exit(main())
