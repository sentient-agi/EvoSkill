#!/usr/bin/env python3
"""Tests for the tau3 digest outcome-kind label.  Run: python3 -m pytest -q

WHY THIS EXISTS. On 2026-08-04 the iteration-1 evolver prompt claimed one episode
"ran 2033s and finished normally". It had not: that episode terminated `timeout`
at reward 0.0 with an EMPTY `reward_basis`, meaning no check ever ran. The 0.0 was
the absence of a verdict, not a wrong answer.

The fix moved that distinction out of hand-written prose and into a computed
`kind=` field on every sample. These tests exist so the field cannot quietly
degenerate into something that agrees with the data only by coincidence:

  * the two CROSSED cases (a graded timeout, an ungraded user_stop) do not occur
    in the epoch-0 batch, where termination and basis agree perfectly on all 96
    episodes. A classifier keying on `termination_reason` would pass every test
    written from that batch alone and still be wrong. Those two cases are the
    whole point — they are what make this a test rather than a restatement.
  * the shell-quoting case is a real regression: the launcher `eval`s the derived
    values, and a bare `ungraded(timeout)` is a shell syntax error that killed it
    before any gate could run.
"""
import pathlib
import re
import shlex
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILDER = ROOT / "scripts" / "tau3_build_digests.py"
LAUNCHER = ROOT / "scripts" / "tau3_evolver.sh"


def _classify():
    """Load classify_outcome without executing the module body (which needs a cache)."""
    src = BUILDER.read_text()
    start = src.index("def classify_outcome")
    end = src.index("def render_grade")
    ns: dict = {}
    exec(src[start:end], ns)
    return ns["classify_outcome"]


classify = _classify()


def fa(basis, term="user_stop"):
    return {"reward_basis": basis, "termination_reason": term}


@pytest.mark.parametrize("basis,term,score,want", [
    # the three shapes that actually occur in the epoch-0 train batch
    (["DB"], "user_stop", 0.0, "fail"),
    (["DB"], "user_stop", 1.0, "pass"),
    (["ACTION"], "user_stop", 1.0, "pass"),
    ([], "timeout", 0.0, "ungraded(timeout)"),
])
def test_shapes_present_in_the_real_batch(basis, term, score, want):
    assert classify(fa(basis, term), score) == want


@pytest.mark.parametrize("basis,term,score,want", [
    # THE CROSSED CASES. Neither occurs in epoch 0, so neither is covered by
    # "it reproduces the census". A termination-keyed classifier fails both.
    (["DB"], "timeout", 0.0, "fail"),
    ([], "user_stop", 0.0, "ungraded(user_stop)"),
    # and a reason the batch has never produced must still be labelled, not dropped
    ([], "max_errors", 0.0, "ungraded(max_errors)"),
])
def test_label_keys_on_reward_basis_not_on_termination(basis, term, score, want):
    assert classify(fa(basis, term), score) == want


def test_missing_verdict_is_not_a_failure():
    assert classify(None, None) == "no_verdict"


def test_basis_without_a_score_is_not_silently_bucketed():
    assert classify(fa(["DB"]), None) == "unscored"


def test_absent_basis_key_is_treated_as_empty():
    assert classify({"termination_reason": "timeout"}, 0.0) == "ungraded(timeout)"


def test_absent_termination_is_named_not_blank():
    assert classify({"reward_basis": []}, 0.0) == "ungraded(unknown)"


def test_pass_threshold_matches_the_rest_of_the_pipeline():
    # 0.999, the same threshold the digest builder and runner use for PASS
    assert classify(fa(["DB"]), 0.999) == "pass"
    assert classify(fa(["DB"]), 0.998) == "fail"


# ── the launcher must survive a label containing shell metacharacters ─────────

def test_launcher_shell_quotes_the_derived_census():
    """`eval "$DERIVED"` runs these as assignments; bare parens are a syntax error."""
    src = LAUNCHER.read_text()
    for var in ("KINDS", "TERMS"):
        m = re.search(rf'print\("{var}=" \+ (\w+)', src)
        assert m, f"{var} assignment not found in the launcher"
        assert m.group(1) == "shlex", (
            f"{var} is pasted into `eval` unquoted; a kind label like "
            f"ungraded(timeout) would be a shell syntax error")


def test_a_kind_label_is_eval_safe_once_quoted():
    label = "fail:64,pass:26,ungraded(timeout):6"
    line = f"KINDS={shlex.quote(label)}"
    out = subprocess.run(["bash", "-c", f'eval "{line}"; printf %s "$KINDS"'],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout == label


def test_the_unquoted_form_really_does_break():
    """Guard against the test above passing for the wrong reason.

    Assert on the SYNTAX ERROR, not on the exit code. `bash -c 'eval "..."; echo ok'`
    exits 0 even when the eval fails, because the trailing command succeeds — the
    first version of this test asserted returncode != 0 and failed for that reason
    while the underlying breakage was real. The launcher runs under `set -euo
    pipefail`, so it aborts; a bare `bash -c` does not.
    """
    out = subprocess.run(
        ["bash", "-c", 'set -e; eval "KINDS=fail:64,ungraded(timeout):6"'],
        capture_output=True, text=True)
    assert out.returncode != 0 and "syntax error" in out.stderr, (
        "bare parens no longer break eval — if so this whole guard is vacuous "
        f"and the shlex requirement above should be re-justified. stderr={out.stderr!r}")



# ── the shared transcript renderer must not eat a customer's words ────────────
# `compact_ssb_user_query` runs on EVERY role=="user" message. In tau2/tau3 that
# is a live customer speaking, so a generic "strip everything before the first
# ### heading" rule silently deletes the start of their message. Found by
# cognitive proofread 2026-08-04; no live corruption had occurred, because the
# tau2 end markers have no whitespace after the hashes.

def test_customer_utterance_with_a_hash_heading_is_not_truncated():
    from src.harness.agent import compact_ssb_user_query
    hazards = [
        "I need my limit raised.\n### Note\nplease hurry",
        "Please check ### 12345 on my statement",
        "Thank you! Have a great day. ###TRANSFER###",
        "Thanks again! Take care!\n\n###STOP###",
    ]
    for s in hazards:
        assert compact_ssb_user_query(s) == s, (
            f"a customer utterance was truncated: {s!r} -> "
            f"{compact_ssb_user_query(s)!r}")


def test_the_documented_ssb_compaction_still_works():
    """Guard the other direction: the real SSB template must still compact."""
    from src.harness.agent import compact_ssb_user_query
    ssb = ("You are a spreadsheet expert. Do the following.\n"
           "### instruction\nFix the totals column.\n")
    out = compact_ssb_user_query(ssb)
    assert out.startswith("### instruction"), out
    assert "spreadsheet expert" not in out, "the preamble should still be stripped"

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
