"""The three sealed-test enforcement points must agree, and must be testable.

The live hook, the post-hoc audit and invariants.check_no_heldout_reachable all decide
"does this text name held-out benchmark material". They carried three separate copies of
that rule; consolidating them into src/leak_patterns.py fixed the PATTERNS but left the
audit's own membership test on the raw split form, so it certified CLEAN on an SSB v2
held-out read written as a filename while invariants FAILED the same log.

The third re-audit's own words: "grep -rn 'heldout|leak_hook|leak_audit|id_forms|
held_forms' tests/ returns nothing; no test imports src/leak_patterns.py" -- so every
verification of this machinery had been one-off and manual, which is why two rounds of
fixes each left a hole outside whatever pairs were tried by hand.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "scripts" / "evolver_leak_audit.py"


def _session(path: Path, filename: str, body: str) -> Path:
    rows = [
        {"type": "assistant", "session_id": "s1", "message": {"content": [
            {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": filename}}]}},
        {"type": "user", "session_id": "s1", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": body}]}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


SSBV2_SPLIT = {"train": ["Debugging/01_01", "Debugging/01_02"],
               "val": ["Debugging/05_05"], "test": ["Template/16_02"]}
TAU3_SPLIT = {"train": ["task_001"], "val": ["task_097"], "test": ["task_052"]}


@pytest.mark.parametrize("split,filename,want_rc", [
    # SSB v2 held-out, BOTH spellings -- the filename form is the one that used to pass
    (SSBV2_SPLIT, "/peer/digests/Debugging_05_05.md", 4),
    (SSBV2_SPLIT, "/peer/digests/Debugging/05_05.md", 4),
    (SSBV2_SPLIT, "/peer/digests/Template_16_02.md", 4),
    # ...and train in either spelling stays clean
    (SSBV2_SPLIT, "/own/digests/Debugging_01_02.md", 0),
    (SSBV2_SPLIT, "/own/digests/Debugging/01_02.md", 0),
    (TAU3_SPLIT, "/peer/digests/task_097.md", 4),
    (TAU3_SPLIT, "/own/digests/task_001.md", 0),
])
def test_post_hoc_audit_sees_both_id_spellings(tmp_path, split, filename, want_rc):
    sp = tmp_path / "split.json"; sp.write_text(json.dumps(split))
    sess = _session(tmp_path / "s.jsonl", filename, f"# {Path(filename).stem}\nbody text")
    r = subprocess.run([sys.executable, str(AUDIT), str(sess), str(sp)],
                       capture_output=True, text=True, cwd=REPO)
    assert r.returncode == want_rc, f"{filename} -> rc={r.returncode}\n{r.stdout}\n{r.stderr}"


def test_audit_and_invariants_agree(tmp_path):
    """The artifact an operator runs to certify an epoch must not disagree with the loop."""
    from src.orchestrator.invariants import check_no_heldout_reachable
    sp = tmp_path / "split.json"; sp.write_text(json.dumps(SSBV2_SPLIT))
    sess = _session(tmp_path / "s.jsonl", "/peer/digests/Debugging_05_05.md",
                    "# Debugging_05_05\nexpected: SUM over B2:B40")
    rc = subprocess.run([sys.executable, str(AUDIT), str(sess), str(sp)],
                        capture_output=True, text=True, cwd=REPO).returncode
    inv = str(getattr(check_no_heldout_reachable(sess, sp), "status", "?")).upper()
    assert (rc == 4) and (inv == "FAIL"), f"audit rc={rc} but invariants={inv} — they disagree"


@pytest.mark.parametrize("command,should_match", [
    ("cp -r /x/data/tau2/domains/banking_knowledge/tasks /tmp/specs", True),
    ("cat /x/domains/banking_knowledge/tasks/task_052.json", True),
    ("cp /d/spreadsheetbench-v2/*/*/*_answer.xlsx /tmp/all/", True),
    ("cat /d/data/spreadsheetbench-v2/Debugging/05_05/1_answer.xlsx", True),
    ("cat /env/tools.py", False),
    ("ls digests_e1_train", False),
    ("cat notes/tasks.md", False),          # not under domains/ -- must not fire
])
def test_shared_leak_paths(command, should_match):
    from src.leak_patterns import LEAK_PATH_RE
    assert bool(LEAK_PATH_RE.search(command)) is should_match, command


def test_hook_runs_under_the_system_interpreter():
    """Claude Code invokes the hook as `python3 <path>`, not with the project venv.

    src/leak_patterns.py therefore has to stay stdlib-only; it was briefly placed under
    src/harness/, whose __init__ imports pydantic, and the hook died on import and exited
    1 -- which does not block.
    """
    hook = REPO / "scripts" / "evolver_leak_hook.py"
    payload = json.dumps({"tool_name": "Read", "tool_input": {"file_path": "/d/task_097.json"}})
    r = subprocess.run(["/usr/bin/python3", str(hook)], input=payload,
                       capture_output=True, text=True, cwd=REPO,
                       env={"PATH": "/usr/bin:/bin", "EVOSKILL_EVOLVER_TRAIN_IDS": "task_001"})
    assert r.returncode == 2, f"rc={r.returncode}, stderr={r.stderr[:300]}"
    assert "BLOCKED" in r.stderr and "BLIND" not in r.stderr


# ── The parent-directory escape, and the probe that has to be able to fail ────────────────
DOM = "third_party/tau3-bench/data/tau2/domains/banking_knowledge"


def _hook(command: str, deny: str = "") -> tuple[int, str]:
    r = subprocess.run(["/usr/bin/python3", str(REPO / "scripts" / "evolver_leak_hook.py")],
                       input=json.dumps({"tool_name": "Bash", "tool_input": {"command": command}}),
                       capture_output=True, text=True, cwd=REPO,
                       env={"PATH": "/usr/bin:/bin", "EVOSKILL_EVOLVER_TRAIN_IDS": "task_001",
                            "EVOSKILL_LEAK_DENY_PATHS": deny})
    return r.returncode, r.stderr


@pytest.mark.parametrize("command,blocked", [
    # THE WHOLE-SUBTREE ESCAPES: neither command names a guarded path, and both take the
    # 97-task pool with them. macOS seatbelt covered these; Linux had nothing.
    (f"cp -r {DOM} /tmp/bk && cat /tmp/bk/tasks.json", True),
    (f"cp -r {DOM}/ /tmp/bk", True),                      # one keystroke away
    (f"cd {DOM} && cat tasks.json", True),
    (f"tar czf /tmp/d.tgz third_party/tau3-bench/src/tau2/domains/telecom", True),
    # THE CORPUS THE EVOLVER IS DELIBERATELY GIVEN. tau3_evolver.sh grants documents/ and its
    # design note is explicit that the environment stays readable; over-blocking here would
    # forbid env-aware evolution, so these must survive every spelling.
    (f"ls {DOM}/documents | head", False),
    (f"cd {DOM}/documents && grep -rl wire .", False),
    ("cat third_party/tau3-bench/src/tau2/domains/banking_knowledge/tools.py", False),
])
def test_whole_subtree_escapes_blocked_corpus_readable(command, blocked):
    rc, err = _hook(command)
    assert (rc == 2) is blocked, f"rc={rc} for: {command}\n{err[:300]}"


def test_deny_list_probe_can_actually_fail():
    """The launcher proves the deny list is armed. That proof must be falsifiable.

    tau3's deny roots sit under paths LEAK_PATH_RE matches anyway, so while the pattern
    check ran first the probe was answered by the pattern and would have passed with
    EVOSKILL_LEAK_DENY_PATHS unset -- a guard that cannot fail (#154). The hook now
    reports the operator-declared root first, and tau3_evolver.sh requires THAT reason.
    """
    deny = str(REPO / DOM / "tasks")
    probe = f"cat {deny}/__held_out_probe__"
    want = "reads a declared held-out data root"

    rc, err = _hook(probe, deny=deny)
    assert rc == 2 and want in err, f"armed probe did not cite the deny root:\n{err[:300]}"

    rc, err = _hook(probe, deny="")           # the same probe, deny list NOT configured
    assert want not in err, (
        "the probe reports the deny root as the reason even with the deny list unset, so "
        "the launcher's check passes whether or not it is armed")


def test_launcher_requires_the_deny_root_reason():
    """Structural: the shell must compare against the specific reason, not just BLOCKED."""
    sh = (REPO / "scripts" / "tau3_evolver.sh").read_text()
    assert '_probe_want="reads a declared held-out data root"' in sh
    assert '_denied_ok "$_probe_want"' in sh, "probe still accepts any BLOCKED banner"


def test_tau3_runs_declare_a_non_empty_deny_list():
    """tau3_evolver.sh's --hide loop is guarded on this being non-empty; on tau3 it was not."""
    from src.orchestrator.loop import _require_tau3_deny_paths
    paths = _require_tau3_deny_paths("banking_knowledge").split(":")
    assert paths and all(Path(p).exists() for p in paths), paths
    # ...and it refuses rather than handing back an affirmative empty list
    for bad in ("", "nosuchdomain"):
        with pytest.raises(RuntimeError):
            _require_tau3_deny_paths(bad)
