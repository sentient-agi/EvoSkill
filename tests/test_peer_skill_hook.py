"""The peer-skill hook, which had no test at all.

It exists so one arm cannot derive its skill set from another arm's answers to the same
train tasks — that would measure copying, not evolution. Three successive fixes each
closed one spelling and left another open, and every check was a launcher probe naming
`skills_v1`, so the `skills_v1_staging` spelling stayed ALLOWED across all of them.

The hard case: every arm names its staging `skills_v{N}_staging`, so after `cd <peer run>`
a relative path has the SAME first segment as this run's own. Name alone cannot answer it;
where the command stands is what decides, and that is what the hook now asks.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / "scripts" / "evolver_peer_skill_hook.py"
OWN, PEER = "/runs/armB", "/runs/armA"
STAGING = f"{OWN}/skills_v1_staging"


def _rc(command: str, cwd: str = OWN, staging: str = STAGING) -> int:
    payload = {"tool_name": "Bash", "cwd": cwd, "tool_input": {"command": command}}
    return subprocess.run([sys.executable, str(HOOK)], input=json.dumps(payload),
                          capture_output=True, text=True,
                          env={"PATH": "/usr/bin:/bin",
                               "EVOSKILL_EVOLVER_STAGING": staging}).returncode


@pytest.mark.parametrize("command,cwd", [
    # THE HOLE: same read, relative spelling, after a cd into the peer.
    (f"cd {PEER} && cat skills_v1_staging/a/SKILL.md", OWN),
    (f"cd {PEER} && for f in skills_v1_staging/*/SKILL.md; do cat \"$f\"; done", OWN),
    # ...the spellings earlier rounds did cover, which must stay covered
    (f"cd {PEER} && for f in skills_v1/*/SKILL.md; do cat \"$f\"; done", OWN),
    (f"cd {PEER} && for d in skills_v1 skills_v2; do ls -la $d; done", OWN),
    (f"cat {PEER}/skills_v1_staging/a/SKILL.md", OWN),
    (f"ls -la {PEER}/skills_v1", OWN),
    ("cat ../armA/skills_v1_staging/a/SKILL.md", OWN),
    # unresolvable moves are not proof of ownership
    ("cd $PEER && cat skills_v1_staging/a/SKILL.md", OWN),
    ("cd && cat skills_v1_staging/a/SKILL.md", OWN),
    # ...and a session already standing in the peer needs no cd at all
    ("cat skills_v1_staging/a/SKILL.md", PEER),
])
def test_peer_reads_are_denied(command, cwd):
    assert _rc(command, cwd) == 2, f"ALLOWED: {command!r} (cwd={cwd})"


@pytest.mark.parametrize("command,cwd", [
    # ITS OWN, every ordinary spelling. Over-blocking here is silent — a denied evolver
    # just works around it — so these matter as much as the denials.
    ("cat skills_v1_staging/a/SKILL.md", OWN),
    ("ls -la ./skills_v1_staging", OWN),
    (f"cd {OWN} && cat skills_v1_staging/a/SKILL.md", OWN),
    (f"cd {STAGING} && ls a/SKILL.md", OWN),
    (f"cat {STAGING}/a/SKILL.md", OWN),
    ("mkdir -p skills_v1_staging/new && cp x skills_v1_staging/new/SKILL.md", OWN),
    # #215: prose that merely contains the word
    ("echo 'the skills library and docs/skills notes'", OWN),
])
def test_own_staging_is_allowed(command, cwd):
    assert _rc(command, cwd) == 0, f"DENIED: {command!r} (cwd={cwd})"


def test_the_two_spellings_of_one_read_agree():
    """The defining symptom: absolute denied, relative allowed, same bytes."""
    rel = _rc(f"cd {PEER} && cat skills_v1_staging/a/SKILL.md")
    abs_ = _rc(f"cat {PEER}/skills_v1_staging/a/SKILL.md")
    assert rel == abs_ == 2, f"relative={rel} absolute={abs_} — the same read, two answers"


def test_launcher_probes_cover_the_staging_spelling():
    """Structural: the launcher's must-deny probes named only `skills_v1`."""
    sh = (Path(__file__).resolve().parents[1] / "scripts" / "tau3_evolver.sh").read_text()
    assert "skills_v1_staging/a/SKILL.md" in sh, "no probe names the staging spelling"


def test_unset_staging_fails_closed():
    """With no staging configured the hook has no basis to call anything its own, and it
    denies every versioned skill path rather than waving them through.

    Pinned deliberately: fail-OPEN here would mean a run that forgot to export
    EVOSKILL_EVOLVER_STAGING silently lost peer protection entirely while the launcher
    still printed its probes as passing. The launcher refuses to dispatch without the
    variable, so the cost of failing closed is a loud stop, not a contaminated run."""
    assert _rc("cat skills_v1_staging/a/SKILL.md", OWN, staging="") == 2
    assert _rc("echo 'skills and docs'", OWN, staging="") == 0   # ...prose still passes
