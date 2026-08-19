"""What the evolver profile grants, proved by EXECUTING reads under it.

Two grants were far wider than the comments beside them claimed, and both widenings came
from a coarse rule written for a different purpose:

  ~/.claude was granted as a whole subpath with only `projects` taken back. Its siblings
  carry the same material in other shapes — file-history/ holds snapshots of every file any
  Claude Code session on the host has edited (on the authors' machine, held-out ids from
  the shipped smoke splits), history.jsonl the prompts, backups/ and session-env/ whole
  copies. A sandboxed evolver read all of them past both hooks.

  /var and /private were granted as whole subpaths as "the minimum needed to run a shell",
  and every arm's evolver work dir is $TMPDIR/evoskill_tau3_evolver_e<N> — under /var/folders.
  So one arm could read AND write a peer arm's cwd, its dispatched PROMPT.md included: the
  cross-arm leak the WORK_TAG change was meant to close, reopened by a system grant.

These tests run the real profile through containment_probe(), which attempts the read
rather than reasoning about SBPL. Rule order in that profile is load-bearing (#296) and has
been inverted once already, so nothing here inspects the profile TEXT.
"""
import tempfile
from pathlib import Path

import pytest

from src.orchestrator.evolver_dispatch import (
    EvolverJob, _sandbox_exec_available, containment_probe,
)

pytestmark = pytest.mark.skipif(
    not _sandbox_exec_available(),
    reason="sandbox-exec is macOS-only; containment is not enforceable on this host")


@pytest.fixture
def job(tmp_path):
    """A job whose work dir sits where the real one does — under $TMPDIR, beside its peers."""
    tmp = Path(tempfile.gettempdir())
    own = tmp / "evoskill_tau3_evolver_e1"
    own.mkdir(exist_ok=True)
    (own / "mine.md").write_text("this run's own working file\n")
    return EvolverJob(epoch=1, work_dir=own, prompt="x", model="opus",
                      log_path=own / "e1.log", grants=[], writes=[own])


def test_a_peer_arms_work_dir_is_unreadable(job):
    peer = Path(tempfile.gettempdir()) / "evoskill_tau3_evolver_e9"
    peer.mkdir(exist_ok=True)
    (peer / "PROMPT.md").write_text("PEER ARM'S DISPATCHED PROMPT\n")
    for target in (peer / "PROMPT.md", peer):
        readable, detail = containment_probe(job, target)
        assert not readable, f"peer arm's {target.name} is READABLE: {detail}"


@pytest.mark.parametrize("rel", [
    "projects", "file-history", "backups", "session-env", "shell-snapshots",
    "history.jsonl",
])
def test_other_sessions_content_under_home_claude_is_unreadable(job, rel):
    target = Path.home() / ".claude" / rel
    if not target.exists():
        pytest.skip(f"~/.claude/{rel} absent on this host")
    readable, detail = containment_probe(job, target)
    assert not readable, f"~/.claude/{rel} is READABLE by the evolver: {detail}"


def test_the_job_can_still_read_its_own_work_dir(job):
    """The denies above are by NAME and match this job's dir too; its own grant must win.
    Without this, the whole file passes by breaking the evolver."""
    readable, detail = containment_probe(job, job.work_dir / "mine.md")
    assert readable, f"the evolver cannot read its own working file: {detail}"
