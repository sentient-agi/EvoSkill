"""scripts/daytona/ssb_leg_daytona_loop.py + daytona_lib_loop.py: the pieces of the SSB v2 leg library that the
Claude Code legs share (the run's declared split, the leg machines' delete rule).

NO DAYTONA, NO OPENROUTER, NO CLAUDE. The Daytona client is an in-memory fake; the ledger is redirected into the
test's temp dir.
"""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "daytona"))
os.environ.setdefault("EVOSKILL_DAYTONA_POLL_S", "0")

import daytona_lib_loop as dlx  # noqa: E402
import ssb_leg_daytona_loop as M  # noqa: E402

# The run's split (57/57/104): a leg runs on whichever split its run's SPEC.json declares; this one is only the
# default for a command with no run to ask.
SPLIT = ROOT / M.PROJECT_SPLIT[0]
N_TRAIN, N_VAL, N_TEST = 57, 57, 104


class FakeSandbox:
    def __init__(self, name, sid):
        self.name, self.id, self.state = name, sid, "started"


class FakeClient:
    def __init__(self):
        self.live, self.deleted = {}, set()

    def get(self, key):
        for sb in self.live.values():
            if key in (sb.id, sb.name):
                return sb
        raise RuntimeError("Sandbox not found (404)")

    def delete(self, sb, timeout=None, wait=True):
        self.live.pop(sb.id, None)
        self.deleted.add(sb.id)

    def list(self):
        return list(self.live.values())


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(dlx, "LEDGER", str(tmp_path / "ledger.jsonl"))
    rd = tmp_path / "ssbv2_fx_run"
    rd.mkdir()
    spec = {"spec": {"solver_model": "claude-sonnet-5", "agent_turns": 45, "n_samples": 3,
                     "split_path": str(SPLIT), "split_sha256": M.sha256_file(SPLIT)}}
    (rd / "SPEC.json").write_text(json.dumps(spec))
    return types.SimpleNamespace(tmp=tmp_path, rd=rd, client=FakeClient())


def ledger(w):
    p = Path(dlx.LEDGER)
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def _declare(w, split, sha=None):
    sp = json.loads((w.rd / "SPEC.json").read_text())
    sp["spec"].update(split_path=str(split), split_sha256=sha or M.sha256_file(split))
    (w.rd / "SPEC.json").write_text(json.dumps(sp))


# ══ the split: the run's declaration, not a code pin ═══════════════════════════════════════════════
def test_the_split_is_not_a_baseline_code_pin_and_hashes_as_documented():
    assert "split_sha256" not in M.EXPECTED["baseline_runtime"]
    assert M.sha256_file(SPLIT) == M.PROJECT_SPLIT[1]
    sp = json.loads(SPLIT.read_text())
    assert (len(sp["train"]), len(sp["val"]), len(sp["test"])) == (N_TRAIN, N_VAL, N_TEST)
    assert sp["dropped"]["n"] == 48 and not set(sp["dropped"]["task_ids"]) & set(sp["train"] + sp["val"] + sp["test"])
    assert M.default_split() == SPLIT


def test_declared_split_and_default_split_follow_the_runs_spec(world, tmp_path):
    assert M.declared_split(world.rd) == (SPLIT.resolve(), M.PROJECT_SPLIT[1])
    assert M.default_split(world.rd) == SPLIT.resolve()
    bare = tmp_path / "no_spec_run"
    bare.mkdir()
    assert M.declared_split(bare) == (None, None) and M.default_split(bare) == SPLIT
    other = tmp_path / "other_split.json"
    other.write_text(json.dumps({"train": [], "val": [], "test": []}))
    _declare(world, other)
    assert M.default_split(world.rd) == other.resolve()
    assert M.declared_split(world.rd) == (other.resolve(), M.sha256_file(other))
    (world.rd / "SPEC.json").write_text("{not json")
    with pytest.raises(M.Refused, match="not a readable spec"):
        M.declared_split(world.rd)


# ══ the leg machines' delete rule ════════════════════════════════════════════════════════════════
def test_delete_refuses_foreign_and_protected_sandboxes(world, monkeypatch):
    c = world.client
    ours = FakeSandbox(f"{dlx.PREFIX}v1tr-s01-1", "00000001-ours")
    other = FakeSandbox("evoskill-grade-t-030406", "27d6e707-other")
    c.live.update({ours.id: ours, other.id: other})
    assert dlx.delete_by_id(c, other.id, "t") is False and other.id in c.live
    monkeypatch.setattr(dlx, "PROTECTED_ID_PREFIXES", ("9847633d",))
    assert dlx.delete_by_id(c, "9847633d-protected", "t") is False
    assert dlx.delete_by_id(c, ours.id, "t") is True and ours.id in c.deleted and ours.id not in c.live
    ev = {e["event"] for e in ledger(world)}
    assert {"delete_refused_not_ours", "delete_refused_protected"} <= ev
