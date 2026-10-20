"""The run's ONE Claude Code credential (src/orchestrator/cc_account.py), and the leg driver's two scheduling rules:
every session of a leg starts on that credential, and a val leg admits no unit while a train leg of the same run still
has units waiting for a seat (training attempts come first). Offline, tmp only; no token is ever real."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts" / "daytona", ROOT / "scripts" / "cc_solver"):
    sys.path.insert(0, str(p))

from src.orchestrator import cc_account as ACC  # noqa: E402
import cc_leg_driver as D  # noqa: E402
import cc_solver as C  # noqa: E402


@pytest.fixture
def token(tmp_path):
    f = tmp_path / "token_standin"
    f.write_text("sk-ant-oat01-" + "a" * 60 + "\n")
    os.chmod(f, 0o600)
    return f


def test_the_declared_token_file_wins_over_the_environment_and_the_api_key(token, tmp_path):
    other = tmp_path / "other_token"
    other.write_text("x")
    other.chmod(0o600)
    env = {ACC.TOKEN_FILE_ENV: str(other), ACC.API_KEY_ENV: "sk-ant-api03-x"}
    c = ACC.resolve(token, env)
    assert (c.kind, c.token_file, c.source) == (ACC.TOKEN_FILE, token, "declared")
    c = ACC.resolve(None, env)
    assert (c.kind, c.token_file, c.source) == (ACC.TOKEN_FILE, other, ACC.TOKEN_FILE_ENV)
    c = ACC.resolve("  ", {ACC.API_KEY_ENV: "sk-ant-api03-x"})
    assert (c.kind, c.token_file, c.source) == (ACC.API_KEY, None, ACC.API_KEY_ENV)


def test_no_credential_and_an_unusable_token_file_are_refused(token, tmp_path):
    with pytest.raises(ValueError, match="no Claude Code credential"):
        ACC.resolve(None, {})
    with pytest.raises(ValueError, match="no Claude Code credential"):
        ACC.resolve(None, {ACC.API_KEY_ENV: "   "})
    with pytest.raises(ValueError, match="not a regular file"):
        ACC.resolve(tmp_path / "absent", {})
    link = tmp_path / "link"
    link.symlink_to(token)
    with pytest.raises(ValueError, match="not a regular file"):
        ACC.resolve(link, {})
    token.chmod(0o640)
    with pytest.raises(ValueError, match="want mode 0600"):
        ACC.resolve(token, {})
    # check=False names the path without a stat (a path the caller will check itself)
    assert ACC.resolve(tmp_path / "absent", {}, check=False).token_file == tmp_path / "absent"


def test_the_token_file_is_never_opened(token, monkeypatch):
    import builtins
    import io
    real = io.open

    def guarded(file, *a, **k):
        if str(file) == str(token):
            raise AssertionError("cc_account opened the token file")
        return real(file, *a, **k)
    monkeypatch.setattr(io, "open", guarded)
    monkeypatch.setattr(builtins, "open", guarded)
    assert ACC.resolve(token, {}).token_file == token


def test_an_api_key_run_is_recognised_and_widens_the_allowed_key_sources(token):
    key = {ACC.API_KEY_ENV: "sk-ant-api03-x"}
    assert ACC.uses_api_key(key) is True
    assert ACC.uses_api_key({**key, ACC.TOKEN_FILE_ENV: str(token)}) is False
    assert ACC.uses_api_key(key, declared=str(token)) is False
    assert ACC.uses_api_key({}) is False                                 # no credential: False, never raises
    assert ACC.allowed_api_key_sources(key) == ACC.OAUTH_API_KEY_SOURCES + ACC.API_KEY_SOURCES
    assert ACC.allowed_api_key_sources({**key, ACC.TOKEN_FILE_ENV: str(token)}) == ACC.OAUTH_API_KEY_SOURCES


def test_the_cli_names_the_credential_and_never_the_secret(token, monkeypatch, capsys):
    monkeypatch.setenv(ACC.TOKEN_FILE_ENV, str(token))
    assert ACC.main([]) == 0
    out = capsys.readouterr().out
    assert str(token) in out and "sk-ant-oat01" not in out
    monkeypatch.delenv(ACC.TOKEN_FILE_ENV)
    assert ACC.main([]) == 1


def test_the_solver_queue_launches_with_token_for_and_scans_with_the_launch_token():
    seen = {}

    class Box:
        status = "running"
        rec: dict = {}

        def prepare(self): pass
        def up(self, init_root=False): pass
        def probe(self): pass
        def upload(self): pass
        def toolchain(self): pass

        def launch(self, tok):
            self.tok_used = tok
            seen["launch"] = tok

        def collect(self, tok): seen["collect"] = tok
        def wipe(self): return True
        def result(self): return {}

    q = object.__new__(C.Queue)
    q.m = SimpleNamespace(lock=__import__("threading").Lock())
    q.guard, q.stopping, q.tok = {"admitting": True, "reason": None}, False, "declared"
    q.token_for = lambda b: "resolved-now"
    b = Box()
    C.Queue._setup(q, b)
    C.Queue._finish(q, b)
    assert seen == {"launch": "resolved-now", "collect": "resolved-now"}


def test_live_train_leases(tmp_path, monkeypatch):
    from src.orchestrator import pool_container_evolver as pce
    run_dir = tmp_path / "myrun"
    run_dir.mkdir()
    leases = tmp_path / "pool" / "leases"
    leases.mkdir(parents=True)
    monkeypatch.setattr(pce, "pool_machines_file", lambda rd, runtime=None: tmp_path / "pool" / "machine.json")
    now = time.time()
    rows = {"ccleg-myrun_v1_train-1": (now + 600, os.getpid()), "ccleg-myrun_v1_val-1": (now + 600, os.getpid()),
            "ccleg-myrun_base_train-0": (now - 5, os.getpid()), "ccleg-other_v1_train-1": (now + 600, os.getpid()),
            "evolver-run-x": (now + 600, os.getpid()), "ccleg-myrun_v2_train-2": (now + 600, 2 ** 22 + 12345)}
    for h, (until, pid) in rows.items():
        (leases / f"{h}.json").write_text(json.dumps({"holder": h, "sandbox_id": "s", "until_ts": until, "pid": pid}))
    run = SimpleNamespace(run_dir=run_dir)
    assert D.LegRun.live_train_leases(run) == ["ccleg-myrun_v1_train-1"]   # expired, other run, dead pid excluded


def _queue(key, live, pending=3, inflight=0):
    q = object.__new__(D.LegQueue)
    q.drv = SimpleNamespace(key=key, train_legs_waiting=lambda: live)
    q.guard, q.pending, q.inflight, q.held = {"admitting": True}, list(range(pending)), list(range(inflight)), False
    return q


def test_a_val_leg_holds_admission_while_a_train_leg_is_live(monkeypatch):
    calls = []
    monkeypatch.setattr(C.Queue, "admit", lambda self: calls.append("admit"))
    monkeypatch.setattr(D.time, "sleep", lambda s: calls.append(f"sleep {s}"))
    q = _queue("val", ["ccleg-r_v1_train-1"])
    D.LegQueue.admit(q)
    assert calls == [f"sleep {D.PRIORITY_HOLD_S}"] and q.held          # held, nothing in flight -> no spin
    q.inflight = [1]
    calls.clear()
    D.LegQueue.admit(q)
    assert calls == []                                                   # in-flight val units finish, none admitted
    q.drv.train_legs_waiting = lambda: []
    D.LegQueue.admit(q)
    assert calls == ["admit"] and not q.held                             # no train unit waits: admission resumes
    calls.clear()
    D.LegQueue.admit(_queue("train", ["ccleg-r_v1_train-1"]))
    assert calls == ["admit"]                                            # a train leg is never held


def test_val_takes_free_seats_once_no_train_unit_waits(tmp_path):
    """A live train leg holds val only while its STATE.json has a unit still 'queued' (waiting for a seat); once every
    train unit has a seat, val takes the free ones. An unreadable state counts as waiting (train first)."""
    store = tmp_path / "run" / "r_base_val"
    tr = tmp_path / "run" / "r_base_train"
    tr.mkdir(parents=True)
    store.mkdir()
    holder = "ccleg-r_base_train-20261004T045106"
    run = SimpleNamespace(store=store, live_train_leases=lambda: [holder])
    st = tr / "STATE.json"
    st.write_text(json.dumps({"units": {"t#s1": {"state": "running"}, "t#s2": {"state": "queued"}}}))
    assert D.LegRun.train_legs_waiting(run) == [holder]                  # a unit still waits: val holds
    st.write_text(json.dumps({"units": {"t#s1": {"state": "running"}, "t#s2": {"state": "graded"}}}))
    assert D.LegRun.train_legs_waiting(run) == []                        # all admitted: val takes the free seats
    st.write_text("{not json")
    assert D.LegRun.train_legs_waiting(run) == [holder]                  # unreadable: train first
