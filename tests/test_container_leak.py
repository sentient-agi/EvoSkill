"""Regression tests for the Docker sandbox container leak (2026-07-30).

The leak fed itself. `_cleanup_docker_container` discarded a container from the
process's own-containers set unconditionally, so any `docker rm` that failed or
hit its 15s timeout both orphaned the container AND dropped it from the atexit
safety net. Heavy load slowed the daemon, slow removals timed out, each timeout
added a permanent orphan, and the orphans slowed the daemon further: a 13-CPU
host went 203 -> 460 -> 653 -> 1023 containers and 8.8GB of 13.9GB VM memory,
which starved the SSB scorer past the liveness gate and produced 12 false
inactivity kills in a single midgate.

Separately, atexit does not fire on SIGTERM, so `kill <pid>` of a runner leaked
its whole in-flight set (measured: two killed judge legs left 545 containers).
"""

import asyncio
import signal

import src.harness.direct.executor as ex


class _FakeProc:
    def __init__(self, rc: int = 0, hang: bool = False):
        self._rc = rc
        self._hang = hang

    async def wait(self):
        if self._hang:
            await asyncio.sleep(3600)
        return self._rc

    def kill(self):
        pass


def _patch_rm(monkeypatch, **kw):
    async def _fake_exec(*args, **kwargs):
        return _FakeProc(**kw)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", _fake_exec)


def test_successful_removal_forgets_the_container(monkeypatch):
    _patch_rm(monkeypatch, rc=0)
    ex._own_containers.add("evoskill-sandbox-ok")
    asyncio.run(ex._cleanup_docker_container("evoskill-sandbox-ok"))
    assert "evoskill-sandbox-ok" not in ex._own_containers


def test_failed_removal_KEEPS_the_container_tracked(monkeypatch):
    """A nonzero `docker rm` must not drop the atexit safety net."""
    _patch_rm(monkeypatch, rc=1)
    ex._own_containers.add("evoskill-sandbox-fail")
    try:
        asyncio.run(ex._cleanup_docker_container("evoskill-sandbox-fail"))
        assert "evoskill-sandbox-fail" in ex._own_containers, (
            "failed removal silently orphaned the container"
        )
    finally:
        ex._own_containers.discard("evoskill-sandbox-fail")


def test_timed_out_removal_KEEPS_the_container_tracked(monkeypatch):
    """The load-driven case: `docker rm` exceeds its timeout under contention."""
    _patch_rm(monkeypatch, hang=True)
    monkeypatch.setattr(
        asyncio, "wait_for", _raise_timeout
    )
    ex._own_containers.add("evoskill-sandbox-slow")
    try:
        asyncio.run(ex._cleanup_docker_container("evoskill-sandbox-slow"))
        assert "evoskill-sandbox-slow" in ex._own_containers, (
            "timed-out removal silently orphaned the container"
        )
    finally:
        ex._own_containers.discard("evoskill-sandbox-slow")


async def _raise_timeout(*args, **kwargs):
    raise asyncio.TimeoutError


def test_sigterm_handler_is_installed():
    """atexit alone is not enough — SIGTERM must also trigger cleanup."""
    h = signal.getsignal(signal.SIGTERM)
    assert callable(h), f"SIGTERM is {h}, so a kill would leak every container"
    assert h not in (signal.SIG_DFL, signal.SIG_IGN)


def test_exit_cleanup_removes_every_tracked_container_in_batches(monkeypatch):
    """Names are passed many-per-call, so collect everything after `rm -f`."""
    removed = []
    calls = []

    class _SP:
        @staticmethod
        def run(cmd, **kwargs):
            assert cmd[:3] == ["docker", "rm", "-f"], cmd
            calls.append(len(cmd) - 3)
            removed.extend(cmd[3:])

            class _R:
                returncode = 0

            return _R()

    monkeypatch.setitem(__import__("sys").modules, "subprocess", _SP)
    ex._own_containers.update({"c-one", "c-two"})
    try:
        ex._cleanup_all_docker_containers()
        assert set(removed) >= {"c-one", "c-two"}
    finally:
        ex._own_containers.discard("c-one")
        ex._own_containers.discard("c-two")
