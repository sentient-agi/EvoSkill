"""Shared pytest configuration for the EvoSkill test suite."""

import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# Live-claude gate (L0). Nothing collected from tests/ may start the real `claude` CLI unless
# EVOSKILL_ALLOW_LIVE_CLAUDE=1 is set. The list and the reason live in tests/live_claude_gate.py.
#
# THIS FILE IS ONE LAYER, NOT THE GATE. `--noconftest`, or `--confcutdir` below tests/ (also via
# PYTEST_ADDOPTS), switches every hook here off at once. So every LIVE_FILES entry also carries a
# module-level require_live_claude() as its first statement, and every LIVE_ITEMS test checks the
# opt-in as its own first statement.
# ---------------------------------------------------------------------------

import importlib.machinery as _mach
import importlib.util as _ilu
import os as _os
import shutil as _shutil
import sys as _sys
import tempfile as _tempfile

_spec = _ilu.spec_from_file_location("evoskill_live_claude_gate",
                                     Path(__file__).with_name("live_claude_gate.py"))
live_gate = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(live_gate)


def pytest_ignore_collect(collection_path, config):
    """Directory walks (`pytest`, `pytest tests`) never import a
    live file. pytest does not consult this hook for paths named on the command line; those are
    caught by pytest_pycollect_makemodule below."""
    if not live_gate.allowed() and live_gate.is_live_file(collection_path):
        return True
    return None


class _LiveGated(pytest.File):
    """Stands in for a live module named explicitly on the command line. It never imports the
    module, it only reports the skip."""

    def collect(self):
        pytest.skip(f"{self.path.name} {live_gate.REASON}")


@pytest.hookimpl(tryfirst=True)
def pytest_pycollect_makemodule(module_path, parent):
    if not live_gate.allowed() and live_gate.is_live_file(module_path):
        return _LiveGated.from_parent(parent, path=module_path)
    return None


@pytest.hookimpl(wrapper=True)
def pytest_collect_file(file_path, parent):
    """`--doctest-modules` adds a DoctestModule for a named .py file, and it IMPORTS the module; it
    is not a pytest_pycollect_makemodule product, so the hook above never sees it. For a live
    file only the _LiveGated stand-in survives."""
    res = yield
    if live_gate.allowed() or not live_gate.is_live_file(file_path):
        return res
    return [c for c in (res or []) if isinstance(c, _LiveGated)]


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    """LIVE_ITEMS: skipped HERE, before any fixture, matched by the item's resolved file and its
    function name. Not a skip MARKER (`-p no:skipping` ignores markers) and not the nodeid string
    (`--rootdir` or `-c` elsewhere turns it into `::test_...`, which matched nothing)."""
    if live_gate.allowed():
        return
    if live_gate.is_live_test(item.path, getattr(item, "originalname", None) or item.name):
        pytest.skip(live_gate.REASON)


_SDK_TRANSPORT = "claude_agent_sdk._internal.transport.subprocess_cli"


def _sdk_to_stub(module, stub: str) -> None:
    cls = module.SubprocessCLITransport
    if "_evoskill_real_find_cli" not in cls.__dict__:
        cls._evoskill_real_find_cli = cls.__dict__.get("_find_cli")
    cls._find_cli = lambda self: stub


class _SdkStubFinder:
    """claude_agent_sdk starts its BUNDLED `claude` (site-packages/.../_bundled/claude) before it
    ever searches PATH, so the PATH stub does not cover it. This points the SDK's own CLI lookup at
    the stub, in THIS process, the moment its transport module is first imported (importing the
    SDK eagerly would cost every session about a second). An explicit ClaudeAgentOptions(cli_path=)
    and an SDK used in a child process are not covered."""

    def __init__(self, stub: str):
        self.stub = stub

    def find_spec(self, name, path=None, target=None):
        if name != _SDK_TRANSPORT:
            return None
        spec = _mach.PathFinder.find_spec(name, path, target)
        if spec is None or not isinstance(spec.loader, _mach.SourceFileLoader):
            return spec
        stub = self.stub

        class _Loader(_mach.SourceFileLoader):
            def exec_module(self, module):
                super().exec_module(module)
                _sdk_to_stub(module, stub)
        spec.loader = _Loader(spec.loader.name, spec.loader.path)
        return spec


def pytest_configure(config):
    """Second safety net, for code paths nobody listed: without the opt-in, a stub named
    `claude` goes first on PATH for this process and every child that inherits its env, and the
    Agent SDK's own CLI lookup resolves to it too (_SdkStubFinder). The stub refuses (exit 97)
    instead of starting a session. A child given an explicit PATH or an absolute path to the
    binary is not covered; that is what the file list above is for."""
    if live_gate.allowed():
        return
    stub_dir = _tempfile.mkdtemp(prefix="evoskill_noclaude_")
    stub = Path(stub_dir) / "claude"
    stub.write_text("#!/bin/sh\n"
                    "echo 'EVOSKILL: the real claude CLI is blocked under pytest "
                    f"(set {live_gate.LIVE_ENV}=1 to opt in)' >&2\n"
                    "exit 97\n")
    stub.chmod(0o755)
    config._evoskill_noclaude = (stub_dir, _os.environ.get("PATH"))
    _os.environ["PATH"] = stub_dir + _os.pathsep + _os.environ.get("PATH", "")
    finder = _SdkStubFinder(str(stub))
    config._evoskill_sdk_finder = finder
    _sys.meta_path.insert(0, finder)
    if _SDK_TRANSPORT in _sys.modules:                  # imported before this ran (a plugin)
        _sdk_to_stub(_sys.modules[_SDK_TRANSPORT], str(stub))


def pytest_unconfigure(config):
    finder = getattr(config, "_evoskill_sdk_finder", None)
    if finder is not None:
        if finder in _sys.meta_path:
            _sys.meta_path.remove(finder)
        mod = _sys.modules.get(_SDK_TRANSPORT)
        cls = getattr(mod, "SubprocessCLITransport", None)
        if cls is not None and "_evoskill_real_find_cli" in cls.__dict__:
            real = cls.__dict__["_evoskill_real_find_cli"]
            if real is not None:
                cls._find_cli = real
            else:
                del cls._find_cli
            del cls._evoskill_real_find_cli
    saved = getattr(config, "_evoskill_noclaude", None)
    if not saved:
        return
    stub_dir, old_path = saved
    if old_path is None:
        _os.environ.pop("PATH", None)
    else:
        _os.environ["PATH"] = old_path
    _shutil.rmtree(stub_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def _no_real_credential(monkeypatch):
    """No test may resolve a real Claude Code credential (src/orchestrator/cc_account.py): the token-file variable and
    the API key are removed for every test, so a test that forgets to declare its own stand-in fails instead of
    reading a real token file or billing a real key."""
    from src.orchestrator import cc_account
    monkeypatch.delenv(cc_account.TOKEN_FILE_ENV, raising=False)
    monkeypatch.delenv(cc_account.API_KEY_ENV, raising=False)
