"""Claude Code pins: explicit model ids, --effort, no model/refusal fallback, no auto-update,
subagents forced onto the model, and the HALT when a session reports another model.

NOTHING HERE STARTS `claude`. Every call site is exercised with subprocess.run monkeypatched to a
recorder that writes a synthetic stream-json log, and a refusing `claude` stub is first on PATH as
a second line of defence (its marker file must never appear).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.orchestrator import claude_pins as CP  # noqa: E402


def stream(model="claude-opus-5-5", *, version="2.1.281", assistant=None, sub=None, usage=None,
           result="done", init=True) -> str:
    ev = []
    if init:
        ev.append({"type": "system", "subtype": "init", "model": model,
                   "claude_code_version": version, "session_id": "s"})
    for m in (assistant if assistant is not None else [model, model]):
        ev.append({"type": "assistant", "parent_tool_use_id": None,
                   "message": {"model": m, "content": [{"type": "text", "text": "hi"}]}})
    for m in (sub or []):
        ev.append({"type": "assistant", "parent_tool_use_id": "toolu_1",
                   "message": {"model": m, "content": []}})
    ev.append({"type": "result", "subtype": "success", "result": result,
               "modelUsage": usage if usage is not None else {model: {"outputTokens": 5}}})
    return "launcher banner line\n" + "\n".join(json.dumps(e) for e in ev) + "\n"


@pytest.fixture(autouse=True)
def refusing_claude(tmp_path, monkeypatch):
    b = tmp_path / "stubbin"
    b.mkdir()
    marker = tmp_path / "CLAUDE_WAS_CALLED"
    (b / "claude").write_text(f"#!/bin/sh\ntouch {marker}\necho refused >&2\nexit 97\n")
    (b / "claude").chmod(0o755)
    monkeypatch.setenv("PATH", f"{b}:{os.environ.get('PATH', '')}")
    monkeypatch.delenv("EVOSKILL_CLAUDE_EFFORT", raising=False)
    monkeypatch.delenv("EVOSKILL_EXPECT_CLAUDE_CODE_VERSION", raising=False)
    monkeypatch.delenv("EVOSKILL_AO_MODEL", raising=False)
    yield
    assert not marker.exists(), "a test reached the real claude path"


# ══ the rules ═══════════════════════════════════════════════════════════════════════════════════
def test_effort_default_optout_and_refusal(monkeypatch):
    assert CP.effort({}) == "high"
    for off in ("", "none", "OFF", "default"):
        assert CP.effort({"EVOSKILL_CLAUDE_EFFORT": off}) is None
    assert CP.effort({"EVOSKILL_CLAUDE_EFFORT": "XHigh"}) == "xhigh"
    with pytest.raises(CP.ModelPinError):
        CP.effort({"EVOSKILL_CLAUDE_EFFORT": "turbo"})


def test_effort_flag_only_for_subscription_models():
    assert CP.effort_argv("claude-opus-5-5", {}) == ["--effort", "high"]
    assert CP.effort_argv("opus[1m]", {}) == ["--effort", "high"]
    assert CP.effort_argv("deepseek/deepseek-v4-flash-0731", {}) == []
    assert CP.effort_argv("claude-opus-5-5", {"EVOSKILL_CLAUDE_EFFORT": "none"}) == []


def test_expected_model_is_exact_family():
    assert CP.expected_model("claude-opus-5-5") == "claude-opus-5-5"
    assert CP.expected_model("claude-opus-5-5[1m]") == "claude-opus-5-5"
    assert CP.expected_model("opus[1m]") is None
    assert CP.expected_model("deepseek/deepseek-v4.1-flash") is None
    assert CP.model_matches("claude-opus-5-5", "claude-opus-5-5")
    assert CP.model_matches("claude-opus-5-5-20260915", "claude-opus-5-5")
    assert CP.model_matches("claude-opus-5-5[1m]", "claude-opus-5-5")
    assert not CP.model_matches("claude-opus-5", "claude-opus-5-5"), "the alias-flip target"
    assert not CP.model_matches("claude-opus-5-5", "claude-opus-5"), "prefix is not a match"


def test_pinned_env_sets_every_pin_and_forces_subagents():
    e = CP.pinned_env("claude-opus-5-5", {"PATH": "/x"})
    assert e["DISABLE_AUTOUPDATER"] == "1"
    assert e["CLAUDE_CODE_NO_MODEL_FALLBACK"] == "1"
    assert e["CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK"] == "1"
    assert e["CLAUDE_CODE_EFFORT_LEVEL"] == "high"
    assert e["CLAUDE_CODE_SUBAGENT_MODEL"] == "claude-opus-5-5"
    assert e["CLAUDE_CODE_SUBAGENT_MODEL_FORCE"] == "1"
    g = CP.pinned_env("deepseek/deepseek-v4-flash-0731", {"CLAUDE_CODE_EFFORT_LEVEL": "max"})
    assert "CLAUDE_CODE_SUBAGENT_MODEL" not in g
    assert g["CLAUDE_CODE_EFFORT_LEVEL"] == "max", "a gateway model's env is not rewritten"
    assert CP.pinned_env("opus[1m]", {}).get("CLAUDE_CODE_SUBAGENT_MODEL") is None


def test_shell_lines_eval_in_bash_3_2():
    txt = CP.shell_lines("claude-opus-5-5", {})
    r = subprocess.run(["/bin/bash", "-c", txt + 'echo "${EFFORT_ARGV[*]}|$DISABLE_AUTOUPDATER|'
                        '$CLAUDE_CODE_SUBAGENT_MODEL_FORCE|$CLAUDE_CODE_NO_MODEL_FALLBACK"'],
                       capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "--effort high|1|1|1"
    r = subprocess.run([sys.executable, "-m", "src.orchestrator.claude_pins", "--shell",
                        "--model", "claude-opus-5-5"], cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "EVOSKILL_CLAUDE_EFFORT": "bogus"})
    assert r.returncode == 4 and "REFUSING" in r.stderr


# ══ the model check ═════════════════════════════════════════════════════════════════════════════
def test_check_passes_on_an_opus_55_session(tmp_path):
    log = tmp_path / "e1.log"
    log.write_text(stream(usage={"claude-opus-5-5": {"outputTokens": 9},
                                 "claude-haiku-4-5-20251001": {"outputTokens": 17}}))
    c = CP.check_session(log, "claude-opus-5-5")
    assert c["ok"] and c["checked"]
    assert c["resolved_model"] == "claude-opus-5-5" and c["claude_code_version"] == "2.1.281"
    assert any("haiku" in n for n in c["notes"]), "background haiku is recorded, not asserted"


@pytest.mark.parametrize("kw,needle", [
    ({"model": "claude-opus-5"}, "system/init model 'claude-opus-5'"),
    ({"assistant": ["claude-opus-5-5", "claude-opus-5"]}, "assistant message(s) from 'claude-opus-5'"),
    ({"sub": ["claude-haiku-4-5"]}, "SUBAGENT"),
    ({"usage": {"claude-opus-5-5": {}, "claude-opus-5": {}}}, "modelUsage"),
])
def test_check_fails_on_any_other_model(tmp_path, kw, needle):
    c = CP.check_session(stream(**kw), "claude-opus-5-5")
    assert not c["ok"] and any(needle in v for v in c["violations"]), c["violations"]


def test_check_fails_when_the_session_reported_nothing_and_on_a_version_change(tmp_path):
    c = CP.check_session("not json at all\n", "claude-opus-5-5")
    assert not c["ok"] and "NO model" in c["violations"][0]
    c = CP.check_session(stream(version="2.1.282"), "claude-opus-5-5", expect_cc_version="2.1.281")
    assert not c["ok"] and "claude_code_version" in c["violations"][0]


def test_alias_requests_are_recorded_not_asserted():
    c = CP.check_session(stream(model="claude-opus-5"), "opus[1m]")
    assert c["ok"] and not c["checked"] and c["resolved_model"] == "claude-opus-5"


def test_result_text_reads_stream_and_legacy_json():
    assert CP.result_text(stream(result='{"a": 1}')) == '{"a": 1}'
    assert CP.result_text(json.dumps({"type": "result", "result": "X"})) == "X"


# ══ the call sites ══════════════════════════════════════════════════════════════════════════════
def _author(monkeypatch, tmp_path, out_stream):
    from src.orchestrator import prompt_author as PA
    rd = tmp_path / "run"
    (rd / "digests_e1_train").mkdir(parents=True)
    (rd / "oracle").mkdir()
    prompt = rd / "ITER1_PROMPT.md"

    def fake_run(argv, **kw):
        kw["stdout"].write(out_stream)
        kw["stdout"].flush()
        prompt.write_text("x" * 2000)
        fake_run.argv, fake_run.env = list(argv), dict(kw.get("env") or {})
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(PA, "_ao_sandbox", lambda *a, **k: ([], None))
    # author_prompt imports subprocess locally, so the module attribute is what it resolves
    monkeypatch.setattr(subprocess, "run", fake_run)
    pol = ROOT / "src/orchestrator/PROMPT_POLICY.md"
    return PA, rd, prompt, pol, fake_run


def test_prompt_author_sends_pins_and_records_the_resolved_model(monkeypatch, tmp_path):
    PA, rd, prompt, pol, fake_run = _author(monkeypatch, tmp_path, stream())
    monkeypatch.setattr(PA, "_author_user_prompt", lambda *a, **k: "MSG")
    info = PA.author_prompt(facts=_FakeFacts(), out=prompt, epoch=1, digests=rd / "digests_e1_train",
                            oracle=rd / "oracle", log=rd / "e1_prompt_author.log",
                            staging=rd / "skills_v1_staging", model="claude-opus-5-5", policy=pol)
    assert fake_run.argv[fake_run.argv.index("--effort") + 1] == "high"
    assert fake_run.argv[fake_run.argv.index("--model") + 1] == "claude-opus-5-5"
    assert fake_run.env["CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK"] == "1"
    assert info["resolved_model"] == "claude-opus-5-5" and info["claude_code_version"] == "2.1.281"
    assert info["model_check"]["ok"] and info["pins"]["effort_argv"] == ["--effort", "high"]


class _FakeFacts:
    def as_block(self):
        return "FACTS"


# ══ the loop HALTs ══════════════════════════════════════════════════════════════════════════════
def _ctx(tmp_path, evolver_model="claude-opus-5-5"):
    from src.orchestrator import loop as L

    class Spec:
        benchmark, domain, variant, name = "ssbv2", "spreadsheetbench-v2", "", "t"
        evolver_harness = "claude_cli"
        prompt_author_model = None
        solver_provider_order = ("deepinfra",)
        solver_allow_fallbacks = False
    Spec.evolver_model = evolver_model
    rd = tmp_path / "run"
    rd.mkdir(exist_ok=True)
    return L, L.Ctx(rd, 1, Spec(), rd / "oracle")


def test_evolver_model_step_passes_fails_and_halts(tmp_path):
    L, ctx = _ctx(tmp_path)
    names = [s.name for s in L.epoch_steps()]
    i = names.index("evolver ran on the declared model")
    assert names[i - 1] == "skills load through the real consumer"
    assert i < names.index("promoted set matches staging")
    assert L.epoch_steps()[i].run is None
    log = ctx.run_dir / "tau3_evolver_e1.log"
    ok, d = L._v_evolver_model(ctx)
    assert not ok and "no evolver log" in d
    log.write_text(stream())
    ok, d = L._v_evolver_model(ctx)
    assert ok and "claude-opus-5-5" in d
    rec = L._record_evolver_model(ctx, halt=True)
    assert rec["ok"] and (ctx.run_dir / "e1_EVOLVER_MODEL.json").is_file()
    log.write_text(stream(assistant=["claude-opus-5-5", "claude-opus-5"]))
    ok, d = L._v_evolver_model(ctx)
    assert not ok and "claude-opus-5" in d
    with pytest.raises(L.Halt, match="did NOT run on claude-opus-5-5"):
        L._record_evolver_model(ctx, halt=True)
    L2, alias = _ctx(tmp_path, "opus[1m]")
    ok, d = L2._v_evolver_model(alias)
    assert ok and d.startswith(L2.UNMEASURED), "an alias is UNMEASURED, never passed silently"


def test_prompt_authored_verify_rederives_the_model_from_the_log(tmp_path):
    L, ctx = _ctx(tmp_path)
    log = ctx.run_dir / "e1_prompt_author.log"
    log.write_text("=== prompt author\n" + stream())
    ctx.paths.prompt.write_text("x" * 2000)
    rec = {"model": "claude-opus-5-5", "seconds": 1, "bytes": 2000, "policy": "P.md",
           "log": str(log), "session_mode": "single-shot", "model_check": {"ok": True},
           "pins": {"effort_argv": ["--effort", "high"]}}
    (ctx.run_dir / "e1_PROMPT_AUTHOR.json").write_text(json.dumps(rec))
    ok, d = L._v_prompt_authored(ctx)
    assert ok and "session ran on claude-opus-5-5" in d
    log.write_text("=== prompt author\n" + stream(model="claude-opus-5"))
    ok, d = L._v_prompt_authored(ctx)
    assert not ok and "did not run on claude-opus-5-5" in d
    # a record from before the check (no model_check / pins) is judged exactly as before
    rec2 = {k: v for k, v in rec.items() if k not in ("model_check", "pins")}
    (ctx.run_dir / "e1_PROMPT_AUTHOR.json").write_text(json.dumps(rec2))
    ok, d = L._v_prompt_authored(ctx)
    assert ok


def test_run_author_prompt_halts_after_recording_a_mismatch(tmp_path, monkeypatch):
    L, ctx = _ctx(tmp_path)
    from src.orchestrator import prompt_author as PA
    monkeypatch.setattr(L, "_prompt_policy", lambda c: None)
    monkeypatch.setattr(L, "_evidence_mode", lambda c: "current_epoch")
    monkeypatch.setattr(PA, "derive_facts", lambda *a, **k: _FakeFacts())
    monkeypatch.setattr(PA, "author_prompt", lambda **k: {
        "model": "claude-opus-5-5", "model_check": {"checked": True, "ok": False,
                                                    "expected_model": "claude-opus-5-5",
                                                    "violations": ["init model 'claude-opus-5'"]}})
    with pytest.raises(L.Halt, match="did NOT run on claude-opus-5-5"):
        L._run_author_prompt(ctx)
    assert (ctx.run_dir / "e1_PROMPT_AUTHOR.json").is_file(), "recorded BEFORE the HALT"
