"""src/orchestrator/stateful_evolver.py's renderer and layout steps: the two
fixed templates under src/orchestrator/prompts/ rendered with four placeholders (render_system_prompt,
render_epoch_message), the render step + its verify (run_render_session_prompt, v_session_prompt_rendered), and the
evolver's material built from a leg's store (run_build_evo_layout, v_evo_layout over evo_layout).

Fixture: a tiny run dir under tmp_path -- a RUNTIME.json satisfying validate_session's nine rules, a split, two fake
Claude Code leg stores (the base leg = epoch 0, skills_v1's train leg = epoch 1; 2 tasks x 2 samples each) at the
loop's leg names. No network, no model call, no write outside tmp_path; nothing under .cache/evolver_state.
"""
from __future__ import annotations

import calendar
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.orchestrator.loop as L  # noqa: E402
from src.orchestrator import claude_pins  # noqa: E402
from src.orchestrator import evo_layout as EL  # noqa: E402
from src.orchestrator import stateful_evolver as SE  # noqa: E402
from src.orchestrator import stateful_layout as SL  # noqa: E402

TASKS = ("Cat/01_01", "Cat/01_02")
N_SAMPLES = 2
# STATE carries the run's grade(task, snapshot) per unit -- a number from 0 to 1, or
# "grader_failed" for a rollout whose grade() raised twice
SCORES = {("Cat/01_01", 1): 1.0, ("Cat/01_01", 2): 0.0,
          ("Cat/01_02", 1): 0.5, ("Cat/01_02", 2): "grader_failed"}


def _store(tmp: Path, name: str, leg_json: Path) -> Path:
    """A Claude Code leg's store as scripts/daytona/cc_leg_driver.py writes it, and the leg JSON naming it."""
    store = tmp / "stores" / name
    units = {}
    for t in TASKS:
        for k in range(1, N_SAMPLES + 1):
            units[f"{t}#s{k}"] = {"state": "graded", "score": SCORES[(t, k)]}
            rd = store / "runs" / f"{EL.safe_id(t)}__s{k}"
            work = rd.joinpath(*EL.CONTAINER_WORK)
            (work / "out").mkdir(parents=True)
            (rd / "stream.jsonl").write_text(f'{{"type":"system","subtype":"init","leg":"{name}","task":"{t}","sample":{k}}}\n'
                                             f'{{"type":"result","leg":"{name}"}}\n')
            # Claude Code's own session transcript (the trace; it carries every user message)
            proj = rd.joinpath(*EL.CC_PROJECTS) / "-evo-work"
            proj.mkdir(parents=True)
            sid = f"sess-{name}-{EL.safe_id(t)}-{k}"
            (proj / f"{sid}.jsonl").write_text(f'{{"type":"user","message":{{"role":"user","content":"solve {t}"}},"leg":"{name}"}}\n'
                                               f'{{"type":"assistant","leg":"{name}","task":"{t}","sample":{k}}}\n')
            if (t, k) == ("Cat/01_02", 2):                                  # one session saved a large tool output
                (proj / sid / "tool-results").mkdir(parents=True)
                (proj / sid / "tool-results" / "big.txt").write_text(f"big output {name}\n")
            (work / "task.md").write_text(f"# {t} sample {k} ({name})\n")
            (work / "out" / "answer.xlsx").write_bytes(f"XLSX {name} {t} {k}".encode())
    (store / "STATE.json").write_text(json.dumps({"units": units}, indent=1))
    leg_json.write_text(json.dumps({"task_ids": list(TASKS), "n_samples": N_SAMPLES, "daytona": {"store": str(store)}}))
    return store


def _runtime(r, **over) -> dict:
    rt = {"daytona_leg": {"pool": True, "require_snapshot_ref": "daytona-abc:daytona", "pool_ttl_minutes": 1440},
          "claude": {"effort": "default"}, "evolver_session": "stateful", "prompt_mode": "template",
          "evolver_host": "daytona", "evolver_evidence": "all_epochs", "evolver_evidence_layout": "evo",
          "evolver_daytona": {"placement": "pool_container"}, "base_leg": str(r.base)}
    rt.update(over)
    return rt


def _set_runtime(r, **over) -> None:
    (r.rd / "RUNTIME.json").write_text(json.dumps(_runtime(r, **over)))


@pytest.fixture
def run(tmp_path, monkeypatch):
    rd = tmp_path / "outputs" / "runX"
    rd.mkdir(parents=True)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": list(TASKS), "val": ["Cat/02_01"], "test": ["Cat/03_01"]}))
    base = rd / "e0_base_train.json"
    _store(tmp_path, "base", base)
    _store(tmp_path, "v1", L.EpochPaths(rd, 1).leg("train"))
    r = SimpleNamespace(rd=rd, split=split, base=base)
    _set_runtime(r)
    monkeypatch.setenv(claude_pins.EFFORT_ENV, "default")
    return r


def sctx(r, epoch, n_samples=N_SAMPLES, **kw):
    spec = SimpleNamespace(benchmark="ssbv2", split_path=str(r.split), n_samples=n_samples, name=r.rd.name,
                           evolver_harness="claude_cli", evolver_model="claude-opus-5-5")
    base = dict(base_leg=str(r.base), evolver_evidence="all_epochs", evidence_layout="evo", evolver_session="stateful",
                prompt_mode="template", evolver_host="daytona", evolver_daytona={"placement": "pool_container"},
                leg_pool=True)
    base.update(kw)
    return L.Ctx(r.rd, epoch, spec, r.rd / "oracle", **base)


def _render(r, *epochs):
    """The render step for each epoch in order; returns the last epoch's ctx."""
    ctx = None
    for e in epochs:
        ctx = sctx(r, e)
        SE.run_render_session_prompt(ctx)
    return ctx


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _record(r, epoch) -> dict:
    return json.loads((r.rd / L.RENDER_RECORD.format(epoch=epoch)).read_text())


# ══ 1. the two templates: every placeholder filled, nothing else computed ═══════════════════════════════════
def test_render_system_prompt_fills_n_samples_and_names_the_container_layout():
    sp = SE.render_system_prompt(n_samples=3)
    text = sp["text"]
    assert not re.findall(r"\{[A-Za-z_][A-Za-z_0-9-]*\}", text)          # no placeholder left
    assert "runs every training task 3 times" in text
    assert SL.forbidden_mac_strings(text) == [] and "/Users" not in text
    # the fixed layout the evolver is told about is stateful_layout's, root for root
    for root in (SL.VM_TRACES, SL.VM_SCORES, SL.VM_SOLVER_WS, SL.VM_SKILLS, SL.VM_STAGING, SL.VM_WORKSPACE, SL.VM_REFERENCE):
        assert f"`{root}/" in text, root
    assert "/evo/traces/epoch_K/" in text and "/evo/scores/epoch_K.json" in text and "This session lasts the whole run" in text
    assert "`<skill-name>/SKILL.md`" in text and "no network beyond the model API" in text
    for gone in ("/evo/grader", "/evo/references", "/evo/inputs", "/evo/notes", "/evo/work", "MESSAGE_e", "INDEX.md",
                 "```scores", "digest"):
        assert gone not in text, gone
    assert sp == {"text": text, "sha256": hashlib.sha256(text.encode()).hexdigest(), "bytes": len(text.encode()),
                  "n_samples": 3, "template": str(SE.SYSTEM_TEMPLATE), "template_sha256": _sha(SE.SYSTEM_TEMPLATE)}
    assert SE.render_system_prompt(n_samples=3) == sp                   # pure
    assert "runs every training task 1 times" in SE.render_system_prompt(n_samples=1)["text"]
    for bad in (0, -1, True, "3", None):
        with pytest.raises(ValueError):
            SE.render_system_prompt(n_samples=bad)


def test_render_epoch_message_fills_the_times_n_and_n_minus_1():
    msg = SE.render_epoch_message(1, utc_now="2026-09-28 10:00", deadline_utc="2026-09-28 11:30")
    text = msg["text"]
    assert text.startswith("It is 2026-09-28 10:00 UTC; you have until 2026-09-28 11:30 UTC for this epoch")
    assert "You are at epoch 1." in text and "Please analyze the solver rollouts, diagnose" in text and "newest" not in text and "compose `skills_v1` under `/evo/staging/`" in text
    assert not re.findall(r"\{[A-Za-z_][A-Za-z_0-9-]*\}", text) and SL.forbidden_mac_strings(text) == []
    assert msg == {"text": text, "sha256": hashlib.sha256(text.encode()).hexdigest(), "bytes": len(text.encode()),
                   "epoch": 1, "utc_now": "2026-09-28 10:00", "deadline_utc": "2026-09-28 11:30",
                   "template": str(SE.MESSAGE_TEMPLATE), "template_sha256": _sha(SE.MESSAGE_TEMPLATE)}
    t3 = SE.render_epoch_message(3, utc_now="2026-09-28 10:00", deadline_utc="2026-09-28 11:30")["text"]
    assert "You are at epoch 3." in t3 and "Please analyze the solver rollouts, diagnose" in t3 and "`skills_v3`" in t3
    assert t3.replace("epoch 3", "epoch 1").replace("epoch 2", "epoch 0").replace("skills_v3", "skills_v1") == text
    for bad in ("2026-09-28T10:00", "10:00", "", None, "2026-09-28 10:00:00"):
        with pytest.raises(SE.RenderError, match="utc_now"):
            SE.render_epoch_message(1, utc_now=bad, deadline_utc="2026-09-28 11:30")
    with pytest.raises(SE.RenderError, match="deadline_utc"):
        SE.render_epoch_message(1, utc_now="2026-09-28 10:00", deadline_utc="soon")
    for bad in (0, True, "1"):
        with pytest.raises(ValueError):
            SE.render_epoch_message(bad, utc_now="2026-09-28 10:00", deadline_utc="2026-09-28 11:30")
    assert time.strptime(SE.utc_str(0), SE.UTC_FMT) and SE.utc_str(0) == "1970-01-01 00:00"


# ══ 2. the render step: the message, the system prompt once, the record ════════════════════════════════════
def test_render_step_writes_the_message_the_system_prompt_once_and_a_record_that_round_trips(run):
    r = run
    ctx1 = _render(r, 1)
    p1, sysp = ctx1.paths.prompt, r.rd / SL.SYSTEM_PROMPT_FILE
    assert p1.name == "ITER1_PROMPT.md" and p1.is_file() and sysp.is_file()
    rec1 = _record(r, 1)
    assert rec1["mode"] == "template" and rec1["session"] == "stateful" and rec1["model_call"] is False and rec1["epoch"] == 1
    assert rec1["prompt"]["path"] == str(p1) and rec1["prompt"]["sha256"] == _sha(p1) and rec1["prompt"]["bytes"] == p1.stat().st_size
    assert rec1["system_prompt"]["path"] == str(sysp) and rec1["system_prompt"]["sha256"] == _sha(sysp)
    assert rec1["system_prompt"]["rendered_epoch"] == 1 and rec1["n_samples"] == N_SAMPLES and rec1["evolver_timeout_s"] == 5400
    assert rec1["prompt"]["template_sha256"] == _sha(SE.MESSAGE_TEMPLATE) and rec1["system_prompt"]["template_sha256"] == _sha(SE.SYSTEM_TEMPLATE)
    assert rec1["renderer"]["sha256"] == _sha(Path(SE.__file__)) and rec1["record"] == "e1_PROMPT_RENDER.json"
    for gone in ("score_table", "facts", "grants", "path_view", "evidence", "prior_library", "trace_stats"):
        assert gone not in rec1, gone
    # the message carries the send time and the deadline = send time + RUNTIME evolver_timeout_s (recorded)
    now, dl = rec1["prompt"]["utc_now"], rec1["prompt"]["deadline_utc"]
    secs = [calendar.timegm(time.strptime(x, SE.UTC_FMT)) for x in (now, dl)]
    assert secs[1] - secs[0] == 5400 and abs(secs[0] - time.time()) < 120
    assert p1.read_text() == SE.render_epoch_message(1, utc_now=now, deadline_utc=dl)["text"]
    assert sysp.read_text() == SE.render_system_prompt(n_samples=N_SAMPLES)["text"]
    assert f"It is {now} UTC; you have until {dl} UTC" in p1.read_text()
    # epoch 2: a new message, the SAME system prompt bytes, the record naming the epoch-1 render
    sha = _sha(sysp)
    ctx2 = _render(r, 2)
    assert ctx2.paths.prompt.name == "ITER2_PROMPT.md" and "You are at epoch 2." in ctx2.paths.prompt.read_text()
    assert _sha(sysp) == sha
    rec2 = _record(r, 2)
    assert rec2["system_prompt"]["sha256"] == sha and rec2["system_prompt"]["rendered_epoch"] == 1 and rec2["epoch"] == 2
    assert {f.name for f in r.rd.iterdir() if f.name.startswith(("ITER", "e1_", "e2_", "EVOLVER"))} == \
        {"ITER1_PROMPT.md", "ITER2_PROMPT.md", "e1_PROMPT_RENDER.json", "e2_PROMPT_RENDER.json", "EVOLVER_SYSTEM.md"}


def test_render_step_refuses_overwrites_and_never_rewrites_the_system_prompt(run):
    r = run
    _render(r, 1)
    with pytest.raises(L.Halt, match="refusing to render over ITER1_PROMPT.md"):
        SE.run_render_session_prompt(sctx(r, 1))
    sysp = r.rd / SL.SYSTEM_PROMPT_FILE
    sha = _sha(sysp)
    # epoch 1 re-run after the message was moved aside: an identical system prompt is kept
    (r.rd / "ITER1_PROMPT.md").rename(r.rd / "ITER1_PROMPT.md.aside")
    (r.rd / "e1_PROMPT_RENDER.json").rename(r.rd / "e1_PROMPT_RENDER.json.aside")
    SE.run_render_session_prompt(sctx(r, 1))
    assert _sha(sysp) == sha
    # ... but not one with other bytes
    (r.rd / "ITER1_PROMPT.md").unlink()
    (r.rd / "e1_PROMPT_RENDER.json").unlink()
    sysp.write_text(sysp.read_text() + "\nedited\n")
    with pytest.raises(L.Halt, match="exists with other bytes"):
        SE.run_render_session_prompt(sctx(r, 1))
    sysp.unlink()
    SE.run_render_session_prompt(sctx(r, 1))
    assert _sha(sysp) == sha
    # an AO record for this epoch is evidence too
    (r.rd / "ITER1_PROMPT.md").unlink()
    (r.rd / "e1_PROMPT_RENDER.json").unlink()
    (r.rd / "e1_PROMPT_AUTHOR.json").write_text("{}")
    with pytest.raises(L.Halt, match="refusing to render over e1_PROMPT_AUTHOR.json"):
        SE.run_render_session_prompt(sctx(r, 1))
    (r.rd / "e1_PROMPT_AUTHOR.json").unlink()
    SE.run_render_session_prompt(sctx(r, 1))
    # epoch 2 needs the epoch-1 system prompt and does not rewrite it
    _render(r, 2)
    assert _sha(sysp) == sha
    (r.rd / "ITER2_PROMPT.md").unlink()
    (r.rd / "e2_PROMPT_RENDER.json").unlink()
    sysp.write_text(sysp.read_text() + "\nedited\n")
    with pytest.raises(L.Halt, match="edited since epoch 1"):
        SE.run_render_session_prompt(sctx(r, 2))
    sysp.unlink()
    with pytest.raises(L.Halt, match="needs EVOLVER_SYSTEM.md"):
        SE.run_render_session_prompt(sctx(r, 2))
    sysp.write_text(SE.render_system_prompt(n_samples=N_SAMPLES)["text"])
    # a SPEC n_samples that changed since epoch 1: the session's system prompt is fixed for the run
    with pytest.raises(L.Halt, match="no longer re-renders"):
        SE.run_render_session_prompt(sctx(r, 2, n_samples=3))
    # a bad RUNTIME evolver_timeout_s, and a per_epoch run, never reach the renderer
    with pytest.raises(L.Halt, match="evolver_timeout_s"):
        SE.run_render_session_prompt(sctx(r, 2, evolver_timeout_s=0))
    _set_runtime(r, evolver_session="per_epoch")
    with pytest.raises(L.Halt, match="evolver_session='per_epoch'"):
        SE.run_render_session_prompt(sctx(r, 2, evolver_session="per_epoch"))


# ══ 3. the verify: pass, and fail on every tampering ═══════════════════════════════════════════════════════
def test_v_session_prompt_rendered_passes_and_fails_on_tampering(run):
    r = run
    ctx1 = _render(r, 1)
    ok, det = SE.v_session_prompt_rendered(ctx1)
    assert ok and "re-render byte-identically" in det and "no model call" in det, det
    ctx = _render(r, 2)
    ok, det = SE.v_session_prompt_rendered(ctx)
    assert ok and "epoch-1 render" in det, det
    p, sysp, rec_p = ctx.paths.prompt, r.rd / SL.SYSTEM_PROMPT_FILE, r.rd / "e2_PROMPT_RENDER.json"
    orig, sorig, rorig = p.read_text(), sysp.read_text(), rec_p.read_text()

    def fails(needle):
        ok, det = SE.v_session_prompt_rendered(ctx)
        assert not ok and needle in det, (needle, det)

    def restore():
        p.write_text(orig)
        sysp.write_text(sorig)
        rec_p.write_text(rorig)
        assert SE.v_session_prompt_rendered(ctx)[0]
    p.write_text(orig.replace("epoch 2", "epoch 3"))                      # the message edited after the render
    fails("edited after the render")
    rec = json.loads(rorig)                                                # ... even with the record's sha "fixed":
    rec["prompt"]["sha256"] = _sha(p)                                      # the re-render from the recorded times differs
    rec_p.write_text(json.dumps(rec))
    fails("re-rendering the message now")
    restore()
    sysp.write_text(sorig + "\n")                                          # the system prompt edited
    fails("edited after the epoch-1 render")
    restore()
    rec = json.loads(rorig)
    rec["prompt"]["utc_now"] = "yesterday"                                 # the record no longer re-renders
    rec_p.write_text(json.dumps(rec))
    fails("no longer re-render")
    rec = json.loads(rorig)
    rec["system_prompt"]["rendered_epoch"] = 2
    rec_p.write_text(json.dumps(rec))
    fails("rendered_epoch 2")
    rec_p.write_text(json.dumps({"mode": "ao"}))
    fails("not a stateful template render record")
    rec_p.write_text("{nope")
    fails("cannot attribute")
    rec_p.unlink()
    fails("cannot attribute")
    restore()
    # the epoch-1 record attributes the system prompt; SPEC n_samples must still give the same system prompt
    e1 = r.rd / "e1_PROMPT_RENDER.json"
    e1.rename(r.rd / "e1.aside")
    fails("epoch-1 render record is needed")
    (r.rd / "e1.aside").rename(e1)
    rec1 = json.loads(e1.read_text())
    rec1["system_prompt"]["sha256"] = "0" * 64
    e1.write_text(json.dumps(rec1))
    fails("not the epoch-1 render")
    e1.write_text(json.dumps(json.loads(e1.read_text()) | {"system_prompt": json.loads(rorig)["system_prompt"]}))
    assert SE.v_session_prompt_rendered(ctx)[0]
    ok, det = SE.v_session_prompt_rendered(sctx(r, 2, n_samples=3))
    assert not ok and "re-rendering the system prompt now gives" in det and "n_samples" in det, det
    sysp.unlink()
    fails("no system prompt")
    restore()
    (r.rd / "e2_PROMPT_AUTHOR.json").write_text("{}")
    fails("authored by the AO")
    (r.rd / "e2_PROMPT_AUTHOR.json").unlink()
    p.unlink()
    fails("has not rendered it yet")
    restore()


# ══ 4. the evolver's material: evo_e{K} from the leg stores ════════════════════════════════════════════════
def test_run_build_evo_layout_builds_each_epoch_from_its_leg_store_and_v_evo_layout_rederives(run, capsys):
    r = run
    ctx1 = sctx(r, 1)
    assert ctx1.paths.evo == r.rd / "evo_e0" and ctx1.paths.evo_dirs() == [(r.rd / "evo_e0").resolve()]
    SE.run_build_evo_layout(ctx1)
    d0 = r.rd / "evo_e0"
    assert "[evo layout] evo_e0: 4 rollouts" in capsys.readouterr().out
    assert sorted(p.name for p in (d0 / "traces").iterdir()) == ["Cat_01_01_s1.jsonl", "Cat_01_01_s2.jsonl",
                                                                 "Cat_01_02_s1.jsonl", "Cat_01_02_s2",
                                                                 "Cat_01_02_s2.jsonl"]
    store = Path(json.loads(r.base.read_text())["daytona"]["store"])
    log, comp = EL.transcript_of(store / "runs/Cat_01_02__s2")
    assert (d0 / "traces/Cat_01_02_s2.jsonl").read_bytes() == log.read_bytes()             # the session transcript
    assert b'"type":"user"' in (d0 / "traces/Cat_01_02_s2.jsonl").read_bytes()              # every user message
    assert (d0 / "traces/Cat_01_02_s2/tool-results/big.txt").read_text() == "big output base\n"   # its saved outputs
    assert not (d0 / "traces/Cat_01_01_s1").exists()                                       # none when none were saved
    assert (d0 / "solver_workspaces/Cat_01_02_s2/out/answer.xlsx").read_bytes() == b"XLSX base Cat/01_02 2"
    assert (d0 / "solver_workspaces/Cat_01_01_s1/task.md").is_file()
    # the evolver sees a score from 0 to 1 or grader_failed only: the STATE score verbatim (the run's grade(task, snapshot)), a fraction and grader_failed included, no verdict mapped
    assert json.loads((d0 / "scores.json").read_text()) == [
        {"task_id": "Cat_01_01", "sample": 1, "score": 1.0}, {"task_id": "Cat_01_01", "sample": 2, "score": 0.0},
        {"task_id": "Cat_01_02", "sample": 1, "score": 0.5}, {"task_id": "Cat_01_02", "sample": 2, "score": "grader_failed"}]
    assert sorted(p.name for p in d0.iterdir()) == ["scores.json", "solver_workspaces", "traces"]   # nothing else
    ok, det = SE.v_evo_layout(ctx1)
    assert ok and det.startswith("1 epoch dir(s) evo_e0..evo_e0") and "4 rollouts" in det and "nothing computed" in det, det
    SE.run_build_evo_layout(ctx1)                                          # an existing dir is left alone
    assert EL.check(d0, leg_json=r.base) == [] and EL.summary(d0) == {"rollouts": 4, "scored": 3, "grader_failed": 1,
                                                                        "mean_score": 0.5}
    # epoch 2: evo_e1 from skills_v1's train leg; the verify covers evo_e0 and evo_e1
    ctx2 = sctx(r, 2)
    SE.run_build_evo_layout(ctx2)
    d1 = r.rd / "evo_e1"
    assert b'"leg":"v1"' in (d1 / "traces/Cat_01_01_s1.jsonl").read_bytes()
    ok, det = SE.v_evo_layout(ctx2)
    assert ok and det.startswith("2 epoch dir(s) evo_e0..evo_e1") and "8 rollouts" in det, det
    # tampering with either dir fails the verify, naming the dir and the remedy
    tr = d1 / "traces/Cat_01_01_s1.jsonl"
    keep = tr.read_bytes()
    tr.write_bytes(keep + b"\n{}")
    ok, det = SE.v_evo_layout(ctx2)
    assert not ok and det.startswith("evo_e1: Cat_01_01_s1: trace differs") and "REMEDY: move the bad dir OUT" in det, det
    tr.write_bytes(keep)
    (d0 / "traces/extra.jsonl").write_text("{}\n")
    ok, det = SE.v_evo_layout(ctx2)
    assert not ok and "evo_e0: 1 file(s) in traces/ that are no measured rollout" in det, det
    (d0 / "traces/extra.jsonl").unlink()
    big = d1 / "traces/Cat_01_02_s2/tool-results/big.txt"                 # a saved tool output edited
    keep_big = big.read_text()
    big.write_text("edited\n")
    ok, det = SE.v_evo_layout(ctx2)
    assert not ok and "Cat_01_02_s2: the trace's companion files differ from the session's" in det, det
    big.write_text(keep_big)
    (d1 / "solver_workspaces/Cat_01_02_s2").rename(d1 / "ws.aside")
    ok, det = SE.v_evo_layout(ctx2)
    assert not ok and "Cat_01_02_s2: solver workspace missing" in det, det
    (d1 / "ws.aside").rename(d1 / "solver_workspaces/Cat_01_02_s2")
    sc = d1 / "scores.json"
    keep = sc.read_text()
    assert '"score": 0.0' in keep
    sc.write_text(keep.replace('"score": 0.0', '"score": 1.0'))
    ok, det = SE.v_evo_layout(ctx2)
    assert not ok and "scores.json differs from the store's scores" in det, det
    sc.write_text(keep)
    st1 = Path(json.loads(L.EpochPaths(r.rd, 1).leg("train").read_text())["daytona"]["store"]) / "STATE.json"
    keep_st = st1.read_text()                                              # the store's score moved after the build
    st = json.loads(keep_st)
    st["units"]["Cat/01_01#s1"]["score"] = 0.25
    st1.write_text(json.dumps(st))
    ok, det = SE.v_evo_layout(ctx2)
    assert not ok and det.startswith("evo_e1: scores.json differs from the store's scores"), det
    st1.write_text(keep_st)
    (d0 / "link").symlink_to(d1 / "scores.json")
    ok, det = SE.v_evo_layout(ctx2)
    assert not ok and "symlink in the layout: link" in det, det
    (d0 / "link").unlink()
    assert SE.v_evo_layout(ctx2)[0]
    # an earlier epoch's dir absent / a leg not on disk
    d0.rename(r.rd / "evo_e0.aside")
    ok, det = SE.v_evo_layout(ctx2)
    assert not ok and "evo_e0: absent" in det, det
    (r.rd / "evo_e0.aside").rename(d0)
    leg1 = L.EpochPaths(r.rd, 1).leg("train")
    leg1.rename(r.rd / "leg.aside")
    ok, det = SE.v_evo_layout(ctx2)
    assert not ok and "evo_e1: its leg" in det and "not on disk" in det, det
    (r.rd / "leg.aside").rename(leg1)
    assert SE.v_evo_layout(ctx2)[0]


def test_run_build_evo_layout_halts_on_a_missing_leg_an_ungraded_unit_and_refuses_to_rebuild(run, tmp_path):
    r = run
    with pytest.raises(L.Halt, match="not on disk"):
        SE.run_build_evo_layout(sctx(r, 1, base_leg=""))
    with pytest.raises(L.Halt, match="not on disk"):
        SE.run_build_evo_layout(sctx(r, 3))                                # skills_v2's train leg does not exist
    leg1 = L.EpochPaths(r.rd, 1).leg("train")
    d = json.loads(leg1.read_text())
    st = Path(d["daytona"]["store"]) / "STATE.json"
    units = json.loads(st.read_text())
    units["units"]["Cat/01_02#s2"]["state"] = "running"
    st.write_text(json.dumps(units))
    with pytest.raises(L.Halt, match="evo_e1 could not be built: Cat/01_02#s2: not graded"):
        SE.run_build_evo_layout(sctx(r, 2))
    assert not (r.rd / "evo_e1").exists() and not list(r.rd.glob("evo_e1.building.*"))   # nothing half-built left
    # A graded unit carries a STATE score (0..1 or grader_failed); a verdict-era row, a
    # number outside 0..1, a bool or any other string is refused, naming the unit, and nothing is left half-built
    for bad, shown in (({"verdict": "passed"}, "None"), ({"score": 1.5}, "1.5"), ({"score": -0.25}, "-0.25"),
                       ({"score": float("nan")}, "nan"), ({"score": "maybe"}, "'maybe'"), ({"score": True}, "True"),
                       ({"score": None}, "None")):
        units["units"]["Cat/01_02#s2"] = {"state": "graded", **bad}
        st.write_text(json.dumps(units))
        with pytest.raises(L.Halt, match=re.escape(f"evo_e1 could not be built: Cat/01_02#s2: score {shown} is not a "
                                                   "number from 0 to 1 or 'grader_failed'")):
            SE.run_build_evo_layout(sctx(r, 2))
        assert not (r.rd / "evo_e1").exists() and not list(r.rd.glob("evo_e1.building.*"))
    assert EL.GRADER_FAILED == "grader_failed"                             # the literal plugin_api.GRADER_FAILED records
    assert not hasattr(EL, "VERDICTS") and not hasattr(EL, "EVOLVER_SCORE")   # no verdict -> score map is left
    d["daytona"]["store"] = str(tmp_path / "nowhere")
    leg1.write_text(json.dumps(d))
    with pytest.raises(L.Halt, match="names no store dir"):
        SE.run_build_evo_layout(sctx(r, 2))
    # evo_layout's own refusals: an existing out dir, a bad epoch, the id maps
    with pytest.raises(EL.LayoutError, match="exists"):
        EL.build(r.rd, leg_json=r.base, epoch=0)
    for bad in (-1, True, "0"):
        with pytest.raises(ValueError):
            EL.epoch_dir(r.rd, bad)
    assert EL.safe_id("Cat/01_01") == "Cat_01_01" and EL.unit_of("a/b/c", 3) == "a_b_c_s3"
    assert EL.check(r.rd / "evo_e9", leg_json=r.base) == [f"{r.rd / 'evo_e9'} is not a directory"]
