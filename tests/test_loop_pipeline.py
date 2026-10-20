"""Offline tests for RUNTIME pipeline_val in src/orchestrator/loop.py: training rollouts first, the next evolver
starts as soon as the training rollouts are done, validation rollouts overlap with it and never block the loop.

A SIMULATED 3-epoch drive through the real `_drive` / `advance` / pipeline code: legs, the evolver and the
clock are fakes (a virtual clock; a leg is a pid that is "alive" until its end time and then writes its
result JSON), the step list is epoch_steps() replaced by steps carrying the REAL step names (so the
pipeline transform, the val bundle and the re-grade remedy path are the production code), and several verifies
are the real ones (loaded-promoted, infra exclusions, the outcome reconcile, the operator turns).

No sandbox, no model, no network, no claude CLI. The only real processes are two `python -c sleep`
dummies (pid-identity / pgrep tests), killed at the end of their test.
"""
from __future__ import annotations

import json
import shutil
import sys
import time as _time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import src.orchestrator.loop as L  # noqa: E402

Step = L.Step
VAL_D4 = L.LEG_D4_STEP.format(key="val")


# ══ the simulated world ═════════════════════════════════════════════════════════════════════════════
def write_leg(out: Path, skills: str, bad: bool = False) -> None:
    """A minimal SSB-shaped leg JSON; `bad` breaks the episode grid (n x n_samples != total)."""
    n, ns = 4, 3
    out.write_text(json.dumps({
        "skills_dir": skills, "n": n, "n_samples": ns, "sample_pass": 5,
        "sample_total": n * ns + (1 if bad else 0), "turn_exhausted": 0,
        "episodes_excluded_infra": 0, "episodes_timeout_scored_zero": 0,
        "per_task": {f"T{i}": [1.0, 0.0, 0.0] for i in range(n)}}))


class World:
    def __init__(self, rd: Path, dur: dict, evolver_s: float = 50.0, fail=(), die=()):
        self.rd, self.dur, self.evolver_s = rd, dict(dur), evolver_s
        self.fail, self.die = set(fail), set(die)
        self.t = 1000.0
        self.procs = {}              # pid -> {"end", "cmd", "on_end", "done", "lstart"}
        self.legs = {}               # tag -> pid of its latest launch
        self.launches = {}           # tag -> launch count
        self.events = []             # (t, kind, epoch, key)
        self.next_pid = 50000
        self.halt_once = set()       # epochs whose evolver raises Halt once
        self.needs_regrade = set()   # epochs whose val grader-failure check needs the re-grade
        self.remedy_s = 30.0
        self.skill_text = ""         # appended to every SKILL.md the evolver writes (a redo with other skills)

    def ev(self, kind, epoch, key="", t=None):
        self.events.append((self.t if t is None else t, kind, epoch, key))

    def first(self, kind, epoch, key=""):
        return next(i for i, e in enumerate(self.events) if e[1:] == (kind, epoch, key))

    def at(self, kind, epoch, key=""):
        return self.events[self.first(kind, epoch, key)][0]

    def spawn(self, dur, cmd, on_end=None) -> int:
        pid = self.next_pid
        self.next_pid += 1
        self.procs[pid] = {"end": self.t + dur, "cmd": cmd, "on_end": on_end, "done": False,
                           "lstart": f"start-{pid}"}
        return pid

    def advance(self, dt):
        """Move the clock forward, ending every process whose end falls inside the interval AT its end
        time, in end order (so what a process does when it ends happens at that virtual time)."""
        target = self.t + max(0.0, float(dt))
        while True:
            due = sorted((p["end"], pid) for pid, p in self.procs.items()
                         if not p["done"] and p["end"] <= target)
            if not due:
                self.t = target
                return
            end, pid = due[0]
            self.t = max(self.t, end)
            self.procs[pid]["done"] = True
            if self.procs[pid]["on_end"]:
                self.procs[pid]["on_end"]()

    def alive(self, pid):
        p = self.procs.get(pid)
        return bool(p) and not p["done"]

    def cmd(self, pid):
        return self.procs[pid]["cmd"] if self.alive(pid) else ""

    def lstart(self, pid):
        return self.procs[pid]["lstart"] if self.alive(pid) else ""

    def dispatch(self, ctx, key):
        """The fake leg launcher: _dispatch_leg's contract (0 = measured; a live same-tag leg = its pid)."""
        out = ctx.paths.leg(key)
        if out.is_file():
            return 0
        tag = out.stem
        if tag in self.legs and self.alive(self.legs[tag]):
            return self.legs[tag]
        e, skills = ctx.epoch, str(ctx.paths.promoted)

        def end(e=e, key=key, out=out, skills=skills):
            if (e, key) not in self.die:
                write_leg(out, skills, bad=(e, key) in self.fail)
            self.ev("end", e, key, t=self.procs[self.legs[out.stem]]["end"])
        pid = self.spawn(self.dur[(e, key)],
                         f"/py scripts/daytona/cc_leg_driver.py leg --runner "
                         f"run_ssbv2_claude_code --tag {tag} --key {key}", end)
        self.legs[tag] = pid
        self.launches[tag] = self.launches.get(tag, 0) + 1
        self.ev("launch", e, key)
        return pid


class FakeTime:
    """loop.py's `time` module in the simulation: a virtual clock; everything else is the real module."""
    def __init__(self, W):
        self.W = W

    def time(self):
        return self.W.t

    def sleep(self, s):
        self.W.advance(s)

    def __getattr__(self, n):
        return getattr(_time, n)


def fake_steps(W: World, d4: bool = False):
    """epoch_steps() for the simulation: the REAL step names (the pipeline transform and the val bundle
    select by name), fake digests / evolver / promotion, and the real leg verifies."""
    def v_digests(c):
        return c.paths.digests.is_dir(), "digests"

    def r_digests(c):
        if c.epoch > 1:   # epoch N+1's evidence is built from train_N: it must exist, val_N need not
            assert L.EpochPaths(c.run_dir, c.epoch - 1).leg("train").is_file()
        c.paths.digests.mkdir(parents=True)
        W.ev("digests", c.epoch)

    def v_staging(c):
        return (c.paths.staging / "s" / "SKILL.md").is_file(), "staging"

    def r_evolver(c):
        if c.epoch in W.halt_once:
            W.halt_once.discard(c.epoch)
            raise L.Halt(f"injected evolver failure in epoch {c.epoch}")
        W.ev("evolver_start", c.epoch)
        W.advance(W.evolver_s)
        (c.paths.staging / "s").mkdir(parents=True)
        (c.paths.staging / "s" / "SKILL.md").write_text(f"skill v{c.epoch}{W.skill_text}\n")
        W.ev("evolver_end", c.epoch)

    def v_promoted(c):
        return (c.paths.promoted / "s" / "SKILL.md").is_file(), "promoted"

    def r_promote(c):
        shutil.copytree(c.paths.staging, c.paths.promoted)
        W.ev("promote", c.epoch)

    def v_trainpair(c):
        return c.paths.leg("train").is_file(), "train pairs"

    def v_valpair(c):
        W.ev("valcheck", c.epoch)
        return c.paths.leg("val").is_file(), "val pairs"

    def v_d4(c):
        if c.epoch in W.needs_regrade and not (c.run_dir / f"regraded_e{c.epoch}").exists():
            return False, "val leg: 2 sample(s) were GRADER FAILED. RE-GRADE NEEDED: ..."
        return True, "val leg: no grader failure"

    def r_d4(c):
        (c.run_dir / f"regraded_e{c.epoch}").write_text("x")
        W.ev("regrade", c.epoch)

    steps = [
        Step("evidence surface is train-only", v_digests, run=r_digests),
        Step(L.TEMPLATE_CONVERSATION_STEP, L._v_operator_turns_recorded),
        Step("skills load through the real consumer", v_staging, run=r_evolver),
        Step("promoted set matches staging", v_promoted, run=r_promote),
        Step(L.TRAIN_LEG_STEP, lambda c: L._v_leg_loaded_promoted(c, "train"),
             run=lambda c: L._run_leg(c, "train")),
        Step(L.VAL_LEG_STEP, lambda c: L._v_leg_loaded_promoted(c, "val"),
             run=lambda c: L._run_leg(c, "val")),
    ]
    if d4:
        steps.append(Step(VAL_D4, v_d4, run=r_d4))
    steps += [
        Step("train pairs comparably with base", v_trainpair),
        Step("val pairs comparably with base", v_valpair),
        Step("train: no infra exclusions", lambda c: L._v_no_infra_exclusions(c, "train")),
        Step("val: no infra exclusions", lambda c: L._v_no_infra_exclusions(c, "val")),
        Step("train outcomes reconcile", lambda c: L._v_outcomes_reconcile(c, "train")),
        Step("val outcomes reconcile", lambda c: L._v_outcomes_reconcile(c, "val")),
    ]
    return steps


SPEC = NS(name="simrun", benchmark="ssbv2", split_path="", n_samples=3, n_epochs=3, cache_tag="t",
          solver_model="m")


@pytest.fixture
def sim(tmp_path, monkeypatch):
    def make(dur, *, fail=(), die=(), pipeline=True, pool=True, d4=False, evolver_s=50.0, skip_last=False,
             leg_timeout_s=100000):
        rd = tmp_path / "simrun"
        rd.mkdir()
        W = World(rd, dur, evolver_s, fail, die)
        monkeypatch.setattr(L, "epoch_steps", lambda: fake_steps(W, d4=d4))
        monkeypatch.setattr(L, "_dispatch_leg", W.dispatch)
        monkeypatch.setattr(L, "_proc_cmd", W.cmd)
        monkeypatch.setattr(L, "_proc_lstart", W.lstart)
        monkeypatch.setattr(L, "time", FakeTime(W))

        def no_spawn(c):
            raise AssertionError("a background re-grade was spawned where none was expected")
        monkeypatch.setattr(L, "_pipeline_spawn_remedy", no_spawn)
        orig_poll = L._pipeline_poll

        def logged_poll(ctx, **k):
            W.ev("poll", ctx.epoch)
            return orig_poll(ctx, **k)
        monkeypatch.setattr(L, "_pipeline_poll", logged_poll)
        kw = dict(leg_timeout_s=leg_timeout_s)
        if pipeline is not None:
            kw.update(pipeline_val=pipeline, leg_pool=pool)
        if skip_last:
            kw.update(skip_last_train=True)
        # the operator's one --drive --until-epoch 3 invocation, recorded under epoch 1
        L._conv_once(L.Ctx(rd, 1, SPEC, rd / "oracle", **kw), "invocation:1:sim", speaker="operator",
                     channel="invocation", content="simulated drive",
                     meta={"argv": f"--run-dir {rd} --epoch 1 --step --drive --until-epoch 3",
                           "mode": "drive"})
        return W, rd, kw
    return make


def drive(rd, kw, epoch=1, until=3, max_steps=500):
    ctx = L.Ctx(rd, epoch, SPEC, rd / "oracle", **kw)
    said = []
    rc = L._drive(ctx, NS(until_epoch=until, max_steps=max_steps), rd, SPEC, rd / "oracle", kw,
                  lambda text, key, c=None: said.append((key, text)), lambda c: "")
    return rc, said


def ran_steps(said):
    """The step names the drive RAN, in order, from its report keys (report:<epoch>:<n>:<name>)."""
    out = []
    for key, _text in said:
        parts = key.split(":", 3)
        if parts[0] == "report" and len(parts) == 4 and parts[2].isdigit():
            out.append((int(parts[1]), parts[3]))
    return out


def reg(rd):
    return json.loads((L.operator_dir(rd) / L.PIPELINE_VAL_FILE).read_text())


DUR = {(1, "train"): 10, (2, "train"): 10, (3, "train"): 10,
       (1, "val"): 70, (2, "val"): 100, (3, "val"): 100}


# ══ 1. the pipelined 3-epoch drive ═══════════════════════════════════════════════════════════════════
def test_pipelined_drive_order_val_failure_recorded_and_final_await(sim, capsys):
    W, rd, kw = sim(DUR, fail={(1, "val"), (3, "val")})
    rc, said = drive(rd, kw)
    out = capsys.readouterr().out
    assert rc == 0, out

    for n in (1, 2):
        # the next evolver is dispatched after train_N ended and BEFORE val_N ended
        assert W.at("end", n, "train") <= W.at("evolver_start", n + 1) < W.at("end", n, "val")
        # train_{N+1} is launched while val_N still runs
        assert W.at("launch", n + 1, "train") < W.at("end", n, "val")
        # the evidence of epoch N+1 was built from train_N only (asserted inside r_digests), before val_N
        assert W.at("digests", n + 1) < W.at("end", n, "val")

    # val_N's checks run at the FIRST step boundary after it ended (never earlier, never later)
    polls = [e[0] for e in W.events if e[1] == "poll"]
    for n in (1, 2, 3):
        t_end = W.at("end", n, "val")
        assert W.at("valcheck", n) == min(t for t in polls if t >= t_end)

    r = reg(rd)
    # val_1 FAILED after train_2 had started: recorded durably (step + detail + remedy) ...
    v1 = r["legs"]["1"]
    assert v1["state"] == "failed" and v1["failure"]["step"] == "val outcomes reconcile"
    assert "episode grid does not reconcile" in v1["failure"]["detail"]
    assert "--val-check --step" in v1["failure"]["remedy"]
    assert W.at("launch", 2, "train") < v1["checked_t"]
    # ... one clear line in the drive output names the epoch, the check and the remedy ...
    assert "[pipeline] VAL e1 (simrun_v1_val): FAILED 'val outcomes reconcile'" in out
    assert "REMEDY:" in out and "Evolution CONTINUES" in out
    # ... and the loop CONTINUED: the epoch-3 evolver ran after the failure was recorded
    assert W.first("valcheck", 1) < W.first("evolver_start", 3)
    assert all(k[:2] != "halt" for k, _ in said)

    # the END awaited every val leg: DRIVE DONE only after val_3 ended, and lists the failures
    assert r["legs"]["2"]["state"] == "passed" and r["legs"]["3"]["state"] == "failed"
    assert W.t >= W.at("end", 3, "val")
    assert "PIPELINE END" in out and "VAL FAILURES" in out and "e1 'val outcomes reconcile'" in out
    assert "e3 'val outcomes reconcile'" in out
    assert out.index("PIPELINE END") < out.index("DRIVE DONE")
    assert set(r["train"]) == {"1", "2", "3"}

    # held-out: epoch 3's val failed -> the rule skips it and selects 2; epoch 1 alone qualifies for none
    gate = L._pipeline_heldout_gate(L.Ctx(rd, 3, SPEC, rd / "oracle", **kw))
    e, why = L.last_successful_epoch(rd, 3, "", 3, val_gate=gate)
    assert e == 2 and why[0][0] == 3 and "FAILED" in why[0][1] and "val outcomes reconcile" in why[0][1]
    e1, why1 = L.last_successful_epoch(rd, 1, "", 3, val_gate=gate)
    assert e1 is None and "FAILED" in why1[0][1]
    ok, det = L._v_heldout_epoch(L.Ctx(rd, 3, SPEC, rd / "oracle", **kw))
    assert not ok and "selects epoch 2" in det
    ok, det = L._v_heldout_epoch(L.Ctx(rd, 2, SPEC, rd / "oracle", **kw))
    assert ok and "val re-verified now" in det

    # the one drive invocation covers every epoch it crossed
    for n in (2, 3):
        ok, det = L._v_operator_turns_recorded(L.Ctx(rd, n, SPEC, rd / "oracle", **kw))
        assert ok, det


def test_pipelined_step_sequence_never_awaits_val(sim):
    W, rd, kw = sim(DUR)
    rc, said = drive(rd, kw)
    assert rc == 0
    seq = ran_steps(said)
    for n in (1, 2, 3):
        assert [s for e, s in seq if e == n] == ["evidence surface is train-only",
                                                 "skills load through the real consumer",
                                                 "promoted set matches staging", L.TRAIN_LEG_STEP]
    assert not any(s == L.VAL_LEG_STEP for _e, s in seq)
    # epoch 3 completed while val_2 and val_3 were still running (only the end waited for them)
    r = reg(rd)
    assert r["train"]["3"]["t"] < W.at("end", 2, "val") < W.at("end", 3, "val") <= W.t
    assert all(r["legs"][k]["state"] == "passed" for k in ("1", "2", "3"))


# ══ 2. switch off = the original order ═══════════════════════════════════════════════════════════════
def _sig(steps):
    code = lambda f: None if f is None else getattr(f, "__code__", f)      # noqa: E731
    return [(s.name, code(s.verify), code(s.run), s.operator_gate, s.defect) for s in steps]


@pytest.mark.parametrize("pipeline", [None, False])
def test_switch_off_is_the_serial_program(pipeline, monkeypatch):
    monkeypatch.setattr(L, "_prompt_mode", lambda c: "template")
    base = L.epoch_steps()
    for mode, want in (("ao", base), ("template", L._template_steps(L.epoch_steps()))):
        c = NS(prompt_mode=mode, **({} if pipeline is None else {"pipeline_val": pipeline}))
        assert _sig(L._steps_for(c)) == _sig(want)


@pytest.mark.parametrize("pipeline", [None, False])
def test_switch_off_drive_awaits_val_inside_each_epoch(sim, capsys, pipeline):
    W, rd, kw = sim(DUR, pipeline=pipeline)
    rc, said = drive(rd, kw)
    out = capsys.readouterr().out
    assert rc == 0, out
    seq = ran_steps(said)
    for n in (1, 2, 3):
        assert [s for e, s in seq if e == n] == ["evidence surface is train-only",
                                                 "skills load through the real consumer",
                                                 "promoted set matches staging", L.TRAIN_LEG_STEP,
                                                 L.VAL_LEG_STEP]
    for n in (1, 2):
        assert W.at("end", n, "val") <= W.at("evolver_start", n + 1)      # serial: val blocks
    assert not any(e[1] == "poll" for e in W.events)
    assert not (L.operator_dir(rd) / L.PIPELINE_VAL_FILE).exists()
    assert "[pipeline]" not in out and "PIPELINE" not in out


def test_pipeline_transform_of_the_real_program(monkeypatch):
    monkeypatch.setattr(L, "_prompt_mode", lambda c: "template")
    for mode in ("ao", "template"):
        serial = L._serial_steps_for(NS(prompt_mode=mode))
        piped = L._steps_for(NS(prompt_mode=mode, pipeline_val=True))
        sn, pn = [s.name for s in serial], [s.name for s in piped]
        removed = [n for n in sn if n in L.PIPELINE_VAL_STEPS]
        assert removed and not set(removed) & set(pn)
        assert pn[pn.index(L.TRAIN_LEG_STEP) + 1] == L.VAL_TRACKED_STEP
        # every other step is the SAME object in the same order
        keep = [s for s in serial if s.name not in L.PIPELINE_VAL_STEPS
                and s.name != L.TRAIN_LEG_STEP]
        pk = [s for s in piped if s.name not in (L.TRAIN_LEG_STEP, L.VAL_TRACKED_STEP)]
        assert _sig(keep) == _sig(pk)
        bundle = [s.name for s in L._val_bundle(NS(prompt_mode=mode))]
        assert bundle == removed
        assert (VAL_D4 in bundle) == (mode == "template")
        b = {s.name: s for s in L._val_bundle(NS(prompt_mode=mode))}
        assert b[L.VAL_LEG_STEP].run is None                 # never awaits the val leg
        if mode == "template":
            assert b[VAL_D4].run is not None                 # the grader-failure step keeps its re-grade runner


# ══ 3. held-out refusal until the val legs are complete and checked ════════════════════════════════════
def test_heldout_refuses_until_val_complete_and_plan_shows_it(sim, monkeypatch, capsys):
    W, rd, kw = sim(DUR)
    real_final = L._pipeline_final
    monkeypatch.setattr(L, "_pipeline_final", lambda ctx, say=None: [])
    rc, _said = drive(rd, kw)
    assert rc == 0
    ctx3 = L.Ctx(rd, 3, SPEC, rd / "oracle", **kw)
    gate = L._pipeline_heldout_gate(ctx3)
    e, why = L.last_successful_epoch(rd, 3, "", 3, val_gate=gate)
    # val_2 and val_3 still run: neither epoch may be selected; val_1 ended and passed its checks
    assert e == 1, why
    assert [k for k, _w in why] == [3, 2]
    assert all("no val leg" in w or "RUNNING" in w for _k, w in why)

    ctx3.readonly = True
    plan = L._pipeline_plan_text(ctx3)
    assert "await TRAIN_N only" in plan
    assert "e2 simrun_v2_val: running" in plan and "ALIVE — its checks run at the first step boundary" in plan
    assert "held-out rule now: epoch 1" in plan
    assert "e1 simrun_v1_val: passed" in plan

    # the end: await and check every val leg; now epoch 3 qualifies
    ctx3.readonly = False
    assert real_final(ctx3) == []
    assert L.last_successful_epoch(rd, 3, "", 3, val_gate=gate)[0] == 3
    # UNCHECKED (no record) and CHANGED bytes are refused too
    r = reg(rd)
    rec3 = r["legs"].pop("3")
    (L.operator_dir(rd) / L.PIPELINE_VAL_FILE).write_text(json.dumps(r))
    e, why = L.last_successful_epoch(rd, 3, "", 3, val_gate=gate)
    assert e == 2 and "UNCHECKED" in why[0][1]
    r["legs"]["3"] = rec3
    (L.operator_dir(rd) / L.PIPELINE_VAL_FILE).write_text(json.dumps(r))
    v3 = L.EpochPaths(rd, 3).leg("val")
    write_leg(v3, str(L.EpochPaths(rd, 3).promoted), bad=False)
    v3.write_text(v3.read_text() + " ")
    e, why = L.last_successful_epoch(rd, 3, "", 3, val_gate=gate)
    assert e == 2 and "changed since its val checks passed" in why[0][1]


# ══ 4. restart safety ════════════════════════════════════════════════════════════════════════════════
def test_restart_reattaches_a_running_val_leg_by_pid_identity(sim, capsys):
    dur = dict(DUR)
    dur[(1, "val")] = 300
    W, rd, kw = sim(dur)
    W.halt_once.add(2)
    rc, _ = drive(rd, kw)
    assert rc == 1                                            # the injected HALT in epoch 2's evolver
    tag = L.EpochPaths(rd, 1).leg("val").stem
    pid = reg(rd)["legs"]["1"]["pid"]
    assert reg(rd)["legs"]["1"]["state"] == "running" and W.alive(pid)
    assert reg(rd)["legs"]["1"]["identity"] == {"lstart": f"start-{pid}", "cmd_has_tag": True}

    rc, _ = drive(rd, kw, epoch=2)                            # a NEW loop process, same record
    assert rc == 0
    r = reg(rd)
    assert W.launches[tag] == 1 and r["legs"]["1"]["pid"] == pid     # re-attached, never relaunched
    assert r["legs"]["1"]["state"] == "passed"
    assert W.at("valcheck", 1) >= W.at("end", 1, "val")


def test_restart_does_not_trust_a_recycled_pid(sim, capsys):
    """VAL NEVER BLOCKS: the recycled pid reads as a dead leg, which is RECORDED, not a HALT."""
    dur = dict(DUR)
    dur[(1, "val")] = 300
    W, rd, kw = sim(dur, die={(1, "val")})
    W.halt_once.add(2)
    assert drive(rd, kw)[0] == 1
    pid = reg(rd)["legs"]["1"]["pid"]
    W.procs[pid]["lstart"] = "a different process now holds this pid"
    capsys.readouterr()
    rc, _ = drive(rd, kw, epoch=2)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert reg(rd)["legs"]["1"]["state"] == "died" and "DIED WITHOUT A RESULT" in out
    assert W.launches[L.EpochPaths(rd, 1).leg("val").stem] == 1          # never relaunched by the loop


def test_a_val_leg_that_died_is_recorded_and_evolution_continues(sim, capsys):
    """Item 2: a val leg that dies without a result is a RECORDED val failure with the exact relaunch command;
    evolution continues, the held-out rule skips the epoch, the end lists it, and only the operator's
    --val-check --step relaunches it."""
    W, rd, kw = sim(DUR, die={(1, "val")})
    rc, said = drive(rd, kw)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "EXITED WITHOUT WRITING simrun_v1_val.json" in out and "DIED WITHOUT A RESULT" in out
    v1 = reg(rd)["legs"]["1"]
    c1 = L.Ctx(rd, 1, SPEC, rd / "oracle", **kw)
    assert v1["state"] == "died" and v1["failure"]["step"] == L.VAL_DIED_STEP
    assert L._val_check_cmd(c1) in v1["failure"]["remedy"]
    assert v1["failed_t"] <= W.at("evolver_start", 3)                    # evolution went on past it
    assert all(k[:2] != "halt" for k, _ in said)
    assert "VAL FAILURES" in out and f"e1 '{L.VAL_DIED_STEP}'" in out
    gate = L._pipeline_heldout_gate(L.Ctx(rd, 3, SPEC, rd / "oracle", **kw))
    assert L.last_successful_epoch(rd, 3, "", 3, val_gate=gate)[0] == 3
    e1, why1 = L.last_successful_epoch(rd, 1, "", 3, val_gate=gate)
    assert e1 is None and "no val leg" in why1[0][1]                     # no result: the rule skips it
    ok, det = gate(1)
    assert not ok and "DIED" in det
    tag = L.EpochPaths(rd, 1).leg("val").stem
    assert W.launches[tag] == 1
    rc, _ = drive(rd, kw, epoch=3)                            # a restart neither HALTs nor relaunches it
    assert rc == 0 and W.launches[tag] == 1
    # the operator's decision: the record's relaunch command, then the end checks it
    W.die.clear()
    assert L._val_check_program(c1, step=True) == 0
    r = reg(rd)
    assert W.launches[tag] == 2 and r["legs"]["1"]["state"] == "running"
    assert r["superseded"][-1]["state"] == "died"
    rc, _ = drive(rd, kw, epoch=3)
    assert rc == 0 and reg(rd)["legs"]["1"]["state"] == "passed"


def test_an_unreadable_record_halts_and_is_never_replaced(sim):
    W, rd, kw = sim(DUR)
    p = L.operator_dir(rd) / L.PIPELINE_VAL_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{not json")
    rc, _ = drive(rd, kw)
    assert rc == 1 and p.read_text() == "{not json"


# ══ 5. the re-grade path runs ONCE in the background and never blocks the loop ══════════════════════════
def test_d4_regrade_runs_once_in_the_background(sim, monkeypatch, capsys):
    W, rd, kw = sim(DUR, d4=True)
    W.needs_regrade.add(1)
    spawned = []

    def fake_spawn(c):
        spawned.append((W.t, c.epoch))
        cc = L.Ctx(rd, c.epoch, SPEC, rd / "oracle", **kw)
        pid = W.spawn(W.remedy_s, f"python -m src.orchestrator.loop --run-dir {rd} --epoch {c.epoch} "
                                  f"--val-check --step",
                      on_end=lambda: L._val_check_program(cc, step=True))
        return {"pid": pid, "identity": {"lstart": f"start-{pid}"}, "log": "sim.log"}
    monkeypatch.setattr(L, "_pipeline_spawn_remedy", fake_spawn)
    rc, _ = drive(rd, kw)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert len(spawned) == 1 and spawned[0][1] == 1
    assert [e for e in W.events if e[1] == "regrade"] == [(spawned[0][0] + W.remedy_s, "regrade", 1, "")]
    # the loop did not wait for it: the epoch-3 evolver started at the boundary that spawned it
    assert W.at("evolver_start", 3) == spawned[0][0]
    assert "is running ONCE in the background" in out
    assert out.count("[pipeline] VAL e1 (simrun_v1_val): all") >= 1
    r = reg(rd)
    assert r["legs"]["1"]["state"] == "passed" and r["legs"]["1"]["remedy"]["attempted_for"]


# ══ 6. seat guard ════════════════════════════════════════════════════════════════════════
def test_seat_guard_lifts_only_with_pipeline_and_pool(tmp_path, monkeypatch):
    rd = tmp_path / "r"
    rd.mkdir()
    script = tmp_path / "leg.sh"
    script.write_text("#!/bin/sh\n")
    spec = NS(name="r", benchmark="ssbv2", split_path="s.json", n_samples=3, cache_tag="t",
              solver_model="m", solver_provider_order=())
    monkeypatch.setattr(L, "_live_runners", lambda ctx=None: [11, 12, 13])
    for kw, halts in ((dict(), True), (dict(pipeline_val=True), True), (dict(leg_pool=True), True),
                      (dict(pipeline_val=True, leg_pool=True), False)):
        ctx = L.Ctx(rd, 2, spec, rd / "oracle", leg_script=str(script), **kw)
        if halts:
            with pytest.raises(L.Halt, match="third") as ei:
                L._leg_argv_env(ctx, "val")
            assert ("pool is not true" in str(ei.value)) == bool(kw.get("pipeline_val"))
        else:
            argv, _env = L._leg_argv_env(ctx, "val")
            assert argv == ["bash", str(script), "r_v2_val", "val", "skills_v2"]


def test_only_a_needed_regrade_is_run_in_the_background(monkeypatch, tmp_path):
    rd = tmp_path / "x"
    rd.mkdir()
    c = L.Ctx(rd, 1, SPEC, rd / "oracle", pipeline_val=True)
    for detail, runnable in (("2 GRADER FAILED. RE-GRADE NEEDED: ...", True),
                             ("2 GRADER FAILED. RE-GRADE FAILED AGAIN for 1: ...", False)):
        monkeypatch.setattr(L, "_val_bundle", lambda ctx, d=detail: [
            Step(VAL_D4, lambda cc, d=d: (False, d), run=lambda cc: None)])
        res = L._pipeline_check_val(c)
        assert res["state"] == "failed" and res["remedy_runnable"] is runnable
        assert ("runs it in the background" in L._pipeline_remedy_text(c, res)) is runnable


# ══ 7. a val record counts only while it is CURRENT ═══════════════════════════════════
def _supersede_epoch(rd: Path, n: int) -> None:
    """What 'redo eN' does in this project: move the epoch's artifacts aside. The operator dir is untouched.
    (The checker's helper, copied from its adversarial suite.)"""
    sup = rd / f"_superseded_e{n}"
    sup.mkdir()
    ep = L.EpochPaths(rd, n)
    for p in (ep.leg("train"), ep.leg("val"), ep.promoted, ep.staging):
        if p.exists():
            shutil.move(str(p), str(sup / p.name))


def test_redo_of_a_checked_epoch_launches_its_val_leg_again(sim, capsys):
    """The checker's test_BUG_redo_of_a_checked_epoch_never_launches_its_val_leg, now expected to PASS: the
    stale epoch-1 record is superseded (its promoted set is gone), the redone epoch's val leg is launched, and
    the end reports that leg, not the old one."""
    W, rd, kw = sim(DUR)
    rc, _ = drive(rd, kw, epoch=1, until=1)
    assert rc == 0
    tag = L.EpochPaths(rd, 1).leg("val").stem
    assert W.launches[tag] == 1
    _supersede_epoch(rd, 1)
    capsys.readouterr()
    rc, _ = drive(rd, kw, epoch=1, until=1)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert W.launches[tag] == 2, (f"val leg of the redone epoch was never launched; the drive said: "
                                  f"{out[out.find('DRIVE DONE'):][:200]!r}")
    r = reg(rd)
    assert r["superseded"][0]["state"] == "passed" and "is gone" in r["superseded"][0]["why"]
    assert r["legs"]["1"]["state"] == "passed" and r["legs"]["1"]["pid"] != r["superseded"][0]["pid"]
    assert "SUPERSEDED" in out and "NOT counted" in out


def test_redo_while_old_val_leg_record_says_running_launches_the_new_leg(sim, capsys):
    """The checker's test_BUG_redo_while_old_val_leg_record_says_running_halts_misleadingly, now expected to
    PASS: no HALT about the OLD pid; the redone epoch's val leg is launched."""
    dur = dict(DUR)
    dur[(1, "val")] = 400
    W, rd, kw = sim(dur, die={(1, "val")})
    W.halt_once.add(2)
    assert drive(rd, kw)[0] == 1                            # e2's evolver halts; val_1 still 'running'
    old = reg(rd)["legs"]["1"]["pid"]
    W.procs[old]["done"] = True                             # the operator kills the old val leg for the redo
    _supersede_epoch(rd, 1)
    shutil.rmtree(L.EpochPaths(rd, 2).digests, ignore_errors=True)
    shutil.rmtree(L.EpochPaths(rd, 1).digests, ignore_errors=True)
    capsys.readouterr()
    rc, _ = drive(rd, kw, epoch=1, until=1)
    out = capsys.readouterr().out
    tag = L.EpochPaths(rd, 1).leg("val").stem
    assert W.launches[tag] == 2, "the redone epoch's val leg was not launched (stale 'running' record reused)"
    assert rc == 0 and "EXITED WITHOUT WRITING" in out       # the NEW leg (die set) is recorded, not HALTed
    assert reg(rd)["legs"]["1"]["pid"] != old


def test_a_superseded_leg_still_running_is_never_adopted(sim, capsys):
    """Redo with DIFFERENT skills while the old val leg still runs: the launcher would re-attach to it (same
    tag, same epoch) and it would write the redone epoch's result from the old skill set. The epoch's entry is
    a recorded failure naming it; its later result is never adopted; the operator's relaunch works once the
    stale leg and its result are out of the way."""
    dur = dict(DUR)
    dur[(1, "val")] = 1000
    W, rd, kw = sim(dur)
    W.halt_once.add(2)
    assert drive(rd, kw)[0] == 1
    old = reg(rd)["legs"]["1"]["pid"]
    assert W.alive(old)
    _supersede_epoch(rd, 1)
    shutil.rmtree(L.EpochPaths(rd, 1).digests, ignore_errors=True)
    W.skill_text = " (redo)"
    capsys.readouterr()
    rc, _ = drive(rd, kw, epoch=1, until=1)
    out = capsys.readouterr().out
    tag = L.EpochPaths(rd, 1).leg("val").stem
    c1 = L.Ctx(rd, 1, SPEC, rd / "oracle", **kw)
    assert rc == 0, out
    assert W.launches[tag] == 1                               # not re-attached, not a second writer
    v1 = reg(rd)["legs"]["1"]
    assert v1["state"] == "blocked" and v1["promoted_sha256"] == L._promoted_sha(c1)
    assert f"stop pid {old}" in v1["failure"]["remedy"] and "--val-check --step" in v1["failure"]["remedy"]
    assert "STILL RUNNING" in out and "NOT LAUNCHED" in out and "VAL FAILURES" in out
    # the stale leg ends and writes the result file: never adopted by the poll
    W.advance(2000)
    assert L.EpochPaths(rd, 1).leg("val").is_file()
    L._pipeline_poll(c1)
    assert reg(rd)["legs"]["1"]["state"] == "blocked"
    # the operator's relaunch refuses a result that may be the superseded leg's ...
    assert L._val_check_program(c1, step=True) == 1
    assert reg(rd)["legs"]["1"]["state"] == "blocked" and "move" in reg(rd)["legs"]["1"]["failure"]["remedy"]
    # ... and relaunches once it is moved aside; the new leg is checked against the redone skills
    L.EpochPaths(rd, 1).leg("val").rename(rd / "stale_v1_val.json")
    assert L._val_check_program(c1, step=True) == 0
    assert W.launches[tag] == 2 and reg(rd)["legs"]["1"]["state"] == "running"
    W.advance(1100)
    L._pipeline_poll(c1)
    assert reg(rd)["legs"]["1"]["state"] == "passed"


def test_the_end_never_prints_passed_for_a_missing_result(sim, capsys):
    W, rd, kw = sim(DUR)
    assert drive(rd, kw, epoch=1, until=1)[0] == 0
    c1 = L.Ctx(rd, 1, SPEC, rd / "oracle", **kw)
    assert reg(rd)["legs"]["1"]["state"] == "passed"
    L.EpochPaths(rd, 1).leg("val").rename(rd / "moved_v1_val.json")
    capsys.readouterr()
    # the listing alone (no poll first) does not count it ...
    fails = L._pipeline_list(c1, reg(rd))
    out = capsys.readouterr().out
    assert fails and fails[0][0] == 1 and "PASSED" not in out and "NOT COUNTED" in out
    # ... and the end supersedes it and lists it as not tracked
    fails = L._pipeline_final(c1)
    out = capsys.readouterr().out
    assert fails and fails[0][0] == 1
    assert "PASSED" not in out and "NOT TRACKED" in out and "is gone" in out
    assert "1" not in reg(rd)["legs"] and reg(rd)["superseded"][-1]["state"] == "passed"


def test_a_changed_result_is_superseded_and_checked_again(sim, capsys):
    W, rd, kw = sim(DUR)
    assert drive(rd, kw, epoch=1, until=1)[0] == 0
    c1 = L.Ctx(rd, 1, SPEC, rd / "oracle", **kw)
    write_leg(L.EpochPaths(rd, 1).leg("val"), str(c1.paths.promoted), bad=True)
    L._pipeline_poll(c1)
    r = reg(rd)
    assert "changed since it was checked" in r["superseded"][-1]["why"]
    assert r["legs"]["1"]["state"] == "failed" and r["legs"]["1"]["failure"]["step"] == "val outcomes reconcile"


def test_heldout_gate_refuses_a_val_record_of_another_promoted_set(sim):
    W, rd, kw = sim(DUR)
    assert drive(rd, kw)[0] == 0
    ep = L.EpochPaths(rd, 3)
    for d in (ep.promoted, ep.staging):                      # same bytes in both: promotion itself is fine
        (d / "s" / "SKILL.md").write_text("skill v3, edited after its val leg ran\n")
    gate = L._pipeline_heldout_gate(L.Ctx(rd, 3, SPEC, rd / "oracle", **kw))
    e, why = L.last_successful_epoch(rd, 3, "", 3, val_gate=gate)
    assert e == 2 and "not the one on disk" in why[0][1]


# ══ 8. VAL NEVER BLOCKS ═══════════════════════════════════════════════════════════════════
def test_a_val_leg_past_its_own_timeout_is_recorded_and_not_awaited(sim, capsys):
    dur = dict(DUR)
    dur[(1, "val")] = 1000
    W, rd, kw = sim(dur, leg_timeout_s=200)
    rc, said = drive(rd, kw)
    out = capsys.readouterr().out
    assert rc == 0, out
    v1 = reg(rd)["legs"]["1"]
    pid = v1["pid"]
    assert v1["state"] == "timed_out" and v1["failure"]["step"] == L.VAL_TIMEOUT_STEP
    # measured from ITS OWN dispatch, not from when the end started waiting
    assert 200 <= v1["failed_t"] - v1["dispatched_t"] < 200 + L.PIPELINE_POLL_S + 1
    assert W.alive(pid) and W.t < W.procs[pid]["end"]           # the drive did not wait for it
    c1 = L.Ctx(rd, 1, SPEC, rd / "oracle", **kw)
    assert f"stop pid {pid}" in v1["failure"]["remedy"] and L._val_check_cmd(c1) in v1["failure"]["remedy"]
    assert "TIMED OUT" in out and f"e1 '{L.VAL_TIMEOUT_STEP}'" in out
    assert all(k[:2] != "halt" for k, _ in said)
    gate = L._pipeline_heldout_gate(L.Ctx(rd, 3, SPEC, rd / "oracle", **kw))
    e1, why1 = L.last_successful_epoch(rd, 1, "", 3, val_gate=gate)
    assert e1 is None and "no val leg" in why1[0][1]
    ok, det = gate(1)
    assert not ok and "TIMED_OUT" in det
    # its relaunch command refuses to start a second leg over the live one
    assert L._val_check_program(c1, step=True) == 1 and W.launches[L.EpochPaths(rd, 1).leg("val").stem] == 1
    # a result it writes later is checked at the next boundary
    W.advance(1000)
    L._pipeline_poll(c1)
    assert reg(rd)["legs"]["1"]["state"] == "passed"


def test_val_death_during_a_train_await_is_recorded_not_halted(sim, capsys):
    """The checker's test_val_death_during_train_await_halts_the_drive documented a HALT; under 'val never
    blocks' the death is recorded by the tick poll while train_2 is awaited, and train_2 carries on."""
    dur = dict(DUR)
    dur[(1, "val")] = 200
    dur[(2, "train")] = 500
    W, rd, kw = sim(dur, die={(1, "val")})
    rc, _ = drive(rd, kw)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "EXITED WITHOUT WRITING simrun_v1_val.json" in out
    v1 = reg(rd)["legs"]["1"]
    assert v1["state"] == "died" and W.at("launch", 2, "train") < v1["failed_t"] < W.at("end", 2, "train")
    assert W.launches[L.EpochPaths(rd, 2).leg("train").stem] == 1
    assert W.at("evolver_start", 3) > v1["failed_t"]


def test_a_non_halt_error_in_the_poll_is_recorded_not_raised(sim, monkeypatch, capsys):
    W, rd, kw = sim(DUR)
    real = L._pipeline_check_val
    boom = {"n": 0}

    def flaky(c, **k):
        if c.epoch == 1 and boom["n"] == 0:
            boom["n"] += 1
            raise OSError("disk hiccup")
        return real(c, **k)
    monkeypatch.setattr(L, "_pipeline_check_val", flaky)
    rc, _ = drive(rd, kw)
    out = capsys.readouterr().out
    assert rc == 0, out
    r = reg(rd)
    assert boom["n"] == 1 and r["poll_errors"][0]["epoch"] == "1" and "OSError: disk hiccup" in r["poll_errors"][0]["error"]
    assert "the poll could not process this val leg" in out and "Traceback" not in out
    assert r["legs"]["1"]["state"] == "passed"               # checked at a later boundary


def test_the_regrade_is_marked_attempted_only_after_it_started(sim, monkeypatch, capsys):
    W, rd, kw = sim(DUR, d4=True)
    W.needs_regrade.add(1)
    calls, snaps = [], []

    def spawn(c):
        calls.append(W.t)
        if len(calls) == 1:
            raise OSError("cannot fork")
        snaps.append(json.loads(json.dumps(reg(rd)["legs"]["1"])))
        cc = L.Ctx(rd, c.epoch, SPEC, rd / "oracle", **kw)
        pid = W.spawn(W.remedy_s, f"python -m src.orchestrator.loop --run-dir {rd} --epoch {c.epoch} "
                                  f"--val-check --step", on_end=lambda: L._val_check_program(cc, step=True))
        return {"pid": pid, "identity": {"lstart": f"start-{pid}"}, "log": "sim.log"}
    monkeypatch.setattr(L, "_pipeline_spawn_remedy", spawn)
    rc, _ = drive(rd, kw)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert len(calls) == 2 and calls[0] <= calls[1]          # the failed start was retried at a later boundary
    before = snaps[0]                                        # the record as the SECOND start began
    assert before["state"] == "failed" and "attempted_for" not in before["remedy"]
    assert before["remedy"]["error"].startswith("OSError") and before["remedy"]["spawn_failures"] == 1
    assert "background start FAILED" in before["failure"]["remedy"]
    v1 = reg(rd)["legs"]["1"]
    assert v1["state"] == "passed" and v1["remedy"]["attempted_for"] == v1["leg_sha256"]
    assert "did NOT start" in out


def test_a_regrade_that_never_starts_is_tried_a_bounded_number_of_times(sim, monkeypatch, capsys):
    W, rd, kw = sim(DUR, d4=True)
    W.needs_regrade.add(1)
    calls = []

    def boom(c):
        calls.append(W.t)
        raise OSError("cannot fork")
    monkeypatch.setattr(L, "_pipeline_spawn_remedy", boom)
    rc, _ = drive(rd, kw)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert len(calls) == L.PIPELINE_REMEDY_SPAWN_TRIES
    v1 = reg(rd)["legs"]["1"]
    assert v1["state"] == "failed" and "does not try again" in v1["failure"]["remedy"]
    assert "VAL FAILURES" in out


def test_val_alive_without_a_recorded_start_time(tmp_path, monkeypatch):
    """Item 3: no recorded start time -> alive only if the result is absent AND the pid names the leg's tag."""
    rd = tmp_path / "r"
    rd.mkdir()
    c = L.Ctx(rd, 1, SPEC, rd / "oracle", pipeline_val=True)
    tag = c.paths.leg("val").stem
    cmds = {777: f"python scripts/run_ssbv2_base30.py --tag {tag}",
            778: "python scripts/run_ssbv2_base30.py --tag another_run_v1_val"}
    monkeypatch.setattr(L, "_proc_cmd", lambda pid: cmds.get(int(pid), ""))
    monkeypatch.setattr(L, "_proc_lstart", lambda pid: "Fri Sep 25 10:00:00 2026" if int(pid) in cmds else "")
    base = {"tag": tag, "pattern": "run_ssbv2", "state": "running"}
    for ident in (None, {"lstart": "", "cmd_has_tag": False}):
        assert L._val_alive(c, {**base, "pid": 777, "identity": ident})
        assert not L._val_alive(c, {**base, "pid": 778, "identity": ident})   # a recycled pid, another leg
        assert not L._val_alive(c, {**base, "pid": 779, "identity": ident})   # gone
        write_leg(c.paths.leg("val"), str(c.paths.promoted))
        assert not L._val_alive(c, {**base, "pid": 777, "identity": ident})   # its result exists
        c.paths.leg("val").unlink()
    # a recorded start time: the recorded identity decides, as before
    ok = {"lstart": "Fri Sep 25 10:00:00 2026", "cmd_has_tag": True}
    assert L._val_alive(c, {**base, "pid": 777, "identity": ok})
    assert not L._val_alive(c, {**base, "pid": 777, "identity": {**ok, "lstart": "another start"}})


# ══ 9. SKIP LAST TRAIN (RUNTIME skip_last_train) ═════════════════════════════════════════════
def test_skip_last_train_launches_only_val_at_the_last_epoch(sim, monkeypatch, capsys):
    W, rd, kw = sim(DUR, skip_last=True)
    rc, said = drive(rd, kw)
    out = capsys.readouterr().out
    assert rc == 0, out
    stem = lambda n, k: L.EpochPaths(rd, n).leg(k).stem       # noqa: E731
    # 2 training rollouts and 3 val rollouts
    assert [W.launches.get(stem(n, "train"), 0) for n in (1, 2, 3)] == [1, 1, 0]
    assert [W.launches.get(stem(n, "val"), 0) for n in (1, 2, 3)] == [1, 1, 1]
    seq = ran_steps(said)
    for n in (1, 2):                                         # every other epoch is unchanged
        assert [s for e, s in seq if e == n] == ["evidence surface is train-only",
                                                 "skills load through the real consumer",
                                                 "promoted set matches staging", L.TRAIN_LEG_STEP]
    assert [s for e, s in seq if e == 3] == ["evidence surface is train-only",
                                             "skills load through the real consumer",
                                             "promoted set matches staging", L.VAL_TRACKED_STEP]
    # the last epoch's train-leg checks are recorded "skipped by design (no later evolver reads it)"
    c3 = L.Ctx(rd, 3, SPEC, rd / "oracle", **kw)
    rows = {s.name: (ok, d) for s, ok, d, _q in L.evaluate(c3)[0]}
    for name in (L.TRAIN_LEG_STEP, "train pairs comparably with base", "train: no infra exclusions",
                 "train outcomes reconcile"):
        assert rows[name][0] and rows[name][1].startswith(L.UNMEASURED) and L.SKIPPED_BY_DESIGN in rows[name][1]
    tr3 = reg(rd)["train"]["3"]
    assert tr3["skipped_by_design"] and tr3["leg_sha256"] is None
    assert set(tr3["train_checks"]) == {L.TRAIN_LEG_STEP, "train pairs comparably with base",
                                        "train: no infra exclusions", "train outcomes reconcile"}
    assert set(tr3["train_checks"].values()) == {L.SKIPPED_BY_DESIGN}
    assert not reg(rd)["train"]["2"].get("skipped_by_design")
    # its val leg was checked and passed; the held-out rule accepts epoch 3 on it
    assert reg(rd)["legs"]["3"]["state"] == "passed" and W.first("valcheck", 3) >= 0
    gate = L._pipeline_heldout_gate(c3)
    assert L.last_successful_epoch(rd, 3, "", 3, val_gate=gate, skip_train_epoch=3)[0] == 3
    e, why = L.last_successful_epoch(rd, 3, "", 3, val_gate=gate)          # without the skip: needs train_3
    assert e == 2 and "no train leg" in why[0][1]
    ok, det = L._v_heldout_epoch(c3)
    assert ok and L.SKIPPED_BY_DESIGN in det, det
    c3.readonly = True
    assert "skip_last_train (RUNTIME): epoch 3" in L._pipeline_plan_text(c3)


def test_skip_last_train_last_epoch_needs_its_val_leg_to_pass(sim):
    W, rd, kw = sim(DUR, skip_last=True, fail={(3, "val")})
    assert drive(rd, kw)[0] == 0
    c3 = L.Ctx(rd, 3, SPEC, rd / "oracle", **kw)
    e, why = L.last_successful_epoch(rd, 3, "", 3, val_gate=L._pipeline_heldout_gate(c3), skip_train_epoch=3)
    assert e == 2 and why[0][0] == 3 and "FAILED" in why[0][1]
    ok, det = L._v_heldout_epoch(c3)
    assert not ok and "selects epoch 2" in det


def test_a_skipped_train_record_counts_only_at_the_last_epoch(sim):
    W, rd, kw = sim(DUR, skip_last=True)
    assert drive(rd, kw)[0] == 0
    p = L.operator_dir(rd) / L.PIPELINE_VAL_FILE
    d = json.loads(p.read_text())
    d["train"]["2"]["skipped_by_design"] = True
    p.write_text(json.dumps(d))
    gate = L._pipeline_heldout_gate(L.Ctx(rd, 3, SPEC, rd / "oracle", **kw))
    ok, det = gate(2)
    assert not ok and "only the run's last epoch may skip it" in det and "epoch 3" in det


def test_skip_last_train_leaves_other_epochs_programs_unchanged(monkeypatch):
    monkeypatch.setattr(L, "_prompt_mode", lambda c: "template")
    spec = NS(**{**vars(SPEC), "n_epochs": 3})
    for mode in ("ao", "template"):
        for e in (1, 2):
            on = NS(prompt_mode=mode, pipeline_val=True, skip_last_train=True, spec=spec, epoch=e)
            off = NS(prompt_mode=mode, pipeline_val=True, spec=spec, epoch=e)
            assert _sig(L._steps_for(on)) == _sig(L._steps_for(off))
        last = L._steps_for(NS(prompt_mode=mode, pipeline_val=True, skip_last_train=True, spec=spec, epoch=3))
        plain = L._steps_for(NS(prompt_mode=mode, pipeline_val=True, spec=spec, epoch=3))
        assert [s.name for s in last] == [s.name for s in plain]
        for s in last:
            if s.name in L.PIPELINE_TRAIN_STEPS:
                assert s.verify is L._v_train_skipped_by_design and s.run is None
        # the serial order ignores the key (main refuses it there)
        serial = NS(prompt_mode=mode, skip_last_train=True, spec=spec, epoch=3)
        assert _sig(L._steps_for(serial)) == _sig(L._serial_steps_for(NS(prompt_mode=mode)))


def test_skip_last_train_runtime_key_is_validated():
    assert L._pipeline_runtime({}) == {}
    assert L._pipeline_runtime({"pipeline_val": True, "skip_last_train": True, "daytona_leg": {"pool": True}}) \
        == {"pipeline_val": True, "skip_last_train": True, "leg_pool": True}
    assert L._pipeline_runtime({"pipeline_val": False, "skip_last_train": False}) == \
        {"pipeline_val": False, "skip_last_train": False}
    # absent + pipeline_val true = true (the last epoch's train leg feeds no later evolver);
    # an explicit false is honoured; pipeline off adds nothing
    assert L._pipeline_runtime({"pipeline_val": True}) == {"pipeline_val": True, "skip_last_train": True}
    assert L._pipeline_runtime({"pipeline_val": True, "skip_last_train": False}) == \
        {"pipeline_val": True, "skip_last_train": False}
    assert L._pipeline_runtime({"pipeline_val": False}) == {"pipeline_val": False}
    for rt, msg in (({"skip_last_train": True}, "valid only with pipeline_val=true"),
                    ({"pipeline_val": False, "skip_last_train": True}, "valid only with pipeline_val=true"),
                    ({"pipeline_val": True, "skip_last_train": "true"}, "skip_last_train must be true or false"),
                    ({"pipeline_val": "true"}, "pipeline_val must be true or false")):
        with pytest.raises(ValueError, match=msg):
            L._pipeline_runtime(rt)


def test_skip_last_train_refuses_a_drive_past_the_last_epoch(sim, capsys):
    W, rd, kw = sim(DUR, skip_last=True)
    rc, _ = drive(rd, kw, until=4)
    out = capsys.readouterr().out
    assert rc == 1 and "skip_last_train" in out and "--until-epoch 3" in out
    assert not W.launches and not any(e[1] == "evolver_start" for e in W.events)
