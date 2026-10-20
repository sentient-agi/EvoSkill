#!/usr/bin/env python3
"""THE LEG PROCESS the orchestrator loop launches when the SOLVER IS CLAUDE CODE: one leg of Claude Code rollouts on
the run's ONE Daytona machine, graded in place, laid out as a leg the loop reads.

scripts/ssbv2_leg_cc.sh (RUNTIME leg_script) starts this program detached with `leg --tag <tag> ...`; the loop knows
the leg by that pid, its start time and the tag on its command line (its per-run dispatch record). While a leg runs,
its process also holds <run>/.machine/legs/<tag>.pid (cc_leg_lib.hold_leg_pid): the per-run record of the live leg,
and the reason a second driver of the same tag refuses.

WHAT ONE `leg` DOES, in order
  1. validate (no Daytona): the run's SPEC / INFRA / RUNTIME, the tag <NAME>_v<N>_<key>, the split (SPEC's file, sha),
     n_samples, the cache dir the loop hands over, the promoted skill set (rows, the Daytona probe record
     .operator/CC_SKILLS_PROBE.json) or a base leg, the run's ONE credential (cc_solver.read_credential: INFRA
     claude_code.token_file, else $EVOSKILL_CC_TOKEN_FILE -- a regular file, 0600 -- else $ANTHROPIC_API_KEY), the
     run's frozen plug-in (plugin_runner.verify_frozen: its task list and the user's grader), the data overlay, the
     grader's /data view (plugin_runner.material_layout); --check-only stops here (exit 0 / 4), --dry-run writes
     <run>/<tag>.DRYRUN.json and exits 7.
  2. the units: split[key] x N (sample-major, cc_solver_lib.train_selection), each prepared on the host exactly as
     cc_solver.py prepare does (the overlay's input bytes, the task's own text with its input and output paths);
     STATE.json (resume: graded samples kept and re-verified).
  3. JOIN the run's pool machine (cc_pool_host.join: attach or spawn the host, pick, lease, ack, identity, the
     idempotent installs); wipe this tag's leftover cc-* containers; the in-place grader (cc_pool_host.InPlaceGrader)
     and its install() BEFORE any rollout (the runner, the pinned plug-in files, the /data links, the leg's task
     files == the host's bytes; a GradeError FAILS the leg).
  4. the QUEUE (LegQueue: cc_solver.Queue with the skills expectations, the pool machine's retire_at as the deadline,
     the wait/resume hook): every completed unit's whole out.tgz is kept on the machine and one grade job enqueued
     BEFORE its container is wiped; a grade poller thread downloads results.jsonl every 60 s and writes each graded
     sample (its two records) at grade time. The score is the run's frozen grade(task, snapshot) (0..1); a grader
     failure is re-graded ONCE, and a second one is RECORDED as grader_failed (the leg continues); a `refused` line
     (a loop fault) FAILS the leg.
  5. the GUARD: a session REJECTED for a usage limit (its last rate_limit_info says status 'rejected', or it ended on
     HTTP 429) closes admission -> once nothing is in flight the driver WAITS until the reset that session reported
     (STATE.json / MACHINE.json record the wait; the lease renews; grading polls continue), then admits again, with
     no extra call to read the limit; a reset more than cc_leg_lib.WAIT_MAX_S away or past the leg's deadline
     (RUNTIME leg_timeout_s - 15 min) writes FAILED with that reason instead. A defect (init / thinking / credential
     hit / setup failures) ends the leg as FAILED. A session without a measured outcome (an api_error, a rejection
     included) is re-run once.
  6. the result: <run>/<tag>.json written atomically ONLY when every unit completed AND was graded AND every
     check passed (LegLayout.verify: STATE.json, both record copies and per_task agree sample by sample, the counts
     re-derive) -> exit 0. Otherwise <run>/<tag>.FAILED.json {INCOMPLETE, problems} -> exit 5.
     Always: containers wiped, the machine's units/work dirs cleaned, the lease released LAST. The machine stays.

Other subcommands: `probe-skills --run-dir R` (the Daytona probe that decides the skill placement, four haiku turns,
writes <run>/.operator/CC_SKILLS_PROBE.json), `status --run-dir R`.
Interpreter: the venv with the `daytona` extra (the Daytona SDK). No OpenRouter key is ever read for an SSB leg:
OPENROUTER_API_KEY is removed from this process's environment at start and any of the seven OpenRouter pins in it is
a refusal.

A run whose SPEC benchmark is "tau3" runs its units through the tau3 bridge (scripts/tau3_cc/tau3_leg.py, imported
only then): Tau3Box / poll_batch / judge / init_problems in place of the SSB ones, no workbook and no task message (the
bridge sends TAU3_FIRST_MESSAGE.md), the container wall = the bridge's agent time + 40 min (cc_leg.wall_min is not
read: the agent clock lives in the bridge), grading under /opt/tau3/venv/bin/python. The customer's OpenRouter key is
read ONCE per leg at run time (never by validate, --check-only or --dry-run) into memory and reaches each bridge only as
fd 4; the Claude credential reaches it on fd 3, as it reaches every SSB container.
RUNTIME cc_leg.concurrency caps the rollout containers of ALL the run's legs on the pool machine together
(LegQueue.admit; <pool root>/inflight/<holder>.json); --concurrency (the loop's EVO_CONC) stays the leg's own ceiling.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import signal
import statistics
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO), str(REPO / "scripts" / "cc_solver")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
if __name__ == "__main__":                       # one module object, whichever way this file is entered
    sys.modules.setdefault("cc_leg_driver", sys.modules["__main__"])

import cc_leg_lib as G  # noqa: E402  (the pure half: layout, record, leg JSON, skills, wait plan, STATE.json)
import cc_pool_host as H  # noqa: E402  (READ-ONLY reuse: join, PoolMachineHandle, InPlaceGrader, settings)
import cc_solver as C  # noqa: E402  (READ-ONLY reuse: Box, Queue, judge, poll_batch, read_credential)
import cc_solver_lib as L  # noqa: E402
import ssb_leg_daytona_loop as LL  # noqa: E402  (READ-ONLY reuse: leg_knobs, data overlay, task_index, Refused)
from src.orchestrator import data_overlay  # noqa: E402
from src.orchestrator import plugin_runner as PR  # noqa: E402  (material_layout: the grader's /data view)

ctl = L.ctl
q = shlex.quote
Refuse = C.Refuse

DRIVER = G.DRIVER                                # "scripts/daytona/cc_leg_driver.py" (the leg JSON's driver field)
EXIT_OK, EXIT_REFUSED, EXIT_FAILED, EXIT_DRYRUN = 0, 4, 5, 7
PROBE_RECORD = Path(".operator") / "CC_SKILLS_PROBE.json"
PROBE_SKILL = "probe-skill"
PROBE_SKILL_MD = ("---\nname: probe-skill\ndescription: Probe skill for the delivery check.\n---\n"
                  "When invoked, reply with the first line of the text you were given.\n")
PROBE_MSG_ONE = "Reply with exactly: OK"
PROBE_MSG_TWO = ("Use the Skill tool to invoke the skill named probe-skill, then reply with exactly the first line "
                 "of what the skill gave you.")
PROBE_MODEL = "haiku"                            # the skills probe: four cheap turns
FALLBACK_SKILLS_DIR = f"{ctl.EVO_HOME}/.claude/skills"   # the user skills dir: the probe's fallback placement
STOP_LEG_FILE = "STOP_LEG"                       # in the leg's store: aborts a waiting / running leg as FAILED
PRIORITY_HOLD_S = 30.0                           # a val leg held behind a live train leg re-checks this often
POOL_INFLIGHT_DIR = "inflight"                   # <pool root>/inflight/<holder>.json: each live leg's containers
TAU3 = "tau3"                                    # SPEC benchmark of a tau3 run (its units go through the tau3 bridge)
SUBSET_ENV = "EVOSKILL_CC_TASK_SUBSET"           # test-only: a comma list of task ids (a few-task live check)
SMOKE_MARK = "smoke"                             # SUBSET_ENV and an n_samples != SPEC's only for such a run dir name
OPENROUTER_PINS = ("EVOSKILL_SOLVER_PROVIDER_ORDER", "EVOSKILL_SOLVER_PROVIDER_ONLY", "EVOSKILL_SOLVER_ALLOW_FALLBACKS",
                   "EVOSKILL_SOLVER_REQUIRE_PARAMETERS", "EVOSKILL_SOLVER_PROVIDER_QUANTIZATIONS",
                   "EVOSKILL_SOLVER_EXPECT_PROVIDER", "EVOSKILL_SOLVER_EXPECT_MODEL")
TAG_RE = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)_(?:v(?P<epoch>[1-9][0-9]*)|(?P<base>base))_(?P<key>train|val|test)$")
GRADE_POLL_S = 60.0                              # results.jsonl downloaded this often
PAUSE_TICK_S = 60.0                              # a rate-limit wait sleeps in these ticks (STOP_LEG checked each)
LEG_DEADLINE_MARGIN_S = 900                      # leg_deadline = start + leg_timeout_s - this
GRADE_WAIT_MARGIN_S = 600                        # after the queue: grades must land before the deadline minus this
UNIT_RETRIES = 1                                 # an incomplete (api_error / died / poll_failed) unit is re-run once
JOB_TIMEOUT_S = 5400                             # plugin_grade_runner --job-timeout
GIB = 1 << 30


# ───────────────────────────────────────────────────────────────────────────── helpers
def log(msg: str) -> None:
    C.log(msg)                                   # stdout (nohup -> <run>/<tag>.log) + <store>/driver.log once opened


def read_json(p) -> dict:
    return json.loads(Path(p).read_text())


def parse_tag(tag: str) -> tuple:
    """(run name, epoch, key) of a leg tag: <NAME>_v<N>_<train|val> (a loop leg, epoch N >= 1) or
    <NAME>_base_<train|val> (the base leg, epoch 0: the solver with no skills, run by this driver under the same
    prompt delivery as every epoch leg)."""
    m = TAG_RE.match(tag or "")
    if not m:
        raise Refuse(f"tag {tag!r} is not <run>_v<N>_<train|val|test> or <run>_base_<train|val|test>")
    return m["name"], (0 if m["base"] else int(m["epoch"])), m["key"]


def stream_result(b) -> dict:
    p = Path(b.rd) / "stream.jsonl"
    s = L.parse_stream(p.read_text(errors="replace") if p.is_file() else "")
    return s["result"] or {}


def infra_error_of(r: dict) -> bool:
    """An API / auth / rate-limit failure ends with is_error + terminal_reason api_error (or an
    api_error_status) -- infrastructure, never a measured outcome."""
    return bool(r.get("is_error")) and (r.get("terminal_reason") == "api_error" or r.get("api_error_status") is not None)


def unit_completed(b) -> bool:
    """Box.result()'s `completed` predicate, evaluated BEFORE the wipe (the grade job is enqueued then): the box's
    own measured() (a tau3 box answers with its bridge's rc)."""
    return b.measured()


def tau3_mod():
    """scripts/tau3_cc/tau3_leg.py, imported only by a tau3 leg (an SSB leg imports nothing of it)."""
    p = str(REPO / "scripts" / "tau3_cc")
    if p not in sys.path:
        sys.path.insert(0, p)
    import tau3_leg
    return tau3_leg


def probe_argv(prompt: str, max_turns: int, model: str = PROBE_MODEL) -> list:
    """A short call with the probe's prompt, turn budget and model and the solver's own flags and env."""
    return ["claude", *L.ctl.CC_EXTRA_ARGS, "-p", prompt, "--model", model, "--max-turns", str(int(max_turns)), "--output-format",
            "stream-json", "--verbose", "--permission-mode", "bypassPermissions", "--strict-mcp-config",
            "--disallowedTools", ",".join(L.CC_DISALLOWED), "--thinking-display", L.THINKING_DISPLAY]


def probe_inner(prompt: str, max_turns: int, model: str = PROBE_MODEL) -> str:
    """The bash run inside the probe container as evo: cwd /evo/work (where Claude Code looks for the project
    skills), evo's own HOME, the credential on fd 3; the stream comes back on stdout."""
    texts = {"CC_PROBE_MSG": prompt}
    return "\n".join([L.env_exports(), f"cd {L.WS_CONTAINER}", L._b64_var("CC_PROBE_MSG", prompt),
                      f"{L.cred_fd_var()}=3 timeout 600 "
                      + L.shell_argv(probe_argv(prompt, max_turns, model), texts) + " 2>/tmp/cc_probe.stderr",
                      "echo CC_PROBE_RC=$?", "tail -c 1500 /tmp/cc_probe.stderr | sed 's/^/STDERR: /'",
                      "rm -f /tmp/cc_probe.stderr"])


def skill_lines(skills_dir) -> list:
    """One `[SKILL] <name>: <description first line>` per skill (loop._v_leg_loaded_promoted's log fallback)."""
    out = []
    for f in sorted(Path(skills_dir).glob("*/SKILL.md")):
        fm, _body = G.parse_frontmatter(f.read_text(errors="replace"))
        first = ((fm.get("description") or "").splitlines() or [""])[0]
        out.append(f"[SKILL] {f.parent.name}: {first}")
    return out


def _on_sigterm(signum, frame):
    """SIGTERM -> SystemExit, so LegRun.run's `finally` still wipes the containers and releases the lease."""
    raise SystemExit(128 + signum)


def leg_limits(conc: int, mem_max_gib: float) -> dict:
    """cc_solver_lib.limits_for(conc) with the per-container memory cap RUNTIME cc_leg declares (default: at most 30
    rollouts, 1 GiB each); /tmp stays inside the cap."""
    lim = L.limits_for(conc)
    cap = int(float(mem_max_gib) * GIB) // 4096 * 4096
    if lim["mem_max_bytes"] > cap:
        lim["mem_max_bytes"] = cap
        lim["tmp_bytes"] = min(int(lim["tmp_bytes"]), cap * 2 // 3 // 4096 * 4096)
    return lim


# ───────────────────────────────────────────────────────────────────────────── the boxes
class LegBox(C.Box):
    """cc_solver.Box for a loop leg: on a SKILL leg the promoted set's rows join the upload tgz
    (/evo/skills/<name>/..., root 0644/0755 under the read-only /evo bind), the workspace gets exactly ONE link
    /evo/work/.claude/skills -> /evo/skills (made after `evoctl extract`, before the handover counts links), and
    the listing inside must show the link + every skill file. A BASE leg's LegBox is cc_solver's Box byte for
    byte (no rows, no link, no .claude dir), so base rollouts see exactly the stock solver's workspace."""
    skill_leg = False
    grade_error = None                           # set by the driver when the unit's grade job could not be enqueued

    @classmethod
    def for_leg(cls, skill_rows, model=None) -> type:
        """The Box class of one leg: the rows and expectations as class attributes (Queue instantiates it with
        cc_solver's 6-argument signature). `model` = the run's SPEC solver_model (passed as --model and checked in
        init / session checks; cc_solver_lib.MODEL is only the default)."""
        rows = list(skill_rows or [])
        return type(cls.__name__, (cls,), {
            "model": model or L.MODEL, "skill_rows": rows, "skill_leg": bool(rows), "expect_link_count": 1 if rows else 0,
            "expect_links": G.expect_links_for(bool(rows)), "extra_dirs": tuple(sorted(G.extra_dirs_for(rows)))})

    def measured(self) -> bool:
        """A measured outcome (Box.result()'s `completed`): collected, claude's rc read, no api_error session."""
        return self.status == "collected" and self.rc is not None and not infra_error_of(stream_result(self))

    def skills_link_cmd(self) -> str:
        cl = f"{self.cdir}/fs{L.WS_CONTAINER}/.claude"
        link, vm = cl + "/skills", f"{self.cdir}/fs{G.SKILLS_VM}"
        return (f"set -e; test -d {q(vm)}; install -d -o {ctl.EVO_UID} -g {ctl.EVO_GID} -m 0755 {q(cl)}; "
                f"ln -s {q(G.SKILLS_VM)} {q(link)}; chown -h {ctl.EVO_UID}:{ctl.EVO_GID} {q(link)}; "
                f"test \"$(readlink {q(link)})\" = {q(G.SKILLS_VM)}; echo LINK_OK")

    def after_extract(self):
        if not self.skill_leg:
            return
        code, out = self.m.sh(self.skills_link_cmd(), timeout=60, what="skills link")
        self.rec["skills_link"] = {"link": G.SKILLS_LINK, "target": G.SKILLS_VM, "rows": len(self.skill_rows),
                                   "out": out.strip()[-200:]}
        self.save()
        self.fail_on("LINK_OK" not in out, f"skills link failed: {out[-300:]}")


class ProbeBox(LegBox):
    """The skills probe's container: no task file, the probe set placed as a skill leg places it, one-turn and
    two-turn haiku calls run synchronously (the stream on stdout), and the fallback placement written as evo."""

    def __init__(self, m, stamp: str, wall_s: int, skill_rows: list, probe_dir: Path):
        self.m = m
        self.unit = self.task = "skills-probe"
        self.rd = Path(probe_dir)
        self.rd.mkdir(parents=True, exist_ok=True)
        self.prep = {"unit": self.unit, "task_id": self.unit, "sample_idx": None, "spreadsheet_filename": None}
        self.um = ""
        self.rows = []
        rows = list(skill_rows)
        self.skill_rows, self.skill_leg = rows, bool(rows)
        self.expect_link_count = 1 if rows else 0
        self.expect_links = G.expect_links_for(bool(rows))
        self.extra_dirs = tuple(sorted(G.extra_dirs_for(rows)))
        self.cid = f"cc-probe-{stamp}-{uuid.uuid4().hex[:4]}"
        self.cdir = ctl.cdir(self.cid)
        self.wall_s = int(wall_s)
        self.spec = L.container_spec(self.cid, wall_s=wall_s, limits=L.PROBE_LIMITS)
        self.rec = {"role": "skills-probe", "unit": self.unit, "task": self.unit, "cid": self.cid,
                    "machine_run": m.dir.name, "sandbox_id": m.sid, "spec": self.spec,
                    "argv": probe_argv(PROBE_MSG_ONE, 1), "env": L.cc_env(), "skills": G.skill_names(rows)}
        self.off, self.fail, self.stopped, self.rc = 0, 0, None, None
        self.t_launch = self.t_done = None
        self.status, self.pl, self.stream_fh, self.guard = "new", None, None, None
        self.wiped = self.result_written = False
        self.res = None

    def call(self, tok: str, label: str, prompt: str, max_turns: int, model: str = PROBE_MODEL) -> dict:
        cmd = self.token_cmd(probe_inner(prompt, max_turns, model), cwd=L.WS_CONTAINER, detached=False, tok=tok)
        t0 = L.now_utc()
        code, out = self.m.sh(cmd, timeout=700, env={"CLAUDE_CODE_OAUTH_TOKEN": tok}, what=f"probe {label}",
                              retries=1)
        (self.rd / f"PROBE_{label}.jsonl").write_text(out)
        s = L.parse_stream(out)
        rec = {"label": label, "cid": self.cid, "prompt": prompt, "max_turns": int(max_turns), "model": model,
               "t_request_utc": t0, "t_response_utc": L.now_utc(), "rc_line": [ln for ln in out.splitlines()
                                                                                if ln.startswith("CC_PROBE_RC=")][-1:],
               "stderr_tail": [ln[8:] for ln in out.splitlines() if ln.startswith("STDERR: ")][-10:],
               "init": s["init"], "result": s["result"], "tool_uses": s["tool_uses"], "stream_text": out}
        self.rec.setdefault("calls", []).append({k: v for k, v in rec.items() if k != "stream_text"})
        self.save()
        return rec

    def remove_primary_link(self) -> None:
        """The primary placement taken away (host side: the link and the .claude dir in the workspace bind), so the
        fallback placement is measured ALONE."""
        cl = f"{self.cdir}/fs{L.WS_CONTAINER}/.claude"
        code, out = self.m.sh(f"set -e; rm -f {q(cl + '/skills')}; rmdir {q(cl)}; test ! -e {q(cl)}; echo RM_OK",
                              timeout=60, what="remove skills link")
        self.rec["primary_link_removed"] = out.strip()[-200:]
        self.save()
        self.fail_on("RM_OK" not in out, f"removing the primary placement failed: {out[-300:]}")

    def write_fallback(self, text: str) -> str:
        """The FALLBACK placement (4.2's user skills dir) written by `enter` as evo into the rw HOME."""
        import base64
        b64 = base64.b64encode(text.encode()).decode()
        d = f"{FALLBACK_SKILLS_DIR}/{PROBE_SKILL}"
        code, out = self.enter(["/bin/bash", "-c", f"set -e; mkdir -p {q(d)} && printf %s {b64} | base64 -d > "
                                f"{q(d + '/SKILL.md')} && ls -la {q(d)} && echo FB_OK"], cwd="/tmp",
                               what="fallback skill")
        self.rec["fallback_written"] = out.strip()[-300:]
        self.save()
        self.fail_on("FB_OK" not in out, f"fallback placement failed: {out[-300:]}")
        return d


# ───────────────────────────────────────────────────────────────────────────── the queue
class LegQueue(C.Queue):
    """cc_solver.Queue with the driver's hooks: the rejections noted for the rate-limit wait, the grade job enqueued
    between collect and wipe, and the completion verdict after the result is written."""

    def __init__(self, drv: "LegRun", *a, judge_fn=None, **kw):
        super().__init__(*a, **kw)
        self.drv = drv
        self.held = False
        self.capped = False                      # admission held to the pool-wide cap
        self.judge_fn = judge_fn or C.judge      # a tau3 leg judges with tau3_leg.judge

    def admit(self):
        """Train rollouts go first: a VAL leg admits no new unit while a TRAIN leg of the same run holds a live lease on
        the pool and still has units WAITING for a seat (train_legs_waiting); its units already in flight finish, and
        once the train leg has admitted all of its units, val takes the seats it frees. Every admission, and this
        leg's count of rollout containers, stays under the pool's cap (pooled_admit); a leg with no room and nothing in
        flight waits PRIORITY_HOLD_S (no busy loop)."""
        held = False
        if self.drv.key == "val" and self.guard["admitting"] and self.pending:
            live = self.drv.train_legs_waiting()
            if live:
                if not self.held:
                    log(f"PRIORITY: val admission held while train leg(s) have units waiting for a seat: {live}; "
                        f"{len(self.pending)} unit(s) wait, {len(self.inflight)} in flight finish")
                    self.held = True
                held = True
            elif self.held:
                log(f"PRIORITY: no train leg has units waiting; val admission resumes ({len(self.pending)} unit(s) pending)")
                self.held = False
        capped = False
        if getattr(self.drv, "pool_reg", None) is not None:
            capped = self.pooled_admit(held)
        elif not held:                           # a queue outside LegRun.run (no pool record): its own ceiling only
            super().admit()
        if self.inflight:
            return
        if held:
            time.sleep(PRIORITY_HOLD_S)
        elif capped:
            self.drv.sleep(PRIORITY_HOLD_S)

    def pooled_admit(self, held: bool) -> bool:
        """The concurrency cap is per MACHINE (two val legs once ran 59 at once under a per-leg cap): under the run's
        pool registry flock, the room = RUNTIME cc_leg.concurrency - the rollout containers the run's OTHER live legs
        hold (<pool root>/inflight/<holder>.json); this leg admits up to min(its own concurrency, the room) and
        records its own count. Returns True when the room, not the leg's own ceiling, holds a pending unit back."""
        drv = self.drv
        with drv.pool_reg.lock():
            others = drv.pool_others()
            room = int(drv.cc["concurrency"]) - others
            if not held:
                self.conc = max(0, min(drv.conc, room))
                super().admit()
            drv.pool_note(len(self.inflight))
        capped = (not held and self.guard["admitting"] and bool(self.pending) and len(self.inflight) >= self.conc
                  and self.conc < drv.conc)
        if capped != self.capped:
            log(f"POOL: admission held to the machine's cap {drv.cc['concurrency']}: other legs hold "
                f"{int(drv.cc['concurrency']) - room}, this leg {len(self.inflight)} in flight, {len(self.pending)} "
                f"pending" if capped else f"POOL: room again under the machine's cap ({len(self.pending)} pending)")
            self.capped = capped
        return capped

    def on_ended(self, b):
        g = self.judge_fn(b, None, self.expect_skills)       # the second slot (a retired threshold) is not read
        b.guard = g
        lr = g["last_rate_limit"] or {}
        self.drv.note_reading(b, g)
        self.readings.append({"utc": g["utc"], "unit": b.unit, "status": lr.get("status"),
                              "five_hour": (lr.get("five_hour") or {}).get("utilization"),
                              "seven_day": (lr.get("seven_day") or {}).get("utilization"),
                              "thinking_chars": g["thinking"]["chars"], "stop_reasons": g["stop_reasons"]})
        log(f"{b.unit}: guard reading 5h {self.readings[-1]['five_hour']} 7d {self.readings[-1]['seven_day']} "
            f"status {lr.get('status')}; thinking {g['thinking']['with_text']}/{g['thinking']['blocks']} blocks "
            f"with text, {g['thinking']['chars']} chars; reasons {g['stop_reasons']}")
        if g["stop_reasons"]:
            self.close("; ".join(g["stop_reasons"]), by=b.unit, kind=self.stop_kind(g["stop_reasons"]))
        b.status_ended = b.status
        self.futs[self.ex.submit(self._finish, b)] = (b, "collect")

    def _setup(self, b):
        self.drv.on_setup(b, "setup")
        super()._setup(b)
        if b.status == "running":
            self.drv.on_setup(b, "running")

    def _teardown(self, b):
        """A box that never launched: its setup failed (re-queued once) or admission closed while it was being set
        up (re-queued as it is: a rate-limit wait that ends with `reopen` must run it)."""
        super()._teardown(b)
        self.drv.on_not_launched(b)

    def _finish(self, b):
        try:
            b.collect(getattr(b, "tok_used", None) or self.tok)
            self.drv.on_collected(b)             # the snapshot kept + the grade job, BEFORE the wipe
        finally:
            b.wiped = b.wipe()
            b.res = b.result()
            b.result_written = True
            self.drv.on_result(b)


# ───────────────────────────────────────────────────────────────────────────── one leg
class LegRun:
    """One leg, end to end. Every collaborator is injectable (the tests drive the same code path with fakes):
    join_fn (cc_pool_host.join), box_cls (LegBox.for_leg), grader_cls (cc_pool_host.InPlaceGrader), poll_fn
    (cc_solver.poll_batch), clock / sleep, plugin (the run's frozen plug-in in cc_leg_lib.frozen_plugin's shape;
    default: read and verified from the run dir), judge_fn (cc_solver.judge), init_problems_fn
    (cc_solver_lib.init_problems) and, for a tau3 leg, key_fn (the customer's OpenRouter key; default
    tau3_leg.read_openrouter_key); a tau3 leg's defaults are tau3_leg's."""

    def __init__(self, a, *, join_fn=None, box_cls=None, grader_cls=None, poll_fn=None,
                 clock=None, sleep=None, plugin=None, env=None, poll_s=None, grade_poll_s=None, judge_fn=None,
                 init_problems_fn=None, key_fn=None):
        self.a = a
        self.poll_s = C.POLL_S if poll_s is None else float(poll_s)           # the queue's stream poll (tests: 0)
        self.grade_poll_s = GRADE_POLL_S if grade_poll_s is None else float(grade_poll_s)
        self.join_fn = join_fn or H.join
        self.given = {"box_cls": box_cls, "poll_fn": poll_fn, "judge_fn": judge_fn,
                      "init_problems_fn": init_problems_fn}                  # a tau3 leg keeps what a test injects
        self.box_base = box_cls or LegBox
        self.grader_cls = grader_cls or H.InPlaceGrader
        self.poll_fn = poll_fn or C.poll_batch
        self.judge_fn = judge_fn or C.judge
        self.init_problems_fn = init_problems_fn or L.init_problems
        self.key_fn = key_fn
        self.tau3, self.T, self.tau3_block, self.tau3_ctx = False, None, None, None
        self.pool_reg, self.holder, self.pool_n = None, None, None
        self.clock = clock or time.time
        self.sleep = sleep or time.sleep
        self.plugin_given = plugin
        self.env = os.environ if env is None else env
        self.problems: list = []
        self.fatal: list = []
        self.samples: dict = {}                  # unit -> {"score": float | None (grader_failed), "turn_exhausted"}
        self.results: dict = {}                  # unit -> RESULT.json dict (completed sessions)
        self.jobs: dict = {}                     # grade job id -> unit
        self.early: list = []                    # grade lines that landed before their unit's RESULT (next poll)
        self.queue_ended = False                 # after the queue every RESULT is written: no line waits any more
        self.failed_units: set = set()
        self.retries: dict = {}
        self.pauses: list = []                   # the rate-limit waits
        self.rejections: list = []               # sessions rejected for a usage limit since admission last opened
        self.rollout_s: list = []
        self.cred = None                         # cc_account.Credential (kind, token file); the secret is self.tok
        self.cred_from = None                    # a token file's path or the variable's name, for records
        self.tok = None
        self.pid_fh = None                       # <run>/.machine/legs/<tag>.pid, held while the leg runs
        self.handle = self.lease = self.grader = self.queue = None
        self.state = None
        self.kept: list = []
        self.requeue: list = []
        self.glock = threading.Lock()            # grader calls come from the queue's workers and the poller
        self.slock = threading.Lock()            # STATE.json writes
        self._poll_stop = threading.Event()
        self._poller = None
        self.t0 = None
        self.leg_deadline_ts = None
        self.log_fh = None

    # ── validate ─────────────────────────────────────────────────────────────────────────────────────────────────
    def validate(self) -> dict:
        a = self.a
        self.env.pop("OPENROUTER_API_KEY", None)
        pins = [k for k in OPENROUTER_PINS if self.env.get(k)]
        if pins:
            raise Refuse(f"OpenRouter pin(s) {pins} set in the environment: a Claude Code leg reads no provider pin "
                         f"(a DeepSeek leftover) -- unset them")
        rd = Path(a.run_dir).resolve()
        for f in ("SPEC.json", "INFRA.json", "RUNTIME.json"):
            if not (rd / f).is_file():
                raise Refuse(f"no {f} in {rd}")
        self.run_dir, self.name = rd, rd.name
        self.smoke = SMOKE_MARK in rd.name
        sp = G.spec_of(rd)
        if sp.get("solver_harness") != "claude_code":
            raise Refuse(f"SPEC solver_harness {sp.get('solver_harness')!r} != 'claude_code'")
        if not str(sp.get("solver_model") or "").startswith("claude-"):
            raise Refuse(f"SPEC solver_model {sp.get('solver_model')!r} is not a Claude model")
        if sp.get("name") != rd.name:
            raise Refuse(f"SPEC name {sp.get('name')!r} != the run dir name {rd.name!r}")
        self.spec, self.model = sp, str(sp["solver_model"])
        self.tau3 = sp.get("benchmark") == TAU3
        self.budget = int(sp.get("agent_turns") or L.MAX_TURNS)
        try:
            inf, self.infra_record = LL.load_infra(rd)        # validated (infra.load), as the pool host reads it
        except LL.Refused as e:
            raise Refuse(str(e))
        if inf is None or (inf.get("solver") or {}).get("harness") != "claude_code":
            raise Refuse(f"INFRA solver.harness {((inf or {}).get('solver') or {}).get('harness')!r} != 'claude_code'")
        self.inf = inf
        self.token_file_declared = (inf.get("claude_code") or {}).get("token_file") or None
        rt = read_json(rd / "RUNTIME.json")
        self.rt = rt
        try:
            self.cc = H.cc_leg_settings(rd)
            self.dl = LL.load_runtime_block(rd)
        except LL.Refused as e:
            raise Refuse(str(e))
        lt = rt.get("leg_timeout_s")
        if not isinstance(lt, int) or isinstance(lt, bool) or lt <= LEG_DEADLINE_MARGIN_S + 600:
            raise Refuse(f"RUNTIME leg_timeout_s {lt!r}: an integer well above {LEG_DEADLINE_MARGIN_S} s")
        self.leg_timeout_s = int(lt)
        # the tag
        name, epoch, key = parse_tag(a.tag)
        if name != rd.name:
            raise Refuse(f"tag {a.tag!r} names run {name!r}, not {rd.name!r}")
        if key != a.key:
            raise Refuse(f"tag {a.tag!r} is a {key} leg but --key says {a.key!r}")
        self.tag, self.epoch, self.key = a.tag, epoch, key
        if (rd / f"{a.tag}.json").exists():
            raise Refuse(f"{rd / (a.tag + '.json')} exists: that leg is measured (never overwritten)")
        if (rd / "HOLD_LEGS").exists() and not a.check_only and not a.dry_run:
            raise Refuse(f"{rd / 'HOLD_LEGS'} is present: legs are on hold")
        # the split
        split = Path(a.split).resolve()
        if split != Path(str(sp.get("split_path"))).resolve():
            raise Refuse(f"--split {split} != SPEC split_path {sp.get('split_path')}")
        if L.sha256_file(split) != sp.get("split_sha256"):
            raise Refuse(f"{split.name} does not hash to SPEC split_sha256")
        self.split_path, self.split_sha256 = split, str(sp["split_sha256"])
        n = int(a.n_samples)
        if n != int(sp.get("n_samples") or 0) and not self.smoke:
            raise Refuse(f"--n-samples {n} != SPEC n_samples {sp.get('n_samples')}")
        self.n_samples = n
        # the loop hands every leg EVO_CONC=30; RUNTIME cc_leg.concurrency is the run's ceiling (2 on the smoke dir)
        conc = int(a.concurrency)
        if conc < 1:
            raise Refuse(f"--concurrency {conc} < 1")
        self.conc_requested = conc
        self.conc = min(conc, int(self.cc["concurrency"]))
        env_model = self.env.get("EVOSKILL_SOLVER_MODEL")
        if env_model and env_model != self.model:
            raise Refuse(f"EVOSKILL_SOLVER_MODEL {env_model!r} != SPEC solver_model {self.model!r}")
        # the cache dir the loop hands over (one per epoch, shared by train and val)
        cd = Path(a.cache_dir)
        if not cd.is_absolute():
            raise Refuse(f"--cache-dir {a.cache_dir!r} must be absolute (the loop's EVOSKILL_CACHE_DIR)")
        want = G.cache_dir_for(sp["name"], epoch).name
        if cd.name != want:
            raise Refuse(f"--cache-dir {cd.name!r} is not the epoch's cache dir name {want!r}")
        self.cache_dir = cd.resolve()
        # the promoted set (a skill leg) or none (a base leg)
        self.skills_dir = None
        self.skill_rows: list = []
        self.expect_skills: list = []
        self.judge_skills: list | str = []       # a base leg's sessions list no skill; a skill leg's: whatever CC loads
        self.probe_rec = None
        if a.skills_dir:
            sd = Path(a.skills_dir).resolve()
            want_sd = (rd / f"skills_v{epoch}").resolve()
            if sd != want_sd:
                raise Refuse(f"--skills-dir {sd} is not the epoch's promoted dir {want_sd}")
            try:
                self.skill_rows = G.skill_rows(sd)
            except ValueError as e:
                raise Refuse(f"skill set {sd.name}: {e}")
            # Claude Code decides what a skill is: the loop has no SKILL.md policy of its own and does not refuse an
            # EMPTY set (the solver then runs with no skills). The names are a record; each session lists whatever
            # Claude Code itself loads (L.ANY_SKILLS).
            self.expect_skills = G.skill_names(self.skill_rows)
            self.judge_skills = L.ANY_SKILLS if self.skill_rows else []
            self.skills_dir = sd
            self.probe_rec = self.probe_record_check(rd)
        # the run's one credential (checked here; its value is read again at run time and never kept by validate)
        _secret, self.cred, self.cred_from = C.read_credential(self.token_file_declared, self.env)
        del _secret
        if not H.CC_LOCAL.is_file():
            raise Refuse(f"{H.CC_LOCAL}: the pinned Claude Code {ctl.CC_VERSION} linux-x64 binary is not on this host")
        # the frozen plug-in, the overlay, the knobs, the grader's /data view
        # The run's frozen task list, not the loop's own reading of the dataset: the whole frozen plug-in, every pin
        # re-hashed (cc_leg_lib.frozen_plugin -> plugin_runner.verify_frozen), since the legs execute its code (the
        # user's grade(task, snapshot)).
        try:
            self.plugin = self.plugin_given if self.plugin_given is not None else G.frozen_plugin(rd)
            self.task_rows = {t["id"]: t for t in self.plugin["tasks"]}
            self.questions, self.idx = G.tasks_json_questions(self.plugin["tasks"])
        except (OSError, ValueError, KeyError) as e:
            raise Refuse(f"the run's frozen plug-in is not usable ({type(e).__name__}: {e})")
        # The user's material at the /data paths the adapter saw (the ONE definition the
        # adapter upload uses), linked on the machine by the grader's install(), so a grader written against it runs
        try:
            self.layout_map = PR.material_layout(Path(str(inf["data"]["root"])), self.split_path, REPO)
        except (KeyError, TypeError, ValueError) as e:
            raise Refuse(f"the grader's /data view: {type(e).__name__}: {e}")
        try:
            self.overlay = LL.data_overlay_from_infra(inf, self.env, set_process_env=True)
            self.knobs = LL.leg_knobs(inf, self.env)
        except LL.Refused as e:
            raise Refuse(str(e))
        # the units: split[key] x N, sample-major
        try:
            sel = L.train_selection(split, samples=n, key=key, expect_sha256=self.split_sha256)
        except ValueError as e:
            raise Refuse(f"selection: {e}")
        task_ids = list(sel["task_ids"])
        subset = self.env.get(SUBSET_ENV)
        if subset:
            if not self.smoke:
                raise Refuse(f"{SUBSET_ENV} is accepted only for a run dir whose name contains {SMOKE_MARK!r}")
            want_t = [t.strip() for t in subset.split(",") if t.strip()]
            off = [t for t in want_t if t not in task_ids]
            if off or not want_t:
                raise Refuse(f"{SUBSET_ENV} names task(s) outside split[{key!r}]: {off}")
            task_ids = [t for t in task_ids if t in want_t]
        self.task_ids = task_ids
        self.units = [u["unit"] for u in sel["units"] if u["task"] in set(task_ids)]
        missing = [t for t in task_ids if t not in self.questions or t not in self.idx]
        if missing:
            raise Refuse(f"dataset lacks task(s) {missing[:4]}")
        # the plan
        self.wall_s = int(self.cc["wall_min"]) * 60
        self.instruction = L.SOLVER_PREAMBLE           # bare metal: nothing precedes the task's message
        if self.tau3:
            # tau3: the units go through the bridge (tau3_leg). Its only limit is 20
            # minutes of AGENT time, inside the bridge; the container wall is a backstop for a hung process
            # (agent_time_s + 40 min), never cc_leg.wall_min. The bucket is keyed by the first message's template.
            T = self.T = tau3_mod()
            self.tau3_block = T.leg_settings(self)
            self.wall_s = int(self.tau3_block["hard_stop_s"])
            self.instruction = T.HB.FIRST_MESSAGE_TEMPLATE
            self.box_base = self.given["box_cls"] or T.Tau3Box
            self.poll_fn = self.given["poll_fn"] or T.poll_batch
            self.judge_fn = self.given["judge_fn"] or T.judge
            self.init_problems_fn = self.given["init_problems_fn"] or T.init_problems
        self.limits = leg_limits(self.conc, self.cc["mem_max_gib_per_container"])
        # the machine time the leg leases: one rollout's wall + the longest rate-limit wait + 30 min
        self.need_s = float(self.wall_s + G.WAIT_MAX_S + 1800)
        self.store = G.store_for(self.name, self.tag, self.dl["store_root"]).resolve()
        self.tasks_dir = G.tasks_dir_for(self.tag, self.dl["tasks_root"]).resolve()
        self.bucket = G.bucket_for(self.model, self.instruction, self.skills_dir)
        self.layout = G.LegLayout(rd, self.tag, key, epoch, self.bucket, self.skills_dir, self.store, self.cache_dir,
                                  self.tasks_dir, model=self.model, n_samples=n, budget=self.budget,
                                  task_ids=task_ids, split_path=split, questions=self.questions)
        # resume?
        st = G.state_read(self.store / "STATE.json")
        if st is not None and (st.get("tag") != self.tag or st.get("bucket") != self.bucket):
            raise Refuse(f"{self.store / 'STATE.json'} belongs to tag {st.get('tag')!r} bucket {st.get('bucket')!r}, "
                         f"not this leg ({self.tag}, {self.bucket})")
        self.prev_state = st
        self.kept, self.requeue = G.resume_units(st, self.units) if st else ([], list(self.units))
        return self.plan()

    def probe_record_check(self, rd: Path) -> dict:
        p = rd / PROBE_RECORD
        if not p.is_file():
            raise Refuse(f"no skills probe record at {p}: run `cc_leg_driver.py probe-skills --run-dir {rd}` first "
                         f"(a skill leg needs the primary placement proven on Daytona)")
        try:
            rec = read_json(p)
        except ValueError as e:
            raise Refuse(f"{p} is not JSON: {e}")
        prim = rec.get("primary") or {}
        if prim.get("ok") is not True:
            raise Refuse(f"{p}: the primary skill placement did not pass ({prim.get('problems')}); the fallback "
                         f"placement (the user skills dir) is not supported by the container controller")
        if rec.get("cc_version") != ctl.CC_VERSION:
            raise Refuse(f"{p} was made with Claude Code {rec.get('cc_version')!r}, not {ctl.CC_VERSION}")
        return {"path": str(p), "utc": rec.get("utc"), "sandbox_id": rec.get("sandbox_id")}

    def plan(self) -> dict:
        mj = None
        try:
            from src.orchestrator import pool_container_evolver as pce
            regf = pce.pool_machines_file(self.run_dir)
            if regf.is_file():
                m = read_json(regf)
                mj = {"file": str(regf), "host_state": m.get("host_state"), "current": m.get("current"),
                      "host_pid": m.get("host_pid"), "machines": {n: (r or {}).get("state")
                                                                  for n, r in (m.get("machines") or {}).items()}}
        except Exception as e:  # noqa: BLE001
            mj = {"error": f"{type(e).__name__}: {e}"}
        p = {"driver": DRIVER, "tag": self.tag, "key": self.key, "epoch": self.epoch,
             "run_dir": str(self.run_dir), "model": self.model, "budget": self.budget,
             "n_tasks": len(self.task_ids), "n_samples": self.n_samples, "n_units": len(self.units),
             "units_kept_from_earlier_attempt": len(self.kept), "units_to_run": len(self.requeue),
             "concurrency": self.conc, "concurrency_requested": self.conc_requested,
             "concurrency_ceiling": int(self.cc["concurrency"]), "wall_min": self.wall_s // 60, "limits": self.limits,
             "need_s": self.need_s, "rate_limit_wait_max_s": G.WAIT_MAX_S, "leg_timeout_s": self.leg_timeout_s,
             "store": str(self.store), "cache_dir": str(self.cache_dir), "tasks_dir": str(self.tasks_dir),
             "bucket": self.bucket, "skills_dir": str(self.skills_dir) if self.skills_dir else None,
             "skills": self.expect_skills, "skills_probe": self.probe_rec,
             "credential": {"kind": self.cred.kind, "from": self.cred_from},
             "split_file": str(self.split_path), "split_sha256": self.split_sha256,
             "data_overlay": str(self.overlay) if self.overlay else None, "knobs": self.knobs,
             "pool_machine": mj, "smoke": {"run": self.smoke, "task_subset": self.env.get(SUBSET_ENV) or None},
             "written_utc": L.now_utc()}
        if self.tau3:
            p["tau3"] = dict(self.tau3_block)             # the bridge, the first message, the clock
        return p

    def dry_run(self) -> int:
        p = self.run_dir / f"{self.tag}{G.DRYRUN_SUFFIX}"
        G.write_json_atomic(p, {"DRY_RUN": True, **self.plan()})
        print(f"DRY RUN: {p} written; nothing created, uploaded or launched")
        return EXIT_DRYRUN

    # ── the units on the Mac (cc_solver.py prepare's layout) ─────────────────────────────────────────────────────
    def prepare_units(self, units: list) -> None:
        """<store>/runs/<task_>__s<j>/{PREP.json, user_message.txt} + <store>/ws/<task_>__s<j>/spreadsheets/<input>
        (the overlay's bytes): exactly what cc_solver.py prepare writes, so cc_solver.Box runs it unchanged. The
        user message = the task's own text + its input and output paths (L.PROMPT_DELIVERY). Never a reference workbook
        (check_upload_set). A tau3 leg's units carry no file and no message (tau3_leg.prepare_units)."""
        if self.tau3:
            return self.T.prepare_units(self, units)
        golden_shas, golden_paths = {}, []
        for tid, info in self.idx.items():
            if tid in set(self.task_ids):
                for p in {Path(info["golden"]), data_overlay.source(info["golden"], self.overlay)}:
                    if Path(p).is_file():
                        golden_shas[L.sha256_file(p)] = tid
                golden_paths.append(str(info["golden"]))
        for unit in units:
            task, k0 = G.sample_index(unit)
            j = k0 + 1
            qd = json.loads(self.questions[task])
            canonical = qd["spreadsheet_path"]
            src = data_overlay.source(canonical, self.overlay)
            if not Path(src).is_file():
                raise Refuse(f"{task}: input {src} missing")
            ud = L.unit_dir_name(unit)
            ws = self.store / "ws" / ud
            if ws.exists():
                shutil.rmtree(ws)
            (ws / "spreadsheets").mkdir(parents=True)
            dst = ws / "spreadsheets" / Path(canonical).name
            shutil.copyfile(src, dst)
            rows = L.upload_rows(ws)
            in_sha = L.sha256_file(src)
            probs = L.check_upload_set(rows, spreadsheet_filename=qd["spreadsheet_filename"], input_sha256=in_sha,
                                       golden_shas=golden_shas, golden_paths=golden_paths)
            if probs:
                raise Refuse(f"{unit}: upload set: {probs}")
            um = L.task_user_message(qd["instruction"], qd["spreadsheet_filename"])
            rd = L.run_dir(self.store, unit)
            rd.mkdir(parents=True, exist_ok=True)
            (rd / "user_message.txt").write_text(um)
            prep = {"schema": L.SCHEMA + ":prep", "unit": unit, "task_id": task, "sample_idx": j,
                    "safe_id": L.safe_id(task), "unit_dir": ud, "category": qd["category"],
                    "spreadsheet_filename": qd["spreadsheet_filename"], "canonical_input": canonical,
                    "input_source": str(src), "overlay_applied": str(src) != os.path.normpath(canonical),
                    "input_sha256": in_sha, "workspace": str(ws), "workspace_container": L.WS_CONTAINER,
                    "output_dir_container": f"{L.WS_CONTAINER}/output", "upload_rows": rows, "upload_problems": [],
                    "golden_shas_checked": len(golden_shas),
                    "user_message_sha256": L.sha256_bytes(um.encode()),
                    "user_message_source": "the task's instruction (dataset) + its input and output paths "
                                           "(cc_solver_lib.task_user_message)",
                    "instruction_sha256": L.sha256_bytes(self.instruction.encode()),
                    "prompt_delivery": L.PROMPT_DELIVERY,
                    "question": qd, "leg": {"tag": self.tag, "key": self.key, "epoch": self.epoch,
                                            "skills_dir": str(self.skills_dir) if self.skills_dir else None},
                    "data_overlay": {"patches_json": str(self.overlay) if self.overlay else None},
                    "created_utc": L.now_utc()}
            L.jdump(rd / "PREP.json", prep)

    # ── STATE.json ───────────────────────────────────────────────────────────────────────────────────────────────
    def state_write(self) -> None:
        with self.slock:
            self.state["guard"] = dict(self.queue.guard) if self.queue is not None else self.state.get("guard", {})
            self.state["pauses"] = list(self.pauses)
            self.state["problems"] = list(self.problems) + list(self.fatal)
            G.state_write(self.store / "STATE.json", self.state)

    def state_unit(self, unit: str, **fields) -> None:
        with self.slock:
            G.state_set_unit(self.state, unit, **fields)
        self.state_write()

    # ── hooks the queue calls ────────────────────────────────────────────────────────────────────────────────────
    def note_reading(self, b, g: dict) -> None:
        """A rejected session's reading is kept for the wait (rate_limit_wait reads the reset it reported)."""
        why = [r for r in g.get("stop_reasons") or [] if r.startswith(("quota ", "HTTP 429"))]
        if why:
            self.rejections.append({"unit": b.unit, "utc": g.get("utc"), "reasons": why,
                                    "brief": g.get("last_rate_limit")})
        if b.t_launch and b.t_done:
            self.rollout_s.append(float(b.t_done - b.t_launch))

    def on_setup(self, b, phase: str) -> None:
        """The queue's setup thread: "setup" when the container is being prepared, "running" once claude runs."""
        self.state_unit(b.unit, state=phase, cid=b.cid)

    def on_collected(self, b) -> None:
        """Between collect and wipe: a completed session's whole out.tgz is kept on the
        machine (its sha == the Mac's runs/<ud>/container_out.tgz, the bytes collect downloaded) and its grade job
        enqueued; an incomplete one (api_error, the wall, a dead container) is left for on_result to re-run.
        EVERY completed unit gets a job: the run's grade(task, snapshot) decides what a rollout that saved nothing
        scores (the loop has no not_eligible of its own)."""
        try:
            if not unit_completed(b):
                return
            task, k0 = G.sample_index(b.unit)
            # A relaunched leg re-runs a unit whose earlier attempt may already have a result line
            # on the machine under the plain id: the runner's RESUME skips a job_id that is already in
            # results.jsonl, so the plain id would hand attempt 1's score to attempt 2's session. The id is
            # therefore unique per attempt ("#a<n>", n >= 2; the worker's JOB_ID_RE admits '#', as the regrade
            # suffix does), and poll() ignores the stale line because its id was never sent by this process.
            att = int((self.state or {}).get("attempt") or 1)
            jid = H.job_id_for(task, k0) + (f"#a{att}" if att > 1 else "")
            mac_tgz = Path(b.rd) / "container_out.tgz"
            snap = L.sha256_file(mac_tgz)
            with self.glock:
                r = self.grader.enqueue_unit(job_id=jid, task_id=task, cdir=b.cdir, snapshot_sha256=snap,
                                             mac_tgz=mac_tgz, cid=b.cid, unit=b.unit, collected_utc=L.now_utc())
            self.jobs[jid] = b.unit
            self.state_unit(b.unit, grade_job_id=jid, snapshot_sha256=snap)
            log(f"{b.unit}: grade job {jid} enqueued (snapshot {snap[:12]} kept "
                f"{'RE-UPLOADED' if r['kept'].get('reuploaded') else 'ok'}; worker "
                f"{'launched' if r['worker'].get('launched') else 'running' if r['worker'].get('running') else 'idle'})")
        except (H.GradeError, H.HandleError, Refuse, OSError) as e:
            b.grade_error = f"{type(e).__name__}: {e}"
            log(f"{b.unit}: grade job NOT enqueued: {b.grade_error}")

    def on_result(self, b) -> None:
        res = b.res or {}
        unit = b.unit
        if res.get("completed"):
            if getattr(b, "grade_error", None):
                self.fail(f"{unit}: grading: {b.grade_error}", by=unit)
                self.state_unit(unit, state="failed", result=True)
            else:
                self.state_unit(unit, state="collected", result=True)
            self.results[unit] = res             # AFTER the STATE row: poll_grades takes a unit's line only from here
            return
        n = self.retries.get(unit, 0)
        why = (f"status {res.get('status')!r} rc {res.get('claude_rc')!r} subtype {res.get('subtype')!r} "
               f"terminal {res.get('terminal_reason')!r} api_error_status {res.get('api_error_status')!r}")
        kind = (self.queue.guard.get("kind") if self.queue is not None else None)
        if n < UNIT_RETRIES and not self.fatal and kind != "defect" and not getattr(self.queue, "stopping", False):
            self.retries[unit] = n + 1
            self.queue.pending.appendleft(unit)
            self.state_unit(unit, state="queued", result=False)
            log(f"{unit}: not a measured outcome ({why}); re-queued (attempt {n + 2} of {UNIT_RETRIES + 1})")
            return
        self.fail(f"{unit}: not completed after {n + 1} attempt(s): {why}", by=unit)
        self.state_unit(unit, state="failed", result=False)

    def on_not_launched(self, b) -> None:
        """The queue tore a box down without a launch: `not_launched` (the guard closed meanwhile) goes back to
        the front of the queue unchanged; a setup failure is re-tried once, then fatal."""
        unit = b.unit
        if b.status == "not_launched":
            self.queue.pending.appendleft(unit)
            self.state_unit(unit, state="queued", cid=None)
            log(f"{unit}: set up but not launched (admission closed); re-queued")
            return
        n = self.retries.get(unit, 0)
        why = f"setup failed: {b.rec.get('setup_error') or b.rec.get('refused') or b.status}"
        kind = self.queue.guard.get("kind") if self.queue is not None else None
        if n < UNIT_RETRIES and not self.fatal and kind != "defect" and not getattr(self.queue, "stopping", False):
            self.retries[unit] = n + 1
            self.queue.pending.appendleft(unit)
            self.state_unit(unit, state="queued", cid=None)
            log(f"{unit}: {why}; re-queued (attempt {n + 2} of {UNIT_RETRIES + 1})")
            return
        self.fail(f"{unit}: {why} (after {n + 1} attempt(s))", by=unit)
        self.state_unit(unit, state="failed")

    def fail(self, msg: str, by: str | None = None) -> None:
        """A fatal problem: recorded, admission closed as a defect (in-flight units finish; the leg FAILS)."""
        self.fatal.append(msg)
        if by:
            self.failed_units.add(by)
        log(f"FATAL: {msg}")
        if self.queue is not None:
            self.queue.close(msg, by=by, kind="defect")

    # ── samples at grade time ────────────────────────────────────────────────────────────────────────────────────
    def unit_inputs(self, unit: str) -> tuple:
        """(the unit's stream, its user message) from its run dir: the first message the tau3 bridge
        actually sent (runs/<ud>/first_message.txt, written at collect) when the unit has one; an SSB unit never does."""
        rd = L.run_dir(self.store, unit)
        stream = (rd / "stream.jsonl").read_text(errors="replace") if (rd / "stream.jsonl").is_file() else ""
        fm = rd / "first_message.txt"
        return stream, (fm if fm.is_file() else rd / "user_message.txt").read_text()

    def write_graded_sample(self, unit: str, res: dict, line: dict, score, created_at) -> dict:
        """The unit's two record copies at grade time. inline_score = the grader's score
        (None for GRADER_FAILED); trace.cc carries the accepted line's status, score and reason and the graded
        snapshot's sha. No DIFF.txt and no snapshot dirs: the graded snapshot is runs/<ud>/container_out.tgz."""
        stream, um = self.unit_inputs(unit)
        task, _k0 = G.sample_index(unit)
        cc = {"cid": res.get("cid"), "unit": unit, "subtype": res.get("subtype"), "terminal_reason": res.get("terminal_reason"),
              "usage": res.get("usage"), "modelUsage": res.get("modelUsage"), "thinking": res.get("thinking"),
              "rate_limit_last": (res.get("rate_limit") or {}).get("last"), "grade_job_id": line.get("job_id"),
              "graded_utc": G.naive_utc(created_at), "attempt": self.state.get("attempt") if self.state else 1,
              "grade_status": line.get("status"), "grade_score": score, "grade_reason": line.get("reason"),
              "snapshot_sha256": line.get("snapshot_sha256")}
        trace = G.build_trace(stream_text=stream, user_message=um, result=res,
                              score=None if score == G.GRADER_FAILED else score, output_saved=bool(res.get("output")),
                              model=self.model, max_turns=self.budget, cc=cc)
        w = self.layout.write_sample(unit, trace, created_at, question=self.questions[task])
        self.samples[unit] = {"score": trace["inline_score"], "turn_exhausted": bool(trace["budget_exhausted"])}
        return w

    def accept_grade(self, unit: str, line: dict, score) -> None:
        """A unit's final line -> its sample and its STATE row (state graded, score = the number from 0 to 1, or
        GRADER_FAILED after a failed re-grade)."""
        res = self.results.get(unit)
        if res is None:
            self.fail(f"{unit}: a grade line arrived for a unit without a completed RESULT", by=unit)
            return
        created_at = line.get("ended_utc") or L.now_utc()
        try:
            w = self.write_graded_sample(unit, res, line, score, created_at)
        except (ValueError, OSError, KeyError) as e:
            self.fail(f"{unit}: sample: {type(e).__name__}: {e}", by=unit)
            return
        self.state_unit(unit, state="graded", score=score, graded_utc=L.now_utc(), result=True)
        log(f"{unit}: graded {score} ({line.get('job_id')}, {line.get('seconds')}s); record {w['stem']}")

    def poll_grades(self) -> int:
        """Download results.jsonl; a new line -> its sample, decided by the runner's score_of: ok -> the score; grader_failed -> ONE re-grade, and a second grader_failed is RECORDED (the leg
        continues); refused (a loop fault: the snapshot sha, an unknown task, the staged plug-in, the extraction) ->
        the leg FAILS. Returns the number of new lines."""
        with self.glock:
            try:
                new = self.grader.poll()
                if self.grader.pending():
                    self.grader.ensure_worker()          # a worker that exited with jobs left is relaunched
            except (H.GradeError, H.HandleError, Refuse, OSError) as e:
                log(f"grade poll failed: {type(e).__name__}: {e}")
                return 0
            lines, self.early = self.early + new, []
        for line in lines:
            jid = str(line.get("job_id"))
            unit = self.jobs.get(jid)
            if unit is None:
                continue
            if unit not in self.results and not self.queue_ended:
                # The job is enqueued BEFORE the wipe and the RESULT written after it, so a fast grade (a
                # rollout that saved nothing is a job now) can land in between: taken at the next poll, not a fault
                with self.glock:
                    self.early.append(line)
                continue
            status, score, reason = H.score_of(line)
            if status == "refused":
                self.fail(f"{unit}: grade job {jid} refused by the runner (a loop fault, not a grade): {reason}",
                          by=unit)
                self.state_unit(unit, state="failed")
                continue
            if status == G.GRADER_FAILED:
                base = jid.split("#regrade", 1)[0]
                if base in self.grader.regrades:
                    log(f"{unit}: grader failed twice ({reason}; job {jid}): recorded as {G.GRADER_FAILED}, the leg "
                        f"continues")
                    self.accept_grade(unit, line, G.GRADER_FAILED)
                    continue
                try:
                    with self.glock:
                        rg = self.grader.regrade(jid)
                except (H.GradeError, H.HandleError, Refuse, OSError) as e:
                    self.fail(f"{unit}: grader failed ({reason}) and the re-grade could not be enqueued: {e}", by=unit)
                    continue
                self.jobs[rg["job_id"]] = unit
                self.state_unit(unit, grade_job_id=rg["job_id"])
                log(f"{unit}: grader failed ({reason}); re-graded ONCE as {rg['job_id']}")
                continue
            self.accept_grade(unit, line, score)
        return len(new)

    def _poll_loop(self) -> None:
        while not self._poll_stop.wait(self.grade_poll_s):
            try:
                self.poll_grades()
            except Exception as e:  # noqa: BLE001 -- the poller never dies silently
                log(f"grade poller: {type(e).__name__}: {e}")

    # ── the rate-limit wait ──────────────────────────────────────────────────────────────────────────────────────
    def median_rollout_s(self):
        return statistics.median(self.rollout_s) if self.rollout_s else None

    def live_train_leases(self) -> list:
        """Holders of this run's live TRAIN-leg leases on the pool (leases/<holder>.json, until_ts ahead, pid alive)."""
        from src.orchestrator import pool_container_evolver as pce
        d = pce.pool_machines_file(self.run_dir).parent / "leases"
        want = f"ccleg-{self.run_dir.name}_"
        out, now = [], time.time()
        for f in sorted(d.glob("*.json")) if d.is_dir() else []:
            try:
                rec = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            h = str(rec.get("holder") or "")
            if not (h.startswith(want) and "_train-" in h) or float(rec.get("until_ts") or 0) <= now:
                continue
            try:
                os.kill(int(rec.get("pid") or 0), 0)
            except (OSError, ValueError):
                continue
            out.append(h)
        return out

    def train_legs_waiting(self) -> list:
        """The live train leases
        (live_train_leases) whose leg still has a unit WAITING for a seat -- its store's STATE.json has a unit in state
        "queued" (the store is <this leg's store>/../<tag>, the tag from the holder "ccleg-<tag>-<stamp>"). An
        unreadable state counts as waiting (train first)."""
        out = []
        for h in self.live_train_leases():
            tag = h[len("ccleg-"):].rsplit("-", 1)[0]
            try:
                units = json.loads((Path(self.store).parent / tag / "STATE.json").read_text()).get("units") or {}
            except (OSError, ValueError):
                out.append(h)
                continue
            if any((u or {}).get("state") == "queued" for u in units.values()):
                out.append(h)
        return out

    # ── the pool-wide rollout cap (read and written under the pool registry's flock) ──────────────────────────────
    def pool_others(self) -> int:
        """The rollout containers the run's OTHER live legs hold now: the n of <pool root>/inflight/<holder>.json of
        every other holder whose pid is alive (a dead holder's file is removed, never counted)."""
        d = self.pool_reg.root / POOL_INFLIGHT_DIR
        n = 0
        for f in sorted(d.glob("*.json")) if d.is_dir() else []:
            try:
                rec = json.loads(f.read_text())
                held = int(rec.get("n") or 0)
            except (OSError, ValueError, TypeError, AttributeError):
                continue
            if rec.get("holder") == self.holder:
                continue
            if not H.P._pid_alive(rec.get("pid")):
                f.unlink(missing_ok=True)
                continue
            n += held
        return n

    def pool_note(self, n: int) -> None:
        """This leg's count of rollout containers, written when it changes: {holder, pid, n, updated_utc}."""
        if n == self.pool_n:
            return
        G.write_json_atomic(self.pool_reg.root / POOL_INFLIGHT_DIR / f"{self.holder}.json",
                            {"holder": self.holder, "pid": os.getpid(), "n": int(n), "updated_utc": L.now_utc()})
        self.pool_n = n

    def on_idle_closed(self) -> bool:
        """Nothing in flight, admission closed, units pending. kind "defect" -> False (the leg FAILS); kind "quota" (a
        session was rejected for a usage limit) -> wait until the reset the rejected sessions reported
        (cc_leg_lib.rate_limit_wait: bounded by WAIT_MAX_S and the leg's deadline), then admit again. No call is made
        to read the limit: a session rejected again simply closes admission again."""
        qu = self.queue
        kind = qu.guard.get("kind")
        if kind != "quota" or self.fatal:
            return False
        reason = qu.guard.get("reason") or ""
        now = self.clock()
        plan = G.rate_limit_wait(self.rejections, now, leg_deadline_ts=self.leg_deadline_ts, pending=len(qu.pending),
                                 conc=self.conc, median_rollout_s=self.median_rollout_s())
        if plan["reason"]:
            self.fail(plan["reason"])
            return False
        wake = plan["until_ts"] + G.PAUSE_MARGIN_S
        pause = {"reason": reason, "window": plan["window"], "until_utc": plan["until_utc"],
                 "wake_utc": G.utc_str(wake), "pending": len(qu.pending), "started_utc": L.now_utc(),
                 "remaining_estimate_s": plan["remaining_estimate_s"],
                 "rejected_units": [r.get("unit") for r in self.rejections], "n": len(self.pauses) + 1}
        self.pauses.append(pause)
        self.handle.rec.setdefault("pauses", []).append(pause)
        self.handle.save()
        self.state_write()
        when = (f"no reset was reported, so a fixed {G.PAUSE_NO_READING_S} s" if plan["window"] == "no_reset_reported"
                else f"the reported {plan['window']} reset {plan['until_utc']}")
        log(f"RATE LIMIT: {len(self.rejections)} session(s) rejected ({reason}); no new attempt until "
            f"{pause['wake_utc']} ({when}; waiting {int(wake - now)} s); {len(qu.pending)} unit(s) pending")
        while self.clock() < wake:
            if (self.store / STOP_LEG_FILE).exists():
                self.fail(f"{STOP_LEG_FILE} present in {self.store}: leg aborted by the operator during a "
                          f"rate-limit wait")
                return False
            if self.fatal:
                return False
            self.sleep(min(PAUSE_TICK_S, max(0.0, wake - self.clock())))
        pause["ended_utc"] = L.now_utc()
        self.rejections = []
        qu.reopen(f"waited until {pause['wake_utc']} for the usage limit's reset ({plan['window']})")
        self.state_write()
        return True

    # ── the run ──────────────────────────────────────────────────────────────────────────────────────────────────
    def resume_check(self) -> None:
        """A relaunch: graded samples kept and re-verified, their RESULT read back; the rest re-queued (their earlier
        RESULT / GRADE archived by the queue's admit); leftover cids noted for the wipe. A kept sample's score is its
        STATE score (grader_failed ones kept too), and both record copies must carry it
        (LegLayout.kept_sample_problems)."""
        st = self.prev_state
        self.state = st
        st["attempt"] = int(st.get("attempt") or 1) + 1
        st["resumed_utc"] = L.now_utc()
        st["run_dir"] = str(self.run_dir)
        for u in list(self.kept):
            score = (st["units"].get(u) or {}).get("score")
            probs = self.layout.kept_sample_problems(u, self.questions[G.sample_index(u)[0]], score=score)
            rp = L.run_dir(self.store, u) / "RESULT.json"
            res = read_json(rp) if rp.is_file() else None
            if not res or not res.get("completed"):
                probs.append(f"{u}: no completed RESULT.json beside the kept sample")
            if probs:
                log(f"{u}: kept sample not usable ({probs[:2]}); re-run")
                self.kept.remove(u)
                self.requeue.append(u)
                G.state_set_unit(st, u, state="queued", score=None)
                continue
            stem = G.record_stem(self.questions[G.sample_index(u)[0]], self.model, G.sample_index(u)[1])
            tr = read_json(self.layout.record_paths(stem)[0]).get("trace") or {}
            self.samples[u] = {"score": None if score == G.GRADER_FAILED else float(score),
                               "turn_exhausted": bool(tr.get("budget_exhausted"))}
            self.results[u] = res
        self.requeue = [u for u in self.units if u in set(self.requeue)]
        self.leftover_cids = sorted({str(su.get("cid")) for u, su in st["units"].items()
                                     if su.get("cid") and su.get("state") in ("setup", "running", "collected")
                                     and u not in set(self.kept)})
        log(f"RESUME attempt {st['attempt']}: {len(self.kept)} graded sample(s) kept, {len(self.requeue)} unit(s) to run, "
            f"{len(self.leftover_cids)} leftover container(s) of this tag")

    def run(self) -> int:
        self.t0 = self.clock()
        self.leg_deadline_ts = self.t0 + self.leg_timeout_s - LEG_DEADLINE_MARGIN_S
        self.store.mkdir(parents=True, exist_ok=True)
        self.log_fh = open(self.store / "driver.log", "a")
        C._LOG["fh"] = self.log_fh
        try:
            self.pid_fh = G.hold_leg_pid(self.run_dir, self.tag, argv=list(sys.argv), store=str(self.store))
        except RuntimeError as e:
            log(f"REFUSED: {e}")
            self.log_fh.close()
            C._LOG["fh"] = None
            return EXIT_REFUSED
        stamp = time.strftime("%Y%m%dT%H%M%S")
        holder = self.holder = f"ccleg-{self.tag}-{stamp}"
        self.leftover_cids = []
        log(f"LEG {self.tag}: {len(self.units)} unit(s) of {len(self.task_ids)} task(s) x {self.n_samples}, "
            f"concurrency {self.conc}, limits {self.limits}, wall {self.wall_s // 60} min, "
            f"bucket {self.bucket}, store {self.store}, cache {self.cache_dir}, tasks {self.tasks_dir}, "
            f"deadline {G.utc_str(self.leg_deadline_ts)}")
        if self.skills_dir:
            for ln in skill_lines(self.skills_dir):
                print(ln, flush=True)
            log(f"skill leg: {self.skills_dir} ({len(self.skill_rows)} file(s), {self.expect_skills}); probe {self.probe_rec}")
        else:
            log("base leg: no skills dir (init.skills must be empty)")
        if self.prev_state is not None:
            self.resume_check()
        else:
            self.state = G.state_new(tag=self.tag, key=self.key, epoch=self.epoch, run_dir=self.run_dir, bucket=self.bucket,
                                     skills_dir=self.skills_dir, stamp=stamp, units=self.units)
        self.state["plan"] = self.plan()
        self.state_write()
        self.prepare_units(self.requeue)
        self.layout.write_cache_env()
        rc = EXIT_FAILED
        path = None
        try:
            tok, self.cred, self.cred_from = C.read_credential(self.token_file_declared, self.env)
            self.tok = tok
            log(f"credential: {self.cred.kind} ({self.cred_from}); every session of this leg starts on it")
            if self.tau3:
                # the customer's OpenRouter key, read ONCE into this process's memory;
                # it reaches each bridge only as fd 4 (never an argv, an env file, the disk, a log or a container)
                key = (self.key_fn or self.T.read_openrouter_key)()
                self.tau3_ctx = {"run": self.name, "tag": self.tag, "data_root": self.tau3_block["data_root"],
                                 "user_sim_model": self.tau3_block["customer_model"], "key_fn": lambda: key,
                                 "user_llm_args": dict(self.tau3_block.get("customer_llm_args") or {})}
                log(f"tau3: the customer's OpenRouter key read into memory; customer {self.tau3_block['customer_model']} "
                    f"llm args {self.tau3_block.get('customer_llm_args') or {}}, "
                    f"agent time {self.tau3_block['agent_time_s']} s, hard stop {self.wall_s} s")
            self.handle, self.lease = self.join_fn(self.run_dir, holder, self.need_s, tag=self.tag, epoch=self.epoch,
                                                   key=self.key, store_dir=self.store / "handle", note=log)
            h = self.handle
            h.token_name = self.cred_from
            self.state["machine"] = {"sandbox_id": h.sid, "name": h.name, "retire_at_ts": h.retire_at_ts,
                                     "row": h.row}
            self.state["lease"] = dict(self.lease.summary(), holder=holder) if self.lease is not None else {}
            self.state_write()
            log(f"joined pool machine {h.sid} ({h.name}; {h.machine_name}); lease {holder}; retire_at "
                f"{G.utc_str(h.retire_at_ts) if h.retire_at_ts else '?'}")
            if self.tau3:
                # the bridge installed (sha-verified) and any bridge an earlier attempt of this tag left running ended
                self.state["tau3"] = {"bridge": self.T.install_bridge(h),
                                      "leftover_bridges": self.T.kill_leftover_bridges(h, self.T.bridge_root(self.name,
                                                                                                             self.tag))}
                self.state_write()
                log(f"tau3 bridge at {self.T.VM_BRIDGE_DIR}: {self.state['tau3']['bridge']['uploaded'] or 'already there'}; "
                    f"leftover bridges {self.state['tau3']['leftover_bridges']['killed'] or 'none'}")
            if self.leftover_cids:
                live = set(h.containers()["cc"])
                own = [c for c in self.leftover_cids if c in live]
                if own:
                    res = h.wipe(own, tag=self.tag)
                    log(f"leftover container(s) of this tag wiped: {res}")
                    self.state.setdefault("leftover_wipes", []).append(res)
            # a tau3 leg grades under the tau3 venv (the user's tau3 grader imports tau2), offline
            gkw = {"runner_py": self.T.VM_TAU3_PY} if self.tau3 else {}
            self.grader = self.grader_cls(h, run_name=self.name, tag=self.tag, knobs=self.knobs, plugin=self.plugin,
                                          task_rows=self.task_rows, layout=self.layout_map, overlay=self.overlay,
                                          max_procs=int(self.cc["grade_max_procs"]),
                                          max_heavy=int(self.cc["grade_max_heavy"]), job_timeout=JOB_TIMEOUT_S,
                                          store=self.store / "s01" / "grade", log=log, **gkw)
            # The grader's setup proven BEFORE any rollout (the runner, the pinned plug-in
            # files, the /data links, the leg's task files == the Mac's bytes): a fault here FAILS the leg unspent.
            # The record is the leg JSON's daytona.grade.setup (InPlaceGrader.summary).
            try:
                self.grader.install(self.task_ids)
            except (H.GradeError, H.HandleError) as e:
                raise Refuse(f"the grader's setup on the machine: {e}")
            self.state_write()
            self.pool_reg = H.P.Registry(H.pool_root(self.run_dir))          # the pool-wide rollout cap's lock
            box_cls = (self.box_base.for_leg(self.skill_rows, model=self.model, ctx=self.tau3_ctx) if self.tau3
                       else self.box_base.for_leg(self.skill_rows, model=self.model))
            self.queue = qu = LegQueue(self, h, self.store, list(self.requeue), stamp, self.wall_s, self.limits, tok,
                                       self.conc, box_cls=box_cls,
                                       poll=self.poll_fn, ttl_s=int(h.ttl) * 60 if h.ttl else None,
                                       poll_s=self.poll_s,
                                       expect_skills=self.judge_skills, deadline_ts=h.retire_at_ts,
                                       on_idle_closed=self.on_idle_closed, judge_fn=self.judge_fn)
            self.state["queue_started_utc"] = L.now_utc()
            self._poller = threading.Thread(target=self._poll_loop, name="cc-leg-grade-poll", daemon=True)
            self._poller.start()
            qu.run()
            self.queue_ended = True
            self.state["all_done_utc"] = L.now_utc()
            self.wait_for_grades()
            path = self.finish()
            rc = EXIT_OK if path == self.layout.leg_path else EXIT_FAILED
        except Refuse as e:
            log(f"REFUSED: {e}")
            self.problems.append(f"refused: {e}")
        except BaseException as e:  # noqa: BLE001 -- SIGTERM / Ctrl-C / crash: still tear everything down
            log(f"CRASH: {type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
            self.problems.append(f"crashed: {type(e).__name__}: {e}")
        finally:
            self._poll_stop.set()
            if self._poller is not None:
                self._poller.join(timeout=30)
            if path is None and not (self.run_dir / f"{self.tag}{G.FAILED_SUFFIX}").exists():
                try:
                    self.layout.write_failed(self.problems + self.fatal or ["leg ended before its result"],
                                             self.daytona_block(partial=True))
                except Exception as e:  # noqa: BLE001
                    log(f"FAILED file: {type(e).__name__}: {e}")
            self.teardown()
            log(f"LEG {self.tag} rc {rc}: {len(self.samples)}/{len(self.units)} sample(s) graded; problems "
                f"{(self.problems + self.fatal)[:3]}")
            try:
                self.log_fh.close()
            except OSError:
                pass
            C._LOG["fh"] = None
            if self.pid_fh is not None:
                self.pid_fh.close()                  # the pid file's lock ends with the leg (the file stays)
                self.pid_fh = None
        return rc

    def wait_for_grades(self) -> None:
        """After the queue: every enqueued job must have its line (the worker relaunched when it exited with jobs
        left), bounded by the leg's deadline."""
        t_end = self.leg_deadline_ts - GRADE_WAIT_MARGIN_S
        last = 0
        while True:
            self.poll_grades()
            pend = sorted({u for j, u in self.jobs.items() if u not in self.samples and u not in self.failed_units})
            if not pend or self.fatal:
                return
            if self.clock() > t_end:
                self.fail(f"{len(pend)} grade job(s) without a result at the leg's deadline: {pend[:4]}")
                return
            if len(pend) != last:
                log(f"waiting for {len(pend)} grade result(s): {pend[:3]}")
                last = len(pend)
            self.sleep(self.grade_poll_s)

    def containment(self) -> dict:
        boxes = self.queue.boxes if self.queue is not None else []
        c = {"units": len(boxes), "launched": 0, "collected": 0, "probe_pre_problems": 0, "probe_post_problems": 0,
             "listing_equal_manifest": 0, "token_scan_hits": 0, "down_wipe_verified": 0, "log_round_trip_equal": 0,
             "toolchain_problems": 0, "init_problems": 0, "thinking_problems": 0, "session_check_violations": 0,
             "skills_link": 0, "launched_without_link": 0, "memory_peak_max_bytes": None, "cg_root_memory_max": None,
             "setup_failures": 0, "not_launched": 0, "superseded_attempts": 0}
        # A unit re-queued after an infra failure (e.g. HTTP 429 at the usage limit) has an earlier box whose
        # session died; Claude Code logs the failed call as a "<synthetic>" model message. The model-identity checks
        # (init / thinking / session_check) judge each unit's FINAL attempt, the measured one; every other containment
        # count (probes, listing, token scan, wipes, round trips) still covers every box.
        final = {}
        for b in boxes:
            final[b.unit] = b                    # admission order: the last box of a unit is its measured attempt
        peaks = []
        for b in boxes:
            rec = b.rec
            c["launched"] += bool(b.t_launch)
            c["collected"] += bool(rec.get("log_round_trip"))
            c["not_launched"] += b.status == "not_launched"
            if b.t_launch and getattr(b, "skill_leg", False) and not rec.get("skills_link"):
                c["launched_without_link"] += 1
            c["probe_pre_problems"] += bool(rec.get("probe_pre_problems"))
            c["probe_post_problems"] += bool(rec.get("probe_post_problems"))
            c["listing_equal_manifest"] += ("listing_problems" in rec and not rec.get("listing_problems"))
            c["toolchain_problems"] += bool(rec.get("toolchain_problems"))
            c["token_scan_hits"] += len((rec.get("token_scan") or {}).get("hits") or []) + len(rec.get("container_token_hits") or [])
            c["down_wipe_verified"] += bool((rec.get("down_wipe") or {}).get("verified"))
            c["log_round_trip_equal"] += bool((rec.get("log_round_trip") or {}).get("equal"))
            c["skills_link"] += bool(rec.get("skills_link"))
            c["setup_failures"] += bool(rec.get("setup_error"))
            res = getattr(b, "res", None) or {}
            if final.get(b.unit) is not b:
                c["superseded_attempts"] += 1
            else:
                init = res.get("init")
                if init:
                    c["init_problems"] += bool(self.init_problems_fn(init, model=self.model,
                                                                     expect_skills=self.judge_skills))
                c["thinking_problems"] += bool(res.get("thinking_problems"))
                sc = res.get("session_check") or {}
                c["session_check_violations"] += len(sc.get("violations") or [])
            m = re.search(r"== memory.peak\n(\d+)", str(rec.get("cgroup_end") or ""))
            if m:
                peaks.append(int(m.group(1)))
            facts = (rec.get("prepare") or {}).get("facts") or {}
            if facts.get("cg_root_memory_max"):
                c["cg_root_memory_max"] = facts["cg_root_memory_max"]
        c["memory_peak_max_bytes"] = max(peaks) if peaks else None
        return c

    def collect_problems(self) -> list:
        P = list(self.fatal) + list(self.problems)
        qu = self.queue
        if qu is not None and qu.pending:
            P.append(f"{len(qu.pending)} unit(s) never ran: admission closed ({qu.guard.get('reason')})")
        not_graded = [u for u in self.units if u not in self.samples]
        if not_graded:
            P.append(f"{len(not_graded)} unit(s) not graded: {not_graded[:4]}")
        c = self.containment()
        if c["token_scan_hits"] or c["probe_pre_problems"] or c["probe_post_problems"] or c["init_problems"] or \
                c["thinking_problems"] or c["session_check_violations"]:
            P.append(f"containment: {c}")
        if c["units"] and (c["down_wipe_verified"] != c["units"] or c["log_round_trip_equal"] != c["collected"]):
            wipes = [b.unit for b in qu.boxes if not (b.rec.get('down_wipe') or {}).get('verified')]
            P.append(f"containment: wipes verified {c['down_wipe_verified']} of {c['units']} ({wipes[:3]}), log round "
                     f"trips equal {c['log_round_trip_equal']} of {c['collected']} collected")
        if self.skills_dir is not None:
            drift = G.check_skill_rows(self.skill_rows, self.skills_dir)
            if drift:
                P.append(f"the promoted dir changed under the leg: {drift[:2]}")
            if c["launched_without_link"]:
                P.append(f"{c['launched_without_link']} container(s) ran without the skills link")
        return P

    def daytona_block(self, partial: bool = False) -> dict:
        h, qu = self.handle, self.queue
        summ = (h.summary if h is not None else {}) or {}
        mrec = {}
        try:
            from src.orchestrator import pool_container_evolver as pce
            mrec = read_json(pce.pool_machines_file(self.run_dir))
        except Exception:  # noqa: BLE001
            pass
        cost = sum(float((r.get("cost") or 0.0)) for r in self.results.values())
        d = {"driver": DRIVER, "driver_sha256": L.sha256_file(Path(__file__)), "store": str(self.store),
             "cache_dir": str(self.cache_dir), "tasks_dir": str(self.tasks_dir), "bucket": self.bucket,
             "sandbox_id": h.sid if h else None, "sandbox_name": h.name if h else None,
             "machine": dict(h.row) if h else None,
             "lease": ({"holder": self.lease.holder, "path": str(self.lease.path), "acknowledged": True,
                        **{k: v for k, v in self.lease.summary().items() if k in ("cap_ts", "writes", "errors")}}
                       if self.lease is not None else (self.state or {}).get("lease")),
             "pool": {"registry": (h.rec.get("registry") if h else None),
                      "host_pid": mrec.get("host_pid"), "host_script": ((h.rec.get("registry") or {}).get("host_script")
                                                                        if h else None),
                      "result_guarantee": G.CC_RESULT_GUARANTEE,
                      "machines_at_result": {n: (r or {}).get("state") for n, r in (mrec.get("machines") or {}).items()},
                      "generation": mrec.get("generation"), "join": (h.rec.get("join") if h else None)},
             "spend": {"pauses": list(self.pauses),
                       "cost_usd_notional": round(cost, 4),
                       "machine_hours_share": round((self.clock() - self.t0) / 3600.0, 3) if self.t0 else None,
                       "credential": {"kind": self.cred.kind if self.cred else None, "from": self.cred_from}},
             "containment": self.containment(),
             "grade": (self.grader.summary() if self.grader is not None else None),
             "skills": ({"dir": str(self.skills_dir), "names": self.expect_skills,
                         "tree_sha256": G.skills_tree_sha256(self.skills_dir),
                         "library_sha256": G.skill_library_mod().library_sha256(self.skills_dir),
                         "probe": self.probe_rec} if self.skills_dir else None),
             "knobs": self.knobs, "infra": {"file": str(self.run_dir / "INFRA.json"),
                                             "sha256": L.sha256_file(self.run_dir / "INFRA.json")},
             "prompt_mode": self.rt.get("prompt_mode"), "data_overlay": (data_overlay.record(self.overlay)
                                                                          if self.overlay else None),
             "data_overlay_from": "INFRA.json",
             "plan": {"n_tasks": len(self.task_ids), "n_samples": self.n_samples, "concurrency": self.conc,
                      "wall_min": self.wall_s // 60, "limits": self.limits, "need_s": self.need_s,
                      "leg_timeout_s": self.leg_timeout_s, "leg_deadline_utc": G.utc_str(self.leg_deadline_ts)},
             "guard": dict(qu.guard) if qu is not None else None,
             "guard_readings": list(qu.readings) if qu is not None else None,
             "queue_errors": list(qu.errors) if qu is not None else None,
             "max_inflight": qu.max_inflight if qu is not None else None,
             "retries": dict(self.retries), "attempt": (self.state or {}).get("attempt"),
             "kept_from_earlier_attempt": list(self.kept),
             "host_summary": {k: summ.get(k) for k in ("sandbox_id", "sandbox_name", "granted_ttl_min", "ready_utc")},
             "notes": [f"{self.tag}: Claude Code {ctl.CC_VERSION} leg on the run's pool machine, graded in place "
                       f"({L.now_utc()})"] + (["PARTIAL: the leg ended before its result"] if partial else [])}
        return d

    def runtime_env(self) -> dict:
        h = self.handle
        summ, ident = h.summary, h.identity
        ls = self.inf.get("leg_sandbox") or {}
        bundle = summ.get("bundle") or {}
        knobs = self.knobs
        thread_env = {k: knobs[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                            "NUMEXPR_MAX_THREADS", "MAX_CONCURRENCY") if k in knobs} or None
        return G.runtime_env_record(
            sandbox_id=h.sid, sandbox_name=h.name, readback=h.rec.get("readback_join"),
            image={"dockerfile": ls.get("dockerfile"), "dockerfile_sha256": ls.get("dockerfile_sha256"),
                   "daytona_snapshot_ref": ls.get("snapshot_ref"), "image_id_reported_to_runner": ls.get("image_id")},
            libreoffice=ident.get("libreoffice"), solver_stack=ident.get("solver_stack"),
            solver_python=ident.get("solver_python"), grading_openpyxl=ident.get("runner_openpyxl") or ls.get("grading_openpyxl"),
            # graded by the run's grade(task, snapshot), no pinned scorer of the loop's own
            runner_python=ident.get("runner_python"), scorer_sha256=None,
            grader=self.grader.grader_record() if self.grader else None, shim_sha256=bundle.get("shim_sha256"),
            split_file=str(self.split_path), split_sha256=self.split_sha256,
            skills=({"dir": str(self.skills_dir), "tree_sha256": G.skills_tree_sha256(self.skills_dir),
                     "names": self.expect_skills} if self.skills_dir else None),
            cache_dir=str(self.cache_dir), knobs=knobs, infra={"file": str(self.run_dir / "INFRA.json"),
                                                               "sha256": L.sha256_file(self.run_dir / "INFRA.json")},
            egress={"machine": summ.get("egress"), "solver_containers": ctl.ALLOW,
                    "grading": (self.T.EGRESS_GRADING if self.tau3 else
                                "in place on the pool machine (env -u OPENROUTER_API_KEY; no key on the machine)")},
            grade_selftest=summ.get("grade_selftest"), cc_version=ctl.CC_VERSION, cc_sha256=ctl.CC_SHA256,
            thread_env=thread_env)

    def solver_harness(self) -> dict:
        results = [self.results[u] for u in self.units if u in self.results]
        text = facts = None
        if self.skills_dir:
            seen: dict = {}
            for r in results:
                for s in ((r.get("init") or {}).get("skills") or []):
                    seen[str(s)] = seen.get(str(s), 0) + 1
            facts = G.skills_delivery_facts(self.skills_dir, init_skills_seen=seen, sessions=len(results),
                                            probe=(self.probe_rec or {}).get("path"))
            text = G.skills_delivery_text(self.skills_dir.name, facts["tree_sha256"][:12])
        h = G.solver_harness_record(results, model=self.model, max_turns=self.budget, skills_delivery=text,
                                    skills_delivery_facts=facts)
        if self.tau3:
            h["prompt_delivery"] = self.T.PROMPT_DELIVERY       # the bridge sends the first message
        return h

    def finish(self) -> Path:
        """The leg JSON through LegLayout: the assembled leg re-read against STATE.json and both record copies
        (LegLayout.verify), written only when nothing is wrong; otherwise the FAILED file."""
        # A wipe that failed at collect time is retried HERE, before the containment judgement: a wipe can fail once
        # and pass on the retry, and a leg must not be refused on that stale count.
        qu = self.queue
        if qu is not None:
            for b in qu.boxes:
                if b.status != "new" and not b.wiped:
                    first = dict(b.rec.get("down_wipe") or {})
                    b.wiped = b.wipe()
                    log(f"{b.unit}: wipe retried at finish: {'verified' if b.wiped else 'STILL unverified'} "
                        f"(first attempt: {json.dumps(first, default=str)[:240]})")
        problems = self.collect_problems()
        daytona = self.daytona_block()
        if problems:
            log(f"leg NOT written: {problems[:4]}")
            return self.layout.write_leg({}, {}, {}, daytona, problems, verify=False)
        try:
            runtime_env = self.runtime_env()
            harness = self.solver_harness()
            leg = self.layout.assemble(self.samples, runtime_env=runtime_env, solver_harness=harness, daytona=daytona)
        except ValueError as e:
            return self.layout.write_leg({}, {}, {}, daytona, [f"assemble: {e}"], verify=False)
        probs = self.layout.verify(leg)
        if probs:
            log(f"leg refused by its own re-read (STATE / records / per_task): {probs[:3]}")
        path = self.layout.write_leg(self.samples, runtime_env, harness, daytona, probs, verify=False)
        if path == self.layout.leg_path:
            log(f"LEG WRITTEN {path}: sample_pass {leg['sample_pass']}/{leg['sample_total']} pass3 {leg['pass3']} "
                f"all3 {leg['all3']} turn_exhausted {leg['turn_exhausted']} grader_failed {leg['grader_failed']} "
                f"(STATE, records and per_task agree)")
            try:
                with self.glock:
                    log(f"machine clean-up: {self.grader.cleanup()}")
            except (H.GradeError, H.HandleError, Refuse, OSError) as e:
                log(f"machine clean-up failed (results kept): {e}")
        return path

    def teardown(self) -> None:
        """Every container of this leg wiped (verified), STATE.json final, the lease released LAST (after the last
        wipe, never before)."""
        qu = self.queue
        try:
            if qu is not None:
                qu.shutdown()
                for b in qu.boxes:
                    if b.status != "new" and not b.wiped:
                        b.wiped = b.wipe()
                    if b.t_launch and not b.result_written:
                        try:
                            b.res = b.result()
                            b.result_written = True
                            self.on_result(b)
                        except Exception as e:  # noqa: BLE001
                            log(f"{b.unit}: result: {e}")
        except Exception as e:  # noqa: BLE001
            log(f"teardown: {type(e).__name__}: {e}")
        finally:
            if self.state is not None:
                try:
                    self.state["ended_utc"] = L.now_utc()
                    self.state_write()
                except Exception as e:  # noqa: BLE001
                    log(f"STATE.json: {e}")
            if self.pool_reg is not None:            # this leg holds no rollout container any more
                try:
                    with self.pool_reg.lock():
                        (self.pool_reg.root / POOL_INFLIGHT_DIR / f"{self.holder}.json").unlink(missing_ok=True)
                except OSError as e:
                    log(f"pool inflight record: {e}")
            if self.handle is not None:
                try:
                    ok = self.handle.release()
                    log(f"lease released: {ok}")
                except Exception as e:  # noqa: BLE001
                    log(f"lease release: {type(e).__name__}: {e}")


# ───────────────────────────────────────────────────────────────────────────── the skills probe
def probe_verdict(text: str, placement_dir: str, *, two_turn: bool) -> dict:
    """What one probe stream proves about a placement: init.skills == [probe-skill], the Skill tool present, the
    init otherwise clean; on the two-turn call the Skill tool_use naming probe-skill, its `Launching skill:` result,
    the injected `Base directory for this skill: <dir>/probe-skill` line, the final text carrying it, and
    rollout_evidence.skill_loads (the loop's reader) reporting the load from the converted messages."""
    s = L.parse_stream(text)
    init = s["init"] or {}
    evs, _bad = L.events(text)
    base_line = f"Base directory for this skill: {placement_dir}/{PROBE_SKILL}"
    v = {"init_skills": init.get("skills"), "tools_has_skill": "Skill" in set(init.get("tools") or []),
         "init_problems": L.init_problems(init, model=None, cwd=L.WS_CONTAINER, expect_skills=[PROBE_SKILL]),
         "result_subtype": (s["result"] or {}).get("subtype"), "is_error": (s["result"] or {}).get("is_error"),
         "n_events": s["n_events"], "base_line": base_line}
    probs = []
    if v["init_skills"] != [PROBE_SKILL]:
        probs.append(f"init.skills {v['init_skills']!r} != ['{PROBE_SKILL}']")
    if not v["tools_has_skill"]:
        probs.append("init.tools lacks Skill")
    if v["init_problems"]:
        probs.append(f"init problems {v['init_problems']}")
    if not s["result"]:
        probs.append("no result event")
    if two_turn:
        use_ids = [c["id"] for e in evs if e.get("type") == "assistant"
                   for c in ((e.get("message") or {}).get("content") or [])
                   if isinstance(c, dict) and c.get("type") == "tool_use" and c.get("name") == "Skill"
                   and PROBE_SKILL in json.dumps(c.get("input") or {})]
        launched = base_seen = False
        for e in evs:
            if e.get("type") != "user":
                continue
            for c in ((e.get("message") or {}).get("content") or []):
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "tool_result" and c.get("tool_use_id") in use_ids:
                    cont = c.get("content")
                    if isinstance(cont, list):
                        cont = "\n".join(x.get("text", "") for x in cont if isinstance(x, dict))
                    if str(cont or "").startswith(f"Launching skill: {PROBE_SKILL}"):
                        launched = True
                if c.get("type") == "text" and str(c.get("text") or "").startswith(base_line):
                    base_seen = True
        final = str(next((e.get("result") for e in evs if e.get("type") == "result"), "") or "")
        v.update(skill_tool_use=bool(use_ids), launching_ok=launched, base_dir_line_seen=base_seen,
                 result_has_base_line=base_line in final, result_text=final[:300])
        try:
            RE = G.rollout_evidence_mod()
            tr = G.build_trace(stream_text=text, user_message=PROBE_MSG_TWO, result={"subtype": v["result_subtype"], "init": init},
                               score=0.0, output_saved=False, model=PROBE_MODEL)
            rec = G.build_record(question=json.dumps({"task_id": "probe/skills"}), model=PROBE_MODEL, bucket="0" * 12,
                                 trace=tr, created_at=time.time())
            sk = RE.skill_loads(RE.direct_trace(rec)["lines"], [PROBE_SKILL])
            v["skill_loads"] = {"loaded": sk.get("loaded"), "skill_tool_calls": sk.get("skill_tool_calls")}
        except Exception as e:  # noqa: BLE001
            v["skill_loads"] = {"error": f"{type(e).__name__}: {e}"}
        if not use_ids:
            probs.append("no Skill tool_use naming probe-skill")
        if not launched:
            probs.append("no `Launching skill: probe-skill` tool result")
        if not base_seen:
            probs.append(f"no user text starting with {base_line!r}")
        # The model's echo (result_has_base_line) is recorded, not required. PROBE_MSG_TWO's "the first
        # line of what the skill gave you" has two valid readings (the tool result's "Launching skill: ..." or the
        # injected text's first line), and haiku may pick either. The injected base-directory line (base_seen,
        # the exact placement path), the Skill tool_use, the Launching result and skill_loads prove the placement.
        if (v.get("skill_loads") or {}).get("loaded") != [PROBE_SKILL]:
            probs.append(f"rollout_evidence.skill_loads reports {v.get('skill_loads')}")
    v["problems"] = probs
    v["ok"] = not probs
    return v


class SkillsProbe:
    """`probe-skills`: join the run's machine, one container with the probe set placed as a skill leg places it,
    the one-turn and two-turn haiku calls, the same two with the fallback placement, down --wipe, the record."""

    def __init__(self, a, *, join_fn=None, box_cls=None, skills_dir=None):
        self.a = a
        self.join_fn = join_fn or H.join
        self.box_cls = box_cls or ProbeBox
        self.skills_dir = Path(skills_dir) if skills_dir else None

    def run(self) -> int:
        a = self.a
        os.environ.pop("OPENROUTER_API_KEY", None)
        rd = Path(a.run_dir).resolve()
        inf = L.infra_of(rd)
        if (inf.get("solver") or {}).get("harness") != "claude_code":
            raise Refuse("INFRA solver.harness != claude_code")
        tok, cred, cred_from = C.read_credential((inf.get("claude_code") or {}).get("token_file") or None)
        stamp = time.strftime("%Y%m%dT%H%M%S")
        pdir = rd / ".operator" / f"cc_skills_probe_{stamp}_{os.getpid()}"
        pdir.mkdir(parents=True, exist_ok=True)
        C._LOG["fh"] = open(pdir / "probe.log", "a")
        sd = self.skills_dir
        if sd is None:
            sd = pdir / "skills"
            (sd / PROBE_SKILL).mkdir(parents=True, exist_ok=True)
            (sd / PROBE_SKILL / "SKILL.md").write_text(PROBE_SKILL_MD)
        rows = G.skill_rows(sd)
        if PROBE_SKILL not in G.skill_names(rows):
            raise Refuse(f"probe set {sd}: names {G.skill_names(rows)} (needs {PROBE_SKILL})")
        rec = {"cc_version": ctl.CC_VERSION, "model": PROBE_MODEL, "utc": L.now_utc(), "run_dir": str(rd),
               "credential": {"kind": cred.kind, "from": cred_from},
               "probe_dir": str(pdir), "skills_dir": str(sd), "skill_names": G.skill_names(rows),
               "placement_primary": {"path": G.SKILLS_LINK, "target": G.SKILLS_VM},
               "placement_fallback": {"path": FALLBACK_SKILLS_DIR}, "primary": None, "fallback": None, "both": None,
               "streams": {}, "problems": []}
        handle = box = None
        rc = EXIT_FAILED
        try:
            handle, lease = self.join_fn(rd, f"ccprobe-{stamp}", 3600.0, tag=f"probe_{rd.name}", key="probe",
                                         store_dir=pdir / "handle", note=log)
            handle.token_name = cred_from
            rec["sandbox_id"], rec["sandbox_name"] = handle.sid, handle.name
            box = self.box_cls(handle, stamp[4:].lower(), 3600, rows, pdir / "container")
            with handle.lock:
                box.prepare()
                box.up(init_root=False)
            box.probe()
            box.upload()
            box.toolchain()
            one = box.call(tok, "primary_one", PROBE_MSG_ONE, 1)
            v1 = probe_verdict(one["stream_text"], G.SKILLS_LINK, two_turn=False)
            two = box.call(tok, "primary_two", PROBE_MSG_TWO, 3)
            v2 = probe_verdict(two["stream_text"], G.SKILLS_LINK, two_turn=True)
            rec["primary"] = {"ok": v1["ok"] and v2["ok"], "one_turn": v1, "two_turn": v2,
                              "problems": v1["problems"] + v2["problems"], "listing_link": box.rec.get("skills_link")}
            rec["streams"]["primary_one"] = self.relevant(one["stream_text"])
            rec["streams"]["primary_two"] = self.relevant(two["stream_text"])
            log(f"PRIMARY placement ({G.SKILLS_LINK}): {'OK' if rec['primary']['ok'] else 'FAILED ' + str(rec['primary']['problems'])}")
            # the fallback (4.2): one reading with BOTH placements present (what init.skills shows then), then the
            # primary link removed and the same two calls against the fallback ALONE -- recorded, used only if the
            # primary failed
            try:
                fb_dir = box.write_fallback((sd / PROBE_SKILL / "SKILL.md").read_text())
                b1 = box.call(tok, "both_one", PROBE_MSG_ONE, 1)
                b_init = L.parse_stream(b1["stream_text"])["init"] or {}
                rec["both"] = {"init_skills": b_init.get("skills"), "tools_has_skill": "Skill" in set(b_init.get("tools") or [])}
                rec["streams"]["both_one"] = self.relevant(b1["stream_text"])
                box.remove_primary_link()
                f1 = box.call(tok, "fallback_one", PROBE_MSG_ONE, 1)
                fv1 = probe_verdict(f1["stream_text"], FALLBACK_SKILLS_DIR, two_turn=False)
                f2 = box.call(tok, "fallback_two", PROBE_MSG_TWO, 3)
                fv2 = probe_verdict(f2["stream_text"], FALLBACK_SKILLS_DIR, two_turn=True)
                rec["fallback"] = {"ok": fv1["ok"] and fv2["ok"], "dir": fb_dir, "one_turn": fv1, "two_turn": fv2,
                                   "problems": fv1["problems"] + fv2["problems"],
                                   "note": "measured ALONE (the primary link removed host-side first)"}
                rec["streams"]["fallback_one"] = self.relevant(f1["stream_text"])
                rec["streams"]["fallback_two"] = self.relevant(f2["stream_text"])
                log(f"FALLBACK placement ({FALLBACK_SKILLS_DIR}): {'OK' if rec['fallback']['ok'] else 'FAILED ' + str(rec['fallback']['problems'])}")
            except Refuse as e:
                rec["fallback"] = {"ok": False, "problems": [str(e)]}
            rc = EXIT_OK if rec["primary"]["ok"] else EXIT_FAILED
        except Refuse as e:
            rec["problems"].append(f"refused: {e}")
            log(f"REFUSED: {e}")
            rc = EXIT_REFUSED if handle is None else EXIT_FAILED
        except BaseException as e:  # noqa: BLE001
            rec["problems"].append(f"crashed: {type(e).__name__}: {e}")
            log(f"CRASH: {type(e).__name__}: {e}\n{traceback.format_exc()[-1200:]}")
        finally:
            if box is not None and box.status != "new":
                rec["wiped"] = box.wipe()
            if handle is not None:
                rec["lease_released"] = handle.release()
            out = rd / PROBE_RECORD
            G.write_json_atomic(out, rec)
            G.write_json_atomic(pdir / "CC_SKILLS_PROBE.json", rec)
            log(f"probe record {out}: primary ok={((rec.get('primary') or {}).get('ok'))} fallback ok="
                f"{((rec.get('fallback') or {}).get('ok'))} rc {rc}")
            C._LOG["fh"].close()
            C._LOG["fh"] = None
        return rc

    @staticmethod
    def relevant(text: str) -> list:
        """The events a reader needs: init (skills, tools), every Skill tool_use / tool_result, the base-directory
        text, the result."""
        out = []
        for e in L.events(text)[0]:
            t = e.get("type")
            if t == "system" and e.get("subtype") == "init":
                out.append({"type": "init", "skills": e.get("skills"), "tools": e.get("tools"), "cwd": e.get("cwd"),
                            "model": e.get("model"), "claude_code_version": e.get("claude_code_version")})
            elif t in ("assistant", "user"):
                for c in ((e.get("message") or {}).get("content") or []):
                    if not isinstance(c, dict):
                        continue
                    if c.get("type") == "tool_use" and c.get("name") == "Skill":
                        out.append({"type": "tool_use", "name": "Skill", "input": c.get("input")})
                    elif c.get("type") == "tool_result":
                        cont = c.get("content")
                        if isinstance(cont, list):
                            cont = "\n".join(x.get("text", "") for x in cont if isinstance(x, dict))
                        out.append({"type": "tool_result", "head": str(cont or "")[:200]})
                    elif c.get("type") == "text" and str(c.get("text") or "").startswith("Base directory"):
                        out.append({"type": "text", "head": str(c.get("text"))[:200]})
            elif t == "result":
                out.append({"type": "result", "subtype": e.get("subtype"), "is_error": e.get("is_error"),
                            "num_turns": e.get("num_turns"), "text": str(e.get("result") or "")[:300]})
        return out


# ───────────────────────────────────────────────────────────────────────────── commands
def cmd_leg(a) -> int:
    run = LegRun(a)
    try:
        plan = run.validate()
    except Refuse as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    if a.dry_run:
        return run.dry_run()
    if a.check_only:
        print(f"check-only OK: {plan['tag']} {plan['n_units']} unit(s) ({plan['units_to_run']} to run), skills "
              f"{plan['skills'] or 'none'}, bucket {plan['bucket']}, pool machine {plan['pool_machine']}")
        return EXIT_OK
    return run.run()


def cmd_probe(a) -> int:
    try:
        return SkillsProbe(a, skills_dir=a.skills_dir or None).run()
    except Refuse as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED


def cmd_status(a) -> int:
    rd = Path(a.run_dir).resolve()
    try:
        dl = LL.load_runtime_block(rd)
    except LL.Refused as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    root = Path(dl["store_root"]) / rd.name
    legs = sorted(p for p in root.glob("*/STATE.json") if not a.tag or p.parent.name == a.tag)
    if not legs:
        print(f"no leg STATE.json under {root}")
    for p in legs:
        try:
            st = G.state_read(p)
        except ValueError as e:
            print(f"{p}: {e}")
            continue
        by: dict = {}
        for u, d in (st.get("units") or {}).items():
            by[d.get("state")] = by.get(d.get("state"), 0) + 1
        g = st.get("guard") or {}
        print(f"{st.get('tag')}: attempt {st.get('attempt')} started {st.get('started_utc')} updated {st.get('updated_utc')} "
              f"ended {st.get('ended_utc') or '-'}; units {by}; guard admitting={g.get('admitting')} kind={g.get('kind')} "
              f"reason={g.get('reason')}; pauses {len(st.get('pauses') or [])}; machine "
              f"{(st.get('machine') or {}).get('sandbox_id')}; problems {(st.get('problems') or [])[:2]}")
        leg = rd / f"{st.get('tag')}.json"
        failed = rd / f"{st.get('tag')}{G.FAILED_SUFFIX}"
        print(f"  result: {'WRITTEN ' + str(leg) if leg.is_file() else 'FAILED ' + str(failed) if failed.is_file() else 'none yet'}")
    return H.cmd_status(a) if a.pool else 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("leg")
    p.add_argument("--tag", required=True)
    p.add_argument("--key", required=True, choices=("train", "val", "test"))   # test: the held-out leg
    p.add_argument("--run-dir", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--skills-dir", default="", help="the promoted dir (skill leg) or '' (base leg)")
    p.add_argument("--n-samples", type=int, required=True)
    p.add_argument("--concurrency", type=int, required=True)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("probe-skills")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--skills-dir", default="", help="a set holding probe-skill (default: a dummy set)")
    p = sub.add_parser("status")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--tag", default="")
    p.add_argument("--pool", action="store_true", help="also the pool host's status")
    a = ap.parse_args(argv)
    signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        return {"leg": cmd_leg, "probe-skills": cmd_probe, "status": cmd_status}[a.cmd](a)
    except Refuse as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED


if __name__ == "__main__":
    sys.exit(main())
