"""tau3 banking units in the Claude Code LEG DRIVER: the loop runs a tau3 run with the SAME leg driver as SSB
(scripts/daytona/cc_leg_driver.py), and this module is the tau3 half of it. The driver imports it ONLY when the run's
SPEC benchmark is "tau3", so an SSB leg imports nothing of it.

One tau3 unit is the Daytona smoke's flow (scripts/daytona/tau3_cc_smoke.py), reused BY IMPORT, on the run's ONE pool
machine (scripts/daytona/Dockerfile.tau3_leg: the SSB leg image + /opt/tau3/venv with tau2 1.0.1; the data root
mirrored at its host path):
  - the container is the SSB solver's (cc_solver.Box.prepare / up / PRE probe), nothing uploaded into it on a base
    leg; a skill leg gets the promoted set and the one .claude/skills link as an SSB skill leg does; no output/ and no
    LibreOffice probe (the tau3 solver needs neither);
  - the BRIDGE (tau3_cc_smoke.BRIDGE_FILES: scripts/tau3_cc/tau2_cc_harness.py + tau2_bundle.py + tau3_retrieval.py,
    which the harness imports from its own dir; installed per leg at VM_BRIDGE_DIR, each file sha-verified and
    renamed into place) runs on the machine as root under /opt/tau3/venv, in its unit dir
    /opt/cc/<run>/<tag>/bridge/<task_>__s<k>/ (root 0700, invisible to every container): job.json (no secret;
    tau3_cc_smoke.unit_job: retrieval variant alltools (KB_search_bm25 + KB_search_dense + the read-only KB shell;
    the documents' embeddings from the data root's .embeddings_cache, a KB_search_dense query through OpenRouter
    with the customer's key, the shell through sandbox-runtime with the data root, /opt, /root, /home and /Users
    denied) and the harness's defaults -- 20 minutes of AGENT time the only limit, the customer
    openrouter/openai/gpt-5-nano with empty llm args, retries inside the bridge), bridge.pid, bridge.log, rc,
    out/{claude.stream.jsonl, claude.stdin.jsonl, claude.stderr, harness.json}, snapshot/tau2/ (the bundle);
  - the launch is tau3_cc_smoke.launch_cmd: the Claude credential on fd 3 (a subscription token or an API key; the
    job's token_kind, the kind of the run's one credential, tells the harness which) and the OpenRouter key on fd 4,
    here-strings from the SDK exec's env, never in an argv, a file or a container (PATH set to
    tau3_cc_smoke.VM_PATH: srt, rg, bwrap, socat, node for the KB shell); the key was read ONCE per leg on the host
    (read_openrouter_key: the environment, else the repository's .env) into the driver's memory; the bridge pid is
    moved into the cgroup BRIDGE_CGROUP (cpu.weight 10000, made by the pool machine's setup);
  - claude runs in the container through `evoctl enter --keep-fd3` (the harness's container mode); the poll reads the
    bridge's stream, its rc file and the container's / the bridge's liveness (poll_batch); the driver's hard backstop
    is agent_time_s + 40 min (tau3_cc_smoke.hard_stop_s): SIGTERM to the bridge, which kills claude and exits 75;
  - bridge rc 0 = a graded end; 75 = an infrastructure end (re-run once, never graded); anything else = not completed;
  - collect: `evoctl down` FIRST (no solver process left), then, for rc 0 only, the bundle copied into the container's
    /evo/work/tau2/ (what the user's grader reads and what the evolver sees as
    /evo/solver_workspaces/epoch_K/<id>_s<k>/tau2/), then `evoctl pack --part /evo/work --part /home/evo/.claude` (the
    snapshot InPlaceGrader keeps; Claude Code's own transcript is the evolver's trace, as for SSB), the unit dir
    downloaded to runs/<ud>/bridge/, every downloaded byte scanned for the credential, its prefix, the key and
    sk-or-v1-.
Grading is the run's frozen setup/benchmark_grader.py (= the user's tau3_grader.py) through plugin_grade_runner under
/opt/tau3/venv/bin/python, offline (cc_pool_host.InPlaceGrader(runner_py=VM_TAU3_PY)).
"""
from __future__ import annotations

import base64
import json
import os
import re
import shlex
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(REPO), str(REPO / "scripts" / "daytona"), str(REPO / "scripts" / "cc_solver"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cc_leg_driver as D  # noqa: E402  (LegBox, log; the driver imports this module lazily, never the reverse at import)
import cc_leg_lib as G  # noqa: E402
import cc_pool_host as H  # noqa: E402  (READ-ONLY reuse: CC_ROOT_VM, HandleError, _sha_map)
import cc_solver as C  # noqa: E402  (READ-ONLY reuse: Box, judge, Refuse, TokenHit, CHUNK_B)
import cc_solver_lib as L  # noqa: E402
import tau2_cc_harness as HB  # noqa: E402  (the bridge: stream_argv, FIRST_MESSAGE_TEMPLATE, SERVER, exit codes)
import tau3_cc_smoke as S  # noqa: E402  (READ-ONLY reuse: unit_job, validate_job, hard_stop_s, launch_cmd, scans)

ctl = L.ctl
# the release's extra flags (evolver_container_ctl.CC_EXTRA_ARGS; none for 2.1.281) ride in the job's claude_cmd
CLAUDE_CMD = {"claude_cmd": ["claude", *ctl.CC_EXTRA_ARGS]} if ctl.CC_EXTRA_ARGS else {}
q = shlex.quote
log = D.log

DOMAIN, VARIANT = S.DOMAIN, S.VARIANT        # banking_knowledge / alltools
VM_TAU3_PY = S.VM_TAU3_PY                    # /opt/tau3/venv/bin/python: the bridge's and the grader's interpreter
VM_BRIDGE_DIR = f"{H.CC_ROOT_VM}/tau3bin"    # every BRIDGE_FILES entry, per leg, sha-verified
VM_HARNESS = f"{VM_BRIDGE_DIR}/tau2_cc_harness.py"
BRIDGE_CGROUP = "/sys/fs/cgroup/evobridge"   # cpu.weight 10000 (the pool machine's setup makes it)
BRIDGE_FILES = S.BRIDGE_FILES                # (tau2_cc_harness.py, tau2_bundle.py, tau3_retrieval.py) on the host
PROMPT_DELIVERY = "tau2_cc_harness.FIRST_MESSAGE_TEMPLATE via the tau3 bridge"
KILL_GRACE_S = 30                            # a bridge SIGTERMed by the driver has this long before SIGKILL
EGRESS_GRADING = ("in place on the pool machine (env -u OPENROUTER_API_KEY); the customer's OpenRouter key in each "
                  "bridge process's memory only (fd 4); grading offline")
# `balive U`: the pid in U/bridge.pid when that process is alive AND runs in U (its cwd: a reused pid is never taken)
BRIDGE_FN = ('balive() { local p; p=$(cat "$1/bridge.pid" 2>/dev/null); [ -n "$p" ] && [ -d "/proc/$p" ] && '
             '[ "$(readlink "/proc/$p/cwd" 2>/dev/null)" = "$1" ] && echo "$p"; }')


# ───────────────────────────────────────────────────────────────────────────── small helpers
def short_id(task_id: str) -> str:
    """banking_knowledge/task_004 -> task_004 (the tau2 task's own id, its file's stem)."""
    dom, sep, rest = str(task_id).partition("/")
    if dom != DOMAIN or not sep or not rest:
        raise C.Refuse(f"{task_id!r} is not a {DOMAIN}/<task> id")
    return rest


def bridge_root(run: str, tag: str) -> str:
    """/opt/cc/<run>/<tag>/bridge: the leg's bridge unit dirs (root 0700)."""
    return f"{H.CC_ROOT_VM}/{run}/{tag}/bridge"


def kill_sh(vm_unit: str, grace_s: int = KILL_GRACE_S) -> str:
    """SIGTERM to the unit's bridge when it is alive (the harness kills claude, writes harness.json, exits 75), then
    SIGKILL after grace_s. Prints BRIDGE_ENDED <pid> or BRIDGE_GONE."""
    return (f'{BRIDGE_FN}; U={q(vm_unit)}; P=$(balive "$U"); if [ -n "$P" ]; then kill -TERM "$P" 2>/dev/null; '
            f'i=0; while [ $i -lt {int(grace_s)} ] && [ -d "/proc/$P" ]; do sleep 1; i=$((i + 1)); done; '
            f'kill -KILL "$P" 2>/dev/null; echo "BRIDGE_ENDED $P"; else echo BRIDGE_GONE; fi')


def _jload(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _sha_or_empty(p: Path) -> str:
    """The file's sha256, or the empty string's when it is absent (a bridge that never started claude streams
    nothing on either side)."""
    return L.sha256_file(p) if Path(p).is_file() else L.sha256_bytes(b"")


def read_openrouter_key() -> str:
    """The customer's OpenRouter key, read ONCE per leg on the host into memory (scripts/tau3_cc/openrouter_key.py:
    OPENROUTER_API_KEY from the environment, else the repository's .env; never printed); it reaches each bridge only
    as fd 4."""
    try:
        return S.openrouter_key()
    except S.Refuse as e:
        raise C.Refuse(str(e)) from None


# ───────────────────────────────────────────────────────────────────────────── validate (no Daytona, no key)
def task_problems(tasks: list, root: Path) -> list:
    """Every frozen task must be a tau3 banking task: no input, ONE answer file = its tau2 task file under the data
    root (tau3_cc_smoke.task_file: where the bridge's task_file and TAU2_DATA_DIR point), grading domain
    banking_knowledge and retrieval variant alltools (the variant the bridge runs)."""
    probs = []
    for t in tasks:
        tid = str(t.get("id"))
        try:
            want = S.task_file(short_id(tid), root)
        except C.Refuse as e:
            probs.append(str(e))
            continue
        g = t.get("grading") or {}
        if t.get("inputs"):
            probs.append(f"{tid}: inputs {t.get('inputs')} (a tau3 task has none)")
        if [Path(p) for p in t.get("answer_files") or []] != [want]:
            probs.append(f"{tid}: answer_files {t.get('answer_files')} != [{want}]")
        elif not want.is_file():
            probs.append(f"{tid}: {want} is missing on this host")
        if g.get("domain") != DOMAIN or g.get("retrieval_variant") != VARIANT:
            probs.append(f"{tid}: grading {dict(g)} (want domain {DOMAIN}, retrieval_variant {VARIANT})")
    return probs


def leg_settings(run) -> dict:
    """validate()'s tau3 half: SPEC names the customer's model, the frozen tasks are tau3 banking tasks under INFRA
    data.root, and one unit's job passes the harness's own load_job. Returns the plan's "tau3" block; its hard_stop_s
    (agent_time_s + 40 min) is the leg's container wall -- cc_leg.wall_min (SSB's 20-minute wall) is never read for
    tau3: the agent clock lives in the bridge."""
    um = run.spec.get("user_sim_model")
    if not isinstance(um, str) or not um.strip() or um == "none":
        raise C.Refuse(f"SPEC user_sim_model {um!r}: a tau3 run names the customer's model")
    root = Path(str((run.inf.get("data") or {}).get("root") or ""))
    if not root.is_absolute():
        raise C.Refuse(f"INFRA data.root {str(root)!r} is not an absolute path")
    probs = task_problems(run.plugin["tasks"], root)
    if probs:
        raise C.Refuse(f"the run's frozen tasks are not tau3 {DOMAIN} tasks under {root}: {probs[:3]}")
    ula = (getattr(run, "rt", None) or {}).get("tau3_user_llm_args") or {}   # optional customer llm args
    if not isinstance(ula, dict):
        raise C.Refuse(f"RUNTIME tau3_user_llm_args {ula!r} is not a JSON object")
    t0 = run.task_ids[0]
    job = S.unit_job(short_id(t0), 1, "cc-tau3-plan", f"{bridge_root(run.name, run.tag)}/{L.unit_dir_name(t0 + '#s1')}",
                     model=run.model, overrides={"user_sim_model": um, **({"user_llm_args": ula} if ula else {}), **CLAUDE_CMD},
                     root=root)
    try:
        eff = S.validate_job(job)
    except S.Refuse as e:
        raise C.Refuse(str(e)) from None
    return {"bridge_sha256": {p.name: L.sha256_file(p) for p in BRIDGE_FILES}, "bridge_dir": VM_BRIDGE_DIR,
            "first_message_template_sha256": HB.first_message_sha256(), "prompt_delivery": PROMPT_DELIVERY,
            "agent_time_s": eff["agent_time_s"], "hard_stop_s": S.hard_stop_s(eff), "customer_model": um,
            "customer_llm_args": dict(ula),
            "data_root": str(root), "interpreter": VM_TAU3_PY}


def prepare_units(run, units: list) -> None:
    """<store>/runs/<task_>__s<j>/{PREP.json, user_message.txt}: what cc_solver.Box.__init__ reads. Nothing is
    uploaded (upload_rows []) and the user message is empty: the bridge writes the first message
    (the harness's FIRST_MESSAGE_TEMPLATE with tau2's policy and the customer's first message); the one actually sent is
    kept at collect as runs/<ud>/first_message.txt."""
    for unit in units:
        task, k0 = G.sample_index(unit)
        ud = L.unit_dir_name(unit)
        rd = L.run_dir(run.store, unit)
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "first_message.txt").unlink(missing_ok=True)
        (rd / "user_message.txt").write_text("")
        qd = json.loads(run.questions[task])
        prep = {"schema": L.SCHEMA + ":prep", "unit": unit, "task_id": task, "sample_idx": k0 + 1,
                "safe_id": L.safe_id(task), "unit_dir": ud, "category": qd["category"], "spreadsheet_filename": None,
                "workspace_container": L.WS_CONTAINER, "upload_rows": [], "upload_problems": [],
                "user_message_sha256": L.sha256_bytes(b""),
                "user_message_source": "none: the tau3 bridge sends the first message (tau2_cc_harness.FIRST_MESSAGE_TEMPLATE)",
                "instruction_sha256": L.sha256_bytes(run.instruction.encode()), "prompt_delivery": PROMPT_DELIVERY,
                "question": qd, "task": run.task_rows[task],
                "leg": {"tag": run.tag, "key": run.key, "epoch": run.epoch,
                        "skills_dir": str(run.skills_dir) if run.skills_dir else None},
                "created_utc": L.now_utc()}
        L.jdump(rd / "PREP.json", prep)


# ───────────────────────────────────────────────────────────────────────────── the machine (once per leg)
def install_bridge(handle) -> dict:
    """Every BRIDGE_FILES entry (the harness, its bundle writer, its alltools module) at VM_BRIDGE_DIR under its own
    name, uploaded only when the machine's sha differs, renamed into place (a leg launching a bridge meanwhile never
    reads a partial file), each sha-verified (InPlaceGrader._install_runner's pattern). Idempotent."""
    if len({p.name for p in BRIDGE_FILES}) != len(BRIDGE_FILES):
        raise C.Refuse(f"two bridge files share a name: {[p.name for p in BRIDGE_FILES]}")
    want = {f"{VM_BRIDGE_DIR}/{p.name}": L.sha256_file(p) for p in BRIDGE_FILES}
    src = {f"{VM_BRIDGE_DIR}/{p.name}": p for p in BRIDGE_FILES}

    def shas() -> dict:
        code, out = handle.sh("sha256sum " + " ".join(q(p) for p in want) + " 2>/dev/null", timeout=120,
                              what="bridge sha")
        return H._sha_map(out)
    try:
        got = shas()
        up = sorted(p for p in want if got.get(p) != want[p])
        if up:
            handle.sh(f"mkdir -p {q(VM_BRIDGE_DIR)} && chmod 0700 {q(VM_BRIDGE_DIR)}", timeout=60, what="bridge dir")
            for dst in up:
                part = f"{dst}.{os.getpid()}.part"
                handle.sb.fs.upload_file(str(src[dst]), part, timeout=300)
                handle.sh(f"mv -f {q(part)} {q(dst)}", timeout=60, what="install bridge")
            got = shas()
    except H.HandleError as e:
        raise C.Refuse(f"installing the tau3 bridge: {e}") from None
    if got != want:
        raise C.Refuse(f"the tau3 bridge on the machine does not verify ({got} != {want})")
    return {"dir": VM_BRIDGE_DIR, "sha256": want, "uploaded": up}


def kill_leftover_bridges(handle, root: str) -> dict:
    """At every leg start, before the queue: a bridge of this tag left alive by an earlier attempt (a crash) is
    SIGTERMed (SIGKILL after KILL_GRACE_S), then the tag's bridge root is removed whole."""
    cmd = (f'{BRIDGE_FN}; R={q(root)}; for d in "$R"/*/; do d="${{d%/}}"; P=$(balive "$d"); [ -n "$P" ] || continue; '
           f'kill -TERM "$P" 2>/dev/null; i=0; while [ $i -lt {KILL_GRACE_S} ] && [ -d "/proc/$P" ]; do sleep 1; '
           f'i=$((i + 1)); done; kill -KILL "$P" 2>/dev/null; echo "KILLED $P $d"; done; rm -rf -- "$R"; '
           f'test ! -e "$R" && echo LEFTOVER_CLEAN')
    try:
        code, out = handle.sh(cmd, timeout=600, what="leftover bridges")
    except H.HandleError as e:
        raise C.Refuse(f"leftover bridges: {e}") from None
    if "LEFTOVER_CLEAN" not in out:
        raise C.Refuse(f"the leg's bridge root {root} could not be cleaned: {out[-300:]}")
    return {"root": root, "killed": re.findall(r"^KILLED (\d+) ", out, re.M)}


# ───────────────────────────────────────────────────────────────────────────── the guard's readings
def init_problems(init, *, model: str | None = L.MODEL, cwd: str | None = L.WS_CONTAINER, expect_skills=None) -> list:
    """cc_solver_lib.init_problems with ONE difference: the session must list exactly one MCP server, the bridge's
    in-process "tau2", connected (SSB's check refuses any server)."""
    probs = [p for p in L.init_problems(init, model=model, cwd=cwd, expect_skills=expect_skills)
             if not p.startswith("mcp servers ")]
    if init:
        sv = init.get("mcp_servers") or []
        if not (len(sv) == 1 and isinstance(sv[0], dict) and sv[0].get("name") == HB.SERVER
                and sv[0].get("status") == "connected"):
            probs.append(f"mcp servers {sv} (want exactly {HB.SERVER!r}, connected)")
    return probs


def judge(b, threshold: float, expect_skills=None) -> dict:
    """cc_solver.judge with its "init ..." stop reason recomputed through init_problems. A bridge that ended without
    claude ever starting (an infrastructure end, e.g. the customer's first call failing) has no init event: that is
    a re-run, not a defect -- only a graded end (rc 0) must show one."""
    g = C.judge(b, threshold, expect_skills)
    p = Path(b.rd) / "stream.jsonl"
    s = L.parse_stream(p.read_text(errors="replace") if p.is_file() else "")
    ip = init_problems(s["init"], model=getattr(b, "model", L.MODEL), expect_skills=expect_skills) \
        if (s["init"] or b.rc == HB.EXIT_GRADED) else []
    reasons = [r for r in g["stop_reasons"] if not r.startswith("init ")]
    if ip:
        at = sum(1 for r in reasons if r.startswith(("quota ", "HTTP 429")))      # C.judge's order: quota, init, thinking
        reasons.insert(at, f"init {ip}")
    return dict(g, init_problems=ip, stop_reasons=reasons)


def _parse_poll(out: str) -> dict:
    """cc_solver.poll_batch's sections ("@@CID@@ <cid>", "<size> <rc> <du> <alive>", "@@B64@@<data>")."""
    res = {}
    for sec in (out or "").split("@@CID@@ ")[1:]:
        first, _, rest = sec.partition("\n")
        head, sep, b64 = rest.partition("@@B64@@")
        try:
            size, rc, du, alive = head.strip().splitlines()[-1].split()
            data = base64.b64decode(b64.strip()) if sep and b64.strip() else b""
            res[first.strip()] = (int(size), rc, int(du), alive, data)
        except (ValueError, IndexError):
            res[first.strip()] = None
    return res


def poll_batch(m, boxes: list) -> dict:
    """ONE machine exec for every running tau3 unit -> cc_solver.poll_batch's {cid: (size, rc, du, alive, data)}:
    size / data = the bridge's out/claude.stream.jsonl from the box's offset (at most cc_solver.CHUNK_B); rc = the
    bridge's rc file, else RUNNING (read FIRST, so a set rc means the stream is final); du and the container's
    liveness from `evoctl tail --du`; alive = 1 only while the container AND the bridge live. A bridge gone without
    an rc therefore ends the unit as container_died (re-run once); its rc is re-read once after 2 s first (the launch
    shell writes rc just after the bridge exits)."""
    items = " ".join(q(f"{b.cid}:{b.off}:{b.vm_unit}") for b in boxes)
    cmd = (f'{BRIDGE_FN}; for s in {items}; do c="${{s%%:*}}"; r="${{s#*:}}"; o="${{r%%:*}}"; u="${{r#*:}}"; '
           f'f="$u/out/claude.stream.jsonl"; echo "@@CID@@ $c"; '
           f'set -- $({ctl.PY} {ctl.CTL} tail --cid "$c" --name {L.STDERR_NAME} --offset 0 --max 0 --du 2>/dev/null '
           f'| head -1); du="${{3:--1}}"; ca="${{4:-0}}"; case "$du" in ""|*[!0-9-]*) du=-1;; esac; '
           f'b=$(balive "$u"); rc=$(cat "$u/rc" 2>/dev/null); '
           f'if [ -z "$b" ] && [ -z "$rc" ]; then sleep 2; rc=$(cat "$u/rc" 2>/dev/null); fi; '
           f'case "$rc" in ""|*[!0-9]*) rc=RUNNING;; esac; n=$(stat -c %s "$f" 2>/dev/null || echo 0); '
           f'al=0; [ "$ca" = 1 ] && [ -n "$b" ] && al=1; echo "$n $rc $du $al"; '
           f'printf "@@B64@@"; tail -c +$((o + 1)) "$f" 2>/dev/null | head -c {C.CHUNK_B} | base64 -w0; echo; done')
    try:
        code, out = m.sh(cmd, timeout=240, what="poll", retries=2)
    except H.HandleError as e:                  # a failed poll is counted per box by poll_once, never a crash
        raise C.Refuse(str(e)) from None
    return _parse_poll(out)


# ───────────────────────────────────────────────────────────────────────────── one unit
class Tau3Box(D.LegBox):
    """A tau3 unit (the module docstring): cc_solver.Box's container, the bridge on the machine. The queue drives it
    exactly as an SSB box (setup, launch, poll, judge, collect, the grade job, wipe, result); measured() = collected
    with the bridge's rc 0."""
    ctx: dict = {}                               # run, tag, data_root, user_sim_model, key_fn (for_leg)

    @classmethod
    def for_leg(cls, skill_rows, model=None, ctx=None) -> type:
        """LegBox.for_leg + the leg's tau3 context. The key itself is never stored: ctx["key_fn"]() returns it."""
        B = super().for_leg(skill_rows, model=model)
        B.ctx = dict(ctx or {})
        return B

    def __init__(self, m, root, unit, stamp, wall_s, limits):
        super().__init__(m, root, unit, stamp, wall_s, limits)
        task, k0 = G.sample_index(unit)
        self.short_id, self.sample = short_id(task), k0 + 1
        self.bridge_root = bridge_root(self.ctx["run"], self.ctx["tag"])
        self.vm_unit = f"{self.bridge_root}/{L.unit_dir_name(unit)}"
        self.rec["argv"] = HB.stream_argv(self.model)                  # no --max-turns: the agent clock is the only limit
        self.rec["env"] = HB.claude_vars(ctl.EVO_HOME, S.TOKEN_FD)     # what claude sees (no secret: the fd number)
        self.rec["vm_unit"] = self.vm_unit

    def measured(self) -> bool:
        return self.status == "collected" and self.rc == HB.EXIT_GRADED

    # -- setup
    def upload(self):
        """Nothing of the task enters the container (the conversation is the bridge's). A skill leg: the promoted
        set as an SSB skill leg gets it (extract, the one .claude/skills link). Then /evo/work handed to evo and the
        listing inside checked -- no output/ (the tau3 solver writes none). The PREP upload rows (self.rows) are uploaded
        with the skill rows, as cc_solver.Box.upload does; tau3_leg.prepare_units sets them to [], so an EvoSkill leg
        uploads the skill rows only."""
        rows = list(self.rows) + list(self.skill_rows)
        if rows:
            tmp = self.m.dir / f"upload_{self.cid}.tgz"
            tmp.write_bytes(L.build_upload_tgz(rows))
            self.m.sb.fs.upload_file(str(tmp), f"{self.cdir}/state/upload.tgz", timeout=900)
            tmp.unlink()
            code, out, res = self.m.ctl(f"extract --cid {self.cid}", timeout=600, what="extract")
            self.fail_on(not res.get("ok"), f"extract refused: {res.get('problems') or out[-400:]}")
            self.after_extract()
        w = f"{self.cdir}/fs{L.WS_CONTAINER}"
        code, out = self.m.sh(f"set -e; W={q(w)}; test -d \"$W\"; test \"$(find \"$W\" -mindepth 1 -type l | wc -l)\" "
                              f"= {int(self.expect_link_count)}; chown -R {ctl.EVO_UID}:{ctl.EVO_GID} \"$W\"; "
                              f"find \"$W\" -printf '%y %u %m %s %P\\n'; echo WS_OK", timeout=120, what="workspace owner")
        self.rec["workspace_host_view"] = out.strip().splitlines()[-12:]
        self.fail_on("WS_OK" not in out, f"workspace handover failed: {out[-400:]}")
        from src.orchestrator import daytona_evolver as de
        code, out = self.enter([C.PY, "-c", de.VM_LISTING_PY, "/Users", ctl.VM_ROOT, ctl.POLICY_DIR], what="listing")
        try:
            listing = json.loads(out.strip().splitlines()[-1])
        except (ValueError, IndexError):
            self.fail_on(True, f"listing inside failed: {out[-400:]}")
        probs = [p for p in L.listing_problems(listing, rows, expect_links=self.expect_links,
                                               extra_dirs=self.extra_dirs) if p != "output/ missing"]
        self.rec["listing_after_upload"] = listing
        self.rec["listing_problems"] = probs
        self.save()
        log(f"{self.task}: {len(rows)} skill file(s) uploaded; listing inside {len(listing)} entries, "
            f"problems {probs}")
        self.fail_on(bool(probs), "listing inside != manifest: " + "; ".join(probs))

    def toolchain(self):
        """No LibreOffice / Python stack probe: the tau3 solver uses neither (the smoke ran none)."""
        self.rec["toolchain"] = {"skipped": "tau3: the solver's tools are the bridge's tau2 MCP server"}
        self.status = "ready"

    # -- the bridge
    def launch(self, tok: str):
        self.tok_used = tok                      # the credential this session starts with (collect scans for it)
        key = self.ctx["key_fn"]()
        try:
            job = S.unit_job(self.short_id, self.sample, self.cid, self.vm_unit, model=self.model,
                             overrides={"user_sim_model": self.ctx["user_sim_model"],
                                        **({"user_llm_args": self.ctx["user_llm_args"]} if self.ctx.get("user_llm_args") else {}),
                                        # an optional preamble before the first message; no EvoSkill leg's ctx sets it
                                        **({"first_message_prefix": self.ctx["first_message_prefix"]}
                                           if self.ctx.get("first_message_prefix") else {}),
                                        **CLAUDE_CMD},
                             root=Path(self.ctx["data_root"]), token_kind=S.token_kind())
            S.validate_job(job)
            cmd = S.launch_cmd(self.vm_unit, tok, key, py=VM_TAU3_PY, harness=VM_HARNESS)
        except S.Refuse as e:
            self.fail_on(True, str(e))
        bp = f"{self.vm_unit}/bridge.pid"
        cmd += (f" ; sleep 3; P=$(cat {q(bp)} 2>/dev/null); {{ echo \"$P\" > {BRIDGE_CGROUP}/cgroup.procs; }} "
                f"2>/dev/null; echo \"BRIDGE_PID=$P\"")
        jf = self.rd / "job.json"
        jf.write_text(json.dumps(job, indent=1) + "\n")
        self.rec.update(job=job, job_sha256=L.sha256_file(jf),
                        env=HB.claude_vars(ctl.EVO_HOME, S.TOKEN_FD, job["token_kind"]))   # the fd variable of its kind
        u = q(self.vm_unit)
        self.m.sh(f"rm -rf -- {u} && install -d -m 0700 {q(self.bridge_root)} {u} {q(self.vm_unit + '/snapshot')}",
                  timeout=60, what="unit dir")
        self.m.sb.fs.upload_file(str(jf), f"{self.vm_unit}/job.json", timeout=120)
        code, out = self.m.sh(f"sha256sum {q(self.vm_unit + '/job.json')} | cut -d' ' -f1", timeout=60, what="job sha")
        self.fail_on(out.strip().splitlines()[-1:] != [self.rec["job_sha256"]], "job.json does not verify on the machine")
        self.rec["launched_utc"] = L.now_utc()
        self.save()
        # never retried: a second bridge
        code, out = self.m.sh(cmd, timeout=180, env={S.TOK_ENV: tok, S.KEY_ENV: key}, what="launch", retries=1)
        pid = re.search(r"BRIDGE_PID=(\d+)", out or "")
        self.rec["bridge_pid"] = int(pid.group(1)) if pid else None
        self.stream_fh = open(self.rd / "stream.jsonl", "wb")
        self.t_launch = time.time()
        self.status = "running"               # last: the poller picks the box up only now
        log(f"{self.task}: bridge launched (pid {self.rec['bridge_pid']}) for {self.cid}; agent time "
            f"{job.get('agent_time_s', HB.JOB_DEFAULTS['agent_time_s'])} s")

    def end_bridge(self) -> str:
        """The bridge ended for sure (a no-op after its own end): SIGTERM, SIGKILL after KILL_GRACE_S."""
        try:
            code, out = self.m.sh(kill_sh(self.vm_unit), timeout=KILL_GRACE_S + 60, what="end bridge", retries=2)
            return out.strip()[-120:]
        except (C.Refuse, H.HandleError) as e:
            return f"failed: {e}"

    def stop(self, why: str):
        """The hard backstop (or the disk stop): SIGTERM to the BRIDGE, not stop-session -- the harness kills claude,
        writes harness.json and exits 75 (an infrastructure end: the unit is re-run)."""
        self.stopped = why
        self.rec["stopped"] = {"why": why, "utc": L.now_utc()}
        log(f"{self.task}: STOP ({why}): SIGTERM to the bridge")
        try:
            code, out = self.m.sh(f'{BRIDGE_FN}; P=$(balive {q(self.vm_unit)}); [ -n "$P" ] && kill -TERM "$P"; '
                                  f'echo "SIGTERM ${{P:-none}}"', timeout=60, what="stop bridge", retries=1)
            self.rec["stopped"]["out"] = out.strip()[-80:]
        except (C.Refuse, H.HandleError) as e:
            log(f"{self.task}: SIGTERM to the bridge failed: {e}")

    def collect(self, tok: str):
        rec = self.rec
        try:
            self.stream_fh.close()
        except Exception:  # noqa: BLE001
            pass
        cg = f"{ctl.CG_PARENT}/{self.cid}"
        code, out = self.m.sh(f"for f in memory.peak memory.events cpu.stat pids.peak; do echo \"== $f\"; "
                              f"cat {cg}/$f 2>/dev/null; done", timeout=60, what="cgroup")
        rec["cgroup_end"] = out.strip()[-1500:]
        from src.orchestrator import pool_container_evolver as pce
        code, out, res = self.m.ctl(f"probe --cid {self.cid} --phase post", timeout=300, what="post probe")
        rec["probe_post_problems"] = pce.post_probe_problems(rec.get("probe_pre") or {}, res.get("probe") or {})
        code, out, res = self.m.ctl(f"down --cid {self.cid}", timeout=300, what="down")
        rec["down_before_pack"] = res
        rec["bridge_end"] = self.end_bridge()
        # the bundle into the container's /evo/work/tau2 -- only for a graded end, AFTER down (no solver process can
        # race it); rm and cp follow no link (a tau2 the solver planted is removed, never written through)
        if self.rc == HB.EXIT_GRADED:
            w = f"{self.cdir}/fs{L.WS_CONTAINER}"
            code, out = self.m.sh(f"set -e; U={q(self.vm_unit)}; W={q(w)}; test \"$(cat \"$U/rc\")\" = 0; "
                                  f"test -d \"$U/snapshot/tau2\" && test ! -L \"$U/snapshot/tau2\"; "
                                  f"test -d \"$W\" && test ! -L \"$W\"; rm -rf -- \"$W/tau2\"; "
                                  f"cp -R -P -- \"$U/snapshot/tau2\" \"$W/tau2\"; test -f \"$W/tau2/simulation.json\"; "
                                  f"echo SHIP_OK", timeout=300, what="ship bundle")
            rec["bundle_shipped"] = "SHIP_OK" in out
            self.fail_on("SHIP_OK" not in out, f"the graded bundle could not be copied into /evo/work: {out[-300:]}")
        else:
            rec["bundle_shipped"] = False
        parts = S.CONTAINER_PARTS                # /evo/work + /home/evo/.claude (the bridge's stream is not in /evo/out)
        code, out, res = self.m.ctl(f"pack --cid {self.cid} " + " ".join(f"--part {q(p)}" for p in parts),
                                    timeout=900, what="pack")
        rec["pack"] = {k: res.get(k) for k in ("ok", "bytes", "packed", "refused")}
        self.save()
        if not res.get("ok"):
            raise C.Refuse(f"{self.task}: pack refused: {res.get('problems') or out[-300:]}")
        local = self.rd / "container_out.tgz"
        self.m.sb.fs.download_file(f"{self.cdir}/state/out.tgz", str(local))
        rec["download_refused_members"] = S.safe_extract(local, self.rd / "container")
        # the unit dir (job.json, bridge.log, rc, out/*, snapshot/tau2/*) -> runs/<ud>/bridge/
        code, out = self.m.sh(f"tar -czf {q(self.vm_unit + '.tgz')} -C {q(self.vm_unit)} . && echo TAR_OK",
                              timeout=300, what="unit tar")
        self.fail_on("TAR_OK" not in out, f"unit tar failed: {out[-300:]}")
        btgz = self.rd / "bridge.tgz"
        self.m.sb.fs.download_file(f"{self.vm_unit}.tgz", str(btgz))
        bdir = self.rd / "bridge"
        rec["bridge_refused_members"] = S.safe_extract(btgz, bdir)
        btgz.unlink()
        streamed, machine = self.rd / "stream.jsonl", bdir / "out" / "claude.stream.jsonl"
        rec["log_round_trip"] = {"streamed_sha256": _sha_or_empty(streamed), "bridge_sha256": _sha_or_empty(machine)}
        rec["log_round_trip"]["equal"] = rec["log_round_trip"]["streamed_sha256"] == rec["log_round_trip"]["bridge_sha256"]
        ts = self.t_done or time.time()
        rec["container_times"] = {"t_start": str(int(self.t_launch)), "t_end": str(int(ts)),
                                  "rc": None if self.rc is None else str(self.rc)}
        rec["output"] = None
        # every downloaded byte: the credential and its prefix (daytona_evolver.token_scan; the prefix of the
        # credential's kind), the key and sk-or-v1-
        from src.orchestrator import daytona_evolver as de
        hits, n = de.token_scan([self.rd], tok)
        sec = S.secret_scan([self.rd], {"claude-token": tok, "token-prefix": S.token_prefix(),
                                        "openrouter-key": self.ctx["key_fn"](), "openrouter-prefix": S.OR_PREFIX})
        rec["token_scan"] = {"hits": sorted(set(hits) | set(sec["hits"])), "scanned_files": n}
        scan = (rec["down_before_pack"] or {}).get("scan") or {}
        mhits = [h for part in scan.values() for h in (part or {}).get("hits") or []] + \
            list(((rec["down_before_pack"] or {}).get("tmpfs_scan") or {}).get("hits") or [])
        rec["container_token_hits"] = mhits
        self.save()
        if rec["token_scan"]["hits"] or mhits:
            raise C.TokenHit(f"{self.task}: A SECRET'S BYTES FOUND ({len(rec['token_scan']['hits'])} local, "
                             f"{len(mhits)} in the container) -- rotate the Claude credential ({self.m.token_name}) "
                             f"/ the OpenRouter key")
        # the first message the bridge actually sent (the record's user message), and the harness's own brief
        fm = self.rd / "first_message.txt"
        fm.unlink(missing_ok=True)
        stdin = bdir / "out" / "claude.stdin.jsonl"
        text = S.first_user_message(stdin.read_text(errors="replace")) if stdin.is_file() else None
        if text is not None:
            fm.write_text(text)
        rec["first_message"] = None if text is None else {
            "sha256": L.sha256_bytes(text.encode("utf-8")), "chars": len(text),
            "template_sha256": HB.first_message_sha256(),
            "matches_template": S.first_message_check(text, None, HB.FIRST_MESSAGE_TEMPLATE)["matches_template"]}
        hj = _jload(bdir / "out" / "harness.json") or {}
        cc = hj.get("cc") or {}
        rec["bridge"] = {"rc": self.rc, "end": hj.get("end"), "exit_code": hj.get("exit_code"),
                         "graded_bundle": hj.get("graded_bundle"), "agent_time": hj.get("agent_time"),
                         "customer": hj.get("customer"), "mcp_servers": cc.get("mcp_servers"),
                         "id_sources": cc.get("id_sources"), "transcript": cc.get("transcript")}
        code, out = self.m.sh(f"rm -rf -- {q(self.vm_unit)} {q(self.vm_unit + '.tgz')}; echo RM_OK", timeout=120,
                              what="unit rm")
        self.save()
        self.status = "collected"

    def wipe(self):
        """The bridge ended (if it still runs), its unit dir removed, then the container `down --wipe`d."""
        try:
            code, out = self.m.sh(kill_sh(self.vm_unit) + f"; rm -rf -- {q(self.vm_unit)} {q(self.vm_unit + '.tgz')}",
                                  timeout=KILL_GRACE_S + 120, what="bridge wipe", retries=2)
            self.rec["bridge_wipe"] = out.strip()[-120:]
        except (C.Refuse, H.HandleError) as e:
            self.rec["bridge_wipe"] = f"failed: {e}"
        return super().wipe()

    def result(self) -> dict:
        """Box.result(), then the tau3 verdict: completed = measured() (the bridge's rc 0, collected), infra_error =
        any other rc, the init judged by init_problems (one tau2 server), the bridge's brief."""
        out = super().result()
        out.update(completed=self.measured(), infra_error=self.rc != HB.EXIT_GRADED,
                   init_problems=init_problems(out.get("init"), model=self.model,
                                               expect_skills=L.ANY_SKILLS if self.skill_leg else None),
                   bridge=self.rec.get("bridge"), bundle_shipped=self.rec.get("bundle_shipped"))
        L.jdump(self.rd / "RESULT.json", out)
        return out
