#!/usr/bin/env python3
"""The STATEFUL evolver session: ONE long-lived container and ONE claude session for the whole run, on the run's
pool machine. RUNTIME `evolver_session: "stateful"`.

The loop's dispatch argv is UNCHANGED: `scripts/daytona/evolver_container.py <epoch> <prompt> <digests>
<staging>` routes here (cmd_dispatch) when the run's RUNTIME.json says "stateful". Run-level commands:

    evolver_session.py status  --run-dir R                       the state record (+ `status --cid` from the machine)
    evolver_session.py finish  --run-dir R [--abort] [--machine-absent-ok]   down --wipe, ledger, record

Per epoch (the base run() order: gate -> prepare -> create -> probes -> upload -> launch -> stream -> collect;
finally delete; install), what differs from the per-epoch container driver:
  prepare   the delta manifest of THIS epoch (stateful_layout.build_stateful_manifest, container-native paths);
            receipt.daytona.container.session, upload_manifest_cumulative, path_view;
  create    epoch 1: ContainerMixin's attach + provision, then the state record .cache/evolver_state/<run>/
            container.json is CREATED; epoch >= 2: reattach() (a lease per epoch; the pool's current machine must be
            the session's and the container up, else rebuild() from the newest verified snapshot);
  probes    epoch 1 / rebuild: `up` + the PRE probe (saved in the state); epoch >= 2: `epoch-begin --clear /evo/staging`;
  upload    the delta tgz -> `extract --incremental`; the listing inside == the CUMULATIVE manifest outside the
            evolver-writable trees (/evo/staging, /evo/evolver_workspace, /evo/work, /evo/out);
  launch    epoch 1 `--session-id <sid>`, epoch N `--resume <sid>`, `--append-system-prompt` from the EVO_SYSTEM base64 variable (no file);
  collect   `epoch-end` (freeze -> scan -> pack -> thaw; the container stays up) -> download -> audits -> snapshot to
            .cache/evolver_state/<run>/e<N>/ (transcripts + notes + receipt); a turn the API rejected for a usage
            limit / an HTTP 429 is recorded `failed`, so the dispatch's re-run after the reset resumes it;
  delete    the lease is released and the state written; NEVER `down --wipe` (that is `finish`).
Nothing here touches the machine before the receipt and trace are written; every exit code is the base's.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import shlex
import shutil
import sys
import tarfile
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import daytona_lib_evo as dle  # noqa: E402
import evolver_container as EC  # noqa: E402
import evolver_container_ctl as ctl  # noqa: E402
import tau3_evolver_daytona as T  # noqa: E402
from src.orchestrator import claude_pins  # noqa: E402
from src.orchestrator import daytona_evolver as de  # noqa: E402
from src.orchestrator import pool_container_evolver as pce  # noqa: E402
from src.orchestrator import stateful_layout as SL  # noqa: E402

Refuse = T.Refuse
log = T.log
jdump = T.jdump
q = shlex.quote
EXIT_USAGE, EXIT_EGRESS, EXIT_MANIFEST = T.EXIT_USAGE, T.EXIT_EGRESS, T.EXIT_MANIFEST
EXIT_TOKEN, EXIT_IDENTITY, EXIT_PIN = T.EXIT_TOKEN, T.EXIT_IDENTITY, T.EXIT_PIN
EXIT_CLAUDE_RC, EXIT_TIMEOUT, EXIT_DELETE = T.EXIT_CLAUDE_RC, T.EXIT_TIMEOUT, T.EXIT_DELETE
DRIVER = pce.DRIVER                     # the ledger rows keep the container driver's name (ledger_container_down keys on it)
PY, CTL = ctl.PY, ctl.CTL
TEARDOWN_SCHEMA = "evoskill.evolver_session_teardown/1"
DEFAULT_MAX_SECONDS = 86400             # egress proxy life when RUNTIME daytona_leg.pool_ttl_minutes is absent
FINISH_LEASE_S = 900                    # the teardown's lease cap
DOWN_RETRIES = 3
# epoch statuses in the state record
S_PREPARED, S_UPLOADED, S_LAUNCHED, S_COLLECTED, S_FAILED = "prepared", "uploaded", "launched", "collected", "failed"


# ───────────────────────────────────────────────────────────────────────────── small helpers
def new_cid() -> str:
    """Run-scoped container id: run-<yyyymmddtHHMMSS>-<hex6>; ctl.CID_RE accepts it."""
    return f"run-{time.strftime('%Y%m%dt%H%M%S').lower()}-{uuid.uuid4().hex[:6]}"


def probe_sha256(probe) -> str:
    """The creation-time PRE probe's hash, re-checked every epoch. ONE definition:
    stateful_layout.probe_sha256 (the receipt check re-derives with it) when present, else the same canonical
    form here (json.dumps sort_keys, compact separators)."""
    f = getattr(SL, "probe_sha256", None)
    if f is not None:
        return f(probe)
    return hashlib.sha256(json.dumps(probe, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def pool_ttl_seconds(run_dir) -> int:
    """RUNTIME daytona_leg.pool_ttl_minutes * 60 (the egress proxy's max_seconds), DEFAULT_MAX_SECONDS when absent."""
    try:
        v = (json.loads((Path(run_dir) / "RUNTIME.json").read_text()).get("daytona_leg") or {}).get("pool_ttl_minutes")
    except (OSError, ValueError):
        v = None
    try:
        return int(float(v) * 60) if v is not None and not isinstance(v, bool) and float(v) > 0 else DEFAULT_MAX_SECONDS
    except (TypeError, ValueError):
        return DEFAULT_MAX_SECONDS


def copy_tree_nofollow(src: Path, dst: Path) -> int:
    """dst/ = the regular files of src/ through plain directories, no symlink followed (one anywhere is refused:
    the members came from the container's writable trees). Returns the file count; an absent src -> empty dst."""
    src, dst = Path(src), Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    if not src.exists() and not src.is_symlink():
        return 0
    if src.is_symlink() or not src.is_dir():
        raise Refuse(EXIT_MANIFEST, f"{src} is not a plain directory")
    n = 0
    for dp, dns, fns in os.walk(src, followlinks=False):
        rel = Path(dp).relative_to(src)
        for d in sorted(dns):
            if (Path(dp) / d).is_symlink():
                raise Refuse(EXIT_MANIFEST, f"symlink {Path(dp, d)} in a tree copied to the host")
            (dst / rel / d).mkdir(exist_ok=True)
        for f in sorted(fns):
            p = Path(dp) / f
            if p.is_symlink() or not p.is_file():
                raise Refuse(EXIT_MANIFEST, f"{p} is not a regular file")
            shutil.copyfile(p, dst / rel / f)
            n += 1
    return n


# ───────────────────────────────────────────────────────────────────────────── the driver
class StatefulMixin:
    """Mixed BEFORE ContainerMixin: StatefulEvolverRun(StatefulMixin, ContainerEvolverRun). The machine-facing
    steps that differ in a stateful session (module doc); everything else is ContainerMixin's / the base's."""

    placement = pce.PLACEMENT
    session = SL.STATEFUL
    # no prompt / system-prompt FILE in the container; both travel as base64 shell variables (launch)
    sysprompt_vm_file = None
    prompt_vm_file = None

    # ── identity: the run's cid / sid from the state record (or minted at the first epoch)
    def cinit(self, **kw) -> None:
        EC.ContainerMixin.cinit(self, **kw)
        self.state, self.state_error = None, None
        try:
            self.state = SL.load_state(self.run_dir)
        except ValueError as e:
            self.state_error = str(e)
        st = self.state
        self.attached_from = str(SL.state_path(self.run_dir))
        self.fresh = st is None or bool(st.get("torn_down"))
        if self.fresh:
            self.cid, self.sid_session, self.created_epoch = new_cid(), str(uuid.uuid4()), self.epoch
            self.prior_rebuilds = []
        else:
            self.cid, self.sid_session, self.created_epoch = st["cid"], st["sid"], int(st["created_epoch"])
            self.prior_rebuilds = list(st.get("rebuilds") or [])
        self.cdir = ctl.cdir(self.cid)
        prior = ((st or {}).get("epochs") or {}).get(str(self.epoch)) if not self.fresh else None
        self.prior_epoch_status = (prior or {}).get("status")
        # --resume: any epoch after the container's first, or a re-run of an epoch whose turn was already launched
        self.resume = (not self.fresh) and (self.epoch > self.created_epoch
                                            or self.prior_epoch_status in (S_LAUNCHED, S_FAILED))
        self.uploaded_before = [r for r in ((st or {}).get("uploaded") or []) if int(r.get("epoch", 0)) < self.epoch] \
            if not self.fresh else []
        self.rebuilt = None
        self.launched = False
        self.collected = False
        self.system_prompt = None
        self.view = []

    # ── the layout of THIS epoch (container paths; nothing has to exist)
    def readonly_roots(self) -> list:
        # an EMPTY skills_vK uploads no file, so /evo/skills/epoch_K does not exist in the container: not a root
        skills = [SL.skills_dir(K) for K in range(1, self.epoch)
                  if any(f.is_file() for f in (self.run_dir / f"skills_v{K}").rglob("*"))]
        return ([SL.traces_dir(K) for K in range(self.epoch)] + [SL.solver_ws_dir(K) for K in range(self.epoch)]
                + [SL.VM_SCORES] + skills + [SL.VM_REFERENCE])

    def add_dirs_for(self) -> list:
        return self.readonly_roots() + [SL.VM_STAGING, SL.VM_WORKSPACE]

    def download_parts(self) -> list:
        return [SL.VM_STAGING.lstrip("/"), SL.VM_WORKSPACE.lstrip("/"), de.VM_PROJECTS.lstrip("/"), de.VM_OUT.lstrip("/")]

    def after_hashes(self, after) -> None:
        self.receipt["evidence_sha256_after_vm"] = SL.after_hashes_vm(after, self.view)

    def build_manifest(self, prev_lib):
        """The seam: the delta rows of this epoch, container-keyed (stateful_layout.build_stateful_manifest). The
        evidence dirs the loop passed must be exactly evo_e0..evo_e{N-1} (the path view's), prev_lib skills_v{N-1}."""
        try:
            self.view = SL.path_view(self.run_dir, self.epoch)
            want_ev = sorted(str(Path(pm.mac).parent) for pm in self.view if pm.role == SL.ROLE_TRACES)
            got_ev = sorted(os.path.realpath(str(d)) for d in self.evidence_dirs)
            if got_ev != want_ev:
                raise ValueError(f"evidence dirs {got_ev} are not this run's evo_e0..evo_e{self.epoch - 1} {want_ev}")
            if self.epoch > 1 and os.path.realpath(str(prev_lib)) != next(pm.mac for pm in self.view if pm.role == SL.ROLE_SEED):
                raise ValueError(f"previous library {prev_lib} is not this run's skills_v{self.epoch - 1}")
            rows, gen = SL.build_stateful_manifest(
                run_dir=self.run_dir, epoch=self.epoch, env_text=de.render_env_file(self.env),
                hooks_text=de.hook_settings_text(), split=self.split, uploaded_before=list(self.uploaded_before))
        except (ValueError, KeyError) as e:
            raise Refuse(EXIT_MANIFEST, f"upload manifest (stateful): {e}")
        return rows, gen

    def container_spec(self) -> dict:
        try:
            limits = pce.container_limits_cfg(self.cfg)
        except ValueError as e:
            raise Refuse(EXIT_USAGE, str(e))
        return {"schema": ctl.SCHEMA, "cid": self.cid, "rw": [dict(r) for r in SL.STATEFUL_RW], "ro_rebind": [],
                "readonly_roots": sorted(set(self.readonly_roots())), "limits": limits,
                "cc": {"version": ctl.CC_VERSION, "sha256": ctl.CC_SHA256},
                "egress": {"allow": ctl.ALLOW, "must_reach": list(de.MUST_REACH), "must_refuse": list(de.MUST_REFUSE),
                           "max_seconds": pool_ttl_seconds(self.run_dir), "idle_timeout_s": 1800}}

    # ── prepare: the session's refusals, then the base prepare with the seams, then the session facts
    def prepare(self) -> None:
        rd = self.run_dir
        if self.state_error:
            raise Refuse(EXIT_USAGE, f"state record: {self.state_error}")
        try:
            mode = SL.evolver_session(rd)
        except ValueError as e:
            raise Refuse(EXIT_USAGE, str(e))
        if mode != SL.STATEFUL:
            raise Refuse(EXIT_USAGE, f"RUNTIME.json {SL.SESSION_KEY}={mode!r}: evolver_session.py drives only a "
                                     f"'stateful' run (per-epoch runs use evolver_container.py's own driver)")
        env_mode = os.environ.get(SL.ENV_SESSION)
        if env_mode != SL.STATEFUL:
            raise Refuse(EXIT_USAGE, f"{SL.ENV_SESSION}={env_mode!r} != RUNTIME.json's {SL.STATEFUL!r} (the loop "
                                     f"exports it; an env that disagrees with the file is refused)")
        try:
            lv = claude_pins.effort(os.environ)
        except claude_pins.ModelPinError as e:
            raise Refuse(EXIT_USAGE, str(e))
        if lv is not None:
            raise Refuse(EXIT_USAGE, f"stateful evolver: EVOSKILL_CLAUDE_EFFORT must opt out (default); got --effort {lv}")
        try:
            tm = de.evolver_template_mode(rd)
        except ValueError as e:
            raise Refuse(EXIT_USAGE, f"prompt mode: {e}")
        if not tm:
            raise Refuse(EXIT_USAGE, "stateful evolver: RUNTIME prompt_mode must be 'template' (the loop renders the message)")
        if self.mode != "all_epochs":
            raise Refuse(EXIT_USAGE, f"stateful evolver: EVOSKILL_EVOLVER_EVIDENCE must be all_epochs, got {self.mode!r}")
        want = rd / SL.SYSTEM_PROMPT_FILE
        sp = os.environ.get(SL.ENV_SYSTEM_PROMPT)
        if not sp or Path(sp).absolute() != want.absolute():
            raise Refuse(EXIT_USAGE, f"{SL.ENV_SYSTEM_PROMPT} must name {want}, got {sp!r}")
        if want.is_symlink() or not want.is_file():
            raise Refuse(EXIT_USAGE, f"{want} missing (rendered once at epoch 1 by the loop)")
        sb = want.read_bytes()
        if len(sb) > de.MAX_PROMPT_BYTES:
            raise Refuse(EXIT_USAGE, f"system prompt is {len(sb)} B > {de.MAX_PROMPT_BYTES} (one argv string)")
        self.system_prompt, self.system_prompt_sha256 = want, de.sha256_bytes(sb)
        st = self.state
        if not self.fresh:
            epochs = {int(k): v for k, v in (st.get("epochs") or {}).items()}
            newest = max(epochs) if epochs else self.created_epoch
            if self.prior_epoch_status == S_COLLECTED:
                raise Refuse(EXIT_USAGE, f"epoch {self.epoch} was already collected in session {st['sid'][:8]} "
                                         f"(container {st['cid']}); a turn is taken once. Remedy: `evolver_session.py "
                                         f"finish --run-dir {rd} --abort` for a new session")
            if self.epoch < self.created_epoch or any(k > self.epoch for k in epochs):
                raise Refuse(EXIT_USAGE, f"epoch {self.epoch} is earlier than the session's newest epoch {newest}; a "
                                         f"session cannot go back (finish --abort for a new session)")
            if self.epoch > newest + 1:
                raise Refuse(EXIT_USAGE, f"epoch {self.epoch} follows epoch {newest} in this session; epochs "
                                         f"{newest + 1}..{self.epoch - 1} never ran")
            if st.get("system_prompt_sha256") != self.system_prompt_sha256:
                raise Refuse(EXIT_USAGE, f"{want.name} changed since epoch 1 (sha {self.system_prompt_sha256[:12]} != "
                                         f"the session's {str(st.get('system_prompt_sha256'))[:12]}); the system prompt "
                                         f"is fixed for the run")
        super().prepare()
        if not self.fresh:
            e1 = [r for r in self.uploaded_before if r["vm_path"] == de.VM_ENV_FILE]
            now_sha = de.sha256_bytes(de.render_env_file(self.env).encode())
            if e1 and e1[0]["sha256"] != now_sha:
                raise Refuse(EXIT_USAGE, "the evolver env (model / pins) changed since epoch 1; the container's "
                                         f"{de.VM_ENV_FILE} is epoch 1's")
        dd, c = self.receipt["daytona"], self.c()
        dd["driver"] = str(Path(__file__).relative_to(REPO))
        dd["session_id"] = self.sid_session
        c["session"] = {"mode": SL.STATEFUL, "cid": self.cid, "sid": self.sid_session, "created_epoch": self.created_epoch,
                        "epoch": self.epoch, "resume": self.resume, "attached_from": self.attached_from, "kept": True,
                        "system_prompt_sha256": self.system_prompt_sha256, "rebuilt": None,
                        "prior_attempt": self.prior_epoch_status}
        self.receipt["upload_manifest_cumulative"] = list(self.uploaded_before) + [dict(r, epoch=self.epoch) for r in self.rows]
        self.receipt["path_view"] = [pm.as_dict() for pm in self.view]
        self.receipt["readonly_roots"] = self.readonly_roots()
        dd["snapshot_dir"] = str(SL.snapshot_dir(rd, self.epoch))
        dd["workspace_dir"] = str(rd / SL.WORKSPACE_DIR.format(epoch=self.epoch))
        self.save_receipt()
        log(f"stateful session {self.sid_session[:8]} container {self.cid}: epoch {self.epoch} "
            f"({'resume' if self.resume else 'first turn'}; created at epoch {self.created_epoch}); delta "
            f"{len(self.rows)} rows, cumulative {len(self.receipt['upload_manifest_cumulative'])}")

    # ── the state record
    def _write_state(self) -> None:
        if self.state is not None:
            SL.write_state(self.run_dir, self.state)

    def _mark(self, status: str, **extra) -> None:
        if self.state is None:
            return
        cur = dict((self.state.setdefault("epochs", {})).get(str(self.epoch)) or {})
        cur.update(status=status, receipt=str(self.receipt_path), utc=dle.now(), **extra)
        self.state["epochs"][str(self.epoch)] = cur
        self._write_state()

    def _state_create(self) -> None:
        c = self.c()
        self.state = {
            "schema": SL.STATE_SCHEMA, "run": SL.run_name(self.run_dir), "run_dir": str(self.run_dir),
            "cid": self.cid, "sandbox_id": self.sb.id, "sandbox_name": self.sb.name, "sid": self.sid_session,
            "created_epoch": self.epoch, "created_utc": dle.now(), "system_prompt_sha256": self.system_prompt_sha256,
            "container": self._container_record(c),
            "generated": {p: base64.b64encode(b).decode() for p, b in (self.gen or {}).items()},
            "uploaded": [], "epochs": {}, "rebuilds": list(self.prior_rebuilds), "torn_down": None}
        self._mark(S_PREPARED)

    def _container_record(self, c: dict) -> dict:
        return {"spec": self.spec, "facts": c.get("facts"), "limits": c.get("limits"), "plan": c.get("plan"),
                "init_sha256": c.get("init_sha256"), "expected_top": c.get("expected_top"),
                "expected_mounts": c.get("expected_mounts"), "probe_pre": None, "probe_pre_sha256": None,
                "cc_upload": c.get("cc_upload"), "pre_upload": None}

    def _receipt_from_state_container(self) -> None:
        cc = self.state["container"]
        c = self.c()
        c.update(spec=cc["spec"], facts=cc["facts"], limits=cc["limits"], plan=cc["plan"], init_sha256=cc["init_sha256"],
                 expected_top=cc["expected_top"], expected_mounts=cc["expected_mounts"], probe_pre=cc["probe_pre"],
                 probe_pre_sha256=cc["probe_pre_sha256"], cc_upload=cc["cc_upload"])
        self.spec, self.plan_ = cc["spec"], cc["plan"]

    # ── create: epoch 1 provisions; later epochs reattach (or rebuild)
    def create(self) -> None:
        cc_local = EC.ensure_cc_binary(self.cfg, self.run_dir)
        self.client = dle.client()
        row = self.attach_machine(self.registry_file())
        dd = self.receipt["daytona"]
        self.sb = self.client.get(row["sandbox_id"])
        rb = dle.readback(self.sb)
        dd.update(sandbox_id=self.sb.id, sandbox_name=self.sb.name, readback_create=rb)
        mp = pce.machine_problems(rb, name=self.sb.name, state=rb.get("state"), row=row,
                                  require_snapshot_ref=self.leg_snapshot_ref(), prefixes=(pce.POOL_PREFIX,))
        if row.get("sandbox_name") and row.get("sandbox_name") != self.sb.name:
            mp.append(f"sandbox name {self.sb.name!r} != the registry's {row.get('sandbox_name')!r}")
        self.save_receipt()
        if mp:
            raise Refuse(EXIT_EGRESS if any("allow" in p or "network" in p for p in mp) else EXIT_IDENTITY,
                         "pool machine: " + "; ".join(mp))
        log(f"pool machine {self.sb.id[:8]} {self.sb.name}: allow {rb.get('domain_allow_list')!r}, snapshot "
            f"{rb.get('snapshot_ref')}; lease {(dd.get('lease') or {}).get('path')}")
        if self.fresh:
            dle.ledger(event="container_attach", driver=DRIVER, tag=self.run_dir.name, id=self.sb.id, cid=self.cid,
                       epoch=self.epoch, session=self.sid_session)
            self.install_ctl()
            self._provision(cc_local, wipe_own=[])
            self._state_create()
        else:
            self.reattach(cc_local)

    def _provision(self, cc_local: Path, *, wipe_own: list) -> None:
        """The 'no other evolver container on the machine' rule (a stale container of THIS run's old cid is wiped
        first, as cmd_cleanup does), Claude Code, `prepare --spec-b64`, the plan re-derived. The legs' `cc-*`
        containers (solver units) are not a conflict -- the run's legs share this machine
        (EC.evolver_container_conflicts, the per-epoch driver's rule)."""
        code, out, lst = self.ctl("list", what="list", retries=2)
        have = list(lst.get("containers") or [])
        others = [x for x in EC.evolver_container_conflicts(have) if x not in wipe_own]
        if others:
            raise Refuse(EXIT_IDENTITY, f"evolver container(s) already on the machine: {others} "
                                        f"(scripts/daytona/evolver_container.py cleanup)")
        for old in [x for x in have if x in wipe_own]:
            code, out, res = self.ctl(f"down --cid {q(old)} --wipe", timeout=300, what="down wipe", retries=DOWN_RETRIES)
            ok = bool(res.get("verified") and res.get("wiped"))
            dle.ledger(event="container_down", driver=DRIVER, tag=self.run_dir.name, id=self.sb.id, cid=old, verified=ok,
                       stale=True)
            self.c().setdefault("wiped_stale", []).append({"cid": old, "verified": res.get("verified"), "wiped": res.get("wiped")})
            if not ok:
                raise Refuse(EXIT_IDENTITY, f"stale container {old} of this run could not be wiped: {res}")
        self.install_cc(cc_local)
        b64 = base64.b64encode(json.dumps(self.spec).encode()).decode()
        code, out, res = self.ctl(f"prepare --spec-b64 {b64}", timeout=900, what="prepare")
        c = self.c()
        c["prepare"] = {k: res.get(k) for k in ("ok", "problems", "cdir", "etc", "expected_top", "cc_host_sha256")}
        c.update(facts=res.get("facts"), limits=res.get("limits"), plan=res.get("plan"))
        if res.get("ok"):
            pl = ctl.plan(self.spec, res["facts"], res["limits"])
            if pl != res.get("plan"):
                raise Refuse(EXIT_IDENTITY, "the machine's plan differs from the one this driver derives")
            c["init_sha256"] = hashlib.sha256(ctl.render_init(self.spec, res["facts"], res["limits"], pl).encode()).hexdigest()
            c["expected_top"], c["expected_mounts"] = ctl.expected_top(pl), ctl.expected_mount_points(pl)
            self.plan_ = pl
        self.save_receipt()
        if not res.get("ok"):
            raise Refuse(EXIT_IDENTITY, f"prepare refused: {res.get('problems') or out[-400:]}")

    def reattach(self, cc_local: Path) -> None:
        """Epoch >= 2 (or a re-run): the pool's current machine must be the session's and its container up with
        nothing else beside it; otherwise rebuild(). Records container_reattach."""
        st, c = self.state, self.c()
        why = None
        if self.sb.id != st["sandbox_id"]:
            why = f"the pool's current machine {self.sb.id[:8]} is not the session's {str(st['sandbox_id'])[:8]}"
        self.install_ctl()
        if why is None:
            code, out, status = self.ctl(f"status --cid {q(self.cid)}", timeout=60, what="status", retries=2)
            code, out, lst = self.ctl("list", what="list", retries=2)
            c["attach"] = {"status": {k: status.get(k) for k in ("cid", "exists", "init_alive", "proxy_alive")},
                           "list": lst.get("containers"), "ledger": None}
            if not status.get("init_alive"):
                why = f"the session's container {self.cid} is not up"
            elif EC.evolver_container_conflicts(lst.get("containers") or []) != [self.cid]:
                # the legs' cc-* containers beside it are not a reason to rebuild (see _provision)
                why = f"evolver containers on the machine {lst.get('containers')} != [{self.cid}]"
        if why is not None:
            self.rebuild(why, cc_local)
            return
        dle.ledger(event=SL.LEDGER_REATTACH, driver=DRIVER, tag=self.run_dir.name, id=self.sb.id, cid=self.cid,
                   epoch=self.epoch, session=self.sid_session)
        c["attach"]["ledger"] = SL.LEDGER_REATTACH
        self._receipt_from_state_container()
        self.save_receipt()
        log(f"reattached to container {self.cid} on {self.sb.id[:8]} (epoch {self.epoch}, session {self.sid_session[:8]})")

    def _newest_verified_snapshot(self) -> tuple:
        """(dir, epoch) of the newest snapshot of this run that verifies and belongs to this session, else (None, None)."""
        root = SL.STATE_ROOT / SL.run_name(self.run_dir)
        skipped = []
        if root.is_dir():
            for d in sorted((x for x in root.iterdir() if x.is_dir() and x.name[:1] == "e" and x.name[1:].isdigit()),
                            key=lambda x: int(x.name[1:]), reverse=True):
                probs = SL.verify_snapshot(d)
                if not probs:
                    try:
                        man = json.loads((d / SL.SNAPSHOT_MANIFEST).read_text())
                    except (OSError, ValueError) as e:
                        probs = [f"manifest unreadable: {e}"]
                    else:
                        if man.get("sid") != self.sid_session:
                            probs = [f"snapshot of another session {str(man.get('sid'))[:8]}"]
                if not probs:
                    self.c()["snapshots_skipped"] = skipped
                    return d, int(d.name[1:])
                skipped.append({"snapshot": str(d), "problems": probs[:3]})
        self.c()["snapshots_skipped"] = skipped
        return None, None

    def rebuild(self, why: str, cc_local: Path, *, snapshot: tuple | None = None) -> None:
        """A NEW cid on the current machine (a stale container of the old cid on the same machine is wiped
        first), prepare, then `restore` from the newest verified snapshot of this session (or the one given). The
        base run() then continues with probes() (up + PRE probe + epoch-begin) and upload() (the cumulative
        re-upload of the epochs before this one + this epoch's delta), and launches with --resume."""
        st, c = self.state, self.c()
        if snapshot is None:
            snap, M = self._newest_verified_snapshot()
        else:
            snap, M = snapshot
        collected_any = any((e or {}).get("status") == S_COLLECTED for e in (st.get("epochs") or {}).values())
        if snap is None and collected_any:
            raise Refuse(EXIT_IDENTITY, f"rebuild ({why}): no verified snapshot of session {self.sid_session[:8]} under "
                                        f"{SL.STATE_ROOT / SL.run_name(self.run_dir)}; nothing to resume from "
                                        f"({c.get('snapshots_skipped')})")
        old_cid, old_sandbox = self.cid, st["sandbox_id"]
        self.cid = new_cid()
        self.cdir = ctl.cdir(self.cid)
        self.spec = self.container_spec()
        c.update(cid=self.cid, cdir=self.cdir, spec=self.spec)
        try:
            tr = json.loads(self.trace_path.read_text())
            tr.update(container=self.cid, rebuilt_from=old_cid)
            jdump(self.trace_path, tr)
        except (OSError, ValueError):
            pass
        dle.ledger(event="container_attach", driver=DRIVER, tag=self.run_dir.name, id=self.sb.id, cid=self.cid,
                   epoch=self.epoch, session=self.sid_session, rebuild_of=old_cid, why=why)
        self._provision(cc_local, wipe_own=[old_cid] if self.sb.id == old_sandbox else [])
        if snap is not None:
            self._restore(snap)
        else:
            self.resume = False                   # nothing was ever collected: a fresh session under the same sid
        rec = {"why": why, "from_snapshot": str(snap) if snap else None, "from_epoch": M, "old_cid": old_cid,
               "old_sandbox_id": old_sandbox, "new_cid": self.cid, "new_sandbox_id": self.sb.id, "utc": dle.now()}
        self.rebuilt = rec
        c.setdefault("session", {})["rebuilt"] = rec
        c["session"]["cid"] = self.cid
        c["session"]["resume"] = self.resume
        st.update(cid=self.cid, sandbox_id=self.sb.id, sandbox_name=self.sb.name)
        st["container"] = self._container_record(c)
        st.setdefault("rebuilds", []).append({"epoch": self.epoch, **rec})
        st["uploaded"] = [r for r in st.get("uploaded") or [] if int(r.get("epoch", 0)) < self.epoch]
        self._mark(S_PREPARED, rebuilt=rec["utc"])
        self.save_receipt()
        log(f"REBUILT container {self.cid} on {self.sb.id[:8]} ({why}); restored from {snap}")

    def _restore(self, snap: Path) -> None:
        """e<M>/projects/** as home/evo/.claude/projects/... and e<M>/workspace/** as evo/evolver_workspace/... ->
        `evoctl restore` (before `up`; members must be regular files under those two roots)."""
        tgz = self.q / "restore.tgz"
        n = 0
        with tarfile.open(tgz, "w:gz", format=tarfile.PAX_FORMAT) as tf:
            for sub, arc_root in (("projects", de.VM_PROJECTS.lstrip("/")),
                                  (SL.SNAPSHOT_WORKSPACE, SL.VM_WORKSPACE.lstrip("/"))):
                base = Path(snap) / sub
                if not base.is_dir():
                    continue
                for f in sorted(base.rglob("*")):
                    if f.is_symlink():
                        raise Refuse(EXIT_MANIFEST, f"symlink in the snapshot {f}")
                    if f.is_file():
                        ti = tf.gettarinfo(str(f), arcname=f"{arc_root}/{f.relative_to(base).as_posix()}")
                        ti.uid = ti.gid = 0
                        ti.uname = ti.gname = "root"
                        with open(f, "rb") as fh:
                            tf.addfile(ti, fh)
                        n += 1
        self.sb.fs.upload_file(str(tgz), f"{self.cdir}/state/restore.tgz", timeout=1800)
        code, out, res = self.ctl(f"restore --cid {q(self.cid)}", timeout=900, what="restore")
        self.c()["restore"] = {"snapshot": str(snap), "members": n, **res}
        self.save_receipt()
        if not res.get("ok"):
            raise Refuse(EXIT_IDENTITY, f"restore refused: {res.get('problems') or out[-400:]}")

    # ── probes: up + PRE probe once per container; epoch-begin every later epoch
    def probes(self) -> None:
        st, c = self.state, self.c()
        if self.fresh or self.rebuilt:
            EC.ContainerMixin.probes(self)
            cc = st["container"]
            cc["probe_pre"] = c.get("probe_pre")
            cc["probe_pre_sha256"] = c["probe_pre_sha256"] = probe_sha256(c.get("probe_pre"))
            cc["pre_upload"] = dict(self.listing["pre_upload"])
            self._write_state()
            if self.rebuilt:
                self._epoch_begin()
            return
        self._epoch_begin()
        self.listing = {"pre_upload": dict(st["container"].get("pre_upload") or {"users_exists": None})}

    def _epoch_begin(self) -> None:
        # A RE-RUN of an epoch whose turn was already launched (status launched/failed) keeps /evo/staging as the
        # evolver left it -- its work in progress; nothing re-seeds it. A new epoch clears it (the seed upload follows).
        keep = self.prior_epoch_status in (S_LAUNCHED, S_FAILED) and not self.rebuilt
        cmd = f"epoch-begin --cid {q(self.cid)}" + ("" if keep else f" --clear {q(SL.VM_STAGING)}")
        if keep:
            log(f"epoch-begin without --clear: epoch {self.epoch} re-run (prior attempt {self.prior_epoch_status}); "
                f"{SL.VM_STAGING} kept as the evolver left it")
        code, out, res = self.ctl(cmd, timeout=300, what="epoch-begin")
        if not res.get("ok") and any("still running" in str(p) for p in res.get("problems") or []):
            code, out, stop = self.ctl(f"stop-session --cid {q(self.cid)}", timeout=120, what="stop")
            self.c()["epoch_begin_stop_session"] = stop
            log(f"epoch-begin: a session was still running in {self.cid}; stopped it ({stop}) and retrying")
            code, out, res = self.ctl(cmd, timeout=300, what="epoch-begin")
        self.c()["epoch_begin"] = res
        self.save_receipt()
        if not res.get("ok") or not res.get("init_alive"):
            raise Refuse(EXIT_IDENTITY, f"epoch-begin refused: {res.get('problems') or out[-400:]}")

    # ── upload: the delta (or the cumulative re-upload after a rebuild), then the cumulative listing check
    def _generated_bytes(self, vm_path: str) -> bytes:
        if vm_path in (self.gen or {}):
            return self.gen[vm_path]
        g = (self.state or {}).get("generated") or {}
        if vm_path not in g:
            raise Refuse(EXIT_MANIFEST, f"generated row {vm_path}: its bytes are not in the state record")
        return base64.b64decode(g[vm_path])

    def _rehashed(self, rows: list) -> list:
        """The rows of earlier epochs, their host sources re-hashed NOW: a changed source is a refusal."""
        for r in rows:
            if r["source"] == "generated":
                if de.sha256_bytes(self._generated_bytes(r["vm_path"])) != r["sha256"]:
                    raise Refuse(EXIT_MANIFEST, f"generated row {r['vm_path']} no longer re-renders to its uploaded sha")
                continue
            p = Path(r["source"])
            if p.is_symlink() or not p.is_file() or de.sha256_file(p) != r["sha256"]:
                raise Refuse(EXIT_MANIFEST, f"{r['source']} (uploaded at epoch {r.get('epoch')}) changed on the host or "
                                            f"is gone; the container cannot be rebuilt from changed evidence")
        return rows

    def _upload_rows(self, rows: list, *, incremental: bool) -> dict:
        tgz = self.q / "upload.tgz"
        with tarfile.open(tgz, "w:gz", format=tarfile.PAX_FORMAT) as tf:
            for r in rows:
                arc = r["vm_path"].lstrip("/")
                if r["source"] == "generated":
                    data = self._generated_bytes(r["vm_path"])
                    ti = tarfile.TarInfo(arc)
                    ti.size, ti.mode, ti.mtime = len(data), 0o644, int(time.time())
                    tf.addfile(ti, io.BytesIO(data))
                else:
                    ti = tf.gettarinfo(r["source"], arcname=arc)
                    ti.uid = ti.gid = 0
                    ti.uname = ti.gname = "root"
                    with open(r["source"], "rb") as fh:
                        tf.addfile(ti, fh)
        self.sb.fs.upload_file(str(tgz), f"{self.cdir}/state/upload.tgz", timeout=1800)
        code, out, res = self.ctl(f"extract --cid {q(self.cid)}" + (" --incremental" if incremental else ""),
                                  timeout=900, what="extract")
        self.c().setdefault("extract", []).append({"rows": len(rows), "incremental": incremental, **res})
        if not res.get("ok"):
            raise Refuse(EXIT_MANIFEST, f"extract refused: {res.get('problems') or out[-400:]}")
        return res

    def upload(self) -> None:
        st, c, N = self.state, self.c(), self.epoch
        older = [r for r in st.get("uploaded") or [] if int(r.get("epoch", 0)) < N]
        prev_this = [r for r in st.get("uploaded") or [] if int(r.get("epoch", 0)) == N]
        delta = [dict(r, epoch=N) for r in self.rows]
        if self.rebuilt:
            to_upload, how, incremental = self._rehashed(older) + delta, "cumulative re-upload after a rebuild", True
        elif prev_this:
            if [(r["vm_path"], r["sha256"]) for r in prev_this] != [(r["vm_path"], r["sha256"]) for r in delta]:
                raise Refuse(EXIT_MANIFEST, f"the delta of epoch {N} differs from the rows its earlier attempt uploaded "
                                            f"(status {self.prior_epoch_status}); remedy: evolver_session.py finish "
                                            f"--run-dir {self.run_dir} --abort for a new session")
            to_upload = [r for r in delta if ctl._under(r["vm_path"], SL.VM_STAGING)]
            how, incremental = "re-seed only (the container already holds this epoch's delta)", True
        else:
            to_upload, how, incremental = delta, "delta", not self.fresh
        if to_upload:
            self._upload_rows(to_upload, incremental=incremental)
        after = self.listing_inside(list(de.listing_roots()))
        self.listing["after_upload"] = after
        jdump(self.listing_path, self.listing)
        cumulative = older + delta

        def writable(p):
            return any(ctl._under(p, w) for w in SL.EVOLVER_WRITABLE)
        man = {r["vm_path"]: r["sha256"] for r in cumulative}
        files = {p: h for p, t, h, _b in after if t == "F"}
        links = [p for p, t, _h, _b in after if t == "L"]
        f_out = {p: h for p, h in files.items() if not writable(p)}
        m_out = {p: h for p, h in man.items() if not writable(p)}
        extra, missing = sorted(set(f_out) - set(m_out)), sorted(set(m_out) - set(f_out))
        diff = sorted(p for p in set(f_out) & set(m_out) if f_out[p] != m_out[p])
        wmissing = sorted(p for p, h in man.items() if writable(p) and files.get(p) != h)
        users = self.listing["pre_upload"].get("users_exists")
        log(f"upload ({how}): {len(to_upload)} rows sent; {len(files)} files inside; cumulative manifest {len(man)}; "
            f"extra {len(extra)} missing {len(missing)} differing {len(diff)} symlinks {len(links)} "
            f"writable-tree rows absent {len(wmissing)}")
        if extra or missing or diff or links or wmissing or users is not False:
            raise Refuse(EXIT_MANIFEST, f"container listing != cumulative manifest (extra {extra[:3]}, missing "
                                        f"{missing[:3]}, diff {diff[:3]}, links {links[:3]}, writable-tree rows absent "
                                        f"{wmissing[:3]}, /Users before upload {users!r})")
        st["uploaded"] = cumulative
        c["upload"] = {"how": how, "rows_sent": len(to_upload), "delta_rows": len(delta), "cumulative_rows": len(cumulative)}
        self._mark(S_UPLOADED)
        self.save_receipt()

    # ── launch
    def claude_argv(self) -> list:
        argv = ["claude", *ctl.CC_EXTRA_ARGS, "-p", "__PROMPT__", "--model", self.model]   # extra: none for 2.1.281
        argv += ["--resume", self.sid_session] if self.resume else ["--session-id", self.sid_session]
        argv += ["--append-system-prompt", "__SYSPROMPT__"]
        for d in self.add_dirs:
            argv += ["--add-dir", d]
        argv += ["--disallowed-tools", "WebFetch", "WebSearch"]
        ea = claude_pins.effort_argv(self.model, os.environ)
        if ea:
            raise Refuse(EXIT_USAGE, f"stateful evolver: no --effort flag (the effort is left unset); got {ea}")
        argv += ["--settings", f"{de.VM_HOOKS}/settings.json", "--thinking-display", "summarized",
                 "--dangerously-skip-permissions", "--output-format", "stream-json", "--verbose"]
        if self.max_turns:
            argv += ["--max-turns", str(self.max_turns)]
        return argv

    def launch(self) -> None:
        """The base launch with ONE difference: the message and the appended system prompt reach `claude` as base64
        shell variables decoded inside the container's bash, never as files the evolver could see (no
        /evo/work/MESSAGE_e<N>.md, no SYSTEM.md). The receipt's launch_argv keeps the placeholders; the texts
        themselves are the host records (ITER{N}_PROMPT.md, EVOLVER_SYSTEM.md)."""
        self._mark(S_LAUNCHED, launched_utc=dle.now())       # before the exec: a launch that fails mid-way may have started
        argv = self.claude_argv()
        if "__PROMPT__" not in argv or "__SYSPROMPT__" not in argv:
            raise Refuse(EXIT_USAGE, "argv must carry the __PROMPT__ and __SYSPROMPT__ placeholders")
        inner = " ".join(q(a) for a in argv).replace(q("__PROMPT__"), '"$EVO_MESSAGE"') \
                                            .replace(q("__SYSPROMPT__"), '"$EVO_SYSTEM"')
        msg_b64 = base64.b64encode(Path(self.prompt).read_bytes()).decode()
        sys_b64 = base64.b64encode(Path(self.system_prompt).read_bytes()).decode()
        inner = (f"exec >{de.VM_OUT}/evolver.log 2>{de.VM_OUT}/evolver.stderr </dev/null; "
                 f"set -a; . {de.VM_ENV_FILE}; set +a; cd {de.VM_WORK}; "
                 f'EVO_MESSAGE="$(printf %s {msg_b64} | base64 -d; printf x)"; EVO_MESSAGE="${{EVO_MESSAGE%x}}"; '
                 f'EVO_SYSTEM="$(printf %s {sys_b64} | base64 -d; printf x)"; EVO_SYSTEM="${{EVO_SYSTEM%x}}"; '
                 f"{self.fd_env}=3 {inner} & "
                 f"echo $! > {de.VM_OUT}/pid; wait $!; "
                 f"echo $? > {de.VM_OUT}/rc.tmp && mv {de.VM_OUT}/rc.tmp {de.VM_OUT}/rc")
        cmd = (self.enter_cmd(inner, cwd=de.VM_WORK, detached=True)
               + f" ; sleep 8; {EC.PY} {EC.CTL} tail --cid {self.cid} --offset 0 --max 1500")
        if self.token in cmd:
            raise Refuse(EXIT_TOKEN, "token would appear in a command string")
        self.receipt["daytona"]["launch_argv"] = argv
        self.receipt["daytona"]["launched_utc"] = dle.now()
        self.receipt["daytona"]["prompt_delivery"] = "message + appended system prompt as base64 shell variables (no file)"
        self.save_receipt()
        code, out = self.sh(cmd, timeout=180, env={T.SECRET_ENV: self.token}, what="launch",
                            retries=1)                # never re-run: a second claude, a truncated log
        self.launched = True
        log(f"launched in {self.cid}; first poll: {out[:200]!r}")

    # ── collect: epoch-end (the container stays up), download, audits, snapshot
    def collect(self) -> None:
        c, dd = self.c(), self.receipt["daytona"]
        if getattr(self, "claude_rc", None) is None and self.launched and "epoch_end" not in c:
            code, out, res = self.ctl(f"stop-session --cid {q(self.cid)}", timeout=120, what="stop")
            c["salvage_stop_session"] = res
        roots = self.readonly_roots()
        if "after_run_readonly" not in getattr(self, "listing", {}) and self.container_alive():
            after = self.listing_inside([str(r) for r in roots])
            self.listing["after_run_readonly"] = after
            jdump(self.listing_path, self.listing)
            self.after_hashes(after)
            code, out, res = self.ctl(f"probe --cid {q(self.cid)} --phase post", timeout=300, what="post probe", retries=2)
            c["probe_post"] = res.get("probe") or {}
            c["probe_post_problems"] = pce.post_probe_problems(c.get("probe_pre") or {}, c["probe_post"])
        if "epoch_end" not in c:
            parts = ["/" + p.lstrip("/") for p in self.download_parts()]
            code, out, res = self.ctl(f"epoch-end --cid {q(self.cid)} " + " ".join(f"--part {q(p)}" for p in parts),
                                      timeout=900, what="epoch-end")
            c["epoch_end"] = res
            self.save_receipt()
            if not res.get("ok"):
                raise Refuse(EXIT_MANIFEST, f"epoch-end refused: {res.get('problems') or out[-400:]}")
        ee = c["epoch_end"]
        local = self.q / "out.tgz"
        self.sb.fs.download_file(f"{self.cdir}/state/out.tgz", str(local))
        ex = self.q / "extract"
        if ex.exists():
            shutil.rmtree(ex)
        ex.mkdir()
        refused = []
        with tarfile.open(local, "r:gz") as tf:
            for m in tf.getmembers():
                name = m.name
                if name.startswith("/") or ".." in Path(name).parts or not (m.isfile() or m.isdir()):
                    refused.append(name)
                    continue
                tf.extract(m, ex, filter="data")
        dd["download_refused_members"] = refused + [r["part"] for r in ee.get("refused") or []]
        vm_log = ex / de.VM_OUT.lstrip("/") / "evolver.log"
        streamed = self.q / self.stream_name
        same = vm_log.is_file() and streamed.is_file() and de.sha256_file(vm_log) == de.sha256_file(streamed)
        dd["log_round_trip"] = {"streamed_sha256": de.sha256_file(streamed) if streamed.is_file() else None,
                                "vm_sha256": de.sha256_file(vm_log) if vm_log.is_file() else None, "equal": same}
        proj = ex / de.VM_PROJECTS.lstrip("/")
        dd["session_audit"] = de.audit_sessions([streamed, proj])
        # No held-out id scan at collect: the container held only the manifest (listing == manifest is the
        # containment proof), and nothing scans what the session WROTE either.
        hits, n = de.token_scan([ex, streamed], self.token)
        self.receipt["token_scan"] = {"hits": hits, "scanned_files": n}
        dd["readback_end"] = dle.readback(self.sb)
        scan = ee.get("scan") or {}
        mhits = [h for part in scan.values() for h in (part or {}).get("hits") or []] + \
            list((ee.get("tmpfs_scan") or {}).get("hits") or [])
        self.save_receipt()
        log(f"collected (container kept): log round-trip equal={same}; token scan {len(hits)} hit(s) in {n} files; "
            f"container state {len(mhits)} prefix hit(s); refused members "
            f"{len(dd['download_refused_members'])}")
        if hits or mhits:
            self._mark(S_FAILED, why="token/key prefix found")
            raise Refuse(EXIT_TOKEN, f"TOKEN/KEY PREFIX FOUND ({len(hits)} downloaded, {len(mhits)} in the container "
                                     f"state: {mhits[:3]}) -- nothing installed; rotate the credential")
        if c.get("probe_post_problems"):
            self._mark(S_FAILED, why="post probe")
            raise Refuse(EXIT_IDENTITY, "post probe: " + "; ".join(c["probe_post_problems"][:3]))
        self.collected = True
        if not same:
            self._mark(S_FAILED, why="streamed log != container log")
            raise Refuse(EXIT_MANIFEST, "streamed log != container log")
        snap = self.snapshot(contaminated=False)
        dd["snapshot_dir"] = str(snap)
        if getattr(self, "rate_limited", None):
            # rejected for a usage limit: not a turn taken -- the dispatch's re-run after the reset resumes it
            self._mark(S_FAILED, why="rejected for a usage limit / HTTP 429", snapshot=str(snap),
                       claude_rc=getattr(self, "claude_rc", None), rate_limited=self.rate_limited)
        else:
            self._mark(S_COLLECTED, snapshot=str(snap), claude_rc=getattr(self, "claude_rc", None))
        self.save_receipt()

    def snapshot(self, *, contaminated: bool) -> Path:
        """.cache/evolver_state/<run>/e<N>/{MANIFEST.json, projects/, notes/, receipt.json}, written
        beside as a tmp dir and renamed into place; re-verified unless marked contaminated."""
        d = SL.snapshot_dir(self.run_dir, self.epoch)
        tmp = d.with_name(d.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        for parent in (SL.STATE_ROOT, d.parent):
            parent.mkdir(parents=True, exist_ok=True)
            os.chmod(parent, 0o700)
        tmp.mkdir()
        os.chmod(tmp, 0o700)
        ex = self.q / "extract"
        n_proj = copy_tree_nofollow(ex / de.VM_PROJECTS.lstrip("/"), tmp / "projects")
        n_ws = copy_tree_nofollow(ex / SL.VM_WORKSPACE.lstrip("/"), tmp / SL.SNAPSHOT_WORKSPACE)
        self.save_receipt()
        shutil.copyfile(self.receipt_path, tmp / "receipt.json")
        man = SL.snapshot_manifest(tmp)
        man.update(run=SL.run_name(self.run_dir), cid=self.cid, sandbox_id=self.sb.id, sid=self.sid_session,
                   epoch=self.epoch, utc=dle.now(), claude_rc=getattr(self, "claude_rc", None),
                   contaminated=bool(contaminated), projects_files=n_proj, workspace_files=n_ws)
        (tmp / SL.SNAPSHOT_MANIFEST).write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
        if d.exists():
            shutil.rmtree(d)
        os.rename(tmp, d)
        probs = SL.verify_snapshot(d)
        if probs and not contaminated:
            raise Refuse(EXIT_MANIFEST, f"snapshot {d} does not verify: {probs[:3]}")
        log(f"snapshot {d}: {n_proj} transcript file(s), {n_ws} workspace file(s)" + (" CONTAMINATED" if contaminated else ""))
        return d

    # ── delete: the lease goes, the container stays
    def delete(self) -> bool:
        dd = (getattr(self, "receipt", None) or {}).get("daytona")
        if getattr(self, "sb", None) is None:
            self.release_lease()
            if getattr(self, "receipt", None):
                self.save_receipt()
            return True
        self.release_lease()
        try:
            self._write_state()
        except (OSError, ValueError) as e:
            log(f"state record not written: {type(e).__name__}: {e}")
        c = self.c()
        c["kept"] = True
        dd["container_removed_utc"] = None
        self.save_receipt()
        log(f"container {self.cid} kept on {self.sb.id[:8]} (stateful session {self.sid_session[:8]}); lease released")
        return True

    # ── install: the base's (from the host-shaped staging path), then the notes
    def install(self) -> None:
        ex = self.q / "extract"
        src, dst = ex / SL.VM_STAGING.lstrip("/"), ex / str(self.staging).lstrip("/")
        if src.is_dir() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.rename(src, dst)                       # the base install reads ex/<host staging path>
        super().install()
        ws_src, ws_dst = ex / SL.VM_WORKSPACE.lstrip("/"), self.run_dir / SL.WORKSPACE_DIR.format(epoch=self.epoch)
        if ws_dst.exists():
            shutil.rmtree(ws_dst)
        n = copy_tree_nofollow(ws_src, ws_dst)
        dd = self.receipt["daytona"]
        dd["workspace_dir"], dd["workspace_files"] = str(ws_dst), n
        self.save_receipt()
        if self.state is not None:
            cur = dict(self.state["epochs"].get(str(self.epoch)) or {})
            cur.update(receipt=str(self.receipt_path), installed_utc=dle.now(), workspace_dir=str(ws_dst))
            self.state["epochs"][str(self.epoch)] = cur
            self._write_state()
        log(f"installed the evolver's workspace -> {ws_dst.name} ({n} file(s))")


class StatefulEvolverRun(StatefulMixin, EC.ContainerEvolverRun):
    pass


# ───────────────────────────────────────────────────────────────────────────── run-level commands
def _lock_or_refuse(rd: Path):
    """The run's dispatch lock, EXCLUSIVE (kept open while the command runs), or None (logged) while a dispatch of
    the run is live (tau3_evolver_daytona.hold_dispatch_lock)."""
    lock = T.exclusive_dispatch_lock(rd)
    if lock is None:
        log(f"a dispatch of {rd.name} is live; refusing")
    return lock


def _load_state_or_fail(rd: Path):
    try:
        st = SL.load_state(rd)
    except ValueError as e:
        log(f"state record: {e}")
        return None, EXIT_USAGE
    if st is None:
        log(f"no session state for {rd.name} under {SL.STATE_ROOT}")
        return None, EXIT_USAGE
    return st, 0


def _pool_current_row(rd: Path) -> tuple:
    """(machine.json record or None, the current machine's row or None)."""
    regf = pce.pool_machines_file(rd)
    try:
        rec = json.loads(regf.read_text())
    except (OSError, ValueError):
        return None, None
    if not isinstance(rec, dict) or rec.get("schema") != pce.POOL_MACHINES_SCHEMA:
        return rec, None
    row = (rec.get("machines") or {}).get(rec.get("current") or "") or None
    if row is not None:
        row = dict(row, name=rec.get("current"))
    return rec, row


def cmd_status(a) -> int:
    rd = Path(a.run_dir).resolve()
    st, rc = _load_state_or_fail(rd)
    if rc:
        return rc
    out = {"state": st, "state_path": str(SL.state_path(rd)),
           "snapshots": {d.name: SL.verify_snapshot(d) for d in sorted((SL.STATE_ROOT / SL.run_name(rd)).glob("e*"))
                         if d.is_dir()}}
    rec, row = _pool_current_row(rd)
    out["pool_machine"] = row
    if row and row.get("sandbox_id") == st.get("sandbox_id") and not st.get("torn_down"):
        try:
            sb = dle.client().get(st["sandbox_id"])
            code, o, _ = dle.run(sb, f"{PY} {CTL} status --cid {q(st['cid'])}", timeout=120)
            out["machine_status"] = EC.parse_json_tail(o)
        except Exception as e:                                  # noqa: BLE001
            out["machine_status"] = {"error": f"{type(e).__name__}: {e}"}
    else:
        out["machine_status"] = None if row else "no current pool machine"
    print(json.dumps(out, indent=1, default=str))
    return 0


def teardown_record_problems(record: dict, ledger_path=None) -> list:
    """The self-check of a teardown record (the loop's run_session_finish re-verifies the same way)."""
    probs = []
    if not isinstance(record, dict) or record.get("schema") != TEARDOWN_SCHEMA:
        return ["not a teardown record"]
    if record.get("machine_absent"):
        return probs
    dn = record.get("down") or {}
    if dn.get("verified") is not True or dn.get("wiped") is not True:
        probs.append(f"down not verified (verified {dn.get('verified')!r}, wiped {dn.get('wiped')!r})")
    if not pce.ledger_container_down(ledger_path or dle.LEDGER, sandbox_id=record.get("sandbox_id") or "",
                                     cid=record.get("cid") or ""):
        probs.append("no ledgered, verified 'container_down' row for this container")
    return probs


def cmd_finish(a) -> int:
    """`down --cid C --wipe` on the session's machine (a short lease of the pool machine), ledgered and verified;
    <run>/EVOLVER_SESSION_TEARDOWN.json; state.torn_down. --abort = an operator abandoning the run.
    --machine-absent-ok: the pool's machine.json no longer names the session's sandbox (it was deleted with the
    pool): a record without a down; the container cannot outlive its machine."""
    rd = Path(a.run_dir).resolve()
    lock = _lock_or_refuse(rd)
    if lock is None:
        return T.EXIT_LIVE
    st, rc = _load_state_or_fail(rd)
    if rc:
        return rc
    record_path = rd / SL.TEARDOWN_RECORD
    if st.get("torn_down"):
        log(f"session {st['sid'][:8]} already torn down at {st['torn_down'].get('utc')}: {st['torn_down'].get('record')}")
        print(json.dumps({"ok": True, "already": True, "record": st["torn_down"].get("record")}))
        return 0
    cid, sid_box = st["cid"], st["sandbox_id"]
    rec, row = _pool_current_row(rd)
    present = bool(row) and row.get("sandbox_id") == sid_box and row.get("state") not in ("deleted", "released", "failed")
    epochs = [{"epoch": int(k), **(v or {})} for k, v in sorted((st.get("epochs") or {}).items(), key=lambda kv: int(kv[0]))]
    record = {"schema": TEARDOWN_SCHEMA, "run": st["run"], "cid": cid, "sandbox_id": sid_box, "sid": st["sid"],
              "epochs": epochs, "down": None, "ledgered": False, "utc": dle.now(), "abort": bool(a.abort)}
    if not present:
        if not a.machine_absent_ok:
            log(f"the session's machine {str(sid_box)[:8]} is not the pool's current ready machine ({row}); pass "
                f"--machine-absent-ok when machine.json shows it deleted/absent (the container died with it)")
            return EXIT_IDENTITY
        record["machine_absent"] = {"machine_json": row, "utc": dle.now()}
        log(f"machine {str(sid_box)[:8]} absent from the pool: recording the teardown without a down")
    else:
        regf = pce.pool_machines_file(rd)
        holder = pce.lease_holder(cid)
        lease = pce.PoolLease(regf.parent / pce.POOL_LEASES_DIR, holder=holder, sandbox_id=sid_box,
                              cap_ts=time.time() + FINISH_LEASE_S,
                              info={"cid": cid, "run": st["run"], "epoch": "finish", "role": "evolver-finish"})
        try:
            lease.write()
        except OSError as e:
            log(f"cannot write the lease: {e}")
            return EXIT_IDENTITY
        ack, awhy = EC.ContainerMixin.await_lease_ack(None, regf, row, holder)
        if not ack:
            lease.release()
            log(f"lease on {row.get('name')} not acknowledged: {awhy}")
            return EXIT_IDENTITY
        lease.start()
        down = {}
        try:
            sb = dle.client().get(sid_box)
            for attempt in range(DOWN_RETRIES):
                code, out, _ = dle.run(sb, f"{PY} {CTL} down --cid {q(cid)} --wipe", timeout=300)
                down = EC.parse_json_tail(out)
                if down.get("verified") and down.get("wiped"):
                    break
                time.sleep(5)
            ok = bool(down.get("verified") and down.get("wiped"))
            dle.ledger(event="container_down", driver=DRIVER, tag=st["run"], id=sid_box, cid=cid, verified=ok,
                       session_finish=True, abort=bool(a.abort))
        finally:
            lease.release()
        record["down"] = down
        record["ledgered"] = pce.ledger_container_down(dle.LEDGER, sandbox_id=sid_box, cid=cid)
    probs = teardown_record_problems(record)
    record["problems"] = probs
    jdump(record_path, record)
    if probs:
        log(f"teardown NOT verified: {probs}; record {record_path}; state left live")
        print(json.dumps({"ok": False, "problems": probs, "record": str(record_path)}))
        return EXIT_DELETE
    st["torn_down"] = {"utc": dle.now(), "record": str(record_path), "abort": bool(a.abort)}
    SL.write_state(rd, st)
    log(f"session {st['sid'][:8]} torn down (container {cid}; {'machine absent' if 'machine_absent' in record else 'down verified + ledgered'})")
    print(json.dumps({"ok": True, "record": str(record_path), "cid": cid, "abort": bool(a.abort)}))
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(prog="evolver_session.py", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("status")
    s.add_argument("--run-dir", required=True)
    s = sub.add_parser("finish")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--abort", action="store_true")
    s.add_argument("--machine-absent-ok", action="store_true")
    a = ap.parse_args(argv)
    fn = {"status": cmd_status, "finish": cmd_finish}[a.cmd]
    return fn(a)


if __name__ == "__main__":
    sys.exit(main())
