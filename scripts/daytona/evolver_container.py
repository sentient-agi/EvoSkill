#!/usr/bin/env python3
"""Dispatch ONE epoch's evolver in an isolated CONTAINER on the run's POOL machine (RUNTIME evolver_host "daytona"
+ evolver_daytona.placement "pool_container"). Same positional argv, receipts and exit codes as
scripts/daytona/tau3_evolver_daytona.py; only WHERE the session runs differs. The evolver gets its own container on
the leg machine (not a machine of its own) so the run uses one machine while its rollouts keep their concurrency.

    evolver_container.py <epoch> <prompt> <digests_dir> <staging_dir>    # the loop's dispatch
    evolver_container.py ao --run-dir ... (tau3_evolver_daytona.py ao's flags)
    evolver_container.py smoke-container --run-dir R [--sandbox-id ID] # container + probes + upload, NO model call
    evolver_container.py probe --run-dir R [--sandbox-id ID]           # + a 2-turn Claude session
    evolver_container.py cleanup --run-dir R [--sandbox-id ID]         # stop + wipe leftover containers

The machine side is scripts/daytona/evolver_container_ctl.py (uploaded as /opt/evoskill-evolver/bin/evoctl.py): what
the container sees, its namespaces, cgroup and egress proxy -- read its docstring. The loop's checks:
src/orchestrator/pool_container_evolver.py (containment) + daytona_evolver.check_upload_manifest (unchanged).

DISPATCH, in order (exit codes as tau3_evolver_daytona.py; every non-zero HALTs the loop):
  1. a per-epoch session only: the launcher's own gates (tau3_evolver.sh, EVOSKILL_TAU3_GATE_ONLY=1); a stateful
     session runs none (its material is the evo layout and its texts are fixed templates);
  2. EvolverRun.prepare() UNCHANGED (the credential, prompt size, grants, upload manifest, receipt + trace written
     before anything touches a machine), plus the container spec (writable = the staging dir only) [2, 23];
  3. the pool machine: the CURRENT machine of the run pool's <store_root>/<run>/pool/run/machine.json (state ready,
     api.anthropic.com in its allow list, retire_at_ts - now >= timeout + 30 min; re-read for up to RUNTIME
     evolver_daytona.machine_wait_s), a LEASE in the pool's leases/<holder>.json that the pool host must list in
     leased_by before anything runs (renewed while held, removed after the teardown), a fresh readback (exactly
     openrouter.ai + api.anthropic.com, the leg snapshot); no other evolver container on it [25];
  4. evoctl + the egress proxy uploaded and sha-checked; the Claude Code binary uploaded once per machine
     (sha256 = INFRA claude_code.linux_x64_sha256) to a path only the container sees; `prepare` [25];
  5. `up` (namespaces, pivot_root, cgroup, proxy in the container's netns) and the PRE probe from inside, as evo:
     identity, capabilities, the mount table = the plan, no machine path or process visible, writes only where
     granted, egress api.anthropic.com only [25 identity / 22 egress / 26 LibreOffice pin];
  6. upload one tgz, extract at the manifest paths; the listing INSIDE the container equals the manifest [23];
  7. launch detached (the credential on fd 3 only); stream the log back byte-exactly;
  8. collect: after-run hashes inside, the POST probe, `down` (freeze, tmpfs scan, kill, verify), pack the staging +
     transcripts + out dir with no symlink followed, download, log round-trip, token scan [24];
  9. teardown: `down --wipe` verified + ledgered, lease released; a failure leaves the machine and its rollouts
     alone [29 if the teardown is unverified]; install as today; claude rc != 0 -> [27], timeout -> [28] -- except
     the DEADLINE stop: the staging is installed as it stands and the exit is 0.
A turn the API rejected for a usage limit / an HTTP 429 waits for the reported reset and runs again
(tau3_evolver_daytona.run_resuming).
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
import signal
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
import evolver_container_ctl as ctl  # noqa: E402
import tau3_evolver_daytona as T  # noqa: E402
from src.orchestrator import daytona_evolver as de  # noqa: E402
from src.orchestrator import pool_container_evolver as pce  # noqa: E402

Refuse = T.Refuse
log = T.log
jdump = T.jdump
EXIT_USAGE, EXIT_EGRESS, EXIT_MANIFEST = T.EXIT_USAGE, T.EXIT_EGRESS, T.EXIT_MANIFEST
EXIT_TOKEN, EXIT_IDENTITY, EXIT_PIN = T.EXIT_TOKEN, T.EXIT_IDENTITY, T.EXIT_PIN
EXIT_CLAUDE_RC, EXIT_TIMEOUT, EXIT_DELETE = T.EXIT_CLAUDE_RC, T.EXIT_TIMEOUT, T.EXIT_DELETE
DRIVER = pce.DRIVER
CTL_SRC = HERE / "evolver_container_ctl.py"
PROXY_SRC = REPO / "scripts/evolver_egress_proxy.py"
CC_CACHE = REPO / ".cache" / "claude_code" / ctl.CC_VERSION / "linux-x64" / "claude"
CC_URL = f"https://downloads.claude.ai/claude-code-releases/{ctl.CC_VERSION}/linux-x64/claude"
POLL_S = T.POLL_S
CHUNK = T.CHUNK
PY, CTL = ctl.PY, ctl.CTL
q = shlex.quote


def sha256_file(p) -> str:
    return de.sha256_file(p)


def infra_cc_pin(run_dir: Path) -> str | None:
    try:
        return (json.loads((run_dir / "INFRA.json").read_text()).get("claude_code") or {}).get("linux_x64_sha256")
    except (OSError, ValueError):
        return None


def ensure_cc_binary(cfg: dict, run_dir: Path | None = None) -> Path:
    """The Claude Code linux-x64 binary on THIS host, verified: sha256 = daytona_evolver.CC_LINUX_X64_SHA (= the run's
    INFRA claude_code.linux_x64_sha256 when it names one) and the release's size. cfg cc_linux_x64_binary names a
    local copy; else the cache, fetched once from the pinned release URL (no install.sh, no npm)."""
    want = ctl.CC_SHA256          # the pool pin (evolver_container_ctl; EVOSKILL_CC_VERSION, default 2.1.281)
    pin = infra_cc_pin(run_dir) if run_dir else None
    if pin and pin != want:
        raise Refuse(EXIT_PIN, f"INFRA claude_code.linux_x64_sha256 {pin[:12]} != the driver's pin {want[:12]}")
    p = Path(cfg.get("cc_linux_x64_binary") or CC_CACHE)
    if not p.is_file():
        if cfg.get("cc_linux_x64_binary"):
            raise Refuse(EXIT_USAGE, f"cc_linux_x64_binary {p} missing")
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + f".{os.getpid()}.part")
        import urllib.request
        log(f"fetching Claude Code {ctl.CC_VERSION} linux-x64 into {p} (once)")
        urllib.request.urlretrieve(CC_URL, tmp)
        if sha256_file(tmp) != want or tmp.stat().st_size != ctl.CC_BYTES:
            tmp.unlink()
            raise Refuse(EXIT_PIN, "downloaded Claude Code binary does not match the pinned sha256/size")
        os.chmod(tmp, 0o755)
        os.replace(tmp, p)
    if sha256_file(p) != want or p.stat().st_size != ctl.CC_BYTES:
        raise Refuse(EXIT_PIN, f"{p}: not the pinned Claude Code {ctl.CC_VERSION} (sha256/size)")
    return p


def parse_json_tail(out: str) -> dict:
    for ln in reversed(out.strip().splitlines()):
        ln = ln.strip()
        if ln.startswith("{"):
            try:
                return json.loads(ln)
            except ValueError:
                continue
    return {}



def evolver_container_conflicts(containers) -> list:
    """The containers on the machine that CONFLICT with a new evolver container: every one whose cid does not start
    with `cc-` (the Claude Code solver legs share the machine; their `cc-<unit>-…` containers -- a val leg's 30 may be
    live when the epoch's evolver starts -- are not a conflict; another evolver container `e<N>-…` / stateful `run-…`
    is). Elements are cid strings or dicts with a `cid`."""
    out = []
    for c in containers or []:
        cid = str(c.get("cid") if isinstance(c, dict) else c)
        if not cid.startswith("cc-"):
            out.append(c)
    return out

class ContainerMixin:
    """The machine-facing half of tau3_evolver_daytona.EvolverRun (and AORun), re-pointed at a container on the
    pool machine. prepare / claude_argv / install / readonly_roots / download_parts / after_hashes / run are the
    base's; every step that touched a VM is replaced here."""

    placement = pce.PLACEMENT

    def cinit(self, *, sandbox_id: str | None = None, registry: Path | None = None,
              episode_check: bool = False) -> None:
        self.sandbox_override, self.registry_override = sandbox_id, registry
        self.episode_check = episode_check
        self.cid = f"e{self.epoch}-{time.strftime('%Y%m%dt%H%M%S').lower()}-{uuid.uuid4().hex[:6]}"
        self.cdir = ctl.cdir(self.cid)
        self.lease = None                             # pce.PoolLease: the pool's leases/<holder>.json

    # ── helpers
    def ctl(self, args: str, *, timeout: int = 600, what: str = "", retries: int = 1, env=None) -> tuple:
        code, out = self.sh(f"{PY} {CTL} {args}", timeout=timeout, env=env, what=what or args.split()[0],
                            retries=retries)
        return code, out, parse_json_tail(out)

    def c(self) -> dict:
        return self.receipt["daytona"].setdefault("container", {})

    def container_spec(self) -> dict:
        ro_roots = sorted({str(Path(r).absolute()) for r in self.readonly_roots()})
        if isinstance(self, T.AORun):
            rw = [{"path": str(Path(self.writable_dir).absolute()), "owner": "root", "mode": "1777"}]
        else:
            rw = [{"path": str(Path(self.staging).absolute()), "owner": "evo", "mode": "0755"}]
        ro_rebind = sorted(r for r in ro_roots if any(ctl._under(r, w["path"]) and r != w["path"] for w in rw))
        try:
            limits = pce.container_limits_cfg(self.cfg)
        except ValueError as e:
            raise Refuse(EXIT_USAGE, str(e))
        return {"schema": ctl.SCHEMA, "cid": self.cid, "rw": rw, "ro_rebind": ro_rebind, "readonly_roots": ro_roots,
                "limits": limits, "cc": {"version": ctl.CC_VERSION, "sha256": ctl.CC_SHA256},
                "egress": {"allow": ctl.ALLOW, "must_reach": list(de.MUST_REACH), "must_refuse": list(de.MUST_REFUSE),
                           "max_seconds": int(self.timeout_s) + 3 * 3600, "idle_timeout_s": 1800}}

    # ── 2-3 on the host: the base prepare + the container spec
    def prepare(self) -> None:
        super().prepare()
        try:
            pl = pce.placement(self.cfg) if self.cfg.get("placement") else pce.PLACEMENT
        except ValueError as e:
            raise Refuse(EXIT_USAGE, str(e))
        if pl != pce.PLACEMENT:
            raise Refuse(EXIT_USAGE, f"evolver_container.py runs placement {pce.PLACEMENT!r}, cfg says {pl!r}")
        try:
            pce.machine_wait_s(self.cfg)
        except ValueError as e:
            raise Refuse(EXIT_USAGE, str(e))
        bad = [r["vm_path"] for r in self.rows if not any(ctl._under(r["vm_path"], t) for t in ctl.LISTING_ROOTS)]
        if bad:
            raise Refuse(EXIT_MANIFEST, f"manifest rows outside {ctl.LISTING_ROOTS}: {bad[:3]}")
        spec = self.container_spec()
        vs = ctl.validate_spec(spec)
        if vs:
            raise Refuse(EXIT_MANIFEST, f"container spec: {vs[:3]}")
        self.spec = spec
        dd = self.receipt["daytona"]
        dd.update(driver=str(Path(__file__).relative_to(REPO)), placement=pce.PLACEMENT, prefix=None,
                  resources_requested=None, domain_allow_list=pce.MACHINE_ALLOW_STR,
                  container_egress_allow=ctl.ALLOW, deleted=None, deleted_utc=None)
        dd["container"] = {"cid": self.cid, "cdir": self.cdir, "spec": spec,
                           "ctl_sha256": sha256_file(CTL_SRC), "egress_proxy_sha256": sha256_file(PROXY_SRC),
                           "leg_dockerfile_sha256": de.sha256_file(de.BASE_DOCKERFILE)}
        self.save_receipt()
        try:
            tr = json.loads(self.trace_path.read_text())
            tr.update(host="daytona", placement=pce.PLACEMENT, container=self.cid)
            jdump(self.trace_path, tr)
        except (OSError, ValueError):
            pass
        log(f"container {self.cid}: writable {[r['path'] for r in spec['rw']]}, re-bound read-only "
            f"{len(spec['ro_rebind'])}, limits {spec['limits'] or 'defaults'}")

    # ── 3-4: the pool machine, the controller, Claude Code, prepare
    def registry_file(self) -> Path:
        """--registry (a probe naming a pool machine.json), else the run's pool: <store_root>/<run>/pool/run/machine.json."""
        return Path(self.registry_override) if self.registry_override else pce.pool_machines_file(self.run_dir)

    def attach_machine(self, regf: Path) -> dict:
        """The machine row to host the container, LEASED before anything touches it: pce.pick_pool_machine -- the
        CURRENT machine of the pool's machine.json, "ready", api.anthropic.com in its allow list, retire_at_ts - now >=
        timeout + 30 min -- re-read every MACHINE_POLL_S for up to RUNTIME evolver_daytona.machine_wait_s (a rollover
        replacement is `starting` for minutes), then REFUSED [25]. Once picked: leases/<holder>.json in the pool's
        format, and the host must list it in that machine's leased_by while the machine is still ready (a release
        decided in the same tick shows as "released": the lease goes and the pick starts over). The lease is renewed
        from a thread and removed by delete()."""
        dd = self.receipt["daytona"]
        need = float(self.timeout_s) + pce.EVOLVER_MARGIN_S
        try:
            wait_s = pce.machine_wait_s(self.cfg)
        except ValueError as e:
            raise Refuse(EXIT_USAGE, str(e))
        holder = pce.lease_holder(self.cid)
        t0 = time.time()
        deadline, attempts, last_why = t0 + wait_s, [], None
        while True:
            try:
                rec = json.loads(regf.read_text())
                row, why = pce.pick_pool_machine(rec, now=time.time(), need_s=need, sandbox_id=self.sandbox_override)
            except (OSError, ValueError) as e:
                rec, row, why = None, None, f"no readable pool machine.json at {regf} ({type(e).__name__})"
            if row is not None:
                picked_ts = time.time()
                self.lease = pce.PoolLease(
                    regf.parent / pce.POOL_LEASES_DIR, holder=holder, sandbox_id=row["sandbox_id"],
                    cap_ts=picked_ts + float(self.timeout_s) + 3600 + 1800,
                    info={"cid": self.cid, "run": self.run_dir.name, "epoch": self.epoch,
                          "role": "ao" if isinstance(self, T.AORun) else "evolver"})
                try:
                    self.lease.write()
                except OSError as e:
                    self.lease = None
                    raise Refuse(EXIT_IDENTITY, f"cannot write the lease in {regf.parent / pce.POOL_LEASES_DIR}: {e}")
                ack, awhy = self.await_lease_ack(regf, row, holder)
                if ack:
                    self.lease.start()
                    dd["pool_machine"] = {"registry": str(regf), "format": "pool", "name": row["name"], "row": row,
                                          "why": why, "generation": rec.get("generation"),
                                          "host_pid": rec.get("host_pid"), "picked_ts": round(picked_ts, 3),
                                          "need_s": need, "waited_s": round(picked_ts - t0, 1),
                                          "attempts": attempts[-20:]}
                    dd["lease"] = {**self.lease.summary(), "acknowledged": True, "ack": awhy,
                                   "ack_after_s": round(time.time() - picked_ts, 1)}
                    self.save_receipt()
                    return row
                self.lease.release()
                self.lease = None
                why = f"lease on {row['name']} not acknowledged: {awhy}"
            attempts.append({"after_s": round(time.time() - t0, 1), "why": why[:300]})
            # a CLOSED (or closing / dead) pool host is never re-opened while the loop waits on this evolver -- only a
            # leg join starts a pool host -- so waiting machine_wait_s cannot help: refuse at once (final-integration
            # check C3; it happens only after a halt longer than pool_keep_idle_s between a train leg and its evolver)
            if isinstance(rec, dict) and (rec.get("host_state") in ("closing", "closed")
                                          or not pce.pid_alive(rec.get("host_pid"))):
                dd["pool_machine"] = {"registry": str(regf), "format": "pool", "row": None, "why": why,
                                      "need_s": need, "waited_s": round(time.time() - t0, 1),
                                      "attempts": attempts[-20:], "host_closed": True}
                self.save_receipt()
                raise Refuse(EXIT_IDENTITY, f"pool machine: {why}. The run's pool host is closed (its keep window "
                                            f"pool_keep_idle_s ended with no work) and nothing restarts it while the "
                                            f"loop waits on this evolver (only a leg join starts one). Remedy: start a "
                                            f"pool host with any leg join of this run (its machine is then kept "
                                            f"pool_keep_idle_s), then relaunch the drive")
            if time.time() >= deadline:
                dd["pool_machine"] = {"registry": str(regf), "format": "pool", "row": None, "why": why,
                                      "need_s": need, "waited_s": round(time.time() - t0, 1),
                                      "attempts": attempts[-20:]}
                self.save_receipt()
                raise Refuse(EXIT_IDENTITY, f"pool machine: {why} (after {int(time.time() - t0)} s; "
                                            f"evolver_daytona.machine_wait_s {int(wait_s)})")
            if why != last_why:
                log(f"pool machine not usable yet: {why}; re-reading for up to {int(deadline - time.time())} s")
                last_why = why
            time.sleep(max(0.0, min(pce.MACHINE_POLL_S, deadline - time.time())))

    def await_lease_ack(self, regf: Path, row: dict, holder: str) -> tuple:
        """(True|False, why): the pool host lists `holder` in leased_by of the picked machine (still ready)."""
        t_end = time.time() + pce.LEASE_ACK_S
        while True:
            try:
                rec = json.loads(regf.read_text())
            except (OSError, ValueError) as e:     # a rewrite in progress: read again
                v, why = None, f"machine.json unreadable ({type(e).__name__})"
            else:
                v, why = pce.lease_acknowledged(rec, name=row["name"], sandbox_id=row["sandbox_id"], holder=holder)
            if v is not None:
                return v, why
            if time.time() >= t_end:
                return False, f"{why} after {int(pce.LEASE_ACK_S)} s (is the pool host alive and reading leases/?)"
            time.sleep(max(0.0, min(2.0, pce.LEASE_ACK_S / 20)))

    def release_lease(self) -> None:
        """Remove this dispatch's lease (only while it is still ours) -- after the container is torn down."""
        dd = (getattr(self, "receipt", None) or {}).get("daytona")
        dd = {} if dd is None else dd
        if getattr(self, "lease", None) is not None:
            ok = self.lease.release()
            dd["lease"] = {**(dd.get("lease") or {}), **self.lease.summary(), "released": ok,
                           "released_utc": dle.now()}
            self.lease = None
            log(f"lease {Path(dd['lease']['path']).name} released={ok}")

    def create(self) -> None:
        cc_local = ensure_cc_binary(self.cfg, self.run_dir)
        self.client = dle.client()
        regf = self.registry_file()
        row = self.attach_machine(regf)
        dd = self.receipt["daytona"]
        self.sb = self.client.get(row["sandbox_id"])
        rb = dle.readback(self.sb)
        dd.update(sandbox_id=self.sb.id, sandbox_name=self.sb.name, readback_create=rb)
        want_snap = self.leg_snapshot_ref()
        mp = pce.machine_problems(rb, name=self.sb.name, state=rb.get("state"), row=row,
                                  require_snapshot_ref=want_snap, prefixes=(pce.POOL_PREFIX,))
        if row.get("sandbox_name") and row.get("sandbox_name") != self.sb.name:
            mp.append(f"sandbox name {self.sb.name!r} != the registry's {row.get('sandbox_name')!r}")
        self.save_receipt()
        if mp:
            raise Refuse(EXIT_EGRESS if any("allow" in p or "network" in p for p in mp) else EXIT_IDENTITY,
                         "pool machine: " + "; ".join(mp))
        dle.ledger(event="container_attach", driver=DRIVER, tag=self.run_dir.name, id=self.sb.id, cid=self.cid,
                   epoch=self.epoch)
        log(f"pool machine {self.sb.id[:8]} {self.sb.name}: allow {rb.get('domain_allow_list')!r}, snapshot "
            f"{rb.get('snapshot_ref')}; lease {(dd.get('lease') or {}).get('path')}")
        # no other evolver container may be on the machine
        self.install_ctl()
        code, out, lst = self.ctl("list", what="list", retries=2)
        # The Claude Code solver legs share this machine; their `cc-*` containers (a val leg's 30 may be live when
        # the epoch's evolver starts) are not a conflict -- only another EVOLVER container is. cids: evolver
        # `e<N>-…` / stateful `run-…`, solver `cc-<unit>-…`.
        others = evolver_container_conflicts(lst.get("containers"))
        if others:
            raise Refuse(EXIT_IDENTITY, f"evolver container(s) already on the machine: {others} "
                                        f"(scripts/daytona/evolver_container.py cleanup)")
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
            c["init_sha256"] = hashlib.sha256(ctl.render_init(self.spec, res["facts"], res["limits"], pl)
                                              .encode()).hexdigest()
            c["expected_top"], c["expected_mounts"] = ctl.expected_top(pl), ctl.expected_mount_points(pl)
            self.plan_ = pl
        self.save_receipt()
        if not res.get("ok"):
            raise Refuse(EXIT_IDENTITY, f"prepare refused: {res.get('problems') or out[-400:]}")

    def leg_snapshot_ref(self) -> str | None:
        if self.cfg.get("require_leg_snapshot_ref"):
            return self.cfg["require_leg_snapshot_ref"]
        try:
            return (json.loads((self.run_dir / "RUNTIME.json").read_text()).get("daytona_leg") or {}) \
                .get("require_snapshot_ref")
        except (OSError, ValueError):
            return None

    def install_ctl(self) -> None:
        self.sh(f"mkdir -p {ctl.BIN} && chmod 0700 {ctl.BASE}", timeout=60, what="mkdir bin")
        for src, dst in ((CTL_SRC, ctl.CTL), (PROXY_SRC, ctl.PROXY_FILE)):
            self.sb.fs.upload_file(str(src), dst, timeout=300)
        code, out = self.sh(f"sha256sum {ctl.CTL} {ctl.PROXY_FILE}", timeout=60, what="ctl sha")
        got = {ln.split()[1]: ln.split()[0] for ln in out.splitlines() if len(ln.split()) == 2}
        want = {ctl.CTL: sha256_file(CTL_SRC), ctl.PROXY_FILE: sha256_file(PROXY_SRC)}
        if got != want:
            raise Refuse(EXIT_IDENTITY, f"controller upload does not verify ({got} != {want})")

    def install_cc(self, local: Path) -> None:
        code, out = self.sh(f"sha256sum {ctl.CC_HOST} 2>/dev/null | cut -d' ' -f1; stat -c %s {ctl.CC_HOST} "
                            f"2>/dev/null", timeout=120, what="cc sha")
        lines = out.split()
        c = self.c()
        if lines[:1] == [ctl.CC_SHA256] and lines[1:2] == [str(ctl.CC_BYTES)]:
            c["cc_upload"] = "already on the machine (sha verified)"
            return
        part = ctl.CC_HOST + ".part"
        self.sh(f"mkdir -p {os.path.dirname(ctl.CC_HOST)} && rm -f {part}", timeout=60, what="cc dir")
        t = time.time()
        self.sb.fs.upload_file(str(local), part, timeout=3600)
        code, out = self.sh(f"test \"$(sha256sum {part} | cut -d' ' -f1)\" = {ctl.CC_SHA256} && "
                            f"test \"$(stat -c %s {part})\" = {ctl.CC_BYTES} && chmod 0755 {part} && "
                            f"mv {part} {ctl.CC_HOST} && echo CC_OK", timeout=300, what="cc verify")
        if "CC_OK" not in out:
            raise Refuse(EXIT_PIN, "uploaded Claude Code binary does not verify on the machine")
        c["cc_upload"] = f"uploaded {ctl.CC_BYTES} B in {time.time() - t:.0f}s, sha verified"

    # ── 5: up + the PRE probe
    def probes(self) -> None:
        code, out, res = self.ctl(f"up --cid {self.cid}", timeout=300, what="up")
        c = self.c()
        c["up"] = res
        self.save_receipt()
        if not res.get("ok"):
            raise Refuse(EXIT_IDENTITY, f"container did not come up: {res.get('problems') or out[-600:]}")
        dle.ledger(event="container_up", driver=DRIVER, tag=self.run_dir.name, id=self.sb.id, cid=self.cid,
                   init_ns=(res.get("state") or {}).get("ns"))
        code, out, res = self.ctl(f"probe --cid {self.cid} --phase pre"
                                  + (" --episode-check" if self.episode_check else ""), timeout=1800, what="probe")
        pre = res.get("probe") or {}
        c["probe_pre"] = pre
        I, N = pre.get("identity") or {}, pre.get("net") or {}
        dd = self.receipt["daytona"]
        dd["identity"] = {"whoami": I.get("user"), "arch": I.get("arch"), "libreoffice": I.get("libreoffice"),
                          "cc_file": f"{ctl.CC_VERSION} {I.get('cc_sha256')}",
                          "cc_sha256": I.get("cc_sha256"), "cc_version": I.get("cc_version"),
                          "cc_binaries": str(len(I.get("files_named_claude") or [])),
                          "solver_stack": I.get("solver_stack"),
                          "opt_entries": ",".join(I.get("opt_entries") or []),
                          "users_exists": "no" if I.get("users_files") == 0 else "yes",
                          "secret_env_names": ",".join(I.get("secret_env_names") or []),
                          "proxy_env_names": ",".join(sorted(I.get("proxy_env") or {})),
                          "evo_uid": str(I.get("uid")), "evo_lo_profile": I.get("evo_lo_profile_owner"),
                          "policy_dir": str(I.get("policy_dir_entries"))}
        dd["egress_probe"], dd["egress_direct"], dd["egress_raw"] = N.get("http"), N.get("direct"), N.get("raw")
        dd["egress_dns"] = {"container_dns": N.get("dns"), "accepted": False,
                            "note": "the container has no resolver and no route; every lookup must fail"}
        probs = pce.container_probe_problems(
            pre, expected_top=c.get("expected_top") or [], expected_mounts=c.get("expected_mounts") or [],
            rw_paths=[r["path"] for r in self.spec["rw"]], ro_rebind=self.spec["ro_rebind"],
            require_libreoffice=self.cfg.get("require_libreoffice"))
        c["probe_pre_problems"] = probs
        self.save_receipt()
        log(f"probe: {len(probs)} problem(s); {(pre.get('view') or {}).get('canaries_checked')} machine paths, "
            f"0 visible = {(pre.get('view') or {}).get('canaries_visible_n') == 0}; top {(pre.get('view') or {}).get('top')}")
        if probs:
            net = [p for p in probs if p.startswith("net:")]
            code_ = EXIT_PIN if all("LibreOffice" in p for p in probs) else (EXIT_EGRESS if net and len(net) == len(probs)
                                                                           else EXIT_IDENTITY)
            raise Refuse(code_, "container probe: " + "; ".join(probs[:5]))
        self.listing = {"pre_upload": {"users_exists": I.get("users_files") != 0,
                                       "users_files": I.get("users_files"),
                                       "semantics": "files under /Users INSIDE the container before the upload (it "
                                                    "holds only the empty mount-point dirs of the writable dirs)"}}

    # ── 6: upload + the listing inside
    def upload(self) -> None:
        tgz = self.q / "upload.tgz"
        with tarfile.open(tgz, "w:gz", format=tarfile.PAX_FORMAT) as tf:
            for r in self.rows:
                arc = r["vm_path"].lstrip("/")
                if r["source"] == "generated":
                    data = self.gen[r["vm_path"]]
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
        code, out, res = self.ctl(f"extract --cid {self.cid}", timeout=900, what="extract")
        if not res.get("ok"):
            raise Refuse(EXIT_MANIFEST, f"extract refused: {res.get('problems') or out[-400:]}")
        self.listing["after_upload"] = self.listing_inside(list(de.listing_roots()))
        jdump(self.listing_path, self.listing)
        files = {p: h for p, t, h, _b in self.listing["after_upload"] if t == "F"}
        links = [p for p, t, _h, _b in self.listing["after_upload"] if t == "L"]
        man = {r["vm_path"]: r["sha256"] for r in self.rows}
        extra, missing = sorted(set(files) - set(man)), sorted(set(man) - set(files))
        diff = sorted(p for p in set(files) & set(man) if files[p] != man[p])
        log(f"upload: {len(files)} files seen inside the container; manifest {len(man)}; extra {len(extra)} "
            f"missing {len(missing)} differing {len(diff)} symlinks {len(links)}")
        if extra or missing or diff or links or self.listing["pre_upload"]["users_exists"]:
            raise Refuse(EXIT_MANIFEST, f"container listing != manifest (extra {extra[:3]}, missing {missing[:3]}, "
                                        f"diff {diff[:3]}, links {links[:3]})")

    def listing_inside(self, roots: list) -> list:
        cmd = (f"{PY} {CTL} enter --cid {self.cid} --cwd /tmp -- {PY} -c {q(de.VM_LISTING_PY)} "
               + " ".join(q(r) for r in roots))
        code, out = self.sh(cmd, timeout=900, what="listing", retries=2)
        try:
            return json.loads(out.strip().splitlines()[-1])
        except (ValueError, IndexError):
            raise Refuse(EXIT_MANIFEST, f"listing inside the container failed: {out[-400:]}")

    # ── 7: the credential-bearing command
    def enter_cmd(self, inner: str, *, cwd: str, detached: bool) -> str:
        """The ONE shape a credential-bearing command takes: the credential leaves the SDK exec's env (`env -u`)
        before any process of ours starts, and reaches the container only as fd 3 (a here-string pipe)."""
        base = (f"env -u {T.SECRET_ENV} {PY} {CTL} enter --cid {self.cid} --cwd {cwd} --keep-fd3"
                + (f" --log {self.cdir}/state/launch.err" if detached else "")
                + f" -- /bin/bash -c {q(inner)}")
        redir = f' 3<<<"${T.SECRET_ENV}"'
        if detached:
            return f"(setsid nohup {base} </dev/null >/dev/null 2>&1{redir} &)"
        return base + " </dev/null" + redir

    # ── 8: launch + stream
    def launch(self) -> None:
        argv = self.claude_argv()
        inner = " ".join(q(a) for a in argv).replace(q("__PROMPT__"), '"$(cat ' + self.prompt_vm_file + ')"')
        # the base's __SYSPROMPT__ expansion (tau3_evolver_daytona.EvolverRun.launch, verbatim); every run that
        # leaves sysprompt_vm_file None carries no placeholder in its argv
        if self.sysprompt_vm_file:
            inner = inner.replace(q("__SYSPROMPT__"), '"$(cat ' + self.sysprompt_vm_file + ')"')
        elif "__SYSPROMPT__" in argv:
            raise Refuse(EXIT_USAGE, "argv carries __SYSPROMPT__ but no system-prompt file is set")
        inner = (f"exec >{de.VM_OUT}/evolver.log 2>{de.VM_OUT}/evolver.stderr </dev/null; "
                 f"set -a; . {de.VM_ENV_FILE}; set +a; cd {de.VM_WORK}; "
                 f"{self.fd_env}=3 {inner} & "
                 f"echo $! > {de.VM_OUT}/pid; wait $!; "
                 f"echo $? > {de.VM_OUT}/rc.tmp && mv {de.VM_OUT}/rc.tmp {de.VM_OUT}/rc")
        cmd = (self.enter_cmd(inner, cwd=de.VM_WORK, detached=True)
               + f" ; sleep 8; {PY} {CTL} tail --cid {self.cid} --offset 0 --max 1500")
        if self.token in cmd:
            raise Refuse(EXIT_TOKEN, "token would appear in a command string")
        self.receipt["daytona"]["launch_argv"] = argv
        self.receipt["daytona"]["launched_utc"] = dle.now()
        self.save_receipt()
        code, out = self.sh(cmd, timeout=180, env={T.SECRET_ENV: self.token}, what="launch",
                            retries=1)                # never re-run: a second claude, a truncated log
        self.launched = True
        log(f"launched in {self.cid}; first poll: {out[:200]!r}")

    def stream(self) -> str:
        """Pull the container's log byte-exactly (host-side read, O_NOFOLLOW) until claude exits."""
        local = self.q / self.stream_name
        off, t0, status, polls = 0, time.time(), "done", 0
        rw_max = int((self.c().get("limits") or {}).get("rw_max_bytes") or (5 << 30))
        with open(local, "ab") as fh:
            while True:
                polls += 1
                du = " --du"                         # the writable dirs' size, every poll (small trees)
                try:
                    code, out = self.sh(f"{PY} {CTL} tail --cid {self.cid} --offset {off} --max {CHUNK}{du}",
                                        timeout=180, what="poll")
                    head, sep, b64 = out.partition("@@B64@@")
                    if not sep:
                        raise ValueError("no marker in poll output")
                    size_s, rc_s, du_s, alive_s = head.strip().splitlines()[-1].split()
                    data = base64.b64decode(b64.strip()) if b64.strip() else b""
                    self._poll_fail = 0
                except Exception as e:  # noqa: BLE001 -- a lost poll is retried, bounded
                    self._poll_fail = getattr(self, "_poll_fail", 0) + 1
                    if self._poll_fail > 40:
                        raise Refuse(EXIT_TIMEOUT, f"polling failed 40x in a row: {type(e).__name__}: {e}")
                    time.sleep(POLL_S)
                    continue
                if data:
                    fh.write(data)
                    fh.flush()
                    off += len(data)
                if int(du_s) > rw_max and status != "timeout":
                    log(f"the container's writable dirs hold {int(du_s) >> 20} MiB > {rw_max >> 20} MiB: stopping")
                    self.c()["stopped_for_disk"] = int(du_s)
                    self.ctl(f"stop-session --cid {self.cid}", timeout=120, what="stop")
                    status, t0 = "timeout", float("inf")
                    continue
                if alive_s == "0" and rc_s == "RUNNING":
                    raise Refuse(EXIT_TIMEOUT, "the container died while claude was running")
                if rc_s != "RUNNING" and off >= int(size_s):
                    self.claude_rc = int(rc_s)
                    break
                if rc_s == "RUNNING" and time.time() - t0 > self.timeout_s:   # an exit before it is never a deadline stop
                    # The epoch message says /evo/staging/ is taken as it stands at the deadline. The stop is recorded as a DEADLINE stop, so
                    # install() takes the staging and run() returns 0 (the disk-limit stop above stays a failure).
                    log(f"DEADLINE after {self.timeout_s}s: stopping claude in the container; staging is taken as it stands")
                    self.ctl(f"stop-session --cid {self.cid}", timeout=120, what="stop")
                    self.deadline_stop = True
                    self.receipt["daytona"]["deadline_stop"] = {"after_s": int(self.timeout_s), "utc": dle.now()}
                    status = "timeout"
                    t0 = float("inf")
                    continue
                if not data:
                    time.sleep(POLL_S)
        log(f"stream finished ({status}): {off} bytes, claude rc {getattr(self, 'claude_rc', None)}")
        return status

    # ── 9: collect
    def container_alive(self) -> bool:
        try:
            code, out, res = self.ctl(f"status --cid {self.cid}", timeout=60, what="status", retries=2)
        except Refuse:
            return False
        return bool(res.get("init_alive"))

    def collect(self) -> None:
        """Also the SALVAGE path after a failure (base run()): whatever the container still holds comes home."""
        c = self.c()
        roots = self.readonly_roots()
        if "after_run_readonly" not in getattr(self, "listing", {}) and self.container_alive():
            after = self.listing_inside([str(r) for r in roots])
            self.listing["after_run_readonly"] = after
            jdump(self.listing_path, self.listing)
            self.after_hashes(after)
            code, out, res = self.ctl(f"probe --cid {self.cid} --phase post", timeout=300, what="post probe",
                                      retries=2)
            c["probe_post"] = res.get("probe") or {}
            c["probe_post_problems"] = pce.post_probe_problems(c.get("probe_pre") or {}, c["probe_post"])
        code, out, res = self.ctl(f"down --cid {self.cid}", timeout=300, what="down")
        c["down_before_pack"] = res
        parts = ["/" + p.lstrip("/") for p in self.download_parts()]
        code, out, res = self.ctl(f"pack --cid {self.cid} " + " ".join(f"--part {q(p)}" for p in parts),
                                  timeout=900, what="pack")
        c["pack"] = res
        self.save_receipt()
        if not res.get("ok"):
            raise Refuse(EXIT_MANIFEST, f"pack refused: {res.get('problems') or out[-400:]}")
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
        dd = self.receipt["daytona"]
        dd["download_refused_members"] = refused + [r["part"] for r in res.get("refused") or []]
        vm_log = ex / de.VM_OUT.lstrip("/") / "evolver.log"
        streamed = self.q / self.stream_name
        same = vm_log.is_file() and streamed.is_file() and de.sha256_file(vm_log) == de.sha256_file(streamed)
        dd["log_round_trip"] = {"streamed_sha256": de.sha256_file(streamed) if streamed.is_file() else None,
                                "vm_sha256": de.sha256_file(vm_log) if vm_log.is_file() else None, "equal": same}
        proj = ex / de.VM_PROJECTS.lstrip("/")
        dd["session_audit"] = de.audit_sessions([streamed, proj])
        dd["heldout_hits"] = de.heldout_id_hits([streamed, proj, ex / str(self.staging).lstrip("/")], self.split) \
            if getattr(self, "split", None) else []
        hits, n = de.token_scan([ex, streamed], self.token)
        self.receipt["token_scan"] = {"hits": hits, "scanned_files": n}
        dd["readback_end"] = dle.readback(self.sb)
        scan = c["down_before_pack"].get("scan") or {}
        mhits = [h for part in scan.values() for h in part.get("hits") or []] + \
            list((c["down_before_pack"].get("tmpfs_scan") or {}).get("hits") or [])
        self.save_receipt()
        log(f"collected: log round-trip equal={same}; token scan {len(hits)} hit(s) in {n} files; container "
            f"state {len(mhits)} prefix hit(s); refused members {len(dd['download_refused_members'])}")
        if hits or mhits:
            raise Refuse(EXIT_TOKEN, f"TOKEN/KEY PREFIX FOUND ({len(hits)} downloaded, {len(mhits)} in the container "
                                     f"state: {mhits[:3]}) -- nothing installed; rotate the credential")
        if c.get("probe_post_problems"):
            raise Refuse(EXIT_IDENTITY, "post probe: " + "; ".join(c["probe_post_problems"][:3]))
        self.collected = True
        if not same:
            raise Refuse(EXIT_MANIFEST, "streamed log != container log")

    # ── 10: teardown (the machine stays)
    def delete(self) -> bool:
        if getattr(self, "sb", None) is None:
            self.release_lease()                      # a refusal between the lease and the attach
            if getattr(self, "receipt", None):
                self.save_receipt()
            return True
        c = self.c()
        ok = False
        try:
            code, out, res = self.ctl(f"down --cid {self.cid} --wipe", timeout=300, what="down wipe", retries=3)
            c["down"] = res
            c["removed"] = ok = bool(res.get("verified") and res.get("wiped"))
        except Refuse as e:
            c["down"] = {"error": str(e)}
        dle.ledger(event="container_down", driver=DRIVER, tag=self.run_dir.name, id=self.sb.id, cid=self.cid,
                   verified=ok)
        self.release_lease()                          # only now: the machine had to stay for the teardown
        self.receipt["daytona"]["container_removed_utc"] = dle.now()
        self.save_receipt()
        log(f"container {self.cid} torn down: verified={ok}; machine {self.sb.id[:8]} left running")
        return ok


class ContainerEvolverRun(ContainerMixin, T.EvolverRun):
    pass


class ContainerAORun(ContainerMixin, T.AORun):
    pass


# ───────────────────────────────────────────────────────────────────────────── subcommands
def cmd_dispatch(a) -> int:
    run_dir = Path(os.environ.get("EVOSKILL_TAU3_RUN_DIR") or "").resolve()
    if not run_dir.is_dir():
        log("EVOSKILL_TAU3_RUN_DIR unset or missing")
        return EXIT_USAGE
    epoch = int(a.epoch)
    prompt, digests, staging = (Path(p) if Path(p).is_absolute() else (REPO / p)
                                for p in (a.prompt, a.digests, a.staging))
    model = os.environ.get("EVOSKILL_EVOLVER_MODEL", "")
    if not model.startswith("claude-"):
        log(f"EVOSKILL_EVOLVER_MODEL must be an explicit claude-* id, got {model!r}")
        return EXIT_USAGE
    mode = os.environ.get("EVOSKILL_EVOLVER_EVIDENCE", "current_epoch")
    prior = [p for p in os.environ.get("EVOSKILL_PRIOR_DIGESTS", "").split(os.pathsep) if p] \
        if mode == "all_epochs" else []
    oracle = Path(os.environ.get("EVOSKILL_TAU3_ORACLE") or (run_dir / "oracle"))
    if not oracle.is_absolute():
        oracle = REPO / oracle
    timeout_s = int(os.environ.get("EVOSKILL_EVOLVER_TIMEOUT_S") or 7200)
    # THE STATEFUL SESSION: the loop's argv is unchanged; a run whose RUNTIME.json says `evolver_session: "stateful"`
    # is driven by evolver_session.StatefulEvolverRun (ONE container and ONE claude session for the whole run).
    # "per_epoch" or absent = this driver. A typo refuses; an env that says stateful for a per-epoch run refuses.
    from src.orchestrator import stateful_layout as SL
    try:
        session = SL.evolver_session(run_dir)
    except ValueError as e:
        log(f"RUNTIME.json: {e}")
        return EXIT_USAGE
    env_session = os.environ.get(SL.ENV_SESSION)
    if session == SL.STATEFUL:
        from evolver_session import StatefulEvolverRun as Cls    # lazy: evolver_session imports this module
        log(f"stateful session: RUNTIME.json {SL.SESSION_KEY}={session!r} -> scripts/daytona/evolver_session.py")
    elif env_session not in (None, "", SL.PER_EPOCH):
        log(f"{SL.ENV_SESSION}={env_session!r} but RUNTIME.json {SL.SESSION_KEY} is {session!r}: refusing")
        return EXIT_USAGE
    else:
        Cls = ContainerEvolverRun
    T.hold_dispatch_lock(run_dir)
    q0 = REPO / ".cache/daytona_evolver" / run_dir.name / f"e{epoch}_{time.strftime('%Y%m%dT%H%M%S')}"

    def make_run():
        q_ = REPO / ".cache/daytona_evolver" / run_dir.name / f"e{epoch}_{time.strftime('%Y%m%dT%H%M%S')}"
        log(f"POOL-CONTAINER EVOLVER e{epoch}: run {run_dir.name}, model {model}, evidence {mode} "
            f"({len(prior) + 1} dir(s)), timeout {timeout_s}s, quarantine {q_}")
        run = Cls(epoch=epoch, prompt=prompt, digests=digests, staging=staging, run_dir=run_dir,
                  prior=prior, model=model, oracle=oracle, cfg=T._cfg(), timeout_s=timeout_s,
                  receipt_path=run_dir / f"e{epoch}_EVOLVER_CONTAINMENT.json",
                  trace_path=run_dir / f"tau3_evolver_e{epoch}.trace.json",
                  listing_path=run_dir / f"e{epoch}_EVOLVER_VM_LISTING.json",
                  log_path=run_dir / f"tau3_evolver_e{epoch}.log", quarantine=q_, mode=mode)
        run.cinit()
        return run
    what = f"the evolver's epoch-{epoch} turn"
    if session == SL.STATEFUL:
        # The launcher's gates are the DIGEST layout's (evidence_surface's digests_e<j>_train naming, the *.md facts,
        # the prompt gate); a stateful session's material is <run>/evo_e{K}/ (raw logs, workspaces, scores) and its
        # texts are fixed templates, so that gate does not apply. The driver's own checks stand: the model pin, the
        # evidence dirs == this run's evo_e0..evo_e{N-1} (build_manifest), the split, the upload listing == the
        # manifest.
        log("stateful session: the launcher's digest-layout gate (tau3_evolver.sh) does not apply to the evo "
            "layout; the driver's own checks run")
        return T.run_resuming(make_run, gate_argv=None, what=what)
    gate = ["bash", str(REPO / "scripts/tau3_evolver.sh"), str(epoch), str(prompt), str(digests),
            str(q0 / "gate_staging")]
    return T.run_resuming(make_run, gate_argv=gate, what=what)


def cmd_ao(a) -> int:
    run_dir = Path(a.run_dir).resolve()
    a.message, a.out, a.log, a.oracle = (str(Path(x).absolute()) for x in (a.message, a.out, a.log, a.oracle))
    a.evidence = [str(Path(x).absolute()) for x in a.evidence]
    a.literal = [str(Path(x).absolute()) for x in (a.literal or [])]
    T.hold_dispatch_lock(run_dir)

    def make_run():
        q_ = REPO / ".cache/daytona_ao" / run_dir.name / f"e{a.epoch}_{time.strftime('%Y%m%dT%H%M%S')}"
        run = ContainerAORun(message=Path(a.message), out=Path(a.out), literals=[Path(x) for x in a.literal],
                             log_header=f"=== prompt author: epoch {a.epoch} model {a.model} host daytona\n",
                             epoch=int(a.epoch), prompt=Path(a.message), digests=Path(a.evidence[-1]),
                             staging=Path(a.out).parent, run_dir=run_dir, prior=list(a.evidence), model=a.model,
                             oracle=Path(a.oracle), cfg=T._cfg(), timeout_s=int(a.timeout_s),
                             receipt_path=run_dir / f"e{a.epoch}_AO_DAYTONA.json",
                             trace_path=q_ / "unused_trace.json",
                             listing_path=run_dir / f"e{a.epoch}_AO_VM_LISTING.json",
                             log_path=Path(a.log), quarantine=q_, mode="all_epochs")
        if a.split:
            run.split = Path(a.split).absolute()
        run.cinit()
        return run
    return T.run_resuming(make_run, gate_argv=None, what=f"the AO's epoch-{a.epoch} turn")


def _probe_run(a, *, prompt_text=None, max_turns=4, install=False) -> tuple:
    run_dir = Path(a.run_dir).resolve()
    T.hold_dispatch_lock(run_dir)
    qd = REPO / ".cache/evolver_container" / f"_{a.cmd}_{time.strftime('%Y%m%dT%H%M%S')}"
    qd.mkdir(parents=True)
    staging = qd / "probe_staging"
    prompt = qd / "PROBE_PROMPT.md"
    prompt.write_text(prompt_text(run_dir, staging) if prompt_text else "unused\n")
    _spec = run_dir / "SPEC.json"
    _declared = json.loads(_spec.read_text()).get("spec", {}).get("split_path") if _spec.is_file() else None
    os.environ.setdefault("EVOSKILL_TAU3_EVOLVER_SPLIT", _declared or str(REPO / "examples/ssbv2/split.json"))
    cfg = T._cfg()
    if not cfg:                       # no env: the run's own RUNTIME.json evolver_daytona (+ this placement)
        try:
            cfg = dict(json.loads((run_dir / "RUNTIME.json").read_text()).get("evolver_daytona") or {})
        except (OSError, ValueError):
            cfg = {}
        cfg["placement"] = pce.PLACEMENT
    run = ContainerEvolverRun(epoch=1, prompt=prompt, digests=run_dir / "digests_e1_train", staging=staging,
                              run_dir=run_dir, prior=[], model=a.model, oracle=run_dir / "oracle", cfg=cfg,
                              timeout_s=900, receipt_path=qd / "receipt.json", trace_path=qd / "trace.json",
                              listing_path=qd / "listing.json", log_path=qd / "probe_evolver.log",
                              quarantine=qd / "q", max_turns=max_turns, install=install, mode="all_epochs")
    run.cinit(sandbox_id=a.sandbox_id, registry=Path(a.registry) if a.registry else None, episode_check=True)
    return run, qd


def cmd_smoke_container(a) -> int:
    """NO model call, no credential use: attach, prepare, up, PRE probe, upload + listing, POST probe, down, wipe."""
    run, qd = _probe_run(a)
    rc = 0
    try:
        run.prepare()
        run.create()
        run.probes()
        run.upload()
        code, out, res = run.ctl(f"probe --cid {run.cid} --phase post", timeout=300, what="post probe")
        run.c()["probe_post"] = res.get("probe") or {}
        run.c()["probe_post_problems"] = pce.post_probe_problems(run.c().get("probe_pre") or {},
                                                                 run.c()["probe_post"])
        code, out, res = run.ctl(f"down --cid {run.cid}", timeout=300, what="down")
        run.c()["down_before_pack"] = res
        if run.c()["probe_post_problems"]:
            rc = EXIT_IDENTITY
    except Refuse as e:
        log(f"REFUSED: {e}")
        rc = e.code
    finally:
        if not run.delete():
            rc = rc or EXIT_DELETE
    run.save_receipt()
    rep = {"rc": rc, "cid": run.cid, "receipt": str(run.receipt_path),
           "probe_pre_problems": run.c().get("probe_pre_problems"), "probe_post_problems":
               run.c().get("probe_post_problems"), "down": run.c().get("down")}
    jdump(qd / "SMOKE_REPORT.json", rep)
    log("SMOKE REPORT: " + json.dumps(rep, default=str)[:1500])
    return rc


def _probe_prompt(run_dir: Path, staging: Path) -> str:
    return ("This is an infrastructure probe, not an evolution task. Do exactly these steps and nothing else.\n"
            "1. Run ONE Bash command: id; grep -E '^(Cap|NoNewPrivs)' /proc/self/status; ls -A / /opt /home "
            f"/Users; ls {run_dir}; ls /proc | grep -c '^[0-9]'; ls /opt/evoskill 2>&1 | head -1\n"
            f"2. Write the file {staging}/probe-skill/SKILL.md containing exactly:\n"
            "---\nname: probe-skill\ndescription: infrastructure probe\n---\nprobe\n"
            "3. Reply with the single word DONE.\n")


def cmd_probe(a) -> int:
    """The operator's live smoke: the container probes, a 2-tool-turn Claude session that lists its view and writes
    one file to staging; the receipt re-checked with the loop's own checks."""
    run, qd = _probe_run(a, prompt_text=_probe_prompt, max_turns=4)
    rc = run.run(gate_argv=None)
    rep = {"rc": rc, "cid": run.cid}
    if rc == 0:
        shutil.copyfile(qd / "q" / "evolver.log", qd / "probe_evolver.log")
        ok, det = pce.check_pool_container_receipt(
            run.receipt, log=qd / "probe_evolver.log", require_libreoffice=run.cfg.get("require_libreoffice"),
            require_snapshot_ref=run.leg_snapshot_ref(), staging=run.staging, prefixes=(pce.POOL_PREFIX,))
        m = de.check_upload_manifest(run_dir=run.run_dir, epoch=1, split=run.split, evidence_dirs=run.evidence_dirs,
                                     prev_library=run.prev_lib, prompt=run.prompt, oracle=run.run_dir / "oracle",
                                     staging=run.staging, receipt=run.receipt, listing=run.listing)
        from src.orchestrator.claude_pins import check_session
        rep.update(receipt_check=[ok, det], manifest_check=[m.status, m.detail],
                   check_session=check_session(qd / "probe_evolver.log", a.model),
                   staged=sorted(str(p.relative_to(qd / "q" / "extract")) for p in (qd / "q" / "extract").rglob("*")
                                 if p.is_file() and "probe_staging" in str(p)))
    jdump(qd / "PROBE_REPORT.json", rep)
    log("PROBE REPORT: " + json.dumps(rep, default=str)[:2000])
    return rc


def _live_stateful_session(run_dir, sandbox_id) -> str:
    """'' or why `cleanup` must not wipe: the run's stateful session record (.cache/evolver_state/<run>/container.json)
    is not torn down (or cannot be read), or any run's record names `sandbox_id` and is not torn down.
    `evolver_session.py finish` is the teardown that ledgers and records it."""
    from src.orchestrator import stateful_layout as SL
    if run_dir:
        try:
            st = SL.load_state(Path(run_dir).resolve())
        except ValueError as e:
            return f"the run's stateful session record is unreadable ({e})"
        if st is not None and not st.get("torn_down"):
            return (f"a live stateful session's container {st.get('cid')} on {str(st.get('sandbox_id'))[:8]} "
                    f"(session {str(st.get('sid'))[:8]})")
    if sandbox_id and SL.STATE_ROOT.is_dir():
        for p in sorted(SL.STATE_ROOT.glob(f"*/{SL.STATE_FILE}")):
            try:
                st = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            if isinstance(st, dict) and st.get("sandbox_id") == sandbox_id and not st.get("torn_down"):
                return (f"a live stateful session's container {st.get('cid')} of run {st.get('run')!r} is on "
                        f"{str(sandbox_id)[:8]}")
    return ""


def cmd_cleanup(a) -> int:
    """Stop and wipe the evolver containers left on the run's pool machine (the machine stays), refused while a
    dispatch of the run is live (its per-run lock) or its stateful session is not torn down."""
    run_dir = Path(a.run_dir).resolve()
    lock = T.exclusive_dispatch_lock(run_dir)
    if lock is None:
        log(f"a dispatch of {run_dir.name} is live; refusing cleanup")
        return T.EXIT_LIVE
    force = bool(getattr(a, "force", False))
    why = _live_stateful_session(run_dir, None)
    if why and not force:
        log(f"{why}; refusing cleanup: use `evolver_session.py finish --run-dir R` (or --force)")
        return T.EXIT_LIVE
    d = dle.client()
    sid = a.sandbox_id
    if not sid:
        regf = Path(a.registry) if a.registry else pce.pool_machines_file(run_dir)
        reg = json.loads(regf.read_text()) if regf.is_file() else {}
        if reg.get("schema") == pce.POOL_MACHINES_SCHEMA:          # the run pool's machine.json: its current machine
            row = (reg.get("machines") or {}).get(reg.get("current") or "") or {}
            sid = row.get("sandbox_id")
    if not sid:
        log("no machine named (--sandbox-id, or a current machine in the run pool's machine.json)")
        return EXIT_USAGE
    why = _live_stateful_session(None, sid)
    if why and not force:
        log(f"{why}; refusing cleanup: use `evolver_session.py finish --run-dir R` (or --force)")
        return T.EXIT_LIVE
    sb = d.get(sid)
    code, out, _ = dle.run(sb, f"{PY} {CTL} list", timeout=120)
    lst = parse_json_tail(out)
    for cid in lst.get("containers") or []:
        code, out, _ = dle.run(sb, f"{PY} {CTL} down --cid {q(cid)} --wipe", timeout=300)
        res = parse_json_tail(out)
        dle.ledger(event="container_down", driver=DRIVER, tag="cleanup", id=sid, cid=cid,
                   verified=bool(res.get("verified") and res.get("wiped")))
        log(f"{cid}: verified={res.get('verified')} wiped={res.get('wiped')}")
    log(f"{len(lst.get('containers') or [])} container(s) on {sid[:8]}; the machine is left running")
    lock.close()
    return 0


def _on_sigterm(signum, frame):
    """SIGTERM (prompt_author's AO timeout) -> SystemExit, so run()'s `finally` still tears the container down
    and removes the pool lease (the default action would skip both and leave a container on the machine)."""
    raise SystemExit(128 + signum)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    signal.signal(signal.SIGTERM, _on_sigterm)
    if argv and argv[0] in ("probe", "smoke-container"):
        ap = argparse.ArgumentParser(prog=f"evolver_container.py {argv[0]}")
        ap.add_argument("--run-dir", required=True)
        ap.add_argument("--model", default="claude-opus-5-5")
        ap.add_argument("--sandbox-id")
        ap.add_argument("--registry", help="a pool machine.json other than the run pool's")
        a = ap.parse_args(argv[1:])
        a.cmd = argv[0].replace("-", "_")
        return cmd_probe(a) if argv[0] == "probe" else cmd_smoke_container(a)
    if argv and argv[0] == "cleanup":
        ap = argparse.ArgumentParser(prog="evolver_container.py cleanup")
        ap.add_argument("--run-dir", required=True, help="the run (its dispatch lock, its pool machine.json)")
        ap.add_argument("--sandbox-id")
        ap.add_argument("--registry")
        ap.add_argument("--force", action="store_true",
                        help="wipe even a LIVE stateful session's container (normally: evolver_session.py finish)")
        return cmd_cleanup(ap.parse_args(argv[1:]))
    if argv and argv[0] == "ao":
        ap = argparse.ArgumentParser(prog="evolver_container.py ao")
        for k in ("--run-dir", "--epoch", "--message", "--out", "--log", "--model", "--oracle"):
            ap.add_argument(k, required=True)
        ap.add_argument("--evidence", action="append", required=True)
        ap.add_argument("--literal", action="append")
        ap.add_argument("--split")
        ap.add_argument("--timeout-s", default="3600")
        return cmd_ao(ap.parse_args(argv[1:]))
    ap = argparse.ArgumentParser(prog="evolver_container.py")
    for k in ("epoch", "prompt", "digests", "staging"):
        ap.add_argument(k)
    return cmd_dispatch(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
