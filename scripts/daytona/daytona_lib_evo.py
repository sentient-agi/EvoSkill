"""Daytona helpers for the EVOLVER and adaptor sandboxes, host side. A copy of daytona_lib_loop.py that differs only
in the sandbox name prefix (evoskill-evo-), the owner label and the ledger's driver name. A separate prefix is
load-bearing: ssb_leg_daytona_loop.py cmd_cleanup deletes EVERY listed sandbox with the loop prefix whenever no leg
driver is alive, which would kill an evolver running between legs; and delete_by_id here refuses any name without
THIS prefix, so neither driver can delete the other's sandboxes. Same ledger file (one place to audit every sandbox).

Hard rules enforced here (inherited): DAYTONA_* read from the environment, else from the repo's .env, in-process and
never printed; every sandbox ledgered before create, when its id is first visible, and after create (readback);
delete by recorded id only, verified and ledgered; ttl backstop on every sandbox. The Claude Code credential
reaches the sandbox only as the env of ONE SDK exec call (the launch), never as Sandbox env_vars (stored
server-side and returned by the API), never in a command string.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import shlex
import threading
import time
from pathlib import Path

PREFIX = "evoskill-evo-"
OWNER_LABEL = "evoskill-evo"
REPO = Path(__file__).resolve().parents[2]
# Secrets come from the environment, else from the repo's .env (EVOSKILL_ENV_FILE names another file).
ENV_FILE = os.environ.get("EVOSKILL_ENV_FILE") or str(REPO / ".env")
# One JSONL ledger for every sandbox the drivers create; rows are told apart by the name prefix and the owner label.
LEDGER = os.environ.get("EVOSKILL_DAYTONA_LEDGER") or str(REPO / ".cache" / "daytona" / "ledger.jsonl")
_lock = threading.Lock()


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _env_file_values() -> dict:
    if not os.path.isfile(ENV_FILE):
        return {}
    from dotenv import dotenv_values
    return dotenv_values(ENV_FILE)


def secret(name: str) -> str:
    """NAME from the environment, else from ENV_FILE; '' when neither sets it. Never printed or logged."""
    return (os.environ.get(name) or "").strip() or (_env_file_values().get(name) or "").strip()


def ledger(**kw) -> None:
    kw = {"ts": now(), "driver": "tau3_evolver_daytona", **kw}
    with _lock:
        os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
        with open(LEDGER, "a") as fh:
            fh.write(json.dumps(kw, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


def load_daytona_env() -> None:
    k = secret("DAYTONA_API_KEY")
    if not k:
        raise SystemExit(f"DAYTONA_API_KEY is set neither in the environment nor in {ENV_FILE}")
    os.environ["DAYTONA_API_KEY"] = k
    for extra in ("DAYTONA_API_URL", "DAYTONA_TARGET"):
        v = secret(extra)
        if v:
            os.environ[extra] = v


def client():
    from daytona import Daytona, DaytonaConfig
    load_daytona_env()
    return Daytona(DaytonaConfig(api_key=os.environ["DAYTONA_API_KEY"]))


def readback(sb) -> dict:
    """A FRESH read of the sandbox's settings. `refreshed` says whether the refresh succeeded; a False
    here means the values below are the ones cached at create, not a measurement."""
    refreshed = False
    for _ in range(4):
        try:
            sb.refresh_data()
            refreshed = True
            break
        except Exception:
            time.sleep(5)
    bi = getattr(sb, "build_info", None)
    return {"refreshed": refreshed, "cpu": getattr(sb, "cpu", None), "memory": getattr(sb, "memory", None),
            "disk": getattr(sb, "disk", None), "state": str(getattr(sb, "state", None)),
            "sandbox_class": getattr(sb, "sandbox_class", None), "target": getattr(sb, "target", None),
            "runner_id": getattr(sb, "runner_id", None), "created_at": getattr(sb, "created_at", None),
            "auto_stop_interval": getattr(sb, "auto_stop_interval", None),
            "auto_delete_interval": getattr(sb, "auto_delete_interval", None),
            "auto_destroy_at": getattr(sb, "auto_destroy_at", None),
            "network_block_all": getattr(sb, "network_block_all", None),
            "network_allow_list": getattr(sb, "network_allow_list", None),
            "domain_allow_list": getattr(sb, "domain_allow_list", None),
            "snapshot": getattr(sb, "snapshot", None),
            "snapshot_ref": getattr(bi, "snapshot_ref", None) if bi is not None else None}


def create(d, params, requested: dict, tag: str, timeout: float = 3600, on_logs=None):
    """Create; ledger request -> id (as soon as visible) -> readback. Returns (sb, id, err, secs)."""
    name = params.name
    assert name and name.startswith(PREFIX), name
    params.labels = dict(params.labels or {})
    params.labels.update({"owner": OWNER_LABEL, "tag": tag})
    ledger(event="create_requested", tag=tag, name=name, requested=requested,
           ttl_minutes=getattr(params, "ttl_minutes", None),
           auto_stop_interval=getattr(params, "auto_stop_interval", None),
           domain_allow_list=getattr(params, "domain_allow_list", None))
    box: dict = {}
    t0 = time.time()

    def _go():
        try:
            if on_logs is not None:
                box["sb"] = d.create(params, timeout=timeout, on_snapshot_create_logs=on_logs)
            else:
                box["sb"] = d.create(params, timeout=timeout)
        except BaseException as e:  # noqa: BLE001
            box["err"] = e

    th = threading.Thread(target=_go, daemon=True)
    th.start()
    seen_id = None
    while th.is_alive():
        th.join(10)
        if seen_id is None:
            try:
                s = d.get(name)
                seen_id = s.id
                ledger(event="id_seen_during_create", tag=tag, name=name, id=seen_id,
                       state=str(s.state), cpu=s.cpu, memory=s.memory, disk=s.disk)
            except Exception:
                pass
    dt = round(time.time() - t0, 1)
    if "err" in box:
        err = box["err"]
        sid = seen_id
        if sid is None:
            try:
                sid = d.get(name).id
            except Exception:
                sid = None
        ledger(event="create_failed", tag=tag, name=name, id=sid, seconds=dt,
               error=f"{type(err).__name__}: {str(err)[:1500]}")
        return None, sid, err, dt
    sb = box["sb"]
    rb = readback(sb)
    mism = {k: (requested.get(k), rb.get(k)) for k in ("cpu", "memory", "disk")
            if requested.get(k) is not None and requested.get(k) != rb.get(k)}
    ledger(event="created", tag=tag, name=name, id=sb.id, seconds=dt, readback=rb,
           requested=requested, resource_mismatch=mism or None)
    return sb, sb.id, None, dt


def delete_by_id(d, sid: str, tag: str) -> bool:
    """Delete (not stop) the sandbox with THIS id, verify it is gone, ledger both."""
    if not sid:
        return True
    ok = False
    for attempt in range(6):
        try:
            sb = d.get(sid)
        except Exception as e:
            if "not found" in str(e).lower() or "404" in str(e):
                ok = True
                break
            time.sleep(5)
            continue
        if not str(sb.name).startswith(PREFIX):
            ledger(event="delete_refused_not_ours", tag=tag, id=sid, name=sb.name)
            return False
        try:
            d.delete(sb, timeout=180, wait=True)
            ok = True
            break
        except Exception as e:
            ledger(event="delete_retry", tag=tag, id=sid, attempt=attempt, error=str(e)[:300])
            time.sleep(10)
    state = None
    try:
        state = str(d.get(sid).state)
    except Exception as e:
        state = f"get-failed: {str(e)[:160]}"
    gone = ok and ("destroy" in state.lower() or "not found" in state.lower()
                   or "get-failed" in state.lower())
    ledger(event="deleted" if gone else "delete_UNVERIFIED", tag=tag, id=sid, post_state=state)
    return gone


def list_ours(d) -> list[dict]:
    rows = []
    for sb in d.list():
        if str(sb.name).startswith(PREFIX):
            rows.append({"id": sb.id, "name": sb.name, "state": str(sb.state)})
    return rows


def list_all_ids(d) -> list[str]:
    return [f"{sb.id[:8]} {sb.name} {sb.state}" for sb in d.list()]


def run(sb, cmd: str, timeout: int = 600, env: dict | None = None):
    """Run a bash command; returns (exit_code, combined output, seconds). `env` goes to the SDK
    as process environment -- the only way a secret may enter a sandbox."""
    t0 = time.time()
    r = sb.process.exec("bash -c " + shlex.quote(cmd + "\n") + " 2>&1", timeout=timeout, env=env)
    return r.exit_code, (r.result or ""), round(time.time() - t0, 2)
