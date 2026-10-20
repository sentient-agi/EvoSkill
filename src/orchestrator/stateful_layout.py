"""Stateful evolver: the container-native layout, the Mac <-> container path view and the state record on the Mac.

RUNTIME `evolver_session: "stateful"` = ONE long-lived container and ONE claude session for the whole run; absent
or "per_epoch" = the one-container-per-epoch driver (nothing here is called on that path).

The container layout (K = epoch index, N = the current epoch):

    /evo/traces/epoch_K/<id>_s<i>.jsonl         K = 0..N-1  <- <run>/evo_e{K}/traces/           (the raw session logs)
    /evo/scores/epoch_K.json                    K = 0..N-1  <- <run>/evo_e{K}/scores.json        (task id, sample, score)
    /evo/solver_workspaces/epoch_K/<id>_s<i>/   K = 0..N-1  <- <run>/evo_e{K}/solver_workspaces/ (the solver's cwd at exit)
    /evo/skills/epoch_K/                        K = 1..N-1  <- <run>/skills_vK                   (role skills_snapshot)
    /evo/staging/                               rw          <- <run>/skills_v{N-1} seed at N >= 2 (cleared first)
    /evo/evolver_workspace/                     rw          the evolver's own, persistent; never uploaded, never cleared
    /evo/reference/                                         <- <run>/reference (the benchmark adapter's copy; NOT scanned)

No message or system-prompt FILE reaches the container: both travel on the launch command line (base64 shell
variables, scripts/daytona/evolver_session.py). No grader / references / inputs dirs: the adapter's reference/ is
the evolver's whole view of the benchmark, uploaded as the adapter wrote it, with no name or hash scan.
"""
from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
from dataclasses import dataclass
from pathlib import Path

from src.orchestrator import daytona_evolver as de
from src.orchestrator import evo_layout as EL

# ── RUNTIME key ────────────────────────────────────────────────────────────────────────────────────
SESSION_KEY = "evolver_session"
PER_EPOCH = "per_epoch"
STATEFUL = "stateful"
SESSIONS = (PER_EPOCH, STATEFUL)

# ── container paths ───────────────────────────────────────────────────────────────────────────────
VM_TRACES = "/evo/traces"                    # /evo/traces/epoch_K            <- evo_e{K}/traces
VM_SCORES = "/evo/scores"                    # /evo/scores/epoch_K.json       <- evo_e{K}/scores.json
VM_SOLVER_WS = "/evo/solver_workspaces"      # /evo/solver_workspaces/epoch_K <- evo_e{K}/solver_workspaces
VM_SKILLS = "/evo/skills"                    # /evo/skills/epoch_K            <- skills_vK (K >= 1)
VM_STAGING = "/evo/staging"                  # rw; <- skills_v{N-1} seed at epoch N >= 2
VM_WORKSPACE = "/evo/evolver_workspace"      # rw; the evolver's own, persistent
VM_REFERENCE = "/evo/reference"              # <- <run>/reference (the adapter's copy)
# The evolver has NO grader of its own: a stateful evolver tests a skill through the next epoch's train leg, graded
# by the loop, and reads the scores.
STATEFUL_RW = ({"path": VM_STAGING, "owner": "evo", "mode": "0755"},
               {"path": VM_WORKSPACE, "owner": "evo", "mode": "0755"})
EVOLVER_WRITABLE = (VM_STAGING, VM_WORKSPACE, de.VM_WORK, de.VM_OUT)   # excluded from listing equality
REFERENCE_DIR_NAME = "reference"             # <run>/reference (scripts/daytona/setup_ao_probe.py fills it)

# ── manifest roles ────────────────────────────────────────────────────────────────────────────────
ROLE_TRACES = "traces"
ROLE_SCORES = "scores"
ROLE_SOLVER_WS = "solver_workspaces"
ROLE_SKILLS = "skills_snapshot"
ROLE_SEED = "seed"
ROLE_REFERENCE = "reference"
ROLE_HOOK, ROLE_ENV, ROLE_POLICY = "hook", "env", "policy"      # daytona_evolver.build_upload_manifest's
EVO_ROLES = (ROLE_TRACES, ROLE_SOLVER_WS, ROLE_SCORES)         # the per-epoch material (evo_layout)
DIR_ROLES = (ROLE_TRACES, ROLE_SOLVER_WS, ROLE_SKILLS, ROLE_SEED, ROLE_REFERENCE)
FILE_ROLES = (ROLE_SCORES,)
STATEFUL_ROLES = frozenset({*DIR_ROLES, *FILE_ROLES, ROLE_HOOK, ROLE_ENV, ROLE_POLICY})
UNSCANNED_ROLES = frozenset({ROLE_TRACES, ROLE_SOLVER_WS, ROLE_SCORES, ROLE_REFERENCE})   # the solver's own
#                                                                    material and the adapter's copy: no name / hash scan

# ── run-dir files, state record, env ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT_FILE = "EVOLVER_SYSTEM.md"     # <run>/EVOLVER_SYSTEM.md, rendered ONCE at epoch 1 (a Mac record)
WORKSPACE_DIR = "e{epoch}_evolver_workspace"  # <run>/e{N}_evolver_workspace/ (installed copy of /evo/evolver_workspace)
TEARDOWN_RECORD = "EVOLVER_SESSION_TEARDOWN.json"
STATE_ROOT = de.REPO / ".cache/evolver_state"
STATE_FILE = "container.json"
STATE_SCHEMA = "evoskill.evolver_session/1"
SNAPSHOT_SCHEMA = "evoskill.evolver_snapshot/1"
SNAPSHOT_MANIFEST = "MANIFEST.json"
SNAPSHOT_WORKSPACE = "workspace"             # <snapshot>/workspace/ = the evolver's workspace at collect time
ENV_SESSION = "EVOSKILL_EVOLVER_SESSION"
ENV_SYSTEM_PROMPT = "EVOSKILL_EVOLVER_SYSTEM_PROMPT"
LEDGER_REATTACH = "container_reattach"
RECEIPT_FILE = "e{epoch}_EVOLVER_CONTAINMENT.json"
LISTING_FILE = "e{epoch}_EVOLVER_VM_LISTING.json"
# Never in the message / system prompt: a Mac path or a Mac-side dir name (the templates are container-native).
_FORBIDDEN_NEEDLES = ("/Users/", "evo_e", "digests_e", "_staging")
_SKILLS_V_DIR_RE = re.compile(r"skills_v\d+[/_]", re.IGNORECASE)   # `skills_v{K}` alone is a label; followed by
#                                                                    "/" or "_" it is a Mac dir (skills_v1/, skills_v1_staging)
_EVO_DIR_RE = re.compile(r"^evo_e(\d+)$")


# ── the RUNTIME reader ────────────────────────────────────────────────────────────────────────────
def evolver_session(run_dir=None, *, runtime: dict | None = None) -> str:
    """RUNTIME.json `evolver_session`: "per_epoch" (absent) or "stateful"; anything else raises ValueError
    ("a typo must not silently mean either"). File-based: the loop, the driver and the upload check read the
    SAME file. A run dir with no RUNTIME.json (offline callers, unit fixtures) -> "per_epoch"."""
    if runtime is None:
        rt_file = Path(run_dir) / "RUNTIME.json" if run_dir is not None else None
        if rt_file is None or not rt_file.is_file():
            return PER_EPOCH
        try:
            runtime = json.loads(rt_file.read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"{rt_file} unreadable ({type(e).__name__}): refusing to guess {SESSION_KEY}")
    if not isinstance(runtime, dict):
        raise ValueError("RUNTIME.json must be an object")
    v = runtime.get(SESSION_KEY, PER_EPOCH)
    if not isinstance(v, str) or v not in SESSIONS:
        raise ValueError(f"RUNTIME.json {SESSION_KEY}={v!r} is not one of {SESSIONS}. Absent means "
                         f"'per_epoch'; a typo must not silently mean either.")
    return v


# ── the layout ────────────────────────────────────────────────────────────────────────────────────
def _epoch_index(k, *, floor: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or k < floor:
        raise ValueError(f"epoch index {k!r}: an integer >= {floor}")
    return k


def traces_dir(K: int) -> str:
    return f"{VM_TRACES}/epoch_{_epoch_index(K, floor=0)}"


def scores_file(K: int) -> str:
    return f"{VM_SCORES}/epoch_{_epoch_index(K, floor=0)}.json"


def solver_ws_dir(K: int) -> str:
    return f"{VM_SOLVER_WS}/epoch_{_epoch_index(K, floor=0)}"


def skills_dir(K: int) -> str:
    return f"{VM_SKILLS}/epoch_{_epoch_index(K, floor=1)}"


def evo_epoch(mac_dir) -> int:
    """<run>/evo_e{K} -> K (the epoch whose rollouts it holds; 0 = the base leg). ValueError for any other name."""
    m = _EVO_DIR_RE.match(Path(str(mac_dir)).name)
    if not m:
        raise ValueError(f"{mac_dir} is not an evo_e<K> dir")
    return int(m.group(1))


@dataclass(frozen=True)
class PathMap:
    """One Mac dir (or file) and where it sits in the container. `epoch` = the epoch that UPLOADS it (the delta
    of epoch N is every entry with epoch == N)."""
    mac: str
    vm: str
    role: str
    epoch: int

    def as_dict(self) -> dict:
        return {"mac": self.mac, "vm": self.vm, "role": self.role, "epoch": self.epoch}


def path_view(run_dir, epoch: int) -> list[PathMap]:
    """Ordered, one entry per Mac dir/file the container holds at epoch `epoch` (CUMULATIVE over the run):

        evo_e{K}/traces             -> traces_dir(K),    role traces,            epoch K+1   for K in 0..epoch-1
        evo_e{K}/solver_workspaces  -> solver_ws_dir(K), role solver_workspaces, epoch K+1
        evo_e{K}/scores.json        -> scores_file(K),   role scores,            epoch K+1
        skills_v{K}                 -> skills_dir(K),    role skills_snapshot,   epoch K+1   for K in 1..epoch-1
        skills_v{epoch-1}           -> VM_STAGING,       role seed,              epoch `epoch` (epoch >= 2)
        reference                   -> VM_REFERENCE,     role reference,         epoch 1

    Mac paths are realpath'd and must lie inside THIS run dir (ValueError otherwise). A pure mapping: nothing is
    required to exist."""
    N = _epoch_index(epoch, floor=1)
    rd = Path(os.path.realpath(str(run_dir)))
    root = str(rd)

    def mac(p) -> str:
        real = os.path.realpath(str(p))
        if not real.startswith(root + os.sep):
            raise ValueError(f"{p} resolves to {real}, outside this run dir {root}: every mapped path is THIS run's")
        return real

    out: list[PathMap] = []
    for K in range(N):
        d = EL.epoch_dir(rd, K)
        out.append(PathMap(mac(d / EL.TRACES), traces_dir(K), ROLE_TRACES, K + 1))
        out.append(PathMap(mac(d / EL.WORKSPACES), solver_ws_dir(K), ROLE_SOLVER_WS, K + 1))
        out.append(PathMap(mac(d / EL.SCORES_JSON), scores_file(K), ROLE_SCORES, K + 1))
    for K in range(1, N):
        out.append(PathMap(mac(rd / f"skills_v{K}"), skills_dir(K), ROLE_SKILLS, K + 1))
    if N >= 2:
        out.append(PathMap(mac(rd / f"skills_v{N - 1}"), VM_STAGING, ROLE_SEED, N))
    out.append(PathMap(mac(rd / REFERENCE_DIR_NAME), VM_REFERENCE, ROLE_REFERENCE, 1))
    return out


def _longest(view, attr: str, path: str, role: str | None):
    """The view entry whose `attr` root equals `path` or is a proper directory prefix of it, the longest root
    winning; a tie (skills_v{N-1} is BOTH the read-only snapshot and the writable seed) resolves to the EARLIER
    entry, i.e. the snapshot, unless `role` picks one."""
    best = None
    for pm in view:
        if role is not None and pm.role != role:
            continue
        root = getattr(pm, attr)
        if path == root or path.startswith(root + "/"):
            if best is None or len(root) > len(getattr(best, attr)):
                best = pm
    return best


def vm_of(view, mac_path: str, *, role: str | None = None) -> str:
    p = os.path.realpath(str(mac_path))
    pm = _longest(view, "mac", p, role)
    if pm is None:
        raise KeyError(f"{mac_path} is not mapped into the container by the path view")
    return pm.vm + p[len(pm.mac):]


def mac_of(view, vm_path: str, *, role: str | None = None) -> str:
    p = posixpath.normpath(str(vm_path))
    if not p.startswith("/"):
        raise KeyError(f"{vm_path!r} is not an absolute container path")
    pm = _longest(view, "vm", p, role)
    if pm is None:
        raise KeyError(f"{vm_path} is not a container path the path view maps")
    return pm.mac + p[len(pm.vm):]


# ── the state record (.cache/evolver_state/<run>/container.json) ──────────────────────────────────
def run_name(run_dir) -> str:
    name = Path(os.path.abspath(str(run_dir))).name
    if not name:
        raise ValueError(f"run dir {run_dir!r} has no name")
    return name


def state_path(run_dir) -> Path:
    return STATE_ROOT / run_name(run_dir) / STATE_FILE


def snapshot_dir(run_dir, epoch) -> Path:
    return STATE_ROOT / run_name(run_dir) / f"e{_epoch_index(epoch, floor=1)}"


def _check_state(run_dir, state) -> dict:
    if not isinstance(state, dict):
        raise ValueError("the state record must be an object")
    if state.get("schema") != STATE_SCHEMA:
        raise ValueError(f"state record schema {state.get('schema')!r} != {STATE_SCHEMA!r}")
    if state.get("run") != run_name(run_dir):
        raise ValueError(f"state record names run {state.get('run')!r}, this run dir is {run_name(run_dir)!r}")
    return state


def load_state(run_dir) -> dict | None:
    p = state_path(run_dir)
    if not p.is_file():
        return None
    try:
        st = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"{p} unreadable ({type(e).__name__}): refusing to guess the session state")
    return _check_state(run_dir, st)


def write_state(run_dir, state: dict) -> None:
    """Atomic: a tmp file beside the record (0600), fsync, os.replace. The run's state dir (and the root) are 0700."""
    _check_state(run_dir, state)
    p = state_path(run_dir)
    for d in (STATE_ROOT, p.parent):
        d.mkdir(parents=True, exist_ok=True)
        os.chmod(d, 0o700)
    tmp = p.with_name(f"{p.name}.tmp.{os.getpid()}")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(state, fh, indent=1, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            tmp.unlink()


# ── snapshots (.cache/evolver_state/<run>/e<N>/) ──────────────────────────────────────────────────
def _walk_files(d: Path) -> dict:
    out = {}
    for dirpath, dirnames, filenames in os.walk(d, followlinks=False):
        dirnames.sort()
        for n in list(dirnames):
            if (Path(dirpath) / n).is_symlink():
                raise ValueError(f"symlink in the snapshot: {Path(dirpath, n).relative_to(d)}")
        for n in sorted(filenames):
            p = Path(dirpath) / n
            rel = p.relative_to(d).as_posix()
            if p.is_symlink():
                raise ValueError(f"symlink in the snapshot: {rel}")
            if rel == SNAPSHOT_MANIFEST:
                continue
            if not p.is_file():
                raise ValueError(f"not a regular file in the snapshot: {rel}")
            out[rel] = p
    return out


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot_manifest(d: Path) -> dict:
    d = Path(d)
    if not d.is_dir():
        raise ValueError(f"snapshot dir {d} does not exist")
    files = {}
    for rel, p in _walk_files(d).items():
        files[rel] = {"sha256": _sha256(p), "bytes": p.stat().st_size}
    return {"schema": SNAPSHOT_SCHEMA, "files": files, "n_files": len(files),
            "bytes": sum(v["bytes"] for v in files.values())}


def verify_snapshot(d: Path) -> list:
    d = Path(d)
    if not d.is_dir():
        return [f"snapshot dir {d} does not exist"]
    mf = d / SNAPSHOT_MANIFEST
    if not mf.is_file() or mf.is_symlink():
        return [f"{mf} missing"]
    try:
        man = json.loads(mf.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return [f"{mf} unreadable ({type(e).__name__})"]
    if not isinstance(man, dict) or man.get("schema") != SNAPSHOT_SCHEMA:
        return [f"{mf} schema {getattr(man, 'get', lambda k: None)('schema')!r} != {SNAPSHOT_SCHEMA!r}"]
    probs = []
    if man.get("contaminated"):
        probs.append("snapshot is marked contaminated (a held-out id reached the session); unusable")
    files = man.get("files")
    if not isinstance(files, dict):
        return probs + [f"{mf} has no files map"]
    try:
        on_disk = _walk_files(d)
    except ValueError as e:
        return probs + [str(e)]
    for rel, want in sorted(files.items()):
        if not isinstance(rel, str) or rel.startswith("/") or ".." in rel.split("/") or rel == SNAPSHOT_MANIFEST:
            probs.append(f"manifest lists an unsafe path {rel!r}")
            continue
        p = on_disk.get(rel)
        if p is None:
            probs.append(f"listed file missing: {rel}")
            continue
        if not isinstance(want, dict):
            probs.append(f"manifest entry for {rel} is not an object")
            continue
        size = p.stat().st_size
        if want.get("bytes") != size:
            probs.append(f"{rel}: {size} bytes on disk, manifest says {want.get('bytes')!r}")
        if want.get("sha256") != _sha256(p):
            probs.append(f"{rel}: sha256 differs from the manifest")
    for rel in sorted(set(on_disk) - set(k for k in files if isinstance(k, str))):
        probs.append(f"unlisted file in the snapshot: {rel}")
    return probs


# ── helpers ───────────────────────────────────────────────────────────────────────────────────────
def _under(p: str, d: str) -> bool:
    d = d.rstrip("/")
    return p == d or p.startswith(d + "/")


def probe_sha256(probe: dict) -> str:
    return hashlib.sha256(json.dumps(probe, sort_keys=True).encode()).hexdigest()


def forbidden_mac_strings(text: str) -> list:
    """Every Mac-side name in `text` that a container-native prompt must not carry: a macOS home path, `evo_e`, `digests_e`,
    `_staging`, `skills_v<digits>` followed by `/` or `_`, and the dataset root's own path. Case-folded."""
    text = str(text)
    low = text.lower()
    needles = [(n, n.lower()) for n in _FORBIDDEN_NEEDLES]
    try:
        root = str(de.data_root()).lower()
        if root and root != "/":
            needles.append((f"the dataset root {root}", root))
    except (OSError, ValueError, TypeError):
        pass
    hits = []

    def _report(label: str, positions: list) -> None:
        if not positions:
            return
        first = positions[0]
        ln = text.count("\n", 0, first) + 1
        a, b = text.rfind("\n", 0, first) + 1, text.find("\n", first)
        line = text[a:(b if b >= 0 else len(text))].strip()
        hits.append(f"{label}: {len(positions)} occurrence(s), first at line {ln}: {line[:120]!r}")

    for label, needle in needles:
        pos, i = [], low.find(needle)
        while i >= 0:
            pos.append(i)
            i = low.find(needle, i + 1)
        _report(label, pos)
    _report("skills_v<K>/ or skills_v<K>_ (a Mac dir)", [m.start() for m in _SKILLS_V_DIR_RE.finditer(text)])
    return hits


# ── the manifest ──────────────────────────────────────────────────────────────────────────────────
def _files_under(d: Path, what: str) -> list:
    d = Path(d)
    if d.is_symlink() or not d.is_dir():
        raise ValueError(f"REFUSING: {what} {d} is not a plain directory")
    out = []
    for f in sorted(d.rglob("*")):
        if f.is_symlink():
            raise ValueError(f"REFUSING: symlink in {what}: {f}")
        if f.is_file():
            out.append(f)
    return out


def _row_of(vm_path: str, role: str, epoch: int, *, source: str | None = None, data: bytes | None = None) -> dict:
    try:
        r = de._row(vm_path, role, source, data=data)
    except OSError as e:
        raise ValueError(f"REFUSING manifest row {vm_path}: source {source} unreadable ({type(e).__name__})")
    r["epoch"] = int(epoch)
    return r


def _dir_rows(pm: PathMap, epoch: int) -> list:
    what = f"{pm.role} dir {Path(pm.mac).name}"
    return [_row_of(f"{pm.vm}/{f.relative_to(pm.mac).as_posix()}", pm.role, epoch, source=str(f))
            for f in _files_under(Path(pm.mac), what)]


def _file_row(pm: PathMap, epoch: int) -> dict:
    p = Path(pm.mac)
    if p.is_symlink() or not p.is_file():
        raise ValueError(f"REFUSING: {pm.role} file {p} is not a regular file")
    return _row_of(pm.vm, pm.role, epoch, source=str(p))


def _real(p) -> str:
    return os.path.realpath(str(p))


def build_stateful_manifest(*, run_dir, epoch: int, env_text: str, hooks_text: str, split, uploaded_before: list) -> tuple:
    """(delta rows, generated {vm_path: bytes}) for THIS epoch of a stateful session: exactly the path_view entries
    with .epoch == epoch, plus the hook / env / policy rows at epoch 1 as daytona_evolver.build_upload_manifest
    builds them. Each row is daytona_evolver._row (vm_path, role, source, sha256, bytes) + "epoch".

    Refuses (ValueError): a dir that is not THIS run's; a symlink source anywhere; a container the `uploaded_before`
    rows say does not hold an earlier epoch's entry; a vm_path already in `uploaded_before` (extract --incremental is
    create-only) -- except under /evo/staging, which `epoch-begin` clears before every upload; any vm_path under
    /Users; and, for the skills / seed / hook / env / policy rows only, everything daytona_evolver
    ._refuse_forbidden_sources refuses. The solver's own material (traces, workspaces, scores) and the adapter's
    reference/ are uploaded as they are: no name or hash scan (the reference folder is the adapter's own copy)."""
    rd = Path(_real(run_dir))
    N = _epoch_index(epoch, floor=1)
    if not isinstance(uploaded_before, list) or any(not isinstance(r, dict) or not isinstance(r.get("vm_path"), str)
                                                    for r in uploaded_before):
        raise ValueError("uploaded_before must be a list of manifest rows ({vm_path, ...})")
    view = path_view(rd, N)
    delta = [pm for pm in view if pm.epoch == N]
    before_paths = {r["vm_path"] for r in uploaded_before}
    for pm in view:
        # an EMPTY dir (e.g. an empty skills_vK: no skills) uploads no rows, so nothing is under it
        empty = pm.role in DIR_ROLES and Path(pm.mac).is_dir() and not any(f.is_file() for f in Path(pm.mac).rglob("*"))
        if pm.epoch < N and not empty and not any(_under(p, pm.vm) for p in before_paths):
            raise ValueError(f"REFUSING: the container does not hold {pm.vm} ({pm.role}, uploaded at epoch "
                             f"{pm.epoch}); uploaded_before lists no row under it")
    rows, gen = [], {}
    for pm in delta:
        if pm.role in FILE_ROLES:
            rows.append(_file_row(pm, N))
        elif pm.role in DIR_ROLES:
            rows += _dir_rows(pm, N)
        else:                                             # pragma: no cover -- the view has no other role
            raise ValueError(f"unknown view role {pm.role!r}")
    if N == 1:
        rows.append(_row_of(f"{de.VM_HOOKS}/evolver_sync_agent_hook.py", ROLE_HOOK, 1, source=str(de.SYNC_HOOK)))
        gen[f"{de.VM_HOOKS}/settings.json"] = hooks_text.encode()
        rows.append(_row_of(f"{de.VM_HOOKS}/settings.json", ROLE_HOOK, 1, data=gen[f"{de.VM_HOOKS}/settings.json"]))
        gen[de.VM_ENV_FILE] = env_text.encode()
        rows.append(_row_of(de.VM_ENV_FILE, ROLE_ENV, 1, data=gen[de.VM_ENV_FILE]))
        gen[de.VM_POLICY] = de.policy_settings_text().encode()
        rows.append(_row_of(de.VM_POLICY, ROLE_POLICY, 1, data=gen[de.VM_POLICY]))
    seen = set()
    for r in rows:
        vp = r["vm_path"]
        if not vp.startswith("/") or posixpath.normpath(vp) != vp:
            raise ValueError(f"REFUSING manifest row {vp!r}: not a normalised absolute container path")
        if _under(vp, "/Users"):
            raise ValueError(f"REFUSING manifest row {vp}: nothing goes under /Users in a stateful session")
        if vp in seen:
            raise ValueError(f"REFUSING manifest row {vp}: listed twice in this epoch's delta")
        seen.add(vp)
        if vp in before_paths and not _under(vp, VM_STAGING):
            raise ValueError(f"REFUSING manifest row {vp}: already in the container (extract --incremental is "
                             f"create-only; only /evo/staging is cleared between epochs)")
    de._refuse_forbidden_sources([r for r in rows if r.get("role") not in UNSCANNED_ROLES], split)
    return rows, gen


def after_hashes_vm(after_listing, view) -> dict:
    """{vm root: daytona_evolver.dir_sha256_from_rows(after_listing, vm root)} for every mapped dir entry of the
    view (informational: the loop judges the Mac bytes, not the container's)."""
    return {pm.vm: de.dir_sha256_from_rows(after_listing, pm.vm) for pm in view if pm.role in DIR_ROLES}


def _load_json(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def _benchmark(run_dir: Path):
    """SPEC.json's spec.benchmark, or None when it cannot be read."""
    try:
        raw = json.loads((Path(run_dir) / "SPEC.json").read_text())
    except (OSError, ValueError):
        return None
    sp = raw.get("spec", raw) if isinstance(raw, dict) else None
    return sp.get("benchmark") if isinstance(sp, dict) else None


def check_stateful_upload(*, run_dir, epoch: int, split, state: dict, receipt: dict, listing: dict):
    """The #213 replacement for a stateful session, a Check: the container held exactly this run's material,
    cumulatively. Re-derives from the Mac bytes NOW:
    - the run's RUNTIME.json says stateful + template;
    - `receipt["upload_manifest"]` == the delta build_stateful_manifest builds now (every source unchanged);
      `receipt["path_view"]` (when recorded) == the run's view; roles in STATEFUL_ROLES;
    - `receipt["upload_manifest_cumulative"]` == `state["uploaded"]` == e1..eN receipts' deltas concatenated;
    - per view entry (cumulative rows): every dir's rows == the Mac dir now, file for file; scores.json at its sha;
      this epoch's seed == skills_v{N-1} at /evo/staging (none at epoch 1); reference/ whole, mirrored, unscanned;
      hook / env / policy once, at epoch 1; NO prompt or system-prompt row (they travel on the command line);
    - the skills / seed / hook / env / policy rows: source unchanged, no forbidden name / sha, nothing under /Users;
    - the cumulative listing: no symlink; regular files outside EVOLVER_WRITABLE == the cumulative manifest exactly;
      the seed rows present at their sha; /Users empty before the first upload."""
    from src.orchestrator.invariants import _fail, _ok, _skip
    n = "evolver container held only this run's material (stateful session)"
    rd = Path(_real(run_dir))
    N = _epoch_index(epoch, floor=1)
    rc = receipt
    if not rc or rc.get("containment") != "daytona":
        return _skip(n, "no daytona receipt (containment is not 'daytona')", "#213")
    rows = rc.get("upload_manifest") or []
    if not rows:
        return _skip(n, "receipt carries an empty upload_manifest", "#213")
    try:
        tm = de.evolver_template_mode(rd)
        ses = evolver_session(rd)
    except ValueError as e:
        return _fail(n, f"the run's RUNTIME.json cannot be read: {e}", "#213")
    if not isinstance(state, dict):
        return _fail(n, "no state record (stateful_layout.load_state) to judge the cumulative manifest against", "#213")
    probs = []
    if ses != STATEFUL:
        probs.append(f"RUNTIME.json {SESSION_KEY}={ses!r}: this check judges a stateful session only")
    if not tm:
        probs.append("a stateful session requires prompt_mode=template (the loop renders the message)")
    try:
        view = path_view(rd, N)
    except ValueError as e:
        return _fail(n, f"the run's path view cannot be built: {e}", "#213")
    if rc.get("path_view") is not None and rc["path_view"] != [pm.as_dict() for pm in view]:
        probs.append("the receipt's path_view is not the run's path view at this epoch")
    state_rows = [r for r in (state.get("uploaded") or []) if isinstance(r, dict)]

    def _epoch_of(r) -> int:
        try:
            return int(r.get("epoch", 0))
        except (TypeError, ValueError):
            return 0

    def _key(r) -> tuple:
        return (r.get("vm_path"), r.get("role"), r.get("source"), r.get("sha256"), r.get("bytes"))
    # (1) the delta re-derived from the Mac bytes now
    venv = (rc.get("daytona") or {}).get("vm_env") or {}
    try:
        now_rows, _gen = build_stateful_manifest(
            run_dir=rd, epoch=N, env_text=de.render_env_file(venv), hooks_text=de.hook_settings_text(), split=split,
            uploaded_before=[r for r in state_rows if _epoch_of(r) < N])
    except (ValueError, OSError) as e:
        probs.append(f"the delta does not rebuild from the Mac bytes now: {str(e)[:300]}")
        now_rows = None
    if now_rows is not None:
        want, got = {_key(r) for r in now_rows}, {_key(r) for r in rows}
        missing, extra = sorted(want - got, key=str), sorted(got - want, key=str)
        if missing or extra or len(rows) != len(now_rows):
            probs.append(f"upload_manifest != the delta rebuilt now: {len(missing)} row(s) missing or changed "
                         f"(e.g. {[m[0] for m in missing[:2]]}), {len(extra)} not in the rebuild "
                         f"(e.g. {[m[0] for m in extra[:2]]})")
    if [r for r in rows if _epoch_of(r) not in (0, N)]:
        probs.append(f"upload_manifest rows carry another epoch than {N}")
    for role in sorted({r.get("role") for r in rows} - STATEFUL_ROLES, key=str):
        probs.append(f"unknown manifest role {role!r}")
    # (2) the cumulative manifest: the receipt's == the state's == the receipts' deltas concatenated
    cum = rc.get("upload_manifest_cumulative")
    receipts = {N: rc}
    unreadable = []
    for k in range(1, N):
        rk = _load_json(rd / RECEIPT_FILE.format(epoch=k))
        if not isinstance(rk, dict):
            unreadable.append(k)
        else:
            receipts[k] = rk
    if unreadable:
        probs.append(f"earlier receipt(s) unreadable: {[RECEIPT_FILE.format(epoch=k) for k in unreadable]}")
    if not isinstance(cum, list) or not cum:
        probs.append("receipt carries no upload_manifest_cumulative")
        cum_rows = [r for r in state_rows]
    else:
        cum_rows = [r for r in cum if isinstance(r, dict)]
        if [(_key(r), _epoch_of(r)) for r in cum_rows] != [(_key(r), _epoch_of(r)) for r in state_rows]:
            probs.append(f"upload_manifest_cumulative ({len(cum_rows)} rows) != the state record's uploaded rows "
                         f"({len(state_rows)})")
        if not unreadable:
            joined = []
            for k in range(1, N + 1):
                joined += [(k, r) for r in (receipts[k].get("upload_manifest") or []) if isinstance(r, dict)]
            if [_key(r) for r in cum_rows] != [_key(r) for _k, r in joined]:
                probs.append(f"upload_manifest_cumulative != the e1..e{N} receipts' upload_manifest concatenated")
            elif [1 for (k, _r), cr in zip(joined, cum_rows) if _epoch_of(cr) not in (0, k)]:
                probs.append("a cumulative row carries another epoch than the receipt that uploaded it")
    by_role: dict = {}
    for r in cum_rows:
        by_role.setdefault(r.get("role"), []).append(r)
    for role in sorted(set(by_role) - STATEFUL_ROLES, key=str):
        probs.append(f"unknown cumulative manifest role {role!r}")
    # (3) every cumulative row: an absolute container path outside /Users; the scanned roles: source unchanged, no
    #     forbidden name / sha (the solver's own material and the adapter's reference/ are not scanned)
    # The SSB dataset tree's forbidden byte streams. A run of another benchmark on a host with no SSB tree
    # configured (de.data_root_or_none() is None) has none to hold: the SSB loop exports INFRA data.root.
    root = de.data_root_or_none()
    if root is None and _benchmark(rd) != "ssbv2":
        bad_sha = {}
    else:
        try:
            root = de.data_root()
            bad_sha = de.forbidden_shas(de.dataset_index(root), de.split_roles(split), root)
        except (OSError, ValueError, TypeError, KeyError) as e:
            probs.append(f"the forbidden dataset hashes cannot be derived ({type(e).__name__}: {str(e)[:100]})")
            bad_sha = {}
    split_real = _real(split) if split else None
    for r in cum_rows:
        vp, src = str(r.get("vm_path") or ""), r.get("source")
        if not vp.startswith("/") or _under(vp, "/Users"):
            probs.append(f"{vp!r}: not an absolute container path outside /Users")
        if src and src != "generated":
            if not os.path.isfile(src) or os.path.islink(src):
                probs.append(f"{vp}: source {src} missing (or a symlink) now")
            elif de.sha256_file(src) != r.get("sha256"):
                probs.append(f"{vp}: source bytes changed since upload")
        if r.get("role") in UNSCANNED_ROLES:
            continue
        why = de.validation_name(vp) or (de.validation_source(src) if src and src != "generated" else "")
        if why:
            probs.append(f"{vp}: {why}")
        if r.get("sha256") in bad_sha:
            probs.append(f"{vp}: sha equals a forbidden file ({bad_sha[r['sha256']]})")
        s = (vp + " " + (src or "")).lower()
        if "golden" in s or "dataset.json" in s or (split_real and src and src != "generated" and _real(src) == split_real):
            probs.append(f"{vp}: forbidden name")
    try:
        de._refuse_forbidden_sources([r for r in cum_rows if r.get("role") not in UNSCANNED_ROLES], split)
    except ValueError as e:
        probs.append(str(e)[:300])
    # (4) per view entry: the rows == the Mac dir / file now
    for pm in view:
        mine = [r for r in by_role.get(pm.role, []) if _under(str(r.get("vm_path")), pm.vm)]
        if pm.role == ROLE_SEED:
            continue                                      # judged on THIS epoch's delta below
        p = Path(pm.mac)
        if pm.role in FILE_ROLES:
            if len(mine) != 1 or not p.is_file() or mine[0].get("sha256") != de.sha256_file(p):
                probs.append(f"{pm.vm}: not exactly {p.name} at its current sha")
            continue
        want = {f"{pm.vm}/{f.relative_to(p).as_posix()}": de.sha256_file(f) for f in p.rglob("*") if f.is_file()} \
            if p.is_dir() else {}
        got = {r["vm_path"]: r.get("sha256") for r in mine}
        if (not want and pm.role != ROLE_SKILLS) or got != want:        # an empty skill set is a set
            probs.append(f"{pm.vm} rows != {Path(pm.mac).relative_to(rd)} file-for-file")
        if pm.role == ROLE_REFERENCE:
            for r in mine:                                # each row mirrors ITS file (no scan of the content)
                vp = str(r.get("vm_path"))
                if _real(r.get("source") or "/nonexistent") != _real(p / vp[len(pm.vm) + 1:]):
                    probs.append(f"{vp}: reference row not mirrored from {REFERENCE_DIR_NAME}/{vp[len(pm.vm) + 1:]}")
                    break
        elif pm.role in EVO_ROLES:
            before = (receipts.get(pm.epoch) or {}).get("evidence_sha256_before") or {}
            evo_dir = str(Path(pm.mac).parent)
            if pm.epoch in receipts and before and evo_dir not in before:
                probs.append(f"{Path(evo_dir).name}: e{pm.epoch}'s receipt records no pre-dispatch hash of it")
    for role, vmroot in ((ROLE_TRACES, VM_TRACES), (ROLE_SOLVER_WS, VM_SOLVER_WS), (ROLE_SCORES, VM_SCORES),
                         (ROLE_SKILLS, VM_SKILLS), (ROLE_REFERENCE, VM_REFERENCE)):
        mapped = [pm.vm for pm in view if pm.role == role]
        stray = [r.get("vm_path") for r in by_role.get(role, []) if not any(_under(str(r.get("vm_path")), v) for v in mapped)]
        if stray:
            probs.append(f"{len(stray)} {role} row(s) outside the mapped {vmroot} entries: {stray[:2]}")
    seed = [r for r in rows if r.get("role") == ROLE_SEED]
    if N == 1:
        if seed:
            probs.append(f"{len(seed)} seed row(s) at epoch 1")
    else:
        pl = rd / f"skills_v{N - 1}"
        want = {f"{VM_STAGING}/{f.relative_to(pl).as_posix()}": de.sha256_file(f) for f in pl.rglob("*") if f.is_file()} \
            if pl.is_dir() else {}
        if not pl.is_dir() or {r["vm_path"]: r.get("sha256") for r in seed} != want:   # empty set: no seed rows
            probs.append(f"seed rows != skills_v{N - 1} file-for-file at {VM_STAGING}")
    # (5) no prompt / system row; hook / env / policy once, at epoch 1
    if [r for r in cum_rows if r.get("role") in ("prompt", "system")]:
        probs.append("a prompt / system-prompt row is in the manifest: the message and the system prompt travel on "
                     "the launch command line, never as files the evolver sees")
    once = [r for r in rows if r.get("role") in (ROLE_HOOK, ROLE_ENV, ROLE_POLICY)]
    if (N == 1) != bool(once) or (N == 1 and len(once) != 4):
        probs.append(f"hook / env / policy rows travel once, at epoch 1 ({len(once)} such row(s) in epoch {N}'s delta)")
    hk = {r.get("vm_path"): r for r in by_role.get(ROLE_HOOK, [])}
    if set(hk) != {f"{de.VM_HOOKS}/evolver_sync_agent_hook.py", f"{de.VM_HOOKS}/settings.json"}:
        probs.append("hook rows are not exactly the sync hook + its settings")
    else:
        if hk[f"{de.VM_HOOKS}/settings.json"].get("sha256") != de.sha256_bytes(de.hook_settings_text().encode()):
            probs.append("hook settings differ from the rendered #212-only settings")
        if hk[f"{de.VM_HOOKS}/evolver_sync_agent_hook.py"].get("source") != str(de.SYNC_HOOK):
            probs.append("hook script row does not come from scripts/evolver_sync_agent_hook.py")
    envr = by_role.get(ROLE_ENV, [])
    if [r.get("vm_path") for r in envr] != [de.VM_ENV_FILE]:
        probs.append("env rows are not exactly the one env file")
    elif not venv or envr[0].get("sha256") != de.sha256_bytes(de.render_env_file(venv).encode()):
        probs.append("env file does not re-render from the receipt's recorded vm_env")
    elif [k for k in venv if re.search(r"TOKEN|KEY|SECRET|ANTHROPIC_", k)]:
        probs.append("recorded vm_env carries secret-shaped keys")
    pol = by_role.get(ROLE_POLICY, [])
    if [r.get("vm_path") for r in pol] != [de.VM_POLICY] or pol[0].get("sha256") != de.sha256_bytes(de.policy_settings_text().encode()):
        probs.append("policy rows are not exactly the rendered managed-settings.json")
    # (6) the cumulative listing
    lst = listing
    if not lst or "after_upload" not in lst:
        probs.append(f"no container listing ({LISTING_FILE.format(epoch=N)} after_upload)")
    else:
        if (lst.get("pre_upload") or {}).get("users_exists") is not False:
            probs.append("/Users existed (or was not probed) in the container before the first upload")
        after = [x for x in lst["after_upload"] if isinstance(x, (list, tuple)) and len(x) == 4]
        links = [p for p, t, _h, _b in after if t == "L"]
        if links:
            probs.append(f"{len(links)} symlink(s) in the container after upload: {links[:2]}")
        files = {p: h for p, t, h, _b in after if t == "F"}

        def _writable(p) -> bool:
            return any(_under(str(p), w) for w in EVOLVER_WRITABLE)
        man = {}
        for r in cum_rows:                                # the LAST row of a path wins (the re-seeded staging)
            man[str(r.get("vm_path"))] = r.get("sha256")
        man_ro = {p: h for p, h in man.items() if not _writable(p)}
        files_ro = {p: h for p, h in files.items() if not _writable(p)}
        extra = sorted(set(files_ro) - set(man_ro))
        missing = sorted(set(man_ro) - set(files_ro))
        diff = sorted(p for p in set(files_ro) & set(man_ro) if files_ro[p] != man_ro[p])
        if extra:
            probs.append(f"{len(extra)} file(s) in the container not in the manifest: {extra[:3]}")
        if missing:
            probs.append(f"{len(missing)} manifest file(s) absent in the container: {missing[:3]}")
        if diff:
            probs.append(f"{len(diff)} file(s) in the container differ from the manifest: {diff[:3]}")
        man_rw = {str(r.get("vm_path")): r.get("sha256") for r in seed}     # this epoch's seed only
        absent = sorted(p for p in man_rw if p not in files)
        changed = sorted(p for p in man_rw if p in files and files[p] != man_rw[p])
        if absent or changed:
            probs.append(f"seed rows under {VM_STAGING} not at their sha: {len(absent)} absent {absent[:2]}, "
                         f"{len(changed)} changed {changed[:2]}")
    if probs:
        return _fail(n, "; ".join(probs[:6]) + (f" (+{len(probs)-6} more)" if len(probs) > 6 else ""),
                     "#213", problems=probs)
    counts = {}
    for r in rows:
        counts[r.get("role")] = counts.get(r.get("role"), 0) + 1
    counts = dict(sorted(counts.items()))
    n_traces = sum(1 for r in by_role.get(ROLE_TRACES, []))
    return _ok(n, f"epoch {N}: {len(rows)} delta file(s) {counts}, every one re-derived from this run's own bytes "
                  f"now; cumulative {len(cum_rows)} file(s) == the state record == the e1..e{N} receipts; "
                  f"{n_traces} raw session log(s) + their workspaces and scores as the leg stores wrote them; "
                  f"reference/ mirrored whole (unscanned); no prompt file in the container; the listing outside "
                  f"{list(EVOLVER_WRITABLE)} equals the cumulative manifest exactly; nothing under /Users",
               "#213", counts=counts, cumulative=len(cum_rows), template_mode=tm)
