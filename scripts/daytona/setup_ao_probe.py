#!/usr/bin/env python3
"""Run the benchmark adapter's ONLY turn -- the setup message -- on a Daytona sandbox built from the solver image.

Same containment as the epoch AO (tau3_evolver_daytona.AORun): unprivileged user `evo`, the run's Claude Code
credential on fd 3, egress api.anthropic.com only, VM listing == upload manifest, token scan, sandbox deleted in
`finally`.

What the adapter gets, and nothing else:
  - the setup message (the user's brief pasted verbatim), as /evo/work/MESSAGE.md;
  - --data-view full (the default: what a real user hands to the adapter): the data root EXACTLY as the user hands it,
    uploaded at its own path: every folder (Visualization too) and every file, reference workbooks of every split set
    included. Checked to be a complete byte-exact mirror.
    Plus every other file the brief names, at its path: the split file and the run's RUNTIME.json ao.brief_files
    (repo-relative; absent = BRIEF_FILES, the grader source src/agent_profiles/ssbv2_agent/scorer.py and the image's
    Dockerfile; the tau3 brief names no repo file, [], its view is /data/<data root> + /data/split.json).
    --data-view train: the TRAIN-ONLY view (each category's dataset.json reduced to its training records,
    plus those records' input workbooks; no reference workbook, no val/test record, no other folder; refused if
    any uploaded file equals a reference workbook or a non-train input). Kept for comparison.
  - <run>/setup/plugin_api.py and benchmark_plugin_skeleton.py (written here when absent: ensure_given_files), and the
    user's grader, --grader FILE, as <run>/setup/benchmark_grader.py (the user writes the grader and hands it to the
    adapter, which neither writes nor tests it; the freeze pins it and the loop runs it).
PATHS ON THE ADAPTER'S MACHINE (neutral paths, never the host's): what the user hands over is under /data by its own
name (/data/<data root name>/, /data/<split file name>, /data/<repository name>/<the brief's repo files>: vm_layout),
the outputs are /evo/setup/ and /evo/reference/ (installed into <run>/setup/ and <run>/reference/ on the host). The
user instruction and the guidance must name these paths: a host path in either is refused.
It may write only into /evo/setup/ and /evo/reference/ (both root-owned 1777 on the VM: it can add files, not
change the given ones). The plug-in it writes lists every task WITH its set and carries no grader; <run>/reference/ is
everything the user handed over that the evolver may use (the loop copies it to the evolver's machine unchanged, no
scan). The plug-in (OUT_FILES) and the reference dir are installed on the host only if claude exits 0 and the local
copies are absent. No clean grader.

  setup_ao_probe.py --run-dir outputs/<run> --message <USER_INSTRUCTION.md> --grader <the user's grader .py> [--timeout-s 3600]

--split defaults to the split the run declares (SPEC.json split_path), else ssb_leg_daytona_loop.PROJECT_SPLIT. The
data root and the repaired-file overlay come from the run's INFRA.json block "data" by the leg driver's own rule
(ssb_leg_daytona_loop.data_overlay_from_infra); without INFRA.json, the environment's SSB root
(daytona_evolver.data_root_or_none: EVOSKILL_SSBV2_DATA or SSBV2_ROOT) and the bare EVOSKILL_SSBV2_DATA_OVERLAY. With
an overlay the view's train inputs are the REPAIRED bytes (what the solver gets; paths unchanged), and the receipt
records the overlay. A turn the API rejected for a usage limit / an HTTP 429 waits for the reported reset and runs
again (tau3_evolver_daytona.run_resuming).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import tau3_evolver_daytona as T  # noqa: E402
import ssb_leg_daytona_loop as L  # noqa: E402  (READ-ONLY reuse: INFRA.json, overlay rule, declared split)
from src.orchestrator import data_overlay  # noqa: E402  (repaired files: bytes, path unchanged)
from src.orchestrator import plugin_runner as PR  # noqa: E402  (the /data view: material_layout)

de, dle, REPO, Refuse, log = T.de, T.dle, T.REPO, T.Refuse, T.log

CATEGORIES = ("Debugging", "Financial_Model", "Template")
# The setup agent is the BENCHMARK ADAPTER and writes ONE file, the plug-in (the stateful evolver needs no template
# and no plan from it).
OUT_FILES = ("benchmark_plugin.py",)
GIVEN_FILES = ("plugin_api.py", "benchmark_plugin_skeleton.py", "benchmark_grader.py")   # the last: the user's grader
REFERENCE_DIR_NAME = "reference"
DATA_VIEWS = ("full", "train")
# Neutral paths: the adapter's machine holds what the user handed over under /data and its two output dirs under
# /evo -- never the host's paths. The user instruction and the guidance name these paths. The /data view is
# plugin_runner's (material_layout), shared with the grader's setup.
VM_DATA = PR.MATERIAL_ROOT
VM_SETUP = f"{de.VM_ROOT}/setup"
VM_REFERENCE = f"{de.VM_ROOT}/{REFERENCE_DIR_NAME}"
SKELETON = '''"""Skeleton of a benchmark plug-in. Copy it to benchmark_plugin.py and fill it in."""
from __future__ import annotations

from pathlib import Path

from plugin_api import Task


def list_tasks(root: Path) -> list[Task]:
    """One Task per task of the benchmark under `root`: id, category, instruction, inputs (absolute paths),
    answer_files (absolute paths), grading (the grader's per-task arguments other than files, as strings) and
    split ("train", "validation", "test", or None for a task the run does not use)."""
    raise NotImplementedError
'''


def ensure_given_files(setup_dir: Path, grader: Path) -> None:
    """The given files: plugin_api.py (the canonical copy, refused if a different one is there), the skeleton, and the
    user's grader (a copy of `grader`, refused if a different one is there)."""
    setup_dir.mkdir(parents=True, exist_ok=True)
    grader = Path(grader)
    if not grader.is_file():
        raise SystemExit(f"REFUSING: no grader file at {grader}")
    if not (setup_dir / GIVEN_FILES[2]).exists():
        shutil.copyfile(grader, setup_dir / GIVEN_FILES[2])
    elif (setup_dir / GIVEN_FILES[2]).read_bytes() != grader.read_bytes():
        raise SystemExit(f"REFUSING: {setup_dir / GIVEN_FILES[2]} differs from {grader}")
    api = REPO / "src" / "orchestrator" / "plugin_api.py"
    if not (setup_dir / GIVEN_FILES[0]).exists():
        shutil.copyfile(api, setup_dir / GIVEN_FILES[0])
    elif (setup_dir / GIVEN_FILES[0]).read_bytes() != api.read_bytes():
        raise SystemExit(f"REFUSING: {setup_dir / GIVEN_FILES[0]} differs from {api}")
    if not (setup_dir / GIVEN_FILES[1]).exists():
        (setup_dir / GIVEN_FILES[1]).write_text(SKELETON)


def tree_sha256(root: Path) -> tuple:
    """(sha256 over (relative path, file sha) pairs, file count) of a directory tree; symlinks refused."""
    h, n = hashlib.sha256(), 0
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise SystemExit(f"REFUSING: symlink in {root}: {p}")
        if p.is_file():
            h.update(p.relative_to(root).as_posix().encode() + b"\0" + de.sha256_file(p).encode() + b"\n")
            n += 1
    return h.hexdigest(), n
# The files the brief names besides the data root (items 1, 3, 4 of SETUP_MESSAGE): uploaded at their own paths in
# the full view. The split file is the run's own (resolved at run time). The default of RUNTIME ao.brief_files.
BRIEF_FILES = ("src/agent_profiles/ssbv2_agent/scorer.py", "scripts/daytona/Dockerfile.ssb_leg")


def brief_files_for(run_dir: Path) -> tuple:
    """The repo-relative files the run's brief names besides the data root and the split: RUNTIME.json
    ao.brief_files ([] for the tau3 brief), absent = BRIEF_FILES (the SSB pair)."""
    p = Path(run_dir) / "RUNTIME.json"
    ao = (json.loads(p.read_text()) if p.is_file() else {}).get("ao")
    if not isinstance(ao, dict) or "brief_files" not in ao:
        return BRIEF_FILES
    bf = ao["brief_files"]
    if not isinstance(bf, list) or not all(isinstance(x, str) and x and not os.path.isabs(x) for x in bf):
        raise SystemExit(f"REFUSING: RUNTIME.json ao.brief_files must be a list of repo-relative paths, got {bf!r}")
    return tuple(bf)


def vm_layout(data_root: Path, split: Path, repo: Path = REPO) -> dict:
    """Host path -> its path on the adapter's machine: the data root, the split file and the repository the user
    instruction names files in, each under /data by its own name. Delegates to
    plugin_runner.material_layout, the one definition the grader's setup uses too."""
    try:
        return PR.material_layout(data_root, split, repo)
    except ValueError as e:
        raise SystemExit(f"REFUSING: {e} (on the adapter's machine)")


def vm_path(p, layout: dict) -> str:
    """p with its longest layout prefix replaced by that prefix's path on the adapter's machine."""
    s = str(p)
    for mac in sorted(layout, key=len, reverse=True):
        if s == mac or s.startswith(mac + "/"):
            return layout[mac] + s[len(mac):]
    raise SystemExit(f"REFUSING: {s} has no path on the adapter's machine")


def build_full_view(data_root: Path, dst: Path, overlay=None) -> dict:
    """The data root exactly as the user hands it: every file under it (every folder, reference workbooks of every
    split set included), the overlay's repaired bytes where it lists the file (paths unchanged). Symlinks refused."""
    n_files, n_bytes, folders = 0, 0, set()
    for src in sorted(data_root.rglob("*")):
        if src.is_symlink():
            raise SystemExit(f"symlink in the data root: {src}")
        if not src.is_file():
            continue
        rel = src.relative_to(data_root)
        out = dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(data_overlay.source(src, overlay) if overlay else src, out)
        n_files += 1
        n_bytes += out.stat().st_size
        folders.add(rel.parts[0] if len(rel.parts) > 1 else ".")
    return {"files": n_files, "bytes": n_bytes, "top_level": sorted(folders)}


def build_train_view(data_root: Path, split: Path, dst: Path, overlay=None) -> dict:
    """dataset.json per category reduced to TRAIN records, plus their input files (the overlay's repaired
    bytes where it lists the file). Returns a summary."""
    train = set(json.loads(split.read_text())["train"])
    n_rec, n_files = 0, 0
    for cat in CATEGORIES:
        recs = json.loads((data_root / cat / "dataset.json").read_text())
        keep = [r for r in recs if f"{cat}/{r['id']}" in train]
        (dst / cat).mkdir(parents=True, exist_ok=True)
        (dst / cat / "dataset.json").write_text(json.dumps(keep, indent=2, ensure_ascii=False))
        for r in keep:
            src = data_root / cat / r["spreadsheet_path"]
            if not src.is_file():
                raise SystemExit(f"train input missing: {src}")
            out = dst / cat / r["spreadsheet_path"]
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(data_overlay.source(src, overlay) if overlay else src, out)
            n_files += 1
        n_rec += len(keep)
    if n_rec != len(train):
        raise SystemExit(f"train view has {n_rec} records, split has {len(train)} train ids")
    return {"records": n_rec, "input_files": n_files}


class SetupRun(T.AORun):
    MIRROR_ROOTS = ("/Users", VM_DATA)          # /Users must stay absent on this machine: nothing mirrors the host

    def __init__(self, *, setup_dir: Path, view_local: Path, data_root: Path, reference_dir: Path, layout: dict,
                 overlay=None, data_view: str = "full", brief_files=(), guidance: Path | None = None, **kw):
        super().__init__(out=setup_dir / OUT_FILES[0], literals=[],
                         log_header="=== setup AO: model %s host daytona\n" % kw["model"], **kw)
        self.setup_dir, self.view_local = setup_dir, view_local
        self.data_root, self.reference_dir = data_root, reference_dir
        self.layout = layout
        self.vm_data_root = vm_path(data_root, layout)
        self.overlay = overlay
        if data_view not in DATA_VIEWS:
            raise ValueError(f"data_view {data_view!r} not in {DATA_VIEWS}")
        self.data_view = data_view
        self.brief_files = [Path(p) for p in brief_files] if data_view == "full" else []
        # The benchmark-agnostic AO guidance is kept apart from the benchmark-specific user brief: the guidance is
        # APPENDED to Claude Code's default system prompt (--append-system-prompt, the default prompt
        # is kept) and the user message is the brief alone. None = the older one-message form.
        self.guidance = Path(guidance) if guidance else None
        self.sysprompt_vm_file = f"{de.VM_WORK}/AO_GUIDANCE.md" if self.guidance else None

    def prepare(self) -> None:
        self.token, self.fd_env, self.credential = T.load_credential(self.cfg)
        if len(self.message.read_bytes()) > de.MAX_PROMPT_BYTES:
            raise Refuse(T.EXIT_USAGE, "setup message too large")
        if self.guidance is not None and len(self.guidance.read_bytes()) > de.MAX_PROMPT_BYTES:
            raise Refuse(T.EXIT_USAGE, "AO guidance too large")
        for f in OUT_FILES:
            if (self.setup_dir / f).exists():
                raise Refuse(T.EXIT_USAGE, f"{self.setup_dir / f} already exists")
        if self.reference_dir.exists():
            raise Refuse(T.EXIT_USAGE, f"{self.reference_dir} already exists")
        macs = (*self.layout, str(self.run_dir), str(self.setup_dir), str(self.reference_dir))
        for label, p in (("user instruction", self.message), ("guidance", self.guidance)):
            named = [m for m in macs if p is not None and m in p.read_text(encoding="utf-8")]
            if named:
                raise Refuse(T.EXIT_USAGE, f"the {label} names the host path {named[0]}; the adapter's machine has "
                                           f"no such path (the paths there: {self.layout}, {VM_SETUP}, {VM_REFERENCE})")
        ok_df, det_df, rec_df = de.dockerfile_check()
        if not ok_df:
            raise Refuse(T.EXIT_IDENTITY, f"image Dockerfile: {det_df}")
        self.env = de.vm_env(self.model, env=os.environ)
        env_text = de.render_env_file(self.env).encode()
        rows = [de._row(f"{de.VM_WORK}/MESSAGE.md", "message", str(self.message))]
        if self.guidance is not None:
            rows.append(de._row(self.sysprompt_vm_file, "system_prompt", str(self.guidance)))
        for f in sorted(self.view_local.rglob("*")):
            if f.is_symlink():
                raise Refuse(T.EXIT_MANIFEST, f"symlink in the data view: {f}")
            if f.is_file():
                rows.append(de._row(f"{self.vm_data_root}/{f.relative_to(self.view_local).as_posix()}", "dataview",
                                    str(f)))
        for p in self.brief_files:
            rows.append(de._row(vm_path(p, self.layout), "brief_file", str(p)))
        for name in GIVEN_FILES:
            rows.append(de._row(f"{VM_SETUP}/{name}", "setup_input", str(self.setup_dir / name)))
        rows.append(de._row(de.VM_ENV_FILE, "env", data=env_text))
        pol = de.policy_settings_text().encode()
        rows.append(de._row(de.VM_POLICY, "policy", data=pol))
        # The shared guard refuses every dataset.json, the dataset tree and the split file by path (right for the
        # evolver). The data view (and, in the full view, the brief's own files) are intended here, so they are
        # checked on their own terms instead. All other rows go through the shared guard.
        own_terms = ("dataview", "brief_file")
        de._refuse_forbidden_sources([r for r in rows if r["role"] not in own_terms], self.split)
        view_real = os.path.realpath(str(self.view_local))
        for r in rows:
            if r["role"] == "dataview" and not os.path.realpath(r["source"]).startswith(view_real + os.sep):
                raise Refuse(T.EXIT_MANIFEST, f"data-view row not from the view copy: {r['source']}")
        if self.data_view == "train":
            # every dataset.json holds training records only, every other view file is byte-equal to a training input
            train = set(json.loads(Path(self.split).read_text())["train"])
            train_inputs = set()
            for cat in CATEGORIES:
                for r in json.loads((self.data_root / cat / "dataset.json").read_text()):
                    if f"{cat}/{r['id']}" in train:
                        src = self.data_root / cat / r["spreadsheet_path"]
                        train_inputs.add(de.sha256_file(data_overlay.source(src, self.overlay) if self.overlay else src))
            for r in rows:
                if r["role"] != "dataview":
                    continue
                if os.path.basename(r["source"]) == "dataset.json":
                    cat = Path(r["source"]).parent.name
                    ids = {f"{cat}/{x['id']}" for x in json.loads(Path(r["source"]).read_text())}
                    if not ids <= train:
                        raise Refuse(T.EXIT_MANIFEST, f"{r['vm_path']} holds non-train records: {sorted(ids - train)[:3]}")
                elif r["sha256"] not in train_inputs:
                    raise Refuse(T.EXIT_MANIFEST, f"{r['vm_path']} is not a training task's input")
        else:
            # full view: a COMPLETE byte-exact mirror of the data root (overlay bytes where listed), each file at its
            # own path; the brief's files are exactly the split + BRIEF_FILES, byte-equal to their sources
            want = {}
            for src in sorted(self.data_root.rglob("*")):
                if src.is_file() and not src.is_symlink():
                    want[str(src.relative_to(self.data_root))] = de.sha256_file(
                        data_overlay.source(src, self.overlay) if self.overlay else src)
            got = {}
            for r in rows:
                if r["role"] != "dataview":
                    continue
                rel = os.path.relpath(os.path.realpath(r["source"]), view_real)
                if r["vm_path"] != f"{self.vm_data_root}/{Path(rel).as_posix()}":
                    raise Refuse(T.EXIT_MANIFEST, f"data-view row {r['vm_path']} is not at its own path")
                got[rel] = r["sha256"]
            if got != want:
                diff = sorted(set(want) ^ set(got)) or sorted(k for k in want if want[k] != got.get(k))
                raise Refuse(T.EXIT_MANIFEST, f"data view is not the data root, byte for byte: {diff[:3]}")
            need = sorted({os.path.realpath(str(self.split)), *(os.path.realpath(str(p)) for p in self.brief_files)})
            have = sorted(os.path.realpath(r["source"]) for r in rows if r["role"] == "brief_file")
            if have != need:
                raise Refuse(T.EXIT_MANIFEST, f"brief files {have} != {need}")
        # the forbidden byte streams of the dataset tree daytona_evolver reads (de.data_root()): only when this view IS
        # that tree (a tau3 data root is another benchmark's, and its full view is exempt anyway)
        ssb_root = de.data_root_or_none()
        same_tree = ssb_root is not None and os.path.realpath(str(self.data_root)) == os.path.realpath(str(ssb_root))
        bad = de.forbidden_shas(de.dataset_index(), de.split_roles(self.split)) if same_tree else {}
        guarded = [r for r in rows if not (self.data_view == "full" and r["role"] in own_terms)]
        clash = [r["vm_path"] for r in guarded if r["sha256"] in bad]
        if clash:
            raise Refuse(T.EXIT_MANIFEST, f"upload holds a reference workbook or non-train input: {clash[:3]}")
        heldout_refs = sorted({r["vm_path"] for r in rows if r["role"] == "dataview" and r["sha256"] in bad})
        # the full view hands over the user's own material as it is: the data root (every held-out task in it), the split
        # and the user's grader, which is not scanned for held-out ids (a grader may name a dropped task in a comment,
        # as the tau3 grader does). The message, the guidance and our two given files are scanned in every view.
        scanned = [g for g in GIVEN_FILES if not (self.data_view == "full" and g == GIVEN_FILES[2])]
        hits = de.heldout_id_hits([self.message, *(self.setup_dir / g for g in scanned),
                                   *([self.guidance] if self.guidance else [])], self.split)
        if hits:
            raise Refuse(T.EXIT_MANIFEST, f"held-out task id in the upload: {hits[:3]}")
        self.rows, self.gen = rows, {de.VM_ENV_FILE: env_text, de.VM_POLICY: pol}
        self.add_dirs = [VM_DATA, VM_SETUP, VM_REFERENCE]
        self.evidence_dirs, self.inputs = [], None
        counts = {}
        for r in rows:
            counts[r["role"]] = counts.get(r["role"], 0) + 1
        self.receipt = {
            "role": "setup-ao", "sandboxed": True, "containment": "daytona", "work_dir": de.VM_WORK,
            "upload_manifest": rows, "upload_counts": counts, "forbidden_hashes_checked": len(bad),
            "setup_dir": str(self.setup_dir), "data_root": str(self.data_root), "split": str(self.split),
            "data_view": self.data_view, "brief_files": [str(p) for p in self.brief_files],
            "vm_paths": {"layout": self.layout, "data_root": self.vm_data_root, "setup_dir": VM_SETUP,
                         "reference_dir": VM_REFERENCE},
            "system_prompt": ("Claude Code's default system prompt + the AO guidance appended "
                              f"(--append-system-prompt, from {self.sysprompt_vm_file})") if self.guidance
            else "Claude Code's default system prompt (nothing appended)",
            "guidance": {"path": str(self.guidance), "sha256": de.sha256_file(self.guidance)} if self.guidance else None,
            "heldout_or_nontrain_files_uploaded": len(heldout_refs),
            "data_overlay": data_overlay.record(self.overlay) if self.overlay else None,
            "daytona": {"driver": str(Path(__file__).relative_to(REPO)), "prefix": de.EVO_PREFIX,
                        "resources_requested": de.RESOURCES, "domain_allow_list": de.ALLOW_DOMAINS,
                        "dockerfile": rec_df, "cc_version": de.CC_VERSION, "session_id": self.sid_session,
                        "config": {k: v for k, v in self.cfg.items() if k != "oauth_token_file"},
                        "credential": self.credential, "oauth_token_file": self.credential["token_file"],
                        "vm_env_names": sorted(self.env), "vm_env": self.env},
            "written_utc": dle.now()}
        self.save_receipt()
        self.writable_dirs = [VM_SETUP, VM_REFERENCE]                  # both 1777 on the VM (the base class)
        self.staging = Path(de.VM_WORK)
        log(f"setup manifest: {len(rows)} files {counts}; data view {self.data_view}: "
            + (f"{len(heldout_refs)} of the upload's data files are val/test/non-train files (by design: the data "
               f"root as the user hands it)" if self.data_view == "full" else f"0 of {len(bad)} forbidden hashes")
            + "; receipt written")

    def claude_argv(self) -> list:
        argv = super().claude_argv()
        if self.guidance is None:
            return argv
        i = argv.index("--session-id") + 2
        return argv[:i] + ["--append-system-prompt", "__SYSPROMPT__"] + argv[i:]

    def readonly_roots(self) -> list:
        return [VM_DATA]

    def after_hashes(self, after) -> None:
        before = self.listing.get("after_upload") or []
        self.receipt["readonly_unchanged_on_vm"] = {
            d: de.dir_sha256_from_rows(before, d) == de.dir_sha256_from_rows(after, d)
            for d in self.readonly_roots()}

    def download_parts(self) -> list:
        return [VM_SETUP.lstrip("/"), VM_REFERENCE.lstrip("/"), de.VM_PROJECTS.lstrip("/"), de.VM_OUT.lstrip("/")]

    def install(self) -> None:
        ex = self.q / "extract"
        body = (self.q / self.stream_name).read_bytes()
        self.log_path.write_bytes(self.log_header.encode() + body)
        sess = ex / de.VM_PROJECTS.lstrip("/")
        if sess.is_dir():
            dst = self.q / "ao_session"
            shutil.copytree(sess, dst, dirs_exist_ok=True)
            self.receipt["daytona"]["session_transcript_dir"] = str(dst)
        src = ex / VM_SETUP.lstrip("/")
        present = sorted(p.name for p in src.iterdir()) if src.is_dir() else []
        extra = [n for n in present if n not in OUT_FILES + GIVEN_FILES and not n.startswith(".")]
        written = {f: (src / f).is_file() and not (src / f).is_symlink() for f in OUT_FILES}
        installed = []
        if getattr(self, "claude_rc", -1) == 0:
            for f, ok in written.items():
                if ok and not (self.setup_dir / f).exists():
                    shutil.copyfile(src / f, self.setup_dir / f)
                    installed.append(f)
        self.receipt["daytona"].update(claude_rc=getattr(self, "claude_rc", None), written=written,
                                       installed=installed, extra_files=extra,
                                       out_sha256={f: de.sha256_file(src / f) for f, ok in written.items() if ok})
        self.receipt["heldout_hits_in_outputs"] = de.heldout_id_hits(
            [self.setup_dir / f for f in installed], self.split) if installed else []
        # the reference dir the adapter filled: installed whole, no scan (the reference folder is the adapter's copy of
        # what the user handed over; the evolver gets it unchanged)
        ref_src = ex / VM_REFERENCE.lstrip("/")
        ref = {"path": str(self.reference_dir), "installed": False, "files": 0, "tree_sha256": None}
        if getattr(self, "claude_rc", -1) == 0 and ref_src.is_dir() and not self.reference_dir.exists():
            shutil.copytree(ref_src, self.reference_dir, symlinks=False)
            sha, n = tree_sha256(self.reference_dir)
            ref.update(installed=True, files=n, tree_sha256=sha)
        self.receipt["reference_dir"] = ref
        self.save_receipt()
        log(f"setup AO installed: {installed}; extra files written: {extra}; reference dir: "
            f"{ref['files']} file(s), installed={ref['installed']}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--message", required=True)
    ap.add_argument("--model", default="claude-sonnet-5-5")    # the adapter's model
    ap.add_argument("--timeout-s", type=int, default=3600)
    ap.add_argument("--guidance", default=None,
                    help="the adapter guidance TEMPLATE ({setup_dir}, {reference_dir}); rendered for this run and "
                         "appended to Claude Code's default system prompt; --message is then the user instruction")
    ap.add_argument("--data-view", choices=DATA_VIEWS, default="full",
                    help="full (default): the data root as the user hands it + every file the brief "
                         "names; train: the train-only view")
    ap.add_argument("--grader", required=True, help="the user's grader: a .py defining grade(task, snapshot), given to the "
                    "adapter as setup/benchmark_grader.py")
    ap.add_argument("--split", default=None, help="default: the run's SPEC.json split_path, else "
                    f"{L.PROJECT_SPLIT[0]}")
    return ap


def make_run(a, cache_root: Path = REPO / ".cache/daytona_ao") -> SetupRun:
    """Everything main() does before the sandbox: the data view under <cache_root>/<run>/setup_<ts>/, the guidance
    render, the given files and the SetupRun (its prepare() is the offline manifest check; run() creates the VM)."""
    run_dir = Path(a.run_dir).resolve()
    try:
        split = Path(a.split).resolve() if a.split else L.default_split(run_dir)
        inf, _rec = L.load_infra(run_dir)
        overlay = L.data_overlay_from_infra(inf, dict(os.environ), set_process_env=False) if inf is not None \
            else (os.environ.get(data_overlay.ENV) or None)
    except L.Refused as e:
        raise SystemExit(f"REFUSING: {e}")
    data_root = Path(inf["data"]["root"]) if inf is not None and inf["data"].get("root") else de.data_root_or_none()
    if data_root is None:
        raise SystemExit("REFUSING: no data root: the run's INFRA.json data.root, else EVOSKILL_SSBV2_DATA / SSBV2_ROOT")
    ts = time.strftime("%Y%m%dT%H%M%S")
    q = Path(cache_root) / run_dir.name / f"setup_{ts}"
    if a.data_view == "full":
        view = q / "full_view"
        summ = build_full_view(data_root, view, overlay)
        brief = [split, *(REPO / b for b in brief_files_for(run_dir))]
    else:
        view = q / "train_view"
        summ = build_train_view(data_root, split, view, overlay)
        brief = []
    log(f"{a.data_view} data view: {summ} at {view} (split {split}; overlay {overlay or 'none'})")
    guidance = None
    if a.guidance:
        g = Path(a.guidance).read_text(encoding="utf-8")
        for k, v in (("{setup_dir}", VM_SETUP), ("{reference_dir}", VM_REFERENCE)):     # the adapter machine's paths
            if k not in g:
                raise SystemExit(f"REFUSING: guidance template lacks {k}")
            g = g.replace(k, v)
        if "{" in g or "}" in g:
            raise SystemExit("REFUSING: guidance has an unfilled brace after rendering")
        guidance = q / "AO_GUIDANCE.md"
        guidance.write_text(g, encoding="utf-8")
        log(f"AO guidance rendered: {guidance} ({len(g.encode())} B) -> appended system prompt")
    ensure_given_files(run_dir / "setup", Path(a.grader).resolve())
    layout = vm_layout(data_root, split)
    log(f"paths on the adapter's machine: {layout}; outputs {VM_SETUP}, {VM_REFERENCE}")
    run = SetupRun(setup_dir=run_dir / "setup", view_local=view, data_root=data_root, overlay=overlay,
                   data_view=a.data_view, brief_files=brief, guidance=guidance, layout=layout,
                   reference_dir=run_dir / REFERENCE_DIR_NAME, message=Path(a.message).resolve(), epoch=0,
                   prompt=Path(a.message).resolve(), digests=view, staging=run_dir / "setup", run_dir=run_dir,
                   prior=[], model=a.model, oracle=run_dir / REFERENCE_DIR_NAME, cfg=T._cfg(),
                   timeout_s=a.timeout_s, receipt_path=run_dir / "setup_probe" / "SETUP_AO_DAYTONA.json",
                   trace_path=q / "unused_trace.json",
                   listing_path=run_dir / "setup_probe" / "SETUP_AO_VM_LISTING.json",
                   log_path=run_dir / "setup_probe" / "setup_ao.log", quarantine=q, mode="setup")
    run.split = split
    log(f"SETUP AO: run {run_dir.name}, model {a.model}, timeout {a.timeout_s}s, quarantine {q}")
    return run


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    T.hold_dispatch_lock(Path(a.run_dir).resolve())
    # each attempt gets a fresh view dir and run object; a rejected turn installs nothing, so a re-run starts clean
    return T.run_resuming(lambda: make_run(a), gate_argv=None, what="the benchmark adapter's turn")


if __name__ == "__main__":
    sys.exit(main())
