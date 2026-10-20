"""scripts/daytona/cc_pool_host.py on a TAU3 run: the run's one pool machine built from
scripts/daytona/Dockerfile.tau3_leg and proven by the tau3 steps, the cpu.weight of its cgroups (the bridges'
evobridge included), and the grader's interpreter its legs pass (InPlaceGrader(runner_py=...)).

NO DAYTONA, NO CLAUDE, NO KEY, NO NETWORK. tests/test_cc_pool_host.py's fake SDK and sandbox (a temp dir) with a
sandbox subclass that answers the tau3 steps as the machine would: the data root's tgz really extracted at its Mac path
inside the sandbox dir and checked against its own MANIFEST.sha256 (count, no link), grade_worker.py / leg.env / the
repo mirror dir really checked, the venv check and tau3_cc_smoke.TAU3_CHECK_PY answered from a table (the client's
tau3_venv / tau3_check; tau3_check_fail = the alltools check failing: a TAU3_CHECK_FAIL line and NO TAU3_CHECK line,
exit 1). The image's build context is the real tau3_cc_smoke.write_build_dir (a local `git archive` of
third_party/tau3-bench + the Mac's sandbox-runtime tree). The data root is a small one made here with its own
manifest. The ledger is the world fixture's temp file.
"""
from __future__ import annotations

import hashlib
import json
import re
import shlex
import sys
import tarfile
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts" / "daytona", ROOT / "tests"):
    sys.path.insert(0, str(_p))

import test_cc_pool_host as TH  # noqa: E402  (the fake machine; its fake daytona module lands before cc_pool_host)
import cc_pool_host as H  # noqa: E402
import daytona_lib_evo as dle_real  # noqa: E402
import ssb_leg_daytona_loop as M  # noqa: E402
import ssb_leg_pool as P  # noqa: E402
import tau3_cc_smoke as S  # noqa: E402
from src.orchestrator import pool_container_evolver as pce  # noqa: E402

world = TH.world
# the machine's image context is a `git archive` of the pinned tau3-bench checkout (third_party/README.md)
needs_tau3_checkout = pytest.mark.skipif(not (S.TAU3_REPO / ".git").exists(),
                                         reason="no third_party/tau3-bench checkout (third_party/README.md)")
sha = TH.sha
RUN = "tau3_cc_fx"
TAU3_REF = "daytona-70c842b98285d64d2944f338e2a51956857c8084a524915dd8ab765d28f6a25a:daytona"   # the tau3 image's
TAU3_DF_SHA = hashlib.sha256(S.DOCKERFILE.read_bytes()).hexdigest()
TAU3_PY = S.VM_TAU3_PY
TASKS = {"train": ["banking_knowledge/task_004", "banking_knowledge/task_005"], "val": ["banking_knowledge/task_001"]}
VENV_EQUAL = ("tau3_check_venv: equal: 76 packages, CPython 3.12.13 (/opt/tau3/venv/bin/python vs "
              "/opt/tau3/check/tau3_venv_freeze.txt)\n")
TAU3_STEPS = ["identity", "egress", "tau3 data", "tau3 dirs", "tau3 grade files", "cgroups", "cpu weights",
              "tau3 venv", "tau3 check", "ctl sha", "mkdir bin", "ctl sha", "cc sha", "cc dir", "cc verify"]


# ══ the fake tau3 machine ═══════════════════════════════════════════════════════════════════════
class Tau3Sandbox(TH.FakeSandbox):
    """test_cc_pool_host.FakeSandbox plus the tau3 steps of CCMachine.prove."""

    def play(self, step, script, env):
        c = self.client
        if step == "tau3 data":               # set -e; R=<root>; test ! -e "$R"; ...; tar -xzf; sha256sum -c; counts
            root = shlex.split(re.search(r"R=(\S+);", script).group(1))[0]
            dest, tgz = self.path(root), self.path(H.TAU3_DATA_TGZ_VM)
            if dest.exists():
                return "", 1
            dest.mkdir(parents=True)
            with tarfile.open(tgz) as tf:
                tf.extractall(dest, filter="data")
            tgz.unlink()
            rows = [ln.split("  ", 1) for ln in (dest / H.TAU3_MANIFEST).read_text().splitlines()]
            if any(not (dest / rel).is_file() or sha((dest / rel).read_bytes()) != s for s, rel in rows):
                return "sha256sum: WARNING: 1 computed checksum did NOT match\n", 1
            if any(p.is_symlink() for p in dest.rglob("*")) or \
                    sum(1 for p in dest.rglob("*") if p.is_file()) != len(rows) + 1:
                return "", 1
            return "DATA_OK\n", 0
        if step == "tau3 dirs":
            for d in shlex.split(script.split("mkdir -p ", 1)[1]):
                self.path(d).mkdir(parents=True, exist_ok=True)
            return "", 0
        if step == "tau3 grade files":       # sha256sum <worker> <manifest> && leg.env checks && test -d <repo>
            out = "".join(f"{sha(self.path(p).read_bytes())}  {p}\n"
                          for p in shlex.split(script.split("sha256sum ", 1)[1].split(" && ", 1)[0])
                          if self.path(p).is_file())
            envf = self.path("/opt/leg/leg.env")
            text = envf.read_text() if envf.is_file() else ""
            repo = shlex.split(re.search(r"test -d (\S+) && echo LEGENV_OK", script).group(1))[0]
            if text.strip() and not re.search(r"OPENROUTER|TOKEN|EVOSKILL_SOLVER", text) and self.path(repo).is_dir():
                out += "LEGENV_OK\n"
            return out, 0
        if step == "tau3 venv":
            assert script.split("\n", 1)[1].strip() == f"{H.ctl.PY} {S.VM_CHECK} --python {TAU3_PY} --pin {S.VM_PIN}"
            return getattr(c, "tau3_venv", VENV_EQUAL), 0
        if step == "tau3 check":
            assert all(f"-u {k}" in script for k in S.KEY_ENV_NAMES) and f" {TAU3_PY} -c " in script
            assert shlex.quote(S.TAU3_CHECK_PY) in script                     # the smoke's own check, verbatim
            root = shlex.split(re.search(r"TAU2_DATA_DIR=(\S+) ", script).group(1))[0]
            res = {"tau2_version": "1.0.1", "data_dir": root, "n_tasks": 97, "litellm": "1.81.11", "httpx": "0.28.1",
                   "openrouter_models_status": 200, "openrouter_models_n": 462, "openrouter_has_gpt_5_nano": True,
                   "sandbox": {"platform": "linux", "required": ["srt", "rg", "bwrap", "socat"],
                               "srt_package_version": "0.0.23"},
                   "embeddings": {"shape": [698, 3072], "ids_match": True},
                   "shell": {"ok": {"ls_lists_kb_documents": True, "task_file_unreadable": True,
                                    "opt_unreadable": True, "control_reads_the_task_file": True}}}
            res.update(getattr(c, "tau3_check", {}))
            fail = getattr(c, "tau3_check_fail", None)
            if fail is not None:                       # TAU3_CHECK_PY's failure shape: no TAU3_CHECK line, exit 1
                return "noise\nTAU3_CHECK_FAIL " + json.dumps(dict(res, reasons=fail)) + "\n", 1
            return "noise\nTAU3_CHECK " + json.dumps(res) + "\n", 0
        return super().play(step, script, env)


def make_data_root(d: Path) -> Path:
    """A tau3 data root as scripts/tau3_make_data_root.py lays it out: tau2/domains/banking_knowledge/{db.json,
    tasks/*.json} + MANIFEST.sha256 (sorted sha256sum lines)."""
    files = {"tau2/domains/banking_knowledge/db.json": b'{"users": {}}'}
    for t in TASKS["train"] + TASKS["val"]:
        files[f"tau2/domains/{t.replace('banking_knowledge/', 'banking_knowledge/tasks/')}.json"] = \
            json.dumps({"id": t.split("/")[1]}).encode()
    for rel, data in files.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_bytes(data)
    (d / H.TAU3_MANIFEST).write_text("".join(f"{sha(files[r])}  {r}\n" for r in sorted(files)))
    return d


@pytest.fixture
def tau3(world):
    """A tau3 run dir beside the world's: SPEC benchmark tau3 (the freeze's split, plug-in ids), INFRA leg_sandbox =
    Dockerfile.tau3_leg + the smoke's snapshot, data.root = a small manifest-verified root; the fake client answers
    the tau3 steps and reads back the tau3 snapshot."""
    rd = world.tmp / "outputs" / RUN
    (rd / "setup").mkdir(parents=True)
    split = rd / "setup" / "split.json"
    split.write_text(json.dumps({"domain": "banking_knowledge", **TASKS, "test": ["banking_knowledge/task_002"]}))
    data = make_data_root(world.tmp / "tau3_data_fx")
    (rd / "SPEC.json").write_text(json.dumps({"spec": {
        "name": RUN, "benchmark": "tau3", "solver_harness": "claude_code", "solver_model": "claude-sonnet-5-5",
        "split_path": str(split), "split_sha256": M.sha256_file(split), "n_samples": 3}}))
    inf = json.loads(TH.INFRA_SRC.read_text())
    inf["leg_sandbox"].update(dockerfile="scripts/daytona/Dockerfile.tau3_leg", dockerfile_sha256=TAU3_DF_SHA,
                              image_id="x86daytona-" + TAU3_DF_SHA[:12], snapshot_ref=TAU3_REF)
    inf["claude_sandbox"]["base_dockerfile_sha256"] = TAU3_DF_SHA     # INFRA's rule: the Claude image is FROM the leg's
    inf["data"].update(root=str(data), overlay_patches_json=None, overlay_patches_sha256=None)
    (rd / "INFRA.json").write_text(json.dumps(inf))
    rt = json.loads((world.rd / "RUNTIME.json").read_text())
    rt["daytona_leg"]["require_snapshot_ref"] = TAU3_REF
    (rd / "RUNTIME.json").write_text(json.dumps(rt))
    world.client.snapshot_ref = TAU3_REF
    world.client.sandbox_cls = Tau3Sandbox
    return types.SimpleNamespace(**{**vars(world), "rd": rd, "data": data, "split_file": split, "inf": inf,
                                    "store": world.store})


def s_image_srt(o, name="m01"):
    """The srt tree sha the machine's summary records for its image (tau3_cc_smoke.write_build_dir's meta)."""
    return json.loads((o.store / name / "summary.json").read_text())["image"]["srt"]["tree_sha256"]


def owner(t, name="p1"):
    o = TH.make_owner(t, pool_dir=t.store / RUN / "pool" / "run" / name)
    o.reap()
    return o


def write_infra(t, **leg_sandbox):
    inf = json.loads(json.dumps(t.inf))
    inf["leg_sandbox"].update(leg_sandbox)
    inf["claude_sandbox"]["base_dockerfile_sha256"] = inf["leg_sandbox"]["dockerfile_sha256"]
    (t.rd / "INFRA.json").write_text(json.dumps(inf))


# ══ machine_cfg ═════════════════════════════════════════════════════════════════════════════════
def test_machine_cfg_takes_a_tau3_runs_image_from_infra_and_verifies_its_data_root(tau3):
    cfg = H.machine_cfg(tau3.rd)
    assert cfg.benchmark == "tau3" and cfg.tau3 is True
    assert cfg.dockerfile_sha256 == TAU3_DF_SHA and cfg.image_id == "x86daytona-" + TAU3_DF_SHA[:12]
    assert cfg.image_id != M.EXPECTED["image_id"]
    assert cfg.rt["require_snapshot_ref"] == TAU3_REF and cfg.require_snapshot_ref == TAU3_REF
    assert cfg.require_libreoffice == M.EXPECTED["libreoffice"]
    assert cfg.tasks == TASKS["train"] + TASKS["val"] and cfg.n_samples == 3 and cfg.data_overlay is None
    assert cfg.data == {"root": str(tau3.data), "files": 4, "tasks": 3,
                        "manifest_sha256": sha((tau3.data / H.TAU3_MANIFEST).read_bytes())}
    assert cfg.tag == f"pool_{RUN}" and cfg.code == P.pool_code(RUN)


def test_machine_cfg_refuses_a_tau3_image_that_is_not_infras_and_a_data_root_off_its_manifest(tau3):
    write_infra(tau3, dockerfile_sha256="0" * 64, image_id="x86daytona-" + "0" * 12)
    with pytest.raises(M.Refused, match=rf"Dockerfile.tau3_leg sha256 {TAU3_DF_SHA[:12]} != INFRA 000000000000"):
        H.machine_cfg(tau3.rd)
    ssb_sha = M.sha256_file(M.DOCKERFILE)                   # the SSB image, named for a tau3 run
    write_infra(tau3, dockerfile="scripts/daytona/Dockerfile.ssb_leg", dockerfile_sha256=ssb_sha,
                image_id="x86daytona-" + ssb_sha[:12])
    with pytest.raises(M.Refused, match="is not scripts/daytona/Dockerfile.tau3_leg"):
        H.machine_cfg(tau3.rd)
    write_infra(tau3)
    assert H.machine_cfg(tau3.rd).tau3
    (tau3.data / "tau2" / "stray.json").write_text("{}")   # a file the manifest does not list
    with pytest.raises(M.Refused, match="INFRA data.root: data root: .* differs from its manifest"):
        H.machine_cfg(tau3.rd)


def test_an_ssb_run_keeps_its_image_and_takes_no_tau3_path(world):
    cfg = H.machine_cfg(world.rd)
    assert cfg.benchmark is None and cfg.tau3 is False and cfg.data is None
    assert cfg.image_id == M.EXPECTED["image_id"] and cfg.dockerfile_sha256 == M.EXPECTED["dockerfile_sha256"]
    sp = json.loads((world.rd / "SPEC.json").read_text())
    sp["spec"]["benchmark"] = "ssbv2"
    (world.rd / "SPEC.json").write_text(json.dumps(sp))
    cfg = H.machine_cfg(world.rd)
    assert cfg.benchmark == "ssbv2" and cfg.tau3 is False and cfg.image_id == M.EXPECTED["image_id"]


# ══ the machine ═════════════════════════════════════════════════════════════════════════════════
@needs_tau3_checkout
def test_the_tau3_machine_is_built_from_the_tau3_image_and_proven_in_order(tau3):
    o = owner(tau3)
    (o.store / "m01" / "build").mkdir(parents=True)
    (o.store / "m01" / "build" / "left_over").write_text("x")      # a context left by an earlier attempt is rewritten
    m = o.create_machine("m01")
    sb = tau3.client.get(m.sid)
    assert [s for s, _ in sb.calls] == TAU3_STEPS, [s for s, _ in sb.calls]
    calls = {st: script.split("\n", 1)[1].strip() for st, script in sb.calls}     # the marker line and the newline off
    # the image: Dockerfile.tau3_leg next to its clean build context, created as the Shard creates a machine
    params = tau3.client.created[-1]
    build = o.store / "m01" / "build"
    assert params.image == ("dockerfile", str(build / "Dockerfile.tau3_leg")) and not (build / "left_over").exists()
    assert sha((build / "Dockerfile.tau3_leg").read_bytes()) == TAU3_DF_SHA
    assert (build / "tau3_ctx" / "tau2-bench" / "src" / "tau2" / "__init__.py").is_file()
    assert (build / "tau3_ctx" / "check" / "tau3_venv_freeze.txt").is_file()
    srt = build / "tau3_ctx" / "srt" / "sandbox-runtime"                 # the KB shell's sandbox-runtime (step 9b)
    assert json.loads((srt / "package.json").read_text())["version"] == "0.0.23"
    assert (srt / "dist" / "cli.js").stat().st_mode & 0o111 and s_image_srt(o) == S.srt_files()[1]["tree_sha256"]
    assert params.resources == M.RESOURCES and params.ttl_minutes == 1440 and params.ephemeral is True
    assert params.domain_allow_list == pce.MACHINE_ALLOW_STR and params.public is False
    assert params.name.startswith(f"{pce.POOL_PREFIX}{P.pool_code(RUN)}-m01-") and params.labels["shard"] == "m01"
    s = json.loads((o.store / "m01" / "summary.json").read_text())
    assert s["image"]["dockerfile_sha256"] == TAU3_DF_SHA and s["readback"]["snapshot_ref"] == TAU3_REF
    # the data root AT ITS MAC PATH, every file the Mac's, MANIFEST checked; then the grader's files
    root = str(tau3.data)
    for p in tau3.data.rglob("*"):
        if p.is_file():
            assert sb.path(p).read_bytes() == p.read_bytes(), p
    assert not sb.path(H.TAU3_DATA_TGZ_VM).exists() and "sha256sum -c --quiet MANIFEST.sha256" in calls["tau3 data"]
    assert sb.path(H.GRADE_WORKER_VM).read_bytes() == H.GRADE_WORKER_SRC.read_bytes()
    assert sb.path(P.SANDBOX_REPO).is_dir()
    text = sb.path("/opt/leg/leg.env").read_text()
    env = {ln.split("=", 1)[0][len("export "):]: shlex.split(ln.split("=", 1)[1])[0] for ln in text.splitlines()}
    assert env["EVOSHIM_IMAGE_ID"] == "x86daytona-" + TAU3_DF_SHA[:12]
    assert env["EVOSKILL_RECALC_TIMEOUT_SECONDS"] == "900"
    assert not re.search(r"OPENROUTER|TOKEN|EVOSKILL_SOLVER", text)
    b = s["bundle"]
    assert b["data_root"] == root and b["files"] == 4 and b["tasks"] == 3
    assert b["verified_on_machine"] == {H.GRADE_WORKER_VM: sha(H.GRADE_WORKER_SRC.read_bytes()),
                                        f"{root}/{H.TAU3_MANIFEST}": sha((tau3.data / H.TAU3_MANIFEST).read_bytes())}
    assert "EVOSHIM_IMAGE_ID" in b["leg_env_keys"]
    # the cgroups (the Shard's delegation + the bridges' cgroup) and their weights; no shim / SSB self-test
    assert calls["cgroups"] == H.CGROUP_CMD + "; mkdir -p $CG/evobridge"
    want = dict(H.CPU_WEIGHTS, evobridge=10000)
    assert s["cgroup_weights"] == {"want": want, "readback": want, "equal": True}
    assert sb.path("/sys/fs/cgroup/evobridge/cpu.weight").read_text().strip() == "10000"
    assert not {"unpack", "verify", "selftest", "grade"} & {st for st, _ in sb.calls}
    # the tau3 venv / tau2 / data root / openrouter check = grade_selftest
    g = s["grade_selftest"]
    assert g["venv_equals_pin"] is True and g["interpreter"] == TAU3_PY and g["data_root"] == root
    assert g["tau3_check"]["n_tasks"] == 97 and g["tau3_check"]["data_dir"] == root
    assert f"TAU2_DATA_DIR={root} " in calls["tau3 check"]
    assert s["install"]["cc"]["cc"].startswith("uploaded") and s["granted_ttl_min"] == 1440
    assert all(k in s for k in H.SUMMARY_KEYS)
    # the summary carries what the leg driver's join checks
    row = TH.machines_json(o.reg)["machines"]["m01"]
    inf, _ = M.load_infra(tau3.rd)
    assert H.identity_problems(s, row, dle_real.readback(sb), inf) == []
    o.close()
    assert tau3.client.deleted == [m.sid]


@needs_tau3_checkout
def test_a_leg_joins_the_tau3_machine(tau3, tmp_path):
    h = TH.HostThread(tau3).start()
    try:
        dle = TH.fake_dle(tau3.client)
        handle, lease = H.join(tau3.rd, "ccleg-tau3-t1", 3600, tag="t1", store_dir=tmp_path / "handle",
                               client=tau3.client, dle=dle, poll_s=0.02,
                               spawn=lambda d, r: pytest.fail("spawned beside a live host"), **TH.cc_kw(tau3))
        try:
            assert handle.sid == h.machine.sid and handle.rec["readback_join"]["snapshot_ref"] == TAU3_REF
            assert handle.rec["host_summary"]["grade_selftest"]["tau3_check"]["tau2_version"] == "1.0.1"
            assert handle.rec["host_summary"]["bundle"]["data_root"] == str(tau3.data)
            assert handle.rec["install"]["cc"]["cc"].startswith("already")
        finally:
            handle.release()
    finally:
        h.stop()


@needs_tau3_checkout
@pytest.mark.parametrize("answer, why", [
    ({"tau3_venv": "tau3_check_venv: 1 difference(s) between ...\n  litellm 1.81.11 != 1.81.12\n"},
     "tau3 venv differs from its pin"),
    ({"tau3_check": {"n_tasks": 95}}, "with 95 tasks"),
    ({"tau3_check": {"tau2_version": None, "import_error": "ModuleNotFoundError: No module named 'tau2'"}},
     "tau2 None"),
    ({"tau3_check": {"openrouter_models_status": None, "openrouter_error": "ConnectError"}}, "openrouter /models None"),
    # the alltools check failing (KB shell leak / no srt / no cache entry): TAU3_CHECK_FAIL and no TAU3_CHECK line --
    # the host finds no tau2 version, no data dir, no openrouter status in it and refuses the machine
    ({"tau3_check_fail": ["KB shell probe: ['task_file_unreadable']"]}, "tau2 None"),
    ({"tau3_check_fail": ["sandbox binaries missing on PATH: ['srt', 'bwrap']"]}, "openrouter /models None"),
    ({"snapshot_ref": "daytona-rebuilt:daytona"}, "Daytona snapshot daytona-rebuilt:daytona != the baseline's"),
])
def test_a_tau3_machine_that_fails_a_check_is_deleted(tau3, answer, why):
    for k, v in answer.items():
        setattr(tau3.client, k, v)
    o = owner(tau3)
    with pytest.raises(M.Abort, match=why):
        o.create_machine("m01")
    m = o.machines[0]
    assert m.failed and m.deleted is True and tau3.client.deleted == [m.sid] and not tau3.client.live
    assert TH.machines_json(o.reg)["current"] is None


@needs_tau3_checkout
def test_a_tau3_cpu_weight_readback_that_differs_keeps_the_machine(tau3):
    tau3.client.cpu_weight_kept = {"evobridge": 100}
    o = owner(tau3)
    m = o.create_machine("m01")
    assert m.usable() and TH.machines_json(o.reg)["current"] == "m01"
    rec = json.loads((o.store / "m01" / "summary.json").read_text())["cgroup_weights"]
    assert rec["equal"] is False and rec["readback"]["evobridge"] == 100 and rec["want"]["evobridge"] == 10000
    o.close()


# ══ the grader's interpreter on a tau3 leg ══════════════════════════════════════════════════════
def test_a_tau3_legs_grader_runs_under_the_tau3_venv(world, tmp_path):
    gs = TH.grading_setup(world, tmp_path)
    g = TH.make_grader(world, gs, tmp_path / "sg", runner_py=TAU3_PY)
    cmd = g.launch_cmd()
    assert TAU3_PY == "/opt/tau3/venv/bin/python" and f"{TAU3_PY} -u /opt/grade/plugin_grade_runner.py run" in cmd
    assert H.RUNNER_PY not in cmd and g.grader_record()["interpreter"] == TAU3_PY
    assert TH.make_grader(world, gs, tmp_path / "sg2").launch_cmd() == TH.frozen_launch_cmd(g)


def test_nothing_here_reads_a_key_or_names_the_seat_guards_pattern():
    src = Path(H.__file__).read_text()
    assert src.count("run_ssbv2") <= 3 and "openrouter_key" not in src and "read_token" not in src
