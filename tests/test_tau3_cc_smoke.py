"""The tau3 Daytona smoke driver (scripts/daytona/tau3_cc_smoke.py) and its image (scripts/daytona/Dockerfile.tau3_leg),
offline: what can be checked before a run on Daytona.

No Daytona call, no model, no network, no real token or key: the Daytona side is a fake SDK / sandbox / ledger, the
secrets are test strings spelled in pieces, and the one end-to-end run is local: the driver's OWN launch command
(minus setsid, which macOS lacks) starts the real harness (scripts/tau3_cc/tau2_cc_harness.py) under the tau3-bench
venv with the driver's OWN job (its container mode), evoctl stood in for by tests/fake_claude_cc.py --evoctl, Claude
Code by the fake, the customer by a scripted one; the conversation is then graded by the user's grader through the
driver's own host-side grade call.

What must hold:
  - the image: Dockerfile.tau3_leg = Dockerfile.ssb_leg's bytes + the /opt/tau3 layer (uv 0.10.12, CPython
    3.12.13, `uv sync --frozen --no-dev --extra knowledge --python 3.12.13` into /opt/tau3/venv, the venv-pin check,
    uv removed), then step 9b (alltools: apt nodejs ripgrep bubblewrap socat, sandbox-runtime 0.0.23 at
    /usr/local/lib/node_modules/..., the srt link, the version assertions), then step 7's assertions again; its COPY
    sources are the driver's clean context only;
  - the context: `git archive 363133a` of the six paths with uv.lock patched to the pinned sha, plus the check and its
    pin, plus the host's sandbox-runtime tree file by file with its exec bits (a symlink, another version, a lost exec
    bit or a file named claude refused); nothing of data/, .venv, .git; deterministic; the Daytona SDK resolves exactly
    the three context dirs next to the copy;
  - the bridge is three files (the harness imports tau3_retrieval from its own dir); the grade asks for alltools;
  - TAU3_CHECK_PY (the machine check) run here under the tau3 venv: TAU3_CHECK with the sandbox binaries, the cached
    embeddings [698, 3072] and the hardened KB shell reading nothing of the data root or /opt while its control does;
    with the deny entries removed it prints TAU3_CHECK_FAIL and no TAU3_CHECK (the leak is caught);
  - the plan: train tasks only; every job passes the harness's own load_job, in its container mode (fd 3 = the token,
    fd 4 = the key), with no secret in it; a --job-set can never move a path, an fd or the container;
  - the launch: neither secret in the command string; both reach the bridge on fds 3 / 4 only, never in its
    environment, argv or any file; claude gets fd 3 and the harness's variables, never the key's fd;
  - the machine is deleted on every exit (a refusal, a failed create whose id was seen, SIGTERM in the middle of a
    rollout) and the running containers are torn down first;
  - --dry-run constructs nothing that reaches Daytona or a secret and writes nothing.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
DRIVER = ROOT / "scripts" / "daytona" / "tau3_cc_smoke.py"
# the curated data root (scripts/tau3_make_data_root.py + the embeddings precompute); tests that need it skip without it
DATA_ROOT = Path(os.environ.get("EVOSKILL_TAU3_DATA_ROOT") or ROOT / "third_party" / "tau3_data_363133a")
FAKE = TESTS / "fake_claude_cc.py"
SDK_PY = Path(sys.executable)   # the Daytona SDK is the `daytona` extra of this venv
TOK = "sk-ant-" + "oat01-" + "TESTONLY" + hashlib.sha256(b"smoke token").hexdigest()[:48]
KEY = "sk-or-" + "v1-" + "TESTONLY" + hashlib.sha256(b"smoke key").hexdigest()[:48]
CUSTOMER = "cc_smoke_test_customer"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


S = _load(DRIVER, "tau3_cc_smoke")
G = _load(TESTS / "tau3_golden.py", "tau3_golden")
H = S.H


def _need_data_root():
    if not (DATA_ROOT / "MANIFEST.sha256").is_file():
        pytest.skip(f"the curated data root {DATA_ROOT} is absent")


def _need_tau3_repo():
    if not (S.TAU3_REPO / ".git").exists():
        pytest.skip(f"{S.TAU3_REPO} is not a git checkout")
    src = S.default_srt_src()
    if src is None or not (src / "package.json").is_file():
        pytest.skip(f"no installed sandbox-runtime {S.SRT_VERSION} ({S.SRT_SRC_ENV}, or npm install -g)")


def _need_tau3_venv_and_srt():
    _need_data_root()
    if not G.TAU3_PY.is_file():
        pytest.skip(f"the tau3-bench venv {G.TAU3_PY} is absent")
    if not shutil.which("srt", path=S.host_tool_path()) or not shutil.which("rg", path=S.host_tool_path()):
        pytest.skip(f"srt / rg are not on {S.host_tool_path()}")


DEAD_PROXY = {k: "http://127.0.0.1:9" for k in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy",
                                                  "all_proxy")}


# ─────────────────────────────────────────────────────────────── the image
def test_the_dockerfile_is_the_ssb_leg_bytes_plus_the_tau3_layer():
    base, full = S.BASE_DOCKERFILE.read_bytes(), S.DOCKERFILE.read_bytes()
    assert full.startswith(base) and len(full) > len(base)
    tail = full[len(base):].decode()
    meta = S.dockerfile_meta()
    assert meta["copy_sources"] == ["tau3_ctx/tau2-bench", "tau3_ctx/check", "tau3_ctx/srt/sandbox-runtime"]
    for piece in ("pip install --no-cache-dir uv==0.10.12",
                  "UV_PYTHON_INSTALL_DIR=/opt/tau3/python uv python install --no-bin 3.12.13",
                  "UV_PROJECT_ENVIRONMENT=/opt/tau3/venv",
                  "uv sync --frozen --no-dev --extra knowledge --python 3.12.13",
                  "tau3_check_venv.py --python /opt/tau3/venv/bin/python",
                  "equal: 76 packages, CPython 3.12.13",
                  "pip uninstall -y uv && rm -rf /root/.cache",
                  S.UVLOCK_SHA_PATCHED):
        assert piece in tail, piece
    step7 = 'test -z "$(find / -xdev -name claude -type f 2>/dev/null)"'
    assert base.decode().count(step7) == 1 and tail.count(step7) == 1
    run_sync = tail.index("UV_PROJECT_ENVIRONMENT=/opt/tau3/venv UV_LINK_MODE=copy")
    assert tail.index("COPY tau3_ctx/tau2-bench") < run_sync < tail.index(step7)
    assert "! command -v claude && ! command -v uv" in tail
    assert tail.rstrip().endswith("WORKDIR /workspace")


def test_step_9b_installs_the_kb_shells_sandbox_between_step_9_and_step_10():
    tail = S.DOCKERFILE.read_text()[len(S.BASE_DOCKERFILE.read_text()):]
    i9, i9b, i10 = (tail.index(f"# ---- {n}. ") for n in ("9", "9b", "10"))
    assert i9 < i9b < i10
    step = tail[i9b:i10]
    srt = "/usr/local/lib/node_modules/@anthropic-ai/sandbox-runtime"
    assert srt == S.VM_SRT and not srt.startswith("/opt")
    for piece in ("apt-get install -y --no-install-recommends \\\n        nodejs ripgrep bubblewrap socat",
                  "apt-get clean && rm -rf /var/lib/apt/lists/*",
                  f"COPY tau3_ctx/srt/sandbox-runtime {srt}\n",
                  'chmod 0755 "$SRT/dist/cli.js"',
                  'find "$SRT" -type f -name apply-seccomp -exec chmod 0755 {} +',
                  "ln -s ../lib/node_modules/@anthropic-ai/sandbox-runtime/dist/cli.js /usr/local/bin/srt",
                  'test "$(node -p "require(\'$SRT/package.json\').version")" = 0.0.23',
                  "srt --version",
                  "command -v srt && command -v rg && command -v bwrap && command -v socat && command -v node",
                  'test -z "$(find "$SRT" -name claude)"',
                  "> /opt/tau3/sandbox_versions.txt",
                  "grep -q '^sandbox-runtime 0.0.23 ' /opt/tau3/sandbox_versions.txt"):
        assert piece in step, piece
    # step 10 (step 7's assertions) still follows, unchanged: no file named claude anywhere, no uv
    step10 = tail[i10:]
    assert 'test -z "$(find / -xdev -name claude -type f 2>/dev/null)"' in step10
    assert "! command -v claude && ! command -v uv" in step10
    assert "npm " not in step and "pip " not in step and "curl" not in step   # srt comes from the context only


def test_the_dockerfile_meta_refuses_another_copy_layout(monkeypatch, tmp_path):
    base = S.BASE_DOCKERFILE.read_bytes()
    for bad in (S.DOCKERFILE.read_bytes().replace(b"COPY tau3_ctx/srt/sandbox-runtime /usr/local/lib/node_modules/",
                                                  b"COPY tau3_ctx/srt/sandbox-runtime /opt/node_modules/"),
                S.DOCKERFILE.read_bytes().replace(b"COPY tau3_ctx/srt/sandbox-runtime", b"COPY tau3_ctx/srt")):
        f = tmp_path / "Dockerfile.tau3_leg"
        f.write_bytes(bad)
        assert bad.startswith(base)
        monkeypatch.setattr(S, "DOCKERFILE", f)
        with pytest.raises(S.Refuse):
            S.dockerfile_meta()


def test_the_build_context_is_the_clean_patched_archive():
    _need_tau3_repo()
    files, meta = S.context_files()
    names = sorted(files)
    assert all(n.startswith(("tau2-bench/", "check/", "srt/sandbox-runtime/")) for n in names)
    assert not [n for n in names if n.startswith(("tau2-bench/data/", "tau2-bench/.venv", "tau2-bench/.git"))]
    top = {n.split("/")[1] for n in names if n.startswith("tau2-bench/")}
    assert top == {"pyproject.toml", "uv.lock", "README.md", "LICENSE", ".python-version", "src"}
    assert hashlib.sha256(files["tau2-bench/uv.lock"][0]).hexdigest() == S.UVLOCK_SHA_PATCHED
    assert files["tau2-bench/pyproject.toml"][0] == subprocess.run(
        ["git", "-C", str(S.TAU3_REPO), "show", f"{S.TAU3_COMMIT}:pyproject.toml"], capture_output=True).stdout
    assert files["check/tau3_check_venv.py"][0] == S.CHECK_VENV.read_bytes()
    assert files["check/tau3_venv_freeze.txt"][0] == S.FREEZE.read_bytes()
    assert meta["bytes"] < S.CONTEXT_MAX_BYTES, meta["bytes"]
    again = S.context_files()[1]
    assert again["context_sha256"] == meta["context_sha256"]                         # deterministic
    assert again["srt"]["tree_sha256"] == meta["srt"]["tree_sha256"]
    # the srt tree: the host's installed package, every file byte-equal with its exec bit, nothing else
    srt = {n[len("srt/sandbox-runtime/"):]: v for n, v in files.items() if n.startswith("srt/sandbox-runtime/")}
    src = S.default_srt_src()
    on_disk = {p.relative_to(src).as_posix(): p for p in src.rglob("*") if p.is_file()}
    assert set(srt) == set(on_disk) and len(srt) == meta["srt"]["n_files"] > 2000
    for rel, p in on_disk.items():
        assert srt[rel][0] == p.read_bytes(), rel
        assert srt[rel][1] == (0o755 if p.stat().st_mode & 0o111 else 0o644), rel
    execs = sorted(r for r, (_, m) in srt.items() if m == 0o755)
    assert "dist/cli.js" in execs and "vendor/seccomp/x64/apply-seccomp" in execs
    assert "vendor/seccomp/arm64/apply-seccomp" in execs and execs == meta["srt"]["executables"]
    assert json.loads(srt["package.json"][0])["version"] == "0.0.23" == meta["srt"]["version"]
    assert meta["srt"]["tree_sha256"] == S.tree_sha256(files, "srt/sandbox-runtime/")
    assert not [n for n in names if Path(n).name == "claude"]


def _fake_srt(d: Path, version: str = "0.0.23") -> Path:
    (d / "dist").mkdir(parents=True)
    (d / "vendor" / "seccomp" / "x64").mkdir(parents=True)
    (d / "package.json").write_text(json.dumps({"name": "@anthropic-ai/sandbox-runtime", "version": version}))
    (d / "dist" / "index.js").write_text("export {}\n")
    for f in (d / "dist" / "cli.js", d / "vendor" / "seccomp" / "x64" / "apply-seccomp"):
        f.write_text("#!/bin/sh\n")
        f.chmod(0o755)
    return d


def test_the_srt_tree_is_refused_unless_it_is_the_pinned_plain_executable_tree(tmp_path):
    files, meta = S.srt_files(_fake_srt(tmp_path / "ok"))
    assert sorted(files) == ["srt/sandbox-runtime/dist/cli.js", "srt/sandbox-runtime/dist/index.js",
                             "srt/sandbox-runtime/package.json", "srt/sandbox-runtime/vendor/seccomp/x64/apply-seccomp"]
    assert files["srt/sandbox-runtime/dist/index.js"][1] == 0o644 and meta["executables"] == [
        "dist/cli.js", "vendor/seccomp/x64/apply-seccomp"]
    assert S.srt_files(tmp_path / "ok")[1]["tree_sha256"] == meta["tree_sha256"]
    (tmp_path / "ok" / "dist" / "cli.js").chmod(0o644)                         # a lost exec bit changes the identity
    with pytest.raises(S.Refuse, match="not executable"):
        S.srt_files(tmp_path / "ok")
    with pytest.raises(S.Refuse, match="0.0.24, not @anthropic-ai/sandbox-runtime 0.0.23"):
        S.srt_files(_fake_srt(tmp_path / "v24", "0.0.24"))
    d = _fake_srt(tmp_path / "link")
    (d / "dist" / "evil").symlink_to("/etc/passwd")
    with pytest.raises(S.Refuse, match="dist/evil is a symlink"):
        S.srt_files(d)
    d = _fake_srt(tmp_path / "dirlink")
    (d / "node_modules").symlink_to(tmp_path / "ok", target_is_directory=True)
    with pytest.raises(S.Refuse, match="node_modules is a symlink"):
        S.srt_files(d)
    d = _fake_srt(tmp_path / "claude")
    (d / "dist" / "claude").write_text("x")
    with pytest.raises(S.Refuse, match="named claude"):
        S.srt_files(d)
    d = _fake_srt(tmp_path / "noseccomp")
    (d / "vendor" / "seccomp" / "x64" / "apply-seccomp").unlink()
    with pytest.raises(S.Refuse, match="no vendored apply-seccomp"):
        S.srt_files(d)
    with pytest.raises(S.Refuse, match="not the installed sandbox-runtime tree"):
        S.srt_files(tmp_path / "absent")


def test_the_lock_patch_applies_only_where_it_says():
    patch = S.UVLOCK_PATCH.read_text()
    with pytest.raises(S.Refuse, match="does not apply"):
        S.apply_patch("nothing\nlike\nthe lock\n", patch, "uv.lock")
    with pytest.raises(S.Refuse, match="one diff of"):
        S.apply_patch("x", patch, "other.lock")


def test_the_daytona_sdk_uploads_exactly_the_three_context_dirs(tmp_path):
    _need_tau3_repo()
    if not SDK_PY.is_file():
        pytest.skip(f"{SDK_PY} (the interpreter with the Daytona SDK) is absent")
    meta = S.write_build_dir(tmp_path / "build")
    code = ("import json, sys\nfrom daytona import Image\nimg = Image.from_dockerfile(sys.argv[1])\n"
            "print(json.dumps([[c.source_path, c.archive_path] for c in img._context_list]))")
    r = subprocess.run([str(SDK_PY), "-c", code, meta["build_dockerfile"]], capture_output=True, text=True,
                       cwd=str(tmp_path), env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    ctx = json.loads(r.stdout.strip().splitlines()[-1])
    build = str(tmp_path / "build")
    assert [a for _s, a in ctx] == ["tau3_ctx/tau2-bench", "tau3_ctx/check", "tau3_ctx/srt/sandbox-runtime"]
    assert all(Path(s).resolve().is_relative_to(Path(build).resolve()) for s, _a in ctx)
    size = sum(p.stat().st_size for s, _a in ctx for p in Path(s).rglob("*") if p.is_file())
    assert size == meta["bytes"] < S.CONTEXT_MAX_BYTES
    assert (tmp_path / "build" / "Dockerfile.tau3_leg").read_bytes() == S.DOCKERFILE.read_bytes()
    # the build dir carries the modes (the SDK's tar keeps them): the srt entry point and its seccomp helpers execute
    srt = tmp_path / "build" / "tau3_ctx" / "srt" / "sandbox-runtime"
    for rel in ("dist/cli.js", "vendor/seccomp/x64/apply-seccomp", "vendor/seccomp/arm64/apply-seccomp"):
        assert (srt / rel).stat().st_mode & 0o777 == 0o755, rel
    assert (srt / "package.json").stat().st_mode & 0o777 == 0o644
    assert not [p for p in srt.rglob("*") if p.is_symlink()]


def test_write_build_dir_is_deterministic(tmp_path):
    _need_tau3_repo()

    def tree(d: Path) -> str:
        return S.tree_sha256({p.relative_to(d).as_posix(): (p.read_bytes(), p.stat().st_mode & 0o777)
                              for p in sorted(d.rglob("*")) if p.is_file()})
    m1, m2 = S.write_build_dir(tmp_path / "b1"), S.write_build_dir(tmp_path / "b2")
    for k in ("context_sha256", "dockerfile_sha256", "n_files", "bytes"):
        assert m1[k] == m2[k], k
    assert m1["srt"]["tree_sha256"] == m2["srt"]["tree_sha256"]
    assert tree(tmp_path / "b1") == tree(tmp_path / "b2")
    assert tree(tmp_path / "b1" / "tau3_ctx") == m1["context_sha256"]      # the meta's sha is the bytes + modes on disk
    with pytest.raises(S.Refuse, match="is not empty"):
        S.write_build_dir(tmp_path / "b1")


# ─────────────────────────────────────────────────────────────── the plan
def test_tasks_are_train_tasks_only(tmp_path):
    split = {"domain": "banking_knowledge", "train": ["task_a", "task_b", "task_c"], "val": ["task_v"],
             "test": ["task_t"], "dropped": {"task_ids": ["task_d"]}}
    tasks = tmp_path / "tau2" / "domains" / "banking_knowledge" / "tasks"
    tasks.mkdir(parents=True)
    for t in ("task_a", "task_b", "task_v", "task_d"):
        (tasks / f"{t}.json").write_text("{}")
    assert S.pick_tasks(split, None, 2, tmp_path) == ["task_a", "task_b"]
    assert S.pick_tasks(split, "task_b", 2, tmp_path) == ["task_b"]
    for bad, why in (("task_v", "not in split"), ("task_d", "not in split"), ("task_c", "missing"),
                     ("task_a,task_a", "repeated")):
        with pytest.raises(S.Refuse, match=why):
            S.pick_tasks(split, bad, 2, tmp_path)


def test_the_real_split_gives_train_tasks_with_files():
    _need_data_root()
    split = S.load_split()
    tasks = S.pick_tasks(split, None, 2, DATA_ROOT)
    assert tasks == split["train"][:2]
    assert not set(tasks) & set(split["dropped"]["task_ids"])


def test_every_job_passes_the_harness_in_its_container_mode():
    _need_data_root()
    cid = S.make_cid("task_005", 1, "20261001T001234", salt="ab12")
    assert S.ctl.CID_RE.match(cid) and H.CID_RE.match(cid)
    job = S.unit_job("task_005", 1, cid, "/opt/tau3smoke/units/task_005_s1", root=DATA_ROOT)
    eff = S.validate_job(job)
    assert eff["container"] == {"evoctl": [S.ctl.PY, S.ctl.CTL], "cid": cid, "cwd": "/evo/work"}
    assert (eff["token_fd"], eff["key_fd"], eff["home"]) == (3, 4, "/home/evo")
    assert Path(eff["task_file"]).parents[4] == DATA_ROOT
    lp = H.launch_plan(eff)
    argv = lp["argv"]
    assert argv[:9] == [S.ctl.PY, S.ctl.CTL, "enter", "--cid", cid, "--cwd", "/evo/work", "--keep-fd3", "--"]
    assert argv[9] == "/usr/bin/env" and "claude" in argv and "--max-turns" not in argv
    pairs = argv[10:argv.index("claude")]
    want = {f"{k}={v}" for k, v in {**H.CC_ENV, **H.TAU2_ENV}.items()}
    assert want <= set(pairs) and "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR=3" in pairs and "HOME=/home/evo" in pairs
    assert lp["pass_fds"] == (3,)
    text = json.dumps(job)
    assert S.TOKEN_PREFIX not in text and S.OR_PREFIX not in text and "TOKEN=" not in text


def test_a_job_set_never_moves_a_path_an_fd_or_the_container():
    assert S.parse_job_sets(["agent_time_s=600", "customer_attempts=2"]) == {"agent_time_s": 600,
                                                                              "customer_attempts": 2}
    for bad in ("workdir=/x", "token_fd=5", "container=null", "home=/root", "nope=1", "noequals"):
        with pytest.raises(S.Refuse):
            S.parse_job_sets([bad])


def test_the_hard_stop_follows_the_agent_clock():
    assert S.hard_stop_s({"agent_time_s": 1200.0}) == 1200.0 + S.CUSTOMER_ALLOWANCE_S
    assert S.hard_stop_s({"agent_time_s": 1200.0}, 30) == 1800.0
    assert S.hard_stop_s({"wall_s": 100.0}) == 100.0 + S.HARD_STOP_MARGIN_S


def test_the_harness_settings_are_reported():
    hv = S.harness_settings()
    assert hv["container_mode"] is True
    for k in ("first_message_fills_policy_and_customer", "no_claude_max_turns",
              "agent_time_s_default", "customer_model_is_gpt_5_nano", "empty_customer_llm_args",
              "job_variant_is_alltools", "harness_default_variant", "harness_imports_tau3_retrieval"):
        assert hv[k] is not None and not str(hv[k]).startswith("unknown"), (k, hv[k])
    assert hv["first_message_fills_policy_and_customer"] is True and hv["no_claude_max_turns"] is True
    assert hv["customer_model_is_gpt_5_nano"] is True and hv["empty_customer_llm_args"] is True
    assert hv["harness_imports_tau3_retrieval"] is True and hv["harness_default_variant"] == "alltools"
    assert hv["job_variant_is_alltools"] is True and hv["retrieval_sha256"] == S.sha256_file(S.TR.__file__)


# ─────────────────────────────────────────────────────────────── alltools: the bridge, the grade, the machine check
def test_the_bridge_is_three_files_and_the_variant_is_alltools():
    assert S.VARIANT == S.TR.VARIANT == "alltools"
    names = [p.name for p in S.BRIDGE_FILES]
    assert names == ["tau2_cc_harness.py", "tau2_bundle.py", "tau3_retrieval.py"]
    assert len(set(p.parent for p in S.BRIDGE_FILES)) == 1 and all(p.is_file() for p in S.BRIDGE_FILES)
    assert S.BRIDGE_FILES[2] == (ROOT / "scripts" / "tau3_cc" / "tau3_retrieval.py").resolve()
    job = S.unit_job("task_005", 1, "cc-tau3-x", "/opt/tau3smoke/units/task_005_s1", root=DATA_ROOT)
    assert job["retrieval_variant"] == "alltools"


def test_the_upload_ships_and_sha_checks_every_bridge_file(tmp_path, monkeypatch):
    """Smoke.upload_files: every BRIDGE_FILES entry + the grader + the split go to /opt/tau3smoke and are sha-checked;
    a wrong sha on any of them refuses."""
    _need_data_root()
    _need_tau3_repo()
    answers = {}

    def hook(cmd, env):
        if cmd.startswith("sha256sum ") and "/opt/tau3smoke/bin/" in cmd:
            return 0, "".join(f"{answers.get(p, h)}  {p}\n" for p, h in want.items()), 0.0
        if "echo DATA_OK" in cmd:
            return 0, "DATA_OK\n", 0.0
        return None
    dle = FakeDle(run_hook=hook)
    sm = _smoke(tmp_path, dle)
    sm.m.sb, sm.m.sid = FakeSandbox(dle), "fakeid01-0000"
    want = {f"{S.VM_BIN}/{p.name}": S.sha256_file(p) for p in [*S.BRIDGE_FILES, S.GRADER]}
    want[S.VM_SPLIT] = S.sha256_file(S.SPLIT)
    sm.upload_files()
    ups = [c[2] for c in dle.calls if c[0] == "upload"]
    assert [u for u in ups if u.startswith(S.VM_BIN)] == [f"{S.VM_BIN}/{n}" for n in (
        "tau2_cc_harness.py", "tau2_bundle.py", "tau3_retrieval.py", S.GRADER.name)]
    assert sm.report["uploads"]["files"] == want
    answers[f"{S.VM_BIN}/tau3_retrieval.py"] = "0" * 64
    with pytest.raises(S.Refuse, match="does not verify"):
        sm.upload_files()
    sm.logfh.close()
    S.CS._LOG["fh"] = None


def test_the_grade_asks_for_alltools_and_finds_the_shells_tools():
    first = S.GRADE_PY.splitlines()[0]
    assert first == "DOMAIN, VARIANT = 'banking_knowledge', 'alltools'"
    assert 'grading={"domain": DOMAIN, "retrieval_variant": VARIANT}' in S.GRADE_PY
    compile(S.GRADE_PY, "GRADE_PY", "exec")
    cmd = S.grade_vm_cmd("/data/x/task.json", "/opt/x/snapshot")
    assert f" PATH={S.VM_PATH} " in cmd and all(f"-u {k}" in cmd for k in S.KEY_ENV_NAMES)
    assert S.VM_PATH.split(":")[:4] == ["/usr/local/sbin", "/usr/local/bin", "/usr/sbin", "/usr/bin"]


def _run_check(code: str, tmp_path: Path) -> tuple:
    """TAU3_CHECK_PY (or a variant) under the tau3-bench venv here, as cc_pool_host runs it on the machine: cd to a
    scratch dir, TAU2_DATA_DIR = the data root, no key, the host's srt / rg on PATH, every proxy a dead port."""
    env = G.child_env(DATA_ROOT, HOME=tmp_path, PATH=S.host_tool_path(), **DEAD_PROXY)
    r = subprocess.run([str(G.TAU3_PY), "-c", code], capture_output=True, text=True, env=env, cwd=str(tmp_path),
                       timeout=600)
    return r.returncode, r.stdout, r.stderr


def test_the_machine_check_compiles_and_carries_tau3_retrievals_constants():
    code = S.TAU3_CHECK_PY
    compile(code, "TAU3_CHECK_PY", "exec")
    head = code.splitlines()[:4]
    assert head[0] == "DOMAIN, VARIANT, PROBE_TASK = 'banking_knowledge', 'alltools', 'task_004'"
    assert head[1] == f"DENY_FIXED, CACHE_DIRNAME = {list(S.TR.DENY_READ_FIXED)!r}, '.embeddings_cache'"
    assert list(S.TR.DENY_READ_FIXED) == ["/opt", "/root", "/home", "/Users"]
    assert head[2] == "EMBEDDER_TYPE, EMBEDDER_MODEL, EMBED_DIM = 'openai', 'text-embedding-3-large', 3072"
    assert head[3] == "SRT_VERSION = '0.0.23'"
    # the same hardening as tau3_retrieval: srt settings' filesystem.denyRead += the data root + the fixed list
    assert 'deny = list(dict.fromkeys([str(root.resolve()), str(root), *DENY_FIXED]))' in code
    assert tuple(S.TR.deny_read_paths(DATA_ROOT)) == (str(DATA_ROOT.resolve()), *S.TR.DENY_READ_FIXED)
    assert 'fs["denyRead"] = list(dict.fromkeys([*fs.get("denyRead", []), *deny]))' in code
    assert "SM._required_binaries()" in code                    # tau2's own per-platform list


def test_the_machine_check_passes_here_and_reads_nothing_it_must_not(tmp_path):
    """TAU3_CHECK_PY under the tau3 venv on a macOS host (darwin: tau2 requires srt + rg; bwrap / socat are Linux-only):
    the cached embeddings load as [698, 3072], the hardened shell lists the KB and reads neither the task file nor
    /opt, the control reads the task file; OpenRouter is unreachable here (dead proxy) and only recorded."""
    _need_tau3_venv_and_srt()
    before = S.verify_data_root(DATA_ROOT)
    rc, out, err = _run_check(S.TAU3_CHECK_PY, tmp_path)
    assert rc == 0, (out[-2000:], err[-2000:])
    assert "TAU3_CHECK_FAIL" not in out
    res = S.parse_tail(out, "TAU3_CHECK")
    assert res["tau2_version"] == "1.0.1" and res["n_tasks"] == 97 and res["data_dir"] == str(DATA_ROOT)
    assert res.get("openrouter_models_status") is None and "openrouter_error" in res      # the dead proxy
    sb = res["sandbox"]
    assert sb["platform"] == "darwin" and sb["required"] == ["srt", "rg"] and sb["srt_package_version"] == "0.0.23"
    assert sb["which"]["srt"] and sb["which"]["rg"] and sb["which"]["node"]
    assert sb["path_inherited"] == S.host_tool_path()
    assert sb["inherited_path_finds"] == ["node", "rg", "srt"]
    emb = res["embeddings"]
    assert emb["shape"] == [698, 3072] and emb["documents"] == 698 and emb["ids_match"] is True
    assert emb["entry_in_root"] is True and emb["read_from_copy"] is True
    assert emb["entry"] in emb["files"] and "metadata.json" in emb["files"]
    sh = res["shell"]
    assert sh["ok"] == {"ls_lists_kb_documents": True, "task_file_unreadable": True, "opt_unreadable": True,
                        "control_reads_the_task_file": True}
    assert sh["deny_read_added"] == [str(DATA_ROOT), "/opt", "/root", "/home", "/Users"]
    assert sh["task_file"] == f"{str(DATA_ROOT).lstrip('/')}/tau2/domains/banking_knowledge/tasks/task_004.json"
    pr = sh["probes"]
    assert pr["cat_task"]["stdout"]["chars"] == 0 and pr["ls_opt"]["stdout"]["chars"] == 0
    assert pr["control_cat_task"]["stdout_chars"] == 80 and pr["control_cat_task"]["names_the_task"] is True
    assert "task_004" not in json.dumps({k: v for k, v in pr.items() if k != "control_cat_task"}).replace(
        sh["task_file"], "")                                     # nothing of the task's contents in the record
    assert S.verify_data_root(DATA_ROOT) == before                        # the data root (its cache included) is unchanged


def test_the_machine_check_catches_the_leak_without_the_deny_list(tmp_path):
    """The same check with the extra deny entries removed: the hardened shell reads the task file (the expected
    actions) and /opt, so it prints TAU3_CHECK_FAIL and NO TAU3_CHECK line, exit 1 -- the pool host's setup then
    finds no TAU3_CHECK and refuses the machine."""
    _need_tau3_venv_and_srt()
    line = '    deny = list(dict.fromkeys([str(root.resolve()), str(root), *DENY_FIXED]))\n'
    assert S.TAU3_CHECK_PY.count(line) == 1
    rc, out, err = _run_check(S.TAU3_CHECK_PY.replace(line, "    deny = []\n"), tmp_path)
    assert rc == 1, (out[-2000:], err[-2000:])
    assert not [ln for ln in out.splitlines() if ln.startswith("TAU3_CHECK ")]
    assert S.parse_tail(out, "TAU3_CHECK") == {}
    fail = S.parse_tail(out, "TAU3_CHECK_FAIL")
    assert fail["shell"]["ok"]["task_file_unreadable"] is False and fail["shell"]["ok"]["opt_unreadable"] is False
    assert fail["shell"]["ok"]["control_reads_the_task_file"] is True
    assert any("task_file_unreadable" in r for r in fail["reasons"])
    assert fail["embeddings"]["shape"] == [698, 3072]           # the other parts still ran and are recorded


def test_the_machine_check_fails_without_the_shells_binaries(tmp_path):
    """No srt on PATH (and no rg): (a) fails and so does (c) -- tau2's SandboxManager refuses to build -- so the
    check prints TAU3_CHECK_FAIL only."""
    _need_tau3_venv_and_srt()
    env = G.child_env(DATA_ROOT, HOME=tmp_path, PATH="/usr/bin:/bin", **DEAD_PROXY)
    r = subprocess.run([str(G.TAU3_PY), "-c", S.TAU3_CHECK_PY], capture_output=True, text=True, env=env,
                       cwd=str(tmp_path), timeout=600)
    assert r.returncode == 1 and S.parse_tail(r.stdout, "TAU3_CHECK") == {}
    fail = S.parse_tail(r.stdout, "TAU3_CHECK_FAIL")
    assert any("sandbox binaries missing" in x for x in fail["reasons"]), fail["reasons"]
    assert any(x.startswith("KB shell") for x in fail["reasons"]), fail["reasons"]


# ─────────────────────────────────────────────────────────────── secrets
def test_the_launch_command_never_holds_a_secret():
    cmd = S.launch_cmd("/opt/tau3smoke/units/task_005_s1", TOK, KEY)
    assert TOK not in cmd and KEY not in cmd and S.TOKEN_PREFIX not in cmd and S.OR_PREFIX not in cmd
    assert cmd.startswith("(setsid nohup env -u CLAUDE_CODE_OAUTH_TOKEN -u TAU3_OR_KEY ") and cmd.endswith("&)")
    assert '3<<<"$CLAUDE_CODE_OAUTH_TOKEN" 4<<<"$TAU3_OR_KEY"' in cmd
    assert "/opt/tau3/venv/bin/python /opt/tau3smoke/bin/tau2_cc_harness.py --job job.json" in cmd
    # the bridge's PATH is explicit: srt (/usr/local/bin) and rg / bwrap / socat / node (/usr/bin) for the KB shell
    assert f"-u TAU3_OR_KEY PATH={S.VM_PATH} PYTHON_DOTENV_DISABLED=1 " in cmd
    assert "PATH=/opt/tools/bin:/x " in S.launch_cmd("/opt/u", TOK, KEY, path="/opt/tools/bin:/x")
    with pytest.raises(S.Refuse):
        S.launch_cmd(f"/opt/x/{TOK}", TOK, KEY)
    with pytest.raises(S.Refuse):
        S.launch_cmd("/opt/x", TOK, "")


def test_the_redactor_and_the_scan(tmp_path):
    red = S.Redactor()
    red.add(TOK, "claude-token")
    red.add(KEY, "openrouter-key")
    out = red(f"a {TOK} b {KEY} c " + "sk-or-" + "v1-" + "abc123 d")
    assert TOK not in out and KEY not in out and "sk-or-" + "v1-abc" not in out
    (tmp_path / "a.txt").write_text("nothing here")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.log").write_text(f"leak {KEY}")
    res = S.secret_scan([tmp_path], {"openrouter-key": KEY, "claude-token": TOK})
    assert res["files"] == 2 and res["hits"] == [f"{tmp_path / 'sub' / 'b.log'}:openrouter-key"]


def test_a_downloaded_tar_cannot_place_a_link_or_escape(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in (("ok/file.txt", b"x"), ("../escape.txt", b"y"), ("/abs.txt", b"z")):
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
        ln = tarfile.TarInfo("ok/link")
        ln.type, ln.linkname = tarfile.SYMTYPE, "/etc/passwd"
        tf.addfile(ln)
    tgz = tmp_path / "x.tgz"
    tgz.write_bytes(buf.getvalue())
    refused = S.safe_extract(tgz, tmp_path / "out")
    assert set(refused) == {"/abs.txt", "../escape.txt", "ok/link"}
    assert [p.name for p in (tmp_path / "out").rglob("*") if p.is_file()] == ["file.txt"]


def test_the_data_root_tgz_extracts_to_its_manifest(tmp_path):
    _need_data_root()
    tgz = tmp_path / "d.tgz"
    tgz.write_bytes(S.data_tgz(DATA_ROOT))
    dest = tmp_path / "root"
    assert S.safe_extract(tgz, dest) == []
    assert S.verify_data_root(dest)["tasks"] == 97


# ─────────────────────────────────────────────────────────────── reading a unit
def _synthetic_unit(d: Path, *, customer: str = "I lost my card.") -> Path:
    tmpl = S.H.FIRST_MESSAGE_TEMPLATE
    first = tmpl.replace("{policy}", "THE POLICY").replace("{customer}", customer)
    (d / "out").mkdir(parents=True)
    (d / "snapshot" / "tau2").mkdir(parents=True)
    (d / "rc").write_text("0\n")
    home = d / "home"
    tr = home / ".claude" / "projects" / "-evo-work" / "sess-1.jsonl"
    tr.parent.mkdir(parents=True)
    tr.write_text("{}\n")
    (d / "out" / "harness.json").write_text(json.dumps({
        "exit_code": 0, "end": {"reason": "user_stop", "detail": None}, "where": "container cc-x",
        "agent_time": {"limit_s": 1200.0, "used_s": 42.5, "turns": 2, "turns_s": [30.0, 12.5]},
        "cc": {"tool_calls": 1, "id_sources": {"_meta": 1}, "mcp_servers": [{"name": "tau2", "status": "connected"}],
               "tau2_tools_listed": ["mcp__tau2__get_user"], "mcp_methods": ["initialize", "tools/call",
                                                                             "tools/list"],
               "session_id": "sess-1", "transcript": str(tr)}}))
    (d / "out" / "claude.stdin.jsonl").write_text(
        json.dumps({"type": "control_request", "request_id": "r1", "request": {"subtype": "initialize"}}) + "\n"
        + json.dumps({"type": "user", "message": {"role": "user", "content": first}}) + "\n")
    tu = {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "toolu_01abc", "name": "mcp__tau2__get_user", "input": {}}]}}
    own = {"type": "assistant", "message": {"content": [
        {"type": "tool_use", "id": "toolu_01skill", "name": "Skill", "input": {"skill": "x"}}]}}
    (d / "out" / "claude.stream.jsonl").write_text("".join(json.dumps(e) + "\n" for e in (
        tu, tu, own, {"type": "result", "duration_ms": 30000})))          # a repeated event counts once
    msgs = [{"role": "assistant", "content": "Hi! How can I help you today?"},
            {"role": "user", "content": customer},
            {"role": "assistant", "content": None, "generation_time_seconds": 30.0,
             "tool_calls": [{"id": "toolu_01abc", "name": "get_user", "arguments": {}, "requestor": "assistant"}]},
            {"role": "tool", "id": "toolu_01abc", "content": "{}", "requestor": "assistant"},
            {"role": "assistant", "content": "Done.", "generation_time_seconds": 12.5},
            {"role": "user", "content": "###STOP###"}]
    (d / "snapshot" / "tau2" / "simulation.json").write_text(json.dumps({"messages": msgs,
                                                                         "termination_reason": "user_stop"}))
    (d / "snapshot" / "tau2" / "run.json").write_text(json.dumps({
        "tau2": {"version": "1.0.1", "commit": "363133a"}, "data_dir": str(DATA_ROOT),
        "retrieval_variant": "alltools",
        "harness": {"first_message_sha256": hashlib.sha256(tmpl.encode()).hexdigest(),
                    "retrieval": {"variant": "alltools", "embedder": {"key": True}}}}))
    return d


def test_a_collected_unit_is_summarized_and_checked(tmp_path):
    u = _synthetic_unit(tmp_path / "u")
    s = S.summarize_unit(u, container_dir=Path("/"))
    assert (s["exit_code"], s["harness_exit_code"], s["termination"]) == (0, 0, "user_stop")
    assert s["agent_seconds"] == 42.5 and s["agent_seconds_tau2_generation"] == 42.5
    assert s["agent_seconds_cc_results"] == 30.0
    assert (s["tool_calls"], s["tool_calls_tau2"], s["tool_call_ids_toolu"]) == (1, 1, 1)
    assert s["calls_paired_with_results"] is True
    assert s["first_message"]["matches_template"] is True
    assert s["first_message"]["customer_part_is_the_customers_first_message"] is True
    assert s["first_message_template_is_the_harness"] is True
    assert s["transcript"] and Path(s["transcript"]).is_file()
    assert s["tool_calls_by_name"] == {
        "stream": {"Skill": 1, "get_user": 1}, "tau2": {"get_user": 1},
        "retrieval": {"KB_search_bm25": 0, "KB_search_dense": 0, "shell": 0, "domain_tools": 1},
        "claude_code_own_tools": ["Skill"]}
    assert s["retrieval"] == {"variant": "alltools", "embedder": {"key": True}}    # run.json's (no harness.json key)
    assert s["run_json"]["retrieval_variant"] == "alltools"
    checks = S.pass_checks(s, {"score": 1.0}, {"score": 1.0}, [], str(DATA_ROOT))
    assert all(checks.values()), {k: v for k, v in checks.items() if not v}
    assert checks["run_json_variant_alltools"] is True
    assert not S.pass_checks(dict(s, run_json=dict(s["run_json"], retrieval_variant="bm25_grep")), {"score": 1.0},
                             {"score": 1.0}, [], str(DATA_ROOT))["run_json_variant_alltools"]
    assert not S.pass_checks(s, {"score": 1.0}, {"score": 0.0}, [], str(DATA_ROOT))["grades_equal"]
    assert not S.pass_checks(s, {"score": 1.0}, {"raised": "X"}, [], str(DATA_ROOT))["graded_on_mac"]
    assert not S.pass_checks(s, {"score": 1.0}, {"score": 1.0}, ["f:openrouter-key"],
                             str(DATA_ROOT))["no_secret_in_files"]
    other = S.first_message_check("You are someone else.\n", "hi")
    assert other["matches_template"] is False and other["sha256"]


# ─────────────────────────────────────────────────────────────── the run's lifecycle with a fake Daytona
class FakeFS:
    def __init__(self, dle):
        self.dle = dle

    def upload_file(self, src, dst, timeout=None):
        self.dle.calls.append(("upload", str(src), dst))

    def download_file(self, src, dst):
        self.dle.calls.append(("download", src, str(dst)))


class FakeSandbox:
    def __init__(self, dle):
        self.id = "fakeid01-0000"
        self.fs = FakeFS(dle)


class FakeDle:
    PREFIX = "paper-ssb-0924-evo-"

    def __init__(self, create_error=None, run_hook=None):
        self.calls, self.rows, self.deleted = [], [], []
        self.create_error, self.run_hook = create_error, run_hook

    def create(self, d, params, requested, tag, timeout=3600, on_logs=None):
        self.calls.append(("create", params.name, tag, params.domain_allow_list, params.ttl_minutes))
        if on_logs:
            on_logs("step 1/10")
        if self.create_error is not None:
            return None, "fakeid01-seen", self.create_error, 2.0
        return FakeSandbox(self), "fakeid01-0000", None, 1.0

    def readback(self, sb):
        return {"domain_allow_list": "openrouter.ai,api.anthropic.com", "network_block_all": False, "cpu": 8,
                "memory": 32, "disk": 40, "snapshot_ref": "snap-x"}

    def ledger(self, **kw):
        self.rows.append(kw)

    def delete_by_id(self, d, sid, tag):
        self.deleted.append(sid)
        return True

    def list_all_ids(self, d):
        return []

    def run(self, sb, cmd, timeout=600, env=None):
        self.calls.append(("run", cmd, sorted(env) if env else None))
        if self.run_hook is not None:
            r = self.run_hook(cmd, env)
            if r is not None:
                return r
        return 0, "", 0.0


class FakeParams:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class FakeImage:
    @staticmethod
    def from_dockerfile(p):
        return ("image", p)


class FakeDeps(S.Deps):
    def __init__(self, dle):
        self.dle = dle
        self.sdk_used = []

    def client(self):
        return object()

    def sdk(self):
        return FakeParams, FakeImage, (lambda **kw: dict(kw))

    def credential(self, token_file=None):
        return TOK, S.cc_account.Credential(S.cc_account.TOKEN_FILE, Path("/nonexistent/token-file"), "declared"), \
            "/nonexistent/token-file"

    def openrouter_key(self):
        return KEY

    def sleep(self, s):
        pass


class FakeBox:
    def __init__(self):
        self.status, self.wiped, self.downs, self.rec = "ready", False, 0, {}
        self.cdir = "/opt/evoskill-evolver/c/fake"

    def down(self):
        self.downs += 1
        return {"verified": True, "scan": {}}

    def wipe(self):
        self.wiped = True
        return True

    def save(self):
        pass


def _smoke(tmp_path, dle, *extra):
    a = S.build_parser().parse_args(["run", "--out", str(tmp_path / "out"), "--data-root", str(DATA_ROOT), *extra])
    stamp = time.strftime("%Y%m%dT%H%M%S")
    plan = S.make_plan(a, stamp)
    return S.Smoke(a, FakeDeps(dle), tmp_path / "out" / stamp, stamp, plan)


@pytest.mark.parametrize("exc, rc", [(S.Refuse("egress"), 4), (SystemExit(143), 6), (KeyboardInterrupt(), 6),
                                     (RuntimeError("boom"), 6)])
def test_the_machine_is_deleted_when_a_step_fails(tmp_path, monkeypatch, exc, rc):
    _need_data_root()
    _need_tau3_repo()
    dle = FakeDle()
    sm = _smoke(tmp_path, dle)

    def fail(self):
        raise exc
    monkeypatch.setattr(S.Smoke, "machine_checks", fail)
    assert sm.run() == rc
    assert dle.deleted == ["fakeid01-0000"]
    create = [c for c in dle.calls if c[0] == "create"][0]
    assert create[1].startswith("paper-ssb-0924-evo-tau3smoke-") and create[3] == "openrouter.ai,api.anthropic.com"
    rep = json.loads((sm.out / "SMOKE_REPORT.json").read_text())
    assert rep["rc"] == rc and rep["machine_delete"]["verified_gone"] is True
    assert (sm.out / "machine" / "build_logs.txt").read_text().strip() == "step 1/10"
    assert not S.secret_scan([sm.out / "driver.log", sm.out / "machine"], {"t": TOK, "k": KEY})["hits"]


def test_a_failed_create_whose_id_was_seen_is_deleted(tmp_path):
    _need_data_root()
    _need_tau3_repo()
    dle = FakeDle(create_error=RuntimeError("org quota"))
    sm = _smoke(tmp_path, dle)
    assert sm.run() == 4
    assert dle.deleted == ["fakeid01-seen"]


def test_an_interrupt_during_the_build_still_finds_the_machine_by_name(tmp_path, monkeypatch):
    _need_data_root()
    _need_tau3_repo()
    dle = FakeDle()

    def interrupted(d, params, requested, tag, timeout=3600, on_logs=None):
        dle.calls.append(("create", params.name, tag))
        raise KeyboardInterrupt()                  # Ctrl-C while the image builds: no id was ever returned
    monkeypatch.setattr(dle, "create", interrupted)
    looked = []

    class Client:
        def get(self, name):
            looked.append(name)
            return type("SB", (), {"id": "fakeid01-late"})()
    sm = _smoke(tmp_path, dle)
    monkeypatch.setattr(sm.deps, "client", lambda: Client())
    assert sm.run() == 6
    assert looked and looked[0].startswith("paper-ssb-0924-evo-tau3smoke-")
    assert dle.deleted == ["fakeid01-late"]


def test_sigterm_during_a_rollout_tears_the_containers_down_then_deletes(tmp_path, monkeypatch):
    _need_data_root()
    _need_tau3_repo()
    seen = {"polls": 0}

    def hook(cmd, env):
        if 'print("POLL "' in cmd:
            seen["polls"] += 1
            if seen["polls"] == 1:
                raise SystemExit(143)             # what cmd_run's SIGTERM handler raises
        return None
    dle = FakeDle(run_hook=hook)
    sm = _smoke(tmp_path, dle)
    for step in ("machine_checks", "install", "upload_files", "tau3_check"):
        monkeypatch.setattr(S.Smoke, step, lambda self: None)
    boxes = []

    def fake_setup(self, u, stop_s):
        u.hard_stop_s, u.box = stop_s, FakeBox()
        boxes.append(u.box)
        u.status, u.t_launch, u.job = "running", time.time(), {"task_file": "x"}
    monkeypatch.setattr(S.Smoke, "setup_and_launch", fake_setup)
    assert sm.run() == 6
    assert len(boxes) == 2 and all(b.downs >= 1 and b.wiped for b in boxes)
    assert dle.deleted == ["fakeid01-0000"]
    assert [u.status for u in sm.units] == ["torn_down", "torn_down"]
    assert seen["polls"] == 1


def test_the_launch_exec_alone_carries_the_secrets(tmp_path, monkeypatch):
    """setup_and_launch: the secrets travel only as the env of the ONE launch exec, never in a command or a file."""
    _need_data_root()
    _need_tau3_repo()

    def hook(cmd, env):
        if cmd.startswith("sha256sum") and "job.json" in cmd:
            p = [c for c in dle.calls if c[0] == "upload" and c[2].endswith("/job.json")][-1][1]
            return 0, hashlib.sha256(Path(p).read_bytes()).hexdigest() + "\n", 0.0
        return None
    dle = FakeDle(run_hook=hook)
    sm = _smoke(tmp_path, dle)
    sm.secrets()
    sm.m.sb, sm.m.sid = FakeSandbox(dle), "fakeid01-0000"
    u = sm.units[0]

    class Box(FakeBox):
        def prepare(self):
            pass

        def up(self, init_root):
            self.init_root = init_root

        def probe(self):
            pass
    monkeypatch.setattr(S, "SmokeBox", lambda *a, **k: Box())
    sm.setup_and_launch(u, 3600.0)
    runs = [c for c in dle.calls if c[0] == "run"]
    with_env = [c for c in runs if c[2]]
    assert len(with_env) == 1 and with_env[0][2] == ["CLAUDE_CODE_OAUTH_TOKEN", "TAU3_OR_KEY"]
    assert "setsid nohup" in with_env[0][1] and u.box.init_root is True
    assert not [c for c in runs if TOK in c[1] or KEY in c[1]]
    job = json.loads((u.local / "job.json").read_text())
    assert job["container"]["cid"] == u.cid and job["token_fd"] == 3 and job["key_fd"] == 4
    assert not S.secret_scan([sm.out], {"t": TOK, "k": KEY})["hits"]
    sm.logfh.close()
    S.CS._LOG["fh"] = None                              # cc_solver's log handle (Smoke.finish resets it otherwise)


# ─────────────────────────────────────────────────────────────── dry run
def test_a_dry_run_reaches_nothing_and_writes_nothing(tmp_path, monkeypatch, capsys):
    _need_data_root()
    _need_tau3_repo()

    class Boom:
        def __init__(self, *a, **k):
            raise AssertionError("a dry run built the Daytona / secrets side")
    monkeypatch.setattr(S, "Deps", Boom)
    monkeypatch.setattr(S.CS, "read_token", lambda *a, **k: pytest.fail("a dry run read a token"))
    monkeypatch.setattr(S.CS, "read_credential", lambda *a, **k: pytest.fail("a dry run read a credential"))
    monkeypatch.setattr(S.cc_account, "check_token_file", lambda *a, **k: pytest.fail("a dry run checked a token file"))
    monkeypatch.setattr(S.ORK, "openrouter_key", lambda *a, **k: pytest.fail("a dry run read the OpenRouter key"))
    before = sorted(p.name for p in S.OUT_ROOT.iterdir()) if S.OUT_ROOT.is_dir() else None
    out = tmp_path / "out"
    assert S.main(["run", "--dry-run", "--out", str(out), "--data-root", str(DATA_ROOT)]) == 0
    assert not out.exists()
    assert (sorted(p.name for p in S.OUT_ROOT.iterdir()) if S.OUT_ROOT.is_dir() else None) == before
    text = capsys.readouterr().out
    plan = json.loads(text[:text.rindex("DRY RUN")])
    assert plan["tasks"] == S.load_split()["train"][:2] and len(plan["units"]) == 2
    assert plan["allow_list"] == "openrouter.ai,api.anthropic.com"
    assert "<TOKEN>" not in plan["launch_cmd_shape"] and S.TOKEN_PREFIX not in text and S.OR_PREFIX not in text
    assert plan["units"][0]["claude_argv"][2] == "enter"
    assert [u["job"]["retrieval_variant"] for u in plan["units"]] == ["alltools", "alltools"]
    assert plan["retrieval"]["variant"] == "alltools" and plan["bridge_path"] == S.VM_PATH
    assert plan["retrieval"]["shell_deny_read"] == [str(DATA_ROOT), "/opt", "/root", "/home", "/Users"]
    assert [Path(b).name for b in plan["upload"]["bridge"]] == ["tau2_cc_harness.py", "tau2_bundle.py",
                                                                "tau3_retrieval.py"]
    assert plan["image"]["srt"]["version"] == "0.0.23" and plan["image"]["copy_sources"][2] == \
        "tau3_ctx/srt/sandbox-runtime"
    assert f"PATH={S.VM_PATH} " in plan["launch_cmd_shape"]


def test_a_dry_run_of_the_named_tasks_plans_alltools_jobs(capsys):
    _need_data_root()
    _need_tau3_repo()
    assert S.main(["run", "--dry-run", "--tasks", "task_004,task_005", "--sample", "1", "--data-root", str(DATA_ROOT)]) == 0
    text = capsys.readouterr().out
    plan = json.loads(text[:text.rindex("DRY RUN")])
    assert plan["tasks"] == ["task_004", "task_005"] and [u["unit"] for u in plan["units"]] == ["task_004_s1",
                                                                                               "task_005_s1"]
    assert {u["job"]["retrieval_variant"] for u in plan["units"]} == {"alltools"}


# ─────────────────────────────────────────────────────────────── end to end, locally
WRAPPER = r'''
import hashlib, json, os, sys
cfg = json.loads(open(os.environ["SMOKE_TEST_CFG"]).read())


def is_open(fd):
    try:
        os.fstat(fd)
        return True
    except OSError:
        return False


rec = {"env_names": sorted(os.environ), "fd3_open": is_open(3), "fd4_open": is_open(4), "argv": sys.argv[1:],
       "secret_in_env": any(hashlib.sha256(v.encode()).hexdigest() in (cfg["tok_sha"], cfg["key_sha"])
                            for v in os.environ.values())}
sys.path[:0] = [cfg["tests_dir"], cfg["harness_dir"]]
import tau3_golden as G
import tau2_cc_harness as H
from tau2.data_model.message import UserMessage
from tau2.registry import registry
from tau2.user.user_simulator_base import HalfDuplexUser, UserState


class Customer(HalfDuplexUser):
    def __init__(self, tools=None, instructions=None, llm=None, llm_args=None):
        super().__init__(instructions=instructions, tools=tools)
        self.n = 0

    def get_init_state(self, message_history=None):
        return UserState(system_messages=[], messages=list(message_history or []))

    def generate_next_message(self, message, state):
        state.messages.append(message)
        self.n += 1
        out = UserMessage(role="user", content=G.OPENING if self.n == 1 else G.STOP, cost=0.0)
        state.messages.append(out)
        return out, state


registry.register_user(Customer, cfg["customer"])
code = H.main(sys.argv[1:])
rec["harness_file"] = H.__file__
rec["retrieval_file"] = sys.modules["tau3_retrieval"].__file__
import litellm
rec["litellm_key_sha256"] = hashlib.sha256((litellm.openrouter_key or "").encode()).hexdigest()
rec["code"] = code
open(cfg["rec"], "w").write(json.dumps(rec))
sys.exit(code)
'''


def test_the_drivers_launch_and_job_drive_the_real_harness_end_to_end(tmp_path):
    """The driver's own launch command (no setsid) and job (container mode) start the real harness; evoctl is the
    fake's --evoctl stand-in, claude the fake, the customer scripted; then the driver's summary, checks and host grade.
    The bridge runs from a directory holding exactly the three BRIDGE_FILES, as /opt/tau3smoke/bin on the machine
    (the harness imports tau3_retrieval from its own dir), with retrieval alltools: the KB shell's srt / rg come from
    the launch command's PATH (the host's tool dirs in front of VM_PATH), the document embeddings from the data root's
    cache, and every proxy is a dead port (nothing reaches the network)."""
    _need_tau3_venv_and_srt()
    task = "task_005"                                   # train; three agent actions, DB basis
    tf = S.task_file(task, DATA_ROOT)
    raw = json.loads(tf.read_text())
    unit, evo_work, home = tmp_path / "unit", tmp_path / "evo_work", tmp_path / "home"
    for p in (unit / "snapshot", evo_work, home, tmp_path / "bridge_home"):
        p.mkdir(parents=True)
    acts = [{"requestor": a["requestor"], "name": a["name"], "arguments": a.get("arguments") or {},
             "id": G.call_id(raw["id"], i, a["requestor"])}
            for i, a in enumerate((raw.get("evaluation_criteria") or {}).get("actions") or [])]
    assert acts and all(a["requestor"] == "assistant" for a in acts)
    script = tmp_path / "script.json"
    script.write_text(json.dumps({"log": str(tmp_path / "fake_log.json"), "session_id": "sess-smoke-e2e",
                                  "actions": acts, "to_user": G.TO_USER, "closing": G.CLOSING,
                                  "report_cwd": "/evo/work"}))
    emap = tmp_path / "evoctl_map.json"
    emap.write_text(json.dumps({"base_env": {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": "/nonexistent"},
                                "dirs": {"/evo/work": str(evo_work)}, "log": str(tmp_path / "evoctl_log.json"),
                                "script": str(script)}))
    cid = S.make_cid(task, 1, "20261001T000000", salt="e2e0")
    job = S.unit_job(task, 1, cid, str(unit), root=DATA_ROOT)
    assert job["home"] == "/home/evo"
    job.update(home=str(home), user=CUSTOMER, close_grace_s=5.0, mid_turn_grace_s=0.5)
    job["container"]["evoctl"] = [str(G.TAU3_PY), str(FAKE), "--evoctl", str(emap)]
    S.validate_job(job)
    (unit / "job.json").write_text(json.dumps(job, indent=1))
    wrapper = tmp_path / "bridge_wrapper.py"
    wrapper.write_text(WRAPPER)
    bridge_bin = tmp_path / "bin"                       # what upload_files / install_bridge put on the machine
    bridge_bin.mkdir()
    for f in S.BRIDGE_FILES:
        (bridge_bin / f.name).write_bytes(f.read_bytes())
    assert sorted(p.name for p in bridge_bin.iterdir()) == ["tau2_bundle.py", "tau2_cc_harness.py", "tau3_retrieval.py"]
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"rec": str(tmp_path / "bridge_rec.json"), "tests_dir": str(TESTS),
                               "harness_dir": str(bridge_bin), "customer": CUSTOMER,
                               "tok_sha": hashlib.sha256(TOK.encode()).hexdigest(),
                               "key_sha": hashlib.sha256(KEY.encode()).hexdigest()}))
    assert job["retrieval_variant"] == "alltools"
    cmd = S.launch_cmd(str(unit), TOK, KEY, py=str(G.TAU3_PY), harness=str(wrapper), detached=False,
                       path=f"{S.host_tool_path()}:{S.VM_PATH}")
    env = G.child_env(DATA_ROOT, HOME=tmp_path / "bridge_home", SMOKE_TEST_CFG=cfg,
                      CLAUDE_CODE_OAUTH_TOKEN=TOK, TAU3_OR_KEY=KEY, **DEAD_PROXY)
    r = subprocess.run(["/bin/bash", "-c", cmd], env=env, capture_output=True, text=True, timeout=600)
    log = (unit / "bridge.log").read_text() if (unit / "bridge.log").is_file() else ""
    assert (unit / "rc").read_text().strip() == "0", f"rc {r.returncode}: {log[-3000:]}"
    br = json.loads((tmp_path / "bridge_rec.json").read_text())
    assert br["code"] == 0 and br["fd3_open"] and br["fd4_open"] and not br["secret_in_env"]
    assert br["harness_file"] == str(bridge_bin / "tau2_cc_harness.py")
    assert br["retrieval_file"] == str(bridge_bin / "tau3_retrieval.py")       # imported from the harness's own dir
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in br["env_names"] and "TAU3_OR_KEY" not in br["env_names"]
    assert br["litellm_key_sha256"] == hashlib.sha256(KEY.encode()).hexdigest()
    ev = json.loads((tmp_path / "evoctl_log.json").read_text())
    assert ev["problems"] == [] and ev["keep_fd3"] and ev["cid"] == cid and ev["cwd"] == "/evo/work"
    # the token's fd 3 and the listing's own directory fd (the lowest free number): an inherited key fd 4 would
    # have pushed the listing to 5
    assert ev["fds"] == [3, 4], ev["fds"]
    want = {f"{k}={v}" for k, v in {**H.CC_ENV, **H.TAU2_ENV}.items()}
    assert want <= set(ev["pairs"]) and "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR=3" in ev["pairs"]
    fk = json.loads((tmp_path / "fake_log.json").read_text())
    assert fk["argv_problems"] == [] and not fk.get("protocol_problems")
    assert fk["token_sha256"] == hashlib.sha256(TOK.encode()).hexdigest()
    assert not [k for k, v in fk["env"].items() if v in (TOK, KEY) or k in ("CLAUDE_CODE_OAUTH_TOKEN", "TAU3_OR_KEY")]
    s = S.summarize_unit(unit, container_dir=Path("/"))
    assert (s["exit_code"], s["harness_exit_code"], s["termination"]) == (0, 0, "user_stop")
    assert s["tool_calls"] == len(acts) == s["tool_calls_tau2"] == s["tool_call_ids_toolu"]
    assert s["run_json"]["retrieval_variant"] == "alltools"
    rec = s["retrieval"]
    assert rec and rec["variant"] == "alltools" and rec["embedder"]["key"] is True and rec["cache"]["readonly"] is True
    assert rec["shell_deny_read"] == [str(DATA_ROOT), "/opt", "/root", "/home", "/Users"]
    assert S.TOKEN_PREFIX not in json.dumps(rec) and S.OR_PREFIX not in json.dumps(rec)
    names = [a["name"] for a in acts]
    assert s["tool_calls_by_name"]["tau2"] == {n: names.count(n) for n in names}
    assert s["tool_calls_by_name"]["retrieval"]["domain_tools"] == len(acts)
    listed = (json.loads((unit / "out" / "harness.json").read_text()).get("cc") or {}).get("tau2_tools_listed") or []
    assert {f"mcp__tau2__{t}" for t in ("KB_search_bm25", "KB_search_dense", "shell")} <= set(listed), listed
    assert s["tau2_tools_listed_n"] == len(listed) == 17          # alltools' 17 tools, listed to Claude Code
    assert s["first_message"]["matches_template"] is True
    assert s["first_message"]["customer_part_is_the_customers_first_message"] is True
    gm = S.grade_local(tf, unit / "snapshot")
    assert gm == {"score": 1.0, "type": "float"}, gm
    checks = S.pass_checks(s, gm, gm, [], str(DATA_ROOT))
    assert all(checks.values()), {k: v for k, v in checks.items() if not v}
    assert not S.secret_scan([tmp_path], {"t": TOK, "k": KEY})["hits"]
