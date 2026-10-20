"""src/orchestrator/infra.py + <run>/INFRA.json: the operator-owned infrastructure of a run (models, pins, sandboxes,
the grader's timeouts, the benchmark data and its repaired-file overlay) and agree(), which compares it with every
other statement of the same facts.

OFFLINE: no Daytona, no model call, no network, no LibreOffice. Every test builds its own run dir (SPEC / RUNTIME /
INFRA of a Claude Code solver run, the report's configuration) in tmp_path, from the constants the repository holds
today, plus a temp MIRROR of the repo files agree() reads, so a planted disagreement never touches the real tree. The
token file is a temp stand-in (a guard asserts agree() never opens it) and the data root and overlay are small temp
trees.
"""
from __future__ import annotations

import builtins
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.orchestrator import claude_pins, data_overlay, infra  # noqa: E402

RECALC, UNO = infra.RECALC_TIMEOUT_ENV, infra.DATATABLE_UNO_TIMEOUT_ENV
OVL = infra.DATA_OVERLAY_ENV
LEG_DOCKERFILE, CLAUDE_DOCKERFILE = "scripts/daytona/Dockerfile.ssb_leg", "scripts/daytona/Dockerfile.ssb_evolver"
CTL = "scripts/daytona/evolver_container_ctl.py"
MIRROR = (infra.LEG_DRIVER_BASE, infra.LEG_DRIVER_LOOP, infra.LEG_LIB_LOOP, infra.EVOLVER_MODULE,
          infra.PINS_MODULE, infra.RUNNER, infra.CC_SOLVER_LIB, LEG_DOCKERFILE, CLAUDE_DOCKERFILE, CTL)
MODEL = "claude-sonnet-5-5"
RUN = "ssbv2_cc_fixture"
# What a shell left over from an OpenRouter solver run would carry.
OPENROUTER_SHELL = {"EVOSKILL_SOLVER_MODEL": "deepseek/deepseek-v4.1-flash",
                    "EVOSKILL_SOLVER_PROVIDER_ORDER": "deepinfra", "EVOSKILL_SOLVER_PROVIDER_ONLY": "deepinfra",
                    "EVOSKILL_SOLVER_ALLOW_FALLBACKS": "0", "EVOSKILL_SOLVER_REQUIRE_PARAMETERS": "1",
                    "EVOSKILL_SOLVER_PROVIDER_QUANTIZATIONS": "fp8", "EVOSKILL_SOLVER_EXPECT_PROVIDER": "DeepInfra",
                    "EVOSKILL_SOLVER_EXPECT_MODEL": "deepseek/deepseek-v4.1-flash-20260910"}


def C(rel, *names) -> dict:
    """Constants of a repo file, read the way agree() reads them."""
    return infra._constants(ROOT / rel, names)


def sub(path: Path, old: str, new: str) -> None:
    s = path.read_text()
    assert s.count(old) == 1, (path, old)
    path.write_text(s.replace(old, new))


def plant(path: Path, line: str) -> None:
    """Append one top-level assignment: agree() reads the LAST assignment of a name, so this restates it."""
    path.write_text(path.read_text() + "\n" + line + "\n")


def edit_json(path: Path, fn) -> None:
    d = json.loads(path.read_text())
    fn(d)
    path.write_text(json.dumps(d, indent=1))


def _set(path, value):
    def f(d):
        *head, last = path.split(".")
        for h in head:
            d = d[h]
        d[last] = value
    return f


def _del(path):
    def f(d):
        *head, last = path.split(".")
        for h in head:
            d = d[h]
        del d[last]
    return f


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ══ the fixture: a Claude Code solver run built from the repository's own constants ═════════════
def make_overlay(tmp: Path, *, src_null: bool = True) -> types.SimpleNamespace:
    """A data root with two input workbooks (stand-ins), a patched copy of one of them, and its PATCHES.json in the
    shipped shape: dst relative to the PATCHES.json directory, src null (the run's data root) unless asked."""
    root = tmp / "SpreadsheetBench-v2" / "data" / "spreadsheetbench-v2"
    rel = "Financial_Model/spreadsheet/06_x/06_01_input.xlsx"
    (root / rel).parent.mkdir(parents=True)
    (root / rel).write_bytes(b"canonical bytes")
    (root / "Debugging" / "a.xlsx").parent.mkdir(parents=True)
    (root / "Debugging" / "a.xlsx").write_bytes(b"untouched")
    odir = tmp / "overlay"
    (odir / "spreadsheetbench-v2" / rel).parent.mkdir(parents=True)
    (odir / "spreadsheetbench-v2" / rel).write_bytes(b"repaired bytes")
    pj = odir / "PATCHES.json"
    pj.write_text(json.dumps({"src": None if src_null else str(root), "dst": "spreadsheetbench-v2",
                              "files": [{"task": "Financial_Model/06_01", "path": rel,
                                         "sha256_before": sha(root / rel),
                                         "sha256_after": sha(odir / "spreadsheetbench-v2" / rel)}]}, indent=1))
    return types.SimpleNamespace(root=root, pj=pj, rel=rel, canonical=root / rel,
                                 patched=odir / "spreadsheetbench-v2" / rel)


def cc_infra(run: str, token, ov) -> dict:
    ev = C(infra.EVOLVER_MODULE, "ALLOW_DOMAINS", "RESOURCES", "BASE_DOCKERFILE_SHA", "CC_VERSION",
           "CC_LINUX_X64_SHA")
    lb = C(infra.LEG_DRIVER_BASE, "RESOURCES", "ALLOW_DOMAINS")
    exp = C(infra.LEG_DRIVER_LOOP, "EXPECTED")["EXPECTED"]
    leg_sha = infra.sha256_file(ROOT / LEG_DOCKERFILE)
    tf = str(token) if token else None
    return {
        "schema": infra.SCHEMA, "run": run,
        "solver": {"model": MODEL, "gateway": infra.GATEWAY_ANTHROPIC_OAUTH, "key_file": tf,
                   "key_variable": "CLAUDE_CODE_OAUTH_TOKEN", "provider_order": [], "provider_only": [],
                   "allow_fallbacks": False, "require_parameters": False, "quantizations": [],
                   "expect_provider": "firstParty", "expect_served_model": MODEL, "reasoning_effort": "none",
                   "harness": infra.HARNESS_CLAUDE_CODE},
        "evolver": {"model": MODEL, "effort": "default", "harness": "claude_cli", "host": "daytona",
                    "sandbox": True, "timeout_s": 7200},
        "ao": {"model": MODEL, "effort": "default", "host": "daytona"},
        "claude_code": {"version": ev["CC_VERSION"], "linux_x64_sha256": ev["CC_LINUX_X64_SHA"], "token_file": tf,
                        "no_model_fallback": True},
        "claude_sandbox": {"dockerfile": CLAUDE_DOCKERFILE,
                           "dockerfile_sha256": infra.sha256_file(ROOT / CLAUDE_DOCKERFILE),
                           "base_dockerfile_sha256": ev["BASE_DOCKERFILE_SHA"], "snapshot_ref": "daytona-evo:daytona",
                           "libreoffice": exp["libreoffice"], "resources": ev["RESOURCES"],
                           "egress": ev["ALLOW_DOMAINS"].split(","), "ttl_minutes": 180},
        "leg_sandbox": {"host": "daytona", "dockerfile": LEG_DOCKERFILE, "dockerfile_sha256": leg_sha,
                        "image_id": "x86daytona-" + leg_sha[:12], "snapshot_ref": exp["snapshot_ref"],
                        "libreoffice": exp["libreoffice"], "grading_openpyxl": exp["grading_openpyxl_prefix"],
                        "resources": lb["RESOURCES"], "egress": lb["ALLOW_DOMAINS"].split(","), "max_sandboxes": 6,
                        "concurrency": 30, "max_heavy_per_shard": 2, "ttl_minutes": 300, "shard_wall_min": 240,
                        "leg_timeout_s": 43200},
        "grader": {"recalc_timeout_s": 900, "datatable_uno_timeout_s": 900},
        "data": {"root": str(ov.root), "overlay_patches_json": str(ov.pj),
                 "overlay_patches_sha256": infra.sha256_file(ov.pj)},
        "guards": {"leg_spend_cap_usd": 40.0, "run_spend_cap_usd": 200.0, "min_account_balance_usd": 100.0},
        "other_models": {"user_sim": "none", "judge": "none"},
    }


def cc_spec(run: str, inf: dict, benchmark: str = "ssbv2") -> dict:
    return {"spec": {"name": run, "benchmark": benchmark, "solver_model": inf["solver"]["model"],
                     "solver_provider_order": [], "solver_allow_fallbacks": False,
                     "solver_harness": infra.HARNESS_CLAUDE_CODE, "evolver_model": inf["evolver"]["model"],
                     "evolver_harness": "claude_cli", "user_sim_model": "none", "judge_model": "none",
                     "agent_turns": infra._cc_constants(ROOT)["MAX_TURNS"], "cost_ceiling_usd": 200.0,
                     "n_samples": 3, "n_epochs": 3}}


def cc_runtime(inf: dict) -> dict:
    cs, ls, g = inf["claude_sandbox"], inf["leg_sandbox"], inf["guards"]
    rt = {"leg_script": "scripts/" + infra.LEG_SCRIPT_CC, "prompt_author_model": inf["ao"]["model"],
          "prompt_mode": "template", "evolver_host": "daytona", "evolver_sandbox": True,
          "evolver_timeout_s": inf["evolver"]["timeout_s"], "leg_timeout_s": ls["leg_timeout_s"],
          "evolver_session": "stateful", "pipeline_val": True, "skip_last_train": True,
          "evolver_daytona": {"placement": "pool_container", "ttl_minutes": cs["ttl_minutes"],
                              "require_libreoffice": cs["libreoffice"], "require_snapshot_ref": cs["snapshot_ref"]},
          "claude": {"ao_model": inf["ao"]["model"], "evolver_model_from_spec": inf["evolver"]["model"],
                     "effort": inf["evolver"]["effort"],
                     "expect_claude_code_version": inf["claude_code"]["version"]},
          "daytona_leg": {"max_sandboxes": ls["max_sandboxes"], "concurrency": ls["concurrency"],
                          "max_heavy_per_shard": ls["max_heavy_per_shard"], "ttl_minutes": ls["ttl_minutes"],
                          "shard_wall_min": ls["shard_wall_min"], "require_snapshot_ref": ls["snapshot_ref"],
                          "require_libreoffice": ls["libreoffice"], "leg_spend_cap_usd": g["leg_spend_cap_usd"],
                          "run_spend_cap_usd": g["run_spend_cap_usd"],
                          "min_account_balance_usd": g["min_account_balance_usd"]}}
    if inf["claude_code"]["token_file"]:
        rt["evolver_daytona"]["oauth_token_file"] = inf["claude_code"]["token_file"]
    return rt


def build_run(tmp: Path, *, token: bool = True, benchmark: str = "ssbv2") -> types.SimpleNamespace:
    ov = make_overlay(tmp / "data")
    tok = None
    if token:
        tok = tmp / "secrets" / "token_standin"
        tok.parent.mkdir()
        tok.write_text("not a token")
        tok.chmod(0o600)
    rd = tmp / "runs" / RUN
    rd.mkdir(parents=True)
    inf = cc_infra(RUN, tok, ov)
    (rd / "INFRA.json").write_text(json.dumps(inf, indent=1))
    (rd / "SPEC.json").write_text(json.dumps(cc_spec(RUN, inf, benchmark), indent=1))
    (rd / "RUNTIME.json").write_text(json.dumps(cc_runtime(inf), indent=1))
    repo = tmp / "repo"
    for rel in MIRROR:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, repo / rel)
    return types.SimpleNamespace(rd=rd, repo=repo, token=tok, ov=ov, tmp=tmp)


@pytest.fixture
def fx(tmp_path, monkeypatch):
    for k in (infra.DATA_ROOT_ENV, infra.SSBV2_ROOT_ENV, OVL, "EVOSKILL_CC_VERSION"):
        monkeypatch.delenv(k, raising=False)
    return build_run(tmp_path)


def agree(fx, env=None, **kw):
    return infra.agree(fx.rd, {} if env is None else env, repo=fx.repo, **kw)


def fields(items) -> set:
    return {(d.source, d.field) for d in items}


def inf_of(fx) -> dict:
    return json.loads((fx.rd / "INFRA.json").read_text())


# ══ the fixture agrees ══════════════════════════════════════════════════════════════════════════
def test_the_fixture_validates_agrees_and_never_opens_the_token(fx, monkeypatch):
    inf = inf_of(fx)
    assert infra.validate(inf) == [] and infra.solver_is_claude_code(inf)
    banned = {str(fx.token)}
    real_open = io.open

    def guarded(file, *a, **k):
        if str(file) in banned:
            raise AssertionError(f"agree() opened the token {file}")
        return real_open(file, *a, **k)
    monkeypatch.setattr(io, "open", guarded)
    monkeypatch.setattr(builtins, "open", guarded)
    assert agree(fx) == []


def test_the_derived_environment_of_a_claude_code_run(fx):
    inf = inf_of(fx)
    assert infra.solver_env(inf) == {"EVOSKILL_SOLVER_MODEL": MODEL}
    assert not (set(infra.derived_env(inf)) & set(infra.SOLVER_ROUTING_KEYS))
    ce = infra.claude_env(inf)
    assert ce["EVOSKILL_CLAUDE_EFFORT"] == "default" and ce["EVOSKILL_AO_MODEL"] == MODEL
    assert ce[infra.CC_TOKEN_FILE_ENV] == str(fx.token), "the token file travels as its PATH only"
    assert claude_pins.effort(ce) is None, "'default' is claude_pins' opt-out: no --effort flag"
    assert infra.grader_env(inf) == {RECALC: "900", UNO: "900"}
    assert infra.data_env(inf) == {OVL: str(fx.ov.pj)}


# ══ schema: closed and complete ═════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("mutate, needle", [
    (_set("gpu", "a100"), "unknown top-level key 'gpu'"),
    (_set("solver.temperature", 0.2), "unknown key solver.temperature"),
    (_del("grader.recalc_timeout_s"), "missing key grader.recalc_timeout_s"),
    (_del("guards"), "missing block 'guards'"),
    (_del("run"), "missing key 'run'"),
    (_set("grader.recalc_timeout_s", "900"), "grader.recalc_timeout_s must be an integer"),
    (_set("grader.datatable_uno_timeout_s", 0), "grader.datatable_uno_timeout_s must be an integer"),
    (_set("leg_sandbox.max_sandboxes", True), "leg_sandbox.max_sandboxes must be an integer"),
    (_set("evolver.model", "opus[1m]"), "explicit claude-* model id"),
    (_set("ao.model", "claude-opus-5-5[1m]"), "explicit claude-* model id"),
    (_set("evolver.effort", "turbo"), "evolver.effort must be one of"),
    (_set("claude_sandbox.dockerfile_sha256", "abc"), "sha256"),
    (_set("schema", "evoskill-infra/2"), "schema must be one of"),
    (_set("solver.key_variable", "sk-or-v1-abc"), "environment variable NAME"),
    (_set("claude_code.token_file", "~/.cc_token"), "absolute path"),
    (_set("leg_sandbox.resources", {"cpu": 8, "memory": 32}), "exactly cpu, memory, disk"),
    (_set("ao.effort", "xhigh"), "one variable (EVOSKILL_CLAUDE_EFFORT"),
    (_set("guards.leg_spend_cap_usd", 300.0), "exceeds guards.run_spend_cap_usd"),
    (_set("leg_sandbox.image_id", "x86daytona-000000000000"), "derives it from the Dockerfile sha"),
    (_set("claude_sandbox.base_dockerfile_sha256", "0" * 64), "built FROM the leg image"),
    (_set("observability", {"solver_traces_phoenix": False}), "unknown top-level key 'observability'"),
    (_del("data"), "missing block 'data'"),
    (_del("data.overlay_patches_sha256"), "missing key data.overlay_patches_sha256"),
    (_set("data.mirror", "/x"), "unknown key data.mirror"),
    (_set("data.root", "SpreadsheetBench-v2/data"), "data.root must be an absolute path"),
    (_set("data.root", None), "data.root must be an absolute path"),
    (_set("data.overlay_patches_json", "PATCHES.json"), "an absolute path or null"),
    (_set("data.overlay_patches_sha256", "8ebc"), "sha256 hex digest or null"),
    (_set("data.overlay_patches_sha256", None), "must both be null (no overlay) or both be set"),
    (_set("data.overlay_patches_json", None), "must both be null (no overlay) or both be set"),
])
def test_schema_refuses(fx, mutate, needle):
    d = inf_of(fx)
    mutate(d)
    assert any(needle in p for p in infra.validate(d)), infra.validate(d)
    f = fx.tmp / "bad_INFRA.json"
    f.write_text(json.dumps(d))
    with pytest.raises(infra.InfraError) as ei:
        infra.load(f)
    assert any(needle in p for p in ei.value.problems)


def test_schema_allows_comment_keys_and_reports_every_problem_at_once(fx):
    d = inf_of(fx)
    d["_why"] = "comment"
    d["grader"]["_why"] = "comment"
    assert infra.validate(d) == []
    d["solver"]["temperature"] = 1
    d["guards"]["run_spend_cap_usd"] = "150"
    assert len(infra.validate(d)) == 2
    (fx.tmp / "INFRA.json").write_text("{not json")
    with pytest.raises(infra.InfraError, match="not readable JSON"):
        infra.load(fx.tmp / "INFRA.json")
    assert infra.load_run(fx.tmp / "nowhere") is None
    with pytest.raises(infra.InfraError, match="no "):
        infra.load(fx.tmp / "absent.json")


@pytest.mark.parametrize("mutate, needle", [
    (_set("solver.provider_order", ["deepinfra"]), "solver.provider_order must be [] for a Claude Code solver"),
    (_set("solver.provider_only", ["deepinfra"]), "solver.provider_only must be [] for a Claude Code solver"),
    (_set("solver.quantizations", ["fp8"]), "solver.quantizations must be [] for a Claude Code solver"),
    (_set("solver.gateway", "openrouter"), "requires solver.gateway 'anthropic_oauth'"),
    (_set("solver.harness", "direct"), "requires solver.harness 'claude_code'"),
    (_set("solver.key_file", "/elsewhere/.env"), "!= claude_code.token_file"),
    (_set("solver.expect_served_model", MODEL + "-20260901"), "!= solver.model"),
    (_set("solver.reasoning_effort", "high"), "solver.reasoning_effort must be 'none' for a Claude Code solver"),
    (_set("leg_sandbox.host", "local"), "leg_sandbox.host must be 'daytona' for a Claude Code solver"),
    (_set("evolver.effort", "none"), "evolver.effort must be one of"),      # only "default" spells the opt-out
    (_set("ao.effort", "high"), "one variable (EVOSKILL_CLAUDE_EFFORT"),
    (_set("solver.provider_order", "deepinfra"), "solver.provider_order must be a list"),
])
def test_claude_code_cross_rules_refuse(fx, mutate, needle):
    d = inf_of(fx)
    mutate(d)
    assert any(needle in p for p in infra.validate(d)), infra.validate(d)


def test_the_openrouter_shape_is_a_separate_gateway(fx):
    """gateway openrouter (a model behind OpenRouter) requires its pins and its key file; the two shapes never mix."""
    def openrouter(d):
        d["solver"].update(gateway="openrouter", harness="direct", provider_order=["deepinfra"],
                           provider_only=["deepinfra"], quantizations=["fp8"], key_file="/k/.env",
                           expect_served_model="deepseek/x-20260910", model="deepseek/x")
        return d
    assert infra.validate(openrouter(inf_of(fx))) == []
    d = openrouter(inf_of(fx))
    d["solver"]["quantizations"] = []
    assert any("solver.quantizations must be a non-empty list" in p for p in infra.validate(d))
    d = openrouter(inf_of(fx))
    d["solver"]["key_file"] = None
    assert any("solver.key_file must be an absolute path for gateway 'openrouter'" in p for p in infra.validate(d))
    d = openrouter(inf_of(fx))
    d["solver"]["harness"] = "claude_code"
    probs = infra.validate(d)
    assert any("requires solver.gateway 'anthropic_oauth'" in p for p in probs)
    assert any("must be [] for a Claude Code solver" in p for p in probs)


def test_effort_default_is_accepted_and_means_no_flag():
    assert infra._t_effort("default") is None and infra._t_effort("high") is None
    for bad in ("none", "off", "", "DEFAULT", None):
        assert infra._t_effort(bad), bad
    assert claude_pins.effort({"EVOSKILL_CLAUDE_EFFORT": "default"}) is None
    assert claude_pins.effort({}) == "high", "an unset variable still means high: INFRA must say 'default'"
    assert infra.same_env_value("EVOSKILL_CLAUDE_EFFORT", "default", " DEFAULT ")
    assert not infra.same_env_value("EVOSKILL_CLAUDE_EFFORT", "default", "high")


# ══ agreement ═══════════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("fname, mutate, want", [
    ("RUNTIME.json", lambda d: d.__setitem__("solver_env", {"EVOSKILL_SOLVER_PROVIDER_ORDER": "deepinfra"}),
     ("RUNTIME.json", "solver_env")),
    ("RUNTIME.json", lambda d: d.__setitem__("solver_env", {}), ("RUNTIME.json", "solver_env")),
    ("RUNTIME.json", lambda d: d.__setitem__("leg_script", "/x/scripts/ssbv2_leg_daytona.sh"),
     ("RUNTIME.json", "leg_script")),
    ("RUNTIME.json", lambda d: d.pop("leg_script"), ("RUNTIME.json", "leg_script")),
    ("RUNTIME.json", lambda d: d["claude"].__setitem__("effort", "high"), ("RUNTIME.json", "claude.effort")),
    ("RUNTIME.json", lambda d: d.__setitem__("leg_timeout_s", 21600), ("RUNTIME.json", "leg_timeout_s")),
    ("RUNTIME.json", lambda d: d.__setitem__("evolver_timeout_s", 5400), ("RUNTIME.json", "evolver_timeout_s")),
    ("RUNTIME.json", lambda d: d.__setitem__("evolver_host", "local"), ("RUNTIME.json", "evolver_host")),
    ("RUNTIME.json", lambda d: d["evolver_daytona"].__setitem__("oauth_token_file", "/x/other_token"),
     ("RUNTIME.json", "evolver_daytona.oauth_token_file")),
    ("RUNTIME.json", lambda d: d["evolver_daytona"].__setitem__("require_snapshot_ref", "daytona-x"),
     ("RUNTIME.json", "evolver_daytona.require_snapshot_ref")),
    ("RUNTIME.json", lambda d: d["daytona_leg"].__setitem__("concurrency", 120),
     ("RUNTIME.json", "daytona_leg.concurrency")),
    ("RUNTIME.json", lambda d: d["daytona_leg"].__setitem__("require_snapshot_ref", "daytona-y"),
     ("RUNTIME.json", "daytona_leg.require_snapshot_ref")),
    ("RUNTIME.json", lambda d: d.__setitem__("prompt_author_model", "claude-opus-5-5"),
     ("RUNTIME.json", "prompt_author_model")),
    ("SPEC.json", lambda d: d["spec"].__setitem__("solver_harness", "direct"), ("SPEC.json", "solver_harness")),
    ("SPEC.json", lambda d: d["spec"].__setitem__("solver_model", "deepseek/deepseek-v4.1-flash"),
     ("SPEC.json", "solver_model")),
    ("SPEC.json", lambda d: d["spec"].__setitem__("agent_turns", 30),
     (f"{infra.CC_SOLVER_LIB} MAX_TURNS", "agent_turns")),
    ("SPEC.json", lambda d: d["spec"].__setitem__("evolver_model", "claude-opus-5"), ("SPEC.json", "evolver_model")),
    ("SPEC.json", lambda d: d["spec"].__setitem__("cost_ceiling_usd", 100.0), ("SPEC.json", "cost_ceiling_usd")),
    ("INFRA.json", lambda d: d.__setitem__("run", "some_other_run"), ("run dir", "name")),
    ("INFRA.json", lambda d: (d["solver"].__setitem__("model", "claude-sonnet-4-5"),
                              d["solver"].__setitem__("expect_served_model", "claude-sonnet-4-5")),
     ("SPEC.json", "solver_model")),        # the leg passes SPEC's model; cc_solver_lib.MODEL is only a default
    ("INFRA.json", lambda d: d["leg_sandbox"].__setitem__("leg_timeout_s", 21600), ("RUNTIME.json", "leg_timeout_s")),
    ("INFRA.json", lambda d: d["claude_sandbox"].__setitem__("ttl_minutes", 240),
     ("RUNTIME.json", "evolver_daytona.ttl_minutes")),
    ("INFRA.json", lambda d: d["leg_sandbox"].__setitem__("max_sandboxes", 12),
     ("RUNTIME.json", "daytona_leg.max_sandboxes")),
])
def test_a_single_disagreement_is_named(fx, fname, mutate, want):
    edit_json(fx.rd / fname, mutate)
    got = agree(fx)
    assert want in fields(got), [str(d) for d in got]


def test_the_runtime_check_names_exactly_the_solver_env_block(fx):
    edit_json(fx.rd / "RUNTIME.json",
              lambda d: d.__setitem__("solver_env", {"EVOSKILL_SOLVER_EXPECT_MODEL": "deepseek/x"}))
    got = {(d.source, d.field): d for d in agree(fx)}
    assert set(got) == {("RUNTIME.json", "solver_env")}
    assert "reads no OpenRouter pin" in got[("RUNTIME.json", "solver_env")].note


def test_an_absent_key_is_compared_after_its_consumers_default(fx):
    edit_json(fx.rd / "RUNTIME.json", lambda d: d["daytona_leg"].pop("leg_spend_cap_usd"))
    edit_json(fx.rd / "RUNTIME.json", lambda d: d.pop("prompt_author_model"))
    got = {(d.source, d.field): d for d in agree(fx)}
    defaults = C(infra.LEG_DRIVER_LOOP, "EXPECTED", "DEFAULTS")["DEFAULTS"]
    assert got[("RUNTIME.json", "daytona_leg.leg_spend_cap_usd")].found == defaults["leg_spend_cap_usd"]
    assert got[("RUNTIME.json", "prompt_author_model")].found == infra.LOOP_DEFAULTS["prompt_author_model"]


def test_the_token_file_may_be_left_out_of_the_runtime(fx):
    """evolver_daytona.oauth_token_file left out: the driver takes the run's one credential from the environment."""
    edit_json(fx.rd / "RUNTIME.json", lambda d: d["evolver_daytona"].pop("oauth_token_file"))
    assert agree(fx) == []


@pytest.mark.parametrize("rel, old, new, want", [
    (infra.CC_SOLVER_LIB, None, "MAX_TURNS = 40", (f"{infra.CC_SOLVER_LIB} MAX_TURNS", "agent_turns")),
    (infra.EVOLVER_MODULE, None, 'CC_VERSION = "2.1.999"', (infra.EVOLVER_MODULE, "CC_VERSION")),
    (infra.EVOLVER_MODULE, None, 'CC_LINUX_X64_SHA = "' + "0" * 64 + '"', (infra.EVOLVER_MODULE, "CC_LINUX_X64_SHA")),
    (infra.EVOLVER_MODULE, None, 'ALLOW_DOMAINS = "api.anthropic.com,github.com"',
     (infra.EVOLVER_MODULE, "ALLOW_DOMAINS")),
    (infra.LEG_DRIVER_BASE, None, 'ALLOW_DOMAINS = "openrouter.ai,x.io"',
     (f"{infra.LEG_DRIVER_BASE} ALLOW_DOMAINS", "ALLOW_DOMAINS")),
    (infra.LEG_DRIVER_BASE, None, 'KNOBS = {"EVOSKILL_RECALC_TIMEOUT_SECONDS": "180"}',
     (f"{infra.LEG_DRIVER_BASE} KNOBS", RECALC)),
    (infra.LEG_DRIVER_LOOP, None, 'EXPECTED = {"libreoffice": "LibreOffice 25.8.0.1"}',
     (f"{infra.LEG_DRIVER_LOOP} EXPECTED", "libreoffice")),
    (infra.PINS_MODULE, '"CLAUDE_CODE_NO_MODEL_FALLBACK": "1",', '',
     (infra.PINS_MODULE, "PIN_ENV_ALWAYS.CLAUDE_CODE_NO_MODEL_FALLBACK")),
    (LEG_DOCKERFILE, "FROM ", "# drift\nFROM ", ("disk", f"sha256({LEG_DOCKERFILE})")),
    (CLAUDE_DOCKERFILE, "FROM ", "# drift\nFROM ", ("disk", f"sha256({CLAUDE_DOCKERFILE})")),
])
def test_code_constants_that_run_are_compared(fx, rel, old, new, want):
    if old is None:
        plant(fx.repo / rel, new)
    else:
        sub(fx.repo / rel, old, new)
    got = agree(fx)
    assert want in fields(got), [str(d) for d in got]


@pytest.mark.parametrize("rel, line", [
    (infra.LEG_DRIVER_BASE, 'LEG_SETTINGS = {"EVOSKILL_SOLVER_EXPECT_MODEL": "deepseek/x-20261001"}'),
    (infra.LEG_LIB_LOOP, 'ENV_FILE = "/elsewhere/.env"'),
    (infra.RUNNER, 'THINKING_EFFORT = "high"'),
    (infra.CC_SOLVER_LIB, 'MODEL = "claude-sonnet-6"'),
])
def test_the_openrouter_instrument_and_the_model_default_are_not_compared(fx, rel, line):
    """LEG_SETTINGS, ENV_FILE and the OpenRouter runner's reasoning effort state an instrument a Claude Code solver
    never runs; cc_solver_lib.MODEL is only a default (the leg passes SPEC's solver_model as --model)."""
    plant(fx.repo / rel, line)
    assert agree(fx) == []


def test_a_missing_cc_solver_lib_is_a_named_disagreement(fx):
    (fx.repo / infra.CC_SOLVER_LIB).unlink()
    got = {(d.source, d.field): d for d in agree(fx)}
    assert set(got) == {(f"{infra.CC_SOLVER_LIB} MAX_TURNS", "agent_turns")}
    assert got[(f"{infra.CC_SOLVER_LIB} MAX_TURNS", "agent_turns")].found is None


def test_a_pinned_claude_code_release_is_compared_with_its_own_entry(fx, monkeypatch):
    rel = C(CTL, "CC_RELEASES").get("CC_RELEASES") or {}
    other = next((v for v in rel if v != inf_of(fx)["claude_code"]["version"]), None)
    if other is None:
        pytest.skip("evolver_container_ctl.CC_RELEASES lists no second release")
    monkeypatch.setenv("EVOSKILL_CC_VERSION", other)
    assert {f[1] for f in fields(agree(fx))} >= {"CC_VERSION", "CC_LINUX_X64_SHA"}
    edit_json(fx.rd / "INFRA.json", lambda d: d["claude_code"].update(version=other,
                                                                      linux_x64_sha256=rel[other][0]))
    edit_json(fx.rd / "RUNTIME.json", lambda d: d["claude"].__setitem__("expect_claude_code_version", other))
    assert agree(fx) == []


def test_a_tau3_run_skips_the_ssb_instrument(tmp_path, monkeypatch):
    for k in (infra.DATA_ROOT_ENV, infra.SSBV2_ROOT_ENV, OVL, "EVOSKILL_CC_VERSION"):
        monkeypatch.delenv(k, raising=False)
    for bm in ("tau3", "ssbv2"):
        t = build_run(tmp_path / bm, benchmark=bm)
        edit_json(t.rd / "SPEC.json", lambda d: d["spec"].__setitem__("agent_turns", 7))
        plant(t.repo / infra.LEG_DRIVER_LOOP, 'EXPECTED = {"libreoffice": "other"}')
        got = fields(infra.agree(t.rd, {infra.DATA_ROOT_ENV: "/elsewhere"}, repo=t.repo))
        if bm == "tau3":
            assert got == set()
        else:                                    # the same statements bind an SSB run
            assert {(f"{infra.CC_SOLVER_LIB} MAX_TURNS", "agent_turns"),
                    (f"{infra.LEG_DRIVER_LOOP} EXPECTED", "libreoffice"), ("env", infra.DATA_ROOT_ENV)} <= got


def test_secret_files_are_checked_by_stat_only_and_the_token_is_one_file(fx):
    fx.token.chmod(0o644)
    got = {(d.source, d.field): d for d in agree(fx)}
    assert set(got) == {("file", "claude_code.token_file")} and got[("file", "claude_code.token_file")].found == "0o644"
    fx.token.unlink()
    got = {(d.source, d.field): d for d in agree(fx)}
    assert got[("file", "claude_code.token_file")].found == "missing"
    assert got[("file", "solver.key_file")].found == "missing"


def test_missing_spec_or_runtime_is_a_disagreement_and_invalid_infra_raises(fx):
    (fx.rd / "SPEC.json").unlink()
    (fx.rd / "RUNTIME.json").unlink()
    assert {("SPEC.json", "(file)"), ("RUNTIME.json", "(file)")} <= fields(agree(fx))
    edit_json(fx.rd / "INFRA.json", _set("grader.recalc_timeout_s", "nine hundred"))
    with pytest.raises(infra.InfraError):
        agree(fx)
    rec = infra.check(fx.rd, {}, repo=fx.repo)
    assert rec["valid"] is False and rec["ok"] is False and rec["problems"]


# ══ the one credential: a token file, or ANTHROPIC_API_KEY ══════════════════════════════════════
def test_an_api_key_run_declares_no_token_file_and_needs_the_key(tmp_path, monkeypatch):
    for k in (infra.DATA_ROOT_ENV, infra.SSBV2_ROOT_ENV, OVL, "EVOSKILL_CC_VERSION"):
        monkeypatch.delenv(k, raising=False)
    k = build_run(tmp_path, token=False)
    inf = json.loads((k.rd / "INFRA.json").read_text())
    assert infra.validate(inf) == [] and inf["claude_code"]["token_file"] is None
    assert infra.CC_TOKEN_FILE_ENV not in infra.claude_env(inf)
    assert infra.agree(k.rd, {infra.CC_API_KEY_ENV: "sk-ant-api03-x"}, repo=k.repo) == []
    got = infra.agree(k.rd, {}, repo=k.repo)
    assert fields(got) == {("env", infra.CC_API_KEY_ENV)}, [str(d) for d in got]
    assert infra.agree(k.rd, {}, repo=k.repo, check_env=False) == []
    got = infra.agree(k.rd, {infra.CC_API_KEY_ENV: "sk-ant-api03-x", infra.CC_TOKEN_FILE_ENV: "/x/token"}, repo=k.repo)
    assert fields(got) == {("env", infra.CC_TOKEN_FILE_ENV)} and "declares no token file" in got[0].note


def test_a_token_file_in_the_shell_must_be_the_declared_one(fx):
    assert agree(fx, {infra.CC_TOKEN_FILE_ENV: str(fx.token)}) == []
    assert agree(fx, {infra.CC_TOKEN_FILE_ENV: str(fx.token.parent / "." / fx.token.name)}) == []
    assert fields(agree(fx, {infra.CC_TOKEN_FILE_ENV: "/x/other"})) == {("env", infra.CC_TOKEN_FILE_ENV)}


# ══ the environment ═════════════════════════════════════════════════════════════════════════════
def test_env_conflicts_are_disagreements_and_unset_is_not(fx):
    inf = infra.load(fx.rd / "INFRA.json")
    assert infra.env_conflicts(inf, {}) == []
    assert infra.env_conflicts(inf, infra.derived_env(inf)) == [], "INFRA's own values agree"
    got = infra.env_conflicts(inf, {RECALC: "180", UNO: "900"})
    assert [(d.source, d.field, d.infra, d.found) for d in got] == [("env", RECALC, "900", "180")]
    same_meaning = {"EVOSKILL_CLAUDE_EFFORT": " DEFAULT ", UNO: "0900", "EVOSKILL_SOLVER_MODEL": MODEL}
    assert infra.env_conflicts(inf, same_meaning) == []
    other = {"EVOSKILL_CLAUDE_EFFORT": "high", "EVOSKILL_AO_MODEL": "opus[1m]",
             "EVOSKILL_EXPECT_CLAUDE_CODE_VERSION": "2.1.280", "EVOSKILL_EVOLVER_MODEL": "claude-opus-5"}
    assert {d.field for d in infra.env_conflicts(inf, other)} == set(other)


def test_cc_env_conflicts_refuse_every_openrouter_pin(fx):
    inf = infra.load(fx.rd / "INFRA.json")
    for k in infra.SOLVER_ROUTING_KEYS:
        got = infra.env_conflicts(inf, {k: "deepinfra"})
        assert [(d.source, d.field, d.infra, d.found) for d in got] == [("env", k, None, "deepinfra")], k
        assert "reads no OpenRouter pin" in got[0].note
        assert infra.env_conflicts(inf, {k: ""}) == [], "empty counts as unset"
    got = infra.env_conflicts(inf, dict(OPENROUTER_SHELL))
    assert {d.field for d in got} == set(infra.SOLVER_ROUTING_KEYS) | {"EVOSKILL_SOLVER_MODEL"}


def test_agree_reads_the_real_environment_by_default_and_can_skip_it(fx, monkeypatch):
    for k in list(infra.derived_env(infra.load(fx.rd / "INFRA.json"))) + list(infra.SOLVER_ROUTING_KEYS):
        monkeypatch.delenv(k, raising=False)
    assert infra.agree(fx.rd, repo=fx.repo) == []
    monkeypatch.setenv(RECALC, "300")
    assert ("env", RECALC) in fields(infra.agree(fx.rd, repo=fx.repo))
    assert infra.agree(fx.rd, repo=fx.repo, check_env=False) == []


def _cli_env(fx) -> dict:
    scrub = set(infra.derived_env(inf_of(fx))) | set(infra.SOLVER_ROUTING_KEYS) | {
        OVL, infra.DATA_ROOT_ENV, infra.SSBV2_ROOT_ENV, "EVOSKILL_CC_VERSION"}
    env = {k: v for k, v in os.environ.items() if k not in scrub}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def test_cli_exit_codes(fx):
    """The CLI reads the REAL repository; the fixture was built from its constants, so it agrees."""
    env = _cli_env(fx)

    def cli(rd, *extra):
        return subprocess.run([sys.executable, "-m", "src.orchestrator.infra", "check", "--run-dir", str(rd),
                               *extra], cwd=ROOT, env=env, capture_output=True, text=True)
    r = cli(fx.rd, "--no-env")
    assert r.returncode == infra.EXIT_OK and "AGREE" in r.stdout, r.stdout + r.stderr
    rec = json.loads(cli(fx.rd, "--json").stdout)
    assert rec["ok"] and rec["valid"], rec
    env.update({RECALC: "180", "EVOSKILL_SOLVER_PROVIDER_ORDER": "deepinfra"})
    r = cli(fx.rd)
    assert r.returncode == infra.EXIT_DISAGREE, r.stdout + r.stderr
    assert "DISAGREE" in r.stdout and f"env: {RECALC}" in r.stdout and "env: EVOSKILL_SOLVER_PROVIDER_ORDER" in r.stdout
    r = cli(fx.rd, "--no-env", "--json")
    assert r.returncode == infra.EXIT_OK and json.loads(r.stdout)["ok"]
    empty = fx.tmp / "no_infra_run"
    empty.mkdir()
    r = cli(empty)
    assert r.returncode == infra.EXIT_INVALID and "INVALID" in r.stdout


# ══ the data block (dataset root + the repaired-file overlay) ═══════════════════════════════════
def _null_overlay(d):
    d["data"]["overlay_patches_json"] = d["data"]["overlay_patches_sha256"] = None


def test_the_overlay_variable_is_one_name_and_joins_the_derived_env(fx):
    assert infra.DATA_OVERLAY_ENV == data_overlay.ENV
    assert infra.DATA_ROOT_ENV == data_overlay.DATA_ROOT_ENV and infra.SSBV2_ROOT_ENV == data_overlay.SSBV2_ROOT_ENV
    assert infra.SSBV2_DATA_SUFFIX == data_overlay.SSBV2_DATA_SUFFIX
    inf = inf_of(fx)
    assert infra.derived_env(inf)[OVL] == str(fx.ov.pj)
    _null_overlay(inf)
    assert infra.validate(inf) == [] and OVL not in infra.derived_env(inf) and infra.data_env(inf) == {}


def test_overlay_env_conflicts_follow_the_declaration(fx):
    inf, pj = inf_of(fx), fx.ov.pj
    assert infra.env_conflicts(inf, {OVL: str(pj)}) == []
    assert infra.env_conflicts(inf, {OVL: str(pj.parent / "." / pj.name)}) == []          # same path
    assert infra.env_conflicts(inf, {OVL: ""}) == []                                      # "" = unset
    got = infra.env_conflicts(inf, {OVL: "/other/PATCHES.json"})
    assert [(d.field, d.infra, d.found) for d in got] == [(OVL, str(pj), "/other/PATCHES.json")]
    _null_overlay(inf)
    assert infra.env_conflicts(inf, {}) == [] and infra.env_conflicts(inf, {OVL: ""}) == []
    got = infra.env_conflicts(inf, {OVL: str(pj)})
    assert [(d.field, d.infra) for d in got] == [(OVL, None)] and "declares no data overlay" in got[0].note


def test_the_data_block_agrees_and_null_overlay_agrees_too(fx):
    assert agree(fx) == [] and agree(fx, {OVL: str(fx.ov.pj)}) == []
    edit_json(fx.rd / "INFRA.json", _null_overlay)
    assert agree(fx) == []
    assert fields(agree(fx, {OVL: str(fx.ov.pj)})) == {("env", OVL)}
    assert agree(fx, {OVL: str(fx.ov.pj)}, check_env=False) == []


def test_a_different_root_disagrees_on_disk_and_with_an_explicit_overlay_src(tmp_path, fx):
    edit_json(fx.rd / "INFRA.json", _set("data.root", str(tmp_path / "elsewhere")))
    got = fields(agree(fx, check_env=False))
    assert ("disk", "data.root") in got and ("file", "data.overlay_patches_json entries") in got
    # an overlay that names its canonical tree is compared with data.root
    ov = make_overlay(tmp_path / "explicit", src_null=False)
    edit_json(fx.rd / "INFRA.json", lambda d: d["data"].update(
        root=str(fx.ov.root), overlay_patches_json=str(ov.pj), overlay_patches_sha256=infra.sha256_file(ov.pj)))
    got = {(d.source, d.field): d for d in agree(fx, check_env=False)}
    assert set(got) == {("PATCHES.json", "src")} and got[("PATCHES.json", "src")].found == str(ov.root)


def test_a_literal_data_constant_in_the_leg_driver_is_compared(fx):
    plant(fx.repo / infra.LEG_DRIVER_BASE, 'DATA = Path("/elsewhere/data/spreadsheetbench-v2")')
    got = {(d.source, d.field): d for d in agree(fx)}
    assert got[(f"{infra.LEG_DRIVER_BASE} DATA", "DATA")].found == "/elsewhere/data/spreadsheetbench-v2"
    plant(fx.repo / infra.LEG_DRIVER_BASE, f'DATA = Path("{fx.ov.root}")')
    assert agree(fx) == []


def test_the_environment_that_moves_the_evolvers_data_root_disagrees(fx):
    root = fx.ov.root
    assert fields(agree(fx, {infra.DATA_ROOT_ENV: "/elsewhere/ssb"})) == {("env", infra.DATA_ROOT_ENV)}
    assert fields(agree(fx, {infra.SSBV2_ROOT_ENV: "/elsewhere"})) == {("env", infra.SSBV2_ROOT_ENV)}
    assert agree(fx, {infra.SSBV2_ROOT_ENV: str(root.parents[1]) + "/"}) == []
    assert agree(fx, {infra.DATA_ROOT_ENV: str(root), infra.SSBV2_ROOT_ENV: "/elsewhere"}) == []
    assert agree(fx, {infra.DATA_ROOT_ENV: "/elsewhere/ssb"}, check_env=False) == []


def test_the_overlay_is_pinned_by_sha_and_every_entry_is_reverified(fx):
    want = infra.sha256_file(fx.ov.pj)
    edit_json(fx.rd / "INFRA.json", _set("data.overlay_patches_sha256", "0" * 64))
    got = {(d.source, d.field): d for d in agree(fx)}
    assert set(got) == {("file", "sha256(data.overlay_patches_json)")}
    assert got[("file", "sha256(data.overlay_patches_json)")].found == want
    edit_json(fx.rd / "INFRA.json", _set("data.overlay_patches_json", str(fx.tmp / "absent.json")))
    got = {(d.source, d.field): d for d in agree(fx, check_env=False)}
    assert got[("file", "sha256(data.overlay_patches_json)")].found == "missing"
    fx.ov.patched.write_bytes(b"drifted")                       # the patched copy no longer matches
    edit_json(fx.rd / "INFRA.json", lambda d: d["data"].update(overlay_patches_json=str(fx.ov.pj),
                                                               overlay_patches_sha256=want))
    got = {(d.source, d.field): d for d in agree(fx, check_env=False)}
    assert set(got) == {("file", "data.overlay_patches_json entries")}
    assert "does not hash to sha256_after" in got[("file", "data.overlay_patches_json entries")].found


# ══ data_overlay ════════════════════════════════════════════════════════════════════════════════
def test_a_null_src_is_the_runs_data_root_and_a_relative_dst_is_beside_patches_json(tmp_path, monkeypatch):
    for k in (data_overlay.DATA_ROOT_ENV, data_overlay.SSBV2_ROOT_ENV, OVL):
        monkeypatch.delenv(k, raising=False)
    ov = make_overlay(tmp_path)
    want = {data_overlay._key(ov.canonical): data_overlay._key(ov.patched)}
    assert data_overlay.verify(ov.pj, root=ov.root) == want
    with pytest.raises(data_overlay.OverlayError, match="src is null"):
        data_overlay.verify(ov.pj)                               # no root given, none in the environment
    monkeypatch.setenv(data_overlay.SSBV2_ROOT_ENV, str(ov.root.parents[1]))
    assert data_overlay.verify(ov.pj) == want
    monkeypatch.setenv(data_overlay.DATA_ROOT_ENV, str(ov.root))
    assert data_overlay.verify(ov.pj) == want
    assert data_overlay.source(ov.canonical, ov.pj) == Path(data_overlay._key(ov.patched))
    other = ov.root / "Debugging" / "a.xlsx"
    assert data_overlay.source(other, ov.pj) == other
    rec = data_overlay.record(ov.pj)
    assert rec["files"] == [{"canonical": data_overlay._key(ov.canonical), "patched": data_overlay._key(ov.patched),
                             "sha256": sha(ov.patched)}]


def test_data_overlay_verify_is_uncached_and_every_failure_is_an_overlay_error(tmp_path):
    ov = make_overlay(tmp_path, src_null=False)
    pj = ov.pj
    m = data_overlay.load(pj)
    assert len(m) == 1 and data_overlay.verify(pj) == m
    rec = json.loads(pj.read_text())
    rec["files"][0]["sha256_after"] = "0" * 64
    pj.write_text(json.dumps(rec))
    assert data_overlay.load(pj) == m                             # cached for this process
    with pytest.raises(data_overlay.OverlayError, match="sha256_after"):
        data_overlay.verify(pj)
    rec["files"][0]["path"] = "../../etc/hosts"
    pj.write_text(json.dumps(rec))
    with pytest.raises(data_overlay.OverlayError, match="relative, without"):
        data_overlay.verify(pj)
    pj.write_text("{not json")
    with pytest.raises(data_overlay.OverlayError, match="unreadable"):
        data_overlay.verify(pj)
    with pytest.raises(data_overlay.OverlayError, match="unreadable"):
        data_overlay.verify(tmp_path / "absent.json")
    assert data_overlay.same_path(str(pj), str(pj.parent / ".." / pj.parent.name / pj.name))
    assert not data_overlay.same_path("", str(pj)) and not data_overlay.same_path(None, None)


def test_the_shipped_overlay_is_portable_and_its_patched_bytes_match(tmp_path):
    """examples/ssbv2/overlay/PATCHES.json names no machine path: src null (the run's data root), dst relative.
    Every patched workbook it ships hashes to its sha256_after (the canonical side needs the dataset)."""
    pj = ROOT / "examples/ssbv2/overlay/PATCHES.json"
    rec = json.loads(pj.read_text())
    assert rec["src"] is None and not Path(rec["dst"]).is_absolute()
    src, dst = data_overlay.trees(pj, rec, root=tmp_path)
    assert src == tmp_path and dst == pj.parent / rec["dst"]
    assert len(rec["files"]) == 5
    for f in rec["files"]:
        assert sha(dst / f["path"]) == f["sha256_after"] != f["sha256_before"], f["path"]


# ══ the grade driver takes the overlay from INFRA.json ══════════════════════════════════════════
def test_grade_config_takes_the_overlay_from_infra(fx, monkeypatch):
    sys.path.insert(0, str(ROOT / "scripts" / "daytona"))
    import ssb_grade_daytona as G
    for k in (RECALC, UNO, OVL):
        monkeypatch.delenv(k, raising=False)
    ov = make_overlay(fx.tmp / "explicit", src_null=False)
    edit_json(fx.rd / "INFRA.json", lambda d: d["data"].update(
        root=str(ov.root), overlay_patches_json=str(ov.pj), overlay_patches_sha256=infra.sha256_file(ov.pj)))
    t = "Financial_Model/06_01"
    idx = {t: {"input": ov.canonical, "golden": ov.canonical, "answer_position": "A1"}}
    jobs = fx.tmp / "jobs.jsonl"
    jobs.write_text(json.dumps({"job_id": f"{t}/input/grade", "mode": "grade", "task_id": t,
                                "output": str(ov.canonical)}) + "\n")

    def cfg(env):
        return G.GradeConfig(jobs_file=jobs, out_dir=fx.tmp / "out", run_dir=fx.rd, label="t",
                             idx=idx, env=env, probe_task=t)
    try:
        c = cfg({})
    except G.Refused as e:
        if "grader code differs" in str(e):
            pytest.skip(f"the grade driver's own code pins refuse first: {e}")
        raise
    j = c.all_jobs[0]
    assert c.data_overlay == str(ov.pj) and OVL not in os.environ
    assert j["input_sha256"] == j["output_sha256"] == sha(ov.patched)     # the bytes the sandbox gets
    assert c.record()["data_overlay"]["patches_json_sha256"] == infra.sha256_file(ov.pj)
    with pytest.raises(G.Refused, match="differs from INFRA.json"):
        cfg({OVL: "/other/PATCHES.json"})
    edit_json(fx.rd / "INFRA.json", _null_overlay)
    c = cfg({})
    assert c.data_overlay is None and c.all_jobs[0]["input_sha256"] == sha(ov.canonical)
    assert c.record()["data_overlay"] is None
    with pytest.raises(G.Refused, match="declares no data overlay"):
        cfg({OVL: str(ov.pj)})
