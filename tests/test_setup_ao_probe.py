"""scripts/daytona/setup_ao_probe.py OFFLINE: the benchmark adapter's upload manifest, built by make_run() (everything
main() does before the sandbox) and checked by SetupRun.prepare() -- no Daytona, no model call, no network; the OAuth
token is a 0600 stand-in (T._cfg is replaced).

The files the brief names besides the data root and the split are RUNTIME.json ao.brief_files (absent = BRIEF_FILES,
the SSB pair). A tau3 run declares [] and hands the adapter --split examples/tau3_banking/split.json and --grader
examples/tau3_banking/grader.py: the /data view is /data/<tau3 data root> + /data/split.json. The forbidden byte streams
of daytona_evolver's dataset tree are computed only when the view IS that tree, and in the full view the user's grader
is not scanned for held-out ids (the tau3 grader names the dropped task_102 in a comment). --grader is required and the
given grader is copied, never replaced.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _jload(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def _jdump(p: Path, d) -> None:
    Path(p).write_text(json.dumps(d, indent=1))


@pytest.fixture
def S(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO / "scripts" / "daytona"))
    import setup_ao_probe as S
    for k in ("EVOSKILL_SSBV2_DATA_OVERLAY", "EVOSKILL_SSBV2_DATA", "SSBV2_ROOT"):
        monkeypatch.delenv(k, raising=False)
    return S


TAU3_SPLIT = REPO / "examples" / "tau3_banking" / "split.json"
TAU3_GRADER = REPO / "examples" / "tau3_banking" / "grader.py"
TAU3_MESSAGE = REPO / "examples" / "tau3_banking" / "instruction.md"
TAU3_FILES = ("tau2/domains/banking_knowledge/db.json", "tau2/domains/banking_knowledge/documents/doc_001.md",
              "tau2/domains/banking_knowledge/prompts/policy.md", "tau2/domains/banking_knowledge/tasks/task_004.json",
              "tau2/user_simulator/guidelines.md", ".embeddings_cache/kb.json")


def tau3_run(tmp_path: Path, monkeypatch, S):
    """A scratch tau3 run dir (RUNTIME ao.brief_files [], no split of its own yet: the adapter runs before the freeze
    writes it) and a small tau3-shaped data root that the run's INFRA.json data block names (load_infra is replaced
    by that block; INFRA.json validation is tested on its own)."""
    root = tmp_path / "tau3_data"
    for rel in TAU3_FILES:
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(f"bytes of {rel}\n")
    rd = tmp_path / "runs" / "tau3_scratch"
    (rd / "setup").mkdir(parents=True)
    _jdump(rd / "RUNTIME.json", {"prompt_mode": "template", "ao": {"brief_files": []}})
    data = {"root": str(root), "overlay_patches_json": None, "overlay_patches_sha256": None}
    monkeypatch.setattr(S.L, "load_infra", lambda run_dir: ({"data": dict(data)}, None))
    return rd, root


def _prepared(S, monkeypatch, tmp_path, rd: Path, argv: list):
    """make_run(<argv>) with the token stand-in, then prepare(): the manifest rows and the receipt, nothing created."""
    tok = tmp_path / "secrets" / "ao_token_standin"
    tok.parent.mkdir(parents=True, exist_ok=True)
    tok.write_text("sk-ant-oat01-NOT-A-TOKEN")
    tok.chmod(0o600)
    monkeypatch.setattr(S.T, "_cfg", lambda: {"oauth_token_file": str(tok)})
    (rd / "setup_probe").mkdir(exist_ok=True)
    run = S.make_run(S.build_parser().parse_args(["--run-dir", str(rd), *argv]), cache_root=tmp_path / "cache")
    run.prepare()
    assert run.token == "sk-ant-oat01-NOT-A-TOKEN"
    return run


# ══ 1. the tau3 brief ═════════════════════════════════════════════════════════════════════════════════════════
def test_the_tau3_manifest_is_the_data_root_the_split_and_the_given_files(S, tmp_path, monkeypatch):
    rd, root = tau3_run(tmp_path, monkeypatch, S)
    assert S.brief_files_for(rd) == ()
    run = _prepared(S, monkeypatch, tmp_path, rd, ["--message", str(TAU3_MESSAGE), "--grader", str(TAU3_GRADER),
                                                    "--split", str(TAU3_SPLIT)])
    rows = {r["vm_path"]: r for r in run.rows}
    data = {f"/data/{root.name}/{p.relative_to(root).as_posix()}" for p in root.rglob("*") if p.is_file()}
    assert len(data) == len(TAU3_FILES)
    setup = {f"/evo/setup/{n}" for n in ("plugin_api.py", "benchmark_plugin_skeleton.py", "benchmark_grader.py")}
    assert set(rows) == {"/evo/work/MESSAGE.md", "/data/split.json", *data, *setup, S.de.VM_ENV_FILE, S.de.VM_POLICY}
    assert {r["role"] for r in run.rows} == {"message", "dataview", "brief_file", "setup_input", "env", "policy"}
    assert [r["vm_path"] for r in run.rows if r["role"] == "brief_file"] == ["/data/split.json"]
    assert rows["/data/split.json"]["sha256"] == _sha(TAU3_SPLIT)
    assert rows["/evo/setup/benchmark_grader.py"]["sha256"] == _sha(TAU3_GRADER)
    assert rows["/evo/work/MESSAGE.md"]["sha256"] == _sha(TAU3_MESSAGE)
    for p in data:                                           # byte-exact, at its own path
        assert rows[p]["sha256"] == _sha(root / p[len(f"/data/{root.name}/"):])
    assert not [r for r in run.rows if "scorer" in r["vm_path"] or "Dockerfile" in r["vm_path"]
                or "scorer" in str(r.get("source")) or "Dockerfile" in str(r.get("source"))]
    rec = run.receipt
    assert rec["brief_files"] == [str(TAU3_SPLIT)] and rec["data_root"] == str(root)
    assert rec["vm_paths"]["data_root"] == f"/data/{root.name}" and rec["forbidden_hashes_checked"] == 0
    assert rec["heldout_or_nontrain_files_uploaded"] == 0 and rec["data_overlay"] is None
    assert (rd / "setup_probe" / "SETUP_AO_DAYTONA.json").is_file()


def test_the_tau3_split_and_grader_pass_the_held_out_checks_on_their_own_terms(S):
    de = S.de
    roles = de.split_roles(TAU3_SPLIT)
    assert sorted(set(roles.values())) == ["dropped", "test", "train", "val"] and roles["task_102"] == "dropped"
    # the user's grader names the DROPPED task_102 (a comment): a hit, which is why the full view does not scan it
    assert de.heldout_id_hits([TAU3_GRADER], TAU3_SPLIT) == [f"{TAU3_GRADER.name}:task_102"]
    assert de.heldout_id_hits([TAU3_MESSAGE, REPO / "src/orchestrator/plugin_api.py"], TAU3_SPLIT) == []


# ══ 2. the SSB brief: today's rows ════════════════════════════════════════════════════════════════════════════
def ssb_run(tmp_path: Path, *, ao=None) -> tuple:
    """A run dir with a SPEC naming the split and a RUNTIME (no ao block unless `ao`), no INFRA.json; the data root a
    tiny SSB-shaped tree (three categories, one train + one val task each) that the test hands over as
    EVOSKILL_SSBV2_DATA, the root daytona_evolver also reads."""
    root = tmp_path / "ssbdata" / "spreadsheetbench-v2"
    for i, cat in enumerate(("Debugging", "Financial_Model", "Template")):
        recs = []
        for j in (1, 2):
            tid = f"0{i}_0{j}"
            d = root / cat / tid
            d.mkdir(parents=True)
            (d / "in.xlsx").write_bytes(f"input {cat} {tid}".encode())
            (d / "golden.xlsx").write_bytes(f"golden {cat} {tid}".encode())
            recs.append({"id": tid, "spreadsheet_path": f"{tid}/in.xlsx", "golden_response_path": f"{tid}/golden.xlsx"})
        _jdump(root / cat / "dataset.json", recs)
    split = tmp_path / "splits" / "split.json"
    split.parent.mkdir(parents=True)
    _jdump(split, {"train": [f"{c}/0{i}_01" for i, c in enumerate(("Debugging", "Financial_Model", "Template"))],
                   "val": [f"{c}/0{i}_02" for i, c in enumerate(("Debugging", "Financial_Model", "Template"))],
                   "test": []})
    rd = tmp_path / "runs" / "ssb_ao_scratch"
    rd.mkdir(parents=True)
    _jdump(rd / "SPEC.json", {"spec": {"name": rd.name, "benchmark": "ssbv2", "split_path": str(split)}})
    rt = {"prompt_mode": "template", "evolver_session": "stateful"}
    if ao is not None:
        rt["ao"] = ao
    _jdump(rd / "RUNTIME.json", rt)
    msg = tmp_path / "MESSAGE.md"
    msg.write_text("The data root is /data/spreadsheetbench-v2; the split is /data/split.json.\n")
    grader = tmp_path / "user_grader.py"
    grader.write_text("def grade(task, snapshot):\n    return 0.0\n")
    return rd, root, split, msg, grader


def test_an_ssb_run_without_an_ao_block_uploads_todays_brief_files(S, tmp_path, monkeypatch):
    rd, root, split, msg, grader = ssb_run(tmp_path)
    monkeypatch.setenv("EVOSKILL_SSBV2_DATA", str(root))       # daytona_evolver's tree = this view's root
    assert S.brief_files_for(rd) == S.BRIEF_FILES == ("src/agent_profiles/ssbv2_agent/scorer.py",
                                                      "scripts/daytona/Dockerfile.ssb_leg")
    run = _prepared(S, monkeypatch, tmp_path, rd, ["--message", str(msg), "--grader", str(grader)])
    brief = [(r["vm_path"], r["source"]) for r in run.rows if r["role"] == "brief_file"]
    assert sorted(brief) == sorted([("/data/split.json", str(split)),
                                    *((f"/data/EvoSkill/{b}", str(REPO / b)) for b in S.BRIEF_FILES)])
    assert run.receipt["brief_files"] == [str(split), *(str(REPO / b) for b in S.BRIEF_FILES)]
    # the dataset tree's forbidden streams are computed (this view IS that tree): goldens + val inputs, uploaded by design
    assert run.receipt["forbidden_hashes_checked"] == 9 and run.receipt["heldout_or_nontrain_files_uploaded"] == 9
    setup = {r["vm_path"]: r["sha256"] for r in run.rows if r["role"] == "setup_input"}
    assert setup["/evo/setup/benchmark_grader.py"] == _sha(grader)


def test_ao_brief_files_is_read_from_runtime_and_checked(S, tmp_path, monkeypatch):
    rd, root, split, msg, grader = ssb_run(tmp_path, ao={"brief_files": ["scripts/daytona/Dockerfile.ssb_leg"]})
    monkeypatch.setenv("EVOSKILL_SSBV2_DATA", str(root))
    assert S.brief_files_for(rd) == ("scripts/daytona/Dockerfile.ssb_leg",)
    run = _prepared(S, monkeypatch, tmp_path, rd, ["--message", str(msg), "--grader", str(grader)])
    assert sorted(r["vm_path"] for r in run.rows if r["role"] == "brief_file") == sorted([
        "/data/split.json", f"/data/EvoSkill/scripts/daytona/Dockerfile.ssb_leg"])
    assert S.brief_files_for(tmp_path / "no_such_run") == S.BRIEF_FILES
    for bad in ("scorer.py", ["/abs/scorer.py"], [""], [3]):
        _jdump(rd / "RUNTIME.json", {"ao": {"brief_files": bad}})
        with pytest.raises(SystemExit, match="ao.brief_files must be a list of repo-relative paths"):
            S.brief_files_for(rd)
    _jdump(rd / "RUNTIME.json", {"ao": {}})
    assert S.brief_files_for(rd) == S.BRIEF_FILES


def test_the_held_out_id_scan(S, tmp_path, monkeypatch):
    """The message is scanned in every view; the user's grader only in the train view."""
    rd, root, split, msg, grader = ssb_run(tmp_path)
    monkeypatch.setenv("EVOSKILL_SSBV2_DATA", str(root))
    grader.write_text("# a note about Template/02_02\ndef grade(task, snapshot):\n    return 0.0\n")
    run = _prepared(S, monkeypatch, tmp_path, rd, ["--message", str(msg), "--grader", str(grader)])   # full view
    assert run.receipt["data_view"] == "full"
    for view in ("full", "train"):
        msg.write_text("Look at Debugging/00_02 first.\n")
        with pytest.raises(S.Refuse, match="held-out task id in the upload"):
            _prepared(S, monkeypatch, tmp_path, rd, ["--message", str(msg), "--grader", str(grader), "--data-view", view])
        msg.write_text("The data root is /data/spreadsheetbench-v2.\n")
    with pytest.raises(S.Refuse, match=r"held-out task id in the upload: \['benchmark_grader.py:Template/02_02'\]"):
        _prepared(S, monkeypatch, tmp_path, rd, ["--message", str(msg), "--grader", str(grader), "--data-view", "train"])


# ══ 3. the user's grader ═════════════════════════════════════════════════════════════════════════════════
def test_grader_is_required_and_given_verbatim(S, tmp_path, capsys):
    with pytest.raises(SystemExit) as ei:
        S.build_parser().parse_args(["--run-dir", "x", "--message", "m"])
    assert ei.value.code == 2 and "--grader" in capsys.readouterr().err
    setup = tmp_path / "setup"
    with pytest.raises(SystemExit, match="no grader file"):
        S.ensure_given_files(setup, tmp_path / "missing.py")
    g = tmp_path / "g.py"
    g.write_text("def grade(task, snapshot):\n    return 1.0\n")
    S.ensure_given_files(setup, g)
    assert (setup / "benchmark_grader.py").read_bytes() == g.read_bytes()
    assert (setup / "plugin_api.py").read_bytes() == (REPO / "src/orchestrator/plugin_api.py").read_bytes()
    assert (setup / "benchmark_plugin_skeleton.py").read_text() == S.SKELETON
    S.ensure_given_files(setup, g)                            # the same file again: fine
    g2 = tmp_path / "g2.py"
    g2.write_text("def grade(task, snapshot):\n    return 0.0\n")
    with pytest.raises(SystemExit, match="differs from"):
        S.ensure_given_files(setup, g2)
    assert os.path.samefile(S.REPO, REPO)
