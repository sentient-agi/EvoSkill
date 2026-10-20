"""Offline tests for the Daytona evolver host (src/orchestrator/daytona_evolver.py and the loop's daytona
branches). No network, no sandbox, no model call."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.orchestrator import daytona_evolver as de  # noqa: E402
from src.orchestrator import evidence_surface as es  # noqa: E402


# ── a fake dataset + split + run ─────────────────────────────────────────────────────────────────
def _fake(tmp_path, monkeypatch):
    root = tmp_path / "data"
    cat = root / "Template"
    (cat / "spreadsheet/01_a").mkdir(parents=True)
    (cat / "spreadsheet/02_b").mkdir(parents=True)
    tasks = []
    for tid, folder in (("01_01", "01_a"), ("01_02", "01_a"), ("02_01", "02_b")):
        (cat / f"spreadsheet/{folder}/{tid}_input.xlsx").write_bytes(f"INPUT {tid}".encode())
        (cat / f"spreadsheet/{folder}/{tid}_golden.xlsx").write_bytes(f"GOLDEN {tid}".encode())
        tasks.append({"id": tid, "spreadsheet_path": f"spreadsheet/{folder}/{tid}_input.xlsx",
                      "golden_response_path": f"spreadsheet/{folder}/{tid}_golden.xlsx"})
    (cat / "dataset.json").write_text(json.dumps(tasks))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["Template/01_01", "Template/01_02"],
                                 "val": ["Template/02_01"], "test": [], "unused": []}))
    monkeypatch.setenv("EVOSKILL_SSBV2_DATA", str(root))
    rd = tmp_path / "outputs" / "runx"
    ev = rd / "digests_e1_train"
    ev.mkdir(parents=True)
    for s in ("Template_01_01", "Template_01_02"):
        (ev / f"{s}.md").write_text(f"# {s}\n**Result: 1/3 samples passed.**\n")
    (rd / "oracle").mkdir()
    (rd / "oracle/README.md").write_text("no goldens")
    (rd / "ITER1_PROMPT.md").write_text("prompt body " * 200)
    return SimpleNamespace(root=root, split=split, rd=rd, ev=ev)


# ── T1: inputs dir ───────────────────────────────────────────────────────────────────────────────
def test_inputs_dir_train_only_and_every_planted_violation_fails(tmp_path, monkeypatch):
    f = _fake(tmp_path, monkeypatch)
    out = de.build_inputs_dir(f.rd, f.split)
    assert out["status"] == "built"
    d = de.inputs_dir(f.rd)
    assert sorted(p.name for p in d.iterdir()) == ["MANIFEST.json", "Template_01_01", "Template_01_02"]
    assert de.verify_inputs_dir(f.rd, f.split).status == "PASS"
    # a golden's bytes under a train task's input name
    x = d / "Template_01_01/01_01_input.xlsx"
    good = x.read_bytes()
    x.write_bytes(b"GOLDEN 01_01")
    assert de.verify_inputs_dir(f.rd, f.split).status == "FAIL"
    x.write_bytes(good)
    # a val input planted as an extra subdir
    (d / "Template_02_01").mkdir()
    (d / "Template_02_01/02_01_input.xlsx").write_bytes(b"INPUT 02_01")
    assert de.verify_inputs_dir(f.rd, f.split).status == "FAIL"
    for p in (d / "Template_02_01").iterdir():
        p.unlink()
    (d / "Template_02_01").rmdir()
    # an extra file inside a task dir
    (d / "Template_01_02/extra.txt").write_text("x")
    assert de.verify_inputs_dir(f.rd, f.split).status == "FAIL"
    (d / "Template_01_02/extra.txt").unlink()
    # a symlink
    (d / "link").symlink_to(d / "Template_01_01")
    assert de.verify_inputs_dir(f.rd, f.split).status == "FAIL"
    (d / "link").unlink()
    assert de.verify_inputs_dir(f.rd, f.split).status == "PASS"
    # a missing task
    y = d / "Template_01_02/01_02_input.xlsx"
    yb = y.read_bytes()
    y.unlink()
    (d / "Template_01_02").rmdir()
    assert de.verify_inputs_dir(f.rd, f.split).status == "FAIL"
    (d / "Template_01_02").mkdir()
    y.write_bytes(yb)
    assert de.verify_inputs_dir(f.rd, f.split).status == "PASS"


# ── T3: the VM listing reproduces dir_sha256 exactly ─────────────────────────────────────────────
def test_vm_listing_reproduces_dir_sha256(tmp_path):
    d = tmp_path / "ev dir"
    (d / "sub ü").mkdir(parents=True)
    (d / "a b.md").write_text("x")
    (d / "sub ü/ç.txt").write_text("y")
    r = subprocess.run([sys.executable, "-c", de.VM_LISTING_PY, str(d)], capture_output=True, text=True)
    rows = json.loads(r.stdout)
    assert de.dir_sha256_from_rows(rows, str(d)) == es.dir_sha256(d)
    rows.append([str(d / "extra"), "F", "0" * 64, 1])
    assert de.dir_sha256_from_rows(rows, str(d)) != es.dir_sha256(d)


# ── T2: the manifest check (the #213 replacement) ────────────────────────────────────────────────
def _manifest_setup(f, epoch=1):
    de.build_inputs_dir(f.rd, f.split)
    venv = {"HOME": de.VM_HOME}
    env_text, hooks_text = de.render_env_file(venv), de.hook_settings_text()
    rows, gen = de.build_upload_manifest(
        run_dir=f.rd, epoch=epoch, prompt=f.rd / "ITER1_PROMPT.md", evidence_dirs=[f.ev],
        oracle=f.rd / "oracle", staging=f.rd / "skills_v1_staging",
        prev_library=f.rd / "skills_v0", inputs=de.inputs_dir(f.rd), env_text=env_text,
        hooks_text=hooks_text)
    receipt = {"containment": "daytona", "sandboxed": True, "upload_manifest": rows,
               "evidence_sha256_before": es.hashes([f.ev]), "daytona": {"vm_env": venv}}
    listing = {"pre_upload": {"users_exists": False},
               "after_upload": [[r["vm_path"], "F", r["sha256"], r["bytes"]] for r in rows]}
    return receipt, listing


def _check(f, receipt, listing, epoch=1):
    return de.check_upload_manifest(run_dir=f.rd, epoch=epoch, split=f.split, evidence_dirs=[f.ev],
                                    prev_library=f.rd / f"skills_v{epoch - 1}",
                                    prompt=f.rd / "ITER1_PROMPT.md", oracle=f.rd / "oracle",
                                    staging=f.rd / f"skills_v{epoch}_staging", receipt=receipt,
                                    listing=listing)


def test_manifest_check_passes_clean_and_fails_each_violation(tmp_path, monkeypatch):
    f = _fake(tmp_path, monkeypatch)
    receipt, listing = _manifest_setup(f)
    assert _check(f, receipt, listing).status == "PASS"
    # an extra file on the VM
    bad = json.loads(json.dumps(listing))
    bad["after_upload"].append(["/evo/other_run/skills_v1/a/SKILL.md", "F", "a" * 64, 3])
    assert _check(f, receipt, bad).status == "FAIL"
    # a symlink on the VM
    bad = json.loads(json.dumps(listing))
    bad["after_upload"].append(["/root/evo/l", "L", "/etc", 0])
    assert _check(f, receipt, bad).status == "FAIL"
    # a golden's bytes in the manifest
    g = json.loads(json.dumps(receipt))
    g["upload_manifest"][0]["sha256"] = hashlib.sha256(b"GOLDEN 01_01").hexdigest()
    assert _check(f, g, listing).status == "FAIL"
    # an evidence row outside the granted dirs
    g = json.loads(json.dumps(receipt))
    g["upload_manifest"].append({"vm_path": str(f.rd / "digests_e2_train/x.md"), "role": "evidence",
                                 "source": str(f.ev / "Template_01_01.md"),
                                 "sha256": de.sha256_file(f.ev / "Template_01_01.md"), "bytes": 1})
    assert _check(f, g, listing).status == "FAIL"
    # a seed row at epoch 1
    g = json.loads(json.dumps(receipt))
    g["upload_manifest"].append({"vm_path": str(f.rd / "skills_v1_staging/a/SKILL.md"), "role": "seed",
                                 "source": "generated", "sha256": "b" * 64, "bytes": 1})
    assert _check(f, g, listing).status == "FAIL"
    # the prompt changed after upload
    (f.rd / "ITER1_PROMPT.md").write_text("changed " * 300)
    assert _check(f, receipt, listing).status == "FAIL"


def test_manifest_check_skips_when_not_daytona(tmp_path, monkeypatch):
    f = _fake(tmp_path, monkeypatch)
    receipt, listing = _manifest_setup(f)
    assert _check(f, {**receipt, "containment": "sandbox-exec"}, listing).status == "SKIP"
    assert _check(f, {**receipt, "upload_manifest": []}, listing).status == "SKIP"


def test_manifest_builder_refuses_dataset_and_split_sources(tmp_path, monkeypatch):
    f = _fake(tmp_path, monkeypatch)
    de.build_inputs_dir(f.rd, f.split)
    kw = dict(run_dir=f.rd, epoch=1, evidence_dirs=[f.ev], oracle=f.rd / "oracle",
              staging=f.rd / "skills_v1_staging", prev_library=f.rd / "skills_v0",
              inputs=de.inputs_dir(f.rd), env_text="", hooks_text="", split=f.split)
    golden = f.root / "Template/spreadsheet/01_a/01_01_golden.xlsx"
    with pytest.raises(ValueError):                       # a source inside the dataset tree
        de.build_upload_manifest(prompt=golden, **kw)
    with pytest.raises(ValueError):                       # the split file itself
        de.build_upload_manifest(prompt=f.split, **kw)
    # a NAME containing "golden" outside the dataset tree is fine (e.g. a skill named golden-rules)
    (f.ev / "golden-rules.md").write_text("x")
    rows, _ = de.build_upload_manifest(prompt=f.rd / "ITER1_PROMPT.md", **kw)
    assert any(r["vm_path"].endswith("golden-rules.md") for r in rows)
    assert any(r["role"] == "policy" and r["vm_path"] == de.VM_POLICY for r in rows)


# ── no grading-validation material reaches an agent ──────────────────
def _validation_tree(tmp_path, f):
    """setup/GRADING_VALIDATION.json in the run (what freeze pins), a plugin_validate out dir in
    <run>.operator/ with a grading-driver shard inside it, and a free-standing validation out dir."""
    (f.rd / "setup").mkdir(exist_ok=True)
    gv = f.rd / "setup/GRADING_VALIDATION.json"
    gv.write_text('{"kind": "grading_validation", "V4": {"passed_heldout": 99}}')
    op = f.rd.parent / f"{f.rd.name}.operator" / "validation_x"
    (op / "shard1/run_1").mkdir(parents=True)
    for n in ("BUILD.json", "jobs.json", "jobs.jsonl", "PLUGIN_VALIDATION.json",
              "HELDOUT_DETAIL.operator.json"):
        (op / n).write_text("{}")
    for n in ("jobs.jsonl", "jobs_normalized.jsonl", "results.jsonl"):
        (op / "shard1" / n).write_text("{}")
    (op / "shard1/run_1/summary.json").write_text("{}")
    free = tmp_path / "scratch_validation"
    free.mkdir()
    for n in ("BUILD.json", "jobs.jsonl", "results.jsonl"):
        (free / n).write_text("{}")
    return gv, op, free


def test_validation_source_names_the_files_the_operator_dir_and_any_out_dir(tmp_path, monkeypatch):
    f = _fake(tmp_path, monkeypatch)
    gv, op, free = _validation_tree(tmp_path, f)
    assert "GRADING_VALIDATION.json" in de.validation_source(gv)
    for p in (op / "PLUGIN_VALIDATION.json", op / "HELDOUT_DETAIL.operator.json",
              tmp_path / "x" / "plugin_validation.json", tmp_path / "anything.operator.json"):
        assert de.validation_source(p).startswith("a grading-validation file"), p
    assert "under a validation out dir" in de.validation_source(op / "jobs.json")
    assert "under a validation out dir" in de.validation_source(op / "shard1/run_1/summary.json")
    assert "under a validation out dir" in de.validation_source(free / "results.jsonl")
    (f.rd.parent / f"{f.rd.name}.operator" / "CONVERSATION.jsonl").write_text("")
    assert "under the operator dir" in de.validation_source(
        f.rd.parent / f"{f.rd.name}.operator" / "CONVERSATION.jsonl")
    rg = tmp_path / "rg"
    rg.mkdir()
    (rg / "REGRADE.json").write_text("{}")
    assert "REGRADE.json" in de.validation_source(rg / "results.jsonl")
    gd = tmp_path / "grade_smoke/out"                     # the grading driver's own --out-dir
    (gd / "run_1").mkdir(parents=True)
    for n in ("jobs_normalized.jsonl", "results.jsonl", "run_1/summary.json"):
        (gd / n).write_text("{}")
    assert "jobs_normalized.jsonl" in de.validation_source(gd / "run_1/summary.json")
    # a symlink is judged by where it points
    (f.ev / "looks_fine.md").symlink_to(op / "jobs.json")
    assert "under a validation out dir" in de.validation_source(f.ev / "looks_fine.md")
    (f.ev / "looks_fine.md").unlink()
    # not validation material: the run's own files, a skill named like a check, a lone BUILD.json
    lone = tmp_path / "lone"
    lone.mkdir()
    (lone / "BUILD.json").write_text("{}")
    for p in (f.ev / "Template_01_01.md", f.rd / "ITER1_PROMPT.md", f.rd / "setup/PLAN.json",
              tmp_path / "skills_v1/validation-rules/SKILL.md", lone / "notes.md"):
        assert de.validation_source(p) == "", p


def test_upload_refuses_and_checks_fail_on_validation_material(tmp_path, monkeypatch):
    f = _fake(tmp_path, monkeypatch)
    gv, op, free = _validation_tree(tmp_path, f)
    de.build_inputs_dir(f.rd, f.split)
    kw = dict(run_dir=f.rd, epoch=1, oracle=f.rd / "oracle", staging=f.rd / "skills_v1_staging",
              prev_library=f.rd / "skills_v0", inputs=de.inputs_dir(f.rd), env_text="",
              hooks_text="", split=f.split)
    # as the prompt, from the operator dir
    with pytest.raises(ValueError, match="grading-validation"):
        de.build_upload_manifest(prompt=op / "PLUGIN_VALIDATION.json", evidence_dirs=[f.ev], **kw)
    # as evidence: a free-standing validation out dir named as an evidence dir
    with pytest.raises(ValueError, match="under a validation out dir"):
        de.build_upload_manifest(prompt=f.rd / "ITER1_PROMPT.md", evidence_dirs=[free], **kw)
    # as evidence: the pinned copy dropped into a granted evidence dir
    (f.ev / "GRADING_VALIDATION.json").write_bytes(gv.read_bytes())
    with pytest.raises(ValueError, match="GRADING_VALIDATION.json"):
        de.build_upload_manifest(prompt=f.rd / "ITER1_PROMPT.md", evidence_dirs=[f.ev], **kw)
    (f.ev / "GRADING_VALIDATION.json").unlink()
    # as a seed file at epoch 2 (the name is judged on the source and on the VM path)
    lib = f.rd / "skills_v1/a"
    lib.mkdir(parents=True)
    (lib / "SKILL.md").write_text("---\nname: a\ndescription: d\n---\n")
    (lib / "PLUGIN_VALIDATION.json").write_text("{}")
    with pytest.raises(ValueError, match="PLUGIN_VALIDATION.json"):
        de.build_upload_manifest(**{**kw, "epoch": 2, "prev_library": f.rd / "skills_v1",
                                    "staging": f.rd / "skills_v2_staging"},
                                 prompt=f.rd / "ITER1_PROMPT.md", evidence_dirs=[f.ev])
    # the AO's rows go through the same refusal (the driver calls it with them)
    with pytest.raises(ValueError, match="grading-validation"):
        de._refuse_forbidden_sources([de._row(str(gv), "literal", str(gv))], f.split)
    # a validation file NAME on the VM is refused even when the source looks ordinary
    ordinary = f.rd / "ITER1_PROMPT.md"
    with pytest.raises(ValueError, match="PLUGIN_VALIDATION.json"):
        de._refuse_forbidden_sources([de._row(f"{de.VM_WORK}/PLUGIN_VALIDATION.json", "literal",
                                              str(ordinary))], f.split)
    # the clean build is still accepted
    rows, _ = de.build_upload_manifest(prompt=f.rd / "ITER1_PROMPT.md", evidence_dirs=[f.ev], **kw)
    assert not [r for r in rows if r["source"] != "generated" and de.validation_source(r["source"])]

    # the after-the-fact checks fail a receipt that holds it, whatever the row's role
    receipt, listing = _manifest_setup(f)
    assert _check(f, receipt, listing).status == "PASS"
    for p in (gv, op / "shard1/results.jsonl"):
        bad = json.loads(json.dumps(receipt))
        bad["upload_manifest"].append(de._row(str(p), "evidence", str(p)))
        c = _check(f, bad, listing)
        assert c.status == "FAIL" and ("grading-validation" in c.detail
                                       or "validation out dir" in c.detail), c.detail
    msg = f.rd / "e1_AUTHOR_MESSAGE.md"
    msg.write_text("author message")
    rows = [de._row(f"{de.VM_WORK}/MESSAGE.md", "message", str(msg))]
    rows += [de._row(str(p), "evidence", str(p)) for p in sorted(f.ev.iterdir())]
    rows.append(de._row(str(f.rd / "oracle/README.md"), "oracle", str(f.rd / "oracle/README.md")))
    rows.append(de._row(de.VM_ENV_FILE, "env", data=de.render_env_file({"HOME": "/root"}).encode()))
    rows.append(de._row(de.VM_POLICY, "policy", data=de.policy_settings_text().encode()))
    rec = {"role": "ao", "containment": "daytona", "upload_manifest": rows,
           "evidence_sha256_before": es.hashes([f.ev])}
    lst = {"pre_upload": {"users_exists": False},
           "after_upload": [[r["vm_path"], "F", r["sha256"], r["bytes"]] for r in rows]}
    akw = dict(run_dir=f.rd, epoch=1, message=msg, evidence_dirs=[f.ev], oracle=f.rd / "oracle")
    assert de.check_ao_upload(receipt=rec, listing=lst, **akw).status == "PASS"
    bad = json.loads(json.dumps(rec))
    bad["upload_manifest"].append(de._row(str(gv), "literal", str(gv)))
    c = de.check_ao_upload(receipt=bad, listing=lst, **akw)
    assert c.status == "FAIL" and "grading-validation" in c.detail, c.detail


# ── T4: the daytona receipt check ────────────────────────────────────────────────────────────────
def _good_receipt(tmp_path, ledger, log):
    rb = {"refreshed": True, "cpu": 8, "memory": 32, "disk": 40, "domain_allow_list": "api.anthropic.com",
          "network_allow_list": None, "network_block_all": False}
    probe = [{"url": u, "reachable": True} for u in de.MUST_REACH] + \
            [{"url": u, "reachable": False} for u in de.MUST_REFUSE]
    ok, det, rec = de.dockerfile_check()
    ledger.write_text(json.dumps({"event": "deleted", "id": "sb-1", "driver": de.DRIVER_NAME}) + "\n")
    return {"containment": "daytona", "sandboxed": True,
            "evidence_sha256_before": {"/x": "h"}, "evidence_sha256_after_vm": {"/x": "h"},
            "inputs_sha256_before": "i", "inputs_sha256_after_vm": "i",
            "token_scan": {"hits": [], "scanned_files": 5},
            "daytona": {"sandbox_id": "sb-1", "readback_create": rb, "readback_end": dict(rb),
                        "egress_probe": probe, "dockerfile": rec,
                        "egress_direct": [{"target": "https://example.com", "reached": False}],
                        "egress_raw": [{"target": "example.com:22", "tcp_connected": False},
                                       {"target": "udp 1.1.1.1:53", "answered": False},
                                       {"target": "[2606:4700:4700::1111]:443", "tcp_connected": False},
                                       {"target": "169.254.169.254:80", "tcp_connected": False}],
                        "log_round_trip": {"equal": True, "streamed_sha256": de.sha256_file(log)},
                        "session_audit": {"files": 1, "denied_tool_uses": [], "nested_claude": [],
                                          "api_or_token_refs": []},
                        "identity": {"cc_sha256": de.CC_LINUX_X64_SHA,
                                     "libreoffice": "LibreOffice 25.2.3.2 520(Build:2)"}}}


def _log(tmp_path, api="CLAUDE_CODE_OAUTH_TOKEN", mcp=(), remote=False):
    p = tmp_path / "evolver.log"
    lines = [json.dumps({"type": "system", "subtype": "init", "apiKeySource": api,
                         "mcp_servers": list(mcp), "model": "claude-opus-5-5",
                         "claude_code_version": "2.1.281", "tools": ["Bash"]})]
    if remote:
        lines.append(json.dumps({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "RemoteTrigger", "input": {}}]}}))
    p.write_text("\n".join(lines) + "\n")
    return p


def test_daytona_receipt_check(tmp_path):
    led = tmp_path / "ledger.log"
    lg0 = _log(tmp_path)
    r = _good_receipt(tmp_path, led, lg0)
    ok, det = de.check_daytona_receipt(r, log=lg0, ledger_path=led,
                                       require_libreoffice="LibreOffice 25.2.3.2 520(Build:2)")
    assert ok, det
    for mut in (lambda x: x["daytona"]["readback_end"].update(refreshed=False),
                lambda x: x["daytona"]["egress_direct"][0].update(reached=True),
                lambda x: x["daytona"]["egress_raw"][2].update(tcp_connected=True),
                lambda x: x["daytona"]["session_audit"].update(nested_claude=["s.jsonl"]),
                lambda x: x["daytona"]["session_audit"].update(denied_tool_uses=["s.jsonl:WebSearch"]),
                lambda x: x["daytona"]["log_round_trip"].update(streamed_sha256="0" * 64)):
        x = json.loads(json.dumps(r)); mut(x)
        assert not de.check_daytona_receipt(x, log=lg0, ledger_path=led)[0]
    cases = []
    x = json.loads(json.dumps(r)); x["daytona"]["readback_end"]["domain_allow_list"] = "api.anthropic.com,github.com"; cases.append(x)
    x = json.loads(json.dumps(r)); x["daytona"]["readback_create"]["network_allow_list"] = "0.0.0.0/0"; cases.append(x)
    x = json.loads(json.dumps(r)); x["daytona"]["egress_probe"][1]["reachable"] = True; cases.append(x)
    x = json.loads(json.dumps(r)); x["daytona"]["readback_create"]["cpu"] = 4; cases.append(x)
    x = json.loads(json.dumps(r)); x["token_scan"]["hits"] = ["/q/f"]; cases.append(x)
    x = json.loads(json.dumps(r)); x["evidence_sha256_after_vm"] = {"/x": "CHANGED"}; cases.append(x)
    x = json.loads(json.dumps(r)); x["daytona"]["sandbox_id"] = "sb-2"; cases.append(x)
    x = json.loads(json.dumps(r)); x["daytona"]["identity"]["cc_sha256"] = "0" * 64; cases.append(x)
    for c in cases:
        ok, det = de.check_daytona_receipt(c, log=lg0, ledger_path=led)
        assert not ok
    for kw in ({"api": "ANTHROPIC_API_KEY"}, {"mcp": ["claude.ai Docs"]}, {"remote": True}):
        lg = _log(tmp_path, **kw)
        x = json.loads(json.dumps(r)); x["daytona"]["log_round_trip"]["streamed_sha256"] = de.sha256_file(lg)
        ok, det = de.check_daytona_receipt(x, log=lg, ledger_path=led)
        assert not ok, det


# ── env file / hooks carry no secret ─────────────────────────────────────────────────────────────
def test_vm_env_has_no_secret_and_hooks_are_sync_only():
    e = de.vm_env("claude-opus-5-5", env={})
    assert e["ENABLE_CLAUDEAI_MCP_SERVERS"] == "false" and e["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert e["IS_SANDBOX"] == "1" and e["HOME"] == de.VM_HOME
    deny = json.loads(de.policy_settings_text())["permissions"]["deny"]
    assert deny == [*de.DENIED_TOOLS, *de.DENIED_READS]
    # the validation files and every *.operator path, from the root
    assert {f"Read(//**/{f})" for f in de.VALIDATION_FILES} | {"Read(//**/*.operator/**)"} <= set(deny)
    assert not [k for k in e if "TOKEN" in k or "KEY" in k]
    txt = de.render_env_file(e)
    assert de.TOKEN_PREFIX not in txt and "unset ANTHROPIC_API_KEY" in txt
    h = json.loads(de.hook_settings_text())
    assert [m["matcher"] for m in h["hooks"]["PreToolUse"]] == ["Agent"]


def test_token_scan_reports_paths_only(tmp_path):
    (tmp_path / "a.txt").write_text("nothing")
    (tmp_path / "b.txt").write_text("xx " + de.TOKEN_PREFIX + "abc")
    hits, n = de.token_scan([tmp_path], de.TOKEN_PREFIX + "zzzz")
    assert n == 2 and hits == [str(tmp_path / "b.txt")]


def test_dockerfile_is_solver_recipe_plus_cc():
    ok, det, rec = de.dockerfile_check()
    assert ok, det
    assert rec["base_sha256"] == de.BASE_DOCKERFILE_SHA


# ── T8 / loop: the 'ls' of this run's own skills path fails local, passes daytona (manifest) ────
def test_loop_host_validation_and_steps():
    from src.orchestrator import loop as L
    names = [s.name for s in L.epoch_steps()]
    assert names.index("evolver inputs are train-only") < names.index("evolver prompt authored by the AO")
    ctx = SimpleNamespace(evolver_host="mars", spec=SimpleNamespace(evolver_harness="claude_cli",
                                                                    benchmark="ssbv2"),
                          evolver_sandbox=True, evolver_daytona={})
    with pytest.raises(L.Halt):
        L._evolver_host(ctx)
    ctx.evolver_host = "daytona"
    ctx.spec.evolver_harness = "direct"
    with pytest.raises(L.Halt):
        L._evolver_host(ctx)
    ctx.spec.evolver_harness = "claude_cli"
    ctx.evolver_sandbox = False
    with pytest.raises(L.Halt):
        L._evolver_host(ctx)
    ctx.evolver_host = "local"
    assert L._evolver_host(ctx) == "local"


def test_peer_grep_flags_own_skills_path_locally():
    """The halted e1 shape: an `ls` of this run's own base-leg skills path trips the LOCAL #213 grep
    (this is why daytona mode replaces it with the manifest check)."""
    from src.orchestrator.invariants import check_no_peer_artifacts_read
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        rd = Path(td) / "outputs" / "ssbv2_x"
        rd.mkdir(parents=True)
        log = rd / "tau3_evolver_e1.log"
        cmd = f"ls -la {REPO}/.cache/ssbv2_ssbv2_x_base_train_skills 2>/dev/null"
        log.write_text(json.dumps({"type": "system", "subtype": "init", "session_id": "s"}) + "\n" +
                       json.dumps({"type": "assistant", "session_id": "s", "message": {"content": [
                           {"type": "tool_use", "name": "Bash", "input": {"command": cmd}}]}}) + "\n")
        assert check_no_peer_artifacts_read(log, rd).status == "FAIL"


# ── the AO role's upload check ───────────────────────────────────────────────────────────────────
def test_ao_upload_check(tmp_path, monkeypatch):
    f = _fake(tmp_path, monkeypatch)
    msg = f.rd / "e1_AUTHOR_MESSAGE.md"
    msg.write_text("author message")
    (f.rd / "SPEC.json").write_text("{}")
    env = de.render_env_file({"HOME": "/root"}).encode()
    rows = [de._row(f"{de.VM_WORK}/MESSAGE.md", "message", str(msg))]
    rows += [de._row(str(p), "evidence", str(p)) for p in sorted(f.ev.iterdir())]
    rows.append(de._row(str(f.rd / "oracle/README.md"), "oracle", str(f.rd / "oracle/README.md")))
    rows.append(de._row(str(f.rd / "SPEC.json"), "literal", str(f.rd / "SPEC.json")))
    rows.append(de._row(de.VM_ENV_FILE, "env", data=env))
    rows.append(de._row(de.VM_POLICY, "policy", data=de.policy_settings_text().encode()))
    rec = {"role": "ao", "containment": "daytona", "upload_manifest": rows,
           "evidence_sha256_before": es.hashes([f.ev])}
    lst = {"pre_upload": {"users_exists": False},
           "after_upload": [[r["vm_path"], "F", r["sha256"], r["bytes"]] for r in rows]}
    kw = dict(run_dir=f.rd, epoch=1, message=msg, evidence_dirs=[f.ev], oracle=f.rd / "oracle")
    assert de.check_ao_upload(receipt=rec, listing=lst, **kw).status == "PASS"
    # an extra literal the AO is not granted (the run's split / a val leg)
    bad = json.loads(json.dumps(rec))
    (f.rd / "runx_base_val.json").write_text("{}")
    bad["upload_manifest"].append(de._row(str(f.rd / "runx_base_val.json"), "literal", str(f.rd / "runx_base_val.json")))
    assert de.check_ao_upload(receipt=bad, listing=lst, **kw).status == "FAIL"
    # an extra file on the VM
    l2 = json.loads(json.dumps(lst))
    l2["after_upload"].append(["/evo/x/y.md", "F", "c" * 64, 1])
    assert de.check_ao_upload(receipt=rec, listing=l2, **kw).status == "FAIL"
    # the message changed after upload
    msg.write_text("different")
    assert de.check_ao_upload(receipt=rec, listing=lst, **kw).status == "FAIL"
    assert de.check_ao_upload(receipt={**rec, "role": "evolver"}, listing=lst, **kw).status == "SKIP"


# ── session audit + held-out scan ──────────────────────────────────────────────────
def test_session_audit_flags_nested_claude_denied_tools_and_token_refs(tmp_path):
    def tu(name, cmd=None):
        inp = {"command": cmd} if cmd is not None else {}
        return json.dumps({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": name, "input": inp}]}})
    d = tmp_path / "projects/-evo-work"
    d.mkdir(parents=True)
    (d / "s1.jsonl").write_text("\n".join([tu("Bash", "ls -la"), tu("Read")]) + "\n")
    clean = de.audit_sessions([tmp_path / "projects"])
    assert clean["files"] == 1 and not clean["nested_claude"] and not clean["denied_tool_uses"]
    (d / "sub.jsonl").write_text("\n".join([tu("Bash", "cd /tmp && claude -p hi"), tu("WebSearch"),
                                           tu("Bash", "curl https://api.anthropic.com/v1/messages")]) + "\n")
    a = de.audit_sessions([tmp_path / "projects"])
    assert a["nested_claude"] and a["denied_tool_uses"] and a["api_or_token_refs"]
    # "claude" inside a word or path is not a nested session
    (d / "sub.jsonl").write_text(tu("Bash", "ls /usr/local/lib/claude-code; echo claudette") + "\n")
    assert not de.audit_sessions([tmp_path / "projects"])["nested_claude"]


def test_heldout_scan_catches_safe_and_escaped_forms(tmp_path, monkeypatch):
    f = _fake(tmp_path, monkeypatch)
    lg = tmp_path / "log.jsonl"
    lg.write_text('{"x": "cat /data/a/digests_e1_train/Template_01_01.md"}\n')
    assert de.heldout_id_hits([lg], f.split) == []            # train id: fine
    lg.write_text('{"x": "ls evolver_inputs_train/Template_02_01/"}\n')
    assert de.heldout_id_hits([lg], f.split)                   # val id, safe form
    lg.write_text('{"x": "task \\"Template/02_01\\" said"}\n')
    assert de.heldout_id_hits([lg], f.split)                   # val id, escaped canonical form
