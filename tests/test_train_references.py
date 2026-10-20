"""Offline tests for the evolver's reference grants: the evolver sees the TRAINING tasks' reference
workbooks, never a validation/test one, never a non-train input; training inputs are OFF by default.
src/orchestrator/daytona_evolver.py: evolver_grants, build/verify_references_dir, the
"train_reference" upload role, check_upload_manifest and check_daytona_receipt. No network, no
sandbox, no model call."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.orchestrator import daytona_evolver as de  # noqa: E402
from src.orchestrator import evidence_surface as es  # noqa: E402
from tests.test_evolver_daytona import _good_receipt, _log  # noqa: E402


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ── a fake dataset: 2 train, 1 val, 1 test, 1 excluded_prior_run sharing a train answer workbook ───
def _fake(tmp_path, monkeypatch, grants=None):
    root = tmp_path / "data"
    cat = root / "Template"
    tasks = []

    def task(tid, folder, golden_name=None, golden_bytes=None):
        d = cat / "spreadsheet" / folder
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{tid}_input.xlsx").write_bytes(f"INPUT {tid}".encode())
        gname = golden_name or f"{tid}_golden.xlsx"
        if golden_bytes is not None or not (d / gname).exists():
            (d / gname).write_bytes(golden_bytes if golden_bytes is not None else f"GOLDEN {tid}".encode())
        tasks.append({"id": tid, "spreadsheet_path": f"spreadsheet/{folder}/{tid}_input.xlsx",
                      "golden_response_path": f"spreadsheet/{folder}/{gname}"})
    task("01_01", "01_a")
    task("01_02", "01_a")
    task("01_03", "01_a", golden_name="01_01_golden.xlsx")      # excluded: SAME answer file as 01_01
    task("02_01", "02_b")                                        # val
    task("03_01", "03_c")                                        # test
    # a stray grader scratch copy of a train golden, left in the dataset tree (as on the real data)
    (cat / "spreadsheet/01_a/01_01_golden.xlsx.fresh_datatable_scratch.d5.xlsx").write_bytes(b"GOLDEN 01_01")
    (cat / "dataset.json").write_text(json.dumps(tasks))
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"train": ["Template/01_01", "Template/01_02"], "val": ["Template/02_01"],
                                 "test": ["Template/03_01"], "unused": [],
                                 "excluded_prior_run": {"task_ids": ["Template/01_03"]}}))
    monkeypatch.setenv("EVOSKILL_SSBV2_DATA", str(root))
    rd = tmp_path / "outputs" / "runr"
    ev = rd / "digests_e1_train"
    ev.mkdir(parents=True)
    for s in ("Template_01_01", "Template_01_02"):
        (ev / f"{s}.md").write_text(f"# {s}\n**Result: 1/3 samples passed.**\n")
    (rd / "oracle").mkdir()
    (rd / "oracle/README.md").write_text("scoring rule")
    (rd / "ITER1_PROMPT.md").write_text("prompt body " * 200)
    if grants is not None:
        (rd / "RUNTIME.json").write_text(json.dumps({"evolver_daytona": dict(grants)}))
    return SimpleNamespace(root=root, cat=cat, split=split, rd=rd, ev=ev)


def _manifest(f, *, inputs=False, references=True):
    g = de.evolver_grants(f.rd)
    if inputs:
        de.build_inputs_dir(f.rd, f.split)
    if references:
        de.build_references_dir(f.rd, f.split)
    venv = {"HOME": de.VM_HOME}
    rows, _gen = de.build_upload_manifest(
        run_dir=f.rd, epoch=1, prompt=f.rd / "ITER1_PROMPT.md", evidence_dirs=[f.ev],
        oracle=f.rd / "oracle", staging=f.rd / "skills_v1_staging", prev_library=f.rd / "skills_v0",
        inputs=de.inputs_dir(f.rd) if inputs else None,
        references=de.references_dir(f.rd) if references else None, grants=g,
        env_text=de.render_env_file(venv), hooks_text=de.hook_settings_text(), split=f.split)
    receipt = {"containment": "daytona", "sandboxed": True, "upload_manifest": rows, "grants": dict(g),
               "evidence_sha256_before": es.hashes([f.ev]), "daytona": {"vm_env": venv},
               "references_dir": str(de.references_dir(f.rd)) if references else None,
               "references_sha256_before": es.dir_sha256(de.references_dir(f.rd)) if references else None}
    listing = {"pre_upload": {"users_exists": False},
               "after_upload": [[r["vm_path"], "F", r["sha256"], r["bytes"]] for r in rows]}
    return receipt, listing


def _check(f, receipt, listing, grants=None):
    return de.check_upload_manifest(run_dir=f.rd, epoch=1, split=f.split, evidence_dirs=[f.ev],
                                    prev_library=f.rd / "skills_v0", prompt=f.rd / "ITER1_PROMPT.md",
                                    oracle=f.rd / "oracle", staging=f.rd / "skills_v1_staging",
                                    receipt=receipt, listing=listing, grants=grants)


def _cp(x):
    return json.loads(json.dumps(x))


# ── grants: RUNTIME.json only, booleans only, absent = off ───────────────────────────────────────
def test_grants_come_from_runtime_booleans_only_absent_is_off(tmp_path):
    rd = tmp_path / "run"
    rd.mkdir()
    g = de.evolver_grants(rd)                                   # no RUNTIME.json: the legacy default
    assert (g["train_inputs"], g["train_references"]) == (True, False)
    assert g["source"].startswith("no RUNTIME.json")
    (rd / "RUNTIME.json").write_text(json.dumps({"evolver_daytona": {"ttl_minutes": 180}}))
    g = de.evolver_grants(rd)                                   # both absent -> both OFF
    assert (g["train_inputs"], g["train_references"]) == (False, False)
    (rd / "RUNTIME.json").write_text(json.dumps({}))
    assert not de.evolver_grants(rd)["train_inputs"]
    (rd / "RUNTIME.json").write_text(json.dumps({"evolver_daytona": {"train_references": True}}))
    g = de.evolver_grants(rd)
    assert (g["train_inputs"], g["train_references"]) == (False, True)
    for bad in ({"evolver_daytona": {"train_references": "true"}},
                {"evolver_daytona": {"train_inputs": 1}},
                {"evolver_daytona": ["train_references"]}):
        (rd / "RUNTIME.json").write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            de.evolver_grants(rd)
    (rd / "RUNTIME.json").write_text("{not json")
    with pytest.raises(ValueError):
        de.evolver_grants(rd)
    assert de.grants_equal({"train_inputs": False, "train_references": True, "source": "a"},
                           {"train_inputs": False, "train_references": True, "source": "b"})
    assert not de.grants_equal({"train_inputs": False, "train_references": True}, {})


# ── the references dir: train goldens only ───────────────────────────────────────────────────────
def test_build_references_dir_is_train_goldens_only(tmp_path, monkeypatch):
    f = _fake(tmp_path, monkeypatch)
    out = de.build_references_dir(f.rd, f.split)
    assert out["status"] == "built", out
    d = de.references_dir(f.rd)
    assert sorted(p.name for p in d.iterdir()) == ["MANIFEST.json", "Template_01_01", "Template_01_02"]
    assert (d / "Template_01_01/01_01_golden.xlsx").read_bytes() == b"GOLDEN 01_01"
    assert (d / "Template_01_02/01_02_golden.xlsx").read_bytes() == b"GOLDEN 01_02"
    c = de.verify_references_dir(f.rd, f.split)
    assert c.status == "PASS", c.detail
    # 01_01's answer file is also the excluded task's answer file and has a scratch copy: tolerated
    # (same bytes, never graded in this run), reported, and refused under strict.
    assert set(c.evidence["tolerated"]) == {"Template_01_01"}
    assert any(lb.startswith("golden:excluded_prior_run:") for lb in c.evidence["tolerated"]["Template_01_01"])
    assert any(lb.startswith("dataset-file:") for lb in c.evidence["tolerated"]["Template_01_01"])
    assert de.verify_references_dir(f.rd, f.split, strict=True).status == "FAIL"
    # the MANIFEST names no held-out id and no dataset path
    mtxt = (d / "MANIFEST.json").read_text()
    assert "02_01" not in mtxt and "03_01" not in mtxt and str(f.root) not in mtxt
    assert set(json.loads(mtxt)) == {"what", "layout", "files"}
    assert de.heldout_id_hits([d], f.split) == []
    # idempotent: an existing verified dir is left alone
    assert de.build_references_dir(f.rd, f.split)["status"] == "exists-verified"
    # strict build refuses before writing anything
    f2 = _fake(tmp_path / "strict", monkeypatch)
    with pytest.raises(ValueError):
        de.build_references_dir(f2.rd, f2.split, strict=True)
    assert not de.references_dir(f2.rd).exists()


def test_verify_references_dir_fails_every_planted_violation(tmp_path, monkeypatch):
    f = _fake(tmp_path, monkeypatch)
    de.build_references_dir(f.rd, f.split)
    d = de.references_dir(f.rd)
    ok = lambda: de.verify_references_dir(f.rd, f.split).status  # noqa: E731
    assert ok() == "PASS"
    # 1. a VAL golden planted as its own task subdir
    (d / "Template_02_01").mkdir()
    (d / "Template_02_01/02_01_golden.xlsx").write_bytes(b"GOLDEN 02_01")
    c = de.verify_references_dir(f.rd, f.split)
    assert c.status == "FAIL" and "not a train task subdir" in c.detail
    (d / "Template_02_01/02_01_golden.xlsx").unlink()
    (d / "Template_02_01").rmdir()
    assert ok() == "PASS"
    # 2. a VAL golden's bytes under a train task's reference name
    x = d / "Template_01_02/01_02_golden.xlsx"
    good = x.read_bytes()
    x.write_bytes(b"GOLDEN 02_01")
    c = de.verify_references_dir(f.rd, f.split)
    assert c.status == "FAIL" and "golden:val:Template/02_01" in c.detail
    # 3. a TEST input's bytes under a train task's reference name
    x.write_bytes(b"INPUT 03_01")
    c = de.verify_references_dir(f.rd, f.split)
    assert c.status == "FAIL" and "input:test:Template/03_01" in c.detail
    # 4. a TRAIN input's bytes (inputs never ride in the references dir, granted or not)
    x.write_bytes(b"INPUT 01_02")
    assert ok() == "FAIL"
    x.write_bytes(good)
    assert ok() == "PASS"
    # 5. a test input as an extra file inside a train task's subdir
    (d / "Template_01_02/03_01_input.xlsx").write_bytes(b"INPUT 03_01")
    assert ok() == "FAIL"
    (d / "Template_01_02/03_01_input.xlsx").unlink()
    # 6. a file from outside (a digest) at the top level, and inside a subdir
    (d / "notes.md").write_text("x")
    assert ok() == "FAIL"
    (d / "notes.md").unlink()
    (d / "Template_01_01/extra.md").write_text("x")
    assert ok() == "FAIL"
    (d / "Template_01_01/extra.md").unlink()
    # 7. a symlink (to the dataset golden itself)
    (d / "Template_01_01/link.xlsx").symlink_to(f.cat / "spreadsheet/02_b/02_01_golden.xlsx")
    assert ok() == "FAIL"
    (d / "Template_01_01/link.xlsx").unlink()
    # 8. a missing train task
    y = d / "Template_01_02/01_02_golden.xlsx"
    y.unlink()
    (d / "Template_01_02").rmdir()
    assert ok() == "FAIL"
    (d / "Template_01_02").mkdir()
    y.write_bytes(good)
    assert ok() == "PASS"
    # 9. a MANIFEST that disagrees / carries an extra key
    mf = d / "MANIFEST.json"
    m0 = mf.read_text()
    mj = json.loads(m0)
    mj["files"][0][2] = "0" * 64
    mf.write_text(json.dumps(mj))
    assert ok() == "FAIL"
    mj = json.loads(m0)
    mj["heldout"] = ["Template/02_01"]
    mf.write_text(json.dumps(mj))
    assert ok() == "FAIL"
    mf.write_text(m0)
    assert ok() == "PASS"
    # a pre-existing dir that does not verify is refused, never overwritten
    x.write_bytes(b"GOLDEN 02_01")
    with pytest.raises(ValueError):
        de.build_references_dir(f.rd, f.split)
    assert x.read_bytes() == b"GOLDEN 02_01"


def test_train_golden_equal_to_a_heldout_file_is_refused(tmp_path, monkeypatch):
    # the val task's answer workbook carries the same bytes as a train one: the split is not
    # project-level. Asserted at build (nothing written) and at verify (dir built before the change).
    f = _fake(tmp_path, monkeypatch)
    de.build_references_dir(f.rd, f.split)
    (f.cat / "spreadsheet/02_b/02_01_golden.xlsx").write_bytes(b"GOLDEN 01_02")
    idx, roles = de.dataset_index(f.root), de.split_roles(f.split)
    assert de.heldout_reference_collisions(idx, roles, f.root) == ["Template/01_02 = golden:val:Template/02_01"]
    c = de.verify_references_dir(f.rd, f.split)
    assert c.status == "FAIL" and "held-out" in c.detail
    f2 = _fake(tmp_path / "b", monkeypatch)
    (f2.cat / "spreadsheet/03_c/03_01_golden.xlsx").write_bytes(b"GOLDEN 01_01")    # a TEST golden
    with pytest.raises(ValueError, match="held-out"):
        de.build_references_dir(f2.rd, f2.split)
    assert not de.references_dir(f2.rd).exists()
    f3 = _fake(tmp_path / "c", monkeypatch)
    (f3.cat / "spreadsheet/03_c/03_01_input.xlsx").write_bytes(b"GOLDEN 01_02")     # a TEST input
    with pytest.raises(ValueError, match="held-out"):
        de.build_references_dir(f3.rd, f3.split)
    f4 = _fake(tmp_path / "d", monkeypatch)
    (f4.cat / "spreadsheet/01_a/01_03_input.xlsx").write_bytes(b"GOLDEN 01_02")     # an excluded task's INPUT
    with pytest.raises(ValueError, match="forbidden"):
        de.build_references_dir(f4.rd, f4.split)
    assert not de.references_dir(f4.rd).exists()


# ── the upload manifest: role "train_reference", behind the grant ────────────────────────────────
def test_manifest_with_train_references_passes_and_flag_off_fails(tmp_path, monkeypatch):
    f = _fake(tmp_path, monkeypatch, grants={"train_references": True})
    receipt, listing = _manifest(f)
    roles = [r["role"] for r in receipt["upload_manifest"]]
    assert roles.count("train_reference") == 3 and "inputs" not in roles       # 2 workbooks + MANIFEST
    c = _check(f, receipt, listing)
    assert c.status == "PASS", c.detail
    assert c.evidence["grants"] == {"train_inputs": False, "train_references": True}
    # the flag OFF (RUNTIME.json) but the references on the box
    (f.rd / "RUNTIME.json").write_text(json.dumps({"evolver_daytona": {}}))
    c = _check(f, receipt, listing)
    assert c.status == "FAIL" and "does not grant train_references" in c.detail
    # ... or the flag passed in explicitly
    (f.rd / "RUNTIME.json").write_text(json.dumps({"evolver_daytona": {"train_references": True}}))
    assert _check(f, receipt, listing).status == "PASS"
    c = _check(f, {**receipt, "grants": None}, listing,
               grants={"train_inputs": False, "train_references": False, "source": "t"})
    assert c.status == "FAIL" and "does not grant train_references" in c.detail
    # the receipt records grants other than the run's
    c = _check(f, {**receipt, "grants": {"train_inputs": True, "train_references": True}}, listing)
    assert c.status == "FAIL" and "receipt records grants" in c.detail
    # the builder refuses references the grants do not cover, and grants without the dir
    kw = dict(run_dir=f.rd, epoch=1, prompt=f.rd / "ITER1_PROMPT.md", evidence_dirs=[f.ev],
              oracle=f.rd / "oracle", staging=f.rd / "skills_v1_staging", prev_library=f.rd / "skills_v0",
              env_text="", hooks_text="", split=f.split)
    with pytest.raises(ValueError):
        de.build_upload_manifest(references=de.references_dir(f.rd), **kw)          # grants=None: no refs
    with pytest.raises(ValueError):
        de.build_upload_manifest(references=de.references_dir(f.rd),
                                 grants={"train_inputs": False, "train_references": False}, **kw)
    with pytest.raises(ValueError):
        de.build_upload_manifest(grants={"train_inputs": False, "train_references": True}, **kw)
    with pytest.raises(ValueError):                                                 # not this run's dir
        de.build_upload_manifest(references=f.ev, grants={"train_inputs": False, "train_references": True}, **kw)


def test_manifest_planted_violations_with_train_references(tmp_path, monkeypatch):
    f = _fake(tmp_path, monkeypatch, grants={"train_references": True})
    receipt, listing = _manifest(f)
    assert _check(f, receipt, listing).status == "PASS"
    d = de.references_dir(f.rd)
    # a VAL golden planted in the references dir, uploaded as a train_reference row
    (d / "Template_02_01").mkdir()
    p = d / "Template_02_01/02_01_golden.xlsx"
    p.write_bytes(b"GOLDEN 02_01")
    g = _cp(receipt)
    g["upload_manifest"].append(de._row(str(p), "train_reference", str(p)))
    lst = _cp(listing)
    lst["after_upload"].append([str(p), "F", _sha(b"GOLDEN 02_01"), 12])
    c = _check(f, g, lst)
    assert c.status == "FAIL" and "references dir does not verify" in c.detail
    p.unlink()
    (d / "Template_02_01").rmdir()
    assert _check(f, receipt, listing).status == "PASS"
    # a train golden smuggled as EVIDENCE (outside the references dir): still a forbidden sha
    (f.ev / "Template_01_01_ref.xlsx").write_bytes(b"GOLDEN 01_01")
    g = _cp(receipt)
    g["upload_manifest"].append(de._row(str(f.ev / "Template_01_01_ref.xlsx"), "evidence",
                                        str(f.ev / "Template_01_01_ref.xlsx")))
    c = _check(f, g, listing)
    assert c.status == "FAIL" and "forbidden" in c.detail
    (f.ev / "Template_01_01_ref.xlsx").unlink()
    # a train_reference row whose source lies outside the references dir
    outside = tmp_path / "elsewhere.xlsx"
    outside.write_bytes(b"GOLDEN 01_02")
    g = _cp(receipt)
    g["upload_manifest"].append(de._row(str(d / "Template_01_02/copy.xlsx"), "train_reference", str(outside)))
    assert _check(f, g, listing).status == "FAIL"
    # a TEST input uploaded under any role
    t_in = tmp_path / "t.xlsx"
    t_in.write_bytes(b"INPUT 03_01")
    g = _cp(receipt)
    g["upload_manifest"].append(de._row(str(f.ev / "t.xlsx"), "evidence", str(t_in)))
    assert _check(f, g, listing).status == "FAIL"
    # a references-dir file on the VM that the manifest does not list
    lst = _cp(listing)
    lst["after_upload"].append([str(d / "Template_01_01/extra.xlsx"), "F", "e" * 64, 1])
    assert _check(f, receipt, lst).status == "FAIL"
    # one reference row dropped (the dir was not uploaded whole)
    g = _cp(receipt)
    g["upload_manifest"] = [r for r in g["upload_manifest"] if not r["vm_path"].endswith("01_02_golden.xlsx")]
    assert _check(f, g, listing).status == "FAIL"
    # the pre-dispatch hash of the references dir does not match the rows
    c = _check(f, {**receipt, "references_sha256_before": "0" * 64}, listing)
    assert c.status == "FAIL" and "pre-dispatch hash" in c.detail
    # the references dir changed after upload
    (d / "Template_01_02/01_02_golden.xlsx").write_bytes(b"GOLDEN 01_02 v2")
    assert _check(f, receipt, listing).status == "FAIL"


def test_builder_pairs_roles_with_their_dirs(tmp_path, monkeypatch):
    f = _fake(tmp_path, monkeypatch)
    de.build_references_dir(f.rd, f.split)
    d = de.references_dir(f.rd)
    ref = d / "Template_01_01/01_01_golden.xlsx"
    golden = f.cat / "spreadsheet/01_a/01_01_golden.xlsx"
    for rows in ([de._row(str(ref), "evidence", str(ref))],                      # refs file as evidence
                 [de._row(str(ref), "literal", str(ref))],                       # ... as an AO literal
                 [de._row(str(d / "x.xlsx"), "train_reference", str(golden))],   # dataset-root source
                 [de._row(str(f.ev / "a.md"), "train_reference", str(f.ev / "Template_01_01.md"))],
                 [de._row(str(d / "g.xlsx"), "train_reference", data=b"GOLDEN 01_01")]):
        with pytest.raises(ValueError):
            de._refuse_forbidden_sources(rows, f.split)
    de._refuse_forbidden_sources([de._row(str(ref), "train_reference", str(ref))], f.split)


# ── training inputs are optional (off by default) ─────────────────────────────────────────────────
def test_inputs_are_optional_and_refused_when_not_granted(tmp_path, monkeypatch):
    f = _fake(tmp_path, monkeypatch, grants={})
    receipt, listing = _manifest(f, inputs=False, references=False)
    assert {r["role"] for r in receipt["upload_manifest"]} == {"prompt", "evidence", "oracle", "hook", "env", "policy"}
    c = _check(f, receipt, listing)
    assert c.status == "PASS", c.detail
    # inputs on the box while the run does not grant them
    de.build_inputs_dir(f.rd, f.split)
    g = _cp(receipt)
    extra = [de._row(str(p), "inputs", str(p)) for p in sorted(de.inputs_dir(f.rd).rglob("*")) if p.is_file()]
    g["upload_manifest"] += extra
    lst = _cp(listing)
    lst["after_upload"] += [[r["vm_path"], "F", r["sha256"], r["bytes"]] for r in extra]
    c = _check(f, g, lst)
    assert c.status == "FAIL" and "does not grant train_inputs" in c.detail
    # granted -> the same rows pass
    (f.rd / "RUNTIME.json").write_text(json.dumps({"evolver_daytona": {"train_inputs": True}}))
    g["grants"] = de.evolver_grants(f.rd)
    assert _check(f, g, lst).status == "PASS"
    # granted but the dir omitted from the build
    kw = dict(run_dir=f.rd, epoch=1, prompt=f.rd / "ITER1_PROMPT.md", evidence_dirs=[f.ev],
              oracle=f.rd / "oracle", staging=f.rd / "skills_v1_staging", prev_library=f.rd / "skills_v0",
              env_text="", hooks_text="", split=f.split)
    with pytest.raises(ValueError):
        de.build_upload_manifest(grants={"train_inputs": True, "train_references": False}, **kw)
    with pytest.raises(ValueError):
        de.build_upload_manifest(inputs=de.inputs_dir(f.rd),
                                 grants={"train_inputs": False, "train_references": False}, **kw)
    # both granted together
    (f.rd / "RUNTIME.json").write_text(json.dumps({"evolver_daytona": {"train_inputs": True,
                                                                       "train_references": True}}))
    receipt2, listing2 = _manifest(f, inputs=True, references=True)
    c = _check(f, receipt2, listing2)
    assert c.status == "PASS", c.detail
    assert c.evidence["counts"]["inputs"] == 3 and c.evidence["counts"]["train_reference"] == 3


# ── the receipt: every granted read-only dir hashes the same before and after ─────────────────────
def test_daytona_receipt_hashes_each_granted_dir(tmp_path):
    led = tmp_path / "ledger.log"
    lg = _log(tmp_path)
    r = _good_receipt(tmp_path, led, lg)
    assert de.check_daytona_receipt(r, log=lg, ledger_path=led)[0]
    # inputs not granted: no inputs hash is required
    x = _cp(r)
    for k in ("inputs_sha256_before", "inputs_sha256_after_vm"):
        x.pop(k)
    x["inputs_dir"] = None
    ok, det = de.check_daytona_receipt(x, log=lg, ledger_path=led)
    assert ok, det
    # ... but a manifest that carries inputs rows needs them
    x["upload_manifest"] = [{"vm_path": "/data/x/evolver_inputs_train/a", "role": "inputs"}]
    ok, det = de.check_daytona_receipt(x, log=lg, ledger_path=led)
    assert not ok and "inputs changed" in det
    # references granted: the hash must match before/after and exist
    x = _cp(r)
    x.update(references_dir="/data/x/evolver_references_train", references_sha256_before="h",
             references_sha256_after_vm="h")
    assert de.check_daytona_receipt(x, log=lg, ledger_path=led)[0]
    x["references_sha256_after_vm"] = "CHANGED"
    ok, det = de.check_daytona_receipt(x, log=lg, ledger_path=led)
    assert not ok and "train references changed" in det
    x = _cp(r)
    x["upload_manifest"] = [{"vm_path": "/data/x/evolver_references_train/a", "role": "train_reference"}]
    ok, det = de.check_daytona_receipt(x, log=lg, ledger_path=led)
    assert not ok and "train references changed" in det


# ── split key "dropped": dropped tasks are NON-TRAIN everywhere ─────────────────────
def _with_dropped(f, *, share_train_golden=False):
    """The fake plus two DROPPED tasks: 04_01 (was test) and 05_01 (was unused; with share_train_golden its
    answer workbook has train 01_02's bytes). The split lists them under "dropped" only, as the
    shipped examples/ssbv2/split.json does."""
    tasks = json.loads((f.cat / "dataset.json").read_text())
    for tid, folder in (("04_01", "04_d"), ("05_01", "05_e")):
        d = f.cat / "spreadsheet" / folder
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{tid}_input.xlsx").write_bytes(f"INPUT {tid}".encode())
        gold = b"GOLDEN 01_02" if (share_train_golden and tid == "05_01") else f"GOLDEN {tid}".encode()
        (d / f"{tid}_golden.xlsx").write_bytes(gold)
        tasks.append({"id": tid, "spreadsheet_path": f"spreadsheet/{folder}/{tid}_input.xlsx",
                      "golden_response_path": f"spreadsheet/{folder}/{tid}_golden.xlsx"})
    (f.cat / "dataset.json").write_text(json.dumps(tasks))
    sp = json.loads(f.split.read_text())
    sp["dropped"] = {"reason": "own reference fails self-grading", "n": 2,
                     "task_ids": ["Template/04_01", "Template/05_01"],
                     "removed_from": {"test": ["Template/04_01"], "unused": ["Template/05_01"]}}
    f.split.write_text(json.dumps(sp))
    return f


def test_dropped_tasks_are_non_train_for_every_evolver_rule(tmp_path, monkeypatch):
    f = _with_dropped(_fake(tmp_path, monkeypatch, grants={"train_references": True}))
    roles = de.split_roles(f.split)
    assert roles["Template/04_01"] == roles["Template/05_01"] == de.DROPPED_ROLE
    idx = de.dataset_index(f.root)
    bad = de.forbidden_shas(idx, roles, f.root)
    for tid in ("04_01", "05_01"):                      # golden AND input of each dropped task
        assert bad[_sha(f"GOLDEN {tid}".encode())] == f"golden:dropped:Template/{tid}"
        assert bad[_sha(f"INPUT {tid}".encode())] == f"input:dropped:Template/{tid}"
    # the references dir: the train goldens only, none of a dropped task's
    out = de.build_references_dir(f.rd, f.split)
    assert out["status"] == "built"
    d = de.references_dir(f.rd)
    assert sorted(p.name for p in d.iterdir()) == ["MANIFEST.json", "Template_01_01", "Template_01_02"]
    # the inputs dir: train only
    de.build_inputs_dir(f.rd, f.split)
    assert sorted(p.name for p in de.inputs_dir(f.rd).iterdir()) == ["MANIFEST.json", "Template_01_01",
                                                                     "Template_01_02"]
    # a dropped task's files planted in either dir FAIL verification
    (d / "Template_04_01").mkdir()
    (d / "Template_04_01/04_01_golden.xlsx").write_bytes(b"GOLDEN 04_01")
    c = de.verify_references_dir(f.rd, f.split)
    assert c.status == "FAIL" and "not a train task subdir" in c.detail
    (d / "Template_04_01/04_01_golden.xlsx").unlink()
    (d / "Template_04_01").rmdir()
    x = d / "Template_01_02/01_02_golden.xlsx"
    good = x.read_bytes()
    x.write_bytes(b"GOLDEN 05_01")                     # a dropped golden's bytes under a train name
    c = de.verify_references_dir(f.rd, f.split)
    assert c.status == "FAIL" and "golden:dropped:Template/05_01" in c.detail
    x.write_bytes(good)
    assert de.verify_references_dir(f.rd, f.split).status == "PASS"
    # the upload: a dropped task's input or golden under any role is a forbidden sha
    receipt, listing = _manifest(f)
    assert _check(f, receipt, listing).status == "PASS"
    for name, data in (("d_in.xlsx", b"INPUT 04_01"), ("d_gold.xlsx", b"GOLDEN 05_01")):
        (f.ev / name).write_bytes(data)
        g = _cp(receipt)
        g["upload_manifest"].append(de._row(str(f.ev / name), "evidence", str(f.ev / name)))
        c = _check(f, g, listing)
        assert c.status == "FAIL" and "dropped" in c.detail, c.detail
        (f.ev / name).unlink()
    # a dropped id in a log or transcript, in either spelling, is flagged like a held-out id
    lg = tmp_path / "evolver.log"
    for text in ('{"x": "Template/04_01"}', "open Template_05_01.md"):
        lg.write_text(text)
        assert de.heldout_id_hits([lg], f.split), text
    lg.write_text("Template/01_01 Template_01_02")      # train ids: fine
    assert de.heldout_id_hits([lg], f.split) == []


def test_a_train_golden_equal_to_a_dropped_golden_is_refused(tmp_path, monkeypatch):
    # dropped tasks are never evolver-visible: a train reference carrying a dropped task's answer
    # workbook bytes would show it, so the builder refuses before writing (strict or not)
    f = _with_dropped(_fake(tmp_path, monkeypatch), share_train_golden=True)
    with pytest.raises(ValueError, match="forbidden"):
        de.build_references_dir(f.rd, f.split)
    assert not de.references_dir(f.rd).exists()
    idx, roles = de.dataset_index(f.root), de.split_roles(f.split)
    labels = de.dataset_sha_labels(idx, roles, f.root)
    refused, _tol = de.reference_label_verdict(labels[_sha(b"GOLDEN 01_02")])
    assert refused == ["golden:dropped:Template/05_01"]


def test_a_dropped_role_overrides_any_other_listing(tmp_path, monkeypatch):
    # fail-closed: a split that lists a task in train AND dropped never makes it train here
    # (plugin_runner.validate reports that split as malformed)
    f = _fake(tmp_path, monkeypatch)
    sp = json.loads(f.split.read_text())
    sp["dropped"] = {"task_ids": ["Template/01_02"]}
    f.split.write_text(json.dumps(sp))
    roles = de.split_roles(f.split)
    assert roles["Template/01_02"] == "dropped" and roles["Template/01_01"] == "train"
    assert _sha(b"INPUT 01_02") in de.forbidden_shas(de.dataset_index(f.root), roles, f.root)
    de.build_references_dir(f.rd, f.split)
    assert sorted(p.name for p in de.references_dir(f.rd).iterdir()) == ["MANIFEST.json", "Template_01_01"]


# ── the shipped drop-48 project split (offline read of the local dataset; skipped if absent) ───
DROP48_SPLIT = REPO / "examples/ssbv2/split.json"
DROP48_SHA = "5be714178186e14406b70efd8d00c51f2b135b0c61038ca8b2467d0f87a1da5d"


@pytest.mark.skipif(not (DROP48_SPLIT.is_file() and (de.data_root_or_none() or Path("/nonexistent")).is_dir()),
                    reason="needs the local SSB v2 dataset and the drop-48 split")
def test_real_drop48_split_references_are_57_train_goldens_and_dropped_stay_forbidden(tmp_path, monkeypatch):
    monkeypatch.delenv("EVOSKILL_SSBV2_DATA", raising=False)
    assert de.sha256_file(DROP48_SPLIT) == DROP48_SHA
    sp = json.loads(DROP48_SPLIT.read_text())
    dropped = sp["dropped"]["task_ids"]
    root = de.data_root()
    idx, roles = de.dataset_index(root), de.split_roles(DROP48_SPLIT)
    assert len(dropped) == 48 and all(roles[t] == "dropped" for t in dropped)
    # 9 of the 48 are also in the historical excluded_prior_run record; 39 were in train/val/test/unused
    assert len(set(dropped) & set(sp["excluded_prior_run"]["task_ids"])) == 9
    train = sorted(t for t, r in roles.items() if r == "train")
    assert len(train) == 57 and not set(train) & set(dropped)
    bad = de.forbidden_shas(idx, roles, root)
    for t in dropped:                                   # every dropped golden and input is forbidden
        assert de.sha256_file(idx[t]["golden"]) in bad and de.sha256_file(idx[t]["input"]) in bad
    labels = de.dataset_sha_labels(idx, roles, root)
    assert de.heldout_reference_collisions(idx, roles, root, labels) == []
    verdicts = {t: de.reference_label_verdict(labels[de.sha256_file(idx[t]["golden"])]) for t in train}
    assert not [t for t, (refused, _tol) in verdicts.items() if refused]     # no train golden = a dropped one
    tolerated = sorted(t for t, (_r, tol) in verdicts.items() if tol)
    assert len(tolerated) == 19                         # none of them dropped
    assert not [lb for _r, tol in verdicts.values() for lb in tol if lb.split(":")[-1] in set(dropped)]
    rd = tmp_path / "outputs" / "run_drop48"
    rd.mkdir(parents=True)
    out = de.build_references_dir(rd, DROP48_SPLIT)
    assert out["status"] == "built" and len(out["tolerated"]) == 19
    subdirs = sorted(p.name for p in de.references_dir(rd).iterdir() if p.is_dir())
    assert subdirs == sorted(t.replace("/", "_") for t in train) and len(subdirs) == 57
    assert de.verify_references_dir(rd, DROP48_SPLIT).status == "PASS"
    assert de.heldout_id_hits([de.references_dir(rd) / "MANIFEST.json"], DROP48_SPLIT) == []
    # a references dir holding a dropped task's golden (Template/16_09) fails verification on the
    # drop-48 split
    t = "Template/16_09"
    assert t in dropped
    extra = de.references_dir(rd) / t.replace("/", "_")
    extra.mkdir()
    (extra / Path(idx[t]["golden"]).name).write_bytes(Path(idx[t]["golden"]).read_bytes())
    c = de.verify_references_dir(rd, DROP48_SPLIT)
    assert c.status == "FAIL" and "Template_16_09: not a train task subdir" in c.detail
