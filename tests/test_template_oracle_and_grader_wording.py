"""What the evolver is told about the oracle, the clean grader and a grader failure must be true.

(a) oracle/README.md says the reference workbooks are withheld, which contradicts D5. With RUNTIME
    prompt_mode="template" the upload carries NO oracle row (daytona_evolver.build_upload_manifest /
    check_upload_manifest, evolver_template_mode), the clean grader must travel instead, and the
    Daytona driver neither --add-dir's nor hashes the oracle dir. AO mode is unchanged.
(b) grader_clean is for READING: the rendered prompt says it runs only inside the grading image's
    LibreOffice setup, which the evolver's machine does not have -- a claim made only after
    daytona_evolver.evolver_image_grader_tooling() confirms it from the evolver's Dockerfile. The
    MANIFEST pins the README (readme_sha256), verify_grader_dir checks it, and the template-mode
    upload refuses a grader_clean whose MANIFEST pins no README.
(c) the grader-failed sentence is the loop's behaviour: re-grade once, halt if it fails again.

No network, no sandbox, no model call.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/daytona"))
import src.orchestrator.evolver_template as ET  # noqa: E402
from src.orchestrator import daytona_evolver as de  # noqa: E402
from src.orchestrator import evidence_surface as es  # noqa: E402
from src.orchestrator.invariants import check_prompt_not_analytical, held_out_figures  # noqa: E402
from tests.test_train_references import _fake  # noqa: E402

def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _grader(rd: Path, split: Path, *, step2: bool = True) -> Path:
    """A clean-grader dir as scripts/make_clean_grader.py leaves it: step2=True pins the README and
    records the evolver image as free of the grader's tooling; step2=False is a pre-step-2 build."""
    g = rd / de.GRADER_DIRNAME
    (g / "docker_scripts").mkdir(parents=True)
    files = {"scorer.py": "def score_and_diff(a, b, c, d):\n    return (True, 1, 1, '')\n",
             "docker_scripts/scan.py": "print('scan')\n"}
    for rel, body in files.items():
        (g / rel).write_text(body)
    readme = "# The grader\n\nThis code is here to READ.\n"
    (g / "README.md").write_text(readme)
    man = {"what": "clean grader", "split": str(split),
           "files": [{"file": rel, "clean_sha256": _sha(body.encode()), "task_id_hits": []}
                     for rel, body in files.items()]}
    if step2:
        man["readme_sha256"] = _sha(readme.encode())
        man["evolver_image"] = {"grader_tooling_absent": True}
    (g / "MANIFEST.json").write_text(json.dumps(man))
    return g


def _runtime(f, *, template: bool | None, grader=True, references=True):
    rt = {"evolver_daytona": {"train_references": references, "grader_clean": grader}}
    if template is not None:
        rt["prompt_mode"] = "template" if template else "ao"
    (f.rd / "RUNTIME.json").write_text(json.dumps(rt))


def _build(f, *, template_mode=None, grader=True, references=True):
    g = de.evolver_grants(f.rd)
    if references and not de.references_dir(f.rd).is_dir():
        de.build_references_dir(f.rd, f.split)
    venv = {"HOME": de.VM_HOME}
    rows, _gen = de.build_upload_manifest(
        run_dir=f.rd, epoch=1, prompt=f.rd / "ITER1_PROMPT.md", evidence_dirs=[f.ev],
        oracle=f.rd / "oracle", staging=f.rd / "skills_v1_staging", prev_library=f.rd / "skills_v0",
        references=de.references_dir(f.rd) if references else None, grants=g,
        grader=de.grader_dir(f.rd) if grader else None, grader_granted=grader,
        template_mode=template_mode, env_text=de.render_env_file(venv),
        hooks_text=de.hook_settings_text(), split=f.split)
    receipt = {"containment": "daytona", "sandboxed": True, "upload_manifest": rows, "grants": dict(g),
               "grader_granted": grader, "evidence_sha256_before": es.hashes([f.ev]),
               "daytona": {"vm_env": venv},
               "references_sha256_before": es.dir_sha256(de.references_dir(f.rd)) if references else None,
               "grader_sha256_before": es.dir_sha256(de.grader_dir(f.rd)) if grader else None}
    listing = {"pre_upload": {"users_exists": False},
               "after_upload": [[r["vm_path"], "F", r["sha256"], r["bytes"]] for r in rows]}
    return receipt, listing


def _check(f, receipt, listing, **kw):
    return de.check_upload_manifest(run_dir=f.rd, epoch=1, split=f.split, evidence_dirs=[f.ev],
                                    prev_library=f.rd / "skills_v0", prompt=f.rd / "ITER1_PROMPT.md",
                                    oracle=f.rd / "oracle", staging=f.rd / "skills_v1_staging",
                                    receipt=receipt, listing=listing, **kw)


def _cp(x):
    return json.loads(json.dumps(x))


def _oracle_row(f):
    p = f.rd / "oracle/README.md"
    return de._row(str(p), "oracle", str(p))


# ══ (a) no oracle README in template mode ═══════════════════════════════════════════════════════
def test_prompt_modes_are_the_loops():
    import src.orchestrator.loop as L
    assert de.PROMPT_MODES == L.PROMPT_MODES and de.TEMPLATE_MODE in L.PROMPT_MODES


def test_template_mode_is_read_from_runtime_and_checked(tmp_path):
    assert de.evolver_template_mode(tmp_path) is False                 # no RUNTIME.json: the AO world
    rt = tmp_path / "RUNTIME.json"
    for body, want in (({}, False), ({"prompt_mode": "ao"}, False), ({"prompt_mode": "template"}, True)):
        rt.write_text(json.dumps(body))
        assert de.evolver_template_mode(tmp_path) is want
    for bad in ({"prompt_mode": "tempate"}, {"prompt_mode": True}, ["prompt_mode"]):
        rt.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            de.evolver_template_mode(tmp_path)
    rt.write_text("{not json")
    with pytest.raises(ValueError):
        de.evolver_template_mode(tmp_path)
    assert de.evolver_template_mode(runtime={"prompt_mode": "template"}) is True


def test_template_upload_carries_no_oracle_row_and_passes(tmp_path, monkeypatch):
    f = _fake(tmp_path, monkeypatch)
    _grader(f.rd, f.split)
    _runtime(f, template=True)
    receipt, listing = _build(f)
    roles = [r["role"] for r in receipt["upload_manifest"]]
    assert "oracle" not in roles, "the AO-era oracle README must not reach a template-mode evolver"
    assert roles.count("grader") == 4 and roles.count("train_reference") == 3   # 2 goldens + MANIFEST
    assert not any("/oracle/" in r["vm_path"] for r in receipt["upload_manifest"])
    c = _check(f, receipt, listing)
    assert c.status == "PASS", c.detail
    assert c.evidence["template_mode"] is True and "no oracle README" in c.detail
    receipt["template_mode"] = True                                 # what the driver records
    assert _check(f, receipt, listing).status == "PASS"


def test_template_check_fails_an_oracle_row(tmp_path, monkeypatch):
    f = _fake(tmp_path, monkeypatch)
    _grader(f.rd, f.split)
    _runtime(f, template=True)
    receipt, listing = _build(f)
    x, lx = _cp(receipt), _cp(listing)
    row = _oracle_row(f)
    x["upload_manifest"].append(row)
    lx["after_upload"].append([row["vm_path"], "F", row["sha256"], row["bytes"]])
    c = _check(f, x, lx)
    assert c.status == "FAIL" and "prompt_mode=template uploads no oracle README" in c.detail, c.detail


def test_ao_mode_keeps_exactly_the_oracle_readme(tmp_path, monkeypatch):
    f = _fake(tmp_path, monkeypatch)
    _grader(f.rd, f.split)
    _runtime(f, template=None)                                      # prompt_mode absent = "ao"
    receipt, listing = _build(f)
    orc = [r for r in receipt["upload_manifest"] if r["role"] == "oracle"]
    assert [r["vm_path"] for r in orc] == [str(f.rd / "oracle/README.md")]
    c = _check(f, receipt, listing)
    assert c.status == "PASS", c.detail
    assert c.evidence["template_mode"] is False
    # an AO-mode upload re-checked after the run declared template mode: the oracle row is refused
    _runtime(f, template=True)
    c = _check(f, receipt, listing)
    assert c.status == "FAIL" and "uploads no oracle README" in c.detail
    # ...and an AO-mode upload that dropped the README still fails the AO rule
    _runtime(f, template=False)
    x, lx = _cp(receipt), _cp(listing)
    x["upload_manifest"] = [r for r in x["upload_manifest"] if r["role"] != "oracle"]
    lx["after_upload"] = [e for e in lx["after_upload"] if "/oracle/" not in e[0]]
    c = _check(f, x, lx)
    assert c.status == "FAIL" and "oracle rows are not exactly oracle/README.md" in c.detail


def test_receipt_template_mode_must_match_runtime(tmp_path, monkeypatch):
    f = _fake(tmp_path, monkeypatch)
    _grader(f.rd, f.split)
    _runtime(f, template=True)
    receipt, listing = _build(f)
    x = _cp(receipt)
    x["template_mode"] = False
    c = _check(f, x, listing)
    assert c.status == "FAIL" and "receipt records template_mode=False != the run's True" in c.detail


def test_template_mode_needs_the_clean_grader(tmp_path, monkeypatch):
    f = _fake(tmp_path, monkeypatch)
    _grader(f.rd, f.split)
    _runtime(f, template=True, grader=False)
    with pytest.raises(ValueError, match="clean grader"):
        _build(f, grader=False)
    # a receipt built without the grader (AO-style call), checked under template mode
    _runtime(f, template=None, grader=False)
    receipt, listing = _build(f, grader=False)
    x, lx = _cp(receipt), _cp(listing)
    x["upload_manifest"] = [r for r in x["upload_manifest"] if r["role"] != "oracle"]
    lx["after_upload"] = [e for e in lx["after_upload"] if "/oracle/" not in e[0]]
    _runtime(f, template=True, grader=False)
    c = _check(f, x, lx)
    assert c.status == "FAIL" and "grader_clean is not granted" in c.detail, c.detail


def test_explicit_template_mode_must_agree_with_runtime(tmp_path, monkeypatch):
    f = _fake(tmp_path, monkeypatch)
    _grader(f.rd, f.split)
    _runtime(f, template=None)
    with pytest.raises(ValueError, match="RUNTIME.json declares prompt_mode=ao"):
        _build(f, template_mode=True)
    receipt, listing = _build(f)
    c = _check(f, receipt, listing, template_mode=True)
    assert c.status == "FAIL" and "declares prompt_mode=ao" in c.detail
    # no RUNTIME.json at all: an explicit value is used (the grants are then passed explicitly)
    (f.rd / "RUNTIME.json").unlink()
    rows, _ = de.build_upload_manifest(
        run_dir=f.rd, epoch=1, prompt=f.rd / "ITER1_PROMPT.md", evidence_dirs=[f.ev],
        oracle=f.rd / "oracle", staging=f.rd / "skills_v1_staging", prev_library=f.rd / "skills_v0",
        references=de.references_dir(f.rd), grants={"train_inputs": False, "train_references": True},
        grader=de.grader_dir(f.rd), grader_granted=True, template_mode=True,
        env_text="", hooks_text="", split=f.split)
    assert "oracle" not in {r["role"] for r in rows}


def test_template_upload_refuses_a_pre_step2_clean_grader(tmp_path, monkeypatch):
    f = _fake(tmp_path, monkeypatch)
    _grader(f.rd, f.split, step2=False)
    _runtime(f, template=True)
    with pytest.raises(ValueError, match="pins no README"):
        _build(f)
    # the same dir is accepted in AO mode, as before (verify_grader_dir without the requirement)
    assert de.verify_grader_dir(f.rd, f.split).status == "PASS"
    c = de.verify_grader_dir(f.rd, f.split, require_readme_pin=True)
    assert c.status == "FAIL" and "pins no README" in c.detail and "grader_tooling_absent" in c.detail


def test_verify_grader_dir_checks_the_pinned_readme(tmp_path, monkeypatch):
    f = _fake(tmp_path, monkeypatch)
    g = _grader(f.rd, f.split)
    assert de.verify_grader_dir(f.rd, f.split, require_readme_pin=True).status == "PASS"
    (g / "README.md").write_text("# The grader\n\nIt runs with a LibreOffice profile.\n")
    c = de.verify_grader_dir(f.rd, f.split)
    assert c.status == "FAIL" and "README.md bytes differ from its MANIFEST readme_sha256" in c.detail
    (g / "README.md").unlink()
    c = de.verify_grader_dir(f.rd, f.split)
    assert c.status == "FAIL" and "README.md is absent" in c.detail


def test_driver_template_mode_uploads_mounts_and_hashes_no_oracle(tmp_path, monkeypatch):
    """The Daytona driver's prepare() (all on the Mac, before any SDK call; fake token, no sandbox)."""
    import tau3_evolver_daytona as drv
    f = _fake(tmp_path, monkeypatch)
    _grader(f.rd, f.split)
    monkeypatch.setenv("EVOSKILL_TAU3_EVOLVER_SPLIT", str(f.split))
    tok = tmp_path / "fake_token"
    tok.write_text(de.TOKEN_PREFIX + "FAKEFAKEFAKE")
    os.chmod(tok, 0o600)

    def prepare(tag):
        q = tmp_path / f"q_{tag}"
        r = drv.EvolverRun(epoch=1, prompt=f.rd / "ITER1_PROMPT.md", digests=f.ev,
                           staging=f.rd / "skills_v1_staging", run_dir=f.rd, prior=[],
                           model="claude-opus-5-5", oracle=f.rd / "oracle",
                           cfg={"oauth_token_file": str(tok)}, timeout_s=600,
                           receipt_path=q / "receipt.json", trace_path=q / "trace.json",
                           listing_path=q / "listing.json", log_path=q / "e.log", quarantine=q / "q")
        r.prepare()
        return r
    oracle = str(f.rd / "oracle")
    de.build_references_dir(f.rd, f.split)
    _runtime(f, template=True)
    r = prepare("template")
    assert "oracle" not in {x["role"] for x in r.rows}
    assert oracle not in r.add_dirs and oracle not in r.readonly_roots()
    assert r.receipt["template_mode"] is True and r.receipt["oracle_readme_uploaded"] is False
    lst = {"pre_upload": {"users_exists": False},
           "after_upload": [[x["vm_path"], "F", x["sha256"], x["bytes"]] for x in r.rows]}
    c = de.check_upload_manifest(run_dir=f.rd, epoch=1, split=f.split, evidence_dirs=r.evidence_dirs,
                                 prev_library=f.rd / "skills_v0", prompt=f.rd / "ITER1_PROMPT.md",
                                 oracle=f.rd / "oracle", staging=f.rd / "skills_v1_staging",
                                 receipt=r.receipt, listing=lst)
    assert c.status == "PASS", c.detail
    _runtime(f, template=None)
    r = prepare("ao")
    assert [x["role"] for x in r.rows].count("oracle") == 1
    assert oracle in r.add_dirs and oracle in r.readonly_roots()
    assert r.receipt["template_mode"] is False
    # a template run whose grader_clean predates step 2 is refused before any sandbox exists
    shutil.rmtree(de.grader_dir(f.rd))
    _grader(f.rd, f.split, step2=False)
    _runtime(f, template=True)
    with pytest.raises(drv.Refuse) as e:
        prepare("old_grader")
    assert e.value.code == drv.EXIT_MANIFEST and "pins no README" in str(e.value)


# ══ (b) the clean grader is for reading ═════════════════════════════════════════════════════════
def test_the_real_evolver_image_has_no_grader_tooling():
    absent, det, rec = de.evolver_image_grader_tooling()
    assert absent, det
    assert rec["removed_by"].startswith("RUN rm -rf /opt/evoskill")
    assert rec["run_instructions_mentioning_docker"] == 0
    assert de.main(["grader-tooling"]) == 0


@pytest.mark.parametrize("layer,why", [
    ("RUN useradd -m evo\n", "never removes /opt/evoskill"),
    ("RUN rm -rf /opt/evoskill\nRUN mkdir -p /opt/evoskill/verifier-lo-profile\n", "brings grader tooling back"),
    ("RUN rm -rf /opt/evoskill\nRUN apt-get install -y \\\n    docker.io\n", "mentions docker"),
    ("# RUN rm -rf /opt/evoskill\nRUN true\n", "never removes /opt/evoskill"),
])
def test_grader_tooling_check_catches_planted_images(tmp_path, layer, why):
    base = de.BASE_DOCKERFILE.read_bytes()
    evo = tmp_path / "Dockerfile.evo"
    evo.write_bytes(base + ("\n" + layer).encode())
    absent, det, _rec = de.evolver_image_grader_tooling(evo, de.BASE_DOCKERFILE)
    assert not absent and why in det, det
    evo.write_bytes(base + b"\nRUN rm -rf /opt/evoskill && test -z \"$(ls -A /opt)\"\n")
    assert de.evolver_image_grader_tooling(evo, de.BASE_DOCKERFILE)[0] is True


def test_prompt_grader_sentence_says_read_only_and_not_on_this_machine(tmp_path):
    s = ET.grader_sentence(tmp_path / "grader_clean", True, True)
    assert s.startswith(f"The grader's code is in `{tmp_path / 'grader_clean'}`: the grader's own "
                        f"implementation with comments and docstrings removed, the same program.")
    assert "It is there to read" in s and "That setup is not on your machine" in s
    s = ET.grader_sentence(tmp_path, False, None)                    # not the solver's image
    assert "It is there to read" in s and "not on your machine" not in s
    assert "the same program" not in s


def test_machine_section_restores_the_tooling_sentence():
    mf = {"image_recipe": "scripts/daytona/Dockerfile.ssb_leg", "image_sha12": "681a617ef3ce",
          "python": "3.11", "openpyxl": "3.1.3", "pandas": "2.2.0", "numpy": "1.26.4",
          "libreoffice": "LibreOffice 25.2.3.2"}
    text = ET._machine_section(mf, None, None)
    assert "plus the Claude Code CLI; the grader's own tooling is not in it." in text


def test_tooling_claim_is_made_only_when_the_image_confirms_it(monkeypatch):
    assert ET._grader_tooling_absent({}) is None
    assert ET._grader_tooling_absent({"python": "3.11"}) is True
    monkeypatch.setattr(de, "evolver_image_grader_tooling", lambda: (False, "planted docker", {}))
    with pytest.raises(ET.RenderError, match="not confirmed free of the grader's tooling"):
        ET._grader_tooling_absent({"python": "3.11"})


# ══ (c) the grader-failed sentence is the loop's step-2 behaviour ═══════════════════════════════
def _facts(tmp_path):
    newest = tmp_path / "digests_e1_train"
    newest.mkdir()
    return SimpleNamespace(n_tasks=2, n_samples=3, n_episodes=6, passed=2, reliable=0, unstable=2,
                           never_passes=0, result_headers_parsed=2, exhausted_episodes=0,
                           termination={}, tasks_with_exhausted=0, tasks_all_exhausted=0,
                           turns_min=0, turns_max=0, variant="", basis_counts={}), newest


def test_grader_failed_policy_is_regrade_once_then_halt(tmp_path):
    assert ET.REGRADE_ATTEMPTS == 1
    assert ET.GRADER_FAILED_POLICY == ("The loop re-grades such an episode once, from the saved "
                                       "workbook, and halts the run if the re-grade fails too.")
    facts, newest = _facts(tmp_path)
    counts = {"passed": 2, "graded_wrong": 3, "not_eligible": 1, "grader_failed": 0}
    text = ET._facts_section(facts, counts, True, newest)
    assert ("The loop re-grades such an episode once, from the saved workbook, and halts the run if "
            "the re-grade fails too, so none remains in this batch.") in text
    text = ET._facts_section(facts, {"passed": 2}, False, newest)
    assert ET.GRADER_FAILED_POLICY in text and "so none remains in this batch" not in text
    assert "stops if it still fails" not in text


def test_new_sentences_pass_the_prompt_checks(tmp_path):
    """The not-analytical rule and the held-out figure rule on every sentence about the grader."""
    body = "\n\n".join([ET.GRADER_FAILED_POLICY, ET.grader_sentence(tmp_path, True, True),
                        "plus the Claude Code CLI; the grader's own tooling is not in it."])
    p = tmp_path / "ITER1_PROMPT.md"
    p.write_text(body)
    c = check_prompt_not_analytical(p)
    assert c.status == "PASS", c.detail
    assert held_out_figures(body) == []


