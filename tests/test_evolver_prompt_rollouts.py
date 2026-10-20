"""The evolver prompt in the ROLLOUTS layout: src/orchestrator/evolver_template.py + the loop's prompt
checks that read it.

The evolver sees (1) high-fidelity rollout traces, (2) the path to the snapshot of each epoch's skills, so it
knows which skills were available and not only which ones were loaded, (2.5) a table of the skills rather than
their full text, and (3) each rollout's score in the prompt itself, so it need not open every trace to find it.
The prompt may name train task ids.

What these pin, on a SYNTHETIC two-epoch run built in tmp with the real rollout-evidence builder
(tests/test_rollout_evidence.make_run + rollout_evidence.build), never the live run:
  (a) the skill set is a table (name, SKILL.md description, files + sizes, where), never the skills' text;
  (b) every evidence dir's row names the skill snapshot its solver carried and the skills in it;
  (c) one per-rollout line per train task, re-derived by #208 field by field from INDEX.json and the leg JSON,
      and #208 fails on each kind of tampering;
  (d) the trace / grade / INDEX layout, stated from the INDEX and the trace bytes;
  #253 narrowed (train ids pass in both spellings, held-out / dropped ids and held-out figures still fail);
  #258 (every INDEX.md / _library/ / rollouts/ named; a named subpath must exist in the granted dir);
  the digests layout renders without touching any of it (the byte-identity of the live run's ITER1/ITER2 was
  checked separately on the real files and is reported, not pinned here: those files belong to the run).
No network, no model call, no write outside tmp.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.orchestrator.evolver_template as ET  # noqa: E402
import src.orchestrator.loop as L  # noqa: E402
from src.orchestrator import invariants as INV  # noqa: E402
from src.orchestrator import rollout_evidence as RE  # noqa: E402
from src.orchestrator import skill_library as SL  # noqa: E402
from tests import digest_info_fixture as DIF  # noqa: E402
from tests import test_rollout_evidence as TRE  # noqa: E402

PARAGRAPH = ("A verdict comes from comparing cell values between the saved output and the reference "
             "workbook, and nothing else.")
E1_TEXTS = None                                               # make_run's default: 2 of 6 pass
E2_TEXTS = {("Debugging/03_08", 0): TRE.PASS_TXT, ("Debugging/03_08", 1): TRE.FAIL_TXT,
            ("Debugging/03_08", 2): TRE.NOOUT_TXT, ("Financial_Model/06_01", 0): TRE.FAIL_TXT,
            ("Financial_Model/06_01", 1): TRE.PASS_TXT, ("Financial_Model/06_01", 2): TRE.PASS_TXT}


def _digests(r, library=None):
    """Per-task digests the loop's other consumers parse (Result line, turns, budget, the library header,
    one verdict line per sample), written from the leg the evidence is built from."""
    leg = json.loads(r.leg.read_text())
    head = [SL.header_line("skills_v1", SL.library_sha256(library)[:12])] if library else []
    for i, t in enumerate(leg["task_ids"]):
        sc = leg["per_task"][str(i)]
        k = sum(1 for x in sc if x >= 0.999)
        (r.out / f"{t.replace('/', '_')}.md").write_text("\n".join(
            [f"# {t}  ({t.split('/')[0]})", "",
             f"**Result: {k}/3 samples passed.**  turns=[2, 2, 45]  "
             f"budget_exhausted=['', '', 'Turn budget exhausted (45/45 turns)']", *head, "",
             "## Instruction (what the solver was asked to do)", "", "Fix it.", "", "Input file: `in.xlsx`", ""]
            + [f"## Rollout s{j} — {'PASS' if sc[j] >= 0.999 else 'FAIL'} (2 turns)\n" for j in range(3)]))


def make_rollouts_run(tmp: Path, e2_texts=None) -> SimpleNamespace:
    """<tmp>/run: digests_e1_train (base leg, no skills) and digests_e2_train (skills_v1), each with its
    per-task digests, a truthful DIGEST_INFO.json and rollout evidence built by rollout_evidence.build;
    the legs at the loop's names; skills_v1; a clean-grader dir; the template paragraph."""
    a = TRE.make_run(tmp / "a", library=False, texts=E1_TEXTS)
    b = TRE.make_run(tmp / "b", library=True, texts=e2_texts or E2_TEXTS)
    for r in (a, b):                                        # an SSB v2 leg states both totals
        leg = json.loads(r.leg.read_text())
        leg["sample_total"] = sum(len(v) for v in leg["per_task"].values())
        r.leg.write_text(json.dumps(leg))
    rd = tmp / "run"
    rd.mkdir()
    base = rd / "run_base_train.json"
    shutil.copyfile(a.leg, base)
    shutil.copyfile(b.leg, rd / "run_v1_train.json")
    shutil.copytree(b.lib, rd / "skills_v1")
    for r, j, lib in ((a, 1, None), (b, 2, b.lib)):
        _digests(r, lib)
        RE.build(r.out, run_dir=r.rd, leg_json=r.leg, split_path=r.split, n_samples=3, library=lib, epoch=j)
        d = rd / f"digests_e{j}_train"
        shutil.copytree(r.out, d)
        DIF.write_truthful_info(d)
    (rd / "grader_clean").mkdir()
    (rd / "grader_clean" / "scorer.py").write_text("def score_and_diff(*a):\n    return 0\n")
    (rd / "setup").mkdir()
    (rd / "setup" / "EVOLVER_TEMPLATE.md").write_text(PARAGRAPH + "\n")
    (rd / "oracle").mkdir()
    return SimpleNamespace(rd=rd, base=base, split=a.split, lib=rd / "skills_v1")


def _counts(rd, epoch):
    idx = json.loads((rd / f"digests_e{epoch}_train" / "INDEX.json").read_text())
    c = idx["counts"]
    return {k: c[k] for k in ET.OUTCOME_KEYS} | {"status": "ok", "reportable": True}


def render(r, epoch, *, layout="rollouts", **kw):
    facts = ET.facts_for_epoch(r.rd, epoch, base_leg=r.base)
    ep = L.EpochPaths(r.rd, epoch)
    return ET.render_parts(r.rd, epoch, evidence_dirs=ep.evidence_dirs(), staging_dir=ep.staging,
                           prior_library=r.lib if epoch == 2 else None, facts=facts,
                           template_paragraph_path=r.rd / "setup" / "EVOLVER_TEMPLATE.md", machine_facts={},
                           train_references=False, leg_json=(r.base if epoch == 1 else
                                                             L.EpochPaths(r.rd, 1).leg("train")),
                           outcomes=kw.pop("outcomes", None) or _counts(r.rd, epoch),
                           evidence_layout=layout, **kw)


def ctx_for(r, epoch, *, layout="rollouts"):
    spec = SimpleNamespace(benchmark="ssbv2", split_path=str(r.split), n_samples=3, name=r.rd.name,
                           domain="spreadsheetbench-v2", variant="", evolver_harness="claude_cli",
                           prompt_author_model="claude-opus-5-5")
    return L.Ctx(r.rd, epoch, spec, r.rd / "oracle", base_leg=str(r.base), evolver_evidence="all_epochs",
                 evidence_layout=layout)


CHECKS = (("#208", L._v_prompt_numbers), ("#253", L._v_prompt_no_leakage), ("#262", L._v_prompt_not_analytical),
          ("#258", L._v_prompt_names_evidence))


def checks(ctx, text):
    ctx.paths.prompt.write_text(text)
    return {name: fn(ctx) for name, fn in CHECKS}


@pytest.fixture
def run2(tmp_path):
    return make_rollouts_run(tmp_path)


def _row(rd, epoch, task, k):
    idx = json.loads((rd / f"digests_e{epoch}_train" / "INDEX.json").read_text())
    return next(x for x in idx["rows"] if x["task"] == task and x["sample"] == k)


# ══ the rendered prompt ════════════════════════════════════════════════════════════════════════════
def test_epoch2_renders_the_four_changes_and_passes_every_check(run2):
    r = run2
    parts = render(r, 2)
    text = parts["text"]
    e1, e2 = r.rd / "digests_e1_train", r.rd / "digests_e2_train"
    # (a) a table, never the skills' text
    assert "# alpha\n" not in text and "print(1)" not in text
    assert ("alpha\n  description: Use when alpha applies.\n  files: SKILL.md (" in text
            and "scripts/x.py (9 B)" in text)
    assert f"read their files in `{r.rd / 'skills_v2_staging'}/` or `{e2 / '_library'}/`" in text
    assert "`skills_v1` holds 2 skills in 4 files" in text
    assert parts["prior_library"]["rendered"] == "table"
    # (b) each epoch's snapshot and what was available
    assert (f"- `{e1}`: epoch 1, produced by the solver carrying no skills (the base leg): no skill was "
            f"available to its rollouts.") in text
    assert f"whose exact files are in `{e2}/_library/`. All 2 skills were available to every one of its " \
           f"rollouts: `alpha`, `beta`." in text
    assert f"its per-rollout scores are in `{e1}/INDEX.md`" in text
    # (c) the per-rollout lines, each equal to what the INDEX row says
    assert ET.ROLLOUT_SECTION in text and "skills: A = alpha, B = beta\n" in text
    tab = ET.parse_rollout_table(text)
    assert tab["bad"] == [] and sorted(tab["tasks"]) == ["Debugging_03_08", "Financial_Model_06_01"]
    for (task, k), want in ((("Debugging/03_08", 0), "passed"), (("Debugging/03_08", 1), "wrong"),
                            (("Debugging/03_08", 2), "no-output")):
        row = _row(r.rd, 2, task, k)
        p = tab["tasks"][task.replace("/", "_")][k]
        assert p["outcome"] == want and int(p["turns"]) == row["turns"]
        assert bool(p["cap"]) == row["turn_cap_hit"] and p["loaded"] == "A"
    assert tab["tasks"]["Debugging_03_08"][1]["m"] == "3" and tab["tasks"]["Debugging_03_08"][1]["g"] == "12"
    assert tab["tasks"]["Debugging_03_08"][2]["m"] is None, "no-output states no cells (its 0/0 counts no workbook)"
    assert f"`{e2}/rollouts/<task file stem>/s<k>.trace.jsonl`" in text
    # (d) the layout, from the INDEX / the bytes
    assert "The loop shortens nothing in it" in text and "with the reasoning text its model API returned" in text
    assert "`INDEX.md`, and the same rows as JSON in `INDEX.json`" in text
    assert "Each directory also holds one `<task file stem>.md` per task" in text
    assert "[SKILL LOADED:" not in text.split("## Per-rollout")[0].split("older, shorter summary")[0]
    # every loop check
    ctx = ctx_for(r, 2)
    for name, (ok, detail) in checks(ctx, text).items():
        assert ok, f"{name}: {detail}"
    assert "per-rollout lines (6 samples) re-derive" in L._v_prompt_numbers(ctx)[1]
    assert parts["bytes"] < 120_000 and parts["evidence_layout"] == "rollouts"
    assert [x["path"] for x in parts["rollout_index"]] == [str(e1 / "INDEX.json"), str(e2 / "INDEX.json")]


def test_epoch1_names_no_skill_and_passes_every_check(run2):
    r = run2
    text = render(r, 1)["text"]
    assert "No skill was available to these rollouts." in text and "skills: " not in text
    assert "## The skill set you are revising" not in text
    tab = ET.parse_rollout_table(text)
    assert tab["legend"] is None and all(p["loaded"] is None for ss in tab["tasks"].values() for p in ss.values())
    ctx = ctx_for(r, 1)
    for name, (ok, detail) in checks(ctx, text).items():
        assert ok, f"{name}: {detail}"


def test_the_digests_layout_never_reaches_the_rollouts_code(run2, monkeypatch):
    """Absent / "digests": the prompt as before (the live run's ITER1/ITER2 were re-rendered byte for byte
    with this code; see the module doc). Here: none of the new functions runs, the skills' text is pasted."""
    for name in ("_rollout_evidence", "_skill_set_table", "_rollout_table_section", "_evidence_section_rollouts",
                 "_trace_stats"):
        monkeypatch.setattr(ET, name, lambda *a, **k: (_ for _ in ()).throw(AssertionError("rollouts code ran")))
    text = render(run2, 2, layout="digests")["text"]
    assert "`alpha/SKILL.md`:" in text and "print(1)" in text and ET.ROLLOUT_SECTION not in text
    with pytest.raises(ET.RenderError, match="rollouts layout only"):
        render(run2, 2, layout="digests", regraded=[{"task": "Debugging/03_08", "sample": 1,
                                                    "outcome": "graded_wrong"}])
    with pytest.raises(ET.RenderError, match="is not one of"):
        render(run2, 2, layout="rollout")
    assert ET.LAYOUTS == RE.LAYOUTS


# ══ #208: every per-rollout line re-derives ═══════════════════════════════════════════════════════════
@pytest.mark.parametrize("old,new,needle", [
    ("s1 wrong 3/12 cells", "s1 passed 3/12 cells", "records score"),              # pass/fail vs the leg
    ("s1 wrong 3/12 cells", "s1 wrong 4/12 cells", "states cells (4, 12)"),       # cells vs INDEX
    ("s1 wrong 3/12 cells, 2 turns", "s1 wrong 3/12 cells, 3 turns", "3 turns"),   # turns vs INDEX
    ("s2 no-output, 45 turns (cap)", "s2 no-output, 45 turns", "(cap) mark"),     # the cap mark
    ("s2 no-output, 45 turns (cap), loaded A", "s2 no-output, 45 turns (cap), loaded B", "skills loaded"),
    ("s2 no-output, 45 turns", "s2 wrong, 45 turns", "says not_eligible"),         # outcome vs INDEX
    ("skills: A = alpha, B = beta", "skills: A = alpha, B = gamma", "legend"),
    ("s2 no-output, 45 turns", "s2 no-output 0/0 cells, 45 turns", "states cells (0, 0)"),
])
def test_208_fails_on_each_tampered_field(run2, old, new, needle):
    r = run2
    text = render(r, 2)["text"]
    line = next(ln for ln in text.split("\n") if ln.startswith("Debugging_03_08 | ") or ln.startswith("skills: "))
    assert old in text, (old, line)
    ctx = ctx_for(r, 2)
    ctx.paths.prompt.write_text(text.replace(old, new, 1))
    ok, detail = L._v_prompt_numbers(ctx)
    assert not ok and needle in detail, detail


def test_208_fails_on_a_missing_duplicated_or_unparseable_line(run2):
    r = run2
    text = render(r, 2)["text"]
    ctx = ctx_for(r, 2)
    fm = next(ln for ln in text.split("\n") if ln.startswith("Financial_Model_06_01 | "))
    for bad, needle in ((text.replace(fm + "\n", ""), "missing"),
                        (text.replace(fm, fm + "\n" + fm), "does not parse"),
                        (text.replace(fm, fm.replace(" turns", " turn")), "does not parse"),
                        (text.replace(fm, fm + " | s3 passed 1/1 cells, 2 turns, loaded none"), "not in the leg")):
        ctx.paths.prompt.write_text(bad)
        ok, detail = L._v_prompt_numbers(ctx)
        assert not ok and needle in detail, (needle, detail)
    # the rollouts layout REQUIRES the lines
    ctx.paths.prompt.write_text("\n".join(ln for ln in text.split("\n") if " | s0 " not in ln
                                          and not ln.startswith("skills: ")))
    ok, detail = L._v_prompt_numbers(ctx)
    assert not ok and "no per-rollout line" in detail


def test_208_takes_the_cell_fractions_out_of_the_batch_scan(run2):
    """A task whose graded cells equal the batch's episode count must not read as a /<episodes> figure."""
    r = run2
    text = render(r, 2)["text"]
    assert "3/12 cells" in text
    tab = ET.parse_rollout_table(text)
    assert "3/12" not in tab["residual"] and "3 of 6 episodes" in tab["residual"]
    # and in the digests layout the old scan is untouched: a stray N/6 figure still fails #208 there
    ctx = ctx_for(r, 2, layout="digests")
    ctx.paths.prompt.write_text(text.replace("3/12 cells", "5/6 cells"))
    ok, detail = L._v_prompt_numbers(ctx)
    assert not ok and "(5, 6)" in detail


def test_208_passes_a_cells_fraction_on_the_batch_denominator(tmp_path):
    """The batch is 6 episodes; a sample graded on 6 cells states "5/6 cells". Left in the headline scan that
    reads as an unsourced /6 batch figure; taken out and re-derived, the correct prompt passes."""
    five_of_six = TRE.FAIL_TXT.replace("3/12 graded cells", "5/6 graded cells")
    r = make_rollouts_run(tmp_path, e2_texts={**E2_TEXTS, ("Financial_Model/06_01", 0): five_of_six})
    text = render(r, 2)["text"]
    assert "s0 wrong 5/6 cells" in text
    ctx = ctx_for(r, 2)
    ctx.paths.prompt.write_text(text)
    ok, detail = L._v_prompt_numbers(ctx)
    assert ok, detail
    ctx.paths.prompt.write_text(text.replace("s0 wrong 5/6 cells", "s0 wrong 4/6 cells"))
    ok, detail = L._v_prompt_numbers(ctx)
    assert not ok and "states cells (4, 6)" in detail, detail


def test_208_fails_when_the_index_was_built_from_another_leg(run2):
    r = run2
    text = render(r, 2)["text"]
    ctx = ctx_for(r, 2)
    ctx.paths.prompt.write_text(text)
    leg = L.EpochPaths(r.rd, 1).leg("train")
    leg.write_text(leg.read_text() + " ")
    ok, detail = L._v_prompt_numbers(ctx)
    assert not ok and "not the current bytes" in detail
    with pytest.raises(ET.RenderError, match="was built from"):
        render(r, 2)


# ══ #253 narrowed ══════════════════════════════════════════════════════════════════════════════════
def test_253_names_train_ids_and_refuses_every_non_train_id_and_heldout_figure(run2):
    r = run2
    ctx = ctx_for(r, 2)
    body = "Batch: 4 of 6 episodes passed.\n"
    for extra, ok_want in (("See Debugging/03_08 and rollouts/Debugging_03_08/s0.trace.jsonl.", True),
                           ("See Debugging/05_01.", False), ("See rollouts/Debugging_05_01/.", False),  # val
                           ("See Template_06_20.", False),                                              # test
                           ("See Template/16_09.", False), ("See Template_02_01.", False),              # dropped, unused
                           ("It scored 3 of 6 on val.", False)):
        ctx.paths.prompt.write_text(body + extra + "\n")
        ok, detail = L._v_prompt_no_leakage(ctx)
        assert ok is ok_want, (extra, detail)
    assert any(s.name == "prompt names no held-out task id, no held-out figure" and s.verify is L._v_prompt_no_leakage
               for s in L.epoch_steps())


def test_253_invariants_both_spellings(tmp_path):
    sp = tmp_path / "s.json"
    sp.write_text(json.dumps({"train": ["Cat/01_01"], "val": ["Cat/02_01"], "test": ["Cat/03_01", "task_9"]}))
    pr = tmp_path / "p.md"
    for body, want in (("rollouts/Cat_01_01/s0 and Cat/01_01", "PASS"), ("Cat_02_01", "FAIL"),
                       ("x Cat/03_01.", "FAIL"), ("task_9", "FAIL"), ("task_90", "PASS"),
                       ("14 of 57 on validation", "FAIL")):
        pr.write_text(body + "\n")
        assert INV.check_prompt_no_leakage(pr, sp).status == want, body
    assert INV.id_mentions("a Cat_01_01 b", ["Cat/01_01"], allowed=["Cat/01_01"]) == []


# ══ #258: the paths inside the granted dirs ═══════════════════════════════════════════════════════════
def test_258_requires_index_library_rollouts_and_refuses_unreal_subpaths(run2):
    r = run2
    text = render(r, 2)["text"]
    ctx = ctx_for(r, 2)
    e1, e2 = r.rd / "digests_e1_train", r.rd / "digests_e2_train"
    for bad, needle in ((text.replace(f"`{e1}/INDEX.md`", "the INDEX"), "never names digests_e1_train/INDEX.md"),
                        (text.replace(f"`{e2}/_library/`", "its library"),
                         "never names digests_e2_train/_library/"),
                        (text.replace(f"`{e2}/rollouts/<task file stem>/s<k>.trace.jsonl`", "the traces")
                             .replace(f"`{e2}/rollouts/<task file stem>/s<k>.grade.txt`", "the grades"),
                         "never names digests_e2_train/rollouts/"),
                        (text + f"\nSee `{e2}/rollouts/Debugging_03_08/s7.trace.jsonl`.\n", "not in the granted dir"),
                        (text + f"\nSee `{e2}/../skills_v1/alpha`.\n", "not in the granted dir")):
        ctx.paths.prompt.write_text(bad)
        ok, detail = L._v_prompt_names_evidence(ctx)
        assert not ok and needle in detail, (needle, detail)
    ctx.paths.prompt.write_text(text + f"\nSee `{e2}/rollouts/Debugging_03_08/s1.trace.jsonl`.\n")
    assert L._v_prompt_names_evidence(ctx)[0]


# ══ the renderer refuses an INDEX it would misstate ════════════════════════════════════════════════════
def test_render_refuses_counts_or_libraries_the_index_contradicts(run2):
    r = run2
    c = _counts(r.rd, 2)
    with pytest.raises(ET.RenderError, match="the facts state"):
        render(r, 2, outcomes=dict(c, graded_wrong=c["graded_wrong"] + 1, not_eligible=c["not_eligible"] - 1))
    with pytest.raises(ET.RenderError, match="re-graded, but its INDEX row is graded_wrong"):
        render(r, 2, regraded=[{"task": "Debugging/03_08", "sample": 1, "outcome": "graded_wrong"}])
    ip = r.rd / "digests_e2_train" / "INDEX.json"
    idx = json.loads(ip.read_text())
    idx["producing_library"]["sha256_12"] = "000000000000"
    ip.write_text(json.dumps(idx))
    with pytest.raises(ET.RenderError, match="EVIDENCE INDEX"):
        render(r, 2)


def test_render_refuses_a_dir_without_rollout_evidence(run2):
    shutil.rmtree(run2.rd / "digests_e1_train" / "rollouts")
    with pytest.raises(ET.RenderError, match="digests_e1_train has no readable rollout evidence"):
        render(run2, 2)


# ══ units ══════════════════════════════════════════════════════════════════════════════════════════
def test_regraded_and_no_count_samples_render_and_parse():
    rows = [{"task": "C/01", "task_file": "C_01", "sample": 0, "outcome": "grader_failed", "cells_matched": None,
             "cells_graded": None, "turns": 45, "turn_cap_hit": True, "skills_loaded": ["b"]},
            {"task": "C/01", "task_file": "C_01", "sample": 1, "outcome": "graded_wrong", "cells_matched": None,
             "cells_graded": None, "turns": None, "turn_cap_hit": False, "skills_loaded": []}]
    lines = ET.rollout_table_lines({"rows": rows}, ET.skill_labels(["a", "b"]),
                                   {("C/01", 0): "graded_wrong"})
    assert lines == ["skills: A = a, B = b",
                     "C_01 | s0 wrong (re-graded), 45 turns (cap), loaded B | s1 wrong, ? turns, loaded none"]
    tab = ET.parse_rollout_table("\n".join(lines))
    assert tab["bad"] == [] and tab["tasks"]["C_01"][0]["regraded"] and tab["legend"] == {"A": "a", "B": "b"}


def test_skill_labels_and_descriptions():
    lab = ET.skill_labels([f"s{i:02d}" for i in range(28)])
    assert lab["s00"] == "A" and lab["s25"] == "Z" and lab["s26"] == "AA" and lab["s27"] == "AB"
    assert ET.skill_description("---\nname: x\ndescription: >\n  Use when a\n  b applies.\n---\nbody\n") == \
        "Use when a b applies."
    assert ET.skill_description("---\nname: x\ndescription: 'quoted: yes'\n---\n") == "quoted: yes"
    assert ET.skill_description("no frontmatter") is None
    assert ET.skill_description("---\nname: x\n---\ndescription: in the body\n") is None


def test_the_loop_passes_the_layout_and_the_regrade_rows_to_the_renderer():
    src = Path(L.__file__).read_text()
    body = src[src.index("def _render_prompt_parts"):src.index("def _run_render_prompt")]
    assert "evidence_layout=layout" in body and 'for x in oc.get("rows") or []' in body
    assert '"rollout_index"' in src[src.index("def _v_prompt_rendered"):src.index("def _dir_mention_problems")]


# ══ no adapter paragraph ═══════════════════════════════════════════════════════════════════════
def test_render_without_an_adapter_paragraph_holds_only_the_loops_verdict_text(tmp_path):
    """The benchmark adapter writes no EVOLVER_TEMPLATE.md. With
    template_paragraph_path=None the verdict section is the loop's own sentences, the record says so, every
    prompt check still passes, and the text differs from the with-paragraph render by exactly that paragraph.
    A path given must still exist (a run whose setup wrote a paragraph re-renders as before)."""
    r = make_rollouts_run(tmp_path)
    with_p = render(r, 1)
    facts = ET.facts_for_epoch(r.rd, 1, base_leg=r.base)
    ep = L.EpochPaths(r.rd, 1)
    common = dict(evidence_dirs=ep.evidence_dirs(), staging_dir=ep.staging, prior_library=None, facts=facts,
                  machine_facts={}, train_references=False, leg_json=r.base, outcomes=_counts(r.rd, 1),
                  evidence_layout="rollouts")
    parts = ET.render_parts(r.rd, 1, template_paragraph_path=None, **common)
    assert parts["template_paragraph"] is None and parts["template_paragraph_sha256"] is None
    assert PARAGRAPH in with_p["text"] and PARAGRAPH not in parts["text"]
    _, _, rest = parts["text"].partition("## How a verdict is produced\n\n")
    assert rest.split("\n\n## ")[0].startswith("The grader's code is in `"), rest[:160]
    assert with_p["text"].replace(PARAGRAPH + "\n\n", "") == parts["text"]
    res = checks(ctx_for(r, 1), parts["text"])
    assert all(ok for ok, _ in res.values()), res
    with pytest.raises(ET.RenderError, match="no AO template paragraph"):
        ET.render_parts(r.rd, 1, template_paragraph_path=r.rd / "setup" / "missing.md", **common)
