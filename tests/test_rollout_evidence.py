"""Tests for the high-fidelity rollout evidence (src/orchestrator/rollout_evidence.py): the evolver sees
1) high-fidelity rollout traces, 2) the path to the snapshot of each epoch's available skills, 3) each
rollout's score.

What these pin:
  * the LAYOUT: rollouts/<stem>/s<k>.trace.jsonl (whole conversation, nothing clipped, reasoning kept, the
    system prompt stored once) + s<k>.grade.txt (the grader's bytes) + solver_context/ + INDEX.json/.md;
  * every INDEX row RE-DERIVES from its own files and from the leg (outcomes.py), and check_dir fails on each
    kind of tampering;
  * TRAIN ONLY: a non-train id in any file is redacted at build and refused at check; a held-out task dir under
    rollouts/ is HELD-OUT MATERIAL for evidence_surface (the loop's train-only check);
  * skills AVAILABLE (library snapshot, cross-checked with the system prompt's list) vs LOADED (from the trace);
  * the Claude Code adapter (stream-json + optional session JSONL);
  * the upload manifest check (daytona_evolver.evidence_index_problems);
  * the loop's layout flag and the skill-uptake read from INDEX.
No network, no model call, no write outside tmp.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_MOD = Path(os.environ.get("EVOSKILL_TEST_ROLLOUT_EVIDENCE", ROOT / "src/orchestrator/rollout_evidence.py"))
_spec = importlib.util.spec_from_file_location("rollout_evidence_under_test", _MOD)
RE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RE)
from src.orchestrator import evidence_surface as ES  # noqa: E402

MODEL = "deepseek/deepseek-v4.1-flash"
TRAIN = ["Debugging/03_08", "Financial_Model/06_01"]
VAL = ["Debugging/05_01"]
TEST = ["Template/06_20"]
DROPPED = ["Template/16_09"]
UNUSED = ["Template/02_01"]
T0 = dt.datetime(2026, 9, 26, 6, 0, 0)
HDR = "(scorer's own diff, computed at scoring time; full, uncapped)"
PASS_TXT = f"PASS — 12/12 graded cells match\n{HDR}\n\nAll cells match\n"
FAIL_TXT = f"FAIL — 3/12 graded cells match\n{HDR}\n\n'Sheet1'!B4: expected=5, got=None\n"
NOOUT_TXT = f"FAIL — 0/0 graded cells match\n{HDR}\n\nOutput file not found\n"
LONG = "row " + "x" * 5000 + " END-OF-LONG-OUTPUT"          # a digest would clip this to 1,200 + 600


def _system(skills):
    lines = "\n".join(f"- {n}: Use when {n} applies." for n in skills) or "(No skills available yet.)"
    return (f"You are a spreadsheet expert.\n\n## Skills\n\nSkills are detailed instruction sets.\n\n{lines}\n\n"
            f"IMPORTANT: You ONLY have access to these tool(s): bash, skill.\n")


def _messages(task, k, *, skills, load=None, capped=False, mention=None):
    msgs = [{"role": "system", "content": _system(skills)},
            {"role": "user", "content": f"Task: {task}\n\nFix it.\n"}]
    calls = []
    if load:
        calls.append({"id": f"c{k}a", "type": "function", "function": {"name": "skill",
                                                                        "arguments": json.dumps({"name": load})}})
    calls.append({"id": f"c{k}b", "type": "function",
                  "function": {"name": "bash", "arguments": json.dumps({"command": "python3 scan.py " + "y" * 3000})}})
    reasoning = "Let me look." + (f" Maybe a sibling task like {mention} exists." if mention else "")
    msgs.append({"role": "assistant", "content": "Starting.", "reasoning": reasoning,
                 "reasoning_details": [{"type": "reasoning.text", "text": reasoning, "format": "unknown", "index": 0}],
                 "tool_calls": calls})
    if load:
        msgs.append({"role": "tool", "tool_call_id": f"c{k}a",
                     "content": f"Base directory for this skill: /x/.claude/skills/{load}\n\n---\nname: {load}\n"
                                f"description: d\n---\n# {load}\nbody\n"})
    msgs.append({"role": "tool", "tool_call_id": f"c{k}b", "content": LONG})
    if capped:
        msgs.append({"role": "user", "content": "[HARNESS] Reached turn budget (45/45). Episode terminated."})
    else:
        msgs.append({"role": "assistant", "content": '{"finished": true, "confidence": 80}', "reasoning": "done",
                     "reasoning_details": [{"type": "reasoning.text", "text": "done", "format": "unknown", "index": 0}]})
    return msgs


def _record(task, k, when, score, **kw):
    q = json.dumps({"task_id": task, "instruction": "Fix it."})
    capped = kw.get("capped", False)
    tr = {"model": MODEL, "messages": _messages(task, k, **kw), "num_turns": 45 if capped else 2,
          "budget_exhausted": "Turn budget exhausted (45/45 turns)" if capped else "",
          "is_error": capped, "parse_error": "No JSON object found" if capped else None,
          "output": {"finished": None if capped else True}, "result": "r", "total_cost_usd": 0.01,
          "duration_ms": 1000, "usage": {"prompt_tokens": 1}, "inline_score": score,
          "tool_schemas": [{"type": "function", "function": {"name": "bash"}}],
          "cached_transcript": "### rendering", "uuid": "u", "session_id": "s"}
    return q, {"version": "1.0", "created_at": when.isoformat(), "trace": tr,
               "cache_key": {"tree_hash": "f22fc16850cf", "question_hash": "x", "question": q, "sdk": "direct",
                             "model": MODEL, "effort": ""}}


def _library(tmp: Path, names=("alpha", "beta"), label="skills_v1") -> Path:
    lib = tmp / label
    for n in names:
        (lib / n / "scripts").mkdir(parents=True)
        (lib / n / "SKILL.md").write_text(f"---\nname: {n}\ndescription: Use when {n} applies.\n---\n# {n}\n")
        (lib / n / "scripts" / "x.py").write_text("print(1)\n")
    return lib


def make_run(tmp: Path, *, library=True, texts=None, mention=None, leg_tasks=None, skills_told=None):
    """A run dir with a split, a leg store (one final attempt) whose cache records and snapshots outcomes.py
    links, and optionally skills_v1. Returns a namespace."""
    rd = tmp / "run"
    rd.mkdir(parents=True)
    split = tmp / "split.json"
    split.write_text(json.dumps({"train": TRAIN, "val": VAL, "test": TEST, "unused": UNUSED,
                                 "dropped": {"task_ids": DROPPED}}))
    lib = _library(rd) if library else None
    names = ["alpha", "beta"] if library else []
    store = tmp / "store" / "leg_x"
    att = store / "s01"
    texts = texts or {("Debugging/03_08", 0): PASS_TXT, ("Debugging/03_08", 1): FAIL_TXT,
                      ("Debugging/03_08", 2): NOOUT_TXT, ("Financial_Model/06_01", 0): FAIL_TXT,
                      ("Financial_Model/06_01", 1): PASS_TXT, ("Financial_Model/06_01", 2): FAIL_TXT}
    scores, cap = {}, 0
    tasks = leg_tasks or TRAIN
    for i, ((t, k), txt) in enumerate(sorted(texts.items())):
        when = T0 + dt.timedelta(minutes=i)
        snap = att / "tasks" / "_snapshots" / t.replace("/", "_") / f"iter_0_s{k}"
        snap.mkdir(parents=True)
        (snap / "DIFF.txt").write_text(txt)
        ts = when.replace(tzinfo=dt.timezone.utc).timestamp()
        os.utime(snap / "DIFF.txt", (ts, ts))
        capped = (k == 2)
        cap += capped
        score = 1.0 if txt is PASS_TXT else 0.0
        q, rec = _record(t, k, when + dt.timedelta(seconds=1), score, skills=skills_told if skills_told is not None
                         else names, load=("alpha" if library else None), capped=capped,
                         mention=(mention if (t, k) == ("Debugging/03_08", 1) else None))
        stem = RE.OC._record_path_stem(q, "direct", MODEL, "", k)
        p = att / "cache" / "f22fc16850cf" / f"{stem}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec))
        scores.setdefault(t, [None, None, None])[k] = score
    leg = {"tag": "leg_x", "model": MODEL, "budget": 45, "n_samples": 3, "task_ids": tasks,
           "per_task": {str(i): scores[t] for i, t in enumerate(tasks)},
           "sample_pass": sum(1 for t in tasks for x in scores[t] if x >= 0.999), "turn_exhausted": cap,
           "daytona": {"store": str(store)}}
    legp = rd / "leg_x_train.json"
    legp.write_text(json.dumps(leg))
    out = rd / ("digests_e2_train" if library else "digests_e1_train")
    out.mkdir()
    # the per-task digests the loop's other steps read (a minimal stand-in: the verdict line per sample,
    # which rollout_evidence.digest_agreement cross-checks)
    for t in tasks:
        (out / f"{t.replace('/', '_')}.md").write_text(
            f"# {t}\n\n" + "".join(f"## Rollout s{k} — {'PASS' if scores[t][k] >= 0.999 else 'FAIL'} (2 turns)\n\n"
                                    for k in range(3)))
    return SimpleNamespace(rd=rd, split=split, lib=lib, leg=legp, out=out, att=att)


def build(r, **kw):
    return RE.build(r.out, run_dir=r.rd, leg_json=r.leg, split_path=r.split, n_samples=3, library=r.lib,
                    epoch=2 if r.lib else 1, **kw)


def check(r, **kw):
    return RE.check_dir(r.out, split_path=r.split, n_samples=3, library=r.lib, leg_json=r.leg, run_dir=r.rd, **kw)


# ══ the layout ═════════════════════════════════════════════════════════════════════════════════════
def test_build_writes_the_layout_and_every_row_rederives(tmp_path):
    r = make_run(tmp_path)
    s = build(r)
    assert s["counts"]["rollouts"] == 6 and s["counts"]["tasks"] == 2
    assert (s["counts"]["passed"], s["counts"]["graded_wrong"], s["counts"]["not_eligible"]) == (2, 3, 1)
    assert s["counts"]["turn_cap_hit"] == 2 and s["counts"]["rollouts_that_loaded_a_skill"] == 6
    idx = json.loads((r.out / "INDEX.json").read_text())
    assert idx["schema"] == RE.SCHEMA and idx["evidence_dir"] == "digests_e2_train"
    for t in TRAIN:
        for k in range(3):
            d = r.out / "rollouts" / t.replace("/", "_")
            assert (d / f"s{k}.trace.jsonl").is_file() and (d / f"s{k}.grade.txt").is_file()
    assert sorted(p.name for p in (r.out / "solver_context").iterdir())[0].startswith("system_prompt.")
    assert len(list((r.out / "solver_context").glob("system_prompt.*.txt"))) == 1     # once per dir
    assert (r.out / "_library" / "alpha" / "SKILL.md").is_file()                       # the snapshot
    assert check(r) == []
    row = next(x for x in idx["rows"] if x["task"] == "Debugging/03_08" and x["sample"] == 1)
    assert row["outcome"] == "graded_wrong" and (row["cells_matched"], row["cells_graded"]) == (3, 12)
    assert row["skills_available"] == {"library": "skills_v1", "path": "_library", "names": ["alpha", "beta"]}
    assert row["skills_loaded"] == ["alpha"] and row["skills_not_loaded"] == ["beta"]
    assert row["trace"] == "rollouts/Debugging_03_08/s1.trace.jsonl"
    noout = next(x for x in idx["rows"] if x["task"] == "Debugging/03_08" and x["sample"] == 2)
    assert noout["outcome"] == "not_eligible" and noout["turn_cap_hit"] and noout["termination"] == "turn_cap"
    # the grade is the grader's bytes
    assert (r.out / row["grade"]).read_text() == FAIL_TXT


def test_the_trace_is_whole_and_ordered(tmp_path):
    r = make_run(tmp_path)
    build(r)
    lines = [json.loads(x) for x in (r.out / "rollouts/Debugging_03_08/s2.trace.jsonl").read_text().splitlines()]
    h, e = lines[0], lines[-1]
    assert h["type"] == "header" and h["schema"] == RE.TRACE_SCHEMA and (h["task"], h["sample"]) == ("Debugging/03_08", 2)
    assert h["system_prompt"].startswith("solver_context/system_prompt.")
    assert lines[1] == {"type": "message", "n": 0, "role": "system", "ref": h["system_prompt"],
                        "sha256": hashlib.sha256((r.out / h["system_prompt"]).read_bytes()).hexdigest()}
    body = lines[2:-1]
    assert [x["n"] for x in body] == list(range(1, len(body) + 1))
    a = next(x for x in body if x["role"] == "assistant")
    assert a["reasoning"] == "Let me look." and "reasoning_details" not in a          # identical -> dropped
    assert h["stats"]["reasoning_details_identical_dropped"] == 1
    assert a["tool_calls"][1]["arguments"]["command"].endswith("y" * 3000)            # full arguments
    tool = [x for x in body if x["role"] == "tool"]
    assert tool[0]["name"] == "skill" and "# alpha\nbody" in tool[0]["content"]      # skill body kept in full
    assert tool[1]["content"] == LONG                                                   # never clipped
    assert body[-1]["from_harness"] is True and body[-1]["content"].startswith("[HARNESS]")
    assert e == {**e, "type": "end", "turns": 45, "turn_cap_hit": True, "termination": "turn_cap"}
    assert "### rendering" not in json.dumps(lines) and "uuid" not in h        # cached_transcript not copied


def test_index_md_rerenders_and_carries_no_digest_marker(tmp_path):
    r = make_run(tmp_path)
    build(r)
    md = (r.out / "INDEX.md").read_text()
    assert md == RE.render_md(json.loads((r.out / "INDEX.json").read_text()))
    for pat in RE.DIGEST_PATTERNS:
        assert not pat.search(md), pat.pattern
    # the readers that glob *.md: the launcher's batch-rate gate, derive_facts, the #234 section split
    assert not re.search(r"\*\*Result: (\d+)/(\d+) samples passed\.\*\*", md)
    assert "## Rollout " not in md and "[SKILL LOADED:" not in md
    assert "`./_library/alpha/SKILL.md`" in md and "skills_v1" in md


def test_base_surface_has_no_library_and_refuses_skill_text(tmp_path):
    r = make_run(tmp_path, library=False)
    build(r)
    idx = json.loads((r.out / "INDEX.json").read_text())
    assert idx["producing_library"] is None and not (r.out / "_library").exists()
    assert all(x["skills_available"] == {"library": None, "path": None, "names": []} and x["skills_loaded"] == []
               for x in idx["rows"])
    assert check(r) == []
    # a base rollout whose tool output carries skill frontmatter is contamination
    r2 = make_run(tmp_path / "b", library=False)
    rec = next((r2.att / "cache" / "f22fc16850cf").glob("*.json"))
    d = json.loads(rec.read_text())
    d["trace"]["messages"][-2]["content"] = "---\nname: xlsx-helper\ndescription: x\n---\nbody"
    rec.write_text(json.dumps(d))
    with pytest.raises(RE.EvidenceError, match="BASE LEG CONTAMINATED"):
        build(r2)
    assert not (r2.out / "rollouts").exists() and not (r2.out / "INDEX.json").exists()


# ══ train only ════════════════════════════════════════════════════════════════════════════════════
@pytest.mark.parametrize("mention", ["Debugging/05_01", "Template_06_20", "Template/16_09", "Template_02_01"])
def test_a_non_train_id_in_a_trace_is_redacted_and_counted(tmp_path, mention):
    r = make_run(tmp_path, mention=mention)
    build(r)
    t = (r.out / "rollouts/Debugging_03_08/s1.trace.jsonl").read_text()
    assert mention not in t and RE.REDACTION in t
    idx = json.loads((r.out / "INDEX.json").read_text())
    assert idx["counts"]["redactions"] == 1
    assert ES.text_id_hits(r.out, ES.IdScanner.for_split(r.split)) == []
    assert check(r) == [] and ES.surface_problems(r.out, r.split) == []


def test_train_ids_are_not_redacted(tmp_path):
    r = make_run(tmp_path, mention="Financial_Model/06_01")
    build(r)
    assert "Financial_Model/06_01" in (r.out / "rollouts/Debugging_03_08/s1.trace.jsonl").read_text()


def test_the_scanner_covers_what_heldout_id_hits_flags(tmp_path):
    """Parity with daytona_evolver.heldout_id_hits (the post-run audit): whatever it flags, this flags."""
    from src.orchestrator import daytona_evolver as de
    r = make_run(tmp_path)
    corpus = ["x Debugging/05_01 y", "'Template/06_20'", "Template_16_09.", "a/Debugging/05_01", "Debugging_05_01__s0",
              "Debugging/05_010", "XDebugging/05_01", "Debugging/03_08", "(Template_06_20)", "Template/02_01\n"]
    sc = ES.IdScanner.for_split(r.split)
    for i, text in enumerate(corpus):
        f = tmp_path / f"c{i}.txt"
        f.write_text(text)
        theirs = {h.split(":", 1)[1] for h in de.heldout_id_hits([f], r.split)}
        assert theirs <= set(sc.find(text)), (text, theirs, sc.find(text))
    assert sc.find("Debugging/03_08 and Debugging_03_08") == []                          # train
    assert sc.find("a/Debugging/05_01") == ["Debugging/05_01"]                           # superset of theirs


def test_leg_with_a_non_train_task_is_refused(tmp_path):
    r = make_run(tmp_path, texts={("Debugging/05_01", k): PASS_TXT for k in range(3)}, leg_tasks=["Debugging/05_01"])
    with pytest.raises(RE.EvidenceError, match="NOT train tasks"):
        build(r)


def test_surface_problems_judges_the_rollouts_subtree(tmp_path):
    r = make_run(tmp_path)
    build(r)
    (r.out / "Debugging_03_08.md").write_text("# a digest\n")
    assert ES.surface_problems(r.out, r.split) == []
    held = r.out / "rollouts" / "Debugging_05_01"
    held.mkdir()
    (held / "s0.trace.jsonl").write_text("{}\n")
    assert any("HELD-OUT MATERIAL" in p for p in ES.surface_problems(r.out, r.split))
    held.joinpath("s0.trace.jsonl").unlink()
    held.rmdir()
    (r.out / "rollouts" / "Debugging_03_08" / "notes.txt").write_text("x")
    assert any("other than s<k>.trace.jsonl" in p for p in ES.surface_problems(r.out, r.split))
    (r.out / "rollouts" / "Debugging_03_08" / "notes.txt").unlink()
    (r.out / "solver_context" / "extra.txt").write_text("x")
    assert any("unexpected entries" in p for p in ES.surface_problems(r.out, r.split))
    (r.out / "solver_context" / "extra.txt").unlink()
    (r.out / "Debugging_03_08.md").write_text("see Template/06_20\n")                    # any file, any layout
    assert any("NON-TRAIN TASK IDS" in p for p in ES.surface_problems(r.out, r.split))


# ══ check_dir fails on each kind of tampering ═══════════════════════════════════════════════════════
def _mutate(r, what):
    tr = r.out / "rollouts/Debugging_03_08/s1.trace.jsonl"
    gr = r.out / "rollouts/Debugging_03_08/s1.grade.txt"
    ix = r.out / "INDEX.json"
    if what == "trace_byte":
        tr.write_text(tr.read_text().replace("Let me look.", "Let me look!"))
    elif what == "grade_deleted":
        gr.unlink()
    elif what == "stray_file":
        (r.out / "rollouts/Debugging_03_08/s9.trace.jsonl").write_text("{}\n")
    elif what == "index_outcome":
        d = json.loads(ix.read_text())
        next(x for x in d["rows"] if x["sample"] == 1 and x["task"] == "Debugging/03_08")["outcome"] = "passed"
        ix.write_text(json.dumps(d))
    elif what == "index_loaded":
        d = json.loads(ix.read_text())
        d["rows"][0]["skills_loaded"] = ["alpha", "beta"]
        ix.write_text(json.dumps(d))
    elif what == "row_dropped":
        d = json.loads(ix.read_text())
        d["rows"] = d["rows"][1:]
        ix.write_text(json.dumps(d))
    elif what == "library_drift":
        (r.out / "_library/alpha/SKILL.md").write_text("changed")
    elif what == "index_md":
        (r.out / "INDEX.md").write_text("# edited\n")
    elif what == "leg_changed":
        d = json.loads(r.leg.read_text())
        d["note"] = "rewritten"
        r.leg.write_text(json.dumps(d))
    elif what == "heldout_text":
        g = r.out / "rollouts/Debugging_03_08/s0.grade.txt"
        g.write_text(g.read_text() + "Debugging/05_01\n")


@pytest.mark.parametrize("what,needle", [
    ("trace_byte", "trace bytes differ"), ("grade_deleted", "grade missing"), ("stray_file", "not in INDEX.json"),
    ("index_outcome", "re-derives as graded_wrong"), ("index_loaded", "skills loaded re-derive"),
    ("row_dropped", "rows != train x 3"), ("library_drift", "not byte-identical"),
    ("index_md", "does not re-render"), ("leg_changed", "changed since the evidence was built"),
    ("heldout_text", "NON-TRAIN TASK IDS")])
def test_check_dir_fails_on_tampering(tmp_path, what, needle):
    r = make_run(tmp_path)
    build(r)
    assert check(r) == []
    _mutate(r, what)
    probs = check(r)
    assert any(needle in p for p in probs), probs


def test_builder_never_builds_over_evidence(tmp_path):
    r = make_run(tmp_path)
    build(r)
    with pytest.raises(RE.EvidenceError, match="already exists"):
        build(r)


def test_system_prompt_skill_list_must_equal_the_library(tmp_path):
    r = make_run(tmp_path, skills_told=["alpha"])
    with pytest.raises(RE.EvidenceError, match="system prompt listed skills"):
        build(r)


def test_a_cache_record_that_disagrees_with_the_digest_cache_is_refused(tmp_path):
    r = make_run(tmp_path)
    other = tmp_path / "digest_cache"
    (other / "f22fc16850cf").mkdir(parents=True)
    for p in (r.att / "cache" / "f22fc16850cf").glob("*.json"):
        (other / "f22fc16850cf" / p.name).write_bytes(p.read_bytes())
    victim = sorted((other / "f22fc16850cf").glob("*.json"))[0]
    victim.write_text(victim.read_text().replace("Let me look.", "Other."))
    with pytest.raises(RE.EvidenceError, match="not byte-identical in the digest cache"):
        build(r, cross_check_cache=other)
    assert not (r.out / "INDEX.json").exists()


def test_digest_agreement_catches_a_digest_with_another_verdict(tmp_path):
    r = make_run(tmp_path)
    build(r)
    (r.out / "Debugging_03_08.md").write_text("## Rollout s0 — FAIL (2 turns)\n## Rollout s1 — FAIL (2 turns)\n"
                                              "## Rollout s2 — FAIL (45 turns)\n")
    assert any("the digest says FAIL" in p for p in check(r))


# ══ Claude Code adapter ═════════════════════════════════════════════════════════════════════════════
def _cc_stream():
    ev = [{"type": "system", "subtype": "init", "cwd": "/evo/work", "session_id": "S", "uuid": "U", "model": "claude-sonnet-5",
           "tools": ["Bash", "Read"], "claude_code_version": "2.1.281", "messaging_socket_path": "/tmp/x"},
          {"type": "assistant", "message": {"id": "m1", "model": "claude-sonnet-5", "content": [
              {"type": "thinking", "thinking": "", "signature": "sig"}]}},
          {"type": "system", "subtype": "thinking_tokens", "estimated_tokens": 5},
          {"type": "assistant", "message": {"id": "m1", "model": "claude-sonnet-5", "content": [
              {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls " + "z" * 4000}}]}},
          {"type": "rate_limit_event", "rate_limit_info": {"status": "allowed"}},
          {"type": "user", "message": {"role": "user", "content": [
              {"type": "tool_result", "tool_use_id": "t1", "content": LONG, "is_error": False}]}},
          {"type": "assistant", "message": {"id": "m2", "model": "claude-sonnet-5", "content": [
              {"type": "thinking", "thinking": "summary of my plan", "signature": "s2"}]}},
          {"type": "assistant", "message": {"id": "m2", "model": "claude-sonnet-5", "content": [
              {"type": "text", "text": "Done."}], "stop_reason": "end_turn"}},
          {"type": "result", "subtype": "error_max_turns", "is_error": True, "num_turns": 45, "terminal_reason": "max_turns",
           "result": "Done.", "total_cost_usd": 0.5, "duration_ms": 9, "usage": {"output_tokens": 3}}]
    return "\n".join(json.dumps(e) for e in ev) + "\n{torn"


def test_cc_trace_groups_one_reply_per_message_id_and_keeps_everything():
    tr = RE.cc_trace(_cc_stream(), user_message="Task: Debugging/03_08\n", system_prompt="appended")
    ls = tr["lines"]
    assert [x["role"] for x in ls] == ["user", "assistant", "tool", "assistant"]
    assert [x["n"] for x in ls] == [1, 2, 3, 4]
    a1, tool, a2 = ls[1], ls[2], ls[3]
    assert a1["tool_calls"] == [{"id": "t1", "name": "Bash", "arguments": {"command": "ls " + "z" * 4000}}]
    assert a1["reasoning_redacted"] == 1 and a1["reasoning"] == ""
    assert tool["name"] == "Bash" and tool["content"] == LONG and tool["is_error"] is False
    assert a2["reasoning"] == "summary of my plan" and a2["content"] == "Done." and a2["stop_reason"] == "end_turn"
    e = tr["end"]
    assert e["turn_cap_hit"] is True and e["termination"] == "turn_cap" and e["turns"] == 45 and e["model_replies"] == 2
    assert tr["stats"]["skipped:system/thinking_tokens"] == 1 and tr["stats"]["skipped:rate_limit_event"] == 1
    assert tr["stats"]["unparsed_stream_lines"] == 1
    assert "session_id" not in tr["init"] and "messaging_socket_path" not in tr["init"] and tr["tools"] == ["Bash", "Read"]
    assert tr["system"] == "appended" and "not recorded" in tr["system_note"]


def test_cc_trace_takes_messages_from_the_session_file():
    sess = [{"type": "queue-operation", "operation": "enqueue", "content": "Task"},
            {"type": "user", "message": {"role": "user", "content": "Task: Debugging/03_08\n"}},
            {"type": "attachment", "attachment": {"type": "environment"},
             "rendered": [{"content": "<system-reminder>env</system-reminder>"}]},
            {"type": "assistant", "message": {"id": "m1", "content": [{"type": "text", "text": "Hi"}]}},
            {"type": "cost-state"}]
    tr = RE.cc_trace(_cc_stream(), session_text="\n".join(json.dumps(x) for x in sess))
    roles = [(x["role"], x.get("from_harness", False)) for x in tr["lines"]]
    assert roles == [("user", False), ("user", True), ("assistant", False)]
    assert tr["lines"][1]["kind"] == "attachment" and tr["end"]["turns"] == 45


def test_a_claude_code_record_routes_through_the_cc_adapter():
    rec = {"cache_key": {"sdk": "claude_code"}, "trace": {"stream_jsonl": _cc_stream(), "user_message": "T"}}
    assert RE.trace_for_record(rec)["solver"] == "claude_code"
    with pytest.raises(RE.EvidenceError, match="stream_jsonl"):
        RE.trace_for_record({"cache_key": {"sdk": "claude_code"}, "trace": {}})
    with pytest.raises(RE.EvidenceError, match="no trace adapter"):
        RE.trace_for_record({"cache_key": {"sdk": "codex"}, "trace": {}})


def test_skill_loads_from_a_cc_trace():
    ls = [{"type": "message", "role": "assistant", "tool_calls": [{"id": "a", "name": "Skill", "arguments": {"skill": "alpha"}},
                                                                   {"id": "b", "name": "Skill", "arguments": {"skill": "gamma"}}]},
          {"type": "message", "role": "tool", "tool_call_id": "a", "content": "loaded", "is_error": False},
          {"type": "message", "role": "tool", "tool_call_id": "b", "content": "no such skill", "is_error": True}]
    s = RE.skill_loads(ls, ["alpha", "beta"])
    assert s["loaded"] == ["alpha"] and s["not_loaded"] == ["beta"] and s["skill_tool_errors"] == ["gamma"]


# ══ the upload manifest check ═══════════════════════════════════════════════════════════════════════
def _rows(d):
    return [{"vm_path": str(p), "role": "evidence", "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(Path(os.path.realpath(d)).rglob("*")) if p.is_file()]


def test_upload_rows_must_carry_every_indexed_file_at_its_sha(tmp_path):
    from src.orchestrator import daytona_evolver as de
    r = make_run(tmp_path)
    build(r)
    rows = _rows(r.out)
    assert de.evidence_index_problems(rows, [r.out]) == []
    dropped = [x for x in rows if not x["vm_path"].endswith("s1.trace.jsonl")]
    assert any("not evidence rows at the sha" in p for p in de.evidence_index_problems(dropped, [r.out]))
    wrong = [dict(x, sha256="0" * 64) if x["vm_path"].endswith("s0.grade.txt") else x for x in rows]
    assert any("not evidence rows at the sha" in p for p in de.evidence_index_problems(wrong, [r.out]))
    extra = rows + [{"vm_path": os.path.realpath(r.out) + "/rollouts/Debugging_03_08/s7.grade.txt", "role": "evidence",
                     "sha256": "1" * 64}]
    assert any("does not list" in p for p in de.evidence_index_problems(extra, [r.out]))
    no_index = [x for x in rows if not x["vm_path"].endswith("INDEX.json")]
    assert any("INDEX.json is not an evidence row" in p for p in de.evidence_index_problems(no_index, [r.out]))
    # a legacy dir (no INDEX.json) may carry no rollout rows
    (r.out / "INDEX.json").rename(r.out / "INDEX.json.x")
    assert any("no INDEX.json lists them" in p for p in de.evidence_index_problems(rows, [r.out]))


# ══ the loop ═══════════════════════════════════════════════════════════════════════════════════════
def _ctx(r, epoch=2, layout="rollouts", benchmark="ssbv2"):
    from src.orchestrator import loop as L
    spec = SimpleNamespace(split_path=str(r.split), n_samples=3, benchmark=benchmark, name="run",
                           solver_model=MODEL, evolver_harness="claude_cli")
    return L, L.Ctx(r.rd, epoch, spec, r.rd / "oracle", evidence_layout=layout, base_leg=str(r.leg))


def test_loop_refuses_an_unknown_or_unbuildable_layout(tmp_path):
    r = make_run(tmp_path)
    L, ctx = _ctx(r, layout="rollout")
    with pytest.raises(L.Halt, match="is not one of"):
        L._evidence_layout(ctx)
    L, ctx = _ctx(r, benchmark="tau3")
    with pytest.raises(L.Halt, match="only the SSB v2 surface"):
        L._evidence_layout(ctx)
    L, ctx = _ctx(r, layout="digests")
    assert L._evidence_layout(ctx) == "digests"


def test_loop_rollouts_layout_checks_and_uptake(tmp_path):
    """Epoch 1 surface (base leg) in the loop's own naming: the completeness step re-derives it; a missing
    INDEX is reported with the build command; #234 reads uptake from a next-epoch INDEX."""
    r = make_run(tmp_path, library=False)
    L, ctx = _ctx(r, epoch=1)
    assert r.out.name == "digests_e1_train"
    assert L._layout_missing(ctx, r.out, 1) and "digest step builds it" in L._layout_missing(ctx, r.out, 1)[0]
    L._ensure_rollout_evidence(ctx, r.out, None)
    assert (r.out / "INDEX.json").is_file() and L._layout_missing(ctx, r.out, 1) == []
    assert L._rollout_evidence_problems(ctx, [r.out]) == []
    (r.out / "rollouts/Debugging_03_08/s0.grade.txt").write_text("PASS — 1/1 graded cells match\n")
    probs = L._rollout_evidence_problems(ctx, [r.out])
    assert probs and "REMEDY" in probs[0] and "rollout_evidence.py build" in probs[0]
    # #234 from the traces: v1's rollouts are digests_e2_train, built by skills_v1
    r2 = make_run(tmp_path / "two")
    build(r2)
    L, ctx2 = _ctx(r2, epoch=1)
    ok, detail = L._v_skills_were_consulted(ctx2)
    assert ok and "6/6 rollouts loaded a skill = 100% [rollout traces" in detail, detail


def test_loop_train_only_single_dir_scans_text(tmp_path):
    r = make_run(tmp_path, library=False)
    L, ctx = _ctx(r, epoch=1, layout="digests")
    for t in TRAIN:
        (r.out / f"{t.replace('/', '_')}.md").write_text("# digest\n")
    ok, detail = L._v_digests_train_only(ctx)
    assert ok, detail
    (r.out / "Debugging_03_08.md").write_text("the solver wondered about Debugging_05_01\n")
    ok, detail = L._v_digests_train_only(ctx)
    assert not ok and "NON-TRAIN TASK IDS" in detail


def test_upload_manifest_build_and_check_carry_the_rollout_evidence(tmp_path, monkeypatch):
    """build_upload_manifest refuses (before any machine) a rollout subtree its INDEX.json does not describe;
    check_upload_manifest passes a faithful upload and names the traces it verified."""
    from tests.test_evolver_daytona import _check, _fake, _manifest_setup
    f = _fake(tmp_path, monkeypatch)
    rows = []
    for stem in ("Template_01_01", "Template_01_02"):
        d = f.ev / "rollouts" / stem
        d.mkdir(parents=True)
        (d / "s0.trace.jsonl").write_text('{"type": "header"}\n')
        (d / "s0.grade.txt").write_text(PASS_TXT)
        rows.append({"task": stem.replace("_", "/", 1), "sample": 0, "trace": f"rollouts/{stem}/s0.trace.jsonl",
                     "trace_sha256": hashlib.sha256(b'{"type": "header"}\n').hexdigest(),
                     "grade": f"rollouts/{stem}/s0.grade.txt",
                     "grade_sha256": hashlib.sha256(PASS_TXT.encode()).hexdigest()})
    (f.ev / "INDEX.json").write_text(json.dumps({"schema": RE.SCHEMA, "rows": rows, "solver_context": []}))
    (f.ev / "INDEX.md").write_text("# Rollout evidence\n")
    receipt, listing = _manifest_setup(f)
    c = _check(f, receipt, listing)
    assert c.status == "PASS" and "2 rollout trace(s) + grades exactly as their INDEX.json lists them" in c.detail, c.detail
    # a receipt whose rows lost one trace
    bad = dict(receipt, upload_manifest=[r for r in receipt["upload_manifest"]
                                         if not r["vm_path"].endswith("Template_01_02/s0.trace.jsonl")])
    c = _check(f, bad, listing)
    assert c.status == "FAIL" and any("not evidence rows at the sha" in p for p in c.evidence["problems"]), c.detail
    # the trace changes after INDEX.json was written: refused at build time
    (f.ev / "rollouts/Template_01_01/s0.trace.jsonl").write_text('{"type": "header", "x": 1}\n')
    with pytest.raises(ValueError, match="does not match its INDEX.json"):
        _manifest_setup(f)
