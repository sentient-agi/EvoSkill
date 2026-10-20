"""cc_leg_lib (scripts/cc_solver/): the pure half of the Claude Code leg driver. NO network, NO Daytona, NO key.

What is pinned here is the CONTRACT with the loop's readers: the record stem / question hash are RunCache's, the
stream -> messages conversion is the reference conversion's byte for byte, the frozen plug-in is read only after
plugin_runner.verify_frozen, ONE score chain (STATE.json score -> both record copies' inline_score -> the leg's
per_task, a grader_failed rollout null in the last two) that LegLayout.verify cross-checks, the leg JSON carries every
key loop.py / evolver_template read, skill rows + listing + init expectations, the SKILL.md policy, the rate-limit
wait plan and STATE.json /2.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cc_solver"))
import cc_solver_lib as L  # noqa: E402
import cc_leg_lib as G  # noqa: E402

OC = G.outcomes_mod()
RE = G.rollout_evidence_mod()
SL = G.skill_library_mod()

MODEL = "claude-sonnet-5"
TRAIN = ["Debugging/03_08", "Financial_Model/06_01"]
VAL = ["Debugging/05_01"]
N = 2
# real Claude Code session streams, when a cc_solver out root's runs/ dir is named (each run dir: stream.jsonl,
# RESULT.json, user_message.txt); the tests that need them skip otherwise
_STREAMS_DIR = os.environ.get("EVOSKILL_TEST_CC_RUNS_DIR")
REAL_STREAMS = sorted(Path(_STREAMS_DIR).glob("*/stream.jsonl")) if _STREAMS_DIR else []
GF = G.GRADER_FAILED


# ═══════════════════════════════════════════════════════════════ fixtures: streams, split, run dir, a graded leg
def cc_stream(*, capped: bool = False, skill: str | None = None, skills_listed=(), text: str = "Done.") -> str:
    init = {"type": "system", "subtype": "init", "cwd": "/evo/work", "session_id": "S1", "uuid": "U1",
            "model": MODEL, "tools": ["Bash", "Edit", "Read", "Skill", "Write"], "claude_code_version": "2.1.281",
            "permissionMode": "bypassPermissions", "apiKeySource": "none", "mcp_servers": [], "skills": list(skills_listed),
            "plugins": [{"name": "agents-md", "path": "builtin"}], "slash_commands": [], "memory_paths": {}}
    ev = [init,
          {"type": "assistant", "message": {"id": "m1", "model": MODEL, "content": [
              {"type": "thinking", "thinking": "look at the workbook first", "signature": "sig"}]}},
          {"type": "system", "subtype": "thinking_tokens", "estimated_tokens": 5},
          {"type": "assistant", "message": {"id": "m1", "model": MODEL, "content": [
              {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls spreadsheets"}}]}},
          {"type": "rate_limit_event", "rate_limit_info": {"status": "allowed", "unifiedWindows": {
              "five_hour": {"utilization": 0.2, "resetsAt": 1790495400}, "seven_day": {"utilization": 0.3, "resetsAt": 1790618400}}}},
          {"type": "user", "message": {"role": "user", "content": [
              {"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "in.xlsx"}], "is_error": False}]}}]
    if skill:
        ev += [{"type": "assistant", "message": {"id": "m2", "model": MODEL, "content": [
                    {"type": "tool_use", "id": "t2", "name": "Skill", "input": {"skill": skill}}]}},
               {"type": "user", "message": {"role": "user", "content": [
                   {"type": "tool_result", "tool_use_id": "t2", "content": f"Launching skill: {skill}", "is_error": False}]}},
               {"type": "user", "message": {"role": "user", "content": [
                   {"type": "text", "text": f"Base directory for this skill: /evo/work/.claude/skills/{skill}\nbody"}]}}]
    ev += [{"type": "assistant", "message": {"id": "m3", "model": MODEL, "content": [{"type": "text", "text": text}],
                                             "stop_reason": "end_turn"}},
           {"type": "result", "subtype": "error_max_turns" if capped else "success", "is_error": capped,
            "num_turns": 46 if capped else 3, "terminal_reason": "max_turns" if capped else "completed",
            "result": text, "total_cost_usd": 0.5, "duration_ms": 9000, "usage": {"output_tokens": 3}}]
    return "\n".join(json.dumps(e) for e in ev) + "\n"


def make_split(tmp: Path) -> tuple:
    p = tmp / "split.json"
    p.write_text(json.dumps({"train": TRAIN, "val": VAL, "test": [], "unused": [],
                             "dropped": {"task_ids": ["Template/01_01"]}}))
    return p, hashlib.sha256(p.read_bytes()).hexdigest()


def make_run(tmp: Path, name: str = "fake_cc_run") -> tuple:
    split, sha = make_split(tmp)
    rd = tmp / name
    rd.mkdir()
    (rd / "SPEC.json").write_text(json.dumps({"spec": {"name": name, "solver_model": MODEL, "n_samples": N,
                                                       "split_path": str(split), "split_sha256": sha, "agent_turns": 45,
                                                       "solver_harness": "claude_code"}}))
    (rd / "INFRA.json").write_text(json.dumps({"leg_sandbox": {
        "dockerfile": "scripts/daytona/Dockerfile.ssb_leg", "dockerfile_sha256": "a3" * 32,
        "snapshot_ref": "daytona-88:daytona", "image_id": "x86daytona-681a617ef3ce", "grading_openpyxl": "3.1.5",
        "libreoffice": "LibreOffice 25.2.3.2 520(Build:2)"}, "data": {}}))
    (rd / "RUNTIME.json").write_text(json.dumps({"evolver_evidence_layout": "rollouts"}))
    return rd, split, sha


def question_of(task: str) -> str:
    return json.dumps({"task_id": task, "category": task.split("/")[0], "instruction": "Fill the model.",
                       "spreadsheet_path": f"/data/{task}/in.xlsx", "spreadsheet_filename": "in.xlsx"})


def make_layout(tmp: Path, rd: Path, *, key="train", epoch=0, skills_dir=None, questions=True) -> G.LegLayout:
    tag = G.leg_tag(rd.name, epoch, key)
    return G.LegLayout(rd, tag, key, epoch, "0123456789ab", skills_dir, tmp / "store" / rd.name / tag,
                       tmp / f"{rd.name}_e{epoch}_cache", tmp / f"ssbv2_{tag}_tasks",
                       questions={t: question_of(t) for t in TRAIN + VAL} if questions else None)


def lay_out(tmp: Path, rd: Path, *, scores: dict, streams: dict | None = None, skills_dir=None, key="train",
            epoch=0) -> tuple:
    """Every (task, k) of the layout's tasks x N as the driver leaves it at grade time: the record through
    LegLayout.write_sample and the unit's STATE.json row (graded, the score). scores[(task, k)] = a number from 0 to
    1 | GF. Returns (layout, units_results) in the driver's memory form (score None for GF)."""
    lay = make_layout(tmp, rd, key=key, epoch=epoch, skills_dir=skills_dir)
    units = [f"{t}#s{k + 1}" for t in lay.task_ids for k in range(N)]
    st = G.state_new(tag=lay.tag, key=key, epoch=epoch, run_dir=rd, bucket=lay.bucket, skills_dir=skills_dir,
                     stamp="20260930T040000", units=units)
    t0 = dt.datetime(2026, 9, 30, 4, 0, 0)
    ur = {}
    for i, task in enumerate(lay.task_ids):
        for k in range(N):
            s = scores[(task, k)]
            unit = f"{task}#s{k + 1}"
            stream = (streams or {}).get((task, k)) or cc_stream(capped=(s == 0.0 and k == 1))
            res = L.parse_stream(stream)["result"] or {}
            trace = G.build_trace(stream_text=stream, user_message=f"Task {task}",
                                  result={"subtype": res.get("subtype"), "num_turns": res.get("num_turns"),
                                          "cost": res.get("total_cost_usd"), "duration_ms": res.get("duration_ms"),
                                          "is_error": res.get("is_error"), "result_text_head": "Done.",
                                          "init": {"tools": ["Bash", "Read"], "session_id": "S1"}},
                                  score=None if s == GF else s, output_saved=True, model=MODEL, cc={"unit": unit})
            lay.write_sample(unit, trace, t0 + dt.timedelta(minutes=10 * i + k), question=question_of(task))
            G.state_set_unit(st, unit, state="graded", score=s, snapshot_sha256="ab" * 32, result=True,
                             grade_job_id=f"{L.safe_id(task)}__s{k}", graded_utc="2026-09-30T04:30:00Z")
            ur[unit] = {"score": trace["inline_score"], "turn_exhausted": bool(trace["budget_exhausted"])}
    G.state_write(lay.state_path, st)
    return lay, ur


def make_skills(tmp: Path, names=("alpha", "beta")) -> Path:
    d = tmp / "skills_v1"
    for n in names:
        (d / n).mkdir(parents=True)
        (d / n / "SKILL.md").write_text(f"---\nname: {n}\ndescription: Use {n} when the sheet needs it.\n---\n"
                                        f"# {n}\n\nRun `python3 helper.py`.\n")
    (d / names[0] / "helper.py").write_text("print('hi')\n")
    return d


def runtime_env_ok(**kw) -> dict:
    return G.runtime_env_record(solver_stack="3.1.3 2.2.0 1.26.4", libreoffice="LibreOffice 25.2.3.2 520(Build:2)",
                                split_sha256="ea24", **kw)


def solver_harness_ok(**kw) -> dict:
    return G.solver_harness_record([{"init": {"tools": ["Bash", "Read"]}, "argv_flags": ["-p"], "env": {"A": "1"}},
                                    {"init": {"tools": ["Bash", "Skill"]}, "env": {"B": "2"}}], model=MODEL, **kw)


# ═══════════════════════════════════════════════════════════════ the record: stem, hash, messages, budget
def test_record_stem_and_question_hash_are_run_caches():
    q = question_of(TRAIN[0])
    for k in range(3):
        stem = G.record_stem(q, MODEL, k)
        assert stem == OC._record_path_stem(q, "direct", MODEL, "", k)
        # build_rollout_digests.true_sample_idx's formula (sdk=direct, EVOSKILL_SOLVER_MODEL, sample=i), verbatim
        keyed = f"{q.strip()}|sdk=direct|model={MODEL}|sample={k}"
        assert hashlib.sha256(keyed.encode("utf-8")).hexdigest()[:12] == stem
    # run_cache.py:863-870: question_hash over the keyed string WITHOUT the sample, hash_length 12
    assert G.question_hash(q, MODEL) == hashlib.sha256(f"{q.strip()}|sdk=direct|model={MODEL}".encode()).hexdigest()[:12]
    rec = G.build_record(question=q, model=MODEL, bucket="0123456789ab", trace={"inline_score": 1.0},
                         created_at="2026-09-27T04:46:57.238Z")
    assert rec["cache_key"] == {"tree_hash": "0123456789ab", "question_hash": G.question_hash(q, MODEL), "question": q,
                                "sdk": "direct", "model": MODEL, "effort": ""}
    assert rec["created_at"] == "2026-09-27T04:46:57.238000" and rec["version"] == "1.0"
    t = 1790485407                                     # a unit's t_end
    assert G.naive_utc(t) == dt.datetime.fromtimestamp(t, dt.timezone.utc).replace(tzinfo=None).isoformat()
    assert G.epoch_of(G.naive_utc(t)) == float(t)      # outcomes._created_epochs' UTC reading of a naive created_at
    assert G.naive_utc(dt.datetime(2026, 9, 27, 1, 3, 27, tzinfo=dt.timezone.utc)) == "2026-09-27T01:03:27"
    assert G.naive_utc("2026-09-27T01:03:27+02:00") == "2026-09-26T23:03:27"


# the reference stream -> messages conversion, an independent copy of the converter's first version, without a system
# row (the solver has no recorded system prompt: L.PROMPT_DELIVERY)
def _glue_to_messages(stream_text, user_msg):
    msgs = [{"role": "user", "content": user_msg}]
    cur_id = None
    for ln in stream_text.split("\n"):
        if not ln.strip():
            continue
        e = json.loads(ln)
        t = e.get("type")
        m = e.get("message") if isinstance(e.get("message"), dict) else None
        if t == "assistant" and m:
            if m.get("id") != cur_id:
                cur_id = m.get("id")
                msgs.append({"role": "assistant", "content": ""})
            a = msgs[-1]
            for b in m.get("content") or []:
                if b.get("type") == "text":
                    a["content"] = (a["content"] + b.get("text", "")) if a["content"] else b.get("text", "")
                elif b.get("type") == "thinking":
                    a["reasoning"] = a.get("reasoning", "") + (b.get("thinking") or "")
                elif b.get("type") == "tool_use":
                    a.setdefault("tool_calls", []).append({
                        "id": b["id"], "type": "function",
                        "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {}, ensure_ascii=False)}})
        elif t == "user" and m and isinstance(m.get("content"), list):
            cur_id = None
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    c = b.get("content")
                    if isinstance(c, list):
                        c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict) and x.get("type") == "text")
                    msgs.append({"role": "tool", "tool_call_id": b.get("tool_use_id"), "content": c or ""})
    for a in msgs:
        if a["role"] == "assistant" and not a.get("content") and a.get("tool_calls"):
            a["content"] = None
    return msgs


def _glue_model_replies(stream_text):
    ids = set()
    for ln in stream_text.split("\n"):
        if not ln.strip():
            continue
        e = json.loads(ln)
        if e.get("type") == "assistant" and isinstance(e.get("message"), dict):
            ids.add(e["message"].get("id"))
    return len(ids)


def test_to_messages_and_model_replies_are_glues_byte_for_byte():
    streams = [cc_stream(), cc_stream(capped=True, skill="alpha")]
    if REAL_STREAMS:
        streams += [p.read_text(errors="replace") for p in REAL_STREAMS[:3]]
    for s in streams:
        # the ONE intended divergence from the reference -- text blocks the CLI injects into a user event
        # (a loaded skill's body) are kept as from_harness user messages; the reference drops them. Everything else
        # stays byte-for-byte.
        ours = [x for x in G.to_messages(s, "U") if not x.get("from_harness")]
        a = json.dumps(ours, ensure_ascii=False, sort_keys=True)
        b = json.dumps(_glue_to_messages(s, "U"), ensure_ascii=False, sort_keys=True)
        assert a == b
        assert G.model_replies(s) == _glue_model_replies(s)
    m = G.to_messages(cc_stream(skill="alpha"), "U")
    assert [x["role"] for x in m] == ["user", "assistant", "tool", "assistant", "tool", "user", "assistant"]
    assert m[0] == {"role": "user", "content": "U"}           # no system row: the instruction IS the user message
    assert m[1]["content"] is None and m[1]["reasoning"] == "look at the workbook first"
    assert m[1]["tool_calls"][0]["function"] == {"name": "Bash", "arguments": '{"command": "ls spreadsheets"}'}
    assert m[2] == {"role": "tool", "tool_call_id": "t1", "content": "in.xlsx"}
    assert m[4]["content"] == "Launching skill: alpha"
    assert m[5] == {"role": "user", "content": "Base directory for this skill: /evo/work/.claude/skills/alpha\nbody",
                    "from_harness": True}                     # the skill's body, as the CLI injected it
    assert m[6]["content"] == "Done."
    assert G.model_replies(cc_stream(skill="alpha")) == 3


def test_budget_exhausted_iff_error_max_turns_and_direct_trace_reads_it():
    assert G.budget_exhausted_text("error_max_turns") == "Turn budget exhausted (45/45 turns)"
    assert G.budget_exhausted_text("success") == "" and G.budget_exhausted_text(None, 30) == ""
    for capped in (False, True):
        s = cc_stream(capped=capped)
        res = L.parse_stream(s)["result"]
        tr = G.build_trace(stream_text=s, user_message="U", score=0.0, output_saved=True,
                           result={"subtype": res["subtype"], "num_turns": res["num_turns"], "init": {"tools": ["Bash"]}})
        rec = G.build_record(question=question_of(TRAIN[0]), model=MODEL, bucket="0123456789ab", trace=tr,
                             created_at=time.time())
        d = RE.direct_trace(rec)
        assert d["solver"] == "direct" and d["end"]["turn_cap_hit"] is capped
        assert d["end"]["termination"] == ("turn_cap" if capped else "completed")
        assert d["end"]["turns"] == tr["num_turns"] == 2 and tr["num_turns_cc_reported"] == (46 if capped else 3)
        assert d["system"] is None and tr["output"] == {"finished": True} and tr["stream_jsonl"] == s
        assert RE.trace_for_record(rec)["solver"] == "direct"      # sdk "direct" routes to direct_trace


def test_skill_loads_read_the_skill_tool_call_from_a_converted_stream():
    def lines(stream):
        tr = G.build_trace(stream_text=stream, user_message="U", score=1.0, output_saved=True,
                           result={"subtype": "success", "num_turns": 3, "init": {}})
        rec = G.build_record(question=question_of(TRAIN[0]), model=MODEL, bucket="0123456789ab", trace=tr,
                             created_at=time.time())
        return RE.direct_trace(rec)["lines"]
    sk = RE.skill_loads(lines(cc_stream(skill="alpha")), ["alpha", "beta"])
    assert sk["loaded"] == ["alpha"] and sk["not_loaded"] == ["beta"] and sk["skill_tool_calls"] == 1
    base = RE.skill_loads(lines(cc_stream()), [])
    assert base["loaded"] == [] and base["skill_tool_calls"] == 0 and not base["skill_tool_ok_outside_library"]


def test_cc_bucket_is_12_hex_stable_and_distinguishes_inputs():
    a = G.cc_bucket("2.1.281", MODEL, "p1", "s1")
    assert len(a) == 12 and int(a, 16) >= 0 and a == G.cc_bucket("2.1.281", MODEL, "p1", "s1")
    assert len({a, G.cc_bucket("2.1.282", MODEL, "p1", "s1"), G.cc_bucket("2.1.281", MODEL, "p2", "s1"),
                G.cc_bucket("2.1.281", MODEL, "p1", "s2"), G.cc_bucket("2.1.281", "claude-opus-5-5", "p1", "s1")}) == 5
    assert G.skills_tree_sha256(None) == hashlib.sha256(b"").hexdigest()
    assert G.bucket_for(MODEL, "prompt") == G.cc_bucket("2.1.281", MODEL, hashlib.sha256(b"prompt").hexdigest(),
                                                       hashlib.sha256(b"").hexdigest())


def test_skills_tree_sha_follows_the_files(tmp_path):
    d = make_skills(tmp_path)
    a = G.skills_tree_sha256(d)
    (d / "alpha" / "helper.py").write_text("print('changed')\n")
    assert G.skills_tree_sha256(d) != a and G.bucket_for(MODEL, "p", d) != G.bucket_for(MODEL, "p", None)


# ═══════════════════════════════════════════════════════════════ the leg JSON
def test_assemble_leg_keys_order_and_counts(tmp_path):
    ur = {"Debugging/03_08#s1": {"score": 1.0, "turn_exhausted": False},
          "Debugging/03_08#s2": {"score": 0.0, "turn_exhausted": True},
          "Financial_Model/06_01#s1": {"score": 1.0}, "Financial_Model/06_01#s2": {"score": 1.0}}
    leg = G.assemble_leg(tag="r_v1_train", key="train", epoch=1, model=MODEL, budget=45, n_samples=N, task_ids=TRAIN,
                         units_results=ur, skills_dir=tmp_path / "skills_v1", runtime_env=runtime_env_ok(),
                         solver_harness=solver_harness_ok(), daytona={"store": "/s"})
    assert leg["per_task"] == {"0": [1.0, 0.0], "1": [1.0, 1.0]} and leg["task_ids"] == TRAIN
    assert (leg["sample_pass"], leg["sample_total"], leg["pass3"], leg["all3"], leg["turn_exhausted"], leg["n"]) == \
        (3, 4, 2, 1, 1, 2)
    assert leg["episodes_excluded_infra"] == 0 and leg["episodes_timeout_scored_zero"] == 0
    assert leg["skills_dir"] == str(tmp_path / "skills_v1") and leg["origin"] == {"cc_solver": 4}
    for k in ("tag", "key", "epoch", "model", "budget", "n_samples", "written_utc", "runtime_env", "solver_harness",
              "daytona"):
        assert k in leg
    assert leg["solver_harness"]["tools"] == ["Bash", "Read", "Skill"] and leg["solver_harness"]["kind"] == "claude_code"
    assert leg["solver_harness"]["env_keys"] == ["A", "B"] and "skills_delivery" not in leg["solver_harness"]
    assert leg["runtime_env"]["solver_stack"].split().__len__() == 3
    assert leg["grader_failed"] == []                               # a clean leg still carries the key
    # Raw scores; a pass is full marks (>= 0.999), a fraction is a fail in the counts, and
    # a grader_failed rollout (None) is null in per_task, listed in grader_failed and fails all3 -- never a number
    ur3 = {"Debugging/03_08#s1": {"score": 1.0}, "Debugging/03_08#s2": {"score": 0.5},
           "Debugging/03_08#s3": {"score": None, "turn_exhausted": True},
           "Financial_Model/06_01#s1": {"score": 1}, "Financial_Model/06_01#s2": {"score": 0.9995},
           "Financial_Model/06_01#s3": {"score": 1.0}}
    leg3 = G.assemble_leg(tag="r_v1_train", key="train", epoch=1, model=MODEL, budget=45, n_samples=3, task_ids=TRAIN,
                          units_results=ur3, skills_dir=None, runtime_env=runtime_env_ok(),
                          solver_harness=solver_harness_ok(), daytona={})
    assert leg3["per_task"] == {"0": [1.0, 0.5, None], "1": [1.0, 0.9995, 1.0]}
    assert isinstance(leg3["per_task"]["1"][0], float)
    assert (leg3["sample_pass"], leg3["sample_total"], leg3["pass3"], leg3["all3"], leg3["turn_exhausted"]) == \
        (4, 6, 2, 1, 1)
    assert leg3["grader_failed"] == ["Debugging/03_08#s3"] and leg3["episodes_excluded_infra"] == 0
    assert json.loads(json.dumps(leg3))["per_task"]["0"][2] is None          # null in the leg JSON
    assert G.leg_counts(leg3["per_task"], TRAIN) == {"sample_pass": 4, "sample_total": 6, "pass3": 2, "all3": 1,
                                                     "grader_failed": ["Debugging/03_08#s3"]}
    with pytest.raises(ValueError, match="without a result"):
        G.assemble_leg(tag="t", key="train", epoch=1, model=MODEL, budget=45, n_samples=N, task_ids=TRAIN,
                       units_results=dict(list(ur.items())[:3]), skills_dir=None, runtime_env=runtime_env_ok(),
                       solver_harness=solver_harness_ok(), daytona={})
    with pytest.raises(ValueError, match="not a task"):
        G.assemble_leg(tag="t", key="train", epoch=1, model=MODEL, budget=45, n_samples=N, task_ids=TRAIN,
                       units_results={**ur, "Debugging/05_01#s1": {"score": 1.0}}, skills_dir=None,
                       runtime_env=runtime_env_ok(), solver_harness=solver_harness_ok(), daytona={})
    with pytest.raises(ValueError, match="solver_stack"):
        G.runtime_env_record(solver_stack="3.1.3", libreoffice="LO")
    with pytest.raises(ValueError, match="unknown keys"):
        G.runtime_env_record(solver_stack="1 2 3", libreoffice="LO", bogus=1)


def test_solver_harness_delivery_text_and_facts(tmp_path):
    d = make_skills(tmp_path)
    facts = G.skills_delivery_facts(d, init_skills_seen={"alpha": 4, "beta": 4}, sessions=4, probe="/p.json")
    text = G.skills_delivery_text("skills_v1", facts["tree_sha256"][:12])
    h = solver_harness_ok(skills_delivery=text, skills_delivery_facts=facts)
    assert "Base directory for this skill: /evo/work/.claude/skills/<name>" in h["skills_delivery"]
    assert facts["tree_sha256"][:12] in h["skills_delivery"] and "`skills_v1`" in h["skills_delivery"]
    assert facts["names"] == ["alpha", "beta"] and facts["library_sha256"] == SL.library_sha256(d)
    assert facts["how"] == "claude_code_project_skills_symlink" and facts["path"] == "/evo/work/.claude/skills"
    assert h["skills_delivery_facts"]["init_skills_seen"] == {"alpha": 4, "beta": 4}
    # evolver_template appends the string when the leg records it (the template reads solver_harness by key)
    sys.path.insert(0, str(ROOT))
    from src.orchestrator import evolver_template as ET
    leg = tmp_path / "leg.json"
    leg.write_text(json.dumps({"solver_harness": h}))
    txt = ET.solver_observe_text(leg, budget_rule=False)
    assert "Claude Code agent" in txt and "Base directory for this skill" in txt and "`Bash`" in txt


# ═══════════════════════════════════════════════════════════════ the record writer, the leg writer, verify
def test_write_sample_writes_only_the_two_records_and_the_kept_check_reads_the_state_score(tmp_path):
    rd, split, sha = make_run(tmp_path)
    scores = {(TRAIN[0], 0): 1.0, (TRAIN[0], 1): 0.25, (TRAIN[1], 0): GF, (TRAIN[1], 1): 0.0}
    lay, ur = lay_out(tmp_path, rd, scores=scores)
    w = lay.written["Debugging/03_08#s2"]
    assert w["k"] == 1 and w["stem"] == G.record_stem(question_of(TRAIN[0]), MODEL, 1) and w["inline_score"] == 0.25
    assert set(w) == {"unit", "task", "k", "stem", "records", "record_sha256", "created_at", "inline_score"}
    a, b = (Path(p) for p in w["records"])
    assert a.read_bytes() == b.read_bytes() and a.parent.name == b.parent.name == "0123456789ab"
    assert a.is_relative_to(lay.attempt / "cache") and b.is_relative_to(lay.cache_dir)
    assert hashlib.sha256(a.read_bytes()).hexdigest() == w["record_sha256"]
    # The records are ALL a sample writes -- no snapshot dirs, DIFF.txt, LAYOUT.md,
    # runner.log or tasks dir (the graded snapshot is the unit's runs/<ud>/container_out.tgz)
    recs = {Path(p) for x in lay.written.values() for p in x["records"]}
    assert len(recs) == 2 * len(TRAIN) * N
    assert {p for p in lay.store.rglob("*") if p.is_file()} == {r for r in recs if r.is_relative_to(lay.store)} | {
        lay.state_path}
    assert {p for p in lay.cache_dir.rglob("*") if p.is_file()} == {r for r in recs if r.is_relative_to(lay.cache_dir)}
    assert not lay.tasks_dir.exists() and not list(tmp_path.rglob("DIFF.txt"))
    # a grader_failed rollout's record says null, never a number
    gf = lay.written[f"{TRAIN[1]}#s1"]
    assert gf["inline_score"] is None
    assert json.loads(Path(gf["records"][0]).read_text())["trace"]["inline_score"] is None
    # the kept-sample check (resume): both copies, identical, inline_score == the STATE score (null <-> grader_failed)
    q0, q1 = question_of(TRAIN[0]), question_of(TRAIN[1])
    assert lay.kept_sample_problems("Debugging/03_08#s2", q0, score=0.25) == []
    assert lay.kept_sample_problems(f"{TRAIN[1]}#s1", q1, score=GF) == []
    assert lay.kept_sample_problems("Debugging/03_08#s2", q0, score=1.0) == [
        "Debugging/03_08#s2: record inline_score 0.25 != STATE score 1.0"]
    assert lay.kept_sample_problems(f"{TRAIN[1]}#s1", q1, score=0.0)          # a grader failure is never a 0
    assert lay.kept_sample_problems(f"{TRAIN[1]}#s2", q1, score=GF)           # nor a 0 a grader failure
    assert lay.kept_sample_problems("Debugging/03_08#s2", q0, score=None)     # not graded
    b.write_bytes(b.read_bytes() + b" ")
    assert lay.kept_sample_problems("Debugging/03_08#s2", q0, score=0.25) == ["Debugging/03_08#s2: the two record "
                                                                              "copies differ"]
    os.remove(a)
    assert lay.kept_sample_problems("Debugging/03_08#s2", q0, score=0.25) == ["Debugging/03_08#s2: record missing"]
    with pytest.raises(ValueError, match="names"):
        lay.write_sample("Debugging/03_08#s1", {"inline_score": 0.0}, time.time(), question=question_of(TRAIN[1]))


def test_write_leg_verifies_then_writes_once_and_fails_loudly(tmp_path):
    rd, split, sha = make_run(tmp_path)
    scores = {(TRAIN[0], 0): 1.0, (TRAIN[0], 1): 0.0, (TRAIN[1], 0): GF, (TRAIN[1], 1): 1.0}
    lay, ur = lay_out(tmp_path, rd, scores=scores)
    p = lay.write_leg(ur, runtime_env_ok(), solver_harness_ok(), {"driver": "test"}, ["a unit died"])
    assert p == lay.failed_path and json.loads(p.read_text())["INCOMPLETE"] is True and not lay.leg_path.exists()
    assert json.loads(p.read_text())["problems"] == ["a unit died"]
    # a leg whose recorded score is not STATE's is refused by verify
    bad = dict(ur)
    bad["Debugging/03_08#s2"] = {"score": 1.0}
    p = lay.write_leg(bad, runtime_env_ok(), solver_harness_ok(), {}, [])
    assert p == lay.failed_path and json.loads(p.read_text())["problems"][0] == \
        "Debugging/03_08#s2: per_task 1.0 != STATE score 0.0"
    # a grader_failed rollout does NOT fail the leg: it is written, null in per_task, listed in grader_failed
    p = lay.write_leg(ur, runtime_env_ok(), solver_harness_ok(), {}, [])
    leg = json.loads(p.read_text())
    assert p == lay.leg_path and leg["per_task"] == {"0": [1.0, 0.0], "1": [None, 1.0]}
    assert (leg["sample_pass"], leg["pass3"], leg["all3"], leg["grader_failed"]) == (2, 2, 0, [f"{TRAIN[1]}#s1"])
    assert leg["daytona"]["store"] == str(lay.store)
    with pytest.raises(FileExistsError):
        lay.write_leg(ur, runtime_env_ok(), solver_harness_ok(), {}, [])


def test_verify_cross_checks_state_the_two_records_and_per_task(tmp_path):
    rd, split, sha = make_run(tmp_path)
    scores = {(TRAIN[0], 0): 1.0, (TRAIN[0], 1): 0.5, (TRAIN[1], 0): GF, (TRAIN[1], 1): 0.0}
    lay, ur = lay_out(tmp_path, rd, scores=scores)
    leg = lay.assemble(ur, runtime_env=runtime_env_ok(), solver_harness=solver_harness_ok(), daytona={})
    assert lay.verify(leg) == []
    u05, ugf = f"{TRAIN[0]}#s2", f"{TRAIN[1]}#s1"
    # per_task against STATE: a grader failure is never a number
    bad = copy.deepcopy(leg)
    bad["per_task"]["1"][0] = 0.0
    assert f"{ugf}: per_task 0.0 != STATE score 'grader_failed'" in lay.verify(bad)
    # the counts re-derive from per_task
    for name, v in (("sample_pass", 2), ("pass3", 2), ("all3", 1), ("grader_failed", [])):
        bad = copy.deepcopy(leg)
        bad[name] = v
        probs = lay.verify(bad)
        assert len(probs) == 1 and probs[0].startswith("counts do not re-derive") and name in probs[0]
    bad = copy.deepcopy(leg)
    bad["per_task"]["0"] = [1.0]
    assert any("want 2 entries" in p for p in lay.verify(bad))
    # STATE against the records and per_task
    st = G.state_read(lay.state_path)
    st["units"][u05]["score"] = 0.75
    G.state_write(lay.state_path, st)
    probs = lay.verify(leg)
    assert f"{u05}: per_task 0.5 != STATE score 0.75" in probs
    assert f"{u05}: record inline_score 0.5 != STATE score 0.75" in probs
    for field, v, want in (("score", 1.5, "is not a number from 0 to 1"), ("score", True, "is not a number"),
                           ("score", "passed", "is not a number"), ("score", None, "is not a number"),
                           ("state", "collected", "not graded")):
        st = G.state_read(lay.state_path)
        st["units"][u05].update({"state": "graded", "score": 0.5, field: v})
        G.state_write(lay.state_path, st)
        assert any(p.startswith(u05) and want in p for p in lay.verify(leg)), (field, v)
    st["units"][u05].update({"state": "graded", "score": 0.5})
    G.state_write(lay.state_path, st)
    assert lay.verify(leg) == []
    # the two record copies
    rp = Path(lay.written[u05]["records"][1])
    keep = rp.read_bytes()
    rp.write_bytes(keep.replace(b'"inline_score": 0.5', b'"inline_score": 1.0'))
    assert lay.verify(leg) == [f"{u05}: the two record copies differ"]
    rp.write_bytes(keep)
    # without the questions no record can be found: named, never passed silently
    nq = make_layout(tmp_path, rd, questions=False)
    assert sum("no question to find its records by" in p for p in nq.verify(leg)) == len(TRAIN)
    # a verdict-era STATE (/1) or none at all
    lay.state_path.write_text(json.dumps({"schema": "evoskill.cc_leg_state/1", "units": {}}))
    assert len(lay.verify(leg)) == 1 and "cc_leg_state/1" in lay.verify(leg)[0]
    lay.state_path.unlink()
    assert lay.verify(leg) == [f"{lay.state_path} missing"]
    # a val leg is verified the same way (no train-only evidence step any more)
    lv, urv = lay_out(tmp_path, rd, scores={(VAL[0], 0): 1.0, (VAL[0], 1): 0.0}, key="val")
    assert lv.task_ids == VAL and lv.tag == "fake_cc_run_base_val"
    legv = lv.assemble(urv, runtime_env=runtime_env_ok(), solver_harness=solver_harness_ok(), daytona={})
    assert lv.verify(legv) == [] and legv["sample_pass"] == 1 and legv["grader_failed"] == []


# ═══════════════════════════════════════════════════════════════ the frozen plug-in; the pinned constants
def _freeze_by_hand(rd: Path, tasks: list) -> dict:
    """setup/ holding every file plugin_runner.SETUP_FILES names and the record plugin_runner.freeze writes over
    them, without running the sandboxed list_tasks child (test_plugin_runner covers freeze itself)."""
    PR = G.plugin_runner_mod()
    setup = rd / "setup"
    setup.mkdir()
    for n in PR.SETUP_FILES:
        if n == "plugin_api.py":
            shutil.copyfile(PR.API_FILE, setup / n)
        else:
            (setup / n).write_text(f"# {n}\n")
    blobs = {PR.TASKS_JSON: PR.canonical_json(tasks), PR.VALIDATION_JSON: PR.canonical_json({"ok": True}),
             PR.SPLIT_JSON: PR.canonical_json(PR.split_from_tasks(tasks))}
    for n, b in blobs.items():
        (setup / n).write_bytes(b)
    pin = {n: {"path": str(setup / n), "sha256": PR.sha256_bytes(b)} for n, b in blobs.items()}
    rec = {"api_version": PR.A.API_VERSION,
           "files": {n: {"path": str(setup / n), "sha256": PR.sha256_file(setup / n)} for n in PR.SETUP_FILES},
           "canonical_plugin_api": {"path": str(PR.API_FILE), "sha256": PR.sha256_file(PR.API_FILE)},
           "validation": pin[PR.VALIDATION_JSON], "tasks_json": pin[PR.TASKS_JSON], "split_json": pin[PR.SPLIT_JSON]}
    rec["record_sha256"] = PR._record_sha(rec)
    (setup / PR.PLUGIN_JSON).write_bytes(PR.canonical_json(rec))
    return rec


def test_frozen_plugin_verifies_every_pin_and_turns_drift_into_value_error(tmp_path, monkeypatch):
    PR = G.plugin_runner_mod()
    rd, split, sha = make_run(tmp_path)
    tasks = [{"id": t, "category": t.split("/")[0], "instruction": "Fill the model.", "inputs": [f"/data/{t}/in.xlsx"],
              "answer_files": [f"/data/{t}/gold.xlsx"], "grading": {"answer_position": "'S'!A1"}, "split": "train"}
             for t in TRAIN]
    rec = _freeze_by_hand(rd, tasks)
    calls = []
    real = PR.verify_frozen
    monkeypatch.setattr(PR, "verify_frozen", lambda run_dir, **kw: calls.append(Path(run_dir)) or real(run_dir, **kw))
    fp = G.frozen_plugin(rd)
    assert calls == [rd]                                             # every pin, through plugin_runner's own check
    # A1: the three pinned setup files, the user's grader among them
    assert fp["files"] == rec["files"] and set(fp["files"]) == {"benchmark_plugin.py", "benchmark_grader.py",
                                                                "plugin_api.py"}
    assert fp["tasks"] == tasks and fp["tasks_json_sha256"] == rec["tasks_json"]["sha256"]
    qs, idx = G.tasks_json_questions(fp["tasks"])                     # unchanged: the questions and the anti-leak idx
    assert json.loads(qs[TRAIN[0]])["spreadsheet_filename"] == "in.xlsx"
    assert idx[TRAIN[1]]["golden"] == Path(f"/data/{TRAIN[1]}/gold.xlsx")
    # the grader file changed after freeze (frozen_plugin itself never reads it): refused, as ValueError
    grader = rd / "setup" / "benchmark_grader.py"
    grader.write_text("def grade(task, snapshot):\n    return 1.0\n")
    with pytest.raises(ValueError, match="benchmark_grader.py.*changed.*re-freeze"):
        G.frozen_plugin(rd)
    grader.write_text("# benchmark_grader.py\n")
    assert G.frozen_plugin(rd)["files"] == rec["files"]
    (rd / "setup" / "tasks.json").write_bytes(PR.canonical_json(tasks[:1]))
    with pytest.raises(ValueError, match="tasks.json"):
        G.frozen_plugin(rd)
    (rd / "setup" / "PLUGIN.json").unlink()
    with pytest.raises(ValueError, match="missing"):
        G.frozen_plugin(rd)


def test_tasks_json_questions_accepts_a_tau3_task_and_keeps_the_ssb_question_bytes():
    """A task with NO input and one answer file (a tau3 task: its tau2 task file) is a conversation; the one-input SSB
    row's question string keeps the SSB key set byte for byte; any other shape is refused."""
    ssb = {"id": TRAIN[0], "category": "Debugging", "instruction": "Fill the model.", "inputs": ["/data/x/in.xlsx"],
           "answer_files": ["/data/x/gold.xlsx"], "grading": {"answer_position": "'S'!A1"}, "split": "train"}
    tau = {"id": "banking_knowledge/task_004", "category": "banking_knowledge", "instruction": "",
           "inputs": [], "answer_files": ["/r/tau2/domains/banking_knowledge/tasks/task_004.json"],
           "grading": {"domain": "banking_knowledge", "retrieval_variant": "bm25_grep"}, "split": "train"}
    qs, idx = G.tasks_json_questions([ssb, tau])
    assert qs[TRAIN[0]] == ('{"task_id": "Debugging/03_08", "category": "Debugging", "instruction": "Fill the model.", '
                            '"spreadsheet_path": "/data/x/in.xlsx", "spreadsheet_filename": "in.xlsx"}')
    assert idx[TRAIN[0]] == {"input": Path("/data/x/in.xlsx"), "golden": Path("/data/x/gold.xlsx"),
                             "answer_position": "'S'!A1", "category": "Debugging", "split": "train"}
    assert json.loads(qs[tau["id"]]) == {"task_id": tau["id"], "category": "banking_knowledge", "instruction": ""}
    assert idx[tau["id"]] == {"input": None, "golden": Path(tau["answer_files"][0]), "answer_position": None,
                              "category": "banking_knowledge", "split": "train"}
    for bad in (dict(ssb, inputs=["/a.xlsx", "/b.xlsx"]), dict(tau, answer_files=[]),
                dict(tau, answer_files=["/a.json", "/b.json"]), dict(ssb, answer_files=[])):
        with pytest.raises(ValueError, match="the leg runs one of each"):
            G.tasks_json_questions([bad])


def test_the_grader_primitive_constants_and_what_is_gone():
    from src.orchestrator import plugin_api as A
    assert G.GRADER_FAILED == A.GRADER_FAILED == "grader_failed"
    assert G.PASS_THRESHOLD == OC.PASS_THRESHOLD
    assert G.STATE_SCHEMA == "evoskill.cc_leg_state/2"
    assert "grader" in G.RUNTIME_ENV_KEYS and "scorer_sha256" in G.RUNTIME_ENV_KEYS
    env = runtime_env_ok(grader={"worker": "plugin_grade/1", "interpreter": "/opt/evoskill/venv/bin/python"})
    assert env["grader"]["worker"] == "plugin_grade/1" and env["scorer_sha256"] is None
    g = G.CC_RESULT_GUARANTEE
    assert "grade(task, snapshot)" in g and "grader_failed" in g and "per_task agree" in g
    assert "outcomes.py" not in g and "2442e5aa" not in g and "no sample is grader_failed" not in g
    for x, ok in ((0.0, True), (1, True), (0.5, True), (1.0000001, False), (-0.1, False), (float("nan"), False),
                  (True, False), (None, False), ("1", False)):
        assert G.is_score(x) is ok, x
    assert G.score_agrees(GF, None) and G.score_agrees(0.5, 0.5) and G.score_agrees(1, 1.0)
    assert not G.score_agrees(GF, 0.0) and not G.score_agrees(0.0, None) and not G.score_agrees(None, None)
    assert G.build_trace(stream_text=cc_stream(), user_message="U", result={}, score=None,
                         output_saved=False)["inline_score"] is None
    assert G.build_trace(stream_text=cc_stream(), user_message="U", result={}, score=1,
                         output_saved=True)["inline_score"] == 1.0
    for gone in ("import_base", "main", "NO_OUTPUT_DIFF", "LAYOUT_MD", "IMPORT_GUARANTEE", "IMPORT_DRIVER",
                 "GRADE_WORKER", "DIFF_MTIME_LEAD_S", "ITER", "dataset_questions", "frozen_tasks", "tasks_output_path",
                 "runner_log_lines", "_resolve_grade", "_unit_input", "_unit_output", "_read_json"):
        assert not hasattr(G, gone), gone
    for gone in ("snapshot_dirs", "append_runner_log", "ensure_runner_log", "evidence_problems"):
        assert not hasattr(G.LegLayout, gone), gone
    assert callable(G.outcomes_mod) and callable(G.rollout_evidence_mod)   # kept: record stems, the skills probe


# ═══════════════════════════════════════════════════════════════ skills: rows, listing, init, policy
def test_skill_rows_and_refusals(tmp_path):
    d = make_skills(tmp_path)
    rows = G.skill_rows(d)
    assert [r["rel"] for r in rows] == ["alpha/SKILL.md", "alpha/helper.py", "beta/SKILL.md"]
    assert all(r["vm_path"] == f"/evo/skills/{r['rel']}" and len(r["sha256"]) == 64 for r in rows)
    assert G.skill_names(rows) == ["alpha", "beta"] and G.check_skill_rows(rows, d) == []
    assert G.extra_dirs_for(rows) == {"/evo/skills", "/evo/work/.claude", "/evo/skills/alpha", "/evo/skills/beta"}
    assert G.expect_links_for(True) == {"/evo/work/.claude/skills": "/evo/skills"} and G.expect_links_for(False) is None
    (d / "alpha" / "helper.py").write_text("changed")
    assert any("differ" in p for p in G.check_skill_rows(rows, d))
    bad = [dict(rows[0], vm_path="/evo/work/x")]
    assert any("not under /evo/skills" in p for p in G.check_skill_rows(bad))
    # Claude Code's implementation is canonical: a reserved or irregular name, a dir without
    # SKILL.md and a stray top-level file all upload; Claude Code decides what loads. Only a symlink is refused.
    (d / "synced").mkdir()
    (d / "synced" / "SKILL.md").write_text("---\nname: synced\ndescription: x\n---\n")
    (d / "Bad Name").mkdir()
    (d / "Bad Name" / "SKILL.md").write_text("---\nname: Bad Name\n---\nno description line\n")
    (d / "gamma").mkdir()
    (d / "gamma" / "notes.txt").write_text("no SKILL.md here")
    (d / "README.md").write_text("a stray top-level file")
    rows = G.skill_rows(d)
    assert {r["rel"] for r in rows} >= {"synced/SKILL.md", "Bad Name/SKILL.md", "gamma/notes.txt", "README.md"}
    assert G.skill_names(rows) == ["Bad Name", "alpha", "beta", "synced"] and G.check_skill_rows(rows, d) == []
    for x in ("synced", "Bad Name", "gamma"):
        shutil.rmtree(d / x)
    os.remove(d / "README.md")
    e = tmp_path / "empty_set"
    e.mkdir()
    assert G.skill_rows(e) == [] and G.skill_names([]) == []                     # an empty set: no skills
    os.symlink(d / "alpha" / "SKILL.md", d / "beta" / "link.md")
    with pytest.raises(ValueError, match="symlink"):
        G.skill_rows(d)


def test_listing_problems_expect_links_and_extra_dirs():
    ws = L.WS_CONTAINER
    rows = [{"vm_path": f"{ws}/spreadsheets/in.xlsx", "sha256": "aa"}]
    base = [[ws, "D", "", 0], [L.OUT_CONTAINER, "D", "", 0], [f"{ws}/output", "D", "", 0],
            [f"{ws}/spreadsheets", "D", "", 0], [f"{ws}/spreadsheets/in.xlsx", "F", "aa", 1]]
    assert L.listing_problems(base, rows) == []
    link = [f"{ws}/.claude/skills", "L", "/evo/skills", 0]
    skills = [[f"{ws}/.claude", "D", "", 0], ["/evo/skills", "D", "", 0], ["/evo/skills/alpha", "D", "", 0],
              ["/evo/skills/alpha/SKILL.md", "F", "bb", 2]]
    srows = [{"vm_path": "/evo/skills/alpha/SKILL.md", "sha256": "bb"}]
    extra = {"/evo/skills", f"{ws}/.claude", "/evo/skills/alpha"}
    links = {f"{ws}/.claude/skills": "/evo/skills"}
    assert L.listing_problems(base + [link] + skills, rows + srows, expect_links=links, extra_dirs=extra) == []
    assert L.listing_problems(base + [link] + skills, rows + srows)                       # default refuses the link
    assert any("missing" in p for p in L.listing_problems(base + skills, rows + srows, expect_links=links, extra_dirs=extra))
    second = [f"{ws}/l2", "L", "/etc", 0]
    assert any("symlinks" in p for p in L.listing_problems(base + [link, second] + skills, rows + srows,
                                                            expect_links=links, extra_dirs=extra))
    wrong = [f"{ws}/.claude/skills", "L", "/tmp", 0]
    assert any("another target" in p for p in L.listing_problems(base + [wrong] + skills, rows + srows,
                                                                  expect_links=links, extra_dirs=extra))
    assert any("files" in p for p in L.listing_problems(base + [link] + skills, rows, expect_links=links, extra_dirs=extra))
    differ = [dict(srows[0], sha256="cc")]
    assert any("files" in p for p in L.listing_problems(base + [link] + skills, rows + differ, expect_links=links,
                                                         extra_dirs=extra))
    assert any("unexpected dirs" in p for p in L.listing_problems(base + [link] + skills, rows + srows,
                                                                   expect_links=links))


def test_init_problems_expect_skills_three_ways():
    init = L.parse_stream(cc_stream(skills_listed=["alpha", "beta"]))["init"]
    assert L.init_problems(init, expect_skills=["beta", "alpha"]) == []
    assert any("skills" in p for p in L.init_problems(init))                          # default: any skill is a problem
    assert any("skills" in p for p in L.init_problems(init, expect_skills=[]))
    assert any("skills" in p for p in L.init_problems(init, expect_skills=["alpha"]))
    empty = L.parse_stream(cc_stream())["init"]
    assert L.init_problems(empty) == [] == L.init_problems(empty, expect_skills=[])
    assert any("skills [] !=" in p for p in L.init_problems(empty, expect_skills=["alpha"]))


def test_the_loop_adds_no_skill_rules_of_its_own(tmp_path):
    """Claude Code's implementation of skills is canonical (the evolver is told to write only name and description):
    no frontmatter-key,
    name == directory, description / listing length, "!`" line or size rule -- none of these is refused."""
    d = make_skills(tmp_path)
    (d / "alpha" / "SKILL.md").write_text("---\nname: other\ndescription: " + "x" * 1600 + "\nmodel: opus\n---\nbody\n!`echo hi`\n")
    for i in range(6):
        (d / f"s{i}").mkdir()
        (d / f"s{i}" / "SKILL.md").write_text(f"---\nname: s{i}\ndescription: {'y' * 1500}\n---\nbody\n")
    rows = G.skill_rows(d)
    assert G.check_skill_rows(rows, d) == [] and len(G.skill_names(rows)) == 8
    assert not hasattr(G, "cc_skill_policy_problems")
    assert G.parse_frontmatter("---\nname: 'q'\ndescription: \"w\"\n---\nbody") == ({"name": "q", "description": "w"}, "body")


# ═══════════════════════════════════════════════════════════════ the usage-limit wait
def _rejected(r5=1000.0, r7=5000.0, kind="five_hour"):
    return {"brief": {"status": "rejected", "rateLimitType": kind, "five_hour": {"utilization": 1.0, "resetsAt": r5},
                      "seven_day": {"utilization": 0.4, "resetsAt": r7}}}


def test_rate_limit_wait_sleeps_until_the_latest_reported_reset():
    now = 100.0
    w = G.rate_limit_wait([_rejected(r5=1000.0)], now, leg_deadline_ts=now + 43200, pending=40, conc=30,
                          median_rollout_s=480.0)
    assert w["until_ts"] == 1000.0 and w["window"] == "five_hour" and w["reason"] is None
    assert w["wait_s"] == 1000.0 + G.PAUSE_MARGIN_S - now and w["beyond_deadline"] is False
    assert w["remaining_estimate_s"] == 1200.0                      # 2 rounds x 480 s < the 20-min floor
    two = G.rate_limit_wait([_rejected(r5=1000.0), _rejected(r5=2000.0)], now, leg_deadline_ts=now + 43200,
                            pending=1, conc=30, median_rollout_s=None)
    assert two["until_ts"] == 2000.0                                # the later of the reported resets
    past = G.rate_limit_wait([_rejected(r5=50.0)], now, leg_deadline_ts=now + 43200, pending=1, conc=30,
                             median_rollout_s=None)
    assert past["until_ts"] == now                                  # a reset already past = now
    bare = G.rate_limit_wait([{"brief": None}], now, leg_deadline_ts=now + 43200, pending=1, conc=30,
                             median_rollout_s=None)
    assert bare["until_ts"] == now + G.PAUSE_NO_READING_S and bare["window"] == "no_reset_reported"


def test_rate_limit_wait_fails_the_leg_past_its_deadline_or_the_longest_wait():
    now = 100.0
    late = G.rate_limit_wait([_rejected(r5=now + 4 * 3600)], now, leg_deadline_ts=now + 4 * 3600 + 600,
                             pending=171, conc=30, median_rollout_s=600.0)
    assert late["beyond_deadline"] is True and late["remaining_estimate_s"] == 6 * 600.0
    assert late["reason"].startswith("usage limit: the limit resets at ") and "beyond this leg's deadline" in late["reason"]
    weekly = G.rate_limit_wait([_rejected(r7=now + 3 * 86400, kind="seven_day")], now, leg_deadline_ts=now + 43200,
                               pending=1, conc=30, median_rollout_s=None)
    assert weekly["window"] == "seven_day" and f"at most {G.WAIT_MAX_S // 3600} h" in weekly["reason"]


# ═══════════════════════════════════════════════════════════════ STATE.json
def test_state_round_trip_and_resume_selection(tmp_path):
    units = [f"{TRAIN[0]}#s1", f"{TRAIN[0]}#s2", f"{TRAIN[1]}#s1", f"{TRAIN[1]}#s2"]
    st = G.state_new(tag="r_v1_train", key="train", epoch=1, run_dir=tmp_path, bucket="0123456789ab",
                     skills_dir=tmp_path / "skills_v1", stamp="20260927T000000", units=units,
                     machine={"sandbox_id": "sb", "retire_at_ts": 1.0})
    # /2 rows carry score + snapshot_sha256
    assert st["schema"] == "evoskill.cc_leg_state/2"
    assert st["units"][units[3]] == {"state": "queued", "cid": None, "result": False, "grade_job_id": None,
                                     "graded_utc": None, "score": None, "snapshot_sha256": None}
    G.state_set_unit(st, units[0], state="graded", score=0.5, snapshot_sha256="ab" * 32, grade_job_id="j1")
    G.state_set_unit(st, units[1], state="graded", score=GF, grade_job_id="j2#regrade1")
    G.state_set_unit(st, units[2], state="running", cid="cc-x")
    with pytest.raises(ValueError):
        G.state_set_unit(st, units[3], state="bogus")
    p = tmp_path / "STATE.json"
    G.state_write(p, st)
    back = G.state_read(p)
    assert back["schema"] == G.STATE_SCHEMA and back["units"][units[0]]["score"] == 0.5 and back["updated_utc"]
    assert back["units"][units[1]]["score"] == "grader_failed"
    assert back["units"][units[2]] == {"state": "running", "cid": "cc-x", "result": False, "grade_job_id": None,
                                       "graded_utc": None, "score": None, "snapshot_sha256": None}
    # a graded sample is kept on a relaunch, a grader_failed one included (it is data, not a hole)
    assert G.resume_units(back, units) == ([units[0], units[1]], [units[2], units[3]])
    assert G.state_read(tmp_path / "none.json") is None
    for schema in ("other", "evoskill.cc_leg_state/1"):                 # a verdict-era store is never resumed
        p.write_text(json.dumps({"schema": schema, "units": {}}))
        with pytest.raises(ValueError, match="schema"):
            G.state_read(p)


@pytest.mark.skipif(not REAL_STREAMS, reason="EVOSKILL_TEST_CC_RUNS_DIR names no recorded Claude Code streams")
def test_real_stream_converts_and_terminates_like_the_reader_expects():
    p = REAL_STREAMS[0]
    s = p.read_text(errors="replace")
    res = json.loads((p.parent / "RESULT.json").read_text())
    tr = G.build_trace(stream_text=s, user_message=(p.parent / "user_message.txt").read_text(), result=res, score=1.0,
                       output_saved=bool(res.get("output")))
    assert tr["num_turns"] >= 1 and tr["messages"][0]["role"] == "user" and tr["messages"][1]["role"] == "assistant"
    assert (tr["budget_exhausted"] != "") == (res.get("subtype") == "error_max_turns")
    assert tr["tools"] == res["init"]["tools"] and tr["solver_harness"].startswith("claude_code 2.1.281")


def test_to_messages_and_model_replies_skip_driver_marker_lines():
    """The skills probe captures `claude`'s stdout plus the driver's own `CC_PROBE_RC=0`
    (and `STDERR: ...`) lines; the converter must ignore such non-event lines (as cc_solver_lib.events does), so
    rollout_evidence.skill_loads can read the probe's trace. A well-formed stream converts exactly as before."""
    s = cc_stream(skill="alpha")
    marked = s.rstrip("\n") + "\nCC_PROBE_RC=0\nSTDERR: Warning: no stdin data received in 3s\n"
    assert G.to_messages(marked, "U") == G.to_messages(s, "U")
    assert G.model_replies(marked) == G.model_replies(s) == 3


def test_to_messages_keeps_the_text_the_cli_injects_after_a_skill_launch():
    """Trace fidelity: Claude Code answers a Skill call with the tool result
    "Launching skill: X" and then injects the skill's body as a USER TEXT BLOCK ("Base directory for this skill: ...").
    A converter that kept only tool results would leave the evolver's trace without the skill text the solver read."""
    ev = [
        {"type": "system", "subtype": "init", "session_id": "s"},
        {"type": "assistant", "message": {"id": "m1", "content": [
            {"type": "tool_use", "id": "t1", "name": "Skill", "input": {"skill": "probe-skill"}}]}},
        {"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "Launching skill: probe-skill"},
            {"type": "text", "text": "Base directory for this skill: /evo/work/.claude/skills/probe-skill\n\nWhen invoked, reply with the first line."}]}},
        {"type": "assistant", "message": {"id": "m2", "content": [{"type": "text", "text": "done"}]}},
        {"type": "result", "subtype": "success", "result": "done"},
    ]
    s = "\n".join(json.dumps(e) for e in ev) + "\n"
    m = G.to_messages(s, "U")
    roles = [x["role"] for x in m]
    assert roles == ["user", "assistant", "tool", "user", "assistant"]
    assert m[2] == {"role": "tool", "tool_call_id": "t1", "content": "Launching skill: probe-skill"}
    assert m[3]["from_harness"] is True and m[3]["content"].startswith("Base directory for this skill: ")
    assert "When invoked, reply with the first line." in m[3]["content"]
    assert G.model_replies(s) == 2
