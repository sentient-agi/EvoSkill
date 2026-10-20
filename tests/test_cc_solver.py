"""Claude Code as the SSB v2 solver on Daytona (scripts/cc_solver/): the parts that decide what the solver sees and
what the numbers mean. NO network, NO Daytona, NO model call, NO key.

  * the sample: seed 926 over split['test'] of the pinned drop-48 split is deterministic, pinned, and refused on a
    drifted split;
  * the stream: init / result / every rate_limit_info (first and last, both windows, resetsAt), a torn last line;
  * prompt parity: the user message is the direct solver's own `solver_instruction` expression (ast, not a copy),
    byte-equal to an independent rendering; the append text differs from prompt.md in the working-directory line
    only; the launch script hands claude exactly that argv (bash round trip, trailing newline kept), the token on
    fd 3 only, no effort pin;
  * the upload set: exactly the input workbook; a reference workbook (by bytes, name or path) or any extra file is
    refused; the tar holds regular files under evo/work/ only; the in-container listing check;
  * the container spec, the claude flags (no --bare, no --effort), the init-event checks, the grade summary never
    carrying a diff body, the quota delta.
"""
from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "cc_solver"))
import cc_solver_lib as L  # noqa: E402
import cc_solver as C  # noqa: E402

# the drop-48 split the reported SSB runs used (shipped as the example split; same bytes)
SPLIT = ROOT / "examples" / "ssbv2" / "split.json"
SPLIT_SHA256 = "5be714178186e14406b70efd8d00c51f2b135b0c61038ca8b2467d0f87a1da5d"
SSBV2_ROOT = Path(os.environ.get("SSBV2_ROOT") or "/nonexistent-ssbv2-root")   # the SSB v2 checkout, when present

PINNED_SAMPLE = ["Template/06_20", "Financial_Model/13_03", "Debugging/02_09", "Template/15_04", "Template/06_17",
                 "Financial_Model/12_02", "Debugging/04_06", "Debugging/01_03", "Template/06_02", "Template/14_09"]


# ───────────────────────────────────────────────────────────────────────────── sampling
@pytest.mark.skipif(not SPLIT.is_file(), reason="the drop-48 split is not on this machine")
def test_sample_is_deterministic_and_pinned():
    a, b = L.sample_tasks(SPLIT, expect_sha256=SPLIT_SHA256), L.sample_tasks(SPLIT, expect_sha256=SPLIT_SHA256)
    assert a["task_ids"] == b["task_ids"] == PINNED_SAMPLE
    assert a["seed"] == 926 and a["key"] == "test" and a["pool_size"] == 104 and a["n"] == 10
    assert a["split_sha256"] == SPLIT_SHA256
    test = json.loads(SPLIT.read_text())["test"]
    assert set(a["task_ids"]) <= set(test) and len(set(a["task_ids"])) == 10
    assert L.sample_tasks(SPLIT, seed=927)["task_ids"] != PINNED_SAMPLE


def test_sample_refuses_a_drifted_split(tmp_path):
    p = tmp_path / "split.json"
    p.write_text(json.dumps({"test": [f"T/{i:02d}" for i in range(30)], "dropped": {"task_ids": []}}))
    with pytest.raises(ValueError, match="sha256"):
        L.sample_tasks(p, expect_sha256=SPLIT_SHA256)
    s1 = L.sample_tasks(p, expect_sha256=None)
    s2 = L.sample_tasks(p, expect_sha256=None)
    assert s1["task_ids"] == s2["task_ids"] and len(s1["task_ids"]) == 10


def test_sample_never_draws_a_dropped_task(tmp_path):
    p = tmp_path / "split.json"
    p.write_text(json.dumps({"test": ["A/1", "A/2"], "dropped": {"task_ids": ["A/1"]}}))
    with pytest.raises(ValueError, match="dropped"):
        L.sample_tasks(p, n=2, expect_sha256=None)


# ───────────────────────────────────────────────────────────────────────────── the stream
def _rl(five, seven, status="allowed", r5=1790430600, r7=1790877600):
    return {"type": "rate_limit_event", "rate_limit_info": {
        "status": status, "resetsAt": r5, "rateLimitType": "five_hour", "overageStatus": "rejected",
        "isUsingOverage": False, "unifiedWindows": {"five_hour": {"utilization": five, "resetsAt": r5},
                                                    "seven_day": {"utilization": seven, "resetsAt": r7}}}}


def _stream(tail_torn=True):
    ev = [{"type": "system", "subtype": "init", "cwd": L.WS_CONTAINER, "session_id": "s", "uuid": "u",
           "tools": ["Bash", "Edit", "Read", "Write", "Skill", "ToolSearch"], "mcp_servers": [],
           "model": "claude-sonnet-5", "permissionMode": "bypassPermissions", "apiKeySource": "none",
           "claude_code_version": L.ctl.CC_VERSION, "skills": [], "plugins": [
               {"name": "agents-md", "path": "builtin", "source": "agents-md@builtin"}]},
          {"type": "assistant", "message": {"model": "claude-sonnet-5", "content": [
              {"type": "tool_use", "name": "Bash", "input": {"command": "ls"}}]}},
          _rl(0.12, 0.36),
          {"type": "assistant", "message": {"model": "claude-sonnet-5", "content": [{"type": "text", "text": "x"}]}},
          _rl(0.15, 0.37, status="allowed_warning"),
          {"type": "result", "subtype": "error_max_turns", "is_error": True, "num_turns": 45, "duration_ms": 600000,
           "duration_api_ms": 500000, "total_cost_usd": 3.21, "usage": {"input_tokens": 10, "output_tokens": 20},
           "modelUsage": {"claude-sonnet-5": {"costUSD": 3.21}}, "result": "done", "permission_denials": []}]
    text = "\n".join(json.dumps(e) for e in ev) + "\n"
    return text + ('{"type": "assist' if tail_torn else "")


def test_parse_stream_result_and_rate_limits():
    s = L.parse_stream(_stream())
    assert s["n_unparsed_lines"] == 1 and s["n_events"] == 6
    assert s["result"]["subtype"] == "error_max_turns" and s["result"]["num_turns"] == 45
    assert s["result"]["duration_ms"] == 600000 and s["result"]["total_cost_usd"] == 3.21
    rl = s["rate_limit"]
    assert rl["n"] == 2
    assert rl["first"]["five_hour"]["utilization"] == 0.12 and rl["first"]["seven_day"]["utilization"] == 0.36
    assert rl["last"]["five_hour"]["utilization"] == 0.15 and rl["last"]["seven_day"]["utilization"] == 0.37
    assert rl["last"]["status"] == "allowed_warning"
    assert rl["first"]["five_hour"]["resetsAt"] == 1790430600
    assert rl["first"]["five_hour"]["resetsAt_utc"].endswith("Z")
    assert rl["first"]["seven_day"]["resetsAt"] == 1790877600
    assert s["tool_uses"] == {"Bash": 1} and s["assistant_models"] == {"claude-sonnet-5": 2}
    assert "session_id" not in s["init"] and "uuid" not in s["init"]


def test_rate_limit_info_found_when_nested_and_absent_stream():
    nested = json.dumps({"type": "stream_event", "event": {"x": [_rl(0.5, 0.6)]}})
    s = L.parse_stream(nested + "\n")
    assert s["rate_limit"]["first"]["five_hour"]["utilization"] == 0.5
    e = L.parse_stream("")
    assert e["result"] is None and e["rate_limit"] == {"n": 0, "first": None, "last": None}


def test_init_problems_clean_and_dirty():
    init = L.parse_stream(_stream())["init"]
    assert L.init_problems(init) == []
    dirty = dict(init, skills=["dataviz"], memory_paths={"auto": "/home/evo/.claude/projects/x/memory/"},
                 mcp_servers=[{"name": "m"}], tools=init["tools"] + ["WebFetch"], apiKeySource="ANTHROPIC_API_KEY",
                 plugins=[{"name": "p", "path": "/home/evo/.claude/plugins/p"}], cwd="/tmp", model="claude-opus-5")
    probs = " | ".join(L.init_problems(dirty))
    for word in ("skills", "memory", "mcp", "WebFetch", "apiKeySource", "plugins", "cwd", "model"):
        assert word in probs, word
    assert L.init_problems(None) == ["no init event"]


# ───────────────────────────────────────────────────────────────────────────── the solver's first message
@pytest.mark.parametrize("ins,fn", [
    ("Complete the model.\nFill in all empty cells.", "06_20_input.xlsx"),
    ("Fix {braces} and 'quotes' and \"dq\" and $HOME `x`", "Relative vs Absolute DIfference_input.xlsx"),
])
def test_the_first_message_is_bare_metal_the_task_text_and_its_paths_only(ins, fn):
    """The solver prompt holds only bare-metal minimal information; adding instructions is the evolver's
    business -- no preamble, no task-id line, no prompt.md."""
    got = L.task_user_message(ins, fn)
    assert got == (f"{ins}\n\nInput spreadsheet: ./spreadsheets/{fn}\n"
                   f"Save your modified file to: ./output/{fn}\n")
    assert L.SOLVER_PREAMBLE == "" and not got.startswith("Task:") and "scored" not in got
    assert L.PROMPT_DELIVERY.startswith("stock system prompt") and "task's own text" in L.PROMPT_DELIVERY


def test_launch_script_hands_claude_exactly_the_argv(tmp_path):
    um = L.task_user_message("Fix it's \"all\" $(rm -rf /) `x`\n\n", "a b_input.xlsx")
    inner = L.launch_inner(um)
    out, work = tmp_path / "out", tmp_path / "work"
    out.mkdir()
    work.mkdir()
    fake = tmp_path / "claude"
    fake.write_text("#!/usr/bin/env python3\nimport json,os,sys\n"
                    "print(json.dumps({'argv':sys.argv[1:],'fd3':os.read(3,64).decode(),'cwd':os.getcwd(),"
                    "'env':dict(os.environ)}))\n")
    fake.chmod(0o755)
    x = (inner.replace(L.OUT_CONTAINER, str(out)).replace(L.WS_CONTAINER, str(work))
         .replace("CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR=3 claude ", f"CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR=3 {fake} "))
    r = subprocess.run(["/bin/bash", "-c", 'bash -c "$0" 3<<<"FAKE-TOKEN"', x], capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", "CLAUDE_CODE_EFFORT_LEVEL": "high", "ANTHROPIC_API_KEY": "k"})
    assert r.returncode == 0, r.stderr
    got = json.loads((out / L.STREAM_NAME).read_text())
    assert got["argv"] == L.claude_argv(um)[1:] and "--append-system-prompt" not in got["argv"]
    assert got["fd3"].strip() == "FAKE-TOKEN" and os.path.realpath(got["cwd"]) == os.path.realpath(work)
    env = got["env"]
    assert env["CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR"] == "3"
    for k in L.FORBIDDEN_ENV:
        assert k not in env, k
    for k, v in L.CC_ENV.items():
        assert env[k] == v
    assert (out / "rc").read_text().strip() == "0" and (out / "pid").read_text().strip().isdigit()


def test_claude_flags():
    argv = L.claude_argv("U")
    assert argv[:3] == ["claude", "-p", "U"]
    assert "--bare" not in argv and "--effort" not in argv
    assert argv.count("--thinking-display") == 1 and argv[argv.index("--thinking-display") + 1] == "summarized"
    assert L.has_thinking_display(argv) and not L.has_thinking_display(argv[:-2])
    i = argv.index
    assert argv[i("--model") + 1] == "claude-sonnet-5" and argv[i("--max-turns") + 1] == "45"
    assert argv[i("--output-format") + 1] == "stream-json" and "--verbose" in argv
    assert argv[i("--permission-mode") + 1] == "bypassPermissions" and "--strict-mcp-config" in argv
    assert "--append-system-prompt" not in argv          # the solver keeps Claude Code's stock system prompt
    dis = argv[i("--disallowedTools") + 1].split(",")
    assert set(L.CC_DISALLOWED_REQUESTED) <= set(dis) and "Bash" not in dis
    assert L.cc_env()["BASH_DEFAULT_TIMEOUT_MS"] == "600000"
    assert not [k for k in L.CC_ENV if "EFFORT" in k or "TOKEN" in k or k.startswith("ANTHROPIC_")]


def test_token_never_in_a_command_string():
    box = SimpleNamespace(cid="cc-x-0926t0000-ab12", cdir=L.ctl.cdir("cc-x-0926t0000-ab12"))
    tok = L.TOKEN_PREFIX + "A" * 40
    cmd = C.Box.token_cmd(box, L.launch_inner(L.task_user_message("Fix it.", "a_input.xlsx")), cwd="/tmp",
                          detached=False, tok=tok)
    assert tok not in cmd and '3<<<"$CLAUDE_CODE_OAUTH_TOKEN"' in cmd and cmd.startswith("env -u CLAUDE_CODE_OAUTH_TOKEN")
    det = C.Box.token_cmd(box, "true", cwd=L.WS_CONTAINER, detached=True, tok=tok)
    assert det.startswith("(setsid nohup env -u CLAUDE_CODE_OAUTH_TOKEN") and "--keep-fd3" in det
    with pytest.raises(C.Refuse):
        C.Box.token_cmd(box, "echo " + tok, cwd="/tmp", detached=False, tok=tok)


# ───────────────────────────────────────────────────────────────────────────── the upload set
def _ws(tmp_path, name="06_20_input.xlsx", data=b"PK-input-bytes"):
    ws = tmp_path / "ws"
    (ws / "spreadsheets").mkdir(parents=True)
    (ws / "output").mkdir()
    (ws / "spreadsheets" / name).write_bytes(data)
    return ws


def test_upload_set_exactly_the_input(tmp_path):
    ws = _ws(tmp_path)
    rows = L.upload_rows(ws)
    ok = L.check_upload_set(rows, spreadsheet_filename="06_20_input.xlsx",
                            input_sha256=L.sha256_bytes(b"PK-input-bytes"), golden_shas={}, golden_paths=[])
    assert ok == [] and [r["vm_path"] for r in rows] == [f"{L.WS_CONTAINER}/spreadsheets/06_20_input.xlsx"]


def test_upload_set_refuses_any_reference_file(tmp_path):
    golden = b"PK-golden-bytes"
    ws = _ws(tmp_path, data=golden)                           # the input slot holds reference bytes
    rows = L.upload_rows(ws)
    p = L.check_upload_set(rows, spreadsheet_filename="06_20_input.xlsx", input_sha256=L.sha256_bytes(golden),
                           golden_shas={L.sha256_bytes(golden): "Template/06_20"}, golden_paths=[])
    assert any("reference workbook" in x for x in p)
    ws2 = _ws(tmp_path / "b")
    (ws2 / "spreadsheets" / "06_20_golden.xlsx").write_bytes(b"other")
    p2 = L.check_upload_set(L.upload_rows(ws2), spreadsheet_filename="06_20_input.xlsx",
                            input_sha256=L.sha256_bytes(b"PK-input-bytes"), golden_shas={}, golden_paths=[])
    assert any("upload set" in x for x in p2) and any("reference-like" in x for x in p2)
    ws3 = _ws(tmp_path / "c")
    src = ws3 / "spreadsheets" / "06_20_input.xlsx"
    p3 = L.check_upload_set(L.upload_rows(ws3), spreadsheet_filename="06_20_input.xlsx",
                            input_sha256=L.sha256_bytes(b"PK-input-bytes"), golden_shas={}, golden_paths=[str(src)])
    assert any("source is a reference" in x for x in p3)
    # a legitimate input whose name contains an ordinary word ("References") is not a reference workbook
    ws4 = _ws(tmp_path / "d", name="Incorrect Cross Sheet References_input.xlsx")
    assert L.check_upload_set(L.upload_rows(ws4), spreadsheet_filename="Incorrect Cross Sheet References_input.xlsx",
                              input_sha256=L.sha256_bytes(b"PK-input-bytes"), golden_shas={}, golden_paths=[]) == []


def test_upload_rows_refuse_links(tmp_path):
    ws = _ws(tmp_path)
    (ws / "spreadsheets" / "link.xlsx").symlink_to("/etc/hosts")
    with pytest.raises(ValueError, match="symlink"):
        L.upload_rows(ws)


def test_upload_tar_and_listing_check(tmp_path):
    ws = _ws(tmp_path, name="a b_input.xlsx")
    rows = L.upload_rows(ws)
    with tarfile.open(fileobj=io.BytesIO(L.build_upload_tgz(rows)), mode="r:gz") as tf:
        ms = tf.getmembers()
    assert [m.name for m in ms] == ["evo/work/spreadsheets/a b_input.xlsx"]
    assert all(m.isfile() and m.name.startswith(L.ctl.UPLOAD_TOPS) for m in ms)
    f = rows[0]["vm_path"]
    good = [[L.WS_CONTAINER, "D", "", 0], [L.OUT_CONTAINER, "D", "", 0], [f"{L.WS_CONTAINER}/spreadsheets", "D", "", 0],
            [f"{L.WS_CONTAINER}/output", "D", "", 0], [f, "F", rows[0]["sha256"], rows[0]["bytes"]]]
    assert L.listing_problems(good, rows) == []
    assert L.listing_problems(good + [["/evo/x/golden.xlsx", "F", "ab", 1]], rows)
    assert L.listing_problems(good + [[f"{L.WS_CONTAINER}/l", "L", "/etc", 0]], rows)
    assert L.listing_problems(good[:3] + good[4:], rows) == ["output/ missing"]


@pytest.mark.skipif(not (SSBV2_ROOT / "data/spreadsheetbench-v2/Template/dataset.json").is_file(),
                    reason="the SSB v2 dataset is not on this machine")
def test_real_workspace_builder_uploads_no_reference(tmp_path):
    """The repo's own builder on a sampled task: after the skills link is dropped the upload set is the one input
    workbook, and its bytes equal no reference workbook of the whole dataset."""
    from src.agent_profiles.ssbv2_agent.dataset import TASK_INDEX, load_ssbv2
    tw = pytest.importorskip("src.agent_profiles.ssbv2_agent.task_workspace")   # the harness-free solver's builder
    setup_task_workspace, task_ws_name = tw.setup_task_workspace, tw.task_ws_name
    load_ssbv2(SSBV2_ROOT)
    t = PINNED_SAMPLE[0]
    info = TASK_INDEX[t]
    skills = tmp_path / "skills"
    skills.mkdir()
    ws = setup_task_workspace(task_id=t, spreadsheet_path=info["spreadsheet_path"], tasks_cache_root=tmp_path,
                              shared_skills_dir=skills)
    assert ws.name == task_ws_name(t) == L.safe_id(t)
    (ws / ".claude" / "skills").unlink()
    (ws / ".claude").rmdir()
    golden_shas = {L.sha256_file(i["golden_path"]): tid for tid, i in TASK_INDEX.items()}
    rows = L.upload_rows(ws)
    fn = Path(info["spreadsheet_path"]).name
    assert L.check_upload_set(rows, spreadsheet_filename=fn, input_sha256=L.sha256_file(info["spreadsheet_path"]),
                              golden_shas=golden_shas, golden_paths=[i["golden_path"] for i in TASK_INDEX.values()]) == []
    assert sorted(p.name for p in ws.iterdir()) == ["output", "spreadsheets"]


# ───────────────────────────────────────────────────────────────────────────── container / grade / table
def test_container_spec_valid_and_nothing_writable_below_users():
    cid = L.make_cid("Financial_Model/13_03", "0926t2000", "ab12")
    spec = L.container_spec(cid, wall_s=5400)
    assert L.ctl.validate_spec(spec) == [] and spec["rw"] == [] and spec["egress"]["allow"] == "api.anthropic.com"
    assert spec["limits"]["mem_max_bytes"] * 10 <= 32 * L.GIB
    pl = L.ctl.plan(spec, {"sysdirs": {"bin": "symlink:usr/bin"}, "masks_present": []},
                    L.ctl.resolve_limits({"mem_total": 32 * L.GIB, "ncpu": 8, "cg_root_cpu_max": "800000 100000"},
                                         spec["limits"]))
    rw = [m["dst"] for m in pl if m["kind"] == "bind" and not m["ro"]]
    assert L.WS_CONTAINER in rw and not [d for d in rw if d.startswith("/Users")]


def test_grade_brief_never_carries_the_diff_body():
    rec = {"status": "ok", "passed": False, "n_match": 3, "n_total": 10, "host": {"outcome": "graded_wrong"},
           "diff_file_text": "FAIL — 3/10 graded cells match\n(scorer's own diff)\n\nSheet1!B2: expected 42, got 41\n",
           "diff": "Sheet1!B2: expected 42, got 41"}
    g = L.grade_brief(rec)
    assert g["verdict"] == "graded_wrong" and g["scorer_header"] == "FAIL — 3/10 graded cells match"
    assert "expected" not in json.dumps(g)
    assert L.grade_brief(None) == {"verdict": "not_eligible_no_output"}
    assert L.grade_brief({"status": "timeout", "host": {}})["verdict"] == "grader_timeout"
    assert L.grade_brief(dict(rec, passed=True))["verdict"] == "pass"


def test_resolve_output_like_the_direct_solver(tmp_path):
    assert L.resolve_output(tmp_path, "a.xlsx") is None
    (tmp_path / "z.xlsx").write_bytes(b"z")
    (tmp_path / "b.xlsx").write_bytes(b"b")
    assert L.resolve_output(tmp_path, "a.xlsx").name == "b.xlsx"
    (tmp_path / "a.xlsx").write_bytes(b"a")
    assert L.resolve_output(tmp_path, "a.xlsx").name == "a.xlsx"


def test_table_renders():
    t = L.table([{"task": "Template/06_20", "subtype": "success", "num_turns": 12, "minutes": 4.25, "cost": 1.5,
                  "grade": {"verdict": "pass", "n_match": 9, "n_total": 9}},
                 {"task": "Debugging/02_09", "grade": {"verdict": "not_eligible_no_output"}}])
    assert "Template/06_20" in t and "9/9" in t and "4.2" in t and "not_eligible_no_output" in t


def test_poll_batch_parses_every_container():
    d1 = base64.b64encode(b'{"type":"system"}\n').decode()

    class M:
        def sh(self, cmd, **kw):
            return 0, (f"@@CID@@ cc-a-1\n120 RUNNING 4096 1\n@@B64@@{d1}\n@@CID@@ cc-b-2\n0 0 100 1\n@@B64@@\n"
                       f"@@CID@@ cc-c-3\nevoctl: boom\n")
    r = C.poll_batch(M(), [SimpleNamespace(cid=c, off=0) for c in ("cc-a-1", "cc-b-2", "cc-c-3")])
    assert r["cc-a-1"] == (120, "RUNNING", 4096, "1", b'{"type":"system"}\n')
    assert r["cc-b-2"] == (0, "0", 100, "1", b"") and r["cc-c-3"] is None


# ═════════════════════════════════════════════════════════════════════════════ the TRAIN base rollouts (171 units)
@pytest.mark.skipif(not SPLIT.is_file(), reason="the drop-48 split is not on this machine")
def test_train_selection_is_57_train_tasks_x_3():
    s = L.train_selection(SPLIT, expect_sha256=SPLIT_SHA256)
    split = json.loads(SPLIT.read_text())
    assert s["split_sha256"] == SPLIT_SHA256 and s["key"] == "train" and s["samples"] == 3
    assert s["n_tasks"] == 57 and s["n_units"] == 171 and s["task_ids"] == split["train"]
    ids = [u["unit"] for u in s["units"]]
    assert len(set(ids)) == 171
    assert ids[:57] == [f"{t}#s1" for t in split["train"]] and ids[-1] == f"{split['train'][-1]}#s3"  # sample-major
    for u in s["units"]:
        assert L.parse_unit(u["unit"]) == (u["task"], u["sample_idx"]) and u["sample_idx"] in (1, 2, 3)
    others = set(split["val"]) | set(split["test"]) | set(split["unused"]) | set(split["dropped"]["task_ids"])
    assert not {u["task"] for u in s["units"]} & others
    assert L.units_split_problems(s["units"], SPLIT, "train") == []
    assert L.units_split_problems(s["units"][:2], SPLIT, "test")          # never graded / run as test
    assert L.units_split_problems([{"unit": split["val"][0] + "#s1", "task": split["val"][0], "sample_idx": 1}],
                                  SPLIT, "train")


def test_train_selection_refuses_a_split_whose_train_meets_another_set(tmp_path):
    p = tmp_path / "split.json"
    p.write_text(json.dumps({"train": ["A/01", "A/02"], "val": ["A/03"], "test": ["A/02"], "unused": [],
                             "dropped": {"task_ids": []}}))
    with pytest.raises(ValueError, match="shares tasks"):
        L.train_selection(p, expect_sha256=None)
    p.write_text(json.dumps({"train": ["A/01", "A/02"], "val": [], "test": [], "dropped": {"task_ids": ["A/01"]}}))
    with pytest.raises(ValueError, match="shares tasks"):
        L.train_selection(p, expect_sha256=None)
    p.write_text(json.dumps({"train": ["A/01", "A/02"], "val": ["B/01"], "test": ["C/01"]}))
    s = L.train_selection(p, expect_sha256=None, samples=2)
    assert [u["unit"] for u in s["units"]] == ["A/01#s1", "A/02#s1", "A/01#s2", "A/02#s2"]
    with pytest.raises(ValueError, match="sha256"):
        L.train_selection(p, expect_sha256=SPLIT_SHA256)


def test_unit_ids_dirs_and_container_ids():
    assert L.unit_id("Template/06_20", None) == "Template/06_20" and L.unit_id("Template/06_20", 2) == "Template/06_20#s2"
    assert L.parse_unit("Financial_Model/13_03#s3") == ("Financial_Model/13_03", 3)
    assert L.parse_unit("Debugging/02_09") == ("Debugging/02_09", None)
    for bad in ("Debugging/02_09#s0", "Debugging/02_09#x1", "../etc#s1", "Debugging/02_09#s1#s2", ""):
        with pytest.raises(ValueError):
            L.parse_unit(bad)
    assert L.unit_dir_name("Financial_Model/13_03#s3") == "Financial_Model_13_03__s3"
    assert L.unit_dir_name("Template/06_20") == "Template_06_20"                  # the test probe's dirs unchanged
    cids = {L.make_cid(f"Financial_Model/13_03#s{k}", "0926t2200", "ab12") for k in (1, 2, 3)}
    assert len(cids) == 3 and all(L.ctl.CID_RE.match(c) for c in cids)


def test_unit_dir_is_the_repo_workspace_name():
    tw = pytest.importorskip("src.agent_profiles.ssbv2_agent.task_workspace")
    for t, k in (("Financial_Model/13_03", 2), ("Template/06_20", None)):
        assert L.unit_dir_name(L.unit_id(t, k)) == tw.task_ws_name(t, k)


def _root_with_selection(tmp_path, units=("A/01#s1", "A/01#s2", "B/02#s1")):
    root = tmp_path / "S"
    root.mkdir()
    us = [{"unit": u, "task": L.parse_unit(u)[0], "sample_idx": L.parse_unit(u)[1]} for u in units]
    (root / L.SELECTION_FILE).write_text(json.dumps({"key": "train", "samples": 2, "units": us, "task_ids": []}))
    return root


def test_select_units_all_rest_ungraded_and_task_ids(tmp_path):
    root = _root_with_selection(tmp_path)
    sel = C.load_selection(root)
    assert C.select_units("all", sel, root) == ["A/01#s1", "A/01#s2", "B/02#s1"]
    assert C.select_units("B/02#s1,A/01", sel, root) == ["A/01#s1", "A/01#s2", "B/02#s1"]
    assert C.select_units("A/01#s2", sel, root) == ["A/01#s2"]
    with pytest.raises(C.Refuse):
        C.select_units("C/03", sel, root)
    rd = L.run_dir(root, "A/01#s1")
    rd.mkdir(parents=True)
    (rd / "RESULT.json").write_text(json.dumps({"completed": True}))
    rd2 = L.run_dir(root, "A/01#s2")
    rd2.mkdir(parents=True)
    (rd2 / "RESULT.json").write_text(json.dumps({"completed": False}))       # an infra error: redone by rest
    assert C.select_units("rest", sel, root) == ["A/01#s2", "B/02#s1"]
    assert C.select_units("ungraded", sel, root) == ["A/01#s1"]
    (rd / "GRADE.json").write_text("{}")
    assert C.select_units("ungraded", sel, root) == []
    (root / L.SAMPLE_FILE).write_text(json.dumps({"task_ids": ["A/01"]}))
    with pytest.raises(C.Refuse, match="exactly one"):
        C.load_selection(root)


@pytest.mark.skipif(not SPLIT.is_file(), reason="the drop-48 split is not on this machine")
def test_cmd_sample_train_writes_its_own_selection(tmp_path, capsys):
    root = tmp_path / "S"
    assert C.main(["sample", "--out", str(root), "--split", str(SPLIT), "--kind", "train"]) == 0
    s = json.loads((root / L.SELECTION_FILE).read_text())
    assert s["n_units"] == 171 and not (root / L.SAMPLE_FILE).exists()
    assert C.main(["sample", "--out", str(root), "--split", str(SPLIT), "--kind", "train"]) == 0          # idempotent
    assert C.main(["sample", "--out", str(root), "--split", str(SPLIT), "--kind", "test10"]) == 4         # one selection per out root
    assert C.main(["sample", "--out", str(root), "--split", str(SPLIT), "--kind", "train", "--samples", "2"]) == 4


def test_limits_for_20_on_one_machine():
    lim = L.limits_for(20)
    assert lim["mem_max_bytes"] == 1536 << 20 and lim["mem_high_ratio"] == 0.85 and lim["cpus"] == 2.0
    assert lim["pids_max"] == 2048 and lim["tmp_bytes"] == 1 * L.GIB and lim["rw_max_bytes"] == 1 * L.GIB
    assert 20 * lim["mem_max_bytes"] + L.PROBE_LIMITS["mem_max_bytes"] <= 31 * L.GIB
    assert L.limits_for(10) == L.LIMITS and L.limits_for(1) == L.LIMITS
    with pytest.raises(ValueError):
        L.limits_for(31)
    facts = {"mem_total": 32 * L.GIB, "ncpu": 8, "cg_root_cpu_max": "800000 100000"}
    for spec_lim in (lim, L.PROBE_LIMITS):
        spec = L.container_spec(L.make_cid("Template/06_20#s1", "0926t2200", "ab12"), wall_s=5400, limits=spec_lim)
        assert L.ctl.validate_spec(spec) == []
        r = L.ctl.resolve_limits(facts, spec["limits"])
        assert r["mem_max_bytes"] == spec_lim["mem_max_bytes"] and r["cpu_max"] == f"{int(spec_lim['cpus'] * 1e5)} 100000"
        assert r["mem_high_bytes"] == int(spec_lim["mem_max_bytes"] * 0.85) // 4096 * 4096


def test_only_a_rejection_stops_and_it_names_its_reset():
    b = lambda five, seven=0.3, status="allowed": L.rl_brief(_rl(five, seven, status=status)["rate_limit_info"])  # noqa: E731
    # no utilization level and no warning stops admission: only a session the API rejected for a usage limit
    assert L.rejection_reason(b(0.02)) is None and L.rejection_reason(b(1.0)) is None
    assert L.rejection_reason(b(0.95, status="allowed_warning")) is None and L.rejection_reason(None) is None
    why = L.rejection_reason(b(0.1, status="rejected"))
    assert why.startswith("quota rejected (five_hour") and "resets 20" in why
    assert L.rejection_reset(b(0.1, status="rejected")) == (1790430600.0, "five_hour")
    assert L.rejection_reset(None) == (None, None)


def _think(text):
    return {"type": "assistant", "message": {"model": "claude-sonnet-5", "content": [
        {"type": "thinking", "thinking": text, "signature": "sig"}]}}


def test_thinking_blocks_are_counted_and_checked():
    s = L.parse_stream(_stream() + "\n" + json.dumps(_think("Let me open the workbook.")) + "\n" +
                       json.dumps(_think("")) + "\n")
    assert s["thinking"] == {"blocks": 2, "with_text": 1, "chars": 25, "redacted": 0}
    argv = L.claude_argv("U")
    assert L.thinking_problems(s["thinking"], argv) == []
    empty = L.parse_stream(json.dumps(_think("")) + "\n" + json.dumps(_think("")) + "\n")["thinking"]
    assert "none carries text" in L.thinking_problems(empty, argv)[0]
    assert "lacks --thinking-display" in L.thinking_problems({"blocks": 0}, argv[:-2])[0]
    assert L.thinking_problems(L.parse_stream("")["thinking"], argv) == []       # no thinking = no verdict


def test_token_file_option(tmp_path):
    tok = L.TOKEN_PREFIX + "Z" * 60
    p = tmp_path / "cc_token"
    p.write_text(tok + "\n")
    p.chmod(0o600)
    assert C.read_token(p) == tok
    p.chmod(0o644)
    with pytest.raises(C.Refuse, match="readable by others"):
        C.read_token(p)
    p.chmod(0o600)
    p.write_text("sk-ant-api03-" + "Z" * 60)
    with pytest.raises(C.Refuse, match="one OAuth token"):
        C.read_token(p)
    link = tmp_path / "link"
    link.symlink_to(p)
    with pytest.raises(C.Refuse, match="regular file"):
        C.read_token(link)
    seen = {}
    orig = C.cmd_run
    req = ["--out", str(tmp_path), "--split", str(SPLIT), "--run-dir", str(tmp_path), "--tasks", "all"]
    try:
        C.cmd_run = lambda a: seen.setdefault("a", a) and 0
        C.main(["run", *req])
        assert seen["a"].token_file == "" and seen.pop("a").concurrency == 20   # default: the environment's
        C.main(["run", *req, "--token-file", "/data/cc_token", "--concurrency", "12"])
        assert seen["a"].token_file == "/data/cc_token" and seen["a"].concurrency == 12
    finally:
        C.cmd_run = orig


def test_grade_job_labels_are_distinct_per_sample():
    import re
    labs = {C.job_label(k) for k in (1, 2, 3)}
    assert len(labs) == 3 and C.job_label(None) == C.JOB_LABEL
    for lab in labs:
        assert re.fullmatch(r"[A-Za-z0-9._:/=+@#-]{1,240}", f"Financial_Model/13_03/saved/{lab}/grade")


def test_per_task_pass_at_3():
    def row(u, v):
        return {"unit": u, "task": L.parse_unit(u)[0], "grade": {"verdict": v} if v else {}}
    rows = [row("A/01#s1", "pass"), row("A/01#s2", "graded_wrong"), row("A/01#s3", "not_eligible_no_output"),
            row("B/02#s1", "graded_wrong"), row("B/02#s2", "graded_wrong"), row("B/02#s3", "graded_wrong"),
            row("C/03#s1", "pass"), row("C/03#s2", None), row("C/03#s3", "not_graded_incomplete")]
    pt = {t["task"]: t for t in L.per_task(rows)}
    assert pt["A/01"] == dict(pt["A/01"], measured=3, passed=1, pass_at_k=1, k=3)
    assert pt["B/02"]["pass_at_k"] == 0 and pt["C/03"]["measured"] == 1 and pt["C/03"]["k"] == 1
    s = L.pass_summary(rows, 3)
    assert s["measured"] == 7 and s["passed"] == 2 and s["pass_at_1"] == round(2 / 7, 4)
    assert s["tasks_with_3_measured"] == 2 and s["pass_at_3"] == 0.5 and s["tasks_any_pass"] == 2
    assert "A/01" in L.task_table(L.per_task(rows))


# ───────────────────────────────────────────────────────────────────────────── the queue (fake machine, fake boxes)
EVENTS: list = []
EV_LOCK = __import__("threading").Lock()


def _ev(kind, unit):
    with EV_LOCK:
        EVENTS.append((kind, unit))


class FakeBox:
    """Box's surface as the Queue drives it; no machine. STREAMS[unit] = the stream the rollout writes."""
    STREAMS: dict = {}
    FAIL_SETUP: set = set()

    def __init__(self, m, root, unit, stamp, wall_s, limits):
        self.m, self.unit, self.task = m, unit, unit
        self.rd = L.run_dir(root, unit)
        self.rd.mkdir(parents=True, exist_ok=True)
        self.cid = L.make_cid(unit, stamp, "ab12")
        self.rec = {"argv": L.claude_argv("U")}
        self.status, self.off, self.fail, self.stopped, self.rc = "new", 0, 0, None, None
        self.t_launch = self.stream_fh = self.guard = None
        self.wiped = self.result_written = False
        self.polls = 0

    def save(self):
        pass

    def prepare(self):
        _ev("prepare", self.unit)
        self.status = "prepared"

    def up(self, init_root):
        assert init_root is False                      # the quota container initialised the root cgroup
        self.status = "up"

    def probe(self):
        if self.unit in self.FAIL_SETUP:
            raise C.Refuse(f"{self.unit}: PRE probe: boom")

    def upload(self):
        pass

    def toolchain(self):
        self.status = "ready"

    def launch(self, tok):
        assert tok == "TOK"
        self.stream_fh = open(self.rd / "stream.jsonl", "wb")
        self.t_launch = __import__("time").time()
        _ev("launch", self.unit)
        self.status = "running"

    def stop(self, why):
        self.stopped = why

    def collect(self, tok):
        self.stream_fh.close()
        _ev("collect", self.unit)
        self.status = "collected"

    def wipe(self):
        _ev("wipe", self.unit)
        return True

    def result(self):
        r = {"unit": self.unit, "completed": self.status == "collected"}
        (self.rd / "RESULT.json").write_text(json.dumps(r))
        return r


def _fake_poll(m, boxes):
    out = {}
    for b in boxes:
        b.polls += 1
        data = (FakeBox.STREAMS.get(b.unit) or _stream_with(0.2)).encode()
        out[b.cid] = (len(data), "0", 0, "1", data) if b.polls >= 2 else (0, "RUNNING", 0, "1", b"")
    return out


def _queue(tmp_path, units, conc, *, ttl_s=240 * 60):
    import threading as _th
    import time as _t
    m = SimpleNamespace(lock=_th.Lock(), t_created=_t.time(), ttl=240, rec={}, save=lambda: None, dir=tmp_path)
    EVENTS.clear()
    return C.Queue(m, tmp_path, list(units), "20260926T220000", 5400, L.limits_for(conc), "TOK", conc,
                   box_cls=FakeBox, poll=_fake_poll, ttl_s=ttl_s, poll_s=0)


def _stream_with(five, status="allowed"):
    return "\n".join(json.dumps(e) for e in [json.loads(ln) for ln in _stream(tail_torn=False).splitlines()]
                     + [_rl(five, 0.3, status=status), _think("I will recalc.")]) + "\n"


def test_queue_refills_with_a_fixed_number_in_flight(tmp_path):
    FakeBox.STREAMS, FakeBox.FAIL_SETUP = {}, set()
    units = [f"A/{i:02d}#s1" for i in range(1, 8)]
    qu = _queue(tmp_path, units, 3)
    qu.run()
    qu.shutdown()
    launched = [u for k, u in EVENTS if k == "launch"]
    assert sorted(launched) == units and not qu.pending and qu.guard["admitting"] is True
    assert qu.max_inflight == 3
    held = 0
    for k, u in EVENTS:                   # a container is set up only while fewer than 3 are held (refill after wipe)
        held += {"prepare": 1, "wipe": -1}.get(k, 0)
        assert 0 <= held <= 3
    for u in units:                       # each unit: collected, then wiped, then its RESULT
        seq = [k for k, x in EVENTS if x == u]
        assert seq == ["prepare", "launch", "collect", "wipe"]
        assert json.loads((L.run_dir(tmp_path, u) / "RESULT.json").read_text())["completed"]
    assert len(qu.readings) == 7 and all(r["thinking_chars"] == len("I will recalc.") for r in qu.readings)
    assert all(r["five_hour"] == 0.2 and r["status"] == "allowed" and not r["stop_reasons"] for r in qu.readings)


def test_queue_guard_stops_admission_and_lets_in_flight_finish(tmp_path):
    FakeBox.FAIL_SETUP = set()
    units = [f"A/{i:02d}#s1" for i in range(1, 9)]
    FakeBox.STREAMS = {u: _stream_with(0.40) for u in units}
    FakeBox.STREAMS["A/03#s1"] = _stream_with(0.91, status="rejected")   # the third rollout's LAST reading: rejected
    qu = _queue(tmp_path, units, 2)
    qu.run()
    qu.shutdown()
    launched = [u for k, u in EVENTS if k == "launch"]
    prepared = [u for k, u in EVENTS if k == "prepare"]
    assert qu.guard["admitting"] is False and "quota rejected" in qu.guard["reason"]
    assert qu.guard["by"] == "A/03#s1"
    # admitted before the stop: u1..u3, and u4 if a slot freed before u3 ended (thread timing); nothing after it
    assert sorted(prepared) in (units[:3], units[:4]) and set(units[:3]) <= set(launched) <= set(prepared)
    assert list(qu.pending) == units[len(prepared):] and qu.guard["pending_at_stop"] == len(qu.pending)
    assert all(("wipe", u) in EVENTS for u in prepared)       # in-flight ones finished, the unlaunched torn down
    for u in launched:
        assert json.loads((L.run_dir(tmp_path, u) / "RESULT.json").read_text())["completed"]
    assert sorted(r["unit"] for r in qu.readings) == sorted(launched)
    assert qu.readings[0]["thinking_chars"] == len("I will recalc.")


def test_queue_guard_stops_on_a_rejection_and_not_on_a_warning(tmp_path):
    FakeBox.FAIL_SETUP = set()
    units = [f"A/{i:02d}#s1" for i in range(1, 5)]
    FakeBox.STREAMS = {u: _stream_with(0.2) for u in units}
    FakeBox.STREAMS["A/01#s1"] = _stream_with(0.97, status="allowed_warning")    # a warning: admission goes on
    FakeBox.STREAMS["A/02#s1"] = _stream_with(0.2, status="rejected")
    qu = _queue(tmp_path, units, 1)
    qu.run()
    qu.shutdown()
    assert [u for k, u in EVENTS if k == "launch"] == ["A/01#s1", "A/02#s1"]
    assert "quota rejected" in qu.guard["reason"] and qu.guard["by"] == "A/02#s1" and qu.guard["kind"] == "quota"


def test_queue_guard_on_thinking_without_text_and_setup_failures_and_ttl(tmp_path):
    units = [f"A/{i:02d}#s1" for i in range(1, 5)]
    FakeBox.FAIL_SETUP = set()
    FakeBox.STREAMS = {u: _stream(tail_torn=False) + json.dumps(_think("")) + "\n" for u in units}
    qu = _queue(tmp_path / "t", units, 1)
    qu.run()
    qu.shutdown()
    assert "none carries text" in qu.guard["reason"] and len([1 for k, _ in EVENTS if k == "launch"]) == 1
    FakeBox.STREAMS = {}
    FakeBox.FAIL_SETUP = set(units[:3])
    qu = _queue(tmp_path / "f", units, 1)
    qu.run()
    qu.shutdown()
    assert "3 setup failures" in qu.guard["reason"] and list(qu.pending) == ["A/04#s1"]
    assert [u for k, u in EVENTS if k == "wipe"] == units[:3]        # every failed setup is torn down
    FakeBox.FAIL_SETUP = set()
    qu = _queue(tmp_path / "x", units, 2, ttl_s=5400 + 10 * 60)
    qu.run()
    qu.shutdown()
    assert "TTL" in qu.guard["reason"] and not EVENTS and len(qu.pending) == 4


# ───────────────────────────────────────────────────────────────────────────── the loop's leg driver's keywords
# (cc_leg_driver). Every default keeps the behaviour the tests above pin.
def test_infra_of_reads_the_given_run_dir(tmp_path):
    (tmp_path / "INFRA.json").write_text(json.dumps({"leg_sandbox": {"libreoffice": "LO"}}))
    assert L.infra_of(tmp_path) == {"leg_sandbox": {"libreoffice": "LO"}}
    assert C.infra(tmp_path) == L.infra_of(tmp_path)


def test_box_class_defaults_are_cc_solvers():
    assert C.Box.expect_link_count == 0 and C.Box.expect_links is None and C.Box.extra_dirs == () and C.Box.skill_rows == []


def test_judge_passes_expect_skills_to_init_problems(tmp_path):
    base = _stream_with(0.2).splitlines()                # a clean reading: the only stop reason can be the init
    init = json.loads(base[0])
    assert init["subtype"] == "init"
    init["skills"] = ["alpha"]
    lines = [json.dumps(init)] + base[1:]
    rd = tmp_path / "runs" / "A_01__s1"
    rd.mkdir(parents=True)
    (rd / "stream.jsonl").write_text("\n".join(lines) + "\n")
    b = SimpleNamespace(rd=rd, status="collected", rc=0, unit="A/01#s1", rec={"argv": L.claude_argv("U")})
    assert any(r.startswith("init") for r in C.judge(b, 0.9)["stop_reasons"])            # default: a skill is a defect
    assert C.judge(b, 0.9, expect_skills=["alpha"])["stop_reasons"] == []
    assert any("skills" in r for r in C.judge(b, 0.9, expect_skills=["beta"])["stop_reasons"])
    assert C.Queue.stop_kind(["quota rejected (five_hour; resets 2026-10-01T00:00:00Z)"]) == "quota"
    assert C.Queue.stop_kind(["HTTP 429 api_error"]) == "quota"
    assert C.Queue.stop_kind(["quota x", "init ['skills']"]) == "defect" and C.Queue.stop_kind([]) == "defect"


def test_queue_close_kind_reopen_and_on_idle_closed_hook(tmp_path):
    FakeBox.FAIL_SETUP = set()
    units = [f"A/{i:02d}#s1" for i in range(1, 6)]
    FakeBox.STREAMS = {u: _stream_with(0.40) for u in units}
    FakeBox.STREAMS["A/02#s1"] = _stream_with(0.95, status="rejected")   # the second rollout is rejected
    resumed = []

    def on_idle_closed():
        if resumed:
            return False
        resumed.append(qu.guard["reason"])
        FakeBox.STREAMS["A/02#s1"] = _stream_with(0.40)   # the re-read is clean: admit again
        qu.reopen("waited for the reset")
        return True
    qu = _queue(tmp_path, units, 1)
    qu.on_idle_closed = on_idle_closed
    qu.run()
    qu.shutdown()
    assert resumed and "quota rejected" in resumed[0] and qu.guard["admitting"] is True and not qu.pending
    assert sorted(u for k, u in EVENTS if k == "launch") == units
    assert qu.guard["reopenings"][0]["note"] == "waited for the reset" and "kind" not in qu.guard
    # a closed queue with pending units and NO hook returns as before (the guard keeps its kind)
    FakeBox.STREAMS = {u: _stream_with(0.40) for u in units}
    FakeBox.STREAMS["A/01#s1"] = _stream_with(0.95, status="rejected")
    qu = _queue(tmp_path / "n", units, 1)
    qu.run()
    qu.shutdown()
    assert qu.guard["admitting"] is False and qu.guard["kind"] == "quota" and len(qu.pending) == 4
    # a defect stop is never reopened by the hook's contract: the hook sees kind "defect"
    FakeBox.STREAMS = {u: _stream(tail_torn=False) + json.dumps(_think("")) + "\n" for u in units}
    seen = []
    qu = _queue(tmp_path / "d", units, 1)
    qu.on_idle_closed = lambda: seen.append(qu.guard["kind"]) or False
    qu.run()
    qu.shutdown()
    assert seen == ["defect"] and qu.guard["admitting"] is False


def test_queue_deadline_ts_replaces_the_machine_ttl(tmp_path):
    import time as _t
    units = [f"A/{i:02d}#s1" for i in range(1, 3)]
    FakeBox.STREAMS, FakeBox.FAIL_SETUP = {}, set()
    qu = _queue(tmp_path, units, 2)
    qu.deadline_ts = _t.time() + 5400 + 10 * 60            # a rollout (5400 s) + the margin would outlive it
    qu.run()
    qu.shutdown()
    assert "TTL" in qu.guard["reason"] and qu.guard["kind"] == "ttl" and len(qu.pending) == 2 and not EVENTS
    qu = _queue(tmp_path / "ok", units, 2, ttl_s=60)        # the machine TTL alone would refuse...
    qu.deadline_ts = _t.time() + 5400 + 25 * 60             # ...but the pool machine's retire_at_ts governs
    qu.run()
    qu.shutdown()
    assert qu.guard["admitting"] is True and not qu.pending


def test_a_warning_is_never_a_stop():
    """The API warns from 0.75 of the week and 0.90 of the five hours; a warning at any utilization is not a stop, a
    rejection is."""
    weekly = {"status": "allowed_warning", "rateLimitType": "seven_day", "utilization": 0.99, "resetsAt": 1_800_000_000}
    assert L.rejection_reason(weekly) is None
    assert L.rejection_reason(dict(weekly, status="allowed")) is None
    why = L.rejection_reason(dict(weekly, status="rejected"))
    assert why.startswith("quota rejected (seven_day") and L.rejection_reset(dict(weekly, status="rejected")) == \
        (1_800_000_000.0, "seven_day")
