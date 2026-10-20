"""plugin_runner (plugin_api 3): static checks, sandboxed list_tasks, validation, freeze / verify, the /data view
(material_layout).

The plug-in is the single source of the benchmark's tasks and their sets: the loop reads no dataset of its own,
and the sandbox (not an import allow-list) is what the code may do. The run's grader is the user's own file setup/benchmark_grader.py -- here only read (AST) and pinned, never imported
or run; the plug-in (GOOD_PLUGIN) stays list_tasks only. The sandbox tests plant
plug-ins that each try ONE forbidden thing (network, a write outside the child's temp dir, a read of a .env / token
file / ~/.claude / <repo>/outputs / <repo>/.cache, fork, exec, a signal to another process) and require the attempt
to FAIL inside the child. Where a real secret path is probed the child only opens and closes it; nothing is ever
read or printed. macOS only (sandbox-exec).
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from src.orchestrator import plugin_api as A
from src.orchestrator import plugin_runner as R

REPO = Path(__file__).resolve().parents[1]
needs_sandbox = pytest.mark.skipif(not os.path.exists("/usr/bin/sandbox-exec"), reason="macOS sandbox-exec only")

GOOD_PLUGIN = '''"""Test plug-in: the adapter's shape for a SpreadsheetBench-like layout, with the split read from a file."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from plugin_api import Task

CATEGORIES = ("Debugging", "Financial_Model", "Template")
SPLIT = {"train": "train", "val": "validation", "test": "test"}


def _abs(path):
    return Path(os.path.abspath(path))


def list_tasks(root: Path) -> list[Task]:
    root = _abs(Path(root))
    sets = json.loads((root / "split.json").read_text())
    role = {t: SPLIT[k] for k in SPLIT for t in sets.get(k, [])}
    tasks = []
    for category in CATEGORIES:
        cat_dir = root / category
        with open(cat_dir / "dataset.json", encoding="utf-8") as f:
            records = json.load(f)
        for rec in records:
            tid = f"{category}/{rec['id']}"
            tasks.append(Task(id=tid, category=category, instruction=rec["instruction"],
                              inputs=(_abs(cat_dir / rec["spreadsheet_path"]),),
                              answer_files=(_abs(cat_dir / rec["golden_response_path"]),),
                              grading={"answer_position": rec["answer_position"]}, split=role.get(tid)))
    return tasks
'''

# the user's grader file, handed to the adapter; plugin_runner only reads and pins it.
GOOD_GRADER = '''"""Test grader: the user's shape -- grade(task, snapshot) -> a score from 0 to 1."""
from __future__ import annotations

from pathlib import Path


def _output(snapshot: Path, name: str) -> Path | None:
    p = snapshot / "output" / name
    return p if p.is_file() else None


def grade(task, snapshot: Path) -> float:
    out = _output(Path(snapshot), Path(task.inputs[0]).name)
    if out is None:
        return 0.0
    return 1.0 if out.read_bytes() == Path(task.answer_files[0]).read_bytes() else 0.0
'''

# Debugging/01_01 and 01_02 share one reference workbook (an answer-sharing pair, both in train).
RECORDS = {
    "Debugging": [("01_01", "01"), ("01_02", "01"), ("02_01", "02")],
    "Financial_Model": [("01_01", "01"), ("02_01", "02")],
    "Template": [("01_01", "01"), ("02_01", "02")],
}
SPLIT = {"train": ["Debugging/01_01", "Debugging/01_02", "Financial_Model/01_01"],
         "val": ["Template/01_01", "Financial_Model/02_01"],
         "test": ["Debugging/02_01"]}          # Template/02_01 is in no set


def make_data(root: Path) -> Path:
    for cat, recs in RECORDS.items():
        rows = []
        for rid, fam in recs:
            inp = f"spreadsheet/{rid}/{rid}_input.xlsx"
            gold = f"spreadsheet/{fam}/{fam}_golden.xlsx"
            for rel, content in ((inp, f"input {cat} {rid}"), (gold, f"golden {cat} {fam}")):
                p = root / cat / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                if not p.exists():
                    p.write_text(content)
            rows.append({"id": rid, "instruction": f"Do task {cat} {rid}.", "spreadsheet_path": inp,
                         "golden_response_path": gold, "answer_position": "'Sheet1'!A1:B2"})
        (root / cat / "dataset.json").write_text(json.dumps(rows))
    (root / "split.json").write_text(json.dumps(SPLIT))
    return root


ROLE = {**{t: "train" for t in SPLIT["train"]}, **{t: "validation" for t in SPLIT["val"]},
        **{t: "test" for t in SPLIT["test"]}}


def tasks_for(root: Path) -> list[dict]:
    out = []
    for cat, recs in RECORDS.items():
        for rid, fam in recs:
            tid = f"{cat}/{rid}"
            out.append({"id": tid, "category": cat, "instruction": f"Do task {cat} {rid}.",
                        "inputs": [str(root / cat / f"spreadsheet/{rid}/{rid}_input.xlsx")],
                        "answer_files": [str(root / cat / f"spreadsheet/{fam}/{fam}_golden.xlsx")],
                        "grading": {"answer_position": "'Sheet1'!A1:B2"}, "split": ROLE.get(tid)})
    return out


@pytest.fixture
def data(tmp_path):
    return make_data(tmp_path / "data")


def make_run(tmp_path: Path, root: Path) -> Path:
    run = tmp_path / "run"
    setup = run / "setup"
    setup.mkdir(parents=True)
    (setup / "benchmark_plugin.py").write_text(GOOD_PLUGIN)
    (setup / "benchmark_grader.py").write_text(GOOD_GRADER)
    shutil.copyfile(R.API_FILE, setup / "plugin_api.py")
    (run / "INFRA.json").write_text(json.dumps({"data": {"root": str(root)}}))
    return run


@pytest.fixture
def run(tmp_path, data):
    return make_run(tmp_path, data), data


# ─────────────────────────────────────────────────────────────── plugin_api
def test_api_is_version_3_with_a_split_the_grader_failed_label_and_no_group():
    t = A.Task(id="a", category="c", instruction="i", inputs=(), answer_files=(), grading={}, split=None)
    with pytest.raises(Exception):
        t.id = "b"
    assert A.API_VERSION == 3 and A.SPLITS == ("train", "validation", "test")
    assert A.GRADER_FAILED == "grader_failed"
    assert [f for f in A.Task.__dataclass_fields__] == list(R.TASK_KEYS)
    assert not hasattr(A, "GraderRef") and "group" not in A.Task.__dataclass_fields__
    assert R.SETUP_FILES == ("benchmark_plugin.py", "benchmark_grader.py", "plugin_api.py")


# ─────────────────────────────────────────────────────────────── static_check: parse + one function, nothing else
def test_static_accepts_the_plugin_and_any_import():
    r = R.static_check(GOOD_PLUGIN)
    assert r["ok"], r["problems"]
    assert "list_tasks" in r["top_level_names"] and "CATEGORIES" in r["top_level_names"]
    # no import allow-list (csv above; subprocess here): what the code may DO is the sandbox's rule
    assert R.static_check(GOOD_PLUGIN + "\nimport subprocess\n")["ok"]


@pytest.mark.parametrize("mutate, why", [
    (lambda s: s.replace("def list_tasks(root: Path)", "def list_tasks_x(root: Path)"), "no top-level def list_tasks"),
    (lambda s: s.replace("def list_tasks(root: Path)", "def list_tasks(root: Path, extra)"), "exactly one argument"),
    (lambda s: s + "\ndef broken(:\n", "does not parse"),
])
def test_static_structure(mutate, why):
    r = R.static_check(mutate(GOOD_PLUGIN))
    assert not r["ok"] and any(why in p for p in r["problems"]), r["problems"]


# ─────────────────────────────────────────────────────────────── static_check_grader: the user's grader file, AST only
SIG = "def grade(task, snapshot: Path)"


@pytest.mark.parametrize("sig", [SIG, "def grade(task, snapshot, /)", "def grade(t, s=None)"])
def test_static_grader_accepts_two_positional_arguments_and_any_top_level_import(sig):
    src = GOOD_GRADER.replace(SIG, sig)
    r = R.static_check_grader(src)
    assert r["ok"], r["problems"]
    assert {"grade", "_output"} <= set(r["top_level_names"]) and r["sha256"] == R.sha256_bytes(src.encode())
    # no lazy-import rule for the grader file: it is imported only in the grader's setup, never here
    top = src.replace("from pathlib import Path\n", "from pathlib import Path\n\nimport openpyxl\nimport subprocess\n", 1)
    assert R.static_check_grader(top)["ok"]


@pytest.mark.parametrize("mutate, why", [
    (lambda s: s.replace(SIG, "def grade_x(task, snapshot: Path)"),
     "no top-level def grade(task, snapshot) in benchmark_grader.py"),
    (lambda s: s.replace(SIG, "async def grade(task, snapshot: Path)"),
     "no top-level def grade(task, snapshot) in benchmark_grader.py"),
    (lambda s: s.replace(SIG, "def _grade(task, snapshot: Path)")
     + "\n\nclass G:\n    def grade(self, snapshot):\n        return _grade(self, snapshot)\n",
     "no top-level def grade(task, snapshot) in benchmark_grader.py"),
    (lambda s: s.replace(SIG, "def grade(task)"), "grade must take exactly two arguments (task, snapshot)"),
    (lambda s: s.replace(SIG, "def grade(task, snapshot, extra)"), "exactly two arguments"),
    (lambda s: s.replace(SIG, "def grade(*a)"), "exactly two arguments"),
    (lambda s: s.replace(SIG, "def grade(task, *snapshot)"), "exactly two arguments"),
    (lambda s: s.replace(SIG, "def grade(task, snapshot, **kw)"), "exactly two arguments"),
    (lambda s: s.replace(SIG, "def grade(task, *, snapshot)"), "exactly two arguments"),
    (lambda s: s.replace(SIG, "def grade(task, snapshot, *, how='x')"), "exactly two arguments"),
    (lambda s: s + "\ndef broken(:\n", "does not parse"),
])
def test_static_grader_structure(mutate, why):
    src = mutate(GOOD_GRADER)
    assert src != GOOD_GRADER
    r = R.static_check_grader(src)
    assert not r["ok"] and any(why in p for p in r["problems"]), r["problems"]


def test_the_plugin_check_is_not_the_grader_check():
    """Each file is held to its own function: a plug-in need not define grade, a grader need not define list_tasks."""
    assert "def grade" not in GOOD_PLUGIN and R.static_check(GOOD_PLUGIN)["ok"]
    assert not R.static_check_grader(GOOD_PLUGIN)["ok"]
    assert "def list_tasks" not in GOOD_GRADER and R.static_check_grader(GOOD_GRADER)["ok"]
    assert not R.static_check(GOOD_GRADER)["ok"]


# ─────────────────────────────────────────────────────────────── sandboxed run: positive
@needs_sandbox
def test_run_list_tasks_good_plugin(tmp_path, data):
    """A list_tasks-only plug-in with no grader file next to it (plugin_validate's path, plugin_api 3 included)."""
    setup = tmp_path / "setup"
    setup.mkdir()
    (setup / "benchmark_plugin.py").write_text(GOOD_PLUGIN)
    assert "def grade" not in GOOD_PLUGIN and not (setup / "benchmark_grader.py").exists()
    tasks = R.run_list_tasks(setup / "benchmark_plugin.py", data, timeout=60)
    assert sorted(tasks, key=lambda t: t["id"]) == sorted(tasks_for(data), key=lambda t: t["id"])
    assert R.validate(tasks)["ok"]


# ─────────────────────────────────────────────────────────────── sandboxed run: planted plug-ins
PLANT_HEAD = '''from pathlib import Path
from plugin_api import Task
def list_tasks(root):
    root = str(root)
    try:
'''
PLANT_TAIL = '''
    except OSError as e:
        raise RuntimeError("BLOCKED errno=%s" % e.errno)
    raise RuntimeError("FORBIDDEN_OK")
'''


def plant(setup: Path, body: str) -> Path:
    setup.mkdir(parents=True, exist_ok=True)
    p = setup / "benchmark_plugin.py"
    p.write_text(PLANT_HEAD + textwrap.indent(textwrap.dedent(body).strip("\n"), " " * 8) + PLANT_TAIL)
    return p


def run_planted(plugin, root, **k) -> str:
    with pytest.raises(R.PluginRunError) as ei:
        R.run_plugin(plugin, root, timeout=60, **k)
    return str(ei.value)


def assert_blocked(msg):
    assert "BLOCKED" in msg and "FORBIDDEN_OK" not in msg, msg


@needs_sandbox
def test_sandbox_blocks_network_to_a_live_local_listener(tmp_path, data):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(5)
    try:
        port = srv.getsockname()[1]
        p = plant(tmp_path / "setup", f"""
            import socket
            socket.create_connection(("127.0.0.1", {port}), timeout=5).close()
        """)
        assert_blocked(run_planted(p, data))
    finally:
        srv.close()


@needs_sandbox
@pytest.mark.parametrize("target", ["data_root", "setup_dir", "pytest_tmp"])
def test_sandbox_blocks_writes_outside_its_temp_dir(tmp_path, data, target):
    setup = tmp_path / "setup"
    dest = {"data_root": data / "escape.txt", "setup_dir": setup / "escape.txt",
            "pytest_tmp": tmp_path / "escape.txt"}[target]
    p = plant(setup, f"open({str(dest)!r}, 'w').write('x')")
    assert_blocked(run_planted(p, data))
    assert not dest.exists()


@needs_sandbox
def test_sandbox_allows_its_own_temp_dir_and_setup_dir(tmp_path, data):
    setup = tmp_path / "setup"
    setup.mkdir()
    (setup / "NOTES.json").write_text("{}")
    p = plant(setup, f"""
        import os
        open(os.environ["TMPDIR"] + "/scratch.txt", "w").write("x")
        open({str(setup / "NOTES.json")!r}).read()
        raise RuntimeError("READ_AND_WRITE_OK")
    """)
    msg = run_planted(p, data)
    assert "READ_AND_WRITE_OK" in msg, msg


@needs_sandbox
@pytest.mark.parametrize("name", [".env", ".env.local", "my_cc_oauth_token.txt", "cc_oauth_token"])
def test_sandbox_denies_secret_names_even_inside_the_data_root(tmp_path, data, name):
    secret = data / "Debugging" / name
    secret.write_text("PLANTED-NOT-A-REAL-SECRET")
    p = plant(tmp_path / "setup", f"open({str(secret)!r}).close()")
    assert_blocked(run_planted(p, data))


@needs_sandbox
def test_sandbox_named_denies_win_over_the_data_root_grant(tmp_path, data):
    """<repo>/outputs, <repo>/.cache and ~/.claude are denied by NAME, after every grant: here the fake repo
    and home sit INSIDE the data root, which is granted, and the reads must still fail."""
    fake_repo, fake_home = data / "fakerepo", data / "fakehome"
    for f in (fake_repo / "outputs/other_run/RESULT.json", fake_repo / ".cache/x/y.json",
              fake_home / ".claude/projects/p.jsonl"):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("{}")
    own_setup = fake_repo / "outputs/this_run/setup"
    for path in (fake_repo / "outputs/other_run/RESULT.json", fake_repo / ".cache/x/y.json",
                 fake_home / ".claude/projects/p.jsonl"):
        p = plant(own_setup, f"open({str(path)!r}).close()")
        assert_blocked(run_planted(p, data, repo=fake_repo, home=fake_home))
    (own_setup / "NOTES.json").write_text("{}")
    p = plant(own_setup, f"""
        open({str(own_setup / "NOTES.json")!r}).read()
        raise RuntimeError("OWN_SETUP_OK")
    """)
    assert "OWN_SETUP_OK" in run_planted(p, data, repo=fake_repo, home=fake_home)


REAL_PROBES = {
    # open+close or list names only; nothing is read or printed
    "repo_env": ("open", REPO / ".env"),
    "home_claude": ("list", Path.home() / ".claude"),
    "repo_outputs": ("list", REPO / "outputs"),
    "repo_cache": ("list", REPO / ".cache"),
    "home_dir": ("list", Path.home()),
    "repo_src": ("open", REPO / "src/orchestrator/loop.py"),
}


@needs_sandbox
@pytest.mark.parametrize("probe", sorted(REAL_PROBES))
def test_sandbox_denies_the_real_host_paths(tmp_path, data, probe):
    how, path = REAL_PROBES[probe]
    if not path.exists():
        pytest.skip(f"{path} not on this host")
    body = f"open({str(path)!r}).close()" if how == "open" else f"import os\nos.listdir({str(path)!r})"
    assert_blocked(run_planted(plant(tmp_path / "setup", body), data))


@needs_sandbox
def test_sandbox_blocks_exec_and_fork(tmp_path, data):
    p = plant(tmp_path / "setup", 'import subprocess\nsubprocess.run(["/bin/echo", "x"], check=True)')
    assert_blocked(run_planted(p, data))
    p = plant(tmp_path / "setup", """
        import os
        pid = os.fork()
        if pid == 0:
            os._exit(0)
    """)
    assert_blocked(run_planted(p, data))


@needs_sandbox
def test_sandbox_blocks_signals_to_other_processes(tmp_path, data):
    victim = subprocess.Popen(["/bin/sleep", "60"])
    try:
        p = plant(tmp_path / "setup", f"import os, signal\nos.kill({victim.pid}, signal.SIGTERM)")
        assert_blocked(run_planted(p, data))
        assert victim.poll() is None, "the sandboxed child killed another process"
    finally:
        victim.send_signal(signal.SIGKILL)
        victim.wait()


@needs_sandbox
def test_child_environment_is_scrubbed(tmp_path, data, monkeypatch):
    monkeypatch.setenv("EVOSKILL_TEST_PARENT_SECRET", "planted")
    p = plant(tmp_path / "setup", 'import os\nraise RuntimeError("ENV" + "LIST:" + ",".join(sorted(os.environ)))')
    msg = run_planted(p, data)
    names = msg.split("ENVLIST:")[-1].split("\n")[0].split(",")
    assert "EVOSKILL_TEST_PARENT_SECRET" not in names
    assert set(names) <= {"PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE", "__CF_USER_TEXT_ENCODING"}, names


@needs_sandbox
def test_timeout_kills_the_child(tmp_path, data):
    p = plant(tmp_path / "setup", "while True:\n    pass")
    with pytest.raises(R.PluginRunError, match="did not finish"):
        R.run_plugin(p, data, timeout=3)


@needs_sandbox
@pytest.mark.parametrize("body, why", [
    ("return [{'id': 'x'}]", "not plugin_api.Task"),
    ("return ()", "not list"),
    ("from dataclasses import dataclass\n@dataclass(frozen=True)\nclass Task:\n    id: str\nreturn [Task('x')]",
     "not plugin_api.Task"),
    ("return [Task(id='x', category='c', instruction='i', inputs=('rel/path',), answer_files=(), grading={}, "
     "split='train')]", "tuple of absolute Paths"),
    ("return [Task(id='x', category='c', instruction='i', inputs=(), answer_files=(), grading={'k': 1}, "
     "split='train')]", "grading"),
    ("return [Task(id='x', category='c', instruction='i', inputs=(), answer_files=(), grading={}, "
     "split='dev')]", "split"),
])
def test_child_type_checks_against_the_loops_own_dataclass(tmp_path, data, body, why):
    setup = tmp_path / "setup"
    setup.mkdir()
    src = PLANT_HEAD.split("    root = str(root)")[0] + textwrap.indent(body, "    ") + "\n"
    (setup / "benchmark_plugin.py").write_text(src)
    with pytest.raises(R.PluginRunError, match=why):
        R.run_plugin(setup / "benchmark_plugin.py", data, timeout=60)


def test_profile_rule_order(tmp_path):
    prof = R.build_profile(tmp=tmp_path, python=sys.executable, prefixes=[sys.base_prefix], data_root=tmp_path,
                           setup_dir=tmp_path / "s", repo=REPO, home=Path.home()).splitlines()
    idx = {line: i for i, line in enumerate(prof)}
    grant = prof.index(f'(allow file-read* (subpath "{os.path.realpath(tmp_path)}"))')   # first: the grant block
    named = idx[f'(deny file-read* (subpath "{os.path.realpath(REPO)}/outputs"))']
    assert idx['(deny file-read* (subpath "/Users"))'] < grant < named
    assert prof[-2:] == [r'(deny file-read* (regex #"/\.env[^/]*(/|$)"))', r'(deny file-read* (regex #"cc_oauth_token"))']
    assert "(deny network*)" in idx and "(deny process-exec)" in idx and "(deny file-write*)" in idx


# ─────────────────────────────────────────────────────────────── validate: the plug-in on its own terms
def by_id(tasks, tid):
    return next(t for t in tasks if t["id"] == tid)


def test_validate_accepts_the_right_tasks(data):
    v = R.validate(tasks_for(data))
    assert v["ok"], v["problems"]
    assert v["n_tasks"] == 7 and v["n_answer_groups"] == 6 and v["answer_groups_spanning_sets"] == 0
    assert v["split_counts"] == {"train": 3, "val": 2, "test": 1, "unused": 1}
    assert v["per_category"] == {"Debugging": 3, "Financial_Model": 2, "Template": 2}


@pytest.mark.parametrize("name, fn, why", [
    ("duplicate", lambda t: t.append(dict(t[0])), "duplicate ids"),
    ("answer_as_input", lambda t: t[0].update(inputs=list(t[0]["answer_files"])), "an input is an answer file"),
    ("no_train", lambda t: [x.update(split="test") for x in t], "no task is in the train set"),
    ("bad_split", lambda t: t[0].update(split="dev"), "split not one of"),
    ("shared_answer_across_sets", lambda t: by_id(t, "Debugging/01_02").update(split="test"),
     "sharing a reference file are in different sets"),
])
def test_validate_refuses(data, name, fn, why):
    t = tasks_for(data)
    fn(t)
    v = R.validate(t)
    assert not v["ok"] and any(why in p for p in v["problems"]), (name, v["problems"])


def test_validate_missing_file_and_input_bytes_equal_to_a_reference(data):
    (data / "Template/spreadsheet/02_01/02_01_input.xlsx").unlink()
    v = R.validate(tasks_for(data))
    assert any("missing file" in p for p in v["problems"])
    make_data(data)
    shutil.copyfile(data / "Template/spreadsheet/01/01_golden.xlsx", data / "Template/spreadsheet/01_01/01_01_input.xlsx")
    v = R.validate(tasks_for(data))
    assert any("same bytes as a reference file" in p for p in v["problems"]), v["problems"]


def test_a_shared_answer_with_an_unused_task_is_fine(data):
    """A task in no set never runs: its shared reference file constrains nothing."""
    t = tasks_for(data)
    by_id(t, "Debugging/01_02").update(split=None)
    assert R.validate(t)["ok"]


def test_split_from_tasks_and_the_split_reader_agree(data):
    sp = R.split_from_tasks(tasks_for(data))
    assert {k: sorted(v) for k, v in sp.items()} == {"train": sorted(SPLIT["train"]), "val": sorted(SPLIT["val"]),
                                                    "test": sorted(SPLIT["test"]), "unused": ["Template/02_01"]}
    _sp, roles = R._load_split(sp)
    assert roles["Debugging/01_01"] == "train" and roles["Template/01_01"] == "val" and roles["Template/02_01"] == "unused"


# ─────────────────────────────────────────────────────────────── check / freeze / verify
@needs_sandbox
def test_check_passes_on_a_good_run_and_carries_the_task_list(run):
    rd, root = run
    rep = R.check(rd)                                   # the data root comes from INFRA.json
    assert rep["ok"], rep["problems"]
    assert rep["run"]["n_tasks"] == 7 and rep["data_root"] == str(root)
    assert sorted(t["id"] for t in rep["tasks"]) == sorted(t["id"] for t in tasks_for(root))
    assert sorted(rep["files"]) == ["benchmark_grader.py", "benchmark_plugin.py", "plugin_api.py"]
    assert rep["files"]["benchmark_grader.py"] == R.sha256_file(rd / "setup/benchmark_grader.py")
    assert rep["grader_static"]["ok"] and "grade" in rep["grader_static"]["top_level_names"]


def _grader(sig):
    return lambda s: (s / "benchmark_grader.py").write_text(GOOD_GRADER.replace(SIG, sig))


@needs_sandbox
@pytest.mark.parametrize("name, mutate, why", [
    ("stub_differs", lambda s: (s / "plugin_api.py").write_text("# not canonical\n"), "differs from src/orchestrator"),
    ("static_fail", lambda s: (s / "benchmark_plugin.py").write_text("def list_tasks(root, x):\n    return []\n"),
     "not attempted"),
    ("raises", lambda s: (s / "benchmark_plugin.py").write_text("def list_tasks(root):\n    raise KeyError('x')\n"),
     "list_tasks raised"),
    ("no_train", lambda s: (s / "benchmark_plugin.py").write_text(
        GOOD_PLUGIN.replace('"train": "train"', '"train": "test"')), "no task is in the train set"),
    # the user's grader file
    ("grader_missing", lambda s: (s / "benchmark_grader.py").unlink(), "setup/benchmark_grader.py missing"),
    ("grader_no_grade", _grader("def grade_x(task, snapshot: Path)"),
     "grader static: no top-level def grade(task, snapshot) in benchmark_grader.py"),
    ("grader_one_arg", _grader("def grade(task)"), "grader static: grade must take exactly two arguments"),
    ("grader_star_args", _grader("def grade(*a)"), "grader static: grade must take exactly two arguments"),
    ("grader_kw_only", _grader("def grade(task, *, snapshot)"), "grader static: grade must take exactly two arguments"),
])
def test_check_refuses(run, name, mutate, why):
    rd, root = run
    mutate(rd / "setup")
    rep = R.check(rd)
    assert not rep["ok"] and any(why in p for p in rep["problems"]), (name, rep["problems"])
    if name.startswith("grader_"):          # the plug-in itself still ran: one report, every problem in it
        assert rep["run"]["n_tasks"] == 7 and len(rep["problems"]) == 1, rep["problems"]


@needs_sandbox
def test_a_top_level_third_party_import_fails_the_plugin_but_not_the_grader(run):
    """benchmark_plugin.py is imported by the `python -I -S` child: its top level is the standard library and
    plugin_api only. benchmark_grader.py is only read here, so the same import at its top level is fine."""
    rd, root = run
    g = rd / "setup/benchmark_grader.py"
    g.write_text(GOOD_GRADER.replace("from pathlib import Path\n", "from pathlib import Path\n\nimport openpyxl\n", 1))
    assert "import openpyxl" in g.read_text()
    rep = R.check(rd)
    assert rep["ok"], rep["problems"]
    p = rd / "setup/benchmark_plugin.py"
    p.write_text(GOOD_PLUGIN.replace("import os\n", "import os\n\nimport openpyxl\n", 1))
    assert "import openpyxl" in p.read_text()
    rep = R.check(rd)
    assert not rep["ok"] and any("list_tasks raised" in x and "openpyxl" in x for x in rep["problems"]), rep["problems"]


@needs_sandbox
def test_freeze_writes_tasks_and_split_then_verify(run):
    rd, root = run
    rec = R.freeze(rd)
    setup = rd / "setup"
    for name in ("PLUGIN.json", "PLUGIN_VALIDATION.json", "tasks.json", "split.json"):
        assert (setup / name).is_file(), name
    assert sorted(rec["files"]) == sorted(R.SETUP_FILES) and "operator_signature" not in rec
    # the user's grader file is pinned like the other two (no new key)
    assert sorted(rec["files"]) == ["benchmark_grader.py", "benchmark_plugin.py", "plugin_api.py"]
    for name in R.SETUP_FILES:
        assert rec["files"][name] == {"path": str(setup / name), "sha256": R.sha256_file(setup / name)}, name
    assert rec["api_version"] == 3 and "plugin_api 3" in rec["what"]
    assert rec["validation"]["sha256"] == R.sha256_file(setup / "PLUGIN_VALIDATION.json")
    tasks = json.loads((setup / "tasks.json").read_text())
    assert sorted(t["id"] for t in tasks) == sorted(t["id"] for t in tasks_for(root))
    assert rec["tasks_json"]["sha256"] == R.sha256_file(setup / "tasks.json") and rec["tasks_json"]["n_tasks"] == 7
    split = json.loads((setup / "split.json").read_text())
    assert split == R.split_from_tasks(tasks) and rec["split_json"]["counts"] == {"train": 3, "val": 2, "test": 1, "unused": 1}
    assert "tasks" not in json.loads((setup / "PLUGIN_VALIDATION.json").read_text())
    rep = R.verify_frozen(rd, recheck=True)
    assert rep["ok"] and rep["rechecked"]
    assert R.frozen_tasks(rd) == tasks


@needs_sandbox
@pytest.mark.parametrize("target", list(R.SETUP_FILES) + ["PLUGIN_VALIDATION.json", "tasks.json", "split.json"])
def test_verify_refuses_drift(run, target):
    rd, root = run
    R.freeze(rd)
    path = rd / "setup" / target
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(R.PluginDrift, match="changed"):
        R.verify_frozen(rd)


@needs_sandbox
def test_verify_refuses_an_edited_record_and_an_older_api(run):
    rd, root = run
    R.freeze(rd)
    pj = rd / "setup/PLUGIN.json"
    rec = json.loads(pj.read_text())
    rec["data_root"] = "/somewhere/else"
    pj.write_text(json.dumps(rec))
    with pytest.raises(R.PluginDrift, match="edited after freeze"):
        R.verify_frozen(rd)
    rec = json.loads(pj.read_text())
    rec["data_root"] = str(root)
    rec["api_version"] = 2
    rec["record_sha256"] = R._record_sha(rec)
    pj.write_text(json.dumps(rec))
    with pytest.raises(R.PluginDrift, match="api_version 2, not 3"):
        R.verify_frozen(rd)
    # an api-2 record's shape: benchmark_grader.py not pinned (every freeze of 0928..0929b)
    rec["api_version"] = 3
    del rec["files"]["benchmark_grader.py"]
    rec["record_sha256"] = R._record_sha(rec)
    pj.write_text(json.dumps(rec))
    with pytest.raises(R.PluginDrift) as ei:
        R.verify_frozen(rd)
    assert ei.value.report["problems"] == [
        "PLUGIN.json pins ['benchmark_plugin.py', 'plugin_api.py'], lacking ['benchmark_grader.py']"]


@needs_sandbox
def test_recheck_catches_a_plugin_whose_list_changed_but_whose_source_did_not(run):
    """The data root moved under the plug-in: the frozen files still hash, a fresh list_tasks differs."""
    rd, root = run
    R.freeze(rd)
    (root / "Template/dataset.json").write_text(json.dumps(json.loads((root / "Template/dataset.json").read_text())[:1]))
    assert R.verify_frozen(rd)["ok"]
    with pytest.raises(R.PluginDrift, match="recheck"):
        R.verify_frozen(rd, recheck=True)


@needs_sandbox
def test_freeze_refuses_a_failing_check_and_an_existing_record(run):
    rd, root = run
    good = (rd / "setup/benchmark_plugin.py").read_text()
    (rd / "setup/benchmark_plugin.py").write_text("def list_tasks(root):\n    return []\n")
    with pytest.raises(R.FreezeRefused, match="check failed"):
        R.freeze(rd)
    assert not (rd / "setup/PLUGIN.json").exists()
    (rd / "setup/benchmark_plugin.py").write_text(good)
    R.freeze(rd)
    with pytest.raises(R.FreezeRefused, match="exists"):
        R.freeze(rd)
    R.freeze(rd, force=True)
    assert list((rd / "setup").glob("PLUGIN.json.superseded.*"))


@needs_sandbox
def test_cli_check_freeze_verify(run):
    rd, root = run
    base = [sys.executable, "-m", "src.orchestrator.plugin_runner"]
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    r = subprocess.run(base + ["check", "--run-dir", str(rd)], cwd=REPO, capture_output=True, text=True, env=env,
                       timeout=120)
    assert r.returncode == 0 and json.loads(r.stdout)["ok"], r.stdout + r.stderr
    r = subprocess.run(base + ["freeze", "--run-dir", str(rd)], cwd=REPO, capture_output=True, text=True, env=env,
                       timeout=120)
    assert r.returncode == 0 and json.loads(r.stdout)["split_counts"]["train"] == 3, r.stdout + r.stderr
    r = subprocess.run(base + ["verify", "--run-dir", str(rd)], cwd=REPO, capture_output=True, text=True, env=env,
                       timeout=120)
    assert r.returncode == 0 and json.loads(r.stdout)["ok"], r.stdout + r.stderr


@needs_sandbox
def test_a_task_id_in_the_plugin_source_is_not_refused(run):
    """The setup files are not scanned for held-out task ids: the plug-in may name any task."""
    rd, root = run
    p = rd / "setup/benchmark_plugin.py"
    p.write_text(p.read_text() + "\n# see Debugging/02_01 for an example\n")
    assert R.check(rd)["ok"]


# ─────────────────────────────────────────────────────────────── material_layout: the user's material under /data


def test_material_layout_names_each_source_under_data_by_its_own_name():
    lay = R.material_layout(Path("/m/SpreadsheetBench-v2/data/spreadsheetbench-v2"), Path("/m/splits/drop48/split.json"),
                            Path("/m/EvoSkill"))
    assert R.MATERIAL_ROOT == "/data"
    assert lay == {"/m/SpreadsheetBench-v2/data/spreadsheetbench-v2": "/data/spreadsheetbench-v2",
                   "/m/splits/drop48/split.json": "/data/split.json", "/m/EvoSkill": "/data/EvoSkill"}
    assert R.material_layout(Path("/d/x"), Path("/s.json"))[str(R.REPO)] == "/data/EvoSkill"
    assert R.material_layout(Path("/d/x"), Path("/s.json"), Path("/m/EvoSkill-fork"))["/m/EvoSkill-fork"] == "/data/EvoSkill"
    with pytest.raises(ValueError, match="two sources map to one path under /data"):
        R.material_layout(Path("/a/data"), Path("/b/data"), Path("/m/EvoSkill"))


def test_setup_ao_probe_delegates_its_vm_layout(monkeypatch):
    """The adapter's upload and the grader's setup use ONE definition of the /data view."""
    monkeypatch.syspath_prepend(str(REPO / "scripts" / "daytona"))
    import setup_ao_probe as S
    assert S.VM_DATA == R.MATERIAL_ROOT and S.REPO == R.REPO
    args = (Path("/m/sb/data/spreadsheetbench-v2"), Path("/m/split.json"))
    assert S.vm_layout(*args) == R.material_layout(*args)
    assert S.vm_layout(*args, Path("/m/EvoSkill")) == R.material_layout(*args, Path("/m/EvoSkill"))
    with pytest.raises(SystemExit, match="REFUSING: two sources map to one path"):
        S.vm_layout(Path("/a/data"), Path("/b/data"))
