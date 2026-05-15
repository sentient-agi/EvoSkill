"""Tests for Harbor integration — loader, scorer, agent, config validation, docker/daytona."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.cli.config import (
    DatasetConfig,
    HarborConfig,
    HarnessConfig,
    ProjectConfig,
    ScorerConfig,
)


# ── helpers ──────────────────────────────────────────────────────────────────


def _make_harbor_cfg(tmp_path, *, harbor_enabled=True, dataset_source="harbor",
                     harbor_tasks_root=None, inner_agent="claude-code",
                     inner_model="anthropic/claude-sonnet-4-5", env="docker"):
    (tmp_path / ".evoskill").mkdir(exist_ok=True)
    root = harbor_tasks_root or str(tmp_path / "tasks")
    return ProjectConfig(
        harness=HarnessConfig(name="claude"),
        dataset=DatasetConfig(source=dataset_source, harbor_tasks_root=root),
        scorer=ScorerConfig(type="harbor" if dataset_source == "harbor" else "multi_tolerance"),
        harbor=HarborConfig(
            enabled=harbor_enabled,
            inner_agent=inner_agent,
            inner_model=inner_model,
            env=env,
        ),
        project_root=tmp_path,
    )


def _make_task_dir(root: Path, org: str, name: str, *,
                   category: str = "default", difficulty: str = "default") -> Path:
    """Create a minimal Harbor task directory with task.toml."""
    digest = "abc123"
    task_dir = root / org / name / digest
    task_dir.mkdir(parents=True)
    toml = (
        f'[task]\nname = "{org}/{name}"\n\n'
        f'[metadata]\ncategory = "{category}"\ndifficulty = "{difficulty}"\n'
    )
    (task_dir / "task.toml").write_text(toml)
    return task_dir


# ── harbor_loader ────────────────────────────────────────────────────────────


class TestHarborLoader:

    def test_load_discovers_tasks(self, tmp_path):
        from src.api.harbor_loader import load_harbor_tasks
        root = tmp_path / "tasks"
        _make_task_dir(root, "org1", "task_a")
        _make_task_dir(root, "org1", "task_b")
        cfg = _make_harbor_cfg(tmp_path, harbor_tasks_root=str(root))
        df = load_harbor_tasks(cfg)
        assert len(df) == 2
        assert set(df["question"]) == {"org1/task_a", "org1/task_b"}

    def test_load_populates_task_path_index(self, tmp_path):
        from src.api.harbor_loader import TASK_PATH_INDEX, load_harbor_tasks
        root = tmp_path / "tasks"
        _make_task_dir(root, "myorg", "mytask")
        cfg = _make_harbor_cfg(tmp_path, harbor_tasks_root=str(root))
        load_harbor_tasks(cfg)
        assert "myorg/mytask" in TASK_PATH_INDEX
        assert TASK_PATH_INDEX["myorg/mytask"].is_dir()

    def test_load_sets_ground_truth_sentinel(self, tmp_path):
        from src.api.harbor_loader import load_harbor_tasks
        root = tmp_path / "tasks"
        _make_task_dir(root, "org", "t1")
        cfg = _make_harbor_cfg(tmp_path, harbor_tasks_root=str(root))
        df = load_harbor_tasks(cfg)
        assert df["ground_truth"].iloc[0] == "1.0"

    def test_load_reads_category(self, tmp_path):
        from src.api.harbor_loader import load_harbor_tasks
        root = tmp_path / "tasks"
        _make_task_dir(root, "org", "t1", category="math")
        cfg = _make_harbor_cfg(tmp_path, harbor_tasks_root=str(root))
        df = load_harbor_tasks(cfg)
        assert df["category"].iloc[0] == "math"

    def test_load_filters_by_include(self, tmp_path):
        from src.api.harbor_loader import load_harbor_tasks
        root = tmp_path / "tasks"
        _make_task_dir(root, "org", "keep_me")
        _make_task_dir(root, "org", "skip_me")
        cfg = _make_harbor_cfg(tmp_path, harbor_tasks_root=str(root))
        cfg.dataset.harbor_include = ["org/keep*"]
        df = load_harbor_tasks(cfg)
        assert len(df) == 1
        assert df["question"].iloc[0] == "org/keep_me"

    def test_load_filters_by_exclude(self, tmp_path):
        from src.api.harbor_loader import load_harbor_tasks
        root = tmp_path / "tasks"
        _make_task_dir(root, "org", "good")
        _make_task_dir(root, "org", "bad")
        cfg = _make_harbor_cfg(tmp_path, harbor_tasks_root=str(root))
        cfg.dataset.harbor_exclude = ["org/bad"]
        df = load_harbor_tasks(cfg)
        assert len(df) == 1
        assert df["question"].iloc[0] == "org/good"

    def test_load_filters_by_difficulty(self, tmp_path):
        from src.api.harbor_loader import load_harbor_tasks
        root = tmp_path / "tasks"
        _make_task_dir(root, "org", "easy_one", difficulty="easy")
        _make_task_dir(root, "org", "hard_one", difficulty="hard")
        cfg = _make_harbor_cfg(tmp_path, harbor_tasks_root=str(root))
        cfg.dataset.harbor_difficulty = ["easy"]
        df = load_harbor_tasks(cfg)
        assert len(df) == 1
        assert df["question"].iloc[0] == "org/easy_one"

    def test_load_respects_limit(self, tmp_path):
        from src.api.harbor_loader import load_harbor_tasks
        root = tmp_path / "tasks"
        for i in range(5):
            _make_task_dir(root, "org", f"task_{i:02d}")
        cfg = _make_harbor_cfg(tmp_path, harbor_tasks_root=str(root))
        cfg.dataset.harbor_limit = 3
        df = load_harbor_tasks(cfg)
        assert len(df) == 3

    def test_load_raises_on_missing_root(self, tmp_path):
        from src.api.harbor_loader import HarborLoadError, load_harbor_tasks
        cfg = _make_harbor_cfg(tmp_path, harbor_tasks_root=str(tmp_path / "nonexistent"))
        with pytest.raises(HarborLoadError, match="does not exist"):
            load_harbor_tasks(cfg)

    def test_load_raises_on_empty_root(self, tmp_path):
        from src.api.harbor_loader import HarborLoadError, load_harbor_tasks
        root = tmp_path / "empty_tasks"
        root.mkdir()
        cfg = _make_harbor_cfg(tmp_path, harbor_tasks_root=str(root))
        with pytest.raises(HarborLoadError, match="no task.toml"):
            load_harbor_tasks(cfg)

    def test_load_raises_on_blank_root(self, tmp_path):
        from src.api.harbor_loader import HarborLoadError, load_harbor_tasks
        cfg = _make_harbor_cfg(tmp_path, harbor_tasks_root="")
        with pytest.raises(HarborLoadError):
            load_harbor_tasks(cfg)

    def test_load_filters_leave_nothing_raises(self, tmp_path):
        from src.api.harbor_loader import HarborLoadError, load_harbor_tasks
        root = tmp_path / "tasks"
        _make_task_dir(root, "org", "only_task")
        cfg = _make_harbor_cfg(tmp_path, harbor_tasks_root=str(root))
        cfg.dataset.harbor_include = ["nonexistent/*"]
        with pytest.raises(HarborLoadError, match="no tasks to run"):
            load_harbor_tasks(cfg)

    def test_resolve_task_dir_returns_path(self, tmp_path):
        from src.api.harbor_loader import (
            TASK_PATH_INDEX,
            load_harbor_tasks,
            resolve_task_dir,
        )
        root = tmp_path / "tasks"
        td = _make_task_dir(root, "org", "t1")
        cfg = _make_harbor_cfg(tmp_path, harbor_tasks_root=str(root))
        load_harbor_tasks(cfg)
        assert resolve_task_dir("org/t1") == td

    def test_resolve_task_dir_returns_none_for_unknown(self):
        from src.api.harbor_loader import resolve_task_dir
        assert resolve_task_dir("nonexistent/task") is None

    def test_load_relative_path_resolved_to_project_root(self, tmp_path):
        from src.api.harbor_loader import load_harbor_tasks
        root = tmp_path / "rel_tasks"
        _make_task_dir(root, "org", "t1")
        cfg = _make_harbor_cfg(tmp_path, harbor_tasks_root="rel_tasks")
        df = load_harbor_tasks(cfg)
        assert len(df) == 1


# ── harbor_scorer ────────────────────────────────────────────────────────────


class TestHarborScorer:

    def test_valid_envelope(self):
        from src.evaluation.harbor_scorer import harbor_reward_scorer
        envelope = json.dumps({"reward": 0.75, "task_id": "org/t1", "exit_status": "verified"})
        assert harbor_reward_scorer("q", envelope, "gt") == 0.75

    def test_perfect_score(self):
        from src.evaluation.harbor_scorer import harbor_reward_scorer
        envelope = json.dumps({"reward": 1.0})
        assert harbor_reward_scorer("q", envelope, "gt") == 1.0

    def test_zero_score(self):
        from src.evaluation.harbor_scorer import harbor_reward_scorer
        envelope = json.dumps({"reward": 0.0})
        assert harbor_reward_scorer("q", envelope, "gt") == 0.0

    def test_empty_prediction_returns_zero(self):
        from src.evaluation.harbor_scorer import harbor_reward_scorer
        assert harbor_reward_scorer("q", "", "gt") == 0.0

    def test_none_prediction_returns_zero(self):
        from src.evaluation.harbor_scorer import harbor_reward_scorer
        assert harbor_reward_scorer("q", None, "gt") == 0.0

    def test_invalid_json_returns_zero(self):
        from src.evaluation.harbor_scorer import harbor_reward_scorer
        assert harbor_reward_scorer("q", "not json", "gt") == 0.0

    def test_missing_reward_key_returns_zero(self):
        from src.evaluation.harbor_scorer import harbor_reward_scorer
        envelope = json.dumps({"task_id": "org/t1"})
        assert harbor_reward_scorer("q", envelope, "gt") == 0.0

    def test_non_numeric_reward_returns_zero(self):
        from src.evaluation.harbor_scorer import harbor_reward_scorer
        envelope = json.dumps({"reward": "not_a_number"})
        assert harbor_reward_scorer("q", envelope, "gt") == 0.0

    def test_json_list_returns_zero(self):
        from src.evaluation.harbor_scorer import harbor_reward_scorer
        assert harbor_reward_scorer("q", "[1, 2, 3]", "gt") == 0.0

    def test_ground_truth_is_ignored(self):
        from src.evaluation.harbor_scorer import harbor_reward_scorer
        envelope = json.dumps({"reward": 0.5})
        assert harbor_reward_scorer("q", envelope, "anything") == 0.5
        assert harbor_reward_scorer("q", envelope, "") == 0.5


# ── HarborAgent ──────────────────────────────────────────────────────────────


class TestHarborAgent:

    def _make_agent(self, tmp_path, **kwargs):
        from src.harness.harbor import HarborAgent
        defaults = dict(
            project_root=tmp_path,
            skills_source_dir=tmp_path / "skills",
            inner_agent="claude-code",
            inner_model="anthropic/claude-sonnet-4-5",
        )
        defaults.update(kwargs)
        return HarborAgent(**defaults)

    def test_default_jobs_dir(self, tmp_path):
        agent = self._make_agent(tmp_path)
        assert agent.jobs_dir == tmp_path / ".evoskill" / "harbor_jobs"

    def test_custom_jobs_dir(self, tmp_path):
        custom = tmp_path / "my_jobs"
        agent = self._make_agent(tmp_path, jobs_dir=custom)
        assert agent.jobs_dir == custom

    def test_n_concurrent_minimum_is_one(self, tmp_path):
        agent = self._make_agent(tmp_path, n_concurrent=0)
        assert agent.n_concurrent == 1

    def test_build_command_basic(self, tmp_path):
        agent = self._make_agent(tmp_path)
        task_dir = tmp_path / "task"
        task_dir.mkdir()
        cmd = agent._build_command(task_dir, "job-1")
        assert cmd[0] == "harbor"
        assert cmd[1] == "run"
        assert "-p" in cmd
        assert str(task_dir) in cmd
        assert "-a" in cmd
        idx = cmd.index("-a")
        assert cmd[idx + 1] == "claude-code"

    def test_build_command_includes_model(self, tmp_path):
        agent = self._make_agent(tmp_path)
        cmd = agent._build_command(tmp_path, "j1")
        idx = cmd.index("-m")
        assert cmd[idx + 1] == "anthropic/claude-sonnet-4-5"

    def test_build_command_includes_extra_args(self, tmp_path):
        agent = self._make_agent(tmp_path, extra_args=["--verbose", "--debug"])
        cmd = agent._build_command(tmp_path, "j1")
        assert "--verbose" in cmd
        assert "--debug" in cmd

    def test_build_skills_mount_returns_none_if_no_dir(self, tmp_path):
        agent = self._make_agent(tmp_path, skills_source_dir=tmp_path / "nodir")
        assert agent._build_skills_mount() is None

    def test_build_skills_mount_returns_none_if_empty(self, tmp_path):
        skills = tmp_path / "skills"
        skills.mkdir()
        agent = self._make_agent(tmp_path, skills_source_dir=skills)
        assert agent._build_skills_mount() is None

    def test_build_skills_mount_returns_dict_with_content(self, tmp_path):
        skills = tmp_path / "skills"
        skills.mkdir()
        (skills / "my_skill.md").write_text("skill content")
        agent = self._make_agent(tmp_path, skills_source_dir=skills)
        mount = agent._build_skills_mount()
        assert mount is not None
        assert mount["type"] == "bind"
        assert mount["target"] == "/skills"
        assert mount["read_only"] is True

    def test_build_command_includes_mounts_when_skills_exist(self, tmp_path):
        skills = tmp_path / "skills"
        skills.mkdir()
        (skills / "s.md").write_text("x")
        agent = self._make_agent(tmp_path, skills_source_dir=skills)
        cmd = agent._build_command(tmp_path, "j1")
        assert "--mounts-json" in cmd
        assert "--ak" in cmd

    def test_read_reward_from_result_json(self, tmp_path):
        agent = self._make_agent(tmp_path)
        job_dir = tmp_path / "job"
        job_dir.mkdir()
        result = {"verifier_result": {"rewards": {"reward": 0.8}}}
        (job_dir / "result.json").write_text(json.dumps(result))
        reward, status = agent._read_reward(job_dir)
        assert reward == 0.8
        assert status == "verified"

    def test_read_reward_from_reward_txt(self, tmp_path):
        agent = self._make_agent(tmp_path)
        job_dir = tmp_path / "job"
        verifier = job_dir / "verifier"
        verifier.mkdir(parents=True)
        (verifier / "reward.txt").write_text("0.5")
        reward, status = agent._read_reward(job_dir)
        assert reward == 0.5
        assert status == "verified"

    def test_read_reward_from_reward_json(self, tmp_path):
        agent = self._make_agent(tmp_path)
        job_dir = tmp_path / "job"
        verifier = job_dir / "verifier"
        verifier.mkdir(parents=True)
        (verifier / "reward.json").write_text(json.dumps({"score": 0.9}))
        reward, status = agent._read_reward(job_dir)
        assert reward == 0.9
        assert status == "verified"

    def test_read_reward_returns_none_for_missing_dir(self, tmp_path):
        agent = self._make_agent(tmp_path)
        reward, status = agent._read_reward(tmp_path / "nonexistent")
        assert reward is None
        assert status == "no_job_dir"

    def test_read_reward_returns_none_for_empty_dir(self, tmp_path):
        agent = self._make_agent(tmp_path)
        job_dir = tmp_path / "empty_job"
        job_dir.mkdir()
        reward, status = agent._read_reward(job_dir)
        assert reward is None
        assert status == "no_reward"

    def test_run_unknown_task_returns_error_trace(self, tmp_path):
        agent = self._make_agent(tmp_path)
        with patch("src.api.harbor_loader.TASK_PATH_INDEX", {}):
            trace = asyncio.run(agent.run("nonexistent/task"))
        assert trace.is_error
        assert "unknown harbor task id" in trace.result

    def test_run_missing_task_toml_returns_error_trace(self, tmp_path):
        from src.api.harbor_loader import TASK_PATH_INDEX
        agent = self._make_agent(tmp_path)
        empty_dir = tmp_path / "no_toml"
        empty_dir.mkdir()
        with patch.dict(TASK_PATH_INDEX, {"org/t1": empty_dir}, clear=True):
            trace = asyncio.run(agent.run("org/t1"))
        assert trace.is_error
        assert "no task.toml" in trace.result

    def test_error_trace_has_zero_reward_envelope(self, tmp_path):
        agent = self._make_agent(tmp_path)
        with patch("src.api.harbor_loader.TASK_PATH_INDEX", {}):
            trace = asyncio.run(agent.run("bad/task"))
        envelope = json.loads(trace.output.final_answer)
        assert envelope["reward"] == 0.0
        assert envelope["exit_status"] == "error"


# ── Config validation ────────────────────────────────────────────────────────


class TestHarborConfigValidation:

    def test_harbor_tasks_root_path_absolute(self, tmp_path):
        cfg = _make_harbor_cfg(tmp_path, harbor_tasks_root="/absolute/path")
        assert cfg.harbor_tasks_root_path == Path("/absolute/path")

    def test_harbor_tasks_root_path_relative(self, tmp_path):
        cfg = _make_harbor_cfg(tmp_path, harbor_tasks_root="rel/tasks")
        assert cfg.harbor_tasks_root_path == tmp_path / "rel/tasks"

    def test_harbor_tasks_root_path_docker_override(self, tmp_path):
        cfg = _make_harbor_cfg(tmp_path, harbor_tasks_root="local/tasks")
        overrides = json.dumps({"harbor_tasks_root": "/mnt/harbor_tasks"})
        with patch.dict("os.environ", {"EVOSKILL_PATH_OVERRIDES": overrides}):
            assert cfg.harbor_tasks_root_path == Path("/mnt/harbor_tasks")

    def test_make_scorer_returns_harbor_scorer(self, tmp_path):
        from src.cli.shared import make_scorer
        cfg = _make_harbor_cfg(tmp_path)
        scorer = make_scorer(cfg)
        from src.evaluation.harbor_scorer import harbor_reward_scorer
        assert scorer is harbor_reward_scorer


# ── Docker launcher harbor support ───────────────────────────────────────────


class TestDockerHarbor:

    def test_compose_skips_csv_mount_for_harbor(self, tmp_path):
        from src.docker.launcher import _build_compose
        cfg = _make_harbor_cfg(tmp_path, harbor_tasks_root=str(tmp_path / "tasks"))
        (tmp_path / ".evoskill").mkdir(exist_ok=True)
        compose = _build_compose(cfg, [])
        volumes = compose["services"]["evoskill"]["volumes"]
        assert not any("/mnt/dataset" in v for v in volumes)

    def test_compose_mounts_external_harbor_tasks(self, tmp_path):
        from src.docker.launcher import _build_compose
        project = tmp_path / "project"
        project.mkdir()
        (project / ".evoskill").mkdir()
        ext_tasks = tmp_path / "external_tasks"
        ext_tasks.mkdir()
        cfg = _make_harbor_cfg(project, harbor_tasks_root=str(ext_tasks))
        compose = _build_compose(cfg, [])
        volumes = compose["services"]["evoskill"]["volumes"]
        assert any("/mnt/harbor_tasks" in v for v in volumes)

    def test_compose_sets_harbor_path_override(self, tmp_path):
        from src.docker.launcher import _build_compose
        project = tmp_path / "project"
        project.mkdir()
        (project / ".evoskill").mkdir()
        ext_tasks = tmp_path / "external_tasks"
        ext_tasks.mkdir()
        cfg = _make_harbor_cfg(project, harbor_tasks_root=str(ext_tasks))
        compose = _build_compose(cfg, [])
        env = compose["services"]["evoskill"]["env_with_values"]
        override_str = [e for e in env if "EVOSKILL_PATH_OVERRIDES" in e]
        assert override_str
        overrides = json.loads(override_str[0].split("=", 1)[1])
        assert overrides["harbor_tasks_root"] == "/mnt/harbor_tasks"

    def test_compose_no_mount_for_internal_harbor_tasks(self, tmp_path):
        from src.docker.launcher import _build_compose
        (tmp_path / ".evoskill").mkdir(exist_ok=True)
        tasks = tmp_path / "tasks"
        tasks.mkdir()
        cfg = _make_harbor_cfg(tmp_path, harbor_tasks_root=str(tasks))
        compose = _build_compose(cfg, [])
        volumes = compose["services"]["evoskill"]["volumes"]
        assert not any("/mnt/harbor_tasks" in v for v in volumes)


# ── Init harbor → harness mapping ────────────────────────────────────────────


class TestInitHarborMapping:

    def test_harness_to_harbor_agent_mapping(self):
        from src.cli.commands.init import _HARNESS_TO_HARBOR_AGENT
        assert _HARNESS_TO_HARBOR_AGENT["claude"] == "claude-code"
        assert _HARNESS_TO_HARBOR_AGENT["opencode"] == "opencode"
        assert _HARNESS_TO_HARBOR_AGENT["codex"] == "codex"
        assert _HARNESS_TO_HARBOR_AGENT["goose"] == "goose"
        assert _HARNESS_TO_HARBOR_AGENT["openhands"] == "openhands"

    def test_write_config_harbor_auto_derives_inner_agent(self, tmp_path):
        from src.cli.commands.init import _write_config
        import tomllib
        path = tmp_path / "config.toml"
        _write_config(path, {
            "harness": "claude",
            "dataset_source": "harbor",
            "dataset_path": "",
            "question_col": "",
            "gt_col": "",
            "category_col": "",
            "data_dirs": [],
            "execution": "local",
            "remote": None,
            "harbor_tasks_root": "/some/path",
        })
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        assert raw["harbor"]["inner_agent"] == "claude-code"
        assert raw["harbor"]["enabled"] is True
        assert raw["scorer"]["type"] == "harbor"
        assert raw["dataset"]["source"] == "harbor"

    def test_write_config_harbor_env_daytona(self, tmp_path):
        from src.cli.commands.init import _write_config
        import tomllib
        path = tmp_path / "config.toml"
        _write_config(path, {
            "harness": "opencode",
            "dataset_source": "harbor",
            "dataset_path": "",
            "question_col": "",
            "gt_col": "",
            "category_col": "",
            "data_dirs": [],
            "execution": "daytona",
            "remote": None,
            "harbor_tasks_root": "/tasks",
        })
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        assert raw["harbor"]["env"] == "daytona"
        assert raw["harbor"]["inner_agent"] == "opencode"

    def test_write_config_csv_mode_no_harbor_section(self, tmp_path):
        from src.cli.commands.init import _write_config
        import tomllib
        path = tmp_path / "config.toml"
        _write_config(path, {
            "harness": "claude",
            "dataset_source": "csv",
            "dataset_path": "/data.csv",
            "question_col": "q",
            "gt_col": "a",
            "category_col": "cat",
            "data_dirs": [],
            "execution": "local",
            "remote": None,
            "harbor_tasks_root": "",
        })
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        assert "harbor" not in raw
        assert raw["dataset"]["source"] == "csv"
