from __future__ import annotations

import sys
import tomllib
import types
from pathlib import Path

import pytest


def _write_project(tmp_path: Path, config_text: str) -> Path:
    project_root = tmp_path / "project"
    evoskill_dir = project_root / ".evoskill"
    evoskill_dir.mkdir(parents=True)
    (evoskill_dir / "config.toml").write_text(config_text)
    (evoskill_dir / "task.md").write_text("# Task\n\nAnswer questions.\n\n---\n\n# Constraints\n\n- no web\n")
    return project_root


@pytest.mark.parametrize(
    ("harness", "expected"),
    [
        ("claude", "anthropic/claude-sonnet-4-6"),
        ("opencode", "anthropic/claude-sonnet-4-6"),
        ("goose", "anthropic/claude-sonnet-4-6"),
        ("openhands", "anthropic/claude-sonnet-4-6"),
        ("codex", "gpt-5.1-codex-mini"),
    ],
)
def test_default_model_for_harness(harness: str, expected: str) -> None:
    from src.harness.model_aliases import default_model_for_harness

    assert default_model_for_harness(harness) == expected


@pytest.mark.parametrize("harness", ["claude", "opencode", "goose", "openhands"])
def test_load_config_normalizes_legacy_sonnet_model(tmp_path: Path, harness: str) -> None:
    from src.cli.config import load_config

    project_root = _write_project(
        tmp_path,
        f'[harness]\nname = "{harness}"\nmodel = "sonnet"\n',
    )

    cfg = load_config(project_root)

    assert cfg.harness.model == "anthropic/claude-sonnet-4-6"


@pytest.mark.parametrize(
    ("harness", "expected"),
    [
        ("claude", "anthropic/claude-sonnet-4-6"),
        ("goose", "anthropic/claude-sonnet-4-6"),
        ("codex", "gpt-5.1-codex-mini"),
    ],
)
def test_load_config_applies_harness_default_when_model_missing(
    tmp_path: Path,
    harness: str,
    expected: str,
) -> None:
    from src.cli.config import load_config

    project_root = _write_project(
        tmp_path,
        f'[harness]\nname = "{harness}"\n',
    )

    cfg = load_config(project_root)

    assert cfg.harness.model == expected


def test_load_config_reads_timeout_and_retry_settings(tmp_path: Path) -> None:
    from src.cli.config import load_config

    project_root = _write_project(
        tmp_path,
        (
            '[harness]\n'
            'name = "codex"\n'
            'timeout_seconds = 42\n'
            'max_retries = 1\n'
        ),
    )

    cfg = load_config(project_root)

    assert cfg.harness.timeout_seconds == 42
    assert cfg.harness.max_retries == 1


def test_dataset_path_resolves_relative_to_project_root(tmp_path: Path) -> None:
    from src.cli.config import load_config

    project_root = _write_project(
        tmp_path,
        (
            "[dataset]\n"
            'path = "data/questions.csv"\n'
        ),
    )

    cfg = load_config(project_root)

    assert cfg.dataset_path == project_root / "data/questions.csv"


def test_dataset_path_preserves_absolute_path(tmp_path: Path) -> None:
    from src.cli.config import load_config

    absolute_path = tmp_path / "external" / "questions.csv"
    project_root = _write_project(
        tmp_path,
        (
            "[dataset]\n"
            f'path = "{absolute_path}"\n'
        ),
    )

    cfg = load_config(project_root)

    assert cfg.dataset_path == absolute_path


def test_load_config_reads_explicit_config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.cli.config import load_config

    project_root = _write_project(
        tmp_path,
        '[harness]\nname = "claude"\n',
    )
    alternate_config = project_root / ".evoskill" / "config.openrouter.toml"
    alternate_config.write_text(
        (
            '[harness]\n'
            'name = "opencode"\n'
            'model = "openrouter/openai/gpt-5-mini"\n'
            '\n'
            '[dataset]\n'
            'path = "data/questions.csv"\n'
        )
    )
    monkeypatch.chdir(project_root)

    cfg = load_config(config_path=Path(".evoskill/config.openrouter.toml"))

    assert cfg.harness.name == "opencode"
    assert cfg.harness.model == "openrouter/openai/gpt-5-mini"
    assert cfg.dataset_path == project_root / "data/questions.csv"


def test_load_config_explicit_config_path_preserves_project_root(tmp_path: Path) -> None:
    from src.cli.config import load_config

    project_root = _write_project(tmp_path, '[harness]\nname = "claude"\n')
    alternate_config = project_root / ".evoskill" / "config.codex.toml"
    alternate_config.write_text('[harness]\nname = "codex"\n')

    cfg = load_config(config_path=alternate_config)

    assert cfg.project_root == project_root
    assert cfg.harness.name == "codex"
    assert cfg.harness.model == "gpt-5.1-codex-mini"


def test_load_config_explicit_config_requires_evoskill_root(tmp_path: Path) -> None:
    from src.cli.config import load_config

    config_path = tmp_path / "config.toml"
    config_path.write_text('[harness]\nname = "claude"\n')

    with pytest.raises(SystemExit):
        load_config(config_path=config_path)


def test_officeqa_openrouter_config_loads() -> None:
    from src.cli.config import load_config

    example_root = Path(__file__).resolve().parents[1] / "examples" / "officeqa"
    config_path = example_root / ".evoskill" / "config.openrouter.toml"

    cfg = load_config(config_path=config_path)

    assert cfg.project_root == example_root
    assert cfg.harness.name == "opencode"
    assert cfg.harness.model == "openrouter/openai/gpt-5-mini"
    assert cfg.dataset_path == example_root / "data" / "officeqa_sample.csv"
    assert cfg.harness.data_dirs == ["data/treasury_bulletins"]


def test_load_config_applies_timeout_and_retry_defaults(tmp_path: Path) -> None:
    from src.cli.config import load_config

    project_root = _write_project(
        tmp_path,
        '[harness]\nname = "claude"\n',
    )

    cfg = load_config(project_root)

    assert cfg.harness.timeout_seconds == 1200
    assert cfg.harness.max_retries == 3


def test_init_write_config_uses_harness_default_model(tmp_path: Path) -> None:
    from src.cli.commands.init import _write_config

    config_path = tmp_path / "config.toml"
    _write_config(
        config_path,
        {
            "harness": "codex",
            "dataset_path": "./data/questions.csv",
            "question_col": "question",
            "gt_col": "answer",
            "category_col": "difficulty",
            "data_dirs": ["./docs"],
        },
    )

    rendered = config_path.read_text()
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    assert raw["harness"]["name"] == "codex"
    assert raw["harness"]["model"] == "gpt-5.1-codex-mini"
    assert raw["harness"]["data_dirs"] == ["./docs"]
    assert raw["harness"]["timeout_seconds"] == 1200
    assert raw["harness"]["max_retries"] == 3
    assert raw["evolution"]["mode"] == "skill_only"
    assert raw["dataset"]["ground_truth_column"] == "answer"
    assert raw["dataset"]["category_column"] == "difficulty"
    assert "# Agent runtime used to execute EvoSkill runs." in rendered
    assert "# Additional folders the agent can interact with during runs." in rendered
    assert "# What EvoSkill is allowed to optimize: skills or the base prompt." in rendered
    assert "# CSV column containing the expected answer." in rendered
    assert "# Scoring rule used to compare predictions against ground truth." in rendered


def test_init_prompt_defaults_keep_existing_category_column(tmp_path: Path) -> None:
    from src.cli.commands.init import _load_prompt_defaults

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        (
            "[harness]\n"
            'name = "goose"\n'
            'data_dirs = ["docs"]\n'
            "\n"
            "[dataset]\n"
            'path = "./data/input.csv"\n'
            'question_column = "prompt"\n'
            'ground_truth_column = "answer"\n'
            'category_column = "difficulty"\n'
        )
    )

    defaults = _load_prompt_defaults(config_path)

    assert defaults["harness"] == "goose"
    assert defaults["dataset_path"] == "./data/input.csv"
    assert defaults["question_col"] == "prompt"
    assert defaults["gt_col"] == "answer"
    assert defaults["category_col"] == "difficulty"
    assert defaults["data_dirs_raw"] == "docs"


def test_build_claudecode_options_strips_anthropic_prefix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.harness.claude.options import build_claudecode_options

    class FakeClaudeAgentOptions:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.model = None

    fake_module = types.SimpleNamespace(ClaudeAgentOptions=FakeClaudeAgentOptions)
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_module)

    options = build_claudecode_options(
        system="system",
        schema={"type": "object"},
        tools=["Read"],
        project_root=tmp_path,
        model="anthropic/claude-sonnet-4-6",
    )

    assert options.model == "claude-sonnet-4-6"
