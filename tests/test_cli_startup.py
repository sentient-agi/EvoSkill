from __future__ import annotations

import importlib
import sys
import tomllib
from pathlib import Path

from click.testing import CliRunner


def test_pyproject_declares_evoskill_console_script() -> None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())

    assert data["project"]["scripts"]["evoskill"] == "src.cli.main:cli"


def test_cli_import_does_not_import_task_registry() -> None:
    for module_name in ("src", "src.api", "src.api.task_registry", "src.cli.main"):
        sys.modules.pop(module_name, None)

    importlib.import_module("src.cli.main")

    assert "src.api.task_registry" not in sys.modules


def test_cli_help_does_not_import_heavy_runtime_modules() -> None:
    heavy_modules = (
        "src.cli.commands.run",
        "src.cli.commands.init",
        "src.cli.commands.eval",
        "src.api.task_registry",
        "pandas",
        "questionary",
        "claude_agent_sdk",
        "openhands",
        "mcp",
    )
    for module_name in ("src", "src.api", "src.cli.main", *heavy_modules):
        sys.modules.pop(module_name, None)

    cli = importlib.import_module("src.cli.main").cli
    result = CliRunner().invoke(cli, ["--help"])

    assert result.exit_code == 0
    for module_name in heavy_modules:
        assert module_name not in sys.modules
