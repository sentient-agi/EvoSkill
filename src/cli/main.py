"""EvoSkill CLI entry point."""

import os
from importlib import import_module
from pathlib import Path

import click

# Generic names commonly used in provider .env files, mapped onto the names
# the OpenAI-compatible harnesses expect. Real environment variables win.
_ENV_ALIASES = {
    "API_KEY": "OPENAI_API_KEY",
    "BASE_URL": "OPENAI_BASE_URL",
    "MODEL": "EVOSKILL_MODEL",
}


def _find_dotenv() -> Path | None:
    """Return the closest .env between cwd and the project root, if any."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
        if (parent / ".evoskill").is_dir():
            break
    return None


def load_dotenv(path: Path | None = None) -> None:
    """Load KEY=VALUE pairs from .env into os.environ without overriding."""
    dotenv_path = path or _find_dotenv()
    if dotenv_path is None:
        return
    try:
        lines = dotenv_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        if not key:
            continue
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)
        alias = _ENV_ALIASES.get(key.upper())
        if alias:
            os.environ.setdefault(alias, value)


_COMMAND_SPECS = {
    "init": ("src.cli.commands.init", "init_cmd", "Initialize a new EvoSkill project in the current directory."),
    "run": ("src.cli.commands.run", "run_cmd", "Run the self-improvement loop."),
    "eval": ("src.cli.commands.eval", "eval_cmd", "Evaluate the best skills on the validation set."),
    "skills": ("src.cli.commands.skills", "skills_cmd", "List all skills learned so far."),
    "diff": ("src.cli.commands.diff", "diff_cmd", "Diff baseline vs best, or between two specific iterations."),
    "logs": ("src.cli.commands.logs", "logs_cmd", "Show recent run history."),
    "reset": ("src.cli.commands.reset", "reset_cmd", "Delete all program branches and frontier tags for a clean slate."),
    "remote": ("src.cli.commands.remote", "remote_group", "Manage remote EvoSkill runs."),
}


class LazyGroup(click.Group):
    def list_commands(self, ctx):
        return sorted(_COMMAND_SPECS)

    def get_command(self, ctx, cmd_name):
        spec = _COMMAND_SPECS.get(cmd_name)
        if spec is None:
            return None
        module_name, attr_name, _ = spec
        module = import_module(module_name)
        return getattr(module, attr_name)

    def format_commands(self, ctx, formatter):
        rows = [(name, spec[2]) for name, spec in sorted(_COMMAND_SPECS.items())]
        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)


@click.group(cls=LazyGroup)
def cli():
    """EvoSkill CLI."""
    load_dotenv()
