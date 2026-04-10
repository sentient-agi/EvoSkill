"""EvoSkill CLI entry point."""

from __future__ import annotations

from importlib import import_module

import click

_COMMANDS = {
    "diff": {
        "module": "src.cli.commands.diff",
        "attr": "diff_cmd",
        "help": "Diff baseline vs best, or between two specific iterations.",
    },
    "eval": {
        "module": "src.cli.commands.eval",
        "attr": "eval_cmd",
        "help": "Evaluate the best skills on the validation set.",
    },
    "init": {
        "module": "src.cli.commands.init",
        "attr": "init_cmd",
        "help": "Initialize a new EvoSkill project in the current directory.",
    },
    "logs": {
        "module": "src.cli.commands.logs",
        "attr": "logs_cmd",
        "help": "Show recent run history.",
    },
    "reset": {
        "module": "src.cli.commands.reset",
        "attr": "reset_cmd",
        "help": "Delete all program branches and frontier tags for a clean slate.",
    },
    "run": {
        "module": "src.cli.commands.run",
        "attr": "run_cmd",
        "help": "Run the self-improvement loop.",
    },
    "skills": {
        "module": "src.cli.commands.skills",
        "attr": "skills_cmd",
        "help": "List all skills learned so far.",
    },
}


class LazyGroup(click.Group):
    """Load command modules only when they are actually requested."""

    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted(_COMMANDS)

    def get_command(self, ctx: click.Context, name: str) -> click.Command | None:
        target = _COMMANDS.get(name)
        if target is None:
            return None

        module = import_module(target["module"])
        return getattr(module, target["attr"])

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        rows = [
            (name, _COMMANDS[name]["help"])
            for name in self.list_commands(ctx)
        ]
        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)


@click.group(cls=LazyGroup)
def cli() -> None:
    """Run EvoSkill commands."""
