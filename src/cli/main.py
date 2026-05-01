"""EvoSkill CLI entry point."""

from importlib import import_module

import click

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
