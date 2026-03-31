"""EvoSkill CLI entry point."""

import click

from src.cli.commands.init import init_cmd
from src.cli.commands.run import run_cmd
from src.cli.commands.eval import eval_cmd
from src.cli.commands.skills import skills_cmd
from src.cli.commands.diff import diff_cmd
from src.cli.commands.logs import logs_cmd
from src.cli.commands.reset import reset_cmd


@click.group()
def cli():
    pass


cli.add_command(init_cmd)
cli.add_command(run_cmd)
cli.add_command(eval_cmd)
cli.add_command(skills_cmd)
cli.add_command(diff_cmd)
cli.add_command(logs_cmd)
cli.add_command(reset_cmd)
