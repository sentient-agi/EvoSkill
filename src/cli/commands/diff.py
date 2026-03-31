"""evoskill diff — diff baseline vs best, or between two iterations."""

import subprocess

import click
from rich.console import Console
from rich.syntax import Syntax

from src.cli.config import load_config

console = Console()


def _git(*args, cwd=None) -> str:
    result = subprocess.run(['git', *args], cwd=cwd, capture_output=True, text=True)
    return result.stdout


def _best_branch(cwd=None) -> str | None:
    """Return the branch name of the best frontier member."""
    tags = _git('tag', '--list', 'frontier/*', cwd=cwd).splitlines()
    if not tags:
        return None
    best_tag = sorted(tags)[-1]
    return best_tag.replace('frontier/', 'program/')


def _resolve_branch(iteration: int, cwd=None) -> str | None:
    """Find the program branch for a given iteration number."""
    branches = _git('branch', '--list', f'program/iter-*-{iteration}', cwd=cwd).splitlines()
    for b in branches:
        b = b.strip().lstrip('* ')
        if b.startswith('program/iter-'):
            return b
    return None


@click.command('diff')
@click.argument('iter_a', required=False, type=int)
@click.argument('iter_b', required=False, type=int)
def diff_cmd(iter_a: int, iter_b: int):
    """Diff baseline vs best, or between two specific iterations.

    Examples:

      evoskill diff          # baseline vs current best

      evoskill diff 3 7      # iteration 3 vs iteration 7
    """
    cfg = load_config()
    cwd = cfg.project_root

    if iter_a is None and iter_b is None:
        branch_a = 'program/base'
        branch_b = _best_branch(cwd)
        if branch_b is None:
            console.print('  No frontier found yet. Run [bold]evoskill run[/bold] first.')
            return
        label = f'baseline → best ({branch_b})'
    else:
        if iter_b is None:
            console.print('  Provide two iteration numbers, e.g. [bold]evoskill diff 3 7[/bold]')
            return
        branch_a = _resolve_branch(iter_a, cwd)
        branch_b = _resolve_branch(iter_b, cwd)
        if branch_a is None or branch_b is None:
            console.print(f'  Could not find branches for iterations {iter_a} and {iter_b}.')
            return
        label = f'iter {iter_a} → iter {iter_b}'

    diff_output = _git('diff', f'{branch_a}..{branch_b}', '--', '.claude/', cwd=cwd).strip()

    console.print(f'\n  [bold]{label}[/bold]\n')
    if not diff_output:
        console.print('  No differences found.')
    else:
        console.print(Syntax(diff_output, 'monokai', line_numbers=False))
