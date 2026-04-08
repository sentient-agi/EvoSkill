"""evoskill eval — evaluate the best skills on the validation set."""

from __future__ import annotations

import asyncio

import click
from rich.console import Console
from rich.table import Table

from src.cli.config import load_config
from src.cli.commands.run import _load_and_split, _make_scorer
from src.agent_profiles import Agent
from src.agent_profiles.skill_generator import get_project_root
from src.evaluation import evaluate_agent_parallel
from src.harness import build_base_agent_factory
from src.registry import ProgramManager
from src.schemas import AgentResponse

console = Console()


@click.command('eval')
@click.option('--verbose', is_flag=True, default=False, help='Show per-question results.')
def eval_cmd(verbose: bool):
    """Evaluate the best skills on the validation set."""
    cfg = load_config()

    try:
        _, val_data = _load_and_split(cfg)
    except FileNotFoundError:
        console.print(f'[red]Error:[/red] Dataset not found at {cfg.dataset_path}')
        raise SystemExit(1)

    manager = ProgramManager(cwd=get_project_root())
    best = manager.get_best_from_frontier()
    if best:
        manager.switch_to(best)
    else:
        console.print('\n  No frontier found — evaluating base program.\n')
        best = 'base'

    console.print(f'\n  Evaluating [bold]{best}[/bold] on {len(val_data)} samples...\n')

    agent = Agent(
        build_base_agent_factory(
            harness=cfg.harness.name,
            task_description=cfg.task_description,
            model=cfg.harness.model,
            data_dirs=cfg.harness.data_dirs,
        ),
        AgentResponse,
    )
    scorer = _make_scorer(cfg)

    qa_data = [(q, a) for q, a, _ in val_data]
    results = asyncio.run(
        evaluate_agent_parallel(agent, qa_data, max_concurrent=cfg.evolution.concurrency)
    )

    scores = []
    combined = []
    for r, (q, a) in zip(results, qa_data):
        got = r.trace.output.final_answer if r.trace and r.trace.output else ''
        s = scorer(q, got, a)
        scores.append(s)
        combined.append((q, a, got, s))

    accuracy = sum(1 for s in scores if s >= 0.8) / len(scores) if scores else 0.0

    if verbose:
        table = Table(box=None, pad_edge=False, show_header=True, header_style='bold')
        table.add_column('Question', max_width=50)
        table.add_column('Expected', max_width=20)
        table.add_column('Got', max_width=20)
        table.add_column('Score', width=7)
        for q, a, got, s in combined:
            style = 'green' if s >= 0.8 else 'red'
            table.add_row(q[:50], a[:20], (got or '—')[:20], f'{s:.0%}', style=style)
        console.print(table)

    n_correct = sum(1 for s in scores if s >= 0.8)
    console.print(f'  Accuracy: [bold]{accuracy:.1%}[/bold]  ({n_correct}/{len(scores)} correct)\n')
