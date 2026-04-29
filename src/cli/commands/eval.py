"""evoskill eval — evaluate the best skills on the validation set."""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.command('eval')
@click.option('--verbose', is_flag=True, default=False, help='Show per-question results.')
@click.option('--config', 'config_path', type=click.Path(dir_okay=False, path_type=Path),
              default=None, help='Load a specific config TOML file.')
def eval_cmd(verbose: bool, config_path: Path | None):
    """Evaluate the best skills on the validation set."""
    from src.harness import Agent, set_sdk
    from src.agent_profiles.base_agent.base_agent import make_base_agent_options
    from src.cli.config import load_config
    from src.cli.shared import load_and_split, make_scorer
    from src.evaluation import evaluate_agent_parallel
    from src.registry import ProgramManager
    from src.schemas import AgentResponse
    cfg = load_config(config_path=config_path)
    sdk = cfg.harness.name
    set_sdk(sdk)

    if sdk == "openhands":
        console.print("  [yellow]Warning:[/yellow] OpenHands does not support native structured output.")
        console.print("  Using fallback JSON extraction which may be less reliable.\n")

    try:
        _, val_data = load_and_split(cfg)
    except FileNotFoundError:
        console.print(f'[red]Error:[/red] Dataset not found at {cfg.dataset_path}')
        raise SystemExit(1)

    manager = ProgramManager(cwd=cfg.project_root)
    best = manager.get_best_from_frontier()
    if best:
        manager.switch_to(best)
    else:
        console.print('\n  No frontier found — evaluating base program.\n')
        best = 'base'

    console.print(f'\n  Evaluating [bold]{best}[/bold] on {len(val_data)} samples...\n')

    agent = Agent(
        make_base_agent_options(
            model=cfg.harness.model,
            data_dirs=cfg.harness.data_dirs,
            project_root=cfg.project_root,
        ),
        AgentResponse,
        timeout_seconds=cfg.harness.timeout_seconds,
        max_retries=cfg.harness.max_retries,
    )
    scorer = make_scorer(cfg)

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
