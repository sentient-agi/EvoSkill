"""evoskill logs — show run history."""

from pathlib import Path

import click
from rich.console import Console

from src.cli.config import load_config

console = Console()


def _parse_summary(text: str) -> tuple[str, str, str]:
    """Extract baseline, final, improvement from a report markdown."""
    baseline = final = improvement = '?'
    for line in text.splitlines():
        if '| Baseline |' in line:
            baseline = line.split('|')[2].strip()
        elif '| Final |' in line:
            final = line.split('|')[2].strip()
        elif '| Improvement |' in line:
            improvement = line.split('|')[2].strip()
    return baseline, final, improvement


@click.command('logs')
@click.option('--last', default=5, show_default=True, help='Number of recent runs to show.')
def logs_cmd(last: int):
    """Show recent run history."""
    cfg = load_config()
    reports_dir = cfg.evoskill_dir / 'reports'

    if not reports_dir.exists():
        console.print('  No runs yet. Run [bold]evoskill run[/bold] first.')
        return

    reports = sorted(reports_dir.glob('run-*.md'), reverse=True)
    if not reports:
        console.print('  No reports found.')
        return

    for report_path in reports[:last]:
        text = report_path.read_text()
        baseline, final, improvement = _parse_summary(text)
        console.print(f'  [bold]{report_path.stem}[/bold]  {baseline} → {final}  ({improvement})')
