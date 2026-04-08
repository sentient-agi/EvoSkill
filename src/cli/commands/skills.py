"""evoskill skills — list all learned skills."""

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from src.cli.config import load_config

console = Console()


def _get_skills_dir(cfg) -> Path:
    """Get the skills directory based on the configured harness."""
    harness = cfg.harness.name
    if harness == "openhands":
        return cfg.project_root / ".agents" / "skills"
    elif harness == "opencode":
        return cfg.project_root / ".opencode" / "skills"
    else:
        return cfg.project_root / ".claude" / "skills"


@click.command('skills')
def skills_cmd():
    """List all skills learned so far."""
    cfg = load_config()
    skills_dir = _get_skills_dir(cfg)

    if not skills_dir.exists() or not any(skills_dir.iterdir()):
        console.print('  No skills yet. Run [bold]evoskill run[/bold] first.')
        return

    table = Table(box=None, pad_edge=False, show_header=True, header_style='bold')
    table.add_column('Skill', min_width=30)
    table.add_column('Size', width=8)
    table.add_column('Preview')

    for skill_dir in sorted(skills_dir.iterdir()):
        if not skill_dir.is_dir():
            continue
        skill_file = skill_dir / 'SKILL.md'
        if not skill_file.exists():
            continue
        content = skill_file.read_text().strip()
        size = f'{len(content):,}b'
        preview = next(
            (ln.strip() for ln in content.splitlines() if ln.strip() and not ln.strip().startswith('#')),
            '',
        )[:70]
        table.add_row(skill_dir.name, size, preview)

    console.print(table)
