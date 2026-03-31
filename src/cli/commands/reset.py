"""evoskill reset — wipe all program branches and start fresh."""

import click
from rich.console import Console

from src.cli.config import load_config
from src.agent_profiles.skill_generator import get_project_root
from src.registry import ProgramManager

console = Console()


@click.command("reset")
@click.option("--yes", is_flag=True, default=False, help="Skip confirmation prompt.")
def reset_cmd(yes: bool):
    """Delete all program branches and frontier tags for a clean slate."""
    cfg = load_config()
    manager = ProgramManager(cwd=get_project_root())

    # Show what will be deleted
    branches = [b for b in manager._git_list_branches() if b.startswith(manager.BRANCH_PREFIX)]
    tags = [t for t in manager._git_list_tags() if t.startswith(manager.FRONTIER_PREFIX)]

    if not branches and not tags:
        console.print("  Nothing to reset — no program branches or frontier tags found.")
        return

    console.print(f"\n  This will permanently delete:")
    console.print(f"    [red]{len(branches)}[/red] program branches: {', '.join(branches[:5])}{'...' if len(branches) > 5 else ''}")
    if tags:
        console.print(f"    [red]{len(tags)}[/red] frontier tags: {', '.join(tags)}")
    console.print(f"    loop checkpoint + feedback history (if present)\n")

    if not yes:
        click.confirm("  Proceed?", abort=True)

    stats = manager.reset_all()

    console.print(
        f"\n  [green]Reset complete.[/green]  "
        f"Deleted {stats['branches']} branch(es), "
        f"{stats['tags']} tag(s), "
        f"{stats['files']} state file(s).\n"
    )
