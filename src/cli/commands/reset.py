"""evoskill reset — wipe all program branches and start fresh."""

import shutil
from pathlib import Path

import click
from rich.console import Console

from src.agent_profiles.skill_generator import get_project_root
from src.registry import ProgramManager

console = Console()


@click.command("reset")
@click.option("--yes", is_flag=True, default=False, help="Skip confirmation prompt.")
def reset_cmd(yes: bool):
    """Delete all program branches and frontier tags for a clean slate."""
    project_root = Path(get_project_root())
    manager = ProgramManager(cwd=project_root)
    evoskill_dir = project_root / ".evoskill"

    # Show what will be deleted
    branches = [b for b in manager._git_list_branches() if b.startswith(manager.BRANCH_PREFIX)]
    tags = [t for t in manager._git_list_tags() if t.startswith(manager.FRONTIER_PREFIX)]
    has_evoskill_dir = evoskill_dir.exists()

    if not branches and not tags and not has_evoskill_dir:
        console.print("  Nothing to reset — no program branches, frontier tags, or .evoskill directory found.")
        return

    console.print(f"\n  This will permanently delete:")
    console.print(f"    [red]{len(branches)}[/red] program branches: {', '.join(branches[:5])}{'...' if len(branches) > 5 else ''}")
    if tags:
        console.print(f"    [red]{len(tags)}[/red] frontier tags: {', '.join(tags)}")
    console.print("    loop checkpoint + feedback history (if present)")
    if has_evoskill_dir:
        console.print("    [red].evoskill/[/red] directory\n")
    else:
        console.print("")

    if not yes:
        click.confirm("  Proceed?", abort=True)

    stats = manager.reset_all()
    removed_evoskill_dir = False
    if evoskill_dir.exists():
        shutil.rmtree(evoskill_dir)
        removed_evoskill_dir = True

    console.print(
        f"\n  [green]Reset complete.[/green]  "
        f"Deleted {stats['branches']} branch(es), "
        f"{stats['tags']} tag(s), "
        f"{stats['files']} state file(s)"
    )
    if removed_evoskill_dir:
        console.print("  Removed .evoskill directory.\n")
    else:
        console.print("")
