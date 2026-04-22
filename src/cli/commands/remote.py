"""evoskill remote — status, logs, stop, download for remote runs."""

from __future__ import annotations

import click
from rich.console import Console

from src.cli.config import load_config
from src.remote.base import RunInfo

console = Console()


def _get_remote_backend(cfg):
    """Return the appropriate remote backend based on config."""
    target = cfg.remote.target
    if target == "daytona":
        from src.remote.daytona import DaytonaBackend
        return DaytonaBackend()
    raise ValueError(f"Unsupported remote target: {target}")


def _load_run_info(cfg) -> RunInfo | None:
    return RunInfo.load(cfg.project_root)


@click.command("status")
def remote_status():
    """Check the status of a remote run."""
    cfg = load_config()
    run_info = _load_run_info(cfg)
    if not run_info:
        console.print("  No active remote run found.")
        console.print("  Start one with: evoskill run --remote")
        return

    backend = _get_remote_backend(cfg)
    status = backend.status(cfg, run_info)

    console.print(f"\n  Run:     {run_info.run_id}")
    console.print(f"  Target:  {run_info.target}")
    console.print(f"  Started: {run_info.started_at}")
    console.print(f"  Status:  {status}")

    if "completed" in status.lower():
        console.print(f"\n  [bold]Next:[/bold]")
        console.print(f"    evoskill remote download     pull results to local")
        console.print(f"    evoskill remote logs          see full run output")
        console.print(f"    evoskill remote stop          clean up sandbox\n")
    elif "failed" in status.lower() or "error" in status.lower():
        console.print(f"\n  [bold]Debug:[/bold]")
        console.print(f"    evoskill remote logs          see error traceback")
        console.print(f"    evoskill remote stop          clean up sandbox\n")
    elif "running" in status.lower():
        console.print(f"\n  [bold]Next:[/bold]")
        console.print(f"    evoskill remote logs -f       stream live output")
        console.print(f"    evoskill remote stop           cancel the run\n")


@click.command("logs")
@click.option("--follow", "-f", is_flag=True, default=False,
              help="Follow log output.")
@click.option("--tail", "-n", default=0, type=int,
              help="Show last N lines (default: all).")
def remote_logs(follow: bool, tail: int):
    """Stream logs from a remote run."""
    cfg = load_config()
    run_info = _load_run_info(cfg)
    if not run_info:
        console.print("  No active remote run found.")
        return

    backend = _get_remote_backend(cfg)
    lines = list(backend.logs(cfg, run_info, follow=follow))

    if tail > 0:
        lines = lines[-tail:]

    for line in lines:
        console.print(line)


@click.command("stop")
def remote_stop():
    """Stop a running remote job."""
    cfg = load_config()
    run_info = _load_run_info(cfg)
    if not run_info:
        console.print("  No active remote run found.")
        return

    backend = _get_remote_backend(cfg)
    backend.stop(cfg, run_info)
    RunInfo.clear(cfg.project_root)
    console.print(f"  Stopped and cleaned up: {run_info.run_id}")


@click.command("download")
def remote_download():
    """Download results from a completed remote run."""
    cfg = load_config()
    run_info = _load_run_info(cfg)
    if not run_info:
        console.print("  No active remote run found.")
        return

    backend = _get_remote_backend(cfg)

    console.print("  Downloading results...", end="")
    backend.download(cfg, run_info)
    console.print(" [green]done[/green]")

    console.print(f"\n  [bold]Downloaded:[/bold]")
    console.print(f"    .claude/skills/          best skill set")
    console.print(f"    .claude/program.yaml     best program config + score")
    if cfg.remote.download.reports:
        console.print(f"    .evoskill/reports/        run summary")
    if cfg.remote.download.all_branches:
        console.print(f"    git branches             all program/* branches")
    if cfg.remote.download.cache:
        console.print(f"    .cache/runs/             evaluation cache")
    if cfg.remote.download.feedback_history:
        console.print(f"    feedback_history.md       for local continuation")

    console.print(f"\n  [bold]Next:[/bold]")
    console.print(f"    evoskill run              continue evolving locally")
    console.print(f"    evoskill skills           see discovered skills")
    console.print(f"    evoskill remote stop      clean up remote sandbox\n")


@click.group("remote")
def remote_group():
    """Manage remote EvoSkill runs."""


remote_group.add_command(remote_status)
remote_group.add_command(remote_logs)
remote_group.add_command(remote_stop)
remote_group.add_command(remote_download)
