"""evoskill run — run the self-improvement loop with rich terminal output."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from src.cli.config import load_config
from src.cli.report import RunReport, SkillEntry

console = Console()


# ── display helpers ──────────────────────────────────────────────────────────

def _build_table(rows: list[dict], baseline_score: float | None) -> Table:
    table = Table(box=None, pad_edge=False, show_header=True, header_style="bold")
    table.add_column("Iter",     style="dim",   width=6)
    table.add_column("Accuracy", width=9)
    table.add_column("Δ",        width=9)
    table.add_column("Skills",   width=7)
    table.add_column("Frontier", width=18)
    table.add_column("Status")

    for row in rows:
        score_pct = f"{row['score']:.1%}"
        delta_str = "—"
        if row["delta"] is not None:
            sign = "+" if row["delta"] >= 0 else ""
            delta_str = f"{sign}{row['delta']:.1%}"

        status_text = Text(row["status"])
        if "new best" in row["status"]:
            status_text.stylize("green bold")
        elif "discarded" in row["status"]:
            status_text.stylize("dim")
        elif "baseline" in row["status"]:
            status_text.stylize("cyan")

        frontier_ids = ", ".join(str(n) for n in row["frontier_ids"])
        table.add_row(
            str(row["iter"]),
            score_pct,
            delta_str,
            str(row["n_skills"]),
            f"[{frontier_ids}]" if frontier_ids else "—",
            status_text,
        )

    if baseline_score is not None:
        best_row = max(rows, key=lambda r: r["score"]) if rows else None
        best_str = f"{best_row['score']:.1%} (iter {best_row['iter']})" if best_row else "—"
        table.caption = f"Best: {best_str}"

    return table


class LoopDisplay:
    """Manages the rich live table + inline proposer output."""

    def __init__(self, verbose: bool = False, quiet: bool = False):
        self.verbose = verbose
        self.quiet = quiet
        self.rows: list[dict] = []
        self.baseline_score: float | None = None
        self.skills_kept: list[SkillEntry] = []
        self.skills_proposed: int = 0
        self._start_time = time.time()
        self._live: Live | None = None
        self._current_iter: int = 0
        self._last_proposal: dict | None = None  # carries action/target into eval_result
        self._last_skill_written: dict | None = None  # carries actual skill name into eval_result

    def start(self) -> None:
        self._live = Live(console=console, refresh_per_second=4, transient=False)
        self._live.start()

    def stop(self) -> None:
        if self._live:
            self._live.stop()

    def _refresh(self) -> None:
        if self._live:
            self._live.update(_build_table(self.rows, self.baseline_score))

    def on_event(self, event: str, data: dict[str, Any]) -> None:
        if event == "baseline":
            self.baseline_score = data["score"]
            self.rows.append({
                "iter": 0,
                "score": data["score"],
                "delta": None,
                "n_skills": data.get("n_skills", 0),
                "frontier_ids": [1],
                "status": "baseline",
            })
            self._refresh()

        elif event == "iter_start":
            self._current_iter = data["iteration"]
            if not self.quiet:
                self._live.console.print(
                    f"\n── Iteration {data['iteration']} {'─' * 40}",
                    style="bold",
                )

        elif event == "sample":
            if not self.quiet:
                icon = "✓" if data["passed"] else "✗"
                style = "green" if data["passed"] else "red"
                self._live.console.print(
                    f"   {icon} [{data['category']}] {data['question'][:60]}",
                    style=style,
                )

        elif event == "proposal":
            self._last_proposal = data

        elif event == "skill_written":
            self._last_skill_written = data

        elif event == "eval_result":
            self.skills_proposed += 1
            prev_score = self.baseline_score or 0.0
            if self.rows:
                prev_score = self.rows[-1]["score"]

            delta = data["score"] - prev_score
            frontier_ids = [i + 1 for i, _ in enumerate(data["frontier"])]
            n_skills = data.get("n_skills", 0)

            if data["added"]:
                previous_best = max((r["score"] for r in self.rows), default=float("-inf"))
                if data["score"] > previous_best:
                    status = "★ new best"
                elif data["score"] == previous_best:
                    status = "tied best"
                else:
                    status = "kept"

                action = (self._last_proposal or {}).get("action", "create")
                target = (self._last_proposal or {}).get("target_skill") or ""

                if action == "edit" and target:
                    skill_name = target
                else:
                    # Use the actual skill name detected after generation
                    skill_name = (self._last_skill_written or {}).get("name") or data["child_name"].split("/")[-1]

                self.skills_kept.append(SkillEntry(
                    name=skill_name,
                    iteration=self._current_iter,
                    score_delta=delta,
                    action=action,
                ))
            else:
                status = "discarded"

            self._last_proposal = None
            self._last_skill_written = None

            self.rows.append({
                "iter": self._current_iter,
                "score": data["score"],
                "delta": delta,
                "n_skills": n_skills,
                "frontier_ids": frontier_ids,
                "status": status,
            })
            self._refresh()

            if not self.quiet:
                score_str = f"{data['score']:.1%}"
                sign = "+" if delta >= 0 else ""
                self._live.console.print(
                    f"  Score: {score_str}  ({sign}{delta:.1%})  → {status}",
                    style="green" if data["added"] else "dim",
                )


# ── command ──────────────────────────────────────────────────────────────────

def _get_remote_backend(cfg):
    """Return the appropriate remote backend based on config."""
    target = cfg.remote.target
    if target == "daytona":
        from src.remote.daytona import DaytonaBackend
        return DaytonaBackend()
    raise ValueError(f"Unsupported remote target: {target}")


@click.command("run")
@click.option("--continue", "continue_loop", is_flag=True, default=False,
              help="Resume from the current frontier.")
@click.option("--verbose", is_flag=True, default=False,
              help="Show full failure examples and per-sample results.")
@click.option("--quiet", is_flag=True, default=False,
              help="Show progress table only, no inline proposer output.")
@click.option("--config", "config_path", type=click.Path(dir_okay=False, path_type=Path),
              default=None, help="Load a specific config TOML file.")
@click.option("--docker", is_flag=True, default=False,
              help="Run the loop inside a Docker container.")
@click.option("--rebuild", is_flag=True, default=False,
              help="Force rebuild the Docker image before running.")
@click.option("--remote", is_flag=True, default=False,
              help="Run the loop on a remote Daytona sandbox.")
def run_cmd(continue_loop: bool, verbose: bool, quiet: bool, config_path: Path | None,
            docker: bool, rebuild: bool, remote: bool):
    """Run the self-improvement loop."""
    # Auto-select execution mode from config if no flag given.
    # Skip inside containers (EVOSKILL_REMOTE=1) to avoid recursion.
    if not docker and not remote and not os.environ.get("EVOSKILL_REMOTE"):
        from src.cli.config import load_config as _lc
        _cfg = _lc(config_path=config_path)
        if _cfg.execution == 'docker':
            docker = True
        elif _cfg.execution == 'daytona':
            remote = True

    if remote:
        cfg = load_config(config_path=config_path)
        if not cfg.remote:
            console.print("[red]Error:[/red] No [remote] section in config.toml. "
                          "Add remote config first.")
            raise SystemExit(1)

        backend = _get_remote_backend(cfg)
        console.print(f"\n  [bold]EvoSkill Remote[/bold] — {cfg.remote.target}\n")

        try:
            console.print("  [1/4] Creating sandbox...", end="")
            backend.setup(cfg)
            console.print(f" [green]done[/green]")

            dataset_path = cfg.dataset_path.resolve()
            project_root = cfg.project_root.resolve()
            external_dataset = not dataset_path.is_relative_to(project_root)
            external_dirs = [d for d in cfg.harness.data_dirs
                             if not Path(d).resolve().is_relative_to(project_root)]

            console.print("  [2/4] Uploading...")
            def _upload_log(msg):
                console.print(f"         {msg}")
            backend.upload(cfg, log=_upload_log)
            console.print(f"         [green]done[/green]")
            console.print(f"         project files → /workspace/")
            if external_dataset:
                console.print(f"         dataset ({dataset_path.name}) → /mnt/dataset/")
            if external_dirs:
                for d in external_dirs:
                    name = Path(d).name
                    console.print(f"         {name} → /mnt/data/{name}/")

            console.print("  [3/4] Installing EvoSkill...", end="")
            console.print(f" [green]done[/green]")

            extra_args = []
            if continue_loop:
                extra_args.append("--continue")
            if verbose:
                extra_args.append("--verbose")
            if quiet:
                extra_args.append("--quiet")

            console.print("  [4/4] Starting loop...", end="")
            run_info = backend.run(cfg, extra_args=extra_args or None)
            console.print(f" [green]done[/green]")

        except Exception:
            console.print(f" [red]failed[/red]")
            console.print("\n  Cleaning up sandbox...", end="")
            backend.cleanup_current(cfg)
            console.print(" [green]done[/green]\n")
            raise

        console.print(f"\n  Run: {run_info.run_id}")
        console.print(f"\n  [bold]Next steps:[/bold]")
        console.print(f"    evoskill remote status       check progress")
        console.print(f"    evoskill remote logs -f       stream live output")
        console.print(f"    evoskill remote download     pull results when done")
        console.print(f"    evoskill remote stop          cancel the run\n")
        return

    if docker:
        from src.docker.launcher import launch_docker

        cfg = load_config(config_path=config_path)
        extra_args = []
        if continue_loop:
            extra_args.append("--continue")
        if verbose:
            extra_args.append("--verbose")
        if quiet:
            extra_args.append("--quiet")
        launch_docker(cfg, extra_args=extra_args, rebuild=rebuild)
        return

    from src.harness import Agent, set_sdk
    from src.agent_profiles.base_agent.base_agent import make_base_agent_options_from_task
    from src.agent_profiles.prompt_generator.prompt_generator import (
        make_prompt_generator_options,
    )
    from src.agent_profiles.prompt_proposer.prompt_proposer import (
        make_prompt_proposer_options,
    )
    from src.agent_profiles.skill_generator.skill_generator import (
        make_skill_generator_options,
    )
    from src.agent_profiles.skill_proposer.skill_proposer import (
        make_skill_proposer_options,
    )
    from src.cli.shared import load_and_split, make_scorer
    from src.loop import LoopAgents, LoopConfig, SelfImprovingLoop
    from src.registry import ProgramManager, ProgramManagerError
    from src.schemas import (
        AgentResponse,
        PromptGeneratorResponse,
        PromptProposerResponse,
        SkillProposerResponse,
        ToolGeneratorResponse,
    )
    cfg = load_config(config_path=config_path)

    # Validate task.md
    if not cfg.task_description:
        console.print("[red]Error:[/red] .evoskill/task.md has no Task section. Fill it in first.")
        raise SystemExit(1)
    if not cfg.task_constraints and not quiet:
        console.print("[yellow]Warning:[/yellow] No constraints defined in task.md — skills may be unconstrained.")

    console.print(f"\n  [bold]EvoSkill[/bold] — {cfg.evolution.mode}  |  {cfg.harness.name}  |  {cfg.evolution.iterations} iterations\n")

    if cfg.harness.name == "openhands":
        console.print("  [yellow]Warning:[/yellow] OpenHands does not support native structured output.")
        console.print("  Using fallback JSON extraction which may be less reliable.\n")

    # Map harness to sdk
    sdk = cfg.harness.name  # "claude" or "opencode"
    set_sdk(sdk)

    # Load dataset
    try:
        train_pools, val_data = load_and_split(cfg)
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] Dataset not found at {cfg.dataset_path}")
        raise SystemExit(1)

    console.print(f"  Dataset: {cfg.dataset_path}  ({len(val_data)} val samples)\n")

    # Build agents — use task.md description as the base agent prompt
    base_factory = make_base_agent_options_from_task(
        cfg.task_description,
        model=cfg.harness.model,
        data_dirs=cfg.harness.data_dirs,
        project_root=cfg.project_root,
    )
    agents = LoopAgents(
        base=Agent(
            base_factory,
            AgentResponse,
            timeout_seconds=cfg.harness.timeout_seconds,
            max_retries=cfg.harness.max_retries,
        ),
        skill_proposer=Agent(
            make_skill_proposer_options(
                project_root=cfg.project_root,
                model=cfg.harness.model,
            ),
            SkillProposerResponse,
            timeout_seconds=cfg.harness.timeout_seconds,
            max_retries=cfg.harness.max_retries,
        ),
        prompt_proposer=Agent(
            make_prompt_proposer_options(
                project_root=cfg.project_root,
                model=cfg.harness.model,
            ),
            PromptProposerResponse,
            timeout_seconds=cfg.harness.timeout_seconds,
            max_retries=cfg.harness.max_retries,
        ),
        skill_generator=Agent(
            make_skill_generator_options(
                project_root=cfg.project_root,
                model=cfg.harness.model,
            ),
            ToolGeneratorResponse,
            timeout_seconds=cfg.harness.timeout_seconds,
            max_retries=cfg.harness.max_retries,
        ),
        prompt_generator=Agent(
            make_prompt_generator_options(
                project_root=cfg.project_root,
                model=cfg.harness.model,
            ),
            PromptGeneratorResponse,
            timeout_seconds=cfg.harness.timeout_seconds,
            max_retries=cfg.harness.max_retries,
        ),
    )
    manager = ProgramManager(cwd=cfg.project_root)

    loop_config = LoopConfig(
        max_iterations=cfg.evolution.iterations,
        frontier_size=cfg.evolution.frontier_size,
        no_improvement_limit=cfg.evolution.no_improvement_limit,
        concurrency=cfg.evolution.concurrency,
        evolution_mode=cfg.evolution.mode,
        failure_sample_count=cfg.evolution.failure_samples,
        categories_per_batch=cfg.evolution.failure_samples,
        continue_mode=continue_loop,
    )

    display = LoopDisplay(verbose=verbose, quiet=quiet)
    display.start()

    try:
        loop = SelfImprovingLoop(
            loop_config, agents, manager, train_pools, val_data,
            scorer=make_scorer(cfg),
            on_event=display.on_event,
            task_constraints=cfg.task_constraints,
        )
        result = asyncio.run(loop.run())
    except ProgramManagerError as exc:
        console.print(f"\n[red]Error:[/red] {exc}\n")
        raise SystemExit(1) from exc
    finally:
        display.stop()

    # Build and print report
    report = RunReport(
        baseline_score=display.baseline_score or 0.0,
        final_score=result.best_score,
        iterations_completed=result.iterations_completed,
        best_program=result.best_program,
        rows=display.rows,
        skills_kept=display.skills_kept,
        skills_proposed=display.skills_proposed,
        project_root=cfg.project_root,
        total_cost_usd=result.total_cost_usd,
    )
    report.print_summary()

    report_path = report.save()
    console.print(f"  Full report: {report_path}\n")
