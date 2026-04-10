"""evoskill run — run the self-improvement loop with rich terminal output."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

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
                "iter": 1,
                "score": data["score"],
                "delta": None,
                "n_skills": 0,
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

        elif event == "status":
            # Generic status message from harness
            if not self.quiet and self._live:
                msg = data.get("message", "")
                if msg:
                    self._live.console.print(f"  [dim]{msg}[/dim]")

        elif event == "sample_start":
            # Fired when a sample evaluation begins
            if self.verbose and self._live:
                question = data.get("question", "")
                self._live.console.print(
                    f"  [dim]→ {question[:60]}...[/dim]"
                )

        elif event == "sample":
            if self.verbose:
                icon = "✓" if data["passed"] else "✗"
                style = "green" if data["passed"] else "red"
                self._live.console.print(
                    f"  [{data['category']}] {icon} {data['question'][:60]}",
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
                is_best = not self.rows or data["score"] >= max(r["score"] for r in self.rows)
                status = "★ new best" if is_best else "kept"

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

@click.command("run")
@click.option("--continue", "continue_loop", is_flag=True, default=False,
              help="Resume from the current frontier.")
@click.option("--verbose", is_flag=True, default=False,
              help="Show full failure examples and per-sample results.")
@click.option("--quiet", is_flag=True, default=False,
              help="Show progress table only, no inline proposer output.")
def run_cmd(continue_loop: bool, verbose: bool, quiet: bool):
    """Run the self-improvement loop."""
    from src.agent_profiles import Agent
    from src.agent_profiles.prompt_generator.prompt_generator import make_prompt_generator_options
    from src.agent_profiles.prompt_proposer.prompt_proposer import make_prompt_proposer_options
    from src.agent_profiles.skill_generator import get_project_root
    from src.agent_profiles.skill_generator.skill_generator import make_skill_generator_options
    from src.agent_profiles.skill_proposer.skill_proposer import make_skill_proposer_options
    from src.cli.config import load_config
    from src.cli.shared import load_and_split, make_scorer
    from src.harness import build_base_agent_factory
    from src.loop import LoopAgents, LoopConfig, SelfImprovingLoop
    from src.registry import ProgramManager
    from src.schemas import (
        AgentResponse,
        PromptGeneratorResponse,
        PromptProposerResponse,
        SkillProposerResponse,
        ToolGeneratorResponse,
    )

    cfg = load_config()

    # Validate task.md
    if not cfg.task_description:
        console.print("[red]Error:[/red] .evoskill/task.md has no Task section. Fill it in first.")
        raise SystemExit(1)
    if not cfg.task_constraints and not quiet:
        console.print("[yellow]Warning:[/yellow] No constraints defined in task.md — skills may be unconstrained.")

    harness = cfg.harness.name  # "claude", "opencode", or "openhands"
    console.print(f"\n  [bold]EvoSkill[/bold] — {cfg.evolution.mode}  |  {harness}  |  {cfg.evolution.iterations} iterations\n")

    # Load dataset
    try:
        train_pools, val_data = load_and_split(cfg)
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] Dataset not found at {cfg.dataset_path}")
        raise SystemExit(1)

    console.print(f"  Dataset: {cfg.dataset_path}  ({len(val_data)} val samples)\n")

    # Build base agent factory through the shared harness builder
    model = cfg.harness.model
    base_factory = build_base_agent_factory(
        harness=harness,
        task_description=cfg.task_description,
        model=model,
        data_dirs=cfg.harness.data_dirs,
    )

    # Build meta-agents through harness factories
    agents = LoopAgents(
        base=Agent(base_factory, AgentResponse),
        skill_proposer=Agent(make_skill_proposer_options(harness, model), SkillProposerResponse),
        prompt_proposer=Agent(make_prompt_proposer_options(harness, model), PromptProposerResponse),
        skill_generator=Agent(make_skill_generator_options(harness, model), ToolGeneratorResponse),
        prompt_generator=Agent(make_prompt_generator_options(harness, model), PromptGeneratorResponse),
    )
    manager = ProgramManager(cwd=get_project_root())

    loop_config = LoopConfig(
        max_iterations=cfg.evolution.iterations,
        frontier_size=cfg.evolution.frontier_size,
        no_improvement_limit=cfg.evolution.no_improvement_limit,
        concurrency=cfg.evolution.concurrency,
        evolution_mode=cfg.evolution.mode,
        failure_sample_count=cfg.evolution.failure_samples,
        categories_per_batch=cfg.evolution.failure_samples,
        continue_mode=continue_loop,
        harness=harness,
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
        harness=harness,
    )
    report.print_summary()

    report_path = report.save()
    console.print(f"  Full report: {report_path}\n")
