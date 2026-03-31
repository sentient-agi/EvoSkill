"""evoskill run — run the self-improvement loop with rich terminal output."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import click
import pandas as pd
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from src.cli.config import load_config, ProjectConfig
from src.cli.report import RunReport, SkillEntry
from src.agent_profiles import (
    Agent,
    base_agent_options,
    skill_proposer_options,
    prompt_proposer_options,
    skill_generator_options,
    prompt_generator_options,
    set_sdk,
)
from src.agent_profiles.base_agent.base_agent import make_base_agent_options_from_task
from src.agent_profiles.skill_generator import get_project_root
from src.loop import SelfImprovingLoop, LoopConfig, LoopAgents
from src.registry import ProgramManager
from src.schemas import (
    AgentResponse,
    SkillProposerResponse,
    PromptProposerResponse,
    ToolGeneratorResponse,
    PromptGeneratorResponse,
)

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
                is_best = data["score"] >= max(r["score"] for r in self.rows)
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


# ── dataset helpers ──────────────────────────────────────────────────────────

def _load_and_split(cfg: ProjectConfig):
    from src.api.data_utils import stratified_split

    data = pd.read_csv(cfg.dataset_path)

    renames: dict[str, str] = {}
    if cfg.dataset.question_column != "question":
        renames[cfg.dataset.question_column] = "question"
    if cfg.dataset.ground_truth_column != "ground_truth":
        renames[cfg.dataset.ground_truth_column] = "ground_truth"
    if renames:
        data.rename(columns=renames, inplace=True)

    # Map category column: explicit config takes priority, then fall back to "default"
    if cfg.dataset.category_column and cfg.dataset.category_column in data.columns:
        if cfg.dataset.category_column != "category":
            data.rename(columns={cfg.dataset.category_column: "category"}, inplace=True)
    elif "category" not in data.columns:
        data["category"] = "default"

    return stratified_split(
        data,
        train_ratio=cfg.dataset.train_ratio,
        val_ratio=cfg.dataset.val_ratio,
    )


def _infer_provider(model: str) -> str:
    """Infer the LLM provider from the model name."""
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        return "openai"
    if model.startswith("gemini"):
        return "google"
    return "anthropic"  # fallback


async def _call_llm(provider: str, model: str, prompt: str) -> str:
    """Call the appropriate LLM provider and return the raw text response."""
    if provider == "anthropic":
        import anthropic
        client = anthropic.AsyncAnthropic()
        response = await client.messages.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    if provider == "openai":
        try:
            import openai
        except ImportError:
            raise RuntimeError("openai package not installed. Run: uv add openai")
        client = openai.AsyncOpenAI()
        response = await client.chat.completions.create(
            model=model,
            max_tokens=16,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    if provider == "google":
        try:
            from google import genai
        except ImportError:
            raise RuntimeError("google-genai package not installed. Run: uv add google-genai")
        client = genai.Client()
        response = await client.aio.models.generate_content(model=model, contents=prompt)
        return response.text

    raise ValueError(f"Unknown provider: {provider}")


def _make_scorer(cfg: ProjectConfig):
    from src.loop.runner import _score_multi_tolerance

    if cfg.scorer.type == "exact":
        def exact(question: str, predicted: str, ground_truth: str) -> float:
            return 1.0 if predicted.strip().lower() == ground_truth.strip().lower() else 0.0
        return exact

    if cfg.scorer.type == "multi_tolerance":
        return _score_multi_tolerance

    if cfg.scorer.type == "llm":
        import asyncio

        rubric = cfg.scorer.rubric or "Award 1.0 if correct, 0.0 if wrong."
        model = cfg.scorer.model or "claude-sonnet-4-6"
        provider = cfg.scorer.provider or _infer_provider(model)

        async def _llm_score(question: str, predicted: str, ground_truth: str) -> float:
            prompt = (
                f"Question: {question}\n"
                f"Expected: {ground_truth}\n"
                f"Got: {predicted}\n\n"
                f"Rubric: {rubric}\n\n"
                "Reply with only a number between 0.0 and 1.0."
            )
            try:
                text = await _call_llm(provider, model, prompt)
                return float(text.strip())
            except (ValueError, Exception):
                return 0.0

        def llm_scorer(question: str, predicted: str, ground_truth: str) -> float:
            return asyncio.get_event_loop().run_until_complete(
                _llm_score(question, predicted, ground_truth)
            )
        return llm_scorer

    if cfg.scorer.type == "script":
        import subprocess, shlex

        def script_scorer(question: str, predicted: str, ground_truth: str) -> float:
            cmd = cfg.scorer.command.format(predicted=predicted, expected=ground_truth)
            result = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
            try:
                return float(result.stdout.strip())
            except ValueError:
                return 0.0
        return script_scorer

    # fallback
    return _score_multi_tolerance


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
    cfg = load_config()

    # Validate task.md
    if not cfg.task_description:
        console.print("[red]Error:[/red] .evoskill/task.md has no Task section. Fill it in first.")
        raise SystemExit(1)
    if not cfg.task_constraints and not quiet:
        console.print("[yellow]Warning:[/yellow] No constraints defined in task.md — skills may be unconstrained.")

    console.print(f"\n  [bold]EvoSkill[/bold] — {cfg.evolution.mode}  |  {cfg.harness.name}  |  {cfg.evolution.iterations} iterations\n")

    # Map harness to sdk
    sdk = cfg.harness.name  # "claude" or "opencode"
    set_sdk(sdk)

    # Load dataset
    try:
        train_pools, val_data = _load_and_split(cfg)
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] Dataset not found at {cfg.dataset_path}")
        raise SystemExit(1)

    console.print(f"  Dataset: {cfg.dataset_path}  ({len(val_data)} val samples)\n")

    # Build agents — use task.md description as the base agent prompt
    base_factory = make_base_agent_options_from_task(
        cfg.task_description, model=cfg.harness.model, data_dirs=cfg.harness.data_dirs
    )
    agents = LoopAgents(
        base=Agent(base_factory, AgentResponse),
        skill_proposer=Agent(skill_proposer_options, SkillProposerResponse),
        prompt_proposer=Agent(prompt_proposer_options, PromptProposerResponse),
        skill_generator=Agent(skill_generator_options, ToolGeneratorResponse),
        prompt_generator=Agent(prompt_generator_options, PromptGeneratorResponse),
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
    )

    display = LoopDisplay(verbose=verbose, quiet=quiet)
    display.start()

    try:
        loop = SelfImprovingLoop(
            loop_config, agents, manager, train_pools, val_data,
            scorer=_make_scorer(cfg),
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
    )
    report.print_summary()

    report_path = report.save()
    console.print(f"  Full report: {report_path}\n")
