"""Wire HALOAgent into EvoSkill's SelfImprovingLoop.

Uses prompt_only evolution mode — the proposer/generator analyze HALO's
failure traces and rewrite instructions.txt, which HALO picks up on
the next iteration.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any, Callable

from examples.appworld2.scripts.halo_agent import (
    HALOAgent,
    SEPARATOR,
    parse_task_id,
    read_eval_result,
)
from src.harness.agent import Agent
from src.loop import SelfImprovingLoop, LoopConfig, LoopAgents
from src.registry import ProgramManager
from src.schemas import (
    AgentResponse,
    PromptProposerResponse,
    PromptGeneratorResponse,
    SkillProposerResponse,
    ToolGeneratorResponse,
)


def model_slug(model: str) -> str:
    """Return a filesystem-safe short model label for experiment names."""
    return model.replace("/", "_").replace(":", "_").split("-")[0]


def halo_scorer(
    question: str,
    predicted: str,
    ground_truth: str,
    halo_root: Path,
    experiment_name: str,
) -> float:
    """Score using AppWorld's official evaluation (written by HALOAgent).

    HALOAgent.run() calls evaluate_dataset() in the same process as
    run_experiment(), which produces the official eval JSON. We read it here.

    Falls back to answer comparison if eval JSON is missing.
    """
    import json as _json

    task_id = ground_truth.strip()

    # Primary: read official AppWorld evaluation JSON
    eval_path = (
        halo_root / "experiments" / "outputs" / experiment_name
        / "evaluations" / f"on_only_{task_id}.json"
    )
    passed, score = read_eval_result(eval_path)
    if score > 0.0:
        return score

    # Fallback: answer comparison for when eval JSON is missing
    gt_path = halo_root / "data" / "tasks" / task_id / "ground_truth" / "answer.json"
    if not gt_path.exists():
        return 0.0

    gt_answer = _json.loads(gt_path.read_text())
    if gt_answer is None:
        return 0.0  # Side-effect task without eval JSON — can't score

    try:
        from appworld.common.evaluation import do_answers_match
        if do_answers_match(predicted, gt_answer):
            return 1.0
    except Exception:
        pass

    if isinstance(gt_answer, str) and "," in gt_answer:
        gt_items = sorted(s.strip().lower() for s in gt_answer.split(",") if s.strip())
        pred_items = sorted(s.strip().lower() for s in predicted.split(",") if s.strip())
        if gt_items == pred_items:
            return 1.0

    if str(predicted).strip().lower() == str(gt_answer).strip().lower():
        return 1.0

    return 0.0


def make_halo_scorer(
    halo_root: Path,
    experiment_name: str,
) -> Callable[[str, str, str], float]:
    """Create a scorer closure for SelfImprovingLoop."""
    halo_root = Path(halo_root)

    def scorer(question: str, predicted: str, ground_truth: str) -> float:
        return halo_scorer(question, predicted, ground_truth, halo_root, experiment_name)

    return scorer


def build_evolution_loop(
    halo_root: str | Path,
    model: str | None = None,
    runner_provider: str | None = None,
    evolver_harness: str = "claude",
    evolver_models: list[str] | None = None,
    experiment_name: str = "evoskill_evolution",
    max_iterations: int = 5,
    frontier_size: int = 2,
    failure_samples: int = 2,
    no_improvement_limit: int = 3,
    train_pools: dict | None = None,
    val_data: list | None = None,
) -> SelfImprovingLoop:
    """Build a SelfImprovingLoop with HALOAgent as the base agent.

    Returns:
        Configured SelfImprovingLoop ready to run.
    """
    from examples.appworld2.scripts.build_config import get_default_model, get_default_experiment_name
    from src.harness import set_sdk
    halo_root = Path(halo_root)
    model = model or get_default_model()
    set_sdk(evolver_harness)

    # Config: prompt_only mode, concurrency=1
    config = LoopConfig(
        max_iterations=max_iterations,
        frontier_size=frontier_size,
        concurrency=1,  # AppWorld tasks are stateful
        failure_sample_count=failure_samples,
        no_improvement_limit=no_improvement_limit,
        evolution_mode="prompt_only",
    )

    # Base agent: HALOAgent wrapping HALO's runner
    base_agent = HALOAgent(
        halo_root=halo_root,
        model=model,
        provider=runner_provider,
        experiment_name=experiment_name,
    )

    # Evolution agents: EvoSkill's prompt proposer/generator
    from src.agent_profiles.prompt_proposer import make_prompt_proposer_options
    from src.agent_profiles.prompt_generator import make_prompt_generator_options
    from src.agent_profiles.skill_proposer import make_skill_proposer_options
    from src.agent_profiles.skill_generator import make_skill_generator_options

    def cycling_options(factory):
        models = list(evolver_models or [])
        index = 0

        def options():
            nonlocal index
            selected_model = None
            if models:
                selected_model = models[index % len(models)]
                index += 1
            return factory(project_root=str(halo_root), model=selected_model)

        return options

    agents = LoopAgents(
        base=base_agent,
        skill_proposer=Agent(cycling_options(make_skill_proposer_options), SkillProposerResponse),
        prompt_proposer=Agent(cycling_options(make_prompt_proposer_options), PromptProposerResponse),
        skill_generator=Agent(cycling_options(make_skill_generator_options), ToolGeneratorResponse),
        prompt_generator=Agent(cycling_options(make_prompt_generator_options), PromptGeneratorResponse),
    )

    # Program manager — lives in appworld2/ (its own git repo, not HALO's).
    # Init git at runtime if it doesn't exist (the repo is not shipped with
    # the source — it's created on first run for ProgramManager's versioning).
    appworld2_root = Path(__file__).resolve().parent.parent
    if not (appworld2_root / ".git").exists():
        import subprocess
        subprocess.run(["git", "init"], cwd=appworld2_root, capture_output=True)
        subprocess.run(["git", "add", "-A"], cwd=appworld2_root, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial: EvoSkill AppWorld2 setup"],
            cwd=appworld2_root, capture_output=True,
        )
    manager = ProgramManager(cwd=appworld2_root)

    # Scorer reads from HALO's eval output
    scorer = make_halo_scorer(halo_root, experiment_name)

    # Build the loop
    loop = SelfImprovingLoop(
        config=config,
        agents=agents,
        manager=manager,
        train_pools=train_pools or {},
        val_data=val_data or [],
        scorer=scorer,
    )

    # Prompt path inside .claude/ so ProgramManager's git tracks it.
    loop._prompt_path = appworld2_root / ".claude" / "prompts" / "instructions.txt"

    # Move feedback and checkpoint paths inside .claude/ so they survive
    # git branch switches (ProgramManager stages .claude/ on commit).
    loop._feedback_path = appworld2_root / ".claude" / "feedback_history.md"
    loop._checkpoint_path = appworld2_root / ".claude" / "loop_checkpoint.json"
    for p in [loop._feedback_path, loop._checkpoint_path]:
        p.parent.mkdir(parents=True, exist_ok=True)
    if not loop._feedback_path.exists():
        loop._feedback_path.touch()

    # Pre-create base program with prompt as plain text string.
    from src.registry.models import ProgramConfig
    prompt_text = loop._prompt_path.read_text()
    base_config = ProgramConfig(
        name="base",
        system_prompt=prompt_text,
        allowed_tools=[],
        output_format={},
    )
    if "base" not in manager.list_programs():
        manager.create_program("base", base_config)

    return loop


async def main(
    halo_root: str | None = None,
    model: str | None = None,
    runner_provider: str | None = None,
    evolver_harness: str = "claude",
    evolver_models: list[str] | None = None,
    dataset: str = "dev",
    max_iterations: int = 5,
    n_train: int = 20,
    n_val: int = 17,
) -> None:
    """Run the evolution loop."""
    from examples.appworld2.scripts.build_config import get_appworld_root, get_default_model
    halo_path = Path(halo_root) if halo_root else get_appworld_root()
    model = model or get_default_model()
    os.environ["APPWORLD_ROOT"] = str(halo_path)

    # Load task IDs and split into train/val
    from appworld.task import load_task_ids
    all_ids = load_task_ids(dataset)

    train_ids = all_ids[:n_train]
    val_ids = all_ids[n_train : n_train + n_val]

    # Build train_pools and val_data in EvoSkill's expected format
    # Read instructions from specs.json for each task
    import json

    train_pools: dict[str, list[tuple[str, str]]] = {"default": []}
    for tid in train_ids:
        specs_path = halo_path / "data" / "tasks" / tid / "specs.json"
        instruction = json.loads(specs_path.read_text())["instruction"]
        question = f"{tid}{SEPARATOR}{instruction}"
        train_pools["default"].append((question, tid))  # ground_truth = task_id

    val_data: list[tuple[str, str, str]] = []
    for tid in val_ids:
        specs_path = halo_path / "data" / "tasks" / tid / "specs.json"
        instruction = json.loads(specs_path.read_text())["instruction"]
        question = f"{tid}{SEPARATOR}{instruction}"
        val_data.append((question, tid, "default"))

    print(f"=== EvoSkill Evolution on AppWorld ===")
    print(f"Runner model: {model}")
    if runner_provider:
        print(f"Runner provider: {runner_provider}")
    print(f"Evolver harness: {evolver_harness}")
    if evolver_models:
        print(f"Evolver models: {', '.join(evolver_models)}")
    print(f"Train: {len(train_ids)} tasks, Val: {len(val_ids)} tasks")
    print(f"Max iterations: {max_iterations}")
    print()

    loop = build_evolution_loop(
        halo_root=halo_path,
        model=model,
        runner_provider=runner_provider,
        evolver_harness=evolver_harness,
        evolver_models=evolver_models,
        experiment_name=f"evoskill_{model_slug(model)}_{dataset}",
        max_iterations=max_iterations,
        train_pools=train_pools,
        val_data=val_data,
    )

    result = await loop.run()

    print(f"\n{'='*60}")
    print(f"Best: {result.best_program} ({result.best_score:.2%})")
    print(f"Frontier: {result.frontier}")
    print(f"Total cost: ${result.total_cost_usd:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EvoSkill evolution on AppWorld via HALO")
    parser.add_argument("--halo-root", default=None, help="AppWorld root (default: from config.json)")
    parser.add_argument("--model", default=None, help="AppWorld runner model (default: from config.json)")
    parser.add_argument("--runner-provider", default=None, help="Provider for the AppWorld runner, e.g. openrouter")
    parser.add_argument("--evolver-harness", default="claude", help="EvoSkill harness for proposer/generator agents")
    parser.add_argument(
        "--evolver-model",
        action="append",
        dest="evolver_models",
        help="Evolver model to use. Repeat to cycle models in order.",
    )
    parser.add_argument("--dataset", default="dev")
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--n-train", type=int, default=20)
    parser.add_argument("--n-val", type=int, default=17)
    args = parser.parse_args()

    asyncio.run(main(
        halo_root=args.halo_root,
        model=args.model,
        runner_provider=args.runner_provider,
        evolver_harness=args.evolver_harness,
        evolver_models=args.evolver_models,
        dataset=args.dataset,
        max_iterations=args.max_iterations,
        n_train=args.n_train,
        n_val=args.n_val,
    ))
