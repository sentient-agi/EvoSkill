"""Baseline run: evaluate AppWorld tasks through HALOAgent.

Uses the exact same code path as SelfImprovingLoop — HALOAgent.run()
for execution + in-process AppWorld evaluation for scoring.

Usage:
    # Run all 57 dev tasks
    python -m examples.appworld2.scripts.run_baseline

    # Run first 10 tasks
    python -m examples.appworld2.scripts.run_baseline --n-tasks 10

    # Run a specific task
    python -m examples.appworld2.scripts.run_baseline --task-id 50e1ac9_1

    # Run multiple specific tasks
    python -m examples.appworld2.scripts.run_baseline --task-id 50e1ac9_1 --task-id 530b157_1

    # Run on test_normal split
    python -m examples.appworld2.scripts.run_baseline --dataset test_normal --n-tasks 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Ensure EvoSkill root is on path BEFORE importing our modules
EVOSKILL_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(EVOSKILL_ROOT))

# Set APPWORLD_ROOT early so appworld imports can find data
APPWORLD2_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("APPWORLD_ROOT", str(APPWORLD2_ROOT))

from examples.appworld2.scripts.halo_agent import HALOAgent, SEPARATOR, read_eval_result
from examples.appworld2.scripts.build_config import get_appworld_root
from appworld.task import load_task_ids


async def run_baseline(
    n_tasks: int | None = None,
    task_ids_override: list[str] | None = None,
    model: str | None = None,
    provider: str | None = None,
    dataset: str = "dev",
    experiment_name: str | None = None,
) -> None:

    from examples.appworld2.scripts.build_config import get_default_model, get_default_experiment_name
    appworld_root = get_appworld_root()
    model = model or get_default_model()
    experiment_name = experiment_name or f"baseline_{model}_{dataset}"

    # Load task IDs
    os.environ["APPWORLD_ROOT"] = str(appworld_root)

    # Determine which tasks to run
    if task_ids_override:
        task_ids = task_ids_override
        total_in_split = "?"
    else:
        all_task_ids = load_task_ids(dataset)
        total_in_split = str(len(all_task_ids))
        if n_tasks is not None:
            task_ids = all_task_ids[:n_tasks]
        else:
            task_ids = all_task_ids  # run all

    print(f"=== Baseline Run: {len(task_ids)}/{total_in_split} tasks ===")
    print(f"Model: {model}")
    print(f"Experiment: {experiment_name}")
    print(f"AppWorld root: {appworld_root}")
    print()

    # Create HALOAgent — same as SelfImprovingLoop uses
    agent = HALOAgent(
        halo_root=appworld_root,
        model=model,
        provider=provider,
        experiment_name=experiment_name,
    )

    results = []
    start_time = time.time()

    for i, task_id in enumerate(task_ids):
        task_start = time.time()

        # Read instruction
        specs_path = appworld_root / "data" / "tasks" / task_id / "specs.json"
        if not specs_path.exists():
            print(f"\n[{i+1}/{len(task_ids)}] Task: {task_id} — SKIPPED (specs.json not found)")
            results.append({
                "task_id": task_id, "passed": False, "score": 0.0,
                "answer": None, "turns": 0, "time_s": 0,
            })
            continue

        specs = json.loads(specs_path.read_text())
        instruction = specs["instruction"]
        question = f"{task_id}{SEPARATOR}{instruction}"

        print(f"\n[{i+1}/{len(task_ids)}] Task: {task_id}")
        print(f"  Instruction: {instruction[:60]}...")

        try:
            # Same call SelfImprovingLoop makes
            trace = await agent.run(question)

            answer = trace.output.final_answer if trace.output else None
            passed = not trace.is_error
            score = 1.0 if passed else 0.0

            # Read official eval for accurate score
            eval_path = (
                appworld_root / "experiments" / "outputs" / experiment_name
                / "evaluations" / f"on_only_{task_id}.json"
            )
            if eval_path.exists():
                eval_passed, eval_score = read_eval_result(eval_path)
                passed = eval_passed
                score = eval_score

            elapsed = time.time() - task_start
            status = "PASS" if passed else "FAIL"
            answer_short = str(answer)[:40] if answer else "NONE"
            print(f"  {status} — {answer_short} (score={score:.1f}, {elapsed:.0f}s)")

            results.append({
                "task_id": task_id,
                "passed": passed,
                "score": score,
                "answer": answer,
                "turns": trace.num_turns,
                "time_s": elapsed,
            })

        except Exception as e:
            elapsed = time.time() - task_start
            print(f"  ERROR: {e} ({elapsed:.0f}s)")
            results.append({
                "task_id": task_id,
                "passed": False,
                "score": 0.0,
                "answer": None,
                "turns": 0,
                "time_s": elapsed,
            })

    # Aggregate evaluation — scenario_goal_completion across all tasks.
    # Run as subprocess to avoid SQLite threading conflicts from HALOAgent's
    # ThreadPoolExecutor leaving stale connections.
    scenario_metrics = {}
    if not task_ids_override:
        import subprocess as _sp
        print(f"\nRunning aggregate evaluation on {dataset}...")
        eval_result = _sp.run(
            [sys.executable, "-c", (
                "import os, json; "
                f"os.environ.setdefault('APPWORLD_ROOT', {repr(str(appworld_root))}); "
                "from appworld.evaluator import evaluate_dataset; "
                f"d = evaluate_dataset(experiment_name={repr(experiment_name)}, "
                f"dataset_name={repr(dataset)}, suppress_errors=True); "
                "print('EVAL_JSON:' + json.dumps(d.get('aggregate', {}) if isinstance(d, dict) else {}))"
            )],
            capture_output=True, text=True,
        )
        print(eval_result.stdout)
        if eval_result.stderr:
            print(eval_result.stderr[-500:])
        for line in eval_result.stdout.splitlines():
            if line.startswith("EVAL_JSON:"):
                scenario_metrics = json.loads(line.removeprefix("EVAL_JSON:"))
                break

    # Summary
    total_time = time.time() - start_time
    passed = sum(1 for r in results if r["passed"])

    print(f"\n{'='*60}")
    print(f"RESULTS: {experiment_name} ({len(results)} tasks)")
    print(f"{'='*60}")
    print(f"Passed:      {passed}/{len(results)} ({100*passed/len(results):.1f}%)")
    if scenario_metrics:
        tgc = scenario_metrics.get("task_goal_completion", "N/A")
        sgc = scenario_metrics.get("scenario_goal_completion", "N/A")
        print(f"Task goal completion:     {tgc}")
        print(f"Scenario goal completion: {sgc}")
    print(f"Total time:  {total_time/60:.1f} min")
    print(f"Avg time:    {total_time/len(results):.0f}s/task")
    print()
    print("Per-task:")
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  {r['task_id']}: {status} (score={r['score']:.1f}, turns={r['turns']})")

    # Save results
    results_path = Path("examples/appworld2/data")
    results_path.mkdir(parents=True, exist_ok=True)
    suffix = "_".join(task_ids_override) if task_ids_override else f"{dataset}_{len(task_ids)}"
    results_file = results_path / f"{experiment_name}_{suffix}.json"
    with open(results_file, "w") as f:
        json.dump({
            "results": results,
            "passed": passed,
            "total": len(results),
            "accuracy": passed / len(results) if results else 0,
            "model": model,
            "dataset": dataset,
            "experiment_name": experiment_name,
            "scenario_metrics": scenario_metrics,
        }, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AppWorld baseline via HALOAgent")
    parser.add_argument("--n-tasks", type=int, default=None, help="Number of tasks (default: all in split)")
    parser.add_argument("--task-id", action="append", dest="task_ids", help="Specific task ID(s) to run (repeatable)")
    parser.add_argument("--model", default=None, help="Model (default: from config.json)")
    parser.add_argument("--runner-provider", default=None, help="Provider for the model, e.g. openrouter")
    parser.add_argument("--dataset", default="dev", help="Dataset split")
    parser.add_argument("--experiment-name", default=None, help="Experiment name (default: baseline_{model}_{dataset})")
    args = parser.parse_args()

    asyncio.run(run_baseline(
        n_tasks=args.n_tasks,
        task_ids_override=args.task_ids,
        model=args.model,
        provider=args.runner_provider,
        dataset=args.dataset,
        experiment_name=args.experiment_name,
    ))
