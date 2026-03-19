#!/usr/bin/env python3
"""Run baseline benchmark of ClaudeCodeAgent on OSWorld (no skill evolution)."""

import argparse
import asyncio
import datetime
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from src.agent_profiles.skill_generator import get_project_root
from src.osworld.data import load_osworld_tasks
from src.osworld.env_pool import EnvPool

logger = logging.getLogger("osworld.baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ClaudeCodeAgent baseline on OSWorld benchmark"
    )

    # OSWorld paths
    parser.add_argument(
        "--osworld-root", type=str, required=True,
        help="Path to OSWorld project root",
    )
    parser.add_argument(
        "--dataset-path", type=str, default=None,
        help="Path to task dataset JSON (default: {osworld-root}/evaluation_examples/test_nogdrive.json)",
    )
    parser.add_argument(
        "--examples-dir", type=str, default=None,
        help="Path to examples directory (default: {osworld-root}/evaluation_examples/examples)",
    )

    # VM config
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel VMs (default: 1)")
    parser.add_argument("--provider-name", type=str, default="vmware", help="VM provider (default: vmware)")
    parser.add_argument("--path-to-vm", type=str, default=None, help="Path to VM image")
    parser.add_argument("--headless", action="store_true", help="Run VMs headless")

    # Agent config
    parser.add_argument("--agent-model", type=str, default="claude-sonnet-4-5-20250929", help="Model for agent")
    parser.add_argument("--agent-timeout", type=int, default=1200, help="Agent timeout in seconds (default: 1200)")
    parser.add_argument("--max-steps-hint", type=int, default=15, help="Max steps hint for agent (default: 15)")

    # Task selection
    parser.add_argument("--domain", type=str, default="all", help="Domain to test (default: all)")
    parser.add_argument("--max-tasks", type=int, default=None, help="Limit total number of tasks (for quick tests)")
    parser.add_argument("--specific-task-id", type=str, default=None, help="Run only a specific task ID")

    # Timing
    parser.add_argument("--setup-time", type=float, default=60.0, help="Seconds to wait after env.reset() (default: 60)")
    parser.add_argument("--settle-time", type=float, default=20.0, help="Seconds to wait before env.evaluate() (default: 20)")

    # Output
    parser.add_argument("--result-dir", type=str, default="./results/baseline", help="Directory to save results")
    parser.add_argument("--clear-results", action="store_true", help="Clear existing results before running (disables resume)")

    return parser.parse_args()


def save_task_result(result_dir: Path, task: "OSWorldTask", trace_data: dict, score: float) -> dict:
    """Save a single task result immediately. Returns the result entry dict."""
    output = trace_data.get("output")
    cost = trace_data.get("total_cost_usd") or 0.0

    entry = {
        "domain": task.domain,
        "task_id": task.id,
        "score": score,
        "cost_usd": cost,
        "duration_ms": trace_data.get("duration_ms") or 0,
        "status": output.status if output else "error",
        "is_error": trace_data.get("is_error", True),
    }

    # Save individual trace
    task_dir = result_dir / task.domain / task.id
    task_dir.mkdir(parents=True, exist_ok=True)
    with open(task_dir / "result.txt", "w") as f:
        f.write(f"{score}\n")
    try:
        serializable = {
            k: v for k, v in trace_data.items()
            if k not in ("messages", "output", "raw_structured_output")
        }
        if output:
            serializable["output"] = output.model_dump()
        with open(task_dir / "trace.json", "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"Failed to save trace for {task.domain}/{task.id}: {e}")

    return entry


def load_completed_tasks(result_dir: Path) -> dict[str, dict]:
    """Load already-completed task results from result_dir. Returns {task_id: entry}."""
    completed = {}
    if not result_dir.exists():
        return completed
    for domain_dir in result_dir.iterdir():
        if not domain_dir.is_dir() or domain_dir.name.startswith("."):
            continue
        for task_dir in domain_dir.iterdir():
            result_file = task_dir / "result.txt"
            if result_file.exists():
                try:
                    score = float(result_file.read_text().strip())
                    trace_file = task_dir / "trace.json"
                    entry = {
                        "domain": domain_dir.name,
                        "task_id": task_dir.name,
                        "score": score,
                        "is_error": False,
                    }
                    if trace_file.exists():
                        trace = json.loads(trace_file.read_text(encoding="utf-8"))
                        entry["cost_usd"] = trace.get("total_cost_usd") or 0.0
                        entry["duration_ms"] = trace.get("duration_ms") or 0
                        output = trace.get("output")
                        entry["status"] = output.get("status", "unknown") if output else "unknown"
                        entry["is_error"] = trace.get("is_error", False)
                    completed[task_dir.name] = entry
                except (ValueError, json.JSONDecodeError) as e:
                    logger.warning(f"Skipping corrupt result {task_dir}: {e}")
    return completed


def print_summary(task_results: list[dict], total_cost: float, result_dir: Path, model: str):
    """Print and save the summary of all results."""
    by_domain = defaultdict(list)
    for entry in task_results:
        by_domain[entry["domain"]].append(entry["score"])

    print(f"\n{'Domain':<25} {'Score':>8} {'Pass':>6} {'Count':>6}")
    print("-" * 50)
    all_scores = []
    summary_by_domain = {}
    for domain in sorted(by_domain.keys()):
        scores = by_domain[domain]
        all_scores.extend(scores)
        mean_score = sum(scores) / len(scores)
        pass_count = sum(1 for s in scores if s >= 1.0)
        print(f"{domain:<25} {mean_score:>7.1%} {pass_count:>5}/{len(scores):<5}")
        summary_by_domain[domain] = {
            "score": round(mean_score, 4),
            "count": len(scores),
            "pass_count": pass_count,
        }

    overall = sum(all_scores) / len(all_scores) if all_scores else 0.0
    overall_pass = sum(1 for s in all_scores if s >= 1.0)
    print("-" * 50)
    print(f"{'Overall':<25} {overall:>7.1%} {overall_pass:>5}/{len(all_scores):<5}")
    print(f"\nTotal cost: ${total_cost:.2f}")

    results_json = {
        "model": model,
        "timestamp": datetime.datetime.now().isoformat(),
        "num_tasks": len(task_results),
        "overall_score": round(overall, 4),
        "overall_pass_count": overall_pass,
        "total_cost_usd": round(total_cost, 2),
        "by_domain": summary_by_domain,
        "tasks": task_results,
    }
    results_path = result_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to {results_path}")


async def main(args: argparse.Namespace):
    osworld_root = Path(args.osworld_root).resolve()
    dataset_path = args.dataset_path or str(osworld_root / "evaluation_examples" / "test_nogdrive.json")
    examples_dir = args.examples_dir or str(osworld_root / "evaluation_examples" / "examples")
    result_dir = Path(args.result_dir)
    if args.clear_results and result_dir.exists():
        import shutil
        shutil.rmtree(result_dir)
        print(f"Cleared results directory: {result_dir}")
    result_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load tasks
    print(f"Loading tasks from {dataset_path}...")
    tasks = load_osworld_tasks(dataset_path, examples_dir)

    # Filter
    if args.specific_task_id:
        tasks = [t for t in tasks if t.id == args.specific_task_id]
        if not tasks:
            print(f"Task ID '{args.specific_task_id}' not found")
            sys.exit(1)
    elif args.domain != "all":
        tasks = [t for t in tasks if t.domain == args.domain]
        if not tasks:
            print(f"No tasks found for domain '{args.domain}'")
            sys.exit(1)

    if args.max_tasks and len(tasks) > args.max_tasks:
        tasks = tasks[:args.max_tasks]

    # Check for completed tasks (resume support)
    completed = load_completed_tasks(result_dir)
    pending_tasks = [t for t in tasks if t.id not in completed]
    if completed:
        print(f"Resuming: {len(completed)} tasks already completed, {len(pending_tasks)} remaining")
    else:
        # Summarize
        domain_counts = defaultdict(int)
        for t in tasks:
            domain_counts[t.domain] += 1
        print(f"Tasks: {len(tasks)} across {len(domain_counts)} domain(s)")
        for domain, count in sorted(domain_counts.items()):
            print(f"  {domain}: {count}")

    if not pending_tasks:
        print("All tasks already completed!")
        task_results = list(completed.values())
        total_cost = sum(e.get("cost_usd", 0) for e in task_results)
        print_summary(task_results, total_cost, result_dir, args.agent_model)
        return

    # 2. Create agent
    project_root = get_project_root()

    # Add OSWorld to path for imports
    if str(osworld_root) not in sys.path:
        sys.path.insert(0, str(osworld_root))

    from mm_agents.claude_code_sdk import ClaudeCodeAgent

    vm_tools_path = str(osworld_root / "mm_agents" / "claude_code_sdk" / "vm_tools.py")
    agent = ClaudeCodeAgent(
        model=args.agent_model,
        cwd=project_root,
        max_steps_hint=args.max_steps_hint,
        vm_tools_path=vm_tools_path,
        timeout_seconds=args.agent_timeout,
    )

    # 3. Create VM pool
    env_kwargs = {
        "provider_name": args.provider_name,
        "path_to_vm": args.path_to_vm,
        "action_space": "pyautogui",
        "screen_size": (1920, 1080),
        "headless": args.headless,
        "os_type": "Ubuntu",
        "require_a11y_tree": True,
    }

    env_pool = EnvPool(
        num_envs=args.num_envs,
        env_kwargs=env_kwargs,
        osworld_root=osworld_root,
    )

    # 4. Run tasks concurrently (bounded by num_envs), saving each result immediately
    task_results = list(completed.values())
    total_cost = sum(e.get("cost_usd", 0) for e in task_results)
    results_lock = asyncio.Lock()
    completed_count = 0

    async def run_and_save(task, task_num):
        nonlocal total_cost, completed_count
        print(f"\n[{task_num}/{len(pending_tasks)}] {task.domain}/{task.id}: {task.instruction[:80]}...")
        trace_data, score = await env_pool.run_task(
            task, agent,
            settle_time=args.settle_time,
            setup_time=args.setup_time,
        )
        entry = save_task_result(result_dir, task, trace_data, score)
        async with results_lock:
            task_results.append(entry)
            total_cost += entry.get("cost_usd", 0)
            completed_count += 1
            print(f"\nCompleted {completed_count}/{len(pending_tasks)}: {task.domain}/{task.id} -> score={score:.2f}")

    print(f"\nRunning baseline: model={args.agent_model}, envs={args.num_envs}, tasks={len(pending_tasks)} pending")
    try:
        # Launch all tasks — env_pool's queue bounds concurrency to num_envs
        coros = [run_and_save(task, i + 1) for i, task in enumerate(pending_tasks)]
        await asyncio.gather(*coros)

        print_summary(task_results, total_cost, result_dir, args.agent_model)

    except KeyboardInterrupt:
        print(f"\n\nInterrupted! {len(task_results)} tasks saved so far. Re-run to resume.")

    finally:
        env_pool.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    asyncio.run(main(args))
