#!/usr/bin/env python3
"""Run self-improving agent loop on OSWorld benchmark."""

import argparse
import asyncio
import sys
from pathlib import Path

from src.agent_profiles import (
    Agent,
    skill_proposer_options,
    prompt_proposer_options,
    skill_generator_options,
    prompt_generator_options,
)
from src.agent_profiles.skill_generator import get_project_root
from src.registry import ProgramManager
from src.schemas import (
    SkillProposerResponse,
    PromptProposerResponse,
    ToolGeneratorResponse,
    PromptGeneratorResponse,
)

from src.osworld.config import OSWorldLoopConfig
from src.osworld.data import load_osworld_tasks, stratified_split_tasks
from src.osworld.env_pool import EnvPool
from src.osworld.runner import OSWorldLoop, OSWorldLoopAgents, LoopResult


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run self-improving loop on OSWorld")

    # OSWorld paths
    parser.add_argument(
        "--osworld-root",
        type=str,
        required=True,
        help="Path to OSWorld project root",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to task dataset JSON (default: {osworld-root}/evaluation_examples/test_nogdrive.json)",
    )
    parser.add_argument(
        "--examples-dir",
        type=str,
        default=None,
        help="Path to examples directory (default: {osworld-root}/evaluation_examples/examples)",
    )

    # VM config
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel VMs (default: 1)")
    parser.add_argument("--provider-name", type=str, default="vmware", help="VM provider (default: vmware)")
    parser.add_argument("--path-to-vm", type=str, default=None, help="Path to VM image")
    parser.add_argument("--headless", action="store_true", help="Run VMs headless")

    # Agent config
    parser.add_argument("--agent-model", type=str, default="claude-sonnet-4-5-20250929", help="Model for base agent")
    parser.add_argument("--agent-timeout", type=int, default=1200, help="Agent timeout in seconds (default: 1200)")
    parser.add_argument("--max-steps-hint", type=int, default=15, help="Max steps hint for agent (default: 15)")

    # Task selection
    parser.add_argument("--domain", type=str, default="all", help="Domain to test (default: all)")
    parser.add_argument("--train-ratio", type=float, default=0.18, help="Train split ratio (default: 0.18)")
    parser.add_argument("--val-ratio", type=float, default=0.12, help="Validation split ratio (default: 0.12)")

    # Loop config
    parser.add_argument("--mode", type=str, choices=["skill_only", "prompt_only"], default="skill_only")
    parser.add_argument("--max-iterations", type=int, default=20, help="Max loop iterations (default: 20)")
    parser.add_argument("--frontier-size", type=int, default=3, help="Frontier size (default: 3)")
    parser.add_argument("--no-improvement-limit", type=int, default=5, help="Stop after N iterations without improvement")
    parser.add_argument("--failure-threshold", type=float, default=1.0, help="Score below which a task is a failure (default: 1.0)")
    parser.add_argument("--failure-samples", type=int, default=3, help="Samples to test per iteration (default: 3)")
    parser.add_argument("--no-reset-feedback", action="store_true", help="Don't reset feedback history")
    parser.add_argument("--continue", dest="continue_loop", action="store_true", help="Continue from existing frontier")

    # Timing
    parser.add_argument("--setup-time", type=float, default=60.0, help="Seconds to wait after env.reset() (default: 60)")
    parser.add_argument("--settle-time", type=float, default=20.0, help="Seconds to wait before env.evaluate() (default: 20)")

    return parser.parse_args()


async def main(args: argparse.Namespace):
    osworld_root = Path(args.osworld_root).resolve()
    dataset_path = args.dataset_path or str(osworld_root / "evaluation_examples" / "test_nogdrive.json")
    examples_dir = args.examples_dir or str(osworld_root / "evaluation_examples" / "examples")

    # 1. Load and split tasks
    print(f"Loading tasks from {dataset_path}...")
    tasks = load_osworld_tasks(dataset_path, examples_dir)

    if args.domain != "all":
        tasks = [t for t in tasks if t.domain == args.domain]
        if not tasks:
            print(f"No tasks found for domain '{args.domain}'")
            sys.exit(1)

    train_pools, val_data = stratified_split_tasks(
        tasks, train_ratio=args.train_ratio, val_ratio=args.val_ratio
    )

    # Print summary
    domains = list(train_pools.keys())
    total_train = sum(len(pool) for pool in train_pools.values())
    print(f"Domains ({len(domains)}): {', '.join(domains)}")
    print(f"Training pools: {', '.join(f'{d}: {len(p)}' for d, p in train_pools.items())}")
    print(f"Total training: {total_train}, Validation: {len(val_data)}")

    # 2. Create agents
    project_root = get_project_root()

    # Add OSWorld to path for imports
    if str(osworld_root) not in sys.path:
        sys.path.insert(0, str(osworld_root))

    from mm_agents.claude_code_sdk import ClaudeCodeAgent

    vm_tools_path = str(osworld_root / "mm_agents" / "claude_code_sdk" / "vm_tools.py")
    base_agent = ClaudeCodeAgent(
        model=args.agent_model,
        cwd=project_root,
        max_steps_hint=args.max_steps_hint,
        vm_tools_path=vm_tools_path,
        timeout_seconds=args.agent_timeout,
    )

    loop_agents = OSWorldLoopAgents(
        skill_proposer=Agent(skill_proposer_options, SkillProposerResponse),
        prompt_proposer=Agent(prompt_proposer_options, PromptProposerResponse),
        skill_generator=Agent(skill_generator_options, ToolGeneratorResponse),
        prompt_generator=Agent(prompt_generator_options, PromptGeneratorResponse),
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

    # 4. Create loop
    config = OSWorldLoopConfig(
        max_iterations=args.max_iterations,
        frontier_size=args.frontier_size,
        no_improvement_limit=args.no_improvement_limit,
        evolution_mode=args.mode,
        failure_sample_count=args.failure_samples,
        categories_per_batch=args.failure_samples,
        reset_feedback=not args.no_reset_feedback,
        continue_mode=args.continue_loop,
        num_envs=args.num_envs,
        provider_name=args.provider_name,
        path_to_vm=args.path_to_vm,
        headless=args.headless,
        max_steps_hint=args.max_steps_hint,
        setup_time=args.setup_time,
        settle_time=args.settle_time,
        failure_threshold=args.failure_threshold,
        agent_model=args.agent_model,
        agent_timeout=args.agent_timeout,
        osworld_root=str(osworld_root),
        dataset_path=dataset_path,
        examples_dir=examples_dir,
    )

    manager = ProgramManager(cwd=project_root)

    loop = OSWorldLoop(
        config=config,
        agents=loop_agents,
        manager=manager,
        base_agent=base_agent,
        env_pool=env_pool,
        train_pools=train_pools,
        val_data=val_data,
    )

    # 5. Run
    print(f"\nRunning loop: mode={args.mode}, envs={args.num_envs}, "
          f"model={args.agent_model}")
    try:
        result = await loop.run()
        print(f"\nBest: {result.best_program} ({result.best_score:.2%})")
        print(f"Frontier: {result.frontier}")
    finally:
        env_pool.close()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
