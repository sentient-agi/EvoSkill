"""Self-improving agent loop module.

This module provides a modular, composable interface for running
self-improving agent loops with git-based versioning.

Example usage:
    from src.loop import SelfImprovingLoop, LoopConfig, LoopAgents
    from src.harness import Agent
    from src.agent_profiles import base_agent_options, proposer_options
    from src.registry import ProgramManager

    agents = LoopAgents(
        base=Agent(base_agent_options, AgentResponse),
        proposer=Agent(proposer_options, ProposerResponse),
        skill_generator=Agent(skill_generator_options, ToolGeneratorResponse),
        prompt_generator=Agent(prompt_generator_options, PromptGeneratorResponse),
    )
    manager = ProgramManager(cwd=get_project_root())
    train_data = [(q, a) for q, a in my_train_set]
    val_data = [(q, a) for q, a in my_val_set]

    config = LoopConfig(max_iterations=10, frontier_size=5)
    loop = SelfImprovingLoop(config, agents, manager, train_data, val_data)
    result = await loop.run()
"""

from .config import LoopConfig
from .runner import SelfImprovingLoop, LoopAgents, LoopResult

__all__ = ["SelfImprovingLoop", "LoopConfig", "LoopAgents", "LoopResult"]
