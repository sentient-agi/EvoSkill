"""Agent profile configurations — what each agent role does.

Each subdirectory defines a specific agent role (system prompt, tools, schema).
Harness/SDK logic lives in src.harness, not here.
"""

from .proposer import proposer_options
from .skill_generator import skill_generator_options
from .solver import solver_options, make_solver_options
from .officeqa_agent import officeqa_agent_options, make_officeqa_agent_options
from .dabstep_agent import dabstep_agent_options, make_dabstep_agent_options
from .sealqa_agent import sealqa_agent_options, make_sealqa_agent_options
from .livecodebench_agent import (
    livecodebench_agent_options,
    make_livecodebench_agent_options,
)
from .prompt_generator import prompt_generator_options
from .skill_proposer import skill_proposer_options
from .prompt_proposer import prompt_proposer_options
from .skill_evolver import skill_evolver_options, make_skill_evolver_options

__all__ = [
    "proposer_options",
    "skill_generator_options",
    "solver_options",
    "make_solver_options",
    "officeqa_agent_options",
    "make_officeqa_agent_options",
    "dabstep_agent_options",
    "make_dabstep_agent_options",
    "sealqa_agent_options",
    "make_sealqa_agent_options",
    "livecodebench_agent_options",
    "make_livecodebench_agent_options",
    "prompt_generator_options",
    "skill_proposer_options",
    "prompt_proposer_options",
    "skill_evolver_options",
    "make_skill_evolver_options",
]
