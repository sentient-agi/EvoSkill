#!/usr/bin/env python3
"""Run self-improving agent loop on any supported benchmark."""

import asyncio
import subprocess

import pandas as pd

from src.loop import SelfImprovingLoop, GEPALoop, LoopConfig, LoopAgents
from src.agent_profiles import (
    Agent,
    set_sdk,
    make_base_agent_options,
    make_sealqa_agent_options,
    make_dabstep_agent_options,
    make_livecodebench_agent_options,
    make_gdpval_agent_options,
    make_frames_agent_options,
    skill_proposer_options,
    prompt_proposer_options,
    skill_generator_options,
    prompt_generator_options,
)
from src.agent_profiles.skill_generator import get_project_root
from src.api.data_utils import stratified_split
from src.evaluation.sealqa_scorer import score_sealqa
from src.evaluation.reward import score_answer
from src.evaluation.dabstep_scorer import question_scorer
from src.evaluation.livecodebench.livecodebench_scorer import score_livecodebench
from src.evaluation.gdpval_scorer import score_gdpval  # Note: Returns 0.0, use run_eval_comb.py for full scoring
from src.registry import ProgramManager
from src.schemas import (
    AgentResponse,
    SkillProposerResponse,
    PromptProposerResponse,
    ToolGeneratorResponse,
    PromptGeneratorResponse,
)
from scripts.load_dataset import EvalSettings, prepare_run_dir

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal, Optional


class LoopSettings(EvalSettings):
    """Settings for the self-improving loop, extending EvalSettings."""
    mode: Literal["skill_only", "prompt_only"] = Field(
        default="skill_only", description="Evolution mode",
    )
    max_iterations: int = Field(default=20, description="Max improvement iterations")
    frontier_size: int = Field(default=3, description="Number of top programs to keep")
    no_improvement_limit: int = Field(
        default=5, description="Stop after N iterations without improvement",
    )
    concurrency: int = Field(default=1, description="Concurrent evaluations")
    categories_per_batch: int = Field(
        default=2, description="Categories to sample per iteration",
    )
    samples_per_category: int = Field(
        default=1, description="Samples per category per iteration",
    )
    no_cache: bool = Field(default=False, description="Disable run caching")
    no_reset_feedback: bool = Field(
        default=False, description="Don't reset feedback history on start",
    )
    continue_loop: bool = Field(
        default=False,
        description="Continue from existing frontier instead of starting fresh",
    )
    optimizer: Literal["evoskill", "gepa"] = Field(
        default="evoskill",
        description='Optimizer: "evoskill" (proposer+generator pipeline) or "gepa" (dspy.GEPA)',
    )
    gepa_reflection_model: Optional[str] = Field(
        default=None,
        description="Model for dspy.GEPA reflection LM. Defaults to --model if not set.",
    )


# Scorer wrappers matching (question, predicted, ground_truth) -> float
def _sealqa_scorer(question: str, predicted: str, ground_truth: str) -> float:
    return score_sealqa(question, ground_truth, predicted)


def _officeqa_scorer(question: str, predicted: str, ground_truth: str) -> float:
    return float(score_answer(ground_truth, predicted))


def _dabstep_scorer(question: str, predicted: str, ground_truth: str) -> float:
    return 1.0 if question_scorer(predicted, ground_truth) else 0.0


def _livecodebench_scorer(question: str, predicted: str, ground_truth: str) -> float:
    return score_livecodebench(question, ground_truth, predicted)


def build_train_val(
    data: pd.DataFrame,
    category_col: str,
    answer_col: str,
    question_col: str,
    settings: LoopSettings,
) -> tuple[dict[str, list[tuple[str, str]]], list[tuple[str, str, str]]]:
    """Build train_pools and val_data from a DataFrame using stratified split."""
    data = data.rename(columns={category_col: "category", answer_col: "ground_truth"})
    if question_col != "question":
        data = data.rename(columns={question_col: "question"})

    train_pools, val_data, _test = stratified_split(
        data, train_ratio=settings.train_ratio, val_ratio=settings.val_ratio,
    )

    total_train = sum(len(p) for p in train_pools.values())
    print(f"Train: {total_train}, Val: {len(val_data)}, Test (held-out): {len(_test)}")
    for cat, pool in sorted(train_pools.items()):
        print(f"  {cat}: train={len(pool)}")

    return train_pools, val_data


PROMPT = """You are an expert data analyst and you will answer factoid questions by loading and referencing the files/documents listed below.
You have these files available:
{context_files}

Here is the question you need to answer:
{question}

Here are the guidelines you must follow when answering the question above:
{guidelines}
"""


async def main(settings: LoopSettings):
    set_sdk(settings.sdk)

    dataset_path = settings.dataset_path
    dataset_name = dataset_path.name
    prompt_path = (Path(get_project_root()) / "src" / "agent_profiles" / "base_agent" / "prompt.txt")

    data = pd.read_csv(dataset_path)

    if dataset_name == "seal-0.csv":
        train_pools, val_data = build_train_val(data, "topic", "answer", "question", settings)
        agent_options = make_sealqa_agent_options(model=settings.model, provider=settings.provider)
        scorer = _sealqa_scorer
        prompt_path = (Path(get_project_root()) / "src" / "agent_profiles" / "sealqa_agent" / "prompt.txt")
    elif dataset_name == "officeqa.csv":
        train_pools, val_data = build_train_val(data, "difficulty", "answer", "question", settings)
        agent_options = make_base_agent_options(model=settings.model, provider=settings.provider)
        scorer = _officeqa_scorer
        prompt_path = (Path(get_project_root()) / "src" / "agent_profiles" / "base_agent" / "prompt.txt")
    elif dataset_name == "dabstep_data.csv":
        # DABstep needs formatted prompts — use level as category
        data_dir = Path(settings.data_dir).resolve()
        context_file_names = sorted(f.name for f in data_dir.iterdir() if f.is_file())
        context_files_text = "\n".join(f"- {data_dir / name}" for name in context_file_names)

        # Format questions with prompt template
        data["formatted_question"] = data.apply(
            lambda row: PROMPT.format(
                context_files=context_files_text,
                question=row["question"],
                guidelines=row["guidelines"],
            ),
            axis=1,
        )
        train_pools, val_data = build_train_val(
            data, "level", "answer", "formatted_question", settings,
        )
        agent_options = make_dabstep_agent_options(model=settings.model, data_dir=settings.data_dir)
        scorer = _dabstep_scorer
    elif dataset_name == "livecodebench_v6.csv":
        train_pools, val_data = build_train_val(
            data, "difficulty", "public_test_cases", "formatted_question", settings,
        )
        agent_options = make_livecodebench_agent_options(model=settings.model, provider=settings.provider)
        scorer = _livecodebench_scorer
        prompt_path = (Path(get_project_root()) / "src" / "agent_profiles" / "livecodebench_agent" / "prompt.txt")
    elif dataset_name in ("frames.csv", "frames_filtered.csv"):
        train_pools, val_data = build_train_val(data, "reasoning_types", "Answer", "Prompt", settings)
        agent_options = make_frames_agent_options(model=settings.model, provider=settings.provider)
        scorer = _sealqa_scorer
    elif dataset_name == "gdpval.csv":
        # GDPval dataset - treated as CSV
        train_pools, val_data = build_train_val(
            data, "sector", "rubric_json", "prompt", settings,
        )
        # GDPval reference files are in data_directories/reference_files
        gdpval_ref_dir = str(Path(get_project_root()) / "data_directories" / "reference_files")
        agent_options = make_gdpval_agent_options(model=settings.model, data_dir=gdpval_ref_dir)
        scorer = score_gdpval
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Create isolated run dir — base agent runs from here, skills land here
    if settings.session:
        session_name = settings.session
    else:
        model_slug = (settings.model or "default").replace("/", "_")
        session_name = f"{model_slug}_evolved"
    run_dir = prepare_run_dir(session_name)
    print(f"Run directory: {run_dir}")

    # Init a git repo in the run dir for ProgramManager branch tracking
    if not (run_dir / ".git").exists():
        subprocess.run(["git", "init"], cwd=run_dir, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=run_dir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "init", "--allow-empty"], cwd=run_dir, capture_output=True)

    # Point all agents at the run dir so skills, git, everything is isolated
    run_dir_str = str(run_dir)

    # Base agent (opencode): inject run_dir into options dict
    original_factory = agent_options
    def agent_factory():
        opts = original_factory() if callable(original_factory) else original_factory
        if isinstance(opts, dict):
            opts["run_dir"] = run_dir_str
        return opts

    # Claude SDK agents: set cwd to run dir
    skill_proposer_options.cwd = run_dir_str
    prompt_proposer_options.cwd = run_dir_str
    skill_generator_options.cwd = run_dir_str
    prompt_generator_options.cwd = run_dir_str

    agents = LoopAgents(
        base=Agent(agent_factory, AgentResponse),
        skill_proposer=Agent(skill_proposer_options, SkillProposerResponse),
        prompt_proposer=Agent(prompt_proposer_options, PromptProposerResponse),
        skill_generator=Agent(skill_generator_options, ToolGeneratorResponse),
        prompt_generator=Agent(prompt_generator_options, PromptGeneratorResponse),
    )
    manager = ProgramManager(cwd=run_dir_str)

    config = LoopConfig(
        max_iterations=settings.max_iterations,
        frontier_size=settings.frontier_size,
        no_improvement_limit=settings.no_improvement_limit,
        concurrency=settings.concurrency,
        evolution_mode=settings.mode,
        categories_per_batch=settings.categories_per_batch,
        samples_per_category=settings.samples_per_category,
        cache_enabled=not settings.no_cache,
        reset_feedback=not settings.no_reset_feedback,
        continue_mode=settings.continue_loop,
    )

    print(f"Dataset: {dataset_name}")
    print(f"Loop: mode={settings.mode}, sdk={settings.sdk}, model={settings.model}")
    print(f"Config: max_iter={settings.max_iterations}, cats_per_batch={settings.categories_per_batch}, samples_per_cat={settings.samples_per_category}")

    if settings.optimizer == "gepa":
        student_model = settings.model or "claude-opus-4-5-20251101"
        reflection_model = settings.gepa_reflection_model or student_model
        provider = settings.provider or "anthropic"
        print(f"Optimizer: dspy.GEPA (model={student_model}, reflection={reflection_model}, provider={provider})")
        loop = GEPALoop(
            config, agents, manager, train_pools, val_data,
            scorer=scorer,
            student_model=student_model,
            reflection_model=reflection_model,
            provider=provider,
            prompt_path=prompt_path,
        )
    else:
        print(f"Optimizer: EvoSkill (two-step proposer+generator)")
        loop = SelfImprovingLoop(config, agents, manager, train_pools, val_data, scorer=scorer, session=session_name)

    result = await loop.run()

    print(f"Best: {result.best_program} ({result.best_score:.2%})")
    print(f"Frontier: {result.frontier}")

    # Export best skills — they're already in run_dir/.claude/skills/ on the best branch
    if result.best_program != "base" and settings.mode == "skill_only":
        exported = loop.export_best_skills(run_dir=run_dir)
        if exported:
            print(f"Exported skills to {run_dir}: {exported}")


if __name__ == "__main__":
    settings = LoopSettings()
    try:
        asyncio.run(main(settings))
    except Exception as e:
        import traceback
        print(f"\n[FATAL] Loop crashed: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise
