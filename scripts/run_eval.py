#!/usr/bin/env python3
"""Run full evaluation on OfficeQA dataset."""

import asyncio
from pathlib import Path

import pandas as pd
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)

from src.agent_profiles import (
    Agent,
    base_agent_options,
    make_base_agent_options,
    sealqa_agent_options, 
    make_sealqa_agent_options,
    dabstep_agent_options, 
    make_dabstep_agent_options,
    make_livecodebench_agent_options,
    make_gdpval_agent_options,
    make_frames_agent_options,
    set_sdk,
)
from src.api.data_utils import stratified_split
from src.evaluation.eval_full import evaluate_full, load_results
from src.agent_profiles.skill_generator import get_project_root
from src.schemas import AgentResponse
from src.evaluation.sealqa_scorer import score_sealqa
from src.evaluation.reward import score_answer
from src.evaluation.dabstep_scorer import question_scorer
from src.evaluation.livecodebench.livecodebench_scorer import score_livecodebench
from src.evaluation.gdpval_scorer import score_gdpval_with_judge
from scripts.load_dataset import load_dabstep, load_livecode, load_officeqa, load_sealqa, load_gdpval, load_frames, prepare_run_dir, list_active_skills, EvalSettings

PROMPT = """You are an expert data analyst and you will answer factoid questions by loading and referencing the files/documents listed below.
You have these files available:
{context_files}

Here is the question you need to answer:
{question}

Here are the guidelines you must follow when answering the question above:
{guidelines}
"""



async def main(settings: EvalSettings):
    set_sdk(settings.sdk)

    # Load dataset
    dataset_path = settings.dataset_path
    dataset_name = dataset_path.name
    data = pd.read_csv(dataset_path)

    # Slice dataset before splitting (e.g., first 120 questions)
    if settings.dataset_slice:
        data = data.head(settings.dataset_slice)
        print(f"Sliced dataset to first {settings.dataset_slice} rows")

    if dataset_name == "officeqa.csv":
        items = load_officeqa(data, settings)
        agent_options = (
            make_base_agent_options(model=settings.model)
            if settings.model
            else base_agent_options
        )
    elif dataset_name == "seal-0.csv":
        items = load_sealqa(data, settings)
        agent_options = make_sealqa_agent_options(model=settings.model, provider=settings.provider)
    elif dataset_name == "dabstep_data.csv":
        items = load_dabstep(data, settings, PROMPT)
        agent_options = make_dabstep_agent_options(model=settings.model, data_dir=settings.data_dir)
    elif dataset_name == "livecodebench_v6.csv":
        items = load_livecode(data, settings)
        agent_options = make_livecodebench_agent_options(model=settings.model, provider=settings.provider)
    elif dataset_name in ("frames.csv", "frames_filtered.csv"):
        items = load_frames(data, settings)
        agent_options = make_frames_agent_options(model=settings.model, provider=settings.provider)
    elif dataset_name == "gdpval.csv":
        # Set up output directory for GDPval deliverables
        gdpval_output_dir = Path(get_project_root()) / "output" / "gdpval_deliverables"
        items = load_gdpval(data, settings, output_base_dir=gdpval_output_dir)
        # GDPval reference files are in data_directories/reference_files
        gdpval_ref_dir = str(Path(get_project_root()) / "data_directories" / "reference_files")
        agent_options = make_gdpval_agent_options(model=settings.model, data_dir=gdpval_ref_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Prepare isolated run directory for opencode (avoids skill conflicts between runs)
    include_skills = not settings.no_skills
    if settings.session:
        session_name = settings.session
    else:
        model_slug = (settings.model or "default").replace("/", "_")
        session_name = f"{model_slug}_{'evolved' if include_skills else 'baseline'}"
    run_dir = prepare_run_dir(session_name, include_skills=include_skills)
    print(f"Run directory: {run_dir}")

    # Wrap agent_options to inject run_dir for opencode
    original_factory = agent_options
    def agent_factory():
        opts = original_factory() if callable(original_factory) else original_factory
        if isinstance(opts, dict):
            opts["run_dir"] = str(run_dir)
        return opts

    agent = Agent(agent_factory, AgentResponse)

    model_info = f" (model: {settings.model})" if settings.model else " (model: opus)"
    print(f"Agent configured{model_info}")

    await evaluate_full(
        agent=agent,
        items=items,
        output_path=settings.output,
        max_concurrent=settings.max_concurrent,
        resume=settings.resume,
    )

    # Summary
    all_results = load_results(settings.output)
    successful = [r for r in all_results if r.error is None]
    failed = [r for r in all_results if r.error is not None]

    # Score successful results (for officeqa use score_answer from reward.py)
    correct = 0
    scored = 0
    for r in successful:
        predicted = None
        if r.trace: 
            if r.trace.output and r.trace.output.final_answer:
                predicted = str(r.trace.output.final_answer)
            elif r.trace.result and r.trace.result.strip():
                predicted = r.trace.result.strip()
        
        if predicted:
            scored += 1
            if dataset_name == "officeqa.csv":
                score = score_answer(str(r.ground_truth), predicted)
                if score > 0:
                    correct += 1
            elif dataset_name == "seal-0.csv":
                score = score_sealqa(r.question, str(r.ground_truth), predicted)
                if score > 0:
                    correct += 1
            elif dataset_name == "dabstep_data.csv":
                score = question_scorer(predicted, str(r.ground_truth))
                if score:
                    correct += 1
            elif dataset_name in ("frames.csv", "frames_filtered.csv"):
                score = score_sealqa(str(r.ground_truth), predicted)
                if score > 0:
                    correct += 1
            elif dataset_name == "livecodebench_v6.csv":
                score = score_livecodebench(r.question, str(r.ground_truth), predicted)
                if score > 0:
                    correct += 1
            # elif dataset_name == "gdpval.csv":
                # GDPval uses async multimodal scoring with file comparison
                # Parse rubric_info from ground_truth
                # Skipping scoring due to complexity
                # import json
                # rubric_info = json.loads(r.ground_truth)
                # task_id = rubric_info["task_id"]
                # generated_dir = Path(rubric_info["generated_dir"])
                
                # # Get reference deliverable paths from the dataset
                # ref_deliverables = rubric_info.get("deliverable_files", [])
                # if isinstance(ref_deliverables, str):
                #     ref_deliverables = json.loads(ref_deliverables)
                
                # # Get GDPval base path
                # from src.evaluation.gdpval_scorer import get_gdpval_base_path
                # gdpval_base = get_gdpval_base_path()
                
                # # Run async scoring
                # score, rationale = await score_gdpval_with_judge(
                #     task_id=task_id,
                #     prompt=r.question,
                #     rubric_json=rubric_info.get("rubric_json", ""),
                #     generated_dir=generated_dir,
                #     reference_deliverable_paths=ref_deliverables,
                #     gdpval_base_path=gdpval_base,
                # )
                # if score > 0:
                #     correct += 1
                # # Store rationale in result metadata for debugging
                # print(f"  GDPval task {task_id}: score={score}, rationale={rationale[:100]}...")

    print(f"\n{'=' * 50}")
    print(f"Total completed: {len(all_results)}/{len(data)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    if failed:
        print(f"Failed indices: {[r.index for r in failed]}")
    print(f"Accuracy: {correct}/{scored} ({correct/scored*100:.1f}%)" if scored != 0 else "Accuracy: N/A (no answers to score)")
    print(f"Results saved to: {settings.output}")


if __name__ == "__main__":
    settings = EvalSettings()
    asyncio.run(main(settings))
