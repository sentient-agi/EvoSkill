from src.api.data_utils import stratified_split
import ast
import json
import os
import pandas as pd
import shutil
from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)
from pathlib import Path
from typing import Literal, Optional

from src.agent_profiles.skill_generator import get_project_root

PROJECT_ROOT = Path(get_project_root())
SKILLS_DIR = PROJECT_ROOT / ".claude" / "skills"
RUNS_DIR = PROJECT_ROOT / ".evoskill-runs"

# Skill names that are meta-tools (not task-specific evolved skills)
META_SKILLS = {"skill-creator"}


def prepare_run_dir(session_name: str, exclude_dataset: bool = False) -> Path:
    """Create an isolated directory for an opencode run.

    Each session gets its own dir with opencode.json, .env, and data symlinks.
    If the session already has skills (from a prior loop run), they are preserved.
    Otherwise a fresh .claude/skills/ with only meta-skills is created.

    Args:
        session_name: Name for the session directory.
        exclude_dataset: If True, skip symlinking .dataset/ to prevent
                        ground-truth contamination (e.g. for gdpval).

    Returns:
        Path to the run directory.
    """
    run_dir = RUNS_DIR / session_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Always refresh opencode.json and .env
    opencode_json = PROJECT_ROOT / "opencode.json"
    if opencode_json.exists():
        shutil.copy2(str(opencode_json), str(run_dir / "opencode.json"))

    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        shutil.copy2(str(env_file), str(run_dir / ".env"))

    # Symlink data directories the agent may need
    symlink_dirs = ["data_directories", ".opencode"]
    if not exclude_dataset:
        symlink_dirs.append(".dataset")
    for data_dir_name in symlink_dirs:
        src = PROJECT_ROOT / data_dir_name
        dest = run_dir / data_dir_name
        if src.exists() and not dest.exists():
            os.symlink(str(src), str(dest))

    # If session already has skills (from a loop run), keep them
    # Otherwise create fresh with only meta-skills
    run_skills = run_dir / ".claude" / "skills"
    existing_evolved = [
        d.name for d in run_skills.iterdir()
        if d.is_dir() and d.name not in META_SKILLS and (d / "SKILL.md").exists()
    ] if run_skills.exists() else []

    if existing_evolved:
        # Ensure meta-skills are present
        if SKILLS_DIR.exists():
            for skill_dir in SKILLS_DIR.iterdir():
                if skill_dir.is_dir() and skill_dir.name in META_SKILLS:
                    dest = run_skills / skill_dir.name
                    if not dest.exists():
                        shutil.copytree(str(skill_dir), str(dest))
    else:
        # Fresh session: only meta-skills
        if run_skills.exists():
            shutil.rmtree(str(run_skills))
        run_skills.mkdir(parents=True, exist_ok=True)
        if SKILLS_DIR.exists():
            for skill_dir in SKILLS_DIR.iterdir():
                if skill_dir.is_dir() and skill_dir.name in META_SKILLS:
                    shutil.copytree(str(skill_dir), str(run_skills / skill_dir.name))

    return run_dir


def list_active_skills() -> list[str]:
    """List non-meta skills currently on disk."""
    if not SKILLS_DIR.exists():
        return []
    return [
        d.name for d in SKILLS_DIR.iterdir()
        if d.is_dir() and d.name not in META_SKILLS and (d / "SKILL.md").exists()
    ]


def split_held_out(
    data: pd.DataFrame,
    settings: "EvalSettings",
    category_col: str,
    extra_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Apply stratified split and return only the held-out test set.

    This is the single place where train/val/test splitting happens for eval.
    Every load_* function should call this instead of doing its own split.

    Args:
        data: DataFrame with 'question' and 'ground_truth' columns already renamed.
        settings: EvalSettings with train_ratio, val_ratio, dataset_slice.
        category_col: Name of the column to use as category for stratification.
        extra_cols: Additional columns to preserve through the split.

    Returns:
        DataFrame containing only the held-out test rows.
    """
    data = data.rename(columns={category_col: "category"})
    _train, _val, test_data = stratified_split(
        data,
        train_ratio=settings.train_ratio,
        val_ratio=settings.val_ratio,
        max_examples=settings.dataset_slice,
        extra_cols=extra_cols,
    )

    # Build column names for the test DataFrame
    cols = ["question", "ground_truth", "category"]
    if extra_cols:
        cols.extend(extra_cols)
    test_df = pd.DataFrame(test_data, columns=cols)

    n_train = sum(1 for _ in _train.values() for _ in _)  # flatten train pools
    print(f"Split: {n_train} train, {len(_val)} val, {len(test_data)} test (slice={settings.dataset_slice}, ratio={settings.train_ratio}/{settings.val_ratio})")
    return test_df


class EvalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        cli_parse_args=True,
    )
    output: Path = Field(
        default=Path("results/eval_results.pkl"), description="Output pkl file path"
    )
    max_concurrent: int = Field(default=8, description="Max concurrent evaluations")
    resume: bool = Field(default=True, description="Resume from existing results")
    difficulty: Literal["all", "easy", "hard"] = Field(
        default="all", description="Filter by difficulty"
    )
    topic: str = Field(
        default="all", description="Filter by topic"
    )
    level: str = Field(
        default="all", description="Filter by level"
    )
    platform: str = Field(
        default="all", description="Filter by platform"
    )
    dataset_slice: Optional[int] = Field(
        default=None, description="Limit dataset to first N rows (passed as max_examples to stratified split)"
    )
    offset: int = Field(
        default=0, description="Skip the first N questions"
    )
    model: Optional[str] = Field(
        default="claude-opus-4-5-20251101",
        description="Model for base agent (opus, sonnet, haiku)",
    )
    dataset_path: Path = Field(
        default=Path(".dataset/officeqa.csv").expanduser(),
        description="Path to evaluation dataset CSV",
    )
    data_dir: str = Field(
        default="DABstep-data/data/context", description="Path to shared context files directory"
    )
    sdk: Literal["claude", "opencode"] = Field(
        default="claude",
        description="SDK to use: 'claude' or 'opencode'",
    )
    provider: Optional[str] = Field(
        default=None, description="Provider ID for opencode SDK (e.g., gemini, arc). Required when --sdk=opencode."
    )
    no_skills: bool = Field(
       default=False,
       description="Run baseline without evolved skills (temporarily hides .claude/skills/)" 
    )
    train_ratio: float = Field(
        default=0.13, description="Train ratio for stratified split"
    )
    val_ratio: float = Field(
        default=0.13, description="Val ratio for stratified split"
    )
    session: Optional[str] = Field(
        default=None, description="Session name for isolated run dir (e.g., 'gemini_baseline'). Auto-generated if not set.",
    )
    prompt_file: Optional[Path] = Field(
        default=None, description="Path to a prompt file to use instead of the default prompt.txt (e.g. .evoskill-runs/<session>/gepa_prompt.txt)"
    )

def load_officeqa(data: pd.DataFrame, settings: EvalSettings) -> list[tuple]:
    data = data.rename(columns={"answer": "ground_truth"})
    test_df = split_held_out(data, settings, category_col="difficulty")

    # Filter by difficulty if requested
    if settings.difficulty != "all":
        test_df = test_df[test_df["category"] == settings.difficulty]

    print(f"Evaluating: {len(test_df)} samples (difficulty={settings.difficulty})")

    items = [
        (int(i), str(row["question"]), str(row["ground_truth"])) for i, row in test_df.iterrows()
    ]

    return items

def load_sealqa(data: pd.DataFrame, settings: EvalSettings) -> list[tuple]:
    data = data.rename(columns={"answer": "ground_truth"})
    test_df = split_held_out(data, settings, category_col="topic")

    # Filter by topic if requested
    if settings.topic != "all":
        test_df = test_df[test_df["category"] == settings.topic]

    items = [
        (idx, row["question"], row["ground_truth"])
        for idx, row in test_df.iterrows()
    ]

    if settings.offset:
        items = items[settings.offset:]

    active = list_active_skills()
    mode = "baseline (no skills)" if settings.no_skills else f"skills: {active or 'none'}"
    print(f"Evaluating: {len(items)} samples (topic={settings.topic}, {mode})")
    print(f"  sdk={settings.sdk} model={settings.model} provider={settings.provider or 'default'}")

    return items

def load_dabstep(data: pd.DataFrame, settings: EvalSettings, PROMPT) -> list[tuple]:
    # Filter by level if requested
    if settings.level != "all":
        data = data[data["level"].astype(str) == settings.level]

    # Limit to dataset_slice if specified
    if settings.dataset_slice is not None:
        data = data.head(settings.dataset_slice)

    print(f"Dataset: {len(data)} samples (level={settings.level})")

    # Auto-discover context files from data-dir
    data_dir = Path(settings.data_dir).resolve()
    context_file_names = sorted(f.name for f in data_dir.iterdir() if f.is_file())
    context_files_text = "\n".join(f"- {data_dir / name}" for name in context_file_names)
    print(f"Context files ({len(context_file_names)}): {', '.join(context_file_names)}")

    # Prepare items: (task_id, formatted_prompt, answer)
    items = [
        (
            row["task_id"],
            PROMPT.format(
                context_files=context_files_text,
                question=row["question"],
                guidelines=row["guidelines"],
            ),
            row["answer"],
        )
        for _, row in data.iterrows()
    ]

    return items

def load_livecode(data: pd.DataFrame, settings: EvalSettings) -> list[tuple]:
    # Filter by platform if requested
    if settings.platform != "all":
        data = data[data["platform"] == settings.platform]

    # Rename for stratified split
    data = data.rename(columns={"formatted_question": "question", "public_test_cases": "ground_truth"})

    # Stratified split by difficulty
    test_df = split_held_out(data, settings, category_col="difficulty")

    print(f"Evaluating: {len(test_df)} samples (platform={settings.platform})")

    # Prepare items: (index, formatted_question, public_test_cases)
    items = [
        (idx, row["question"], row["ground_truth"])
        for idx, row in test_df.iterrows()
    ]

    return items


def _simplify_reasoning_category_frames(reasoning_types: str) -> str:
    """Collapse fine-grained reasoning_types into broad categories.

    Priority order (first match wins):
      1. 4+ types → "Complex reasoning"
      2. Contains "Post processing" → "Post processing"
      3. Contains "Tabular reasoning" → "Tabular reasoning"
      4. Contains "Temporal reasoning" → "Temporal reasoning"
      5. Contains "Numerical reasoning" → "Numerical reasoning"
      6. Everything else → "Multiple constraints"
    """
    parts = [p.strip() for p in reasoning_types.split("|")]
    if len(parts) >= 4:
        return "Complex reasoning"
    if "Post processing" in parts:
        return "Post processing"
    if "Tabular reasoning" in parts:
        return "Tabular reasoning"
    if "Temporal reasoning" in parts:
        return "Temporal reasoning"
    if "Numerical reasoning" in parts:
        return "Numerical reasoning"
    return "Multiple constraints"


def prepare_frames_data(data: pd.DataFrame) -> pd.DataFrame:
    """Rename columns and simplify categories for FRAMES dataset.

    Standardizes to stratified_split's expected columns:
    question, ground_truth, category.
    """
    data = data.rename(columns={
        "Prompt": "question",
        "Answer": "ground_truth",
        "reasoning_types": "category",
    })
    data["category"] = data["category"].apply(_simplify_reasoning_category_frames)
    return data


def load_frames(data: pd.DataFrame, settings: EvalSettings) -> list[tuple]:
    data = prepare_frames_data(data)

    _train, _val, test_data = stratified_split(
        data, train_ratio=settings.train_ratio, val_ratio=settings.val_ratio,
        max_examples=settings.dataset_slice,
    )
    data = pd.DataFrame(test_data, columns=["question", "answer", "category"])
    print(f"Sampled dataset: {settings.dataset_slice}, Held-out test set: {len(data)} samples (train={settings.train_ratio:.0%}, val={settings.val_ratio:.0%})")

    items = [
        (idx, row["question"], row["ground_truth"])
        for idx, row in test_df.iterrows()
    ]

    if settings.offset:
        items = items[settings.offset:]
    active = list_active_skills()
    mode = "baseline (no skills)" if settings.no_skills else f"skills: {active or 'none'}"
    print(f"Evaluating: {len(items)} samples ({mode})")
    print(f"  sdk={settings.sdk} model={settings.model} provider={settings.provider or 'default'}")

    return items


def load_gdpval(data: pd.DataFrame, settings: EvalSettings, output_base_dir: Path | None = None) -> list[tuple]:
    """Load GDPval dataset items.
    
    GDPval has columns: task_id, sector, occupation, prompt,
    reference_files, deliverable_files, rubric_json, etc.
    
    Args:
        data: The DataFrame containing GDPval data
        settings: Evaluation settings
        output_base_dir: Base directory where agent should save deliverables
                        (if None, uses project_root/output/gdpval_deliverables)
    """
    from src.agent_profiles.skill_generator import get_project_root
    
    data = data.rename(columns={"prompt": "question", "rubric_json": "ground_truth"})
    extra = ["task_id", "deliverable_files", "reference_files"]
    test_df = split_held_out(data, settings, category_col="sector", extra_cols=extra)

    active = list_active_skills()
    mode = "baseline (no skills)" if settings.no_skills else f"skills: {active or 'none'}"
    print(f"  sdk={settings.sdk} model={settings.model} provider={settings.provider or 'default'}")

    # Set up output directory for generated deliverables
    if output_base_dir is None:
        output_base_dir = Path(get_project_root()) / "output" / "gdpval_deliverables"
    output_base_dir.mkdir(parents=True, exist_ok=True)
    print(f"  deliverables_dir={output_base_dir}")

    # Prepare items: (index, prompt, rubric_info_json)
    items = []

    # Apply offset
    if settings.offset:
        test_data = test_data[settings.offset:]

    ref_base = Path(get_project_root()) / "data_directories"

    for idx, (prompt, rubric_json, sector, task_id, deliverable_files, reference_files) in enumerate(test_data):
        # Create task-specific deliverable directory
        task_deliverable_dir = output_base_dir / task_id / "deliverables"
        task_deliverable_dir.mkdir(parents=True, exist_ok=True)

        # Parse reference file paths and resolve to absolute paths
        try:
            ref_list = ast.literal_eval(reference_files) if isinstance(reference_files, str) else []
        except (ValueError, SyntaxError):
            ref_list = []
        resolved_refs = [str(ref_base / ref_path) for ref_path in ref_list]

        # Parse expected deliverable filenames
        try:
            del_list = ast.literal_eval(deliverable_files) if isinstance(deliverable_files, str) else []
        except (ValueError, SyntaxError):
            del_list = []
        expected_names = [Path(f).name for f in del_list]

        ref_section = ""
        if resolved_refs:
            ref_lines = "\n".join(f"- {p}" for p in resolved_refs)
            ref_section = f"\n\nREFERENCE FILES for this task (use Read tool to access):\n{ref_lines}"

        deliverable_section = ""
        if expected_names:
            del_lines = "\n".join(f"- {name}" for name in expected_names)
            deliverable_section = f"\n\nEXPECTED DELIVERABLE(S) — you MUST use these exact filenames:\n{del_lines}"

        # Store rubric info and deliverable directory for scoring
        rubric_info = {
            "task_id": task_id,
            "rubric_json": rubric_json,
            "deliverable_files": del_list,
            "reference_files": ref_list,
            "generated_dir": str(task_deliverable_dir),
        }

        # Enhanced prompt with references, expected deliverables, and save location
        enhanced_prompt = f"""{prompt}{ref_section}{deliverable_section}

IMPORTANT: You must save your deliverable file(s) to the following directory:
{task_deliverable_dir}

Create the deliverable(s) exactly as requested and save them to the specified directory path.

You have been given everything you needed through the prompt and all (if any) reference files. Finish the task without further user input."""

        items.append((idx, enhanced_prompt, json.dumps(rubric_info)))

    print(f"Evaluating: {len(items)} samples (category={settings.topic}, {mode})")
    return items