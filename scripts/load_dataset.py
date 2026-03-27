from src.api.data_utils import stratified_split
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
META_SKILLS = {"skill-creator", "brainstorming"}


def prepare_run_dir(session_name: str, include_skills: bool) -> Path:
    """Create an isolated directory for an opencode run.

    Each session gets its own dir with opencode.json and .env copied in.
    If the session dir already has skills (e.g., from a prior loop run),
    they are preserved — the eval reuses them.

    Args:
        session_name: Name for the session directory.
        include_skills: Whether to copy evolved skills into the run dir.
            Ignored if the session already has evolved skills (from a loop).

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

    # Symlink data directories the agent may need (e.g., treasury docs for officeqa)
    for data_dir_name in ["treasury_bulletins_parsed", "DABstep-data"]:
        src = PROJECT_ROOT / data_dir_name
        dest = run_dir / data_dir_name
        if src.exists() and not dest.exists():
            os.symlink(str(src), str(dest))

    # If session already has skills (from a loop run), keep them
    run_skills = run_dir / ".claude" / "skills"
    existing_evolved = [
        d.name for d in run_skills.iterdir()
        if d.is_dir() and d.name not in META_SKILLS and (d / "SKILL.md").exists()
    ] if run_skills.exists() else []

    if existing_evolved:
        # Session has evolved skills from a loop — preserve them
        # Just ensure meta-skills are present
        if SKILLS_DIR.exists():
            for skill_dir in SKILLS_DIR.iterdir():
                if skill_dir.is_dir() and skill_dir.name in META_SKILLS:
                    dest = run_skills / skill_dir.name
                    if not dest.exists():
                        shutil.copytree(str(skill_dir), str(dest))
    elif include_skills and SKILLS_DIR.exists():
        # Fresh session, copy all skills (meta + evolved)
        if run_skills.exists():
            shutil.rmtree(str(run_skills))
        shutil.copytree(str(SKILLS_DIR), str(run_skills))
    else:
        # Baseline: only copy meta-skills (skill-creator, etc.)
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
        default=None, description="Truncate dataset to first N rows BEFORE train/val/test splitting (use to shrink the total pool)"
    )
    num_samples: Optional[int] = Field(
        default=None, description="Limit to first N items AFTER splitting (use to cap how many questions to evaluate)"
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
    provider: str = Field(
        default=None, description="Provider ID for opencode SDK (e.g., gemini, arc). Required when --sdk=opencode."
    )
    held_out: bool = Field(
        default=True,
        description="Evaluate only on the held-out test set (excludes train/val samples)",
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

def load_officeqa(data: pd.DataFrame, settings: EvalSettings) -> list[tuple]:
    if settings.held_out:
        data.rename(columns={"answer": "ground_truth", "difficulty": "category"}, inplace=True)
        _train, _val, test_data = stratified_split(data, train_ratio=settings.train_ratio, val_ratio=settings.val_ratio)
        # Rebuild dataframe from held-out tuples
        data = pd.DataFrame(test_data, columns=["question", "answer", "difficulty"])
        print(f"Held-out test set: {len(data)} samples (train={settings.train_ratio:.0%}, val={settings.val_ratio:.0%})")
    else:
        print(f"Full dataset: {len(data)} samples")

    # Filter by difficulty if requested
    if settings.difficulty != "all":
        data = data[data["difficulty"] == settings.difficulty]

    # Limit to num_samples if specified
    if settings.num_samples is not None:
        data = data.head(settings.num_samples)

    print(f"Evaluating: {len(data)} samples (difficulty={settings.difficulty})")

    # Prepare items with index
    items = [
        (int(i), str(row["question"]), str(row["answer"])) for i, row in data.iterrows()
    ]

    return items

def load_sealqa(data: pd.DataFrame, settings: EvalSettings) -> list[tuple]:
    if settings.held_out:
        data.rename(columns={"topic": "category", "answer": "ground_truth"}, inplace=True)
        _train, _val, test_data = stratified_split(data, train_ratio=settings.train_ratio, val_ratio=settings.val_ratio)
        # Rebuild dataframe from held-out tuples
        data = pd.DataFrame(test_data, columns=["question", "answer", "topic"])
        print(f"Held-out test set: {len(data)} samples (train={settings.train_ratio:.0%}, val={settings.val_ratio:.0%})")
    else:
        print(f"Full dataset: {len(data)} samples")

    # Filter by topic if requested
    if settings.topic != "all":
        data = data[data["topic"] == settings.topic]

    items = [
        (idx, row["question"], row["answer"])
        for idx, row in data.iterrows()
    ]

    #Apply offset and limit
    if settings.offset:
        items = items[settings.offset:]
    if settings.num_samples is not None:
        items = items[:settings.num_samples]

    # Report config
    active = list_active_skills()
    mode = "baseline (no skills)" if settings.no_skills else f"skills: {active or 'none'}"
    print(f"Evaluating: {len(items)} samples (topic={settings.topic}, {mode})")
    print(f"  sdk={settings.sdk} model={settings.model} provider={settings.provider or 'default'}")

    return items

def load_dabstep(data: pd.DataFrame, settings: EvalSettings, PROMPT) -> list[tuple]:
    # Filter by level if requested
    if settings.level != "all":
        data = data[data["level"].astype(str) == settings.level]

    # Limit to num_samples if specified
    if settings.num_samples is not None:
        data = data.head(settings.num_samples)

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

    # Filter by difficulty if requested
    if settings.difficulty != "all":
        data = data[data["difficulty"] == settings.difficulty]

    # Limit to num_samples if specified
    if settings.num_samples is not None:
        data = data.head(settings.num_samples)

    print(
        f"Dataset: {len(data)} samples (platform={settings.platform}, difficulty={settings.difficulty})"
    )
    # print(f"SDK: {args.sdk}, Model: {args.model}")

    # Prepare items: (index, formatted_question, public_test_cases)
    items = [
        (
            idx,
            row["formatted_question"],
            row["public_test_cases"],
        )
        for idx, row in data.iterrows()
    ]

    return items