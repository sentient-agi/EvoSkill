"""Helper functions for the self-improving loop."""

from pathlib import Path
from typing import TYPE_CHECKING

from src.harness.opencode.skill_utils import (
    ensure_skill_frontmatter,
    normalize_project_skill_frontmatter,
)

if TYPE_CHECKING:
    from src.harness import AgentTrace
    from src.schemas import ProposerResponse, SkillProposerResponse, PromptProposerResponse


def build_proposer_query(
    traces_with_answers: list[tuple["AgentTrace", str, str, str]],
    feedback_history: str,
    evolution_mode: str = "skill_only",
    truncation_level: int = 0,
    task_constraints: str = "",
    project_root: str | Path | None = None,
) -> str:
    """Build the query for the proposer agent from multiple failure traces.

    Args:
        traces_with_answers: List of (trace, agent_answer, ground_truth, category) tuples.
        feedback_history: Previous feedback history.
        evolution_mode: "skill_only" or "prompt_only" - affects trace truncation.
        truncation_level: Context reduction level (0=full, 1=moderate, 2=aggressive).
        task_constraints: Optional task-specific constraints to include in the query.

    Returns:
        Formatted query string for the proposer.
    """
    # Truncation level settings: (head_chars, tail_chars, feedback_lines, max_failures)
    TRUNCATION_SETTINGS = [
        (60_000, 60_000, None, None),    # Level 0: full
        (20_000, 10_000, 20, 3),         # Level 1: moderate
        (5_000, 2_000, 5, 2),            # Level 2: aggressive
    ]
    head_chars, tail_chars, feedback_lines, max_failures = TRUNCATION_SETTINGS[
        min(truncation_level, len(TRUNCATION_SETTINGS) - 1)
    ]

    # Apply max_failures limit
    if max_failures is not None and len(traces_with_answers) > max_failures:
        traces_with_answers = traces_with_answers[:max_failures]

    # Apply feedback truncation
    if feedback_lines is not None:
        feedback_lines_list = feedback_history.split("\n")
        if len(feedback_lines_list) > feedback_lines:
            feedback_history = "\n".join(feedback_lines_list[-feedback_lines:])

    # Get existing skills for context
    skills_dir = Path(project_root) / ".claude" / "skills" if project_root else Path(".claude/skills")
    existing_skills = []
    if skills_dir.exists():
        for skill_dir in skills_dir.iterdir():
            if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                existing_skills.append(skill_dir.name)
    skills_list = "\n".join([f"- {s}" for s in existing_skills]) or "None"

    # Collect categories for summary
    categories = [cat for _, _, _, cat in traces_with_answers]
    category_summary = ", ".join(sorted(set(categories)))

    # Build failure summaries with truncation-level-aware settings
    failure_sections = []
    for i, (trace, agent_answer, ground_truth, category) in enumerate(traces_with_answers, 1):
        # For prompt mode, use more aggressive truncation to focus on patterns
        # For skill mode, keep full trace to see tool usage (but respect truncation level)
        if evolution_mode == "prompt_only":
            # Prompt mode uses tighter truncation even at level 0
            effective_head = min(head_chars, 20_000)
            effective_tail = min(tail_chars, 10_000)
        else:
            effective_head = head_chars
            effective_tail = tail_chars

        trace_summary = trace.summarize(head_chars=effective_head, tail_chars=effective_tail)

        failure_sections.append(f"""### Failure {i} [Category: {category}]
{trace_summary}

Agent Answer: {agent_answer}
Ground Truth: {ground_truth}
""")

    failures_text = "\n".join(failure_sections)

    constraints_section = f"\n## Task Constraints\n{task_constraints}\n" if task_constraints else ""

    return f"""## Existing Skills (check before proposing new ones)
{skills_list}
{constraints_section}
## Previous Attempts Feedback
{feedback_history}

## Current Failures ({len(traces_with_answers)} samples across categories: {category_summary})

Analyze the patterns across these failures to identify a GENERAL improvement, not a fix for any single case.

{failures_text}

## Your Task
1. Check if any EXISTING skill should have handled these failures
2. If yes → propose EDITING that skill (action="edit", target_skill="skill-name")
3. If no → propose a NEW skill (action="create")
4. Reference any related DISCARDED iterations and explain how your proposal differs
5. Identify what COMMON pattern or capability gap caused these failures across categories"""


def build_skill_query(proposer_trace: "AgentTrace[ProposerResponse]") -> str:
    """Build the query for the skill generator agent.

    Args:
        proposer_trace: The trace from the proposer agent.

    Returns:
        Formatted query string for the skill generator.
    """
    return f"""Proposed tool or skill (high level description): {proposer_trace.output.proposed_skill_or_prompt}

Justification: {proposer_trace.output.justification}"""


def build_prompt_query(
    proposer_trace: "AgentTrace[ProposerResponse]", original_prompt: str
) -> str:
    """Build the query for the prompt generator agent.

    Args:
        proposer_trace: The trace from the proposer agent.
        original_prompt: The original system prompt to optimize.

    Returns:
        Formatted query string for the prompt generator.
    """
    return f"""## Original Prompt
{original_prompt}

## Proposed Change
{proposer_trace.output.proposed_skill_or_prompt}

## Justification
{proposer_trace.output.justification}"""


def append_feedback(
    path: Path,
    iteration: str,
    proposal: str,
    justification: str,
    outcome: str | None = None,
    score: float | None = None,
    parent_score: float | None = None,
    active_skills: list[str] | None = None,
    failure_category: str | None = None,
    root_cause: str | None = None,
) -> None:
    """Append feedback entry to history file with outcome tracking.

    Args:
        path: Path to the feedback history file.
        iteration: Iteration identifier (e.g., "iter-1").
        proposal: The skill or prompt that was proposed.
        justification: Why this change was proposed.
        outcome: "improved", "no_improvement", or "discarded".
        score: The score achieved after applying this proposal.
        parent_score: The parent's score before this proposal.
        active_skills: List of skills that were active during evaluation.
        failure_category: Category of failure (e.g., "methodology", "formatting").
        root_cause: Brief description of root cause.
    """
    # Build outcome section if available
    outcome_section = ""
    if outcome is not None:
        delta = (score - parent_score) if (score is not None and parent_score is not None) else None
        delta_str = f" ({delta:+.4f})" if delta is not None else ""
        score_str = f" (score: {score:.4f}{delta_str})" if score is not None else ""
        outcome_section = f"\n**Outcome**: {outcome.upper()}{score_str}"

    # Build diagnostic section
    diagnostic_section = ""
    if active_skills:
        diagnostic_section += f"\n**Active Skills**: {', '.join(active_skills)}"
    if failure_category:
        diagnostic_section += f"\n**Failure Category**: {failure_category}"
    if root_cause:
        diagnostic_section += f"\n**Root Cause**: {root_cause}"

    entry = f"""
## {iteration}
**Proposal**: {proposal}
**Justification**: {justification}{outcome_section}{diagnostic_section}

"""
    with open(path, "a") as f:
        f.write(entry)


def read_feedback_history(path: Path) -> str:
    """Read feedback history or return default message.

    Args:
        path: Path to the feedback history file.

    Returns:
        Contents of feedback file or default message.
    """
    if path.exists():
        return path.read_text()
    return "No previous attempts."


def update_prompt_file(file_path: Path, new_prompt: str) -> None:
    """Write the new prompt to prompt.txt.

    The Agent reads this file at runtime on each run().

    Args:
        file_path: Path to the prompt file.
        new_prompt: The new prompt content.
    """
    file_path.write_text(new_prompt.strip())


def build_skill_query_from_skill_proposer(
    proposer_trace: "AgentTrace[SkillProposerResponse]",
) -> str:
    """Build the query for the skill generator from a skill proposer trace.

    Args:
        proposer_trace: The trace from the skill proposer agent.

    Returns:
        Formatted query string for the skill generator.
    """
    return f"""Proposed tool or skill (high level description): {proposer_trace.output.proposed_skill}

Justification: {proposer_trace.output.justification}"""


def build_prompt_query_from_prompt_proposer(
    proposer_trace: "AgentTrace[PromptProposerResponse]",
    original_prompt: str,
) -> str:
    """Build the query for the prompt generator from a prompt proposer trace.

    Args:
        proposer_trace: The trace from the prompt proposer agent.
        original_prompt: The original system prompt to optimize.

    Returns:
        Formatted query string for the prompt generator.
    """
    return f"""## Original Prompt
{original_prompt}

## Proposed Change
{proposer_trace.output.proposed_prompt_change}

## Justification
{proposer_trace.output.justification}"""
