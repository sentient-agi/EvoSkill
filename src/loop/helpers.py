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
    traces_with_answers: list[tuple["AgentTrace", str, str, str, str]],
    feedback_history: str,
    evolution_mode: str = "skill_only",
    truncation_level: int = 0,
    task_constraints: str = "",
    project_root: str | Path | None = None,
    past_traces_index: str = "",
    runtime_proposals: str = "",
) -> str:
    """Build the query for the proposer agent from multiple failure traces.

    Args:
        traces_with_answers: List of (trace, agent_answer, ground_truth, category, question) tuples.
        feedback_history: Previous feedback history.
        evolution_mode: "skill_only" or "prompt_only" - affects trace truncation.
        truncation_level: Context reduction level (0=full, 1=moderate, 2=aggressive).
        task_constraints: Optional task-specific constraints to include in the query.
        past_traces_index: Lightweight index of past traces (progressive disclosure).
        runtime_proposals: Unvalidated insights from the background reviewer.

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
    categories = [cat for _, _, _, cat, _ in traces_with_answers]
    category_summary = ", ".join(sorted(set(categories)))

    # Write current failures to files AND present a compact inline index.
    # Each failure gets:
    #   - A short "tool call skeleton" inline (tool names only, no thinking, no results)
    #   - A reference to the full transcript file (Read on demand)
    #
    # This keeps the query small regardless of trace size / failure count.
    failure_dir = None
    if project_root:
        failure_dir = Path(project_root) / ".cache" / "current_failures"
        failure_dir.mkdir(parents=True, exist_ok=True)

    failure_sections = []
    for i, (trace, agent_answer, ground_truth, category, question) in enumerate(traces_with_answers, 1):
        # Build compact skeleton: just tool names in turn order (no thinking, no tool results).
        skeleton_lines = []
        try:
            from src.harness.agent import _render_turn_transcript  # reuse for block parsing
        except ImportError:
            _render_turn_transcript = None
        try:
            # Lightweight pass: scan messages for ToolUseBlocks in order
            from claude_agent_sdk import AssistantMessage, ToolUseBlock
            turn_n = 0
            for msg in getattr(trace, "messages", []) or []:
                if not isinstance(msg, AssistantMessage):
                    continue
                turn_n += 1
                tool_names = [getattr(b, "name", "?") for b in getattr(msg, "content", []) or [] if isinstance(b, ToolUseBlock)]
                if tool_names:
                    skeleton_lines.append(f"  T{turn_n}: {' → '.join(tool_names)}")
        except Exception:
            pass
        skeleton = "\n".join(skeleton_lines) or "  (no tool calls captured)"

        # Write the full transcript to a file the evolver can Read on demand
        full_path_ref = "(full trace file not written — project_root not provided)"
        if failure_dir is not None:
            q_hash = abs(hash(question)) % (10 ** 8)
            fail_file = failure_dir / f"failure-{i}_{q_hash}.md"
            full = trace.summarize(head_chars=head_chars, tail_chars=tail_chars)
            fail_file.write_text(
                f"# Current Failure #{i}\n\n"
                f"**Question**: {question}\n"
                f"**Category**: {category}\n"
                f"**Agent Answer**: {agent_answer}\n"
                f"**Ground Truth**: {ground_truth}\n\n"
                f"## Full Transcript\n\n{full}\n"
            )
            full_path_ref = str(fail_file)

        failure_sections.append(
            f"### Failure {i} [Category: {category}]\n"
            f"**Question**: {question[:300]}\n"
            f"**Agent Answer**: {agent_answer[:200]}\n"
            f"**Ground Truth**: {str(ground_truth)[:200]}\n\n"
            f"**Tool call skeleton** (Read the full trace file for thinking, text, tool I/O):\n"
            f"{skeleton}\n\n"
            f"**Full trace**: `{full_path_ref}`\n"
        )

    failures_text = "\n".join(failure_sections)

    constraints_section = f"\n## Task Constraints\n{task_constraints}\n" if task_constraints else ""

    runtime_section = ""
    if runtime_proposals:
        runtime_section = f"""
## Runtime Proposals (from background review — UNVALIDATED candidates)
These insights were extracted automatically from recent solver traces by a lightweight
reviewer. They represent candidate improvements but have NOT been validated.

{runtime_proposals}
"""

    past_traces_section = ""
    if past_traces_index:
        past_traces_section = f"""
{past_traces_index}
"""

    return f"""## Existing Skills (check before proposing new ones)
{skills_list}
{constraints_section}
## Previous Attempts Feedback
{feedback_history}
{runtime_section}{past_traces_section}
## Current Failures ({len(traces_with_answers)} samples across categories: {category_summary})

Analyze the patterns across these failures to identify a GENERAL improvement, not a fix for any single case.

{failures_text}

## Your Task
1. For each failure above, decide if the tool-call skeleton is enough signal. If you need to see thinking, text output, or full tool I/O, Read the `Full trace` file for that failure.
2. Check PAST TRACES INDEX — use Read on any past trace file to see how earlier iterations approached the same question.
3. Check if any EXISTING skill should have handled these failures.
4. If yes → propose EDITING that skill (action="edit", target_skill="skill-name")
5. If no → propose a NEW skill (action="create")
6. Reference any related DISCARDED iterations and explain how your proposal differs
7. Identify what COMMON pattern or capability gap caused these failures across categories

Do NOT create or edit skills that are duplicative of existing ones. Prefer editing over creating when possible.

Always prefer the `Write` tool for creating new skill files (it auto-creates parent directories). Do NOT use `Bash mkdir` — it will be denied."""


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
