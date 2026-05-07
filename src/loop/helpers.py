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
    phase: str = "accuracy",
    phase_metrics: dict | None = None,
    project_skills_dir: str | Path | None = None,
    iter_traces_dir: str | Path | None = None,
    solver_prompt: str = "",
    data_root: str | Path | None = None,
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

    # Get existing skills for context. Prefer the explicit
    # `project_skills_dir` (correct under workspace/project split). Fall
    # back to `project_root/.claude/skills` for backward compat.
    if project_skills_dir is not None:
        skills_dir = Path(project_skills_dir)
    elif project_root is not None:
        skills_dir = Path(project_root) / ".claude" / "skills"
    else:
        skills_dir = Path(".claude/skills")
    existing_skills = []
    if skills_dir.exists():
        for skill_dir in skills_dir.iterdir():
            if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                existing_skills.append(skill_dir.name)
    skills_list = "\n".join([f"- {s}" for s in existing_skills]) or "None"

    # Collect categories for summary
    categories = [cat for _, _, _, cat, _ in traces_with_answers]
    category_summary = ", ".join(sorted(set(categories)))

    # Write current-iteration traces to files AND present a compact inline
    # index. Each entry gets:
    #   - A short "tool call skeleton" inline (tool names only, no thinking, no results)
    #   - A reference to the full transcript file (Read on demand)
    #
    # The directory is `current_iter_traces/` — same name regardless of phase
    # (accuracy vs efficiency); previously it was `current_failures/` which
    # was misleading in Phase 2 where the entries are PASSING samples.
    failure_dir = None
    if iter_traces_dir is not None:
        failure_dir = Path(iter_traces_dir)
        failure_dir.mkdir(parents=True, exist_ok=True)
    elif project_root is not None:
        failure_dir = Path(project_root) / ".cache" / "current_iter_traces"
        failure_dir.mkdir(parents=True, exist_ok=True)

    # Phase-dependent labels for per-trace headers and section titles.
    is_efficiency = (phase == "efficiency")
    item_label = "Passing Sample" if is_efficiency else "Failure"
    section_title = (
        "Training Samples (all passed — analyze for efficiency)"
        if is_efficiency
        else f"Current Failures"
    )

    failure_sections = []
    for i, (trace, agent_answer, ground_truth, category, question) in enumerate(traces_with_answers, 1):
        # Build compact skeleton: just tool names in turn order (no thinking, no tool results).
        skeleton_lines = []
        try:
            from src.harness.agent import _render_turn_transcript  # reuse for block parsing
        except ImportError:
            _render_turn_transcript = None
        try:
            # Lightweight pass: scan messages for ToolUseBlocks in order.
            # We render `name(key_arg)` rather than just `name` so the evolver
            # can see WHICH file the agent Read, WHAT pattern the agent
            # Grepped, etc. — diagnostic signal for ~70% of failures without
            # needing to open the trace file. Each block is capped to ~80
            # chars to keep the skeleton compact.
            from claude_agent_sdk import AssistantMessage, ToolUseBlock

            # Strip the dataset root from rendered file paths so the
            # evolver sees `treasury_bulletins_parsed/transformed/foo.txt`
            # instead of `/Users/dastin/.../officeqa/data/treasury_bulletins_parsed/transformed/foo.txt`.
            # The full path is in the trace file if anyone needs it; the
            # skeleton is meant to be scannable.
            _data_root_prefixes = [
                "/Users/dastin/dev/officeqa/data/",
                # Trailing-slash + non-trailing both, defensive
                "/Users/dastin/dev/officeqa/data",
            ]
            if data_root:
                dr = str(data_root).rstrip("/") + "/"
                if dr not in _data_root_prefixes:
                    _data_root_prefixes.insert(0, dr)

            def _trim_root(s: str) -> str:
                for prefix in _data_root_prefixes:
                    if s.startswith(prefix):
                        return s[len(prefix):].lstrip("/")
                return s

            def _short_input(name: str, inp: dict) -> str:
                """Compact one-line summary of a tool's key argument(s).

                Renders the tool's primary argument verbatim (no truncation),
                with the dataset root stripped from any file paths so the
                evolver sees relative locations like
                `treasury_bulletins_parsed/transformed/foo.txt`.
                """
                if not isinstance(inp, dict):
                    return ""
                # Per-tool key-field selection. Keep this list small; for
                # tools not listed, fall back to the first string-ish value.
                key_field_map = {
                    "Read": "file_path",
                    "Write": "file_path",
                    "Edit": "file_path",
                    "Glob": "pattern",
                    "Grep": "pattern",
                    "Bash": "command",
                    "WebFetch": "url",
                    "WebSearch": "query",
                    "Skill": "skill",
                    "Task": "description",
                    "TodoWrite": None,           # too noisy to summarize
                    "BashOutput": "bash_id",
                }
                if name in key_field_map:
                    field = key_field_map[name]
                    if field is None:
                        return ""
                    val = inp.get(field, "")
                else:
                    # Unknown tool — pick first string-ish value as a hint.
                    val = next((v for v in inp.values() if isinstance(v, str)), "")
                if not val:
                    return ""
                val = str(val).replace("\n", " ").strip()
                val = _trim_root(val)
                return f"({val})"

            turn_n = 0
            for msg in getattr(trace, "messages", []) or []:
                if not isinstance(msg, AssistantMessage):
                    continue
                turn_n += 1
                tool_calls = []
                for b in getattr(msg, "content", []) or []:
                    if not isinstance(b, ToolUseBlock):
                        continue
                    name = getattr(b, "name", "?")
                    inp = getattr(b, "input", None) or {}
                    tool_calls.append(f"{name}{_short_input(name, inp)}")
                if tool_calls:
                    skeleton_lines.append(f"  T{turn_n}: {' → '.join(tool_calls)}")
        except Exception:
            pass
        skeleton = "\n".join(skeleton_lines) or "  (no tool calls captured)"

        # Detect timeout up front — used to suppress misleading cost=$0 in
        # both the per-failure file body and the section header below. The
        # SDK only emits total_cost_usd in the terminal ResultMessage, which
        # never arrives when the wall clock fires, so a partial-run trace
        # carries cost=0 even though the agent really did burn tokens.
        _reason = (
            getattr(trace, "parse_error", None)
            or getattr(trace, "result", None)
            or ""
        )
        _reason_str = str(_reason).strip().splitlines()[0][:200] if _reason else ""
        is_timeout = bool(
            getattr(trace, "output", None) is None
            and (
                "TimeoutError" in _reason_str
                or "timed out" in _reason_str.lower()
            )
        )
        cost_val = getattr(trace, "total_cost_usd", 0) or 0
        cost_str = "n/a (run cut short)" if is_timeout else f"${cost_val:.4f}"

        # Write the full transcript to a file the evolver can Read on demand
        full_path_ref = "(full trace file not written — project_root not provided)"
        if failure_dir is not None:
            q_hash = abs(hash(question)) % (10 ** 8)
            file_prefix = "sample" if is_efficiency else "failure"
            fail_file = failure_dir / f"{file_prefix}-{i}_{q_hash}.md"
            full = trace.summarize(head_chars=head_chars, tail_chars=tail_chars)
            header = "Passing Sample" if is_efficiency else "Current Failure"
            fail_file.write_text(
                f"# {header} #{i}\n\n"
                f"**Question**: {question}\n"
                f"**Category**: {category}\n"
                f"**Agent Answer**: {agent_answer}\n"
                f"**Ground Truth**: {ground_truth}\n"
                f"**Turns**: {getattr(trace, 'num_turns', '?')}, "
                f"**Cost**: {cost_str}\n\n"
                f"## Full Transcript\n\n{full}\n"
            )
            full_path_ref = str(fail_file)

        turns = getattr(trace, "num_turns", "?")

        # Surface run-level failure mode (timeout / crash / parse error) to
        # the evolver as a STATUS BANNER at the top of the section. When
        # `trace.output` is None the agent produced no parseable answer —
        # the failure cause is structural (out of time / crashed) rather
        # than analytical (wrong answer), and the evolver should respond
        # accordingly: "trim turns / shorten searches / cut work" for
        # timeouts vs "improve reasoning skill" for wrong answers. Burying
        # this fact mid-paragraph alongside the question text loses the
        # signal — the banner makes it impossible to miss.
        # `is_timeout` and `_reason_str` were computed up front (see above)
        # so cost rendering in the file body could match this header.
        # Compact one-liner status flag for non-output failures. The verbose
        # explanation of WHY timeouts happen and what to do about them lives
        # in the evolver's system prompt — repeating it on every timeout
        # failure here just bloats the prompt and the evolver tunes it out.
        status_banner = ""
        if getattr(trace, "output", None) is None:
            if is_timeout:
                status_banner = f"\n> ⚠️ **TIMEOUT** — agent did not reach `final_answer` (focus on brevity / targeted lookups, not reasoning quality)\n"
            else:
                status_banner = f"\n> ⚠️ **CRASH** — `{(_reason_str or 'unknown')[:120]}`\n"

        # Always show the full question — the evolver needs every constraint
        # to diagnose root causes (units, rounding rules, scope, etc. are
        # often buried late in the prompt). The 300-char truncation we used
        # to apply for non-timeout failures was hiding load-bearing detail.
        question_text = question
        agent_answer_block = (
            ""
            if is_timeout
            else f"**Agent Answer**: {agent_answer[:200]}\n"
        )
        # Surface the agent's `reasoning` field inline. This is the highest-
        # leverage diagnostic signal: it shows WHY the agent landed where it
        # did (wrong column, wrong formula, missing step) without forcing a
        # Read of the full trace file. Cap at ~600 chars — long enough to
        # carry the load-bearing chain-of-thought, short enough not to bloat
        # the prompt when there are several failures.
        reasoning_block = ""
        if not is_timeout and getattr(trace, "output", None) is not None:
            reasoning = str(getattr(trace.output, "reasoning", "") or "").strip()
            if reasoning:
                truncated = reasoning[:600]
                if len(reasoning) > 600:
                    truncated += "…"
                reasoning_block = (
                    f"**Agent Reasoning** (excerpt — full version in trace file):\n"
                    f"> {truncated}\n\n"
                )
        skeleton_block = (
            ""
            if is_timeout
            else (
                f"**Tool call skeleton** (Read the full trace file for thinking, text, tool I/O):\n"
                f"{skeleton}\n\n"
            )
        )

        # cost_str was computed up front (timeouts → "n/a (run cut short)"
        # to avoid the misleading $0.0000).
        failure_sections.append(
            f"### {item_label} {i} [Category: {category}]  (turns={turns}, cost={cost_str})\n"
            f"{status_banner}"
            f"**Question**: {question_text}\n"
            f"**Ground Truth**: {str(ground_truth)[:200]}\n"
            f"{agent_answer_block}"
            f"{reasoning_block}"
            f"\n{skeleton_block}"
            f"**Full trace**: `{full_path_ref}`\n"
        )

    failures_text = "\n".join(failure_sections)

    constraints_section = f"\n## Task Constraints\n{task_constraints}\n" if task_constraints else ""

    # Base agent's system prompt — gives the evolver the same priors the base
    # agent runs under (dataset description, file-system rules, answer format,
    # etc.) so evolved skills don't duplicate or contradict it.
    base_prompt_section = (
        "\n## Base agent's system prompt (read this — the base agent already knows the following; do NOT duplicate it in your skill, build on top of it)\n\n"
        f"```\n{solver_prompt.strip()}\n```\n"
        if solver_prompt else ""
    )

    # Runtime context — small note telling the evolver where the dataset
    # lives. The tool-call skeletons below show paths relative to this
    # data_root (e.g. `Read(treasury_bulletins_parsed/foo.txt)`) so the
    # evolver can resolve them if it needs to read a file itself.
    runtime_context_section = (
        f"\n## Runtime context\n- `data_root` = `{data_root}`  (relative paths in tool-call skeletons below resolve against this)\n"
        if data_root else ""
    )

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

    if is_efficiency:
        metrics = phase_metrics or {}
        accuracy = metrics.get("accuracy")
        threshold = metrics.get("accuracy_threshold")
        avg_turns = metrics.get("avg_turns")
        avg_cost = metrics.get("avg_cost_usd")
        metrics_lines = []
        if accuracy is not None and threshold is not None:
            metrics_lines.append(
                f"- Current accuracy: {accuracy:.2f} (threshold: {threshold:.2f})"
            )
        if avg_turns is not None:
            metrics_lines.append(f"- Average turns per sample: {avg_turns:.1f}")
        if avg_cost is not None:
            metrics_lines.append(f"- Average cost per sample: ${avg_cost:.4f}")
        metrics_block = "\n".join(metrics_lines) if metrics_lines else "(metrics unavailable)"

        return f"""{base_prompt_section}{runtime_context_section}## Existing Skills (check before proposing edits)
{skills_list}
{constraints_section}
## Previous Attempts Feedback
{feedback_history}
{runtime_section}
## Phase: Efficiency Optimization

This iteration's {len(traces_with_answers)} training sample(s) all passed. The current best program has reached the configured accuracy threshold — note this is not 100%, so some validation samples may still be failing.

### Current performance
{metrics_block}

## Passing Training Samples ({len(traces_with_answers)} samples across categories: {category_summary})

{failures_text}
{past_traces_section}
## Phase

efficiency — see your system prompt's "Phase: efficiency" section for the full task description.
"""

    return f"""{base_prompt_section}{runtime_context_section}## Existing Skills (check before proposing new ones)
{skills_list}
{constraints_section}
## Previous Attempts Feedback
{feedback_history}
{runtime_section}
## {section_title} ({len(traces_with_answers)} samples across categories: {category_summary})

{failures_text}
{past_traces_section}
## Phase

accuracy — see your system prompt's "Phase: accuracy" section for the full task description.
"""


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


_GIT_CONFLICT_MARKERS = ("<<<<<<<", "=======", ">>>>>>>")


def _strip_git_conflict_markers(text: str) -> tuple[str, bool]:
    """Strip git merge conflict markers from text.

    When `feedback_history.md` collides with a stash-pop during the loop's
    branch switching, unresolved conflict markers can end up inside the file.
    Without this strip those literal markers reach the evolver as part of
    its prompt and look like real "iter-skill-N" entries to the model,
    biasing its output.

    Strategy: prefer the "ours" side (above `=======`) of each conflict
    block, drop the "theirs" side. This biases toward the most-recent
    feedback the loop actually wrote (rather than whatever was stashed).

    Returns: (cleaned_text, had_markers).
    """
    if not any(m in text for m in _GIT_CONFLICT_MARKERS):
        return text, False
    out_lines: list[str] = []
    in_theirs = False
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("<<<<<<<"):
            # Start of "ours" — keep its body
            in_theirs = False
            continue
        if stripped.startswith("=======") and (line.strip() == "=======" or line.startswith("=======")):
            # Switch to "theirs" — drop body until end-marker
            in_theirs = True
            continue
        if stripped.startswith(">>>>>>>"):
            # End of conflict block
            in_theirs = False
            continue
        if not in_theirs:
            out_lines.append(line)
    return "\n".join(out_lines), True


def read_feedback_history(path: Path) -> str:
    """Read feedback history, stripping git conflict markers if present.

    Args:
        path: Path to the feedback history file.

    Returns:
        Contents of feedback file (cleaned of any merge markers) or a
        default message when the file is absent.
    """
    if not path.exists():
        return "No previous attempts."
    raw = path.read_text()
    cleaned, had_markers = _strip_git_conflict_markers(raw)
    if had_markers:
        # Loud warning so the operator can investigate the underlying
        # branch-switching race that created the conflict.
        import warnings
        warnings.warn(
            f"Git merge conflict markers detected in {path}; stripped before "
            "passing to evolver. Investigate the branch-switching path that "
            "wrote the file mid-stash-pop.",
            stacklevel=2,
        )
    return cleaned


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
