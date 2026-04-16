"""Background reviewer: extracts reusable insights from solver traces.

After each solver run, fires a lightweight Claude Code agent (Haiku) to review
the trace and produce structured proposals for the offline reflector.
Uses Claude Code SDK auth — no separate ANTHROPIC_API_KEY needed.
Proposals are stored in the trace DB for the reflector to consume.
"""

import asyncio
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

REVIEW_MODEL = "haiku"

REVIEW_SYSTEM_PROMPT = """\
You are a trace reviewer for an AI agent optimization system. You analyze solver \
execution traces to extract reusable insights that could improve future runs.

You will be given a solver's full turn-by-turn trace (tool calls, results, reasoning) \
along with the question, answer, ground truth, and score.

Extract ONLY insights that are generalizable across questions. Skip anything specific \
to a single question's answer.

Examples of good insights:
- type=data_path: "Treasury bulletins at /Users/dastin/dev/officeqa/" — Agent found PDFs at this path after 10 turns of searching.
- type=pitfall: "Reading .skill files returns binary garbage" — .skill files are ZIP archives, not text.
- type=approach: "Read PDF ToC first to find report page offsets" — 2-step approach works reliably.
- type=skill_idea: "Systematic chart feature counting protocol" — Segment-by-segment scanning with verification.

If there are no useful insights, leave the proposals array empty."""

REVIEW_USER_TEMPLATE = """\
## Solver Trace to Review

Question: {question}
Ground Truth: {ground_truth}
Agent Answer: {agent_answer}
Score: {score}
Iteration: {iteration}
Active Skills: {active_skills}

## Full Execution Trace
{trace_summary}
"""


class ReviewProposalItem(BaseModel):
    """A single insight from the reviewer."""
    type: str = Field(description="One of: data_path, tool_pattern, approach, pitfall, skill_idea")
    title: str = Field(description="Short description, under 80 chars")
    detail: str = Field(description="Explanation with evidence from the trace, 1-3 sentences")
    priority: str = Field(default="medium", description="high, medium, or low")


class ReviewOutput(BaseModel):
    """Structured output from the background reviewer."""
    proposals: list[ReviewProposalItem] = Field(
        default_factory=list,
        description="List of reusable insights extracted from the trace. Empty if nothing notable.",
    )


@dataclass
class RuntimeProposal:
    """A single insight extracted by the background reviewer."""
    type: str
    title: str
    detail: str
    priority: str
    iteration: str
    question: str
    score: float


class BackgroundReviewer:
    """Reviews solver traces using a lightweight Claude Code agent (Haiku)."""

    def __init__(self, db_path: Path | str, model: str = REVIEW_MODEL):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._model = model
        self._init_schema()

    def _init_schema(self) -> None:
        conn = sqlite3.connect(str(self._db_path))
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS runtime_proposals (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                type        TEXT NOT NULL,
                title       TEXT NOT NULL,
                detail      TEXT NOT NULL,
                priority    TEXT NOT NULL DEFAULT 'medium',
                iteration   TEXT NOT NULL,
                question    TEXT NOT NULL,
                score       REAL NOT NULL,
                consumed    INTEGER NOT NULL DEFAULT 0,
                created_at  TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_proposals_consumed ON runtime_proposals(consumed);
            CREATE INDEX IF NOT EXISTS idx_proposals_priority ON runtime_proposals(priority);
        """)
        conn.commit()
        conn.close()

    def _make_agent(self):
        """Create a lightweight Agent for trace review.

        Uses the harness-agnostic build_options so the reviewer works under any SDK.
        No tools means the agent must respond with structured JSON in a single pass.
        """
        from src.harness import Agent, build_options

        options = build_options(
            system=REVIEW_SYSTEM_PROMPT,
            schema=ReviewOutput.model_json_schema(),
            tools=[],  # No tools — pure text analysis, single turn
            model=self._model,
            permission_mode="plan",  # Read-only (ignored on non-Claude SDKs)
        )
        return Agent(options, ReviewOutput, name="background_reviewer")

    async def review_trace(
        self,
        iteration: str,
        question: str,
        ground_truth: str,
        agent_answer: str,
        score: float,
        trace_summary: str,
        active_skills: list[str] | None = None,
    ) -> list[RuntimeProposal]:
        """Review a single trace and store proposals."""
        user_msg = REVIEW_USER_TEMPLATE.format(
            question=question,
            ground_truth=ground_truth,
            agent_answer=agent_answer,
            score=score,
            iteration=iteration,
            active_skills=", ".join(active_skills or []) or "none",
            trace_summary=trace_summary[:50_000],
        )

        try:
            agent = self._make_agent()
            trace_result = await agent.run(user_msg)
            if trace_result.output is None:
                logger.warning(f"Background review: no structured output (parse_error={trace_result.parse_error})")
                return []
            proposals_raw = trace_result.output.proposals
        except Exception as e:
            logger.warning(f"Background review failed: {e}")
            return []

        # Store proposals
        proposals = []
        conn = sqlite3.connect(str(self._db_path))
        for p in proposals_raw:
            proposal = RuntimeProposal(
                type=p.type,
                title=p.title,
                detail=p.detail,
                priority=p.priority,
                iteration=iteration,
                question=question,
                score=score,
            )
            conn.execute(
                """INSERT INTO runtime_proposals
                   (type, title, detail, priority, iteration, question, score, consumed, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)""",
                (
                    proposal.type,
                    proposal.title,
                    proposal.detail,
                    proposal.priority,
                    proposal.iteration,
                    proposal.question,
                    proposal.score,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            proposals.append(proposal)
        conn.commit()
        conn.close()

        if proposals:
            _titles = ", ".join(p.title[:40] for p in proposals)
            _log_msg = f"Background review: {len(proposals)} proposals — {_titles}"
            logger.info(_log_msg)
            print(f"  [REVIEW] {_log_msg}", flush=True)

        return proposals

    async def review_traces_batch(
        self,
        traces: list[dict],
    ) -> list[RuntimeProposal]:
        """Review multiple traces sequentially (one Claude Code subprocess at a time)."""
        all_proposals = []
        for t in traces:
            try:
                proposals = await self.review_trace(**t)
                all_proposals.extend(proposals)
            except Exception as e:
                logger.warning(f"Background review failed for trace: {e}")
        return all_proposals

    def get_unconsumed_proposals(self, limit: int = 20) -> list[dict]:
        """Get proposals not yet shown to the reflector."""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """SELECT * FROM runtime_proposals
               WHERE consumed = 0
               ORDER BY
                   CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                   created_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        result = [dict(r) for r in rows]
        conn.close()
        return result

    def mark_consumed(self, proposal_ids: list[int]) -> None:
        """Mark proposals as consumed after the reflector has seen them."""
        if not proposal_ids:
            return
        conn = sqlite3.connect(str(self._db_path))
        placeholders = ",".join("?" * len(proposal_ids))
        conn.execute(
            f"UPDATE runtime_proposals SET consumed = 1 WHERE id IN ({placeholders})",
            proposal_ids,
        )
        conn.commit()
        conn.close()

    def format_proposals_for_reflector(self, max_chars: int = 10_000) -> str:
        """Format unconsumed proposals as a section for the reflector query.

        Returns formatted text and marks proposals as consumed.
        """
        proposals = self.get_unconsumed_proposals()
        if not proposals:
            return ""

        sections = []
        total = 0
        consumed_ids = []

        for p in proposals:
            entry = (
                f"- **[{p['type']}] {p['title']}** ({p['priority']} priority, "
                f"from {p['iteration']}, score={p['score']:.2f})\n"
                f"  {p['detail']}\n"
            )
            if total + len(entry) > max_chars:
                break
            sections.append(entry)
            total += len(entry)
            consumed_ids.append(p["id"])

        self.mark_consumed(consumed_ids)
        return "\n".join(sections)
