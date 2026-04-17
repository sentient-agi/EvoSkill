"""SQLite-backed trace storage for cross-iteration learning.

Traces are persisted to both SQLite (for querying) and individual markdown
files (for the evolver to Read on demand via progressive disclosure).
"""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class TraceDB:
    """Persists solver traces so the evolver can recall past attempts."""

    def __init__(self, db_path: Path | str):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._traces_dir = self._path.parent / "traces"
        self._traces_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS traces (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                iteration       TEXT    NOT NULL,
                question        TEXT    NOT NULL,
                ground_truth    TEXT    NOT NULL,
                agent_answer    TEXT    NOT NULL,
                score           REAL    NOT NULL,
                trace_summary   TEXT    NOT NULL,
                trace_file      TEXT    NOT NULL DEFAULT '',
                active_skills   TEXT    NOT NULL DEFAULT '[]',
                num_turns       INTEGER NOT NULL DEFAULT 0,
                category        TEXT    NOT NULL DEFAULT '',
                phase           TEXT    NOT NULL DEFAULT 'train',
                created_at      TEXT    NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_traces_question ON traces(question);
            CREATE INDEX IF NOT EXISTS idx_traces_iteration ON traces(iteration);
        """)
        # Migration: add columns if missing (for existing DBs)
        for col, typedef in [("trace_file", "TEXT NOT NULL DEFAULT ''"),
                             ("num_turns", "INTEGER NOT NULL DEFAULT 0")]:
            try:
                self._conn.execute(f"SELECT {col} FROM traces LIMIT 1")
            except sqlite3.OperationalError:
                self._conn.execute(f"ALTER TABLE traces ADD COLUMN {col} {typedef}")
        self._conn.commit()

    @staticmethod
    def _question_hash(question: str) -> str:
        return hashlib.sha256(question.encode()).hexdigest()[:8]

    def insert(
        self,
        iteration: str,
        question: str,
        ground_truth: str,
        agent_answer: str,
        score: float,
        trace_summary: str,
        active_skills: list[str] | None = None,
        active_skill_contents: dict[str, str] | None = None,
        num_turns: int = 0,
        category: str = "",
        phase: str = "train",
    ) -> None:
        # Write trace to individual file for Read-on-demand
        q_hash = self._question_hash(question)
        trace_filename = f"{iteration}_{q_hash}.md"
        trace_file = self._traces_dir / trace_filename
        skills_list = active_skills or []
        skill_contents = active_skill_contents or {}

        # Embed the skill content AS IT WAS at this iteration, so a future
        # evolver reading this trace has the exact context the solver had.
        # Skills evolve across iterations — "what was the agent guided by?"
        # cannot be answered from the current disk state.
        skill_snapshot_section = ""
        if skill_contents:
            parts = ["## Active Skill Snapshots (exact content at this iteration)\n"]
            for name in sorted(skill_contents.keys()):
                parts.append(f"### Skill: {name}\n")
                parts.append("```markdown")
                parts.append(skill_contents[name].rstrip())
                parts.append("```\n")
            skill_snapshot_section = "\n".join(parts) + "\n"

        file_content = (
            f"# Trace: {iteration}\n\n"
            f"**Question**: {question}\n"
            f"**Ground Truth**: {ground_truth}\n"
            f"**Agent Answer**: {agent_answer}\n"
            f"**Score**: {score:.4f}\n"
            f"**Turns**: {num_turns}\n"
            f"**Active Skills**: {', '.join(skills_list) or 'none'}\n"
            f"**Category**: {category}\n\n"
            f"{skill_snapshot_section}"
            f"## Full Execution Trace\n\n"
            f"{trace_summary}\n"
        )
        trace_file.write_text(file_content)

        # Persist to SQLite
        self._conn.execute(
            """INSERT INTO traces
               (iteration, question, ground_truth, agent_answer, score,
                trace_summary, trace_file, active_skills, num_turns,
                category, phase, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                iteration,
                question,
                ground_truth,
                agent_answer,
                score,
                trace_summary,
                str(trace_file),
                json.dumps(skills_list),
                num_turns,
                category,
                phase,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    def query_by_question(self, question: str, limit: int = 10) -> list[dict]:
        """Exact match on question text, most recent first."""
        rows = self._conn.execute(
            """SELECT * FROM traces
               WHERE question = ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (question, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_traces(self, limit: int = 100) -> list[dict]:
        """Get all traces, most recent first."""
        rows = self._conn.execute(
            """SELECT * FROM traces
               ORDER BY created_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def generate_index(
        self,
        failed_questions: list[str] | None = None,  # accepted for backward compat; now unused
        limit: int = 200,
    ) -> str:
        """Generate a comprehensive markdown index of ALL past traces.

        Shows every persisted trace (success + failure, across all iterations
        and questions), grouped by question for readability. Each row
        references the per-trace .md file — the evolver reads those on
        demand via the Read tool.

        The trace files include the EXACT skill contents that were active
        at the moment of each trace, so the evolver can reason about
        "what was the solver guided by at iter-3?" even if skills have
        since evolved.
        """
        rows = self.get_all_traces(limit=limit)
        if not rows:
            return ""

        # Sort: group by question, then chronological (iter number ascending)
        def iter_sort_key(r):
            iter_name = r.get("iteration", "")
            # "iter-skill-3" -> 3, "iter-3" -> 3, "iter-1" -> 1
            parts = iter_name.rsplit("-", 1)
            try:
                return (r["question"], int(parts[-1]))
            except (ValueError, KeyError):
                return (r["question"], 0)

        rows_sorted = sorted(rows, key=iter_sort_key)

        by_question: dict[str, list[dict]] = {}
        for row in rows_sorted:
            q_short = row["question"][:80]
            by_question.setdefault(q_short, []).append(row)

        lines = [
            "## Past Traces Index — ALL iterations, ALL questions\n",
            "Each row links to a .md file containing the full turn-by-turn transcript",
            "AND the exact skill contents that were active at that iteration.",
            "Use the Read tool on any file below to get the complete context.\n",
            f"Total traces recorded: **{len(rows)}** across {len(by_question)} questions.\n",
        ]

        for q_short, traces in by_question.items():
            lines.append(f"### Question: {q_short}...")
            lines.append("| Iteration | Score | Turns | Active Skills | File |")
            lines.append("|-----------|-------|-------|---------------|------|")
            for t in traces:
                skills = json.loads(t["active_skills"])
                skills_str = ", ".join(skills) if skills else "—"
                lines.append(
                    f"| {t['iteration']} | {t['score']:.2f} | {t['num_turns']} "
                    f"| {skills_str} | `{t['trace_file']}` |"
                )
            lines.append("")

        return "\n".join(lines)

    def close(self) -> None:
        self._conn.close()
