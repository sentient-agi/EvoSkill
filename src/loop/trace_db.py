"""SQLite-backed trace storage for cross-iteration learning.

Traces are persisted to both SQLite (for querying) and individual markdown
files (for the evolver to Read on demand via progressive disclosure).
"""

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _iter_num(iteration: str) -> int:
    """Parse the numeric suffix from iteration names like 'iter-3', 'iter-skill-7'."""
    parts = iteration.rsplit("-", 1)
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return 0


def _select_representative_rows(rows: list[dict], max_n: int) -> list[dict]:
    """Pick up to max_n rows: first, latest, best, worst, biggest jump, then fill gaps.

    Input rows must already be sorted chronologically (ascending).
    Returns rows in chronological order (for readability in the table).
    """
    if len(rows) <= max_n:
        return list(rows)

    picked: set[int] = {0, len(rows) - 1}  # first + latest

    # Best by score
    picked.add(max(range(len(rows)), key=lambda i: rows[i]["score"]))

    # Worst score (excluding the first baseline iteration if possible)
    if len(rows) > 2:
        picked.add(min(range(1, len(rows)), key=lambda i: rows[i]["score"]))
    else:
        picked.add(min(range(len(rows)), key=lambda i: rows[i]["score"]))

    # Biggest absolute score change between consecutive iterations
    if len(rows) > 1:
        jumps = [
            (abs(rows[i]["score"] - rows[i - 1]["score"]), i)
            for i in range(1, len(rows))
        ]
        _, jump_idx = max(jumps)
        picked.add(jump_idx)

    # Fill any remaining budget by widening gaps
    while len(picked) < max_n and len(picked) < len(rows):
        sorted_picked = sorted(picked)
        # Find the largest gap between consecutive picked indices
        biggest_gap = 0
        mid = None
        for i in range(len(sorted_picked) - 1):
            gap = sorted_picked[i + 1] - sorted_picked[i]
            if gap > biggest_gap:
                biggest_gap = gap
                mid = (sorted_picked[i] + sorted_picked[i + 1]) // 2
        if mid is None or mid in picked:
            break
        picked.add(mid)

    return [rows[i] for i in sorted(picked)]


class TraceDB:
    """Persists solver traces so the evolver can recall past attempts."""

    def __init__(self, db_path: Path | str):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._traces_dir = self._path.parent / "traces"
        self._traces_dir.mkdir(parents=True, exist_ok=True)
        # Skill snapshots are content-addressed — one file per unique (skill, content)
        self._snapshots_dir = self._path.parent / "skill_snapshots"
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_schema()

    def _snapshot_skill(self, skill_name: str, content: str) -> tuple[Path, str]:
        """Content-addressed skill snapshot — dedup across iterations.

        Returns (snapshot_file_path, short_content_hash). Writes the file
        only if a snapshot with the same hash doesn't already exist.
        """
        content_hash = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:10]
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in skill_name)
        snapshot_file = self._snapshots_dir / f"{safe_name}_{content_hash}.md"
        if not snapshot_file.exists():
            snapshot_file.write_text(content)
        return snapshot_file, content_hash

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

        # Snapshot each skill's content, deduplicated by content hash.
        # Multiple iterations using the same skill content share one file.
        # The trace file references the snapshot path — evolver Reads on demand.
        skill_snapshot_section = ""
        if skill_contents:
            parts = [
                "## Active Skill Snapshots\n",
                "Content at this iteration — snapshots are deduplicated by content hash.",
                "If two iterations reference the same file, the skill was identical between them.\n",
                "| Skill | Hash | Snapshot file |",
                "|-------|------|---------------|",
            ]
            for name in sorted(skill_contents.keys()):
                snap_path, content_hash = self._snapshot_skill(name, skill_contents[name])
                parts.append(f"| {name} | `{content_hash}` | `{snap_path}` |")
            parts.append("")
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
        failed_questions: list[str] | None = None,  # accepted for backward compat; unused
        limit: int = 500,
        per_question_cap: int = 5,
    ) -> str:
        """Compact index of past traces with trajectory + priority selection.

        For each question:
          - One-line score trajectory (all iterations, most compact signal)
          - A table of AT MOST `per_question_cap` representative rows
            selected by priority: first, latest, best, worst, biggest jump
          - Each row references the per-trace .md file (Read on demand)

        Trace files contain the full transcript AND reference-by-path the
        exact skill snapshots active at that iteration.

        Args:
            limit: Max total traces to pull from DB.
            per_question_cap: Max rows shown per question (others summarized
                in the trajectory line).
        """
        rows = self.get_all_traces(limit=limit)
        if not rows:
            return ""

        by_question: dict[str, list[dict]] = {}
        for row in rows:
            by_question.setdefault(row["question"], []).append(row)

        # Sort each question's rows chronologically (by iter number)
        for q in by_question:
            by_question[q].sort(key=lambda r: _iter_num(r["iteration"]))

        lines = [
            "## Past Traces Index — ALL iterations, ALL questions\n",
            "Each question shows its score trajectory plus a few representative iterations.",
            "Each trace file contains the full turn-by-turn transcript AND references to",
            "the exact skill snapshots active at that iteration (Read the trace file, then",
            "follow the snapshot paths to see skill content as of that moment).\n",
            f"Total traces recorded: **{len(rows)}** across {len(by_question)} questions.",
            f"Per-question row cap: {per_question_cap} (trajectory line shows all iterations).\n",
        ]

        for question, q_rows in by_question.items():
            q_short = question[:80]
            lines.append(f"### Question: {q_short}...")

            # B. Trajectory line — one-line summary of ALL iterations for this question
            traj_parts = [f"{r['iteration']}:{r['score']:.2f}" for r in q_rows]
            best_row = max(q_rows, key=lambda r: r["score"])
            lines.append(
                f"**Trajectory** ({len(q_rows)} iter{'s' if len(q_rows) != 1 else ''}): "
                + " → ".join(traj_parts)
                + f"  — best: {best_row['iteration']} @ {best_row['score']:.2f}"
            )
            lines.append("")

            # A. Priority selection — pick representative rows if above cap
            selected = (
                q_rows
                if len(q_rows) <= per_question_cap
                else _select_representative_rows(q_rows, per_question_cap)
            )

            # Table with Δ (delta vs previous iteration in the FULL list)
            lines.append("| Iteration | Score | Δ | Turns | Active Skills | File |")
            lines.append("|-----------|-------|---|-------|---------------|------|")
            idx_in_full = {r["iteration"]: i for i, r in enumerate(q_rows)}
            for r in selected:
                i = idx_in_full.get(r["iteration"], 0)
                if i > 0:
                    d = r["score"] - q_rows[i - 1]["score"]
                    delta = f"{d:+.2f}"
                else:
                    delta = "—"
                skills = json.loads(r["active_skills"])
                skills_str = ", ".join(skills) if skills else "—"
                lines.append(
                    f"| {r['iteration']} | {r['score']:.2f} | {delta} | {r['num_turns']} "
                    f"| {skills_str} | `{r['trace_file']}` |"
                )
            if len(q_rows) > per_question_cap:
                lines.append(
                    f"\n_Showing {len(selected)}/{len(q_rows)} iterations "
                    f"(first/last/best/worst/biggest-jump). Trajectory above covers all._"
                )
            lines.append("")

        return "\n".join(lines)

    def close(self) -> None:
        self._conn.close()
