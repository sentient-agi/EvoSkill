"""Generate post-run terminal summary and markdown report."""

from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SkillEntry:
    name: str
    iteration: int
    score_delta: float
    action: str = 'create'


@dataclass
class RunReport:
    baseline_score: float
    final_score: float
    iterations_completed: int
    best_program: str
    rows: list[dict]
    skills_kept: list[SkillEntry]
    skills_proposed: int
    project_root: Path = field(default_factory=Path.cwd)
    total_cost_usd: float = 0.0

    @property
    def improvement(self) -> float:
        return self.final_score - self.baseline_score

    @property
    def evoskill_dir(self) -> Path:
        return self.project_root / '.evoskill'

    def print_summary(self) -> None:
        sign = '+' if self.improvement >= 0 else ''
        lines = [
            '',
            '  EvoSkill — Run Complete',
            f'  Baseline: {self.baseline_score:.1%} → Final: {self.final_score:.1%}',
            f'  ({sign}{self.improvement:.1%})',
            f'  Iterations: {self.iterations_completed}  |  Skills kept: {len(self.skills_kept)} of {self.skills_proposed} proposed',
            f'  Total cost: ${self.total_cost_usd:.4f}',
        ]
        if self.skills_kept:
            lines.append('  Skills (by accuracy impact):')
            for i, sk in enumerate(sorted(self.skills_kept, key=lambda s: s.score_delta, reverse=True), 1):
                sign_sk = '+' if sk.score_delta >= 0 else ''
                label = 'edited' if sk.action == 'edit' else 'created'
                lines.append(f'   {i}. {sk.name:<40} {label} {sign_sk}{sk.score_delta:.1%}')
        for line in lines:
            print(line)

    def save(self) -> Path:
        reports_dir = self.evoskill_dir / 'reports'
        reports_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')
        path = reports_dir / f'run-{timestamp}.md'
        path.write_text(self._render_markdown(), encoding='utf-8')
        return path

    def _render_markdown(self) -> str:
        sign = '+' if self.improvement >= 0 else ''
        ts = datetime.now().strftime('%Y-%m-%d %H:%M')
        lines = [
            f'# EvoSkill Run Report — {ts}',
            '',
            '## Summary',
            '',
            '| | |',
            '|---|---|',
            f'| Baseline | {self.baseline_score:.1%} |',
            f'| Final | {self.final_score:.1%} |',
            f'| Improvement | {sign}{self.improvement:.1%} |',
            f'| Iterations | {self.iterations_completed} |',
            f'| Skills kept | {len(self.skills_kept)} of {self.skills_proposed} proposed |',
            f'| Best program | `{self.best_program}` |',
            f'| Total cost | ${self.total_cost_usd:.4f} |',
            '',
            '## Iteration Log',
            '',
            '| Iter | Accuracy | Δ | Skills | Status |',
            '|------|----------|---|--------|--------|',
        ]
        for row in self.rows:
            delta_str = '—'
            if row.get('delta') is not None:
                sign_d = '+' if row['delta'] >= 0 else ''
                delta_str = f'{sign_d}{row["delta"]:.1%}'
            lines.append(f'| {row["iter"]} | {row["score"]:.1%} | {delta_str} | {row["n_skills"]} | {row["status"]} |')

        lines += ['', '## Skills', '']
        for sk in sorted(self.skills_kept, key=lambda s: s.score_delta, reverse=True):
            sign_sk = '+' if sk.score_delta >= 0 else ''
            action_label = 'Edited' if sk.action == 'edit' else 'Created'
            lines.append(f'### {sk.name}')
            lines.append(f'- {action_label} at iteration {sk.iteration}')
            lines.append(f'- Accuracy impact: **{sign_sk}{sk.score_delta:.1%}**')
            lines.append('')
            skill_path = self.project_root / '.claude' / 'skills' / sk.name / 'SKILL.md'
            if skill_path.exists():
                lines.append('```markdown')
                lines.append(skill_path.read_text().strip())
                lines.append('```')
            lines.append('')

        return '\n'.join(lines)
