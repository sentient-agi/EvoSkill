"""evoskill init — interactive project setup."""

import json
import tomllib
from pathlib import Path

import click
from src.harness.model_aliases import default_model_for_harness

EVOSKILL_DIR = '.evoskill'
DEFAULT_CONFIG = {
    'harness': {
        'name': 'claude',
        'model': default_model_for_harness('claude'),
        'data_dirs': [],
        'timeout_seconds': 1200,
        'max_retries': 3,
    },
    'evolution': {
        'mode': 'skill_unified',
        'iterations': 20,
        'frontier_size': 3,
        'concurrency': 4,
        'no_improvement_limit': 5,
        'failure_samples': 3,
        'accuracy_threshold': None,
        'evolver_model': None,
    },
    'dataset': {
        'path': './data/questions.csv',
        'question_column': 'question',
        'ground_truth_column': 'ground_truth',
        'train_ratio': 0.18,
        'val_ratio': 0.12,
    },
    'scorer': {
        'type': 'multi_tolerance',
    },
}

TASK_MD_TEMPLATE = (
    '# Task\n\n'
    'Describe what your agent should do here.\n'
    'This is the prompt your agent receives at runtime.\n\n'
    '## Examples\n'
    '- "Example question?" → "Example answer"\n\n'
    '## Output format\n'
    'Return only the answer, no explanation.\n\n'
    '---\n\n'
    '# Constraints\n\n'
    '- Must answers include units? → yes\n'
    '- Should the agent use external APIs? → no\n'
)


def _require_non_empty(value: str) -> bool | str:
    if value.strip():
        return True
    return 'This field is required.'


def _format_toml_value(value: object) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, int | float):
        return str(value)
    raise TypeError(f'Unsupported TOML value: {type(value)!r}')


def _append_toml_field(lines: list[str], comment: str, key: str, value: object) -> None:
    lines.append(f'# {comment}')
    if value is None:
        # TOML has no null. Render the field commented out so users see the
        # key (and can uncomment to enable) without it being interpreted.
        lines.append(f'# {key} =')
    else:
        lines.append(f'{key} = {_format_toml_value(value)}')


def _append_toml_list_field(lines: list[str], comment: str, key: str, values: list[str]) -> None:
    lines.append(f'# {comment}')
    if not values:
        lines.append(f'{key} = []')
        return

    lines.append(f'{key} = [')
    for value in values:
        lines.append(f'    {_format_toml_value(value)},')
    lines.append(']')


def _render_config(config: dict) -> str:
    lines: list[str] = [
        '# EvoSkill project configuration.',
        '# Edit these defaults if your task or runtime needs different behavior.',
        '',
        '[harness]',
    ]
    _append_toml_field(lines, 'Agent runtime used to execute EvoSkill runs.', 'name', config['harness']['name'])
    lines.append('')
    _append_toml_field(lines, 'Default model for the selected runtime.', 'model', config['harness']['model'])
    lines.append('')
    _append_toml_list_field(lines, 'Additional folders the agent can interact with during runs.', 'data_dirs', config['harness']['data_dirs'])
    lines.append('')
    _append_toml_field(lines, 'Maximum time allowed for one agent attempt, in seconds.', 'timeout_seconds', config['harness']['timeout_seconds'])
    lines.append('')
    _append_toml_field(lines, 'How many times to retry a failed or timed-out agent attempt.', 'max_retries', config['harness']['max_retries'])
    lines.extend([
        '',
        '[evolution]',
    ])
    _append_toml_field(lines, 'What EvoSkill is allowed to optimize: skills or the base prompt.', 'mode', config['evolution']['mode'])
    lines.append('')
    _append_toml_field(lines, 'Maximum number of improvement iterations to run.', 'iterations', config['evolution']['iterations'])
    lines.append('')
    _append_toml_field(lines, 'How many top programs to keep in the frontier at once.', 'frontier_size', config['evolution']['frontier_size'])
    lines.append('')
    _append_toml_field(lines, 'How many evaluations can run in parallel.', 'concurrency', config['evolution']['concurrency'])
    lines.append('')
    _append_toml_field(lines, 'Stop after this many iterations with no improvement.', 'no_improvement_limit', config['evolution']['no_improvement_limit'])
    lines.append('')
    _append_toml_field(lines, 'How many failing examples to sample when proposing changes.', 'failure_samples', config['evolution']['failure_samples'])
    lines.append('')
    _append_toml_field(lines, 'Switch to efficiency optimization once frontier accuracy reaches this value (e.g. 0.8). Leave unset to disable Phase 2.', 'accuracy_threshold', config['evolution']['accuracy_threshold'])
    lines.append('')
    _append_toml_field(lines, "Override the model used by evolver agents. Leave unset to inherit harness.model.", 'evolver_model', config['evolution']['evolver_model'])
    lines.extend([
        '',
        '[dataset]',
    ])
    _append_toml_field(lines, 'Path to the dataset CSV. Relative paths are resolved from .evoskill/.', 'path', config['dataset']['path'])
    lines.append('')
    _append_toml_field(lines, 'CSV column containing the question or task input.', 'question_column', config['dataset']['question_column'])
    lines.append('')
    _append_toml_field(lines, 'CSV column containing the expected answer.', 'ground_truth_column', config['dataset']['ground_truth_column'])
    lines.append('')
    _append_toml_field(lines, 'CSV column used to group or stratify examples.', 'category_column', config['dataset']['category_column'])
    lines.append('')
    _append_toml_field(lines, 'Fraction of each category used for training samples.', 'train_ratio', config['dataset']['train_ratio'])
    lines.append('')
    _append_toml_field(lines, 'Fraction of each category used for validation samples.', 'val_ratio', config['dataset']['val_ratio'])
    lines.extend([
        '',
        '[scorer]',
    ])
    _append_toml_field(lines, 'Scoring rule used to compare predictions against ground truth.', 'type', config['scorer']['type'])
    return '\n'.join(lines) + '\n'


def _write_config(path: Path, answers: dict) -> None:
    config = {
        'harness': dict(DEFAULT_CONFIG['harness']),
        'evolution': dict(DEFAULT_CONFIG['evolution']),
        'dataset': dict(DEFAULT_CONFIG['dataset']),
        'scorer': dict(DEFAULT_CONFIG['scorer']),
    }
    config['harness']['name'] = answers['harness']
    config['harness']['model'] = default_model_for_harness(answers['harness'])
    config['harness']['data_dirs'] = answers['data_dirs']
    config['evolution']['mode'] = answers.get('mode', DEFAULT_CONFIG['evolution']['mode'])
    config['evolution']['accuracy_threshold'] = answers.get('accuracy_threshold')
    config['evolution']['evolver_model'] = answers.get('evolver_model')
    config['dataset']['path'] = answers['dataset_path']
    config['dataset']['question_column'] = answers['question_col']
    config['dataset']['ground_truth_column'] = answers['gt_col']
    config['dataset']['category_column'] = answers['category_col']

    path.write_text(_render_config(config), encoding='utf-8')


def _load_prompt_defaults(config_path: Path) -> dict[str, str]:
    defaults = {
        'harness': DEFAULT_CONFIG['harness']['name'],
        'dataset_path': DEFAULT_CONFIG['dataset']['path'],
        'question_col': DEFAULT_CONFIG['dataset']['question_column'],
        'gt_col': DEFAULT_CONFIG['dataset']['ground_truth_column'],
        'category_col': 'category',
        'data_dirs_raw': '',
        'mode': DEFAULT_CONFIG['evolution']['mode'],
        'accuracy_threshold_raw': '',
        'evolver_model_raw': '',
    }
    if not config_path.exists():
        return defaults

    with open(config_path, 'rb') as f:
        raw = tomllib.load(f)

    harness = raw.get('harness', {})
    dataset = raw.get('dataset', {})
    evolution = raw.get('evolution', {})

    defaults['harness'] = harness.get('name', defaults['harness'])
    defaults['dataset_path'] = dataset.get('path', defaults['dataset_path'])
    defaults['mode'] = evolution.get('mode', defaults['mode'])
    if evolution.get('accuracy_threshold') is not None:
        defaults['accuracy_threshold_raw'] = str(evolution['accuracy_threshold'])
    if evolution.get('evolver_model'):
        defaults['evolver_model_raw'] = str(evolution['evolver_model'])
    defaults['question_col'] = dataset.get('question_column', defaults['question_col'])
    defaults['gt_col'] = dataset.get('ground_truth_column', defaults['gt_col'])
    defaults['category_col'] = dataset.get('category_column', defaults['category_col']) or defaults['category_col']
    defaults['data_dirs_raw'] = ','.join(harness.get('data_dirs', []))
    return defaults


@click.command('init')
def init_cmd():
    """Initialize a new EvoSkill project in the current directory."""
    import questionary

    cwd = Path.cwd()
    evoskill_dir = cwd / EVOSKILL_DIR
    config_path = evoskill_dir / 'config.toml'
    prompt_defaults = _load_prompt_defaults(config_path)

    if evoskill_dir.exists():
        overwrite = questionary.confirm(
            f'  {cwd}/{EVOSKILL_DIR}/ already exists. Reinitialize? (config.toml and task.md will be overwritten)',
            default=False,
        ).ask()
        if not overwrite:
            click.echo('  Aborted.')
            return

    click.echo('  EvoSkill — Project Setup')

    harness = questionary.select(
        'Which agent runtime do you want to use?',
        choices=['claude', 'opencode', 'codex', 'goose', 'openhands'],
        default=prompt_defaults['harness'],
    ).ask()

    dataset_path = questionary.text(
        'Path to the dataset CSV?',
        default=prompt_defaults['dataset_path'],
        validate=_require_non_empty,
    ).ask()
    question_col = questionary.text(
        'Question/input column name?',
        default=prompt_defaults['question_col'],
        validate=_require_non_empty,
    ).ask()
    gt_col = questionary.text(
        'Answer column name?',
        default=prompt_defaults['gt_col'],
        validate=_require_non_empty,
    ).ask()
    category_col = questionary.text(
        'Category/difficulty column name?',
        default=prompt_defaults['category_col'],
        validate=_require_non_empty,
    ).ask()
    data_dirs_raw = questionary.text(
        'Additional folders the agent can interact with? (comma-separated paths, leave blank if none)',
        default=prompt_defaults['data_dirs_raw'],
    ).ask()

    mode = questionary.select(
        'Evolution mode?',
        choices=[
            questionary.Choice(
                'skill_unified — single evolver agent proposes + writes the skill in one pass (recommended)',
                value='skill_unified',
            ),
            questionary.Choice(
                'skill_only — split proposer/generator, evolves skills only (legacy)',
                value='skill_only',
            ),
            questionary.Choice(
                'prompt_only — evolves the base prompt instead of skills',
                value='prompt_only',
            ),
        ],
        default=prompt_defaults['mode'],
    ).ask()

    accuracy_threshold_raw = questionary.text(
        'Accuracy threshold to enable Phase 2 efficiency optimization (e.g. 0.8). Leave blank to disable.',
        default=prompt_defaults['accuracy_threshold_raw'],
    ).ask()

    evolver_model_raw = questionary.text(
        'Override model for evolver agents? Leave blank to inherit harness.model.',
        default=prompt_defaults['evolver_model_raw'],
    ).ask()

    required = [harness, dataset_path, question_col, gt_col, category_col, data_dirs_raw, mode]
    if any(v is None for v in required) or any(v is None for v in [accuracy_threshold_raw, evolver_model_raw]):
        click.echo('\n  Aborted.')
        raise SystemExit(1)

    # Parse optional inputs. Empty / "none" → unset (None), so config.toml renders the field commented out.
    accuracy_threshold: float | None = None
    if accuracy_threshold_raw.strip():
        try:
            accuracy_threshold = float(accuracy_threshold_raw)
        except ValueError:
            click.echo(f'  ⚠️  Could not parse "{accuracy_threshold_raw}" as a number — Phase 2 left disabled.')
    evolver_model: str | None = evolver_model_raw.strip() or None

    data_dirs = [v.strip() for v in data_dirs_raw.split(',') if v.strip()]

    for subdir in ('data', 'skills', 'frontier', 'reports', 'logs', 'prompts'):
        d = evoskill_dir / subdir
        d.mkdir(parents=True, exist_ok=True)

    _write_config(
        evoskill_dir / 'config.toml',
        {
            'harness': harness,
            'dataset_path': dataset_path,
            'question_col': question_col,
            'gt_col': gt_col,
            'category_col': category_col,
            'data_dirs': data_dirs,
            'mode': mode,
            'accuracy_threshold': accuracy_threshold,
            'evolver_model': evolver_model,
        },
    )
    (evoskill_dir / 'task.md').write_text(TASK_MD_TEMPLATE)

    click.echo(f'  ✓ Created {cwd}/{EVOSKILL_DIR}')
    click.echo('')
    click.echo(f'    Runtime: {harness}')
    click.echo(f'    Dataset: {dataset_path}')
    click.echo('    Columns:')
    click.echo(f'      question: {question_col}')
    click.echo(f'      answer: {gt_col}')
    click.echo(f'      category: {category_col}')
    click.echo(f'    Extra data dirs: {", ".join(data_dirs) if data_dirs else "none"}')
    click.echo('')
    click.echo('    Next:')
    click.echo(f'      1. Fill in {cwd}/{EVOSKILL_DIR}/task.md')
    click.echo(f'      2. Review {cwd}/{EVOSKILL_DIR}/config.toml if needed')
    click.echo('      3. Run: evoskill run')
