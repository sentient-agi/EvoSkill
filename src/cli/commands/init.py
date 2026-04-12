"""evoskill init — interactive project setup."""

import tomllib
from pathlib import Path

import click

EVOSKILL_DIR = '.evoskill'
DEFAULT_CONFIG = {
    'harness': {
        'name': 'claude',
        'model': 'sonnet',
        'data_dirs': [],
    },
    'evolution': {
        'mode': 'skill_only',
        'iterations': 20,
        'frontier_size': 3,
        'concurrency': 4,
        'no_improvement_limit': 5,
        'failure_samples': 3,
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


def _write_config(path: Path, answers: dict) -> None:
    import tomli_w

    config = {
        'harness': dict(DEFAULT_CONFIG['harness']),
        'evolution': dict(DEFAULT_CONFIG['evolution']),
        'dataset': dict(DEFAULT_CONFIG['dataset']),
        'scorer': dict(DEFAULT_CONFIG['scorer']),
    }
    config['harness']['name'] = answers['harness']
    config['harness']['data_dirs'] = answers['data_dirs']
    config['evolution']['mode'] = answers['mode']
    config['dataset']['path'] = answers['dataset_path']
    config['dataset']['question_column'] = answers['question_col']
    config['dataset']['ground_truth_column'] = answers['gt_col']
    if answers['category_col']:
        config['dataset']['category_column'] = answers['category_col']
    else:
        config['dataset'].pop('category_column', None)

    with open(path, 'wb') as f:
        tomli_w.dump(config, f)


def _load_prompt_defaults(config_path: Path) -> dict[str, str]:
    defaults = {
        'harness': DEFAULT_CONFIG['harness']['name'],
        'mode': DEFAULT_CONFIG['evolution']['mode'],
        'dataset_path': DEFAULT_CONFIG['dataset']['path'],
        'question_col': DEFAULT_CONFIG['dataset']['question_column'],
        'gt_col': DEFAULT_CONFIG['dataset']['ground_truth_column'],
        'category_col': '',
        'data_dirs_raw': '',
    }
    if not config_path.exists():
        return defaults

    with open(config_path, 'rb') as f:
        raw = tomllib.load(f)

    harness = raw.get('harness', {})
    evolution = raw.get('evolution', {})
    dataset = raw.get('dataset', {})

    defaults['harness'] = harness.get('name', defaults['harness'])
    defaults['mode'] = evolution.get('mode', defaults['mode'])
    defaults['dataset_path'] = dataset.get('path', defaults['dataset_path'])
    defaults['question_col'] = dataset.get('question_column', defaults['question_col'])
    defaults['gt_col'] = dataset.get('ground_truth_column', defaults['gt_col'])
    defaults['category_col'] = dataset.get('category_column', defaults['category_col']) or ''
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
        'Which harness?',
        choices=['claude', 'opencode'],
        default=prompt_defaults['harness'],
    ).ask()

    mode = questionary.select(
        'Evolution mode?',
        choices=[
            questionary.Choice('skill_only  — agent learns new skills (recommended)', value='skill_only'),
            questionary.Choice('prompt_only — rewrites the base prompt instead', value='prompt_only'),
        ],
        default=prompt_defaults['mode'],
    ).ask()

    dataset_path = questionary.text('Dataset path?', default=prompt_defaults['dataset_path']).ask()
    question_col = questionary.text('Question column name?', default=prompt_defaults['question_col']).ask()
    gt_col = questionary.text('Ground truth column name?', default=prompt_defaults['gt_col']).ask()
    category_col = questionary.text(
        'Category column name? (for stratified sampling, leave blank if none)',
        default=prompt_defaults['category_col'],
    ).ask()
    data_dirs_raw = questionary.text(
        'Extra data directories for the agent? (comma-separated paths, leave blank if none)',
        default=prompt_defaults['data_dirs_raw'],
    ).ask()

    if any(v is None for v in [harness, mode, dataset_path, question_col, gt_col, category_col, data_dirs_raw]):
        click.echo('\n  Aborted.')
        raise SystemExit(1)

    data_dirs = [v.strip() for v in data_dirs_raw.split(',') if v.strip()]

    for subdir in ('data', 'skills', 'frontier', 'reports', 'logs', 'prompts'):
        d = evoskill_dir / subdir
        d.mkdir(parents=True, exist_ok=True)

    _write_config(
        evoskill_dir / 'config.toml',
        {
            'harness': harness,
            'mode': mode,
            'dataset_path': dataset_path,
            'question_col': question_col,
            'gt_col': gt_col,
            'category_col': category_col,
            'data_dirs': data_dirs,
        },
    )
    (evoskill_dir / 'task.md').write_text(TASK_MD_TEMPLATE)

    click.echo(f'  ✓ Created {cwd}/{EVOSKILL_DIR}')
    click.echo(f'    → Fill in {cwd}/{EVOSKILL_DIR}/task.md  (task description + constraints)')
    click.echo(f'    → Place your dataset at: {dataset_path}')
    click.echo(f'    → Run: evoskill run')
