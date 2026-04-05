"""evoskill init — interactive project setup."""

from pathlib import Path

import click
import questionary
import tomli_w

EVOSKILL_DIR = '.evoskill'

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
    config = {
        'harness': {
            'name': answers['harness'],
            'model': 'sonnet',
            'data_dirs': answers['data_dirs'],
        },
        'evolution': {
            'mode': answers['mode'],
            'iterations': 20,
            'frontier_size': 3,
            'concurrency': 4,
            'no_improvement_limit': 5,
        },
        'dataset': {
            'path': answers['dataset_path'],
            'question_column': answers['question_col'],
            'ground_truth_column': answers['gt_col'],
        },
        'scorer': {
            'type': 'multi_tolerance',
        },
    }
    if answers['category_col']:
        config['dataset']['category_column'] = answers['category_col']
    config['dataset']['train_ratio'] = 0.18
    config['dataset']['val_ratio'] = 0.12
    with open(path, 'wb') as f:
        tomli_w.dump(config, f)


@click.command('init')
def init_cmd():
    """Initialize a new EvoSkill project in the current directory."""
    cwd = Path.cwd()
    evoskill_dir = cwd / EVOSKILL_DIR

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
        default='claude',
    ).ask()

    mode = questionary.select(
        'Evolution mode?',
        choices=[
            questionary.Choice('skill_only  — agent learns new skills (recommended)', value='skill_only'),
            questionary.Choice('prompt_only — rewrites the base prompt instead', value='prompt_only'),
        ],
    ).ask()

    dataset_path = questionary.text('Dataset path?', default='./data/questions.csv').ask()
    question_col = questionary.text('Question column name?', default='question').ask()
    gt_col = questionary.text('Ground truth column name?', default='ground_truth').ask()
    category_col = questionary.text(
        'Category column name? (for stratified sampling, leave blank if none)', default=''
    ).ask()
    data_dirs_raw = questionary.text(
        'Extra data directories for the agent? (comma-separated paths, leave blank if none)', default=''
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
    click.echo(f'    → Fill in {cwd}/task.md  (task description + constraints)')
    click.echo(f'    → Place your dataset at: {dataset_path}')
    click.echo(f'    → Run: evoskill run')
