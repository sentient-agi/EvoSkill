"""evoskill init — interactive project setup."""

import json
import subprocess
import tomllib
from pathlib import Path

import click
from src.harness.model_aliases import default_model_for_harness

EVOSKILL_DIR = '.evoskill'
DEFAULT_CONFIG = {
    'harness': {
        'name': 'claude',
        'model': default_model_for_harness('claude'),
        'data_dirs': ['/absolute/path/to/data_dir'],
        'timeout_seconds': 1200,
        'max_retries': 3,
    },
    'evolution': {
        'mode': 'skill_only',
        'iterations': 7,
        'frontier_size': 3,
        'concurrency': 4,
        'no_improvement_limit': 5,
        'failure_samples': 3,
    },
    'dataset': {
        'path': '/absolute/path/to/questions.csv',
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
    ]
    if config.get('execution', 'local') != 'local':
        _append_toml_field(lines, 'Execution mode: "local", "docker", or "daytona".', 'execution', config['execution'])
        lines.append('')

    lines.append('[harness]')
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
    lines.extend([
        '',
        '[dataset]',
    ])
    _append_toml_field(lines, 'Absolute path to the dataset CSV.', 'path', config['dataset']['path'])
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

    if 'remote' in config and config['remote']:
        rc = config['remote']
        lines.extend(['', '', '[remote]'])
        _append_toml_field(lines, 'Remote execution backend.', 'target', rc['target'])

        if 'daytona' in rc:
            dc = rc['daytona']
            lines.extend(['', '[remote.daytona]'])
            _append_toml_field(lines, 'Daytona API key. Can also use DAYTONA_API_KEY env var.', 'api_key', dc['api_key'])
            lines.append('')
            _append_toml_field(lines, 'Base Docker image for the sandbox.', 'image', dc['image'])
            lines.append('')
            _append_toml_field(lines, 'vCPUs allocated to the sandbox.', 'cpu', dc['cpu'])
            lines.append('')
            _append_toml_field(lines, 'Memory in GB allocated to the sandbox.', 'memory', dc['memory'])
            lines.append('')
            _append_toml_field(lines, 'Disk in GB allocated to the sandbox.', 'disk', dc['disk'])
            lines.append('')
            _append_toml_field(lines, 'Auto-stop timeout in minutes. 0 = never auto-stop.', 'timeout', dc['timeout'])

        if 'download' in rc:
            dl = rc['download']
            lines.extend(['', '[remote.download]'])
            _append_toml_field(lines, 'Download all program branches, not just the best.', 'all_branches', dl['all_branches'])
            lines.append('')
            _append_toml_field(lines, 'Download evaluation cache to avoid re-running locally.', 'cache', dl['cache'])
            lines.append('')
            _append_toml_field(lines, 'Download run reports.', 'reports', dl['reports'])
            lines.append('')
            _append_toml_field(lines, 'Download feedback history for local continuation.', 'feedback_history', dl['feedback_history'])

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
    config['dataset']['path'] = answers['dataset_path']
    config['dataset']['question_column'] = answers['question_col']
    config['dataset']['ground_truth_column'] = answers['gt_col']
    config['dataset']['category_column'] = answers['category_col']

    if answers.get('execution', 'local') != 'local':
        config['execution'] = answers['execution']
    if answers.get('remote'):
        config['remote'] = answers['remote']

    path.write_text(_render_config(config), encoding='utf-8')


def _load_prompt_defaults(config_path: Path) -> dict[str, str]:
    defaults = {
        'harness': DEFAULT_CONFIG['harness']['name'],
        'dataset_path': DEFAULT_CONFIG['dataset']['path'],
        'question_col': DEFAULT_CONFIG['dataset']['question_column'],
        'gt_col': DEFAULT_CONFIG['dataset']['ground_truth_column'],
        'category_col': 'category',
        'data_dirs_raw': '',
    }
    if not config_path.exists():
        return defaults

    with open(config_path, 'rb') as f:
        raw = tomllib.load(f)

    harness = raw.get('harness', {})
    dataset = raw.get('dataset', {})

    defaults['harness'] = harness.get('name', defaults['harness'])
    defaults['dataset_path'] = dataset.get('path', defaults['dataset_path'])
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

    click.echo('  EvoSkill — Project Setup\n')

    # 1. Agent runtime
    harness = questionary.select(
        'Which agent runtime?',
        choices=['claude', 'opencode', 'codex', 'goose', 'openhands'],
        default=prompt_defaults['harness'],
    ).ask()

    if harness == 'openhands':
        click.echo('\n  Note: OpenHands uses fallback JSON extraction for structured output.')
        click.echo('  Consider Claude, OpenCode, or Goose for more reliable results.\n')

    # 2. Dataset
    dataset_path = questionary.text(
        'Absolute path to dataset CSV?',
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
        'Category column name? (leave blank if none)',
        default=prompt_defaults['category_col'],
    ).ask()
    data_dirs_raw = questionary.text(
        'Additional data directories? (comma-separated absolute paths, blank if none)',
        default=prompt_defaults['data_dirs_raw'],
    ).ask()

    # 3. Execution mode
    exec_mode = questionary.select(
        'How do you want to run EvoSkill?',
        choices=[
            questionary.Choice('Local — run directly on this machine', value='local'),
            questionary.Choice('Docker — run in a container (local or remote via DOCKER_HOST)', value='docker'),
            questionary.Choice('Daytona — run on a managed Daytona sandbox', value='daytona'),
        ],
        default='local',
    ).ask()

    remote_config = None
    if exec_mode == 'daytona':
        import os
        env_key = os.environ.get('DAYTONA_API_KEY', '')
        api_key_input = questionary.text(
            'Daytona API key?',
            default=env_key if env_key else '',
        ).ask()

        image_input = questionary.text(
            'Docker image for sandbox? (build with: docker build -t evoskill .)',
            default='',
        ).ask()

        if api_key_input and api_key_input.strip():
            remote_config = {
                'target': 'daytona',
                'daytona': {
                    'api_key': api_key_input.strip(),
                    'image': image_input.strip() if image_input else '',
                    'cpu': 4,
                    'memory': 8,
                    'disk': 10,
                    'timeout': 60,
                },
                'download': {
                    'all_branches': False,
                    'cache': False,
                    'reports': True,
                    'feedback_history': False,
                },
            }
        else:
            click.echo('  No API key provided. Skipping Daytona setup.')

    if any(v is None for v in [harness, dataset_path, question_col, gt_col, data_dirs_raw, exec_mode]):
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
            'dataset_path': dataset_path,
            'question_col': question_col,
            'gt_col': gt_col,
            'category_col': category_col or '',
            'data_dirs': data_dirs,
            'execution': exec_mode,
            'remote': remote_config,
        },
    )
    (evoskill_dir / 'task.md').write_text(TASK_MD_TEMPLATE)

    # Save init-time state (original branch for reset landing)
    original_branch = 'main'
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=cwd, capture_output=True, text=True, check=True,
        )
        original_branch = result.stdout.strip() or 'main'
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    state = {'original_branch': original_branch}
    (evoskill_dir / 'state.json').write_text(json.dumps(state, indent=2) + '\n')

    click.echo(f'\n  ✓ Created {cwd}/{EVOSKILL_DIR}')
    click.echo(f'    Runtime:   {harness}')
    click.echo(f'    Dataset:   {dataset_path}')
    click.echo(f'    Columns:   {question_col}, {gt_col}' + (f', {category_col}' if category_col else ''))
    click.echo(f'    Data dirs: {", ".join(data_dirs) if data_dirs else "none"}')
    click.echo(f'    Execution: {exec_mode}')
    if remote_config:
        click.echo(f'    Image:     {remote_config["daytona"].get("image") or "(not set — build and push first)"}')
    click.echo('')
    click.echo('    Next:')
    click.echo(f'      1. Edit {EVOSKILL_DIR}/task.md with your task description')
    if exec_mode == 'docker':
        click.echo('      2. Build image: docker build -t evoskill .')
        click.echo('      3. Run: evoskill run --docker')
    elif exec_mode == 'daytona':
        if not remote_config or not remote_config['daytona'].get('image'):
            click.echo('      2. Build and push image:')
            click.echo('         docker build -t evoskill .')
            click.echo('         docker tag evoskill <registry>/evoskill:latest')
            click.echo('         docker push <registry>/evoskill:latest')
            click.echo('         Then set image in .evoskill/config.toml')
        click.echo(f'      {"3" if not remote_config or not remote_config["daytona"].get("image") else "2"}. Run: evoskill run --remote')
    else:
        click.echo('      2. Run: evoskill run')
    click.echo('')
    click.echo('    Note: Default iterations is set to 7. Increase iterations in')
    click.echo('    .evoskill/config.toml for better skill discovery.')
