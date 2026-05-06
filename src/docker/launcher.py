"""Docker launcher — run EvoSkill in a local container via docker compose."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from src.cli.config import ProjectConfig

IMAGE_NAME = "evoskill"
COMPOSE_FILE = "docker-compose.yml"

# API key env vars to forward from host into the container.
# Only the name is written to docker-compose.yml (not the value).
_API_KEY_VARS = [
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY",
    "LLM_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
    "GROQ_API_KEY", "MISTRAL_API_KEY", "TOGETHER_API_KEY",
    "DEEPSEEK_API_KEY", "XAI_API_KEY",
]


def _build_compose(cfg: ProjectConfig, extra_args: list[str]) -> dict:
    """Build the docker-compose config dict from ProjectConfig."""
    project_root = cfg.project_root.resolve()
    volumes = [f"{project_root}:/workspace"]

    # Path overrides for dataset/data_dirs inside the container
    path_overrides: dict[str, str] = {}

    # Mount dataset if external
    dataset_path = cfg.dataset_path.resolve()
    try:
        dataset_path.relative_to(project_root)
    except ValueError:
        container_dataset = f"/mnt/dataset/{dataset_path.name}"
        volumes.append(f"{dataset_path}:{container_dataset}:ro")
        path_overrides["dataset_path"] = container_dataset

    # Mount external data dirs
    container_data_dirs = []
    for d in cfg.harness.data_dirs:
        p = Path(d).resolve()
        try:
            p.relative_to(project_root)
        except ValueError:
            container_path = f"/mnt/data/{p.name}"
            volumes.append(f"{p}:{container_path}:ro")
            container_data_dirs.append(container_path)

    if container_data_dirs:
        path_overrides["data_dirs"] = ",".join(container_data_dirs)

    # Environment: static values (written to file) + API keys (forwarded by name)
    env_with_values = [
        "EVOSKILL_REMOTE=1",
        "CLAUDE_CODE_ACCEPT_TOS=yes",
    ]
    if path_overrides:
        env_with_values.append(f"EVOSKILL_PATH_OVERRIDES={json.dumps(path_overrides)}")

    # Only forward keys that are actually set in the host environment
    env_forward = [k for k in _API_KEY_VARS if k in os.environ]

    # Build the evoskill run command
    import shlex
    cmd = "pip install --no-deps -e . > /dev/null 2>&1 && evoskill run"
    if extra_args:
        cmd += " " + " ".join(shlex.quote(a) for a in extra_args)

    return {
        "services": {
            "evoskill": {
                "image": IMAGE_NAME,
                "container_name": "evoskill-run",
                "working_dir": "/workspace",
                "volumes": volumes,
                "env_with_values": env_with_values,
                "env_forward": env_forward,
                "command": f'bash -c "{cmd}"',
            }
        }
    }


def _write_compose(cfg: ProjectConfig, compose: dict) -> Path:
    """Write docker-compose.yml to .evoskill/."""
    import yaml

    compose_path = cfg.evoskill_dir / COMPOSE_FILE
    svc = compose["services"]["evoskill"]

    compose_dict = {
        "services": {
            "evoskill": {
                "image": svc["image"],
                "container_name": svc["container_name"],
                "working_dir": svc["working_dir"],
                "volumes": svc["volumes"],
                "environment": svc["env_with_values"] + svc["env_forward"],
                "command": svc["command"],
            }
        }
    }

    compose_path.write_text(yaml.safe_dump(compose_dict, default_flow_style=False))
    return compose_path


def launch_docker(
    cfg: ProjectConfig,
    extra_args: list[str] | None = None,
    rebuild: bool = False,
) -> None:
    """Build image if needed and launch evoskill in a Docker container."""
    from rich.console import Console
    console = Console()

    project_root = cfg.project_root
    dockerfile = project_root / "Dockerfile"
    if not dockerfile.exists():
        console.print("[red]Error:[/red] No Dockerfile found in project root.")
        sys.exit(1)

    # Check if image exists locally
    result = subprocess.run(
        ["docker", "images", "-q", IMAGE_NAME],
        capture_output=True, text=True,
    )
    image_exists = bool(result.stdout.strip())

    # Build if needed
    if rebuild or not image_exists:
        action = "Rebuilding" if rebuild else "Building"
        console.print(f"\n  {action} Docker image...", end="")
        build_result = subprocess.run(
            ["docker", "build", "-t", IMAGE_NAME, str(project_root)],
            capture_output=True, text=True,
        )
        if build_result.returncode != 0:
            console.print(" [red]failed[/red]")
            console.print(build_result.stderr[-500:])
            sys.exit(1)
        console.print(" [green]done[/green]")

    # Generate docker-compose.yml
    compose = _build_compose(cfg, extra_args or [])
    compose_path = _write_compose(cfg, compose)

    # Stop any existing container
    subprocess.run(
        ["docker", "compose", "-f", str(compose_path), "down"],
        capture_output=True,
    )

    # Launch
    console.print("  Starting container...", end="")
    up_result = subprocess.run(
        ["docker", "compose", "-f", str(compose_path), "up", "-d"],
        capture_output=True, text=True,
    )
    if up_result.returncode != 0:
        console.print(" [red]failed[/red]")
        console.print(up_result.stderr[-500:])
        sys.exit(1)
    console.print(" [green]done[/green]")

    cf = compose_path
    console.print(f"\n  [bold]EvoSkill running in Docker[/bold]")
    console.print(f"\n  [bold]Next:[/bold]")
    console.print(f"    Logs:   docker compose -f {cf} logs -f")
    console.print(f"    Stop:   docker compose -f {cf} down")
    console.print(f"    Skills: evoskill skills\n")
