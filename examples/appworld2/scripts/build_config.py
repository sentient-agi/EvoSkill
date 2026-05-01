"""Build runner_config matching HALO's exact jsonnet configuration.

All paths are loaded from config.json — single source of truth.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


def load_config() -> dict:
    """Load config.json from the appworld2 root."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
    return json.loads(CONFIG_PATH.read_text())


def get_appworld_root() -> Path:
    """Get the AppWorld root path from config.json."""
    config = load_config()
    root = Path(config["appworld_root"])
    if not root.exists():
        raise FileNotFoundError(f"AppWorld root not found: {root}")
    # Also set env var so AppWorld SDK can find data
    os.environ.setdefault("APPWORLD_ROOT", str(root))
    return root


def get_prompts_path() -> Path:
    """Get the prompts directory path (inside appworld2)."""
    root = get_appworld_root()
    prompts = root / "experiments" / "prompts"
    if not (prompts / "api_predictor.txt").exists():
        raise FileNotFoundError(f"Prompts not found at: {prompts}")
    return prompts


def get_default_model() -> str:
    """Get the default model from config.json."""
    return load_config().get("model", "claude-sonnet-4-20250514")


def get_default_experiment_name() -> str:
    """Get the default experiment name from config.json."""
    return load_config().get("experiment_name", "evoskill")


def build_runner_config(
    model: str | None = None,
    dataset: str = "dev",
    max_steps: int | None = None,
    max_predicted_apis: int = 20,
) -> dict:
    """Build a runner_config dict matching HALO's exact setup.

    Args:
        model: Model name (default: from config.json).
        dataset: Dataset split name.
        max_steps: Max agent turns per task (default: from config.json).
        max_predicted_apis: Max APIs from predictor.

    Returns:
        Config dict ready for run_experiment().
    """
    config = load_config()
    model = model or config.get("model", "claude-sonnet-4-20250514")
    max_steps = max_steps or config.get("max_steps", 50)
    prompts = get_prompts_path()

    # Determine model type based on prefix
    if model.startswith("anthropic/") or model.startswith("claude"):
        model_type = "litellm"
        model_settings = {
            "api_type": "chat_completions",
            "temperature": 0.0,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }
    else:
        model_type = "openai"
        model_settings = {
            "api_type": "chat_completions",
            "temperature": 0.0,
            "seed": 100,
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "store": False,
        }

    model_config = {
        "type": model_type,
        "name": model,
        "settings": model_settings,
    }

    return {
        "agent": {
            "model": {
                **model_config,
                "settings": {
                    **model_settings,
                    "tool_choice": "auto",
                    "parallel_tool_calls": True,
                },
            },
            "max_steps": max_steps,
            "retrieve_apis": True,
            "prompt_file_path": str(prompts / "function_calling_agent" / "instructions.txt"),
            "demo_messages_file_path": str(prompts / "function_calling_agent" / "demos.json"),
        },
        "api_predictor": {
            "mode": "predicted",
            "model_config": model_config,
            "prompt_file_path": str(prompts / "api_predictor.txt"),
            "demo_task_ids": ["82e2fac_1", "29caf6f_1", "d0b1f43_1"],
            "max_predicted_apis": max_predicted_apis,
        },
        "appworld": {
            "start_servers": True,
            "show_server_logs": False,
            "remote_apis_url": "http://localhost:{port}",
            "remote_mcp_url": "http://localhost:{port}",
            "mcp_server_kwargs": {
                "output_type": "both_but_empty_text",
            },
            "random_seed": 100,
            "include_direct_functions": True,
            "direct_function_separator": "__",
        },
        "logger": {
            "color": True,
            "verbose": True,
        },
        "dataset": dataset,
    }
