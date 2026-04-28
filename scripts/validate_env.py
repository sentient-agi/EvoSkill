"""
validate_env.py — EvoSkill Environment Validator
--------------------------------------------------
Checks that your environment is correctly configured before running
the self-improvement loop. Detects missing API keys, unsupported
Python versions, missing dependencies, and invalid SDK/model combos.

Usage:
    python scripts/validate_env.py                        # check all
    python scripts/validate_env.py --sdk claude           # check Claude only
    python scripts/validate_env.py --sdk opencode --model google/gemini-2.0-flash-exp
    python scripts/validate_env.py --list-free-options    # show free/low-cost combos
"""

import argparse
import importlib
import os
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional


# ── ANSI colours ────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

OK   = f"{GREEN}✔{RESET}"
WARN = f"{YELLOW}⚠{RESET}"
FAIL = f"{RED}✘{RESET}"
INFO = f"{CYAN}ℹ{RESET}"


# ── SDK / model registry ────────────────────────────────────────────────────
@dataclass
class SDKConfig:
    name: str
    env_var: str
    signup_url: str
    free: bool
    models: list[str]
    notes: str = ""


SDK_REGISTRY: dict[str, SDKConfig] = {
    "claude": SDKConfig(
        name="Claude (Anthropic)",
        env_var="ANTHROPIC_API_KEY",
        signup_url="https://console.anthropic.com",
        free=False,
        models=[
            "claude-opus-4-5-20250514",
            "claude-sonnet-4-5-20250514",
            "claude-haiku-4-5-20251001",   # lowest cost
        ],
        notes="Free tier available at console.anthropic.com — Haiku is cheapest.",
    ),
    "opencode": SDKConfig(
        name="OpenCode (multi-provider via OpenRouter)",
        env_var="OPENROUTER_API_KEY",
        signup_url="https://openrouter.ai",
        free=True,
        models=[
            "google/gemini-2.0-flash-exp",        # free on OpenRouter
            "deepseek-ai/DeepSeek-V3",             # very low cost
            "google/gemini-2.5-pro-preview",
            "openai/gpt-4o",
        ],
        notes="Gemini Flash is FREE on OpenRouter — no credit card needed.",
    ),
}

FREE_COMBOS: list[tuple[str, str, str]] = [
    ("opencode", "google/gemini-2.0-flash-exp",  "Free on OpenRouter (no billing required)"),
    ("opencode", "deepseek-ai/DeepSeek-V3",       "~$0.001/1k tokens — extremely cheap"),
    ("claude",   "claude-haiku-4-5-20251001",     "Cheapest Anthropic model"),
]


# ── Check helpers ────────────────────────────────────────────────────────────
@dataclass
class CheckResult:
    label: str
    passed: bool
    message: str
    hint: str = ""


def check_python_version() -> CheckResult:
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 12)
    return CheckResult(
        label="Python version",
        passed=ok,
        message=f"Python {major}.{minor} detected",
        hint="" if ok else "EvoSkill requires Python 3.12+. Use pyenv or conda to upgrade.",
    )


def check_package(package: str, import_name: Optional[str] = None) -> CheckResult:
    name = import_name or package
    try:
        importlib.import_module(name)
        return CheckResult(label=f"Package: {package}", passed=True, message="installed")
    except ImportError:
        return CheckResult(
            label=f"Package: {package}",
            passed=False,
            message="not found",
            hint=f"Run: pip install {package}  (or: uv sync)",
        )


def check_uv() -> CheckResult:
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True, timeout=5)
        ver = result.stdout.strip()
        return CheckResult(label="uv (package manager)", passed=True, message=ver)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return CheckResult(
            label="uv (package manager)",
            passed=False,
            message="not found",
            hint="Recommended: pip install uv  or  curl -Ls https://astral.sh/uv/install.sh | sh",
        )


def check_git() -> CheckResult:
    try:
        result = subprocess.run(["git", "--version"], capture_output=True, text=True, timeout=5)
        return CheckResult(label="git", passed=True, message=result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return CheckResult(
            label="git",
            passed=False,
            message="not found",
            hint="git is required for EvoSkill's branch-based program versioning.",
        )


def check_docker() -> CheckResult:
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=10)
        ok = result.returncode == 0
        return CheckResult(
            label="Docker",
            passed=ok,
            message="running" if ok else "not running",
            hint="" if ok else "Docker is needed only for LiveCodeBench. Other benchmarks work without it.",
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return CheckResult(
            label="Docker",
            passed=False,
            message="not installed",
            hint="Optional — only required for LiveCodeBench sandbox. Skip if using OfficeQA or SealQA.",
        )


def check_api_key(sdk: str) -> CheckResult:
    cfg = SDK_REGISTRY.get(sdk)
    if not cfg:
        return CheckResult(label=f"API key ({sdk})", passed=False, message=f"Unknown SDK: {sdk}")
    val = os.environ.get(cfg.env_var, "")
    if val:
        masked = val[:6] + "..." + val[-4:] if len(val) > 12 else "***"
        return CheckResult(
            label=f"API key — {cfg.env_var}",
            passed=True,
            message=f"set ({masked})",
        )
    return CheckResult(
        label=f"API key — {cfg.env_var}",
        passed=False,
        message="not set",
        hint=f"export {cfg.env_var}=your-key-here  |  Sign up: {cfg.signup_url}",
    )


def check_model(sdk: str, model: str) -> CheckResult:
    cfg = SDK_REGISTRY.get(sdk)
    if not cfg:
        return CheckResult(label="Model", passed=False, message=f"Unknown SDK: {sdk}")
    known = model in cfg.models
    return CheckResult(
        label=f"Model: {model}",
        passed=True,   # don't block on unknown — user may use a new model
        message="recognised" if known else f"not in known list for {sdk} (may still work)",
        hint="" if known else f"Known models for {sdk}: {', '.join(cfg.models)}",
    )


def check_dataset(task: str) -> CheckResult:
    paths = {
        "sealqa":   ".dataset/seal-0.csv",
        "dabstep":  ".dataset/dabstep_data.csv",
        "base":     ".dataset/new_runs_base/solved_dataset.csv",
    }
    path = paths.get(task)
    if not path:
        return CheckResult(label=f"Dataset ({task})", passed=False, message="Unknown task")
    exists = os.path.isfile(path)
    return CheckResult(
        label=f"Dataset ({task}): {path}",
        passed=exists,
        message="found" if exists else "missing",
        hint="" if exists else f"Place your dataset CSV at: {path}",
    )


# ── Renderer ─────────────────────────────────────────────────────────────────
def render(results: list[CheckResult]) -> int:
    """Print results, return number of failures."""
    failures = 0
    for r in results:
        icon = OK if r.passed else (WARN if r.hint else FAIL)
        status_color = GREEN if r.passed else (YELLOW if r.hint else RED)
        print(f"  {icon}  {r.label:<42} {status_color}{r.message}{RESET}")
        if r.hint:
            print(f"       {INFO} {r.hint}")
        if not r.passed:
            failures += 1
    return failures


def print_free_options() -> None:
    print(f"\n{BOLD}{CYAN}── Free / low-cost model options ──{RESET}\n")
    print(f"  {'SDK':<12} {'Model':<40} Notes")
    print(f"  {'-'*12} {'-'*40} {'-'*40}")
    for sdk, model, note in FREE_COMBOS:
        print(f"  {sdk:<12} {model:<40} {note}")
    print()
    print(f"  {INFO} Example command using the free Gemini Flash option:")
    print(f"     uv run python scripts/run_eval.py --sdk opencode --model google/gemini-2.0-flash-exp")
    print(f"\n  {INFO} Limit samples during exploration to keep costs near zero:")
    print(f"     uv run python scripts/run_eval.py --sdk opencode --model google/gemini-2.0-flash-exp --num-samples 20\n")


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate your EvoSkill environment before running.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--sdk",   default=None, help="SDK to validate: claude | opencode")
    parser.add_argument("--model", default=None, help="Model name to validate")
    parser.add_argument("--task",  default=None, help="Task to check dataset for: sealqa | dabstep | base")
    parser.add_argument("--list-free-options", action="store_true", help="Show free/low-cost model options")
    args = parser.parse_args()

    if args.list_free_options:
        print_free_options()
        return

    sdks_to_check = [args.sdk] if args.sdk else list(SDK_REGISTRY.keys())

    print(f"\n{BOLD}EvoSkill — Environment Validator{RESET}")
    print("─" * 54)

    # ── System checks
    print(f"\n{BOLD}System{RESET}")
    system_results = [
        check_python_version(),
        check_uv(),
        check_git(),
        check_docker(),
    ]
    sys_failures = render(system_results)

    # ── Dependency checks
    print(f"\n{BOLD}Python dependencies{RESET}")
    deps = ["anthropic", "openai", "pydantic", "tqdm", "pandas"]
    dep_results = [check_package(d) for d in deps]
    dep_failures = render(dep_results)

    # ── API key checks
    print(f"\n{BOLD}API keys{RESET}")
    key_results = [check_api_key(sdk) for sdk in sdks_to_check]
    key_failures = render(key_results)

    # ── Model check (if specified)
    model_failures = 0
    if args.sdk and args.model:
        print(f"\n{BOLD}Model{RESET}")
        model_results = [check_model(args.sdk, args.model)]
        model_failures = render(model_results)

    # ── Dataset check (if task specified)
    dataset_failures = 0
    if args.task:
        print(f"\n{BOLD}Dataset{RESET}")
        dataset_results = [check_dataset(args.task)]
        dataset_failures = render(dataset_results)

    # ── Summary
    total = sys_failures + dep_failures + key_failures + model_failures + dataset_failures
    print("\n" + "─" * 54)
    if total == 0:
        print(f"  {OK}  {GREEN}{BOLD}All checks passed — you're good to go!{RESET}")
    else:
        print(f"  {FAIL}  {RED}{BOLD}{total} check(s) failed.{RESET} Resolve the hints above, then re-run.")
        print(f"\n  {INFO} No paid plan? Run:  python scripts/validate_env.py --list-free-options")
    print()


if __name__ == "__main__":
    main()