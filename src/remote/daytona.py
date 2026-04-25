"""Daytona remote execution backend."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterator

from src.cli.config import ProjectConfig
from src.remote.base import RemoteBackend, RunInfo
from src.remote.sync import (
    bundle_create_args,
    bundle_unbundle_args,
    download_file_list,
    remap_data_dirs,
    upload_file_list,
)

SESSION_ID = "evoskill-run"


def _make_client(api_key: str):
    """Create a Daytona client. Isolated for mocking."""
    from daytona import Daytona, DaytonaConfig

    return Daytona(DaytonaConfig(api_key=api_key))


def _get_sandbox(client, sandbox_id: str):
    """Retrieve an existing sandbox by ID. Isolated for mocking."""
    return client.get(sandbox_id)


def _create_sandbox_params(*, image, env_vars, auto_stop_interval, cpu, memory, disk):
    """Build CreateSandboxFromImageParams. Isolated for mocking when daytona isn't installed."""
    from daytona import CreateSandboxFromImageParams, Resources

    return CreateSandboxFromImageParams(
        image=image,
        env_vars=env_vars,
        auto_stop_interval=auto_stop_interval,
        resources=Resources(cpu=cpu, memory=memory, disk=disk),
    )


def _is_under(path: Path, parent: Path) -> bool:
    """Check if path is under parent directory."""
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _collect_api_keys() -> dict[str, str]:
    """Collect LLM API keys from environment."""
    keys = [
        "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY",
        "LLM_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY",
        "GROQ_API_KEY", "MISTRAL_API_KEY", "TOGETHER_API_KEY",
        "DEEPSEEK_API_KEY", "XAI_API_KEY",
    ]
    return {k: os.environ[k] for k in keys if k in os.environ}


class DaytonaBackend(RemoteBackend):
    """Run EvoSkill on a Daytona sandbox."""

    def __init__(self):
        self._client = None
        self._sandbox = None
        self._path_overrides: dict[str, str] = {}

    def _ensure_client(self, cfg: ProjectConfig):
        """Lazily initialize the Daytona client from config."""
        if self._client is None:
            daytona_cfg = cfg.remote.daytona
            if not daytona_cfg or not daytona_cfg.api_key:
                raise ValueError(
                    "Daytona API key required. Set it in config.toml "
                    "[remote.daytona] api_key or DAYTONA_API_KEY env var."
                )
            self._client = _make_client(daytona_cfg.api_key)
        return self._client

    def setup(self, cfg: ProjectConfig) -> None:
        self._ensure_client(cfg)
        daytona_cfg = cfg.remote.daytona

        if not daytona_cfg.image:
            raise ValueError(
                "No Docker image configured for Daytona. "
                "Build your image with 'docker build -t evoskill .' then "
                "push to a registry and set image in config.toml:\n"
                "  [remote.daytona]\n"
                '  image = "your-registry/evoskill:latest"'
            )

        env_vars = _collect_api_keys()
        env_vars["CLAUDE_CODE_ACCEPT_TOS"] = "yes"
        env_vars["EVOSKILL_REMOTE"] = "1"

        params = _create_sandbox_params(
            image=daytona_cfg.image,
            env_vars=env_vars,
            auto_stop_interval=daytona_cfg.timeout,
            cpu=daytona_cfg.cpu,
            memory=daytona_cfg.memory,
            disk=daytona_cfg.disk,
        )
        self._sandbox = self._client.create(params)

    def upload(self, cfg: ProjectConfig) -> None:
        sandbox = self._sandbox
        project_root = cfg.project_root

        # 1. Create and upload git bundle
        with tempfile.NamedTemporaryFile(suffix=".bundle", delete=False) as f:
            bundle_path = f.name

        args = bundle_create_args(bundle_path, all_branches=True)
        subprocess.run(args, cwd=str(project_root), check=True)

        bundle_bytes = Path(bundle_path).read_bytes()
        sandbox.fs.upload_file(bundle_bytes, "/workspace/repo.bundle")
        sandbox.process.exec(
            "cd /workspace && git init && git bundle unbundle repo.bundle && rm repo.bundle",
            cwd="/workspace",
        )

        # 2. Upload project files (skip .git — handled by bundle)
        files = upload_file_list(project_root)
        for file_path in files:
            rel = file_path.relative_to(project_root)
            if rel.parts and rel.parts[0] == ".git":
                continue
            remote_path = f"/workspace/{rel}"
            parent = str(Path(remote_path).parent)
            sandbox.fs.create_folder(parent, mode="755")
            sandbox.fs.upload_file(file_path.read_bytes(), remote_path)

        # 3. Upload dataset if external
        dataset_path = cfg.dataset_path.resolve()
        if not _is_under(dataset_path, project_root.resolve()):
            container_dataset = f"/mnt/dataset/{dataset_path.name}"
            sandbox.fs.create_folder("/mnt/dataset", mode="755")
            sandbox.fs.upload_file(dataset_path.read_bytes(), container_dataset)
            self._path_overrides["dataset_path"] = container_dataset

        # 4. Upload external data dirs via tar (chunked if > 50MB)
        MAX_CHUNK = 50 * 1024 * 1024
        mappings = remap_data_dirs(cfg.harness.data_dirs, project_root)
        container_data_dirs = []

        for mapping in mappings:
            container_data_dirs.append(mapping.container_path)
            if mapping.needs_upload:
                with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
                    tar_path = f.name
                subprocess.run(
                    ["tar", "czf", tar_path, "-C", str(mapping.host_path.parent), mapping.host_path.name],
                    check=True,
                )
                tar_bytes = Path(tar_path).read_bytes()
                Path(tar_path).unlink(missing_ok=True)

                if len(tar_bytes) <= MAX_CHUNK:
                    remote_tar = f"/tmp/{mapping.host_path.name}.tar.gz"
                    sandbox.fs.upload_file(tar_bytes, remote_tar)
                    sandbox.process.exec(
                        f"mkdir -p {mapping.container_path} && "
                        f"tar xzf {remote_tar} -C /mnt/data/ && "
                        f"rm {remote_tar}",
                    )
                else:
                    sandbox.process.exec("mkdir -p /tmp/chunks")
                    for i in range(0, len(tar_bytes), MAX_CHUNK):
                        chunk = tar_bytes[i:i + MAX_CHUNK]
                        sandbox.fs.upload_file(chunk, f"/tmp/chunks/part_{i:010d}")
                    sandbox.process.exec(
                        f"cat /tmp/chunks/part_* > /tmp/combined.tar.gz && "
                        f"mkdir -p {mapping.container_path} && "
                        f"tar xzf /tmp/combined.tar.gz -C /mnt/data/ && "
                        f"rm -rf /tmp/chunks /tmp/combined.tar.gz",
                    )

        if container_data_dirs:
            self._path_overrides["data_dirs"] = ",".join(container_data_dirs)

    def run(self, cfg: ProjectConfig, extra_args: list[str] | None = None) -> RunInfo:
        from daytona import SessionExecuteRequest

        sandbox = self._sandbox

        # Install EvoSkill (--no-deps: image already has all runtime deps)
        sandbox.process.exec(
            "cd /workspace && pip install --no-deps -e . 2>&1",
            cwd="/workspace",
        )

        # Preflight checks
        preflight = sandbox.process.exec(
            "echo '=== evoskill ===' && which evoskill && "
            "echo '=== ANTHROPIC_API_KEY ===' && "
            "([ -n \"$ANTHROPIC_API_KEY\" ] && echo 'set' || echo 'NOT SET') && "
            "echo '=== python ===' && python --version && "
            "echo '=== git ===' && git --version",
        )
        if "NOT SET" in preflight.result and cfg.harness.name == "claude":
            raise RuntimeError(
                f"Preflight failed: ANTHROPIC_API_KEY not set in sandbox.\n"
                f"Full diagnostics:\n{preflight.result}"
            )

        # Build run script with path overrides
        env_lines = []
        if self._path_overrides:
            overrides_json = json.dumps(self._path_overrides)
            env_lines.append(f"export EVOSKILL_PATH_OVERRIDES='{overrides_json}'")

        cmd = "evoskill run"
        if extra_args:
            cmd += " " + " ".join(extra_args)

        # Build run command (no script file needed — session executes directly)
        env_prefix = ""
        if env_lines:
            env_prefix = " && ".join(env_lines) + " && "

        full_cmd = f"cd /workspace && {env_prefix}{cmd}"

        # Launch via Daytona session (survives API call returning)
        sandbox.process.create_session(SESSION_ID)
        req = SessionExecuteRequest(command=full_cmd, run_async=True)
        response = sandbox.process.execute_session_command(SESSION_ID, req)

        # Save run info
        run_info = RunInfo(
            run_id=f"daytona-{sandbox.id}",
            target="daytona",
            extra={
                "sandbox_id": sandbox.id,
                "session_id": SESSION_ID,
                "cmd_id": response.cmd_id,
            },
        )
        run_info.save(cfg.project_root)
        return run_info

    def status(self, cfg: ProjectConfig, run_info: RunInfo) -> str:
        sandbox_id = run_info.extra.get("sandbox_id")
        session_id = run_info.extra.get("session_id", SESSION_ID)
        cmd_id = run_info.extra.get("cmd_id")
        if not sandbox_id or not cmd_id:
            raise ValueError("No sandbox_id or cmd_id in run info")

        client = self._ensure_client(cfg)
        sandbox = _get_sandbox(client, sandbox_id)

        cmd_info = sandbox.process.get_session_command(session_id, cmd_id)
        is_running = cmd_info.exit_code is None

        # Get logs for progress info
        logs = sandbox.process.get_session_command_logs(session_id, cmd_id)
        log_text = logs.output or logs.stdout or ""

        last_score = ""
        has_error = False
        error_line = ""
        for line in log_text.splitlines()[-20:]:
            if "Score:" in line:
                last_score = f"\n  Last:    {line.strip()}"
            if "Error" in line or "Traceback" in line or "Exception" in line:
                has_error = True
                error_line = line.strip()

        if is_running:
            return f"running{last_score}"
        elif cmd_info.exit_code != 0 or has_error:
            return f"failed (exit_code={cmd_info.exit_code})\n  Error:   {error_line}"
        else:
            return f"completed{last_score}"

    def logs(self, cfg: ProjectConfig, run_info: RunInfo, follow: bool = False) -> Iterator[str]:
        sandbox_id = run_info.extra.get("sandbox_id")
        session_id = run_info.extra.get("session_id", SESSION_ID)
        cmd_id = run_info.extra.get("cmd_id")
        if not sandbox_id or not cmd_id:
            raise ValueError("No sandbox_id or cmd_id in run info")

        client = self._ensure_client(cfg)
        sandbox = _get_sandbox(client, sandbox_id)

        if follow:
            last_size = 0
            while True:
                logs = sandbox.process.get_session_command_logs(session_id, cmd_id)
                content = logs.output or logs.stdout or ""
                if len(content) > last_size:
                    for line in content[last_size:].splitlines():
                        yield line
                    last_size = len(content)

                cmd_info = sandbox.process.get_session_command(session_id, cmd_id)
                if cmd_info.exit_code is not None:
                    # Final read
                    logs = sandbox.process.get_session_command_logs(session_id, cmd_id)
                    final = logs.output or logs.stdout or ""
                    if len(final) > last_size:
                        for line in final[last_size:].splitlines():
                            yield line
                    yield f"--- Run completed (exit_code={cmd_info.exit_code}) ---"
                    break
                time.sleep(5)
        else:
            logs = sandbox.process.get_session_command_logs(session_id, cmd_id)
            content = logs.output or logs.stdout or ""
            for line in content.splitlines()[-50:]:
                yield line

    def download(self, cfg: ProjectConfig, run_info: RunInfo) -> None:
        sandbox_id = run_info.extra.get("sandbox_id")
        if not sandbox_id:
            raise ValueError("No sandbox_id in run info")

        client = self._ensure_client(cfg)
        sandbox = _get_sandbox(client, sandbox_id)
        download_cfg = cfg.remote.download
        project_root = cfg.project_root

        # 1. Identify best program
        result = sandbox.process.exec(
            "git tag -l 'frontier/*' --sort=-v:refname | head -1",
            cwd="/workspace",
        )
        best_tag = result.result.strip()
        best_branch = best_tag.split(":")[0].replace("frontier/", "program/") if best_tag else None

        # 2. Create git bundle on remote
        if download_cfg.all_branches:
            sandbox.process.exec("git bundle create /tmp/results.bundle --all", cwd="/workspace")
        elif best_branch:
            sandbox.process.exec(f"git bundle create /tmp/results.bundle {best_branch}", cwd="/workspace")

        # 3. Download bundle
        bundle_bytes = sandbox.fs.download_file("/tmp/results.bundle")
        with tempfile.NamedTemporaryFile(suffix=".bundle", delete=False) as f:
            f.write(bundle_bytes)
            local_bundle = f.name

        # 4. Unbundle locally and create branches from the refs
        result = subprocess.run(
            bundle_unbundle_args(local_bundle),
            cwd=str(project_root), check=True, capture_output=True, text=True,
        )
        for line in result.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) == 2 and parts[1].startswith("refs/heads/"):
                sha, ref = parts
                branch_name = ref.removeprefix("refs/heads/")
                subprocess.run(
                    ["git", "branch", "-f", branch_name, sha],
                    cwd=str(project_root), capture_output=True,
                )

        # 5. Download file-based results
        paths = download_file_list(download_cfg)
        for remote_rel in paths:
            remote_path = f"/workspace/{remote_rel}"
            local_path = project_root / remote_rel
            try:
                content = sandbox.fs.download_file(remote_path)
                local_path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(content, bytes):
                    local_path.write_bytes(content)
            except Exception:
                pass

    def stop(self, cfg: ProjectConfig, run_info: RunInfo) -> None:
        sandbox_id = run_info.extra.get("sandbox_id")
        if not sandbox_id:
            raise ValueError("No sandbox_id in run info")

        client = self._ensure_client(cfg)
        sandbox = _get_sandbox(client, sandbox_id)
        client.delete(sandbox)

    def cleanup_current(self, cfg: ProjectConfig) -> None:
        """Delete the sandbox created during this session (for crash recovery)."""
        if not self._sandbox:
            return
        try:
            client = self._ensure_client(cfg)
            client.delete(self._sandbox)
        except Exception:
            pass
