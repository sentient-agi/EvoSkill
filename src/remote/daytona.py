"""Daytona remote execution backend."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
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

        env_vars = _collect_api_keys()

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

        # 4. Upload external data dirs (tar + upload + extract for speed)
        mappings = remap_data_dirs(cfg.harness.data_dirs, project_root)
        container_data_dirs = []

        for mapping in mappings:
            container_data_dirs.append(mapping.container_path)
            if mapping.needs_upload:
                # Tar locally, upload single file, extract on sandbox
                with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
                    tar_path = f.name
                subprocess.run(
                    ["tar", "czf", tar_path, "-C", str(mapping.host_path.parent), mapping.host_path.name],
                    check=True,
                )
                tar_bytes = Path(tar_path).read_bytes()
                remote_tar = f"/tmp/{mapping.host_path.name}.tar.gz"
                sandbox.fs.upload_file(tar_bytes, remote_tar)
                sandbox.process.exec(
                    f"mkdir -p {mapping.container_path} && "
                    f"tar xzf {remote_tar} -C /mnt/data/ && "
                    f"rm {remote_tar}",
                )
                Path(tar_path).unlink(missing_ok=True)

        if container_data_dirs:
            self._path_overrides["data_dirs"] = ",".join(container_data_dirs)

    def run(self, cfg: ProjectConfig, extra_args: list[str] | None = None) -> RunInfo:
        sandbox = self._sandbox

        # Install EvoSkill and capture output for debugging
        install_result = sandbox.process.exec(
            "cd /workspace && pip install -e . 2>&1 | tail -5",
            cwd="/workspace",
        )

        # Write a launcher script so background execution is reliable
        env_lines = []
        if self._path_overrides:
            overrides_json = json.dumps(self._path_overrides)
            env_lines.append(f'export EVOSKILL_PATH_OVERRIDES=\'{overrides_json}\'')

        cmd = "evoskill run"
        if extra_args:
            cmd += " " + " ".join(extra_args)

        script = "#!/bin/bash\n"
        script += "cd /workspace\n"
        for line in env_lines:
            script += line + "\n"
        script += f"{cmd} > /workspace/evoskill.log 2>&1\n"
        script += "echo $? > /workspace/evoskill.exit_code\n"

        sandbox.fs.upload_file(script.encode(), "/workspace/run.sh")
        sandbox.process.exec("chmod +x /workspace/run.sh")

        # Launch via nohup + disown so it survives session end
        sandbox.process.exec(
            "nohup bash /workspace/run.sh > /dev/null 2>&1 & disown",
        )

        # Brief pause then verify it started
        import time
        time.sleep(2)
        check = sandbox.process.exec(
            "pgrep -f 'evoskill run' > /dev/null 2>&1 && echo 'started' || "
            "(cat /workspace/evoskill.log 2>/dev/null; echo 'FAILED_TO_START')",
        )
        if "FAILED_TO_START" in check.result:
            # Show what went wrong
            raise RuntimeError(
                f"Remote run failed to start. Output:\n{check.result}"
            )

        # Save run info
        run_info = RunInfo(
            run_id=f"daytona-{sandbox.id}",
            target="daytona",
            extra={"sandbox_id": sandbox.id},
        )
        run_info.save(cfg.project_root)
        return run_info

    def status(self, cfg: ProjectConfig, run_info: RunInfo) -> str:
        sandbox_id = run_info.extra.get("sandbox_id")
        if not sandbox_id:
            raise ValueError("No sandbox_id in run info")

        client = self._ensure_client(cfg)
        sandbox = _get_sandbox(client, sandbox_id)

        # Check if evoskill process is still running
        proc_check = sandbox.process.exec(
            "pgrep -f 'evoskill run' > /dev/null 2>&1 && echo 'alive' || echo 'done'",
        )
        is_running = "alive" in proc_check.result

        # Read checkpoint for progress
        checkpoint = sandbox.process.exec(
            "cat /workspace/.claude/loop_checkpoint.json 2>/dev/null || echo '{}'",
        )
        iteration = ""
        try:
            data = json.loads(checkpoint.result)
            if "iteration" in data:
                max_iter = cfg.evolution.iterations
                iteration = f" (iteration {data['iteration']}/{max_iter})"
        except (json.JSONDecodeError, ValueError):
            pass

        # Read last lines of log for score or errors
        tail = sandbox.process.exec(
            "tail -20 /workspace/evoskill.log 2>/dev/null || echo ''",
        )
        last_score = ""
        has_error = False
        error_line = ""
        for line in tail.result.splitlines():
            if "Score:" in line:
                last_score = f"\n  Last:    {line.strip()}"
            if "Error" in line or "Traceback" in line or "Exception" in line:
                has_error = True
                error_line = line.strip()

        if is_running:
            return f"running{iteration}{last_score}"
        elif has_error:
            return f"failed{iteration}\n  Error:   {error_line}"
        else:
            return f"completed{iteration}{last_score}"

    def logs(self, cfg: ProjectConfig, run_info: RunInfo, follow: bool = False) -> Iterator[str]:
        sandbox_id = run_info.extra.get("sandbox_id")
        if not sandbox_id:
            raise ValueError("No sandbox_id in run info")

        client = self._ensure_client(cfg)
        sandbox = _get_sandbox(client, sandbox_id)

        if follow:
            # Stream: read current content then poll for new lines
            import time
            last_size = 0
            while True:
                result = sandbox.process.exec(
                    "cat /workspace/evoskill.log 2>/dev/null || echo ''",
                )
                content = result.result
                if len(content) > last_size:
                    new_content = content[last_size:]
                    for line in new_content.splitlines():
                        yield line
                    last_size = len(content)

                # Check if process is still running
                proc_check = sandbox.process.exec(
                    "pgrep -f 'evoskill run' > /dev/null 2>&1 && echo 'alive' || echo 'done'",
                )
                if "done" in proc_check.result:
                    # One final read
                    result = sandbox.process.exec("cat /workspace/evoskill.log 2>/dev/null || echo ''")
                    if len(result.result) > last_size:
                        for line in result.result[last_size:].splitlines():
                            yield line
                    yield "--- Run completed ---"
                    break
                time.sleep(5)
        else:
            result = sandbox.process.exec(
                "tail -50 /workspace/evoskill.log 2>/dev/null || echo 'No logs yet'",
            )
            for line in result.result.splitlines():
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

        # 4. Unbundle locally
        args = bundle_unbundle_args(local_bundle)
        subprocess.run(args, cwd=str(project_root), check=True)

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
        sandbox.stop()
