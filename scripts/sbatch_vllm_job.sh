#!/bin/bash
#SBATCH --job-name=evoskill-vllm
#SBATCH --account=llms-ar
#SBATCH --time=12:00:00
#SBATCH --partition=l40s_normal_q
#SBATCH --output=job-outputs/vllm-%j.out
#SBATCH --error=job-outputs/vllm-%j.err
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

if [ -f .env ]; then
    set -a; source .env; set +a
fi

export PATH="$HOME/.local/bin:$HOME/.npm-global/bin:$HOME/.opencode/bin:$PATH"
export PYTHONPATH="${PYTHONPATH:-}:$SLURM_SUBMIT_DIR"
export PYTHONUNBUFFERED=1
unset CLAUDECODE

module reset
module load Python/3.13.1-GCCcore-14.2.0
module load vLLM/0.18.0

export HF_HOME=/projects/llms-lab/hub

mkdir -p job-outputs

# --- vLLM configuration (override via env vars) ---
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3.5-9B}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
PORT=$((10000 + RANDOM % 10000))

# Unique cache directory per job
JOB_CACHE="/tmp/noahpro/job_${SLURM_JOB_ID}"
mkdir -p "$JOB_CACHE/torch_compile_cache" "$JOB_CACHE/torch_inductor"

export XDG_CACHE_HOME="$JOB_CACHE"
export TORCH_HOME="$JOB_CACHE/torch"
export TORCHINDUCTOR_CACHE_DIR="$JOB_CACHE/torch_inductor"

# --- Start vLLM server ---
echo "Starting vLLM server on port $PORT with model $VLLM_MODEL..."
apptainer exec --nv --cleanenv --containall \
    --bind /common/data/models --bind /tmp --bind /projects --bind /home/noahpro \
    --pwd "$SLURM_SUBMIT_DIR" \
    --ipc \
    --bind "$JOB_CACHE/torch_compile_cache":/home/noahpro/.cache/vllm/torch_compile_cache \
    --env TMPDIR=/tmp --env XDG_CACHE_HOME="$XDG_CACHE_HOME" --env TORCH_HOME="$TORCH_HOME" \
    --env TORCHINDUCTOR_CACHE_DIR="$TORCHINDUCTOR_CACHE_DIR" \
    /common/containers/vllm-openai-0.18.0.sif \
    vllm serve --port $PORT --model "$VLLM_MODEL" --trust-remote-code --gpu-memory-utilization 0.9 --max-model-len "$VLLM_MAX_MODEL_LEN" --enable-auto-tool-choice --tool-call-parser hermes &

VLLM_PID=$!

# --- Cleanup trap ---
cleanup() {
    echo "Cleaning up..."
    kill $VLLM_PID 2>/dev/null || true
    rm -rf "$JOB_CACHE"
    echo "Cleanup done."
}
trap cleanup EXIT

echo "Waiting for vLLM server..."
until curl -s "http://localhost:$PORT/v1/models" > /dev/null; do
    sleep 15
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "ERROR: vLLM server died during startup"
        exit 1
    fi
done
echo "vLLM server is UP on port $PORT"

# --- Export vLLM config as env vars (session dir patched by prepare_run_dir) ---
export VLLM_BASE_URL="http://localhost:${PORT}/v1"
export VLLM_MODEL
export VLLM_MAX_MODEL_LEN
echo "Exported VLLM_BASE_URL=$VLLM_BASE_URL"

# --- Run the experiment ---
echo "Running: uv run $@"
uv run "$@"

echo "Job complete."
