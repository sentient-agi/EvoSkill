#!/bin/bash
#SBATCH --job-name=vllm
#SBATCH --time=24:00:00
#SBATCH --partition=a100_normal_q
#SBATCH --output=job-outputs/%j-vllm.out
#SBATCH --error=job-outputs/%j-vllm.err
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

if [ -f .env ]; then
    set -a; source .env; set +a
fi

# Use the shared vLLM env
source /projects/llms-lab/creative-bench/.venv/bin/activate

export HF_HOME=/projects/llms-lab/huggingface
export XET_CACHE_DIR="${HOME}/.cache/xet"
mkdir -p "${XET_CACHE_DIR}"
mkdir -p job-outputs

PORT="${VLLM_PORT:-9999}"

# Only add --port if user didn't pass one explicitly.
PORT_ARGS=()
case " $* " in
  *" --port "*) ;;
  *) PORT_ARGS=(--port "${PORT}") ;;
esac

echo "Starting vLLM server on port ${PORT}..."
python -m vllm.entrypoints.openai.api_server "${PORT_ARGS[@]}" "$@"
