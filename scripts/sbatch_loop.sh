#!/bin/bash
#SBATCH --job-name=evoskill-loop
#SBATCH --time=06:00:00
#SBATCH --partition=normal_q
#SBATCH --output=job-outputs/loop-%j.out
#SBATCH --error=job-outputs/loop-%j.err
#SBATCH --mem=16G

cd "$SLURM_SUBMIT_DIR"

if [ -f .env ]; then
    set -a; source .env; set +a
fi

export PATH="$HOME/.local/bin:$HOME/.npm-global/bin:$HOME/.opencode/bin:$PATH"
export PYTHONPATH=$PYTHONPATH:$SLURM_SUBMIT_DIR
export PYTHONUNBUFFERED=1

mkdir -p job-outputs

echo "Running EvoSkill loop..."
uv run scripts/run_loop_comb.py "$@"

echo "Job complete."
