#!/bin/bash
#SBATCH --job-name=eval-comb
#SBATCH --account=llms-lab
#SBATCH --time=04:00:00
#SBATCH --partition=normal_q
#SBATCH --output=job-outputs/evalcomb-%j.out
#SBATCH --error=job-outputs/evalcomb-%j.err
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kuzoto@vt.edu

module reset
module load Python/3.13.1-GCCcore-14.2.0

export PATH="$HOME/.opencode/bin:$PATH"
source ~/research/EvoSkill/.env
export OPENCODE_ENABLE_EXA=true
export PYTHONPATH=$PYTHONPATH:$SLURM_SUBMIT_DIR
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p job-outputs

# Ensure the dataset is accessible
if [ ! -f ".dataset/seal-0.csv" ]; then
    echo "Dataset not found at .dataset/seal-0.csv"
    exit 1
fi

echo "Running single evaluation entry..."
~/.local/bin/uv run scripts/run_eval_comb.py "$@"

echo "Job complete."