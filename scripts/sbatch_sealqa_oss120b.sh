#!/bin/bash
#SBATCH --job-name=sealqa-gemini
#SBATCH --account=llms-ar
#SBATCH --time=04:00:00
#SBATCH --partition=normal_q
#SBATCH --output=job-outputs/sealqa-%j.out
#SBATCH --error=job-outputs/sealqa-%j.err
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=noahpro@gmail.com

module reset
module load Python/3.13.1-GCCcore-14.2.0

export PATH="$HOME/.opencode/bin:$PATH"
source ~/.secrets/gemini.env
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

echo "Running SEAL-QA evaluation with Gemini 3.1 Flash..."
~/.local/bin/uv run scripts/run_sealqa_opencode.py "$@"

echo "Job complete."
