#!/bin/bash
#SBATCH --job-name=evoskill-loop
#SBATCH --account=llms-ar
#SBATCH --time=06:00:00
#SBATCH --partition=normal_q
#SBATCH --output=job-outputs/loop-%j.out
#SBATCH --error=job-outputs/loop-%j.err
#SBATCH --mem=16G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=noahpro@gmail.com

module reset
module load Python/3.13.1-GCCcore-14.2.0
module load nodejs/22.17.1-GCCcore-14.3.0

export PATH="$HOME/.npm-global/bin:$HOME/.opencode/bin:$PATH"
source ~/.secrets/gemini.env
export OPENCODE_ENABLE_EXA=true
export PYTHONPATH=$PYTHONPATH:$SLURM_SUBMIT_DIR
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p job-outputs

if [ ! -f ".dataset/seal-0.csv" ]; then
    echo "Dataset not found at .dataset/seal-0.csv"
    exit 1
fi

echo "Running EvoSkill loop on SEAL-QA (8 train, 3 val, ~1.5 epochs)..."
~/.local/bin/uv run scripts/run_loop_sealqa.py "$@"

echo "Job complete."
