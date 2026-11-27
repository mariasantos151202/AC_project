#!/bin/bash
#SBATCH --job-name=MODELS_10-18_25-11
#SBATCH --partition=normal-a100-40
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=f202500007hpcvlabuportog
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out 
#SBATCH --error=logs/%x_%A_%a.err

#Environment setup
set -eo pipefail

BASE_DIR="/projects/F202500007HPCVLABUPORTO/cpereira/HPCvLAB"
RUN_SCRIPT=$(BASE_DIR)/"run_rnn.py"

conda activate ac2

# Run

echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

nvidia-smi || true
# Avoid TensorFlow grabbing all GPU memory at once
export TF_FORCE_GPU_ALLOW_GROWTH=true

srun python "$RUN_SCRIPT"

echo "End time: $(date)"