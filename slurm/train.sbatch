#!/bin/bash
#SBATCH -A project_2015432
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH -J multiview_train
#SBATCH -o slurm-%x-%j.out

set -euo pipefail

module purge || true
module load pytorch/2.7 || true

python -V
nvidia-smi || true

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Activate repo-local venv
if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

TRAIN_CONFIG="${MULTIVIEW_TRAIN_CONFIG:-config/train.yaml}"
multiview train --config "$TRAIN_CONFIG"
