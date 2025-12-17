#!/bin/bash
#SBATCH -A project_2015432
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH -J mvot_train
#SBATCH -o slurm-%x-%j.out

set -euo pipefail

module purge || true

python -V
nvidia-smi || true

# Run from repo root (defaults to directory where you called sbatch).
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Activate repo-local venv (recommended).
if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

# For 2-4 GPUs, request more GPUs and set `runtime.device: "0,1"` (or "0,1,2,3") in config/train.yaml.
mvot train --config config/train.yaml
