#!/bin/bash
#SBATCH -A project_2015432
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH -J mvot_label
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

# Label videos with YOLO proposals + SAM3 refinement.
# Edit `config/labeling.yaml` and set `sam3.checkpoint` (or pass --sam3-checkpoint).
mvot label --config config/labeling.yaml
