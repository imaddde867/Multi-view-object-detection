#!/bin/bash
#SBATCH -A project_2015432
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH -J multiview_label
#SBATCH -o slurm-%x-%j.out

set -euo pipefail

module purge || true
module load pytorch/2.7 || true

python -V
nvidia-smi || true

# Run from repo root
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Activate repo-local venv (recommended).
if [[ -f ".venv/bin/activate" ]]; then
  source .venv/bin/activate
fi

# Keeping cache in scratch.
export HF_HOME="${HF_HOME:-/scratch/project_2015432/exp/.hf}"

# Label videos with YOLO proposals + SAM3 refinement.
# Edit `config/labeling.yaml` and set `sam3.checkpoint` (or pass --sam3-checkpoint).
LABEL_CONFIG="${MULTIVIEW_LABEL_CONFIG:-config/labeling.yaml}"
SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-}"

cmd=(multiview label --config "$LABEL_CONFIG")
if [[ -n "$SAM3_CHECKPOINT" ]]; then
  cmd+=(--sam3-checkpoint "$SAM3_CHECKPOINT")
fi

"${cmd[@]}"
