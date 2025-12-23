#!/usr/bin/env bash
set -euo pipefail

# Submits label → train → run as three SLURM jobs with dependencies.
#
# Usage (from repo root):
#   bash slurm/submit_pipeline.sh
#
# Optional overrides:
#   MULTIVIEW_LABEL_CONFIG=config/labeling.yaml \
#   MULTIVIEW_TRAIN_CONFIG=config/train.yaml \
#   MULTIVIEW_RUN_CONFIG=config/system_demo_tuned.yaml \
#   SAM3_CHECKPOINT=/path/to/sam3.pt \
#   bash slurm/submit_pipeline.sh

LABEL_CONFIG="${MULTIVIEW_LABEL_CONFIG:-config/labeling.yaml}"
TRAIN_CONFIG="${MULTIVIEW_TRAIN_CONFIG:-config/train.yaml}"
# Default run config is the tuned demo; override with MULTIVIEW_RUN_CONFIG for custom runs.
RUN_CONFIG="${MULTIVIEW_RUN_CONFIG:-config/system_demo_tuned.yaml}"
SAM3_CHECKPOINT="${SAM3_CHECKPOINT:-}"

echo "Submitting Multiview pipeline:"
echo "  label config: $LABEL_CONFIG"
echo "  train config: $TRAIN_CONFIG"
echo "  run   config: $RUN_CONFIG"
if [[ -n "$SAM3_CHECKPOINT" ]]; then
  echo "  sam3 checkpoint: $SAM3_CHECKPOINT"
fi

label_job="$(sbatch --parsable --export=ALL,MULTIVIEW_LABEL_CONFIG="$LABEL_CONFIG",SAM3_CHECKPOINT="$SAM3_CHECKPOINT" slurm/label.sbatch)"
echo "Submitted label job: $label_job"

train_job="$(sbatch --parsable --dependency=afterok:"$label_job" --export=ALL,MULTIVIEW_TRAIN_CONFIG="$TRAIN_CONFIG" slurm/train.sbatch)"
echo "Submitted train job: $train_job (afterok:$label_job)"

run_job="$(sbatch --parsable --dependency=afterok:"$train_job" --export=ALL,MULTIVIEW_RUN_CONFIG="$RUN_CONFIG" slurm/run.sbatch)"
echo "Submitted run job: $run_job (afterok:$train_job)"

echo
echo "Monitor:"
echo "  squeue -u \"$USER\""
echo "  tail -f slurm-multiview_label-$label_job.out"
echo "  tail -f slurm-multiview_train-$train_job.out"
echo "  tail -f slurm-multiview_run-$run_job.out"
