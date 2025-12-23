# Cleanup Summary

- Consolidated YOLO loading into `multiview/utils/yolo.py` and removed dead association logic while keeping the CLI pipeline intact (`label`, `train`, `run`, `verify`).
- Pruned configs to the minimal set: `labeling.yaml`, `train.yaml`, `system_demo.yaml`, `system_demo_tuned.yaml`. SLURM defaults now point at the tuned demo; override `MULTIVIEW_RUN_CONFIG` for custom runs.
- Dropped generated/unused artifacts (`multiview.egg-info/`, `GEMINI.md`, `notes.txt`, `config/system.yaml`, `config/system_puhti_*`).

## Removed Files (and why)
- `config/system.yaml`, `config/system_puhti_g78_tuned.yaml`, `config/system_puhti_spatial.yaml`: outdated/duplicate run configs; demos + your copies cover system runs.
- `multiview.egg-info/`: build artifact.
- `GEMINI.md`, `notes.txt`: unused placeholders.

## Dependency Audit
- Runtime deps (unchanged): `ultralytics`, `opencv-python`, `numpy`, `scipy`, `pyyaml`, `tqdm`.
- Optional extras: `torch`, `torchvision` for tracking embeddings; `sam3` from source for labeling.

## Module Structure (post-cleanup)
- `multiview/cli.py` — CLI entrypoint for label/train/run/verify.
- `multiview/labeling/` — YOLO proposals + SAM3 refinement (`pipeline.py`, `proposals.py`, `sam3_refiner.py`, `masks.py`).
- `multiview/training/train.py` — YOLO training wrapper.
- `multiview/system/run.py` — multi-view detection + tracking, global association, overlays.
- `multiview/tracking/` — per-camera tracker, global ID assigner, embedders.
- `multiview/dataset/verify.py` — dataset format checks.
- `multiview/utils/` — boxes, video IO, splits, YAML helpers, shared YOLO loader.
- `scripts/debug_global_association.py` — association sanity checks.
- `config/` — `labeling.yaml`, `train.yaml`, `system_demo.yaml`, `system_demo_tuned.yaml`.
- `slurm/` — label/train/run job scripts and pipeline submit helper.

## Breaking Changes
- Run configs trimmed to demos; copy a `config/system_demo*.yaml` file for custom runs and point SLURM via `MULTIVIEW_RUN_CONFIG`.
- SLURM `run.sbatch` / `submit_pipeline.sh` now default to `config/system_demo_tuned.yaml`.
