# Multi-View Object Detection & Tracking (SAM3 -> YOLO -> Multi-Camera)

End-to-end pipeline for multi-view object detection and tracking across synchronized cameras:
1) Auto-label videos with YOLO proposals + SAM3 refinement.
2) Train a YOLO detector.
3) Run multi-view detection + tracking with camera groups.

## Quick look

![Multi-view tracking overlay](results/showcase/training/sam3_autolabel_v2/val_batch0_pred.jpg)

## Quickstart: run the demo clips

Option A (tuned; uses repo-root `demo1.mov` + `demo2.mp4`; requires `torch/torchvision`):

```bash
multiview run --config config/system_demo_tuned.yaml
```

Option B (colorhist embedder; keep inputs under `data/raw/`):

```bash
mkdir -p data/raw/demo_videos
ln -s "$PWD/demo1.mov" data/raw/demo_videos/demo1.mov
ln -s "$PWD/demo2.mp4" data/raw/demo_videos/demo2.mp4

multiview run --config config/system_demo.yaml
```

Outputs (per run):
- `results/system/demo_tuned/demo.json` + `results/system/demo_tuned/demo.avi`
- `results/system/demo_run/demo.json` + `results/system/demo_run/demo.avi`

If you do not have CUDA, set `runtime.device: cpu` and `runtime.half: false` in the config.

If `.mov` fails to open, convert once (requires `ffmpeg`) and update your camera path:

```bash
ffmpeg -i demo1.mov -c:v libx264 -pix_fmt yuv420p data/raw/demo_videos/demo1.mp4
```

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

Optional:
- `pip install -e sam3` to enable SAM3 labeling.
- `pip install -e ".[tracking]"` to enable the `torch_resnet18` embedder (used by `config/system_demo_tuned.yaml`).

## Project layout

- `multiview/` core pipeline + CLI.
- `config/` configs for labeling (`labeling.yaml`), training (`train.yaml`), and the two demo system runs (`system_demo*.yaml`; copy one as your template).
- `data/raw/` raw videos and ground-truth assets (`testing_videos/`, `multiclass_ground_truth/`, `multiclass_ground_truth_images/`).
- `data/processed/` full datasets (local); `data/processed/showcase/` small tracked samples.
- `checkpoints/yolo/` local YOLO base weights.
- `checkpoints/sam3/` SAM3 checkpoints (place `sam3.pt` here).
- `results/training/` training runs (includes `weights/best.pt`).
- `results/system/` system outputs (videos + JSON; either per-run folders or group files).
- `results/showcase/` curated artifacts for demos.
- `runs/` MLflow runs.
- `sam3/` SAM3 source (editable install for labeling).
- `scripts/` debugging utilities.
- `slurm/` Puhti job scripts.

## Pipeline

### 1) Label videos (SAM3 -> YOLO)

Edit `config/labeling.yaml`, then:

```bash
multiview label --config config/labeling.yaml
```

Override the SAM3 checkpoint if needed:
`multiview label --config config/labeling.yaml --sam3-checkpoint /path/to/sam3.pt`

Dataset layout:

```text
data/processed/<dataset_name>/
  dataset.yaml
  train/images/*.jpg
  train/labels/*.txt
  val/...
  test/...
  meta.json
  stats.json
```

Verify:

```bash
multiview verify --dataset data/processed/<dataset_name>/dataset.yaml
```

Defaults (see `config/labeling.yaml`):
- Classes: `person,car,bus`
- Proposal remap: `truck/motorcycle/bicycle -> car`
- Proposal model: `checkpoints/yolo/yolo11m.pt`
- SAM3 checkpoint: `checkpoints/sam3/sam3.pt`

### 2) Train YOLO

Edit `config/train.yaml`, then:

```bash
multiview train --config config/train.yaml
```

### 3) Run multi-view detection + tracking

For demos, use:

```bash
multiview run --config config/system_demo_tuned.yaml
```

For custom camera groups, copy one of the demo configs, update `cameras` + `groups`, and pass it via `--config`.

Debug global ID association by adding to your config:

```yaml
debug:
  global_assoc: true
  log_path: results/system/demo_run/demo_global_assoc.jsonl # optional
```

The JSONL log includes per-frame local/global IDs, embeddings, cost matrices, and accept/reject decisions. Video overlays now show `G<global_id> L<local_id>` to verify failures quickly.

Optional synthetic sanity check:

```bash
python scripts/debug_global_association.py
```

## Showcase (tracked)

- `data/processed/showcase/sam3_autolabel_v2/` (viz + metadata)
- `results/showcase/training/sam3_autolabel_v2/`
- `results/showcase/system/sam3_autolabel_v2/` (`g34_demo.mp4`, `g34.json`)

## Data + git

- `data/raw/`, `data/processed/`, and most of `results/` are ignored by git.
- Local assets also live in `checkpoints/`, `sam3/`, and `runs/` (kept out of git).
- Root `demo*.mp4` / `demo*.mov` are ignored; store working videos under `data/raw/`.

## SLURM (Puhti)

Use the job scripts in `slurm/`:

```bash
bash slurm/submit_pipeline.sh
```

Defaults target the demo configs (`system_demo_tuned.yaml` for the run step); override `MULTIVIEW_*_CONFIG` to point at your own YAMLs.
