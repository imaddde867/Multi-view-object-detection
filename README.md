# Multi-View Object Detection & Tracking (SAM3 → YOLO → Multi-Camera)

This repo implements an end-to-end pipeline for multi-view detection and tracking across synchronized cameras:

1) Label videos with YOLO proposals + SAM3 mask refinement.
2) Train a YOLO detector.
3) Run multi-view detection + tracking with camera groups.

## Visual showcase

<p align="center">
  <img src="data/processed/showcase/sam3_autolabel_v2/viz/val/Cam3_f000070.jpg" width="49%" alt="SAM3 auto-labeling visualization sample" />
  <img src="results/showcase/training/sam3_autolabel_v2/val_batch0_pred.jpg" width="49%" alt="YOLO validation predictions sample" />
</p>

<p align="center">
  <em>Left: SAM3 auto-labeling overlay. Right: YOLO validation predictions.</em>
</p>

## Demo (no training required)

Use the tracked artifacts for a quick walkthrough:

- Labeling samples: `data/processed/showcase/sam3_autolabel_v2/viz/val/`
- Training metrics: `results/showcase/training/sam3_autolabel_v2/results.png`
- Tracking output: `results/showcase/system/sam3_autolabel_v2/g34_demo.mp4`

## Showcase artifacts (tracked)

- Labeling samples: `data/processed/showcase/sam3_autolabel_v2/` (viz + metadata)
- Training metrics: `results/showcase/training/sam3_autolabel_v2/`
- Evaluation demo: `results/showcase/system/sam3_autolabel_v2/` (`g34_demo.mp4`, `g34.json`)

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

Optional:
- `pip install -e ".[tracking]"` to enable the `torch_resnet18` embedder.

## SAM3

SAM3 is installed from source. Follow https://github.com/facebookresearch/sam3 and ensure `sam3` is importable in the same environment.

## 1) Label videos (SAM3 → YOLO)

Edit `config/labeling.yaml`, then run:

```bash
multiview label --config config/labeling.yaml --sam3-checkpoint /path/to/sam3.pt
```

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

## 2) Train YOLO

Edit `config/train.yaml`, then:

```bash
multiview train --config config/train.yaml
```

## 3) Run multi-view detection + tracking

Edit `config/system.yaml`, then:

```bash
multiview run --config config/system.yaml
```

Outputs per-group JSON and optional videos.

## Data and outputs

- `data/raw/` and most of `data/processed/` are ignored by git.
- Track curated samples in `data/processed/showcase/`.
- Most of `results/` is ignored; track demos and proof artifacts in `results/showcase/`.

Showcase paths:
- Labeling: set `out: data/processed/showcase/<dataset_name>` in `config/labeling.yaml`.
- Training: set `project: results/showcase/training` in `config/train.yaml`.
- Evaluation: set `output.dir: results/showcase/system/<run_name>` in `config/system.yaml`.

## SLURM

See `slurm/label.sbatch`, `slurm/train.sbatch`, and `slurm/run.sbatch`.
Submit the full pipeline with:

```bash
bash slurm/submit_pipeline.sh
```
