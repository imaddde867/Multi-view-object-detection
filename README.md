# Multi-View Object Detection & Tracking (SAM3 → YOLO → Multi-Camera)

Production-oriented, end-to-end pipeline:

1) **SAM3-based video labeling** (YOLO proposals → SAM3 mask refinement → tight YOLO boxes)  
2) **YOLO training** (newer/stronger models supported)  
3) **Multi-view detection + tracking** with configurable camera groups (1+2, 3+4, 5+6, …)

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

Optional extras:
- `pip install -e ".[tracking]"` to enable the `torch_resnet18` embedder.

### SAM3 install (required for labeling)

SAM3 is installed from source. Follow https://github.com/facebookresearch/sam3 and ensure `sam3` is importable in the same environment as this repo.

## 1) Data generation (SAM3 → YOLO format)

Edit `config/labeling.yaml` (videos, output path, SAM3 checkpoint), then run:

```bash
multiview label --config config/labeling.yaml --sam3-checkpoint /path/to/sam3.pt
```

Outputs a standard Ultralytics dataset (location comes from `out:` in the config):

```text
data/processed/sam3_autolabel_allcams/
  dataset.yaml
  train/images/*.jpg
  train/labels/*.txt
  val/...
  test/...
  meta.json
  stats.json
```

Sanity-check the dataset (format + multi-view consistency):

```bash
multiview verify --dataset data/processed/sam3_autolabel_allcams/dataset.yaml
```

## 2) Train a stronger YOLO model

Edit `config/train.yaml`, then:

```bash
multiview train --config config/train.yaml
```

Notes:
- Default `model: yolov8n.pt` is configurable; set to any Ultralytics-supported checkpoint.
- Multi-GPU: set `runtime.device: "0,1"` (or `"0,1,2,3"`) in `config/train.yaml` and request multiple GPUs in SLURM.

## 3) Multi-view detection + tracking (camera pairs/groups)

Define cameras + groups in `config/system.yaml`, then:

```bash
multiview run --config config/system.yaml
```

This writes per-group JSON to `results/system/` and (optionally) rendered videos.

## Data and outputs

- `data/raw/` and most of `data/processed/` are ignored by git.
- Track curated samples in `data/processed/showcase/`.
- Most of `results/` is ignored; track demos and proof artifacts in `results/showcase/`.

## SLURM

Example job files:
- `slurm/label.sbatch`
- `slurm/train.sbatch`
- `slurm/run.sbatch`

End-to-end submission helper:
- `bash slurm/submit_pipeline.sh`
