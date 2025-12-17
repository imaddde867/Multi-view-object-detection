# End-to-End Pipeline (SAM3 → YOLO → Multi-View Tracking)

This repository provides a clean, modular pipeline:

1. Label videos with **YOLO proposals + SAM3 mask refinement** and export **YOLO-format boxes**
2. Train a stronger YOLO detector on the generated dataset
3. Run multi-camera detection + tracking with configurable camera groups

## 0) Environment

```bash
pip install -r requirements.txt
pip install -e .
```

### SAM3 (required for labeling)

SAM3 is installed from source. Follow the official instructions:
https://github.com/facebookresearch/sam3

Then set `sam3.checkpoint` in `config/labeling.yaml` or pass `--sam3-checkpoint`.

## 1) Label videos (YOLO format)

```bash
mvot label --config config/labeling.yaml --sam3-checkpoint /path/to/sam3.pt
```

## 2) Train YOLO

```bash
mvot train --config config/train.yaml
```

Tip: set `runtime.device: "0,1"` (or `"0,1,2,3"`) for multi-GPU training.

## 3) Run multi-camera detect + track

Edit `config/system.yaml`:

- `cameras`: all video sources (or live sources if supported by OpenCV)
- `groups`: camera pairs/groups like `[cam1, cam2]`, `[cam3, cam4]`, etc.

Then:

```bash
mvot run --config config/system.yaml
```

Outputs:

- `results/system/<group>.json`
- `results/system/<group>.avi` (if `output.write_video: true`)

