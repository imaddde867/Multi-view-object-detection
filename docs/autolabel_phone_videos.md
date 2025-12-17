# Auto-Label Videos (YOLO proposals → SAM3-tight YOLO boxes)

This repo can generate YOLO-format annotations from your own phone videos by:

1. Running a YOLO detector to propose **person / car / bus** candidates
2. Using **SAM3** to refine each proposal into a segmentation mask
3. Converting each mask to a **tight bounding box** and exporting **YOLO labels**

The main entry point is: `mvot label`

## 1) Install dependencies

Base dependencies (already used by this project):

```bash
pip install -r requirements.txt
pip install -e .
```

SAM3 is required for labeling (installed from source):

- Follow https://github.com/facebookresearch/sam3
- Ensure `sam3` is importable in the same Python environment as this repo
- Download/locate a SAM3 checkpoint file (e.g., `sam3.pt`)

## 2) Run auto-labeling on videos

Example (2 phone videos):

```bash
mvot label \
  --videos data/raw/phone/CamA.mp4 data/raw/phone/CamB.mp4 \
  --out data/processed/phone_autolabel_sam3 \
  --proposal-model yolov8n.pt \
  --targets person,car,bus \
  --source-map truck=car,motorcycle=car,bicycle=car \
  --conf 0.35 --iou 0.5 --frame-stride 5 \
  --sam3-checkpoint /absolute/path/to/sam3.pt \
  --save-viz
```

Tip: edit `config/labeling.yaml` instead of passing many CLI flags.

## 3) Outputs

The script writes a standard Ultralytics dataset:

```text
data/processed/phone_autolabel_sam3/
  dataset.yaml
  meta.json
  stats.json
  train/
    images/*.jpg
    labels/*.txt
  val/
    images/*.jpg
    labels/*.txt
  test/
    images/*.jpg
    labels/*.txt
  viz/                # only if --save-viz
```

Each label file is YOLO format:

`class_id x_center y_center width height` (all normalized to `[0..1]`).

## 4) Train YOLO on the generated dataset

```bash
mvot train --data data/processed/phone_autolabel_sam3/dataset.yaml --model yolo11m.pt --epochs 100 --imgsz 960
```

## Notes / tuning

- If you get too many false positives, increase `--conf` (e.g. `0.45`) and/or increase `--min-box-area`.
- If boxes are slightly “fat”, increase `sam3.mask_close` (e.g. `7` or `9`) in `config/labeling.yaml` to clean small holes before boxing.
- For faster labeling, increase `--frame-stride` (e.g. `10` → ~3 FPS on 30 FPS videos).
