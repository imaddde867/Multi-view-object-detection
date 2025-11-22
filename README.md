# Multi-View 3D Object Detection

Full EPFL Terrace pipeline: YOLO detections on all six cameras → cross-view matching → DLT triangulation with depth/reprojection checks.

## Quick Start

```bash
pip install -r requirements.txt
python create_calibration.py            # writes epfl-calibration.json
python main.py --frame 150             # multi-view triangulation for a single frame
```

Useful CLI flags: `--all-frames`, `--max-frames N`, `--min-conf`, `--pixel-tolerance`, `--merge-distance`, `--labels-dir PATH`, `--output-obj cloud.obj`.

## Pipeline

1. `yolo_loader.py` reads Nat's YOLO outputs (`Yolov8_Multi-view-main/yolo_labels`) and converts them to pixel-space detections for each camera.
2. `multiview_matching.py` pairs same-class detections between cameras, rejects bad matches with reprojection error, and adds supporting views.
3. `Triangulator` validates depths, triangulates pairs/batches, and exports 3D points via `main.py`.

Result: `output_cloud.obj` plus per-frame stats.

## YOLO Utilities

Located under `Yolov8_Multi-view-main/`:

- `EPFL_to_Yolov8.py` – convert EPFL labels → YOLO format using the shared dataset root.
- `Read_images_labels.py` – split YOLO labels/images into `data_train-test/`.
- `Check_train_test.py`, `Classes_check.py` – sanity checks for the split.
- `First_detect/`, `Second_detect/` – Nat’s training/detection artifacts (weights, logs).

All scripts assume the dataset folders at repo root (`multiclass_ground_truth_images`, `bounding_boxes_EPFL_cross`).

## Triangulator API

```python
from triangulation import Triangulator

triangulator = Triangulator(projection_matrices)
points_3d = triangulator.triangulate_points(0, 1, pts_cam0, pts_cam1)
errors, mean_error = triangulator.compute_reprojection_error(points_3d[0], [(u0, v0), (u1, v1)])
```

Helpers: `triangulate_dlt`, `project_point`, `compute_reprojection_error`, `is_in_front_of_camera`.

## Repo Layout

- `main.py`, `multiview_matching.py`, `yolo_loader.py`, `triangulation.py`: primary pipeline.
- `Yolov8_Multi-view-main/`: YOLO training/inference utilities & outputs.
- `legacy/feature_detection.py`: historical ORB feature code retained for reference.

⚠️ Objects must be visible in ≥2 cameras to be triangulated.
