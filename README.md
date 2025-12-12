# Multi-View Object Tracking

This project implements a multi-view object detection and tracking pipeline using **YOLOv8** and **DeepSORT**. It specifically addresses severe class imbalance (rare "Bus" class vs. frequent "Car/Person" classes) through data balancing strategies and cross-camera matching.

## Project Structure

The project has been reorganized for clarity. **Always run scripts from the project root.**

```text
├── config/           # Configuration files (data.yaml)
├── data/             # Data storage
│   ├── raw/          # Original images, videos, and ground truth
│   └── processed/    # Generated YOLO datasets (train/val/test splits)
├── scripts/          # Python executables (Data prep, Training, Tracking)
├── results/          # Training runs, matched JSONs, and output videos
├── docs/             # Documentation and experiment logs
└── requirements.txt  # Project dependencies
```

## Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Important Usage Note**
   All scripts must be executed from the **root directory** to ensure file paths resolve correctly:
   ```bash
   # CORRECT 
   python scripts/train_yolo.py ...

   # INCORRECT
   cd scripts && python train_yolo.py ...
   ```

## Workflow

### 1. Data Preparation
Choose an experiment strategy to generate the dataset.

*   **Option A: Balanced 3-Class (Person, Car, Bus)**
    *   *Strategy:* Oversamples rare Bus frames (15x), undersamples common frames.
    ```bash
    python scripts/2_Read_images_labels_balanced.py
    ```

*   **Option B: Standard 2-Class (Person, Car only)**
    *   *Strategy:* Removes Bus class entirely to focus on core performance.
    ```bash
    python scripts/2_Read_images_labels_car_person.py
    ```

### 2. Training
Train the YOLOv8 model using the generated dataset.

```bash
# For Balanced dataset (Option A)
python scripts/train_yolo.py --mode balanced --epochs 50

# For 2-Class dataset (Option B)
python scripts/train_yolo.py --mode 2class --epochs 50
```
*Results are saved to `results/Detection_Balanced` or `results/Detection_2Class`.*

### 3. Cross-Camera Matching & Tracking
After training, use the best weights to detect and track objects across camera views.

1.  **Match Detections (Cam 1 & Cam 2):**
    ```bash
    python scripts/4_matching_cam1_cam2.py
    ```
2.  **Run DeepSORT Tracking:**
    ```bash
python scripts/5_track_matched.py
```

## Immediate Fixes (Quick Wins)

**Fix 1 – Aggressive Data Augmentation (`scripts/train_yolo.py`)**  
Phone footage swings wildly in lighting, color balance, and motion blur, so the YOLO training call now injects heavy visual perturbations (HSV shifts, ±10° tilt, ±10% translations, 0.5 scale swings, 50% horizontal flips, mosaic/mixup/copy-paste, and light blur). The training block is:

```python
results = model.train(
    data=dataset_path,
    epochs=epochs,
    imgsz=img_size,
    batch=batch_size,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0005,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    blur=0.01,
    patience=10,
    device=device,
)
```

**Fix 2 – Phone-Mode Detector (`scripts/3_save_detection.py`)**  
Supply `--phone-mode` and the script switches to `detect_objects_phone_mode`, which lowers YOLO to `conf=0.25`/`iou=0.45`, keeps the same 960 px inference size, and reuses DeepSORT-friendly JSON exports. Use it whenever the inputs are handheld or low light:

```bash
python scripts/3_save_detection.py --phone-mode --video1 <phone_cam_A> --video2 <phone_cam_B>
```

**Fix 3 – Phone-Mode / Independent Matching (`scripts/4_matching_cam1_cam2.py`)**  
Handheld captures break the old geometry-based matching, so two escape hatches now exist:

1. `--phone-mode` (existing) switches to appearance/ReID matching with a tunable `--appearance-threshold`.
2. `--independent-tracking` skips cross-camera matching entirely and hands out disjoint ID ranges (Cam1 → 1‑10000, Cam2 → 10001‑20000) so each stream can be narrated separately without spurious correspondences.

## Demo Playbook

Keep these commands handy for tomorrow's presentation so you can jump between pre-recorded and live feeds without editing any scripts.

### Ready-to-Show Assets

* `demo_material/demo_cam34.avi` – side-by-side Cam3/Cam4 detections (latest render).
* `demo_material/detection_demo.avi` – curated inference highlight reel.
* `demo_material/training_metrics.csv` – raw training curves linked from `docs/results.md`.

Regenerate `demo_cam34.avi` anytime with the command below so your narrative matches the latest thresholds.

### Dual-Video Playback (Cam3 + Cam4)

```bash
python scripts/3_save_detection.py \
  --model demo_material/yolov8m_best.pt \
  --video1 data/raw/testing_videos/Cam3.mp4 \
  --video2 data/raw/testing_videos/Cam4.mp4 \
  --out_json demo_material/demo_cam34.json \
  --out_video demo_material/demo_cam34.avi \
  --imgsz 960 --conf 0.35 --iou 0.5 --slowdown 1 \
  --box_shrink 0.15 --nms_iou 0.6 --frame_stride 1
```

* `--imgsz 960` matches training resolution for tighter boxes.
* `--conf`/`--iou` can be nudged on the fly (e.g., `--conf 0.45` if you see flicker).
* Set `--slowdown 2` if you prefer a slower side-by-side playback.
* `--box_shrink 0.1-0.2` trims boxes inward so people/car overlays stay tight even when the raw YOLO box includes shadows.
* `--nms_iou 0.5-0.7` applies an extra per-camera suppression pass to eliminate duplicate boxes on the same subject.
* `--frame_stride 2` halves the number of processed frames for faster turnaround, and `--skip_json` writes video only (no JSON log).
* Use `--device cuda:0 --half` on GPU machines to force CUDA/FP16 and squeeze out extra FPS.

### Live Dual-Camera Demo

```bash
python scripts/live_dual_demo.py \
  --model demo_material/yolov8m_best.pt \
  --source1 0 --source2 1 \
  --imgsz 896 --conf 0.4 --iou 0.45 \
  --box_shrink 0.15 --nms_iou 0.6 \
  --device cuda:0 --half
```

* Use numeric IDs (`0`, `1`, …) for webcams/capture cards, or drop in video file paths.
* Press `q` to exit; add `--record_out demo_material/live_capture.avi` to archive the combined feed.
* If CUDA is unavailable, omit `--device/--half` and the script will fall back to CPU.

### Live Camera Sanity Check

```bash
yolo predict model=demo_material/yolov8m_best.pt source=0 show=True conf=0.35
```

Run it once per camera ID (`source=1`, `source=2`, …) to verify the lab hardware before switching to your dual-camera helper or the script above.

## Experiments
Detailed logs of data leakage issues and class imbalance experiments can be found in `docs/results.md`.
