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

## Demo Playbook

Keep these commands handy for tomorrow's presentation so you can jump between pre-recorded and live feeds without editing any scripts.

### Dual-Video Playback (Cam3 + Cam4)

```bash
python scripts/3_save_detection.py \
  --model demo_material/yolov8m_best.pt \
  --video1 data/raw/testing_videos/Cam3.mp4 \
  --video2 data/raw/testing_videos/Cam4.mp4 \
  --out_json demo_material/demo_cam34.json \
  --out_video demo_material/demo_cam34.avi \
  --imgsz 960 --conf 0.35 --iou 0.5 --slowdown 1 --box_shrink 0.15
```

* `--imgsz 960` matches training resolution for tighter boxes.
* `--conf`/`--iou` can be nudged on the fly (e.g., `--conf 0.45` if you see flicker).
* Set `--slowdown 2` if you prefer a slower side-by-side playback.
* `--box_shrink 0.1-0.2` trims boxes inward so people/car overlays stay tight even when the raw YOLO box includes shadows.

### Live Camera Sanity Check

```bash
yolo predict model=demo_material/yolov8m_best.pt source=0 show=True conf=0.35
```

Run it once per camera ID (`source=1`, `source=2`, …) to verify the lab hardware before switching to your dual-camera helper or the script above.

## Experiments
Detailed logs of data leakage issues and class imbalance experiments can be found in `docs/results.md`.
