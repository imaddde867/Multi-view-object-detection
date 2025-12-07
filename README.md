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

## Experiments
Detailed logs of data leakage issues and class imbalance experiments can be found in `docs/results.md`.