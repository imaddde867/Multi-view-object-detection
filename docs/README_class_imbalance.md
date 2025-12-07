# Class Imbalance & Model Improvement Experiments

This document describes the steps taken to address the severe class imbalance (few buses vs. many cars/persons) and instructions for running the comparative experiments.

## 1. Problem
The dataset is heavily imbalanced:
- ~3000 cars
- ~1300 persons
- ~64 buses (very few unique frames)

This causes the model to likely ignore buses or misclassify background as objects.

## 2. Solutions Implemented

### Experiment A: Balanced 3-Class Training
We created a new data preparation script `2_Read_images_labels_balanced.py` that:
1.  Identifies frames containing **Bus** instances.
2.  **Oversamples** these frames in the training set (repeats them 15x).
3.  **Undersamples** the majority frames (keeps 50%) to reduce noise and dominance of cars.
4.  Ensures validation and test sets remain distinct (no data leakage).

**To run this experiment:**
```bash
# 1. Generate labels (if not done)
python 1_EPFL_Yolov8_all_class.py

# 2. Create balanced dataset
python 2_Read_images_labels_balanced.py

# 3. Train
python train_yolo.py --mode balanced --epochs 50
```

### Experiment B: 2-Class Model (Person + Car)
We exclude the "Bus" class entirely to see if the model performs better on the core classes without the confusion of the underrepresented class.

**To run this experiment:**
```bash
# 1. Generate labels for 2 classes
python 1_EPFL_YOLO_car_person.py

# 2. Create dataset
python 2_Read_images_labels_car_person.py

# 3. Train
python train_yolo.py --mode 2class --epochs 50
```

## 3. Requirements
Install the YOLOv8 library:
```bash
pip install ultralytics
```

## 4. File Structure
- `2_Read_images_labels_balanced.py`: **NEW** Script for balanced splitting.
- `train_yolo.py`: **NEW** Unified training script.
- `data_train_balanced/`: Output folder for Exp A (Ignored by git).
- `data_train_val_test_car_person/`: Output folder for Exp B (Ignored by git).
