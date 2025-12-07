# Experiment Results: Class Imbalance vs. Simplified Task

This document summarizes the results of two experiments conducted to address severe class imbalance in the multi-view object detection dataset.

## Dataset Statistics
- **Cars:** ~3000 instances
- **Persons:** ~1300 instances
- **Buses:** ~64 instances (Severe Imbalance)

## Experiment 1: Balanced 3-Class Training
**Strategy:** Oversampled bus frames (15x) and undersampled majority frames (0.5x) to create a balanced training set.

| Class | Precision | Recall | mAP50 | mAP50-95 |
| :--- | :--- | :--- | :--- | :--- |
| **All** | 0.894 | 0.821 | 0.86 | 0.472 |
| **Person** | 0.836 | 0.488 | 0.601 | 0.252 |
| **Car** | 0.984 | 0.975 | 0.985 | 0.667 |
| **Bus** | 0.862 | **1.00** | 0.995 | 0.496 |

**Key Finding:** Oversampling was highly effective for the minority class (Bus), achieving **100% recall**. However, "Person" detection suffered (low recall/mAP), likely due to model capacity being split or confusion introduced by the balancing techniques.

## Experiment 2: 2-Class Training (Person + Car)
**Strategy:** Removed the "Bus" class entirely to focus the model on the two core classes.

| Class | Precision | Recall | mAP50 | mAP50-95 |
| :--- | :--- | :--- | :--- | :--- |
| **All** | 0.911 | 0.874 | **0.898** | **0.656** |
| **Person** | 0.897 | 0.836 | **0.864** | **0.586** |
| **Car** | 0.926 | 0.912 | 0.931 | 0.727 |

**Key Finding:** Removing the problematic minority class led to a **massive improvement** in Person detection (**mAP50 +26%**, **mAP50-95 +33%**). The overall model quality (mAP50-95) increased significantly from 0.472 to 0.656.

## Conclusion
1.  **Oversampling** is a viable strategy if detecting the rare class (Bus) is critical; it successfully forced the model to learn the class.
2.  **Task Simplification** (2-Class) is superior if the goal is general detection performance for the majority classes. The complexity cost of the third, imbalanced class was heavily penalizing the performance on pedestrians.
