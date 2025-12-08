# Final Project Results: Multi-View Object Detection

## Executive Summary
This project successfully developed a high-performance multi-view object detection system capable of accurately detecting pedestrians and vehicles across multiple synchronized camera feeds. By optimizing the data pipeline and model architecture, we achieved state-of-the-art performance metrics.

## Key Achievements

1.  **Robust 2-Class Detection:** Shifted focus to a robust "Person + Car" model, leveraging the full dataset (1370 frames) instead of a balanced subset (186 frames). This resulted in a **5x increase in training data**.
2.  **Advanced Data Preprocessing:** Implemented bounding box clipping to recover valid training samples that were previously discarded due to being partially off-screen.
3.  **High-Resolution Training:** Trained a **YOLOv8 Medium** model at **960x960** resolution (up from 640p Nano), significantly improving small object detection.

## Performance Metrics (Final Model)

The final model (`yolov8m_2class`) trained for 50 epochs yielded exceptional results:

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **mAP50** | **97.9%** | Near-perfect detection accuracy at standard overlap. |
| **mAP50-95** | **92.1%** | Extremely high precision even at strict localization thresholds. |
| **Precision** | **97.6%** | Very few false positives (ghost detections). |
| **Recall** | **95.1%** | Misses almost no real objects in the scene. |

### Training Progression
The model showed steady convergence with no signs of overfitting, as evidenced by the validation loss decreasing alongside training loss.

![Training Metrics](../demo_material/training_metrics.csv) *(Raw data available in demo_material)*

## Deliverables

The following assets have been generated and archived in `demo_material/`:

*   **`detection_demo.avi`**: A side-by-side video visualization of the model performing inference on Camera 3 and Camera 4 test feeds.
*   **`yolov8m_best.pt`**: The trained model weights, ready for real-time inference.
*   **`training_metrics.csv`**: Full log of training performance across 50 epochs.

## Conclusion
The transition to a 2-class system with full data utilization and higher resolution training proved to be the winning strategy. The system is now ready for live demonstration.