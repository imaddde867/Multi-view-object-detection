import cv2
from ultralytics import YOLO
import json

model_path = "results/Detection/03_10epochs_416img/detect/03_10epochs_416img/weights/best.pt"
video1_path = "data/raw/testing_videos/Cam3.mp4"
video2_path = "data/raw/testing_videos/Cam4.mp4"
output_json_path = "results/multi_view_detections.json"

model = YOLO(model_path)

cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

if not cap1.isOpened() or not cap2.isOpened():
    print("Error: Could not open one or both videos.")
    exit()

matched_data = []
frame_num = 0

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        break

    results1 = model(frame1)[0]
    results2 = model(frame2)[0]

    cam1_dets = []
    for bbox, conf, cls in zip(results1.boxes.xyxy.tolist(),
                               results1.boxes.conf.tolist(),
                               results1.boxes.cls.tolist()):
        cam1_dets.append({
            "bbox": bbox,
            "conf": float(conf),
            "class": int(cls)
        })

    cam2_dets = []
    for bbox, conf, cls in zip(results2.boxes.xyxy.tolist(),
                               results2.boxes.conf.tolist(),
                               results2.boxes.cls.tolist()):
        cam2_dets.append({
            "bbox": bbox,
            "conf": float(conf),
            "class": int(cls)
        })

    matched_data.append({
        "frame": frame_num,
        "cam1": cam1_dets,
        "cam2": cam2_dets
    })

    frame_num += 1

with open(output_json_path, "w") as f:
    json.dump(matched_data, f, indent=2)

cap1.release()
cap2.release()
print(f"✅ Detections saved to {output_json_path}")
