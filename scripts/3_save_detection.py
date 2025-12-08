import cv2
from ultralytics import YOLO
import json

model_path = "/home/treenut/multi_view/Yolov8_Multi-view/Detection/05_44epochs_480img_size/weights/best.pt"
video1_path = "/home/treenut/multi_view/Yolov8_Multi-view/testing_videos/Cam3.mp4"
video2_path = "/home/treenut/multi_view/Yolov8_Multi-view/testing_videos/Cam4.mp4"
output_json_path = "./multi_view_detections.json"
output_video_path = "./detection_visualization.avi"

# Classes
CLASS_NAMES = {
    0: "person",
    1: "car"
}

# Slowdown factor (slower video)
SLOWDOWN_FACTOR = 2

model = YOLO(model_path)

cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

if not cap1.isOpened() or not cap2.isOpened():
    print("❌ Error: Cannot open video(s).")
    exit()

# Get video dimensions
width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))

# FPS settings
input_fps = int(cap1.get(cv2.CAP_PROP_FPS))
input_fps = max(1, input_fps)

output_fps = max(1, int(input_fps / SLOWDOWN_FACTOR))

print(f"🎥 Input FPS: {input_fps}")
print(f"🐌 Output FPS (slowed): {output_fps}")

out = None
matched_data = {
    "metadata": {
        "cam1": {"width": width1, "height": height1},
        "cam2": {"width": width2, "height": height2},
        "fps": input_fps
    },
    "frames": []
}
frame_num = 0

print("🎬 Processing videos and generating slowed visualization...")

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        break

    vis_frame1 = frame1.copy()
    vis_frame2 = frame2.copy()

    results1 = model(frame1)[0]
    results2 = model(frame2)[0]

    cam1_dets = []
    for bbox, conf, cls in zip(results1.boxes.xyxy.tolist(),
                               results1.boxes.conf.tolist(),
                               results1.boxes.cls.tolist()):
        class_id = int(cls)
        class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")

        cam1_dets.append({
            "bbox": bbox,
            "confidence": float(conf),
            "class_id": class_id,
            "class_name": class_name
        })

        x1, y1, x2, y2 = map(int, bbox)
        color = (0, 255, 0) if class_name == "person" else (0, 0, 255)
        cv2.rectangle(vis_frame1, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis_frame1, f"{class_name} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cam2_dets = []
    for bbox, conf, cls in zip(results2.boxes.xyxy.tolist(),
                               results2.boxes.conf.tolist(),
                               results2.boxes.cls.tolist()):
        class_id = int(cls)
        class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")

        cam2_dets.append({
            "bbox": bbox,
            "confidence": float(conf),
            "class_id": class_id,
            "class_name": class_name
        })

        x1, y1, x2, y2 = map(int, bbox)
        color = (0, 255, 0) if class_name == "person" else (0, 0, 255)
        cv2.rectangle(vis_frame2, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis_frame2, f"{class_name} {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    matched_data["frames"].append({
        "frame": frame_num,
        "cam1": cam1_dets,
        "cam2": cam2_dets
    })

    # Combine views
    h = min(vis_frame1.shape[0], vis_frame2.shape[0])
    vis1 = cv2.resize(vis_frame1, (int(vis_frame1.shape[1] * h / vis_frame1.shape[0]), h))
    vis2 = cv2.resize(vis_frame2, (int(vis_frame2.shape[1] * h / vis_frame2.shape[0]), h))
    combined = cv2.hconcat([vis1, vis2])

    # FPS slowed output
    if out is None:
        height, width = combined.shape[:2]
        out = cv2.VideoWriter(output_video_path,
                              cv2.VideoWriter_fourcc(*'XVID'),
                              output_fps,
                              (width, height))

    out.write(combined)

    frame_num += 1
    if frame_num % 50 == 0:
        print(f"  Processed {frame_num} frames...")

# Save JSON
with open(output_json_path, "w") as f:
    json.dump(matched_data, f, indent=2)

cap1.release()
cap2.release()
if out:
    out.release()
cv2.destroyAllWindows()

print(f"\n✅ JSON saved to {output_json_path}")
print(f"🐌 Slowed video saved to {output_video_path}")
print(f"📊 Total frames processed: {frame_num}")
print(f"📊 Cam1 total detections: {sum(len(f['cam1']) for f in matched_data['frames'])}")
print(f"📊 Cam2 total detections: {sum(len(f['cam2']) for f in matched_data['frames'])}")