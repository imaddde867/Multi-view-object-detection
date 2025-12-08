import cv2
from ultralytics import YOLO
import json
import argparse
import os

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="results/Detection_2Class/yolov8m_2class/weights/best.pt", help="Path to trained YOLO model")
parser.add_argument("--video1", type=str, default="data/raw/testing_videos/Cam3.mp4", help="Path to first video")
parser.add_argument("--video2", type=str, default="data/raw/testing_videos/Cam4.mp4", help="Path to second video")
parser.add_argument("--out_json", type=str, default="./multi_view_detections.json", help="Output JSON path")
parser.add_argument("--out_video", type=str, default="./detection_visualization.avi", help="Output video path")
parser.add_argument("--imgsz", type=int, default=960, help="Inference image size (matches 960 training resolution)")
parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold for YOLO inference")
parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for YOLO NMS")
parser.add_argument("--slowdown", type=float, default=2.0, help="Slowdown factor for output FPS (1 keeps realtime)")
parser.add_argument("--box_shrink", type=float, default=0.0, help="Fraction (0-0.4) to shrink boxes for display/JSON")
parser.add_argument("--nms_iou", type=float, default=0.65, help="Per-camera NMS IoU to remove duplicate boxes of the same class")
args = parser.parse_args()

model_path = args.model
video1_path = args.video1
video2_path = args.video2
output_json_path = args.out_json
output_video_path = args.out_video
img_size = max(320, args.imgsz)
conf_thres = max(0.01, min(0.99, args.conf))
iou_thres = max(0.1, min(0.99, args.iou))
slowdown_factor = max(0.5, args.slowdown)
box_shrink = min(0.4, max(0.0, args.box_shrink))
nms_iou = max(0.1, min(0.95, args.nms_iou))

print(f"Model: {model_path}")
print(f"Video 1: {video1_path}")
print(f"Video 2: {video2_path}")

# Classes
CLASS_NAMES = {
    0: "person",
    1: "car"
}

# Slowdown factor (slower video)
SLOWDOWN_FACTOR = slowdown_factor


def clip_bbox(bbox, width, height):
    """Clip a bbox to frame limits and drop degenerate boxes."""
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def shrink_bbox(bbox, shrink_factor):
    """Pull bbox edges toward center for cleaner visualization."""
    if shrink_factor <= 0:
        return bbox
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    delta_w = int(width * shrink_factor / 2)
    delta_h = int(height * shrink_factor / 2)
    new_bbox = (x1 + delta_w, y1 + delta_h, x2 - delta_w, y2 - delta_h)
    return new_bbox if new_bbox[0] < new_bbox[2] and new_bbox[1] < new_bbox[3] else bbox


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def apply_nms(detections, iou_threshold):
    if len(detections) <= 1:
        return detections
    ordered = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept = []
    while ordered:
        current = ordered.pop(0)
        kept.append(current)
        ordered = [det for det in ordered
                   if det["class_id"] != current["class_id"]
                   or compute_iou(det["bbox"], current["bbox"]) < iou_threshold]
    return kept

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

    results1 = model(frame1, imgsz=img_size, conf=conf_thres, iou=iou_thres, verbose=False)[0]
    results2 = model(frame2, imgsz=img_size, conf=conf_thres, iou=iou_thres, verbose=False)[0]

    cam1_dets = []
    for bbox, conf, cls in zip(results1.boxes.xyxy.tolist(),
                               results1.boxes.conf.tolist(),
                               results1.boxes.cls.tolist()):
        class_id = int(cls)
        class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")

        clipped_box = clip_bbox(bbox, width1, height1)
        if not clipped_box:
            continue

        shrunk_box = shrink_bbox(clipped_box, box_shrink)

        cam1_dets.append({
            "bbox": list(map(int, shrunk_box)),
            "confidence": float(conf),
            "class_id": class_id,
            "class_name": class_name
        })

    cam1_dets = apply_nms(cam1_dets, nms_iou)

    for det in cam1_dets:
        x1, y1, x2, y2 = det["bbox"]
        class_name = det["class_name"]
        conf = det["confidence"]
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

        clipped_box = clip_bbox(bbox, width2, height2)
        if not clipped_box:
            continue

        shrunk_box = shrink_bbox(clipped_box, box_shrink)

        cam2_dets.append({
            "bbox": list(map(int, shrunk_box)),
            "confidence": float(conf),
            "class_id": class_id,
            "class_name": class_name
        })

    cam2_dets = apply_nms(cam2_dets, nms_iou)

    for det in cam2_dets:
        x1, y1, x2, y2 = det["bbox"]
        class_name = det["class_name"]
        conf = det["confidence"]
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
