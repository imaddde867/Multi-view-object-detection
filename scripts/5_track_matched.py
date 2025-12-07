import cv2
import json
import random
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- Paths ---
matched_json_path = "results/multi_view_detections_matched.json"
tracked_json_path = "results/multi_view_tracked_objects.json"
video1_path = "data/raw/testing_videos/Cam3.mp4"
video2_path = "data/raw/testing_videos/Cam4.mp4"
output_video_path = "results/multi_view_1st_tracked_visualization.avi"

# --- Load matched detections ---
with open(matched_json_path, "r") as f:
    matched_data = json.load(f)

# --- Get video dimensions ---
cap_temp = cv2.VideoCapture(video1_path)
cam1_width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
cam1_height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap_temp.release()

cap_temp = cv2.VideoCapture(video2_path)
cam2_width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
cam2_height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap_temp.release()

print(f"📐 Cam1: {cam1_width}x{cam1_height}, Cam2: {cam2_width}x{cam2_height}")

# --- Initialize DeepSORT ---
tracker = DeepSort(max_age=30)

# --- Colors ---
random.seed(42)
colors = {}

tracked_data_output = []

print("\n🔍 Running DeepSORT tracking on matched detections...")

# Read Cam1 frames for ReID
cap1 = cv2.VideoCapture(video1_path)

# --- Reduce FPS / skip frames to save memory ---
frame_skip = 1  # process every frame, increase if memory issues

for frame_num, frame_info in enumerate(matched_data):
    ret, frame_cam1 = cap1.read()
    if not ret:
        print("⚠️ End of Cam1 video.")
        break

    if frame_num % frame_skip != 0:
        continue

    # Optional: downscale to reduce memory
    frame_cam1 = cv2.resize(frame_cam1, (cam1_width//2, cam1_height//2))
    scale_x = cam1_width / frame_cam1.shape[1]
    scale_y = cam1_height / frame_cam1.shape[0]

    objects = frame_info["objects"]
    detections = []
    object_map = []

    for obj in objects:
        # Only track classes 0=person, 1=car, 2=bus
        if obj.get("class") not in [0, 1, 2]:
            continue

        bbox = obj.get("cam1_bbox")
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(int, bbox)
        # scale down bbox
        x1 = int(x1 / scale_x)
        y1 = int(y1 / scale_y)
        x2 = int(x2 / scale_x)
        y2 = int(y2 / scale_y)

        if x2 <= x1 or y2 <= y1:
            continue

        conf = obj.get("confidence", 0.9)
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, obj.get("class")))
        object_map.append(obj)

    tracks = tracker.update_tracks(detections, frame=frame_cam1)
    tracked_objects = []

    for t in tracks:
        if not t.is_confirmed():
            continue
        track_id = t.track_id
        l, t1, r, b = t.to_ltrb()
        tracker_bbox = [int(l), int(t1), int(r), int(b)]

        # Match with original using IoU
        best_match_obj = None
        best_iou = 0
        for obj in object_map:
            ox1, oy1, ox2, oy2 = obj["cam1_bbox"]
            # scale down original bbox
            ox1 = int(ox1 / scale_x)
            oy1 = int(oy1 / scale_y)
            ox2 = int(ox2 / scale_x)
            oy2 = int(oy2 / scale_y)

            xi1 = max(l, ox1)
            yi1 = max(t1, oy1)
            xi2 = min(r, ox2)
            yi2 = min(b, oy2)

            inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            area1 = (r - l) * (b - t1)
            area2 = (ox2 - ox1) * (oy2 - oy1)
            union = area1 + area2 - inter
            iou = inter / union if union > 0 else 0

            if iou > best_iou:
                best_iou = iou
                best_match_obj = obj

        if best_iou > 0.3 and best_match_obj:
            cam1_bbox = best_match_obj["cam1_bbox"]
            cam2_bbox = best_match_obj.get("cam2_bbox")
        else:
            cam1_bbox = tracker_bbox
            cam2_bbox = None

        tracked_objects.append({
            "id": track_id,
            "cam1_bbox": cam1_bbox,
            "cam2_bbox": cam2_bbox,
            "class": obj.get("class")
        })

        if track_id not in colors:
            colors[track_id] = [random.randint(0, 255) for _ in range(3)]

    tracked_data_output.append({
        "frame": frame_num,
        "objects": tracked_objects
    })

    if frame_num % 50 == 0:
        print(f"  Frame {frame_num}, objects: {len(tracked_objects)}")

cap1.release()

# --- Save tracked JSON ---
with open(tracked_json_path, "w") as f:
    json.dump(tracked_data_output, f, indent=2)

print(f"\n✅ Tracked objects saved to {tracked_json_path}")

# ============================================================
#                 VISUALIZATION
# ============================================================

cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

fps = 10
out = None
frame_num = 0
last_bboxes_cam1 = {}
last_bboxes_cam2 = {}

print("\n🎬 Creating live visualization...")

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2 or frame_num >= len(tracked_data_output):
        break

    frame_info = tracked_data_output[frame_num]

    for obj in frame_info["objects"]:
        obj_id = obj["id"]
        color = colors[obj_id]

        # --- CAM1 ---
        if obj.get("cam1_bbox") and len(obj["cam1_bbox"]) == 4:
            try:
                x1, y1, x2, y2 = map(int, obj["cam1_bbox"])
                last_bboxes_cam1[obj_id] = (x1, y1, x2, y2)
                cv2.rectangle(frame1, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame1, str(obj_id), (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            except Exception as e:
                print(f"⚠️ Skipping CAM1 bbox for object {obj_id}: {obj['cam1_bbox']} - {e}")

        # --- CAM2 ---
        if obj.get("cam2_bbox") and len(obj["cam2_bbox"]) == 4:
            try:
                x1, y1, x2, y2 = map(int, obj["cam2_bbox"])
                last_bboxes_cam2[obj_id] = (x1, y1, x2, y2)
                cv2.rectangle(frame2, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame2, str(obj_id), (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            except Exception as e:
                print(f"⚠️ Skipping CAM2 bbox for object {obj_id}: {obj['cam2_bbox']} - {e}")

    # Resize both to same height
    h = min(frame1.shape[0], frame2.shape[0])
    frame1 = cv2.resize(frame1, (int(frame1.shape[1] * h / frame1.shape[0]), h))
    frame2 = cv2.resize(frame2, (int(frame2.shape[1] * h / frame2.shape[0]), h))

    combined = cv2.hconcat([frame1, frame2])

    if out is None:
        height, width = combined.shape[:2]
        out = cv2.VideoWriter(output_video_path,
                              cv2.VideoWriter_fourcc(*'XVID'),
                              fps, (width, height))

    out.write(combined)
    cv2.imshow("Multi-View Tracked Objects", combined)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_num += 1

cap1.release()
cap2.release()
if out:
    out.release()
cv2.destroyAllWindows()

print(f"✅ Tracked video saved to {output_video_path}")
