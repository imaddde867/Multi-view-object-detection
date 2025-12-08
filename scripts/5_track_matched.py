import cv2
import json
import random
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- Paths ---
matched_json_path = "./multi_view_detections_matched.json"
tracked_json_path = "./multi_view_tracked_objects.json"
video1_path = "/home/treenut/multi_view/Yolov8_Multi-view/testing_videos/Cam3.mp4"
video2_path = "/home/treenut/multi_view/Yolov8_Multi-view/testing_videos/Cam4.mp4"
output_video_path = "./multi_view_5th_tracked_visualization.avi"

# Class mapping
CLASS_NAMES = {
    0: "person",
    1: "car"
}

# --- Load matched detections ---
print("📂 Loading matched detections...")
with open(matched_json_path, "r") as f:
    matched_data = json.load(f)

# --- Get video dimensions ---
cap_temp = cv2.VideoCapture(video1_path)
cam1_width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
cam1_height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap_temp.get(cv2.CAP_PROP_FPS))
cap_temp.release()

cap_temp = cv2.VideoCapture(video2_path)
cam2_width = int(cap_temp.get(cv2.CAP_PROP_FRAME_WIDTH))
cam2_height = int(cap_temp.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap_temp.release()

print(f"📐 Cam1: {cam1_width}x{cam1_height}, Cam2: {cam2_width}x{cam2_height}, FPS: {fps}")

# --- Initialize TWO DeepSORT trackers (one per camera) ---
tracker_cam1 = DeepSort(
    max_age=30,
    n_init=3,
    max_iou_distance=0.7,
    max_cosine_distance=0.4,
    nn_budget=100
)

tracker_cam2 = DeepSort(
    max_age=30,
    n_init=3,
    max_iou_distance=0.7,
    max_cosine_distance=0.4,
    nn_budget=100
)

# --- Global ID management ---
class GlobalIDManager:
    def __init__(self):
        self.cam1_to_global = {}  # cam1_track_id -> global_id
        self.cam2_to_global = {}  # cam2_track_id -> global_id
        self.next_global_id = 1
        self.global_id_history = {}  # global_id -> {"cam1_ids": set(), "cam2_ids": set()}
    
    def get_or_create_global_id(self, cam1_id, cam2_id):
        """
        Match cam1 and cam2 track IDs to a single global ID
        """
        global_id = None
        
        # Check if either camera ID already has a global ID
        if cam1_id and cam1_id in self.cam1_to_global:
            global_id = self.cam1_to_global[cam1_id]
        elif cam2_id and cam2_id in self.cam2_to_global:
            global_id = self.cam2_to_global[cam2_id]
        
        # Create new global ID if none exists
        if global_id is None:
            global_id = self.next_global_id
            self.next_global_id += 1
            self.global_id_history[global_id] = {"cam1_ids": set(), "cam2_ids": set()}
        
        # Associate camera IDs with global ID
        if cam1_id:
            self.cam1_to_global[cam1_id] = global_id
            self.global_id_history[global_id]["cam1_ids"].add(cam1_id)
        if cam2_id:
            self.cam2_to_global[cam2_id] = global_id
            self.global_id_history[global_id]["cam2_ids"].add(cam2_id)
        
        return global_id
    
    def get_global_id_for_cam1(self, cam1_id):
        return self.cam1_to_global.get(cam1_id)
    
    def get_global_id_for_cam2(self, cam2_id):
        return self.cam2_to_global.get(cam2_id)

id_manager = GlobalIDManager()

# --- Colors ---
random.seed(42)
colors = {}

tracked_data_output = []

print("\n🔍 Running dual-camera DeepSORT tracking...")

# Open both video captures for ReID
cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

# --- Helper function for IoU ---
def compute_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)
    
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0

# --- Process frames ---
for frame_num, frame_info in enumerate(matched_data):
    ret1, frame_cam1 = cap1.read()
    ret2, frame_cam2 = cap2.read()
    
    if not ret1 or not ret2:
        print("⚠️ End of video.")
        break

    objects = frame_info["objects"]
    
    # Separate detections for each camera
    cam1_detections = []
    cam2_detections = []
    cam1_object_map = []
    cam2_object_map = []
    
    for obj in objects:
        class_id = obj.get("class_id", 0)
        conf = obj.get("confidence", 0.9)
        
        if conf < 0.5 or class_id not in [0, 1]:
            continue
        
        # Cam1 detections
        bbox1 = obj.get("cam1_bbox")
        if bbox1 and len(bbox1) == 4:
            x1, y1, x2, y2 = map(int, bbox1)
            if x2 > x1 and y2 > y1:
                cam1_detections.append(([x1, y1, x2 - x1, y2 - y1], conf, class_id))
                cam1_object_map.append(obj)
        
        # Cam2 detections
        bbox2 = obj.get("cam2_bbox")
        if bbox2 and len(bbox2) == 4:
            x1, y1, x2, y2 = map(int, bbox2)
            if x2 > x1 and y2 > y1:
                cam2_detections.append(([x1, y1, x2 - x1, y2 - y1], conf, class_id))
                cam2_object_map.append(obj)
    
    # Update both trackers
    tracks_cam1 = tracker_cam1.update_tracks(cam1_detections, frame=frame_cam1)
    tracks_cam2 = tracker_cam2.update_tracks(cam2_detections, frame=frame_cam2)
    
    # Build track dictionaries
    cam1_tracks_dict = {}
    for t in tracks_cam1:
        if t.is_confirmed():
            l, t1, r, b = t.to_ltrb()
            cam1_tracks_dict[t.track_id] = {
                "bbox": [int(l), int(t1), int(r), int(b)],
                "class_id": t.get_det_class() if hasattr(t, 'get_det_class') else 0
            }
    
    cam2_tracks_dict = {}
    for t in tracks_cam2:
        if t.is_confirmed():
            l, t1, r, b = t.to_ltrb()
            cam2_tracks_dict[t.track_id] = {
                "bbox": [int(l), int(t1), int(r), int(b)],
                "class_id": t.get_det_class() if hasattr(t, 'get_det_class') else 0
            }
    
    # Match tracks to original matched objects
    tracked_objects = []
    processed_cam1_ids = set()
    processed_cam2_ids = set()
    
    # Process matched objects (objects with both cam1 and cam2 bboxes)
    for obj in objects:
        if not obj.get("cam1_bbox") or not obj.get("cam2_bbox"):
            continue
        
        cam1_bbox = obj["cam1_bbox"]
        cam2_bbox = obj["cam2_bbox"]
        
        # Find best matching track in cam1
        best_cam1_id = None
        best_cam1_iou = 0
        for track_id, track_data in cam1_tracks_dict.items():
            if track_id in processed_cam1_ids:
                continue
            iou = compute_iou(cam1_bbox, track_data["bbox"])
            if iou > best_cam1_iou:
                best_cam1_iou = iou
                best_cam1_id = track_id
        
        # Find best matching track in cam2
        best_cam2_id = None
        best_cam2_iou = 0
        for track_id, track_data in cam2_tracks_dict.items():
            if track_id in processed_cam2_ids:
                continue
            iou = compute_iou(cam2_bbox, track_data["bbox"])
            if iou > best_cam2_iou:
                best_cam2_iou = iou
                best_cam2_id = track_id
        
        # Only proceed if we have at least one good match
        if (best_cam1_iou > 0.3 or best_cam2_iou > 0.3):
            # Get or create global ID
            global_id = id_manager.get_or_create_global_id(
                best_cam1_id if best_cam1_iou > 0.3 else None,
                best_cam2_id if best_cam2_iou > 0.3 else None
            )
            
            tracked_objects.append({
                "id": global_id,
                "cam1_bbox": cam1_bbox,
                "cam2_bbox": cam2_bbox,
                "class_id": obj.get("class_id", 0),
                "class_name": obj.get("class_name", "unknown"),
                "cam1_track_id": best_cam1_id,
                "cam2_track_id": best_cam2_id
            })
            
            if best_cam1_id and best_cam1_iou > 0.3:
                processed_cam1_ids.add(best_cam1_id)
            if best_cam2_id and best_cam2_iou > 0.3:
                processed_cam2_ids.add(best_cam2_id)
            
            if global_id not in colors:
                colors[global_id] = [random.randint(0, 255) for _ in range(3)]
    
    # Process cam1-only tracks
    for track_id, track_data in cam1_tracks_dict.items():
        if track_id in processed_cam1_ids:
            continue
        
        global_id = id_manager.get_global_id_for_cam1(track_id)
        if global_id is None:
            global_id = id_manager.get_or_create_global_id(track_id, None)
        
        tracked_objects.append({
            "id": global_id,
            "cam1_bbox": track_data["bbox"],
            "cam2_bbox": None,
            "class_id": track_data["class_id"],
            "class_name": CLASS_NAMES.get(track_data["class_id"], "unknown"),
            "cam1_track_id": track_id,
            "cam2_track_id": None
        })
        
        if global_id not in colors:
            colors[global_id] = [128, 128, 128]
    
    # Process cam2-only tracks
    for track_id, track_data in cam2_tracks_dict.items():
        if track_id in processed_cam2_ids:
            continue
        
        global_id = id_manager.get_global_id_for_cam2(track_id)
        if global_id is None:
            global_id = id_manager.get_or_create_global_id(None, track_id)
        
        tracked_objects.append({
            "id": global_id,
            "cam1_bbox": None,
            "cam2_bbox": track_data["bbox"],
            "class_id": track_data["class_id"],
            "class_name": CLASS_NAMES.get(track_data["class_id"], "unknown"),
            "cam1_track_id": None,
            "cam2_track_id": track_id
        })
        
        if global_id not in colors:
            colors[global_id] = [128, 128, 128]
    
    tracked_data_output.append({
        "frame": frame_num,
        "objects": tracked_objects
    })
    
    if frame_num % 50 == 0:
        print(f"  Frame {frame_num}: {len(tracked_objects)} tracked objects")

cap1.release()
cap2.release()

# --- Save tracked JSON ---
with open(tracked_json_path, "w") as f:
    json.dump(tracked_data_output, f, indent=2)

print(f"\n✅ Tracked objects saved to {tracked_json_path}")

# ============================================================
#                 VISUALIZATION
# ============================================================

cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

output_fps = max(1, fps // 2)  # 2x slower

out = None
frame_num = 0

print("\n🎬 Creating visualization...")

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2 or frame_num >= len(tracked_data_output):
        break

    frame_info = tracked_data_output[frame_num]

    for obj in frame_info["objects"]:
        obj_id = obj["id"]
        color = tuple(colors[obj_id])
        class_name = obj.get("class_name", "unknown")

        # --- CAM1 ---
        if obj.get("cam1_bbox") and len(obj["cam1_bbox"]) == 4:
            try:
                x1, y1, x2, y2 = map(int, obj["cam1_bbox"])
                cv2.rectangle(frame1, (x1, y1), (x2, y2), color, 2)

                label = f"ID:{obj_id} {class_name}"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame1, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
                cv2.putText(frame1, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            except:
                pass

        # --- CAM2 ---
        if obj.get("cam2_bbox") and len(obj["cam2_bbox"]) == 4:
            try:
                x1, y1, x2, y2 = map(int, obj["cam2_bbox"])
                cv2.rectangle(frame2, (x1, y1), (x2, y2), color, 2)

                label = f"ID:{obj_id} {class_name}"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame2, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
                cv2.putText(frame2, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            except:
                pass

    # Combine frames
    h = min(frame1.shape[0], frame2.shape[0])
    frame1_resized = cv2.resize(frame1, (int(frame1.shape[1] * h / frame1.shape[0]), h))
    frame2_resized = cv2.resize(frame2, (int(frame2.shape[1] * h / frame2.shape[0]), h))
    combined = cv2.hconcat([frame1_resized, frame2_resized])

    # Count objects in each view
    cam1_count = sum(1 for obj in frame_info["objects"] if obj.get("cam1_bbox"))
    cam2_count = sum(1 for obj in frame_info["objects"] if obj.get("cam2_bbox"))
    both_count = sum(1 for obj in frame_info["objects"] if obj.get("cam1_bbox") and obj.get("cam2_bbox"))
    
    info_text = f"Frame: {frame_num} | Total: {len(frame_info['objects'])} | Both views: {both_count} | Cam1: {cam1_count} | Cam2: {cam2_count}"
    cv2.putText(combined, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if out is None:
        height, width = combined.shape[:2]
        out = cv2.VideoWriter(output_video_path,
                              cv2.VideoWriter_fourcc(*'XVID'),
                              output_fps, (width, height))

    out.write(combined)

    display_frame = cv2.resize(combined, (1280, 480))
    cv2.imshow("Multi-Camera Tracking (Press 'q' to close)", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

    frame_num += 1

cap1.release()
cap2.release()
if out:
    out.release()
cv2.destroyAllWindows()

print(f"\n✅ Video saved to {output_video_path}")

# Statistics
unique_ids = set(obj["id"] for frame in tracked_data_output for obj in frame["objects"])
total_both_views = sum(1 for frame in tracked_data_output 
                      for obj in frame["objects"] 
                      if obj.get("cam1_bbox") and obj.get("cam2_bbox"))
total_objects = sum(len(frame["objects"]) for frame in tracked_data_output)

print(f"\n📊 Tracking statistics:")
print(f"   Unique tracked IDs: {len(unique_ids)}")
print(f"   Frames processed: {len(tracked_data_output)}")
print(f"   Total object instances: {total_objects}")
print(f"   Objects visible in both cameras: {total_both_views} ({total_both_views/total_objects*100:.1f}%)")
print(f"   Average objects per frame: {total_objects / len(tracked_data_output):.1f}")