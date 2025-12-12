import argparse
import cv2
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
import random

parser = argparse.ArgumentParser(description="Match detections between two camera views.")
parser.add_argument("--input-json", type=str, default="./multi_view_detections.json", help="Path to detection JSON from step 3")
parser.add_argument("--video1", type=str, default="data/raw/testing_videos/Cam3.mp4", help="Path to camera 1 video")
parser.add_argument("--video2", type=str, default="data/raw/testing_videos/Cam4.mp4", help="Path to camera 2 video")
parser.add_argument("--out-json", type=str, default="./multi_view_detections_matched.json", help="Output JSON with matched objects")
parser.add_argument("--out-video", type=str, default="./matching_visualization.avi", help="Output visualization video")
parser.add_argument("--match-threshold", type=float, default=0.3, help="Similarity threshold for geometric matching")
parser.add_argument("--appearance-threshold", type=float, default=0.7, help="Similarity threshold for appearance-based matching")
parser.add_argument("--phone-mode", action="store_true", help="Disable geometric matching and match via appearance features (recommended for handheld footage)")
parser.add_argument(
    "--independent-tracking",
    action="store_true",
    help="Track each camera independently and skip cross-camera matching (handheld footage quick fix)",
)
args = parser.parse_args()

input_json_path = args.input_json
video1_path = args.video1
video2_path = args.video2
output_json_path = args.out_json
output_video_path = args.out_video
geometric_threshold = max(0.0, min(1.0, args.match_threshold))
appearance_threshold = max(0.0, min(1.0, args.appearance_threshold))
phone_mode = args.phone_mode
independent_tracking = args.independent_tracking

# --- Load detection data ---
print("📂 Loading detection data from JSON...")
with open(input_json_path, "r") as f:
    detection_data = json.load(f)

metadata = detection_data["metadata"]
frames_data = detection_data["frames"]

w1, h1 = metadata["cam1"]["width"], metadata["cam1"]["height"]
w2, h2 = metadata["cam2"]["width"], metadata["cam2"]["height"]
fps = metadata["fps"]

def _features_exist(frames):
    for frame in frames:
        for det in frame.get("cam1", []):
            if det.get("features") is not None:
                return True
        for det in frame.get("cam2", []):
            if det.get("features") is not None:
                return True
    return False

appearance_features_available = _features_exist(frames_data)
if phone_mode and not appearance_features_available:
    print("⚠️ Phone mode requested but no appearance features were found in the detection JSON. Matches may be empty.")

print(f"✅ Loaded {len(frames_data)} frames of detections")
print(f"   Cam1: {w1}x{h1}, Cam2: {w2}x{h2}, FPS: {fps}")

# --- Helper functions ---
def clamp_bbox(bbox, max_width, max_height):
    x1, y1, x2, y2 = bbox
    return [max(0, min(x1, max_width)), max(0, min(y1, max_height)),
            max(0, min(x2, max_width)), max(0, min(y2, max_height))]

def bbox_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    union_area = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter_area
    return inter_area / union_area if union_area > 0 else 0

def normalize_bbox(bbox, width, height):
    x1, y1, x2, y2 = bbox
    return [x1/width, y1/height, x2/width, y2/height]

def compute_similarity(bbox1, bbox2, w1, h1, w2, h2):
    norm_bbox1 = normalize_bbox(bbox1, w1, h1)
    norm_bbox2 = normalize_bbox(bbox2, w2, h2)
    iou = bbox_iou(norm_bbox1, norm_bbox2)
    center1 = [(norm_bbox1[0]+norm_bbox1[2])/2, (norm_bbox1[1]+norm_bbox1[3])/2]
    center2 = [(norm_bbox2[0]+norm_bbox2[2])/2, (norm_bbox2[1]+norm_bbox2[3])/2]
    center_dist = np.sqrt((center1[0]-center2[0])**2 + (center1[1]-center2[1])**2)
    size1 = (norm_bbox1[2]-norm_bbox1[0])*(norm_bbox1[3]-norm_bbox1[1])
    size2 = (norm_bbox2[2]-norm_bbox2[0])*(norm_bbox2[3]-norm_bbox2[1])
    size_sim = min(size1, size2)/max(size1, size2) if max(size1, size2)>0 else 0
    return 0.4*iou + 0.4*(1-center_dist) + 0.2*size_sim

def match_detections(cam1_dets, cam2_dets, w1, h1, w2, h2, threshold=0.3):
    if not cam1_dets or not cam2_dets:
        return []
    cost_matrix = np.zeros((len(cam1_dets), len(cam2_dets)))
    for i, det1 in enumerate(cam1_dets):
        for j, det2 in enumerate(cam2_dets):
            if det1['class_id'] != det2['class_id']:
                cost_matrix[i, j] = 0
            else:
                cost_matrix[i, j] = compute_similarity(det1['bbox'], det2['bbox'], w1, h1, w2, h2)
    cost_matrix = 1 - cost_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = [(i, j, 1-cost_matrix[i,j]) for i,j in zip(row_ind, col_ind) if 1-cost_matrix[i,j]>=threshold]
    return matches


def compute_feature_similarity(features1, features2):
    if features1 is None or features2 is None:
        return 0.0
    feat1 = np.asarray(features1, dtype=np.float32)
    feat2 = np.asarray(features2, dtype=np.float32)
    if feat1.size == 0 or feat2.size == 0:
        return 0.0
    norm1 = np.linalg.norm(feat1)
    norm2 = np.linalg.norm(feat2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return float(np.dot(feat1, feat2) / (norm1 * norm2))


def match_by_appearance(detections_cam1, detections_cam2, threshold=0.7):
    """
    Match objects using visual appearance instead of geometry.
    Uses the same ReID-style features exported in detection JSON.
    """
    if not detections_cam1 or not detections_cam2:
        return []

    matches = []
    matched_cam2 = set()

    for idx1, det1 in enumerate(detections_cam1):
        feat1 = det1.get("features")
        if feat1 is None:
            continue
        best_match = None
        best_score = threshold
        for idx2, det2 in enumerate(detections_cam2):
            if idx2 in matched_cam2:
                continue
            if det1.get("class_id") != det2.get("class_id"):
                continue
            score = compute_feature_similarity(feat1, det2.get("features"))
            if score > best_score:
                best_score = score
                best_match = idx2
        if best_match is not None:
            matches.append((idx1, best_match, best_score))
            matched_cam2.add(best_match)
    return matches

def draw_matches(frame1, frame2, matched_objects, colors):
    vis_frame1 = frame1.copy()
    vis_frame2 = frame2.copy()
    
    for obj in matched_objects:
        obj_id = obj["id"]
        color = colors.get(obj_id, (255, 255, 255))
        class_name = obj["class_name"]
        
        if obj["cam1_bbox"]:
            x1, y1, x2, y2 = map(int, obj["cam1_bbox"])
            cv2.rectangle(vis_frame1, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{obj_id} {class_name}"
            cv2.putText(vis_frame1, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        if obj["cam2_bbox"]:
            x1, y1, x2, y2 = map(int, obj["cam2_bbox"])
            cv2.rectangle(vis_frame2, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{obj_id} {class_name}"
            cv2.putText(vis_frame2, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return vis_frame1, vis_frame2

# --- Open videos for visualization ---
cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

if not cap1.isOpened() or not cap2.isOpened():
    print("❌ Error: Cannot open video(s).")
    exit()

if independent_tracking:
    print("🆔 Independent tracking enabled: skipping cross-camera matching.")
elif phone_mode:
    print(f"📱 Phone mode enabled: using appearance-based matching (threshold={appearance_threshold}).")
else:
    print(f"📐 Geometric matching enabled (threshold={geometric_threshold}).")

matched_data = []
next_object_id = 1
cam1_track_id = 1
cam2_track_id = 10001

random.seed(42)
colors = {}

out = None

print("🎬 Processing and matching detections from JSON...")

for frame_idx, frame_info in enumerate(frames_data):
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        break

    frame_num = frame_info["frame"]
    cam1_dets = frame_info["cam1"]
    cam2_dets = frame_info["cam2"]

    # Clamp bboxes to frame dimensions
    for det in cam1_dets:
        det["bbox"] = clamp_bbox(det["bbox"], w1, h1)
    for det in cam2_dets:
        det["bbox"] = clamp_bbox(det["bbox"], w2, h2)

    if independent_tracking:
        matches = []
    elif phone_mode:
        matches = match_by_appearance(cam1_dets, cam2_dets, threshold=appearance_threshold)
    else:
        matches = match_detections(cam1_dets, cam2_dets, w1, h1, w2, h2, threshold=geometric_threshold)

    matched_objects = []
    matched_cam1 = set()
    matched_cam2 = set()

    if independent_tracking:
        for det in cam1_dets:
            object_id = cam1_track_id
            matched_objects.append({
                "id": object_id,
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "cam1_bbox": det["bbox"],
                "cam2_bbox": None,
                "confidence": det["confidence"],
                "match_score": 0.0,
            })
            if object_id not in colors:
                colors[object_id] = (random.randint(0, 255),
                                     random.randint(0, 255),
                                     random.randint(0, 255))
            cam1_track_id += 1

        for det in cam2_dets:
            object_id = cam2_track_id
            matched_objects.append({
                "id": object_id,
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "cam1_bbox": None,
                "cam2_bbox": det["bbox"],
                "confidence": det["confidence"],
                "match_score": 0.0,
            })
            if object_id not in colors:
                colors[object_id] = (random.randint(0, 255),
                                     random.randint(0, 255),
                                     random.randint(0, 255))
            cam2_track_id += 1
    else:
        for i, j, sim in matches:
            matched_objects.append({
                "id": next_object_id,
                "class_id": cam1_dets[i]["class_id"],
                "class_name": cam1_dets[i]["class_name"],
                "cam1_bbox": cam1_dets[i]["bbox"],
                "cam2_bbox": cam2_dets[j]["bbox"],
                "confidence": (cam1_dets[i]["confidence"] + cam2_dets[j]["confidence"]) / 2,
                "match_score": float(sim)
            })
            matched_cam1.add(i)
            matched_cam2.add(j)
            
            if next_object_id not in colors:
                colors[next_object_id] = (random.randint(0, 255), 
                                         random.randint(0, 255), 
                                         random.randint(0, 255))
            
            next_object_id += 1

        for i, det in enumerate(cam1_dets):
            if i not in matched_cam1:
                matched_objects.append({
                    "id": next_object_id,
                    "class_id": det["class_id"],
                    "class_name": det["class_name"],
                    "cam1_bbox": det["bbox"],
                    "cam2_bbox": None,
                    "confidence": det["confidence"],
                    "match_score": 0.0
                })
                
                if next_object_id not in colors:
                    colors[next_object_id] = (128, 128, 128)
                
                next_object_id += 1

        for j, det in enumerate(cam2_dets):
            if j not in matched_cam2:
                matched_objects.append({
                    "id": next_object_id,
                    "class_id": det["class_id"],
                    "class_name": det["class_name"],
                    "cam1_bbox": None,
                    "cam2_bbox": det["bbox"],
                    "confidence": det["confidence"],
                    "match_score": 0.0
                })
                
                if next_object_id not in colors:
                    colors[next_object_id] = (128, 128, 128)
                
                next_object_id += 1

    matched_data.append({"frame": frame_num, "objects": matched_objects})
    
    vis_frame1, vis_frame2 = draw_matches(frame1, frame2, matched_objects, colors)
    
    h = min(vis_frame1.shape[0], vis_frame2.shape[0])
    vis_frame1_resized = cv2.resize(vis_frame1, (int(vis_frame1.shape[1] * h / vis_frame1.shape[0]), h))
    vis_frame2_resized = cv2.resize(vis_frame2, (int(vis_frame2.shape[1] * h / vis_frame2.shape[0]), h))
    combined = cv2.hconcat([vis_frame1_resized, vis_frame2_resized])
    
    num_matched = sum(1 for obj in matched_objects if obj["cam1_bbox"] and obj["cam2_bbox"])
    info = f"Frame: {frame_num} | Matched: {num_matched} | Cam1 only: {len(cam1_dets)-len(matched_cam1)} | Cam2 only: {len(cam2_dets)-len(matched_cam2)}"
    cv2.putText(combined, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if out is None:
        height, width = combined.shape[:2]
        out = cv2.VideoWriter(
            output_video_path, 
            cv2.VideoWriter_fourcc(*'XVID'), 
            fps / 2,  # 2× slower
            (width, height)
        )
    
    out.write(combined)
    
    display_frame = cv2.resize(combined, (1280, 480))
    cv2.imshow("Matching Visualization (Press 'q' to close)", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
    
    if frame_idx % 50 == 0:
        print(f"  Frame {frame_num}: {num_matched} matched, {len(matched_objects)} total objects")

with open(output_json_path, "w") as f:
    json.dump(matched_data, f, indent=2)

cap1.release()
cap2.release()
if out:
    out.release()
cv2.destroyAllWindows()

print(f"\n✅ Matched detections saved to {output_json_path}")
print(f"✅ Visualization saved to {output_video_path}")
print(f"   Total frames: {len(matched_data)}")

total_matched = sum(1 for frame in matched_data 
                   for obj in frame["objects"] 
                   if obj["cam1_bbox"] and obj["cam2_bbox"])
total_objects = sum(len(frame["objects"]) for frame in matched_data)
print(f"\n📊 Matching statistics:")
print(f"   Total objects: {total_objects}")
print(f"   Matched across cameras: {total_matched}")
print(f"   Match rate: {total_matched/total_objects*100:.1f}%")
