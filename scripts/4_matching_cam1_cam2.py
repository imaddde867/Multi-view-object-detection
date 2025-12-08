import cv2
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
import random

input_json_path = "./multi_view_detections.json"
video1_path = "/home/treenut/multi_view/Yolov8_Multi-view/testing_videos/Cam3.mp4"
video2_path = "/home/treenut/multi_view/Yolov8_Multi-view/testing_videos/Cam4.mp4"
output_json_path = "./multi_view_detections_matched.json"
output_video_path = "./matching_visualization.avi"

# --- Load detection data ---
print("📂 Loading detection data from JSON...")
with open(input_json_path, "r") as f:
    detection_data = json.load(f)

metadata = detection_data["metadata"]
frames_data = detection_data["frames"]

w1, h1 = metadata["cam1"]["width"], metadata["cam1"]["height"]
w2, h2 = metadata["cam2"]["width"], metadata["cam2"]["height"]
fps = metadata["fps"]

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

matched_data = []
next_object_id = 1

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

    matches = match_detections(cam1_dets, cam2_dets, w1, h1, w2, h2)

    matched_objects = []
    matched_cam1 = set()
    matched_cam2 = set()

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