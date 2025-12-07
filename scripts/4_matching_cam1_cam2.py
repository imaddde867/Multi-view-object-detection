import cv2
from ultralytics import YOLO
import json
import numpy as np
from scipy.optimize import linear_sum_assignment

model_path = "results/Detection/03_10epochs_416img/detect/03_10epochs_416img/weights/best.pt"
video1_path = "data/raw/testing_videos/Cam3.mp4"
video2_path = "data/raw/testing_videos/Cam4.mp4"
output_json_path = "results/multi_view_detections_matched.json"

model = YOLO(model_path)
cap1 = cv2.VideoCapture(video1_path)
cap2 = cv2.VideoCapture(video2_path)

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
            if det1['class'] != det2['class']:
                cost_matrix[i, j] = 0
            else:
                cost_matrix[i, j] = compute_similarity(det1['bbox'], det2['bbox'], w1, h1, w2, h2)
    cost_matrix = 1 - cost_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches = [(i, j, 1-cost_matrix[i,j]) for i,j in zip(row_ind, col_ind) if 1-cost_matrix[i,j]>=threshold]
    return matches

# --- Frame processing & matching ---
#offset_frames = 28
#for _ in range(offset_frames):
#    ret, _ = cap2.read()
#    if not ret:
#        break

matched_data = []
frame_num = 0
next_object_id = 1

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if not ret1 or not ret2:
        break

    results1 = model(frame1)[0]
    results2 = model(frame2)[0]

    cam1_dets = [{"bbox": clamp_bbox(bbox, frame1.shape[1], frame1.shape[0]),
                  "conf": float(conf), "class": int(cls)}
                 for bbox, conf, cls in zip(results1.boxes.xyxy.tolist(),
                                             results1.boxes.conf.tolist(),
                                             results1.boxes.cls.tolist())
                 if bbox[2] > bbox[0] and bbox[3] > bbox[1]]

    cam2_dets = [{"bbox": clamp_bbox(bbox, frame2.shape[1], frame2.shape[0]),
                  "conf": float(conf), "class": int(cls)}
                 for bbox, conf, cls in zip(results2.boxes.xyxy.tolist(),
                                             results2.boxes.conf.tolist(),
                                             results2.boxes.cls.tolist())
                 if bbox[2] > bbox[0] and bbox[3] > bbox[1]]

    matches = match_detections(cam1_dets, cam2_dets, frame1.shape[1], frame1.shape[0], frame2.shape[1], frame2.shape[0])

    matched_objects = []
    matched_cam1 = set()
    matched_cam2 = set()

    for i,j,sim in matches:
        matched_objects.append({
            "id": next_object_id,
            "class": cam1_dets[i]["class"],
            "cam1_bbox": cam1_dets[i]["bbox"],
            "cam2_bbox": cam2_dets[j]["bbox"],
            "confidence": (cam1_dets[i]["conf"]+cam2_dets[j]["conf"])/2,
            "match_score": sim
        })
        matched_cam1.add(i)
        matched_cam2.add(j)
        next_object_id += 1

    for i, det in enumerate(cam1_dets):
        if i not in matched_cam1:
            matched_objects.append({
                "id": next_object_id,
                "class": det["class"],
                "cam1_bbox": det["bbox"],
                "cam2_bbox": None,
                "confidence": det["conf"],
                "match_score": 0.0
            })
            next_object_id += 1

    for j, det in enumerate(cam2_dets):
        if j not in matched_cam2:
            matched_objects.append({
                "id": next_object_id,
                "class": det["class"],
                "cam1_bbox": None,
                "cam2_bbox": det["bbox"],
                "confidence": det["conf"],
                "match_score": 0.0
            })
            next_object_id += 1

    matched_data.append({"frame": frame_num, "objects": matched_objects})
    frame_num += 1

with open(output_json_path, "w") as f:
    json.dump(matched_data, f, indent=2)

cap1.release()
cap2.release()
print(f"✅ Matched detections saved to {output_json_path}")
