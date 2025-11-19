import json
import cv2
import numpy as np
import os
import glob
from triangulation import Triangulator
from feature_detection import FeatureDetector

def load_calibration(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    projections = {}
    for cam_id, cam_data in data['cameras'].items():
        P = np.array(cam_data['P'], dtype=np.float32)
        # Map "camera0" to index 0, "camera1" to 1, etc...
        idx = int(cam_id.replace('camera', ''))
        projections[idx] = P
    
    # Return a list of matrices sorted by camera index
    return [projections[i] for i in sorted(projections.keys())]

def load_images_for_frame(base_path, frame_number, camera_indices):
    images = {}
    # Format frame number to 8 digits (for example 150 -> "00000150")
    frame_str = f"{frame_number:08d}"
    
    print(f"--- Loading Frame {frame_number} ---")
    
    for cam_idx in camera_indices:
        # Construct path: multiclass_ground_truth_images/c0/00000150.jpg
        # our folders are named like 'c0', 'c1', etc.
        img_path = os.path.join(base_path, f"c{cam_idx}", f"{frame_str}.jpg")
        
        if os.path.exists(img_path):
            print(f"Loading: {img_path}")
            images[cam_idx] = cv2.imread(img_path)
        else:
            print(f"Warning: Image not found at {img_path}")
            
    return images

def main():
    # 1. Configuration
    # specific path based on your folder structure
    image_folder = "multiclass_ground_truth_images" 
    calibration_file = "epfl-calibration.json"
    
    # We will use 4 cameras for reconstruction
    camera_indices = [0, 1, 2, 3] 
    
    # Pick a frame where people are visible (Frame 150 is usually good in this dataset)
    frame_num = 150 

    # 2. Load Calibration
    if not os.path.exists(calibration_file):
        print(f"Error: Calibration file '{calibration_file}' not found. Run create_calibration.py first.")
        return

    projections = load_calibration(calibration_file)
    triangulator = Triangulator(projections)
    detector = FeatureDetector()

    # 3. Load Images
    images = load_images_for_frame(image_folder, frame_num, camera_indices)
    
    if len(images) < 2:
        print("Not enough images loaded to perform triangulation.")
        return

    # 4. Run Pipeline
    print("\n--- Step 1: Feature Detection & Matching ---")
    # We match Camera 0 with Camera 1, then Camera 0 with Camera 2, etc.
    # A simple star-topology matching (Reference Camera = 0)
    
    ref_cam = camera_indices[0]
    ref_img = images[ref_cam]
    ref_kp, ref_desc = detector.detect_features(ref_img)
    
    points_3d = []
    
    for other_cam in camera_indices[1:]:
        if other_cam not in images: continue
        
        target_img = images[other_cam]
        target_kp, target_desc = detector.detect_features(target_img)
        
        # Match features
        matches = detector.match_features(ref_desc, target_desc, ref_kp, target_kp)
        print(f"Matched Cam {ref_cam} <-> Cam {other_cam}: {len(matches)} matches found.")
        
        # Get coordinates
        pts_ref = []
        pts_target = []
        for m in matches:
            pts_ref.append(ref_kp[m.queryIdx].pt)
            pts_target.append(target_kp[m.trainIdx].pt)
            
        pts_ref = np.array(pts_ref)
        pts_target = np.array(pts_target)
        
        # Triangulate these specific matches
        if len(matches) > 0:
            new_points_3d = triangulator.triangulate_points(
                ref_cam, other_cam, pts_ref, pts_target
            )
            points_3d.extend(new_points_3d)

    points_3d = np.array(points_3d)
    print(f"\n--- Step 2: Triangulation Results ---")
    print(f"Total raw 3D points generated: {len(points_3d)}")

    # 5. Simple Visualization (Optional textual output)
    if len(points_3d) > 0:
        # Calculate center of mass to see where the points are roughly
        center = np.mean(points_3d, axis=0)
        print(f"Center of reconstruction (X, Y, Z): {center}")
        
        # Save to a simple OBJ file to view in MeshLab or online viewer
        output_obj = "output_cloud.obj"
        with open(output_obj, "w") as f:
            for p in points_3d:
                f.write(f"v {p[0]} {p[1]} {p[2]}\n")
        print(f"Saved 3D point cloud to '{output_obj}'")

if __name__ == "__main__":
    main()