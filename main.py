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
        idx = int(cam_id.replace('camera', ''))
        projections[idx] = P
    
    return [projections[i] for i in sorted(projections.keys())]

def load_images_for_frame(base_path, frame_number, camera_indices):
    images = {}
    frame_str = f"{frame_number:08d}"
    
    print(f"\n--- Loading Frame {frame_number} ---")
    
    for cam_idx in camera_indices:
        img_path = os.path.join(base_path, f"c{cam_idx}", f"{frame_str}.jpg")
        
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images[cam_idx] = img
                print(f"✓ Loaded camera {cam_idx}: {img.shape}")
            else:
                print(f"✗ Failed to read: {img_path}")
        else:
            print(f"✗ Not found: {img_path}")
            
    return images

def main():
    image_folder = "multiclass_ground_truth_images" 
    calibration_file = "epfl-calibration.json"
    
    camera_indices = [0, 1, 2, 3, 4, 5]
    frame_num = 150
    reprojection_threshold = 1.5

    if not os.path.exists(calibration_file):
        print(f"Error: Calibration file '{calibration_file}' not found. Run create_calibration.py first.")
        return

    print("=" * 60)
    print("EPFL Multi-View 3D Reconstruction Pipeline")
    print("=" * 60)

    projections = load_calibration(calibration_file)
    triangulator = Triangulator(projections)
    detector = FeatureDetector()

    images = load_images_for_frame(image_folder, frame_num, camera_indices)
    
    if len(images) < 2:
        print("✗ Not enough images loaded to perform triangulation.")
        return

    print(f"\n✓ Loaded {len(images)} images")

    print("\n--- Step 1: Feature Detection ---")
    features = {}
    for cam_idx in sorted(images.keys()):
        kp, des = detector.detect_features(images[cam_idx])
        if des is not None:
            features[cam_idx] = (kp, des)
            print(f"Camera {cam_idx}: {len(kp)} features detected")
        else:
            print(f"Camera {cam_idx}: No features detected")

    print("\n--- Step 2: Feature Matching & Triangulation ---")
    all_points_3d = []
    match_stats = []
    
    for i, cam1 in enumerate(sorted(features.keys())):
        for cam2 in sorted(features.keys())[i+1:]:
            kp1, des1 = features[cam1]
            kp2, des2 = features[cam2]
            
            matches = detector.match_features(des1, des2, kp1, kp2)
            
            if len(matches) < 8:
                print(f"Camera {cam1} <-> {cam2}: {len(matches)} matches (skipped)")
                continue
            
            pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.array([kp2[m.trainIdx].pt for m in matches])
            
            points_3d = triangulator.triangulate_points(cam1, cam2, pts1, pts2)
            
            if len(points_3d) > 0:
                all_points_3d.extend(points_3d)
                match_stats.append((cam1, cam2, len(matches), len(points_3d)))
                print(f"Camera {cam1} <-> {cam2}: {len(matches)} matches → {len(points_3d)} valid 3D points")
            else:
                print(f"Camera {cam1} <-> {cam2}: {len(matches)} matches → 0 valid 3D points")

    all_points_3d = np.array(all_points_3d)
    print(f"\n✓ Total 3D points before filtering: {len(all_points_3d)}")
    
    if len(all_points_3d) > 0:
        print(f"  Min coords: {np.min(all_points_3d, axis=0)}")
        print(f"  Max coords: {np.max(all_points_3d, axis=0)}")
        print(f"  Mean coords: {np.mean(all_points_3d, axis=0)}")

    print("\n--- Step 3: Reprojection Error Filtering ---")
    if len(all_points_3d) > 0:
        filtered_points = []
        depth_failures = 0
        bounds_failures = 0
        
        for pt_3d in all_points_3d:
            valid = True
            depth_ok = True
            bounds_ok = True
            
            for cam_idx in sorted(features.keys()):
                try:
                    pt_2d_proj = triangulator.P_matrices[cam_idx] @ np.append(pt_3d, 1)
                    
                    if pt_2d_proj[2] <= 0:
                        depth_ok = False
                        valid = False
                        break
                    
                    u_proj = pt_2d_proj[0] / pt_2d_proj[2]
                    v_proj = pt_2d_proj[1] / pt_2d_proj[2]
                    
                    h, w = images[cam_idx].shape[:2]
                    if u_proj < -200 or u_proj >= w + 200 or v_proj < -200 or v_proj >= h + 200:
                        bounds_ok = False
                        valid = False
                        break
                        
                except:
                    valid = False
                    break
            
            if not depth_ok:
                depth_failures += 1
            elif not bounds_ok:
                bounds_failures += 1
            elif valid:
                filtered_points.append(pt_3d)
        
        filtered_points = np.array(filtered_points)
        print(f"✓ After depth/bounds check: {len(filtered_points)} points")
        print(f"  (Depth failures: {depth_failures}, Bounds failures: {bounds_failures})")
        
        if len(filtered_points) > 0:
            center = np.mean(filtered_points, axis=0)
            bounds = np.max(filtered_points, axis=0) - np.min(filtered_points, axis=0)
            print(f"\nReconstruction bounds:")
            print(f"  Center (X, Y, Z): {center}")
            print(f"  Size (X, Y, Z): {bounds}")
            print(f"  Points in valid region: {len(filtered_points)}")

            output_obj = "output_cloud.obj"
            with open(output_obj, "w") as f:
                for p in filtered_points:
                    f.write(f"v {p[0]} {p[1]} {p[2]}\n")
            print(f"\n✓ Saved 3D point cloud to '{output_obj}'")

    print("\n" + "=" * 60)
    print("Reconstruction complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()