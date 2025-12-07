import os
from glob import glob
from collections import defaultdict
import cv2

# Paths
image_root = "multiclass_ground_truth_images"
label_root = "multiclass_ground_truth/bounding_boxes_EPFL_cross"
yolo_label_root = "yolo_labels"

os.makedirs(yolo_label_root, exist_ok=True)

# Map classes to IDs
folder_to_class = {
    "gt_files242_person": 0,
    "gt_files242_car": 1,
    "gt_files242_bus": 2
}

def convert_to_yolo(x, y, w, h, img_w, img_h):
    """Convert absolute bounding box to YOLO format (normalized)"""
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    return x_center, y_center, w_norm, h_norm

def get_image_dimensions(img_path, cache):
    """Get image dimensions with caching to avoid repeated reads"""
    if img_path not in cache:
        if not os.path.exists(img_path):
            return None
        img = cv2.imread(img_path)
        if img is None:
            return None
        cache[img_path] = img.shape[:2]  # (height, width)
    return cache[img_path]

# Aggregate all annotations by (frame, camera)
annotations = defaultdict(list)
img_dim_cache = {}

print("Reading annotations from all classes...")

for folder_name, class_id in folder_to_class.items():
    label_path = os.path.join(label_root, folder_name, "visible_frame")
    label_files = glob(os.path.join(label_path, "*.txt"))
    
    print(f"  -> Processing {folder_name} ({len(label_files)} files)")

    for lf in label_files:
        base = os.path.basename(lf)
        parts = base.split("_")
        if len(parts) != 3:
            print(f"  Unexpected filename format: {base}")
            continue
        
        frame_num = int(parts[1].replace("frame", ""))
        cam_num = parts[2].replace(".txt", "").replace("cam", "")
        
        # Construct image path
        img_name = f"{frame_num:08d}.jpg"
        img_path = os.path.join(image_root, f"c{cam_num}", img_name)
        
        # Get image dimensions
        dims = get_image_dimensions(img_path, img_dim_cache)
        if dims is None:
            print(f"  Image not found or unreadable: {img_path}")
            continue
        
        h, w = dims
        
        # Read bounding boxes
        with open(lf, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                
                x_abs, y_abs, w_abs, h_abs = map(float, parts)
                
                # Convert to YOLO format
                x_c, y_c, w_n, h_n = convert_to_yolo(x_abs, y_abs, w_abs, h_abs, w, h)
                
                # Validate normalized coordinates
                if not (0 <= x_c <= 1 and 0 <= y_c <= 1 and 0 < w_n <= 1 and 0 < h_n <= 1):
                    print(f"  Invalid bbox in {base}: {x_c}, {y_c}, {w_n}, {h_n}")
                    continue
                
                # Store annotation
                key = (frame_num, cam_num)
                annotations[key].append(f"{class_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")

# Write aggregated YOLO labels
print(f"\nWriting {len(annotations)} YOLO label files...")

for (frame_num, cam_num), yolo_lines in annotations.items():
    yolo_file_path = os.path.join(yolo_label_root, f"{frame_num:08d}_c{cam_num}.txt")
    with open(yolo_file_path, "w") as f:
        f.write("\n".join(yolo_lines))

print(f"YOLO label conversion complete!")
print(f"   Total frames processed: {len(annotations)}")
print(f"   Labels saved to: {yolo_label_root}/")