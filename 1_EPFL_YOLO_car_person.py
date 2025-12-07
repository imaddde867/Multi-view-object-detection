import os
from glob import glob
from collections import defaultdict
import cv2

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths (using absolute paths based on script location)
image_root = os.path.join(script_dir, "multiclass_ground_truth_images")
label_root = os.path.join(script_dir, "multiclass_ground_truth/bounding_boxes_EPFL_cross")
yolo_label_root = os.path.join(script_dir, "yolo_labels_person_car")

print(f"Script directory: {script_dir}")
print(f"Label root: {label_root}")
print(f"Image root: {image_root}")
print(f"Output root: {yolo_label_root}\n")

os.makedirs(yolo_label_root, exist_ok=True)

# Map classes to IDs (only person and car)
folder_to_class = {
    "gt_files242_person": 0,
    "gt_files242_car": 1
}

def convert_to_yolo(x_center, y_center, w, h, img_w, img_h):
    """
    Convert from center format (x_center, y_center, width, height) in pixels
    to YOLO format (normalized x_center, y_center, width, height)
    """
    # Normalize to [0, 1] range
    x_center_norm = x_center / img_w
    y_center_norm = y_center / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    
    # Clip to valid range [0, 1]
    x_center_norm = max(0, min(1, x_center_norm))
    y_center_norm = max(0, min(1, y_center_norm))
    w_norm = max(0, min(1, w_norm))
    h_norm = max(0, min(1, h_norm))
    
    return x_center_norm, y_center_norm, w_norm, h_norm

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
total_boxes = 0

print("Reading annotations for person and car classes...")

for folder_name, class_id in folder_to_class.items():
    label_path = os.path.join(label_root, folder_name, "visible_frame")
    
    print(f"\n  Checking: {label_path}")
    if not os.path.exists(label_path):
        print(f"  Path does not exist!")
        continue
    
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
                
                # Assume format is: x_center, y_center, width, height (in pixels)
                x_center, y_center, w_box, h_box = map(float, parts[:4])
                
                total_boxes += 1
                
                # Convert to YOLO format (normalized)
                x_c, y_c, w_n, h_n = convert_to_yolo(x_center, y_center, w_box, h_box, w, h)
                
                # Skip boxes with zero area
                if w_n <= 0 or h_n <= 0:
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
print(f"   Total bounding boxes: {total_boxes}")
print(f"   Classes: person (0), car (1)")
print(f"   Labels saved to: {yolo_label_root}/")