import os
from pathlib import Path
import cv2
from glob import glob

# Define colors for each class (BGR)
class_colors = {
    "bus": (0, 0, 255),
    "car": (255, 0, 0),
    "person": (0, 255, 0)
}

# Mapping from folder names to class labels
folder_to_class = {
    "gt_files242_bus": "bus",
    "gt_files242_car": "car",
    "gt_files242_person": "person"
}

REPO_ROOT = Path(__file__).resolve().parents[1]

image_root = REPO_ROOT / "multiclass_ground_truth_images"
label_root = REPO_ROOT / "bounding_boxes_EPFL_cross"

# Choose which frame to show
frame_to_show = "221"  # Example: frame 221

def get_multiview_for_frame(frame_id):
    """Collect all available images and label files for this frame across all object types."""
    views = {}

    for folder_name, class_name in folder_to_class.items():
        label_path = label_root / folder_name / "visible_frame"
        label_files = glob(str(label_path / f"det_frame{int(frame_id)}_cam*.txt"))

        for lf in label_files:
            cam_num = os.path.basename(lf).split("_")[2].replace(".txt", "").replace("cam", "")
            img_name = f"{int(frame_id):08d}.jpg"
            img_path = image_root / f"c{cam_num}" / img_name

            if not img_path.exists():
                continue

            # Group by camera number
            if cam_num not in views:
                views[cam_num] = {"img": img_path, "labels": []}
            views[cam_num]["labels"].append((lf, class_name))

    return views


def draw_bboxes(img, label_path, class_name):
    """Draw bounding boxes for one label file."""
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                x, y, w, h = map(float, parts)
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                color = class_colors.get(class_name, (255, 255, 255))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, class_name, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


# --- Visualization ---
views = get_multiview_for_frame(frame_to_show)

if not views:
    print(f"No images found for frame {frame_to_show}")
else:
    imgs = []
    for cam_num in sorted(views.keys()):
        img_path = views[cam_num]["img"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Draw all bounding boxes for this camera
        for label_path, class_name in views[cam_num]["labels"]:
            draw_bboxes(img, label_path, class_name)

        # Add camera ID text
        cv2.putText(img, f"Cam {cam_num}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        imgs.append(img)

    # Resize and concatenate all views horizontally
    h_min = min(im.shape[0] for im in imgs)
    resized = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min)) for im in imgs]
    concat_img = cv2.hconcat(resized)

    cv2.imshow(f"Frame {frame_to_show} - Multi-view (All Classes)", concat_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
