import os
import shutil
import random
from glob import glob
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
YOLO_DIR = Path(__file__).resolve().parent

# Paths
image_root = REPO_ROOT / "multiclass_ground_truth_images"  # c0, c1, ...
yolo_label_root = YOLO_DIR / "yolo_labels"
output_root = YOLO_DIR / "data_train-test"
train_ratio = 0.8

# Ensure output subfolders exist
for split in ["train", "test"]:
    os.makedirs(output_root / split / "images", exist_ok=True)
    os.makedirs(output_root / split / "labels", exist_ok=True)

# Collect all label files
all_labels = glob(str(yolo_label_root / "*.txt"))
frames = sorted(list({os.path.basename(f).split("_")[0] for f in all_labels}))
random.shuffle(frames)

# Split frames into train/test
num_train = int(len(frames) * train_ratio)
train_frames = set(frames[:num_train])
test_frames = set(frames[num_train:])

print(f"Total frames: {len(frames)}, Train: {len(train_frames)}, Test: {len(test_frames)}")

# Copy labels and corresponding images
for label_file in all_labels:
    base = os.path.basename(label_file)
    frame_num, cam_part = base.split("_")
    cam_num = cam_part.split(".")[0]
    
    split = "train" if frame_num in train_frames else "test"
    
    img_src = image_root / cam_num / f"{frame_num}.jpg"
    img_dst = output_root / split / "images" / f"{frame_num}_{cam_num}.jpg"
    label_dst = output_root / split / "labels" / f"{frame_num}_{cam_num}.txt"
    
    # Skip if image is missing
    if not img_src.exists():
        print(f"⚠️ Image missing: {img_src}")
        continue

    shutil.copy2(img_src, img_dst)
    shutil.copy2(label_file, label_dst)

print("Train/test split completed successfully!")
