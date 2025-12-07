import os
import shutil
import random
from glob import glob
from collections import defaultdict

# -----------------------------
# Paths
# -----------------------------
image_root = "data/raw/multiclass_ground_truth_images"
yolo_label_root = "data/processed/yolo_labels"
output_root = "data/processed/data_train-test"

# -----------------------------
# Split ratios
# -----------------------------
train_ratio = 0.7
val_ratio   = 0.15
test_ratio  = 0.15

# -----------------------------
# Reproducibility
# -----------------------------
random.seed(42)

# -----------------------------
# Create output folders
# -----------------------------
for split in ["train", "val", "test"]:
    for subfolder in ["images", "labels"]:
        os.makedirs(os.path.join(output_root, split, subfolder), exist_ok=True)

# -----------------------------
# Collect all label files
# -----------------------------
all_labels = glob(os.path.join(yolo_label_root, "*.txt"))
if not all_labels:
    print("No label files found! Check your yolo_labels directory.")
    exit(1)

# -----------------------------
# Group labels by frame number
# -----------------------------
frame_to_labels = defaultdict(list)
for label_file in all_labels:
    base = os.path.basename(label_file)
    if "_" not in base:
        print(f"Unexpected label filename format: {base}")
        continue
    frame_num = base.split("_")[0]
    frame_to_labels[frame_num].append(label_file)

frames = sorted(frame_to_labels.keys())
random.shuffle(frames)

# -----------------------------
# Map classes to frames
# -----------------------------
class_to_frames = defaultdict(set)
for frame_num, label_files in frame_to_labels.items():
    for label_file in label_files:
        with open(label_file, "r") as f:
            for line in f:
                class_id = int(line.split()[0])
                class_to_frames[class_id].add(frame_num)

# -----------------------------
# Initial random split by frames
# -----------------------------
num_train = int(len(frames) * train_ratio)
num_val   = int(len(frames) * val_ratio)

train_frames = set(frames[:num_train])
val_frames   = set(frames[num_train:num_train + num_val])
test_frames  = set(frames[num_train + num_val:])

# -----------------------------
# Ensure all classes exist in each split
# -----------------------------
def ensure_class_in_split(target_frames, split_name):
    added_frames = set()
    for class_id, frames_with_class in class_to_frames.items():
        if not frames_with_class & target_frames:
            chosen = random.choice(list(frames_with_class))
            added_frames.add(chosen)
    return added_frames

# Add missing classes to splits
train_frames |= ensure_class_in_split(train_frames, "train")
val_frames   |= ensure_class_in_split(val_frames, "val")
test_frames  |= ensure_class_in_split(test_frames, "test")

# Remove any duplicated frames across splits
all_frames = set()
for split_frames in [train_frames, val_frames, test_frames]:
    all_frames |= split_frames

if len(all_frames) != len(train_frames | val_frames | test_frames):
    print("Warning: some frames may still be duplicated across splits!")

# -----------------------------
# Copy images and labels
# -----------------------------
stats = {"train": 0, "val": 0, "test": 0}
missing_images = []

for frame_num, label_files in frame_to_labels.items():
    if frame_num in train_frames:
        split = "train"
    elif frame_num in val_frames:
        split = "val"
    else:
        split = "test"

    for label_file in label_files:
        base = os.path.basename(label_file)
        cam_part = "_".join(base.split("_")[1:]).replace(".txt","")  # handles multiple underscores

        img_src = os.path.join(image_root, cam_part, f"{frame_num}.jpg")
        if not os.path.exists(img_src):
            missing_images.append(img_src)
            continue

        img_dst = os.path.join(output_root, split, "images", f"{frame_num}_{cam_part}.jpg")
        label_dst = os.path.join(output_root, split, "labels", f"{frame_num}_{cam_part}.txt")

        shutil.copy2(img_src, img_dst)
        shutil.copy2(label_file, label_dst)
        stats[split] += 1

# -----------------------------
# Print stats
# -----------------------------
print("\nTrain/Val/Test split completed!")
print(f"Train: {stats['train']} pairs")
print(f"Val:   {stats['val']} pairs")
print(f"Test:  {stats['test']} pairs")
print(f"Total: {sum(stats.values())} pairs")

if missing_images:
    print(f"\nWarning: {len(missing_images)} images missing, first 10:")
    for img in missing_images[:10]:
        print(f"   - {img}")

print(f"\nDataset saved to: {output_root}/")
