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
output_root = "data/processed/data_train_val_test_car_person"

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

print("Creating train/val/test split (keeping ALL camera views together)...")
print(f"   Ratios: Train={train_ratio}, Val={val_ratio}, Test={test_ratio}\n")

# -----------------------------
# CLEAN OUTPUT DIRECTORY FIRST
# -----------------------------
if os.path.exists(output_root):
    print(f"Cleaning existing output directory: {output_root}")
    shutil.rmtree(output_root)
    print("   Old files removed\n")

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

print(f"Found {len(all_labels)} label files")

# -----------------------------
# Group labels by frame number (all cameras for same frame stay together)
# -----------------------------
frame_to_labels = defaultdict(list)
for label_file in all_labels:
    base = os.path.basename(label_file)
    if "_" not in base:
        print(f"Unexpected label filename format: {base}")
        continue
    
    # Extract frame number (e.g., "00000149" from "00000149_c3.txt")
    frame_num = base.split("_")[0]
    frame_to_labels[frame_num].append(label_file)

print(f"Found {len(frame_to_labels)} unique frames")

# Check camera coverage
camera_counts = [len(labels) for labels in frame_to_labels.values()]
print(f"Camera views per frame: min={min(camera_counts)}, max={max(camera_counts)}, avg={sum(camera_counts)/len(camera_counts):.1f}")

# Show example
sample_frame = list(frame_to_labels.keys())[0]
print(f"Example: Frame {sample_frame} has {len(frame_to_labels[sample_frame])} camera views:")
for lf in frame_to_labels[sample_frame][:3]:
    print(f"   - {os.path.basename(lf)}")
if len(frame_to_labels[sample_frame]) > 3:
    print(f"   ... and {len(frame_to_labels[sample_frame]) - 3} more")

# -----------------------------
# Analyze class distribution per frame
# -----------------------------
frame_classes = {}  # frame_num -> set of class_ids present in that frame
for frame_num, label_files in frame_to_labels.items():
    classes_in_frame = set()
    for label_file in label_files:
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    class_id = int(parts[0])
                    classes_in_frame.add(class_id)
    frame_classes[frame_num] = classes_in_frame

print(f"\nClass distribution:")
class_to_frames = defaultdict(set)
for frame_num, classes in frame_classes.items():
    for class_id in classes:
        class_to_frames[class_id].add(frame_num)

for class_id in sorted(class_to_frames.keys()):
    print(f"   Class {class_id}: appears in {len(class_to_frames[class_id])} frames")

# -----------------------------
# Split frames into train/val/test
# -----------------------------
frames = sorted(frame_to_labels.keys())
random.shuffle(frames)

num_train = int(len(frames) * train_ratio)
num_val   = int(len(frames) * val_ratio)

train_frames = set(frames[:num_train])
val_frames   = set(frames[num_train:num_train + num_val])
test_frames  = set(frames[num_train + num_val:])

print(f"\nInitial split (by frames, not images):")
print(f"   Train: {len(train_frames)} frames")
print(f"   Val:   {len(val_frames)} frames")
print(f"   Test:  {len(test_frames)} frames")

# -----------------------------
# Verify class balance (and fix if needed)
# -----------------------------
def check_class_coverage(split_frames, split_name):
    """Check which classes are present in a split"""
    classes_present = set()
    for frame in split_frames:
        classes_present.update(frame_classes[frame])
    return classes_present

print(f"\nChecking class coverage...")
train_classes = check_class_coverage(train_frames, "train")
val_classes = check_class_coverage(val_frames, "val")
test_classes = check_class_coverage(test_frames, "test")

all_classes = set(class_to_frames.keys())
print(f"   Train has classes: {sorted(train_classes)}")
print(f"   Val has classes:   {sorted(val_classes)}")
print(f"   Test has classes:  {sorted(test_classes)}")

# Fix missing classes by moving entire frames
def fix_missing_classes(target_frames, source_frames, target_name):
    """Move frames from source to target to ensure all classes are present"""
    target_classes = check_class_coverage(target_frames, target_name)
    missing_classes = all_classes - target_classes
    
    moved_frames = set()
    for class_id in missing_classes:
        # Find frames in source that have this class
        candidates = [f for f in source_frames if class_id in frame_classes[f]]
        if candidates:
            chosen = random.choice(candidates)
            moved_frames.add(chosen)
            print(f"   Moving frame {chosen} to {target_name} (missing class {class_id})")
    
    return moved_frames

# Balance train
moved_to_train = fix_missing_classes(train_frames, val_frames | test_frames, "train")
train_frames |= moved_to_train
val_frames -= moved_to_train
test_frames -= moved_to_train

# Balance val  
moved_to_val = fix_missing_classes(val_frames, train_frames | test_frames, "val")
val_frames |= moved_to_val
train_frames -= moved_to_val
test_frames -= moved_to_val

# Balance test
moved_to_test = fix_missing_classes(test_frames, train_frames | val_frames, "test")
test_frames |= moved_to_test
train_frames -= moved_to_test
val_frames -= moved_to_test

# Verify no overlap
assert len(train_frames & val_frames) == 0, "Train and val overlap!"
assert len(train_frames & test_frames) == 0, "Train and test overlap!"
assert len(val_frames & test_frames) == 0, "Val and test overlap!"
assert len(train_frames) + len(val_frames) + len(test_frames) == len(frames), "Frame count mismatch!"

print(f"\nFinal split (frames only, ALL cameras per frame stay together):")
print(f"   Train: {len(train_frames)} frames")
print(f"   Val:   {len(val_frames)} frames")
print(f"   Test:  {len(test_frames)} frames")

# -----------------------------
# Copy ALL camera views for each frame
# -----------------------------
stats = {"train": 0, "val": 0, "test": 0}
missing_images = []

print(f"\nCopying files...")

# Map frames to splits
frame_split_map = {}
for frame in train_frames:
    frame_split_map[frame] = "train"
for frame in val_frames:
    frame_split_map[frame] = "val"
for frame in test_frames:
    frame_split_map[frame] = "test"

# Copy all files
for frame_num, label_files in frame_to_labels.items():
    split = frame_split_map[frame_num]
    
    # Copy ALL camera views of this frame to the SAME split
    for label_file in label_files:
        base = os.path.basename(label_file)
        parts = base.split("_")
        cam_part = "_".join(parts[1:]).replace(".txt", "")

        # Source image path
        img_src = os.path.join(image_root, cam_part, f"{frame_num}.jpg")
        
        if not os.path.exists(img_src):
            missing_images.append(img_src)
            continue

        # Destination paths
        img_dst = os.path.join(output_root, split, "images", f"{frame_num}_{cam_part}.jpg")
        label_dst = os.path.join(output_root, split, "labels", f"{frame_num}_{cam_part}.txt")

        shutil.copy2(img_src, img_dst)
        
        # Filter label file to only keep person (0) and car (1)
        with open(label_file, "r") as f_in, open(label_dst, "w") as f_out:
            for line in f_in:
                parts = line.strip().split()
                if len(parts) > 0:
                    cls_id = int(parts[0])
                    if cls_id in [0, 1]:  # Keep only person and car
                        f_out.write(line)
        
        stats[split] += 1

# Calculate images per frame for each split
for split in ["train", "val", "test"]:
    split_frames = train_frames if split == "train" else (val_frames if split == "val" else test_frames)
    total_images = sum(len(frame_to_labels[f]) for f in split_frames)
    print(f"   {split}: {len(split_frames)} frames x ~{total_images/len(split_frames):.1f} cameras = {total_images} images")

# -----------------------------
# Create YOLO dataset.yaml
# -----------------------------
yaml_content = f"""# Dataset configuration for YOLO training
path: {output_root}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: val/images  # val images (relative to 'path')
test: test/images  # test images (relative to 'path')

# Classes
names:
  0: person
  1: car

# Number of classes
nc: 2
"""

yaml_path = os.path.join(output_root, "dataset.yaml")
with open(yaml_path, 'w') as f:
    f.write(yaml_content)

# -----------------------------
# Print final stats
# -----------------------------
print("\nDataset split completed!")
print(f"   Train: {stats['train']} images from {len(train_frames)} frames")
print(f"   Val:   {stats['val']} images from {len(val_frames)} frames")
print(f"   Test:  {stats['test']} images from {len(test_frames)} frames")
print(f"   Total: {sum(stats.values())} images from {len(frames)} frames")

if missing_images:
    print(f"\nWarning: {len(missing_images)} images missing")

print(f"\nDataset saved to: {output_root}/")
print(f"YAML config: {yaml_path}")

# Verify no frame appears in multiple splits
print("\nVerifying frame integrity...")
sample_checks = random.sample(list(frame_to_labels.keys()), min(5, len(frame_to_labels)))
all_good = True
for frame in sample_checks:
    splits_found = []
    for split in ["train", "val", "test"]:
        pattern = os.path.join(output_root, split, "images", f"{frame}_*.jpg")
        if glob(pattern):
            splits_found.append(split)
    
    if len(splits_found) == 1:
        print(f"   Frame {frame}: all cameras in {splits_found[0]}")
    else:
        print(f"   Frame {frame}: found in {splits_found}!")
        all_good = False

if all_good:
    print("\nAll frames correctly grouped with all their cameras!")
else:
    print("\nERROR: Some frames split across multiple sets!")