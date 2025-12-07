import os
import shutil
import random
from glob import glob
from collections import defaultdict

# -----------------------------
# Configuration
# -----------------------------
IMAGE_ROOT = "data/raw/multiclass_ground_truth_images"
YOLO_LABEL_ROOT = "data/processed/yolo_labels"
OUTPUT_ROOT = "data/processed/data_train_balanced"

# Split Ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Balancing Parameters (Train set only)
# Bus is class 2.
BUS_OVERSAMPLE_FACTOR = 15  # Repeat bus frames this many times
OTHERS_UNDERSAMPLE_RATIO = 0.5  # Keep this fraction of non-bus frames

# Classes
CLASS_NAMES = {0: 'person', 1: 'car', 2: 'bus'}

# -----------------------------
# Setup
# -----------------------------
random.seed(42)

if os.path.exists(OUTPUT_ROOT):
    print(f"Cleaning existing output directory: {OUTPUT_ROOT}")
    shutil.rmtree(OUTPUT_ROOT)

for split in ["train", "val", "test"]:
    for subfolder in ["images", "labels"]:
        os.makedirs(os.path.join(OUTPUT_ROOT, split, subfolder), exist_ok=True)

# -----------------------------
# Data Collection
# -----------------------------
print("Collecting labels...")
all_labels = glob(os.path.join(YOLO_LABEL_ROOT, "*.txt"))
if not all_labels:
    print("No label files found!")
    exit(1)

# Group by frame
frame_to_labels = defaultdict(list)
frame_classes = defaultdict(set)

for label_file in all_labels:
    base = os.path.basename(label_file)
    frame_num = base.split("_")[0]
    frame_to_labels[frame_num].append(label_file)
    
    # Check classes in this file
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                frame_classes[frame_num].add(int(parts[0]))

frames = sorted(frame_to_labels.keys())
print(f"   Found {len(frames)} unique frames.")

# -----------------------------
# Stratified Split
# -----------------------------
# Separate frames into "has_bus" and "no_bus"
bus_frames = [f for f in frames if 2 in frame_classes[f]]
other_frames = [f for f in frames if 2 not in frame_classes[f]]

random.shuffle(bus_frames)
random.shuffle(other_frames)

print(f"   Bus frames: {len(bus_frames)}")
print(f"   Other frames: {len(other_frames)}")

def split_list(lst):
    n = len(lst)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    return lst[:n_train], lst[n_train:n_train+n_val], lst[n_train+n_val:]

train_bus, val_bus, test_bus = split_list(bus_frames)
train_others, val_others, test_others = split_list(other_frames)

# Combine for initial sets
train_frames = train_bus + train_others
val_frames = val_bus + val_others
test_frames = test_bus + test_others

print(f"\nInitial Split (Frames):")
print(f"   Train: {len(train_frames)} (Bus: {len(train_bus)})")
print(f"   Val:   {len(val_frames)} (Bus: {len(val_bus)})")
print(f"   Test:  {len(test_frames)} (Bus: {len(test_bus)})")

# -----------------------------
# Balancing (Train Only)
# -----------------------------
print("\nBalancing Training Set...")

# 1. Oversample Bus
train_bus_balanced = train_bus * BUS_OVERSAMPLE_FACTOR
print(f"   Oversampling Bus: {len(train_bus)} -> {len(train_bus_balanced)} (x{BUS_OVERSAMPLE_FACTOR})")

# 2. Undersample Others
n_others_keep = int(len(train_others) * OTHERS_UNDERSAMPLE_RATIO)
train_others_balanced = random.sample(train_others, n_others_keep)
print(f"   Undersampling Others: {len(train_others)} -> {len(train_others_balanced)} (x{OTHERS_UNDERSAMPLE_RATIO})")

# Final Train Set (List of frame IDs, can contain duplicates)
final_train_frames = train_bus_balanced + train_others_balanced
random.shuffle(final_train_frames)

print(f"   Final Balanced Train Frames: {len(final_train_frames)}")
print(f"   Ratio (Bus Frames / Total): {len(train_bus_balanced) / len(final_train_frames):.2%}")

# -----------------------------
# Copy Files
# -----------------------------
print("\nCopying files...")

def copy_frame(frame_id, split_name, suffix=""):
    """Copies all images/labels for a frame to the split folder.
       suffix: added to filename to handle duplicates (oversampling).
    """
    labels = frame_to_labels[frame_id]
    for label_file in labels:
        base = os.path.basename(label_file)
        parts = base.split("_")
        cam_part = "_".join(parts[1:]).replace(".txt", "")
        
        # Source Image
        img_src = os.path.join(IMAGE_ROOT, cam_part, f"{frame_id}.jpg")
        if not os.path.exists(img_src):
            continue
            
        # Destination Names
        # If suffix is present, insert it before extension
        if suffix:
            img_dst_name = f"{frame_id}_{cam_part}_{suffix}.jpg"
            lbl_dst_name = f"{frame_id}_{cam_part}_{suffix}.txt"
        else:
            img_dst_name = f"{frame_id}_{cam_part}.jpg"
            lbl_dst_name = f"{frame_id}_{cam_part}.txt"
            
        img_dst = os.path.join(OUTPUT_ROOT, split_name, "images", img_dst_name)
        lbl_dst = os.path.join(OUTPUT_ROOT, split_name, "labels", lbl_dst_name)
        
        shutil.copy2(img_src, img_dst)
        shutil.copy2(label_file, lbl_dst)

# Process Train (with duplicates)
# We need to handle filenames for duplicates.
# Strategy: Use a counter for each frame in the list.
frame_counts = defaultdict(int)

for frame in final_train_frames:
    count = frame_counts[frame]
    suffix = f"copy{count}" if count > 0 else ""
    copy_frame(frame, "train", suffix)
    frame_counts[frame] += 1

# Process Val/Test (No duplicates)
for frame in val_frames:
    copy_frame(frame, "val")
for frame in test_frames:
    copy_frame(frame, "test")

# -----------------------------
# Generate dataset.yaml
# -----------------------------
yaml_content = f"""
path: {os.path.abspath(OUTPUT_ROOT)}
train: train/images
val: val/images
test: test/images

nc: 3
names: ['person', 'car', 'bus']
"""

with open(os.path.join(OUTPUT_ROOT, "dataset.yaml"), "w") as f:
    f.write(yaml_content)

print(f"\nBalanced dataset created at: {OUTPUT_ROOT}")
print(f"Config saved to: {os.path.join(OUTPUT_ROOT, 'dataset.yaml')}")
