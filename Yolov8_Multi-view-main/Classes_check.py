import os
from glob import glob
from collections import Counter
from pathlib import Path

YOLO_DIR = Path(__file__).resolve().parent
label_root = YOLO_DIR / "data_train-test"

# Class mapping
class_names = {0: "person", 1: "car", 2: "bus"}

# Counts how many images contain each class
images_per_class = Counter()

for split in ["train", "test"]:
    label_files = glob(str(label_root / split / "labels" / "*.txt"))
    for lbl_file in label_files:
        with open(lbl_file, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        classes_in_file = {int(line.split()[0]) for line in lines}  # unique classes in image
        for cls in classes_in_file:
            images_per_class[cls] += 1

print("Number of images containing each class (train + test):")
for cls_id, count in sorted(images_per_class.items()):
    cls_name = class_names.get(cls_id, f"Unknown({cls_id})")
    print(f"Class {cls_id} ({cls_name}): {count} images")
