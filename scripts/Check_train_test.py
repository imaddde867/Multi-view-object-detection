import os
from collections import defaultdict

# Paths
data_root = "data/processed/data_train_val_test_car_person"
splits = ["train", "val", "test"]

def get_files(split_name, file_type):
    """Get list of files for a given split and type (images/labels)"""
    path = os.path.join(data_root, split_name, file_type)
    if not os.path.exists(path):
        return []
    
    if file_type == "labels":
        return [f for f in os.listdir(path) if f.endswith(".txt")]
    else:  # images
        return [f for f in os.listdir(path) if f.lower().endswith((".jpg", ".png", ".jpeg"))]

def extract_frame_num(filename):
    """Extract frame number from filename (e.g., '00000001_c0.txt' -> '00000001')"""
    return filename.split("_")[0]

print("="*60)
print("🔍 DATASET VALIDATION CHECK")
print("="*60)

# Collect data for all splits
split_data = {}
all_frame_nums = defaultdict(list)  # frame_num -> [list of splits it appears in]

for split in splits:
    split_path = os.path.join(data_root, split)
    if not os.path.exists(split_path):
        print(f"\n⚠️  {split.upper()} split not found at {split_path}")
        continue
    
    labels = get_files(split, "labels")
    images = get_files(split, "images")
    
    # Extract frame numbers from labels
    frame_nums = set([extract_frame_num(f) for f in labels])
    
    # Track which frames appear in which splits
    for frame in frame_nums:
        all_frame_nums[frame].append(split)
    
    split_data[split] = {
        "labels": labels,
        "images": images,
        "frame_nums": frame_nums,
        "label_count": len(labels),
        "image_count": len(images)
    }

# Print statistics for each split
print("\n📊 DATASET STATISTICS")
print("-" * 60)
for split in splits:
    if split not in split_data:
        continue
    
    data = split_data[split]
    print(f"\n{split.upper()}:")
    print(f"   Images: {data['image_count']}")
    print(f"   Labels: {data['label_count']}")
    print(f"   Unique frames: {len(data['frame_nums'])}")
    
    # Check for mismatches
    if data['image_count'] != data['label_count']:
        print(f"   ⚠️  Mismatch: {abs(data['image_count'] - data['label_count'])} files don't have pairs")

# Check for data leakage (frames appearing in multiple splits)
print("\n" + "="*60)
print("🔒 DATA LEAKAGE CHECK")
print("="*60)

leakage_found = False
for frame_num, splits_list in all_frame_nums.items():
    if len(splits_list) > 1:
        if not leakage_found:
            print("\n❌ DATA LEAKAGE DETECTED!")
            print("   The following frames appear in multiple splits:")
            leakage_found = True
        print(f"   Frame {frame_num}: {', '.join(splits_list)}")

if not leakage_found:
    print("\n✅ No data leakage detected!")
    print("   Each frame appears in only one split.")

# Check for orphaned files (images without labels or vice versa)
print("\n" + "="*60)
print("🔗 ORPHANED FILES CHECK")
print("="*60)

for split in splits:
    if split not in split_data:
        continue
    
    data = split_data[split]
    
    # Get basenames without extensions
    label_bases = set([os.path.splitext(f)[0] for f in data['labels']])
    image_bases = set([os.path.splitext(f)[0] for f in data['images']])
    
    # Find orphans
    images_without_labels = image_bases - label_bases
    labels_without_images = label_bases - image_bases
    
    if images_without_labels or labels_without_images:
        print(f"\n{split.upper()}:")
        
        if images_without_labels:
            print(f"   ⚠️  {len(images_without_labels)} images without labels")
            if len(images_without_labels) <= 5:
                for img in sorted(images_without_labels):
                    print(f"      - {img}")
            else:
                for img in sorted(list(images_without_labels)[:3]):
                    print(f"      - {img}")
                print(f"      ... and {len(images_without_labels) - 3} more")
        
        if labels_without_images:
            print(f"   ⚠️  {len(labels_without_images)} labels without images")
            if len(labels_without_images) <= 5:
                for lbl in sorted(labels_without_images):
                    print(f"      - {lbl}")
            else:
                for lbl in sorted(list(labels_without_images)[:3]):
                    print(f"      - {lbl}")
                print(f"      ... and {len(labels_without_images) - 3} more")
    else:
        print(f"\n{split.upper()}: ✅ All files have matching pairs")

# Summary
print("\n" + "="*60)
print("📝 SUMMARY")
print("="*60)

total_images = sum(data['image_count'] for data in split_data.values())
total_labels = sum(data['label_count'] for data in split_data.values())
total_frames = len(all_frame_nums)

print(f"\nTotal unique frames: {total_frames}")
print(f"Total images: {total_images}")
print(f"Total labels: {total_labels}")

if total_images == total_labels:
    print("\n✅ Dataset appears to be properly structured!")
else:
    print(f"\n⚠️  Warning: Image/label count mismatch ({abs(total_images - total_labels)} difference)")

if not leakage_found:
    print("✅ No data leakage between splits!")
else:
    print("❌ Data leakage detected - frames appear in multiple splits!")

print("\n" + "="*60)