import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os

def generate_plots():
    csv_path = 'demo_material/training_metrics.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Strip whitespace from column names just in case
    df.columns = df.columns.str.strip()

    plt.figure(figsize=(10, 6))
    
    # Plot Losses
    plt.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', linestyle='--', alpha=0.7)
    plt.plot(df['epoch'], df['train/cls_loss'], label='Train Class Loss', linestyle='--', alpha=0.7)
    plt.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', linewidth=2)
    plt.plot(df['epoch'], df['val/cls_loss'], label='Val Class Loss', linewidth=2)
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Convergence: Loss vs Epochs')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    output_path = 'docs/training_metrics.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()

    # Plot mAP
    plt.figure(figsize=(10, 6))
    plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@50', color='green', linewidth=2)
    plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@50-95', color='blue', linewidth=2)
    
    plt.xlabel('Epochs')
    plt.ylabel('mAP Score')
    plt.title('Model Accuracy: mAP over Training')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    output_path_map = 'docs/training_map.png'
    plt.savefig(output_path_map, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path_map}")
    plt.close()

def extract_video_frame():
    video_path = 'demo_material/demo_cam34.avi'
    output_dir = 'docs/candidate_frames'
    
    if not os.path.exists(video_path):
        print(f"Error: {video_path} not found.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        print("Error: Video has 0 frames or could not be read.")
        return

    print(f"Video has {total_frames} frames. Extracting candidates...")

    # Extract 20 frames evenly distributed, skipping the very first few to avoid black screens
    num_samples = 20
    start_frame = min(10, total_frames - 1) # Start a bit in
    end_frame = max(total_frames - 10, start_frame) # End a bit early
    
    indices = [int(start_frame + (end_frame - start_frame) * i / (num_samples - 1)) for i in range(num_samples)]
    
    saved_count = 0
    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            output_path = os.path.join(output_dir, f'candidate_batch2_{i+1:02d}_frame_{idx}.jpg')
            cv2.imwrite(output_path, frame)
            saved_count += 1
            print(f"  Saved {output_path}")
        else:
            print(f"  Warning: Could not read frame {idx}")
    
    cap.release()
    print(f"Done. Saved {saved_count} candidate frames to {output_dir}")

if __name__ == "__main__":
    generate_plots()
    extract_video_frame()
