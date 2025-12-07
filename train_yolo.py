import argparse
import os
import sys

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ Ultralytics YOLOv8 not installed. Please run: pip install ultralytics")
    sys.exit(1)

def train(mode, epochs, img_size, batch_size):
    print(f"Starting training in {mode} mode...")
    
    # Select Dataset
    if mode == "balanced":
        dataset_path = os.path.abspath("data_train_balanced/dataset.yaml")
        project_name = "Detection_Balanced"
    elif mode == "2class":
        dataset_path = os.path.abspath("data_train_val_test_car_person/dataset.yaml")
        project_name = "Detection_2Class"
    else:
        print(f"Unknown mode: {mode}")
        return

    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Dataset config not found: {dataset_path}")
        print(f"   Please run the corresponding data preparation script first.")
        return

    print(f"Dataset: {dataset_path}")
    print(f"Epochs: {epochs}, Img Size: {img_size}, Batch: {batch_size}")

    # Check for device (MPS for Mac M-series, CUDA, or CPU)
    import torch
    if torch.backends.mps.is_available():
        device = "mps"
        print("Apple Metal (MPS) GPU detected and enabled!")
    elif torch.cuda.is_available():
        device = "0"
        print("NVIDIA CUDA GPU detected and enabled!")
    else:
        device = "cpu"
        print("No GPU detected. Training on CPU (this will be slow).")

    # Load Model (Nano model for speed, can be changed)
    model = YOLO("yolov8n.pt") 

    # Train
    results = model.train(
        data=dataset_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=project_name,
        name=f"yolov8n_{mode}",
        exist_ok=True, # Overwrite if exists
        patience=10, # Early stopping
        verbose=True,
        device=device
    )

    print(f"Training completed. Results saved in {project_name}/yolov8n_{mode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Multi-view Dataset")
    parser.add_argument("--mode", type=str, required=True, choices=["balanced", "2class"], help="Training mode")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--img", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    
    args = parser.parse_args()
    
    train(args.mode, args.epochs, args.img, args.batch)
