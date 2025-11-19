# Multi-View 3D Object Reconstruction

This project reconstructs a 3D point cloud from multiple 2D images. It uses the ORB algorithm for feature detection, matches features across views, and then uses triangulation to recover the 3D coordinates.

## Getting Started

1.  **Clone and Install:**
    ```bash
    git clone <repository-url>
    cd Multi-view-object-detection
    pip install -r requirements.txt
    ```

2.  **Generate Calibration Data:**
    This step is required if you are using the EPFL dataset.
    ```bash
    python create_calibration.py
    ```

3.  **Run Reconstruction:**
    ```bash
    python main.py
    ```

## Output

The reconstructed 3D point cloud is saved to `output_cloud.obj`, which can be opened with a 3D viewer like [MeshLab](https://www.meshlab.net/).
