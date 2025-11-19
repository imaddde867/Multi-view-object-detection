# Multi-View 3D Object Triangulation

Convert matched 2D detections across 6 EPFL cameras into 3D world coordinates using Direct Linear Transform (DLT).

## Quick Start

```bash
pip install -r requirements.txt
python create_calibration.py
```

## Pipeline

```
YOLO (or other detector)
  ↓
2D Detections per camera
  ↓
Match Across Views
  ↓
Triangulator.triangulate_points()
  ↓
3D Positions (X, Y, Z)
```

## Core API

**File:** `triangulation.py`

```python
from triangulation import Triangulator
import json
import numpy as np

# Load calibration
with open('epfl-calibration.json') as f:
    data = json.load(f)
    projections = [np.array(data['cameras'][f'camera{i}']['P']) for i in range(6)]

triangulator = Triangulator(projections)

# Triangulate matched 2D points from two cameras
points_3d = triangulator.triangulate_points(
    cam_idx1=0,           # Reference camera
    cam_idx2=1,           # Target camera
    points1=pts_cam0,     # Nx2 array
    points2=pts_cam1      # Nx2 array
)
# Returns: Mx3 array (M <= N, invalid points filtered)

# Validate quality
errors, mean_error = triangulator.compute_reprojection_error(
    point_3d=points_3d[0],
    points_2d=[(u0,v0), (u1,v1), ...]
)
```

### Methods

| Method | Input | Output | Purpose |
|--------|-------|--------|---------|
| `triangulate_points(cam1, cam2, pts1, pts2)` | Camera indices, Nx2 arrays | Mx3 array | 2-view triangulation with depth validation |
| `triangulate_dlt(points_2d)` | List of (u,v) per camera | (X,Y,Z) | Multi-view DLT (all 6 cameras) |
| `compute_reprojection_error(pt_3d, pts_2d)` | 3D point, 2D observations | (errors, mean) | Quality metric (pixels) |
| `is_in_front_of_camera(pt_3d, cam_idx)` | 3D point, camera index | bool | Depth check |

## For YOLO Team

### Input Format

```python
# Per frame, per camera: list of detections
detections = {
    0: [{'bbox': (x, y, w, h), 'class': 'person', 'conf': 0.95},
        {'bbox': (x, y, w, h), 'class': 'car', 'conf': 0.87}],
    1: [{'bbox': (x, y, w, h), 'class': 'person', 'conf': 0.92}],
    # ... cameras 2-5
}
```

### Match & Triangulate

```python
# Match detections across cameras (by class + spatial proximity)
matched = {
    'person_1': {0: (u0, v0), 1: (u1, v1), 2: (u2, v2)},
    'car_1': {0: (u0, v0), 3: (u3, v3)},
}

# Triangulate each
for obj_id, views in matched.items():
    cam_indices = sorted(views.keys())
    pts_ref = np.array([views[cam_indices[0]]])
    pts_tgt = np.array([views[cam_indices[1]]])
    
    pt_3d = triangulator.triangulate_points(
        cam_indices[0], cam_indices[1],
        pts_ref, pts_tgt
    )
    
    if len(pt_3d) > 0:
        print(f"{obj_id}: {pt_3d[0]}")
```

### Optional: Multi-View Refinement

```python
# Use all visible cameras for better accuracy
points_all = [views.get(i, None) for i in range(6)]
if sum(1 for p in points_all if p) >= 2:
    pt_3d = triangulator.triangulate_dlt(points_all)
```

## What We Fixed

| Issue | Solution |
|-------|----------|
| Random point cloud | Added all 6 cameras + depth validation |
| Poor matching | RANSAC epipolar filtering (F-matrix, 5px threshold) |
| Behind-camera points | Depth checks (Z > 0) |
| Out-of-bounds reprojection | Image bounds tolerance (±200px) |

**Results (Frame 150):** 387 raw → 156 valid points

## Dataset

**EPFL Terrace Multi-View Multi-Class**
- 6 cameras (360×288)
- Classes: Person, Car, Bus
- Coordinate scale: ~10k units
- Calibration: `epfl-calibration.json`

## Output

- `output_cloud.obj` — 3D points (MeshLab viewer)
- Console logs — Feature counts, match rates, errors

---

⚠️ **Important:** This reconstructs ANY matched 2D points. Use YOLO to extract objects first. Raw feature matching produces background noise.

**Next:** Implement YOLO → match detections → feed to `triangulate_points()` → get 3D positions.
