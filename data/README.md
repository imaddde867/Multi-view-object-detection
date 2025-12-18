# Data (not tracked)

This repo keeps data out of git. Place inputs/outputs under these paths:

- `data/raw/`        raw videos (one folder per capture/session)
- `data/processed/`  generated YOLO datasets (from `multiview label`)
- `data/processed/showcase/`  small, tracked sample datasets for demos

Example layout:

```
data/raw/testing_videos/Cam5.mp4
data/raw/testing_videos/Cam6.mp4
data/raw/multiclass_ground_truth/...
data/raw/multiclass_ground_truth_images/...
data/processed/sam3_autolabel_v2/dataset.yaml
```

Large artifacts (datasets, runs, weights) should live outside git; track only curated samples in `data/processed/showcase/`.
