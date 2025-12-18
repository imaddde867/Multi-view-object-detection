# Data (not tracked)

This repo keeps data out of git. Place inputs/outputs under these paths:

- `data/raw/`        raw videos (one folder per capture/session)
- `data/processed/`  generated YOLO datasets (from `multiview label`)
- `data/processed/showcase/`  small, tracked sample datasets for demos

Example layout:

```
data/raw/session_01/Cam5.mp4
data/raw/session_01/Cam6.mp4
data/processed/sam3_autolabel_allcams/dataset.yaml
```

Large artifacts (datasets, runs, weights) should live outside git; track only curated samples in `data/processed/showcase/`.
