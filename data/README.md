# Data (not tracked)

This repo keeps data out of git. Place inputs/outputs under these paths:

- `data/raw/`        raw videos (one folder per capture/session)
- `data/processed/`  generated YOLO datasets (from `mvot label`)

Example layout:

```
data/raw/session_01/Cam1.mp4
data/raw/session_01/Cam2.mp4
data/processed/sam3_autolabel_allcams/dataset.yaml
```

Large artifacts (datasets, runs, weights) should live outside git and be shared via tarball if needed.
