# Showcase Dataset (tracked)

Small, curated samples to prove labeling quality and enable quick demos.

Recommended contents:
- A minimal YOLO dataset subset (images/labels)
- `dataset.yaml`, `meta.json`, `stats.json`

Keep this folder compact; full datasets should live outside git.
To track a dataset, set `out: data/processed/showcase/<dataset_name>` in `config/labeling.yaml`.
