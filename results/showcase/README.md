# Showcase Results (tracked)

Curated artifacts for demos, reports, and reproducibility.

Recommended contents:
- `training/<run_name>/` with metrics (e.g., `results.csv`, `results.png`, `args.yaml`)
- `system/<run_name>/` with JSON outputs and rendered videos
- Representative visualization frames or short demo clips

Keep this folder lightweight; for large media, prefer smaller clips or use Git LFS.
To track runs:
- Set `project: results/showcase/training` in `config/train.yaml`
- Set `output.dir: results/showcase/system/<run_name>` in your system config (copy one of `config/system_demo*.yaml`)
