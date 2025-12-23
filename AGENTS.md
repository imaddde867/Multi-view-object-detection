# Repository Guidelines

## Project Structure & Module Organization
- `multiview/`: core pipeline and CLI (`labeling/`, `training/`, `tracking/`, `system/`, `utils/`).
- `config/`: YAML configs for labeling (`labeling.yaml`), training (`train.yaml`), and system demos (`system_demo*.yaml`; copy to make your own run config).
- `data/raw/` and `data/processed/`: input videos and YOLO-style datasets.
- `checkpoints/`: local model weights (`yolo/`, `sam3/`).
- `results/` and `runs/`: outputs and MLflow runs (typically git-ignored).
- `scripts/` and `slurm/`: debugging utilities and Puhti job scripts.

## Build, Test, and Development Commands
Install dependencies and the editable package:
```bash
pip install -r requirements.txt
pip install -e .
```
Optional extras:
```bash
pip install -e ".[tracking]"  # torch/torchvision embedder
pip install -e sam3           # SAM3 labeling from source
```
Core workflows:
```bash
multiview label --config config/labeling.yaml
multiview train --config config/train.yaml
multiview run --config config/system_demo_tuned.yaml  # or your custom copy
multiview verify --dataset data/processed/<dataset_name>/dataset.yaml
```
Demo run uses `config/system_demo_tuned.yaml` (repo-root demo videos).

## Coding Style & Naming Conventions
- Python 3.10+; follow existing module boundaries in `multiview/`.
- Use 4-space indentation, `snake_case` for functions/modules, `CamelCase` for classes.
- Configs are YAML under `config/` with descriptive names (e.g., `system_demo_tuned.yaml`).
- Dataset layout follows `data/processed/<dataset_name>/` with `train/`, `val/`, `test/`.
- No formatter/linter is enforced; keep style consistent with nearby files.

## Testing Guidelines
- No automated test framework or coverage targets are defined.
- Use `multiview verify` to validate datasets and a small demo run to smoke-test changes.

## Commit & Pull Request Guidelines
- Commit subjects are short, imperative, and descriptive (e.g., "Add tuned demo config").
- PRs should include: a brief summary, configs used, and where outputs are written.
- Do not commit large assets; `data/`, `checkpoints/`, `results/`, and `runs/` are intended to stay local.

## Configuration & Assets Tips
- Place SAM3 weights at `checkpoints/sam3/sam3.pt` if using labeling.
- If running on CPU, set `runtime.device: cpu` and `runtime.half: false` in configs.
