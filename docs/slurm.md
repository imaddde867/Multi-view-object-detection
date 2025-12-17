# SLURM Usage

This repo includes example SLURM job files in `slurm/`.

## Interactive allocation (example)

```bash
salloc -A project_2015432 -p gpu --gres=gpu:v100:1 \
       --cpus-per-task=16 --mem=128G --time=24:00:00
```

Then run:

```bash
mvot label --config config/labeling.yaml --sam3-checkpoint /path/to/sam3.pt
mvot train --config config/train.yaml
mvot run --config config/system.yaml
```

## Batch jobs

- Labeling: `sbatch slurm/label.sbatch`
- Training: `sbatch slurm/train.sbatch`
- System run: `sbatch slurm/run.sbatch`

## Scaling to 2–4 V100 GPUs

1) Request more GPUs in SLURM (example 2 GPUs):

```bash
#SBATCH --gres=gpu:v100:2
```

2) Set `runtime.device` in `config/train.yaml`:

```yaml
runtime:
  device: "0,1"
```

Ultralytics handles multi-GPU internally when multiple devices are provided.

