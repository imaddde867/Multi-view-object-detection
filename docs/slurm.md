# SLURM Usage

This repo includes example SLURM job files in `slurm/`.

## Puhti notes (CSC)

- Do **not** run `mvot label/train/run` on the **login node** (no GPUs).
- Load the CUDA-enabled PyTorch module inside jobs/sessions:

```bash
module purge
module load pytorch/2.7
```

## Interactive allocation (example)

```bash
salloc -A project_2015432 -p gpu --gres=gpu:v100:1 \
       --cpus-per-task=16 --mem=128G --time=24:00:00
```

Then (on the allocated node), run:

```bash
mvot label --config config/labeling.yaml --sam3-checkpoint /path/to/sam3.pt
mvot train --config config/train.yaml
mvot run --config config/system.yaml
```

Tip: on Puhti, `srun --pty` is less error-prone than `salloc + ssh`:

```bash
srun -A project_2015432 -p gpu --gres=gpu:v100:1 --cpus-per-task=16 --mem=128G --time=00:20:00 --pty bash -l
```

## Batch jobs

- Labeling: `sbatch slurm/label.sbatch`
- Training: `sbatch slurm/train.sbatch`
- System run: `sbatch slurm/run.sbatch`

## One-command end-to-end submission

Submit label → train → run with dependencies:

```bash
bash slurm/submit_pipeline.sh
```

You can override configs/checkpoint via env vars:

```bash
SAM3_CHECKPOINT=/scratch/project_2015432/exp/checkpoints/sam3/sam3.pt \
MVOT_LABEL_CONFIG=config/labeling.yaml \
MVOT_TRAIN_CONFIG=config/train.yaml \
MVOT_RUN_CONFIG=config/system.yaml \
bash slurm/submit_pipeline.sh
```

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
