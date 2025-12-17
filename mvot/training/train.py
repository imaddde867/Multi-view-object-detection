from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainConfig:
    data: str
    model: str
    epochs: int
    imgsz: int
    batch: int
    project: str
    name: str
    seed: int
    workers: int
    device: str
    half: bool


def _default_cfg() -> dict[str, Any]:
    return {
        "data": "",
        "model": "yolo11m.pt",
        "epochs": 100,
        "imgsz": 960,
        "batch": 16,
        "project": "results/training",
        "name": "",
        "seed": 0,
        "workers": 8,
        "runtime": {"device": "", "half": False},
    }


def _as_train_config(cfg: dict[str, Any]) -> TrainConfig:
    base = _default_cfg()
    for k, v in cfg.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k].update(v)
        else:
            base[k] = v

    runtime = base["runtime"]
    return TrainConfig(
        data=str(base["data"]).strip(),
        model=str(base["model"]).strip(),
        epochs=int(base["epochs"]),
        imgsz=int(base["imgsz"]),
        batch=int(base["batch"]),
        project=str(base["project"]).strip(),
        name=str(base["name"]).strip(),
        seed=int(base["seed"]),
        workers=int(base["workers"]),
        device=str(runtime.get("device", "")).strip(),
        half=bool(runtime.get("half", False)),
    )


def train_yolo(cfg: dict[str, Any]) -> None:
    c = _as_train_config(cfg)
    if not c.data:
        raise ValueError("Missing dataset yaml path. Set `data` or pass `--data`.")
    data_path = Path(c.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

    try:
        from ultralytics import YOLO
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: ultralytics. Install with `pip install -r requirements.txt`.") from e

    try:
        import torch

        if not c.device:
            if torch.cuda.is_available():
                device = "0"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        else:
            device = c.device
    except Exception:
        device = c.device or "cpu"

    run_name = c.name or f"{Path(c.model).stem}"

    project_dir = Path(c.project)
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / run_name).mkdir(parents=True, exist_ok=True)
    (project_dir / run_name / "mvot_train_config.json").write_text(json.dumps(c.__dict__, indent=2))

    model = YOLO(c.model)

    if c.half:
        try:
            model.model.half()
        except Exception:
            pass

    print(f"Training model={c.model} data={data_path} device={device} epochs={c.epochs} imgsz={c.imgsz} batch={c.batch}")
    model.train(
        data=str(data_path),
        epochs=c.epochs,
        imgsz=c.imgsz,
        batch=c.batch,
        seed=c.seed,
        workers=c.workers,
        device=device,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        verbose=True,
    )

