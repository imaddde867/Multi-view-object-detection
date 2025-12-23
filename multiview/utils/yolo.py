from __future__ import annotations

from pathlib import Path
from typing import Any


def ensure_weights(weights: str) -> None:
    weights_path = Path(weights)
    if not weights_path.suffix:
        return
    if weights_path.is_absolute() or weights_path.parent != Path("."):
        if not weights_path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {weights_path}")
        return
    if not weights_path.exists():
        print(
            "⚠️ YOLO weights not found locally. Ultralytics will try to download them. "
            "If you're offline, download weights and pass a local path."
        )


def _extract_names(model: Any) -> dict[int, str]:
    names_obj: Any = getattr(model, "names", None)
    if isinstance(names_obj, dict):
        return {int(k): str(v) for k, v in names_obj.items()}
    if isinstance(names_obj, (list, tuple)):
        return {i: str(n) for i, n in enumerate(names_obj)}
    return {}


def load_yolo_model(weights: str, *, device: str = "", half: bool = False) -> tuple[Any, dict[int, str]]:
    try:
        from ultralytics import YOLO
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: ultralytics. Install with `pip install -r requirements.txt`.") from e

    ensure_weights(weights)

    model = YOLO(weights)
    if device:
        try:
            model.to(device)
        except Exception:
            pass
    if half:
        try:
            model.model.half()
        except Exception:
            pass
    names = _extract_names(model)
    return model, names
