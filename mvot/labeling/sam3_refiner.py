from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from mvot.labeling.masks import clean_binary_mask, mask_to_xyxy


@dataclass(frozen=True)
class Sam3Config:
    checkpoint: str
    load_from_hf: bool
    device: str
    confidence: float
    mask_close: int


class Sam3BoxRefiner:
    def __init__(self, cfg: Sam3Config):
        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "SAM3 is required for labeling but is not installed.\n"
                "Install it from https://github.com/facebookresearch/sam3 and ensure `sam3` is importable."
            ) from e

        if cfg.checkpoint:
            model = build_sam3_image_model(
                checkpoint_path=cfg.checkpoint,
                load_from_HF=False,
                device=cfg.device,
                enable_segmentation=True,
                enable_inst_interactivity=False,
            )
        else:
            if not cfg.load_from_hf:
                raise ValueError("SAM3 checkpoint required. Set `sam3.checkpoint` or enable `sam3.load_from_hf`.")
            model = build_sam3_image_model(
                checkpoint_path=None,
                load_from_HF=True,
                device=cfg.device,
                enable_segmentation=True,
                enable_inst_interactivity=False,
            )

        self.processor = Sam3Processor(model, device=cfg.device, confidence_threshold=float(cfg.confidence))
        self.cfg = cfg

    def _predict_masks_from_boxes(self, image_rgb: np.ndarray, boxes_xyxy: np.ndarray) -> list[np.ndarray | None]:
        proc = self.processor
        boxes = boxes_xyxy.astype(np.float32)

        candidates: list[tuple[str, dict[str, Any]]] = [
            ("predict", {"image": image_rgb, "boxes": boxes}),
            ("predict", {"image": image_rgb, "box": boxes}),
            ("process_image", {"image": image_rgb, "boxes": boxes}),
            ("__call__", {"image": image_rgb, "boxes": boxes}),
        ]

        last_err: Exception | None = None
        for name, kwargs in candidates:
            try:
                fn = getattr(proc, name)
            except Exception:
                continue
            try:
                out = fn(**kwargs)
            except TypeError as e:
                last_err = e
                continue

            masks = _extract_masks(out)
            if masks is None:
                continue

            masks = _normalize_masks(masks, expected_n=len(boxes))
            return masks

        raise RuntimeError(
            "Could not call SAM3 processor with a boxes API. "
            "The installed SAM3 version has an incompatible interface."
        ) from last_err

    def refine_xyxy(
        self,
        frame_bgr: np.ndarray,
        boxes_xyxy: list[tuple[float, float, float, float]],
    ) -> list[tuple[float, float, float, float]]:
        if not boxes_xyxy:
            return []
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        masks = self._predict_masks_from_boxes(image_rgb, np.array(boxes_xyxy, dtype=np.float32))

        refined: list[tuple[float, float, float, float]] = []
        for fallback_xyxy, mask in zip(boxes_xyxy, masks):
            if mask is None:
                refined.append(fallback_xyxy)
                continue
            m = mask
            if m.dtype != bool:
                m = m > 0.5
            m = clean_binary_mask(m, close_kernel=self.cfg.mask_close)
            xyxy = mask_to_xyxy(m)
            refined.append(xyxy if xyxy is not None else fallback_xyxy)
        return refined


def _extract_masks(out: Any) -> Any | None:
    if out is None:
        return None
    if isinstance(out, dict):
        for key in ("masks", "mask", "pred_masks"):
            if key in out:
                return out[key]
        return None
    if isinstance(out, (list, tuple)) and out:
        for item in out:
            if isinstance(item, dict) and ("masks" in item or "mask" in item):
                return item.get("masks", item.get("mask"))
        if len(out) >= 1:
            return out[0]
    return None


def _normalize_masks(masks: Any, *, expected_n: int) -> list[np.ndarray | None]:
    if isinstance(masks, list):
        if len(masks) == expected_n:
            return [m if m is None else np.asarray(m) for m in masks]
        if len(masks) and isinstance(masks[0], (list, tuple, np.ndarray)):
            # Sometimes SAM outputs a list of candidate masks per prompt/box.
            out: list[np.ndarray | None] = []
            for item in masks[:expected_n]:
                if item is None:
                    out.append(None)
                else:
                    arr = np.asarray(item)
                    if arr.ndim >= 3:
                        out.append(arr[0])
                    else:
                        out.append(arr)
            while len(out) < expected_n:
                out.append(None)
            return out
        return [None] * expected_n

    arr = np.asarray(masks)
    if arr.ndim == 2:
        return [arr] * expected_n
    if arr.ndim == 3 and arr.shape[0] == expected_n:
        return [arr[i] for i in range(expected_n)]
    if arr.ndim >= 4 and arr.shape[0] == expected_n:
        return [arr[i, 0] for i in range(expected_n)]
    return [None] * expected_n

