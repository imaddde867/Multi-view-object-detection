from __future__ import annotations

import inspect
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

    def refine_xyxy_with_prompts(
        self,
        frame_bgr: np.ndarray,
        boxes_xyxy: list[tuple[float, float, float, float]],
        prompts: list[str],
    ) -> list[tuple[float, float, float, float]]:
        """
        Refine boxes using SAM3.

        Prefer a native "boxes API" if the installed SAM3 version supports it. If not,
        fall back to prompt-based segmentation on per-box crops.
        """
        if not boxes_xyxy:
            return []
        if len(prompts) != len(boxes_xyxy):
            raise ValueError("Expected prompts to match boxes length.")

        try:
            return self.refine_xyxy(frame_bgr, boxes_xyxy)
        except RuntimeError as e:
            msg = str(e).lower()
            if "boxes api" not in msg and "incompatible interface" not in msg:
                raise
            return [self._refine_box_via_prompt_crop(frame_bgr, box, prompt) for box, prompt in zip(boxes_xyxy, prompts)]

    def _predict_masks_from_boxes(self, image_rgb: np.ndarray, boxes_xyxy: np.ndarray) -> list[np.ndarray | None]:
        proc = self.processor
        boxes = boxes_xyxy.astype(np.float32)

        candidates: list[tuple[str, dict[str, Any]]] = [
            ("predict", {"image": image_rgb, "boxes": boxes}),
            ("predict", {"image": image_rgb, "box": boxes}),
            ("predict", {"image": image_rgb, "bboxes": boxes}),
            ("predict", {"image": image_rgb, "input_boxes": boxes}),
            ("process_image", {"image": image_rgb, "boxes": boxes}),
            ("__call__", {"image": image_rgb, "boxes": boxes}),
        ]

        last_err: Exception | None = None
        for name, kwargs in candidates:
            fn = getattr(proc, name, None)
            if fn is None:
                continue
            try:
                out = _call_with_supported_kwargs(fn, kwargs)
            except Exception as e:
                last_err = e
                continue

            masks = _extract_masks(out)
            if masks is None:
                continue

            masks = _normalize_masks(masks, expected_n=len(boxes))
            return masks

        raise RuntimeError(
            "Could not call SAM3 processor with a boxes API. The installed SAM3 version has an incompatible interface."
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

    def _refine_box_via_prompt_crop(
        self,
        frame_bgr: np.ndarray,
        xyxy: tuple[float, float, float, float],
        prompt: str,
        *,
        pad_frac: float = 0.15,
    ) -> tuple[float, float, float, float]:
        h, w = frame_bgr.shape[:2]
        x0, y0, x1, y1 = [float(v) for v in xyxy]
        bw = max(1.0, x1 - x0)
        bh = max(1.0, y1 - y0)
        pad_x = bw * float(pad_frac)
        pad_y = bh * float(pad_frac)
        cx0 = int(max(0.0, x0 - pad_x))
        cy0 = int(max(0.0, y0 - pad_y))
        cx1 = int(min(float(w), x1 + pad_x))
        cy1 = int(min(float(h), y1 + pad_y))
        if cx1 <= cx0 or cy1 <= cy0:
            return xyxy

        crop_bgr = frame_bgr[cy0:cy1, cx0:cx1]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        masks_scores = _predict_masks_from_text(self.processor, crop_rgb, prompt)
        if not masks_scores:
            return xyxy

        # Pick best by score, else by mask area.
        best_mask = None
        best_score = -1.0
        best_area = -1
        for mask, score in masks_scores:
            if mask is None:
                continue
            m = mask
            if m.dtype != bool:
                m = m > 0.5
            m = clean_binary_mask(m, close_kernel=self.cfg.mask_close)
            area = int(m.sum())
            if score is not None and score > best_score:
                best_score = float(score)
                best_area = area
                best_mask = m
            elif score is None and area > best_area:
                best_area = area
                best_mask = m
        if best_mask is None:
            return xyxy

        refined_crop = mask_to_xyxy(best_mask)
        if refined_crop is None:
            return xyxy
        rx0, ry0, rx1, ry1 = refined_crop
        # Offset back to full image coords.
        out = (rx0 + cx0, ry0 + cy0, rx1 + cx0, ry1 + cy0)
        return out


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


def _call_with_supported_kwargs(fn: Any, kwargs: dict[str, Any]) -> Any:
    """
    Call a function with only the kwargs it supports. This makes the integration
    resilient to SAM3 API changes across versions.
    """
    try:
        sig = inspect.signature(fn)
    except Exception:
        return fn(**kwargs)

    accepted = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            accepted[k] = v
    return fn(**accepted)


def _predict_masks_from_text(proc: Any, image_rgb: np.ndarray, prompt: str) -> list[tuple[np.ndarray | None, float | None]]:
    """
    Attempt a prompt-based segmentation call, returning a list of (mask, score).
    """
    candidates: list[tuple[str, dict[str, Any]]] = [
        ("predict", {"image": image_rgb, "text": prompt}),
        ("predict", {"image": image_rgb, "texts": [prompt]}),
        ("predict", {"image": image_rgb, "prompt": prompt}),
        ("predict", {"image": image_rgb, "prompts": [prompt]}),
        ("process_image", {"image": image_rgb, "text": prompt}),
        ("process_image", {"image": image_rgb, "texts": [prompt]}),
        ("__call__", {"image": image_rgb, "text": prompt}),
        ("__call__", {"image": image_rgb, "texts": [prompt]}),
    ]
    last_err: Exception | None = None
    for name, kwargs in candidates:
        fn = getattr(proc, name, None)
        if fn is None:
            continue
        try:
            out = _call_with_supported_kwargs(fn, kwargs)
        except Exception as e:
            last_err = e
            continue
        masks = _extract_masks(out)
        if masks is None:
            continue
        mask_list = _normalize_masks(masks, expected_n=_infer_mask_count(masks) or 1)
        scores = _extract_scores(out, expected_n=len(mask_list))
        return list(zip(mask_list, scores))
    if last_err is not None:
        return []
    return []


def _infer_mask_count(masks: Any) -> int | None:
    try:
        arr = np.asarray(masks)
        if arr.ndim == 3:
            return int(arr.shape[0])
        if arr.ndim >= 4:
            return int(arr.shape[0])
    except Exception:
        return None
    return None


def _extract_scores(out: Any, *, expected_n: int) -> list[float | None]:
    if isinstance(out, dict):
        for key in ("scores", "score", "confidences", "confidence"):
            if key in out:
                s = out[key]
                try:
                    arr = np.asarray(s, dtype=np.float32).reshape(-1)
                    if arr.size >= expected_n:
                        return [float(v) for v in arr[:expected_n]]
                except Exception:
                    pass
                break
    return [None] * expected_n
