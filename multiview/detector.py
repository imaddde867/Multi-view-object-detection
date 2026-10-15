"""The single YOLO adapter behind the label and run paths.

Both pipelines used to carry their own copy of the same adapter: how to
unpack an ultralytics result into raw boxes, and how to remap source class
names onto target classes. This module owns that: the label path and the
run path each call one interface, and the unpack/remap behavior is pinned
by tests in exactly one place.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from multiview.utils.boxes import Det
from multiview.utils.yolo import load_yolo_model

RawDet = tuple[tuple[float, float, float, float], float, int]


def parse_kv_map(mapping_str: str) -> dict[str, str]:
    """Parse "a=b,c=d" into a dict, as used for detector.source_map."""
    if not mapping_str:
        return {}
    mapping: dict[str, str] = {}
    for part in mapping_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid mapping entry: {part} (expected k=v)")
        k, v = part.split("=", 1)
        mapping[k.strip()] = v.strip()
    return mapping


def unpack_boxes(boxes: Any) -> list[RawDet]:
    """Unpack one ultralytics ``boxes`` object into (xyxy, score, cls_id) tuples.

    Pins the single copy of the tensor-unpacking idiom the two pipelines
    used to duplicate. Empty boxes unpack to no detections.
    """
    xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
    scores = boxes.conf.cpu().numpy().astype(np.float32)
    cls_ids = boxes.cls.cpu().numpy().astype(np.int32)
    raw: list[RawDet] = []
    for b, s, c in zip(xyxy, scores, cls_ids):
        x0, y0, x1, y1 = [float(v) for v in b.tolist()]
        raw.append(((x0, y0, x1, y1), float(s), int(c)))
    return raw


def to_dets(
    raw: Sequence[RawDet],
    *,
    names: dict[int, str],
    source_map: dict[str, str],
    target_to_id: dict[str, int],
) -> list[Det]:
    """Remap raw detections onto target class ids, dropping non-targets."""
    dets: list[Det] = []
    for xyxy, score, cls_id in raw:
        src_name = names.get(int(cls_id), str(int(cls_id)))
        tgt_name = source_map.get(src_name, src_name)
        if tgt_name not in target_to_id:
            continue
        dets.append(Det(xyxy=xyxy, cls_id=target_to_id[tgt_name], score=float(score)))
    return dets


class YoloDetector:
    """Deep adapter over an ultralytics YOLO model.

    Interface: load weights, expose class names, predict frames to raw
    detections or remapped ``Det`` objects. The model import is deferred so
    tests and config code never need ultralytics installed.
    """

    def __init__(self, weights: str, *, device: str = "", half: bool = False):
        self.device = str(device or "").strip()
        self.half = bool(half)
        self.model, self.names = load_yolo_model(weights, device=self.device, half=self.half)

    def predict(
        self,
        frames_bgr: Sequence[np.ndarray],
        *,
        conf: float,
        iou: float,
        imgsz: int | None = None,
        max_det: int = 0,
    ) -> list[list[RawDet]]:
        max_det_i = int(max_det)
        kwargs: dict[str, Any] = {
            "conf": float(conf),
            "iou": float(iou),
            "half": bool(self.half),
            "verbose": False,
        }
        if imgsz is not None:
            kwargs["imgsz"] = int(imgsz)
        kwargs["max_det"] = max_det_i if max_det_i > 0 else None
        results = self.model.predict(list(frames_bgr), **kwargs)
        out: list[list[RawDet]] = []
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None or len(boxes) == 0:
                out.append([])
                continue
            out.append(unpack_boxes(boxes))
        return out

    def predict_dets(
        self,
        frames_bgr: Sequence[np.ndarray],
        *,
        conf: float,
        iou: float,
        source_map: dict[str, str],
        target_to_id: dict[str, int],
        imgsz: int | None = None,
        max_det: int = 0,
    ) -> list[list[Det]]:
        """Predict one call per frame and remap onto target classes."""
        raw_batch = self.predict(frames_bgr, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det)
        return [
            to_dets(raw, names=self.names, source_map=source_map, target_to_id=target_to_id)
            for raw in raw_batch
        ]
