from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Det:
    xyxy: tuple[float, float, float, float]
    cls_id: int
    score: float


def clip_xyxy(xyxy: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float] | None:
    x0, y0, x1, y1 = [float(v) for v in xyxy]
    # Coordinate convention: (x0, y0) inclusive, (x1, y1) exclusive.
    x0 = max(0.0, min(float(width - 1), x0))
    y0 = max(0.0, min(float(height - 1), y0))
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def area_xyxy(xyxy: tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = xyxy
    return max(0.0, x1 - x0) * max(0.0, y1 - y0)


def to_yolo_line(cls_id: int, xyxy: tuple[float, float, float, float], width: int, height: int) -> str | None:
    clipped = clip_xyxy(xyxy, width=width, height=height)
    if clipped is None:
        return None
    x0, y0, x1, y1 = clipped
    xc = (x0 + x1) / 2.0 / width
    yc = (y0 + y1) / 2.0 / height
    bw = (x1 - x0) / width
    bh = (y1 - y0) / height
    if bw <= 0 or bh <= 0:
        return None
    return f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def nms_xyxy(dets: Iterable[Det], iou_thr: float) -> list[Det]:
    det_list = list(dets)
    if not det_list:
        return []

    boxes = np.array([d.xyxy for d in det_list], dtype=np.float32)
    scores = np.array([d.score for d in det_list], dtype=np.float32)
    order = scores.argsort()[::-1]

    x0 = boxes[:, 0]
    y0 = boxes[:, 1]
    x1 = boxes[:, 2]
    y1 = boxes[:, 3]
    areas = np.maximum(0.0, x1 - x0) * np.maximum(0.0, y1 - y0)

    keep_idx: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep_idx.append(i)
        if order.size == 1:
            break
        xx0 = np.maximum(x0[i], x0[order[1:]])
        yy0 = np.maximum(y0[i], y0[order[1:]])
        xx1 = np.minimum(x1[i], x1[order[1:]])
        yy1 = np.minimum(y1[i], y1[order[1:]])
        w = np.maximum(0.0, xx1 - xx0)
        h = np.maximum(0.0, yy1 - yy0)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_thr)[0]
        order = order[inds + 1]

    return [det_list[i] for i in keep_idx]
