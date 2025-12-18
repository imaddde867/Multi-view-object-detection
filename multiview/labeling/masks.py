from __future__ import annotations

import cv2
import numpy as np


def clean_binary_mask(mask: np.ndarray, *, close_kernel: int = 0) -> np.ndarray:
    m = mask.astype(np.uint8)
    if close_kernel and close_kernel > 1:
        k = int(close_kernel)
        if k % 2 == 0:
            k += 1
        kernel = np.ones((k, k), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    num, labels = cv2.connectedComponents(m)
    if num <= 1:
        return m.astype(bool)

    counts = np.bincount(labels.reshape(-1))
    counts[0] = 0
    keep = int(counts.argmax())
    return (labels == keep)


def mask_to_xyxy(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    x0 = float(xs.min())
    y0 = float(ys.min())
    # +1 to produce an exclusive max corner (consistent with cv2 slicing and YOLO width/height).
    x1 = float(xs.max() + 1)
    y1 = float(ys.max() + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)
