from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from multiview.utils.boxes import Det
from multiview.utils.yolo import load_yolo_model


def parse_kv_map(mapping_str: str) -> dict[str, str]:
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


@dataclass(frozen=True)
class Proposal:
    det: Det
    src_name: str
    tgt_name: str


class YoloProposer:
    def __init__(self, weights: str, *, device: str = "", half: bool = False):
        self.model, self.names = load_yolo_model(weights, device=device, half=half)

    def propose(
        self,
        frame_bgr: np.ndarray,
        *,
        conf: float,
        iou: float,
        target_to_id: dict[str, int],
        source_map: dict[str, str],
        min_box_area: int,
        imgsz: int | None = None,
    ) -> list[Proposal]:
        kwargs: dict[str, Any] = {"conf": float(conf), "iou": float(iou), "verbose": False}
        if imgsz is not None:
            kwargs["imgsz"] = int(imgsz)
        result = self.model.predict(frame_bgr, **kwargs)[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = boxes.conf.cpu().numpy().astype(np.float32)
        cls_ids = boxes.cls.cpu().numpy().astype(np.int32)

        proposals: list[Proposal] = []
        for box, score, cls_id in zip(xyxy, scores, cls_ids):
            src_name = self.names.get(int(cls_id), str(int(cls_id)))
            tgt_name = source_map.get(src_name, src_name)
            if tgt_name not in target_to_id:
                continue
            x0, y0, x1, y1 = [float(v) for v in box.tolist()]
            if (x1 - x0) * (y1 - y0) < float(min_box_area):
                continue
            proposals.append(
                Proposal(det=Det(xyxy=(x0, y0, x1, y1), cls_id=target_to_id[tgt_name], score=float(score)), src_name=src_name, tgt_name=tgt_name)
            )
        return proposals
