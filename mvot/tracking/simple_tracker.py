from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from mvot.tracking.embeddings import Embedder, cosine_sim
from mvot.utils.boxes import Det


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    xx0 = max(ax0, bx0)
    yy0 = max(ay0, by0)
    xx1 = min(ax1, bx1)
    yy1 = min(ay1, by1)
    inter = max(0.0, xx1 - xx0) * max(0.0, yy1 - yy0)
    a_area = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    b_area = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = a_area + b_area - inter
    return float(inter / (union + 1e-9))


@dataclass
class Track:
    track_id: int
    cls_id: int
    xyxy: tuple[float, float, float, float]
    score: float
    embedding: np.ndarray
    age: int = 0
    hits: int = 1


class SimpleTracker:
    def __init__(
        self,
        embedder: Embedder,
        *,
        max_age: int = 30,
        iou_weight: float = 0.5,
        appearance_weight: float = 0.5,
        match_threshold: float = 0.5,
        min_iou: float = 0.05,
    ):
        self.embedder = embedder
        self.max_age = int(max_age)
        self.iou_weight = float(iou_weight)
        self.appearance_weight = float(appearance_weight)
        self.match_threshold = float(match_threshold)
        self.min_iou = float(min_iou)
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, frame_bgr: np.ndarray, dets: list[Det]) -> list[Track]:
        for t in self.tracks:
            t.age += 1

        det_embeddings = [self.embedder.embed(frame_bgr, d.xyxy) for d in dets]

        if not self.tracks:
            for d, emb in zip(dets, det_embeddings):
                self.tracks.append(Track(track_id=self._next_id, cls_id=d.cls_id, xyxy=d.xyxy, score=d.score, embedding=emb))
                self._next_id += 1
            self.tracks = [t for t in self.tracks if t.age <= self.max_age]
            return list(self.tracks)

        cost = np.ones((len(self.tracks), len(dets)), dtype=np.float32) * 1e3
        for i, trk in enumerate(self.tracks):
            for j, det in enumerate(dets):
                if trk.cls_id != det.cls_id:
                    continue
                iou = iou_xyxy(trk.xyxy, det.xyxy)
                if iou < self.min_iou:
                    continue
                sim = cosine_sim(trk.embedding, det_embeddings[j])
                match_score = self.iou_weight * iou + self.appearance_weight * max(0.0, sim)
                cost[i, j] = 1.0 - match_score

        row_ind, col_ind = linear_sum_assignment(cost)
        assigned_tracks = set()
        assigned_dets = set()

        for i, j in zip(row_ind.tolist(), col_ind.tolist()):
            if cost[i, j] > 1.0 - self.match_threshold:
                continue
            trk = self.tracks[i]
            det = dets[j]
            trk.xyxy = det.xyxy
            trk.score = det.score
            trk.embedding = det_embeddings[j]
            trk.age = 0
            trk.hits += 1
            assigned_tracks.add(i)
            assigned_dets.add(j)

        for j, det in enumerate(dets):
            if j in assigned_dets:
                continue
            self.tracks.append(Track(track_id=self._next_id, cls_id=det.cls_id, xyxy=det.xyxy, score=det.score, embedding=det_embeddings[j]))
            self._next_id += 1

        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        return list(self.tracks)

