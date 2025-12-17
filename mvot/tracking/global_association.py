from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from mvot.tracking.embeddings import cosine_sim


@dataclass(frozen=True)
class GlobalTrackView:
    cam: str
    local_id: int
    cls_id: int
    xyxy: tuple[float, float, float, float]
    embedding: np.ndarray


class GlobalIDAssigner:
    def __init__(self, *, match_threshold: float = 0.75):
        self.match_threshold = float(match_threshold)
        self._next_gid = 1
        self._map: dict[tuple[str, int], int] = {}

    def get(self, cam: str, local_id: int) -> int | None:
        return self._map.get((cam, local_id))

    def ensure(self, cam: str, local_id: int) -> int:
        key = (cam, local_id)
        gid = self._map.get(key)
        if gid is None:
            gid = self._next_gid
            self._next_gid += 1
            self._map[key] = gid
        return gid

    def unify(self, a: tuple[str, int], b: tuple[str, int]) -> int:
        ga = self._map.get(a)
        gb = self._map.get(b)
        if ga is None and gb is None:
            gid = self._next_gid
            self._next_gid += 1
            self._map[a] = gid
            self._map[b] = gid
            return gid
        if ga is None:
            self._map[a] = gb
            return gb  # type: ignore[return-value]
        if gb is None:
            self._map[b] = ga
            return ga
        if ga == gb:
            return ga
        keep = min(ga, gb)
        drop = max(ga, gb)
        for k, v in list(self._map.items()):
            if v == drop:
                self._map[k] = keep
        return keep

    def associate_anchor(
        self,
        anchor_cam: str,
        anchor_tracks: list[GlobalTrackView],
        other_cam: str,
        other_tracks: list[GlobalTrackView],
    ) -> None:
        if not anchor_tracks or not other_tracks:
            return

        cost = np.ones((len(anchor_tracks), len(other_tracks)), dtype=np.float32) * 1e3
        for i, ta in enumerate(anchor_tracks):
            for j, tb in enumerate(other_tracks):
                if ta.cls_id != tb.cls_id:
                    continue
                sim = cosine_sim(ta.embedding, tb.embedding)
                cost[i, j] = 1.0 - sim

        row_ind, col_ind = linear_sum_assignment(cost)
        for i, j in zip(row_ind.tolist(), col_ind.tolist()):
            sim = 1.0 - float(cost[i, j])
            if sim < self.match_threshold:
                continue
            self.unify((anchor_cam, anchor_tracks[i].local_id), (other_cam, other_tracks[j].local_id))

