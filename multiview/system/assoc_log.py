"""The global-association debug JSONL seam.

The documented log (see README: debug.global_assoc) is one JSON object
per frame with `frame`, `tracks`, and `association` sections. The
association section is built by GlobalIDAssigner; the tracks section used
to be assembled inline in the run loop from track dicts plus view
embeddings. This module owns the file handle and the line format, so the
assigner knows nothing about JSONL and the run loop's debug path is one
call.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from multiview.tracking.global_association import GlobalTrackView


class JsonlAssocLogger:
    """Two-call recorder: the assigner supplies the association section via
    ``record`` during assignment; the run loop supplies the tracks section
    via ``flush``, which writes the complete line."""

    def __init__(self, fp) -> None:
        self._fp = fp
        self._association: dict[str, Any] | None = None

    def record(self, frame_idx: int, views_by_cam: dict[str, list[GlobalTrackView]], payload: dict[str, Any]) -> None:
        self._association = payload

    def close(self) -> None:
        self._fp.close()

    def flush(
        self,
        frame_idx: int,
        per_cam_tracks: dict[str, list[dict[str, Any]]],
        views_by_cam: dict[str, list[GlobalTrackView]],
        global_ids: dict[tuple[str, int], int],
    ) -> None:
        views_by_key = {(v.cam, v.local_id): v for views in views_by_cam.values() for v in views}
        tracks: dict[str, list[dict[str, Any]]] = {}
        for cam, trs in per_cam_tracks.items():
            entries = []
            for tr in trs:
                key = (cam, int(tr["local_id"]))
                view = views_by_key.get(key)
                entries.append(
                    {
                        "camera_id": cam,
                        "local_id": int(tr["local_id"]),
                        "global_id": int(global_ids.get(key, -1)),
                        "bbox_xyxy": [float(v) for v in tr["bbox_xyxy"]],
                        "score": float(tr["score"]),
                        "cls_id": int(tr["cls_id"]),
                        "cls_name": str(tr["cls_name"]),
                        "embedding": (
                            view.embedding.astype(np.float32, copy=False).reshape(-1).tolist()
                            if view is not None
                            else []
                        ),
                    }
                )
            tracks[cam] = entries
        line = {"frame": int(frame_idx), "tracks": tracks, "association": self._association}
        self._fp.write(json.dumps(line, default=str) + "\n")
        self._association = None
