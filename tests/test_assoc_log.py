"""Tests for the global-association debug logger (the JSONL seam)."""

from __future__ import annotations

import io
import json

import numpy as np

from multiview.system.assoc_log import JsonlAssocLogger
from multiview.tracking.global_association import GlobalIDAssigner, GlobalTrackView


def _view(cam: str, local: int, emb=(1.0, 0.0)) -> GlobalTrackView:
    return GlobalTrackView(
        cam=cam, local_id=local, cls_id=0, xyxy=(0.0, 0.0, 10.0, 10.0),
        embedding=np.asarray(emb, dtype=np.float32),
    )


def _per_cam_tracks() -> dict[str, list[dict]]:
    return {
        "cam1": [
            {"local_id": 1, "cls_id": 0, "cls_name": "person", "bbox_xyxy": [0, 0, 10, 10], "score": 0.9},
        ]
    }


def test_writes_one_jsonl_line_per_frame_with_both_sections() -> None:
    buf = io.StringIO()
    logger = JsonlAssocLogger(buf)
    assigner = GlobalIDAssigner(match_threshold=0.75, max_age=30)
    views = {"cam1": [_view("cam1", 1)]}

    global_ids = assigner.assign_frame(0, views, recorder=logger)
    logger.flush(0, _per_cam_tracks(), views, global_ids)

    lines = buf.getvalue().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["frame"] == 0
    track = payload["tracks"]["cam1"][0]
    assert track["local_id"] == 1
    assert track["global_id"] == global_ids[("cam1", 1)]
    assert track["cls_name"] == "person"
    assert track["embedding"] == [1.0, 0.0]
    assert payload["association"]["frame"] == 0
    assert payload["association"]["threshold"] == 0.75
    assert isinstance(payload["association"]["steps"], list)


def test_two_frames_write_two_lines() -> None:
    buf = io.StringIO()
    logger = JsonlAssocLogger(buf)
    assigner = GlobalIDAssigner(match_threshold=0.75, max_age=30)
    for f in (0, 1):
        views = {"cam1": [_view("cam1", 1)]}
        global_ids = assigner.assign_frame(f, views, recorder=logger)
        logger.flush(f, _per_cam_tracks(), views, global_ids)
    frames = [json.loads(line)["frame"] for line in buf.getvalue().splitlines()]
    assert frames == [0, 1]


def test_association_section_resets_after_flush() -> None:
    buf = io.StringIO()
    logger = JsonlAssocLogger(buf)
    views = {"cam1": [_view("cam1", 1)]}
    assigner = GlobalIDAssigner(match_threshold=0.75, max_age=30)
    global_ids = assigner.assign_frame(0, views, recorder=logger)
    logger.flush(0, _per_cam_tracks(), views, global_ids)
    # a frame flushed without a recorded association must not leak the old one
    logger.flush(1, {}, {}, {})
    assert json.loads(buf.getvalue().splitlines()[1])["association"] is None
