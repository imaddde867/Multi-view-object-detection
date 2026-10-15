"""Additional contract pins identified by the test-suite audit.

These are characterization tests: each one kills a specific mutant the
audit found surviving in the first pass, so each guards a behavior that
was previously unpinned.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

from multiview.dataset.layout import frame_name, split_frame_name
from multiview.detector import YoloDetector
from multiview.frames import iter_group_frames
from multiview.system.assoc_log import JsonlAssocLogger
from multiview.tracking.global_association import GlobalIDAssigner, GlobalTrackView


def _view(cam: str, local: int, emb=(1.0, 0.0), cls: int = 0) -> GlobalTrackView:
    return GlobalTrackView(
        cam=cam, local_id=local, cls_id=cls, xyxy=(0.0, 0.0, 10.0, 10.0),
        embedding=np.asarray(emb, dtype=np.float32),
    )


class TestFrameNameLastSplit:
    def test_group_name_containing_marker_splits_at_last(self) -> None:
        name = frame_name("f_team", "cam1", 12)
        assert name == "f_team_cam1_f000012"
        assert split_frame_name(name) == ("f_team_cam1_f", "000012")


class TestFramesErrorsAndFallbacks:
    def test_unknown_align_policy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown align"):
            iter_group_frames({}, align="bogus")

    def test_timestamp_falls_back_to_first_cam(self) -> None:
        class FakeCap:
            def __init__(self, script):
                self.script = list(script)
                self._last = (False, ("", 0.0))

            def read(self):
                if not self.script:
                    return False, None
                frame = self.script.pop(0)
                self._last = (True, frame)
                return True, frame[0]

            def get(self, prop):
                return self._last[1][1]

        caps = {
            "cam1": FakeCap([("a0", 100.0), ("a1", 200.0)]),
            "cam2": FakeCap([("b0", 100.0), ("b1", 200.0)]),
        }
        out = list(iter_group_frames(caps, align="timestamp", sync_cam=None))
        assert [f["cam1"] for _, f in out] == ["a0", "a1"]


class TestDetectorHalfForwarding:
    def test_half_is_forwarded_to_the_model(self, monkeypatch) -> None:
        monkeypatch.setattr("multiview.detector.load_yolo_model", lambda w, *, device="", half=False: (object(), {}))
        calls = {}

        class FakeModel:
            def predict(self, frames, **kwargs):
                calls.update(kwargs)
                return []

        det = YoloDetector("w.pt", half=True)
        det.model = FakeModel()
        det.predict([np.zeros((4, 4, 3))], conf=0.3, iou=0.5)
        assert calls["half"] is True

    def test_imgsz_omitted_when_not_given(self, monkeypatch) -> None:
        monkeypatch.setattr("multiview.detector.load_yolo_model", lambda w, *, device="", half=False: (object(), {}))
        calls = {}

        class FakeModel:
            def predict(self, frames, **kwargs):
                calls.update(kwargs)
                return []

        det = YoloDetector("w.pt")
        det.model = FakeModel()
        det.predict([np.zeros((4, 4, 3))], conf=0.3, iou=0.5)
        # old label path did not pass imgsz when proposal_imgsz was unset
        assert "imgsz" not in calls


class TestAssignerClassChange:
    def test_class_change_evicts_and_remints(self) -> None:
        a = GlobalIDAssigner(match_threshold=0.75, max_age=30)
        first = a.assign_frame(0, {"cam1": [_view("cam1", 1, cls=0)]})
        assert first[("cam1", 1)] == 1
        # same embedding, different class: association must break
        changed = a.assign_frame(1, {"cam1": [_view("cam1", 1, cls=1)]})
        assert changed[("cam1", 1)] == 2

    def test_cam_order_drives_anchor_choice(self) -> None:
        # with cam_order [cam2, cam1], cam2 is the anchor that mints the gid;
        # the recorder's merge step names anchor/other in iteration order.
        class Rec:
            def __init__(self):
                self.steps = []

            def record(self, frame_idx, views, payload):
                self.steps = [s for s in payload["steps"] if s["type"] == "merge_assigned"]

        a = GlobalIDAssigner(match_threshold=0.75, max_age=30)
        a.assign_frame(0, {"cam1": [_view("cam1", 1)], "cam2": [_view("cam2", 5)]})
        rec = Rec()
        mapping = a.assign_frame(
            1, {"cam2": [_view("cam2", 5)], "cam1": [_view("cam1", 1)]}, cam_order=["cam2", "cam1"], recorder=rec
        )
        assert rec.steps, "expected a merge step between the two cameras"
        assert rec.steps[0]["anchor_cam"] == "cam2"
        assert rec.steps[0]["other_cam"] == "cam1"
        assert mapping[("cam1", 1)] == mapping[("cam2", 5)]


class TestLoggerEdgeCases:
    def test_track_without_global_id_gets_sentinel(self) -> None:
        buf = io.StringIO()
        logger = JsonlAssocLogger(buf)
        tracks = {"cam1": [{"local_id": 1, "cls_id": 0, "cls_name": "person", "bbox_xyxy": [0, 0, 1, 1], "score": 0.5}]}
        logger.flush(0, tracks, {}, {})  # no views, no ids
        entry = json.loads(buf.getvalue())["tracks"]["cam1"][0]
        assert entry["global_id"] == -1
        assert entry["embedding"] == []

    def test_close_closes_handle(self) -> None:
        buf = io.StringIO()
        logger = JsonlAssocLogger(buf)
        logger.close()
        assert buf.closed


