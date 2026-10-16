"""Tests for the single YOLO adapter shared by the label and run paths."""

from __future__ import annotations

import numpy as np
import pytest

from multiview.detector import YoloDetector, parse_kv_map, to_dets, unpack_boxes
from multiview.utils.boxes import Det


class _FakeTensor:
    """Stands in for an ultralytics tensor: .cpu().numpy() only."""

    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def cpu(self) -> "_FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


class _FakeBoxes:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self.xyxy = _FakeTensor(xyxy)
        self.conf = _FakeTensor(conf)
        self.cls = _FakeTensor(cls)

    def __len__(self) -> int:
        return len(self.xyxy._arr)


def _boxes(n: int = 2) -> _FakeBoxes:
    return _FakeBoxes(
        np.array([[10.0, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0]])[:n],
        np.array([0.9, 0.5], dtype=np.float32)[:n],
        np.array([0, 2], dtype=np.float32)[:n],
    )


class TestParseKvMap:
    def test_empty(self) -> None:
        assert parse_kv_map("") == {}

    def test_pairs_with_spaces(self) -> None:
        assert parse_kv_map(" truck = car , motorcycle=car ") == {"truck": "car", "motorcycle": "car"}

    def test_invalid_entry_raises(self) -> None:
        with pytest.raises(ValueError, match="expected k=v"):
            parse_kv_map("truck")


class TestUnpackBoxes:
    def test_rows_to_raw_tuples(self) -> None:
        raw = unpack_boxes(_boxes(2))
        # float32 widening is the pinned production behavior
        assert [r[0] for r in raw] == [(10.0, 20.0, 30.0, 40.0), (1.0, 2.0, 3.0, 4.0)]
        assert [r[1] for r in raw] == pytest.approx([0.9, 0.5])
        assert [r[2] for r in raw] == [0, 2]

    def test_empty_boxes(self) -> None:
        assert unpack_boxes(_boxes(0)) == []


class TestToDets:
    names = {0: "person", 1: "car", 2: "bus"}
    targets = {"person": 0, "car": 1}

    def test_remap_and_filter(self) -> None:
        # bus is not a target and is dropped; person passes through untouched.
        raw = [((1.0, 1.0, 2.0, 2.0), 0.8, 0), ((0.0, 0.0, 4.0, 4.0), 0.7, 2)]
        dets = to_dets(raw, names=self.names, source_map={}, target_to_id=self.targets)
        assert dets == [Det(xyxy=(1.0, 1.0, 2.0, 2.0), cls_id=0, score=pytest.approx(0.8))]

    def test_source_map_renames(self) -> None:
        raw = [((0.0, 0.0, 4.0, 4.0), 0.7, 2)]
        dets = to_dets(raw, names=self.names, source_map={"bus": "person"}, target_to_id=self.targets)
        assert dets[0].cls_id == 0
        assert dets[0].score == pytest.approx(0.7)

    def test_unknown_cls_id_falls_back_to_str(self) -> None:
        # id 99 is not in names; its string form is used for the lookup and dropped
        raw = [((0.0, 0.0, 4.0, 4.0), 0.7, 99)]
        dets = to_dets(raw, names=self.names, source_map={}, target_to_id=self.targets)
        assert dets == []


class TestYoloDetector:
    def test_loads_weights_and_names_via_loader(self, monkeypatch) -> None:
        seen = {}

        def fake_load(weights: str, *, device: str = "", half: bool = False):
            seen.update(weights=weights, device=device, half=half)
            return object(), {0: "person"}

        monkeypatch.setattr("multiview.detector.load_yolo_model", fake_load)
        det = YoloDetector("weights.pt", device="cpu", half=True)
        assert seen == {"weights": "weights.pt", "device": "cpu", "half": True}
        assert det.names == {0: "person"}

    def test_predict_passes_kwargs_and_zero_max_det_omits_the_key(self, monkeypatch) -> None:
        monkeypatch.setattr("multiview.detector.load_yolo_model", lambda w, *, device="", half=False: (object(), {}))
        det = YoloDetector("w.pt")
        calls = {}

        class FakeModel:
            def predict(self, frames, **kwargs):
                calls["frames"] = frames
                calls["kwargs"] = kwargs
                return [object(), object()]

        det.model = FakeModel()
        det.predict([np.zeros((4, 4, 3)), np.zeros((4, 4, 3))], conf=0.3, iou=0.6, imgsz=640, max_det=0)
        assert calls["frames"] is not None and len(calls["frames"]) == 2
        kw = calls["kwargs"]
        assert kw["conf"] == 0.3 and kw["iou"] == 0.6 and kw["imgsz"] == 640
        # 0 means "unset": the key must be absent, not None. ultralytics
        # raises TypeError on max_det=None and ValueError on max_det=0.
        assert "max_det" not in kw
        assert kw["verbose"] is False
    def test_predict_positive_max_det_passes_through(self, monkeypatch) -> None:
        monkeypatch.setattr("multiview.detector.load_yolo_model", lambda w, *, device="", half=False: (object(), {}))
        det = YoloDetector("w.pt")
        calls = {}

        class FakeModel:
            def predict(self, frames, **kwargs):
                calls.update(kwargs)
                return [_FakeResult(_boxes(0))]

        det.model = FakeModel()
        out = det.predict([np.zeros((4, 4, 3))], conf=0.3, iou=0.5, imgsz=640, max_det=7)
        assert calls["max_det"] == 7
        assert out == [[]]

    def test_predict_dets_end_to_end_with_stub_model(self, monkeypatch) -> None:
        monkeypatch.setattr("multiview.detector.load_yolo_model", lambda w, *, device="", half=False: (object(), {0: "person", 1: "car"}))
        det = YoloDetector("w.pt")

        class FakeModel:
            def predict(self, frames, **kwargs):
                return [_FakeResult(_boxes(2))]

        det.model = FakeModel()
        dets = det.predict_dets(
            [np.zeros((4, 4, 3))],
            conf=0.3,
            iou=0.5,
            source_map={"truck": "car"},
            target_to_id={"person": 0, "car": 1},
        )
        assert dets == [[Det(xyxy=(10.0, 20.0, 30.0, 40.0), cls_id=0, score=pytest.approx(0.9))]]


class _FakeResult:
    def __init__(self, boxes) -> None:
        self.boxes = boxes
