"""Contract check against a real ultralytics install (skipped when absent).

The max_det guard in multiview/detector.py exists because ultralytics
validates its predict kwargs before it ever touches a frame: max_det=None
raises TypeError and max_det=0 raises ValueError (it must be >= 1). The
unit tests pin what we send; this pins what the library accepts, so the
"unset means omit" contract cannot drift back into a forwarded None.

The model is built from the yaml bundled with ultralytics, so this needs no
weights download and no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

ultralytics = pytest.importorskip("ultralytics")

from ultralytics import YOLO  # noqa: E402

from multiview.detector import YoloDetector  # noqa: E402

FRAME = np.zeros((64, 64, 3), dtype=np.uint8)


@pytest.fixture()
def real_model(monkeypatch) -> YOLO:
    model = YOLO("yolo11n.yaml")
    monkeypatch.setattr(
        "multiview.detector.load_yolo_model", lambda w, *, device="", half=False: (model, dict(model.names))
    )
    return model


def test_unset_max_det_is_accepted_and_uses_the_library_default(real_model) -> None:
    det = YoloDetector("unused.pt")
    out = det.predict([FRAME], conf=0.25, iou=0.5, max_det=0)
    assert len(out) == 1
    assert det.model.predictor.args.max_det == 300


def test_explicit_max_det_is_accepted(real_model) -> None:
    out = real_model.predict(FRAME, conf=0.25, iou=0.5, max_det=5, verbose=False)
    assert real_model.predictor.args.max_det == 5
    assert len(out) == 1


def test_forwarding_none_is_what_ultralytics_rejects(real_model) -> None:
    with pytest.raises(TypeError, match="max_det=None"):
        real_model.predict(FRAME, conf=0.25, iou=0.5, max_det=None, verbose=False)
    with pytest.raises(ValueError, match="max_det=0"):
        real_model.predict(FRAME, conf=0.25, iou=0.5, max_det=0, verbose=False)
