"""Multi-camera smoke over real video files: alignment, identity, JSONL.

Two synthetic cameras film the same scene at different frame rates (10 and
25 fps). Frames are decoded from disk by the real run loop; only the YOLO
model is stubbed, and the stub localizes the drawn box by color, so every
box center is an exact reading of *which source frame* was decoded. That
makes timestamp alignment measurable instead of assumed: a naive (not
timestamp-aligned) read drifts by whole seconds at stride>1.

The stride case is the aging-clock regression. SimpleTracker ages once per
processed frame while the assigner ticks on raw video indices, so both
max_age knobs are configured in processed frames and the global window is
scaled by the stride. Un-scaled, a 30-frame global window at stride 5 is 6
processed frames, and a reappearing object re-mints a global id.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from multiview.system import run as run_mod
from multiview.utils.boxes import Det

WIDTH, HEIGHT = 320, 240
BOX = 60
X0 = 20
TOP = 100
VELOCITY = 40.0  # px/s: the box position doubles as a clock
DURATION = 4.0
SLOW_FPS, FAST_FPS = 10.0, 25.0
GREEN = (0, 255, 0)


def _time_from_center(cx: float) -> float:
    """Invert the box position into the source timestamp. XVID blurs box
    edges by ~1px, which is ~0.03s at this velocity."""
    return (cx - BOX / 2.0 - X0) / VELOCITY


def _draw_box(frame: np.ndarray, t: float, velocity: float) -> None:
    x = int(round(X0 + velocity * t))
    frame[TOP : TOP + BOX, x : x + BOX] = GREEN


def _write_video(path: Path, fps: float, *, visible, duration: float, velocity: float) -> int:
    n = int(round(duration * fps))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"XVID"), float(fps), (WIDTH, HEIGHT))
    assert writer.isOpened(), "cv2 cannot write AVI here"
    try:
        for i in range(n):
            frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            if visible(i / fps):
                _draw_box(frame, i / fps, velocity)
            writer.write(frame)
    finally:
        writer.release()
    return n


def _center_x(frame: np.ndarray) -> float | None:
    mask = (frame[..., 1] > 200) & (frame[..., 0] < 100) & (frame[..., 2] < 100)
    if not mask.any():
        return None
    return float(np.where(mask)[1].mean())


def _stub_detector(seen: list[list[float | None]]) -> type:
    """Stand-in for YoloDetector: same call shape, real frames in, exact boxes out."""

    class StubDetector:
        def __init__(self, weights: str, *, device: str = "", half: bool = False) -> None:
            self.names = {0: "person"}

        def predict_dets(
            self,
            frames_bgr,
            *,
            conf: float,
            iou: float,
            source_map,
            target_to_id,
            imgsz=None,
            max_det: int = 0,
        ):
            assert max_det is not None, "the run seam must not forward max_det=None"
            row: list[float | None] = []
            out = []
            for frame in frames_bgr:
                cx = _center_x(frame)
                row.append(cx)
                if cx is None:
                    out.append([])
                    continue
                x0 = cx - BOX / 2.0
                out.append([Det(xyxy=(x0, float(TOP), x0 + BOX, float(TOP + BOX)), cls_id=0, score=0.9)])
            seen.append(row)
            return out

    return StubDetector


def _cfg(videos: dict[str, Path], out_dir: Path, *, stride: int, max_age: int = 30, global_max_age: int = 30) -> dict:
    return {
        "cameras": {cam: {"source": str(path)} for cam, path in (("slow", videos["slow"]), ("fast", videos["fast"]))},
        "groups": {"g": ["slow", "fast"]},
        "detector": {"model": "stub.pt", "imgsz": 96, "conf": 0.2, "iou": 0.5, "targets": "person"},
        "tracker": {
            "max_age": max_age,
            "match_threshold": 0.5,
            "global_match_threshold": 0.75,
            "global_max_age": global_max_age,
            "embedder": {"type": "colorhist", "bins": 8},
        },
        "run": {"frame_stride": stride, "max_frames": None, "groups": ["g"]},
        "output": {"dir": str(out_dir), "write_video": False},
        "runtime": {"device": "cpu", "half": False},
        "debug": {"global_assoc": True, "log_path": str(out_dir / "assoc.jsonl")},
    }


@pytest.fixture(scope="module")
def videos(tmp_path_factory):
    """Both cameras see the box for the whole clip."""
    root = tmp_path_factory.mktemp("mv_videos")
    out = {"slow": root / "slow.avi", "fast": root / "fast.avi"}
    out["n_slow"] = _write_video(out["slow"], SLOW_FPS, visible=lambda t: True, duration=DURATION, velocity=VELOCITY)
    out["n_fast"] = _write_video(out["fast"], FAST_FPS, visible=lambda t: True, duration=DURATION, velocity=VELOCITY)
    return out


@pytest.fixture(scope="module")
def gap_videos(tmp_path_factory):
    """The box disappears for 1.0s..7.0s: long enough that the local tracks
    die (max_age 5 processed frames) and only the global window can carry the
    identity across."""
    root = tmp_path_factory.mktemp("mv_gap_videos")
    out = {"slow": root / "slow.avi", "fast": root / "fast.avi"}
    is_visible = lambda t: t < 1.0 or t >= 7.0  # noqa: E731
    out["n_slow"] = _write_video(out["slow"], SLOW_FPS, visible=is_visible, duration=8.0, velocity=20.0)
    out["n_fast"] = _write_video(out["fast"], FAST_FPS, visible=is_visible, duration=8.0, velocity=20.0)
    return out


def _run(tmp_path: Path, monkeypatch, videos: dict, *, stride: int, **cfg_kw):
    seen: list[list[float | None]] = []
    monkeypatch.setattr(run_mod, "YoloDetector", _stub_detector(seen))
    out_dir = tmp_path / f"out_stride{stride}"
    run_mod.run_multiview(_cfg(videos, out_dir, stride=stride, **cfg_kw))
    payload = json.loads((out_dir / "g.json").read_text())
    lines = [json.loads(ln) for ln in (out_dir / "assoc.jsonl").read_text().splitlines()]
    return payload, lines, seen


def _gids_by_frame(payload: dict) -> dict[int, set[int]]:
    out: dict[int, set[int]] = {}
    for frame in payload["frames"]:
        gids = {int(tr["global_id"]) for tracks in frame["cameras"].values() for tr in tracks}
        out[int(frame["frame"])] = gids
    return out


class TestStrideOne:
    def test_reports_every_sync_frame_and_aligns_the_other_camera(self, tmp_path, monkeypatch, videos) -> None:
        payload, _lines, seen = _run(tmp_path, monkeypatch, videos, stride=1)
        frames = payload["frames"]
        assert [f["frame"] for f in frames] == list(range(videos["n_slow"]))
        assert len(seen) == videos["n_slow"]
        for k, (slow_cx, fast_cx) in enumerate(seen):
            sync_t = k / SLOW_FPS
            assert abs(_time_from_center(slow_cx) - sync_t) <= 0.03
            # The other camera holds the first frame at or after the sync
            # timestamp: it never lags, and it leads by at most one of its own
            # frames. A naive read lags by 0.06s per frame here.
            dt = _time_from_center(fast_cx) - sync_t
            assert -0.03 <= dt <= 1.0 / FAST_FPS + 0.03, f"frame {k}: fast cam off by {dt:.3f}s"

    def test_one_object_two_cameras_one_stable_global_id(self, tmp_path, monkeypatch, videos) -> None:
        payload, _lines, _seen = _run(tmp_path, monkeypatch, videos, stride=1)
        local_ids = {"slow": set(), "fast": set()}
        for frame in payload["frames"]:
            for cam, tracks in frame["cameras"].items():
                assert len(tracks) == 1, f"expected one track, got {tracks}"
                local_ids[cam].add(int(tracks[0]["local_id"]))
        assert all(gids == {1} for gids in _gids_by_frame(payload).values())
        assert local_ids == {"slow": {1}, "fast": {1}}

    def test_jsonl_schema_matches_the_documented_log(self, tmp_path, monkeypatch, videos) -> None:
        payload, lines, _seen = _run(tmp_path, monkeypatch, videos, stride=1)
        assert [ln["frame"] for ln in lines] == [f["frame"] for f in payload["frames"]]
        for line in lines:
            assert set(line) == {"frame", "tracks", "association"}
            assert set(line["tracks"]) == {"slow", "fast"}
            for cam, tracks in line["tracks"].items():
                assert len(tracks) == 1
                tr = tracks[0]
                assert tr["camera_id"] == cam
                assert tr["global_id"] != -1
                assert len(tr["embedding"]) == 24
            assoc = line["association"]
            assert assoc["frame"] == line["frame"]
            assert set(assoc) == {"frame", "threshold", "max_age", "steps"}
            assert assoc["max_age"] == 30  # stride 1: config window == assigner window

        by_frame = {ln["frame"]: ln for ln in lines}
        for frame in payload["frames"]:
            for cam, tracks in frame["cameras"].items():
                logged = by_frame[frame["frame"]]["tracks"][cam][0]
                assert logged["global_id"] == int(tracks[0]["global_id"])
                assert logged["local_id"] == int(tracks[0]["local_id"])


class TestStrideFive:
    def test_global_ids_survive_the_stride_gap(self, tmp_path, monkeypatch, videos) -> None:
        # 40 source frames at stride 5 = 8 processed frames. A 30-frame global
        # window means 30 processed frames, so nothing may expire here.
        payload, _lines, _seen = _run(tmp_path, monkeypatch, videos, stride=5)
        frames = payload["frames"]
        assert [f["frame"] for f in frames] == [0, 5, 10, 15, 20, 25, 30, 35]
        assert all(gids == {1} for gids in _gids_by_frame(payload).values())
        local_ids = {cam: {int(tr["local_id"]) for f in frames for tr in f["cameras"][cam]} for cam in ("slow", "fast")}
        assert local_ids == {"slow": {1}, "fast": {1}}

    def test_alignment_holds_under_stride(self, tmp_path, monkeypatch, videos) -> None:
        _payload, _lines, seen = _run(tmp_path, monkeypatch, videos, stride=5)
        assert len(seen) == 8
        for k, (slow_cx, fast_cx) in enumerate(seen):
            sync_t = (k * 5) / SLOW_FPS
            assert abs(_time_from_center(slow_cx) - sync_t) <= 0.03
            dt = _time_from_center(fast_cx) - sync_t
            assert -0.03 <= dt <= 1.0 / FAST_FPS + 0.03, f"raw frame {k * 5}: fast cam off by {dt:.3f}s"

    def test_jsonl_reports_the_effective_global_window(self, tmp_path, monkeypatch, videos) -> None:
        _payload, lines, _seen = _run(tmp_path, monkeypatch, videos, stride=5)
        assert [ln["frame"] for ln in lines] == [0, 5, 10, 15, 20, 25, 30, 35]
        assert all(ln["association"]["max_age"] == 150 for ln in lines)

    def test_identity_survives_a_detection_gap_whose_local_tracks_died(self, tmp_path, monkeypatch, gap_videos) -> None:
        # The box is gone for 12 processed frames at stride 5 (60 source
        # frames). Local tracks (max_age 5) die, so only the global window
        # (30 processed frames = 150 source frames) can carry the identity.
        payload, _lines, _seen = _run(tmp_path, monkeypatch, gap_videos, stride=5, max_age=5)
        by_frame = _gids_by_frame(payload)
        assert by_frame[0] == {1} and by_frame[5] == {1}
        assert by_frame[70] == {1} and by_frame[75] == {1}, f"global id churned: {by_frame}"
        # The local ids did churn: the association is what stayed stable.
        local_after = {int(tr["local_id"]) for tr in payload["frames"][-1]["cameras"]["slow"]}
        assert local_after == {2}
